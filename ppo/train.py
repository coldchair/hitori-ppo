"""Train MaskablePPO on hitori-gym."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList

from ppo.callbacks import CriticOutputHeadWarmupCallback, EvalAndCheckpointCallback, JsonlTrainMetricsCallback
from ppo.env import make_env, make_vec_env
from ppo.evaluation import evaluate_all_sizes, plot_learning_curves_per_size, resolve_eval_vec_envs
from ppo.policies import make_policy_spec


def _configure_stdio_line_buffered() -> None:
    """Avoid silent logs when stdout is a pipe (e.g. ``conda run``)."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, OSError, ValueError):
            pass


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_training_device(requested: str) -> str:
    """Pick a torch device string for SB3; CUDA > MPS > CPU when ``requested`` is ``auto``."""

    raw = requested.strip()
    key = raw.lower()
    if key == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"
    if key == "cpu":
        return "cpu"
    if key == "mps":
        if not _mps_available():
            raise ValueError("Requested --device mps but MPS is not available (needs Apple Silicon PyTorch build).")
        return "mps"
    if key == "cuda" or key.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(
                "Requested CUDA device but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build in this env, or use --device cpu."
            )
        return raw
    raise ValueError(f"Unknown --device {requested!r}; use auto, cpu, cuda, cuda:N, or mps.")


def parse_size_list(raw: str | None, fallback: list[int]) -> list[int]:
    if raw is None:
        return fallback
    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not sizes:
        raise ValueError("size list must not be empty")
    return sizes


def parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("float list must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--train-sizes", type=str, default=None)
    parser.add_argument("--eval-sizes", type=str, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--sampling-strategy", choices=["round_robin", "weighted", "curriculum"], default="round_robin")
    parser.add_argument("--train-weights", type=str, default=None)
    parser.add_argument("--curriculum-thresholds", type=str, default=None)
    parser.add_argument("--reward-mode", choices=["sparse", "shaped"], default="sparse")
    parser.add_argument("--model", choices=["structured", "cnn", "cnnv2", "unet"], default="unet")
    parser.add_argument(
        "--num-attention-blocks",
        type=int,
        default=0,
        help="U-Net only: number of AxialAttentionBlock at bottleneck (0 = none). Ignored for --model cnn|cnnv2|structured.",
    )
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Episodes per eval board size during periodic eval (learning_curve.csv / checkpoints).",
    )
    parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=200,
        help="Episodes per eval board size for final_eval.txt only (more stable estimate).",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Base int for the fixed eval puzzle suite: periodic + final eval share it. "
        "Episode ep at eval_sizes[k] uses options['episode_seed']=eval_seed + k*1000 + ep (see ppo/evaluation.py). "
        "Default: --seed + 10000 (same periodic base as before; final eval no longer uses seed+20000).",
    )
    parser.add_argument("--max-steps", type=int, default=1_000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--dup-coef", type=float, default=1.0)
    parser.add_argument("--adj-coef", type=float, default=0.5)
    parser.add_argument("--comp-coef", type=float, default=1.5)
    parser.add_argument(
        "--shaped-reward-norm",
        choices=["none", "inv_board_area"],
        default="none",
        help="Only when --reward-mode shaped: inv_board_area divides (step penalty + constraint deltas) by "
        "(board_size**2); success/failure bonuses are not scaled; sparse ignores this.",
    )
    parser.add_argument(
        "--shaped-step-penalty",
        type=float,
        default=-0.05,
        help="Only when --reward-mode shaped: per-step penalty added before area norm; sparse ignores this.",
    )
    parser.add_argument(
        "--shaped-dense-mult",
        type=float,
        default=None,
        help="Only when --reward-mode shaped: multiply dense return (after inv_board_area scaling if any) by this. "
        "Default: 15 with --shaped-reward-norm inv_board_area so episode dense sum is ~10 vs +10 terminal; else 1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="RL policy device: auto (CUDA if available, else Apple MPS, else CPU), or cpu|cuda|cuda:N|mps.",
    )
    parser.add_argument(
        "--eval-vec-envs",
        type=int,
        default=None,
        help="Parallel eval env slots (thread pool for env step/reset; one model, batched GPU predict). "
        "1 = serial. Default=min(--n-envs, total eval episodes).",
    )
    parser.add_argument(
        "--train-random-pad-offset",
        action="store_true",
        help="When board n < max_size, place the n×n board at a random offset inside the max_size×max_size "
        "observation (training only). Eval and final eval always use top-left (no random offset).",
    )
    parser.add_argument(
        "--pretrain-policy",
        type=Path,
        default=None,
        help="Optional path to a MaskablePPO .zip (e.g. SFT output sft_maskable_ppo_best.zip). "
        "Copies **policy** weights (actor + shared backbone + critic heads) into a new PPO built with "
        "CLI --n-steps / --batch-size / etc.; checkpoint must match --model, padded obs/action spaces, "
        "and U-Net --num-attention-blocks. Saved n_steps/batch_size in the zip are ignored. "
        "Also triggers an extra eval at csv timesteps=0 before the first PPO rollout.",
    )
    parser.add_argument(
        "--critic-warmup-timesteps",
        type=int,
        default=0,
        help="If > 0: while SB3 num_timesteps is below this threshold, freeze backbone and actor "
        "and train only the critic scalar output head (final Linear after vf features), so random "
        "value initialization does not backprop into the actor. 0 disables. Compared after each "
        "rollout (same counter as --total-timesteps).",
    )
    return parser.parse_args()


def default_out_dir(reward_mode: str, model_name: str) -> Path:
    if model_name == "structured":
        return Path(f"runs/maskable_ppo_{reward_mode}")
    return Path(f"runs/maskable_ppo_{reward_mode}_{model_name}")


def format_board_tag(sizes: list[int], max_size: int, sampling_strategy: str) -> str:
    unique_sizes = sorted(set(sizes))
    if len(unique_sizes) == 1:
        size = unique_sizes[0]
        if size == max_size:
            tag = f"{size}x{size}"
        else:
            tag = f"{size}x{size}_max{max_size}"
    else:
        size_tag = "-".join(str(size) for size in unique_sizes)
        tag = f"mix{size_tag}_max{max_size}"
    if sampling_strategy == "round_robin":
        return tag
    return f"{tag}_{sampling_strategy}"


def _resolve_pretrain_checkpoint(path: Path) -> Path:
    p = path.expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    if p.is_file():
        return p
    if p.suffix != ".zip":
        alt = p.with_suffix(".zip")
        if alt.is_file():
            return alt
    raise FileNotFoundError(f"--pretrain-policy not found: {path} (resolved: {p})")


def _infer_policy_model_name(policy: object) -> str:
    from ppo.policies import HitoriCNNV2Policy, HitoriUNetPolicy

    if isinstance(policy, HitoriUNetPolicy):
        return "unet"
    if isinstance(policy, HitoriCNNV2Policy):
        return "cnnv2"
    fe = getattr(policy, "features_extractor", None)
    if fe is None:
        raise ValueError(f"cannot infer --model from policy type {type(policy)!r} (no features_extractor)")
    name = type(fe).__name__
    if name == "HitoriStructuredFeatureExtractor":
        return "structured"
    if name == "HitoriCNNFeatureExtractor":
        return "cnn"
    raise ValueError(
        f"unknown pretrained policy stack: policy={type(policy).__name__}, features_extractor={name!r}. "
        "Supported: unet, cnnv2, structured, cnn (must match CLI --model)."
    )


def _unet_attention_block_count(policy: object) -> int:
    fe = policy.features_extractor
    backbone = getattr(fe, "backbone", None)
    if backbone is None:
        raise AssertionError("internal: UNet policy missing features_extractor.backbone")
    attn = getattr(backbone, "attention", None)
    if attn is None:
        raise AssertionError("internal: UNet backbone missing attention")
    return len(attn)


def _assert_pretrain_policy_matches(*, model: MaskablePPO, env: Any, args: argparse.Namespace) -> None:
    if env.observation_space != model.observation_space:
        raise ValueError(
            "pretrain checkpoint observation_space does not match training env:\n"
            f"  env:  {env.observation_space}\n"
            f"  ckpt: {model.observation_space}\n"
            "Use the same padded board layout (--max-size, wrappers) as the checkpoint run."
        )
    if env.action_space != model.action_space:
        raise ValueError(
            f"pretrain checkpoint action_space mismatch: env={env.action_space} ckpt={model.action_space}"
        )

    loaded_name = _infer_policy_model_name(model.policy)
    if loaded_name != args.model:
        raise ValueError(
            f"pretrain policy / --model mismatch: checkpoint is {loaded_name!r}, CLI has --model {args.model!r}."
        )

    if args.model == "unet":
        n_saved = _unet_attention_block_count(model.policy)
        if int(n_saved) != int(args.num_attention_blocks):
            raise ValueError(
                "pretrain U-Net --num-attention-blocks mismatch: "
                f"checkpoint has {n_saved}, CLI has {args.num_attention_blocks}."
            )


def main() -> None:
    _configure_stdio_line_buffered()
    args = parse_args()
    if args.shaped_dense_mult is None:
        args.shaped_dense_mult = (
            15.0 if args.reward_mode == "shaped" and args.shaped_reward_norm == "inv_board_area" else 1.0
        )
    train_sizes = parse_size_list(args.train_sizes, fallback=[args.size])
    eval_sizes = parse_size_list(args.eval_sizes, fallback=train_sizes)
    train_weights = parse_float_list(args.train_weights)
    curriculum_thresholds = parse_float_list(args.curriculum_thresholds)

    inferred_max_size = max(train_sizes + eval_sizes)
    if args.max_size is None:
        args.max_size = inferred_max_size
    if args.max_size < inferred_max_size:
        raise ValueError(f"--max-size ({args.max_size}) must be >= max requested size ({inferred_max_size})")

    if args.n_envs is None:
        args.n_envs = len(train_sizes)
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if args.num_attention_blocks < 0:
        raise ValueError("--num-attention-blocks must be >= 0")
    if int(args.critic_warmup_timesteps) < 0:
        raise ValueError("--critic-warmup-timesteps must be >= 0")
    if train_weights is not None and len(train_weights) != len(train_sizes):
        raise ValueError("--train-weights must match the number of --train-sizes")
    if curriculum_thresholds is not None and len(curriculum_thresholds) != max(len(train_sizes) - 1, 0):
        raise ValueError("--curriculum-thresholds must have len(train_sizes)-1 values")
    if int(args.final_eval_episodes) < 1:
        raise ValueError("--final-eval-episodes must be >= 1")

    eval_suite_seed = int(args.eval_seed) if args.eval_seed is not None else int(args.seed) + 10_000
    total_eval_tasks = len(eval_sizes) * int(args.eval_episodes)
    eval_vec_envs = resolve_eval_vec_envs(args.n_envs, args.eval_vec_envs, total_eval_tasks)
    final_total_eval_tasks = len(eval_sizes) * int(args.final_eval_episodes)
    final_eval_vec_envs = resolve_eval_vec_envs(args.n_envs, args.eval_vec_envs, final_total_eval_tasks)

    if args.out_dir is None:
        args.out_dir = default_out_dir(args.reward_mode, args.model)

    device = resolve_training_device(args.device)
    print(
        f"PPO device: {device} | eval_vec_envs: {eval_vec_envs} (periodic) / {final_eval_vec_envs} (final) | "
        f"train_random_pad_offset={args.train_random_pad_offset} | "
        f"reward_mode={args.reward_mode} shaped_norm={args.shaped_reward_norm} shaped_dense_mult={args.shaped_dense_mult} "
        f"shaped_step_penalty={args.shaped_step_penalty} | critic_warmup_timesteps={args.critic_warmup_timesteps} | "
        f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | mps_available={_mps_available()}",
        flush=True,
    )
    print(
        f"eval suite: seed={eval_suite_seed} periodic_episodes/size={args.eval_episodes} "
        f"final_episodes/size={args.final_eval_episodes} eval_sizes={eval_sizes}",
        flush=True,
    )

    run_dir = args.out_dir / f"{format_board_tag(train_sizes, args.max_size, args.sampling_strategy)}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_suite.json").write_text(
        json.dumps(
            {
                "eval_seed": int(eval_suite_seed),
                "eval_sizes": list(eval_sizes),
                "episode_seed_formula": "episode_seed = eval_seed + index_in_eval_sizes * 1000 + episode_index",
                "periodic_eval_episodes": int(args.eval_episodes),
                "final_eval_episodes": int(args.final_eval_episodes),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.n_envs == 1 and len(train_sizes) == 1 and args.sampling_strategy == "round_robin":
        env = make_env(
            size=train_sizes[0],
            max_size=args.max_size,
            seed=args.seed,
            reward_mode=args.reward_mode,
            dup_coef=args.dup_coef,
            adj_coef=args.adj_coef,
            comp_coef=args.comp_coef,
            shaped_reward_norm=args.shaped_reward_norm,
            shaped_step_penalty=args.shaped_step_penalty,
            shaped_dense_mult=args.shaped_dense_mult,
            random_pad_offset=args.train_random_pad_offset,
        )
    else:
        env, env_descriptions = make_vec_env(
            train_sizes=train_sizes,
            max_size=args.max_size,
            n_envs=args.n_envs,
            seed=args.seed,
            reward_mode=args.reward_mode,
            dup_coef=args.dup_coef,
            adj_coef=args.adj_coef,
            comp_coef=args.comp_coef,
            shaped_reward_norm=args.shaped_reward_norm,
            shaped_step_penalty=args.shaped_step_penalty,
            shaped_dense_mult=args.shaped_dense_mult,
            vec_env_type=args.vec_env,
            sampling_strategy=args.sampling_strategy,
            train_weights=train_weights,
            curriculum_thresholds=curriculum_thresholds,
            random_pad_offset=args.train_random_pad_offset,
        )
        print(f"training env setup: {env_descriptions}", flush=True)

    if args.pretrain_policy is not None:
        ckpt = _resolve_pretrain_checkpoint(args.pretrain_policy)
        print(f"loading --pretrain-policy from {ckpt}", flush=True)
        loaded = MaskablePPO.load(str(ckpt), env=env, device=device, print_system_info=False)
        try:
            _assert_pretrain_policy_matches(model=loaded, env=env, args=args)
            policy, policy_kwargs = make_policy_spec(args.model, num_attention_blocks=args.num_attention_blocks)
            model = MaskablePPO(
                policy,
                env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                ent_coef=args.ent_coef,
                policy_kwargs=policy_kwargs,
                seed=args.seed,
                verbose=1,
                device=device,
            )
            model.policy.load_state_dict(loaded.policy.state_dict(), strict=True)
        finally:
            del loaded
        print(
            f"pretrain OK: policy weights copied from zip; PPO rollout uses CLI "
            f"n_steps={args.n_steps}, batch_size={args.batch_size}, "
            f"lr={args.learning_rate}, ent_coef={args.ent_coef}, gamma={args.gamma}, gae_lambda={args.gae_lambda}",
            flush=True,
        )
        model.set_random_seed(args.seed)
    else:
        policy, policy_kwargs = make_policy_spec(args.model, num_attention_blocks=args.num_attention_blocks)
        model = MaskablePPO(
            policy,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            verbose=1,
            device=device,
        )

    eval_callback = EvalAndCheckpointCallback(
        eval_sizes=eval_sizes,
        max_size=args.max_size,
        out_dir=run_dir,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_seed=int(eval_suite_seed),
        max_steps=args.max_steps,
        reward_mode=args.reward_mode,
        dup_coef=args.dup_coef,
        adj_coef=args.adj_coef,
        comp_coef=args.comp_coef,
        shaped_reward_norm=args.shaped_reward_norm,
        shaped_step_penalty=args.shaped_step_penalty,
        shaped_dense_mult=args.shaped_dense_mult,
        eval_vec_envs=eval_vec_envs,
        total_timesteps=args.total_timesteps,
        eval_at_timestep_zero=args.pretrain_policy is not None,
    )
    train_metrics_jsonl = run_dir / "train_metrics.jsonl"
    cb_list: list[Any] = [
        eval_callback,
        JsonlTrainMetricsCallback(train_metrics_jsonl, verbose=1),
    ]
    if int(args.critic_warmup_timesteps) > 0:
        cb_list.insert(
            0,
            CriticOutputHeadWarmupCallback(until_timesteps=int(args.critic_warmup_timesteps), verbose=1),
        )
    learn_callbacks = CallbackList(cb_list)

    print("starting model.learn ...", flush=True)
    train_learn_t0 = time.perf_counter()
    model.learn(total_timesteps=args.total_timesteps, callback=learn_callbacks, progress_bar=False)
    learn_wall_s = time.perf_counter() - train_learn_t0

    final_model_path = run_dir / f"maskable_ppo_{args.reward_mode}_final"
    model.save(final_model_path)
    env.close()

    print("starting final eval ...", flush=True)
    _t_final_eval = time.perf_counter()
    periodic_eval_wall_s = float(eval_callback._eval_wall_sum)
    pure_train_learn_s = learn_wall_s - periodic_eval_wall_s

    final_rows, final_eval_env_steps = evaluate_all_sizes(
        model=model,
        eval_sizes=eval_sizes,
        max_size=args.max_size,
        episodes=int(args.final_eval_episodes),
        eval_seed=int(eval_suite_seed),
        max_steps=args.max_steps,
        reward_mode=args.reward_mode,
        dup_coef=args.dup_coef,
        adj_coef=args.adj_coef,
        comp_coef=args.comp_coef,
        shaped_reward_norm=args.shaped_reward_norm,
        shaped_step_penalty=args.shaped_step_penalty,
        shaped_dense_mult=args.shaped_dense_mult,
        timesteps=int(model.num_timesteps),
        vec_envs=final_eval_vec_envs,
    )
    _final_eval_s = time.perf_counter() - _t_final_eval
    print(
        f"learn_wall_s={learn_wall_s:.3f} pure_train_wall_s={pure_train_learn_s:.3f} "
        f"periodic_eval_wall_sum_s={periodic_eval_wall_s:.3f} final_eval_wall_s={_final_eval_s:.3f} "
        f"final_eval_env_steps={final_eval_env_steps} final_eval_vec_envs={final_eval_vec_envs} "
        f"sizes={len(eval_sizes)} final_episodes_per_size={args.final_eval_episodes}",
        flush=True,
    )
    with (run_dir / "final_eval.txt").open("w") as f:
        for row in final_rows:
            f.write(
                f"[{row.size}x{row.size}]\n"
                f"timesteps={row.timesteps}\n"
                f"mean_reward={row.mean_reward:.6f}\n"
                f"solve_rate={row.solve_rate:.6f}\n"
                f"mean_length={row.mean_length:.6f}\n\n"
            )

    plot_learning_curves_per_size(run_dir / "learning_curve.csv", run_dir)
    print(f"saved final model: {final_model_path}.zip", flush=True)
    print(f"saved per-size curves: {run_dir}/learning_curve_size*.png", flush=True)
    print(f"saved train metrics jsonl: {train_metrics_jsonl}", flush=True)
    summary = ", ".join(
        (
            f"{row.size}x{row.size}: "
            f"reward={row.mean_reward:.3f}, "
            f"solve={row.solve_rate:.3f}, "
            f"len={row.mean_length:.2f}"
        )
        for row in final_rows
    )
    print(f"final eval: {summary}", flush=True)


if __name__ == "__main__":
    main()
