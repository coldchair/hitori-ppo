#!/usr/bin/env python3
"""
Evaluate a saved MaskablePPO checkpoint (.zip) on a range of board sizes.

Uses the same vectorized eval path as training (``evaluate_all_sizes`` with
``vec_envs`` parallel slots, default 256): one model on device, batched ``predict``,
ThreadPoolExecutor for env step/reset.

Run from repo root::

    python -m ppo.eval_checkpoint runs/.../maskable_ppo_sparse_final.zip \\
        --max-size 20 --min-size 4 --eval-max-size 11 --episodes 50 --vec-envs 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hitori_env  # noqa: F401 - register Hitori-v2
import numpy as np
from sb3_contrib import MaskablePPO

from ppo.env import make_env
from ppo.evaluation import EvalRow, evaluate_all_sizes
from ppo.train import resolve_training_device


def _resolve_checkpoint_zip(path: Path) -> Path:
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
    raise FileNotFoundError(f"checkpoint not found: {path} (resolved: {p})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path, help="Path to MaskablePPO .zip (with or without .zip suffix).")
    p.add_argument(
        "--max-size",
        type=int,
        default=20,
        help="Padded observation side (must match checkpoint); env pads n×n boards into max_size×max_size.",
    )
    p.add_argument("--min-size", type=int, default=4, help="Smallest n×n board side to evaluate (inclusive).")
    p.add_argument(
        "--eval-max-size",
        type=int,
        default=None,
        help="Largest n×n board side to evaluate (inclusive). Default: same as --max-size.",
    )
    p.add_argument("--episodes", type=int, default=50, help="Episodes per board size.")
    p.add_argument("--eval-seed", type=int, default=20_000, help="Base seed offset per size (same scheme as train eval).")
    p.add_argument("--max-steps", type=int, default=1_000, help="Max steps per episode.")
    p.add_argument("--vec-envs", type=int, default=256, help="Parallel eval slots (ThreadPoolExecutor).")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--reward-mode", choices=["sparse", "shaped"], default="sparse")
    p.add_argument("--dup-coef", type=float, default=1.0)
    p.add_argument("--adj-coef", type=float, default=0.5)
    p.add_argument("--comp-coef", type=float, default=1.5)
    p.add_argument("--shaped-reward-norm", choices=["none", "inv_board_area"], default="none")
    p.add_argument("--shaped-step-penalty", type=float, default=-0.05)
    p.add_argument("--shaped-dense-mult", type=float, default=1.0)
    p.add_argument("--out-json", type=Path, default=None, help="Optional path to write metrics JSON.")
    return p.parse_args()


def _rows_to_jsonable(rows: list[EvalRow]) -> list[dict[str, float | int]]:
    return [
        {
            "size": int(r.size),
            "timesteps": int(r.timesteps),
            "mean_reward": float(r.mean_reward),
            "solve_rate": float(r.solve_rate),
            "mean_length": float(r.mean_length),
        }
        for r in rows
    ]


def main() -> None:
    args = parse_args()
    max_eval = int(args.eval_max_size) if args.eval_max_size is not None else int(args.max_size)
    if args.min_size < 1 or max_eval < args.min_size:
        raise SystemExit("invalid --min-size / --eval-max-size range")
    if args.max_size < max_eval:
        raise SystemExit(f"--max-size ({args.max_size}) must be >= --eval-max-size ({max_eval})")

    ckpt = _resolve_checkpoint_zip(args.checkpoint)
    device = resolve_training_device(args.device)
    eval_sizes = list(range(int(args.min_size), max_eval + 1))

    # Template env for SB3 load (spaces must match checkpoint).
    env = make_env(
        size=int(args.min_size),
        max_size=int(args.max_size),
        reward_mode=args.reward_mode,
        dup_coef=args.dup_coef,
        adj_coef=args.adj_coef,
        comp_coef=args.comp_coef,
        shaped_reward_norm=args.shaped_reward_norm,
        shaped_step_penalty=args.shaped_step_penalty,
        shaped_dense_mult=args.shaped_dense_mult,
        random_pad_offset=False,
    )
    try:
        print(f"loading {ckpt} | device={device}", flush=True)
        model = MaskablePPO.load(str(ckpt), env=env, device=device, print_system_info=False)
    except Exception as e:
        env.close()
        raise SystemExit(f"failed to load checkpoint: {e}") from e

    try:
        rows, env_steps = evaluate_all_sizes(
            model=model,
            eval_sizes=eval_sizes,
            max_size=int(args.max_size),
            episodes=int(args.episodes),
            eval_seed=int(args.eval_seed),
            max_steps=int(args.max_steps),
            reward_mode=args.reward_mode,
            dup_coef=args.dup_coef,
            adj_coef=args.adj_coef,
            comp_coef=args.comp_coef,
            shaped_reward_norm=args.shaped_reward_norm,
            shaped_step_penalty=args.shaped_step_penalty,
            shaped_dense_mult=args.shaped_dense_mult,
            timesteps=int(model.num_timesteps),
            vec_envs=int(args.vec_envs),
        )
    finally:
        env.close()

    print(f"eval env steps (sum of lengths): {env_steps}", flush=True)
    print(f"{'size':>4}  {'solve_rate':>10}  {'mean_reward':>12}  {'mean_length':>12}", flush=True)
    for r in rows:
        print(f"{r.size:4d}  {r.solve_rate:10.4f}  {r.mean_reward:12.4f}  {r.mean_length:12.2f}", flush=True)

    overall_solve = float(np.mean([r.solve_rate for r in rows])) if rows else 0.0
    print(f"\nmean solve_rate across sizes: {overall_solve:.4f}", flush=True)

    if args.out_json is not None:
        payload = {
            "checkpoint": str(ckpt),
            "max_size": int(args.max_size),
            "eval_sizes": eval_sizes,
            "episodes_per_size": int(args.episodes),
            "vec_envs": int(args.vec_envs),
            "eval_seed": int(args.eval_seed),
            "model_num_timesteps": int(model.num_timesteps),
            "total_eval_env_steps": int(env_steps),
            "mean_solve_rate_across_sizes": overall_solve,
            "per_size": _rows_to_jsonable(rows),
        }
        out_path = args.out_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
