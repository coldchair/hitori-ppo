#!/usr/bin/env python3
"""
Supervised fine-tuning (behavioral cloning) of the U-Net Maskable policy on solver data.

Builds step-level (observation, action_mask, action) tuples by replaying expert shading in
**row-major order**: at each step, among cells that are black in the solution but not yet
shaded, pick the first (top-to-bottom, left-to-right) that is currently legal under the
env action mask (mimics scanning the board like a human).

Saves a MaskablePPO checkpoint compatible with ``python -m ppo.train`` / ``MaskablePPO.load``
(best validation cross-entropy).

Train/validation split is by **whole puzzle (one .npz trajectory)**, never by mixing steps
from the same board across train and val.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import hitori_env  # noqa: F401 - registers env
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hitori_env.envs.hitori import HitoriEnv

from ppo.env import ActionMaskWrapper, PadToMaxBoardSizeWrapper, make_env
from ppo.policies import make_policy_spec
from ppo.train import resolve_training_device


def _unwrap_hitori(env: gym.Env) -> HitoriEnv:
    cur: gym.Env = env
    while not isinstance(cur, HitoriEnv):
        if not hasattr(cur, "env"):
            raise TypeError(f"could not find HitoriEnv under {type(env)}")
        cur = cur.env
    return cur


def _sync_hitori_state(core: HitoriEnv, game_grid: np.ndarray, shaded: np.ndarray) -> None:
    core._game_grid = game_grid.astype(np.uint32, copy=True)
    core._shaded = shaded.astype(bool, copy=True)
    core._row_counts = [Counter(core._game_grid[r, :]) for r in range(core.size)]
    core._col_counts = [Counter(core._game_grid[:, c]) for c in range(core.size)]
    core._articulation_points = set()
    core._action_mask = core._compute_next_action_mask()


def _padded_observation(pad_env: PadToMaxBoardSizeWrapper, core: HitoriEnv) -> dict[str, np.ndarray]:
    return pad_env._pad_obs(core._get_obs())


def _inner_to_padded_action(inner_row: int, inner_col: int, max_n: int, dr: int, dc: int) -> int:
    r = dr + inner_row
    c = dc + inner_col
    return int(r * max_n + c)


def _build_trajectory_row_major(
    pad_env: PadToMaxBoardSizeWrapper,
    core: HitoriEnv,
    game: np.ndarray,
    target_shaded: np.ndarray,
    max_n: int,
) -> list[tuple[dict[str, np.ndarray], np.ndarray, int]] | None:
    """
    Return list of (padded_obs, padded_action_mask, padded_action_index).
    Row-major greedy among legal cells that are still-to-shade in the target.
    """

    n = int(core.size)
    if game.shape != (n, n) or target_shaded.shape != (n, n):
        return None

    _sync_hitori_state(core, game, np.zeros((n, n), dtype=bool))
    dr, dc = int(pad_env._pad_dr), int(pad_env._pad_dc)

    steps: list[tuple[dict[str, np.ndarray], np.ndarray, int]] = []
    remaining = int(target_shaded.sum())
    if remaining == 0:
        return None

    while remaining > 0:
        obs = _padded_observation(pad_env, core)
        mask = np.asarray(pad_env.action_masks(), dtype=np.int8).reshape(max_n, max_n)
        chosen: int | None = None
        for i in range(n):
            for j in range(n):
                if not target_shaded[i, j] or core._shaded[i, j]:
                    continue
                a = _inner_to_padded_action(i, j, max_n, dr, dc)
                r, c = divmod(a, max_n)
                if mask[r, c] == 1:
                    chosen = a
                    break
            if chosen is not None:
                break
        if chosen is None:
            return None

        mask_flat = np.asarray(pad_env.action_masks(), dtype=np.int8)
        steps.append((obs, mask_flat, chosen))

        _, _, terminated, _, _ = pad_env.step(chosen)
        if terminated and int(target_shaded.sum() - core._shaded.sum()) > 0:
            return None
        remaining = int(target_shaded.sum() - core._shaded.sum())

    if not np.array_equal(core._shaded, target_shaded):
        return None
    return steps


def _load_npz_paths(data_root: Path, min_size: int, max_size: int, per_size: int) -> list[tuple[int, Path]]:
    paths: list[tuple[int, Path]] = []
    for sz in range(min_size, max_size + 1):
        d = data_root / f"size_{sz:02d}"
        if not d.is_dir():
            continue
        files = sorted(d.glob("item_*.npz"))
        if per_size > 0:
            files = files[:per_size]
        for p in files:
            paths.append((sz, p))
    return paths


def _load_sample(path: Path, sz: int) -> tuple[np.ndarray, np.ndarray] | None:
    z = np.load(path)
    g = z["game_grid"]
    s = z["shaded"]
    if g.shape != (sz, sz) or s.shape != (sz, sz):
        return None
    return g.astype(np.uint32), s.astype(bool)


def _obs_to_tensors(
    batch_obs: list[dict[str, np.ndarray]], device: torch.device
) -> dict[str, torch.Tensor]:
    game = torch.stack([torch.as_tensor(o["game_grid"], dtype=torch.float32) for o in batch_obs]).to(device)
    shaded = torch.stack([torch.as_tensor(o["shaded"], dtype=torch.float32) for o in batch_obs]).to(device)
    return {"game_grid": game, "shaded": shaded}


def _masks_to_numpy(action_masks: torch.Tensor) -> np.ndarray:
    return action_masks.detach().cpu().numpy().astype(np.int8)


def _sft_loss_batch(
    policy: torch.nn.Module,
    obs: dict[str, torch.Tensor],
    action_masks: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    dist = policy.get_distribution(obs, action_masks=_masks_to_numpy(action_masks))
    logp = dist.log_prob(actions)
    return -logp.mean()


def _accuracy_batch(
    policy: torch.nn.Module,
    obs: dict[str, torch.Tensor],
    action_masks: torch.Tensor,
    actions: torch.Tensor,
) -> float:
    with torch.no_grad():
        dist = policy.get_distribution(obs, action_masks=_masks_to_numpy(action_masks))
        logits = dist.distribution.logits
        pred = logits.argmax(dim=-1)
        return float((pred == actions).float().mean().item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=REPO / "runs/solver_supervision_mix4_20")
    p.add_argument("--min-size", type=int, default=4)
    p.add_argument("--max-size", type=int, default=20)
    p.add_argument("--max-n", type=int, default=20, help="Padded board side (must match train.py / checkpoint).")
    p.add_argument("--per-size", type=int, default=100, help="Max .npz files per size; <=0 means all.")
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of **puzzles** (full trajectories / env episodes) held out for validation, not step fraction.",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--num-attention-blocks", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out-dir", type=Path, default=REPO / "runs/sft_unet_solver_bc")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_s = resolve_training_device(args.device)
    device = torch.device(device_s)

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"data root not found: {data_root}")

    max_n = int(args.max_n)
    paths = _load_npz_paths(data_root, args.min_size, args.max_size, int(args.per_size))
    if not paths:
        raise SystemExit(f"no npz under {data_root} for sizes {args.min_size}-{args.max_size}")

    env_cache: dict[int, tuple[PadToMaxBoardSizeWrapper, HitoriEnv]] = {}

    def get_env(n: int) -> tuple[PadToMaxBoardSizeWrapper, HitoriEnv]:
        if n not in env_cache:
            e = make_env(
                n,
                max_n,
                reward_mode="sparse",
                random_pad_offset=False,
            )
            core = _unwrap_hitori(e)
            pad = e
            if not isinstance(pad, PadToMaxBoardSizeWrapper):
                raise TypeError("expected PadToMaxBoardSizeWrapper outer")
            env_cache[n] = (pad, core)
        return env_cache[n]

    StepRec = tuple[dict[str, np.ndarray], np.ndarray, int]
    trajectories: list[list[StepRec]] = []
    skipped = 0
    for sz, path in paths:
        loaded = _load_sample(path, sz)
        if loaded is None:
            skipped += 1
            continue
        game, target = loaded
        pad, core = get_env(sz)
        pad.reset(seed=0)
        traj = _build_trajectory_row_major(pad, core, game, target, max_n)
        if traj is None:
            skipped += 1
            continue
        trajectories.append(traj)

    if not trajectories:
        raise SystemExit("no valid trajectories (all skipped); check solver data / ordering")

    rng = random.Random(args.seed)
    rng.shuffle(trajectories)

    n_env = len(trajectories)
    if n_env >= 10:
        n_val_envs = max(1, int(n_env * float(args.val_ratio)))
    else:
        n_val_envs = max(1, n_env // 10)
    if n_env > 1:
        n_val_envs = min(n_val_envs, n_env - 1)
    else:
        n_val_envs = 0

    val_trajs = trajectories[:n_val_envs]
    train_trajs = trajectories[n_val_envs:]
    train_records: list[StepRec] = [step for t in train_trajs for step in t]
    val_records: list[StepRec] = [step for t in val_trajs for step in t]

    n_steps_train = len(train_records)
    n_steps_val = len(val_records)

    print(
        f"data_root={data_root}\n"
        f"puzzles: total={n_env}  train={len(train_trajs)}  val={len(val_trajs)}  skipped={skipped}\n"
        f"steps: train={n_steps_train}  val={n_steps_val}  (split by puzzle, not by step)\n"
        f"max_n={max_n}  device={device_s}",
        flush=True,
    )

    vec = DummyVecEnv(
        [
            lambda: make_env(
                args.min_size,
                max_n,
                reward_mode="sparse",
                random_pad_offset=False,
            )
        ]
    )
    policy_cls, policy_kwargs = make_policy_spec("unet", num_attention_blocks=args.num_attention_blocks)
    # SB3 requires n_steps/batch_size for the wrapper; BC never calls learn(). train.py --pretrain-policy
    # copies policy.load_state_dict only and builds PPO with its own --n-steps / --batch-size.
    model = MaskablePPO(
        policy_cls,
        vec,
        learning_rate=args.lr,
        n_steps=128,
        batch_size=min(args.batch_size, max(1, len(train_records))),
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        verbose=0,
        device=device_s,
    )
    policy = model.policy.to(device)

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    def iter_batches(recs: list[tuple[dict[str, np.ndarray], np.ndarray, int]], bs: int, shuffle: bool):
        idx = list(range(len(recs)))
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            sel = idx[i : i + bs]
            batch = [recs[j] for j in sel]
            obs_list = [b[0] for b in batch]
            masks = torch.stack([torch.as_tensor(b[1], dtype=torch.float32) for b in batch]).to(device)
            acts = torch.stack([torch.tensor(b[2], dtype=torch.long, device=device) for b in batch])
            yield _obs_to_tensors(obs_list, device), masks, acts

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(int(args.epochs)):
        t0 = time.perf_counter()
        policy.train()
        train_losses: list[float] = []
        for obs, masks, acts in iter_batches(train_records, args.batch_size, shuffle=True):
            optim.zero_grad(set_to_none=True)
            loss = _sft_loss_batch(policy, obs, masks, acts)
            loss.backward()
            optim.step()
            train_losses.append(float(loss.item()))

        policy.eval()
        val_losses: list[float] = []
        val_accs: list[float] = []
        with torch.no_grad():
            for obs, masks, acts in iter_batches(val_records, args.batch_size, shuffle=False):
                loss = _sft_loss_batch(policy, obs, masks, acts)
                val_losses.append(float(loss.item()))
                val_accs.append(_accuracy_batch(policy, obs, masks, acts))

        tr = float(np.mean(train_losses)) if train_losses else 0.0
        vl = float(np.mean(val_losses)) if val_losses else float("nan")
        va = float(np.mean(val_accs)) if val_accs else float("nan")
        history.append({"epoch": epoch, "train_nll": tr, "val_nll": vl, "val_acc": va, "wall_s": time.perf_counter() - t0})
        val_str = f"{vl:.4f}" if val_losses else "nan"
        acc_str = f"{va:.4f}" if val_accs else "nan"
        print(f"epoch {epoch+1}/{args.epochs}  train_nll={tr:.4f}  val_nll={val_str}  val_acc={acc_str}", flush=True)

        if val_losses and math.isfinite(vl) and vl < best_val:
            best_val = vl
            best_epoch = epoch
            best_path = out_dir / "sft_maskable_ppo_best"
            model.save(str(best_path))
            meta = {
                "best_epoch": best_epoch,
                "best_val_nll": best_val,
                "val_acc_at_best": float(va),
                "max_n": max_n,
                "num_attention_blocks": args.num_attention_blocks,
                "min_size": args.min_size,
                "max_size": args.max_size,
                "data_root": str(data_root),
                "train_steps": len(train_records),
                "val_steps": len(val_records),
            }
            (out_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    final_path = out_dir / "sft_maskable_ppo_last"
    model.save(str(final_path))

    best_zip = out_dir / "sft_maskable_ppo_best.zip"
    summary = {
        "best_epoch": best_epoch,
        "best_val_nll": best_val if best_epoch >= 0 else None,
        "history": history,
        "saved_best": str(best_zip) if best_zip.is_file() else None,
        "saved_last": str(out_dir / "sft_maskable_ppo_last.zip"),
    }
    (out_dir / "sft_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    load_hint = str(out_dir / "sft_maskable_ppo_best") if best_zip.is_file() else str(out_dir / "sft_maskable_ppo_last")
    if best_epoch >= 0 and math.isfinite(best_val):
        print(f"\nBest val NLL={best_val:.4f} at epoch {best_epoch + 1}", flush=True)
    else:
        print("\nNo validation improvement tracked; use last checkpoint.", flush=True)
    print(
        f"Load: MaskablePPO.load({load_hint!r}, env=..., device=...)\n"
        f"(SB3 writes {load_hint}.zip)",
        flush=True,
    )
    vec.close()


if __name__ == "__main__":
    main()
