#!/usr/bin/env python3
"""
Run SFT ``train_sft`` ablations over ``--per-size`` (data budget per board side), then
``eval_checkpoint`` on each ``sft_maskable_ppo_best.zip``.

Default: **12** ``per_size`` tiers from 20 to 1000, **denser at small N** (similar spirit to
20,50,100,200,500,1000 extended to 12 points). Default GPUs CUDA **2–7** (six cards): jobs are
**paired small+large** (smallest with largest, etc.), **one pair per GPU**, two train+eval runs
per GPU **sequentially** so two jobs never share the same GPU concurrently. Use ``--no-gpu-pairing``
for the old round-robin schedule (many jobs can run on the same GPU at once if workers allow).

Each experiment is a **sibling** folder named ``per_size_<N>`` (no extra campaign directory
wrapping the checkpoints).

- If ``--out-root`` is ``runs/sft_ablation`` → runs go to ``runs/sft_ablation/per_size_<N>/``,
  summaries to ``runs/sft_ablation/``.
- If ``--out-root`` is ``runs/sft_ablation/solver_bc_20`` (``…/sft_ablation/<tag>``) → runs are
  **flattened** to ``runs/sft_ablation/per_size_<N>/`` (not under ``<tag>``), while
  ``ablation_summary.*`` still go under ``runs/sft_ablation/solver_bc_20/``.
  If you run **several** such campaigns with the same ``per_size`` grid, use distinct
  one-level roots (e.g. ``runs/sft_ablation_bc20``) to avoid clobbering the same
  ``per_size_*`` folders.

Example (repo root)::

    python scripts/sft_per_size_ablation.py \\
        --data-root runs/solver_supervision_mix4_20 \\
        --out-root runs/sft_ablation \\
        --gpus 2,3,4,5,6,7

Override tiers::

    python scripts/sft_per_size_ablation.py --per-sizes 20,200,400,600,800,1000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def _rel_to_repo(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p.resolve())


# Twelve tiers 20..1000: more points at small per-size (gaps grow toward 1000).
DEFAULT_PER_SIZES = [20, 35, 55, 80, 120, 180, 260, 360, 500, 650, 820, 1000]
DEFAULT_GPUS = [2, 3, 4, 5, 6, 7]


def _runs_parent_and_summary_dir(out_root: Path) -> tuple[Path, Path]:
    """
    ``out_root`` = where ablation_summary.* is written.

    Per-size training dirs always live under ``runs_parent / f'per_size_{ps}'``:
    - ``runs/sft_ablation`` -> runs_parent = out_root, summary = out_root
    - ``runs/sft_ablation/tag`` -> runs_parent = parent (sft_ablation), summary = out_root (tag/)
    """
    out_root = out_root.resolve()
    parts = out_root.parts
    if len(parts) >= 2 and parts[-2] == "sft_ablation":
        return out_root.parent, out_root
    return out_root, out_root


def _parse_int_list(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty list")
    return [int(p) for p in parts]


def _run_one(
    *,
    per_size: int,
    gpu: int,
    run_dir: Path,
    python_exe: str,
    data_root: Path,
    min_size: int,
    max_size: int,
    max_n: int,
    val_ratio: float,
    epochs: int,
    batch_size: int,
    lr: float,
    num_attention_blocks: int,
    seed: int,
    eval_episodes: int,
    eval_vec_envs: int,
    eval_seed: int,
    reward_mode: str,
    shaped_reward_norm: str,
    shaped_step_penalty: float,
    shaped_dense_mult: float,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "ablation_worker.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    def _sh(cmd: list[str], phase: str) -> None:
        line = " ".join(cmd)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"\n### {phase} gpu={gpu} per_size={per_size}\n{line}\n")
            lf.flush()
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        wall = time.perf_counter() - t0
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(proc.stdout or "")
            lf.write(f"\nexit={proc.returncode} wall_s={wall:.1f}\n")
            lf.flush()
        if proc.returncode != 0:
            raise RuntimeError(f"{phase} failed (exit {proc.returncode}) log={log_path}")

    train_cmd = [
        python_exe,
        "-m",
        "ppo.train_sft",
        "--data-root",
        str(data_root),
        "--min-size",
        str(min_size),
        "--max-size",
        str(max_size),
        "--max-n",
        str(max_n),
        "--per-size",
        str(per_size),
        "--val-ratio",
        str(val_ratio),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--num-attention-blocks",
        str(num_attention_blocks),
        "--out-dir",
        str(run_dir),
        "--device",
        "cuda",
        "--seed",
        str(seed),
    ]
    _sh(train_cmd, "train_sft")

    ckpt = run_dir / "sft_maskable_ppo_best.zip"
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint after SFT: {ckpt}")

    eval_json = run_dir / "eval_after_sft.json"
    eval_cmd = [
        python_exe,
        "-m",
        "ppo.eval_checkpoint",
        str(ckpt),
        "--max-size",
        str(max_n),
        "--min-size",
        str(min_size),
        "--eval-max-size",
        str(max_size),
        "--episodes",
        str(eval_episodes),
        "--vec-envs",
        str(eval_vec_envs),
        "--eval-seed",
        str(eval_seed),
        "--device",
        "cuda",
        "--reward-mode",
        reward_mode,
        "--shaped-reward-norm",
        shaped_reward_norm,
        "--shaped-step-penalty",
        str(shaped_step_penalty),
        "--shaped-dense-mult",
        str(shaped_dense_mult),
        "--out-json",
        str(eval_json),
    ]
    _sh(eval_cmd, "eval_checkpoint")

    with eval_json.open(encoding="utf-8") as f:
        ev = json.load(f)
    row: dict[str, Any] = {
        "per_size": per_size,
        "gpu": gpu,
        "run_dir": _rel_to_repo(run_dir),
        "mean_solve_rate_across_sizes": float(ev.get("mean_solve_rate_across_sizes", 0.0)),
    }
    for item in ev.get("per_size", []):
        sz = int(item["size"])
        row[f"solve_rate_{sz}"] = float(item["solve_rate"])
        row[f"mean_reward_{sz}"] = float(item["mean_reward"])
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", type=Path, default=REPO / "runs/sft_ablation")
    p.add_argument(
        "--per-sizes",
        type=str,
        default=",".join(str(x) for x in DEFAULT_PER_SIZES),
        help=f"Comma-separated per-size limits (default: {DEFAULT_PER_SIZES}).",
    )
    p.add_argument("--gpus", type=str, default=",".join(str(g) for g in DEFAULT_GPUS), help="Comma-separated CUDA device ids.")
    p.add_argument("--data-root", type=Path, default=REPO / "runs/solver_supervision_mix4_20")
    p.add_argument("--min-size", type=int, default=4)
    p.add_argument("--max-size", type=int, default=20)
    p.add_argument("--max-n", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-attention-blocks", type=int, default=0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--eval-vec-envs", type=int, default=256)
    p.add_argument("--eval-seed", type=int, default=20_000)
    p.add_argument("--reward-mode", choices=["sparse", "shaped"], default="sparse")
    p.add_argument("--shaped-reward-norm", choices=["none", "inv_board_area"], default="none")
    p.add_argument("--shaped-step-penalty", type=float, default=-0.05)
    p.add_argument("--shaped-dense-mult", type=float, default=1.0)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Thread pool size. Default: with GPU pairing, one thread per GPU that has work; "
        "with --no-gpu-pairing, min(len(per_sizes), len(gpus)).",
    )
    p.add_argument(
        "--no-gpu-pairing",
        action="store_true",
        help="Legacy: assign GPUs round-robin per experiment (multiple jobs may use the same GPU at once).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned runs only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    per_sizes = _parse_int_list(args.per_sizes)
    gpus = _parse_int_list(args.gpus)
    if not per_sizes:
        raise SystemExit("no per-sizes")
    if len(set(per_sizes)) != len(per_sizes):
        print("warning: duplicate values in --per-sizes; pairing / run dirs may be ambiguous.", flush=True)

    out_root: Path = args.out_root.resolve()
    runs_parent, summary_dir = _runs_parent_and_summary_dir(out_root)
    runs_parent.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    python_exe = str(args.python.resolve())

    n = len(per_sizes)
    jobs: list[tuple[int, int, Path]] = []
    by_gpu: dict[int, list[tuple[int, Path]]] | None = None
    workers: int

    if args.no_gpu_pairing:
        if len(gpus) < n:
            print(
                f"warning: fewer GPUs ({len(gpus)}) than experiments ({n}); GPUs will repeat round-robin.",
                flush=True,
            )
        workers = args.max_workers if args.max_workers is not None else min(n, max(1, len(gpus)))
        workers = max(1, min(workers, n))
        for i, ps in enumerate(per_sizes):
            gpu = int(gpus[i % len(gpus)])
            run_dir = runs_parent / f"per_size_{ps}"
            jobs.append((ps, gpu, run_dir))
    else:
        if n % 2 != 0:
            raise SystemExit(
                f"GPU small/large pairing needs an even number of --per-sizes (got {n}). "
                "Use --no-gpu-pairing for an odd count."
            )
        sizes_sorted = sorted(per_sizes)
        half = n // 2
        pairs: list[tuple[int, int]] = [
            (sizes_sorted[k], sizes_sorted[n - 1 - k]) for k in range(half)
        ]
        by_gpu = defaultdict(list)
        for j, (s_lo, s_hi) in enumerate(pairs):
            gpu = int(gpus[j % len(gpus)])
            by_gpu[gpu].append((s_lo, runs_parent / f"per_size_{s_lo}"))
            by_gpu[gpu].append((s_hi, runs_parent / f"per_size_{s_hi}"))
        jobs = [(ps, g, rd) for g in sorted(by_gpu) for ps, rd in by_gpu[g]]
        n_shards = len(by_gpu)
        workers = args.max_workers if args.max_workers is not None else n_shards
        workers = max(1, min(workers, n_shards))

    print(
        f"repo={REPO}\nout_root={out_root}\nruns_parent={runs_parent}\nsummary_dir={summary_dir}\n"
        f"gpu_pairing={not args.no_gpu_pairing} jobs={len(jobs)} workers={workers}",
        flush=True,
    )
    if by_gpu is None:
        for ps, gpu, rd in jobs:
            print(f"  per_size={ps:4d}  gpu={gpu}  -> {_rel_to_repo(rd)}", flush=True)
    else:
        for gpu in sorted(by_gpu):
            seq = [ps for ps, _ in by_gpu[gpu]]
            print(f"  gpu={gpu}  sequential per_size={seq}", flush=True)

    if args.dry_run:
        return

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    def _run_kwargs() -> dict[str, Any]:
        return dict(
            python_exe=python_exe,
            data_root=args.data_root.resolve(),
            min_size=int(args.min_size),
            max_size=int(args.max_size),
            max_n=int(args.max_n),
            val_ratio=float(args.val_ratio),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            num_attention_blocks=int(args.num_attention_blocks),
            seed=int(args.seed),
            eval_episodes=int(args.eval_episodes),
            eval_vec_envs=int(args.eval_vec_envs),
            eval_seed=int(args.eval_seed),
            reward_mode=str(args.reward_mode),
            shaped_reward_norm=str(args.shaped_reward_norm),
            shaped_step_penalty=float(args.shaped_step_penalty),
            shaped_dense_mult=float(args.shaped_dense_mult),
        )

    def _task(tup: tuple[int, int, Path]) -> dict[str, Any]:
        ps, gpu, rd = tup
        return _run_one(per_size=ps, gpu=gpu, run_dir=rd, **_run_kwargs())

    def _task_gpu_shard(gpu: int, shard: list[tuple[int, Path]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ps, rd in shard:
            out.append(_run_one(per_size=ps, gpu=gpu, run_dir=rd, **_run_kwargs()))
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        if args.no_gpu_pairing:
            futs = {ex.submit(_task, j): j for j in jobs}
            for fut in as_completed(futs):
                ps, gpu, _ = futs[fut]
                try:
                    results.append(fut.result())
                    print(f"[ok] per_size={ps} gpu={gpu}", flush=True)
                except Exception as e:
                    errors.append(f"per_size={ps} gpu={gpu}: {e}")
                    print(f"[fail] per_size={ps} gpu={gpu}: {e}", flush=True)
        else:
            assert by_gpu is not None
            futs = {ex.submit(_task_gpu_shard, g, by_gpu[g]): g for g in sorted(by_gpu)}
            for fut in as_completed(futs):
                gpu = futs[fut]
                try:
                    rows = fut.result()
                    results.extend(rows)
                    for row in rows:
                        print(f"[ok] per_size={row['per_size']} gpu={gpu}", flush=True)
                except Exception as e:
                    errors.append(f"gpu={gpu} shard: {e}")
                    print(f"[fail] gpu={gpu} shard: {e}", flush=True)

    results.sort(key=lambda r: int(r["per_size"]))
    summary_path = summary_dir / "ablation_summary.json"
    summary_path.write_text(
        json.dumps({"results": results, "errors": errors}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {summary_path}", flush=True)

    # Flat CSV for spreadsheets
    if results:
        sizes = list(range(int(args.min_size), int(args.max_size) + 1))
        fieldnames = ["per_size", "gpu", "mean_solve_rate_across_sizes"] + [f"solve_rate_{s}" for s in sizes]
        csv_path = summary_dir / "ablation_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"wrote {csv_path}", flush=True)

    if errors:
        raise SystemExit(f"completed with {len(errors)} failure(s); see ablation_summary.json errors[]")


if __name__ == "__main__":
    main()
