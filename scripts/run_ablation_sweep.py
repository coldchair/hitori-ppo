#!/usr/bin/env python3
"""
Parallel ablation launcher for ``python -m ppo.train`` on multiple GPUs.

Baseline (fixed across runs except the ablation axes below):
  --model unet --train-sizes 4,5,6,7,8,9,10,11 --eval-sizes 4,5,6,7,8,9,10,11 --max-size 20
  --vec-env subproc --sampling-strategy round_robin
  --n-envs 128 --total-timesteps 2000000 --eval-freq 20000 --eval-episodes 50 --seed 44
  --eval-vec-envs 256

Axes:
  1) --num-attention-blocks in {0, 1, 2}
  2) (--n-steps, --batch-size) scaled together: (32,256), (64,512), (128,1024), (256,2048), (512,4096)
     (chosen so n_steps * n_envs is divisible by batch_size with n_envs=128)
  3) --train-random-pad-offset on or off
  4) --reward-mode in {sparse, shaped}

Runs up to 8 concurrent processes (CUDA 0–7). Blocks until all jobs finish.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def format_board_tag(sizes: list[int], max_size: int, sampling_strategy: str) -> str:
    """Mirror ``ppo.train.format_board_tag`` (avoid importing ``ppo.train`` / torch)."""

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


# Fixed baseline (do not change per user request).
FIXED_ARGS: list[str] = [
    "python",
    "-u",
    "-m",
    "ppo.train",
    "--model",
    "unet",
    "--train-sizes",
    "4,5,6,7,8,9,10,11",
    "--eval-sizes",
    "4,5,6,7,8,9,10,11",
    "--max-size",
    "20",
    "--vec-env",
    "subproc",
    "--sampling-strategy",
    "round_robin",
    "--n-envs",
    "128",
    "--total-timesteps",
    "2000000",
    "--eval-freq",
    "20000",
    "--eval-episodes",
    "50",
    "--seed",
    "44",
    "--eval-vec-envs",
    "256",
    "--device",
    "cuda",
    "--shaped-reward-norm",
    "inv_board_area",
]

OUT_ROOT = "runs/maskable_ppo_unet_mix4_to_11_2M_ablation"

# (n_steps, batch_size): buffer = n_steps * 128 must be divisible by batch_size
STEP_BATCH_GRID: list[tuple[int, int]] = [
    (32, 256),  # x0.25 vs (128,1024)
    (64, 512),  # x0.5
    (128, 1024),  # baseline
    (256, 2048),  # x2
    (512, 4096),  # x4
]

ATTENTION_BLOCKS = [0, 1, 2]
REWARD_MODES: tuple[str, ...] = ("sparse", "shaped")

# Must match FIXED_ARGS / train.py ``run_dir = args.out_dir / f"{format_board_tag(...)}_seed{SEED}"``.
_TRAIN_SIZES: list[int] = [4, 5, 6, 7, 8, 9, 10, 11]
_MAX_SIZE: int = 20
_SAMPLING: str = "round_robin"
_SEED: int = 44


@dataclass(frozen=True)
class Job:
    attn: int
    n_steps: int
    batch_size: int
    random_pad: bool
    reward_mode: str
    out_dir: str
    gpu: int


def _out_dir_tag(attn: int, n_steps: int, batch_size: int, random_pad: bool, reward_mode: str) -> str:
    pad = "pad" if random_pad else "nopad"
    return f"{OUT_ROOT}/att{attn}_ns{n_steps}_bs{batch_size}_{pad}_{reward_mode}_seed44"


def build_jobs() -> list[Job]:
    jobs: list[Job] = []
    for attn in ATTENTION_BLOCKS:
        for n_steps, batch_size in STEP_BATCH_GRID:
            for random_pad in (False, True):
                for reward_mode in REWARD_MODES:
                    jobs.append(
                        Job(
                            attn=attn,
                            n_steps=n_steps,
                            batch_size=batch_size,
                            random_pad=random_pad,
                            reward_mode=reward_mode,
                            out_dir=_out_dir_tag(attn, n_steps, batch_size, random_pad, reward_mode),
                            gpu=-1,
                        )
                    )
    return jobs


def task_run_dir(job: Job) -> Path:
    """Same directory as ``learning_curve.csv`` (``train.py`` ``run_dir``)."""

    tag = format_board_tag(_TRAIN_SIZES, _MAX_SIZE, _SAMPLING)
    return REPO_ROOT / job.out_dir / f"{tag}_seed{_SEED}"


def job_command(job: Job) -> list[str]:
    cmd = list(FIXED_ARGS)
    cmd.extend(
        [
            "--reward-mode",
            job.reward_mode,
            "--num-attention-blocks",
            str(job.attn),
            "--n-steps",
            str(job.n_steps),
            "--batch-size",
            str(job.batch_size),
            "--out-dir",
            job.out_dir,
        ]
    )
    if job.random_pad:
        cmd.append("--train-random-pad-offset")
    return cmd


def run_one(job: Job) -> tuple[Job, int, float]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    cmd = job_command(job)
    run_dir = task_run_dir(job)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "cli.log"
    t0 = time.perf_counter()
    print(f"[GPU {job.gpu}] START {job.out_dir} log={log_path.relative_to(REPO_ROOT)}", flush=True)
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8", buffering=1) as logf:
        logf.write(f"# started_utc={started}\n")
        logf.write(f"# CUDA_VISIBLE_DEVICES={job.gpu}\n")
        logf.write(f"# cwd={REPO_ROOT}\n")
        logf.write(f"# cmd: {shlex.join(cmd)}\n")
        logf.write("# --- subprocess output ---\n")
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.perf_counter() - t0
    with log_path.open("a", encoding="utf-8", buffering=1) as logf:
        logf.write(f"\n# --- end exit_code={proc.returncode} elapsed_s={elapsed:.3f} ---\n")
    print(f"[GPU {job.gpu}] END code={proc.returncode} {elapsed:.1f}s {job.out_dir}", flush=True)
    return job, proc.returncode, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated CUDA device ids to use.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not run.")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpu_ids:
        print("No GPUs in --gpus", file=sys.stderr)
        return 2

    jobs = build_jobs()
    print(f"Total jobs: {len(jobs)} | GPUs: {gpu_ids} | max parallel: {len(gpu_ids)}", flush=True)

    if args.dry_run:
        for i, job in enumerate(jobs):
            gpu_hint = gpu_ids[i % len(gpu_ids)]
            log_path = task_run_dir(job) / "cli.log"
            print(f"(example GPU {gpu_hint}) log -> {log_path.relative_to(REPO_ROOT)}")
            print("  " + " ".join(job_command(job)))
        return 0

    # Assign round-robin GPU id for logging only; scheduling uses free GPU from pool.
    pending = list(jobs)
    free_gpus = list(gpu_ids)
    futures: dict[Future, tuple[Job, int]] = {}
    failed: list[tuple[Job, int]] = []

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        while pending or futures:
            while pending and free_gpus:
                gpu = free_gpus.pop(0)
                job = pending.pop(0)
                job_run = Job(
                    job.attn,
                    job.n_steps,
                    job.batch_size,
                    job.random_pad,
                    job.reward_mode,
                    job.out_dir,
                    gpu,
                )
                fut = ex.submit(run_one, job_run)
                futures[fut] = (job_run, gpu)
            if not futures:
                break
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                job_run, gpu = futures.pop(fut)
                try:
                    _, code, _ = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[GPU {gpu}] EXCEPTION {job_run.out_dir}: {exc}", flush=True)
                    failed.append((job_run, -1))
                    free_gpus.append(gpu)
                    continue
                if code != 0:
                    failed.append((job_run, code))
                free_gpus.append(gpu)
            free_gpus.sort()

    if failed:
        print("\nFailed jobs:", flush=True)
        for job, code in failed:
            print(f"  code={code} {job.out_dir}", flush=True)
        return 1
    print("\nAll jobs completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
