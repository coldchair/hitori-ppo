#!/usr/bin/env python3
"""
Run all ``final_exp`` PPO jobs (groups 0–4) with a **global order**: lower group first,
then jobs listed in definition order. Eight worker threads bind to CUDA devices 0–7;
each thread repeatedly pulls the next job from a queue (one GPU, many jobs serially).

Output layout::

    runs/final_exp/{group}/{slug}/cli.log   # full launcher + ``ppo.train`` stdout/stderr
    runs/final_exp/{group}/{slug}/<run_tag>_seed44/...   # SB3 checkpoints / curves

``slug`` is a short filesystem-safe name; ``train.py`` still appends ``*_seed44`` under ``--out-dir``.

Group 1 singles use **500k** timesteps; the group-1 mix run still uses **2M**.

Example::

    cd /path/to/hitori-ppo && python scripts/run_final_exps.py
    python scripts/run_final_exps.py --dry-run
    python scripts/run_final_exps.py --gpus 0,1,2,3
    python scripts/run_final_exps.py --groups 2          # only length-extrapolation jobs
    python scripts/run_final_exps.py --groups 0,1,3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PRETRAIN_ZIP = REPO / "runs/sft_ablation/per_size_1000/sft_maskable_ppo_best.zip"


def _sizes(a: int, b: int) -> str:
    return ",".join(str(i) for i in range(a, b + 1))


def _common_prefix(py: str) -> list[str]:
    return [
        py,
        "-m",
        "ppo.train",
        "--learning-rate",
        "1e-4",
        "--max-size",
        "20",
        "--vec-env",
        "subproc",
        "--sampling-strategy",
        "round_robin",
        "--n-envs",
        "128",
        "--batch-size",
        "256",
        "--n-steps",
        "32",
        "--eval-freq",
        "50000",
        "--eval-episodes",
        "50",
        "--seed",
        "44",
        "--eval-vec-envs",
        "256",
        "--device",
        "cuda",
    ]


def build_jobs() -> list[tuple[int, str, Path, list[str]]]:
    """(group_id, slug, out_dir_abs, argv tail after common prefix, including ``--out-dir``)."""
    out_base = "runs/final_exp"
    jobs: list[tuple[int, str, Path, list[str]]] = []

    def od(group: int, slug: str, tail: list[str]) -> None:
        rel = f"{out_base}/{group}/{slug}"
        out_abs = (REPO / rel).resolve()
        jobs.append((group, slug, out_abs, tail + ["--out-dir", rel]))

    # ----- group 0: final recipe, 4M, shaped, pretrain + 100k critic warmup -----
    common_g0 = [
        "--model",
        "unet",
        "--num-attention-blocks",
        "0",
        "--total-timesteps",
        "4000000",
        "--reward-mode",
        "shaped",
        "--critic-warmup-timesteps",
        "100000",
        "--pretrain-policy",
        str(PRETRAIN_ZIP),
    ]
    od(0, "unet_train_eval_4-11", common_g0 + ["--train-sizes", _sizes(4, 11), "--eval-sizes", _sizes(4, 11)])
    od(0, "unet_train_eval_4-15", common_g0 + ["--train-sizes", _sizes(4, 15), "--eval-sizes", _sizes(4, 15)])
    od(0, "unet_train_eval_4-20", common_g0 + ["--train-sizes", _sizes(4, 20), "--eval-sizes", _sizes(4, 20)])

    # ----- group 1: no pretrain / warmup, shaped; singles (500k) + mix (2M) -----
    common_no_pt_2m = [
        "--model",
        "unet",
        "--num-attention-blocks",
        "0",
        "--total-timesteps",
        "2000000",
        "--reward-mode",
        "shaped",
        "--critic-warmup-timesteps",
        "0",
    ]
    common_no_pt_single = [
        "--model",
        "unet",
        "--num-attention-blocks",
        "0",
        "--total-timesteps",
        "500000",
        "--reward-mode",
        "shaped",
        "--critic-warmup-timesteps",
        "0",
    ]
    for s in range(4, 12):
        od(1, f"single_train{s}_eval{s}", common_no_pt_single + ["--train-sizes", str(s), "--eval-sizes", str(s)])
    od(1, "mix_train_4-11_eval_4-11", common_no_pt_2m + ["--train-sizes", _sizes(4, 11), "--eval-sizes", _sizes(4, 11)])

    # ----- group 2: length extrapolation, 2M -----
    ev = _sizes(4, 15)
    for train_hi in (4, 5, 6, 7, 8):
        ts = _sizes(4, train_hi)
        od(2, f"extrap_train_{ts.replace(',', '-')}_eval_4-15", common_no_pt_2m + ["--train-sizes", ts, "--eval-sizes", ev])

    # ----- group 3: architectures, 2M -----
    ts311, ev415 = _sizes(4, 11), _sizes(4, 15)
    for m in ("unet", "cnn", "cnnv2", "structured"):
        extra = [
            "--model",
            m,
            "--num-attention-blocks",
            "0",
            "--total-timesteps",
            "2000000",
            "--reward-mode",
            "shaped",
            "--critic-warmup-timesteps",
            "0",
            "--train-sizes",
            ts311,
            "--eval-sizes",
            ev415,
        ]
        od(3, f"model_{m}_train_4-11_eval_4-15", extra)

    # ----- group 4: sparse/shaped × pretrain/no, 2M, unet -----
    ts311, ev415 = _sizes(4, 11), _sizes(4, 15)

    def g4(slug: str, reward_mode: str, *, pretrain: bool) -> None:
        tail = [
            "--model",
            "unet",
            "--num-attention-blocks",
            "0",
            "--train-sizes",
            ts311,
            "--eval-sizes",
            ev415,
            "--total-timesteps",
            "2000000",
            "--reward-mode",
            reward_mode,
        ]
        if pretrain:
            tail += [
                "--pretrain-policy",
                str(PRETRAIN_ZIP),
                "--critic-warmup-timesteps",
                "100000",
            ]
        else:
            tail += ["--critic-warmup-timesteps", "0"]
        od(4, slug, tail)

    g4("sparse_no_pretrain", "sparse", pretrain=False)
    g4("sparse_pretrain_warm100k", "sparse", pretrain=True)
    g4("shaped_no_pretrain", "shaped", pretrain=False)
    g4("shaped_pretrain_warm100k", "shaped", pretrain=True)

    return jobs


def _tail_uses_pretrain(tail: list[str]) -> bool:
    return "--pretrain-policy" in tail


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated CUDA device ids (one worker each).")
    p.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated experiment group ids to run (default: all). Known groups: 0–4.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned commands only.")
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    args = p.parse_args()

    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise SystemExit("no GPUs in --gpus")

    py = str(args.python.resolve())
    prefix = _common_prefix(py)
    jobs = build_jobs()

    if args.groups is not None:
        allow = {int(x.strip()) for x in args.groups.split(",") if x.strip()}
        jobs = [j for j in jobs if j[0] in allow]
        if not jobs:
            raise SystemExit(f"no jobs left after --groups {args.groups!r} (allowed={sorted(allow)})")

    if not args.dry_run and any(_tail_uses_pretrain(tail) for _, _, _, tail in jobs) and not PRETRAIN_ZIP.is_file():
        raise SystemExit(f"pretrain zip not found (at least one selected job needs it): {PRETRAIN_ZIP}")

    print(f"repo={REPO} jobs={len(jobs)} workers={len(gpus)} gpus={gpus}", flush=True)
    for i, (g, slug, out_abs, tail) in enumerate(jobs):
        cmd = prefix + tail
        print(f"  [{i+1:02d}/{len(jobs)}] group={g} slug={slug} log={out_abs / 'cli.log'}", flush=True)
        if args.dry_run:
            print("      " + " ".join(cmd), flush=True)

    if args.dry_run:
        return

    lock = threading.Lock()
    next_i = 0
    errors: list[str] = []

    def worker(gpu: int) -> None:
        nonlocal next_i
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        while True:
            with lock:
                global_idx = next_i
                if global_idx >= len(jobs):
                    return
                next_i += 1
                group, slug, out_abs, tail = jobs[global_idx]
            cmd = prefix + tail
            out_abs.mkdir(parents=True, exist_ok=True)
            log_path = out_abs / "cli.log"
            wall_t0 = time.perf_counter()
            print(
                f"[start] job={global_idx + 1}/{len(jobs)} group={group} slug={slug} gpu={gpu} log={log_path}",
                flush=True,
            )
            with log_path.open("w", encoding="utf-8") as lf:
                lf.write(f"CUDA_VISIBLE_DEVICES={gpu}\n{' '.join(cmd)}\n\n")
                lf.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(REPO),
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            wall = time.perf_counter() - wall_t0
            if proc.returncode != 0:
                msg = f"group={group} slug={slug} gpu={gpu} exit={proc.returncode} log={log_path}"
                with lock:
                    errors.append(msg)
                print(f"[fail] {msg} wall_s={wall:.1f}", flush=True)
            else:
                print(f"[ok] group={group} slug={slug} gpu={gpu} wall_s={wall:.1f}", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,), name=f"w{gpu}") for gpu in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        print("--- failures ---", flush=True)
        for e in errors:
            print(e, flush=True)
        raise SystemExit(1)
    print("all jobs finished.", flush=True)


if __name__ == "__main__":
    main()
