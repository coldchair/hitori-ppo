#!/usr/bin/env python3
"""
Parse all ``final_eval.txt`` under an ablation run root (e.g. ``runs/maskable_ppo_unet_mix4_to_11_2M_ablation``).

Expects job folder names like:
  ``att{0,1,2}_ns{n}_bs{b}_{pad|nopad}_{sparse|shaped}_seed44``

Writes a CSV and prints grouped means (by attention blocks, step×batch, pad, reward-mode).
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


JOB_TAG_RE = re.compile(
    r"att(?P<attn>\d+)_ns(?P<ns>\d+)_bs(?P<bs>\d+)_(?P<pad>pad|nopad)_(?P<rm>sparse|shaped)_seed\d+"
)

SIZE_HEADER_RE = re.compile(r"^\[(\d+)x(\d+)\]\s*$")
KV_RE = re.compile(r"^(\w+)=(.+)$")


def parse_final_eval(path: Path) -> dict[int, dict[str, float]]:
    """Per board size -> metrics (solve_rate, mean_reward, mean_length, timesteps)."""

    by_size: dict[int, dict[str, float]] = {}
    cur: int | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        m = SIZE_HEADER_RE.match(line)
        if m:
            cur = int(m.group(1))
            by_size[cur] = {}
            continue
        if not line or cur is None:
            continue
        km = KV_RE.match(line)
        if not km:
            continue
        key, val_s = km.group(1), km.group(2).strip()
        if key == "timesteps":
            by_size[cur][key] = float(int(val_s))
        else:
            by_size[cur][key] = float(val_s)
    return by_size


def job_params_from_relpath(rel: str) -> dict[str, str | int] | None:
    """Extract ablation fields from ``.../att0_ns32_bs256_nopad_sparse_seed44/...``."""

    for part in Path(rel).parts:
        m = JOB_TAG_RE.fullmatch(part)
        if m:
            return {
                "attn": int(m.group("attn")),
                "n_steps": int(m.group("ns")),
                "batch_size": int(m.group("bs")),
                "random_pad": m.group("pad") == "pad",
                "reward_mode": m.group("rm"),
            }
    return None


def summarize_sizes(by_size: dict[int, dict[str, float]]) -> dict[str, float]:
    sizes = sorted(by_size)
    solves = [by_size[s]["solve_rate"] for s in sizes if "solve_rate" in by_size[s]]
    rewards = [by_size[s]["mean_reward"] for s in sizes if "mean_reward" in by_size[s]]
    lengths = [by_size[s]["mean_length"] for s in sizes if "mean_length" in by_size[s]]
    out: dict[str, float] = {
        "n_sizes": float(len(sizes)),
        "mean_solve_all_sizes": float(statistics.mean(solves)) if solves else float("nan"),
        "min_solve": float(min(solves)) if solves else float("nan"),
        "max_solve": float(max(solves)) if solves else float("nan"),
        "mean_mean_reward": float(statistics.mean(rewards)) if rewards else float("nan"),
        "mean_mean_length": float(statistics.mean(lengths)) if lengths else float("nan"),
    }
    for s in sizes:
        prefix = f"size{s}"
        if "solve_rate" in by_size[s]:
            out[f"{prefix}_solve"] = by_size[s]["solve_rate"]
        if "mean_reward" in by_size[s]:
            out[f"{prefix}_reward"] = by_size[s]["mean_reward"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("runs/maskable_ppo_unet_mix4_to_11_2M_ablation"),
        help="Ablation root containing att*_ns*_bs*_... job directories.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write one row per run to this CSV (default: print path only, still print tables).",
    )
    args = p.parse_args()
    root: Path = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"root not found or not a directory: {root}")

    paths = sorted(root.glob("**/final_eval.txt"))
    if not paths:
        raise SystemExit(f"no final_eval.txt under {root}")

    rows: list[dict[str, str | int | float]] = []
    for fp in paths:
        rel = str(fp.relative_to(root))
        jp = job_params_from_relpath(rel)
        if jp is None:
            print(f"[skip] cannot parse job tag from: {rel}", flush=True)
            continue
        try:
            by_size = parse_final_eval(fp)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] parse error {fp}: {e}", flush=True)
            continue
        if not by_size:
            print(f"[skip] empty metrics: {rel}", flush=True)
            continue
        summ = summarize_sizes(by_size)
        row: dict[str, str | int | float] = {
            "relpath": rel,
            **jp,
            "step_batch": f"{jp['n_steps']}_{jp['batch_size']}",
            **summ,
        }
        rows.append(row)

    if not rows:
        raise SystemExit("no valid runs parsed")

    # Stable column order for CSV
    base_cols = [
        "relpath",
        "attn",
        "n_steps",
        "batch_size",
        "step_batch",
        "random_pad",
        "reward_mode",
        "n_sizes",
        "mean_solve_all_sizes",
        "min_solve",
        "max_solve",
        "mean_mean_reward",
        "mean_mean_length",
    ]
    size_cols = [c for c in sorted(rows[0]) if c.startswith("size") and c.endswith("_solve")]
    reward_cols = [c for c in sorted(rows[0]) if c.startswith("size") and c.endswith("_reward")]
    fieldnames = base_cols + size_cols + reward_cols

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in sorted(rows, key=lambda r: (-float(r["mean_solve_all_sizes"]), str(r["relpath"]))):
                w.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"wrote {len(rows)} rows -> {args.csv}", flush=True)

    def grp_mean(key: str) -> list[tuple[str, float, int]]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            k = str(row[key])
            buckets[k].append(float(row["mean_solve_all_sizes"]))
        out = [(k, statistics.mean(v), len(v)) for k, v in sorted(buckets.items())]
        return sorted(out, key=lambda t: -t[1])

    print(f"\n=== Parsed {len(rows)} runs under {root} ===\n", flush=True)
    print("Top 15 by mean_solve_all_sizes (equal weight per eval size):\n", flush=True)
    top = sorted(rows, key=lambda r: -float(r["mean_solve_all_sizes"]))[:15]
    for i, r in enumerate(top, 1):
        print(
            f"{i:2d}  att={r['attn']}  ns={r['n_steps']:3d}  bs={r['batch_size']:4d}  "
            f"pad={r['random_pad']!s:5}  {r['reward_mode']:7s}  "
            f"mean_solve={r['mean_solve_all_sizes']:.4f}  min={r['min_solve']:.3f}  "
            f"11x11={r.get('size11_solve', float('nan')):.3f}  "
            f"path={r['relpath']}",
            flush=True,
        )

    for label, key in [
        ("num_attention_blocks (--num-attention-blocks)", "attn"),
        ("n_steps × batch_size (step_batch)", "step_batch"),
        ("train_random_pad_offset", "random_pad"),
        ("reward_mode", "reward_mode"),
    ]:
        print(f"\n--- Mean solve (all sizes) by {label} ---\n", flush=True)
        for k, mean_v, cnt in grp_mean(key):
            print(f"  {k!s:16}  n={cnt:2d}  mean_solve={mean_v:.4f}", flush=True)

    # Best sparse vs shaped at same other settings? Quick diff not automated — user can use CSV.

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
