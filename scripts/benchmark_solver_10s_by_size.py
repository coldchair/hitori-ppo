#!/usr/bin/env python3
"""
Benchmark bundled ``hitori-solver`` (smart CSP): for each board side length,
run a fixed number of random puzzles (same distribution as ``HitoriEnv.reset``)
and record whether a solution is found within a wall-clock limit.

Uses the same generator import path and ``_solve_smart_core`` pattern as
``generate_solver_supervision_dataset.py`` (SIGALRM / setitimer per puzzle on Linux).

Example (repo root)::

    python scripts/benchmark_solver_10s_by_size.py \\
        --min-size 4 --max-size 25 --per-size 100 --workers 128 --time-limit 10

Writes ``summary.json`` under ``--out-dir`` by default; prints a Markdown table to stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SOLVER_SRC = REPO / "hitori-solver" / "source"
GENERATOR_PATH = REPO / "hitori_env" / "envs" / "hitori_generator.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("hitori_generator_standalone", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load spec for {GENERATOR_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_random_hitori_game


def _solve_smart_core(puzzle: list[list[int]]) -> tuple[int, list[list[Any]] | None]:
    """Mirror ``hitori-solver/source/hitori.py::solve_smart`` (same as generate_solver_supervision_dataset)."""

    from hitori_rules import (  # noqa: PLC0415
        black_allowed,
        cell_surrounded,
        check_solution,
        copy as hitori_copy_state,
        fc_black,
        fc_white,
        load_puzzle,
        test_white_connected,
    )

    counter = 0
    nodes = [[0, 0]]
    states = [load_puzzle(puzzle)]

    while nodes:
        i, j = nodes.pop()
        state = states.pop()

        if check_solution(state):
            return counter, state["puzzle"]

        if i >= len(state["puzzle"]):
            continue

        state = cell_surrounded(state, i, j)
        if not state:
            continue

        for d in state["domain"][i][j]:
            if d == "V":
                states.append(state)
                nodes.append([i, j + 1] if j + 1 < len(state["puzzle"][i]) else [i + 1, 0])
                continue

            if d == "W":
                white_state = hitori_copy_state(state)
                white_state["domain"][i][j] = "W"
                counter += 1
                new_domain_white = fc_white(white_state, i, j)
                if not new_domain_white:
                    continue
                states.append(white_state)
                nodes.append([i, j + 1] if j + 1 < len(state["puzzle"][i]) else [i + 1, 0])

            if d == "B" and black_allowed(state, i, j):
                black_state = hitori_copy_state(state)
                black_state["puzzle"][i][j] = "B"
                black_state["domain"][i][j] = "B"
                counter += 1
                if not test_white_connected(black_state["domain"], black_state["edges"]):
                    continue
                black_state = fc_black(black_state, i, j)
                if not black_state:
                    continue
                states.append(black_state)
                nodes.append([i, j + 1] if j + 1 < len(state["puzzle"][i]) else [i + 1, 0])
    return counter, None


class _Timeout(Exception):
    pass


def _alarm_handler(_signum: int, _frame: Any) -> None:
    raise _Timeout()


def _worker_loop(
    task_q: mp.Queue,
    res_q: mp.Queue,
    repo_root: str,
    solver_src: str,
    time_limit_s: float,
) -> None:
    sys.path.insert(0, solver_src)
    sys.path.insert(0, repo_root)

    generate_random_hitori_game = _load_generator()

    while True:
        task = task_q.get()
        if task is None:
            break
        size: int = task["size"]
        gen_seed: int = task["gen_seed"]

        grid = generate_random_hitori_game(size, seed=gen_seed)
        puzzle = grid.astype(int).tolist()

        nodes = -1
        sol = None
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, time_limit_s, 0.0)
            try:
                nodes, sol = _solve_smart_core(puzzle)
            except _Timeout:
                nodes, sol = -1, None
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
        else:
            nodes, sol = _solve_smart_core(puzzle)

        ok = sol is not None
        res_q.put(
            {
                "size": size,
                "gen_seed": gen_seed,
                "ok": bool(ok),
                "solve_nodes": int(nodes) if nodes is not None and nodes >= 0 else -1,
            }
        )


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _print_markdown_table(rows: list[dict[str, Any]]) -> None:
    print("", flush=True)
    print("| size | trials | solved ≤ limit | rate |", flush=True)
    print("|-----:|-------:|---------------:|-----:|", flush=True)
    for r in rows:
        print(
            f"| {int(r['size'])} | {int(r['trials'])} | {int(r['solved'])} | {float(r['solve_rate']):.4f} |",
            flush=True,
        )
    tot_t = sum(int(r["trials"]) for r in rows)
    tot_s = sum(int(r["solved"]) for r in rows)
    print(
        f"| **all** | {tot_t} | {tot_s} | {tot_s / max(tot_t, 1):.4f} |",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-size", type=int, default=4)
    p.add_argument("--max-size", type=int, default=25)
    p.add_argument("--per-size", type=int, default=100, help="Number of random puzzles per board size.")
    p.add_argument("--workers", type=int, default=128, help="Parallel solver processes.")
    p.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Wall-clock seconds per puzzle (SIGALRM / setitimer in each worker).",
    )
    p.add_argument(
        "--rng-base",
        type=int,
        default=900_000,
        help="Base for generator seeds; each task uses rng_base + sequential index.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "runs/solver_10s_benchmark",
        help="Directory for summary.json (and optional copy of stdout table).",
    )
    p.add_argument(
        "--summary-name",
        type=str,
        default="summary.json",
        help="Written under --out-dir.",
    )
    args = p.parse_args()

    if not SOLVER_SRC.is_dir():
        raise SystemExit(f"Solver source not found: {SOLVER_SRC}")
    if not GENERATOR_PATH.is_file():
        raise SystemExit(f"Generator not found: {GENERATOR_PATH}")
    if args.min_size < 1 or args.max_size < args.min_size:
        raise SystemExit("invalid size range")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.per_size < 1:
        raise SystemExit("--per-size must be >= 1")

    if not hasattr(signal, "SIGALRM"):
        print("warning: SIGALRM unavailable; puzzles are solved without hard time limit", flush=True)

    sizes = list(range(args.min_size, args.max_size + 1))
    tasks: list[dict[str, int]] = []
    idx = 0
    for size in sizes:
        for _trial in range(args.per_size):
            tasks.append({"size": size, "gen_seed": int(args.rng_base + idx)})
            idx += 1

    n_tasks = len(tasks)
    print(
        f"benchmark: sizes {args.min_size}-{args.max_size}, {args.per_size} trials each, "
        f"n_tasks={n_tasks}, workers={args.workers}, time_limit={args.time_limit}s",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    task_q: mp.Queue = ctx.Queue(maxsize=max(512, n_tasks + args.workers))
    res_q: mp.Queue = ctx.Queue()
    procs: list[mp.Process] = []
    for _ in range(args.workers):
        proc = ctx.Process(
            target=_worker_loop,
            args=(task_q, res_q, str(REPO), str(SOLVER_SRC), float(args.time_limit)),
            daemon=True,
        )
        proc.start()
        procs.append(proc)

    t0 = time.perf_counter()
    try:
        for t in tasks:
            task_q.put(t)
        for _ in range(args.workers):
            task_q.put(None)

        raw: list[dict[str, Any]] = []
        for _ in range(n_tasks):
            raw.append(res_q.get())
    finally:
        for proc in procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()

    elapsed = time.perf_counter() - t0

    solved_by_size: dict[int, int] = {s: 0 for s in sizes}
    trials_by_size: dict[int, int] = {s: 0 for s in sizes}
    for row in raw:
        sz = int(row["size"])
        trials_by_size[sz] = trials_by_size.get(sz, 0) + 1
        if row["ok"]:
            solved_by_size[sz] = solved_by_size.get(sz, 0) + 1

    rows: list[dict[str, Any]] = []
    for s in sizes:
        t_n = trials_by_size[s]
        s_n = solved_by_size[s]
        rows.append(
            {
                "size": s,
                "trials": t_n,
                "solved": s_n,
                "failed": t_n - s_n,
                "solve_rate": (s_n / t_n) if t_n else 0.0,
            }
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "min_size": args.min_size,
        "max_size": args.max_size,
        "per_size": args.per_size,
        "workers": args.workers,
        "time_limit_s": args.time_limit,
        "rng_base": args.rng_base,
        "wall_clock_s": elapsed,
        "n_tasks": n_tasks,
        "by_size": rows,
    }
    summary_path = out_dir / args.summary_name
    _atomic_write_json(summary_path, summary)
    print(f"wrote {summary_path}", flush=True)
    print(f"wall_clock={elapsed:.1f}s", flush=True)
    _print_markdown_table(rows)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
