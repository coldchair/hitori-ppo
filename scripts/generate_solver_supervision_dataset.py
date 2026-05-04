#!/usr/bin/env python3
"""
Generate (puzzle, solver shading) pairs using ``hitori_env``'s generator distribution
and the bundled ``hitori-solver`` smart CSP, with per-puzzle wall-clock limit and
high parallelism. Suitable for later supervised init of policy / value nets.

Uses the same puzzle distribution as ``HitoriEnv.reset`` (``generate_random_hitori_game``),
loaded via importlib from ``hitori_env/envs/hitori_generator.py`` so Gymnasium / pygame
are not required.

Example (repo root)::

    python scripts/generate_solver_supervision_dataset.py \\
        --out-dir runs/solver_supervision_mix4_20 \\
        --rng-base 100000 --workers 128 --time-limit 60

Resume: re-run the same command; existing ``size_*/item_*.npz`` are counted and only
missing samples are generated.

Scheduling: keep at most ``--workers`` tasks in flight. Whenever a worker finishes, the
main process **tops up** the queue. Among sizes that still need saved samples, the next
task picks the size with the **smallest current in-flight count** (tie: larger board
first), so slow large boards get parallel slots and small boards cannot monopolize the
128 workers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import random
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOLVER_SRC = REPO / "hitori-solver" / "source"
GENERATOR_PATH = REPO / "hitori_env" / "envs" / "hitori_generator.py"
STATE_NAME = "dataset_state.json"


def _load_generator():
    spec = importlib.util.spec_from_file_location("hitori_generator_standalone", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load spec for {GENERATOR_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_random_hitori_game


def _solve_smart_core(puzzle: list[list[int]]) -> tuple[int, list[list[Any]] | None]:
    """Mirror ``hitori-solver/source/hitori.py::solve_smart`` (no broken ``@timeit``)."""

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


def _solution_to_shaded(solution: list[list[Any]], size: int) -> np.ndarray:
    out = np.zeros((size, size), dtype=np.bool_)
    for i in range(size):
        for j in range(size):
            v = solution[i][j]
            out[i, j] = v == "B"
    return out


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
        shaded = None
        if ok:
            shaded = _solution_to_shaded(sol, size)

        res_q.put(
            {
                "size": size,
                "gen_seed": gen_seed,
                "ok": ok,
                "puzzle": np.asarray(grid, dtype=np.uint32),
                "shaded": shaded,
                "solve_nodes": int(nodes) if nodes is not None else -1,
            }
        )


def _size_dir(out_dir: Path, size: int) -> Path:
    return out_dir / f"size_{size:02d}"


def _count_saved(out_dir: Path, size: int) -> int:
    d = _size_dir(out_dir, size)
    if not d.is_dir():
        return 0
    return len(list(d.glob("item_*.npz")))


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


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "rng_base": None, "next_seed_cursor": 0, "attempts": {}}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=REPO / "runs/solver_supervision_mix4_20")
    p.add_argument("--min-size", type=int, default=4)
    p.add_argument("--max-size", type=int, default=20)
    p.add_argument("--per-size", type=int, default=1000, help="Max saved (solved) samples per board size.")
    p.add_argument("--workers", type=int, default=128, help="Parallel solver processes.")
    p.add_argument(
        "--time-limit",
        type=float,
        default=60.0,
        help="Wall-clock seconds per puzzle (SIGALRM / setitimer in each worker).",
    )
    p.add_argument(
        "--rng-base",
        type=int,
        default=100_000,
        help="Base offset for generator seeds (avoid colliding with common train seeds like 44).",
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

    if not hasattr(signal, "SIGALRM"):
        print("warning: SIGALRM unavailable; puzzles are solved without hard time limit", flush=True)

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / STATE_NAME
    state = _load_state(state_path)
    prev_base = state.get("rng_base")
    state["rng_base"] = args.rng_base
    if prev_base is not None and prev_base != args.rng_base:
        state["next_seed_cursor"] = 0
    state.setdefault("attempts", {})
    cursor = int(state.get("next_seed_cursor", 0))

    sizes = list(range(args.min_size, args.max_size + 1))
    need: dict[int, int] = {}
    for s in sizes:
        have = _count_saved(out_dir, s)
        need[s] = max(0, args.per_size - have)
        print(f"size {s:2d}: have {have:4d} / {args.per_size}  need {need[s]:4d}", flush=True)

    if all(v == 0 for v in need.values()):
        print("nothing to do (all quotas met).", flush=True)
        return

    ctx = mp.get_context("spawn")
    q_cap = max(256, args.workers * 4)
    task_q: mp.Queue = ctx.Queue(maxsize=q_cap)
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

    def next_gen_seed() -> int:
        nonlocal cursor
        s = int(args.rng_base + cursor)
        cursor += 1
        return s

    inflight_per_size: dict[int, int] = {s: 0 for s in sizes}

    def pick_size_for_dispatch() -> int | None:
        """Among sizes still under quota, pick one with fewest in-flight jobs (tie → larger n)."""

        candidates = [s for s in sizes if need[s] > 0]
        if not candidates:
            return None
        return min(candidates, key=lambda s: (inflight_per_size[s], -s))

    in_flight = 0
    manifest_path = out_dir / "manifest.jsonl"
    manifest_f = manifest_path.open("a", encoding="utf-8", buffering=1)

    def top_up_inflight() -> None:
        """While below worker cap and some size still needs data, dispatch one task (real-time)."""

        nonlocal in_flight
        while in_flight < args.workers:
            s = pick_size_for_dispatch()
            if s is None:
                break
            task_q.put({"size": s, "gen_seed": next_gen_seed()})
            in_flight += 1
            inflight_per_size[s] += 1

    top_up_inflight()

    t0 = time.perf_counter()
    total_ok = 0
    total_fail = 0

    try:
        while any(need[s] > 0 for s in sizes) or in_flight > 0:
            if in_flight == 0 and any(need[s] > 0 for s in sizes):
                top_up_inflight()
                if in_flight == 0:
                    break

            row = res_q.get()
            in_flight -= 1
            sz = int(row["size"])
            inflight_per_size[sz] -= 1

            if not row["ok"]:
                total_fail += 1
                state["attempts"][str(sz)] = int(state["attempts"].get(str(sz), 0)) + 1
                top_up_inflight()
                state["next_seed_cursor"] = cursor
                _atomic_write_json(state_path, state)
                continue

            idx = _count_saved(out_dir, sz)
            if idx >= args.per_size:
                need[sz] = 0
                top_up_inflight()
                state["next_seed_cursor"] = cursor
                _atomic_write_json(state_path, state)
                continue

            d = _size_dir(out_dir, sz)
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"item_{idx:06d}.npz"
            puzzle = row["puzzle"].astype(np.uint32, copy=False)
            shaded = row["shaded"].astype(np.bool_, copy=False)
            np.savez_compressed(
                path,
                game_grid=puzzle,
                shaded=shaded,
                gen_seed=np.int64(row["gen_seed"]),
                solve_nodes=np.int64(row["solve_nodes"]),
            )

            rec = {
                "size": sz,
                "item": idx,
                "path": str(path.relative_to(out_dir)),
                "gen_seed": int(row["gen_seed"]),
                "solve_nodes": int(row["solve_nodes"]),
            }
            manifest_f.write(json.dumps(rec, sort_keys=True) + "\n")
            manifest_f.flush()

            need[sz] = max(0, args.per_size - (idx + 1))
            total_ok += 1
            state["attempts"][str(sz)] = int(state["attempts"].get(str(sz), 0)) + 1

            if (total_ok + total_fail) % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"progress ok={total_ok} fail={total_fail} cursor={cursor} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            top_up_inflight()

            state["next_seed_cursor"] = cursor
            _atomic_write_json(state_path, state)
    finally:
        manifest_f.close()
        for _ in procs:
            task_q.put(None)
        for proc in procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    elapsed = time.perf_counter() - t0
    print(f"done. saved_ok={total_ok} timeouts_or_unsat={total_fail} elapsed={elapsed:.1f}s", flush=True)
    for s in sizes:
        print(f"  size {s:2d}: {_count_saved(out_dir, s)} files", flush=True)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
