"""Evaluation helpers for PPO training."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sb3_contrib import MaskablePPO

from ppo.env import ResizableEvalSlotEnv, make_env

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class EvalRow:
    size: int
    timesteps: int
    mean_reward: float
    solve_rate: float
    mean_length: float


def _stack_obs_dict(singles: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not singles:
        raise ValueError("empty observation batch")
    keys = singles[0].keys()
    return {k: np.stack([s[k] for s in singles]) for k in keys}


def _run_episode(
    model: MaskablePPO,
    env: Any,
    episode_seed: int,
    max_steps: int,
) -> tuple[float, int, bool]:
    """One full episode; matches prior evaluate_model semantics."""

    obs, _ = env.reset(seed=episode_seed)
    total_reward = 0.0

    for step in range(1, max_steps + 1):
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)

        if terminated or truncated:
            env_reward = float(info.get("env_reward", reward))
            return total_reward, step, bool(terminated and env_reward > 0)

    return total_reward, max_steps, False


def evaluate_model(
    model: MaskablePPO,
    size: int,
    max_size: int,
    episodes: int,
    seed: int,
    max_steps: int,
    reward_mode: str,
    dup_coef: float,
    adj_coef: float,
    comp_coef: float,
    shaped_reward_norm: str = "none",
    shaped_step_penalty: float = -0.05,
    shaped_dense_mult: float = 1.0,
) -> tuple[EvalRow, int]:
    rewards: list[float] = []
    lengths: list[int] = []
    solved: list[bool] = []

    env = make_env(
        size=size,
        max_size=max_size,
        reward_mode=reward_mode,
        dup_coef=dup_coef,
        adj_coef=adj_coef,
        comp_coef=comp_coef,
        shaped_reward_norm=shaped_reward_norm,
        shaped_step_penalty=shaped_step_penalty,
        shaped_dense_mult=shaped_dense_mult,
        random_pad_offset=False,
    )
    try:
        for episode in range(episodes):
            reward, length, is_solved = _run_episode(model, env, seed + episode, max_steps)
            rewards.append(reward)
            lengths.append(length)
            solved.append(is_solved)
    finally:
        env.close()

    total_env_steps = int(sum(lengths))
    return (
        EvalRow(
            size=size,
            timesteps=int(model.num_timesteps),
            mean_reward=float(np.mean(rewards)),
            solve_rate=float(np.mean(solved)),
            mean_length=float(np.mean(lengths)),
        ),
        total_env_steps,
    )


def _step_eval_slot(args: tuple[int, ResizableEvalSlotEnv, int]) -> tuple[int, dict[str, np.ndarray], float, bool, bool, dict]:
    slot, env, action = args
    obs, reward, term, trunc, info = env.step(int(action))
    return slot, obs, float(reward), bool(term), bool(trunc), info


def _reset_eval_slot(args: tuple[int, ResizableEvalSlotEnv, int, int]) -> tuple[int, dict[str, np.ndarray], int]:
    slot, env, size, seed = args
    obs, _ = env.reset(seed=None, options={"board_size": int(size), "episode_seed": int(seed)})
    return slot, obs, int(size)


def evaluate_all_sizes_vectorized(
    *,
    model: MaskablePPO,
    eval_sizes: list[int],
    max_size: int,
    tasks: list[tuple[int, int]],
    max_steps: int,
    timesteps: int,
    n_slots: int,
    reward_mode: str,
    dup_coef: float,
    adj_coef: float,
    comp_coef: float,
    shaped_reward_norm: str = "none",
    shaped_step_penalty: float = -0.05,
    shaped_dense_mult: float = 1.0,
) -> tuple[list[EvalRow], int]:
    """
    One ``model`` on the training device: each env step stacks obs → batched ``predict`` → parallel ``step``.

    ``tasks`` is ``(board_size, episode_seed)`` in evaluation order (same total count as serial eval).

    Returns ``(rows, total_env_steps)`` where ``total_env_steps`` counts every ``env.step`` during this eval.
    """

    if not tasks:
        return (
            [
                EvalRow(size=s, timesteps=timesteps, mean_reward=0.0, solve_rate=0.0, mean_length=0.0)
                for s in eval_sizes
            ],
            0,
        )

    n_slots = max(1, min(int(n_slots), len(tasks)))
    envs = [
        ResizableEvalSlotEnv(
            max_size=max_size,
            reward_mode=reward_mode,
            dup_coef=dup_coef,
            adj_coef=adj_coef,
            comp_coef=comp_coef,
            shaped_reward_norm=shaped_reward_norm,
            shaped_step_penalty=shaped_step_penalty,
            shaped_dense_mult=shaped_dense_mult,
        )
        for _ in range(n_slots)
    ]

    task_ptr = 0
    active = [False] * n_slots
    current_obs: list[dict[str, np.ndarray] | None] = [None] * n_slots
    cur_size = [0] * n_slots
    cur_return = [0.0] * n_slots
    cur_len = [0] * n_slots
    by_size: dict[int, list[tuple[float, int, bool]]] = defaultdict(list)
    eval_env_steps = 0

    def pull_next_task() -> tuple[int, int] | None:
        nonlocal task_ptr
        if task_ptr >= len(tasks):
            return None
        t = tasks[task_ptr]
        task_ptr += 1
        return t

    try:
        with ThreadPoolExecutor(max_workers=n_slots) as pool:
            initial: list[tuple[int, ResizableEvalSlotEnv, int, int]] = []
            for i in range(n_slots):
                nxt = pull_next_task()
                if nxt is None:
                    break
                size, seed = nxt
                initial.append((i, envs[i], size, seed))

            if not initial:
                return (
                    [
                        EvalRow(size=s, timesteps=timesteps, mean_reward=0.0, solve_rate=0.0, mean_length=0.0)
                        for s in eval_sizes
                    ],
                    0,
                )

            reset_out = list(pool.map(_reset_eval_slot, initial))
            for (slot, obs, size), _ in zip(reset_out, initial, strict=True):
                current_obs[slot] = obs
                active[slot] = True
                cur_size[slot] = size
                cur_return[slot] = 0.0
                cur_len[slot] = 0

            while any(active):
                active_ix = [i for i in range(n_slots) if active[i]]
                obs_b = _stack_obs_dict([current_obs[i] for i in active_ix])
                masks = np.stack([envs[i].action_masks() for i in active_ix])
                actions, _ = model.predict(obs_b, action_masks=masks, deterministic=True)
                actions_flat = np.asarray(actions).reshape(-1)
                step_args = [(ix, envs[ix], int(actions_flat[j])) for j, ix in enumerate(active_ix)]
                eval_env_steps += len(step_args)

                need_reset: list[tuple[int, ResizableEvalSlotEnv, int, int]] = []
                for slot, obs, reward, term, trunc, info in pool.map(_step_eval_slot, step_args):
                    cur_return[slot] += reward
                    cur_len[slot] += 1
                    current_obs[slot] = obs

                    ended = bool(term or trunc)
                    if ended:
                        env_reward = float(info.get("env_reward", reward))
                        solved = bool(term and env_reward > 0)
                        by_size[cur_size[slot]].append((cur_return[slot], cur_len[slot], solved))
                        nxt = pull_next_task()
                        if nxt is None:
                            active[slot] = False
                            current_obs[slot] = None
                        else:
                            need_reset.append((slot, envs[slot], nxt[0], nxt[1]))
                    elif cur_len[slot] >= max_steps:
                        by_size[cur_size[slot]].append((cur_return[slot], max_steps, False))
                        nxt = pull_next_task()
                        if nxt is None:
                            active[slot] = False
                            current_obs[slot] = None
                        else:
                            need_reset.append((slot, envs[slot], nxt[0], nxt[1]))

                if need_reset:
                    reset_batch = list(pool.map(_reset_eval_slot, need_reset))
                    for (slot, obs, size), _ in zip(reset_batch, need_reset, strict=True):
                        current_obs[slot] = obs
                        cur_size[slot] = size
                        cur_return[slot] = 0.0
                        cur_len[slot] = 0
    finally:
        for env in envs:
            env.close()

    eval_rows: list[EvalRow] = []
    for size in eval_sizes:
        rows = by_size.get(size, [])
        if not rows:
            eval_rows.append(
                EvalRow(
                    size=size,
                    timesteps=timesteps,
                    mean_reward=0.0,
                    solve_rate=0.0,
                    mean_length=0.0,
                )
            )
            continue
        rewards, lengths, solved_flags = zip(*rows, strict=True)
        eval_rows.append(
            EvalRow(
                size=size,
                timesteps=timesteps,
                mean_reward=float(np.mean(rewards)),
                solve_rate=float(np.mean(solved_flags)),
                mean_length=float(np.mean(lengths)),
            )
        )
    return eval_rows, eval_env_steps


def evaluate_all_sizes(
    *,
    model: MaskablePPO,
    eval_sizes: list[int],
    max_size: int,
    episodes: int,
    eval_seed: int,
    max_steps: int,
    reward_mode: str,
    dup_coef: float,
    adj_coef: float,
    comp_coef: float,
    shaped_reward_norm: str = "none",
    shaped_step_penalty: float = -0.05,
    shaped_dense_mult: float = 1.0,
    timesteps: int,
    vec_envs: int,
) -> tuple[list[EvalRow], int]:
    """Evaluate every board size; serial if ``vec_envs`` <= 1 else batched like training.

    **Deterministic puzzle suite:** for each ``size`` at index ``k`` in ``eval_sizes`` and each
    ``episode_index`` in ``0 .. episodes-1``, the env is reset with
    ``options['episode_seed'] = eval_seed + k * 1000 + episode_index``. The same triple
    ``(eval_seed, k, episode_index)`` always yields the same initial board (for fixed env code).

    Returns ``(eval_rows, total_env_steps)`` where ``total_env_steps`` is the number of ``env.step`` calls.
    """

    tasks: list[tuple[int, int]] = []
    for offset, size in enumerate(eval_sizes):
        base = eval_seed + offset * 1_000
        for ep in range(episodes):
            tasks.append((size, base + ep))

    if vec_envs <= 1:
        rows: list[EvalRow] = []
        total_steps = 0
        for offset, size in enumerate(eval_sizes):
            row, n_steps = evaluate_model(
                model=model,
                size=size,
                max_size=max_size,
                episodes=episodes,
                seed=eval_seed + offset * 1_000,
                max_steps=max_steps,
                reward_mode=reward_mode,
                dup_coef=dup_coef,
                adj_coef=adj_coef,
                comp_coef=comp_coef,
                shaped_reward_norm=shaped_reward_norm,
                shaped_step_penalty=shaped_step_penalty,
                shaped_dense_mult=shaped_dense_mult,
            )
            rows.append(row)
            total_steps += n_steps
        return rows, total_steps

    return evaluate_all_sizes_vectorized(
        model=model,
        eval_sizes=eval_sizes,
        max_size=max_size,
        tasks=tasks,
        max_steps=max_steps,
        timesteps=timesteps,
        n_slots=vec_envs,
        reward_mode=reward_mode,
        dup_coef=dup_coef,
        adj_coef=adj_coef,
        comp_coef=comp_coef,
        shaped_reward_norm=shaped_reward_norm,
        shaped_step_penalty=shaped_step_penalty,
        shaped_dense_mult=shaped_dense_mult,
    )


def resolve_eval_vec_envs(n_envs: int, explicit: int | None, total_tasks: int) -> int:
    """Parallel eval slots; default matches ``--n-envs``, capped by total eval episodes."""

    cap = max(1, int(total_tasks))
    if explicit is not None:
        return max(1, min(int(explicit), cap))
    return max(1, min(int(n_envs), cap))


def plot_learning_curves_per_size(csv_path: Path, output_dir: Path) -> None:
    """Write ``learning_curve_size{N}.png`` for each board size (reward + solve_rate twin axes).

    Reads the cumulative ``learning_curve.csv`` so plots update as new eval rows are appended.
    """

    if not csv_path.is_file():
        return
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return

    if data.shape == ():
        data = np.array([data], dtype=data.dtype)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = np.unique(data["size"])
    for size in sizes:
        size_data = data[data["size"] == size]
        fig, ax_reward = plt.subplots(figsize=(7.5, 4.25))
        ax_reward.plot(
            size_data["timesteps"],
            size_data["mean_reward"],
            marker="o",
            color="tab:blue",
            linewidth=1.2,
            markersize=4,
            label="mean_reward",
        )
        ax_reward.set_xlabel("Timesteps")
        ax_reward.set_ylabel("Mean episode reward")
        ax_reward.grid(True, alpha=0.25)
        ax_reward.set_title(f"{int(size)}×{int(size)}")

        ax_solve = ax_reward.twinx()
        ax_solve.plot(
            size_data["timesteps"],
            size_data["solve_rate"],
            marker="s",
            color="tab:green",
            linewidth=1.2,
            markersize=4,
            label="solve_rate",
        )
        ax_solve.set_ylabel("Solve rate")
        ax_solve.set_ylim(-0.02, 1.02)

        lines, labels = ax_reward.get_legend_handles_labels()
        lines2, labels2 = ax_solve.get_legend_handles_labels()
        ax_reward.legend(lines + lines2, labels + labels2, loc="lower right")
        fig.tight_layout()
        out_png = output_dir / f"learning_curve_size{int(size)}.png"
        fig.savefig(out_png, dpi=160)
        plt.close(fig)


def plot_learning_curve(csv_path: Path, output_path: Path) -> None:
    """Backward-compatible: write per-size PNGs next to ``output_path``'s parent."""

    plot_learning_curves_per_size(csv_path, output_path.parent)
