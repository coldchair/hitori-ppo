"""Environment wrappers and builders for PPO training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import hitori_env  # noqa: F401 - registers hitori_env/Hitori-v2
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from ppo.sampling import CurriculumSizeSampler, WeightedSizeSampler


class ActionMaskWrapper(gym.Wrapper):
    """Expose action_masks through wrappers so MaskablePPO can find it."""

    def action_masks(self) -> np.ndarray:
        return self.env.unwrapped.action_masks()


@dataclass(frozen=True)
class ConstraintStats:
    duplicates: int
    adjacent: int
    disconnected: int


def count_unshaded_duplicates(game_grid: np.ndarray, shaded: np.ndarray) -> int:
    duplicate_count = 0
    size = int(game_grid.shape[0])

    for row in range(size):
        values = game_grid[row, ~shaded[row]]
        _, counts = np.unique(values, return_counts=True)
        duplicate_count += int(np.sum(np.maximum(counts - 1, 0)))

    for col in range(size):
        values = game_grid[~shaded[:, col], col]
        _, counts = np.unique(values, return_counts=True)
        duplicate_count += int(np.sum(np.maximum(counts - 1, 0)))

    return duplicate_count


def count_adjacent_shaded_conflicts(shaded: np.ndarray) -> int:
    horizontal = np.logical_and(shaded[:, :-1], shaded[:, 1:]).sum()
    vertical = np.logical_and(shaded[:-1, :], shaded[1:, :]).sum()
    return int(horizontal + vertical)


def count_unshaded_disconnected_components(shaded: np.ndarray) -> int:
    size = int(shaded.shape[0])
    unshaded = ~shaded
    total_unshaded = int(unshaded.sum())
    if total_unshaded == 0:
        return 0

    visited = np.zeros_like(unshaded, dtype=bool)
    components = 0

    for start_row, start_col in np.argwhere(unshaded):
        start = (int(start_row), int(start_col))
        if visited[start]:
            continue

        components += 1
        queue: deque[tuple[int, int]] = deque([start])
        visited[start] = True

        while queue:
            row, col = queue.popleft()
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < size and 0 <= nc < size and unshaded[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    return max(components - 1, 0)


def constraint_stats(obs: dict[str, np.ndarray]) -> ConstraintStats:
    game_grid = np.asarray(obs["game_grid"])
    shaded = np.asarray(obs["shaded"], dtype=bool)
    return ConstraintStats(
        duplicates=count_unshaded_duplicates(game_grid, shaded),
        adjacent=count_adjacent_shaded_conflicts(shaded),
        disconnected=count_unshaded_disconnected_components(shaded),
    )


class ConstraintRewardWrapper(gym.Wrapper):
    """Reward shaping based on progress toward the Hitori constraints.

    With ``shaped_reward_norm == "inv_board_area"``, only the dense part (step penalty + constraint deltas)
    is divided by ``board_size**2``; then multiplied by ``shaped_dense_mult``; terminal bonuses are not scaled.
    """

    def __init__(
        self,
        env: gym.Env,
        dup_coef: float = 1.0,
        adj_coef: float = 0.5,
        comp_coef: float = 1.5,
        success_bonus: float = 10.0,
        failure_penalty: float = -10.0,
        step_penalty: float = -0.05,
        *,
        shaped_reward_norm: str = "none",
        shaped_dense_mult: float = 1.0,
    ):
        super().__init__(env)
        self.dup_coef = dup_coef
        self.adj_coef = adj_coef
        self.comp_coef = comp_coef
        self.success_bonus = success_bonus
        self.failure_penalty = failure_penalty
        self.step_penalty = step_penalty
        if shaped_reward_norm not in ("none", "inv_board_area"):
            raise ValueError(f"shaped_reward_norm must be 'none' or 'inv_board_area', got {shaped_reward_norm!r}")
        self.shaped_reward_norm = shaped_reward_norm
        self.shaped_dense_mult = float(shaped_dense_mult)
        self._last_stats: ConstraintStats | None = None
        self._board_size: int | None = None

    def _scale_dense(self, dense: float) -> float:
        if self.shaped_reward_norm == "none":
            return float(dense)
        assert self.shaped_reward_norm == "inv_board_area"
        if self._board_size is None:
            return float(dense)
        area = float(max(1, self._board_size) ** 2)
        return float(dense) / area

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._board_size = int(self.env.unwrapped.size)
        self._last_stats = constraint_stats(obs)
        return obs, info

    def step(self, action: int):
        before = self._last_stats
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        after = constraint_stats(obs)
        self._last_stats = after

        if before is None:
            before = after

        dense = self.step_penalty
        dense += self.dup_coef * (before.duplicates - after.duplicates)
        dense += self.adj_coef * (before.adjacent - after.adjacent)
        dense += self.comp_coef * (before.disconnected - after.disconnected)
        dense = self._scale_dense(dense) * self.shaped_dense_mult

        solved = bool(terminated and env_reward > 0)
        failed = bool((terminated and env_reward <= 0) or truncated)
        terminal = 0.0
        if solved:
            terminal = self.success_bonus
        elif failed:
            terminal = self.failure_penalty

        reward = dense + terminal

        info = dict(info)
        info["env_reward"] = float(env_reward)
        info["constraint_stats"] = {
            "duplicates": after.duplicates,
            "adjacent": after.adjacent,
            "disconnected": after.disconnected,
        }
        return obs, float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


class PadToMaxBoardSizeWrapper(gym.Wrapper):
    """Pad observations and action masks to a fixed maximum board size."""

    def __init__(self, env: gym.Env, max_size: int, *, random_pad_offset: bool = False):
        super().__init__(env)
        self.board_size = int(env.unwrapped.size)
        self.max_size = max_size
        self.random_pad_offset = bool(random_pad_offset)
        self._pad_dr = 0
        self._pad_dc = 0
        if self.max_size < self.board_size:
            raise ValueError(f"max_size={max_size} must be >= board_size={self.board_size}")

        self.observation_space = spaces.Dict(
            {
                "game_grid": spaces.Box(
                    low=0,
                    high=self.max_size,
                    shape=(self.max_size, self.max_size),
                    dtype=np.uint32,
                ),
                "shaded": spaces.MultiBinary((self.max_size, self.max_size)),
            }
        )
        self.action_space = spaces.Discrete(self.max_size * self.max_size)

    def _sample_pad_offsets(self) -> None:
        margin = self.max_size - self.board_size
        if not self.random_pad_offset or margin <= 0:
            self._pad_dr = 0
            self._pad_dc = 0
            return
        rng = self.np_random
        self._pad_dr = int(rng.integers(0, margin + 1))
        self._pad_dc = int(rng.integers(0, margin + 1))

    def _pad_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        game_grid = np.zeros((self.max_size, self.max_size), dtype=np.uint32)
        shaded = np.zeros((self.max_size, self.max_size), dtype=bool)
        n = self.board_size
        dr, dc = self._pad_dr, self._pad_dc
        game_grid[dr : dr + n, dc : dc + n] = obs["game_grid"]
        shaded[dr : dr + n, dc : dc + n] = obs["shaded"]
        return {
            "game_grid": game_grid,
            "shaded": shaded,
        }

    def _with_board_info(self, info: dict) -> dict:
        info = dict(info)
        info["board_size"] = self.board_size
        info["max_size"] = self.max_size
        info["pad_dr"] = int(self._pad_dr)
        info["pad_dc"] = int(self._pad_dc)
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._sample_pad_offsets()
        return self._pad_obs(obs), self._with_board_info(info)

    def step(self, action: int):
        row, col = divmod(int(action), self.max_size)
        dr, dc = self._pad_dr, self._pad_dc
        n = self.board_size
        if row < dr or row >= dr + n or col < dc or col >= dc + n:
            raise ValueError(
                f"received padded action {(row, col)} outside active board "
                f"region dr={dr}, dc={dc}, board_size={n}; this should be prevented by the action mask"
            )

        inner_row, inner_col = row - dr, col - dc
        mapped_action = inner_row * n + inner_col
        obs, reward, terminated, truncated, info = self.env.step(mapped_action)
        return self._pad_obs(obs), reward, terminated, truncated, self._with_board_info(info)

    def action_masks(self) -> np.ndarray:
        base_mask = np.asarray(self.env.action_masks(), dtype=np.int8).reshape(self.board_size, self.board_size)
        padded_mask = np.zeros((self.max_size, self.max_size), dtype=np.int8)
        dr, dc = self._pad_dr, self._pad_dc
        n = self.board_size
        padded_mask[dr : dr + n, dc : dc + n] = base_mask
        return padded_mask.reshape(-1)


class ResizableEvalSlotEnv(gym.Env):
    """One VecEval slot: ``reset(..., options=)`` picks board size (recreates inner env when size changes)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_size: int,
        reward_mode: str,
        dup_coef: float,
        adj_coef: float,
        comp_coef: float,
        shaped_reward_norm: str = "none",
        shaped_step_penalty: float = -0.05,
        shaped_dense_mult: float = 1.0,
    ):
        super().__init__()
        self.max_size = int(max_size)
        self.reward_mode = reward_mode
        self.dup_coef = dup_coef
        self.adj_coef = adj_coef
        self.comp_coef = comp_coef
        self.shaped_reward_norm = shaped_reward_norm
        self.shaped_step_penalty = shaped_step_penalty
        self.shaped_dense_mult = shaped_dense_mult
        self._inner: gym.Env | None = None
        self._board_size: int | None = None

        template = make_env(
            4,
            self.max_size,
            reward_mode=reward_mode,
            dup_coef=dup_coef,
            adj_coef=adj_coef,
            comp_coef=comp_coef,
            shaped_reward_norm=shaped_reward_norm,
            shaped_step_penalty=shaped_step_penalty,
            shaped_dense_mult=shaped_dense_mult,
            random_pad_offset=False,
        )
        self.observation_space = template.observation_space
        self.action_space = template.action_space
        template.close()

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # noqa: ARG002
        if options is None or "board_size" not in options or "episode_seed" not in options:
            raise ValueError("ResizableEvalSlotEnv.reset requires options['board_size'] and options['episode_seed']")
        size = int(options["board_size"])
        ep_seed = int(options["episode_seed"])
        if self._inner is None or self._board_size != size:
            if self._inner is not None:
                self._inner.close()
            self._inner = make_env(
                size=size,
                max_size=self.max_size,
                reward_mode=self.reward_mode,
                dup_coef=self.dup_coef,
                adj_coef=self.adj_coef,
                comp_coef=self.comp_coef,
                shaped_reward_norm=self.shaped_reward_norm,
                shaped_step_penalty=self.shaped_step_penalty,
                shaped_dense_mult=self.shaped_dense_mult,
                random_pad_offset=False,
            )
            self._board_size = size
        assert self._inner is not None
        return self._inner.reset(seed=ep_seed)

    def step(self, action):
        assert self._inner is not None
        return self._inner.step(action)

    def action_masks(self) -> np.ndarray:
        assert self._inner is not None
        return self._inner.action_masks()

    def close(self):
        if self._inner is not None:
            self._inner.close()
            self._inner = None
            self._board_size = None
        super().close()


def make_resizable_eval_slot_factory(
    *,
    max_size: int,
    reward_mode: str,
    dup_coef: float,
    adj_coef: float,
    comp_coef: float,
    shaped_reward_norm: str = "none",
    shaped_step_penalty: float = -0.05,
    shaped_dense_mult: float = 1.0,
):
    def _factory() -> ResizableEvalSlotEnv:
        return ResizableEvalSlotEnv(
            max_size=max_size,
            reward_mode=reward_mode,
            dup_coef=dup_coef,
            adj_coef=adj_coef,
            comp_coef=comp_coef,
            shaped_reward_norm=shaped_reward_norm,
            shaped_step_penalty=shaped_step_penalty,
            shaped_dense_mult=shaped_dense_mult,
        )

    return _factory


def make_env(
    size: int,
    max_size: int | None = None,
    seed: int | None = None,
    reward_mode: str = "sparse",
    dup_coef: float = 1.0,
    adj_coef: float = 0.5,
    comp_coef: float = 1.5,
    *,
    shaped_reward_norm: str = "none",
    shaped_step_penalty: float = -0.05,
    shaped_dense_mult: float = 1.0,
    random_pad_offset: bool = False,
) -> gym.Env:
    if max_size is None:
        max_size = size

    env = gym.make("hitori_env/Hitori-v2", size=size)
    env = ActionMaskWrapper(env)
    if reward_mode == "shaped":
        env = ConstraintRewardWrapper(
            env,
            dup_coef=dup_coef,
            adj_coef=adj_coef,
            comp_coef=comp_coef,
            step_penalty=shaped_step_penalty,
            shaped_reward_norm=shaped_reward_norm,
            shaped_dense_mult=shaped_dense_mult,
        )
    env = PadToMaxBoardSizeWrapper(env, max_size=max_size, random_pad_offset=random_pad_offset)
    if seed is not None:
        env.reset(seed=seed)
    return env


class MultiSizeHitoriEnv(gym.Env):
    """Episode-wise board-size sampler over a fixed padded observation/action space."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        sizes: list[int],
        max_size: int,
        reward_mode: str,
        dup_coef: float,
        adj_coef: float,
        comp_coef: float,
        size_sampler: WeightedSizeSampler | CurriculumSizeSampler,
        *,
        shaped_reward_norm: str = "none",
        shaped_step_penalty: float = -0.05,
        shaped_dense_mult: float = 1.0,
        random_pad_offset: bool = False,
    ):
        super().__init__()
        self.sizes = sorted(set(sizes))
        self.max_size = max_size
        self.reward_mode = reward_mode
        self.dup_coef = dup_coef
        self.adj_coef = adj_coef
        self.comp_coef = comp_coef
        self.size_sampler = size_sampler
        self.envs = {
            size: make_env(
                size=size,
                max_size=max_size,
                reward_mode=reward_mode,
                dup_coef=dup_coef,
                adj_coef=adj_coef,
                comp_coef=comp_coef,
                shaped_reward_norm=shaped_reward_norm,
                shaped_step_penalty=shaped_step_penalty,
                shaped_dense_mult=shaped_dense_mult,
                random_pad_offset=random_pad_offset,
            )
            for size in self.sizes
        }
        template_env = self.envs[self.sizes[0]]
        self.observation_space = template_env.observation_space
        self.action_space = template_env.action_space
        self.current_size = self.sizes[0]
        self.current_env = self.envs[self.current_size]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        assert self.np_random is not None
        self.current_size = self.size_sampler.sample(self.np_random)
        self.current_env = self.envs[self.current_size]
        child_seed = int(self.np_random.integers(0, 2**31 - 1))
        obs, info = self.current_env.reset(seed=child_seed, options=options)
        info = dict(info)
        info["sampled_size"] = self.current_size
        info["size_sampler"] = self.size_sampler.describe()
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        info = dict(info)
        info["sampled_size"] = self.current_size
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.current_env.action_masks()

    def set_training_progress(self, progress: float) -> None:
        self.size_sampler.set_progress(progress)

    def close(self) -> None:
        for env in self.envs.values():
            env.close()
        super().close()


def make_env_factory(
    *,
    size: int,
    max_size: int,
    seed: int | None,
    reward_mode: str,
    dup_coef: float,
    adj_coef: float,
    comp_coef: float,
    shaped_reward_norm: str = "none",
    shaped_step_penalty: float = -0.05,
    shaped_dense_mult: float = 1.0,
    random_pad_offset: bool = False,
):
    """Return a thunk for VecEnv creation."""

    def _factory():
        return make_env(
            size=size,
            max_size=max_size,
            seed=seed,
            reward_mode=reward_mode,
            dup_coef=dup_coef,
            adj_coef=adj_coef,
            comp_coef=comp_coef,
            shaped_reward_norm=shaped_reward_norm,
            shaped_step_penalty=shaped_step_penalty,
            shaped_dense_mult=shaped_dense_mult,
            random_pad_offset=random_pad_offset,
        )

    return _factory


def assign_env_sizes(train_sizes: list[int], n_envs: int) -> list[int]:
    """Round-robin assign board sizes to vectorized env slots."""

    if not train_sizes:
        raise ValueError("train_sizes must not be empty")
    return [train_sizes[index % len(train_sizes)] for index in range(n_envs)]


def make_vec_env(
    *,
    train_sizes: list[int],
    max_size: int,
    n_envs: int,
    seed: int | None,
    reward_mode: str,
    dup_coef: float,
    adj_coef: float,
    comp_coef: float,
    shaped_reward_norm: str = "none",
    shaped_step_penalty: float = -0.05,
    shaped_dense_mult: float = 1.0,
    vec_env_type: str = "dummy",
    sampling_strategy: str = "round_robin",
    train_weights: list[float] | None = None,
    curriculum_thresholds: list[float] | None = None,
    random_pad_offset: bool = False,
) -> tuple[VecEnv, list[str]]:
    """Build a vectorized env whose slots may use different board sizes."""

    if sampling_strategy == "round_robin":
        assigned_sizes = assign_env_sizes(train_sizes, n_envs)
        env_fns = [
            make_env_factory(
                size=size,
                max_size=max_size,
                seed=None if seed is None else seed + index,
                reward_mode=reward_mode,
                dup_coef=dup_coef,
                adj_coef=adj_coef,
                comp_coef=comp_coef,
                shaped_reward_norm=shaped_reward_norm,
                shaped_step_penalty=shaped_step_penalty,
                shaped_dense_mult=shaped_dense_mult,
                random_pad_offset=random_pad_offset,
            )
            for index, size in enumerate(assigned_sizes)
        ]
        descriptions = [f"fixed:{size}" for size in assigned_sizes]
    else:
        env_fns = []
        descriptions = []
        for index in range(n_envs):
            if sampling_strategy == "weighted":
                sampler = WeightedSizeSampler(train_sizes, weights=train_weights)
            elif sampling_strategy == "curriculum":
                sampler = CurriculumSizeSampler(
                    train_sizes,
                    thresholds=curriculum_thresholds,
                    weights=train_weights,
                )
            else:
                raise ValueError(f"unsupported sampling_strategy={sampling_strategy!r}")

            def _factory(
                sampler=sampler,
                train_sizes=train_sizes,
                max_size=max_size,
                reward_mode=reward_mode,
                dup_coef=dup_coef,
                adj_coef=adj_coef,
                comp_coef=comp_coef,
                shaped_reward_norm=shaped_reward_norm,
                shaped_step_penalty=shaped_step_penalty,
                shaped_dense_mult=shaped_dense_mult,
                seed=seed,
                index=index,
                random_pad_offset=random_pad_offset,
            ):
                env = MultiSizeHitoriEnv(
                    sizes=train_sizes,
                    max_size=max_size,
                    reward_mode=reward_mode,
                    dup_coef=dup_coef,
                    adj_coef=adj_coef,
                    comp_coef=comp_coef,
                    size_sampler=sampler,
                    shaped_reward_norm=shaped_reward_norm,
                    shaped_step_penalty=shaped_step_penalty,
                    shaped_dense_mult=shaped_dense_mult,
                    random_pad_offset=random_pad_offset,
                )
                if seed is not None:
                    env.reset(seed=seed + index)
                return env

            env_fns.append(_factory)
            descriptions.append(sampler.describe())

    if vec_env_type == "dummy":
        return DummyVecEnv(env_fns), descriptions
    if vec_env_type == "subproc":
        return SubprocVecEnv(env_fns), descriptions

    raise ValueError(f"unsupported vec_env_type={vec_env_type!r}")
