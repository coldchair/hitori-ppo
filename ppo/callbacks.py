"""Callbacks for PPO training."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import JSONOutputFormat

from ppo.evaluation import EvalRow, evaluate_all_sizes, plot_learning_curves_per_size


def set_policy_critic_output_only_training(policy: Any, critic_output_only: bool) -> None:
    """
    If ``critic_output_only`` is True, freeze backbone / actor and train only the scalar value
    readout (``value_net``): for U-Net policies this is the final ``Linear`` after ``value_head``;
    for MLP ``MultiInputPolicy`` this is the last ``Linear`` after ``mlp_extractor`` (the whole
    MLP extractor including the vf stack is frozen).

    If False, all parameters are marked trainable again.
    """

    if not critic_output_only:
        for p in policy.parameters():
            p.requires_grad = True
        return

    if hasattr(policy, "value_head"):
        for p in policy.features_extractor.parameters():
            p.requires_grad = False
        for p in policy.action_net.parameters():
            p.requires_grad = False
        for p in policy.value_head.parameters():
            p.requires_grad = False
        for p in policy.value_net.parameters():
            p.requires_grad = True
        return

    for p in policy.features_extractor.parameters():
        p.requires_grad = False
    for p in policy.mlp_extractor.parameters():
        p.requires_grad = False
    for p in policy.action_net.parameters():
        p.requires_grad = False
    for p in policy.value_net.parameters():
        p.requires_grad = True


class CriticOutputHeadWarmupCallback(BaseCallback):
    """Until ``until_timesteps`` env steps (SB3 ``num_timesteps``), only update the critic scalar head."""

    def __init__(self, until_timesteps: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.until_timesteps = int(until_timesteps)
        self._last_critic_only: bool | None = None

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        if self.until_timesteps <= 0:
            return
        if self.verbose:
            print(
                f"[critic-warmup] while num_timesteps < {self.until_timesteps}, only the critic scalar "
                "head is trained (backbone + actor frozen before each PPO update phase).",
                flush=True,
            )

    def _on_rollout_end(self) -> None:
        if self.until_timesteps <= 0:
            return
        want = self.num_timesteps < self.until_timesteps
        set_policy_critic_output_only_training(self.model.policy, want)
        if self.verbose and want and self._last_critic_only is None:
            print(
                f"[critic-warmup] first policy update at num_timesteps={self.num_timesteps}: "
                "critic-output-only mode active",
                flush=True,
            )
        if (
            self.verbose
            and self._last_critic_only is True
            and not want
        ):
            print(
                f"[critic-warmup] timesteps={self.num_timesteps} >= {self.until_timesteps}: "
                "unfroze backbone+actor for subsequent updates",
                flush=True,
            )
        self._last_critic_only = want

    def _on_training_end(self) -> None:
        if self.until_timesteps <= 0:
            return
        set_policy_critic_output_only_training(self.model.policy, False)


def _fmt_duration(seconds: float) -> str:
    if seconds != seconds or seconds < 0:  # NaN or invalid
        return "n/a"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


class JsonlTrainMetricsCallback(BaseCallback):
    """Append Stable-Baselines3 ``logger`` KV dumps to a JSONL file (same keys as CLI tables)."""

    def __init__(self, jsonl_path: Path, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.jsonl_path = Path(jsonl_path)
        self._json_writer: JSONOutputFormat | None = None

    def _init_callback(self) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._json_writer = JSONOutputFormat(str(self.jsonl_path))
        self.model.logger.output_formats.append(self._json_writer)
        if self.verbose:
            print(f"[jsonl] train metrics -> {self.jsonl_path}", flush=True)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if self._json_writer is None:
            return
        try:
            self.model.logger.output_formats.remove(self._json_writer)
        except ValueError:
            pass
        self._json_writer.close()
        self._json_writer = None


class EvalAndCheckpointCallback(BaseCallback):
    def __init__(
        self,
        eval_sizes: list[int],
        max_size: int,
        out_dir: Path,
        eval_freq: int,
        eval_episodes: int,
        eval_seed: int,
        max_steps: int,
        reward_mode: str,
        dup_coef: float,
        adj_coef: float,
        comp_coef: float,
        shaped_reward_norm: str,
        shaped_step_penalty: float,
        shaped_dense_mult: float,
        eval_vec_envs: int,
        total_timesteps: int,
        verbose: int = 1,
        *,
        eval_at_timestep_zero: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.eval_at_timestep_zero = bool(eval_at_timestep_zero)
        self.eval_sizes = eval_sizes
        self.max_size = max_size
        self.out_dir = out_dir
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.dup_coef = dup_coef
        self.adj_coef = adj_coef
        self.comp_coef = comp_coef
        self.shaped_reward_norm = shaped_reward_norm
        self.shaped_step_penalty = shaped_step_penalty
        self.shaped_dense_mult = shaped_dense_mult
        self.eval_vec_envs = eval_vec_envs
        if eval_freq <= 0:
            raise ValueError("eval_freq must be positive")
        planned = total_timesteps // eval_freq
        self._total_periodic_evals = max(1, planned)
        self._completed_periodic_evals = 0
        self.rows: list[EvalRow] = []
        self.csv_path = out_dir / "learning_curve.csv"
        self.checkpoint_dir = out_dir / "checkpoints"
        self._best_mean_solve: float = -1.0
        self._best_timesteps: int = 0
        self._next_eval_timestep = eval_freq
        self._train_wall_t0: float | None = None
        self._eval_wall_sum: float = 0.0

    def _sync_curriculum_progress(self) -> None:
        progress = 1.0 - float(self.model._current_progress_remaining)
        if hasattr(self.training_env, "has_attr") and self.training_env.has_attr("set_training_progress"):
            self.training_env.env_method("set_training_progress", progress)

    def _init_callback(self) -> None:
        self._train_wall_t0 = time.perf_counter()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._next_eval_timestep = self.eval_freq
        self._eval_wall_sum = 0.0
        self._completed_periodic_evals = 0
        with self.csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "size", "mean_reward", "solve_rate", "mean_length"])
        # Runs once learn() has attached ``self.model`` (before first rollout). Avoids relying on
        # ``_on_training_start`` across SB3 minor versions.
        if self.eval_at_timestep_zero:
            self._run_eval(csv_timesteps=0, bump_periodic_counter=False)

    def _run_eval(self, *, csv_timesteps: int, bump_periodic_counter: bool) -> None:
        pure_train_before_s = 0.0
        elapsed_total_s = 0.0
        if self._train_wall_t0 is not None:
            now = time.perf_counter()
            elapsed_total_s = now - self._train_wall_t0
            pure_train_before_s = elapsed_total_s - self._eval_wall_sum

        done_evals = self._completed_periodic_evals
        total_evals = self._total_periodic_evals
        eta_start_s = float("nan")
        if bump_periodic_counter and done_evals > 0 and self._train_wall_t0 is not None:
            eta_start_s = elapsed_total_s * (total_evals - done_evals) / done_evals

        tag = "pretrain@timesteps=0 " if not bump_periodic_counter else ""
        print(
            f"[eval] {tag}start csv_timesteps={csv_timesteps} train_ts={self.num_timesteps} "
            f"pure_train_wall_s={pure_train_before_s:.3f} elapsed_total_s={elapsed_total_s:.3f} "
            f"periodic_eval={done_evals}/{total_evals} eta_remaining_s≈{_fmt_duration(eta_start_s)}",
            flush=True,
        )
        eval_t0 = time.perf_counter()
        eval_rows, eval_env_steps = evaluate_all_sizes(
            model=self.model,
            eval_sizes=self.eval_sizes,
            max_size=self.max_size,
            episodes=self.eval_episodes,
            eval_seed=self.eval_seed,
            max_steps=self.max_steps,
            reward_mode=self.reward_mode,
            dup_coef=self.dup_coef,
            adj_coef=self.adj_coef,
            comp_coef=self.comp_coef,
            shaped_reward_norm=self.shaped_reward_norm,
            shaped_step_penalty=self.shaped_step_penalty,
            shaped_dense_mult=self.shaped_dense_mult,
            timesteps=int(csv_timesteps),
            vec_envs=self.eval_vec_envs,
        )

        for row in eval_rows:
            self.rows.append(row)

        eval_wall_s = time.perf_counter() - eval_t0
        self._eval_wall_sum += eval_wall_s
        if bump_periodic_counter:
            self._completed_periodic_evals += 1
        pure_train_wall_s = 0.0
        elapsed_total_s = 0.0
        if self._train_wall_t0 is not None:
            elapsed_total_s = time.perf_counter() - self._train_wall_t0
            pure_train_wall_s = elapsed_total_s - self._eval_wall_sum

        k = self._completed_periodic_evals
        n = self._total_periodic_evals
        eta_remaining_s = float("nan")
        if bump_periodic_counter:
            if k > 0 and k < n:
                eta_remaining_s = elapsed_total_s * (n - k) / k
            elif k >= n:
                eta_remaining_s = 0.0

        tag_done = "pretrain@0 " if not bump_periodic_counter else ""
        print(
            f"[eval] {tag_done}done pure_train_wall_s={pure_train_wall_s:.3f} eval_wall_s={eval_wall_s:.3f} "
            f"eval_env_steps={eval_env_steps} timesteps={eval_rows[0].timesteps} "
            f"eval_vec_envs={self.eval_vec_envs} sizes={len(self.eval_sizes)} "
            f"episodes_per_size={self.eval_episodes} "
            f"elapsed_total_s={elapsed_total_s:.3f} periodic_eval={k}/{n} "
            f"eta_remaining_s≈{_fmt_duration(eta_remaining_s)}",
            flush=True,
        )

        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            for row in eval_rows:
                writer.writerow([row.timesteps, row.size, row.mean_reward, row.solve_rate, row.mean_length])

        plot_learning_curves_per_size(self.csv_path, self.out_dir)

        mean_solve = float(np.mean([row.solve_rate for row in eval_rows]))
        ts = int(eval_rows[0].timesteps)
        if mean_solve > self._best_mean_solve + 1e-12:
            self._best_mean_solve = mean_solve
            self._best_timesteps = ts
            best_stem = self.checkpoint_dir / f"maskable_ppo_{self.reward_mode}_best"
            self.model.save(str(best_stem))
            meta_path = self.checkpoint_dir / "best_checkpoint.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "timesteps": ts,
                        "mean_solve_rate": mean_solve,
                        "per_size_solve_rate": {str(row.size): row.solve_rate for row in eval_rows},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                f"[checkpoint] new best mean_solve_rate={mean_solve:.6f} (all eval sizes) timesteps={ts} "
                f"-> {best_stem}.zip",
                flush=True,
            )

        if self.verbose:
            summary = ", ".join(
                (
                    f"{row.size}x{row.size}: "
                    f"reward={row.mean_reward:.3f}, "
                    f"solve={row.solve_rate:.3f}, "
                    f"len={row.mean_length:.2f}"
                )
                for row in eval_rows
            )
            print(f"eval @ {eval_rows[0].timesteps}: {summary}", flush=True)
        if bump_periodic_counter:
            while self._next_eval_timestep <= self.num_timesteps:
                self._next_eval_timestep += self.eval_freq

    def _on_step(self) -> bool:
        self._sync_curriculum_progress()
        if self.num_timesteps < self._next_eval_timestep:
            return True
        self._run_eval(csv_timesteps=int(self.num_timesteps), bump_periodic_counter=True)
        return True
