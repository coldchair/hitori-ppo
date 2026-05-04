"""Random baseline for hitori-gym using the environment action mask."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import hitori_env  # noqa: F401 - registers hitori_env/Hitori-v2
import numpy as np


@dataclass
class EpisodeResult:
    solved: bool
    stuck: bool
    reward: float
    length: int


def run_episode(size: int, seed: int, max_steps: int, rng: np.random.Generator) -> EpisodeResult:
    env = gym.make("hitori_env/Hitori-v2", size=size)
    _, _ = env.reset(seed=seed)

    total_reward = 0.0
    solved = False
    stuck = False

    try:
        for step in range(1, max_steps + 1):
            action_mask = env.unwrapped.action_masks()
            valid_actions = np.flatnonzero(action_mask)

            if len(valid_actions) == 0:
                stuck = True
                return EpisodeResult(solved=False, stuck=stuck, reward=total_reward, length=step - 1)

            action = int(rng.choice(valid_actions))
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                solved = terminated and reward > 0
                stuck = terminated and reward <= 0
                return EpisodeResult(solved=solved, stuck=stuck, reward=total_reward, length=step)
    finally:
        env.close()

    return EpisodeResult(solved=False, stuck=False, reward=total_reward, length=max_steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results = [
        run_episode(
            size=args.size,
            seed=args.seed + episode,
            max_steps=args.max_steps,
            rng=rng,
        )
        for episode in range(args.episodes)
    ]

    rewards = np.array([result.reward for result in results], dtype=float)
    lengths = np.array([result.length for result in results], dtype=float)
    solved = np.array([result.solved for result in results], dtype=bool)
    stuck = np.array([result.stuck for result in results], dtype=bool)

    print(f"Random + Mask baseline, size={args.size}, episodes={args.episodes}, seed={args.seed}")
    print(f"solve_rate={solved.mean():.3f}")
    print(f"stuck_rate={stuck.mean():.3f}")
    print(f"avg_reward={rewards.mean():.3f}")
    print(f"avg_length={lengths.mean():.2f}")
    print(f"reward_std={rewards.std():.3f}")


if __name__ == "__main__":
    main()
