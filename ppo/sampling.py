"""Board-size sampling strategies for mixed-size training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_weights(weights: list[float]) -> np.ndarray:
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.ndim != 1 or weight_array.size == 0:
        raise ValueError("weights must be a non-empty 1D list")
    if np.any(weight_array < 0):
        raise ValueError("weights must be non-negative")
    total = float(weight_array.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return weight_array / total


@dataclass
class WeightedSizeSampler:
    sizes: list[int]
    weights: np.ndarray

    def __init__(self, sizes: list[int], weights: list[float] | None = None):
        self.sizes = list(sizes)
        if not self.sizes:
            raise ValueError("sizes must not be empty")
        if weights is None:
            weights = [1.0] * len(self.sizes)
        if len(weights) != len(self.sizes):
            raise ValueError("weights must have the same length as sizes")
        self.weights = normalize_weights(weights)

    def sample(self, rng: np.random.Generator) -> int:
        index = int(rng.choice(len(self.sizes), p=self.weights))
        return self.sizes[index]

    def set_progress(self, progress: float) -> None:
        del progress

    def describe(self) -> str:
        parts = [f"{size}:{weight:.3f}" for size, weight in zip(self.sizes, self.weights)]
        return f"weighted[{', '.join(parts)}]"


@dataclass
class CurriculumSizeSampler:
    sizes: list[int]
    base_weights: np.ndarray
    thresholds: list[float]
    progress: float = 0.0

    def __init__(
        self,
        sizes: list[int],
        thresholds: list[float] | None = None,
        weights: list[float] | None = None,
    ):
        self.sizes = list(sizes)
        if not self.sizes:
            raise ValueError("sizes must not be empty")
        if weights is None:
            weights = [1.0] * len(self.sizes)
        if len(weights) != len(self.sizes):
            raise ValueError("weights must have the same length as sizes")
        self.base_weights = normalize_weights(weights)

        if thresholds is None:
            thresholds = [index / len(self.sizes) for index in range(1, len(self.sizes))]
        if len(thresholds) != max(len(self.sizes) - 1, 0):
            raise ValueError("curriculum thresholds must have len(sizes) - 1 entries")
        if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
            raise ValueError("curriculum thresholds must be in [0, 1]")
        if list(thresholds) != sorted(thresholds):
            raise ValueError("curriculum thresholds must be sorted ascending")
        self.thresholds = list(thresholds)

    def unlocked_count(self) -> int:
        return 1 + sum(self.progress >= threshold for threshold in self.thresholds)

    def sample(self, rng: np.random.Generator) -> int:
        count = self.unlocked_count()
        weights = self.base_weights[:count]
        weights = weights / weights.sum()
        index = int(rng.choice(count, p=weights))
        return self.sizes[index]

    def set_progress(self, progress: float) -> None:
        self.progress = float(np.clip(progress, 0.0, 1.0))

    def describe(self) -> str:
        unlocked = self.sizes[: self.unlocked_count()]
        return f"curriculum[progress={self.progress:.3f}, unlocked={unlocked}]"

