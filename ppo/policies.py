"""Policy definitions for Hitori PPO training."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from ppo.networks import (
    HitoriCNNFeatureExtractor,
    HitoriCNNV2FeatureExtractor,
    HitoriStructuredFeatureExtractor,
    HitoriUNetFeatureExtractor,
    board_mask_from_observations,
)


class HitoriUNetPolicy(MaskableMultiInputActorCriticPolicy):
    """Custom maskable policy whose actor operates directly on a feature map."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = HitoriUNetFeatureExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        value_channels: int = 64,
    ):
        if not share_features_extractor:
            raise ValueError("HitoriUNetPolicy requires share_features_extractor=True")

        self.value_channels = value_channels
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=[] if net_arch is None else net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(value_channels=self.value_channels)
        return data

    def _build(self, lr_schedule: Schedule) -> None:
        self.action_net = nn.Conv2d(self.features_dim, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Conv2d(self.features_dim, self.value_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.value_net = nn.Linear(self.value_channels, 1)
        self.mlp_extractor = nn.Identity()

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_head: np.sqrt(2),
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _value_from_feature_map(self, feature_map: th.Tensor, board_mask: th.Tensor) -> th.Tensor:
        value_features = self.value_head(feature_map) * board_mask
        denom = board_mask.sum(dim=(2, 3)).clamp(min=1.0)
        pooled = value_features.sum(dim=(2, 3)) / denom
        return self.value_net(pooled)

    def _distribution_from_feature_map(self, feature_map: th.Tensor, board_mask: th.Tensor) -> MaskableDistribution:
        action_map = self.action_net(feature_map) * board_mask
        action_logits = action_map.flatten(start_dim=1)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def forward(
        self,
        obs: PyTorchObs,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        feature_map = self.extract_features(obs)
        board_mask = board_mask_from_observations(obs)
        values = self._value_from_feature_map(feature_map, board_mask)
        distribution = self._distribution_from_feature_map(feature_map, board_mask)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        feature_map = self.extract_features(obs)
        board_mask = board_mask_from_observations(obs)
        distribution = self._distribution_from_feature_map(feature_map, board_mask)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self._value_from_feature_map(feature_map, board_mask)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: PyTorchObs, action_masks: np.ndarray | None = None) -> MaskableDistribution:
        feature_map = self.extract_features(obs)
        board_mask = board_mask_from_observations(obs)
        distribution = self._distribution_from_feature_map(feature_map, board_mask)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        feature_map = self.extract_features(obs)
        board_mask = board_mask_from_observations(obs)
        return self._value_from_feature_map(feature_map, board_mask)


class HitoriCNNV2Policy(HitoriUNetPolicy):
    """CNNV2: same-resolution 32-channel map; actor/critic heads per architecture spec (critic: Conv 32→32, no mid ReLU)."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = HitoriCNNV2FeatureExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        value_channels: int = 32,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs if features_extractor_kwargs is not None else {},
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            value_channels=value_channels,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        self.action_net = nn.Conv2d(self.features_dim, 1, kernel_size=1)
        self.value_head = nn.Conv2d(self.features_dim, self.value_channels, kernel_size=1)
        self.value_net = nn.Linear(self.value_channels, 1)
        self.mlp_extractor = nn.Identity()

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_head: np.sqrt(2),
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def make_policy_spec(
    model_name: str,
    *,
    num_attention_blocks: int = 1,
) -> tuple[str | type[MaskableMultiInputActorCriticPolicy], dict[str, Any]]:
    if model_name == "structured":
        return "MultiInputPolicy", {
            "features_extractor_class": HitoriStructuredFeatureExtractor,
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        }
    if model_name == "cnn":
        return "MultiInputPolicy", {
            "features_extractor_class": HitoriCNNFeatureExtractor,
            "net_arch": {"pi": [128, 128], "vf": [128, 128]},
        }
    if model_name == "cnnv2":
        return HitoriCNNV2Policy, {
            "features_extractor_class": HitoriCNNV2FeatureExtractor,
            "features_extractor_kwargs": {},
            "net_arch": [],
            "value_channels": 32,
        }
    if model_name == "unet":
        if num_attention_blocks < 0:
            raise ValueError(f"num_attention_blocks must be >= 0, got {num_attention_blocks}")
        return HitoriUNetPolicy, {
            "features_extractor_class": HitoriUNetFeatureExtractor,
            "features_extractor_kwargs": {
                "base_channels": 64,
                "out_channels": 64,
                "num_attention_blocks": int(num_attention_blocks),
            },
            "net_arch": [],
        }

    choices = ", ".join(["structured", "cnn", "cnnv2", "unet"])
    raise ValueError(f"unknown model {model_name!r}; choose one of: {choices}")
