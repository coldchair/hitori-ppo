"""Network components for Hitori PPO training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def board_mask_from_observations(observations: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return a 1xHxW mask for active board positions."""

    return observations["game_grid"].float().gt(0).float().unsqueeze(1)


def build_hitori_cell_features(observations: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build per-cell channels used by all Hitori policies."""

    game = observations["game_grid"].float()
    shaded = observations["shaded"].float()
    board_mask = game.gt(0).float()
    unshaded = (1.0 - shaded) * board_mask

    same_row_value = game.unsqueeze(3) == game.unsqueeze(2)
    row_counts = (same_row_value.float() * unshaded.unsqueeze(2)).sum(dim=3)

    same_col_value = game.unsqueeze(2) == game.unsqueeze(1)
    col_counts = (same_col_value.float() * unshaded.unsqueeze(1)).sum(dim=2)

    board_cells = board_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
    board_size = torch.sqrt(board_cells)
    norm = torch.clamp(board_size - 1.0, min=1.0)

    row_duplicate_pressure = torch.clamp(row_counts - unshaded, min=0.0) / norm
    col_duplicate_pressure = torch.clamp(col_counts - unshaded, min=0.0) / norm

    padded_shaded = F.pad(shaded.unsqueeze(1), (1, 1, 1, 1), mode="constant", value=0.0)
    adjacent_shaded = (
        padded_shaded[:, :, 1:-1, :-2]
        + padded_shaded[:, :, 1:-1, 2:]
        + padded_shaded[:, :, :-2, 1:-1]
        + padded_shaded[:, :, 2:, 1:-1]
    ).squeeze(1) / 4.0

    max_value = game.amax(dim=(1, 2), keepdim=True).clamp(min=1.0)
    normalized_game = game / max_value
    batch_size, height, width = game.shape
    row_coord = torch.linspace(0.0, 1.0, steps=height, device=game.device, dtype=game.dtype).view(1, height, 1)
    col_coord = torch.linspace(0.0, 1.0, steps=width, device=game.device, dtype=game.dtype).view(1, 1, width)
    row_coord = row_coord.expand(batch_size, -1, width) * board_mask
    col_coord = col_coord.expand(batch_size, height, -1) * board_mask
    board_size_ratio = (board_size / float(height)).expand(-1, height, width) * board_mask

    return torch.stack(
        [
            normalized_game,
            shaded,
            row_duplicate_pressure,
            col_duplicate_pressure,
            adjacent_shaded * board_mask,
            board_mask,
            row_coord,
            col_coord,
            board_size_ratio,
        ],
        dim=1,
    )


class HitoriStructuredFeatureExtractor(BaseFeaturesExtractor):
    """Flattened feature extractor for padded Hitori boards."""

    def __init__(self, observation_space: spaces.Dict):
        board_space = observation_space["game_grid"]
        self.max_size = int(board_space.shape[0])
        channels = 9
        super().__init__(observation_space, features_dim=channels * self.max_size * self.max_size)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return build_hitori_cell_features(observations).flatten(start_dim=1)


class HitoriCNNFeatureExtractor(BaseFeaturesExtractor):
    """Small CNN feature extractor for padded Hitori boards."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        board_space = observation_space["game_grid"]
        self.max_size = int(board_space.shape[0])
        super().__init__(observation_space, features_dim=features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.max_size * self.max_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.cnn(build_hitori_cell_features(observations))


def _group_norm_groups(channels: int, preferred: int = 8) -> int:
    g = min(preferred, channels)
    while g > 0 and channels % g != 0:
        g -= 1
    return max(1, g)


class CNNV2ResBlock(nn.Module):
    """Residual block with two 3×3 convs; optional dilation (same H×W)."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")
        pad = dilation
        ng = _group_norm_groups(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(ng, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(ng, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.act(x)


class HitoriCNNV2Backbone(nn.Module):
    """Stem + 4 residual blocks (spec CNNV2)."""

    def __init__(self, in_channels: int = 9, stem_channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.block1 = CNNV2ResBlock(stem_channels, dilation=1)
        self.block2 = CNNV2ResBlock(stem_channels, dilation=1)
        self.block3 = CNNV2ResBlock(stem_channels, dilation=2)
        self.block4 = CNNV2ResBlock(stem_channels, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class HitoriCNNV2FeatureExtractor(BaseFeaturesExtractor):
    """9-channel cell features → 32-channel spatial map for CNNV2 actor/critic heads."""

    def __init__(self, observation_space: spaces.Dict):
        board_space = observation_space["game_grid"]
        self.max_size = int(board_space.shape[0])
        out_ch = 32
        super().__init__(observation_space, features_dim=out_ch)
        self.backbone = HitoriCNNV2Backbone(in_channels=9, stem_channels=out_ch)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        x = build_hitori_cell_features(observations)
        feat = self.backbone(x)
        return feat * board_mask_from_observations(observations)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        return self.act(x)


class AxialAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.row_attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm_row = nn.LayerNorm(channels)
        self.norm_col = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.SiLU(),
            nn.Linear(channels * 4, channels),
        )
        self.norm_ffn = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        row_tokens = x.permute(0, 2, 3, 1).reshape(batch_size * height, width, channels)
        row_attn_out, _ = self.row_attn(row_tokens, row_tokens, row_tokens, need_weights=False)
        row_tokens = self.norm_row(row_tokens + row_attn_out)

        col_tokens = row_tokens.reshape(batch_size, height, width, channels).permute(0, 2, 1, 3).reshape(
            batch_size * width, height, channels
        )
        col_attn_out, _ = self.col_attn(col_tokens, col_tokens, col_tokens, need_weights=False)
        col_tokens = self.norm_col(col_tokens + col_attn_out)

        ffn_out = self.ffn(col_tokens)
        col_tokens = self.norm_ffn(col_tokens + ffn_out)

        return col_tokens.reshape(batch_size, width, height, channels).permute(0, 3, 2, 1)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class HitoriUNetBackbone(nn.Module):
    """Compact U-Net style backbone that preserves board resolution."""

    def __init__(
        self,
        in_channels: int = 9,
        base_channels: int = 64,
        out_channels: int = 64,
        num_attention_blocks: int = 1,
    ):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.attention = nn.Sequential(
            *[AxialAttentionBlock(base_channels * 8, num_heads=8) for _ in range(num_attention_blocks)]
        )
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up3 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, ceil_mode=True))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, ceil_mode=True))
        bottleneck = self.bottleneck(F.max_pool2d(enc3, kernel_size=2, ceil_mode=True))
        bottleneck = self.attention(bottleneck)
        dec1 = self.up1(bottleneck, enc3)
        dec2 = self.up2(dec1, enc2)
        dec3 = self.up3(dec2, enc1)
        return self.out_conv(dec3)


class HitoriUNetFeatureExtractor(BaseFeaturesExtractor):
    """U-Net feature extractor that returns a same-resolution feature map."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        base_channels: int = 64,
        out_channels: int = 64,
        num_attention_blocks: int = 1,
    ):
        super().__init__(observation_space, features_dim=out_channels)
        self.backbone = HitoriUNetBackbone(
            in_channels=9,
            base_channels=base_channels,
            out_channels=out_channels,
            num_attention_blocks=num_attention_blocks,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        feature_map = self.backbone(build_hitori_cell_features(observations))
        return feature_map * board_mask_from_observations(observations)
