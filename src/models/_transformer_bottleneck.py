"""Transformer body for the U-Net bottleneck.

A small ``nn.TransformerEncoder`` stack that operates on the flattened
bottleneck feature map. Drop-in replacement for the ``DoubleConv`` body of
the ``Bottleneck`` module in ``src.models.custom_unet`` -- the signature is
``(B, C, H, W) -> (B, C, H, W)`` and channels are preserved.

Why here: attention is global by construction but quadratic in tokens; the
bottleneck is the only place in the U where the spatial grid is small enough
to make global attention cheap (8x8 = 64 tokens at 256x256 input).
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class TransformerBottleneck(nn.Module):
    """Stack of TransformerEncoder layers operating on bottleneck tokens.

    Args:
        channels: Channel count of the bottleneck feature map. Output has the
            same channel count -- this module is shape-preserving.
        num_heads: Attention heads per layer. ``channels`` must be divisible
            by ``num_heads``.
        num_layers: Stacked ``nn.TransformerEncoderLayer`` blocks.
        mlp_ratio: Feed-forward hidden dim = ``channels * mlp_ratio``.
        spatial_size: Side length of the square positional embedding. If the
            forward input has a different spatial size, the embedding is
            bilinearly interpolated to match (so the module also accepts
            non-default and non-square inputs at inference).

    Raises:
        ValueError: If ``channels`` is not divisible by ``num_heads``.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: int = 2,
        spatial_size: int = 8,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads "
                f"({num_heads}) for nn.MultiheadAttention."
            )
        if spatial_size < 1:
            raise ValueError(f"spatial_size must be >= 1 (got {spatial_size})")

        self.channels = int(channels)
        self.spatial_size = int(spatial_size)

        # 2D positional embedding stored as [1, C, H, W] for easy add + interp.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, channels, self.spatial_size, self.spatial_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * mlp_ratio,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels (got {c}). "
                "TransformerBottleneck is shape-preserving and built for a "
                "fixed channel count."
            )
        # Pos embed: resize if input spatial differs from configured size.
        if (h, w) == (self.spatial_size, self.spatial_size):
            pos = self.pos_embed
        else:
            pos = F.interpolate(
                self.pos_embed, size=(h, w), mode="bilinear", align_corners=False
            )
        x = x + pos
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        tokens = self.encoder(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)
