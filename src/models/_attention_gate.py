"""Attention Gate for U-Net skip connections (Oktay et al. 2018).

Used as a drop-in for the ``skip_refine`` slot of the ``Up`` block defined in
``src.models.custom_unet``. The gate learns a spatial attention map alpha in
``[0, 1]`` from two co-resolution feature maps -- the encoder skip and the
decoder gating signal -- and returns ``alpha * skip``. The intent is to
suppress decoder-irrelevant skip regions (background, clutter) and emphasise
foreground, which helps underrepresented classes whose gradient is otherwise
drowned by the majority pixels.

Reference: Oktay et al. (2018), "Attention U-Net: Learning Where to Look for
the Pancreas", MIDL.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class AttentionGate(nn.Module):
    """Additive attention gate.

    Both inputs must share spatial dimensions (the decoder gating tensor must
    already be upsampled before the gate is called -- which is exactly what
    happens inside ``Up.forward`` after ``self.up(x)``).

    Args:
        skip_channels: Channels of the encoder skip tensor.
        gating_channels: Channels of the decoder gating tensor (the upsampled
            feature being passed into the next decoder stage).
        inter_channels: Width of the intermediate projection. Defaults to
            ``max(skip_channels // 2, 1)``; must be ``>= 1``.

    Raises:
        ValueError: If ``inter_channels < 1``.
    """

    def __init__(
        self,
        skip_channels: int,
        gating_channels: int,
        inter_channels: int | None = None,
    ) -> None:
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 1)
        if inter_channels < 1:
            raise ValueError(f"inter_channels must be >= 1 (got {inter_channels})")

        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_gating = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        # Pre-activation ReLU on the additive merge (Oktay et al. eq. 2).
        self.relu = nn.ReLU(inplace=False)

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] != gating.shape[-2:]:
            raise ValueError(
                "AttentionGate expects matching spatial dims for skip and "
                f"gating; got skip={tuple(skip.shape)} gating={tuple(gating.shape)}"
            )
        merged = self.relu(self.W_skip(skip) + self.W_gating(gating))
        alpha = self.psi(merged)  # [B, 1, H, W] in (0, 1)
        return skip * alpha
