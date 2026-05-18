"""Hand-implemented building blocks for the custom LWIR U-Net.

Three blocks live here:

* :class:`DepthwiseSeparableConv` -- 3x3 depthwise + 1x1 pointwise + GroupNorm
  + SiLU. The workhorse used everywhere except the stem.
* :class:`StandardConvBlock` -- plain 3x3 Conv + GroupNorm + SiLU. Used only
  at the stem, where the input has 1 channel (so depthwise conv is degenerate).
* :class:`DoubleDSConv` -- two :class:`DepthwiseSeparableConv` blocks back to
  back. Drop-in replacement for the encoder/decoder double-conv pattern.

Why GroupNorm + SiLU and not BatchNorm + ReLU?

* GroupNorm is batch-size-independent. The thesis runs sometimes need
  batch=1 (smoke tests, OOM recovery, deployment); BatchNorm misbehaves at
  batch=1, GroupNorm doesn't.
* SiLU (a.k.a. swish) has been the consensus successor to ReLU in modern
  encoder-decoder vision models -- smoother gradient near zero, no dead
  units. The downside (more compute) is offset by depthwise-separable
  convs cutting the per-block FLOP cost.

Group-count safety: ``nn.GroupNorm(num_groups=G, num_channels=C)`` requires
``C % G == 0``. At very small widths (e.g. stem at 32 channels with G=8 that's
4 ch/group, fine; with G=16 it would be 2 ch/group which is workable but
noisier). The helper :func:`_safe_num_groups` clamps to ``min(G, C // 4)``
so any block survives down to ~4 channels without crashing.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _safe_num_groups(groups_norm: int, num_channels: int) -> int:
    """Pick a GroupNorm group count that divides ``num_channels`` cleanly.

    Strategy: aim for at most ``groups_norm`` groups, but back off to keep
    at least 4 channels per group (so the norm has enough statistical
    support per group). If even that's not divisible, walk down until we
    find a divisor of ``num_channels``.
    """
    target = min(groups_norm, max(num_channels // 4, 1))
    if num_channels % target == 0:
        return target
    # Walk down to the largest divisor of num_channels that's <= target.
    for g in range(target, 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # unreachable for num_channels >= 1


def _gn(num_channels: int, groups_norm: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=_safe_num_groups(groups_norm, num_channels),
        num_channels=num_channels,
    )


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise + 1x1 pointwise, each followed by GroupNorm + SiLU.

    Pattern (MobileNet-style, two activations):
        DWConv(3x3, groups=in) -> GN -> SiLU -> PWConv(1x1) -> GN -> SiLU

    Args:
        in_channels: Channels into the depthwise step.
        out_channels: Channels out of the pointwise step.
        groups_norm: Target number of GroupNorm groups; clamped per channel
            count via :func:`_safe_num_groups`.

    Raises:
        ValueError: If ``in_channels < 1`` or ``out_channels < 1``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups_norm: int = 8,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError(
                f"Channel counts must be >= 1 (got in={in_channels}, "
                f"out={out_channels})"
            )

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.dw_norm = _gn(in_channels, groups_norm)
        self.dw_act = nn.SiLU(inplace=True)

        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.pw_norm = _gn(out_channels, groups_norm)
        self.pw_act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_act(self.dw_norm(self.depthwise(x)))
        x = self.pw_act(self.pw_norm(self.pointwise(x)))
        return x


class StandardConvBlock(nn.Module):
    """Plain 3x3 Conv + GroupNorm + SiLU. Used only for the stem.

    Reason for not using DepthwiseSeparableConv here: the first conv sees a
    1-channel input, so a depthwise (groups=1) collapses to a regular 3x3
    conv with a single 1x3x3 filter per output -- which is exactly what a
    regular conv would compute anyway, just with an extra useless pointwise
    afterwards.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups_norm: int = 8,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError(
                f"Channel counts must be >= 1 (got in={in_channels}, "
                f"out={out_channels})"
            )
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm = _gn(out_channels, groups_norm)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DoubleDSConv(nn.Module):
    """Two :class:`DepthwiseSeparableConv` blocks back-to-back.

    Mirrors the role of ``DoubleConv`` from ``custom_unet`` -- the conv body
    of every encoder Down and decoder Up stage in :class:`CustomLWIRUNet`.
    Channel expansion happens in the first block; the second preserves channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups_norm: int = 8,
    ) -> None:
        super().__init__()
        self.block1 = DepthwiseSeparableConv(in_channels, out_channels, groups_norm)
        self.block2 = DepthwiseSeparableConv(out_channels, out_channels, groups_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


def init_silu_weights(module: nn.Module) -> None:
    """Kaiming init for Conv / ConvTranspose layers, treating SiLU as
    leaky_relu(a=0.01).

    SiLU sits between linear and ReLU near the origin; PyTorch doesn't ship
    a dedicated nonlinearity for it, so we approximate with a small-leak
    leaky_relu. The exact value of ``a`` is not load-bearing here -- it just
    keeps the gain a touch below the pure-ReLU value.

    GroupNorm gets the standard scale=1, shift=0 init (PyTorch's default).
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.01
            )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
