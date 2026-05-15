"""Hand-implemented U-Net for MassMIND.

This module deliberately avoids ``segmentation_models_pytorch`` and any
prebuilt architecture: every layer is constructed with ``torch.nn``
primitives so the structure is fully visible. The architecture follows
Ronneberger et al. (2015) -- 4 down/up levels, double-conv blocks,
max-pool downsampling, transposed-conv upsampling, concatenative skip
connections, a 1x1 output head.

The decomposition into ``DoubleConv``, ``Down``, ``Up``, ``OutConv`` and a
standalone ``Bottleneck`` is intentional. Two upcoming modifications are
designed to drop in at fixed seams:

* **Attention gates** replace the ``nn.Identity`` skip-refiner held on each
  ``Up`` block. The forward signature is preserved: ``(skip, gating) -> skip'``.
* **Transformer bottleneck** replaces the ``self.bottleneck`` submodule.
  Any module with signature ``(B, C, H, W) -> (B, C, H, W)`` fits.

Channel progression at ``base_channels=64``::

    1  -> 64 -> 128 -> 256 -> 512  (encoder; 4 down levels)
                              \
                               1024  (bottleneck)
                              /
       64 <- 128 <- 256 <- 512    (decoder)
       |
       1x1 conv -> num_classes
"""
from __future__ import annotations

import logging
from typing import Final

import torch
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_BASE_CHANNELS: Final[int] = 64
DEFAULT_DEPTH: Final[int] = 4


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d (stride 2) -> DoubleConv. Halves spatial dims."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """ConvTranspose2d -> concat(skip) -> DoubleConv.

    The skip path goes through ``self.skip_refine``, which defaults to
    ``nn.Identity``. Step 2 of the plan replaces it with an Attention Gate
    that takes both the encoder skip and the upsampled decoder feature
    (the "gating signal") and returns a re-weighted skip.

    Args:
        in_channels: channels coming up from the deeper decoder/bottleneck.
        skip_channels: channels of the matching encoder skip connection.
        out_channels: channels produced by the post-concat double conv.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, skip_channels, kernel_size=2, stride=2
        )
        # Placeholder skip refiner. Identity means "raw skip" -- the classic
        # U-Net behaviour. Step 2 swaps this for an AttentionGate.
        self.skip_refine: nn.Module = nn.Identity()
        self.conv = DoubleConv(skip_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd input dims: pad the upsampled feature to the skip size
        # before concat. (Demonstrator U-Nets crop; padding is safer for
        # arbitrary inputs without losing border pixels.)
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.size(-2) - x.size(-2)
            dw = skip.size(-1) - x.size(-1)
            x = nn.functional.pad(
                x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
            )

        # Identity by default; attention gate later. AttentionGate will accept
        # (skip, gating=x); Identity accepts only one arg, so dispatch here.
        if isinstance(self.skip_refine, nn.Identity):
            skip = self.skip_refine(skip)
        else:
            skip = self.skip_refine(skip, x)

        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    """1x1 conv to ``num_classes`` channels."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Bottleneck(nn.Module):
    """Bottleneck block sitting at the bottom of the U.

    Pools the deepest encoder feature once more before doubling channels.
    Pooling inside the bottleneck (rather than in the last encoder ``Down``)
    keeps the deepest encoder feature available as a skip into the first
    decoder ``Up`` block, matching Ronneberger et al. (2015) exactly.

    For the basis U-Net the body is just ``DoubleConv``. Step 3 of the plan
    swaps the body for a small Transformer stack; the pool stays.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class CustomUNet(nn.Module):
    """Hand-implemented 4-level U-Net.

    Args:
        num_classes: Number of output classes. 7 for MassMIND.
        in_channels: Input channels. 1 for LWIR thermal.
        base_channels: Channel count at the first encoder level. Doubles per
            downsample. With ``base_channels=64`` and ``depth=4`` this produces
            a ~31 M-parameter network -- the canonical Ronneberger U-Net size.
        depth: Number of downsampling/upsampling levels.

    Forward:
        Input  ``[B, in_channels, H, W]`` -> Output logits
        ``[B, num_classes, H, W]``. ``H`` and ``W`` should be divisible by
        ``2 ** depth`` for the encoder skips to align with the decoder
        upsamples (the forward pads on size mismatch as a safety net).
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1 (got {depth})")
        if base_channels < 1:
            raise ValueError(f"base_channels must be >= 1 (got {base_channels})")
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.depth = int(depth)

        # Encoder: first block keeps spatial size; each subsequent Down halves it.
        # Channel sequence: base, 2*base, 4*base, ..., (2**(depth-1))*base.
        enc_channels = [base_channels * (2**i) for i in range(depth)]
        self.in_conv = DoubleConv(in_channels, enc_channels[0])
        self.downs = nn.ModuleList(
            Down(enc_channels[i], enc_channels[i + 1]) for i in range(depth - 1)
        )

        # Bottleneck doubles channels one more time.
        bottleneck_channels = enc_channels[-1] * 2
        self.bottleneck = Bottleneck(enc_channels[-1], bottleneck_channels)

        # Decoder: mirror of the encoder. At each level we have:
        #   in_channels  = channels coming up (bottleneck or previous decoder)
        #   skip_channels = matching encoder channels
        #   out_channels = same as skip_channels (so each Up halves channels)
        self.ups = nn.ModuleList()
        up_in = bottleneck_channels
        for i in reversed(range(depth)):
            skip_c = enc_channels[i]
            self.ups.append(
                Up(in_channels=up_in, skip_channels=skip_c, out_channels=skip_c)
            )
            up_in = skip_c

        self.out_conv = OutConv(enc_channels[0], num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming for convs, ones/zeros for BN. Matches U-Net references."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: collect skips as we go down. The bottleneck pools internally,
        # so every encoder feature -- including the deepest -- remains
        # available as a skip into a matching Up block.
        skips: list[torch.Tensor] = []
        x = self.in_conv(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        for up in self.ups:
            x = up(x, skips.pop())

        return self.out_conv(x)


def build_custom_unet(
    num_classes: int = 7,
    in_channels: int = 1,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    depth: int = DEFAULT_DEPTH,
) -> CustomUNet:
    """Construct a from-scratch U-Net and log its parameter count.

    Same args as :class:`CustomUNet`. Returned model is on CPU; move it to
    a device after construction.
    """
    model = CustomUNet(
        num_classes=num_classes,
        in_channels=in_channels,
        base_channels=base_channels,
        depth=depth,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Built CustomUNet: in_channels=%d, classes=%d, base=%d, depth=%d, "
        "params=%.2fM (trainable=%.2fM)",
        in_channels,
        num_classes,
        base_channels,
        depth,
        n_params / 1e6,
        n_trainable / 1e6,
    )
    return model
