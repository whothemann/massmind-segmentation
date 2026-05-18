"""From-scratch U-Net tailored for single-channel LWIR semantic segmentation.

Combines findings from the VGG16UNetExt architecture probe:

* the Transformer bottleneck helps (probe's `trans_aux` was the strongest variant);
* Attention Gates did **not** combine well with the Transformer in the probe,
  so they are omitted here;
* deep-supervision aux heads accelerate convergence at zero inference cost.

Design choices that make this model "modern" rather than a recapitulation of
the 2015 U-Net:

* **SiLU** (a.k.a. swish) instead of ReLU -- smoother gradient near zero,
  no dead units, the consensus replacement for ReLU in modern encoder
  architectures.
* **GroupNorm** instead of BatchNorm -- batch-size-independent, which
  matters for batch=1 evaluation, AMP edge cases, and downstream deployment.
* **Depthwise-separable convolutions** everywhere except the stem -- the
  stem sees 1-channel input where depthwise is degenerate. DSConv cuts the
  per-block FLOP and parameter cost dramatically without changing the
  receptive field.

Architecture (with defaults: stem_channels=32, multipliers (1,2,4,8,8)):

    [B, 1, H, W]
        |-- Stem (full res)
        |    StandardConvBlock(1 -> 32)
        |    StandardConvBlock(32 -> 32)             -> skip[0]
        |-- Encoder
        |    MaxPool2x2 + DoubleDSConv(32 -> 64)     -> skip[1] (stride 2)
        |    MaxPool2x2 + DoubleDSConv(64 -> 128)    -> skip[2] (stride 4)
        |    MaxPool2x2 + DoubleDSConv(128 -> 256)   -> skip[3] (stride 8)
        |    MaxPool2x2 + DoubleDSConv(256 -> 256)   -> bottleneck in (stride 16)
        |-- Bottleneck (no pool)
        |    TransformerBottleneck(256, layers=2, heads=8, spatial=16)
        |-- Decoder
        |    ConvT(256 -> 256) + concat skip[3] + DoubleDSConv(512 -> 256)
        |    ConvT(256 -> 128) + concat skip[2] + DoubleDSConv(256 -> 128) -> aux_deep
        |    ConvT(128 -> 64)  + concat skip[1] + DoubleDSConv(128 -> 64)  -> aux_shallow
        |    ConvT(64  -> 32)  + concat skip[0] + DoubleDSConv(64  -> 32)
        |-- Heads
             OutConv 1x1 (32 -> num_classes)               (main)
             OutConv 1x1 (128 -> num_classes) + bilinear up (aux_deep,    w=0.2)
             OutConv 1x1 (64  -> num_classes) + bilinear up (aux_shallow, w=0.4)

The aux heads only contribute in training mode -- in eval the forward returns
just the main logits, so the model is byte-identical to the no-aux variant
at inference.
"""
from __future__ import annotations

import logging
from typing import Any, Final

import torch
from torch import nn

from ._blocks import (
    DepthwiseSeparableConv,
    DoubleDSConv,
    StandardConvBlock,
    init_silu_weights,
)
from ._transformer_bottleneck import TransformerBottleneck

logger = logging.getLogger(__name__)

# Match VGG16UNetExt exactly so the loss combination in src.train._compute_loss
# applies the same weights across both architectures -- otherwise the fair
# comparison between aux-head variants is poisoned by a hidden weight delta.
AUX_HEAD_WEIGHT_SHALLOW: Final[float] = 0.4
AUX_HEAD_WEIGHT_DEEP: Final[float] = 0.2

DEFAULT_CHANNEL_MULTIPLIERS: Final[tuple[int, ...]] = (1, 2, 4, 8, 8)


class _OutConv1x1(nn.Module):
    """1x1 conv head, exposed as a named submodule so tests can introspect it.

    Same role as ``custom_unet.OutConv``, kept local to avoid import cycles
    and to leave that module untouched.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CustomLWIRUNet(nn.Module):
    """Hand-implemented U-Net for 1-channel LWIR thermal imagery.

    Args:
        num_classes: Output classes (7 for MassMIND).
        in_channels: Input channels. Must be 1 -- this model is purpose-built
            for single-channel LWIR.
        stem_channels: Channels out of the stem. Together with
            ``channel_multipliers`` this fixes the width schedule.
        channel_multipliers: Five integers; the channel counts at each level
            are ``stem_channels * multiplier``. The first multiplier governs
            the stem itself, the remaining four govern the encoder stages.
            Default ``(1, 2, 4, 8, 8)`` gives widths ``(32, 64, 128, 256, 256)``.
        transformer_layers: Number of ``nn.TransformerEncoderLayer`` blocks
            in the bottleneck.
        transformer_heads: Attention heads in the bottleneck. The deepest
            channel width must be divisible by this.
        use_aux_heads: If True, attach two 1x1 deep-supervision heads to the
            decoder mid-levels. In training mode the forward returns a tuple
            ``(main, aux_shallow, aux_deep)``; in eval mode it returns just
            ``main``.
        groups_norm: Target number of GroupNorm groups. Per-block clamped
            via :func:`._blocks._safe_num_groups`.
        transformer_config: Extra kwargs forwarded to ``TransformerBottleneck``
            (e.g. ``mlp_ratio``, ``spatial_size``). Sensible defaults for
            256 channels at 16x16 are applied.

    Raises:
        ValueError: If ``in_channels != 1``, ``channel_multipliers`` has the
            wrong length, or the deepest width isn't divisible by
            ``transformer_heads``.

    Forward:
        Input ``[B, 1, H, W]`` (H, W divisible by 16).

        Output:
            * If ``use_aux_heads=False`` or the model is in ``eval()`` mode:
              a single ``[B, num_classes, H, W]`` tensor.
            * Else (``training`` and ``use_aux_heads``): a tuple
              ``(main, aux_shallow, aux_deep)`` of three same-shape tensors.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        stem_channels: int = 32,
        channel_multipliers: tuple[int, ...] = DEFAULT_CHANNEL_MULTIPLIERS,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        use_aux_heads: bool = True,
        groups_norm: int = 8,
        transformer_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                f"CustomLWIRUNet is single-channel only (got in_channels={in_channels})"
            )
        if len(channel_multipliers) != 5:
            raise ValueError(
                "channel_multipliers must have 5 entries "
                f"(stem + 4 encoder stages); got {len(channel_multipliers)}"
            )
        if stem_channels < 1:
            raise ValueError(f"stem_channels must be >= 1 (got {stem_channels})")

        widths = tuple(stem_channels * m for m in channel_multipliers)
        deepest = widths[-1]
        if deepest % transformer_heads != 0:
            raise ValueError(
                f"deepest width {deepest} must be divisible by "
                f"transformer_heads={transformer_heads}"
            )

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.stem_channels = int(stem_channels)
        self.channel_multipliers = tuple(channel_multipliers)
        self.widths = widths
        self.use_aux_heads = bool(use_aux_heads)
        self.groups_norm = int(groups_norm)

        # --- Stem (full resolution; no pool). Two StandardConvBlocks: first
        # lifts 1 -> stem_channels, second deepens the stem at the same width.
        # Depthwise is degenerate at 1 input channel, hence standard convs here.
        self.stem = nn.Sequential(
            StandardConvBlock(in_channels, widths[0], groups_norm),
            StandardConvBlock(widths[0], widths[0], groups_norm),
        )

        # --- Encoder: 4 stages, each MaxPool(2) + DoubleDSConv. The pool
        # is the only thing that changes resolution; the DSConv expands
        # channels per the multiplier schedule.
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(4)]
        )
        self.encoder_blocks = nn.ModuleList()
        for i in range(4):
            self.encoder_blocks.append(
                DoubleDSConv(widths[i], widths[i + 1], groups_norm)
            )

        # --- Bottleneck (no extra pool). Spatial size at 256 input is
        # H/16 = 16, so the transformer sees 16*16 = 256 tokens -- the right
        # ballpark for global self-attention.
        tcfg: dict[str, Any] = dict(transformer_config or {})
        tcfg.setdefault("mlp_ratio", 2)
        tcfg.setdefault("spatial_size", 16)
        self.bottleneck = TransformerBottleneck(
            channels=deepest,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            **tcfg,
        )

        # --- Decoder: 4 stages, each ConvT(in -> skip_ch) + concat(skip)
        # + DoubleDSConv(2*skip_ch -> skip_ch). Skips are consumed deep -> shallow.
        # No attention gates (probe finding: AG doesn't combine with Transformer).
        skip_channels = [widths[3], widths[2], widths[1], widths[0]]  # deep -> shallow
        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        up_in = deepest  # 256 by default
        for skip_c in skip_channels:
            self.up_convs.append(
                nn.ConvTranspose2d(up_in, skip_c, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                DoubleDSConv(skip_c * 2, skip_c, groups_norm)
            )
            up_in = skip_c

        # --- Heads.
        # Main head: 1x1 at full input resolution (decoder ends at stem width).
        self.out_conv = _OutConv1x1(widths[0], num_classes)
        # Aux heads attach to decoder stage 2 (deeper, w=0.2) and stage 3
        # (shallower, w=0.4), matching the VGG16UNetExt convention. They are
        # bilinear-upsampled to the input resolution inside forward() so all
        # three heads share one mask in the loss.
        if use_aux_heads:
            self.aux_head_deep = _OutConv1x1(widths[2], num_classes)     # 128ch
            self.aux_head_shallow = _OutConv1x1(widths[1], num_classes)  # 64ch
        else:
            self.aux_head_deep = None
            self.aux_head_shallow = None

        # Kaiming init treating SiLU as leaky_relu(a=0.01). GroupNorm keeps
        # PyTorch's default (scale=1, shift=0).
        init_silu_weights(self)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 4 or x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected input [B, {self.in_channels}, H, W], got {tuple(x.shape)}"
            )

        # Stem keeps spatial dims; saved as the shallowest skip.
        s0 = self.stem(x)
        skips: list[torch.Tensor] = [s0]

        # Encoder: pool then conv. Each stage doubles stride.
        feat = s0
        for pool, block in zip(self.pools, self.encoder_blocks, strict=True):
            feat = block(pool(feat))
            skips.append(feat)
        # skips: [s0 (str 1), s1 (str 2), s2 (str 4), s3 (str 8), s4 (str 16)]

        # Bottleneck operates on the deepest skip (s4).
        out = self.bottleneck(skips[-1])

        # Decoder: deep -> shallow, consuming skips[3..0].
        decoder_outs: list[torch.Tensor] = []
        for up, block, skip in zip(
            self.up_convs,
            self.decoder_blocks,
            (skips[3], skips[2], skips[1], skips[0]),
            strict=True,
        ):
            out = up(out)
            # Pad in case of odd input sizes (defensive; with H,W div by 16
            # this is a no-op).
            if out.shape[-2:] != skip.shape[-2:]:
                dh = skip.size(-2) - out.size(-2)
                dw = skip.size(-1) - out.size(-1)
                out = nn.functional.pad(
                    out, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
                )
            out = block(torch.cat([skip, out], dim=1))
            decoder_outs.append(out)
        # decoder_outs[0]=stage1 (str 8), [1]=stage2 (str 4), [2]=stage3 (str 2), [3]=stage4 (str 1)

        main_logits = self.out_conv(decoder_outs[-1])

        if (
            self.use_aux_heads
            and self.training
            and self.aux_head_deep is not None
            and self.aux_head_shallow is not None
        ):
            target_size = main_logits.shape[-2:]
            aux_deep = self.aux_head_deep(decoder_outs[1])     # weight 0.2
            aux_shallow = self.aux_head_shallow(decoder_outs[2])  # weight 0.4
            aux_deep = nn.functional.interpolate(
                aux_deep, size=target_size, mode="bilinear", align_corners=False,
            )
            aux_shallow = nn.functional.interpolate(
                aux_shallow, size=target_size, mode="bilinear", align_corners=False,
            )
            return main_logits, aux_shallow, aux_deep

        return main_logits


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_custom_lwir_unet(
    num_classes: int = 7,
    in_channels: int = 1,
    stem_channels: int = 32,
    channel_multipliers: tuple[int, ...] = DEFAULT_CHANNEL_MULTIPLIERS,
    transformer_layers: int = 2,
    transformer_heads: int = 8,
    use_aux_heads: bool = True,
    groups_norm: int = 8,
    transformer_config: dict[str, Any] | None = None,
) -> CustomLWIRUNet:
    """Construct a :class:`CustomLWIRUNet` and log a parameter-count breakdown.

    Same args as :class:`CustomLWIRUNet`. Returned model is on CPU; move it
    to a device after construction.
    """
    model = CustomLWIRUNet(
        num_classes=num_classes,
        in_channels=in_channels,
        stem_channels=stem_channels,
        channel_multipliers=channel_multipliers,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        use_aux_heads=use_aux_heads,
        groups_norm=groups_norm,
        transformer_config=transformer_config,
    )

    p_stem = _count_params(model.stem)
    p_enc = sum(_count_params(b) for b in model.encoder_blocks)
    p_bot = _count_params(model.bottleneck)
    p_dec = sum(_count_params(b) for b in model.decoder_blocks) + sum(
        _count_params(u) for u in model.up_convs
    )
    p_heads = _count_params(model.out_conv) + (
        (_count_params(model.aux_head_deep) if model.aux_head_deep else 0)
        + (_count_params(model.aux_head_shallow) if model.aux_head_shallow else 0)
    )
    p_total = sum(p.numel() for p in model.parameters())
    p_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        "Built CustomLWIRUNet: in=%d, classes=%d, widths=%s, "
        "trans_layers=%d, heads=%d, aux=%s, total=%.2fM (trainable=%.2fM)",
        in_channels,
        num_classes,
        model.widths,
        transformer_layers,
        transformer_heads,
        use_aux_heads,
        p_total / 1e6,
        p_trainable / 1e6,
    )
    logger.info(
        "  breakdown: stem=%.3fM enc=%.3fM bottleneck=%.3fM dec=%.3fM heads=%.3fM",
        p_stem / 1e6,
        p_enc / 1e6,
        p_bot / 1e6,
        p_dec / 1e6,
        p_heads / 1e6,
    )
    return model
