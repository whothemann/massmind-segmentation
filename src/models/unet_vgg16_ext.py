"""VGG16 U-Net extension: pretrained encoder + custom-decoder swappable seams.

Combines:

* the ImageNet-pretrained VGG16 encoder from ``segmentation_models_pytorch``
  (the same one our 0.80-mIoU baseline uses), with our channel-mean
  adaptation from RGB to single-channel LWIR;
* a custom decoder built from the same ``DoubleConv`` / ``Up`` / ``OutConv``
  modules as ``src.models.custom_unet``, exposing the same two seams:
    1. ``Up.skip_refine`` (Identity / AttentionGate)
    2. ``Bottleneck.body``  (DoubleConv / TransformerBottleneck).

VGG16-specific structural choice: the bottleneck **omits** the MaxPool of
the from-scratch ``Bottleneck``. VGG16 already pools 5 times in the encoder;
at 256x256 input the deepest feature is 8x8, and an extra pool drops that
to 4x4 (16 tokens) -- too coarse to feed the transformer. The bottleneck
here is therefore body-only.

SMP VGG16 encoder (depth=5) emits 6 feature maps; this code uses indices
``[1..4]`` as decoder skips and ``[5]`` as the bottleneck input. A final
stride-2 -> stride-1 stage upsamples back to input resolution before the
1x1 head, since the shallowest skip is at stride 2.
"""
from __future__ import annotations

import logging
from typing import Any, Final

import segmentation_models_pytorch as smp
import torch
from torch import nn

from ._adapt import adapt_conv_to_one_channel, replace_module
from ._attention_gate import AttentionGate
from ._transformer_bottleneck import TransformerBottleneck
from .custom_unet import DoubleConv, OutConv, Up

logger = logging.getLogger(__name__)

# SMP VGG16 encoder (depth=5) emits 6 features at strides [1, 2, 4, 8, 16, 32]
# with the following channel counts. NB: ``segmentation_models_pytorch`` v0.5
# returns ``[64, 128, 256, 512, 512, 512]`` (the first feature is already a
# post-stem-conv 64-channel map, NOT the raw input).
_VGG16_ENCODER_CHANNELS: Final[tuple[int, ...]] = (64, 128, 256, 512, 512, 512)

# Path to the first Conv2d inside the bare SMP encoder (not wrapped in Unet).
_VGG16_FIRST_CONV_PATH: Final[str] = "features.0"


class VGG16Bottleneck(nn.Module):
    """Bottleneck wrapper for the VGG16 extension.

    Mirrors ``src.models.custom_unet.Bottleneck`` in design (a single
    swappable body) but **omits the MaxPool**: VGG16 has already pooled 32x
    by this point. The body is the actual computation -- either a
    ``DoubleConv`` (default) or a ``TransformerBottleneck``.

    Attribute ``body`` is the public seam tests inspect to confirm the
    correct variant is wired up.
    """

    def __init__(self, channels: int, body: nn.Module) -> None:
        super().__init__()
        self.channels = int(channels)
        self.body = body

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class VGG16UNetExt(nn.Module):
    """SMP VGG16 encoder + custom decoder with AttGate / Transformer seams.

    Args:
        num_classes: Output classes (7 for MassMIND).
        in_channels: 1 (LWIR; first conv is channel-mean-adapted from the
            pretrained 3-channel weights) or 3.
        encoder_weights: ``"imagenet"`` for pretrained weights (default),
            ``None`` for random init.
        use_transformer_bottleneck: If True, the bottleneck body is a
            ``TransformerBottleneck`` instead of a ``DoubleConv``.
        use_attention_gates: If True, every ``Up`` block's ``skip_refine``
            slot is filled with an ``AttentionGate`` (otherwise
            ``nn.Identity``).
        transformer_config: Extra kwargs forwarded to
            ``TransformerBottleneck`` (``num_heads``, ``num_layers``,
            ``mlp_ratio``, ``spatial_size``). Unused when
            ``use_transformer_bottleneck=False``.

    Raises:
        ValueError: If ``in_channels`` is neither 1 nor 3.

    Forward:
        Input  ``[B, in_channels, H, W]`` (H, W divisible by 32).
        Output ``[B, num_classes, H, W]``.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 1,
        encoder_weights: str | None = "imagenet",
        use_transformer_bottleneck: bool = False,
        use_attention_gates: bool = False,
        transformer_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if in_channels not in (1, 3):
            raise ValueError(f"in_channels must be 1 or 3 (got {in_channels})")

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.use_attention_gates = bool(use_attention_gates)
        self.use_transformer_bottleneck = bool(use_transformer_bottleneck)

        # --- Encoder: SMP factory, built as 3-channel for clean weight loading.
        self.encoder = smp.encoders.get_encoder(
            "vgg16",
            in_channels=3,
            depth=5,
            weights=encoder_weights,
        )
        if in_channels == 1:
            first_conv = _get_module(self.encoder, _VGG16_FIRST_CONV_PATH)
            if not isinstance(first_conv, nn.Conv2d):
                raise RuntimeError(
                    f"Expected nn.Conv2d at {_VGG16_FIRST_CONV_PATH}, "
                    f"got {type(first_conv).__name__}"
                )
            replace_module(
                self.encoder,
                _VGG16_FIRST_CONV_PATH,
                adapt_conv_to_one_channel(first_conv),
            )

        # --- Bottleneck. Channels stay at 512 (VGG16's deepest) on both paths;
        # neither body changes channel count. This is intentional -- the
        # TransformerBottleneck is shape-preserving by construction, so the
        # DoubleConv variant matches it for ablation parity.
        bottleneck_channels = _VGG16_ENCODER_CHANNELS[5]  # 512
        if use_transformer_bottleneck:
            tcfg: dict[str, Any] = dict(transformer_config or {})
            tcfg.setdefault("num_heads", 8)
            tcfg.setdefault("num_layers", 2)
            tcfg.setdefault("mlp_ratio", 2)
            tcfg.setdefault("spatial_size", 8)
            body: nn.Module = TransformerBottleneck(
                channels=bottleneck_channels, **tcfg
            )
        else:
            body = DoubleConv(bottleneck_channels, bottleneck_channels)
        self.bottleneck = VGG16Bottleneck(channels=bottleneck_channels, body=body)

        # --- Decoder. 4 Up blocks consuming features[4..1] (deep -> shallow).
        # in_channels of each Up is the previous out (or bottleneck), skip is
        # the matching encoder feature, out is the encoder's same-level
        # channel count -- matching the "out_channels = skip_channels"
        # convention from custom_unet.Up.
        skip_channels = [
            _VGG16_ENCODER_CHANNELS[4],  # 512  (stride 16)
            _VGG16_ENCODER_CHANNELS[3],  # 512  (stride 8)
            _VGG16_ENCODER_CHANNELS[2],  # 256  (stride 4)
            _VGG16_ENCODER_CHANNELS[1],  # 128  (stride 2)
        ]
        self.ups = nn.ModuleList()
        up_in = bottleneck_channels
        for skip_c in skip_channels:
            self.ups.append(
                Up(in_channels=up_in, skip_channels=skip_c, out_channels=skip_c)
            )
            up_in = skip_c

        # --- Attention gates on each Up's skip_refine slot, if enabled.
        # After Up.up (the ConvTranspose), the gating feature has skip_c
        # channels -- that's how custom_unet.Up was designed -- so
        # gating_channels == skip_channels for the gate.
        if use_attention_gates:
            for up, skip_c in zip(self.ups, skip_channels, strict=True):
                up.skip_refine = AttentionGate(
                    skip_channels=skip_c, gating_channels=skip_c
                )

        # --- Final stride-2 -> stride-1 stage. The shallowest skip is at
        # stride 2 (feats[1]), so after 4 Ups we land at half-resolution;
        # this stage brings us back to the input resolution before the head.
        final_in = skip_channels[-1]                # 128
        final_mid = max(final_in // 2, 16)          # 64
        self.final_up = nn.ConvTranspose2d(
            final_in, final_mid, kernel_size=2, stride=2
        )
        self.final_conv = DoubleConv(final_mid, final_mid)
        self.out_conv = OutConv(final_mid, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SMP encoder returns 6 feature maps at strides [1, 2, 4, 8, 16, 32].
        feats = self.encoder(x)
        if len(feats) != 6:
            raise RuntimeError(
                f"Expected 6 VGG16 encoder features, got {len(feats)}. "
                "API mismatch -- update _VGG16_ENCODER_CHANNELS."
            )

        out = self.bottleneck(feats[5])
        # Decoder consumes skips deep -> shallow: feats[4], [3], [2], [1].
        for up, skip in zip(
            self.ups, (feats[4], feats[3], feats[2], feats[1]), strict=True
        ):
            out = up(out, skip)
        out = self.final_up(out)
        out = self.final_conv(out)
        return self.out_conv(out)


def _get_module(root: nn.Module, path: str) -> nn.Module:
    """Walk a dotted path (with integer-string segments) to a submodule."""
    target: nn.Module = root
    for part in path.split("."):
        target = target[int(part)] if part.isdigit() else getattr(target, part)
    return target


def build_unet_vgg16_ext(
    num_classes: int = 7,
    in_channels: int = 1,
    encoder_weights: str | None = "imagenet",
    use_transformer_bottleneck: bool = False,
    use_attention_gates: bool = False,
    transformer_config: dict[str, Any] | None = None,
) -> VGG16UNetExt:
    """Construct a :class:`VGG16UNetExt` and log its parameter count.

    Same args as :class:`VGG16UNetExt`. Returned model is on CPU; move it to
    a device after construction.
    """
    model = VGG16UNetExt(
        num_classes=num_classes,
        in_channels=in_channels,
        encoder_weights=encoder_weights,
        use_transformer_bottleneck=use_transformer_bottleneck,
        use_attention_gates=use_attention_gates,
        transformer_config=transformer_config,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Built VGG16UNetExt: in_channels=%d, classes=%d, weights=%s, "
        "trans_bottleneck=%s, att_gates=%s, params=%.2fM (trainable=%.2fM)",
        in_channels,
        num_classes,
        encoder_weights,
        use_transformer_bottleneck,
        use_attention_gates,
        n_params / 1e6,
        n_trainable / 1e6,
    )
    return model
