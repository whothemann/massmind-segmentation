"""U-Net + VGG-16 segmentation model for MassMIND.

Built on top of ``segmentation_models_pytorch`` (SMP). We deliberately
construct the model with ``in_channels=3`` and ImageNet weights, then
*surgically* replace the first conv with a channel-mean-initialised
single-channel kernel via :func:`src.models._adapt.adapt_conv_to_one_channel`.

Why not pass ``in_channels=1`` to SMP directly? SMP does its own first-layer
patching when ``in_channels != 3``, and the strategy is not always
channel-mean (it can be sum-based or partly random depending on version).
Doing it ourselves keeps the initialisation transparent and unit-testable.
"""
from __future__ import annotations

import logging
from typing import Final

import segmentation_models_pytorch as smp
from torch import nn

from ._adapt import adapt_conv_to_one_channel, replace_module

logger = logging.getLogger(__name__)

# Where the first Conv2d lives inside SMP's vgg16 encoder.
_VGG16_FIRST_CONV_PATH: Final[str] = "encoder.features.0"


def build_unet_vgg16(
    num_classes: int = 7,
    in_channels: int = 1,
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Build a U-Net with a VGG-16 encoder, adapted for single-channel input.

    Args:
        num_classes: Number of output classes (7 for MassMIND).
        in_channels: Number of input channels. If ``1`` and ``encoder_weights``
            is set, the first conv is reinitialised by channel-averaging the
            pretrained RGB kernel. If ``3``, the model is returned untouched.
        encoder_weights: Either ``"imagenet"`` (default) or ``None`` for
            random initialisation.

    Returns:
        An ``nn.Module`` whose forward expects ``[B, in_channels, H, W]`` and
        returns logits of shape ``[B, num_classes, H, W]``.

    Raises:
        ValueError: If ``in_channels`` is neither 1 nor 3.
    """
    if in_channels not in (1, 3):
        raise ValueError(f"in_channels must be 1 or 3 (got {in_channels})")

    model = smp.Unet(
        encoder_name="vgg16",
        encoder_weights=encoder_weights,
        in_channels=3,           # build with RGB first so weights load cleanly
        classes=num_classes,
    )

    if in_channels == 1:
        first_conv = _get_first_conv(model)
        if encoder_weights is None:
            # Random init: channel-mean adaptation is meaningless (it would
            # just average noise into more noise) AND it discards the SMP
            # encoder's default random init scale. Replace with a fresh
            # 1-channel Conv2d of the same shape so random init runs on the
            # actual single-channel kernel.
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                dilation=first_conv.dilation,
                bias=first_conv.bias is not None,
            )
            logger.info(
                "Replaced first conv with fresh 1-channel init (no pretrained "
                "weights to adapt)."
            )
        else:
            new_conv = adapt_conv_to_one_channel(first_conv)
        replace_module(model, _VGG16_FIRST_CONV_PATH, new_conv)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Built U-Net+VGG16: in_channels=%d, classes=%d, params=%.2fM, "
        "encoder_weights=%s",
        in_channels,
        num_classes,
        n_params / 1e6,
        encoder_weights,
    )
    return model


def _get_first_conv(model: nn.Module) -> nn.Conv2d:
    """Resolve the dotted attribute path to the encoder's first Conv2d."""
    target: nn.Module = model
    for part in _VGG16_FIRST_CONV_PATH.split("."):
        target = target[int(part)] if part.isdigit() else getattr(target, part)
    if not isinstance(target, nn.Conv2d):
        raise RuntimeError(
            f"Expected nn.Conv2d at {_VGG16_FIRST_CONV_PATH}, got {type(target).__name__}"
        )
    return target
