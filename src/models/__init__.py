"""Model builders.

* :func:`build_unet_vgg16` -- existing-model baseline (SMP + VGG-16 encoder).
* :func:`build_custom_unet` -- hand-implemented from-scratch U-Net basis.
* :func:`build_unet_vgg16_ext` -- VGG-16 encoder + custom decoder with
  swappable AttentionGate / TransformerBottleneck seams.
"""
from __future__ import annotations

from .custom_unet import CustomUNet, build_custom_unet
from .unet import build_unet_vgg16
from .unet_vgg16_ext import VGG16UNetExt, build_unet_vgg16_ext

__all__ = [
    "build_unet_vgg16",
    "build_custom_unet",
    "CustomUNet",
    "build_unet_vgg16_ext",
    "VGG16UNetExt",
]
