"""First-layer adaptation: 3-channel ImageNet-pretrained Conv2d -> single-channel.

We use **channel-mean** initialisation for the new single-channel kernel: the
new weight for output channel ``o`` is the mean across the three input
channels of the pretrained kernel. This preserves activation magnitude when
the input becomes a single channel, unlike summing (which inflates) or
re-initialising (which throws away the pretrained features).

The same helper is reused for SegFormer in a later phase -- the only thing
that differs is which Conv2d to surgically replace.
"""
from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def adapt_conv_to_one_channel(conv: nn.Conv2d) -> nn.Conv2d:
    """Return a new Conv2d with ``in_channels=1`` whose weight is the mean of
    ``conv``'s pretrained 3-channel weights along the input-channel axis.

    Args:
        conv: An ``nn.Conv2d`` whose ``in_channels`` is 3 (typical for an
            ImageNet-pretrained first layer).

    Returns:
        A new ``nn.Conv2d`` with the same out_channels, kernel_size, stride,
        padding, dilation, groups and bias-presence, but ``in_channels=1`` and
        weight initialised to the channel-wise mean of the input.

    Raises:
        ValueError: If ``conv.in_channels != 3`` or ``conv.groups != 1``.
    """
    if conv.in_channels != 3:
        raise ValueError(
            f"Expected in_channels=3 (RGB pretrained), got {conv.in_channels}"
        )
    if conv.groups != 1:
        raise ValueError(
            f"Grouped convolutions are not supported (got groups={conv.groups})"
        )

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        # conv.weight shape: [out, 3, kH, kW]; collapse axis=1 by mean.
        new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    logger.info(
        "Adapted first conv: in=%d -> 1, out=%d, kernel=%s (channel-mean init)",
        conv.in_channels,
        conv.out_channels,
        tuple(conv.kernel_size),
    )
    return new_conv


def replace_module(parent: nn.Module, attr_path: str, new_module: nn.Module) -> None:
    """Replace a nested submodule by dotted attribute path.

    For example, ``replace_module(model, "encoder.features.0", new_conv)`` walks
    ``model.encoder.features``, then sets index ``0`` to ``new_conv``.

    Supports both attribute access (``foo.bar``) and integer index access
    (``foo.0``) inside ``nn.Sequential`` / ``nn.ModuleList``.

    Args:
        parent: The model to mutate in place.
        attr_path: Dotted path to the submodule, e.g. ``"encoder.features.0"``.
        new_module: The replacement.

    Raises:
        AttributeError: If any segment of ``attr_path`` cannot be resolved.
    """
    parts = attr_path.split(".")
    target = parent
    for part in parts[:-1]:
        target = _get_child(target, part)
    last = parts[-1]
    if last.isdigit():
        target[int(last)] = new_module  # type: ignore[index]
    else:
        setattr(target, last, new_module)


def _get_child(module: nn.Module, name: str) -> nn.Module:
    if name.isdigit():
        return module[int(name)]  # type: ignore[index]
    return getattr(module, name)
