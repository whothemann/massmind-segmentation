"""Tests for the model builders and first-layer adaptation."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models import build_unet_vgg16
from src.models._adapt import adapt_conv_to_one_channel, replace_module


class TestAdaptConvToOneChannel:
    def test_weight_is_channel_mean(self) -> None:
        conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=True)
        # Deterministic weight: each input channel is a distinct constant per
        # output channel, so the mean across channels has a known value.
        with torch.no_grad():
            for c in range(3):
                conv.weight[:, c, :, :] = float(c + 1)  # channels: 1, 2, 3
        new_conv = adapt_conv_to_one_channel(conv)
        assert new_conv.in_channels == 1
        assert new_conv.out_channels == 8
        # Expected weight: mean(1, 2, 3) = 2 everywhere.
        torch.testing.assert_close(
            new_conv.weight, torch.full_like(new_conv.weight, 2.0)
        )

    def test_bias_preserved(self) -> None:
        conv = nn.Conv2d(3, 4, 3, bias=True)
        original_bias = conv.bias.detach().clone()
        new_conv = adapt_conv_to_one_channel(conv)
        torch.testing.assert_close(new_conv.bias, original_bias)

    def test_no_bias(self) -> None:
        conv = nn.Conv2d(3, 4, 3, bias=False)
        new_conv = adapt_conv_to_one_channel(conv)
        assert new_conv.bias is None

    def test_geometry_preserved(self) -> None:
        conv = nn.Conv2d(3, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1), dilation=(1, 2))
        new_conv = adapt_conv_to_one_channel(conv)
        assert new_conv.kernel_size == (5, 3)
        assert new_conv.stride == (2, 1)
        assert new_conv.padding == (2, 1)
        assert new_conv.dilation == (1, 2)

    def test_rejects_non_rgb(self) -> None:
        with pytest.raises(ValueError):
            adapt_conv_to_one_channel(nn.Conv2d(4, 8, 3))

    def test_rejects_grouped(self) -> None:
        with pytest.raises(ValueError):
            adapt_conv_to_one_channel(nn.Conv2d(3, 6, 3, groups=3))


class TestReplaceModule:
    def test_replace_via_indexed_path(self) -> None:
        seq = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
        replace_module(seq, "0", nn.Linear(3, 3))
        assert seq[0].in_features == 3

    def test_replace_via_attribute_path(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.head = nn.Linear(2, 1)

        m = M()
        replace_module(m, "head", nn.Linear(4, 1))
        assert m.head.in_features == 4


class TestUnetBuilder:
    def test_single_channel_forward(self) -> None:
        model = build_unet_vgg16(num_classes=7, in_channels=1, encoder_weights=None)
        model.eval()
        x = torch.randn(2, 1, 64, 96)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 7, 64, 96)

    def test_three_channel_forward_unchanged(self) -> None:
        # When in_channels=3 we should NOT touch the first conv.
        model = build_unet_vgg16(num_classes=7, in_channels=3, encoder_weights=None)
        first_conv = model.encoder.features[0]
        assert isinstance(first_conv, nn.Conv2d)
        assert first_conv.in_channels == 3

    def test_invalid_in_channels(self) -> None:
        with pytest.raises(ValueError):
            build_unet_vgg16(in_channels=2, encoder_weights=None)
