"""Tests for the model builders and first-layer adaptation."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models import CustomUNet, build_custom_unet, build_unet_vgg16
from src.models._adapt import adapt_conv_to_one_channel, replace_module
from src.models.custom_unet import Bottleneck, DoubleConv, Down, Up


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


class TestCustomUNetBlocks:
    def test_double_conv_preserves_spatial(self) -> None:
        block = DoubleConv(8, 16)
        out = block(torch.randn(2, 8, 32, 32))
        assert out.shape == (2, 16, 32, 32)

    def test_down_halves_spatial(self) -> None:
        block = Down(16, 32)
        out = block(torch.randn(2, 16, 32, 32))
        assert out.shape == (2, 32, 16, 16)

    def test_up_concats_and_doubles_spatial(self) -> None:
        # up: 32 -> 16 channels, halved -> doubled spatial; skip provides 16.
        block = Up(in_channels=32, skip_channels=16, out_channels=16)
        x = torch.randn(2, 32, 8, 8)
        skip = torch.randn(2, 16, 16, 16)
        out = block(x, skip)
        assert out.shape == (2, 16, 16, 16)

    def test_up_handles_odd_size_mismatch(self) -> None:
        # Decoder upsamples 9 -> 18, but skip from an odd encoder is 17.
        # The Up block must pad rather than crop.
        block = Up(in_channels=32, skip_channels=16, out_channels=16)
        x = torch.randn(1, 32, 9, 9)
        skip = torch.randn(1, 16, 17, 17)
        out = block(x, skip)
        assert out.shape == (1, 16, 17, 17)

    def test_bottleneck_halves_spatial_and_doubles_channels(self) -> None:
        # Bottleneck pools once (preserving the deepest encoder feature as a
        # skip into the first Up) then runs a double conv.
        block = Bottleneck(64, 128)
        out = block(torch.randn(2, 64, 8, 8))
        assert out.shape == (2, 128, 4, 4)


class TestCustomUNet:
    def test_forward_shape_matches_input(self) -> None:
        model = build_custom_unet(num_classes=7, in_channels=1, base_channels=16)
        model.eval()
        x = torch.randn(2, 1, 64, 96)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 7, 64, 96)

    def test_forward_three_channel_input(self) -> None:
        model = build_custom_unet(num_classes=4, in_channels=3, base_channels=8)
        x = torch.randn(1, 3, 32, 48)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 4, 32, 48)

    def test_skip_refiner_default_is_identity(self) -> None:
        # The seam where step 2 (attention gates) will plug in must start as
        # nn.Identity -- otherwise downstream behaviour silently changes.
        model = build_custom_unet(base_channels=8)
        for up in model.ups:
            assert isinstance(up.skip_refine, nn.Identity)

    def test_bottleneck_seam_is_bottleneck_module(self) -> None:
        # Step 3 (transformer bottleneck) swaps model.bottleneck. Verify it
        # exists as a single module rather than being inlined into forward.
        model = build_custom_unet(base_channels=8)
        assert isinstance(model.bottleneck, Bottleneck)

    def test_handles_non_power_of_two_input(self) -> None:
        # 64x96 IS divisible by 16; pick something that isn't to stress the
        # odd-size padding path in Up.forward.
        model = build_custom_unet(num_classes=7, in_channels=1, base_channels=8)
        model.eval()
        x = torch.randn(1, 1, 70, 100)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 7, 70, 100)

    def test_param_count_is_reasonable(self) -> None:
        # base=64, depth=4 should land in the ~30M canonical-U-Net range.
        # We allow a wide window because BatchNorm + 1-channel input shifts it
        # slightly from the original 31M number.
        model = build_custom_unet(num_classes=7, in_channels=1, base_channels=64)
        n_params = sum(p.numel() for p in model.parameters())
        assert 25_000_000 < n_params < 40_000_000, f"got {n_params}"

    def test_rejects_invalid_depth(self) -> None:
        with pytest.raises(ValueError):
            CustomUNet(depth=0)

    def test_rejects_invalid_base_channels(self) -> None:
        with pytest.raises(ValueError):
            CustomUNet(base_channels=0)

    def test_loss_backprop_runs(self) -> None:
        # Sanity-check the whole chain: forward, CE loss, backward.
        model = build_custom_unet(num_classes=3, in_channels=1, base_channels=8)
        x = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 3, (2, 32, 32))
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        # At least one encoder param must have received a non-zero gradient.
        has_grad = any(
            (p.grad is not None and p.grad.abs().sum() > 0)
            for p in model.in_conv.parameters()
        )
        assert has_grad
