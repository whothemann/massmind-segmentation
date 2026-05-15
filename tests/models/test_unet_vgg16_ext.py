"""Tests for VGG16UNetExt and its swappable seams.

Covers:
* the 2x2 ablation (Identity / AttentionGate) x (DoubleConv / Transformer);
* the seam types are wired exactly as the flags request;
* pretrained-weight channel-mean adaptation (skipped if download fails);
* AttentionGate output shape & alpha-bounded magnitude;
* TransformerBottleneck shape preservation and pos-embed resize;
* determinism under a fixed seed.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models import build_unet_vgg16_ext
from src.models._attention_gate import AttentionGate
from src.models._transformer_bottleneck import TransformerBottleneck
from src.models.custom_unet import DoubleConv
from src.models.unet_vgg16_ext import VGG16Bottleneck, VGG16UNetExt


# ---------------------------------------------------------------------------
# Forward / shape tests across the 2x2 ablation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "use_trans,use_att",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_forward_shape(use_trans: bool, use_att: bool) -> None:
    model = build_unet_vgg16_ext(
        num_classes=7,
        in_channels=1,
        encoder_weights=None,
        use_transformer_bottleneck=use_trans,
        use_attention_gates=use_att,
    )
    model.eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 7, 256, 256)


# ---------------------------------------------------------------------------
# Seam wiring tests.
# ---------------------------------------------------------------------------


def test_skip_refine_default_is_identity() -> None:
    model = build_unet_vgg16_ext(encoder_weights=None, use_attention_gates=False)
    for up in model.ups:
        assert isinstance(up.skip_refine, nn.Identity)


def test_skip_refine_swapped_to_attention_gate() -> None:
    model = build_unet_vgg16_ext(encoder_weights=None, use_attention_gates=True)
    for up in model.ups:
        assert isinstance(up.skip_refine, AttentionGate)


def test_bottleneck_body_default_is_double_conv() -> None:
    model = build_unet_vgg16_ext(
        encoder_weights=None, use_transformer_bottleneck=False
    )
    assert isinstance(model.bottleneck, VGG16Bottleneck)
    assert isinstance(model.bottleneck.body, DoubleConv)


def test_bottleneck_body_swapped_to_transformer() -> None:
    model = build_unet_vgg16_ext(
        encoder_weights=None, use_transformer_bottleneck=True
    )
    assert isinstance(model.bottleneck, VGG16Bottleneck)
    assert isinstance(model.bottleneck.body, TransformerBottleneck)


def test_bottleneck_has_no_pool_for_vgg16() -> None:
    # Design decision: the VGG16 bottleneck does NOT include a MaxPool, unlike
    # the from-scratch custom_unet.Bottleneck. Verify the wrapper only carries
    # the body submodule.
    model = build_unet_vgg16_ext(encoder_weights=None)
    children = list(model.bottleneck.named_children())
    assert len(children) == 1 and children[0][0] == "body"


# ---------------------------------------------------------------------------
# Pretrained channel-mean adaptation.
# ---------------------------------------------------------------------------


def test_pretrained_first_conv_is_channel_mean() -> None:
    # Build a reference SMP encoder and compute the channel-mean of its
    # RGB first conv; then verify the model's encoder ends up with exactly
    # that weight after the 1-channel adaptation.
    import segmentation_models_pytorch as smp

    try:
        ref_encoder = smp.encoders.get_encoder(
            "vgg16", in_channels=3, depth=5, weights="imagenet"
        )
    except Exception as e:  # network unavailable or weights URL down
        pytest.skip(f"pretrained VGG16 weights unavailable: {e}")

    rgb_weight = ref_encoder.features[0].weight.detach()
    expected = rgb_weight.mean(dim=1, keepdim=True)

    model = build_unet_vgg16_ext(
        num_classes=7, in_channels=1, encoder_weights="imagenet"
    )
    got = model.encoder.features[0].weight.detach()
    assert got.shape == (64, 1, 3, 3)
    torch.testing.assert_close(got, expected)


def test_invalid_in_channels() -> None:
    with pytest.raises(ValueError):
        build_unet_vgg16_ext(in_channels=2, encoder_weights=None)


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


def test_determinism_same_seed_same_output() -> None:
    torch.manual_seed(0)
    m1 = build_unet_vgg16_ext(
        encoder_weights=None,
        use_transformer_bottleneck=True,
        use_attention_gates=True,
    )
    torch.manual_seed(0)
    m2 = build_unet_vgg16_ext(
        encoder_weights=None,
        use_transformer_bottleneck=True,
        use_attention_gates=True,
    )
    m1.eval()
    m2.eval()
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y1 = m1(x)
        y2 = m2(x)
    torch.testing.assert_close(y1, y2)


# ---------------------------------------------------------------------------
# AttentionGate isolated tests.
# ---------------------------------------------------------------------------


def test_attention_gate_output_shape() -> None:
    gate = AttentionGate(skip_channels=64, gating_channels=128)
    skip = torch.randn(2, 64, 16, 16)
    gating = torch.randn(2, 128, 16, 16)
    out = gate(skip, gating)
    assert out.shape == skip.shape


def test_attention_gate_magnitude_bounded_by_skip() -> None:
    # Output is alpha * skip with alpha in (0, 1); per-pixel magnitude must
    # not exceed the corresponding |skip| value.
    gate = AttentionGate(skip_channels=32, gating_channels=64).eval()
    skip = torch.randn(1, 32, 8, 8)
    gating = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        out = gate(skip, gating)
    assert (out.abs() <= skip.abs() + 1e-5).all()


def test_attention_gate_default_inter_channels() -> None:
    gate = AttentionGate(skip_channels=64, gating_channels=128)
    # W_skip is Sequential(Conv2d, BatchNorm2d); Conv2d.out_channels carries
    # the inter_channels we want to check.
    assert gate.W_skip[0].out_channels == 32


def test_attention_gate_inter_channels_override() -> None:
    gate = AttentionGate(skip_channels=64, gating_channels=128, inter_channels=8)
    assert gate.W_skip[0].out_channels == 8


def test_attention_gate_rejects_zero_inter() -> None:
    with pytest.raises(ValueError):
        AttentionGate(skip_channels=64, gating_channels=128, inter_channels=0)


def test_attention_gate_rejects_spatial_mismatch() -> None:
    gate = AttentionGate(skip_channels=8, gating_channels=8)
    with pytest.raises(ValueError):
        gate(torch.randn(1, 8, 16, 16), torch.randn(1, 8, 8, 8))


# ---------------------------------------------------------------------------
# TransformerBottleneck isolated tests.
# ---------------------------------------------------------------------------


def test_transformer_bottleneck_shape_preserving() -> None:
    bot = TransformerBottleneck(channels=64, num_heads=4, num_layers=2, spatial_size=8)
    x = torch.randn(2, 64, 8, 8)
    out = bot(x)
    assert out.shape == x.shape


def test_transformer_bottleneck_resizes_pos_embed() -> None:
    # Configured spatial_size 8x8 but input is 16x16 -- pos embed must be
    # interpolated to match.
    bot = TransformerBottleneck(channels=64, num_heads=4, num_layers=1, spatial_size=8)
    out = bot(torch.randn(1, 64, 16, 16))
    assert out.shape == (1, 64, 16, 16)


def test_transformer_bottleneck_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError):
        TransformerBottleneck(channels=63, num_heads=8)


def test_transformer_bottleneck_rejects_wrong_channels() -> None:
    bot = TransformerBottleneck(channels=64, num_heads=4, num_layers=1)
    with pytest.raises(ValueError):
        bot(torch.randn(1, 32, 8, 8))


# ---------------------------------------------------------------------------
# Loss backprop sanity check on the full model.
# ---------------------------------------------------------------------------


def test_full_model_backprop_runs() -> None:
    model = build_unet_vgg16_ext(
        num_classes=3,
        in_channels=1,
        encoder_weights=None,
        use_transformer_bottleneck=True,
        use_attention_gates=True,
    )
    x = torch.randn(1, 1, 64, 64)
    target = torch.randint(0, 3, (1, 64, 64))
    loss = nn.functional.cross_entropy(model(x), target)
    loss.backward()
    # Bottleneck body params must receive gradient (gate participates).
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.bottleneck.body.parameters()
    )
    assert has_grad
    # Attention gate params must also receive gradient.
    ag = model.ups[0].skip_refine
    has_ag_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in ag.parameters()
    )
    assert has_ag_grad
