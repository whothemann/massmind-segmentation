"""Tests for CustomLWIRUNet.

Mirrors the structure of ``test_unet_vgg16_ext.py`` for the parts that overlap
(shape tests, determinism, aux-head dispatch) and adds coverage specific to
this architecture: GroupNorm/SiLU/no-BatchNorm/no-AttentionGate invariants,
batch-1 forward safety, and AMP autocast compatibility.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models import build_custom_lwir_unet
from src.models._attention_gate import AttentionGate
from src.models._blocks import (
    DepthwiseSeparableConv,
    DoubleDSConv,
    StandardConvBlock,
)
from src.models._transformer_bottleneck import TransformerBottleneck
from src.models.custom_lwir_unet import CustomLWIRUNet


# ---------------------------------------------------------------------------
# Shape / forward-output dispatch.
# ---------------------------------------------------------------------------


def test_forward_shape_eval_mode() -> None:
    model = build_custom_lwir_unet(num_classes=7, in_channels=1, use_aux_heads=True)
    model.eval()
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    # Eval mode: single tensor regardless of use_aux_heads.
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 7, 256, 256)


def test_forward_shape_train_mode_with_aux_returns_tuple() -> None:
    model = build_custom_lwir_unet(num_classes=7, use_aux_heads=True)
    model.train()
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    assert isinstance(out, tuple) and len(out) == 3
    main, aux_shallow, aux_deep = out
    # All three at input resolution so they share one mask in the loss.
    for t in (main, aux_shallow, aux_deep):
        assert t.shape == (2, 7, 256, 256)


def test_forward_shape_train_mode_no_aux_returns_tensor() -> None:
    model = build_custom_lwir_unet(use_aux_heads=False)
    model.train()
    x = torch.randn(1, 1, 256, 256)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 7, 256, 256)


# ---------------------------------------------------------------------------
# Parameter-count regression. The custom architecture deliberately uses
# depthwise-separable convs to stay much lighter than the VGG16 baselines.
# Pin the actual number here so accidental changes (e.g. accidentally using
# StandardConvBlock in the encoder) get caught.
# ---------------------------------------------------------------------------


def test_parameter_count_regression() -> None:
    model = build_custom_lwir_unet(num_classes=7, use_aux_heads=True)
    n_params = sum(p.numel() for p in model.parameters())
    # Sanity bounds. DSConv-heavy U-Net + 384-ch transformer bottleneck
    # lands at ~4.69 M params with the spec defaults (stem 48, multipliers
    # (1,2,4,8,8), 2 transformer layers). Measured exactly = 4_690_533. If
    # a future edit accidentally swaps DSConv -> standard conv in the
    # encoder/decoder, the count would jump well above 8M and fail this
    # test.
    assert 4_500_000 < n_params < 4_900_000, f"got {n_params} params"


def test_parameter_count_aux_heads_overhead_is_small() -> None:
    # The aux heads add only two 1x1 convs at decoder mid-levels (128 -> 7
    # and 64 -> 7). Total extra weights + biases ~= (128*7 + 7) + (64*7 + 7)
    # = 903 + 455 = 1358.
    base = build_custom_lwir_unet(use_aux_heads=False)
    with_aux = build_custom_lwir_unet(use_aux_heads=True)
    extra = sum(p.numel() for p in with_aux.parameters()) - sum(
        p.numel() for p in base.parameters()
    )
    assert 1_000 < extra < 2_500, f"extra aux params = {extra}"


# ---------------------------------------------------------------------------
# Building-block invariants -- the design intent must be visible in the
# instantiated model graph.
# ---------------------------------------------------------------------------


def test_stem_uses_standard_conv_not_depthwise() -> None:
    # Depthwise on a 1-channel input is degenerate (single 3x3 filter), so
    # the stem must use plain Conv2d via StandardConvBlock. Inspecting both
    # stem submodules.
    model = build_custom_lwir_unet()
    stem_modules = list(model.stem.modules())
    standard_blocks = [m for m in stem_modules if isinstance(m, StandardConvBlock)]
    depthwise_blocks = [m for m in stem_modules if isinstance(m, DepthwiseSeparableConv)]
    assert len(standard_blocks) == 2, "stem must have exactly two StandardConvBlocks"
    assert len(depthwise_blocks) == 0, "stem must not contain DepthwiseSeparableConv"


def test_encoder_and_decoder_use_depthwise_separable() -> None:
    # Every encoder/decoder block is a DoubleDSConv, which contains two
    # DepthwiseSeparableConv submodules. The total count is 4 (encoder) + 4
    # (decoder) = 8 DoubleDSConvs, i.e. 16 DepthwiseSeparableConv instances.
    model = build_custom_lwir_unet()
    n_ds = sum(1 for _ in model.modules() if isinstance(_, DoubleDSConv))
    n_dsconv = sum(1 for _ in model.modules() if isinstance(_, DepthwiseSeparableConv))
    assert n_ds == 8
    assert n_dsconv == 16


def test_uses_silu_not_relu() -> None:
    model = build_custom_lwir_unet()
    relu_count = sum(1 for m in model.modules() if isinstance(m, nn.ReLU))
    silu_count = sum(1 for m in model.modules() if isinstance(m, nn.SiLU))
    assert relu_count == 0, f"unexpected ReLU instances: {relu_count}"
    assert silu_count > 0, "expected at least one SiLU activation"


def test_uses_groupnorm_not_batchnorm() -> None:
    model = build_custom_lwir_unet()
    bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    gn = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    assert bn == 0, f"unexpected BatchNorm2d instances: {bn}"
    assert gn > 0, "expected at least one GroupNorm"


def test_no_attention_gates() -> None:
    # The probe found AttentionGate doesn't combine well with the Transformer
    # bottleneck; this model omits AG by design.
    model = build_custom_lwir_unet()
    ag = sum(1 for m in model.modules() if isinstance(m, AttentionGate))
    assert ag == 0


def test_bottleneck_is_transformer() -> None:
    model = build_custom_lwir_unet()
    assert isinstance(model.bottleneck, TransformerBottleneck)


# ---------------------------------------------------------------------------
# Robustness: batch=1 must work (BatchNorm would fail here; GroupNorm must
# not). Also covers the smallest input divisible by 16.
# ---------------------------------------------------------------------------


def test_batch_size_one_forward_does_not_crash() -> None:
    model = build_custom_lwir_unet(use_aux_heads=True)
    model.eval()
    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 7, 256, 256)
    assert torch.isfinite(y).all()


def test_smaller_input_resolution() -> None:
    # 64x64 is the smallest input divisible by 16 that the model accepts
    # (encoder strides 16x, transformer takes 4x4 = 16 tokens).
    model = build_custom_lwir_unet(use_aux_heads=False)
    model.eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 7, 64, 64)


# ---------------------------------------------------------------------------
# AMP autocast compatibility. CUDA gets FP16; CPU uses BF16 since FP16
# autocast isn't supported on CPU (PyTorch enforces this).
# ---------------------------------------------------------------------------


def test_autocast_forward_no_nan_inf() -> None:
    if torch.cuda.is_available():
        device_type = "cuda"
        amp_dtype = torch.float16
        device = torch.device("cuda")
    else:
        device_type = "cpu"
        amp_dtype = torch.bfloat16
        device = torch.device("cpu")
    model = build_custom_lwir_unet(use_aux_heads=True).to(device)
    x = torch.randn(1, 1, 64, 64, device=device)
    model.train()
    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
        out = model(x)
    assert isinstance(out, tuple) and len(out) == 3
    for t in out:
        assert torch.isfinite(t.float()).all(), "non-finite values in autocast output"


def test_autocast_backward_gradients_flow() -> None:
    if torch.cuda.is_available():
        device_type = "cuda"
        amp_dtype = torch.float16
        device = torch.device("cuda")
    else:
        device_type = "cpu"
        amp_dtype = torch.bfloat16
        device = torch.device("cpu")
    model = build_custom_lwir_unet(num_classes=3, use_aux_heads=True).to(device)
    model.train()
    x = torch.randn(1, 1, 64, 64, device=device)
    target = torch.randint(0, 3, (1, 64, 64), device=device)
    with torch.amp.autocast(device_type=device_type, dtype=amp_dtype):
        out = model(x)
        # Combine the three heads with the standard deep-supervision weights.
        main, aux_shallow, aux_deep = out
        loss = (
            nn.functional.cross_entropy(main, target)
            + 0.4 * nn.functional.cross_entropy(aux_shallow, target)
            + 0.2 * nn.functional.cross_entropy(aux_deep, target)
        )
    loss.backward()
    # Confirm gradients reached the deepest part of the network (transformer
    # body), and not just the heads.
    has_bot_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.bottleneck.parameters()
    )
    assert has_bot_grad, "no gradient on the transformer bottleneck"


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


def test_determinism_same_seed_same_output() -> None:
    torch.manual_seed(0)
    m1 = build_custom_lwir_unet(use_aux_heads=True)
    torch.manual_seed(0)
    m2 = build_custom_lwir_unet(use_aux_heads=True)
    m1.eval(); m2.eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y1 = m1(x)
        y2 = m2(x)
    torch.testing.assert_close(y1, y2)


def test_determinism_train_mode_same_seed_same_output() -> None:
    torch.manual_seed(0)
    m1 = build_custom_lwir_unet(use_aux_heads=True)
    torch.manual_seed(0)
    m2 = build_custom_lwir_unet(use_aux_heads=True)
    m1.train(); m2.train()
    x = torch.randn(1, 1, 64, 64)
    out1 = m1(x)
    out2 = m2(x)
    for a, b in zip(out1, out2, strict=True):
        torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Configuration validation.
# ---------------------------------------------------------------------------


def test_rejects_non_single_channel_input() -> None:
    with pytest.raises(ValueError):
        build_custom_lwir_unet(in_channels=3)


def test_rejects_wrong_channel_multiplier_count() -> None:
    with pytest.raises(ValueError):
        build_custom_lwir_unet(channel_multipliers=(1, 2, 4, 8))  # only 4


def test_rejects_indivisible_transformer_heads() -> None:
    # Default deepest width is stem_channels * 8 = 384; with 7 heads that's
    # not divisible -> ValueError.
    with pytest.raises(ValueError):
        build_custom_lwir_unet(transformer_heads=7)


def test_aux_heads_disabled_by_default_explicit_false() -> None:
    model = build_custom_lwir_unet(use_aux_heads=False)
    assert model.use_aux_heads is False
    assert model.aux_head_deep is None
    assert model.aux_head_shallow is None


# ---------------------------------------------------------------------------
# Integration with the train.py loss combiner.
# ---------------------------------------------------------------------------


def test_compute_loss_combines_three_heads() -> None:
    from src.train import _compute_loss

    model = build_custom_lwir_unet(num_classes=3, use_aux_heads=True)
    model.train()
    x = torch.randn(1, 1, 64, 64)
    target = torch.randint(0, 3, (1, 64, 64))
    out = model(x)
    loss = _compute_loss(out, target, nn.CrossEntropyLoss())
    loss.backward()
    for head in (model.out_conv, model.aux_head_shallow, model.aux_head_deep):
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head.parameters()
        )
        assert has_grad, f"no gradient on {head.__class__.__name__}"


def test_aux_heads_eval_returns_main_logits_only_with_aux_constructed() -> None:
    # With use_aux_heads=True, eval mode must still return a single tensor
    # at the input resolution -- the aux heads are training-only. (Note: we
    # don't byte-compare against the no-aux model because init_silu_weights
    # walks all modules in registration order, so the additional aux heads
    # cause the shared modules to draw from a different RNG state at
    # post-construction re-init. That's an internal RNG quirk, not a bug --
    # checkpoints save the actual weights; nothing user-facing depends on
    # seed-induced byte parity between the two ctor flavours.)
    model = build_custom_lwir_unet(use_aux_heads=True)
    model.eval()
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 7, 64, 64)
