"""Tests for the confusion-matrix-based metrics."""
from __future__ import annotations

import math

import pytest
import torch

from src.metrics import ConfusionMatrixTracker


def test_perfect_prediction_gives_iou_1() -> None:
    tracker = ConfusionMatrixTracker(num_classes=3)
    target = torch.tensor([[0, 1, 2], [2, 1, 0]])
    tracker.update(pred=target, target=target)
    iou = tracker.per_class_iou()
    assert torch.allclose(iou, torch.ones(3))
    assert tracker.mean_iou() == pytest.approx(1.0)
    assert tracker.pixel_accuracy() == pytest.approx(1.0)


def test_iou_matches_hand_computed() -> None:
    # Confusion matrix:
    #   gt=0,pred=0: 3   gt=0,pred=1: 1
    #   gt=1,pred=0: 0   gt=1,pred=1: 4
    # IoU(0) = 3 / (3 + 0 + 1) = 0.75
    # IoU(1) = 4 / (4 + 1 + 0) = 0.80
    target = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    pred = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
    tracker = ConfusionMatrixTracker(num_classes=2)
    tracker.update(pred=pred, target=target)
    iou = tracker.per_class_iou().tolist()
    assert iou[0] == pytest.approx(0.75)
    assert iou[1] == pytest.approx(0.80)
    assert tracker.mean_iou() == pytest.approx((0.75 + 0.80) / 2)


def test_ignore_index_excludes_pixels() -> None:
    target = torch.tensor([0, 0, 1, 1, 255, 255])
    pred = torch.tensor([0, 0, 1, 1, 0, 1])
    tracker = ConfusionMatrixTracker(num_classes=2, ignore_index=255)
    tracker.update(pred=pred, target=target)
    # The two 255-target pixels are dropped, leaving perfect predictions.
    assert tracker.mean_iou() == pytest.approx(1.0)


def test_absent_class_returns_nan() -> None:
    target = torch.tensor([0, 0, 0])
    pred = torch.tensor([0, 0, 0])
    tracker = ConfusionMatrixTracker(num_classes=3)
    tracker.update(pred=pred, target=target)
    iou = tracker.per_class_iou()
    assert iou[0].item() == pytest.approx(1.0)
    assert math.isnan(iou[1].item())
    assert math.isnan(iou[2].item())
    # mean_iou should average only the non-NaN entries.
    assert tracker.mean_iou() == pytest.approx(1.0)


def test_streaming_equals_single_update() -> None:
    torch.manual_seed(0)
    target = torch.randint(0, 7, (1000,))
    pred = torch.randint(0, 7, (1000,))

    single = ConfusionMatrixTracker(num_classes=7)
    single.update(pred=pred, target=target)

    streamed = ConfusionMatrixTracker(num_classes=7)
    for chunk in range(0, 1000, 137):
        streamed.update(pred=pred[chunk:chunk + 137], target=target[chunk:chunk + 137])

    torch.testing.assert_close(single.cm, streamed.cm)


def test_reset_clears_state() -> None:
    tracker = ConfusionMatrixTracker(num_classes=2)
    tracker.update(pred=torch.tensor([0, 1]), target=torch.tensor([0, 1]))
    tracker.reset()
    assert tracker.cm.sum().item() == 0
    assert math.isnan(tracker.mean_iou())


def test_shape_mismatch_raises() -> None:
    tracker = ConfusionMatrixTracker(num_classes=2)
    with pytest.raises(ValueError):
        tracker.update(pred=torch.zeros(3, dtype=torch.long), target=torch.zeros(4, dtype=torch.long))
