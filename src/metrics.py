"""Segmentation metrics: streaming confusion matrix, per-class IoU, mIoU.

Why a confusion matrix and not per-batch IoU?

* Averaging per-batch IoU across a heldout set is **biased** when batches are
  small or class-imbalanced -- rare classes get a noisy IoU per batch that
  destabilises the mean.
* A single confusion matrix accumulated over the whole pass gives the exact
  dataset-level IoU as ``TP / (TP + FP + FN)``. This is the standard for
  semantic segmentation benchmarks (Cityscapes, ADE20K, the MassMIND paper).

Memory cost: ``num_classes x num_classes`` int64 -> 392 bytes for 7 classes.

Usage::

    tracker = ConfusionMatrixTracker(num_classes=7, ignore_index=255)
    for batch in loader:
        logits = model(batch["image"])
        tracker.update(logits.argmax(1), batch["mask"])
    ious = tracker.per_class_iou()
    miou = tracker.mean_iou()
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IoUResult:
    """Container returned by :meth:`ConfusionMatrixTracker.compute`."""

    per_class_iou: torch.Tensor  # shape [num_classes], NaN if class absent
    mean_iou: float
    pixel_accuracy: float

    def as_dict(self, class_names: list[str] | None = None) -> dict[str, float]:
        """Flatten to a single dict for CSV logging.

        Keys: ``mIoU``, ``pixel_acc``, then ``iou_<name>`` per class.
        """
        out: dict[str, float] = {
            "mIoU": self.mean_iou,
            "pixel_acc": self.pixel_accuracy,
        }
        names = class_names or [f"cls{i}" for i in range(len(self.per_class_iou))]
        for n, v in zip(names, self.per_class_iou.tolist()):
            out[f"iou_{n}"] = v
        return out


class ConfusionMatrixTracker:
    """Accumulates a ``[K, K]`` confusion matrix across batches.

    Rows are ground truth, columns are predictions. Pixels with target
    value equal to ``ignore_index`` are excluded.

    Args:
        num_classes: Number of valid classes (``K``).
        ignore_index: Target value to skip (e.g. 255 for augmentation borders).
        device: Where to keep the accumulator. Defaults to CPU; pushing to the
            model's device would save a host transfer but the bincount op is
            cheap on CPU and avoids one MPS quirk.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = torch.device(device)
        self.cm = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Add a batch to the running confusion matrix.

        Args:
            pred: Long tensor of class IDs, any shape, on any device.
            target: Long tensor of the same shape as ``pred``.

        Both tensors are flattened. Predictions outside ``[0, num_classes)``
        are clamped (defensive -- they shouldn't occur from argmax over K logits).
        """
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
        pred = pred.detach().to(self.device).reshape(-1)
        target = target.detach().to(self.device).reshape(-1)

        valid = target != self.ignore_index
        pred = pred[valid].clamp_(0, self.num_classes - 1)
        target = target[valid]

        # Vectorised confusion update: encode (target, pred) as a single index
        # in [0, K*K), bincount, reshape. This is the standard idiom and ~100x
        # faster than a Python loop.
        index = target * self.num_classes + pred
        binc = torch.bincount(index, minlength=self.num_classes ** 2)
        self.cm += binc.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> torch.Tensor:
        """IoU = TP / (TP + FP + FN) per class. NaN if a class has zero union."""
        cm = self.cm.float()
        tp = cm.diag()
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        union = tp + fp + fn
        iou = torch.where(union > 0, tp / union, torch.full_like(tp, float("nan")))
        return iou

    def mean_iou(self) -> float:
        """Mean IoU, ignoring classes that didn't appear (NaN -> skip)."""
        iou = self.per_class_iou()
        finite = iou[~torch.isnan(iou)]
        return float(finite.mean()) if finite.numel() > 0 else float("nan")

    def pixel_accuracy(self) -> float:
        """Fraction of pixels classified correctly across all valid pixels."""
        total = self.cm.sum()
        if total == 0:
            return float("nan")
        return float(self.cm.diag().sum() / total)

    def compute(self) -> IoUResult:
        return IoUResult(
            per_class_iou=self.per_class_iou(),
            mean_iou=self.mean_iou(),
            pixel_accuracy=self.pixel_accuracy(),
        )

    def reset(self) -> None:
        self.cm.zero_()
