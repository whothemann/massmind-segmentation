"""Loss functions for MassMIND semantic segmentation.

Currently provides :class:`FocalLoss` -- a drop-in replacement for
``nn.CrossEntropyLoss(ignore_index=...)`` that down-weights easy-to-classify
pixels and focuses gradient on the hard ones. This is the standard
Lin et al. (2017) formulation extended to dense prediction:

    FL(p_t) = - alpha_t * (1 - p_t) ** gamma * log(p_t)

where ``p_t`` is the softmax probability assigned to the ground-truth class.
The relationship to cross-entropy is exact: ``-log(p_t) == CE(logits, target)``,
so we compute pixel-wise CE first and modulate from there. ``alpha`` can be a
scalar (single down-weighting factor applied to all classes) or a per-class
tensor of length ``num_classes`` for class-balanced focusing.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multi-class focal loss for dense prediction.

    Args:
        gamma: Focusing parameter. ``gamma=0`` reduces to (alpha-weighted) CE;
            larger values down-weight well-classified pixels more aggressively.
            ``gamma=2.0`` is the value from the original paper.
        alpha: Per-class weighting. ``None`` for no weighting; a float for a
            global down-weighting factor; a 1-D tensor of shape ``[num_classes]``
            for per-class weights (typically inverse-frequency for imbalance).
        ignore_index: Target value to ignore (no gradient, excluded from the
            mean). Matches ``nn.CrossEntropyLoss`` semantics.
        reduction: ``"mean"`` (over non-ignored pixels), ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | torch.Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be mean/sum/none (got {reduction!r})")
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W], targets: [B, H, W] (long)
        ce = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index,
        )  # [B, H, W]; ignored positions are returned as 0
        pt = torch.exp(-ce)
        focal = (1.0 - pt).pow(self.gamma) * ce

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Index per-class weights. Clamp targets so ignore_index doesn't
                # OOB into the alpha tensor; ignored positions are masked below.
                safe_targets = targets.clamp(min=0, max=self.alpha.numel() - 1)
                at = self.alpha.to(focal.device)[safe_targets]
                focal = at * focal
            else:
                focal = float(self.alpha) * focal

        valid = targets != self.ignore_index
        if self.reduction == "none":
            return focal
        if self.reduction == "sum":
            return focal[valid].sum()
        # mean over non-ignored pixels; guard against an all-ignored batch
        n = valid.sum()
        if n == 0:
            return focal.sum() * 0.0
        return focal[valid].sum() / n
