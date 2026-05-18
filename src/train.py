"""Training loop for MassMIND semantic segmentation.

Single-file trainer for the U-Net + VGG-16 baseline. Runs on MPS (M-series
Mac), CUDA (Colab T4), or CPU automatically. The same script is intended to
work in both environments -- only the device-dependent knobs (``num_workers``,
mixed precision) auto-adjust.

For laptop smoke tests, use ``--subset 30 --epochs 1`` to validate the loop
end-to-end before pushing to Colab.

Outputs (per run, under ``--output-dir``):

    checkpoint_best.pt     -- best-by-val-mIoU model state
    checkpoint_last.pt     -- final epoch state
    metrics.csv            -- one row per epoch with train/val losses + IoUs
    config.json            -- the resolved argparse args, for reproducibility
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .augmentations import MASK_IGNORE_INDEX, build_pipeline
from .dataset import NUM_CLASSES, MassMINDDataset, make_collate_fn
from .losses import FocalLoss
from .metrics import ConfusionMatrixTracker
from .models import (
    build_custom_lwir_unet,
    build_unet_vgg16,
    build_unet_vgg16_ext,
)
from .models.unet_vgg16_ext import (
    AUX_HEAD_WEIGHT_DEEP,
    AUX_HEAD_WEIGHT_SHALLOW,
)
from .splits import load_splits
from .stats import load_stats

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "sky", "water", "bridge", "obstacle", "living_obs", "background", "self",
]

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = DEFAULT_PROJECT_ROOT / "data" / "massmind"
DEFAULT_SPLIT_PATH = DEFAULT_PROJECT_ROOT / "data" / "splits" / "split.json"
DEFAULT_STATS_PATH = DEFAULT_PROJECT_ROOT / "data" / "splits" / "stats.json"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "runs"


@dataclass
class TrainConfig:
    """Resolved hyperparameters for one training run."""

    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    augmentation: str          # "A", "B", or "C"
    subset: int | None         # cap on training-set size for smoke tests
    num_workers: int
    device: str
    seed: int
    encoder_weights: str | None
    loss: str                  # "ce" or "focal"
    focal_gamma: float
    focal_alpha: float | None
    model: str = "vgg16"               # "vgg16", "vgg16_ext", or "custom_lwir"
    use_attention_gates: bool = False  # only consulted when model == "vgg16_ext"
    use_transformer_bottleneck: bool = False
    use_aux_heads: bool = False        # deep supervision; vgg16_ext & custom_lwir
    amp: bool = False                  # mixed precision (fp16 autocast + GradScaler); CUDA-only
    # custom_lwir-specific overrides (no-ops for other model types):
    stem_channels: int = 48
    transformer_layers: int = 2
    # LR-schedule warmup as a fraction of total training steps. 0.0 = pure
    # cosine (current behaviour, preserved for the pretrained baseline);
    # 0.05 default-on for from-scratch configs (custom_lwir or --no-pretrained).
    warmup_frac: float = 0.0
    warmup_auto_set: bool = False      # True if warmup_frac was auto-defaulted (for logging)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def autodetect_device(requested: str = "auto") -> torch.device:
    """Pick a device. Order: explicit request -> CUDA -> MPS -> CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autodetect_num_workers(device: torch.device, requested: int) -> int:
    """Pick DataLoader num_workers based on platform. macOS spawn semantics
    around DataLoader workers are fragile; default to 0 on Mac/MPS, 4 on CUDA.
    """
    if requested >= 0:
        return requested
    if device.type == "cuda":
        return 4
    if platform.system() == "Darwin":
        return 0
    return 2


def build_dataloaders(
    cfg: TrainConfig,
    data_root: Path,
    split_path: Path,
    stats_path: Path,
) -> tuple[DataLoader, DataLoader, float, float]:
    splits = load_splits(split_path)
    mean, std = load_stats(stats_path)

    train_files = splits["train"]
    if cfg.subset is not None and cfg.subset > 0:
        train_files = train_files[: cfg.subset]
        logger.info("Subsetting training set to first %d files.", cfg.subset)

    val_files = splits["val"]
    if cfg.subset is not None and cfg.subset > 0:
        # Keep validation small but never empty during smoke tests.
        val_files = val_files[: max(8, cfg.subset // 4)]

    train_tf = build_pipeline(cfg.augmentation, mean, std, train=True)
    val_tf = build_pipeline("C", mean, std, train=False)

    train_ds = MassMINDDataset(data_root, train_files, train_tf)
    val_ds = MassMINDDataset(data_root, val_files, val_tf)

    collate = make_collate_fn()
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        collate_fn=collate,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        collate_fn=collate,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader, mean, std


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def _compute_loss(
    output: torch.Tensor | tuple[torch.Tensor, ...],
    masks: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute loss for either a single-tensor or aux-tuple forward output.

    When the model returns a tuple ``(main, aux_shallow, aux_deep)`` (i.e.
    deep supervision is active and the model is in training mode), the
    combined loss matches the README plan::

        L_total = L_main
                + AUX_HEAD_WEIGHT_SHALLOW * L_aux_shallow
                + AUX_HEAD_WEIGHT_DEEP    * L_aux_deep

    All three heads predict at full input resolution against the same mask.
    """
    if isinstance(output, tuple):
        main, aux_shallow, aux_deep = output
        return (
            criterion(main, masks)
            + AUX_HEAD_WEIGHT_SHALLOW * criterion(aux_shallow, masks)
            + AUX_HEAD_WEIGHT_DEEP * criterion(aux_deep, masks)
        )
    return criterion(output, masks)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = False,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    step_scheduler_per_batch: bool = False,
) -> float:
    """One training epoch.

    When ``amp=True`` and ``device.type == "cuda"``, the forward + loss run
    inside ``torch.amp.autocast(fp16)`` and the backward goes through a
    ``GradScaler``. Both are no-ops when AMP is disabled or the device isn't
    CUDA, so the existing CPU/MPS smoke-test path is unaffected.

    The loss is the main-head CE/Focal by default. If the model returns a
    deep-supervision tuple ``(main, aux_shallow, aux_deep)``, the three
    terms are combined with the standard weights via :func:`_compute_loss`.

    If ``scheduler`` is provided and ``step_scheduler_per_batch`` is True,
    the scheduler advances once per batch (needed for SequentialLR with
    LinearLR warmup, which counts ``total_iters`` in steps, not epochs).
    Otherwise the caller is expected to step the scheduler once per epoch.
    """
    model.train()
    use_amp = bool(amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    running_loss = 0.0
    n_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            output = model(images)
            loss = _compute_loss(output, masks, criterion)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and step_scheduler_per_batch:
            scheduler.step()
        bs = images.size(0)
        running_loss += float(loss.detach()) * bs
        n_samples += bs
    return running_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = False,
) -> tuple[float, ConfusionMatrixTracker]:
    """Evaluation pass.

    Wraps forward + loss in ``torch.amp.autocast`` when AMP is requested
    on a CUDA device. The argmax + confusion matrix run in FP32 regardless,
    so the IoU metric is unaffected by autocast precision.
    """
    model.eval()
    use_amp = bool(amp) and device.type == "cuda"
    tracker = ConfusionMatrixTracker(NUM_CLASSES, ignore_index=MASK_IGNORE_INDEX)
    running_loss = 0.0
    n_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)
        tracker.update(pred=preds, target=masks)
        bs = images.size(0)
        running_loss += float(loss.detach()) * bs
        n_samples += bs
    return running_loss / max(n_samples, 1), tracker


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_criterion(cfg: TrainConfig) -> nn.Module:
    """Build the training loss from cfg.loss."""
    if cfg.loss == "ce":
        return nn.CrossEntropyLoss(ignore_index=MASK_IGNORE_INDEX)
    if cfg.loss == "focal":
        return FocalLoss(
            gamma=cfg.focal_gamma,
            alpha=cfg.focal_alpha,
            ignore_index=MASK_IGNORE_INDEX,
        )
    raise ValueError(f"Unknown loss {cfg.loss!r}; expected one of ce, focal.")


def build_model(cfg: TrainConfig) -> nn.Module:
    """Build the segmentation model from cfg.model."""
    if cfg.model == "vgg16":
        return build_unet_vgg16(
            num_classes=NUM_CLASSES,
            in_channels=1,
            encoder_weights=cfg.encoder_weights,
        )
    if cfg.model == "vgg16_ext":
        return build_unet_vgg16_ext(
            num_classes=NUM_CLASSES,
            in_channels=1,
            encoder_weights=cfg.encoder_weights,
            use_attention_gates=cfg.use_attention_gates,
            use_transformer_bottleneck=cfg.use_transformer_bottleneck,
            use_aux_heads=cfg.use_aux_heads,
        )
    if cfg.model == "custom_lwir":
        return build_custom_lwir_unet(
            num_classes=NUM_CLASSES,
            in_channels=1,
            stem_channels=cfg.stem_channels,
            transformer_layers=cfg.transformer_layers,
            use_aux_heads=cfg.use_aux_heads,
        )
    raise ValueError(
        f"Unknown model {cfg.model!r}; expected one of vgg16, vgg16_ext, "
        "custom_lwir."
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler, bool]:
    """Build the LR schedule and report its stepping cadence.

    Returns a tuple ``(scheduler, step_per_batch)``:

    * ``warmup_frac == 0.0`` -> plain ``CosineAnnealingLR`` (T_max = epochs),
      stepped once per epoch. This is byte-identical to the pre-warmup
      behaviour, so the pretrained-VGG16 baseline numerics don't drift.
    * ``warmup_frac > 0.0``  -> ``SequentialLR([LinearLR, CosineAnnealingLR])``
      stepped once per batch. LinearLR ramps from ``base_lr * 1e-3`` to
      ``base_lr`` over the warmup window (PyTorch's ``LinearLR`` requires
      ``start_factor > 0``, so we use 1e-3 rather than literal 0).
    """
    if cfg.warmup_frac <= 0.0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        )
        logger.info(
            "LR schedule: warmup_frac=0.0 (no warmup), cosine over %d epochs",
            cfg.epochs,
        )
        return scheduler, False

    total_steps = max(cfg.epochs * steps_per_epoch, 1)
    warmup_steps = max(int(round(cfg.warmup_frac * total_steps)), 1)
    cosine_steps = max(total_steps - warmup_steps, 1)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
    origin = "auto-set for from-scratch config" if cfg.warmup_auto_set else "user-specified"
    logger.info(
        "LR schedule: warmup_frac=%.3f (%s), total_steps=%d, "
        "warmup_steps=%d, then cosine to 0 over %d steps",
        cfg.warmup_frac,
        origin,
        total_steps,
        warmup_steps,
        cosine_steps,
    )
    return scheduler, True


def run_training(cfg: TrainConfig, output_dir: Path) -> None:
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(
        "Training on device=%s, visible_gpus=%d, CUDA_VISIBLE_DEVICES=%s",
        device,
        visible_gpus,
        os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    )

    train_loader, val_loader, mean, std = build_dataloaders(
        cfg, DEFAULT_DATA_ROOT, DEFAULT_SPLIT_PATH, DEFAULT_STATS_PATH,
    )
    logger.info(
        "train batches=%d (n=%d)  val batches=%d (n=%d)  norm: mean=%.4f std=%.4f",
        len(train_loader), len(train_loader.dataset),
        len(val_loader), len(val_loader.dataset),
        mean, std,
    )

    model = build_model(cfg).to(device)

    criterion = build_criterion(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler, step_scheduler_per_batch = build_scheduler(
        optimizer, cfg, steps_per_epoch=len(train_loader)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    csv_path = output_dir / "metrics.csv"

    fieldnames = (
        ["epoch", "train_loss", "val_loss", "mIoU", "pixel_acc"]
        + [f"iou_{n}" for n in CLASS_NAMES]
        + ["lr", "elapsed_s"]
    )
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(fieldnames)

    best_miou = -1.0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, amp=cfg.amp,
            scheduler=scheduler,
            step_scheduler_per_batch=step_scheduler_per_batch,
        )
        val_loss, tracker = evaluate(
            model, val_loader, criterion, device, amp=cfg.amp,
        )
        result = tracker.compute()
        # SequentialLR (warmup path) is stepped per batch inside the train
        # loop; the plain cosine schedule (no-warmup path) ticks per epoch.
        if not step_scheduler_per_batch:
            scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "mIoU": round(result.mean_iou, 5),
            "pixel_acc": round(result.pixel_accuracy, 5),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
        }
        for name, iou in zip(CLASS_NAMES, result.per_class_iou.tolist()):
            row[f"iou_{name}"] = (
                round(iou, 5) if iou == iou else float("nan")  # NaN check
            )

        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        logger.info(
            "epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  mIoU=%.4f  pixel_acc=%.4f  (%.1fs)",
            epoch, cfg.epochs, train_loss, val_loss, result.mean_iou,
            result.pixel_accuracy, elapsed,
        )

        # Save best
        if result.mean_iou > best_miou and result.mean_iou == result.mean_iou:
            best_miou = result.mean_iou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "config": asdict(cfg),
                },
                output_dir / "checkpoint_best.pt",
            )
            logger.info("  saved new best (mIoU=%.4f)", best_miou)

    # Always save the last epoch too -- useful if training was cut short.
    torch.save(
        {
            "epoch": cfg.epochs,
            "model_state": model.state_dict(),
            "mean": mean,
            "std": std,
            "config": asdict(cfg),
        },
        output_dir / "checkpoint_last.pt",
    )
    logger.info("Training complete. Best val mIoU=%.4f", best_miou)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--augmentation", choices=["A", "B", "C"], default="A")
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use first N training images (for laptop smoke tests).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = autodetect: 0 on Mac, 4 on CUDA).",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip ImageNet weights for the encoder (random init).",
    )
    p.add_argument(
        "--loss",
        choices=["ce", "focal"],
        default="ce",
        help="Training loss: cross-entropy (default) or focal.",
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing parameter for focal loss (only used with --loss focal).",
    )
    p.add_argument(
        "--focal-alpha",
        type=float,
        default=None,
        help="Scalar alpha for focal loss (None = unweighted).",
    )
    p.add_argument(
        "--model",
        choices=["vgg16", "vgg16_ext", "custom_lwir"],
        default="vgg16",
        help="Model variant: SMP VGG16 U-Net (default), our extended VGG16, "
        "or the from-scratch CustomLWIRUNet.",
    )
    p.add_argument(
        "--stem-channels",
        type=int,
        default=48,
        help="custom_lwir only: channels out of the stem (default 48).",
    )
    p.add_argument(
        "--transformer-layers",
        type=int,
        default=2,
        help="custom_lwir only: number of TransformerEncoder layers in the "
        "bottleneck (default 2).",
    )
    p.add_argument(
        "--warmup-frac",
        type=float,
        default=None,
        help="Linear LR warmup as a fraction of total training steps. "
        "If unset: 0.05 for custom_lwir or --no-pretrained (from-scratch "
        "configs), else 0.0 (pure cosine, preserves baseline behaviour).",
    )
    p.add_argument(
        "--attention-gates",
        action="store_true",
        help="Enable AttentionGate skip refiners (model=vgg16_ext only).",
    )
    p.add_argument(
        "--transformer-bottleneck",
        action="store_true",
        help="Enable TransformerBottleneck body (model=vgg16_ext only).",
    )
    p.add_argument(
        "--aux-heads",
        action="store_true",
        help="Enable deep-supervision aux heads (model=vgg16_ext only). "
        "Two 1x1-conv aux heads at decoder mid-levels with weights "
        f"{AUX_HEAD_WEIGHT_SHALLOW}/{AUX_HEAD_WEIGHT_DEEP}; main head "
        "weight is 1.0. Aux heads are dropped at inference.",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (fp16 autocast + GradScaler). CUDA-only; "
        "no-op on CPU/MPS. Off by default to preserve baseline numerics.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run output directory. Default: runs/<augmentation>_<timestamp>.",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    device = autodetect_device(args.device)
    num_workers = autodetect_num_workers(device, args.num_workers)

    # --no-pretrained sanity check. custom_lwir is always trained from scratch,
    # so combining the two flags is harmless but misleading; warn so the run
    # config in config.json doesn't confuse a future reader.
    if args.no_pretrained and args.model == "custom_lwir":
        logger.warning(
            "--no-pretrained is a no-op for --model custom_lwir "
            "(this architecture has no pretrained option). Ignoring."
        )

    # Auto-default warmup_frac. The pretrained baseline keeps pure cosine
    # (preserves prior numerics exactly); from-scratch configs get 5% warmup
    # because the Transformer bottleneck on small datasets is sensitive to
    # early-epoch LR shocks.
    if args.warmup_frac is None:
        is_from_scratch = (args.model == "custom_lwir") or args.no_pretrained
        warmup_frac = 0.05 if is_from_scratch else 0.0
        warmup_auto_set = True
    else:
        warmup_frac = float(args.warmup_frac)
        warmup_auto_set = False

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        augmentation=args.augmentation,
        subset=args.subset,
        num_workers=num_workers,
        device=str(device),
        seed=args.seed,
        encoder_weights=None if args.no_pretrained else "imagenet",
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        model=args.model,
        use_attention_gates=args.attention_gates,
        use_transformer_bottleneck=args.transformer_bottleneck,
        use_aux_heads=args.aux_heads,
        amp=args.amp,
        stem_channels=args.stem_channels,
        transformer_layers=args.transformer_layers,
        warmup_frac=warmup_frac,
        warmup_auto_set=warmup_auto_set,
    )

    # Run isolation: include the PID in the default output dir name. This
    # prevents collisions when two parallel sessions (one per GPU) start
    # within the same second on a multi-GPU machine.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_DIR
        / f"{args.model}_{args.loss}_{timestamp}_pid{os.getpid()}"
    )
    run_training(cfg, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
