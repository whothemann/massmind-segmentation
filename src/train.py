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
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .augmentations import MASK_IGNORE_INDEX, build_pipeline
from .dataset import NUM_CLASSES, MassMINDDataset, make_collate_fn
from .losses import FocalLoss
from .metrics import ConfusionMatrixTracker
from .models import build_unet_vgg16, build_unet_vgg16_ext
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
    model: str = "vgg16"               # "vgg16" or "vgg16_ext"
    use_attention_gates: bool = False  # only consulted when model == "vgg16_ext"
    use_transformer_bottleneck: bool = False
    amp: bool = False                  # mixed precision (fp16 autocast + GradScaler); CUDA-only


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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = False,
) -> float:
    """One training epoch.

    When ``amp=True`` and ``device.type == "cuda"``, the forward + loss run
    inside ``torch.amp.autocast(fp16)`` and the backward goes through a
    ``GradScaler``. Both are no-ops when AMP is disabled or the device isn't
    CUDA, so the existing CPU/MPS smoke-test path is unaffected.
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
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
        )
    raise ValueError(
        f"Unknown model {cfg.model!r}; expected one of vgg16, vgg16_ext."
    )


def run_training(cfg: TrainConfig, output_dir: Path) -> None:
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    logger.info("Device: %s", device)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

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
        )
        val_loss, tracker = evaluate(
            model, val_loader, criterion, device, amp=cfg.amp,
        )
        result = tracker.compute()
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
        choices=["vgg16", "vgg16_ext"],
        default="vgg16",
        help="Model variant: SMP VGG16 U-Net (default) or our extended one.",
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
        amp=args.amp,
    )

    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_DIR
        / f"{args.model}_aug{args.augmentation}_{args.loss}_{int(time.time())}"
    )
    run_training(cfg, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
