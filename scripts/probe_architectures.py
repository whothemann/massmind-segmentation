"""Architecture probe over the 2x2 VGG16UNetExt ablation.

Trains four configurations sequentially in a single process:

    base       no attention gates, no transformer bottleneck  (the "standard" ext)
    att        attention gates only
    trans      transformer bottleneck only
    att_trans  both

Defaults match the user-requested probe profile: 600 training images,
10 epochs, augmentation A, focal loss (gamma=2). Estimated runtime is
~30-45 min / config on a Kaggle P100 (~2-3 hours total). Override any
hyperparameter via the CLI -- e.g. `--subset 30 --epochs 1 --batch-size 2`
for a laptop smoke test.

Outputs (under `--output-dir`, default ``runs/probe_<timestamp>/``):

    summary.json                  results across all configs
    <tag>/metrics.csv             per-epoch metrics
    <tag>/checkpoint_best.pt      best-by-mIoU model state
    <tag>/checkpoint_last.pt      final epoch
    <tag>/config.json             resolved TrainConfig

Usage:

    python scripts/probe_architectures.py                      # full probe
    python scripts/probe_architectures.py --configs base,att   # subset
    python scripts/probe_architectures.py --subset 30 --epochs 1  # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import torch

# Allow running as `python scripts/probe_architectures.py` from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.train import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    TrainConfig,
    autodetect_device,
    autodetect_num_workers,
    run_training,
)

logger = logging.getLogger("probe")


# Order matters: print the table in the order we run them.
PROBE_CONFIGS: tuple[dict, ...] = (
    {"tag": "base",      "attention": False, "transformer": False},
    {"tag": "att",       "attention": True,  "transformer": False},
    {"tag": "trans",     "attention": False, "transformer": True},
    {"tag": "att_trans", "attention": True,  "transformer": True},
)


def _best_miou_from_csv(metrics_csv: Path) -> float:
    """Read the highest mIoU value across all epochs in a metrics CSV."""
    best = float("-inf")
    with metrics_csv.open() as f:
        for row in csv.DictReader(f):
            try:
                v = float(row["mIoU"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v) and v > best:
                best = v
    return best if math.isfinite(best) else float("nan")


def _select_configs(requested: str) -> list[dict]:
    wanted = {t.strip() for t in requested.split(",") if t.strip()}
    valid = {c["tag"] for c in PROBE_CONFIGS}
    unknown = wanted - valid
    if unknown:
        raise ValueError(
            f"Unknown probe configs: {sorted(unknown)}. Valid: {sorted(valid)}"
        )
    return [c for c in PROBE_CONFIGS if c["tag"] in wanted]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--subset", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--augmentation", choices=["A", "B", "C"], default="A")
    p.add_argument("--loss", choices=["ce", "focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=None)
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip ImageNet weights for the VGG16 encoder.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = autodetect: 0 on Mac, 4 on CUDA).",
    )
    p.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--configs",
        default="base,att,trans,att_trans",
        help="Comma-separated subset of {base, att, trans, att_trans}.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Probe output directory. Default: runs/probe_<timestamp>/.",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    selected = _select_configs(args.configs)
    if not selected:
        raise SystemExit("No configs selected; check --configs.")

    device = autodetect_device(args.device)
    num_workers = autodetect_num_workers(device, args.num_workers)
    encoder_weights = None if args.no_pretrained else "imagenet"

    timestamp = int(time.time())
    base_out = args.output_dir or (DEFAULT_OUTPUT_DIR / f"probe_{timestamp}")
    base_out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 78)
    logger.info(
        "Architecture probe: %d config(s)  device=%s  epochs=%d  subset=%s  "
        "batch=%d  loss=%s  aug=%s",
        len(selected),
        device,
        args.epochs,
        args.subset,
        args.batch_size,
        args.loss,
        args.augmentation,
    )
    logger.info("Output: %s", base_out)
    logger.info("=" * 78)

    results: list[dict] = []
    t_probe = time.time()
    for i, meta in enumerate(selected, 1):
        tag = meta["tag"]
        config_out = base_out / tag
        logger.info("")
        logger.info(
            "[%d/%d] %s  (attention=%s, transformer=%s)",
            i,
            len(selected),
            tag,
            meta["attention"],
            meta["transformer"],
        )
        logger.info("-" * 78)

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
            encoder_weights=encoder_weights,
            loss=args.loss,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            model="vgg16_ext",
            use_attention_gates=meta["attention"],
            use_transformer_bottleneck=meta["transformer"],
        )

        t0 = time.time()
        try:
            run_training(cfg, config_out)
            status = "ok"
            error = None
        except Exception as e:  # don't let one config kill the whole probe
            elapsed = time.time() - t0
            logger.exception(
                "[%d/%d] %s FAILED after %.1f min", i, len(selected), tag,
                elapsed / 60,
            )
            results.append({
                "tag": tag,
                "use_attention_gates": meta["attention"],
                "use_transformer_bottleneck": meta["transformer"],
                "best_val_miou": float("nan"),
                "elapsed_s": round(elapsed, 1),
                "status": "failed",
                "error": str(e),
                "config_dir": str(config_out),
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        elapsed = time.time() - t0

        metrics_csv = config_out / "metrics.csv"
        best_miou = (
            _best_miou_from_csv(metrics_csv)
            if metrics_csv.exists()
            else float("nan")
        )

        results.append({
            "tag": tag,
            "use_attention_gates": meta["attention"],
            "use_transformer_bottleneck": meta["transformer"],
            "best_val_miou": (
                round(best_miou, 5) if math.isfinite(best_miou) else float("nan")
            ),
            "elapsed_s": round(elapsed, 1),
            "status": status,
            "error": error,
            "config_dir": str(config_out),
        })
        logger.info(
            "[%d/%d] %s done in %.1f min  best mIoU = %.4f",
            i, len(selected), tag, elapsed / 60, best_miou,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = time.time() - t_probe
    summary = {
        "timestamp": timestamp,
        "elapsed_total_s": round(total, 1),
        "device": str(device),
        "shared": {
            "epochs": args.epochs,
            "subset": args.subset,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "loss": args.loss,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": args.focal_alpha,
            "augmentation": args.augmentation,
            "encoder_weights": encoder_weights,
            "seed": args.seed,
        },
        "results": results,
    }
    (base_out / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info("")
    logger.info("=" * 78)
    logger.info("Probe complete in %.1f min", total / 60)
    logger.info("")
    logger.info(
        "%-10s | %-9s | %-11s | %-9s | %-8s | %s",
        "config", "attention", "transformer", "best mIoU", "time min", "status",
    )
    logger.info("-" * 78)
    for r in results:
        miou_str = (
            f"{r['best_val_miou']:.4f}"
            if isinstance(r["best_val_miou"], float)
            and math.isfinite(r["best_val_miou"])
            else "  nan "
        )
        logger.info(
            "%-10s | %-9s | %-11s | %-9s | %8.1f | %s",
            r["tag"],
            r["use_attention_gates"],
            r["use_transformer_bottleneck"],
            miou_str,
            r["elapsed_s"] / 60,
            r["status"],
        )
    logger.info("=" * 78)
    logger.info("Summary saved to %s", base_out / "summary.json")

    # Non-zero exit if any config failed (useful for CI/scripted use).
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
