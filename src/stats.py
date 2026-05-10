"""Compute single-channel mean/std across the MassMIND training split.

ImageNet's [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] are RGB statistics on
natural photographs and are not meaningful for thermal imagery. We compute
dataset-specific statistics over the training split only (never val/test, to
avoid leakage), and cache the result to ``data/splits/stats.json``.

The computation uses Welford-style streaming sums (sum and sum-of-squares) so
it scales to the full dataset in O(1) memory and stays numerically stable on
float64. All pixel values are normalised to [0, 1] before accumulation:

    8-bit  PNG  -> divide by 255
    16-bit TIFF -> divide by 65535
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .splits import load_splits

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "massmind"
DEFAULT_SPLIT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "splits" / "split.json"
)
DEFAULT_STATS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "splits" / "stats.json"
)


def _load_image_normalised(path: Path) -> np.ndarray:
    """Load an image to a float32 array in [0, 1].

    Handles 8-bit PNG and 16-bit TIFF/PNG transparently. The output is always
    2D (single-channel); if the file unexpectedly has multiple channels we
    average them, since the underlying capture is single-channel thermal.
    """
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        # tifffile is more reliable than cv2 for 16-bit, especially on macOS.
        import tifffile

        arr = tifffile.imread(str(path))
    else:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise IOError(f"Failed to read image: {path}")

    if arr.ndim == 3:
        # Defensive: shouldn't happen for LWIR but average rather than crash.
        arr = arr.mean(axis=-1)

    if arr.dtype == np.uint16:
        scale = 65535.0
    elif arr.dtype == np.uint8:
        scale = 255.0
    else:
        # Float images are assumed already normalised.
        scale = 1.0
    return arr.astype(np.float32) / scale


def compute_stats(
    data_root: Path = DEFAULT_DATA_ROOT,
    split_path: Path = DEFAULT_SPLIT_PATH,
    out_path: Path = DEFAULT_STATS_PATH,
) -> dict[str, float]:
    """Compute training-set mean/std and write them to ``out_path``.

    Args:
        data_root: Path containing the ``data/`` images directory.
        split_path: Path to the split JSON written by ``src.splits``.
        out_path: Where to cache the resulting statistics.

    Returns:
        ``{"mean": float, "std": float, "n_images": int, "n_pixels": int}``.

    Raises:
        FileNotFoundError: If ``split_path`` doesn't exist.
        RuntimeError: If the training split is empty.
    """
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. Run src.splits first."
        )

    splits = load_splits(split_path)
    train_files = splits.get("train", [])
    if not train_files:
        raise RuntimeError("Training split is empty.")

    image_dir = data_root / "data"
    logger.info("Computing mean/std over %d training images", len(train_files))

    # Streaming accumulators in float64 for numerical stability.
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    skipped: list[str] = []

    for name in tqdm(train_files, desc="stats", unit="img"):
        path = image_dir / name
        try:
            arr = _load_image_normalised(path).astype(np.float64)
        except (IOError, OSError) as e:
            logger.warning("Skipping %s: %s", path, e)
            skipped.append(name)
            continue
        total_sum += float(arr.sum())
        total_sq_sum += float((arr * arr).sum())
        total_count += int(arr.size)

    if total_count == 0:
        raise RuntimeError("No pixels accumulated — every training image failed to load.")

    mean = total_sum / total_count
    # Var = E[X^2] - E[X]^2; clamp to >= 0 in case of tiny float drift.
    variance = max(total_sq_sum / total_count - mean * mean, 0.0)
    std = float(np.sqrt(variance))

    payload = {
        "mean": float(mean),
        "std": std,
        "n_images": len(train_files) - len(skipped),
        "n_pixels": total_count,
        "skipped": skipped,
        "split_path": str(split_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    logger.info("mean = %.6f", mean)
    logger.info("std  = %.6f", std)
    if skipped:
        logger.warning("Skipped %d unreadable images.", len(skipped))
    logger.info("Wrote stats to %s", out_path)
    return payload


def load_stats(stats_path: Path = DEFAULT_STATS_PATH) -> tuple[float, float]:
    """Return ``(mean, std)`` from the cached stats file.

    Args:
        stats_path: Path to the JSON file written by :func:`compute_stats`.

    Returns:
        ``(mean, std)`` as floats, ready to feed into normalisation.
    """
    payload = json.loads(Path(stats_path).read_text())
    return float(payload["mean"]), float(payload["std"])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT_PATH)
    p.add_argument("--out", type=Path, default=DEFAULT_STATS_PATH)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    compute_stats(args.data_root, args.split, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
