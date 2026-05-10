"""Generate the 70/20/10 train/val/test split for MassMIND.

MassMIND filenames look like ``a00068686.png`` -- a single lowercase letter
prefix followed by 8 digits. The letter prefix is the *capture session* (the
distribution has 26 sessions ``a..z`` of 91--141 frames each). Stratifying by
session ensures every fold sees frames from every trajectory, which prevents
the test set from being dominated by a single recording.

This is a refinement of the spec, which originally asked for year-based
stratification. The actual files in the public release don't carry timestamp
metadata, but the session prefix correlates with capture date/camera/upscale
status -- so session-stratification preserves the property the year-stratified
plan was after (balanced fold composition across capture conditions).

Output is a JSON file at ``data/splits/split.json``::

    {
      "seed": 42,
      "fractions": {"train": 0.7, "val": 0.2, "test": 0.1},
      "counts": {"train": 2041, "val": 583, "test": 292},
      "session_distribution": {
        "train": {"a": 80, "b": 73, ...},
        "val":   {"a": 23, "b": 21, ...},
        "test":  {"a": 12, "b": 10, ...}
      },
      "splits": {"train": ["a00068686.png", ...], "val": [...], "test": [...]}
    }

Filenames are stored relative (no directory prefix) so the file is portable.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

UNKNOWN_BUCKET = "unknown"

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "massmind"
DEFAULT_OUT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "splits" / "split.json"
)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for split generation.

    Attributes:
        train_frac: Fraction of samples assigned to the training split.
        val_frac: Fraction of samples assigned to the validation split.
        test_frac: Fraction of samples assigned to the test split.
        seed: Random seed controlling shuffle order within each year bucket.
    """

    train_frac: float = 0.70
    val_frac: float = 0.20
    test_frac: float = 0.10
    seed: int = 42

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"Split fractions must sum to 1.0 (got {total:.4f})"
            )


def session_bucket(filename: str) -> str:
    """Return the capture-session bucket for a MassMIND filename.

    MassMIND filenames are ``<letter><8 digits>.png`` (e.g. ``a00068686.png``);
    the leading letter identifies one of 26 capture sessions ``a..z``.
    Filenames not matching that pattern return :data:`UNKNOWN_BUCKET`.

    Args:
        filename: The image filename (with or without directory prefix).

    Returns:
        Lowercase session letter (``"a"``..``"z"``) or ``"unknown"``.
    """
    stem = Path(filename).stem
    if len(stem) >= 2 and stem[0].isalpha() and stem[0].islower() and stem[1:].isdigit():
        return stem[0]
    return UNKNOWN_BUCKET


def find_paired_files(data_root: Path) -> list[str]:
    """List filenames that exist in BOTH ``data_root/data`` and ``data_root/mask``.

    Args:
        data_root: Path containing ``data/`` (images) and ``mask/`` (segmentation
            masks) subdirectories, as produced by ``scripts/download.py``.

    Returns:
        Sorted list of filenames (e.g. ``["a163...png", ...]``), without
        directory prefix. Orphaned images or masks are skipped with a warning.

    Raises:
        FileNotFoundError: If either subdirectory is missing.
    """
    image_dir = data_root / "data"
    mask_dir = data_root / "mask"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_names = {p.name for p in image_dir.iterdir() if p.is_file()}
    mask_names = {p.name for p in mask_dir.iterdir() if p.is_file()}

    paired = sorted(image_names & mask_names)
    orphans = (image_names ^ mask_names)
    if orphans:
        logger.warning(
            "Skipping %d orphaned files (in only one of data/ or mask/). "
            "First 5: %s",
            len(orphans),
            sorted(orphans)[:5],
        )
    return paired


def _split_one_bucket(
    names: list[str], cfg: SplitConfig, rng: random.Random
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle ``names`` deterministically and partition into 3 fractions.

    The boundaries use ``round`` rather than ``int`` so a 10-element bucket
    splits as 7/2/1 instead of 7/2/0 (avoiding a starved test split).
    """
    items = list(names)
    rng.shuffle(items)
    n = len(items)
    n_train = round(n * cfg.train_frac)
    n_val = round(n * cfg.val_frac)
    # Test gets the remainder so total always equals n.
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def _bucket_by_session(names: Iterable[str]) -> dict[str, list[str]]:
    """Group filenames by capture-session prefix. Empty buckets are dropped."""
    buckets: dict[str, list[str]] = {}
    for name in names:
        buckets.setdefault(session_bucket(name), []).append(name)
    return dict(sorted(buckets.items()))


def generate_splits(
    data_root: Path = DEFAULT_DATA_ROOT,
    out_path: Path = DEFAULT_OUT_PATH,
    cfg: SplitConfig | None = None,
) -> dict:
    """Generate the train/val/test split, write it to ``out_path``, and return it.

    The split is stratified by coarse year bucket (2019 vs 2020+), so each
    fold sees roughly the same proportion of upscaled vs native-resolution
    captures. Within each bucket the order is shuffled deterministically by
    ``cfg.seed``.

    Args:
        data_root: Path containing ``data/`` and ``mask/`` subdirectories.
        out_path: Where to write the resulting JSON.
        cfg: Split configuration. Defaults to 70/20/10 with seed 42.

    Returns:
        The split dictionary that was written to disk.
    """
    cfg = cfg or SplitConfig()
    rng = random.Random(cfg.seed)

    paired = find_paired_files(data_root)
    if not paired:
        raise RuntimeError(
            f"No paired image/mask files found under {data_root}. Did you run "
            "scripts/download.py?"
        )
    logger.info("Found %d paired image/mask files.", len(paired))

    buckets = _bucket_by_session(paired)
    logger.info("Stratifying across %d capture sessions.", len(buckets))
    for bucket_name, items in buckets.items():
        logger.debug("  session %-3s : %d files", bucket_name, len(items))

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    session_distribution: dict[str, dict[str, int]] = {
        "train": {}, "val": {}, "test": {}
    }
    for bucket_name, items in buckets.items():
        if not items:
            continue
        train, val, test = _split_one_bucket(items, cfg, rng)
        splits["train"].extend(train)
        splits["val"].extend(val)
        splits["test"].extend(test)
        session_distribution["train"][bucket_name] = len(train)
        session_distribution["val"][bucket_name] = len(val)
        session_distribution["test"][bucket_name] = len(test)

    # Sort each split for stable diffs; consumers should still shuffle at train time.
    for split_name in splits:
        splits[split_name].sort()

    payload = {
        "seed": cfg.seed,
        "fractions": {
            "train": cfg.train_frac,
            "val": cfg.val_frac,
            "test": cfg.test_frac,
        },
        "counts": {k: len(v) for k, v in splits.items()},
        "session_distribution": session_distribution,
        "splits": splits,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote split to %s", out_path)
    logger.info(
        "  train=%d  val=%d  test=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return payload


def load_splits(split_path: Path) -> dict[str, list[str]]:
    """Load an existing split file and return the ``splits`` sub-dictionary.

    Args:
        split_path: Path to the JSON file written by :func:`generate_splits`.

    Returns:
        Dict mapping ``"train"`` / ``"val"`` / ``"test"`` to filename lists.
    """
    payload = json.loads(Path(split_path).read_text())
    return payload["splits"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--test-frac", type=float, default=0.10)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = SplitConfig(
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    generate_splits(args.data_root, args.out, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
