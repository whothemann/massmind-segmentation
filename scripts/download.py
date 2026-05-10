"""Download the MassMIND LWIR dataset from Google Drive.

The MassMIND authors host the data as two zip archives on Google Drive
(linked from the upstream repo's README):

    images : 1T572f0oqy5JmuTvVEwkSUeXLWOSHl4hL
    masks  : 1pHp480_Q-s72RoDf1nD7ERzsv9yZTDE1

We use ``gdown`` to handle Google Drive's confirm-token redirect for large files.
The script is idempotent: if the target directory already contains the expected
number of files, it skips the download.

Usage::

    python scripts/download.py
    python scripts/download.py --root /custom/path --force
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Google Drive file IDs from https://github.com/uml-marine-robotics/MassMIND
GDRIVE_IDS: dict[str, str] = {
    "images.zip": "1T572f0oqy5JmuTvVEwkSUeXLWOSHl4hL",
    "masks.zip": "1pHp480_Q-s72RoDf1nD7ERzsv9yZTDE1",
}

# Per the MassMIND paper: 2,916 LWIR images with paired masks.
EXPECTED_COUNT = 2916

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "massmind"


def _count_images(directory: Path) -> int:
    """Count image files (png/tif/tiff) recursively beneath ``directory``."""
    if not directory.exists():
        return 0
    exts = {".png", ".tif", ".tiff"}
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in exts)


def _download_one(file_id: str, dest: Path) -> None:
    """Download a single Google Drive archive to ``dest`` via gdown."""
    try:
        import gdown
    except ImportError as e:
        raise SystemExit(
            "gdown is required for the download. Install with: pip install gdown"
        ) from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading drive id=%s -> %s", file_id, dest)
    # gdown >=6 takes ``id=`` directly; resume=True picks up partial downloads.
    gdown.download(id=file_id, output=str(dest), quiet=False, resume=True)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(
            f"Download failed or produced an empty file: {dest}. "
            f"Verify https://drive.google.com/file/d/{file_id}/view is still public."
        )


def _extract(archive: Path, target: Path) -> None:
    """Extract a zip archive to ``target``. Creates ``target`` if missing."""
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s -> %s", archive.name, target)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)


def _flatten_single_top_dir(root: Path, expected_name: str) -> None:
    """If extraction produced ``root/<single_dir>/...``, move contents up to ``root``.

    Some archives wrap everything inside a top-level folder; we want a flat layout
    of ``root/*.png`` so that ``data/massmind/data/`` and ``data/massmind/mask/``
    sit at predictable locations.
    """
    entries = [p for p in root.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        if inner.name != expected_name:
            logger.debug("Flattening %s -> %s", inner, root)
            for child in inner.iterdir():
                shutil.move(str(child), str(root / child.name))
            inner.rmdir()


def _verify(image_dir: Path, mask_dir: Path) -> None:
    """Sanity-check that paired images and masks landed in the expected places."""
    n_img = _count_images(image_dir)
    n_msk = _count_images(mask_dir)
    logger.info("Found %d images in %s", n_img, image_dir)
    logger.info("Found %d masks  in %s", n_msk, mask_dir)
    if n_img == 0 or n_msk == 0:
        raise RuntimeError(
            "Extraction produced no images. Inspect "
            f"{image_dir.parent} to see what was actually unpacked."
        )
    if n_img != n_msk:
        logger.warning(
            "Image/mask count mismatch: %d vs %d. Pairing logic in dataset.py "
            "will skip orphaned files, but you should investigate.",
            n_img,
            n_msk,
        )
    if n_img != EXPECTED_COUNT:
        logger.warning(
            "Expected %d images per the paper, found %d. The split file will "
            "still work but downstream stats will not exactly match Table 5.",
            EXPECTED_COUNT,
            n_img,
        )


def _already_present(image_dir: Path, mask_dir: Path) -> bool:
    return (
        _count_images(image_dir) >= EXPECTED_COUNT
        and _count_images(mask_dir) >= EXPECTED_COUNT
    )


def download_massmind(root: Path = DEFAULT_ROOT, force: bool = False) -> None:
    """Download and extract the MassMIND dataset under ``root``.

    Args:
        root: Target directory. Will contain ``data/`` (images) and ``mask/``
            (semantic masks) after extraction.
        force: If True, redownload and re-extract even if files already exist.
    """
    root = root.resolve()
    image_dir = root / "data"
    mask_dir = root / "mask"

    if not force and _already_present(image_dir, mask_dir):
        logger.info("Dataset already present under %s — skipping download.", root)
        return

    archive_dir = root / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for archive_name, file_id in GDRIVE_IDS.items():
        archive_path = archive_dir / archive_name
        if force or not archive_path.exists() or archive_path.stat().st_size == 0:
            _download_one(file_id, archive_path)
        else:
            logger.info("Archive already downloaded: %s", archive_path)

    _extract(archive_dir / "images.zip", image_dir)
    _extract(archive_dir / "masks.zip", mask_dir)

    # Some archives nest one level deep — normalise that.
    _flatten_single_top_dir(image_dir, expected_name="data")
    _flatten_single_top_dir(mask_dir, expected_name="mask")

    _verify(image_dir, mask_dir)
    logger.info("Done. Dataset is at %s", root)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Target directory (default: {DEFAULT_ROOT}).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract even if files already exist.",
    )
    p.add_argument(
        "--keep-archives",
        action="store_true",
        help="Do not delete the downloaded zip archives after extraction.",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    download_massmind(args.root, force=args.force)
    if not args.keep_archives:
        archive_dir = args.root / "_archives"
        if archive_dir.exists():
            logger.info("Removing archive directory %s", archive_dir)
            shutil.rmtree(archive_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
