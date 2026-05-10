"""PyTorch ``Dataset`` for MassMIND LWIR semantic segmentation.

Each item is a ``dict`` with two keys:

    image : torch.FloatTensor of shape [1, H, W], normalised single-channel.
    mask  : torch.LongTensor  of shape [H, W], integer class IDs in [0, 6]
            (with possible 255 sentinels at borders introduced by augmentation).

Image bit depth is detected at load time:

    8-bit  -> divide by 255
    16-bit -> divide by 65535

Pairing relies on filenames matching exactly between ``data/`` and ``mask/``.
The dataset is constructed from a list of filenames (typically a slice of a
split JSON) -- *not* by globbing the directory at runtime, so the training run
is reproducible even if the user adds files to the directory later.

Corrupted files are skipped at ``__getitem__`` time by recursively returning
the next sample, with a warning. This is a deliberate tradeoff: training keeps
moving instead of crashing mid-epoch. ``__len__`` still reports the requested
size; the sample-skipping only matters in the rare corrupted-file case.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Class IDs valid in MassMIND masks (Sky..Self). 255 is the augmentation
# ignore-index. Anything else suggests a label mapping bug.
NUM_CLASSES: int = 7
VALID_CLASS_IDS: frozenset[int] = frozenset(range(NUM_CLASSES))


def _load_image(path: Path) -> np.ndarray:
    """Load an image to a 2D float32 array in [0, 1].

    Handles 8-bit PNG and 16-bit TIFF/PNG. Multichannel inputs (which
    shouldn't occur for LWIR) are averaged to a single channel.
    """
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(str(path))
    else:
        import cv2

        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise IOError(f"cv2.imread returned None for {path}")

    if arr.ndim == 3:
        arr = arr.mean(axis=-1)

    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _load_mask(path: Path) -> np.ndarray:
    """Load a mask to a 2D int64 array of class IDs.

    Masks are stored as 8-bit PNG with pixel values directly equal to class IDs.
    We read with ``IMREAD_UNCHANGED`` to preserve those values exactly (no
    accidental BGR conversion or alpha stripping).
    """
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"cv2.imread returned None for {path}")
    if arr.ndim == 3:
        # Some tools save masks as 3-channel even when only one channel is meaningful.
        arr = arr[..., 0]
    return arr.astype(np.int64)


class MassMINDDataset(Dataset):
    """MassMIND LWIR thermal semantic segmentation dataset.

    Args:
        data_root: Directory containing ``data/`` (images) and ``mask/``
            (semantic masks) subdirectories, as produced by
            ``scripts/download.py``.
        filenames: List of filenames (e.g. ``["a163...png", ...]``) that
            constitute this split. Each filename must exist in BOTH
            ``data_root/data/`` and ``data_root/mask/``.
        transform: An ``albumentations.Compose`` taking ``image=`` and
            ``mask=`` keyword arguments. Use :func:`src.augmentations.build_pipeline`
            to construct one.
        validate_classes: If True, raise on any mask containing a pixel value
            outside ``[0, NUM_CLASSES) | {255}``. Useful for sanity-checking
            during development; turn off for fast training iteration.

    Raises:
        FileNotFoundError: If ``data_root/data`` or ``data_root/mask`` is missing.
    """

    def __init__(
        self,
        data_root: Path,
        filenames: list[str],
        transform: A.Compose,
        validate_classes: bool = False,
    ) -> None:
        self.image_dir = Path(data_root) / "data"
        self.mask_dir = Path(data_root) / "mask"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        self.filenames = list(filenames)
        self.transform = transform
        self.validate_classes = validate_classes
        self._skipped: set[str] = set()

    def __len__(self) -> int:
        return len(self.filenames)

    def _read_pair(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        image = _load_image(self.image_dir / name)
        mask = _load_mask(self.mask_dir / name)
        if image.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch for {name}: image {image.shape} vs mask {mask.shape}"
            )
        if self.validate_classes:
            unique = set(int(v) for v in np.unique(mask))
            invalid = unique - VALID_CLASS_IDS - {255}
            if invalid:
                raise ValueError(
                    f"Mask {name} contains invalid class IDs: {sorted(invalid)}. "
                    f"Expected values in {sorted(VALID_CLASS_IDS)}."
                )
        return image, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        name = self.filenames[idx]
        try:
            image, mask = self._read_pair(name)
        except (IOError, OSError, ValueError) as e:
            if name not in self._skipped:
                logger.warning("Skipping %s (%s); using next sample instead.", name, e)
                self._skipped.add(name)
            # Move on to the next sample. Wraps around so __len__ stays honest.
            return self.__getitem__((idx + 1) % len(self.filenames))

        out = self.transform(image=image, mask=mask)
        image_t: torch.Tensor = out["image"]   # [1, H, W] float
        mask_t: torch.Tensor = out["mask"]      # [H, W] long (or whatever dtype)

        # Albumentations preserves the mask dtype; force long for CrossEntropy.
        if mask_t.dtype != torch.long:
            mask_t = mask_t.long()

        # ToTensorV2 produces [C, H, W] for HWC inputs; LWIR is 2D HW so we get
        # a [H, W] image tensor. Add the channel dim to match the [1, H, W] spec.
        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)

        return {"image": image_t, "mask": mask_t, "filename": name}


def make_dataset(
    data_root: Path,
    split_files: list[str],
    transform: A.Compose,
    validate_classes: bool = False,
) -> MassMINDDataset:
    """Factory wrapping the constructor; thin convenience for trainer code."""
    return MassMINDDataset(
        data_root=data_root,
        filenames=split_files,
        transform=transform,
        validate_classes=validate_classes,
    )


def make_collate_fn() -> Callable[[list[dict]], dict]:
    """Default collate that stacks images/masks and lists filenames.

    PyTorch's default collate would try to stack the filename strings into a
    tuple of tuples, which is fine but noisy; this version produces clean
    ``[B, ...]`` tensors plus a list of names.
    """

    def collate(batch: list[dict]) -> dict:
        images = torch.stack([b["image"] for b in batch], dim=0)
        masks = torch.stack([b["mask"] for b in batch], dim=0)
        names = [b["filename"] for b in batch]
        return {"image": images, "mask": masks, "filenames": names}

    return collate
