"""Albumentations augmentation pipelines for MassMIND.

Three pipelines, all applied on-the-fly:

    Pipeline A ("MassMIND-replicated"):
        Mirrors the augmentation choices from Nirgudkar et al. 2023 (Sec. 5.1,
        5.3). Used for the assignment-required "with augmentation" runs.

    Pipeline B ("Extended"):
        Pipeline A plus modern additions defensible from broader segmentation
        literature: random crop+resize, CLAHE, mild Gaussian noise.

    Pipeline C ("None"):
        Validation/test pipeline. No augmentation; only normalisation and
        tensor conversion.

Design choices baked into all three pipelines:

* Vertical flip is forbidden -- maritime physics has sky on top, water below.
* Brightness/contrast jitter is forbidden -- in LWIR the pixel intensity IS
  the class signal, so jittering it actively hurts.
* Mask interpolation is nearest-neighbour throughout. Albumentations defaults
  to that for ``mask=`` tensors, but we set it explicitly on every transform
  that takes a separate mask interpolation argument.
* Border padding uses ``BORDER_REFLECT_101`` for images and a constant fill
  value for masks. The constant must equal the number of classes (7), which
  the dataset and trainer can treat as ``ignore_index`` -- using class ID 0
  (Sky) would silently relabel border pixels as Sky.
* Normalisation operates on the [0, 1]-rescaled single-channel image using
  the training mean/std produced by ``src.stats``.

The pipelines return ``albumentations.Compose`` objects expecting
``image=`` and ``mask=`` keyword arguments and producing dict outputs with the
same keys -- matching what ``MassMINDDataset.__getitem__`` consumes.
"""
from __future__ import annotations

from typing import Literal

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# Sentinel value for mask border fill. MUST be outside the [0, 6] valid class
# range so the trainer can set ``ignore_index=MASK_IGNORE_INDEX`` without
# accidentally penalising real classes.
MASK_IGNORE_INDEX: int = 255

PipelineName = Literal["A", "B", "C", "none"]


def _final_tensor_ops(mean: float, std: float) -> list[A.BasicTransform]:
    """Normalisation + tensor conversion shared by every pipeline.

    ``A.Normalize(max_pixel_value=1.0)`` skips the implicit /255 scaling, since
    the dataset already normalises bit-depth-correctly to [0, 1] before
    augmentation.
    """
    return [
        A.Normalize(mean=(mean,), std=(std,), max_pixel_value=1.0),
        ToTensorV2(),  # image: HWC -> CHW float tensor; mask: long tensor unchanged.
    ]


def pipeline_a_massmind_replicated(
    mean: float, std: float, train: bool
) -> A.Compose:
    """Pipeline A: MassMIND-paper-replicated augmentation.

    Args:
        mean: Training-set mean (single-channel, in [0, 1] image space).
        std: Training-set std.
        train: If False, the function returns the validation/test pipeline
            (Pipeline C) regardless. Augmentation is only applied at training
            time.

    Returns:
        An ``A.Compose`` taking ``image=`` and ``mask=`` and returning a dict
        with the same keys.
    """
    if not train:
        return pipeline_c_no_augmentation(mean, std)

    transforms: list[A.BasicTransform] = [
        A.HorizontalFlip(p=0.5),
        A.Rotate(
            limit=7,  # +/- 7 degrees, per Nirgudkar et al.
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_REFLECT_101,
            fill_mask=MASK_IGNORE_INDEX,
            crop_border=False,
            p=0.5,
        ),
        *_final_tensor_ops(mean, std),
    ]
    return A.Compose(transforms)


def pipeline_b_extended(mean: float, std: float, train: bool) -> A.Compose:
    """Pipeline B: Pipeline A plus modern additions.

    Adds random crop+resize (geometric), CLAHE (per-image local contrast
    normalisation, defensible for thermal), and mild Gaussian noise (sensor
    variation simulation). All pixel-level transforms are image-only by
    construction in albumentations.

    Args:
        mean: Training-set mean.
        std: Training-set std.
        train: If False, returns Pipeline C.

    Returns:
        An ``A.Compose`` taking ``image=`` and ``mask=``.
    """
    if not train:
        return pipeline_c_no_augmentation(mean, std)

    transforms: list[A.BasicTransform] = [
        A.HorizontalFlip(p=0.5),
        A.Rotate(
            limit=7,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_REFLECT_101,
            fill_mask=MASK_IGNORE_INDEX,
            crop_border=False,
            p=0.5,
        ),
        # Crop a 480x384 window then resize back to native 640x512.
        # Albumentations applies the crop+resize to image and mask identically
        # when both are passed; mask uses INTER_NEAREST automatically.
        A.OneOf(
            [
                A.Sequential(
                    [
                        A.RandomCrop(height=384, width=480, p=1.0),
                        A.Resize(
                            height=512,
                            width=640,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=1.0,
                ),
            ],
            p=0.5,
        ),
        # Pixel-level (image-only) below.
        A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), p=0.3),
        # std_range = sqrt(var_limit). Original spec: var_limit=(0.0, 0.005)
        # -> std_range = (0.0, ~0.0707). Per-channel doesn't matter for 1ch.
        A.GaussNoise(std_range=(0.0, 0.0707), mean_range=(0.0, 0.0), per_channel=False, p=0.2),
        *_final_tensor_ops(mean, std),
    ]
    return A.Compose(transforms)


def pipeline_c_no_augmentation(mean: float, std: float) -> A.Compose:
    """Pipeline C: validation/test pipeline. Normalise + to-tensor only.

    Args:
        mean: Training-set mean.
        std: Training-set std.

    Returns:
        An ``A.Compose`` that applies no spatial or photometric augmentation.
    """
    return A.Compose(_final_tensor_ops(mean, std))


def build_pipeline(
    name: PipelineName, mean: float, std: float, train: bool
) -> A.Compose:
    """Dispatch helper: build a pipeline by name.

    Args:
        name: One of ``"A"``, ``"B"``, ``"C"``, or ``"none"`` (alias for ``"C"``).
        mean: Training-set mean.
        std: Training-set std.
        train: Pipelines A and B fall back to Pipeline C when ``train`` is False.

    Returns:
        The corresponding ``A.Compose``.

    Raises:
        ValueError: If ``name`` isn't one of the supported pipeline keys.
    """
    if name == "A":
        return pipeline_a_massmind_replicated(mean, std, train)
    if name == "B":
        return pipeline_b_extended(mean, std, train)
    if name in ("C", "none"):
        return pipeline_c_no_augmentation(mean, std)
    raise ValueError(f"Unknown pipeline {name!r}; expected one of A, B, C, none.")
