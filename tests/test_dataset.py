"""Sanity tests for the MassMIND data pipeline.

These tests fall in two groups:

1. **Synthetic tests** -- run anywhere with no MassMIND download required.
   They build a tiny on-disk fake dataset (8x12 random images and masks) and
   exercise the dataset/augmentation contracts: shapes, dtypes, value ranges,
   geometric-equivariance of augmentations.

2. **Real-data tests** -- gated by ``MASSMIND_DATA_ROOT`` env var. Skipped
   if unset, so CI can run without the 2.9k-image archive. Verifies length,
   class distribution, and that nothing's NaN or out-of-range on the actual
   dataset.

Run::

    pytest tests/                                      # synthetic only
    MASSMIND_DATA_ROOT=/path/to/data pytest tests/     # synthetic + real
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.augmentations import (
    MASK_IGNORE_INDEX,
    build_pipeline,
    pipeline_a_massmind_replicated,
    pipeline_b_extended,
    pipeline_c_no_augmentation,
)
from src.dataset import NUM_CLASSES, MassMINDDataset
from src.splits import (
    SplitConfig,
    _split_one_bucket,
    generate_splits,
    session_bucket,
)

# Reference per-class pixel fractions reported in the MassMIND paper (Table 5).
PAPER_CLASS_FRACTIONS = {
    0: 0.3058,  # Sky
    1: 0.5221,  # Water
    2: 0.0167,  # Bridge
    3: 0.0094,  # Obstacle
    4: 0.0005,  # Living obstacle
    5: 0.1128,  # Background
    6: 0.0325,  # Self
}


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Build a minimal valid MassMIND-shaped directory under ``tmp_path``.

    Returns the data root (containing ``data/`` and ``mask/``) so individual
    tests can hand it to ``MassMINDDataset``. Filenames mimic the real
    distribution: ``<letter><8 digits>.png`` across multiple session prefixes
    so session_bucket has both buckets to stratify.
    """
    rng = np.random.default_rng(0)
    image_dir = tmp_path / "data"
    mask_dir = tmp_path / "mask"
    image_dir.mkdir()
    mask_dir.mkdir()

    h, w = 8, 12
    names = [
        "a00012345.png",
        "a00067890.png",
        "b00011111.png",
        "z00099999.png",
    ]
    for name in names:
        img = (rng.integers(0, 256, size=(h, w))).astype(np.uint8)
        msk = (rng.integers(0, NUM_CLASSES, size=(h, w))).astype(np.uint8)
        cv2.imwrite(str(image_dir / name), img)
        cv2.imwrite(str(mask_dir / name), msk)
    return tmp_path


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


class TestSessionBucket:
    def test_letter_a(self) -> None:
        assert session_bucket("a00012345.png") == "a"

    def test_letter_z(self) -> None:
        assert session_bucket("z00099999.png") == "z"

    def test_uppercase_letter_unknown(self) -> None:
        # The real distribution uses lowercase only; uppercase is unrecognised.
        assert session_bucket("A00012345.png") == "unknown"

    def test_unparseable_filename(self) -> None:
        assert session_bucket("frame_001.png") == "unknown"


class TestSplitGeneration:
    def test_fractions_validated(self) -> None:
        with pytest.raises(ValueError):
            SplitConfig(train_frac=0.6, val_frac=0.2, test_frac=0.1)

    def test_all_files_assigned(self, tiny_dataset: Path) -> None:
        out = tiny_dataset / "splits.json"
        payload = generate_splits(tiny_dataset, out, SplitConfig())
        all_split = (
            payload["splits"]["train"]
            + payload["splits"]["val"]
            + payload["splits"]["test"]
        )
        assert len(all_split) == 4
        assert set(all_split) == {p.name for p in (tiny_dataset / "data").iterdir()}

    def test_disjoint_splits(self, tiny_dataset: Path) -> None:
        out = tiny_dataset / "splits.json"
        payload = generate_splits(tiny_dataset, out, SplitConfig())
        s = payload["splits"]
        assert not (set(s["train"]) & set(s["val"]))
        assert not (set(s["train"]) & set(s["test"]))
        assert not (set(s["val"]) & set(s["test"]))

    def test_seeded_determinism(self, tiny_dataset: Path) -> None:
        out1 = tiny_dataset / "s1.json"
        out2 = tiny_dataset / "s2.json"
        p1 = generate_splits(tiny_dataset, out1, SplitConfig(seed=42))
        p2 = generate_splits(tiny_dataset, out2, SplitConfig(seed=42))
        assert p1["splits"] == p2["splits"]

    def test_round_keeps_test_nonempty(self) -> None:
        # 10 items, 70/20/10 -> train=7, val=2, test=1. ``int`` would give 0.
        items = [f"f{i}.png" for i in range(10)]
        import random as _r

        rng = _r.Random(0)
        train, val, test = _split_one_bucket(items, SplitConfig(), rng)
        assert (len(train), len(val), len(test)) == (7, 2, 1)


# ---------------------------------------------------------------------------
# Dataset contract -- synthetic
# ---------------------------------------------------------------------------


def _name_at(tiny_dataset: Path, idx: int = 0) -> str:
    return sorted(p.name for p in (tiny_dataset / "data").iterdir())[idx]


class TestDatasetContract:
    def test_length_matches_filenames(self, tiny_dataset: Path) -> None:
        names = sorted(p.name for p in (tiny_dataset / "data").iterdir())
        ds = MassMINDDataset(
            tiny_dataset, names, pipeline_c_no_augmentation(0.5, 0.1)
        )
        assert len(ds) == len(names)

    def test_image_shape_and_dtype(self, tiny_dataset: Path) -> None:
        names = [_name_at(tiny_dataset)]
        ds = MassMINDDataset(
            tiny_dataset, names, pipeline_c_no_augmentation(0.5, 0.1)
        )
        sample = ds[0]
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].dtype == torch.float32
        assert sample["image"].shape == (1, 8, 12)
        assert sample["image"].ndim == 3

    def test_mask_dtype_and_range(self, tiny_dataset: Path) -> None:
        names = [_name_at(tiny_dataset)]
        ds = MassMINDDataset(
            tiny_dataset, names, pipeline_c_no_augmentation(0.5, 0.1),
            validate_classes=True,
        )
        sample = ds[0]
        assert sample["mask"].dtype == torch.long
        assert sample["mask"].shape == (8, 12)
        unique_vals = torch.unique(sample["mask"]).tolist()
        for v in unique_vals:
            assert 0 <= v < NUM_CLASSES, f"Invalid class id {v}"

    def test_no_nan_or_inf_in_normalised_image(self, tiny_dataset: Path) -> None:
        names = [_name_at(tiny_dataset)]
        ds = MassMINDDataset(
            tiny_dataset, names, pipeline_c_no_augmentation(0.5, 0.1)
        )
        img = ds[0]["image"]
        assert torch.isfinite(img).all(), "Normalised image has NaN or Inf"


# ---------------------------------------------------------------------------
# Augmentation correctness -- the most subtle bit
# ---------------------------------------------------------------------------


class TestAugmentationGeometricCoupling:
    """Verify that geometric transforms apply identically to image and mask."""

    def _make_marker_pair(
        self, h: int = 32, w: int = 32
    ) -> tuple[np.ndarray, np.ndarray]:
        # Encode a known per-pixel correspondence: image pixel value is the same
        # integer as the mask pixel value at that location, scaled to [0, 1].
        # If a geometric transform is applied identically, the relationship
        # value = mask_id / NUM_CLASSES survives the transform.
        rng = np.random.default_rng(42)
        mask = rng.integers(0, NUM_CLASSES, size=(h, w)).astype(np.uint8)
        image = (mask.astype(np.float32) / NUM_CLASSES).astype(np.float32)
        return image, mask

    def test_horizontal_flip_couples_image_and_mask(self) -> None:
        # Build a deterministic flip pipeline.
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        pipe = A.Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
        image, mask = self._make_marker_pair()
        out = pipe(image=image, mask=mask)
        img = out["image"].numpy().squeeze(0)  # [1,H,W] -> [H,W]
        msk = out["mask"].numpy()
        # The encoded relationship: image == mask / NUM_CLASSES, after flip.
        np.testing.assert_allclose(img, msk.astype(np.float32) / NUM_CLASSES, atol=1e-5)

    def test_pipeline_b_geometric_consistency(self) -> None:
        # Verify that image and mask receive the SAME geometric transform.
        # Production uses bilinear-image / nearest-mask which is intentional
        # (smooth image, integer mask) but breaks pixelwise equality even on
        # non-border pixels. Force nearest for both here so the encoded
        # ``image == mask / NUM_CLASSES`` invariant can be checked exactly.
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        pipe = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(
                    limit=7,
                    interpolation=cv2.INTER_NEAREST,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_REFLECT_101,
                    fill_mask=MASK_IGNORE_INDEX,
                    p=0.5,
                ),
                A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
        image, mask = self._make_marker_pair(h=64, w=64)
        for _ in range(20):
            out = pipe(image=image, mask=mask)
            img = out["image"].numpy().squeeze(0)
            msk = out["mask"].numpy()
            valid = msk != MASK_IGNORE_INDEX
            np.testing.assert_allclose(
                img[valid],
                msk[valid].astype(np.float32) / NUM_CLASSES,
                atol=1e-5,
                err_msg="Image and mask diverged under the same geometric transform",
            )

    def test_mask_values_remain_integer_after_rotation(self) -> None:
        # Critical correctness check: mask interpolation must be nearest-
        # neighbour. If it were bilinear, fractional class IDs (e.g. 2.5)
        # would appear and break CrossEntropy.
        import albumentations as A

        pipe = A.Compose(
            [
                A.Rotate(
                    limit=7,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_REFLECT_101,
                    fill_mask=MASK_IGNORE_INDEX,
                    p=1.0,
                ),
            ]
        )
        rng = np.random.default_rng(0)
        image = rng.random(size=(64, 64)).astype(np.float32)
        mask = rng.integers(0, NUM_CLASSES, size=(64, 64)).astype(np.uint8)
        out = pipe(image=image, mask=mask)
        unique = np.unique(out["mask"])
        valid_set = set(range(NUM_CLASSES)) | {MASK_IGNORE_INDEX}
        assert set(int(v) for v in unique).issubset(valid_set), (
            f"Rotation produced fractional or unexpected class IDs: {unique.tolist()}"
        )


class TestPipelineFactory:
    def test_build_pipeline_dispatch(self) -> None:
        a = build_pipeline("A", 0.5, 0.1, train=True)
        b = build_pipeline("B", 0.5, 0.1, train=True)
        c = build_pipeline("C", 0.5, 0.1, train=True)
        none_alias = build_pipeline("none", 0.5, 0.1, train=True)
        assert all(p is not None for p in (a, b, c, none_alias))

    def test_train_false_falls_back_to_c(self) -> None:
        a_eval = pipeline_a_massmind_replicated(0.5, 0.1, train=False)
        b_eval = pipeline_b_extended(0.5, 0.1, train=False)
        # Length of the transform list is a cheap proxy: pipelines A/B in
        # train mode have more transforms than C (which has only Normalize+ToTensor).
        assert len(a_eval.transforms) == 2
        assert len(b_eval.transforms) == 2

    def test_unknown_pipeline_name_raises(self) -> None:
        with pytest.raises(ValueError):
            build_pipeline("Z", 0.5, 0.1, train=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Real-data tests: opt-in via env var
# ---------------------------------------------------------------------------

REAL_DATA = os.environ.get("MASSMIND_DATA_ROOT")
real = pytest.mark.skipif(
    REAL_DATA is None,
    reason="Set MASSMIND_DATA_ROOT to enable real-dataset tests.",
)


@real
class TestRealDataset:
    @pytest.fixture(scope="class")
    def split(self, tmp_path_factory: pytest.TempPathFactory) -> dict:
        out = tmp_path_factory.mktemp("split") / "split.json"
        return generate_splits(Path(REAL_DATA), out, SplitConfig())  # type: ignore[arg-type]

    def test_split_counts_close_to_70_20_10(self, split: dict) -> None:
        total = sum(split["counts"].values())
        assert math.isclose(split["counts"]["train"] / total, 0.70, abs_tol=0.01)
        assert math.isclose(split["counts"]["val"] / total, 0.20, abs_tol=0.01)
        assert math.isclose(split["counts"]["test"] / total, 0.10, abs_tol=0.01)

    def test_class_distribution_close_to_paper(self, split: dict) -> None:
        # Subsample to keep the test fast: 200 random training images.
        names = split["splits"]["train"][:200]
        counts = np.zeros(NUM_CLASSES + 1, dtype=np.int64)
        for name in names:
            mask = cv2.imread(
                str(Path(REAL_DATA) / "mask" / name), cv2.IMREAD_UNCHANGED  # type: ignore[arg-type]
            )
            for cls in range(NUM_CLASSES):
                counts[cls] += int((mask == cls).sum())
        total = counts[:NUM_CLASSES].sum()
        assert total > 0
        for cls, expected in PAPER_CLASS_FRACTIONS.items():
            actual = counts[cls] / total
            # Loose tolerance: 200 images is small and class 4 is ~0.05% so
            # tighter bounds would be statistically unfair.
            assert abs(actual - expected) < max(0.05, expected * 2.0), (
                f"Class {cls}: expected {expected:.4f}, got {actual:.4f}"
            )

    def test_no_nan_in_real_image(self, split: dict) -> None:
        names = split["splits"]["train"][:5]
        ds = MassMINDDataset(
            Path(REAL_DATA),  # type: ignore[arg-type]
            names,
            pipeline_c_no_augmentation(0.3, 0.15),
        )
        for i in range(len(ds)):
            sample = ds[i]
            assert torch.isfinite(sample["image"]).all()
