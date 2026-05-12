# MassMIND Segmentation

Next step: implement focal loss (maybe focal loss and dice loss combined)

Semantic segmentation of long-wave infrared (LWIR) maritime imagery from the
[MassMIND dataset](https://github.com/uml-marine-robotics/MassMIND) (Nirgudkar
et al., 2023). Coursework for *Computer Vision — Assignment 2, FEUP 2025/26*.

The assignment asks for two things: **(1)** a custom segmentation architecture
proposed and defended by us, and **(2)** a comparison against at least one
existing model trained on the same data. We're delivering this as:

- **Custom model — "Tier 2 U-Net":** a hand-implemented 4-level U-Net (built on
  top of the in-class demonstrator) with three localised modifications: a
  transformer bottleneck, 1×1-conv skip refinement, and deep supervision via
  auxiliary heads.
- **Existing-model baseline:** U-Net with an ImageNet-pretrained VGG-16 encoder
  (Nirgudkar et al.'s strongest CNN baseline), trained on the same data with
  the same loss and metrics.

## Status

| Phase | State |
|---|---|
| Dataset download + 70/20/10 session-stratified split | ✅ |
| Pixel mean/std + per-class pixel counts | ✅ |
| U-Net + VGG-16 trainer (PyTorch; CUDA / MPS / CPU autodetect) | ✅ |
| Augmentation pipelines A / B / C | ✅ |
| Colab + Kaggle training notebook (live progress + Drive sync) | ✅ |
| Baseline U-Net + VGG-16 runs for A / B / C on Kaggle P100, 20 ep | ✅ — `runs/kaggle_extract/` |
| Tier 2 U-Net architecture (hand-implemented) | ⏳ design done, code pending |
| Dice + CE loss; AdamW + warmup-cosine schedule | ⏳ |
| 50-epoch experiment matrix (5 runs) | ⏳ |
| Threshold sweep (τ = 0.3 and 0.6) for rare-class F1 | ⏳ |
| Final writeup | ⏳ |

## Tier 2 architecture

A 4-level encoder/decoder U-Net (preserving the demonstrator's base
structure: double-conv blocks, max-pool downsampling, transposed-conv
upsampling, concatenative skip connections, 1×1 output conv) with three
deliberate, individually ablatable modifications.

```
Input (1×H×W, LWIR)
   │
   ▼  4× (double conv → max pool)
Enc1 → Enc2 → Enc3 → Enc4
                       │
                       ▼  [Mod 1] Transformer bottleneck (2 layers, 8 heads)
                  bottleneck
                       │
                       ▼  4× (upconv → concat with [Mod 2] 1×1-refined skip → double conv)
                 Dec4 → Dec3 → Dec2 → Dec1
                          │      │      │
                          ▼      ▼      ▼
                       [Mod 3 aux head]  [Mod 3 aux head]  Main head (1×1 conv → 7 classes)
                       (training only)   (training only)
```

### Mod 1 — Transformer bottleneck

Replaces the demonstrator's plain double-conv bottleneck. The encoder Stage-4
feature map is flattened into tokens, summed with learned 2-D positional
embeddings, and passed through **2 stacked `nn.TransformerEncoderLayer`
blocks** (8 attention heads, embed dim 512, FFN hidden 2048, dropout 0.1,
pre-norm), then reshaped back to spatial form for the decoder.

*Why here:* attention is global by construction but quadratic in tokens; at
the bottleneck the spatial grid is tiny (~1024 tokens), so it's cheap. The
MassMIND task has class-level scene structure (water spans the whole frame,
bridges are extended objects, rare classes need global context to be
distinguished from clutter) — exactly the kind of dependencies plain CNNs
struggle to capture through stacked 3×3 convolutions. Reference pattern:
TransUNet (Chen et al. 2021); attention mechanism: Vaswani et al. 2017.

Approx. parameter cost: ~4–6 M.

### Mod 2 — 1×1 conv refinement on skip connections

Before each encoder feature is concatenated with the decoder feature at the
same level, it passes through `Conv2d(C, C, 1) → BatchNorm2d → ReLU`. Channel
count is preserved; nothing is downsampled or expanded.

*Why:* the vanilla U-Net's skip connection forwards encoder features
unchanged — the decoder gets no learned control over what it receives. The
1×1 conv learns per-channel reweighting (a lightweight, parameter-cheap
analogue of Attention U-Net's gating). For LWIR this matters: the same
edge feature can be a boat hull in one image and a wave artefact in another.
Reference: Attention U-Net (Oktay et al. 2018), lightweight variant.

Approx. parameter cost: ~340 K across all four skip levels.

### Mod 3 — Deep supervision with auxiliary heads

Adds 1×1-conv segmentation heads at decoder levels 2 and 3 (each followed by
bilinear upsample to full resolution), each contributing its own Dice + CE
loss term. Total training loss:

```
L_total = L_main  +  0.4 · L_aux_dec3  +  0.2 · L_aux_dec2
```

At inference only the main head is used — the aux heads add **zero**
deployed parameters.

*Why:* gradient for the main output has to flow all the way through the
decoder back to the encoder; for `living_obs` at 0.05% of pixels, that
gradient is dominated by majority-class signal at every intermediate layer.
Auxiliary heads inject direct, full-class-distribution supervision into the
deeper decoder layers. Reference patterns: PSPNet (Zhao et al. 2017),
UNet++ (Zhou et al. 2018).

Approx. parameter cost: ~2 K (negligible).

### Demonstrator vs Tier 2 — at a glance

| Component | Demonstrator (class notebook) | Tier 2 (ours) |
|---|---|---|
| Encoder blocks | Plain double conv | Plain double conv |
| Bottleneck | Plain double conv | Transformer (2 layers, 8 heads) |
| Skip connections | Raw concatenation | 1×1 conv refinement |
| Output supervision | Single head | Main + 2 auxiliary heads |
| Approx. parameter count | ~31 M | ~33 M |
| Approx. model code | ~80 lines | ~250 lines |
| Pretrained weights | None (or optional VGG) | None — from scratch |
| Loss | Cross-entropy | Dice + Cross-entropy (0.5/0.5) |

## Training methodology

- **Loss** — `0.5 · CE + 0.5 · Dice`. CE provides stable pixel-wise gradients;
  Dice is invariant to class size, which is critical for the 0.05% /
  0.9% rare classes. Standard practice for class-imbalanced semantic
  segmentation.
- **Optimizer** — AdamW, `weight_decay = 1e-4`.
- **Schedule** — cosine annealing with **5-epoch linear warmup** from 0 to
  `lr = 1e-4`.
- **Epochs** — **50**, matching the MassMIND paper's protocol.
- **Batch size** — 8 on Colab T4, 16 on Kaggle P100.
- **From scratch** — no pretrained weights for Tier 2. Kaiming init for
  convs, Xavier for linears, zero-init for positional embeddings. (The
  baseline U-Net + VGG-16 model *does* use ImageNet weights, adapted from 3-
  to 1-channel by the channel-mean trick in `src/models/_adapt.py`.)
- **Augmentations** — three pipelines defined in `src/augmentations.py`; see
  table below.

## Experiment matrix

Five runs satisfying and extending the assignment requirements:

| # | Model | Augmentation | Loss | Purpose |
|---|-------|--------------|------|---------|
| 1 | Tier 2 | none (C) | Dice + CE | Assignment-required no-aug baseline |
| 2 | Tier 2 | A — MassMIND-replicated | Dice + CE | Assignment-required with-aug |
| 3 | Tier 2 | B — Extended | Dice + CE | Own extension of MassMIND paper |
| 4 | U-Net + VGG-16 | A — MassMIND-replicated | Dice + CE | Required comparison vs existing model |
| 5* | Tier 2 ablation | A | Dice + CE | Optional — isolate Mod 1 / 2 / 3 contributions |

\* Run 5 ablates the three modifications independently (transformer bottleneck
only, aux-heads only, skip-refinement only, full) so we can attribute IoU
gains to specific architectural choices.

## Project layout

```
massmind_segmentation/
├── data/
│   ├── massmind/                    # raw LWIR images + masks (downloaded, gitignored)
│   └── splits/
│       ├── split.json               # 70/20/10 train/val/test, session-stratified
│       ├── stats.json               # train-set pixel mean & std for normalisation
│       └── class_pixel_counts.json  # global pixel count per class
├── src/
│   ├── dataset.py                   # MassMINDDataset: bit-depth aware, robust to corrupt files
│   ├── splits.py                    # generate stratified split by capture session (a..z)
│   ├── stats.py                     # compute pixel mean/std + class pixel counts
│   ├── augmentations.py             # albumentations pipelines A / B / C
│   ├── metrics.py                   # ConfusionMatrixTracker → IoU, pixel acc
│   ├── train.py                     # single-file trainer (CE loss, AdamW, cosine LR)
│   └── models/
│       ├── unet.py                  # build_unet_vgg16 (SMP backbone) — baseline
│       ├── _adapt.py                # adapt first conv RGB 3-ch → LWIR 1-ch (baseline only)
│       └── tier2_unet.py            # ⏳ Tier 2 hand-built U-Net (pending implementation)
├── scripts/
│   └── download.py                  # idempotent Google-Drive download via gdown
├── notebooks/
│   ├── 01_data_exploration.ipynb    # class balance, image stats, sample renders
│   └── 02_train_colab.ipynb         # Colab/Kaggle driver: live tqdm + plots + Drive sync
├── kaggle_test_run_u_net.ipynb      # actually-run Kaggle copy with embedded outputs
├── runs/
│   └── kaggle_extract/              # outputs extracted from kaggle_test_run_u_net.ipynb:
│                                    #   per-epoch CSVs, full logs, training-curve PNGs,
│                                    #   comparison plot, summary table
├── tests/                           # unit tests: dataset, metrics, models
├── requirements.txt
└── .gitignore
```

## Quickstart

### Local laptop — smoke test

Validates the pipeline end-to-end on Mac (MPS) or CPU in ~1 minute.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/download.py        # ~2 min, ~500 MB; idempotent
python -m src.splits              # writes data/splits/split.json
python -m src.stats               # writes data/splits/stats.json + class_pixel_counts.json

python -m src.train \
    --augmentation A --epochs 1 --subset 30 \
    --output-dir runs/smoke_test
```

`src.train` autodetects device: CUDA → MPS → CPU. `--subset N` caps the
training set to the first N images for fast iteration.

### Colab / Kaggle — full runs

`notebooks/02_train_colab.ipynb` is the runtime driver and works on both
platforms. It:

1. Clones this repo (asks for a GitHub PAT once)
2. Installs deps, downloads the dataset, optionally mounts Google Drive
3. Defines `run_training(aug)` — launches `src.train` as a subprocess, streams
   stdout into a `tqdm` bar + live loss / mIoU plot, and rsyncs the run
   directory to Drive after every epoch (so a runtime crash doesn't wipe
   progress)
4. Has three separate cells (6a / 6b / 6c) so you can run A, B, C independently
5. Overlays training curves and prints a per-class IoU summary table

**Kaggle tips** (lessons learned the hard way):

- **Pick P100**, not T4 ×2. The trainer is FP32-only, so T4's Tensor-Core
  advantage is unused; P100 has higher memory bandwidth and wins on this
  workload. Kaggle's "T4" option is two cards, but the trainer is single-GPU.
- **Enable Internet** in the right-hand notebook settings (one-time phone
  verification on the Kaggle account).
- **Bump `NUM_WORKERS = 4`** in cell 14 — Kaggle gives ~4 vCPUs vs Colab
  free's 2.
- Save Version → Output tab → Download All for the run artefacts.

**Colab tips:**

- Free T4 has only 2 vCPUs, so `NUM_WORKERS = 2` is correct (default).
- Free GPU access is rate-limited; expect cooldowns after heavy use.
- Use `DRIVE_RUNS_DIR = '/content/drive/MyDrive/massmind_runs'` (not bare
  `/content/drive/...`, which is read-only).
- The keepalive cell helps with idle disconnects; pair it with a no-sleep
  laptop setting.

## How the pieces fit together

```
LWIR image (640×512, 8 or 16 bit)
        + 7-class mask                ──► augmentation (A | B | C)
                                          + normalise → tensor
                                                │
                                                ▼
                                ┌──────────────────────────────┐
                                │ Tier 2 U-Net   OR            │
                                │ U-Net + VGG-16 (baseline)    │
                                └──────────────────────────────┘
                                                │
                                                ▼ argmax  (or τ-threshold for rare classes)
                                       per-pixel class ID ∈ [0..6]
                                                │
                                                ▼
                          ConfusionMatrixTracker → mIoU, Precision, Recall, F1, per-class IoU
```

### Class scheme

| ID | Class | Pixel share | Notes |
|----|-------|-------------|-------|
| 0 | sky | 30.3 % | usually top of frame |
| 1 | water | 51.1 % | dominant class |
| 2 | bridge | 1.6 % | static, urban |
| 3 | obstacle | 0.9 % | inanimate (buoys, boats, kayaks) |
| 4 | living_obs | **0.05 %** | animate (humans, birds) — extremely rare |
| 5 | background | 10.9 % | shoreline, trees, land |
| 6 | self | 3.1 % | the recording vessel itself |
| 255 | (ignore) | — | augmentation border sentinel; trainer uses `ignore_index` |

### Augmentation pipelines (`src/augmentations.py`)

| | Name | Contents |
|---|------|----------|
| A | "MassMIND-replicated" | Rotations ±2/±5/±7°, horizontal flip. Mirrors the paper's Sec. 5.1. |
| B | "Extended"           | A + random crop+resize, CLAHE, mild Gaussian noise. |
| C | "None"               | Normalisation + tensor conversion. Used as the no-aug baseline (Run 1) and as the val/test pipeline. |

Two things deliberately excluded from all three pipelines:

- **Vertical flip** — sky stays on top in maritime imagery; flipping is
  physically wrong.
- **Brightness / contrast jitter** — in LWIR the pixel intensity *is* the
  class signal. The paper explicitly reports this hurt their results, and our
  pipeline B's mild noise injection alone was enough to wipe out the
  `obstacle` class in the baseline runs (see Results).

## Baseline results — U-Net + VGG-16, 20 epochs, P100

These are the *baseline* (existing-model) runs that confirm the pipeline
works end-to-end. From `runs/kaggle_extract/summary.csv`:

| Aug | best ep | val mIoU | sky | water | bridge | obstacle | **living_obs** | background | self |
|-----|---------|----------|-------|-------|--------|----------|----------------|------------|------|
| A   | 18      | 0.739    | 0.986 | 0.990 | 0.718  | 0.641    | **0.000**      | 0.893      | 0.945 |
| B   | 20      | 0.643    | 0.985 | 0.979 | 0.719  | **0.000**| **0.000**      | 0.877      | 0.940 |
| C   | 20      | 0.744    | 0.986 | 0.990 | 0.723  | 0.666    | **0.000**      | 0.896      | 0.946 |

Two findings, both **consistent with the MassMIND paper**:

1. **`living_obs` IoU collapses to exactly 0** with plain cross-entropy and
   `argmax` decoding. The class is 1/2000 as common as water; gradients never
   push a logit high enough to win an `argmax`. The paper hit the same wall —
   their UNet got F1 = 16.1 at τ = 0.6, but F1 = 54.5 at τ = 0.3. Signal is in
   the logits; argmax discards it. Tier 2's Dice + CE loss + auxiliary
   deep-supervision heads are designed to address this directly.
2. **Pipeline B destroys the `obstacle` class** (0.666 → 0.00008). Even
   modest photometric perturbations break LWIR. The MassMIND paper's
   Section 5 explicitly warns against brightness jitter; our Run 4-equivalent
   reproduces the failure mode.

A and C land within 0.005 mIoU of each other for the baseline, suggesting the
geometric augmentations in A add little over plain normalisation when training
from a pretrained encoder.

## Evaluation metrics

Required by the assignment:

- **IoU** per class and macro-averaged (mIoU)
- **Precision** per class
- **Recall** per class
- **Total parameter count** and **trainable parameter count**

Additional, for stronger comparison with the MassMIND paper:

- **F1 per class** (the paper's headline metric)
- **Per-class results at two thresholds: τ = 0.6 (standard) and τ = 0.3 (rare-class)**
- **Training time per epoch** and **inference time per image**

Metrics are computed via the `ConfusionMatrixTracker` in `src/metrics.py`
with per-class breakdown and a macro aggregate.

## Tests

```bash
pytest tests/ -q
```

Covers: dataset loading + augmentation invariants, `ConfusionMatrixTracker`
correctness, model factory + first-conv adaptation.

## References

- Nirgudkar, S., DeFilippo, M., Sacarny, M., Benjamin, M., Robinette, P.
  (2023). *MassMIND: Massachusetts Maritime INfrared Dataset.* International
  Journal of Robotics Research, 42(1–2), 21–32.
  DOI: [10.1177/02783649231153020](https://doi.org/10.1177/02783649231153020).
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Chen, J. et al. (2021). *TransUNet: Transformers Make Strong Encoders for
  Medical Image Segmentation.* arXiv:2102.04306.
- Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the
  Pancreas.* MIDL.
- Zhao, H. et al. (2017). *Pyramid Scene Parsing Network (PSPNet).* CVPR.
- Zhou, Z. et al. (2018). *UNet++: A Nested U-Net Architecture for Medical
  Image Segmentation.* DLMIA.
- Upstream MassMIND repository: <https://github.com/uml-marine-robotics/MassMIND>
