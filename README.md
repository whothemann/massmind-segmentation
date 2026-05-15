# MassMIND Segmentation


Semantic segmentation of long-wave infrared (LWIR) maritime imagery from the
[MassMIND dataset](https://github.com/uml-marine-robotics/MassMIND) (Nirgudkar
et al., 2023). Coursework for *Computer Vision — Assignment 2, FEUP 2025/26*.

The assignment asks for two things: **(1)** a custom segmentation architecture
proposed and defended by us, and **(2)** a comparison against at least one
existing model trained on the same data. We're delivering this as:

- **Custom model — `VGG16UNetExt`:** ImageNet-pretrained VGG-16 encoder (same
  pattern as the in-class demonstrator) wrapped in a hand-implemented decoder
  with three individually-toggleable architectural modifications: an
  Attention-Gate skip-refinement, a Transformer-Encoder bottleneck, and
  deep-supervision auxiliary heads. Every decoder block, the attention gate,
  the transformer body, and the aux-head supervision are built from
  `torch.nn` primitives -- only the encoder backbone is borrowed.
- **Existing-model baseline:** U-Net with the same ImageNet-pretrained VGG-16
  encoder but SMP's default decoder (Nirgudkar et al.'s strongest CNN
  baseline), trained on the same data with the same loss and metrics.

## Status

| Phase | State |
|---|---|
| Dataset download + 70/20/10 session-stratified split | ✅ |
| Pixel mean/std + per-class pixel counts | ✅ |
| U-Net + VGG-16 trainer (PyTorch; CUDA / MPS / CPU autodetect) | ✅ |
| Augmentation pipelines A / B / C | ✅ |
| Colab + Kaggle training notebooks | ✅ |
| Baseline U-Net + VGG-16 runs for A / B / C on Kaggle, 20 ep | ✅ — `runs/kaggle_extract/` |
| `VGG16UNetExt` custom model (`src/models/unet_vgg16_ext.py`) | ✅ |
| Hand-implemented `AttentionGate`, `TransformerBottleneck`, aux heads | ✅ |
| Focal loss (γ=2) + AdamW + cosine schedule | ✅ |
| AMP (mixed precision) opt-in flag for CUDA | ✅ |
| 2×2 architecture probe (`base` / `att` / `trans` / `att_trans`) on T4 | ✅ — see "Architecture probe results" below |
| Deep-supervision probe (`trans_aux`) | ✅ |
| ONNX export script for Netron visualisation | ✅ — `scripts/export_onnx.py` |
| Final 50-epoch training run on full data (winner architecture) | ⏳ |
| Threshold sweep (τ = 0.3 and 0.6) for rare-class F1 | ⏳ |
| Final writeup | ⏳ |

## Custom architecture (`VGG16UNetExt`)

The same encoder/decoder *pattern* as the in-class demonstrator (cell 30 of
`12_Pytorch_SemanticSegmentation.ipynb`: pretrained VGG-16 sliced into
encoder stages + hand-rolled `conv`/`up_conv` decoder helpers + skip
concatenation), with three independently-toggleable modifications. Each
modification is a flag on `build_unet_vgg16_ext()` so we can ablate them
cleanly:

| Flag | Component | Status |
|---|---|---|
| `use_attention_gates=True` | Oktay-style attention gates on each skip | hand-implemented in `src/models/_attention_gate.py` |
| `use_transformer_bottleneck=True` | Multi-head self-attention bottleneck body | hand-implemented in `src/models/_transformer_bottleneck.py` |
| `use_aux_heads=True` | Deep-supervision aux heads at decoder mids | hand-implemented in `src/models/unet_vgg16_ext.py` |

All four flag combinations (none / one / two / three on) plus the
`trans + aux` combination are wired into the probe script
(`scripts/probe_architectures.py`) and were measured on Kaggle T4.

```
Input (1×H×W, LWIR)
   │
   ▼  channel-mean-adapted first conv (3-ch ImageNet → 1-ch LWIR)
   │
   ▼  VGG-16 encoder (pretrained), 6 features at strides [1, 2, 4, 8, 16, 32]
Enc0  Enc1  Enc2  Enc3  Enc4  Enc5
                              │
                              ▼  [Mod A] AttentionGate on each skip (optional)
                              │  [Mod B] Transformer bottleneck (optional)
                         bottleneck
                              │
                              ▼  4× (ConvTranspose → concat skip → DoubleConv)
                         Dec1 → Dec2 → Dec3 → Dec4
                                 │      │      │
                                 ▼      ▼      ▼
                             [Mod C aux head]  [Mod C aux head]  Final upsample
                             (training only)   (training only)         │
                                                                       ▼
                                                                  1×1 Conv → 7 classes
```

### Mod A — Attention Gate skip refinement (`use_attention_gates=True`)

Hand-implemented `AttentionGate` (Oktay et al. 2018) in
`src/models/_attention_gate.py`. Replaces the `Up.skip_refine = nn.Identity()`
default on each decoder block with an additive attention module that takes
the encoder skip and the upsampled decoder gating signal, computes a spatial
attention map `α ∈ (0, 1)` via `(W_skip(skip) + W_gating(gating)) → ReLU →
1×1 conv → BN → sigmoid`, and returns `α · skip`. Channels collapsed to
`skip_channels // 2` internally, BatchNorm throughout.

*Why:* the vanilla U-Net's skip forwards encoder features unchanged — the
decoder gets no learned control over what it receives. The gate suppresses
clutter/background regions of the skip based on the decoder's coarse-grained
context. For LWIR this matters: the same thermal edge can be a boat hull
or a wave artefact, distinguishable only from broader scene structure.

Approx. parameter cost: ~610 K across all four levels (1×1 convs at
512/512/256/128 channels).

### Mod B — Transformer bottleneck (`use_transformer_bottleneck=True`)

Hand-implemented `TransformerBottleneck` in `src/models/_transformer_bottleneck.py`.
Replaces the default `DoubleConv` body of the bottleneck wrapper. The 512-channel
encoder Stage-5 feature map (8×8 at 256-px input, 16×20 at 640×512 native LWIR)
is flattened to tokens, summed with a learnable 2-D positional embedding
(bilinearly resized to runtime spatial), and passed through **2 stacked
`nn.TransformerEncoderLayer` blocks** (8 attention heads, embed dim 512,
FFN hidden 1024, GELU, pre-norm), then reshaped back to spatial form.

The composition is hand-rolled; we use `nn.TransformerEncoderLayer` as a
primitive for the math just as we use `nn.Conv2d` for convolutions.

*Why here:* attention is global by construction but quadratic in tokens; at
the bottleneck the spatial grid is tiny (64 tokens at 256 px), so it's
cheap. The MassMIND task has class-level scene structure — water spans the
whole frame, bridges are extended objects, rare classes need global context
to be distinguished from clutter — exactly the kind of dependency stacked
3×3 convs struggle to capture. Reference pattern: TransUNet (Chen et al.
2021); attention mechanism: Vaswani et al. 2017.

Approx. parameter cost: 4.7 M (replaces the 5.2 M `DoubleConv`-bottleneck,
so the *net* cost is slightly negative).

### Mod C — Deep supervision auxiliary heads (`use_aux_heads=True`)

Two extra 1×1-conv heads attached to the Up2 (deeper, weight 0.2) and Up3
(shallower, weight 0.4) decoder outputs in
`src/models/unet_vgg16_ext.py`, each bilinear-upsampled to the input
resolution. Total training loss:

```
L_total = L_main  +  0.4 · L_aux_shallow  +  0.2 · L_aux_deep
```

The model returns a tuple `(main, aux_shallow, aux_deep)` in `model.train()`
mode and just `main` (a single tensor) in `model.eval()` — so the heads
add **zero deployed parameters** and zero inference cost. Loss combination
lives in `src/train._compute_loss()`.

*Why:* gradient for the main output flows back through the entire decoder
before reaching the bottleneck; for `living_obs` at 0.05 % of pixels, that
gradient is dominated by majority-class signal at every intermediate layer.
Auxiliary heads inject direct full-class-distribution supervision into the
deeper decoder layers. Reference patterns: PSPNet (Zhao et al. 2017),
UNet++ (Zhou et al. 2018).

Approx. parameter cost: ~5 K (negligible; two 1×1 convs of 512×7 and 256×7).

### Demonstrator vs `VGG16UNetExt` — at a glance

| Component | Demonstrator (`12_Pytorch_SemanticSegmentation.ipynb`) | `VGG16UNetExt` (ours, `trans_aux`) |
|---|---|---|
| Encoder | `vgg16_bn(pretrained=True).features` (torchvision) | SMP `vgg16` (`pretrained="imagenet"`) + 3-ch → 1-ch channel-mean adapter |
| Bottleneck | `conv(512, 1024)` (single DoubleConv) | Hand-implemented `TransformerBottleneck` (2 layers, 8 heads) |
| Skip connections | Raw concatenation (`torch.cat`) | Hand-implemented `AttentionGate` (Oktay-style) — opt-in via flag |
| Output supervision | Single 1×1 conv head | Main + 2 aux heads with weighted loss (train-only) |
| Approx. parameter count | ~24 M | ~38 M |
| Approx. model code | ~80 lines | ~530 lines (model files only) |
| Pretrained weights | ImageNet (`pretrained=True` default) | ImageNet (channel-mean-adapted for 1-channel LWIR) |
| Loss | Cross-entropy | Focal loss (γ=2) |

## Training methodology

- **Loss** — `FocalLoss(gamma=2.0)` from Lin et al. (2017), implemented in
  `src/losses.py`. Down-weights easy-to-classify pixels (the majority sky /
  water) and focuses gradient on hard rare-class pixels. Dropped the planned
  Dice+CE because focal loss matched or beat it in our baseline probe at
  the same data scale, with a simpler single-term formulation. Both `--loss
  focal` (default) and `--loss ce` are supported in `src/train.py`.
- **Optimizer** — AdamW, `weight_decay = 1e-4`.
- **Schedule** — `CosineAnnealingLR(T_max=epochs)` from `lr = 1e-4`.
- **Epochs** — 10 for the architecture probe (this README's results); 50 for
  the planned final training run on full data, matching the MassMIND paper.
- **Batch size** — 8 on Kaggle T4 with AMP.
- **Mixed precision** — opt-in via `--amp` CLI flag (default off in
  `src/train.py` to preserve baseline numerics; default *on* in the probe
  driver `scripts/probe_architectures.py`). Implementation: `torch.amp.autocast(fp16)`
  + `GradScaler`, gated on `device.type == "cuda"` so the same code path is a
  no-op on CPU/MPS. Delivered ~2.5× speedup on Kaggle T4.
- **Pretrained encoder** — yes, all `VGG16UNetExt` variants load ImageNet weights
  through SMP and adapt the first conv from 3-ch RGB to 1-ch LWIR via channel-mean
  initialisation in `src/models/_adapt.py`. Matches the pattern used by the
  in-class demonstrator.
- **Augmentations** — three pipelines defined in `src/augmentations.py`; see
  table below.

## Architecture probe results

Five `VGG16UNetExt` configurations trained on Kaggle T4 with focal loss,
600 training images, 10 epochs, AMP on the AMP runs. Same seed, same
hyperparameters across all configs — the only thing that varies is the
architecture flag combination. Probe driver: `scripts/probe_architectures.py`.

| Config | Att. gates | Transformer | Aux heads | Precision | best mIoU | min/cfg |
|---|:---:|:---:|:---:|---|---:|---:|
| `base` | ❌ | ❌ | ❌ | FP32 | 0.579 | 41 |
| `att` | ✅ | ❌ | ❌ | AMP | 0.620 | 16 |
| `trans` | ❌ | ✅ | ❌ | FP32 | 0.640 | 41 |
| `att_trans` | ✅ | ✅ | ❌ | AMP | 0.624 | 14 |
| **`trans_aux`** | ❌ | ✅ | ✅ | AMP | **0.645** | 15 |

Per-class IoU at the best epoch:

| Class | Pixel-% | `base` | `att` | `trans` | `att_trans` | `trans_aux` |
|---|---:|---:|---:|---:|---:|---:|
| sky | 30.3 | 0.983 | 0.985 | 0.984 | 0.984 | 0.985 |
| water | 51.1 | 0.982 | 0.983 | 0.983 | 0.981 | 0.983 |
| **bridge** | 1.6 | 0.330 | 0.410 | 0.424 | 0.386 | **0.450** |
| **obstacle** | 0.9 | 0.059 | 0.222 | 0.351 | 0.297 | 0.351 |
| **living_obs** | 0.05 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| background | 10.9 | 0.813 | 0.826 | 0.826 | 0.812 | 0.830 |
| self | 3.1 | 0.889 | 0.913 | 0.915 | 0.905 | 0.919 |

**Three findings drive the architecture choice:**

1. **The Transformer bottleneck is the dominant contributor:** +6.1 pp mIoU
   over `base`, with the gain concentrated on `obstacle` (+29 pp) and
   `bridge` (+9 pp) — the two mid-rarity classes whose disambiguation
   requires global scene context. Sky/water are already saturated for all
   configs. Matches the TransUNet hypothesis exactly.
2. **Attention Gates help individually but not additively with Transformer:**
   `att` alone is +4.0 pp, but `att_trans` (0.624) is *worse* than `trans`
   alone (0.640). The two mechanisms partially compete for the same
   feature-reweighting role; once the transformer provides strong global
   context, the gates suppress skip detail that the decoder needs back.
3. **Deep supervision yields a small but real gain at the same compute
   budget:** `trans_aux` 0.645 vs `trans` 0.640 (+0.5 pp). The mechanism is
   *convergence acceleration*, not a ceiling lift — at epoch 5 `trans_aux`
   already has `obstacle = 0.125` vs `trans = 0.023` (5× higher). Both
   plateau at similar levels by epoch 10. Bridge gains most (+2.6 pp). Zero
   inference-time cost.

**`living_obs` remains at 0.000 across all five configurations.** This is
not an architecture problem — it's an argmax-decoding limitation at 0.05 %
pixel share. Threshold sweep (τ=0.3 vs 0.6) is the planned next step;
MassMIND paper reports UNet F1 16 → 54 from this single change.

### Planned remaining experiment matrix

| # | Model | Aug | Epochs | Data | Purpose |
|---|---|---|---|---|---|
| 1 | `trans_aux` (winner) | A | 50 | full (2042 train) | Headline final number |
| 2 | `trans_aux` | C (no-aug) | 50 | full | Assignment-required no-aug variant |
| 3 | `trans_aux` | B (extended) | 50 | full | Own extension of MassMIND paper |
| 4 | U-Net + VGG-16 (baseline) | A | 50 | full | Required comparison vs existing |
| 5 | Threshold sweep (τ=0.3 vs 0.6) | — | — | best checkpoint | Rare-class F1 |

## Project layout

```
massmind_segmentation/
├── data/
│   ├── massmind/                          # raw LWIR images + masks (gitignored)
│   └── splits/
│       ├── split.json                     # 70/20/10 session-stratified
│       ├── stats.json                     # train-set pixel mean & std
│       └── class_pixel_counts.json        # global per-class pixel count
├── src/
│   ├── dataset.py                         # MassMINDDataset, bit-depth aware
│   ├── splits.py                          # session-stratified split builder
│   ├── stats.py                           # pixel mean/std + class counts
│   ├── augmentations.py                   # albumentations pipelines A / B / C
│   ├── metrics.py                         # ConfusionMatrixTracker → IoU, pixel acc
│   ├── losses.py                          # FocalLoss (Lin et al. 2017)
│   ├── train.py                           # single-file trainer; AMP, model+loss dispatch
│   └── models/
│       ├── __init__.py                    # builder exports
│       ├── _adapt.py                      # adapt first conv 3-ch → 1-ch via channel mean
│       ├── unet.py                        # build_unet_vgg16 (SMP) — existing-model baseline
│       ├── custom_unet.py                 # from-scratch hand-rolled U-Net (unused, kept for reference)
│       ├── _attention_gate.py             # AttentionGate (Oktay 2018), hand-implemented
│       ├── _transformer_bottleneck.py     # TransformerBottleneck, hand-implemented
│       └── unet_vgg16_ext.py              # build_unet_vgg16_ext + VGG16UNetExt with three seams
├── scripts/
│   ├── download.py                        # idempotent Google-Drive download via gdown
│   ├── probe_architectures.py             # runs the 2×2 architecture probe + aux variants
│   └── export_onnx.py                     # exports any variant to ONNX for Netron
├── notebooks/
│   ├── 01_data_exploration.ipynb          # class balance, image stats, sample renders
│   ├── 02_train_colab.ipynb               # Colab/Kaggle driver: live tqdm + plots
│   └── 03_probe_kaggle.ipynb              # dedicated probe notebook (T4 + AMP)
├── kaggle_test_run_u_net.ipynb            # baseline run with embedded outputs
├── kaggle_test_run_u_net_with_focal_loss.ipynb  # baseline + focal loss (0.80 mIoU)
├── 12_Pytorch_SemanticSegmentation.ipynb  # the in-class demonstrator (reference)
├── exports/                               # ONNX exports for Netron (gitignored, generated)
├── runs/
│   └── kaggle_extract/                    # baseline run artefacts: CSVs, logs, plots
├── tests/                                 # pytest suite (80 tests)
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_models.py                     # CustomUNet + VGG16 builder + adaptation
│   └── models/
│       └── test_unet_vgg16_ext.py         # VGG16UNetExt + AttentionGate + TransformerBottleneck + aux
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

To smoke-test the custom architecture variants instead of the SMP baseline:

```bash
# Plain VGG16UNetExt
python -m src.train --model vgg16_ext --epochs 1 --subset 30

# With attention gates
python -m src.train --model vgg16_ext --attention-gates --epochs 1 --subset 30

# With transformer bottleneck
python -m src.train --model vgg16_ext --transformer-bottleneck --epochs 1 --subset 30

# Full stack (training-time aux heads), focal loss, AMP (CUDA-only)
python -m src.train --model vgg16_ext \
    --attention-gates --transformer-bottleneck --aux-heads \
    --loss focal --amp \
    --epochs 1 --subset 30
```

### Architecture probe (recommended for a clean comparison)

`scripts/probe_architectures.py` runs all six configs (`base`, `att`,
`trans`, `att_trans`, `trans_aux`, `att_trans_aux`) sequentially with
matched hyperparameters and writes a `summary.json` plus per-config
`metrics.csv` and checkpoints:

```bash
python scripts/probe_architectures.py                          # all four core configs
python scripts/probe_architectures.py --configs trans_aux      # just one
python scripts/probe_architectures.py --no-amp                 # FP32 forced
```

### ONNX export (for Netron visualisation)

```bash
pip install onnx onnxscript  # optional, not in requirements.txt

python scripts/export_onnx.py --output exports/base.onnx
python scripts/export_onnx.py --transformer-bottleneck --output exports/trans.onnx
python scripts/export_onnx.py \
    --attention-gates --transformer-bottleneck --aux-heads --training-mode \
    --output exports/full_training.onnx
```

Drop the `.onnx` file into <https://netron.app> to inspect the graph.

### Colab / Kaggle — full runs

Two notebooks cover the two use cases:

- **`notebooks/02_train_colab.ipynb`** — original baseline driver. Streams
  `src.train` stdout into a tqdm bar + live loss/mIoU plot, optional Drive
  sync. One augmentation per cell.
- **`notebooks/03_probe_kaggle.ipynb`** — dedicated architecture-probe
  notebook for Kaggle. Self-contained: clones repo, installs deps, downloads
  data, runs `scripts/probe_architectures.py`, plots training curves and a
  per-class IoU table for every config in one go. Designed for headless
  "Save Version → Save & Run All".

**Kaggle tips** (lessons learned the hard way):

- **Pick T4 ×2**, *not* P100. PyTorch ≥ 2.5 dropped sm_60 from the official
  CUDA binaries, and Kaggle's pre-installed PyTorch now refuses to run on
  P100 (`CUDA error: no kernel image is available for execution on the
  device`). The trainer's single-GPU code only uses one of the two T4s; the
  second slot is unused but harmless.
- **Enable AMP** for T4 — `scripts/probe_architectures.py` does this by
  default. Delivers ~2.5× speedup and ~50 % less activation memory, which
  is required to fit the attention-gate configs at batch=8 on T4's 16 GB.
- **Enable Internet** in the right-hand notebook settings (one-time phone
  verification on the Kaggle account).
- **Save your GitHub PAT as a Kaggle Secret** named `github_pat` (Add-ons →
  Secrets) so the notebook clones in headless "Save & Run All" mode. Public
  repo? Leave it empty.
- **`NUM_WORKERS = 4`** matches Kaggle's ~4 vCPUs.
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
                                │ VGG16UNetExt (custom)  OR    │
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
   the logits; argmax discards it. Our `trans_aux` config (focal loss +
   transformer bottleneck + deep-supervision aux heads) is designed to push
   the logits in the right direction; the threshold sweep at τ=0.3 is the
   complementary decoding-side fix.
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

80 tests, ~30 s on a laptop. Covers:

- Dataset loading + augmentation invariants (`test_dataset.py`)
- `ConfusionMatrixTracker` correctness (`test_metrics.py`)
- SMP-VGG16 builder + channel-mean adaptation (`test_models.py`)
- Hand-rolled `CustomUNet` blocks + forward + backward (`test_models.py`)
- `VGG16UNetExt` full 2×2 ablation, seam wiring (Identity vs AttentionGate,
  DoubleConv vs TransformerBottleneck), pretrained channel-mean adaptation,
  deep-supervision tuple-vs-tensor output dispatch, determinism
  (`tests/models/test_unet_vgg16_ext.py`)

## References

- Nirgudkar, S., DeFilippo, M., Sacarny, M., Benjamin, M., Robinette, P.
  (2023). *MassMIND: Massachusetts Maritime INfrared Dataset.* International
  Journal of Robotics Research, 42(1–2), 21–32.
  DOI: [10.1177/02783649231153020](https://doi.org/10.1177/02783649231153020).
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional
  Networks for Biomedical Image Segmentation.* MICCAI.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017).
  *Focal Loss for Dense Object Detection.* ICCV.
- Chen, J. et al. (2021). *TransUNet: Transformers Make Strong Encoders for
  Medical Image Segmentation.* arXiv:2102.04306.
- Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the
  Pancreas.* MIDL.
- Zhao, H. et al. (2017). *Pyramid Scene Parsing Network (PSPNet).* CVPR.
- Zhou, Z. et al. (2018). *UNet++: A Nested U-Net Architecture for Medical
  Image Segmentation.* DLMIA.
- Upstream MassMIND repository: <https://github.com/uml-marine-robotics/MassMIND>
