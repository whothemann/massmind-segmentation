"""Export a VGG16UNetExt configuration to ONNX for inspection in Netron.

ONNX captures the full computational graph -- operators, tensor shapes,
parameter counts -- which is exactly what https://netron.app renders. Upload
the resulting ``.onnx`` file there to get an interactive view of the
encoder, bottleneck, decoder, and (if enabled) the attention gate and aux
head branches.

By default the model is exported in **eval mode** -- the same graph that
gets deployed at inference. Aux heads are absent there (forward returns a
single tensor in eval), so the graph stays clean. Pass ``--training-mode``
to instead export the train-mode graph, which exposes all three output
heads -- useful for visualising the deep-supervision structure for a
report figure.

Usage:

    # Plain VGG16-Ext baseline
    python scripts/export_onnx.py --output vgg16_ext_base.onnx

    # Our headline config (transformer bottleneck, no attention, no aux)
    python scripts/export_onnx.py --transformer-bottleneck --output trans.onnx

    # Everything turned on, exported in training mode so aux heads are visible
    python scripts/export_onnx.py \\
        --attention-gates --transformer-bottleneck --aux-heads \\
        --training-mode --output full_deep_supervision.onnx
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import torch

# Allow running as `python scripts/export_onnx.py` from the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import build_unet_vgg16_ext  # noqa: E402

logger = logging.getLogger("export_onnx")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--output",
        type=Path,
        default=Path("model.onnx"),
        help="Destination .onnx path (default: model.onnx).",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=7,
        help="Output channel count of the segmentation head (default 7 for MassMIND).",
    )
    p.add_argument(
        "--in-channels",
        type=int,
        choices=[1, 3],
        default=1,
        help="Input channel count -- 1 for LWIR (default), 3 for RGB.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=256,
        help="Dummy-input height. Must be divisible by 32.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=256,
        help="Dummy-input width. Must be divisible by 32.",
    )
    p.add_argument(
        "--attention-gates",
        action="store_true",
        help="Enable AttentionGate skip refiners.",
    )
    p.add_argument(
        "--transformer-bottleneck",
        action="store_true",
        help="Enable TransformerBottleneck body.",
    )
    p.add_argument(
        "--aux-heads",
        action="store_true",
        help="Enable deep-supervision aux heads. Visible in the exported "
        "graph only when --training-mode is also set.",
    )
    p.add_argument(
        "--training-mode",
        action="store_true",
        help="Export the train-mode graph (with aux outputs, if --aux-heads). "
        "Default is eval mode -- single-tensor output, matches deployment.",
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip ImageNet weights for the encoder. Has no effect on graph "
        "topology, only on the values stored in the .onnx file.",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default 17 -- needed for modern attention ops).",
    )
    p.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Allow variable batch size at inference. Netron will display the "
        "batch dim as a symbolic name instead of a concrete number.",
    )
    p.add_argument(
        "--external-data",
        action="store_true",
        help="Store weights in a separate .onnx.data file alongside the graph. "
        "Default is a single self-contained .onnx file -- easier to upload to "
        "netron.app, at the cost of a ~150 MB file size for the full model.",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    if args.height % 32 or args.width % 32:
        raise SystemExit(
            f"--height ({args.height}) and --width ({args.width}) must each "
            "be divisible by 32 (VGG16 encoder pools 5x)."
        )

    encoder_weights = None if args.no_pretrained else "imagenet"

    logger.info(
        "Building VGG16UNetExt for export: classes=%d, in=%d, "
        "attention=%s, transformer=%s, aux=%s, mode=%s",
        args.num_classes,
        args.in_channels,
        args.attention_gates,
        args.transformer_bottleneck,
        args.aux_heads,
        "train" if args.training_mode else "eval",
    )

    model = build_unet_vgg16_ext(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        encoder_weights=encoder_weights,
        use_transformer_bottleneck=args.transformer_bottleneck,
        use_attention_gates=args.attention_gates,
        use_aux_heads=args.aux_heads,
    )

    # In train mode with aux heads the forward returns 3 tensors; we need to
    # name them all so ONNX bookkeeping works. In eval mode, just one output.
    if args.training_mode:
        model.train()
    else:
        model.eval()

    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]] = {}
    if args.training_mode and args.aux_heads:
        output_names = ["logits_main", "logits_aux_shallow", "logits_aux_deep"]
    else:
        output_names = ["logits"]

    if args.dynamic_batch:
        dynamic_axes["image"] = {0: "batch"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    dummy = torch.randn(1, args.in_channels, args.height, args.width)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting to %s (opset=%d, external_data=%s)",
        args.output, args.opset, args.external_data,
    )
    torch.onnx.export(
        model,
        (dummy,),
        str(args.output),
        input_names=["image"],
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes or None,
        external_data=args.external_data,
    )

    size_mb = args.output.stat().st_size / 1e6
    logger.info("Wrote %s  (%.1f MB)", args.output, size_mb)
    logger.info("Open at https://netron.app -- drag the file into the browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
