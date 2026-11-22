import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer.inputs import build_samples


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Outdoor metric depth inference")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", help="Single input image")
    inputs.add_argument("--input-dir", help="Directory of input images")
    camera = parser.add_mutually_exclusive_group(required=True)
    camera.add_argument(
        "--fx", type=float, help="Original image focal length in pixels"
    )
    camera.add_argument(
        "--intrinsics", help="JSON mapping relative image names to focal lengths"
    )
    parser.add_argument("--weights", default=None, help="Model weights directory")
    parser.add_argument(
        "--output", required=True, help="Depth .npy file or output directory"
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--moge2-cls-token", action=argparse.BooleanOptionalAction, default=None,
        help="Use RGB for the MoGe2 global token; disable to use a white image",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ensemble", type=int, choices=[1, 4], default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record per-stage timing and CUDA peak memory",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save colored depth and RGB/depth preview",
    )
    parser.add_argument("--colormap", choices=["cube", "gray"], default="cube")
    parser.add_argument("--reverse-colors", action="store_true")
    parser.add_argument("--visual-min-depth", type=float, default=0.1)
    parser.add_argument("--visual-max-depth", type=float, default=655.0)
    parser.add_argument("--preview-width", type=int, default=768)
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Save additional uint16 metric depth PNG",
    )
    parser.add_argument(
        "--png-scale", type=float, default=100.0, help="Integer PNG units per meter"
    )
    parser.add_argument(
        "--save-inverse", action="store_true", help="Save inverse depth in 1/m"
    )
    parser.add_argument(
        "--save-canonical",
        action="store_true",
        help="Save depth at reference focal length",
    )
    return parser


def main():
    from infer.pipeline import (
        ExportOptions,
        InferenceOptions,
        OutdoorInference,
        PredictionWriter,
        run_inference,
    )

    parser = make_parser()
    args = parser.parse_args()
    if args.weights is None:
        parser.error("--weights must specify a model weights directory")
    try:
        samples = build_samples(
            args.image,
            args.input_dir,
            args.output,
            args.fx,
            args.intrinsics,
            args.recursive,
        )
        writer = PredictionWriter(
            ExportOptions(
                visualize=args.visualize,
                save_png=args.save_png,
                save_inverse=args.save_inverse,
                save_canonical=args.save_canonical,
                png_scale=args.png_scale,
                min_depth=args.visual_min_depth,
                max_depth=args.visual_max_depth,
                colormap=args.colormap,
                reverse_colors=args.reverse_colors,
                preview_width=args.preview_width,
            )
        )
        options = InferenceOptions(args.steps, args.ensemble, args.seed, args.profile,
                                   args.moge2_cls_token)
        if options.steps < 1:
            raise ValueError("--steps must be positive")
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    summary = (
        samples[0].output.with_suffix(".summary.json")
        if args.image
        else Path(args.output).expanduser().resolve() / "summary.json"
    )
    pipeline = OutdoorInference(args.weights, args.device, options)
    run_inference(pipeline, samples, writer, summary)
    print(f"Summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
