import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from infer.camera import FocalMapping
from infer.depth_io import (
    load_depth,
    load_depth_png,
    reciprocal_depth,
    save_depth,
    save_depth_png,
)
from infer.visualization import DepthColorizer


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Depth formats, color coding and focal mapping"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "colorize",
            "decolorize",
            "to-png",
            "from-png",
            "inverse",
            "inverse-to-depth",
            "canonicalize",
            "restore",
        ],
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metadata", help="Optional inference JSON with color and camera settings"
    )
    parser.add_argument("--min-depth", type=float)
    parser.add_argument("--max-depth", type=float)
    parser.add_argument("--colormap", choices=["cube", "gray"])
    parser.add_argument("--reverse-colors", action="store_true", default=None)
    parser.add_argument("--png-scale", type=float)
    parser.add_argument("--fx", type=float)
    parser.add_argument("--original-width", type=int)
    parser.add_argument("--model-width", type=int)
    parser.add_argument("--focal-reference", type=float)
    return parser


def colorizer_from_args(args, metadata: dict) -> DepthColorizer:
    settings = metadata.get("color_encoding", {})
    return DepthColorizer(
        (
            args.min_depth
            if args.min_depth is not None
            else settings.get("min_depth_m", 0.1)
        ),
        (
            args.max_depth
            if args.max_depth is not None
            else settings.get("max_depth_m", 655.0)
        ),
        (
            args.colormap
            if args.colormap is not None
            else settings.get("colormap", "cube")
        ),
        (
            args.reverse_colors
            if args.reverse_colors is not None
            else settings.get("reverse", False)
        ),
    )


def focal_mapping_from_args(args, metadata: dict) -> FocalMapping:
    settings = metadata.get("camera", {})
    fx = args.fx if args.fx is not None else settings.get("fx_original_px")
    width = (
        args.original_width
        if args.original_width is not None
        else settings.get("original_width")
    )
    if fx is None or width is None:
        raise ValueError(
            "Focal conversion requires --fx and --original-width, or camera metadata"
        )
    return FocalMapping(
        fx,
        width,
        (
            args.model_width
            if args.model_width is not None
            else settings.get("model_width", 768)
        ),
        (
            args.focal_reference
            if args.focal_reference is not None
            else settings.get("focal_reference_px", 1000.0)
        ),
    )


def main():
    parser = make_parser()
    args = parser.parse_args()
    output = Path(args.output)
    png_mode = args.mode in ("colorize", "to-png")
    if output.suffix.lower() != (".png" if png_mode else ".npy"):
        parser.error("Output must end with .png for image export, otherwise .npy")
    try:
        metadata = json.loads(Path(args.metadata).read_text()) if args.metadata else {}
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.mode == "colorize":
            colorizer_from_args(args, metadata).colorize(load_depth(args.input)).save(
                output
            )
        elif args.mode == "decolorize":
            with Image.open(args.input) as image:
                depth = colorizer_from_args(args, metadata).decode(image)
            save_depth(output, depth)
        elif args.mode == "to-png":
            scale = 100.0 if args.png_scale is None else args.png_scale
            save_depth_png(output, load_depth(args.input), scale)
        elif args.mode == "from-png":
            save_depth(output, load_depth_png(args.input, args.png_scale))
        elif args.mode in ("inverse", "inverse-to-depth"):
            save_depth(output, reciprocal_depth(load_depth(args.input)))
        else:
            mapping = focal_mapping_from_args(args, metadata)
            depth = load_depth(args.input)
            mapped = (
                mapping.to_canonical(depth)
                if args.mode == "canonicalize"
                else mapping.to_metric(depth)
            )
            save_depth(output, mapped)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(str(output))


if __name__ == "__main__":
    main()
