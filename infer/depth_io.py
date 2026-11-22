import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin


def load_depth(path: str | Path) -> np.ndarray:
    depth = np.load(path, allow_pickle=False)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise ValueError("Expected a numeric H x W depth array")
    return np.asarray(depth, dtype=np.float32)


def save_depth(path: str | Path, depth: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(depth, dtype=np.float32), allow_pickle=False)


def save_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_depth_png(path: str | Path, depth: np.ndarray, scale: float = 100.0) -> dict:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("PNG scale must be positive and finite")
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    encoded[valid] = (
        np.rint(depth[valid].astype(np.float64) * scale)
        .clip(1, 65535)
        .astype(np.uint16)
    )
    info = PngImagePlugin.PngInfo()
    info.add_text("depth_unit", "m")
    info.add_text("depth_scale", str(scale))
    info.add_text("invalid_value", "0")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded).save(path, pnginfo=info)
    return dict(
        scale=scale,
        meters_per_unit=1.0 / scale,
        maximum_representable_depth_m=65535.0 / scale,
        invalid_value=0,
        saturated_pixels=int((valid & (depth > 65535.0 / scale)).sum()),
    )


def load_depth_png(path: str | Path, scale: float | None = None) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("I", "I;16", "I;16B", "I;16L"):
            raise ValueError("Expected a 16-bit grayscale depth PNG")
        if scale is None:
            scale = float(image.info.get("depth_scale", 100.0))
        encoded = np.asarray(image, dtype=np.float32)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("PNG scale must be positive and finite")
    return encoded / scale


def reciprocal_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    result = np.zeros_like(depth)
    np.divide(1.0, depth, out=result, where=np.isfinite(depth) & (depth > 0))
    return result


def depth_statistics(depth: np.ndarray) -> dict:
    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0)
    selected = values[valid]
    summary = dict(
        shape_hw=list(values.shape),
        dtype=str(values.dtype),
        units="m",
        valid_pixels=int(valid.sum()),
    )
    if selected.size:
        quantiles = np.quantile(selected, [0.01, 0.5, 0.99])
        summary.update(
            min_m=float(selected.min()),
            max_m=float(selected.max()),
            mean_m=float(selected.mean()),
            p01_m=float(quantiles[0]),
            median_m=float(quantiles[1]),
            p99_m=float(quantiles[2]),
        )
    return summary
