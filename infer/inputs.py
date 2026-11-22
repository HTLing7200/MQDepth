from dataclasses import dataclass
import json
import math
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class InferenceSample:
    image: Path
    name: str
    fx: float
    output: Path


def read_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def read_focal_lengths(path: str | Path | None) -> dict:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Intrinsics JSON must map image names to focal lengths")
    return value


def focal_length(name: str, common_fx: float | None, table: dict) -> float:
    value = common_fx if common_fx is not None else table.get(name)
    if value is None:
        raise ValueError(f"Missing focal length for {name}")
    if isinstance(value, dict):
        value = value["fx"]
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid focal length for {name}")
    return value


def build_samples(
    image: str | None,
    input_dir: str | None,
    output: str,
    fx: float | None,
    intrinsics: str | None,
    recursive: bool = False,
) -> list[InferenceSample]:
    destination = Path(output).expanduser().resolve()
    table = read_focal_lengths(intrinsics)
    if image is not None:
        source = Path(image).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.suffix.lower() != ".npy":
            raise ValueError("Single-image output must end with .npy")
        return [
            InferenceSample(
                source, source.name, focal_length(source.name, fx, table), destination
            )
        ]
    source = Path(input_dir).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if destination == source:
        raise ValueError("Input and output directories must differ")
    if destination.suffix.lower() == ".npy":
        raise ValueError("Directory inference requires an output directory")
    paths = source.rglob("*") if recursive else source.iterdir()
    exclude_destination = destination.is_relative_to(source)
    images = sorted(
        path
        for path in paths
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not (exclude_destination and path.is_relative_to(destination))
    )
    if not images:
        raise ValueError(f"No input images found in {source}")
    samples = []
    for path in images:
        relative = path.relative_to(source)
        name = relative.as_posix()
        target = destination / relative.parent / f"{relative.name}.npy"
        samples.append(
            InferenceSample(path, name, focal_length(name, fx, table), target)
        )
    return samples
