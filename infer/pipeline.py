from dataclasses import dataclass, replace
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from models import OutdoorDepth
from .camera import FocalMapping
from .depth_io import (
    depth_statistics,
    reciprocal_depth,
    save_depth,
    save_depth_png,
    save_json,
)
from .inputs import InferenceSample, read_image
from .gallery import write_gallery
from .visualization import DepthColorizer


@dataclass(frozen=True)
class InferenceOptions:
    steps: int = 20
    ensemble: int = 4
    seed: int = 42
    profile: bool = False
    moge2_cls_token: bool | None = None


@dataclass(frozen=True)
class ExportOptions:
    visualize: bool = False
    save_png: bool = False
    save_inverse: bool = False
    save_canonical: bool = False
    png_scale: float = 100.0
    min_depth: float = 0.1
    max_depth: float = 655.0
    colormap: str = "cube"
    reverse_colors: bool = False
    preview_width: int = 768


@dataclass
class DepthPrediction:
    depth: np.ndarray
    canonical_depth: np.ndarray
    focal_mapping: FocalMapping
    timings: dict[str, float]
    peak_memory_gib: float | None


class InferenceTimer:
    def __init__(self, device: torch.device, enabled: bool):
        self.device = device
        self.enabled = enabled
        self.synchronize()
        self.started = self.previous = time.perf_counter()
        self.stages = {}
        if enabled and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def mark(self, stage: str) -> None:
        if not self.enabled:
            return
        self.synchronize()
        now = time.perf_counter()
        self.stages[stage] = now - self.previous
        self.previous = now

    def finish(self) -> tuple[dict[str, float], float | None]:
        self.synchronize()
        timings = dict(self.stages, total=time.perf_counter() - self.started)
        peak = None
        if self.enabled and self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.device) / 2**30
        return timings, peak


class OutdoorInference:
    def __init__(
        self,
        weights: str | Path | None = None,
        device: str = "cuda",
        options: InferenceOptions | None = None,
    ):
        self.options = options or InferenceOptions()
        if self.options.steps < 1 or self.options.ensemble not in (1, 4):
            raise ValueError("Steps must be positive; ensemble must be 1 or 4")
        started = time.perf_counter()
        self.model = OutdoorDepth(weights, device, moge2_cls_token=self.options.moge2_cls_token)
        self.options = replace(self.options, moge2_cls_token=self.model.config.moge2_cls_token)
        self.load_seconds = time.perf_counter() - started

    @torch.inference_mode()
    def predict(self, image: Image.Image, fx: float) -> DepthPrediction:
        if not math.isfinite(fx) or fx <= 0:
            raise ValueError("Focal length must be positive and finite")
        model = self.model
        timer = InferenceTimer(model.device, self.options.profile)
        prepared = model.image_processor(image, model.device)
        timer.mark("preprocess")
        generator = torch.Generator(device=model.device).manual_seed(self.options.seed)
        canonical = model(
            prepared.canonical_image,
            prepared.qwen_image,
            num_inference_steps=self.options.steps,
            ensemble_size=self.options.ensemble,
            generator=generator,
            stage_callback=timer.mark if self.options.profile else None,
        )
        metric = model.image_processor.restore_depth(canonical, fx, prepared)
        native_depth = metric[0, 0].float().cpu().numpy()
        canonical_depth = canonical[0, 0].float().cpu().numpy()
        timer.mark("restore_depth")
        timings, peak = timer.finish()
        if not np.isfinite(native_depth).all() or not (native_depth > 0).all():
            raise RuntimeError("Inference returned nonfinite or nonpositive depth")
        focal = FocalMapping(
            fx,
            prepared.original_size[1],
            canonical.shape[-1],
            model.config.focal_reference,
        )
        return DepthPrediction(native_depth, canonical_depth, focal, timings, peak)


class PredictionWriter:
    def __init__(self, options: ExportOptions | None = None):
        self.options = options or ExportOptions()
        self.colorizer = DepthColorizer(
            self.options.min_depth,
            self.options.max_depth,
            self.options.colormap,
            self.options.reverse_colors,
        )
        if self.options.preview_width < 64:
            raise ValueError("Preview width must be at least 64")
        if not math.isfinite(self.options.png_scale) or self.options.png_scale <= 0:
            raise ValueError("PNG scale must be positive and finite")

    def write(
        self,
        sample: InferenceSample,
        image: Image.Image,
        prediction: DepthPrediction,
        inference: InferenceOptions,
    ) -> dict:
        output = sample.output
        if prediction.depth.shape != (image.height, image.width):
            raise ValueError("Prediction must have the original image resolution")
        save_depth(output, prediction.depth)
        artifacts = dict(depth=output.name)
        metadata = dict(
            image=sample.name,
            depth=depth_statistics(prediction.depth),
            camera=prediction.focal_mapping.to_dict(),
            model_size_hw=list(prediction.canonical_depth.shape),
            inference=dict(
                steps=inference.steps, ensemble=inference.ensemble, seed=inference.seed,
                **({"moge2_cls_token": inference.moge2_cls_token}
                   if inference.moge2_cls_token is not None else {}),
            ),
            timing_seconds=prediction.timings,
            peak_memory_gib=prediction.peak_memory_gib,
            artifacts=artifacts,
        )
        if self.options.save_canonical:
            path = output.with_suffix(".canonical.npy")
            save_depth(path, prediction.canonical_depth)
            artifacts["canonical_depth"] = path.name
        if self.options.save_inverse:
            path = output.with_suffix(".inverse.npy")
            save_depth(path, reciprocal_depth(prediction.depth))
            artifacts["inverse_depth"] = path.name
            metadata["inverse_depth_units"] = "1/m"
        if self.options.save_png:
            path = output.with_suffix(".depth.png")
            metadata["depth_png"] = save_depth_png(
                path, prediction.depth, self.options.png_scale
            )
            artifacts["depth_png"] = path.name
        if self.options.visualize:
            path = output.with_suffix(".color.png")
            self.colorizer.colorize(prediction.depth).save(path)
            artifacts["color"] = path.name
            path = output.with_suffix(".preview.jpg")
            self.colorizer.preview(
                image, prediction.depth, self.options.preview_width
            ).save(path, quality=95)
            artifacts["preview"] = path.name
            metadata["color_encoding"] = self.colorizer.to_dict()
        save_json(output.with_suffix(".json"), metadata)
        return metadata


def run_inference(
    pipeline: OutdoorInference,
    samples: list[InferenceSample],
    writer: PredictionWriter,
    summary_path: str | Path,
) -> dict:
    started = time.perf_counter()
    images, gallery_records = [], []
    for index, sample in enumerate(samples, 1):
        image = read_image(sample.image)
        prediction = pipeline.predict(image, sample.fx)
        metadata = writer.write(sample, image, prediction, pipeline.options)
        images.append(dict(image=sample.name, seconds=prediction.timings['total']))
        if writer.options.visualize:
            gallery_records.append((sample.output, metadata))
        print(f"[{index}/{len(samples)}] {sample.name}: {prediction.timings['total']:.3f}s -> {sample.output}",
              flush=True)
    elapsed = time.perf_counter() - started
    inference_seconds = sum(image['seconds'] for image in images)
    summary = dict(
        completed=len(images), model_load_seconds=pipeline.load_seconds,
        elapsed_seconds=elapsed, inference_seconds=inference_seconds,
        mean_inference_seconds=inference_seconds / len(images) if images else 0.,
        images_per_second=len(images) / elapsed if elapsed > 0 else 0.,
        images=images,
    )
    summary_path = Path(summary_path)
    if writer.options.visualize:
        gallery = summary_path.with_suffix('.html')
        write_gallery(gallery_records, gallery)
        summary['gallery'] = gallery.name
    save_json(summary_path, summary)
    return summary
