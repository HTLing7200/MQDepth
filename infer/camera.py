from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FocalMapping:
    fx: float
    original_width: int
    model_width: int = 768
    focal_reference: float = 1000.0

    def __post_init__(self):
        if not math.isfinite(self.fx) or self.fx <= 0:
            raise ValueError("Focal length must be positive and finite")
        if self.original_width <= 0 or self.model_width <= 0:
            raise ValueError("Image widths must be positive")
        if not math.isfinite(self.focal_reference) or self.focal_reference <= 0:
            raise ValueError("Reference focal length must be positive and finite")

    @property
    def image_scale(self) -> float:
        return self.model_width / self.original_width

    @property
    def resized_fx(self) -> float:
        return self.fx * self.image_scale

    @property
    def metric_scale(self) -> float:
        return self.resized_fx / self.focal_reference

    def to_metric(self, canonical_depth: np.ndarray) -> np.ndarray:
        return np.asarray(canonical_depth, dtype=np.float32) * self.metric_scale

    def to_canonical(self, metric_depth: np.ndarray) -> np.ndarray:
        return np.asarray(metric_depth, dtype=np.float32) / self.metric_scale

    def to_dict(self) -> dict:
        return dict(
            fx_original_px=self.fx,
            original_width=self.original_width,
            model_width=self.model_width,
            horizontal_resize_scale=self.image_scale,
            fx_resized_px=self.resized_fx,
            focal_reference_px=self.focal_reference,
            canonical_to_metric_scale=self.metric_scale,
        )
