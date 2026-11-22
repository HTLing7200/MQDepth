from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .config import MQDepthConfig


@dataclass(frozen=True)
class PreparedImage:
    canonical_image: torch.Tensor
    qwen_image: torch.Tensor
    original_size: tuple[int, int]
    horizontal_scale: float


class ImageProcessor:
    def __init__(self, config: MQDepthConfig):
        self.config = config

    @staticmethod
    def image_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
        return tensor / 127.5 - 1

    def canonical_size(self, width: int, height: int) -> tuple[int, int]:
        target_width = self.config.width
        multiple = self.config.size_multiple
        target_height = max(
            multiple, round(height * target_width / width / multiple) * multiple
        )
        return target_width, target_height

    def qwen_size(self, width: int, height: int) -> tuple[int, int]:
        scale = self.config.qwen_long_side / max(width, height)
        return max(1, round(width * scale)), max(1, round(height * scale))

    def __call__(self, image: Image.Image, device: torch.device) -> PreparedImage:
        image = image.convert("RGB")
        width, height = image.size
        canonical_size = self.canonical_size(width, height)
        canonical = image.resize(canonical_size, Image.Resampling.BILINEAR)
        qwen = image.resize(self.qwen_size(width, height), Image.Resampling.BILINEAR)
        canonical_tensor = self.image_to_tensor(canonical).unsqueeze(0).to(device)
        qwen_tensor = self.image_to_tensor(qwen).unsqueeze(0)
        return PreparedImage(
            canonical_image=canonical_tensor,
            qwen_image=qwen_tensor,
            original_size=(height, width),
            horizontal_scale=canonical_size[0] / width,
        )

    def restore_depth(
        self,
        canonical_depth: torch.Tensor,
        focal_length: float,
        prepared: PreparedImage,
    ) -> torch.Tensor:
        focal_scale = (
            focal_length * prepared.horizontal_scale / self.config.focal_reference
        )
        metric_depth = canonical_depth * focal_scale
        return F.interpolate(
            metric_depth,
            size=prepared.original_size,
            mode="bilinear",
            align_corners=False,
        )
