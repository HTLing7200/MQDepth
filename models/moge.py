import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import MQDepthConfig
from .geometry import GeometryEncoder


class GeometryProjection(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(258, output_dim, kernel_size=1),
            nn.GELU(approximate="tanh"),
        )
        self.spatial_norm = nn.LayerNorm(output_dim)
        self.spatial_type_embedding = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.cls_norm = nn.LayerNorm(1024)
        self.metric_projection = nn.Sequential(
            nn.Linear(1025, output_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.metric_type_embedding = nn.Parameter(torch.zeros(1, 1, output_dim))

    @staticmethod
    def normalized_uv(features: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = features.shape
        y = (
            torch.arange(height, device=features.device, dtype=features.dtype) + 0.5
        ) / height * 2 - 1
        x = (
            torch.arange(width, device=features.device, dtype=features.dtype) + 0.5
        ) / width * 2 - 1
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy]).unsqueeze(0).expand(batch_size, -1, -1, -1)

    def project_spatial(self, features: torch.Tensor) -> torch.Tensor:
        uv = self.normalized_uv(features)
        features = self.spatial_projection(torch.cat([features, uv], dim=1))
        features = features.flatten(2).transpose(1, 2)
        return self.spatial_norm(features) + self.spatial_type_embedding

    def project_metric(
        self, cls_token: torch.Tensor, log_scale: torch.Tensor
    ) -> torch.Tensor:
        metric_input = torch.cat(
            [
                self.cls_norm(cls_token),
                log_scale.clamp(-10, 10).unsqueeze(-1),
            ],
            dim=-1,
        )
        metric = self.metric_projection(metric_input).unsqueeze(1)
        return metric + self.metric_type_embedding

    def forward(
        self,
        spatial_features: torch.Tensor,
        cls_token: torch.Tensor,
        log_scale: torch.Tensor,
    ) -> torch.Tensor:
        spatial = self.project_spatial(spatial_features)
        metric = self.project_metric(cls_token, log_scale)
        return torch.cat([metric, spatial], dim=1)


class MoGeVFMConditioner(nn.Module):
    def __init__(
        self,
        config: MQDepthConfig,
        weights: str | Path,
        output_dim: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.config = config
        self._blank_global_cache = {}
        checkpoint = torch.load(
            weights, map_location="cpu", weights_only=True, mmap=True
        )
        self.moge = GeometryEncoder(**checkpoint["model_config"])
        self.moge.load_state_dict(checkpoint["model"], strict=True, assign=True)
        self.moge.to(dtype=dtype)
        self.projection = GeometryProjection(output_dim).to(dtype=dtype)

    def aspect_padding(self, height: int, width: int) -> tuple[int, int, int, int]:
        maximum = self.config.geometry_max_aspect_ratio
        if width / height > maximum:
            total = math.ceil(width / maximum) - height
            top = total // 2
            return 0, 0, top, total - top
        if height / width > maximum:
            total = math.ceil(height / maximum) - width
            left = total // 2
            return left, total - left, 0, 0
        return 0, 0, 0, 0

    def prepare_image(
        self, canonical_image: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        image = ((canonical_image.float() + 1) * 0.5).clamp(0, 1)
        padding = self.aspect_padding(*image.shape[-2:])
        if any(padding):
            image = F.pad(image, padding, mode="replicate")
        return image.to(dtype=self.moge.dtype), padding

    @staticmethod
    def crop_features(
        features: torch.Tensor,
        padding: tuple[int, int, int, int],
        padded_size: tuple[int, int],
    ) -> torch.Tensor:
        if not any(padding):
            return features
        left, right, top, bottom = padding
        image_height, image_width = padded_size
        height, width = features.shape[-2:]
        y0 = round(top / image_height * height)
        y1 = round((image_height - bottom) / image_height * height)
        x0 = round(left / image_width * width)
        x1 = round((image_width - right) / image_width * width)
        return features[..., y0 : max(y0 + 1, y1), x0 : max(x0 + 1, x1)]

    def pool_features(
        self, features: torch.Tensor, original_size: tuple[int, int]
    ) -> torch.Tensor:
        height, width = original_size
        aspect_ratio = width / height
        count = self.config.geometry_pool_tokens
        pool_height = max(1, round((count / aspect_ratio) ** 0.5))
        pool_width = max(1, round((count * aspect_ratio) ** 0.5))
        return F.adaptive_avg_pool2d(features, (pool_height, pool_width))

    def forward(self, canonical_image: torch.Tensor) -> torch.Tensor:
        image, padding = self.prepare_image(canonical_image)
        spatial, cls_token, log_scale = self.moge(
            image, num_tokens=self.config.geometry_tokens
        )
        if not self.config.moge2_cls_token:
            key = (tuple(image.shape), image.device, image.dtype, self.config.geometry_tokens)
            global_features = self._blank_global_cache.get(key)
            if global_features is None:
                cls_token, log_scale = self.moge.global_features(
                    torch.ones_like(image), self.config.geometry_tokens
                )
                global_features = (cls_token.detach().clone(), log_scale.detach().clone())
                if len(self._blank_global_cache) >= 256:
                    self._blank_global_cache.pop(next(iter(self._blank_global_cache)))
                self._blank_global_cache[key] = global_features
            cls_token, log_scale = global_features
        spatial = self.crop_features(spatial, padding, image.shape[-2:])
        spatial = self.pool_features(spatial, canonical_image.shape[-2:])
        return self.projection(spatial, cls_token, log_scale)
