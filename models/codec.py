import math

import torch
from torch import nn


class DepthColorCodec(nn.Module):
    def __init__(self, min_depth: float, max_depth: float):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.register_buffer(
            "vertices",
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            persistent=False,
        )

    def rgb_to_unit(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.float().clamp(0, 1)
        batch_size, _, height, width = rgb.shape
        vertices = self.vertices.to(device=rgb.device, dtype=rgb.dtype)
        starts = vertices[:-1]
        directions = vertices[1:] - starts
        points = rgb.flatten(2).transpose(1, 2).unsqueeze(-2)
        denominator = directions.square().sum(dim=-1)
        fraction = ((points - starts) * directions).sum(dim=-1) / denominator
        fraction = fraction.clamp(0, 1)
        projected = starts + fraction.unsqueeze(-1) * directions
        distance_squared = (points - projected).square().sum(dim=-1)
        segment = distance_squared.argmin(dim=-1)
        fraction = fraction.gather(-1, segment.unsqueeze(-1)).squeeze(-1)
        unit = (segment + fraction) / (vertices.shape[0] - 1)
        return unit.reshape(batch_size, 1, height, width)

    def unit_to_depth(self, unit: torch.Tensor) -> torch.Tensor:
        log_min = math.log(self.min_depth)
        log_range = math.log(self.max_depth) - log_min
        return torch.exp(unit * log_range + log_min)

    def depth_to_unit(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.float().clamp(self.min_depth, self.max_depth)
        log_min = math.log(self.min_depth)
        log_range = math.log(self.max_depth) - log_min
        return (depth.log() - log_min) / log_range

    def unit_to_rgb(self, unit: torch.Tensor) -> torch.Tensor:
        if unit.ndim != 4 or unit.shape[1] != 1:
            raise ValueError("Expected depth with shape B x 1 x H x W")
        vertices = self.vertices.to(device=unit.device, dtype=torch.float32)
        position = unit.float().clamp(0, 1)[:, 0] * (vertices.shape[0] - 1)
        segment = position.floor().long().clamp(max=vertices.shape[0] - 2)
        fraction = (position - segment).unsqueeze(-1)
        rgb = vertices[segment] + fraction * (vertices[segment + 1] - vertices[segment])
        return rgb.permute(0, 3, 1, 2)

    def encode(self, depth: torch.Tensor) -> torch.Tensor:
        return self.unit_to_rgb(self.depth_to_unit(depth))

    def decode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.unit_to_depth(self.rgb_to_unit(rgb))

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        rgb = (target.float() + 1) * 0.5
        return self.decode_rgb(rgb)


DepthDecoder = DepthColorCodec
