import math

import numpy as np
from PIL import Image, ImageDraw
import torch

from models.codec import DepthColorCodec


class DepthColorizer:
    def __init__(
        self,
        min_depth: float = 0.1,
        max_depth: float = 655.0,
        colormap: str = "cube",
        reverse: bool = False,
    ):
        if not 0 < min_depth < max_depth or not math.isfinite(max_depth):
            raise ValueError("Expected finite 0 < min_depth < max_depth")
        if colormap not in ("cube", "gray"):
            raise ValueError("Unsupported colormap")
        self.codec = DepthColorCodec(min_depth, max_depth)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.colormap = colormap
        self.reverse = reverse

    @torch.inference_mode()
    def colorize(self, depth: np.ndarray) -> Image.Image:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError("Expected H x W depth")
        valid = ~np.isnan(depth) & (depth > 0)
        rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
        for top in range(0, depth.shape[0], 128):
            rows = depth[top : top + 128]
            values = torch.from_numpy(np.array(rows, copy=True))[None, None]
            unit = self.codec.depth_to_unit(values)
            unit = torch.nan_to_num(unit, nan=0.0)
            if self.reverse:
                unit = 1 - unit
            colors = (
                self.codec.unit_to_rgb(unit)
                if self.colormap == "cube"
                else unit.expand(-1, 3, -1, -1)
            )
            array = colors[0].permute(1, 2, 0).numpy()
            rgb[top : top + 128] = np.rint(array * 255).clip(0, 255).astype(np.uint8)
        rgb[~valid] = 0
        if valid.all():
            return Image.fromarray(rgb)
        alpha = (valid.astype(np.uint8) * 255)[..., None]
        return Image.fromarray(np.concatenate([rgb, alpha], axis=-1))

    @torch.inference_mode()
    def decode(self, image: Image.Image) -> np.ndarray:
        alpha = np.asarray(image.getchannel("A")) if "A" in image.getbands() else None
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        depth = np.empty(rgb.shape[:2], dtype=np.float32)
        for top in range(0, rgb.shape[0], 128):
            rows = torch.from_numpy(np.array(rgb[top : top + 128], copy=True)).permute(
                2, 0, 1
            )[None]
            unit = (
                self.codec.rgb_to_unit(rows) if self.colormap == "cube" else rows[:, :1]
            )
            if self.reverse:
                unit = 1 - unit
            depth[top : top + 128] = self.codec.unit_to_depth(unit)[0, 0].numpy()
        if alpha is not None:
            depth[alpha == 0] = np.nan
        return depth

    def colorbar(self, width: int, height: int = 20) -> Image.Image:
        values = np.geomspace(self.min_depth, self.max_depth, width, dtype=np.float32)
        return self.colorize(np.repeat(values[None], height, axis=0)).convert("RGB")

    def preview(
        self, image: Image.Image, depth: np.ndarray, panel_width: int = 768
    ) -> Image.Image:
        image = image.convert("RGB")
        height = max(1, round(image.height * panel_width / image.width))
        source = image.resize((panel_width, height), Image.Resampling.BILINEAR)
        colored = (
            self.colorize(depth)
            .convert("RGB")
            .resize((panel_width, height), Image.Resampling.BILINEAR)
        )
        canvas = Image.new("RGB", (panel_width * 2, height + 100), "white")
        canvas.paste(source, (0, 32))
        canvas.paste(colored, (panel_width, 32))
        drawing = ImageDraw.Draw(canvas)
        drawing.text((12, 10), "RGB", fill="black")
        drawing.text((panel_width + 12, 10), "DEPTH (m)", fill="black")
        margin = 20
        bar_width = canvas.width - margin * 2
        canvas.paste(self.colorbar(bar_width), (margin, height + 45))
        lower, upper = math.log(self.min_depth), math.log(self.max_depth)
        ticks = [self.min_depth, 1.0, 10.0, 100.0, self.max_depth]
        for value in sorted(set(ticks)):
            if not self.min_depth <= value <= self.max_depth:
                continue
            position = (math.log(value) - lower) / (upper - lower)
            x = margin + round(position * (bar_width - 1))
            label = f"{value:g} m"
            box = drawing.textbbox((0, 0), label)
            x = min(
                max(0, x - (box[2] - box[0]) // 2), canvas.width - (box[2] - box[0])
            )
            drawing.text((x, height + 72), label, fill="black")
        return canvas

    def to_dict(self) -> dict:
        return dict(
            colormap=self.colormap,
            min_depth_m=self.min_depth,
            max_depth_m=self.max_depth,
            reverse=self.reverse,
            normalization="logarithmic",
            invalid_pixels="transparent",
        )
