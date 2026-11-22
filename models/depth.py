import math
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaTransformer2DModel,
)
from PIL import Image
from safetensors.torch import load_file
from torch import nn

from .codec import DepthDecoder
from .config import MQDepthConfig
from .moge import MoGeVFMConditioner
from .processing import ImageProcessor
from .qwen import Qwen3MQConditioner


class OutdoorDepth(nn.Module):
    def __init__(self, weights: str | Path | None = None, device: str | torch.device = "cuda",
                 moge2_cls_token: bool | None = None):
        super().__init__()
        if weights is None:
            raise ValueError("Provide a model weights directory")
        directory = Path(weights)
        self.config = MQDepthConfig.from_directory(directory)
        if moge2_cls_token is not None:
            self.config = replace(self.config, moge2_cls_token=moge2_cls_token)
        device = torch.device(device)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.vae = AutoencoderDC.from_pretrained(
            directory / "vae",
            torch_dtype=dtype,
            local_files_only=True,
        )
        self.transformer = SanaTransformer2DModel.from_pretrained(
            directory / "transformer",
            torch_dtype=dtype,
            local_files_only=True,
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            directory / "scheduler",
            local_files_only=True,
        )
        output_dim = int(self.transformer.config.caption_channels)
        self.conditioner = Qwen3MQConditioner(
            config=self.config,
            weights=directory / "qwen",
            output_dim=output_dim,
            dtype=dtype,
        )
        self.vfm_conditioner = MoGeVFMConditioner(
            config=self.config,
            weights=directory / "geometry.pt",
            output_dim=output_dim,
            dtype=dtype,
        )
        self._load_conditioning_weights(directory / "conditioning.safetensors")
        self.depth_decoder = DepthDecoder(self.config.depth_min, self.config.depth_max)
        self.image_processor = ImageProcessor(self.config)
        self.to(device).eval().requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    def _load_conditioning_weights(self, path: str | Path) -> None:
        state = load_file(path)
        connector_state = {
            key.removeprefix("connector."): value
            for key, value in state.items()
            if key.startswith("connector.")
        }
        projection_state = {
            key.removeprefix("geometry."): value
            for key, value in state.items()
            if key.startswith("geometry.")
        }
        self.conditioner.connector.load_state_dict(connector_state, strict=True)
        self.vfm_conditioner.projection.load_state_dict(projection_state, strict=True)

    def encode_images(self, image: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        latents = self.vae.encode(image.to(dtype=vae_dtype)).latent
        shift = getattr(self.vae.config, "shift_factor", None)
        if shift is not None:
            latents = latents - shift
        return latents * float(self.vae.config.scaling_factor)

    def encode_condition(
        self, canonical_image: torch.Tensor, qwen_image: torch.Tensor
    ) -> torch.Tensor:
        query_tokens = self.conditioner(qwen_image)
        geometry_tokens = self.vfm_conditioner(canonical_image)
        return torch.cat([query_tokens, geometry_tokens], dim=1).to(dtype=self.dtype)

    def prepare_latents(
        self, image_latents: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        return torch.randn(
            image_latents.shape,
            device=image_latents.device,
            dtype=torch.float32,
            generator=generator,
        )

    def denoise(
        self,
        latents: torch.Tensor,
        image_latents: torch.Tensor,
        condition: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
        for timestep in self.scheduler.timesteps:
            model_input = torch.cat(
                [
                    latents.to(dtype=image_latents.dtype),
                    image_latents,
                ],
                dim=1,
            )
            timestep_batch = (
                timestep.unsqueeze(0).expand(latents.shape[0]).to(self.device)
            )
            prediction = self.transformer(
                hidden_states=model_input,
                timestep=timestep_batch,
                encoder_hidden_states=condition,
                encoder_attention_mask=None,
            ).sample
            latents = self.scheduler.step(
                prediction.float(), timestep, latents
            ).prev_sample
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=vae_dtype) / float(self.vae.config.scaling_factor)
        shift = getattr(self.vae.config, "shift_factor", None)
        if shift is not None:
            latents = latents + shift
        return self.vae.decode(latents).sample.clamp(-1, 1)

    @staticmethod
    def aggregate_predictions(members: torch.Tensor) -> torch.Tensor:
        return torch.quantile(members.float(), 0.5, dim=0, keepdim=True)

    @torch.inference_mode()
    def forward(
        self,
        canonical_image: torch.Tensor,
        qwen_image: torch.Tensor,
        num_inference_steps: int = 20,
        ensemble_size: int = 4,
        generator: torch.Generator | None = None,
        stage_callback: Callable[[str], None] | None = None,
    ) -> torch.Tensor:
        if canonical_image.shape[0] != 1 or qwen_image.shape[0] != 1:
            raise ValueError("Expected one input image")
        if num_inference_steps < 1 or ensemble_size not in (1, 4):
            raise ValueError("Steps must be positive; ensemble size must be 1 or 4")
        image_latents = self.encode_images(canonical_image)
        image_latents = image_latents.repeat(ensemble_size, 1, 1, 1)
        latents = self.prepare_latents(image_latents, generator)
        if stage_callback is not None:
            stage_callback("vae_encode")
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            condition = self.encode_condition(canonical_image, qwen_image)
            condition = condition.repeat(ensemble_size, 1, 1)
            if stage_callback is not None:
                stage_callback("conditioning")
            latents = self.denoise(
                latents, image_latents, condition, num_inference_steps
            )
            if stage_callback is not None:
                stage_callback("diffusion")
        target = self.decode_latents(latents)
        if stage_callback is not None:
            stage_callback("vae_decode")
        members = self.depth_decoder(target)
        prediction = self.aggregate_predictions(members)
        if stage_callback is not None:
            stage_callback("depth_decode")
        return prediction

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        fx: float,
        steps: int = 20,
        ensemble: int = 4,
        seed: int = 42,
    ) -> np.ndarray:
        if not math.isfinite(fx) or fx <= 0:
            raise ValueError("fx must be positive and finite")
        prepared = self.image_processor(image, self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        canonical_depth = self(
            prepared.canonical_image,
            prepared.qwen_image,
            num_inference_steps=steps,
            ensemble_size=ensemble,
            generator=generator,
        )
        metric_depth = self.image_processor.restore_depth(canonical_depth, fx, prepared)
        return metric_depth[0, 0].float().cpu().numpy()
