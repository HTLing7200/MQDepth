from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .config import MQDepthConfig
from .connector import MQConnector


class Qwen3MQConditioner(nn.Module):
    def __init__(
        self,
        config: MQDepthConfig,
        weights: str | Path,
        output_dim: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.config = config
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            weights,
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.connector = MQConnector(
            in_dim=self.qwen.config.text_config.hidden_size,
            out_dim=output_dim,
            num_layers=config.connector_layers,
            num_heads=config.connector_heads,
        ).to(dtype=dtype)
        tokenizer = self.processor.tokenizer
        self.begin_id = tokenizer.convert_tokens_to_ids("<begin_of_img>")
        self.end_id = tokenizer.convert_tokens_to_ids("<end_of_img>")
        query_tokens = "".join(f"<img{index}>" for index in range(config.queries))
        self.suffix = f"\n<begin_of_img>{query_tokens}<end_of_img>"
        self._use_float32_patch_embedding()

    def _use_float32_patch_embedding(self) -> None:
        visual = self.qwen.model.visual
        patch_embed = visual.patch_embed

        def forward(hidden_states: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                hidden_states = hidden_states.reshape(
                    -1,
                    patch_embed.in_channels,
                    patch_embed.temporal_patch_size,
                    patch_embed.patch_size,
                    patch_embed.patch_size,
                )
                projection = patch_embed.proj
                bias = None if projection.bias is None else projection.bias.float()
                hidden_states = F.conv3d(
                    hidden_states.float(),
                    projection.weight.float(),
                    bias,
                    stride=projection.stride,
                    padding=projection.padding,
                    dilation=projection.dilation,
                    groups=projection.groups,
                )
                hidden_states = hidden_states.reshape(-1, patch_embed.embed_dim)
            return hidden_states.to(visual.pos_embed.weight.dtype)

        patch_embed.forward = forward

    def prepare_inputs(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.config.prompt + self.suffix},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
            do_rescale=False,
            do_normalize=False,
            do_resize=True,
        )
        device = next(self.qwen.parameters()).device
        return {key: value.to(device) for key, value in inputs.items()}

    def extract_queries(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        begin = torch.where(input_ids == self.begin_id)[1]
        end = torch.where(input_ids == self.end_id)[1]
        selected = (positions > begin[:, None]) & (positions < end[:, None])
        return hidden_states[selected].reshape(batch_size, self.config.queries, -1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        inputs = self.prepare_inputs(images)
        outputs = self.qwen.model(
            **inputs,
            output_hidden_states=False,
            use_cache=False,
        )
        queries = self.extract_queries(outputs.last_hidden_state, inputs["input_ids"])
        return self.connector(queries)
