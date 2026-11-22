import math

import torch
from torch import nn
from torch.nn import functional as F


def normalized_view_plane_uv(width, height, aspect_ratio, dtype, device):
    span_x = aspect_ratio / math.sqrt(1 + aspect_ratio**2)
    span_y = 1 / math.sqrt(1 + aspect_ratio**2)
    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim=1024, heads=16):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, tokens, dim = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.heads, dim // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.permute(0, 2, 1, 3).reshape(batch, tokens, dim))


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = FeedForward(dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, 14, stride=14)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 1024
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, dim))
        self.patch_embed = PatchEmbedding(dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(24)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def _positions(self, x, height, width):
        if x.shape[1] - 1 == self.pos_embed.shape[1] - 1 and height == width:
            return self.pos_embed
        positions = self.pos_embed.float()
        grid = int(math.sqrt(positions.shape[1] - 1))
        patches = positions[:, 1:].reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
        patches = F.interpolate(
            patches,
            scale_factor=((height / 14 + 0.1) / grid, (width / 14 + 0.1) / grid),
            mode="bicubic",
            antialias=False,
        )
        patches = patches.permute(0, 2, 3, 1).flatten(1, 2)
        return torch.cat([positions[:, :1], patches], dim=1).to(x.dtype)

    def prepare_tokens(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        x = self.patch_embed(image)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        return x + self._positions(x, height, width)

    def get_intermediate_layers(
        self, image: torch.Tensor, layers: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        x = self.prepare_tokens(image)
        outputs = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in layers:
                value = self.norm(x)
                outputs.append((value[:, 1:], value[:, 0]))
        return outputs

    def forward(
        self, image: torch.Tensor, layers: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.get_intermediate_layers(image, layers)


class DinoEncoder(nn.Module):
    def __init__(self, backbone, intermediate_layers, dim_out):
        super().__init__()
        if backbone != "dinov2_vitl14":
            raise ValueError("Unsupported geometry backbone")
        self.intermediate_layers = intermediate_layers
        self.backbone = DinoBackbone()
        self.output_projections = nn.ModuleList(
            [nn.Conv2d(1024, dim_out, 1) for _ in intermediate_layers]
        )
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, image, token_rows, token_cols, return_class_token=True):
        image = F.interpolate(
            image,
            (token_rows * 14, token_cols * 14),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        outputs = self.backbone(
            (image - self.image_mean) / self.image_std, self.intermediate_layers
        )
        features = torch.stack(
            [
                projection(
                    value.permute(0, 2, 1)
                    .unflatten(2, (token_rows, token_cols))
                    .contiguous()
                )
                for projection, (value, _) in zip(self.output_projections, outputs)
            ],
            dim=1,
        ).sum(dim=1)
        if return_class_token:
            return features, outputs[-1][1]
        return features


class ResidualConvBlock(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(dim, hidden, 3, padding=1, padding_mode="replicate"),
            nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(hidden, dim, 3, padding=1, padding_mode="replicate"),
        )

    def forward(self, x):
        return self.layers(x) + x


class ConvStack(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_res_blocks,
        dim_out,
        num_res_blocks,
        resamplers,
        res_block_in_norm,
        res_block_hidden_norm,
    ):
        super().__init__()
        if (
            res_block_in_norm != "none"
            or res_block_hidden_norm != "none"
            or resamplers[0] != "conv_transpose"
        ):
            raise ValueError("Unsupported geometry configuration")
        dims = dim_res_blocks[:2]
        outputs = [None, None] if dim_out is None else dim_out[:2]
        self.input_blocks = nn.ModuleList(
            [
                nn.Conv2d(source, target, 1) if source is not None else nn.Identity()
                for source, target in zip(dim_in[:2], dims)
            ]
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(*[ResidualConvBlock(dim, dim) for _ in range(count)])
                for dim, count in zip(dims, num_res_blocks[:2])
            ]
        )
        self.output_blocks = nn.ModuleList(
            [
                nn.Conv2d(dim, out, 1) if out is not None else nn.Identity()
                for dim, out in zip(dims, outputs)
            ]
        )
        self.resamplers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(dims[0], dims[1], 2, stride=2),
                    nn.Conv2d(dims[1], dims[1], 3, padding=1, padding_mode="replicate"),
                )
            ]
        )

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        features = None
        for level, input_block in enumerate(self.input_blocks):
            value = input_block(inputs[level])
            features = value if level == 0 else features + value
            features = self.res_blocks[level](features)
            outputs.append(self.output_blocks[level](features))
            if level < len(self.resamplers):
                features = self.resamplers[level](features)
        return outputs


class GeometryEncoder(nn.Module):
    def __init__(self, encoder, neck, points_head, scale_head):
        super().__init__()
        self.encoder = DinoEncoder(**encoder)
        self.neck = ConvStack(**neck)
        self.points_head = ConvStack(**points_head)
        dims = scale_head["dims"]
        layers = []
        for index, (source, target) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(source, target))
            if index < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.scale_head = nn.Sequential(*layers)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def prepare_pyramid(
        self, image: torch.Tensor, num_tokens: int
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        batch_size, _, height, width = image.shape
        aspect_ratio = width / height
        base_height = round((num_tokens / aspect_ratio) ** 0.5)
        base_width = round((num_tokens * aspect_ratio) ** 0.5)
        feature, cls_token = self.encoder(
            image, base_height, base_width, return_class_token=True
        )
        inputs = []
        for level in range(len(self.neck.input_blocks)):
            uv = normalized_view_plane_uv(
                width=base_width * 2**level,
                height=base_height * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=image.dtype,
                device=image.device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
            inputs.append(torch.cat([feature, uv], dim=1) if level == 0 else uv)
        return inputs, cls_token

    def forward(
        self, image: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, cls_token = self.prepare_pyramid(image, num_tokens)
        shared_features = self.neck(inputs)
        point_features = self.points_head(shared_features)
        log_scale = self.scale_head(cls_token).squeeze(-1)
        return point_features[-1], cls_token, log_scale

    def global_features(
        self, image: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aspect_ratio = image.shape[-1] / image.shape[-2]
        height = round((num_tokens / aspect_ratio) ** .5)
        width = round((num_tokens * aspect_ratio) ** .5)
        _, cls_token = self.encoder(image, height, width, return_class_token=True)
        return cls_token, self.scale_head(cls_token).squeeze(-1)
