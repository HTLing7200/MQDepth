import torch
from torch import nn


class MQConnector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=in_dim,
                    nhead=num_heads,
                    dim_feedforward=in_dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        features = hidden_states
        for layer in self.layers:
            features = layer(features)
        return self.proj(self.norm(features))
