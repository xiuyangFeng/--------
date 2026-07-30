from __future__ import annotations

import torch
from torch import nn


class GeometryEncoder(nn.Module):
    def __init__(self, input_dim: int = 15, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.ndim == 1:
            descriptor = descriptor.unsqueeze(0)
        return self.network(descriptor)

