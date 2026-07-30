from __future__ import annotations

import torch
from torch import nn

from .geometry_encoder import GeometryEncoder


class FourierCoordinates(nn.Module):
    def __init__(self, frequencies: int):
        super().__init__()
        bands = 2.0 ** torch.arange(int(frequencies), dtype=torch.float32)
        self.register_buffer("bands", bands, persistent=True)

    @property
    def output_dim(self) -> int:
        return 3 + 6 * len(self.bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        phase = coords[..., None] * self.bands * torch.pi
        return torch.cat(
            [coords, torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)],
            dim=-1,
        )


class WSSPINNModel(nn.Module):
    """Shared smooth coordinate field used unchanged by F0-U/F0-UP/F1."""

    def __init__(self, model_config: dict):
        super().__init__()
        latent = int(model_config["geometry_latent"])
        self.geometry = GeometryEncoder(
            input_dim=15,
            hidden_dim=int(model_config["geometry_hidden"]),
            latent_dim=latent,
        )
        self.coordinates = FourierCoordinates(int(model_config["fourier_frequencies"]))
        hidden = int(model_config["field_hidden"])
        layers = int(model_config["field_layers"])
        input_dim = self.coordinates.output_dim + latent

        def make_trunk() -> nn.Sequential:
            modules: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.SiLU()]
            for _ in range(max(1, layers) - 1):
                modules.extend([nn.Linear(hidden, hidden), nn.SiLU()])
            return nn.Sequential(*modules)

        self.architecture_version = str(
            model_config.get("architecture_version", "shared_v1")
        )
        if self.architecture_version == "shared_v1":
            self.trunk = make_trunk()
            self.head = nn.Linear(hidden, 5)
        elif self.architecture_version == "split_v2":
            self.wss_trunk = make_trunk()
            self.field_trunk = make_trunk()
            self.wss_head = nn.Linear(hidden, 1)
            self.field_head = nn.Linear(hidden, 4)
        else:
            raise ValueError(
                f"unsupported model.architecture_version: {self.architecture_version}"
            )

    def forward(
        self, coords: torch.Tensor, geometry_descriptor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        latent = self.geometry(geometry_descriptor)
        if latent.shape[0] == 1 and coords.shape[0] != 1:
            latent = latent.expand(coords.shape[0], -1)
        features = torch.cat([self.coordinates(coords), latent], dim=-1)
        if self.architecture_version == "shared_v1":
            output = self.head(self.trunk(features))
            log_wss = output[:, 0]
            field = output[:, 1:5]
        else:
            log_wss = self.wss_head(self.wss_trunk(features))[:, 0]
            field = self.field_head(self.field_trunk(features))
        return {
            "log_wss_direct": log_wss,
            "velocity": field[:, 0:3],
            "pressure": field[:, 3],
        }
