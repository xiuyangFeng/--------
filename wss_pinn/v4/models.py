"""Matched PointNet/PointNet++ support encoders with a continuous V4 decoder."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import nn
from torch_scatter import scatter

from .waveform import PERIOD_S, q_nom_torch


def _activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation={name!r}")


def _mlp(channels: list[int], activation: str, *, final_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (cin, cout) in enumerate(zip(channels[:-1], channels[1:])):
        layers.append(nn.Linear(int(cin), int(cout)))
        if final_activation or index + 1 < len(channels) - 1:
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


def _deterministic_fps(points: torch.Tensor, count: int, identity: str, seed: int) -> torch.Tensor:
    count = min(int(count), int(points.size(0)))
    digest = hashlib.sha256(f"{seed}|{identity}".encode("utf-8")).digest()
    current = int.from_bytes(digest[:8], "little") % int(points.size(0))
    selected = torch.empty(count, dtype=torch.long, device=points.device)
    distance = torch.full((points.size(0),), float("inf"), dtype=points.dtype, device=points.device)
    with torch.no_grad():
        for offset in range(count):
            selected[offset] = current
            delta = points - points[current]
            distance = torch.minimum(distance, torch.sum(delta * delta, dim=-1))
            current = int(torch.argmax(distance).item())
    return selected


class PointNetSupportEncoder(nn.Module):
    def __init__(self, in_dim: int, channels: list[int], activation: str):
        super().__init__()
        self.branch = _mlp([in_dim, *channels], activation)

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        batch: torch.Tensor,
        case_ids: list[str],
        seed: int,
    ) -> torch.Tensor:
        del coords, case_ids, seed
        return scatter(self.branch(features), batch, dim=0, reduce="max")


class PointNetPlusPlusSupportEncoder(nn.Module):
    """Hierarchical FPS/kNN encoder used only on support points.

    Query coordinates never enter FPS or kNN, so moving a physics collocation
    point cannot change a discrete neighbour set.  Strong-form derivatives are
    taken only through the smooth decoder below.
    """

    def __init__(self, in_dim: int, config: dict[str, Any], activation: str):
        super().__init__()
        channels = [int(value) for value in config["local_channels"]]
        self.centroids = int(config["centroids"])
        self.neighbors = int(config["neighbors"])
        self.local = _mlp([in_dim + 3, *channels], activation)

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        batch: torch.Tensor,
        case_ids: list[str],
        seed: int,
    ) -> torch.Tensor:
        latents = []
        for case_index, case_id in enumerate(case_ids):
            mask = batch == case_index
            case_coords = coords[mask]
            case_features = features[mask]
            fps = _deterministic_fps(case_coords, self.centroids, case_id, seed)
            centroids = case_coords[fps]
            distance = torch.cdist(centroids, case_coords)
            neighbors = torch.topk(
                distance,
                k=min(self.neighbors, int(case_coords.size(0))),
                largest=False,
                sorted=False,
            ).indices
            local_coords = case_coords[neighbors] - centroids[:, None, :]
            local_features = case_features[neighbors]
            fused = torch.cat([local_coords, local_features], dim=-1)
            encoded = self.local(fused).max(dim=1).values
            latents.append(encoded.max(dim=0).values)
        return torch.stack(latents, dim=0)


class BCRCRFieldModel(nn.Module):
    def __init__(self, config: dict[str, Any], *, backbone: str, transient: bool):
        super().__init__()
        activation = str(config["activation"])
        support_dim = int(config["support_in_dim"])
        latent_dim = int(config["latent_dim"])
        if backbone == "pointnet":
            channels = [int(value) for value in config["pointnet_channels"]]
            if channels[-1] != latent_dim:
                raise ValueError("PointNet latent width mismatch")
            self.support_encoder = PointNetSupportEncoder(support_dim, channels, activation)
        elif backbone == "pointnetpp":
            if int(config["pointnetpp"]["local_channels"][-1]) != latent_dim:
                raise ValueError("PointNet++ latent width mismatch")
            self.support_encoder = PointNetPlusPlusSupportEncoder(
                support_dim, config["pointnetpp"], activation
            )
        else:
            raise ValueError(f"unsupported backbone={backbone!r}")
        self.backbone = backbone
        self.transient = bool(transient)
        bc_channels = [int(value) for value in config["bc_channels"]]
        self.bc_encoder = _mlp([int(config["bc_in_dim"]), *bc_channels], activation)
        time_dim = 4 if self.transient else 0
        decoder = [int(value) for value in config["decoder_channels"]]
        self.decoder = _mlp(
            [3 + time_dim + latent_dim + bc_channels[-1], *decoder], activation
        )
        self.output = nn.Linear(decoder[-1], 4)

    def encode_support(
        self,
        support_coords: torch.Tensor,
        support_features: torch.Tensor,
        support_batch: torch.Tensor,
        bc_vector: torch.Tensor,
        case_ids: list[str],
        seed: int,
    ) -> dict[str, torch.Tensor]:
        geometry_latent = self.support_encoder(
            support_coords, support_features, support_batch, case_ids, seed
        )
        return {
            "geometry_latent": geometry_latent,
            "bc_latent": self.bc_encoder(bc_vector),
        }

    def time_features(self, time_s: torch.Tensor) -> torch.Tensor:
        if not self.transient:
            raise ValueError("steady model has no time features")
        t_norm = time_s / PERIOD_S
        return torch.cat(
            [
                t_norm,
                torch.sin(2.0 * math.pi * t_norm),
                torch.cos(2.0 * math.pi * t_norm),
                q_nom_torch(time_s) / 1.0e-4,
            ],
            dim=-1,
        )

    def decode_query(
        self,
        encoded: dict[str, torch.Tensor],
        query_coords: torch.Tensor,
        query_batch: torch.Tensor,
        query_time_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pieces = [query_coords]
        if self.transient:
            if query_time_s is None:
                raise ValueError("transient decoder requires query_time_s")
            pieces.append(self.time_features(query_time_s))
        elif query_time_s is not None:
            raise ValueError("steady decoder received time input")
        pieces.extend(
            [
                encoded["geometry_latent"][query_batch],
                encoded["bc_latent"][query_batch],
            ]
        )
        return self.output(self.decoder(torch.cat(pieces, dim=-1)))


def build_model(config: Any) -> BCRCRFieldModel:
    return BCRCRFieldModel(
        config["model"],
        backbone=str(config["experiment"]["backbone"]),
        transient=str(config["experiment"]["temporal_mode"]) == "transient_81",
    )
