"""Historical PointNet/PointNet++ anchors with PINN-compatible query heads.

Only the architecture and sampling contracts are inherited.  No WSS
checkpoint is loaded.  Query-coordinate paths use smooth activations and a
differentiable inverse-distance interpolation because the upstream PyG helper
computes its weights under ``torch.no_grad()``.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import nn
from torch_geometric.nn import knn
from torch_geometric.utils import scatter

from training_wss_min.baseline_models import PointNetPlusPlusRegressor


def _activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported smooth activation={name!r}")


def smooth_mlp(channels: list[int], activation: str, *, last_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (cin, cout) in enumerate(zip(channels[:-1], channels[1:])):
        layers.append(nn.Linear(cin, cout))
        is_last = index == len(channels) - 2
        if not is_last or last_act:
            # LayerNorm is point-local. BatchNorm would couple query points and
            # invalidate the usual pointwise PINN derivative interpretation.
            layers.extend([nn.LayerNorm(cout), _activation(activation)])
    return nn.Sequential(*layers)


def differentiable_knn_interpolate(
    features: torch.Tensor,
    support_pos: torch.Tensor,
    query_pos: torch.Tensor,
    support_batch: torch.Tensor,
    query_batch: torch.Tensor,
    *,
    k: int = 3,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """kNN assignment is discrete; distance weights remain differentiable."""
    with torch.no_grad():
        assignment = knn(
            support_pos,
            query_pos,
            k=int(k),
            batch_x=support_batch,
            batch_y=query_batch,
        )
        query_index, support_index = assignment[0], assignment[1]
    delta = support_pos[support_index] - query_pos[query_index]
    distance_sq = torch.sum(delta * delta, dim=-1).clamp_min(float(epsilon))
    weight = distance_sq.reciprocal()
    denominator = scatter(
        weight,
        query_index,
        dim=0,
        dim_size=query_pos.size(0),
        reduce="sum",
    ).clamp_min(float(epsilon))
    numerator = scatter(
        features[support_index] * weight.unsqueeze(-1),
        query_index,
        dim=0,
        dim_size=query_pos.size(0),
        reduce="sum",
    )
    return numerator / denominator.unsqueeze(-1)


class SmoothPointNetField(nn.Module):
    """P2V widths with a continuous support-conditioned query function."""

    def __init__(self, in_dim: int, config: dict[str, Any], activation: str):
        super().__init__()
        support_channels = [int(x) for x in config["support_channels"]]
        decoder_channels = [int(x) for x in config["decoder_channels"]]
        if support_channels != [256, 512] or decoder_channels != [512, 256]:
            raise ValueError("PointNet v1 must preserve the historical P2V widths")
        self.in_dim = int(in_dim)
        self.geometry_dim = max(0, self.in_dim - 3)
        self.local = smooth_mlp(
            [self.in_dim, *support_channels], activation, last_act=True
        )
        self.decoder = smooth_mlp(
            [support_channels[-1] * 2, *decoder_channels],
            activation,
            last_act=True,
        )
        self.output = nn.Linear(decoder_channels[-1], 4)

    def encode_support(
        self,
        support_pos: torch.Tensor,
        support_features: torch.Tensor,
        support_batch: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        local = self.local(support_features)
        global_feature = scatter(local, support_batch, dim=0, reduce="max")
        return {
            "support_pos": support_pos,
            "support_features": support_features,
            "support_batch": support_batch,
            "global_feature": global_feature,
        }

    def decode_query(
        self,
        encoded: dict[str, torch.Tensor],
        query_pos: torch.Tensor,
        query_batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.geometry_dim:
            query_geometry = differentiable_knn_interpolate(
                encoded["support_features"][:, 3:],
                encoded["support_pos"],
                query_pos,
                encoded["support_batch"],
                query_batch,
                k=3,
            )
            query_features = torch.cat([query_pos, query_geometry], dim=-1)
        else:
            query_features = query_pos
        query_local = self.local(query_features)
        fused = torch.cat(
            [query_local, encoded["global_feature"][query_batch]], dim=-1
        )
        return self.output(self.decoder(fused))


class SmoothPointNetPlusPlusField(nn.Module):
    """Pure D2 c125xk128 PointNet++ support encoder plus smooth query head."""

    def __init__(self, in_dim: int, config: dict[str, Any], activation: str):
        super().__init__()
        expected = {
            "sa_nsample": [128, 16, 16],
            "sa_center_counts": [125, 125, 32],
            "sa_grouping": ["knn_cover", "ball", "ball"],
        }
        for key, value in expected.items():
            if list(config[key]) != value:
                raise ValueError(f"PointNet++ v1 must preserve D2 {key}={value}")
        self.interpolation_k = int(config["fp_knn"])
        self.backbone = PointNetPlusPlusRegressor(
            in_dim=int(in_dim),
            width=int(config["width"]),
            sa_ratios=tuple(float(x) for x in config["sa_ratios"]),
            sa_radius=tuple(float(x) for x in config["sa_radius"]),
            sa_nsample=tuple(int(x) for x in config["sa_nsample"]),
            fp_knn=int(config["fp_knn"]),
            head_hidden=int(config["head_hidden"]),
            out_dim=4,
            dropout=0.0,
            sa_center_counts=tuple(int(x) for x in config["sa_center_counts"]),
            sa_center_sampling=str(config["sa_center_sampling"]),
            query_decoder="interpolate",
            sa_grouping=tuple(str(x) for x in config["sa_grouping"]),
            sa_blocks=(0, 0, 0),
            local_geope=False,
            local_geope_feature_indices=(),
            drop_path_rate=0.0,
            local_transformer_stages=(),
            edgeconv_stages=(),
            coarse_attention=False,
        )
        # The upstream head is never used. Removing it avoids silent parameters
        # that receive no gradient in this route.
        self.backbone.head = nn.Identity()
        width = int(config["width"])
        hidden = int(config["head_hidden"])
        self.query_head = nn.Sequential(
            nn.Linear(width + 3, hidden),
            _activation(activation),
            nn.Linear(hidden, hidden),
            _activation(activation),
            nn.Linear(hidden, 4),
        )

    def encode_support(
        self,
        support_pos: torch.Tensor,
        support_features: torch.Tensor,
        support_batch: torch.Tensor,
        **context: Any,
    ) -> dict[str, torch.Tensor]:
        pos, latent, batch = self.backbone.encode_support(
            support_pos, support_features, support_batch, **context
        )[:3]
        return {"support_pos": pos, "support_latent": latent, "support_batch": batch}

    def decode_query(
        self,
        encoded: dict[str, torch.Tensor],
        query_pos: torch.Tensor,
        query_batch: torch.Tensor,
    ) -> torch.Tensor:
        latent = differentiable_knn_interpolate(
            encoded["support_latent"],
            encoded["support_pos"],
            query_pos,
            encoded["support_batch"],
            query_batch,
            k=self.interpolation_k,
        )
        return self.query_head(torch.cat([latent, query_pos], dim=-1))


class ConditionalSmoothPointNetField(nn.Module):
    """V3 branch-conditioned smooth coordinate field without interpolation.

    Geometry features, when enabled, enter only the support/branch encoder.
    The physics query path is exactly ``query_xyz + broadcast(case_latent)``
    followed by Linear+tanh layers, so moving a query never triggers kNN/3NN
    or any query-neighbour discrete selection.
    """

    def __init__(self, in_dim: int, config: dict[str, Any]):
        super().__init__()
        branch = [int(value) for value in config["branch_channels"]]
        decoder = [int(value) for value in config["decoder_channels"]]
        if branch != [64, 128, 256] or decoder != [256, 256, 128]:
            raise ValueError(
                "V3 widths are frozen to branch 64-128-256 and decoder 256-256-128"
            )
        self.in_dim = int(in_dim)
        branch_layers: list[nn.Module] = []
        for cin, cout in zip([self.in_dim, *branch[:-1]], branch):
            branch_layers.extend([nn.Linear(cin, cout), nn.Tanh()])
        self.branch = nn.Sequential(*branch_layers)
        decoder_layers: list[nn.Module] = []
        decoder_inputs = [3 + branch[-1], *decoder[:-1]]
        for cin, cout in zip(decoder_inputs, decoder):
            decoder_layers.extend([nn.Linear(cin, cout), nn.Tanh()])
        self.decoder = nn.Sequential(*decoder_layers)
        self.output = nn.Linear(decoder[-1], 4)

    def encode_support(
        self,
        support_pos: torch.Tensor,
        support_features: torch.Tensor,
        support_batch: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        del support_pos
        local = self.branch(support_features)
        latent = scatter(local, support_batch, dim=0, reduce="max")
        return {"case_latent": latent}

    def decode_query(
        self,
        encoded: dict[str, torch.Tensor],
        query_pos: torch.Tensor,
        query_batch: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat([query_pos, encoded["case_latent"][query_batch]], dim=-1)
        return self.output(self.decoder(fused))


def _deterministic_fps(
    points: torch.Tensor,
    count: int,
    *,
    identity: str,
    seed: int,
) -> torch.Tensor:
    """Small deterministic farthest-point sampler used only on support points."""
    if int(count) <= 0 or int(count) > int(points.size(0)):
        raise ValueError("invalid deterministic FPS centroid count")
    digest = hashlib.sha256(f"{seed}|{identity}".encode("utf-8")).digest()
    selected = torch.empty(int(count), dtype=torch.long, device=points.device)
    current = int.from_bytes(digest[:8], "little") % int(points.size(0))
    distance = torch.full(
        (points.size(0),), float("inf"), dtype=points.dtype, device=points.device
    )
    with torch.no_grad():
        for offset in range(int(count)):
            selected[offset] = current
            delta = points - points[current]
            distance = torch.minimum(distance, torch.sum(delta * delta, dim=-1))
            current = int(torch.argmax(distance).item())
    return selected


class ConditionalFieldV4(nn.Module):
    """Matched global/local field with optional band-limited query PE.

    Global and local arms share exactly the same trainable layers.  The local
    arm replaces max-pool broadcasting with a positive-bandwidth, all-centroid
    Gaussian mixture; no query top-k/kNN decision is made.
    """

    def __init__(self, in_dim: int, config: dict[str, Any]):
        super().__init__()
        branch = [int(value) for value in config["branch_channels"]]
        decoder = [int(value) for value in config["decoder_channels"]]
        if branch != [64, 128, 256] or decoder != [256, 256, 128]:
            raise ValueError("V4 matched widths are frozen")
        self.conditioner = str(config["conditioner"])
        self.query_encoding = str(config["query_encoding"])
        self.local_centroids = int(config["local_centroids"])
        bandwidths = torch.tensor(
            [float(value) for value in config["local_bandwidths_normalized"]],
            dtype=torch.float32,
        )
        frequencies = torch.tensor(
            [
                float(value)
                for value in config[
                    "pe_frequencies_cycles_per_normalized_length"
                ]
            ],
            dtype=torch.float32,
        )
        self.register_buffer("local_bandwidths", bandwidths)
        self.register_buffer("pe_frequencies", frequencies)

        branch_layers: list[nn.Module] = []
        for cin, cout in zip([int(in_dim), *branch[:-1]], branch):
            branch_layers.extend([nn.Linear(cin, cout), nn.Tanh()])
        self.branch = nn.Sequential(*branch_layers)

        query_dim = 3
        if self.query_encoding == "pe":
            query_dim += 3 * 2 * len(frequencies)
        decoder_layers: list[nn.Module] = []
        for cin, cout in zip(
            [query_dim + branch[-1], *decoder[:-1]], decoder
        ):
            decoder_layers.extend([nn.Linear(cin, cout), nn.Tanh()])
        self.decoder = nn.Sequential(*decoder_layers)
        self.output = nn.Linear(decoder[-1], 4)

    def _query_features(self, query_pos: torch.Tensor) -> torch.Tensor:
        if self.query_encoding == "raw":
            return query_pos
        phase = (
            query_pos.unsqueeze(-1)
            * self.pe_frequencies.to(dtype=query_pos.dtype)
            * (2.0 * math.pi)
        )
        encoded = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        return torch.cat([query_pos, encoded.flatten(start_dim=1)], dim=-1)

    def encode_support(
        self,
        support_pos: torch.Tensor,
        support_features: torch.Tensor,
        support_batch: torch.Tensor,
        **context: Any,
    ) -> dict[str, torch.Tensor]:
        local = self.branch(support_features)
        global_feature = scatter(local, support_batch, dim=0, reduce="max")
        if self.conditioner == "global":
            return {"case_latent": global_feature}
        centroid_pos = []
        centroid_feature = []
        centroid_batch = []
        unit_ids = list(context.get("unit_ids") or [])
        global_seed = int(context.get("global_seed", 0))
        epoch = 0 if context.get("evaluation") else int(context.get("epoch", 0))
        for graph_id in range(int(global_feature.size(0))):
            mask = support_batch == graph_id
            graph_points = support_pos[mask]
            graph_features = local[mask]
            identity = unit_ids[graph_id] if graph_id < len(unit_ids) else str(graph_id)
            indices = _deterministic_fps(
                graph_points,
                self.local_centroids,
                identity=identity,
                seed=global_seed + epoch * 1_000_003,
            )
            centroid_pos.append(graph_points[indices])
            centroid_feature.append(graph_features[indices])
            centroid_batch.append(
                torch.full(
                    (len(indices),),
                    graph_id,
                    dtype=torch.long,
                    device=support_pos.device,
                )
            )
        return {
            "case_latent": global_feature,
            "centroid_pos": torch.cat(centroid_pos, dim=0),
            "centroid_feature": torch.cat(centroid_feature, dim=0),
            "centroid_batch": torch.cat(centroid_batch, dim=0),
        }

    def _local_condition(
        self,
        encoded: dict[str, torch.Tensor],
        query_pos: torch.Tensor,
        query_batch: torch.Tensor,
    ) -> torch.Tensor:
        output = query_pos.new_empty((query_pos.size(0), 256))
        bandwidths = self.local_bandwidths.to(dtype=query_pos.dtype)
        for graph_id in range(int(encoded["case_latent"].size(0))):
            query_mask = query_batch == graph_id
            centroid_mask = encoded["centroid_batch"] == graph_id
            query = query_pos[query_mask]
            centroids = encoded["centroid_pos"][centroid_mask]
            features = encoded["centroid_feature"][centroid_mask]
            distance_sq = torch.cdist(query, centroids).square()
            scales = []
            for bandwidth in bandwidths:
                weights = torch.softmax(
                    -distance_sq / (2.0 * torch.square(bandwidth)), dim=1
                )
                scales.append(weights @ features)
            output[query_mask] = torch.stack(scales, dim=0).mean(dim=0)
        return output

    def decode_query(
        self,
        encoded: dict[str, torch.Tensor],
        query_pos: torch.Tensor,
        query_batch: torch.Tensor,
    ) -> torch.Tensor:
        if self.conditioner == "global":
            condition = encoded["case_latent"][query_batch]
        else:
            condition = self._local_condition(encoded, query_pos, query_batch)
        fused = torch.cat([self._query_features(query_pos), condition], dim=-1)
        return self.output(self.decoder(fused))


def build_model(config) -> nn.Module:
    in_dim = 3 if config.input_variant == "xyz" else 6
    activation = str(config["model"]["activation"])
    if config.architecture == "pointnet":
        return SmoothPointNetField(
            in_dim, config["model"]["pointnet"], activation
        )
    if config.architecture == "pointnetpp":
        return SmoothPointNetPlusPlusField(
            in_dim, config["model"]["pointnetpp"], activation
        )
    if config.architecture == "smooth_pointnet":
        if activation != "tanh":
            raise ValueError("V3 smooth_pointnet activation must be tanh")
        return ConditionalSmoothPointNetField(
            in_dim, config["model"]["smooth_pointnet"]
        )
    if config.architecture == "conditional_field_v4":
        if activation != "tanh":
            raise ValueError("V4 conditional field activation must be tanh")
        return ConditionalFieldV4(in_dim, config["model"]["field_v4"])
    raise ValueError(f"unsupported architecture={config.architecture!r}")


build_volume_model = build_model
