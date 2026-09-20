"""Small, opt-in local refinement operators for WSS point-cloud regression.

All modules are residual and zero initialized at their output projection, so enabling
one preserves the parent model at initialization.  They deliberately accept geometry
as explicit tensors (tangent/normal or per-case mm scale) to keep deployment contracts
visible and avoid hidden dataset dependencies.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn
from torch_geometric.nn import knn, radius
from torch_geometric.utils import scatter


def _mlp(channels, final_activation=True):
    layers = []
    for i, (a, b) in enumerate(zip(channels[:-1], channels[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(channels)-2 or final_activation:
            layers += [nn.LayerNorm(b), nn.GELU()]
    return nn.Sequential(*layers)


def tangent_frame(tangent: torch.Tensor, normal: torch.Tensor, eps: float = 1e-6):
    """Return axial and circumferential unit vectors in the wall tangent plane."""
    n = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(eps)
    axial = tangent - (tangent * n).sum(-1, keepdim=True) * n
    # At a rare pole where centerline direction and normal coincide, choose the
    # coordinate axis least aligned with the normal before projecting. This is
    # explicit handling of a degenerate frame, not missing-geometry imputation.
    basis = torch.nn.functional.one_hot(n.abs().argmin(-1), 3).to(n)
    fallback = basis - (basis * n).sum(-1, keepdim=True) * n
    axial = torch.where(axial.norm(dim=-1, keepdim=True) < eps, fallback, axial)
    axial = axial / torch.linalg.vector_norm(axial, dim=-1, keepdim=True).clamp_min(eps)
    circum = torch.linalg.cross(n, axial, dim=-1)
    circum = circum / torch.linalg.vector_norm(circum, dim=-1, keepdim=True).clamp_min(eps)
    return axial, circum, n


def validate_wall_geometry(geometry, pos):
    """Raw deployment geometry: normal3, tangent3, R_mm, coordinate scale_mm."""
    if geometry is None or geometry.ndim != 2 or geometry.shape != (pos.size(0), 8):
        raise ValueError("enabled wall geometry branch requires raw geometry [N,8]")
    if not torch.isfinite(geometry).all():
        raise ValueError("wall geometry contains non-finite values")
    if (geometry[:, :3].norm(dim=-1) < 1e-6).any() or (geometry[:, 3:6].norm(dim=-1) < 1e-6).any():
        raise ValueError("wall geometry requires nonzero raw normals and tangents")
    if (geometry[:, 6:] <= 0).any():
        raise ValueError("wall radius and coordinate scale must be positive millimetres")
    return geometry.to(pos)


def _first_per_group(group, distance, cap):
    """Indices of the closest ``cap`` entries in each group, without dense NxN."""
    order = distance.argsort(stable=True)
    order = order[group[order].argsort(stable=True)]
    sorted_group = group[order]
    ids = torch.arange(order.numel(), device=order.device)
    starts = torch.zeros_like(ids)
    boundary = torch.ones_like(sorted_group, dtype=torch.bool)
    boundary[1:] = sorted_group[1:] != sorted_group[:-1]
    starts[boundary] = ids[boundary]
    rank = ids - starts.cummax(0).values
    return order[rank < cap]


def geometry_neighborhood(pos, batch, geometry, radius_value, nsample, *,
                          directional=False, physical=False):
    """Sparse radius-filtered kNN with balanced axial/circumferential sectors.

    Directional grouping reserves one quarter of the cap for each signed axial
    or circumferential sector, then fills unused slots with the nearest remaining
    candidates. Candidates are the nearest 100 points (or all if fewer) per case;
    torch-cluster's CUDA kNN kernel has a hard k<=100 limit.
    Physical grouping uses a per-center radius in mm; its nearest ``nsample``
    candidates are sufficient when directional grouping is disabled.
    """
    geom = validate_wall_geometry(geometry, pos)
    scale = geom[:, 7] if physical else torch.ones_like(geom[:, 7])
    metric_pos = pos * scale[:, None]
    radius_by_point = torch.as_tensor(radius_value, device=pos.device, dtype=pos.dtype)
    radius_by_point = radius_by_point.expand(pos.size(0))
    if (radius_by_point <= 0).any():
        raise ValueError("neighborhood radii must be positive")
    counts = torch.bincount(batch)
    minimum_case_size = int(counts[counts > 0].min().item())
    requested_k = 100 if directional else int(nsample)
    if requested_k > 100:
        raise ValueError("wall neighborhood kNN candidate count exceeds CUDA limit of 100")
    candidate_k = min(requested_k, minimum_case_size)
    row, col = knn(metric_pos, metric_pos, k=candidate_k, batch_x=batch, batch_y=batch)
    delta = metric_pos[col] - metric_pos[row]
    distance = delta.norm(dim=-1)
    keep = distance <= radius_by_point[row]
    if directional:
        axial, circum, normal = tangent_frame(geom[:, 3:6], geom[:, :3])
        relative = torch.stack(((delta * axial[row]).sum(-1),
                                (delta * circum[row]).sum(-1),
                                (delta * normal[row]).sum(-1)), dim=-1)
        # Across opposing sheets, or predominantly normal jumps, are unsuitable
        # for a wall-tangent patch. Each self edge is retained.
        compatible = ((normal[row] * normal[col]).sum(-1) > 0) & (
            relative[:, 2].abs() <= 0.75 * distance + 1e-8)
        keep &= compatible | (row == col)
    else:
        relative = delta
    row, col, relative, distance = row[keep], col[keep], relative[keep], distance[keep]
    if directional:
        if int(nsample) < 4 or int(nsample) % 4:
            raise ValueError("directional neighborhood cap must be a positive multiple of four")
        axial_dominant = relative[:, 0].abs() >= relative[:, 1].abs()
        sector = torch.where(axial_dominant,
                             (relative[:, 0] < 0).long(),
                             2 + (relative[:, 1] < 0).long())
        reserved = _first_per_group(row * 4 + sector, distance, int(nsample) // 4)
        # Reserved entries sort first; nearest unreserved edges fill shortages.
        priority = distance.clone()
        priority[reserved] -= radius_by_point[row[reserved]] * 2
        selected = _first_per_group(row, priority, int(nsample))
        row, col, relative = row[selected], col[selected], relative[selected]
    return row, col, relative / radius_by_point[row, None]


class DirectionalNeighborhoodRefinement(nn.Module):
    """An anisotropic wall-neighborhood residual branch.

    Neighbors are selected by kNN, then represented in each center's axial,
    circumferential and normal frame.  A normal-distance gate suppresses edges that
    jump across nearby but distinct wall sheets.  ``tangent`` and ``normal`` are
    deployment geometry inputs, not learned quantities.
    """
    def __init__(self, channels: int, out_channels: Optional[int] = None,
                 k: int = 16, hidden: int = 32, normal_gate: float = 0.35):
        super().__init__()
        self.channels = int(channels); self.out_channels = int(out_channels or channels)
        self.k = int(k); self.normal_gate = float(normal_gate)
        self.message = _mlp([channels + 4, hidden, self.out_channels])
        self.fuse = nn.Linear(self.out_channels, self.out_channels)
        nn.init.zeros_(self.fuse.weight); nn.init.zeros_(self.fuse.bias)
        self.last_edge_keep_fraction = 1.0

    def forward(self, pos, x, batch, tangent, normal):
        if pos.ndim != 2 or pos.size(-1) != 3 or x.size(0) != pos.size(0):
            raise ValueError("pos must be [N,3] and x must have matching rows")
        axial, circum, unit_normal = tangent_frame(tangent, normal)
        row, col = knn(pos, pos, k=min(self.k, max(1, pos.size(0))), batch_x=batch, batch_y=batch)
        delta = pos[col] - pos[row]
        # Coordinates are center-local; normal coordinate is used for a soft gate.
        local = torch.stack((
            (delta * axial[row]).sum(-1),
            (delta * circum[row]).sum(-1),
            (delta * unit_normal[row]).sum(-1),
            torch.linalg.vector_norm(delta, dim=-1),
        ), dim=-1)
        keep = local[:, 2].abs() <= self.normal_gate * (local[:, 3] + 1e-6)
        # Keep each center's self edge, guaranteeing nonempty aggregation.
        keep = keep | (row == col)
        self.last_edge_keep_fraction = float(keep.float().mean().detach().cpu())
        msg = self.message(torch.cat((x[col], local), dim=-1))
        pooled = scatter(msg[keep], row[keep], dim=0, dim_size=pos.size(0), reduce="mean")
        return self.fuse(pooled)


class LocalPatchFiLMRefinement(nn.Module):
    """Query-point patch refinement conditioned by an encoded context feature.

    Unlike interpolation-only decoders, this module consumes the actual support
    patch around each query and lets the support context FiLM-modulate that patch.
    The output projection is zero initialized, making the branch an exact no-op at
    initialization while retaining a nonlinear head after aggregation.
    """
    def __init__(self, support_channels: int, query_channels: int,
                 context_channels: Optional[int] = None, out_dim: int = 1,
                 k: int = 16, hidden: int = 64):
        super().__init__()
        self.k = int(k); self.context_channels = int(context_channels or support_channels)
        self.patch = _mlp([support_channels + query_channels + 4, hidden, hidden])
        self.to_gamma = nn.Linear(self.context_channels, hidden)
        self.to_beta = nn.Linear(self.context_channels, hidden)
        self.out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward_patch(self, patch, query_x, context):
        """Read an explicit complete-wall patch prepared before query chunking.

        The patch contains standardized input features and raw relative mm
        coordinates, never labels or support-interpolated substitutes.
        """
        if not isinstance(patch, dict) or "features" not in patch or "relative_mm" not in patch:
            raise ValueError("query_patch_film requires full-wall patch features and relative_mm")
        features, relative_mm = patch["features"], patch["relative_mm"]
        if features.ndim != 3 or features.shape[:2] != (query_x.size(0), self.k):
            raise ValueError(f"patch features must be [Nquery,{self.k},D]")
        if relative_mm.shape != (*features.shape[:2], 3):
            raise ValueError("patch relative_mm must be [Nquery,K,3]")
        if context is None or context.shape != (query_x.size(0), self.context_channels):
            raise ValueError("patch FiLM requires encoded per-query context")
        if not torch.isfinite(features).all() or not torch.isfinite(relative_mm).all():
            raise ValueError("query patch contains non-finite values")
        relative = torch.cat((relative_mm, relative_mm.norm(dim=-1, keepdim=True)), dim=-1)
        q = query_x[:, None, :].expand(-1, self.k, -1)
        h = self.patch(torch.cat((features, q, relative), dim=-1))
        h = h * (1.0 + self.to_gamma(context)[:, None, :]) + self.to_beta(context)[:, None, :]
        return self.out(h.mean(dim=1))

    def forward(self, support_pos, support_x, support_batch, query_pos, query_x,
                query_batch, context: Optional[torch.Tensor] = None):
        k = min(self.k, max(1, support_pos.size(0)))
        row, col = knn(support_pos, query_pos, k=k,
                       batch_x=support_batch, batch_y=query_batch)
        delta = support_pos[col] - query_pos[row]
        rel = torch.cat((delta, torch.linalg.vector_norm(delta, dim=-1, keepdim=True)), dim=-1)
        q = query_x[row]
        h_edge = self.patch(torch.cat((support_x[col], q, rel), dim=-1))
        if context is None:
            ctx = torch.zeros(query_pos.size(0), self.context_channels,
                              device=query_pos.device, dtype=h_edge.dtype)
        else:
            if context.size(0) == query_pos.size(0):
                ctx = context
            elif context.size(0) == support_pos.size(0):
                ctx = scatter(context[col], row, dim=0, dim_size=query_pos.size(0), reduce="mean")
            else:
                raise ValueError("context must be per-query or per-support")
        gamma = self.to_gamma(ctx[row]); beta = self.to_beta(ctx[row])
        h_edge = h_edge * (1.0 + gamma) + beta
        pooled = scatter(h_edge, row, dim=0, dim_size=query_pos.size(0), reduce="mean")
        return self.out(pooled)


class MultiStatisticPool(nn.Module):
    """Max + learned weighted mean + standard deviation, projected to C channels."""
    def __init__(self, channels: int, hidden: int = 16):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.proj = nn.Linear(channels * 3, channels)
        # Preserve max-pool behaviour at initialization; mean/std contributions
        # must earn their influence during training.
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:, :channels].copy_(torch.eye(channels))
            self.proj.bias.zero_()

    def forward(self, values: torch.Tensor, row: torch.Tensor, size: int):
        if values.numel() == 0:
            return values.new_zeros((size, self.proj.out_features))
        mx = scatter(values, row, dim=0, dim_size=size, reduce="max")
        scores = self.score(values).squeeze(-1)
        weights = scatter(scores, row, dim=0, dim_size=size, reduce="max")
        weights = torch.exp(scores - weights[row])
        den = scatter(weights, row, dim=0, dim_size=size, reduce="sum").clamp_min(1e-6)
        mean = scatter(values * weights[:, None], row, dim=0, dim_size=size, reduce="sum") / den[:, None]
        var = scatter((values - mean[row]).square() * weights[:, None], row, dim=0, dim_size=size, reduce="sum") / den[:, None]
        # sqrt(0) has an infinite derivative; even an initially zero projection
        # weight then produces 0*inf = NaN during backward on singleton groups.
        std = (var.clamp_min(0) + 1e-6).sqrt()
        return self.proj(torch.cat((mx, mean, std), dim=-1))


class FixedMmDualScale(nn.Module):
    """Two physical-radius local messages; converts normalized coordinates per case."""
    def __init__(self, channels: int, radii_mm: Tuple[float, float] = (2.5, 10.0),
                 hidden: int = 32):
        super().__init__(); self.radii_mm = tuple(float(r) for r in radii_mm)
        self.mlps = nn.ModuleList(_mlp([channels + 4, hidden, channels]) for _ in self.radii_mm)
        self.fuse = nn.Linear(channels * len(self.radii_mm), channels)
        nn.init.zeros_(self.fuse.weight); nn.init.zeros_(self.fuse.bias)

    def forward(self, pos, x, batch, case_extent_mm):
        # ``case_extent_mm`` is max absolute atlas extent per case, in millimetres.
        if case_extent_mm.ndim == 0: case_extent_mm = case_extent_mm[None]
        scale = case_extent_mm[batch].to(pos).clamp_min(1e-6)
        pooled = []
        pos_mm = pos * scale[:, None]
        for radius_mm, mlp in zip(self.radii_mm, self.mlps):
            # Sparse radius query avoids an O(N^2) distance matrix at deployment.
            row, col = radius(pos_mm, pos_mm, float(radius_mm), batch, batch,
                              max_num_neighbors=64)
            rel = (pos[col] - pos[row]) * scale[row, None] / radius_mm
            msg = mlp(torch.cat((x[col], rel, rel.norm(dim=-1, keepdim=True)), dim=-1))
            pooled.append(scatter(msg, row, dim=0, dim_size=pos.size(0), reduce="max"))
        return self.fuse(torch.cat(pooled, dim=-1))
