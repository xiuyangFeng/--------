"""Small, opt-in local refinement operators for WSS point-cloud regression.

All modules are residual and zero initialized at their output projection, so enabling
one preserves the parent model at initialization.  They deliberately accept geometry
as explicit tensors (tangent/normal or per-case mm scale) to keep deployment contracts
visible and avoid hidden dataset dependencies.
"""
from __future__ import annotations
import copy
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
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
                 k: int = 16, hidden: int = 64, frame: str = "atlas", pool: str = "mean", radius_k: int = 0,
                 sectors: int = 0, rings: int = 1, sector_heads: int = 4):
        super().__init__()
        if frame not in {"atlas", "tangent"} or pool not in {"mean", "attention"}:
            raise ValueError("query patch frame must be atlas|tangent and pool mean|attention")
        self.k = int(k); self.context_channels = int(context_channels or support_channels)
        self.radius_k = int(radius_k)
        self.frame, self.pool = frame, pool
        self.patch = _mlp([support_channels + query_channels + 4, hidden, hidden])
        self.to_gamma = nn.Linear(self.context_channels, hidden)
        self.to_beta = nn.Linear(self.context_channels, hidden)
        self.out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.attn = None
        if pool == "attention":
            # Zero-initialised logits give uniform softmax weights, i.e. exactly the mean pool.
            self.attn = nn.Linear(hidden, 1)
            nn.init.zeros_(self.attn.weight); nn.init.zeros_(self.attn.bias)
        self.last_attention_entropy = None
        # 2026-09-16 dual scale: a second, masked branch over a fixed-mm ball; zero-initialised output = exact no-op at init
        self.patch_r = self.to_gamma_r = self.to_beta_r = self.out_r = self.attn_r = None
        if self.radius_k > 0:
            self.patch_r = _mlp([support_channels + query_channels + 4, hidden, hidden])
            self.to_gamma_r = nn.Linear(self.context_channels, hidden)
            self.to_beta_r = nn.Linear(self.context_channels, hidden)
            self.out_r = nn.Linear(hidden, out_dim)
            nn.init.zeros_(self.out_r.weight); nn.init.zeros_(self.out_r.bias)
            if pool == "attention":
                self.attn_r = nn.Linear(hidden, 1)
                nn.init.zeros_(self.attn_r.weight); nn.init.zeros_(self.attn_r.bias)
        # 2026-09-18 sector tokens (B2): keep the angular arrangement that mean pooling erases.
        # Neighbours of the fixed-mm ball are partitioned by their position in the query's own
        # (axial, circumferential) surface plane into `sectors` angular wedges x `rings` distance
        # rings; each cell becomes one token, tokens interact once, then aggregate.  The output
        # projection is zero initialised, so sectors=0 and sectors>0 both start as exact no-ops.
        self.sectors, self.rings, self.sector_heads = int(sectors), int(rings), int(sector_heads)
        self.sector_pos = self.sector_attn = self.sector_norm = self.sector_out = None
        if self.sectors > 0:
            if self.radius_k <= 0:
                raise ValueError("sector tokens operate on the fixed-mm ball branch; radius_k must be positive")
            if self.rings < 1 or self.sector_heads < 1 or hidden % self.sector_heads:
                raise ValueError("sector tokens need rings >= 1 and hidden divisible by sector_heads")
            n_tokens = self.sectors * self.rings
            # A submodule, not a bare Parameter: paired initialisation derives a module's RNG
            # substream from the parameter's parent name, and a bare Parameter here would make
            # that parent the whole query_patch module and re-draw its existing tensors.
            self.sector_pos = nn.Embedding(n_tokens, hidden)
            nn.init.normal_(self.sector_pos.weight, std=0.02)
            self.sector_attn = nn.MultiheadAttention(hidden, self.sector_heads, batch_first=True)
            self.sector_norm = nn.LayerNorm(hidden)
            self.sector_out = nn.Linear(hidden, out_dim)
            nn.init.zeros_(self.sector_out.weight); nn.init.zeros_(self.sector_out.bias)

    def _sector_index(self, relative_mm, query_geometry, radius_mm_max):
        """Cell id per neighbour: angular wedge in the query's (axial, circumferential) plane x distance ring."""
        if query_geometry is None or query_geometry.ndim != 2 or query_geometry.size(0) != relative_mm.size(0) \
                or query_geometry.size(1) < 6:
            raise ValueError("sector tokens require raw query geometry [Nquery,8]")
        axial, circum, _ = tangent_frame(query_geometry[:, 3:6].to(relative_mm), query_geometry[:, :3].to(relative_mm))
        a = (relative_mm * axial[:, None, :]).sum(-1)
        c = (relative_mm * circum[:, None, :]).sum(-1)
        ang = torch.atan2(c, a)                                  # [-pi, pi], 0 = downstream, +pi/2 = one side
        wedge = torch.floor((ang + torch.pi) / (2 * torch.pi) * self.sectors).long().clamp(0, self.sectors - 1)
        if self.rings == 1:
            return wedge
        dist = torch.linalg.vector_norm(relative_mm, dim=-1)
        edge = radius_mm_max.clamp_min(1e-6)[:, None] if radius_mm_max.ndim else radius_mm_max
        ring = torch.floor(dist / edge * self.rings).long().clamp(0, self.rings - 1)
        return wedge * self.rings + ring

    def _pool_sectors(self, h, mask, relative_mm, query_geometry):
        """Masked mean inside each cell -> token interaction -> masked mean over tokens."""
        n, k, hid = h.shape
        n_tokens = self.sectors * self.rings
        far = torch.linalg.vector_norm(relative_mm, dim=-1).masked_fill(mask <= 0, 0.0).amax(dim=1)
        cell = self._sector_index(relative_mm, query_geometry, far)
        w = mask.to(h.dtype)
        flat = (torch.arange(n, device=h.device)[:, None] * n_tokens + cell).reshape(-1)
        num = torch.zeros(n * n_tokens, hid, dtype=h.dtype, device=h.device)
        num.index_add_(0, flat, (h * w[:, :, None]).reshape(-1, hid))
        den = torch.zeros(n * n_tokens, dtype=h.dtype, device=h.device)
        den.index_add_(0, flat, w.reshape(-1))
        tok_mask = (den > 0).reshape(n, n_tokens)
        tokens = (num / den.clamp_min(1e-6)[:, None]).reshape(n, n_tokens, hid)
        tokens = tokens + self.sector_pos.weight[None].to(tokens.dtype)
        # Empty cells are masked out of attention; a query whose ball is entirely empty keeps zeros.
        pad = ~tok_mask
        safe = pad.all(dim=1, keepdim=True)
        attended, _ = self.sector_attn(tokens, tokens, tokens,
                                       key_padding_mask=torch.where(safe, torch.zeros_like(pad), pad),
                                       need_weights=False)
        tokens = self.sector_norm(tokens + attended)
        m = tok_mask.to(tokens.dtype)[:, :, None]
        return (tokens * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def _pool_masked(self, h, mask):
        m = mask.to(h.dtype)
        if self.attn_r is None:
            return (h * m[:, :, None]).sum(dim=1) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
        logits = self.attn_r(h).squeeze(-1).float().masked_fill(mask <= 0, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        return (h * weights.to(h.dtype)[:, :, None]).sum(dim=1)

    def _relative(self, relative_mm, query_geometry):
        """Relative coordinates + distance; optionally in the query's (axial, circumferential, normal) frame."""
        if self.frame == "tangent":
            if query_geometry is None or query_geometry.ndim != 2 or query_geometry.size(0) != relative_mm.size(0) \
                    or query_geometry.size(1) < 6:
                raise ValueError("tangent-frame patches require raw query geometry [Nquery,8]")
            axial, circum, normal = tangent_frame(query_geometry[:, 3:6].to(relative_mm), query_geometry[:, :3].to(relative_mm))
            local = torch.stack((
                (relative_mm * axial[:, None, :]).sum(-1),
                (relative_mm * circum[:, None, :]).sum(-1),
                (relative_mm * normal[:, None, :]).sum(-1),
            ), dim=-1)
        else:
            local = relative_mm
        return torch.cat((local, relative_mm.norm(dim=-1, keepdim=True)), dim=-1)

    def _pool(self, h):
        if self.attn is None:
            return h.mean(dim=1)
        weights = torch.softmax(self.attn(h).squeeze(-1).float(), dim=1)  # [Nquery, K]
        if not self.training:
            with torch.no_grad():
                self.last_attention_entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1).mean().detach()
        return (h * weights.to(h.dtype)[:, :, None]).sum(dim=1)

    def forward_patch(self, patch, query_x, context, query_geometry=None):
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
        relative = self._relative(relative_mm, query_geometry)
        q = query_x[:, None, :].expand(-1, self.k, -1)
        h = self.patch(torch.cat((features, q, relative), dim=-1))
        h = h * (1.0 + self.to_gamma(context)[:, None, :]) + self.to_beta(context)[:, None, :]
        out = self.out(self._pool(h))
        if self.radius_k > 0:
            for key in ("radius_features", "radius_relative_mm", "radius_mask"):
                if key not in patch:
                    raise ValueError(f"dual-scale query patch requires {key}")
            fr, rr, mr = patch["radius_features"], patch["radius_relative_mm"], patch["radius_mask"]
            if fr.shape != (query_x.size(0), self.radius_k, features.size(-1)) or rr.shape != (query_x.size(0), self.radius_k, 3) \
                    or mr.shape != (query_x.size(0), self.radius_k):
                raise ValueError("dual-scale patch tensors must be [Nquery,radius_k,D] / [Nquery,radius_k,3] / [Nquery,radius_k]")
            if not torch.isfinite(fr).all() or not torch.isfinite(rr).all():
                raise ValueError("radius patch contains non-finite values")
            rel_r = self._relative(rr, query_geometry)
            q_r = query_x[:, None, :].expand(-1, self.radius_k, -1)
            h_r = self.patch_r(torch.cat((fr, q_r, rel_r), dim=-1))
            h_r = h_r * (1.0 + self.to_gamma_r(context)[:, None, :]) + self.to_beta_r(context)[:, None, :]
            out = out + self.out_r(self._pool_masked(h_r, mr))
            if self.sectors > 0:
                out = out + self.sector_out(self._pool_sectors(h_r, mr, rr, query_geometry))
        return out

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


class MoEHead(nn.Module):
    """Mixture of prediction heads gated by the query's own (masked, standardised) input features.

    Every expert starts as an exact copy of the parent head and the gate's last layer is
    zero initialised, so the initial output equals the single-head model bit for bit.
    """
    def __init__(self, base_head: nn.Module, in_dim: int, experts: int, hidden: int = 16):
        super().__init__()
        if int(experts) < 1 or int(hidden) < 1:
            raise ValueError("MoE head needs at least one expert and a positive gate width")
        self.experts = nn.ModuleList(copy.deepcopy(base_head) for _ in range(int(experts)))
        self.gate = nn.Sequential(nn.Linear(in_dim, int(hidden)), nn.GELU(), nn.Linear(int(hidden), int(experts)))
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)
        self.last_gate_entropy = None

    def forward(self, x, input_x):
        if input_x is None or input_x.size(0) != x.size(0):
            raise ValueError("MoE head requires the query input features for gating")
        weights = torch.softmax(self.gate(input_x).float(), dim=-1)  # [N, E]
        if not self.training:
            with torch.no_grad():
                self.last_gate_entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1).mean().detach()
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [N, E, out]
        return (outputs * weights.to(outputs.dtype)[:, :, None]).sum(dim=1)


class SectionContext(nn.Module):
    """Centreline section tokens (segment x s_local bin) with a tree-wide Transformer.

    ``section`` rows are raw atlas metadata [N,2] = (segment_id in 0..6, s_local_mm); no label
    is read.  Support features are mean-pooled per occupied (case, segment, bin) cell, every
    cell gets a positional description (segment one-hot, bin position, occupancy, log count),
    and all cells of a case exchange information through ``layers`` pre-LN Transformer layers
    (unoccupied cells are masked out).  ``lookup`` returns, for every query, the residual of its
    own cell (nearest occupied cell of the same segment as fallback).  The output projection is
    zero initialised: the module is an exact no-op at initialisation.
    """
    SEGMENTS = 7

    def __init__(self, in_dim: int, feat_channels: int, context_channels: int, bin_mm: float = 4.0,
                 max_bins: int = 64, layers: int = 2, heads: int = 4, hidden: int = 128):
        super().__init__()
        if bin_mm <= 0 or max_bins < 1 or layers < 1 or heads < 1 or hidden < 1 or hidden % heads:
            raise ValueError("invalid section context settings")
        self.bin_mm, self.max_bins = float(bin_mm), int(max_bins)
        self.n_tokens = self.SEGMENTS * self.max_bins
        token_in = int(feat_channels) + int(in_dim) + 1 + self.SEGMENTS + 1 + 1
        self.embed = _mlp([token_in, int(hidden), int(hidden)])
        layer = nn.TransformerEncoderLayer(int(hidden), int(heads), dim_feedforward=2 * int(hidden),
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(layers), enable_nested_tensor=False)
        self.out = nn.Linear(int(hidden), int(context_channels))
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.last_occupied_fraction = None

    def cells(self, section):
        if section is None or section.ndim != 2 or section.size(1) != 2:
            raise ValueError("section context requires [N,2] (segment_id, s_local_mm) metadata")
        segment = section[:, 0].round().long()
        if (segment < 0).any() or (segment >= self.SEGMENTS).any():
            raise ValueError("segment ids must lie in 0..6")
        bins = torch.clamp((section[:, 1].float() / self.bin_mm).floor().long(), 0, self.max_bins - 1)
        return segment * self.max_bins + bins, segment, bins

    def forward(self, x, input_x, section, batch, n_cases):
        cell, _, _ = self.cells(section)
        if batch.numel() != x.size(0) or int(batch.max()) >= n_cases:
            raise ValueError("section batch index mismatch")
        index = batch * self.n_tokens + cell
        size = int(n_cases) * self.n_tokens
        count = scatter(torch.ones(index.numel(), device=x.device, dtype=torch.float32), index, dim=0,
                        dim_size=size, reduce="sum")
        mean_x = scatter(x.float(), index, dim=0, dim_size=size, reduce="mean")
        mean_in = scatter(input_x.float(), index, dim=0, dim_size=size, reduce="mean")
        occupied = count > 0
        token = torch.arange(self.n_tokens, device=x.device)
        seg_onehot = F.one_hot(token // self.max_bins, self.SEGMENTS).float().repeat(int(n_cases), 1)
        position = ((token % self.max_bins).float() / self.max_bins).repeat(int(n_cases))[:, None]
        feats = torch.cat((mean_x, mean_in, torch.log1p(count)[:, None], seg_onehot, position,
                           occupied.float()[:, None]), dim=-1)
        h = self.embed(feats.to(x.dtype)).view(int(n_cases), self.n_tokens, -1)
        padding = ~occupied.view(int(n_cases), self.n_tokens)
        h = self.encoder(h, src_key_padding_mask=padding)
        residual = self.out(h) * occupied.view(int(n_cases), self.n_tokens, 1).to(h.dtype)
        if not self.training:
            self.last_occupied_fraction = occupied.float().mean().detach()
        return residual, occupied.view(int(n_cases), self.n_tokens)

    def lookup(self, tokens, occupied, section, batch):
        _, segment, bins = self.cells(section)
        occupied_q = occupied.view(-1, self.SEGMENTS, self.max_bins)[batch, segment]  # [Nq, max_bins]
        grid = torch.arange(self.max_bins, device=section.device)
        distance = (grid[None, :] - bins[:, None]).abs().float()
        distance = torch.where(occupied_q, distance, torch.full_like(distance, float("inf")))
        has = occupied_q.any(dim=1)
        chosen = torch.where(has, distance.argmin(dim=1), bins)
        out = tokens[batch, segment * self.max_bins + chosen]
        return out * has.to(out.dtype)[:, None]
