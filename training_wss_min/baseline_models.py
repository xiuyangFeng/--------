#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小 WSS 基线模型：MLP、PointNet 和 PointNet++。

三者都遵守训练器的 ``forward(pos, x, batch) -> (N,)`` 接口：

* ``MLP``：每个点独立预测，没有点云上下文；
* ``PointNet``：逐点编码 + 每病例全局 max-pool；
* ``PointNet++``：FPS、ball-query 局部集合抽象和 FP 插值解码，**不含**
  PointNeXt 的倒置瓶颈或残差块。

这是与 ``pointnext.py`` 独立的最简模型组，避免把后续路线中的模块带入
2×3 baseline。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch_geometric.nn import fps, knn, knn_interpolate, radius
from torch_geometric.utils import scatter, to_dense_batch

from .surface import stable_seed


def _mlp(channels: List[int], *, last_act: bool = True) -> nn.Sequential:
    """Linear + BatchNorm + ReLU 的基础 PointNet MLP。"""
    layers: List[nn.Module] = []
    for i, (cin, cout) in enumerate(zip(channels[:-1], channels[1:])):
        layers.append(nn.Linear(cin, cout))
        is_last = i == len(channels) - 2
        if not is_last or last_act:
            layers.extend((nn.BatchNorm1d(cout), nn.ReLU(inplace=True)))
    return nn.Sequential(*layers)


class MLPRegressor(nn.Module):
    """无邻域、无全局池化的逐点 MLP 对照。"""

    def __init__(self, in_dim: int, hidden: Tuple[int, ...], out_dim: int = 1):
        super().__init__()
        self.network = _mlp([in_dim, *hidden, out_dim], last_act=False)

    def forward(self, pos: torch.Tensor, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        del pos, batch
        out = self.network(x)
        return out.squeeze(-1) if out.size(-1) == 1 else out


class PointNetRegressor(nn.Module):
    """经典 PointNet 分割式逐点回归：局部编码 + per-case global max。"""

    def __init__(self, in_dim: int, width: int, head_hidden: int, out_dim: int = 1,
                 dropout: float = 0.0, local_channels: Tuple[int, ...] = (),
                 decoder_channels: Tuple[int, ...] = ()):
        super().__init__()
        local_spec = tuple(local_channels) or (width, width * 2, width * 4)
        decoder_spec = tuple(decoder_channels) or (width * 4, head_hidden)
        if not local_spec or not decoder_spec:
            raise ValueError("PointNet local and decoder channel lists must be non-empty")
        local_dim = local_spec[-1]
        self.local = _mlp([in_dim, *local_spec])
        self.decoder = _mlp([local_dim * 2, *decoder_spec])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(decoder_spec[-1], out_dim)
        # QA-only counter; it is neither a parameter nor a buffer, so historical
        # checkpoint state_dict keys remain byte-for-byte compatible.
        self.support_encode_calls = 0

    def encode_support(self, pos, x, batch, **_):
        del pos
        self.support_encode_calls += 1
        local = self.local(x)
        return scatter(local, batch, dim=0, reduce="max")

    def decode_query(self, encoded, pos, x, batch):
        del pos
        local = self.local(x)
        fused = torch.cat([local, encoded[batch]], dim=-1)
        out = self.out(self.dropout(self.decoder(fused)))
        return out.squeeze(-1) if out.size(-1) == 1 else out

    def forward_support_query(self, support_pos, support_x, support_batch,
                              query_pos, query_x, query_batch, **context):
        if (support_x.data_ptr() == query_x.data_ptr()
                and support_batch.data_ptr() == query_batch.data_ptr()):
            local = self.local(support_x)
            global_feat = scatter(local, support_batch, dim=0, reduce="max")
            fused = torch.cat([local, global_feat[support_batch]], dim=-1)
            out = self.out(self.dropout(self.decoder(fused)))
            return out.squeeze(-1) if out.size(-1) == 1 else out
        encoded = self.encode_support(support_pos, support_x, support_batch, **context)
        return self.decode_query(encoded, query_pos, query_x, query_batch)

    def forward(self, pos: torch.Tensor, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.forward_support_query(pos, x, batch, pos, x, batch)


def _stable_rank_within_groups(keys: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Return each item's zero-based rank after sorting by ``(key, score)``.

    The implementation is tensor-only so adaptive grouping does not introduce a
    Python loop over points or centers in every training forward pass.
    """
    if keys.numel() == 0:
        return torch.empty_like(keys)
    by_score = torch.argsort(scores, stable=True)
    order = by_score[torch.argsort(keys[by_score], stable=True)]
    sorted_keys = keys[order]
    positions = torch.arange(len(order), device=keys.device, dtype=torch.long)
    starts = torch.zeros_like(positions)
    new_group = torch.ones_like(sorted_keys, dtype=torch.bool)
    new_group[1:] = sorted_keys[1:] != sorted_keys[:-1]
    starts[new_group] = positions[new_group]
    group_starts = torch.cummax(starts, dim=0).values
    sorted_ranks = positions - group_starts
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


# torch_cluster 的 CUDA knn 核有 k<=100 的内部断言（knn_cuda.cu:98）；
# 更大的 k 走分块 cdist+topk 的精确 KNN。阈值内保持 torch_cluster 路径，
# 与既有 knn8/knn10 矩阵逐位一致。
_TORCH_CLUSTER_KNN_MAX_K = 100


def _knn_group_chunked(pos_src, batch_src, pos_query, batch_query, k: int):
    """Exact center->source KNN for k beyond the torch_cluster CUDA cap."""
    rows, cols = [], []
    n_graphs = int(batch_query.max().item()) + 1 if batch_query.numel() else 0
    for graph_id in range(n_graphs):
        src_idx = torch.nonzero(batch_src == graph_id, as_tuple=False).flatten()
        query_idx = torch.nonzero(batch_query == graph_id, as_tuple=False).flatten()
        n_src = int(src_idx.numel())
        if n_src == 0 or query_idx.numel() == 0:
            continue
        kk = min(k, n_src)
        src_pos = pos_src[src_idx].float()
        # 每块约 2^27 个距离项（~512MB fp32 瞬时），避免 (m x n) 距离矩阵整块驻留
        chunk = max(1, (1 << 27) // n_src)
        for start in range(0, int(query_idx.numel()), chunk):
            q_chunk = query_idx[start:start + chunk]
            dist = torch.cdist(pos_query[q_chunk].float(), src_pos)
            nn = dist.topk(kk, dim=-1, largest=False).indices
            rows.append(q_chunk.repeat_interleave(kk))
            cols.append(src_idx[nn.reshape(-1)])
    device = pos_query.device
    if not rows:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty.clone()
    return torch.cat(rows), torch.cat(cols)


def _knn_group(pos_src, batch_src, pos_query, batch_query, k: int):
    """Return center->source Euclidean KNN edges."""
    if k > _TORCH_CLUSTER_KNN_MAX_K:
        with torch.no_grad(), torch.autocast(pos_src.device.type, enabled=False):
            return _knn_group_chunked(pos_src, batch_src, pos_query, batch_query, k)
    edge = knn(pos_src, pos_query, k=k, batch_x=batch_src, batch_y=batch_query)
    return edge[0], edge[1]


def _knn_cover_group(pos_src, batch_src, pos_query, batch_query, k: int):
    """KNN-k plus a nearest-center repair edge for every uncovered source."""
    row, col = _knn_group(pos_src, batch_src, pos_query, batch_query, k)
    covered = torch.zeros(pos_src.size(0), dtype=torch.bool, device=pos_src.device)
    covered[col] = True
    missing = torch.nonzero(~covered, as_tuple=False).flatten()
    if missing.numel() == 0:
        return row, col
    inverse = knn(
        pos_query, pos_src[missing], k=1,
        batch_x=batch_query, batch_y=batch_src[missing],
    )
    # inverse[0] indexes the compact missing-source array; inverse[1] indexes centers.
    return torch.cat([row, inverse[1]]), torch.cat([col, missing[inverse[0]]])


def _adaptive_cover_group(
    pos_src, batch_src, pos_query, batch_query, *, radius_value: float,
    target_size: int, candidate_centers: int, overlap_cap: float,
):
    """Coverage-first ragged grouping with a hard pairwise overlap budget.

    Every source has one primary (nearest-center) membership, hence coverage is
    exactly 100%.  A source may receive at most one secondary membership.  For
    each unordered pair of centers, secondary memberships are capped by
    ``floor(min(primary sizes) * overlap_cap)``.  Since final group sizes cannot
    be smaller than their primary sizes, this construction guarantees
    ``|Gi & Gj| / min(|Gi|, |Gj|) <= overlap_cap`` for every pair.
    """
    center_counts = torch.bincount(batch_query)
    if center_counts.numel() == 0:
        raise ValueError("adaptive_cover requires at least one center")
    k = min(int(candidate_centers), int(center_counts.min().item()))
    if k < 2:
        raise ValueError("adaptive_cover requires at least two centers per graph")

    inverse = knn(
        pos_query, pos_src, k=k,
        batch_x=batch_query, batch_y=batch_src,
    )
    source_ids, center_ids = inverse[0], inverse[1]
    distances = torch.linalg.vector_norm(pos_src[source_ids] - pos_query[center_ids], dim=-1)
    # torch-cluster normally returns neighbor-major order, but sort explicitly so
    # primary/alternative ranks stay deterministic across supported versions.
    by_distance = torch.argsort(distances, stable=True)
    order = by_distance[torch.argsort(source_ids[by_distance], stable=True)]
    source_ids = source_ids[order]
    center_ids = center_ids[order]
    distances = distances[order]
    expected = pos_src.size(0) * k
    if source_ids.numel() != expected:
        raise RuntimeError(
            f"adaptive_cover KNN returned {source_ids.numel()} edges, expected {expected}"
        )
    source_grid = source_ids.reshape(pos_src.size(0), k)
    if not torch.equal(source_grid[:, 0], torch.arange(pos_src.size(0), device=pos_src.device)):
        raise RuntimeError("adaptive_cover inverse-KNN source ordering is incomplete")
    centers_by_source = center_ids.reshape(pos_src.size(0), k)
    distances_by_source = distances.reshape(pos_src.size(0), k)
    primary = centers_by_source[:, 0]
    source_index = torch.arange(pos_src.size(0), device=pos_src.device)

    n_centers = pos_query.size(0)
    primary_sizes = torch.bincount(primary, minlength=n_centers)
    group_sizes = primary_sizes.clone()
    used_secondary = torch.zeros(pos_src.size(0), dtype=torch.bool, device=pos_src.device)
    # Pair counters are int32 and remain small; a flat dense table avoids host
    # synchronization and is ~64 MB for the standard batch8 x 500 SA1 centers.
    pair_counts = torch.zeros(n_centers * n_centers, dtype=torch.int32, device=pos_src.device)
    secondary_rows: List[torch.Tensor] = []
    secondary_cols: List[torch.Tensor] = []

    for rank in range(1, k):
        target = centers_by_source[:, rank]
        candidate_distance = distances_by_source[:, rank]
        low = torch.minimum(primary, target)
        high = torch.maximum(primary, target)
        pair_key = low * n_centers + high
        pair_budget = torch.floor(
            torch.minimum(primary_sizes[low], primary_sizes[high]).to(torch.float32)
            * float(overlap_cap)
        ).to(torch.int32)
        active = (
            (~used_secondary)
            & (group_sizes[target] < int(target_size))
            & (candidate_distance <= float(radius_value))
            & (pair_counts[pair_key] < pair_budget)
        )
        active_sources = source_index[active]
        if active_sources.numel() == 0:
            continue
        active_pairs = pair_key[active]
        active_scores = candidate_distance[active]
        pair_remaining = pair_budget[active] - pair_counts[active_pairs]
        pair_rank = _stable_rank_within_groups(active_pairs, active_scores)
        pair_ok = pair_rank < pair_remaining.to(torch.long)
        active_sources = active_sources[pair_ok]
        active_pairs = active_pairs[pair_ok]
        active_scores = active_scores[pair_ok]
        active_targets = target[active_sources]
        if active_sources.numel() == 0:
            continue
        target_remaining = int(target_size) - group_sizes[active_targets]
        target_rank = _stable_rank_within_groups(active_targets, active_scores)
        target_ok = target_rank < target_remaining.to(torch.long)
        accepted_sources = active_sources[target_ok]
        accepted_targets = active_targets[target_ok]
        accepted_pairs = active_pairs[target_ok]
        if accepted_sources.numel() == 0:
            continue
        used_secondary[accepted_sources] = True
        ones_long = torch.ones_like(accepted_targets)
        group_sizes.index_add_(0, accepted_targets, ones_long)
        pair_counts.index_add_(0, accepted_pairs, torch.ones_like(accepted_pairs, dtype=torch.int32))
        secondary_rows.append(accepted_targets)
        secondary_cols.append(accepted_sources)

    if secondary_rows:
        return (
            torch.cat([primary, *secondary_rows]),
            torch.cat([source_index, *secondary_cols]),
        )
    return primary, source_index


def _group(
    pos_src, batch_src, pos_query, batch_query, radius_value: float, nsample: int,
    *, grouping: str = "ball", adaptive_candidate_centers: int = 16,
    overlap_cap: float = 1.0 / 3.0,
):
    if grouping == "ball":
        assign = radius(pos_src, pos_query, radius_value, batch_src, batch_query,
                        max_num_neighbors=nsample)
        return assign[0], assign[1]  # query indices, source indices
    if grouping == "knn":
        return _knn_group(pos_src, batch_src, pos_query, batch_query, nsample)
    if grouping == "knn_cover":
        return _knn_cover_group(pos_src, batch_src, pos_query, batch_query, nsample)
    if grouping == "adaptive_cover":
        return _adaptive_cover_group(
            pos_src, batch_src, pos_query, batch_query,
            radius_value=radius_value, target_size=nsample,
            candidate_centers=adaptive_candidate_centers, overlap_cap=overlap_cap,
        )
    raise ValueError(f"unsupported grouping mode {grouping!r}")


def sample_and_group(pos: torch.Tensor, batch: torch.Tensor, *, ratio: float,
                     radius_value: float, nsample: int, random_start: bool,
                     grouping: str = "ball", adaptive_candidate_centers: int = 16,
                     overlap_cap: float = 1.0 / 3.0):
    """Run the exact FPS + ball-query path used by PointNet++ SA.

    The returned ``row``/``col`` tensors are an auditable assignment table:
    ``row`` indexes query centers and ``col`` indexes source points.  Keeping
    this operation shared prevents the diagnostic visualizer from drifting
    away from the model's actual grouping semantics.
    """
    idx = fps(pos, batch, ratio=ratio, random_start=random_start)
    pos_q, batch_q = pos[idx], batch[idx]
    row, col = _group(
        pos, batch, pos_q, batch_q, radius_value, nsample,
        grouping=grouping, adaptive_candidate_centers=adaptive_candidate_centers,
        overlap_cap=overlap_cap,
    )
    return idx, pos_q, batch_q, row, col


def _drop_neighbors_preserve_one(
    row: torch.Tensor,
    col: torch.Tensor,
    pos_src: torch.Tensor,
    pos_query: torch.Tensor,
    drop_rate: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop neighborhood edges while preserving nearest-center coverage."""
    if not training or drop_rate <= 0.0 or row.numel() == 0:
        return row, col
    distance2 = ((pos_src[col] - pos_query[row]) ** 2).sum(dim=-1)
    nearest = _stable_rank_within_groups(row, distance2) == 0
    random_keep = torch.rand(row.numel(), device=row.device) >= float(drop_rate)
    keep = nearest | random_keep
    return row[keep], col[keep]


def _graph_drop_path_scale(
    batch: torch.Tensor, drop_rate: float, training: bool
) -> torch.Tensor:
    """Return one stochastic-depth scale per graph, broadcast to its nodes."""
    if not training or drop_rate <= 0.0 or batch.numel() == 0:
        return torch.ones((batch.numel(), 1), device=batch.device)
    keep_prob = 1.0 - float(drop_rate)
    n_graphs = int(batch.max().item()) + 1
    graph_scale = (
        (torch.rand(n_graphs, device=batch.device) < keep_prob).to(torch.float32)
        / keep_prob
    )
    return graph_scale[batch].unsqueeze(-1)


class LocalInvResBlock(nn.Module):
    """固定支持 PointNet++ 使用的轻量 PointNeXt-R 同分辨率残差块。

    两条分支都由可学习标量 ``gamma`` 小值启动。没有 post-add 激活，因此
    gamma=0 时对任意输入严格恒等，便于做数值防伪测试和稳定的小样本训练。
    """

    def __init__(self, channels: int, radius_value: float, nsample: int,
                 expansion: int = 2, residual_scale_init: float = 1e-3,
                 drop_path_rate: float = 0.0,
                 neighbor_drop_rate: float = 0.0):
        super().__init__()
        self.radius_value = float(radius_value)
        self.nsample = int(nsample)
        self.drop_path_rate = float(drop_path_rate)
        self.neighbor_drop_rate = float(neighbor_drop_rate)
        self.local = _mlp([channels + 3, channels, channels], last_act=False)
        self.pointwise = _mlp(
            [channels, channels * int(expansion), channels], last_act=False
        )
        init = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        self.gamma_local = nn.Parameter(init.clone())
        self.gamma_pointwise = nn.Parameter(init.clone())
        self.last_branch_norm_ratio = {"local": 0.0, "pointwise": 0.0}
        self.last_drop_path_graph_scale = torch.empty(0)
        self.last_neighbor_keep_rate = 1.0

    @staticmethod
    def _norm_ratio(branch: torch.Tensor, base: torch.Tensor) -> float:
        denom = torch.linalg.vector_norm(base.detach()).clamp_min(1e-12)
        return float((torch.linalg.vector_norm(branch.detach()) / denom).cpu())

    def forward(self, pos, x, batch):
        row, col = _group(
            pos, batch, pos, batch, self.radius_value, self.nsample,
            grouping="ball",
        )
        before = int(row.numel())
        row, col = _drop_neighbors_preserve_one(
            row, col, pos, pos, self.neighbor_drop_rate, self.training
        )
        self.last_neighbor_keep_rate = (
            float(row.numel()) / before if before else 1.0
        )
        grouped = torch.cat([pos[col] - pos[row], x[col]], dim=-1)
        local = scatter(
            self.local(grouped), row, dim=0, dim_size=pos.size(0), reduce="max"
        )
        drop_scale = _graph_drop_path_scale(
            batch, self.drop_path_rate, self.training
        ).to(dtype=x.dtype)
        self.last_drop_path_graph_scale = drop_scale.detach().cpu()
        local_update = drop_scale * self.gamma_local * local
        h = x + local_update
        pointwise_update = drop_scale * self.gamma_pointwise * self.pointwise(h)
        out = h + pointwise_update
        self.last_branch_norm_ratio = {
            "local": self._norm_ratio(local_update, x),
            "pointwise": self._norm_ratio(pointwise_update, h),
        }
        return out


class LocalNeighborhoodTransformer(nn.Module):
    """Transformer over the point tokens inside each SA neighborhood.

    ``group_index`` identifies the SA center owning each edge token.  Attention
    is therefore local to one center's neighborhood: it never mixes different
    centers or different cases.  The input tokens already contain relative xyz
    and, when enabled, LocalGeoPE, so no second positional-encoding contract is
    introduced here.
    """

    def __init__(self, channels: int, heads: int = 4, ffn_ratio: int = 2,
                 dropout: float = 0.0, residual_scale_init: float = 1e-3):
        super().__init__()
        if channels % int(heads) != 0:
            raise ValueError("local transformer channels must be divisible by heads")
        self.channels = int(channels)
        self.heads = int(heads)
        self.norm_attention = nn.LayerNorm(self.channels)
        self.attention = nn.MultiheadAttention(
            self.channels, self.heads, dropout=float(dropout), batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(self.channels)
        hidden = self.channels * int(ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, self.channels),
        )
        init = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        self.gamma_attention = nn.Parameter(init.clone())
        self.gamma_ffn = nn.Parameter(init.clone())
        self.last_group_sizes = torch.empty(0, dtype=torch.long)

    def forward(self, tokens: torch.Tensor, group_index: torch.Tensor,
                n_groups: int) -> torch.Tensor:
        order = torch.argsort(group_index, stable=True)
        dense, valid = to_dense_batch(
            tokens[order], group_index[order], batch_size=int(n_groups)
        )
        if not bool(valid.any(dim=1).all()):
            raise RuntimeError("local transformer received an empty SA neighborhood")
        normalized = self.norm_attention(dense)
        context, _ = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        h = dense + self.gamma_attention * context
        h = h + self.gamma_ffn * self.ffn(self.norm_ffn(h))
        self.last_group_sizes = valid.sum(dim=1).detach().cpu()
        h = h.masked_fill(
            ~valid.unsqueeze(-1), torch.finfo(h.dtype).min
        )
        return h.max(dim=1).values


class StaticEdgeConvCorrection(nn.Module):
    """Residual EdgeConv message on an existing geometry-constrained SA graph."""

    def __init__(self, in_ch: int, out_ch: int, radius_value: float,
                 residual_scale_init: float = 1e-3):
        super().__init__()
        self.radius_value = float(radius_value)
        self.message = _mlp(
            [2 * int(in_ch) + 3, out_ch, out_ch],
            last_act=False,
        )
        self.gamma = nn.Parameter(
            torch.tensor(float(residual_scale_init), dtype=torch.float32)
        )
        self.last_edges = 0

    def forward(self, x, center_idx, row, col, relative_pos):
        center_x = x[center_idx][row]
        scale = max(abs(self.radius_value), 1e-6)
        edge_input = torch.cat(
            [center_x, x[col] - center_x, relative_pos / scale],
            dim=-1,
        )
        self.last_edges = int(row.numel())
        return self.gamma * self.message(edge_input)


class PointNetSetAbstraction(nn.Module):
    """PointNet++ 单尺度 SA，可选 LocalGeoPE、EdgeConv、Transformer 与残差块。"""

    def __init__(self, in_ch: int, out_ch: int, ratio: float, radius_value: float,
                 nsample: int, center_count: int | None = None,
                 center_sampling: str = "fps", stage: int = 0,
                 grouping: str = "ball", adaptive_candidate_centers: int = 16,
                 overlap_cap: float = 1.0 / 3.0, n_blocks: int = 0,
                 invres_expansion: int = 2, residual_scale_init: float = 1e-3,
                 invres_radius_scale: float = 1.0,
                 local_geope: bool = False, drop_path_rates=(),
                 neighbor_drop_rate: float = 0.0,
                 local_geope_attr_dim: int = 3,
                 local_transformer: bool = False,
                 local_transformer_heads: int = 4,
                 local_transformer_ffn_ratio: int = 2,
                 local_transformer_dropout: float = 0.0,
                 edgeconv: bool = False):
        super().__init__()
        self.ratio = ratio
        self.radius_value = radius_value
        self.nsample = nsample
        self.center_count = center_count
        self.center_sampling = center_sampling
        self.stage = stage
        self.grouping = grouping
        self.adaptive_candidate_centers = adaptive_candidate_centers
        self.overlap_cap = overlap_cap
        self.neighbor_drop_rate = float(neighbor_drop_rate)
        self.last_neighbor_keep_rate = 1.0
        self.local = _mlp([in_ch + 3, out_ch, out_ch])
        self.local_geope_attr_dim = int(local_geope_attr_dim)
        if self.local_geope_attr_dim not in {0, 3}:
            raise ValueError("LocalGeoPE semantic attribute width must be 0 or 3")
        self.geo_pe = (
            _mlp([4 + self.local_geope_attr_dim, out_ch, out_ch], last_act=False)
            if local_geope else None
        )
        if self.geo_pe is not None:
            # Exact parent initialization: the relative geometry branch has to
            # earn a non-zero correction without perturbing the starting model.
            nn.init.zeros_(self.geo_pe[-1].weight)
            nn.init.zeros_(self.geo_pe[-1].bias)
        self.local_transformer = (
            LocalNeighborhoodTransformer(
                out_ch, heads=local_transformer_heads,
                ffn_ratio=local_transformer_ffn_ratio,
                dropout=local_transformer_dropout,
                residual_scale_init=residual_scale_init,
            )
            if local_transformer else None
        )
        self.edgeconv = (
            StaticEdgeConvCorrection(
                in_ch, out_ch, radius_value,
                residual_scale_init=residual_scale_init,
            )
            if edgeconv else None
        )
        block_drop_rates = tuple(drop_path_rates) or (0.0,) * int(n_blocks)
        if len(block_drop_rates) != int(n_blocks):
            raise ValueError("drop_path_rates must match PointNeXt-R blocks")
        self.blocks = nn.ModuleList(
            LocalInvResBlock(
                out_ch, radius_value * float(invres_radius_scale), nsample,
                expansion=invres_expansion,
                residual_scale_init=residual_scale_init,
                drop_path_rate=block_drop_rates[block_idx],
                neighbor_drop_rate=neighbor_drop_rate,
            )
            for block_idx in range(int(n_blocks))
        )

    def _fixed_indices(self, pos, batch, *, unit_ids, epoch, global_seed, evaluation):
        selected = []
        for graph_id in range(int(batch.max().item()) + 1):
            members = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
            n = int(members.numel())
            count = min(int(self.center_count), n)
            if self.center_sampling == "random":
                unit_id = unit_ids[graph_id] if unit_ids else graph_id
                seed = stable_seed(global_seed, 0 if evaluation else epoch, unit_id,
                                   self.stage, "sa_center", "eval" if evaluation else "train")
                gen = torch.Generator(device=pos.device)
                gen.manual_seed(seed % (2**63 - 1))
                local = torch.randperm(n, generator=gen, device=pos.device)[:count]
            else:
                local = fps(pos[members], ratio=count / n, random_start=self.training and not evaluation)
                local = local[:count]
                if local.numel() != count:
                    raise RuntimeError(f"FPS produced {local.numel()} centers, expected {count}")
            selected.append(members[local])
        return torch.cat(selected)

    def forward(self, pos, x, batch, *, unit_ids=None, epoch=0, global_seed=0,
                evaluation=False, geo_attr=None):
        if self.center_count is None:
            idx, pos_q, batch_q, row, col = sample_and_group(
                pos, batch, ratio=self.ratio, radius_value=self.radius_value,
                nsample=self.nsample, random_start=self.training,
                grouping=self.grouping,
                adaptive_candidate_centers=self.adaptive_candidate_centers,
                overlap_cap=self.overlap_cap,
            )
        else:
            idx = self._fixed_indices(pos, batch, unit_ids=unit_ids, epoch=epoch,
                                      global_seed=global_seed, evaluation=evaluation)
            pos_q, batch_q = pos[idx], batch[idx]
            row, col = _group(
                pos, batch, pos_q, batch_q, self.radius_value, self.nsample,
                grouping=self.grouping,
                adaptive_candidate_centers=self.adaptive_candidate_centers,
                overlap_cap=self.overlap_cap,
            )
        before = int(row.numel())
        row, col = _drop_neighbors_preserve_one(
            row, col, pos, pos_q, self.neighbor_drop_rate, self.training
        )
        self.last_neighbor_keep_rate = (
            float(row.numel()) / before if before else 1.0
        )
        self.last_indices = idx
        self.last_center_counts = torch.bincount(batch_q).detach().cpu().tolist()
        relative_pos = pos[col] - pos_q[row]
        grouped = torch.cat([relative_pos, x[col]], dim=-1)
        h = self.local(grouped)
        if self.edgeconv is not None:
            h = h + self.edgeconv(x, idx, row, col, relative_pos)
        if self.geo_pe is not None:
            if self.local_geope_attr_dim == 0:
                if geo_attr is not None:
                    raise ValueError("XYZ-only LocalGeoPE must not receive semantic attributes")
            elif geo_attr is None or geo_attr.ndim != 2 or geo_attr.size(1) != self.local_geope_attr_dim:
                raise ValueError("LocalGeoPE requires its declared semantic attribute tensor")
            scale = max(abs(float(self.radius_value)), 1e-6)
            relative_scaled = relative_pos / scale
            edge_terms = [
                relative_scaled,
                torch.linalg.vector_norm(relative_scaled, dim=-1, keepdim=True),
            ]
            if geo_attr is not None:
                edge_terms.append(geo_attr[col] - geo_attr[idx][row])
            edge_geometry = torch.cat(edge_terms, dim=-1)
            h = h + self.geo_pe(edge_geometry)
        if self.local_transformer is None:
            x_q = scatter(h, row, dim=0, dim_size=pos_q.size(0), reduce="max")
        else:
            x_q = self.local_transformer(h, row, pos_q.size(0))
        for block in self.blocks:
            x_q = block(pos_q, x_q, batch_q)
        return pos_q, x_q, batch_q


class FeaturePropagation(nn.Module):
    """PointNet++ 的 3-NN 插值上采样和 skip 拼接。"""

    def __init__(self, coarse_ch: int, skip_ch: int, out_ch: int, k: int):
        super().__init__()
        self.k = k
        self.mlp = _mlp([coarse_ch + skip_ch, out_ch, out_ch])

    def forward(self, pos_fine, x_skip, batch_fine, pos_coarse, x_coarse, batch_coarse):
        up = knn_interpolate(x_coarse, pos_coarse, pos_fine, batch_coarse, batch_fine,
                             k=self.k)
        return self.mlp(torch.cat([up, x_skip], dim=-1))


class CoarseGlobalBlock(nn.Module):
    """Graph-local attention at the coarsest SA level with relative geometry bias."""

    def __init__(self, channels: int, heads: int = 4,
                 residual_scale_init: float = 1e-3):
        super().__init__()
        if channels % int(heads) != 0:
            raise ValueError("coarse attention channels must be divisible by heads")
        self.channels = int(channels)
        self.heads = int(heads)
        self.head_dim = self.channels // self.heads
        self.norm_attn = nn.LayerNorm(self.channels)
        self.qkv = nn.Linear(self.channels, self.channels * 3)
        self.geo_bias = nn.Sequential(
            nn.Linear(4, self.heads),
            nn.ReLU(inplace=True),
            nn.Linear(self.heads, self.heads),
        )
        self.proj = nn.Linear(self.channels, self.channels)
        self.norm_ffn = nn.LayerNorm(self.channels)
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, self.channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels * 2, self.channels),
        )
        init = torch.tensor(float(residual_scale_init), dtype=torch.float32)
        self.gamma_attention = nn.Parameter(init.clone())
        self.gamma_ffn = nn.Parameter(init.clone())
        self.last_attention_row_sums: list[torch.Tensor] = []
        self.last_attention_entropy: list[float] = []
        self.last_attention_mean_distance: list[float] = []

    def forward(self, pos, x, batch):
        normalized = self.norm_attn(x)
        context = torch.zeros_like(x)
        row_sums: list[torch.Tensor] = []
        entropies: list[float] = []
        mean_distances: list[float] = []
        for graph_id in torch.unique(batch, sorted=True):
            idx = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
            n = int(idx.numel())
            qkv = self.qkv(normalized[idx]).reshape(
                n, 3, self.heads, self.head_dim
            )
            q, k, value = qkv.unbind(dim=1)
            logits = torch.einsum("ihd,jhd->hij", q, k) / (self.head_dim ** 0.5)
            delta = pos[idx][:, None, :] - pos[idx][None, :, :]
            distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            bias = self.geo_bias(torch.cat([delta, distance], dim=-1))
            logits = logits + bias.permute(2, 0, 1)
            attention = torch.softmax(logits, dim=-1)
            graph_context = torch.einsum(
                "hij,jhd->ihd", attention, value
            ).reshape(n, self.channels)
            context[idx] = graph_context
            detached = attention.detach().float()
            row_sums.append(detached.sum(dim=-1).cpu())
            entropy = -(detached * detached.clamp_min(1e-12).log()).sum(dim=-1)
            entropies.append(float(entropy.mean().cpu()))
            mean_distances.append(float(
                (detached.mean(dim=0) * distance.squeeze(-1).detach()).sum(dim=-1)
                .mean().cpu()
            ))
        self.last_attention_row_sums = row_sums
        self.last_attention_entropy = entropies
        self.last_attention_mean_distance = mean_distances
        h = x + self.gamma_attention * self.proj(context)
        return h + self.gamma_ffn * self.ffn(self.norm_ffn(h))


class ConservativeSEPKernel(nn.Module):
    """Learnable convex 3-NN interpolation initialized as PyG ``1/d²``.

    The correction can only redistribute non-negative normalized weights; it
    cannot add an unbounded query residual.  ``beta=2`` and the zero-initialized
    correction reproduce ``knn_interpolate`` within floating-point tolerance.
    """

    def __init__(self, channels: int, hidden: int = 32, k: int = 3,
                 beta_init: float = 2.0):
        super().__init__()
        del channels  # The first conservative version conditions on geometry only.
        self.k = int(k)
        self.beta = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))
        self.correction = nn.Sequential(
            nn.Linear(4, int(hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden), 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        empty_index = torch.empty(0, dtype=torch.long)
        empty_weight = torch.empty(0)
        self.last_assignment = (empty_index, empty_weight)
        self.last_weight_entropy = 0.0
        self.last_effective_neighbors = 0.0

    def forward(self, support_x, support_pos, query_pos, support_batch, query_batch):
        with torch.no_grad():
            row, col = _knn_group(
                support_pos, support_batch, query_pos, query_batch, self.k
            )
            relative = support_pos[col] - query_pos[row]
            squared_distance = (relative * relative).sum(dim=-1)
            clamped_distance = squared_distance.clamp_min(1e-16)
            parent_weight = clamped_distance.reciprocal()
            distance = torch.sqrt(clamped_distance)
        edge_geometry = torch.cat([relative, distance.unsqueeze(-1)], dim=-1)
        correction = self.correction(edge_geometry).squeeze(-1)
        beta = self.beta.clamp_min(0.0)
        # Factor the expression around beta=2 so initialization uses the exact
        # same reciprocal operation as PyG instead of a log/exp approximation.
        log_factor = (
            -0.5 * (beta - 2.0) * torch.log(clamped_distance) + correction
        ).clamp(min=-8.0, max=8.0)
        unnormalized = parent_weight * torch.exp(log_factor)
        denominator = scatter(
            unnormalized, row, dim=0, dim_size=query_pos.size(0), reduce="sum"
        ).clamp_min(1e-16)
        weights = unnormalized / denominator[row]
        output = scatter(
            support_x[col] * weights.unsqueeze(-1),
            row, dim=0, dim_size=query_pos.size(0), reduce="sum",
        )
        detached = weights.detach().float()
        self.last_assignment = (row.detach(), detached)
        entropy = scatter(
            -(detached * detached.clamp_min(1e-12).log()),
            row, dim=0, dim_size=query_pos.size(0), reduce="sum",
        )
        self.last_weight_entropy = float(entropy.mean().cpu())
        self.last_effective_neighbors = float(torch.exp(entropy).mean().cpu())
        return output


class PointNetPlusPlusRegressor(nn.Module):
    """经典 single-scale PointNet++ segmentation backbone，用于逐点标量回归。"""

    def __init__(self, in_dim: int, width: int, sa_ratios, sa_radius, sa_nsample,
                 fp_knn: int, head_hidden: int, out_dim: int = 1, dropout: float = 0.0,
                 sa_center_counts=(), sa_center_sampling: str = "fps",
                 query_decoder: str = "interpolate", qad_hidden: int = 32,
                 sa_grouping=(), sa_adaptive_candidate_centers: int = 16,
                 sa_overlap_cap: float = 1.0 / 3.0, stem_channels=(),
                 sa_blocks=(), invres_expansion: int = 2,
                 residual_scale_init: float = 1e-3,
                 invres_radius_scale: float = 1.0,
                 local_geope: bool = False,
                 local_geope_feature_indices=(3, 4, 5),
                 drop_path_rate: float = 0.0,
                 neighbor_drop_rate: float = 0.0,
                 local_transformer_stages=(),
                 local_transformer_heads: int = 4,
                 local_transformer_ffn_ratio: int = 2,
                 local_transformer_dropout: float = 0.0,
                 edgeconv_stages=(),
                 coarse_attention: bool = False,
                 coarse_attention_heads: int = 4,
                 sep_hidden: int = 32, sep_beta_init: float = 2.0):
        super().__init__()
        if not (len(sa_ratios) == len(sa_radius) == len(sa_nsample)):
            raise ValueError("PointNet++ SA ratios/radii/nsample must have equal length")
        stem_spec = tuple(stem_channels) or (width, width)
        if stem_spec[-1] != width:
            raise ValueError("stem_channels must end at model width")
        self.stem = _mlp([in_dim, *stem_spec])
        self.n_stages = len(sa_ratios)
        self.fixed_support_query = bool(sa_center_counts)
        self.fp_knn = fp_knn
        self.query_decoder = query_decoder
        self.support_encode_calls = 0
        self.local_geope = bool(local_geope)
        self.local_geope_feature_indices = tuple(
            int(i) for i in local_geope_feature_indices
        )
        self.local_transformer_stages = tuple(
            int(stage) for stage in local_transformer_stages
        )
        self.edgeconv_stages = tuple(
            int(stage) for stage in edgeconv_stages
        )
        if self.local_geope and len(self.local_geope_feature_indices) not in {0, 3}:
            raise ValueError("LocalGeoPE requires zero (xyz-only) or three semantic feature indices")
        channels = [width * (2 ** i) for i in range(self.n_stages + 1)]
        grouping = tuple(sa_grouping) or ("ball",) * self.n_stages
        blocks = tuple(sa_blocks) or (0,) * self.n_stages
        if len(blocks) != self.n_stages:
            raise ValueError("PointNet++ sa_blocks must match SA stages")
        total_blocks = sum(int(n) for n in blocks)
        if total_blocks > 1:
            all_drop_rates = torch.linspace(
                0.0, float(drop_path_rate), total_blocks
            ).tolist()
        elif total_blocks == 1:
            all_drop_rates = [float(drop_path_rate)]
        else:
            all_drop_rates = []
        stage_drop_rates = []
        offset = 0
        for n_blocks in blocks:
            count = int(n_blocks)
            stage_drop_rates.append(tuple(all_drop_rates[offset:offset + count]))
            offset += count
        self.sa = nn.ModuleList(
            PointNetSetAbstraction(channels[i], channels[i + 1], sa_ratios[i],
                                   sa_radius[i], sa_nsample[i],
                                   int(sa_center_counts[i]) if sa_center_counts else None,
                                   sa_center_sampling, i, grouping[i],
                                   sa_adaptive_candidate_centers, sa_overlap_cap,
                                   blocks[i], invres_expansion, residual_scale_init,
                                   invres_radius_scale, local_geope,
                                   stage_drop_rates[i], neighbor_drop_rate,
                                   len(self.local_geope_feature_indices),
                                   i + 1 in self.local_transformer_stages,
                                   local_transformer_heads,
                                   local_transformer_ffn_ratio,
                                   local_transformer_dropout,
                                   i + 1 in self.edgeconv_stages)
            for i in range(self.n_stages)
        )
        self.coarse_global = (
            CoarseGlobalBlock(
                channels[-1], heads=coarse_attention_heads,
                residual_scale_init=residual_scale_init,
            )
            if coarse_attention else None
        )
        self.fp = nn.ModuleList(
            FeaturePropagation(channels[i + 1], channels[i], channels[i], fp_knn)
            for i in reversed(range(self.n_stages))
        )
        self.head = nn.Sequential(
            nn.Linear(channels[0], head_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, out_dim),
        )
        self.sep_kernel = None
        if query_decoder == "qad_lite":
            # Relative raw features retain Δxyz/Δabscissa/Δradius/Δcurvature;
            # delta_pos and distance make the geometric contract explicit even
            # if a future input feature list omits absolute xyz.
            self.qad_local = _mlp([in_dim + 4, qad_hidden, qad_hidden])
            self.qad_gate = nn.Sequential(
                nn.Linear(channels[0], qad_hidden), nn.Sigmoid(),
            )
            self.qad_out = nn.Linear(qad_hidden, out_dim)
            # At initialization QAD is exactly the historical interpolating
            # decoder; only the residual path has to earn a non-zero update.
            nn.init.zeros_(self.qad_out.weight)
            nn.init.zeros_(self.qad_out.bias)
        elif query_decoder == "sep_kernel":
            self.sep_kernel = ConservativeSEPKernel(
                channels[0], hidden=sep_hidden, k=fp_knn, beta_init=sep_beta_init
            )
        elif query_decoder != "interpolate":
            raise ValueError(f"unsupported query_decoder {query_decoder!r}")

    def encode_support(self, pos, x, batch, *, unit_ids=None, epoch=0,
                       global_seed=0, evaluation=False):
        self.support_encode_calls += 1
        support_input_x = x
        geo_attr = None
        if self.local_geope:
            if self.local_geope_feature_indices and max(self.local_geope_feature_indices) >= support_input_x.size(1):
                raise ValueError("LocalGeoPE feature index exceeds support input width")
            if self.local_geope_feature_indices:
                geo_attr = support_input_x[:, self.local_geope_feature_indices]
        x = self.stem(support_input_x)
        levels = [(pos, x, batch)]
        for stage, sa in enumerate(self.sa):
            pos, x, batch = sa(pos, x, batch, unit_ids=unit_ids, epoch=epoch,
                               global_seed=global_seed, evaluation=evaluation,
                               geo_attr=geo_attr)
            if geo_attr is not None:
                geo_attr = geo_attr[sa.last_indices]
            if self.coarse_global is not None and stage == self.n_stages - 1:
                x = self.coarse_global(pos, x, batch)
            levels.append((pos, x, batch))

        pos_c, x_c, batch_c = levels[-1]
        for fp, level_idx in zip(self.fp, reversed(range(self.n_stages))):
            pos_f, x_skip, batch_f = levels[level_idx]
            x_c = fp(pos_f, x_skip, batch_f, pos_c, x_c, batch_c)
            pos_c, batch_c = pos_f, batch_f
        encoded = (pos_c, x_c, batch_c)
        if self.query_decoder == "qad_lite":
            encoded = (*encoded, support_input_x)
        return encoded

    def decode_query(self, encoded, pos, x, batch):
        support_pos, support_x, support_batch = encoded[:3]
        if (pos.data_ptr() == support_pos.data_ptr()
                and batch.data_ptr() == support_batch.data_ptr()):
            query_x = support_x
        elif self.query_decoder == "sep_kernel":
            query_x = self.sep_kernel(
                support_x, support_pos, pos, support_batch, batch
            )
        else:
            query_x = knn_interpolate(support_x, support_pos, pos,
                                      support_batch, batch, k=self.fp_knn)
        out = self.head(query_x)
        if self.query_decoder == "qad_lite":
            support_input_x = encoded[3]
            if (pos.data_ptr() == support_pos.data_ptr()
                    and x.data_ptr() == support_input_x.data_ptr()
                    and batch.data_ptr() == support_batch.data_ptr()):
                interpolated_input = support_input_x
                interpolated_pos = support_pos
            else:
                interpolated_input = knn_interpolate(
                    support_input_x, support_pos, pos,
                    support_batch, batch, k=self.fp_knn,
                )
                interpolated_pos = knn_interpolate(
                    support_pos, support_pos, pos,
                    support_batch, batch, k=self.fp_knn,
                )
            delta_pos = pos - interpolated_pos
            relative = torch.cat([
                x - interpolated_input,
                delta_pos,
                torch.linalg.vector_norm(delta_pos, dim=-1, keepdim=True),
            ], dim=-1)
            residual = self.qad_out(self.qad_local(relative) * self.qad_gate(query_x))
            out = out + residual
        return out.squeeze(-1) if out.size(-1) == 1 else out

    def forward_support_query(self, support_pos, support_x, support_batch,
                              query_pos, query_x, query_batch, **context):
        encoded = self.encode_support(support_pos, support_x, support_batch, **context)
        return self.decode_query(encoded, query_pos, query_x, query_batch)

    def forward(self, pos, x, batch):
        encoded = self.encode_support(pos, x, batch)
        if self.fixed_support_query:
            return self.decode_query(encoded, pos, x, batch)
        x_c = encoded[1]
        out = self.head(x_c)
        return out.squeeze(-1) if out.size(-1) == 1 else out


def build_baseline_model(model_cfg, in_dim: int) -> nn.Module:
    """按 ``ModelConfig.name`` 构建 2×3 矩阵使用的模型。"""
    name = model_cfg.name
    if name == "mlp":
        return MLPRegressor(in_dim, tuple(model_cfg.mlp_hidden), model_cfg.out_dim)
    if name == "pointnet":
        return PointNetRegressor(in_dim, model_cfg.width, model_cfg.head_hidden,
                                 model_cfg.out_dim, model_cfg.dropout,
                                 tuple(model_cfg.pointnet_local_channels),
                                 tuple(model_cfg.pointnet_decoder_channels))
    if name == "pointnetpp":
        return PointNetPlusPlusRegressor(
            in_dim, model_cfg.width, tuple(model_cfg.sa_ratios),
            tuple(model_cfg.sa_radius), tuple(model_cfg.sa_nsample), model_cfg.fp_knn,
            model_cfg.head_hidden, model_cfg.out_dim, model_cfg.dropout,
            tuple(model_cfg.sa_center_counts), model_cfg.sa_center_sampling,
            model_cfg.query_decoder, model_cfg.qad_hidden,
            tuple(model_cfg.sa_grouping), model_cfg.sa_adaptive_candidate_centers,
            model_cfg.sa_overlap_cap,
            stem_channels=tuple(getattr(model_cfg, "stem_channels", ()) or ()),
            sa_blocks=tuple(model_cfg.sa_blocks),
            invres_expansion=model_cfg.invres_expansion,
            residual_scale_init=model_cfg.residual_scale_init,
            invres_radius_scale=model_cfg.invres_radius_scale,
            local_geope=model_cfg.local_geope,
            local_geope_feature_indices=tuple(model_cfg.local_geope_feature_indices),
            drop_path_rate=model_cfg.drop_path_rate,
            neighbor_drop_rate=model_cfg.neighbor_drop_rate,
            local_transformer_stages=tuple(model_cfg.local_transformer_stages),
            local_transformer_heads=model_cfg.local_transformer_heads,
            local_transformer_ffn_ratio=model_cfg.local_transformer_ffn_ratio,
            local_transformer_dropout=model_cfg.local_transformer_dropout,
            edgeconv_stages=tuple(model_cfg.edgeconv_stages),
            coarse_attention=model_cfg.coarse_attention,
            coarse_attention_heads=model_cfg.coarse_attention_heads,
            sep_hidden=model_cfg.sep_hidden,
            sep_beta_init=model_cfg.sep_beta_init,
        )
    raise ValueError(f"unknown minimal baseline model {name!r}")
