"""G4-c · 测地邻域壁面贴片展开（Phase 0 审计 / Phase 1 训练共用）。"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from pipeline.config import NODE_FEATURE_NAMES
from pipeline.wall_unwrap.grid import denorm_wss_mag, load_norm_stats, r2_score

NODE_IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
X_TAN = slice(NODE_IDX["Tangent_X"], NODE_IDX["Tangent_Z"] + 1)
BRANCH_IDX = NODE_IDX.get("branch_id")
DIST_BIF_IDX = NODE_IDX.get("dist_to_bifurcation")


@dataclass(frozen=True)
class GeodesicPatchConfig:
    patch_h: int = 12
    patch_w: int = 12
    geodesic_radius_mm: float = 3.0
    knn_k: int = 20
    max_seeds_per_graph: int = 40
    seed_stride: int = 0


def _load_graph(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _estimate_normals(xyz: np.ndarray, k: int = 16) -> np.ndarray:
    n = xyz.shape[0]
    k_eff = min(k, max(2, n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(xyz)
    _, idx = nn.kneighbors(xyz)
    normals = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        nb = xyz[idx[i, 1:]]
        c = nb - nb.mean(axis=0, keepdims=True)
        cov = c.T @ c
        _, v = np.linalg.eigh(cov)
        nrm = v[:, 0]
        tn = np.linalg.norm(nrm)
        normals[i] = nrm / tn if tn > 1e-12 else np.array([0.0, 0.0, 1.0])
    return normals


def _estimate_mm_per_norm_unit(
    wall_xyz: np.ndarray,
    int_xyz: np.ndarray,
    int_dist_to_wall_mm: np.ndarray,
    *,
    sample: int = 200,
) -> float:
    n = wall_xyz.shape[0]
    step = max(1, n // sample)
    q = np.arange(0, n, step)
    nn = NearestNeighbors(n_neighbors=1).fit(int_xyz)
    d_norm, j = nn.kneighbors(wall_xyz[q])
    d_norm = d_norm.ravel()
    d_mm = int_dist_to_wall_mm[j.ravel()]
    mask = (d_norm > 1e-9) & (d_mm > 1e-9)
    if not np.any(mask):
        return 1.0
    return float(np.median(d_mm[mask] / d_norm[mask]))


def _build_knn_adj(xyz: np.ndarray, k: int) -> List[List[Tuple[int, float]]]:
    k_eff = min(k + 1, xyz.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff).fit(xyz)
    dist, idx = nn.kneighbors(xyz)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(xyz.shape[0])]
    for i in range(xyz.shape[0]):
        for j in range(1, idx.shape[1]):
            v = int(idx[i, j])
            w = float(dist[i, j])
            if w > 1e-12:
                adj[i].append((v, w))
                adj[v].append((i, w))
    return adj


def _geodesic_from(source: int, adj: Sequence[Sequence[Tuple[int, float]]], n: int) -> np.ndarray:
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[source] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def _local_frame(tangent: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = tangent.astype(np.float64)
    tn = np.linalg.norm(t)
    t = t / tn if tn > 1e-9 else np.array([0.0, 0.0, 1.0])
    n = normal.astype(np.float64)
    nn = np.linalg.norm(n)
    n = n / nn if nn > 1e-9 else np.array([1.0, 0.0, 0.0])
    b = np.cross(n, t)
    bn = np.linalg.norm(b)
    b = b / bn if bn > 1e-12 else np.array([0.0, 1.0, 0.0])
    return t, b


def patch_from_seed_training(
    seed: int,
    *,
    graph_path: Path,
    xyz: np.ndarray,
    wss_norm: np.ndarray,
    wss_phys: np.ndarray,
    norm_r: np.ndarray,
    curv: np.ndarray,
    tangents: np.ndarray,
    normal: np.ndarray,
    geod: np.ndarray,
    geodesic_radius: float,
    patch_h: int,
    patch_w: int,
    bc_inlet: float,
) -> Optional[Dict[str, Any]]:
    in_patch = geod <= geodesic_radius + 1e-9
    idx = np.where(in_patch & np.isfinite(geod))[0]
    if idx.size < 12:
        return None

    center = xyz[seed]
    t_vec, b_vec = _local_frame(tangents[seed], normal[seed])
    rel = xyz[idx] - center
    u = rel @ t_vec
    v = rel @ b_vec
    u_rng = float(np.max(u) - np.min(u))
    v_rng = float(np.max(v) - np.min(v))
    if u_rng < 1e-6 or v_rng < 1e-6:
        return None

    u_bin = np.clip(((u - u.min()) / u_rng * (patch_h - 1)).astype(int), 0, patch_h - 1)
    v_bin = np.clip(((v - v.min()) / v_rng * (patch_w - 1)).astype(int), 0, patch_w - 1)

    counts = np.zeros((patch_h, patch_w), dtype=np.int64)
    wss_sum = np.zeros((patch_h, patch_w), dtype=np.float64)
    wssn_sum = np.zeros((patch_h, patch_w), dtype=np.float64)
    nr_sum = np.zeros((patch_h, patch_w), dtype=np.float64)
    cv_sum = np.zeros((patch_h, patch_w), dtype=np.float64)
    for bi, bj, wn, wp, nr, cv in zip(
        u_bin, v_bin, wss_norm[idx], wss_phys[idx], norm_r[idx], curv[idx]
    ):
        counts[bi, bj] += 1
        wssn_sum[bi, bj] += float(wn)
        wss_sum[bi, bj] += float(wp)
        nr_sum[bi, bj] += float(nr)
        cv_sum[bi, bj] += float(cv)

    occupied = counts > 0
    n_occ = int(occupied.sum())
    if n_occ < 4:
        return None

    wssn_grid = np.full((patch_h, patch_w), np.nan, dtype=np.float32)
    wssp_grid = np.full((patch_h, patch_w), np.nan, dtype=np.float32)
    nr_grid = np.full((patch_h, patch_w), np.nan, dtype=np.float32)
    cv_grid = np.full((patch_h, patch_w), np.nan, dtype=np.float32)
    wssn_grid[occupied] = (wssn_sum[occupied] / counts[occupied]).astype(np.float32)
    wssp_grid[occupied] = (wss_sum[occupied] / counts[occupied]).astype(np.float32)
    nr_grid[occupied] = (nr_sum[occupied] / counts[occupied]).astype(np.float32)
    cv_grid[occupied] = (cv_sum[occupied] / counts[occupied]).astype(np.float32)

    gt_pts = wss_phys[idx]
    pred_pts = wssp_grid[u_bin, v_bin]
    r2_pts = r2_score(gt_pts, pred_pts)

    occ_mask = occupied.astype(np.float32)
    bc_grid = np.full((patch_h, patch_w), bc_inlet, dtype=np.float32)
    x_feat = np.stack([nr_grid, cv_grid, occ_mask, bc_grid], axis=0)
    nan_to = 0.0
    x_feat = np.nan_to_num(x_feat, nan=nan_to)

    return {
        "graph_path": str(graph_path),
        "seed": int(seed),
        "x_feat": x_feat,
        "y_wss_norm": wssn_grid[np.newaxis, ...],
        "y_wss_phys": wssp_grid[np.newaxis, ...],
        "occupied": occ_mask,
        "remap": {
            "u_bin": u_bin.astype(np.int32),
            "v_bin": v_bin.astype(np.int32),
            "wall_idx": idx.astype(np.int32),
            "wss_phys": wss_phys[idx].astype(np.float64),
            "wss_norm": wss_norm[idx].astype(np.float32),
        },
        "occupancy_ratio": float(n_occ / (patch_h * patch_w)),
        "remap_gap_gt_perfect": float(1.0 - r2_pts),
    }


def extract_graph_patch_samples(
    graph_path: Path,
    *,
    cfg: GeodesicPatchConfig,
    stats: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    stats = stats or load_norm_stats()
    data = _load_graph(graph_path)
    x = data.x.numpy()
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    n_wall = int(np.sum(wall))
    if n_wall < 40:
        return []

    xyz = x[wall, :3].astype(np.float64)
    tangents = x[wall, X_TAN].astype(np.float64)
    wss_norm = data.y_wss.numpy()[wall, 0].astype(np.float32)
    wss_phys = denorm_wss_mag(wss_norm, stats).astype(np.float64)
    norm_r = x[wall, NODE_IDX["NormRadius"]].astype(np.float32)
    curv = x[wall, NODE_IDX["Curvature"]].astype(np.float32)
    interior = ~wall
    int_xyz = x[interior, :3].astype(np.float64)
    int_dtw = x[interior, NODE_IDX["dist_to_wall"]].astype(np.float64)
    mm_per_unit = _estimate_mm_per_norm_unit(xyz, int_xyz, int_dtw)
    geodesic_radius_norm = cfg.geodesic_radius_mm / max(mm_per_unit, 1e-9)
    normals = _estimate_normals(xyz, k=16)
    adj = _build_knn_adj(xyz, cfg.knn_k)
    bc_inlet = float(data.global_cond.view(-1)[1].item()) if hasattr(data, "global_cond") else 0.0

    stride = cfg.seed_stride if cfg.seed_stride > 0 else max(1, n_wall // max(1, cfg.max_seeds_per_graph))
    seed_indices = np.arange(0, n_wall, stride)[: cfg.max_seeds_per_graph]

    samples: List[Dict[str, Any]] = []
    for si in seed_indices:
        geod = _geodesic_from(int(si), adj, n_wall)
        row = patch_from_seed_training(
            int(si),
            graph_path=graph_path,
            xyz=xyz,
            wss_norm=wss_norm,
            wss_phys=wss_phys,
            norm_r=norm_r,
            curv=curv,
            tangents=tangents,
            normal=normals,
            geod=geod,
            geodesic_radius=geodesic_radius_norm,
            patch_h=cfg.patch_h,
            patch_w=cfg.patch_w,
            bc_inlet=bc_inlet,
        )
        if row:
            samples.append(row)
    return samples


def merged_graph_point_r2(
    graph_path: Path,
    patch_preds_phys: List[np.ndarray],
    patch_samples: List[Dict[str, Any]],
) -> float:
    """同一 graph 上合并多 patch 点预测 → 壁面点 R²。"""
    data = _load_graph(graph_path)
    x = data.x.numpy()
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    n_wall = int(np.sum(wall))
    stats = load_norm_stats()
    wss_phys = denorm_wss_mag(data.y_wss.numpy()[wall, 0], stats)

    pred_sum = np.zeros(n_wall, dtype=np.float64)
    pred_cnt = np.zeros(n_wall, dtype=np.int64)

    wss_std = float(stats.get("wss", {}).get("std", 1.0))
    wss_mean = float(stats.get("wss", {}).get("mean", 0.0))

    for pred_norm, sample in zip(patch_preds_phys, patch_samples):
        h, w = sample["y_wss_phys"].shape[1], sample["y_wss_phys"].shape[2]
        grid = pred_norm[0, :h, :w] if pred_norm.ndim == 3 else pred_norm[:h, :w]
        grid_phys = grid * wss_std + wss_mean
        remap = sample["remap"]
        pred_pts = grid_phys[remap["u_bin"], remap["v_bin"]]
        for wi, pp in zip(remap["wall_idx"], pred_pts):
            pred_sum[int(wi)] += float(pp)
            pred_cnt[int(wi)] += 1

    mask = pred_cnt > 0
    if not np.any(mask):
        return float("nan")
    merged = np.zeros(n_wall, dtype=np.float64)
    merged[mask] = pred_sum[mask] / pred_cnt[mask]
    return r2_score(wss_phys[mask], merged[mask])
