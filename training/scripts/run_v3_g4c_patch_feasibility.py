#!/usr/bin/env python3
"""G4-c Phase 0 · Geodesic Patch 可行性审计（V3P · 0 重训 · CPU）。

对照路径 G §7 / [V3P_路径G_G4_2D换轨预研方案.md] G4-c：在壁面 kNN 图上做测地邻域
局部贴片展开，审计 occupancy 与 GT 反映射 gap，并与 G4-a baseline gap ~0.16 对照。

产出 ``outputs/field/f0_decision/v3p_g4c_patch_feasibility_<date>.json``。

用法::

    python -m training.scripts.run_v3_g4c_patch_feasibility
    python -m training.scripts.run_v3_g4c_patch_feasibility --max-graphs-per-case 2 --max-seeds-per-graph 20
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from ..core.denylist import filter_case_names
from ..core.splits import SplitSpec
from pipeline.config import NODE_FEATURE_NAMES
from pipeline.wall_unwrap.grid import (
    UnwrapGridConfig,
    denorm_wss_mag,
    graph_to_2d_sample,
    load_norm_stats,
    r2_score,
    remap_grid_to_wall_points,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
G4A_BASELINE_GAP = 0.16

NODE_IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
X_TAN = slice(NODE_IDX["Tangent_X"], NODE_IDX["Tangent_Z"] + 1)
BRANCH_IDX = NODE_IDX.get("branch_id")
DIST_BIF_IDX = NODE_IDX.get("dist_to_bifurcation")

GO_OCCUPANCY_MIN = 0.5
GO_REMAP_GAP_MAX = G4A_BASELINE_GAP


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
    """由 dist_to_wall(mm) / 归一化欧氏距 估计坐标尺度。"""
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


def _patch_from_seed(
    seed: int,
    *,
    xyz: np.ndarray,
    wss_phys: np.ndarray,
    tangents: np.ndarray,
    branch: Optional[np.ndarray],
    dist_bif: Optional[np.ndarray],
    geod: np.ndarray,
    geodesic_radius: float,
    patch_h: int,
    patch_w: int,
    normal: np.ndarray,
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
    for bi, bj, w in zip(u_bin, v_bin, wss_phys[idx]):
        counts[bi, bj] += 1
        wss_sum[bi, bj] += float(w)

    occupied = counts > 0
    n_cells = patch_h * patch_w
    n_occ = int(occupied.sum())
    if n_occ < 4:
        return None

    wss_grid = np.full((patch_h, patch_w), np.nan, dtype=np.float64)
    wss_grid[occupied] = wss_sum[occupied] / counts[occupied]

    gt_pts = wss_phys[idx]
    pred_pts = wss_grid[u_bin, v_bin]
    r2_pts = r2_score(gt_pts, pred_pts)
    remap_gap = 1.0 - r2_pts

    flags: List[str] = []
    occ_ratio = n_occ / n_cells
    if occ_ratio < 0.25:
        flags.append("sparse_patch")
    if branch is not None:
        br = branch[idx]
        if np.unique(np.round(br)).size > 1:
            flags.append("bifurcation_mix")
    if dist_bif is not None and float(np.min(dist_bif[idx])) < 2.0:
        flags.append("near_bifurcation")

    return {
        "seed": int(seed),
        "n_points": int(idx.size),
        "occupancy_ratio": round(occ_ratio, 4),
        "cells_occupied": n_occ,
        "cells_total": n_cells,
        "remap_gap_gt_perfect": round(remap_gap, 4),
        "r2_points_remap": round(r2_pts, 4),
        "flags": sorted(set(flags)),
    }


def _g4a_remap_gap(
    graph_path: Path,
    *,
    cfg: UnwrapGridConfig,
    stats: Mapping[str, Dict[str, float]],
) -> Optional[float]:
    sample = graph_to_2d_sample(graph_path, cfg=cfg, norm_stats=stats)
    if sample is None:
        return None
    gt_grid = sample["y_wss_phys"][0]
    pred_pts, gt_pts = remap_grid_to_wall_points(gt_grid, sample["remap"], use_phys=True, stats=stats)
    r2_pts = r2_score(gt_pts, pred_pts)
    return round(1.0 - r2_pts, 4)


def _audit_graph(
    graph_path: Path,
    *,
    stats: Mapping[str, Dict[str, float]],
    patch_h: int,
    patch_w: int,
    geodesic_radius: float,
    knn_k: int,
    max_seeds: int,
    g4a_cfg: UnwrapGridConfig,
    seed_stride: int,
) -> Optional[Dict[str, Any]]:
    data = _load_graph(graph_path)
    x = data.x.numpy()
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    n_wall = int(np.sum(wall))
    if n_wall < 40:
        return None

    xyz = x[wall, :3].astype(np.float64)
    tangents = x[wall, X_TAN].astype(np.float64)
    wss_phys = denorm_wss_mag(data.y_wss.numpy()[wall, 0], stats).astype(np.float64)
    branch = x[wall, BRANCH_IDX].astype(np.float64) if BRANCH_IDX is not None else None
    dist_bif = x[wall, DIST_BIF_IDX].astype(np.float64) if DIST_BIF_IDX is not None else None
    interior = ~wall
    int_xyz = x[interior, :3].astype(np.float64)
    int_dtw = x[interior, NODE_IDX["dist_to_wall"]].astype(np.float64)
    mm_per_unit = _estimate_mm_per_norm_unit(xyz, int_xyz, int_dtw)
    geodesic_radius_norm = geodesic_radius / max(mm_per_unit, 1e-9)
    normals = _estimate_normals(xyz, k=16)
    adj = _build_knn_adj(xyz, knn_k)

    seed_indices = np.arange(0, n_wall, max(1, seed_stride))[:max_seeds]
    patch_rows: List[Dict[str, Any]] = []
    for si in seed_indices:
        geod = _geodesic_from(int(si), adj, n_wall)
        row = _patch_from_seed(
            int(si),
            xyz=xyz,
            wss_phys=wss_phys,
            tangents=tangents,
            branch=branch,
            dist_bif=dist_bif,
            geod=geod,
            geodesic_radius=geodesic_radius_norm,
            patch_h=patch_h,
            patch_w=patch_w,
            normal=normals,
        )
        if row:
            patch_rows.append(row)

    if not patch_rows:
        return None

    occ = [p["occupancy_ratio"] for p in patch_rows]
    gaps = [p["remap_gap_gt_perfect"] for p in patch_rows]
    flags: List[str] = []
    if float(np.mean(occ)) < GO_OCCUPANCY_MIN:
        flags.append("low_mean_occupancy")
    if float(np.mean(gaps)) >= GO_REMAP_GAP_MAX:
        flags.append("high_remap_gap")
    if any("bifurcation_mix" in p.get("flags", []) for p in patch_rows):
        flags.append("bifurcation_patch")
    if sum(1 for p in patch_rows if "sparse_patch" in p.get("flags", [])) > len(patch_rows) * 0.5:
        flags.append("sparse_patches")

    g4a_gap = _g4a_remap_gap(graph_path, cfg=g4a_cfg, stats=stats)

    return {
        "graph": graph_path.name,
        "n_wall": n_wall,
        "mm_per_norm_unit": round(mm_per_unit, 4),
        "geodesic_radius_norm": round(geodesic_radius_norm, 4),
        "n_patches": len(patch_rows),
        "mean_patch_occupancy": round(float(np.mean(occ)), 4),
        "mean_remap_gap_gt_perfect": round(float(np.mean(gaps)), 4),
        "p95_remap_gap_gt_perfect": round(float(np.percentile(gaps, 95)), 4),
        "g4a_global_remap_gap_gt_perfect": g4a_gap,
        "gap_vs_g4a_delta": round(float(np.mean(gaps)) - g4a_gap, 4) if g4a_gap is not None else None,
        "flags": sorted(set(flags)),
        "patches": patch_rows,
    }


def _audit_case(
    case_rel: str,
    data_root: Path,
    *,
    stats: Mapping[str, Dict[str, float]],
    patch_h: int,
    patch_w: int,
    geodesic_radius: float,
    knn_k: int,
    max_seeds: int,
    max_graphs: int,
    g4a_cfg: UnwrapGridConfig,
    seed_stride: int,
    verbose: bool,
) -> Dict[str, Any]:
    graphs_dir = data_root / case_rel / "processed" / "graphs"
    graph_paths = sorted(graphs_dir.glob("*.pt"))[:max_graphs] if graphs_dir.is_dir() else []
    graph_rows: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, gp in enumerate(graph_paths):
        if verbose:
            print(f"  [{case_rel}] graph {i + 1}/{len(graph_paths)} {gp.name}", flush=True)
        row = _audit_graph(
            gp,
            stats=stats,
            patch_h=patch_h,
            patch_w=patch_w,
            geodesic_radius=geodesic_radius,
            knn_k=knn_k,
            max_seeds=max_seeds,
            g4a_cfg=g4a_cfg,
            seed_stride=seed_stride,
        )
        if row:
            graph_rows.append({k: v for k, v in row.items() if k != "patches"})

    occ = [g["mean_patch_occupancy"] for g in graph_rows]
    gaps = [g["mean_remap_gap_gt_perfect"] for g in graph_rows]
    g4a_gaps = [g["g4a_global_remap_gap_gt_perfect"] for g in graph_rows if g.get("g4a_global_remap_gap_gt_perfect") is not None]
    flags: List[str] = []
    if not graph_rows:
        flags.append("no_graphs")
    elif float(np.mean(occ)) < GO_OCCUPANCY_MIN:
        flags.append("low_occupancy")
    if gaps and float(np.mean(gaps)) >= GO_REMAP_GAP_MAX:
        flags.append("high_remap_gap")
    if any("bifurcation_patch" in g.get("flags", []) for g in graph_rows):
        flags.append("bifurcation")

    return {
        "case": case_rel,
        "n_graphs_audited": len(graph_rows),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "mean_patch_occupancy": round(float(np.mean(occ)), 4) if occ else None,
        "mean_remap_gap_gt_perfect": round(float(np.mean(gaps)), 4) if gaps else None,
        "mean_g4a_global_remap_gap": round(float(np.mean(g4a_gaps)), 4) if g4a_gaps else None,
        "flags": sorted(set(flags)),
        "graphs": graph_rows,
    }


def _summarize(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    occ = [c["mean_patch_occupancy"] for c in cases if c.get("mean_patch_occupancy") is not None]
    gaps = [c["mean_remap_gap_gt_perfect"] for c in cases if c.get("mean_remap_gap_gt_perfect") is not None]
    mean_occ = float(np.mean(occ)) if occ else None
    mean_gap = float(np.mean(gaps)) if gaps else None
    gate = bool(
        mean_occ is not None
        and mean_gap is not None
        and mean_occ >= GO_OCCUPANCY_MIN
        and mean_gap < GO_REMAP_GAP_MAX
    )
    return {
        "n_cases": len(cases),
        "mean_patch_occupancy": round(mean_occ, 4) if mean_occ is not None else None,
        "mean_remap_gap_gt_perfect": round(mean_gap, 4) if mean_gap is not None else None,
        "g4a_baseline_gap": G4A_BASELINE_GAP,
        "phase0_gate": gate,
        "verdict": "go" if gate else "no_go",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P G4-c Geodesic Patch Phase 0 可行性审计")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training" / "splits" / "split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new" / "AG")
    ap.add_argument("--patch-h", type=int, default=16)
    ap.add_argument("--patch-w", type=int, default=16)
    ap.add_argument("--geodesic-radius-mm", type=float, default=3.0)
    ap.add_argument("--knn-k", type=int, default=12)
    ap.add_argument("--max-seeds-per-graph", type=int, default=40)
    ap.add_argument("--seed-stride", type=int, default=0, help="0=auto from n_wall/max_seeds")
    ap.add_argument("--g4a-grid-s", type=int, default=64)
    ap.add_argument("--g4a-sectors", type=int, default=4)
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--max-cases", type=int, default=0, help="0=全 split")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    stats = load_norm_stats()
    g4a_cfg = UnwrapGridConfig(grid_s=args.g4a_grid_s, n_sectors=args.g4a_sectors)

    active = filter_case_names(
        list(split.train_cases) + list(split.val_cases) + list(split.test_cases),
        data_root,
    )
    if args.max_cases > 0:
        active = active[: args.max_cases]

    seed_stride = args.seed_stride
    if seed_stride <= 0:
        seed_stride = 0  # resolved per graph

    print(f"G4-c patch feasibility: {len(active)} cases · patch={args.patch_h}x{args.patch_w} "
          f"radius={args.geodesic_radius_mm}mm", flush=True)

    case_rows: List[Dict[str, Any]] = []
    for case_rel in sorted(active):
        if args.verbose:
            print(f"case {case_rel}", flush=True)
        stride = seed_stride
        if stride <= 0:
            # auto: approximate uniform seed coverage
            stride = max(1, 200 // max(1, args.max_seeds_per_graph))
        case_rows.append(
            _audit_case(
                case_rel,
                data_root,
                stats=stats,
                patch_h=args.patch_h,
                patch_w=args.patch_w,
                geodesic_radius=args.geodesic_radius_mm,
                knn_k=args.knn_k,
                max_seeds=args.max_seeds_per_graph,
                max_graphs=args.max_graphs_per_case,
                g4a_cfg=g4a_cfg,
                seed_stride=stride,
                verbose=args.verbose,
            )
        )

    by_split = {
        "train": _summarize([r for r in case_rows if r["case"] in split.train_cases]),
        "val": _summarize([r for r in case_rows if r["case"] in split.val_cases]),
        "test": _summarize([r for r in case_rows if r["case"] in split.test_cases]),
    }
    overall = _summarize(case_rows)

    report: Dict[str, Any] = {
        "label": "V3P-G4c-patch-feasibility",
        "date": date.today().isoformat(),
        "split": str(args.split.resolve()),
        "data_root": str(data_root),
        "context": "V3P post5463 band · split_AG_v1 · no V3D mixing",
        "patch_h": args.patch_h,
        "patch_w": args.patch_w,
        "geodesic_radius_mm": args.geodesic_radius_mm,
        "knn_k": args.knn_k,
        "max_seeds_per_graph": args.max_seeds_per_graph,
        "g4a_reference": {"grid_s": args.g4a_grid_s, "n_sectors": args.g4a_sectors, "baseline_gap": G4A_BASELINE_GAP},
        "go_thresholds": {
            "mean_patch_occupancy_min": GO_OCCUPANCY_MIN,
            "mean_remap_gap_max": GO_REMAP_GAP_MAX,
            "note": "remap_gap 用 GT 完美网格→点反映射估计结构损失；低于 G4-a 0.16 为 Go",
        },
        "summary": overall,
        "by_split": by_split,
        "interpretation": (
            f"Phase 0 {'PASS' if overall['phase0_gate'] else 'REVIEW'}: "
            f"mean occupancy {overall.get('mean_patch_occupancy', 'n/a')} "
            f"(≥{GO_OCCUPANCY_MIN}), mean remap gap {overall.get('mean_remap_gap_gt_perfect', 'n/a')} "
            f"(<{GO_REMAP_GAP_MAX} vs G4-a {G4A_BASELINE_GAP})."
        ),
        "cases": [{k: v for k, v in c.items() if k != "graphs"} for c in case_rows],
    }

    out = args.output or (
        REPO_ROOT / "outputs" / "field" / "f0_decision" / f"v3p_g4c_patch_feasibility_{date.today():%Y%m%d}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "cases"}, indent=2, ensure_ascii=False))
    print(out, flush=True)


if __name__ == "__main__":
    main()
