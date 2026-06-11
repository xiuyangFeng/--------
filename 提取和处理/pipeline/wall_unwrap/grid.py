"""(Abscissa × θ) 壁面场展开 · 与 run_v3_f0_decision._collect_profiles_2d 对齐。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from pipeline.config import NODE_FEATURE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NORM_PARAMS = REPO_ROOT / "data_new" / "normalization_params_global.json"

NODE_IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
X_TAN = slice(NODE_IDX["Tangent_X"], NODE_IDX["Tangent_Z"] + 1)


@dataclass(frozen=True)
class UnwrapGridConfig:
    grid_s: int = 64
    n_sectors: int = 4


def load_norm_stats(path: Path | str = DEFAULT_NORM_PARAMS) -> Dict[str, Dict[str, float]]:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("statistics", {})


def _denorm_zscore(arr: np.ndarray, stat: Optional[Mapping[str, float]]) -> np.ndarray:
    if not stat:
        return arr.astype(np.float64)
    return arr.astype(np.float64) * float(stat.get("std", 1.0)) + float(stat.get("mean", 0.0))


def denorm_wss_mag(wss_norm: np.ndarray, stats: Mapping[str, Dict[str, float]]) -> np.ndarray:
    return _denorm_zscore(wss_norm, stats.get("wss"))


def compute_theta(xyz: np.ndarray, tan: np.ndarray) -> np.ndarray:
    t_mean = tan.mean(axis=0)
    tn = np.linalg.norm(t_mean)
    t_mean = t_mean / tn if tn > 1e-9 else np.array([0.0, 0.0, 1.0])
    ref = np.array([0.0, 0.0, 1.0]) if abs(t_mean[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = ref - (ref @ t_mean) * t_mean
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(t_mean, e1)
    c = xyz - xyz.mean(axis=0, keepdims=True)
    return np.arctan2(c @ e2, c @ e1)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64).ravel()
    y_pred = y_pred.astype(np.float64).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def graph_to_2d_sample(
    graph_path: Path,
    *,
    cfg: UnwrapGridConfig | None = None,
    norm_stats: Mapping[str, Dict[str, float]] | None = None,
    norm_params_path: Path | str = DEFAULT_NORM_PARAMS,
) -> Optional[Dict[str, Any]]:
    """graph.pt → 2D 特征/标签张量 + 反映射元数据。"""
    cfg = cfg or UnwrapGridConfig()
    stats = norm_stats if norm_stats is not None else load_norm_stats(norm_params_path)

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    x = data.x.numpy()
    y_wss = data.y_wss.numpy()
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    if int(np.sum(wall)) < 20:
        return None

    xyz = x[wall, :3].astype(np.float64)
    absc = x[wall, NODE_IDX["Abscissa"]].astype(np.float64)
    tan = x[wall, X_TAN].astype(np.float64)
    norm_r = x[wall, NODE_IDX["NormRadius"]].astype(np.float32)
    curv = x[wall, NODE_IDX["Curvature"]].astype(np.float32)
    wss_norm = y_wss[wall, 0].astype(np.float32)
    wss_phys = denorm_wss_mag(wss_norm, stats).astype(np.float32)

    a_lo, a_hi = float(absc.min()), float(absc.max())
    if a_hi - a_lo < 1e-9:
        return None

    theta = compute_theta(xyz, tan)
    gs, ns = cfg.grid_s, cfg.n_sectors
    s_bin = np.clip(((absc - a_lo) / (a_hi - a_lo) * gs).astype(int), 0, gs - 1)
    th_bin = np.clip(((theta + math.pi) / (2 * math.pi) * ns).astype(int), 0, ns - 1)

    feat_nr = np.full((gs, ns), np.nan, dtype=np.float32)
    feat_cv = np.full((gs, ns), np.nan, dtype=np.float32)
    tgt_norm = np.full((gs, ns), np.nan, dtype=np.float32)
    tgt_phys = np.full((gs, ns), np.nan, dtype=np.float32)
    counts = np.zeros((gs, ns), dtype=np.int32)

    for si in range(gs):
        row = s_bin == si
        if not np.any(row):
            continue
        row_nr = float(np.mean(norm_r[row]))
        row_cv = float(np.mean(curv[row]))
        row_wss_n = float(np.mean(wss_norm[row]))
        row_wss_p = float(np.mean(wss_phys[row]))
        for ti in range(ns):
            cell = row & (th_bin == ti)
            if np.any(cell):
                feat_nr[si, ti] = float(np.mean(norm_r[cell]))
                feat_cv[si, ti] = float(np.mean(curv[cell]))
                tgt_norm[si, ti] = float(np.mean(wss_norm[cell]))
                tgt_phys[si, ti] = float(np.mean(wss_phys[cell]))
                counts[si, ti] = int(np.sum(cell))
            else:
                feat_nr[si, ti] = row_nr
                feat_cv[si, ti] = row_cv
                tgt_norm[si, ti] = row_wss_n
                tgt_phys[si, ti] = row_wss_p

    occupied = counts > 0
    if not np.any(occupied):
        return None

    bc_inlet = float(data.global_cond.view(-1)[1].item()) if hasattr(data, "global_cond") else 0.0

    x_feat = np.stack(
        [
            feat_nr,
            feat_cv,
            occupied.astype(np.float32),
            np.full((gs, ns), bc_inlet, dtype=np.float32),
        ],
        axis=0,
    )

    parts = graph_path.parts
    case = ""
    if "processed" in parts:
        i = parts.index("processed")
        if i >= 2:
            case = f"{parts[i - 2]}/{parts[i - 1]}"

    return {
        "graph_path": str(graph_path.resolve()),
        "case": case,
        "x_feat": x_feat,
        "y_wss_norm": tgt_norm[np.newaxis, ...],
        "y_wss_phys": tgt_phys[np.newaxis, ...],
        "occupied": occupied,
        "counts": counts,
        "bc_inlet": bc_inlet,
        "abscissa_range": (a_lo, a_hi),
        "remap": {
            "s_bin": s_bin.astype(np.int32),
            "th_bin": th_bin.astype(np.int32),
            "wss_norm": wss_norm,
            "wss_phys": wss_phys,
        },
    }


def remap_grid_to_wall_points(
    pred_grid: np.ndarray,
    remap: Mapping[str, np.ndarray],
    *,
    use_phys: bool = False,
    stats: Mapping[str, Dict[str, float]] | None = None,
) -> np.ndarray:
    """2D 网格预测 → 壁面点（最近 cell 取值，与训练网格对齐）。"""
    s_bin = remap["s_bin"]
    th_bin = remap["th_bin"]
    if pred_grid.ndim == 3:
        pred_grid = pred_grid[0]
    pred_pts = pred_grid[s_bin, th_bin].astype(np.float64)
    if use_phys and stats is not None:
        gt = remap["wss_phys"].astype(np.float64)
    else:
        gt = remap["wss_norm"].astype(np.float64)
    return pred_pts, gt


def collect_graph_paths(cases: List[str], data_root: Path, graphs_subdir: str = "processed/graphs") -> List[Path]:
    out: List[Path] = []
    for case_rel in cases:
        gdir = data_root / case_rel / graphs_subdir
        if gdir.is_dir():
            out.extend(sorted(gdir.glob("*.pt")))
    return out
