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
BRANCH_IDX = NODE_IDX.get("branch_id")
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


def _case_from_graph_path(graph_path: Path) -> str:
    parts = graph_path.parts
    if "processed" in parts:
        i = parts.index("processed")
        if i >= 2:
            return f"{parts[i - 2]}/{parts[i - 1]}"
    return ""


def _wall_subset_to_grid_sample(
    *,
    graph_path: Path,
    xyz: np.ndarray,
    absc: np.ndarray,
    tan: np.ndarray,
    norm_r: np.ndarray,
    curv: np.ndarray,
    wss_norm: np.ndarray,
    wss_phys: np.ndarray,
    bc_inlet: float,
    cfg: UnwrapGridConfig,
    segment: str = "global",
    segment_id: int = -1,
) -> Optional[Dict[str, Any]]:
    if absc.size < 20:
        return None
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

    row_has_pts = np.array([np.any(s_bin == si) for si in range(gs)])
    if not np.any(row_has_pts):
        return None
    if not np.all(row_has_pts):
        ref_rows = np.where(row_has_pts)[0]
        for si in range(gs):
            if row_has_pts[si]:
                continue
            nearest = ref_rows[int(np.argmin(np.abs(ref_rows - si)))]
            feat_nr[si] = feat_nr[nearest]
            feat_cv[si] = feat_cv[nearest]
            tgt_norm[si] = tgt_norm[nearest]
            tgt_phys[si] = tgt_phys[nearest]

    occupied = counts > 0
    if not np.any(occupied):
        return None

    x_feat = np.stack(
        [
            feat_nr,
            feat_cv,
            occupied.astype(np.float32),
            np.full((gs, ns), bc_inlet, dtype=np.float32),
        ],
        axis=0,
    )

    return {
        "graph_path": str(graph_path.resolve()),
        "case": _case_from_graph_path(graph_path),
        "segment": segment,
        "segment_id": segment_id,
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


def graph_to_2d_sample(
    graph_path: Path,
    *,
    cfg: UnwrapGridConfig | None = None,
    norm_stats: Mapping[str, Dict[str, float]] | None = None,
    norm_params_path: Path | str = DEFAULT_NORM_PARAMS,
) -> Optional[Dict[str, Any]]:
    """graph.pt → 单张全局 2D 网格（G4-a）。"""
    samples = graph_to_2d_samples(
        graph_path,
        cfg=cfg,
        norm_stats=norm_stats,
        norm_params_path=norm_params_path,
        unwrap_mode="global",
    )
    return samples[0] if samples else None


def graph_to_2d_branch_samples(
    graph_path: Path,
    *,
    cfg: UnwrapGridConfig | None = None,
    norm_stats: Mapping[str, Dict[str, float]] | None = None,
    norm_params_path: Path | str = DEFAULT_NORM_PARAMS,
    min_wall_points: int = 20,
) -> List[Dict[str, Any]]:
    """graph.pt → 按 branch_id 切段的多张 2D 网格（G4-b）。"""
    if BRANCH_IDX is None:
        return []
    cfg = cfg or UnwrapGridConfig()
    stats = norm_stats if norm_stats is not None else load_norm_stats(norm_params_path)

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    x = data.x.numpy()
    y_wss = data.y_wss.numpy()
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    if int(np.sum(wall)) < min_wall_points * 2:
        return []

    xyz_all = x[wall, :3].astype(np.float64)
    absc_all = x[wall, NODE_IDX["Abscissa"]].astype(np.float64)
    tan_all = x[wall, X_TAN].astype(np.float64)
    norm_r_all = x[wall, NODE_IDX["NormRadius"]].astype(np.float32)
    curv_all = x[wall, NODE_IDX["Curvature"]].astype(np.float32)
    branch_all = x[wall, BRANCH_IDX].astype(np.float64)
    wss_norm_all = y_wss[wall, 0].astype(np.float32)
    wss_phys_all = denorm_wss_mag(wss_norm_all, stats).astype(np.float32)
    bc_inlet = float(data.global_cond.view(-1)[1].item()) if hasattr(data, "global_cond") else 0.0

    out: List[Dict[str, Any]] = []
    for seg_id, label in ((0, "trunk"), (1, "branch")):
        mask = branch_all <= 0.5 if seg_id == 0 else branch_all > 0.5
        if int(mask.sum()) < min_wall_points:
            continue
        sample = _wall_subset_to_grid_sample(
            graph_path=graph_path,
            xyz=xyz_all[mask],
            absc=absc_all[mask],
            tan=tan_all[mask],
            norm_r=norm_r_all[mask],
            curv=curv_all[mask],
            wss_norm=wss_norm_all[mask],
            wss_phys=wss_phys_all[mask],
            bc_inlet=bc_inlet,
            cfg=cfg,
            segment=label,
            segment_id=seg_id,
        )
        if sample is not None:
            out.append(sample)
    return out


def graph_to_2d_samples(
    graph_path: Path,
    *,
    cfg: UnwrapGridConfig | None = None,
    norm_stats: Mapping[str, Dict[str, float]] | None = None,
    norm_params_path: Path | str = DEFAULT_NORM_PARAMS,
    unwrap_mode: str = "global",
) -> List[Dict[str, Any]]:
    """按展开模式返回 1..N 个 2D 样本。"""
    cfg = cfg or UnwrapGridConfig()
    if unwrap_mode == "branch":
        return graph_to_2d_branch_samples(
            graph_path, cfg=cfg, norm_stats=norm_stats, norm_params_path=norm_params_path
        )

    stats = norm_stats if norm_stats is not None else load_norm_stats(norm_params_path)
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    x = data.x.numpy()
    y_wss = data.y_wss.numpy()
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    if int(np.sum(wall)) < 20:
        return []

    sample = _wall_subset_to_grid_sample(
        graph_path=graph_path,
        xyz=x[wall, :3].astype(np.float64),
        absc=x[wall, NODE_IDX["Abscissa"]].astype(np.float64),
        tan=x[wall, X_TAN].astype(np.float64),
        norm_r=x[wall, NODE_IDX["NormRadius"]].astype(np.float32),
        curv=x[wall, NODE_IDX["Curvature"]].astype(np.float32),
        wss_norm=y_wss[wall, 0].astype(np.float32),
        wss_phys=denorm_wss_mag(y_wss[wall, 0].astype(np.float32), stats).astype(np.float32),
        bc_inlet=float(data.global_cond.view(-1)[1].item()) if hasattr(data, "global_cond") else 0.0,
        cfg=cfg,
        segment="global",
        segment_id=-1,
    )
    return [sample] if sample is not None else []


def remap_grid_to_wall_points(
    pred_grid: np.ndarray,
    remap: Mapping[str, np.ndarray],
    *,
    use_phys: bool = False,
    stats: Mapping[str, Dict[str, float]] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """2D 网格预测 → 壁面点（cell 取值）。"""
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


def merged_branch_points_r2(
    segment_preds: List[np.ndarray],
    segment_samples: List[Dict[str, Any]],
    stats: Mapping[str, Dict[str, float]],
) -> float:
    """G4-b：合并同一 graph 各分支段的 3D 壁面点 R²。"""
    wss_std = float(stats.get("wss", {}).get("std", 1.0))
    wss_mean = float(stats.get("wss", {}).get("mean", 0.0))
    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    for pred_grid, sample in zip(segment_preds, segment_samples):
        h, w = sample["y_wss_phys"].shape[1], sample["y_wss_phys"].shape[2]
        pred_norm = pred_grid[0, :h, :w] if pred_grid.ndim == 3 else pred_grid[:h, :w]
        pp, gt = remap_grid_to_wall_points(pred_norm, sample["remap"], stats=stats)
        preds.append(pp * wss_std + wss_mean)
        gts.append(sample["remap"]["wss_phys"])
    if not preds:
        return float("nan")
    return r2_score(np.concatenate(gts), np.concatenate(preds))


def collect_graph_paths(cases: List[str], data_root: Path, graphs_subdir: str = "processed/graphs") -> List[Path]:
    out: List[Path] = []
    for case_rel in cases:
        gdir = data_root / case_rel / graphs_subdir
        if gdir.is_dir():
            out.extend(sorted(gdir.glob("*.pt")))
    return out
