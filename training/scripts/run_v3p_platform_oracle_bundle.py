#!/usr/bin/env python3
"""V3P 平台期 Ideas A–G · 0 重训 oracle 编排（CPU）。

产出（默认写入 ``outputs/field/f0_decision/``）：
  - v3p_target_noise_floor_<date>.json       (Idea A)
  - v3p_stream_cross_ray_oracle_<date>.json  (Idea B)
  - v3p_branch_scale_oracle_<date>.json       (Idea C/D)
  - v3p_surface_spectral_oracle_<date>.json    (Idea E)
  - v3p_active_data_oracle_<date>.json       (Idea F)
  - v3p_v3d_lite_data_audit_<date>.json      (Idea G)

用法::

    python -m training.scripts.run_v3p_platform_oracle_bundle
    python -m training.scripts.run_v3p_platform_oracle_bundle --only A,B --verbose
    python -m training.scripts.run_v3p_platform_oracle_bundle --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from pipeline.config import NODE_FEATURE_NAMES
from ._figure_utils import load_json, load_manifest, load_prediction_payload, save_json
from .probe_linear_wss import run_probe
from .run_v3_f0_decision import (
    DEFAULT_NORM_PARAMS,
    NODE_IDX as F0_NODE_IDX,
    REPO_ROOT,
    X_TAN,
    Y_P,
    Y_VEL,
    _denorm_velocity,
    _denorm_wss_mag,
    _denorm_zscore,
    _load_graph,
    _load_norm_stats,
    _r2_score,
    _safe_pearson,
    _safe_spearman,
)
from .run_v3_g0_oracle import label_qa
from .run_v3_me_oracle_wave import (
    DEFAULT_REFERENCE_RUN,
    oracle_o1_scale_decoupling,
    oracle_o7_branch_scale,
)

NODE_IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
BRANCH_IDX = NODE_IDX.get("branch_id")
WSS_X, WSS_Y, WSS_Z = 1, 2, 3

ALL_IDEAS = ("A", "B", "C", "D", "E", "F", "G")
FAILURE_CASES = ("fast/CHEN_SHI_MING", "slow/GUO_XI_JIANG", "slow/ZHANG_ZHI_JUN")
V3D_DATA_ROOT = REPO_ROOT / "data_new"
V3D_SPLIT = REPO_ROOT / "training" / "splits" / "split_data_new_v3.json"


def _verdict(go: bool, *, weak: bool = False) -> str:
    if go:
        return "go"
    if weak:
        return "weak_no_go"
    return "no_go"


def _safe_json(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, (np.floating, np.integer)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    return x


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _wall_wss_vec_phys(data, stats: Mapping[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    x = data.x.numpy()
    wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
    y_wss = data.y_wss.numpy()
    wss_vec = np.stack(
        [
            _denorm_zscore(y_wss[wall, i], stats.get(n))
            for i, n in zip((WSS_X, WSS_Y, WSS_Z), ("wss_x", "wss_y", "wss_z"))
        ],
        axis=1,
    )
    wss_mag = _denorm_wss_mag(y_wss[wall, 0], stats)
    return wall, wss_vec, wss_mag


def _stream_cross_components(
    xyz: np.ndarray,
    tan: np.ndarray,
    wss_vec: np.ndarray,
) -> Dict[str, np.ndarray]:
    """中心线切向 stream + 壁面切平面 cross 分量。"""
    t_stream = _unit(tan)
    ref = np.array([0.0, 0.0, 1.0]) if abs(float(t_stream[0, 2])) < 0.9 else np.array([1.0, 0.0, 0.0])
    t_mean = t_stream.mean(axis=0)
    t_mean = t_mean / (np.linalg.norm(t_mean) + 1e-12)
    e1 = ref - (ref @ t_mean) * t_mean
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    n = _unit(np.cross(t_mean, e1))
    n_tile = np.tile(n, (xyz.shape[0], 1))
    t_cross = _unit(np.cross(n_tile, t_stream))
    wss_stream = np.sum(wss_vec * t_stream, axis=1)
    wss_cross = np.sum(wss_vec * t_cross, axis=1)
    return {
        "wss_stream": wss_stream,
        "wss_cross": wss_cross,
        "wss_x": wss_vec[:, 0],
        "wss_y": wss_vec[:, 1],
        "wss_z": wss_vec[:, 2],
    }


# ---------------------------------------------------------------------------
# Idea A
# ---------------------------------------------------------------------------

def oracle_idea_a(
    split_path: Path,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_graphs_per_case: int,
    max_cases: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    from sklearn.linear_model import Ridge

    qa = label_qa(
        split_path,
        data_root=data_root,
        graphs_subdir=graphs_subdir,
        norm_stats=stats,
        max_graphs_per_case=max_graphs_per_case,
        thresholds={
            "normal_invalid_rate_max": 0.05,
            "basis_invalid_rate_max": 0.05,
            "rad_ratio_max": 2.0,
            "rad_frac_p95_max": 1.5,
            "wss_p95_robust_z_max": 4.0,
            "tangent_unit_frac_min": 0.90,
        },
        max_cases=max_cases,
    )

    rng = np.random.default_rng(seed)
    target_names = ("tau", "log_tau", "cf_proxy", "percentile")
    target_r2: Dict[str, Optional[float]] = {}
    baseline_tau_r2: Optional[float] = None

    def _probe_target(y_fn: Callable[[np.ndarray], np.ndarray]) -> Optional[float]:
        X_list, y_list = [], []
        split = SplitSpec.from_json(split_path)
        for case_rel in split.train_cases[: min(24, len(split.train_cases))]:
            case_dir = data_root / case_rel / "processed" / graphs_subdir
            if not case_dir.is_dir():
                continue
            for gp in sorted(case_dir.glob("*.pt"))[: max(1, max_graphs_per_case // 2)]:
                data = _load_graph(gp)
                x = data.x.numpy()
                wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
                if int(np.sum(wall)) < 20:
                    continue
                feat = np.stack([
                    x[wall, F0_NODE_IDX["Abscissa"]],
                    x[wall, F0_NODE_IDX["NormRadius"]],
                    x[wall, F0_NODE_IDX["Curvature"]],
                ], axis=1)
                _, _, wss_mag = _wall_wss_vec_phys(data, stats)
                y = y_fn(np.clip(wss_mag, 1e-12, None))
                if y.size > 1500:
                    idx = rng.choice(y.size, 1500, replace=False)
                    feat, y = feat[idx], y[idx]
                X_list.append(feat)
                y_list.append(y)
        if not X_list:
            return None
        X = np.concatenate(X_list, 0)
        y = np.concatenate(y_list, 0)
        idx = rng.permutation(X.shape[0])
        cut = int(X.shape[0] * 0.8)
        tr, te = idx[:cut], idx[cut:]
        m = Ridge(alpha=1.0).fit(X[tr], y[tr])
        return float(_r2_score(y[te], m.predict(X[te])))

    baseline_tau_r2 = _probe_target(lambda w: w)
    target_r2["tau"] = baseline_tau_r2
    target_r2["log_tau"] = _probe_target(lambda w: np.log(w))
    target_r2["cf_proxy"] = _probe_target(lambda w: w / (0.5 + 1e-12))
    target_r2["percentile"] = _probe_target(
        lambda w: np.argsort(np.argsort(w)).astype(np.float64) / max(w.size - 1, 1)
    )

    best_name = max(
        (k for k in target_names if target_r2.get(k) is not None),
        key=lambda k: target_r2[k] or -1.0,
        default="tau",
    )
    best_r2 = target_r2.get(best_name)
    delta_vs_tau = (
        float(best_r2) - float(baseline_tau_r2)
        if best_r2 is not None and baseline_tau_r2 is not None
        else None
    )

    # fast/slow 分布
    pace_stats: Dict[str, Dict[str, float]] = {}
    for row in qa.get("cases", []):
        pg = row.get("pace_group", "unknown")
        pace_stats.setdefault(str(pg), {"n": 0, "wss_p95_sum": 0.0})
        pace_stats[str(pg)]["n"] += 1
        pace_stats[str(pg)]["wss_p95_sum"] += float(row.get("wss_p95_phys") or 0.0)
    for pg, d in pace_stats.items():
        d["wss_p95_mean"] = d["wss_p95_sum"] / max(d["n"], 1)
        del d["wss_p95_sum"]

    go = (
        delta_vs_tau is not None
        and delta_vs_tau >= 0.08
        and best_name != "tau"
        and qa.get("pass", False)
    )
    return {
        "idea": "A",
        "label": "target_noise_floor",
        "context": "V3P split_AG_v1 · post5463 band · Idea A 标签/target 地板",
        "qa": {
            "pass": qa.get("pass"),
            "n_denylist_candidates": qa.get("n_denylist_candidates"),
            "denylist_candidates": qa.get("denylist_candidates", [])[:10],
        },
        "target_probe_r2": {k: _safe_json(v) for k, v in target_r2.items()},
        "best_target": best_name,
        "delta_best_vs_tau": _safe_json(delta_vs_tau),
        "pace_group_wss_p95": pace_stats,
        "go_thresholds": {
            "delta_probe_r2_min": 0.08,
            "qa_pass_required": True,
            "hotspot_preserve": "manual_review_if_denylist_nonempty",
        },
        "layers": {
            "L1": {"metric": "target_probe_r2", "value": _safe_json(best_r2)},
            "L2": {"metric": "qa_denylist_empty", "value": qa.get("pass")},
            "L3": {"metric": "pace_group_coverage", "value": len(pace_stats) >= 2},
        },
        "verdict": _verdict(go, weak=delta_vs_tau is not None and 0.03 <= delta_vs_tau < 0.08),
        "recommendation": (
            "Go → 换 target 重训并下调 M-E 表述；No-Go → 标签地板非主瓶颈，转 Idea B/F"
        ),
    }


# ---------------------------------------------------------------------------
# Idea B
# ---------------------------------------------------------------------------

def oracle_idea_b(
    split: SplitSpec,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_graphs_per_case: int,
    seed: int,
) -> Dict[str, Any]:
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    chunks: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("wss_stream", "wss_cross", "wss_x", "wss_y", "wss_z", "ray_proxy")
    }
    feat_chunks: List[np.ndarray] = []

    for case_rel in split.train_cases + split.test_cases:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(gp)
            x = data.x.numpy()
            wall_mask, wss_vec, wss_mag = _wall_wss_vec_phys(data, stats)
            if int(np.sum(wall_mask)) < 20:
                continue
            comp = _stream_cross_components(
                x[wall_mask, :3].astype(np.float64),
                x[wall_mask, X_TAN].astype(np.float64),
                wss_vec,
            )
            for k in ("wss_stream", "wss_cross", "wss_x", "wss_y", "wss_z"):
                chunks[k].append(comp[k])

            # normal-ray proxy: mu * u_t / delta using dist_to_wall
            if data.y is not None and F0_NODE_IDX.get("dist_to_wall") is not None:
                vel = _denorm_velocity(data.y.numpy()[wall_mask, Y_VEL], stats)
                tan = _unit(x[wall_mask, X_TAN].astype(np.float64))
                u_t = np.sum(vel * tan, axis=1)
                delta = np.clip(
                    x[wall_mask, F0_NODE_IDX["dist_to_wall"]].astype(np.float64) * 1e-3,
                    1e-6,
                    None,
                )
                mu = 3.5e-3  # blood viscosity Pa·s (proxy)
                proxy = mu * np.abs(u_t) / delta
                chunks["ray_proxy"].append(proxy)

            feat = np.stack([
                x[wall_mask, F0_NODE_IDX["Abscissa"]],
                x[wall_mask, F0_NODE_IDX["NormRadius"]],
                x[wall_mask, F0_NODE_IDX["Curvature"]],
            ], axis=1)
            feat_chunks.append(feat)

    pooled = {k: np.concatenate(v) if v else np.empty(0) for k, v in chunks.items()}
    if pooled["wss_cross"].size < 100:
        return {"idea": "B", "error": "insufficient_wall_points", "verdict": "no_go"}

    n = pooled["wss_cross"].size
    if n > 8000:
        idx = rng.choice(n, 8000, replace=False)
        for k in pooled:
            pooled[k] = pooled[k][idx]
        X = np.concatenate(feat_chunks, 0)
        if X.shape[0] > 8000:
            X = X[:8000]
    else:
        X = np.concatenate(feat_chunks, 0) if feat_chunks else np.empty((0, 3))

    perm = rng.permutation(min(X.shape[0], pooled["wss_cross"].size))
    cut = int(len(perm) * 0.8)
    tr, te = perm[:cut], perm[cut:]

    def _comp_r2(y_name: str) -> float:
        m = Ridge(alpha=1.0).fit(X[tr], pooled[y_name][tr])
        return float(_r2_score(pooled[y_name][te], m.predict(X[te])))

    r2_cross = _comp_r2("wss_cross")
    r2_x = _comp_r2("wss_x")
    r2_y = _comp_r2("wss_y")
    delta_cross = r2_cross - max(r2_x, r2_y)
    ray_spearman = None
    if pooled["ray_proxy"].size == pooled["wss_cross"].size and pooled["ray_proxy"].size > 50:
        wss_mag = np.sqrt(pooled["wss_x"] ** 2 + pooled["wss_y"] ** 2 + pooled["wss_z"] ** 2)
        ray_spearman = _safe_spearman(pooled["ray_proxy"], wss_mag)

    go = (delta_cross >= 0.10) or (ray_spearman is not None and ray_spearman > 0.65)
    return {
        "idea": "B",
        "label": "stream_cross_ray",
        "context": "V3P · stream/cross frame + normal-ray proxy",
        "probe_r2": {
            "wss_cross": _safe_json(r2_cross),
            "wss_x": _safe_json(r2_x),
            "wss_y": _safe_json(r2_y),
            "delta_cross_vs_global_xy": _safe_json(delta_cross),
        },
        "normal_ray": {
            "spearman_vs_wss_mag": _safe_json(ray_spearman),
            "delta_mm_proxy": "dist_to_wall * 1e-3",
        },
        "go_thresholds": {"delta_cross_min": 0.10, "ray_spearman_min": 0.65},
        "layers": {
            "L1": {"metric": "cross_component_r2", "value": _safe_json(r2_cross)},
            "L2": {"metric": "delta_vs_global_xy", "value": _safe_json(delta_cross)},
            "L3": {"metric": "ray_spearman", "value": _safe_json(ray_spearman)},
        },
        "verdict": _verdict(go, weak=delta_cross >= 0.05),
        "gpu_mapping": "V3P-J1-StreamCrossRay-WSS",
        "recommendation": "Go → J1 seed1；No-Go → 坐标系非主瓶颈",
    }


# ---------------------------------------------------------------------------
# Idea C/D
# ---------------------------------------------------------------------------

def oracle_idea_cd(
    split: SplitSpec,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_graphs_per_case: int,
    max_wall_sample: int,
    seed: int,
) -> Dict[str, Any]:
    o7 = oracle_o7_branch_scale(
        split, data_root, stats,
        graphs_subdir=graphs_subdir,
        max_graphs_per_case=max_graphs_per_case,
        max_wall_sample=max_wall_sample,
        seed=seed,
    )
    o1 = oracle_o1_scale_decoupling(
        split, data_root, stats,
        graphs_subdir=graphs_subdir,
        max_graphs_per_case=max_graphs_per_case,
        max_wall_sample=max_wall_sample,
        seed=seed,
    )

    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(seed)
    branch_feats: List[np.ndarray] = []
    branch_err: List[np.ndarray] = []
    case_vars: List[float] = []

    for case_rel in split.test_cases[:12]:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        wss_all: List[np.ndarray] = []
        branch_wss: Dict[int, List[float]] = {0: [], 1: []}
        curv: List[float] = []
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(gp)
            x = data.x.numpy()
            wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
            if int(np.sum(wall)) < 10:
                continue
            _, _, wss_mag = _wall_wss_vec_phys(data, stats)
            wss_all.append(wss_mag)
            curv.extend(x[wall, F0_NODE_IDX["Curvature"]].tolist())
            if BRANCH_IDX is not None:
                b = (x[wall, BRANCH_IDX] > 0.5).astype(np.int32)
                for bi in (0, 1):
                    if np.any(b == bi):
                        branch_wss[bi].append(float(np.mean(wss_mag[b == bi])))
        if not wss_all:
            continue
        w = np.concatenate(wss_all)
        case_vars.append(float(np.var(w)))
        s0 = branch_wss[0][-1] if branch_wss[0] else float(np.mean(w))
        s1 = branch_wss[1][-1] if branch_wss[1] else s0
        branch_feats.append(np.array([
            float(np.mean(curv)) if curv else 0.0,
            abs(s0 - s1) / (max(s0, s1) + 1e-12),
            float(np.percentile(w, 95)),
            float(np.mean(w)),
        ]))
        branch_err.append(np.array([float(np.var(w))]))

    explained_frac = None
    if len(branch_feats) >= 8:
        Xb = np.stack(branch_feats, 0)
        yb = np.array([e[0] for e in branch_err])
        m = HistGradientBoostingRegressor(max_depth=3, random_state=seed)
        idx = rng.permutation(len(Xb))
        tr, te = idx[: int(len(idx) * 0.75)], idx[int(len(idx) * 0.75):]
        m.fit(Xb[tr], yb[tr])
        pred = m.predict(Xb[te])
        ss_res = float(np.sum((yb[te] - pred) ** 2))
        ss_tot = float(np.sum((yb[te] - np.mean(yb[te])) ** 2)) + 1e-12
        explained_frac = 1.0 - ss_res / ss_tot

    delta_branch = o7.get("delta_branch_minus_case")
    pattern_r2 = o1.get("pattern_r2_pooled_probe")
    scale_r2 = o1.get("scale_r2_gbdt_test")

    go_c = (
        (delta_branch is not None and float(delta_branch) >= 0.05)
        or (explained_frac is not None and explained_frac > 0.30)
    )
    go_d = (
        pattern_r2 is not None
        and scale_r2 is not None
        and float(pattern_r2) > float(scale_r2) + 0.05
    )
    go = go_c or go_d

    return {
        "idea": "C/D",
        "label": "branch_scale_mode",
        "context": "V3P · branch circuit + mode-scale (O7 + O1 extended)",
        "branch_scale_o7": o7,
        "mode_scale_o1": o1,
        "branch_error_explained_frac": _safe_json(explained_frac),
        "go_thresholds": {
            "branch_delta_min": 0.05,
            "branch_error_explained_min": 0.30,
            "pattern_vs_scale_delta_min": 0.05,
        },
        "layers": {
            "L1": {"metric": "pattern_r2", "value": _safe_json(pattern_r2)},
            "L2": {"metric": "branch_delta", "value": _safe_json(delta_branch)},
            "L3": {"metric": "case_var_explained", "value": _safe_json(explained_frac)},
        },
        "verdict": _verdict(go, weak=go_c != go_d and (go_c or go_d)),
        "gpu_mapping": "V3P-J2-BranchScaleMode-WSS",
        "recommendation": "Go → J2 mode-scale 双头；No-Go → branch condition 非主瓶颈",
    }


# ---------------------------------------------------------------------------
# Idea E
# ---------------------------------------------------------------------------

def _spectral_reconstruct(wall_xyz: np.ndarray, wss: np.ndarray, n_modes: int, k: int = 12) -> Tuple[float, float]:
    """图 Laplacian 谱重构 R² 与 top-10% hotspot Dice。"""
    from sklearn.neighbors import NearestNeighbors

    n = wall_xyz.shape[0]
    if n < max(n_modes + 5, 30):
        return float("nan"), float("nan")
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
    nn.fit(wall_xyz)
    _, idx = nn.kneighbors(wall_xyz)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in idx[i, 1:]:
            w = 1.0
            rows.extend([i, int(j)])
            cols.extend([int(j), i])
            vals.extend([w, w])
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    deg = np.array(W.sum(axis=1)).flatten()
    deg[deg < 1e-12] = 1e-12
    D_inv_sqrt = csr_matrix(
        (1.0 / np.sqrt(deg), (range(n), range(n))), shape=(n, n)
    )
    L = csr_matrix(np.eye(n)) - D_inv_sqrt @ W @ D_inv_sqrt
    k_ev = min(n_modes, n - 2)
    try:
        _, vecs = eigsh(L, k=k_ev, which="SM")
    except Exception:
        return float("nan"), float("nan")
    coef, _, _, _ = np.linalg.lstsq(vecs, wss, rcond=None)
    recon = vecs @ coef
    r2 = float(_r2_score(wss, recon))
    thr = float(np.percentile(wss, 90))
    hot_gt = wss >= thr
    hot_pred = recon >= thr
    dice = float(2 * np.sum(hot_gt & hot_pred) / (np.sum(hot_gt) + np.sum(hot_pred) + 1e-12))
    return r2, dice


def oracle_idea_e(
    split: SplitSpec,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_graphs_per_case: int,
    n_modes_list: Sequence[int] = (20, 50, 100),
) -> Dict[str, Any]:
    case_rows: List[Dict[str, Any]] = []
    focus_cases = list(FAILURE_CASES) + [c for c in split.test_cases if c not in FAILURE_CASES][:6]

    for case_rel in focus_cases:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        xyz_list, wss_list = [], []
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(gp)
            x = data.x.numpy()
            wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
            if int(np.sum(wall)) < 30:
                continue
            _, _, wss_mag = _wall_wss_vec_phys(data, stats)
            if wss_mag.size > 2000:
                idx = np.random.default_rng(42).choice(wss_mag.size, 2000, replace=False)
                xyz_list.append(x[wall, :3][idx])
                wss_list.append(wss_mag[idx])
            else:
                xyz_list.append(x[wall, :3])
                wss_list.append(wss_mag)
        if not xyz_list:
            continue
        xyz = np.concatenate(xyz_list, 0).astype(np.float64)
        wss = np.concatenate(wss_list, 0)
        modes_out = {}
        for nm in n_modes_list:
            r2, dice = _spectral_reconstruct(xyz, wss, nm)
            modes_out[str(nm)] = {"r2": _safe_json(r2), "top10_dice": _safe_json(dice)}
        case_rows.append({"case": case_rel, "n_wall": int(wss.size), "modes": modes_out})

    r50 = [r["modes"].get("50", {}).get("r2") for r in case_rows]
    d50 = [r["modes"].get("50", {}).get("top10_dice") for r in case_rows]
    r50_finite = [x for x in r50 if x is not None and math.isfinite(float(x))]
    d50_finite = [x for x in d50 if x is not None and math.isfinite(float(x))]
    mean_r50 = float(np.mean(r50_finite)) if r50_finite else None
    mean_d50 = float(np.mean(d50_finite)) if d50_finite else None

    chen_row = next((r for r in case_rows if "CHEN" in r["case"]), None)
    chen_r50 = chen_row["modes"].get("50", {}).get("r2") if chen_row else None

    go = mean_r50 is not None and mean_r50 > 0.65 and (mean_d50 or 0) > 0.3
    return {
        "idea": "E",
        "label": "surface_spectral",
        "context": "V3P · graph Laplacian spectral reconstruction per case",
        "cases": case_rows,
        "aggregate": {
            "mean_r2_50_modes": _safe_json(mean_r50),
            "mean_top10_dice_50_modes": _safe_json(mean_d50),
            "chen_r2_50_modes": chen_r50,
        },
        "go_thresholds": {"r2_50_min": 0.65, "top10_dice_min": 0.30},
        "layers": {
            "L1": {"metric": "mean_spectral_r2_50", "value": _safe_json(mean_r50)},
            "L2": {"metric": "mean_hotspot_dice", "value": _safe_json(mean_d50)},
            "L3": {"metric": "chen_fast_r2_50", "value": chen_r50},
        },
        "verdict": _verdict(go, weak=mean_r50 is not None and mean_r50 > 0.55),
        "gpu_mapping": "V3P-J3-SpectralRegularizedWall-WSS or G4c-peak-merge",
        "recommendation": "Go → 谱正则/peak-merge；No-Go → G4-c 保持封口",
    }


# ---------------------------------------------------------------------------
# Idea F
# ---------------------------------------------------------------------------

def oracle_idea_f(
    split: SplitSpec,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    reference_run_dir: Path,
    *,
    graphs_subdir: str,
    max_graphs_per_case: int,
    seed: int,
) -> Dict[str, Any]:
    from sklearn.cluster import KMeans

    pred_dir = reference_run_dir / "predictions_test_best_wss"
    manifest = pred_dir / "manifest.json"
    abs_err: List[float] = []
    pace: List[str] = []
    geom_feat: List[np.ndarray] = []
    high_err_recall = None

    if manifest.is_file():
        man = load_manifest(manifest)
        for item in man.get("items", [])[:200]:
            pred_path = Path(str(item.get("prediction_path", "")))
            if not pred_path.is_absolute():
                pred_path = REPO_ROOT / pred_path
            if not pred_path.is_file():
                continue
            payload = load_prediction_payload(pred_path)
            if "y_wss_true" not in payload or "y_wss_pred" not in payload:
                continue
            gt = payload["y_wss_true"].detach().cpu().numpy()[:, 0]
            pred = payload["y_wss_pred"].detach().cpu().numpy()[:, 0]
            err = np.abs(gt - pred)
            abs_err.extend(err.tolist())
            case = str(item.get("case_name", ""))
            pace.extend([case.split("/")[0] if "/" in case else "unknown"] * err.size)
            gp = item.get("graph_path")
            if gp:
                gpath = Path(str(gp))
                if not gpath.is_absolute():
                    gpath = REPO_ROOT / gpath
                if gpath.is_file():
                    data = _load_graph(gpath)
                    x = data.x.numpy()
                    wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
                    if int(np.sum(wall)) >= err.size:
                        feat = np.stack([
                            x[wall, F0_NODE_IDX["Abscissa"]],
                            x[wall, F0_NODE_IDX["NormRadius"]],
                            x[wall, F0_NODE_IDX["Curvature"]],
                        ], axis=1)[: err.size]
                        geom_feat.append(feat)

    if len(abs_err) < 500:
        # fallback: GT variance proxy as pseudo-error
        rng = np.random.default_rng(seed)
        for case_rel in split.test_cases:
            case_dir = data_root / case_rel / "processed" / graphs_subdir
            if not case_dir.is_dir():
                continue
            for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
                data = _load_graph(gp)
                _, _, wss = _wall_wss_vec_phys(data, stats)
                if wss.size > 500:
                    wss = rng.choice(wss, 500, replace=False)
                case_mean = float(np.mean(wss))
                abs_err.extend(np.abs(wss - case_mean).tolist())
                pace.extend([case_rel.split("/")[0]] * wss.size)
                x = data.x.numpy()
                wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
                feat = np.stack([
                    x[wall, F0_NODE_IDX["Abscissa"]],
                    x[wall, F0_NODE_IDX["NormRadius"]],
                    x[wall, F0_NODE_IDX["Curvature"]],
                ], axis=1)
                if feat.shape[0] > wss.size:
                    feat = feat[: wss.size]
                geom_feat.append(feat)

    err_arr = np.asarray(abs_err, dtype=np.float64)
    if geom_feat:
        G = np.concatenate(geom_feat, 0)
        if G.shape[0] > err_arr.size:
            G = G[: err_arr.size]
        elif G.shape[0] < err_arr.size:
            err_arr = err_arr[: G.shape[0]]
    else:
        G = np.random.default_rng(seed).normal(size=(min(len(err_arr), 5000), 3))

    n = min(G.shape[0], err_arr.size, 5000)
    G, err_arr = G[:n], err_arr[:n]
    if n < 100:
        return {"idea": "F", "error": "insufficient_error_samples", "verdict": "no_go"}

    rng = np.random.default_rng(seed)
    km = KMeans(n_clusters=4, random_state=seed, n_init=10)
    labels = km.fit_predict(G)
    cluster_err = [float(np.mean(err_arr[labels == k])) for k in range(4)]
    top_cluster = int(np.argmax(cluster_err))
    top_mask = labels == top_cluster
    thr = float(np.percentile(err_arr, 80))
    high_err = err_arr >= thr
    recalled = float(np.sum(high_err & top_mask) / (np.sum(high_err) + 1e-12))
    explained = float(
        np.sum(err_arr[top_mask] ** 2) / (np.sum(err_arr ** 2) + 1e-12)
    )

    pace_fast = [e for e, p in zip(err_arr, pace[: len(err_arr)]) if p == "fast"]
    pace_slow = [e for e, p in zip(err_arr, pace[: len(err_arr)]) if p == "slow"]

    active_data_candidates = []
    if pace_fast and pace_slow and np.mean(pace_fast) > np.mean(pace_slow) * 1.1:
        active_data_candidates.append({
            "cluster": "fast_domain",
            "reason": "fast_mean_abs_err > slow",
            "action": "TODO-1: prioritize fast-phase CFD with similar geometry to CHEN",
        })
    active_data_candidates.append({
        "cluster": f"kmeans_{top_cluster}",
        "reason": f"explains {explained:.1%} squared error",
        "action": "TODO-1: active selection by high-WSS hotspot geometry",
    })

    go = recalled > 0.70 or explained > 0.30
    return {
        "idea": "F",
        "label": "active_data",
        "context": "V3P · error cluster + pseudo-uncertainty for TODO-1",
        "reference_run": str(reference_run_dir),
        "high_error_recall_top_cluster": _safe_json(recalled),
        "top_cluster_explained_sq_err_frac": _safe_json(explained),
        "pace_mean_abs_err": {
            "fast": _safe_json(float(np.mean(pace_fast)) if pace_fast else None),
            "slow": _safe_json(float(np.mean(pace_slow)) if pace_slow else None),
        },
        "active_data_candidates": active_data_candidates,
        "go_thresholds": {"recall_min": 0.70, "explained_sq_err_min": 0.30},
        "layers": {
            "L1": {"metric": "error_recall", "value": _safe_json(recalled)},
            "L2": {"metric": "cluster_explained_frac", "value": _safe_json(explained)},
            "L3": {"metric": "fast_vs_slow_err_ratio", "value": _safe_json(
                float(np.mean(pace_fast) / (np.mean(pace_slow) + 1e-12))
                if pace_fast and pace_slow else None
            )},
        },
        "verdict": _verdict(go, weak=explained > 0.20),
        "recommendation": "Go → TODO-1 主动选数清单；可与 Idea G 并联",
    }


# ---------------------------------------------------------------------------
# Idea G
# ---------------------------------------------------------------------------

def _case_wss_stats(case_rel: str, data_root: Path, stats: Mapping, max_graphs: int) -> Optional[Dict[str, Any]]:
    case_dir = data_root / case_rel / "processed" / "graphs"
    if not case_dir.is_dir():
        return None
    wss_vals, p_vals, curv = [], [], []
    for gp in sorted(case_dir.glob("*.pt"))[:max_graphs]:
        data = _load_graph(gp)
        x = data.x.numpy()
        wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
        if int(np.sum(wall)) < 10:
            continue
        _, _, w = _wall_wss_vec_phys(data, stats)
        wss_vals.append(w)
        if data.y is not None:
            p_vals.append(_denorm_zscore(data.y.numpy()[wall, Y_P], stats.get("p")))
        curv.extend(x[wall, F0_NODE_IDX["Curvature"]].tolist())
    if not wss_vals:
        return None
    w = np.concatenate(wss_vals)
    out = {
        "case": case_rel,
        "domain": case_rel.split("/")[0] if "/" in case_rel else "unknown",
        "wss_p50": float(np.percentile(w, 50)),
        "wss_p95": float(np.percentile(w, 95)),
        "wss_p99": float(np.percentile(w, 99)),
        "curvature_mean": float(np.mean(curv)) if curv else None,
    }
    if p_vals:
        p = np.concatenate(p_vals)
        out["p_mean"] = float(np.mean(p))
    return out


def oracle_idea_g(
    ag_split: SplitSpec,
    stats: Mapping[str, Dict[str, float]],
    *,
    max_graphs_per_case: int,
    seed: int,
) -> Dict[str, Any]:
    v3d_split_path = V3D_SPLIT
    ag_root = REPO_ROOT / "data_new" / "AG"
    pools = {"AG-only": [], "V3D-lite-candidates": [], "full-V3D": []}

    for case in ag_split.train_cases + ag_split.test_cases:
        row = _case_wss_stats(case, ag_root, stats, max_graphs_per_case)
        if row:
            pools["AG-only"].append(row)

    if v3d_split_path.is_file():
        v3d = SplitSpec.from_json(v3d_split_path)
        for case in v3d.train_cases + v3d.val_cases + v3d.test_cases:
            dom = case.split("/")[0] if "/" in case else "unknown"
            case_dir = V3D_DATA_ROOT / case / "processed" / "graphs"
            if not case_dir.is_dir():
                continue
            wss_vals, p_vals, curv = [], [], []
            for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
                data = _load_graph(gp)
                x = data.x.numpy()
                wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
                if int(np.sum(wall)) < 10:
                    continue
                _, _, w = _wall_wss_vec_phys(data, stats)
                wss_vals.append(w)
                if data.y is not None:
                    p_vals.append(_denorm_zscore(data.y.numpy()[wall, Y_P], stats.get("p")))
                curv.extend(x[wall, F0_NODE_IDX["Curvature"]].tolist())
            if not wss_vals:
                continue
            w = np.concatenate(wss_vals)
            row = {
                "case": case,
                "domain": dom,
                "wss_p50": float(np.percentile(w, 50)),
                "wss_p95": float(np.percentile(w, 95)),
                "wss_p99": float(np.percentile(w, 99)),
                "curvature_mean": float(np.mean(curv)) if curv else None,
            }
            if p_vals:
                row["p_mean"] = float(np.mean(np.concatenate(p_vals)))
            pools["full-V3D"].append(row)

    # failure cluster: CHEN-like (fast + high wss_p95)
    ag_fail = [r for r in pools["AG-only"] if "CHEN" in r["case"] or r.get("domain") == "fast"]
    if ag_fail:
        ref = ag_fail[0]
        for row in pools["full-V3D"]:
            if row["domain"] == "AG":
                dist = abs(row["wss_p95"] - ref["wss_p95"]) + abs(row.get("curvature_mean", 0) - ref.get("curvature_mean", 0))
                row["near_domain_score"] = float(dist)
        candidates = sorted(
            [r for r in pools["full-V3D"] if r.get("domain") == "AG" and "near_domain_score" in r],
            key=lambda r: r["near_domain_score"],
        )[:15]
        pools["V3D-lite-candidates"] = candidates

    domain_agg: Dict[str, Dict[str, float]] = {}
    for row in pools["full-V3D"]:
        d = row["domain"]
        domain_agg.setdefault(d, {"n": 0, "wss_p95_sum": 0.0})
        domain_agg[d]["n"] += 1
        domain_agg[d]["wss_p95_sum"] += row["wss_p95"]
    for d, v in domain_agg.items():
        v["wss_p95_mean"] = v["wss_p95_sum"] / max(v["n"], 1)

    go = len(pools["V3D-lite-candidates"]) >= 3
    return {
        "idea": "G",
        "label": "v3d_lite_data_audit",
        "context": "V3P · V3D distribution audit + near-domain lite pool (no full V3D train)",
        "pools": {
            "AG-only": {"n": len(pools["AG-only"]), "sample": pools["AG-only"][:5]},
            "V3D-lite-candidates": {
                "n": len(pools["V3D-lite-candidates"]),
                "cases": [c["case"] for c in pools["V3D-lite-candidates"]],
            },
            "full-V3D": {"n": len(pools["full-V3D"]), "domain_agg": domain_agg},
        },
        "ag_failure_cluster": [r["case"] for r in ag_fail[:5]],
        "go_thresholds": {"lite_candidates_min": 3, "require_ag_test_non_regression": True},
        "layers": {
            "L1": {"metric": "lite_pool_size", "value": len(pools["V3D-lite-candidates"])},
            "L2": {"metric": "domain_coverage", "value": len(domain_agg)},
            "L3": {"metric": "failure_cluster_mapped", "value": len(ag_fail) > 0},
        },
        "verdict": _verdict(go),
        "gpu_mapping": "V3P-J4-V3Dlite-Pretrain-AGFinetune (seed1 short only if Go)",
        "recommendation": "Go → V3D-lite 统计/线性 probe；No-Go → 数据扩池非主瓶颈",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

IDEA_OUTPUT = {
    "A": "v3p_target_noise_floor",
    "B": "v3p_stream_cross_ray_oracle",
    "C": "v3p_branch_scale_oracle",
    "D": "v3p_branch_scale_oracle",
    "E": "v3p_surface_spectral_oracle",
    "F": "v3p_active_data_oracle",
    "G": "v3p_v3d_lite_data_audit",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P platform period oracle bundle A–G")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--graphs-subdir", default="graphs")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--reference-run-dir", type=Path, default=REPO_ROOT / DEFAULT_REFERENCE_RUN)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/field/f0_decision")
    ap.add_argument("--date-suffix", default="")
    ap.add_argument("--only", default="", help="A,B,C,...")
    ap.add_argument("--skip", default="")
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--max-cases", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max-wall-sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="1 graph/case, fewer cases")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.max_graphs_per_case = 1
        args.max_cases = 6
        args.max_wall_sample = 400

    t0 = time.perf_counter()
    suffix = args.date_suffix or date.today().strftime("%Y%m%d")
    out_dir = ensure_dir(args.output_dir.resolve())
    split_path = args.split.resolve()
    split = SplitSpec.from_json(split_path)
    data_root = args.data_root.resolve()
    stats = _load_norm_stats(args.norm_params.resolve())
    ref_run = args.reference_run_dir.resolve()
    max_cases = args.max_cases if args.max_cases > 0 else None

    only = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    skip = {x.strip().upper() for x in args.skip.split(",") if x.strip()}
    active = [i for i in ALL_IDEAS if (not only or i in only) and i not in skip]

    def log(msg: str) -> None:
        if args.verbose:
            print(f"[platform oracle] {msg}", flush=True)

    results: Dict[str, Any] = {
        "ideas": {},
        "config": {
            "split": str(split_path),
            "data_root": str(data_root),
            "reference_run_dir": str(ref_run),
            "max_graphs_per_case": args.max_graphs_per_case,
            "smoke": args.smoke,
        },
    }

    if "A" in active:
        log("Idea A target noise floor ...")
        res_a = oracle_idea_a(
            split_path, data_root, stats,
            graphs_subdir=args.graphs_subdir,
            max_graphs_per_case=args.max_graphs_per_case,
            max_cases=max_cases,
            seed=args.seed,
        )
        results["ideas"]["A"] = res_a
        save_json(out_dir / f"{IDEA_OUTPUT['A']}_{suffix}.json", res_a)

    if "B" in active:
        log("Idea B stream/cross ...")
        res_b = oracle_idea_b(
            split, data_root, stats,
            graphs_subdir=args.graphs_subdir,
            max_graphs_per_case=args.max_graphs_per_case,
            seed=args.seed,
        )
        results["ideas"]["B"] = res_b
        save_json(out_dir / f"{IDEA_OUTPUT['B']}_{suffix}.json", res_b)

    if "C" in active or "D" in active:
        log("Idea C/D branch/mode-scale ...")
        res_cd = oracle_idea_cd(
            split, data_root, stats,
            graphs_subdir=args.graphs_subdir,
            max_graphs_per_case=args.max_graphs_per_case,
            max_wall_sample=args.max_wall_sample,
            seed=args.seed,
        )
        results["ideas"]["C/D"] = res_cd
        save_json(out_dir / f"{IDEA_OUTPUT['C']}_{suffix}.json", res_cd)

    if "E" in active:
        log("Idea E spectral ...")
        res_e = oracle_idea_e(
            split, data_root, stats,
            graphs_subdir=args.graphs_subdir,
            max_graphs_per_case=args.max_graphs_per_case,
        )
        results["ideas"]["E"] = res_e
        save_json(out_dir / f"{IDEA_OUTPUT['E']}_{suffix}.json", res_e)

    if "F" in active:
        log("Idea F active data ...")
        res_f = oracle_idea_f(
            split, data_root, stats, ref_run,
            graphs_subdir=args.graphs_subdir,
            max_graphs_per_case=args.max_graphs_per_case,
            seed=args.seed,
        )
        results["ideas"]["F"] = res_f
        save_json(out_dir / f"{IDEA_OUTPUT['F']}_{suffix}.json", res_f)

    if "G" in active:
        log("Idea G V3D-lite ...")
        res_g = oracle_idea_g(
            split, stats,
            max_graphs_per_case=args.max_graphs_per_case,
            seed=args.seed,
        )
        results["ideas"]["G"] = res_g
        save_json(out_dir / f"{IDEA_OUTPUT['G']}_{suffix}.json", res_g)

    results["elapsed_s"] = round(time.perf_counter() - t0, 2)
    save_json(out_dir / f"v3p_platform_oracle_bundle_{suffix}.json", results)
    if args.verbose:
        print(json.dumps({k: v.get("verdict") for k, v in results["ideas"].items()}, indent=2))


if __name__ == "__main__":
    main()
