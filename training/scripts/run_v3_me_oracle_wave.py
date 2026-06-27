#!/usr/bin/env python3
"""V3P M-E · O1–O12 零重训 oracle 波次（工程化精度方案 §5+.x · CPU）。

在 AsymW 微调线封口后，对结构性方向 E-G–E-R 做低成本先验筛子，
产出 ``outputs/field/f0_decision/v3p_me_oracle_wave_<date>.json``。

O6（E-L / I6-a）已完训，本脚本写入引用结果，不重复 GPU 探针。

用法::

    python -m training.scripts.run_v3_me_oracle_wave
    python -m training.scripts.run_v3_me_oracle_wave --only O1,O3,O10 --verbose
    python -m training.scripts.run_v3_me_oracle_wave --skip O6,O11 --max-graphs-per-case 5
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.denylist import filter_case_names
from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from pipeline.config import NODE_FEATURE_NAMES
from pipeline.wall_unwrap.grid import compute_theta
from ._figure_utils import load_json, load_manifest, load_prediction_payload, save_json
from .run_v3_f0_decision import (
    DEFAULT_NORM_PARAMS,
    NODE_IDX as F0_NODE_IDX,
    REPO_ROOT,
    X_TAN,
    Y_P,
    _denorm_wss_mag,
    _denorm_zscore,
    _load_graph,
    _load_norm_stats,
    _r2_score,
    _safe_pearson,
    _safe_spearman,
    oracle_pod,
    oracle_pod_2d,
    oracle_pressure_gradient_wss,
)
from .run_v3_g0_oracle import g0d_residual_baseline, g0c_pressure_geom_gbdt, _collect_pressure_geom

NODE_IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
BRANCH_IDX = NODE_IDX.get("branch_id")

DEFAULT_REFERENCE_RUN = (
    "outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000"
    "_split_AG_v1_seed1_20260619_174001"
)
I6A_REFERENCE = {
    "exp_id": "V3P-I6-a-AsymW-a-post5463",
    "job": 5810,
    "test_wss_r2_wss": 0.44570499658584595,
    "baseline_wss_r2_wss": 0.42863863706588745,
    "delta": 0.017066359519958496,
    "best_wss_epoch": 12,
    "verdict": "weak_no_go",
}

ALL_ORACLES = (
    "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "O11", "O12",
)


def _verdict(go: bool, *, weak: bool = False) -> str:
    if go:
        return "go"
    if weak:
        return "weak_no_go"
    return "no_go"


def _mean_finite(vals: Iterable[float]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else None


GLOBAL_BC_INLET = 1
GLOBAL_BC_O1 = 2


def _graph_global_cond(data) -> np.ndarray:
    if hasattr(data, "global_cond") and data.global_cond is not None:
        return data.global_cond.view(-1).numpy().astype(np.float64)
    return np.zeros(6, dtype=np.float64)


def _collect_case_graphs(
    cases: Sequence[str],
    data_root: Path,
    graphs_subdir: str,
    *,
    max_graphs_per_case: int,
) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for case_rel in cases:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            out.append((case_rel, gp))
    return out


def _wall_wss_phys(data, stats: Mapping[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    x = data.x.numpy()
    wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
    y_wss = data.y_wss.numpy()
    wss = _denorm_wss_mag(y_wss[wall, 0], stats)
    return wall, wss


def _case_scale(wss: np.ndarray) -> float:
    if wss.size == 0:
        return float("nan")
    return float(np.mean(wss))


def oracle_o1_scale_decoupling(
    split: SplitSpec,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_graphs_per_case: int,
    max_wall_sample: int,
    seed: int,
) -> Dict[str, Any]:
    """E-G · 病例级量纲解耦：去尺度模式 vs S_case 可预测性。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    raw_feats: List[np.ndarray] = []
    raw_y: List[np.ndarray] = []
    norm_y: List[np.ndarray] = []
    case_rows: List[Dict[str, Any]] = []

    for case_rel in split.train_cases + split.test_cases:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        bc_inlet, bc_o1, mean_r = [], [], []
        wss_case: List[np.ndarray] = []
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(gp)
            x = data.x.numpy()
            wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
            if int(np.sum(wall)) < 20:
                continue
            _, wss = _wall_wss_phys(data, stats)
            wss_case.append(wss)
            gc = _graph_global_cond(data)
            bc_inlet.append(float(gc[GLOBAL_BC_INLET]))
            bc_o1.append(float(gc[GLOBAL_BC_O1]))
            mean_r.append(float(np.mean(x[wall, F0_NODE_IDX["NormRadius"]])))
        if not wss_case:
            continue
        wss_all = np.concatenate(wss_case)
        s_case = _case_scale(wss_all)
        if not math.isfinite(s_case) or s_case <= 1e-12:
            continue
        w_norm = wss_all / s_case
        case_rows.append({
            "case": case_rel,
            "s_case": s_case,
            "bc_inlet_max": float(np.mean(bc_inlet)),
            "bc_o1_max": float(np.mean(bc_o1)),
            "mean_norm_radius": float(np.mean(mean_r)),
        })
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(gp)
            x = data.x.numpy()
            wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
            if int(np.sum(wall)) < 20:
                continue
            feat = np.stack([
                x[wall, F0_NODE_IDX["Abscissa"]],
                x[wall, F0_NODE_IDX["NormRadius"]],
                x[wall, F0_NODE_IDX["Curvature"]],
            ], axis=1).astype(np.float64)
            _, wss = _wall_wss_phys(data, stats)
            if wss.size > max_wall_sample:
                idx = rng.choice(wss.size, size=max_wall_sample, replace=False)
                feat, wss = feat[idx], wss[idx]
            raw_feats.append(feat)
            raw_y.append(wss)
            norm_y.append(wss / s_case)

    if not raw_feats:
        return {"error": "no_wall_data", "verdict": "no_go"}

    X = np.concatenate(raw_feats, axis=0)
    y_raw = np.concatenate(raw_y, axis=0)
    y_norm = np.concatenate(norm_y, axis=0)
    n = X.shape[0]
    idx = rng.permutation(n)
    cut = int(n * 0.8)
    tr, te = idx[:cut], idx[cut:]
    ridge = Ridge(alpha=1.0)
    ridge.fit(X[tr], y_raw[tr])
    raw_r2 = _r2_score(y_raw[te], ridge.predict(X[te]))
    ridge_n = Ridge(alpha=1.0)
    ridge_n.fit(X[tr], y_norm[tr])
    pattern_r2 = _r2_score(y_norm[te], ridge_n.predict(X[te]))

    s_cases = np.array([r["s_case"] for r in case_rows], dtype=np.float64)
    s_feat = np.array([
        [r["bc_inlet_max"], r["bc_o1_max"], r["mean_norm_radius"], math.log(max(r["s_case"], 1e-12))]
        for r in case_rows
    ], dtype=np.float64)
    s_feat = s_feat[:, :3]
    if len(case_rows) >= 4:
        s_tr, s_te = np.arange(len(case_rows))[: len(case_rows) * 2 // 3], np.arange(len(case_rows))[len(case_rows) * 2 // 3 :]
        gbdt = HistGradientBoostingRegressor(max_depth=3, max_iter=100, random_state=seed)
        gbdt.fit(s_feat[s_tr], np.log(s_cases[s_tr] + 1e-12))
        s_pred = gbdt.predict(s_feat[s_te])
        s_case_r2 = _r2_score(np.log(s_cases[s_te] + 1e-12), s_pred)
    else:
        s_case_r2 = float("nan")

    go_pattern = pattern_r2 >= 0.60
    go_scale = (s_case_r2 if math.isfinite(s_case_r2) else -1.0) >= 0.80
    go = go_pattern and go_scale
    return {
        "direction": "E-G",
        "pattern_r2_pooled_probe": _safe_json(pattern_r2),
        "raw_r2_pooled_probe": _safe_json(raw_r2),
        "s_case_log_r2_gbdt": _safe_json(s_case_r2),
        "n_cases": len(case_rows),
        "n_wall_points": int(y_raw.size),
        "go_thresholds": {"pattern_r2": 0.60, "s_case_r2": 0.80},
        "verdict": _verdict(go),
        "recommendation": (
            "Go → 立项 E-G 模式+标定双头；No-Go → 跨病例 scale 不是主瓶颈"
        ),
    }


def oracle_o2_analytic_prior(
    split_path: Path,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_graphs_per_case: int,
    max_cases: int,
) -> Dict[str, Any]:
    """E-H · Poiseuille 解析先验 + 残差（复用 G0-d）。"""
    rep = g0d_residual_baseline(
        split_path,
        data_root=data_root,
        graphs_subdir=graphs_subdir,
        norm_stats=stats,
        max_graphs_per_case=max_graphs_per_case,
        max_cases=max_cases,
    )
    pooled_red = rep.get("pooled_var_reduction")
    mean_red = rep.get("mean_var_reduction")
    trunk_proxy = pooled_red  # per-case Poiseuille 已含 trunk 主导段
    resid_ratio = 1.0 - float(pooled_red) if pooled_red is not None else None
    go = (
        trunk_proxy is not None
        and float(trunk_proxy) >= 0.50
        and resid_ratio is not None
        and float(resid_ratio) <= 0.60
    )
    return {
        "direction": "E-H",
        "g0d": rep,
        "pooled_var_reduction": pooled_red,
        "mean_var_reduction": mean_red,
        "resid_var_fraction": resid_ratio,
        "go_thresholds": {"trunk_tau0_r2_proxy": 0.50, "resid_var_fraction_max": 0.60},
        "verdict": _verdict(go),
        "recommendation": "Go → E-H 残差头立项；No-Go → 解析先验不足以降方差",
    }


def oracle_o3_lb_pod(
    split_path: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    grid_s: int,
    n_sectors: int,
    n_modes: int,
    max_graphs_per_case: int,
) -> Dict[str, Any]:
    """E-I/E-C · 2D (s,θ)-POD 重构上限（LB 谱 proxy：同口径 POD）。"""
    pod_1d = oracle_pod(
        split_path, grid_size=grid_s, max_graphs_per_case=max_graphs_per_case, modes=[10, 20, 50],
    )
    pod_2d = oracle_pod_2d(
        split_path, stats, grid_s=grid_s, sectors=[1, n_sectors], modes=[10, 20, 50],
        max_graphs_per_case=max_graphs_per_case,
    )
    r2_50_1d = next((m["test_mean_r2"] for m in pod_1d.get("modes", []) if m.get("n_modes") == 50), None)
    r2_50_2d = None
    for block in pod_2d.get("results", []):
        if block.get("n_sectors") == n_sectors:
            for m in block.get("modes", []):
                if m.get("n_modes") == 50:
                    r2_50_2d = m.get("test_mean_r2")
    go = r2_50_2d is not None and float(r2_50_2d) >= 0.90
    if not go and r2_50_1d is not None and r2_50_2d is not None:
        go = float(r2_50_2d) >= float(r2_50_1d) + 0.05 and float(r2_50_2d) >= 0.85
    return {
        "direction": "E-I/E-C",
        "pod_1d": pod_1d,
        "pod_2d": pod_2d,
        "test_r2_50_modes_1d": r2_50_1d,
        "test_r2_50_modes_2d": r2_50_2d,
        "go_thresholds": {"r2_50_2d_min": 0.90, "or_delta_vs_1d": 0.05},
        "verdict": _verdict(bool(go)),
        "recommendation": "Go → 低秩/POD/LB 表示换轨；No-Go → target 侧低秩上限不足",
    }


def oracle_o4_pressure_gradient(
    split_path: Path,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_cases: int,
    max_graphs_per_case: int,
    max_wall_per_graph: int,
    seed: int,
) -> Dict[str, Any]:
    """E-J · 壁面 ∂p/∂s 与 WSS 闭合 + 增量 probe。"""
    from sklearn.ensemble import HistGradientBoostingRegressor

    pg = oracle_pressure_gradient_wss(
        split_path, stats,
        max_cases=max_cases, max_graphs_per_case=max_graphs_per_case,
        max_wall_per_graph=max_wall_per_graph,
    )
    gbdt_full = g0c_pressure_geom_gbdt(
        split_path, data_root=data_root, graphs_subdir=graphs_subdir, norm_stats=stats,
        max_graphs_per_case=max_graphs_per_case, max_wall_per_graph=max_wall_per_graph,
        seed=seed, max_train_cases=max_cases, max_test_cases=max_cases,
    )
    spearman = pg.get("mean_spearman_neg_dpds")
    full_r2 = gbdt_full.get("test_r2")

    # 几何-only 对照：复用 G0-c 特征列 [Abscissa, Curvature, NormRadius, ...]
    split = SplitSpec.from_json(split_path)
    Xtr, ytr = _collect_pressure_geom(
        split.train_cases, data_root=data_root, graphs_subdir=graphs_subdir, norm_stats=stats,
        max_graphs_per_case=max_graphs_per_case, max_wall_per_graph=max_wall_per_graph, k_wall=12,
        seed=seed, max_cases=max_cases,
    )
    Xte, yte = _collect_pressure_geom(
        split.test_cases, data_root=data_root, graphs_subdir=graphs_subdir, norm_stats=stats,
        max_graphs_per_case=max_graphs_per_case, max_wall_per_graph=max_wall_per_graph, k_wall=12,
        seed=seed + 1, max_cases=max_cases,
    )
    geom_r2 = None
    delta = None
    if Xtr.shape[0] >= 50 and Xte.shape[0] >= 20:
        geom_tr, geom_te = Xtr[:, 2:], Xte[:, 2:]
        reg_g = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, l2_regularization=1.0, random_state=0,
        ).fit(geom_tr, ytr)
        geom_r2 = _r2_score(yte, reg_g.predict(geom_te))
        if geom_r2 is not None and full_r2 is not None:
            delta = float(full_r2) - float(geom_r2)

    closed = spearman is not None and abs(float(spearman)) >= 0.30
    go = closed and delta is not None and delta >= 0.05
    return {
        "direction": "E-J",
        "pressure_gradient_corr": pg,
        "gbdt_pressure_geom": gbdt_full,
        "geom_only_test_r2": _safe_json(geom_r2),
        "mean_spearman_neg_dpds": spearman,
        "delta_r2_with_pressure_features": delta,
        "go_thresholds": {"abs_spearman_min": 0.30, "delta_r2_min": 0.05},
        "verdict": _verdict(bool(go)),
        "recommendation": "Go → ∂p/∂s 作 WSS 输入/一致性；No-Go → 压力梯度不携带足够 WSS 信息",
    }


def oracle_o5_freq_band(
    split_path: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    grid_s: int,
    n_sectors: int,
    max_graphs_per_case: int,
    reference_run_dir: Optional[Path],
) -> Dict[str, Any]:
    """E-K · 高频带误差占比（有 reference run 时用预测误差，否则 GT 谱分解 proxy）。"""
    from .run_v3_f0_decision import _collect_profiles_2d

    split = SplitSpec.from_json(split_path)
    profiles = _collect_profiles_2d(
        split.test_cases, stats, grid_s=grid_s, n_sectors=n_sectors,
        max_graphs_per_case=max_graphs_per_case,
    )
    if profiles.shape[0] < 2:
        return {"error": "insufficient_profiles", "verdict": "no_go"}

    mean = profiles.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(profiles - mean, full_matrices=False)
    n_modes = min(50, vt.shape[0])
    basis = vt[:n_modes]
    recon = ((profiles - mean) @ basis.T) @ basis + mean
    resid = profiles - recon
    low_energy = float(np.sum(recon ** 2))
    high_energy = float(np.sum(resid ** 2))
    high_frac = high_energy / max(low_energy + high_energy, 1e-20)

    pred_high_frac = None
    if reference_run_dir and (reference_run_dir / "predictions_test_best_wss" / "manifest.json").is_file():
        manifest = load_manifest(reference_run_dir / "predictions_test_best_wss" / "manifest.json")
        errs: List[float] = []
        for item in (manifest.get("items") or [])[:200]:
            pred_path = REPO_ROOT / item["prediction_path"]
            if not pred_path.is_file():
                continue
            payload = load_prediction_payload(pred_path)
            y_t = payload["y_wss_true"].detach().cpu().numpy()[:, 0]
            y_p = payload["y_wss_pred"].detach().cpu().numpy()[:, 0]
            err = y_t - y_p
            errs.append(float(np.var(err)))
        pred_high_frac = _mean_finite(errs)

    go = high_frac >= 0.50
    return {
        "direction": "E-K",
        "gt_high_freq_energy_fraction": _safe_json(high_frac),
        "pred_error_var_mean_sampled": pred_high_frac,
        "n_test_profiles": int(profiles.shape[0]),
        "reference_run": str(reference_run_dir) if reference_run_dir else None,
        "go_thresholds": {"high_freq_fraction_min": 0.50},
        "verdict": _verdict(go),
        "recommendation": "Go → hotspot/高频残差阶段值得立项；No-Go → 误差非高频主导",
    }


def oracle_o6_reference() -> Dict[str, Any]:
    """E-L / I6-a · 已完成探针引用。"""
    ref = dict(I6A_REFERENCE)
    ref.update({
        "direction": "E-L/E-E",
        "go_thresholds": {"delta_min": 0.02, "best_wss_epoch_min": 20},
        "verdict": "weak_no_go",
        "recommendation": "边际 +0.017 未达 Go；仅作叠加增益参考，不单独立项 GPU",
    })
    return ref


def oracle_o7_branch_scale(
    split: SplitSpec,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_graphs_per_case: int,
    max_wall_sample: int,
    seed: int,
) -> Dict[str, Any]:
    """E-M · per-branch vs per-case 去尺度。"""
    from sklearn.linear_model import Ridge

    if BRANCH_IDX is None:
        return {"error": "branch_id_missing", "verdict": "no_go"}

    rng = np.random.default_rng(seed)

    def _collect(descale: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats, y_raw, y_norm = [], [], []
        for case_rel in split.train_cases + split.test_cases:
            case_dir = data_root / case_rel / "processed" / graphs_subdir
            if not case_dir.is_dir():
                continue
            branch_wss: Dict[int, List[np.ndarray]] = {0: [], 1: []}
            case_wss: List[np.ndarray] = []
            for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
                data = _load_graph(gp)
                x = data.x.numpy()
                wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
                if int(np.sum(wall)) < 20:
                    continue
                _, wss = _wall_wss_phys(data, stats)
                case_wss.append(wss)
                branch = (x[wall, BRANCH_IDX] > 0.5).astype(np.int32)
                for b in (0, 1):
                    wb = branch == b
                    if int(np.sum(wb)) >= 10:
                        branch_wss[b].append(wss[wb])
            if not case_wss:
                continue
            s_case = _case_scale(np.concatenate(case_wss))
            s_branch = {
                b: _case_scale(np.concatenate(v)) if v else float("nan")
                for b, v in branch_wss.items()
            }
            for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
                data = _load_graph(gp)
                x = data.x.numpy()
                wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
                if int(np.sum(wall)) < 20:
                    continue
                feat = np.stack([
                    x[wall, F0_NODE_IDX["Abscissa"]],
                    x[wall, F0_NODE_IDX["NormRadius"]],
                    x[wall, F0_NODE_IDX["Curvature"]],
                ], axis=1).astype(np.float64)
                _, wss = _wall_wss_phys(data, stats)
                branch = (x[wall, BRANCH_IDX] > 0.5).astype(np.int32)
                if descale == "case":
                    scale = np.full(wss.shape, s_case)
                else:
                    scale = np.array([s_branch.get(int(b), s_case) for b in branch], dtype=np.float64)
                scale = np.clip(scale, 1e-12, None)
                if wss.size > max_wall_sample:
                    idx = rng.choice(wss.size, size=max_wall_sample, replace=False)
                    feat, wss, scale = feat[idx], wss[idx], scale[idx]
                feats.append(feat)
                y_raw.append(wss)
                y_norm.append(wss / scale)
        if not feats:
            return np.empty((0, 3)), np.empty(0), np.empty(0)
        return np.concatenate(feats, 0), np.concatenate(y_raw, 0), np.concatenate(y_norm, 0)

    Xc, _, yn_case = _collect("case")
    Xb, _, yn_branch = _collect("branch")
    if Xc.size == 0 or Xb.size == 0:
        return {"error": "no_branch_data", "verdict": "no_go"}

    n = Xc.shape[0]
    idx = rng.permutation(n)
    cut = int(n * 0.8)
    tr, te = idx[:cut], idx[cut:]
    ridge_c = Ridge(alpha=1.0).fit(Xc[tr], yn_case[tr])
    ridge_b = Ridge(alpha=1.0).fit(Xb[tr], yn_branch[tr])
    r2_case = _r2_score(yn_case[te], ridge_c.predict(Xc[te]))
    r2_branch = _r2_score(yn_branch[te], ridge_b.predict(Xb[te]))
    delta = float(r2_branch) - float(r2_case)
    go = delta >= 0.05
    return {
        "direction": "E-M",
        "pattern_r2_per_case_descale": _safe_json(r2_case),
        "pattern_r2_per_branch_descale": _safe_json(r2_branch),
        "delta_branch_minus_case": _safe_json(delta),
        "go_thresholds": {"delta_min": 0.05},
        "verdict": _verdict(go),
        "recommendation": "Go → S_branch 拆分立项；No-Go → 病例级 scale 已够",
    }


def oracle_o8_circuit_proxy(
    split: SplitSpec,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_graphs_per_case: int,
) -> Dict[str, Any]:
    """E-N · branch 统计量 vs circuit proxy (BC × R^-3)。"""
    rows: List[Dict[str, float]] = []
    for case_rel in split.test_cases:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(gp)
            x = data.x.numpy()
            wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
            if int(np.sum(wall)) < 20 or BRANCH_IDX is None:
                continue
            _, wss = _wall_wss_phys(data, stats)
            branch = (x[wall, BRANCH_IDX] > 0.5).astype(np.int32)
            gc = _graph_global_cond(data)
            bc = float(gc[GLOBAL_BC_INLET])
            for b in (0, 1):
                m = branch == b
                if int(np.sum(m)) < 10:
                    continue
                r_mean = float(np.mean(x[wall, F0_NODE_IDX["NormRadius"]][m]))
                proxy = bc * (max(r_mean, 0.05) ** -3)
                rows.append({
                    "branch_mean_wss": float(np.mean(wss[m])),
                    "branch_top10_wss": float(np.mean(np.partition(wss[m], -max(1, int(0.1 * np.sum(m))))[-max(1, int(0.1 * np.sum(m))):])),
                    "circuit_proxy": proxy,
                    "mean_norm_radius": r_mean,
                })

    if len(rows) < 8:
        return {"error": "insufficient_branch_rows", "n_rows": len(rows), "verdict": "no_go"}

    y = np.array([r["branch_mean_wss"] for r in rows])
    x_proxy = np.array([r["circuit_proxy"] for r in rows])
    x_geom = np.array([r["mean_norm_radius"] for r in rows])
    r2_proxy = _r2_score(y, x_proxy) if np.std(x_proxy) > 1e-12 else float("nan")
    r2_geom = _r2_score(y, x_geom) if np.std(x_geom) > 1e-12 else float("nan")
    delta = (float(r2_proxy) - float(r2_geom)) if math.isfinite(r2_proxy) and math.isfinite(r2_geom) else None
    go = r2_proxy is not None and math.isfinite(r2_proxy) and (
        float(r2_proxy) >= 0.70 or (delta is not None and delta >= 0.10)
    )
    return {
        "direction": "E-N",
        "n_branch_rows": len(rows),
        "branch_mean_wss_r2_circuit_proxy": _safe_json(r2_proxy),
        "branch_mean_wss_r2_geom_only": _safe_json(r2_geom),
        "delta_proxy_minus_geom": _safe_json(delta),
        "go_thresholds": {"r2_proxy_min": 0.70, "or_delta_vs_geom": 0.10},
        "verdict": _verdict(bool(go)),
        "recommendation": "Go → circuit latent 立项；No-Go → 分支流量 proxy 不足",
    }


def _frame_index(sample_id: str) -> int:
    m = re.search(r"merged-(\d+)", sample_id)
    return int(m.group(1)) if m else 0


def oracle_o9_temporal(
    split: SplitSpec,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_graphs_per_case: int,
    n_harmonics: int,
) -> Dict[str, Any]:
    """E-O · 81 帧时间低秩/谐波重构。"""
    recon_r2s: List[float] = []
    for case_rel in split.test_cases[: min(8, len(split.test_cases))]:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        series: List[Tuple[int, float]] = []
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(gp)
            _, wss = _wall_wss_phys(data, stats)
            series.append((_frame_index(gp.stem), float(np.mean(wss))))
        if len(series) < 8:
            continue
        series.sort(key=lambda t: t[0])
        y = np.array([v for _, v in series], dtype=np.float64)
        n = y.size
        t = np.arange(n, dtype=np.float64)
        X = [np.ones(n)]
        for k in range(1, n_harmonics + 1):
            w = 2 * math.pi * k / max(n, 1)
            X.append(np.sin(w * t))
            X.append(np.cos(w * t))
        design = np.column_stack(X)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        recon = design @ coef
        recon_r2s.append(_r2_score(y, recon))

    mean_r2 = _mean_finite(recon_r2s)
    go = mean_r2 is not None and mean_r2 >= 0.90
    return {
        "direction": "E-O",
        "n_cases_temporal": len(recon_r2s),
        "mean_temporal_recon_r2": _safe_json(mean_r2),
        "n_harmonics": n_harmonics,
        "per_case_r2": [_safe_json(v) for v in recon_r2s],
        "go_thresholds": {"recon_r2_min": 0.90},
        "verdict": _verdict(bool(go)),
        "recommendation": "Go → 时间系数建模；No-Go → 逐帧独立假设可接受",
    }


def _topk_dice(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.10) -> float:
    n = y_true.size
    if n == 0:
        return float("nan")
    k = max(1, int(round(n * frac)))
    t_idx = set(int(i) for i in np.argpartition(y_true, -k)[-k:])
    p_idx = set(int(i) for i in np.argpartition(y_pred, -k)[-k:])
    inter = len(t_idx & p_idx)
    return 2.0 * inter / (len(t_idx) + len(p_idx))


def oracle_o10_hotspot(
    reference_run_dir: Path,
    *,
    max_items: int,
    top_frac: float,
) -> Dict[str, Any]:
    """E-P · hotspot 定位 vs 幅值低估。"""
    manifest_path = reference_run_dir / "predictions_test_best_wss" / "manifest.json"
    if not manifest_path.is_file():
        return {"error": "missing_predictions", "verdict": "no_go"}

    manifest = load_manifest(manifest_path)
    dices: List[float] = []
    biases: List[float] = []
    for item in (manifest.get("items") or [])[:max_items]:
        pred_path = REPO_ROOT / item["prediction_path"]
        if not pred_path.is_file():
            continue
        payload = load_prediction_payload(pred_path)
        y_t = payload["y_wss_true"].detach().cpu().numpy()[:, 0].astype(np.float64)
        y_p = payload["y_wss_pred"].detach().cpu().numpy()[:, 0].astype(np.float64)
        wall = np.isfinite(y_t) & np.isfinite(y_p)
        y_t, y_p = y_t[wall], y_p[wall]
        if y_t.size < 50:
            continue
        dices.append(_topk_dice(y_t, y_p, top_frac))
        k = max(1, int(round(y_t.size * top_frac)))
        hot = y_t >= np.partition(y_t, -k)[-k]
        biases.append(float(np.mean(y_p[hot] - y_t[hot])))

    mean_dice = _mean_finite(dices)
    mean_bias = _mean_finite(biases)
    recall_ok = mean_dice is not None and mean_dice >= 0.25
    bias_negative = mean_bias is not None and mean_bias < 0
    go = recall_ok and bias_negative
    return {
        "direction": "E-P",
        "reference_run": str(reference_run_dir),
        "mean_topk_dice": _safe_json(mean_dice),
        "mean_hotspot_bias_pred_minus_gt": _safe_json(mean_bias),
        "n_samples": len(dices),
        "go_thresholds": {"topk_dice_min": 0.25, "hotspot_bias_negative": True},
        "verdict": _verdict(bool(go)),
        "recommendation": (
            "定位可+幅值负偏 → E-P/E-Q 标定路线；定位差 → E-A/E-I 表示换轨"
        ),
    }


def oracle_o11_calibration(
    reference_run_dir: Path,
    split: SplitSpec,
    *,
    max_val_items: int,
    max_test_items: int,
) -> Dict[str, Any]:
    """E-Q · 单调校准诊断（val 拟合 · test 评）。"""
    from sklearn.isotonic import IsotonicRegression

    manifest_path = reference_run_dir / "predictions_test_best_wss" / "manifest.json"
    if not manifest_path.is_file():
        return {"error": "missing_predictions", "verdict": "no_go"}

    val_cases = set(split.val_cases)
    test_cases = set(split.test_cases)
    manifest = load_manifest(manifest_path)

    def _collect(case_filter: set, limit: int) -> Tuple[np.ndarray, np.ndarray]:
        yt, yp = [], []
        for item in manifest.get("items") or []:
            if len(yt) >= limit:
                break
            case_name = str(item.get("case_name", ""))
            if case_name not in case_filter:
                continue
            pred_path = REPO_ROOT / item["prediction_path"]
            if not pred_path.is_file():
                continue
            payload = load_prediction_payload(pred_path)
            t = payload["y_wss_true"].detach().cpu().numpy()[:, 0]
            p = payload["y_wss_pred"].detach().cpu().numpy()[:, 0]
            m = np.isfinite(t) & np.isfinite(p)
            yt.append(t[m])
            yp.append(p[m])
        if not yt:
            return np.empty(0), np.empty(0)
        return np.concatenate(yt), np.concatenate(yp)

    yv_t, yv_p = _collect(val_cases, max_val_items)
    yt_t, yt_p = _collect(test_cases, max_test_items)
    if yv_t.size < 100 or yt_t.size < 100:
        return {"error": "insufficient_val_test", "verdict": "no_go"}

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(yv_p, yv_t)
    cal = iso.predict(yt_p)
    r2_raw = _r2_score(yt_t, yt_p)
    r2_cal = _r2_score(yt_t, cal)
    rmse_raw = float(np.sqrt(np.mean((yt_t - yt_p) ** 2)))
    rmse_cal = float(np.sqrt(np.mean((yt_t - cal) ** 2)))
    rmse_drop = (rmse_raw - rmse_cal) / max(rmse_raw, 1e-20)
    delta_r2 = float(r2_cal) - float(r2_raw)
    go = rmse_drop >= 0.10 or delta_r2 >= 0.05
    return {
        "direction": "E-Q",
        "reference_run": str(reference_run_dir),
        "test_r2_raw": _safe_json(r2_raw),
        "test_r2_calibrated": _safe_json(r2_cal),
        "test_rmse_raw": _safe_json(rmse_raw),
        "test_rmse_calibrated": _safe_json(rmse_cal),
        "rmse_relative_drop": _safe_json(rmse_drop),
        "delta_r2": _safe_json(delta_r2),
        "go_thresholds": {"rmse_drop_min": 0.10, "or_delta_r2": 0.05},
        "verdict": _verdict(bool(go)),
        "recommendation": "Go → 幅值标定/ E-G 路线；No-Go → 空间模式不足，校准无效",
    }


def oracle_o12_target_transform(
    split: SplitSpec,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_graphs_per_case: int,
    max_wall_sample: int,
    seed: int,
) -> Dict[str, Any]:
    """E-R · target 变换低秩性 / 方差结构。"""
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    targets: Dict[str, List[np.ndarray]] = {
        "tau": [], "log_tau": [], "cf_proxy": [], "percentile": [],
    }
    for case_rel in split.train_cases + split.test_cases:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(gp)
            _, wss = _wall_wss_phys(data, stats)
            if wss.size > max_wall_sample:
                wss = rng.choice(wss, size=max_wall_sample, replace=False)
            wss = np.clip(wss, 1e-12, None)
            targets["tau"].append(wss)
            targets["log_tau"].append(np.log(wss))
            u_ref = 1.0
            targets["cf_proxy"].append(wss / (0.5 * u_ref ** 2 + 1e-12))
            ranks = np.argsort(np.argsort(wss)).astype(np.float64) / max(wss.size - 1, 1)
            targets["percentile"].append(ranks)

    rows = []
    best_name, best_var = None, float("inf")
    for name, chunks in targets.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks)
        var = float(np.var(arr))
        rows.append({"target": name, "variance": var, "p95": float(np.percentile(arr, 95))})
        if var < best_var:
            best_var, best_name = var, name

    # 简单 probe：几何 → target R²
    X_list, y_best = [], []
    for case_rel in split.train_cases[:20]:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        for gp in sorted(case_dir.glob("*.pt"))[:1]:
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
            _, wss = _wall_wss_phys(data, stats)
            wss = np.clip(wss, 1e-12, None)
            if best_name == "log_tau":
                y = np.log(wss)
            elif best_name == "cf_proxy":
                y = wss
            elif best_name == "percentile":
                y = np.argsort(np.argsort(wss)).astype(np.float64) / max(wss.size - 1, 1)
            else:
                y = wss
            if y.size > max_wall_sample:
                idx = rng.choice(y.size, size=max_wall_sample, replace=False)
                feat, y = feat[idx], y[idx]
            X_list.append(feat)
            y_best.append(y)
    probe_r2 = None
    if X_list:
        X = np.concatenate(X_list, 0)
        y = np.concatenate(y_best, 0)
        n = X.shape[0]
        idx = rng.permutation(n)
        cut = int(n * 0.8)
        tr, te = idx[:cut], idx[cut:]
        m = Ridge(alpha=1.0).fit(X[tr], y[tr])
        probe_r2 = _r2_score(y[te], m.predict(X[te]))

    go = best_name is not None and best_name != "tau" and probe_r2 is not None and probe_r2 > 0.45
    return {
        "direction": "E-R",
        "target_stats": rows,
        "lowest_variance_target": best_name,
        "best_target_probe_r2": _safe_json(probe_r2),
        "go_thresholds": {"prefer_non_tau_target": True, "probe_r2_min": 0.45},
        "verdict": _verdict(bool(go), weak=best_name == "log_tau"),
        "recommendation": "Go → 换 target 训练；需同步重定 M-E 门槛口径",
    }


def _safe_json(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, (np.floating, np.integer)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    return x


def _summarize_verdicts(report: Mapping[str, Any]) -> Dict[str, Any]:
    go, weak, no_go = [], [], []
    active = report.get("config", {}).get("active_oracles", ALL_ORACLES)
    for oid in active:
        block = report.get("oracles", {}).get(oid, {})
        v = block.get("verdict", "unknown")
        direction = block.get("direction", oid)
        if v == "go":
            go.append({"id": oid, "direction": direction})
        elif v == "weak_no_go":
            weak.append({"id": oid, "direction": direction})
        else:
            no_go.append({"id": oid, "direction": direction})
    priority = [
        {"rank": i + 1, **item}
        for i, item in enumerate(go)
    ]
    return {
        "n_go": len(go),
        "n_weak_no_go": len(weak),
        "n_no_go": len(no_go),
        "go_list": go,
        "weak_no_go_list": weak,
        "no_go_list": no_go,
        "gpu_priority_order": priority,
        "next_hop": (
            "Go oracle 对应方向可立项 GPU（须单命题 config + 新 exp_id）；"
            "全 No-Go → 维持 G5 叙事收口 · 母版 post5463 band"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P M-E O1–O12 oracle wave")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training" / "splits" / "split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new" / "AG")
    ap.add_argument("--graphs-subdir", default="graphs")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--reference-run-dir", type=Path, default=REPO_ROOT / DEFAULT_REFERENCE_RUN)
    ap.add_argument("--output-name", default="")
    ap.add_argument("--only", default="", help="逗号分隔 O1,O3,...")
    ap.add_argument("--skip", default="", help="逗号分隔跳过项")
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--max-cases", type=int, default=24)
    ap.add_argument("--max-wall-per-graph", type=int, default=800)
    ap.add_argument("--max-wall-sample", type=int, default=2000)
    ap.add_argument("--grid-s", type=int, default=64)
    ap.add_argument("--n-sectors", type=int, default=8)
    ap.add_argument("--n-modes", type=int, default=50)
    ap.add_argument("--n-harmonics", type=int, default=4)
    ap.add_argument("--max-pred-items", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    split_path = args.split.resolve()
    split = SplitSpec.from_json(split_path)
    data_root = args.data_root.resolve()
    stats = _load_norm_stats(args.norm_params.resolve())
    ref_run = args.reference_run_dir.resolve() if args.reference_run_dir else None

    only = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    skip = {x.strip().upper() for x in args.skip.split(",") if x.strip()}
    active = [o for o in ALL_ORACLES if (not only or o in only) and o not in skip]

    def log(msg: str) -> None:
        if args.verbose:
            print(f"[M-E oracle] {msg}", flush=True)

    oracles: Dict[str, Any] = {}

    if "O1" in active:
        log("O1 E-G scale decoupling ...")
        oracles["O1"] = oracle_o1_scale_decoupling(
            split, data_root, stats,
            graphs_subdir=args.graphs_subdir,
            max_graphs_per_case=args.max_graphs_per_case,
            max_wall_sample=args.max_wall_sample,
            seed=args.seed,
        )

    if "O2" in active:
        log("O2 E-H analytic prior ...")
        oracles["O2"] = oracle_o2_analytic_prior(
            split_path, data_root, stats,
            graphs_subdir=args.graphs_subdir,
            max_graphs_per_case=args.max_graphs_per_case,
            max_cases=args.max_cases,
        )

    if "O3" in active:
        log("O3 E-I/E-C POD ...")
        oracles["O3"] = oracle_o3_lb_pod(
            split_path, stats,
            grid_s=args.grid_s, n_sectors=args.n_sectors, n_modes=args.n_modes,
            max_graphs_per_case=args.max_graphs_per_case,
        )

    if "O4" in active:
        log("O4 E-J pressure gradient ...")
        oracles["O4"] = oracle_o4_pressure_gradient(
            split_path, data_root, stats,
            graphs_subdir=args.graphs_subdir,
            max_cases=args.max_cases,
            max_graphs_per_case=args.max_graphs_per_case,
            max_wall_per_graph=args.max_wall_per_graph,
            seed=args.seed,
        )

    if "O5" in active:
        log("O5 E-K freq band ...")
        oracles["O5"] = oracle_o5_freq_band(
            split_path, stats,
            grid_s=args.grid_s, n_sectors=args.n_sectors,
            max_graphs_per_case=args.max_graphs_per_case,
            reference_run_dir=ref_run,
        )

    if "O6" in active:
        log("O6 E-L reference (I6-a) ...")
        oracles["O6"] = oracle_o6_reference()

    if "O7" in active:
        log("O7 E-M branch scale ...")
        oracles["O7"] = oracle_o7_branch_scale(
            split, data_root, stats,
            graphs_subdir=args.graphs_subdir,
            max_graphs_per_case=args.max_graphs_per_case,
            max_wall_sample=args.max_wall_sample,
            seed=args.seed,
        )

    if "O8" in active:
        log("O8 E-N circuit proxy ...")
        oracles["O8"] = oracle_o8_circuit_proxy(
            split, data_root, stats,
            graphs_subdir=args.graphs_subdir,
            max_graphs_per_case=args.max_graphs_per_case,
        )

    if "O9" in active:
        log("O9 E-O temporal ...")
        oracles["O9"] = oracle_o9_temporal(
            split, data_root, stats,
            graphs_subdir=args.graphs_subdir,
            max_graphs_per_case=min(args.max_graphs_per_case * 3, 81),
            n_harmonics=args.n_harmonics,
        )

    if "O10" in active:
        log("O10 E-P hotspot ...")
        if ref_run is None:
            oracles["O10"] = {"error": "no_reference_run", "verdict": "no_go"}
        else:
            oracles["O10"] = oracle_o10_hotspot(
                ref_run, max_items=args.max_pred_items, top_frac=0.10,
            )

    if "O11" in active:
        log("O11 E-Q calibration ...")
        if ref_run is None:
            oracles["O11"] = {"error": "no_reference_run", "verdict": "no_go"}
        else:
            oracles["O11"] = oracle_o11_calibration(
                ref_run, split,
                max_val_items=args.max_pred_items,
                max_test_items=args.max_pred_items,
            )

    if "O12" in active:
        log("O12 E-R target transform ...")
        oracles["O12"] = oracle_o12_target_transform(
            split, data_root, stats,
            graphs_subdir=args.graphs_subdir,
            max_graphs_per_case=args.max_graphs_per_case,
            max_wall_sample=args.max_wall_sample,
            seed=args.seed,
        )

    report: Dict[str, Any] = {
        "label": "V3P-M-E-oracle-wave",
        "date": date.today().isoformat(),
        "motivation": "AsymW 微调封口后 · 工程方案 §5+.x O1–O12 零重训筛子",
        "split": str(split_path),
        "reference_run_dir": str(ref_run) if ref_run else None,
        "baseline_wss_r2_wss": 0.42863863706588745,
        "config": {
            "max_graphs_per_case": args.max_graphs_per_case,
            "max_cases": args.max_cases,
            "grid_s": args.grid_s,
            "n_sectors": args.n_sectors,
            "active_oracles": active,
        },
        "oracles": oracles,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    report["summary"] = _summarize_verdicts(report)

    out_name = args.output_name or f"v3p_me_oracle_wave_{date.today().strftime('%Y%m%d')}.json"
    out_path = REPO_ROOT / "outputs" / "field" / "f0_decision" / out_name
    ensure_dir(out_path.parent)
    save_json(out_path, report)
    print(f"已写入: {out_path}")
    print(
        f"汇总: Go={report['summary']['n_go']} · "
        f"weak={report['summary']['n_weak_no_go']} · "
        f"No-Go={report['summary']['n_no_go']} · "
        f"{report['elapsed_sec']}s"
    )
    if report["summary"]["go_list"]:
        print("Go 方向:", ", ".join(f"{x['id']}({x['direction']})" for x in report["summary"]["go_list"]))


if __name__ == "__main__":
    main()
