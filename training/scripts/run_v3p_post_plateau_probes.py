#!/usr/bin/env python3
"""V3P 后平台期 K1/K3/K5 CPU 短探针（基于 I6-diag 预测误差）。

产物（默认 ``outputs/field/f0_decision/``）：
  - v3p_k1_branch_error_probe_<date>.json
  - v3p_k3_mode_scale_audit_<date>.json
  - v3p_k5_temporal_bc_probe_<date>.json

用法::

    python -m training.scripts.run_v3p_post_plateau_probes
    python -m training.scripts.run_v3p_post_plateau_probes --only K1,K3 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from ._figure_utils import load_manifest, load_prediction_payload, save_json
from .run_v3_f0_decision import (
    DEFAULT_NORM_PARAMS,
    REPO_ROOT,
    _denorm_wss_mag,
    _load_graph,
    _load_norm_stats,
    _r2_score,
)
from .run_v3_me_oracle_wave import (
    GLOBAL_BC_INLET,
    GLOBAL_BC_O1,
    _case_scale,
    _graph_global_cond,
    _wall_wss_phys,
)
from pipeline.config import NODE_FEATURE_NAMES

NODE_IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
BRANCH_IDX = NODE_IDX.get("branch_id")
DIST_BIF_IDX = NODE_IDX.get("dist_to_bifurcation")

DEFAULT_I6_RUN = REPO_ROOT / (
    "outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a"
    "_wall13000_near2000_split_AG_v1_seed1_20260619_174001"
)
ALL_PROBES = ("K1", "K3", "K5")
FOCUS_CASES = ("fast/CHEN_SHI_MING", "slow/GUO_XI_JIANG", "slow/ZHANG_ZHI_JUN")
FRAME_RE = re.compile(r"merged[-_]?(\d+)", re.I)


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


def _explained_frac(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def _parse_frame_id(sample_id: str, graph_path: Path) -> Optional[int]:
    for src in (sample_id, graph_path.stem):
        m = FRAME_RE.search(src)
        if m:
            return int(m.group(1))
    return None


def _iter_prediction_rows(
    manifest_path: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    cases: Optional[set[str]] = None,
) -> Iterable[Dict[str, Any]]:
    man = load_manifest(manifest_path)
    for item in man.get("items", []):
        case = str(item.get("case_name", ""))
        if cases and case not in cases:
            continue
        pred_path = Path(str(item.get("prediction_path", "")))
        if not pred_path.is_absolute():
            pred_path = REPO_ROOT / pred_path
        if not pred_path.is_file():
            continue
        graph_path = Path(str(item.get("graph_path", "")))
        if not graph_path.is_absolute():
            graph_path = REPO_ROOT / graph_path
        if not graph_path.is_file():
            continue
        payload = load_prediction_payload(pred_path)
        if "y_wss_true" not in payload or "y_wss_pred" not in payload:
            continue
        gt = _denorm_wss_mag(payload["y_wss_true"].detach().cpu().numpy()[:, 0], stats.get("wss"))
        pred = _denorm_wss_mag(payload["y_wss_pred"].detach().cpu().numpy()[:, 0], stats.get("wss"))
        data = _load_graph(graph_path)
        x = data.x.numpy()
        wall = x[:, NODE_IDX["is_wall"]] > 0.5
        if int(np.sum(wall)) != gt.size:
            n = min(int(np.sum(wall)), gt.size)
            gt, pred = gt[:n], pred[:n]
            wall_idx = np.where(wall)[0][:n]
        else:
            wall_idx = np.where(wall)[0]
        sq_err = (gt - pred) ** 2
        sample_id = str(item.get("sample_id", graph_path.stem))
        frame_id = _parse_frame_id(sample_id, graph_path)
        gc = _graph_global_cond(data)
        yield {
            "case": case,
            "pace": case.split("/")[0] if "/" in case else "unknown",
            "sample_id": sample_id,
            "frame_id": frame_id,
            "graph_path": str(graph_path),
            "x_wall": x[wall_idx],
            "gt": gt,
            "pred": pred,
            "sq_err": sq_err,
            "abs_err": np.abs(gt - pred),
            "bc_inlet": float(gc[GLOBAL_BC_INLET]) if gc.size > GLOBAL_BC_INLET else 0.0,
            "bc_o1": float(gc[GLOBAL_BC_O1]) if gc.size > GLOBAL_BC_O1 else 0.0,
            "t_norm": float(gc[0]) if gc.size else 0.0,
        }


def _branch_features(x_wall: np.ndarray, gt: np.ndarray, sq_err: np.ndarray) -> Dict[int, Dict[str, float]]:
    if BRANCH_IDX is None:
        return {}
    branch = (x_wall[:, BRANCH_IDX] > 0.5).astype(np.int32)
    out: Dict[int, Dict[str, float]] = {}
    for b in (0, 1):
        mask = branch == b
        if int(np.sum(mask)) < 10:
            continue
        curv = x_wall[mask, NODE_IDX["Curvature"]]
        absc = x_wall[mask, NODE_IDX["Abscissa"]]
        rad = x_wall[mask, NODE_IDX["NormRadius"]]
        dist_bif = (
            x_wall[mask, DIST_BIF_IDX]
            if DIST_BIF_IDX is not None
            else np.zeros(int(np.sum(mask)))
        )
        out[b] = {
            "n_wall": int(np.sum(mask)),
            "mean_sq_err": float(np.mean(sq_err[mask])),
            "mean_abs_err": float(np.mean(np.sqrt(sq_err[mask]))),
            "curvature_mean": float(np.mean(curv)),
            "curvature_p95": float(np.percentile(curv, 95)),
            "abscissa_mean": float(np.mean(absc)),
            "abscissa_span": float(np.max(absc) - np.min(absc)),
            "radius_mean": float(np.mean(rad)),
            "wss_p95_gt": float(np.percentile(gt[mask], 95)),
            "wss_mean_gt": float(np.mean(gt[mask])),
            "dist_bif_mean": float(np.mean(dist_bif)),
        }
    return out


def probe_k1_branch_error(
    manifest_path: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    seed: int,
) -> Dict[str, Any]:
    """K1 · branch/topology error predictor（I6-diag 平方误差）。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    branch_rows: List[Dict[str, Any]] = []
    point_feats: List[np.ndarray] = []
    point_sq: List[np.ndarray] = []
    case_branch_err: Dict[str, Dict[int, float]] = {}

    for row in _iter_prediction_rows(manifest_path, stats):
        bf = _branch_features(row["x_wall"], row["gt"], row["sq_err"])
        for b, feat in bf.items():
            branch_rows.append({
                "case": row["case"],
                "pace": row["pace"],
                "branch_id": b,
                "bc_inlet": row["bc_inlet"],
                "bc_o1": row["bc_o1"],
                **feat,
            })
            case_branch_err.setdefault(row["case"], {})[b] = feat["mean_sq_err"]

        xw = row["x_wall"]
        feat_cols = [
            xw[:, NODE_IDX["Abscissa"]],
            xw[:, NODE_IDX["NormRadius"]],
            xw[:, NODE_IDX["Curvature"]],
        ]
        if DIST_BIF_IDX is not None:
            feat_cols.append(xw[:, DIST_BIF_IDX])
        if BRANCH_IDX is not None:
            feat_cols.append(xw[:, BRANCH_IDX])
        feat_cols.extend([
            np.full(xw.shape[0], row["bc_inlet"]),
            np.full(xw.shape[0], row["bc_o1"]),
        ])
        point_feats.append(np.stack(feat_cols, axis=1).astype(np.float64))
        point_sq.append(row["sq_err"])

    if len(branch_rows) < 12:
        return {"error": "insufficient_branch_rows", "verdict": "no_go"}

    feat_names = [
        "curvature_mean", "curvature_p95", "abscissa_span", "radius_mean",
        "wss_p95_gt", "dist_bif_mean", "bc_inlet", "bc_o1", "branch_id",
    ]
    Xb = np.array([[r[k] for k in feat_names[:-1]] + [float(r["branch_id"])] for r in branch_rows])
    yb = np.array([r["mean_sq_err"] for r in branch_rows], dtype=np.float64)

    rng = np.random.default_rng(seed)
    case_ids = sorted({r["case"] for r in branch_rows})
    if len(case_ids) >= 4:
        # 病例级 holdout，避免同一 case 多帧泄漏到 train/test
        te_cases = set(rng.choice(case_ids, size=max(1, len(case_ids) // 4), replace=False))
        tr_mask = np.array([r["case"] not in te_cases for r in branch_rows])
        te_mask = ~tr_mask
        if int(np.sum(te_mask)) >= 3 and int(np.sum(tr_mask)) >= 3:
            gbdt = HistGradientBoostingRegressor(max_depth=4, max_iter=200, random_state=seed)
            gbdt.fit(Xb[tr_mask], yb[tr_mask])
            branch_explained = _explained_frac(yb[te_mask], gbdt.predict(Xb[te_mask]))
        else:
            branch_explained = float("nan")
    else:
        idx = rng.permutation(len(Xb))
        cut = max(4, int(len(idx) * 0.75))
        tr, te = idx[:cut], idx[cut:]
        gbdt = HistGradientBoostingRegressor(max_depth=4, max_iter=200, random_state=seed)
        gbdt.fit(Xb[tr], yb[tr])
        branch_explained = _explained_frac(yb[te], gbdt.predict(Xb[te])) if te.size >= 3 else float("nan")

    Xp = np.concatenate(point_feats, axis=0)
    yp = np.concatenate(point_sq, axis=0)
    n = Xp.shape[0]
    if n > 120_000:
        sub = rng.choice(n, size=120_000, replace=False)
        Xp, yp = Xp[sub], yp[sub]
        n = Xp.shape[0]
    idx_p = rng.permutation(n)
    cut_p = int(n * 0.8)
    tr_p, te_p = idx_p[:cut_p], idx_p[cut_p:]
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xp[tr_p], yp[tr_p])
    point_explained = _explained_frac(yp[te_p], ridge.predict(Xp[te_p]))

    focus_stats: Dict[str, Any] = {}
    for fc in FOCUS_CASES:
        if fc not in case_branch_err:
            continue
        focus_stats[fc] = case_branch_err[fc]

    fast_err = [r["mean_sq_err"] for r in branch_rows if r["pace"] == "fast"]
    slow_err = [r["mean_sq_err"] for r in branch_rows if r["pace"] == "slow"]
    chen_err = [r["mean_sq_err"] for r in branch_rows if "CHEN" in r["case"]]

    go_branch = branch_explained > 0.30
    go_point = point_explained > 0.15
    go_strat = (
        chen_err
        and fast_err
        and float(np.mean(chen_err)) > float(np.mean(slow_err)) * 1.2
    )
    go = go_branch or (go_point and go_strat)

    return {
        "probe": "K1",
        "label": "branch_topology_error_predictor",
        "context": "V3P · I6-diag prediction sq_err vs branch/topology features",
        "reference_run": str(manifest_path.parent.parent),
        "n_branch_rows": len(branch_rows),
        "n_point_samples": int(n),
        "branch_error_explained_frac": _safe_json(branch_explained),
        "point_sq_err_explained_frac": _safe_json(point_explained),
        "pace_mean_sq_err": {
            "fast": _safe_json(float(np.mean(fast_err)) if fast_err else None),
            "slow": _safe_json(float(np.mean(slow_err)) if slow_err else None),
        },
        "chen_mean_sq_err": _safe_json(float(np.mean(chen_err)) if chen_err else None),
        "focus_case_branch_err": focus_stats,
        "go_thresholds": {
            "branch_error_explained_min": 0.30,
            "point_explained_min": 0.15,
        },
        "layers": {
            "L1": {"metric": "branch_explained_frac", "value": _safe_json(branch_explained)},
            "L2": {"metric": "point_explained_frac", "value": _safe_json(point_explained)},
            "L3": {"metric": "chen_vs_slow_ratio", "value": _safe_json(
                float(np.mean(chen_err) / (np.mean(slow_err) + 1e-12)) if chen_err and slow_err else None
            )},
        },
        "verdict": _verdict(go, weak=go_branch != go and (go_branch or go_point)),
        "gpu_mapping": "V3P-K1-DualGraphWallBranch_seed1",
        "recommendation": (
            "Go → K1 dual-graph 短 GPU probe；No-Go → branch token 难解释当前误差"
        ),
    }


def probe_k3_mode_scale_audit(
    manifest_path: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    seed: int,
) -> Dict[str, Any]:
    """K3 · mode-scale target audit（绝对 vs 去尺度模式 R2）。"""
    case_gt: Dict[str, List[np.ndarray]] = {}
    case_pred: Dict[str, List[np.ndarray]] = {}
    branch_gt: List[np.ndarray] = []
    branch_pred: List[np.ndarray] = []
    case_bc: Dict[str, Tuple[float, float, float]] = {}

    for row in _iter_prediction_rows(manifest_path, stats):
        case_gt.setdefault(row["case"], []).append(row["gt"])
        case_pred.setdefault(row["case"], []).append(row["pred"])
        if row["case"] not in case_bc:
            xw = row["x_wall"]
            case_bc[row["case"]] = (
                row["bc_inlet"],
                row["bc_o1"],
                float(np.mean(xw[:, NODE_IDX["NormRadius"]])),
            )
        if BRANCH_IDX is not None:
            branch = (row["x_wall"][:, BRANCH_IDX] > 0.5).astype(np.int32)
            for b in (0, 1):
                mask = branch == b
                if int(np.sum(mask)) >= 10:
                    bg = row["gt"][mask]
                    bp = row["pred"][mask]
                    s_b = _case_scale(bg)
                    if math.isfinite(s_b) and s_b > 1e-12:
                        branch_gt.append(bg / s_b)
                        branch_pred.append(bp / s_b)

    abs_gt, abs_pred = [], []
    mode_gt, mode_pred = [], []
    case_scales: List[float] = []
    scale_case_names: List[str] = []

    for case in sorted(set(case_gt) & set(case_pred)):
        gt_all = np.concatenate(case_gt[case])
        pred_all = np.concatenate(case_pred[case])
        s_case = _case_scale(gt_all)
        if not math.isfinite(s_case) or s_case <= 1e-12:
            continue
        abs_gt.append(gt_all)
        abs_pred.append(pred_all)
        mode_gt.append(gt_all / s_case)
        mode_pred.append(pred_all / s_case)
        case_scales.append(s_case)
        scale_case_names.append(case)

    if not abs_gt:
        return {"error": "no_case_data", "verdict": "no_go"}

    y_abs = np.concatenate(abs_gt)
    p_abs = np.concatenate(abs_pred)
    y_mode = np.concatenate(mode_gt)
    p_mode = np.concatenate(mode_pred)
    abs_r2 = float(_r2_score(y_abs, p_abs))
    mode_r2 = float(_r2_score(y_mode, p_mode))

    branch_mode_r2 = float("nan")
    if branch_gt:
        yb = np.concatenate(branch_gt)
        pb = np.concatenate(branch_pred)
        if yb.size == pb.size and yb.size > 100:
            branch_mode_r2 = float(_r2_score(yb, pb))

    # optimal per-case scale calibration (oracle upper bound on scale fix)
    cal_r2_list: List[float] = []
    for case in sorted(case_gt):
        gt_all = np.concatenate(case_gt[case])
        pred_all = np.concatenate(case_pred[case])
        denom = float(np.sum(pred_all ** 2)) + 1e-12
        alpha = float(np.sum(gt_all * pred_all) / denom)
        cal_r2_list.append(float(_r2_score(gt_all, alpha * pred_all)))
    cal_r2_mean = float(np.mean(cal_r2_list)) if cal_r2_list else float("nan")
    scale_gain = cal_r2_mean - abs_r2

    from sklearn.ensemble import HistGradientBoostingRegressor

    scale_r2 = float("nan")
    if len(case_scales) >= 6:
        feats, scales = [], []
        for case, s in zip(scale_case_names, case_scales):
            if case not in case_bc:
                continue
            bc_i, bc_o, mr = case_bc[case]
            feats.append([bc_i, bc_o, mr, math.log(max(s, 1e-12))])
            scales.append(math.log(max(s, 1e-12)))
        if len(feats) >= 6:
            Xs = np.array(feats, dtype=np.float64)
            ys = np.array(scales, dtype=np.float64)
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(Xs))
            cut = max(3, int(len(idx) * 0.7))
            tr, te = idx[:cut], idx[cut:]
            m = HistGradientBoostingRegressor(max_depth=3, random_state=seed)
            m.fit(Xs[tr], ys[tr])
            scale_r2 = float(_r2_score(ys[te], m.predict(Xs[te]))) if te.size else float("nan")

    delta_mode_abs = mode_r2 - abs_r2
    go_mode = delta_mode_abs >= 0.05 and mode_r2 > abs_r2
    go_scale = scale_gain >= 0.03 and (scale_r2 if math.isfinite(scale_r2) else -1) >= 0.5
    go = go_mode or go_scale

    per_case_rows: List[Dict[str, Any]] = []
    for case in sorted(case_gt):
        gt_all = np.concatenate(case_gt[case])
        pred_all = np.concatenate(case_pred[case])
        s = _case_scale(gt_all)
        per_case_rows.append({
            "case": case,
            "abs_r2": _safe_json(float(_r2_score(gt_all, pred_all))),
            "mode_r2": _safe_json(float(_r2_score(gt_all / s, pred_all / s))) if s > 1e-12 else None,
            "calibrated_r2": _safe_json(float(_r2_score(gt_all, pred_all * float(np.sum(gt_all * pred_all) / (np.sum(pred_all ** 2) + 1e-12))))),
        })

    return {
        "probe": "K3",
        "label": "mode_scale_target_audit",
        "context": "V3P · I6-diag absolute vs descaled mode R2 + scale calibration bound",
        "reference_run": str(manifest_path.parent.parent),
        "abs_r2_pooled": _safe_json(abs_r2),
        "mode_r2_per_case_descale": _safe_json(mode_r2),
        "branch_mode_r2": _safe_json(branch_mode_r2),
        "delta_mode_minus_abs": _safe_json(delta_mode_abs),
        "calibrated_r2_mean": _safe_json(cal_r2_mean),
        "scale_calibration_gain": _safe_json(scale_gain),
        "s_case_log_r2_bc_geom": _safe_json(scale_r2),
        "per_case": per_case_rows,
        "go_thresholds": {
            "delta_mode_minus_abs_min": 0.05,
            "scale_calibration_gain_min": 0.03,
            "s_case_r2_min": 0.50,
        },
        "layers": {
            "L1": {"metric": "mode_r2", "value": _safe_json(mode_r2)},
            "L2": {"metric": "scale_calibration_gain", "value": _safe_json(scale_gain)},
            "L3": {"metric": "chen_mode_r2", "value": next(
                (r["mode_r2"] for r in per_case_rows if "CHEN" in r["case"]), None
            )},
        },
        "verdict": _verdict(go, weak=go_mode != go_scale and (go_mode or go_scale)),
        "gpu_mapping": "V3P-K3-ModeScaleHotspot_seed1",
        "recommendation": (
            "Go → K3 mode-scale-hotspot 双头；No-Go → 误差非幅值/模式可拆分主因"
        ),
    }


def probe_k5_temporal_bc(
    manifest_path: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    seed: int,
) -> Dict[str, Any]:
    """K5 · temporal/BC error autocorrelation。"""
    from sklearn.ensemble import HistGradientBoostingRegressor

    case_frames: Dict[str, List[Tuple[int, float, float]]] = {}
    case_bc: Dict[str, Tuple[float, float]] = {}
    case_pace: Dict[str, str] = {}
    case_err_acc: Dict[str, List[float]] = {}

    for row in _iter_prediction_rows(manifest_path, stats):
        case = row["case"]
        case_pace[case] = row["pace"]
        case_bc[case] = (row["bc_inlet"], row["bc_o1"])
        case_err_acc.setdefault(case, []).append(float(np.mean(row["abs_err"])))
        if row["frame_id"] is None:
            continue
        mean_err = float(np.mean(row["abs_err"]))
        case_frames.setdefault(case, []).append((row["frame_id"], mean_err, row["t_norm"]))

    lag1_vals: List[float] = []
    case_autocorr: Dict[str, float] = {}
    for case, rows in case_frames.items():
        rows.sort(key=lambda t: t[0])
        errs = np.array([r[1] for r in rows], dtype=np.float64)
        if errs.size < 8:
            continue
        # 去病例均值后再算 lag-1，避免平滑误差场带来的伪高自相关
        errs = errs - float(np.mean(errs))
        e0 = errs[:-1]
        e1 = errs[1:]
        if float(np.std(e0)) < 1e-12 or float(np.std(e1)) < 1e-12:
            continue
        r = float(np.corrcoef(e0, e1)[0, 1])
        if math.isfinite(r):
            lag1_vals.append(r)
            case_autocorr[case] = r

    fast_ac = [case_autocorr[c] for c in case_autocorr if case_pace.get(c) == "fast"]
    slow_ac = [case_autocorr[c] for c in case_autocorr if case_pace.get(c) == "slow"]
    chen_ac = case_autocorr.get("fast/CHEN_SHI_MING")

    case_mean_err: List[float] = []
    case_feats: List[List[float]] = []
    for case in sorted(case_bc):
        errs = case_err_acc.get(case)
        if not errs:
            continue
        bc_i, bc_o = case_bc[case]
        case_mean_err.append(float(np.mean(errs)))
        case_feats.append([bc_i, bc_o, 1.0 if case_pace.get(case) == "fast" else 0.0])

    bc_explained = float("nan")
    if len(case_feats) >= 6:
        X = np.array(case_feats, dtype=np.float64)
        y = np.array(case_mean_err, dtype=np.float64)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        cut = max(3, int(len(idx) * 0.7))
        tr, te = idx[:cut], idx[cut:]
        m = HistGradientBoostingRegressor(max_depth=3, random_state=seed)
        m.fit(X[tr], y[tr])
        if te.size:
            bc_explained = _explained_frac(y[te], m.predict(X[te]))

    mean_lag1 = float(np.mean(lag1_vals)) if lag1_vals else float("nan")
    fast_slow_delta = (
        float(np.mean(fast_ac) - np.mean(slow_ac)) if fast_ac and slow_ac else None
    )
    # 高 lag1  alone 不足以 Go（平滑误差场天然高相关）；需 BC 可解释或 fast/slow 分层
    go_temporal = (
        math.isfinite(mean_lag1)
        and abs(mean_lag1) >= 0.25
        and fast_slow_delta is not None
        and abs(fast_slow_delta) >= 0.01
    )
    go_bc = math.isfinite(bc_explained) and bc_explained >= 0.30
    go = go_bc or go_temporal

    return {
        "probe": "K5",
        "label": "temporal_bc_error_autocorrelation",
        "context": "V3P · I6-diag per-frame error autocorr + BC case-level explained",
        "reference_run": str(manifest_path.parent.parent),
        "n_cases_with_frames": len(case_frames),
        "mean_lag1_autocorr": _safe_json(mean_lag1),
        "fast_mean_lag1": _safe_json(float(np.mean(fast_ac)) if fast_ac else None),
        "slow_mean_lag1": _safe_json(float(np.mean(slow_ac)) if slow_ac else None),
        "chen_lag1": _safe_json(chen_ac),
        "bc_case_error_explained_frac": _safe_json(bc_explained),
        "case_autocorr": {k: _safe_json(v) for k, v in sorted(case_autocorr.items())},
        "go_thresholds": {
            "mean_lag1_abs_min": 0.25,
            "bc_explained_min": 0.30,
        },
        "layers": {
            "L1": {"metric": "mean_lag1_autocorr", "value": _safe_json(mean_lag1)},
            "L2": {"metric": "bc_explained_frac", "value": _safe_json(bc_explained)},
            "L3": {"metric": "fast_minus_slow_lag1", "value": _safe_json(
                float(np.mean(fast_ac) - np.mean(slow_ac)) if fast_ac and slow_ac else None
            )},
        },
        "verdict": _verdict(go, weak=go_temporal != go_bc and (go_temporal or go_bc)),
        "gpu_mapping": "V3P-K5-TemporalFiLM_seed1",
        "recommendation": (
            "Go → K5 temporal/BC FiLM 短 GPU probe；No-Go → 帧间/BC 非主误差来源"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P 后平台期 K1/K3/K5 CPU 短探针")
    ap.add_argument("--reference-run", type=Path, default=DEFAULT_I6_RUN)
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/field/f0_decision")
    ap.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    ap.add_argument("--only", default="", help="K1,K3,K5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="仅 CHEN + GUO 两例")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    run_dir = args.reference_run.resolve()
    manifest = run_dir / "predictions_test_best_wss" / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"缺少 I6-diag predictions: {manifest}")

    stats = _load_norm_stats(args.norm_params.resolve())
    out_dir = ensure_dir(args.output_dir.resolve())
    suffix = args.date

    only = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    active = [p for p in ALL_PROBES if not only or p in only]

    summary: Dict[str, Any] = {
        "context": "V3P post-plateau K probes",
        "reference_run": str(run_dir),
        "probes": {},
        "elapsed_sec": None,
    }

    if args.smoke:
        # monkey-patch via temp manifest filter — handled in probes by limiting cases
        orig_iter = _iter_prediction_rows

        def _smoke_iter(mpath, st, *, cases=None):
            smoke_cases = {"fast/CHEN_SHI_MING", "slow/GUO_XI_JIANG"}
            want = smoke_cases if cases is None else (cases & smoke_cases)
            yield from orig_iter(mpath, st, cases=want or smoke_cases)

        globals()["_iter_prediction_rows"] = _smoke_iter

    for probe in active:
        if args.verbose:
            print(f"[post-plateau] running {probe} ...", flush=True)
        t_probe = time.perf_counter()
        if probe == "K1":
            res = probe_k1_branch_error(manifest, stats, seed=args.seed)
            path = out_dir / f"v3p_k1_branch_error_probe_{suffix}.json"
        elif probe == "K3":
            res = probe_k3_mode_scale_audit(manifest, stats, seed=args.seed)
            path = out_dir / f"v3p_k3_mode_scale_audit_{suffix}.json"
        elif probe == "K5":
            res = probe_k5_temporal_bc(manifest, stats, seed=args.seed)
            path = out_dir / f"v3p_k5_temporal_bc_probe_{suffix}.json"
        else:
            continue
        save_json(path, res)
        summary["probes"][probe] = {
            "path": str(path),
            "verdict": res.get("verdict"),
            "elapsed_sec": round(time.perf_counter() - t_probe, 2),
        }
        if args.verbose:
            print(f"  -> {path.name} verdict={res.get('verdict')} ({summary['probes'][probe]['elapsed_sec']}s)", flush=True)

    summary["elapsed_sec"] = round(time.perf_counter() - t0, 2)
    summary_path = out_dir / f"v3p_post_plateau_probes_summary_{suffix}.json"
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
