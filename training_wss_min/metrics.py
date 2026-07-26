#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估指标（train 监控 + evaluate 复用）。

指标默认在**原始 WSS 空间**（denormalize 之后）计算，物理可解释。
- field 级：把所有点 pool 起来算（受高 WSS 病例主导）。
- case-mean 级：逐病例算再平均（跨病例更平衡）。
分区：分叉区(近原点) / 狭窄区(小 local_radius) / 高 WSS 区。
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, norm: str = "range") -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if norm == "mean":
        denom = abs(float(y_true.mean()))
    elif norm == "std":
        denom = float(y_true.std())
    else:
        denom = float(y_true.max() - y_true.min())
    return rmse / denom if denom > 1e-12 else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, np.float64) - np.asarray(y_pred, np.float64))))


def nmae(y_true: np.ndarray, y_pred: np.ndarray, norm: str = "range") -> float:
    """Normalized MAE using the same denominator choices as :func:`nrmse`."""
    y_true = np.asarray(y_true, dtype=np.float64)
    value = mae(y_true, y_pred)
    if norm == "mean":
        denom = abs(float(y_true.mean()))
    elif norm == "std":
        denom = float(y_true.std())
    else:
        denom = float(y_true.max() - y_true.min())
    return value / denom if denom > 1e-12 else float("nan")


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "nrmse_range": nrmse(y_true, y_pred, "range"),
        "nrmse_mean": nrmse(y_true, y_pred, "mean"),
        "mae": mae(y_true, y_pred),
        "nmae_range": nmae(y_true, y_pred, "range"),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "n": int(len(y_true)),
    }


def weighted_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           weights: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(w) & (w >= 0)
    yt, yp, w = yt[valid], yp[valid], w[valid]
    w = w / w.sum()
    mean = float(np.sum(w * yt))
    mse = float(np.sum(w * (yt - yp) ** 2))
    mae_value = float(np.sum(w * np.abs(yt - yp)))
    var = float(np.sum(w * (yt - mean) ** 2))
    true_range = float(yt.max() - yt.min())
    return {"r2": 1.0 - mse / var if var > 1e-12 else float("nan"),
            "mae": mae_value, "rmse": float(np.sqrt(mse)),
            "nmae_range": mae_value / true_range if true_range > 1e-12 else float("nan"),
            "n": int(len(yt)), "weight_sum": float(weights.sum())}


def area_top_fraction_mask(values: np.ndarray, weights: np.ndarray,
                           fraction: float = 0.10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")[::-1]
    cumulative = np.cumsum(weights[order] / weights.sum())
    take = cumulative <= fraction
    if len(take):
        take[0] = True
        boundary = int(np.searchsorted(cumulative, fraction, side="left"))
        take[:min(boundary + 1, len(take))] = True
    mask = np.zeros(len(values), dtype=bool)
    mask[order[take]] = True
    return mask


def area_hotspot_metrics(y_true: np.ndarray, y_pred: np.ndarray, pos: np.ndarray,
                         weights: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    xyz = np.asarray(pos, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    true_mask = area_top_fraction_mask(yt, w)
    pred_mask = area_top_fraction_mask(yp, w)
    inter = true_mask & pred_mask
    union = true_mask | pred_mask
    inter_area = float(w[inter].sum())
    union_area = float(w[union].sum())
    pred_area = float(w[pred_mask].sum())
    true_area = float(w[true_mask].sum())
    wt = w[true_mask] / w[true_mask].sum()
    wp = w[pred_mask] / w[pred_mask].sum()
    ct = np.sum(xyz[true_mask] * wt[:, None], axis=0)
    cp = np.sum(xyz[pred_mask] * wp[:, None], axis=0)
    high = weighted_basic_metrics(yt[true_mask], yp[true_mask], w[true_mask])
    return {
        "area_top10_iou": inter_area / union_area if union_area else float("nan"),
        "area_top10_precision": inter_area / pred_area if pred_area else float("nan"),
        "area_top10_recall": inter_area / true_area if true_area else float("nan"),
        "area_hotspot_centroid_dist": float(np.linalg.norm(ct - cp)),
        "high_wss_area_mae": high["mae"], "high_wss_area_rmse": high["rmse"],
        "high_wss_area_r2": high["r2"],
    }


def casebalanced_area_metrics(y_true_by_case: Sequence[np.ndarray],
                              y_pred_by_case: Sequence[np.ndarray],
                              weights_by_case: Sequence[np.ndarray]) -> Dict[str, float]:
    means = [float(np.sum(np.asarray(w) / np.sum(w) * np.asarray(y)))
             for y, w in zip(y_true_by_case, weights_by_case)]
    global_mean = float(np.mean(means))
    mse, mae_values, variance = [], [], []
    for yt, yp, w in zip(y_true_by_case, y_pred_by_case, weights_by_case):
        yt, yp, w = map(lambda x: np.asarray(x, dtype=np.float64), (yt, yp, w))
        w = w / w.sum()
        mse.append(np.sum(w * (yt - yp) ** 2))
        mae_values.append(np.sum(w * np.abs(yt - yp)))
        variance.append(np.sum(w * (yt - global_mean) ** 2))
    mse_mean, var_mean = float(np.mean(mse)), float(np.mean(variance))
    return {"r2": 1.0 - mse_mean / var_mean if var_mean > 1e-12 else float("nan"),
            "mae": float(np.mean(mae_values)), "rmse": float(np.sqrt(mse_mean)),
            "n_cases": len(mse)}


def distribution_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Distribution summary in the model target space (never display-clipped)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    finite = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[finite], yp[finite]
    if yt.size == 0:
        return {}
    out: Dict[str, float] = {}
    for q in (50, 90, 95, 99):
        out[f"true_p{q}"] = float(np.percentile(yt, q))
        out[f"pred_p{q}"] = float(np.percentile(yp, q))
    out.update({
        "true_dynamic_range": float(yt.max() - yt.min()),
        "pred_dynamic_range": float(yp.max() - yp.min()),
        "pred_below_zero_fraction": float(np.mean(yp < 0.0)),
        "pred_above_one_fraction": float(np.mean(yp > 1.0)),
        "n": int(yt.size),
    })
    out["pred_true_dynamic_range_ratio"] = (
        out["pred_dynamic_range"] / out["true_dynamic_range"]
        if out["true_dynamic_range"] > 1e-12 else float("nan")
    )
    return out


def casebalanced_field_metrics(
    y_true_by_case: Sequence[np.ndarray],
    y_pred_by_case: Sequence[np.ndarray],
) -> Dict[str, float]:
    """病例等权的全场指标。

    每个病例先获得总权重 1，再在病例内对点等权。R² 使用所有病例的
    等权全局真值均值作为中心，因此不同于逐病例 R² 的算术平均。
    """
    if len(y_true_by_case) != len(y_pred_by_case):
        raise ValueError("y_true_by_case and y_pred_by_case must have the same length")
    pairs = []
    for yt0, yp0 in zip(y_true_by_case, y_pred_by_case):
        yt = np.asarray(yt0, dtype=np.float64).reshape(-1)
        yp = np.asarray(yp0, dtype=np.float64).reshape(-1)
        if yt.shape != yp.shape:
            raise ValueError("true/pred shape mismatch within a case")
        finite = np.isfinite(yt) & np.isfinite(yp)
        if finite.any():
            pairs.append((yt[finite], yp[finite]))
    if not pairs:
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"),
                "n": 0, "n_cases": 0}

    case_true_means = np.asarray([yt.mean() for yt, _ in pairs], dtype=np.float64)
    global_true_mean = float(case_true_means.mean())
    mse_cases = np.asarray([np.mean((yt - yp) ** 2) for yt, yp in pairs], dtype=np.float64)
    mae_cases = np.asarray([np.mean(np.abs(yt - yp)) for yt, yp in pairs], dtype=np.float64)
    var_cases = np.asarray(
        [np.mean((yt - global_true_mean) ** 2) for yt, _ in pairs], dtype=np.float64
    )
    ss_res = float(mse_cases.mean())
    ss_tot = float(var_cases.mean())
    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"),
        "mae": float(mae_cases.mean()),
        "rmse": float(np.sqrt(ss_res)),
        "n": int(sum(len(yt) for yt, _ in pairs)),
        "n_cases": int(len(pairs)),
    }


def calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        high_pctl: float = 90.0) -> Dict[str, float]:
    """幅值校准指标：专门追踪 high-WSS 系统性低估/爆峰。"""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    if yt.size == 0:
        return {}
    thr = np.percentile(yt, high_pctl)
    high = yt >= thr
    top_true = float(np.mean(yt[high])) if high.any() else float("nan")
    top_pred = float(np.mean(yp[high])) if high.any() else float("nan")
    denom = float(np.var(yt))
    slope = float(np.mean((yt - yt.mean()) * (yp - yp.mean())) / denom) if denom > 1e-12 else float("nan")

    def _ratio(q: float) -> float:
        tv = float(np.percentile(yt, q))
        pv = float(np.percentile(yp, q))
        return pv / tv if abs(tv) > 1e-12 else float("nan")

    ymax = float(np.max(yt))
    pmax = float(np.max(yp))
    return {
        "top10_mean_true": top_true,
        "top10_mean_pred": top_pred,
        "top10_pred_true_ratio": top_pred / top_true if abs(top_true) > 1e-12 else float("nan"),
        "p90_pred_true_ratio": _ratio(90.0),
        "p95_pred_true_ratio": _ratio(95.0),
        "p99_pred_true_ratio": _ratio(99.0),
        "max_pred_true_ratio": pmax / ymax if abs(ymax) > 1e-12 else float("nan"),
        "calibration_slope": slope,
        "n": int(yt.size),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def hotspot_localization_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 pos: np.ndarray,
                                 high_pctl: float = 90.0) -> Dict[str, float]:
    """热点定位指标：区分“幅值低但位置对”与“热点位置错误”。"""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(pos).all(axis=1)
    yt, yp, pos = yt[m], yp[m], pos[m]
    if yt.size == 0:
        return {}
    thr_t = np.percentile(yt, high_pctl)
    thr_p = np.percentile(yp, high_pctl)
    true_mask = yt >= thr_t
    pred_mask = yp >= thr_p
    inter = float(np.logical_and(true_mask, pred_mask).sum())
    union = float(np.logical_or(true_mask, pred_mask).sum())
    n_true = float(true_mask.sum())
    n_pred = float(pred_mask.sum())
    iou = inter / union if union > 0 else float("nan")
    precision = inter / n_pred if n_pred > 0 else float("nan")
    recall = inter / n_true if n_true > 0 else float("nan")
    high_mae = float(np.mean(np.abs(yt[true_mask] - yp[true_mask]))) if n_true > 0 else float("nan")
    high_nrmse_mean = (
        float(np.sqrt(np.mean((yt[true_mask] - yp[true_mask]) ** 2)) / (abs(yt[true_mask].mean()) + 1e-12))
        if n_true > 0 else float("nan")
    )
    peak_t = int(np.argmax(yt))
    peak_p = int(np.argmax(yp))
    peak_dist = float(np.linalg.norm(pos[peak_t] - pos[peak_p]))
    bbox_diag = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))
    true_centroid = pos[true_mask].mean(axis=0) if n_true > 0 else np.full(3, np.nan)
    pred_centroid = pos[pred_mask].mean(axis=0) if n_pred > 0 else np.full(3, np.nan)
    centroid_dist = float(np.linalg.norm(true_centroid - pred_centroid))
    return {
        "high_wss_mae": high_mae,
        "high_wss_nrmse_mean": high_nrmse_mean,
        "top10_iou": iou,
        "top10_precision": precision,
        "top10_recall": recall,
        "spearman_all": _spearman(yt, yp),
        "spearman_high_wss": _spearman(yt[true_mask], yp[true_mask]) if n_true >= 3 else float("nan"),
        "peak_point_dist": peak_dist,
        "bbox_diag": bbox_diag,
        "peak_point_dist_over_bbox": peak_dist / bbox_diag if bbox_diag > 1e-12 else float("nan"),
        "hotspot_centroid_dist": centroid_dist,
        "hotspot_centroid_dist_over_bbox": (
            centroid_dist / bbox_diag if bbox_diag > 1e-12 else float("nan")
        ),
        "n_high_true": int(n_true),
        "n_high_pred": int(n_pred),
    }


def aggregate_hotspot_metrics(per_case: Dict[str, Dict]) -> Dict[str, float]:
    """Aggregate only per-case hotspot metrics; never concatenate case coordinates."""
    keys = (
        "top10_iou", "top10_precision", "top10_recall", "spearman_all",
        "spearman_high_wss", "peak_point_dist_over_bbox",
        "hotspot_centroid_dist_over_bbox",
    )
    out: Dict[str, float] = {"n_cases": int(len(per_case))}
    for key in keys:
        vals = np.asarray([
            reg.get("hotspot", {}).get(key, np.nan) for reg in per_case.values()
        ], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[f"{key}_casemean"] = float(vals.mean()) if vals.size else float("nan")
        out[f"{key}_casemed"] = float(np.median(vals)) if vals.size else float("nan")
    # Selection code historically reads the un-suffixed top10 value.
    out["top10_iou"] = out["top10_iou_casemean"]
    out["top10_precision"] = out["top10_precision_casemean"]
    out["top10_recall"] = out["top10_recall_casemean"]
    return out


def selection_score_r4_composite_v1(agg: Dict, field: Dict, cal: Dict,
                                    hotspot: Dict | None = None) -> float:
    """预先写死的第四轮复合选模分数（越大越好）。

    主项：0.6 * R²_casemean + 0.4 * R²_field
    惩罚：top10/p99 欠估或 max ratio 爆峰；不在看完 test 后改权重。
    """
    r2c = float(agg.get("r2_casemean", float("nan")))
    r2f = float(field.get("r2", float("nan")))
    if not (np.isfinite(r2c) and np.isfinite(r2f)):
        return float("-inf")
    score = 0.6 * r2c + 0.4 * r2f
    top10 = float(cal.get("top10_pred_true_ratio", float("nan")))
    p99 = float(cal.get("p99_pred_true_ratio", float("nan")))
    maxr = float(cal.get("max_pred_true_ratio", float("nan")))
    if np.isfinite(top10):
        # 目标约 0.5–1.0；过低惩罚，过高（>1.2）也惩罚
        score -= 0.15 * max(0.0, 0.45 - top10)
        score -= 0.10 * max(0.0, top10 - 1.20)
    if np.isfinite(p99):
        score -= 0.10 * max(0.0, 0.40 - p99)
    if np.isfinite(maxr) and maxr > 1.5:
        score -= 0.50 * (maxr - 1.5)
    if hotspot:
        iou = float(hotspot.get("top10_iou", float("nan")))
        if np.isfinite(iou):
            score += 0.05 * iou
    return float(score)


# ---------------------------------------------------------------------------
# 分区 mask（在完整点云上算，不依赖采样）
# ---------------------------------------------------------------------------
def region_masks(pos: np.ndarray, local_radius: np.ndarray, y_true_raw: np.ndarray,
                 bif_radius: float = 0.25, narrow_pctl: float = 20.0,
                 high_wss_pctl: float = 90.0) -> Dict[str, np.ndarray]:
    d0 = np.linalg.norm(pos, axis=1)              # 距原点(=flow divider 分叉)
    lr = np.asarray(local_radius, dtype=np.float64)
    yt = np.asarray(y_true_raw, dtype=np.float64)
    return {
        "bifurcation": d0 <= bif_radius,
        "stenosis": lr <= np.nanpercentile(lr, narrow_pctl),
        "high_wss": yt >= np.nanpercentile(yt, high_wss_pctl),
    }


def regional_metrics(pos, local_radius, y_true_raw, y_pred_raw, **kw) -> Dict[str, Dict]:
    masks = region_masks(pos, local_radius, y_true_raw, **kw)
    out = {"overall": basic_metrics(y_true_raw, y_pred_raw)}
    for name, m in masks.items():
        if m.sum() >= 5:
            out[name] = basic_metrics(y_true_raw[m], y_pred_raw[m])
        else:
            out[name] = {"r2": float("nan"), "n": int(m.sum())}
    return out


def aggregate_case_metrics(per_case: Dict[str, Dict]) -> Dict[str, float]:
    """把逐病例 overall 指标聚成 field(pooled) 与 case-mean。"""
    r2s = [v["overall"]["r2"] for v in per_case.values()
           if np.isfinite(v["overall"].get("r2", np.nan))]
    nrmses = [v["overall"]["nrmse_range"] for v in per_case.values()
              if np.isfinite(v["overall"].get("nrmse_range", np.nan))]
    nmaes = [v["overall"]["nmae_range"] for v in per_case.values()
             if np.isfinite(v["overall"].get("nmae_range", np.nan))]
    maes = [v["overall"]["mae"] for v in per_case.values()
            if np.isfinite(v["overall"].get("mae", np.nan))]
    rmses = [v["overall"]["rmse"] for v in per_case.values()
             if np.isfinite(v["overall"].get("rmse", np.nan))]
    return {
        "r2_casemean": float(np.mean(r2s)) if r2s else float("nan"),
        "r2_casemed": float(np.median(r2s)) if r2s else float("nan"),
        "r2_casep10": float(np.percentile(r2s, 10.0)) if r2s else float("nan"),
        "r2_negative_cases": int(np.sum(np.asarray(r2s) < 0.0)) if r2s else 0,
        "r2_failure_rate": float(np.mean(np.asarray(r2s) < 0.0)) if r2s else float("nan"),
        "nrmse_casemean": float(np.mean(nrmses)) if nrmses else float("nan"),
        "nmae_casemean": float(np.mean(nmaes)) if nmaes else float("nan"),
        "mae_casemean": float(np.mean(maes)) if maes else float("nan"),
        "rmse_casemean": float(np.mean(rmses)) if rmses else float("nan"),
        "n_cases": len(r2s),
    }
