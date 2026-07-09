#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估指标（train 监控 + evaluate 复用）。

指标默认在**原始 WSS 空间**（denormalize 之后）计算，物理可解释。
- field 级：把所有点 pool 起来算（受高 WSS 病例主导）。
- case-mean 级：逐病例算再平均（跨病例更平衡）。
分区：分叉区(近原点) / 狭窄区(小 local_radius) / 高 WSS 区。
"""

from __future__ import annotations

from typing import Dict

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


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "nrmse_range": nrmse(y_true, y_pred, "range"),
        "nrmse_mean": nrmse(y_true, y_pred, "mean"),
        "mae": mae(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "n": int(len(y_true)),
    }


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
    maes = [v["overall"]["mae"] for v in per_case.values()
            if np.isfinite(v["overall"].get("mae", np.nan))]
    return {
        "r2_casemean": float(np.mean(r2s)) if r2s else float("nan"),
        "r2_casemed": float(np.median(r2s)) if r2s else float("nan"),
        "nrmse_casemean": float(np.mean(nrmses)) if nrmses else float("nan"),
        "mae_casemean": float(np.mean(maes)) if maes else float("nan"),
        "n_cases": len(r2s),
    }
