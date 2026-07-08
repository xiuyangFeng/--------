#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WSS 全局标准化统计（第二遍）。

坐标逐病例、WSS 全局 —— 这里在所有 bundle 上流式累积壁面 WSS 的统计量：
- 线性 z：mean/std of wss
- 对数 z：mean/std of log(wss + eps)（WSS 近似对数正态，更稳）
统计覆盖全部时间步的壁面点（与后续可能预测全相位一致）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config as C

EPS = 1e-6


def _iter_bundles(cohorts: List[str]):
    for cohort in cohorts:
        for case in C.list_cases(cohort):
            p = C.out_case_dir(cohort, case) / "bundle.npz"
            if p.is_file():
                yield cohort, case, p


def compute_global_wss_stats(cohorts: List[str] | None = None,
                             cfg: C.PipelineConfig | None = None) -> Dict:
    cfg = cfg or C.DEFAULT
    cohorts = cohorts or list(C.COHORTS.values())

    n = 0
    s_lin = ss_lin = 0.0
    s_log = ss_log = 0.0
    vmin, vmax = np.inf, -np.inf
    cases = 0
    for cohort, case, p in _iter_bundles(cohorts):
        with np.load(p, allow_pickle=True) as d:
            wss = d["wall_wss"].astype(np.float64).ravel()
        wss = wss[np.isfinite(wss)]
        if wss.size == 0:
            continue
        cases += 1
        n += wss.size
        s_lin += wss.sum(); ss_lin += (wss * wss).sum()
        lg = np.log(np.clip(wss, 0, None) + EPS)
        s_log += lg.sum(); ss_log += (lg * lg).sum()
        vmin = min(vmin, float(wss.min())); vmax = max(vmax, float(wss.max()))

    if n == 0:
        raise RuntimeError("no bundles found for global WSS stats — run preprocess first")

    mean_lin = s_lin / n
    std_lin = float(np.sqrt(max(ss_lin / n - mean_lin ** 2, 1e-12)))
    mean_log = s_log / n
    std_log = float(np.sqrt(max(ss_log / n - mean_log ** 2, 1e-12)))

    stats = {
        "method": cfg.normalization.wss_method,
        "eps": EPS,
        "n_points": int(n),
        "n_cases": cases,
        "linear": {"mean": mean_lin, "std": std_lin},
        "log": {"mean": mean_log, "std": std_log},
        "raw_min": vmin, "raw_max": vmax,
        "cohorts": cohorts,
    }
    out = C.OUT_ROOT / cfg.normalization.global_stats_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    from . import reporting
    reporting.get_logger().info(
        "[global_stats] %d cases, %d wall-pts, method=%s log(mean=%.4f std=%.4f) -> %s",
        cases, n, stats["method"], mean_log, std_log, out)
    return stats


def load_global_wss_stats(cfg: C.PipelineConfig | None = None) -> Dict:
    cfg = cfg or C.DEFAULT
    return json.loads((C.OUT_ROOT / cfg.normalization.global_stats_name).read_text())


def normalize_wss(wss: np.ndarray, stats: Dict, method: str | None = None) -> np.ndarray:
    method = method or stats["method"]
    if method == "log_z":
        lg = np.log(np.clip(wss, 0, None) + stats["eps"])
        return (lg - stats["log"]["mean"]) / stats["log"]["std"]
    return (wss - stats["linear"]["mean"]) / stats["linear"]["std"]
