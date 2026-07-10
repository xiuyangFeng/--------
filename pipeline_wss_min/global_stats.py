#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WSS 全局标准化统计（第二遍）。

坐标逐病例、WSS 全局 —— 这里在 split 的 train bundle 上流式累积壁面 WSS 的统计量：
- 线性 z：mean/std of wss
- 对数 z：mean/std of log(wss + eps)（WSS 近似对数正态，更稳）
第三轮 clean-data 默认统计 peak 单步；全相位任务可显式切回 all。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config as C

EPS = 1e-6


def _iter_bundles(
    cohorts: List[str],
    split_name: str | None,
    partitions: tuple[str, ...],
):
    for cohort in cohorts:
        cases = (
            C.list_split_cases(cohort, split_name, partitions)
            if split_name else C.list_cases(cohort)
        )
        for case in cases:
            p = C.out_case_dir(cohort, case) / "bundle.npz"
            if p.is_file():
                yield cohort, case, p


def compute_global_wss_stats(cohorts: List[str] | None = None,
                             cfg: C.PipelineConfig | None = None,
                             split_name: str | None = C.DEFAULT_SPLIT_NAME,
                             partitions: tuple[str, ...] = ("train",),
                             timesteps_scope: str = "peak") -> Dict:
    cfg = cfg or C.DEFAULT
    cohorts = cohorts or list(C.COHORTS.values())
    if split_name is None:
        raise RuntimeError("global WSS stats 必须按 split 运行，禁止读取 all-raw bundle")
    if timesteps_scope not in {"peak", "all"}:
        raise ValueError(f"unsupported timesteps_scope={timesteps_scope!r}")

    n = 0
    n_zero = 0
    s_lin = ss_lin = 0.0
    s_log = ss_log = 0.0
    vmin, vmax = np.inf, -np.inf
    cases = 0
    quant_chunks = []
    for cohort, case, p in _iter_bundles(cohorts, split_name, partitions):
        with np.load(p, allow_pickle=True) as d:
            if timesteps_scope == "peak":
                steps = d["steps"].tolist()
                peak = int(d["peak_step"])
                si = steps.index(peak)
                wss = d["wall_wss"][si].astype(np.float64).ravel()
            else:
                wss = d["wall_wss"].astype(np.float64).ravel()
        wss = wss[np.isfinite(wss)]
        if wss.size == 0:
            continue
        cases += 1
        n += wss.size
        n_zero += int((wss <= 0).sum())
        s_lin += wss.sum(); ss_lin += (wss * wss).sum()
        lg = np.log(np.clip(wss, 0, None) + EPS)
        s_log += lg.sum(); ss_log += (lg * lg).sum()
        vmin = min(vmin, float(wss.min())); vmax = max(vmax, float(wss.max()))
        quant_chunks.append(wss.astype(np.float32, copy=False))

    if n == 0:
        raise RuntimeError("no bundles found for global WSS stats — run preprocess first")

    mean_lin = s_lin / n
    std_lin = float(np.sqrt(max(ss_lin / n - mean_lin ** 2, 1e-12)))
    mean_log = s_log / n
    std_log = float(np.sqrt(max(ss_log / n - mean_log ** 2, 1e-12)))
    q = np.concatenate(quant_chunks) if quant_chunks else np.asarray([], dtype=np.float32)
    percentiles = {
        "p50": float(np.percentile(q, 50)) if q.size else float("nan"),
        "p90": float(np.percentile(q, 90)) if q.size else float("nan"),
        "p95": float(np.percentile(q, 95)) if q.size else float("nan"),
        "p99": float(np.percentile(q, 99)) if q.size else float("nan"),
    }

    stats = {
        "method": cfg.normalization.wss_method,
        "eps": EPS,
        "n_points": int(n),
        "n_cases": cases,
        "timesteps_scope": timesteps_scope,
        "zero_frac": float(n_zero / n),
        "linear": {"mean": mean_lin, "std": std_lin},
        "log": {"mean": mean_log, "std": std_log},
        "raw_min": vmin, "raw_max": vmax,
        "raw_percentiles": percentiles,
        "cohorts": cohorts,
        "split_name": split_name,
        "partitions": list(partitions),
    }
    out = C.OUT_ROOT / cfg.normalization.global_stats_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    from . import reporting
    reporting.get_logger().info(
        "[global_stats] split=%s partitions=%s timesteps=%s %d cases, %d wall-pts, "
        "zero_frac=%.6f method=%s log(mean=%.4f std=%.4f) raw(p90=%.4f p99=%.4f max=%.4f) -> %s",
        split_name or "ALL_RAW", ",".join(partitions), timesteps_scope,
        cases, n, stats["zero_frac"], stats["method"],
        mean_log, std_log, percentiles["p90"], percentiles["p99"], vmax, out)
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
