#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稀疏化 + 样本装配（配置驱动，最后一步）。

严格在 归一化之后 才稀疏化。改 SampleConfig 就能扫点数/时间步/输入特征，
不用重跑上游预处理。

每个样本落地为 .npz：
- coords (N,3) 归一化坐标
- geom (N,G) 保留的几何特征 + geom_names（后续用 input_mask 屏蔽）
- y (N,) 全局标准化后的 WSS 目标；y_raw (N,) 原始 WSS
- point_type (N,)
- input_feature_names / target：网络实际吃什么由配置决定
默认：输入 x,y,z；输出 wss；壁面 2000 点；峰值收缩期单样本。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import config as C
from . import global_stats as GS
from . import reporting


# ---------------------------------------------------------------------------
# 稀疏化
# ---------------------------------------------------------------------------
def farthest_point_sample(pts: np.ndarray, k: int, seed: int) -> np.ndarray:
    """最远点采样，返回被选点的索引 (k,)。"""
    n = len(pts)
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    sel = np.empty(k, dtype=np.int64)
    sel[0] = rng.integers(n)
    d = np.linalg.norm(pts - pts[sel[0]], axis=1)
    for i in range(1, k):
        sel[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(pts - pts[sel[i]], axis=1))
    return sel


def _sample_indices(pts: np.ndarray, k: int, method: str, seed: int) -> np.ndarray:
    if k <= 0 or k >= len(pts):
        return np.arange(len(pts))
    if method == "random":
        return np.random.default_rng(seed).choice(len(pts), size=k, replace=False)
    return farthest_point_sample(pts, k, seed)


# ---------------------------------------------------------------------------
# 样本装配
# ---------------------------------------------------------------------------
GEOM_KEYS = ["dist_to_wall", "abscissa_norm", "local_radius", "curvature"]


def _select_steps(bundle, sample_cfg: C.SampleConfig) -> List[int]:
    steps = list(bundle["steps"].tolist())
    if sample_cfg.timesteps == "peak":
        return [int(bundle["peak_step"])]
    if sample_cfg.timesteps == "all":
        return steps
    # 允许直接给步列表
    if isinstance(sample_cfg.timesteps, (list, tuple)):
        return [s for s in sample_cfg.timesteps if s in steps]
    return [int(bundle["peak_step"])]


def build_case_samples(cohort_rel: str, case_name: str, stats: Dict,
                       cfg: C.PipelineConfig | None = None) -> List[Path]:
    cfg = cfg or C.DEFAULT
    scfg = cfg.sample
    bundle_path = C.out_case_dir(cohort_rel, case_name) / "bundle.npz"
    with np.load(bundle_path, allow_pickle=True) as d:
        bundle = {k: d[k] for k in d.files}

    wall_coords = bundle["wall_coords_norm"]          # (Nw,3)
    steps_all = list(bundle["steps"].tolist())
    idx = _sample_indices(wall_coords, scfg.wall_n_points, scfg.wall_sampling, scfg.seed)

    # 稀疏后的静态几何
    coords = wall_coords[idx]
    geom_cols = {
        "dist_to_wall": bundle["wall_dist_to_wall"][idx],
        "abscissa_norm": bundle["wall_abscissa_norm"][idx],
        "local_radius": bundle["wall_local_radius"][idx],
        "curvature": bundle["wall_curvature"][idx],
    }
    geom = np.stack([geom_cols[k] for k in GEOM_KEYS], axis=1).astype(np.float32)

    out_dir = C.OUT_ROOT / "samples" / scfg.name / cohort_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for step in _select_steps(bundle, scfg):
        si = steps_all.index(step)
        wss_raw = bundle["wall_wss"][si][idx].astype(np.float32)
        y = GS.normalize_wss(wss_raw, stats).astype(np.float32)

        payload = dict(
            case=case_name, cohort=cohort_rel, step=np.int32(step),
            is_peak=np.bool_(step == int(bundle["peak_step"])),
            coords=coords.astype(np.float32),
            geom=geom, geom_names=np.array(GEOM_KEYS),
            point_type=np.full(len(idx), 2, dtype=np.int8),   # 全壁面点
            y=y, y_raw=wss_raw,
            input_feature_names=np.array(list(scfg.input_features)),
            target=scfg.target,
            coord_scale=bundle["coord_scale"],
        )
        # 保留但默认屏蔽的 WSS 矢量分量（如有）
        if "wall_wss_vec" in bundle:
            payload["wss_vec"] = bundle["wall_wss_vec"][si][idx].astype(np.float32)

        out_path = out_dir / f"{case_name}__{step}.npz"
        np.savez_compressed(out_path, **payload)
        written.append(out_path)
    return written


def build_all(cohorts: List[str] | None = None, cfg: C.PipelineConfig | None = None) -> None:
    cfg = cfg or C.DEFAULT
    cohorts = cohorts or list(C.COHORTS.values())
    stats = GS.load_global_wss_stats(cfg)

    log = reporting.get_logger()
    manifest = []
    n_cases = 0
    for cohort in cohorts:
        for case in C.list_cases(cohort):
            bp = C.out_case_dir(cohort, case) / "bundle.npz"
            if not bp.is_file():
                continue
            paths = build_case_samples(cohort, case, stats, cfg)
            manifest.extend(str(p.relative_to(C.OUT_ROOT)) for p in paths)
            n_cases += 1

    man_dir = C.OUT_ROOT / "samples" / cfg.sample.name
    man_dir.mkdir(parents=True, exist_ok=True)
    (man_dir / "manifest.json").write_text(json.dumps({
        "sample_name": cfg.sample.name,
        "wall_n_points": cfg.sample.wall_n_points,
        "timesteps": cfg.sample.timesteps,
        "input_features": list(cfg.sample.input_features),
        "target": cfg.sample.target,
        "n_cases": n_cases,
        "n_samples": len(manifest),
        "samples": manifest,
    }, indent=2))
    log.info("[build_samples] set=%s wall_n=%d timesteps=%s -> %d samples from %d cases -> %s",
             cfg.sample.name, cfg.sample.wall_n_points, cfg.sample.timesteps,
             len(manifest), n_cases, man_dir)
