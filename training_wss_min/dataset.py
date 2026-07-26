#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集：直接读 pipeline_wss_min 的 bundle.npz（不经 build_samples）。

为什么直连 bundle 而不是预生成 sample npz：
- 完整点云评估（A 路）需要全部壁面点，sample npz 只有稀疏子集；
- 点数扫描/每 epoch 重采样都只是改配置，直连 bundle 最灵活；
- bundle 是唯一真源，训练与评估共用同一份数据，杜绝口径漂移。

每个病例只加载壁面数组（丢弃巨大的 int_coords，省内存）。坐标已是逐病例
归一化到 [-1,1]（保形）；WSS 用全局 log_z 统计标准化（复用 pipeline 的 stats）。
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C
from . import surface as S


# ---------------------------------------------------------------------------
# 全局 WSS 统计 / 归一化（与 pipeline_wss_min.global_stats 对齐）
# ---------------------------------------------------------------------------
def load_wss_stats(path: str | Path = C.GLOBAL_STATS) -> Dict:
    return json.loads(Path(path).read_text())


def normalize_wss(wss: np.ndarray, stats: Dict) -> np.ndarray:
    if stats["method"] == "log_z":
        lg = np.log(np.clip(wss, 0, None) + stats["eps"])
        return (lg - stats["log"]["mean"]) / stats["log"]["std"]
    return (wss - stats["linear"]["mean"]) / stats["linear"]["std"]


def denormalize_wss(y: np.ndarray, stats: Dict) -> np.ndarray:
    """标准化空间 -> 原始 WSS（mm 物理量级），评估在原始空间算指标。"""
    if stats["method"] == "log_z":
        lg = y * stats["log"]["std"] + stats["log"]["mean"]
        return np.exp(lg) - stats["eps"]
    return y * stats["linear"]["std"] + stats["linear"]["mean"]


def normalize_target(wss: np.ndarray, stats: Dict, mode: str = "global_stats") -> Tuple[np.ndarray, Dict]:
    """Normalize a complete case target and return auditable per-case metadata."""
    wss = np.asarray(wss)
    if mode == "global_stats":
        return normalize_wss(wss, stats), {
            "mode": mode,
            "stats_source": "train-only global statistics",
        }
    if mode != "case_max":
        raise ValueError(f"unsupported target normalization mode: {mode!r}")
    case_max = float(np.nanmax(wss)) if wss.size else float("nan")
    if not np.isfinite(case_max) or case_max <= 1e-12:
        raise ValueError(f"case_max normalization requires finite WSSmax > 1e-12, got {case_max}")
    return wss / case_max, {
        "mode": mode,
        "case_wss_max": case_max,
        "stats_source": "this case's complete peak-step ground-truth wall field",
        "used_as_model_input": False,
        "physical_recovery_enabled": False,
    }


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def load_split(split_path: str | Path) -> Dict:
    sp = json.loads(Path(split_path).read_text())
    seen: Dict[str, str] = {}
    for part in ("train", "val", "test"):
        for label in sp.get(f"{part}_cases", []):
            canonical = canonical_unit_id(label)
            if canonical in seen:
                raise ValueError(
                    f"split partition leakage/duplicate: {canonical} in {seen[canonical]} and {part}"
                )
            seen[canonical] = part
    return sp


def canonical_unit_id(label: str) -> str:
    parts = str(label).strip().split("/")
    if len(parts) == 2 and parts[0] in {"fast", "slow"} and parts[1]:
        return f"AG/{parts[0]}/{parts[1]}"
    if len(parts) == 3 and parts[0] in {"AG", "AAA"} and parts[1] and parts[2]:
        if parts[0] == "AG" and parts[1] not in {"fast", "slow"}:
            raise ValueError(f"invalid AG subset in split ID: {label!r}")
        if parts[0] == "AAA" and parts[1] not in {"ruputer", "unruputer"}:
            raise ValueError(f"invalid AAA subset in split ID: {label!r}")
        return "/".join(parts)
    if len(parts) == 3 and parts[0] == "ILO" and parts[1] and parts[2]:
        if not re.fullmatch(r".+-(?:0|1)", parts[1]):
            raise ValueError(f"invalid ILO patient suffix in split ID: {label!r}")
        if parts[2] != "before":
            raise ValueError(f"active ILO split only permits before, got: {label!r}")
        return "/".join(parts)
    raise ValueError(
        "invalid split case ID (expect legacy AG or canonical AG/AAA/ILO-before): "
        f"{label!r}"
    )


def load_split_cases(split_path: str | Path, partition: str) -> List[Tuple[str, str]]:
    """返回 [(cohort_rel, case_name)]，cohort_rel 形如 'AG/fast'。"""
    sp = load_split(split_path)
    key = partition if partition.endswith("_cases") else f"{partition}_cases"
    out: List[Tuple[str, str]] = []
    for label in sp.get(key, []):
        canonical = canonical_unit_id(label)
        cohort, subset, case = canonical.split("/", 2)
        out.append((f"{cohort}/{subset}", case))
    return out


def expected_partition_count(split_path: str | Path, partition: str) -> Optional[int]:
    """若 split 写入 expected_counts，返回该分区期望病例数。"""
    sp = load_split(split_path)
    ec = sp.get("expected_counts") or {}
    key = partition if not partition.endswith("_cases") else partition.replace("_cases", "")
    if key in ec:
        return int(ec[key])
    cases = sp.get(f"{key}_cases", sp.get(partition, []))
    return len(cases) if cases is not None else None


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def farthest_point_sample(pts: np.ndarray, k: int, seed: int) -> np.ndarray:
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


def build_fps_pool(pts: np.ndarray, k: int, pool_size: int, base_seed: int) -> np.ndarray:
    """预计算 S 个不同起点的 FPS-k 子集，形状 (S, k)。"""
    n = len(pts)
    k_eff = min(n, max(k, 1))
    pool = np.empty((pool_size, k_eff), dtype=np.int64)
    for s in range(pool_size):
        pool[s] = farthest_point_sample(pts, k_eff, base_seed + 104729 * s + 17)
    return pool


def fps_cache_path(case: Dict, cfg: C.DataConfig, k: int) -> Path | None:
    if not cfg.fps_cache_dir:
        return None
    unit_id = str(case.get("unit_id") or f"{case.get('cohort')}/{case.get('case')}")
    return Path(cfg.fps_cache_dir) / unit_id / f"fps_k{int(k)}_pool{int(cfg.fps_pool_size)}.npz"


def _coords_sha256(pos: np.ndarray) -> str:
    arr = np.ascontiguousarray(pos, dtype=np.float32)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def load_fps_cache(case: Dict, cfg: C.DataConfig, k: int) -> Dict[str, np.ndarray] | None:
    """Load and validate one persistent FPS cache once per in-memory case."""
    path = fps_cache_path(case, cfg, k)
    if path is None:
        return None
    memory_key = f"_fps_disk_cache_{int(k)}_{int(cfg.fps_pool_size)}"
    if memory_key in case:
        return case[memory_key]
    if not path.is_file():
        if cfg.fps_cache_required:
            raise FileNotFoundError(f"required FPS cache is missing: {path}")
        return None
    with np.load(path, allow_pickle=False) as payload:
        required = {"fixed", "pool", "n_total", "k", "pool_size", "coords_sha256"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"FPS cache missing fields {missing}: {path}")
        expected_hash = _coords_sha256(case["pos"])
        actual_hash = str(np.asarray(payload["coords_sha256"]).item())
        if actual_hash != expected_hash:
            raise ValueError(f"FPS cache coordinate hash mismatch: {path}")
        n_total = int(np.asarray(payload["n_total"]).item())
        stored_k = int(np.asarray(payload["k"]).item())
        pool_size = int(np.asarray(payload["pool_size"]).item())
        if (n_total, stored_k, pool_size) != (len(case["pos"]), min(len(case["pos"]), int(k)), int(cfg.fps_pool_size)):
            raise ValueError(
                f"FPS cache contract mismatch: {path}; got {(n_total, stored_k, pool_size)}"
            )
        cached = {
            "fixed": payload["fixed"].astype(np.int64, copy=True),
            "pool": payload["pool"].astype(np.int64, copy=True),
        }
    if cached["fixed"].shape != (stored_k,) or cached["pool"].shape != (pool_size, stored_k):
        raise ValueError(f"FPS cache array shape mismatch: {path}")
    case[memory_key] = cached
    return cached


def _norm01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo, hi = np.nanpercentile(finite, 2), np.nanpercentile(finite, 98)
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def geom_sampling_weights(case: Dict, cfg: C.DataConfig) -> np.ndarray:
    """几何加权采样权重（狭窄/高曲率/近分叉更密）。纯几何，无标签泄漏。"""
    curv = _norm01(np.abs(case["curvature"]))
    inv_r = _norm01(1.0 / np.clip(case["local_radius"], 1e-6, None))
    w = np.ones(len(case["pos"]), dtype=np.float64)
    w += cfg.geom_weight_curv * curv
    w += cfg.geom_weight_invradius * inv_r
    if cfg.geom_weight_bifurcation > 0:
        near = _norm01(-np.linalg.norm(case["pos"], axis=1))  # 近原点(分叉) -> 大
        w += cfg.geom_weight_bifurcation * near
    return w / w.sum()


def sample_indices(case: Dict, cfg: C.DataConfig, seed: int,
                   epoch: int = 0, case_index: int = 0,
                   run_seed: int = 0, *, stream: str = "default", n_points: int | None = None,
                   sampling: str | None = None) -> np.ndarray:
    n = len(case["pos"])
    k = cfg.wall_n_points if n_points is None else int(n_points)
    sampling = cfg.sampling if sampling is None else sampling
    if k <= 0 or k >= n:
        return np.arange(n)
    if sampling == "random":
        return np.random.default_rng(seed).choice(n, size=k, replace=False)
    if sampling == "area_random":
        if "surface_area_weights" not in case:
            case["surface_area_weights"], case["surface_area_report"] = S.area_weights_for_case(case)
        return np.random.default_rng(seed).choice(
            n, size=k, replace=False, p=case["surface_area_weights"]
        )
    if sampling == "geom_weighted":
        w = geom_sampling_weights(case, cfg)
        return np.random.default_rng(seed).choice(n, size=k, replace=False, p=w)
    if sampling == "fps_multistart":
        pool_size = max(1, int(cfg.fps_pool_size))
        disk = load_fps_cache(case, cfg, k)
        if disk is not None:
            pool = disk["pool"]
        elif "_fps_pool" not in case or case.get("_fps_pool_k") != min(n, k):
            case["_fps_pool"] = build_fps_pool(
                case["pos"], k, pool_size, case.get("geom_seed", run_seed)
            )
            case["_fps_pool_k"] = min(n, k)
            case["_fps_pool_size"] = pool_size
        if disk is None:
            pool = case["_fps_pool"]
        # Make the multistart choice explicit rather than relying on a hash
        # modulo collision.  With pool_size>1 this guarantees a distinct
        # support/query candidate and an epoch-varying candidate (the epoch
        # stride is deliberately coprime to the standard pool size 8).
        role_offset = {"default": 0, "support": 0, "query": 1, "eval_support": 0}.get(stream)
        if role_offset is None:
            raise ValueError(f"unsupported FPS-multistart stream {stream!r}")
        pool_id = int((run_seed + 100003 * epoch + 7919 * case_index + role_offset) % len(pool))
        return pool[pool_id].copy()
    # fps（确定性）：缓存长度-k 子集（非全序）
    disk = load_fps_cache(case, cfg, k)
    if disk is not None:
        return disk["fixed"].copy()
    cache_key = f"_fps_subset_{k}"
    if cache_key not in case or case.get("_fps_subset_k") != min(n, k):
        case[cache_key] = farthest_point_sample(case["pos"], min(n, max(k, 1)), seed)
        case["_fps_subset_k"] = min(n, k)
    return case[cache_key].copy()


# ---------------------------------------------------------------------------
# 特征标准化（几何输入列用 train 统计 z-score；x,y,z 保持归一化坐标）
# ---------------------------------------------------------------------------
GEOM_FEATURE_KEYS = ("abscissa_norm", "local_radius", "curvature", "coord_scale",
                     "radius_gradient")

# cohort one-hot（逐病例常量）：从 case["cohort"] 的父系（AG/AAA/ILO）派生 0/1，
# 不做 z-score，也不进入 feature_stats。仅当出现在 input_features 时才启用。
COHORT_FEATURE_KEYS = {"cohort_ag": "AG", "cohort_aaa": "AAA", "cohort_ilo": "ILO"}


def cohort_family(cohort_rel: str) -> str:
    """canonical cohort_rel（如 'AG/fast'、'AAA/ruputer'、'ILO/<patient>'）取父系。"""
    return str(cohort_rel).split("/", 1)[0]


def _transform_feature_values(name: str, vals: np.ndarray, transform: str | None) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float64)
    if name == "curvature" and transform == "signed_log1p":
        return np.sign(vals) * np.log1p(np.abs(vals))
    return vals


def compute_feature_stats(
    cases: List[Dict],
    input_features: Tuple[str, ...],
    curvature_transform: str = "signed_log1p",
) -> Dict:
    stats: Dict[str, Dict[str, float]] = {}
    for f in input_features:
        if f in ("x", "y", "z"):
            continue
        if f in COHORT_FEATURE_KEYS:  # cohort one-hot：0/1 常量，不做 z-score
            continue
        if f not in cases[0]:
            raise KeyError(f"feature {f!r} missing from case bundle fields")
        vals = np.concatenate([np.asarray(c[f], dtype=np.float64).ravel() for c in cases])
        transform = curvature_transform if f == "curvature" else "none"
        vals = _transform_feature_values(f, vals, transform)
        vals = vals[np.isfinite(vals)]
        if f == "curvature":  # 端点数值尖峰，先 clip 再统计
            hi = np.percentile(np.abs(vals), 99)
            vals = np.clip(vals, -hi, hi)
            stats[f] = {"mean": float(vals.mean()), "std": float(vals.std() + 1e-6),
                        "clip": float(hi), "transform": transform}
        else:
            stats[f] = {"mean": float(vals.mean()), "std": float(vals.std() + 1e-6),
                        "transform": transform}
    return stats


def compute_train_weight_quantiles(cases: List[Dict]) -> Dict[str, float]:
    """train-only 固定分位，供 target/geom loss 权重使用。"""
    y = np.concatenate([c["y_norm"].ravel() for c in cases]).astype(np.float64)
    curv = np.concatenate([np.abs(c["curvature"]).ravel() for c in cases]).astype(np.float64)
    invr = np.concatenate([
        (1.0 / np.clip(c["local_radius"], 1e-6, None)).ravel() for c in cases
    ]).astype(np.float64)
    return {
        "y_norm_q02": float(np.quantile(y, 0.02)),
        "y_norm_q98": float(np.quantile(y, 0.98)),
        "curv_q02": float(np.quantile(curv, 0.02)),
        "curv_q98": float(np.quantile(curv, 0.98)),
        "invr_q02": float(np.quantile(invr, 0.02)),
        "invr_q98": float(np.quantile(invr, 0.98)),
        "raw_p90": float(np.quantile(
            np.concatenate([c["y_raw"].ravel() for c in cases]).astype(np.float64), 0.90
        )),
    }


def random_rotation(seed: int) -> np.ndarray:
    """确定性随机 3D 旋转矩阵（det=+1）。"""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))            # 唯一化
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.astype(np.float32)


def build_features(case: Dict, idx: np.ndarray, input_features: Tuple[str, ...],
                   feat_stats: Dict, pos_override: np.ndarray | None = None) -> np.ndarray:
    pos = case["pos"][idx] if pos_override is None else pos_override
    cols = []
    for f in input_features:
        if f == "x":
            cols.append(pos[:, 0])
        elif f == "y":
            cols.append(pos[:, 1])
        elif f == "z":
            cols.append(pos[:, 2])
        elif f in COHORT_FEATURE_KEYS:
            hit = cohort_family(case["cohort"]) == COHORT_FEATURE_KEYS[f]
            cols.append(np.full(pos.shape[0], 1.0 if hit else 0.0, dtype=np.float64))
        else:
            st = feat_stats[f]
            v = np.asarray(case[f], dtype=np.float64)[idx]
            v = _transform_feature_values(f, v, st.get("transform"))
            if "clip" in st:
                v = np.clip(v, -st["clip"], st["clip"])
            v = (v - st["mean"]) / st["std"]
            cols.append(v)
    return np.stack(cols, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def _radius_gradient_from_bundle(d) -> np.ndarray:
    """若 bundle 已有 radius_gradient 则用；否则用 abscissa 排序后的稳健差分。"""
    if "wall_radius_gradient" in d.files:
        return d["wall_radius_gradient"].astype(np.float32)
    absc = d["wall_abscissa_norm"].astype(np.float64)
    lr = d["wall_local_radius"].astype(np.float64)
    order = np.argsort(absc, kind="mergesort")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    absc_s = absc[order]
    lr_s = lr[order]
    # 避免重复 abscissa 导致 np.gradient 除零：用前向差分 + 零填充
    dx = np.diff(absc_s)
    dy = np.diff(lr_s)
    g = np.zeros_like(lr_s)
    valid = np.abs(dx) > 1e-12
    g_mid = np.zeros_like(dx)
    g_mid[valid] = dy[valid] / dx[valid]
    if len(g) > 1:
        g[0] = g_mid[0] if valid[0] else 0.0
        g[-1] = g_mid[-1] if valid[-1] else 0.0
        if len(g) > 2:
            g[1:-1] = 0.5 * (g_mid[:-1] + g_mid[1:])
            # 无效段置 0
            bad = ~(valid[:-1] & valid[1:])
            g[1:-1][bad] = 0.0
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    return g[inv].astype(np.float32)


def load_case(cohort_rel: str, case_name: str, wss_stats: Dict,
              target: str = "wss", target_normalization: str = "global_stats",
              data_root: str | Path = C.DATA_ROOT,
              required_frame_version: str | None = None) -> Dict:
    p = Path(data_root) / cohort_rel / case_name / "bundle.npz"
    with np.load(p, allow_pickle=True) as d:
        if required_frame_version is not None:
            if "transform_frame_version" not in d.files:
                raise ValueError(f"missing transform_frame_version: {p}")
            actual_frame = str(np.asarray(d["transform_frame_version"]).item())
            if actual_frame != required_frame_version:
                raise ValueError(
                    f"wrong frame_version for {cohort_rel}/{case_name}: "
                    f"{actual_frame!r} != {required_frame_version!r}"
                )
        steps = d["steps"].tolist()
        peak = int(d["peak_step"])
        si = steps.index(peak)
        pos = d["wall_coords_norm"].astype(np.float32)
        if target == "wss":
            y_raw = d["wall_wss"][si].astype(np.float32)
        elif target == "pressure":
            # 壁面压力 gauge：逐例去均值（相对压力），隔离跨例 DC 偏置，
            # 使指标聚焦于几何可预测的空间压力型态（可为负）。
            p_raw = d["wall_pressure"][si].astype(np.float32)
            y_raw = (p_raw - np.float32(p_raw.mean())).astype(np.float32)
        else:
            raise ValueError(f"unsupported target={target!r} (expect 'wss'|'pressure')")
        y_norm, target_norm_meta = normalize_target(y_raw, wss_stats, target_normalization)
        case = dict(
            cohort=cohort_rel, case=case_name,
            unit_id=f"{cohort_rel}/{case_name}",
            pos=pos,
            wall_coords_raw=d["wall_coords_raw"].astype(np.float64),
            original_stl_path=(str(np.asarray(d["original_stl_path"]).item())
                               if "original_stl_path" in d.files else ""),
            original_stl_scale_to_mm=(float(d["original_stl_scale_to_mm"])
                                      if "original_stl_scale_to_mm" in d.files else float("nan")),
            original_stl_match_score=(float(d["original_stl_match_score"])
                                      if "original_stl_match_score" in d.files else float("nan")),
            wall_crop_applied=(bool(d["wall_crop_applied"])
                               if "wall_crop_applied" in d.files else False),
            wall_crop_frac=(float(d["wall_crop_frac"])
                            if "wall_crop_frac" in d.files else 0.0),
            y_raw=y_raw,
            y_norm=y_norm.astype(np.float32),
            target_normalization=target_normalization,
            target_normalization_meta=target_norm_meta,
            abscissa_norm=d["wall_abscissa_norm"].astype(np.float32),
            local_radius=d["wall_local_radius"].astype(np.float32),
            curvature=d["wall_curvature"].astype(np.float32),
            coord_scale=np.full(len(pos), float(d["coord_scale"]), dtype=np.float32),
            radius_gradient=_radius_gradient_from_bundle(d),
            peak_step=peak,
            coord_scale_scalar=float(d["coord_scale"]),
            bundle_path=str(p),
        )
    return case


def load_partition(split_path: str, partition: str, wss_stats: Dict,
                   strict: bool = True, target: str = "wss",
                   target_normalization: str = "global_stats",
                   data_root: str | Path = C.DATA_ROOT,
                   required_frame_version: str | None = None) -> List[Dict]:
    labels = load_split_cases(split_path, partition)
    cases = []
    missing = []
    for cohort_rel, case_name in labels:
        p = Path(data_root) / cohort_rel / case_name / "bundle.npz"
        if not p.is_file():
            missing.append(f"{cohort_rel}/{case_name}")
            continue
        cases.append(load_case(
            cohort_rel, case_name, wss_stats, target=target,
            target_normalization=target_normalization,
            data_root=data_root, required_frame_version=required_frame_version,
        ))
    if missing:
        msg = (f"missing {len(missing)} bundle(s) in partition={partition!r}: "
               + ", ".join(missing[:5]) + ("..." if len(missing) > 5 else ""))
        if strict:
            raise FileNotFoundError(msg)
    expected = expected_partition_count(split_path, partition)
    if strict and expected is not None and len(cases) != expected:
        raise RuntimeError(
            f"partition={partition!r} loaded {len(cases)} cases, "
            f"expected {expected} (split={split_path})"
        )
    return cases


# ---------------------------------------------------------------------------
# torch Dataset
# ---------------------------------------------------------------------------
class WSSMinDataset(Dataset):
    """训练用：每 epoch（可选）重采样壁面点。评估请直接用 case 全量点云。"""

    def __init__(self, cases: List[Dict], cfg: C.DataConfig, feat_stats: Dict,
                 training: bool = True, base_seed: int = 1234):
        self.cases = cases
        self.cfg = cfg
        self.feat_stats = feat_stats
        self.training = training
        self.base_seed = base_seed
        self.epoch = 0
        support_n = int(cfg.support_n_points or cfg.wall_n_points)
        support_sampling = cfg.support_sampling or cfg.sampling
        query_n = int(cfg.query_n_points or support_n)
        if (training and support_sampling in {"random", "area_random"} and support_n > 0
                and not getattr(cfg, "support_allow_undersized", False)):
            too_small = [
                f"{c['cohort']}/{c['case']}({len(c['pos'])})"
                for c in cases if len(c["pos"]) < support_n
            ]
            if too_small:
                raise ValueError(
                    "random sampling protocol requires the same number of points per case; "
                    f"requested {support_n}, too-small cases: {', '.join(too_small[:5])}"
                )
        if cfg.query_mode == "independent" and any(len(c["pos"]) < query_n for c in cases):
            raise ValueError(f"independent query requires at least {query_n} wall points per case")
        # 给每个 case 一个稳定的 fps seed
        for i, c in enumerate(cases):
            c.setdefault("unit_id", f"{c.get('cohort', 'AG/unknown')}/{c.get('case', i)}")
            c["geom_seed"] = base_seed + 7919 * i
            area_query_used = (
                cfg.query_mode == "independent" and cfg.query_sampling == "area_random"
            )
            if ((support_sampling == "area_random" or area_query_used)
                    and "surface_area_weights" not in c):
                c["surface_area_weights"], c["surface_area_report"] = S.area_weights_for_case(c)
        # 预热 fps_multistart pool，避免首个 epoch 在 worker 内重复计算
        if training and support_sampling == "fps_multistart":
            for i, c in enumerate(cases):
                sample_indices(c, cfg, c["geom_seed"], epoch=0, case_index=i,
                               run_seed=base_seed)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i: int):
        case = self.cases[i]
        support_n = int(self.cfg.support_n_points or self.cfg.wall_n_points)
        support_sampling = self.cfg.support_sampling or self.cfg.sampling
        query_n = int(self.cfg.query_n_points or support_n)
        query_sampling = self.cfg.query_sampling or support_sampling
        epoch = self.epoch if self.training and self.cfg.resample_each_epoch else 0
        support_seed = S.stable_seed(self.base_seed, epoch, case["unit_id"], "support")
        support_idx = sample_indices(
            case, self.cfg, support_seed,
            epoch=self.epoch if self.training else 0,
            case_index=i,
            run_seed=self.base_seed, stream="support", n_points=support_n, sampling=support_sampling,
        )
        if self.cfg.query_mode == "same":
            query_idx = support_idx
        else:
            query_seed = S.stable_seed(self.base_seed, epoch, case["unit_id"], "query")
            query_idx = sample_indices(
                case, self.cfg, query_seed, epoch=epoch, case_index=i,
                run_seed=self.base_seed, stream="query", n_points=query_n, sampling=query_sampling,
            ).copy()
        support_pos = case["pos"][support_idx]
        query_pos = case["pos"][query_idx]
        if self.training and getattr(self.cfg, "rot_aug", False):
            rot = random_rotation(int(support_seed % (2**32)) + 31)
            support_pos = (support_pos @ rot).astype(np.float32)
            query_pos = (query_pos @ rot).astype(np.float32)
        support_feat = build_features(case, support_idx, self.cfg.input_features, self.feat_stats,
                                      pos_override=support_pos)
        query_feat = (support_feat if query_idx is support_idx else
                      build_features(case, query_idx, self.cfg.input_features, self.feat_stats,
                                     pos_override=query_pos))
        return {
            "support_pos": torch.from_numpy(np.ascontiguousarray(support_pos)),
            "support_x": torch.from_numpy(support_feat),
            "support_idx": torch.from_numpy(np.ascontiguousarray(support_idx)),
            "pos": torch.from_numpy(np.ascontiguousarray(query_pos)),
            "x": torch.from_numpy(query_feat),
            "query_idx": torch.from_numpy(np.ascontiguousarray(query_idx)),
            "y": torch.from_numpy(case["y_norm"][query_idx]),
            "y_raw": torch.from_numpy(case["y_raw"][query_idx]),
            # 供几何加权 loss 用（曲率、1/局部半径）
            "curv": torch.from_numpy(np.abs(case["curvature"][query_idx]).astype(np.float32)),
            "invr": torch.from_numpy((1.0 / np.clip(case["local_radius"][query_idx], 1e-6, None)).astype(np.float32)),
            "case": case["case"], "cohort": case["cohort"], "unit_id": case["unit_id"],
        }


def collate(batch: List[Dict]) -> Dict:
    """拼成 PyG 风格 batch：concat 点 + batch 索引。"""
    pos = torch.cat([b["pos"] for b in batch], dim=0)
    x = torch.cat([b["x"] for b in batch], dim=0)
    y = torch.cat([b["y"] for b in batch], dim=0)
    y_raw = torch.cat([b["y_raw"] for b in batch], dim=0)
    batch_idx = torch.cat([
        torch.full((len(b["pos"]),), i, dtype=torch.long) for i, b in enumerate(batch)
    ])
    support_pos = torch.cat([b["support_pos"] for b in batch], dim=0)
    support_x = torch.cat([b["support_x"] for b in batch], dim=0)
    support_batch = torch.cat([
        torch.full((len(b["support_pos"]),), i, dtype=torch.long) for i, b in enumerate(batch)
    ])
    curv = torch.cat([b["curv"] for b in batch], dim=0)
    invr = torch.cat([b["invr"] for b in batch], dim=0)
    return {"pos": pos, "x": x, "y": y, "y_raw": y_raw, "batch": batch_idx,
            "support_pos": support_pos, "support_x": support_x,
            "support_batch": support_batch,
            "support_indices": [b["support_idx"] for b in batch],
            "query_indices": [b["query_idx"] for b in batch],
            "curv": curv, "invr": invr, "cases": [b["case"] for b in batch],
            "unit_ids": [b["unit_id"] for b in batch]}


def case_to_batch(case: Dict, input_features: Tuple[str, ...], feat_stats: Dict,
                  device: str = "cpu") -> Dict:
    """把单个病例的**完整点云**打成 batch（评估用）。"""
    n = len(case["pos"])
    idx = np.arange(n)
    feat = build_features(case, idx, input_features, feat_stats)
    return {
        "pos": torch.from_numpy(case["pos"]).to(device),
        "x": torch.from_numpy(feat).to(device),
        "y": torch.from_numpy(case["y_norm"]).to(device),
        "batch": torch.zeros(n, dtype=torch.long, device=device),
        "cases": [case["case"]],
    }


def worker_init_fn(worker_id: int):
    """DataLoader worker 初始化：派生独立 numpy/torch RNG。"""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    base = int(worker_info.dataset.base_seed)  # type: ignore[attr-defined]
    seed = base + 10007 * (worker_id + 1)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    stats = load_wss_stats()
    tr = load_partition(str(C.DEFAULT_SPLIT), "train", stats)
    print(f"train cases: {len(tr)}  例点数 min/max:",
          min(len(c['pos']) for c in tr), max(len(c['pos']) for c in tr))
    fs = compute_feature_stats(tr, ("x", "y", "z", "curvature"))
    print("feat_stats keys:", list(fs.keys()))
    ds = WSSMinDataset(tr, C.DataConfig(wall_n_points=2000, input_features=("x", "y", "z")),
                       fs, training=True)
    b = collate([ds[0], ds[1]])
    print("batch pos", b["pos"].shape, "x", b["x"].shape, "y", b["y"].shape,
          "batch", b["batch"].shape, "n_graphs", int(b["batch"].max()) + 1)
