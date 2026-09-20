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
from .next_geometry import case_geometry, build_query_patch, full_wall_tree, spatial_geometry_sample


# ---------------------------------------------------------------------------
# 全局 WSS 统计 / 归一化（与 pipeline_wss_min.global_stats 对齐）
# ---------------------------------------------------------------------------
def load_wss_stats(path: str | Path = C.GLOBAL_STATS) -> Dict:
    return json.loads(Path(path).read_text())


def normalize_wss(wss: np.ndarray, stats: Dict) -> np.ndarray:
    if stats["method"] == "log_z":
        floor = float(stats.get("floor", 0.0))  # 全帧统计带下限（Pa）；旧 stats 无该键 → 0，行为不变
        lg = np.log(np.clip(wss, floor, None) + stats["eps"])
        return (lg - stats["log"]["mean"]) / stats["log"]["std"]
    return (wss - stats["linear"]["mean"]) / stats["linear"]["std"]


def denormalize_wss(y: np.ndarray, stats: Dict) -> np.ndarray:
    """标准化空间 -> 原始 WSS（mm 物理量级），评估在原始空间算指标。"""
    if stats["method"] == "log_z":
        lg = y * stats["log"]["std"] + stats["log"]["mean"]
        return np.exp(lg) - stats["eps"]
    return y * stats["linear"]["std"] + stats["linear"]["mean"]


def denormalize_wss_lognormal_mean(mu: np.ndarray, logvar: np.ndarray, stats: Dict,
                                   logvar_min: float = -6.0, logvar_max: float = 2.0) -> np.ndarray:
    """逐点 Jensen 修正：高斯 NLL 头在 log_z 空间给出 (mu, logvar)，物理均值是 exp(mu+sigma^2/2)。

    ``denormalize_wss`` 的 exp(mu) 是**中位数**；对数正态的均值多一个 exp(sigma^2/2) 因子，
    这正是 log->Pa 回变换里被漏掉的那一项。logvar 按训练时同样的上下限截断。
    """
    if stats["method"] != "log_z":
        raise ValueError("pointwise Jensen correction requires log_z target statistics")
    log_mean = np.asarray(mu, dtype=np.float64) * stats["log"]["std"] + stats["log"]["mean"]
    log_var = np.exp(np.clip(np.asarray(logvar, dtype=np.float64), logvar_min, logvar_max)) * (
        float(stats["log"]["std"]) ** 2
    )
    return np.exp(log_mean + 0.5 * log_var) - stats["eps"]


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


def geom_stratified_sample(case: Dict, k: int, seed: int, bins: Tuple[int, ...] = ()) -> np.ndarray:
    """Geometry-only spatial coverage plus curvature/junction quotas.

    ``bins`` is retained for config/checkpoint compatibility; the old feature
    quantile strata did not guarantee spatial coverage and are no longer used.
    """
    if bins and (len(bins) != 3 or min(int(x) for x in bins) < 1):
        raise ValueError(f"stratified_bins must contain three positive integers: {bins}")
    return spatial_geometry_sample(case, k, seed)


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
    if sampling == "geom_stratified":
        return geom_stratified_sample(case, k, seed, getattr(cfg, "stratified_bins", ()))
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
GEOM_FEATURE_KEYS = (
    "abscissa_norm",
    "local_radius",
    "log_local_radius",
    "curvature",
    "coord_scale",
    "radius_gradient",
)
CASE_FEATURE_KEYS = frozenset(C.CASE_FEATURE_KEYS)

# cohort one-hot（逐病例常量）：从 case["cohort"] 的父系（AG/AAA/ILO）派生 0/1，
# 不做 z-score，也不进入 feature_stats。仅当出现在 input_features 时才启用。
COHORT_FEATURE_KEYS = {"cohort_ag": "AG", "cohort_aaa": "AAA", "cohort_ilo": "ILO"}


def cohort_family(cohort_rel: str) -> str:
    """canonical cohort_rel（如 'AG/fast'、'AAA/ruputer'、'ILO/<patient>'）取父系。"""
    return str(cohort_rel).split("/", 1)[0]



_WAVEFORM_CACHE: Dict[str, Dict] = {}


def load_waveform(path: str | Path) -> Dict:
    """协议入口波形（172 例共享）：steps 与逐帧 q_norm/dq_norm/t_sin/t_cos/t_norm。"""
    key = str(Path(path).resolve())
    if key not in _WAVEFORM_CACHE:
        wf = json.loads(Path(path).read_text(encoding="utf-8"))
        wf["_steps"] = np.asarray(wf["steps"], dtype=np.int64)
        for name in C.TIME_FEATURE_KEYS:
            if name not in wf:
                raise KeyError(f"waveform file lacks {name!r}: {path}")
            wf[f"_{name}"] = np.asarray(wf[name], dtype=np.float64)
        _WAVEFORM_CACHE[key] = wf
    return _WAVEFORM_CACHE[key]


def time_feature_value(case: Dict, name: str, frame_index: int | None = None) -> float:
    """某帧的时间特征标量（逐点广播）。frame_index=None 用 case 当前帧（默认峰值帧）。"""
    wf = case.get("waveform")
    if wf is None:
        raise KeyError(f"time feature {name!r} requested but case has no waveform (timesteps='peak'?)")
    k = int(case.get("frame_index", case["peak_index"]) if frame_index is None else frame_index)
    return float(wf[f"_{name}"][k])


def select_frame(case: Dict, frame_index: int) -> None:
    """把 case 的 y_raw/y_norm 切到第 frame_index 帧（就地；仅 timesteps='random_frame' 的 case 有帧数组）。"""
    if "y_raw_frames" not in case:
        raise KeyError("case has no per-frame labels (load with timesteps='random_frame')")
    case["frame_index"] = int(frame_index)
    case["y_raw"] = case["y_raw_frames"][frame_index]
    case["y_norm"] = case["y_norm_frames"][frame_index]


def sample_frame_index(case: Dict, seed: int, peak_frame_prob: float) -> int:
    rng = np.random.default_rng(seed)
    n = int(case["y_raw_frames"].shape[0])
    if rng.random() < float(peak_frame_prob):
        return int(case["peak_index"])
    return int(rng.integers(0, n))


def _transform_feature_values(name: str, vals: np.ndarray, transform: str | None) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float64)
    if name in C.CURVATURE_LIKE_KEYS and transform == "signed_log1p":
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
        if f in C.TIME_FEATURE_KEYS:  # 协议波形相位特征：有界常量，不做 z-score
            stats[f] = {"mean": 0.0, "std": 1.0, "transform": "none"}
            continue
        if f in C.V6_SEMANTIC_FEATURE_KEYS:  # 分支语义 one-hot：0/1，不做 z-score
            stats[f] = {"mean": 0.0, "std": 1.0, "transform": "none"}
            continue
        if f.startswith("radius_over_mm_") or f in {"radius_over_coord_scale", "log_radius_over_coord_scale"}:
            vals = np.concatenate([_derived_geometry_feature(c, f) for c in cases])
            vals = vals[np.isfinite(vals)]
            stats[f] = {"mean": float(vals.mean()), "std": float(vals.std() + 1e-6), "transform": "none"}
            continue
        if f not in cases[0]:
            raise KeyError(f"feature {f!r} missing from case bundle fields")
        if f in CASE_FEATURE_KEYS:
            # 病例级条件每例只计一次；不能按壁面顶点数重复，否则统计会重新
            # 引入网格密度权重。
            vals = np.asarray([float(c[f]) for c in cases], dtype=np.float64)
        else:
            vals = np.concatenate([np.asarray(c[f], dtype=np.float64).ravel() for c in cases])
        transform = curvature_transform if f in C.CURVATURE_LIKE_KEYS else "none"
        vals = _transform_feature_values(f, vals, transform)
        vals = vals[np.isfinite(vals)]
        if f in C.CURVATURE_LIKE_KEYS:  # 端点数值尖峰，先 clip 再统计
            hi = np.percentile(np.abs(vals), 99)
            vals = np.clip(vals, -hi, hi)
            stats[f] = {"mean": float(vals.mean()), "std": float(vals.std() + 1e-6),
                        "clip": float(hi), "transform": transform}
        else:
            stats[f] = {"mean": float(vals.mean()), "std": float(vals.std() + 1e-6),
                        "transform": transform}
    return stats


def _derived_geometry_feature(case: Dict, name: str) -> np.ndarray:
    """Return a geometry-only derived channel before train z-scoring."""
    r = np.asarray(case["local_radius"], dtype=np.float64)
    if name.startswith("radius_over_mm_"):
        token = name.removeprefix("radius_over_mm_").removesuffix("mm")
        scale = float(token)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid fixed-mm feature name: {name!r}")
        return r / scale
    c = max(float(case.get("coord_scale_scalar", 1.0)), 1e-9)
    ratio = r / c
    return np.log(np.clip(ratio, 1e-9, None)) if name == "log_radius_over_coord_scale" else ratio


def compute_train_weight_quantiles(cases: List[Dict]) -> Dict[str, float]:
    """train-only 固定分位，供 target/geom loss 权重使用。"""
    if any(c.get("joint_volume_target", False) for c in cases):
        raise ValueError("joint velocity/pressure labels must not be flattened into mixed-unit quantiles")
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
                   feat_stats: Dict, pos_override: np.ndarray | None = None,
                   frame_index: int | None = None) -> np.ndarray:
    pos = case["pos"][idx] if pos_override is None else pos_override
    cols = []
    for f in input_features:
        if f in C.TIME_FEATURE_KEYS:
            st = feat_stats.get(f, {"mean": 0.0, "std": 1.0})
            value = (time_feature_value(case, f, frame_index) - st["mean"]) / st["std"]
            cols.append(np.full(pos.shape[0], value, dtype=np.float64))
        elif f == "x":
            cols.append(pos[:, 0])
        elif f == "y":
            cols.append(pos[:, 1])
        elif f == "z":
            cols.append(pos[:, 2])
        elif f in COHORT_FEATURE_KEYS:
            hit = cohort_family(case["cohort"]) == COHORT_FEATURE_KEYS[f]
            cols.append(np.full(pos.shape[0], 1.0 if hit else 0.0, dtype=np.float64))
        elif f in CASE_FEATURE_KEYS:
            st = feat_stats[f]
            value = (float(case[f]) - st["mean"]) / st["std"]
            cols.append(np.full(pos.shape[0], value, dtype=np.float64))
        elif f.startswith("radius_over_mm_"):
            # E5 fixed-mm scale channel, e.g. radius_over_mm_5.  Radius is
            # already a deployment-available mm geometry quantity.
            try:
                scale_mm = float(f.removeprefix("radius_over_mm_").removesuffix("mm"))
            except ValueError as exc:
                raise ValueError(f"invalid fixed-mm feature name: {f!r}") from exc
            if not np.isfinite(scale_mm) or scale_mm <= 0:
                raise ValueError(f"fixed-mm scale must be >0, got {f!r}")
            v = np.asarray(case["local_radius"], dtype=np.float64)[idx] / scale_mm
            st = feat_stats.get(f, {"mean": 0.0, "std": 1.0})
            cols.append((v - st.get("mean", 0.0)) / st.get("std", 1.0))
        elif f in {"radius_over_coord_scale", "log_radius_over_coord_scale"}:
            v = _derived_geometry_feature(case, f)[idx]
            st = feat_stats.get(f, {"mean": 0.0, "std": 1.0})
            cols.append((v - st.get("mean", 0.0)) / st.get("std", 1.0))
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



def _attach_v6_point_features(case: Dict, d, names: Tuple[str, ...],
                              point_features_root: str | Path | None,
                              cohort_rel: str, case_name: str) -> None:
    """V6 逐点几何/语义特征：曲面主曲率族来自 sidecar，分支语义/弧长由 bundle 派生。

    只附加 ``names`` 里显式要求的键；``names`` 为空时函数不被调用，旧 run 行为逐位不变。
    """
    surface = tuple(n for n in names if n in C.V6_SURFACE_FEATURE_KEYS)
    if surface:
        if not point_features_root:
            raise ValueError(
                f"{cohort_rel}/{case_name}: {sorted(surface)} require data.point_features_root"
            )
        sidecar_path = Path(point_features_root) / cohort_rel / case_name / "features.npz"
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"missing wall geometry sidecar: {sidecar_path}")
        with np.load(sidecar_path, allow_pickle=False) as sidecar:
            if not np.array_equal(sidecar["wall_node_id_cas"], d["wall_node_id_cas"]):
                raise ValueError(
                    f"geometry sidecar rows do not match the bundle: {sidecar_path}"
                )
            for name in surface:
                key = f"wall_{name}"
                if key not in sidecar.files:
                    raise KeyError(f"{key} missing from {sidecar_path}")
                case[name] = np.asarray(sidecar[key], dtype=np.float32)
    semantic = tuple(n for n in names if n in C.V6_SEMANTIC_FEATURE_KEYS)
    if semantic:
        if "wall_semantic_id" not in d.files:
            raise KeyError(f"{cohort_rel}/{case_name}: bundle has no wall_semantic_id")
        sem = np.asarray(d["wall_semantic_id"]).astype(np.int64)
        for index, key in enumerate(C.V6_SEMANTIC_FEATURE_KEYS):  # contract order: trunk, l/r CIA, 4 terminals
            if key in semantic:
                case[key] = (sem == index).astype(np.float32)
    branch = tuple(n for n in names if n in C.V6_BRANCH_FEATURE_KEYS)
    if branch:
        if "wall_s_local_mm" not in d.files:
            raise KeyError(f"{cohort_rel}/{case_name}: bundle has no wall_s_local_mm")
        s_local = np.asarray(d["wall_s_local_mm"], dtype=np.float32)
        if "s_local_mm" in branch:
            case["s_local_mm"] = s_local
        if "s_local_frac" in branch:
            if "wall_segment_id" not in d.files:
                raise KeyError(f"{cohort_rel}/{case_name}: bundle has no wall_segment_id")
            segment = np.asarray(d["wall_segment_id"]).astype(np.int64)
            frac = np.zeros_like(s_local)
            for sid in np.unique(segment):
                rows = segment == sid
                longest = float(np.max(s_local[rows]))
                if longest > 1e-6:
                    frac[rows] = s_local[rows] / longest
            case["s_local_frac"] = frac.astype(np.float32)


VOLUME_VIEW_VERSION = "v5_volume_view_v1.1"
_VOLUME_FEATURE_PAIRS = (
    ("pos", "vol_coords_norm"), ("wall_coords_raw", "vol_coords_raw"), ("abscissa_norm", "vol_abscissa_norm"),
    ("local_radius", "vol_local_radius"), ("curvature", "vol_curvature"), ("radius_gradient", "vol_radius_gradient"),
    ("rho", "vol_rho"), ("theta_sin", "vol_theta_sin"), ("theta_cos", "vol_theta_cos"), ("dr_ds", "vol_dr_ds"),
    ("dist_to_junction_mm", "vol_dist_to_junction_mm"), ("dist_to_endpoint_mm", "vol_dist_to_endpoint_mm"),
    ("end_zone", "vol_end_zone"),
)


def _load_volume_view(path: Path, required_frame_version: str | None, peak: int) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"volume view missing: {path} (run wss_v5.views.volume_view)")
    with np.load(path, allow_pickle=True) as d:
        vol = {k: d[k] for k in d.files}
    version = str(np.asarray(vol["volume_view_version"]).item())
    if version != VOLUME_VIEW_VERSION:
        raise ValueError(f"volume view version {version!r} != {VOLUME_VIEW_VERSION!r}: {path}")
    if required_frame_version is not None:
        actual = str(np.asarray(vol["transform_frame_version"]).item())
        if actual != required_frame_version:
            raise ValueError(f"volume view frame {actual!r} != {required_frame_version!r}: {path}")
    if int(vol["peak_step"]) != int(peak):
        raise ValueError(f"volume view peak_step {int(vol['peak_step'])} != bundle {peak}: {path}")
    return vol


def _attach_volume(case: Dict, vol: Dict[str, np.ndarray], target: str, stats: Dict) -> None:
    """把内部单元拼到壁面 case 后面；壁面行 = support 池，query 池按目标选择。"""
    n_wall = len(case["pos"])
    n_vol = len(vol["vol_coords_norm"])
    for key, vkey in _VOLUME_FEATURE_PAIRS:
        if key in case:
            base = np.asarray(case[key])
            case[key] = np.concatenate([base, np.asarray(vol[vkey]).astype(base.dtype)])
    case["log_local_radius"] = np.log(
        np.clip(case["local_radius"].astype(np.float64), 1e-6, None)
    ).astype(np.float32)
    case["coord_scale"] = np.full(n_wall + n_vol, case["coord_scale_scalar"], dtype=np.float32)
    radial = np.asarray(vol["vol_radial_aligned"], dtype=np.float32)
    for i, key in enumerate(("nx_aligned", "ny_aligned", "nz_aligned")):
        if key in case:
            case[key] = np.concatenate([case[key], radial[:, i]])
    case["dist_to_wall_mm"] = np.concatenate([
        np.zeros(n_wall, dtype=np.float32), np.asarray(vol["vol_dist_to_wall_mm"], dtype=np.float32)
    ])
    if target == "pressure_mixed":
        y_raw = np.concatenate([case["y_raw"], np.asarray(vol["vol_pressure_rel_peak"], dtype=np.float32)])
        y_norm = (y_raw - stats["linear"]["mean"]) / stats["linear"]["std"]
        case["query_pool"] = np.arange(n_wall + n_vol)
        case["query_groups"] = [np.arange(n_wall), np.arange(n_wall, n_wall + n_vol)]
    elif target == "velocity":
        y_raw = np.concatenate([case["y_raw"], np.asarray(vol["vol_velocity_aligned_peak"], dtype=np.float32)])
        mean = np.asarray(stats["velocity"]["mean"], dtype=np.float32)
        std = np.asarray(stats["velocity"]["std"], dtype=np.float32)
        y_norm = (y_raw - mean) / std
        case["query_pool"] = np.arange(n_wall, n_wall + n_vol)
    elif target == "velocity_pressure":
        # Wall velocity is a finite placeholder, never a supervised target.
        # Keep each task's arithmetic identical to the historical single-task views.
        velocity = np.concatenate([
            np.zeros((n_wall, 3), dtype=np.float32),
            np.asarray(vol["vol_velocity_aligned_peak"], dtype=np.float32),
        ])
        pressure = np.concatenate([
            np.asarray(vol["wall_pressure_rel_peak"], dtype=np.float32),
            np.asarray(vol["vol_pressure_rel_peak"], dtype=np.float32),
        ])
        velocity_z = (velocity - np.asarray(stats["velocity"]["mean"], dtype=np.float32)) / np.asarray(
            stats["velocity"]["std"], dtype=np.float32
        )
        pressure_z = (pressure - stats["linear"]["mean"]) / stats["linear"]["std"]
        y_raw = np.column_stack([velocity, pressure])
        y_norm = np.column_stack([velocity_z, pressure_z])
        case["query_pool"] = np.arange(n_wall + n_vol)
        case["query_groups"] = [np.arange(n_wall), np.arange(n_wall, n_wall + n_vol)]
        case["joint_volume_target"] = True
    else:
        raise ValueError(target)
    case["y_raw"] = y_raw.astype(np.float32)
    case["y_norm"] = y_norm.astype(np.float32)
    case["support_pool"] = np.arange(n_wall)
    case["n_wall"] = n_wall
    case["point_kind"] = np.concatenate([np.zeros(n_wall, dtype=np.int8), np.ones(n_vol, dtype=np.int8)])
    case["volume_view_version"] = VOLUME_VIEW_VERSION
    case["p_ref_pa"] = float(vol["p_ref_pa"])


def _pool_random(pool: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k <= 0 or k >= len(pool):
        return np.asarray(pool).copy()
    return np.asarray(pool)[np.random.default_rng(seed).choice(len(pool), size=k, replace=False)]


def sample_support_indices(case: Dict, cfg: C.DataConfig, seed: int, *, n_points: int,
                           sampling: str, stream: str, **legacy_kwargs) -> np.ndarray:
    """support 采样：有 support_pool（体场目标）时只在壁面行内随机；否则走旧 sample_indices。"""
    if "support_pool" not in case:
        return sample_indices(case, cfg, seed, stream=stream, n_points=n_points, sampling=sampling, **legacy_kwargs)
    if sampling != "random":
        raise NotImplementedError("volume targets only support random support sampling")
    return np.sort(_pool_random(case["support_pool"], int(n_points), seed))


def sample_query_indices(case: Dict, cfg: C.DataConfig, seed: int, *, n_points: int,
                         sampling: str, stream: str, **legacy_kwargs) -> np.ndarray:
    """query 采样：query_groups（壁面+内部混合，按 query_wall_fraction 分配）> query_pool > 旧行为。"""
    if "query_groups" in case:
        if sampling != "random":
            raise NotImplementedError("volume targets only support random query sampling")
        groups = case["query_groups"]
        k_wall = int(round(float(getattr(cfg, "query_wall_fraction", 0.5)) * int(n_points)))
        counts = [k_wall, int(n_points) - k_wall]
        parts = [_pool_random(g, k, seed + 1000003 * gi) for gi, (g, k) in enumerate(zip(groups, counts)) if k > 0]
        return np.sort(np.concatenate(parts))
    if "query_pool" in case:
        if sampling != "random":
            raise NotImplementedError("volume targets only support random query sampling")
        return np.sort(_pool_random(case["query_pool"], int(n_points), seed))
    return sample_indices(case, cfg, seed, stream=stream, n_points=n_points, sampling=sampling, **legacy_kwargs)


def sample_joint_query_indices(case: Dict, cfg: C.DataConfig, seed: int, *, n_points: int,
                               sampling: str = "random", stream: str = "query",
                               **legacy_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Union the historical pressure and velocity draws without changing either draw.

    ``n_points`` is the supervision budget for EACH task, not the union budget.
    Returned masks follow the sorted union and may overlap on interior rows.
    """
    n_wall = int(case["n_wall"])
    interior = np.arange(n_wall, len(case["pos"]))
    n_wall_query = int(round(float(cfg.query_wall_fraction) * int(n_points)))
    if len(interior) < n_points or n_wall < n_wall_query:
        raise ValueError("joint volume queries require the full per-task supervision budget")
    pressure_case = {"query_groups": [np.arange(n_wall), interior]}
    velocity_case = {"query_pool": interior}
    kwargs = dict(n_points=n_points, sampling=sampling, stream=stream, **legacy_kwargs)
    pressure_rows = sample_query_indices(pressure_case, cfg, seed, **kwargs)
    velocity_rows = sample_query_indices(velocity_case, cfg, seed, **kwargs)
    union = np.union1d(pressure_rows, velocity_rows)
    return union, np.isin(union, velocity_rows), np.isin(union, pressure_rows)


def query_rows(case: Dict) -> np.ndarray:
    """评估用全量 query 行：体场目标为 query_pool，否则全部点。"""
    if "query_pool" in case:
        return np.asarray(case["query_pool"])
    return np.arange(len(case["pos"]))


def subset_case(case: Dict, rows: np.ndarray) -> Dict:
    """按行切出子 case（评估分组用）；逐点数组按 rows 索引，标量原样保留。"""
    n = len(case["pos"])
    out = {}
    for k, v in case.items():
        if k in {"support_pool", "query_pool", "query_groups"}:
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[rows]
        else:
            out[k] = v
    return out


def load_case(cohort_rel: str, case_name: str, wss_stats: Dict,
              target: str = "wss", target_normalization: str = "global_stats",
              data_root: str | Path = C.DATA_ROOT,
              required_frame_version: str | None = None,
              case_features: Dict[str, float] | None = None,
              timesteps: str = "peak", waveform_path: str | Path | None = None,
              extra_point_features: Tuple[str, ...] = (),
              point_features_root: str | Path | None = None) -> Dict:
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
        volume = None
        if target == "wss":
            y_raw = d["wall_wss"][si].astype(np.float32)
        elif target == "pressure":
            # 壁面压力 gauge：逐例去均值（相对压力），隔离跨例 DC 偏置，
            # 使指标聚焦于几何可预测的空间压力型态（可为负）。
            p_raw = d["wall_pressure"][si].astype(np.float32)
            y_raw = (p_raw - np.float32(p_raw.mean())).astype(np.float32)
        elif target in C.VOLUME_TARGETS:
            # 体场目标：support 仍是壁面点云；query 来自 volume.npz（内部单元），压力目标把壁面点也混入 query 池。
            volume = _load_volume_view(p.parent / "volume.npz", required_frame_version, peak)
            if target == "pressure_mixed":
                y_raw = volume["wall_pressure_rel_peak"].astype(np.float32)  # 与内部点共用同一 p_ref
            elif target == "velocity_pressure":
                y_raw = np.zeros((len(pos), 4), dtype=np.float32)
                y_raw[:, 3] = volume["wall_pressure_rel_peak"].astype(np.float32)
            else:
                y_raw = np.zeros((len(pos), 3), dtype=np.float32)  # 无滑移；壁面行不会成为速度 query
        else:
            raise ValueError(f"unsupported target={target!r} (expect 'wss'|'pressure'|{C.VOLUME_TARGETS})")
        if volume is None:
            y_norm, target_norm_meta = normalize_target(y_raw, wss_stats, target_normalization)
        else:
            y_norm = np.zeros_like(y_raw)
            target_norm_meta = {"mode": target_normalization, "stats_source": "train-only volume statistics", "target": target}
        local_radius = d["wall_local_radius"].astype(np.float32)
        log_local_radius = np.log(
            np.clip(local_radius.astype(np.float64), 1e-6, None)
        ).astype(np.float32)
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
            local_radius=local_radius,
            log_local_radius=log_local_radius,
            curvature=d["wall_curvature"].astype(np.float32),
            coord_scale=np.full(len(pos), float(d["coord_scale"]), dtype=np.float32),
            radius_gradient=_radius_gradient_from_bundle(d),
            peak_step=peak,
            coord_scale_scalar=float(d["coord_scale"]),
            bundle_path=str(p),
        )
        # V5 视图的可选逐点特征（旧 bundle 无这些键则跳过，行为不变）
        for name in C.V5_POINT_FEATURE_KEYS:
            key = f"wall_{name}"
            if key in d.files:
                case[name] = np.asarray(d[key], dtype=np.float32)
        if "wall_normal_pca_aligned" in d.files:
            normals = np.asarray(d["wall_normal_pca_aligned"], dtype=np.float32)
            case["nx_aligned"], case["ny_aligned"], case["nz_aligned"] = normals[:, 0], normals[:, 1], normals[:, 2]
        if extra_point_features:
            _attach_v6_point_features(case, d, tuple(extra_point_features),
                                      point_features_root, cohort_rel, case_name)
        if volume is not None:
            _attach_volume(case, volume, target, wss_stats)
        if timesteps == "random_frame":
            if target != "wss":
                raise ValueError("timesteps='random_frame' only supports target='wss'")
            frames = d["wall_wss"].astype(np.float32)  # (81, N)
            wf = load_waveform(waveform_path) if waveform_path else None
            if wf is None:
                raise ValueError("timesteps='random_frame' requires waveform_path")
            if not np.array_equal(wf["_steps"], np.asarray(steps, dtype=np.int64)):
                raise ValueError(f"waveform steps differ from bundle steps: {cohort_rel}/{case_name}")
            case["y_raw_frames"] = frames
            case["y_norm_frames"] = normalize_wss(frames, wss_stats).astype(np.float32)
            case["steps"] = np.asarray(steps, dtype=np.int64)
            case["peak_index"] = int(si)
            case["waveform"] = wf
            select_frame(case, int(si))  # 默认停在峰值帧：不改帧时与单帧 case 完全一致
        if case_features:
            unknown = sorted(set(case_features) - CASE_FEATURE_KEYS)
            if unknown:
                raise KeyError(f"unknown case-level features for {case['unit_id']}: {unknown}")
            for key, value in case_features.items():
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(
                        f"non-finite case-level feature {key!r} for {case['unit_id']}"
                    )
                case[key] = value
    return case


def load_case_feature_table(path: str | Path) -> Dict[str, Dict[str, float]]:
    """Load a strict unit_id -> case-level feature mapping."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    features = tuple(payload.get("feature_names", ()))
    if not features or len(set(features)) != len(features):
        raise ValueError(f"invalid feature_names in case feature table: {source}")
    unknown = sorted(set(features) - CASE_FEATURE_KEYS)
    if unknown:
        raise KeyError(f"unknown case-level feature names {unknown}: {source}")
    rows = payload.get("cases")
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"case feature table has no cases: {source}")
    parsed: Dict[str, Dict[str, float]] = {}
    expected = set(features)
    for unit_id, row in rows.items():
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError(
                f"case feature columns mismatch for {unit_id!r}: "
                f"expected={sorted(expected)} actual={sorted(row) if isinstance(row, dict) else type(row)}"
            )
        parsed[str(unit_id)] = {key: float(row[key]) for key in features}
    return parsed


def load_partition(split_path: str, partition: str, wss_stats: Dict,
                   strict: bool = True, target: str = "wss",
                   target_normalization: str = "global_stats",
                   data_root: str | Path = C.DATA_ROOT,
                   required_frame_version: str | None = None,
                   case_features_path: str | Path | None = None,
                   timesteps: str = "peak", waveform_path: str | Path | None = None,
                   extra_point_features: Tuple[str, ...] = (),
                   point_features_root: str | Path | None = None) -> List[Dict]:
    labels = load_split_cases(split_path, partition)
    case_feature_table = (
        load_case_feature_table(case_features_path) if case_features_path else None
    )
    cases = []
    missing = []
    for cohort_rel, case_name in labels:
        unit_id = f"{cohort_rel}/{case_name}"
        p = Path(data_root) / cohort_rel / case_name / "bundle.npz"
        if not p.is_file():
            missing.append(unit_id)
            continue
        if case_feature_table is not None and unit_id not in case_feature_table:
            raise KeyError(
                f"case-level features missing for {unit_id}: {case_features_path}"
            )
        cases.append(load_case(
            cohort_rel, case_name, wss_stats, target=target,
            target_normalization=target_normalization,
            data_root=data_root, required_frame_version=required_frame_version,
            case_features=(
                case_feature_table[unit_id] if case_feature_table is not None else None
            ),
            timesteps=timesteps, waveform_path=waveform_path,
            extra_point_features=extra_point_features,
            point_features_root=point_features_root,
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
        if getattr(cfg, "local_geometry", False) and getattr(cfg, "rot_aug", False):
            raise ValueError("local_geometry currently requires rot_aug=false to keep all vector features aligned")
        if int(getattr(cfg, "query_patch_nsample", 0)) > 0 and not getattr(cfg, "local_geometry", False):
            raise ValueError("query patches require local_geometry=true")
        support_n = int(cfg.support_n_points or cfg.wall_n_points)
        support_sampling = cfg.support_sampling or cfg.sampling
        query_n = int(cfg.query_n_points or support_n)
        if (training and support_sampling in {"random", "area_random"} and support_n > 0
                and not getattr(cfg, "support_allow_undersized", False)):
            too_small = [
                f"{c['cohort']}/{c['case']}({len(c.get('support_pool', c['pos']))})"
                for c in cases if len(c.get("support_pool", c["pos"])) < support_n
            ]
            if too_small:
                raise ValueError(
                    "random sampling protocol requires the same number of points per case; "
                    f"requested {support_n}, too-small cases: {', '.join(too_small[:5])}"
                )
        if cfg.query_mode == "independent" and any(len(query_rows(c)) < query_n for c in cases):
            raise ValueError(f"independent query requires at least {query_n} wall points per case")
        if cfg.target == "velocity_pressure":
            for case in cases:
                n_wall = int(case["n_wall"])
                if (len(case["pos"]) - n_wall < query_n
                        or n_wall < round(cfg.query_wall_fraction * query_n)):
                    raise ValueError("joint volume queries require the full per-task supervision budget")
        # 给每个 case 一个稳定的 fps seed
        for i, c in enumerate(cases):
            c.setdefault("unit_id", f"{c.get('cohort', 'AG/unknown')}/{c.get('case', i)}")
            c["geom_seed"] = base_seed + 7919 * i
            if getattr(cfg, "local_geometry", False):
                case_geometry(c)
            if int(getattr(cfg, "query_patch_nsample", 0)) > 0:
                full_wall_tree(c)
                if getattr(cfg, "timesteps", "peak") == "peak":
                    # Workers are recreated each epoch in the frozen protocol.
                    # Prime shared read-only features before fork, avoiding
                    # repeated full-wall transformations inside every worker.
                    build_query_patch(c, np.empty(0, dtype=np.int64), cfg.input_features,
                                      feat_stats, nsample=int(cfg.query_patch_nsample))
            if support_sampling == "geom_stratified":
                # The voxel search is independent of epoch; keep it on the
                # parent dataset so persistent_workers=false does not redo it.
                geom_stratified_sample(c, support_n, c["geom_seed"],
                                       getattr(cfg, "stratified_bins", ()))
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
        support_idx = sample_support_indices(
            case, self.cfg, support_seed,
            epoch=self.epoch if self.training else 0,
            case_index=i,
            run_seed=self.base_seed, stream="support", n_points=support_n, sampling=support_sampling,
        )
        joint_masks = None
        if self.cfg.query_mode == "same":
            query_idx = support_idx
        else:
            query_seed = S.stable_seed(self.base_seed, epoch, case["unit_id"], "query")
            if self.cfg.target == "velocity_pressure":
                query_idx, velocity_mask, pressure_mask = sample_joint_query_indices(
                    case, self.cfg, query_seed, epoch=epoch, case_index=i,
                    run_seed=self.base_seed, stream="query", n_points=query_n, sampling=query_sampling,
                )
                joint_masks = {"velocity_mask": torch.from_numpy(velocity_mask),
                               "pressure_mask": torch.from_numpy(pressure_mask)}
            else:
                query_idx = sample_query_indices(
                    case, self.cfg, query_seed, epoch=epoch, case_index=i,
                    run_seed=self.base_seed, stream="query", n_points=query_n, sampling=query_sampling,
                ).copy()
        frame_index = None
        y_norm_src, y_raw_src = case["y_norm"], case["y_raw"]
        if getattr(self.cfg, "timesteps", "peak") == "random_frame":
            if self.training:
                frame_seed = S.stable_seed(self.base_seed, epoch, case["unit_id"], "frame")
                frame_index = sample_frame_index(case, frame_seed, self.cfg.peak_frame_prob)
            else:
                frame_index = int(case["peak_index"])
            y_norm_src, y_raw_src = case["y_norm_frames"][frame_index], case["y_raw_frames"][frame_index]
        support_pos = case["pos"][support_idx]
        query_pos = case["pos"][query_idx]
        if self.training and getattr(self.cfg, "rot_aug", False):
            rot = random_rotation(int(support_seed % (2**32)) + 31)
            support_pos = (support_pos @ rot).astype(np.float32)
            query_pos = (query_pos @ rot).astype(np.float32)
        support_feat = build_features(case, support_idx, self.cfg.input_features, self.feat_stats,
                                      pos_override=support_pos, frame_index=frame_index)
        query_feat = (support_feat if query_idx is support_idx else
                      build_features(case, query_idx, self.cfg.input_features, self.feat_stats,
                                     pos_override=query_pos, frame_index=frame_index))
        result = {
            "support_pos": torch.from_numpy(np.ascontiguousarray(support_pos)),
            "support_x": torch.from_numpy(support_feat),
            "support_idx": torch.from_numpy(np.ascontiguousarray(support_idx)),
            "pos": torch.from_numpy(np.ascontiguousarray(query_pos)),
            "x": torch.from_numpy(query_feat),
            "query_idx": torch.from_numpy(np.ascontiguousarray(query_idx)),
            "y": torch.from_numpy(y_norm_src[query_idx]),
            "y_raw": torch.from_numpy(y_raw_src[query_idx]),
            "frame_index": -1 if frame_index is None else int(frame_index),
            # 供几何加权 loss 用（曲率、1/局部半径）
            "curv": torch.from_numpy(np.abs(case["curvature"][query_idx]).astype(np.float32)),
            "invr": torch.from_numpy((1.0 / np.clip(case["local_radius"][query_idx], 1e-6, None)).astype(np.float32)),
            "case": case["case"], "cohort": case["cohort"], "unit_id": case["unit_id"],
        }
        if getattr(self.cfg, "local_geometry", False):
            geometry = case_geometry(case)
            result["support_geometry"] = torch.from_numpy(np.ascontiguousarray(geometry[support_idx]))
            result["query_geometry"] = torch.from_numpy(np.ascontiguousarray(geometry[query_idx]))
        patch_k = int(getattr(self.cfg, "query_patch_nsample", 0))
        if patch_k > 0:
            patch = build_query_patch(case, query_idx, self.cfg.input_features, self.feat_stats,
                                      nsample=patch_k, frame_index=frame_index)
            result["query_patch"] = {key: torch.from_numpy(value) for key, value in patch.items()}
        if joint_masks is not None:
            result.update(joint_masks)
        return result


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
    result = {"pos": pos, "x": x, "y": y, "y_raw": y_raw, "batch": batch_idx,
            "support_pos": support_pos, "support_x": support_x,
            "support_batch": support_batch,
            "support_indices": [b["support_idx"] for b in batch],
            "query_indices": [b["query_idx"] for b in batch],
            "curv": curv, "invr": invr, "cases": [b["case"] for b in batch],
            "unit_ids": [b["unit_id"] for b in batch],
            "frame_indices": [b.get("frame_index", -1) for b in batch]}
    for key in ("support_geometry", "query_geometry"):
        if any(key in item for item in batch):
            if not all(key in item for item in batch):
                raise ValueError(f"cannot collate mixed {key} availability")
            result[key] = torch.cat([item[key] for item in batch], dim=0)
    if any("query_patch" in item for item in batch):
        if not all("query_patch" in item for item in batch):
            raise ValueError("cannot collate mixed query patch availability")
        result["query_patch"] = {key: torch.cat([item["query_patch"][key] for item in batch], dim=0)
                                 for key in ("features", "relative_mm")}
    if any("velocity_mask" in item or "pressure_mask" in item for item in batch):
        if not all("velocity_mask" in item and "pressure_mask" in item for item in batch):
            raise ValueError("cannot collate mixed joint and single-task supervision")
        result.update({key: torch.cat([item[key] for item in batch])
                       for key in ("velocity_mask", "pressure_mask")})
    return result


def local_model_context(batch: Dict, device) -> Dict:
    """Move only opted-in geometry tensors; legacy model calls stay unchanged."""
    out = {key: batch[key].to(device) for key in ("support_geometry", "query_geometry") if key in batch}
    if "query_patch" in batch:
        out["query_patch"] = {key: value.to(device) for key, value in batch["query_patch"].items()}
    return out


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
