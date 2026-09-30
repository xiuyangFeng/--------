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
from . import time_basis as TB
from . import surface as S
from . import longitudinal_geometry as LG
from .next_geometry import case_geometry, case_section, build_query_patch, full_wall_tree, spatial_geometry_sample


# ---------------------------------------------------------------------------
# 全局 WSS 统计 / 归一化（与 pipeline_wss_min.global_stats 对齐）
# ---------------------------------------------------------------------------
def load_wss_stats(path: str | Path = C.GLOBAL_STATS) -> Dict:
    return json.loads(Path(path).read_text())


def normalize_wss(wss: np.ndarray, stats: Dict, offset: np.ndarray | None = None) -> np.ndarray:
    if stats["method"] == "log_z":
        floor = float(stats.get("floor", 0.0))  # 全帧统计带下限（Pa）；旧 stats 无该键 → 0，行为不变
        lg = np.log(np.clip(wss, floor, None) + stats["eps"])
        if "offset" in stats:
            # 残差目标：z = (ln(WSS+eps) − offset − mean_r) / std_r；offset 是逐点几何先验（如 log_tau0）
            if offset is None:
                raise ValueError("residual-target statistics require the per-point offset array")
            return (lg - np.asarray(offset, dtype=np.float64) - stats["offset"]["mean"]) / stats["offset"]["std"]
        return (lg - stats["log"]["mean"]) / stats["log"]["std"]
    if stats["method"] == "logit_z":
        # 有界目标（OSI ∈ [0, scale]）：z = (logit(clip(v/scale, c, 1−c)) − mean) / std；stats['logit'] 只用训练折
        lp = stats["logit"]
        u = np.clip(np.asarray(wss, dtype=np.float64) / float(lp["scale"]), float(lp["clip"]), 1.0 - float(lp["clip"]))
        return (np.log(u) - np.log1p(-u) - lp["mean"]) / lp["std"]
    if stats["method"] == "multi":
        # M1 多头：(N, C) 逐列用各通道子统计量归一化（通道顺序 = stats['channels']）
        arr = np.asarray(wss, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(stats["channels"]):
            raise ValueError(f"multi-channel target must be (N, {len(stats['channels'])}), got {arr.shape}")
        return np.stack([normalize_wss(arr[:, j], stats[name]) for j, name in enumerate(stats["channels"])], axis=1)
    return (wss - stats["linear"]["mean"]) / stats["linear"]["std"]


def standard_logz(wss: np.ndarray, stats: Dict) -> np.ndarray:
    """标准 log_z（只用 stats["log"]），残差目标评估时把真值/预测换回可比口径。"""
    floor = float(stats.get("floor", 0.0))
    lg = np.log(np.clip(wss, floor, None) + stats["eps"])
    return (lg - stats["log"]["mean"]) / stats["log"]["std"]


def residual_to_standard_logz(z_residual: np.ndarray, offset: np.ndarray, stats: Dict) -> np.ndarray:
    """残差 z（模型输出）→ 标准 log_z：ln = z·std_r + mean_r + offset；z_std = (ln − mean)/std。"""
    if "offset" not in stats:
        raise ValueError("stats carry no residual-target offset")
    ln = np.asarray(z_residual, dtype=np.float64) * stats["offset"]["std"] + stats["offset"]["mean"] + np.asarray(offset, dtype=np.float64)
    return (ln - stats["log"]["mean"]) / stats["log"]["std"]


def gate_residual_prediction(z_residual: np.ndarray, offset: np.ndarray, stats: Dict, quantile: float) -> np.ndarray:
    """T3 级联门控：只在偏移特征（一阶段预测）的逐例 top-(1-quantile) 区域保留二阶段残差，区外残差置 0。

    残差 0 对应 ln(WSS) = offset，即一阶段预测本身；在残差 z 空间等于 (0 − mean_r)/std_r。"""
    if "offset" not in stats:
        raise ValueError("residual gating requires residual-target statistics")
    if not (0.0 < float(quantile) < 1.0):
        raise ValueError("residual gate quantile must be in (0, 1)")
    offset = np.asarray(offset, dtype=np.float64)
    out = np.array(z_residual, dtype=np.float64, copy=True)
    if out.shape != offset.shape:
        raise ValueError("residual and offset must align point by point")
    threshold = np.quantile(offset, float(quantile))
    out[offset < threshold] = (0.0 - stats["offset"]["mean"]) / stats["offset"]["std"]
    return out


def frame_stats_view(stats: Dict, frame_index: int) -> Dict:
    """frame_stats 归一化：把 stats 的全局均值/标准差换成第 frame_index 帧的（其余键不变）。

    - WSS（log_z）换 stats['log']，normalize_wss / denormalize_wss 只读这一块，单帧路径原样复用；
    - 体场（linear）换 stats['linear']（压力 Pa，wall∪interior）、stats['velocity']（逐分量 m/s）
      与 stats['speed']，`denormalize_volume_prediction` 与速度评估同样只读这几块。
    """
    frame = stats.get("frame")
    if frame is None:
        raise ValueError("target_normalization='frame_stats' needs per-frame statistics ('frame' block) in the stats file")
    k = int(frame_index)
    view = dict(stats)
    if "log_mean" in frame:
        view["log"] = {"mean": float(frame["log_mean"][k]), "std": float(frame["log_std"][k])}
    else:
        view["linear"] = {**stats.get("linear", {}),
                          "mean": float(frame["linear_mean"][k]), "std": float(frame["linear_std"][k])}
        view["velocity"] = {**stats.get("velocity", {}),
                            "mean": list(frame["velocity_mean"][k]), "std": list(frame["velocity_std"][k])}
        if "speed_mean" in frame:
            view["speed"] = {**stats.get("speed", {}),
                             "mean": float(frame["speed_mean"][k]), "std": float(frame["speed_std"][k])}
    view["_frame_index"] = k
    return view


def denormalize_wss(y: np.ndarray, stats: Dict) -> np.ndarray:
    """标准化空间 -> 原始 WSS（mm 物理量级），评估在原始空间算指标。"""
    if stats["method"] == "log_z":
        lg = y * stats["log"]["std"] + stats["log"]["mean"]
        return np.exp(lg) - stats["eps"]
    if stats["method"] == "logit_z":
        lp = stats["logit"]
        x = np.asarray(y, dtype=np.float64) * lp["std"] + lp["mean"]
        return float(lp["scale"]) / (1.0 + np.exp(-x))
    if stats["method"] == "multi":
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(stats["channels"]):
            raise ValueError(f"multi-channel prediction must be (N, {len(stats['channels'])}), got {arr.shape}")
        return np.stack([denormalize_wss(arr[:, j], stats[name]) for j, name in enumerate(stats["channels"])], axis=1)
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
        # 2026-09-27: synthetic children carry a "~m<NN>" tag after the patient suffix (ILO/NAME-0~m24/before);
        # library IDs are unchanged (the optional group never matches them).
        if not re.fullmatch(r".+-(?:0|1)(?:~m\d+)?", parts[1]):
            raise ValueError(f"invalid ILO patient suffix in split ID: {label!r}")
        if parts[2] not in {"before", "after"}:  # 2026-09-22: post-operative (after) units admitted (user decision 09-21)
            raise ValueError(f"ILO case must be before|after, got: {label!r}")
        return "/".join(parts)
    raise ValueError(
        "invalid split case ID (expect legacy AG or canonical AG/AAA/ILO-before|after): "
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
    """把 case 的 y_raw/y_norm 切到第 frame_index 帧（就地；多帧模式的 case 才有帧标签）。

    壁面 WSS 用内存里的 (81, N) 帧数组；体场目标走 `volume_frames` 懒读句柄。"""
    if "y_raw_frames" not in case:
        if "volume_frames" in case:
            case["frame_index"] = int(frame_index)
            case["y_raw"], case["y_norm"] = volume_frame_labels(case, frame_index)
            return
        raise KeyError("case has no per-frame labels (load with timesteps='random_frame')")
    case["frame_index"] = int(frame_index)
    case["y_raw"] = case["y_raw_frames"][frame_index]
    case["y_norm"] = case["y_norm_frames"][frame_index]


def _normalize_frames(frames: np.ndarray, wss_stats: Dict, target_normalization: str) -> np.ndarray:
    """(T, N) 原始帧 → 标准化帧；frame_stats 逐帧用各自的 log 均值/标准差，其余口径与单帧一致。"""
    if target_normalization == "frame_stats":
        return np.stack([normalize_wss(frames[k], frame_stats_view(wss_stats, k)) for k in range(frames.shape[0])]).astype(np.float32)
    return normalize_wss(frames, wss_stats).astype(np.float32)


def sample_frame_index(case: Dict, seed: int, peak_frame_prob: float) -> int:
    rng = np.random.default_rng(seed)
    n = int(case["y_raw_frames"].shape[0]) if "y_raw_frames" in case else int(case["n_frames"])
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
    stats: Dict[str, Dict[str, float]] = LG.compute_stats(cases, input_features)
    for f in input_features:
        if f in LG.FEATURE_KEYS:
            continue
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
    if LG.validate_feature_pairs(input_features):
        LG.validate_frozen_stats(feat_stats, input_features)
    pos = case["pos"][idx] if pos_override is None else pos_override
    cols = []
    for f in input_features:
        if f in LG.FEATURE_KEYS:
            cols.append(LG.feature_column(case, idx, f, feat_stats))
        elif f in C.TIME_FEATURE_KEYS:
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
    LG.attach_sidecar(case, d, names, point_features_root)
    surface = tuple(n for n in names if n in C.SIDECAR_FEATURE_KEYS and n not in LG.FEATURE_KEYS)
    if surface:
        if not point_features_root:
            raise ValueError(
                f"{cohort_rel}/{case_name}: {sorted(surface)} require data.point_features_root"
            )
        # 一个或多个 sidecar 根目录，按顺序查找每个键（wave-1 同时用 wss_min_geom_v2 与 wss_min_flowref_v1）
        roots = [point_features_root] if isinstance(point_features_root, (str, Path)) else list(point_features_root)
        pending = list(surface)
        for root in roots:
            sidecar_path = Path(root) / cohort_rel / case_name / "features.npz"
            if not sidecar_path.is_file():
                raise FileNotFoundError(f"missing wall geometry sidecar: {sidecar_path}")
            with np.load(sidecar_path, allow_pickle=False) as sidecar:
                if not np.array_equal(sidecar["wall_node_id_cas"], d["wall_node_id_cas"]):
                    raise ValueError(
                        f"geometry sidecar rows do not match the bundle: {sidecar_path}"
                    )
                for name in list(pending):
                    key = f"wall_{name}"
                    if key in sidecar.files:
                        case[name] = np.asarray(sidecar[key], dtype=np.float32)
                        pending.remove(name)
        if pending:
            raise KeyError(f"{cohort_rel}/{case_name}: sidecar keys {pending} missing from {roots}")
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


VOLUME_TIME_SIDECAR_VERSION = "v5_volume_time_sidecar_v1"


class VolumeFrameSource:
    """体场 81 帧标签的懒读句柄（`timesteps != 'peak'` 的体场目标用）。

    为什么不像 WSS 那样把 (81, N) 标签整块放进 case：内部单元每例 3–7×10⁵ 个，81 帧速度
    就是 0.3–0.7 GB/例，136 例放不进内存。`case.h5` 的 `volume_temporal` 是按帧分块的 lzf
    数组，单帧读约 5 ms、整周期读不到 1 s，因此直接按需读，不再落任何缓存副本。

    行布局与 `_attach_volume` 完全一致：前 `n_wall` 行是壁面节点、其后是内部单元。
    - 压力：`p_rel(x, t) = p(x, t) − p_volume_mean(t)`（逐帧参考，与峰值帧视图同定义延伸到整周期），
      壁面行取最近内部单元（与 volume_view v1.1 的壁面压力标签同规则，索引由 A0 的 sidecar 给出）；
    - 速度：旋转进 atlas 对齐系，壁面行按无滑移填 0（速度 query 池本来只含内部行）。

    h5py 句柄不能跨进程，`__getstate__` 会把它丢掉，DataLoader worker 里第一次读时重新打开。
    """

    def __init__(self, h5_path: str | Path, target: str, *, n_wall: int, n_vol: int,
                 wall_nearest_cell_index: np.ndarray, p_volume_mean_pa: np.ndarray,
                 rotation: np.ndarray, steps: np.ndarray, rows: np.ndarray | None = None):
        self.h5_path = str(h5_path)
        self.rows = None if rows is None else np.asarray(rows, dtype=np.int64)
        self.target = str(target)
        self.n_wall, self.n_vol = int(n_wall), int(n_vol)
        self.wall_nearest_cell_index = np.asarray(wall_nearest_cell_index, dtype=np.int64)
        self.p_volume_mean_pa = np.asarray(p_volume_mean_pa, dtype=np.float64)
        self.rotation = np.asarray(rotation, dtype=np.float64)
        self.steps = np.asarray(steps, dtype=np.int64)
        self._h5 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    @property
    def n_frames(self) -> int:
        return len(self.steps)

    def _dataset(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.h5_path, "r")
        key = "volume_temporal/pressure_pa" if self.target == "pressure_mixed" else "volume_temporal/velocity_m_s"
        return self._h5[key]

    def take_rows(self, rows: np.ndarray) -> "VolumeFrameSource":
        """行子集视图（评估分组 / 子采样口径用）；行号是整例 wall∪interior 行空间里的下标。"""
        rows = np.asarray(rows, dtype=np.int64)
        return VolumeFrameSource(self.h5_path, self.target, n_wall=self.n_wall, n_vol=self.n_vol,
                                 wall_nearest_cell_index=self.wall_nearest_cell_index,
                                 p_volume_mean_pa=self.p_volume_mean_pa, rotation=self.rotation,
                                 steps=self.steps, rows=rows if self.rows is None else self.rows[rows])

    def frame(self, frame_index: int) -> np.ndarray:
        """第 k 帧标签：压力 (N,) Pa；速度 (N, 3) m/s（N = 整例行数，或 take_rows 后的子集行数）。"""
        k = int(frame_index)
        raw = self._dataset()[k]
        if self.target == "pressure_mixed":
            interior = raw.astype(np.float64) - self.p_volume_mean_pa[k]
            full = np.concatenate([interior[self.wall_nearest_cell_index], interior])
        else:
            interior = raw.astype(np.float64) @ self.rotation.T
            full = np.concatenate([np.zeros((self.n_wall, 3)), interior])
        return (full if self.rows is None else full[self.rows]).astype(np.float32)

    def frames(self) -> np.ndarray:
        """全部 81 帧（(T, n_wall + n_vol[, 3]) float32）；只在时间基投影这类一次性归约里用。"""
        return np.stack([self.frame(k) for k in range(self.n_frames)])

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


CYCLE_LABEL_KEYS = {"tawss": "wall_tawss", "osi": "wall_osi", "rev_frac": "wall_rev_frac"}


def load_cycle_label(cycle_view_root: str | Path | None, cohort_rel: str, case_name: str, target: str,
                     node_id: np.ndarray) -> np.ndarray:
    """周期积分量标签（wss_v5.views.wall_cycle_v1 的 cycle.npz）：行序 = 视图 bundle 行序，用 wall_node_id_cas 逐位校验。"""
    if not cycle_view_root:
        raise ValueError(f"target={target!r} requires data.cycle_view_root")
    key = CYCLE_LABEL_KEYS[target]
    path = Path(cycle_view_root) / cohort_rel / case_name / "cycle.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing cycle label view: {path}")
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"{path}: missing {key}")
        rows = np.asarray(z["wall_node_id_cas"]).astype(np.int64)
        y = np.asarray(z[key], dtype=np.float32)
    if rows.shape != node_id.shape or not np.array_equal(rows, node_id):
        raise ValueError(f"{cohort_rel}/{case_name}: cycle label rows do not match the view bundle rows")
    if y.shape != node_id.shape or not np.isfinite(y).all():
        raise ValueError(f"{cohort_rel}/{case_name}: invalid {key} labels")
    if target == "osi" and (float(y.min()) < 0.0 or float(y.max()) > C.OSI_MAX + 1e-6):
        raise ValueError(f"{cohort_rel}/{case_name}: OSI labels outside [0, {C.OSI_MAX}]")
    return y


def load_volume_time_sidecar(sidecar_root: str | Path, cohort_rel: str, case_name: str) -> Dict[str, np.ndarray]:
    path = Path(sidecar_root) / cohort_rel / case_name / "sidecar.npz"
    if not path.is_file():
        raise FileNotFoundError(f"volume time sidecar missing: {path} "
                                "(run training_wss_min.experiments.volume_time_20260919.offline.a0_scan)")
    with np.load(path, allow_pickle=False) as d:
        version = str(np.asarray(d["sidecar_version"]).item())
        if version != VOLUME_TIME_SIDECAR_VERSION:
            raise ValueError(f"volume time sidecar version {version!r} != {VOLUME_TIME_SIDECAR_VERSION!r}: {path}")
        return {k: d[k] for k in d.files}


def volume_frame_labels(case: Dict, frame_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """体场第 k 帧的 (y_raw, y_norm)。归一化用 load_case 预取的逐帧 mean/std，与 stats 文件同源。"""
    src = case.get("volume_frames")
    if src is None:
        raise KeyError("case has no volume frame source (load with timesteps='random_frame'/'time_basis')")
    mean, std = case["_volume_frame_norm"]
    k = int(frame_index)
    y_raw = src.frame(k)
    return y_raw, ((y_raw.astype(np.float64) - mean[k]) / std[k]).astype(np.float32)


def _volume_frame_norm(stats: Dict, target: str, target_normalization: str, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """逐帧归一化常数：压力 (T,)；速度 (T, 3)。global_stats 下所有帧共用峰值帧统计（历史口径）。"""
    key = "linear" if target == "pressure_mixed" else "velocity"
    if target_normalization == "frame_stats":
        frame = stats.get("frame")
        if frame is None:
            raise ValueError("target_normalization='frame_stats' needs the 'frame' block in the volume stats file")
        mean = np.asarray(frame[f"{key}_mean"], dtype=np.float64)
        std = np.asarray(frame[f"{key}_std"], dtype=np.float64)
        if len(mean) != n_frames or len(std) != n_frames:
            raise ValueError("per-frame volume statistics must cover every frame of the cycle")
        return mean, std
    mean = np.asarray(stats[key]["mean"], dtype=np.float64)
    std = np.asarray(stats[key]["std"], dtype=np.float64)
    return np.broadcast_to(mean, (n_frames,) + mean.shape).copy(), np.broadcast_to(std, (n_frames,) + std.shape).copy()


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
    # 波 2：Murray 分支流量先验按 segment id 广播到内部单元；log_tau0 用内部单元的 atlas 半径重算
    murray_keys = [k for k in C.VOLUME_EXTENDABLE_SIDECAR_KEYS if k in case]
    if murray_keys:
        if "_wall_segment_id" not in case or "vol_segment_id" not in vol:
            raise KeyError("volume Murray features need wall_segment_id and vol_segment_id")
        wall_seg = np.asarray(case["_wall_segment_id"]).astype(np.int64)
        vol_seg = np.asarray(vol["vol_segment_id"]).astype(np.int64)
        vol_radius = np.clip(np.asarray(vol["vol_local_radius"], dtype=np.float64), 1e-3, None)
        for key in murray_keys:
            wall_values = np.asarray(case[key], dtype=np.float64)[:n_wall]
            if key.startswith("log_tau0_"):
                share_key = "log_q_branch_" + key[len("log_tau0_"):]
                share = np.asarray(case[share_key], dtype=np.float64)[:n_wall] if share_key in case \
                    else wall_values + 3.0 * np.log(np.clip(np.asarray(case["local_radius"], dtype=np.float64)[:n_wall], 1e-3, None))
            else:
                share = wall_values
            table = {}
            for sid in np.unique(vol_seg):
                rows = wall_seg == sid
                if not rows.any():
                    raise ValueError(f"interior segment {sid} has no wall rows to take the Murray share from")
                table[int(sid)] = float(np.median(share[rows]))
            interior = np.array([table[int(sid)] for sid in vol_seg], dtype=np.float64)
            if key.startswith("log_tau0_"):
                interior = interior - 3.0 * np.log(vol_radius)
            case[key] = np.concatenate([wall_values, interior]).astype(np.float32)
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


def _patch_scale_kwargs(cfg) -> Dict:
    """2026-09-16 双尺度 patch 的配置转参数；旧配置（字段缺省）得到空的默认值，输出逐位不变。"""
    return {"radius_mm": float(getattr(cfg, "query_patch_radius_mm", 0.0) or 0.0),
            "radius_k": int(getattr(cfg, "query_patch_radius_k", 0) or 0),
            "offset_norm": str(getattr(cfg, "query_patch_offset_norm", "mm") or "mm")}


def subset_case(case: Dict, rows: np.ndarray) -> Dict:
    """按行切出子 case（评估分组用）；逐点数组按 rows 索引，标量原样保留。"""
    n = len(case["pos"])
    out = {}
    for k, v in case.items():
        if k in {"support_pool", "query_pool", "query_groups"}:
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[rows]
        elif k in ("y_raw_frames", "y_norm_frames") and isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == n:
            out[k] = v[:, rows]   # 多帧标签 (T, N)：按点轴切子集（抽稀视图 / 分组评估）
        elif k == "volume_frames" and isinstance(v, VolumeFrameSource):
            out[k] = v.take_rows(rows)   # 体场懒读句柄：组合行映射，子 case 读到的仍是自己的行
        else:
            out[k] = v
    return out


DENSITY_OVERRIDE_KEYS = ("curv_k1", "curv_k2", "curv_mean", "curv_gauss", "curvedness", "shape_index",
                         "curv_k1_c", "curv_k2_c", "curv_mean_c", "curv_gauss_c", "curvedness_c", "shape_index_c", "tn_dot")


def load_density_level(case: Dict, root: str | Path, level: int) -> Dict:
    """预计算抽稀云（tools/build_density_sidecars）：行索引 + 抽稀云上重算的法向/曲率族/tn_dot；按 case 缓存。"""
    cache = case.setdefault("_density", {})
    level = int(level)
    if level not in cache:
        path = Path(root) / f"L{level}" / case["unit_id"] / "features.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing density sidecar: {path}")
        with np.load(path, allow_pickle=False) as z:
            rows = np.asarray(z["rows"], dtype=np.int64)
            if rows.ndim != 1 or len(rows) == 0 or rows.max() >= len(case["pos"]) or rows.min() < 0:
                raise ValueError(f"invalid density rows: {path}")
            normals = np.asarray(z["wall_normal_pca_aligned"], dtype=np.float32)
            overrides = {"nx_aligned": normals[:, 0], "ny_aligned": normals[:, 1], "nz_aligned": normals[:, 2]}
            for name in DENSITY_OVERRIDE_KEYS:
                key = f"wall_{name}"
                if key in z.files and name in case:
                    overrides[name] = np.asarray(z[key], dtype=np.float32)
        cache[level] = {"rows": rows, "normals": normals, "overrides": overrides}
    return cache[level]


def density_view(case: Dict, level_data: Dict, level: int) -> Dict:
    """把病例换成抽稀云视图：逐点数组按行子集，云端派生输入用抽稀云上重算的值，邻域缓存全部丢弃。"""
    rows = level_data["rows"]
    sub = subset_case(case, rows)
    for key in ("_full_wall_tree", "_full_wall_features", "_fps_pool", "_fps_pool_k", "_fps_pool_size", "_density"):
        sub.pop(key, None)
    for name, values in level_data["overrides"].items():
        if name in sub:
            if len(values) != len(rows):
                raise ValueError(f"density override {name} length mismatch for {case['unit_id']}")
            sub[name] = values
    if "local_geometry" in sub:
        geometry = np.array(sub["local_geometry"], dtype=np.float32, copy=True)
        geometry[:, :3] = level_data["normals"]
        sub["local_geometry"] = geometry
    sub["_density_level"] = int(level)
    return sub


def load_noise_level(case: Dict, root: str | Path, tag: str) -> Dict:
    """预计算噪声云（tools/build_noise_sidecars）：全部点沿法向加相关噪声后的归一化位置 + 噪声云上重算的法向/曲率族/tn_dot；按 case 缓存。"""
    cache = case.setdefault("_noise", {})
    tag = str(tag)
    if tag not in cache:
        path = Path(root) / f"S{tag}" / case["unit_id"] / "features.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing noise sidecar: {path}")
        with np.load(path, allow_pickle=False) as z:
            rows = np.asarray(z["rows"], dtype=np.int64)
            if rows.ndim != 1 or len(rows) != len(case["pos"]) or not np.array_equal(rows, np.arange(len(rows))):
                raise ValueError(f"noise sidecar must cover the full cloud in order: {path}")
            pos = np.asarray(z["wall_pos_norm"], dtype=np.float32)
            if pos.shape != (len(rows), 3) or not np.isfinite(pos).all():
                raise ValueError(f"invalid noise positions: {path}")
            normals = np.asarray(z["wall_normal_pca_aligned"], dtype=np.float32)
            overrides = {"nx_aligned": normals[:, 0], "ny_aligned": normals[:, 1], "nz_aligned": normals[:, 2]}
            for name in DENSITY_OVERRIDE_KEYS:
                key = f"wall_{name}"
                if key in z.files and name in case:
                    overrides[name] = np.asarray(z[key], dtype=np.float32)
        cache[tag] = {"rows": rows, "pos": pos, "normals": normals, "overrides": overrides}
    return cache[tag]


def noise_view(case: Dict, level_data: Dict, tag: str) -> Dict:
    """把病例换成噪声云视图：点数不变，位置换成噪声后的归一化坐标，云端派生输入用噪声云上重算的值，邻域缓存全部丢弃。"""
    rows = level_data["rows"]
    sub = subset_case(case, rows)
    for key in ("_full_wall_tree", "_full_wall_features", "_fps_pool", "_fps_pool_k", "_fps_pool_size", "_density", "_noise"):
        sub.pop(key, None)
    pos = np.asarray(level_data["pos"], dtype=np.float32)
    if pos.shape != sub["pos"].shape:
        raise ValueError(f"noise positions shape mismatch for {case['unit_id']}")
    sub["pos"] = pos
    if "wall_coords_raw" in sub:
        sub["wall_coords_raw"] = (pos.astype(np.float64) * float(sub.get("coord_scale_scalar", 1.0)))
    for name, values in level_data["overrides"].items():
        if name in sub:
            if len(values) != len(rows):
                raise ValueError(f"noise override {name} length mismatch for {case['unit_id']}")
            sub[name] = values
    if "local_geometry" in sub:
        geometry = np.array(sub["local_geometry"], dtype=np.float32, copy=True)
        geometry[:, :3] = level_data["normals"]
        sub["local_geometry"] = geometry
    sub["_noise_level"] = str(tag)
    return sub


def tangent_plane_axes(geometry: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(轴向, 周向) 单位向量 [N,3] ×2：轴向 = 中心线切向投到壁面切平面，周向 = 法向 × 轴向；
    切向与法向平行时退化为与法向最不平行的坐标轴（与 attach_direction_target 同一规则，部署可得几何）。"""
    geometry = np.asarray(geometry, dtype=np.float64)
    normal = geometry[:, :3] / np.clip(np.linalg.norm(geometry[:, :3], axis=1, keepdims=True), 1e-12, None)
    tangent = geometry[:, 3:6]
    axial = tangent - (tangent * normal).sum(1, keepdims=True) * normal
    basis = np.eye(3)[np.argmin(np.abs(normal), axis=1)]
    fallback = basis - (basis * normal).sum(1, keepdims=True) * normal
    axial = np.where(np.linalg.norm(axial, axis=1, keepdims=True) < 1e-6, fallback, axial)
    axial = axial / np.clip(np.linalg.norm(axial, axis=1, keepdims=True), 1e-12, None)
    return axial, np.cross(normal, axial)


def cycle_mean_vector_ratios(vec80: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    """帧 0–79 平均 WSS 向量在 (轴向, 周向) 上的分量 ÷ 平均幅值 (1/80)Σ‖τ‖ → (N, 2)，范数 ≤ 1；vec80 须已在对齐坐标架。"""
    vec80 = np.asarray(vec80, dtype=np.float64)
    if vec80.ndim != 3 or vec80.shape[0] != 80 or vec80.shape[2] != 3:
        raise ValueError(f"expected (80, N, 3) wall shear vectors, got {vec80.shape}")
    mean = vec80.mean(axis=0)
    mag = np.linalg.norm(vec80, axis=2).mean(axis=0)
    axial, circ = tangent_plane_axes(geometry)
    denom = np.clip(mag, 1e-12, None)
    return np.stack([(mean * axial).sum(1) / denom, (mean * circ).sum(1) / denom], axis=1)


def derived_osi(ratio_axial: np.ndarray, ratio_circ: np.ndarray) -> np.ndarray:
    """OSI = ½(1 − min(1, ‖(r_a, r_c)‖))：与 OSI 定义同式，只是 ‖Σ τ‖ / Σ‖τ‖ 由切平面两分量给出。"""
    r = np.sqrt(np.asarray(ratio_axial, dtype=np.float64) ** 2 + np.asarray(ratio_circ, dtype=np.float64) ** 2)
    return 0.5 * (1.0 - np.clip(r, 0.0, 1.0))


def multi_aux_columns(bundle_path: str | Path, aux: Tuple[str, ...], cycle_view_root: str | Path | None,
                      cohort_rel: str, case_name: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """M1 辅助通道原始标签（2026-09-25，data.multi_aux_channels）。返回 ({通道: (N,)}, {'local_geometry': …} 或 {})。
    rev_frac 取 cycle.npz；mean_axial / mean_circ 用帧 0–79 wall_wss_vec（旋到对齐架）与 case_geometry 的法向 / 中心线切向。"""
    bundle_path = Path(bundle_path)
    with np.load(bundle_path, allow_pickle=True) as z:
        node_id = np.asarray(z["wall_node_id_cas"]).astype(np.int64)
        pos = z["wall_coords_norm"].astype(np.float32)
        need_vec = any(a in ("mean_axial", "mean_circ") for a in aux)
        if need_vec:
            steps = z["steps"].tolist()
            if len(steps) != 81 or steps[0] != 1120 or steps[-1] != 1280:
                raise ValueError(f"{bundle_path}: cycle vector labels need the 81-frame 1120..1280 layout")
            vec80 = np.asarray(z["wall_wss_vec"][:80], dtype=np.float64) @ np.asarray(z["transform_rotation"], dtype=np.float64).T
    cols: Dict[str, np.ndarray] = {}
    geo: Dict = {}
    if "rev_frac" in aux:
        cols["rev_frac"] = load_cycle_label(cycle_view_root, cohort_rel, case_name, "rev_frac", node_id)
    if need_vec:
        holder = {"bundle_path": str(bundle_path), "pos": pos}
        ratios = cycle_mean_vector_ratios(vec80, case_geometry(holder))
        cols["mean_axial"], cols["mean_circ"] = ratios[:, 0].astype(np.float32), ratios[:, 1].astype(np.float32)
        geo = {"local_geometry": holder["local_geometry"], "local_geometry_source": holder["local_geometry_source"]}
    return cols, geo


def load_case(cohort_rel: str, case_name: str, wss_stats: Dict,
              target: str = "wss", target_normalization: str = "global_stats",
              data_root: str | Path = C.DATA_ROOT,
              required_frame_version: str | None = None,
              case_features: Dict[str, float] | None = None,
              timesteps: str = "peak", waveform_path: str | Path | None = None,
              extra_point_features: Tuple[str, ...] = (),
              point_features_root: str | Path | None = None,
              time_basis_path: str | Path | None = None, time_basis_k: int = 0,
              volume_time_sidecar_root: str | Path | None = None,
              volume_h5_root: str | Path | None = None,
              cycle_view_root: str | Path | None = None,
              multi_aux_channels: Tuple[str, ...] = ()) -> Dict:
    p = Path(data_root) / cohort_rel / case_name / "bundle.npz"
    multi_aux_channels = tuple(multi_aux_channels or ())
    aux_geometry: Dict = {}
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
        elif target in C.CYCLE_TARGETS:
            # 周期积分量（wss_min_cycle_v1）：逐点标签按视图行序存放，用 wall_node_id_cas 校验后直接取用
            y_raw = load_cycle_label(cycle_view_root, cohort_rel, case_name, target,
                                     np.asarray(d["wall_node_id_cas"]).astype(np.int64))
        elif target == C.MULTI_TARGET:
            # M1 三头：[峰值帧 WSS, TAWSS, OSI] 逐点拼成 (N, 3)；统计量文件 method='multi' 逐列归一化
            node_id = np.asarray(d["wall_node_id_cas"]).astype(np.int64)
            columns = {"wss": d["wall_wss"][si].astype(np.float32)}
            for name in C.MULTI_CHANNELS[1:]:
                columns[name] = load_cycle_label(cycle_view_root, cohort_rel, case_name, name, node_id)
            channels = tuple(C.MULTI_CHANNELS) + multi_aux_channels
            if multi_aux_channels:
                aux_cols, aux_geometry = multi_aux_columns(p, multi_aux_channels, cycle_view_root, cohort_rel, case_name)
                columns.update(aux_cols)
            if tuple(wss_stats.get("channels", ())) != channels or wss_stats.get("method") != "multi":
                raise ValueError(f"wss_cycle_multi requires a method='multi' statistics file with channels {channels}")
            y_raw = np.stack([columns[name] for name in channels], axis=1).astype(np.float32)
        else:
            raise ValueError(f"unsupported target={target!r} (expect 'wss'|'pressure'|{C.CYCLE_TARGETS}|{C.VOLUME_TARGETS})")
        if volume is None:
            # residual-target stats: provisional standard log_z here, replaced after the offset feature is attached
            provisional = ({k: v for k, v in wss_stats.items() if k != "offset"} if "offset" in wss_stats else wss_stats)
            if target_normalization == "frame_stats":
                if target != "wss" or timesteps == "peak":
                    raise ValueError("target_normalization='frame_stats' requires target='wss' in a multi-frame mode")
                y_norm, _ = normalize_target(y_raw, frame_stats_view(provisional, si), "global_stats")
                target_norm_meta = {"mode": "frame_stats", "stats_source": "train-only per-frame log statistics",
                                    "peak_frame_index": int(si)}
            else:
                y_norm, target_norm_meta = normalize_target(y_raw, provisional, target_normalization)
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
        if "wall_segment_id" in d.files:
            case["_wall_segment_id"] = np.asarray(d["wall_segment_id"]).astype(np.int64)
        if aux_geometry:
            case.update(aux_geometry)   # 与 case_geometry 懒算结果相同，只省一次 h5 读取
        if (target in C.CYCLE_TARGETS or target == C.MULTI_TARGET) and "wall_semantic_id" in d.files:
            # 周期积分量落地一致性块按语义血管段（trunk / CIA / 四末支）报段均值误差；只在周期目标下附带，旧目标的 case 逐位不变
            case["_wall_semantic_id"] = np.asarray(d["wall_semantic_id"]).astype(np.int64)
        if extra_point_features:
            _attach_v6_point_features(case, d, tuple(extra_point_features),
                                      point_features_root, cohort_rel, case_name)
        if volume is None and "offset" in wss_stats and target == "wss":
            feature = wss_stats["offset"]["feature"]
            if feature not in case:
                raise KeyError(f"residual target needs sidecar feature {feature!r} loaded for {cohort_rel}/{case_name}")
            case["y_offset"] = np.asarray(case[feature], dtype=np.float64)
            case["y_norm"] = normalize_wss(y_raw, wss_stats, offset=case["y_offset"]).astype(np.float32)
            target_norm_meta = {**target_norm_meta, "residual_offset_feature": feature}
            case["target_normalization_meta"] = target_norm_meta
        if volume is not None:
            _attach_volume(case, volume, target, wss_stats)
        if volume is not None and timesteps != "peak":
            # 体场全周期：81 帧标签懒读（case.h5），只在 case 上放一个句柄 + 逐帧归一化常数
            if target not in ("pressure_mixed", "velocity"):
                raise ValueError(f"multi-frame volume targets support 'pressure_mixed'|'velocity', got {target!r}")
            if not volume_time_sidecar_root or not volume_h5_root:
                raise ValueError("multi-frame volume targets require volume_time_sidecar_root and volume_h5_root")
            side = load_volume_time_sidecar(volume_time_sidecar_root, cohort_rel, case_name)
            if not np.array_equal(side["steps"].astype(np.int64), np.asarray(steps, dtype=np.int64)):
                raise ValueError(f"volume time sidecar steps differ from bundle steps: {cohort_rel}/{case_name}")
            n_wall = int(case["n_wall"])
            n_vol = len(case["pos"]) - n_wall
            if int(side["n_cells"]) != n_vol:
                raise ValueError(f"volume time sidecar cell count differs from the volume view: {cohort_rel}/{case_name}")
            src = VolumeFrameSource(
                Path(volume_h5_root) / f"{cohort_rel}/{case_name}".replace("/", "__") / "case.h5", target,
                n_wall=n_wall, n_vol=n_vol,
                wall_nearest_cell_index=side["wall_nearest_cell_index"],
                p_volume_mean_pa=side["p_volume_mean_pa"], rotation=d["transform_rotation"],
                steps=np.asarray(steps, dtype=np.int64))
            case["volume_frames"] = src
            case["_volume_frame_norm"] = _volume_frame_norm(wss_stats, target, target_normalization, len(steps))
            case["steps"] = np.asarray(steps, dtype=np.int64)
            case["peak_index"] = int(si)
            case["frame_index"] = int(si)
            case["n_frames"] = len(steps)
            case["waveform"] = load_waveform(waveform_path) if waveform_path else None
            case["y_raw"], case["y_norm"] = volume_frame_labels(case, si)   # 峰值帧：行布局与单帧 case 一致
            case["target_normalization_meta"] = {**case["target_normalization_meta"],
                                                 "mode": target_normalization, "timesteps": timesteps,
                                                 "volume_frame_reference": "p_rel(x,t) = p(x,t) - p_volume_mean(t)",
                                                 "volume_time_sidecar_version": VOLUME_TIME_SIDECAR_VERSION}
            if timesteps == "time_basis":
                if target_normalization != "frame_stats":
                    raise ValueError("timesteps='time_basis' requires target_normalization='frame_stats'")
                if not time_basis_path or int(time_basis_k) < 1:
                    raise ValueError("timesteps='time_basis' requires time_basis_path and time_basis_k>=1")
                basis = TB.load_time_basis(time_basis_path, int(time_basis_k))
                TB.check_stats(basis, wss_stats)
                if not np.array_equal(basis["steps"], np.asarray(steps, dtype=np.int64)):
                    raise ValueError(f"time basis steps differ from bundle steps: {cohort_rel}/{case_name}")
                expected = 3 if target == "velocity" else 1
                if int(basis["channels"]) != expected:
                    raise ValueError(f"time basis has {basis['channels']} channels but target={target!r} needs {expected}")
                mean, std = case["_volume_frame_norm"]
                y_frames = src.frames().astype(np.float64)                  # (T, N[, 3])
                if target == "pressure_mixed":
                    z = (y_frames - mean[:, None]) / std[:, None]
                else:
                    z = (y_frames - mean[:, None, :]) / std[:, None, :]
                del y_frames
                case["y_norm"] = TB.project(z, basis).astype(np.float32)    # (N, C*(K+1))
                del z
                case["time_basis"] = basis
                case["target_normalization_meta"] = {**case["target_normalization_meta"],
                                                     "time_basis": basis["path"],
                                                     "time_basis_sha256": basis["sha256"],
                                                     "time_basis_k": basis["k"],
                                                     "time_basis_channels": basis["channels"]}
        if timesteps == "random_frame" and volume is None:
            if target != "wss":
                raise ValueError("timesteps='random_frame' only supports target='wss' on the wall view")
            frames = d["wall_wss"].astype(np.float32)  # (81, N)
            wf = load_waveform(waveform_path) if waveform_path else None
            if wf is None:
                raise ValueError("timesteps='random_frame' requires waveform_path")
            if not np.array_equal(wf["_steps"], np.asarray(steps, dtype=np.int64)):
                raise ValueError(f"waveform steps differ from bundle steps: {cohort_rel}/{case_name}")
            case["y_raw_frames"] = frames
            case["y_norm_frames"] = _normalize_frames(frames, wss_stats, target_normalization)
            case["steps"] = np.asarray(steps, dtype=np.int64)
            case["peak_index"] = int(si)
            case["waveform"] = wf
            select_frame(case, int(si))  # 默认停在峰值帧：不改帧时与单帧 case 完全一致
        if timesteps == "time_basis" and volume is None:
            if target != "wss" or target_normalization != "frame_stats":
                raise ValueError("timesteps='time_basis' on the wall view requires target='wss' with target_normalization='frame_stats'")
            if not time_basis_path or int(time_basis_k) < 1:
                raise ValueError("timesteps='time_basis' requires time_basis_path and time_basis_k>=1")
            basis = TB.load_time_basis(time_basis_path, int(time_basis_k))
            TB.check_stats(basis, wss_stats)
            if not np.array_equal(basis["steps"], np.asarray(steps, dtype=np.int64)):
                raise ValueError(f"time basis steps differ from bundle steps: {cohort_rel}/{case_name}")
            frames = d["wall_wss"].astype(np.float32)  # (81, N)
            y_norm_frames = _normalize_frames(frames, wss_stats, target_normalization)
            case["y_raw_frames"] = frames
            case["y_norm_frames"] = y_norm_frames
            case["y_norm"] = TB.project(y_norm_frames.astype(np.float64), basis).astype(np.float32)  # (N, K+1)
            case["steps"] = np.asarray(steps, dtype=np.int64)
            case["peak_index"] = int(si)
            case["frame_index"] = int(si)
            case["time_basis"] = basis
            case["waveform"] = load_waveform(waveform_path) if waveform_path else None
            case["target_normalization_meta"] = {**case["target_normalization_meta"], "time_basis": basis["path"],
                                                 "time_basis_sha256": basis["sha256"], "time_basis_k": basis["k"]}
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
                   point_features_root: str | Path | None = None,
                   time_basis_path: str | Path | None = None, time_basis_k: int = 0,
                   volume_time_sidecar_root: str | Path | None = None,
                   volume_h5_root: str | Path | None = None,
                   cycle_view_root: str | Path | None = None,
                   multi_aux_channels: Tuple[str, ...] = ()) -> List[Dict]:
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
            time_basis_path=time_basis_path, time_basis_k=time_basis_k,
            volume_time_sidecar_root=volume_time_sidecar_root, volume_h5_root=volume_h5_root,
            cycle_view_root=cycle_view_root, multi_aux_channels=multi_aux_channels,
        ))
        if set(extra_point_features) & set(LG.FEATURE_KEYS):
            cases[-1]["_longitudinal_partition"] = partition
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


def attach_direction_target(case: Dict) -> np.ndarray:
    """切平面 WSS 方向标签 [N,2]：峰值帧 wall_wss_vec 旋到对齐坐标架后投到 (轴向, 周向) 并单位化。

    轴向 = 中心线切向在壁面切平面内的投影，周向 = 法向 × 轴向（与 local_refinement.tangent_frame 同一规则，
    含切向与法向平行时的退化处理）。切平面内幅值为 0 的点给 (0, 0)，损失里按此掩掉。只在 data.direction_target 时调用。
    """
    if "dir_target" in case:
        return case["dir_target"]
    geometry = np.asarray(case_geometry(case), dtype=np.float64)
    with np.load(case["bundle_path"], allow_pickle=True) as z:
        if "wall_wss_vec" not in z.files:
            raise KeyError(f"direction target requires wall_wss_vec in the bundle: {case['bundle_path']}")
        steps = z["steps"].tolist()
        vec = np.asarray(z["wall_wss_vec"][steps.index(int(z["peak_step"]))], dtype=np.float64)
        rotation = np.asarray(z["transform_rotation"], dtype=np.float64)
    if vec.shape != (len(case["pos"]), 3):
        raise ValueError("direction target requires the original full wall case")
    vec = vec @ rotation.T  # V5 视图：aligned = (raw - c) @ R.T，向量同样 @ R.T（与 wall_normal_pca_aligned 一致）
    normal = geometry[:, :3] / np.clip(np.linalg.norm(geometry[:, :3], axis=1, keepdims=True), 1e-12, None)
    tangent = geometry[:, 3:6]
    axial = tangent - (tangent * normal).sum(1, keepdims=True) * normal
    basis = np.eye(3)[np.argmin(np.abs(normal), axis=1)]
    fallback = basis - (basis * normal).sum(1, keepdims=True) * normal
    axial = np.where(np.linalg.norm(axial, axis=1, keepdims=True) < 1e-6, fallback, axial)
    axial = axial / np.clip(np.linalg.norm(axial, axis=1, keepdims=True), 1e-12, None)
    circ = np.cross(normal, axial)
    comps = np.stack([(vec * axial).sum(1), (vec * circ).sum(1)], axis=1)
    mag = np.linalg.norm(comps, axis=1, keepdims=True)
    unit = np.where(mag > 1e-9, comps / np.clip(mag, 1e-12, None), 0.0)
    case["dir_target"] = np.ascontiguousarray(unit, dtype=np.float32)
    full = np.linalg.norm(vec, axis=1)
    case["dir_target_report"] = {
        "in_plane_fraction_median": float(np.median(mag[:, 0] / np.clip(full, 1e-12, None))),
        "reversed_axial_fraction": float(np.mean(comps[:, 0] < 0)),
    }
    return case["dir_target"]


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
        if training and (float(getattr(cfg, "density_aug_prob", 0.0) or 0.0) > 0.0 or float(getattr(cfg, "noise_aug_prob", 0.0) or 0.0) > 0.0):
            # 抽稀视图不能再从 bundle 推导原始几何/截面元数据，先在完整病例上算好并缓存
            for case in cases:
                if getattr(cfg, "local_geometry", False):
                    case_geometry(case)
                if getattr(cfg, "section_tokens", False):
                    case_section(case)
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
            if getattr(cfg, "direction_target", False):
                attach_direction_target(c)
            if getattr(cfg, "section_tokens", False):
                case_section(c)
            if int(getattr(cfg, "query_patch_nsample", 0)) > 0:
                full_wall_tree(c)
                if getattr(cfg, "timesteps", "peak") == "peak":
                    # Workers are recreated each epoch in the frozen protocol.
                    # Prime shared read-only features before fork, avoiding
                    # repeated full-wall transformations inside every worker.
                    build_query_patch(c, np.empty(0, dtype=np.int64), cfg.input_features,
                                      feat_stats, nsample=int(cfg.query_patch_nsample), **_patch_scale_kwargs(cfg))
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
        noise_prob = float(getattr(self.cfg, "noise_aug_prob", 0.0) or 0.0)
        noise_drawn = False
        if self.training and noise_prob > 0.0:
            # 边界噪声增广（2026-09-16）：独立抽签流 "noise"；抽中则整例换成预计算噪声云视图，本轮不再叠加密度增广
            rng_noise = np.random.default_rng(S.stable_seed(self.base_seed, self.epoch, case["unit_id"], "noise"))
            if rng_noise.random() < noise_prob:
                tags = tuple(str(t) for t in self.cfg.noise_aug_levels)
                tag = tags[int(rng_noise.integers(len(tags)))]
                case = noise_view(case, load_noise_level(case, self.cfg.noise_aug_root, tag), tag)
                noise_drawn = True
        density_prob = float(getattr(self.cfg, "density_aug_prob", 0.0) or 0.0)
        if self.training and density_prob > 0.0 and not noise_drawn:
            # 密度增广：按 (run seed, epoch, case) 确定性抽签；抽中则整例换成预计算抽稀云视图
            rng = np.random.default_rng(S.stable_seed(self.base_seed, self.epoch, case["unit_id"], "density"))
            if rng.random() < density_prob:
                levels = tuple(int(l) for l in self.cfg.density_aug_levels)
                level = levels[int(rng.integers(len(levels)))]
                case = density_view(case, load_density_level(case, self.cfg.density_aug_root, level), level)
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
            if "y_norm_frames" in case:
                y_norm_src, y_raw_src = case["y_norm_frames"][frame_index], case["y_raw_frames"][frame_index]
            else:
                # 体场：该帧标签按需从 case.h5 读（整块 81 帧放不进内存）
                y_raw_src, y_norm_src = volume_frame_labels(case, frame_index)
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
        if getattr(self.cfg, "direction_target", False):
            result["y_dir"] = torch.from_numpy(np.ascontiguousarray(attach_direction_target(case)[query_idx]))
        if getattr(self.cfg, "section_tokens", False):
            section = case_section(case)
            result["support_section"] = torch.from_numpy(np.ascontiguousarray(section[support_idx]))
            result["query_section"] = torch.from_numpy(np.ascontiguousarray(section[query_idx]))
        patch_k = int(getattr(self.cfg, "query_patch_nsample", 0))
        if patch_k > 0:
            patch = build_query_patch(case, query_idx, self.cfg.input_features, self.feat_stats,
                                      nsample=patch_k, frame_index=frame_index, **_patch_scale_kwargs(self.cfg))
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
    for key in ("support_geometry", "query_geometry", "support_section", "query_section", "y_dir"):
        if any(key in item for item in batch):
            if not all(key in item for item in batch):
                raise ValueError(f"cannot collate mixed {key} availability")
            result[key] = torch.cat([item[key] for item in batch], dim=0)
    if any("query_patch" in item for item in batch):
        if not all("query_patch" in item for item in batch):
            raise ValueError("cannot collate mixed query patch availability")
        # 2026-09-16: 按首个样本的键遍历（双尺度 patch 多出 radius_features / radius_relative_mm / radius_mask；旧 patch 仍只有两键）
        result["query_patch"] = {key: torch.cat([item["query_patch"][key] for item in batch], dim=0)
                                 for key in batch[0]["query_patch"]}
    if any("velocity_mask" in item or "pressure_mask" in item for item in batch):
        if not all("velocity_mask" in item and "pressure_mask" in item for item in batch):
            raise ValueError("cannot collate mixed joint and single-task supervision")
        result.update({key: torch.cat([item[key] for item in batch])
                       for key in ("velocity_mask", "pressure_mask")})
    return result


def local_model_context(batch: Dict, device) -> Dict:
    """Move only opted-in geometry tensors; legacy model calls stay unchanged."""
    out = {key: batch[key].to(device)
           for key in ("support_geometry", "query_geometry", "support_section", "query_section") if key in batch}
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
