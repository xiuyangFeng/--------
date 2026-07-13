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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C


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


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def load_split(split_path: str | Path) -> Dict:
    return json.loads(Path(split_path).read_text())


def load_split_cases(split_path: str | Path, partition: str) -> List[Tuple[str, str]]:
    """返回 [(cohort_rel, case_name)]，cohort_rel 形如 'AG/fast'。"""
    sp = load_split(split_path)
    key = partition if partition.endswith("_cases") else f"{partition}_cases"
    out: List[Tuple[str, str]] = []
    for label in sp.get(key, []):
        subset, case = label.split("/", 1)          # 'fast/XXX'
        out.append((f"AG/{subset}", case))
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
                   run_seed: int = 0) -> np.ndarray:
    n = len(case["pos"])
    k = cfg.wall_n_points
    if k <= 0 or k >= n:
        return np.arange(n)
    if cfg.sampling == "random":
        return np.random.default_rng(seed).choice(n, size=k, replace=False)
    if cfg.sampling == "geom_weighted":
        w = geom_sampling_weights(case, cfg)
        return np.random.default_rng(seed).choice(n, size=k, replace=False, p=w)
    if cfg.sampling == "fps_multistart":
        pool_size = max(1, int(cfg.fps_pool_size))
        if "_fps_pool" not in case or case.get("_fps_pool_k") != min(n, k):
            case["_fps_pool"] = build_fps_pool(
                case["pos"], k, pool_size, case.get("geom_seed", run_seed)
            )
            case["_fps_pool_k"] = min(n, k)
            case["_fps_pool_size"] = pool_size
        pool = case["_fps_pool"]
        pool_id = int((run_seed + 100003 * epoch + 7919 * case_index) % len(pool))
        return pool[pool_id].copy()
    # fps（确定性）：缓存长度-k 子集（非全序）
    cache_key = "_fps_subset"
    if cache_key not in case or case.get("_fps_subset_k") != min(n, k):
        case[cache_key] = farthest_point_sample(case["pos"], min(n, max(k, 1)), seed)
        case["_fps_subset_k"] = min(n, k)
    return case[cache_key].copy()


# ---------------------------------------------------------------------------
# 特征标准化（几何输入列用 train 统计 z-score；x,y,z 保持归一化坐标）
# ---------------------------------------------------------------------------
GEOM_FEATURE_KEYS = ("abscissa_norm", "local_radius", "curvature", "coord_scale",
                     "radius_gradient")


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
              target: str = "wss") -> Dict:
    p = C.DATA_ROOT / cohort_rel / case_name / "bundle.npz"
    with np.load(p, allow_pickle=True) as d:
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
        case = dict(
            cohort=cohort_rel, case=case_name,
            pos=pos,
            y_raw=y_raw,
            y_norm=normalize_wss(y_raw, wss_stats).astype(np.float32),
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
                   strict: bool = True, target: str = "wss") -> List[Dict]:
    labels = load_split_cases(split_path, partition)
    cases = []
    missing = []
    for cohort_rel, case_name in labels:
        p = C.DATA_ROOT / cohort_rel / case_name / "bundle.npz"
        if not p.is_file():
            missing.append(f"{cohort_rel}/{case_name}")
            continue
        cases.append(load_case(cohort_rel, case_name, wss_stats, target=target))
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
        # 给每个 case 一个稳定的 fps seed
        for i, c in enumerate(cases):
            c["geom_seed"] = base_seed + 7919 * i
        # 预热 fps_multistart pool，避免首个 epoch 在 worker 内重复计算
        if training and cfg.sampling == "fps_multistart":
            for i, c in enumerate(cases):
                sample_indices(c, cfg, c["geom_seed"], epoch=0, case_index=i,
                               run_seed=base_seed)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i: int):
        case = self.cases[i]
        sampling = self.cfg.sampling
        if self.training and self.cfg.resample_each_epoch and sampling in (
            "random", "geom_weighted", "fps_multistart"
        ):
            seed = self.base_seed + 100003 * self.epoch + 7919 * i
        else:
            seed = case["geom_seed"]
        idx = sample_indices(
            case, self.cfg, seed,
            epoch=self.epoch if self.training else 0,
            case_index=i,
            run_seed=self.base_seed,
        )
        pos_sel = case["pos"][idx]
        if self.training and getattr(self.cfg, "rot_aug", False):
            pos_sel = (pos_sel @ random_rotation(seed + 31)).astype(np.float32)
        feat = build_features(case, idx, self.cfg.input_features, self.feat_stats,
                              pos_override=pos_sel)
        return {
            "pos": torch.from_numpy(np.ascontiguousarray(pos_sel)),
            "x": torch.from_numpy(feat),
            "y": torch.from_numpy(case["y_norm"][idx]),
            "y_raw": torch.from_numpy(case["y_raw"][idx]),
            # 供几何加权 loss 用（曲率、1/局部半径）
            "curv": torch.from_numpy(np.abs(case["curvature"][idx]).astype(np.float32)),
            "invr": torch.from_numpy((1.0 / np.clip(case["local_radius"][idx], 1e-6, None)).astype(np.float32)),
            "case": case["case"], "cohort": case["cohort"],
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
    curv = torch.cat([b["curv"] for b in batch], dim=0)
    invr = torch.cat([b["invr"] for b in batch], dim=0)
    return {"pos": pos, "x": x, "y": y, "y_raw": y_raw, "batch": batch_idx,
            "curv": curv, "invr": invr, "cases": [b["case"] for b in batch]}


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
