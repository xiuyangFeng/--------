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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
def load_split_cases(split_path: str | Path, partition: str) -> List[Tuple[str, str]]:
    """返回 [(cohort_rel, case_name)]，cohort_rel 形如 'AG/fast'。"""
    sp = json.loads(Path(split_path).read_text())
    key = partition if partition.endswith("_cases") else f"{partition}_cases"
    out: List[Tuple[str, str]] = []
    for label in sp.get(key, []):
        subset, case = label.split("/", 1)          # 'fast/XXX'
        out.append((f"AG/{subset}", case))
    return out


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


def sample_indices(case: Dict, cfg: C.DataConfig, seed: int) -> np.ndarray:
    n = len(case["pos"])
    k = cfg.wall_n_points
    if k <= 0 or k >= n:
        return np.arange(n)
    if cfg.sampling == "random":
        return np.random.default_rng(seed).choice(n, size=k, replace=False)
    if cfg.sampling == "geom_weighted":
        w = geom_sampling_weights(case, cfg)
        return np.random.default_rng(seed).choice(n, size=k, replace=False, p=w)
    # fps（确定性）：缓存全序，取前 k
    if "_fps_order" not in case:
        case["_fps_order"] = farthest_point_sample(case["pos"], min(n, max(k, 1)), seed)
    order = case["_fps_order"]
    if len(order) >= k:
        return order[:k]
    return order  # 理论不会走到


# ---------------------------------------------------------------------------
# 特征标准化（几何输入列用 train 统计 z-score；x,y,z 保持归一化坐标）
# ---------------------------------------------------------------------------
GEOM_FEATURE_KEYS = ("dist_to_wall", "abscissa_norm", "local_radius", "curvature")


def compute_feature_stats(cases: List[Dict], input_features: Tuple[str, ...]) -> Dict:
    stats: Dict[str, Dict[str, float]] = {}
    for f in input_features:
        if f in ("x", "y", "z"):
            continue
        vals = np.concatenate([np.asarray(c[f], dtype=np.float64).ravel() for c in cases])
        vals = vals[np.isfinite(vals)]
        if f == "curvature":  # 端点数值尖峰，先 clip 再统计
            hi = np.percentile(np.abs(vals), 99)
            vals = np.clip(vals, -hi, hi)
            stats[f] = {"mean": float(vals.mean()), "std": float(vals.std() + 1e-6),
                        "clip": float(hi)}
        else:
            stats[f] = {"mean": float(vals.mean()), "std": float(vals.std() + 1e-6)}
    return stats


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
            v = np.asarray(case[f], dtype=np.float64)[idx]
            st = feat_stats[f]
            if "clip" in st:
                v = np.clip(v, -st["clip"], st["clip"])
            v = (v - st["mean"]) / st["std"]
            cols.append(v)
    return np.stack(cols, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def load_case(cohort_rel: str, case_name: str, wss_stats: Dict) -> Dict:
    p = C.DATA_ROOT / cohort_rel / case_name / "bundle.npz"
    with np.load(p, allow_pickle=True) as d:
        steps = d["steps"].tolist()
        peak = int(d["peak_step"])
        si = steps.index(peak)
        pos = d["wall_coords_norm"].astype(np.float32)
        wss_raw = d["wall_wss"][si].astype(np.float32)
        case = dict(
            cohort=cohort_rel, case=case_name,
            pos=pos,
            y_raw=wss_raw,
            y_norm=normalize_wss(wss_raw, wss_stats).astype(np.float32),
            dist_to_wall=d["wall_dist_to_wall"].astype(np.float32),
            abscissa_norm=d["wall_abscissa_norm"].astype(np.float32),
            local_radius=d["wall_local_radius"].astype(np.float32),
            curvature=d["wall_curvature"].astype(np.float32),
            peak_step=peak,
            coord_scale=float(d["coord_scale"]),
        )
    return case


def load_partition(split_path: str, partition: str, wss_stats: Dict) -> List[Dict]:
    cases = []
    for cohort_rel, case_name in load_split_cases(split_path, partition):
        p = C.DATA_ROOT / cohort_rel / case_name / "bundle.npz"
        if not p.is_file():
            continue
        cases.append(load_case(cohort_rel, case_name, wss_stats))
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

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i: int):
        case = self.cases[i]
        if self.training and self.cfg.resample_each_epoch and self.cfg.sampling != "fps":
            seed = self.base_seed + 100003 * self.epoch + 7919 * i
        else:
            seed = case["geom_seed"]
        idx = sample_indices(case, self.cfg, seed)
        pos_sel = case["pos"][idx]
        if self.training and getattr(self.cfg, "rot_aug", False):
            pos_sel = (pos_sel @ random_rotation(seed + 31)).astype(np.float32)
        feat = build_features(case, idx, self.cfg.input_features, self.feat_stats,
                              pos_override=pos_sel)
        return {
            "pos": torch.from_numpy(np.ascontiguousarray(pos_sel)),
            "x": torch.from_numpy(feat),
            "y": torch.from_numpy(case["y_norm"][idx]),
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
    batch_idx = torch.cat([
        torch.full((len(b["pos"]),), i, dtype=torch.long) for i, b in enumerate(batch)
    ])
    curv = torch.cat([b["curv"] for b in batch], dim=0)
    invr = torch.cat([b["invr"] for b in batch], dim=0)
    return {"pos": pos, "x": x, "y": y, "batch": batch_idx, "curv": curv, "invr": invr,
            "cases": [b["case"] for b in batch]}


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


if __name__ == "__main__":
    stats = load_wss_stats()
    tr = load_partition(str(C.DEFAULT_SPLIT), "train", stats)
    print(f"train cases: {len(tr)}  例点数 min/max:",
          min(len(c['pos']) for c in tr), max(len(c['pos']) for c in tr))
    fs = compute_feature_stats(tr, ("x", "y", "z", "dist_to_wall", "curvature"))
    print("feat_stats keys:", list(fs.keys()))
    ds = WSSMinDataset(tr, C.DataConfig(wall_n_points=2000, input_features=("x", "y", "z")),
                       fs, training=True)
    b = collate([ds[0], ds[1]])
    print("batch pos", b["pos"].shape, "x", b["x"].shape, "y", b["y"].shape,
          "batch", b["batch"].shape, "n_graphs", int(b["batch"].max()) + 1)
