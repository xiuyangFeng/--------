#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高 WSS 优化方案的第一性原理诊断（只读，不训练）。

支撑 `docs/02-推进与变更/WSS高值区域预测优化方案.md` §13 的全部数字。
四组量，全部直接从 `data_wss_min/**/bundle.npz` 测得：

1. **协议天花板**：用真值在 `--support` 个支撑点上"完美预测"，再走模型实际的
   3-NN 反距离插值到全点云。这是当前 support/query 协议下任何模型的上界
   （`baseline_models.PointNetPlusPlusRegressor.decode_query` 用
   `torch_geometric.nn.knn_interpolate`，权重 1/d^2 归一化，这里逐位复现）。
2. **标签稳定性**：8 近邻平滑后热点是否保持、相邻心动时相热点是否漂移。
   回答"尾部标签是否受局部网格/插值影响"与"峰值选步是否是噪声源"。
3. **方差分解**：log(WSS) 的病例间 / 病例内方差比，给一切病例级条件信号
   （出口 RCR、cohort、病例尺度头）的收益定上界。
4. **Poiseuille 先验**：逐例 log(WSS) ~ a + b*log(local_radius) 的指数与 R²，
   衡量一个尚未进入网络的解析特征能解释多少病例内方差。

用法::

    /usr/bin/python3 training_wss_min/tools/diagnose_high_wss_ceilings.py \
        --n-per-cohort 8 --out preflight/high_wss_ceilings.json

注意：`--n-per-cohort` 抽样是分层的（每 cohort 独立抽），seed 固定，结论量级稳健；
要全库复现请传 `--n-per-cohort 0`（较慢，192 例约 30-60 min）。
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOB = "data_wss_min/*/*/*/bundle.npz"
# 本 run 实际使用的 train-only 统计；标准化空间天花板必须用同一份
DEFAULT_STATS = (
    "data_wss_min/fold_stats/q2v_ilo_20260718/q2v_pool2025_control106_global_stats.json"
)


def knn_interpolate_numpy(support_pos, support_val, query_pos, k=3):
    """复现 torch_geometric.nn.knn_interpolate：权重 1/d^2，在 k 个近邻上归一化。"""
    tree = cKDTree(support_pos)
    dist, idx = tree.query(query_pos, k=k)
    dist = np.maximum(dist, 1e-16)
    weight = 1.0 / dist ** 2
    weight /= weight.sum(axis=1, keepdims=True)
    return (support_val[idx] * weight).sum(axis=1)


def top_mask(values: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    return values >= np.percentile(values, 100 * (1 - fraction))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((true - pred) ** 2) / ss_tot)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = float((a | b).sum())
    return float((a & b).sum() / union) if union else float("nan")


def diagnose_case(path: str, support_n: int, knn: int, stats: dict) -> dict:
    bundle = np.load(path, allow_pickle=True)
    pos = bundle["wall_coords_norm"].astype(np.float64)
    raw = bundle["wall_coords_raw"].astype(np.float64)
    n = len(pos)
    steps = list(bundle["steps"])
    peak_i = steps.index(int(bundle["peak_step"]))
    wss_all = bundle["wall_wss"]
    y = wss_all[peak_i].astype(np.float64)
    local_radius = bundle["wall_local_radius"].astype(np.float64)

    row = {
        "case": path.split("data_wss_min/")[1].rsplit("/bundle", 1)[0],
        "cohort": path.split("/")[-4],
        "n_wall": int(n),
        "n_steps": int(len(steps)),
        "diag_mm": float(np.linalg.norm(raw.max(0) - raw.min(0))),
    }

    # --- 1. 协议天花板：真值在 support 点上完美，再插值到全点云 ---
    k_sup = min(support_n, n)
    # hashlib 而非内置 hash()：后者按进程加盐，会让支撑点抽样跨运行不可复现
    case_seed = int(hashlib.sha256(row["case"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(case_seed)
    sup = rng.choice(n, size=k_sup, replace=False)
    y_hat = knn_interpolate_numpy(pos[sup], y[sup], pos, k=knn)
    true_hot, pred_hot = top_mask(y), top_mask(y_hat)
    row.update(
        support_fraction=k_sup / n,
        oracle_r2=r2_score(y, y_hat),
        oracle_top10_ratio=float(y_hat[true_hot].mean() / y[true_hot].mean()),
        oracle_p99_ratio=float(np.percentile(y_hat, 99) / np.percentile(y, 99)),
        oracle_max_ratio=float(y_hat.max() / y.max()),
        oracle_top10_iou=iou(true_hot, pred_hot),
        oracle_spearman_high=spearman(y[true_hot], y_hat[true_hot]),
        oracle_spearman_all=spearman(y, y_hat),
    )
    # 同一天花板，但在模型实际训练的标准化 log 空间里
    z = (np.log(np.clip(y, 0, None) + stats["eps"]) - stats["log"]["mean"]) / stats["log"]["std"]
    row["oracle_r2_normalized"] = r2_score(z, knn_interpolate_numpy(pos[sup], z[sup], pos, k=knn))

    # --- 2. 标签稳定性 ---
    tree = cKDTree(raw)
    nn_dist, _ = tree.query(raw, k=2)
    row["mesh_spacing_mm"] = float(np.median(nn_dist[:, 1]))
    row["support_spacing_mm"] = row["mesh_spacing_mm"] * float(np.sqrt(n / k_sup))
    _, nb = tree.query(raw, k=9)
    # 热点内聚度：热点点的 8 近邻中仍是热点的比例
    row["hotspot_cohesion"] = float(true_hot[nb[true_hot][:, 1:]].mean())
    smoothed = y[nb].mean(axis=1)
    row["smooth8_top10_iou"] = iou(true_hot, top_mask(smoothed))
    row["smooth8_top10_ratio"] = float(smoothed[true_hot].mean() / y[true_hot].mean())
    for offset in (1, 2, 4):
        other = wss_all[min(peak_i + offset, len(steps) - 1)].astype(np.float64)
        row[f"phase+{offset}_top10_iou"] = iou(true_hot, top_mask(other))
        row[f"phase+{offset}_top10_ratio"] = float(other[true_hot].mean() / y[true_hot].mean())
        row[f"phase+{offset}_spearman_high"] = spearman(y[true_hot], other[true_hot])

    # --- 3./4. 方差分解输入 + Poiseuille 先验 ---
    log_y = np.log(y + 1e-6)
    row["log_mean"] = float(log_y.mean())
    row["log_var_within"] = float(log_y.var())
    ok = np.isfinite(local_radius) & (local_radius > 1e-9) & np.isfinite(log_y)
    if int(ok.sum()) > 100:
        # local_radius 已是物理 mm；斜率与 R² 对常数单位因子不变
        log_r = np.log(local_radius[ok])
        slope, intercept = np.polyfit(log_r, log_y[ok], 1)
        row["poiseuille_slope"] = float(slope)
        row["poiseuille_r2_logspace"] = r2_score(log_y[ok], intercept + slope * log_r)
    else:
        row["poiseuille_slope"] = float("nan")
        row["poiseuille_r2_logspace"] = float("nan")
    row["local_radius_median_mm"] = float(np.median(local_radius))
    return row


def summarize(rows: list, keys: list) -> None:
    cohorts = ["ALL"] + sorted({r["cohort"] for r in rows})
    print(f"\n{'metric':30s}" + "".join(f"{c:>11s}" for c in cohorts))
    for key in keys:
        line = f"{key:30s}"
        for cohort in cohorts:
            vals = np.array(
                [r[key] for r in rows if cohort == "ALL" or r["cohort"] == cohort],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            line += f"{np.median(vals):11.4f}" if vals.size else f"{'-':>11s}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-glob", default=DEFAULT_GLOB)
    ap.add_argument("--stats", default=DEFAULT_STATS,
                    help="训练所用的 train-only global stats（标准化空间天花板需要）")
    ap.add_argument("--support", type=int, default=5000, help="支撑点数，须与配置一致")
    ap.add_argument("--knn", type=int, default=3, help="解码器 fp_knn")
    ap.add_argument("--n-per-cohort", type=int, default=8, help="每 cohort 抽样数；0 = 全库")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    stats = json.loads((PROJECT_ROOT / args.stats).read_text())
    if stats.get("method") != "log_z":
        raise ValueError(f"expected log_z stats, got {stats.get('method')!r}")

    paths = sorted(glob.glob(str(PROJECT_ROOT / args.bundle_glob)))
    if not paths:
        raise FileNotFoundError(f"no bundles matched {args.bundle_glob}")
    by_cohort = collections.defaultdict(list)
    for p in paths:
        by_cohort[p.split("/")[-4]].append(p)
    if args.n_per_cohort > 0:
        rng = np.random.default_rng(args.seed)
        picked = []
        for _, group in sorted(by_cohort.items()):
            picked += list(rng.choice(group, size=min(args.n_per_cohort, len(group)),
                                      replace=False))
    else:
        picked = paths

    rows = []
    for i, path in enumerate(picked, 1):
        rows.append(diagnose_case(path, args.support, args.knn, stats))
        print(f"\r  {i}/{len(picked)}", end="", flush=True)
    print()

    print(f"\n===== 协议天花板（真值在 {args.support} 支撑点上完美 -> {args.knn}-NN 插值到全点云）=====")
    summarize(rows, ["n_wall", "support_fraction", "mesh_spacing_mm", "support_spacing_mm",
                     "oracle_r2", "oracle_r2_normalized", "oracle_top10_ratio",
                     "oracle_p99_ratio", "oracle_top10_iou", "oracle_spearman_high"])
    print("\n===== 标签稳定性（热点是否为网格已解析的连续结构；峰值选步是否是噪声源）=====")
    summarize(rows, ["hotspot_cohesion", "smooth8_top10_iou", "smooth8_top10_ratio",
                     "phase+1_top10_iou", "phase+1_top10_ratio", "phase+1_spearman_high",
                     "phase+2_top10_iou", "phase+4_top10_iou", "phase+4_top10_ratio"])
    print("\n===== Poiseuille 先验（单个 log(local_radius) 能解释多少病例内 log 方差）=====")
    summarize(rows, ["poiseuille_slope", "poiseuille_r2_logspace", "local_radius_median_mm"])

    log_means = np.array([r["log_mean"] for r in rows], dtype=np.float64)
    within = np.array([r["log_var_within"] for r in rows], dtype=np.float64)
    between_var, within_var = float(log_means.var()), float(within.mean())
    share = between_var / (between_var + within_var)
    print(f"\n===== log(WSS) 方差分解（{len(rows)} 例）=====")
    print(f"  病例间 var(病例平均 log WSS) = {between_var:.4f}")
    print(f"  病例内平均 var               = {within_var:.4f}")
    print(f"  病例间占比                   = {share:.3f}"
          f"   <- 一切病例级条件信号（出口 RCR / cohort / 尺度头）的收益上界")

    if args.out:
        out = PROJECT_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"config": vars(args),
             "variance_decomposition": {"between": between_var, "within": within_var,
                                        "between_share": share},
             "cases": rows}, indent=1, ensure_ascii=False))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
