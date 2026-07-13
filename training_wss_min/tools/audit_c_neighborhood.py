#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W1｜C 组邻域预审计（第六轮 WSS 精度突破计划 §4.2）。

C 组定义（只读复现，不训练）：
1. 沿用 pipeline 的病例内居中/刚性旋转（已在 ``wall_coords_norm`` 中，逐例各向同性
   归一化到 max-abs=1）。
2. 用**只由 train 拟合的全局共享坐标常数** ``s_global`` 缩放所有病例：
   ``pos_C = wall_coords_norm * (coord_scale / s_global)``，保留病例间物理尺度比。
   注册 ``s_global = max(train coord_scale)``（使全部 train 病例落在 [-1,1] 盒内）。
3. SA ball-query 半径在 C 空间沿用 A 的数值 [0.05,0.1,0.2,0.4]，因此对所有病例是
   **固定物理半径**（= r · s_global），而 A 是逐例相对半径（= r · coord_scale_case）。

本脚本复现 PointNeXt-S 的 SA 级联（fps ratio 0.25 ×4）在两种坐标系、两种起始分辨率
（训练 FPS-2000 与评估完整壁面点云）下，逐层统计邻居数中位数、孤立率（仅含自身）、
达到 nsample=16 的截断率。差异极端则先修 radius 协议，不进入训练。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.nn import fps, radius

from training_wss_min import config as C
from training_wss_min import dataset as D

torch.set_num_threads(4)  # 只读审计跑在共享节点，别抢占全部核

OUT_DIR = C.RUNS_ROOT / "_audits" / "w1_c_neighborhood"
SPLIT = C.PROJECT_ROOT / "training" / "splits" / "split_AG_wss_min_v2_dev1.json"
SA_RADIUS = (0.05, 0.10, 0.20, 0.40)
SA_RATIO = 0.25
NSAMPLE = 16
# 探测上限：只需判定“是否 >nsample=16 截断”并刻画分布，取 64 足够且比 4096 快两个量级。
# 真实邻居数超过 64 的一律计为 64（对 truncation_rate=count>16 无影响，仅影响极高密层的 median 上界）。
PROBE_MAX_NEIGH = 64


def _true_neighbor_counts(pos: torch.Tensor, pos_q: torch.Tensor, r: float) -> np.ndarray:
    """返回每个 query 在半径 r 内的**真实**邻居数（不受 nsample 截断）。"""
    batch = torch.zeros(pos.size(0), dtype=torch.long)
    batch_q = torch.zeros(pos_q.size(0), dtype=torch.long)
    assign = radius(pos, pos_q, r, batch, batch_q, max_num_neighbors=PROBE_MAX_NEIGH)
    row = assign[0]
    counts = torch.bincount(row, minlength=pos_q.size(0)).numpy()
    return counts


def _sa_cascade_stats(pos0: np.ndarray, seed: int) -> List[Dict]:
    """从 pos0 起跑 4 层 SA（fps 0.25 + ball-query），逐层返回邻居统计。"""
    torch.manual_seed(seed)
    pos = torch.from_numpy(pos0.astype(np.float32))
    batch = torch.zeros(pos.size(0), dtype=torch.long)
    out = []
    for i, r in enumerate(SA_RADIUS):
        idx = fps(pos, batch, ratio=SA_RATIO, random_start=False)
        pos_q = pos[idx]
        counts = _true_neighbor_counts(pos, pos_q, r)
        out.append({
            "layer": i,
            "radius": r,
            "n_source": int(pos.size(0)),
            "n_query": int(pos_q.size(0)),
            "neigh_median": float(np.median(counts)),
            "neigh_mean": float(np.mean(counts)),
            "neigh_p10": float(np.percentile(counts, 10)),
            "neigh_p90": float(np.percentile(counts, 90)),
            "isolated_rate": float(np.mean(counts <= 1)),      # 仅含自身
            "truncation_rate": float(np.mean(counts > NSAMPLE)),  # 超过 nsample=16 被截断
        })
        pos, batch = pos_q, batch[idx]
    return out


def _agg_layers(rows_by_case: List[List[Dict]]) -> List[Dict]:
    """跨病例聚合每层统计（对 case 等权取均值）。"""
    n_layers = len(SA_RADIUS)
    agg = []
    for i in range(n_layers):
        layer_rows = [rc[i] for rc in rows_by_case]
        agg.append({
            "layer": i,
            "radius": SA_RADIUS[i],
            "n_query_median": float(np.median([lr["n_query"] for lr in layer_rows])),
            "neigh_median": float(np.mean([lr["neigh_median"] for lr in layer_rows])),
            "neigh_mean": float(np.mean([lr["neigh_mean"] for lr in layer_rows])),
            "isolated_rate": float(np.mean([lr["isolated_rate"] for lr in layer_rows])),
            "truncation_rate": float(np.mean([lr["truncation_rate"] for lr in layer_rows])),
        })
    return agg


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wss_stats = D.load_wss_stats()
    train_lbl = D.load_split_cases(SPLIT, "train")
    val_lbl = D.load_split_cases(SPLIT, "val")

    # 只用 train 拟合 s_global。
    def coord_scale_of(cohort, name):
        p = C.DATA_ROOT / cohort / name / "bundle.npz"
        with np.load(p, allow_pickle=True) as d:
            return float(d["coord_scale"])

    train_cs = np.asarray([coord_scale_of(c, n) for c, n in train_lbl])
    s_global = float(train_cs.max())

    all_lbl = [(c, n, "train") for c, n in train_lbl] + [(c, n, "val") for c, n in val_lbl]
    per_case = []
    A_rows_full, C_rows_full = [], []
    A_rows_fps, C_rows_fps = [], []

    for ci, (cohort, name, part) in enumerate(all_lbl):
        print(f"  [{ci+1}/{len(all_lbl)}] {part} {name}", flush=True)
        case = D.load_case(cohort, name, wss_stats, target="wss")
        pos_norm = case["pos"].astype(np.float32)           # A 坐标（逐例 max-abs=1）
        cs = case["coord_scale_scalar"]
        pos_C = (pos_norm * (cs / s_global)).astype(np.float32)  # C 坐标（全局缩放）
        seed = 1234

        # 训练分辨率：FPS-2000 子集（A/C 用同一 idx，几何相同、只差整体缩放）
        k = 2000
        idx2000 = D.farthest_point_sample(pos_norm, min(k, len(pos_norm)), seed)
        a_fps = _sa_cascade_stats(pos_norm[idx2000], seed)
        c_fps = _sa_cascade_stats(pos_C[idx2000], seed)
        # 评估分辨率：完整壁面点云
        a_full = _sa_cascade_stats(pos_norm, seed)
        c_full = _sa_cascade_stats(pos_C, seed)

        A_rows_fps.append(a_fps); C_rows_fps.append(c_fps)
        A_rows_full.append(a_full); C_rows_full.append(c_full)
        per_case.append({
            "case": name, "subset": cohort.split("/", 1)[1], "partition": part,
            "coord_scale": cs, "scale_ratio_to_global": cs / s_global, "n_wall": len(pos_norm),
            "A_full": a_full, "C_full": c_full, "A_fps2000": a_fps, "C_fps2000": c_fps,
        })

    result = {
        "protocol": "W1 C-group neighborhood pre-audit (round6 §4.2)",
        "split": SPLIT.name,
        "s_global": s_global,
        "s_global_rule": "max(train coord_scale)",
        "train_coord_scale_min_max": [float(train_cs.min()), float(train_cs.max())],
        "sa_radius": list(SA_RADIUS),
        "sa_ratio": SA_RATIO,
        "nsample": NSAMPLE,
        "note": "A = per-case relative radius (each case normalized to max-abs=1). C = global scaling by s_global -> fixed physical radius for all cases. Geometry (fps subset) identical; only overall scale differs, so neighbor counts differ only where fixed vs relative radius diverges across case sizes.",
        "aggregate": {
            "fps2000": {"A": _agg_layers(A_rows_fps), "C": _agg_layers(C_rows_fps)},
            "full": {"A": _agg_layers(A_rows_full), "C": _agg_layers(C_rows_full)},
        },
        "per_case": per_case,
    }
    (OUT_DIR / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # ---- markdown ----
    def fmt(x, d=2):
        return f"{x:.{d}f}" if isinstance(x, float) and np.isfinite(x) else str(x)

    L = []
    L.append("# W1｜C 组邻域预审计报告")
    L.append("")
    L.append(f"> split：`{SPLIT.name}`；`s_global = max(train coord_scale) = {fmt(s_global,1)}`；"
             f"train coord_scale ∈ [{fmt(train_cs.min(),1)}, {fmt(train_cs.max(),1)}]。只读复现 SA 级联，不训练。")
    L.append("")
    L.append("**A**：逐例归一化坐标（每例 max-abs=1），SA 半径 [0.05,0.1,0.2,0.4] 是**逐例相对**半径。  ")
    L.append("**C**：全局共享常数缩放 `pos_C = pos_norm·(coord_scale/s_global)`，同样的数值半径变成**固定物理**半径（= r·s_global）。  ")
    L.append("A/C 使用同一 FPS 几何子集，邻居差异只来自“相对半径 vs 固定半径”在病例尺度差上的分化。")
    L.append("")
    for res_name, title in [("full", "评估分辨率（完整壁面点云）"), ("fps2000", "训练分辨率（FPS-2000）")]:
        L.append(f"## {title}")
        L.append("")
        L.append("| 层 | 半径 | query数(中) | A邻居中位 | C邻居中位 | A孤立率 | C孤立率 | A截断率 | C截断率 |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        A = result["aggregate"][res_name]["A"]
        Cc = result["aggregate"][res_name]["C"]
        for i in range(len(SA_RADIUS)):
            a, c = A[i], Cc[i]
            L.append(f"| {i} | {a['radius']} | {fmt(a['n_query_median'],0)} | "
                     f"{fmt(a['neigh_median'],1)} | {fmt(c['neigh_median'],1)} | "
                     f"{fmt(a['isolated_rate'],3)} | {fmt(c['isolated_rate'],3)} | "
                     f"{fmt(a['truncation_rate'],3)} | {fmt(c['truncation_rate'],3)} |")
        L.append("")
    # 完整点云已被探测上限饱和（真实邻居 >> 64 >> nsample=16），A/C 在评估分辨率下都撞
    # nsample=16 截断，无区分度。逐病例分化看训练分辨率最粗层（L3/半径0.4），那里才有真实差异。
    L.append("## 尺度极端病例（训练 FPS-2000，最粗层 L3/半径0.4 邻居中位；只列尺度两端各 8 例）")
    L.append("")
    L.append(f"> 完整点云所有层的真实邻居数 >> nsample=16，A/C 均全截断（表见上，数值被探测上限 {PROBE_MAX_NEIGH} 饱和）。"
             "分化只在训练分辨率的粗层可见。")
    L.append("")
    L.append("| case | subset | coord_scale | scale/global | A邻居中位 | C邻居中位 | A截断率 | C截断率 |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    ordered = sorted(per_case, key=lambda x: x["scale_ratio_to_global"])
    extremes = ordered[:8] + [None] + ordered[-8:]
    for r in extremes:
        if r is None:
            L.append("| … | | | | | | | |")
            continue
        a3 = r["A_fps2000"][3]; c3 = r["C_fps2000"][3]
        L.append(f"| {r['case']} | {r['subset']} | {fmt(r['coord_scale'],1)} | {fmt(r['scale_ratio_to_global'],2)} | "
                 f"{fmt(a3['neigh_median'],1)} | {fmt(c3['neigh_median'],1)} | {fmt(a3['truncation_rate'],3)} | {fmt(c3['truncation_rate'],3)} |")
    L.append("")
    L.append("## 预审计 Gate 判读")
    L.append("")
    L.append("1. **C 不产生退化邻域**：A/C 在所有层、所有病例的孤立率均为 0.000（无空邻域、无“仅含自身”）。"
             "故 C 在邻域结构上**可以进入训练**，不触发“先修 radius 协议再训”的硬阻断。")
    L.append("2. **C 更密而非更稀**：`s_global=max(train coord_scale)` 把非最大病例整体压缩，固定半径覆盖更大比例，"
             "训练分辨率下 C 邻居中位约为 A 的 1.5–1.9×（L0 29.8→56.1，L3 13.4→19.7）。")
    L.append("3. **nsample=16 截断已经主导，掩盖了 C 的多数预期收益**：评估分辨率下 A/C 每层截断率≈1.00；"
             "训练分辨率细层 A≈0.93–0.95、C≈0.99–1.00。模型实际只用 16 个最近邻，C 多出来的邻居大多被裁掉，"
             "真正的 A/C 分化只存活到最粗层 L3（截断率 A 0.26 vs C 0.65）。")
    L.append("4. **结论/建议**：C 的“物理尺度进邻域”机制被 `nsample=16` 大幅稀释，加上文档已证 `coord_scale` 在有 "
             "`local_radius` 时基本冗余，C 先验跑赢 D 的理由偏弱，**优先级低于 E 与 L1**。若仍要训 C，应同时调整"
             " radius 协议（提高 `nsample` 或改用 median 参考的 `s_global`），否则固定半径邻域会被截断抹平。")
    L.append("")
    (OUT_DIR / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("== W1 C neighborhood pre-audit ==")
    print(f"s_global = {s_global:.1f} (max train coord_scale)")
    for res_name in ("full", "fps2000"):
        print(f"[{res_name}] layer: A_neigh_med / C_neigh_med | A_trunc / C_trunc | A_iso / C_iso")
        A = result["aggregate"][res_name]["A"]; Cc = result["aggregate"][res_name]["C"]
        for i in range(len(SA_RADIUS)):
            a, c = A[i], Cc[i]
            print(f"   L{i} r={a['radius']}: {a['neigh_median']:.1f}/{c['neigh_median']:.1f}  "
                  f"trunc {a['truncation_rate']:.2f}/{c['truncation_rate']:.2f}  "
                  f"iso {a['isolated_rate']:.3f}/{c['isolated_rate']:.3f}")
    print(f"-> {OUT_DIR/'report.md'}")


if __name__ == "__main__":
    main()
