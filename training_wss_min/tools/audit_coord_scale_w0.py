#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W0｜尺度与裁剪只读审计（第六轮 WSS 精度突破计划 §3）。

只读 bundle.npz，不重训、不用 val 选择任何阈值。对 dev1 的 train/val 报告：

1. train/val ``coord_scale`` 范围，val 是否超出 train 范围（OOD 判据）。
2. ``coord_scale`` 与 bbox 对角线、入口面积代理、壁面 local_radius 中位数、
   峰值 WSS case mean/p95/p99 的 Spearman 关联。
3. fast/slow、``wall_crop_applied``、``wall_crop_frac`` 分层。
4. 明确 ``coord_scale`` = 配准（居中/刚性旋转）+裁剪后坐标的最大绝对值，
   不等同血管直径；入口面积仅为 π·r_inlet² 代理。

用法：
    python -m training_wss_min.tools.audit_coord_scale_w0
产物：
    training_wss_min/runs/_audits/w0_coord_scale/audit.json
    training_wss_min/runs/_audits/w0_coord_scale/report.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D

OUT_DIR = C.RUNS_ROOT / "_audits" / "w0_coord_scale"
SPLIT = C.PROJECT_ROOT / "training" / "splits" / "split_AG_wss_min_v2_dev1.json"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _case_row(cohort_rel: str, case_name: str) -> Dict:
    p = C.DATA_ROOT / cohort_rel / case_name / "bundle.npz"
    with np.load(p, allow_pickle=True) as d:
        steps = d["steps"].tolist()
        si = steps.index(int(d["peak_step"]))
        cr = d["wall_coords_raw"].astype(np.float64)
        absc = d["wall_abscissa_norm"].astype(np.float64)
        lr = d["wall_local_radius"].astype(np.float64)
        wss = d["wall_wss"][si].astype(np.float64)
        extent = cr.max(0) - cr.min(0)
        inlet_mask = absc <= max(0.05, np.quantile(absc, 0.05))
        r_inlet = float(np.median(lr[inlet_mask])) if inlet_mask.any() else float("nan")
        return {
            "cohort": cohort_rel,
            "case": case_name,
            "subset": cohort_rel.split("/", 1)[1],  # fast / slow
            "coord_scale": float(d["coord_scale"]),
            "coord_scale_on": str(d["coord_scale_on"]),
            "n_wall": int(cr.shape[0]),
            "bbox_extent_x": float(extent[0]),
            "bbox_extent_y": float(extent[1]),
            "bbox_extent_z": float(extent[2]),
            "bbox_diag": float(np.linalg.norm(extent)),
            "r_inlet_proxy": r_inlet,
            "inlet_area_proxy": float(np.pi * r_inlet ** 2) if np.isfinite(r_inlet) else float("nan"),
            "local_radius_median": float(np.median(lr)),
            "wss_case_mean": float(np.mean(wss)),
            "wss_p95": float(np.percentile(wss, 95)),
            "wss_p99": float(np.percentile(wss, 99)),
            "wall_crop_applied": bool(d["wall_crop_applied"]),
            "wall_crop_frac": float(d["wall_crop_frac"]),
            "unit_factor": float(d["unit_factor"]),
        }


def _range(vals: List[float]) -> Dict:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p05": float(np.percentile(a, 5)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_lbl = D.load_split_cases(SPLIT, "train")
    val_lbl = D.load_split_cases(SPLIT, "val")
    rows = {
        "train": [_case_row(c, n) for c, n in train_lbl],
        "val": [_case_row(c, n) for c, n in val_lbl],
    }
    all_rows = rows["train"] + rows["val"]

    cs_tr = [r["coord_scale"] for r in rows["train"]]
    cs_va = [r["coord_scale"] for r in rows["val"]]
    tr_min, tr_max = min(cs_tr), max(cs_tr)
    val_ood = [r["case"] for r in rows["val"]
               if r["coord_scale"] < tr_min or r["coord_scale"] > tr_max]

    # Spearman：coord_scale vs 各物理量（全 dev1 池，train+val）。
    def col(key):
        return np.asarray([r[key] for r in all_rows], dtype=np.float64)

    cs_all = col("coord_scale")
    spearman = {
        k: _spearman(cs_all, col(k))
        for k in [
            "bbox_diag", "bbox_extent_x", "bbox_extent_y", "bbox_extent_z",
            "inlet_area_proxy", "r_inlet_proxy", "local_radius_median",
            "wss_case_mean", "wss_p95", "wss_p99", "n_wall",
        ]
    }

    strat = {}
    for sub in ("fast", "slow"):
        vals = [r["coord_scale"] for r in all_rows if r["subset"] == sub]
        if vals:
            strat[sub] = _range(vals)
    crop_applied = [r["case"] for r in all_rows if r["wall_crop_applied"]]

    audit = {
        "protocol": "W0 coord_scale/crop read-only audit (round6 §3)",
        "split": str(SPLIT.name),
        "note_coord_scale": "coord_scale = max abs coordinate after registration (centering + rigid rotation) and optional wall crop; NOT vessel diameter. inlet_area_proxy = pi*r_inlet^2 with r_inlet = median wall local_radius near abscissa~0.",
        "coord_scale_range": {
            "train": _range(cs_tr),
            "val": _range(cs_va),
            "val_outside_train_range_cases": val_ood,
            "val_within_train_range": len(val_ood) == 0,
        },
        "spearman_coord_scale_vs": spearman,
        "stratify_fast_slow": strat,
        "wall_crop": {
            "any_applied": len(crop_applied) > 0,
            "applied_cases": crop_applied,
            "frac_nonzero_cases": [r["case"] for r in all_rows if r["wall_crop_frac"] > 0],
        },
        "per_case": all_rows,
    }
    (OUT_DIR / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    # ---- markdown 报告 ----
    def fmt(x, d=3):
        return f"{x:.{d}f}" if isinstance(x, float) and np.isfinite(x) else str(x)

    lines = []
    lines.append("# W0｜尺度与裁剪只读审计报告")
    lines.append("")
    lines.append(f"> split：`{SPLIT.name}`；train {len(rows['train'])} / val {len(rows['val'])}；只读，不重训、不用 val 选阈值。")
    lines.append("")
    lines.append("`coord_scale` = 配准（居中+刚性旋转）与可选壁面裁剪后**坐标最大绝对值**，"
                 "不是血管直径；入口面积为 `π·r_inlet²` 代理（`r_inlet`=近入口壁面 `local_radius` 中位数）。")
    lines.append("")
    lines.append("## 1. coord_scale 范围与 val OOD")
    lines.append("")
    lines.append("| 分区 | n | min | median | max | mean±std |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for part in ("train", "val"):
        r = audit["coord_scale_range"][part]
        lines.append(f"| {part} | {r['n']} | {fmt(r['min'],1)} | {fmt(r['median'],1)} | "
                     f"{fmt(r['max'],1)} | {fmt(r['mean'],1)}±{fmt(r['std'],1)} |")
    lines.append("")
    ood = audit["coord_scale_range"]["val_outside_train_range_cases"]
    if ood:
        lines.append(f"- **val 超出 train coord_scale 范围**：{', '.join(ood)}（尺度 OOD 风险）。")
    else:
        lines.append(f"- **val 全部落在 train coord_scale 范围内** [{fmt(tr_min,1)}, {fmt(tr_max,1)}]，无尺度外插。")
    lines.append("")
    lines.append("## 2. coord_scale 与物理量的 Spearman 关联（dev1 全池）")
    lines.append("")
    lines.append("| 物理量 | Spearman ρ |")
    lines.append("|---|---:|")
    for k, v in spearman.items():
        lines.append(f"| {k} | {fmt(v)} |")
    lines.append("")
    lines.append("解读：ρ 越接近 1 表示 `coord_scale` 越能代表该物理量；与 WSS 幅值（case mean/p95/p99）"
                 "的关联决定尺度标量作为输入能否帮助 WSS 幅值定标。")
    lines.append("")
    lines.append("## 3. fast/slow 分层与壁面裁剪")
    lines.append("")
    lines.append("| subset | n | coord_scale median | min–max |")
    lines.append("|---|---:|---:|---:|")
    for sub, r in strat.items():
        lines.append(f"| {sub} | {r['n']} | {fmt(r['median'],1)} | {fmt(r['min'],1)}–{fmt(r['max'],1)} |")
    lines.append("")
    lines.append(f"- `wall_crop_applied` 病例数：{len(crop_applied)}"
                 + (f"（{', '.join(crop_applied)}）" if crop_applied else "（全 dev1 无壁面裁剪）") + "。")
    lines.append("")
    lines.append("## 4. 逐病例明细（coord_scale 降序）")
    lines.append("")
    lines.append("| case | subset | coord_scale | bbox_diag | inlet_area | lr_median | wss_mean | wss_p95 | wss_p99 | n_wall | crop |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for r in sorted(all_rows, key=lambda x: -x["coord_scale"]):
        lines.append(
            f"| {r['case']} | {r['subset']} | {fmt(r['coord_scale'],1)} | {fmt(r['bbox_diag'],1)} | "
            f"{fmt(r['inlet_area_proxy'],1)} | {fmt(r['local_radius_median'],2)} | {fmt(r['wss_case_mean'],3)} | "
            f"{fmt(r['wss_p95'],2)} | {fmt(r['wss_p99'],2)} | {r['n_wall']} | {'Y' if r['wall_crop_applied'] else '·'} |"
        )
    lines.append("")
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("== W0 coord_scale audit ==")
    print(f"train coord_scale range: [{tr_min:.1f}, {tr_max:.1f}]  val: [{min(cs_va):.1f}, {max(cs_va):.1f}]")
    print(f"val outside train range: {ood or 'none'}")
    print("Spearman coord_scale vs:")
    for k, v in spearman.items():
        print(f"   {k:22s} {v:+.3f}")
    print(f"wall_crop applied cases: {crop_applied or 'none'}")
    print(f"-> {OUT_DIR/'report.md'}")


if __name__ == "__main__":
    main()
