#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较普通分叉原点与实验性三臂等权 flow-divider 原点。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    if Path(_p).is_file():
        try:
            matplotlib.font_manager.fontManager.addfont(_p)
            matplotlib.rcParams["font.family"] = \
                matplotlib.font_manager.FontProperties(fname=_p).get_name()
            break
        except Exception:  # noqa: BLE001
            pass
matplotlib.rcParams["axes.unicode_minus"] = False

from pipeline_wss_min import config as C
from pipeline_wss_min import raw_io
from pipeline_wss_min.archive.alignment_v3 import legacy_registration_for_case
from pipeline_wss_min.preprocess import _resolve_unit_factor
from pipeline_wss_min.registration import compute_transform, untraced_inlet_crop_masks, AXIS_INDEX


OUT_DIR = C.PROJECT_ROOT / "outputs" / "wss_min" / "flow_divider_origin_compare"


def _case_labels(split_name: str) -> List[str]:
    return C.split_case_labels(split_name, ("train", "val", "test"))


def _aligned_metrics(
    wall_pts: np.ndarray,
    int_pts: np.ndarray,
    centerline: Dict[str, np.ndarray],
    reg_cfg,
):
    transform = compute_transform(wall_pts, int_pts, centerline, reg_cfg)
    wall_aln = transform.apply_points(wall_pts)
    int_aln = transform.apply_points(int_pts)
    cl_aln = transform.apply_points(centerline["coords"])
    tgt = AXIS_INDEX[transform.principal_axis_target]
    wall_keep, int_keep, crop_frac, crop_applied = untraced_inlet_crop_masks(
        wall_aln, int_aln, cl_aln, centerline["abscissa"], tgt, reg_cfg)
    if crop_applied:
        wall_aln = wall_aln[wall_keep]
        int_aln = int_aln[int_keep]
    ref = wall_aln if C.DEFAULT.normalization.coord_scale_on == "wall" \
        else np.vstack([wall_aln, int_aln])
    scale = float(np.abs(ref).max()) or 1.0
    wall_norm = wall_aln / scale
    p01 = np.percentile(wall_norm, 1, axis=0)
    p50 = np.percentile(wall_norm, 50, axis=0)
    p99 = np.percentile(wall_norm, 99, axis=0)
    return {
        "transform": transform,
        "wall_norm": wall_norm,
        "coord_scale": scale,
        "com": float(np.linalg.norm(wall_norm.mean(axis=0))),
        "z_p01": float(p01[2]),
        "z_p50": float(p50[2]),
        "z_p99": float(p99[2]),
        "x_p50": float(p50[0]),
        "y_p50": float(p50[1]),
        "crop_applied": bool(crop_applied),
        "crop_frac": float(crop_frac),
    }


def _load_pair(label: str):
    subset, case_name = label.split("/", 1)
    cohort_rel = "AG/" + subset
    case_dir = C.raw_case_dir(cohort_rel, case_name)
    steps = raw_io.list_timesteps(case_dir)
    ref = steps[0]
    wall_native = raw_io.read_wall_geometry(case_dir, case_name, ref)
    int_native = raw_io.read_interior_geometry(case_dir, case_name, ref)
    centerline = raw_io.read_centerline(case_dir)
    unit_factor, unit_anomaly, unit_extent_mismatch = _resolve_unit_factor(
        wall_native, centerline, C.DEFAULT.unit)
    wall_pts = wall_native * unit_factor
    int_pts = int_native * unit_factor

    base_cfg = legacy_registration_for_case(cohort_rel, case_name)
    base_cfg = replace(base_cfg, center_on="bifurcation")
    flow_cfg = replace(base_cfg, center_on="flow_divider")
    base = _aligned_metrics(wall_pts, int_pts, centerline, base_cfg)
    flow = _aligned_metrics(wall_pts, int_pts, centerline, flow_cfg)

    shift_world = flow["transform"].centroid - base["transform"].centroid
    shift_base_frame = shift_world @ base["transform"].rotation
    return {
        "case": label,
        "cohort": cohort_rel,
        "unit_factor": float(unit_factor),
        "unit_anomaly": bool(unit_anomaly),
        "unit_extent_mismatch": bool(unit_extent_mismatch),
        "base_origin_kind": base["transform"].origin_kind,
        "flow_origin_kind": flow["transform"].origin_kind,
        "origin_shift_mm": float(np.linalg.norm(shift_world)),
        "origin_shift_x_mm": float(shift_base_frame[0]),
        "origin_shift_y_mm": float(shift_base_frame[1]),
        "origin_shift_z_mm": float(shift_base_frame[2]),
        "base_coord_scale": base["coord_scale"],
        "flow_coord_scale": flow["coord_scale"],
        "base_com": base["com"],
        "flow_com": flow["com"],
        "delta_com": flow["com"] - base["com"],
        "base_z_p01": base["z_p01"],
        "flow_z_p01": flow["z_p01"],
        "base_z_p50": base["z_p50"],
        "flow_z_p50": flow["z_p50"],
        "base_z_p99": base["z_p99"],
        "flow_z_p99": flow["z_p99"],
        "base_crop_applied": base["crop_applied"],
        "flow_crop_applied": flow["crop_applied"],
        "base_crop_frac": base["crop_frac"],
        "flow_crop_frac": flow["crop_frac"],
        "base_main_axis_source": base["transform"].main_axis_source,
        "flow_main_axis_source": flow["transform"].main_axis_source,
        "base_roll_sign_reliable": bool(base["transform"].roll_sign_reliable),
        "flow_roll_sign_reliable": bool(flow["transform"].roll_sign_reliable),
    }


def _write_csv(rows: List[dict], out_dir: Path) -> Path:
    path = out_dir / "flow_divider_origin_compare_78cases.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot_summary(rows: List[dict], out_dir: Path) -> None:
    shift = np.asarray([r["origin_shift_mm"] for r in rows], dtype=float)
    base_com = np.asarray([r["base_com"] for r in rows], dtype=float)
    flow_com = np.asarray([r["flow_com"] for r in rows], dtype=float)
    dz = np.asarray([r["origin_shift_z_mm"] for r in rows], dtype=float)
    labels = [r["case"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    ax = axes[0, 0]
    ax.hist(shift, bins=24, color="#4C78A8", alpha=0.85)
    ax.set_title("新旧原点距离分布")
    ax.set_xlabel("origin shift (mm)")
    ax.set_ylabel("cases")

    ax = axes[0, 1]
    ax.scatter(base_com, flow_com, s=24, alpha=0.8)
    lo = min(base_com.min(), flow_com.min())
    hi = max(base_com.max(), flow_com.max())
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--")
    ax.set_title("|COM|：当前原点 vs flow-divider")
    ax.set_xlabel("current bifurcation")
    ax.set_ylabel("flow-divider")

    ax = axes[1, 0]
    order = np.argsort(shift)[-15:]
    ax.barh(np.arange(len(order)), shift[order], color="#F58518", alpha=0.85)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([labels[i] for i in order], fontsize=8)
    ax.set_title("原点变化最大的 15 例")
    ax.set_xlabel("origin shift (mm)")

    ax = axes[1, 1]
    ax.scatter(dz, flow_com - base_com, s=24, alpha=0.8, color="#54A24B")
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_title("原点 Z 向移动 vs COM 变化")
    ax.set_xlabel("origin shift z in current frame (mm)")
    ax.set_ylabel("flow COM - current COM")
    fig.tight_layout()
    fig.savefig(out_dir / "flow_divider_origin_summary.png", dpi=160)
    plt.close(fig)


def _write_summary(rows: List[dict], out_dir: Path) -> None:
    shift = np.asarray([r["origin_shift_mm"] for r in rows], dtype=float)
    base_com = np.asarray([r["base_com"] for r in rows], dtype=float)
    flow_com = np.asarray([r["flow_com"] for r in rows], dtype=float)
    delta = flow_com - base_com
    improved = int((flow_com < base_com).sum())
    worsened = int((flow_com > base_com).sum())
    strong_improved = int((delta <= -0.005).sum())
    near_same = int((np.abs(delta) < 0.005).sum())
    strong_worsened = int((delta >= 0.005).sum())
    fallback = [r["case"] for r in rows if r["flow_origin_kind"] != "flow_divider"]
    top_shift = sorted(rows, key=lambda r: r["origin_shift_mm"], reverse=True)[:10]
    top_gain = sorted(rows, key=lambda r: r["delta_com"])[:10]
    top_loss = sorted(rows, key=lambda r: r["delta_com"], reverse=True)[:10]
    summary = {
        "n_cases": len(rows),
        "origin_shift_mm": {
            "mean": float(shift.mean()),
            "median": float(np.median(shift)),
            "p95": float(np.percentile(shift, 95)),
            "max": float(shift.max()),
        },
        "base_com": {"mean": float(base_com.mean()), "median": float(np.median(base_com))},
        "flow_com": {"mean": float(flow_com.mean()), "median": float(np.median(flow_com))},
        "com_improved_cases": improved,
        "com_worsened_cases": worsened,
        "com_strong_improved_cases_delta_lte_-0.005": strong_improved,
        "com_near_same_cases_abs_delta_lt_0.005": near_same,
        "com_strong_worsened_cases_delta_gte_0.005": strong_worsened,
        "flow_fallback_cases": fallback,
        "top_origin_shift": [
            {"case": r["case"], "shift_mm": round(r["origin_shift_mm"], 3)}
            for r in top_shift
        ],
        "top_com_gain": [
            {"case": r["case"], "delta_com": round(r["delta_com"], 4)}
            for r in top_gain
        ],
        "top_com_loss": [
            {"case": r["case"], "delta_com": round(r["delta_com"], 4)}
            for r in top_loss
        ],
    }
    (out_dir / "flow_divider_origin_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# flow-divider 原点对照报告",
        "",
        f"- 病例数：{len(rows)}",
        f"- 原点位移 median/mean/p95/max：{np.median(shift):.2f} / {shift.mean():.2f} / "
        f"{np.percentile(shift, 95):.2f} / {shift.max():.2f} mm",
        f"- 当前原点 |COM| mean/median：{base_com.mean():.4f} / {np.median(base_com):.4f}",
        f"- flow-divider |COM| mean/median：{flow_com.mean():.4f} / {np.median(flow_com):.4f}",
        f"- |COM| 改善/变差：{improved} / {worsened}",
        f"- 明确改善/近似不变/明确变差（阈值 0.005）：{strong_improved} / {near_same} / {strong_worsened}",
        f"- flow-divider fallback 病例数：{len(fallback)}",
        "",
        "## 算法口径",
        "- 当前原点：`DistToBifurcation≈0` 分叉近邻点的普通均值。",
        "- flow-divider 原点：把 `DistToBifurcation≈0` 分叉近邻点聚成三臂（主干端 + 左右髂支起始端），",
        "  再对三臂中心做等权平均。这样对应“左右髂支接入主动脉的三叉连接点”，",
        "  并降低某一臂中心线采样更密时对普通均值的拉偏。",
        "- 本报告只比较刚性配准原点变化；旋转、裁剪和壁面归一化口径保持一致。",
        "",
        "## 原点变化最大的病例",
    ]
    lines.extend([f"- {r['case']}: {r['origin_shift_mm']:.2f} mm" for r in top_shift])
    lines.extend(["", "## |COM| 改善最多"])
    lines.extend([f"- {r['case']}: delta={r['delta_com']:.4f}" for r in top_gain])
    lines.extend(["", "## |COM| 变差最多"])
    lines.extend([f"- {r['case']}: delta={r['delta_com']:.4f}" for r in top_loss])
    (out_dir / "说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(split_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    labels = _case_labels(split_name)
    print(f"[flow_divider_compare] split={split_name} included={len(labels)}")
    for i, label in enumerate(labels, 1):
        rows.append(_load_pair(label))
        if i % 10 == 0 or i == len(labels):
            print(f"  compared {i}/{len(labels)}")
    csv_path = _write_csv(rows, out_dir)
    _plot_summary(rows, out_dir)
    _write_summary(rows, out_dir)
    print(f"[flow_divider_compare] report -> {out_dir}")
    print(f"  csv={csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=C.DEFAULT_SPLIT_NAME)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    run(args.split, Path(args.out_dir))


if __name__ == "__main__":
    main()
