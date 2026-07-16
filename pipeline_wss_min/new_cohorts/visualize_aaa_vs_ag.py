#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AAA-v4 与 AG-v4 共同物理框架并排目视验收。

把 AAA 数据层+坐标架双白名单单元（candidate_units）与 AG 正式纳入病例
（split_AG_wss_min_v1 的 train/val/test）都在同一 v4 STL-landmark 坐标架下
只读重算，放进同一物理尺度框架，出：
  - AG 参考小图（目标摆向）
  - AAA ruputer / unruputer 小图（与 AG 同尺）
  - AAA vs AG 叠加三视图（确认朝向包络一致，含 X-Y 判左右）
  - 逐例 v4 帧指标 + 异常旗标 CSV / summary
供人工签核"AAA 摆向是否与 AG 一致"。只读，不写 data_wss_min，不改磁盘 bundle。

配准配置走 config.registration_for_case（与 preprocess 一致，含逐例 override）；
v4 landmark 帧下左右号由 STL 世界 +X 定，AG 的 9 个 roll_sign_mode override 为
v3 遗留、在 v4 无效，故此图与 bundle 忠实一致。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_wss_min import config as C, raw_io, surface_io
from pipeline_wss_min.new_cohorts.common import candidate_units, split_unit
from pipeline_wss_min.preprocess import _resolve_unit_factor
from pipeline_wss_min.registration import compute_transform


OUT = C.PROJECT_ROOT / "docs" / "02-推进与变更" / "assets_新队列审计" / "alignment_v4_aaa_vs_ag"
POINTS_PER_CASE = 1400


def _cohort_case(unit_id: str) -> tuple[str, str]:
    """把 AAA/AG 的 unit_id 拆成 raw_case_dir 需要的 (cohort_rel, case_name)。"""
    parts = unit_id.split("/")
    if parts[0] == "AG" and len(parts) == 3:      # AG/slow/XXX or AG/fast/XXX
        return f"AG/{parts[1]}", parts[2]
    return split_unit(unit_id)                    # AAA/ruputer|unruputer/XXX


def _load(unit_id: str, seed: int) -> dict:
    cohort, case_name = _cohort_case(unit_id)
    case_dir = C.raw_case_dir(cohort, case_name)
    steps = raw_io.list_timesteps(case_dir)
    wall_native = raw_io.read_wall_geometry(case_dir, case_name, steps[0])
    cl = raw_io.read_centerline(case_dir)
    factor, _, _ = _resolve_unit_factor(wall_native, cl, C.DEFAULT.unit)
    wall = wall_native * factor
    stl = surface_io.select_case_surface(case_dir, wall, factor)
    reg = C.registration_for_case(cohort, case_name, C.DEFAULT.registration)
    trans = compute_transform(
        wall, wall, cl, reg,
        anatomy_pts=stl.points_mm, anatomy_source="original_stl")
    lm_world = np.vstack([
        trans.landmark_trunk_point, trans.landmark_left_point, trans.landmark_right_point])
    lm = trans.apply_points(lm_world)
    if len(wall) > POINTS_PER_CASE:
        idx = np.random.default_rng(seed).choice(len(wall), POINTS_PER_CASE, replace=False)
        wall = wall[idx]
    cloud = trans.apply_points(wall)
    return {
        "unit_id": unit_id,
        "cloud": cloud,
        "landmarks": lm,
        "repair": bool(trans.centerline_translation_applied),
        "origin_kind": str(trans.origin_kind),
        "fork_ratio": float(trans.fork_spread_ratio),
        "lr_sep": float(trans.lr_separation),
        "roll_reliable": bool(trans.roll_sign_reliable),
        "trunk_z_mm": float(lm[0, 2]),
        "left_z_mm": float(lm[1, 2]),
        "right_z_mm": float(lm[2, 2]),
        "lr_world_dx_mm": float(lm_world[1, 0] - lm_world[2, 0]),
    }


def _safe_load(args: tuple[str, int]) -> dict:
    unit_id, seed = args
    try:
        return _load(unit_id, seed)
    except Exception as exc:  # noqa: BLE001 - 单例失败不中断整批
        return {"unit_id": unit_id, "status": "error",
                "reason": f"{type(exc).__name__}: {exc}"}


def _flags(item: dict) -> list[str]:
    flags = []
    if item.get("repair"):
        flags.append("SHIFT")
    if "stl_bifurcation" in item.get("origin_kind", ""):
        flags.append("STL-O")
    if "axial_guard" in item.get("origin_kind", ""):
        flags.append("AX")
    if not item.get("roll_reliable", True):
        flags.append("ROLL?")
    if item.get("trunk_z_mm", 1.0) <= 0 or max(item.get("left_z_mm", -1.0),
                                               item.get("right_z_mm", -1.0)) >= 0:
        flags.append("ZBAD")
    if item.get("lr_world_dx_mm", 1.0) <= 0:
        flags.append("LR?")
    return flags


def _label(unit_id: str) -> str:
    parts = unit_id.split("/")
    return parts[-1][:15]


def _aaa_flag_counts(df: pd.DataFrame) -> dict:
    aaa = df[df["cohort"] == "AAA"]["flags"]
    return {tag: int(aaa.str.contains(pat, regex=True).sum())
            for tag, pat in (("SHIFT", "SHIFT"), ("STL-O", "STL-O"), ("AX", "AX"),
                             ("ROLL?", r"ROLL\?"), ("ZBAD", "ZBAD"), ("LR?", r"LR\?"))}


def _round_up(value: float, step: float = 25.0) -> float:
    return step * math.ceil(value / step)


def _small_multiples(items, title, path, xy_lim, z_abs, cols=8):
    rows = max(1, math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.05, rows * 2.35), squeeze=False)
    for ax, item in zip(axes.ravel(), items):
        cloud, lm = item["cloud"], item["landmarks"]
        ax.scatter(cloud[:, 0], cloud[:, 2], c=cloud[:, 2], cmap="viridis",
                   vmin=-z_abs, vmax=z_abs, s=0.7, linewidths=0)
        ax.scatter(lm[0, 0], lm[0, 2], marker="^", s=18, c="#d62728")   # trunk
        ax.scatter(lm[1, 0], lm[1, 2], marker="o", s=13, c="#ff7f0e")   # +X / left
        ax.scatter(lm[2, 0], lm[2, 2], marker="o", s=13, c="#1f77b4")   # -X / right
        ax.axhline(0, color="0.75", lw=0.5)
        ax.axvline(0, color="0.75", lw=0.5)
        flags = _flags(item)
        suffix = f" [{'|'.join(flags)}]" if flags else ""
        hard = any(f in flags for f in ("ZBAD", "LR?", "ROLL?"))
        ax.set_title(_label(item["unit_id"]) + suffix, fontsize=6.5,
                     color="#b22222" if hard else ("#c07000" if flags else "black"))
        ax.set_xlim(-xy_lim, xy_lim)
        ax.set_ylim(-z_abs, z_abs)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _overlay(aaa_items, ag_items, path, xy_lim, z_abs):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    planes = [(0, 2, "X", "Z"), (1, 2, "Y", "Z"), (0, 1, "X", "Y")]
    for ax, (a, b, la, lb) in zip(axes, planes):
        ag_cloud = np.vstack([it["cloud"] for it in ag_items])
        aaa_cloud = np.vstack([it["cloud"] for it in aaa_items])
        ax.scatter(ag_cloud[:, a], ag_cloud[:, b], s=0.3, alpha=0.05,
                   color="0.45", linewidths=0, label="AG (v4 ref)")
        ax.scatter(aaa_cloud[:, a], aaa_cloud[:, b], s=0.3, alpha=0.06,
                   color="#d62728", linewidths=0, label="AAA (v4)")
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvline(0, color="k", lw=0.6, ls="--")
        lim_a = z_abs if a == 2 else xy_lim
        lim_b = z_abs if b == 2 else xy_lim
        ax.set_xlim(-lim_a, lim_a); ax.set_ylim(-lim_b, lim_b)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{la} (mm)"); ax.set_ylabel(f"{lb} (mm)")
    axes[0].legend(markerscale=12, fontsize=9, framealpha=0.9)
    fig.suptitle("AAA vs AG · one rigid v4 anatomical frame · +Z proximal aorta / -Z iliac / +X patient-left",
                 y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="调试：每队列只取前 N 例")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    aaa_units = candidate_units()
    aaa_units = [u for u in aaa_units if u.startswith("AAA/")]
    ag_labels = C.split_case_labels(C.DEFAULT_SPLIT_NAME, ("train", "val", "test"))
    ag_units = [f"AG/{label}" for label in ag_labels]
    if args.limit is not None:
        aaa_units = aaa_units[:args.limit]
        ag_units = ag_units[:args.limit]

    jobs = [(u, 20260715 + i) for i, u in enumerate(aaa_units + ag_units)]
    results: dict[str, dict] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, item in enumerate(pool.map(_safe_load, jobs), 1):
            results[item["unit_id"]] = item
            note = item.get("reason", "|".join(_flags(item)) or "ok")
            print(f"[{i:3d}/{len(jobs)}] {item['unit_id']} {note}", flush=True)

    errors = [u for u, it in results.items() if it.get("status") == "error"]
    ok_items = [it for it in results.values() if it.get("status") != "error"]
    aaa_items = [results[u] for u in aaa_units if results[u].get("status") != "error"]
    ag_items = [results[u] for u in ag_units if results[u].get("status") != "error"]

    all_points = np.vstack([it["cloud"] for it in ok_items])
    xy_lim = _round_up(float(np.quantile(np.abs(all_points[:, :2]), 0.995)))
    z_abs = _round_up(float(np.quantile(np.abs(all_points[:, 2]), 0.995)))

    _small_multiples(
        ag_items,
        f"AG reference · STL landmark v4 · COMMON PHYSICAL FRAME (mm) · "
        f"+Z=proximal aorta, -Z=iliac branches, +X=patient-left · n={len(ag_items)}",
        OUT / "AG_reference_common_mm_xz.png", xy_lim, z_abs)
    for group, prefix in (("AAA_ruputer", "AAA/ruputer/"), ("AAA_unruputer", "AAA/unruputer/")):
        subset = [it for it in aaa_items if it["unit_id"].startswith(prefix)]
        _small_multiples(
            subset,
            f"{group} · v4 vs AG common frame (mm) · +Z proximal / -Z iliac / +X patient-left · "
            f"n={len(subset)}",
            OUT / f"{group}_common_mm_xz.png", xy_lim, z_abs)
    _overlay(aaa_items, ag_items, OUT / "AAA_vs_AG_ortho_overlay_common_mm.png", xy_lim, z_abs)

    metric_cols = ["unit_id", "origin_kind", "fork_ratio", "lr_sep", "roll_reliable",
                   "trunk_z_mm", "left_z_mm", "right_z_mm", "lr_world_dx_mm", "repair"]
    rows = []
    for it in aaa_items + ag_items:
        row = {k: it.get(k) for k in metric_cols}
        row["cohort"] = "AAA" if it["unit_id"].startswith("AAA/") else "AG"
        row["flags"] = "|".join(_flags(it))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "aaa_vs_ag_frame_metrics.csv", index=False)

    aaa_flagged = df[(df["cohort"] == "AAA") & (df["flags"] != "")]["unit_id"].tolist()
    summary = {
        "convention": "+Z proximal aorta; -Z iliac branches; +X original STL world +X (patient-left)",
        "frame_version": "stl_landmarks_v4",
        "n_aaa": len(aaa_items),
        "n_ag_ref": len(ag_items),
        "limits_mm": {"xy": xy_lim, "z": z_abs},
        "errors": errors,
        "aaa_flagged_units": aaa_flagged,
        "aaa_flag_counts": _aaa_flag_counts(df),
        "per_case_visual_rescaling": False,
    }
    (OUT / "aaa_vs_ag_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"figures + csv -> {OUT}")


if __name__ == "__main__":
    main()
