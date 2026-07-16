#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AAA/ILO v4 坐标架共同物理尺度可视化（原始 STL 关键点版）。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_wss_min import config as C, raw_io, surface_io
from pipeline_wss_min.new_cohorts.common import CLASSIFIED, PASS_CATEGORIES, split_unit
from pipeline_wss_min.preprocess import _resolve_unit_factor
from pipeline_wss_min.registration import compute_transform


OUT = C.PROJECT_ROOT / "docs" / "02-推进与变更" / "assets_新队列审计" / "alignment_v4"
POINTS_PER_CASE = 1400


def _load(unit_id: str, seed: int):
    cohort, case_name = split_unit(unit_id)
    case_dir = C.raw_case_dir(cohort, case_name)
    steps = raw_io.list_timesteps(case_dir)
    wall_native = raw_io.read_wall_geometry(case_dir, case_name, steps[0])
    cl = raw_io.read_centerline(case_dir)
    factor, _, _ = _resolve_unit_factor(wall_native, cl, C.DEFAULT.unit)
    wall = wall_native * factor
    stl = surface_io.select_case_surface(case_dir, wall, factor)
    trans = compute_transform(
        wall, wall, cl, C.DEFAULT.registration,
        anatomy_pts=stl.points_mm, anatomy_source="original_stl")
    if len(wall) > POINTS_PER_CASE:
        idx = np.random.default_rng(seed).choice(len(wall), POINTS_PER_CASE, replace=False)
        wall = wall[idx]
    cloud = trans.apply_points(wall)
    landmarks = trans.apply_points(np.vstack([
        trans.landmark_trunk_point, trans.landmark_left_point, trans.landmark_right_point]))
    return {
        "unit_id": unit_id,
        "cloud": cloud,
        "landmarks": landmarks,
        "repair": trans.centerline_translation_applied,
        "origin_kind": trans.origin_kind,
    }


def _group(unit_id: str):
    if unit_id.startswith("AAA/ruputer/"):
        return "AAA_ruputer"
    if unit_id.startswith("AAA/unruputer/"):
        return "AAA_unruputer"
    return "ILO_before" if unit_id.endswith("/before") else "ILO_after"


def _label(unit_id: str):
    parts = unit_id.split("/")
    if parts[0] == "AAA":
        return parts[-1][:15]
    return f"{parts[1][:12]}/{parts[2][0]}"


def _round_up(value: float, step: float = 25.0):
    return step * math.ceil(value / step)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    classified = pd.read_csv(CLASSIFIED)
    units = classified[classified["cat"].isin(PASS_CATEGORIES)]["unit_id"].tolist()
    items = []
    for i, unit in enumerate(units, 1):
        items.append(_load(unit, 20260714 + i))
        print(f"[{i:3d}/{len(units)}] {unit}", flush=True)

    all_points = np.vstack([item["cloud"] for item in items])
    xy_lim = _round_up(float(np.quantile(np.abs(all_points[:, :2]), 0.995)))
    z_abs = _round_up(float(np.quantile(np.abs(all_points[:, 2]), 0.995)))
    limits = {"xy": xy_lim, "z": z_abs}

    groups = ["AAA_ruputer", "AAA_unruputer", "ILO_before", "ILO_after"]
    for group in groups:
        subset = [item for item in items if _group(item["unit_id"]) == group]
        cols = 8
        rows = math.ceil(len(subset) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.05, rows * 2.35), squeeze=False)
        for ax, item in zip(axes.ravel(), subset):
            cloud = item["cloud"]
            lm = item["landmarks"]
            ax.scatter(cloud[:, 0], cloud[:, 2], c=cloud[:, 2], cmap="viridis",
                       vmin=-z_abs, vmax=z_abs, s=0.7, linewidths=0)
            ax.scatter(lm[0, 0], lm[0, 2], marker="^", s=18, c="#d62728", label="trunk")
            ax.scatter(lm[1, 0], lm[1, 2], marker="o", s=13, c="#ff7f0e", label="+X/left")
            ax.scatter(lm[2, 0], lm[2, 2], marker="o", s=13, c="#1f77b4", label="-X/right")
            ax.axhline(0, color="0.75", lw=0.5)
            ax.axvline(0, color="0.75", lw=0.5)
            flags = []
            if item["repair"]:
                flags.append("SHIFT")
            if "stl_bifurcation" in item["origin_kind"]:
                flags.append("STL-O")
            suffix = f" [{'|'.join(flags)}]" if flags else ""
            ax.set_title(_label(item["unit_id"]) + suffix, fontsize=6.5,
                         color="#b22222" if flags else "black")
            ax.set_xlim(-xy_lim, xy_lim)
            ax.set_ylim(-z_abs, z_abs)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.ravel()[len(subset):]:
            ax.axis("off")
        fig.suptitle(
            f"{group} · STL landmark v4 · COMMON PHYSICAL FRAME (mm) · "
            f"+Z=proximal aorta, -Z=iliac branches · n={len(subset)}",
            fontsize=11, y=0.998)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(OUT / f"{group}_common_mm_xz.png", dpi=150)
        plt.close(fig)

    colors = dict(zip(groups, ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    planes = [(0, 2, "X", "Z"), (1, 2, "Y", "Z"), (0, 1, "X", "Y")]
    for ax, (a, b, la, lb) in zip(axes, planes):
        seen = set()
        for item in items:
            group = _group(item["unit_id"])
            label = group if group not in seen else None
            seen.add(group)
            cloud = item["cloud"]
            ax.scatter(cloud[:, a], cloud[:, b], s=0.35, alpha=0.08,
                       color=colors[group], linewidths=0, label=label)
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvline(0, color="k", lw=0.6, ls="--")
        lim_a = z_abs if a == 2 else xy_lim
        lim_b = z_abs if b == 2 else xy_lim
        ax.set_xlim(-lim_a, lim_a)
        ax.set_ylim(-lim_b, lim_b)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{la} (mm)")
        ax.set_ylabel(f"{lb} (mm)")
    axes[0].legend(markerscale=7, fontsize=8)
    fig.suptitle("AAA + ILO · one rigid anatomical frame · no per-case visual rescaling", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "AAA_ILO_all_ortho_overlay_common_mm.png", dpi=170)
    plt.close(fig)
    (OUT / "visualization_summary.json").write_text(json.dumps({
        "n_units": len(items), "limits_mm": limits,
        "convention": "+Z proximal aorta; -Z iliac branches; +X original STL world +X",
        "per_case_visual_rescaling": False,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
