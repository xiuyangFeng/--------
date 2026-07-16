#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""坐标规整体检图（对标旧 outputs/coord_check_*）——用于 pipeline_wss_min 新对齐流程。

对 split included 病例，逐病例算 配准+归一化，并读峰值收缩期壁面 WSS 上色，产出：
- front_view_pageN.png : 4x4 小图，正视图 y-z，颜色=WSS，红线=原点
- top_view_pageN.png   : 4x4 小图，俯视图 x-y，颜色=WSS，红线=原点
- raw_centroids.png    : 各病例原始(世界系)壁面重心散布
- raw_vs_norm_examples.png : 若干例 原始 vs 归一化(WSS上色) 对照
- coord_check_summary.csv   : 逐病例坐标/配准统计

历史复核入口：
``python -m pipeline_wss_min.archive.alignment_v3.coord_check``
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(_p).is_file():
        try:
            matplotlib.font_manager.fontManager.addfont(_p)
            matplotlib.rcParams["font.family"] = \
                matplotlib.font_manager.FontProperties(fname=_p).get_name()
        except Exception:  # noqa: BLE001
            pass
matplotlib.rcParams["axes.unicode_minus"] = False

from pipeline_wss_min import config as C
from pipeline_wss_min import raw_io
from pipeline_wss_min.archive.alignment_v3 import legacy_registration_for_case
from pipeline_wss_min.preprocess import _resolve_unit_factor
from pipeline_wss_min.registration import compute_transform

OUT_DIR = C.PROJECT_ROOT / "outputs" / "wss_min" / "coord_check_20260707_anatomical_v2"
PER_PAGE = 16          # 4x4
PLOT_POINTS = 6000     # 每病例绘图上限


def _load_included(split_name):
    import json
    sp = json.loads((C.PROJECT_ROOT / "training" / "splits" / f"{split_name}.json").read_text())
    cases = sp.get("train_cases", []) + sp.get("val_cases", []) + sp.get("test_cases", [])
    return [("AG/" + c.split("/")[0], c.split("/")[1], c) for c in cases]


def _case_data(coh, cn, cfg):
    """返回 dict: raw_centroid(mm), norm_coords(N,3), wss(N,), scale, factor, anomaly, rot_det, orth_err。"""
    cd = C.raw_case_dir(coh, cn)
    steps = raw_io.list_timesteps(cd)
    ref = steps[0]
    wall_native = raw_io.read_wall_geometry(cd, cn, ref)
    cl = raw_io.read_centerline(cd)
    factor, anomaly, _ = _resolve_unit_factor(wall_native, cl, cfg.unit)
    wall = wall_native * factor
    reg_cfg = legacy_registration_for_case(coh, cn, cfg.registration)
    T = compute_transform(wall, wall, cl, reg_cfg)
    aligned = (wall - T.centroid) @ T.rotation
    scale = float(np.abs(aligned).max()) or 1.0
    norm = aligned / scale
    peak = raw_io.peak_systole_step(cd, steps)
    wss = raw_io.read_wall_fields(cd, cn, peak)["wss"]
    R = T.rotation
    return dict(
        raw_centroid=wall.mean(axis=0), norm=norm, wss=wss,
        scale=scale, factor=factor, anomaly=anomaly, peak=peak,
        rot_det=float(np.linalg.det(R)),
        orth_err=float(np.abs(R.T @ R - np.eye(3)).max()),
        n_wall=len(wall),
        origin_kind=T.origin_kind,
        main_axis_mode=T.main_axis_mode,
        main_axis_source=T.main_axis_source,
        roll_source=T.roll_source,
        trunk_centering_applied=bool(T.trunk_centering_applied),
        trunk_centering_offset_frac=float(T.trunk_centering_offset_frac),
        origin=T.centroid,
    )


def _subsample(n_tot, seed):
    if n_tot <= PLOT_POINTS:
        return np.arange(n_tot)
    return np.random.default_rng(seed).choice(n_tot, PLOT_POINTS, replace=False)


def _grid_pages(items, view, out_prefix):
    """按 4×4 分页绘制“病例标签、二维坐标、WSS”列表的指定视图。"""
    n = len(items)
    pages = math.ceil(n / PER_PAGE)
    for pg in range(pages):
        chunk = items[pg * PER_PAGE:(pg + 1) * PER_PAGE]
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        for ax in axes:
            ax.axis("off")
        for k, (label, xy, wss) in enumerate(chunk):
            ax = axes[k]; ax.axis("on")
            vmax = np.quantile(wss, 0.97) if len(wss) else 1.0
            ax.scatter(xy[:, 0], xy[:, 1], c=wss, cmap="viridis", s=1.5,
                       vmin=0, vmax=max(vmax, 1e-6), linewidths=0)
            ax.axhline(0, color="red", lw=0.8, alpha=0.7)
            ax.axvline(0, color="red", lw=0.8, alpha=0.7)
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal")
            ax.set_title(label, fontsize=9)
        fig.suptitle(f"归一化壁面点 {view} 视图（红线=原点，颜色=WSS峰值） page {pg+1}/{pages}",
                     fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(OUT_DIR / f"{out_prefix}_page{pg+1}.png", dpi=110)
        plt.close(fig)


def _fig_raw_centroids(cases_stat):
    cents = np.array([s["raw_centroid"] for s in cases_stat])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (a, b, la, lb) in zip(axes, [(0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z")]):
        ax.scatter(cents[:, a], cents[:, b], s=18, c="tab:blue", alpha=0.7)
        ax.set_xlabel(f"{la} (mm)"); ax.set_ylabel(f"{lb} (mm)")
        ax.set_title(f"原始重心 {la}-{lb}"); ax.grid(alpha=0.3)
    fig.suptitle("各病例原始(世界系)壁面重心散布 —— 解剖原点配准前位置各异", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "raw_centroids.png", dpi=120)
    plt.close(fig)


def _fig_raw_vs_norm(examples):
    """绘制病例原始 Y-Z 坐标与归一化 Y-Z 坐标的对照。"""
    m = len(examples)
    fig, axes = plt.subplots(2, m, figsize=(4 * m, 9))
    for j, (label, raw, norm, wss) in enumerate(examples):
        ax0 = axes[0, j]
        ax0.scatter(raw[:, 0], raw[:, 1], s=1.2, c="tab:blue", linewidths=0)
        ax0.set_title(f"{label}\n原始 y-z (mm)", fontsize=9); ax0.set_aspect("equal")
        ax1 = axes[1, j]
        vmax = np.quantile(wss, 0.97) if len(wss) else 1.0
        ax1.scatter(norm[:, 0], norm[:, 1], s=1.2, c=wss, cmap="viridis",
                    vmin=0, vmax=max(vmax, 1e-6), linewidths=0)
        ax1.axhline(0, color="red", lw=0.8); ax1.axvline(0, color="red", lw=0.8)
        ax1.set_xlim(-1.05, 1.05); ax1.set_ylim(-1.05, 1.05); ax1.set_aspect("equal")
        ax1.set_title("归一化 y-z (颜色=WSS)", fontsize=9)
    fig.suptitle("原始 vs 解剖坐标归一化（壁面点，峰值步 WSS 上色）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "raw_vs_norm_examples.png", dpi=120)
    plt.close(fig)


def run(split_name="split_AG_wss_min_v1", tag=None):
    cfg = C.DEFAULT
    global OUT_DIR
    if tag:
        OUT_DIR = C.PROJECT_ROOT / "outputs" / "wss_min" / f"coord_check_{tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = _load_included(split_name)
    print(f"[coord_check] split={split_name} cases={len(cases)}")

    front_items, top_items, stats, examples = [], [], [], []
    for i, (coh, cn, label) in enumerate(cases):
        try:
            d = _case_data(coh, cn, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {label}: {e}"); continue
        idx = _subsample(d["n_wall"], i)
        norm, wss = d["norm"][idx], d["wss"][idx]
        front_items.append((label, norm[:, [1, 2]], wss))   # y-z
        top_items.append((label, norm[:, [0, 1]], wss))     # x-y
        stats.append((label, d))
        if len(examples) < 4:
            examples.append((label, coh, cn))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(cases)}")

    print(f"[coord_check] processed {len(front_items)} cases")
    _grid_pages(front_items, "y-z", "front_view")
    _grid_pages(top_items, "x-y", "top_view")
    _fig_raw_centroids([s for _, s in stats])

    # raw vs norm 例图：重新读 4 个例子的原始 y-z
    ex_pack = []
    for label, coh, cn in examples:
        cd = C.raw_case_dir(coh, cn); steps = raw_io.list_timesteps(cd)
        wall_native = raw_io.read_wall_geometry(cd, cn, steps[0])
        cl = raw_io.read_centerline(cd)
        f, _, _ = _resolve_unit_factor(wall_native, cl, cfg.unit)
        wall = wall_native * f
        d = _case_data(coh, cn, cfg)
        idx = _subsample(len(wall), 0)
        ex_pack.append((label, wall[idx][:, [1, 2]], d["norm"][idx][:, [1, 2]], d["wss"][idx]))
    _fig_raw_vs_norm(ex_pack)

    # 写出逐病例汇总表。
    with open(OUT_DIR / "coord_check_summary.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["case", "unit_factor", "unit_anomaly", "scale_mm", "rot_det",
                    "rot_orth_err", "origin_kind", "main_axis_mode", "main_axis_source", "roll_source",
                    "trunk_centering_applied", "trunk_centering_offset_frac",
                    "origin_x", "origin_y", "origin_z",
                    "n_wall", "peak_step", "coord_min", "coord_max",
                    "wss_min", "wss_max"])
        for label, d in stats:
            w.writerow([label, f"{d['factor']:.4g}", d["anomaly"], f"{d['scale']:.3f}",
                        f"{d['rot_det']:.6f}", f"{d['orth_err']:.2e}",
                        d["origin_kind"], d["main_axis_mode"], d["main_axis_source"], d["roll_source"],
                        d["trunk_centering_applied"], f"{d['trunk_centering_offset_frac']:.4f}",
                        f"{d['origin'][0]:.4f}", f"{d['origin'][1]:.4f}", f"{d['origin'][2]:.4f}",
                        d["n_wall"], d["peak"],
                        f"{d['norm'].min():.4f}", f"{d['norm'].max():.4f}",
                        f"{d['wss'].min():.4f}", f"{d['wss'].max():.4f}"])
    print(f"[coord_check] figures + summary -> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split_AG_wss_min_v1")
    ap.add_argument("--tag", default=None,
                    help="输出目录后缀：outputs/wss_min/coord_check_<tag>；缺省用 _20260707_anatomical_v2")
    args = ap.parse_args()
    run(args.split, args.tag)
