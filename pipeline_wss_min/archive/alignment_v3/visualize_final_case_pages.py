#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""当前 WSS-min split 的最终坐标 QA 分页图。

面向人工复核，不写 bundle：
- 点云同框：每 20 例叠加到同一坐标框架；
- STL 同框：每 20 例 STL 顶点/稀疏面片叠加到同一坐标框架；
- 逐病例 X-Z 小图：类似 alignment_viz 的 small multiples；
- 峰值 WSS 正视/俯视分页图：类似 coord_check 的 front/top view。
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
from pipeline_wss_min.archive.alignment_v3.visualize_stl_point_overlap import (
    _find_stl, _read_stl, _stl_index, _stl_scale_to_pipeline,
)


OUT_DIR = C.PROJECT_ROOT / "outputs" / "wss_min" / "flow_divider_origin_QA_78例"
STL_ROOT = C.PROJECT_ROOT / "stl_data"
PER_PAGE = 20
WALL_POINTS_PER_CASE = 1800
STL_VERTS_PER_CASE = 2200
STL_FACES_PER_CASE = 180
SMALL_POINTS_PER_CASE = 3500


def _sample_idx(n_total: int, n_keep: int, seed: int) -> np.ndarray:
    if n_total <= n_keep:
        return np.arange(n_total)
    return np.random.default_rng(seed).choice(n_total, n_keep, replace=False)


def _case_colors(n_cases: int):
    cmap = plt.get_cmap("tab20" if n_cases <= 20 else "turbo")
    return [cmap(i % 20) if n_cases <= 20 else cmap(i / max(n_cases - 1, 1))
            for i in range(n_cases)]


def _set_2d_frame(ax):
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.42)
    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.42)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])


def _set_3d_frame(ax):
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.view_init(elev=18, azim=-62)
    ax.set_box_aspect((1, 1, 1))
    ax.plot([-1.05, 1.05], [0, 0], [0, 0], color="black", lw=0.7, alpha=0.45)
    ax.plot([0, 0], [-1.05, 1.05], [0, 0], color="black", lw=0.7, alpha=0.45)
    ax.plot([0, 0], [0, 0], [-1.05, 1.05], color="black", lw=0.9, alpha=0.55)


def _load_cases(split_name: str) -> List[Tuple[str, str, str]]:
    labels = C.split_case_labels(split_name, ("train", "val", "test"))
    out = []
    for label in labels:
        subset, case_name = label.split("/", 1)
        out.append(("AG/" + subset, case_name, label))
    return out


def _load_case(
    cohort_rel: str,
    case_name: str,
    label: str,
    stl_paths: Dict[str, List[Path]],
    seed: int,
    center_on: str | None = None,
) -> dict:
    cfg = C.DEFAULT
    case_dir = C.raw_case_dir(cohort_rel, case_name)
    steps = raw_io.list_timesteps(case_dir)
    ref = steps[0]

    wall_native = raw_io.read_wall_geometry(case_dir, case_name, ref)
    int_native = raw_io.read_interior_geometry(case_dir, case_name, ref)
    centerline = raw_io.read_centerline(case_dir)

    unit_factor, unit_anomaly, unit_extent_mismatch = _resolve_unit_factor(
        wall_native, centerline, cfg.unit)
    wall_pts = wall_native * unit_factor
    int_pts = int_native * unit_factor
    reg_cfg = legacy_registration_for_case(cohort_rel, case_name, cfg.registration)
    if center_on:
        reg_cfg = replace(reg_cfg, center_on=center_on)
    transform = compute_transform(wall_pts, int_pts, centerline, reg_cfg)

    wall_aligned = transform.apply_points(wall_pts)
    int_aligned = transform.apply_points(int_pts)
    wall_aligned_for_crop = wall_aligned

    # 与 preprocess 一致：配准后、归一化前裁掉未描主动脉尾巴（覆盖一致病例为 no-op）
    cl_aligned = transform.apply_points(centerline["coords"])
    tgt = AXIS_INDEX[transform.principal_axis_target]
    wall_keep, _int_keep, crop_frac, crop_applied = untraced_inlet_crop_masks(
        wall_aligned, int_aligned, cl_aligned, centerline["abscissa"], tgt, reg_cfg)
    if crop_applied:
        wall_pts = wall_pts[wall_keep]
        wall_aligned = wall_aligned[wall_keep]
        int_aligned = int_aligned[_int_keep]

    ref = wall_aligned if cfg.normalization.coord_scale_on == "wall" \
        else np.vstack([wall_aligned, int_aligned])
    coord_scale = float(np.abs(ref).max()) or 1.0
    wall_norm = wall_aligned / coord_scale

    peak = raw_io.peak_systole_step(case_dir, steps)
    wss = raw_io.read_wall_fields(case_dir, case_name, peak)["wss"]
    if crop_applied:  # WSS 与裁剪后的壁面点保持同序同长
        wss = wss[wall_keep]

    stl_path = _find_stl(cohort_rel, case_name, stl_paths)
    stl_norm = None
    stl_faces = None
    native_to_stl = np.nan
    stl_to_pipeline = np.nan
    stl_scale_kind = ""
    if stl_path is not None:
        stl_raw, stl_faces = _read_stl(stl_path)
        native_to_stl, stl_to_pipeline, stl_scale_kind = _stl_scale_to_pipeline(
            wall_native, wall_pts, centerline, stl_raw, unit_factor, unit_extent_mismatch)
        stl_pts = stl_raw * stl_to_pipeline
        stl_aligned = transform.apply_points(stl_pts)
        if crop_applied:
            _, stl_keep, _, _ = untraced_inlet_crop_masks(
                wall_aligned_for_crop, stl_aligned, cl_aligned,
                centerline["abscissa"], tgt, reg_cfg)
            stl_aligned = stl_aligned[stl_keep]
            stl_faces = None
        stl_norm = stl_aligned / coord_scale

    return {
        "label": label,
        "cohort": cohort_rel,
        "case": case_name,
        "wall_norm": wall_norm.astype(np.float32),
        "wss": wss.astype(np.float32),
        "stl_norm": None if stl_norm is None else stl_norm.astype(np.float32),
        "stl_faces": stl_faces,
        "stl_path": stl_path,
        "unit_factor": float(unit_factor),
        "unit_anomaly": bool(unit_anomaly),
        "unit_extent_mismatch": bool(unit_extent_mismatch),
        "wall_crop_applied": bool(crop_applied),
        "wall_crop_frac": float(crop_frac),
        "native_to_stl": float(native_to_stl) if np.isfinite(native_to_stl) else "",
        "stl_to_pipeline": float(stl_to_pipeline) if np.isfinite(stl_to_pipeline) else "",
        "stl_scale_kind": stl_scale_kind,
        "coord_scale": float(coord_scale),
        "coord_scale_on": cfg.normalization.coord_scale_on,
        "origin_kind": transform.origin_kind,
        "center_on": reg_cfg.center_on,
        "main_axis_mode": transform.main_axis_mode,
        "main_axis_source": transform.main_axis_source,
        "roll_source": transform.roll_source,
        "roll_sign_source": transform.roll_sign_source,
        "roll_sign_reliable": bool(transform.roll_sign_reliable),
        "roll_sign_cos": float(transform.roll_sign_cos),
        "trunk_centering_applied": bool(transform.trunk_centering_applied),
        "trunk_centering_offset_mm": transform.trunk_centering_offset_mm.astype(np.float64),
        "trunk_centering_offset_frac": float(transform.trunk_centering_offset_frac),
        "peak_step": int(peak),
        "n_wall": int(len(wall_norm)),
        "n_stl_vertices": 0 if stl_norm is None else int(len(stl_norm)),
        "seed": seed,
    }


def _chunks(items: List[dict], n: int = PER_PAGE):
    for i in range(0, len(items), n):
        yield i // n + 1, items[i:i + n]


def _page_case_range(page_idx: int, chunk: List[dict]) -> str:
    start = (page_idx - 1) * PER_PAGE + 1
    end = start + len(chunk) - 1
    return f"{start:02d}-{end:02d}"


def _draw_same_frame_page(chunk: List[dict], page_idx: int, total_pages: int,
                          total_cases: int, source: str, out_dir: Path, seed: int) -> Path:
    is_wall = source == "wall"
    title_source = "壁面点云" if is_wall else "STL 顶点/面片"
    colors = _case_colors(len(chunk))
    planes = [(0, 2, "X-Z 主轴视图"), (1, 2, "Y-Z 主轴视图"), (0, 1, "X-Y 横截面视图")]

    fig = plt.figure(figsize=(16.5, 11.4))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    proj_axes = [fig.add_subplot(2, 2, i) for i in (2, 3, 4)]
    missing = []

    for i, (item, color) in enumerate(zip(chunk, colors)):
        pts = item["wall_norm"] if is_wall else item["stl_norm"]
        if pts is None:
            missing.append(item["label"])
            continue

        n_keep = WALL_POINTS_PER_CASE if is_wall else STL_VERTS_PER_CASE
        idx = _sample_idx(len(pts), n_keep, seed + page_idx * 1009 + i * 53)
        if is_wall:
            ax3d.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                         s=0.85, color=color, alpha=0.54, linewidths=0)
        else:
            faces = item["stl_faces"]
            if faces is not None and len(faces) > 0:
                fidx = _sample_idx(len(faces), STL_FACES_PER_CASE,
                                   seed + page_idx * 1009 + i * 59)
                mesh = Poly3DCollection(
                    pts[faces[fidx]],
                    facecolor=(color[0], color[1], color[2], 0.08),
                    edgecolor=(color[0], color[1], color[2], 0.035),
                    linewidths=0.03,
                )
                ax3d.add_collection3d(mesh)
            ax3d.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                         s=0.55, color=color, alpha=0.23, linewidths=0)

        for ax, (a, b, plane_title) in zip(proj_axes, planes):
            ax.scatter(pts[idx, a], pts[idx, b],
                       s=0.5 if is_wall else 0.65,
                       color=color, alpha=0.48 if is_wall else 0.33,
                       linewidths=0)
            ax.set_title(plane_title, fontsize=11)

    _set_3d_frame(ax3d)
    ax3d.set_title(f"{title_source} 同框 3D 视角", fontsize=11, pad=8)
    for ax in proj_axes:
        _set_2d_frame(ax)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[i],
               markersize=5, label=item["label"])
        for i, item in enumerate(chunk)
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=5, fontsize=7, frameon=False)
    miss = f"；缺 STL：{', '.join(missing)}" if missing else ""
    fig.suptitle(
        f"当前 split included {total_cases} 例：{title_source} 同一坐标框架叠加 "
        f"page {page_idx}/{total_pages}（病例 {_page_case_range(page_idx, chunk)}）{miss}",
        fontsize=13, y=0.985,
    )
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    name = "点云同框" if is_wall else "STL同框"
    out = out_dir / f"page{page_idx:02d}_{name}_病例{_page_case_range(page_idx, chunk)}.png"
    fig.savefig(out, dpi=155)
    plt.close(fig)
    return out


def _draw_small_multiples_page(chunk: List[dict], page_idx: int, total_pages: int,
                               total_cases: int, out_dir: Path, seed: int) -> Path:
    fig, axes = plt.subplots(4, 5, figsize=(12.5, 13.5))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for i, (ax, item) in enumerate(zip(axes, chunk)):
        ax.axis("on")
        pts = item["wall_norm"]
        idx = _sample_idx(len(pts), SMALL_POINTS_PER_CASE, seed + page_idx * 701 + i * 31)
        pc = pts[idx]
        ax.scatter(pc[:, 0], pc[:, 2], s=0.65, c=pc[:, 2], cmap="viridis", linewidths=0)
        flags = []
        if item["unit_anomaly"]:
            flags.append("单位")
        if item["wall_crop_applied"]:
            flags.append(f"裁剪{item['wall_crop_frac']:.0%}")
        if item["main_axis_source"] != "centerline_chord":
            flags.append(item["main_axis_source"])
        if item["roll_source"] in ("branches", "curvature"):
            flags.append(item["roll_source"])
        if not item["roll_sign_reliable"]:
            flags.append("LR待复核")
        if item["trunk_centering_applied"]:
            flags.append(f"主干居中{item['trunk_centering_offset_frac']:.2f}")
        flag = " | " + "/".join(flags) if flags else ""
        ax.set_title(f"{item['label']}{flag}", fontsize=7)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhline(0, color="black", lw=0.35, ls="--", alpha=0.38)
        ax.axvline(0, color="black", lw=0.35, ls="--", alpha=0.38)
    fig.suptitle(
        f"当前 split included {total_cases} 例：逐病例 X-Z 主轴视图 page {page_idx}/{total_pages} "
        f"（病例 {_page_case_range(page_idx, chunk)}）",
        fontsize=13, y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = out_dir / f"page{page_idx:02d}_逐病例XZ_病例{_page_case_range(page_idx, chunk)}.png"
    fig.savefig(out, dpi=145)
    plt.close(fig)
    return out


def _draw_wss_view_page(chunk: List[dict], page_idx: int, total_pages: int,
                        total_cases: int, view: str, out_dir: Path, seed: int) -> Path:
    if view == "front":
        dims = (1, 2)
        title = "正视图 Y-Z"
        filename = "正视图YZ_WSS"
    else:
        dims = (0, 1)
        title = "俯视图 X-Y"
        filename = "俯视图XY_WSS"

    fig, axes = plt.subplots(4, 5, figsize=(14, 14))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for i, (ax, item) in enumerate(zip(axes, chunk)):
        ax.axis("on")
        pts = item["wall_norm"]
        idx = _sample_idx(len(pts), SMALL_POINTS_PER_CASE, seed + page_idx * 809 + i * 37)
        pc = pts[idx]
        wss = item["wss"][idx]
        vmax = np.quantile(wss, 0.97) if len(wss) else 1.0
        ax.scatter(pc[:, dims[0]], pc[:, dims[1]], s=1.0, c=wss, cmap="viridis",
                   vmin=0, vmax=max(vmax, 1e-6), linewidths=0)
        ax.axhline(0, color="red", lw=0.55, alpha=0.65)
        ax.axvline(0, color="red", lw=0.55, alpha=0.65)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{item['label']}\npeak={item['peak_step']} p97={vmax:.2g}", fontsize=7)
    fig.suptitle(
        f"当前 split included {total_cases} 例：峰值 WSS {title} page {page_idx}/{total_pages} "
        f"（红线=原点，病例 {_page_case_range(page_idx, chunk)}）",
        fontsize=13, y=0.993,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    out = out_dir / f"page{page_idx:02d}_{filename}_病例{_page_case_range(page_idx, chunk)}.png"
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out


def _draw_trunk_centering_review(items: List[dict], out_dir: Path, seed: int) -> Path | None:
    focused = [item for item in items if item["trunk_centering_applied"]]
    if not focused:
        return None

    cols = 4
    rows = len(focused)
    fig = plt.figure(figsize=(17.0, max(4.2, rows * 3.9)))
    planes = [(0, 2, "X-Z 主轴视图"), (1, 2, "Y-Z 主轴视图"), (0, 1, "X-Y 横截面")]

    for r, item in enumerate(focused):
        wall = item["wall_norm"]
        stl = item["stl_norm"]
        faces = item["stl_faces"]
        wall_idx = _sample_idx(len(wall), min(6500, len(wall)), seed + r * 101)
        stl_idx = None if stl is None else _sample_idx(len(stl), min(6500, len(stl)), seed + r * 103)

        ax3d = fig.add_subplot(rows, cols, r * cols + 1, projection="3d")
        if stl is not None and faces is not None and len(faces) > 0:
            fidx = _sample_idx(len(faces), min(900, len(faces)), seed + r * 107)
            mesh = Poly3DCollection(
                stl[faces[fidx]],
                facecolor=(0.20, 0.42, 0.75, 0.12),
                edgecolor=(0.20, 0.42, 0.75, 0.055),
                linewidths=0.03,
            )
            ax3d.add_collection3d(mesh)
            ax3d.scatter(stl[stl_idx, 0], stl[stl_idx, 1], stl[stl_idx, 2],
                         s=0.6, color="tab:blue", alpha=0.20, linewidths=0)
        ax3d.scatter(wall[wall_idx, 0], wall[wall_idx, 1], wall[wall_idx, 2],
                     s=0.8, color="tab:orange", alpha=0.62, linewidths=0)
        _set_3d_frame(ax3d)
        off = item["trunk_centering_offset_mm"]
        ax3d.set_title(
            f"{item['label']}\n主干二次居中 frac={item['trunk_centering_offset_frac']:.3f} "
            f"offset=({off[0]:.1f},{off[1]:.1f},{off[2]:.1f})mm",
            fontsize=9,
        )

        for c, (a, b, title) in enumerate(planes, start=2):
            ax = fig.add_subplot(rows, cols, r * cols + c)
            if stl is not None and stl_idx is not None:
                ax.scatter(stl[stl_idx, a], stl[stl_idx, b],
                           s=0.8, color="tab:blue", alpha=0.25, linewidths=0,
                           label="STL")
            ax.scatter(wall[wall_idx, a], wall[wall_idx, b],
                       s=0.8, color="tab:orange", alpha=0.58, linewidths=0,
                       label="wall")
            _set_2d_frame(ax)
            ax.set_title(title, fontsize=9)
            if r == 0 and c == 4:
                ax.legend(loc="upper right", fontsize=7, frameon=False)

    fig.suptitle("主干横向二次居中触发病例：STL + 壁面点云三视图复核", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = out_dir / "二次居中病例_STL点云三视图.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _write_summary(items: List[dict], out_dir: Path) -> None:
    with open(out_dir / f"当前{len(items)}例_坐标QA汇总.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case", "cohort", "n_wall", "stl_status", "n_stl_vertices",
            "unit_factor", "unit_anomaly", "unit_extent_mismatch",
            "wall_crop_applied", "wall_crop_frac",
            "coord_scale", "coord_scale_on", "stl_scale_kind", "center_on", "origin_kind",
            "main_axis_mode", "main_axis_source", "roll_source", "roll_sign_source",
            "roll_sign_reliable", "roll_sign_cos",
            "trunk_centering_applied", "trunk_centering_offset_frac",
            "trunk_centering_offset_x_mm", "trunk_centering_offset_y_mm",
            "trunk_centering_offset_z_mm",
            "peak_step", "stl_path",
        ])
        for item in items:
            off = item["trunk_centering_offset_mm"]
            writer.writerow([
                item["label"], item["cohort"], item["n_wall"],
                "ok" if item["stl_norm"] is not None else "missing",
                item["n_stl_vertices"], f"{item['unit_factor']:.8g}",
                item["unit_anomaly"], item["unit_extent_mismatch"],
                item["wall_crop_applied"], f"{item['wall_crop_frac']:.8g}",
                f"{item['coord_scale']:.8g}", item["coord_scale_on"], item["stl_scale_kind"],
                item["center_on"], item["origin_kind"], item["main_axis_mode"], item["main_axis_source"],
                item["roll_source"], item["roll_sign_source"], item["roll_sign_reliable"],
                f"{item['roll_sign_cos']:.8g}",
                item["trunk_centering_applied"],
                f"{item['trunk_centering_offset_frac']:.8g}",
                f"{off[0]:.8g}", f"{off[1]:.8g}", f"{off[2]:.8g}",
                item["peak_step"],
                "" if item["stl_path"] is None else str(item["stl_path"]),
            ])


def run(split_name: str, out_dir: Path, seed: int, center_on: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subdirs = {
        "wall": out_dir / "01_点云同框_每20例",
        "stl": out_dir / "02_STL同框_每20例",
        "xz": out_dir / "03_逐病例XZ主轴视图_每20例",
        "front": out_dir / "04_峰值WSS正视图YZ_每20例",
        "top": out_dir / "05_峰值WSS俯视图XY_每20例",
        "trunk_centering": out_dir / "07_二次居中病例复核",
    }
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(split_name)
    stl_paths = _stl_index()
    items: List[dict] = []
    print(f"[final_qa] split={split_name} included={len(cases)}")
    for i, (cohort, case_name, label) in enumerate(cases):
        item = _load_case(cohort, case_name, label, stl_paths, seed + i * 101, center_on=center_on)
        items.append(item)
        if (i + 1) % 10 == 0 or i + 1 == len(cases):
            print(f"  loaded {i + 1}/{len(cases)}")

    pages = math.ceil(len(items) / PER_PAGE)
    total_cases = len(items)
    for page_idx, chunk in _chunks(items):
        _draw_same_frame_page(chunk, page_idx, pages, total_cases, "wall", subdirs["wall"], seed)
        _draw_same_frame_page(chunk, page_idx, pages, total_cases, "stl", subdirs["stl"], seed)
        _draw_small_multiples_page(chunk, page_idx, pages, total_cases, subdirs["xz"], seed)
        _draw_wss_view_page(chunk, page_idx, pages, total_cases, "front", subdirs["front"], seed)
        _draw_wss_view_page(chunk, page_idx, pages, total_cases, "top", subdirs["top"], seed)
        print(f"  drew page {page_idx}/{pages}")
    review_path = _draw_trunk_centering_review(items, subdirs["trunk_centering"], seed)
    _write_summary(items, out_dir)
    if review_path is not None:
        print(f"  drew trunk-centering review: {review_path}")
    print(f"[final_qa] figures + summary -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=C.DEFAULT_SPLIT_NAME)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--center-on", choices=["bifurcation", "flow_divider", "wall", "all", "centerline"],
                    default=None, help="临时覆盖 RegistrationConfig.center_on，用于原点方案对照")
    args = ap.parse_args()
    run(args.split, Path(args.out_dir), args.seed, center_on=args.center_on)


if __name__ == "__main__":
    main()
