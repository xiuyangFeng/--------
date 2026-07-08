#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""随机抽病例，检查壁面点云与 STL 是否处在同一解剖坐标框架。

本脚本不做 ICP 或额外刚性拟合。流程与 preprocess 保持一致：
1. 壁面/内部 ascii -> mesh 单位换算；
2. 读取中心线并计算同一个分叉原点刚性配准；
3. STL 只按 ascii/STL 原始单位比例换到同一 pipeline 坐标尺度；
4. 点云与 STL 套用同一个 transform、同一个 per-case 归一化尺度；
5. 画叠加图并输出 overlap 距离，核查 x,y,z 输入空间是否规整。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.neighbors import NearestNeighbors

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

from . import config as C
from . import raw_io
from .preprocess import _resolve_unit_factor
from .registration import compute_transform


OUT_DIR = C.PROJECT_ROOT / "outputs" / "wss_min" / "stl_point_overlap_20260707"
STL_ROOT = C.PROJECT_ROOT / "stl_data"
PLOT_POINTS = 3500
PLOT_STL_VERTS = 4500
PLOT_FACES = 2200
OVERLAY_POINTS_PER_CASE = 2600
OVERLAY_STL_VERTS_PER_CASE = 3200
OVERLAY_FACES_PER_CASE = 850


def _norm_key(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def _clean_stem(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("-all", "").replace("_all", "")
    stem = stem.replace("(1)", "").strip()
    return stem


def _bbox_diag(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _load_included(split_name: str) -> List[Tuple[str, str, str]]:
    split_path = C.PROJECT_ROOT / "training" / "splits" / f"{split_name}.json"
    sp = json.loads(split_path.read_text())
    cases = sp.get("train_cases", []) + sp.get("val_cases", []) + sp.get("test_cases", [])
    out = []
    for label in cases:
        subset, case_name = label.split("/", 1)
        out.append(("AG/" + subset, case_name, label))
    return out


def _stl_index() -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for path in sorted(STL_ROOT.glob("*/*.stl")):
        index.setdefault(_norm_key(_clean_stem(path)), []).append(path)
    return index


def _find_stl(cohort_rel: str, case_name: str, index: Dict[str, List[Path]]) -> Path | None:
    key = _norm_key(case_name)
    subset = cohort_rel.split("/")[-1]
    candidates = []
    for path in index.get(key, []):
        score = 0
        if path.parent.name == subset:
            score += 3
        if path.parent.name == "name_data":
            score += 2
        if _norm_key(_clean_stem(path)) == key:
            score += 1
        candidates.append((score, path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (-x[0], str(x[1])))[0][1]


def _read_stl(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import vtk  # type: ignore
        from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("需要 vtk 才能读取 STL") from exc

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(reader.GetOutput())
    tri.Update()
    poly = tri.GetOutput()
    if poly is None or poly.GetPoints() is None:
        raise RuntimeError(f"STL 无有效点: {path}")

    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    raw_polys = vtk_to_numpy(poly.GetPolys().GetData())
    faces = []
    i = 0
    while i < len(raw_polys):
        n = int(raw_polys[i])
        if n == 3:
            faces.append(raw_polys[i + 1:i + 4])
        i += n + 1
    return pts, np.asarray(faces, dtype=np.int64)


def _sample_idx(n_total: int, n_keep: int, seed: int) -> np.ndarray:
    if n_total <= n_keep:
        return np.arange(n_total)
    return np.random.default_rng(seed).choice(n_total, n_keep, replace=False)


def _case_colors(n_cases: int):
    cmap = plt.get_cmap("tab10" if n_cases <= 10 else "turbo")
    return [cmap(i % 10) if n_cases <= 10 else cmap(i / max(n_cases - 1, 1)) for i in range(n_cases)]


def _set_3d_frame(ax):
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])
    ax.view_init(elev=18, azim=-62)
    ax.set_box_aspect((1, 1, 1))
    ax.plot([-1.05, 1.05], [0, 0], [0, 0], color="black", lw=0.7, alpha=0.45)
    ax.plot([0, 0], [-1.05, 1.05], [0, 0], color="black", lw=0.7, alpha=0.45)
    ax.plot([0, 0], [0, 0], [-1.05, 1.05], color="black", lw=0.9, alpha=0.55)


def _set_2d_frame(ax):
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.42)
    ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.42)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])


def _load_case_overlay(
    cohort_rel: str,
    case_name: str,
    label: str,
    stl_path: Path,
    seed: int,
):
    cfg = C.DEFAULT
    case_dir = C.raw_case_dir(cohort_rel, case_name)
    steps = raw_io.list_timesteps(case_dir)
    ref = steps[0]

    wall_native = raw_io.read_wall_geometry(case_dir, case_name, ref)
    int_native = raw_io.read_interior_geometry(case_dir, case_name, ref)
    centerline = raw_io.read_centerline(case_dir)

    unit_factor, unit_anomaly = _resolve_unit_factor(wall_native, centerline, cfg.unit)
    wall_pts = wall_native * unit_factor
    int_pts = int_native * unit_factor

    reg_cfg = C.registration_for_case(cohort_rel, case_name, cfg.registration)
    transform = compute_transform(wall_pts, int_pts, centerline, reg_cfg)
    wall_aligned = transform.apply_points(wall_pts)
    int_aligned = transform.apply_points(int_pts)
    coord_scale = float(np.abs(np.vstack([wall_aligned, int_aligned])).max()) or 1.0
    wall_norm = wall_aligned / coord_scale

    stl_raw, stl_faces = _read_stl(stl_path)
    native_diag = _bbox_diag(wall_native)
    stl_diag = _bbox_diag(stl_raw)
    native_to_stl = stl_diag / native_diag if native_diag > 1e-12 else 1000.0
    stl_to_pipeline = unit_factor / native_to_stl if native_to_stl > 1e-12 else 1.0
    stl_pts = stl_raw * stl_to_pipeline
    stl_aligned = transform.apply_points(stl_pts)
    stl_norm = stl_aligned / coord_scale

    stl_sample = _sample_idx(len(stl_aligned), min(20000, len(stl_aligned)), seed + 17)
    wall_sample = _sample_idx(len(wall_aligned), min(20000, len(wall_aligned)), seed + 29)
    nn_wall = NearestNeighbors(n_neighbors=1).fit(wall_aligned)
    d_stl_to_wall = nn_wall.kneighbors(stl_aligned[stl_sample], return_distance=True)[0][:, 0]
    nn_stl = NearestNeighbors(n_neighbors=1).fit(stl_aligned)
    d_wall_to_stl = nn_stl.kneighbors(wall_aligned[wall_sample], return_distance=True)[0][:, 0]

    return {
        "label": label,
        "case_name": case_name,
        "cohort": cohort_rel,
        "stl_path": stl_path,
        "wall_norm": wall_norm,
        "stl_norm": stl_norm,
        "stl_faces": stl_faces,
        "unit_factor": float(unit_factor),
        "unit_anomaly": bool(unit_anomaly),
        "native_to_stl": float(native_to_stl),
        "stl_to_pipeline": float(stl_to_pipeline),
        "coord_scale": float(coord_scale),
        "origin_kind": transform.origin_kind,
        "main_axis_mode": transform.main_axis_mode,
        "roll_source": transform.roll_source,
        "stl_to_wall_p50_mm": float(np.median(d_stl_to_wall)),
        "stl_to_wall_p95_mm": float(np.quantile(d_stl_to_wall, 0.95)),
        "wall_to_stl_p50_mm": float(np.median(d_wall_to_stl)),
        "wall_to_stl_p95_mm": float(np.quantile(d_wall_to_stl, 0.95)),
        "n_wall": int(len(wall_norm)),
        "n_stl_vertices": int(len(stl_norm)),
        "n_stl_faces": int(len(stl_faces)),
    }


def _draw_3d_grid(items: List[dict], seed: int):
    cols = 5
    rows = int(np.ceil(len(items) / cols))
    fig = plt.figure(figsize=(cols * 3.4, rows * 3.5))
    for i, item in enumerate(items):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        wall = item["wall_norm"]
        stl = item["stl_norm"]
        faces = item["stl_faces"]

        face_idx = _sample_idx(len(faces), PLOT_FACES, seed + i * 13)
        if len(face_idx) > 0:
            mesh = Poly3DCollection(
                stl[faces[face_idx]],
                facecolor=(0.32, 0.46, 0.63, 0.16),
                edgecolor=(0.20, 0.29, 0.39, 0.10),
                linewidths=0.08,
            )
            ax.add_collection3d(mesh)

        point_idx = _sample_idx(len(wall), PLOT_POINTS, seed + i * 19)
        ax.scatter(
            wall[point_idx, 0], wall[point_idx, 1], wall[point_idx, 2],
            s=1.2, c="#f97316", alpha=0.82, linewidths=0, label="wall points",
        )
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=18, azim=-62)
        ax.set_box_aspect((1, 1, 1))
        title = f"{item['label']}\nSTL→点云 p95={item['stl_to_wall_p95_mm']:.3g} mm"
        ax.set_title(title, fontsize=8, pad=2)
    fig.suptitle("随机 10 例：壁面点云(橙) 与 STL(蓝灰半透明) 在同一解剖归一化框架内叠加",
                 fontsize=13, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT_DIR / "01_random10_stl_point_overlap_3d.png", dpi=145)
    plt.close(fig)


def _draw_projection_grid(items: List[dict], seed: int):
    planes = [(0, 2, "X-Z 主轴视图"), (1, 2, "Y-Z 主轴视图"), (0, 1, "X-Y 横截面视图")]
    fig, axes = plt.subplots(len(items), 3, figsize=(11.8, max(3.0, len(items) * 2.0)))
    axes = np.atleast_2d(axes)
    for r, item in enumerate(items):
        wall = item["wall_norm"]
        stl = item["stl_norm"]
        wall_idx = _sample_idx(len(wall), PLOT_POINTS, seed + r * 31)
        stl_idx = _sample_idx(len(stl), PLOT_STL_VERTS, seed + r * 37)
        for c, (a, b, title) in enumerate(planes):
            ax = axes[r, c]
            ax.scatter(stl[stl_idx, a], stl[stl_idx, b], s=2.5,
                       c="#2563eb", alpha=0.38, linewidths=0)
            ax.scatter(wall[wall_idx, a], wall[wall_idx, b], s=0.55,
                       c="#f97316", alpha=0.70, linewidths=0)
            ax.axhline(0, color="black", lw=0.45, ls="--", alpha=0.35)
            ax.axvline(0, color="black", lw=0.45, ls="--", alpha=0.35)
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{item['label']}\np95={item['stl_to_wall_p95_mm']:.3g}mm",
                              fontsize=7, rotation=0, ha="right", va="center", labelpad=34)
    fig.suptitle("随机 10 例 STL 顶点(蓝灰) 与壁面点云(橙) 三投影叠加：同一坐标轴范围 [-1,1]",
                 fontsize=13, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.35, w_pad=0.15)
    fig.savefig(OUT_DIR / "02_random10_stl_point_overlap_projections.png", dpi=155)
    plt.close(fig)


def _draw_zoomed_projection_grid(items: List[dict], seed: int):
    planes = [(0, 2, "X-Z 局部放大"), (1, 2, "Y-Z 局部放大"), (0, 1, "X-Y 局部放大")]
    fig, axes = plt.subplots(len(items), 3, figsize=(11.8, max(3.0, len(items) * 2.0)))
    axes = np.atleast_2d(axes)
    for r, item in enumerate(items):
        wall = item["wall_norm"]
        stl = item["stl_norm"]
        wall_idx = _sample_idx(len(wall), PLOT_POINTS, seed + r * 41)
        stl_idx = _sample_idx(len(stl), PLOT_STL_VERTS, seed + r * 43)
        for c, (a, b, title) in enumerate(planes):
            ax = axes[r, c]
            all_xy = np.vstack([wall[:, [a, b]], stl[:, [a, b]]])
            lo = np.quantile(all_xy, 0.005, axis=0)
            hi = np.quantile(all_xy, 0.995, axis=0)
            span = np.maximum(hi - lo, 1e-3)
            pad = np.maximum(span * 0.18, 0.025)
            ax.scatter(stl[stl_idx, a], stl[stl_idx, b], s=3.0,
                       c="#2563eb", alpha=0.36, linewidths=0)
            ax.scatter(wall[wall_idx, a], wall[wall_idx, b], s=0.65,
                       c="#f97316", alpha=0.72, linewidths=0)
            ax.axhline(0, color="black", lw=0.45, ls="--", alpha=0.32)
            ax.axvline(0, color="black", lw=0.45, ls="--", alpha=0.32)
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{item['label']}\np95={item['stl_to_wall_p95_mm']:.3g}mm",
                              fontsize=7, rotation=0, ha="right", va="center", labelpad=34)
    fig.suptitle("随机 10 例 STL 顶点(蓝) 与壁面点云(橙) 三投影局部放大：检查是否逐点贴合",
                 fontsize=13, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.35, w_pad=0.15)
    fig.savefig(OUT_DIR / "03_random10_stl_point_overlap_zoomed_projections.png", dpi=155)
    plt.close(fig)


def _draw_same_frame_overlay(items: List[dict], source: str, seed: int):
    is_wall = source == "wall"
    title_source = "壁面点云" if is_wall else "STL 顶点/面片"
    out_name = (
        "04_random10_same_frame_pointcloud_overlay.png"
        if is_wall else
        "05_random10_same_frame_stl_overlay.png"
    )
    colors = _case_colors(len(items))
    planes = [(0, 2, "X-Z 主轴视图"), (1, 2, "Y-Z 主轴视图"), (0, 1, "X-Y 横截面视图")]

    fig = plt.figure(figsize=(15.2, 10.8))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    proj_axes = [fig.add_subplot(2, 2, i) for i in (2, 3, 4)]

    for i, (item, color) in enumerate(zip(items, colors)):
        pts = item["wall_norm"] if is_wall else item["stl_norm"]
        n_keep = OVERLAY_POINTS_PER_CASE if is_wall else OVERLAY_STL_VERTS_PER_CASE
        idx = _sample_idx(len(pts), n_keep, seed + i * 53)

        if is_wall:
            ax3d.scatter(
                pts[idx, 0], pts[idx, 1], pts[idx, 2],
                s=1.0, color=color, alpha=0.58, linewidths=0,
            )
        else:
            faces = item["stl_faces"]
            face_idx = _sample_idx(len(faces), OVERLAY_FACES_PER_CASE, seed + i * 59)
            if len(face_idx) > 0:
                mesh = Poly3DCollection(
                    pts[faces[face_idx]],
                    facecolor=(color[0], color[1], color[2], 0.11),
                    edgecolor=(color[0], color[1], color[2], 0.05),
                    linewidths=0.04,
                )
                ax3d.add_collection3d(mesh)
            ax3d.scatter(
                pts[idx, 0], pts[idx, 1], pts[idx, 2],
                s=0.75, color=color, alpha=0.24, linewidths=0,
            )

        for ax, (a, b, plane_title) in zip(proj_axes, planes):
            ax.scatter(
                pts[idx, a], pts[idx, b],
                s=0.65 if is_wall else 0.8,
                color=color,
                alpha=0.50 if is_wall else 0.36,
                linewidths=0,
            )
            ax.set_title(plane_title, fontsize=11)

    _set_3d_frame(ax3d)
    ax3d.set_title(f"{title_source} 10 例同框 3D 视角", fontsize=11, pad=8)
    for ax in proj_axes:
        _set_2d_frame(ax)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[i],
               markersize=5, label=item["label"])
        for i, item in enumerate(items)
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
        ncol=5, fontsize=8, frameon=False,
    )
    fig.suptitle(
        f"随机 10 例 {title_source} 同一坐标框架叠加：同一原点、同一坐标范围、同一相机视角",
        fontsize=14, y=0.985,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.955])
    fig.savefig(OUT_DIR / out_name, dpi=160)
    plt.close(fig)


def _write_summary(items: List[dict]):
    with open(OUT_DIR / "random10_stl_point_overlap_summary.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "case", "cohort", "stl_path", "n_wall", "n_stl_vertices", "n_stl_faces",
            "unit_factor", "unit_anomaly", "native_to_stl", "stl_to_pipeline",
            "coord_scale", "origin_kind", "main_axis_mode", "roll_source",
            "stl_to_wall_p50_mm", "stl_to_wall_p95_mm",
            "wall_to_stl_p50_mm", "wall_to_stl_p95_mm",
        ])
        for item in items:
            writer.writerow([
                item["label"], item["cohort"], item["stl_path"],
                item["n_wall"], item["n_stl_vertices"], item["n_stl_faces"],
                f"{item['unit_factor']:.8g}", item["unit_anomaly"],
                f"{item['native_to_stl']:.8g}", f"{item['stl_to_pipeline']:.8g}",
                f"{item['coord_scale']:.8g}", item["origin_kind"],
                item["main_axis_mode"], item["roll_source"],
                f"{item['stl_to_wall_p50_mm']:.8g}", f"{item['stl_to_wall_p95_mm']:.8g}",
                f"{item['wall_to_stl_p50_mm']:.8g}", f"{item['wall_to_stl_p95_mm']:.8g}",
            ])


def run(split_name: str = "split_AG_wss_min_v1", n_cases: int = 10, seed: int = 20260707,
        tag: str | None = None):
    global OUT_DIR
    if tag:
        OUT_DIR = C.PROJECT_ROOT / "outputs" / "wss_min" / f"stl_point_overlap_{tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = _stl_index()
    cases = _load_included(split_name)
    available = []
    missing = []
    for cohort_rel, case_name, label in cases:
        stl_path = _find_stl(cohort_rel, case_name, index)
        if stl_path is None:
            missing.append(label)
        else:
            available.append((cohort_rel, case_name, label, stl_path))
    if len(available) < n_cases:
        raise RuntimeError(f"可匹配 STL 的病例不足：available={len(available)} requested={n_cases}")

    rng = random.Random(seed)
    selected = rng.sample(available, n_cases)
    print(f"[stl_overlap] split={split_name} available={len(available)} missing_stl={len(missing)}")
    print("[stl_overlap] selected:")
    for _, _, label, stl_path in selected:
        print(f"  {label} <- {stl_path.relative_to(C.PROJECT_ROOT)}")

    items = []
    for i, (cohort_rel, case_name, label, stl_path) in enumerate(selected):
        item = _load_case_overlay(cohort_rel, case_name, label, stl_path, seed + i * 101)
        items.append(item)
        print(f"  ok {label}: stl->wall p95={item['stl_to_wall_p95_mm']:.4g} mm, "
              f"scale={item['stl_to_pipeline']:.4g}")

    _draw_3d_grid(items, seed)
    _draw_projection_grid(items, seed)
    _draw_zoomed_projection_grid(items, seed)
    _draw_same_frame_overlay(items, "wall", seed)
    _draw_same_frame_overlay(items, "stl", seed)
    _write_summary(items)
    print(f"[stl_overlap] figures + summary -> {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split_AG_wss_min_v1")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--tag", default=None,
                    help="输出目录后缀：outputs/wss_min/stl_point_overlap_<tag>；缺省覆盖 20260707 原目录")
    args = ap.parse_args()
    run(args.split, args.n, args.seed, args.tag)
