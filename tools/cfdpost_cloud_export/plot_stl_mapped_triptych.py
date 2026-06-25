#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 map_to_stl_surface 输出的 VTP/CSV 生成 CFD | Pred | Error 三联图（壁面 WSS / 压力）。

默认 ``--render surface``：按 STL 三角面片着色（面片云图）。
``--render scatter``：仅画映射顶点散点（旧行为，视觉上仍像点云）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 改进蓝-白-红（比默认 bwr 更干净）
FIELD_CMAP_NAME = "GNN_BWR"
ERR_CMAP_NAME = "GNN_BWR"


def _register_gnn_colormaps() -> None:
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    if FIELD_CMAP_NAME not in matplotlib.colormaps:
        matplotlib.colormaps.register(
            LinearSegmentedColormap.from_list(
                FIELD_CMAP_NAME,
                [(0.0, "#1f5a9e"), (0.5, "#ffffff"), (1.0, "#c62828")],
                N=256,
            )
        )


def _load_mesh_vtp(vtp_path: Path) -> Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise SystemExit("面片渲染需要 vtk（GNN_vmtk）；请 conda activate GNN_vmtk") from exc

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp_path))
    reader.Update()
    poly = reader.GetOutput()
    verts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    polys_raw = vtk_to_numpy(poly.GetPolys().GetData())
    if len(polys_raw) == 0:
        raise SystemExit(f"VTP 无三角面拓扑，无法面片渲染: {vtp_path}")
    tris = polys_raw.reshape(-1, 4)[:, 1:4].astype(np.int64)

    scalars: dict[str, np.ndarray] = {}
    pd_data = poly.GetPointData()
    for i in range(pd_data.GetNumberOfArrays()):
        arr = pd_data.GetArray(i)
        if arr is not None and arr.GetName():
            scalars[arr.GetName()] = vtk_to_numpy(arr).astype(np.float64)
    return verts, tris, scalars


def _resolve_vtp(csv_path: Path | None, vtp_path: Path | None) -> Path:
    if vtp_path is not None:
        p = vtp_path.resolve()
        if p.is_file():
            return p
        raise SystemExit(f"VTP 不存在: {p}")
    if csv_path is not None:
        p = csv_path.resolve()
        alt = p.with_suffix(".vtp")
        if alt.is_file():
            return alt
        raise SystemExit(f"未找到同名 VTP: {alt}；面片渲染需要 map_to_stl_surface 输出的 *.vtp")
    raise SystemExit("请指定 --vtp 或 --csv（将自动找同名 .vtp）")


def _face_scalar(vertex_scalar: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """三角面标量 = 三顶点均值（与 CFD-Post Contour on triangles 常见口径一致）。"""
    return vertex_scalar[tris].mean(axis=1)


def _plot_surface_panel(
    ax,
    verts: np.ndarray,
    tris: np.ndarray,
    vertex_scalar: np.ndarray,
    *,
    plane: Tuple[int, int],
    cmap: str,
    vmin: float,
    vmax: float,
    axis_names: Tuple[str, str, str],
):
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize
    import matplotlib

    ia, ib = plane
    valid_faces = np.all(np.isfinite(vertex_scalar[tris]), axis=1)
    tris_v = tris[valid_faces]
    if len(tris_v) == 0:
        ax.text(0.5, 0.5, "no valid faces", ha="center", va="center", transform=ax.transAxes)
        return None

    polys = [
        [
            (verts[i0, ia], verts[i0, ib]),
            (verts[i1, ia], verts[i1, ib]),
            (verts[i2, ia], verts[i2, ib]),
        ]
        for i0, i1, i2 in tris_v
    ]
    face_vals = _face_scalar(vertex_scalar, tris_v)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    facecolors = cmap_obj(norm(face_vals))
    pc = PolyCollection(
        polys,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0,
        antialiased=True,
        rasterized=True,
    )
    ax.add_collection(pc)
    xs = verts[:, ia]
    ys = verts[:, ib]
    ax.set_xlim(float(xs.min()), float(xs.max()))
    ax.set_ylim(float(ys.min()), float(ys.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{axis_names[ia]} (mm)")
    ax.set_ylabel(f"{axis_names[ib]} (mm)")
    return matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_obj)


def _plot_scatter_panel(
    ax,
    verts: np.ndarray,
    vertex_scalar: np.ndarray,
    *,
    plane: Tuple[int, int],
    cmap: str,
    vmin: float,
    vmax: float,
    axis_names: Tuple[str, str, str],
    max_points: int,
    seed: int,
):
    ia, ib = plane
    valid = np.isfinite(vertex_scalar)
    pos = verts[valid]
    vals = vertex_scalar[valid]
    rng = np.random.default_rng(seed)
    if len(pos) > max_points:
        idx = rng.choice(len(pos), size=max_points, replace=False)
        pos, vals = pos[idx], vals[idx]
    sc = ax.scatter(
        pos[:, ia], pos[:, ib], c=vals, cmap=cmap,
        s=2.5, alpha=0.85, vmin=vmin, vmax=vmax, rasterized=True,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{axis_names[ia]} (mm)")
    ax.set_ylabel(f"{axis_names[ib]} (mm)")
    return sc


def plot_triptych(
    *,
    vtp_path: Path,
    cfd_key: str,
    pred_key: str,
    err_key: str,
    output_path: Path,
    title: str,
    unit: str,
    plane: str = "xy",
    render: str = "surface",
    field_cmap: str = FIELD_CMAP_NAME,
    err_cmap: str = ERR_CMAP_NAME,
    max_points: int = 80_000,
    seed: int = 42,
) -> dict:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("需要 matplotlib（GNN 环境）") from exc

    _register_gnn_colormaps()

    verts, tris, scalars = _load_mesh_vtp(vtp_path)
    for key in (cfd_key, pred_key, err_key):
        if key not in scalars:
            raise SystemExit(f"VTP 缺少标量 {key!r}，现有: {sorted(scalars.keys())}")

    cfd = scalars[cfd_key]
    pred = scalars[pred_key]
    err = scalars[err_key]
    if "map_valid" in scalars:
        mask = scalars["map_valid"].astype(bool) | (scalars["map_valid"] == 1)
        cfd = np.where(mask, cfd, np.nan)
        pred = np.where(mask, pred, np.nan)
        err = np.where(mask, err, np.nan)

    valid = np.isfinite(cfd) & np.isfinite(pred) & np.isfinite(err)
    if valid.sum() == 0:
        raise SystemExit(f"无有效映射顶点: {vtp_path}")

    plane_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ia, ib = plane_map[plane]
    axis_names = ("x", "y", "z")

    vmin = float(np.nanmin(np.concatenate([cfd[valid], pred[valid]])))
    vmax = float(np.nanmax(np.concatenate([cfd[valid], pred[valid]])))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0
    err_abs = np.abs(err)
    err_vmax = float(np.percentile(err_abs[valid], 99.0))
    if err_vmax <= 0:
        err_vmax = float(np.max(err_abs[valid])) or 1.0

    labels = (f"CFD ({unit})", f"GNN Pred ({unit})", f"|Error| ({unit})")
    arrays = (cfd, pred, err_abs)
    cmaps = (field_cmap, field_cmap, err_cmap)
    vmins = (vmin, vmin, 0.0)
    vmaxs = (vmax, vmax, err_vmax)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    mappable = None
    for ax, arr, lab, cmap, lo, hi in zip(axes, arrays, labels, cmaps, vmins, vmaxs):
        if render == "surface":
            mappable = _plot_surface_panel(
                ax, verts, tris, arr,
                plane=(ia, ib), cmap=cmap, vmin=lo, vmax=hi, axis_names=axis_names,
            )
        else:
            mappable = _plot_scatter_panel(
                ax, verts, arr,
                plane=(ia, ib), cmap=cmap, vmin=lo, vmax=hi,
                axis_names=axis_names, max_points=max_points, seed=seed,
            )
        ax.set_title(lab, fontsize=11, fontweight="bold")
        if mappable is not None:
            fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)

    render_note = "STL tri faces" if render == "surface" else "STL vertices scatter"
    fig.suptitle(f"{title} · {render_note}", fontsize=13, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    n_faces = int(len(tris))
    n_valid_faces = int(np.all(np.isfinite(arrays[0][tris]), axis=1).sum())

    return {
        "vtp": str(vtp_path),
        "output_png": str(output_path),
        "render_mode": render,
        "cfd_key": cfd_key,
        "pred_key": pred_key,
        "err_key": err_key,
        "plane": plane,
        "value_range": {"vmin": vmin, "vmax": vmax},
        "err_p99": err_vmax,
        "mesh": {"vertices": int(len(verts)), "triangles": n_faces, "valid_faces": n_valid_faces},
        "valid_vertices": int(valid.sum()),
        "metric_basis": "err recomputed on STL after field interpolation; surface color = mean of 3 vertex values per triangle",
        "colormaps": {"field": field_cmap, "error": err_cmap},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="STL 映射 VTP → CFD/Pred/Error 三联图")
    parser.add_argument("--csv", default="", help="可选：*.csv（用于定位同名 .vtp）")
    parser.add_argument("--vtp", default="", help="map_to_stl_surface 输出的 *.vtp（含三角面）")
    parser.add_argument("--output", required=True, help="输出 PNG 路径")
    parser.add_argument("--title", default="Surface field comparison")
    parser.add_argument("--variable", choices=("wss", "p", "vel_mag"), default="wss")
    parser.add_argument("--plane", default="xy", choices=("xy", "xz", "yz"))
    parser.add_argument("--render", default="surface", choices=("surface", "scatter"))
    parser.add_argument("--field-cmap", default=FIELD_CMAP_NAME, help="CFD/Pred 色标（默认 GNN 蓝-白-红）")
    parser.add_argument("--err-cmap", default=ERR_CMAP_NAME, help="误差色标")
    parser.add_argument("--max-points", type=int, default=80_000, help="render=scatter 时子采样上限")
    parser.add_argument("--report-json", default="")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    vtp_path = Path(args.vtp) if args.vtp else None
    resolved_vtp = _resolve_vtp(csv_path, vtp_path)

    if args.variable == "wss":
        keys = ("wss_cfd", "wss_pred", "err_wss")
        unit = "Pa"
    elif args.variable == "vel_mag":
        keys = ("vel_mag_cfd", "vel_mag_pred", "err_vel_mag")
        unit = "m/s"
    else:
        keys = ("p_cfd", "p_pred", "err_p")
        unit = "Pa"

    report = plot_triptych(
        vtp_path=resolved_vtp,
        cfd_key=keys[0],
        pred_key=keys[1],
        err_key=keys[2],
        output_path=Path(args.output).resolve(),
        title=args.title,
        unit=unit,
        plane=args.plane,
        render=args.render,
        field_cmap=args.field_cmap,
        err_cmap=args.err_cmap,
        max_points=args.max_points,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
