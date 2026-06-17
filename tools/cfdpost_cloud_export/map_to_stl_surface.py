#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线 B：将 GNN/CSV 点云标量最近邻映射到 STL 壁面顶点，输出面片形式 CSV / VTP。

供 CFD-Post「在三角面上 Contour」或 ParaView 直接打开；比纯散点更接近传统 CFD 云图。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_stl_vertices(stl_path: Path) -> np.ndarray:
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise SystemExit(
            "缺少 vtk，请在 GNN_vmtk 或含 vtk 的环境中运行：conda activate GNN_vmtk"
        ) from exc

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    poly = reader.GetOutput()
    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    if len(pts) == 0:
        raise SystemExit(f"STL 无顶点: {stl_path}")
    return pts


def _write_vtp(
    vertices: np.ndarray,
    triangles: np.ndarray | None,
    scalars: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)

    if triangles is not None and len(triangles) > 0:
        n_tri = len(triangles)
        cells = vtk.vtkCellArray()
        arr = np.hstack([np.full((n_tri, 1), 3, dtype=np.int64), triangles]).ravel()
        cells.SetCells(n_tri, numpy_to_vtkIdTypeArray(arr, deep=True))
        poly.SetPolys(cells)
    else:
        verts = vtk.vtkCellArray()
        for i in range(len(vertices)):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        poly.SetVerts(verts)

    for name, values in scalars.items():
        arr = numpy_to_vtk(values.astype(np.float32), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise RuntimeError(f"写入 VTP 失败: {out_path}")


def _load_stl_mesh(stl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise SystemExit("缺少 vtk") from exc

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    poly = reader.GetOutput()
    verts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    polys = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]
    return verts, polys.astype(np.int64)


def map_csv_to_stl(
    *,
    csv_path: Path,
    stl_path: Path,
    output_dir: Path,
    max_dist: float,
    scalar_columns: list[str],
) -> dict[str, Path]:
    df = pd.read_csv(csv_path)
    for col in ("x", "y", "z"):
        if col not in df.columns:
            raise SystemExit(f"CSV 缺少坐标列 {col}: {csv_path}")

    avail = [c for c in scalar_columns if c in df.columns]
    if not avail:
        raise SystemExit(f"CSV 中无可用标量列（期望之一: {scalar_columns}）")

    from scipy.spatial import cKDTree

    cloud_xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    tree = cKDTree(cloud_xyz)

    stl_verts, stl_tris = _load_stl_mesh(stl_path)
    dists, indices = tree.query(stl_verts, k=1, workers=-1)

    valid = dists <= max_dist
    n_bad = int((~valid).sum())
    if n_bad > 0:
        print(f"[warn] {n_bad}/{len(stl_verts)} 个 STL 顶点在 max_dist={max_dist} 内无匹配点云，标量置 NaN")

    stem = csv_path.stem
    if stem.endswith("__wall"):
        tag = "wall"
    elif stem.endswith("__interior"):
        tag = "interior"
    else:
        tag = "all"
    base = stem.replace("__wall", "").replace("__interior", "").replace("__all", "")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / f"{base}__stl_mapped_{tag}.csv"
    out_vtp = output_dir / f"{base}__stl_mapped_{tag}.vtp"

    mapped: dict[str, np.ndarray] = {}
    for col in avail:
        vals = df[col].to_numpy(dtype=np.float64)[indices]
        vals = vals.astype(np.float64)
        vals[~valid] = np.nan
        mapped[col] = vals

    out_df = pd.DataFrame({"x": stl_verts[:, 0], "y": stl_verts[:, 1], "z": stl_verts[:, 2]})
    out_df["map_dist"] = dists
    out_df["map_valid"] = valid.astype(np.int8)
    for col, vals in mapped.items():
        out_df[col] = vals
    out_df.to_csv(out_csv, index=False, float_format="%.8g")

    _write_vtp(stl_verts, stl_tris, mapped, out_vtp)

    print(f"STL 顶点: {len(stl_verts)} | 三角面: {len(stl_tris)} | 有效映射: {int(valid.sum())}")
    print(f"  CSV: {out_csv}")
    print(f"  VTP: {out_vtp}")
    return {"csv": out_csv, "vtp": out_vtp}


def main() -> None:
    parser = argparse.ArgumentParser(description="点云标量 → STL 面片映射（路线 B）")
    parser.add_argument("--csv", required=True, help="export_for_cfdpost 导出的 __wall.csv 或 __interior.csv")
    parser.add_argument("--stl", required=True, help="病例 STL，如 data_new/AG/slow/GUO_XI_JIANG/GUO_XI_JIANG.stl")
    parser.add_argument(
        "--output-dir",
        default="tools/cfdpost_cloud_export/output/route_interp",
        help="输出目录",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=2.0,
        help="最近邻最大距离（与 CSV 坐标同单位；features 坐标约为 mm 量级时可设 2~5）",
    )
    parser.add_argument(
        "--scalars",
        default="wss_cfd,wss_pred,p_cfd,p_pred,err_wss,err_p",
        help="要映射的列名，逗号分隔",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    stl_path = Path(args.stl).resolve()
    if not stl_path.is_file():
        alt = REPO_ROOT / args.stl
        if alt.is_file():
            stl_path = alt.resolve()
        else:
            raise SystemExit(f"STL 不存在: {args.stl}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    map_csv_to_stl(
        csv_path=csv_path,
        stl_path=stl_path,
        output_dir=output_dir,
        max_dist=args.max_dist,
        scalar_columns=[s.strip() for s in args.scalars.split(",") if s.strip()],
    )


if __name__ == "__main__":
    main()
