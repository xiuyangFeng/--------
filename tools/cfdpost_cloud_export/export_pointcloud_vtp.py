#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线 C：将 export_for_cfdpost 的 CSV 转为 VTP 点云，供 CFD-Post / ParaView 直接 Contour（不插值到面片）。
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


def _write_point_vtp(df: pd.DataFrame, scalar_cols: list[str], out_path: Path) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(xyz, deep=True))

    verts = vtk.vtkCellArray()
    for i in range(len(xyz)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)

    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetVerts(verts)

    for col in scalar_cols:
        if col not in df.columns:
            continue
        arr = numpy_to_vtk(df[col].to_numpy(dtype=np.float32), deep=True)
        arr.SetName(col)
        poly.GetPointData().AddArray(arr)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise RuntimeError(f"写入失败: {out_path}")


def export_vtp_bundle(csv_path: Path, output_dir: Path) -> dict[str, Path]:
    df = pd.read_csv(csv_path)
    stem = csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    scalar_all = [
        "wss_cfd", "wss_pred", "p_cfd", "p_pred",
        "err_wss", "err_p", "vel_mag_cfd", "vel_mag_pred",
        "u_cfd", "v_cfd", "w_cfd", "u_pred", "v_pred", "w_pred",
    ]
    scalars = [c for c in scalar_all if c in df.columns]

    paths: dict[str, Path] = {}

    out_all = output_dir / f"{stem}.vtp"
    _write_point_vtp(df, scalars, out_all)
    paths["all"] = out_all
    print(f"  VTP [all]: {out_all} ({len(df)} points)")

    if "wss_cfd" in df.columns:
        cfd_cols = [c for c in scalars if c.endswith("_cfd") or c in ("u_cfd", "v_cfd", "w_cfd", "vel_mag_cfd")]
        out_cfd = output_dir / f"{stem}__cfd.vtp"
        _write_point_vtp(df, cfd_cols, out_cfd)
        paths["cfd"] = out_cfd
        print(f"  VTP [cfd]: {out_cfd}")

    if "wss_pred" in df.columns:
        pred_cols = [c for c in scalars if c.endswith("_pred") or c.startswith("err_") or c in ("u_pred", "v_pred", "w_pred", "vel_mag_pred")]
        out_pred = output_dir / f"{stem}__pred.vtp"
        _write_point_vtp(df, pred_cols, out_pred)
        paths["pred"] = out_pred
        print(f"  VTP [pred]: {out_pred}")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="CSV 点云 → VTP（路线 C 直接展示）")
    parser.add_argument("--csv", required=True, help="export_for_cfdpost 导出的 CSV")
    parser.add_argument(
        "--output-dir",
        default="tools/cfdpost_cloud_export/output/route_pointcloud",
        help="VTP 输出目录",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    print(f"输入: {csv_path}")
    export_vtp_bundle(csv_path, output_dir)


if __name__ == "__main__":
    main()
