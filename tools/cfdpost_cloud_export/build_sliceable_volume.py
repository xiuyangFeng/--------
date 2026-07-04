#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把"体点云预测场"转换成带体单元（四面体）的非结构网格 .vtu，供 ParaView 直接 Slice 出连续填充截面。

背景
----
GNN/CROWN 预测是离散点云；现有的 ``*__volume*.vtp`` 实际上只是 Fluent 网格单元中心的
点集（全 ``verts``、``polys=0``、无四面体）。ParaView/CFD-Post 对纯点云做 Slice 只会切到
平面附近的稀疏散点，得不到连续填充截面。要得到填充截面，标量场必须挂在**带体单元连接**的
网格上（与文件后缀 .cas/.dat/.cdat 无关）。

本脚本提供两种"补上连接"的方式：

1. ``--cas`` 模式（推荐，最严谨）
   从 Fluent ``.cas`` / ``.cas.gz`` 读出体网格（节点 + 四面体/楔形单元，单元拓扑与时间步
   无关），再把来源点云（``--source``，可为 ``*__volume_merged*.vtp`` 或 ``*__all.csv``）里的
   全部标量场用高斯核插值到网格节点，输出 ``.vtu``。

2. ``--delaunay`` 模式（无 .cas 兜底）
   直接对来源点云做 Delaunay3D 四面体化。Delaunay 是凸包，会在血管凹陷/分叉外侧补料，
   可用 ``--alpha`` 删除外接球过大的单元近似裁剪；效果不如 .cas 模式严谨。

用法
----
    conda activate GNN_vmtk   # 需要 vtk

    # cas 模式（推荐）：用 .cas 单元 + 已合并体场 VTP
    python tools/cfdpost_cloud_export/build_sliceable_volume.py \
        --cas data_new/AG/slow/GUO_XI_JIANG/GUO_XI_JIANG.cas.gz \
        --source outputs/field/postview/v3p_i6diag_t016_report/GUO_XI_JIANG__result_features_merged-1146/GUO_XI_JIANG__volume_merged-1146.vtp \
        --output outputs/field/postview/v3p_i6diag_t016_report/GUO_XI_JIANG__result_features_merged-1146/GUO_XI_JIANG__volume_merged-1146.vtu

    # delaunay 兜底（无 .cas）：直接从点云重建
    python tools/cfdpost_cloud_export/build_sliceable_volume.py \
        --delaunay --alpha 4.0 \
        --source .../*__all.csv \
        --output .../GUO_volume_delaunay.vtu

ParaView
--------
    Open .vtu → Filters → Slice（选一个法向/原点）→ Coloring 选 p_pred / p_cfd / err_p /
    vel_mag_pred 等 → 即得连续填充截面。多切面用 Slice 的 "Plane" + 偏移，或 Filters → Clip。
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


# ----------------------------- IO helpers ----------------------------- #
def _read_source_points(source: Path):
    """读来源点云，返回 (xyz[N,3], {field_name: values[N]})。支持 .vtp / .csv。"""
    suffix = source.suffix.lower()
    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(source)
        for c in ("x", "y", "z"):
            if c not in df.columns:
                raise SystemExit(f"CSV 缺少坐标列 {c}: {source}")
        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        fields = {}
        for col in df.columns:
            if col in ("x", "y", "z"):
                continue
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            if np.isfinite(vals).any():
                fields[col] = vals
        return xyz, fields

    if suffix in (".vtp", ".vtk"):
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        reader = vtk.vtkXMLPolyDataReader() if suffix == ".vtp" else vtk.vtkPolyDataReader()
        reader.SetFileName(str(source))
        reader.Update()
        poly = reader.GetOutput()
        xyz = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
        pd_obj = poly.GetPointData()
        fields = {}
        for i in range(pd_obj.GetNumberOfArrays()):
            arr = pd_obj.GetArray(i)
            if arr is None or arr.GetNumberOfComponents() != 1:
                continue
            fields[arr.GetName()] = vtk_to_numpy(arr).astype(np.float64)
        return xyz, fields

    raise SystemExit(f"不支持的来源格式: {source}（支持 .vtp / .csv）")


def _load_cas_unstructured(cas_path: Path, scale: float = 1.0):
    """读 Fluent .cas/.cas.gz，合并所有体块为单个 vtkUnstructuredGrid（节点 + 单元）。

    ``scale``：节点坐标缩放系数。Fluent .cas 多为 SI 单位（米），而本项目 features/点云为
    毫米，需 ``scale=1000`` 把网格从米换算到毫米与点云对齐。
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    tmp_cas = None
    path = cas_path
    if cas_path.suffix == ".gz":
        tmp = tempfile.NamedTemporaryFile(suffix=".cas", delete=False)
        tmp.close()
        tmp_cas = Path(tmp.name)
        with gzip.open(cas_path, "rb") as fin, open(tmp_cas, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        path = tmp_cas

    try:
        reader = vtk.vtkFLUENTReader()
        reader.SetFileName(str(path))
        reader.Update()
        mb = reader.GetOutput()

        append = vtk.vtkAppendFilter()
        append.MergePointsOn()
        n_blocks = 0
        for i in range(mb.GetNumberOfBlocks()):
            block = mb.GetBlock(i)
            if block is None or block.GetNumberOfCells() == 0:
                continue
            append.AddInputData(block)
            n_blocks += 1
        if n_blocks == 0:
            raise SystemExit(f".cas 中未读到任何体单元: {cas_path}")
        append.Update()
        ug = vtk.vtkUnstructuredGrid()
        ug.DeepCopy(append.GetOutput())
        if scale != 1.0:
            pts = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64) * float(scale)
            ug.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(pts), deep=True))
        return ug
    finally:
        if tmp_cas is not None:
            tmp_cas.unlink(missing_ok=True)


# ----------------------------- interpolation ----------------------------- #
def _source_poly(xyz: np.ndarray, fields: dict[str, np.ndarray]):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(xyz, dtype=np.float64), deep=True))
    verts = vtk.vtkCellArray()
    for i in range(len(xyz)):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetVerts(verts)
    for name, vals in fields.items():
        arr = numpy_to_vtk(np.ascontiguousarray(vals, dtype=np.float32), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)
    return poly


def _interpolate_fields_to_nodes(ug, src_xyz, src_fields, *, radius, sharpness, fallback):
    """把来源点云的标量插值到 ug 的节点（点数据）。高斯核 + 最近邻兜底。"""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    target_xyz = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64)
    src_poly = _source_poly(src_xyz, src_fields)

    kernel = vtk.vtkGaussianKernel()
    kernel.SetRadius(radius)
    kernel.SetSharpness(sharpness)

    interp = vtk.vtkPointInterpolator()
    interp.SetInputData(ug)
    interp.SetSourceData(src_poly)
    interp.SetKernel(kernel)
    interp.SetNullValue(float("nan"))
    interp.Update()
    out = interp.GetOutput()

    mapped: dict[str, np.ndarray] = {}
    for name in src_fields:
        arr = out.GetPointData().GetArray(name)
        if arr is None:
            continue
        mapped[name] = vtk_to_numpy(arr).astype(np.float64)

    if fallback == "nearest":
        from scipy.spatial import cKDTree

        tree = cKDTree(src_xyz)
        _, nn = tree.query(target_xyz, k=1, workers=-1)
        nn = np.asarray(nn, dtype=np.int64)
        for name, vals in mapped.items():
            nan = ~np.isfinite(vals)
            if nan.any():
                vals[nan] = src_fields[name][nn[nan]]
            mapped[name] = vals

    for name, vals in mapped.items():
        arr = numpy_to_vtk(np.ascontiguousarray(vals, dtype=np.float32), deep=True)
        arr.SetName(name)
        ug.GetPointData().AddArray(arr)
    return ug, len(target_xyz)


def _delaunay_from_points(src_xyz, src_fields, *, alpha):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(src_xyz, dtype=np.float64), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    for name, vals in src_fields.items():
        arr = numpy_to_vtk(np.ascontiguousarray(vals, dtype=np.float32), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)

    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(poly)
    if alpha and alpha > 0:
        delaunay.SetAlpha(alpha)
        delaunay.AlphaTetsOn()
    delaunay.Update()
    ug = vtk.vtkUnstructuredGrid()
    ug.DeepCopy(delaunay.GetOutput())
    # Delaunay3D 输出会丢点数据，需重新挂回（点序一致）
    src_poly_pd = poly.GetPointData()
    for i in range(src_poly_pd.GetNumberOfArrays()):
        ug.GetPointData().AddArray(src_poly_pd.GetArray(i))
    return ug


def _write_vtu(ug, out_path: Path):
    import vtk

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(ug)
    writer.SetDataModeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"写 .vtu 失败: {out_path}")


# ----------------------------- main ----------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="体点云 → 带四面体单元的 .vtu（供 ParaView Slice 出填充截面）")
    p.add_argument("--source", required=True, help="来源点云（*__volume_merged*.vtp 或 *__all.csv）")
    p.add_argument("--output", required=True, help="输出 .vtu 路径")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cas", help="Fluent .cas/.cas.gz（提供体单元，推荐）")
    mode.add_argument("--delaunay", action="store_true", help="无 .cas，直接 Delaunay3D 重建")
    p.add_argument("--cas-scale", type=float, default=1000.0,
                   help="cas 节点坐标缩放（Fluent 米→点云毫米，默认 1000）")
    p.add_argument("--radius", type=float, default=2.0, help="cas 模式高斯插值半径 mm（默认 2.0）")
    p.add_argument("--sharpness", type=float, default=2.0)
    p.add_argument("--fallback", default="nearest", choices=["nearest", "mask"], help="kernel 外取值方式")
    p.add_argument("--alpha", type=float, default=0.0, help="delaunay 模式裁剪 alpha（0=不裁剪凸包）")
    p.add_argument("--interior-only", action="store_true", help="仅用 is_wall==0 的点（去壁面，看腔内场）")
    args = p.parse_args()

    src_xyz, src_fields = _read_source_points(Path(args.source).resolve())
    if args.interior_only and "is_wall" in src_fields:
        keep = src_fields["is_wall"] < 0.5
        src_xyz = src_xyz[keep]
        src_fields = {k: v[keep] for k, v in src_fields.items()}
    print(f"来源点云: {len(src_xyz):,} 点, 标量场: {sorted(src_fields)}")

    out_path = Path(args.output).resolve()
    if args.cas:
        ug = _load_cas_unstructured(Path(args.cas).resolve(), scale=args.cas_scale)
        b = ug.GetBounds()
        print(f".cas 网格: {ug.GetNumberOfPoints():,} 节点, {ug.GetNumberOfCells():,} 单元 "
              f"(scale={args.cas_scale:g}, bounds x[{b[0]:.1f},{b[1]:.1f}] "
              f"y[{b[2]:.1f},{b[3]:.1f}] z[{b[4]:.1f},{b[5]:.1f}])")
        ug, n = _interpolate_fields_to_nodes(
            ug, src_xyz, src_fields,
            radius=args.radius, sharpness=args.sharpness, fallback=args.fallback,
        )
        print(f"已插值 {len(src_fields)} 个场到 {n:,} 个网格节点")
    else:
        ug = _delaunay_from_points(src_xyz, src_fields, alpha=args.alpha)
        print(f"Delaunay 网格: {ug.GetNumberOfPoints():,} 节点, {ug.GetNumberOfCells():,} 单元"
              + ("（已 alpha 裁剪）" if args.alpha > 0 else "（凸包，未裁剪）"))

    _write_vtu(ug, out_path)
    print(f"✅ 已写出: {out_path}")
    print(f"   单元数: {ug.GetNumberOfCells():,}（>0 即可在 ParaView Slice 出填充截面）")
    print("   ParaView: Open .vtu → Slice → Coloring 选 p_pred / err_p / vel_mag_pred ...")


if __name__ == "__main__":
    main()
