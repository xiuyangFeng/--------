#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 GNN 点云预测插值到 Fluent 体网格 VTP，输出含 CFD / Pred / Error 的完整体场。

用途：ParaView 对体网格 Slice 查看压力、速度（可选）、壁面 WSS 截面。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PRED_COLS = ["p_pred", "u_pred", "v_pred", "w_pred", "vel_mag_pred", "wss_pred"]
CFD_RENAME = {
    "p": "p_cfd",
    "u": "u_cfd",
    "v": "v_cfd",
    "w": "w_cfd",
    "vel_mag": "vel_mag_cfd",
    "wss": "wss_cfd",
}
ERR_PAIRS = [
    ("err_p", "p_pred", "p_cfd"),
    ("err_u", "u_pred", "u_cfd"),
    ("err_v", "v_pred", "v_cfd"),
    ("err_w", "w_pred", "w_cfd"),
    ("err_vel_mag", "vel_mag_pred", "vel_mag_cfd"),
    ("err_wss", "wss_pred", "wss_cfd"),
]
ABS_ERR = {"err_p": "abs_err_p", "err_wss": "abs_err_wss", "err_vel_mag": "abs_err_vel_mag"}


def _load_volume_vtp(path: Path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    if not reader.CanReadFile(str(path)):
        raise SystemExit(f"无法读取 VTP: {path}")
    reader.Update()
    poly = reader.GetOutput()
    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    arrays: dict[str, np.ndarray] = {}
    pd_obj = poly.GetPointData()
    for i in range(pd_obj.GetNumberOfArrays()):
        arr = pd_obj.GetArray(i)
        name = arr.GetName()
        arrays[name] = vtk_to_numpy(arr).astype(np.float64)
    return poly, pts, arrays


def _write_volume_vtp(
    template_poly,
    points: np.ndarray,
    scalars: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    out = vtk.vtkPolyData()
    out.DeepCopy(template_poly)
    out.GetPoints().SetData(numpy_to_vtk(points.astype(np.float64), deep=True))

    pd_obj = out.GetPointData()
    pd_obj.Initialize()
    for name, values in scalars.items():
        arr = numpy_to_vtk(np.asarray(values, dtype=np.float32), deep=True)
        arr.SetName(name)
        pd_obj.AddArray(arr)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(out)
    if writer.Write() != 1:
        raise RuntimeError(f"写入 VTP 失败: {out_path}")


def _ensure_cfd_names(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(arrays)
    for src, dst in CFD_RENAME.items():
        if dst not in out and src in out:
            out[dst] = out[src].copy()
    if "vel_mag_cfd" not in out and all(k in out for k in ("u_cfd", "v_cfd", "w_cfd")):
        u, v, w = out["u_cfd"], out["v_cfd"], out["w_cfd"]
        out["vel_mag_cfd"] = np.sqrt(u * u + v * v + w * w)
    return out


def _csv_to_source_poly(df: pd.DataFrame, columns: list[str]):
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

    for col in columns:
        if col not in df.columns:
            continue
        arr = numpy_to_vtk(df[col].to_numpy(dtype=np.float32), deep=True)
        arr.SetName(col)
        poly.GetPointData().AddArray(arr)
    return poly


def _vtk_probe(
    target_poly,
    source_poly,
    columns: list[str],
    *,
    radius: float,
    sharpness: float,
) -> dict[str, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    kernel = vtk.vtkGaussianKernel()
    kernel.SetRadius(radius)
    kernel.SetSharpness(sharpness)

    interpolator = vtk.vtkPointInterpolator()
    interpolator.SetInputData(target_poly)
    interpolator.SetSourceData(source_poly)
    interpolator.SetKernel(kernel)
    interpolator.SetNullValue(float("nan"))
    interpolator.Update()
    out = interpolator.GetOutput()

    mapped: dict[str, np.ndarray] = {}
    for col in columns:
        arr = out.GetPointData().GetArray(col)
        if arr is None:
            continue
        mapped[col] = vtk_to_numpy(arr).astype(np.float64)
    return mapped


def _interp_columns(
    target_xyz: np.ndarray,
    target_poly,
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    method: str,
    max_dist: float,
    radius: float,
    sharpness: float,
    k: int,
    power: float,
    fallback: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        n = len(target_xyz)
        return {}, np.full(n, np.nan), np.zeros(n, dtype=np.int8)

    source_poly = _csv_to_source_poly(df, cols)
    mapped = _vtk_probe(target_poly, source_poly, cols, radius=radius, sharpness=sharpness)

    from scipy.spatial import cKDTree

    cloud_xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    tree = cKDTree(cloud_xyz)
    map_dist, nn_idx = tree.query(target_xyz, k=1, workers=-1)
    map_dist = map_dist.astype(np.float64)
    nn_idx = np.asarray(nn_idx, dtype=np.int64)
    valid = (map_dist <= max_dist).astype(np.int8)

    if fallback == "nearest":
        # Gaussian 在 kernel 外返回 NaN；EnSight 等后处理对 NaN 色标会显示 nan nan
        for col in mapped:
            src = df[col].to_numpy(dtype=np.float64)
            out = mapped[col].copy()
            nan_mask = ~np.isfinite(out)
            if nan_mask.any():
                out[nan_mask] = src[nn_idx[nan_mask]]
            mapped[col] = out
    elif fallback == "mask":
        for col in list(mapped.keys()):
            mapped[col] = mapped[col].copy()
            mapped[col][~valid.astype(bool)] = np.nan
    return mapped, map_dist, valid


def export_volume_merged(
    *,
    volume_vtp: Path,
    csv_path: Path,
    output_vtp: Path,
    method: str = "gaussian",
    radius: float = 5.0,
    sharpness: float = 2.0,
    max_dist: float = 12.0,
    k: int = 12,
    power: float = 2.0,
    fallback: str = "nearest",
    include_velocity: bool = True,
    include_wss: bool = True,
    report_json: Path | None = None,
) -> Path:
    template_poly, target_xyz, arrays = _load_volume_vtp(volume_vtp)
    scalars = _ensure_cfd_names(arrays)

    df = pd.read_csv(csv_path)
    for col in ("x", "y", "z"):
        if col not in df.columns:
            raise SystemExit(f"CSV 缺少坐标列 {col}: {csv_path}")

    pred_cols = ["p_pred"]
    if include_velocity:
        pred_cols.extend(["u_pred", "v_pred", "w_pred", "vel_mag_pred"])
    pred_cols = [c for c in pred_cols if c in df.columns]

    print(f"体网格点数: {len(target_xyz):,}")
    print(f"GNN 点云点数: {len(df):,} (壁面 {int((df.get('is_wall', 0) == 1).sum()):,})")
    print(f"插值 pred 列: {pred_cols}")

    pred_mapped, map_dist, map_valid = _interp_columns(
        target_xyz, template_poly, df, pred_cols,
        method=method, max_dist=max_dist, radius=radius, sharpness=sharpness,
        k=k, power=power, fallback=fallback,
    )
    scalars.update(pred_mapped)
    scalars["map_dist_pred"] = map_dist
    scalars["map_valid_pred"] = map_valid.astype(np.float32)

    if include_wss and "wss_pred" in df.columns:
        wall_df = df[df["is_wall"] == 1] if "is_wall" in df.columns else df
        wss_cols = [c for c in ("wss_cfd", "wss_pred") if c in wall_df.columns]
        print(f"WSS 插值（壁面源点 {len(wall_df):,}）: {wss_cols}")
        wss_mapped, wss_dist, wss_valid = _interp_columns(
            target_xyz, template_poly, wall_df, wss_cols,
            method=method, max_dist=max_dist, radius=radius, sharpness=sharpness,
            k=k, power=power, fallback=fallback,
        )
        scalars.update(wss_mapped)
        scalars["map_dist_wss"] = wss_dist
        scalars["map_valid_wss"] = wss_valid.astype(np.float32)

    for err_name, pred_name, cfd_name in ERR_PAIRS:
        if pred_name in scalars and cfd_name in scalars:
            scalars[err_name] = scalars[pred_name] - scalars[cfd_name]
            if err_name in ABS_ERR:
                scalars[ABS_ERR[err_name]] = np.abs(scalars[err_name])

    if "vel_mag_pred" not in scalars and all(k in scalars for k in ("u_pred", "v_pred", "w_pred")):
        u, v, w = scalars["u_pred"], scalars["v_pred"], scalars["w_pred"]
        scalars["vel_mag_pred"] = np.sqrt(u * u + v * v + w * w)
        if "vel_mag_cfd" in scalars:
            scalars["err_vel_mag"] = scalars["vel_mag_pred"] - scalars["vel_mag_cfd"]
            scalars["abs_err_vel_mag"] = np.abs(scalars["err_vel_mag"])

    # 保留原始 CFD 短名（p/u/v/w/vel_mag）便于与旧文件对照
    for short, long in CFD_RENAME.items():
        if long in scalars and short not in scalars:
            scalars[short] = scalars[long]

    output_vtp.parent.mkdir(parents=True, exist_ok=True)
    _write_volume_vtp(template_poly, target_xyz, scalars, output_vtp)

    valid_ratio = float(np.mean(map_valid.astype(bool))) if len(map_valid) else 0.0
    report = {
        "volume_vtp": str(volume_vtp.resolve()),
        "source_csv": str(csv_path.resolve()),
        "output_vtp": str(output_vtp.resolve()),
        "n_volume_points": int(len(target_xyz)),
        "n_source_points": int(len(df)),
        "interpolation": {
            "method": method,
            "radius_mm": radius,
            "sharpness": sharpness,
            "max_dist_mm": max_dist,
            "fallback": fallback,
        },
        "pred_valid_ratio": valid_ratio,
        "scalar_fields": sorted(scalars.keys()),
        "paraview_hint": (
            "Slice → Color: p_cfd|p_pred|err_p (压力); "
            "vel_mag_cfd|vel_mag_pred|err_vel_mag (速度, 可选); "
            "wss_cfd|wss_pred|err_wss (壁面附近, 可选)"
        ),
        "metric_basis": "正式 R² 仍在原始同点点云 CSV；本 VTP 仅供体网格截面可视化",
    }
    if report_json:
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"报告: {report_json}")

    print(f"✅ 已写出: {output_vtp} ({len(target_xyz):,} points, {len(scalars)} fields)")
    print(f"   pred 有效覆盖率: {valid_ratio * 100:.1f}%")
    return output_vtp


def main() -> None:
    parser = argparse.ArgumentParser(description="GNN 点云 → 体网格合并 VTP（CFD+Pred+Error）")
    parser.add_argument("--volume-vtp", required=True, help="CFD 体网格 VTP（如 xxx__volume_cfd-1146.vtp）")
    parser.add_argument("--csv", required=True, help="export_for_cfdpost 的 __all.csv")
    parser.add_argument("--output-vtp", required=True, help="输出合并 VTP 路径")
    parser.add_argument("--report-json", default="", help="可选 mapping 报告 JSON")
    parser.add_argument("--method", default="gaussian", choices=["nearest", "gaussian", "idw"])
    parser.add_argument("--radius", type=float, default=5.0, help="Gaussian 半径 mm（体场默认 5）")
    parser.add_argument("--sharpness", type=float, default=2.0)
    parser.add_argument("--max-dist", type=float, default=12.0, help="最大映射距离 mm（体场默认 12）")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--fallback", default="nearest", choices=["mask", "nearest"])
    parser.add_argument("--no-velocity", action="store_true", help="不插值速度 pred 场")
    parser.add_argument("--no-wss", action="store_true", help="不插值 WSS 场")
    args = parser.parse_args()

    report = Path(args.report_json) if args.report_json else None
    export_volume_merged(
        volume_vtp=Path(args.volume_vtp).resolve(),
        csv_path=Path(args.csv).resolve(),
        output_vtp=Path(args.output_vtp).resolve(),
        method=args.method,
        radius=args.radius,
        sharpness=args.sharpness,
        max_dist=args.max_dist,
        k=args.k,
        power=args.power,
        fallback=args.fallback,
        include_velocity=not args.no_velocity,
        include_wss=not args.no_wss,
        report_json=report,
    )


if __name__ == "__main__":
    main()
