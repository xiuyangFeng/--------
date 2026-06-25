#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路线 B：将 GNN/CSV 点云标量插值/映射到 STL 壁面顶点，输出面片形式 CSV / VTP。

支持 ``nearest`` / ``gaussian`` / ``idw``；默认推荐 ``gaussian``（见 docs/paper_reproduction/05-…）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

Method = Literal["nearest", "gaussian", "idw"]
Fallback = Literal["mask", "nearest"]


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


def _interp_nearest(
    stl_verts: np.ndarray,
    cloud_xyz: np.ndarray,
    cloud_vals: np.ndarray,
    *,
    max_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(cloud_xyz)
    dists, indices = tree.query(stl_verts, k=1, workers=-1)
    out = cloud_vals[indices].astype(np.float64)
    valid = dists <= max_dist
    out[~valid] = np.nan
    return out, dists.astype(np.float64), valid.astype(np.int8)


def _interp_gaussian(
    stl_verts: np.ndarray,
    cloud_xyz: np.ndarray,
    cloud_vals: np.ndarray,
    *,
    radius: float,
    sharpness: float,
    max_dist: float,
    fallback: Fallback,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(cloud_xyz)
    n = len(stl_verts)
    out = np.full(n, np.nan, dtype=np.float64)
    map_dist = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=np.int8)

    neighbor_lists = tree.query_ball_point(stl_verts, r=radius, workers=-1)
    for i, idxs in enumerate(neighbor_lists):
        if len(idxs) == 0:
            if fallback == "nearest":
                d_nn, j = tree.query(stl_verts[i], k=1)
                if d_nn <= max_dist:
                    out[i] = cloud_vals[j]
                    map_dist[i] = float(d_nn)
                    valid[i] = 1
            continue
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        diff = cloud_xyz[idxs_arr] - stl_verts[i]
        d = np.linalg.norm(diff, axis=1)
        w = np.exp(-sharpness * (d / radius) ** 2)
        w_sum = float(w.sum())
        if w_sum <= 0:
            continue
        out[i] = float(np.dot(w, cloud_vals[idxs_arr]) / w_sum)
        map_dist[i] = float(d.min())
        valid[i] = 1

    return out, map_dist, valid


def _interp_idw(
    stl_verts: np.ndarray,
    cloud_xyz: np.ndarray,
    cloud_vals: np.ndarray,
    *,
    k: int,
    power: float,
    max_dist: float,
    fallback: Fallback,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(cloud_xyz)
    n = len(stl_verts)
    out = np.full(n, np.nan, dtype=np.float64)
    map_dist = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=np.int8)
    eps = 1e-8

    dists, indices = tree.query(stl_verts, k=min(k, len(cloud_xyz)), workers=-1)
    if dists.ndim == 1:
        dists = dists[:, None]
        indices = indices[:, None]

    for i in range(n):
        di = dists[i]
        ji = indices[i]
        in_radius = di <= max_dist
        if not np.any(in_radius):
            if fallback == "nearest" and di[0] <= max_dist:
                out[i] = cloud_vals[ji[0]]
                map_dist[i] = float(di[0])
                valid[i] = 1
            continue
        di = di[in_radius]
        ji = ji[in_radius]
        w = 1.0 / (di + eps) ** power
        w_sum = float(w.sum())
        if w_sum <= 0:
            continue
        out[i] = float(np.dot(w, cloud_vals[ji]) / w_sum)
        map_dist[i] = float(di.min())
        valid[i] = 1

    return out, map_dist, valid


def _interp_scalar(
    method: Method,
    stl_verts: np.ndarray,
    cloud_xyz: np.ndarray,
    cloud_vals: np.ndarray,
    *,
    max_dist: float,
    radius: float,
    sharpness: float,
    k: int,
    power: float,
    fallback: Fallback,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if method == "nearest":
        return _interp_nearest(stl_verts, cloud_xyz, cloud_vals, max_dist=max_dist)
    if method == "gaussian":
        return _interp_gaussian(
            stl_verts, cloud_xyz, cloud_vals,
            radius=radius, sharpness=sharpness, max_dist=max_dist, fallback=fallback,
        )
    return _interp_idw(
        stl_verts, cloud_xyz, cloud_vals,
        k=k, power=power, max_dist=max_dist, fallback=fallback,
    )


def _distance_stats(dists: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    d = dists[valid.astype(bool)]
    if len(d) == 0:
        return {"p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
        "p95": float(np.percentile(d, 95)),
        "max": float(np.max(d)),
    }


def map_csv_to_stl(
    *,
    csv_path: Path,
    stl_path: Path,
    output_dir: Path,
    method: Method,
    max_dist: float,
    radius: float,
    sharpness: float,
    k: int,
    power: float,
    fallback: Fallback,
    scalar_columns: list[str],
    report_json: Path | None = None,
) -> dict[str, Path]:
    df = pd.read_csv(csv_path)
    for col in ("x", "y", "z"):
        if col not in df.columns:
            raise SystemExit(f"CSV 缺少坐标列 {col}: {csv_path}")

    avail = [c for c in scalar_columns if c in df.columns]
    if not avail:
        raise SystemExit(f"CSV 中无可用标量列（期望之一: {scalar_columns}）")

    cloud_xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    stl_verts, stl_tris = _load_stl_mesh(stl_path)

    # 误差在 STL 顶点上由插值后的 pred/cfd 重算，避免对 err 二次插值
    derived_err = {
        "err_wss": ("wss_pred", "wss_cfd"),
        "err_p": ("p_pred", "p_cfd"),
        "abs_err_wss": ("wss_pred", "wss_cfd"),
        "abs_err_p": ("p_pred", "p_cfd"),
    }
    base_cols = [c for c in avail if c not in derived_err]

    mapped: dict[str, np.ndarray] = {}
    map_dist: np.ndarray | None = None
    valid: np.ndarray | None = None

    for col in base_cols:
        vals = df[col].to_numpy(dtype=np.float64)
        out, dists, v = _interp_scalar(
            method, stl_verts, cloud_xyz, vals,
            max_dist=max_dist, radius=radius, sharpness=sharpness,
            k=k, power=power, fallback=fallback,
        )
        mapped[col] = out
        if map_dist is None:
            map_dist, valid = dists, v

    if "wss_pred" in mapped and "wss_cfd" in mapped:
        if "err_wss" in avail:
            mapped["err_wss"] = mapped["wss_pred"] - mapped["wss_cfd"]
        if "abs_err_wss" in avail:
            mapped["abs_err_wss"] = np.abs(mapped["wss_pred"] - mapped["wss_cfd"])
    if "p_pred" in mapped and "p_cfd" in mapped:
        if "err_p" in avail:
            mapped["err_p"] = mapped["p_pred"] - mapped["p_cfd"]
        if "abs_err_p" in avail:
            mapped["abs_err_p"] = np.abs(mapped["p_pred"] - mapped["p_cfd"])

    assert map_dist is not None and valid is not None
    n_bad = int((~valid.astype(bool)).sum())
    if n_bad > 0:
        print(f"[warn] {n_bad}/{len(stl_verts)} 个 STL 顶点未成功插值（method={method}）")

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

    out_df = pd.DataFrame({"x": stl_verts[:, 0], "y": stl_verts[:, 1], "z": stl_verts[:, 2]})
    out_df["map_dist"] = map_dist
    out_df["map_valid"] = valid
    for col, vals in mapped.items():
        out_df[col] = vals
    out_df.to_csv(out_csv, index=False, float_format="%.8g")

    _write_vtp(stl_verts, stl_tris, mapped | {"map_dist": map_dist, "map_valid": valid.astype(np.float64)}, out_vtp)

    report = {
        "source_points": str(csv_path),
        "target_geometry": str(stl_path),
        "method": method,
        "params": {
            "max_dist": max_dist,
            "radius": radius,
            "sharpness": sharpness,
            "k": k,
            "power": power,
            "fallback": fallback,
        },
        "coverage": {
            "target_points": int(len(stl_verts)),
            "valid_points": int(valid.sum()),
            "valid_ratio": float(valid.sum() / len(stl_verts)),
        },
        "distance_mm": _distance_stats(map_dist, valid),
        "metric_basis": "metrics are computed on original same-point CSV, not on interpolated STL",
        "figure_only": True,
    }
    report_path = report_json or (output_dir / f"{base}__mapping_report_{tag}.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"方法: {method} | STL 顶点: {len(stl_verts)} | 三角面: {len(stl_tris)} | 有效: {int(valid.sum())} ({report['coverage']['valid_ratio']:.1%})")
    print(f"  CSV: {out_csv}")
    print(f"  VTP: {out_vtp}")
    print(f"  report: {report_path}")
    return {"csv": out_csv, "vtp": out_vtp, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="点云标量 → STL 面片映射（路线 B）")
    parser.add_argument("--csv", required=True, help="export_for_cfdpost 导出的 __wall.csv 或 __interior.csv")
    parser.add_argument("--stl", required=True, help="病例 STL")
    parser.add_argument("--output-dir", default="tools/cfdpost_cloud_export/output/route_interp")
    parser.add_argument("--method", default="gaussian", choices=("nearest", "gaussian", "idw"))
    parser.add_argument("--max-dist", type=float, default=3.0, help="有效邻域/最近邻上限 (mm)")
    parser.add_argument("--radius", type=float, default=3.0, help="Gaussian 核半径 (mm)")
    parser.add_argument("--sharpness", type=float, default=2.0, help="Gaussian 衰减 sharpness")
    parser.add_argument("--k", type=int, default=12, help="IDW 邻居数")
    parser.add_argument("--power", type=float, default=2.0, help="IDW 幂次")
    parser.add_argument("--fallback", default="mask", choices=("mask", "nearest"))
    parser.add_argument(
        "--scalars",
        default="wss_cfd,wss_pred,err_wss,abs_err_wss,p_cfd,p_pred,err_p",
    )
    parser.add_argument("--report-json", default="", help="映射报告 JSON 路径")
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
        method=args.method,
        max_dist=args.max_dist,
        radius=args.radius,
        sharpness=args.sharpness,
        k=args.k,
        power=args.power,
        fallback=args.fallback,
        scalar_columns=[s.strip() for s in args.scalars.split(",") if s.strip()],
        report_json=Path(args.report_json).resolve() if args.report_json else None,
    )


if __name__ == "__main__":
    main()
