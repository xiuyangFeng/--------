#!/usr/bin/env python3
"""Add an exact CFD-wall-mesh VTP to WSS-min postview case packages built from V5 bundles.

The V5 snapshot keeps the Fluent wall triangulation (``topology/wall_triangles``), and the
model predicts on every wall node, so truth/prediction/error can be shown on the real CFD
wall surface without any interpolation.  For each ``<tag>__peak_wss`` case directory this
reads ``_export/<short>__wall.csv`` (same-point scalars in the aligned mm frame), builds the
triangle surface and writes ``<short>__cfd_wall_mesh.vtp`` (all CSV scalars as point data),
then records it in ``manifest_bundle.json`` and the case README.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import vtk
from vtk.util import numpy_support as vnp

from wss_v5 import contract as C


def write_mesh_vtp(xyz: np.ndarray, tris: np.ndarray, scalars: dict[str, np.ndarray], out: Path) -> None:
    pts = vtk.vtkPoints()
    pts.SetData(vnp.numpy_to_vtk(np.ascontiguousarray(xyz, dtype=np.float64), deep=True))
    cells = vtk.vtkCellArray()
    conn = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64), tris.astype(np.int64)]).ravel()
    cells.SetCells(len(tris), vnp.numpy_to_vtkIdTypeArray(np.ascontiguousarray(conn), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    for name, values in scalars.items():
        arr = vnp.numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float64), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)
    poly.GetPointData().SetActiveScalars("wss_cfd" if "wss_cfd" in scalars else next(iter(scalars)))
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out))
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    writer.Write()


def process_case(case_dir: Path, snapshot_root: Path) -> dict:
    manifest_path = case_dir / "manifest_bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical = manifest["case"]
    short = canonical.split("/")[-1]
    csv = case_dir / "_export" / f"{short}__wall.csv"
    df = pd.read_csv(csv)
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    h5_path = C.case_dir(canonical, snapshot_root) / "case.h5"
    with h5py.File(h5_path, "r") as h5:
        valid = h5["wall_static/valid"][()].astype(bool)
        tris = h5["topology/wall_triangles"][()].astype(np.int64)
    if int(valid.sum()) != len(df):
        raise RuntimeError(f"{canonical}: wall CSV rows {len(df)} != valid wall nodes {int(valid.sum())}")
    remap = np.full(len(valid), -1, dtype=np.int64)
    remap[np.flatnonzero(valid)] = np.arange(int(valid.sum()))
    tri_local = remap[tris]
    keep = (tri_local >= 0).all(axis=1)
    tri_local = tri_local[keep]
    scalars = {c: df[c].to_numpy(dtype=np.float64) for c in df.columns if c not in ("x", "y", "z")}
    out = case_dir / f"{short}__cfd_wall_mesh.vtp"
    write_mesh_vtp(xyz, tri_local, scalars, out)
    manifest.setdefault("files", {})["cfd_wall_mesh_vtp"] = out.name
    manifest["cfd_wall_mesh"] = {
        "source": "V5 snapshot topology/wall_triangles (Fluent wall faces triangulated)",
        "n_nodes": int(len(xyz)), "n_triangles": int(len(tri_local)), "triangles_dropped_invalid_node": int((~keep).sum()),
        "interpolation": "none (values at CFD wall nodes)", "frame": "same aligned mm frame as *__wall.csv / *__surface_wall.vtp",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = case_dir / "README_后处理打开说明.md"
    note = (f"\n## 精确 CFD 壁面网格（V5 新增，推荐主图）\n\n"
            f"- `{out.name}`：直接用 Fluent 壁面三角化（{len(tri_local)} 个三角形、{len(xyz)} 个节点）承载同点标量，**无插值**；"
            f"字段与 `_export/{short}__wall.csv` 完全一致（wss_cfd / wss_pred / err_wss / abs_err_wss / *_over_cfd_max / *_selfmax / *_norm / high-risk 掩码）。\n"
            f"- 与 `{short}__surface_wall.vtp`（STL + Gaussian r=3 mm 插值）同一坐标系，可叠加对照；正式数字仍以 CSV 为准。\n")
    if "精确 CFD 壁面网格" not in readme.read_text(encoding="utf-8"):
        readme.write_text(readme.read_text(encoding="utf-8") + note, encoding="utf-8")
    return {"case": canonical, "vtp": str(out), "n_nodes": int(len(xyz)), "n_triangles": int(len(tri_local))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--snapshot-root", default=str(C.SNAPSHOT_ROOT))
    args = ap.parse_args()
    batch = Path(args.batch_dir)
    results = [process_case(d, Path(args.snapshot_root)) for d in sorted(batch.glob("*__peak_wss")) if (d / "manifest_bundle.json").is_file()]
    for r in results:
        print(f"[ok] {r['case']}  nodes={r['n_nodes']} tris={r['n_triangles']} -> {r['vtp']}")
    bm = batch / "batch_manifest.json"
    if bm.is_file():
        data = json.loads(bm.read_text(encoding="utf-8"))
        data["cfd_wall_mesh_vtp"] = {r["case"]: Path(r["vtp"]).name for r in results}
        bm.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
