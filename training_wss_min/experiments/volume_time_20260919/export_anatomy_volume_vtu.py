#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 体内速度：Fluent anatomy blood 体单元 + 已导出点云身份映射 → 可 Slice 的 VTU。

旧工具 ``tools/cfdpost_cloud_export/build_sliceable_volume.py`` 不能直接用于本包：
它只做 cas 米→毫米缩放、合并全部 cell zone、并且丢掉向量数组。本脚本：

* 只保留 Fluent 区名恰好为 ``blood`` 的解剖腔（不含 blood1–5 延长段）
* 节点做 V5 刚体变换 ``(xyz_mm - centroid) @ R_apply``，与现有 interior VTP 同帧
* 按单元中心最近邻把已验收的 CFD/Pred 速度挂到 **CellData**（不插值到壁面、不 Delaunay）
* 写出一份共享网格 VTU，供 EnSight / ParaView Slice 出填充截面

不改变正式同点 R²。
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path("/public/newhome/cy/Digital_twin/GNN")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from wss_v5 import contract as VC

EXP = REPO / "training_wss_min/experiments/volume_time_20260919"
POST = EXP / "postview_median_phases_20260921"
CASE_TAG = "AAA__unruputer__SHEN_FANG_JIN"
CANONICAL = "AAA/unruputer/SHEN_FANG_JIN"
BUNDLE = (
    REPO / "data_wss_v5/views_v5_1/wss_min_view_v1"
    / "AAA/unruputer/SHEN_FANG_JIN/bundle.npz"
)
ARMS = ("VT0", "VTB4")
PHASES = ("accel", "peak", "decel", "trough")
KEEP_ARRAYS = (
    "speed_cfd", "speed_pred", "err_speed", "abs_err_speed",
    "speed_cfd_selfmax", "speed_pred_selfmax",
    "speed_selfmax_error_pred_minus_cfd", "speed_selfmax_abs_error",
    "velocity_cfd_aligned_m_s", "velocity_pred_aligned_m_s", "velocity_error_aligned_m_s",
)
# Fluent / VTK 读入后若坐标已是毫米，bbox 对角线会远大于 0.5 m 量级。
_METRE_BBOX_DIAG_MAX = 2.0


def log(msg: str) -> None:
    print(msg, flush=True)


def _block_name(mb, index: int) -> str:
    import vtk

    meta = mb.GetMetaData(index)
    if meta is None:
        return ""
    name = meta.Get(vtk.vtkCompositeDataSet.NAME())
    return str(name) if name is not None else ""


def load_anatomy_blood_ug(cas_path: Path, n_blood: int):
    """读 .cas/.cas.gz，只返回 anatomy blood 区（米制，未配准）。

    vtkFLUENTReader 对本队列 .cas 常不填 block 名，因此优先匹配区名 ``blood``，
    否则用拓扑审计给出的唯一 cell 计数认区；blood1–5 延长段一律丢弃。
    """
    import vtk

    tmp_cas = None
    path = cas_path
    if cas_path.suffix == ".gz":
        tmp = tempfile.NamedTemporaryFile(suffix=".cas", delete=False)
        tmp.close()
        tmp_cas = Path(tmp.name)
        log(f"  解压 {cas_path.name} → {tmp_cas}")
        with gzip.open(cas_path, "rb") as fin, open(tmp_cas, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        path = tmp_cas
    try:
        t0 = time.time()
        reader = vtk.vtkFLUENTReader()
        reader.SetFileName(str(path))
        reader.Update()
        log(f"  vtkFLUENTReader {time.time() - t0:.1f}s")
        mb = reader.GetOutput()
        blocks = []
        for i in range(mb.GetNumberOfBlocks()):
            block = mb.GetBlock(i)
            name = _block_name(mb, i)
            n_cells = int(block.GetNumberOfCells()) if block is not None else 0
            n_pts = int(block.GetNumberOfPoints()) if block is not None else 0
            log(f"    block[{i}] name={name!r} cells={n_cells:,} points={n_pts:,}")
            blocks.append((name, block, n_cells))
        named = [(n, b, c) for n, b, c in blocks if n.strip().lower() == "blood" and c > 0]
        counted = [(n, b, c) for n, b, c in blocks if c == int(n_blood)]
        if len(named) == 1:
            name, block, n_cells = named[0]
            how = "zone name"
        elif len(counted) == 1:
            name, block, n_cells = counted[0]
            how = f"unique cell count {n_blood}"
        else:
            raise RuntimeError(
                f"could not isolate anatomy blood zone (name matches={len(named)}, "
                f"count matches={len(counted)}): {[(n, c) for n, _, c in blocks]}"
            )
        ug = vtk.vtkUnstructuredGrid()
        ug.DeepCopy(block)
        # 禁止 vtkRemoveUnusedPoints：对本例 CONVEX_POINT_SET 会改坏点号，单元中心最多偏 6.7 mm。
        log(
            f"  kept anatomy blood via {how} (block name={name!r}): "
            f"{ug.GetNumberOfCells():,} cells, {ug.GetNumberOfPoints():,} nodes "
            f"(unused nodes kept; cleaner corrupts polyhedra)"
        )
        return ug
    finally:
        if tmp_cas is not None:
            tmp_cas.unlink(missing_ok=True)


def cell_centers(ug) -> np.ndarray:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    cc = vtk.vtkCellCenters()
    cc.SetInputData(ug)
    cc.VertexCellsOff()
    cc.Update()
    return vtk_to_numpy(cc.GetOutput().GetPoints().GetData()).astype(np.float64)


def transform_nodes(ug, centroid: np.ndarray, rotation_apply: np.ndarray) -> dict:
    """cas 节点 → V5 对齐毫米。若读入已是毫米则不再 ×1000。"""
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    pts = vtk_to_numpy(ug.GetPoints().GetData()).astype(np.float64)
    bbox = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(bbox))
    if diag <= _METRE_BBOX_DIAG_MAX:
        scale = 1000.0
        unit_in = "m"
    else:
        scale = 1.0
        unit_in = "mm"
    pts_mm = pts * scale
    aligned = (pts_mm - centroid) @ rotation_apply
    ug.GetPoints().SetData(numpy_to_vtk(np.ascontiguousarray(aligned, dtype=np.float64), deep=True))
    return {
        "input_unit": unit_in,
        "scale_to_mm": scale,
        "bbox_diag_in": diag,
        "aligned_bbox_mm": aligned.max(axis=0).tolist() + aligned.min(axis=0).tolist(),
        "n_points": int(len(aligned)),
    }


def v5_rotation(bundle_path: Path, wall_xyz_aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(bundle_path, allow_pickle=True) as data:
        wall_raw = data["wall_coords_raw"].astype(np.float64)
        centroid = data["transform_centroid"].astype(np.float64)
        rotation = data["transform_rotation"].astype(np.float64)
    err_r = float(np.abs((wall_raw - centroid) @ rotation - wall_xyz_aligned).max())
    err_rt = float(np.abs((wall_raw - centroid) @ rotation.T - wall_xyz_aligned).max())
    rotation_apply = rotation if err_r <= err_rt else rotation.T
    if min(err_r, err_rt) > 1e-2:
        raise RuntimeError(f"cannot reproduce wall frame (err {err_r:.3g}/{err_rt:.3g} mm)")
    return centroid, rotation_apply, {
        "rotation_convention": "@R" if err_r <= err_rt else "@R.T",
        "repro_err_mm": min(err_r, err_rt),
    }


def read_vtp_arrays(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    xyz = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    arrays = {}
    pdata = poly.GetPointData()
    for i in range(pdata.GetNumberOfArrays()):
        arr = pdata.GetArray(i)
        if arr is None:
            continue
        name = arr.GetName()
        if name not in KEEP_ARRAYS:
            continue
        arrays[name] = vtk_to_numpy(arr).copy()
    return xyz, arrays


def add_cell_array(ug, name: str, values: np.ndarray) -> None:
    from vtk.util.numpy_support import numpy_to_vtk

    arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
    arr.SetName(name)
    ug.GetCellData().AddArray(arr)


def add_cell_ids(ug, name: str, values: np.ndarray) -> None:
    from vtk.util.numpy_support import numpy_to_vtk

    arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.int64), deep=True)
    arr.SetName(name)
    ug.GetCellData().AddArray(arr)


def write_vtu(ug, path: Path) -> None:
    import vtk

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(ug)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise RuntimeError(f"write failed: {path}")


def try_write_ensight(ug, out_dir: Path, basename: str) -> dict:
    import vtk

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkEnSightWriter()
    writer.SetPath(str(out_dir) + "/")
    writer.SetBaseName(basename)
    writer.SetFileName(str(out_dir / basename))
    writer.SetInputData(ug)
    writer.SetTimeStep(0)
    writer.SetNumberOfBlocks(1)
    writer.Write()
    writer.WriteCaseFile(1)
    case_files = sorted(p.name for p in out_dir.glob(f"{basename}*") if p.suffix in {".case", ".geo", ".scl", ".vel", ".C"})
    return {"wrote": True, "files": case_files}


def cell_type_histogram(ug) -> dict[str, int]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    names = {
        int(vtk.VTK_TETRA): "tetra",
        int(vtk.VTK_HEXAHEDRON): "hex",
        int(vtk.VTK_WEDGE): "wedge",
        int(vtk.VTK_PYRAMID): "pyramid",
        int(vtk.VTK_POLYHEDRON): "polyhedron",
        int(vtk.VTK_CONVEX_POINT_SET): "convex_point_set",
    }
    raw = vtk_to_numpy(ug.GetCellTypesArray()).astype(np.int32)
    counts: dict[str, int] = {}
    for value, n in zip(*np.unique(raw, return_counts=True)):
        counts[names.get(int(value), f"type_{int(value)}")] = int(n)
    return counts


def plot_filled_slice(ug, scalar_cfd: str, scalar_pred: str, origin: np.ndarray, normal: np.ndarray,
                      out_png: Path, title: str, unit: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    plane = vtk.vtkPlane()
    plane.SetOrigin(*[float(x) for x in origin])
    plane.SetNormal(*[float(x) for x in normal])
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(ug)
    cutter.Update()
    cut = cutter.GetOutput()
    n_cells = int(cut.GetNumberOfCells())
    n_pts = int(cut.GetNumberOfPoints())
    if n_cells == 0:
        raise RuntimeError("Slice produced zero cells; mesh/plane mismatch")

    # 把 CellData 插到切面节点，画连续填色（对应 CFD-Post 的 node interpolation 观感）
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(cut)
    c2p.PassCellDataOn()
    c2p.Update()
    surf = c2p.GetOutput()
    xyz = vtk_to_numpy(surf.GetPoints().GetData()).astype(np.float64)
    nrm = np.asarray(normal, dtype=np.float64)
    nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(nrm, tmp)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(nrm, e1)
    uv = np.column_stack([(xyz - origin) @ e1, (xyz - origin) @ e2])

    def polygons_and_values(name: str):
        arr = surf.GetPointData().GetArray(name)
        if arr is None:
            raise RuntimeError(f"slice missing point array {name}")
        val = vtk_to_numpy(arr).astype(np.float64)
        polys, colors = [], []
        for i in range(surf.GetNumberOfCells()):
            cell = surf.GetCell(i)
            ids = [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]
            if len(ids) < 3:
                continue
            polys.append(uv[ids])
            colors.append(float(np.mean(val[ids])))
        return polys, np.asarray(colors, dtype=np.float64)

    poly_cfd, val_cfd = polygons_and_values(scalar_cfd)
    poly_pred, val_pred = polygons_and_values(scalar_pred)
    vmin = float(min(np.nanmin(val_cfd), np.nanmin(val_pred)))
    vmax = float(max(np.nanmax(val_cfd), np.nanmax(val_pred)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=140)
    for ax, polys, vals, lab in (
        (axes[0], poly_cfd, val_cfd, "CFD"),
        (axes[1], poly_pred, val_pred, "Pred"),
    ):
        coll = PolyCollection(polys, array=vals, cmap="turbo", edgecolors="none")
        coll.set_clim(vmin, vmax)
        ax.add_collection(coll)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_title(f"{lab}  {unit}")
        ax.set_xlabel("slice u (mm)")
        ax.set_ylabel("slice v (mm)")
        fig.colorbar(coll, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return {
        "slice_cells": n_cells,
        "slice_points": n_pts,
        "filled_polygons": int(len(poly_cfd)),
        "origin_mm": [float(x) for x in origin],
        "normal": [float(x) for x in normal],
        "png": str(out_png),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-root", type=Path, default=POST)
    parser.add_argument("--skip-ensight", action="store_true")
    parser.add_argument("--skip-slice-png", action="store_true")
    parser.add_argument("--nodedata", action="store_true",
                        help="also write CellDataToPointData copy (large; unused extension nodes get smeared)")
    args = parser.parse_args()
    post = args.post_root.resolve()
    case_root = post / CASE_TAG
    out_dir = case_root / "velocity" / "anatomy_volume"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    audit = json.loads((VC.TOPOLOGY_AUDIT_DIR / f"{CASE_TAG}.json").read_text())
    cas_path = Path(audit["fluent_case"]["path"])
    blood_zone = audit["mesh"]["cell_zones"]["blood"]
    n_blood = int(blood_zone["count"])
    first_id = int(blood_zone["first"])
    last_id = int(blood_zone["last"])
    log(f"case {CANONICAL}")
    log(f"cas {cas_path}  anatomy blood cells (audit)={n_blood:,} fluent ids {first_id}–{last_id}")

    peak_vtp = case_root / "velocity" / "VT0" / "peak" / "interior_pointcloud.vtp"
    xyz_ref, _ = read_vtp_arrays(peak_vtp)
    n_vol = len(xyz_ref)
    log(f"interior VTP points {n_vol:,}")
    if n_vol != n_blood:
        raise RuntimeError(f"interior points {n_vol} != anatomy blood cells {n_blood}")

    with np.load(BUNDLE, allow_pickle=True) as data:
        wall_raw = data["wall_coords_raw"].astype(np.float64)
        centroid = data["transform_centroid"].astype(np.float64)
        rotation = data["transform_rotation"].astype(np.float64)
        coord_scale = float(data["coord_scale"])
    # 用峰值 VTP 的壁面外壳核对旋转约定（wall_geometry_only 与点云同帧）
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    wall_reader = vtk.vtkXMLPolyDataReader()
    wall_reader.SetFileName(str(case_root / "velocity" / "VT0" / "peak" / "wall_geometry_only.vtp"))
    wall_reader.Update()
    wall_xyz = vtk_to_numpy(wall_reader.GetOutput().GetPoints().GetData()).astype(np.float64)
    err_r = float(np.abs((wall_raw - centroid) @ rotation - wall_xyz).max())
    err_rt = float(np.abs((wall_raw - centroid) @ rotation.T - wall_xyz).max())
    rotation_apply = rotation if err_r <= err_rt else rotation.T
    rot_meta = {
        "rotation_convention": "@R" if err_r <= err_rt else "@R.T",
        "repro_err_mm": min(err_r, err_rt),
        "coord_scale": coord_scale,
        "bundle": str(BUNDLE),
    }
    log(f"V5 frame {rot_meta['rotation_convention']} repro_err={rot_meta['repro_err_mm']:.3g} mm")

    h5_path = VC.case_dir(CANONICAL) / "case.h5"
    with h5py.File(h5_path, "r") as h5:
        xyz_mm = h5["volume_static/xyz_mm"][()].astype(np.float64)
        cell_id_cas = h5["volume_static/cell_id_cas"][()].astype(np.int64)
    if len(xyz_mm) != n_vol:
        raise RuntimeError(f"case.h5 volume {len(xyz_mm)} != interior {n_vol}")

    log("loading Fluent anatomy blood mesh…")
    ug = load_anatomy_blood_ug(cas_path, n_blood)
    types_before = cell_type_histogram(ug)
    log(f"  cell types: {types_before}")
    centers_native = cell_centers(ug)
    xform = transform_nodes(ug, centroid, rotation_apply)
    log(f"  transform {xform['input_unit']}×{xform['scale_to_mm']:g} → aligned mm, nodes={xform['n_points']:,}")
    centers_aligned = cell_centers(ug)
    n_mesh = len(centers_aligned)
    log(f"  mesh cells after extract: {n_mesh:,}")

    if n_mesh != n_vol or n_mesh != (last_id - first_id + 1):
        raise RuntimeError(
            f"cell count mismatch mesh={n_mesh} volume={n_vol} fluent {first_id}–{last_id}"
        )
    lookup = np.full(int(cell_id_cas.max()) + 1, -1, dtype=np.int64)
    lookup[cell_id_cas] = np.arange(len(cell_id_cas), dtype=np.int64)
    fluent_ids = np.arange(first_id, last_id + 1, dtype=np.int64)
    row = lookup[fluent_ids]
    if np.any(row < 0):
        raise RuntimeError(f"fluent ids missing from cell_id_cas: {int((row < 0).sum())}")
    dist_native = np.linalg.norm(centers_native * xform["scale_to_mm"] - xyz_mm[row], axis=1)
    dist_aligned = np.linalg.norm(centers_aligned - xyz_ref[row], axis=1)
    unique_rows = np.unique(row)
    match = {
        "n_mesh_cells": int(n_mesh),
        "n_volume_rows": int(n_vol),
        "unique_matched_volume_rows": int(len(unique_rows)),
        "unmatched_volume_rows": int(n_vol - len(unique_rows)),
        "mapping": "fluent_cell_id_cas sequential in vtk blood block",
        "fluent_id_first": first_id,
        "fluent_id_last": last_id,
        "centroid_note": (
            "VTK CellCenters is the vertex mean of CONVEX_POINT_SET/hex; "
            "Fluent/V5 xyz_mm is the volume centroid. Offset is geometry, not a wrong cell."
        ),
        "native_centroid_offset_mm": {
            "max": float(dist_native.max()),
            "p99": float(np.percentile(dist_native, 99)),
            "median": float(np.median(dist_native)),
        },
        "aligned_centroid_offset_mm": {
            "max": float(dist_aligned.max()),
            "p99": float(np.percentile(dist_aligned, 99)),
            "median": float(np.median(dist_aligned)),
        },
        "cell_types": types_before,
    }
    log(
        f"  identity {match['mapping']}: unique {len(unique_rows):,}/{n_vol:,}, "
        f"centroid offset aligned med/p99/max "
        f"{match['aligned_centroid_offset_mm']['median']:.3g}/"
        f"{match['aligned_centroid_offset_mm']['p99']:.3g}/"
        f"{match['aligned_centroid_offset_mm']['max']:.3g} mm"
    )
    if len(unique_rows) != n_vol:
        raise RuntimeError(f"volume rows not fully covered: {len(unique_rows)}/{n_vol}")
    if match["aligned_centroid_offset_mm"]["median"] > 2.0:
        raise RuntimeError("median centroid offset > 2 mm; fluent-id order likely wrong")

    add_cell_ids(ug, "volume_row", row)
    add_cell_ids(ug, "cell_id_cas", cell_id_cas[row])
    add_cell_array(ug, "match_dist_mm", dist_aligned.astype(np.float32))

    attached = []
    for arm in ARMS:
        for phase in PHASES:
            vtp = case_root / "velocity" / arm / phase / "interior_pointcloud.vtp"
            log(f"attach {arm}/{phase} ← {vtp.relative_to(post)}")
            xyz, arrays = read_vtp_arrays(vtp)
            if len(xyz) != n_vol:
                raise RuntimeError(f"{vtp} has {len(xyz)} points, expected {n_vol}")
            shift = float(np.linalg.norm(xyz - xyz_ref, axis=1).max())
            if shift > 1e-6:
                raise RuntimeError(f"{vtp} coordinates drifted {shift} mm from VT0/peak")
            for name, vals in arrays.items():
                if len(vals) != n_vol:
                    raise RuntimeError(f"{vtp} array {name} length {len(vals)}")
                add_cell_array(ug, f"{arm}_{phase}_{name}", vals[row])
                attached.append(f"{arm}_{phase}_{name}")
    ug.GetCellData().SetActiveScalars("VT0_peak_speed_cfd")
    log(f"attached {len(attached)} cell arrays")

    vtu_path = out_dir / "anatomy_volume.vtu"
    log(f"write {vtu_path}")
    write_vtu(ug, vtu_path)

    nodedata_path = None
    if args.nodedata:
        log("CellDataToPointData for display copy…")
        c2p = vtk.vtkCellDataToPointData()
        c2p.SetInputData(ug)
        c2p.PassCellDataOn()
        c2p.Update()
        nodedata_path = out_dir / "anatomy_volume_nodedata.vtu"
        write_vtu(c2p.GetOutput(), nodedata_path)

    ensight_meta = {"wrote": False}
    if not args.skip_ensight:
        try:
            log("write EnSight Gold…")
            ensight_meta = try_write_ensight(ug, out_dir / "ensight", "anatomy_volume")
            log(f"  EnSight files: {ensight_meta.get('files')}")
        except Exception as exc:  # noqa: BLE001
            ensight_meta = {"wrote": False, "error": f"{type(exc).__name__}: {exc}"}
            log(f"  EnSight writer skipped: {ensight_meta['error']}")

    slice_meta = {}
    if not args.skip_slice_png:
        origin = np.median(xyz_ref, axis=0)
        # 瘤腔短轴：用点云 PCA 第一法向，保证切到实心圆截面而不是纵剖空心
        cov = np.cov((xyz_ref - origin).T)
        normal = np.linalg.eigh(cov)[1][:, 0]
        slice_meta = plot_filled_slice(
            ug, "VT0_peak_speed_cfd", "VT0_peak_speed_pred", origin, normal,
            out_dir / "plots" / "fig_VT0_peak_slice_speed.png",
            f"{CANONICAL}  VT0 peak  |u|  Fluent anatomy Slice (not a point cloud)",
            "m/s",
        )
        log(f"  slice png cells={slice_meta['slice_cells']:,} polygons={slice_meta['filled_polygons']:,}")

    # 回读核验
    chk = vtk.vtkXMLUnstructuredGridReader()
    chk.SetFileName(str(vtu_path))
    chk.Update()
    got = chk.GetOutput()
    n_cells = int(got.GetNumberOfCells())
    cd = got.GetCellData()
    missing = [n for n in attached if cd.GetArray(n) is None]
    vec = cd.GetArray("VT0_peak_velocity_cfd_aligned_m_s")
    vec_ok = vec is not None and int(vec.GetNumberOfComponents()) == 3 and int(vec.GetNumberOfTuples()) == n_cells
    speed = vtk_to_numpy(cd.GetArray("VT0_peak_speed_cfd"))
    _, peak_arrays = read_vtp_arrays(peak_vtp)
    speed_src = peak_arrays["speed_cfd"][row]
    speed_err = float(np.max(np.abs(speed.astype(np.float64) - speed_src)))
    verification = {
        "vtu_cells": n_cells,
        "vtu_points": int(got.GetNumberOfPoints()),
        "nonzero_cells": n_cells > 0,
        "missing_arrays": missing,
        "vector_ok": bool(vec_ok),
        "attached_arrays": len(attached),
        "speed_cfd_identity_max_abs": speed_err,
        "nodedata_vtu_exists": bool(nodedata_path and nodedata_path.is_file()),
    }
    if n_cells == 0 or missing or not vec_ok or speed_err > 1e-4:
        raise RuntimeError(f"verification failed: {verification}")
    log(f"verify OK cells={n_cells:,} arrays={len(attached)} |Δspeed|max={speed_err:.3g}")

    report = {
        "case": CANONICAL,
        "cas_path": str(cas_path),
        "zone": "blood",
        "excluded_zones": ["blood1", "blood2", "blood3", "blood4", "blood5"],
        "method": "Fluent anatomy connectivity + V5 rigid transform + cell-centroid identity",
        "not_used": ["Delaunay3D", "wall interpolation", "build_sliceable_volume.py cas-scale-only"],
        "transform": {**rot_meta, **xform},
        "identity": match,
        "verification": verification,
        "ensight": ensight_meta,
        "slice_preview": slice_meta,
        "files": {
            "anatomy_volume_celldata": "anatomy_volume.vtu",
            "anatomy_volume_nodedata": "anatomy_volume_nodedata.vtu" if nodedata_path else None,
        },
        "how_to_open": {
            "ParaView": "Open anatomy_volume.vtu → Slice → Coloring: VT0_peak_speed_cfd (Cell Data). "
                        "Filters → Cell Data to Point Data if you want a smoother CFD-Post-like fill.",
            "EnSight": "File → Open anatomy_volume.vtu (VTK XML Unstructured Grid). "
                       "Clip or Plane, color by VT0_peak_speed_cfd. Do not Glyph the old VTP point cloud. "
                       "If VTK reader is missing, try ensight/anatomy_volume.case.",
        },
        "elapsed_s": round(time.time() - t_all, 1),
    }
    (out_dir / "identity_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"done in {report['elapsed_s']}s → {out_dir}")


if __name__ == "__main__":
    main()
