#!/usr/bin/env python3
"""Remove CFD inlet/outlet extensions from the selected velocity point clouds.

The original audit VTP combines a 100,000-point sample from the complete CFD
volume with a cropped WSS wall.  The complete CFD volume contains artificial
inlet/outlet extensions.  This tool closes every open boundary loop of the
corresponding anatomical STL and keeps only source volume points enclosed by
that capped surface.  Wall points and WSS arrays are intentionally omitted from
the new files so that wall no-slip zeros do not obscure velocity visualization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT_DIR = (
    REPO_ROOT
    / "outputs/wss_pinn/audits/v2_v3_profile_secant_wss_20260806"
)

SOURCE_FIELDS = (
    "field_valid",
    "velocity_truth_m_s",
    "velocity_pred_m_s",
    "velocity_error_m_s",
    "speed_truth_m_s",
    "speed_pred_m_s",
    "speed_abs_error_m_s",
    "pressure_truth_relative_pa",
    "pressure_pred_relative_pa",
    "pressure_error_signed_pa",
    "pressure_abs_error_pa",
    "source_index",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_vtp(path: Path):
    import vtk

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read a non-empty VTP: {path}")
    return poly


def _read_stl(path: Path):
    import vtk

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read a non-empty STL: {path}")
    return poly


def _boundary_edge_count(poly) -> int:
    import vtk

    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(poly)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.Update()
    return int(edges.GetOutput().GetNumberOfCells())


def _cap_surface(poly, hole_size: float):
    import vtk

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.Update()

    fill = vtk.vtkFillHolesFilter()
    fill.SetInputData(clean.GetOutput())
    fill.SetHoleSize(float(hole_size))
    fill.Update()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(fill.GetOutput())
    triangles.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(triangles.GetOutput())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()
    capped = normals.GetOutput()
    if _boundary_edge_count(capped) != 0:
        raise RuntimeError("Failed to close every anatomical STL boundary loop")
    return capped


def _inside_mask(points: np.ndarray, closed_surface, tolerance: float) -> np.ndarray:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points.astype(np.float32), deep=True))
    cloud = vtk.vtkPolyData()
    cloud.SetPoints(vtk_points)

    selector = vtk.vtkSelectEnclosedPoints()
    selector.SetInputData(cloud)
    selector.SetSurfaceData(closed_surface)
    selector.SetTolerance(float(tolerance))
    selector.CheckSurfaceOn()
    selector.Update()
    selected = selector.GetOutput().GetPointData().GetArray("SelectedPoints")
    if selected is None:
        raise RuntimeError("vtkSelectEnclosedPoints did not produce SelectedPoints")
    return vtk_to_numpy(selected).astype(bool)


def _point_array(poly, name: str) -> np.ndarray:
    from vtk.util.numpy_support import vtk_to_numpy

    array = poly.GetPointData().GetArray(name)
    if array is None:
        raise RuntimeError(f"Source VTP is missing point array: {name}")
    return vtk_to_numpy(array).copy()


def _bounds(points: np.ndarray) -> list[float]:
    return [
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
        float(points[:, 2].min()),
        float(points[:, 2].max()),
    ]


def _write_vtp(
    path: Path,
    points: np.ndarray,
    arrays: dict[str, np.ndarray],
    field_strings: dict[str, str],
    field_numbers: dict[str, float | int],
) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points.astype(np.float32), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)

    vertices = vtk.vtkCellArray()
    legacy_cells = np.column_stack(
        [np.ones(len(points), dtype=np.int64), np.arange(len(points), dtype=np.int64)]
    ).ravel()
    vertices.SetCells(
        len(points), numpy_to_vtkIdTypeArray(legacy_cells, deep=True)
    )
    poly.SetVerts(vertices)

    for name, values in arrays.items():
        vtk_array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
        vtk_array.SetName(name)
        poly.GetPointData().AddArray(vtk_array)
    poly.GetPointData().SetActiveScalars("speed_truth_m_s")

    for name, value in field_strings.items():
        item = vtk.vtkStringArray()
        item.SetName(name)
        item.InsertNextValue(value)
        poly.GetFieldData().AddArray(item)
    for name, value in field_numbers.items():
        item = vtk.vtkDoubleArray()
        item.SetName(name)
        item.InsertNextValue(float(value))
        poly.GetFieldData().AddArray(item)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP: {path}")


def _resolve_stl(case_id: str) -> Path:
    if case_id == "ILO/ZHANG_JIN_CHUN-1/before":
        return REPO_ROOT / "data_new/ILO/ZHANG_JIN_CHUN-1/before/0zhang_jin_chun-sq.stl"
    case_dir = REPO_ROOT / "data_new" / case_id
    case_name = Path(case_id).parts[-1]
    return case_dir / f"{case_name}.stl"


def _verify_output(path: Path, expected_points: int) -> dict:
    from vtk.util.numpy_support import vtk_to_numpy

    poly = _read_vtp(path)
    if poly.GetNumberOfPoints() != expected_points:
        raise RuntimeError(f"Written point count mismatch: {path}")
    if poly.GetNumberOfVerts() != expected_points or poly.GetNumberOfPolys() != 0:
        raise RuntimeError(f"Written velocity output is not a vertex-only point cloud: {path}")
    names = {
        poly.GetPointData().GetArrayName(index)
        for index in range(poly.GetPointData().GetNumberOfArrays())
    }
    if any("wss" in name.lower() for name in names):
        raise RuntimeError(f"Velocity-only output unexpectedly contains a WSS array: {path}")
    nonfinite: dict[str, int] = {}
    for name in names:
        values = vtk_to_numpy(poly.GetPointData().GetArray(name))
        count = int((~np.isfinite(values)).sum())
        if count:
            nonfinite[name] = count
    if nonfinite:
        raise RuntimeError(f"Written VTP has non-finite arrays {nonfinite}: {path}")
    speed = vtk_to_numpy(poly.GetPointData().GetArray("speed_truth_m_s"))
    if float(np.max(speed)) <= 0.0:
        raise RuntimeError(f"Cropped volume has no non-zero CFD speed: {path}")
    return {
        "points": int(poly.GetNumberOfPoints()),
        "vertex_cells": int(poly.GetNumberOfVerts()),
        "point_arrays": sorted(names),
        "all_point_arrays_finite": True,
        "speed_truth_min_m_s": float(np.min(speed)),
        "speed_truth_max_m_s": float(np.max(speed)),
    }


def process_case(
    *,
    source_path: Path,
    stl_path: Path,
    arm: str,
    selection: str,
    case_id: str,
    hole_size: float,
    tolerance: float,
    overwrite: bool,
) -> dict:
    output_path = source_path.with_name(
        f"{source_path.stem}__anatomical_roi_velocity_pointcloud.vtp"
    )
    report_path = source_path.with_name(
        f"{source_path.stem}__anatomical_roi_velocity_report.json"
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_path}")

    source = _read_vtp(source_path)
    source_points = _point_array(source, "point_kind_0_volume_1_wall")
    volume_mask = source_points.reshape(-1) == 0
    from vtk.util.numpy_support import vtk_to_numpy

    all_points = vtk_to_numpy(source.GetPoints().GetData()).astype(np.float64, copy=True)
    volume_indices = np.flatnonzero(volume_mask)
    volume_points = all_points[volume_indices]

    stl = _read_stl(stl_path)
    boundary_before = _boundary_edge_count(stl)
    closed_stl = _cap_surface(stl, hole_size)
    boundary_after = _boundary_edge_count(closed_stl)
    inside = _inside_mask(volume_points, closed_stl, tolerance)
    kept_source_ids = volume_indices[inside]
    kept_points = all_points[kept_source_ids]
    if not len(kept_points):
        raise RuntimeError(f"Anatomical ROI crop removed every volume point: {case_id}")

    arrays: dict[str, np.ndarray] = {}
    for name in SOURCE_FIELDS:
        values = _point_array(source, name)[kept_source_ids]
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values in selected source field {name}: {source_path}")
        arrays[name] = values
    arrays["point_kind_0_volume_1_wall"] = np.zeros(len(kept_points), dtype=np.int8)
    arrays["roi_inside_capped_stl"] = np.ones(len(kept_points), dtype=np.int8)
    arrays["source_vtp_point_id"] = kept_source_ids.astype(np.int64)
    arrays["speed_error_signed_m_s"] = (
        np.asarray(arrays["speed_pred_m_s"], dtype=np.float64)
        - np.asarray(arrays["speed_truth_m_s"], dtype=np.float64)
    ).astype(np.float32)

    field_strings = {
        "source_pointcloud_vtp": str(source_path),
        "target_anatomical_stl": str(stl_path),
        "roi_definition": "source volume points enclosed by target STL after capping all boundary loops",
        "crop_method": "vtkFillHolesFilter + vtkSelectEnclosedPoints",
        "content": "anatomical ROI volume points only; no wall points and no WSS fields",
        "coordinate_frame": "raw CFD frame in millimetres",
    }
    field_numbers = {
        "source_total_points": source.GetNumberOfPoints(),
        "source_volume_points": len(volume_points),
        "kept_anatomical_volume_points": len(kept_points),
        "removed_extension_points": len(volume_points) - len(kept_points),
        "retained_volume_ratio": len(kept_points) / len(volume_points),
        "stl_boundary_edges_before_capping": boundary_before,
        "stl_boundary_edges_after_capping": boundary_after,
        "fill_hole_size_mm": hole_size,
        "enclosed_point_tolerance": tolerance,
    }
    _write_vtp(output_path, kept_points, arrays, field_strings, field_numbers)
    verification = _verify_output(output_path, len(kept_points))

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "arm": arm,
        "selection": selection,
        "case_id": case_id,
        "source_pointcloud_vtp": str(source_path),
        "source_sha256": _sha256(source_path),
        "target_anatomical_stl": str(stl_path),
        "target_stl_sha256": _sha256(stl_path),
        "output_velocity_pointcloud_vtp": str(output_path),
        "output_sha256": _sha256(output_path),
        "method": {
            "name": "capped STL enclosed-point crop",
            "fill_hole_size_mm": hole_size,
            "enclosed_point_tolerance": tolerance,
            "boundary_edges_before_capping": boundary_before,
            "boundary_edges_after_capping": boundary_after,
        },
        "counts": {
            "source_total_points": int(source.GetNumberOfPoints()),
            "source_volume_points": int(len(volume_points)),
            "kept_anatomical_volume_points": int(len(kept_points)),
            "removed_extension_points": int(len(volume_points) - len(kept_points)),
            "retained_volume_ratio": float(len(kept_points) / len(volume_points)),
        },
        "bounds_mm": {
            "source_volume": _bounds(volume_points),
            "kept_anatomical_volume": _bounds(kept_points),
            "target_stl": [float(value) for value in stl.GetBounds()],
        },
        "fields": sorted(arrays),
        "verification": verification,
        "visualization_note": (
            "Use speed_truth_m_s / speed_pred_m_s / speed_abs_error_m_s. "
            "This is still a point cloud; continuous ParaView slices require a VTU volume mesh."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[{arm} {selection}] {case_id}: kept {len(kept_points):,}/{len(volume_points):,} "
        f"volume points ({len(kept_points) / len(volume_points):.1%}); "
        f"removed {len(volume_points) - len(kept_points):,} extension points"
    )
    print(f"  VTP: {output_path}")
    print(f"  report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop the four selected velocity point clouds to their anatomical STL ROI"
    )
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--fill-hole-size-mm", type=float, default=1_000_000.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    audit_dir = args.audit_dir.resolve()
    reports: list[dict] = []
    for arm in ("V2-E8", "V3-E6"):
        manifest = json.loads(
            (audit_dir / "vtp" / arm / "manifest.json").read_text(encoding="utf-8")
        )
        for case in manifest["cases"]:
            reports.append(
                process_case(
                    source_path=Path(case["vtp"]).resolve(),
                    stl_path=_resolve_stl(case["case_id"]).resolve(),
                    arm=arm,
                    selection=case["selection"],
                    case_id=case["case_id"],
                    hole_size=args.fill_hole_size_mm,
                    tolerance=args.tolerance,
                    overwrite=args.overwrite,
                )
            )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "content": "anatomical ROI velocity/pressure volume point clouds without CFD extensions",
        "cases": reports,
    }
    manifest_path = audit_dir / "vtp/anatomical_roi_velocity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Batch manifest: {manifest_path}")


if __name__ == "__main__":
    main()
