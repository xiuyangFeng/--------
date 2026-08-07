#!/usr/bin/env python3
"""Gaussian-map the selected WSS-PINN wall point clouds onto their STL meshes.

The source VTP files contain a deterministic volume sample plus every wall point,
but only vertex cells.  This tool keeps the source files unchanged, selects only
``point_kind_0_volume_1_wall == 1``, and maps the wall fields to the vertices of
the real triangular STL surface using the repository postview convention:

    Gaussian radius = 3 mm, sharpness = 2, no nearest-neighbour fallback.

Official numerical metrics must continue to come from the original same-point
arrays.  The generated surface files are visualization/postview products only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIT_DIR = (
    REPO_ROOT
    / "outputs/wss_pinn/audits/v2_v3_profile_secant_wss_20260806"
)

SOURCE_ARRAYS = (
    "velocity_truth_m_s",
    "velocity_pred_m_s",
    "speed_truth_m_s",
    "speed_pred_m_s",
    "pressure_truth_relative_pa",
    "pressure_pred_relative_pa",
    "wss_truth_vector_pa",
    "wss_pred_vector_pa",
    "wss_truth_pa",
    "wss_pred_pa",
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


def _read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read a non-empty STL: {path}")

    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64, copy=True)
    cells = poly.GetPolys()
    connectivity = vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64, copy=True)
    offsets = vtk_to_numpy(cells.GetOffsetsArray()).astype(np.int64, copy=True)
    cell_sizes = np.diff(offsets)
    if len(cell_sizes) == 0 or not np.all(cell_sizes == 3):
        raise RuntimeError(f"Target STL is not an all-triangle surface: {path}")
    triangles = connectivity.reshape(-1, 3)
    return points, triangles


def _point_array(poly, name: str) -> np.ndarray:
    from vtk.util.numpy_support import vtk_to_numpy

    array = poly.GetPointData().GetArray(name)
    if array is None:
        raise RuntimeError(f"Source VTP is missing point array: {name}")
    return vtk_to_numpy(array).copy()


def _source_wall_data(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    from vtk.util.numpy_support import vtk_to_numpy

    poly = _read_vtp(path)
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64, copy=True)
    point_kind = _point_array(poly, "point_kind_0_volume_1_wall").reshape(-1)
    wall_mask = point_kind > 0.5
    if not np.any(wall_mask):
        raise RuntimeError(f"Source VTP has no wall points: {path}")

    field_valid = _point_array(poly, "field_valid").reshape(-1)
    wss_valid = _point_array(poly, "wss_valid").reshape(-1)
    if not np.all(field_valid[wall_mask] > 0.5):
        raise RuntimeError(f"Some selected wall points have field_valid != 1: {path}")
    if not np.all(wss_valid[wall_mask] > 0.5):
        raise RuntimeError(f"Some selected wall points have wss_valid != 1: {path}")

    arrays: dict[str, np.ndarray] = {}
    for name in SOURCE_ARRAYS:
        values = _point_array(poly, name)[wall_mask].astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite wall values in {name}: {path}")
        arrays[name] = values

    wall_points = points[wall_mask]
    metadata = {
        "source_total_points": int(len(points)),
        "source_wall_points": int(len(wall_points)),
        "source_wall_bounds_mm": _bounds(wall_points),
    }
    return wall_points, arrays, metadata


def _stack_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, slice]]:
    chunks: list[np.ndarray] = []
    slices: dict[str, slice] = {}
    offset = 0
    for name in SOURCE_ARRAYS:
        values = arrays[name]
        chunk = values[:, None] if values.ndim == 1 else values
        width = chunk.shape[1]
        chunks.append(chunk)
        slices[name] = slice(offset, offset + width)
        offset += width
    return np.concatenate(chunks, axis=1), slices


def _gaussian_map(
    target_points: np.ndarray,
    source_points: np.ndarray,
    source_values: np.ndarray,
    *,
    radius: float,
    sharpness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    tree = cKDTree(source_points)
    neighbor_lists = tree.query_ball_point(target_points, r=radius, workers=-1)
    mapped = np.full((len(target_points), source_values.shape[1]), np.nan, dtype=np.float64)
    map_dist = np.full(len(target_points), np.nan, dtype=np.float64)
    map_valid = np.zeros(len(target_points), dtype=np.int8)
    neighbor_count = np.zeros(len(target_points), dtype=np.int32)

    for index, neighbors in enumerate(neighbor_lists):
        if not neighbors:
            continue
        neighbor_index = np.asarray(neighbors, dtype=np.int64)
        delta = source_points[neighbor_index] - target_points[index]
        distances = np.linalg.norm(delta, axis=1)
        weights = np.exp(-sharpness * (distances / radius) ** 2)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            continue
        mapped[index] = weights @ source_values[neighbor_index] / weight_sum
        map_dist[index] = float(distances.min())
        map_valid[index] = 1
        neighbor_count[index] = len(neighbor_index)

    return mapped, map_dist, map_valid, neighbor_count


def _unstack_arrays(
    mapped: np.ndarray,
    slices: dict[str, slice],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name in SOURCE_ARRAYS:
        values = mapped[:, slices[name]]
        output[name] = values[:, 0] if values.shape[1] == 1 else values
    return output


def _add_derived_fields(arrays: dict[str, np.ndarray]) -> None:
    velocity_error = arrays["velocity_pred_m_s"] - arrays["velocity_truth_m_s"]
    speed_error = arrays["speed_pred_m_s"] - arrays["speed_truth_m_s"]
    pressure_error = (
        arrays["pressure_pred_relative_pa"] - arrays["pressure_truth_relative_pa"]
    )
    wss_error = arrays["wss_pred_pa"] - arrays["wss_truth_pa"]

    arrays["velocity_error_m_s"] = velocity_error
    arrays["speed_error_signed_m_s"] = speed_error
    arrays["speed_abs_error_m_s"] = np.abs(speed_error)
    arrays["pressure_error_signed_pa"] = pressure_error
    arrays["pressure_abs_error_pa"] = np.abs(pressure_error)
    arrays["wss_error_signed_pa"] = wss_error
    arrays["wss_abs_error_pa"] = np.abs(wss_error)

    # Standard postview aliases retain compatibility with the existing plotting tools.
    arrays["vel_mag_cfd"] = arrays["speed_truth_m_s"]
    arrays["vel_mag_pred"] = arrays["speed_pred_m_s"]
    arrays["err_vel_mag"] = speed_error
    arrays["abs_err_vel_mag"] = np.abs(speed_error)
    arrays["p_cfd"] = arrays["pressure_truth_relative_pa"]
    arrays["p_pred"] = arrays["pressure_pred_relative_pa"]
    arrays["err_p"] = pressure_error
    arrays["abs_err_p"] = np.abs(pressure_error)
    arrays["wss_cfd"] = arrays["wss_truth_pa"]
    arrays["wss_pred"] = arrays["wss_pred_pa"]
    arrays["err_wss"] = wss_error
    arrays["abs_err_wss"] = np.abs(wss_error)

    wss_pred_max = float(np.max(arrays["wss_pred_pa"]))
    wss_truth_max = float(np.max(arrays["wss_truth_pa"]))
    if not np.isfinite(wss_pred_max) or wss_pred_max <= 0.0:
        raise RuntimeError(f"Invalid interpolated prediction WSS maximum: {wss_pred_max}")
    if not np.isfinite(wss_truth_max) or wss_truth_max <= 0.0:
        raise RuntimeError(f"Invalid interpolated truth WSS maximum: {wss_truth_max}")
    arrays["wss_pred_over_wss_pred_max"] = arrays["wss_pred_pa"] / wss_pred_max
    arrays["wss_truth_over_wss_truth_max"] = arrays["wss_truth_pa"] / wss_truth_max
    arrays["wss_pred_over_wss_truth_max"] = arrays["wss_pred_pa"] / wss_truth_max


def _write_vtp(
    output_path: Path,
    points: np.ndarray,
    triangles: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    source_path: Path,
    stl_path: Path,
    radius: float,
    sharpness: float,
) -> None:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points.astype(np.float32), deep=True))

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    cell_array = vtk.vtkCellArray()
    legacy_cells = np.hstack(
        [np.full((len(triangles), 1), 3, dtype=np.int64), triangles.astype(np.int64)]
    ).ravel()
    cell_array.SetCells(
        len(triangles), numpy_to_vtkIdTypeArray(legacy_cells, deep=True)
    )
    poly.SetPolys(cell_array)

    for name, values in arrays.items():
        vtk_array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
        vtk_array.SetName(name)
        poly.GetPointData().AddArray(vtk_array)
    poly.GetPointData().SetActiveScalars("wss_truth_pa")

    field_strings = {
        "mapping_method": "gaussian",
        "source_pointcloud_vtp": str(source_path),
        "target_wall_stl": str(stl_path),
        "metric_basis": "visualization only; official metrics remain on original points",
    }
    for name, value in field_strings.items():
        item = vtk.vtkStringArray()
        item.SetName(name)
        item.InsertNextValue(value)
        poly.GetFieldData().AddArray(item)
    for name, value in {
        "mapping_radius_mm": radius,
        "mapping_sharpness": sharpness,
        "wss_pred_surface_max_pa": float(np.max(arrays["wss_pred_pa"])),
        "wss_truth_surface_max_pa": float(np.max(arrays["wss_truth_pa"])),
    }.items():
        item = vtk.vtkDoubleArray()
        item.SetName(name)
        item.InsertNextValue(float(value))
        poly.GetFieldData().AddArray(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(poly)
    writer.SetDataModeToAppended()
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP: {output_path}")


def _bounds(points: np.ndarray) -> list[float]:
    return [
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
        float(points[:, 2].min()),
        float(points[:, 2].max()),
    ]


def _valid_stats(values: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    selected = values[valid.astype(bool)]
    return {
        "p50": float(np.percentile(selected, 50)),
        "p90": float(np.percentile(selected, 90)),
        "p95": float(np.percentile(selected, 95)),
        "p99": float(np.percentile(selected, 99)),
        "max": float(selected.max()),
    }


def _resolve_stl(case_id: str) -> Path:
    if case_id == "ILO/ZHANG_JIN_CHUN-1/before":
        # This is the CFD wall represented by the source wall nodes.  ZJC_SQ_FINAL.stl
        # includes additional geometry and only reaches 68.6% coverage at r=3 mm.
        return REPO_ROOT / "data_new/ILO/ZHANG_JIN_CHUN-1/before/0zhang_jin_chun-sq.stl"
    case_dir = REPO_ROOT / "data_new" / case_id
    case_name = Path(case_id).parts[-1]
    return case_dir / f"{case_name}.stl"


def _verify_written_vtp(path: Path, expected_points: int, expected_cells: int) -> dict:
    from vtk.util.numpy_support import vtk_to_numpy

    poly = _read_vtp(path)
    if poly.GetNumberOfPoints() != expected_points:
        raise RuntimeError(f"Written VTP point count mismatch: {path}")
    if poly.GetNumberOfPolys() != expected_cells:
        raise RuntimeError(f"Written VTP triangle count mismatch: {path}")
    if poly.GetNumberOfVerts() != 0:
        raise RuntimeError(f"Written surface unexpectedly contains vertex cells: {path}")

    nonfinite: dict[str, int] = {}
    point_data = poly.GetPointData()
    for index in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(index)
        values = vtk_to_numpy(array)
        count = int((~np.isfinite(values)).sum())
        if count:
            nonfinite[array.GetName()] = count
    if nonfinite:
        raise RuntimeError(f"Written VTP has non-finite arrays {nonfinite}: {path}")
    return {
        "points": int(poly.GetNumberOfPoints()),
        "triangles": int(poly.GetNumberOfPolys()),
        "point_arrays": int(poly.GetPointData().GetNumberOfArrays()),
        "all_point_arrays_finite": True,
    }


def process_case(
    *,
    source_path: Path,
    stl_path: Path,
    case_id: str,
    selection: str,
    arm: str,
    radius: float,
    sharpness: float,
    overwrite: bool,
) -> dict:
    output_path = source_path.with_name(f"{source_path.stem}__surface_wall.vtp")
    report_path = source_path.with_name(f"{source_path.stem}__mapping_report_wall.json")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output_path}")
    backup_path: Path | None = None
    if output_path.exists() and overwrite:
        backup_dir = output_path.parent / "backup_before_wss_normalized_fields"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / output_path.name
        if not backup_path.exists():
            shutil.copy2(output_path, backup_path)
    if not stl_path.is_file():
        raise FileNotFoundError(f"Target STL does not exist: {stl_path}")

    source_points, source_arrays, source_metadata = _source_wall_data(source_path)
    target_points, triangles = _read_stl(stl_path)
    stacked_values, slices = _stack_arrays(source_arrays)
    mapped, map_dist, map_valid, neighbor_count = _gaussian_map(
        target_points,
        source_points,
        stacked_values,
        radius=radius,
        sharpness=sharpness,
    )
    valid_ratio = float(map_valid.mean())
    if valid_ratio < 0.95:
        raise RuntimeError(
            f"Gaussian coverage below required 95%: {valid_ratio:.2%} for {case_id}"
        )
    if not np.all(map_valid == 1):
        raise RuntimeError(
            f"This audited batch expects finite full coverage, got {valid_ratio:.2%}: {case_id}"
        )

    output_arrays = _unstack_arrays(mapped, slices)
    _add_derived_fields(output_arrays)
    output_arrays["point_kind_0_volume_1_wall"] = np.ones(
        len(target_points), dtype=np.int8
    )
    output_arrays["field_valid"] = map_valid.copy()
    output_arrays["wss_valid"] = map_valid.copy()
    output_arrays["map_dist"] = map_dist.astype(np.float32)
    output_arrays["map_valid"] = map_valid.copy()
    output_arrays["map_neighbor_count"] = neighbor_count.copy()
    output_arrays = {
        name: np.asarray(values, dtype=np.float32)
        if np.asarray(values).dtype.kind == "f"
        else np.asarray(values)
        for name, values in output_arrays.items()
    }

    _write_vtp(
        output_path,
        target_points,
        triangles,
        output_arrays,
        source_path=source_path,
        stl_path=stl_path,
        radius=radius,
        sharpness=sharpness,
    )
    verification = _verify_written_vtp(output_path, len(target_points), len(triangles))

    valid_neighbors = neighbor_count[map_valid.astype(bool)]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "arm": arm,
        "selection": selection,
        "case_id": case_id,
        "source_points": str(source_path),
        "target_geometry": str(stl_path),
        "output_surface_vtp": str(output_path),
        "method": "gaussian",
        "params": {
            "radius_mm": radius,
            "sharpness": sharpness,
            "max_dist_mm": radius,
            "fallback": "mask",
            "source_subset": "point_kind_0_volume_1_wall == 1",
        },
        **source_metadata,
        "target_points": int(len(target_points)),
        "target_triangles": int(len(triangles)),
        "target_bounds_mm": _bounds(target_points),
        "coverage": {
            "valid_points": int(map_valid.sum()),
            "invalid_points": int((map_valid == 0).sum()),
            "valid_ratio": valid_ratio,
        },
        "distance_mm": _valid_stats(map_dist, map_valid),
        "neighbor_count": {
            "min": int(valid_neighbors.min()),
            "p50": float(np.percentile(valid_neighbors, 50)),
            "p95": float(np.percentile(valid_neighbors, 95)),
            "max": int(valid_neighbors.max()),
        },
        "mapped_source_arrays": list(SOURCE_ARRAYS),
        "derived_fields_recomputed_after_mapping": [
            "velocity_error_m_s",
            "speed_error_signed_m_s",
            "speed_abs_error_m_s",
            "pressure_error_signed_pa",
            "pressure_abs_error_pa",
            "wss_error_signed_pa",
            "wss_abs_error_pa",
        ],
        "postview_aliases": [
            "vel_mag_cfd",
            "vel_mag_pred",
            "err_vel_mag",
            "abs_err_vel_mag",
            "p_cfd",
            "p_pred",
            "err_p",
            "abs_err_p",
            "wss_cfd",
            "wss_pred",
            "err_wss",
            "abs_err_wss",
        ],
        "normalized_wss_fields": {
            "wss_pred_over_wss_pred_max": "wss_pred_pa / max(wss_pred_pa) on this interpolated surface",
            "wss_truth_over_wss_truth_max": "wss_truth_pa / max(wss_truth_pa) on this interpolated surface",
            "wss_pred_over_wss_truth_max": "wss_pred_pa / max(wss_truth_pa) on this interpolated surface",
        },
        "normalization_denominators_pa": {
            "wss_pred_surface_max": float(np.max(output_arrays["wss_pred_pa"])),
            "wss_truth_surface_max": float(np.max(output_arrays["wss_truth_pa"])),
        },
        "metric_basis": (
            "visualization only; official WSS/velocity/pressure metrics remain on "
            "the original same-point evaluation arrays, not this interpolated STL"
        ),
        "verification": verification,
        "output_sha256": _sha256(output_path),
    }
    if backup_path is not None:
        report["backup_before_wss_normalized_fields"] = str(backup_path)
        report["backup_sha256"] = _sha256(backup_path)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[{arm} {selection}] {case_id}: "
        f"{len(target_points):,} points, {len(triangles):,} triangles, "
        f"coverage={valid_ratio:.2%}, map_dist_p95={report['distance_mm']['p95']:.4f} mm"
    )
    print(f"  VTP: {output_path}")
    print(f"  report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map the four selected WSS-PINN point-cloud VTPs to wall STL surfaces"
    )
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--radius", type=float, default=3.0, help="Gaussian radius in mm")
    parser.add_argument("--sharpness", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    audit_dir = args.audit_dir.resolve()
    batch_reports: list[dict] = []
    for arm in ("V2-E8", "V3-E6"):
        arm_dir = audit_dir / "vtp" / arm
        manifest_path = arm_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        arm_reports: list[dict] = []
        for case in manifest["cases"]:
            source_path = Path(case["vtp"]).resolve()
            stl_path = _resolve_stl(case["case_id"]).resolve()
            report = process_case(
                source_path=source_path,
                stl_path=stl_path,
                case_id=case["case_id"],
                selection=case["selection"],
                arm=arm,
                radius=args.radius,
                sharpness=args.sharpness,
                overwrite=args.overwrite,
            )
            arm_reports.append(report)
            batch_reports.append(report)

        surface_manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "arm": arm,
            "method": "gaussian",
            "params": {
                "radius_mm": args.radius,
                "sharpness": args.sharpness,
                "max_dist_mm": args.radius,
                "fallback": "mask",
            },
            "metric_basis": "postview visualization only; do not compute official metrics here",
            "cases": arm_reports,
        }
        (arm_dir / "surface_manifest.json").write_text(
            json.dumps(surface_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    batch_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "method": "gaussian",
        "params": {
            "radius_mm": args.radius,
            "sharpness": args.sharpness,
            "max_dist_mm": args.radius,
            "fallback": "mask",
        },
        "cases": batch_reports,
    }
    batch_path = audit_dir / "vtp/surface_mapping_manifest.json"
    batch_path.write_text(
        json.dumps(batch_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Batch manifest: {batch_path}")


if __name__ == "__main__":
    main()
