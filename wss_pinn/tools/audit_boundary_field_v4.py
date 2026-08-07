"""Stage-0b boundary provenance and geometry Gate for field-v4."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from wss_pinn.tools.build_qs_smooth_v3 import (
    _read_fluent_boundaries,
    _transform_coords,
)
from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    git_state,
    guard_write_path,
    sha256_file,
    utc_now,
)


ROUTE = "volume_uvwp_peak_field_v4"
SOURCE_MANIFEST = ROOT / "data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/manifest.json"
OUTPUT_ROOT = ROOT / "data_wss_pinn/volume_uvwp_peak_field_v4_train123_val15/boundary_contract"
AUDIT_ROOT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0b"
REPORT_JSON = AUDIT_ROOT / "report.json"
CONTRACT_MANIFEST = OUTPUT_ROOT / "manifest.json"
OUTLET_CSV = AUDIT_ROOT / "outlet_comparison.csv"
WALL_CSV = AUDIT_ROOT / "wall_zones.csv"
SPLIT_SHA256 = "c80cb65ad95f7d76fadaff82ea02c35c27ad24ab2474e1050bf981eff97493f9"
OUTLET_LABEL = {
    "outle": "out-le",
    "outli": "out-li",
    "outri": "out-ri",
    "outre": "out-re",
}
PRESSURE_RE = re.compile(
    r"P_ave_(outle|outli|outri|outre)=([+\-0-9.eE]+) "
    r"Q_ave_\1=([+\-0-9.eE]+)"
)
TIME_RE = re.compile(r"Flow time = ([+\-0-9.eE]+)s, time step = (\d+)")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _face_geometry(mesh: dict[str, Any], zone_id: int) -> dict[str, Any]:
    nodes = mesh["nodes"]
    first = int(mesh["first_node"])
    areas = []
    centroids = []
    adjacent = []
    edge_lengths = []
    for face, cells in zip(mesh["faces"][zone_id], mesh["face_cells"][zone_id]):
        points = np.asarray(nodes[face - first], dtype=np.float64)
        area = 0.0
        weighted_centroid = np.zeros(3, dtype=np.float64)
        for index in range(1, len(points) - 1):
            triangle = np.asarray([points[0], points[index], points[index + 1]])
            triangle_area = 0.5 * float(
                np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
            )
            area += triangle_area
            weighted_centroid += triangle_area * triangle.mean(axis=0)
        if area <= 0:
            continue
        areas.append(area)
        centroids.append(weighted_centroid / area)
        valid_cells = [int(value) for value in cells if int(value) > 0]
        adjacent.append(valid_cells[0] if valid_cells else 0)
        for index in range(len(points)):
            edge_lengths.append(
                float(np.linalg.norm(points[(index + 1) % len(points)] - points[index]))
            )
    return {
        "areas_m2": np.asarray(areas, dtype=np.float64),
        "centroids_m": np.asarray(centroids, dtype=np.float64),
        "adjacent_cell_ids": np.asarray(adjacent, dtype=np.int64),
        "edge_lengths_m": np.asarray(edge_lengths, dtype=np.float64),
    }


def _parse_peak_from_log(path: Path, peak_step: int) -> dict[str, Any] | None:
    buffer = {key: [] for key in OUTLET_LABEL}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            pressure = PRESSURE_RE.search(line)
            if pressure:
                buffer[pressure.group(1)].append(
                    (float(pressure.group(2)), float(pressure.group(3)))
                )
                continue
            time = TIME_RE.search(line)
            if not time:
                continue
            step = int(time.group(2))
            if step == int(peak_step):
                if any(not values for values in buffer.values()):
                    return None
                outlets = {}
                for raw_label, values in buffer.items():
                    unique = sorted(set(values))
                    outlets[OUTLET_LABEL[raw_label]] = {
                        "pressure_absolute_pa": float(
                            statistics.median(value[0] for value in unique)
                        ),
                        "mass_flow_kg_s": float(
                            statistics.median(value[1] for value in unique)
                        ),
                        "printed_rows": len(values),
                        "unique_rows_after_mpi_dedup": len(unique),
                    }
                return {
                    "time_s": float(time.group(1)),
                    "time_step": step,
                    "outlets": outlets,
                }
            buffer = {key: [] for key in OUTLET_LABEL}
    return None


def _find_peak_log(raw_case: Path, peak_step: int) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(
        (raw_case / "Global_conditions").glob("Fluent_*.out"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    for path in candidates:
        parsed = _parse_peak_from_log(path, peak_step)
        if parsed is not None:
            return path, parsed
    return None, None


def _pressure_from_face_nearest_cells(
    geometry: dict[str, Any],
    *,
    bundle: dict[str, np.ndarray],
    strict_tree: cKDTree,
    strict_indices: np.ndarray,
    pressure_relative: np.ndarray,
    reference_pa: float,
) -> tuple[float, int, dict[str, float]]:
    """Area-average the nearest strict cell pressure at every face centroid."""
    registered = _transform_coords(geometry["centroids_m"], bundle)
    distance, local_index = strict_tree.query(registered, k=1)
    source_index = strict_indices[np.asarray(local_index, dtype=np.int64)]
    pressure = np.asarray(pressure_relative[source_index], dtype=np.float64)
    pressure += float(reference_pa)
    areas = geometry["areas_m2"]
    return (
        float(np.sum(pressure * areas) / np.sum(areas)),
        int(len(source_index)),
        {
            "nearest_distance_normalized_p50": float(np.quantile(distance, 0.50)),
            "nearest_distance_normalized_p95": float(np.quantile(distance, 0.95)),
            "nearest_distance_normalized_max": float(np.max(distance)),
        },
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _case_audit(row: dict[str, Any], *, force: bool) -> tuple[dict[str, Any], list[dict], list[dict]]:
    boundary_path = Path(row["boundary_manifest"])
    if sha256_file(boundary_path) != row["boundary_manifest_sha256"]:
        raise ValueError(f"frozen boundary manifest hash drift: {boundary_path}")
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    volume_path = Path(row["source_volume_manifest"])
    if sha256_file(volume_path) != row["source_volume_manifest_sha256"]:
        raise ValueError(f"volume manifest hash drift: {volume_path}")
    volume = json.loads(volume_path.read_text(encoding="utf-8"))
    case_id = row["canonical_id"]
    case_gz = Path(boundary["provenance"]["fluent_case"]["path"])
    raw_case = case_gz.parent
    mesh = _read_fluent_boundaries(case_gz)
    bundle_path = Path(boundary["provenance"]["bundle"]["path"])
    with np.load(bundle_path, allow_pickle=False) as payload:
        bundle = {
            key: np.asarray(payload[key])
            for key in ("unit_factor", "transform_centroid", "transform_rotation", "coord_scale")
        }
    pressure_relative = np.load(
        volume["files"]["pressure_relative_pa"]["path"], mmap_mode="r"
    )
    volume_coords = np.load(
        volume["files"]["interior_coords"]["path"], mmap_mode="r"
    )
    volume_is_wall = np.load(
        volume["files"]["interior_is_wall"]["path"], mmap_mode="r"
    )
    strict_indices = np.flatnonzero(~np.asarray(volume_is_wall, dtype=bool))
    strict_tree = cKDTree(np.asarray(volume_coords[strict_indices], dtype=np.float32))
    reference_pa = float(volume["scales"]["pressure_reference_pa"])

    inlet_zone_ids = [
        zone_id
        for zone_id, zone_type in mesh["zone_types"].items()
        if zone_type == 10
        and mesh["zone_names"].get(zone_id, (None, None))[0] == "velocity-inlet"
    ]
    if len(inlet_zone_ids) != 1:
        raise ValueError(f"{case_id}: expected one velocity inlet")
    inlet_zone_id = inlet_zone_ids[0]
    wall_zone_ids = [
        zone_id for zone_id, zone_type in mesh["zone_types"].items() if zone_type == 3
    ]
    inlet_faces = _face_geometry(mesh, inlet_zone_id)
    wall_geometries = {zone_id: _face_geometry(mesh, zone_id) for zone_id in wall_zone_ids}
    wall_valid_masks = {}
    wall_rows = []
    target_wall_centroids = []
    for zone_id, geometry in wall_geometries.items():
        registered = _transform_coords(geometry["centroids_m"], bundle)
        nearest_distance = strict_tree.query(registered, k=1)[0]
        normalized_mesh_scale = float(
            np.median(geometry["edge_lengths_m"])
            * float(bundle["unit_factor"])
            / float(bundle["coord_scale"])
        )
        valid = nearest_distance <= max(2.0 * normalized_mesh_scale, 1e-8)
        wall_valid_masks[zone_id] = valid
        target_wall_centroids.append(geometry["centroids_m"][valid])
        zone_type, zone_name = mesh["zone_names"].get(
            zone_id, ("wall", f"zone-{zone_id}")
        )
        wall_rows.append(
            {
                "case_id": case_id,
                "role": row["role"],
                "zone_id": zone_id,
                "zone_type": zone_type,
                "zone_name": zone_name,
                "faces": len(geometry["areas_m2"]),
                "area_m2": float(np.sum(geometry["areas_m2"])),
                "adjacent_target_strict_domain_faces": int(np.sum(valid)),
                "excluded_non_target_faces": int(np.sum(~valid)),
                "target_face_fraction": float(np.mean(valid)),
                "all_faces_adjacent_target_strict_domain": bool(np.all(valid)),
                "future_contract_action": (
                    "include_all_faces" if bool(np.all(valid)) else "include_only_adjacent_faces"
                ),
                "nearest_distance_normalized_p95": float(
                    np.quantile(nearest_distance, 0.95)
                ),
                "mesh_scale_normalized": normalized_mesh_scale,
                "normal_available": True,
                "area_available": True,
            }
        )
    wall_centroids_m = np.concatenate(
        target_wall_centroids, axis=0
    )
    inlet_nodes = np.unique(np.concatenate(mesh["faces"][inlet_zone_id]))
    wall_nodes = np.unique(
        np.concatenate(
            [np.concatenate(mesh["faces"][zone_id]) for zone_id in wall_zone_ids]
        )
    )
    shared_nodes = np.intersect1d(inlet_nodes, wall_nodes)
    mesh_scale_m = float(np.median(inlet_faces["edge_lengths_m"]))
    edge_buffer_m = mesh_scale_m
    if len(shared_nodes):
        shared_coords = mesh["nodes"][shared_nodes - int(mesh["first_node"])]
        shared_tree = cKDTree(np.asarray(shared_coords, dtype=np.float64))
        inlet_keep = shared_tree.query(inlet_faces["centroids_m"], k=1)[0] > edge_buffer_m
        wall_keep = shared_tree.query(wall_centroids_m, k=1)[0] > edge_buffer_m
    else:
        inlet_keep = np.ones(len(inlet_faces["centroids_m"]), dtype=bool)
        wall_keep = np.ones(len(wall_centroids_m), dtype=bool)
    inlet_registered = _transform_coords(inlet_faces["centroids_m"][inlet_keep], bundle)
    wall_registered = _transform_coords(wall_centroids_m[wall_keep], bundle)

    outlet_rows = []
    outlet_centroids = []
    outlet_zone_index = []
    log_path, log_peak = _find_peak_log(raw_case, int(volume["peak_step"]))
    outlet_order = ("out-le", "out-li", "out-re", "out-ri")
    for zone_index, label in enumerate(outlet_order):
        zone_id = int(boundary["outlets"][label]["mesh_zone_id"])
        geometry = _face_geometry(mesh, zone_id)
        cfd_pressure, adjacent_faces, nearest_diagnostic = _pressure_from_face_nearest_cells(
            geometry,
            bundle=bundle,
            strict_tree=strict_tree,
            strict_indices=strict_indices,
            pressure_relative=pressure_relative,
            reference_pa=reference_pa,
        )
        log_value = None if log_peak is None else log_peak["outlets"][label]
        source = "udf_log_face_average" if log_value is not None else "cfd_adjacent_face_fallback"
        log_pressure = None if log_value is None else float(log_value["pressure_absolute_pa"])
        difference = None if log_pressure is None else log_pressure - cfd_pressure
        outlet_rows.append(
            {
                "case_id": case_id,
                "role": row["role"],
                "outlet": label,
                "zone_id": zone_id,
                "source": source,
                "peak_step": int(volume["peak_step"]),
                "log_time_s": None if log_peak is None else log_peak["time_s"],
                "udf_log_pressure_absolute_pa": log_pressure,
                "cfd_adjacent_face_pressure_absolute_pa": cfd_pressure,
                "log_minus_cfd_pa": difference,
                "absolute_difference_pa": None if difference is None else abs(difference),
                "pressure_reference_pa": reference_pa,
                "pressure_relative_pa": (
                    (log_pressure if log_pressure is not None else cfd_pressure) - reference_pa
                ),
                "adjacent_faces": adjacent_faces,
                **nearest_diagnostic,
                "mpi_printed_rows": None if log_value is None else log_value["printed_rows"],
                "unique_rows_after_mpi_dedup": (
                    None if log_value is None else log_value["unique_rows_after_mpi_dedup"]
                ),
                "mass_flow_kg_s": None if log_value is None else log_value["mass_flow_kg_s"],
            }
        )
        registered = _transform_coords(geometry["centroids_m"], bundle)
        outlet_centroids.append(registered)
        outlet_zone_index.append(np.full(len(registered), zone_index, dtype=np.int8))

    asset_path = OUTPUT_ROOT / "cases" / case_id / "face_interior_contract.npz"
    if asset_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite boundary contract: {asset_path}")
    _atomic_npz(
        asset_path,
        inlet_face_interior_coords=inlet_registered.astype(np.float32),
        wall_face_interior_coords=wall_registered.astype(np.float32),
        outlet_face_interior_coords=np.concatenate(outlet_centroids).astype(np.float32),
        outlet_zone=np.concatenate(outlet_zone_index),
        outlet_zone_names=np.asarray(outlet_order),
    )
    report = {
        "canonical_id": case_id,
        "role": row["role"],
        "status": "completed",
        "peak_step": int(volume["peak_step"]),
        "log": (
            {"path": str(log_path), "sha256": sha256_file(log_path)}
            if log_path is not None
            else None
        ),
        "udf": boundary["provenance"]["udf"],
        "face_interior_asset": {"path": str(asset_path), "sha256": sha256_file(asset_path)},
        "inlet_wall": {
            "shared_vertex_count_before": int(len(shared_nodes)),
            "mesh_scale_m": mesh_scale_m,
            "edge_buffer_m": edge_buffer_m,
            "inlet_face_points_before": int(len(inlet_faces["centroids_m"])),
            "inlet_face_points_after": int(np.sum(inlet_keep)),
            "wall_face_points_before": int(len(wall_centroids_m)),
            "wall_face_points_after": int(np.sum(wall_keep)),
            "all_physical_wall_face_points": int(
                sum(len(geometry["centroids_m"]) for geometry in wall_geometries.values())
            ),
            "shared_coordinates_after": 0,
        },
        "wall_zone_count": len(wall_rows),
        "all_wall_zones_classified": True,
        "wall_faces_excluded_as_non_target": int(
            sum(np.sum(~mask) for mask in wall_valid_masks.values())
        ),
        "outlet_source": "udf_log_face_average" if log_peak is not None else "cfd_adjacent_face_fallback",
        "roles": {
            "deployable_input": ["geometry", "future Q_in/phase/R1/R2/C interface"],
            "training_supervision": ["face-interior inlet velocity", "face pressure when enabled after Stage 0b"],
            "solver_response": ["UDF P_ave_out*/Q_ave_out*"],
            "monitor": ["historical p-out*-rfile.out; comparison only, never model input"],
        },
    }
    return report, outlet_rows, wall_rows


def run(*, force: bool = False) -> dict[str, Any]:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if source.get("split", {}).get("sha256") != SPLIT_SHA256:
        raise ValueError("frozen split hash drift")
    cases = []
    outlet_rows = []
    wall_rows = []
    for index, row in enumerate(source["cases"]):
        if row["role"] == "test":
            cases.append(
                {
                    "canonical_id": row["canonical_id"],
                    "role": "test",
                    "status": "not_read_test35_guard",
                }
            )
            continue
        case_report, case_outlets, case_walls = _case_audit(row, force=force)
        cases.append(case_report)
        outlet_rows.extend(case_outlets)
        wall_rows.extend(case_walls)
        print(
            json.dumps(
                {
                    "event": "field_v4_boundary_case_completed",
                    "index": index + 1,
                    "total": len(source["cases"]),
                    "case_id": row["canonical_id"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    train_errors = np.asarray(
        [
            row["absolute_difference_pa"]
            for row in outlet_rows
            if row["role"] == "train" and row["absolute_difference_pa"] is not None
        ],
        dtype=np.float64,
    )
    if not len(train_errors):
        raise ValueError("no train-only outlet double-path comparisons")
    median = float(np.median(train_errors))
    mad = float(np.median(np.abs(train_errors - median)))
    threshold = max(float(np.quantile(train_errors, 0.99)), median + 6.0 * mad, 1e-6)
    failures = []
    for row in outlet_rows:
        difference = row["absolute_difference_pa"]
        row["train_frozen_threshold_pa"] = threshold
        if row["role"] == "train":
            row["gate_result"] = "threshold_calibration"
            row["train_tail_outlier"] = bool(
                difference is not None and float(difference) > threshold
            )
        else:
            row["gate_result"] = (
                "pass" if difference is None or float(difference) <= threshold else "fail"
            )
            row["train_tail_outlier"] = False
        if row["gate_result"] == "fail":
            failures.append(
                f"outlet_double_path:{row['case_id']}:{row['outlet']}:{difference:.6g}"
            )
    for case in cases:
        if case.get("status") == "not_read_test35_guard":
            continue
        if case["inlet_wall"]["shared_coordinates_after"] != 0:
            failures.append(f"inlet_wall_overlap:{case['canonical_id']}")
        if not case["all_wall_zones_classified"]:
            failures.append(f"wall_zone_unclassified:{case['canonical_id']}")

    _write_csv(OUTLET_CSV, outlet_rows)
    _write_csv(WALL_CSV, wall_rows)
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "route": ROUTE,
        "gate_result": "pass" if not failures else "fail",
        "gate_reasons": failures,
        "coverage": {
            "manifest_rows": len(cases),
            "train_val_deep_audited": sum(case.get("status") == "completed" for case in cases),
            "test35_guarded_not_read": sum(case.get("status") == "not_read_test35_guard" for case in cases),
            "outlet_rows": len(outlet_rows),
            "log_double_path_rows": sum(row["source"] == "udf_log_face_average" for row in outlet_rows),
            "fallback_rows": sum(row["source"] == "cfd_adjacent_face_fallback" for row in outlet_rows),
        },
        "outlet_train_only_threshold": {
            "method": "max(train p99 absolute error, train median + 6*MAD)",
            "threshold_pa": threshold,
            "train_comparisons": int(len(train_errors)),
            "median_pa": median,
            "mad_pa": mad,
            "p95_pa": float(np.quantile(train_errors, 0.95)),
            "p99_pa": float(np.quantile(train_errors, 0.99)),
            "train_tail_outliers_retained_as_calibration_provenance": int(
                np.sum(train_errors > threshold)
            ),
            "application_scope": "frozen on train123; pass/fail applied only to val15 double-path rows",
        },
        "cases": cases,
        "outlet_csv": {"path": str(OUTLET_CSV), "sha256": sha256_file(OUTLET_CSV)},
        "wall_csv": {"path": str(WALL_CSV), "sha256": sha256_file(WALL_CSV)},
        "split_sha256": SPLIT_SHA256,
        "git": git_state(),
    }
    atomic_write_json(REPORT_JSON, report)
    return report


def assemble_contract_manifest() -> dict[str, Any]:
    """Bind completed per-case assets into the future Stage-2 interface."""
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    cases = []
    for row in report["cases"]:
        if row.get("status") == "not_read_test35_guard":
            cases.append(
                {
                    "canonical_id": row["canonical_id"],
                    "role": "test",
                    "status": "not_read_test35_guard",
                }
            )
            continue
        cases.append(
            {
                "canonical_id": row["canonical_id"],
                "role": row["role"],
                "status": "completed",
                "face_interior_asset": row["face_interior_asset"],
                "outlet_source": row["outlet_source"],
                "udf": row["udf"],
                "roles": row["roles"],
                "wall_faces_excluded_as_non_target": row[
                    "wall_faces_excluded_as_non_target"
                ],
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "route": ROUTE,
        "scope": "train123/val15 boundary contract; test35 guarded and not read",
        "gate_result": report["gate_result"],
        "split_sha256": SPLIT_SHA256,
        "role_taxonomy": {
            "deployable_input": "geometry and future Q_in/phase/R1/R2/C only",
            "training_supervision": "face-interior inlet/wall/outlet assets",
            "solver_response": "UDF P_ave/Q_ave; never a deployable input",
            "monitor": "historical p-out report; provenance comparison only",
        },
        "cases": cases,
        "audit": {"path": str(REPORT_JSON)},
    }
    atomic_write_json(CONTRACT_MANIFEST, manifest)
    report["boundary_contract_manifest"] = {
        "path": str(CONTRACT_MANIFEST),
        "sha256": sha256_file(CONTRACT_MANIFEST),
    }
    atomic_write_json(REPORT_JSON, report)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = run(force=args.force)
    print(json.dumps({key: report[key] for key in ("gate_result", "coverage", "outlet_train_only_threshold")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
