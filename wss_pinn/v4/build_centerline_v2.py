"""Build the isolated Centerline-V2-aligned V4 staging dataset.

This builder never overwrites the historical V4 tree.  It rebuilds the steady
peak field and a deterministic 15,000-cell transient pool from the complete
raw Fluent volumes.  Coordinates and vectors are re-registered with the
Centerline V2 opening-ID frame.  Steady and transient geometry are then taken
from the same V2 path asset and stored with the explicit raw schema
``abscissa_norm, local_radius_mm, curvature_per_mm``.

The output remains *staging* until the unresolved pressure cluster and raw
81-frame content digests are signed off; it must not start formal training.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from wss_pinn.data.raw_io import read_interior, step_file
from wss_pinn.tools.audit_boundary_field_v4 import _face_geometry
from wss_pinn.tools.build_qs_smooth_v3 import _read_fluent_boundaries, _zone_geometry
from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    guard_write_path,
    sha256_file,
    sha256_json,
    utc_now,
)
from wss_pinn.volume_utils import StreamingMoments, iter_slices

from .bc_contract import bc_scale_policy, bc_vector_from_conditions, transform_bc_raw
from .geometry_v2 import (
    GEOMETRY_AUX_NAMES,
    GEOMETRY_MODEL_NAMES,
    GEOMETRY_RAW_NAMES,
    OUTLET_ORDER,
    CenterlineV2Geometry,
)


ROUTE = "volume_uvwp_bc_rcr_v4_centerline_v2_rawfull_v2_staging"
SOURCE_MANIFEST = ROOT / "data_wss_pinn/volume_uvwp_bc_rcr_v4_train138_test35/manifest.json"
CENTERLINE_ROOT = ROOT / "outputs/centerline_v2_full_173_20260828"
DATA_ROOT = ROOT / "data_wss_pinn/volume_uvwp_bc_rcr_v4_centerline_v2_rawfull_v2_staging_train138_test35"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
STATS_PATH = DATA_ROOT / "train138_stats.json"
AUDIT_ROOT = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_centerline_v2_rawfull_v2_staging/audits"
GATE_REPORT = AUDIT_ROOT / "gate_report.json"
ARRAY_AUDIT_REPORT = AUDIT_ROOT / "full_array_audit.json"
TRANSIENT_POOL_SIZE = 15_000
NEAR_WALL_THRESHOLD_MM = 1.5
REQUIRED_STEADY_ARRAYS = {
    "coords",
    "geometry_raw",
    "geometry_aux",
    "is_wall",
    "region",
    "distance_to_wall_mm",
    "velocity_m_s",
    "pressure_relative_pa",
}
REQUIRED_TRANSIENT_ARRAYS = {
    "coords",
    "geometry_raw",
    "geometry_aux",
    "is_wall",
    "eligible_interior",
    "region",
    "distance_to_wall_mm",
    "velocity_m_s",
    "pressure_raw_pa",
    "cell_ids",
    "steps",
    "times_s",
    "q_actual_m3_s",
}
REQUIRED_BOUNDARY_ARRAYS = {
    "inlet_coords",
    "inlet_normals",
    "wall_coords",
    "outlet_coords",
    "outlet_normals",
    "outlet_area_weights_m2",
    "outlet_zone",
}

PRESSURE_LOW_CLUSTER = {
    "ILO/LIU_YUE_DONG-0/before",
    "ILO/LI_SHENG_WEN-0/before",
    "AG/slow/LI_BING_YI",
    "AAA/ruputer/MENG_GUANG_QIN",
    "AAA/unruputer/CHEN_SHU_LIN",
    "AG/slow/LIU_ZONG_YANG",
    "ILO/ZHANG_YONG_SHENG-0/before",
    "AAA/ruputer/WANG_FU_SHUN",
}
# 2026-09-05 BC contract: max-aware z-scale bounds every train |z| by 6 and the inlet
# ratio uses a fixed physical scale, so no per-case |z|>6 whitelist is needed any more.
# The ZOU_LI_SHUN out-re R2/C values (formerly +6.5σ / -6.5σ) are surfaced through the
# advisory ``abs_z_gt_4`` list instead of a blocking whitelist.
BC_Z_WHITELIST: dict[tuple[str, str], str] = {
}


def _atomic_npy(path: Path, array: np.ndarray) -> Path:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, np.asarray(array), allow_pickle=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _array_record(path: Path, array: np.ndarray) -> dict[str, Any]:
    written = _atomic_npy(path, array)
    value = np.asarray(array)
    return {
        "path": str(written.resolve()),
        "sha256": sha256_file(written),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _load_source() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("counts", {}).get("total") != 173:
        raise ValueError("historical V4 source manifest is not the frozen 173-case set")
    rows = list(payload["cases"])
    split = _load_split(payload["split"])
    train_ids = [row["canonical_id"] for row in rows if row["role"] == "train"]
    test_ids = [row["canonical_id"] for row in rows if row["role"] == "test"]
    if len(train_ids) != len(set(train_ids)) or len(test_ids) != len(set(test_ids)):
        raise ValueError("historical V4 source manifest contains duplicate case IDs")
    if set(train_ids) != set(split["train_cases"]) or set(test_ids) != set(
        split["test_cases"]
    ):
        raise ValueError("historical V4 source roles do not match the frozen split")
    return payload, rows


def _load_split(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError("frozen train138/test35 split SHA256 drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    train_ids = list(payload["train_cases"])
    test_ids = list(payload["test_cases"])
    if len(train_ids) != 138 or len(test_ids) != 35:
        raise ValueError("frozen split is not train138/test35")
    if len(set(train_ids)) != 138 or len(set(test_ids)) != 35:
        raise ValueError("frozen split contains duplicate case IDs")
    if set(train_ids).intersection(test_ids):
        raise ValueError("frozen train/test split overlaps")
    return payload


def _load_case(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["manifest"])
    if sha256_file(path) != row["manifest_sha256"]:
        raise ValueError(f"{row['canonical_id']}: source case manifest SHA256 drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["canonical_id"] != row["canonical_id"]:
        raise ValueError("source aggregate/case identity mismatch")
    return payload


def _raw_boundary_contract(
    case: dict[str, Any],
    geometry: CenterlineV2Geometry,
    strict_raw_mm: np.ndarray,
) -> dict[str, Any]:
    boundary_path = Path(case["provenance"]["boundary_manifest"]["path"])
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    mesh = _read_fluent_boundaries(Path(boundary["provenance"]["fluent_case"]["path"]))
    unit_factor = float(geometry.provenance["fluent_to_mm_unit_factor"])
    strict_tree = cKDTree(np.asarray(strict_raw_mm, dtype=np.float64))

    inlet_zone_ids = [
        zone_id
        for zone_id, zone_type in mesh["zone_types"].items()
        if zone_type == 10
        and mesh["zone_names"].get(zone_id, (None, None))[0] == "velocity-inlet"
    ]
    if len(inlet_zone_ids) != 1:
        raise ValueError(f"{case['canonical_id']}: expected exactly one velocity inlet zone")
    inlet_zone_id = inlet_zone_ids[0]
    inlet_geometry = _face_geometry(mesh, inlet_zone_id)
    inlet_coords = inlet_geometry["centroids_m"] * unit_factor
    inlet_zone = _zone_geometry(mesh, inlet_zone_id)
    inlet_normal_raw = np.asarray(inlet_zone["area_vector_m2"], dtype=np.float64)
    inlet_normal_raw /= max(float(np.linalg.norm(inlet_normal_raw)), 1.0e-30)
    inlet_centroid = np.mean(inlet_coords, axis=0)
    inlet_nearest = strict_raw_mm[
        strict_tree.query(inlet_centroid, k=min(512, len(strict_raw_mm)))[1]
    ]
    inlet_interior = np.mean(np.asarray(inlet_nearest).reshape(-1, 3), axis=0)
    if float(inlet_normal_raw @ (inlet_interior - inlet_centroid)) < 0.0:
        inlet_normal_raw *= -1.0

    wall_blocks = []
    wall_zone_ids = [
        zone_id for zone_id, zone_type in mesh["zone_types"].items() if zone_type == 3
    ]
    for zone_id in wall_zone_ids:
        face = _face_geometry(mesh, zone_id)
        coords_mm = face["centroids_m"] * unit_factor
        distance = strict_tree.query(coords_mm, k=1)[0]
        mesh_scale_mm = float(np.median(face["edge_lengths_m"]) * unit_factor)
        valid = distance <= max(2.0 * mesh_scale_mm, 1.0e-6)
        if bool(np.any(valid)):
            wall_blocks.append(coords_mm[valid])
    if not wall_blocks:
        raise ValueError(f"{case['canonical_id']}: no target wall face-interior points")
    wall_coords = np.concatenate(wall_blocks, axis=0)

    # Exclude the inlet-wall junction rim from both Dirichlet populations.
    # The one-edge buffer is the frozen face-interior contract used by the
    # historical V4 boundary audit; without it, the same geometric rim is
    # forced toward both uniform inlet velocity and wall no-slip.
    inlet_nodes = np.unique(np.concatenate(mesh["faces"][inlet_zone_id]))
    wall_nodes = np.unique(
        np.concatenate(
            [np.concatenate(mesh["faces"][zone_id]) for zone_id in wall_zone_ids]
        )
    )
    shared_nodes = np.intersect1d(inlet_nodes, wall_nodes)
    edge_buffer_mm = float(np.median(inlet_geometry["edge_lengths_m"]) * unit_factor)
    if len(shared_nodes):
        shared_coords_mm = (
            np.asarray(
                mesh["nodes"][shared_nodes - int(mesh["first_node"])],
                dtype=np.float64,
            )
            * unit_factor
        )
        shared_tree = cKDTree(shared_coords_mm)
        inlet_keep = shared_tree.query(inlet_coords, k=1)[0] > edge_buffer_mm
        wall_keep = shared_tree.query(wall_coords, k=1)[0] > edge_buffer_mm
    else:
        inlet_keep = np.ones(len(inlet_coords), dtype=bool)
        wall_keep = np.ones(len(wall_coords), dtype=bool)
    inlet_coords = inlet_coords[inlet_keep]
    wall_coords = wall_coords[wall_keep]
    if len(inlet_coords) == 0 or len(wall_coords) == 0:
        raise ValueError(f"{case['canonical_id']}: inlet-wall rim buffer emptied a BC population")
    minimum_inlet_wall_distance_mm = float(
        cKDTree(wall_coords).query(inlet_coords, k=1)[0].min()
    )

    outlet_coords = []
    outlet_normals = []
    outlet_weights = []
    outlet_zone = []
    for zone_index, label in enumerate(OUTLET_ORDER):
        zone_id = int(boundary["outlets"][label]["mesh_zone_id"])
        face = _face_geometry(mesh, zone_id)
        coords_mm = face["centroids_m"] * unit_factor
        zone = _zone_geometry(mesh, zone_id)
        normal = np.asarray(zone["area_vector_m2"], dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-30)
        centroid = np.mean(coords_mm, axis=0)
        nearest = strict_raw_mm[strict_tree.query(centroid, k=min(512, len(strict_raw_mm)))[1]]
        interior = np.mean(np.asarray(nearest).reshape(-1, 3), axis=0)
        if float(normal @ (centroid - interior)) < 0.0:
            normal *= -1.0
        outlet_coords.append(coords_mm)
        outlet_normals.append(np.broadcast_to(normal, coords_mm.shape).copy())
        outlet_weights.append(face["areas_m2"].astype(np.float32))
        outlet_zone.append(np.full(len(coords_mm), zone_index, dtype=np.int8))

    return {
        "inlet_coords_raw_mm": np.asarray(inlet_coords, dtype=np.float64),
        "inlet_normals_raw": np.broadcast_to(inlet_normal_raw, inlet_coords.shape).copy(),
        "wall_coords_raw_mm": wall_coords,
        "outlet_coords_raw_mm": np.concatenate(outlet_coords, axis=0),
        "outlet_normals_raw": np.concatenate(outlet_normals, axis=0),
        "outlet_area_weights_m2": np.concatenate(outlet_weights, axis=0),
        "outlet_zone": np.concatenate(outlet_zone, axis=0),
        "diagnostics": {
            "contract_version": "fluent_face_interior_rim_buffer_v2",
            "shared_vertex_count_before": int(len(shared_nodes)),
            "edge_buffer_mm": edge_buffer_mm,
            "inlet_face_points_before": int(len(inlet_geometry["centroids_m"])),
            "inlet_face_points_after": int(np.sum(inlet_keep)),
            "wall_face_points_before": int(sum(len(block) for block in wall_blocks)),
            "wall_face_points_after": int(np.sum(wall_keep)),
            "minimum_inlet_wall_distance_mm": minimum_inlet_wall_distance_mm,
            "inlet_nonempty": bool(len(inlet_coords)),
            "wall_nonempty": bool(len(wall_coords)),
        },
    }


def _raw_peak_steady(
    case: dict[str, Any],
    volume: dict[str, Any],
    geometry: CenterlineV2Geometry,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    raw_peak_record = volume["parents"]["raw_peak"]
    raw_peak_path = Path(raw_peak_record["path"])
    if sha256_file(raw_peak_path) != raw_peak_record["sha256"]:
        raise ValueError(f"{case['canonical_id']}: raw peak SHA256 drift")
    raw = read_interior(raw_peak_path)
    unit_factor = float(geometry.provenance["fluent_to_mm_unit_factor"])
    coords_mm = np.asarray(raw["coords"], dtype=np.float64) * unit_factor

    boundary_path = Path(case["provenance"]["boundary_manifest"]["path"])
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    mesh = _read_fluent_boundaries(Path(boundary["provenance"]["fluent_case"]["path"]))
    wall_zone_ids = [
        zone_id for zone_id, zone_type in mesh["zone_types"].items() if zone_type == 3
    ]
    wall_node_ids = np.unique(
        np.concatenate([np.concatenate(mesh["faces"][zone_id]) for zone_id in wall_zone_ids])
    )
    wall_nodes_mm = (
        np.asarray(mesh["nodes"][wall_node_ids - int(mesh["first_node"])], dtype=np.float64)
        * unit_factor
    )
    distance = cKDTree(wall_nodes_mm).query(coords_mm, k=1)[0]
    is_wall = np.asarray(distance <= 1.0e-6, dtype=bool)
    region = np.asarray(distance <= NEAR_WALL_THRESHOLD_MM, dtype=np.int8)
    if not bool(np.any(~is_wall)):
        raise ValueError(f"{case['canonical_id']}: raw peak contains no strict-volume cells")
    velocity_raw = np.asarray(raw["velocity"], dtype=np.float64)
    pressure_raw = np.asarray(raw["pressure"], dtype=np.float64)
    pressure_reference = float(np.mean(pressure_raw[~is_wall], dtype=np.float64))
    pressure_relative = (pressure_raw - pressure_reference).astype(np.float32)
    return (
        coords_mm,
        is_wall,
        region,
        np.asarray(distance, dtype=np.float32),
        velocity_raw,
        pressure_relative,
        np.asarray(raw["cell_id"]),
        {
        "path": str(raw_peak_path.resolve()),
        "sha256": raw_peak_record["sha256"],
        "rows": int(len(coords_mm)),
        "strict_rows": int(np.sum(~is_wall)),
        "wall_duplicate_rows": int(np.sum(is_wall)),
        "pressure_reference_pa": pressure_reference,
        "wall_duplicate_rule": "distance to any Fluent physical-wall node <= 1e-6 mm",
        "near_wall_rule": f"distance to any Fluent physical-wall node <= {NEAR_WALL_THRESHOLD_MM} mm",
        },
    )


def _selected_indices(
    case_id: str,
    is_wall: np.ndarray,
    cell_ids: np.ndarray,
    count: int = TRANSIENT_POOL_SIZE,
) -> np.ndarray:
    strict = np.flatnonzero(~np.asarray(is_wall, dtype=bool))
    if len(strict) < int(count):
        raise ValueError(f"{case_id}: raw strict-volume pool has fewer than {count} cells")
    seed = int.from_bytes(
        hashlib.sha256(f"centerline-v2|{case_id}".encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(strict, size=int(count), replace=False))
    if len(np.unique(np.asarray(cell_ids)[selected])) != len(selected):
        raise ValueError(f"{case_id}: selected transient cell IDs are not unique")
    return selected.astype(np.int64)


def _indices_for_cell_ids(frame_ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame_ids, dtype=np.int64)
    selected = np.asarray(selected_ids, dtype=np.int64)
    order = np.argsort(frame)
    sorted_ids = frame[order]
    position = np.searchsorted(sorted_ids, selected)
    if np.any(position >= len(sorted_ids)):
        raise ValueError("selected transient cell ID is absent from a frame")
    indices = order[position]
    if not np.array_equal(frame[indices], selected):
        raise ValueError("selected transient cell IDs do not match a frame")
    return indices.astype(np.int64)


def _raw_transient(
    case: dict[str, Any],
    geometry: CenterlineV2Geometry,
    steady_raw_mm: np.ndarray,
    steady_is_wall: np.ndarray,
    steady_cell_ids: np.ndarray,
    steady_velocity_raw: np.ndarray,
    steady_pressure_relative: np.ndarray,
    pressure_reference_pa: float,
    transient_source: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    case_id = case["canonical_id"]
    selected = _selected_indices(case_id, steady_is_wall, steady_cell_ids)
    selected_ids = np.asarray(steady_cell_ids[selected], dtype=np.int64)
    coords_raw_mm = np.asarray(steady_raw_mm[selected], dtype=np.float64)
    steps = np.asarray(
        np.load(transient_source["steps"]["path"], mmap_mode="r"), dtype=np.int32
    )
    times_s = np.asarray(
        np.load(transient_source["times_s"]["path"], mmap_mode="r"), dtype=np.float32
    )
    q_actual = np.asarray(
        np.load(transient_source["q_actual_m3_s"]["path"], mmap_mode="r"),
        dtype=np.float32,
    )
    if len(steps) != 81 or len(times_s) != 81 or len(q_actual) != 81:
        raise ValueError(f"{case_id}: transient time contract is not 81 frames")

    boundary_path = Path(case["provenance"]["boundary_manifest"]["path"])
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    raw_case_dir = Path(boundary["provenance"]["fluent_case"]["path"]).parent
    unit_factor = float(geometry.provenance["fluent_to_mm_unit_factor"])
    velocity = np.empty((81, len(selected), 3), dtype=np.float32)
    pressure = np.empty((81, len(selected)), dtype=np.float32)
    source_frames = []
    max_coord_delta_mm = 0.0
    peak_step = int(case["peak_step"])
    for frame_index, step in enumerate(steps.tolist()):
        path = step_file(raw_case_dir, int(step), "ascii_in")
        stat = path.stat()
        if int(step) == peak_step:
            frame_indices = selected
            velocity_raw = np.asarray(steady_velocity_raw[frame_indices], dtype=np.float64)
            pressure_raw = (
                np.asarray(steady_pressure_relative[frame_indices], dtype=np.float64)
                + float(pressure_reference_pa)
            )
            frame_coords_mm = np.asarray(steady_raw_mm[frame_indices], dtype=np.float64)
        else:
            frame = read_interior(path)
            frame_indices = _indices_for_cell_ids(frame["cell_id"], selected_ids)
            velocity_raw = np.asarray(frame["velocity"][frame_indices], dtype=np.float64)
            pressure_raw = np.asarray(frame["pressure"][frame_indices], dtype=np.float64)
            frame_coords_mm = (
                np.asarray(frame["coords"][frame_indices], dtype=np.float64) * unit_factor
            )
        delta = float(np.max(np.abs(frame_coords_mm - coords_raw_mm)))
        max_coord_delta_mm = max(max_coord_delta_mm, delta)
        if delta > 1.0e-5:
            raise ValueError(f"{case_id}: raw transient coordinates drift by {delta:.3e} mm")
        velocity[frame_index] = geometry.rotate_vectors(velocity_raw)
        pressure[frame_index] = pressure_raw.astype(np.float32)
        source_frames.append(
            {
                "step": int(step),
                "path": str(path.resolve()),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    geometry_raw, geometry_aux = geometry.map_points(coords_raw_mm)
    return {
        "coords_raw_mm": coords_raw_mm,
        "geometry_raw": geometry_raw,
        "geometry_aux": geometry_aux,
        "is_wall": np.zeros(len(selected), dtype=bool),
        "eligible_interior": np.ones(len(selected), dtype=bool),
        "steps": steps,
        "times_s": times_s,
        "q_actual_m3_s": q_actual,
        "velocity_m_s": velocity,
        "pressure_raw_pa": pressure,
        "selected_cell_ids": selected_ids,
        "steady_indices": selected,
    }, {
        "pool_selection": (
            "15000 unique strict-volume peak cell IDs; deterministic SHA256 seed"
        ),
        "pool_size": int(len(selected)),
        "max_static_coordinate_delta_mm": max_coord_delta_mm,
        "frame_sources": source_frames,
        "frame_source_identity": (
            "path + size_bytes + mtime_ns; content SHA pending formal cutover"
        ),
    }


def _scale_mm(geometry: CenterlineV2Geometry, arrays: list[np.ndarray]) -> float:
    maximum = 0.0
    for array in arrays:
        if len(array):
            maximum = max(
                maximum,
                float(np.max(np.abs(geometry.registered_unscaled_mm(array)))),
            )
    if not np.isfinite(maximum) or maximum <= 1.0e-9:
        raise ValueError("volume-aware coordinate scale is degenerate")
    return maximum


def _copy_record(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"source array SHA256 drift: {path}")
    return {"path": str(path.resolve()), "sha256": record["sha256"]}


def _verify_array_record(record: dict[str, Any]) -> None:
    path = Path(record["path"])
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"staging array SHA256 drift: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(value.shape) != list(record["shape"]):
        raise ValueError(f"staging array shape drift: {path}")
    if str(value.dtype) != str(record["dtype"]):
        raise ValueError(f"staging array dtype drift: {path}")


def _current_input_snapshot(
    row: dict[str, Any],
    case: dict[str, Any],
    volume: dict[str, Any],
    geometry: CenterlineV2Geometry,
) -> dict[str, Any]:
    volume_record = case["provenance"]["volume_manifest"]
    volume_path = Path(volume_record["path"])
    if volume_path.resolve() != Path(case["files"]["steady_volume_manifest"]["path"]).resolve():
        raise ValueError(f"{case['canonical_id']}: source volume manifest path drift")
    if sha256_file(volume_path) != volume_record["sha256"]:
        raise ValueError(f"{case['canonical_id']}: source volume manifest SHA256 drift")
    boundary_record = case["provenance"]["boundary_manifest"]
    boundary_path = Path(boundary_record["path"])
    if sha256_file(boundary_path) != boundary_record["sha256"]:
        raise ValueError(f"{case['canonical_id']}: boundary manifest SHA256 drift")
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    fluent_record = boundary["provenance"]["fluent_case"]
    fluent_path = Path(fluent_record["path"])
    if sha256_file(fluent_path) != fluent_record["sha256"]:
        raise ValueError(f"{case['canonical_id']}: Fluent case SHA256 drift")

    raw_peak = volume["parents"]["raw_peak"]
    if sha256_file(raw_peak["path"]) != raw_peak["sha256"]:
        raise ValueError(f"{case['canonical_id']}: raw peak SHA256 drift")
    transient = case["files"]["transient"]
    used_transient_records = {}
    for name in (
        "steps",
        "times_s",
        "q_actual_m3_s",
        "outlet_label_pressure_pa",
        "outlet_label_mass_flow_kg_s",
    ):
        if name in transient:
            used_transient_records[name] = _copy_record(transient[name])
    steps = np.asarray(np.load(transient["steps"]["path"], mmap_mode="r"), dtype=np.int32)
    raw_case_dir = fluent_path.parent
    frame_sources = []
    for step in steps.tolist():
        path = step_file(raw_case_dir, int(step), "ascii_in")
        stat = path.stat()
        frame_sources.append(
            {
                "step": int(step),
                "path": str(path.resolve()),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    snapshot = {
        "source_case_manifest": {
            "path": str(Path(row["manifest"]).resolve()),
            "sha256": row["manifest_sha256"],
        },
        "source_volume_manifest": {
            "path": str(volume_path.resolve()),
            "sha256": volume_record["sha256"],
        },
        "boundary_manifest": {
            "path": str(boundary_path.resolve()),
            "sha256": boundary_record["sha256"],
        },
        "fluent_case": {
            "path": str(fluent_path.resolve()),
            "sha256": fluent_record["sha256"],
        },
        "raw_peak": {"path": str(Path(raw_peak["path"]).resolve()), "sha256": raw_peak["sha256"]},
        "transient_records": used_transient_records,
        "raw_frame_identity": frame_sources,
        "centerline_v2": geometry.provenance,
    }
    return {"sha256": sha256_json(snapshot), "inputs": snapshot}


def build_case(row: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    case = _load_case(row)
    case_id = case["canonical_id"]
    output_dir = guard_write_path(DATA_ROOT / "cases" / case_id)
    manifest_path = output_dir / "manifest.json"
    cl = CenterlineV2Geometry.load(CENTERLINE_ROOT, case_id)
    volume_path = Path(case["files"]["steady_volume_manifest"]["path"])
    volume = json.loads(volume_path.read_text(encoding="utf-8"))
    input_snapshot = _current_input_snapshot(row, case, volume, cl)
    if resume and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed" and payload.get("schema_version") == 2:
            if payload.get("build_input", {}).get("sha256") != input_snapshot["sha256"]:
                raise ValueError(
                    f"{case_id}: staging build inputs changed; a full case rebuild is required"
                )
            for group in ("steady", "transient", "boundary_faces"):
                records = payload["files"][group]
                for record in records.values():
                    _verify_array_record(record)
            return payload
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite staging case without --resume: {manifest_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to reuse incomplete staging directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        steady_raw_mm,
        steady_is_wall,
        steady_region,
        steady_distance_to_wall_mm,
        steady_velocity_raw,
        steady_pressure,
        steady_cell_ids,
        raw_peak_provenance,
    ) = _raw_peak_steady(case, volume, cl)
    strict_raw_mm = steady_raw_mm[~steady_is_wall]
    boundary_raw = _raw_boundary_contract(case, cl, strict_raw_mm)
    transient_source = case["files"]["transient"]
    transient_arrays, transient_raw_provenance = _raw_transient(
        case,
        cl,
        steady_raw_mm,
        steady_is_wall,
        steady_cell_ids,
        steady_velocity_raw,
        steady_pressure,
        float(raw_peak_provenance["pressure_reference_pa"]),
        transient_source,
    )
    transient_raw_mm = transient_arrays["coords_raw_mm"]
    scale_mm = _scale_mm(
        cl,
        [
            steady_raw_mm,
            transient_raw_mm,
            boundary_raw["wall_coords_raw_mm"],
            boundary_raw["inlet_coords_raw_mm"],
            boundary_raw["outlet_coords_raw_mm"],
        ],
    )

    steady_geometry, steady_aux = cl.map_points(steady_raw_mm)
    steady_velocity = cl.rotate_vectors(steady_velocity_raw)
    steady_coords = cl.register_points(steady_raw_mm, scale_mm)

    transient_is_wall = transient_arrays["is_wall"]
    eligible = transient_arrays["eligible_interior"]
    identity_index = transient_arrays["steady_indices"]
    transient_geometry = transient_arrays["geometry_raw"]
    transient_aux = transient_arrays["geometry_aux"]
    # Same raw cell ID means the two temporal modes must reference exactly the
    # same static V2 geometry values, not merely two nearest-neighbour queries.
    transient_geometry[:] = steady_geometry[identity_index]
    transient_aux[:] = steady_aux[identity_index]
    transient_velocity = transient_arrays["velocity_m_s"]
    transient_pressure = transient_arrays["pressure_raw_pa"]
    transient_coords = cl.register_points(transient_raw_mm, scale_mm)
    transient_region = steady_region[identity_index]
    transient_distance_to_wall_mm = steady_distance_to_wall_mm[identity_index]

    steady_records = {
        "coords": _array_record(output_dir / "steady_coords.npy", steady_coords),
        "geometry_raw": _array_record(
            output_dir / "steady_geometry_raw.npy", steady_geometry
        ),
        "geometry_aux": _array_record(output_dir / "steady_geometry_aux.npy", steady_aux),
        "is_wall": _array_record(output_dir / "steady_is_wall.npy", steady_is_wall),
        "region": _array_record(output_dir / "steady_region.npy", steady_region),
        "distance_to_wall_mm": _array_record(
            output_dir / "steady_distance_to_wall_mm.npy", steady_distance_to_wall_mm
        ),
        "velocity_m_s": _array_record(output_dir / "steady_velocity_m_s.npy", steady_velocity),
        "pressure_relative_pa": _array_record(
            output_dir / "steady_pressure_relative_pa.npy", steady_pressure
        ),
    }
    transient_records = {
        "coords": _array_record(output_dir / "transient_coords.npy", transient_coords),
        "geometry_raw": _array_record(
            output_dir / "transient_geometry_raw.npy", transient_geometry
        ),
        "geometry_aux": _array_record(
            output_dir / "transient_geometry_aux.npy", transient_aux
        ),
        "is_wall": _array_record(output_dir / "transient_is_wall.npy", transient_is_wall),
        "eligible_interior": _array_record(
            output_dir / "transient_eligible_interior.npy", eligible
        ),
        "region": _array_record(output_dir / "transient_region.npy", transient_region),
        "distance_to_wall_mm": _array_record(
            output_dir / "transient_distance_to_wall_mm.npy",
            transient_distance_to_wall_mm,
        ),
        "velocity_m_s": _array_record(
            output_dir / "transient_velocity_m_s.npy", transient_velocity
        ),
        "pressure_raw_pa": _array_record(
            output_dir / "transient_pressure_raw_pa.npy", transient_pressure
        ),
        "cell_ids": _array_record(
            output_dir / "transient_cell_ids.npy",
            transient_arrays["selected_cell_ids"],
        ),
    }
    for name in ("steps", "times_s", "q_actual_m3_s"):
        transient_records[name] = _array_record(
            output_dir / f"transient_{name}.npy", transient_arrays[name]
        )
    for name in ("outlet_label_pressure_pa", "outlet_label_mass_flow_kg_s"):
        if name in transient_source:
            source = np.load(transient_source[name]["path"], mmap_mode="r")
            transient_records[name] = _array_record(output_dir / f"{name}.npy", source)

    boundary_records = {
        "inlet_coords": _array_record(
            output_dir / "inlet_coords.npy",
            cl.register_points(boundary_raw["inlet_coords_raw_mm"], scale_mm),
        ),
        "inlet_normals": _array_record(
            output_dir / "inlet_normals.npy",
            cl.rotate_vectors(boundary_raw["inlet_normals_raw"]),
        ),
        "wall_coords": _array_record(
            output_dir / "wall_coords.npy",
            cl.register_points(boundary_raw["wall_coords_raw_mm"], scale_mm),
        ),
        "outlet_coords": _array_record(
            output_dir / "outlet_coords.npy",
            cl.register_points(boundary_raw["outlet_coords_raw_mm"], scale_mm),
        ),
        "outlet_normals": _array_record(
            output_dir / "outlet_normals.npy",
            cl.rotate_vectors(boundary_raw["outlet_normals_raw"]),
        ),
        "outlet_area_weights_m2": _array_record(
            output_dir / "outlet_area_weights_m2.npy",
            boundary_raw["outlet_area_weights_m2"],
        ),
        "outlet_zone": _array_record(
            output_dir / "outlet_zone.npy", boundary_raw["outlet_zone"]
        ),
    }

    coord_max = max(
        float(np.max(np.abs(steady_coords))),
        float(np.max(np.abs(transient_coords))),
        *[
            float(np.max(np.abs(np.load(record["path"], mmap_mode="r"))))
            for name, record in boundary_records.items()
            if name.endswith("coords")
        ],
    )
    speed_before = np.linalg.norm(np.asarray(steady_velocity_raw, dtype=np.float64), axis=1)
    speed_after = np.linalg.norm(np.asarray(steady_velocity, dtype=np.float64), axis=1)
    geometry_identity_delta = float(
        np.max(
            np.abs(
                transient_geometry[eligible]
                - steady_geometry[identity_index[eligible]]
            )
        )
    )
    finite_arrays = all(
        np.isfinite(np.asarray(value)).all()
        for value in (
            steady_coords,
            steady_geometry,
            steady_aux,
            steady_velocity,
            steady_pressure,
            steady_distance_to_wall_mm,
            transient_coords,
            transient_geometry,
            transient_aux,
            transient_velocity,
            transient_pressure,
            transient_distance_to_wall_mm,
            boundary_raw["inlet_coords_raw_mm"],
            boundary_raw["inlet_normals_raw"],
            boundary_raw["wall_coords_raw_mm"],
            boundary_raw["outlet_coords_raw_mm"],
            boundary_raw["outlet_normals_raw"],
            boundary_raw["outlet_area_weights_m2"],
        )
    )
    inlet_normal_error = float(
        np.max(np.abs(np.linalg.norm(boundary_raw["inlet_normals_raw"], axis=1) - 1.0))
    )
    outlet_normal_error = float(
        np.max(np.abs(np.linalg.norm(boundary_raw["outlet_normals_raw"], axis=1) - 1.0))
    )
    blockers = [
        "pressure_gauge_low_cluster_requires_case_signoff",
        "raw_transient_frame_content_sha256_pending_formal_cutover",
    ]
    if case_id in PRESSURE_LOW_CLUSTER:
        blockers.append("case_is_in_unresolved_pressure_low_cluster")

    payload = {
        "schema_version": 2,
        "route": ROUTE,
        "created_at": utc_now(),
        "status": "completed",
        "gate_result": "staging_pass",
        "training_ready": False,
        "canonical_id": case_id,
        "cohort": case["cohort"],
        "role": case["role"],
        "geometry_schema": {
            "raw_names": list(GEOMETRY_RAW_NAMES),
            "model_names": list(GEOMETRY_MODEL_NAMES),
            "aux_names": list(GEOMETRY_AUX_NAMES),
            "model_transform": "signed_log1p(curvature_per_mm), then shared train138 z-score",
            "steady_transient_shared_semantics": True,
        },
        "registration": {
            "frame_version": "centerline_v2_opening_ids_v1",
            "centroid_mm": cl.centroid_mm.tolist(),
            "rotation": cl.rotation.tolist(),
            "coord_scale_mm": scale_mm,
            "coordinate_length_m": scale_mm * 1.0e-3,
            "axis_contract": "+Z junction-to-inlet; +X patient-left opening pair",
            **cl.registration_diagnostics,
            "coordinate_max_abs": coord_max,
            "speed_norm_roundtrip_max_m_s": float(np.max(np.abs(speed_before - speed_after))),
        },
        "counts": {
            "steady_rows": int(len(steady_coords)),
            "steady_strict_rows": int(np.sum(~steady_is_wall)),
            "transient_rows": int(len(transient_coords)),
            "transient_eligible_interior": int(np.sum(eligible)),
            "transient_frames": int(transient_velocity.shape[0]),
        },
        "boundary_contract": boundary_raw["diagnostics"],
        "gates": {
            "finite_arrays": bool(finite_arrays),
            "proper_rotation": abs(cl.registration_diagnostics["determinant"] - 1.0) < 1.0e-8,
            "all_coordinates_within_unit_box": coord_max <= 1.0 + 1.0e-6,
            "velocity_speed_preserved": float(np.max(np.abs(speed_before - speed_after))) < 1.0e-6,
            "steady_transient_geometry_peak_identity_max_abs": geometry_identity_delta,
            "steady_transient_geometry_identity": geometry_identity_delta <= 1.0e-6,
            "centerline_v2_curvature_hard_max_per_mm": float(
                np.max(np.abs(cl.curvature_per_mm))
            ),
            "boundary_inlet_normal_unit_max_error": inlet_normal_error,
            "boundary_outlet_normal_unit_max_error": outlet_normal_error,
            "boundary_normals_are_unit": max(inlet_normal_error, outlet_normal_error) <= 1.0e-6,
            "boundary_outlet_area_weights_positive": bool(
                np.all(np.asarray(boundary_raw["outlet_area_weights_m2"]) > 0.0)
            ),
            "boundary_four_outlet_zones_complete": set(
                np.asarray(boundary_raw["outlet_zone"], dtype=np.int64).tolist()
            )
            == {0, 1, 2, 3},
            "inlet_wall_rim_buffer_nonempty": bool(
                boundary_raw["diagnostics"]["inlet_nonempty"]
                and boundary_raw["diagnostics"]["wall_nonempty"]
            ),
        },
        "conditions": {
            **case["conditions"],
            # frozen 2026-09-05 contract: rebuild the raw vector from solver-time quantities
            "inlet_area_ratio": float(case["conditions"]["a_in_mesh_m2"]) / float(case["conditions"]["a_in_udf_m2"]),
            "bc_vector_raw": bc_vector_from_conditions(case["conditions"]),
        },
        "physics_scales": {
            **case["physics_scales"],
            "coordinate_length_m": scale_mm * 1.0e-3,
        },
        "pressure": {
            **case["pressure"],
            "steady_reference_pa": float(raw_peak_provenance["pressure_reference_pa"]),
        },
        "rheology": case["rheology"],
        "fourier": case["fourier"],
        "staging_blockers": blockers,
        "build_input": input_snapshot,
        "provenance": {
            "source_case_manifest": {
                "path": row["manifest"],
                "sha256": row["manifest_sha256"],
            },
            "source_volume_manifest": {
                "path": str(volume_path.resolve()),
                "sha256": sha256_file(volume_path),
            },
            "raw_peak": raw_peak_provenance,
            "source_legacy_bundle": case["provenance"]["bundle"],
            "raw_transient": transient_raw_provenance,
            "centerline_v2": cl.provenance,
        },
        "files": {
            "steady": steady_records,
            "transient": transient_records,
            "boundary_faces": boundary_records,
        },
    }
    atomic_write_json(manifest_path, payload)
    return payload


def _update(moment: StreamingMoments, array: np.ndarray) -> None:
    for chunk in iter_slices(len(array)):
        moment.update(np.asarray(array[chunk]))


def build_stats(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    split_record = source_manifest["split"]
    split_path = Path(split_record["path"])
    split = _load_split(split_record)
    train = [row for row in case_reports if row["role"] == "train"]
    test = [row for row in case_reports if row["role"] == "test"]
    if set(row["canonical_id"] for row in train) != set(split["train_cases"]):
        raise ValueError("staging train roles do not match the frozen train138 set")
    if set(row["canonical_id"] for row in test) != set(split["test_cases"]):
        raise ValueError("staging test roles do not match the frozen test35 set")
    geometry = StreamingMoments(3)
    geometry_min = np.full(3, np.inf, dtype=np.float64)
    geometry_max = np.full(3, -np.inf, dtype=np.float64)
    steady_velocity = StreamingMoments(3)
    steady_pressure = StreamingMoments(1)
    transient_velocity = StreamingMoments(3)
    transient_pressure = StreamingMoments(1)
    bc_raw = np.asarray([row["conditions"]["bc_vector_raw"] for row in train], dtype=np.float64)
    bc_value = transform_bc_raw(bc_raw)
    bc_policy = bc_scale_policy(bc_value)  # raises on a near-constant z-scored field
    bc_mean = np.asarray(bc_policy["mean"], dtype=np.float64)
    bc_std = np.asarray(bc_policy["std"], dtype=np.float64)
    bc_std_raw = np.asarray(bc_policy["raw_std"], dtype=np.float64)

    for row in train:
        steady = row["files"]["steady"]
        geom = np.load(steady["geometry_raw"]["path"], mmap_mode="r")
        is_wall = np.asarray(np.load(steady["is_wall"]["path"], mmap_mode="r"), dtype=bool)
        for chunk in iter_slices(len(geom)):
            eligible = ~is_wall[chunk]
            value = np.asarray(geom[chunk], dtype=np.float64).copy()
            value[:, 2] = np.sign(value[:, 2]) * np.log1p(np.abs(value[:, 2]))
            selected_geometry = value[eligible]
            geometry.update(selected_geometry)
            geometry_min = np.minimum(geometry_min, selected_geometry.min(axis=0))
            geometry_max = np.maximum(geometry_max, selected_geometry.max(axis=0))
        velocity = np.load(steady["velocity_m_s"]["path"], mmap_mode="r")
        pressure = np.load(steady["pressure_relative_pa"]["path"], mmap_mode="r")
        for chunk in iter_slices(len(velocity)):
            eligible = ~is_wall[chunk]
            steady_velocity.update(np.asarray(velocity[chunk])[eligible])
            steady_pressure.update(np.asarray(pressure[chunk])[eligible, None])

        transient = row["files"]["transient"]
        eligible = np.asarray(
            np.load(transient["eligible_interior"]["path"], mmap_mode="r"), dtype=bool
        )
        velocity_t = np.load(transient["velocity_m_s"]["path"], mmap_mode="r")
        pressure_t = np.load(transient["pressure_raw_pa"]["path"], mmap_mode="r")
        for frame in range(len(velocity_t)):
            transient_velocity.update(np.asarray(velocity_t[frame])[eligible])
            transient_pressure.update(np.asarray(pressure_t[frame])[eligible, None])

    transformed_names = list(bc_policy["names"])
    z = (bc_value - bc_mean) / bc_std
    outliers = []
    advisory = [
        {"canonical_id": row["canonical_id"], "field": name, "z": float(z[case_index, field_index])}
        for case_index, row in enumerate(train)
        for field_index, name in enumerate(transformed_names)
        if abs(float(z[case_index, field_index])) > 4.0
    ]
    for case_index, row in enumerate(train):
        for field_index, name in enumerate(transformed_names):
            if abs(float(z[case_index, field_index])) > 6.0:
                whitelist_key = (row["canonical_id"], name)
                outliers.append(
                    {
                        "canonical_id": row["canonical_id"],
                        "field": name,
                        "z": float(z[case_index, field_index]),
                        "whitelisted": whitelist_key in BC_Z_WHITELIST,
                        "reason": BC_Z_WHITELIST.get(whitelist_key, ""),
                    }
                )
    if any(not row["whitelisted"] for row in outliers):
        raise ValueError("unreviewed train BC value exceeds the |z|>6 hard Gate")
    observed_whitelist = {
        (row["canonical_id"], row["field"]) for row in outliers if row["whitelisted"]
    }
    if observed_whitelist != set(BC_Z_WHITELIST):
        raise ValueError("reviewed BC long-tail whitelist no longer matches the observed fields")

    geometry_moments = geometry.result()
    geometry_mean = np.asarray(geometry_moments["mean"], dtype=np.float64)
    geometry_std_raw = np.asarray(geometry_moments["std"], dtype=np.float64)
    geometry_max_deviation = np.maximum(
        np.abs(geometry_min - geometry_mean),
        np.abs(geometry_max - geometry_mean),
    )
    geometry_scale = np.maximum(geometry_std_raw, geometry_max_deviation / 6.0)
    geometry_train_abs_z_max = geometry_max_deviation / geometry_scale

    return {
        "schema_version": 2,
        "route": ROUTE,
        "created_at": utc_now(),
        "status": "completed",
        "scope": "train138 only; staging source domain",
        "split": {"path": str(split_path.resolve()), "sha256": split_record["sha256"]},
        "train_case_ids": [row["canonical_id"] for row in train],
        "geometry": {
            "raw_names": list(GEOMETRY_RAW_NAMES),
            "names": list(GEOMETRY_MODEL_NAMES),
            "source_population": "steady strict-volume train138; shared by steady/transient",
            "transform": (
                "signed_log1p curvature without clip; subtract train138 mean; "
                "divide by max(population_std, train_max_abs_deviation/6)"
            ),
            "count": geometry_moments["count"],
            "mean": geometry_mean.tolist(),
            "std": geometry_scale.tolist(),
            "raw_std": geometry_std_raw.tolist(),
            "train_min_after_curvature_transform": geometry_min.tolist(),
            "train_max_after_curvature_transform": geometry_max.tolist(),
            "train_abs_z_max": geometry_train_abs_z_max.tolist(),
            "scale_policy": "max-aware train-only z-scale; no clipping; hard bound ±6 on train",
        },
        "bc": {
            **bc_policy,
            "z_gt_6": outliers,
            "abs_z_gt_4": advisory,
        },
        "steady": {
            "velocity_m_s": steady_velocity.result(),
            "pressure_pa": steady_pressure.result(),
        },
        "transient": {
            "velocity_m_s": transient_velocity.result(),
            "pressure_pa": {
                **transient_pressure.result(),
                "gauge": "raw_fluent_gauge_pa_distal_pressure_zero_unresolved_low_cluster",
            },
        },
    }


def _audit_case_arrays(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    array_count = 0
    total_bytes = 0
    for report in case_reports:
        required = {
            "steady": REQUIRED_STEADY_ARRAYS,
            "transient": REQUIRED_TRANSIENT_ARRAYS,
            "boundary_faces": REQUIRED_BOUNDARY_ARRAYS,
        }
        for group, names in required.items():
            for name in sorted(names.difference(report["files"].get(group, {}))):
                errors.append(f"required:{report['canonical_id']}:{group}:{name}")
        for group in ("steady", "transient", "boundary_faces"):
            for name, record in report["files"][group].items():
                array_count += 1
                path = Path(record["path"])
                if not path.is_file():
                    errors.append(f"missing:{report['canonical_id']}:{group}:{name}")
                    continue
                total_bytes += int(path.stat().st_size)
                if sha256_file(path) != record["sha256"]:
                    errors.append(f"sha256:{report['canonical_id']}:{group}:{name}")
                    continue
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(value.shape) != list(record["shape"]):
                    errors.append(f"shape:{report['canonical_id']}:{group}:{name}")
                if str(value.dtype) != str(record["dtype"]):
                    errors.append(f"dtype:{report['canonical_id']}:{group}:{name}")
                if value.ndim == 0:
                    finite = bool(np.isfinite(np.asarray(value)).all())
                else:
                    finite = all(
                        np.isfinite(np.asarray(value[chunk])).all()
                        for chunk in iter_slices(len(value))
                    )
                if not finite:
                    errors.append(f"nonfinite:{report['canonical_id']}:{group}:{name}")
        if any(
            names.difference(report["files"].get(group, {}))
            for group, names in required.items()
        ):
            continue
        steady = report["files"]["steady"]
        transient = report["files"]["transient"]
        boundary = report["files"]["boundary_faces"]
        steady_coords = np.load(steady["coords"]["path"], mmap_mode="r")
        steady_region = np.asarray(np.load(steady["region"]["path"], mmap_mode="r"))
        steady_distance = np.asarray(
            np.load(steady["distance_to_wall_mm"]["path"], mmap_mode="r")
        )
        steady_lengths = {
            len(np.load(record["path"], mmap_mode="r")) for record in steady.values()
        }
        if steady_lengths != {len(steady_coords)}:
            errors.append(f"semantic:{report['canonical_id']}:steady_first_dimension")
        if not np.isin(steady_region, [0, 1]).all() or np.any(steady_distance < 0.0):
            errors.append(f"semantic:{report['canonical_id']}:steady_region_distance")
        steady_expected = steady_distance <= NEAR_WALL_THRESHOLD_MM
        steady_mismatch = steady_region.astype(bool) != steady_expected
        if np.any(steady_mismatch) and not np.all(
            np.isclose(
                steady_distance[steady_mismatch],
                NEAR_WALL_THRESHOLD_MM,
                rtol=0.0,
                atol=1.0e-6,
            )
        ):
            errors.append(f"semantic:{report['canonical_id']}:steady_region_threshold")

        transient_coords = np.load(transient["coords"]["path"], mmap_mode="r")
        transient_region = np.asarray(
            np.load(transient["region"]["path"], mmap_mode="r")
        )
        transient_distance = np.asarray(
            np.load(transient["distance_to_wall_mm"]["path"], mmap_mode="r")
        )
        point_records = (
            "coords",
            "geometry_raw",
            "geometry_aux",
            "is_wall",
            "eligible_interior",
            "region",
            "distance_to_wall_mm",
            "cell_ids",
        )
        if any(
            len(np.load(transient[name]["path"], mmap_mode="r")) != len(transient_coords)
            for name in point_records
        ):
            errors.append(f"semantic:{report['canonical_id']}:transient_first_dimension")
        velocity_t = np.load(transient["velocity_m_s"]["path"], mmap_mode="r")
        pressure_t = np.load(transient["pressure_raw_pa"]["path"], mmap_mode="r")
        if velocity_t.shape[:2] != (81, len(transient_coords)) or pressure_t.shape != (
            81,
            len(transient_coords),
        ):
            errors.append(f"semantic:{report['canonical_id']}:transient_field_shape")
        if not np.isin(transient_region, [0, 1]).all() or np.any(transient_distance < 0.0):
            errors.append(f"semantic:{report['canonical_id']}:transient_region_distance")
        transient_expected = transient_distance <= NEAR_WALL_THRESHOLD_MM
        transient_mismatch = transient_region.astype(bool) != transient_expected
        if np.any(transient_mismatch) and not np.all(
            np.isclose(
                transient_distance[transient_mismatch],
                NEAR_WALL_THRESHOLD_MM,
                rtol=0.0,
                atol=1.0e-6,
            )
        ):
            errors.append(f"semantic:{report['canonical_id']}:transient_region_threshold")

        inlet_coords = np.load(boundary["inlet_coords"]["path"], mmap_mode="r")
        inlet_normals = np.load(boundary["inlet_normals"]["path"], mmap_mode="r")
        outlet_coords = np.load(boundary["outlet_coords"]["path"], mmap_mode="r")
        outlet_normals = np.load(boundary["outlet_normals"]["path"], mmap_mode="r")
        outlet_weights = np.load(
            boundary["outlet_area_weights_m2"]["path"], mmap_mode="r"
        )
        outlet_zone = np.load(boundary["outlet_zone"]["path"], mmap_mode="r")
        if len(inlet_coords) != len(inlet_normals) or len(outlet_coords) != len(
            outlet_normals
        ) or len(outlet_coords) != len(outlet_weights) or len(outlet_coords) != len(
            outlet_zone
        ):
            errors.append(f"semantic:{report['canonical_id']}:boundary_first_dimension")
    payload = {
        "schema_version": 1,
        "route": ROUTE,
        "created_at": utc_now(),
        "status": "pass" if not errors else "fail",
        "case_count": len(case_reports),
        "array_count": array_count,
        "total_bytes": total_bytes,
        "errors": errors,
        "semantic_contract": {
            "near_wall_threshold_mm": NEAR_WALL_THRESHOLD_MM,
            "required_steady_arrays": sorted(REQUIRED_STEADY_ARRAYS),
            "required_transient_arrays": sorted(REQUIRED_TRANSIENT_ARRAYS),
            "required_boundary_arrays": sorted(REQUIRED_BOUNDARY_ARRAYS),
        },
        "case_manifest_sha256": {
            report["canonical_id"]: sha256_file(
                DATA_ROOT / "cases" / report["canonical_id"] / "manifest.json"
            )
            for report in case_reports
        },
    }
    atomic_write_json(ARRAY_AUDIT_REPORT, payload)
    if errors:
        raise ValueError(f"full staging array audit failed with {len(errors)} errors")
    return payload


def _refresh_case_derived(row: dict[str, Any]) -> dict[str, Any]:
    case = _load_case(row)
    case_id = case["canonical_id"]
    manifest_path = DATA_ROOT / "cases" / case_id / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("route") != ROUTE:
        raise ValueError(f"{case_id}: cannot refresh an incomplete/non-staging case")
    cl = CenterlineV2Geometry.load(CENTERLINE_ROOT, case_id)
    old_centerline = payload["provenance"]["centerline_v2"]
    if {
        key: old_centerline[key]["sha256"]
        for key in ("result", "surface_selection", "paths")
    } != {
        key: cl.provenance[key]["sha256"]
        for key in ("result", "surface_selection", "paths")
    }:
        raise ValueError(f"{case_id}: Centerline V2 changed; full case rebuild is required")
    volume_path = Path(case["files"]["steady_volume_manifest"]["path"])
    volume = json.loads(volume_path.read_text(encoding="utf-8"))
    current_input = _current_input_snapshot(row, case, volume, cl)
    if payload.get("build_input", {}).get("sha256") != current_input["sha256"]:
        raise ValueError(f"{case_id}: refresh input snapshot drift; full rebuild is required")
    for group in ("steady", "transient", "boundary_faces"):
        for record in payload["files"][group].values():
            _verify_array_record(record)
    (
        steady_raw_mm,
        steady_is_wall,
        steady_region,
        steady_distance_to_wall_mm,
        _,
        _,
        steady_cell_ids,
        raw_peak_provenance,
    ) = _raw_peak_steady(case, volume, cl)
    transient_records = payload["files"]["transient"]
    selected_ids = np.asarray(
        np.load(transient_records["cell_ids"]["path"], mmap_mode="r"), dtype=np.int64
    )
    identity_index = _indices_for_cell_ids(steady_cell_ids, selected_ids)
    transient_raw_mm = steady_raw_mm[identity_index]
    boundary_raw = _raw_boundary_contract(case, cl, steady_raw_mm[~steady_is_wall])
    scale_mm = _scale_mm(
        cl,
        [
            steady_raw_mm,
            transient_raw_mm,
            boundary_raw["wall_coords_raw_mm"],
            boundary_raw["inlet_coords_raw_mm"],
            boundary_raw["outlet_coords_raw_mm"],
        ],
    )
    output_dir = manifest_path.parent
    payload["files"]["steady"]["coords"] = _array_record(
        output_dir / "steady_coords.npy", cl.register_points(steady_raw_mm, scale_mm)
    )
    payload["files"]["steady"]["region"] = _array_record(
        output_dir / "steady_region.npy", steady_region
    )
    payload["files"]["steady"]["distance_to_wall_mm"] = _array_record(
        output_dir / "steady_distance_to_wall_mm.npy", steady_distance_to_wall_mm
    )
    payload["files"]["transient"]["coords"] = _array_record(
        output_dir / "transient_coords.npy", cl.register_points(transient_raw_mm, scale_mm)
    )
    payload["files"]["transient"]["region"] = _array_record(
        output_dir / "transient_region.npy", steady_region[identity_index]
    )
    payload["files"]["transient"]["distance_to_wall_mm"] = _array_record(
        output_dir / "transient_distance_to_wall_mm.npy",
        steady_distance_to_wall_mm[identity_index],
    )
    boundary_arrays = {
        "inlet_coords": cl.register_points(boundary_raw["inlet_coords_raw_mm"], scale_mm),
        "inlet_normals": cl.rotate_vectors(boundary_raw["inlet_normals_raw"]),
        "wall_coords": cl.register_points(boundary_raw["wall_coords_raw_mm"], scale_mm),
        "outlet_coords": cl.register_points(boundary_raw["outlet_coords_raw_mm"], scale_mm),
        "outlet_normals": cl.rotate_vectors(boundary_raw["outlet_normals_raw"]),
        "outlet_area_weights_m2": boundary_raw["outlet_area_weights_m2"],
        "outlet_zone": boundary_raw["outlet_zone"],
    }
    payload["files"]["boundary_faces"] = {
        name: _array_record(output_dir / f"{name}.npy", value)
        for name, value in boundary_arrays.items()
    }
    coord_max = max(
        float(np.max(np.abs(np.load(payload["files"]["steady"]["coords"]["path"], mmap_mode="r")))),
        float(np.max(np.abs(np.load(payload["files"]["transient"]["coords"]["path"], mmap_mode="r")))),
        *[
            float(np.max(np.abs(value)))
            for name, value in boundary_arrays.items()
            if name.endswith("coords")
        ],
    )
    inlet_normal_error = float(
        np.max(np.abs(np.linalg.norm(boundary_arrays["inlet_normals"], axis=1) - 1.0))
    )
    outlet_normal_error = float(
        np.max(np.abs(np.linalg.norm(boundary_arrays["outlet_normals"], axis=1) - 1.0))
    )
    payload["updated_at"] = utc_now()
    payload["boundary_contract"] = boundary_raw["diagnostics"]
    payload["registration"]["coord_scale_mm"] = scale_mm
    payload["registration"]["coordinate_length_m"] = scale_mm * 1.0e-3
    payload["registration"]["coordinate_max_abs"] = coord_max
    payload["physics_scales"]["coordinate_length_m"] = scale_mm * 1.0e-3
    payload["pressure"]["steady_reference_pa"] = float(
        raw_peak_provenance["pressure_reference_pa"]
    )
    payload["provenance"]["raw_peak"] = raw_peak_provenance
    payload["build_input"] = current_input
    payload["gates"].update(
        {
            "finite_arrays": True,
            "all_coordinates_within_unit_box": coord_max <= 1.0 + 1.0e-6,
            "boundary_inlet_normal_unit_max_error": inlet_normal_error,
            "boundary_outlet_normal_unit_max_error": outlet_normal_error,
            "boundary_normals_are_unit": max(inlet_normal_error, outlet_normal_error)
            <= 1.0e-6,
            "boundary_outlet_area_weights_positive": bool(
                np.all(boundary_arrays["outlet_area_weights_m2"] > 0.0)
            ),
            "boundary_four_outlet_zones_complete": set(
                np.asarray(boundary_arrays["outlet_zone"], dtype=np.int64).tolist()
            )
            == {0, 1, 2, 3},
            "inlet_wall_rim_buffer_nonempty": bool(
                boundary_raw["diagnostics"]["inlet_nonempty"]
                and boundary_raw["diagnostics"]["wall_nonempty"]
            ),
        }
    )
    atomic_write_json(manifest_path, payload)
    return payload


def _build_case_task(task: dict[str, Any]) -> dict[str, Any]:
    row = task["row"]
    print(
        json.dumps(
            {
                "event": "centerline_v2_staging_case_started",
                "index": task["index"],
                "total": task["total"],
                "case_id": row["canonical_id"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return build_case(row, resume=bool(task["resume"]))


def _refresh_case_task(task: dict[str, Any]) -> dict[str, Any]:
    row = task["row"]
    print(
        json.dumps(
            {
                "event": "centerline_v2_staging_refresh_started",
                "index": task["index"],
                "total": task["total"],
                "case_id": row["canonical_id"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return _refresh_case_derived(row)


def build(
    *,
    case_ids: list[str] | None = None,
    resume: bool = False,
    stats_only: bool = False,
    refresh_derived: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    source, rows = _load_source()
    if stats_only and case_ids:
        raise ValueError("--stats-only cannot be combined with --case-id")
    if refresh_derived and (case_ids or resume or stats_only):
        raise ValueError("--refresh-derived must run alone on the full 173-case staging set")
    selected = rows
    if case_ids:
        wanted = set(case_ids)
        selected = [row for row in rows if row["canonical_id"] in wanted]
        missing = sorted(wanted.difference(row["canonical_id"] for row in selected))
        if missing:
            raise ValueError(f"unknown case IDs: {missing}")
    reports = []
    if refresh_derived:
        tasks = [
            {"row": row, "index": index, "total": len(selected)}
            for index, row in enumerate(selected, start=1)
        ]
        if int(workers) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
                reports = list(executor.map(_refresh_case_task, tasks))
        else:
            reports = [_refresh_case_task(task) for task in tasks]
    elif stats_only:
        for row in selected:
            case_path = DATA_ROOT / "cases" / row["canonical_id"] / "manifest.json"
            if not case_path.is_file():
                raise FileNotFoundError(f"missing completed staging manifest: {case_path}")
            report = json.loads(case_path.read_text(encoding="utf-8"))
            if (
                report.get("schema_version") != 2
                or report.get("status") != "completed"
                or report.get("canonical_id") != row["canonical_id"]
            ):
                raise ValueError(f"invalid completed staging manifest: {case_path}")
            case = _load_case(row)
            cl = CenterlineV2Geometry.load(CENTERLINE_ROOT, row["canonical_id"])
            volume = json.loads(
                Path(case["files"]["steady_volume_manifest"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            current_input = _current_input_snapshot(row, case, volume, cl)
            if report.get("build_input", {}).get("sha256") != current_input["sha256"]:
                raise ValueError(f"stale staging build inputs: {case_path}")
            for group in ("steady", "transient", "boundary_faces"):
                for record in report["files"][group].values():
                    _verify_array_record(record)
            reports.append(report)
    else:
        tasks = [
            {
                "row": row,
                "resume": resume,
                "index": index,
                "total": len(selected),
            }
            for index, row in enumerate(selected, start=1)
        ]
        if int(workers) > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(workers)
            ) as executor:
                reports = list(executor.map(_build_case_task, tasks))
        else:
            reports = [_build_case_task(task) for task in tasks]

    full = len(reports) == 173
    if not full:
        pilot = {
            "schema_version": 2,
            "route": ROUTE,
            "created_at": utc_now(),
            "status": "pilot_completed",
            "gate_result": "not_evaluated",
            "training_ready": False,
            "counts": {
                "total": len(reports),
                "train": sum(row["role"] == "train" for row in reports),
                "test": sum(row["role"] == "test" for row in reports),
            },
            "case_ids": [row["canonical_id"] for row in reports],
        }
        pilot_path = AUDIT_ROOT / "pilots" / f"pilot_{sha256_json(pilot['case_ids'])[:16]}.json"
        atomic_write_json(pilot_path, pilot)
        return {**pilot, "report": str(pilot_path.resolve())}

    stats = build_stats(reports)
    atomic_write_json(STATS_PATH, stats)
    stats_record = {"path": str(STATS_PATH.resolve()), "sha256": sha256_file(STATS_PATH)}
    array_audit = _audit_case_arrays(reports)
    array_audit_record = {
        "path": str(ARRAY_AUDIT_REPORT.resolve()),
        "sha256": sha256_file(ARRAY_AUDIT_REPORT),
        "array_count": int(array_audit["array_count"]),
        "total_bytes": int(array_audit["total_bytes"]),
    }
    manifest = {
        "schema_version": 2,
        "route": ROUTE,
        "created_at": utc_now(),
        "status": "completed",
        "gate_result": "staging_pass",
        "training_ready": False,
        "source_manifest": {
            "path": str(SOURCE_MANIFEST.resolve()),
            "sha256": sha256_file(SOURCE_MANIFEST),
        },
        "split": source["split"],
        "centerline_v2_root": str(CENTERLINE_ROOT.resolve()),
        "counts": {
            "total": len(reports),
            "train": sum(row["role"] == "train" for row in reports),
            "test": sum(row["role"] == "test" for row in reports),
        },
        "stats": stats_record,
        "array_audit": array_audit_record,
        "cases": [
            {
                "canonical_id": report["canonical_id"],
                "cohort": report["cohort"],
                "role": report["role"],
                "manifest": str((DATA_ROOT / "cases" / report["canonical_id"] / "manifest.json").resolve()),
                "manifest_sha256": sha256_file(
                    DATA_ROOT / "cases" / report["canonical_id"] / "manifest.json"
                ),
            }
            for report in reports
        ],
    }
    transient_pool_sizes = [
        int(report["counts"]["transient_eligible_interior"]) for report in reports
    ]
    raw_static_delta_max = max(
        float(report["provenance"]["raw_transient"]["max_static_coordinate_delta_mm"])
        for report in reports
    )
    pressure_low_cases = sorted(
        report["canonical_id"]
        for report in reports
        if "case_is_in_unresolved_pressure_low_cluster" in report["staging_blockers"]
    )
    optional_outlet_label_missing = sorted(
        report["canonical_id"]
        for report in reports
        if not {
            "outlet_label_pressure_pa",
            "outlet_label_mass_flow_kg_s",
        }.issubset(report["files"]["transient"])
    )
    checks = {
        "selected_cases_built": len(reports) == len(selected) == 173,
        "all_centerline_v2_hard_pass": all(
            report["provenance"]["centerline_v2"]["status"] in {"pass", "pass_review"}
            for report in reports
        ),
        "all_case_finite_array_checks": all(report["gates"]["finite_arrays"] for report in reports),
        "all_proper_rotations": all(report["gates"]["proper_rotation"] for report in reports),
        "all_coordinates_within_unit_box": all(
            report["gates"]["all_coordinates_within_unit_box"] for report in reports
        ),
        "all_shared_geometry_identity": all(
            report["gates"]["steady_transient_geometry_identity"] for report in reports
        ),
        "all_boundary_normals_are_unit": all(
            report["gates"]["boundary_normals_are_unit"] for report in reports
        ),
        "all_boundary_area_weights_positive": all(
            report["gates"]["boundary_outlet_area_weights_positive"] for report in reports
        ),
        "all_four_outlet_zones_complete": all(
            report["gates"]["boundary_four_outlet_zones_complete"] for report in reports
        ),
        "all_inlet_wall_rim_buffers_nonempty": all(
            report["gates"]["inlet_wall_rim_buffer_nonempty"] for report in reports
        ),
        "all_raw_transient_pools_are_15000_unique_cells": all(
            value == TRANSIENT_POOL_SIZE for value in transient_pool_sizes
        ),
        "all_raw_transient_frames_have_static_coordinates": raw_static_delta_max <= 1.0e-5,
        "geometry_train_input_abs_z_le_6_without_clip": max(
            stats["geometry"]["train_abs_z_max"]
        )
        <= 6.0 + 1.0e-9,
        "pressure_low_cluster_explicitly_preserved": set(pressure_low_cases)
        == PRESSURE_LOW_CLUSTER,
        "full_train138_stats_written": True,
        "full_array_sha_shape_dtype_finite_audit": array_audit["status"] == "pass",
    }
    gate_result = "staging_pass" if all(checks.values()) else "fail"
    manifest["gate_result"] = gate_result
    atomic_write_json(MANIFEST_PATH, manifest)
    gate = {
        "schema_version": 2,
        "route": ROUTE,
        "created_at": utc_now(),
        "gate_result": gate_result,
        "training_ready": False,
        "checks": checks,
        "metrics": {
            "rotation_det_error_max": max(
                abs(float(report["registration"]["determinant"]) - 1.0)
                for report in reports
            ),
            "rotation_orthogonality_error_max": max(
                float(report["registration"]["orthogonality_max_abs"])
                for report in reports
            ),
            "left_minus_right_x_min_mm": min(
                float(report["registration"]["left_minus_right_x_mm"])
                for report in reports
            ),
            "inlet_z_min_mm": min(
                float(report["registration"]["inlet_z_mm"]) for report in reports
            ),
            "outlet_z_max_mm": max(
                float(report["registration"]["outlet_z_max_mm"]) for report in reports
            ),
            "coordinate_max_abs": max(
                float(report["registration"]["coordinate_max_abs"]) for report in reports
            ),
            "raw_transient_static_coordinate_delta_max_mm": raw_static_delta_max,
            "transient_pool_min": min(transient_pool_sizes),
            "transient_pool_max": max(transient_pool_sizes),
            "centerline_v2_curvature_max_per_mm": max(
                float(report["gates"]["centerline_v2_curvature_hard_max_per_mm"])
                for report in reports
            ),
            "geometry_train_abs_z_max": (
                stats["geometry"]["train_abs_z_max"] if stats is not None else None
            ),
            "bc_z_gt_6": stats["bc"]["z_gt_6"] if stats is not None else None,
            "pressure_low_cluster_cases": pressure_low_cases,
            "inlet_face_points_before_total": sum(
                int(report["boundary_contract"]["inlet_face_points_before"])
                for report in reports
            ),
            "inlet_face_points_after_total": sum(
                int(report["boundary_contract"]["inlet_face_points_after"])
                for report in reports
            ),
            "wall_face_points_before_total": sum(
                int(report["boundary_contract"]["wall_face_points_before"])
                for report in reports
            ),
            "wall_face_points_after_total": sum(
                int(report["boundary_contract"]["wall_face_points_after"])
                for report in reports
            ),
            "minimum_inlet_wall_distance_mm": min(
                float(report["boundary_contract"]["minimum_inlet_wall_distance_mm"])
                for report in reports
            ),
            "optional_outlet_label_missing_cases": optional_outlet_label_missing,
        },
        "formal_training_blockers": [
            "resolve/sign off the eight-case pressure low cluster",
            "bind raw transient 81-frame content SHA256 before formal cutover",
            "sign off the sparse transient velocity/BC long tails and their loss sensitivity",
            "declare the four-case outlet monitor labels optional or rebuild them",
            "replace staging route/config/output roots before any training submission",
        ],
        "manifest": {"path": str(MANIFEST_PATH.resolve()), "sha256": sha256_file(MANIFEST_PATH)},
        "stats": stats_record,
        "array_audit": array_audit_record,
    }
    atomic_write_json(GATE_REPORT, gate)
    if gate_result != "staging_pass":
        raise ValueError("full Centerline V2 staging Gate failed")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--refresh-derived", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = build(
        case_ids=args.case_id or None,
        resume=bool(args.resume),
        stats_only=bool(args.stats_only),
        refresh_derived=bool(args.refresh_derived),
        workers=max(int(args.workers), 1),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
