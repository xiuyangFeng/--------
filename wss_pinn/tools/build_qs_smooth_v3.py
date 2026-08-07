"""Build and audit the quasi-steady smooth-field V3 boundary assets.

The V1 volume sidecars remain immutable.  This tool derives only:

* a deterministic train123/val15/test35 split;
* inlet/outlet coordinates and peak targets recovered from the original
  Fluent ``.cas.gz`` mesh and ``bc_metadata.json``;
* train123-only field statistics; and
* aggregate manifests and a machine-readable hard Gate report.

Legacy Fluent case files mix text headers with binary payloads.  The parser
below intentionally reads only node coordinates and physical boundary face
zones (wall, pressure-outlet, velocity-inlet); large interior face sections
are skipped without materialising them.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    guard_write_path,
    sha256_file,
    utc_now,
)


ROUTE = "volume_uvwp_peak_qs_smooth_v3"
SOURCE_SPLIT = (
    ROOT
    / "wss_pinn/configs/splits/"
    "split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_"
    "train138_test35_exclude_SHI_YUN_XI_v1.json"
)
SOURCE_MANIFEST = (
    ROOT / "data_wss_pinn/volume_uvwp_peak_v1_train138_test35/manifest.json"
)
OUTPUT_ROOT = (
    ROOT / "data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35"
)
SPLIT_OUTPUT = (
    ROOT
    / "wss_pinn/configs/splits/"
    "split_WSS_PINN_AG_AAA_ILO_qs_smooth_v3_train123_val15_test35_s1234.json"
)
AUDIT_ROOT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_boundary"

VAL_QUOTAS = {
    "AG": 7,
    "AAA/ruputer": 2,
    "AAA/unruputer": 3,
    "ILO/0": 2,
    "ILO/1": 1,
}
PROTECTED_RELATED_GROUP = {
    "AG/slow/HOU_SHEN_QIAN",
    "AG/slow/KANG_XI_MING",
}
EXPECTED_OUTLETS = ("out-le", "out-li", "out-re", "out-ri")
OUTLET_TO_BC_INDEX = {"out-le": 1, "out-li": 2, "out-re": 3, "out-ri": 4}

HEADER_RE = re.compile(
    rb"^\(\s*(3010|2010|3013|2013|39)\s+\(([^)]*)\)"
)
ZONE_RE = re.compile(rb"^\(39 \(([0-9]+)\s+(\S+)\s+(\S+)")


def _stratum(case_id: str) -> str:
    if case_id.startswith("AG/"):
        return "AG"
    if case_id.startswith("AAA/ruputer/"):
        return "AAA/ruputer"
    if case_id.startswith("AAA/unruputer/"):
        return "AAA/unruputer"
    if case_id.startswith("ILO/"):
        return "ILO/1" if "-1/" in case_id else "ILO/0"
    raise ValueError(f"unknown case stratum: {case_id}")


def derive_split(seed: int = 1234) -> dict[str, Any]:
    parent = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    train138 = list(parent["train_cases"])
    validation: list[str] = []
    for stratum, quota in VAL_QUOTAS.items():
        candidates = [
            case_id
            for case_id in train138
            if _stratum(case_id) == stratum
            and case_id not in PROTECTED_RELATED_GROUP
        ]
        candidates.sort(
            key=lambda case_id: hashlib.sha256(
                f"{seed}|{stratum}|{case_id}".encode("utf-8")
            ).hexdigest()
        )
        validation.extend(candidates[: int(quota)])
    validation = sorted(validation)
    validation_set = set(validation)
    train123 = [case_id for case_id in train138 if case_id not in validation_set]
    test35 = list(parent["test_cases"])
    if len(train123) != 123 or len(validation) != 15 or len(test35) != 35:
        raise ValueError("derived split counts are not 123/15/35")
    if set(train123) & validation_set or set(train123) & set(test35):
        raise ValueError("derived split overlap")
    return {
        "schema_version": 1,
        "split_version": SPLIT_OUTPUT.stem,
        "route": ROUTE,
        "created_at": "2026-08-05T00:00:00+00:00",
        "algorithm": (
            "within frozen train138, protect the known related AG pair, then "
            "rank each stratum by SHA256(seed|stratum|case_id) and take fixed quotas"
        ),
        "seed": int(seed),
        "source_split": str(SOURCE_SPLIT.resolve()),
        "source_split_sha256": sha256_file(SOURCE_SPLIT),
        "test35_unchanged": test35 == parent["test_cases"],
        "validation_quotas": VAL_QUOTAS,
        "protected_related_group": sorted(PROTECTED_RELATED_GROUP),
        "counts": {"train": 123, "val": 15, "test": 35},
        "train_cases": train123,
        "val_cases": validation,
        "test_cases": test35,
    }


def _open_binary_payload(handle) -> None:
    while True:
        value = handle.read(1)
        if not value:
            raise EOFError("missing Fluent binary payload opener")
        if value == b"(":
            return
        if not value.isspace():
            raise ValueError(f"unexpected byte before Fluent payload: {value!r}")


def _finish_binary_section(handle, index: bytes) -> None:
    marker = b"End of Binary Section   " + index + b")"
    while True:
        line = handle.readline()
        if not line:
            raise EOFError(f"missing Fluent section marker {marker!r}")
        if marker in line:
            return


def _read_fluent_boundaries(case_gz: Path) -> dict[str, Any]:
    """Return physical boundary faces and nodes from a legacy Fluent case."""
    with tempfile.NamedTemporaryFile(prefix="qs_v3_", suffix=".cas", delete=False) as tmp:
        temporary = Path(tmp.name)
        with gzip.open(case_gz, "rb") as source:
            shutil.copyfileobj(source, tmp, length=8 * 1024 * 1024)
    nodes = None
    first_node = 1
    face_zones: dict[int, list[np.ndarray]] = {}
    face_cells: dict[int, list[np.ndarray]] = {}
    zone_types: dict[int, int] = {}
    zone_names: dict[int, tuple[str, str]] = {}
    try:
        with temporary.open("rb") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                match = HEADER_RE.match(line)
                if match is None:
                    continue
                index = match.group(1)
                values = match.group(2).split()
                if index in {b"3010", b"2010"}:
                    header = [int(value, 16) for value in values]
                    _, first_node, last_node, _, dimension = header[:5]
                    count = last_node - first_node + 1
                    _open_binary_payload(handle)
                    dtype = np.dtype("<f8" if index == b"3010" else "<f4")
                    payload = handle.read(count * dimension * dtype.itemsize)
                    nodes = np.frombuffer(payload, dtype=dtype).reshape(
                        count, dimension
                    ).copy()
                    _finish_binary_section(handle, index)
                elif index in {b"3013", b"2013"}:
                    header = [int(value, 16) for value in values]
                    zone_id, first_face, last_face, zone_type, element_type = header[:5]
                    face_count = last_face - first_face + 1
                    if zone_type not in {3, 5, 10}:
                        _finish_binary_section(handle, index)
                        continue
                    _open_binary_payload(handle)
                    faces: list[np.ndarray] = []
                    cells: list[np.ndarray] = []
                    # Fluent uses 32-bit integer connectivity here even in 3013.
                    if element_type == 0:
                        for _ in range(face_count):
                            face_type = int(
                                np.frombuffer(handle.read(4), dtype="<i4", count=1)[0]
                            )
                            node_count = {2: 2, 3: 3, 4: 4}.get(face_type)
                            if face_type == 5:
                                node_count = int(
                                    np.frombuffer(
                                        handle.read(4), dtype="<i4", count=1
                                    )[0]
                                )
                            if node_count is None or node_count < 3:
                                raise ValueError(
                                    f"unsupported Fluent mixed face type={face_type}"
                                )
                            record = np.frombuffer(
                                handle.read((node_count + 2) * 4),
                                dtype="<i4",
                                count=node_count + 2,
                            )
                            faces.append(record[:node_count].astype(np.int64))
                            cells.append(record[node_count : node_count + 2].astype(np.int64))
                    else:
                        node_count = {2: 2, 3: 3, 4: 4}.get(element_type)
                        if element_type == 5:
                            for _ in range(face_count):
                                polygon_nodes = int(
                                    np.frombuffer(
                                        handle.read(4), dtype="<i4", count=1
                                    )[0]
                                )
                                if polygon_nodes < 3:
                                    raise ValueError(
                                        f"invalid Fluent polygon size={polygon_nodes}"
                                    )
                                record = np.frombuffer(
                                    handle.read((polygon_nodes + 2) * 4),
                                    dtype="<i4",
                                    count=polygon_nodes + 2,
                                )
                                faces.append(record[:polygon_nodes].astype(np.int64))
                                cells.append(
                                    record[polygon_nodes : polygon_nodes + 2].astype(np.int64)
                                )
                            face_zones[zone_id] = faces
                            face_cells[zone_id] = cells
                            zone_types[zone_id] = zone_type
                            _finish_binary_section(handle, index)
                            continue
                        if node_count is None:
                            raise ValueError(
                                f"unsupported Fluent face element={element_type}"
                            )
                        record_width = node_count + 2
                        payload = handle.read(face_count * record_width * 4)
                        records = np.frombuffer(payload, dtype="<i4").reshape(
                            face_count, record_width
                        )
                        faces = [row[:node_count].astype(np.int64) for row in records]
                        cells = [row[node_count:].astype(np.int64) for row in records]
                    face_zones[zone_id] = faces
                    face_cells[zone_id] = cells
                    zone_types[zone_id] = zone_type
                    _finish_binary_section(handle, index)
                elif index == b"39":
                    zone_match = ZONE_RE.match(line)
                    if zone_match:
                        zone_names[int(zone_match.group(1))] = (
                            zone_match.group(2).decode("utf-8"),
                            zone_match.group(3).decode("utf-8"),
                        )
    finally:
        temporary.unlink(missing_ok=True)
    if nodes is None:
        raise ValueError(f"Fluent case contains no node coordinates: {case_gz}")
    return {
        "nodes": nodes,
        "first_node": int(first_node),
        "faces": face_zones,
        "face_cells": face_cells,
        "zone_types": zone_types,
        "zone_names": zone_names,
    }


def _zone_geometry(mesh: dict[str, Any], zone_id: int) -> dict[str, Any]:
    nodes = mesh["nodes"]
    first_node = int(mesh["first_node"])
    faces = mesh["faces"][zone_id]
    unique = np.unique(np.concatenate(faces)).astype(np.int64)
    area = 0.0
    area_vector = np.zeros(3, dtype=np.float64)
    for face in faces:
        points = nodes[face - first_node]
        for index in range(1, len(points) - 1):
            cross = np.cross(points[index] - points[0], points[index + 1] - points[0])
            area += float(np.linalg.norm(cross) * 0.5)
            area_vector += cross * 0.5
    return {
        "coords_m": np.asarray(nodes[unique - first_node], dtype=np.float64),
        "area_m2": float(area),
        "area_vector_m2": area_vector,
        "faces": int(len(faces)),
        "unique_points": int(len(unique)),
        "face_node_repetitions": int(sum(len(face) for face in faces) - len(unique)),
    }


def _transform_coords(raw_m: np.ndarray, bundle: dict[str, np.ndarray]) -> np.ndarray:
    coords_mm = raw_m * float(bundle["unit_factor"])
    return (
        (coords_mm - bundle["transform_centroid"]) @ bundle["transform_rotation"]
    ) / float(bundle["coord_scale"])


def _nearest_interior_mean(
    coords: np.ndarray,
    strict_indices: np.ndarray,
    point: np.ndarray,
    *,
    count: int = 512,
) -> np.ndarray:
    """Mean of the nearest strict-volume cell centres without building a huge tree."""
    best_distance = np.empty((0,), dtype=np.float64)
    best_points = np.empty((0, 3), dtype=np.float64)
    for start in range(0, len(strict_indices), 250_000):
        indices = strict_indices[start : start + 250_000]
        points = np.asarray(coords[indices], dtype=np.float64)
        distance = np.sum(np.square(points - point[None, :]), axis=1)
        merged_distance = np.concatenate([best_distance, distance])
        merged_points = np.concatenate([best_points, points], axis=0)
        keep = min(int(count), len(merged_distance))
        selected = np.argpartition(merged_distance, keep - 1)[:keep]
        best_distance = merged_distance[selected]
        best_points = merged_points[selected]
    if not len(best_points):
        raise ValueError("strict-volume coordinate set is empty")
    return best_points.mean(axis=0)


def _case_paths(volume_manifest: dict[str, Any]) -> tuple[Path, Path, Path]:
    raw_peak = Path(volume_manifest["parents"]["raw_peak"]["path"])
    raw_case = raw_peak.parent.parent
    cases = sorted(raw_case.glob("*.cas.gz"))
    if len(cases) != 1:
        # A small number of migrated cases retain both the case read by the
        # production Fluent journal and an older/renamed copy.  The journal is
        # the auditable source of truth for which mesh generated the fields.
        journal_case_names: set[str] = set()
        for journal in sorted(raw_case.glob("*.jou")):
            journal_text = journal.read_text(encoding="utf-8", errors="ignore")
            journal_case_names.update(
                match.group(1)
                for match in re.finditer(
                    r"(?:^|[/\\])([^/\\\s\"']+\.cas\.gz)(?:[\"']|\s|$)",
                    journal_text,
                    flags=re.MULTILINE,
                )
            )
        journal_cases = [path for path in cases if path.name in journal_case_names]
        if len(journal_cases) == 1:
            cases = journal_cases
        else:
            raise ValueError(
                f"cannot select one journal-backed .cas.gz in {raw_case}: "
                f"files={[path.name for path in cases]}, "
                f"journal_names={sorted(journal_case_names)}"
            )
    bc_metadata = raw_case / "processed/coord_normalized/bc_metadata.json"
    bundle = Path(volume_manifest["parents"]["bundle"]["path"])
    return cases[0], bc_metadata, bundle


def _inlet_udf_area(raw_case: Path) -> tuple[float, Path]:
    candidates = sorted(raw_case.glob("udf-inlet*.c")) + sorted(
        raw_case.glob("libudf/src/udf-inlet*.c")
    )
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(
            r"DEFINE_PROFILE\s*\(\s*my_inlet.*?F_PROFILE.*?\)\s*/\s*"
            r"([0-9.eE+-]+)\s*;",
            text,
            flags=re.S,
        )
        if match:
            return float(match.group(1)), path
    raise ValueError(f"cannot recover inlet area from UDF in {raw_case}")


def _outlet_udf_zone_ids(udf_path: Path) -> dict[str, int]:
    text = udf_path.read_text(encoding="utf-8", errors="ignore")
    thread_to_label = {"t1": "out-le", "t2": "out-li", "t3": "out-ri", "t4": "out-re"}
    output: dict[str, int] = {}
    for thread, label in thread_to_label.items():
        match = re.search(
            rf"\b{thread}\s*=\s*Lookup_Thread\s*\(\s*d\s*,\s*([0-9]+)\s*\)",
            text,
        )
        if match:
            output[label] = int(match.group(1))
    if set(output) != set(EXPECTED_OUTLETS):
        raise ValueError(f"cannot recover four outlet thread IDs from {udf_path}")
    return output


def _build_case(
    row: dict[str, Any],
    role: str,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    volume_manifest_path = Path(row["manifest"])
    volume_manifest = json.loads(volume_manifest_path.read_text(encoding="utf-8"))
    canonical_id = str(row["canonical_id"])
    case_gz, bc_path, bundle_path = _case_paths(volume_manifest)
    if not bc_path.is_file():
        raise FileNotFoundError(bc_path)
    mesh = _read_fluent_boundaries(case_gz)
    udf_area, udf_path = _inlet_udf_area(case_gz.parent)
    outlet_zone_ids = _outlet_udf_zone_ids(udf_path)
    inlet_zone_ids = []
    for zone_id, zone_type in mesh["zone_types"].items():
        type_name, name = mesh["zone_names"].get(zone_id, ("unknown", f"zone-{zone_id}"))
        if zone_type == 10 and type_name == "velocity-inlet":
            inlet_zone_ids.append(zone_id)
    if len(inlet_zone_ids) != 1 or any(
        mesh["zone_types"].get(zone_id) != 5 for zone_id in outlet_zone_ids.values()
    ):
        raise ValueError(
            f"{canonical_id}: UDF/mesh boundary-zone contract mismatch"
        )
    inlet_zone_id = inlet_zone_ids[0]
    inlet_name = mesh["zone_names"][inlet_zone_id][1]
    outlet_names = list(EXPECTED_OUTLETS)
    inlet_geometry = _zone_geometry(mesh, inlet_zone_id)
    outlet_geometry = {
        name: _zone_geometry(mesh, outlet_zone_ids[name]) for name in outlet_names
    }
    with np.load(bundle_path, allow_pickle=False) as bundle_npz:
        bundle = {
            key: np.asarray(bundle_npz[key])
            for key in (
                "unit_factor",
                "transform_centroid",
                "transform_rotation",
                "coord_scale",
            )
        }
    inlet_coords = _transform_coords(inlet_geometry["coords_m"], bundle)
    outlet_coords_by_zone = {
        name: _transform_coords(item["coords_m"], bundle)
        for name, item in outlet_geometry.items()
    }
    all_wall_node_ids = np.unique(
        np.concatenate(
            [
                np.concatenate(mesh["faces"][zone_id])
                for zone_id, zone_type in mesh["zone_types"].items()
                if zone_type == 3
            ]
        )
    ).astype(np.int64)
    all_wall_coords = _transform_coords(
        mesh["nodes"][all_wall_node_ids - int(mesh["first_node"])], bundle
    )

    files = volume_manifest["files"]
    strict_coords = np.load(files["interior_coords"]["path"], mmap_mode="r")
    strict_is_wall = np.load(files["interior_is_wall"]["path"], mmap_mode="r")
    strict_indices = np.flatnonzero(~np.asarray(strict_is_wall, dtype=bool))
    inlet_centroid = inlet_coords.mean(axis=0)
    inward_hint = _nearest_interior_mean(
        strict_coords, strict_indices, inlet_centroid, count=min(512, len(strict_indices))
    ) - inlet_centroid
    raw_normal = np.asarray(inlet_geometry["area_vector_m2"], dtype=np.float64)
    normal_registered = raw_normal @ np.asarray(bundle["transform_rotation"], dtype=np.float64)
    normal_registered /= max(float(np.linalg.norm(normal_registered)), 1e-30)
    if float(np.dot(normal_registered, inward_hint)) < 0:
        normal_registered *= -1.0

    bc = json.loads(bc_path.read_text(encoding="utf-8"))
    key = f"merged-{int(volume_manifest['peak_step'])}"
    if key not in bc.get("data", {}):
        raise ValueError(f"{canonical_id}: exact peak key {key} missing in bc_metadata")
    peak_values = [float(value) for value in bc["data"][key]]
    if len(peak_values) != 5 or not np.isfinite(peak_values).all():
        raise ValueError(f"{canonical_id}: invalid peak boundary values")
    inlet_flow = peak_values[0]
    inlet_speed = inlet_flow / float(inlet_geometry["area_m2"])
    inlet_velocity = np.broadcast_to(
        normal_registered * inlet_speed, inlet_coords.shape
    ).copy()
    pressure_reference = float(volume_manifest["scales"]["pressure_reference_pa"])
    outlet_coords = []
    outlet_pressure = []
    outlet_zone = []
    for zone_index, name in enumerate(outlet_names):
        coords = outlet_coords_by_zone[name]
        target = peak_values[OUTLET_TO_BC_INDEX[name]] - pressure_reference
        outlet_coords.append(coords)
        outlet_pressure.append(np.full(len(coords), target, dtype=np.float64))
        outlet_zone.append(np.full(len(coords), zone_index, dtype=np.int8))
    outlet_coords_array = np.concatenate(outlet_coords, axis=0)
    outlet_pressure_array = np.concatenate(outlet_pressure, axis=0)
    outlet_zone_array = np.concatenate(outlet_zone, axis=0)

    # The frozen sidecar wall is sometimes an exact subset of the Fluent wall
    # vertices and sometimes a different surface tessellation of the same
    # lumen.  Audit sidecar -> union(all physical Fluent wall zones), then
    # normalize the mismatch by the sidecar's own vertex spacing.  The reverse
    # direction is invalid because Fluent can retain extra disconnected wall
    # zones that intentionally do not belong to the frozen lumen surface.
    sidecar_wall = np.asarray(
        np.load(files["wall_coords"]["path"], mmap_mode="r"), dtype=np.float64
    )
    wall_distance = cKDTree(np.asarray(all_wall_coords, dtype=np.float64)).query(
        sidecar_wall, k=1
    )[0]
    sidecar_spacing = cKDTree(sidecar_wall).query(sidecar_wall, k=2)[0][:, 1]
    spacing_p99 = float(np.quantile(sidecar_spacing, 0.99))
    alignment_p99 = float(np.quantile(wall_distance, 0.99))
    alignment_max = float(wall_distance.max())
    alignment_p99_spacing_ratio = alignment_p99 / max(spacing_p99, 1e-12)
    alignment_max_spacing_ratio = alignment_max / max(spacing_p99, 1e-12)
    inlet_area_relative_error = abs(udf_area - inlet_geometry["area_m2"]) / udf_area

    case_dir = guard_write_path(output_root / "cases" / canonical_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    boundary_npz = case_dir / "peak_boundaries.npz"
    temporary_npz = boundary_npz.with_name(f".{boundary_npz.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary_npz,
        inlet_coords=inlet_coords.astype(np.float32),
        inlet_velocity_m_s=inlet_velocity.astype(np.float32),
        outlet_coords=outlet_coords_array.astype(np.float32),
        outlet_pressure_relative_pa=outlet_pressure_array.astype(np.float32),
        outlet_zone=outlet_zone_array.astype(np.int8),
        outlet_zone_names=np.asarray(outlet_names),
        wall_coords=all_wall_coords.astype(np.float32),
    )
    os.replace(temporary_npz, boundary_npz)
    gate_reasons = []
    gate_warnings = []
    if inlet_geometry["unique_points"] < 16:
        gate_reasons.append("inlet_too_few_unique_points")
    if len(all_wall_coords) < 100:
        gate_reasons.append("wall_too_few_unique_points")
    if any(item["unique_points"] < 16 for item in outlet_geometry.values()):
        gate_reasons.append("outlet_too_few_unique_points")
    if inlet_area_relative_error > 1e-3:
        # ``BC_Inlet`` is the measured, exact-timestep volume flow from
        # Global_conditions/vf-in-rfile.out.  Dividing it by the true mesh
        # inlet area recovers the actual uniform velocity even when a migrated
        # UDF retained a stale denominator; preserve that discrepancy as an
        # audit warning instead of rejecting otherwise observable BC data.
        gate_warnings.append("inlet_udf_denominator_differs_from_mesh_area")
    exact_vertex_alignment = alignment_p99 <= 5e-5 and alignment_max <= 1e-4
    same_surface_tessellation = (
        alignment_p99_spacing_ratio <= 1.5
        and alignment_max_spacing_ratio <= 2.0
    )
    if not (exact_vertex_alignment or same_surface_tessellation):
        gate_reasons.append("registered_wall_alignment_mismatch")
    if not all(
        np.isfinite(array).all()
        for array in (inlet_coords, inlet_velocity, outlet_coords_array, outlet_pressure_array)
    ):
        gate_reasons.append("nonfinite_boundary_asset")
    case_report = {
        "canonical_id": canonical_id,
        "cohort": row["cohort"],
        "role": role,
        "status": "completed" if not gate_reasons else "failed",
        "gate_result": "pass" if not gate_reasons else "fail",
        "gate_reasons": gate_reasons,
        "gate_warnings": gate_warnings,
        "peak_step": int(volume_manifest["peak_step"]),
        "provenance": {
            "fluent_case": {"path": str(case_gz), "sha256": sha256_file(case_gz)},
            "bc_metadata": {"path": str(bc_path), "sha256": sha256_file(bc_path)},
            "udf": {"path": str(udf_path), "sha256": sha256_file(udf_path)},
            "bundle": {
                "path": str(bundle_path),
                "sha256": volume_manifest["parents"]["bundle"]["sha256"],
            },
            "volume_manifest": {
                "path": str(volume_manifest_path),
                "sha256": row["manifest_sha256"],
            },
        },
        "units": {
            "coordinates": "dimensionless registered coordinates",
            "inlet_velocity": "m/s; uniform normal profile from peak flow / true inlet area",
            "outlet_pressure": "Pa; peak zone-mean pressure minus fixed strict-volume case gauge",
        },
        "inlet": {
            "zone": inlet_name,
            "faces": inlet_geometry["faces"],
            "unique_points": inlet_geometry["unique_points"],
            "area_m2": inlet_geometry["area_m2"],
            "udf_area_m2": udf_area,
            "mesh_udf_area_relative_error": inlet_area_relative_error,
            "udf_equivalent_peak_flow_m3_s": inlet_speed * udf_area,
            "velocity_source": (
                "exact bc_metadata BC_Inlet measured flow / true Fluent inlet area; "
                "UDF denominator retained as provenance diagnostic"
            ),
            "peak_flow_m3_s": inlet_flow,
            "uniform_speed_m_s": inlet_speed,
            "registered_inward_normal": normal_registered.tolist(),
        },
        "wall": {
            "physical_zone_count": int(
                sum(1 for value in mesh["zone_types"].values() if value == 3)
            ),
            "unique_points": int(len(all_wall_coords)),
            "source": "all Fluent wall face zones",
        },
        "outlets": {
            name: {
                "zone_index": index,
                "mesh_zone": mesh["zone_names"][outlet_zone_ids[name]][1],
                "mesh_zone_id": outlet_zone_ids[name],
                "faces": outlet_geometry[name]["faces"],
                "unique_points": outlet_geometry[name]["unique_points"],
                "area_m2": outlet_geometry[name]["area_m2"],
                "peak_absolute_pressure_pa": peak_values[OUTLET_TO_BC_INDEX[name]],
                "peak_relative_pressure_pa": (
                    peak_values[OUTLET_TO_BC_INDEX[name]] - pressure_reference
                ),
            }
            for index, name in enumerate(outlet_names)
        },
        "alignment": {
            "method": "sidecar wall -> union(all Fluent wall zones)",
            "sidecar_wall_points": int(len(sidecar_wall)),
            "fluent_wall_points": int(len(all_wall_coords)),
            "sidecar_spacing_p99": spacing_p99,
            "nearest_distance_p50": float(np.quantile(wall_distance, 0.50)),
            "nearest_distance_p95": float(np.quantile(wall_distance, 0.95)),
            "nearest_distance_p99": alignment_p99,
            "nearest_distance_max": alignment_max,
            "p99_to_spacing_p99_ratio": alignment_p99_spacing_ratio,
            "max_to_spacing_p99_ratio": alignment_max_spacing_ratio,
            "alignment_class": (
                "exact_vertices" if exact_vertex_alignment else "same_surface_retessellated"
            ),
        },
        "boundary_npz": {
            "path": str(boundary_npz.resolve()),
            "sha256": sha256_file(boundary_npz),
        },
    }
    case_manifest_path = atomic_write_json(case_dir / "manifest.json", case_report)
    aggregate_row = {
        "canonical_id": canonical_id,
        "cohort": row["cohort"],
        "role": role,
        "source_volume_manifest": str(volume_manifest_path.resolve()),
        "source_volume_manifest_sha256": row["manifest_sha256"],
        "boundary_manifest": str(case_manifest_path.resolve()),
        "boundary_manifest_sha256": sha256_file(case_manifest_path),
        "boundary_npz": str(boundary_npz.resolve()),
        "boundary_npz_sha256": sha256_file(boundary_npz),
    }
    return aggregate_row, case_report


def _stream_stats(
    rows: list[dict[str, Any]],
    split_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    train_rows = [row for row in rows if row["role"] == "train"]
    total_count = 0
    for row in train_rows:
        manifest = json.loads(
            Path(row["source_volume_manifest"]).read_text(encoding="utf-8")
        )
        total_count += int(manifest["counts"]["strict_interior"])
    curvature_tmp = Path(tempfile.gettempdir()) / f"qs_v3_curvature_{os.getpid()}.dat"
    curvature = np.memmap(curvature_tmp, dtype="float32", mode="w+", shape=(total_count,))
    cursor = 0
    try:
        for row in train_rows:
            manifest = json.loads(
                Path(row["source_volume_manifest"]).read_text(encoding="utf-8")
            )
            files = manifest["files"]
            geometry = np.load(files["interior_geometry"]["path"], mmap_mode="r")
            is_wall = np.load(files["interior_is_wall"]["path"], mmap_mode="r")
            selected = np.asarray(geometry[~np.asarray(is_wall, dtype=bool), 2], dtype=np.float32)
            curvature[cursor : cursor + len(selected)] = np.abs(selected)
            cursor += len(selected)
        curvature.flush()
        curvature_clip = float(np.quantile(curvature, 0.99))
    finally:
        del curvature
        curvature_tmp.unlink(missing_ok=True)

    sums = {
        "geometry": np.zeros(3, dtype=np.float64),
        "geometry_sq": np.zeros(3, dtype=np.float64),
        "velocity": np.zeros(3, dtype=np.float64),
        "velocity_sq": np.zeros(3, dtype=np.float64),
        "pressure": np.zeros(1, dtype=np.float64),
        "pressure_sq": np.zeros(1, dtype=np.float64),
    }
    count = 0
    speed_sq_sum = 0.0
    chunk_size = 1_000_000
    for row in train_rows:
        manifest = json.loads(
            Path(row["source_volume_manifest"]).read_text(encoding="utf-8")
        )
        files = manifest["files"]
        geometry = np.load(files["interior_geometry"]["path"], mmap_mode="r")
        velocity = np.load(files["velocity_m_s"]["path"], mmap_mode="r")
        pressure = np.load(files["pressure_relative_pa"]["path"], mmap_mode="r")
        is_wall = np.load(files["interior_is_wall"]["path"], mmap_mode="r")
        strict = np.flatnonzero(~np.asarray(is_wall, dtype=bool))
        for start in range(0, len(strict), chunk_size):
            indices = strict[start : start + chunk_size]
            geom = np.asarray(geometry[indices], dtype=np.float64).copy()
            geom[:, 2] = np.clip(geom[:, 2], -curvature_clip, curvature_clip)
            vel = np.asarray(velocity[indices], dtype=np.float64)
            pres = np.asarray(pressure[indices], dtype=np.float64)[:, None]
            sums["geometry"] += geom.sum(axis=0)
            sums["geometry_sq"] += np.square(geom).sum(axis=0)
            sums["velocity"] += vel.sum(axis=0)
            sums["velocity_sq"] += np.square(vel).sum(axis=0)
            sums["pressure"] += pres.sum(axis=0)
            sums["pressure_sq"] += np.square(pres).sum(axis=0)
            speed_sq_sum += float(np.square(vel).sum())
            count += len(indices)
    if count != total_count:
        raise ValueError("train123 stats count drift")

    def mean_std(sum_value: np.ndarray, square_sum: np.ndarray) -> tuple[list, list]:
        mean = sum_value / count
        variance = np.maximum(square_sum / count - np.square(mean), 0.0)
        return mean.tolist(), np.sqrt(variance).tolist()

    geometry_mean, geometry_std = mean_std(sums["geometry"], sums["geometry_sq"])
    velocity_mean, velocity_std = mean_std(sums["velocity"], sums["velocity_sq"])
    pressure_mean, pressure_std = mean_std(sums["pressure"], sums["pressure_sq"])
    return {
        "schema_version": 3,
        "route": ROUTE,
        "created_at": utc_now(),
        "status": "completed",
        "scope": "train123 strict-volume only",
        "split": {"path": str(SPLIT_OUTPUT.resolve()), "sha256": split_sha256},
        "population": {
            "roles": ["train"],
            "wall_duplicate_rows_excluded": True,
            "interior_is_wall": False,
        },
        "case_asset_sha256": {
            row["canonical_id"]: {
                "volume_manifest": row["source_volume_manifest_sha256"],
                "boundary_manifest": row["boundary_manifest_sha256"],
            }
            for row in train_rows
        },
        "geometry": {
            "names": ["abscissa_norm", "local_radius", "curvature_signed_log1p"],
            "count": count,
            "curvature_clip_quantile": 0.99,
            "curvature_clip_abs": curvature_clip,
            "mean": geometry_mean,
            "std": geometry_std,
        },
        "velocity_m_s": {
            "names": ["u", "v", "w"],
            "count": count,
            "mean": velocity_mean,
            "std": velocity_std,
        },
        "pressure_relative_pa": {
            "names": ["p"],
            "count": count,
            "mean": pressure_mean,
            "std": pressure_std,
        },
        "physics_nondimensionalization": {
            "density_kg_m3": 1060.0,
            "velocity_m_s": 1.0,
            "pressure_pa": 1060.0,
        },
        "train_diagnostics": {
            "velocity_speed_rms_m_s": float(np.sqrt(speed_sq_sum / count))
        },
    }


def build(seed: int = 1234, *, force: bool = False) -> dict[str, Any]:
    output_root = guard_write_path(OUTPUT_ROOT)
    audit_root = guard_write_path(AUDIT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "manifest.json").exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing V3 assets without --force: {output_root}"
        )
    split = derive_split(seed)
    atomic_write_json(SPLIT_OUTPUT, split)
    split_sha256 = sha256_file(SPLIT_OUTPUT)
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    rows_by_id = {row["canonical_id"]: row for row in source["cases"]}
    role_by_id = {
        **{case_id: "train" for case_id in split["train_cases"]},
        **{case_id: "val" for case_id in split["val_cases"]},
        **{case_id: "test" for case_id in split["test_cases"]},
    }
    aggregate_rows = []
    case_reports = []
    for index, canonical_id in enumerate(sorted(role_by_id), start=1):
        print(
            json.dumps(
                {
                    "event": "qs_v3_boundary_case_start",
                    "index": index,
                    "total": len(role_by_id),
                    "case_id": canonical_id,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        aggregate_row, report = _build_case(
            rows_by_id[canonical_id], role_by_id[canonical_id], output_root
        )
        aggregate_rows.append(aggregate_row)
        case_reports.append(report)
    stats = _stream_stats(aggregate_rows, split_sha256, output_root)
    stats_path = atomic_write_json(output_root / "field_stats.json", stats)
    failed = [row for row in case_reports if row["gate_result"] != "pass"]
    warned = [row for row in case_reports if row.get("gate_warnings")]
    aggregate = {
        "schema_version": 3,
        "route": ROUTE,
        "created_at": utc_now(),
        "status": "completed" if not failed else "failed",
        "split": {"path": str(SPLIT_OUTPUT.resolve()), "sha256": split_sha256},
        "source_volume_manifest": {
            "path": str(SOURCE_MANIFEST.resolve()),
            "sha256": sha256_file(SOURCE_MANIFEST),
        },
        "field_stats": {
            "path": str(stats_path.resolve()),
            "sha256": sha256_file(stats_path),
        },
        "counts": {"train": 123, "val": 15, "test": 35, "total": 173},
        "cases": aggregate_rows,
    }
    manifest_path = atomic_write_json(output_root / "manifest.json", aggregate)
    csv_path = guard_write_path(audit_root / "cases.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "canonical_id",
                "cohort",
                "role",
                "gate_result",
                "gate_warnings",
                "inlet_points",
                "outlet_points_min",
                "inlet_area_relative_error",
                "alignment_p99",
                "alignment_max",
            ],
        )
        writer.writeheader()
        for report in case_reports:
            writer.writerow(
                {
                    "canonical_id": report["canonical_id"],
                    "cohort": report["cohort"],
                    "role": report["role"],
                    "gate_result": report["gate_result"],
                    "gate_warnings": ";".join(report.get("gate_warnings", [])),
                    "inlet_points": report["inlet"]["unique_points"],
                    "outlet_points_min": min(
                        value["unique_points"] for value in report["outlets"].values()
                    ),
                    "inlet_area_relative_error": report["inlet"][
                        "mesh_udf_area_relative_error"
                    ],
                    "alignment_p99": report["alignment"]["nearest_distance_p99"],
                    "alignment_max": report["alignment"]["nearest_distance_max"],
                }
            )
    report = {
        "schema_version": 1,
        "route": ROUTE,
        "created_at": utc_now(),
        "status": "completed" if not failed else "failed",
        "gate_result": "pass" if not failed else "fail",
        "case_count": len(case_reports),
        "passed_cases": len(case_reports) - len(failed),
        "failed_cases": [row["canonical_id"] for row in failed],
        "warning_case_count": len(warned),
        "warning_cases": {
            row["canonical_id"]: row["gate_warnings"] for row in warned
        },
        "split": {"path": str(SPLIT_OUTPUT.resolve()), "sha256": split_sha256},
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "field_stats": {
            "path": str(stats_path.resolve()),
            "sha256": sha256_file(stats_path),
        },
        "cases_csv": {"path": str(csv_path.resolve()), "sha256": sha256_file(csv_path)},
        "coverage": {
            "inlet_mesh_and_peak_flow": len(case_reports) - len(failed),
            "four_outlet_meshes_and_peak_pressures": len(case_reports) - len(failed),
            "registered_frame_alignment": len(case_reports) - len(failed),
        },
    }
    atomic_write_json(audit_root / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.seed, force=args.force), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
