"""Stage-0 builder and hard Gate for the V4 BC/RCR route.

The builder deliberately references immutable upstream assets and writes only
the new V4 tree.  Transient caches are derived from the already registered
15k-point CSVs (81 frames per case), avoiding a second scan of the 4.3 TB raw
full-volume ASCII tree while still enforcing exact cross-time point identity.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from wss_pinn.tools.audit_boundary_field_v4 import (
    OUTLET_LABEL as LOG_OUTLET_LABEL,
    PRESSURE_RE,
    TIME_RE,
    _face_geometry,
)
from wss_pinn.tools.build_qs_smooth_v3 import (
    _nearest_interior_mean,
    _read_fluent_boundaries,
    _transform_coords,
    _zone_geometry,
)
from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    guard_write_path,
    sha256_file,
    sha256_json,
    utc_now,
)

from .bc_contract import (
    BC_TRANSFORM_DESCRIPTION,
    INLET_AREA_RATIO_INDEX,
    bc_scale_policy,
    bc_vector_raw as _bc_vector_raw,
    transform_bc_raw,
)
from .config import DATA_ROOT, MANIFEST_PATH, OUTLET_ORDER, ROUTE, SPLIT_PATH, STATS_PATH
from .waveform import DEFAULT_FOURIER, PERIOD_S, parse_udf, q_nom_numpy, q_nom_peak


SOURCE_SPLIT = (
    ROOT
    / "wss_pinn/configs/splits/"
    "split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_train138_test35_exclude_SHI_YUN_XI_v1.json"
)
SOURCE_VOLUME_MANIFEST = ROOT / "data_wss_pinn/volume_uvwp_peak_v1_train138_test35/manifest.json"
SOURCE_BOUNDARY_MANIFEST = (
    ROOT / "data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/manifest.json"
)
SOURCE_FACE_MANIFEST = (
    ROOT / "data_wss_pinn/volume_uvwp_peak_field_v4_train123_val15/boundary_contract/manifest.json"
)
SOURCE_STEADY_STATS = ROOT / "data_wss_pinn/volume_uvwp_peak_v1_train138_test35/field_stats.json"
AUDIT_ROOT = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4/audits/stage0"
GATE_REPORT = AUDIT_ROOT / "gate_report.json"
CASE_REPORTS = AUDIT_ROOT / "cases"

PROTECTED_RELATED_GROUP = {
    "AG/slow/HOU_SHEN_QIAN",
    "AG/slow/KANG_XI_MING",
}
CSV_COLUMNS = [
    "x",
    "y",
    "z",
    "Abscissa",
    "NormRadius",
    "Curvature",
    "u",
    "v",
    "w",
    "p",
    "is_wall",
]
RHEOLOGY_REFERENCE = {
    "mu_inf_pa_s": 0.0035,
    "mu_zero_pa_s": 0.16,
    "lambda_s": 8.2,
    "a": 0.64,
    "n": 0.2128,
}


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


def derive_split(seed: int = 1234, folds: int = 5) -> dict[str, Any]:
    source = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    train = list(source["train_cases"])
    test = list(source["test_cases"])
    if len(train) != 138 or len(test) != 35 or set(train) & set(test):
        raise ValueError("source split is not the frozen train138/test35 contract")
    fold_cases = [[] for _ in range(folds)]
    for stratum in sorted({_stratum(value) for value in train}):
        members = [
            value
            for value in train
            if _stratum(value) == stratum and value not in PROTECTED_RELATED_GROUP
        ]
        members.sort(
            key=lambda value: hashlib.sha256(
                f"{seed}|cv5|{stratum}|{value}".encode("utf-8")
            ).hexdigest()
        )
        for index, value in enumerate(members):
            fold_cases[index % folds].append(value)
    protected_fold = min(range(folds), key=lambda index: len(fold_cases[index]))
    fold_cases[protected_fold].extend(sorted(PROTECTED_RELATED_GROUP))
    flattened = [value for fold in fold_cases for value in fold]
    if set(flattened) != set(train) or len(flattened) != len(train):
        raise ValueError("CV folds do not exactly partition train138")
    fold_payload = []
    for index, validation in enumerate(fold_cases):
        validation = sorted(validation)
        validation_set = set(validation)
        fold_payload.append(
            {
                "fold": index,
                "train_cases": [value for value in train if value not in validation_set],
                "validation_cases": validation,
                "stratum_counts": {
                    stratum: sum(_stratum(value) == stratum for value in validation)
                    for stratum in sorted({_stratum(value) for value in train})
                },
            }
        )
    payload = {
        "schema_version": 1,
        "route": ROUTE,
        "created_at": utc_now(),
        "seed": seed,
        "source_split": str(SOURCE_SPLIT.resolve()),
        "source_split_sha256": sha256_file(SOURCE_SPLIT),
        "counts": {"train": 138, "test": 35, "cv_folds": folds},
        "train_cases": train,
        "test_cases": test,
        "cv": {
            "algorithm": "stratified deterministic SHA256 round-robin within train138",
            "protected_related_group": sorted(PROTECTED_RELATED_GROUP),
            "protected_group_fold": protected_fold,
            "folds": fold_payload,
        },
        "test35_unchanged": True,
    }
    payload["content_sha256"] = sha256_json(payload)
    return payload


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _source_maps() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    volume = json.loads(SOURCE_VOLUME_MANIFEST.read_text(encoding="utf-8"))
    boundary = json.loads(SOURCE_BOUNDARY_MANIFEST.read_text(encoding="utf-8"))
    faces = json.loads(SOURCE_FACE_MANIFEST.read_text(encoding="utf-8"))
    return (
        {row["canonical_id"]: row for row in volume["cases"]},
        {row["canonical_id"]: row for row in boundary["cases"]},
        {row["canonical_id"]: row for row in faces["cases"]},
    )


def _largest_fluent_log(raw_case: Path) -> Path | None:
    candidates = list((raw_case / "Global_conditions").glob("Fluent_*.out"))
    if not candidates:
        candidates = list(raw_case.glob("Fluent_*.out"))
    return max(candidates, key=lambda value: value.stat().st_size) if candidates else None


def _parse_export_outlet_log(path: Path | None, expected_steps: list[int]) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    wanted = set(expected_steps)
    collected: dict[int, dict[str, tuple[float, float]]] = {}
    buffer: dict[str, list[tuple[float, float]]] = {key: [] for key in LOG_OUTLET_LABEL}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            pressure = PRESSURE_RE.search(line)
            if pressure:
                buffer[pressure.group(1)].append(
                    (float(pressure.group(2)), float(pressure.group(3)))
                )
                continue
            time_match = TIME_RE.search(line)
            if not time_match:
                continue
            step = int(time_match.group(2))
            if step in wanted and all(buffer.values()):
                values = {}
                for raw_label, rows in buffer.items():
                    unique = sorted(set(rows))
                    values[LOG_OUTLET_LABEL[raw_label]] = (
                        float(statistics.median(value[0] for value in unique)),
                        float(statistics.median(value[1] for value in unique)),
                    )
                collected[step] = values
            buffer = {key: [] for key in LOG_OUTLET_LABEL}
    if any(step not in collected for step in expected_steps):
        return None
    pressure = np.asarray(
        [[collected[step][label][0] for label in OUTLET_ORDER] for step in expected_steps],
        dtype=np.float32,
    )
    mass_flow = np.asarray(
        [[collected[step][label][1] for label in OUTLET_ORDER] for step in expected_steps],
        dtype=np.float32,
    )
    return {"pressure_pa": pressure, "mass_flow_kg_s": mass_flow}


def _transient_frame_paths(bc_metadata_path: Path) -> list[Path]:
    # ``coord_normalized`` is an older per-case frame.  V4 must instead read
    # the raw-mm registered-feature table and apply the frozen WSS-min bundle
    # transform so every time frame matches ``stl_landmarks_v4``.
    feature_root = bc_metadata_path.parent.parent / "features"
    paths = sorted(
        feature_root.glob("result_features_merged-*.csv"),
        key=lambda path: int(path.stem.rsplit("-", 1)[1]),
    )
    if len(paths) != 81:
        raise ValueError(f"expected 81 registered frames beside {bc_metadata_path}, got {len(paths)}")
    return paths


def _read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=CSV_COLUMNS, dtype=np.float32)


def _outlet_face_contract(
    mesh: dict[str, Any],
    boundary: dict[str, Any],
    bundle: dict[str, np.ndarray],
    strict_coords: np.ndarray,
    strict_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    all_coords = []
    all_normals = []
    all_weights = []
    all_zones = []
    for zone_index, label in enumerate(OUTLET_ORDER):
        zone_id = int(boundary["outlets"][label]["mesh_zone_id"])
        geometry = _face_geometry(mesh, zone_id)
        registered = _transform_coords(geometry["centroids_m"], bundle)
        zone = _zone_geometry(mesh, zone_id)
        raw_normal = np.asarray(zone["area_vector_m2"], dtype=np.float64)
        registered_normal = raw_normal @ np.asarray(bundle["transform_rotation"], dtype=np.float64)
        registered_normal /= max(float(np.linalg.norm(registered_normal)), 1.0e-30)
        centroid = registered.mean(axis=0)
        interior = _nearest_interior_mean(
            strict_coords,
            strict_indices,
            centroid,
            count=min(512, len(strict_indices)),
        )
        outward_hint = centroid - interior
        if float(np.dot(registered_normal, outward_hint)) < 0:
            registered_normal *= -1.0
        all_coords.append(registered.astype(np.float32))
        all_normals.append(
            np.broadcast_to(registered_normal.astype(np.float32), registered.shape).copy()
        )
        all_weights.append(geometry["areas_m2"].astype(np.float32))
        all_zones.append(np.full(len(registered), zone_index, dtype=np.int8))
    return {
        "outlet_coords": np.concatenate(all_coords),
        "outlet_normals": np.concatenate(all_normals),
        "outlet_area_weights_m2": np.concatenate(all_weights),
        "outlet_zone": np.concatenate(all_zones),
    }


def _case_complete(case_manifest: Path, build_transient: bool) -> bool:
    if not case_manifest.is_file():
        return False
    try:
        payload = json.loads(case_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("gate_result") != "pass":
        return False
    files = payload.get("files", {})
    boundary_faces = files.get("boundary_faces")
    if not isinstance(boundary_faces, dict) or not boundary_faces.get("path"):
        return False
    required = [Path(boundary_faces["path"])]
    if build_transient:
        transient_report = payload.get("transient", {})
        transient_files = files.get("transient", {})
        required_report_fields = {
            "max_bc_monitor_vs_udf_relative_difference",
            "target_domain_interior_rows",
            "excluded_non_target_interior_rows",
        }
        required_file_fields = {
            "coords",
            "geometry_raw",
            "is_wall",
            "eligible_interior",
            "steps",
            "times_s",
            "q_actual_m3_s",
            "velocity_m_s",
            "pressure_raw_pa",
        }
        if not required_report_fields.issubset(transient_report):
            return False
        if not required_file_fields.issubset(transient_files):
            return False
        if any(
            not isinstance(transient_files[key], dict)
            or not transient_files[key].get("path")
            for key in required_file_fields
        ):
            return False
        required.extend(Path(transient_files[key]["path"]) for key in transient_files)
    return all(path.is_file() for path in required)


def _build_case(task: dict[str, Any]) -> dict[str, Any]:
    case_id = task["case_id"]
    role = task["role"]
    build_transient = bool(task["build_transient"])
    resume = bool(task["resume"])
    case_dir = DATA_ROOT / "cases" / case_id
    case_manifest_path = case_dir / "manifest.json"
    if resume and _case_complete(case_manifest_path, build_transient):
        return json.loads(case_manifest_path.read_text(encoding="utf-8"))

    volume_row = task["volume_row"]
    boundary_row = task["boundary_row"]
    face_row = task.get("face_row")
    volume_path = Path(volume_row["manifest"])
    boundary_path = Path(boundary_row["boundary_manifest"])
    if sha256_file(volume_path) != volume_row["manifest_sha256"]:
        raise ValueError(f"{case_id}: frozen volume manifest hash drift")
    if sha256_file(boundary_path) != boundary_row["boundary_manifest_sha256"]:
        raise ValueError(f"{case_id}: frozen boundary manifest hash drift")
    volume = json.loads(volume_path.read_text(encoding="utf-8"))
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    raw_case = Path(boundary["provenance"]["fluent_case"]["path"]).parent
    udf_path = Path(boundary["provenance"]["udf"]["path"])
    parsed_udf = parse_udf(udf_path)
    if parsed_udf["fourier"] != DEFAULT_FOURIER:
        raise ValueError(f"{case_id}: inlet Fourier coefficients differ from frozen cohort")
    for key, value in RHEOLOGY_REFERENCE.items():
        if not math.isclose(float(parsed_udf["rheology"][key]), value, rel_tol=1.0e-10):
            raise ValueError(f"{case_id}: rheology drift in {key}")

    files = volume["files"]
    strict_coords = np.load(files["interior_coords"]["path"], mmap_mode="r")
    strict_is_wall = np.load(files["interior_is_wall"]["path"], mmap_mode="r")
    strict_indices = np.flatnonzero(~np.asarray(strict_is_wall, dtype=bool))
    bundle_path = Path(boundary["provenance"]["bundle"]["path"])
    with np.load(bundle_path, allow_pickle=False) as payload:
        bundle = {
            key: np.asarray(payload[key])
            for key in ("unit_factor", "transform_centroid", "transform_rotation", "coord_scale")
        }
    mesh = _read_fluent_boundaries(Path(boundary["provenance"]["fluent_case"]["path"]))
    outlet_contract = _outlet_face_contract(
        mesh, boundary, bundle, strict_coords, strict_indices
    )

    if face_row and face_row.get("face_interior_asset"):
        face_path = Path(face_row["face_interior_asset"]["path"])
        with np.load(face_path, allow_pickle=False) as payload:
            inlet_coords = np.asarray(payload["inlet_face_interior_coords"], dtype=np.float32)
            wall_coords = np.asarray(payload["wall_face_interior_coords"], dtype=np.float32)
        face_source = {"path": str(face_path), "sha256": sha256_file(face_path)}
    else:
        boundary_npz = Path(boundary_row["boundary_npz"])
        with np.load(boundary_npz, allow_pickle=False) as payload:
            inlet_coords = np.asarray(payload["inlet_coords"], dtype=np.float32)
            wall_coords = np.asarray(payload["wall_coords"], dtype=np.float32)
        face_source = {"path": str(boundary_npz), "sha256": sha256_file(boundary_npz)}
    inlet_normal = np.asarray(boundary["inlet"]["registered_inward_normal"], dtype=np.float32)
    inlet_normals = np.broadcast_to(inlet_normal, inlet_coords.shape).copy()

    boundary_faces_path = case_dir / "boundary_faces.npz"
    _atomic_npz(
        boundary_faces_path,
        inlet_coords=inlet_coords,
        inlet_normals=inlet_normals,
        wall_coords=wall_coords,
        outlet_coords=outlet_contract["outlet_coords"],
        outlet_normals=outlet_contract["outlet_normals"],
        outlet_area_weights_m2=outlet_contract["outlet_area_weights_m2"],
        outlet_zone=outlet_contract["outlet_zone"],
        outlet_zone_names=np.asarray(OUTLET_ORDER),
    )

    _, q_peak_nominal = q_nom_peak(parsed_udf["fourier"])
    a_mesh = float(boundary["inlet"]["area_m2"])
    a_udf = float(parsed_udf["a_udf_m2"])
    q_actual_peak = q_peak_nominal * a_mesh / a_udf
    rcr = parsed_udf["rcr"]
    outlet_areas = np.asarray(
        [float(boundary["outlets"][label]["area_m2"]) for label in OUTLET_ORDER],
        dtype=np.float64,
    )
    conductance = np.asarray([1.0 / (item["R1"] + item["R2"]) for item in rcr])
    mass_characteristic = 1060.0 * q_actual_peak * conductance / conductance.sum()
    rcr_pressure_scale = np.asarray(
        [(item["R1"] + item["R2"]) * mass_characteristic[index] for index, item in enumerate(rcr)],
        dtype=np.float64,
    )
    # frozen 2026-09-05 contract: index 1 is the inlet area ratio A_mesh/A_udf (= Q_actual/Q_nom), not Q_actual_peak
    bc_vector_raw = _bc_vector_raw(a_mesh, a_udf, outlet_areas, rcr)

    transient_files: dict[str, dict[str, Any]] = {}
    transient_report: dict[str, Any] = {"status": "not_built"}
    if build_transient:
        bc_path = Path(boundary["provenance"]["bc_metadata"]["path"])
        bc_metadata = json.loads(bc_path.read_text(encoding="utf-8"))
        frame_paths = _transient_frame_paths(bc_path)
        steps = [int(path.stem.rsplit("-", 1)[1]) for path in frame_paths]
        if any(right - left != 2 for left, right in zip(steps[:-1], steps[1:])):
            raise ValueError(f"{case_id}: exported field stride is not exactly two solver steps")
        times = (np.asarray(steps, dtype=np.float64) - steps[0]) * 0.005
        if not np.allclose(times, np.linspace(0.0, PERIOD_S, 81), rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{case_id}: time mapping is not 0..0.8 s at 0.01 s")
        first = _read_frame(frame_paths[0])
        raw_coords_mm = (
            first[["x", "y", "z"]].to_numpy(dtype=np.float64)
            * float(bundle["unit_factor"])
            / 1000.0
        )
        coords = (
            (raw_coords_mm - np.asarray(bundle["transform_centroid"], dtype=np.float64))
            @ np.asarray(bundle["transform_rotation"], dtype=np.float64)
            / float(bundle["coord_scale"])
        ).astype(np.float32)
        geometry_raw = first[["Abscissa", "NormRadius", "Curvature"]].to_numpy(dtype=np.float32)
        is_wall = first["is_wall"].to_numpy(dtype=np.float32) > 0.5
        target_tree = cKDTree(
            np.asarray(strict_coords[strict_indices], dtype=np.float32)
        )
        feature_interior = np.flatnonzero(~is_wall)
        feature_distance, _ = target_tree.query(coords[feature_interior], k=1)
        eligible_interior = np.zeros(len(coords), dtype=bool)
        eligible_interior[feature_interior[np.asarray(feature_distance) <= 5.0e-5]] = True
        if int(np.sum(eligible_interior)) < 256:
            raise ValueError(
                f"{case_id}: fewer than 256 transient sample points belong to the frozen target domain"
            )
        row_count = len(first)
        velocity_path = case_dir / "transient_velocity_m_s.npy"
        pressure_path = case_dir / "transient_pressure_raw_pa.npy"
        temporary_velocity = velocity_path.with_name(f".{velocity_path.name}.{os.getpid()}.tmp.npy")
        temporary_pressure = pressure_path.with_name(f".{pressure_path.name}.{os.getpid()}.tmp.npy")
        velocity = np.lib.format.open_memmap(
            temporary_velocity, mode="w+", dtype="float32", shape=(81, row_count, 3)
        )
        pressure = np.lib.format.open_memmap(
            temporary_pressure, mode="w+", dtype="float32", shape=(81, row_count)
        )
        max_static_delta = 0.0
        q_observed = []
        try:
            for frame_index, frame_path in enumerate(frame_paths):
                frame = first if frame_index == 0 else _read_frame(frame_path)
                if len(frame) != row_count:
                    raise ValueError(f"{case_id}: transient point count changed at {frame_path.name}")
                raw_frame_coords_mm = (
                    frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
                    * float(bundle["unit_factor"])
                    / 1000.0
                )
                frame_coords = (
                    (raw_frame_coords_mm - np.asarray(bundle["transform_centroid"], dtype=np.float64))
                    @ np.asarray(bundle["transform_rotation"], dtype=np.float64)
                    / float(bundle["coord_scale"])
                ).astype(np.float32)
                static = np.concatenate(
                    [
                        frame_coords,
                        frame[["Abscissa", "NormRadius", "Curvature"]].to_numpy(dtype=np.float32),
                    ],
                    axis=1,
                )
                reference = np.concatenate([coords, geometry_raw], axis=1)
                delta = float(np.max(np.abs(static - reference)))
                max_static_delta = max(max_static_delta, delta)
                if delta > 1.0e-6 or not np.array_equal(
                    frame["is_wall"].to_numpy(dtype=np.float32) > 0.5, is_wall
                ):
                    raise ValueError(f"{case_id}: cross-time point identity/order drift")
                velocity[frame_index] = (
                    frame[["u", "v", "w"]].to_numpy(dtype=np.float64)
                    @ np.asarray(bundle["transform_rotation"], dtype=np.float64)
                ).astype(np.float32)
                pressure[frame_index] = frame["p"].to_numpy(dtype=np.float32)
                bc_key = f"merged-{steps[frame_index]}"
                if bc_key not in bc_metadata["data"]:
                    raise ValueError(f"{case_id}: exact BC key missing: {bc_key}")
                q_observed.append(float(bc_metadata["data"][bc_key][0]))
            velocity.flush()
            pressure.flush()
            os.replace(temporary_velocity, velocity_path)
            os.replace(temporary_pressure, pressure_path)
        finally:
            del velocity, pressure
            temporary_velocity.unlink(missing_ok=True)
            temporary_pressure.unlink(missing_ok=True)
        q_expected = q_nom_numpy(times, parsed_udf["fourier"]) * a_mesh / a_udf
        q_error = np.asarray(q_observed) - q_expected
        relative_q_error = float(
            np.max(np.abs(q_error))
            / max(float(np.max(np.abs(q_expected))), 1.0e-30)
        )
        # ``BC_Inlet`` is a solver monitor, not the velocity profile truth.
        # Some AG cases differ from the analytic UDF×mesh-area flow by up to
        # about 2%.  V4 therefore uses the parsed UDF contract as truth and
        # keeps this comparison only as a gross-provenance warning Gate.
        if relative_q_error > 3.0e-2:
            raise ValueError(
                f"{case_id}: BC monitor differs from UDF-derived actual inlet flow by >3%"
            )

        # Peak frame alignment proves the registered CSV vector frame and raw
        # gauge are consistent with the frozen full-volume peak sidecar.
        peak_index = steps.index(int(volume["peak_step"]))
        interior_indices = np.flatnonzero(eligible_interior)
        distance, nearest_local = target_tree.query(coords[interior_indices], k=1)
        nearest = strict_indices[np.asarray(nearest_local, dtype=np.int64)]
        velocity_peak = np.load(velocity_path, mmap_mode="r")[peak_index, interior_indices]
        volume_velocity = np.load(files["velocity_m_s"]["path"], mmap_mode="r")[nearest]
        pressure_peak = np.load(pressure_path, mmap_mode="r")[peak_index, interior_indices]
        volume_pressure = np.load(files["pressure_relative_pa"]["path"], mmap_mode="r")[nearest]
        pressure_reference = float(volume["scales"]["pressure_reference_pa"])
        velocity_alignment = float(np.max(np.abs(velocity_peak - volume_velocity)))
        pressure_alignment = float(
            np.max(np.abs((pressure_peak - pressure_reference) - volume_pressure))
        )
        if float(np.max(distance)) > 1.0e-5 or velocity_alignment > 1.0e-4 or pressure_alignment > 0.1:
            raise ValueError(f"{case_id}: registered transient peak does not align with frozen sidecar")

        log_path = _largest_fluent_log(raw_case)
        outlet_labels = _parse_export_outlet_log(log_path, steps)
        label_status = "complete" if outlet_labels is not None else "unavailable"
        if outlet_labels is not None:
            outlet_pressure_path = case_dir / "outlet_label_pressure_pa.npy"
            outlet_mass_path = case_dir / "outlet_label_mass_flow_kg_s.npy"
            _atomic_npy(outlet_pressure_path, outlet_labels["pressure_pa"])
            _atomic_npy(outlet_mass_path, outlet_labels["mass_flow_kg_s"])
            transient_files["outlet_label_pressure_pa"] = {
                "path": str(outlet_pressure_path.resolve()),
                "sha256": sha256_file(outlet_pressure_path),
            }
            transient_files["outlet_label_mass_flow_kg_s"] = {
                "path": str(outlet_mass_path.resolve()),
                "sha256": sha256_file(outlet_mass_path),
            }

        static_paths = {
            "coords": case_dir / "transient_coords.npy",
            "geometry_raw": case_dir / "transient_geometry_raw.npy",
            "is_wall": case_dir / "transient_is_wall.npy",
            "eligible_interior": case_dir / "transient_eligible_interior.npy",
            "steps": case_dir / "transient_steps.npy",
            "times_s": case_dir / "transient_times_s.npy",
            "q_actual_m3_s": case_dir / "transient_q_actual_m3_s.npy",
        }
        for key, array in (
            ("coords", coords),
            ("geometry_raw", geometry_raw),
            ("is_wall", is_wall),
            ("eligible_interior", eligible_interior),
            ("steps", np.asarray(steps, dtype=np.int32)),
            ("times_s", times.astype(np.float32)),
            ("q_actual_m3_s", q_expected.astype(np.float32)),
        ):
            _atomic_npy(static_paths[key], np.asarray(array))
        transient_files.update(
            {
                key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for key, path in static_paths.items()
            }
        )
        transient_files["velocity_m_s"] = {
            "path": str(velocity_path.resolve()),
            "sha256": sha256_file(velocity_path),
        }
        transient_files["pressure_raw_pa"] = {
            "path": str(pressure_path.resolve()),
            "sha256": sha256_file(pressure_path),
        }
        transient_report = {
            "status": "completed",
            "frames": 81,
            "rows": row_count,
            "strict_interior_rows": int(np.sum(~is_wall)),
            "wall_rows": int(np.sum(is_wall)),
            "target_domain_interior_rows": int(np.sum(eligible_interior)),
            "excluded_non_target_interior_rows": int(
                np.sum(~is_wall) - np.sum(eligible_interior)
            ),
            "target_domain_fraction_of_feature_interior": float(
                np.sum(eligible_interior) / max(int(np.sum(~is_wall)), 1)
            ),
            "step_first": steps[0],
            "step_last": steps[-1],
            "solver_step_s": 0.005,
            "field_export_stride": 2,
            "field_interval_s": 0.01,
            "max_static_identity_delta": max_static_delta,
            "max_bc_monitor_vs_udf_relative_difference": relative_q_error,
            "inlet_flow_training_truth": "parsed_udf_q_nom_times_a_mesh_over_a_udf",
            "bc_inlet_monitor_role": "diagnostic_only_not_training_truth",
            "peak_registered_coord_max_distance": float(np.max(distance)),
            "peak_velocity_alignment_max_m_s": velocity_alignment,
            "peak_pressure_alignment_max_pa": pressure_alignment,
            "outlet_label_reference": label_status,
            "outlet_log": (
                {"path": str(log_path), "sha256": sha256_file(log_path)}
                if log_path is not None
                else None
            ),
        }

    mismatch = abs(a_mesh - a_udf) / a_udf
    case_report = {
        "schema_version": 1,
        "route": ROUTE,
        "created_at": utc_now(),
        "canonical_id": case_id,
        "cohort": volume_row["cohort"],
        "role": role,
        "status": "completed",
        "gate_result": "pass",
        "provenance": {
            "volume_manifest": {"path": str(volume_path), "sha256": volume_row["manifest_sha256"]},
            "boundary_manifest": {"path": str(boundary_path), "sha256": boundary_row["boundary_manifest_sha256"]},
            "boundary_face_source": face_source,
            "udf": {"path": str(udf_path), "sha256": sha256_file(udf_path)},
            "bundle": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
        },
        "peak_step": int(volume["peak_step"]),
        "conditions": {
            "outlet_order": list(OUTLET_ORDER),
            "a_in_mesh_m2": a_mesh,
            "a_in_udf_m2": a_udf,
            "mesh_udf_area_relative_error": mismatch,
            "mesh_udf_mismatch": mismatch > 1.0e-3,
            "q_nominal_peak_m3_s": q_peak_nominal,
            "q_actual_peak_m3_s": q_actual_peak,
            "inlet_area_ratio": a_mesh / a_udf,
            "outlet_area_m2": outlet_areas.tolist(),
            "rcr_mass_flow_basis": rcr,
            "bc_vector_raw": bc_vector_raw,
            "outlet_valid_mask": [1, 1, 1, 1],
        },
        "physics_scales": {
            "density_kg_m3": 1060.0,
            "U_c_m_s": q_actual_peak / a_mesh,
            "L_c_m": 2.0 * math.sqrt(a_mesh / math.pi),
            "coordinate_length_m": float(volume["scales"]["length_m"]),
            "rcr_pressure_scale_pa": rcr_pressure_scale.tolist(),
        },
        "pressure": {
            "steady": "case_strict_volume_mean_centered_pa",
            "transient": "raw_fluent_gauge_pa_distal_pressure_zero",
            "rcr_pressure_definition": "predicted_face_area_mean_raw_gauge",
            "steady_reference_pa": float(volume["scales"]["pressure_reference_pa"]),
        },
        "rheology": parsed_udf["rheology"],
        "fourier": parsed_udf["fourier"],
        "transient": transient_report,
        "files": {
            "boundary_faces": {
                "path": str(boundary_faces_path.resolve()),
                "sha256": sha256_file(boundary_faces_path),
            },
            "steady_volume_manifest": {
                "path": str(volume_path.resolve()),
                "sha256": volume_row["manifest_sha256"],
            },
            "transient": transient_files,
        },
    }
    atomic_write_json(case_manifest_path, case_report)
    atomic_write_json(CASE_REPORTS / f"{case_id.replace('/', '__')}.json", case_report)
    return case_report


def _transformed_bc(raw: np.ndarray) -> np.ndarray:
    """Physical BC transform of the frozen 2026-09-05 contract (see ``bc_contract``)."""

    return transform_bc_raw(np.asarray(raw, dtype=np.float64))


def _running_stats(arrays: list[np.ndarray]) -> tuple[int, np.ndarray, np.ndarray]:
    count = 0
    total = None
    total_sq = None
    for array in arrays:
        values = np.asarray(array, dtype=np.float64)
        flat = values.reshape(-1, values.shape[-1])
        if total is None:
            total = np.zeros(flat.shape[1], dtype=np.float64)
            total_sq = np.zeros(flat.shape[1], dtype=np.float64)
        count += len(flat)
        total += flat.sum(axis=0)
        total_sq += np.square(flat).sum(axis=0)
    if not count or total is None or total_sq is None:
        raise ValueError("cannot compute empty statistics")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    return count, mean, np.sqrt(variance)


def _build_stats(case_reports: list[dict[str, Any]], split: dict[str, Any]) -> dict[str, Any]:
    train = [report for report in case_reports if report["role"] == "train"]
    if {report["canonical_id"] for report in train} != set(split["train_cases"]):
        raise ValueError("train-only stats case set does not equal frozen train138")
    bc_raw = np.asarray([report["conditions"]["bc_vector_raw"] for report in train], dtype=np.float64)
    bc_value = _transformed_bc(bc_raw)
    bc_policy = bc_scale_policy(bc_value)  # raises on a near-constant z-scored field

    curvature_values = []
    for report in train:
        path = Path(report["files"]["transient"]["geometry_raw"]["path"])
        geometry = np.load(path, mmap_mode="r")
        eligible = np.load(
            report["files"]["transient"]["eligible_interior"]["path"],
            mmap_mode="r",
        )
        curvature_values.append(
            np.abs(np.asarray(geometry[eligible, 2], dtype=np.float32))
        )
    curvature_clip = float(np.quantile(np.concatenate(curvature_values), 0.99))

    geometry_arrays = []
    velocity_arrays = []
    pressure_arrays = []
    du_dt_rms_terms = []
    rcr_reference_terms = []
    for report in train:
        transient = report["files"]["transient"]
        geometry = np.load(transient["geometry_raw"]["path"], mmap_mode="r")
        eligible = np.load(transient["eligible_interior"]["path"], mmap_mode="r")
        selected = np.asarray(geometry[eligible], dtype=np.float64)
        selected[:, 2] = np.sign(selected[:, 2]) * np.log1p(
            np.minimum(np.abs(selected[:, 2]), curvature_clip)
        )
        geometry_arrays.append(selected)
        velocity = np.load(transient["velocity_m_s"]["path"], mmap_mode="r")[:, eligible]
        pressure = np.load(transient["pressure_raw_pa"]["path"], mmap_mode="r")[:, eligible]
        velocity_arrays.append(np.asarray(velocity))
        pressure_arrays.append(np.asarray(pressure)[..., None])
        du_dt = np.diff(np.asarray(velocity, dtype=np.float64), axis=0) / 0.01
        du_dt_rms_terms.append(np.mean(np.square(du_dt), axis=(0, 1)))
        if "outlet_label_pressure_pa" in transient:
            p = np.load(transient["outlet_label_pressure_pa"]["path"], mmap_mode="r").astype(np.float64)
            m = np.load(transient["outlet_label_mass_flow_kg_s"]["path"], mmap_mode="r").astype(np.float64)
            dp = np.gradient(p, 0.01, axis=0)
            dm = np.gradient(m, 0.01, axis=0)
            rcr = report["conditions"]["rcr_mass_flow_basis"]
            scales = np.asarray(report["physics_scales"]["rcr_pressure_scale_pa"])
            residual = []
            for outlet, item in enumerate(rcr):
                value = (
                    item["R2"] * item["C"] * dp[:, outlet]
                    + p[:, outlet]
                    - (item["R1"] + item["R2"]) * m[:, outlet]
                    - item["R1"] * item["R2"] * item["C"] * dm[:, outlet]
                ) / max(scales[outlet], 1.0e-30)
                residual.append(np.mean(np.square(value)))
            rcr_reference_terms.append(residual)

    geometry_count, geometry_mean, geometry_std = _running_stats(geometry_arrays)
    velocity_count, velocity_mean, velocity_std = _running_stats(velocity_arrays)
    pressure_count, pressure_mean, pressure_std = _running_stats(pressure_arrays)
    steady_stats = json.loads(SOURCE_STEADY_STATS.read_text(encoding="utf-8"))

    # Log-space RCR/area collinearity is descriptive and uses all 173 cases;
    # it never enters standardization or training decisions.
    all_area = np.asarray(
        [report["conditions"]["outlet_area_m2"] for report in case_reports], dtype=np.float64
    )
    all_rcr = np.asarray(
        [
            [[item["R1"], item["R2"], item["C"]] for item in report["conditions"]["rcr_mass_flow_basis"]]
            for report in case_reports
        ],
        dtype=np.float64,
    )
    collinearity = {}
    for parameter_index, name in enumerate(("R1", "R2", "C")):
        rows = []
        for outlet in range(4):
            x = np.log10(all_area[:, outlet])
            y = np.log10(all_rcr[:, outlet, parameter_index])
            slope, intercept = np.polyfit(x, y, 1)
            predicted = slope * x + intercept
            ss_res = float(np.sum(np.square(y - predicted)))
            ss_tot = float(np.sum(np.square(y - y.mean())))
            rows.append(
                {
                    "outlet": OUTLET_ORDER[outlet],
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r2": 1.0 - ss_res / max(ss_tot, 1.0e-30),
                }
            )
        collinearity[name] = rows

    stats = {
        "schema_version": 1,
        "route": ROUTE,
        "created_at": utc_now(),
        "scope": "train138 only except explicitly descriptive 173-case collinearity",
        "split": {"path": str(SPLIT_PATH.resolve()), "sha256": sha256_file(SPLIT_PATH)},
        "train_case_ids": [report["canonical_id"] for report in train],
        "bc": bc_policy,
        "geometry": {
            "names": ["abscissa_norm", "local_radius", "curvature_signed_log1p"],
            "count": geometry_count,
            "mean": geometry_mean.tolist(),
            "std": np.where(geometry_std < 1.0e-12, 1.0, geometry_std).tolist(),
            "curvature_clip_abs": curvature_clip,
            "curvature_clip_quantile": 0.99,
        },
        "steady": {
            "velocity_m_s": steady_stats["velocity_m_s"],
            "pressure_pa": steady_stats["pressure_relative_pa"],
            "source": {"path": str(SOURCE_STEADY_STATS), "sha256": sha256_file(SOURCE_STEADY_STATS)},
        },
        "transient": {
            "velocity_m_s": {
                "count": velocity_count,
                "mean": velocity_mean.tolist(),
                "std": np.where(velocity_std < 1.0e-12, 1.0, velocity_std).tolist(),
            },
            "pressure_pa": {
                "count": pressure_count,
                "mean": pressure_mean.tolist(),
                "std": np.where(pressure_std < 1.0e-12, 1.0, pressure_std).tolist(),
                "gauge": "raw_fluent_gauge_pa_distal_pressure_zero",
            },
        },
        "label_reference": {
            "time_fd_velocity_derivative_rms_m_s2": np.sqrt(
                np.mean(np.asarray(du_dt_rms_terms), axis=0)
            ).tolist(),
            "rcr_discrete_normalized_rms": (
                np.sqrt(np.mean(np.asarray(rcr_reference_terms), axis=0)).tolist()
                if rcr_reference_terms
                else None
            ),
            "rcr_cases_available": len(rcr_reference_terms),
            "spatial_momentum_reference": {
                "status": "sensitivity_band_only",
                "known_k48_k96_point_relative_difference": "approximately 1.0-2.2x",
            },
        },
        "rcr_area_collinearity_all173": collinearity,
    }
    return stats


def build(*, workers: int = 1, resume: bool = False, build_transient: bool = True) -> dict[str, Any]:
    guard_write_path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    guard_write_path(AUDIT_ROOT).mkdir(parents=True, exist_ok=True)
    split = derive_split()
    atomic_write_json(SPLIT_PATH, split)
    volume_map, boundary_map, face_map = _source_maps()
    all_cases = split["train_cases"] + split["test_cases"]
    missing = [
        value for value in all_cases if value not in volume_map or value not in boundary_map
    ]
    if missing:
        raise ValueError(f"source manifests missing cases: {missing[:5]}")
    tasks = [
        {
            "case_id": case_id,
            "role": "train" if case_id in set(split["train_cases"]) else "test",
            "build_transient": build_transient,
            "resume": resume,
            "volume_row": volume_map[case_id],
            "boundary_row": boundary_map[case_id],
            "face_row": face_map.get(case_id),
        }
        for case_id in all_cases
    ]
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            case_reports = list(executor.map(_build_case, tasks))
    else:
        case_reports = [_build_case(task) for task in tasks]
    case_reports.sort(key=lambda report: all_cases.index(report["canonical_id"]))
    if any(report.get("gate_result") != "pass" for report in case_reports):
        raise RuntimeError("one or more V4 case Gates failed")
    if not build_transient:
        raise RuntimeError("formal V4 Gate requires transient caches")
    stats = _build_stats(case_reports, split)
    atomic_write_json(STATS_PATH, stats)
    manifest = {
        "schema_version": 1,
        "route": ROUTE,
        "created_at": utc_now(),
        "status": "completed",
        "gate_result": "pass",
        "split": {"path": str(SPLIT_PATH.resolve()), "sha256": sha256_file(SPLIT_PATH)},
        "stats": {"path": str(STATS_PATH.resolve()), "sha256": sha256_file(STATS_PATH)},
        "source_manifests": {
            "volume": {"path": str(SOURCE_VOLUME_MANIFEST), "sha256": sha256_file(SOURCE_VOLUME_MANIFEST)},
            "boundary": {"path": str(SOURCE_BOUNDARY_MANIFEST), "sha256": sha256_file(SOURCE_BOUNDARY_MANIFEST)},
            "face": {"path": str(SOURCE_FACE_MANIFEST), "sha256": sha256_file(SOURCE_FACE_MANIFEST)},
        },
        "counts": {
            "total": len(case_reports),
            "train": sum(report["role"] == "train" for report in case_reports),
            "test": sum(report["role"] == "test" for report in case_reports),
            "transient_frames_each": 81,
        },
        "cases": [
            {
                "canonical_id": report["canonical_id"],
                "cohort": report["cohort"],
                "role": report["role"],
                "manifest": str((DATA_ROOT / "cases" / report["canonical_id"] / "manifest.json").resolve()),
                "manifest_sha256": sha256_file(DATA_ROOT / "cases" / report["canonical_id"] / "manifest.json"),
            }
            for report in case_reports
        ],
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    mismatch_cases = [
        report["canonical_id"]
        for report in case_reports
        if report["conditions"]["mesh_udf_mismatch"]
    ]
    gate = {
        "schema_version": 1,
        "route": ROUTE,
        "created_at": utc_now(),
        "gate_result": "pass",
        "checks": {
            "cases_173": len(case_reports) == 173,
            "train138_test35": manifest["counts"]["train"] == 138 and manifest["counts"]["test"] == 35,
            "all_81_frames": all(report["transient"]["frames"] == 81 for report in case_reports),
            "cross_time_identity": all(report["transient"]["max_static_identity_delta"] <= 1.0e-6 for report in case_reports),
            "exact_time_mapping": all(report["transient"]["field_interval_s"] == 0.01 for report in case_reports),
            "train_stats_exclude_test35": set(stats["train_case_ids"]) == set(split["train_cases"]),
            "fourier_consistent": all(report["fourier"] == DEFAULT_FOURIER for report in case_reports),
            "rheology_consistent": all(report["rheology"] == RHEOLOGY_REFERENCE for report in case_reports),
            "positive_boundary_conditions": all(
                min(report["conditions"]["bc_vector_raw"]) > 0 for report in case_reports
            ),
            "known_a_udf_a_mesh_mismatch_count_is_6": len(mismatch_cases) == 6,
            "bc_inlet_ratio_fixed_scale": stats["bc"]["scale_policy"][INLET_AREA_RATIO_INDEX] == "fixed_physical",
            "bc_train_abs_z_within_6": max(stats["bc"]["train_abs_z_max"]) <= 6.0 + 1.0e-9,
            "outlet_label_reference_available_train": stats["label_reference"]["rcr_cases_available"] > 0,
        },
        "known_a_udf_a_mesh_mismatch_cases": mismatch_cases,
        "known_mismatch_count": len(mismatch_cases),
        "manifest": {"path": str(MANIFEST_PATH.resolve()), "sha256": sha256_file(MANIFEST_PATH)},
        "stats": {"path": str(STATS_PATH.resolve()), "sha256": sha256_file(STATS_PATH)},
        "split": {"path": str(SPLIT_PATH.resolve()), "sha256": sha256_file(SPLIT_PATH)},
    }
    if not all(gate["checks"].values()):
        gate["gate_result"] = "fail"
    atomic_write_json(GATE_REPORT, gate)
    if gate["gate_result"] != "pass":
        raise RuntimeError("V4 Stage-0 Gate failed")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    result = build(
        workers=max(int(args.workers), 1),
        resume=bool(args.resume),
        build_transient=not bool(args.metadata_only),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
