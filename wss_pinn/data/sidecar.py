from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ..physics.wall_shear import map_stl_normals_to_wall
from ..utils import atomic_write_json, guard_write_path, sha256_file, utc_now
from .raw_io import load_cases, read_interior, step_file
from .sampling import build_three_slot_sampling


def _atomic_savez(path: Path, **arrays: Any) -> None:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _case_output_dir(sidecar_root: Path, case: dict) -> Path:
    parts = [sidecar_root, case["cohort"]]
    relative = str(case["case_id"]).split("/")
    return Path(*parts, *relative)


def _reuse_completed_case(case: dict, sidecar_root: str | Path) -> dict[str, Any] | None:
    output_dir = _case_output_dir(Path(sidecar_root), case)
    manifest_path = output_dir / "physics_manifest.json"
    if not manifest_path.exists():
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                f"incomplete sidecar directory requires review before resume: {output_dir}"
            )
        return None
    manifest = load_sidecar_manifest(manifest_path, verify=True)
    if manifest.get("status") != "completed":
        raise RuntimeError(f"existing sidecar is not completed: {manifest_path}")
    existing_canonical = manifest.get(
        "canonical_id", f"{manifest['cohort']}/{manifest['case_id']}"
    )
    existing_role = manifest.get("role", "unspecified")
    if existing_canonical != case["canonical_id"]:
        raise RuntimeError(f"existing sidecar canonical ID drift: {manifest_path}")
    if existing_role != case["role"]:
        raise RuntimeError(f"existing sidecar role drift: {manifest_path}")
    return {
        "case_id": case["case_id"],
        "canonical_id": case["canonical_id"],
        "cohort": case["cohort"],
        "role": case["role"],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _geometry_descriptor(wall_coords_norm: np.ndarray) -> np.ndarray:
    xyz = np.asarray(wall_coords_norm, dtype=np.float32)
    return np.concatenate(
        [
            xyz.mean(axis=0),
            xyz.std(axis=0),
            xyz.min(axis=0),
            xyz.max(axis=0),
            np.asarray(
                [
                    np.log1p(len(xyz)),
                    np.mean(np.linalg.norm(xyz, axis=1)),
                    np.std(np.linalg.norm(xyz, axis=1)),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def _transform_raw_interior(
    raw_coords: np.ndarray, bundle: dict[str, np.ndarray]
) -> np.ndarray:
    coords_mm = raw_coords * float(bundle["unit_factor"])
    aligned = (coords_mm - bundle["transform_centroid"]) @ bundle["transform_rotation"]
    return aligned / float(bundle["coord_scale"])


def align_raw_interior_to_bundle(
    interior: dict[str, np.ndarray],
    bundle: dict[str, np.ndarray],
    *,
    tolerance: float = 5e-5,
) -> tuple[np.ndarray, np.ndarray, str]:
    raw_aligned_norm = _transform_raw_interior(interior["coords"], bundle)
    bundle_int_coords = np.asarray(bundle["int_coords_norm"], dtype=np.float64)
    if len(raw_aligned_norm) == len(bundle_int_coords):
        direct_delta = np.max(
            np.abs(raw_aligned_norm - bundle_int_coords), axis=1
        )
        if float(np.max(direct_delta)) <= tolerance:
            return (
                np.arange(len(bundle_int_coords), dtype=np.int64),
                direct_delta,
                "direct_row",
            )

    tree = cKDTree(raw_aligned_norm)
    _, raw_indices = tree.query(bundle_int_coords, k=1, workers=-1)
    raw_indices = np.asarray(raw_indices, dtype=np.int64)
    if len(np.unique(raw_indices)) != len(raw_indices):
        raise ValueError("spatial raw/bundle alignment is not one-to-one")
    coordinate_delta = np.max(
        np.abs(raw_aligned_norm[raw_indices] - bundle_int_coords), axis=1
    )
    if float(np.max(coordinate_delta)) > tolerance:
        raise ValueError(
            "raw/bundle spatial-subset alignment drift: "
            f"max={float(np.max(coordinate_delta)):.3e}"
        )
    return raw_indices, coordinate_delta, "spatial_subset"


def build_case_sidecar(
    case: dict,
    *,
    sidecar_root: str | Path,
    wall_points: int,
    near_wall_points: int,
    core_points: int,
    seed: int,
) -> dict[str, Any]:
    raw_case_dir = Path(case["raw_case_dir"]).resolve()
    bundle_path = Path(case["bundle_path"]).resolve()
    output_dir = guard_write_path(_case_output_dir(Path(sidecar_root), case))
    if (output_dir / "physics_manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite sidecar: {output_dir}")

    with np.load(bundle_path, allow_pickle=False) as source:
        bundle = {key: np.asarray(source[key]) for key in source.files}
    peak_step = int(bundle["peak_step"])
    raw_peak_path = step_file(raw_case_dir, peak_step, "ascii_in")
    interior = read_interior(raw_peak_path)
    raw_indices, coordinate_delta, alignment_mode = align_raw_interior_to_bundle(
        interior, bundle
    )

    sampled = build_three_slot_sampling(
        len(bundle["wall_coords_norm"]),
        bundle["int_type"],
        wall_points,
        near_wall_points,
        core_points,
        seed,
    )
    wall_idx = sampled["wall"]
    near_idx = sampled["near_wall"]
    core_idx = sampled["core"]
    stl_path = Path(str(bundle["original_stl_path"].item()))
    wall_normals, normal_map_distance = map_stl_normals_to_wall(
        np.asarray(bundle["wall_coords_raw"])[wall_idx],
        stl_path,
        float(bundle["original_stl_scale_to_mm"]),
        rotation=np.asarray(bundle["transform_rotation"]),
    )
    normal_finite = np.isfinite(wall_normals).all(axis=1)
    if not normal_finite.all():
        raise ValueError("mapped wall normals contain NaN/Inf")

    sampling_payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "case_id": case["case_id"],
        "cohort": case["cohort"],
        "seed": int(seed),
        "slots": {
            "wall": wall_idx.tolist(),
            "near_wall": near_idx.tolist(),
            "core": core_idx.tolist(),
        },
        "candidate_counts": {
            "wall": int(len(bundle["wall_coords_norm"])),
            "near_wall": int(np.sum(bundle["int_type"] == 1)),
            "core": int(np.sum(bundle["int_type"] == 0)),
        },
    }
    sampling_path = atomic_write_json(output_dir / "sampling_manifest.json", sampling_payload)

    static_path = output_dir / "physics_static.npz"
    _atomic_savez(
        static_path,
        wall_indices=wall_idx,
        wall_coords=np.asarray(bundle["wall_coords_norm"])[wall_idx].astype(np.float32),
        wall_coords_mm=(
            np.asarray(bundle["wall_coords_norm"])[wall_idx]
            * float(bundle["coord_scale"])
        ).astype(np.float32),
        wall_normals=wall_normals.astype(np.float32),
        wall_normal_map_distance_mm=normal_map_distance.astype(np.float32),
        near_wall_indices=near_idx,
        near_wall_coords=np.asarray(bundle["int_coords_norm"])[near_idx].astype(np.float32),
        near_wall_dist_mm=np.asarray(bundle["int_dist_to_wall"])[near_idx].astype(np.float32),
        core_indices=core_idx,
        core_coords=np.asarray(bundle["int_coords_norm"])[core_idx].astype(np.float32),
        core_dist_mm=np.asarray(bundle["int_dist_to_wall"])[core_idx].astype(np.float32),
        geometry_descriptor=_geometry_descriptor(bundle["wall_coords_norm"]),
        coord_scale_mm=np.asarray(float(bundle["coord_scale"]), dtype=np.float32),
        unit_factor=np.asarray(float(bundle["unit_factor"]), dtype=np.float64),
        transform_centroid=np.asarray(bundle["transform_centroid"], dtype=np.float64),
        transform_rotation=np.asarray(bundle["transform_rotation"], dtype=np.float64),
    )

    step_index = int(np.flatnonzero(bundle["steps"] == peak_step)[0])
    fields_path = output_dir / "fields" / "step_peak.npz"
    _atomic_savez(
        fields_path,
        step=np.asarray(peak_step, dtype=np.int32),
        wall_wss=np.asarray(bundle["wall_wss"])[step_index, wall_idx].astype(np.float32),
        wall_wss_vector=np.asarray(bundle["wall_wss_vec"])[step_index, wall_idx].astype(
            np.float32
        ),
        wall_pressure=np.asarray(bundle["wall_pressure"])[step_index, wall_idx].astype(
            np.float32
        ),
        near_wall_velocity=np.asarray(interior["velocity"])[
            raw_indices[near_idx]
        ].astype(np.float32),
        near_wall_pressure=np.asarray(interior["pressure"])[
            raw_indices[near_idx]
        ].astype(np.float32),
        core_velocity=np.asarray(interior["velocity"])[
            raw_indices[core_idx]
        ].astype(np.float32),
        core_pressure=np.asarray(interior["pressure"])[
            raw_indices[core_idx]
        ].astype(np.float32),
    )
    all_velocity = np.vstack(
        [
            interior["velocity"][raw_indices[near_idx]],
            interior["velocity"][raw_indices[core_idx]],
        ]
    )
    velocity_scale = float(
        np.quantile(np.linalg.norm(all_velocity, axis=1), 0.95)
    )
    velocity_scale = max(velocity_scale, 1e-6)

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "case_id": case["case_id"],
        "canonical_id": case["canonical_id"],
        "cohort": case["cohort"],
        "role": case["role"],
        "timesteps": {"peak": peak_step, "neighbors_included": False},
        "slots": {
            "wall": int(len(wall_idx)),
            "near_wall": int(len(near_idx)),
            "core": int(len(core_idx)),
        },
        "units": {
            "raw_coordinates": "m (case-specific unit factor audited against geometry)",
            "sidecar_coordinates": "dimensionless registered coordinates",
            "coord_scale_mm": float(bundle["coord_scale"]),
            "velocity": "m/s",
            "pressure": "Pa gauge-offset arbitrary",
            "wss": "Pa",
            "time": "s; Fluent step dt=0.005 s",
        },
        "characteristic_scales": {
            "length_m": float(bundle["coord_scale"]) * 1e-3,
            "velocity_m_s": velocity_scale,
            "density_kg_m3": 1060.0,
        },
        "normal_audit": {
            "source": "original_stl_nearest_face",
            "stl_path": str(stl_path),
            "orientation": "source_facet_winding; scalar WSS insensitive to global flip",
            "finite_fraction": float(normal_finite.mean()),
            "unit_norm_max_error": float(
                np.max(np.abs(np.linalg.norm(wall_normals, axis=1) - 1.0))
            ),
            "map_distance_mm_p50": float(np.median(normal_map_distance)),
            "map_distance_mm_p95": float(np.quantile(normal_map_distance, 0.95)),
        },
        "alignment": {
            "mode": alignment_mode,
            "raw_id_column": interior["id_column"],
            "row_contract": (
                "direct row when equal and coordinate-identical; otherwise unique "
                "raw-to-frozen-transform spatial subset matching"
            ),
            "raw_rows": int(len(interior["coords"])),
            "bundle_rows": int(len(bundle["int_coords_norm"])),
            "coord_delta_max": float(np.max(coordinate_delta)),
            "coord_delta_p99": float(np.quantile(coordinate_delta, 0.99)),
        },
        "parents": {
            "raw_peak": {
                "path": str(raw_peak_path.resolve()),
                "sha256": sha256_file(raw_peak_path),
            },
            "bundle": {
                "path": str(bundle_path),
                "sha256": sha256_file(bundle_path),
            },
            "stl": {"path": str(stl_path), "sha256": sha256_file(stl_path)},
            "sampling_manifest": {
                "path": str(sampling_path),
                "sha256": sha256_file(sampling_path),
            },
        },
        "files": {
            "static": {"path": str(static_path), "sha256": sha256_file(static_path)},
            "peak_fields": {"path": str(fields_path), "sha256": sha256_file(fields_path)},
        },
    }
    manifest_path = atomic_write_json(output_dir / "physics_manifest.json", manifest)
    return {
        "case_id": case["case_id"],
        "canonical_id": case["canonical_id"],
        "cohort": case["cohort"],
        "role": case["role"],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def build_sidecars_from_config(config) -> dict[str, Any]:
    cases = load_cases(config["data"]["split_path"])
    sampling = config["sampling"]
    rows = []
    for index, case in enumerate(cases):
        row = _reuse_completed_case(
            case, config["paths"]["sidecar_root"]
        )
        event = "sidecar_reused"
        if row is None:
            row = build_case_sidecar(
                case,
                sidecar_root=config["paths"]["sidecar_root"],
                wall_points=int(sampling["wall_points"]),
                near_wall_points=int(sampling["near_wall_points"]),
                core_points=int(sampling["core_points"]),
                seed=int(sampling["seed"]) + index * 1000,
            )
            event = "sidecar_completed"
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": event,
                    "index": index + 1,
                    "total": len(cases),
                    "canonical_id": row["canonical_id"],
                    "role": row["role"],
                    "manifest": row["manifest"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    aggregate = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "split_label": config["data"]["split_label"],
        "cases": rows,
    }
    path = atomic_write_json(sampling["manifest_path"], aggregate)
    aggregate["path"] = str(path)
    aggregate["sha256"] = sha256_file(path)
    return aggregate


def load_sidecar_manifest(path: str | Path, verify: bool = True) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if verify:
        for parent in payload.get("parents", {}).values():
            parent_path = Path(parent["path"])
            if not parent_path.exists() or sha256_file(parent_path) != parent["sha256"]:
                raise ValueError(f"sidecar parent drift: {parent_path}")
        for item in payload.get("files", {}).values():
            item_path = Path(item["path"])
            if not item_path.exists() or sha256_file(item_path) != item["sha256"]:
                raise ValueError(f"sidecar file drift: {item_path}")
    return payload
