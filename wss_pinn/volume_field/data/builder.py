"""Build mmap-friendly full-volume peak ``u,v,w,p`` sidecars.

Unlike the historical fixed 8k-near/8k-core sidecar, this contract stores the
entire frozen interior volume.  Training can therefore draw a fresh uniform
sample from every eligible CFD cell at every epoch without rereading large
Fluent ASCII files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wss_pinn.data.raw_io import load_cases, read_interior, step_file
from wss_pinn.data.sidecar import align_raw_interior_to_bundle
from wss_pinn.utils import atomic_write_json, guard_write_path, sha256_file, utc_now

from ..utils import StreamingMoments, atomic_save_npy, iter_slices


SCHEMA_VERSION = 2
GEOMETRY_NAMES = (
    "abscissa_norm",
    "local_radius",
    "curvature_signed_log1p",
)
REQUIRED_BUNDLE_KEYS = {
    "steps",
    "peak_step",
    "transform_frame_version",
    "unit_extent_mismatch",
    "unit_factor",
    "coord_scale",
    "transform_centroid",
    "transform_rotation",
    "int_coords_norm",
    "int_type",
    "int_dist_to_wall",
    "int_abscissa_norm",
    "int_local_radius",
    "int_curvature",
    "wall_coords_norm",
    "wall_abscissa_norm",
    "wall_local_radius",
    "wall_curvature",
}


def _case_dir(root: Path, case: dict[str, Any]) -> Path:
    return root / case["cohort"] / Path(str(case["case_id"]))


def _geometry(abscissa: np.ndarray, radius: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    output = np.column_stack(
        [
            np.asarray(abscissa, dtype=np.float32),
            np.asarray(radius, dtype=np.float32),
            np.sign(curvature) * np.log1p(np.abs(curvature)),
        ]
    ).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("geometry features contain NaN/Inf")
    if np.any(output[:, 1] <= 0):
        raise ValueError("local_radius must be positive")
    return output


def rotate_velocity_to_registered(
    velocity_raw: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """Rotate row-vector velocity components into the registered frame."""
    velocity = np.asarray(velocity_raw, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape [3,3]")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6):
        raise ValueError("velocity rotation is not orthogonal")
    return (velocity @ matrix).astype(np.float32)


def _write_case_arrays(output_dir: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, value in arrays.items():
        path = atomic_save_npy(output_dir / f"{name}.npy", value)
        records[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
        }
    return records


def build_case(
    case: dict[str, Any],
    *,
    sidecar_root: str | Path,
) -> dict[str, Any]:
    """Build one immutable full-volume sidecar and return its aggregate row."""
    root = guard_write_path(sidecar_root)
    output_dir = guard_write_path(_case_dir(root, case))
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise RuntimeError(f"existing case sidecar is incomplete: {manifest_path}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"existing case sidecar schema is stale: {manifest_path}; "
                "archive/rebuild it instead of mixing normalization contracts"
            )
        if payload.get("route") != "volume_uvwp_peak_v1":
            raise RuntimeError(f"existing case sidecar route drift: {manifest_path}")
        if payload.get("canonical_id") != case["canonical_id"]:
            raise RuntimeError(f"case identity drift: {manifest_path}")
        if payload.get("role") != case["role"]:
            raise RuntimeError(f"case role drift: {manifest_path}")
        for record in payload.get("files", {}).values():
            path = Path(record["path"])
            if not path.exists() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"existing case file drift: {path}")
        return {
            "canonical_id": case["canonical_id"],
            "role": case["role"],
            "cohort": case["cohort"],
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
        }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to reuse incomplete sidecar directory: {output_dir}")

    bundle_path = Path(case["bundle_path"]).resolve()
    raw_case_dir = Path(case["raw_case_dir"]).resolve()
    with np.load(bundle_path, allow_pickle=False) as source:
        missing = sorted(REQUIRED_BUNDLE_KEYS.difference(source.files))
        if missing:
            raise KeyError(f"bundle missing volume-field keys: {missing}")
        if str(source["transform_frame_version"].item()) != "stl_landmarks_v4":
            raise ValueError("volume-field route requires stl_landmarks_v4")
        if bool(source["unit_extent_mismatch"].item()):
            raise ValueError("unit_extent_mismatch=true")
        bundle = {name: np.asarray(source[name]) for name in source.files}

    peak_step = int(bundle["peak_step"].item())
    steps = np.asarray(bundle["steps"], dtype=np.int64)
    if peak_step not in set(steps.tolist()):
        raise ValueError("peak_step is not present in bundle steps")
    raw_peak = step_file(raw_case_dir, peak_step, "ascii_in")
    interior = read_interior(raw_peak)
    raw_indices, coordinate_delta, alignment_mode = align_raw_interior_to_bundle(
        interior, bundle
    )

    coords = np.asarray(bundle["int_coords_norm"], dtype=np.float32)
    region = np.asarray(bundle["int_type"], dtype=np.int8)
    distance_to_wall = np.asarray(bundle["int_dist_to_wall"], dtype=np.float32)
    # A few frozen bundles contain exact wall-coordinate duplicates inside the
    # volume table.  They are retained for audit/data provenance but excluded
    # from strict-volume support/query and all PDE residuals.
    volume_is_wall = distance_to_wall <= 1e-6
    if (
        len(coords) != len(raw_indices)
        or len(coords) != len(region)
        or len(coords) != len(volume_is_wall)
    ):
        raise ValueError("interior coordinate/region/alignment length mismatch")
    if not np.isin(region, [0, 1]).all():
        raise ValueError("int_type contains values outside core=0/near-wall=1")
    geometry = _geometry(
        bundle["int_abscissa_norm"],
        bundle["int_local_radius"],
        bundle["int_curvature"],
    )
    wall_coords = np.asarray(bundle["wall_coords_norm"], dtype=np.float32)
    wall_geometry = _geometry(
        bundle["wall_abscissa_norm"],
        bundle["wall_local_radius"],
        bundle["wall_curvature"],
    )
    # Coordinates are stored in the registered WSS-min frame.  Velocity is a
    # vector and must be rotated by the exact same row-vector convention;
    # historical WSS-target sidecars omitted this transform and are therefore
    # not valid supervision for this route.
    velocity_raw = np.asarray(interior["velocity"], dtype=np.float64)[raw_indices]
    velocity = rotate_velocity_to_registered(
        velocity_raw, bundle["transform_rotation"]
    )
    pressure = np.asarray(interior["pressure"], dtype=np.float64)[raw_indices]
    strict_volume = ~volume_is_wall
    if not bool(np.any(strict_volume)):
        raise ValueError("case has no strict-volume rows after wall-duplicate removal")
    pressure_reference = float(
        np.mean(pressure[strict_volume], dtype=np.float64)
    )
    pressure_relative = (pressure - pressure_reference).astype(np.float32)
    required = (coords, geometry, wall_coords, wall_geometry, velocity, pressure_relative)
    if not all(np.isfinite(value).all() for value in required):
        raise ValueError("volume-field sidecar source contains NaN/Inf")
    speed = np.linalg.norm(velocity.astype(np.float64), axis=1)
    if float(np.quantile(speed, 0.95)) <= 1e-6:
        raise ValueError("peak velocity field is degenerate")

    files = _write_case_arrays(
        output_dir,
        {
            "interior_coords": coords,
            "interior_geometry": geometry,
            "interior_region": region,
            "interior_is_wall": volume_is_wall.astype(np.bool_),
            "velocity_m_s": velocity,
            "pressure_relative_pa": pressure_relative,
            "wall_coords": wall_coords,
            "wall_geometry": wall_geometry,
        },
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "completed",
        "route": "volume_uvwp_peak_v1",
        "canonical_id": case["canonical_id"],
        "case_id": case["case_id"],
        "cohort": case["cohort"],
        "role": case["role"],
        "peak_step": peak_step,
        "pressure_target": "case_strict_volume_mean_centered_pa",
        "counts": {
            "interior": int(len(coords)),
            "strict_interior": int(np.sum(~volume_is_wall)),
            "volume_wall_duplicates": int(np.sum(volume_is_wall)),
            "near_wall": int(np.sum(region == 1)),
            "core": int(np.sum(region == 0)),
            "wall": int(len(wall_coords)),
        },
        "geometry_features": list(GEOMETRY_NAMES),
        "boundary_mask": {
            "source": "int_dist_to_wall <= 1e-6 mm",
            "use": (
                "audit/data provenance only; excluded from support/query/PDE; "
                "no-slip uses independent wall_coords"
            ),
        },
        "units": {
            "interior_coords": "dimensionless registered coordinates",
            "velocity_m_s": "m/s",
            "pressure_relative_pa": (
                "Pa; raw Fluent gauge pressure minus the fixed strict-volume case mean"
            ),
            "local_radius": "mm in the registered WSS-min geometry contract",
        },
        "vector_frame": {
            "coordinates": "stl_landmarks_v4 registered frame",
            "velocity": "raw Fluent vector right-multiplied by transform_rotation",
        },
        "scales": {
            "length_m": float(bundle["coord_scale"].item()) * 1e-3,
            "pressure_reference_pa": pressure_reference,
            "density_kg_m3": 1060.0,
        },
        "alignment": {
            "mode": alignment_mode,
            "raw_rows": int(len(interior["coords"])),
            "bundle_rows": int(len(coords)),
            "coord_delta_max": float(np.max(coordinate_delta)),
            "coord_delta_p99": float(np.quantile(coordinate_delta, 0.99)),
        },
        "field_summary": {
            "velocity_speed_p95_m_s": float(np.quantile(speed, 0.95)),
            "velocity_speed_max_m_s": float(np.max(speed)),
            "pressure_relative_min_pa": float(np.min(pressure_relative)),
            "pressure_relative_max_pa": float(np.max(pressure_relative)),
            "pressure_relative_strict_mean_pa": float(
                np.mean(pressure_relative[strict_volume], dtype=np.float64)
            ),
            "pressure_relative_all_rows_mean_pa": float(
                np.mean(pressure_relative, dtype=np.float64)
            ),
        },
        "parents": {
            "bundle": {"path": str(bundle_path), "sha256": sha256_file(bundle_path)},
            "raw_peak": {"path": str(raw_peak.resolve()), "sha256": sha256_file(raw_peak)},
        },
        "files": files,
    }
    written = atomic_write_json(manifest_path, payload)
    return {
        "canonical_id": case["canonical_id"],
        "role": case["role"],
        "cohort": case["cohort"],
        "manifest": str(written.resolve()),
        "manifest_sha256": sha256_file(written),
    }


def compute_train_stats(rows: list[dict[str, Any]], output: str | Path, split: Path) -> Path:
    """Compute normalization from train-only rows eligible for model sampling."""
    geometry = StreamingMoments(3)
    velocity = StreamingMoments(3)
    pressure = StreamingMoments(1)
    speed_sq_sum = 0.0
    speed_count = 0
    train_rows = [row for row in rows if row["role"] == "train"]
    curvature_samples = []
    for row_index, row in enumerate(train_rows):
        manifest = json.loads(Path(row["manifest"]).read_text(encoding="utf-8"))
        geom = np.load(manifest["files"]["interior_geometry"]["path"], mmap_mode="r")
        is_wall = np.load(
            manifest["files"]["interior_is_wall"]["path"], mmap_mode="r"
        )
        eligible = np.flatnonzero(~np.asarray(is_wall, dtype=bool))
        if len(eligible) == 0:
            raise ValueError(f"train case has no strict-volume rows: {row['canonical_id']}")
        take = min(len(eligible), 10_000)
        rng = np.random.default_rng(1234 + row_index * 7919)
        indices = rng.choice(eligible, size=take, replace=False)
        curvature_samples.append(np.abs(np.asarray(geom[indices, 2], dtype=np.float64)))
    curvature_clip = float(np.quantile(np.concatenate(curvature_samples), 0.99))
    if not np.isfinite(curvature_clip) or curvature_clip <= 0:
        raise ValueError("train-only curvature clip is invalid")

    for row in train_rows:
        manifest = json.loads(Path(row["manifest"]).read_text(encoding="utf-8"))
        geom = np.load(manifest["files"]["interior_geometry"]["path"], mmap_mode="r")
        vel = np.load(manifest["files"]["velocity_m_s"]["path"], mmap_mode="r")
        pres = np.load(
            manifest["files"]["pressure_relative_pa"]["path"], mmap_mode="r"
        )
        is_wall = np.load(
            manifest["files"]["interior_is_wall"]["path"], mmap_mode="r"
        )
        for chunk in iter_slices(len(geom)):
            eligible = ~np.asarray(is_wall[chunk], dtype=bool)
            if not bool(np.any(eligible)):
                continue
            geom_block = np.asarray(geom[chunk], dtype=np.float64).copy()
            geom_block[:, 2] = np.clip(
                geom_block[:, 2], -curvature_clip, curvature_clip
            )
            geometry.update(geom_block[eligible])
            velocity.update(np.asarray(vel[chunk])[eligible])
            pressure.update(np.asarray(pres[chunk])[eligible])
            block = np.asarray(vel[chunk], dtype=np.float64)[eligible]
            speed_sq_sum += float(np.square(block).sum(dtype=np.float64))
            speed_count += int(len(block))
    velocity_rms = float(np.sqrt(speed_sq_sum / max(speed_count, 1)))
    if velocity_rms <= 1e-6:
        raise ValueError("global train velocity field is degenerate")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "completed",
        "route": "volume_uvwp_peak_v1",
        "scope": "train138 strict-volume only",
        "population": {
            "roles": ["train"],
            "interior_is_wall": False,
            "wall_duplicate_rows_excluded": True,
        },
        "split": {"path": str(split.resolve()), "sha256": sha256_file(split)},
        "case_manifest_sha256": {
            row["canonical_id"]: row["manifest_sha256"] for row in train_rows
        },
        "geometry": {
            "names": list(GEOMETRY_NAMES),
            "curvature_clip_abs": curvature_clip,
            "curvature_clip_quantile": 0.99,
            **geometry.result(),
        },
        "velocity_m_s": {"names": ["u", "v", "w"], **velocity.result()},
        "pressure_relative_pa": {"names": ["p"], **pressure.result()},
        "train_diagnostics": {"velocity_speed_rms_m_s": velocity_rms},
        "physics_nondimensionalization": {
            "velocity_m_s": 1.0,
            "density_kg_m3": 1060.0,
            "pressure_pa": 1060.0,
        },
    }
    return atomic_write_json(output, payload)


def build_dataset(
    *,
    split_path: str | Path,
    sidecar_root: str | Path,
    aggregate_path: str | Path,
    stats_path: str | Path,
) -> dict[str, Any]:
    """Build or verify all split cases, train-only stats and aggregate manifest."""
    split = Path(split_path).resolve()
    rows = []
    cases = load_cases(split)
    for index, case in enumerate(cases, start=1):
        row = build_case(case, sidecar_root=sidecar_root)
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "volume_sidecar_ready",
                    "index": index,
                    "total": len(cases),
                    "canonical_id": row["canonical_id"],
                    "role": row["role"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    stats = compute_train_stats(rows, stats_path, split)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "completed",
        "route": "volume_uvwp_peak_v1",
        "split": {"path": str(split), "sha256": sha256_file(split)},
        "field_stats": {"path": str(stats.resolve()), "sha256": sha256_file(stats)},
        "cases": rows,
    }
    aggregate = atomic_write_json(aggregate_path, payload)
    payload["path"] = str(aggregate.resolve())
    payload["sha256"] = sha256_file(aggregate)
    return payload
