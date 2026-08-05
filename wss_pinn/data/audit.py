"""Hard Gate for the full-volume uvwp sidecar contract."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from wss_pinn.data.raw_io import load_cases, read_interior
from wss_pinn.data.alignment import align_raw_interior_to_bundle
from wss_pinn.utils import sha256_file, utc_now

from wss_pinn.volume_utils import iter_slices
from .builder import GEOMETRY_NAMES, SCHEMA_VERSION


REQUIRED_FILES = {
    "interior_coords",
    "interior_geometry",
    "interior_region",
    "interior_is_wall",
    "velocity_m_s",
    "pressure_relative_pa",
    "wall_coords",
    "wall_geometry",
}


def _finite_mmap(array: np.ndarray) -> bool:
    return all(np.isfinite(np.asarray(array[chunk])).all() for chunk in iter_slices(len(array)))


def _valid_stats_vector(section: dict[str, Any], width: int) -> bool:
    try:
        mean = np.asarray(section.get("mean", []), dtype=np.float64)
        std = np.asarray(section.get("std", []), dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        mean.shape == (width,)
        and std.shape == (width,)
        and np.isfinite(mean).all()
        and np.isfinite(std).all()
        and np.all(std > 0)
    )


def _positive_finite_scalar(value: Any) -> bool:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(scalar) and scalar > 0)


def audit_case(
    row: dict[str, Any],
    *,
    expected_role: str,
    deep_source: bool,
    sample_seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "canonical_id": row.get("canonical_id"),
        "role": row.get("role"),
        "status": "failed",
        "severity": "critical",
        "error": "",
    }
    try:
        manifest_path = Path(row["manifest"])
        if sha256_file(manifest_path) != row["manifest_sha256"]:
            raise ValueError("aggregate-to-case manifest hash drift")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        canonical_id = str(manifest["canonical_id"])
        report["canonical_id"] = canonical_id
        if canonical_id != row["canonical_id"]:
            raise ValueError("aggregate/case canonical ID mismatch")
        if manifest["role"] != expected_role or row["role"] != expected_role:
            raise ValueError("split/aggregate/case role mismatch")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("case sidecar schema drift")
        if manifest.get("route") != "volume_uvwp_peak_v1":
            raise ValueError("unexpected sidecar route")
        if manifest.get("pressure_target") != "case_strict_volume_mean_centered_pa":
            raise ValueError("pressure target/gauge contract drift")
        if manifest.get("geometry_features") != list(GEOMETRY_NAMES):
            raise ValueError("geometry feature contract drift")
        if set(manifest.get("files", {})) != REQUIRED_FILES:
            raise KeyError("case file set does not exactly match the v1 contract")
        for parent in manifest["parents"].values():
            path = Path(parent["path"])
            if not path.exists() or sha256_file(path) != parent["sha256"]:
                raise ValueError(f"parent hash drift: {path}")
        arrays = {}
        for name, record in manifest["files"].items():
            path = Path(record["path"])
            if not path.exists() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"sidecar file hash drift: {path}")
            array = np.load(path, mmap_mode="r")
            if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
                raise ValueError(f"sidecar shape/dtype drift: {name}")
            if not _finite_mmap(array):
                raise ValueError(f"sidecar contains NaN/Inf: {name}")
            arrays[name] = array

        n = len(arrays["interior_coords"])
        for name in (
            "interior_geometry",
            "interior_region",
            "interior_is_wall",
            "velocity_m_s",
            "pressure_relative_pa",
        ):
            if len(arrays[name]) != n:
                raise ValueError(f"interior length mismatch: {name}")
        if arrays["interior_coords"].shape[1:] != (3,):
            raise ValueError("interior_coords must have shape [N,3]")
        if arrays["interior_geometry"].shape[1:] != (3,):
            raise ValueError("interior_geometry must have shape [N,3]")
        if arrays["velocity_m_s"].shape[1:] != (3,):
            raise ValueError("velocity must have shape [N,3]")
        region = np.asarray(arrays["interior_region"])
        if not np.isin(region, [0, 1]).all():
            raise ValueError("interior_region contains invalid values")
        is_wall = np.asarray(arrays["interior_is_wall"], dtype=bool)
        strict_count = int(np.sum(~is_wall))
        if strict_count < 10_000:
            raise ValueError("strict interior cannot supply independent 5k+5k samples")
        if len(arrays["wall_coords"]) < 1024:
            raise ValueError("wall cloud cannot supply the no-slip batch")
        counts = manifest.get("counts", {})
        expected_counts = {
            "interior": n,
            "strict_interior": strict_count,
            "volume_wall_duplicates": int(np.sum(is_wall)),
            "near_wall": int(np.sum(region == 1)),
            "core": int(np.sum(region == 0)),
            "wall": int(len(arrays["wall_coords"])),
        }
        if counts != expected_counts:
            raise ValueError("manifest count summary drift")
        length_m = float(manifest.get("scales", {}).get("length_m", 0.0))
        if not np.isfinite(length_m) or length_m <= 0:
            raise ValueError("invalid case geometry length scale")
        if manifest.get("scales", {}).get("density_kg_m3") != 1060.0:
            raise ValueError("case density contract drift")
        if manifest.get("vector_frame", {}).get("velocity") != (
            "raw Fluent vector right-multiplied by transform_rotation"
        ):
            raise ValueError("registered velocity-frame contract drift")
        geometry = arrays["interior_geometry"]
        abscissa_min = min(float(np.min(geometry[s, 0])) for s in iter_slices(n))
        abscissa_max = max(float(np.max(geometry[s, 0])) for s in iter_slices(n))
        radius_min = min(float(np.min(geometry[s, 1])) for s in iter_slices(n))
        if abscissa_min < -1e-6 or abscissa_max > 1.0 + 1e-6:
            raise ValueError("abscissa is outside [0,1]")
        if radius_min <= 0:
            raise ValueError("local_radius is non-positive")
        pressure_mean = float(
            sum(
                np.asarray(arrays["pressure_relative_pa"][s], dtype=np.float64)[
                    ~np.asarray(arrays["interior_is_wall"][s], dtype=bool)
                ].sum()
                for s in iter_slices(n)
            )
            / strict_count
        )
        if abs(pressure_mean) > 1e-2:
            raise ValueError(f"strict-volume pressure gauge mean drift: {pressure_mean}")
        speed = np.linalg.norm(np.asarray(arrays["velocity_m_s"]), axis=1)
        speed_p95 = float(np.quantile(speed[~is_wall], 0.95))
        if speed_p95 <= 1e-6:
            raise ValueError("velocity field is degenerate")
        volume_wall_speed_max = (
            float(np.max(speed[is_wall])) if bool(np.any(is_wall)) else 0.0
        )
        if volume_wall_speed_max > 1e-5:
            raise ValueError(
                f"volume wall-duplicate velocity violates no-slip: {volume_wall_speed_max}"
            )

        if deep_source:
            bundle_path = Path(manifest["parents"]["bundle"]["path"])
            raw_peak = Path(manifest["parents"]["raw_peak"]["path"])
            with np.load(bundle_path, allow_pickle=False) as source:
                bundle = {name: np.asarray(source[name]) for name in source.files}
            raw = read_interior(raw_peak)
            raw_indices, delta, mode = align_raw_interior_to_bundle(raw, bundle)
            if mode != manifest["alignment"]["mode"]:
                raise ValueError("source alignment mode drift")
            rng = np.random.default_rng(int(sample_seed))
            selected = np.sort(rng.choice(n, size=min(n, 2048), replace=False))
            rotation = np.asarray(bundle["transform_rotation"], dtype=np.float64)
            expected_velocity = np.asarray(raw["velocity"], dtype=np.float64)[
                raw_indices[selected]
            ] @ rotation
            velocity_error = float(
                np.max(
                    np.abs(
                        expected_velocity
                        - np.asarray(arrays["velocity_m_s"][selected], dtype=np.float64)
                    )
                )
            )
            if velocity_error > 1e-6:
                raise ValueError(f"registered velocity mismatch: {velocity_error:.3e}")
            expected_pressure = np.asarray(raw["pressure"], dtype=np.float64)[raw_indices]
            expected_pressure -= float(manifest["scales"]["pressure_reference_pa"])
            pressure_error = float(
                np.max(
                    np.abs(
                        expected_pressure[selected]
                        - np.asarray(
                            arrays["pressure_relative_pa"][selected], dtype=np.float64
                        )
                    )
                )
            )
            if pressure_error > 2e-3:
                raise ValueError(f"relative pressure mismatch: {pressure_error:.3e}")
            report.update(
                {
                    "deep_velocity_max_error_m_s": velocity_error,
                    "deep_pressure_max_error_pa": pressure_error,
                    "deep_coord_delta_max": float(np.max(delta)),
                }
            )

        report.update(
            {
                "status": "passed",
                "severity": "none",
                "interior_rows": n,
                "strict_interior_rows": strict_count,
                "volume_wall_duplicate_rows": int(np.sum(is_wall)),
                "wall_rows": int(len(arrays["wall_coords"])),
                "velocity_speed_p95_m_s": speed_p95,
                "volume_wall_speed_max_m_s": volume_wall_speed_max,
                "pressure_relative_mean_pa": pressure_mean,
                "error": "",
            }
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def audit_dataset(
    aggregate_path: str | Path,
    split_path: str | Path,
    stats_path: str | Path,
    *,
    deep_source: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    aggregate_path = Path(aggregate_path)
    split_path = Path(split_path)
    stats_path = Path(stats_path)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    expected_cases = load_cases(split_path)
    expected = {case["canonical_id"]: case["role"] for case in expected_cases}
    aggregate_rows = aggregate.get("cases", [])
    invalid_row_indices = [
        index
        for index, row in enumerate(aggregate_rows)
        if not isinstance(row.get("canonical_id"), str) or not row["canonical_id"]
    ]
    aggregate_ids = [
        row["canonical_id"]
        for row in aggregate_rows
        if isinstance(row.get("canonical_id"), str) and row["canonical_id"]
    ]
    duplicate_ids = sorted(
        case_id for case_id, count in Counter(aggregate_ids).items() if count > 1
    )
    rows_by_id = {
        row["canonical_id"]: row
        for row in aggregate_rows
        if isinstance(row.get("canonical_id"), str) and row["canonical_id"]
    }
    reports = []
    for index, canonical_id in enumerate(sorted(expected), start=1):
        row = rows_by_id.get(canonical_id, {"canonical_id": canonical_id})
        report = audit_case(
            row,
            expected_role=expected[canonical_id],
            deep_source=deep_source,
            sample_seed=1234 + index * 97,
        )
        reports.append(report)
        if progress:
            print(
                json.dumps(
                    {
                        "event": "volume_case_audited",
                        "index": index,
                        "total": len(expected),
                        "canonical_id": canonical_id,
                        "status": report["status"],
                        "error": report["error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    unexpected = sorted(set(rows_by_id).difference(expected))
    missing = sorted(set(expected).difference(rows_by_id))
    failed = [row for row in reports if row["status"] != "passed"]

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_scope_ok = stats.get("scope") == "train138 strict-volume only"
    train_ids = {case_id for case_id, role in expected.items() if role == "train"}
    stats_manifest_hashes = stats.get("case_manifest_sha256", {})
    stats_ids = set(stats_manifest_hashes)
    stats_membership_ok = stats_ids == train_ids
    stats_split_ok = stats.get("split", {}).get("sha256") == sha256_file(split_path)
    stats_contract_ok = (
        stats.get("schema_version") == SCHEMA_VERSION
        and aggregate.get("schema_version") == SCHEMA_VERSION
        and stats.get("status") == "completed"
        and stats.get("route") == "volume_uvwp_peak_v1"
        and stats_scope_ok
        and stats.get("population", {}).get("interior_is_wall") is False
        and stats.get("geometry", {}).get("names") == list(GEOMETRY_NAMES)
        and _valid_stats_vector(stats.get("geometry", {}), 3)
        and _positive_finite_scalar(
            stats.get("geometry", {}).get("curvature_clip_abs")
        )
        and _valid_stats_vector(stats.get("velocity_m_s", {}), 3)
        and _valid_stats_vector(stats.get("pressure_relative_pa", {}), 1)
        and stats.get("physics_nondimensionalization", {})
        == {
            "velocity_m_s": 1.0,
            "density_kg_m3": 1060.0,
            "pressure_pa": 1060.0,
        }
    )
    stats_manifest_hashes_ok = stats_membership_ok and all(
        rows_by_id.get(case_id, {}).get("manifest_sha256") == manifest_sha256
        for case_id, manifest_sha256 in stats_manifest_hashes.items()
    )
    split_sha256 = sha256_file(split_path)
    aggregate_split_ok = aggregate.get("split", {}).get("sha256") == split_sha256
    aggregate_stats_ok = (
        aggregate.get("field_stats", {}).get("sha256") == sha256_file(stats_path)
        and Path(aggregate.get("field_stats", {}).get("path", "")).resolve()
        == stats_path.resolve()
    )
    role_counts = Counter(report.get("role") for report in reports)
    gate = (
        aggregate.get("status") == "completed"
        and aggregate.get("route") == "volume_uvwp_peak_v1"
        and aggregate_split_ok
        and aggregate_stats_ok
        and not invalid_row_indices
        and not duplicate_ids
        and not missing
        and not unexpected
        and not failed
        and stats_contract_ok
        and stats_membership_ok
        and stats_manifest_hashes_ok
        and stats_split_ok
        and role_counts == Counter(expected.values())
    )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "route": "volume_uvwp_peak_v1",
        "mode": "deep_source" if deep_source else "sidecar",
        "gate_result": "pass" if gate else "fail",
        "split": {
            "path": str(split_path.resolve()),
            "sha256": split_sha256,
            "expected": len(expected),
            "missing": missing,
            "unexpected": unexpected,
            "invalid_aggregate_row_indices": invalid_row_indices,
            "duplicate_ids": duplicate_ids,
            "aggregate_binding_passed": aggregate_split_ok,
            "role_counts": dict(role_counts),
        },
        "field_stats": {
            "path": str(stats_path.resolve()),
            "sha256": sha256_file(stats_path),
            "contract_passed": stats_contract_ok,
            "train_membership_passed": stats_membership_ok,
            "train_manifest_hashes_passed": stats_manifest_hashes_ok,
            "split_binding_passed": stats_split_ok,
            "aggregate_binding_passed": aggregate_stats_ok,
        },
        "summary": {
            "total": len(reports),
            "passed": len(reports) - len(failed),
            "failed": len(failed),
            "failed_cases": [
                {"canonical_id": row["canonical_id"], "error": row["error"]}
                for row in failed
            ],
        },
        "rows": reports,
    }
