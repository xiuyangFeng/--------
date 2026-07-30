from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from wss_pinn.data.raw_io import (
    INTERIOR_COLUMNS,
    load_cases,
    read_interior,
    step_file,
)
from wss_pinn.data.sidecar import align_raw_interior_to_bundle
from wss_pinn.utils import atomic_write_json, guard_write_path, sha256_file, utc_now


REQUIRED_BUNDLE_KEYS = {
    "steps",
    "peak_step",
    "unit_extent_mismatch",
    "transform_frame_version",
    "unit_factor",
    "coord_scale",
    "transform_centroid",
    "transform_rotation",
    "original_stl_path",
    "original_stl_scale_to_mm",
    "wall_coords_norm",
    "wall_coords_raw",
    "wall_wss",
    "wall_pressure",
    "wall_wss_vec",
    "int_coords_norm",
    "int_type",
    "int_dist_to_wall",
}
EXPECTED_INTERIOR_COLUMNS = {
    *INTERIOR_COLUMNS["xyz"],
    INTERIOR_COLUMNS["pressure"],
    *INTERIOR_COLUMNS["velocity"],
}


def _finite(value: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(value)).all())


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _header_columns(path: Path) -> set[str]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        header = handle.readline().strip()
    if "," in header:
        return {value.strip() for value in header.split(",")}
    return set(header.split())


def audit_case(
    case: dict[str, Any],
    *,
    deep: bool,
    wall_points: int,
    near_wall_points: int,
    core_points: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "canonical_id": case["canonical_id"],
        "cohort": case["cohort"],
        "role": case["role"],
        "status": "failed",
        "severity": "critical",
    }
    try:
        raw_case_dir = Path(case["raw_case_dir"])
        bundle_path = Path(case["bundle_path"])
        row["raw_case_exists"] = raw_case_dir.is_dir()
        row["bundle_exists"] = bundle_path.is_file()
        if not row["raw_case_exists"] or not row["bundle_exists"]:
            raise FileNotFoundError(
                f"raw={row['raw_case_exists']} bundle={row['bundle_exists']}"
            )

        with np.load(bundle_path, allow_pickle=False) as source:
            missing = sorted(REQUIRED_BUNDLE_KEYS.difference(source.files))
            if missing:
                raise KeyError(f"bundle missing keys: {missing}")
            frame = str(source["transform_frame_version"].item())
            row["frame_version"] = frame
            if frame != "stl_landmarks_v4":
                raise ValueError(f"unexpected frame version: {frame}")
            if bool(source["unit_extent_mismatch"].item()):
                raise ValueError("unit_extent_mismatch=true")
            steps = np.asarray(source["steps"], dtype=np.int64)
            peak_step = int(source["peak_step"].item())
            if peak_step not in set(steps.tolist()):
                raise ValueError("peak_step is absent from steps")
            row["step_count"] = int(len(steps))
            row["peak_step"] = peak_step
            wall_count = int(len(source["wall_coords_norm"]))
            int_type = np.asarray(source["int_type"], dtype=np.int8)
            interior_count = int(len(int_type))
            near_count = int(np.sum(int_type == 1))
            core_count = int(np.sum(int_type == 0))
            invalid_type_count = int(np.sum(~np.isin(int_type, [0, 1])))
            row.update(
                {
                    "wall_candidates": wall_count,
                    "interior_candidates": interior_count,
                    "near_wall_candidates": near_count,
                    "core_candidates": core_count,
                    "invalid_int_type_count": invalid_type_count,
                }
            )
            if (
                wall_count < wall_points
                or near_count < near_wall_points
                or core_count < core_points
                or invalid_type_count
            ):
                raise ValueError("physics slot candidate contract failed")
            stl_path = Path(str(source["original_stl_path"].item()))
            row["stl_exists"] = stl_path.is_file()
            if not row["stl_exists"]:
                raise FileNotFoundError(f"missing original STL: {stl_path}")

            raw_peak_path = step_file(raw_case_dir, peak_step, "ascii_in")
            row["raw_peak_exists"] = raw_peak_path.is_file()
            header_missing = sorted(
                EXPECTED_INTERIOR_COLUMNS.difference(_header_columns(raw_peak_path))
            )
            header_columns = _header_columns(raw_peak_path)
            id_columns = sorted(
                set(INTERIOR_COLUMNS["id_candidates"]).intersection(header_columns)
            )
            row["raw_id_column"] = id_columns[0] if id_columns else ""
            if not id_columns:
                header_missing.append("cellnumber|nodenumber")
            row["raw_header_missing"] = "|".join(header_missing)
            if header_missing:
                raise KeyError(f"raw peak header missing: {header_missing}")

            if deep:
                step_index = int(np.flatnonzero(steps == peak_step)[0])
                required_arrays = (
                    source["wall_coords_norm"],
                    source["wall_coords_raw"],
                    source["wall_wss"][step_index],
                    source["wall_pressure"][step_index],
                    source["wall_wss_vec"][step_index],
                    source["int_coords_norm"],
                    source["int_dist_to_wall"],
                )
                if not all(_finite(value) for value in required_arrays):
                    raise ValueError("bundle contains NaN/Inf in required arrays")
                bundle = {
                    key: np.asarray(source[key])
                    for key in (
                        "unit_factor",
                        "transform_centroid",
                        "transform_rotation",
                        "coord_scale",
                    )
                }
                bundle_int_coords = np.asarray(
                    source["int_coords_norm"], dtype=np.float64
                )
                interior = read_interior(raw_peak_path)
                row["raw_interior_rows"] = int(len(interior["cell_id"]))
                row["raw_id_column"] = interior["id_column"]
                if len(np.unique(interior["cell_id"])) != len(
                    interior["cell_id"]
                ):
                    raise ValueError("raw peak IDs are not unique")
                if not _finite(interior["coords"]) or not _finite(
                    interior["velocity"]
                ) or not _finite(interior["pressure"]):
                    raise ValueError("raw peak contains NaN/Inf")
                raw_indices, coordinate_delta, alignment_mode = (
                    align_raw_interior_to_bundle(interior, {
                        **bundle,
                        "int_coords_norm": bundle_int_coords,
                    })
                )
                row["alignment_mode"] = alignment_mode
                row["matched_raw_rows"] = int(len(raw_indices))
                row["coord_delta_max"] = float(np.max(coordinate_delta))
                row["coord_delta_p99"] = float(np.quantile(coordinate_delta, 0.99))
                if row["coord_delta_max"] > 5e-5:
                    raise ValueError(
                        f"raw/bundle alignment drift: {row['coord_delta_max']:.3e}"
                    )
                speed = np.linalg.norm(interior["velocity"][raw_indices], axis=1)
                row["velocity_p95_m_s"] = float(np.quantile(speed, 0.95))
                row["velocity_max_m_s"] = float(np.max(speed))
                row["pressure_min_pa"] = float(
                    np.min(interior["pressure"][raw_indices])
                )
                row["pressure_max_pa"] = float(
                    np.max(interior["pressure"][raw_indices])
                )
                if row["velocity_p95_m_s"] <= 1e-6:
                    raise ValueError("velocity scale is degenerate")

        row["status"] = "passed"
        row["severity"] = "none"
        row["error"] = ""
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def audit_split(
    split_path: Path,
    *,
    deep: bool,
    wall_points: int,
    near_wall_points: int,
    core_points: int,
    max_cases: int | None = None,
    case_ids: set[str] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    cases = load_cases(split_path)
    if case_ids is not None:
        cases = [
            case for case in cases if str(case["canonical_id"]) in case_ids
        ]
        missing_requested = sorted(
            case_ids.difference(
                str(case["canonical_id"]) for case in cases
            )
        )
        if missing_requested:
            raise ValueError(f"requested case IDs absent from split: {missing_requested}")
    if max_cases is not None:
        cases = cases[: int(max_cases)]
    rows = []
    for index, case in enumerate(cases, start=1):
        row = audit_case(
            case,
            deep=deep,
            wall_points=wall_points,
            near_wall_points=near_wall_points,
            core_points=core_points,
        )
        rows.append(row)
        if progress:
            print(
                json.dumps(
                    {
                        "event": "case_audited",
                        "index": index,
                        "total": len(cases),
                        "canonical_id": row["canonical_id"],
                        "status": row["status"],
                        "error": row["error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    passed = [row for row in rows if row["status"] == "passed"]
    failed = [row for row in rows if row["status"] != "passed"]
    role_counts = Counter(row["role"] for row in rows)
    cohort_counts = Counter(row["cohort"] for row in rows)
    duplicate_groups = split_payload.get("duplicate_geometry_groups", [])
    role_by_id = {row["canonical_id"]: row["role"] for row in rows}
    group_violations = [
        group
        for group in duplicate_groups
        if len({role_by_id.get(case_id) for case_id in group}) > 1
    ]
    expected = split_payload.get("expected_counts", {})
    expected_ok = not max_cases and case_ids is None and (
        (not expected)
        or (
            role_counts.get("train", 0) == int(expected.get("train", 0))
            and role_counts.get("test", 0) == int(expected.get("test", 0))
        )
    )
    if max_cases is not None or case_ids is not None:
        expected_ok = True
    gate_passed = not failed and not group_violations and expected_ok
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "mode": "deep" if deep else "metadata",
        "gate_result": "pass" if gate_passed else "fail",
        "intended_use": "WSS-PINN grouped train138/test36 full-dataset training",
        "grain": "one canonical patient/phase unit per row",
        "split": {
            "path": str(split_path.resolve()),
            "sha256": sha256_file(split_path),
            "split_version": split_payload.get("split_version"),
            "expected_counts": expected,
            "actual_role_counts": dict(role_counts),
            "cohort_counts": dict(cohort_counts),
            "duplicate_group_violations": group_violations,
            "expected_counts_passed": expected_ok,
        },
        "thresholds": {
            "frame_version": "stl_landmarks_v4",
            "wall_candidates_min": wall_points,
            "near_wall_candidates_min": near_wall_points,
            "core_candidates_min": core_points,
            "coord_delta_max": 5e-5,
            "finite_required": True,
            "unique_peak_cell_ids": True,
        },
        "summary": {
            "total": len(rows),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": len(passed) / max(len(rows), 1),
            "failed_cases": [
                {
                    "canonical_id": row["canonical_id"],
                    "severity": row["severity"],
                    "error": row["error"],
                }
                for row in failed
            ],
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("metadata", "deep"), default="deep")
    parser.add_argument("--wall-points", type=int, default=5000)
    parser.add_argument("--near-wall-points", type=int, default=8000)
    parser.add_argument("--core-points", type=int, default=8000)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--case-ids", nargs="+")
    args = parser.parse_args()
    output_dir = guard_write_path(args.output_dir)
    report = audit_split(
        Path(args.split),
        deep=args.mode == "deep",
        wall_points=args.wall_points,
        near_wall_points=args.near_wall_points,
        core_points=args.core_points,
        max_cases=args.max_cases,
        case_ids=set(args.case_ids) if args.case_ids else None,
        progress=True,
    )
    json_path = atomic_write_json(output_dir / "report.json", report)
    csv_path = _atomic_write_csv(output_dir / "cases.csv", report["rows"])
    print(
        json.dumps(
            {
                "status": report["status"],
                "gate_result": report["gate_result"],
                "summary": report["summary"],
                "report": str(json_path),
                "cases_csv": str(csv_path),
            },
            ensure_ascii=False,
        )
    )
    if report["gate_result"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
