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

from wss_pinn.data.raw_io import load_cases
from wss_pinn.data.sidecar import load_sidecar_manifest
from wss_pinn.utils import atomic_write_json, guard_write_path, sha256_file, utc_now


STATIC_FIELDS = {
    "wall_coords",
    "wall_normals",
    "near_wall_coords",
    "core_coords",
    "geometry_descriptor",
}
FIELD_FIELDS = {
    "wall_wss",
    "near_wall_velocity",
    "near_wall_pressure",
    "core_velocity",
    "core_pressure",
}


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
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


def audit(
    aggregate_path: Path,
    split_path: Path,
    *,
    wall_points: int,
    near_wall_points: int,
    core_points: int,
    progress: bool = False,
) -> dict[str, Any]:
    expected_cases = load_cases(split_path)
    expected = {
        str(case["canonical_id"]): str(case["role"]) for case in expected_cases
    }
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    rows = []
    seen: set[str] = set()
    aggregate_cases = aggregate.get("cases", [])
    for index, item in enumerate(aggregate_cases, start=1):
        row: dict[str, Any] = {
            "status": "failed",
            "severity": "critical",
            "manifest": str(item.get("manifest", "")),
        }
        try:
            manifest_path = Path(item["manifest"])
            if sha256_file(manifest_path) != item["manifest_sha256"]:
                raise ValueError("aggregate-to-case manifest hash drift")
            manifest = load_sidecar_manifest(manifest_path, verify=True)
            canonical_id = str(
                item.get(
                    "canonical_id",
                    manifest.get(
                        "canonical_id",
                        f"{manifest['cohort']}/{manifest['case_id']}",
                    ),
                )
            )
            role = str(item.get("role", manifest.get("role", "unspecified")))
            row.update({"canonical_id": canonical_id, "role": role})
            if canonical_id in seen:
                raise ValueError("duplicate canonical ID in aggregate manifest")
            seen.add(canonical_id)
            if expected.get(canonical_id) != role:
                raise ValueError(
                    f"split role mismatch: expected={expected.get(canonical_id)!r} "
                    f"actual={role!r}"
                )
            if (
                "canonical_id" in manifest
                and manifest.get("canonical_id") != canonical_id
            ):
                raise ValueError("aggregate/case canonical ID mismatch")
            if "role" in manifest and manifest.get("role") != role:
                raise ValueError("aggregate/case role mismatch")
            slots = manifest["slots"]
            actual_slots = {
                "wall": int(slots["wall"]),
                "near_wall": int(slots["near_wall"]),
                "core": int(slots["core"]),
            }
            row.update({f"{key}_points": value for key, value in actual_slots.items()})
            required_slots = {
                "wall": wall_points,
                "near_wall": near_wall_points,
                "core": core_points,
            }
            if actual_slots != required_slots:
                raise ValueError(
                    f"sidecar slot mismatch: {actual_slots} != {required_slots}"
                )
            with np.load(
                manifest["files"]["static"]["path"], allow_pickle=False
            ) as static:
                missing = STATIC_FIELDS.difference(static.files)
                if missing:
                    raise KeyError(f"static sidecar missing fields: {sorted(missing)}")
                static_arrays = [np.asarray(static[key]) for key in STATIC_FIELDS]
                normals = np.asarray(static["wall_normals"])
                if len(static["wall_coords"]) != wall_points:
                    raise ValueError("wall static row count mismatch")
                if len(static["near_wall_coords"]) != near_wall_points:
                    raise ValueError("near-wall static row count mismatch")
                if len(static["core_coords"]) != core_points:
                    raise ValueError("core static row count mismatch")
                normal_error = float(
                    np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0))
                )
                row["normal_unit_max_error"] = normal_error
                if normal_error > 1e-5:
                    raise ValueError(f"wall normals are not unit: {normal_error:.3e}")
                if not all(np.isfinite(value).all() for value in static_arrays):
                    raise ValueError("static sidecar contains NaN/Inf")
            with np.load(
                manifest["files"]["peak_fields"]["path"], allow_pickle=False
            ) as fields:
                missing = FIELD_FIELDS.difference(fields.files)
                if missing:
                    raise KeyError(f"field sidecar missing fields: {sorted(missing)}")
                arrays = [np.asarray(fields[key]) for key in FIELD_FIELDS]
                if len(fields["wall_wss"]) != wall_points:
                    raise ValueError("wall WSS row count mismatch")
                if len(fields["near_wall_velocity"]) != near_wall_points:
                    raise ValueError("near-wall field row count mismatch")
                if len(fields["core_velocity"]) != core_points:
                    raise ValueError("core field row count mismatch")
                if not all(np.isfinite(value).all() for value in arrays):
                    raise ValueError("field sidecar contains NaN/Inf")
                speed = np.linalg.norm(
                    np.vstack(
                        [
                            fields["near_wall_velocity"],
                            fields["core_velocity"],
                        ]
                    ),
                    axis=1,
                )
                row["velocity_p95_m_s"] = float(np.quantile(speed, 0.95))
                if row["velocity_p95_m_s"] <= 1e-6:
                    raise ValueError("sidecar velocity scale is degenerate")
            if float(manifest["alignment"]["coord_delta_max"]) > 5e-5:
                raise ValueError("sidecar alignment contract failed")
            row.update({"status": "passed", "severity": "none", "error": ""})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if progress:
            print(
                json.dumps(
                    {
                        "event": "sidecar_audited",
                        "index": index,
                        "total": len(aggregate_cases),
                        "canonical_id": row.get("canonical_id"),
                        "status": row["status"],
                        "error": row["error"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    missing_ids = sorted(set(expected).difference(seen))
    unexpected_ids = sorted(seen.difference(expected))
    failed = [row for row in rows if row["status"] != "passed"]
    role_counts = Counter(row.get("role") for row in rows)
    gate_passed = (
        aggregate.get("status") == "completed"
        and not failed
        and not missing_ids
        and not unexpected_ids
        and role_counts == Counter(expected.values())
    )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "gate_result": "pass" if gate_passed else "fail",
        "intended_use": "WSS-PINN grouped full-dataset training/evaluation",
        "grain": "one immutable physics sidecar per canonical patient/phase unit",
        "aggregate_manifest": {
            "path": str(aggregate_path.resolve()),
            "sha256": sha256_file(aggregate_path),
            "status": aggregate.get("status"),
        },
        "split": {
            "path": str(split_path.resolve()),
            "sha256": sha256_file(split_path),
            "expected_total": len(expected),
            "missing_ids": missing_ids,
            "unexpected_ids": unexpected_ids,
            "actual_role_counts": dict(role_counts),
        },
        "thresholds": {
            "slots": {
                "wall": wall_points,
                "near_wall": near_wall_points,
                "core": core_points,
            },
            "normal_unit_max_error": 1e-5,
            "coord_delta_max": 5e-5,
            "finite_required": True,
            "parent_and_file_hashes_required": True,
        },
        "summary": {
            "total": len(rows),
            "passed": len(rows) - len(failed),
            "failed": len(failed),
            "pass_rate": (len(rows) - len(failed)) / max(len(rows), 1),
            "failed_cases": [
                {
                    "canonical_id": row.get("canonical_id"),
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wall-points", type=int, default=5000)
    parser.add_argument("--near-wall-points", type=int, default=8000)
    parser.add_argument("--core-points", type=int, default=8000)
    args = parser.parse_args()
    output_dir = guard_write_path(args.output_dir)
    report = audit(
        Path(args.manifest),
        Path(args.split),
        wall_points=args.wall_points,
        near_wall_points=args.near_wall_points,
        core_points=args.core_points,
        progress=True,
    )
    report_path = atomic_write_json(output_dir / "report.json", report)
    cases_path = _atomic_csv(output_dir / "cases.csv", report["rows"])
    print(
        json.dumps(
            {
                "status": "completed",
                "gate_result": report["gate_result"],
                "summary": report["summary"],
                "report": str(report_path),
                "cases_csv": str(cases_path),
            },
            ensure_ascii=False,
        )
    )
    if report["gate_result"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
