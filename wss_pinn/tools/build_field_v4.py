"""Build the independent train123/val15 data contract for field-v4.

The tool never copies volume arrays.  It binds the immutable V1/V3 assets by
SHA256, drops the development-exposed test35 rows, and materialises one fixed
5000-point validation query per val case.  Prediction-only query coordinate
files are separate from the index files used by the truth evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from wss_pinn.data.dataset import _take_strict_volume
from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    git_state,
    guard_write_path,
    sha256_file,
    sha256_json,
    utc_now,
)


ROUTE = "volume_uvwp_peak_field_v4"
SOURCE_ROOT = ROOT / "data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35"
SOURCE_MANIFEST = SOURCE_ROOT / "manifest.json"
SOURCE_STATS = SOURCE_ROOT / "field_stats.json"
OUTPUT_ROOT = ROOT / "data_wss_pinn/volume_uvwp_peak_field_v4_train123_val15"
OUTPUT_MANIFEST = OUTPUT_ROOT / "manifest.json"
QUERY_MANIFEST = OUTPUT_ROOT / "validation_queries/manifest.json"
AUDIT_OUTPUT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_data/report.json"
SPLIT_SHA256 = "c80cb65ad95f7d76fadaff82ea02c35c27ad24ab2474e1050bf981eff97493f9"


def _case_seed(case_id: str, seed: int) -> int:
    identity = int.from_bytes(
        hashlib.sha256(case_id.encode("utf-8")).digest()[:8], "little"
    )
    return (int(seed) + identity) % (2**63 - 1)


def _atomic_save(path: Path, value: np.ndarray) -> None:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, value, allow_pickle=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _volume_manifest(row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(row["source_volume_manifest"])
    if sha256_file(path) != row.get("source_volume_manifest_sha256"):
        raise ValueError(f"source volume manifest hash drift: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("route") != "volume_uvwp_peak_v1"
        or payload.get("status") != "completed"
        or payload.get("canonical_id") != row.get("canonical_id")
    ):
        raise ValueError(f"source volume manifest contract drift: {path}")
    return path, payload


def build(*, seed: int = 1234, query_points: int = 5000, force: bool = False) -> dict[str, Any]:
    if int(query_points) != 5000:
        raise ValueError("field-v4 development validation is frozen to 5000 points/case")
    if OUTPUT_MANIFEST.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing v4 data contract: {OUTPUT_MANIFEST}")

    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    stats = json.loads(SOURCE_STATS.read_text(encoding="utf-8"))
    if (
        source.get("route") != "volume_uvwp_peak_qs_smooth_v3"
        or source.get("status") != "completed"
        or source.get("split", {}).get("sha256") != SPLIT_SHA256
        or stats.get("route") != "volume_uvwp_peak_qs_smooth_v3"
        or stats.get("scope") != "train123 strict-volume only"
        or stats.get("split", {}).get("sha256") != SPLIT_SHA256
    ):
        raise ValueError("frozen V3 source data/stats contract drift")
    if source.get("field_stats", {}).get("sha256") != sha256_file(SOURCE_STATS):
        raise ValueError("V3 manifest-to-field-stats binding drift")

    rows = []
    query_rows = []
    source_roles = {role: 0 for role in ("train", "val", "test")}
    for row in source.get("cases", []):
        role = str(row.get("role"))
        source_roles[role] = source_roles.get(role, 0) + 1
        if role == "test":
            continue
        if role not in {"train", "val"}:
            raise ValueError(f"unexpected source role={role!r}")
        manifest_path, volume = _volume_manifest(row)
        output_row: dict[str, Any] = {
            "canonical_id": row["canonical_id"],
            "cohort": row["cohort"],
            "role": role,
            "source_volume_manifest": str(manifest_path.resolve()),
            "source_volume_manifest_sha256": sha256_file(manifest_path),
        }
        if role == "val":
            files = volume["files"]
            is_wall_path = Path(files["interior_is_wall"]["path"])
            coords_path = Path(files["interior_coords"]["path"])
            if sha256_file(is_wall_path) != files["interior_is_wall"]["sha256"]:
                raise ValueError(f"interior wall mask hash drift: {is_wall_path}")
            if sha256_file(coords_path) != files["interior_coords"]["sha256"]:
                raise ValueError(f"interior coordinate hash drift: {coords_path}")
            is_wall = np.load(is_wall_path, mmap_mode="r")
            coords = np.load(coords_path, mmap_mode="r")
            rng = np.random.default_rng(_case_seed(row["canonical_id"], seed))
            indices = _take_strict_volume(rng, is_wall, int(query_points))
            query_coords = np.asarray(coords[indices], dtype=np.float32)
            case_dir = OUTPUT_ROOT / "validation_queries" / row["canonical_id"]
            index_path = case_dir / "indices.npy"
            query_coords_path = case_dir / "coords.npy"
            _atomic_save(index_path, indices.astype(np.int64))
            _atomic_save(query_coords_path, query_coords)
            query_contract = {
                "case_id": row["canonical_id"],
                "role": "val",
                "points": int(len(indices)),
                "seed": int(seed),
                "sampling": "uniform strict-volume without replacement",
                "indices": {
                    "path": str(index_path.resolve()),
                    "sha256": sha256_file(index_path),
                },
                "prediction_only_coords": {
                    "path": str(query_coords_path.resolve()),
                    "sha256": sha256_file(query_coords_path),
                },
            }
            query_contract["query_sha256"] = sha256_json(query_contract)
            output_row["fixed_validation_query"] = query_contract
            query_rows.append(query_contract)
        rows.append(output_row)

    counts = {
        "train": sum(row["role"] == "train" for row in rows),
        "val": sum(row["role"] == "val" for row in rows),
        "total": len(rows),
    }
    if counts != {"train": 123, "val": 15, "total": 138}:
        raise ValueError(f"v4 role counts drift: {counts}")
    if source_roles != {"train": 123, "val": 15, "test": 35}:
        raise ValueError(f"source role counts drift: {source_roles}")

    query_manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "route": ROUTE,
        "purpose": "fixed val15 query coordinates shared by every Stage-1 arm",
        "truth_access": "none; coordinate files contain no u/v/w/p targets",
        "split_sha256": SPLIT_SHA256,
        "seed": int(seed),
        "query_points_per_case": int(query_points),
        "cases": query_rows,
    }
    atomic_write_json(QUERY_MANIFEST, query_manifest)

    manifest = {
        "schema_version": 4,
        "created_at": utc_now(),
        "status": "completed",
        "route": ROUTE,
        "scope": "train123/val15 only; test35 rows intentionally absent",
        "counts": counts,
        "split": {
            "path": source["split"]["path"],
            "sha256": SPLIT_SHA256,
            "allowed_roles": ["train", "val"],
            "test35_guard": "reject",
        },
        "field_stats": {
            "path": str(SOURCE_STATS.resolve()),
            "sha256": sha256_file(SOURCE_STATS),
            "source_route": "volume_uvwp_peak_qs_smooth_v3",
            "scope": "train123 strict-volume only",
        },
        "validation_queries": {
            "path": str(QUERY_MANIFEST.resolve()),
            "sha256": sha256_file(QUERY_MANIFEST),
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST.resolve()),
            "sha256": sha256_file(SOURCE_MANIFEST),
        },
        "cases": rows,
    }
    atomic_write_json(OUTPUT_MANIFEST, manifest)

    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "gate_result": "pass",
        "gate_reasons": [],
        "route": ROUTE,
        "counts": counts,
        "source_counts": source_roles,
        "test35_rows_read_for_targets": 0,
        "test35_rows_in_v4_manifest": 0,
        "manifest": {"path": str(OUTPUT_MANIFEST), "sha256": sha256_file(OUTPUT_MANIFEST)},
        "validation_queries": {"path": str(QUERY_MANIFEST), "sha256": sha256_file(QUERY_MANIFEST)},
        "field_stats": {"path": str(SOURCE_STATS), "sha256": sha256_file(SOURCE_STATS)},
        "split_sha256": SPLIT_SHA256,
        "query_hashes": {
            row["case_id"]: row["query_sha256"] for row in query_rows
        },
        "git": git_state(),
    }
    atomic_write_json(AUDIT_OUTPUT, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--query-points", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = build(seed=args.seed, query_points=args.query_points, force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
