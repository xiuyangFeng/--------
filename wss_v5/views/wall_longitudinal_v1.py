"""Export uniform optional value/mask sidecars from a geometry candidate.

All candidate cases and all selected training-view wall rows are preserved.
Only node IDs and original coordinates are read from the training bundle;
no WSS, pressure, CFD boundary conditions or anatomical label overrides enter
the exported feature values. Reference and point-cloud fields never fallback
to each other. This command neither modifies the candidate nor starts training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from training_wss_min.longitudinal_geometry import (
    SCHEMA, FIELD_SPECS, FEATURE_KEYS, checked_pair, sha256_file,
)


def align_rows(candidate_ids, candidate_xyz, view_ids, view_xyz) -> np.ndarray:
    """Exact node correspondence and raw float32 coordinate identity; no NN."""
    source_ids, target_ids = np.asarray(candidate_ids), np.asarray(view_ids)
    if source_ids.ndim != 1 or target_ids.ndim != 1:
        raise ValueError("wall node IDs must be one-dimensional")
    if not np.issubdtype(source_ids.dtype, np.integer) or not np.issubdtype(target_ids.dtype, np.integer):
        raise ValueError("wall node IDs must be integers")
    if len(set(source_ids.tolist())) != len(source_ids) or len(set(target_ids.tolist())) != len(target_ids):
        raise ValueError("duplicate wall node IDs cannot be aligned")
    source_xyz, target_xyz = np.asarray(candidate_xyz), np.asarray(view_xyz)
    if source_xyz.shape != (len(source_ids), 3) or target_xyz.shape != (len(target_ids), 3):
        raise ValueError("original wall coordinates must be N x 3")
    if not np.isfinite(source_xyz).all() or not np.isfinite(target_xyz).all():
        raise ValueError("nonfinite original wall coordinates")
    order = np.argsort(source_ids)
    where = np.searchsorted(source_ids[order], target_ids)
    if np.any(where >= len(order)):
        raise ValueError("training-view nodes absent from geometry candidate")
    rows = order[where]
    if not np.array_equal(source_ids[rows], target_ids):
        raise ValueError("training-view nodes absent from geometry candidate")
    # Both production formats store raw coordinates as float32. Compare that
    # recorded precision exactly, never accept a nearest geometric neighbour.
    if not np.array_equal(source_xyz[rows].astype(np.float32), target_xyz.astype(np.float32)):
        raise ValueError("candidate and training view have different original wall coordinates")
    return rows


def export_case(canonical_id: str, candidate: Path, view_root: Path, out_root: Path) -> dict:
    source_dir = candidate / "cases" / canonical_id.replace("/", "__")
    geometry_path, report_path = source_dir / "geometry.npz", source_dir / "report.json"
    bundle_path = view_root / canonical_id / "bundle.npz"
    report = json.loads(report_path.read_text())
    if report["canonical_id"] != canonical_id:
        raise ValueError(f"candidate report identity mismatch: {canonical_id}")
    with np.load(geometry_path, allow_pickle=False) as geometry, np.load(bundle_path, allow_pickle=False) as bundle:
        if str(bundle["canonical_id"].item()) != canonical_id:
            raise ValueError(f"training bundle identity mismatch: {canonical_id}")
        rows = align_rows(geometry["wall_node_id"], geometry["wall_xyz_mm"],
                          bundle["wall_node_id_cas"], bundle["wall_coords_raw"])
        n = len(rows)
        payload = {
            "longitudinal_schema": np.asarray(SCHEMA), "canonical_id": np.asarray(canonical_id),
            "wall_node_id_cas": np.asarray(bundle["wall_node_id_cas"], dtype=np.int64),
            "wall_coords_raw": np.asarray(bundle["wall_coords_raw"]).copy(),
            "candidate_wall_rows": rows.astype(np.int64),
        }
        counts = {}
        for source, prefix in (("ref", "wall_map_"), ("pc", "wall_map_pc_")):
            for name, (value_key, mask_key, _) in FIELD_SPECS.items():
                feature = f"geom_{source}_{name}"
                values, valid = checked_pair(geometry[prefix + value_key][rows],
                                             geometry[prefix + mask_key][rows], feature, n)
                if name == "log_area":
                    if np.any(values[valid] <= 0):
                        raise ValueError(f"{canonical_id}: valid area must be positive")
                    values[valid] = np.log(values[valid])
                values[~valid] = np.nan
                encoded = values.astype(np.float32)
                checked_pair(encoded, valid, feature, n)
                payload[f"wall_{feature}"] = encoded
                payload[f"wall_{feature}_valid"] = valid
                counts[feature] = int(valid.sum())
        source_n = len(geometry["wall_node_id"])
    provenance = {
        "schema": SCHEMA, "canonical_id": canonical_id, "geometry_only": True,
        "candidate": str(candidate.resolve()), "source_geometry": str(geometry_path.resolve()),
        "source_geometry_sha256": sha256_file(geometry_path), "source_report_sha256": sha256_file(report_path),
        "training_view_bundle": str(bundle_path.resolve()), "training_view_bundle_sha256": sha256_file(bundle_path),
        "export_code_sha256": sha256_file(Path(__file__)),
        "schema_code_sha256": sha256_file(Path(__file__).resolve().parents[2] / "training_wss_min/longitudinal_geometry.py"),
        "view_identity_fields_read": ["canonical_id", "wall_node_id_cas", "wall_coords_raw"],
        "candidate_fields_used": ["wall_node_id", "wall_xyz_mm"] + [
            prefix + key for prefix in ("wall_map_", "wall_map_pc_")
            for value, mask, _ in FIELD_SPECS.values() for key in (value, mask)],
        "node_alignment": "exact_unique_ID_mapping_and_exact_raw_float32_coordinates",
        "source_wall_points": source_n, "view_wall_points": n, "exported_wall_points": n,
        "view_points_dropped": 0, "valid_count_by_feature": counts,
        "feature_names": list(FEATURE_KEYS),
        "units_by_feature": {f"geom_{src}_{name}": spec[2] for src in ("ref", "pc") for name, spec in FIELD_SPECS.items()},
        "mask_semantics": "per-field source validity; invalid raw values are NaN; no source substitution",
        "network_semantics": "train-valid-only z-score; invalid standardised value=0 with mandatory binary mask",
        "supervision_filter": "none; geometry validity never removes WSS supervision rows",
        "anatomy_labels_or_overrides_consumed": False, "cfd_solution_values_consumed": False,
        "surface_guided_candidate": True, "bare_pointcloud_portability_proven": False,
        "training_started": False, "training_approval_changed": False,
    }
    payload["provenance_json"] = np.asarray(json.dumps(provenance, ensure_ascii=False, sort_keys=True))
    dest = out_root / canonical_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "features.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing sidecar: {path}")
    temporary = dest / f"features.{os.getpid()}.tmp.npz"
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    provenance["features_sha256"] = sha256_file(path)
    (dest / "report.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--expected-cases", type=int, default=170)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.candidate, args.view_root, args.out_root = args.candidate.resolve(), args.view_root.resolve(), args.out_root.resolve()
    if args.out_root == args.candidate or args.out_root == args.view_root:
        parser.error("sidecar output must be separate from the candidate and training view")
    if args.out_root.exists() and any(args.out_root.iterdir()):
        parser.error("--out-root must be new or empty; existing exports are immutable")
    manifest_path = args.candidate / "coverage_manifest.json"
    if not manifest_path.exists():
        manifest_path = args.candidate / "refinement_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases = [row["canonical_id"] for row in manifest["cases"]]
    split = json.loads(args.split.read_text())
    selected = [cid for part in ("train_cases", "val_cases", "test_cases") for cid in split.get(part, [])]
    if (len(cases) != args.expected_cases or len(cases) != len(set(cases))
            or len(selected) != len(set(selected)) or set(cases) != set(selected)):
        parser.error("candidate identities must exactly equal all split partitions and --expected-cases")
    for cid in cases:
        if not (args.view_root / cid / "bundle.npz").is_file():
            parser.error(f"training view missing a required case: {cid}")
    if args.workers == 1:
        rows = [export_case(cid, args.candidate, args.view_root, args.out_root) for cid in sorted(cases)]
    else:
        from itertools import repeat
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(export_case, sorted(cases), repeat(args.candidate), repeat(args.view_root), repeat(args.out_root)))
    summary = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": str(args.candidate), "candidate_manifest_sha256": sha256_file(manifest_path),
        "view_root": str(args.view_root), "split": str(args.split.resolve()), "split_sha256": sha256_file(args.split),
        "case_count": len(rows), "feature_names": list(FEATURE_KEYS), "all_cases_same_fields": True,
        "all_view_wall_points_preserved": True, "no_case_coverage_hold": True,
        "training_started": False, "training_approval_changed": False, "cases": rows,
    }
    (args.out_root / "longitudinal_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.out_root), "cases": len(rows), "fields": len(FEATURE_KEYS),
                      "all_view_wall_points_preserved": True, "training_started": False}))


if __name__ == "__main__":
    main()
