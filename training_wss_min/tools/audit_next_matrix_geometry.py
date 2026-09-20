#!/usr/bin/env python3
"""Audit frozen wall geometry and build the E9 public-Q / true-inlet-area input.

Reads an explicit allowlist of geometry and public nominal waveform datasets.
No CFD solution or case-specific boundary impedance enters the feature table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from training_wss_min.next_geometry import case_geometry, source_h5_for_bundle


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def inlet_feature_from_arrays(vectors, q_peak, declared_area):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) == 0 or not np.isfinite(vectors).all():
        raise ValueError("invalid inlet face area vectors")
    norms = np.linalg.norm(vectors, axis=1)
    area = float(norms.sum())
    planarity = float(np.linalg.norm(vectors.sum(axis=0)) / max(area, 1e-30))
    if np.any(norms <= 0) or not np.isfinite(q_peak) or q_peak <= 0 or area <= 0:
        raise ValueError("nonpositive inlet geometry or public nominal flow")
    if not np.isclose(area, declared_area, rtol=1e-7, atol=1e-14):
        raise ValueError("inlet area disagrees with frozen interface metadata")
    if planarity < 0.99:
        raise ValueError("inlet face orientation/planarity gate failed")
    return {"inlet_area_m2": area, "inlet_planarity": planarity,
            "inlet_velocity_nominal_m_s": float(q_peak / area)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--view-root", type=Path, default=Path("data_wss_v5/views/wss_min_view_v1"))
    p.add_argument("--output-dir", type=Path, default=Path("training_wss_min/experiments/wss_direct_recovery_20260912/inlet_audit"))
    p.add_argument("--baseline-feature-stats", type=Path,
                   default=Path("training_wss_min/runs/v6_followup_20260909/M2_a5_independent_k3_s1234/feature_stats.json"))
    args = p.parse_args()
    root = args.view_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    waveform_path = root / "protocol_inlet_waveform.json"
    wf = json.loads(waveform_path.read_text())
    nominal = np.asarray(wf["q_nom_m3s"], dtype=np.float64)
    q_peak = float(wf["q_peak_m3s"])
    if not np.isclose(q_peak, nominal.max(), rtol=0, atol=1e-16) or int(wf["peak_step"]) != 1162:
        raise ValueError("public waveform does not match the requested peak-step protocol")
    split = json.loads((root / "split_V5_train138_test34.json").read_text())
    names = split["train_cases"] + split.get("val_cases", []) + split["test_cases"]
    if len(names) != len(set(names)) or len(names) != 172:
        raise ValueError("expected 172 unique frozen cases")
    records, features, errors = [], {}, []
    for unit_id in names:
        try:
            bundle_path = root / unit_id / "bundle.npz"
            source = source_h5_for_bundle(bundle_path)
            with np.load(bundle_path, allow_pickle=False) as z:
                case = {"pos": z["wall_coords_norm"].astype(np.float32), "bundle_path": str(bundle_path)}
                radius = z["wall_local_radius"].copy()
            geom = case_geometry(case)
            with h5py.File(source, "r") as h:
                if not np.array_equal(nominal, h["conditions/q_nom_m3s"][:]):
                    raise ValueError("case nominal waveform is not bit-identical to public protocol")
                vectors = h["interfaces/inlet/face_area_vec_m2"][:]
                result = inlet_feature_from_arrays(vectors, q_peak, float(h["interfaces/inlet"].attrs["area_m2"]))
            if not np.array_equal(radius, geom[:, 6]):
                raise ValueError("local physical radius differs from frozen view")
            records.append({"unit_id": unit_id, "status": "pass", "bundle_path": str(bundle_path),
                            "frozen_source_h5": str(source), "geometry_shape": list(geom.shape),
                            "geometry_sha256": array_hash(geom), "inlet_area_vectors_sha256": array_hash(vectors),
                            "public_q_peak_m3s": q_peak, "nominal_waveform_bit_identical": True,
                            "coord_scale_mm": float(geom[0, 7]), **result})
            features[unit_id] = {"inlet_velocity_nominal_m_s": result["inlet_velocity_nominal_m_s"]}
        except Exception as exc:
            errors.append({"unit_id": unit_id, "error": f"{type(exc).__name__}: {exc}"})
        if (len(records) + len(errors)) % 25 == 0:
            print(f"audited {len(records) + len(errors)}/172, errors={len(errors)}", flush=True)
    audit = {"status": "pass" if not errors else "failed", "cases_expected": 172,
             "cases_passed": len(records), "public_waveform_path": str(waveform_path),
             "public_waveform_sha256": hashlib.sha256(waveform_path.read_bytes()).hexdigest(),
             "feature_definition": "public protocol peak Q [m^3/s] / sum norm(frozen inlet face area vectors) [m^2]",
             "interpretation": "Geometry-dependent reference speed under the common inlet protocol; not patient-specific flow information.",
             "input_provenance": "Frozen inlet boundary mesh geometry; same inlet surface can be triangulated at deployment. No flow solution needed.",
             "read_allowlist": ["wall_static/node_id_cas", "wall_static/xyz_mm", "wall_static/atlas_row", "geometry/atlas_table",
                                "interfaces/inlet/face_area_vec_m2", "conditions/q_nom_m3s (public equality audit only)"],
             "excluded_inputs": ["CFD velocity", "CFD pressure", "WSS labels", "measured inlet flux", "RCR", "unreviewed replacement centerline"],
             "records": records, "errors": errors}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if errors:
        print(json.dumps(errors, indent=2))
        raise SystemExit("geometry audit failed; E9 feature table was not produced")
    table = {"feature_names": ["inlet_velocity_nominal_m_s"], "source_audit": str(output / "audit.json"), "cases": features}
    (output / "case_features.json").write_text(json.dumps(table, indent=2) + "\n")
    # Preserve the entire historical 25D preprocessing contract; fit only the
    # new case-level feature, once per TRAIN case as compute_feature_stats does.
    from training_wss_min.dataset import compute_feature_stats
    stats = json.loads(args.baseline_feature_stats.read_text())
    added = compute_feature_stats([features[name] for name in split["train_cases"]],
                                  ("inlet_velocity_nominal_m_s",))
    stats.update(added)
    (output / "feature_stats_26d.json").write_text(json.dumps(stats, indent=2) + "\n")
    stats_provenance = {"baseline_feature_stats": str(args.baseline_feature_stats.resolve()),
                        "baseline_feature_stats_sha256": hashlib.sha256(args.baseline_feature_stats.read_bytes()).hexdigest(),
                        "new_feature_statistics": added, "fit_cases": split["train_cases"],
                        "weighting": "one observation per training case; existing CASE_FEATURE_KEYS implementation",
                        "test_cases_used_for_statistics": False}
    (output / "feature_stats_provenance.json").write_text(json.dumps(stats_provenance, indent=2) + "\n")
    areas = np.asarray([r["inlet_area_m2"] * 1e6 for r in records])
    speeds = np.asarray([r["inlet_velocity_nominal_m_s"] for r in records])
    text = (f"# Frozen geometry and E9 input audit\n\n172/172 cases passed.\n\n"
            f"- Public peak Q: {q_peak:.12g} m³/s (step 1162; bit-identical for all cases).\n"
            f"- Inlet area range: {areas.min():.3f}–{areas.max():.3f} mm², derived from actual inlet face geometry.\n"
            f"- Q/A range: {speeds.min():.6f}–{speeds.max():.6f} m/s.\n"
            "- Local normal/tangent/radius/scale metadata passed frozen node identity, raw coordinate and finiteness checks.\n\n"
            "This feature describes the geometry-dependent reference speed under a common prescribed waveform. "
            "It adds no patient-specific flow or outlet information. Only the existing frozen geometry and the public nominal waveform are used; "
            "no CFD solution, measured flux, RCR, or replacement geometry is read. "
            "The source inlet boundary must be available at deployment, or its surface must be triangulated to obtain the same area definition.\n")
    (output / "README.md").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
