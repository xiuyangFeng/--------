#!/usr/bin/env python3
"""Independently verify the VELWSS2 derived-WSS postview packs; no inference."""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
ROOT = next(p for p in OUT.parents if (p / "training_wss_min").is_dir())
sys.path.insert(0, str(ROOT))
from verify_selected_postview import Audit, derived_selfmax, load_npz, read_vtp  # noqa: E402

DATA = ROOT / "data_wss_v5/views/wss_min_view_v1"
EXP = ROOT / "training_wss_min/experiments/vf6_velocity_to_wss_20260916"
MODEL = "VF6_wss"
ROLES = ("best", "median", "worst")
SELECT_PATH = OUT.parent / "启发式v2_R2分布_20260914/B_VF6_wss_selection.json"


def verify_case(audit: Audit, role: str, selection: dict) -> None:
    d, label = OUT / MODEL / role, f"{MODEL}/{role}"
    m = json.loads((d / "manifest.json").read_text())
    start = len(audit.checks)
    audit.check(label + "/schema", m.get("schema_version") == 3)
    uid = m["case_id"]
    audit.check(label + "/selected_id", uid == selection[role]["case_id"])
    pred = load_npz(EXP / "predictions/legacy" / (uid.replace("/", "__") + ".npz"))
    b = load_npz(DATA / uid / "bundle.npz")
    official = json.loads((EXP / "metrics/legacy.json").read_text())["s1234_calibrated"]["per_case"][uid]["overall"]
    wall = b["wall_coords_aligned_mm"].astype(float)
    audit.check(label + "/frame", m["peak_step"] == int(b["peak_step"]) == 1162
                and m["peak_index"] == int(np.where(b["steps"] == 1162)[0][0]) == 21
                and m["frame"] == str(b["transform_frame_version"]) == "v5_atlas_frame_v1"
                and m["coordinate_unit"] == "mm")
    audit.check(label + "/seed", m["seed"] == 1234 and m["split"] == "test")
    audit.equal(label + "/selection_r2", m["selection"]["r2_case_mean"], selection[role]["r2"])
    audit.check(label + "/selection_separate", "three-seed" in m["selection"]["basis"]
                and "not ensemble" in m["selection"]["basis"]
                and m["visualization_metrics"]["seed"] == 1234)
    for name, path in m["files"].items():
        if path is not None:
            audit.check(label + "/relative_pointer/" + name, not Path(path).is_absolute() and (d / path).is_file())
    point = read_vtp(d / m["files"]["pointcloud"])
    audit.check(label + "/pointcloud_vertices", len(point["triangles"]) == 0 and point["vertices"] == len(point["xyz"]))
    true, pred_pa = pred["truth_wss_pa"].astype(float), pred["s1234_calibrated_wss_pa"].astype(float)
    audit.equal(label + "/truth_mother", true, b["wall_wss"][21])
    audit.equal(label + "/raw_wall_mm", pred["wall_mm"], b["wall_coords_raw"])
    audit.check(label + "/valid_all", bool(np.asarray(pred["valid_mask"]).all()))
    expected = {
        "wss_cfd_pa": true, "wss_pred_pa": pred_pa,
        "wss_error_pred_minus_cfd_pa": pred_pa - true, "wss_abs_error_pa": abs(pred_pa - true),
        "row_index": np.arange(len(wall), dtype=np.int64),
    }
    audit.equal(label + "/coordinates", point["xyz"], wall)
    cmax, pmax = float(true.max()), float(pred_pa.max())
    contract = m["display_normalization"]
    audit.check(label + "/positive_finite_denominators", np.isfinite([cmax, pmax]).all() and cmax > 0 and pmax > 0)
    audit.close(label + "/cfd_max_domain", contract["cfd_max"], cmax, 0)
    audit.close(label + "/pred_max_domain", contract["pred_max"], pmax, 0)
    audit.check(label + "/denominator_scope", contract["denominator_scope"] == "wall"
                and contract["source_point_count"] == len(true) and contract["prefix"] == "wss"
                and contract["source_unit"] == "Pa" and contract["unit"] == "dimensionless"
                and contract["amplitude_difference_removed"] is True)
    expected.update(derived_selfmax(true, pred_pa, "wss", cmax, pmax))
    audit.compare_arrays(label + "/point_array", point["arrays"], expected)
    audit.normalization(label + "/point_fielddata", point, contract, cmax, pmax)
    audit.csv(label + "/point_csv", d / m["files"]["original_points_csv"], wall, expected)
    r2 = 1 - ((pred_pa - true) ** 2).sum() / ((true - true.mean()) ** 2).sum()
    mae = abs(pred_pa - true).mean()
    audit.close(label + "/r2_official", r2, official["r2"], 1e-12)
    audit.close(label + "/mae_official", mae, official["mae"], 1e-12)
    audit.equal(label + "/metric_metadata_r2", m["visualization_metrics"]["r2"], official["r2"])
    audit.equal(label + "/metric_metadata_n", m["visualization_metrics"]["n"], len(true))
    native = read_vtp(d / m["files"]["native_cfd_wall"])
    surface = read_vtp(d / m["files"]["gaussian_surface"])
    aligned = read_vtp(d / m["files"]["aligned_geometry"])
    report = json.loads((d / m["files"]["mapping_report"]).read_text())
    native_expected = dict(expected)
    audit.equal(label + "/native_coordinates", native["xyz"], wall)
    audit.compare_arrays(label + "/native_array", native["arrays"], native_expected)
    audit.normalization(label + "/native_fielddata", native, contract, cmax, pmax)
    geometry = m["geometry"]
    with h5py.File(geometry["snapshot_source"]) as h:
        valid = h["wall_static/valid"][:].astype(bool)
        audit.equal(label + "/native_node_ids", b["wall_node_id_cas"], h["wall_static/node_id_cas"][:][valid])
        remap = np.full(len(valid), -1, int)
        remap[valid] = np.arange(valid.sum())
        tri = remap[h["topology/wall_triangles"][:]]
        tri = tri[np.all(tri >= 0, axis=1)]
        audit.equal(label + "/native_triangles", native["triangles"], tri)
        raw = h["wall_static/xyz_mm"][:][valid]
        reconstructed = (raw - b["transform_centroid"]) @ b["transform_rotation"].T
        audit.check(label + "/atlas_registration", np.max(abs(reconstructed - wall)) < 1e-3)
    sx, sa, st = surface["xyz"], surface["arrays"], surface["triangles"]
    audit.equal(label + "/surface_coordinates", sx, aligned["xyz"])
    audit.equal(label + "/surface_topology", st, aligned["triangles"])
    audit.check(label + "/real_surface_triangles", len(st) > 0 and st.min() >= 0 and st.max() < len(sx)
                and len(st) == len(aligned["triangles"]))
    audit.check(label + "/native_cfd_triangles", len(native["triangles"]) > 0 and native["triangles"].min() >= 0
                and native["triangles"].max() < len(wall))
    audit.equal(label + "/surface_signed_error", sa["wss_error_pred_minus_cfd_pa"], sa["wss_pred_pa"] - sa["wss_cfd_pa"])
    audit.equal(label + "/surface_absolute_error", sa["wss_abs_error_pred_minus_cfd_pa"], abs(sa["wss_error_pred_minus_cfd_pa"]))
    audit.check(label + "/coverage", np.all(sa["map_valid"] == 1) and report["coverage"]["valid_ratio"] == 1
                and report["coverage"]["target_points"] == len(sx) and report["source_points"] == len(wall))
    for field, values in derived_selfmax(sa["wss_cfd_pa"], sa["wss_pred_pa"], "wss", cmax, pmax).items():
        audit.equal(label + "/surface_selfmax/" + field, sa[field], values)
    audit.normalization(label + "/surface_fielddata", surface, contract, cmax, pmax)
    audit.check(label + "/mapping_normalization", report["display_normalization"] == contract)
    tree = cKDTree(wall)
    distances, _ = tree.query(sx, k=1)
    audit.check(label + "/mapping_distances", np.max(abs(distances - sa["map_dist_mm"])) < 1e-10)
    audit.close(label + "/mapping_max_distance", report["distance_mm"]["max"], distances.max(), 1e-10)
    audit.check(label + "/gaussian_parameters", report["params"]["radius_mm"] == 3
                and report["params"]["sharpness"] == 2 and report["params"]["max_dist_mm"] == 3)
    samples = np.unique(np.linspace(0, len(sx) - 1, 64, dtype=int))
    maxerror = 0.0
    for index in samples:
        ids = tree.query_ball_point(sx[index], 3.0)
        distance = np.linalg.norm(wall[ids] - sx[index], axis=1)
        weights = np.exp(-2 * (distance / 3) ** 2)
        weights /= weights.sum()
        for field, values in [("wss_cfd_pa", true), ("wss_pred_pa", pred_pa)]:
            maxerror = max(maxerror, abs(float(weights @ values[ids]) - float(sa[field][index])))
    audit.check(label + "/independent_gaussian_64", maxerror < 1e-9,
                {"samples": len(samples), "max_absolute_difference": maxerror})
    qc = report["cross_wall_checks"]
    disconnected, opposite = sa["disconnected_local_patch_weight_fraction"], sa["opposite_normal_weight_fraction"]
    dc, oc = int((disconnected > 1e-12).sum()), int((opposite > 1e-12).sum())
    audit.check(label + "/cross_wall_qc", dc == qc["disconnected_stencil_count"] and oc == qc["opposite_normal_stencil_count"]
                and abs(disconnected.max() - qc["max_disconnected_weight"]) < 1e-12
                and abs(opposite.max() - qc["max_opposite_normal_weight"]) < 1e-12)
    audit.check(label + "/geometry_source_sha", audit.sha(geometry["stl_source"]) == geometry["stl_sha256"])
    audit.csv(label + "/mapped_csv", d / "surface_gaussian/mapped_vertices.csv", sx, sa)
    source_arrays = {
        "wss_cfd_pa": true, "wss_pred_pa": pred_pa,
        "wss_error_pred_minus_cfd_pa": pred_pa - true, "wss_abs_error_pa": abs(pred_pa - true),
        **derived_selfmax(true, pred_pa, "wss", cmax, pmax),
    }
    audit.csv(label + "/gaussian_source_csv", d / "_export/gaussian_source.csv.gz", wall, source_arrays)
    for path, sha in m["source_sha256"].items():
        audit.check(label + "/source_sha/" + Path(path).name, audit.sha(path) == sha)
    for path, sha in m["output_sha256"].items():
        audit.check(label + "/output_sha/" + path, audit.sha(d / path) == sha)
    audit.cases.append({
        "model": MODEL, "role": role, "case_id": uid, "point_count": len(wall),
        "s1234_r2": official["r2"], "three_seed_selection_r2": m["selection"]["r2_case_mean"],
        "surface": {
            "points": len(sx), "triangles": len(st), "coverage": 1.0,
            "nearest_max_mm": float(distances.max()), "independent_kernel_max_error": maxerror,
            "disconnected_stencils": dc, "max_disconnected_weight": float(disconnected.max()),
            "opposite_normal_stencils": oc, "max_opposite_normal_weight": float(opposite.max()),
        },
        "checks": len(audit.checks) - start,
        "passed": all(c["passed"] for c in audit.checks[start:]),
    })
    print(MODEL, role, "passed", audit.cases[-1]["passed"], "checks", audit.cases[-1]["checks"], flush=True)


def main() -> int:
    audit = Audit()
    audit.helper_branches()
    selection = json.loads(SELECT_PATH.read_text())
    try:
        for role in ROLES:
            verify_case(audit, role, selection)
    except Exception as error:
        audit.check("unhandled_verification_exception", False, repr(error))
        _write(audit)
        raise
    return 0 if _write(audit) else 1


def _write(audit: Audit) -> bool:
    errors = [c for c in audit.checks if not c["passed"]]
    result = {
        "status": "passed" if not errors else "failed",
        "schema_version": 3,
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "VELWSS2 three-case derived-WSS postview audit from existing caches; no inference",
        "verifier": Path(__file__).name,
        "verifier_sha256": audit.sha(__file__),
        "case_count": len(audit.cases),
        "check_count": len(audit.checks),
        "failed_check_count": len(errors),
        "cases": audit.cases,
        "errors": errors,
    }
    (OUT / MODEL / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["status", "case_count", "check_count", "failed_check_count"]}), flush=True)
    return not errors


if __name__ == "__main__":
    raise SystemExit(main())
