#!/usr/bin/env python3
"""Independently verify X5D_v51 best/median/worst postview packs; no inference."""
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
sys.path.insert(0, str(OUT))
from verify_selected_postview import Audit, derived_selfmax, load_npz, read_vtp  # noqa: E402

DATA = ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1"
RUN = ROOT / "training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s1234"
MODEL = "X5D_v51"
ROLES = ("best", "median", "worst")


def verify_case(audit: Audit, role: str, selection: dict) -> None:
    dest, label = OUT / MODEL / role, f"{MODEL}/{role}"
    manifest = json.loads((dest / "manifest.json").read_text())
    start = len(audit.checks)
    audit.check(label + "/schema", manifest.get("schema_version") == 3)
    cid = manifest["case_id"]
    audit.check(label + "/selected_id", cid == selection[role]["case_id"])
    pred = load_npz(RUN / "eval/ckpt_best/predictions/test" / cid / "predictions.npz")
    bundle = load_npz(DATA / cid / "bundle.npz")
    official = json.loads((RUN / "eval/ckpt_best/metrics.json").read_text())["test"]["per_case"][cid]["overall"]
    wall = bundle["wall_coords_aligned_mm"].astype(float)
    peak_index = int(np.where(bundle["steps"] == 1162)[0][0])
    audit.check(label + "/frame", manifest["peak_step"] == int(bundle["peak_step"]) == 1162
                and manifest["peak_index"] == peak_index == 21
                and manifest["frame"] == str(bundle["transform_frame_version"]) == "v5_atlas_frame_v1"
                and manifest["coordinate_unit"] == "mm")
    audit.check(label + "/seed", manifest["seed"] == 1234 and manifest["split"] == "test")
    audit.check(label + "/not_longitudinal", "not X5D_long" in manifest["selection"]["basis"])
    audit.equal(label + "/selection_r2", manifest["selection"]["r2_case_mean"], selection[role]["r2"])
    audit.check(label + "/selection_separate", "five-seed" in manifest["selection"]["basis"]
                and "not ensemble" in manifest["selection"]["basis"]
                and manifest["visualization_metrics"]["seed"] == 1234)
    for name, path in manifest["files"].items():
        if path is not None:
            audit.check(label + "/relative_pointer/" + name, not Path(path).is_absolute() and (dest / path).is_file())
    point = read_vtp(dest / manifest["files"]["pointcloud"])
    audit.check(label + "/pointcloud_vertices", len(point["triangles"]) == 0 and point["vertices"] == len(point["xyz"]))
    index = pred["row_index"].astype(np.int64)
    true, pred_pa = pred["true_pa"].astype(float), pred["pred_pa"].astype(float)
    audit.equal(label + "/truth_mother", true, bundle["wall_wss"][21, index])
    audit.equal(label + "/coordinates", point["xyz"], wall[index])
    expected = {
        "wss_cfd_pa": true, "wss_pred_pa": pred_pa,
        "wss_error_pred_minus_cfd_pa": pred_pa - true, "wss_abs_error_pa": abs(pred_pa - true),
        "wss_cfd_norm": pred["true_norm"].astype(float), "wss_pred_norm": pred["pred_norm"].astype(float),
        "row_index": index,
    }
    cmax, pmax = float(true.max()), float(pred_pa.max())
    contract = manifest["display_normalization"]
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
    audit.csv(label + "/point_csv", dest / manifest["files"]["original_points_csv"], wall[index], expected)
    r2 = 1 - ((pred_pa - true) ** 2).sum() / ((true - true.mean()) ** 2).sum()
    mae = abs(pred_pa - true).mean()
    audit.close(label + "/r2_official", r2, official["r2"], 1e-8)
    audit.close(label + "/mae_official", mae, official["mae"], 1e-6)
    audit.equal(label + "/metric_metadata_r2", manifest["visualization_metrics"]["r2"], official["r2"])
    audit.equal(label + "/metric_metadata_n", manifest["visualization_metrics"]["n"], len(true))
    native = read_vtp(dest / manifest["files"]["native_cfd_wall"])
    surface = read_vtp(dest / manifest["files"]["gaussian_surface"])
    aligned = read_vtp(dest / manifest["files"]["aligned_geometry"])
    report = json.loads((dest / manifest["files"]["mapping_report"]).read_text())
    order = np.argsort(index)
    native_expected = {key: value[order] if np.ndim(value) else value for key, value in expected.items()}
    audit.equal(label + "/native_coordinates", native["xyz"], wall)
    audit.compare_arrays(label + "/native_array", native["arrays"], native_expected)
    audit.normalization(label + "/native_fielddata", native, contract, cmax, pmax)
    geometry = manifest["geometry"]
    with h5py.File(geometry["snapshot_source"]) as handle:
        valid = handle["wall_static/valid"][:].astype(bool)
        audit.equal(label + "/native_node_ids", bundle["wall_node_id_cas"], handle["wall_static/node_id_cas"][:][valid])
        remap = np.full(len(valid), -1, int)
        remap[valid] = np.arange(valid.sum())
        tri = remap[handle["topology/wall_triangles"][:]]
        tri = tri[np.all(tri >= 0, axis=1)]
        audit.equal(label + "/native_triangles", native["triangles"], tri)
        raw = handle["wall_static/xyz_mm"][:][valid]
        reconstructed = (raw - bundle["transform_centroid"]) @ bundle["transform_rotation"].T
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
    ordered_true, ordered_pred = true[order], pred_pa[order]
    for sample in samples:
        ids = tree.query_ball_point(sx[sample], 3.0)
        distance = np.linalg.norm(wall[ids] - sx[sample], axis=1)
        weights = np.exp(-2 * (distance / 3) ** 2)
        weights /= weights.sum()
        for field, values in [("wss_cfd_pa", ordered_true), ("wss_pred_pa", ordered_pred)]:
            maxerror = max(maxerror, abs(float(weights @ values[ids]) - float(sa[field][sample])))
    audit.check(label + "/independent_gaussian_64", maxerror < 1e-9,
                {"samples": len(samples), "max_absolute_difference": maxerror})
    qc = report["cross_wall_checks"]
    disconnected, opposite = sa["disconnected_local_patch_weight_fraction"], sa["opposite_normal_weight_fraction"]
    dc, oc = int((disconnected > 1e-12).sum()), int((opposite > 1e-12).sum())
    audit.check(label + "/cross_wall_qc", dc == qc["disconnected_stencil_count"] and oc == qc["opposite_normal_stencil_count"]
                and abs(disconnected.max() - qc["max_disconnected_weight"]) < 1e-12
                and abs(opposite.max() - qc["max_opposite_normal_weight"]) < 1e-12)
    audit.check(label + "/geometry_source_sha", audit.sha(geometry["stl_source"]) == geometry["stl_sha256"])
    audit.csv(label + "/mapped_csv", dest / "surface_gaussian/mapped_vertices.csv", sx, sa)
    source_arrays = {
        "wss_cfd_pa": ordered_true, "wss_pred_pa": ordered_pred,
        "wss_error_pred_minus_cfd_pa": ordered_pred - ordered_true,
        "wss_abs_error_pa": abs(ordered_pred - ordered_true),
        **derived_selfmax(ordered_true, ordered_pred, "wss", cmax, pmax),
    }
    audit.csv(label + "/gaussian_source_csv", dest / "_export/gaussian_source.csv.gz", wall, source_arrays)
    for path, digest in manifest["source_sha256"].items():
        audit.check(label + "/source_sha/" + Path(path).name, audit.sha(path) == digest)
    for path, digest in manifest["output_sha256"].items():
        audit.check(label + "/output_sha/" + path, audit.sha(dest / path) == digest)
    audit.cases.append({
        "model": MODEL, "role": role, "case_id": cid, "point_count": len(true),
        "s1234_r2": official["r2"], "five_seed_selection_r2": manifest["selection"]["r2_case_mean"],
        "surface": {
            "points": len(sx), "triangles": len(st), "coverage": 1.0,
            "nearest_max_mm": float(distances.max()), "independent_kernel_max_error": maxerror,
            "disconnected_stencils": dc, "max_disconnected_weight": float(disconnected.max()),
            "opposite_normal_stencils": oc, "max_opposite_normal_weight": float(opposite.max()),
        },
        "checks": len(audit.checks) - start,
        "passed": all(item["passed"] for item in audit.checks[start:]),
    })
    print(MODEL, role, "passed", audit.cases[-1]["passed"], "checks", audit.cases[-1]["checks"], flush=True)


def main() -> int:
    audit = Audit()
    audit.helper_branches()
    selection = json.loads((OUT / MODEL / "selection.json").read_text())
    try:
        for role in ROLES:
            verify_case(audit, role, selection)
    except Exception as error:
        audit.check("unhandled_verification_exception", False, repr(error))
        _write(audit)
        raise
    return 0 if _write(audit) else 1


def _write(audit: Audit) -> bool:
    errors = [item for item in audit.checks if not item["passed"]]
    result = {
        "status": "passed" if not errors else "failed",
        "schema_version": 3,
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Finished v5.1 X5D_v51 three-case wall postview from caches; no inference",
        "verifier": Path(__file__).name,
        "verifier_sha256": audit.sha(__file__),
        "case_count": len(audit.cases),
        "check_count": len(audit.checks),
        "failed_check_count": len(errors),
        "cases": audit.cases,
        "errors": errors,
    }
    (OUT / MODEL / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["status", "case_count", "check_count", "failed_check_count"]}), flush=True)
    return not errors


if __name__ == "__main__":
    raise SystemExit(main())
