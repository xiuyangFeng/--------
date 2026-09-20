"""Independent read-only audit of candidate arrays and geometry-source hashes.

Writes only audit.json and audit_summary.md, never edits candidate cases. Run
the full-array/source audit in Slurm; no CFD labels are accessed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .geometry_candidate import (DEFAULT_OUTPUT, SCHEMA, all_cases, code_fingerprints,
                                 digest_arrays, interpolate_valid_field, write_json)
from . import contract as C


def distribution(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(a):
        return {"count": 0}
    return {"count": len(a), "mean": float(a.mean()), "min": float(a.min()),
            "p10": float(np.percentile(a, 10)), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def audit_case(report_path: Path, source_root: Path | None = None):
    r = json.loads(report_path.read_text())
    errors, notes = [], []
    check = lambda condition, message: errors.append(message) if not condition else None
    check(r.get("schema") == SCHEMA, "schema_mismatch")
    check(r.get("algorithm_code") == code_fingerprints(), "current_core_source_hash_mismatch")
    check(hashlib.sha256(json.dumps(r.get("config"), sort_keys=True).encode()).hexdigest() == r.get("config_sha256"), "recorded_config_hash_mismatch")
    check(r.get("training_allowed") is False and r.get("awaiting_user_review") is True and r.get("user_approved") is False,
          "review_training_gate_invalid")
    allowed_fields = {"wall_static/xyz_mm", "wall_static/node_id_cas", "topology/wall_triangles", "geometry/atlas_table", "geometry attributes"}
    check(set(r["source"]["fields_read"]) <= allowed_fields and r["source"]["label_fields_read"] == [], "declared_non_geometry_field_access")
    check(r["source"]["frozen_assets_modified"] is False, "declared_frozen_source_modification")
    # Re-read only the same geometry arrays to verify the frozen source remains
    # byte-identical to the build's recorded input fingerprint.
    source = (C.case_dir(r["canonical_id"], root=source_root) / "case.h5"
              if source_root is not None else Path(r["source"]["case_h5"]))
    with h5py.File(source, "r") as h5:
        wall_source = h5["wall_static/xyz_mm"][()].astype(float)
        tri_source = h5["topology/wall_triangles"][()]
        table_source = h5["geometry/atlas_table"][()]
        ids_source = h5["wall_static/node_id_cas"][()]
    check(digest_arrays({"wall": wall_source, "triangles": tri_source, "atlas": table_source}) == r["source"]["geometry_arrays_sha256"], "frozen_geometry_source_hash_mismatch")
    a = np.load(report_path.parent / "geometry.npz", allow_pickle=False)
    wall = a["wall_xyz_mm"]
    ns, nw = len(a["section_segment_id"]), len(wall)
    check(np.array_equal(wall, wall_source.astype(np.float32)), "wall_points_differ_from_source")
    check(np.array_equal(a["wall_triangles"], tri_source), "wall_triangles_differ_from_source")
    check(np.array_equal(a["wall_node_id"], ids_source), "wall_identity_order_differs_from_source")
    check(np.all((a["wall_triangles"] >= 0) & (a["wall_triangles"] < nw)), "triangle_index_outside_wall")
    check(np.isfinite(wall).all() and np.isfinite(a["centerline_xyz_mm"]).all(), "nonfinite_coordinates")
    expected_segments = {row["segment_id"] for row in r["segments"]}
    check(set(np.unique(a["section_segment_id"])) == expected_segments == set(np.unique(a["centerline_segment_id"])), "segment_identity_mismatch")
    check(len(expected_segments) == 7 and len(r["P3_CIA"]) == 2 and r["P2_topology"]["valid"] and r["P2_topology"]["expected_aortoiliac_pattern_valid"], "unexpected_topology")
    check(r["opening_audit"]["boundary_loops"] == 5 and r["opening_audit"]["valid"], "opening_topology_invalid")
    check(set(np.unique(a["wall_map_segment_id"])) <= expected_segments, "wall_mapping_unknown_segment")
    finite_but_invalid = {}
    for prefix, wall_prefix, wall_valid_key in (("section_", "wall_map_", "wall_map_valid"),
                                                ("pc_section_", "wall_map_pc_", "wall_map_pc_valid")):
        valid = a[prefix + "valid"]
        area = a[prefix + "area_mm2"]
        check(valid.shape == area.shape == (ns,), prefix + "field_shape")
        check(np.all(np.isfinite(area[valid]) & (area[valid] > 0)), prefix + "invalid_positive_area")
        req = a[prefix + "req_mm"]
        check(np.allclose(np.pi * req[valid] ** 2, area[valid], rtol=5e-6), prefix + "equivalent_radius_inconsistent")
        circularity = a[prefix + "roundness"]
        check(np.all(np.isfinite(circularity[valid]) & (circularity[valid] > 0) & (circularity[valid] <= 1 + 1e-5)), prefix + "roundness_outside_range")
        check(np.all(np.isfinite(a[prefix + "eccentricity"][valid]) & (a[prefix + "eccentricity"][valid] >= 0)), prefix + "eccentricity_invalid")
        offsets, polygon = a[prefix + "polygon_offsets"], a[prefix + "polygon_xyz_mm"]
        check(len(offsets) == ns + 1 and offsets[0] == 0 and offsets[-1] == len(polygon) and np.all(np.diff(offsets) >= 0), prefix + "polygon_offsets")
        check(np.all(np.diff(offsets)[valid] >= 3), prefix + "valid_section_has_no_polygon")
        wm = a[wall_prefix + "area_mm2"]
        wv = a[wall_valid_key]
        check(wm.shape == wv.shape == (nw,), wall_prefix + "field_shape")
        check(not np.any(wv & a["wall_map_ambiguous"]), wall_prefix + "ambiguous_marked_valid")
        check(np.all(np.isfinite(wm[wv]) & (wm[wv] > 0)), wall_prefix + "nonfinite_valid_area")
        check(np.isnan(wm[~wv]).all(), wall_prefix + "invalid_area_not_nan")
        raw_wm = a[wall_prefix + "raw_area_mm2"]
        check(np.array_equal(wm[wv], raw_wm[wv]), wall_prefix + "valid_raw_area_mismatch")
        finite_but_invalid[prefix] = int(np.sum(~wv & np.isfinite(wm)))
        check(not np.any(valid & ~a[prefix + "raw_contour_valid"]), prefix + "pipeline_valid_without_raw_contour")
        check(not np.any(valid & (a[prefix + "same_segment_nonlocal_crossing_count"] > 0)), prefix + "nonlocal_same_segment_crossing_marked_valid")
        # Independently replay interpolation invariants, including both masks.
        for seg in expected_segments:
            si = np.flatnonzero(a["section_segment_id"] == seg)
            wi = np.flatnonzero(a["wall_map_segment_id"] == seg)
            ss = a["section_s_local_mm"][si].astype(float)
            check(np.all(np.diff(ss) > 0), prefix + "nonincreasing_section_s")
            v, good = interpolate_valid_field(a["wall_map_s_local_mm"][wi].astype(float), ss, area[si].astype(float), valid[si])
            good &= ~a["wall_map_ambiguous"][wi]
            # Output s is float32 while original interpolation used float64;
            # endpoint comparisons can differ on <1e-5 mm boundaries only.
            disagreements = good != wv[wi]
            if np.any(disagreements):
                q = a["wall_map_s_local_mm"][wi][disagreements]
                near_boundary = np.min(np.abs(q[:, None] - ss), axis=1) < 2e-5
                check(np.all(near_boundary), wall_prefix + "interpolation_mask_mismatch")
            compare = good & wv[wi]
            check(np.allclose(v[compare], wm[wi][compare], rtol=2e-4, atol=1e-3), wall_prefix + "area_interpolation_mismatch")
        for suffix in ("area_slope_per_mm", "roundness", "eccentricity", "upstream_min_area_ratio", "downstream_min_area_ratio", "upstream_min_distance_mm", "downstream_min_distance_mm"):
            fv = a[wall_prefix + suffix + "_valid"]
            x = a[wall_prefix + suffix]
            check(x.shape == fv.shape == (nw,), wall_prefix + suffix + "_shape")
            check(np.isfinite(x[fv]).all() and np.isnan(x[~fv]).all(), wall_prefix + suffix + "_mask_values")
    check(a["section_intersection_offsets"][-1] == len(a["section_intersection_segments_xyz_mm"]), "intersection_offsets_invalid")
    check(a["wall_cia_segment_id"].shape == (nw,), "CIA_wall_shape")
    check(np.all(~a["wall_cia_geometry_valid"] | np.isin(a["wall_cia_segment_id"], [c["segment_id"] for c in r["P3_CIA"]])), "CIA_geometry_mask_invalid")
    for cia in r["P3_CIA"]:
        wi = a["wall_cia_segment_id"] == cia["segment_id"]
        if not cia["semantic_valid"]:
            check(not a["wall_cia_semantic_valid"][wi].any(), "CIA_unknown_semantics_marked_valid")
    paired = a["section_valid"] & a["pc_section_valid"]
    paired_errors = np.abs(a["pc_section_area_mm2"][paired] / a["section_area_mm2"][paired] - 1)
    p1 = r["P1_centerline"]
    check(p1["outside_core_points"] == int(np.sum(a["centerline_audit_core_mask"] & ~a["centerline_inside_reference"])), "inside_report_array_mismatch")
    for item in r["stability"]:
        for variant in item["variants"]:
            check(variant["valid"] == variant["pipeline_valid"], "P6_valid_not_pipeline_valid")
            check(not variant["pipeline_valid"] or variant["raw_contour_valid"], "P6_pipeline_valid_without_raw_contour")
            check(not variant["pipeline_valid"] or variant.get("same_segment_nonlocal_crossing_count", 0) == 0, "P6_same_segment_nonlocal_crossing_marked_valid")
            check(variant["reason"] == variant["pipeline_reason"], "P6_reason_not_pipeline_reason")
            check(not variant["comparison_valid"] or variant["pipeline_valid"], "P6_comparison_without_pipeline_validity")
            check(variant["comparison_valid"] or variant["relative_area_change"] is None, "P6_formal_difference_despite_invalid_comparison")
        variants = {v["name"]: v for v in item["variants"]}
        both = variants["pointcloud_full"]["pipeline_valid"] and variants["pointcloud_half"]["pipeline_valid"]
        check(item["pc_density_comparison_valid"] == both, "P6_density_comparison_mask")
        check(both or item["pc_density_relative_change"] is None, "P6_invalid_density_difference_not_null")
        check(len(item["variants"]) == 10, "P6_missing_shift_tilt_source_variants")
    for v in r["opening_stability"]:
        check(v["valid"] == v["pipeline_valid"], "opening_valid_not_pipeline_valid")
        check(v["comparison_valid"] or v["relative_area_change"] is None, "opening_invalid_comparison_not_null")
    notes.append("common_valid_section_error_excludes_invalid_and_nonstar_sections; coverage_required")
    output = {"canonical_id": r["canonical_id"], "checks_pass": not errors, "errors": sorted(set(errors)), "notes": notes,
              "status": r["status"], "warnings": r["warnings"], "source_hash_matches": "frozen_geometry_source_hash_mismatch" not in errors,
              "finite_raw_area_with_invalid_mask": finite_but_invalid,
              "paired_section_area_error": distribution(paired_errors.tolist()),
              "reference_section_coverage": float(a["section_valid"].mean()), "pointcloud_section_coverage": float(a["pc_section_valid"].mean()),
              "reference_wall_coverage": float(a["wall_map_valid"].mean()), "pointcloud_wall_coverage": float(a["wall_map_pc_valid"].mean()),
              "outside_core_points": p1["outside_core_points"], "outside_core_atlas_rows": p1["outside_core_atlas_rows"]}
    a.close()
    return output, r, paired_errors.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--split", type=Path, default=C.SPLIT_PATH,
                    help="geometry cohort identity to audit")
    ap.add_argument("--source-root", type=Path,
                    help="optional read-only snapshot root; otherwise use each report's recorded source")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        ap.error("full-array and source-hash audit must run in a Slurm allocation")
    expected = all_cases(args.split)
    paths = {p.parent.name.replace("__", "/"): p for p in (args.root / "cases").glob("*/report.json")}
    rows, reports, errors, pooled = [], [], [], []
    for cid in expected:
        if cid not in paths:
            errors.append(f"missing:{cid}")
            continue
        try:
            row, report, paired = audit_case(paths[cid], source_root=args.source_root)
            rows.append(row)
            reports.append(report)
            pooled.extend(paired)
            if not row["checks_pass"]:
                errors.append(f"array_or_source_check_failed:{cid}")
            print(json.dumps({"case": cid, "checks_pass": row["checks_pass"], "errors": row["errors"]}), flush=True)
        except Exception as exc:
            errors.append(f"audit_exception:{cid}:{exc}")
    extra = sorted(set(paths) - set(expected))
    if extra:
        errors.append(f"unexpected_cases:{extra}")
    warnings = Counter(w for r in reports for w in r["warnings"])
    inlet_errors = [abs(r["inlet_pointcloud_area_mm2"] / r["inlet_reference_area_mm2"] - 1)
                    for r in reports if r.get("inlet_pointcloud_valid") and r.get("inlet_reference_area_mm2")]
    density_valid, density_all = [], []
    variant_stats = {}
    for r in reports:
        for item in r["stability"]:
            v = {a["name"]: a for a in item["variants"]}
            density_all.append(item.get("pc_density_relative_change"))
            if item["pc_density_comparison_valid"]:
                density_valid.append(abs(item["pc_density_relative_change"]))
            for name, vv in v.items():
                stat = variant_stats.setdefault(name, {"probes": 0, "raw_contour_valid": 0, "pipeline_valid": 0, "comparison_valid": 0, "changes": []})
                stat["probes"] += 1
                for flag in ("raw_contour_valid", "pipeline_valid", "comparison_valid"):
                    stat[flag] += int(vv[flag])
                change = vv.get("relative_to_same_source_baseline")
                if vv["comparison_valid"] and change is not None:
                    stat["changes"].append(abs(change))
    for stat in variant_stats.values():
        stat["common_valid_same_source_relative_area_change_abs"] = distribution(stat.pop("changes"))
    config_hash_counts = dict(Counter(r["config_sha256"] for r in reports))
    if len(config_hash_counts) != 1:
        errors.append("candidate_configs_not_uniform")
    audit = {
        "schema": "wss_v6_geometry_candidate_audit_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "expected_cases": len(expected), "audited_cases": len(rows),
        "split_path": str(args.split.resolve()),
        "source_root_override": str(args.source_root.resolve()) if args.source_root is not None else None,
        "all_mechanical_checks_pass": not errors and len(rows) == len(expected), "errors": errors,
        "user_approved": False, "awaiting_user_review": True, "training_allowed": False,
        "mechanical_checks_do_not_validate_anatomical_accuracy_or_authorize_training": True,
        "algorithm_code": code_fingerprints(), "candidate_schema_counts": dict(Counter(r["schema"] for r in reports)),
        "candidate_code_hash_counts": dict(Counter(r["algorithm_code"]["combined_sha256"] for r in reports)),
        "candidate_config_hash_counts": config_hash_counts,
        "status_counts": dict(Counter(r["status"] for r in reports)), "warning_counts": dict(warnings),
        "reference_section_coverage": distribution([r["reference_section_coverage"] for r in rows]),
        "pointcloud_section_coverage": distribution([r["pointcloud_section_coverage"] for r in rows]),
        "reference_wall_coverage": distribution([r["reference_wall_coverage"] for r in rows]),
        "pointcloud_wall_coverage": distribution([r["pointcloud_wall_coverage"] for r in rows]),
        "pooled_common_valid_section_area_error": distribution(pooled),
        "case_p90_common_valid_section_area_error": distribution([r["paired_section_area_error"].get("p90") for r in rows]),
        "inlet_pointcloud_reference_area_error": distribution(inlet_errors),
        "half_density_common_valid_relative_area_change": distribution(density_valid),
        "half_density_probes_total": len(density_all), "half_density_probes_both_valid": len(density_valid),
        "P6_stability_by_variant": variant_stats,
        "exceptions_for_visual_review": [r for r in rows if r["warnings"]], "cases": rows,
        "schema_clarifications": {
            "invalid_area_values": "wall_map_area_mm2 and wall_map_pc_area_mm2 are NaN wherever valid=false; separate wall_map_raw_area_mm2 and wall_map_pc_raw_area_mm2 retain pre-mask values only for visual diagnosis. Derived slope/shape/context invalid values are also NaN.",
            "group_mapping": "feature_manifest G1_inlet=experiment G1; G2_section=G2; experiment G3=G1+G2; G3_slope=G4; G4_shape=G5; G5_context=G6; G6_CIA=G7",
            "area_error_scope": "agreement on common algorithm-valid sections only, not whole-cohort geometric accuracy; reference geometry itself is CFD-derived and masks exclude branch/nonstar failures",
        },
    }
    write_json(args.root / "audit.json", audit)
    lines = ["# P 几何候选独立审计", "", f"检查 {len(rows)}/{len(expected)} 例；机械检查{'全部通过' if audit['all_mechanical_checks_pass'] else '有失败，见 audit.json'}。仍全部等待用户几何确认，不能进入训练。", "",
             f"候选状态：{dict(audit['status_counts'])}。异常分类：{dict(warnings)}。", "",
             "| 量 | 中位数 | P90 | 范围 |", "| --- | ---: | ---: | --- |"]
    for key, label in [("reference_section_coverage", "参考截面有效覆盖"), ("pointcloud_section_coverage", "点云截面有效覆盖"),
                       ("pointcloud_wall_coverage", "点云壁面映射有效覆盖"), ("case_p90_common_valid_section_area_error", "每例共同有效截面面积误差 P90"),
                       ("inlet_pointcloud_reference_area_error", "点云入口对参考面积误差"), ("half_density_common_valid_relative_area_change", "共同有效减半密度面积变化")]:
        d = audit[key]
        if d.get("count"):
            lines.append(f"| {label} | {d['median']:.2%} | {d['p90']:.2%} | {d['min']:.2%}–{d['max']:.2%} |")
    lines += ["", "面积误差只描述两种算法共同有效的截面，必须与覆盖率一起解释；不代表分叉、非星形截面或全部壁面均达到这一误差。", "",
              "旧几何源按构建记录的数组哈希逐例复核；数组和拓扑、两源 valid/mask、插值、CIA 语义标记已核查。正式壁面面积在无效位置均为 NaN，另存 raw 数组仅供诊断。P6 原始轮廓和完整局部截面 gate 分开，正式差值仅在 comparison_valid 时保留。", "",
              "实验编号映射：G1 入口，G2 局部面积，G3 两者组合，G4 坡度，G5 非圆/偏心，G6 上下游，G7 CIA。当前 feature_candidates 的旧编号见 audit.json 的 schema_clarifications。", "", "需图审病例：", ""]
    lines += [f"- {r['canonical_id']}：{', '.join(r['warnings'])}" for r in rows if r["warnings"]]
    (args.root / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(json.dumps({"audit": str(args.root / 'audit.json'), "cases": len(rows), "pass": audit['all_mechanical_checks_pass'], "errors": errors}), flush=True)
    if not audit["all_mechanical_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
