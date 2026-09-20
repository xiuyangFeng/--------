"""Re-slice the six largest reference perturbations for visual diagnosis only.

This writes independent review artifacts, never changes candidate masks or
core geometry. Slurm-only because it performs new geometric intersections.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .geometry_candidate import DEFAULT_CONFIG, DEFAULT_OUTPUT, SCHEMA, code_fingerprints, section_geometry_gate, write_json
from .section_features import SurfaceSlicer, contains_origin, frame


def polygon_plane_hits(polygon, center, tangent, xyz, sid, s, segment, local_s, radius):
    e1, e2 = frame(tangent)
    candidates = np.flatnonzero(np.abs((xyz - center) @ tangent) <= .6)
    hits = []
    for i in candidates:
        p = np.column_stack(((polygon - xyz[i]) @ e1, (polygon - xyz[i]) @ e2))
        if len(p) >= 3 and contains_origin(p):
            hits.append(int(i))
    out = []
    for seg in np.unique(sid[hits]):
        rows = np.array([i for i in hits if sid[i] == seg], int)
        out.append({"segment_id": int(seg), "atlas_rows": rows.tolist(), "s_local_min_mm": float(s[rows].min()),
                    "s_local_max_mm": float(s[rows].max()), "samples": len(rows),
                    "same_branch_remote_samples": int(np.sum(np.abs(s[rows] - local_s) > max(5., 2 * radius))) if seg == segment else None})
    return out


def slim(result, center, tangent, xyz, sid, s, segment, local_s, radius):
    keys = ("valid", "reason", "raw_contour_valid", "raw_contour_reason", "pipeline_valid", "pipeline_reason",
            "area_mm2", "perimeter_mm", "req_mm", "roundness", "eccentricity", "centroid_mm", "contour_count", "closed_contour_count",
            "same_segment_crossing_group_count", "same_segment_nonlocal_crossing_count", "same_segment_plane_intersections")
    return {**{k: result.get(k) for k in keys}, "center_mm": center, "normal": tangent,
            "s_local_mm": local_s, "polygon_xyz_mm": result["polygon_xyz_mm"],
            "intersection_segments_xyz_mm": result["intersection_segments_xyz_mm"],
            "centerline_plane_hits_inside_selected_contour": polygon_plane_hits(result["polygon_xyz_mm"], center, tangent, xyz, sid, s, segment, local_s, radius)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--priority-file", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        ap.error("new slicing diagnostics must run in a Slurm allocation")
    priority_path = args.priority_file or args.root / "review_priority_stability.json"
    priority = json.loads(priority_path.read_text())
    gate_config = json.loads(DEFAULT_CONFIG.read_text())
    selected = [r for r in priority["variants"] if r["source"] == "reference_surface"][:args.top]
    output = args.output or args.root / ("priority_stability_diagnostics_" + SCHEMA.split("_")[-1].replace(".", "p"))
    records = []
    for rank, item in enumerate(selected, 1):
        cid, k = item["canonical_id"], item["section_index_zero_based"]
        case = args.root / "cases" / cid.replace("/", "__")
        report = json.loads((case / "report.json").read_text())
        z = np.load(case / "geometry.npz", allow_pickle=False)
        xyz, sid, ss = z["centerline_xyz_mm"].astype(float), z["centerline_segment_id"], z["centerline_s_local_mm"].astype(float)
        c, t = z["section_center_mm"][k].astype(float), z["section_tangent"][k].astype(float)
        local_s, radius = float(z["section_s_local_mm"][k]), float(z["section_radius_mis_mm"][k])
        seg = next(s for s in report["segments"] if s["segment_id"] == item["segment_id"])
        slicer = SurfaceSlicer(z["wall_xyz_mm"].astype(float), z["wall_triangles"])
        name = item["variant"]
        vc, vt, vs = c.copy(), t.copy(), local_s
        sign = -1 if "minus" in name else 1
        if name.startswith("shift"):
            delta = sign * report["config"]["stability_shift_mm"]
            vc += delta * t
            vs += delta
        elif name.startswith("tilt"):
            e1, _ = frame(t)
            angle = np.radians(report["config"]["stability_tilt_deg"])
            vt = np.cos(angle) * t + sign * np.sin(angle) * e1
        else:
            raise ValueError(name)
        before = section_geometry_gate(slicer.slice(c, t), c, t, seg, local_s, xyz, sid, gate_config, ss)
        after = section_geometry_gate(slicer.slice(vc, vt), vc, vt, seg, vs, xyz, sid, gate_config, ss)
        ba, aa = before.get("area_mm2"), after.get("area_mm2")
        row = {"rank": rank, "canonical_id": cid, "section_index_zero_based": k, "segment_id": seg["segment_id"], "segment_role": seg["role"],
               "variant": name, "stored_priority": item, "baseline": slim(before, c, t, xyz, sid, ss, seg["segment_id"], local_s, radius),
               "perturbed": slim(after, vc, vt, xyz, sid, ss, seg["segment_id"], vs, radius),
               "recomputed_relative_area_change": aa / ba - 1 if ba and aa else None,
               "closed_contour_count_changed": before["closed_contour_count"] != after["closed_contour_count"],
               "selected_contour_centroid_shift_mm": float(np.linalg.norm(after["centroid_mm"] - before["centroid_mm"])) if "centroid_mm" in after and "centroid_mm" in before else None,
               "note": "Recomputed from saved float32 review coordinates, not used to modify original masks; changed contour count or remote same-branch crossings need visual interpretation, not automatic truth labels.",
               "source_candidate_schema": report["schema"], "applied_gate_schema": SCHEMA, "applied_gate_config": gate_config,
               "awaiting_user_review": True, "training_allowed": False}
        dest = output / f"rank_{rank:02d}_{cid.replace('/', '__')}_section_{k}.json"
        write_json(dest, row)
        records.append({"rank": rank, "canonical_id": cid, "section_index_zero_based": k, "variant": name,
                        "diagnostic_json": str(dest.resolve()), "closed_contour_count_changed": row["closed_contour_count_changed"],
                        "baseline_area_mm2": ba, "perturbed_area_mm2": aa, "recomputed_relative_area_change": row["recomputed_relative_area_change"],
                        "baseline_gate": before["pipeline_valid"], "perturbed_gate": after["pipeline_valid"],
                        "baseline_plane_hits": row["baseline"]["centerline_plane_hits_inside_selected_contour"],
                        "perturbed_plane_hits": row["perturbed"]["centerline_plane_hits_inside_selected_contour"]})
        z.close()
        print(json.dumps({"rank": rank, "case": cid, "before": ba, "after": aa, "closed_contour_count_changed": row["closed_contour_count_changed"]}), flush=True)
    write_json(output / "manifest.json", {"schema": "geometry_priority_stability_contours_v1", "slurm_job_id": os.environ["SLURM_JOB_ID"],
                                          "source_priority_file": str(priority_path.resolve()), "applied_gate_schema": SCHEMA,
                                          "algorithm_code": code_fingerprints(), "awaiting_user_review": True, "training_allowed": False, "diagnostics": records})


if __name__ == "__main__":
    main()
