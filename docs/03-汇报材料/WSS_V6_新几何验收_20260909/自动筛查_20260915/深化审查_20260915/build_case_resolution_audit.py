"""Reproduce case-level evidence without modifying candidate arrays or policy."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
from resolve_bifurcation_routes import route_arrays

CAND = ROOT / "outputs/wss_v6_geometry_candidate_20260909/cases"


def candidate_record(scan):
    c = scan["candidate"]
    return min(scan["records"], key=lambda r: abs(r["s_mm"] - c["s_mm"])) if c else None


def project_arc(path, point):
    edge = np.diff(path, axis=0)
    lengths = np.linalg.norm(edge, axis=1)
    a = np.clip(np.sum((np.asarray(point) - path[:-1]) * edge, axis=1) / np.maximum(lengths**2, 1e-20), 0., 1.)
    q = path[:-1] + a[:, None] * edge
    d = np.linalg.norm(q - point, axis=1)
    i = int(d.argmin())
    return float(np.r_[0., np.cumsum(lengths)][i] + a[i] * lengths[i]), float(d[i])


def cia_interval_measurements(z, old, recovery):
    lookup = {j["parent_segment"]: j for j in old["junctions"]}
    root = lookup[0]["scans"][0]
    if root["candidate"]:
        root_variants = [{"routes": [[1], [2]], "scan": root}]
    elif recovery and recovery["junctions"][0]["resolved_by_route_consensus"]:
        root_variants = recovery["junctions"][0]["descendant_routes"]
    else:
        return []
    measurements = []
    for v in root_variants:
        rec = candidate_record(v["scan"])
        for side_i, side in enumerate([1, 2]):
            proximal = rec["daughter_points_mm"][side_i]
            distal_scan = lookup[side]["scans"][0]
            if not distal_scan["candidate"]:
                continue
            distal_rec = candidate_record(distal_scan)
            for leaf_i, leaf in enumerate(lookup[side]["daughter_segments"]):
                if len(v["routes"][side_i]) > 1 and v["routes"][side_i][-1] != leaf:
                    continue
                route = [side, leaf]
                path, ss = route_arrays(z, route)
                distal = distal_rec["daughter_points_mm"][leaf_i]
                s0, e0 = project_arc(path, proximal)
                s1, e1 = project_arc(path, distal)
                length = s1 - s0
                chord = float(np.linalg.norm(np.asarray(distal) - proximal))
                measurements.append({"side_segment": side, "route": route, "root_pair_routes": v["routes"], "proximal_point_mm": proximal, "distal_point_mm": distal, "projection_error_mm": max(e0, e1), "proximal_route_arc_mm": s0, "distal_route_arc_mm": s1, "interval_length_mm": length, "interval_chord_mm": chord, "interval_tortuosity": length / chord if chord > 0 else None, "positive_interval": length > 0})
    return measurements


def main():
    baseline = BASE / "case_resolution_input_baseline.csv"
    if not baseline.exists():
        triage = pd.read_csv(BASE / "case_triage.csv")
        triage[triage.needs_user_geometry].to_csv(baseline, index=False)
    triage = pd.read_csv(baseline)
    rows = []
    details = []
    for entry in triage[triage.needs_user_geometry].to_dict("records"):
        cid = entry["canonical_id"]
        key = cid.replace("/", "__")
        cdir = CAND / key
        rep = json.loads((cdir / "report.json").read_text())
        z = np.load(cdir / "geometry.npz")
        old = json.loads((cdir / "bifurcation_locator.json").read_text())
        rp = BASE / "bifurcation_route_recovery" / (key + ".json")
        recovery = json.loads(rp.read_text()) if rp.exists() else None
        p1 = rep["P1_centerline"]
        outside = []
        for i in p1["outside_core_atlas_rows"]:
            seg = int(z["centerline_segment_id"][i])
            s = float(z["centerline_s_local_mm"][i])
            end = float(z["centerline_s_local_mm"][z["centerline_segment_id"] == seg].max())
            outside.append({"row": i, "segment": seg, "s_mm": s, "nearest_segment_end_distance_mm": min(s, end - s)})
        measures = cia_interval_measurements(z, old, recovery)
        record = {"canonical_id": cid, "key": key, "source_geometry_sha256": hashlib.sha256((cdir / "geometry.npz").read_bytes()).hexdigest(), "source_report_sha256": hashlib.sha256((cdir / "report.json").read_bytes()).hexdigest(), "prior_review_causes": [], "source_centerline": p1, "outside_point_locations": outside, "bifurcation_recovery": None if recovery is None else [{k:v for k,v in j.items() if k != "descendant_routes"} for j in recovery["junctions"]], "cia_slice_transition_intervals": measures, "interpretation": []}
        row = {"canonical_id": cid, "previously_needs_user_geometry": True, "source_report_warnings": ";".join(rep["warnings"]), "closed_reference_valid": p1["closed_reference_valid"], "residual_edges": p1["remaining_boundary_or_nonmanifold_edges"], "outside_core_points": p1["outside_core_points"], "outside_max_distance_to_segment_end_mm": max((x["nearest_segment_end_distance_mm"] for x in outside), default=0.), "bifurcation_old_undetermined": int(entry["bif_undetermined"]), "bifurcation_route_consensus_resolved": bool(recovery and all(j["resolved_by_route_consensus"] for j in recovery["junctions"])), "coverage_pass": entry["coverage_pass"]}
        if recovery:
            j = recovery["junctions"][0]
            row.update(bifurcation_route_spread_mm=j["route_center_spread_mm"], bifurcation_new_s_lo_mm=j["consensus_bracket_mm"][0] if j["consensus_bracket_mm"] else None, bifurcation_new_s_hi_mm=j["consensus_bracket_mm"][1] if j["consensus_bracket_mm"] else None)
        for side, label in [(1, "left"), (2, "right")]:
            ss = [m for m in measures if m["side_segment"] == side]
            row[f"graph_{label}_cia_length_mm"] = entry[f"cia_{label}_length_mm"]
            row[f"graph_{label}_cia_tortuosity"] = entry[f"cia_{label}_tortuosity"]
            row[f"transition_{label}_cia_length_min_mm"] = min((m["interval_length_mm"] for m in ss), default=None)
            row[f"transition_{label}_cia_length_max_mm"] = max((m["interval_length_mm"] for m in ss), default=None)
            row[f"transition_{label}_cia_tortuosity_min"] = min((m["interval_tortuosity"] for m in ss), default=None)
            row[f"transition_{label}_cia_tortuosity_max"] = max((m["interval_tortuosity"] for m in ss), default=None)
        old_cia_outlier = any(entry[f"cia_{s}_length_mm"] < 15 or entry[f"cia_{s}_length_mm"] > 120 or entry[f"cia_{s}_tortuosity"] > 2 for s in ["left", "right"])
        row["graph_cia_band_outlier"] = old_cia_outlier
        row["transition_cia_intervals_computable"] = bool(measures) and all(m["positive_interval"] and m["projection_error_mm"] < 1e-3 for m in measures)
        row["transition_cia_inside_old_bands"] = bool(measures) and all(15 <= m["interval_length_mm"] <= 120 and 1 <= m["interval_tortuosity"] <= 2 for m in measures)
        causes = []
        if not p1["closed_reference_valid"] or p1["outside_core_points"]:
            causes.append("source_inside_or_surface_audit")
        if entry["bif_undetermined"]:
            causes.append("bifurcation_truncated_scan")
        if old_cia_outlier:
            causes.append("graph_based_cia_band")
        if not entry["coverage_pass"]:
            causes.append("coverage")
        if not causes:
            causes.append("stale_hardcoded_review_case")
        record["prior_review_causes"] = causes
        if row["bifurcation_route_consensus_resolved"]:
            record["interpretation"].append("The shorter graph daughter truncated the old search before the persistent wall-contour split. Four descendant-route combinations now agree; review of failure-to-find alone can be cleared.")
        if entry["bif_undetermined"] and not row["bifurcation_route_consensus_resolved"]:
            record["interpretation"].append("Descendant-route transitions disagree or contain wide ambiguity gaps. Keep this location unresolved; a distant transition must not be accepted to remove a warning.")
        if old_cia_outlier and row["transition_cia_intervals_computable"]:
            record["interpretation"].append("The old CIA uses graph bifurcation nodes that may precede physical lumen separation. The additional transition-to-transition lengths are reproducible topology-defined measurements, not a replacement anatomical convention or proof of a faulty source geometry.")
        if causes == ["stale_hardcoded_review_case"]:
            record["interpretation"].append("No current source warning or numerical review cause; remove from the old hardcoded human-review list.")
        row["prior_review_causes"] = ";".join(causes)
        row["audit_scope"] = "independent_sidecar_no_candidate_mutation"
        rows.append(row)
        details.append(record)
    pd.DataFrame(rows).to_csv(BASE / "case_resolution_audit.csv", index=False)
    result = {"schema": "v6_case_resolution_evidence_v1", "scope": "The 29 cases flagged by the prior sidecar; preserves existing data and training permissions.", "algorithm_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__), BASE / "resolve_bifurcation_routes.py"]}, "limitations": ["This audit does not repair source triangles or centerlines; source repair validation is a separate artifact.", "Slice-transition intervals are not wall carina coordinates and do not replace the original graph atlas.", "CIA length/tortuosity bands are diagnostic prioritization, not normative anatomical acceptance limits."], "summary": {"previous_review_cases": len(rows), "outside_core_cases": sum(r["outside_core_points"] > 0 for r in rows), "residual_edge_cases": sum(not r["closed_reference_valid"] for r in rows), "bifurcation_old_undetermined": sum(r["bifurcation_old_undetermined"] for r in rows), "bifurcation_route_consensus_resolved": sum(r["bifurcation_route_consensus_resolved"] for r in rows), "graph_cia_outliers": sum(r["graph_cia_band_outlier"] for r in rows), "cia_outliers_now_inside_diagnostic_bands": sum(r["graph_cia_band_outlier"] and r["transition_cia_inside_old_bands"] for r in rows), "stale_hardcoded_review_cases": [r["canonical_id"] for r in rows if r["prior_review_causes"] == "stale_hardcoded_review_case"]}, "cases": details}
    (BASE / "case_resolution_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
