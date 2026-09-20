"""Read-only recovery of censored daughter scans using descendant paths.

The old locator stops at the shorter daughter graph segment. Graph junctions
can occur upstream of a wall carina, so scan each descendant-path combination
and require agreement. These slice-transition positions remain sidecars, not
replacement graph nodes or anatomical approvals.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from wss_v5.bifurcation_locator import _scan, _plain
from wss_v5.section_features import SurfaceSlicer

BASE = Path(__file__).resolve().parent
CAND = ROOT / "outputs/wss_v6_geometry_candidate_20260909/cases"


def paths_to_leaves(children, seg):
    return [[seg]] if not children[seg] else [[seg] + p for c in children[seg] for p in paths_to_leaves(children, c)]


def route_arrays(z, route):
    xyz, ss = [], []
    offset = 0.
    for seg in route:
        rows = np.flatnonzero(z["centerline_segment_id"] == seg)
        rows = rows[np.argsort(z["centerline_s_local_mm"][rows])]
        p = z["centerline_xyz_mm"][rows].astype(float)
        s = z["centerline_s_local_mm"][rows].astype(float)
        if xyz:
            # Maintain continuous arc parameter while retaining source points.
            offset += float(np.linalg.norm(p[0] - xyz[-1][-1]))
        xyz.append(p)
        ss.append(s - s[0] + offset)
        offset = float(ss[-1][-1])
    return np.concatenate(xyz), np.concatenate(ss)


def run_case(key, out, step=1.):
    start = time.monotonic()
    p = CAND / key
    rep = json.loads((p / "report.json").read_text())
    z = np.load(p / "geometry.npz")
    children = {int(s["segment_id"]): s["children"] for s in rep["segments"]}
    old = json.loads((p / "bifurcation_locator.json").read_text())
    failed = [j for j in old["junctions"] if any(sc["status"] == "no_persistent_two_contour_transition" for sc in j["scans"])]
    slicer = SurfaceSlicer(z["wall_xyz_mm"].astype(float), z["wall_triangles"].astype(np.int64))
    junctions = []
    for j in failed:
        parent = j["parent_segment"]
        left, right = [paths_to_leaves(children, c) for c in j["daughter_segments"]]
        variants = []
        for l in left:
            for r in right:
                a, sa = route_arrays(z, l)
                b, sb = route_arrays(z, r)
                xyz = np.concatenate([a, b])
                ss = np.concatenate([sa, sb])
                sid = np.concatenate([np.ones(len(a), int), np.full(len(b), 2, int)])
                scan = _scan(xyz, sid, ss, 0, [1, 2], slicer, step=step)
                variants.append({"routes": [l, r], "scan": scan})
        candidates = [v["scan"]["candidate"] for v in variants]
        complete = bool(candidates) and all(c is not None for c in candidates)
        centers = np.array([c["center_mm"] for c in candidates if c is not None])
        spread = float(np.max(np.linalg.norm(centers[:, None] - centers[None], axis=2))) if len(centers) else None
        bracket = [float(min(c["bracket_mm"][0] for c in candidates)), float(max(c["bracket_mm"][1] for c in candidates))] if complete else None
        consensus = complete and spread <= 2.0 and bracket[1] - bracket[0] <= 2.0
        junctions.append({"parent_segment": parent, "old_status": j["scans"][0]["status"], "old_scan_end_mm": j["scans"][0]["records"][-1]["s_mm"], "descendant_routes": variants, "all_routes_have_transition": complete, "route_center_spread_mm": spread, "consensus_bracket_mm": bracket, "consensus_slice_center_mm": centers.mean(axis=0).tolist() if consensus else None, "consensus_center_definition": "mean_of_four_route_variant_slice_centers_not_a_surface_carina", "resolved_by_route_consensus": consensus})
    result = {"schema": "v6_bifurcation_descendant_route_recovery_v1", "canonical_id": rep["canonical_id"], "key": key, "source_geometry_sha256": hashlib.sha256((p / "geometry.npz").read_bytes()).hexdigest(), "source_report_sha256": hashlib.sha256((p / "report.json").read_bytes()).hexdigest(), "algorithm_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__), ROOT / "wss_v5/bifurcation_locator.py", ROOT / "wss_v5/section_features.py"]}, "source_unchanged": True, "training_allowed": False, "definition": "local_slice_shared_to_separate_descendant_route_contours_not_wall_carina_or_replacement_graph_node", "parameters": {"step_mm": step, "minimum_two_state_extent_mm": 1.0, "max_route_bracket_width_mm": 2.0, "max_route_center_spread_mm": 2.0}, "junctions": junctions, "seconds": time.monotonic() - start}
    out.mkdir(parents=True, exist_ok=True)
    (out / (key + ".json")).write_text(json.dumps(_plain(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"case": rep["canonical_id"], "seconds": result["seconds"], "junctions": [{k:v for k,v in j.items() if k != "descendant_routes"} for j in junctions]}), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--step", type=float, default=1.)
    ap.add_argument("--out", type=Path, default=BASE / "bifurcation_route_recovery")
    args = ap.parse_args()
    for key in args.keys:
        run_case(key, args.out, args.step)
