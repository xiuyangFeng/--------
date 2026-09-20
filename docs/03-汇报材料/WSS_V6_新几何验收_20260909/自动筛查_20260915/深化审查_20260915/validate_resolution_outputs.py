"""Validate recovery evidence and source provenance; emits a compact audit."""
from pathlib import Path
import hashlib
import json
import numpy as np
from test_route_recovery import test_descendant_extension_recovers_only_a_real_persistent_split

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]


def main():
    test_descendant_extension_recovers_only_a_real_persistent_split()
    rows = []
    for f in sorted((BASE / "bifurcation_route_recovery").glob("*.json")):
        x = json.loads(f.read_text())
        case = ROOT / "outputs/wss_v6_geometry_candidate_20260909/cases" / x["key"]
        assert x["source_geometry_sha256"] == hashlib.sha256((case / "geometry.npz").read_bytes()).hexdigest()
        assert x["source_report_sha256"] == hashlib.sha256((case / "report.json").read_bytes()).hexdigest()
        for name, digest in x["algorithm_sha256"].items():
            assert digest == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for j in x["junctions"]:
            if j["resolved_by_route_consensus"]:
                assert len(j["descendant_routes"]) == 4
                assert j["route_center_spread_mm"] <= 2
                assert j["consensus_bracket_mm"][1] - j["consensus_bracket_mm"][0] <= 2
                for v in j["descendant_routes"]:
                    sc = v["scan"]
                    c = sc["candidate"]
                    records = sc["records"]
                    i = next(i for i, r in enumerate(records) if r["s_mm"] == c["s_mm"])
                    assert any(r["state"] == "one" for r in records[:i])
                    support = records[i:i + sc["persistence_samples"]]
                    assert len(support) == sc["persistence_samples"]
                    assert all(r["state"] == "two" for r in support)
                    assert support[-1]["s_mm"] - support[0]["s_mm"] >= 1.0
                    for r in support:
                        probes = np.asarray(r["daughter_points_mm"])
                        assert probes.shape == (2, 3)
                        assert np.max(np.abs((probes - r["center_mm"]) @ np.asarray(r["normal"]))) <= 1e-4
                        assert len(r["daughter_contour_ids"][0]) == len(r["daughter_contour_ids"][1]) == 1
                        assert r["daughter_contour_ids"][0] != r["daughter_contour_ids"][1]
            else:
                assert j["consensus_slice_center_mm"] is None
        rows.append({"canonical_id": x["canonical_id"], "source_hashes_match": True, "algorithm_hashes_match": True, "consensus": all(j["resolved_by_route_consensus"] for j in x["junctions"]), "recovery_sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    assert len(rows) == 10
    summary = {"schema": "v6_route_recovery_validation_v1", "synthetic_known_split_and_no_split_regressions": "pass", "source_and_algorithm_hashes": "pass", "accepted_transition_local_coplanarity_unique_ownership_and_1mm_persistence": "pass", "n_cases": len(rows), "n_consensus": sum(r["consensus"] for r in rows), "cases": rows}
    (BASE / "case_resolution_validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
