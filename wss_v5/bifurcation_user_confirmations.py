"""Provenance-bound application of two explicitly confirmed first separations.

This sidecar records a user choice, not an algorithm consensus or a training
approval. It never changes source geometry, route scans, or graph coordinates.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

CASE_IDS = ("AAA/unruputer/LIU_XING_GUO", "AAA/unruputer/SUN_SHU_MING")
SCHEMA = "v6_bifurcation_user_first_separation_confirmation_v1"
RULE = "earliest_valid_independently_owned_two_contour_scan_record_without_persistence_requirement"
ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _first_event(variant, step_mm):
    records = variant["scan"]["records"]
    s = np.asarray([r["s_mm"] for r in records], dtype=float)
    if not len(s) or not np.all(np.isfinite(s)) or np.any(np.diff(s) <= 0):
        raise ValueError("route scan positions must be finite and strictly increasing")
    for index, record in enumerate(records):
        if record["state"] != "two":
            continue
        ids = record["daughter_contour_ids"]
        if len(ids) != 2 or any(len(x) != 1 for x in ids) or ids[0] == ids[1]:
            raise ValueError("first separate record lacks unique independent ownership")
        center = np.asarray(record["center_mm"], dtype=float)
        normal = np.asarray(record["normal"], dtype=float)
        probes = np.asarray(record["daughter_points_mm"], dtype=float)
        if (center.shape != (3,) or normal.shape != (3,) or probes.shape != (2, 3)
                or not all(np.all(np.isfinite(x)) for x in (center, normal, probes))
                or not np.isclose(np.linalg.norm(normal), 1., atol=1e-6)
                or np.max(np.abs((probes-center) @ normal)) > 1e-4):
            raise ValueError("first separate record has invalid or noncoplanar geometry")
        return {"routes": copy.deepcopy(variant["routes"]),
                "record_index_0based": index, "scan_step_mm": step_mm,
                "s_mm": record["s_mm"], "center_mm": record["center_mm"],
                "normal": record["normal"], "daughter_points_mm": record["daughter_points_mm"],
                "daughter_contour_ids": ids, "state": "two",
                "source_record_sha256": _json_sha(record)}
    raise ValueError("confirmed route has no independently separated record")


def build_confirmations(recovery_dir, source_root, user_statement, confirmed_at):
    """Build only the two authorized root choices from unchanged raw scans."""
    if not user_statement.strip() or not confirmed_at.strip():
        raise ValueError("exact user statement and confirmation time are required")
    recovery_dir, source_root = Path(recovery_dir), Path(source_root)
    cases = []
    for cid in CASE_IDS:
        key = cid.replace("/", "__")
        evidence_path = recovery_dir / (key + ".json")
        evidence = json.loads(evidence_path.read_text())
        if evidence["canonical_id"] != cid or evidence["key"] != key:
            raise ValueError("confirmation evidence case identity mismatch")
        for field, filename in (("source_geometry_sha256", "geometry.npz"),
                                ("source_report_sha256", "report.json")):
            if evidence[field] != _sha(source_root / "cases" / key / filename):
                raise ValueError("confirmation evidence does not match frozen source")
        for path, digest in evidence.get("algorithm_sha256", {}).items():
            if _sha(ROOT / path) != digest:
                raise ValueError("route extraction algorithm provenance changed")
        roots = [j for j in evidence["junctions"] if j["parent_segment"] == 0]
        if len(roots) != 1 or len(roots[0]["descendant_routes"]) != 4:
            raise ValueError("exactly one root junction and four route scans are required")
        junction = roots[0]
        first = [_first_event(v, evidence["parameters"]["step_mm"])
                 for v in junction["descendant_routes"]]
        first.sort(key=lambda e: (e["s_mm"], tuple(tuple(r) for r in e["routes"]),
                                  e["record_index_0based"]))
        earliest = [e for e in first if e["s_mm"] == first[0]["s_mm"]]
        cases.append({"canonical_id": cid, "parent_segment": 0,
                      "decision_origin": "explicit_user_confirmation",
                      "resolved_by_user_confirmation": True,
                      "selected_event": first[0], "earliest_event_set": earliest,
                      "first_event_per_route": first,
                      "canonical_tie_break": "minimum_s_then_lexicographic_route_pair_then_record_index",
                      "s_definition": "from_old_root_graph_node_along_extended_route_pair_mm",
                      "coordinate_definition": "scan_plane_center_in_frozen_source_mm_not_surface_carina",
                      "source_geometry_sha256": evidence["source_geometry_sha256"],
                      "source_report_sha256": evidence["source_report_sha256"],
                      "recovery_sha256": _sha(evidence_path),
                      "original_algorithm_resolved_by_route_consensus": junction["resolved_by_route_consensus"],
                      "original_algorithm_route_center_spread_mm": junction["route_center_spread_mm"],
                      "original_algorithm_disagreement_preserved": True})
    return {"schema": SCHEMA, "confirmed_at": confirmed_at,
            "user_statement_verbatim": user_statement,
            "interpretation_rule": RULE,
            "authorized_case_ids": list(CASE_IDS), "cases": cases,
            "confirmation_code_sha256": _sha(__file__),
            "scope": "only_selected_root_separation_positions_for_two_named_cases",
            "training_permission_granted": False, "whole_case_anatomy_approval_granted": False,
            "graph_or_source_geometry_modified": False}


def validate_confirmations(confirmations, recovery_dir, source_root):
    """Rebuild the exact choices and hashes; reject edited or stale sidecars."""
    if isinstance(confirmations, (str, Path)):
        confirmations = json.loads(Path(confirmations).read_text())
    if confirmations.get("schema") != SCHEMA:
        raise ValueError("unsupported confirmation schema")
    expected = build_confirmations(recovery_dir, source_root,
                                   confirmations["user_statement_verbatim"],
                                   confirmations["confirmed_at"])
    if confirmations != expected:
        raise ValueError("confirmation differs from exact authorized choice or provenance")
    return copy.deepcopy(expected)


def apply_confirmations(resolution, validated_confirmations):
    """Return an augmented copy; callers validate provenance before applying."""
    if (validated_confirmations.get("schema") != SCHEMA
            or set(validated_confirmations.get("authorized_case_ids", [])) != set(CASE_IDS)
            or validated_confirmations.get("training_permission_granted") is not False):
        raise ValueError("confirmation scope exceeds the two authorized root positions")
    confirmations = {c["canonical_id"]: c for c in validated_confirmations["cases"]}
    if set(confirmations) != set(CASE_IDS):
        raise ValueError("confirmation case scope mismatch")
    result = copy.deepcopy(resolution)
    applied = set()
    for case in result["cases"]:
        cid = case["canonical_id"]
        if cid not in confirmations:
            continue
        c = confirmations[cid]
        roots = [b for b in case["bifurcation_recovery"] if b["parent_segment"] == 0]
        if len(roots) != 1:
            raise ValueError("confirmed root is absent or duplicated in resolution")
        roots[0]["user_confirmation"] = copy.deepcopy(c)
        roots[0]["resolved_by_user_confirmation"] = True
        applied.add(cid)
    if applied != set(CASE_IDS):
        raise ValueError("both authorized cases must be present in the resolution")
    result["user_confirmation_summary"] = {"confirmed_cases": len(applied),
        "interpretation_rule": RULE, "user_statement_verbatim": validated_confirmations["user_statement_verbatim"],
        "confirmed_at": validated_confirmations["confirmed_at"], "training_permission_granted": False}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recovery-dir", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--user-statement", required=True)
    ap.add_argument("--confirmed-at", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = build_confirmations(args.recovery_dir, args.source_root, args.user_statement, args.confirmed_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({c["canonical_id"]: c["selected_event"] for c in result["cases"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
