import copy
import hashlib
import json

import pytest

from wss_v5.bifurcation_user_confirmations import (
    CASE_IDS, apply_confirmations, build_confirmations, validate_confirmations,
)


def fixture_evidence(tmp_path):
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    source = tmp_path / "source"
    routes = [[[1, 5], [2, 3]], [[1, 5], [2, 4]], [[1, 6], [2, 3]], [[1, 6], [2, 4]]]
    for cid in CASE_IDS:
        key = cid.replace("/", "__")
        case = source / "cases" / key
        case.mkdir(parents=True)
        (case / "geometry.npz").write_bytes(b"unchanged-source-array")
        (case / "report.json").write_text("{}")
        variants = []
        for route_index, route in enumerate(routes):
            # A single early two event followed by ambiguity must win over
            # the later sustained transition. Tied routes retain their xyz.
            records = [{"s_mm": i*.25, "state": "two" if i in (1, 4, 5, 6) else "ambiguous",
                        "center_mm": [float(route_index), 0., 0.], "normal": [0., 0., 1.],
                        "daughter_points_mm": [[0., 0., 0.], [1., 1., 0.]],
                        "daughter_contour_ids": [[0], [1]]} for i in range(7)]
            variants.append({"routes": route, "scan": {"records": records,
                            "candidate": {"s_mm": 1.}}})
        data = {"canonical_id": cid, "key": key,
                "source_geometry_sha256": hashlib.sha256((case / "geometry.npz").read_bytes()).hexdigest(),
                "source_report_sha256": hashlib.sha256((case / "report.json").read_bytes()).hexdigest(),
                "algorithm_sha256": {}, "parameters": {"step_mm": .25},
                "junctions": [{"parent_segment": 0, "descendant_routes": list(reversed(variants)),
                               "resolved_by_route_consensus": False, "route_center_spread_mm": 10.}]}
        (recovery / (key + ".json")).write_text(json.dumps(data))
    return recovery, source


def test_first_actual_event_and_deterministic_ties_without_mutation(tmp_path):
    recovery, source = fixture_evidence(tmp_path)
    confirmation = build_confirmations(recovery, source, "两例以首次隔离点为准", "2026-09-16 Asia/Shanghai")
    for c in confirmation["cases"]:
        assert c["selected_event"]["s_mm"] == .25
        assert c["selected_event"]["record_index_0based"] == 1
        assert c["selected_event"]["routes"] == [[1, 5], [2, 3]]
        assert len(c["earliest_event_set"]) == 4
        assert len({tuple(e["center_mm"]) for e in c["earliest_event_set"]}) == 4
    validated = validate_confirmations(confirmation, recovery, source)
    resolution = {"cases": [{"canonical_id": cid, "bifurcation_recovery": [
        {"parent_segment": 0, "resolved_by_route_consensus": False}]} for cid in CASE_IDS]
        + [{"canonical_id": "other", "training_allowed": False, "bifurcation_recovery": []}]}
    original = copy.deepcopy(resolution)
    result = apply_confirmations(resolution, validated)
    assert resolution == original
    for c in result["cases"][:2]:
        assert c["bifurcation_recovery"][0]["resolved_by_user_confirmation"]
        assert c["bifurcation_recovery"][0]["resolved_by_route_consensus"] is False
    assert result["cases"][-1] == original["cases"][-1]
    assert confirmation["training_permission_granted"] is False


def test_rejects_tampered_coordinate_and_changed_source(tmp_path):
    recovery, source = fixture_evidence(tmp_path)
    confirmation = build_confirmations(recovery, source, "确认两例首次隔离", "2026-09-16 Asia/Shanghai")
    edited = copy.deepcopy(confirmation)
    edited["cases"][0]["selected_event"]["center_mm"][0] = 123.
    with pytest.raises(ValueError, match="differs"):
        validate_confirmations(edited, recovery, source)
    case = source / "cases" / CASE_IDS[0].replace("/", "__")
    (case / "geometry.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen source"):
        validate_confirmations(confirmation, recovery, source)


def test_rejects_ambiguous_ownership_and_widened_approval(tmp_path):
    recovery, source = fixture_evidence(tmp_path)
    confirmation = build_confirmations(recovery, source, "确认两例首次隔离", "2026-09-16 Asia/Shanghai")
    confirmation["training_permission_granted"] = True
    with pytest.raises(ValueError, match="scope"):
        apply_confirmations({"cases": []}, confirmation)
    f = recovery / (CASE_IDS[0].replace("/", "__") + ".json")
    data = json.loads(f.read_text())
    data["junctions"][0]["descendant_routes"][0]["scan"]["records"][1]["daughter_contour_ids"] = [[0], [0]]
    f.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="independent ownership"):
        build_confirmations(recovery, source, "确认两例首次隔离", "2026-09-16 Asia/Shanghai")
