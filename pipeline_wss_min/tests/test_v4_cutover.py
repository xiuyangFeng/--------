from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline_wss_min import config as C
from pipeline_wss_min import reporting
from pipeline_wss_min import v4_cutover as cutover
from pipeline_wss_min.preprocess import _spatially_align_wall_fields_to_reference
from pipeline_wss_min.v4_cutover import (
    AG_EXCLUDED_OR_PENDING,
    ag_units,
    check_v4_bundle,
    finalize_review,
    promote,
    snapshot_manifest,
)


def _write_case(root: Path, unit: str, *, frame: str = "stl_landmarks_v4",
                flip_z: bool = False, bad_rotation: bool = False) -> Path:
    case_dir = root / unit
    case_dir.mkdir(parents=True)
    rotation = np.eye(3)
    if bad_rotation:
        rotation[0, 0] = 2.0
    centroid = np.array([10.0, 20.0, 30.0])
    aligned = np.array([
        [-2.0, 0.0, -2.0], [2.0, 0.0, -2.0],
        [0.0, 0.0, 5.0], [0.0, 1.0, 1.0],
    ])
    raw = aligned @ np.linalg.inv(rotation) + centroid
    scale = float(np.abs(aligned).max())
    landmarks_aligned = np.array([
        [0.0, 0.0, -5.0 if flip_z else 5.0],
        [2.0, 0.0, -2.0],
        [-2.0, 0.0, -2.0],
    ])
    landmarks = landmarks_aligned @ np.linalg.inv(rotation) + centroid
    np.savez_compressed(
        case_dir / "bundle.npz",
        steps=np.array([1, 2]), peak_step=np.int32(2), coord_scale=np.float32(scale),
        transform_centroid=centroid, transform_rotation=rotation,
        transform_frame_version=np.asarray(frame),
        transform_landmark_source=np.asarray("original_stl"),
        transform_landmark_trunk_point=landmarks[0],
        transform_landmark_left_point=landmarks[1],
        transform_landmark_right_point=landmarks[2],
        wall_coords_norm=(aligned / scale).astype(np.float32),
        wall_coords_raw=raw.astype(np.float32),
        wall_nodenumber=np.arange(len(raw), dtype=np.int64),
        wall_wss=np.ones((2, len(raw)), dtype=np.float32),
        original_stl_path=np.asarray("dummy.stl"),
        original_stl_match_score=np.float64(0.0),
    )
    report = {
        "status": "ok", "frame_version": frame, "landmark_source": "original_stl",
        "nodenumber_alignment_ok": True, "wall_coord_mismatch_n_steps": 0,
        "unit_anomaly": False, "unit_extent_mismatch": False, "wall_crop_frac": 0.0,
        "centerline_translation_applied": False, "origin_kind": "flow_divider",
    }
    reporting.write_case_report(case_dir, report)
    return case_dir


def test_out_case_dir_supports_isolated_root(tmp_path: Path):
    assert C.out_case_dir("AG/fast", "CASE", out_root=tmp_path) == (
        tmp_path.resolve() / "AG/fast/CASE"
    )


def test_formal_ag_v4_split_has_77_and_no_excluded_or_pending_leakage():
    units = ag_units(C.DEFAULT_SPLIT_NAME)
    assert len(units) == 77
    assert not (set(units) & AG_EXCLUDED_OR_PENDING)


def test_v4_bundle_hard_gate_accepts_consistent_bundle(tmp_path: Path):
    unit = "AG/fast/CASE"
    _write_case(tmp_path, unit)
    row = check_v4_bundle(unit, tmp_path, enforce_ag_physical_limits=True)
    assert row["qa_pass"] is True
    assert row["reason"] == ""
    assert row["normalization_reconstruction_max_abs_error"] < 2e-5


def test_v4_bundle_hard_gate_rejects_legacy_and_bad_geometry(tmp_path: Path):
    legacy = "AG/fast/LEGACY"
    flipped = "AG/fast/FLIPPED"
    bad_rotation = "AG/fast/BAD_ROT"
    _write_case(tmp_path, legacy, frame="legacy_centerline_v3")
    _write_case(tmp_path, flipped, flip_z=True)
    _write_case(tmp_path, bad_rotation, bad_rotation=True)
    legacy_row = check_v4_bundle(legacy, tmp_path, enforce_ag_physical_limits=True)
    flipped_row = check_v4_bundle(flipped, tmp_path, enforce_ag_physical_limits=True)
    rotation_row = check_v4_bundle(bad_rotation, tmp_path, enforce_ag_physical_limits=True)
    assert not legacy_row["qa_pass"] and "wrong_frame_version" in legacy_row["reason"]
    assert not flipped_row["qa_pass"] and "axial_landmark_sign_failed" in flipped_row["reason"]
    assert not rotation_row["qa_pass"] and "rotation_not_orthonormal" in rotation_row["reason"]


def test_snapshot_manifest_hashes_bundle_and_report(tmp_path: Path):
    _write_case(tmp_path, "AG/fast/CASE")
    output = tmp_path / "manifest.json"
    payload = snapshot_manifest(tmp_path, output, C.DEFAULT_SPLIT_NAME)
    assert payload["n_bundles"] == 1
    assert payload["n_reports"] == 1
    assert output.is_file()
    loaded = json.loads(output.read_text())
    assert all(len(row["sha256"]) == 64 for row in loaded["files"])

    # Migration docs and operators may naturally pass the active AG directory.
    # It must produce the same data-root-relative manifest, never an empty one.
    output_from_ag = tmp_path / "manifest_from_ag.json"
    payload_from_ag = snapshot_manifest(tmp_path / "AG", output_from_ag, C.DEFAULT_SPLIT_NAME)
    assert payload_from_ag["legacy_root"] == payload["legacy_root"]
    assert payload_from_ag["files"] == payload["files"]


def test_finalize_review_applies_approvals_and_exclusions(tmp_path: Path):
    rows = [
        {"unit_id": "AG/fast/KEEP", "qa_pass": True, "soft_flags": "CROP"},
        {"unit_id": "AG/slow/DROP", "qa_pass": True, "soft_flags": ""},
        {"unit_id": "AAA/ruputer/KEEP", "qa_pass": True, "soft_flags": "AX"},
        {"unit_id": "AAA/unruputer/DROP", "qa_pass": True, "soft_flags": "STL-O"},
    ]
    (tmp_path / "v4_dataset_manifest.json").write_text(json.dumps({"rows": rows}))
    decisions = {
        "frame_version": "stl_landmarks_v4",
        "decision_date": "2026-07-16",
        "visual_review_signed_off": True,
        "excluded_units": [
            {"unit_id": "AG/slow/DROP", "reason": "manual"},
            {"unit_id": "AAA/unruputer/DROP", "reason": "manual"},
        ],
        "approved_review_units": ["AG/fast/KEEP", "AAA/ruputer/KEEP"],
        "auto_accept_all_unflagged_except_exclusions": True,
        "expected_final_counts": {"AG": 1, "AAA": 1},
    }
    decision_path = tmp_path / "decisions.json"
    decision_path.write_text(json.dumps(decisions))
    payload = finalize_review(tmp_path, decision_path)
    assert payload["summary"]["ready_for_promotion"] is True
    assert payload["summary"]["n_review_pending"] == 0
    assert json.loads((tmp_path / "v4_final_whitelist.json").read_text())["counts"] == {
        "AG": 1, "AAA": 1,
    }
    final = {row["unit_id"]: row for row in payload["rows"]}
    assert final["AG/fast/KEEP"]["final_status"] == "approved_by_user"
    assert final["AG/slow/DROP"]["final_status"] == "excluded_by_user"


def test_case_report_atomic_write_leaves_no_temp_file(tmp_path: Path):
    reporting.write_case_report(tmp_path, {"status": "ok"})
    assert json.loads((tmp_path / "report.json").read_text()) == {"status": "ok"}
    assert not list(tmp_path.glob(".report.*.tmp.json"))


def test_spatial_remap_repairs_coordinate_permutation_only():
    ref_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    ref_ids = np.array([10, 11, 12])
    fields = {
        "coords": ref_coords[[1, 2, 0]],
        "nodenumber": ref_ids.copy(),
        "wss": np.array([101.0, 102.0, 100.0]),
    }
    aligned, changed, distance = _spatially_align_wall_fields_to_reference(
        fields, ref_coords, ref_ids, step=1,
    )
    assert changed is True and distance == 0.0
    np.testing.assert_array_equal(aligned["coords"], ref_coords)
    np.testing.assert_array_equal(aligned["nodenumber"], ref_ids)
    np.testing.assert_array_equal(aligned["wss"], [100.0, 101.0, 102.0])

    moved = dict(fields, coords=fields["coords"] + np.array([0.0, 0.1, 0.0]))
    _, changed, _ = _spatially_align_wall_fields_to_reference(
        moved, ref_coords, ref_ids, step=2,
    )
    assert changed is False


def test_promote_requires_explicit_visual_review_confirmation(tmp_path: Path):
    try:
        promote(tmp_path / "staging", tmp_path / "active", tmp_path / "snapshot",
                tmp_path / "reports", confirm_reviewed=False)
    except RuntimeError as exc:
        assert "confirm-reviewed" in str(exc)
    else:
        raise AssertionError("promotion must be blocked without explicit review confirmation")


def test_promote_materializes_final_whitelist_and_keeps_full_staging(tmp_path: Path):
    active = tmp_path / "active"
    staging = tmp_path / "staging"
    reports = tmp_path / "reports"
    snapshot = tmp_path / "snapshot"
    formal = ag_units()
    excluded = "AG/slow/WANG_DENG_FENG"
    eligible = [unit for unit in formal if unit != excluded]
    for unit in formal:
        _write_case(staging, unit)
        _write_case(active, unit)
    for i in range(7):
        _write_case(active, f"AG/slow/LEGACY_EXTRA_{i}")
    reports.mkdir()
    (reports / "v4_manual_review_summary.json").write_text(json.dumps({
        "visual_review_signed_off": True,
        "ready_for_promotion": True,
    }))
    (reports / "v4_final_whitelist.json").write_text(json.dumps({
        "AG": eligible,
        "AAA": [],
        "counts": {"AG": 76, "AAA": 0},
    }))
    snapshot_manifest(
        active, reports / "AG_legacy_v3_snapshot_manifest.json", C.DEFAULT_SPLIT_NAME,
    )
    record = promote(
        staging, active, snapshot, reports, confirm_reviewed=True,
    )
    assert record["n_active_v4_bundles"] == 76
    assert len(list((active / "AG").glob("*/*/bundle.npz"))) == 76
    assert not (active / excluded / "bundle.npz").exists()
    assert len(list((staging / "AG").glob("*/*/bundle.npz"))) == 77
    assert len(list((snapshot / "AG").glob("*/*/bundle.npz"))) == 84


def test_promote_rolls_back_when_record_commit_fails(tmp_path: Path, monkeypatch):
    active = tmp_path / "active"
    staging = tmp_path / "staging"
    reports = tmp_path / "reports"
    snapshot = tmp_path / "snapshot"
    formal = ag_units()
    eligible = [u for u in formal if u != "AG/slow/WANG_DENG_FENG"]
    for unit in formal:
        _write_case(staging, unit)
        _write_case(active, unit)
    for i in range(7):
        _write_case(active, f"AG/slow/LEGACY_EXTRA_{i}")
    reports.mkdir()
    (reports / "v4_manual_review_summary.json").write_text(json.dumps({
        "visual_review_signed_off": True, "ready_for_promotion": True,
    }))
    (reports / "v4_final_whitelist.json").write_text(json.dumps({
        "AG": eligible, "AAA": [], "counts": {"AG": 76, "AAA": 0},
    }))
    snapshot_manifest(active, reports / "AG_legacy_v3_snapshot_manifest.json", C.DEFAULT_SPLIT_NAME)
    old_hashes = {
        p.relative_to(active): cutover._sha256(p)
        for p in active.glob("AG/*/*/*") if p.is_file()
    }
    real_write = cutover._write_json_atomic

    def fail_record(path, payload):
        if Path(path).name == "promotion_record.json":
            raise OSError("injected record commit failure")
        return real_write(path, payload)

    monkeypatch.setattr(cutover, "_write_json_atomic", fail_record)
    try:
        promote(staging, active, snapshot, reports, confirm_reviewed=True)
    except OSError as exc:
        assert "injected" in str(exc)
    else:
        raise AssertionError("fault injection must abort promotion")
    assert len(list((active / "AG").glob("*/*/bundle.npz"))) == 84
    assert not snapshot.exists()
    assert not (reports / "promotion_record.json").exists()
    assert old_hashes == {
        p.relative_to(active): cutover._sha256(p)
        for p in active.glob("AG/*/*/*") if p.is_file()
    }
