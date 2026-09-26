"""v0.14 rebuild_report: uint8 branch ids in refreshed pages, provenance refresh, the offline writer lock,
and the regression-key contract for summaries that now carry provenance."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_report import _minimal_report_inputs  # noqa: E402

from wss_deploy import rebuild_report as RB, report as R, regress  # noqa: E402

SEGMENT_KEYS = {"ms": ("mesh", "segment"), "ps": ("cloud", "segment"), "cs": ("centerline", "segment")}


def _page(tmp_path, *, mesh_segment, cloud_segment, center_segment):
    meta, mesh, cloud, centerline = _minimal_report_inputs()
    mesh, cloud, centerline = ({**mesh, "segment": mesh_segment}, {**cloud, "segment": cloud_segment},
                               {**centerline, "segment": center_segment})
    job = tmp_path / "job"
    job.mkdir()
    R.build_html(job / "report.html", meta, mesh, cloud, centerline)
    return job, {"mesh": mesh, "cloud": cloud, "centerline": centerline}


@pytest.mark.parametrize("values, expected_dt", [
    ((np.ones(4, np.int16), np.full(4, 2, np.int16), np.array([0, 7], np.int16)), {"ms": "u8", "ps": "u8", "cs": "u8"}),   # v0.14 uint8 page
    ((np.full(4, 300, np.int32), np.full(4, -1, np.int32), np.array([256, 1], np.int32)), None),                           # int32 (= pre-v0.14 layout)
    ((np.ones(4, np.int16), np.full(4, 300, np.int32), np.array([3, 4], np.int16)), {"ms": "u8", "cs": "u8"}),             # mixed
])
def test_refresh_ui_only_decodes_branch_ids_with_the_recorded_dtype(tmp_path, monkeypatch, values, expected_dt):
    job, original = _page(tmp_path, mesh_segment=values[0], cloud_segment=values[1], center_segment=values[2])
    html = (job / "report.html").read_text(encoding="utf-8")
    _, arrays = RB._embedded_json(html, "wss-report-arrays")
    assert arrays.get("dt") == expected_dt
    seen = {}
    real = R.build_html

    def capture(out_path, meta, mesh, cloud, centerline, **kwargs):
        seen.update(mesh=mesh, cloud=cloud, centerline=centerline)
        return real(out_path, meta, mesh, cloud, centerline, **kwargs)

    monkeypatch.setattr(R, "build_html", capture)
    result = RB.refresh_ui_only(job)
    assert result["ui_only"] is True
    for key, (part, field) in SEGMENT_KEYS.items():
        decoded, source = seen[part][field], original[part][field]
        assert np.array_equal(decoded.astype(np.int64), np.asarray(source).astype(np.int64)), key
        assert decoded.dtype == (np.uint8 if (expected_dt or {}).get(key) == "u8" else np.int32), key
    refreshed = (job / "report.html").read_text(encoding="utf-8")
    assert RB._embedded_json(refreshed, "wss-report-arrays")[1] == arrays          # science payload untouched


def test_rebuild_cli_refuses_while_a_service_holds_the_lock_and_force_overrides(tmp_path, capsys):
    from wss_deploy.service import acquire_lock
    root = tmp_path / "jobs"
    root.mkdir()
    lock = acquire_lock(root, purpose="serve")
    try:
        assert RB.main(["--jobs-root", str(root)]) == 2
        err = capsys.readouterr().err
        assert "service upgrade --rebuild" in err and "--force" in err
        assert RB.main(["--jobs-root", str(root), "--force"]) == 0             # no finished jobs: nothing to do
        assert "--force" in capsys.readouterr().err
    finally:
        lock.release()
    assert RB.main(["--jobs-root", str(root)]) == 0                            # lock free: runs, and releases it
    again = acquire_lock(root, purpose="serve")
    again.release()


def test_finish_refreshes_the_provenance_keys(tmp_path, monkeypatch):
    from wss_deploy import schema
    job = tmp_path / "job"
    job.mkdir()
    meta, mesh, cloud, centerline = _minimal_report_inputs()
    R.build_html(job / "report.html", meta, mesh, cloud, centerline)
    np.savez_compressed(job / "field.npz", pts=np.zeros((1, 3), np.float32))
    stale = {**meta, "analysis_version": "2026-01-01", "git_dirty": None}
    monkeypatch.setattr(RB, "write_run_manifest", lambda *a, **k: None)
    RB._finish(job, {}, stale, ["summary.json"], {})
    summary = json.loads((job / "summary.json").read_text(encoding="utf-8"))
    expected = schema.summary_provenance()
    assert {k: summary[k] for k in expected} == expected and summary["analysis_version"] == schema.ANALYSIS_VERSION


def test_regression_ignores_provenance_and_compares_the_cycle_block():
    assert "cycle" in regress.SUMMARY_KEYS and "fields" not in regress.SUMMARY_KEYS
    old = {"peak": {"p99_pa": 1.0}, "cycle": {"fields": {"tawss": {"p99": 2.0}}}}
    new = {**old, "analysis_version": "2026-09-24", "deploy_version": "0.14", "git_describe": "x-dirty",
           "git_dirty": True, "code_source_hash": "a" * 64}
    diffs = []
    for key in regress.SUMMARY_KEYS:
        if key in old or key in new:
            regress._walk(old.get(key), new.get(key), key, 1e-5, diffs)
    assert diffs == []
    nested = []
    regress._walk({"a": {"git_dirty": False}}, {"a": {"git_dirty": True, "analysis_version": "v"}}, "x", 1e-5, nested)
    assert nested == []
    changed = []
    regress._walk(old["cycle"], {"fields": {"tawss": {"p99": 2.5}}}, "cycle", 1e-5, changed)
    assert changed and changed[0][1] == "numeric"
