"""``service upgrade``: (v0.14) preflight → stop → start → the new service rebuilds pending jobs and refreshes stale
reports in the background; a failed rebuild is reported, never keeps the service down."""
import json

import pytest

from wss_deploy import service as S


def _job(root, job_id, summary, status="done"):
    d = root / job_id
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"id": job_id, "status": status}), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return d


def test_pending_rebuilds_finds_cycle_results_without_cycle_findings(tmp_path):
    _job(tmp_path, "old_cycle", {"cycle": {"fields": {}}, "findings": {"items": []}})
    _job(tmp_path, "new_cycle", {"cycle": {"fields": {}}, "findings": {"items": [], "cycle_criteria": {}}})
    _job(tmp_path, "wall", {"findings": {"items": []}})
    _job(tmp_path, "running_cycle", {"cycle": {"fields": {}}}, status="running")
    assert S.pending_rebuilds(tmp_path) == ["old_cycle"]


def test_pending_rebuilds_also_finds_three_head_results_without_rrt_ecap(tmp_path):
    """v0.13: a TAWSS / OSI summary without the derived RRT / ECAP blocks gets the full rebuild on upgrade."""
    done = {"items": [], "cycle_criteria": {}}
    _job(tmp_path, "m1_v012", {"cycle": {"fields": {"tawss": {}, "osi": {}}}, "findings": done})
    _job(tmp_path, "m1_v013", {"cycle": {"fields": {"tawss": {}, "osi": {}, "rrt": {}, "ecap": {}}}, "findings": done})
    _job(tmp_path, "tawss_only", {"cycle": {"fields": {"tawss": {}}}, "findings": done})
    assert S.pending_rebuilds(tmp_path) == ["m1_v012"]


def test_upgrade_order_and_start_even_when_rebuild_fails(tmp_path, monkeypatch):
    """v0.14: the rebuild runs inside the *new* service (it holds the jobs-root lock) after the start."""
    _job(tmp_path, "old_cycle", {"cycle": {"fields": {}}, "findings": {"items": []}})
    calls = []
    monkeypatch.setattr(S, "preflight", lambda root, cfg, out: calls.append(("preflight",)))
    monkeypatch.setattr(S, "locate", lambda root, port=None: {"pid": 1, "port": 8765, "root_matches": True, "managed": True, "own": True})
    monkeypatch.setattr(S, "load_config", lambda root: {"port": 8765, "exists": True})
    monkeypatch.setattr(S, "stop", lambda root, port, **k: calls.append(("stop", port)))

    def broken(job_dir):
        calls.append(("rebuild", job_dir.name)); raise RuntimeError("boom")
    import wss_deploy.rebuild_report as R
    import wss_deploy.report_freshness as F
    from tests._c_helpers import manager
    monkeypatch.setattr(R, "rebuild", broken)
    monkeypatch.setattr(F, "stale_jobs", lambda root: [tmp_path / "old_cycle"])
    monkeypatch.setattr(F, "refresh", lambda job_dir, source: calls.append(("refresh", job_dir.name, source)))

    def start(root, **k):
        calls.append(("start", k.get("port")))
        service_manager = manager(root)          # the new service: loads the jobs and runs the queued maintenance
        service_manager.jobs["old_cycle"] = {"id": "old_cycle", "status": "done", "owner": "o", "version": 1}
        service_manager._run_maintenance()
        return {"pid": 2}
    monkeypatch.setattr(S, "start", start)
    result = S.upgrade(tmp_path, out=lambda *_: None)
    assert calls == [("preflight",), ("stop", 8765), ("start", 8765), ("rebuild", "old_cycle"), ("refresh", "old_cycle", "upgrade")]
    assert result["upgrade"]["rebuild_failed"] == ["old_cycle"] and result["upgrade"]["refreshed"] == 1


def test_upgrade_refuses_a_service_on_another_jobs_root(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "locate", lambda root, port=None: {"pid": 1, "port": 8765, "root_matches": False, "jobs_root": "/elsewhere"})
    monkeypatch.setattr(S, "load_config", lambda root: {"port": 8765, "exists": True})
    with pytest.raises(S.ServiceError):
        S.upgrade(tmp_path, out=lambda *_: None, preflight_check=False)
