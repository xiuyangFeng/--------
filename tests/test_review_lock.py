"""Review sign-off locks a finished result; reopening lifts the lock; the decision reaches the result files."""
from __future__ import annotations

import http.client
import json
import threading

import pytest

from wss_deploy.jobs import JobError, JobManager, fresh_review
from wss_deploy.server import ServiceHTTPServer, SessionStore

META_BLOCK = '<!--WSS_META_START--><script id="wss-report-meta" type="application/json">{}</script><!--WSS_META_END-->'


def _manager(tmp_path):
    return JobManager(tmp_path, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {},
                      mapping_validator=lambda *_a, **_k: [])


def _finished(manager, owner="owner", case_id="case", *, with_files=True):
    job = manager.create(owner, content=b"solid", filename=f"{case_id}.stl", case_id=case_id)
    with manager.lock:
        record = manager.jobs[job["id"]]
        record.update(status="done", stage="B", phase="计算完成", mapping={"3": "out-le", "4": "out-li", "5": "out-re", "6": "out-ri"},
                      a={"stage": "A", "input_check": {"ok": True, "status": "pass"}, "centerline": {"hard_pass": True, "openings": []},
                         "proposal": {"mapping": {"3": "out-le", "4": "out-li", "5": "out-re", "6": "out-ri"}}},
                      summary={"peak": {"p99_pa": 1.0}, "fields": {"wss": {"units": "Pa"}}})
        manager._event(record, "finished")
        job_dir = manager.root / job["id"]
        if with_files:
            (job_dir / "summary.json").write_text(json.dumps({"case_id": case_id, "peak": {"p99_pa": 1.0}, "audit": {"x": 1},
                                                              "fields": {"wss": {"units": "Pa", "location": "wall", "kind": "scalar"}},
                                                              "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21}]}), encoding="utf-8")
            (job_dir / "report.html").write_text("<html><body>" + META_BLOCK + "</body></html>", encoding="utf-8")
            (job_dir / "run_manifest.json").write_text(json.dumps({"outputs": {"summary.json": {"path": "summary.json"}, "report.html": {"path": "report.html"}}}), encoding="utf-8")
    return manager.get(job["id"], owner)


def test_new_jobs_start_unreviewed_and_light_snapshot_carries_review_and_family(tmp_path):
    manager = _manager(tmp_path)
    job = _finished(manager)
    assert job["review"] == fresh_review()
    row = manager.list("owner")[0]
    assert row["review"]["status"] == "unreviewed" and row["family"] == "wall"


def test_approve_locks_retry_override_and_delete_until_reopened(tmp_path):
    manager = _manager(tmp_path)
    job = _finished(manager)
    with pytest.raises(JobError):
        manager.review(job["id"], "owner", {"version": job["version"], "decision": "approve", "reviewer": ""})
    approved = manager.review(job["id"], "owner", {"version": job["version"], "decision": "approve", "reviewer": "R-01", "note": "看过"})
    assert approved["review"]["status"] == "reviewed" and approved["review"]["by"] == "R-01" and approved["review"]["version"] == approved["version"]
    assert approved["summary"]["review"]["status"] == "reviewed"
    assert approved["events"][-1]["action"] == "review_approved"
    version = approved["version"]
    with pytest.raises(JobError) as info:
        manager.retry(job["id"], "owner", {"version": version})
    assert info.value.status == 409 and "重新打开" in str(info.value)
    with pytest.raises(JobError) as info:
        manager.confirm(job["id"], "owner", {"version": version, "override": True, "acknowledged": True, "stage": "B",
                                             "mapping": {"3": "out-li", "4": "out-le", "5": "out-re", "6": "out-ri"}})
    assert info.value.status == 409
    with pytest.raises(JobError) as info:
        manager.delete(job["id"], "owner", {"version": version})
    assert info.value.status == 409 and (tmp_path / job["id"]).is_dir()
    batch = manager.delete_many("owner", [{"id": job["id"], "version": version}])
    assert batch["deleted_count"] == 0 and batch["results"][0]["error"]["status"] == 409
    with pytest.raises(JobError) as info:  # approving twice is refused
        manager.review(job["id"], "owner", {"version": version, "decision": "approve", "reviewer": "R-02"})
    assert info.value.status == 409
    with pytest.raises(JobError):  # reopening requires a reason
        manager.review(job["id"], "owner", {"version": version, "decision": "reopen", "reviewer": "R-02", "note": ""})
    reopened = manager.review(job["id"], "owner", {"version": version, "decision": "reopen", "reviewer": "R-02", "note": "出口命名可能有误"})
    assert reopened["review"]["status"] == "reopened" and reopened["events"][-1]["action"] == "review_reopened"
    assert [item["status"] for item in reopened["review_history"]] == ["reviewed", "reopened"]
    # unlocked again: override recompute is accepted and queues stage B
    queued = manager.confirm(job["id"], "owner", {"version": reopened["version"], "override": True, "acknowledged": True, "stage": "B",
                                                  "mapping": {"3": "out-li", "4": "out-le", "5": "out-re", "6": "out-ri"}})
    assert queued["status"] == "queued"


def test_rerun_of_a_reviewed_job_is_allowed_and_starts_unreviewed(tmp_path):
    manager = _manager(tmp_path)
    job = _finished(manager)
    approved = manager.review(job["id"], "owner", {"version": job["version"], "decision": "approve", "reviewer": "R-01"})
    new = manager.rerun(job["id"], "owner", {"version": approved["version"]})
    assert new["source_job_id"] == job["id"] and new["review"]["status"] == "unreviewed"
    assert manager.get(job["id"], "owner")["review"]["status"] == "reviewed"


def test_review_is_written_to_summary_report_meta_and_manifest(tmp_path):
    manager = _manager(tmp_path)
    job = _finished(manager)
    manager.review(job["id"], "owner", {"version": job["version"], "decision": "approve", "reviewer": "R-01", "note": "ok"})
    job_dir = tmp_path / job["id"]
    summary = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["review"]["status"] == "reviewed" and summary["review"]["by"] == "R-01"
    assert summary["audit"]["review"]["status"] == "reviewed" and summary["audit"]["x"] == 1
    assert summary["audit"]["review_history"][-1]["status"] == "reviewed"
    html = (job_dir / "report.html").read_text(encoding="utf-8")
    embedded = json.loads(html.split('type="application/json">', 1)[1].split("</script>", 1)[0])
    assert embedded["review"]["status"] == "reviewed" and embedded["review"]["by"] == "R-01"
    manifest = json.loads((job_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["audit"]["review"]["status"] == "reviewed"
    assert set(manifest["outputs"]) == {"summary.json", "report.html"}


def test_review_without_result_files_only_updates_the_record(tmp_path):
    manager = _manager(tmp_path)
    job = _finished(manager, with_files=False)
    approved = manager.review(job["id"], "owner", {"version": job["version"], "decision": "approve", "reviewer": "R-01"})
    assert approved["review"]["status"] == "reviewed"
    assert not (tmp_path / job["id"] / "summary.json").exists()


def test_review_needs_done_status_owner_and_version(tmp_path):
    manager = _manager(tmp_path)
    queued = manager.create("owner", content=b"solid", filename="x.stl")
    with pytest.raises(JobError) as info:
        manager.review(queued["id"], "owner", {"version": queued["version"], "decision": "approve", "reviewer": "R"})
    assert info.value.status == 409
    job = _finished(manager)
    with pytest.raises(JobError) as info:
        manager.review(job["id"], "someone-else", {"version": job["version"], "decision": "approve", "reviewer": "R"})
    assert info.value.status == 404
    with pytest.raises(JobError) as info:
        manager.review(job["id"], "owner", {"version": job["version"] - 1, "decision": "approve", "reviewer": "R"})
    assert info.value.status == 409
    with pytest.raises(JobError):
        manager.review(job["id"], "owner", {"version": job["version"], "decision": "sign", "reviewer": "R"})


def test_review_survives_reload_and_recomputation_resets_it(tmp_path):
    manager = _manager(tmp_path)
    job = _finished(manager)
    manager.review(job["id"], "owner", {"version": job["version"], "decision": "approve", "reviewer": "R-01"})
    manager.close()
    reloaded = _manager(tmp_path)
    assert reloaded.locked(reloaded.jobs[job["id"]]) is True
    record = reloaded.get(job["id"], "owner")
    reopened = reloaded.review(job["id"], "owner", {"version": record["version"], "decision": "reopen", "reviewer": "R-02", "note": "复算"})
    queued = reloaded.confirm(job["id"], "owner", {"version": reopened["version"], "override": True, "acknowledged": True, "stage": "B",
                                                    "mapping": {"3": "out-li", "4": "out-le", "5": "out-re", "6": "out-ri"}})
    assert queued["status"] == "queued"
    assert reloaded.run_next(timeout=0.5)
    done = reloaded.get(job["id"], "owner")
    assert done["status"] == "done" and done["review"]["status"] == "unreviewed"


def test_http_review_route_requires_csrf_and_reports_lock(tmp_path):
    manager = JobManager(tmp_path / "jobs", stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {},
                         mapping_validator=lambda *_a, **_k: [])
    sessions = SessionStore(tmp_path / "sessions.json", shared=False)
    server = ServiceHTTPServer(("127.0.0.1", 0), manager, sessions)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    api = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        api.request("GET", "/api/session")
        response = api.getresponse(); session = json.loads(response.read())
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        owner = sessions.lookup(cookie)[1]["owner"]
        headers = {"Cookie": cookie, "X-CSRF-Token": session["csrf_token"], "Content-Type": "application/json"}
        job = _finished(manager, owner=owner)
        api.request("POST", f"/api/jobs/{job['id']}/review", body=json.dumps({"version": job["version"], "decision": "approve", "reviewer": "R-01"}),
                    headers={"Cookie": cookie, "Content-Type": "application/json"})
        response = api.getresponse(); response.read()
        assert response.status == 403  # missing CSRF token
        api.request("POST", f"/api/jobs/{job['id']}/review", body=json.dumps({"version": job["version"], "decision": "approve", "reviewer": "R-01", "note": "签字"}), headers=headers)
        response = api.getresponse(); result = json.loads(response.read())
        assert response.status == 200 and result["job"]["review"]["status"] == "reviewed"
        version = result["job"]["version"]
        api.request("POST", f"/api/jobs/{job['id']}/delete", body=json.dumps({"version": version}), headers=headers)
        response = api.getresponse(); result = json.loads(response.read())
        assert response.status == 409 and "重新打开" in result["error"]["message"]
        api.request("GET", "/api/jobs?page=1&page_size=5", headers={"Cookie": cookie})
        response = api.getresponse(); listing = json.loads(response.read())
        assert listing["jobs"][0]["review"]["status"] == "reviewed"
    finally:
        api.close(); server.shutdown(); server.server_close(); manager.close()
