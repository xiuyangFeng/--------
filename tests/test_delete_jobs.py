"""Deleting jobs removes the record, moves the directory to the trash and closes the live stream."""
from __future__ import annotations

import http.client
import json
import queue
import threading

import pytest

from wss_deploy.jobs import DELETED_LOG, DELETING_PREFIX, TRASH_DIR, JobError, JobManager
from wss_deploy.server import ServiceHTTPServer, SessionStore


def _manager(tmp_path):
    return JobManager(tmp_path, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})


def _finished(manager, owner="owner", case_id="case"):
    job = manager.create(owner, content=b"solid", filename=f"{case_id}.stl", case_id=case_id)
    with manager.lock:
        record = manager.jobs[job["id"]]
        record.update(status="done", stage="B", phase="计算完成")
        manager._event(record, "finished")
        (manager.root / job["id"] / "report.html").write_text("<html></html>", encoding="utf-8")
        (manager.root / job["id"] / "history").mkdir()
        (manager.root / job["id"] / "history" / "old.json").write_text("{}", encoding="utf-8")
    return manager.get(job["id"], owner)


def test_delete_moves_directory_to_trash_and_logs_provenance_on_purge(tmp_path):
    manager = _manager(tmp_path)
    job = _finished(manager)
    listener, snapshot = manager.subscribe(job["id"], "owner")
    assert snapshot["status"] == "done"
    result = manager.delete(job["id"], "owner", {"version": job["version"]})
    assert result["deleted"] is True and result["trashed"] is True and result["removed_bytes"] > 0 and result["expires_at"]
    assert not (tmp_path / job["id"]).exists() and (tmp_path / TRASH_DIR / job["id"] / "trash.json").is_file()
    assert not list(tmp_path.glob(DELETING_PREFIX + "*"))
    with pytest.raises(JobError):
        manager.get(job["id"], "owner")
    farewell = listener.get_nowait()
    assert farewell["status"] == "deleted" and farewell["final"] is True
    assert job["id"] not in manager._subscribers
    assert not (tmp_path / DELETED_LOG).exists()   # provenance is written when the trash entry is purged
    manager.purge(job["id"], "owner")
    lines = (tmp_path / DELETED_LOG).read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["id"] == job["id"] and record["status_before"] == "done" and record["deleted_at"] and record["purge_reason"] == "manual"
    assert not (tmp_path / TRASH_DIR / job["id"]).exists()


def test_delete_checks_owner_version_and_running_state(tmp_path):
    manager = _manager(tmp_path)
    job = _finished(manager)
    with pytest.raises(JobError) as info:
        manager.delete(job["id"], "someone-else", {"version": job["version"]})
    assert info.value.status == 404
    with pytest.raises(JobError) as info:
        manager.delete(job["id"], "owner", {"version": job["version"] - 1})
    assert info.value.status == 409
    with manager.lock:
        manager.jobs[job["id"]]["status"] = "running"
    with pytest.raises(JobError) as info:
        manager.delete(job["id"], "owner", {"version": job["version"]})
    assert info.value.status == 409 and (tmp_path / job["id"]).is_dir()


def test_delete_many_reports_each_item_and_keeps_others(tmp_path):
    manager = _manager(tmp_path)
    first, second = _finished(manager, case_id="a"), _finished(manager, case_id="b")
    other = _finished(manager, owner="bob", case_id="c")
    with manager.lock:
        manager.jobs[second["id"]]["status"] = "running"
    result = manager.delete_many("owner", [
        {"id": first["id"], "version": first["version"]},
        {"id": second["id"], "version": second["version"]},
        {"id": other["id"], "version": other["version"]},
        {"id": "not/valid", "version": 1},
    ])
    assert result["deleted_count"] == 1 and result["failed_count"] == 3
    assert result["results"][0]["deleted"] is True
    assert result["results"][1]["error"]["status"] == 409
    assert result["results"][2]["error"]["status"] == 404
    assert result["results"][3]["error"]["status"] == 400
    assert not (tmp_path / first["id"]).exists() and (tmp_path / second["id"]).is_dir() and (tmp_path / other["id"]).is_dir()
    assert manager.query("owner")["total"] == 1


def test_queued_job_deleted_before_worker_runs_is_skipped(tmp_path):
    manager = _manager(tmp_path)
    job = manager.create("owner", content=b"solid", filename="x.stl")
    assert job["status"] == "queued"
    manager.delete(job["id"], "owner", {"version": job["version"]})
    assert manager.run_next(timeout=0.2) is True  # queue entry consumed without error
    assert manager.tasks.empty()


def test_unfinished_deletion_is_purged_at_startup(tmp_path):
    leftover = tmp_path / f"{DELETING_PREFIX}old_1234"
    leftover.mkdir()
    (leftover / "job.json").write_text(json.dumps({"id": "old", "status": "done"}), encoding="utf-8")
    manager = _manager(tmp_path)
    assert not leftover.exists() and manager.jobs == {}


def test_http_delete_routes(tmp_path):
    manager = JobManager(tmp_path / "jobs", stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})
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
        one, two = _finished(manager, owner=owner, case_id="one"), _finished(manager, owner=owner, case_id="two")
        api.request("POST", f"/api/jobs/{one['id']}/delete", body=json.dumps({"version": one["version"]}), headers=headers)
        response = api.getresponse(); result = json.loads(response.read())
        assert response.status == 200 and result["deleted"] is True and not (manager.root / one["id"]).exists()
        api.request("POST", "/api/jobs/delete", body=json.dumps({"jobs": [{"id": two["id"], "version": two["version"]}]}), headers=headers)
        response = api.getresponse(); result = json.loads(response.read())
        assert response.status == 200 and result["deleted_count"] == 1 and not (manager.root / two["id"]).exists()
        api.request("GET", f"/api/jobs/{two['id']}", headers={"Cookie": cookie})
        response = api.getresponse(); response.read()
        assert response.status == 404
    finally:
        api.close(); server.shutdown(); server.server_close(); manager.close()
