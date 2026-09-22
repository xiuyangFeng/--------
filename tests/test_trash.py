"""C4: deletion goes to the trash; restore keeps the id and state; expiry is clock-driven."""
from __future__ import annotations

import json

import pytest

from wss_deploy.jobs import DELETED_LOG, TRASH_DAYS, TRASH_DIR, JobError, JobManager
from tests._c_helpers import Service, call, finished, manager


class Clock:
    def __init__(self, now=1_800_000_000.0): self.now = now
    def __call__(self): return self.now


def test_delete_restore_keeps_state_and_report_opens(tmp_path):
    clock = Clock()
    mgr = manager(tmp_path, clock=clock)
    job = finished(mgr, case_id="A")
    result = mgr.delete(job["id"], "owner", {"version": job["version"]})
    assert result["trashed"] is True
    listing = mgr.trash_list("owner")["items"]
    assert [item["id"] for item in listing] == [job["id"]] and listing[0]["days_left"] == TRASH_DAYS and listing[0]["status_before"] == "done"
    assert listing[0]["family"] == "wall" and listing[0]["release_id"] == "REL_A"
    assert mgr.trash_list("someone-else")["items"] == [] and mgr.trash_list("someone-else", all_owners=True)["items"]
    with pytest.raises(JobError) as info:
        mgr.restore(job["id"], "someone-else")
    assert info.value.status == 404
    restored = mgr.restore(job["id"], "owner")
    assert restored["id"] == job["id"] and restored["status"] == "done" and restored["events"][-1]["action"] == "restored"
    assert (tmp_path / job["id"] / "report.html").is_file() and not (tmp_path / TRASH_DIR / job["id"]).exists()
    assert not (tmp_path / job["id"] / "trash.json").exists()
    assert mgr.get(job["id"], "owner")["summary"]["peak"]["p99_pa"] == 1.0
    # the record can be deleted again and the newer trash copy supersedes nothing stale
    again = mgr.delete(job["id"], "owner", {"version": restored["version"]})
    assert again["trashed"] is True and (tmp_path / TRASH_DIR / job["id"] / "job.json").is_file()


def test_expired_entries_are_purged_with_injected_clock(tmp_path):
    clock = Clock()
    mgr = manager(tmp_path, clock=clock)
    old, fresh = finished(mgr, case_id="old"), finished(mgr, case_id="fresh")
    mgr.delete(old["id"], "owner", {"version": old["version"]})
    clock.now += (TRASH_DAYS - 1) * 86400
    mgr.delete(fresh["id"], "owner", {"version": fresh["version"]})
    assert mgr.purge_expired() == []
    clock.now += 2 * 86400
    assert mgr.purge_expired() == [old["id"]]
    assert not (tmp_path / TRASH_DIR / old["id"]).exists() and (tmp_path / TRASH_DIR / fresh["id"]).is_dir()
    record = json.loads((tmp_path / DELETED_LOG).read_text(encoding="utf-8").splitlines()[-1])
    assert record["id"] == old["id"] and record["purge_reason"] == "expired" and record["deleted_by"] == "owner"
    # start-up scan on a fresh manager
    clock.now += TRASH_DAYS * 86400
    reloaded = JobManager(tmp_path, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {}, clock=clock)
    assert not (tmp_path / TRASH_DIR / fresh["id"]).exists()
    reloaded.close()


def test_locked_job_cannot_be_trashed_and_queued_job_restores_as_interrupted(tmp_path):
    mgr = manager(tmp_path)
    job = finished(mgr, case_id="A")
    approved = mgr.review(job["id"], "owner", {"version": job["version"], "decision": "approve", "reviewer": "R"})
    with pytest.raises(JobError) as info:
        mgr.delete(job["id"], "owner", {"version": approved["version"]})
    assert info.value.status == 409
    queued = mgr.create("owner", content=b"q", filename="q.stl")
    mgr.delete(queued["id"], "owner", {"version": queued["version"]})
    restored = mgr.restore(queued["id"], "owner")
    assert restored["status"] == "interrupted"


def test_http_trash_routes(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        job = finished(service.manager, owner=owner, case_id="A")
        status, payload, _ = call(api, "POST", f"/api/jobs/{job['id']}/delete", headers, {"version": job["version"]})
        assert status == 200 and payload["trashed"] is True and payload["expires_at"]
        status, payload, _ = call(api, "GET", "/api/trash", {"Cookie": cookie})
        assert status == 200 and payload["items"][0]["id"] == job["id"]
        status, payload, _ = call(api, "POST", f"/api/trash/{job['id']}/restore", headers, {})
        assert status == 200 and payload["job"]["status"] == "done"
        status, payload, _ = call(api, "GET", f"/api/jobs/{job['id']}/report", {"Cookie": cookie})
        assert status == 200
        version = service.manager.get(job["id"], owner)["version"]
        call(api, "POST", f"/api/jobs/{job['id']}/delete", headers, {"version": version})
        status, payload, _ = call(api, "POST", f"/api/trash/{job['id']}/purge", headers, {})
        assert status == 200 and payload["purged"] is True
        status, payload, _ = call(api, "GET", "/api/trash", {"Cookie": cookie})
        assert payload["items"] == []
    finally:
        api.close(); service.close()


def test_daily_scan_from_worker_loop_purges_expired_entries(tmp_path):
    clock = Clock()
    mgr = manager(tmp_path, clock=clock)
    job = finished(mgr, case_id="A")
    mgr.delete(job["id"], "owner", {"version": job["version"]})
    assert mgr.maybe_purge_expired() == []                      # the start-up scan just ran; nothing is due
    clock.now += TRASH_DAYS * 86400 + 1
    assert mgr.maybe_purge_expired() == [job["id"]]             # a day has passed and the entry is expired
    assert mgr.trash_list("owner")["items"] == [] and mgr.maybe_purge_expired() == []
    lines = [json.loads(line) for line in (tmp_path / DELETED_LOG).read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["id"] == job["id"] and lines[-1]["purge_reason"] == "expired"
