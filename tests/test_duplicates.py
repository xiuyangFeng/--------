"""C2: the input SHA256 is recorded at upload; duplicates are detected and may reuse stage A."""
from __future__ import annotations

import hashlib
import json

import pytest

from wss_deploy.jobs import JobError
from tests._c_helpers import MAPPING, Service, call, finished, manager, multipart


def test_create_records_sha_and_find_by_input_orders_newest_first(tmp_path):
    mgr = manager(tmp_path)
    first = finished(mgr, case_id="a", content=b"solid one")
    second = mgr.create("owner", content=b"solid one", filename="b.stl", case_id="b")
    other = mgr.create("someone", content=b"solid one", filename="c.stl")
    assert first["input_sha256"] == hashlib.sha256(b"solid one").hexdigest() == second["input_sha256"]
    rows = mgr.find_by_input("owner", first["input_sha256"])
    assert [row["job_id"] for row in rows] == [second["id"], first["id"]]
    assert rows[1]["reusable"] is True and rows[1]["mapping_confirmed"] is True and rows[1]["family"] == "wall"
    assert rows[0]["reusable"] is False and rows[0]["status"] == "queued"
    assert other["id"] not in {row["job_id"] for row in rows}   # owner isolation


def test_ask_raises_409_with_existing_runs_and_reuse_clones_stage_a(tmp_path):
    mgr = manager(tmp_path)
    source = finished(mgr, case_id="a", content=b"solid one", patient_id="P-1", tags=["AAA"])
    with pytest.raises(JobError) as info:
        mgr.create("owner", content=b"solid one", filename="again.stl", on_duplicate="ask")
    assert info.value.status == 409 and info.value.payload["duplicate"] is True and info.value.payload["reusable"] == source["id"]
    assert info.value.payload["existing"][0]["job_id"] == source["id"]
    reused = mgr.create("owner", content=b"solid one", filename="again.stl", case_id="again", patient_id="P-2", tags=["followup"],
                        device="cpu", seed_count=1, on_duplicate="reuse")
    assert reused["reused_from"] == source["id"] and reused["source_job_id"] == source["id"]
    assert reused["status"] == "queued" and reused["stage"] == "B" and reused["mapping"] == MAPPING
    assert reused["case_id"] == "again" and reused["patient_id"] == "P-2" and reused["tags"] == ["followup"]
    assert reused["compute"] == {"device": "cpu", "seed_count": 1, "threads": None}
    assert reused["input_sha256"] == source["input_sha256"]
    assert (tmp_path / reused["id"] / "centerline" / "atlas.npz").is_file() and (tmp_path / reused["id"] / "stage_a.json").is_file()
    while mgr.run_next(timeout=0.2):   # drains the stale stage-A entry of the source, then runs stage B of the clone
        pass
    done = mgr.get(reused["id"], "owner")
    assert done["status"] == "done" and "centerline" not in done["summary"]["timing_s"]
    forced = mgr.create("owner", content=b"solid one", filename="again.stl", on_duplicate="force")
    assert forced["stage"] == "A" and "reused_from" not in forced
    assert mgr.list("owner")[0]["reused_from"] is None and mgr.get(reused["id"], "owner")["reused_from"] == source["id"]


def test_reuse_without_reusable_source_reports_reuse_unavailable(tmp_path):
    mgr = manager(tmp_path)
    mgr.create("owner", content=b"solid two", filename="x.stl")    # still queued: no mapping
    with pytest.raises(JobError) as info:
        mgr.create("owner", content=b"solid two", filename="y.stl", on_duplicate="reuse")
    assert info.value.status == 409 and info.value.payload["reuse_unavailable"] is True and info.value.payload["reusable"] is None


def test_batch_reports_duplicates_per_item(tmp_path):
    mgr = manager(tmp_path)
    source = finished(mgr, case_id="a", content=b"solid one")
    result = mgr.create_batch("owner", [{"content": b"solid one", "filename": "dup.stl"}, {"content": b"fresh", "filename": "new.stl"},
                                        {"content": b"solid one", "filename": "reuse.stl", "on_duplicate": "reuse"}], on_duplicate="ask")
    assert result["created_count"] == 2 and result["failed_count"] == 0 and result["duplicate_count"] == 1
    assert result["results"][0]["duplicate"] is True and result["results"][0]["reusable"] == source["id"] and "job" not in result["results"][0]
    assert result["results"][1]["job"]["stage"] == "A"
    assert result["results"][2]["reused_from"] == source["id"] and result["results"][2]["job"]["stage"] == "B"


def test_http_upload_on_duplicate_field(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        source = finished(service.manager, owner=owner, case_id="a", content=b"solid one")
        body, ctype = multipart("dup", [("stl", b"solid one", "again.stl")])
        upload = {"Cookie": cookie, "X-CSRF-Token": headers["X-CSRF-Token"], "Content-Type": ctype}
        status, payload, _ = call(api, "POST", "/api/jobs", upload, body)
        assert status == 409 and payload["duplicate"] is True and payload["reusable"] == source["id"] and payload["error"]["message"]
        body, ctype = multipart("dup", [("stl", b"solid one", "again.stl"), ("on_duplicate", "reuse", None), ("case_id", "AGAIN", None)])
        status, payload, _ = call(api, "POST", "/api/jobs", {**upload, "Content-Type": ctype}, body)
        assert status == 201 and payload["reused_from"] == source["id"] and payload["job"]["case_id"] == "AGAIN" and payload["job"]["stage"] == "B"
        body, ctype = multipart("dup", [("stl", b"solid one", "one.stl"), ("stl", b"solid one", "two.stl"), ("on_duplicate", "ask", None)])
        status, payload, _ = call(api, "POST", "/api/jobs/batch", {**upload, "Content-Type": ctype}, body)
        assert status == 201 and payload["duplicate_count"] == 2 and payload["created_count"] == 0
        assert all(item["duplicate"] for item in payload["results"])
    finally:
        api.close(); service.close()
