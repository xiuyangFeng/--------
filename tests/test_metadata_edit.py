"""§15.2: case identifiers can be corrected after the fact, on any status, unless the result is signed off."""
from __future__ import annotations

import json

import pytest

from wss_deploy.jobs import JobError
from tests._c_helpers import Service, call, finished, manager


def test_partial_edit_merges_onto_current_values_and_revalidates(tmp_path):
    mgr = manager(tmp_path)
    job = finished(mgr, case_id="A", patient_id="P-001", scan_label="基线", scan_date="2026-09-01",
                   tags=["AAA", "随访"], notes="原始备注")
    updated = mgr.update_metadata(job["id"], "owner", {"scan_label": "复查", "tags": ["AAA", "aaa", "术后"]})
    assert updated["case_id"] == "A" and updated["patient_id"] == "P-001" and updated["scan_date"] == "2026-09-01"
    assert updated["scan_label"] == "复查" and updated["notes"] == "原始备注"
    assert updated["tags"] == ["AAA", "术后"]  # case-insensitive de-duplication of _metadata
    record = mgr.get(job["id"], "owner")
    assert record["events"][-1]["action"] == "metadata_updated"
    assert record["events"][-1]["changed"] == ["scan_label", "tags"]

    summary = json.loads((mgr.root / job["id"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_metadata"] == {"patient_id": "P-001", "scan_label": "复查", "scan_date": "2026-09-01",
                                        "tags": ["AAA", "术后"], "notes": "原始备注"}
    assert summary["case_id"] == "A"
    report = (mgr.root / job["id"] / "report.html").read_text(encoding="utf-8")
    assert "复查" in report and json.loads(report.split('type="application/json">')[1].split("</script>")[0])["case_metadata"]["scan_label"] == "复查"

    renamed = mgr.update_metadata(job["id"], "owner", {"case_id": "A-2", "notes": ""})
    assert renamed["case_id"] == "A-2" and renamed["notes"] == ""
    summary = json.loads((mgr.root / job["id"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["case_id"] == "A-2" and summary["case_metadata"]["notes"] == ""
    assert mgr.jobs[job["id"]]["summary"]["case_metadata"]["notes"] == ""
    mgr.close()


def test_validation_lock_and_ownership(tmp_path):
    mgr = manager(tmp_path)
    job = finished(mgr, case_id="A")

    def edit(payload, owner="owner"):
        with pytest.raises(JobError) as error:
            mgr.update_metadata(job["id"], owner, payload)
        return error.value

    assert "日期" in str(edit({"scan_date": "2026-13-01"}))
    assert edit({"tags": ["x"] * 13}).status == 400
    assert edit({"case_id": "   "}).status == 400
    assert edit({"patient_id": "p" * 81}).status == 400
    assert edit({"notes": "bad\x00text"}).status == 400
    assert edit({"case_id": "B"}, owner="someone-else").status == 404
    assert edit({"case_id": "B", "version": 999}).status == 409
    assert mgr.get(job["id"], "owner")["case_id"] == "A"  # nothing was written

    mgr.review(job["id"], "owner", {"version": mgr.get(job["id"], "owner")["version"],
                                    "decision": "approve", "reviewer": "R-1"})
    assert edit({"case_id": "B"}).status == 409
    mgr.close()


def test_metadata_of_an_unfinished_job_is_editable(tmp_path):
    mgr = manager(tmp_path)
    job = mgr.create("owner", content=b"solid", filename="q.stl", case_id="Q")
    updated = mgr.update_metadata(job["id"], "owner", {"patient_id": "P-9", "case_id": "Q-1"})
    assert updated["status"] == "queued" and updated["patient_id"] == "P-9" and updated["case_id"] == "Q-1"
    assert json.loads((mgr.root / job["id"] / "job.json").read_text(encoding="utf-8"))["case_id"] == "Q-1"
    mgr.close()


def test_http_metadata_route(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        job = finished(service.manager, owner=owner, case_id="A")
        status, payload, _ = call(api, "POST", f"/api/jobs/{job['id']}/metadata", headers,
                                  {"version": job["version"], "case_id": "A-新", "tags": ["AAA"], "notes": "改过"})
        assert status == 200 and payload["job"]["case_id"] == "A-新" and payload["job"]["tags"] == ["AAA"]
        assert payload["job"]["version"] > job["version"]
        status, payload, _ = call(api, "GET", f"/api/jobs/{job['id']}", {"Cookie": cookie})
        assert payload["case_id"] == "A-新" and payload["notes"] == "改过"
        status, payload, _ = call(api, "POST", f"/api/jobs/{job['id']}/metadata", headers, {"scan_date": "x"})
        assert status == 400 and "error" in payload
        status, _, _ = call(api, "POST", f"/api/jobs/{job['id']}/metadata",
                            {"Cookie": cookie, "Content-Type": "application/json"}, {"case_id": "B"})
        assert status == 403  # CSRF
    finally:
        api.close(); service.close()
