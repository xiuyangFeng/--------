"""C8: annotations are stored as a sidecar, mirrored into summary.json + the embedded report meta, and refused when locked."""
from __future__ import annotations

import json

import pytest

from wss_deploy.jobs import JobError
from tests._c_helpers import Service, call, finished, manager


def _embedded(job_dir):
    html = (job_dir / "report.html").read_text(encoding="utf-8")
    return json.loads(html.split('type="application/json">', 1)[1].split("</script>", 1)[0])


def test_annotations_roundtrip_validation_and_lock(tmp_path):
    mgr = manager(tmp_path)
    job = finished(mgr, case_id="A")
    items = [{"id": "A1", "xyz_mm": [1, 2, 3], "text": "瘤颈", "branch": "主动脉", "segment_id": 0, "s_from_root_mm": 42.5},
             {"xyz_mm": [4.0, 5.0, 6.0], "text": "第二处", "color": "#ff0000"}]
    result = mgr.annotations(job["id"], "owner", {"items": items, "version": job["version"]})
    doc = result["annotations"]
    assert doc["schema_version"] == "wss-deploy.annotations/v1" and [i["id"] for i in doc["items"]] == ["A1", "A2"]
    assert doc["items"][0]["s_from_root_mm"] == 42.5 and doc["items"][1]["color"] == "#ff0000" and doc["items"][0]["color"] == "#d97706"
    assert result["version"] == job["version"] + 1
    job_dir = tmp_path / job["id"]
    assert json.loads((job_dir / "annotations.json").read_text(encoding="utf-8")) == doc
    assert json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))["annotations"] == doc
    assert _embedded(job_dir)["annotations"] == doc
    snapshot = mgr.get(job["id"], "owner")
    assert snapshot["annotations"] == doc and snapshot["events"][-1]["action"] == "annotations_updated"
    assert mgr.list("owner")[0]["annotations_count"] == 2
    for bad in ({"items": [{"xyz_mm": [1, 2], "text": "x"}]}, {"items": [{"xyz_mm": [1, 2, 3], "text": ""}]},
                {"items": [{"xyz_mm": [1, 2, 3], "text": "x", "color": "red"}]}, {"items": "no"},
                {"items": [{"id": "A1", "xyz_mm": [1, 2, 3], "text": "x"}, {"id": "A1", "xyz_mm": [1, 2, 3], "text": "y"}]}):
        with pytest.raises(JobError):
            mgr.annotations(job["id"], "owner", bad)
    with pytest.raises(JobError) as info:
        mgr.annotations(job["id"], "owner", {"items": [], "version": 1})
    assert info.value.status == 409
    version = mgr.get(job["id"], "owner")["version"]
    mgr.review(job["id"], "owner", {"version": version, "decision": "approve", "reviewer": "R"})
    with pytest.raises(JobError) as info:
        mgr.annotations(job["id"], "owner", {"items": []})
    assert info.value.status == 409
    # survives a reload of the job store
    mgr.close()
    reloaded = manager(tmp_path)
    assert reloaded.get(job["id"], "owner")["annotations"]["items"][0]["text"] == "瘤颈"
    reloaded.close()


def test_http_annotations_put_and_file(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        job = finished(service.manager, owner=owner, case_id="A")
        status, payload, _ = call(api, "GET", f"/api/jobs/{job['id']}/files/annotations.json", {"Cookie": cookie})
        assert status == 404
        status, payload, _ = call(api, "PUT", f"/api/jobs/{job['id']}/annotations", headers, {"items": [{"xyz_mm": [0, 0, 0], "text": "t"}]})
        assert status == 200 and payload["annotations"]["items"][0]["id"] == "A1"
        status, payload, _ = call(api, "GET", f"/api/jobs/{job['id']}/files/annotations.json", {"Cookie": cookie})
        assert status == 200 and payload["items"][0]["text"] == "t"
        status, _, _ = call(api, "PUT", f"/api/jobs/{job['id']}/annotations", {"Cookie": cookie, "Content-Type": "application/json"}, {"items": []})
        assert status == 403
    finally:
        api.close(); service.close()


def test_rebuild_embeds_sidecars(tmp_path):
    from wss_deploy.rebuild_report import embed_sidecars
    (tmp_path / "annotations.json").write_text(json.dumps({"schema_version": "wss-deploy.annotations/v1", "items": [{"id": "A1"}]}), encoding="utf-8")
    (tmp_path / "findings_review.json").write_text(json.dumps({"items": {"F1": {"decision": "rejected", "note": ""}}, "added": []}), encoding="utf-8")
    meta = embed_sidecars(tmp_path, {"findings": {"items": [{"id": "F1"}]}})
    assert meta["annotations"]["items"][0]["id"] == "A1" and meta["findings"]["review"]["items"]["F1"]["decision"] == "rejected"
    assert meta["findings"]["items"] == [{"id": "F1"}]
    assert embed_sidecars(tmp_path / "missing", {"x": 1}) == {"x": 1}
