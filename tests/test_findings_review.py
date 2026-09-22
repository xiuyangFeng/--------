"""C16: reviewer decisions are merged into summary.findings.review and drive the one-page layout."""
from __future__ import annotations

import json

import pytest

from wss_deploy.jobs import JobError
from wss_deploy.onepager import render_onepage
from tests._c_helpers import Service, call, finished, manager


def test_findings_review_roundtrip_and_onepage(tmp_path):
    mgr = manager(tmp_path)
    job = finished(mgr, case_id="A")
    payload = {"items": {"F1": {"decision": "confirmed", "note": "与影像一致"}, "F2": {"decision": "rejected", "note": "切口效应"}, "F3": {"decision": None, "note": ""}},
               "added": [{"xyz_mm": [1, 2, 3], "branch": "左髂外", "text": "人工发现：局部狭窄", "kind": "manual", "severity": "attention"}]}
    result = mgr.findings_review(job["id"], "owner", payload)
    doc = result["findings_review"]
    assert doc["schema_version"] == "wss-deploy.findings_review/v1" and doc["items"]["F1"]["decision"] == "confirmed"
    assert doc["added"][0]["id"] == "M1" and doc["added"][0]["kind"] == "manual" and doc["added"][0]["severity"] == "attention"
    job_dir = tmp_path / job["id"]
    summary = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["findings"]["review"] == doc and len(summary["findings"]["items"]) == 3
    html = (job_dir / "report.html").read_text(encoding="utf-8")
    embedded = json.loads(html.split('type="application/json">', 1)[1].split("</script>", 1)[0])
    assert embedded["findings"]["review"]["items"]["F2"]["decision"] == "rejected"
    assert mgr.get(job["id"], "owner")["findings_review"]["items"]["F1"]["note"] == "与影像一致"
    assert mgr.get(job["id"], "owner")["events"][-1]["action"] == "findings_review_updated"
    page = render_onepage(summary, {"id": job["id"]})
    body, _, appendix = page.partition("附录：已驳回的自动发现")
    assert "☑ 已确认" in body and "☐ 未判定" in body and "【人工】人工发现：局部狭窄" in body and "与影像一致" in body
    assert "主动脉低 WSS 区" not in body and "主动脉低 WSS 区" in appendix and "切口效应" in appendix
    assert "术语说明" in page and "空间 p99" in page
    for bad in ({"items": {"F1": {"decision": "maybe"}}}, {"items": []}, {"items": {}, "added": [{"xyz_mm": [1, 2, 3], "text": ""}]},
                {"items": {}, "added": [{"xyz_mm": "x", "text": "t"}]}):
        with pytest.raises(JobError):
            mgr.findings_review(job["id"], "owner", bad)
    version = mgr.get(job["id"], "owner")["version"]
    mgr.review(job["id"], "owner", {"version": version, "decision": "approve", "reviewer": "R"})
    with pytest.raises(JobError) as info:
        mgr.findings_review(job["id"], "owner", {"items": {}})
    assert info.value.status == 409
    version = mgr.get(job["id"], "owner")["version"]
    mgr.review(job["id"], "owner", {"version": version, "decision": "reopen", "reviewer": "R", "note": "改判"})
    assert mgr.findings_review(job["id"], "owner", {"items": {"F1": {"decision": "rejected"}}})["findings_review"]["items"]["F1"]["decision"] == "rejected"


def test_http_findings_review_route_and_annotations_in_onepage(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        job = finished(service.manager, owner=owner, case_id="A")
        status, payload, _ = call(api, "PUT", f"/api/jobs/{job['id']}/findings_review", headers, {"items": {"F1": {"decision": "confirmed"}}, "added": []})
        assert status == 200 and payload["findings_review"]["items"]["F1"]["decision"] == "confirmed" and payload["version"] == job["version"] + 1
        status, payload, _ = call(api, "GET", f"/api/jobs/{job['id']}/files/findings_review.json", {"Cookie": cookie})
        assert status == 200 and payload["items"]["F1"]["decision"] == "confirmed"
        call(api, "PUT", f"/api/jobs/{job['id']}/annotations", headers, {"items": [{"xyz_mm": [0, 0, 0], "text": "钉在瘤囊", "branch": "主动脉", "s_from_root_mm": 88}]})
        status, page, _ = call(api, "GET", f"/api/jobs/{job['id']}/onepage", {"Cookie": cookie})
        page = page.decode("utf-8")
        assert status == 200 and "<h2>标注</h2>" in page and "钉在瘤囊" in page and "88" in page
    finally:
        api.close(); service.close()
