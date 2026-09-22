"""C15: a job bundle holds the white-listed files, sidecars, onepage.html and README; multi-select nests zips."""
from __future__ import annotations

import io
import json
import zipfile

from wss_deploy.bundle import bundle_name, job_bundle, multi_bundle
from wss_deploy.server import OUTPUT_FILES
from tests._c_helpers import Service, call, finished


def test_job_bundle_contents_and_compression(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "summary.json").write_text(json.dumps({"case_id": "A", "run_identity": "r" * 64}), encoding="utf-8")
    (job_dir / "report.html").write_text("<html>report</html>", encoding="utf-8")
    (job_dir / "field.npz").write_bytes(b"PK\x03\x04binary")
    (job_dir / "annotations.json").write_text("{}", encoding="utf-8")
    (job_dir / "narrative.json").write_text('{"edited": "\u5ba1\u9605\u4eba\u7ed3\u8bba"}', encoding="utf-8")
    (job_dir / "secret.txt").write_text("not exported", encoding="utf-8")
    (job_dir / "snapshots.json").write_text('{"items": []}', encoding="utf-8")
    (job_dir / "snapshot_front.png").write_bytes(b"\x89PNG\r\n\x1a\npixels")
    (job_dir / "snapshot_BAD.png").write_bytes(b"\x89PNG\r\n\x1a\npixels")  # not a name the service can produce
    data = job_bundle(job_dir, {"id": "job", "case_id": "A", "review": {"status": "reviewed", "by": "R"}}, OUTPUT_FILES, onepage_html="<html>one</html>")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        assert set(names) == {"summary.json", "report.html", "field.npz", "annotations.json", "snapshots.json",
                              "narrative.json", "snapshot_front.png", "onepage.html", "README.txt"}
        assert archive.getinfo("field.npz").compress_type == zipfile.ZIP_STORED and archive.getinfo("report.html").compress_type == zipfile.ZIP_DEFLATED
        assert archive.getinfo("snapshot_front.png").compress_type == zipfile.ZIP_STORED
        readme = archive.read("README.txt").decode("utf-8")
        assert "r" * 64 in readme and "reviewed（R）" in readme and "- onepage.html" in readme
        assert "- narrative.json" in readme and "narrative.json（若存在）" in readme and "morphology" in readme
        assert archive.testzip() is None
    assert bundle_name({"id": "20260921_x", "case_id": "LIU 病例/1"}) == "LIU_病例_1_20260921_x.zip"
    outer = multi_bundle([("a.zip", data), ("a.zip", data)])
    with zipfile.ZipFile(io.BytesIO(outer)) as archive:
        assert archive.namelist() == ["a.zip", "a_2.zip"] and archive.getinfo("a.zip").compress_type == zipfile.ZIP_STORED


def test_http_bundle_routes(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        a = finished(service.manager, owner=owner, case_id="A")
        b = finished(service.manager, owner=owner, case_id="B")
        queued = service.manager.create(owner, content=b"q", filename="q.stl")
        status, data, response = call(api, "GET", f"/api/jobs/{a['id']}/bundle.zip", {"Cookie": cookie})
        assert status == 200 and response.getheader("Content-Type") == "application/zip" and f"A_{a['id']}.zip" in response.getheader("Content-Disposition")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert {"summary.json", "report.html", "onepage.html", "README.txt", "field.npz", "run_manifest.json"} <= set(archive.namelist())
            assert "局限性声明" in archive.read("onepage.html").decode("utf-8")
        status, _, _ = call(api, "GET", f"/api/jobs/{queued['id']}/bundle.zip", {"Cookie": cookie})
        assert status == 409
        status, data, response = call(api, "POST", "/api/jobs/bundle", headers, {"ids": [a["id"], b["id"]]})
        assert status == 200 and response.getheader("Content-Type") == "application/zip"
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert sorted(archive.namelist()) == sorted([f"A_{a['id']}.zip", f"B_{b['id']}.zip"])
        # a signed-off job still downloads
        service.manager.review(a["id"], owner, {"version": service.manager.get(a["id"], owner)["version"], "decision": "approve", "reviewer": "R"})
        status, _, _ = call(api, "GET", f"/api/jobs/{a['id']}/bundle.zip", {"Cookie": cookie})
        assert status == 200
    finally:
        api.close(); service.close()
