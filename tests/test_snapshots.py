"""§15.1: the 3-D report posts its rendered views; they are stored, served, bundled and printed.

Pictures illustrate an existing result and change no number, so a signed-off job stays writable.
"""
from __future__ import annotations

import base64
import io
import json
import zipfile

import pytest

from wss_deploy.jobs import JobError, MAX_SNAPSHOTS
from wss_deploy.onepager import NO_SNAPSHOTS_HINT, render_onepage
from wss_deploy.server import OUTPUT_FILES, output_file
from tests._c_helpers import Service, call, finished, manager

PNG_B64 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAC"
           "hwGA60e6kgAAAABJRU5ErkJggg==")
PNG = base64.b64decode(PNG_B64)


def image(name="front", caption="前视 · WSS · Pa", **extra):
    return {"name": name, "png_base64": PNG_B64, "width": 1600, "height": 1200, "caption": caption, "view": name, **extra}


def test_snapshots_are_stored_atomically_and_merge_or_replace(tmp_path):
    mgr = manager(tmp_path)
    job = finished(mgr, case_id="A")
    result = mgr.snapshots(job["id"], "owner", {"images": [image("front"), image("left", "左视 · WSS · Pa")]})
    document = result["snapshots"]
    assert document["schema_version"] == "wss-deploy.snapshots/v1" and document["updated_by"] == "owner"
    assert [item["name"] for item in document["items"]] == ["front", "left"]
    assert document["items"][0] == {"name": "front", "file": "snapshot_front.png", "caption": "前视 · WSS · Pa",
                                    "view": "front", "bytes": len(PNG), "created_at": document["items"][0]["created_at"],
                                    "width": 1600, "height": 1200}
    job_dir = mgr.root / job["id"]
    assert (job_dir / "snapshot_front.png").read_bytes() == PNG
    assert json.loads((job_dir / "snapshots.json").read_text(encoding="utf-8")) == document
    assert not list(job_dir.glob("*.tmp"))
    record = mgr.get(job["id"], "owner")
    assert record["snapshots"] == document and result["version"] == record["version"]
    assert record["events"][-1]["action"] == "snapshots_updated" and record["events"][-1]["count"] == 2
    # summary.json is untouched: pictures are not results.
    assert "snapshots" not in json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))

    merged = mgr.snapshots(job["id"], "owner", {"images": [image("top", "上视")], "replace": False})["snapshots"]
    assert [item["name"] for item in merged["items"]] == ["front", "left", "top"]
    replaced = mgr.snapshots(job["id"], "owner", {"images": [image("current", "当前视图")]})["snapshots"]
    assert [item["name"] for item in replaced["items"]] == ["current"]
    assert sorted(path.name for path in job_dir.glob("snapshot_*.png")) == ["snapshot_current.png"]
    mgr.close()


def test_snapshot_validation(tmp_path):
    mgr = manager(tmp_path)
    job = finished(mgr, case_id="A")

    def post(payload):
        with pytest.raises(JobError) as error:
            mgr.snapshots(job["id"], "owner", payload)
        return error.value

    assert post({"images": []}).status == 400
    assert post({"images": [image() for _ in range(MAX_SNAPSHOTS + 1)]}).status == 400
    assert "名称" in str(post({"images": [image("Front")]}))
    assert "名称" in str(post({"images": [image("a" * 33)]}))
    assert "名称" in str(post({"images": [image("../evil")]}))
    assert "重复" in str(post({"images": [image("front"), image("front")]}))
    assert "PNG" in str(post({"images": [dict(image(), png_base64=base64.b64encode(b"GIF89a junk").decode())]}))
    assert "base64" in str(post({"images": [dict(image(), png_base64="not base64!!")]}))
    assert post({"images": [dict(image(), png_base64=base64.b64encode(
        PNG + b"\0" * (6 * 1024 * 1024)).decode())]}).status == 413
    assert "布尔" in str(post({"images": [image()], "replace": "yes"}))
    assert "整数" in str(post({"images": [dict(image(), width=0)]}))
    # version, when given, must match
    assert post({"images": [image()], "version": 1}).status == 409
    assert not list((mgr.root / job["id"]).glob("snapshot_*.png"))
    mgr.close()


def test_only_finished_jobs_but_a_signed_off_one_still_accepts_pictures(tmp_path):
    mgr = manager(tmp_path)
    queued = mgr.create("owner", content=b"solid", filename="q.stl", case_id="Q")
    with pytest.raises(JobError) as error:
        mgr.snapshots(queued["id"], "owner", {"images": [image()]})
    assert error.value.status == 409
    job = finished(mgr, case_id="A")
    mgr.review(job["id"], "owner", {"version": mgr.get(job["id"], "owner")["version"],
                                    "decision": "approve", "reviewer": "R-1"})
    assert mgr.locked(mgr.jobs[job["id"]])
    assert mgr.snapshots(job["id"], "owner", {"images": [image()]})["snapshots"]["items"]
    with pytest.raises(JobError):
        mgr.snapshots(job["id"], "other-owner", {"images": [image()]})
    mgr.close()


def test_output_file_whitelist_only_accepts_the_snapshot_pattern(tmp_path):
    (tmp_path / "snapshot_front.png").write_bytes(PNG)
    (tmp_path / "snapshot_BAD.png").write_bytes(PNG)
    (tmp_path / "other.png").write_bytes(PNG)
    (tmp_path / "input.stl").write_bytes(b"solid")
    assert output_file(tmp_path, "snapshot_front.png") is not None
    assert output_file(tmp_path, "snapshot_BAD.png") is None
    assert output_file(tmp_path, "other.png") is None
    assert output_file(tmp_path, "input.stl") is None
    assert "snapshots.json" in OUTPUT_FILES


def test_onepage_embeds_the_pictures_or_shows_a_hint(tmp_path):
    mgr = manager(tmp_path)
    job = finished(mgr, case_id="A")
    summary = json.loads((mgr.root / job["id"] / "summary.json").read_text(encoding="utf-8"))
    html = render_onepage(summary, job, job_dir=mgr.root / job["id"])
    assert "<h2>配图</h2>" in html and NO_SNAPSHOTS_HINT in html
    mgr.snapshots(job["id"], "owner", {"images": [image("front"), image("left", "左视 · WSS · Pa"),
                                                  image("top", "上视 · WSS · Pa")]})
    html = render_onepage(summary, job, job_dir=mgr.root / job["id"])
    assert html.count("data:image/png;base64," + PNG_B64) == 3
    assert "左视 · WSS · Pa" in html and "shots cols3" in html and NO_SNAPSHOTS_HINT not in html
    assert "page-break-inside:avoid" in html and "max-width:100%" in html
    mgr.close()


def test_http_snapshot_routes_serving_and_bundling(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        job = finished(service.manager, owner=owner, case_id="A")
        status, payload, _ = call(api, "POST", f"/api/jobs/{job['id']}/snapshots", headers,
                                  {"images": [image("front"), image("slice", "截面 · 速度 · m/s")]})
        assert status == 200 and [item["name"] for item in payload["snapshots"]["items"]] == ["front", "slice"]
        assert payload["version"] > job["version"]

        status, data, response = call(api, "GET", f"/api/jobs/{job['id']}/files/snapshot_front.png", {"Cookie": cookie})
        assert status == 200 and response.getheader("Content-Type") == "image/png" and data == PNG
        assert response.getheader("Content-Disposition") is None  # shown inline, not downloaded
        status, payload, _ = call(api, "GET", f"/api/jobs/{job['id']}/files/snapshots.json", {"Cookie": cookie})
        assert status == 200 and payload["schema_version"] == "wss-deploy.snapshots/v1"
        status, _, _ = call(api, "GET", f"/api/jobs/{job['id']}/files/snapshot_missing.png", {"Cookie": cookie})
        assert status == 404
        status, _, _ = call(api, "GET", f"/api/jobs/{job['id']}/files/input.stl", {"Cookie": cookie})
        assert status == 404

        status, body, _ = call(api, "GET", f"/api/jobs/{job['id']}/onepage", {"Cookie": cookie})
        assert status == 200 and b"data:image/png;base64," in body
        status, data, _ = call(api, "GET", f"/api/jobs/{job['id']}/bundle.zip", {"Cookie": cookie})
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert {"snapshots.json", "snapshot_front.png", "snapshot_slice.png"} <= set(archive.namelist())
            assert archive.getinfo("snapshot_front.png").compress_type == zipfile.ZIP_STORED
            assert b"data:image/png;base64," in archive.read("onepage.html")

        # CSRF is required like every other mutation
        status, _, _ = call(api, "POST", f"/api/jobs/{job['id']}/snapshots",
                            {"Cookie": cookie, "Content-Type": "application/json"}, {"images": [image()]})
        assert status == 403
        status, _, _ = call(api, "POST", f"/api/jobs/{job['id']}/snapshots", headers,
                            {"images": [dict(image(), png_base64=base64.b64encode(b"nope").decode())]})
        assert status == 400
    finally:
        api.close(); service.close()


def test_http_snapshot_body_limit_is_larger_than_other_json_routes(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        job = finished(service.manager, owner=owner, case_id="A")
        # ~1 MB body: far beyond the 64 KB limit that applies to annotations and friends.
        big = base64.b64encode(PNG + b"\0" * (1024 * 1024)).decode()
        status, payload, _ = call(api, "POST", f"/api/jobs/{job['id']}/snapshots", headers,
                                  {"images": [dict(image(), png_base64=big)]})
        assert status == 200 and payload["snapshots"]["items"][0]["bytes"] > 1024 * 1024
        status, _, _ = call(api, "PUT", f"/api/jobs/{job['id']}/annotations", headers,
                            {"items": [{"id": "A1", "xyz_mm": [0, 0, 0], "text": "x" * 200}] * 1})
        assert status == 200
    finally:
        api.close(); service.close()
