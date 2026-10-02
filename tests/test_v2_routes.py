"""Workspace v2 routes (WORKSPACE_V2_CONTRACT.md §1–§2): static whitelist, manifest / arrays / inputcheck / offline /
model cards, with the existing session, owner isolation, administrator and CSRF rules."""
from __future__ import annotations

import base64
import http.client
import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from wss_deploy import server, v2_data as V, v2_offline as O
from wss_deploy.jobs import JobManager
from wss_deploy.paths import PROJECT_ROOT
from wss_deploy.server import ServiceHTTPServer, SessionStore
from wss_deploy.users import UserStore

from tests._c_helpers import FakeRelease, call, stage_a_stub, stage_b_stub

DEV_ROOT = PROJECT_ROOT / "outputs" / "wss_deploy_jobs"
WALL = "20260929_230442_a7273f1670e4"
VOLUME = "20260929_230450_d1428792b846"
OLD = "20260918_152113_e2e53937b25a"
COPIED = ("job.json", "summary.json", "report.html", "stage_a.json", "input.stl", "input_clean_mm.stl")
pytestmark = pytest.mark.skipif(not all((DEV_ROOT / job / "report.html").is_file() for job in (WALL, VOLUME, OLD)),
                                reason="development job copies are not present")


@pytest.fixture(scope="module")
def jobs_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2jobs")
    for job in (WALL, VOLUME, OLD):
        (root / job).mkdir()
        for name in COPIED:
            if (DEV_ROOT / job / name).is_file():
                shutil.copy2(DEV_ROOT / job / name, root / job / name)
    return root


class _Service:
    def __init__(self, root: Path, tmp_path: Path, *, shared=False, users=None):
        self.manager = JobManager(root, release=FakeRelease(), stage_a_fn=stage_a_stub, stage_b_fn=stage_b_stub,
                                  mapping_validator=lambda *_a, **_k: [])
        self.sessions = SessionStore(tmp_path / "sessions.json", shared=shared, users=users,
                                     legacy_owner=None if shared else "admin")
        self.server = ServiceHTTPServer(("127.0.0.1", 0), self.manager, self.sessions)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def api(self):
        return http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=30)

    def session(self, api):
        status, payload, response = call(api, "GET", "/api/session", {})
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        return {"Cookie": cookie, "X-CSRF-Token": payload["csrf_token"], "Content-Type": "application/json"}

    def login(self, api, username, password):
        status, payload, response = call(api, "POST", "/api/session", {"Content-Type": "application/json"},
                                         {"username": username, "password": password})
        assert status == 200, payload
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        return {"Cookie": cookie, "X-CSRF-Token": payload["csrf_token"], "Content-Type": "application/json"}

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.manager.close()


@pytest.fixture
def service(jobs_root, tmp_path):
    V.clear_cache()
    svc = _Service(jobs_root, tmp_path)
    yield svc
    svc.close()
    V.clear_cache()


def _get(api, path, headers):
    api.request("GET", path, headers=headers)
    response = api.getresponse()
    return response.status, response.read(), response


# --------------------------------------------------------------------------------------------- manifest and arrays
def test_manifest_arrays_and_revalidation(service):
    api = service.api(); headers = service.session(api)
    try:
        status, body, response = _get(api, f"/api/v2/jobs/{WALL}/manifest", {"Cookie": headers["Cookie"]})
        assert status == 200, body
        manifest = json.loads(body)
        assert manifest["schema"] == V.SCHEMA and manifest["job"]["id"] == WALL
        etag = response.getheader("ETag")
        assert manifest["data_version"] in etag and response.getheader("Cache-Control") == server.REPORT_CACHE
        status, body, _ = _get(api, f"/api/v2/jobs/{WALL}/manifest", {"Cookie": headers["Cookie"], "If-None-Match": etag})
        assert status == 304 and body == b""
        V.clear_cache()
        data = V.load(service.manager.root / WALL)
        for key in ("mv", "f.wss.r", "f.tawss.d", "f.rrt.r", "ps"):
            status, body, response = _get(api, manifest["arrays"][key]["url"], {"Cookie": headers["Cookie"], "Accept-Encoding": "gzip"})
            assert status == 200 and response.getheader("Content-Type") == "application/octet-stream"
            assert response.getheader("Content-Encoding") is None                   # never gzipped
            assert len(body) == manifest["arrays"][key]["bytes"] and body == data.report.raw(key)
            assert response.getheader("X-WSS-Dtype") == manifest["arrays"][key]["dtype"]
            tag = response.getheader("ETag")
            assert manifest["data_version"] in tag and key in tag
            status, body, _ = _get(api, manifest["arrays"][key]["url"], {"Cookie": headers["Cookie"], "If-None-Match": tag})
            assert status == 304
        for bad in ("nope", "f.wss.x", "..", "mv%2F..", "a" * 80):
            status, body, _ = _get(api, f"/api/v2/jobs/{WALL}/arrays/{bad}", {"Cookie": headers["Cookie"]})
            assert status == 404, bad
        status, body, _ = _get(api, f"/api/v2/jobs/{VOLUME}/manifest", {"Cookie": headers["Cookie"]})
        volume = json.loads(body)
        assert status == 200 and volume["result"]["family"] == "volume"
        assert volume["job"]["companions"][0] == {"release_id": "M1_3head_3seed_20260922", "job_id": WALL, "role": "primary"}
        status, body, _ = _get(api, f"/api/v2/jobs/{OLD}/manifest", {"Cookie": headers["Cookie"]})
        assert status == 200 and [f["id"] for f in json.loads(body)["fields"]] == ["wss"]
    finally:
        api.close()


def test_login_owner_and_status_rules(service):
    api = service.api()
    try:
        for path in (f"/api/v2/jobs/{WALL}/manifest", f"/api/v2/jobs/{WALL}/arrays/mv", f"/api/v2/jobs/{WALL}/inputcheck",
                     "/api/v2/model-cards"):
            status, body, _ = _get(api, path, {})
            assert status == 401, path
        status, payload, _ = call(api, "POST", f"/api/v2/jobs/{WALL}/offline", {"Content-Type": "application/json"}, {})
        assert status == 401
        headers = service.session(api)
        status, body, _ = _get(api, "/api/v2/jobs/20990101_000000_missing/manifest", {"Cookie": headers["Cookie"]})
        assert status == 404
        status, body, _ = _get(api, "/api/v2/jobs/nope/nothing", {"Cookie": headers["Cookie"]})
        assert status == 404
        with service.manager.lock:
            service.manager.jobs[WALL]["status"] = "running"
        try:
            for path in (f"/api/v2/jobs/{WALL}/manifest", f"/api/v2/jobs/{WALL}/arrays/mv"):
                status, body, _ = _get(api, path, {"Cookie": headers["Cookie"]})
                payload = json.loads(body)
                assert status == 409 and payload["status"] == "running" and payload["error"]["message"]
            status, payload, _ = call(api, "POST", f"/api/v2/jobs/{WALL}/offline", headers, {})
            assert status == 409
        finally:
            with service.manager.lock:
                service.manager.jobs[WALL]["status"] = "done"
        # another owner's job is invisible (same 404 as a missing job)
        with service.manager.lock:
            service.manager.jobs[VOLUME]["owner"] = "someone-else"
        try:
            for path in (f"/api/v2/jobs/{VOLUME}/manifest", f"/api/v2/jobs/{VOLUME}/arrays/mv", f"/api/v2/jobs/{VOLUME}/inputcheck"):
                status, body, _ = _get(api, path, {"Cookie": headers["Cookie"]})
                assert status == 404, path
            status, payload, _ = call(api, "POST", f"/api/v2/jobs/{VOLUME}/offline", headers, {})
            assert status == 404
        finally:
            with service.manager.lock:
                service.manager.jobs[VOLUME]["owner"] = "admin"
        # the primary of a companion is listed only when the caller may see it
        with service.manager.lock:
            service.manager.jobs[WALL]["owner"] = "someone-else"
        try:
            status, body, _ = _get(api, f"/api/v2/jobs/{VOLUME}/manifest", {"Cookie": headers["Cookie"]})
            assert status == 200 and json.loads(body)["job"]["companions"] == []
        finally:
            with service.manager.lock:
                service.manager.jobs[WALL]["owner"] = "admin"
    finally:
        api.close()


def test_shared_mode_users_see_only_their_jobs_and_administrators_read_any(jobs_root, tmp_path):
    users = UserStore(tmp_path / "users.json")
    users.add("admin", "admin-pass-123", admin=True); users.add("bob", "bob-pass-12345")
    svc = _Service(jobs_root, tmp_path, shared=True, users=users)
    api = svc.api()
    try:
        with svc.manager.lock:
            svc.manager.jobs[OLD]["owner"] = "bob"
        bob = svc.login(api, "bob", "bob-pass-12345")
        assert _get(api, f"/api/v2/jobs/{WALL}/manifest", {"Cookie": bob["Cookie"]})[0] == 404
        assert _get(api, f"/api/v2/jobs/{OLD}/manifest", {"Cookie": bob["Cookie"]})[0] == 200
        admin = svc.login(api, "admin", "admin-pass-123")
        assert _get(api, f"/api/v2/jobs/{OLD}/manifest", {"Cookie": admin["Cookie"]})[0] == 200    # read-only any owner
        assert _get(api, f"/api/v2/jobs/{OLD}/arrays/pv", {"Cookie": admin["Cookie"]})[0] == 200
    finally:
        with svc.manager.lock:
            svc.manager.jobs[OLD]["owner"] = "admin"
        api.close(); svc.close()


# -------------------------------------------------------------------------------------------------- input check
def test_inputcheck_for_a_finished_and_a_failed_job(service):
    api = service.api(); headers = service.session(api)
    try:
        status, body, _ = _get(api, f"/api/v2/jobs/{WALL}/inputcheck", {"Cookie": headers["Cookie"]})
        payload = json.loads(body)
        assert status == 200 and payload["status"] == "done" and payload["mesh"]["source"] == "stage_a_preview"
        assert len(payload["openings"]) == 5 and payload["mapping_proposal"]["mapping"]
        assert payload["input_check"]["quality"]["opening_geometry"] == payload["input_check"]["opening_geometry"]
        # a job that failed its input check: stage A kept only the input check, the display mesh is the upload
        with service.manager.lock:
            job = service.manager.jobs[OLD]
            saved = {key: job.get(key) for key in ("status", "a", "error")}
            job["status"] = "failed"
            job["error"] = {"message": "输入检查未通过：开口数 4 ≠ 5", "category": "input_geometry", "retryable": False,
                            "admin_detail": "server side detail"}
            job["a"] = {"stage": "input", "created_at": "2026-09-30T00:00:00+08:00",
                        "input_check": {"status": "fail", "ok": False, "scale_factor": 1.0, "clean_stl": None,
                                        "quality": {"grade": "fail", "opening_geometry": [{"opening_index": 0}]}},
                        "centerline": None, "proposal": None}
        try:
            status, body, _ = _get(api, f"/api/v2/jobs/{OLD}/inputcheck", {"Cookie": headers["Cookie"]})
            payload = json.loads(body)
            assert status == 200 and payload["status"] == "failed" and payload["error"]["category"] == "input_geometry"
            assert payload["openings"] == [] and payload["mapping_proposal"] is None
            assert payload["mesh"]["source"] in {"clean_stl", "input_stl"} and payload["mesh"]["n_faces"] <= V.MAX_INPUT_DISPLAY_FACES
            vertices = np.frombuffer(base64.b64decode(payload["mesh"]["vertices"]), "<f4")
            assert vertices.size == payload["mesh"]["n_vertices"] * 3
            status, body, _ = _get(api, f"/api/v2/jobs/{OLD}/manifest", {"Cookie": headers["Cookie"]})
            assert status == 409
        finally:
            with service.manager.lock:
                service.manager.jobs[OLD].update(saved)
    finally:
        api.close()


# ----------------------------------------------------------------------------------------------------- offline
def test_offline_download_requires_csrf_validates_the_body_and_hides_the_name(service, tmp_path, monkeypatch):
    static = tmp_path / "static"; v2 = static / "v2"; v2.mkdir(parents=True)
    (v2 / "bundle.json").write_text(json.dumps({"legacy_scripts": ["three.min.js"], "scripts": ["core_util.js"],
                                                "offline_exclude": [], "styles": ["v2.css"], "pages": ["index.html"]}), encoding="utf-8")
    (v2 / "index.html").write_text("<!doctype html><html><head><title>x</title></head><body><div id='app'></div></body></html>", encoding="utf-8")
    (v2 / "v2.css").write_text("body{}", encoding="utf-8"); (v2 / "core_util.js").write_text("var WSSV2={};", encoding="utf-8")
    (static / "three.min.js").write_text("var THREE={};", encoding="utf-8")
    monkeypatch.setattr(O, "V2_DIR", v2); monkeypatch.setattr(O, "STATIC_DIR", static)
    api = service.api(); headers = service.session(api)
    try:
        status, payload, _ = call(api, "POST", f"/api/v2/jobs/{WALL}/offline", {**headers, "X-CSRF-Token": "wrong"}, {})
        assert status == 403
        for body in ({"hide_name": "yes"}, {"bookmarks": {"a": 1}}, {"bookmarks": [1]}, {"view": [1]},
                     {"bookmarks": [{}] * (server.MAX_OFFLINE_BOOKMARKS + 1)}):
            status, payload, _ = call(api, "POST", f"/api/v2/jobs/{WALL}/offline", headers, body)
            assert status == 400, body
        status, payload, _ = call(api, "POST", f"/api/v2/jobs/{WALL}/offline", headers, '{"view": {"x": NaN}}')
        assert status == 400
        api.request("POST", f"/api/v2/jobs/{WALL}/offline", body=json.dumps({"hide_name": True, "bookmarks": [{"name": "LV_GUO_YOU 瘤颈"}]}),
                    headers=headers)
        response = api.getresponse(); page = response.read().decode("utf-8")
        assert response.status == 200 and response.getheader("Content-Type").startswith("text/html")
        disposition = response.getheader("Content-Disposition")
        assert disposition.startswith("attachment;") and "WSS_" in disposition and ".html" in disposition
        assert "%E7%97%85%E4%BE%8B" in disposition                                # 「病例」 in the UTF-8 file name
        assert "LV_GUO_YOU" not in page and 'id="wssv2-manifest"' in page and 'id="wssv2-arrays"' in page
        api.request("POST", f"/api/v2/jobs/{WALL}/offline", body="{}", headers=headers)
        response = api.getresponse(); page = response.read().decode("utf-8")
        assert response.status == 200 and "LV_GUO_YOU" in response.getheader("Content-Disposition")
        (v2 / "core_util.js").unlink()                                              # a listed file is missing
        status, payload, _ = call(api, "POST", f"/api/v2/jobs/{WALL}/offline", headers, {})
        assert status == 503 and "core_util.js" in payload["error"]["message"]
    finally:
        api.close()


# ----------------------------------------------------------------------------------------------- static + cards
def test_static_whitelist_pages_and_example(service, tmp_path, monkeypatch):
    v2 = tmp_path / "v2"; v2.mkdir()
    (v2 / "bundle.json").write_text(json.dumps({"legacy_scripts": ["three.min.js"], "scripts": ["core_util.js", "absent.js"],
                                                "styles": ["v2.css"], "pages": ["index.html", "help_input.html"],
                                                "assets": ["example_aaa.stl"], "dev": ["dev_viewer.html"],
                                                "generated": ["example_report.html"]}), encoding="utf-8")
    for name, text in {"index.html": "<!doctype html><title>v2</title>", "help_input.html": "<p>help</p>", "v2.css": "body{}",
                       "core_util.js": "var a=1;" * 2000, "example_aaa.stl": "solid x\nendsolid x\n", "dev_viewer.html": "<p>dev</p>",
                       "secret.js": "var s=1;", "example_report.html": "<!doctype html><script>1</script>"}.items():
        (v2 / name).write_text(text, encoding="utf-8")
    (v2 / "sub").mkdir(); (v2 / "sub" / "x.js").write_text("1", encoding="utf-8")
    monkeypatch.setattr(server, "V2_DIR", v2)
    api = service.api()
    try:
        status, body, response = _get(api, "/v2", {})
        assert status == 301 and response.getheader("Location") == "/v2/"
        status, body, response = _get(api, "/v2?x=1", {})
        assert status == 301 and response.getheader("Location") == "/v2/?x=1"
        status, body, response = _get(api, "/v2/", {})
        assert status == 200 and body.startswith(b"<!doctype html>") and response.getheader("Set-Cookie")   # page opens a session
        assert "script-src 'self';" in response.getheader("Content-Security-Policy")
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        for name, ctype in (("core_util.js", "application/javascript"), ("v2.css", "text/css"), ("help_input.html", "text/html"),
                            ("example_aaa.stl", "model/stl"), ("dev_viewer.html", "text/html")):
            status, body, response = _get(api, f"/static/v2/{name}", {"Accept-Encoding": "gzip"})
            assert status == 200 and response.getheader("Content-Type").startswith(ctype), name
            assert "script-src 'self';" in response.getheader("Content-Security-Policy")
            tag = response.getheader("ETag")
            assert _get(api, f"/static/v2/{name}", {"If-None-Match": tag})[0] == 304
        assert _get(api, "/static/v2/core_util.js", {"Accept-Encoding": "gzip"})[2].getheader("Content-Encoding") == "gzip"
        for name in ("secret.js", "absent.js", "bundle.json", "example_report.html", "sub/x.js", "..%2Fapp.js", "%2e%2e/app.js",
                     "sub%2Fx.js", "three.min.js"):
            assert _get(api, f"/static/v2/{name}", {})[0] == 404, name
        assert _get(api, "/static/app.js", {})[0] == 404                             # S7: the classic workbench is retired …
        status, _, response = _get(api, "/", {})
        assert status == 302 and response.getheader("Location") == "/v2/"          # … and its address opens the workspace
        status, body, response = _get(api, "/v2/example", {"Cookie": cookie})
        assert status == 200 and "'unsafe-inline'" in response.getheader("Content-Security-Policy")
        (v2 / "example_report.html").unlink()
        assert _get(api, "/v2/example", {"Cookie": cookie})[0] == 404
        (v2 / "index.html").unlink()
        assert _get(api, "/v2/", {})[0] == 404
        (v2 / "bundle.json").write_text("not json", encoding="utf-8")
        assert _get(api, "/static/v2/core_util.js", {})[0] == 404
    finally:
        api.close()


def test_example_requires_a_session_in_shared_mode(jobs_root, tmp_path, monkeypatch):
    v2 = tmp_path / "v2"; v2.mkdir(); (v2 / "example_report.html").write_text("<!doctype html>", encoding="utf-8")
    monkeypatch.setattr(server, "V2_DIR", v2)
    users = UserStore(tmp_path / "users.json"); users.add("admin", "admin-pass-123", admin=True)
    svc = _Service(jobs_root, tmp_path, shared=True, users=users)
    api = svc.api()
    try:
        assert _get(api, "/v2/example", {})[0] == 401
        headers = svc.login(api, "admin", "admin-pass-123")
        assert _get(api, "/v2/example", {"Cookie": headers["Cookie"]})[0] == 200
    finally:
        api.close(); svc.close()


def test_model_cards_and_health_build_ids(service, monkeypatch, tmp_path):
    from wss_deploy import registry as REG
    monkeypatch.setattr(REG, "RETIRED_RELEASE_ROOT", tmp_path / "retired")       # no retired packages
    api = service.api(); headers = service.session(api)
    try:
        status, body, _ = _get(api, "/api/v2/model-cards", {"Cookie": headers["Cookie"]})
        cards = json.loads(body)["cards"]
        assert status == 200 and set(cards) == {"REL_A"}
        # 2026-10-02: a retired package's card is sent too (old results keep their own, marked name)
        (tmp_path / "retired" / "X5D_v51_5seed_20260916").mkdir(parents=True)
        (tmp_path / "retired" / "X5D_v51_5seed_20260916" / "release.json").write_text("{}", encoding="utf-8")
        status, body, _ = _get(api, "/api/v2/model-cards", {"Cookie": headers["Cookie"]})
        cards = json.loads(body)["cards"]
        assert set(cards) == {"REL_A", "X5D_v51_5seed_20260916"} and cards["X5D_v51_5seed_20260916"]["short_name"] == "峰值 WSS（旧）"
        (tmp_path / "retired" / "X5D_v51_5seed_20260916" / "release.json").unlink()

        class Cards:
            @staticmethod
            def load(release_id, release_dir=None):
                return {"release_id": release_id, "display_name": "测试卡"}
        monkeypatch.setattr(V, "_model_cards_module", lambda: Cards)
        status, body, _ = _get(api, "/api/v2/model-cards", {"Cookie": headers["Cookie"]})
        assert json.loads(body)["cards"] == {"REL_A": {"release_id": "REL_A", "display_name": "测试卡"}}
        status, body, _ = _get(api, "/api/health", {"Cookie": headers["Cookie"]})
        health = json.loads(body)
        assert status == 200 and health["ui_build"] == server.static_build_id()
        assert isinstance(health["ui_build_v2"], str) and health["ui_build_v2"] == server.static_build_id("v2")
    finally:
        api.close()
