"""v0.15 merge glue: batch limits on the session, the static build id on logged-in health, one-pager time text."""
from __future__ import annotations

import wss_deploy
from wss_deploy import server
from wss_deploy.onepager import _full_time, _short_time
from tests._c_helpers import Service, call


def test_session_reports_batch_limits_for_browser_chunking(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        status, payload, _ = call(api, "GET", "/api/session", {})
        assert status == 200
        assert payload["max_upload_bytes"] == server.MAX_UPLOAD_BYTES
        assert payload["max_batch_bytes"] == server.MAX_BATCH_BYTES and payload["max_batch_files"] == server.MAX_BATCH_FILES
    finally:
        api.close(); service.close()


def test_logged_in_health_carries_the_static_build_id_but_anonymous_health_does_not(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        status, payload, _ = call(api, "GET", "/api/health", {})
        assert status == 200 and set(payload) == {"ok", "version"}
        cookie, _, _ = service.session(api)
        status, payload, _ = call(api, "GET", "/api/health", {"Cookie": cookie})
        assert status == 200 and payload["version"] == wss_deploy.__version__
        build = payload["ui_build"]
        assert isinstance(build, str) and len(build) == 12 and int(build, 16) >= 0
        assert build == server.static_build_id()      # stable across calls (cached by stat signature)
    finally:
        api.close(); service.close()


def test_static_build_id_follows_the_file_contents(tmp_path, monkeypatch):
    static = tmp_path / "static"; static.mkdir()
    (static / "app.js").write_text("var a = 1;", encoding="utf-8"); (static / "app.css").write_text("body{}", encoding="utf-8")
    monkeypatch.setattr(server, "STATIC_DIR", static)
    monkeypatch.setattr(server, "STATIC_FILES", {"app.js", "app.css", "missing.js"})
    monkeypatch.setattr(server, "_STATIC_BUILD", {})
    first = server.static_build_id()
    assert len(first) == 12 and first == server.static_build_id()
    (static / "app.js").write_text("var a = 2;", encoding="utf-8")
    import os; os.utime(static / "app.js", ns=(os.stat(static / "app.js").st_atime_ns, os.stat(static / "app.js").st_mtime_ns + 10_000_000))
    second = server.static_build_id()
    assert second != first and len(second) == 12
    monkeypatch.setattr(server, "STATIC_DIR", tmp_path / "nowhere")
    monkeypatch.setattr(server, "_STATIC_BUILD", {})
    assert server.static_build_id() == ""              # unreadable directory: empty, never an exception


def test_one_pager_time_text_accepts_iso_with_offset_and_plain_local_forms():
    assert _full_time("2026-09-26T14:00:05+08:00") == "2026-09-26 14:00:05"
    assert _full_time("2026-09-24 18:23:40") == "2026-09-24 18:23:40"
    assert _short_time("2026-09-26T14:00:05+08:00") == "2026-09-26 14:00"
    assert _full_time(None) == "—" and _short_time("") == "—"
