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


def test_redact_paths_strips_project_root_and_home_everywhere():
    from pathlib import Path as _P
    from wss_deploy.paths import PROJECT_ROOT
    from wss_deploy.schema import redact_paths, model_release_metadata, ANALYSIS_VERSION
    from wss_deploy.service import needs_rebuild
    run = str(PROJECT_ROOT / "training_wss_min/runs/x/M1_s7")
    value = {"source_runs": [{"seed": 7, "path": run, "test34_metrics": run + "/eval/metrics.json"}],
             "note": f"trained at {run}", "n": 3, "home": str(_P.home() / "secret.txt")}
    out = redact_paths(value)
    text = repr(out)
    assert str(PROJECT_ROOT) not in text and "<project>/training_wss_min/runs/x/M1_s7" in text
    assert out["n"] == 3 and out["note"].startswith("trained at <project>/")
    if str(_P.home()) not in (str(PROJECT_ROOT), "/"):
        assert out["home"] == "~/secret.txt"
    assert value["source_runs"][0]["path"] == run          # the input is not mutated

    class FakeRelease:
        info = {"release": "REL", "source_runs": value["source_runs"]}
        name = "REL"
    meta = model_release_metadata(FakeRelease())
    assert meta["source_runs"][0]["path"] == "<project>/training_wss_min/runs/x/M1_s7"
    assert str(PROJECT_ROOT) not in repr(meta)
    # the analysis version moved so every finished job is rebuilt (and cleaned) by the next service upgrade
    # 2026-09-30 (C line) bumped the analysis version; every older summary, including 2026-09-26, is rebuilt.
    assert ANALYSIS_VERSION == "2026-09-30" and needs_rebuild({"analysis_version": "2026-09-26"}) and not needs_rebuild({"analysis_version": ANALYSIS_VERSION})


def test_pre_v014_summaries_with_server_paths_are_queued_for_rebuild():
    from wss_deploy.paths import PROJECT_ROOT
    from wss_deploy.schema import ANALYSIS_VERSION
    from wss_deploy.service import needs_rebuild
    old = {"peak": {}, "model_release": {"source_runs": [{"path": str(PROJECT_ROOT / "training_wss_min/runs/x")}]}}
    assert needs_rebuild(old) is True                                   # no analysis_version + absolute path
    clean = {"peak": {}, "model_release": {"source_runs": [{"path": "<project>/training_wss_min/runs/x"}]}}
    assert needs_rebuild(clean) is False                                # no analysis_version, already redacted, no cycle probe
    legacy = {"peak": {}, "input_check": {"clean_stl": str(PROJECT_ROOT / "outputs/wss_deploy_jobs/j/input_clean_mm.stl")}}
    assert needs_rebuild(legacy) is True                                # v0.3-era record: the input paths qualify too
    assert needs_rebuild({"analysis_version": ANALYSIS_VERSION, "model_release": old["model_release"]}) is False   # version wins
