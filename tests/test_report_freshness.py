"""§19.7: stale report templates are refreshed (UI only) when the report is opened, once, and never break the page."""
from __future__ import annotations

import http.client
import json
import os
import shutil
import threading
import time
from pathlib import Path

import pytest

from wss_deploy import report_freshness as F
from wss_deploy import rebuild_report as RB
from wss_deploy.cli import main as cli_main
from wss_deploy.jobs import JobManager
from wss_deploy.server import ServiceHTTPServer, SessionStore

# Checked-in copy of a finished X5D job (LV_GUO_YOU, 2026-09-20; restored from the jobs-root trash after the
# live job was deleted on 09-23): job.json, summary.json, report.html, run_manifest.json, input.stl only, byte-identical
# except job.json's session owner id, replaced by a same-length placeholder.
# WSS_DEPLOY_TEST_FIXTURES may point at another fixtures root holding the same job id.
JOB = "20260920_173929_a9ec139d6cdd"
FIXTURES = Path(os.environ.get("WSS_DEPLOY_TEST_FIXTURES") or Path(__file__).resolve().parent / "fixtures")
SOURCE = FIXTURES / JOB
FIXTURE_FILES = ("job.json", "summary.json", "report.html", "run_manifest.json")


def _require_fixture() -> None:
    missing = [name for name in FIXTURE_FILES if not (SOURCE / name).is_file()]
    if missing:
        pytest.fail(f"test fixture job missing: {SOURCE} lacks {', '.join(missing)} "
                    f"(restore tests/fixtures/{JOB}/ or set WSS_DEPLOY_TEST_FIXTURES)", pytrace=False)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    F._failed.clear()
    monkeypatch.delenv(F.ENV_SWITCH, raising=False)
    yield
    F._failed.clear()


def _copy(tmp_path) -> tuple[Path, str]:
    _require_fixture()
    root = tmp_path / "jobs"
    target = root / JOB
    target.mkdir(parents=True)
    for name in FIXTURE_FILES:
        shutil.copy2(SOURCE / name, target / name)
    owner = json.loads((target / "job.json").read_text(encoding="utf-8"))["owner"]
    return root, owner


class _Server:
    def __init__(self, root: Path, owner: str):
        self.manager = JobManager(root, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})
        self.sessions = SessionStore(root.parent / "sessions.json", shared=False, legacy_owner=owner)
        self.server = ServiceHTTPServer(("127.0.0.1", 0), self.manager, self.sessions)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        api = self.api()
        api.request("GET", "/api/session")
        response = api.getresponse(); response.read()
        self.cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        api.close()

    def api(self):
        return http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=30)

    def report(self, path=f"/api/jobs/{JOB}/report") -> tuple[int, bytes]:
        api = self.api()
        try:
            api.request("GET", path, headers={"Cookie": self.cookie})
            response = api.getresponse()
            return response.status, response.read()
        finally:
            api.close()

    def close(self):
        self.server.shutdown(); self.server.server_close(); self.manager.close()


def _counting(monkeypatch, fn=None, *, delay=0.0):
    calls = []
    real = RB.refresh_ui_only

    def wrapper(job_dir):
        calls.append(Path(job_dir).name)
        time.sleep(delay)
        return (fn or real)(job_dir)
    monkeypatch.setattr(RB, "refresh_ui_only", wrapper)
    return calls


def test_stale_report_is_refreshed_once_and_scientific_data_is_untouched(tmp_path, monkeypatch):
    root, owner = _copy(tmp_path)
    job_dir = root / JOB
    before = (job_dir / "report.html").read_text(encoding="utf-8")
    meta_before = RB._embedded_json(before, "wss-report-meta")[1]
    arrays_before = RB._embedded_json(before, "wss-report-arrays")[1]
    assert F.is_stale(job_dir)
    calls = _counting(monkeypatch)
    service = _Server(root, owner)
    try:
        status, body = service.report()
        assert status == 200 and calls == [JOB]
        marker = json.loads((job_dir / F.MARKER).read_text(encoding="utf-8"))
        assert marker["fingerprint"] == F.ui_fingerprint() and marker["source"] == "auto" and marker["refreshed_at"]
        after = body.decode("utf-8")
        assert after == (job_dir / "report.html").read_text(encoding="utf-8")
        assert RB._embedded_json(after, "wss-report-meta")[1] == meta_before
        assert RB._embedded_json(after, "wss-report-arrays")[1] == arrays_before
        manifest = json.loads((job_dir / "run_manifest.json").read_text(encoding="utf-8"))
        from wss_deploy.io_utils import file_sha256
        assert manifest["outputs"]["report.html"]["sha256"] == file_sha256(job_dir / "report.html")
        # second open, other routes to the same file: nothing to do
        assert service.report()[0] == 200 and service.report(f"/api/jobs/{JOB}/files/report.html")[0] == 200
        assert service.report(f"/jobs/{JOB}/report.html")[0] == 200
        assert calls == [JOB] and not F.is_stale(job_dir)
    finally:
        service.close()


def test_concurrent_requests_refresh_only_once(tmp_path, monkeypatch):
    root, owner = _copy(tmp_path)
    calls = _counting(monkeypatch, lambda job_dir: {"job": Path(job_dir).name}, delay=0.6)
    service = _Server(root, owner)
    try:
        results = []
        threads = [threading.Thread(target=lambda: results.append(service.report()[0])) for _ in range(3)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=30)
        assert results == [200, 200, 200] and calls == [JOB]
    finally:
        service.close()


def test_failed_refresh_serves_the_original_report_and_is_not_retried(tmp_path, monkeypatch, caplog):
    root, owner = _copy(tmp_path)
    job_dir = root / JOB
    F.write_marker(job_dir, "0" * 64, "cli")
    original = (job_dir / "report.html").read_bytes()

    def broken(_job_dir):
        raise RuntimeError("模板损坏")
    calls = _counting(monkeypatch, broken)
    service = _Server(root, owner)
    try:
        status, body = service.report()
        assert status == 200 and body == original and calls == [JOB]
        assert json.loads((job_dir / F.MARKER).read_text(encoding="utf-8"))["fingerprint"] == "0" * 64
        assert "报告模板自动刷新失败" in caplog.text
        status, body = service.report()
        assert status == 200 and body == original and calls == [JOB]      # same template version: not retried
    finally:
        service.close()


def test_switch_off_leaves_reports_alone(tmp_path, monkeypatch):
    root, owner = _copy(tmp_path)
    monkeypatch.setenv(F.ENV_SWITCH, "0")
    calls = _counting(monkeypatch, lambda job_dir: {})
    service = _Server(root, owner)
    try:
        assert service.report()[0] == 200 and calls == [] and F.is_stale(root / JOB)
    finally:
        service.close()


def test_fingerprint_follows_source_changes(tmp_path):
    for name in F.UI_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    first = F.ui_fingerprint(tmp_path)
    assert first == F.ui_fingerprint(tmp_path) and len(first) == 64
    time.sleep(0.01)
    (tmp_path / "static/report_common.js").write_text("changed viewer", encoding="utf-8")
    assert F.ui_fingerprint(tmp_path) != first
    assert F.ui_fingerprint() != first                        # the package's own sources give another value


def test_cli_reports_check_and_refresh(tmp_path, monkeypatch, capsys):
    root, _ = _copy(tmp_path)
    calls = _counting(monkeypatch, lambda job_dir: {"report_bytes": 1})
    assert cli_main(["reports", "refresh", "--check", "--jobs-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert JOB in out and "1 个需要刷新" in out and calls == []
    assert cli_main(["reports", "refresh", "--all", "--jobs-root", str(root)]) == 0
    assert calls == [JOB] and json.loads((root / JOB / F.MARKER).read_text(encoding="utf-8"))["source"] == "cli"
    assert cli_main(["reports", "refresh", "--all", "--jobs-root", str(root)]) == 0 and calls == [JOB]
    assert cli_main(["reports", "refresh", "--job", JOB, "--force", "--jobs-root", str(root)]) == 0 and calls == [JOB, JOB]
    with pytest.raises(SystemExit):
        cli_main(["reports", "refresh", "--job", "no_such_job", "--jobs-root", str(root)])


def test_auto_refresh_waits_for_upgrade_when_template_code_changed_after_start(tmp_path, monkeypatch):
    """A running service must not render new viewer scripts with the template module it loaded at start."""
    import time
    from wss_deploy import report_freshness as F
    pkg = tmp_path / "pkg"; (pkg / "static").mkdir(parents=True)
    for name in F.PY_TEMPLATES:
        (pkg / name).write_text("TEMPLATE = 'v1'\n", encoding="utf-8")
    monkeypatch.setattr(F, "PACKAGE_DIR", pkg)
    monkeypatch.setattr(F, "_code_snapshot", dict(F._py_state(pkg)))
    monkeypatch.setattr(F, "_code_warned", set())
    job = tmp_path / "job"; job.mkdir(); (job / "report.html").write_text("<html></html>", encoding="utf-8")
    calls = []
    assert F.code_changed() is False
    time.sleep(0.01)
    (pkg / "volume_report.py").write_text("TEMPLATE = 'v2 with new controls'\n", encoding="utf-8")
    assert F.code_changed() is True
    assert F.ensure_fresh(job, refresher=lambda d: calls.append(d)) == "code_changed" and calls == []
    assert not (job / F.MARKER).exists()
    # the offline path (CLI / service upgrade runs in a fresh process) is not blocked
    assert F.ensure_fresh(job, source="cli", refresher=lambda d: calls.append(d) or {}) == "refreshed" and len(calls) == 1
