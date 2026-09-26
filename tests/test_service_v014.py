"""v0.14 service management: single-writer lock (J3), upgrade preflight / drain / in-service maintenance (J6),
analysis-version rebuild detection (J9), token redaction and --token-file (J10), /api/ready probing (J5)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from wss_deploy import jobs as J
from wss_deploy import service as S
from wss_deploy.cli import main as cli_main
from wss_deploy.schema import ANALYSIS_VERSION


# ------------------------------------------------------------------------------------------ J3 lock
def test_lock_is_exclusive_names_its_holder_and_is_released(tmp_path):
    lock = S.acquire_lock(tmp_path, purpose="serve", port=9999)
    try:
        holder = S.lock_holder(tmp_path)
        assert holder["held"] is True and holder["pid"] == os.getpid() and holder["purpose"] == "serve" and holder["port"] == 9999
        with pytest.raises(S.ServiceLockError) as info:
            S.acquire_lock(tmp_path, purpose="jobs claim")
        assert str(os.getpid()) in str(info.value) and "写锁" in str(info.value) and info.value.holder["purpose"] == "serve"
    finally:
        lock.release()
    assert S.lock_holder(tmp_path)["held"] is False              # stale content, nothing to clean up
    S.acquire_lock(tmp_path, purpose="serve").release()


def test_lock_held_by_another_process_blocks_offline_commands_unless_forced(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    code = ("import sys, time; from wss_deploy import service as S; "
            "l = S.acquire_lock(sys.argv[1], purpose='serve'); print('locked', flush=True); time.sleep(60)")
    holder = subprocess.Popen([sys.executable, "-c", code, str(root)], stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(SystemExit) as info:
            cli_main(["jobs", "claim", "--owner", "old", "--user", "alice", "--jobs-root", str(root)])
        assert f"PID {holder.pid}" in str(info.value) and "--force" in str(info.value)
        assert cli_main(["jobs", "claim", "--owner", "old", "--user", "alice", "--jobs-root", str(root), "--force"]) == 0
        messages = []
        assert S.offline_lock(root, force=True, purpose="x", out=messages.append) is None and "--force" in messages[0]
        with pytest.raises(S.ServiceLockError):
            S.offline_lock(root, force=False, purpose="x")
    finally:
        holder.kill(); holder.wait()
    lock = S.offline_lock(root, force=False, purpose="x")        # released when the process died
    assert lock is not None
    lock.release()


def test_serve_refuses_a_locked_jobs_root(tmp_path, capsys):
    lock = S.acquire_lock(tmp_path, purpose="serve")
    try:
        assert cli_main(["serve", "--port", "1", "--jobs-root", str(tmp_path)]) == 1
        assert "写锁" in capsys.readouterr().err
    finally:
        lock.release()


def test_reports_refresh_refuses_while_locked_but_check_is_read_only(tmp_path, capsys, monkeypatch):
    import wss_deploy.report_freshness as F
    job = tmp_path / "20260101_000000_a"
    job.mkdir()
    (job / "job.json").write_text(json.dumps({"id": job.name, "status": "done"}))
    (job / "report.html").write_text("<html></html>")
    monkeypatch.setattr(F, "refresh", lambda job_dir, source: {"report_bytes": 1})
    lock = S.acquire_lock(tmp_path, purpose="serve")
    try:
        assert cli_main(["reports", "refresh", "--check", "--jobs-root", str(tmp_path)]) == 0
        with pytest.raises(SystemExit) as info:
            cli_main(["reports", "refresh", "--all", "--jobs-root", str(tmp_path)])
        assert "写锁" in str(info.value)
    finally:
        lock.release()
    assert cli_main(["reports", "refresh", "--all", "--jobs-root", str(tmp_path)]) == 0
    assert "✓ 20260101_000000_a" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------ J10 token handling
def test_token_values_are_redacted_and_token_file_is_parsed():
    argv = ["python", "-m", "wss_deploy.cli", "serve", "--token", "s3cret", "--port", "1", "--token=other"]
    assert S.redact_argv(argv) == ["python", "-m", "wss_deploy.cli", "serve", "--token", "***", "--port", "1", "--token=***"]
    opts = S.serve_options(["python", "-m", "wss_deploy.cli", "serve", "--token-file", "/x/tok", "--port", "9001"])
    assert opts["token_file"] == "/x/tok" and opts["token"] is None and opts["port"] == 9001


def test_adopt_reads_the_token_file_and_never_saves_a_token_in_argv(tmp_path, monkeypatch):
    token_file = tmp_path / "tok"
    token_file.write_text("file-token\n")
    info = {"pid": 4242, "own": True, "host": "0.0.0.0", "port": 9002, "device": "cpu", "python": sys.executable, "cwd": str(tmp_path),
            "argv": [sys.executable, "-m", "wss_deploy.cli", "serve", "--token", "argv-token", "--port", "9002"],
            "token": None, "token_file": "tok", "log_file": None}
    monkeypatch.setattr(S, "process_env", lambda pid: {"TZ": "UTC"})
    S.adopt(tmp_path, info, out=lambda *_: None)
    saved = (tmp_path / S.CONFIG_FILE).read_text(encoding="utf-8")
    assert "argv-token" not in saved and "***" in saved and S.read_token(tmp_path) == "file-token"


# ------------------------------------------------------------------------------------------ J9 analysis version
def _job(root, job_id, summary, status="done"):
    d = root / job_id
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({"id": job_id, "status": status}), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return d


def test_pending_rebuilds_compares_analysis_version_and_falls_back_to_key_probes(tmp_path):
    done = {"items": [], "cycle_criteria": {}}
    _job(tmp_path, "a_old_version", {"analysis_version": "2026-01-01", "findings": done})
    _job(tmp_path, "b_current", {"analysis_version": ANALYSIS_VERSION, "cycle": {"fields": {"tawss": {}, "osi": {}}}, "findings": {"items": []}})
    _job(tmp_path, "c_legacy_cycle", {"cycle": {"fields": {}}, "findings": {"items": []}})
    _job(tmp_path, "d_legacy_wall", {"findings": {"items": []}})
    assert S.pending_rebuilds(tmp_path) == ["a_old_version", "c_legacy_cycle"]
    assert S.needs_rebuild({"analysis_version": "2026-09-24"}, current="2026-10-01") is True


# ------------------------------------------------------------------------------------------ J5 readiness probe
def test_wait_healthy_uses_ready_and_reports_the_failing_checks(monkeypatch):
    answers = [{"ok": False, "checks": {"releases": {"ok": False}, "gpu": {"ok": True}}}, {"ok": True, "version": "x"}]
    monkeypatch.setattr(S, "probe_ready", lambda host, port, timeout=2: answers.pop(0) if answers else None)
    monkeypatch.setattr(S.time, "sleep", lambda _s: None)
    detail = {}
    assert S.wait_healthy("127.0.0.1", 1, timeout=5, detail=detail)["ok"] is True
    assert S.failing_checks({"checks": {"releases": {"ok": False}, "gpu": {"ok": True}}}) == ["releases"]


def test_probe_ready_falls_back_to_health_on_an_older_service():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/health":
                body = b'{"ok": true, "version": "0.13.0"}'
                self.send_response(200)
            else:
                body = b'{"error": {"message": "not found"}}'
                self.send_response(404)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert S.probe_ready("127.0.0.1", server.server_port) == {"ok": True, "version": "0.13.0"}
    finally:
        server.shutdown()


# ------------------------------------------------------------------------------------------ J6 upgrade
def test_preflight_failure_aborts_before_anything_is_stopped(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(S, "load_config", lambda root: {"port": 8765, "exists": True, "env": {}})
    monkeypatch.setattr(S, "locate", lambda root, port=None: calls.append("locate") or {"pid": 1, "port": 8765, "root_matches": True, "managed": True})
    monkeypatch.setattr(S, "stop", lambda *a, **k: calls.append("stop"))

    def run(argv, **kwargs):
        calls.append("import" if argv[1] == "-c" else "doctor")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="Traceback\nImportError: cannot import name 'x'")
    monkeypatch.setattr(S.subprocess, "run", run)
    with pytest.raises(S.ServiceError) as info:
        S.upgrade(tmp_path, out=lambda *_: None)
    assert "服务保持运行" in str(info.value) and "ImportError" in str(info.value) and calls == ["import"]


def test_preflight_doctor_failures_are_listed(tmp_path, monkeypatch):
    rows = [{"status": "ok", "label": "PyTorch", "detail": "x"}, {"status": "fail", "label": "特征合同", "detail": "hash mismatch"}]

    def run(argv, **kwargs):
        if argv[1] == "-c":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        assert "--skip" in argv and "service" in argv
        return subprocess.CompletedProcess(argv, 1, stdout=json.dumps({"checks": rows}), stderr="")
    monkeypatch.setattr(S.subprocess, "run", run)
    with pytest.raises(S.ServiceError) as info:
        S.preflight(tmp_path, {"env": {}}, out=lambda *_: None)
    assert "特征合同" in str(info.value) and "hash mismatch" in str(info.value)
    rows[1]["status"] = "warn"
    monkeypatch.setattr(S.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"checks": rows}), stderr=""))
    assert S.preflight(tmp_path, {"env": {}}, out=lambda *_: None) == {"checks": 2, "warn": 1}


def test_upgrade_order_queues_maintenance_for_the_new_service(tmp_path, monkeypatch):
    _job(tmp_path, "old_cycle", {"cycle": {"fields": {}}, "findings": {"items": []}})
    calls = []
    monkeypatch.setattr(S, "preflight", lambda root, cfg, out: calls.append("preflight"))
    monkeypatch.setattr(S, "locate", lambda root, port=None: {"pid": 1, "port": 8765, "root_matches": True, "managed": True, "own": True})
    monkeypatch.setattr(S, "load_config", lambda root: {"port": 8765, "exists": True})
    monkeypatch.setattr(S, "stop", lambda root, port, **k: calls.append(("stop", port, k.get("drain_s"))))

    def start(root, **kwargs):
        request = json.loads((tmp_path / J.MAINTENANCE_FILE).read_text(encoding="utf-8"))
        calls.append(("start", kwargs.get("port"), request["rebuild"]))
        # the new service runs the maintenance and records its result
        J.atomic_json(tmp_path / J.MAINTENANCE_RESULT, {"request_id": request["request_id"], "finished_at": "now",
                                                         "rebuilt": [], "rebuild_failed": [{"id": "old_cycle", "error": "boom"}],
                                                         "refreshed": 2, "refresh_failed": []})
        return {"pid": 2}
    monkeypatch.setattr(S, "start", start)
    lines = []
    result = S.upgrade(tmp_path, out=lines.append, rebuild=["extra"], drain_s=30.0)
    assert calls == ["preflight", ("stop", 8765, 30.0), ("start", 8765, ["extra", "old_cycle"])]
    assert result["upgrade"]["rebuild_failed"] == ["old_cycle"] and result["upgrade"]["refreshed"] == 2
    assert result["upgrade"]["maintenance"] == "done" and any("✗ 重建 old_cycle" in line for line in lines)


def test_upgrade_keeps_the_maintenance_request_when_the_new_service_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "locate", lambda root, port=None: None)
    monkeypatch.setattr(S, "load_config", lambda root: {"port": 8765, "exists": True})

    def start(root, **kwargs):
        raise S.ServiceError("新服务未能就绪")
    monkeypatch.setattr(S, "start", start)
    with pytest.raises(S.ServiceError):
        S.upgrade(tmp_path, out=lambda *_: None, preflight_check=False, rebuild=["x"])
    assert json.loads((tmp_path / J.MAINTENANCE_FILE).read_text(encoding="utf-8"))["rebuild"] == ["x"]


def test_drain_writes_the_flag_for_the_service_pid_and_waits_for_running_jobs(tmp_path, monkeypatch):
    states = [{"running": 1, "queued": 2}, {"running": 1, "queued": 2}, {"running": 0, "queued": 2}]
    monkeypatch.setattr(S, "active_jobs", lambda root: states.pop(0) if len(states) > 1 else states[0])
    monkeypatch.setattr(S, "_alive", lambda pid: True)
    lines = []
    result = S.drain(tmp_path, 4321, timeout=10, out=lines.append, poll=0.01)
    assert result["drained"] is True and json.loads((tmp_path / J.DRAIN_FILE).read_text())["pid"] == 4321
    assert "排空完成" in lines[-1]


def test_serve_reads_the_token_file_holds_the_lock_and_routes_the_access_log(tmp_path, monkeypatch, capsys):
    import logging
    import wss_deploy.server as server
    seen = {}

    def fake_serve(**kwargs):
        seen.update(kwargs)
        seen["holder"] = S.lock_holder(tmp_path)
        logging.getLogger("wss_deploy.access").info("127.0.0.1 GET /api/health 200 1 0ms")
    monkeypatch.setattr(server, "serve", fake_serve)
    token_file = tmp_path / "token"
    token_file.write_text("from-file\n")
    access = logging.getLogger("wss_deploy.access")
    before, level = list(access.handlers), access.level
    try:
        assert cli_main(["serve", "--port", "9", "--jobs-root", str(tmp_path), "--token-file", str(token_file)]) == 0
        assert seen["token"] == "from-file" and seen["holder"]["held"] and seen["holder"]["purpose"] == "serve"
        assert "from-file" not in json.dumps(seen["holder"])
        for handler in access.handlers:
            handler.flush()
        assert "GET /api/health" in (tmp_path / S.ACCESS_LOG).read_text(encoding="utf-8") and access.propagate is False
        assert S.lock_holder(tmp_path)["held"] is False                         # released on the way out
        assert cli_main(["serve", "--port", "9", "--jobs-root", str(tmp_path), "--token", "argv-secret"]) == 0
        assert "--token-file" in capsys.readouterr().err and seen["token"] == "argv-secret"
        assert "argv-secret" not in (tmp_path / S.LOCK_FILE).read_text(encoding="utf-8")
    finally:
        for handler in list(access.handlers):
            if handler not in before:
                access.removeHandler(handler); handler.close()
        access.propagate = True
        access.setLevel(level)                                                  # no global logging state leaks


def test_user_role_passwd_and_disable_end_sessions(tmp_path, capsys, monkeypatch):
    from wss_deploy.users import UserStore, generation_of
    monkeypatch.setenv("WSS_DEPLOY_PASSWORD", "correct horse battery")
    assert cli_main(["user", "add", "dora", "--jobs-root", str(tmp_path)]) == 0
    store = UserStore(tmp_path / "users.json")
    generation = generation_of(store.load()["dora"])
    assert cli_main(["user", "role", "dora", "--admin", "--jobs-root", str(tmp_path)]) == 0
    row = store.load()["dora"]
    assert row["role"] == "admin" and generation_of(row) == generation + 1 and "会话均已失效" in capsys.readouterr().out
    assert cli_main(["user", "role", "dora", "--user", "--jobs-root", str(tmp_path)]) == 0
    assert store.load()["dora"]["role"] == "user"
    with pytest.raises(SystemExit):
        cli_main(["user", "role", "dora", "--jobs-root", str(tmp_path)])
    with pytest.raises(SystemExit):
        cli_main(["user", "role", "dora", "--admin", "--user", "--jobs-root", str(tmp_path)])
    assert cli_main(["user", "passwd", "dora", "--jobs-root", str(tmp_path)]) == 0
    assert cli_main(["user", "disable", "dora", "--jobs-root", str(tmp_path)]) == 0
    assert "会话均已失效" in capsys.readouterr().out
    with pytest.raises(SystemExit) as info:
        cli_main(["user", "--help"])
    assert info.value.code == 0 and "所有已登录会话立即失效" in capsys.readouterr().out
