"""v0.15 上线前可运维加固 (O1–O6): deployment clock, doctor pre-launch checks, status / ready summary,
jobs du / prune-cache, service rehearse and crash visibility."""
from __future__ import annotations

import json
import logging
import os
import re
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from wss_deploy import clock
from wss_deploy import doctor as D
from wss_deploy import housekeeping as H
from wss_deploy import service as S
from wss_deploy.cli import main as cli_main
from tests._c_helpers import Service, call, finished, manager

PROJECT = Path(__file__).resolve().parents[1]
FIXED_TS = 1790000000          # 2026-09-21T14:13:20Z


@pytest.fixture(autouse=True)
def _isolated_clock(tmp_path, monkeypatch):
    """Every test resolves the zone against its own empty jobs root (never the production service.json)."""
    monkeypatch.delenv(clock.ENV_TZ, raising=False)
    clock.configure(tmp_path / "clock_root")
    yield
    clock.configure(None)


def _snapshot(root: Path) -> list:
    return sorted((str(p.relative_to(root)), p.lstat().st_size, p.lstat().st_mtime_ns, stat.S_IMODE(p.lstat().st_mode))
                  for p in root.rglob("*"))


# ------------------------------------------------------------------------------------------ O1 clock
def test_now_iso_is_stable_under_a_fixed_zone(monkeypatch):
    monkeypatch.setenv(clock.ENV_TZ, "Asia/Shanghai")
    assert clock.iso(FIXED_TS) == "2026-09-21T22:13:20+08:00"
    assert clock.strftime("%Y%m%d_%H%M%S", FIXED_TS) == "20260921_221320"
    stamp = clock.now_iso()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00", stamp)
    assert clock.now_iso()[-6:] == "+08:00" and clock.now_local().utcoffset().total_seconds() == 8 * 3600
    monkeypatch.setenv(clock.ENV_TZ, "UTC")
    assert clock.iso(FIXED_TS) == "2026-09-21T14:13:20+00:00" and clock.iso(None) is None
    assert clock.iso(0) == "1970-01-01T00:00:00+00:00"


def test_zone_priority_env_then_service_json_then_system_and_the_shell_tz_is_ignored(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    root.mkdir()
    clock.configure(root)
    monkeypatch.setenv("TZ", "America/Los_Angeles")            # the calling shell never decides
    monkeypatch.setattr(clock, "system_zone", lambda: ("Asia/Shanghai", ZoneInfo("Asia/Shanghai")))
    info = clock.resolve()
    assert info["source"] == "system" and info["name"] == "Asia/Shanghai" and clock.iso(FIXED_TS).endswith("+08:00")
    (root / "service.json").write_text(json.dumps({"host": "127.0.0.1", "env": {"TZ": "UTC"}}), encoding="utf-8")
    assert clock.resolve()["source"] == "service.json" and clock.iso(FIXED_TS).endswith("+00:00")
    clock.configure(root, tz="Asia/Tokyo")                       # a --env TZ= about to be saved
    assert clock.resolve()["source"] == "pending" and clock.iso(FIXED_TS).endswith("+09:00")
    monkeypatch.setenv(clock.ENV_TZ, "Europe/Berlin")
    assert clock.resolve()["source"] == clock.ENV_TZ and clock.iso(FIXED_TS).endswith("+02:00")
    monkeypatch.setenv(clock.ENV_TZ, "Mars/Olympus")             # invalid: falls through, reported
    info = clock.resolve()
    assert info["source"] == "pending" and any("Mars/Olympus" in e for e in info["errors"])
    # an old service.json without env.TZ (and one without env at all) still reads fine
    monkeypatch.delenv(clock.ENV_TZ)
    clock.configure(root)
    (root / "service.json").write_text(json.dumps({"host": "0.0.0.0", "port": 8765}), encoding="utf-8")
    assert clock.resolve()["source"] == "system"
    # describe(root) for another root never changes the configured one
    other = tmp_path / "other"
    other.mkdir()
    (other / "service.json").write_text(json.dumps({"env": {"TZ": "UTC"}}), encoding="utf-8")
    assert clock.describe(other)["name"] == "UTC" and clock.resolve()["jobs_root"] == str(root)


def test_log_formatter_job_dates_and_job_ids_use_the_deployment_zone(tmp_path, monkeypatch):
    from wss_deploy import jobs as J
    monkeypatch.setenv(clock.ENV_TZ, "Asia/Shanghai")
    record = logging.LogRecord("wss_deploy.service", logging.INFO, __file__, 1, "hello", (), None)
    record.created, record.msecs = FIXED_TS, 0
    line = clock.LogFormatter("%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z").format(record)
    assert line == "2026-09-21T22:13:20+0800 INFO wss_deploy.service hello"
    assert J._date(FIXED_TS) == "2026-09-21T22:13:20+08:00"
    mgr = manager(tmp_path / "jobs")
    try:
        job = mgr.create("o", content=b"solid", filename="a.stl")
        assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{12}", job["id"]) and len(job["id"]) == 28     # format and length unchanged
        assert job["created_at"].endswith("+08:00")
    finally:
        mgr.close()
    from wss_deploy.export_table import export_filename
    assert re.fullmatch(r"wss_summary_\d{8}_\d{4}\.csv", export_filename("csv"))


def test_serve_and_access_log_lines_carry_the_configured_offset(tmp_path, monkeypatch):
    """server.log / access.log use clock.LogFormatter (not the process TZ)."""
    import wss_deploy.server as server
    monkeypatch.setenv(clock.ENV_TZ, "UTC")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    root = tmp_path / "jobs"
    seen = {}

    def fake_serve(**kwargs):
        seen.update(kwargs)
        logging.getLogger("wss_deploy.access").info("127.0.0.1 GET /api/health 200 1 0ms")
    monkeypatch.setattr(server, "serve", fake_serve)
    access = logging.getLogger("wss_deploy.access")
    before, level, hook = list(access.handlers), access.level, threading.excepthook
    try:
        assert cli_main(["serve", "--port", "9", "--jobs-root", str(root)]) == 0
        for handler in access.handlers:
            handler.flush()
        text = (root / S.ACCESS_LOG).read_text(encoding="utf-8")
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+0000 INFO wss_deploy.access", text)
        assert getattr(threading.excepthook, "_wss_crash_logging", False)
    finally:
        for handler in list(access.handlers):
            if handler not in before:
                access.removeHandler(handler); handler.close()
        access.propagate = True
        access.setLevel(level)
        threading.excepthook = hook


# ------------------------------------------------------------------------------------------ O2 doctor
def _users(root: Path, rows: dict) -> None:
    (root / "users.json").write_text(json.dumps(rows), encoding="utf-8")
    os.chmod(root / "users.json", 0o600)


def test_exposure_requires_password_login_with_an_enabled_admin(tmp_path, monkeypatch):
    monkeypatch.delenv("WSS_DEPLOY_TRUST_PROXY", raising=False)
    root = tmp_path / "jobs"
    root.mkdir()
    rows = {r["key"]: r for r in D.check_exposure(root)}
    assert rows["bind_login"]["status"] == "ok" and "tls" not in rows                 # no service.json → loopback
    rows = {r["key"]: r for r in D.check_exposure(root, "0.0.0.0")}
    assert rows["bind_login"]["status"] == "fail" and "user add" in rows["bind_login"]["advice"]
    assert rows["tls"]["status"] == "info" and "反代 TLS" in rows["tls"]["advice"] and "直接暴露 HTTP" in rows["tls"]["detail"]
    _users(root, {"bob": {"role": "user", "disabled": False}, "old": {"role": "admin", "disabled": True}})
    assert {r["key"]: r for r in D.check_exposure(root, "0.0.0.0")}["bind_login"]["status"] == "fail"
    _users(root, {"bob": {"role": "user"}, "admin": {"role": "admin", "disabled": False}})
    assert {r["key"]: r for r in D.check_exposure(root, "0.0.0.0")}["bind_login"]["status"] == "ok"
    # the host comes from service.json when not given
    (root / "users.json").unlink()
    (root / "service.json").write_text(json.dumps({"host": "0.0.0.0", "port": 8765}), encoding="utf-8")
    assert {r["key"]: r for r in D.check_exposure(root)}["bind_login"]["status"] == "fail"
    # loopback behind a trusted proxy still needs login; no "direct HTTP" row
    (root / "service.json").write_text(json.dumps({"host": "127.0.0.1", "env": {"WSS_DEPLOY_TRUST_PROXY": "1"}}), encoding="utf-8")
    rows = {r["key"]: r for r in D.check_exposure(root)}
    assert rows["bind_login"]["status"] == "fail" and "tls" not in rows
    monkeypatch.setenv("WSS_DEPLOY_TRUST_PROXY", "1")
    rows = {r["key"]: r for r in D.check_exposure(root, "0.0.0.0")}
    assert rows["tls"]["status"] == "info" and "WSS_DEPLOY_TRUST_PROXY=1" in rows["tls"]["detail"]


def test_secret_file_permissions(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    assert D.check_secret_perms(root)[0]["status"] == "ok"
    for name in D.SECRET_FILES:
        (root / name).write_text("{}")
        os.chmod(root / name, 0o600)
    assert D.check_secret_perms(root)[0]["status"] == "ok"
    os.chmod(root / ".sessions.json", 0o644)
    row = D.check_secret_perms(root)[0]
    assert row["status"] == "warn" and ".sessions.json 为 0644" in row["detail"] and row["advice"] == f"chmod 600 {root / '.sessions.json'}"


def test_disk_thresholds_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setattr("wss_deploy.service.disk_free_gb", lambda path: 30.0)
    monkeypatch.delenv(D.MIN_FREE_ENV, raising=False)
    assert D.disk_thresholds() == (5.0, 20.0)
    assert {r["key"]: r["status"] for r in D.check_jobs_root(tmp_path)}["disk"] == "ok"
    monkeypatch.setenv(D.MIN_FREE_ENV, "50")
    assert D.disk_thresholds() == (50.0, 50.0)
    assert {r["key"]: r["status"] for r in D.check_jobs_root(tmp_path)}["disk"] == "fail"
    monkeypatch.setenv(D.MIN_FREE_ENV, "5,100")
    assert {r["key"]: r["status"] for r in D.check_jobs_root(tmp_path)}["disk"] == "warn"
    monkeypatch.setenv(D.MIN_FREE_ENV, "abc")
    assert D.disk_thresholds() == (5.0, 20.0)
    # /api/ready uses the same thresholds
    monkeypatch.setenv(D.MIN_FREE_ENV, "1000000")
    mgr = manager(tmp_path / "m")
    try:
        disk = mgr.health()["checks"]["disk"]
        assert disk["ok"] is False and disk["fail_below_gb"] == 1000000.0
    finally:
        mgr.close()


def test_log_rotation_derived_cache_and_release_pins(tmp_path, monkeypatch):
    import hashlib
    root = tmp_path / "jobs"
    root.mkdir()
    (root / "access.log").write_bytes(b"x" * 100)
    row = D.check_log_rotation(root)[0]
    assert row["status"] == "ok" and "20 MB × 5" in row["detail"]
    monkeypatch.setattr("wss_deploy.server.LOG_MAX_BYTES", 50)
    row = D.check_log_rotation(root)[0]
    assert row["status"] == "warn" and "access.log" in row["detail"] and "轮转没有生效" in row["detail"]
    # derived caches: information only
    job = root / "20260101_000000_aaaaaaaaaaaa"
    (job / "geometry_cache").mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"id": job.name, "status": "done"}))
    (job / "geometry_cache" / "mesh-x.npz").write_bytes(b"c" * 2048)
    (root / ".tmp").mkdir()
    (root / ".tmp" / "upload_x.stl").write_bytes(b"t" * 10)
    row = D.check_derived_cache(root)[0]
    assert row["status"] == "info" and "geometry_cache 1 个任务 1 个文件 2.0 KB" in row["detail"] and ".tmp 1 个文件" in row["detail"]
    # release pins against env/releases.sha256
    rel = tmp_path / "releases" / "REL"
    rel.mkdir(parents=True)
    (rel / "release.json").write_text("{}")
    (rel / "MANIFEST.sha256").write_text("")

    class Registry:
        root = tmp_path / "releases"
    monkeypatch.setattr("wss_deploy.registry.ReleaseRegistry", lambda *a, **k: Registry())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()   # noqa: E731
    pins = {"REL/release.json": digest(rel / "release.json"), "REL/MANIFEST.sha256": digest(rel / "MANIFEST.sha256")}
    monkeypatch.setattr(D, "_pinned_release_hashes", lambda: dict(pins))
    assert D.check_release_pins()[0]["status"] == "ok"
    (rel / "release.json").write_text('{"changed": 1}')
    row = D.check_release_pins()[0]
    assert row["status"] == "warn" and "REL/release.json" in row["detail"]
    monkeypatch.setattr(D, "_pinned_release_hashes", lambda: {"OTHER/release.json": "0" * 64})
    assert D.check_release_pins()[0]["status"] == "info"


def test_tz_row_reports_source_and_invalid_names(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    root.mkdir()
    monkeypatch.setattr(clock, "system_zone", lambda: ("Asia/Shanghai", ZoneInfo("Asia/Shanghai")))
    row = D.check_tz(root)[0]
    assert row["status"] == "info" and "Asia/Shanghai" in row["detail"] and "--env TZ=Asia/Shanghai" in row["advice"]
    (root / "service.json").write_text(json.dumps({"env": {"TZ": "Asia/Shanghai"}}), encoding="utf-8")
    assert D.check_tz(root)[0]["status"] == "ok"
    (root / "service.json").write_text(json.dumps({"env": {"TZ": "Nowhere/City"}}), encoding="utf-8")
    row = D.check_tz(root)[0]
    assert row["status"] == "warn" and "Nowhere/City" in row["detail"]


def test_doctor_never_writes_to_the_jobs_root(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    job = root / "20260101_000000_aaaaaaaaaaaa"
    (job / "geometry_cache").mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"id": job.name, "status": "done"}))
    (job / "geometry_cache" / "mesh-x.npz").write_bytes(b"c")
    (root / ".tmp").mkdir()
    (root / "service.json").write_text(json.dumps({"host": "0.0.0.0", "env": {"TZ": "UTC"}}))
    _users(root, {"admin": {"role": "admin"}})
    (root / "access.log").write_text("x\n")
    before = _snapshot(root)
    rows = D.run_checks(root, port=1, skip={"torch", "vmtk", "releases", "features", "glossary", "git"})
    assert {"bind_login", "tls", "secret_perms", "log_rotation", "derived_cache", "tz", "disk"} <= {r["key"] for r in rows}
    assert _snapshot(root) == before
    assert "ℹ" in D.format_rows(rows) and "硬错" in D.format_rows(rows)


def test_hard_exposure_error_fails_doctor_and_blocks_the_upgrade_preflight(tmp_path, monkeypatch):
    """A shared bind without an enabled admin is a ✗: ``doctor`` exits 1 and ``service upgrade`` stops before
    touching the running service; the preflight judges the ``--host`` / ``--env`` overrides being started."""
    root = tmp_path / "jobs"
    root.mkdir()
    code = cli_main(["doctor", "--json", "--jobs-root", str(root), "--host", "0.0.0.0",
                     "--skip", "torch,vmtk,releases,features,glossary,service,git,reports"])
    assert code == 1
    # real preflight subprocess (import check + doctor) on a jobs root whose saved config is shared without users
    (root / "service.json").write_text(json.dumps({"host": "0.0.0.0", "port": 8765, "python": sys.executable, "cwd": str(PROJECT),
                                                   "env": {"CUDA_VISIBLE_DEVICES": ""}}), encoding="utf-8")
    with pytest.raises(S.ServiceError) as info:
        S.preflight(root, out=lambda *_: None)
    assert "共享绑定与登录" in str(info.value) and "服务保持运行" in str(info.value)
    # upgrade: the override host and TZ reach the doctor subprocess, and nothing is stopped
    seen, calls = {}, []

    def run(argv, **kwargs):
        if argv[1] == "-c":
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        seen.update(argv=argv, env=kwargs.get("env"))
        rows = [{"status": "fail", "label": "共享绑定与登录", "detail": "没有 users.json"}]
        return subprocess.CompletedProcess(argv, 1, stdout=json.dumps({"checks": rows}), stderr="")
    monkeypatch.setattr(S.subprocess, "run", run)
    monkeypatch.setattr(S, "stop", lambda *a, **k: calls.append("stop"))
    monkeypatch.setattr(S, "locate", lambda root, port=None: {"pid": 1, "port": 8765, "root_matches": True, "managed": True, "own": True})
    (root / "service.json").write_text(json.dumps({"host": "127.0.0.1", "port": 8765}), encoding="utf-8")
    with pytest.raises(S.ServiceError):
        S.upgrade(root, out=lambda *_: None, host="0.0.0.0", env_items=["TZ=Asia/Shanghai"])
    assert calls == [] and seen["argv"][seen["argv"].index("--host") + 1] == "0.0.0.0"
    assert seen["env"]["TZ"] == "Asia/Shanghai" and seen["env"][clock.ENV_TZ] == "Asia/Shanghai"
    assert json.loads((root / "service.json").read_text())["host"] == "127.0.0.1"      # nothing saved on failure


# ------------------------------------------------------------------------------------------ O3 status / ready
def test_status_reports_login_maintenance_and_a_crashed_service(tmp_path, monkeypatch, capsys):
    root = tmp_path / "jobs"
    root.mkdir()
    port = S_free_port()
    (root / "service.json").write_text(json.dumps({"host": "0.0.0.0", "port": port, "log_file": str(root / "server.log")}))
    _users(root, {"admin": {"role": "admin"}})
    (root / "maintenance_result.json").write_text(json.dumps({
        "request_id": "r1", "requested_at": "2026-09-24T17:40:04+08:00", "started_at": "2026-09-24T17:40:05+08:00",
        "finished_at": "2026-09-24T17:40:06+08:00", "rebuilt": ["a"], "rebuild_failed": [], "refreshed": 5, "refresh_failed": [],
        "complete": True}))
    (root / "server.log").write_text("\n".join(f"line {i}" for i in range(50)) + "\nTraceback: boom\n")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (root / "service.pid").write_text(json.dumps({"pid": dead.pid, "started_at": "2026-09-25T10:00:00+08:00",
                                                  "log_file": str(root / "server.log")}))
    info = S.status(root)
    assert info["running"] is False and info["stale_pid_file"] == dead.pid and info["crashed"] is True
    assert info["login"] == "password" and info["log_tail"][-1] == "Traceback: boom" and len(info["log_tail"]) == S.LOG_TAIL_LINES
    assert info["maintenance"]["upgraded_at"] == "2026-09-24T17:40:04+08:00" and info["maintenance"]["rebuilt"] == 1
    assert info["start_command"].startswith("python -m wss_deploy.cli service start --jobs-root")
    text = S.format_status(info)
    assert "进程" in text and "已不在" in text and "Traceback: boom" in text and "重新启动：python -m wss_deploy.cli service start" in text
    assert "最近升级" in text and "刷新报告 5" in text and "用户名登录" in text
    assert cli_main(["service", "status", "--json", "--jobs-root", str(root)]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["crashed"] is True and payload["login"] == "password" and payload["disk_free_gb"] is not None
    # without a pid file nothing is called a crash; loopback without proxy = login-free
    (root / "service.pid").unlink()
    (root / "service.json").write_text(json.dumps({"host": "127.0.0.1", "port": port}))
    info = S.status(root)
    assert "crashed" not in info and info["login"] == "none"


def S_free_port() -> int:
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_ready_summary_only_for_administrators(tmp_path):
    from wss_deploy.users import UserStore
    store = UserStore(tmp_path / "users.json")
    store.add("boss", "correct horse battery", admin=True)
    store.add("user1", "correct horse battery")
    service = Service(tmp_path, shared=True, users=store)
    api = service.api()
    try:
        status, payload, _ = call(api, "GET", "/api/ready", {})
        assert payload == {"ok": payload["ok"], "version": payload["version"]}           # anonymous: nothing else
        status, _, cookie, _ = service.login(api, {"username": "user1", "password": "correct horse battery"})
        status, payload, _ = call(api, "GET", "/api/ready", {"Cookie": cookie})
        assert "summary" not in payload
        status, _, cookie, _ = service.login(api, {"username": "boss", "password": "correct horse battery"})
        finished(service.manager, owner="boss", case_id="A")
        status, payload, _ = call(api, "GET", "/api/ready", {"Cookie": cookie})
        summary = payload["summary"]
        assert summary["login"] == "password" and summary["bind"]["shared"] is True and summary["pid"] == os.getpid()
        assert set(summary["queue"]) == {"queued", "running", "awaiting_input", "awaiting_confirmation"}
        for key in ("version", "uptime_s", "started_at", "maintenance", "disk_free_gb", "tz", "last_start"):
            assert key in summary, key
        assert status == (200 if payload["ok"] else 503)
    finally:
        api.close(); service.close()


# ------------------------------------------------------------------------------------------ O4 du / prune-cache
def _job_dir(root: Path, job_id: str, status: str, *, cache: int = 0, age_days: float = 0) -> Path:
    job = root / job_id
    job.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"id": job_id, "status": status, "case_id": job_id[-4:]}))
    (job / "field.npz").write_bytes(b"f" * 1000)
    (job / "report.html").write_bytes(b"r" * 500)
    if cache:
        (job / "geometry_cache").mkdir()
        for index in range(2):
            path = job / "geometry_cache" / f"mesh-{index}.npz"
            path.write_bytes(b"c" * cache)
            old = time.time() - age_days * 86400
            os.utime(path, (old, old))
    return job


def test_jobs_du_lists_jobs_by_size_with_parts(tmp_path, capsys):
    root = tmp_path / "jobs"
    _job_dir(root, "20260101_000000_000000000001", "done", cache=5000)
    _job_dir(root, "20260101_000000_000000000002", "done")
    (root / ".tmp").mkdir()
    (root / ".tmp" / "spool").write_bytes(b"s" * 7)
    before = _snapshot(root)
    info = H.usage(root)
    assert [row["id"] for row in info["jobs"]] == ["20260101_000000_000000000001", "20260101_000000_000000000002"]
    big = info["jobs"][0]
    assert big["geometry_cache"] == 10000 and big["field_npz"] == 1000 and big["report_html"] == 500
    assert big["bytes"] == big["field_npz"] + big["report_html"] + big["geometry_cache"] + big["history"] + big["other"]
    assert info["tmp"]["bytes"] == 7 and info["geometry_cache_bytes"] == 10000
    assert cli_main(["jobs", "du", "--jobs-root", str(root), "--top", "1"]) == 0
    text = capsys.readouterr().out
    assert "000000000001" in text and "000000000002" not in text and "显示前 1 个" in text
    assert cli_main(["jobs", "du", "--jobs-root", str(root), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["jobs"]) == 2
    assert _snapshot(root) == before                                              # read only


def test_prune_cache_is_dry_run_by_default_and_only_touches_derived_files(tmp_path, capsys):
    root = tmp_path / "jobs"
    old = _job_dir(root, "20260101_000000_000000000001", "done", cache=100, age_days=40)
    fresh = _job_dir(root, "20260101_000000_000000000002", "done", cache=100, age_days=1)
    active = _job_dir(root, "20260101_000000_000000000003", "awaiting_confirmation", cache=100, age_days=40)
    (root / ".tmp").mkdir()
    stale = root / ".tmp" / "bundle_x.zip"
    stale.write_bytes(b"z")
    os.utime(stale, (time.time() - 40 * 86400,) * 2)
    plan = H.prune_plan(root, 30)
    paths = sorted(item["path"] for item in plan["items"])
    assert paths == [".tmp/bundle_x.zip", f"{old.name}/geometry_cache/mesh-0.npz", f"{old.name}/geometry_cache/mesh-1.npz"]
    assert [row["id"] for row in plan["skipped_active"]] == [active.name]
    before = _snapshot(root)
    assert cli_main(["jobs", "prune-cache", "--older-than", "30", "--jobs-root", str(root)]) == 0
    assert "未删除任何文件" in capsys.readouterr().out and _snapshot(root) == before
    assert cli_main(["jobs", "prune-cache", "--older-than", "30", "--jobs-root", str(root), "--yes", "--dry-run"]) == 0
    assert _snapshot(root) == before
    with pytest.raises(SystemExit):
        cli_main(["jobs", "prune-cache", "--jobs-root", str(root)])                # --older-than is required
    lock = S.acquire_lock(root, purpose="serve")
    try:
        # the running service holds the writer lock: refused unless --force (checked in-process: flock is per open file)
        code = subprocess.run([sys.executable, "-m", "wss_deploy.cli", "jobs", "prune-cache", "--older-than", "30", "--jobs-root", str(root),
                               "--yes"], cwd=str(PROJECT), env={**os.environ, "PYTHONPATH": str(PROJECT)}, capture_output=True, text=True)
        assert code.returncode != 0 and "写锁" in code.stderr
    finally:
        lock.release()
    assert cli_main(["jobs", "prune-cache", "--older-than", "30", "--jobs-root", str(root), "--yes"]) == 0
    assert not (old / "geometry_cache").exists() and not stale.exists()
    assert len(list((fresh / "geometry_cache").iterdir())) == 2 and len(list((active / "geometry_cache").iterdir())) == 2
    assert (old / "field.npz").is_file() and (old / "report.html").is_file() and (old / "job.json").is_file()
    log = [json.loads(line) for line in (root / H.PRUNE_LOG).read_text().splitlines()]
    assert log[-1]["files"] == 3 and log[-1]["bytes"] == 201


# ------------------------------------------------------------------------------------------ O5 rehearse
def _source_root(tmp_path: Path) -> tuple[Path, str]:
    src = tmp_path / "source"
    mgr = manager(src)
    try:
        job = finished(mgr, owner="admin", case_id="REH")
    finally:
        mgr.close()
    return src, job["id"]


def _fixture_root(tmp_path: Path) -> tuple[Path, str]:
    """S7: the smoke test reads the workspace data of the job (parsed from its report.html), so it needs a real finished
    job: the checked-in LV_GUO_YOU copy (tests/fixtures, see test_report_freshness.py)."""
    import shutil
    job_id = "20260920_173929_a9ec139d6cdd"
    fixture = Path(__file__).resolve().parent / "fixtures" / job_id
    target = tmp_path / "source" / job_id
    target.mkdir(parents=True)
    for name in ("job.json", "summary.json", "report.html", "run_manifest.json"):
        shutil.copy2(fixture / name, target / name)
    return tmp_path / "source", job_id


def test_rehearse_runs_the_code_on_disk_against_a_copy_and_cleans_up(tmp_path):
    src, job_id = _fixture_root(tmp_path)
    before = _snapshot(src)
    lines = []
    result = __import__("wss_deploy.rehearse", fromlist=["rehearse"]).rehearse(src, workdir=tmp_path, timeout=120, out=lines.append)
    assert result["ok"], json.dumps(result, ensure_ascii=False)[:3000]
    names = [row["name"] for row in result["steps"]]
    assert names == ["复制任务", "临时账号", "启动临时服务", "就绪（未登录）", "登录", "任务列表与详情", "打开工作区与一页纸", "就绪（管理员摘要）"]
    assert result["jobs"] == [job_id] and not Path(result["dir"]).exists()          # cleaned up
    assert _snapshot(src) == before                                                # the source is only read
    assert "演练通过" in lines[-1]


def test_rehearse_reports_a_server_that_dies_and_still_cleans_up(tmp_path):
    from wss_deploy import rehearse as R
    src, _ = _source_root(tmp_path)

    def launcher(workdir, port, cfg, env):
        assert env["CUDA_VISIBLE_DEVICES"] == "" and env["WSS_DEPLOY_TRUST_PROXY"] == "1" and env["WSS_DEPLOY_PRELOAD"] == "0"
        assert "WSS_DEPLOY_TOKEN" not in env and "WSS_DEPLOY_LEGACY_OWNER" not in env
        (workdir / "server.log").write_text("Traceback (most recent call last):\nImportError: boom\n")
        return subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    result = R.rehearse(src, workdir=tmp_path, timeout=20, out=lambda *_: None, launcher=launcher)
    assert result["ok"] is False and not Path(result["dir"]).exists()
    failed = [row for row in result["steps"] if row["ok"] is False]
    assert failed[0]["name"] == "就绪（未登录）" and "已退出" in failed[0]["detail"]
    assert result["server_log_tail"][-1] == "ImportError: boom"
    assert all(row["ok"] is None for row in result["steps"][result["steps"].index(failed[0]) + 1:])
    with pytest.raises(R.RehearsalError):
        R.pick_jobs(src, ["does_not_exist"])


# ------------------------------------------------------------------------------------------ O6 crash visibility
def test_a_dead_worker_is_logged_and_makes_ready_503(tmp_path, caplog):
    service = Service(tmp_path)
    mgr = service.manager
    calls = {"n": 0}
    original = mgr.run_next

    def run_next(timeout=0.1):
        calls["n"] += 1
        if threading.current_thread().name == "wss-worker-1":
            raise SystemExit("library called exit()")
        return original(timeout=timeout)
    mgr.run_next = run_next
    api = service.api()
    try:
        with caplog.at_level(logging.CRITICAL, logger="wss_deploy.service"):
            mgr.start()
            deadline = time.time() + 5
            while time.time() < deadline and all(w.is_alive() for w in mgr._workers):
                time.sleep(0.05)
        assert not mgr._workers[0].is_alive() and mgr._workers[1].is_alive()
        records = [r for r in caplog.records if "wss-worker-1 died" in r.getMessage()]
        assert records and "diagnostic_id=" in records[0].getMessage()
        health = mgr.health()
        assert health["ok"] is False and health["checks"]["worker"]["alive"] == 1
        assert health["checks"]["worker"]["failures"][0]["thread"] == "wss-worker-1"
        status, payload, _ = call(api, "GET", "/api/ready", {})
        assert status == 503 and payload["ok"] is False
    finally:
        api.close(); service.close()


def test_crash_logging_hook_and_request_thread_errors_carry_a_diagnostic_id(tmp_path, caplog):
    previous = threading.excepthook
    try:
        S.install_crash_logging()
        S.install_crash_logging()                                # idempotent
        with caplog.at_level(logging.CRITICAL, logger="wss_deploy.service"):
            thread = threading.Thread(target=lambda: 1 / 0, name="wss-precompute")
            thread.start(); thread.join()
        assert any("Thread wss-precompute died; diagnostic_id=" in r.getMessage() for r in caplog.records)
    finally:
        threading.excepthook = previous
    service = Service(tmp_path)
    try:
        with caplog.at_level(logging.ERROR):
            try:
                raise RuntimeError("stream broke after headers")
            except RuntimeError:
                service.server.handle_error(None, ("127.0.0.1", 1))
        assert any("Request thread failed; diagnostic_id=" in r.getMessage() for r in caplog.records)
    finally:
        service.close()


def test_precompute_thread_is_restarted_by_the_next_schedule(tmp_path):
    mgr = manager(tmp_path / "jobs")
    mgr._precompute_enabled = True
    mgr._precompute_fn = lambda *a, **k: {"ok": True}
    try:
        dead = threading.Thread(target=lambda: None, name="wss-precompute")
        dead.start(); dead.join()
        mgr._pc_thread = dead                                    # the loop died
        mgr._schedule_precompute({"id": "x", "status": "awaiting_confirmation"})
        assert mgr._pc_thread is not dead and mgr._pc_thread.name == "wss-precompute" and mgr._pc_thread.ident is not None
    finally:
        mgr.close()
