"""§19.6: `cli service` against real loopback sandbox services (random high ports, temporary jobs roots, CPU only).

Covers start → status → restart → stop, adoption of a hand-started process without a PID file (argv, cwd,
WSS_DEPLOY_* / CUDA_VISIBLE_DEVICES / TZ and the token are carried over), the rotating log with a UTC offset,
the PID-reuse guard, and `cli submit` / `cli jobs list` over HTTP.  Every process started here is killed at the end.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

import wss_deploy
from wss_deploy import service as S
from wss_deploy.cli import main as cli_main
from wss_deploy.paths import PROJECT_ROOT

# Checked-in copy of LV_GUO_YOU's input STL (tests/fixtures/<job>/input.stl, shared with test_report_freshness);
# WSS_DEPLOY_TEST_FIXTURES may point at another fixtures root holding the same job id.
FIXTURES = Path(os.environ.get("WSS_DEPLOY_TEST_FIXTURES") or Path(__file__).resolve().parent / "fixtures")
SOURCE_STL = FIXTURES / "20260920_173929_a9ec139d6cdd" / "input.stl"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def sandbox(tmp_path):
    """Temporary jobs root; afterwards every serve process of this root (or cwd) is killed."""
    root = tmp_path / "jobs"
    root.mkdir()
    yield root
    for info in S.scan_processes():
        if str(info["jobs_root"]).startswith(str(tmp_path)) or (info.get("cwd") or "").startswith(str(tmp_path)):
            try:
                os.killpg(info["pid"], signal.SIGKILL)
            except OSError:
                try:
                    os.kill(info["pid"], signal.SIGKILL)
                except OSError:
                    pass


def _clean_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not (k.startswith("WSS_DEPLOY_") or k in ("CUDA_VISIBLE_DEVICES", "TZ"))}
    env.update(PYTHONPATH=str(PROJECT_ROOT), **extra)
    return env


def _wait_health(port: int, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        health = S.probe_health("127.0.0.1", port, timeout=2)
        if health and health.get("ok"):
            return health
        time.sleep(0.3)
    raise AssertionError(f"service on {port} did not become healthy")


def test_start_status_restart_stop(sandbox, capsys):
    port = _free_port()
    lines: list[str] = []
    record = S.start(sandbox, host="127.0.0.1", port=port, device="cpu", env_items=["CUDA_VISIBLE_DEVICES=", "TZ=Asia/Shanghai"],
                     timeout=90, out=lines.append)
    try:
        assert record["health"]["version"] == wss_deploy.__version__ and "服务已就绪" in lines[-1]
        pid_file = json.loads((sandbox / S.PID_FILE).read_text(encoding="utf-8"))
        assert pid_file["pid"] == record["pid"] and pid_file["port"] == port and pid_file["version"] == wss_deploy.__version__
        assert set(pid_file) >= {"pid", "started_at", "host", "port", "argv", "version"}
        config = json.loads((sandbox / S.CONFIG_FILE).read_text(encoding="utf-8"))
        assert config["env"] == {"CUDA_VISIBLE_DEVICES": "", "TZ": "Asia/Shanghai"} and config["device"] == "cpu"
        info = S.status(sandbox, port)
        assert info["running"] and info["managed"] and info["healthy"] and info["pid"] == record["pid"] and info["version"] == wss_deploy.__version__
        assert info["env"] == {"CUDA_VISIBLE_DEVICES": "", "TZ": "Asia/Shanghai"} and info["gpu"]["available"] is False
        assert set(info["queue"]) == {"running", "queued", "awaiting_input", "awaiting_confirmation"}
        text = S.format_status(info)
        assert "由 service 管理" in text and f":{port}/" in text and "PID" in text and "运行时长" in text
        assert cli_main(["service", "status", "--jobs-root", str(sandbox), "--port", str(port)]) == 0
        assert "由 service 管理" in capsys.readouterr().out
        assert cli_main(["service", "status", "--jobs-root", str(sandbox), "--port", str(port), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["pid"] == record["pid"]
        log = (sandbox / S.LOG_FILE).read_text(encoding="utf-8")
        assert re.search(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+0800 INFO wss_deploy\.service Serving wss_deploy " + re.escape(wss_deploy.__version__), log, re.M)
        assert cli_main(["service", "logs", "--jobs-root", str(sandbox), "-n", "5"]) == 0 and "Serving" in capsys.readouterr().out
        # a second start is refused
        with pytest.raises(S.ServiceError):
            S.start(sandbox, timeout=10, out=lines.append)
        restarted = S.restart(sandbox, port, timeout=90, out=lines.append)
        assert restarted["pid"] != record["pid"] and not S._alive(record["pid"])
        assert S.process_env(restarted["pid"])["TZ"] == "Asia/Shanghai" and S.process_env(restarted["pid"])["CUDA_VISIBLE_DEVICES"] == ""
        assert S.status(sandbox, port)["pid"] == restarted["pid"]
        assert "Service stopping" in (sandbox / S.LOG_FILE).read_text(encoding="utf-8")   # SIGTERM took the clean path
        stopped = S.stop(sandbox, port, out=lines.append)
        assert stopped["stopped"] and not stopped["killed"] and not (sandbox / S.PID_FILE).exists()
        assert cli_main(["service", "status", "--jobs-root", str(sandbox), "--port", str(port)]) == 3
        assert "未运行" in capsys.readouterr().out
        assert S.stop(sandbox, port, out=lines.append)["stopped"] is False
    finally:
        if S._alive(record["pid"]):
            os.kill(record["pid"], signal.SIGKILL)


def test_hand_started_process_is_found_and_adopted_by_restart(sandbox, tmp_path):
    port = _free_port()
    env = _clean_env(CUDA_VISIBLE_DEVICES="", TZ="Asia/Shanghai", WSS_DEPLOY_LEGACY_OWNER="owner-x",
                     WSS_DEPLOY_TOKEN="manual-token-123", WSS_DEPLOY_AUTO_REFRESH_REPORTS="0")
    with open(tmp_path / "manual.log", "ab") as log:
        manual = subprocess.Popen([sys.executable, "-u", "-m", "wss_deploy.cli", "serve", "--host", "127.0.0.1", "--port", str(port),
                                   "--jobs-root", "jobs", "--device", "cpu"], cwd=str(tmp_path), env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    try:
        _wait_health(port)
        assert not (sandbox / S.PID_FILE).exists() and not (sandbox / S.CONFIG_FILE).exists()
        info = S.status(sandbox, port)
        assert info["running"] and info["managed"] is False and info["pid"] == manual.pid and info["root_matches"]
        assert info["env"] == {"CUDA_VISIBLE_DEVICES": "", "TZ": "Asia/Shanghai", "WSS_DEPLOY_AUTO_REFRESH_REPORTS": "0",
                               "WSS_DEPLOY_LEGACY_OWNER": "owner-x"}
        assert info["token_in_env"] is True and info["log_file"] is None
        assert "手工启动" in S.format_status(info) and "manual-token-123" not in S.format_status(info)
        # status is read-only: nothing was written
        assert not (sandbox / S.CONFIG_FILE).exists() and not (sandbox / S.TOKEN_FILE).exists()
        lines: list[str] = []
        restarted = S.restart(sandbox, port, timeout=90, out=lines.append)
        assert manual.wait(timeout=30) is not None and restarted["pid"] != manual.pid
        assert any("已接管" in line for line in lines) and any("令牌已从运行中进程" in line for line in lines)
        config = S.load_config(sandbox)
        assert config["env"] == info["env"] and config["cwd"] == str(tmp_path) and config["python"] == sys.executable
        assert config["device"] == "cpu" and config["port"] == port and config["adopted_from"]["pid"] == manual.pid
        assert "manual-token-123" not in (sandbox / S.CONFIG_FILE).read_text(encoding="utf-8")
        assert S.read_token(sandbox) == "manual-token-123"
        assert stat.S_IMODE((sandbox / S.TOKEN_FILE).stat().st_mode) == 0o600
        new_env = S.process_env(restarted["pid"])
        assert {k: new_env.get(k) for k in info["env"]} == info["env"]
        assert "WSS_DEPLOY_TOKEN" not in new_env           # loopback: the token is only injected for shared hosts
        assert os.readlink(f"/proc/{restarted['pid']}/cwd") == str(tmp_path)
        after = S.status(sandbox, port)
        assert after["managed"] and after["healthy"] and after["log_file"] == str(sandbox / S.LOG_FILE)
        S.stop(sandbox, port, out=lines.append)
        assert not S._alive(restarted["pid"])
    finally:
        if manual.poll() is None:
            manual.kill()


def test_guards_never_touch_unrelated_processes(sandbox):
    assert S.serve_options(["/bin/bash", "-c", "python -m wss_deploy.cli serve --port 8765"]) is None
    assert S.serve_options(["python", "-c", "print('-m wss_deploy.cli serve')"]) is None
    opts = S.serve_options(["/env/bin/python", "-u", "-m", "wss_deploy.cli", "serve", "--port=9000", "--host", "0.0.0.0", "--token", "t"])
    assert opts["port"] == 9000 and opts["host"] == "0.0.0.0" and opts["token"] == "t" and opts["python"] == "/env/bin/python"
    assert S.serve_options(["python", "wss_deploy/cli.py", "serve"])["port"] == 8765
    sleeper = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        (sandbox / S.PID_FILE).write_text(json.dumps({"pid": sleeper.pid, "port": 8765}))
        messages: list[str] = []
        assert S.stop(sandbox, _free_port(), out=messages.append)["stopped"] is False
        assert sleeper.poll() is None and "没有在运行" in messages[-1]
        assert S.locate(sandbox, _free_port()) is None
    finally:
        sleeper.kill(); sleeper.wait()


def test_shared_start_uses_the_token_file_and_config_rules(sandbox):
    cfg = S.load_config(sandbox)
    cfg.update(host="0.0.0.0", env={"TZ": "Asia/Shanghai"})
    env = S._child_env(cfg, sandbox)
    token = S.read_token(sandbox)
    assert token and env["WSS_DEPLOY_TOKEN"] == token and stat.S_IMODE((sandbox / S.TOKEN_FILE).stat().st_mode) == 0o600
    assert S._child_env(cfg, sandbox)["WSS_DEPLOY_TOKEN"] == token        # generated once, then reused
    assert env["TZ"] == "Asia/Shanghai" and env["PYTHONPATH"].split(os.pathsep)[0] == str(PROJECT_ROOT)
    with pytest.raises(S.ServiceError):
        S.apply_overrides(cfg, env_items=["WSS_DEPLOY_TOKEN=x"])
    with pytest.raises(S.ServiceError):
        S.apply_overrides(cfg, env_items=["HOME=/tmp"])
    assert S.apply_overrides(cfg, env_items=["CUDA_VISIBLE_DEVICES=1"], unset_env=["TZ"])["env"] == {"CUDA_VISIBLE_DEVICES": "1"}
    assert S.token(sandbox, out=lambda *_a, **_k: None) == 0


def test_cli_submit_and_jobs_list_against_a_sandbox_service(sandbox, tmp_path, capsys):
    if not SOURCE_STL.is_file():
        pytest.fail(f"test fixture missing: {SOURCE_STL} (restore tests/fixtures/ or set WSS_DEPLOY_TEST_FIXTURES)", pytrace=False)
    port = _free_port()
    stl = tmp_path / "LV_copy.stl"
    shutil.copy2(SOURCE_STL, stl)
    # /bin/false as the VMTK interpreter: stage A starts for real but stops at the centreline without spawning VMTK.
    record = S.start(sandbox, host="127.0.0.1", port=port, device="cpu", timeout=90, out=lambda *_: None,
                     env_items=["CUDA_VISIBLE_DEVICES=", "WSS_DEPLOY_VMTK_PYTHON=/bin/false", "WSS_DEPLOY_LEGACY_OWNER=cli-owner"])
    url = f"http://127.0.0.1:{port}"
    try:
        code = cli_main(["submit", str(stl), "--server", url, "--patient-id", "P-9", "--scan-date", "2026-01-02", "--tags", "AAA,随访",
                         "--units", "mm", "--scan-label", "基线"])
        out = capsys.readouterr().out
        assert code == 0 and "✓ LV_copy.stl → 任务" in out and f"{url}/#job=" in out and "新建 1" in out
        job_id = re.search(r"#job=([A-Za-z0-9_-]+)", out)[1]
        assert cli_main(["jobs", "list", "--server", url, "--json"]) == 0
        jobs = json.loads(capsys.readouterr().out)
        job = next(item for item in jobs if item["id"] == job_id)
        assert job["patient_id"] == "P-9" and job["scan_date"] == "2026-01-02" and job["tags"] == ["AAA", "随访"] and job["stage"] == "A"
        assert job["status"] in {"queued", "running", "failed"} and job["family_label"] == "壁面 WSS"
        deadline = time.time() + 60
        while time.time() < deadline:
            client = S.ServiceClient(url); client.connect()
            state = next(item for item in client.jobs()["jobs"] if item["id"] == job_id)
            if state["status"] == "failed":
                break
            time.sleep(0.5)
        assert state["status"] == "failed" and state["stage"] == "A"          # stage A really ran (ingest passed, centreline refused)
        assert cli_main(["jobs", "list", "--server", url]) == 0
        assert job_id in capsys.readouterr().out
        code = cli_main(["submit", str(stl), "--server", url, "--units", "mm"])
        assert code == 2 and "同一几何已有 1 个任务" in capsys.readouterr().out
        with pytest.raises(SystemExit):
            cli_main(["submit", str(tmp_path / "missing.stl"), "--server", url])
    finally:
        S.stop(sandbox, port, out=lambda *_: None)
        if S._alive(record["pid"]):
            os.kill(record["pid"], signal.SIGKILL)
