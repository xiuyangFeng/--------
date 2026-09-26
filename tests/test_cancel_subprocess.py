"""N1 (§15.5): a cancelled job kills the whole vessel_geom process group, and the attempt clock is reported.

The real VMTK command is replaced by a harmless child that itself spawns a grandchild, so the test
proves the *group* is terminated rather than only the direct child.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from wss_deploy import centerline as CL
from tests._c_helpers import manager

# Writes its own pid and its grandchild's pid, then sleeps far longer than any assertion window.
SPAWNER = (
    "import os, subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
    "open(sys.argv[1], 'w').write(f'{os.getpid()} {child.pid}')\n"
    "sys.stdout.flush()\n"
    "time.sleep(30)\n"
)


def _pids(path, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            parts = path.read_text(encoding="utf-8").split()
            if len(parts) == 2:
                return [int(value) for value in parts]
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError("child never reported its pids")


def _dead(pid, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return True
        time.sleep(0.05)
    return False


def _fake_command(monkeypatch, tmp_path, script, *args):
    monkeypatch.setattr(CL, "VESSEL_GEOM_DIR", tmp_path)
    monkeypatch.setattr(CL, "vessel_geom_command",
                        lambda *_a, **_k: [sys.executable, "-c", script, *[str(a) for a in args]])


def test_cancelled_run_kills_the_process_group(tmp_path, monkeypatch):
    pidfile = tmp_path / "pids.txt"
    _fake_command(monkeypatch, tmp_path, SPAWNER, pidfile)
    out_dir = tmp_path / "centerline"
    started = time.time()
    cancel_at = started + 0.2
    with pytest.raises(InterruptedError) as error:
        CL.run_vessel_geom(tmp_path / "input.stl", out_dir, cancelled=lambda: time.time() >= cancel_at)
    assert "任务已取消" in str(error.value)
    assert time.time() - started < 3.0
    parent_pid, grandchild_pid = _pids(pidfile)
    assert _dead(parent_pid) and _dead(grandchild_pid)
    # The transcript is still written so the operator can see how far the run got.
    assert (out_dir / "vessel_geom.stdout.txt").is_file()
    assert not list(out_dir.glob("*.part"))


def test_timeout_kills_the_group_and_raises_a_typed_toolchain_error(tmp_path, monkeypatch):
    from wss_deploy.errors import ToolchainError
    pidfile = tmp_path / "pids.txt"
    _fake_command(monkeypatch, tmp_path, SPAWNER, pidfile)
    with pytest.raises(ToolchainError) as error:
        CL.run_vessel_geom(tmp_path / "input.stl", tmp_path / "centerline", timeout=1)
    parent_pid, grandchild_pid = _pids(pidfile)
    assert _dead(parent_pid) and _dead(grandchild_pid)
    # v0.14: a short Chinese message for the user, the tool details for admins; not retryable.
    assert "超时" in error.value.user_message and "timeout" in error.value.admin_detail
    assert error.value.category == "toolchain" and error.value.retryable is False


def test_successful_run_reports_topology_and_keeps_the_transcript(tmp_path, monkeypatch):
    script = (
        "import json, os, sys\n"
        "out = sys.argv[1]\n"
        "os.makedirs(os.path.join(out, 'centerline'), exist_ok=True)\n"
        "json.dump({'extraction': {'hard_pass': True, 'selected_attempt': 2}, 'graph_topology': {'segments': 7}},\n"
        "          open(os.path.join(out, 'run.json'), 'w'))\n"
        "json.dump({'surface_report': {'openings': [{'opening_id': 0}], 'area_mm2': 12.5}},\n"
        "          open(os.path.join(out, 'centerline', 'result.json'), 'w'))\n"
        "print('stdout line')\n"
        "print('stderr line', file=sys.stderr)\n"
    )
    out_dir = tmp_path / "centerline"
    _fake_command(monkeypatch, tmp_path, script, out_dir)
    result = CL.run_vessel_geom(tmp_path / "input.stl", out_dir, cancelled=lambda: False)
    assert result["hard_pass"] is True and result["attempt"] == 2 and result["topology"] == {"segments": 7}
    assert result["openings"] == [{"opening_id": 0}] and result["surface_report"] == {"area_mm2": 12.5}
    transcript = (out_dir / "vessel_geom.stdout.txt").read_text(encoding="utf-8")
    assert "stdout line" in transcript and "--- stderr ---" in transcript and "stderr line" in transcript
    assert not list(out_dir.glob("*.part"))


def test_failed_run_still_reports_the_stderr_tail(tmp_path, monkeypatch):
    from wss_deploy.errors import ToolchainError
    _fake_command(monkeypatch, tmp_path, "import sys; print('boom detail', file=sys.stderr); sys.exit(3)")
    with pytest.raises(ToolchainError) as error:
        CL.run_vessel_geom(tmp_path / "input.stl", tmp_path / "centerline")
    # v0.14: the stderr tail moved to admin_detail; the user message stays free of tool output.
    assert "rc=3" in error.value.admin_detail and "boom detail" in error.value.admin_detail
    assert "boom detail" not in error.value.user_message and "中心线提取失败" in error.value.user_message
    assert error.value.to_record(diagnostic_id="x")["category"] == "toolchain"


def test_missing_outputs_after_success_are_a_toolchain_error(tmp_path, monkeypatch):
    from wss_deploy.errors import ToolchainError
    _fake_command(monkeypatch, tmp_path, "print('no outputs written')")
    with pytest.raises(ToolchainError) as error:
        CL.run_vessel_geom(tmp_path / "input.stl", tmp_path / "centerline")
    assert "run.json" in error.value.admin_detail or "No such file" in error.value.admin_detail


def test_attempt_seconds_reset_per_attempt_and_abort_event(tmp_path):
    """``compute_seconds`` accumulates over retries; ``attempt_seconds`` is only the current attempt."""
    state = {"cancel": True}

    def stage_a(*_args, cancelled=None, **_kwargs):
        time.sleep(0.05)
        if state["cancel"]:
            raise InterruptedError("任务已取消")
        return {"stage": "A", "input_check": {"status": "fail", "errors": ["stop here"]},
                "centerline": {"hard_pass": False}, "proposal": {}}

    mgr = manager(tmp_path, stage_a=stage_a)
    job = mgr.create("owner", content=b"solid", filename="c.stl", case_id="C")
    assert mgr.run_next(timeout=1)
    record = mgr.get(job["id"], "owner")
    assert record["status"] == "cancelled"
    aborted = [event for event in record["events"] if event["action"] == "attempt_aborted"]
    assert len(aborted) == 1 and aborted[0]["attempt_s"] >= 0
    first_attempt = record["timing"]["attempt_s"]
    assert record["timing"]["compute_s"] == pytest.approx(first_attempt, abs=0.05)

    state["cancel"] = False
    mgr.retry(job["id"], "owner", {"version": record["version"]})
    assert mgr.jobs[job["id"]]["attempt_seconds"] == 0.0  # reset at enqueue, before the worker starts
    assert mgr.run_next(timeout=1)
    record = mgr.get(job["id"], "owner")
    assert record["status"] == "failed"
    assert record["timing"]["compute_s"] >= record["timing"]["attempt_s"]
    assert len([event for event in record["events"] if event["action"] == "attempt_aborted"]) == 1
    mgr.close()


def test_cancelling_a_queued_job_keeps_the_historical_event_stream(tmp_path):
    """Nothing ran, so there is no aborted attempt: only the worker writes ``attempt_aborted``."""
    mgr = manager(tmp_path)
    job = mgr.create("owner", content=b"solid", filename="c.stl", case_id="C")
    snapshot = mgr.cancel(job["id"], "owner", {"version": job["version"]})
    assert snapshot["status"] == "cancelled" and snapshot["timing"]["attempt_s"] == 0.0
    actions = [event["action"] for event in mgr.get(job["id"], "owner")["events"]]
    assert actions == ["created", "cancel_requested"]
    mgr.close()
