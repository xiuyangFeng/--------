"""CPU-only queue regression tests. GPU queries and training commands are mocked."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from training_wss_min.tools import run_next_matrix_queue as queue


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    arms = [{"id": aid, "config": f"{aid}.json", "phase": phase}
            for aid, phase in (("B0", 1), ("A0", 0), ("C0", 2), ("A1", 0))]
    for arm in arms:
        (config_dir / arm["config"]).write_text(json.dumps({"name": f"run_{arm['id']}"}))
    (config_dir / "matrix.json").write_text(json.dumps({"arms": arms}))
    args = argparse.Namespace(config_dir=config_dir, matrix=None,
                              experiment_dir=tmp_path / "experiment", slots_per_gpu=2,
                              min_free_mib=7000, launch_interval_seconds=3,
                              poll_seconds=1, terminate_timeout_seconds=1)
    return args


@pytest.fixture
def runtime(monkeypatch):
    class Runtime:
        now = 0.0
        processes = []
        events = []
        spawn_error_at = None
        fail_stages = set()
        ignored_term = False

        def sleep(self, seconds):
            self.now += seconds
            assert self.now < 1000, "queue stalled"

        def popen(self, command, **kwargs):
            stage = "train" if "--config" in command else "eval_" + command[command.index("--checkpoint") + 1]
            aid = (Path(command[command.index("--config") + 1]).stem if stage == "train"
                   else Path(command[command.index("--run-dir") + 1]).name.removeprefix("run_"))
            if self.spawn_error_at == len(self.processes):
                raise OSError("injected spawn failure")
            process = Process(self, aid, stage, kwargs)
            self.processes.append(process)
            self.events.append(("launch", aid, stage, self.now, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            return process

        def killpg(self, pid, sig):
            process = next(p for p in self.processes if p.pid == pid)
            self.events.append(("signal", process.aid, process.stage, self.now, sig))
            if sig == signal.SIGKILL or not self.ignored_term:
                process.returncode = -sig

    class Process:
        def __init__(self, owner, aid, stage, kwargs):
            self.owner, self.aid, self.stage = owner, aid, stage
            self.pid = 10000 + len(owner.processes)
            self.stream = kwargs["stdout"]
            self.kwargs = kwargs
            self.started = owner.now
            self.returncode = None
            self.waits = 0

        def poll(self):
            if self.returncode is None and self.owner.now - self.started >= 10:
                self.returncode = 1 if (self.aid, self.stage) in self.owner.fail_stages else 0
            return self.returncode

        def wait(self, timeout=None):
            self.waits += 1
            if self.poll() is None:
                if timeout is not None:
                    raise subprocess.TimeoutExpired("mock", timeout)
                raise AssertionError("unbounded wait on live fake process")
            self.owner.events.append(("wait", self.aid, self.stage, self.owner.now, self.returncode))
            return self.returncode

    rt = Runtime()
    monkeypatch.setattr(queue.time, "monotonic", lambda: rt.now)
    monkeypatch.setattr(queue.time, "sleep", rt.sleep)
    monkeypatch.setattr(queue.subprocess, "Popen", rt.popen)
    monkeypatch.setattr(queue.os, "killpg", rt.killpg)
    monkeypatch.setattr(queue, "free_memory", lambda gpu: 16000)
    return rt


def test_json_state_all_arms_real_log_paths_and_success(sandbox, runtime, monkeypatch):
    snapshots = []
    write = queue.atomic_write_status

    def capture(path, state):
        snapshots.append(json.loads(json.dumps(state)))  # Original TextIOWrapper bug fails here.
        write(path, state)

    monkeypatch.setattr(queue, "atomic_write_status", capture)
    assert queue.MatrixQueue(sandbox).run() == 0
    assert set(snapshots[0]["arms"]) == {"A0", "A1", "B0", "C0"}
    assert all(r["status"] == "pending" for r in snapshots[0]["arms"].values())
    state = json.loads((sandbox.experiment_dir / "queue_status.json").read_text())
    assert state["status"] == "complete"
    for aid, record in state["arms"].items():
        assert list(record["stages"]) == ["train", "eval_best", "eval_last"]
        for stage, step in record["stages"].items():
            assert step["log"] == str(sandbox.experiment_dir / "logs" / f"{aid}_{stage}.log")
            assert "_fh" not in step
            assert step["returncode"] == 0
    assert len(runtime.processes) == 12
    assert all(p.waits and p.stream.closed and p.kwargs["start_new_session"] for p in runtime.processes)
    assert not (sandbox.experiment_dir / ".queue.lock").exists()


def test_arbitrary_phase_barrier_and_one_eval_per_gpu(sandbox, runtime):
    assert queue.MatrixQueue(sandbox).run() == 0
    events = runtime.events
    for later, earlier in (("B0", ("A0", "A1")), ("C0", ("B0",))):
        launched = next(i for i, e in enumerate(events) if e[:3] == ("launch", later, "train"))
        for aid in earlier:
            finished = next(i for i, e in enumerate(events) if e[:3] == ("wait", aid, "eval_last"))
            assert finished < launched
    active = {}
    for action, aid, stage, _, extra in events:
        if action == "launch":
            if stage.startswith("eval_"):
                assert not any(gpu == extra and s.startswith("eval_") for _, s, gpu in active.values())
            active[(aid, stage)] = (aid, stage, extra)
        elif action == "wait":
            active.pop((aid, stage), None)


def test_failed_lower_phase_blocks_combinations(sandbox, runtime):
    runtime.fail_stages = {("A0", "eval_best")}
    runner = queue.MatrixQueue(sandbox)
    assert runner.run() == 1
    assert runner.state["status"] == "failed"
    assert runner.state["arms"]["A0"]["failed_stage"] == "eval_best"
    assert runner.state["arms"]["A1"]["status"] == "complete"
    assert all(runner.state["arms"][aid]["status"] == "blocked" for aid in ("B0", "C0"))
    assert all(p.aid in {"A0", "A1"} for p in runtime.processes)


def test_warmup_interval_and_memory_gate(sandbox, runtime, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    sandbox.launch_interval_seconds = 6
    queries = []

    def memory(gpu):
        queries.append(runtime.now)
        return 0 if runtime.now < 5 else 20000

    monkeypatch.setattr(queue, "free_memory", memory)
    assert queue.MatrixQueue(sandbox).run() == 0
    launches = [e[3] for e in runtime.events if e[0] == "launch"]
    assert launches[0] >= 5
    assert all(b - a >= 6 for a, b in zip(launches, launches[1:]))
    assert len(queries) > 1


def test_spawn_failure_reaps_existing_children_and_records_all_arms(sandbox, runtime):
    runtime.spawn_error_at = 1
    runner = queue.MatrixQueue(sandbox)
    with pytest.raises(OSError, match="injected spawn failure"):
        runner.run()
    assert len(runtime.processes) == 1
    assert runtime.processes[0].waits and runtime.processes[0].stream.closed
    state = json.loads(runner.path.read_text())
    assert state["status"] == "failed"
    assert state["arms"]["A1"]["stages"]["train"]["error"].startswith("OSError:")
    assert all(r["status"] in queue.TERMINAL for r in state["arms"].values())
    assert not runner.active


@pytest.mark.parametrize("persistent", [False, True])
def test_status_failure_after_launch_reaps_and_preserves_exception(sandbox, runtime, monkeypatch, persistent):
    write = queue.atomic_write_status
    writes = 0

    def faulty_write(path, state):
        nonlocal writes
        writes += 1
        if writes == 2 or (persistent and writes > 2):
            raise TypeError("injected serialization failure")
        write(path, state)

    monkeypatch.setattr(queue, "atomic_write_status", faulty_write)
    runner = queue.MatrixQueue(sandbox)
    with pytest.raises(TypeError, match="injected serialization failure"):
        runner.run()
    assert len(runtime.processes) == 1
    assert runtime.processes[0].waits and runtime.processes[0].stream.closed
    assert runner.state["status"] == "failed"
    assert runner.state["arms"]["A0"]["status"] == "failed"
    assert not runner.active
    if not persistent:
        assert json.loads(runner.path.read_text())["status"] == "failed"
    assert not (sandbox.experiment_dir / ".queue.lock").exists()


def test_signal_terminates_all_owned_processes_and_escalates(sandbox, runtime, monkeypatch):
    runtime.ignored_term = True
    runner = queue.MatrixQueue(sandbox)
    sleep = runtime.sleep

    def stop_after_launch(seconds):
        sleep(seconds)
        runner.stop(signal.SIGTERM, None)

    monkeypatch.setattr(queue.time, "sleep", stop_after_launch)
    assert runner.run() == 1
    assert len(runtime.processes) == 2
    assert all(p.waits == 2 and p.returncode == -signal.SIGKILL and p.stream.closed for p in runtime.processes)
    assert runner.state["stop_signal"] == signal.SIGTERM
    assert all(r["status"] == "cancelled" for r in runner.state["arms"].values())
    assert not runner.active


@pytest.mark.parametrize("artifact", ["status", "run", "log", "lock"])
def test_existing_execution_artifacts_never_overwritten(sandbox, runtime, artifact):
    sandbox.experiment_dir.mkdir()
    if artifact == "status":
        target = sandbox.experiment_dir / "queue_status.json"
    elif artifact == "log":
        target = sandbox.experiment_dir / "logs" / "A0_train.log"
        target.parent.mkdir()
    elif artifact == "lock":
        target = sandbox.experiment_dir / ".queue.lock"
    else:
        target = queue.ROOT / "training_wss_min/runs/run_A0" / "checkpoint.pt"
        target.parent.mkdir(parents=True)
    target.write_text("historical evidence")
    with pytest.raises(FileExistsError):
        queue.MatrixQueue(sandbox).run()
    assert target.read_text() == "historical evidence"
    assert not runtime.processes
    if artifact != "status":
        assert not (sandbox.experiment_dir / "queue_status.json").exists()


@pytest.mark.parametrize("bad", ["duplicate_id", "duplicate_run", "negative_phase", "empty_matrix"])
def test_matrix_validation_before_status_creation(sandbox, runtime, bad):
    path = sandbox.config_dir / "matrix.json"
    matrix = json.loads(path.read_text())
    if bad == "duplicate_id":
        matrix["arms"].append(matrix["arms"][0])
    elif bad == "duplicate_run":
        (sandbox.config_dir / "A1.json").write_text(json.dumps({"name": "run_A0"}))
    elif bad == "negative_phase":
        matrix["arms"][0]["phase"] = -1
    else:
        matrix["arms"] = []
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError):
        queue.MatrixQueue(sandbox).run()
    assert not sandbox.experiment_dir.exists()
    assert not runtime.processes


@pytest.mark.parametrize("job,devices,count", [("", "0", "1"), ("abc", "0", "1"),
    ("123", "", "1"), ("123", "0,0", "2"), ("123", "-1", "1"), ("123", "0,1", "1")])
def test_slurm_checks(sandbox, runtime, monkeypatch, job, devices, count):
    monkeypatch.setenv("SLURM_JOB_ID", job)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", devices)
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", count)
    with pytest.raises(RuntimeError):
        queue.MatrixQueue(sandbox)
    assert not runtime.processes


def test_gpu_query_failure_fails_before_any_child(sandbox, runtime, monkeypatch):
    def fail(gpu):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(queue, "free_memory", fail)
    runner = queue.MatrixQueue(sandbox)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run()
    assert not runtime.processes
    assert json.loads(runner.path.read_text())["status"] == "failed"


def test_atomic_serialization_failure_preserves_last_valid_status(tmp_path):
    path = tmp_path / "queue_status.json"
    queue.atomic_write_status(path, {"status": "valid"})
    with (tmp_path / "log").open("w") as handle:
        with pytest.raises(TypeError):
            queue.atomic_write_status(path, {"_fh": handle})
    assert json.loads(path.read_text()) == {"status": "valid"}
    assert not list(tmp_path.glob(".queue_status.json.*"))


def test_atomic_replace_failure_preserves_status_and_removes_temp(tmp_path, monkeypatch):
    path = tmp_path / "queue_status.json"
    queue.atomic_write_status(path, {"status": "valid"})

    def fail(source, dest):
        raise OSError("replace failed")

    monkeypatch.setattr(queue.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        queue.atomic_write_status(path, {"status": "new"})
    assert json.loads(path.read_text()) == {"status": "valid"}
    assert not list(tmp_path.glob(".queue_status.json.*"))


def test_real_cpu_child_reaped_after_first_launch_save_failure(sandbox, monkeypatch):
    """Real short-lived CPU sleeper verifies killpg/wait, never imports training or CUDA."""
    actual_popen = subprocess.Popen
    children = []
    monkeypatch.setattr(queue, "free_memory", lambda gpu: 16000)
    write = queue.atomic_write_status
    writes = 0

    def popen(_command, **kwargs):
        process = actual_popen([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)
        children.append((process, kwargs["stdout"]))
        return process

    def fault(path, state):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise TypeError("first launch status failure")
        write(path, state)

    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    monkeypatch.setattr(queue, "atomic_write_status", fault)
    runner = queue.MatrixQueue(sandbox)
    try:
        with pytest.raises(TypeError, match="first launch status failure"):
            runner.run()
        assert len(children) == 1
        child, log = children[0]
        assert child.returncode is not None and log.closed
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)
        assert json.loads(runner.path.read_text())["status"] == "failed"
    finally:
        for child, _ in children:
            if child.poll() is None:
                child.kill()
            child.wait()


def test_nested_run_names_preserve_existing_config_protocol(sandbox):
    path = sandbox.config_dir / "A0.json"
    path.write_text(json.dumps({"name": "matrix_20260911/group/A0"}))
    records = queue.read_arms(sandbox.config_dir, sandbox.config_dir / "matrix.json")
    assert records["A0"]["run_name"] == "matrix_20260911/group/A0"


def test_current_matrix_config_names_are_readable(sandbox):
    config_dir = Path(queue.__file__).resolve().parents[1] / "configs/wss_direct_recovery_20260912"
    records = queue.read_arms(config_dir, config_dir / "matrix.json")
    source = json.loads((config_dir / "matrix.json").read_text())["arms"]
    assert set(records) == {a["id"] for a in source}
    assert all("/" in r["run_name"] for r in records.values())


@pytest.mark.parametrize("name", ["../escape", "/absolute", "matrix/../escape", "matrix/./A0",
                                  "matrix//A0", "matrix/A0/", "", "matrix/../../outside"])
def test_unsafe_run_paths_rejected(sandbox, name):
    (sandbox.config_dir / "A0.json").write_text(json.dumps({"name": name}))
    with pytest.raises(ValueError, match="invalid run name"):
        queue.read_arms(sandbox.config_dir, sandbox.config_dir / "matrix.json")


def test_nested_run_symlink_escape_rejected(sandbox):
    run_root = queue.ROOT / "training_wss_min/runs"
    run_root.mkdir(parents=True)
    (run_root / "matrix").symlink_to(sandbox.config_dir, target_is_directory=True)
    (sandbox.config_dir / "A0.json").write_text(json.dumps({"name": "matrix/A0"}))
    with pytest.raises(ValueError, match="escapes runs directory"):
        queue.read_arms(sandbox.config_dir, sandbox.config_dir / "matrix.json")
