"""Atomic queue transactions, safe retries, stage handoffs and admission gates."""
from __future__ import annotations

import copy
import json
import multiprocessing
from pathlib import Path
import signal

import pytest

from training_wss_min.tools import m2_optimization_common as common
from training_wss_min.tools import run_m2_optimization as queue


def _claim_process(experiment_path, owner):
    # Real independently opened flock descriptors, not a shared in-process mutex.
    common.EXP = Path(experiment_path)
    while True:
        with common.locked_state() as state:
            pending = next((arm for arm in state["arms"].values() if arm["status"] == "pending"), None)
            if pending is None:
                return
            pending.update(status="claimed", worker=owner)
            state["claims"].append(pending["id"])
            state["transactions"] += 1


def _arm(aid="A", status="pending", stage="train"):
    return dict(id=aid, status=status, next_stage=stage, failed_stage=stage,
                run_name=f"experiment/{aid}", config=f"{aid}.json", stages={}, attempts=[],
                min_free_mib=7000.)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    exp, runs, configs = tmp_path / "experiment", tmp_path / "runs", tmp_path / "configs"
    exp.mkdir(); configs.mkdir()
    source = tmp_path / "source.py"
    source.write_text("frozen_source = True\n")
    hashes = {"source.py": common.sha(source)}
    monkeypatch.setattr(common, "EXP", exp)
    monkeypatch.setattr(queue, "EXP", exp)
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "RUNS", runs)
    monkeypatch.setattr(queue, "CONFIGS", configs)
    monkeypatch.setattr(queue, "fingerprints", lambda phase="base": copy.deepcopy(hashes))
    monkeypatch.setattr(queue, "manifest_path", lambda phase="base": configs / f"{phase}.json")
    monkeypatch.setenv("SLURM_JOB_ID", "unit-worker")
    monkeypatch.setattr(queue, "visible_gpu", lambda: "0")
    return dict(exp=exp, runs=runs, configs=configs, hashes=hashes)


def _save_state(sandbox, arms, workers=None):
    state = dict(phase="base", status="running", source_sha256=sandbox["hashes"],
                 arms={arm["id"]: arm for arm in arms}, workers=workers or {})
    common.save_json(common.state_path(), state)
    return state


def _read_state():
    return json.loads(common.state_path().read_text())


def test_multiple_processes_claim_each_arm_once_without_lost_updates(sandbox):
    state = _save_state(sandbox, [_arm(f"A{i:03}") for i in range(80)])
    state.update(claims=[], transactions=0)
    common.save_json(common.state_path(), state)
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_claim_process, args=(str(sandbox["exp"]), str(i))) for i in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("concurrent state lock did not make progress")
        assert process.exitcode == 0
    final = _read_state()
    assert final["transactions"] == 80
    assert len(final["claims"]) == len(set(final["claims"])) == 80
    assert all(arm["status"] == "claimed" for arm in final["arms"].values())


def test_atomic_json_failure_and_aborted_transaction_preserve_last_valid_state(sandbox):
    _save_state(sandbox, [_arm()])
    old = common.state_path().read_bytes()
    with pytest.raises(ValueError):
        common.save_json(common.state_path(), {"bad": float("nan")})
    assert common.state_path().read_bytes() == old
    assert not list(sandbox["exp"].glob("*.tmp"))
    with pytest.raises(RuntimeError):
        with common.locked_state() as state:
            state["arms"]["A"]["status"] = "complete"
            raise RuntimeError("abort")
    assert common.state_path().read_bytes() == old


@pytest.mark.parametrize("peak,expected", [(1000., 7000.), (6000., 9548.), (12000., 17048.)])
def test_memory_admission_uses_peak_multiplier_and_absolute_floor(peak, expected):
    assert queue.required_memory({"max_memory_reserved_mib": peak}) == expected


@pytest.mark.parametrize("peak", [0., -1., float("nan"), float("inf")])
def test_memory_admission_rejects_missing_or_invalid_measurements(peak):
    with pytest.raises(ValueError):
        queue.required_memory({"max_memory_reserved_mib": peak})


def test_retry_prevalidates_all_arms_before_moving_any_artifacts(sandbox):
    _save_state(sandbox, [_arm("bad", "failed"), _arm("done", "complete")])
    failed = sandbox["runs"] / "experiment/bad"
    failed.mkdir(parents=True)
    marker = failed / "evidence.txt"
    marker.write_text("failed training evidence")
    before = common.state_path().read_bytes()
    with pytest.raises(RuntimeError):
        queue.retry("base", ["bad", "done"])
    assert marker.read_text() == "failed training evidence"
    assert common.state_path().read_bytes() == before
    assert not (sandbox["exp"] / "failed_attempts").exists()


def test_retrying_only_failed_eval_preserves_successful_training_and_best_eval(sandbox):
    arm = _arm("A", "failed", "eval_last")
    arm["stages"] = {"train": {"returncode": 0, "log": "train.log"},
                     "eval_best": {"returncode": 0, "log": "best.log"},
                     "eval_last": {"returncode": 1, "log": "failed-last.log"}}
    arm["attempts"] = [dict(stage="eval_last", returncode=1, log="failed-last.log")]
    _save_state(sandbox, [arm])
    run = sandbox["runs"] / arm["run_name"]
    run.mkdir(parents=True)
    (run / "ckpt_best.pt").write_bytes(b"successful checkpoint")
    queue.retry("base", ["A"])
    final = _read_state()["arms"]["A"]
    assert final["status"] == "pending" and final["next_stage"] == "eval_last"
    assert final["stages"]["train"] == arm["stages"]["train"]
    assert final["stages"]["eval_best"] == arm["stages"]["eval_best"]
    assert final["attempts"] == arm["attempts"]
    assert (run / "ckpt_best.pt").read_bytes() == b"successful checkpoint"


def test_failed_training_retry_archives_evidence_before_starting_clean_run(sandbox):
    arm = _arm("A", "failed", "train")
    arm["stages"] = {"train": {"returncode": 1, "log": "failed.log"}}
    arm["attempts"] = [dict(stage="train", returncode=1, log="failed.log")]
    _save_state(sandbox, [arm])
    run = sandbox["runs"] / arm["run_name"]
    run.mkdir(parents=True)
    (run / "history.jsonl").write_text("partial history")
    queue.retry("base", ["A"])
    final = _read_state()["arms"]["A"]
    assert not run.exists()
    assert (Path(final["failed_run_archives"][0]) / "history.jsonl").read_text() == "partial history"
    assert final["stages"] == {} and final["attempts"] == arm["attempts"]
    assert final["status"] == "pending" and final["next_stage"] == "train"


def test_retry_archive_io_failure_keeps_every_successful_move_in_state(sandbox, monkeypatch):
    _save_state(sandbox, [_arm("A", "failed"), _arm("B", "failed")])
    for aid in ("A", "B"):
        run = sandbox["runs"] / f"experiment/{aid}"
        run.mkdir(parents=True)
        (run / "evidence.txt").write_text(aid)
    original_move = queue.shutil.move
    calls = [0]

    def move_once(source, destination):
        calls[0] += 1
        if calls[0] == 2:
            raise OSError("simulated second archive failure")
        return original_move(source, destination)

    monkeypatch.setattr(queue.shutil, "move", move_once)
    with pytest.raises(OSError, match="second archive failure"):
        queue.retry("base", ["A", "B"])
    state = _read_state()
    archive = Path(state["arms"]["A"]["failed_run_archives"][0])
    assert (archive / "evidence.txt").read_text() == "A"
    assert (sandbox["runs"] / "experiment/B/evidence.txt").read_text() == "B"


def test_retry_refuses_running_workers_or_changed_sources(sandbox):
    _save_state(sandbox, [_arm("A", "failed")], workers={"other": {"status": "running"}})
    with pytest.raises(RuntimeError, match="workers have ended"):
        queue.retry("base", ["A"])
    with common.locked_state() as state:
        state["workers"] = {}
        state["source_sha256"] = {"source.py": "changed"}
    with pytest.raises(RuntimeError, match="same source"):
        queue.retry("base", ["A"])


class _FakeProcess:
    def __init__(self, pid, returncode=0):
        self.pid, self.returncode = pid, returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = -signal.SIGTERM
        return self.returncode


def _fake_worker_environment(monkeypatch, immediate=True):
    processes, commands, handlers = [], [], {}

    def popen(command, **kwargs):
        commands.append(command)
        process = _FakeProcess(10000 + len(commands), 0 if immediate else None)
        processes.append(process)
        return process

    clock_value = [1000.]

    def now():
        clock_value[0] += 20
        return clock_value[0]

    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    monkeypatch.setattr(queue.signal, "signal", lambda signum, handler: handlers.__setitem__(signum, handler))
    monkeypatch.setattr(queue, "free_memory", lambda gpu: 24000.)
    monkeypatch.setattr(queue.time, "time", now)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(queue.os, "killpg", lambda pid, signum: None)
    return processes, commands, handlers


def test_worker_hands_off_train_best_last_and_preserves_all_attempts(sandbox, monkeypatch):
    _save_state(sandbox, [_arm()])
    _, commands, _ = _fake_worker_environment(monkeypatch)
    assert queue.worker("base", 2) == 0
    final = _read_state()["arms"]["A"]
    assert final["status"] == "complete"
    assert [attempt["stage"] for attempt in final["attempts"]] == ["train", "eval_best", "eval_last"]
    assert all(attempt["returncode"] == 0 for attempt in final["attempts"])
    assert len(commands) == 3
    assert commands[0][3] == "training_wss_min.train"
    assert [command[command.index("--checkpoint") + 1] for command in commands[1:]] == ["best", "last"]
    assert queue.finalize("base") == 0
    assert _read_state()["status"] == "complete"


def test_interrupted_worker_keeps_actual_stage_attempt_for_retry(sandbox, monkeypatch):
    _save_state(sandbox, [_arm()])
    _, commands, handlers = _fake_worker_environment(monkeypatch, immediate=False)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: handlers[signal.SIGTERM](signal.SIGTERM, None))
    assert queue.worker("base", 1) == 1
    arm = _read_state()["arms"]["A"]
    assert len(commands) == 1 and arm["status"] == "interrupted"
    assert len(arm["attempts"]) == 1
    attempt = arm["attempts"][0]
    assert attempt["stage"] == "train" and attempt["returncode"] == -signal.SIGTERM
    assert attempt["ended_at"] and attempt["elapsed_seconds"] >= 0
    assert Path(attempt["log"]).exists()


def test_worker_does_not_claim_an_arm_without_enough_memory(sandbox, monkeypatch):
    _save_state(sandbox, [_arm()])
    _, commands, handlers = _fake_worker_environment(monkeypatch)
    monkeypatch.setattr(queue, "free_memory", lambda gpu: 6999.)
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: handlers[signal.SIGTERM](signal.SIGTERM, None))
    assert queue.worker("base", 1) == 1
    assert commands == []
    assert _read_state()["arms"]["A"]["status"] == "pending"


def test_failed_training_never_advances_to_evaluation_and_cannot_finalize(sandbox, monkeypatch):
    _save_state(sandbox, [_arm()])
    _, commands, _ = _fake_worker_environment(monkeypatch)

    def failed_process(command, **kwargs):
        commands.append(command)
        return _FakeProcess(10001, 17)

    monkeypatch.setattr(queue.subprocess, "Popen", failed_process)
    queue.worker("base", 2)
    arm = _read_state()["arms"]["A"]
    assert len(commands) == 1
    assert arm["status"] == "failed" and arm["failed_stage"] == "train"
    assert list(arm["stages"]) == ["train"]
    assert queue.finalize("base") == 1
    assert _read_state()["status"] == "incomplete"


def test_source_change_between_stages_stops_before_new_process_launch(sandbox, monkeypatch):
    _save_state(sandbox, [_arm()])
    _, commands, _ = _fake_worker_environment(monkeypatch)
    calls = [0]

    def hashes(phase="base"):
        calls[0] += 1
        return sandbox["hashes"] if calls[0] <= 2 else {"source.py": "changed"}

    monkeypatch.setattr(queue, "fingerprints", hashes)
    with pytest.raises(RuntimeError, match="Sources changed"):
        queue.worker("base", 2)
    arm = _read_state()["arms"]["A"]
    assert len(commands) == 1
    assert arm["stages"]["train"]["returncode"] == 0
    assert "eval_best" not in arm["stages"]
    assert arm["status"] == "interrupted" and arm["failed_stage"] == "eval_best"
    assert _read_state()["workers"]["unit-worker"]["source_changed"] is True


def test_base_and_combination_states_are_separate_files(sandbox):
    _save_state(sandbox, [_arm("base")])
    common.save_json(common.state_path("combination"), {"arms": {"combo": {"status": "pending"}}})
    with common.locked_state("combination") as state:
        state["arms"]["combo"]["status"] = "claimed"
    assert _read_state()["arms"]["base"]["status"] == "pending"
    assert json.loads(common.state_path("combination").read_text())["arms"]["combo"]["status"] == "claimed"


def _prepare_gates(sandbox, matching=True):
    arms = [dict(id=f"A{i}", parent="MO0", family="L", config=f"A{i}.json",
                 run_name=f"experiment/A{i}", hypothesis="toy") for i in range(20)]
    common.save_json(sandbox["configs"] / "base.json", {"arms": arms})
    hashes = sandbox["hashes"] if matching else {"source.py": "old"}
    gate = dict(passed=True, source_sha256=hashes, source_sha256_end=hashes,
                arms={arm["id"]: {"max_memory_reserved_mib": 1000.} for arm in arms})
    common.save_json(sandbox["exp"] / "runtime_preflight.json", gate)
    common.save_json(sandbox["exp"] / "cli_smoke.json", dict(
        passed=True, source_sha256=hashes, source_sha256_end=hashes,
    ))


def test_initialize_rejects_stale_source_gate_without_creating_queue(sandbox):
    _prepare_gates(sandbox, matching=False)
    with pytest.raises(RuntimeError, match="matching immutable sources"):
        queue.initialize("base")
    assert not common.state_path().exists()


def test_initialize_cannot_overwrite_existing_run_or_queue(sandbox):
    _prepare_gates(sandbox)
    existing = sandbox["runs"] / "experiment/A0"
    existing.mkdir(parents=True)
    (existing / "keep.pt").write_bytes(b"keep")
    with pytest.raises(RuntimeError, match="overwrite existing run"):
        queue.initialize("base")
    assert (existing / "keep.pt").read_bytes() == b"keep"
    assert not common.state_path().exists()
    _save_state(sandbox, [_arm()])
    before = common.state_path().read_bytes()
    with pytest.raises(RuntimeError, match="Queue already exists"):
        queue.initialize("base")
    assert common.state_path().read_bytes() == before
