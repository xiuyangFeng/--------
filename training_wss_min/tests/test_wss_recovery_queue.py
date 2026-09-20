"""Selection and phase-barrier tests; training/GPU/verification are mocked."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import pytest

from training_wss_min.tools import run_next_matrix_queue as base
from training_wss_min.tools import run_wss_recovery_queue as queue
from training_wss_min.tests.test_next_matrix_queue import runtime  # shared fake subprocess runtime


def summary(norm=.8, pa=.6, mae=2., iou=.4, p10=.2, negative=()):
    return dict(norm_r2=norm, pa_r2=pa, pa_mae=mae, iou=iou, pa_p10=p10,
                negative_cases=list(negative), case_ids=["A", "B", "C"])


def metrics(values):
    per_case = {case: {"overall": {"r2": -.1 if case in values["negative_cases"] else .5}}
                for case in values["case_ids"]}
    return {"test": {
        "normalized": {"field_casebalanced": {"r2": values["norm_r2"]}, "per_case": per_case},
        "field_casebalanced": {"r2": values["pa_r2"], "mae": 999.},
        "field": {"mae": values["pa_mae"]},
        "hotspot": {"top10_iou_casemean": values["iou"]},
        "aggregate": {"r2_casep10": values["pa_p10"], "n_cases": len(per_case)},
        "per_case": per_case,
    }}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    arms = [{"id": aid, "config": f"{aid}.json", "phase": int(aid.startswith("C"))}
            for aid in [*(f"E{i}" for i in range(10)), *(f"C{i}" for i in range(1, 5))]]
    for arm in arms:
        (config_dir / arm["config"]).write_text(json.dumps({"name": f"run_{arm['id']}"}))
    arms[-1]["candidate_configs"] = {}
    for aid in queue.STRUCTURES:
        candidate = config_dir / aid / "C4.json"
        candidate.parent.mkdir()
        candidate.write_text(json.dumps({"name": "run_C4", "structure": aid}))
        arms[-1]["candidate_configs"][aid] = f"{aid}/C4.json"
    (config_dir / "matrix.json").write_text(json.dumps({"arms": arms}))
    return argparse.Namespace(config_dir=config_dir, matrix=None, experiment_dir=tmp_path / "experiment",
                              slots_per_gpu=2, min_free_mib=7000, launch_interval_seconds=3,
                              poll_seconds=1, terminate_timeout_seconds=1)


@pytest.fixture
def verification(monkeypatch):
    values = {f"E{i}": summary() for i in range(10)}
    values["E2"] = summary(.81, .61)
    values["E3"] = summary(.82, .62)
    monkeypatch.setattr(queue.RecoveryQueue, "verify_gate", lambda self: {"status": "passed"})
    monkeypatch.setattr(queue.RecoveryQueue, "verify_source", lambda self: None)

    def verify(self, aid, checkpoint):
        run = queue.ROOT / "training_wss_min/runs" / self.state["arms"][aid]["run_name"]
        path = run / "eval" / f"ckpt_{checkpoint}" / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics(values.get(aid, summary()))))
        return {"status": "passed"}

    monkeypatch.setattr(queue.RecoveryQueue, "verify_evaluation", verify)
    return values


def test_selection_prefers_qualified_structure_over_higher_score_with_regression():
    values = {"E0": summary(), "E2": summary(.82, .62), "E3": summary(.9, .64, mae=2.05),
              "E4": summary(.83, .59), "E5": summary(.81, .61)}
    result = queue.choose_structure(values)
    assert result["selected_structure"] == "E2"
    assert result["eligible_structures"] == ["E2", "E5"]
    assert not result["forced_exploratory"]


def test_no_eligible_structure_still_runs_best_structure_with_forced_label_and_tie_breaks():
    values = {"E0": summary(), "E2": summary(.79, .5), "E3": summary(.8, .51),
              "E4": summary(.8, .52), "E5": summary(.8, .52)}
    result = queue.choose_structure(values)
    assert result["selected_structure"] == "E4"
    assert result["forced_exploratory"]
    assert result["ranking"] == ["E4", "E5", "E3", "E2"]


def test_same_number_of_negative_cases_is_insufficient_if_identity_changes():
    result = queue.qualification(summary(.82, .62, negative=["B"]), summary(negative=["A"]))
    assert not result["qualified"]
    assert result["additional_negative_cases"] == ["B"]
    assert not result["checks"]["no_additional_negative_cases"]


@pytest.mark.parametrize("field,value", [("norm_r2", .8049), ("pa_r2", .6), ("pa_mae", 2.0401),
                                        ("iou", .3899), ("pa_p10", .1799)])
def test_each_pre_registered_metric_gate_can_reject(field, value):
    candidate = summary(.81, .61)
    candidate[field] = value
    assert not queue.qualification(candidate, summary())["qualified"]


def test_exact_inclusive_thresholds_pass():
    assert queue.qualification(summary(.805, .600001, 2.04, .39, .18), summary())["qualified"]


def test_real_schema_uses_field_mae_not_casebalanced_mae(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics(summary(.81, .61))))
    result = queue.metric_summary(path)
    assert result["pa_mae"] == 2.
    assert result["norm_r2"] == .81
    assert len(result["source_sha256"]) == 64


def test_full_14_arm_queue_selects_frozen_c4_without_rewriting_and_saves_predictions(sandbox, runtime, verification):
    frozen = {p: p.read_bytes() for p in sandbox.config_dir.rglob("*.json")}
    runner = queue.RecoveryQueue(sandbox)
    assert runner.run() == 0
    assert len(runtime.processes) == 42
    assert set(runner.state["arms"]) == queue.EXPECTED_ARMS
    assert all(r["status"] == "complete" for r in runner.state["arms"].values())
    c4 = runner.state["arms"]["C4"]
    assert c4["selected_structure"] == "E3"
    assert c4["config"].endswith("E3/C4.json")
    assert c4["run_name"] == "run_C4"
    assert c4["constituents"] == ["E3", "E9"]
    assert runner.state["arms"]["C2"]["forced_exploratory"]  # E6 has no gain, still executes.
    assert all(p.read_bytes() == data for p, data in frozen.items())
    for record in runner.state["arms"].values():
        for stage in ("eval_best", "eval_last"):
            assert "--save-predictions" in record["stages"][stage]["cmd"]
            assert record["stages"][stage]["prediction_verification"]["status"] == "passed"
    # Complete all E arms (including last evaluations) before any C train.
    first_combo = min(i for i, e in enumerate(runtime.events) if e[0] == "launch" and e[1].startswith("C"))
    for aid in (f"E{i}" for i in range(10)):
        last_wait = next(i for i, e in enumerate(runtime.events) if e[:3] == ("wait", aid, "eval_last"))
        assert last_wait < first_combo


def test_failed_evaluation_blocks_all_combinations(sandbox, runtime, verification):
    runtime.fail_stages = {("E2", "eval_best")}
    runner = queue.RecoveryQueue(sandbox)
    assert runner.run() == 1
    assert runner.state["arms"]["E2"]["failed_stage"] == "eval_best"
    assert all(runner.state["arms"][f"C{i}"]["status"] == "blocked" for i in range(1, 5))
    assert not any(p.aid.startswith("C") for p in runtime.processes)


def test_prediction_verification_failure_blocks_last_eval_and_combinations(sandbox, runtime, verification, monkeypatch):
    original = queue.RecoveryQueue.verify_evaluation

    def verify(self, aid, checkpoint):
        if aid == "E3" and checkpoint == "best":
            raise ValueError("independent full-wall recomputation disagrees")
        return original(self, aid, checkpoint)

    monkeypatch.setattr(queue.RecoveryQueue, "verify_evaluation", verify)
    runner = queue.RecoveryQueue(sandbox)
    assert runner.run() == 1
    assert runner.state["arms"]["E3"]["failed_stage"] == "verify_eval_best"
    assert not any(p.aid == "E3" and p.stage == "eval_last" for p in runtime.processes)
    assert all(runner.state["arms"][f"C{i}"]["status"] == "blocked" for i in range(1, 5))


def test_corrupt_metrics_block_selection_without_launching_placeholder_c4(sandbox, runtime, verification):
    verification["E4"]["norm_r2"] = float("nan")
    runner = queue.RecoveryQueue(sandbox)
    assert runner.run() == 1
    assert "non-finite" in runner.state["combination_selection_error"]
    assert not runner.state["phase1_prepared"]
    assert all(runner.state["arms"][f"C{i}"]["status"] == "blocked" for i in range(1, 5))
    assert not any(p.aid.startswith("C") for p in runtime.processes)


def test_gate_failure_starts_no_child(sandbox, runtime, monkeypatch):
    def fail(self):
        raise ValueError("source fingerprint changed after GPU smoke")

    monkeypatch.setattr(queue.RecoveryQueue, "verify_gate", fail)
    with pytest.raises(ValueError, match="fingerprint"):
        queue.RecoveryQueue(sandbox).run()
    assert not runtime.processes


def test_source_change_between_stages_stops_and_reaps_running_training(sandbox, runtime, verification, monkeypatch):
    checks = 0

    def verify(self):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("source/config changed")

    monkeypatch.setattr(queue.RecoveryQueue, "verify_source", verify)
    runner = queue.RecoveryQueue(sandbox)
    with pytest.raises(RuntimeError, match="source/config changed"):
        runner.run()
    assert len(runtime.processes) == 1
    assert runtime.processes[0].waits > 0 and runtime.processes[0].stream.closed
    assert runner.state["status"] == "failed"
    assert all(record["status"] in base.TERMINAL for record in runner.state["arms"].values())


def test_c4_placeholder_cannot_launch(sandbox, runtime):
    runner = queue.RecoveryQueue(sandbox)
    with pytest.raises(RuntimeError, match="placeholder"):
        runner.launch("C4", "0", "train")
    assert not runtime.processes


def test_candidate_config_cannot_escape_or_change_run_identity(sandbox, runtime):
    path = sandbox.config_dir / "E2/C4.json"
    path.write_text(json.dumps({"name": "extra_fifteenth_arm"}))
    with pytest.raises(ValueError, match="one C4 run name"):
        queue.RecoveryQueue(sandbox)


@pytest.mark.parametrize("error", [False, True])
def test_main_reports_on_success_and_exception_preserving_queue_result(sandbox, monkeypatch, error):
    called = []

    class Runner:
        def __init__(self, args):
            pass

        def run(self):
            if error:
                raise RuntimeError("queue original exception")
            return 7

    monkeypatch.setattr(queue, "RecoveryQueue", Runner)
    monkeypatch.setattr(queue, "write_report_safely", lambda args: called.append(args.experiment_dir))
    arguments = ["--experiment-dir", str(sandbox.experiment_dir), "--config-dir", str(sandbox.config_dir)]
    if error:
        with pytest.raises(RuntimeError, match="original exception"):
            queue.main(arguments)
    else:
        assert queue.main(arguments) == 7
    assert called == [sandbox.experiment_dir]


def test_report_failure_is_recorded_separately_and_never_raised(sandbox, monkeypatch):
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 9))
    queue.write_report_safely(sandbox)
    failure = json.loads((sandbox.experiment_dir / "report_failure.json").read_text())
    assert failure["returncode"] == 9
    assert Path(failure["log"]).exists()
