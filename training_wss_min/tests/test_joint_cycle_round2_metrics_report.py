"""Physical-metric and formal-report integrity checks; no model/data evaluation."""
from copy import deepcopy
import json

import numpy as np
import pytest

from training_wss_min.joint_cycle import case_metrics as old_case_metrics, mean_records, weighted_fields
from training_wss_min.joint_cycle_round2_metrics import case_metrics
from training_wss_min.tools import report_joint_cycle_round2_v52 as report


@pytest.mark.parametrize("task,channels", [("velocity", 3), ("pressure", 1), ("wss", 1)])
def test_preserves_original_segments_and_uses_fixed_peak_frame(task, channels):
    rng = np.random.default_rng(11)
    y = rng.uniform(.1, 3, (9, 80, channels))
    p = y + rng.normal(0, .03, y.shape)
    w = rng.uniform(.1, 1, 9)
    old, new = old_case_metrics(task, y, p, w, .01), case_metrics(task, y, p, w, .01)
    for key, value in old.items():
        if key == "speed":
            assert all(new[key][stage] == metrics for stage, metrics in value.items())
        else:
            assert new[key] == value
    assert list(new["per_frame"]) == [f"{f:02d}" for f in range(80)]
    expected = weighted_fields(y[:, 21:22], p[:, 21:22], w)
    assert all(new["peak_frame21"][key] == value for key, value in expected.items())
    reduced = mean_records([new, new])
    assert reduced["per_frame"]["21"]["rmse"] == new["peak_frame21"]["rmse"]


def test_increment_errors_not_derivatives_or_periodic_closure():
    y = np.arange(80, dtype=float)[None, :, None] * np.array([1., 2.])[:, None, None]
    p = 2 * y + 10
    out = case_metrics("pressure", y, p, np.array([3., 1.]), .01)
    np.testing.assert_allclose(out["temporal_difference"]["mse"], 1.75)
    np.testing.assert_allclose(out["endpoint_difference"]["mse"], 1.75 * 79**2)
    # A constant prediction bias affects values, but not their increments.
    shifted = case_metrics("pressure", y, y + 12, np.ones(2), .01)
    assert shifted["cycle"]["mae"] == 12
    assert shifted["temporal_difference"]["rmse"] == 0
    assert shifted["endpoint_difference"]["rmse"] == 0


def test_zero_energy_stays_null_and_zero_vector_keeps_direction_penalty():
    out = case_metrics("velocity", np.zeros((2, 80, 3)), np.ones((2, 80, 3)), np.ones(2), .01)
    for key in ("peak_frame21", "temporal_difference", "endpoint_difference"):
        assert out[key]["r2"] is None and out[key]["relative_l2"] is None
    assert out["peak_frame21"]["direction_cosine"] is None
    assert out["peak_frame21"]["direction_weight_coverage"] == 0
    out = case_metrics("velocity", np.ones((2, 80, 3)), np.zeros((2, 80, 3)), np.ones(2), .01)
    assert out["peak_frame21"]["direction_cosine"] == 0
    assert out["peak_frame21"]["direction_weight_coverage"] == 1
    json.dumps(out, allow_nan=False)


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def formal_run(tmp_path):
    out = tmp_path / "P1_f0_s1234"
    out.mkdir()
    split = {"train_cases": [f"train{i}" for i in range(206)], "test_cases": [f"test{i}" for i in range(55)],
             "fold": 0, "n_folds": 5}
    split_path = tmp_path / "split.json"
    write_json(split_path, split)
    stats_path = tmp_path / "stats.json"
    write_json(stats_path, {"split_path": str(split_path), "split_sha256": report.digest(split_path),
                           "complete_train_partition": True, "train_ids": split["train_cases"]})
    cfg = {"schema": "joint_cycle_round2_v52_v1", "seed": 1234, "arm": "P1", "variant": "J1", "tasks": ["pressure"],
           "out_dir": str(out), "data": {"stats_path": str(stats_path), "split_path": str(split_path),
                                           "split_sha256": report.digest(split_path), "fold": 0},
           "train": {"phases": 80, "epochs": 150, "selection": "last"}, "eval": {"partition": "test"}, "model": {}}
    path = tmp_path / "P1.json"
    write_json(path, cfg)
    write_json(out / "config.json", cfg)
    code = tmp_path / "frozen_code.py"
    code.write_text("# frozen fixture\n")
    write_json(out / "provenance.json", {"smoke": False, "config_path": str(path), "config_sha256": report.digest(path),
                                         "stats_sha256": report.digest(stats_path), "parameter_count": 100,
                                         "code": {str(code): report.digest(code)}})
    write_json(out / "training_complete.json", {"smoke": False, "epochs": 150, "seconds": 25})
    y = np.arange(240).reshape(3, 80, 1).astype(float)
    task_metrics = {"pressure": case_metrics("pressure", y, y + 1, np.ones(3), .01)}
    cases = [{"unit_id": unit, "partition": "test", "tasks": task_metrics,
              "cache_load_encode_decode_seconds": 1 + i / 100} for i, unit in enumerate(split["test_cases"])]
    (out / "per_case.jsonl").write_text("\n".join(json.dumps(r) for r in cases) + "\n")
    metrics = {"status": "complete", "arm": "P1", "variant": "J1", "seed": 1234, "phase_count": 80,
               "partition": "test", "n_units": 55, "selection": "last", "parameter_count": 100,
               "tasks": task_metrics, "inference": {"n_units": 55}, "peak_cuda_memory_bytes": 1024}
    write_json(out / "metrics.json", metrics)
    (out / "history.jsonl").write_text(json.dumps({"epoch": 150, "seconds": 25, "peak_cuda_memory_bytes": 1024}) + "\n")
    return path, cfg, report.base.split_contract([cfg]), out, cases


def test_formal_report_accepts_all55_and_exposes_progress_and_cost(formal_run):
    path, cfg, contract, _, _ = formal_run
    row, cases = report.load_arm(path, cfg, contract)
    assert not row["errors"]
    assert row["formal_complete"] and len(cases) == 55
    assert row["last_completed_epoch"] == 150
    assert row["sampled_query_median_s"] == 1.27
    np.testing.assert_allclose(row["sampled_query_p95_s"], 1.513)
    assert row["robustness"]["pressure"]["cycle"]["n_valid_r2"] == 55


def test_report_rejects_missing_units_smoke_and_source_mutations(formal_run):
    path, cfg, contract, out, cases = formal_run
    (out / "per_case.jsonl").write_text("\n".join(json.dumps(r) for r in cases[:-1]) + "\n")
    row, records = report.load_arm(path, cfg, contract)
    assert not row["formal_complete"] and not records
    assert any("55 unique" in e for e in row["errors"])
    (out / "per_case.jsonl").write_text("\n".join(json.dumps(r) for r in cases) + "\n")
    provenance = report.read_json(out / "provenance.json")
    provenance["smoke"] = True
    write_json(out / "provenance.json", provenance)
    row, records = report.load_arm(path, cfg, contract)
    assert not records and any("smoke" in e for e in row["errors"])
    provenance["smoke"] = False
    write_json(out / "provenance.json", provenance)
    path.parent.joinpath("frozen_code.py").write_text("# modified\n")
    row, records = report.load_arm(path, cfg, contract)
    assert not records and any("code fingerprint" in e for e in row["errors"])


def test_report_rejects_aggregate_change_and_missing_new_metric(formal_run):
    path, cfg, contract, out, cases = formal_run
    metrics = report.read_json(out / "metrics.json")
    metrics["tasks"]["pressure"]["per_frame"]["21"]["mae"] += .5
    write_json(out / "metrics.json", metrics)
    row, records = report.load_arm(path, cfg, contract)
    assert not records and any("aggregate/per_case mismatch" in e for e in row["errors"])
    cases[0] = deepcopy(cases[0])
    del cases[0]["tasks"]["pressure"]["temporal_difference"]
    (out / "per_case.jsonl").write_text("\n".join(json.dumps(r) for r in cases) + "\n")
    row, records = report.load_arm(path, cfg, contract)
    assert not records and any("temporal_difference" in e for e in row["errors"])


def test_pairing_excludes_null_values_and_scheduler_failure_is_not_running():
    def record(r2, mae):
        return {"tasks": {"pressure": {"cycle": {"r2": r2, "mae": mae}}}}
    records = {"P1": {"a": record(.8, 1), "b": record(None, 2)},
               "Ip": {"a": record(.6, 2), "b": record(.3, 3), "c": record(.9, 1)}}
    delta = report.paired_delta("P1", "Ip", records)
    assert delta["n_common_units"] == 2
    assert delta["metrics"]["pressure_cycle_r2"]["n_pairs"] == 1
    assert delta["metrics"]["pressure_cycle_r2"]["improvement_fraction"] == 1
    assert delta["metrics"]["pressure_cycle_mae"]["mean_delta"] == -1
    row = {"status": "running", "formal_complete": False, "has_metrics_artifact": False, "warnings": []}
    audit = {"array_job": "123", "jobs": {"123_0": {"state": "OUT_OF_MEMORY", "job_id": "123_0", "raw_state": "OUT_OF_MEMORY", "exit_code": "1:0"}}}
    report.base.apply_scheduler_evidence(row, 0, audit)
    assert row["status"] == "failed" and row["status_evidence"] == "slurm_terminal_failure"
