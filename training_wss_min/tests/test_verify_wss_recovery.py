from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from training_wss_min import dataset as D, metrics as M
from training_wss_min.tools import verify_wss_recovery as V


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def production_metrics(names, truths, predictions):
    per_case = {}
    for name, y, p in zip(names, truths, predictions):
        positions = np.column_stack([np.arange(len(y)), np.zeros((len(y), 2))])
        per_case[name] = {"overall": M.basic_metrics(y, p),
                          "hotspot": M.hotspot_localization_metrics(y, p, positions)}
    return {"per_case": per_case, "aggregate": M.aggregate_case_metrics(per_case),
            "field": M.basic_metrics(np.concatenate(truths), np.concatenate(predictions)),
            "field_casebalanced": M.casebalanced_field_metrics(truths, predictions),
            "hotspot": M.aggregate_hotspot_metrics(per_case)}


@pytest.fixture
def complete_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    names = ["AAA/a/case_a", "ILO/b/case_b"]
    split = tmp_path / "split.json"
    save(split, {"train_cases": ["AAA/train/case"], "test_cases": names})
    stats = {"method": "log_z", "eps": 1e-6, "log": {"mean": .4, "std": 1.2},
             "split_sha256": V.sha256(split)}
    stats_path = tmp_path / "stats.json"
    save(stats_path, stats)
    save(run / "wss_global_stats.json", stats)
    cfg = {"data": {"data_root": str(tmp_path / "data"), "wss_stats_path": str(stats_path),
                     "split_path": str(split), "target": "wss", "target_normalization": "global_stats",
                     "timesteps": "peak", "required_frame_version": "test_frame"},
           "train": {"epochs": 2, "seed": 1234, "selection_rule": "train_loss"}}
    save(run / "config.json", cfg)
    history = [{"epoch": e, "train_loss": loss, "selection_score": -loss,
                "train_mse_norm": loss, "train_mae_norm": loss, "train_rmse_norm": loss ** .5,
                "lr": .001} for e, loss in enumerate([1., .5])]
    (run / "history.jsonl").write_text("\n".join(json.dumps(row) for row in history) + "\n")
    for which in ("best", "last"):
        torch.save({"epoch": 1, "model": {"weight": torch.tensor([1.])}}, run / f"ckpt_{which}.pt")
    raws = [np.array([1., 2., 4., 8.], np.float32), np.array([.4, 1., 2., 3., 6., 10.], np.float32)]
    norms = [D.normalize_wss(y, stats) for y in raws]
    for name, y in zip(names, raws):
        source = tmp_path / "data" / name / "bundle.npz"
        source.parent.mkdir(parents=True)
        np.savez(source, wall_wss=np.stack([y * .1, y]), steps=[1100, 1162], peak_step=1162,
                 wall_coords_norm=np.zeros((len(y), 3)), transform_frame_version="test_frame")
    for which, offset in (("best", .1), ("last", -.2)):
        evaldir = run / "eval" / f"ckpt_{which}"
        entries = []
        pred_norms = [y.astype(np.float64) + offset for y in norms]
        pred_raws = [D.denormalize_wss(y, stats) for y in pred_norms]
        for name, yn, yr, pn, pr in zip(names, norms, raws, pred_norms, pred_raws):
            relative = Path(name) / "predictions.npz"
            target = evaldir / "predictions" / "test" / relative
            target.parent.mkdir(parents=True)
            np.savez(target, pred_norm=pn, true_norm=yn, pred_pa_unclipped=pr,
                     pred_pa=np.maximum(pr, 0), true_pa=yr, row_index=np.arange(len(yr)))
            entries.append({"unit_id": name, "file": str(relative), "n_points": len(yr),
                            "sha256": V.sha256(target), "metric_clipping": "nonnegative after inverse log_z"})
        save(evaldir / "predictions" / "test" / "manifest.json",
             {"schema_version": 1, "target": "wss", "cases": entries})
        save(evaldir / "predictions" / "manifest.json",
             {"schema_version": 1, "target": "wss", "checkpoint": which,
              "checkpoint_sha256": V.sha256(run / f"ckpt_{which}.pt"),
              "config_sha256": V.sha256(run / "config.json"),
              "wss_stats_sha256": V.sha256(run / "wss_global_stats.json"),
              "evaluation_split_path": str(split), "partitions": {"test":
                  [{**item, "file": str(Path("test") / item["file"])} for item in entries]}})
        physical = production_metrics(names, raws, pred_raws)
        physical["normalized"] = production_metrics(names, norms, pred_norms)
        physical["evaluation_split_path"] = str(split)
        save(evaldir / "metrics.json", {"test": physical})
    return run


def test_independent_verifier_accepts_real_metric_contract(complete_run):
    result = V.verify_run(complete_run, expected_epochs=2, expected_cases=2)
    assert result["passed"], result.get("error")
    assert set(result["checkpoints"]) == {"best", "last"}
    assert result["training"]["epoch_indexing"] == "zero-based"
    assert str(complete_run / "eval" / "ckpt_last" / "metrics.json") in result["files_sha256"]
    assert V.read_json(complete_run / "acceptance.json")["passed"]


def test_best_acceptance_does_not_require_last_evaluation(complete_run):
    shutil.rmtree(complete_run / "eval" / "ckpt_last")
    result = V.verify_run(complete_run, "best", expected_epochs=2, expected_cases=2)
    assert result["passed"], result.get("error")
    assert set(result["checkpoints"]) == {"best"}
    assert not (complete_run / "acceptance.json").exists()


@pytest.mark.parametrize("problem", ["short_history", "history_nonfinite", "bad_epoch", "npz_hash", "metric_drift", "source_labels", "changed_split"])
def test_verifier_rejects_incomplete_or_tampered_results(complete_run, problem):
    if problem in ("short_history", "history_nonfinite"):
        path = complete_run / "history.jsonl"
        rows = path.read_text().splitlines()
        if problem == "short_history":
            rows.pop()
        else:
            row = json.loads(rows[0])
            row["train_loss"] = float("nan")
            rows[0] = json.dumps(row)
        path.write_text("\n".join(rows))
    elif problem == "bad_epoch":
        torch.save({"epoch": 0, "model": {"weight": torch.tensor([1.])}}, complete_run / "ckpt_last.pt")
    elif problem == "npz_hash":
        path = next((complete_run / "eval" / "ckpt_best" / "predictions" / "test").rglob("predictions.npz"))
        with path.open("ab") as stream:
            stream.write(b"changed")
    elif problem == "metric_drift":
        path = complete_run / "eval" / "ckpt_best" / "metrics.json"
        value = V.read_json(path)
        value["test"]["field_casebalanced"]["r2"] += .1
        save(path, value)
    elif problem == "source_labels":
        cfg = V.read_json(complete_run / "config.json")
        path = next(Path(cfg["data"]["data_root"]).rglob("bundle.npz"))
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["wall_wss"][1] *= 2
        np.savez(path, **arrays)
    elif problem == "changed_split":
        cfg = V.read_json(complete_run / "config.json")
        path = Path(cfg["data"]["split_path"])
        path.write_text(path.read_text() + "\n")
    result = V.verify_run(complete_run, expected_epochs=2, expected_cases=2)
    assert not result["passed"]
    assert "error" in result
    assert not V.read_json(complete_run / "acceptance.json")["passed"]


def test_metric_recompute_case_balance_and_negative_cases():
    ys = [np.array([0., 1.]), np.array([2., 3., 4., 5.])]
    ps = [ys[0] + 4, ys[1]]
    result = V.summarize(ys, ps, ["small", "large"])
    assert result["field_casebalanced"]["mae"] == 2.
    assert result["field"]["mae"] == pytest.approx(8 / 6)
    assert result["negative_cases"] == ["small"]
    assert result["aggregate"]["r2_negative_cases"] == 1


def test_frozen_source_hash_is_required_when_preflight_is_provided(complete_run):
    cfg = V.read_json(complete_run / "config.json")
    sources = list(Path(cfg["data"]["data_root"]).rglob("bundle.npz"))
    evidence = complete_run / "preflight.json"
    fingerprints = {str(p.resolve()): {"sha256": V.sha256(p)} for p in sources}
    save(evidence, {"passed": True, "slurm_job_id": "test_job", "data_fingerprints": fingerprints})
    result = V.verify_run(complete_run, expected_epochs=2, expected_cases=2, preflight_path=evidence)
    assert result["passed"], result.get("error")
    fingerprints[str(sources[0].resolve())]["sha256"] = "0" * 64
    save(evidence, {"passed": True, "slurm_job_id": "test_job", "data_fingerprints": fingerprints})
    result = V.verify_run(complete_run, expected_epochs=2, expected_cases=2, preflight_path=evidence)
    assert not result["passed"] and "sha256" in result["error"]


def test_prediction_checkpoint_provenance_is_required(complete_run):
    path = complete_run / "eval" / "ckpt_best" / "predictions" / "manifest.json"
    manifest = V.read_json(path)
    manifest["checkpoint_sha256"] = "0" * 64
    save(path, manifest)
    result = V.verify_run(complete_run, expected_epochs=2, expected_cases=2)
    assert not result["passed"] and "prediction_provenance" in result["error"]
