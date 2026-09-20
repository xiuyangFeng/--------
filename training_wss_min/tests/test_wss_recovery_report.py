import json

from training_wss_min.tools import report_wss_recovery as R


def setup_run(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RUNS", tmp_path / "runs")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"test_cases": ["AG/fast/a"]}))
    cfg = {"name": "matrix/E0", "train": {"seed": 1234, "epochs": 400},
           "data": {"split_path": str(split), "query_mode": "independent", "input_features": ["x", "y", "z"],
                    "support_n_points": 5000}}
    path = tmp_path / "E0.json"
    path.write_text(json.dumps(cfg))
    run = R.RUNS / cfg["name"]
    return path, run


def test_pending_run_never_reports_metrics(tmp_path, monkeypatch):
    cfg, run = setup_run(tmp_path, monkeypatch)
    result = R.run_evidence({"id": "E0"}, cfg, {"status": "pending"})
    assert not result["accepted"]
    assert result["best"] is None and result["last"] is None
    assert result["history"]["rows"] == 0


def accepted_fixture(run, cfg):
    run.mkdir(parents=True)
    (run / "config.json").write_text(cfg.read_text())
    (run / "history.jsonl").write_text("".join(json.dumps({"epoch": i, "train_loss": .5, "elapsed_s": i}) + "\n" for i in range(400)))
    metrics = {"aggregate": {"r2_negative_cases": 0, "n_cases": 1},
               "field_casebalanced": {"r2": .6}, "field": {"mae": 2., "rmse": 4.},
               "per_case": {"AG/fast/a": {}},
               "normalized": {"field_casebalanced": {"r2": .7}, "per_case": {"AG/fast/a": {}}}}
    files, checkpoints = {}, {}
    for which in ("best", "last"):
        cp = run / f"ckpt_{which}.pt"
        cp.write_bytes(which.encode())
        checkpoints[which] = {"epoch": 399, "sha256": R.sha(cp)}
        path = run / "eval" / f"ckpt_{which}" / "metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"test": metrics}))
        files[str(path.resolve())] = R.sha(path)
    for name in ("history.jsonl", "config.json"):
        path = run / name
        files[str(path.resolve())] = R.sha(path)
    acceptance = {"passed": True, "training": {"epochs": 400, "checkpoints": checkpoints}, "files_sha256": files}
    for name in ("acceptance.json", "acceptance_best.json", "acceptance_last.json"):
        (run / name).write_text(json.dumps(acceptance))


def test_complete_report_requires_acceptance_hashes(tmp_path, monkeypatch):
    cfg, run = setup_run(tmp_path, monkeypatch)
    accepted_fixture(run, cfg)
    result = R.run_evidence({"id": "E0"}, cfg, {"status": "complete"})
    assert result["accepted"] and result["best"]["metrics"]["field_casebalanced"]["r2"] == .6
    path = run / "eval/ckpt_best/metrics.json"
    path.write_text(path.read_text().replace('0.6', '0.9'))
    rejected = R.run_evidence({"id": "E0"}, cfg, {"status": "complete"})
    assert not rejected["accepted"]
    assert not rejected["accepted_checkpoints"]["best"]
    assert rejected["best"] is None and rejected["last"] is None


def test_queue_complete_cannot_replace_missing_training_epoch(tmp_path, monkeypatch):
    cfg, run = setup_run(tmp_path, monkeypatch)
    accepted_fixture(run, cfg)
    path = run / "history.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    result = R.run_evidence({"id": "E0"}, cfg, {"status": "complete"})
    assert not result["accepted"] and not result["history"]["epochs_complete"]
    assert result["history"]["rows"] == 399


def test_workbook_mapping_keeps_physical_and_normalized_space_distinct():
    values = R.metric_values({"field_casebalanced": {"r2": .5, "mae": 2.}, "field": {"mae": 3.},
                             "normalized": {"field_casebalanced": {"r2": .7, "mae": .2}, "field": {"mae": .3}}})
    assert values[6] == .5 and values[24] == .7
    assert values[8] == 3. and values[29] == .3
    assert values[54] == 2. and values[56] == .2
