"""Isolated two-epoch CLI integration checks and full historical M2 re-evaluation."""
from __future__ import annotations
import copy
import json
import math
import os
import subprocess
import sys
from training_wss_min.tools.m2_optimization_common import (
    ANCHOR_RUN, EXP, MATRIX, PREFIX, ROOT, RUNS, fingerprints, save_json, stamp,
)


def compare(old, new, path="", checks=None):
    checks = [] if checks is None else checks
    if path == "/efficiency":
        return checks
    if isinstance(old, dict):
        for key, value in old.items():
            if key not in new:
                raise AssertionError(f"Missing original metric: {path}/{key}")
            compare(value, new[key], path + "/" + key, checks)
    elif isinstance(old, list):
        assert len(old) == len(new), path
        for i, value in enumerate(old):
            compare(value, new[i], path + f"/{i}", checks)
    elif isinstance(old, (float, int)) and not isinstance(old, bool):
        if math.isnan(float(old)):
            assert math.isnan(float(new)), path
        else:
            assert math.isclose(old, new, rel_tol=1e-4, abs_tol=2e-5), (path, old, new)
        checks.append(path)
    return checks


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CLI GPU smoke/evaluation must use Slurm")
    job = os.environ["SLURM_JOB_ID"]
    folder = EXP / f"smoke_{job}"
    folder.mkdir(parents=True, exist_ok=False)
    source = fingerprints()
    matrix = json.loads(MATRIX.read_text())
    records = {}
    for arm in matrix["arms"]:
        if arm["id"] not in {"MO0", "MO-L3", "MO-S2", "MO-S3", "MO-S5"}:
            continue
        config = copy.deepcopy(arm["resolved_config"])
        config["name"] = f"{PREFIX}/smoke_{job}/{arm['id']}"
        config["train"].update(epochs=2, min_epoch=2, eval_every=2)
        path = folder / f"{arm['id']}.json"
        save_json(path, config)
        command = [sys.executable, "-u", "-m", "training_wss_min.train", "--config", str(path)]
        start = stamp()
        with (folder / f"{arm['id']}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        run = RUNS / config["name"]
        history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
        assert [r["epoch"] for r in history] == [0, 1]
        assert all(math.isfinite(r["train_loss"]) for r in history)
        init = json.loads((run / "initialization.json").read_text())
        assert init["shared_tensors_exact"] and not init["trained_checkpoint_loaded"]
        records[arm["id"]] = dict(command=command, returncode=0, started_at=start, ended_at=stamp(),
                                   run_name=config["name"], epochs=2, initialization=init)
        print(f"{arm['id']} two-epoch CLI smoke passed", flush=True)
        save_json(folder / "results.json", dict(passed=False, source_sha256=source, arms=records))
    command = [sys.executable, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(ANCHOR_RUN),
               "--partitions", "test", "--allow-test", "--checkpoint", "best", "--no-plots",
               "--output-dir", str(folder / "historical_m2_eval")]
    with (folder / "historical_m2_eval.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    old = json.loads((ANCHOR_RUN / "eval/ckpt_best/metrics.json").read_text())["test"]
    new = json.loads((folder / "historical_m2_eval/metrics.json").read_text())["test"]
    checks = compare(old, new)
    assert abs(old["field_casebalanced"]["r2"] - new["field_casebalanced"]["r2"]) <= 1e-4
    end = fingerprints()
    assert source == end, "Sources changed during integration smoke"
    payload = dict(passed=True, job_id=job, source_sha256=source, source_sha256_end=end, arms=records,
                   legacy_full_test34=dict(numerical_checks=len(checks), rtol=1e-4, atol=2e-5,
                     original_r2=old["field_casebalanced"]["r2"], reproduced_r2=new["field_casebalanced"]["r2"],
                     evaluation_dir=str(folder / "historical_m2_eval")))
    save_json(folder / "results.json", payload)
    save_json(EXP / "cli_smoke.json", payload)
    print(f"All five CLI smokes and {len(checks)} historical numerical comparisons passed", flush=True)


if __name__ == "__main__":
    main()
