#!/usr/bin/env python3
"""Audit the completed bridge SEP control follow-up (SAME -> SEP only).

与 analyze_q1v_radius_r60_same_results.py 同源：单臂追加，父实验取自已冻结的
SA1-scale 矩阵结果（pointnetpp_sa1_scale_results_analysis.json），不重算父实验。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "training_wss_min/preflight/bridge_sep_followup_submission.json"
PREPARED = ROOT / "training_wss_min/preflight/bridge_sep_followup_prepared.json"
PARENT_ANALYSIS = ROOT / "training_wss_min/preflight/pointnetpp_sa1_scale_results_analysis.json"
OUT_JSON = ROOT / "training_wss_min/preflight/bridge_sep_followup_results_analysis.json"
OUT_CSV = ROOT / "training_wss_min/preflight/bridge_sep_followup_results_summary.csv"
PARENT_EXP_ID = "bridge_rand5000_fixed500_k64"
CHILD_EXP_ID = "bridge_rand5000_fixed500_k64_sep"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and subset(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(subset(a, b) for a, b in zip(expected, actual))
    return expected == actual


def audit_csv(path: Path, expected_cases: int) -> tuple[bool, list[str]]:
    problems: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected_cases:
        problems.append(f"{path.name}: {len(rows)} rows (expected {expected_cases})")
    case_ids = [r.get("case", "") for r in rows]
    if len(set(case_ids)) != expected_cases or any(not c for c in case_ids):
        problems.append(f"{path.name}: duplicate/missing case IDs")
    for r in rows:
        for key, value in r.items():
            if key in ("case", "partition") or value in (None, ""):
                continue
            try:
                if not math.isfinite(float(value)):
                    problems.append(f"{path.name}: non-finite {key}")
                    break
            except ValueError:
                problems.append(f"{path.name}: nonnumeric {key}")
                break
    return not problems, problems


def metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    """Mirrors metric_row() in analyze_pointnetpp_sa1_scale_results.py exactly."""
    t = metrics["test"]
    n = t["normalized"]
    return {
        "physical_r2_cb": t["field_casebalanced"]["r2"],
        "physical_r2": t["field"]["r2"],
        "physical_casemean_r2": t["aggregate"]["r2_casemean"],
        "physical_r2_negative_cases": t["aggregate"]["r2_negative_cases"],
        "physical_mae": t["field_casebalanced"]["mae"],
        "physical_rmse": t["field_casebalanced"]["rmse"],
        "physical_high_wss_r2": t["regional_field"]["high_wss"]["r2"],
        "physical_high_wss_mae": t["regional_field"]["high_wss"]["mae"],
        "physical_spearman": t["hotspot"]["spearman_all_casemean"],
        "physical_top10_iou": t["hotspot"]["top10_iou_casemean"],
        "normalized_r2_cb": n["field_casebalanced"]["r2"],
        "normalized_r2": n["field"]["r2"],
        "normalized_casemean_r2": n["aggregate"]["r2_casemean"],
        "normalized_r2_negative_cases": n["aggregate"]["r2_negative_cases"],
        "normalized_p99_ratio": n["calibration"]["p99_pred_true_ratio"],
        "parameters": t["efficiency"].get("parameters"),
    }


def delta(a: Any, b: Any) -> Any:
    return None if a is None or b is None else a - b


def audit_run(run_dir: Path, frozen_config: Path, expected_sha: str) -> dict[str, Any]:
    frozen = load(frozen_config)
    runtime = load(run_dir / "config.json")
    checks: dict[str, bool] = {}
    problems: list[str] = []
    source_sha = sha256(frozen_config)
    checks["prepared_sha256_match"] = source_sha == expected_sha
    checks["runtime_preserves_frozen_config"] = subset(frozen, runtime)
    checks["seed_1234"] = runtime["train"].get("seed") == 1234
    checks["query_mode_independent"] = runtime["data"].get("query_mode") == "independent"
    split_path = Path(runtime["data"].get("split_path", ""))
    split = load(split_path) if split_path.is_file() else {}
    checks["split_106_0_27"] = (
        len(split.get("train_cases", [])) == 106
        and len(split.get("val_cases", [])) == 0
        and len(split.get("test_cases", [])) == 27
        and "train106" in runtime["data"].get("wss_stats_path", "")
    )
    checks["epochs_400"] = runtime["train"].get("epochs") == 400
    history = [json.loads(x) for x in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    checks["history_complete_400"] = len(history) == 400 and [x.get("epoch") for x in history] == list(range(400))
    checks["ckpt_best_and_last"] = (run_dir / "ckpt_best.pt").is_file() and (run_dir / "ckpt_last.pt").is_file()
    checks["selection_rule_train_loss"] = (
        runtime["train"].get("selection_rule") == "train_loss"
        and runtime["train"].get("ckpt_metric") == "train_loss"
    )
    metric_docs: dict[str, Any] = {}
    checks["best_last_eval_complete"] = True
    checks["per_case_27_complete"] = True
    checks["no_nan_inf"] = True
    for checkpoint in ("ckpt_best", "ckpt_last"):
        metric_path = run_dir / "eval" / checkpoint / "metrics.json"
        csv_path = run_dir / "eval" / checkpoint / "per_case_metrics.csv"
        if not metric_path.is_file() or not csv_path.is_file():
            checks["best_last_eval_complete"] = False
            problems.append(f"missing {checkpoint} eval artifact")
            continue
        metrics = load(metric_path)
        metric_docs[checkpoint] = metrics
        ok, csv_problems = audit_csv(csv_path, 27)
        checks["per_case_27_complete"] &= ok and len(metrics.get("test", {}).get("per_case", {})) == 27
        checks["no_nan_inf"] &= finite(metrics) and not csv_problems
        problems.extend(csv_problems)
    for name, good in checks.items():
        if not good and not any(name in x for x in problems):
            problems.append(name)
    best = metric_docs.get("ckpt_best")
    last = metric_docs.get("ckpt_last")
    tail = history[-1] if history else None
    return {
        "experiment_id": CHILD_EXP_ID, "run_dir": str(run_dir), "config": str(frozen_config),
        "config_sha256": source_sha, "checks": checks, "passed": all(checks.values()), "problems": problems,
        "metrics_best": metric_row(best) if best else {}, "metrics_last": metric_row(last) if last else {},
        "final_train_loss": tail.get("train_loss") if tail else None,
        "training_seconds": tail.get("elapsed_s") if tail else None,
    }


DELTA_KEYS = (
    "physical_r2_cb", "physical_r2", "physical_casemean_r2", "physical_r2_negative_cases",
    "physical_mae", "physical_rmse", "physical_high_wss_r2", "physical_high_wss_mae",
    "physical_spearman", "physical_top10_iou", "normalized_r2_cb", "normalized_r2",
    "normalized_casemean_r2", "normalized_p99_ratio",
)


def main() -> None:
    submission = load(SUBMISSION)
    prepared = load(PREPARED)
    if submission.get("status") != "submitted":
        raise RuntimeError("submission is not in submitted state")
    parent_analysis = load(PARENT_ANALYSIS)
    parent = parent_analysis["results"][PARENT_EXP_ID]
    row = prepared["configs"][0]
    run_dir = Path(submission["run_dir"])
    frozen_config = Path(submission["config"])
    record = audit_run(run_dir, frozen_config, submission["config_sha256"])
    if not record["passed"]:
        raise RuntimeError(f"bridge SEP follow-up failed result integrity audit: {record['problems']}")
    m, p = record["metrics_best"], parent["metrics_best"]
    deltas = {key: delta(m[key], p[key]) for key in DELTA_KEYS}
    last_minus_best = record["metrics_last"]["physical_r2_cb"] - record["metrics_best"]["physical_r2_cb"]
    payload = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "bridge_rand5000_fixed500_k64 SEP control follow-up (SAME -> SEP only)",
        "test27_policy": submission["test27_policy"],
        "single_change": submission["single_change"],
        "source_submission": str(SUBMISSION), "source_submission_sha256": sha256(SUBMISSION),
        "source_parent_analysis": str(PARENT_ANALYSIS),
        "slurm_jobs": submission["jobs"],
        "parent": {"experiment_id": PARENT_EXP_ID, "run_dir": parent["run_dir"], "config": parent["config"],
                   "metrics_best": p, "metrics_last": parent["metrics_last"]},
        "child": record,
        "paired_delta_vs_parent_best": deltas,
        "child_last_minus_best_r2_cb": last_minus_best,
        "ranking_note": "Same-protocol engineering comparison on reused historical test27; not an independent confirmation.",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["experiment_id", "reference_id"] + [f"{k}" for k in DELTA_KEYS] + [f"delta_{k}" for k in DELTA_KEYS] + ["last_minus_best_r2_cb", "integrity"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        out_row = {"experiment_id": CHILD_EXP_ID, "reference_id": PARENT_EXP_ID,
                   **{k: m[k] for k in DELTA_KEYS}, **{f"delta_{k}": deltas[k] for k in DELTA_KEYS},
                   "last_minus_best_r2_cb": last_minus_best, "integrity": "passed"}
        writer.writerow(out_row)
    print(json.dumps({"status": "passed", "json": str(OUT_JSON), "csv": str(OUT_CSV),
                      "paired_delta_vs_parent_best": deltas}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
