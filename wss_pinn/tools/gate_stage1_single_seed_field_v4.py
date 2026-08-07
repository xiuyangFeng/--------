"""Freeze the single-seed ranking before Stage-1 confirmation seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wss_pinn.config import ExperimentConfig
from wss_pinn.utils import ROOT, atomic_write_json, git_state, sha256_file, utc_now


CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_field_v4"
DATA_MANIFEST = ROOT / "data_wss_pinn/volume_uvwp_peak_field_v4_train123_val15/manifest.json"
OUTPUT = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage1_single_seed_gate/report.json"
CONFIGS = (
    "g_raw_s1234.json",
    "l_raw_s1234.json",
    "g_pe_s1234.json",
    "l_pe_s1234.json",
)
CONFIRMATION_SEEDS = (1234, 2345, 3456)


def _arm(config_path: Path, expected_val_cases: set[str]) -> tuple[dict[str, Any], list[str]]:
    config = ExperimentConfig.from_json(config_path)
    run_dir = config.run_dir
    summary_path = run_dir / "training_summary.json"
    audit_path = run_dir / "selection_audit.json"
    evaluation_path = run_dir / "evaluation_val15_best_validation_field_cb.json"
    checkpoint_path = run_dir / "checkpoints/best_validation_field_cb.pt"
    required = (summary_path, audit_path, evaluation_path, checkpoint_path)
    failures: list[str] = []
    if any(not path.is_file() for path in required):
        return {
            "experiment_id": config["experiment"]["id"],
            "config": str(config_path.resolve()),
            "status": "missing_outputs",
        }, [f"missing_outputs:{config['experiment']['id']}"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    case_ids = {str(row["case_id"]) for row in evaluation.get("cases", [])}
    score = float(summary.get("best_validation_field_score_cb", float("nan")))
    if summary.get("status") != "completed":
        failures.append(f"training_status:{config['experiment']['id']}")
    if evaluation.get("status") != "completed" or case_ids != expected_val_cases:
        failures.append(f"validation_contract:{config['experiment']['id']}")
    if not np.isfinite(score):
        failures.append(f"nonfinite_score:{config['experiment']['id']}")
    if audit.get("selection_metric") != "validation_field_score_cb":
        failures.append(f"selection_metric:{config['experiment']['id']}")
    if (run_dir / "error.json").exists():
        failures.append(f"error_file:{config['experiment']['id']}")
    return {
        "experiment_id": config["experiment"]["id"],
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "validation_field_score_cb": score,
        "epochs_executed": int(summary.get("epochs_executed", 0)),
        "best_epoch_zero_based": audit.get("best_epoch_zero_based", {}).get("case_balanced"),
        "legacy_best_epoch_zero_based": audit.get("best_epoch_zero_based", {}).get("legacy_batch_weighted"),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
        "evaluation": {
            "path": str(evaluation_path.resolve()),
            "sha256": sha256_file(evaluation_path),
            "val_cases": len(case_ids),
        },
        "gate_result": "pass" if not failures else "fail",
    }, failures


def run() -> dict[str, Any]:
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    expected_val_cases = {
        str(row["canonical_id"])
        for row in manifest.get("cases", [])
        if row.get("role") == "val"
    }
    failures: list[str] = []
    arms = []
    for name in CONFIGS:
        arm, arm_failures = _arm(CONFIG_ROOT / name, expected_val_cases)
        arms.append(arm)
        failures.extend(arm_failures)
    ranked = sorted(arms, key=lambda row: float(row.get("validation_field_score_cb", float("inf"))))
    selected = [row["experiment_id"] for row in ranked[:2]] if len(ranked) == 4 else []
    if len(expected_val_cases) != 15:
        failures.append("val15_manifest_count")
    if len(selected) != 2:
        failures.append("top2_selection")
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "route": "volume_uvwp_peak_field_v4",
        "phase": "single_seed_ranking",
        "primary_selection": "validation_field_score_cb",
        "ranking": ranked,
        "selected_top2": selected,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "gate_result": "pass" if not failures else "fail",
        "gate_reasons": failures,
        "formal_multiseed_training_allowed": not failures,
        "test35_cases_read": 0,
        "wss_used": False,
        "physics_residual_used_for_selection": False,
        "data_manifest": {
            "path": str(DATA_MANIFEST.resolve()),
            "sha256": sha256_file(DATA_MANIFEST),
            "validation_cases": len(expected_val_cases),
        },
        "git": git_state(),
    }
    atomic_write_json(OUTPUT, report)
    return report


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
