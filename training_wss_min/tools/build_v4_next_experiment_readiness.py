#!/usr/bin/env python3
"""汇总 v4 验收、新 split 与四组待提交配置的只读准备状态。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "data_wss_min/pipeline_reports/v4_cutover_20260715_1921"
CONFIGS = [
    ROOT / "training_wss_min/configs/pointnetpp_v4/ag_v4_sa3_e2_global_fps2000.json",
    ROOT / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_locked_sa3_e2_global_fps2000.json",
    ROOT / "training_wss_min/configs/pointnetpp_v4/ag_aaa_v4_stratified_sa3_e2_global_fps2000.json",
    ROOT / "training_wss_min/configs/pointnet_v4/ag_aaa_v4_stratified_e2_global_fps2000.json",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    acceptance_path = REPORT_DIR / "v4_e2_global_acceptance.json"
    preflight_path = REPORT_DIR / "v4_pointnetpp_matrix_preflight.json"
    archive_path = REPORT_DIR / "v4_archive_space_inventory.json"
    split_path = ROOT / "training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
    stats_path = ROOT / "data_wss_min/fold_stats/v4/AG_AAA_v4_stratified_train106_global_stats.json"
    runner = ROOT / "training_wss_min/cluster/run_pointnet_distribution_matrix.slurm"

    acceptance = read_json(acceptance_path)
    preflight = read_json(preflight_path)
    split = read_json(split_path)
    stats = read_json(stats_path)
    archive = read_json(archive_path) if archive_path.exists() else None
    subprocess.run(["bash", "-n", str(runner)], check=True)

    preflight_by_config = {Path(row["config"]).resolve(): row for row in preflight["jobs"]}
    configs = []
    for path in CONFIGS:
        cfg = ExpConfig.from_json(path)
        row = preflight_by_config.get(path.resolve())
        configs.append({
            "path": str(path),
            "sha256": sha256(path),
            "model": cfg.model.name,
            "split_path": str(Path(cfg.data.split_path)),
            "stats_path": str(Path(cfg.data.wss_stats_path)),
            "run_dir": str(cfg.run_dir),
            "run_dir_absent": not cfg.run_dir.exists(),
            "epochs": cfg.train.epochs,
            "seed": cfg.train.seed,
            "sampling": cfg.data.sampling,
            "sample_points": cfg.data.wall_n_points,
            "selection_metric": cfg.train.ckpt_metric,
            "selection_rule": cfg.train.selection_rule,
            "preflight_passed": bool(row and row["cuda_amp_forward_backward"] == "passed"
                                     and row["cpu_two_case_smoke"]["forward_backward"] == "passed"),
            "train_cases": row["train_cases"] if row else None,
            "test_cases": row["test_cases"] if row else None,
        })

    exclusions = set(split["excluded_cases"])
    partitions = set(split["train_cases"] + split["val_cases"] + split["test_cases"])
    gates = {
        "completed_jobs_accepted": acceptance["passed"],
        "matrix_preflight": preflight["status"],
        "split_counts": {
            "train": len(split["train_cases"]),
            "val": len(split["val_cases"]),
            "test": len(split["test_cases"]),
            "strata": {key: value for key, value in split["counts"].items()
                       if key not in {"train", "val", "test"}},
        },
        "split_exclusion_leak_count": len(exclusions & partitions),
        "stats_train_manifest_exact": stats["train_units"] == split["train_cases"],
        "stats_n_cases": stats["n_cases"],
        "stats_sha256": sha256(stats_path),
        "slurm_runner_bash_n": "passed",
        "all_formal_run_dirs_absent": all(row["run_dir_absent"] for row in configs),
        "all_cpu_cuda_preflights_passed": all(row["preflight_passed"] for row in configs),
    }
    ready = (
        gates["completed_jobs_accepted"]
        and gates["matrix_preflight"] == "passed"
        and gates["split_counts"]["train"] == 106
        and gates["split_counts"]["val"] == 0
        and gates["split_counts"]["test"] == 27
        and gates["split_exclusion_leak_count"] == 0
        and gates["stats_train_manifest_exact"]
        and gates["all_formal_run_dirs_absent"]
        and gates["all_cpu_cuda_preflights_passed"]
    )
    output = REPORT_DIR / "v4_next_experiment_readiness.json"
    prior = read_json(output) if output.exists() else {}
    submission = prior.get("submission") if prior.get("submission_performed") else None
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitted_jobs_9167_9170" if submission else (
            "ready_for_submission_after_user_confirmation" if ready else "not_ready"
        ),
        "submission_performed": bool(submission),
        "training_started": bool(submission),
        "protocol": {
            "new_split": str(split_path),
            "new_split_sha256": sha256(split_path),
            "old_ag_test15_locked": False,
            "old_ag_test15_overlap": split["legacy_AG_test15_overlap"],
            "duplicate_group_policy": {
                "duplicate_geometry_groups": split["duplicate_geometry_groups"],
                "exact_duplicate_geometry_groups": split["exact_duplicate_geometry_groups"],
                "explicit_related_groups": split["explicit_related_groups"],
            },
            "statistics": str(stats_path),
            "statistics_point_contribution": stats["point_contribution_by_cohort"],
            "comparability": "new test27 is an independent protocol; do not directly compare its aggregate metrics with common-test15",
        },
        "gates": gates,
        "configs": configs,
        "archive_inventory": {
            "available": archive is not None,
            "path": str(archive_path),
            "status": archive["status"] if archive else "still_running_or_not_generated",
            "estimated_release": archive[
                "estimated_release_after_verified_archive_and_confirmed_deletion"
            ] if archive else None,
        },
        "next_action": (
            "monitor Jobs 9167-9170; accept best/last evaluation and PostView only after each full pipeline completes"
            if submission else "wait for explicit user approval of the split and experiment matrix before sbatch"
        ),
    }
    if submission:
        payload["submission"] = submission
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": payload["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
