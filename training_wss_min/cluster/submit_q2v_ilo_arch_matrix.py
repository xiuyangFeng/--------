#!/usr/bin/env python3
"""Hash-gated, partial-submission-safe launcher for the approved Q2V matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


REPO = Path(__file__).resolve().parents[2]
SBATCH = Path("/public/slurm/bin/sbatch")
PREPARED = REPO / "training_wss_min/preflight/q2v_ilo_arch_matrix_prepared.json"
PREFLIGHT = REPO / "training_wss_min/preflight/q2v_ilo_arch_matrix_gpu_preflight.json"
OUTPUT = REPO / "training_wss_min/preflight/q2v_ilo_arch_matrix_submission.json"
CLUSTER = REPO / "training_wss_min/cluster"

DESIGN = {
    "q2v_ilo_d1_fixed_frozen": ("Q2V-10477", "add ILO-before41 to train; freeze Q2V target and feature stats"),
    "q2v_ilo_d1_fixed_refit": ("D1 fixed frozen", "same train147/test27; refit target and feature stats on train147"),
    "q2v_ilo_d2_extended_frozen": ("Q2V-10477", "add ILO32 to train and ILO9 to test; freeze Q2V stats"),
    "q2v_ilo_d3_pool2025_control": ("Q2V architecture and budget", "new pool2025 test36; AG/AAA train106; ILO32 unused"),
    "q2v_ilo_d3_pool2025_frozen": ("D3 pool2025 control", "add matched ILO32 train units; freeze D3-control stats"),
    "q2v_ilo_d3_pool2025_refit": ("D3 pool2025 frozen", "same train138/test36; refit target and feature stats on train138"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def submit(script: Path, *extra: str) -> str:
    proc = subprocess.run(
        [str(SBATCH), "--parsable", *extra, str(script)],
        cwd=REPO, text=True, capture_output=True,
    )
    if proc.returncode:
        raise RuntimeError(f"sbatch failed for {script.name}: {proc.stderr.strip()}")
    return proc.stdout.strip().split(";", 1)[0]


def config_record(row: dict, task_id: int) -> dict:
    config_path = Path(row["config"])
    cfg = ExpConfig.from_json(config_path)
    split_path = Path(cfg.data.split_path)
    stats_path = Path(cfg.data.wss_stats_path)
    if row["family"] == "architecture":
        n = int(cfg.model.sa_nsample[0]); w = int(cfg.model.width)
        control = "Q2V architecture n16/w32 on the same dev85/val21 protocol"
        unique = f"factor cell nsample={n}, width={w}; all other Q2V settings frozen"
    else:
        control, unique = DESIGN[row["experiment_id"]]
    return {
        "array_task_id": task_id,
        "experiment_id": row["experiment_id"],
        "family": row["family"],
        "config": str(config_path),
        "config_sha256": sha(config_path),
        "split": str(split_path),
        "split_sha256": sha(split_path),
        "target_stats": str(stats_path),
        "target_stats_sha256": sha(stats_path),
        "feature_stats": cfg.data.feature_stats_path,
        "feature_stats_sha256": sha(Path(cfg.data.feature_stats_path)) if cfg.data.feature_stats_path else None,
        "control_group": control,
        "unique_change": unique,
        "expected_run_dir": str(cfg.run_dir),
        "surface_metric_mode": cfg.eval.surface_metric_mode,
        "train_seed": cfg.train.seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", type=Path, default=PREPARED)
    ap.add_argument("--preflight", type=Path, default=PREFLIGHT)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    if not SBATCH.is_file():
        raise FileNotFoundError(SBATCH)
    prepared = load(args.prepared)
    preflight = load(args.preflight)
    if prepared.get("status") != "prepared" or len(prepared.get("configs", [])) != 12:
        raise RuntimeError("prepared manifest is not the exact 12-config matrix")
    if preflight.get("status") != "passed" or len(preflight.get("jobs", [])) != 12:
        raise RuntimeError("formal GPU preflight did not pass all 12 configs")

    prepared_hashes = {row["config"]: row["sha256"] for row in prepared["configs"]}
    preflight_hashes = {row["config"]: row["config_sha256"] for row in preflight["jobs"]}
    if prepared_hashes != preflight_hashes:
        raise RuntimeError("preflight config set/hash differs from prepared manifest")
    for path_text, expected in prepared_hashes.items():
        if sha(Path(path_text)) != expected:
            raise RuntimeError(f"config changed after preflight: {path_text}")

    data_rows = [row for row in prepared["configs"] if row["family"] == "data"]
    arch_rows = [row for row in prepared["configs"] if row["family"] == "architecture"]
    if len(data_rows) != 6 or len(arch_rows) != 6:
        raise RuntimeError("matrix family cardinality changed")
    configs = [config_record(row, i) for rows in (data_rows, arch_rows)
               for i, row in enumerate(rows)]
    existing = [row["expected_run_dir"] for row in configs if Path(row["expected_run_dir"]).exists()]
    if existing:
        raise FileExistsError(f"refuse duplicate submission; run dirs already exist: {existing}")

    scripts = {
        "data_array": CLUSTER / "run_q2v_ilo_data_matrix.slurm",
        "architecture_array": CLUSTER / "run_q2v_arch_dev_matrix.slurm",
        "q2v_export_only": CLUSTER / "run_q2v_10477_postview_export_only.slurm",
        "q2v_zero_shot_test36": CLUSTER / "run_q2v_10477_extended_zero_shot_eval.slurm",
    }
    for path in scripts.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = {
        "schema_version": 1,
        "created_at": now(),
        "status": "submitting",
        "prepared_manifest": str(args.prepared.resolve()),
        "prepared_manifest_sha256": sha(args.prepared),
        "gpu_preflight": str(args.preflight.resolve()),
        "gpu_preflight_sha256": sha(args.preflight),
        "config_hash_gate": "passed",
        "run_dir_absence_gate": "passed",
        "max_array_concurrency": 4,
        "jobs": [],
        "configs": configs,
        "deferred": "architecture winner test27 confirmation remains unsubmitted",
        "frozen": "AreaRandom six-run phase remains unsubmitted and unmixed",
    }
    atomic_write(args.output, payload)
    try:
        data_job = submit(scripts["data_array"], "--array=0-5%4")
        payload["jobs"].append({
            "role": "data_training_array", "job_id": data_job, "array": "0-5%4",
            "script": str(scripts["data_array"]), "script_sha256": sha(scripts["data_array"]),
            "tasks": [{"task_id": i, "experiment_id": row["experiment_id"]}
                      for i, row in enumerate(data_rows)],
        })
        atomic_write(args.output, payload)

        arch_job = submit(scripts["architecture_array"], "--array=0-5%4")
        payload["jobs"].append({
            "role": "architecture_dev_array_val_only", "job_id": arch_job, "array": "0-5%4",
            "script": str(scripts["architecture_array"]),
            "script_sha256": sha(scripts["architecture_array"]),
            "tasks": [{"task_id": i, "experiment_id": row["experiment_id"]}
                      for i, row in enumerate(arch_rows)],
        })
        atomic_write(args.output, payload)

        export_job = submit(scripts["q2v_export_only"])
        payload["jobs"].append({
            "role": "q2v_10477_postview_export_only_27", "job_id": export_job,
            "script": str(scripts["q2v_export_only"]),
            "script_sha256": sha(scripts["q2v_export_only"]),
        })
        atomic_write(args.output, payload)

        zero_job = submit(scripts["q2v_zero_shot_test36"])
        payload["jobs"].append({
            "role": "q2v_10477_zero_shot_extended_test36", "job_id": zero_job,
            "script": str(scripts["q2v_zero_shot_test36"]),
            "script_sha256": sha(scripts["q2v_zero_shot_test36"]),
        })
        payload["status"] = "submitted"
        payload["submitted_at"] = now()
        atomic_write(args.output, payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["failed_at"] = now()
        payload["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write(args.output, payload)
        raise

    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"],
                      "manifest": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
