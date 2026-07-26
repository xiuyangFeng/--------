#!/usr/bin/env python3
"""Hash-gated submission of the Q1V/SAME 0.6x-radius follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training_wss_min.config import ExpConfig


SBATCH = Path("/public/slurm/bin/sbatch")
CONFIG = REPO / "training_wss_min/configs/pointnetpp_q1v_radius_followup_20260719/q1v_radius_r60_same.json"
PREFLIGHT = REPO / "training_wss_min/preflight/q1v_radius_r60_same_gpu_preflight.json"
OUTPUT = REPO / "training_wss_min/preflight/q1v_radius_r60_same_submission.json"
PREFLIGHT_SCRIPT = REPO / "training_wss_min/cluster/run_q1v_radius_r60_same_preflight.slurm"
TRAIN_SCRIPT = REPO / "training_wss_min/cluster/run_q1v_radius_r60_same.slurm"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sbatch(*args: str) -> str:
    result = subprocess.run([str(SBATCH), "--parsable", *args], cwd=REPO, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    return result.stdout.strip().split(";", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not SBATCH.is_file():
        raise FileNotFoundError(SBATCH)
    if args.output.exists():
        raise FileExistsError(f"refusing duplicate/ambiguous submission manifest: {args.output}")
    if not CONFIG.is_file() or not PREFLIGHT_SCRIPT.is_file() or not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError("follow-up config or Slurm script is missing")
    cfg = ExpConfig.from_json(CONFIG)
    if cfg.run_dir.exists():
        raise FileExistsError(f"follow-up run directory already exists: {cfg.run_dir}")
    payload = {
        "schema_version": 1, "created_at": now(), "status": "submitting",
        "purpose": "Q1V-10476 SAME anchored 0.6x-radius historical-test27 follow-up",
        "test27_policy": "user-authorized historical anchor; not independent confirmation",
        "experiment_id": "q1v_radius_r60_same", "config": str(CONFIG.resolve()),
        "config_sha256": sha(CONFIG), "run_dir": str(cfg.run_dir),
        "single_change": "Q1V SAME radius-only: 60% of 0.05/0.10/0.20 -> 0.03/0.06/0.12",
        "gpu_preflight_output": str(PREFLIGHT.resolve()), "jobs": [],
    }
    try:
        if not args.dry_run:
            preflight_job = sbatch("--job-name", "q1vr60_pf", str(PREFLIGHT_SCRIPT))
            payload["jobs"].append({"role": "formal_gpu_preflight", "job_id": preflight_job,
                                    "script": str(PREFLIGHT_SCRIPT), "script_sha256": sha(PREFLIGHT_SCRIPT)})
            dump(args.output, payload)
            train_job = sbatch("--job-name", "q1vr60", f"--dependency=afterok:{preflight_job}", str(TRAIN_SCRIPT))
            payload["jobs"].append({"role": "training_test27_exploratory", "job_id": train_job,
                                    "dependency": f"afterok:{preflight_job}", "script": str(TRAIN_SCRIPT),
                                    "script_sha256": sha(TRAIN_SCRIPT)})
        payload["status"] = "dry_run" if args.dry_run else "submitted"
        payload["submitted_at"] = now()
        dump(args.output, payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["failed_at"] = now()
        dump(args.output, payload)
        raise
    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"],
                      "manifest": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
