#!/usr/bin/env python3
"""Hash-gated submission of the user-authorized Q2V/test27 exploratory matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


REPO = Path(__file__).resolve().parents[2]
SBATCH = Path("/public/slurm/bin/sbatch")
PREPARED = REPO / "training_wss_min/preflight/q2v_sampling_radius_test27_prepared.json"
PREFLIGHT = REPO / "training_wss_min/preflight/q2v_sampling_radius_test27_gpu_preflight.json"
OUTPUT = REPO / "training_wss_min/preflight/q2v_sampling_radius_test27_submission.json"
PREFLIGHT_SCRIPT = REPO / "training_wss_min/cluster/run_q2v_sampling_radius_test27_preflight.slurm"
TRAIN_SCRIPT = REPO / "training_wss_min/cluster/run_q2v_sampling_radius_test27_matrix.slurm"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sbatch(*args: str) -> str:
    result = subprocess.run([str(SBATCH), "--parsable", *args], cwd=REPO,
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    return result.stdout.strip().split(";", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not SBATCH.is_file():
        raise FileNotFoundError(SBATCH)
    if args.output.exists():
        raise FileExistsError(f"refusing duplicate/ambiguous submission manifest: {args.output}")
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    rows = prepared.get("configs", [])
    if prepared.get("status") != "prepared" or len(rows) != 4:
        raise RuntimeError("prepared manifest is not the exact four-config Q2V matrix")
    if any(not Path(row["config"]).is_file() or sha(Path(row["config"])) != row["sha256"] for row in rows):
        raise RuntimeError("config missing or changed after preparation")
    if any(ExpConfig.from_json(row["config"]).run_dir.exists() for row in rows):
        raise FileExistsError("at least one expected run directory already exists")
    if not PREFLIGHT_SCRIPT.is_file() or not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError("matrix Slurm script is missing")

    payload = {
        "schema_version": 1, "created_at": now(), "status": "submitting",
        "purpose": "Q2V-10477 anchored exploratory sampling/radius matrix on historical test27",
        "test27_policy": "user-authorized historical anchor; not independent confirmation",
        "prepared_manifest": str(args.prepared.resolve()), "prepared_manifest_sha256": sha(args.prepared),
        "gpu_preflight_output": str(args.preflight.resolve()),
        "configs": [{**row, "run_dir": str(ExpConfig.from_json(row["config"]).run_dir)} for row in rows],
        "jobs": [],
    }
    try:
        if args.dry_run:
            preflight_job, train_job = None, None
        else:
            preflight_job = sbatch("--job-name", "q2vsr_pf", str(PREFLIGHT_SCRIPT))
            payload["jobs"].append({"role": "formal_gpu_preflight", "job_id": preflight_job,
                                    "script": str(PREFLIGHT_SCRIPT), "script_sha256": sha(PREFLIGHT_SCRIPT)})
            dump(args.output, payload)
            train_job = sbatch("--job-name", "q2vsr", f"--dependency=afterok:{preflight_job}",
                               "--array=0-3%2", str(TRAIN_SCRIPT))
            payload["jobs"].append({"role": "training_test27_exploratory_array", "job_id": train_job,
                                    "array": "0-3%2", "dependency": f"afterok:{preflight_job}",
                                    "script": str(TRAIN_SCRIPT), "script_sha256": sha(TRAIN_SCRIPT)})
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
