#!/usr/bin/env python3
"""Hash-gated submission of the bridge SEP control follow-up (SAME -> SEP only)."""

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
MANIFEST = REPO / "training_wss_min/preflight/bridge_sep_followup_prepared.json"
CONFIG = REPO / "training_wss_min/configs/pointnetpp_sa1_scale_bridge_sep_followup_20260722/bridge_rand5000_fixed500_k64_sep.json"
OUTPUT = REPO / "training_wss_min/preflight/bridge_sep_followup_submission.json"
PREFLIGHT_SCRIPT = REPO / "training_wss_min/cluster/run_bridge_sep_followup_preflight.slurm"
TRAIN_SCRIPT = REPO / "training_wss_min/cluster/run_bridge_sep_followup.slurm"


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
    if not MANIFEST.is_file() or not CONFIG.is_file() or not PREFLIGHT_SCRIPT.is_file() or not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError("follow-up manifest, config, or Slurm script is missing")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    row = manifest["configs"][0]
    if row["config"] != str(CONFIG.resolve()) or row["sha256"] != sha(CONFIG):
        raise RuntimeError("manifest config path/hash does not match follow-up config")
    cfg = ExpConfig.from_json(CONFIG)
    if cfg.run_dir.exists():
        raise FileExistsError(f"follow-up run directory already exists: {cfg.run_dir}")
    payload = {
        "schema_version": 1, "created_at": now(), "status": "submitting",
        "purpose": "bridge_rand5000_fixed500_k64 SEP control follow-up (SAME -> SEP only)",
        "test27_policy": "same-protocol engineering comparison against bridge_rand5000_fixed500_k64 "
                         "(historical test27, reused per SA1-scale matrix policy); not an independent confirmation",
        "experiment_id": "bridge_rand5000_fixed500_k64_sep", "config": str(CONFIG.resolve()),
        "config_sha256": sha(CONFIG), "run_dir": str(cfg.run_dir),
        "single_change": "query_mode: same -> independent (SAME -> SEP); all other saved settings unchanged",
        "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha(MANIFEST),
        "jobs": [],
    }
    try:
        if not args.dry_run:
            preflight_job = sbatch("--job-name", "bridgesep_pf", str(PREFLIGHT_SCRIPT))
            payload["jobs"].append({"role": "geometry_and_gpu_gate", "job_id": preflight_job,
                                    "script": str(PREFLIGHT_SCRIPT), "script_sha256": sha(PREFLIGHT_SCRIPT)})
            dump(args.output, payload)
            train_job = sbatch("--job-name", "bridgesep", f"--dependency=afterok:{preflight_job}", str(TRAIN_SCRIPT))
            payload["jobs"].append({"role": "training_test27_engineering_comparison", "job_id": train_job,
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
