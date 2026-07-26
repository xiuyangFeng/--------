#!/usr/bin/env python3
"""Submit the S3-GEOPE root-cause matrix: one formal gate + one training array.

Mirrors the D2-K64 three-seed submission (gate then afterok array), generalised
to the N prepared tasks in the manifest.  Writes an auditable submission.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
SBATCH = Path("/public/slurm/bin/sbatch")
PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "s3_rootcause_matrix_prepared.json"
OUTPUT = PREFLIGHT / "s3_rootcause_matrix_submission.json"
GATE = ROOT / "training_wss_min/cluster/preflight_s3_rootcause_matrix.slurm"
RUN = ROOT / "training_wss_min/cluster/run_s3_rootcause_matrix.slurm"
MAX_CONCURRENT = 6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sbatch(*args: str) -> str:
    result = subprocess.run([str(SBATCH), "--parsable", *args], cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    return result.stdout.strip().split(";", 1)[0]


def dump(payload: dict) -> None:
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--existing-gate", help="reuse an already submitted formal gate job")
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing duplicate submission: {OUTPUT}")
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or not rows:
        raise RuntimeError("manifest not prepared or empty")
    for row in rows:
        cfg = Path(row["config"])
        if sha(cfg) != row["sha256"]:
            raise RuntimeError(f"config sha drift: {cfg}")
        if ExpConfig.from_json(cfg).run_dir.exists():
            raise RuntimeError(f"run_dir already exists: {cfg}")
    n = len(rows)
    array_spec = f"0-{n - 1}%{MAX_CONCURRENT}"
    payload = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "submitting",
        "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha(MANIFEST),
        "policy": "S3-GEOPE root-cause matrix; frozen split; single-variable arms; one gate; afterok array; no early cancellation",
        "tasks": n, "array": array_spec, "jobs": [],
    }
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            export = f"ALL,MATRIX_MANIFEST={MANIFEST}"
            gate = args.existing_gate or sbatch("--job-name=s3rc_pf", f"--export={export}", str(GATE))
            array = sbatch("--job-name=s3rc", f"--export={export}", f"--dependency=afterok:{gate}", f"--array={array_spec}", str(RUN))
            payload["jobs"] = [
                {"role": "formal_gate", "job_id": gate, "reused": bool(args.existing_gate)},
                {"role": "training_and_best_last_eval", "job_id": array, "array": array_spec, "dependency": f"afterok:{gate}"},
            ]
            payload["status"] = "submitted"
            payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        dump(payload)
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        dump(payload)
        raise
    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"], "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
