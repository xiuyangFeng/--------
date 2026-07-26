#!/usr/bin/env python3
"""Submit the exact six-cell M1/S2/S3 three-seed confirmation array."""
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
MANIFEST = PREFLIGHT / "d2_k64_ilo_structure_three_seed_confirmation_prepared.json"
STATIC = PREFLIGHT / "d2_k64_ilo_structure_three_seed_confirmation_static_audit.json"
OUTPUT = PREFLIGHT / "d2_k64_ilo_structure_three_seed_confirmation_submission.json"
GATE = ROOT / "training_wss_min/cluster/preflight_d2_k64_three_seed_confirmation.slurm"
RUN = ROOT / "training_wss_min/cluster/run_d2_k64_three_seed_confirmation.slurm"


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
    manifest, static = json.loads(MANIFEST.read_text()), json.loads(STATIC.read_text())
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or len(rows) != 6:
        raise RuntimeError("expected exact six prepared confirmation tasks")
    if static.get("status") != "passed" or static.get("manifest_sha256") != sha(MANIFEST):
        raise RuntimeError("static audit missing, failed, or stale")
    if {(r["arm"], r["seed"]) for r in rows} != {(a, s) for a in ("M1", "S2", "S3") for s in (7, 2025)}:
        raise RuntimeError("matrix drift")
    for row in rows:
        cfg = Path(row["config"])
        if sha(cfg) != row["sha256"] or ExpConfig.from_json(cfg).run_dir.exists():
            raise RuntimeError(f"config drift or existing run: {cfg}")
    payload = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "submitting",
               "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha(MANIFEST),
               "policy": "M1/S2/S3 x seeds{7,2025}; frozen split/stats; one gate; six-task array; no early cancellation", "jobs": []}
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            export = f"ALL,MATRIX_MANIFEST={MANIFEST}"
            gate = args.existing_gate or sbatch("--job-name=d2k64c_pf", f"--export={export}", str(GATE))
            array = sbatch("--job-name=d2k64c", f"--export={export}", f"--dependency=afterok:{gate}", "--array=0-5%6", str(RUN))
            payload["jobs"] = [{"role": "formal_gate", "job_id": gate, "reused": bool(args.existing_gate)}, {"role": "training_and_best_last_eval", "job_id": array, "array": "0-5%6", "dependency": f"afterok:{gate}"}]
            payload["status"] = "submitted"; payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        dump(payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"; payload["error"] = f"{type(exc).__name__}: {exc}"; dump(payload); raise
    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"], "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
