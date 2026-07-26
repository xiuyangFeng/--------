#!/usr/bin/env python3
"""Submit the exact three-arm D2-K64 xyz-only ablation."""

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
MANIFEST = REPO / "training_wss_min/preflight/xyz_input_ablation_20260724_prepared.json"
STATIC_AUDIT = REPO / "training_wss_min/preflight/xyz_input_ablation_20260724_static_audit.json"
OUTPUT = REPO / "training_wss_min/preflight/xyz_input_ablation_20260724_submission.json"
GATE_SCRIPT = REPO / "training_wss_min/cluster/preflight_xyz_input_ablation.slurm"
TRAIN_SCRIPT = REPO / "training_wss_min/cluster/run_xyz_input_ablation.slurm"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit(*args: str) -> str:
    proc = subprocess.run([str(SBATCH), "--parsable", *args], cwd=REPO, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "sbatch failed")
    return proc.stdout.strip().split(";", 1)[0]


def dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing duplicate submission record: {OUTPUT}")
    output = OUTPUT.with_suffix(".dryrun.json") if args.dry_run else OUTPUT
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    static = json.loads(STATIC_AUDIT.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or len(rows) != 3:
        raise RuntimeError("manifest is not the exact three-arm xyz-only matrix")
    if static.get("status") != "passed" or static.get("manifest_sha256") != sha(MANIFEST):
        raise RuntimeError("static audit is missing, failed, or stale")
    for row in rows:
        config = Path(row["config"])
        if sha(config) != row["sha256"] or ExpConfig.from_json(config).run_dir.exists():
            raise RuntimeError(f"config drift or existing run: {config}")
    payload = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
               "status": "submitting", "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha(MANIFEST),
               "static_audit": str(STATIC_AUDIT.resolve()), "static_audit_sha256": sha(STATIC_AUDIT),
               "policy": "three xyz-only parent controls; GPU gate then 0-2%3 training array", "jobs": []}
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            export = f"ALL,MATRIX_MANIFEST={MANIFEST}"
            gate = submit("--job-name=d2xyz_pf", f"--export={export}", str(GATE_SCRIPT))
            payload["jobs"].append({"role": "formal_gate", "job_id": gate})
            train = submit("--job-name=d2xyz", f"--export={export}", f"--dependency=afterok:{gate}", "--array=0-2%3", str(TRAIN_SCRIPT))
            payload["jobs"].append({"role": "training_and_eval", "job_id": train, "array": "0-2%3", "dependency": f"afterok:{gate}"})
            payload["status"] = "submitted"
            payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        dump(output, payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        dump(output, payload)
        raise
    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"], "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
