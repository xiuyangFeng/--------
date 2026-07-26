#!/usr/bin/env python3
"""Submit the four-task single-seed D2-K64 structure wave-2 matrix."""

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
MANIFEST = REPO / "training_wss_min/preflight/d2_k64_ilo_structure_wave2_prepared.json"
STATIC_AUDIT = REPO / "training_wss_min/preflight/d2_k64_ilo_structure_wave2_static_audit.json"
OUTPUT = REPO / "training_wss_min/preflight/d2_k64_ilo_structure_wave2_submission.json"
GATE_SCRIPT = REPO / "training_wss_min/cluster/preflight_d2_k64_structure_wave2.slurm"
TRAIN_SCRIPT = REPO / "training_wss_min/cluster/run_d2_k64_structure_wave2.slurm"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit(*args: str) -> str:
    proc = subprocess.run(
        [str(SBATCH), "--parsable", *args],
        cwd=REPO, text=True, capture_output=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "sbatch failed")
    return proc.stdout.strip().split(";", 1)[0]


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing duplicate submission record: {OUTPUT}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    static = json.loads(STATIC_AUDIT.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or len(rows) != 4:
        raise RuntimeError("wave2 manifest is not the exact four-task matrix")
    if static.get("status") != "passed" or static.get("manifest_sha256") != sha(MANIFEST):
        raise RuntimeError("static audit is missing, failed, or stale")
    if {row["seed"] for row in rows} != {1234}:
        raise RuntimeError("wave2 contains a non-1234 seed")
    for row in rows:
        path = Path(row["config"])
        if not path.is_file() or sha(path) != row["sha256"]:
            raise RuntimeError(f"missing or drifted config: {path}")
        if ExpConfig.from_json(path).run_dir.exists():
            raise FileExistsError(f"run dir already exists: {row['run_dir']}")
    for path in (SBATCH, GATE_SCRIPT, TRAIN_SCRIPT):
        if not path.is_file():
            raise FileNotFoundError(path)

    export = f"ALL,MATRIX_MANIFEST={MANIFEST}"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitting",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha(MANIFEST),
        "static_audit": str(STATIC_AUDIT.resolve()),
        "static_audit_sha256": sha(STATIC_AUDIT),
        "policy": (
            "four new seed1234 jobs on mixed138/test36; S3, S4, and matched "
            "independent-query S5C/S5; one common gate; array 0-3 concurrent"
        ),
        "jobs": [],
    }
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            gate = submit(
                "--job-name=d2k64w2_pf", f"--export={export}", str(GATE_SCRIPT)
            )
            payload["jobs"].append({"role": "formal_gate", "job_id": gate})
            train = submit(
                "--job-name=d2k64w2", f"--export={export}",
                f"--dependency=afterok:{gate}", "--array=0-3%4", str(TRAIN_SCRIPT),
            )
            payload["jobs"].append({
                "role": "training_and_eval",
                "job_id": train,
                "array": "0-3%4",
                "dependency": f"afterok:{gate}",
            })
            payload["status"] = "submitted"
            payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        dump(OUTPUT, payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        dump(OUTPUT, payload)
        raise
    print(json.dumps({
        "status": payload["status"],
        "jobs": payload["jobs"],
        "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
