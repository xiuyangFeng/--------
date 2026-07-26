#!/usr/bin/env python3
"""Submit the single-seed S3-GEOPE + SA3 coarse-attention screen."""
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
MANIFEST = PREFLIGHT / "s3_sa3_coarse_attention_single_prepared.json"
OUTPUT = PREFLIGHT / "s3_sa3_coarse_attention_single_submission.json"
GATE = ROOT / "training_wss_min/cluster/preflight_s3_coarse_attention_single.slurm"
RUN = ROOT / "training_wss_min/cluster/run_s3_coarse_attention_single.slurm"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit(*args: str) -> str:
    result = subprocess.run(
        [str(SBATCH), "--parsable", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    return result.stdout.strip().split(";", 1)[0]


def dump(payload: dict) -> None:
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing duplicate submission: {OUTPUT}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or len(rows) != 1:
        raise RuntimeError("single-attention manifest is not prepared")
    row = rows[0]
    config = Path(row["config"])
    if sha(config) != row["sha256"]:
        raise RuntimeError(f"config sha drift: {config}")
    if ExpConfig.from_json(config).run_dir.exists():
        raise RuntimeError(f"run_dir already exists: {row['run_dir']}")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitting",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha(MANIFEST),
        "policy": (
            "one seed1234 S3-GEOPE + SA3 32-center 4-head coarse-attention "
            "screen; no Drop; formal gate then afterok train/eval"
        ),
        "jobs": [],
    }
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            export = f"ALL,MATRIX_MANIFEST={MANIFEST}"
            gate = submit(
                "--job-name=s3attn_pf", f"--export={export}", str(GATE)
            )
            train = submit(
                "--job-name=s3attn",
                f"--export={export}",
                f"--dependency=afterok:{gate}",
                str(RUN),
            )
            payload["jobs"] = [
                {"role": "formal_gate", "job_id": gate},
                {
                    "role": "training_and_best_last_eval",
                    "job_id": train,
                    "dependency": f"afterok:{gate}",
                },
            ]
            payload["status"] = "submitted"
            payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        dump(payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        dump(payload)
        raise
    print(json.dumps({
        "status": payload["status"],
        "jobs": payload["jobs"],
        "output": str(OUTPUT),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
