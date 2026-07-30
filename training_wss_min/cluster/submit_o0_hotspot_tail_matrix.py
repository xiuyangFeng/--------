#!/usr/bin/env python3
"""Submit the two-arm O0 hotspot/tail objective matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_wss_min.config import ExpConfig


PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "o0_hotspot_tail_matrix_20260728_prepared.json"
STATIC = PREFLIGHT / "o0_hotspot_tail_matrix_20260728_static_audit.json"
GPU_AUDIT = PREFLIGHT / "o0_hotspot_tail_matrix_20260728_gpu_smoke.json"
SUBMISSION = PREFLIGHT / "o0_hotspot_tail_matrix_20260728_submission.json"
SBATCH = Path("/public/slurm/bin/sbatch")
GATE = ROOT / "training_wss_min/cluster/preflight_o0_hotspot_tail_matrix.slurm"
RUN = ROOT / "training_wss_min/cluster/run_o0_hotspot_tail_matrix.slurm"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def submit(*args: str) -> str:
    process = subprocess.run(
        [str(SBATCH), "--parsable", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or "sbatch failed")
    return process.stdout.strip().split(";", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = SUBMISSION.with_suffix(".dryrun.json") if args.dry_run else SUBMISSION
    if output.exists():
        raise FileExistsError(f"refusing duplicate submission: {output}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    if (
        manifest.get("status") != "prepared"
        or len(rows) != 2
        or [row["out_dim"] for row in rows] != [2, 1]
        or [row["hotspot_bce_lambda"] for row in rows] != [0.2, 0.0]
        or [row["pinball_lambda"] for row in rows] != [0.0, 0.2]
        or static.get("status") != "passed"
        or static.get("manifest_sha256") != sha256(MANIFEST)
    ):
        raise RuntimeError("O0 hotspot/tail manifest/static audit contract failed")
    for row in rows:
        config_path = Path(row["config"])
        if sha256(config_path) != row["config_sha256"]:
            raise RuntimeError(f"config drift: {config_path}")
        cfg = ExpConfig.from_json(config_path)
        if cfg.run_dir.exists():
            raise RuntimeError(f"run collision: {cfg.run_dir}")

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitting",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "static_audit": str(STATIC.resolve()),
        "static_audit_sha256": sha256(STATIC),
        "gpu_audit": str(GPU_AUDIT.resolve()),
        "policy": (
            "O0 geometry-only parent; H1 top10 hotspot BCE; H2 q90 pinball; "
            "mixed 138/0/36; seed1234; no RCR"
        ),
        "jobs": [],
    }
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            export = (
                f"ALL,O0_HOTTAIL_MANIFEST={MANIFEST},"
                f"O0_HOTTAIL_STATIC_AUDIT={STATIC},"
                f"O0_HOTTAIL_GPU_AUDIT={GPU_AUDIT}"
            )
            gate = submit(
                "--job-name=o0hottail_pf",
                f"--export={export}",
                str(GATE),
            )
            array = submit(
                "--job-name=o0hottail",
                f"--export={export}",
                f"--dependency=afterok:{gate}",
                "--array=0-1%2",
                str(RUN),
            )
            payload["jobs"] = [
                {"role": "gpu_preflight", "job_id": gate},
                {
                    "role": "training_and_best_last_test_eval",
                    "job_id": array,
                    "array": "0-1%2",
                    "dependency": f"afterok:{gate}",
                },
            ]
            payload["status"] = "submitted"
            payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        dump(output, payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        dump(output, payload)
        raise
    print(
        json.dumps(
            {
                "status": payload["status"],
                "jobs": payload["jobs"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
