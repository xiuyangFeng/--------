#!/usr/bin/env python3
"""Submit the fixed-split D2 c125×k64 PointNeXt-R + LocalGeoPE control."""
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
MANIFEST = PREFLIGHT / "d2_c125_k64_pnxr_geope_20260726_prepared.json"
STATIC = PREFLIGHT / "d2_c125_k64_pnxr_geope_20260726_static_audit.json"
OUTPUT = PREFLIGHT / "d2_c125_k64_pnxr_geope_20260726_submission.json"
SBATCH = Path("/public/slurm/bin/sbatch")
GATE = ROOT / "training_wss_min/cluster/preflight_d2_c125_k64_pnxr_geope.slurm"
RUN = ROOT / "training_wss_min/cluster/run_d2_c125_k64_pnxr_geope.slurm"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit(*args: str) -> str:
    proc = subprocess.run([str(SBATCH), "--parsable", *args], cwd=ROOT, text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "sbatch failed")
    return proc.stdout.strip().split(";", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing duplicate submission: {OUTPUT}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or len(rows) != 1 or static.get("status") != "passed" or static.get("manifest_sha256") != sha(MANIFEST):
        raise RuntimeError("prepared/static contract failed")
    row = rows[0]
    cfg = Path(row["config"])
    if sha(cfg) != row["sha256"] or ExpConfig.from_json(cfg).run_dir.exists():
        raise RuntimeError("config drift or run collision")
    output = OUTPUT.with_suffix(".dryrun.json") if args.dry_run else OUTPUT
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitting",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha(MANIFEST),
        "static_audit": str(STATIC.resolve()),
        "policy": "D2 c125×k64 fixed 106/0/27; only PointNeXt-R (1,1,0) plus 7D LocalGeoPE; gate then 400-epoch train/best-last test27",
        "jobs": [],
    }
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            export = f"ALL,MATRIX_MANIFEST={MANIFEST}"
            gate = submit("--job-name=d2c125geo_pf", f"--export={export}", str(GATE))
            train = submit("--job-name=d2c125geo", f"--export={export}", f"--dependency=afterok:{gate}", str(RUN))
            payload["jobs"] = [
                {"role": "formal_gate", "job_id": gate},
                {"role": "training_and_best_last_eval", "job_id": train, "dependency": f"afterok:{gate}"},
            ]
            payload["status"] = "submitted"
            payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"], "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
