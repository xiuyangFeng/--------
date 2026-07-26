#!/usr/bin/env python3
"""One-time recovery submission after the original D2 GPU smoke harness failure."""
from __future__ import annotations

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
OUTPUT = PREFLIGHT / "d2_c125_k64_pnxr_geope_20260726_resubmission.json"
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
    if OUTPUT.exists():
        raise FileExistsError(f"recovery submission already recorded: {OUTPUT}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared" or static.get("status") != "passed" or static.get("manifest_sha256") != sha(MANIFEST):
        raise RuntimeError("prepared/static contract failed")
    row = manifest["configs"][0]
    config = Path(row["config"])
    if sha(config) != row["sha256"] or ExpConfig.from_json(config).run_dir.exists():
        raise RuntimeError("config drift or run collision")
    export = f"ALL,MATRIX_MANIFEST={MANIFEST}"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitting",
        "recovery_reason": "10956 failed only because GPU smoke assumed feature_stats_path; D2 runtime legitimately recomputes train106 feature statistics",
        "replaces": {"formal_gate": "10956", "dependent_training": "10957"},
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha(MANIFEST),
        "jobs": [],
    }
    try:
        gate = submit("--job-name=d2c125geo_pf_r1", f"--export={export}", str(GATE))
        train = submit("--job-name=d2c125geo_r1", f"--export={export}", f"--dependency=afterok:{gate}", str(RUN))
        payload["jobs"] = [
            {"role": "formal_gate", "job_id": gate},
            {"role": "training_and_best_last_eval", "job_id": train, "dependency": f"afterok:{gate}"},
        ]
        payload["status"] = "submitted"
        payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    print(json.dumps({"status": payload["status"], "jobs": payload["jobs"], "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
