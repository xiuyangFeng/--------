#!/usr/bin/env python3
"""Submit the eight-arm REG-P10 local/global Transformer matrix."""

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
MANIFEST = PREFLIGHT / "regp10_transformer_matrix_20260726_prepared.json"
STATIC = PREFLIGHT / "regp10_transformer_matrix_20260726_static_audit.json"
OUTPUT = PREFLIGHT / "regp10_transformer_matrix_20260726_submission.json"
SBATCH = Path("/public/slurm/bin/sbatch")
GATE = ROOT / "training_wss_min/cluster/preflight_regp10_transformer_matrix.slurm"
RUN = ROOT / "training_wss_min/cluster/run_regp10_transformer_matrix.slurm"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submit(*args: str) -> str:
    proc = subprocess.run(
        [str(SBATCH), "--parsable", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "sbatch failed")
    return proc.stdout.strip().split(";", 1)[0]


def dump(path: Path, payload: dict) -> None:
    path.write_text(
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
    expected_subsets = {
        (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)
    }
    local_subsets = {
        tuple(row["local_transformer_stages"])
        for row in rows
        if row["local_transformer_stages"]
    }
    global_rows = [row for row in rows if row["coarse_attention"]]
    if (
        manifest.get("status") != "prepared"
        or len(rows) != 8
        or {row["seed"] for row in rows} != {1234}
        or local_subsets != expected_subsets
        or len(global_rows) != 1
    ):
        raise RuntimeError("expected exact 7-local-subset + 1-global matrix")
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    if (
        static.get("status") != "passed"
        or static.get("manifest_sha256") != sha(MANIFEST)
    ):
        raise RuntimeError("static audit missing, failed, or stale")
    for row in rows:
        config_path = Path(row["config"])
        if sha(config_path) != row["sha256"]:
            raise RuntimeError(f"config drift: {config_path}")
        if ExpConfig.from_json(config_path).run_dir.exists():
            raise RuntimeError(f"run collision: {config_path}")

    out = OUTPUT.with_suffix(".dryrun.json") if args.dry_run else OUTPUT
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitting",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha(MANIFEST),
        "static_audit": str(STATIC.resolve()),
        "static_audit_sha256": sha(STATIC),
        "policy": (
            "REG-P10 seed1234; seven local SA-stage subsets plus one existing "
            "SA3 coarse-global control; GPU gate then array 0-7%4"
        ),
        "tasks": 8,
        "array": "0-7%4",
        "jobs": [],
    }
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            export = f"ALL,MATRIX_MANIFEST={MANIFEST}"
            gate = submit(
                "--job-name=regp10tf_pf",
                f"--export={export}",
                str(GATE),
            )
            array = submit(
                "--job-name=regp10tf",
                f"--export={export}",
                f"--dependency=afterok:{gate}",
                "--array=0-7%4",
                str(RUN),
            )
            payload["jobs"] = [
                {"role": "formal_gate", "job_id": gate},
                {
                    "role": "training_and_best_last_eval",
                    "job_id": array,
                    "array": "0-7%4",
                    "dependency": f"afterok:{gate}",
                },
            ]
            payload["status"] = "submitted"
            payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
        dump(out, payload)
    except Exception as exc:
        payload["status"] = "partial_failed" if payload["jobs"] else "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        dump(out, payload)
        raise
    print(
        json.dumps(
            {"status": payload["status"], "jobs": payload["jobs"], "output": str(out)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
