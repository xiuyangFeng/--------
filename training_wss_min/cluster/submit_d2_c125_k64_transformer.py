#!/usr/bin/env python3
"""Submit SA1/SA2 local-Transformer controls on the fixed D2 PNXR-GeoPE anchor."""
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
MANIFEST = (
    PREFLIGHT / "d2_c125_k64_pnxr_geope_transformer_20260726_prepared.json"
)
STATIC = (
    PREFLIGHT / "d2_c125_k64_pnxr_geope_transformer_20260726_static_audit.json"
)
OUTPUT = (
    PREFLIGHT / "d2_c125_k64_pnxr_geope_transformer_20260726_submission.json"
)
SBATCH = Path("/public/slurm/bin/sbatch")
GATE = (
    ROOT / "training_wss_min/cluster/preflight_d2_c125_k64_transformer.slurm"
)
RUN = ROOT / "training_wss_min/cluster/run_d2_c125_k64_transformer.slurm"


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
    stages = {tuple(row["local_transformer_stages"]) for row in rows}
    if (
        manifest.get("status") != "prepared"
        or len(rows) != 2
        or stages != {(1,), (2,)}
        or {row["seed"] for row in rows} != {1234}
        or any(row["coarse_attention"] for row in rows)
    ):
        raise RuntimeError("expected exact D2 local-Transformer SA1/SA2 matrix")
    static = json.loads(STATIC.read_text(encoding="utf-8"))
    if (
        static.get("status") != "passed"
        or static.get("manifest_sha256") != sha(MANIFEST)
    ):
        raise RuntimeError("static audit missing, failed, or stale")
    for row in rows:
        config_path = Path(row["config"])
        cfg = ExpConfig.from_json(config_path)
        if sha(config_path) != row["sha256"]:
            raise RuntimeError(f"config drift: {config_path}")
        if cfg.run_dir.exists():
            raise RuntimeError(f"run collision: {cfg.run_dir}")
        if (
            tuple(cfg.model.sa_blocks) != (1, 1, 0)
            or not cfg.model.local_geope
            or tuple(cfg.model.local_transformer_stages)
            not in {(1,), (2,)}
        ):
            raise RuntimeError(f"D2 PNXR-GeoPE architecture drift: {config_path}")

    output = OUTPUT.with_suffix(".dryrun.json") if args.dry_run else OUTPUT
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitting",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha(MANIFEST),
        "static_audit": str(STATIC.resolve()),
        "static_audit_sha256": sha(STATIC),
        "policy": (
            "Fixed D2 c125×k64 PNXR+7D LocalGeoPE 106/0/27 seed1234; "
            "only local Transformer SA1 or SA2; GPU gate then array 0-1%2"
        ),
        "tasks": 2,
        "array": "0-1%2",
        "jobs": [],
    }
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            export = f"ALL,MATRIX_MANIFEST={MANIFEST}"
            gate = submit(
                "--job-name=d2tf_pf",
                f"--export={export}",
                str(GATE),
            )
            array = submit(
                "--job-name=d2tf",
                f"--export={export}",
                f"--dependency=afterok:{gate}",
                "--array=0-1%2",
                str(RUN),
            )
            payload["jobs"] = [
                {"role": "formal_gate", "job_id": gate},
                {
                    "role": "training_and_best_last_eval",
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
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
