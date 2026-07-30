#!/usr/bin/env python3
"""Submit the same-seed LSA2-H2 log(local_radius) matrix."""

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
MANIFEST = PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_prepared.json"
STATIC = PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_static_audit.json"
GPU_AUDIT = PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_gpu_smoke.json"
SUBMISSION = PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_submission.json"
SBATCH = Path("/public/slurm/bin/sbatch")
GATE = (
    ROOT
    / "training_wss_min/cluster/preflight_lsa2_h2_logradius_matrix.slurm"
)
RUN = ROOT / "training_wss_min/cluster/run_lsa2_h2_logradius_matrix.slurm"


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
        or [row["role"] for row in rows]
        != ["concurrent_control", "logradius_treatment"]
        or [row["input_dim"] for row in rows] != [6, 7]
        or static.get("status") != "passed"
        or static.get("manifest_sha256") != sha256(MANIFEST)
    ):
        raise RuntimeError("LSA2-H2 log-radius manifest contract failed")
    for row in rows:
        path = Path(row["config"])
        feature_stats = Path(row["feature_stats"])
        if sha256(path) != row["config_sha256"]:
            raise RuntimeError(f"config drift: {path}")
        if sha256(feature_stats) != row["feature_stats_sha256"]:
            raise RuntimeError(f"feature-stat drift: {feature_stats}")
        cfg = ExpConfig.from_json(path)
        if cfg.run_dir.exists():
            raise RuntimeError(f"run collision: {cfg.run_dir}")
        if (
            cfg.data.query_mode != "same"
            or cfg.train.seed != 1234
            or float(cfg.train.loss_pinball_lambda) != 0.20
            or tuple(cfg.model.local_transformer_stages) != (2,)
            or tuple(cfg.model.local_geope_feature_indices) != (3, 4, 5)
            or float(cfg.model.drop_path_rate) != 0.10
            or cfg.data.case_features_path is not None
        ):
            raise RuntimeError(f"frozen H2 anchor contract drift: {path}")

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
            "REG-P10-LSA2 SAME H2 seed1234 concurrent reproduction versus "
            "one added log(local_radius) input; mixed 138/0/36; legacy_vertex; "
            "no RCR, area, H1 combination, or multi-seed"
        ),
        "jobs": [],
    }
    try:
        if args.dry_run:
            payload["status"] = "dry_run"
        else:
            export = (
                f"ALL,LSA2_H2_LOG_MANIFEST={MANIFEST},"
                f"LSA2_H2_LOG_STATIC_AUDIT={STATIC},"
                f"LSA2_H2_LOG_GPU_AUDIT={GPU_AUDIT}"
            )
            gate = submit(
                "--job-name=lsa2h2log_pf",
                f"--export={export}",
                str(GATE),
            )
            array = submit(
                "--job-name=lsa2h2log",
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
