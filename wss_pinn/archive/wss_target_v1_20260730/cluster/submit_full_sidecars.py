#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wss_pinn.config import ExperimentConfig
from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    git_state,
    guard_write_path,
    sha256_file,
    utc_now,
)


SBATCH = Path("/public/slurm/bin/sbatch")
SCRIPT = ROOT / "wss_pinn/cluster/build_full_sidecars.slurm"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-report", required=True)
    parser.add_argument("--dependency-job-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config)
    if config.stage != "p1":
        raise ValueError("full sidecar builder requires a P1 config")
    audit_report = Path(args.audit_report).resolve()
    split_sha256 = sha256_file(config["data"]["split_path"])
    if audit_report.exists():
        audit = json.loads(audit_report.read_text(encoding="utf-8"))
        if audit.get("gate_result") != "pass":
            raise RuntimeError(f"source-data audit failed: {audit_report}")
        if audit.get("split", {}).get("sha256") != split_sha256:
            raise RuntimeError(
                "source-data audit split hash does not match the P1 config"
            )
        audit_sha256 = sha256_file(audit_report)
    elif args.dependency_job_id:
        audit_sha256 = None
    else:
        raise FileNotFoundError(audit_report)

    root = guard_write_path(config["paths"]["sidecar_root"])
    manifest = guard_write_path(config["sampling"]["manifest_path"])
    if manifest.exists():
        raise FileExistsError(f"aggregate sidecar manifest already exists: {manifest}")
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError(f"sidecar root is not empty: {root}")

    command = [
        str(SBATCH),
        "--parsable",
        "--partition=CPU",
        "--time=24:00:00",
        "--mem=48G",
        "--cpus-per-task=4",
        "--job-name=PINN-full-sidecars",
        f"--output={root}/slurm_%j.out",
        f"--error={root}/slurm_%j.err",
    ]
    if args.dependency_job_id:
        command.append(f"--dependency=afterok:{args.dependency_job_id}")
    command.extend(
        [
            (
                "--export=ALL,"
                f"WSS_PINN_CONFIG={config.source},"
                f"WSS_PINN_AUDIT_REPORT={audit_report},"
                f"WSS_PINN_PYTHON={config['cluster']['python']}"
            ),
            str(SCRIPT),
        ]
    )
    record = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "code_ready" if args.dry_run else "submitting",
        "config": str(config.source),
        "config_sha256": sha256_file(config.source),
        "resolved_config_sha256": config.resolved_sha256,
        "split": config["data"]["split_path"],
        "split_sha256": split_sha256,
        "audit_report": str(audit_report),
        "audit_report_sha256": audit_sha256,
        "sidecar_root": str(root),
        "aggregate_manifest": str(manifest),
        "git": git_state(),
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return

    root.mkdir(parents=True, exist_ok=args.resume)
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip())
    job_id = process.stdout.strip().split(";", 1)[0]
    record.update(
        {
            "status": "submitted",
            "submitted_at": utc_now(),
            "job_id": job_id,
            "monitor_commands": [
                f"/public/slurm/bin/squeue -j {job_id}",
                (
                    f"/public/slurm/bin/sacct -j {job_id} "
                    "--format=JobID,State,Elapsed,ExitCode"
                ),
                f"tail -f {root}/slurm_{job_id}.out",
            ],
        }
    )
    output_name = "submission_resume.json" if args.resume else "submission.json"
    output = atomic_write_json(root / output_name, record)
    print(
        json.dumps(
            {"status": "submitted", "job_id": job_id, "output": str(output)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
