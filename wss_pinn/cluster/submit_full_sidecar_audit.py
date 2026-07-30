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

from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    git_state,
    guard_write_path,
    sha256_file,
    utc_now,
)


SBATCH = Path("/public/slurm/bin/sbatch")
SCRIPT = ROOT / "wss_pinn/cluster/audit_full_sidecars.slurm"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dependency-job-id")
    parser.add_argument(
        "--python", default="/public/newhome/cy/.conda/envs/GNN/bin/python"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    split = Path(args.split).resolve()
    if not split.is_file():
        raise FileNotFoundError(split)
    if not manifest.is_file() and not args.dependency_job_id:
        raise FileNotFoundError(manifest)
    output_dir = guard_write_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"sidecar audit output is not empty: {output_dir}")

    command = [
        str(SBATCH),
        "--parsable",
        "--partition=CPU",
        "--time=04:00:00",
        "--mem=24G",
        "--cpus-per-task=2",
        "--job-name=PINN-full-sidecar-audit",
        f"--output={output_dir}/slurm_%j.out",
        f"--error={output_dir}/slurm_%j.err",
    ]
    if args.dependency_job_id:
        command.append(f"--dependency=afterok:{args.dependency_job_id}")
    command.extend(
        [
            (
                "--export=ALL,"
                f"WSS_PINN_MANIFEST={manifest},"
                f"WSS_PINN_SPLIT={split},"
                f"WSS_PINN_AUDIT_OUTPUT={output_dir},"
                f"WSS_PINN_PYTHON={Path(args.python).resolve()}"
            ),
            str(SCRIPT),
        ]
    )
    record = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "code_ready" if args.dry_run else "submitting",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
        "split": str(split),
        "split_sha256": sha256_file(split),
        "dependency_job_id": args.dependency_job_id,
        "output_dir": str(output_dir),
        "git": git_state(),
        "command": command,
    }
    if args.dry_run:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=False)
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
                f"tail -f {output_dir}/slurm_{job_id}.out",
            ],
        }
    )
    output = atomic_write_json(output_dir / "submission.json", record)
    print(
        json.dumps(
            {"status": "submitted", "job_id": job_id, "output": str(output)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
