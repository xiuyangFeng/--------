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
SCRIPT = ROOT / "wss_pinn/cluster/analyze_f1_matrix.slurm"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dependency-job-ids", nargs="+", required=True)
    parser.add_argument(
        "--python", default="/public/newhome/cy/.conda/envs/GNN/bin/python"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_list = Path(args.config_list).resolve()
    configs = [
        (ROOT / line.strip()).resolve()
        for line in config_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not configs:
        raise ValueError("diagnostic config list is empty")
    rows = []
    for path in configs:
        config = ExperimentConfig.from_json(path)
        if config.stage != "f1" or not config["experiment"]["diagnostic_only"]:
            raise ValueError(f"not an F1 diagnostic config: {path}")
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "resolved_sha256": config.resolved_sha256,
                "run_dir": str(config.run_dir),
            }
        )

    output_dir = guard_write_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"matrix output directory is not empty: {output_dir}")
    dependency = ":".join(str(value) for value in args.dependency_job_ids)
    command = [
        str(SBATCH),
        "--parsable",
        "--partition=CPU",
        "--time=01:00:00",
        "--mem=8G",
        "--cpus-per-task=1",
        "--job-name=PINN-F1D-matrix-analysis",
        f"--output={output_dir}/slurm_%j.out",
        f"--error={output_dir}/slurm_%j.err",
        f"--dependency=afterok:{dependency}",
        (
            "--export=ALL,"
            f"WSS_PINN_CONFIG_LIST={config_list},"
            f"WSS_PINN_MATRIX_OUTPUT={output_dir},"
            f"WSS_PINN_PYTHON={Path(args.python).resolve()}"
        ),
        str(SCRIPT),
    ]
    record = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "code_ready" if args.dry_run else "submitting",
        "config_list": str(config_list),
        "config_list_sha256": sha256_file(config_list),
        "configs": rows,
        "dependency_job_ids": args.dependency_job_ids,
        "output_dir": str(output_dir),
        "analyzer": str(ROOT / "wss_pinn/tools/analyze_f1_matrix.py"),
        "analyzer_sha256": sha256_file(
            ROOT / "wss_pinn/tools/analyze_f1_matrix.py"
        ),
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
