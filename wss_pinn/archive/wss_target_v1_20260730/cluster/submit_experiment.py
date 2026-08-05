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
from wss_pinn.tools.preflight import run_preflight
from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    git_state,
    guard_write_path,
    sha256_file,
    utc_now,
)


SBATCH = Path("/public/slurm/bin/sbatch")
SQUEUE = Path("/public/slurm/bin/squeue")
SACCT = Path("/public/slurm/bin/sacct")
PREFLIGHT = ROOT / "wss_pinn/cluster/preflight.slurm"
RUN = ROOT / "wss_pinn/cluster/run_experiment.slurm"


def command(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True)


def job_state(job_id: str) -> str:
    queue = command([str(SQUEUE), "-h", "-j", str(job_id), "-o", "%T"])
    if queue.returncode == 0 and queue.stdout.strip():
        return queue.stdout.strip().splitlines()[0]
    accounting = command(
        [str(SACCT), "-n", "-j", str(job_id), "--format=State", "-P"]
    )
    if accounting.returncode == 0 and accounting.stdout.strip():
        return accounting.stdout.strip().splitlines()[0].split("|", 1)[0]
    return "UNKNOWN"


def submit(arguments: list[str]) -> str:
    process = command([str(SBATCH), "--parsable", *arguments])
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "sbatch failed")
    return process.stdout.strip().split(";", 1)[0]


def resource_args(config: ExperimentConfig) -> list[str]:
    cluster = config["cluster"]
    args = [
        f"--partition={cluster['partition']}",
        f"--time={cluster['time']}",
        f"--mem={cluster['memory']}",
        f"--cpus-per-task={int(cluster['cpus'])}",
    ]
    if cluster.get("qos"):
        args.append(f"--qos={cluster['qos']}")
    if int(cluster["gpus"]) > 0:
        args.append(f"--gres=gpu:{int(cluster['gpus'])}")
    return args


def existing_submission(run_dir: Path) -> dict | None:
    path = run_dir / "submission.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    active = []
    for row in payload.get("jobs", []):
        state = job_state(str(row["job_id"]))
        if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
            active.append({**row, "state": state})
    if active:
        raise RuntimeError(f"valid jobs already exist for this experiment: {active}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    # Code-ready validation is allowed for locked future stages.
    preflight = run_preflight(config, require_science_gate=False)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "code_ready",
                    "stage": config.stage,
                    "science_gate": preflight["science_gate"],
                    "would_submit": preflight["science_gate"]["passed"],
                    "config_sha256": sha256_file(config.source),
                    "resolved_config_sha256": config.resolved_sha256,
                    "run_dir": str(config.run_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    # A real submission may never bypass the scientific gate.
    run_preflight(config, require_science_gate=True)
    run_dir = guard_write_path(config.run_dir)
    existing_submission(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    (run_dir / "slurm").mkdir(parents=True, exist_ok=False)
    reservation = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "reserved",
        "experiment_id": config.experiment_id,
        "stage": config.stage,
        "config": str(config.source),
        "config_sha256": sha256_file(config.source),
        "resolved_config_sha256": config.resolved_sha256,
        "sampling_manifest": config["sampling"]["manifest_path"],
        "sampling_manifest_sha256": sha256_file(config["sampling"]["manifest_path"]),
        "git": git_state(),
    }
    for field in ("resume", "init_checkpoint"):
        value = config["train"].get(field)
        if value:
            checkpoint_path = Path(value)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"{field} checkpoint is missing: {checkpoint_path}"
                )
            reservation[f"{field}_checkpoint"] = {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            }
    for field in (
        "source_data_audit_report",
        "sidecar_audit_report",
        "scientific_gate_report",
    ):
        value = config["experiment"].get(field)
        if value:
            report_path = Path(value)
            if not report_path.is_absolute():
                report_path = ROOT / report_path
            if not report_path.is_file():
                raise FileNotFoundError(f"{field} is missing: {report_path}")
            reservation[field] = {
                "path": str(report_path.resolve()),
                "sha256": sha256_file(report_path),
            }
    atomic_write_json(run_dir / "reservation.json", reservation)
    export = (
        f"ALL,WSS_PINN_CONFIG={config.source},WSS_PINN_RUN_DIR={run_dir},"
        f"WSS_PINN_PYTHON={config['cluster']['python']}"
    )
    common = resource_args(config)
    jobs: list[dict] = []
    submission = {**reservation, "status": "submitting", "jobs": jobs}
    try:
        preflight_id = submit(
            [
                *common,
                f"--job-name={config.experiment_id[:80]}-pf",
                f"--output={run_dir}/slurm/preflight_%j.out",
                f"--error={run_dir}/slurm/preflight_%j.err",
                f"--export={export}",
                str(PREFLIGHT),
            ]
        )
        jobs.append({"role": "gpu_preflight", "job_id": preflight_id})
        train_id = submit(
            [
                *common,
                f"--job-name={config.experiment_id[:80]}",
                f"--output={run_dir}/slurm/train_%j.out",
                f"--error={run_dir}/slurm/train_%j.err",
                f"--dependency=afterok:{preflight_id}",
                f"--export={export}",
                str(RUN),
            ]
        )
        jobs.append(
            {
                "role": "training_and_evaluation",
                "job_id": train_id,
                "dependency": f"afterok:{preflight_id}",
            }
        )
        submission.update(
            {
                "status": "submitted",
                "submitted_at": utc_now(),
                "jobs": jobs,
                "monitor_commands": [
                    f"/public/slurm/bin/squeue -j {preflight_id},{train_id}",
                    f"/public/slurm/bin/sacct -j {preflight_id},{train_id} --format=JobID,State,Elapsed,ExitCode",
                    f"tail -f {run_dir}/slurm/preflight_{preflight_id}.out",
                    f"tail -f {run_dir}/slurm/train_{train_id}.out",
                ],
                "resume_command": (
                    f"python wss_pinn/cluster/submit_experiment.py --config {config.source} "
                    "# after setting train.resume to an explicit checkpoint and a new run_dir"
                ),
                "checkpoint_paths": {
                    "best": str(run_dir / "checkpoints/best.pt"),
                    "last": str(run_dir / "checkpoints/last.pt"),
                },
            }
        )
    except Exception as exc:
        submission["status"] = "partial_failed" if jobs else "failed"
        submission["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(run_dir / "submission.json", submission)
        raise
    output = atomic_write_json(run_dir / "submission.json", submission)
    print(json.dumps({"status": "submitted", "jobs": jobs, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
