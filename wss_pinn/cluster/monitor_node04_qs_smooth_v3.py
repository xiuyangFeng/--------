"""Capture one machine-readable two-host health snapshot for the V3 six arms."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from wss_pinn.utils import ROOT, atomic_write_json, utc_now


SUBMISSION = ROOT / "outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/submission_six_gpu.json"
SNAPSHOTS = ROOT / "outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/monitor_snapshots.jsonl"
REMOTE = "cy@node04"
SQUEUE = "/public/slurm/bin/squeue"
SACCT = "/public/slurm/bin/sacct"


def _ssh(command: str) -> str:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", REMOTE, command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def _local(command: str) -> str:
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout


def _run_host(host: str, command: str) -> str:
    if host == "master":
        return _local(command)
    if host == "node04":
        return _ssh(command)
    raise ValueError(f"unsupported host={host}")


def _tail(path: Path, lines: int = 30) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        try:
            handle.seek(-65_536, 2)
        except OSError:
            handle.seek(0)
        return "\n".join(
            handle.read().decode("utf-8", errors="replace").splitlines()[-lines:]
        )


def _last_json(path: Path) -> dict | None:
    tail = _tail(path, lines=1)
    if not tail:
        return None
    try:
        return json.loads(tail)
    except json.JSONDecodeError:
        return {"unparsed_tail": tail}


def _slurm_status(job_id: str) -> dict:
    queued_result = subprocess.run(
        [SQUEUE, "-h", "-j", job_id, "-o", "%T|%M|%R"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    queued = queued_result.stdout.strip()
    if queued:
        state, elapsed, reason_or_node = queued.split("|", 2)
        return {
            "state": state,
            "elapsed": elapsed,
            "reason_or_node": reason_or_node,
            "source": "squeue",
        }
    accounted = subprocess.run(
        [
            SACCT,
            "-n",
            "-X",
            "-j",
            job_id,
            "--format=State,Elapsed,NodeList,ExitCode",
            "--parsable2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    line = next((line for line in accounted.splitlines() if line.strip()), "")
    if not line:
        return {"state": "UNKNOWN", "source": "none"}
    state, elapsed, node, exit_code = line.split("|", 3)
    return {
        "state": state.split("+")[0],
        "elapsed": elapsed,
        "reason_or_node": node,
        "exit_code": exit_code,
        "source": "sacct",
    }


def snapshot() -> dict:
    submission = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    gpu = {
        host: _run_host(
            host,
            "nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu "
            "--format=csv,noheader,nounits",
        ).strip().splitlines()
        for host in sorted({job["host"] for job in submission["jobs"]})
    }
    jobs = []
    for job in submission["jobs"]:
        host = str(job["host"])
        log = Path(job["log"])
        if job.get("kind") == "slurm_array_task":
            scheduler = _slurm_status(str(job["job_id"]))
            process = json.dumps(scheduler, ensure_ascii=False)
            active = scheduler["state"] in {
                "PENDING",
                "CONFIGURING",
                "RUNNING",
                "COMPLETING",
            }
            running = scheduler["state"] == "RUNNING"
            process_matches = True
        else:
            pid = int(job["pid"])
            process = _run_host(
                host,
                f"if kill -0 {pid} 2>/dev/null; then ps -p {pid} -o pid=,etime=,%cpu=,%mem=,stat=,cmd=; "
                f"else echo EXITED; fi",
            ).strip()
            scheduler = None
            active = process != "EXITED"
            running = active
            process_matches = job["experiment_id"] in process or job["config"] in process
        stat = log.stat() if log.exists() else None
        tail = _tail(log)
        error_tail = _tail(Path(job["error_log"])) if job.get("error_log") else ""
        run_dir = Path(
            job.get("run_dir")
            or (
                ROOT
                / "outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/runs"
                / job["experiment_id"]
            )
        )
        epoch_path = run_dir / "epoch_progress.jsonl"
        epoch_tail = _last_json(epoch_path)
        step_tail = _last_json(run_dir / "training_progress.jsonl")
        errors = [
            token
            for token in (
                "Traceback",
                "CUDA out of memory",
                "non-finite",
                "nan",
                "CUDA error",
            )
            if token.lower() in (tail + "\n" + error_tail).lower()
        ]
        jobs.append(
            {
                **job,
                "process": process,
                "scheduler": scheduler,
                "alive": active,
                "running": running,
                "process_matches_experiment": process_matches,
                "log_size": stat.st_size if stat else 0,
                "log_mtime": stat.st_mtime if stat else None,
                "log_age_seconds": time.time() - stat.st_mtime if stat else None,
                "log_tail": tail,
                "error_log_tail": error_tail,
                "epoch_tail": epoch_tail,
                "step_tail": step_tail,
                "error_tokens": errors,
                "initialization_checkpoint": (
                    run_dir / "checkpoints/initialization.pt"
                ).exists(),
                "best_validation_data_checkpoint": (
                    run_dir / "checkpoints/best_validation_data.pt"
                ).exists(),
                "best_validation_total_checkpoint": (
                    run_dir / "checkpoints/best_validation_total.pt"
                ).exists(),
                "last_checkpoint": (run_dir / "checkpoints/last.pt").exists(),
            }
        )
    result = {
        "schema_version": 1,
        "created_at": utc_now(),
        "hosts": sorted(gpu),
        "gpu": gpu,
        "jobs": jobs,
        "healthy": all(
            job["alive"]
            and job["process_matches_experiment"]
            and not job["error_tokens"]
            for job in jobs
        ),
        "all_running": all(job["running"] for job in jobs),
    }
    SNAPSHOTS.parent.mkdir(parents=True, exist_ok=True)
    with SNAPSHOTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/monitor_latest.json"),
    )
    args = parser.parse_args()
    value = snapshot()
    atomic_write_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
