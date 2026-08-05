"""Gate-protected six-arm launcher: Slurm master4 + direct node04 A100x2.

The cluster GPU partition manages the four RTX 4090 devices on ``master``.
``node04`` exposes two additional A100 devices through SSH, but is not
registered as a GPU GRES node and is currently DOWN in Slurm.  Formal launch
therefore submits E1--E4 as a four-task Slurm array and starts E5--E6 as
public-tree ``nohup`` processes on node04, one arm per GPU.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from wss_pinn.utils import ROOT, atomic_write_json, sha256_file, utc_now


CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_qs_smooth_v3"
CONFIG_LIST = CONFIG_ROOT / "matrix_configs.txt"
MASTER_CONFIG_LIST = CONFIG_ROOT / "matrix_configs_master4.txt"
BOUNDARY_GATE = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_boundary/report.json"
STATIC_GATE = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_preflight/report.json"
GPU_GATE = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_gpu_preflight/report.json"
OUTPUT = ROOT / "outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/submission_six_gpu.json"
REMOTE = "cy@node04"
PYTHON = "/public/newhome/cy/.conda/envs/GNN/bin/python"
PROJECT = "/public/newhome/cy/Digital_twin/GNN"
SBATCH = "/public/slurm/bin/sbatch"
SCANCEL = "/public/slurm/bin/scancel"
SLURM_RUN = ROOT / "wss_pinn/cluster/run_experiment.slurm"


def _gate(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate_result") != "pass":
        raise RuntimeError(f"Gate did not pass: {path}")
    return payload


def _ssh(command: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", REMOTE, command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def _local(command: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def _run_host(host: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess:
    if host == "master":
        return _local(command, check=check)
    if host == "node04":
        return _ssh(command, check=check)
    raise ValueError(f"unsupported host={host}")


def _configs() -> list[str]:
    values = [
        line.strip()
        for line in CONFIG_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(values) != 6 or len(set(values)) != 6:
        raise ValueError("V3 config list must contain six unique configs")
    expected_master = values[:4]
    actual_master = [
        line.strip()
        for line in MASTER_CONFIG_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if actual_master != expected_master:
        raise ValueError("master4 config list must be the first four V3 arms")
    return values


def _node_status(host: str, expected_gpus: int) -> dict:
    command = (
        "id -u; id -g; echo HOME=$HOME; "
        "test -d /public/newhome/cy/Digital_twin/GNN/data_new && echo public_ok; "
        "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits; "
        "echo __COMPUTE_APPS__; "
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )
    result = _run_host(host, command)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if lines[:2] != ["1006", "1007"] or "public_ok" not in lines:
        raise RuntimeError(f"{host} UID/public path check failed: {result.stdout}")
    marker = lines.index("__COMPUTE_APPS__")
    gpu_lines = [line for line in lines[:marker] if line[:1].isdigit() and "," in line]
    gpus = []
    for line in gpu_lines:
        index, uuid, name, total, used, utilization = [
            value.strip() for value in line.split(",")
        ]
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_percent": int(utilization),
            }
        )
    if len(gpus) != expected_gpus:
        raise RuntimeError(f"{host} must expose {expected_gpus} GPUs: {gpus}")
    gpu_index_by_uuid = {gpu["uuid"]: gpu["index"] for gpu in gpus}
    processes = []
    for line in lines[marker + 1 :]:
        values = [value.strip() for value in line.split(",")]
        if len(values) != 4 or values[0] not in gpu_index_by_uuid:
            continue
        uuid, pid_text, process_name, used_memory = values
        pid = int(pid_text)
        owner = _run_host(host, f"ps -p {pid} -o user=", check=False).stdout.strip()
        processes.append(
            {
                "gpu": gpu_index_by_uuid[uuid],
                "gpu_uuid": uuid,
                "pid": pid,
                "owner": owner or None,
                "process_name": process_name,
                "used_memory_mib": int(used_memory),
            }
        )
    return {"host": host, "raw": result.stdout, "gpus": gpus, "processes": processes}


def _dry_run_process(host: str, gpu: int, config: str) -> subprocess.Popen:
    command = (
        f"cd {shlex.quote(PROJECT)} && "
        f"CUDA_VISIBLE_DEVICES={gpu} CUBLAS_WORKSPACE_CONFIG=:4096:8 "
        f"{shlex.quote(PYTHON)} -m wss_pinn.train "
        f"--config {shlex.quote(config)} --dry-run --device cuda"
    )
    argv = (
        ["bash", "-lc", command]
        if host == "master"
        else ["ssh", "-o", "BatchMode=yes", REMOTE, command]
    )
    return subprocess.Popen(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def gpu_preflight() -> dict:
    _gate(BOUNDARY_GATE)
    _gate(STATIC_GATE)
    statuses = {
        "master": _node_status("master", 4),
        "node04": _node_status("node04", 2),
    }
    configs = _configs()
    assignments = [
        (configs[0], "master", 0),
        (configs[1], "master", 1),
        (configs[2], "master", 2),
        (configs[3], "master", 3),
        (configs[4], "node04", 0),
        (configs[5], "node04", 1),
    ]
    processes = [
        (config, host, gpu, _dry_run_process(host, gpu, config))
        for config, host, gpu in assignments
    ]
    results = []
    for config, host, gpu, process in processes:
        stdout, stderr = process.communicate()
        results.append(
            {
                "config": config,
                "host": host,
                "gpu": gpu,
                "returncode": process.returncode,
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            }
        )
        if process.returncode != 0:
            report = {
                "schema_version": 1,
                "created_at": utc_now(),
                "status": "failed",
                "gate_result": "fail",
                "nodes": statuses,
                "results": results,
            }
            atomic_write_json(GPU_GATE, report)
            raise RuntimeError(f"GPU dry-run failed: {config}")
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed",
        "gate_result": "pass",
        "nodes": statuses,
        "assignments": [
            {"config": config, "host": host, "gpu": gpu}
            for config, host, gpu in assignments
        ],
        "results": results,
        "boundary_gate_sha256": sha256_file(BOUNDARY_GATE),
        "static_gate_sha256": sha256_file(STATIC_GATE),
    }
    atomic_write_json(GPU_GATE, report)
    return report


def _validate_run_targets(configs: list[str]) -> list[dict]:
    specs = []
    log_root = ROOT / "outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/host_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for config in configs:
        payload = json.loads((ROOT / config).read_text(encoding="utf-8"))
        experiment_id = payload["experiment"]["id"]
        run_dir = ROOT / payload["paths"]["run_dir"]
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty formal run directory: {run_dir}")
        specs.append(
            {
                "config": config,
                "experiment_id": experiment_id,
                "run_dir": run_dir,
                "host_log": log_root / f"{experiment_id}.log",
            }
        )
    for spec in specs[4:]:
        if spec["host_log"].exists() and spec["host_log"].stat().st_size:
            raise FileExistsError(f"refusing to overwrite formal log: {spec['host_log']}")
    return specs


def _launch_detached_node04(spec: dict, gpu: int) -> int:
    """Start one remote trainer without leaving an SSH-owned shell behind."""
    argv = [PYTHON, "-m", "wss_pinn.train", "--config", spec["config"]]
    python_code = (
        "import os,subprocess;"
        f"log=open({str(spec['host_log'])!r},'ab',buffering=0);"
        "env=os.environ.copy();"
        f"env.update({{'CUDA_VISIBLE_DEVICES':{str(gpu)!r},"
        "'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONUNBUFFERED':'1'});"
        f"process=subprocess.Popen({argv!r},cwd={PROJECT!r},env=env,"
        "stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,"
        "start_new_session=True,close_fds=True);"
        "print(process.pid,flush=True)"
    )
    result = _ssh(f"{shlex.quote(PYTHON)} -c {shlex.quote(python_code)}")
    pid = int(result.stdout.strip().splitlines()[-1])
    verification = _ssh(
        f"sleep 1; kill -0 {pid}; ps -p {pid} -o cmd=",
        check=False,
    )
    if verification.returncode != 0 or spec["config"] not in verification.stdout:
        raise RuntimeError(
            f"node04 detached launch verification failed for {spec['experiment_id']}: "
            f"{verification.stdout} {verification.stderr}"
        )
    return pid


def submit() -> dict:
    _gate(BOUNDARY_GATE)
    _gate(STATIC_GATE)
    gpu_gate = _gate(GPU_GATE)
    if gpu_gate.get("boundary_gate_sha256") != sha256_file(BOUNDARY_GATE):
        raise RuntimeError("GPU preflight is stale relative to the boundary Gate")
    if gpu_gate.get("static_gate_sha256") != sha256_file(STATIC_GATE):
        raise RuntimeError("GPU preflight is stale relative to the static Gate")
    statuses = {
        "master": _node_status("master", 4),
        "node04": _node_status("node04", 2),
    }
    node04_busy = statuses["node04"]["processes"] or any(
        gpu["memory_used_mib"] > 256 or gpu["utilization_percent"] > 5
        for gpu in statuses["node04"]["gpus"]
    )
    if node04_busy:
        raise RuntimeError(
            "node04 GPUs are occupied and node04 is not registered as a Slurm GPU node: "
            f"{statuses['node04']}"
        )
    configs = _configs()
    specs = _validate_run_targets(configs)
    jobs: list[dict] = []
    node04_jobs: list[dict] = []
    slurm_array_id: str | None = None
    try:
        for gpu, spec in enumerate(specs[4:]):
            pid = _launch_detached_node04(spec, gpu)
            job = {
                "kind": "direct_process",
                "experiment_id": spec["experiment_id"],
                "config": spec["config"],
                "config_sha256": sha256_file(ROOT / spec["config"]),
                "run_dir": str(spec["run_dir"].resolve()),
                "host": "node04",
                "gpu": gpu,
                "pid": pid,
                "log": str(spec["host_log"].resolve()),
            }
            node04_jobs.append(job)
            jobs.append(job)

        slurm_array_id = subprocess.run(
            [
                SBATCH,
                "--parsable",
                "--array=0-3%4",
                f"--export=ALL,VOLUME_CONFIG_LIST={MASTER_CONFIG_LIST.resolve()}",
                str(SLURM_RUN),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip().split(";")[0]
        for task_index, spec in enumerate(specs[:4]):
            jobs.append(
                {
                    "kind": "slurm_array_task",
                    "experiment_id": spec["experiment_id"],
                    "config": spec["config"],
                    "config_sha256": sha256_file(ROOT / spec["config"]),
                    "run_dir": str(spec["run_dir"].resolve()),
                    "host": "master",
                    "gpu": "slurm_allocated",
                    "array_job_id": int(slurm_array_id),
                    "array_task_id": task_index,
                    "job_id": f"{slurm_array_id}_{task_index}",
                    "log": str(
                        (
                            ROOT
                            / f"outputs/wss_pinn/slurm/volume_field_train_{slurm_array_id}_{task_index}.out"
                        ).resolve()
                    ),
                    "error_log": str(
                        (
                            ROOT
                            / f"outputs/wss_pinn/slurm/volume_field_train_{slurm_array_id}_{task_index}.err"
                        ).resolve()
                    ),
                }
            )
    except Exception:
        if slurm_array_id is not None:
            subprocess.run([SCANCEL, slurm_array_id], cwd=ROOT, check=False)
        for job in node04_jobs:
            _ssh(f"kill {int(job['pid'])}", check=False)
        raise

    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "submitted",
        "formal_training_submitted": True,
        "route": "volume_uvwp_peak_qs_smooth_v3",
        "hosts": ["master", "node04"],
        "launch_mode": "slurm_master4_plus_direct_node04_2",
        "slurm_array_id": int(slurm_array_id),
        "nodes": statuses,
        "gates": {
            "boundary": sha256_file(BOUNDARY_GATE),
            "static": sha256_file(STATIC_GATE),
            "gpu_preflight": sha256_file(GPU_GATE),
        },
        "jobs": jobs,
    }
    atomic_write_json(OUTPUT, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gpu-preflight", action="store_true")
    group.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    payload = gpu_preflight() if args.gpu_preflight else submit()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
