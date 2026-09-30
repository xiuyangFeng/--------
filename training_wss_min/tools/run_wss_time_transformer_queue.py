#!/usr/bin/env python3
"""Run the Temporal Transformer queue with lifetime locks and artifact acceptance.

The script location selects the (possibly frozen) code tree. The matrix selects
the real output directory. GPU sharing requires an explicit flag and sufficient
free memory; other jobs are never terminated. A successful child exit alone
does not mean an accepted experiment.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
GPU_LOCK_ROOT = Path("/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments")
SOURCE_EXCLUDES = {"runs", "experiments", "configs", "__pycache__", ".git", ".venv", "venv"}


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def fingerprints(config_dir: Path) -> dict:
    paths = []
    for directory, subdirs, names in os.walk(PKG):
        subdirs[:] = sorted(d for d in subdirs if d not in SOURCE_EXCLUDES)
        paths.extend(Path(directory) / name for name in names if name.endswith(".py"))
    paths.extend(config_dir.rglob("*.json"))
    return {str(p.resolve()): sha256(p) for p in sorted(set(paths))}


def fingerprint_digest(values: dict) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def gpu_snapshot(selector: str) -> dict:
    result = subprocess.run(
        ["nvidia-smi", "-i", selector,
         "--query-gpu=index,uuid,name,driver_version,memory.free,memory.total,memory.used",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=20)
    rows = [r for r in csv.reader(result.stdout.splitlines(), skipinitialspace=True) if r]
    if len(rows) != 1 or len(rows[0]) != 7:
        raise RuntimeError(f"--gpu must resolve to exactly one physical GPU: {result.stdout!r}")
    index, uuid, name, driver, free, total, used = (v.strip() for v in rows[0])
    if not uuid.startswith("GPU-") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in uuid):
        raise RuntimeError(f"unexpected physical GPU UUID {uuid!r}")
    processes = subprocess.run(
        ["nvidia-smi", "-i", uuid, "--query-compute-apps=gpu_uuid,pid,process_name",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=20)
    active = []
    for row in csv.reader(processes.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 3:
            raise RuntimeError(f"cannot verify GPU occupancy: {row!r}")
        active.append({"gpu_uuid": row[0].strip(), "pid": int(row[1].strip()), "process_name": row[2].strip()})
    return {"index": index, "uuid": uuid, "name": name, "driver_version": driver,
            "memory_free_mib": int(free), "memory_total_mib": int(total), "memory_used_mib": int(used),
            "compute_processes": active, "checked_at": timestamp()}


def gpu_dispatch_ready(snapshot: dict, allow_shared: bool, min_free_mib: int) -> bool:
    return ((allow_shared or not snapshot["compute_processes"])
            and snapshot["memory_free_mib"] >= min_free_mib)


def finite_tree(value, where: str = "metrics") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where} is not finite")
    if isinstance(value, dict):
        for key, entry in value.items():
            finite_tree(entry, f"{where}.{key}")
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            finite_tree(entry, f"{where}[{index}]")


def contains_config(actual, expected, where="config") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{where} is missing or not an object")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{where}.{key} is missing")
            contains_config(actual[key], value, f"{where}.{key}")
    elif actual != expected:
        raise ValueError(f"{where} does not match the dispatched config")


def validate_outputs(rec: dict, frozen: dict, *, require_acceptance: bool) -> dict:
    cfg = rec["actual_config"]
    paths = {"metrics": Path(rec["done_marker"]), "history": Path(rec["history_path"]),
             "checkpoint": Path(rec["checkpoint_path"])}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing/empty {name}: {path}")
    metrics = json.loads(paths["metrics"].read_text())
    if metrics.get("config_sha256") != rec["config_sha256"]:
        raise ValueError("metrics config SHA does not match the dispatched file")
    contains_config(metrics.get("config"), cfg)
    for key in ("schema", "name", "arm", "fold", "seed"):
        if metrics.get(key) != cfg[key]:
            raise ValueError(f"metrics {key} differs from config")
    if metrics.get("smoke") is not False:
        raise ValueError("only an explicitly non-smoke run can be accepted")
    if cfg["eval"]["partition"] != "test" or metrics.get("partition") != "test":
        raise ValueError("only the registered test partition is accepted")
    split = json.loads(Path(cfg["split_path"]).read_text())
    if metrics.get("n_cases") != len(split["test_cases"]):
        raise ValueError("evaluated case count does not match the test split")
    expected = int(cfg["train"]["epochs"]) * int(cfg["train"]["steps_per_epoch"])
    if expected != 1800 or metrics.get("steps") != expected:
        raise ValueError(f"metrics has {metrics.get('steps')} steps; registered production runs require 1800")
    finite_tree(metrics.get("metrics", {}))
    if not metrics.get("metrics"):
        raise ValueError("metrics report is empty")
    anchor = metrics.get("anchor_check", {})
    error = anchor.get("max_abs_peak_ln_vs_anchor")
    if anchor.get("passed") is not True or error is None or not math.isfinite(float(error)) or not 0 <= float(error) <= 1e-9:
        raise ValueError(f"peak anchor acceptance failed: {anchor}")
    if anchor.get("n_cases") != len(split["test_cases"]):
        raise ValueError("peak anchor was not checked on every test case")
    for key in ("cache_manifest_sha256", "frame_stats_sha256", "split_sha256", "waveform_sha256"):
        if metrics.get(key) != rec["data_fingerprints"][key]:
            raise ValueError(f"metrics {key} differs from launch provenance")
    for key, path in rec["data_fingerprint_paths"].items():
        if sha256(Path(path)) != rec["data_fingerprints"][key]:
            raise ValueError(f"{key} source changed during the queue")
    code_hashes = metrics.get("code_sha256", {})
    if "time_transformer.py" not in code_hashes:
        raise ValueError("metrics does not fingerprint its runner")
    for name, source_hash in code_hashes.items():
        code_path = (PKG / name).resolve()
        if not is_within(code_path, PKG) or source_hash != frozen.get(str(code_path)):
            raise ValueError(f"metrics source SHA does not match frozen code: {name}")
    execution = metrics.get("execution", {})
    if execution.get("hostname") != socket.gethostname():
        raise ValueError("metrics execution host differs from this queue")
    if rec.get("gpu_uuid") and execution.get("cuda_visible_devices") != rec["gpu_uuid"]:
        raise ValueError("metrics CUDA visibility differs from the dispatched GPU UUID")
    if rec.get("gpu_uuid") and execution.get("gpu_uuid") != rec["gpu_uuid"]:
        raise ValueError("metrics GPU UUID differs from the dispatched physical GPU")
    if not execution.get("interpreter") or Path(execution["interpreter"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("metrics interpreter differs from the queue interpreter")
    evaluation = metrics.get("evaluation_contract", {})
    frame_stats = json.loads(Path(cfg["frame_stats_path"]).read_text())
    steps = frame_stats["frame"]["steps"]
    contains_config(evaluation, {"truth_space": "raw_pa_unclipped", "frame_steps": steps,
                    "peak_index": cfg["peak_index"], "peak_step": steps[cfg["peak_index"]],
                    "target_floor": frame_stats["floor"], "target_eps": frame_stats["eps"],
                    "checkpoint": "last"}, "evaluation_contract")
    history = [json.loads(line) for line in paths["history"].read_text().splitlines() if line.strip()]
    if len(history) != int(cfg["train"]["epochs"]):
        raise ValueError("history does not contain exactly the expected epochs")
    for epoch, entry in enumerate(history):
        if entry.get("epoch") != epoch or entry.get("step") != (epoch + 1) * int(cfg["train"]["steps_per_epoch"]):
            raise ValueError(f"history epoch/step sequence differs at epoch {epoch}")
        if "loss_z" not in entry:
            raise ValueError(f"history epoch {epoch} has no loss_z")
        finite_tree(entry, f"history[{epoch}]")
    artifact_hashes = {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}
    accepted = {"schema": "time_transformer_queue_acceptance_v1", "passed": True, "state": "complete",
                "config_sha256": rec["config_sha256"], "matrix_sha256": rec["matrix_sha256"],
                "code_sha256": code_hashes,
                "fingerprint_sha256": fingerprint_digest(frozen), "steps": expected,
                "data_fingerprints": rec["data_fingerprints"], "artifacts": artifact_hashes}
    acceptance_path = Path(rec["acceptance_path"])
    if require_acceptance:
        if not acceptance_path.is_file():
            raise ValueError("existing metrics lacks queue_acceptance.json; preserve it and use a new run directory")
        prior = json.loads(acceptance_path.read_text())
        contains_config(prior, accepted, "queue_acceptance")
        if prior.get("gpu_uuid") != execution.get("gpu_uuid") or prior.get("host") != execution.get("hostname"):
            raise ValueError("existing acceptance provenance differs from metrics execution")
        rec["gpu_uuid"] = prior["gpu_uuid"]
    else:
        write_json(acceptance_path, {**accepted, "accepted_at": timestamp(), "queue_pid": os.getpid(),
                                     "gpu_uuid": rec["gpu_uuid"], "host": socket.gethostname()})
    return accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--gpu", default="0", help="one physical GPU index or UUID")
    ap.add_argument("--allow-shared-gpu", action="store_true",
                    help="explicitly allow coexistence with existing GPU processes")
    ap.add_argument("--min-free-mib", type=int, default=12288,
                    help="minimum free GPU memory before each dispatch; not a memory reservation")
    ap.add_argument("--poll-seconds", type=float, default=20.0)
    ap.add_argument("--only", default="", help="comma-separated registered ids; default all")
    args = ap.parse_args()
    if not 0 < args.poll_seconds <= 60:
        ap.error("--poll-seconds must be in (0, 60]")
    if args.min_free_mib <= 0:
        ap.error("--min-free-mib must be positive")
    config_dir = Path(args.config_dir).resolve()
    matrix = json.loads((config_dir / "matrix.json").read_text())
    exp = Path(matrix["output_experiment_dir"])
    if not exp.is_absolute():
        ap.error("matrix.output_experiment_dir must be an absolute path")
    exp = exp.resolve()
    exp.mkdir(parents=True, exist_ok=True)
    experiment_lock = (exp / ".queue.lock").open("a+")
    try:
        fcntl.flock(experiment_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another queue holds {exp / '.queue.lock'}; no jobs dispatched")
    experiment_lock.seek(0)
    experiment_lock.truncate()
    experiment_lock.write(json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "started_at": timestamp()}))
    experiment_lock.flush()
    logs = exp / "logs"
    logs.mkdir(exist_ok=True)
    jobs = list(matrix["arms"])
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("matrix contains duplicate job ids")
    only = {item for item in args.only.split(",") if item}
    if only - {job["id"] for job in jobs}:
        raise ValueError(f"unregistered --only ids: {sorted(only - {job['id'] for job in jobs})}")
    if only:
        jobs = [job for job in jobs if job["id"] in only]
    if not jobs:
        raise ValueError("matrix selects no jobs")
    frozen = fingerprints(config_dir)
    status = {"host": socket.gethostname(), "pid": os.getpid(), "state": "preflight",
              "gpu_selector": args.gpu, "started_at": timestamp(),
              "gpu_policy": {"allow_shared_gpu": args.allow_shared_gpu,
                             "min_free_mib": args.min_free_mib,
                             "memory_reservation": False},
              "python_executable": sys.executable, "python_executable_resolved": str(Path(sys.executable).resolve()),
              "python_version": sys.version, "conda_prefix": os.environ.get("CONDA_PREFIX"),
              "code_package_dir": str(PKG), "working_directory": str(PKG.parent),
              "config_dir": str(config_dir), "output_experiment_dir": str(exp),
              "matrix_path": str(config_dir / "matrix.json"),
              "matrix_sha256": sha256(config_dir / "matrix.json"),
              "fingerprint_sha256": fingerprint_digest(frozen),
              "executed_outside_slurm": os.environ.get("SLURM_JOB_ID") is None,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "fingerprints": frozen, "jobs": {}}
    status_path = exp / "queue_status.json"
    stopped = threading.Event()
    stop_signal = None
    proc = None
    current_rec = None
    gpu_lock = None

    def save():
        status["updated_at"] = timestamp()
        write_json(status_path, status)

    def cancel(signum, _frame):
        nonlocal stop_signal
        stop_signal = signum
        stopped.set()

    def check_cancelled():
        if stopped.is_set():
            raise InterruptedError(f"queue received signal {stop_signal}")

    def check_fingerprints(phase):
        if fingerprints(config_dir) != frozen:
            raise RuntimeError(f"code/config fingerprint changed {phase}")
        for rec in status["jobs"].values():
            for key, path in rec["data_fingerprint_paths"].items():
                if sha256(Path(path)) != rec["data_fingerprints"][key]:
                    raise RuntimeError(f"data/config metadata changed {phase}: {path}")

    def wait_for_gpu(uuid, rec=None):
        while True:
            check_cancelled()
            snap = gpu_snapshot(uuid)
            status["gpu"] = snap
            if gpu_dispatch_ready(snap, args.allow_shared_gpu, args.min_free_mib):
                status["state"] = "ready"
                status.pop("waiting_reason", None)
                return snap
            reason = ("GPU has active compute processes; shared use disabled"
                      if snap["compute_processes"] and not args.allow_shared_gpu
                      else f"free GPU memory is below {args.min_free_mib} MiB")
            status.update(state="waiting_gpu", waiting_reason=reason)
            if rec is not None:
                rec.update(state="waiting_gpu", gpu=snap)
            save()
            stopped.wait(args.poll_seconds)

    previous_handlers = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for job in jobs:
            cfg_path = (config_dir / job["config"]).resolve()
            if not is_within(cfg_path, config_dir):
                raise ValueError("job config is outside config_dir")
            cfg = json.loads(cfg_path.read_text())
            marker = Path(job["done_marker"]).resolve()
            run_dir = Path(cfg["out_dir"]).resolve()
            if marker != run_dir / "metrics.json":
                raise ValueError(f"{job['id']}: done_marker does not match config out_dir/metrics.json")
            data_paths = {"cache_manifest_sha256": str(Path(cfg["cache_dir"]) / "manifest.json"),
                          "frame_stats_sha256": cfg["frame_stats_path"], "split_sha256": cfg["split_path"],
                          "waveform_sha256": cfg["waveform_path"]}
            rec = {"state": "pending", "config": str(cfg_path), "config_sha256": sha256(cfg_path),
                   "matrix_sha256": status["matrix_sha256"],
                   "actual_config": cfg, "done_marker": str(marker), "run_dir": str(run_dir),
                   "history_path": str(run_dir / "history.jsonl"), "checkpoint_path": str(run_dir / "ckpt_last.pt"),
                   "acceptance_path": str(run_dir / "queue_acceptance.json"), "data_fingerprint_paths": data_paths,
                   "data_fingerprints": {key: sha256(Path(path)) for key, path in data_paths.items()}}
            status["jobs"][job["id"]] = rec
            if marker.exists():
                rec["acceptance"] = validate_outputs(rec, frozen, require_acceptance=True)
                rec.update(state="done", already_completed=True, returncode=0)
            elif any(Path(rec[key]).exists() for key in ("history_path", "checkpoint_path", "acceptance_path")):
                raise ValueError(f"{job['id']}: partial outputs exist; preserve them and use a new run directory")
        save()
        pending = [job for job in jobs if status["jobs"][job["id"]]["state"] != "done"]
        if pending:
            snap = gpu_snapshot(args.gpu)
            uuid = snap["uuid"]
            status["gpu"] = snap
            GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
            lock_path = GPU_LOCK_ROOT / f".node04_gpu_{uuid}.lock"
            status["gpu_lock_path"] = str(lock_path)
            gpu_lock = lock_path.open("a+")
            while True:
                check_cancelled()
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    status.update(state="waiting_gpu_lock", waiting_reason="another cooperative queue holds the GPU lock")
                    status["gpu"] = gpu_snapshot(uuid)
                    save()
                    stopped.wait(args.poll_seconds)
            gpu_lock.seek(0)
            gpu_lock.truncate()
            gpu_lock.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                                       "experiment": matrix["experiment"], "started_at": timestamp()}))
            gpu_lock.flush()
            wait_for_gpu(uuid)
        for job in pending:
            current_rec = status["jobs"][job["id"]]
            snap = wait_for_gpu(uuid, current_rec)
            check_fingerprints("before dispatch")
            check_cancelled()
            log_path = logs / f"{job['id']}.log"
            cmd = [sys.executable, "-u", "-m", "training_wss_min.time_transformer", "--config", current_rec["config"]]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, WSS_GPU_UUID=uuid,
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            with log_path.open("a") as stream:
                stream.write(f"[queue] start {timestamp()} gpu_uuid={uuid} cmd={json.dumps(cmd)}\n")
                stream.flush()
                proc = subprocess.Popen(cmd, cwd=str(PKG.parent), env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                current_rec.update(state="running", pid=proc.pid, gpu=snap, gpu_uuid=uuid,
                                   log=str(log_path), cmd=cmd, started_at=timestamp(), cuda_visible_devices=uuid)
                status.update(state="running", current_job=job["id"])
                status.pop("waiting_reason", None)
                save()
                print(f"{timestamp()} launch {job['id']} pid={proc.pid} gpu_uuid={uuid}", flush=True)
                while proc.poll() is None:
                    check_cancelled()
                    stopped.wait(args.poll_seconds)
                check_cancelled()
                current_rec.update(returncode=proc.returncode, ended_at=timestamp())
                if proc.returncode != 0:
                    raise RuntimeError(f"job {job['id']} exited {proc.returncode}")
                check_fingerprints("after completion")
                current_rec["acceptance"] = validate_outputs(current_rec, frozen, require_acceptance=False)
                current_rec["state"] = "done"
                save()
                print(f"{timestamp()} accepted {job['id']}", flush=True)
                proc = None
        check_cancelled()
        check_fingerprints("before final completion")
        accepted_jobs = status["jobs"]
        accepted_code = [rec["acceptance"]["code_sha256"] for rec in accepted_jobs.values()]
        if any(code != accepted_code[0] for code in accepted_code):
            raise ValueError("accepted jobs do not share the same recorded code fingerprints")
        final_acceptance = {
            "schema": "time_transformer_queue_matrix_acceptance_v1", "passed": True, "state": "complete",
            "matrix_sha256": status["matrix_sha256"], "fingerprint_sha256": fingerprint_digest(frozen),
            "code_sha256": accepted_code[0], "accepted_at": timestamp(), "queue_pid": os.getpid(),
            "selected_job_ids": list(accepted_jobs), "all_matrix_jobs": len(accepted_jobs) == len(matrix["arms"]),
            "jobs": {jid: {"passed": True, "config_sha256": rec["config_sha256"],
                           "metrics_sha256": rec["acceptance"]["artifacts"]["metrics"]["sha256"],
                           "acceptance_path": rec["acceptance_path"]} for jid, rec in accepted_jobs.items()}}
        check_cancelled()
        write_json(exp / "queue_acceptance.json", final_acceptance)
        status.update(state="done", finished_at=timestamp())
        status.pop("current_job", None)
        save()
        print("queue finished: all selected jobs accepted", flush=True)
    except Exception as exc:
        cancelled = stopped.is_set() or isinstance(exc, InterruptedError)
        if proc is not None and proc.poll() is None:
            proc.terminate()  # This queue's direct child only; never another GPU user's PID.
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        state = "cancelled" if cancelled else "failed"
        status.update(state=state, aborted=str(exc), ended_at=timestamp())
        if current_rec is not None and current_rec["state"] != "done":
            current_rec.update(state=state, ended_at=timestamp())
            if proc is not None:
                current_rec["returncode"] = proc.returncode
        save()
        raise SystemExit((128 + int(stop_signal)) if cancelled and stop_signal else str(exc)) from exc
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if gpu_lock is not None:
            gpu_lock.close()
        experiment_lock.close()


if __name__ == "__main__":
    main()
