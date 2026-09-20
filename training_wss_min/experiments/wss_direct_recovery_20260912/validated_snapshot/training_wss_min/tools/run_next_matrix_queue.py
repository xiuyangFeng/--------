"""Memory-aware Slurm queue; process handles never enter the JSON status.

A status file/run directory is an execution record, not a resumable checkpoint.
Use a new experiment directory and new run names for a deliberate fresh attempt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
TERMINAL = {"complete", "failed", "cancelled", "blocked"}


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_write_status(path, state):
    """Serialize before touching disk, then replace the whole status atomically."""
    data = json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def slurm_gpus():
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("submit through sbatch: a numeric SLURM_JOB_ID is required")
    gpus = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if (not all(gpus) or len(set(gpus)) != len(gpus)
            or any(not re.fullmatch(r"(?:[0-9]+|GPU-[\w-]+|MIG-[\w./-]+)", g) for g in gpus)):
        raise RuntimeError("Slurm must expose distinct valid CUDA_VISIBLE_DEVICES")
    allocated = os.environ.get("SLURM_GPUS_ON_NODE")
    if allocated and allocated.isdigit() and len(gpus) > int(allocated):
        raise RuntimeError("CUDA_VISIBLE_DEVICES exceeds SLURM_GPUS_ON_NODE")
    return gpus


def free_memory(gpu):
    output = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    )
    return int(output.strip())


def read_arms(config_dir, matrix):
    arms = json.loads(matrix.read_text())["arms"]
    if not isinstance(arms, list) or not arms:
        raise ValueError("matrix must contain at least one arm")
    records = {}
    runs = set()
    for arm in arms:
        aid = arm["id"]
        if not isinstance(aid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", aid) or aid in records:
            raise ValueError(f"invalid or duplicate arm id: {aid!r}")
        phase = arm.get("phase", 0)
        if type(phase) is not int or phase < 0:
            raise ValueError(f"invalid phase for {aid}")
        config = (config_dir / arm["config"]).resolve()
        if not config.is_relative_to(config_dir.resolve()):
            raise ValueError(f"config must be inside config-dir: {config}")
        run = json.loads(config.read_text())["name"]
        # Existing training configs use matrix_id/arm_id as their run name.
        if (not isinstance(run, str) or not run
                or any(part in {"", ".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part)
                       for part in run.split("/"))):
            raise ValueError(f"invalid run name: {run!r}")
        run_root = (ROOT / "training_wss_min/runs").resolve()
        if not (run_root / run).resolve().is_relative_to(run_root):
            raise ValueError(f"run escapes runs directory: {run!r}")
        if run in runs:
            raise ValueError(f"duplicate run name: {run}")
        runs.add(run)
        records[aid] = {"config": str(config), "run_name": run, "phase": phase,
                        "status": "pending", "stages": {}}
    return records


class MatrixQueue:
    def __init__(self, args):
        self.args = args
        self.gpus = slurm_gpus()
        self.exp = args.experiment_dir
        self.path = self.exp / "queue_status.json"
        records = read_arms(args.config_dir, args.matrix or args.config_dir / "matrix.json")
        self.state = {"job_id": os.environ["SLURM_JOB_ID"], "started_at": stamp(),
                      "status": "running", "gpus": self.gpus,
                      "slots_per_gpu": args.slots_per_gpu,
                      "launch_interval_seconds": args.launch_interval_seconds, "arms": records}
        self.active = {}
        self.stopped = False
        self.next_launch = {g: 0.0 for g in self.gpus}

    def save(self):
        atomic_write_status(self.path, self.state)

    def refuse_overwrite(self):
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite execution status: {self.path}")
        for aid, record in self.state["arms"].items():
            run = ROOT / "training_wss_min/runs" / record["run_name"]
            if run.exists():
                raise FileExistsError(f"refusing to overwrite historical run: {run}")
            for stage in ("train", "eval_best", "eval_last"):
                log = self.exp / "logs" / f"{aid}_{stage}.log"
                if log.exists():
                    raise FileExistsError(f"refusing to overwrite historical log: {log}")

    def stop(self, signum, _frame):
        # Signal handlers do no IO; the finally block terminates and reaps children.
        self.stopped = True
        self.state["stop_signal"] = signum

    def launch(self, aid, gpu, stage):
        record = self.state["arms"][aid]
        run = ROOT / "training_wss_min/runs" / record["run_name"]
        if stage == "train" and run.exists():
            raise FileExistsError(f"run appeared while queued: {run}")
        if stage == "train":
            command = [PY, "-u", "-m", "training_wss_min.train", "--config", record["config"]]
        else:
            command = [PY, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(run),
                       "--partitions", "test", "--allow-test", "--checkpoint", stage[5:], "--no-plots"]
        log = self.exp / "logs" / f"{aid}_{stage}.log"
        record.update(status=stage, gpu=gpu)
        stage_record = {"started_at": stamp(), "log": str(log), "cmd": command}
        record["stages"][stage] = stage_record
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        stream = None
        try:
            stream = log.open("x")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException as exc:
            if stream is not None:
                stream.close()
            stage_record.update(ended_at=stamp(), error=f"{type(exc).__name__}: {exc}")
            record.update(status="failed", failed_stage=stage)
            raise
        # Register immediately: even a subsequent serialization error must reap this child.
        self.active[aid] = {"process": process, "stream": stream, "gpu": gpu, "stage": stage}
        stage_record["pid"] = process.pid
        self.next_launch[gpu] = time.monotonic() + self.args.launch_interval_seconds
        self.save()
        print(stamp(), aid, stage, gpu, flush=True)

    def poll(self):
        for aid, task in list(self.active.items()):
            process = task["process"]
            code = process.poll()
            if code is None:
                continue
            process.wait()
            task["stream"].close()
            record = self.state["arms"][aid]
            stage = task["stage"]
            record["stages"][stage].update(returncode=code, ended_at=stamp())
            del self.active[aid]
            if code:
                record.update(status="failed", failed_stage=stage)
            else:
                record["status"] = {"train": "ready_eval_best", "eval_best": "ready_eval_last",
                                    "eval_last": "complete"}[stage]
            self.save()

    def schedule(self):
        records = self.state["arms"]
        unfinished = [r for r in records.values() if r["status"] not in TERMINAL]
        if not unfinished:
            return
        phase = min(r["phase"] for r in unfinished)
        if any(r["phase"] < phase and r["status"] != "complete" for r in records.values()):
            for record in unfinished:
                record.update(status="blocked", reason="an earlier phase did not complete successfully")
            self.save()
            return
        for gpu in self.gpus:
            if self.stopped:
                return
            on_gpu = [t for t in self.active.values() if t["gpu"] == gpu]
            if len(on_gpu) >= self.args.slots_per_gpu or time.monotonic() < self.next_launch[gpu]:
                continue
            candidate = None
            # Finish train -> best -> last before launching another train; only one eval per GPU.
            if not any(t["stage"].startswith("eval_") for t in on_gpu):
                candidate = next(((aid, r["status"][6:]) for aid, r in records.items()
                                  if r["phase"] == phase and r.get("gpu") == gpu
                                  and r["status"].startswith("ready_")), None)
            if candidate is None:
                candidate = next(((aid, "train") for aid, r in records.items()
                                  if r["phase"] == phase and r["status"] == "pending"), None)
            if candidate is not None and free_memory(gpu) >= self.args.min_free_mib:
                self.launch(candidate[0], gpu, candidate[1])

    def cleanup(self):
        """Terminate every child group, then wait (and escalate) every direct child."""
        errors = []
        for task in self.active.values():
            try:
                os.killpg(task["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except BaseException as exc:
                errors.append(exc)
        for aid, task in list(self.active.items()):
            process = task["process"]
            try:
                try:
                    code = process.wait(timeout=self.args.terminate_timeout_seconds)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = process.wait()
                record = self.state["arms"][aid]
                record["stages"][task["stage"]].update(returncode=code, ended_at=stamp(), interrupted=True)
                record.update(status="cancelled" if self.stopped else "failed", failed_stage=task["stage"])
            except BaseException as exc:
                errors.append(exc)
            finally:
                try:
                    task["stream"].close()
                except BaseException as exc:
                    errors.append(exc)
                del self.active[aid]
        if errors:
            raise errors[0]

    def run(self):
        self.exp.mkdir(parents=True, exist_ok=True)
        lock = self.exp / ".queue.lock"
        # Exclusive claim closes the race between the no-overwrite check and first save.
        with lock.open("x") as stream:
            stream.write(f"pid={os.getpid()} job_id={self.state['job_id']}\n")
        handlers = {}
        error = None
        owned = False
        try:
            self.refuse_overwrite()
            owned = True
            (self.exp / "logs").mkdir(exist_ok=True)
            for sig in (signal.SIGTERM, signal.SIGINT):
                handlers[sig] = signal.signal(sig, self.stop)
            self.save()
            # Fail before any launch if a visible device cannot be queried.
            for gpu in self.gpus:
                free_memory(gpu)
            while not self.stopped and any(r["status"] not in TERMINAL for r in self.state["arms"].values()):
                self.poll()
                self.schedule()
                if any(r["status"] not in TERMINAL for r in self.state["arms"].values()):
                    time.sleep(self.args.poll_seconds)
        except BaseException as exc:
            error = exc
            self.state["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                if owned:
                    cleanup_error = None
                    try:
                        self.cleanup()
                    except BaseException as exc:
                        cleanup_error = exc
                        self.state["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                    for record in self.state["arms"].values():
                        if record["status"] not in TERMINAL:
                            record.update(status="cancelled" if self.stopped else "blocked",
                                          reason="queue interrupted before arm completion")
                    self.state["ended_at"] = stamp()
                    self.state["status"] = ("complete" if error is None and cleanup_error is None
                                            and not self.stopped and all(r["status"] == "complete"
                                                for r in self.state["arms"].values()) else "failed")
                    try:
                        self.save()
                    except BaseException:
                        if error is None and cleanup_error is None:
                            raise
                        # Preserve the original exception while making status-write failure visible.
                        traceback.print_exc()
                    if cleanup_error is not None:
                        if error is None:
                            raise cleanup_error
                        traceback.print_exception(type(cleanup_error), cleanup_error, cleanup_error.__traceback__)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
                lock.unlink(missing_ok=True)
        return 0 if self.state["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=ROOT / "training_wss_min/configs/wss_direct_next_20260911")
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "training_wss_min/experiments/wss_direct_next_20260911")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=7000)
    parser.add_argument("--launch-interval-seconds", type=float, default=60)
    parser.add_argument("--poll-seconds", type=float, default=3)
    parser.add_argument("--terminate-timeout-seconds", type=float, default=20)
    args = parser.parse_args(argv)
    if (args.slots_per_gpu < 1 or args.min_free_mib < 0 or args.launch_interval_seconds < 0
            or args.poll_seconds <= 0 or args.terminate_timeout_seconds <= 0):
        parser.error("invalid slot, memory or timing limit")
    return MatrixQueue(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
