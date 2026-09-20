"""Execute the wave-1 local-information screening matrix on shared Slurm GPUs.

Built on the repaired memory-aware queue: every arm runs train -> eval best -> eval last
(-> eval ema when the arm keeps an EMA checkpoint), one seed each, on the GPUs Slurm
exposes.  ``depends_on`` arms (X13b after X13a) wait for their dependency to complete.
Source/config fingerprints are recorded at start and re-checked before every launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from training_wss_min.tools import run_next_matrix_queue as base

ROOT = base.ROOT
NAME = "wss_local_wave1_20260912"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
STAGE_NEXT = {"train": "ready_eval_best", "eval_best": "ready_eval_last", "eval_last": "ready_eval_ema",
              "eval_ema": "complete"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprints(config_dir=CONFIGS):
    """Core training sources, every arm config and every frozen stats/split file they reference."""
    paths = list((ROOT / "training_wss_min").glob("*.py")) + sorted(config_dir.glob("*.json"))
    for cfg_path in sorted(config_dir.glob("*_s*.json")):
        cfg = json.loads(cfg_path.read_text())
        for key in ("feature_stats_path", "split_path", "wss_stats_path", "waveform_path", "time_basis_path"):
            value = cfg["data"].get(key)
            if value:
                paths.append(Path(value))
        if cfg["train"].get("init_reference_config"):
            paths.append(Path(cfg["train"]["init_reference_config"]))
    return {str(p.resolve()): sha256(p) for p in sorted(set(paths)) if p.is_file()}


def verify_preflight(config_dir=CONFIGS, exp=EXP):
    path = exp / "runtime_preflight.json"
    evidence = json.loads(path.read_text())
    if evidence.get("passed") is not True or not evidence.get("slurm_job_id"):
        raise RuntimeError("a successful Slurm GPU preflight is required before the matrix")
    if fingerprints(config_dir) != evidence["fingerprints"]:
        raise RuntimeError("source/config/protocol changed since the GPU preflight")
    expected = {p.name for p in config_dir.glob("*_s*.json")}
    if set(evidence["arms"]) != expected:
        raise RuntimeError("GPU preflight did not exercise every arm config")
    if evidence.get("anchor_reevaluation", {}).get("passed") is not True:
        raise RuntimeError("anchor (C1) re-evaluation compatibility check did not pass")
    return {"preflight": str(path), "sha256": sha256(path), "job_id": evidence["slurm_job_id"]}


class Wave1Queue(base.MatrixQueue):
    def __init__(self, args):
        super().__init__(args)
        matrix = json.loads((args.matrix or args.config_dir / "matrix.json").read_text())
        self.meta = {arm["id"]: arm for arm in matrix["arms"]}
        if set(self.meta) != set(self.state["arms"]):
            raise ValueError("matrix arms and config records differ")
        for aid, record in self.state["arms"].items():
            record["depends_on"] = list(self.meta[aid].get("depends_on", []))
            record["evaluate"] = list(self.meta[aid].get("evaluate", ["best", "last"]))
            unknown = set(record["depends_on"]) - set(self.meta)
            if unknown:
                raise ValueError(f"{aid} depends on unknown arms {sorted(unknown)}")
            if record["depends_on"]:
                record["status"] = "waiting"
        self.state.update(matrix=matrix["experiment"],
                          expected_training_runs=matrix["expected_training_runs"],
                          expected_evaluations=matrix["expected_evaluations"])

    # ---- gates -------------------------------------------------------------
    def verify_gate(self):
        result = verify_preflight(self.args.config_dir, self.exp)
        self.preflight_path = Path(result["preflight"])
        self.preflight_sha256 = result["sha256"]
        self.frozen = json.loads(self.preflight_path.read_text())["fingerprints"]
        return result

    def verify_source(self):
        if sha256(self.preflight_path) != self.preflight_sha256 or fingerprints(self.args.config_dir) != self.frozen:
            raise RuntimeError("source/config/protocol or preflight evidence changed during matrix execution")

    def refuse_overwrite(self):
        super().refuse_overwrite()
        for aid in self.state["arms"]:
            log = self.exp / "logs" / f"{aid}_eval_ema.log"
            if log.exists():
                raise FileExistsError(f"refusing to overwrite historical log: {log}")

    def run(self):
        self.state["preflight_verification"] = self.verify_gate()
        return super().run()

    # ---- execution ---------------------------------------------------------
    def launch(self, aid, gpu, stage):
        self.verify_source()
        record = self.state["arms"][aid]
        run = ROOT / "training_wss_min/runs" / record["run_name"]
        if stage == "train" and run.exists():
            raise FileExistsError(f"run appeared while queued: {run}")
        if stage == "train":
            for dep in record["depends_on"]:
                if self.state["arms"][dep]["status"] != "complete":
                    raise RuntimeError(f"{aid} launched before dependency {dep} completed")
            command = [base.PY, "-u", "-m", "training_wss_min.train", "--config", record["config"]]
        else:
            command = [base.PY, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(run),
                       "--partitions", "test", "--allow-test", "--checkpoint", stage[5:], "--no-plots"]
            # same-point prediction export exists for the peak-frame evaluator only
            if json.loads(Path(record["config"]).read_text())["data"].get("timesteps", "peak") == "peak":
                command.append("--save-predictions")
        log = self.exp / "logs" / f"{aid}_{stage}.log"
        record.update(status=stage, gpu=gpu)
        stage_record = {"started_at": base.stamp(), "log": str(log), "cmd": command}
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
            stage_record.update(ended_at=base.stamp(), error=f"{type(exc).__name__}: {exc}")
            record.update(status="failed", failed_stage=stage)
            raise
        self.active[aid] = {"process": process, "stream": stream, "gpu": gpu, "stage": stage}
        stage_record["pid"] = process.pid
        self.next_launch[gpu] = time.monotonic() + self.args.launch_interval_seconds
        self.save()
        print(base.stamp(), aid, stage, gpu, flush=True)

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
            record["stages"][stage].update(returncode=code, ended_at=base.stamp())
            del self.active[aid]
            if code:
                record.update(status="failed", failed_stage=stage)
            else:
                status = STAGE_NEXT[stage]
                if status == "ready_eval_ema" and "ema" not in record["evaluate"]:
                    status = "complete"
                record["status"] = status
            self.save()

    def schedule(self):
        records = self.state["arms"]
        for aid, record in records.items():
            if record["status"] != "waiting":
                continue
            deps = [records[d]["status"] for d in record["depends_on"]]
            if all(status == "complete" for status in deps):
                record["status"] = "pending"
            elif any(status in base.TERMINAL for status in deps):
                record.update(status="blocked", reason="a dependency did not complete successfully")
        self.save()
        super().schedule()


def write_report_safely(args):
    log = args.experiment_dir / "logs" / f"report_{time.time_ns()}.log"
    command = [base.PY, "-u", "-m", "training_wss_min.tools.report_wss_local_wave1",
               "--experiment-dir", str(args.experiment_dir), "--config-dir", str(args.config_dir)]
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        if result.returncode:
            base.atomic_write_status(args.experiment_dir / "report_failure.json",
                                     {"returncode": result.returncode, "log": str(log), "reported_at": base.stamp()})
    except Exception as exc:  # the queue status must survive a report failure
        print(f"report generation failed: {exc}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=CONFIGS)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--experiment-dir", type=Path, default=EXP)
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--min-free-mib", type=int, default=7000)
    parser.add_argument("--launch-interval-seconds", type=float, default=60)
    parser.add_argument("--poll-seconds", type=float, default=3)
    parser.add_argument("--terminate-timeout-seconds", type=float, default=20)
    args = parser.parse_args(argv)
    if (args.slots_per_gpu < 1 or args.min_free_mib < 0 or args.launch_interval_seconds < 0
            or args.poll_seconds <= 0 or args.terminate_timeout_seconds <= 0):
        parser.error("invalid slot, memory or timing limit")
    try:
        return Wave1Queue(args).run()
    finally:
        write_report_safely(args)


if __name__ == "__main__":
    raise SystemExit(main())
