#!/usr/bin/env python3
"""时间适配器探针的本机 GPU 队列（Slurm 之外执行，例如 ssh 到 node04）。

读取 configs/<exp>/matrix.json：先跑每折 cache，再跑依赖该折 cache 的各臂 fit / finetune；
按 --gpus × --slots-per-gpu 并发，CUDA_VISIBLE_DEVICES 逐任务指定。
启动时记录代码与配置的 sha256，每次派发前复核；变化即中止（与矩阵队列同一纪律）。
已有完成标记（cache manifest / metrics.json）的任务跳过，可断点续跑。
用法：python -m training_wss_min.tools.run_time_adapter_probe_queue --config-dir <dir> --gpus 0,1 [--slots-per-gpu 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints(config_dir: Path):
    paths = sorted(PKG.glob("*.py")) + sorted(config_dir.glob("*.json"))
    return {str(p.resolve()): sha256(p) for p in paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--slots-per-gpu", type=int, default=2)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--only", default="", help="comma list of job ids to run (default all)")
    args = ap.parse_args()
    config_dir = Path(args.config_dir).resolve()
    matrix = json.loads((config_dir / "matrix.json").read_text())
    exp = PKG / "experiments" / matrix["experiment"]
    logs = exp / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s}
    # 任务类型：cache / fit（time_adapter_probe）或 finetune（time_adapter_finetune）；矩阵未写 kind 的臂按 fit（旧矩阵不变）
    jobs = ([dict(j, kind="cache") for j in matrix.get("cache_jobs", [])]
            + [{"kind": "fit", **j} for j in matrix["arms"]])
    markers = {j["id"]: j["done_marker"] for j in jobs}
    if only:
        jobs = [j for j in jobs if j["id"] in only]
    frozen = fingerprints(config_dir)
    status = {"host": socket.gethostname(), "pid": os.getpid(), "package_dir": str(PKG), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "executed_outside_slurm": os.environ.get("SLURM_JOB_ID") is None, "gpus": args.gpus.split(","),
              "slots_per_gpu": args.slots_per_gpu, "fingerprint_files": len(frozen), "jobs": {}}
    for j in jobs:
        status["jobs"][j["id"]] = {"kind": j["kind"], "state": "done" if Path(j["done_marker"]).is_file() else "pending"}
    status_path = exp / "queue_status.json"

    def save():
        status_path.write_text(json.dumps(status, indent=1, ensure_ascii=False))

    slots = [g for g in args.gpus.split(",") for _ in range(args.slots_per_gpu)]
    running = {}   # job id -> (proc, slot, fh)
    save()
    while True:
        for jid, (proc, slot, fh) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            rec = status["jobs"][jid]
            rec.update(returncode=rc, ended_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       state="done" if rc == 0 else "failed")
            slots.append(slot)
            del running[jid]
            print(f"{time.strftime('%H:%M:%S')} {jid} -> rc {rc}", flush=True)
            save()
        pending = [j for j in jobs if status["jobs"][j["id"]]["state"] == "pending"]
        if not pending and not running:
            break
        for j in pending:
            if not slots:
                break
            dep = j.get("depends_on")
            if dep:
                dstate = status["jobs"][dep]["state"] if dep in status["jobs"] else (
                    "done" if Path(markers[dep]).is_file() else "missing")
                if dstate in ("failed", "missing"):
                    status["jobs"][j["id"]]["state"] = "skipped_dependency_failed"
                    save()
                    continue
                if dstate != "done":
                    continue
            if fingerprints(config_dir) != frozen:
                status["aborted"] = "code/config fingerprint changed during the queue"
                save()
                raise SystemExit(status["aborted"])
            slot = slots.pop(0)
            log = logs / f"{j['id']}.log"
            fh = log.open("a")
            if j["kind"] == "finetune":
                cmd = [sys.executable, "-u", "-m", "training_wss_min.time_adapter_finetune", "--config", str(config_dir / j["config"])]
            else:
                cmd = [sys.executable, "-u", "-m", "training_wss_min.time_adapter_probe", j["kind"], "--config", str(config_dir / j["config"])]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            proc = subprocess.Popen(cmd, cwd=str(PKG.parent), env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[j["id"]] = (proc, slot, fh)
            status["jobs"][j["id"]].update(state="running", gpu=slot, pid=proc.pid, log=str(log), cmd=cmd,
                                           started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            print(f"{time.strftime('%H:%M:%S')} launch {j['id']} on GPU {slot}", flush=True)
            save()
        time.sleep(args.poll_seconds)
    status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    failed = [k for k, v in status["jobs"].items() if v["state"] != "done"]
    print("queue finished; not done:", failed or "none", flush=True)


if __name__ == "__main__":
    main()
