#!/usr/bin/env python3
"""在 node04 直启的单 GPU 时间 Transformer 队列。

启动时冻结新实验脚本与配置指纹；每个任务独占一个显式 CUDA_VISIBLE_DEVICES，
按 matrix.json 顺序串行运行并写 queue_status.json、日志、PID 和 GPU 参数。
旧实验队列不读取也不修改。
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--poll-seconds", type=float, default=20.0)
    ap.add_argument("--only", default="", help="comma-separated ids; default all")
    args = ap.parse_args()
    config_dir = Path(args.config_dir).resolve()
    matrix = json.loads((config_dir / "matrix.json").read_text())
    exp = PKG / "experiments" / matrix["experiment"]
    logs = exp / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = list(matrix["arms"])
    only = {x for x in args.only.split(",") if x}
    if only:
        jobs = [j for j in jobs if j["id"] in only]
    fingerprint_paths = [PKG / "time_transformer.py", Path(__file__), config_dir / "matrix.json"]
    fingerprint_paths += [config_dir / j["config"] for j in jobs]
    frozen = {str(p): sha256(p) for p in fingerprint_paths}
    status = {"host": socket.gethostname(), "pid": os.getpid(), "gpu": str(args.gpu),
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "executed_outside_slurm": os.environ.get("SLURM_JOB_ID") is None,
              "fingerprints": frozen, "jobs": {}}
    for j in jobs:
        marker = Path(j["done_marker"])
        status["jobs"][j["id"]] = {"state": "done" if marker.is_file() else "pending",
                                     "config": j["config"], "done_marker": str(marker)}
    status_path = exp / "queue_status.json"

    def save():
        status_path.write_text(json.dumps(status, indent=1, ensure_ascii=False))

    save()
    for j in jobs:
        rec = status["jobs"][j["id"]]
        if rec["state"] == "done":
            continue
        current = {str(p): sha256(p) for p in fingerprint_paths}
        if current != frozen:
            status["aborted"] = "code/config fingerprint changed before dispatch"
            save()
            raise SystemExit(status["aborted"])
        log_path = logs / f"{j['id']}.log"
        cmd = [sys.executable, "-u", "-m", "training_wss_min.time_transformer",
               "--config", str(config_dir / j["config"])]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4",
                   MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with log_path.open("a") as fh:
            fh.write(f"[queue] start {started} gpu={args.gpu} cmd={' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=str(PKG.parent), env=env, stdout=fh, stderr=subprocess.STDOUT)
            rec.update(state="running", pid=proc.pid, gpu=args.gpu, log=str(log_path), cmd=cmd, started_at=started)
            save()
            print(f"{started} launch {j['id']} pid={proc.pid} gpu={args.gpu}", flush=True)
            while True:
                rc = proc.poll()
                if rc is not None:
                    ended = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    rec.update(state="done" if rc == 0 else "failed", returncode=rc, ended_at=ended)
                    save()
                    print(f"{ended} finish {j['id']} rc={rc}", flush=True)
                    if rc != 0:
                        status["aborted"] = f"job {j['id']} failed; later jobs not dispatched"
                        save()
                        raise SystemExit(status["aborted"])
                    break
                time.sleep(float(args.poll_seconds))
    status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
