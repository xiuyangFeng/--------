#!/usr/bin/env python3
"""标准训练矩阵的本机 GPU 队列（Slurm 之外执行，例如 ssh 到 node04）。

读取 configs/<exp>/matrix.json 的 arms（id / config / run_name）：每臂 train → eval best → eval last（均 --save-predictions），
按 --gpus × --slots-per-gpu 并发，CUDA_VISIBLE_DEVICES 逐任务指定；派发前要求该卡空闲显存 ≥ --min-free-mib（与他人共享卡）。
启动时记录 training_wss_min/*.py、配置 JSON 与配置引用的统计/split 文件的 sha256，每次派发前复核，变化即中止（矩阵纪律）。
可断点续跑：train.log 已有「训练完成」行 / eval/ckpt_*/metrics.json 已存在的阶段跳过（ckpt_last.pt 每轮都写，不能当完成标志）；
train 阶段遇到已存在但未完成的 run 目录拒绝覆盖。可在 Slurm 作业内运行（--gpus 传 $CUDA_VISIBLE_DEVICES）。
用法：python -m training_wss_min.tools.run_local_train_queue --config-dir <dir> --gpus 0,1 --slots-per-gpu 3
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
ROOT = PKG.parent
STAGES = ("train", "eval_best", "eval_last")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprints(config_dir: Path) -> dict:
    paths = sorted(PKG.glob("*.py")) + sorted(config_dir.glob("*.json"))
    for cfg_path in sorted(config_dir.glob("*_s*.json")):
        cfg = json.loads(cfg_path.read_text())
        for key in ("feature_stats_path", "split_path", "wss_stats_path"):
            if cfg["data"].get(key):
                paths.append(Path(cfg["data"][key]))
        if cfg["train"].get("init_reference_config"):
            paths.append(Path(cfg["train"]["init_reference_config"]))
    return {str(Path(p).resolve()): sha256(p) for p in sorted(set(paths)) if Path(p).is_file()}


def free_mib(gpu: str) -> int:
    out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def stage_done(run: Path, stage: str) -> bool:
    if stage == "train":
        log = run / "train.log"
        return log.is_file() and "训练完成" in log.read_text(errors="replace")
    return (run / "eval" / f"ckpt_{stage[5:]}" / "metrics.json").is_file()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--slots-per-gpu", type=int, default=3)
    ap.add_argument("--min-free-mib", type=int, default=9000)
    ap.add_argument("--launch-interval-seconds", type=float, default=60.0)
    ap.add_argument("--poll-seconds", type=float, default=15.0)
    ap.add_argument("--only", default="", help="comma list of arm ids or arm-id prefixes ending in '*' (default all)")
    ap.add_argument("--status-file", default="queue_status.json", help="status JSON name under experiments/<exp>/ (one per concurrent queue)")
    args = ap.parse_args()
    config_dir = Path(args.config_dir).resolve()
    matrix = json.loads((config_dir / "matrix.json").read_text())
    exp = PKG / "experiments" / matrix["experiment"]
    logs = exp / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    only = {s for s in args.only.split(",") if s}
    arms = [a for a in matrix["arms"] if not only or a["id"] in only
            or any(o.endswith("*") and a["id"].startswith(o[:-1]) for o in only)]
    if not arms:
        raise SystemExit("no arm selected")
    frozen = fingerprints(config_dir)
    gpus = args.gpus.split(",")
    status = {"host": socket.gethostname(), "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "package_dir": str(PKG), "config_dir": str(config_dir),
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "executed_outside_slurm": os.environ.get("SLURM_JOB_ID") is None,
              "selected_arms": [a["id"] for a in arms],
              "gpus": gpus, "slots_per_gpu": args.slots_per_gpu, "min_free_mib": args.min_free_mib,
              "fingerprint_files": len(frozen), "fingerprint_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
              "arms": {}}
    for a in arms:
        run = ROOT / "training_wss_min/runs" / a["run_name"]
        done = [s for s in STAGES if stage_done(run, s)]
        if run.exists() and "train" not in done:
            raise FileExistsError(f"refusing to overwrite an unfinished run directory: {run}")
        status["arms"][a["id"]] = {"run": str(run), "stages": {s: {"state": "done_before_start"} for s in done},
                                   "next": next((s for s in STAGES if s not in done), None)}
    status_path = exp / args.status_file

    def save():
        tmp = status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=1, ensure_ascii=False))
        tmp.replace(status_path)

    slots = [g for _ in range(args.slots_per_gpu) for g in gpus]   # 轮转：先每卡一个，再每卡第二个
    running = {}   # arm id -> (proc, slot, fh, stage)
    last_launch = {g: 0.0 for g in gpus}
    save()
    while True:
        for aid, (proc, slot, fh, stage) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            rec = status["arms"][aid]
            rec["stages"][stage].update(returncode=rc, ended_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                        state="done" if rc == 0 else "failed")
            idx = STAGES.index(stage)
            rec["next"] = (STAGES[idx + 1] if idx + 1 < len(STAGES) else None) if rc == 0 else "failed"
            slots.append(slot)
            del running[aid]
            print(f"{time.strftime('%H:%M:%S')} {aid} {stage} -> rc {rc}", flush=True)
            save()
        pending = [a for a in arms if status["arms"][a["id"]]["next"] not in (None, "failed") and a["id"] not in running]
        if not pending and not running:
            break
        # eval 阶段优先（释放训练结果），其次按矩阵顺序派发训练
        pending.sort(key=lambda a: (status["arms"][a["id"]]["next"] == "train", arms.index(a)))
        for a in pending:
            if not slots:
                break
            slot = next((g for g in slots if time.monotonic() - last_launch[g] >= args.launch_interval_seconds
                         and free_mib(g) >= args.min_free_mib), None)
            if slot is None:
                break
            if fingerprints(config_dir) != frozen:
                status["aborted"] = "code/config fingerprint changed during the queue"
                save()
                for proc, *_ in running.values():
                    proc.terminate()
                raise SystemExit(status["aborted"])
            slots.remove(slot)
            aid = a["id"]
            stage = status["arms"][aid]["next"]
            run = ROOT / "training_wss_min/runs" / a["run_name"]
            if stage == "train":
                if run.exists():
                    raise FileExistsError(f"run appeared while queued: {run}")
                cmd = [sys.executable, "-u", "-m", "training_wss_min.train", "--config", str(config_dir / a["config"])]
            else:
                cmd = [sys.executable, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(run), "--partitions", "test",
                       "--allow-test", "--checkpoint", stage[5:], "--no-plots", "--save-predictions"]
            log = logs / f"{aid}_{stage}.log"
            if log.exists():
                raise FileExistsError(f"refusing to overwrite log: {log}")
            fh = log.open("x")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            running[aid] = (proc, slot, fh, stage)
            last_launch[slot] = time.monotonic()
            status["arms"][aid]["stages"][stage] = {"state": "running", "gpu": slot, "pid": proc.pid, "log": str(log), "cmd": cmd,
                                                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            print(f"{time.strftime('%H:%M:%S')} launch {aid} {stage} on GPU {slot}", flush=True)
            save()
        time.sleep(args.poll_seconds)
    status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    failed = [k for k, v in status["arms"].items() if v["next"] is not None]
    print("queue finished; not complete:", failed or "none", flush=True)


if __name__ == "__main__":
    main()
