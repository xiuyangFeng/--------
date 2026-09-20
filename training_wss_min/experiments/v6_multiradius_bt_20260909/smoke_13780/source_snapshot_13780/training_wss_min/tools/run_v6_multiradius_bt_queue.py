"""Slurm-only queue for the six authorized multi-radius/BT arms.

Every slot trains one arm and evaluates its best/last checkpoints before taking
another arm. Source configs and historical runs are never rewritten. Short
preflight runs get their own config/run names and are not scientific results.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "training_wss_min/experiments/v6_multiradius_bt_20260909"
MATRIX = ROOT / "training_wss_min/configs/v6_multiradius_bt_20260909/matrix.json"
RUNS = ROOT / "training_wss_min/runs"


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def visible_gpus():
    # CUDA_VISIBLE_DEVICES belongs to Slurm. Preserve its identifiers (UUIDs
    # or indices); never select a physical GPU outside the allocation.
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw or raw in {"-1", "NoDevFiles"}:
        raise RuntimeError("Slurm must expose its allocated CUDA_VISIBLE_DEVICES")
    return [v.strip() for v in raw.split(",") if v.strip()]


def gpu_free(gpu):
    out = subprocess.check_output([
        "nvidia-smi", "-i", gpu, "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    return int(out.strip())


def fingerprints():
    paths = sorted((ROOT / "training_wss_min").glob("*.py"))
    paths += sorted((ROOT / "training_wss_min/configs/v6_multiradius_bt_20260909").glob("*.json"))
    paths += [Path(__file__).resolve(), ROOT / "training_wss_min/tools/prepare_v6_multiradius_bt_matrix.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, default=MATRIX)
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--slots-per-gpu", type=int, default=2)
    ap.add_argument("--min-free-mib", type=int, default=7000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=2, help="Smoke only")
    ap.add_argument("--gate", type=Path, default=EXP / "training_gate.json")
    args = ap.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU training/evaluation must be submitted with sbatch")
    if args.slots_per_gpu < 1:
        raise ValueError("slots-per-gpu must be positive")
    matrix = json.loads(args.matrix.read_text())
    selected = [a for a in matrix["arms"] if not args.arms or a["id"] in args.arms]
    if args.arms and set(args.arms) != {a["id"] for a in selected}:
        raise ValueError("Unknown or duplicate arm selection")
    if any(a["id"] not in {f"MS{i}" for i in range(1, 7)} for a in selected):
        raise ValueError("Only MS1-MS6 authorized for this queue")
    if not selected or len({a['id'] for a in selected}) != len(selected):
        raise ValueError("Empty or duplicate matrix")
    if not args.smoke:
        gate = json.loads(args.gate.read_text())
        if not gate.get('passed') or gate.get('source_sha256') != fingerprints():
            raise RuntimeError('Formal training requires a passing gate matching all current sources/configs')
        if set(a['id'] for a in selected) != {f'MS{i}' for i in range(1,7)}:
            raise RuntimeError('Formal matrix must contain all six approved arms')
    job = os.environ["SLURM_JOB_ID"]
    state_path = EXP / (f"smoke_{job}/queue_status.json" if args.smoke else "queue_status.json")
    if state_path.exists():
        raise RuntimeError(f"State already exists: {state_path}; inspect previous jobs before restarting")
    log_dir = EXP / (f"smoke_{job}/logs" if args.smoke else "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "job_id": job, "started_at": stamp(), "mode": "smoke" if args.smoke else "formal",
        "status": "running", "gpus": visible_gpus(), "slots_per_gpu": args.slots_per_gpu,
        "min_free_mib": args.min_free_mib, "source_sha256": fingerprints(), "arms": {},
    }
    for arm in selected:
        cfg_path = args.matrix.parent / arm["config"]
        cfg = json.loads(cfg_path.read_text())
        if args.smoke:
            cfg = copy.deepcopy(cfg)
            cfg["name"] = f"v6_multiradius_bt_20260909/smoke_{job}/{Path(arm['config']).stem}"
            cfg["train"].update(epochs=args.epochs, min_epoch=args.epochs, eval_every=args.epochs)
            cfg_path = EXP / f"smoke_{job}/configs" / arm["config"]
            save_json(cfg_path, cfg)
        elif (RUNS / cfg["name"]).exists():
            raise RuntimeError(f"Refusing to overwrite an existing run: {cfg['name']}")
        state["arms"][arm["id"]] = {
            "config": str(cfg_path), "run_name": cfg["name"], "status": "pending",
            "stages": {}, "hypothesis": arm["hypothesis"],
        }
    save_json(state_path, state)
    snapshot_dir = state_path.parent / f"source_snapshot_{job}"
    for relative, expected in state['source_sha256'].items():
        src = ROOT / relative
        if hashlib.sha256(src.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source changed before snapshot: {relative}')
        dst = snapshot_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    save_json(state_path.parent / f'environment_{job}.json', dict(
        python=sys.version,executable=sys.executable,job_id=job,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        snapshot=str(snapshot_dir),source_sha256=state['source_sha256']))
    pending = [a["id"] for a in selected]
    active = {}
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        for task in active.values():
            try:
                os.killpg(task["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def launch(arm_id, gpu, stage):
        rec = state["arms"][arm_id]
        path = log_dir / f"{arm_id}_{stage}.log"
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if stage == "train":
            cmd = [sys.executable, "-u", "-m", "training_wss_min.train", "--config", rec["config"]]
        else:
            cmd = [sys.executable, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(RUNS / rec["run_name"]),
                   "--partitions", "test", "--allow-test", "--checkpoint", stage.removeprefix("eval_"), "--no-plots"]
        handle = path.open("w")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        rec.update(status=stage, gpu=gpu, pid=proc.pid)
        rec["stages"][stage] = {"started_at": stamp(), "log": str(path), "command": cmd}
        active[arm_id] = {"gpu": gpu, "stage": stage, "proc": proc, "log": handle, "start": time.time()}
        print(f"{stamp()} {arm_id} {stage} gpu={gpu} pid={proc.pid}", flush=True)
        save_json(state_path, state)

    last_launch = 0.0
    while (pending or active) and not stopped:
        for arm_id, task in list(active.items()):
            code = task["proc"].poll()
            if code is None:
                continue
            task["log"].close()
            rec = state["arms"][arm_id]
            rec["stages"][task["stage"]].update(ended_at=stamp(), returncode=code, elapsed_seconds=time.time() - task["start"])
            del active[arm_id]
            if code:
                rec.update(status="failed", failed_stage=task["stage"])
                print(f"{stamp()} {arm_id} FAILED stage={task['stage']} code={code}", flush=True)
            elif task["stage"] == "train":
                if args.smoke:
                    rec["status"] = "complete"
                else:
                    launch(arm_id, task["gpu"], "eval_best")
            elif task["stage"] == "eval_best":
                launch(arm_id, task["gpu"], "eval_last")
            else:
                rec["status"] = "complete"
            save_json(state_path, state)
        # Give a process time to load CUDA before checking headroom for another.
        if pending and time.time() - last_launch > 10:
            candidates = []
            for gpu in state["gpus"]:
                count = sum(t["gpu"] == gpu for t in active.values())
                if count >= args.slots_per_gpu:
                    continue
                try:
                    free = gpu_free(gpu)
                except (subprocess.SubprocessError, ValueError) as exc:
                    print(f"{stamp()} GPU observation unavailable {gpu}: {exc}", flush=True)
                    continue
                if free >= args.min_free_mib:
                    candidates.append((count, -free, gpu))
            if candidates:
                gpu = sorted(candidates)[0][2]
                launch(pending.pop(0), gpu, "train")
                last_launch = time.time()
        time.sleep(3)
    if stopped:
        for arm_id, task in active.items():
            try:
                task["proc"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(task["proc"].pid, signal.SIGKILL)
            task["log"].close()
            state["arms"][arm_id]["status"] = "interrupted"
    state["source_sha256_end"] = fingerprints()
    state["source_changed"] = state["source_sha256_end"] != state["source_sha256"]
    success = all(v["status"] == "complete" for v in state["arms"].values()) and not state["source_changed"]
    state.update(status="complete" if success else "incomplete", ended_at=stamp())
    save_json(state_path, state)
    print(f"{stamp()} queue={state['status']} source_changed={state['source_changed']}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
