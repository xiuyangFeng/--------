"""Slurm one-GPU workers share an atomic queue; each runs at most two arms.

Preparing a phase, executing workers and finalising evidence are deliberately
separate: submission is never reported as experimental completion.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from training_wss_min.tools.m2_optimization_common import (
    CONFIGS, EXP, ROOT, RUNS, fingerprints, locked_state, manifest_path,
    save_json, sha, stamp, state_path,
)


def visible_gpu():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    if not os.environ.get("SLURM_JOB_ID") or len(ids) != 1 or ids[0] in {"-1", "NoDevFiles"}:
        raise RuntimeError("A Slurm worker must have exactly one allocated visible GPU")
    return ids[0]


def free_memory(gpu):
    out = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free",
                                   "--format=csv,noheader,nounits"], text=True, timeout=15)
    return float(out.strip())


def required_memory(record):
    peak = float(record["max_memory_reserved_mib"])
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("Preflight must measure a finite positive train/eval memory peak")
    return max(7000., 1.25 * peak + 2048.)


def initialize(phase):
    manifest = json.loads(manifest_path(phase).read_text())
    gate_path = EXP / ("runtime_preflight.json" if phase == "base" else "combination_runtime_preflight.json")
    gate = json.loads(gate_path.read_text())
    start = fingerprints(phase)
    if not gate.get("passed") or gate.get("source_sha256") != start or gate.get("source_sha256_end") != start:
        raise RuntimeError("A passing runtime preflight with matching immutable sources is required")
    if phase == "base":
        smoke = json.loads((EXP / "cli_smoke.json").read_text())
        if not smoke.get("passed") or smoke.get("source_sha256") != start or smoke.get("source_sha256_end") != start:
            raise RuntimeError("Five passing isolated CLI smokes and historical M2 re-evaluation are required")
    ids = [a["id"] for a in manifest["arms"]]
    if len(set(ids)) != len(ids) or not ids or set(ids) != set(gate["arms"]):
        raise RuntimeError("Matrix/preflight arm set mismatch")
    if phase == "base" and len(ids) != 20:
        raise RuntimeError("The fixed base phase has exactly 20 arms")
    if phase == "combination" and len(ids) > 4:
        raise RuntimeError("At most four combinations are authorised")
    with locked_state(phase) as previous:
        if previous is not None:
            raise RuntimeError("Queue already exists; use explicit retry for failed stages")
        state = dict(phase=phase, status="running", started_at=stamp(), source_sha256=start,
            manifest_sha256=sha(manifest_path(phase)), preflight_sha256=sha(gate_path),
            workers={}, arms={})
        for arm in manifest["arms"]:
            run = RUNS / arm["run_name"]
            if run.exists():
                raise RuntimeError(f"Refusing to overwrite existing run {run}")
            state["arms"][arm["id"]] = dict(id=arm["id"], phase=phase, parent=arm["parent"],
                family=arm["family"], config=str(CONFIGS / arm["config"]), run_name=arm["run_name"],
                hypothesis=arm["hypothesis"], status="pending", next_stage="train", stages={},
                attempts=[], min_free_mib=required_memory(gate["arms"][arm["id"]]))
        snapshot = EXP / f"source_snapshot_{phase}"
        if snapshot.exists():
            raise RuntimeError(f"Snapshot already exists: {snapshot}")
        for relative, expected in start.items():
            source = ROOT / relative
            if sha(source) != expected:
                raise RuntimeError("Sources changed while making frozen snapshot")
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        state["source_snapshot"] = str(snapshot)
        save_json(state_path(phase), state)
    print(f"Initialised {phase}: {len(ids)} arms", flush=True)


def worker(phase, slots):
    gpu = visible_gpu()
    job = os.environ["SLURM_JOB_ID"]
    initial_hashes = fingerprints(phase)
    with locked_state(phase) as state:
        if state is None or state["source_sha256"] != initial_hashes or state["status"] != "running":
            raise RuntimeError("Missing/currently invalid queue provenance")
        if job in state["workers"]:
            raise RuntimeError("Worker job already registered")
        state["workers"][job] = dict(status="running", gpu=gpu, started_at=stamp(), pid=os.getpid(),
            python=sys.executable, source_sha256=initial_hashes, slots=slots)
    log_dir = EXP / ("logs" if phase == "base" else "combination_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    active = {}
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        for task in active.values():
            if task["proc"].poll() is None:
                os.killpg(task["proc"].pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def launch(aid, stage, rec):
        if fingerprints(phase) != initial_hashes:
            raise RuntimeError("Sources changed; refusing to launch another stage")
        run = RUNS / rec["run_name"]
        if stage == "train":
            if run.exists():
                raise RuntimeError(f"Existing training directory: {run}")
            command = [sys.executable, "-u", "-m", "training_wss_min.train", "--config", rec["config"]]
        else:
            command = [sys.executable, "-u", "-m", "training_wss_min.evaluate", "--run-dir", str(run),
                       "--partitions", "test", "--allow-test", "--checkpoint", stage.removeprefix("eval_"), "--no-plots"]
        log = log_dir / f"{aid}_{stage}_{job}_{time.time_ns()}.log"
        handle = log.open("w")
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        info = dict(job_id=job, started_at=stamp(), log=str(log), command=command, pid=proc.pid)
        with locked_state(phase) as state:
            target = state["arms"][aid]
            target.update(status=stage, worker=job, gpu=gpu, source_sha256=initial_hashes)
            target["stages"][stage] = info
        active[aid] = dict(stage=stage, proc=proc, handle=handle, start=time.time(), rec=rec)
        print(f"{stamp()} {job} {aid} {stage} gpu={gpu}", flush=True)

    try:
        last_launch = 0.
        while not stopped:
            for aid, task in list(active.items()):
                code = task["proc"].poll()
                if code is None:
                    continue
                task["handle"].close()
                del active[aid]
                stage = task["stage"]
                with locked_state(phase) as state:
                    rec = state["arms"][aid]
                    rec["stages"][stage].update(returncode=code, ended_at=stamp(), elapsed_seconds=time.time() - task["start"])
                    rec["attempts"].append(dict(stage=stage, **rec["stages"][stage]))
                    if code:
                        rec.update(status="failed", failed_stage=stage, source_sha256_end=fingerprints(phase))
                    elif stage == "eval_last":
                        rec.update(status="complete", ended_at=stamp(), source_sha256_end=fingerprints(phase))
                    else:
                        rec["next_stage"] = "eval_best" if stage == "train" else "eval_last"
                        rec["status"] = "pending"
                print(f"{stamp()} {aid} {stage} exit={code}", flush=True)
            if len(active) < slots and time.time() - last_launch >= 15:
                try:
                    free = free_memory(gpu)
                except (subprocess.SubprocessError, ValueError):
                    free = 0.
                claim = None
                with locked_state(phase) as state:
                    # Prefer finishing already trained arms before allocating another model.
                    pending = sorted((r for r in state["arms"].values() if r["status"] == "pending"),
                                     key=lambda r: (r["next_stage"] == "train", list(state["arms"]).index(r["id"])))
                    for rec in pending:
                        if free >= rec["min_free_mib"]:
                            rec.update(status="claimed", worker=job, claimed_at=stamp(), gpu=gpu)
                            claim = copy.deepcopy(rec)
                            break
                if claim:
                    launch(claim["id"], claim["next_stage"], claim)
                    last_launch = time.time()
            with locked_state(phase) as state:
                unfinished = any(r["status"] not in {"complete", "failed", "interrupted"} for r in state["arms"].values())
            if not active and not unfinished:
                break
            time.sleep(5)
    finally:
        for aid, task in active.items():
            if task["proc"].poll() is None:
                os.killpg(task["proc"].pid, signal.SIGTERM)
                try:
                    task["proc"].wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(task["proc"].pid, signal.SIGKILL)
                    task["proc"].wait()
            task["handle"].close()
        end = fingerprints(phase)
        with locked_state(phase) as state:
            for aid, task in active.items():
                rec = state["arms"][aid]
                stage = task["stage"]
                evidence = rec["stages"][stage]
                evidence.update(returncode=task["proc"].returncode, ended_at=stamp(),
                                elapsed_seconds=time.time() - task["start"], interrupted=True)
                rec["attempts"].append(dict(stage=stage, **evidence))
            for aid, rec in state["arms"].items():
                if rec.get("worker") == job and rec["status"] not in {"complete", "failed", "pending"}:
                    rec.update(status="interrupted", failed_stage=rec.get("next_stage", "train"), source_sha256_end=end)
            state["workers"][job].update(status="interrupted" if stopped else "complete", ended_at=stamp(),
                source_sha256_end=end, source_changed=end != initial_hashes)
    return 1 if stopped or end != initial_hashes else 0


def finalize(phase):
    end = fingerprints(phase)
    with locked_state(phase) as state:
        if state is None:
            raise RuntimeError("No queue")
        if any(w["status"] == "running" for w in state["workers"].values()):
            raise RuntimeError("Workers have not all ended")
        changed = end != state["source_sha256"] or any(w.get("source_changed") for w in state["workers"].values())
        success = all(r["status"] == "complete" and all(r["stages"].get(s, {}).get("returncode") == 0
                     for s in ("train", "eval_best", "eval_last")) for r in state["arms"].values())
        state.update(status="complete" if success and not changed else "incomplete", ended_at=stamp(),
                     source_sha256_end=end, source_changed=changed)
        print(f"Finalized {phase}: {state['status']}")
    return 0 if success and not changed else 1


def retry(phase, ids):
    with locked_state(phase) as state:
        if any(w["status"] == "running" for w in state["workers"].values()):
            raise RuntimeError("Retry only after phase workers have ended")
        if state["source_sha256"] != fingerprints(phase):
            raise RuntimeError("Retry must use the same source/configuration")
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Retry requires unique explicit failed arm IDs")
        # Complete validation before moving any partial run directory. A rejected
        # batch must leave both the filesystem and queue state untouched.
        for aid in ids:
            rec = state["arms"].get(aid)
            if rec is None or rec["status"] not in {"failed", "interrupted"}:
                raise RuntimeError(f"Cannot retry nonfailed arm {aid}")
            if rec.get("failed_stage") not in {"train", "eval_best", "eval_last"}:
                raise RuntimeError(f"Unknown failed stage for {aid}")
        for aid in ids:
            rec = state["arms"][aid]
            stage = rec["failed_stage"]
            if stage == "train":
                run = RUNS / rec["run_name"]
                if run.exists():
                    archive = EXP / "failed_attempts" / f"{aid}_{time.time_ns()}"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(run), archive)
                    rec.setdefault("failed_run_archives", []).append(str(archive))
                    save_json(state_path(phase), state)
                rec["stages"] = {}
            rec.update(status="pending", next_stage=stage)
        state.update(status="running", retry_started_at=stamp())
        for key in ("ended_at", "source_sha256_end", "source_changed"):
            state.pop(key, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("initialize", "worker", "finalize", "retry"))
    ap.add_argument("--phase", choices=("base", "combination"), default="base")
    ap.add_argument("--slots", type=int, choices=(1, 2), default=2)
    ap.add_argument("--arms", nargs="+", default=[])
    args = ap.parse_args()
    if args.action == "initialize":
        initialize(args.phase)
    elif args.action == "worker":
        return worker(args.phase, args.slots)
    elif args.action == "finalize":
        return finalize(args.phase)
    else:
        retry(args.phase, args.arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
