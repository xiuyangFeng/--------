"""Borrow one GPU slot using an independent, lease-owning recovery controller.

Only the verified queue coordinator is paused. Its child keeps running. The
controller owns the diagnostic srun and cancels only its uniquely named numeric
Slurm step before resuming the same coordinator process identity.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from training_wss_min.tools.m2_optimization_common import EXP, ROOT, locked_state, save_json, stamp

SLURM = "/public/slurm/bin/"
COMMAND = b"training_wss_min.tools.run_m2_optimization worker"


def identity(pid):
    directory = Path(f"/proc/{pid}")
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": int(pid), "start_ticks": fields[19], "state": fields[0],
            "uid": directory.stat().st_uid}


def same_process(expected):
    try:
        current = identity(expected["pid"])
        return all(current[k] == expected[k] for k in ("pid", "start_ticks", "uid")) and current["state"] not in {"Z", "X"}
    except (FileNotFoundError, ProcessLookupError):
        return False


def verified_worker(job, worker):
    pid = int(worker["pid"])
    current = identity(pid)
    directory = Path(f"/proc/{pid}")
    command = (directory / "cmdline").read_bytes().replace(b"\0", b" ")
    environment = (directory / "environ").read_bytes().split(b"\0")
    if current["uid"] != os.getuid() or COMMAND not in command or f"SLURM_JOB_ID={job}".encode() not in environment:
        raise RuntimeError("Worker process ownership/command/allocation mismatch")
    return current


def active_arms(state, job):
    return [a["id"] for a in state["arms"].values() if str(a.get("worker")) == str(job)
            and a["status"] in {"claimed", "train", "eval_best", "eval_last"}]


def interrupt(signum, frame):
    raise InterruptedError(f"received signal {signum}")


def unique_steps(plan):
    raw = subprocess.check_output([SLURM + "squeue", "--steps", "-h", "-j", plan["job_id"], "-o", "%i|%100j"], text=True, timeout=15)
    steps = set()
    for line in raw.splitlines():
        step, _, name = line.strip().partition("|")
        step, name = step.strip(), name.strip()
        if name == plan["step_name"]:
            prefix, dot, index = step.partition(".")
            if prefix != plan["job_id"] or not dot or not index.isdigit():
                raise RuntimeError("Refusing nonnumeric or foreign diagnostic Slurm step")
            steps.add(step)
    marker = Path(plan["directory"]) / "step_started.json"
    if marker.exists():
        data = json.loads(marker.read_text())
        if data.get("token") != plan["token"] or data.get("job_id") != plan["job_id"] or not str(data.get("step_id", "")).isdigit():
            raise RuntimeError("Diagnostic step marker identity mismatch")
        marked = f"{plan['job_id']}.{data['step_id']}"
        # The exact marker may name a step that has already exited: only return
        # it if Slurm still lists that exact ID, regardless of display name.
        if any(line.strip().split("|", 1)[0] == marked for line in raw.splitlines()):
            steps.add(marked)
    return steps


def stop_step_and_confirm(plan, process):
    # The controller's signal handlers only set a flag, so None proves it never
    # created a diagnostic step; no asynchronous exception can lose the PID.
    if process is None:
        return
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
    absent = 0
    while absent < 3:
        steps = unique_steps(plan)
        for step in steps:
            subprocess.run([SLURM + "scancel", step], check=True, timeout=15)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        absent = absent + 1 if not steps and (process is None or process.poll() is not None) else 0
        if absent < 3:
            time.sleep(2)


def controller(plan_path):
    plan = json.loads(plan_path.read_text())
    directory = Path(plan["directory"])
    evidence = {**plan, "completed": False, "controller_pid": os.getpid(), "started_at": stamp()}
    paused, process = False, None
    termination_requested = False
    def request_abort(signum, frame):
        nonlocal termination_requested
        termination_requested = True
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, request_abort)
    try:
        with locked_state() as state:
            worker = state["workers"].get(plan["job_id"], {})
            current = verified_worker(plan["job_id"], worker)
            if (worker.get("status") != "running" or len(active_arms(state, plan["job_id"])) > 1
                    or not same_process(plan["coordinator"]) or current["state"] in {"T", "t"}):
                return 99
            if termination_requested or not same_process(plan["owner"]):
                return 98
            evidence["other_active_arms"] = active_arms(state, plan["job_id"])
            paused = True
            os.kill(current["pid"], signal.SIGSTOP)
            for _ in range(100):
                if identity(current["pid"])["state"] in {"T", "t"}:
                    break
                time.sleep(.01)
            else:
                raise RuntimeError("Coordinator pause not acknowledged")
        evidence["coordinator_paused_at"] = stamp()
        save_json(directory / "controller.json", evidence)
        save_json(EXP / "borrowed_diagnostic_slot.json", evidence)
        redundant = plan.get("cancel_pending")
        if redundant:
            state = subprocess.check_output([SLURM + "squeue", "-h", "-j", redundant, "-o", "%T"], text=True, timeout=15).strip()
            if state == "PENDING":
                subprocess.run([SLURM + "scancel", "--state=PENDING", redundant], check=True, timeout=15)
                remaining = subprocess.check_output([SLURM + "squeue", "-h", "-j", redundant, "-o", "%T"], text=True, timeout=15).strip()
                if remaining:
                    raise RuntimeError(f"Redundant allocation still listed as {remaining}; refusing duplicate execution")
                evidence["cancelled_pending_allocation"] = redundant
            elif state:
                raise RuntimeError(f"Redundant diagnostic allocation is {state}; refusing duplicate execution")
        command = [SLURM + "srun", f"--jobid={plan['job_id']}", "--overlap", "-n", "1", "-c", "4",
                   f"--job-name={plan['step_name']}", "--time=00:20:00", sys.executable, "-u", "-m",
                   "training_wss_min.tools.borrow_m2_diagnostic_slot", "--run-step", str(plan_path)]
        evidence["command"] = command
        if termination_requested or (directory / "abort_requested").exists() or not same_process(plan["owner"]):
            raise RuntimeError("Borrow aborted before diagnostic launch")
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        evidence["srun_pid"] = process.pid
        save_json(directory / "controller.json", evidence)
        deadline = time.monotonic() + 1200
        while process.poll() is None:
            if termination_requested or (directory / "abort_requested").exists() or not same_process(plan["owner"]):
                raise RuntimeError("Borrow owner exited or requested abort; cancelling only the diagnostic step")
            if time.monotonic() >= deadline:
                raise TimeoutError("Bounded diagnostic step exceeded 1200 seconds")
            time.sleep(1)
        evidence.update(returncode=process.returncode, completed=process.returncode == 0)
        if process.returncode:
            raise RuntimeError(f"Diagnostic step exited {process.returncode}")
        return 0
    except BaseException as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # A separate session and inherited full-lifetime lease let this recovery
        # controller survive the caller's SIGTERM/SIGKILL. Further ordinary
        # termination signals cannot interrupt the critical cleanup sequence.
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGHUP, signal.SIGINT})
        if paused:
            while True:
                try:
                    stop_step_and_confirm(plan, process)
                    evidence["diagnostic_step_absent_confirmed_at"] = stamp()
                    break
                except Exception as exc:
                    evidence["cleanup_retry"] = f"{type(exc).__name__}: {exc}"
                    save_json(directory / "controller.json", evidence)
                    time.sleep(5)
            if same_process(plan["coordinator"]):
                os.kill(plan["coordinator"]["pid"], signal.SIGCONT)
                evidence["coordinator_resumed_at"] = stamp()
            else:
                evidence["coordinator_already_exited"] = True
        evidence["ended_at"] = stamp()
        save_json(directory / "controller.json", evidence)
        save_json(EXP / "borrowed_diagnostic_slot.json", evidence)


def run_step(plan_path):
    plan = json.loads(plan_path.read_text())
    job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID", "")
    if job != plan["job_id"] or not step.isdigit() or os.getuid() != plan["owner"]["uid"]:
        raise RuntimeError("Wrong diagnostic Slurm step identity")
    directory = Path(plan["directory"])
    save_json(directory / "step_started.json", {"token": plan["token"], "job_id": job, "step_id": step,
              "wrapper": identity(os.getpid()), "started_at": stamp()})
    try:
        return subprocess.call(["bash", str(ROOT / "training_wss_min/cluster/compare_m2_training_trajectory.slurm")], cwd=ROOT)
    finally:
        save_json(directory / "step_ended.json", {"token": plan["token"], "job_id": job, "step_id": step, "ended_at": stamp()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path)
    parser.add_argument("--run-step", type=Path)
    parser.add_argument("--cancel-pending", default="13978")
    args = parser.parse_args()
    if args.controller:
        return controller(args.controller)
    if args.run_step:
        return run_step(args.run_step)
    EXP.mkdir(parents=True, exist_ok=True)
    with (EXP / "borrowed_diagnostic_slot.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, interrupt)
        while True:
            selected = None
            with locked_state() as state:
                if state["status"] != "running":
                    raise RuntimeError("Base queue no longer running; use a separate GPU allocation")
                for job, worker in state["workers"].items():
                    if worker["status"] != "running" or len(active_arms(state, job)) > 1:
                        continue
                    current = verified_worker(job, worker)
                    if current["state"] in {"T", "t"}:
                        continue
                    selected = {"job_id": str(job), "coordinator": current}
                    break
            if selected is None:
                time.sleep(5)
                continue
            token = uuid.uuid4().hex
            directory = EXP / "borrowed_slots" / token
            directory.mkdir(parents=True)
            plan = {**selected, "token": token, "step_name": "m2b_" + token[:16],
                    "directory": str(directory), "owner": identity(os.getpid()), "cancel_pending": args.cancel_pending}
            plan_path = directory / "plan.json"
            save_json(plan_path, plan)
            with (directory / "controller.log").open("w") as log:
                process = subprocess.Popen([sys.executable, "-u", "-m", "training_wss_min.tools.borrow_m2_diagnostic_slot",
                    "--controller", str(plan_path)], cwd=ROOT, start_new_session=True,
                    pass_fds=(lease.fileno(),), stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                except BaseException:
                    (directory / "abort_requested").write_text(stamp())
                    # Do not terminate the independent controller: it owns the
                    # lease, exact srun child and safe remote-step cleanup.
                    raise
            if code == 99:
                time.sleep(1)
                continue
            if code:
                raise RuntimeError(f"Recovery controller exited {code}; inspect {directory}")
            print(json.dumps({"completed": True, "evidence": str(directory / "controller.json")}), flush=True)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
