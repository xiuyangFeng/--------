"""Fluent batch jobs on the CPU partition (the solver/mesher must never run on the login node)."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from cfd_auto import guard

SLURM_BIN = "/public/slurm/bin"
FLUENT_MODULE = "fluent/231"
EXCLUDE = "node05"


def _env() -> dict:
    env = dict(os.environ)
    env["PATH"] = f"{SLURM_BIN}:{env.get('PATH', '')}"
    return env


def submit(workdir: Path, journal: Path, mode: str = "solver", ntasks: int = 4, time_limit: str = "01:00:00") -> tuple[str, Path]:
    """Submit ``fluent 3ddp [-meshing] -g -t<n> -i <journal>`` with cwd = workdir. ntasks=0 -> serial Fluent."""
    workdir = guard.assert_inside(workdir, workdir)
    guard.assert_inside(journal, workdir)
    guard.check_journal(Path(journal).read_text(), workdir)
    logs = workdir / "logs"; logs.mkdir(exist_ok=True)
    tag = Path(journal).stem
    tflag, slots = ("", 1) if ntasks == 0 else (f"-t{ntasks}", ntasks)
    mflag = "-meshing" if mode == "meshing" else ""
    script = logs / f"_{tag}.slurm"
    script.write_text(f"""#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude={EXCLUDE}
#SBATCH --ntasks-per-node={slots}
#SBATCH --job-name=cfdauto_{tag}
#SBATCH --output={logs}/{tag}_%j.out
#SBATCH --time={time_limit}
export I_MPI_SHM_LMT=shm
module purge
module load {FLUENT_MODULE}
cd {workdir}
fluent 3ddp {mflag} -g {tflag} -mpi=intel -platform=intel -i {journal} -ssh 2>&1
mv -f {workdir}/*.trn {logs}/ 2>/dev/null
rm -f {workdir}/cleanup-fluent-*.sh
""")
    out = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True, env=_env(), check=True).stdout.strip()
    if not out.isdigit():
        raise RuntimeError(f"sbatch returned {out!r}")
    return out, logs / f"{tag}_{out}.out"


def wait(job_id: str, poll_s: float = 5.0, timeout_s: float = 7200) -> str:
    """Block until the job leaves the queue; return its final sacct state."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        q = subprocess.run(["squeue", "-h", "-j", job_id], capture_output=True, text=True, env=_env()).stdout.strip()
        if not q:
            break
        time.sleep(poll_s)
    else:
        raise TimeoutError(f"job {job_id} still queued after {timeout_s} s")
    st = subprocess.run(["sacct", "-j", job_id, "-n", "-X", "-o", "State"], capture_output=True, text=True, env=_env()).stdout.split()
    return st[0] if st else "UNKNOWN"


def run(workdir: Path, journal: Path, mode: str = "solver", ntasks: int = 4, time_limit: str = "01:00:00") -> Path:
    """Submit, wait and fail loudly on Fluent errors (the batch exits 0 even when a journal line fails)."""
    job, log = submit(workdir, journal, mode, ntasks, time_limit)
    state = wait(job)
    text = log.read_text(errors="replace") if log.exists() else ""
    bad = [ln for ln in text.splitlines() if ln.startswith(("Error:", "Warning: An error or interrupt")) or "invalid command" in ln or "Mesh check failed" in ln]
    if state != "COMPLETED" or bad:
        raise RuntimeError(f"{journal.name}: job {job} {state}; {bad[:5]} (log {log})")
    return log
