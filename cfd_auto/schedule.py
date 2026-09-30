"""Run schedules of a managed CFD run: the journal (2.jou), the kept frames and the Slurm script (fluent.slurm).

Schedules (``protocol["schedules"]["ladder"]`` is the divergence ladder):
  library  dt x n_steps (the library 2.jou)
  gentle   first cardiac cycle at dt/refine, then dt; the step count keeps the library phase of every step from the end of
           the fine cycle (t = N dt - period), so the exported steps and their labels are the library ones
  dt2      the whole run at dt/refine (n_steps x refine native steps); the last cycle's frames at the library phases are
           kept (every ``every * refine`` native steps) and relabelled N -> N / refine after the run (``cfd_auto.relabel``)

Only the needed frames are exported (2026-09-30 probes on v231): the journal saves the case's automatic-export list,
switches it off (``(rpsetvar 'export/automatic '())``) until the first kept step, then restores it with the export
frequency set to the kept-frame spacing. Fluent applies the frequency to the global time-step index (freq 2 -> even
steps), and a run split into two ``dual-time-iterate`` calls writes byte-identical exports to an unsplit run. A library
run thus writes 81 + 81 frames instead of 1280 + 1280 (volume frames are ~130 MB each).

The Slurm script is requeue-safe: a job restarted by Slurm (cluster reboot) first deletes every output of a previous
attempt, because Fluent silently skips an export whose file already exists. It refuses to run outside a cfd_auto work
directory (marker file), keeps a safety-net janitor for unexpected frames, and ends in the library layout (wall frames in
``ascii/``, volume frames in ``ascii_in/``, transcripts and monitors in ``Global_conditions/``).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from cfd_auto import guard

MARKER = ".cfd_auto_workdir"
KEEP_FILE = ".keep_steps"
FRAMES_FILE = "frames.json"
ATTEMPT_MARK = "=== cfd_auto attempt start"
FINISH_MARK = "=== cfd_auto run finished"
TRANSCRIPT_GLOB = "Fluent_cfdauto_*.out"
FREQ_FN = ("(define (cfdauto-freq lst f) (map (lambda (e) (cons (car e) (map (lambda (p) (if (and (pair? p) (eq? (car p) 'frequency)) "
           "(cons 'frequency f) p)) (cdr e)))) lst))")


@dataclasses.dataclass
class Schedule:
    name: str
    stages: list[tuple[float, int]]        # (time step s, native steps), in order
    max_iter: int
    keep: list[int]                        # native steps whose exports are kept
    refine: int = 1                        # native step N carries library label N / refine
    export_every: int = 2                  # export frequency (global step index) inside the kept window
    relax: dict | None = None              # optional under-relaxation overrides {"mom": 0.5, "pressure": 0.2}

    @property
    def n_native(self) -> int:
        return sum(n for _, n in self.stages)

    @property
    def export_from(self) -> int:
        return min(self.keep)

    def label(self, native: int) -> int:
        return native // self.refine

    def to_json(self) -> dict:
        return {**dataclasses.asdict(self), "n_native": self.n_native, "export_from": self.export_from,
                "labels": {str(n): self.label(n) for n in self.keep}}


def make(name: str, proto: dict) -> Schedule:
    s, e, sc = proto["solver"], proto["exports"], proto["schedules"]
    dt, n, it, period = s["time_step_s"], s["n_steps"], s["max_iter"], s["period_s"]
    labels = list(range(e["first_step"], e["last_step"] + 1, e["every"]))
    if name == "library":
        return Schedule("library", [(dt, n)], it, labels, 1, e["every"])
    if name == "gentle":
        r = sc["gentle_refine"]
        fine = int(round(period / (dt / r)))
        if abs(fine * dt / r - period) > 1e-9 or fine >= n or fine >= min(labels):
            raise ValueError("gentle: the fine stage must be one whole period ending before the first exported step")
        return Schedule("gentle", [(dt / r, fine), (dt, n - fine)], it, labels, 1, e["every"])
    if name == "dt2":
        r = sc["dt2_refine"]
        return Schedule("dt2", [(dt / r, n * r)], it, [x * r for x in labels], r, e["every"] * r)
    raise ValueError(f"unknown schedule {name!r}")


def journal(workdir: Path, case: Path, sch: Schedule) -> str:
    lines = [f"/file/read-case {case}", "(define cfdauto-exports (rpgetvar 'export/automatic))", FREQ_FN, "(rpsetvar 'export/automatic '())",
             "/solve/initialize/initialize-flow"]
    for k, v in (sch.relax or {}).items():
        lines.append(f"/solve/set/under-relaxation/{k} {v:g}")
    on = f"(rpsetvar 'export/automatic (cfdauto-freq cfdauto-exports {sch.export_every}))"
    done = 0
    for dt, n in sch.stages:
        lines.append(f"/solve/set/time-step {dt:g}")
        first, last = done + 1, done + n
        if first <= sch.export_from <= last:
            pre = sch.export_from - first
            if pre:
                lines.append(f"/solve/dual-time-iterate {pre} {sch.max_iter}")
            lines += [on, f"/solve/dual-time-iterate {n - pre} {sch.max_iter}"]
        else:
            lines.append(f"/solve/dual-time-iterate {n} {sch.max_iter}")
        done = last
    lines.append("/exit y")
    text = "\n".join(lines) + "\n"
    guard.check_journal(text, workdir)
    return text


def slurm_script(sch: Schedule, cores: int, job_name: str, fluent_module: str, exclude: str | None = "node05", time_limit: str | None = None,
                 make_dirs: tuple[str, ...] = ("export",)) -> str:
    dirs = " ".join(sorted({"ascii", *make_dirs}))
    clear = " ".join(f'"{d}"' for d in sorted(set(make_dirs) - {"ascii"}))
    opt = [f"#SBATCH --exclude={exclude}"] if exclude else []
    opt += [f"#SBATCH --time={time_limit}"] if time_limit else []
    return "\n".join([
        "#!/bin/bash",
        "#SBATCH --partition=CPU",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks-per-node={cores}",
        *opt,
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=Fluent_cfdauto_%j.out",
        "#SBATCH --error=Fluent_cfdauto_%j.err",
        f"# cfd_auto managed run, schedule {sch.name}: {sch.n_native} native steps, {len(sch.keep)} kept frames (native {min(sch.keep)}..{max(sch.keep)}).",
        "# Generated by cfd_auto.schedule; regenerate instead of editing.",
        "set -u",
        'cd "$SLURM_SUBMIT_DIR"',
        f'[ -f {MARKER} ] || {{ echo "cfd_auto: $PWD is not a cfd_auto work directory"; exit 3; }}',
        f'echo "{ATTEMPT_MARK} job=$SLURM_JOB_ID restart=${{SLURM_RESTART_COUNT:-0}} host=$(hostname) $(date -Is)"',
        'echo "job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) start=$(date -Is)" >> run_attempts.log',
        "# requeue safety: Fluent never overwrites an existing export, so every attempt starts from a clean slate",
        "rm -f ./*-[0-9][0-9][0-9][0-9] ./*-rfile.out ./*.trn ./cleanup-fluent-*.sh",
        "rm -rf ascii ascii_in Global_conditions",
        *([f"find {clear} -mindepth 1 -delete 2>/dev/null"] if clear else []),
        f"mkdir -p {dirs}",
        "declare -A KEEP",
        f"while read -r s; do [ -n \"$s\" ] && KEEP[$s]=1; done < {KEEP_FILE}",
        "prune() {  # delete numbered exports that are not kept (older than step $1)",
        "  for f in ./*-[0-9][0-9][0-9][0-9] ascii/*-[0-9][0-9][0-9][0-9]; do",
        "    [ -f \"$f\" ] || continue; n=$((10#${f##*-}))",
        "    [ \"$n\" -lt \"$1\" ] && [ -z \"${KEEP[$n]:-}\" ] && rm -f \"$f\"",
        "  done; return 0",
        "}",
        "LOG=Fluent_cfdauto_$SLURM_JOB_ID.out",
        "( while sleep 120; do cur=$(tail -c 400000 \"$LOG\" 2>/dev/null | grep -o 'time step = [0-9]*' | tail -1 | grep -o '[0-9]*$'); "
        "[ -n \"$cur\" ] && prune $((cur - 5)); done ) &",
        "JANITOR=$!",
        "export I_MPI_SHM_LMT=shm",
        "module purge",
        f"module load {fluent_module}",
        f"time fluent 3ddp -g -t{cores} -mpi=intel -platform=intel -i ./2.jou -ssh",
        "pkill -P $JANITOR 2>/dev/null; kill $JANITOR 2>/dev/null; wait $JANITOR 2>/dev/null",   # its sleep first, or it outlives the loop
        "prune 999999",
        "mkdir -p ascii_in",
        "mv ./*-[0-9][0-9][0-9][0-9] ./ascii_in/ 2>/dev/null",
        "rm -f ./*.trn ./cleanup-fluent-*.sh",
        f'echo "{FINISH_MARK} $(date -Is)"',
        "mkdir -p Global_conditions",
        "mv ./*.out ./Global_conditions/ 2>/dev/null",
        "exit 0",
    ]) + "\n"


def write_run_files(work: Path, case: Path, sch: Schedule, cores: int, job_name: str, fluent_module: str, make_dirs: tuple[str, ...] = ("export",),
                    exclude: str | None = "node05", time_limit: str | None = None) -> dict:
    """2.jou + fluent.slurm + kept-step list + frames.json + the work-directory marker. Refuses a directory that still holds
    exports (the Slurm script clears them itself, but a prepared directory must start clean)."""
    work = guard.assert_inside(work, work)
    (work / MARKER).write_text("cfd_auto work directory: the managed fluent.slurm may delete outputs here\n")
    (work / "2.jou").write_text(journal(work, case, sch))
    (work / "fluent.slurm").write_text(slurm_script(sch, cores, job_name, fluent_module, exclude, time_limit, make_dirs))
    (work / KEEP_FILE).write_text("".join(f"{n}\n" for n in sch.keep))
    rep = {"schedule": sch.to_json(), "cores": cores, "job_name": job_name, "case": str(case)}
    (work / FRAMES_FILE).write_text(json.dumps(rep, indent=1))
    for d in ("ascii", *make_dirs):
        (work / d).mkdir(parents=True, exist_ok=True)
    return rep


def autosave_dirs(case_rp_autosave: str) -> tuple[str, ...]:
    """Relative directory of the template's autosave file name (e.g. 'export/NAME.gz' -> ('export',))."""
    import re
    names = re.findall(r'"([^"]*)"', case_rp_autosave or "")
    if names and "/" in names[0] and not names[0].startswith("/"):
        return (str(Path(names[0]).parent),)
    return ()
