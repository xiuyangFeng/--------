"""Health of a (running or finished) managed CFD run, read from the tail of its Fluent transcript.

A diverged Fluent batch run does not fail the Slurm job (the script goes on to its clean-up and exits 0), so the outcome
is decided here: ``completed`` (last native step reached and the script finished), ``diverged`` (floating point exception
/ divergence message, or the final continuity residual blowing up), ``diverging`` (still running, residual high for
several steps — worth cancelling), ``ended_early`` (the script finished short of the last step without a divergence
sign: node / licence / time limit), ``running``, ``no_transcript``.

WANG_CAI-0/before (2026-09-29, job 16100): the final continuity residual per step went 3e-4 -> 1e-2 -> 0.24 -> 130, then
'Error at Node k: floating point exception' on every rank and 'An error or interrupt occurred while reading the journal'.
"""
from __future__ import annotations

import re
from pathlib import Path

from cfd_auto import schedule

TAIL_BYTES = 3_000_000
STEP_RE = re.compile(r"time step = (\d+)")
# residual row: iteration, continuity, x/y/z-velocity, time per iteration (h:mm:ss), iterations left. The trailing time +
# count tell it apart from report-file prints ("  27  1.3500e-01  9.4542e-01  1.3500e-01" = step, flow time, ...), which
# would otherwise read as a residual that grows with flow time (2026-09-30).
_NUM = r"([0-9.]+e[-+]\d+|nan|inf)"
ROW_RE = re.compile(r"^\s*!?\s*(\d+)\s+" + r"\s+".join([_NUM] * 4) + r"\s+\d+:\d\d:\d\d\s+\d+\s*$", re.I)


def transcript(work: Path, job_id: str | None = None) -> Path | None:
    """Live transcript (in the work dir while running) or the archived one (Global_conditions/ after the run)."""
    work = Path(work)
    pats = [f"Fluent_cfdauto_{job_id}.out"] if job_id else [schedule.TRANSCRIPT_GLOB]
    for base in (work, work / "Global_conditions"):
        for pat in pats:
            hits = sorted(base.glob(pat), key=lambda p: p.stat().st_mtime)
            if hits:
                return hits[-1]
    return None


def _tail(path: Path, nbytes: int = TAIL_BYTES) -> str:
    with open(path, "rb") as fh:
        fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - nbytes))
        return fh.read().decode("latin-1", errors="replace")


def parse(text: str) -> dict:
    """Steps, final continuity residual per finished step and fatal lines of the LAST attempt in ``text``."""
    k = text.rfind(schedule.ATTEMPT_MARK)
    text = text[k:] if k >= 0 else text
    cont, cur, last = {}, None, 0
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m:
            last = int(m.group(1))
            if cur is not None:
                cont[last] = cur
            cur = None
            continue
        r = ROW_RE.match(line)
        if r:
            v = r.group(2).lower()
            cur = float("inf") if v in ("nan", "inf") else float(v)
    return {"last_step": last, "continuity": cont, "pending_continuity": cur, "finished": schedule.FINISH_MARK in text,
            "attempt_started": k >= 0, "text": text}


def classify(work: Path, n_native: int, div: dict, job_id: str | None = None) -> dict:
    """``div`` = protocol['divergence'] (fatal patterns, blow-up / high thresholds)."""
    t = transcript(work, job_id)
    if t is None:
        return {"status": "no_transcript"}
    p = parse(_tail(t))
    text = p.pop("text")
    fatal = next((ln.strip() for ln in text.splitlines() if any(s in ln for s in div["fatal_patterns"])), None)
    steps = sorted(p["continuity"])
    vals = [p["continuity"][s] for s in steps]     # finished steps only: scaled residuals start near 1 in a run's first iterations
    blowup = next((v for v in vals[-5:] if v >= div["continuity_blowup"]), None)
    k = div["continuity_high_steps"]
    high = len(vals) >= k and all(v >= div["continuity_high"] for v in vals[-k:])
    out = {"transcript": str(t), "last_step": p["last_step"], "n_native": n_native, "continuity_last": vals[-5:], "finished": p["finished"]}
    if fatal or blowup is not None:
        out.update(status="diverged", reason=fatal or f"continuity residual {blowup:.3g} >= {div['continuity_blowup']}")
    elif p["last_step"] >= n_native and p["finished"]:
        out["status"] = "completed"
    elif p["finished"]:
        out.update(status="ended_early", reason=f"script finished at step {p['last_step']} of {n_native}")
    elif high:
        out.update(status="diverging", reason=f"continuity residual >= {div['continuity_high']} for the last {k} steps")
    else:
        out["status"] = "running"
    return out
