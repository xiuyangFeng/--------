"""Resumable batch orchestrator: one state machine per unit, all state on disk, safe to stop and restart at any time.

    python -m cfd_auto.orchestrate run    <batch.json> [--interval 120]   # tick until every unit is terminal (nohup-able;
                                                                           # stop: kill $(cat <batch dir>/.orchestrate.run.pid))
    python -m cfd_auto.orchestrate tick   <batch.json> [--dry-run]        # one pass, then exit (cron / manual)
    python -m cfd_auto.orchestrate status <batch.json>
    python -m cfd_auto.orchestrate confirm <batch.json> <unit> [--keys '{"0": "inlet", ...}' | --accept]
    python -m cfd_auto.orchestrate retry  <batch.json> <unit> [--stage prepare|cfd|post]
    python -m cfd_auto.orchestrate release <batch.json> <unit>             # a unit held before CFD (spec hold_before_cfd)
    python -m cfd_auto.orchestrate cancel <batch.json> <unit>

Unit states:  pending -> preparing -> ready -> cfd -> post_pending -> post -> checked | flagged
              side exits: needs_confirmation (opening names), blocked (set-up checks / mesh gate / geometry: a person
              decides, then ``retry``), failed (errors after retries, numerics after the whole ladder)
Stages run as Slurm jobs: 'prepare' (driver: cfd_auto.prepare, which runs its own small Fluent jobs), 'cfd' (the managed
fluent.slurm), 'post' (relabel refined runs, library-layout check, cfd_auto.sanity with the protocol's two-level gates,
optional comparison with a library solution, MANIFEST.json). Batch hooks (e.g. staging + evaluation chain) run once
every unit is terminal.

Why this shape (2026-09-29 recovery batch lessons): state lives in ``<batch dir>/state/*.json`` and every decision is
re-derived from files (driver result files, the Fluent transcript), never from memory, so a restarted orchestrator does
not resubmit (a 30-minute monitor expiry once did); a file lock allows one tick at a time; a submission is recorded
before sbatch and recovered by job name if the tick dies in between; drivers refuse to run code whose fingerprint differs
from the one the tick saw (NFS served stale code to compute nodes); the CFD outcome is read from the transcript because a
diverged Fluent run still ends COMPLETED; a divergence moves the unit down the protocol's schedule ladder (library ->
gentle start -> half step + relabel); a node / licence / requeue failure retries the same schedule.

Batch file (JSON):
  {"batch": "name", "protocol": "aortoiliac_rcr4_v1", "protocol_overrides": {},
   "library": ".../data_new", "work_root": ".../units", "profiles": "<optional dir of refcase profiles>",
   "resources": {"cfd_cores": 92, "max_parallel_cfd": 6, "max_parallel_prepare": 6, "driver_cores": 4, "driver_mem": "48G",
                 "exclude": "node05", "cfd_time_limit": null},
   "units": {"<unit id>": {<spec: see cfd_auto.prepare>}}  or  "plan": "<recover plan.json whose units to take>",
   "hooks": [{"name": "eval", "script": "<sbatch script>", "units": "checked", "env": {}}]}
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cfd_auto import protocol as protocol_mod
from cfd_auto import runlog, schedule, slurm

ROOT = Path(__file__).resolve().parent.parent
PY = os.environ.get("CFD_AUTO_PYTHON", "/public/newhome/cy/.conda/envs/GNN/bin/python")
TERMINAL = {"checked", "flagged", "failed", "blocked", "needs_confirmation", "held"}
DEFAULT_RES = {"cfd_cores": 92, "max_parallel_cfd": 6, "max_parallel_prepare": 6, "driver_cores": 4, "driver_mem": "48G", "exclude": "node05",
               "cfd_time_limit": None, "driver_time_limit": "08:00:00", "prepare_retries": 2}


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def tag(unit: str) -> str:
    return unit.replace("/", "__")


def fingerprint(batch_file: Path) -> str:
    h = hashlib.sha1()
    for p in sorted((ROOT / "cfd_auto").glob("*.py")) + sorted((ROOT / "cfd_auto" / "protocols").glob("*.json")) + [Path(batch_file)]:
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:12]


# ----------------------------------------------------------------------------------------------------------- Slurm
class Slurm:
    """squeue / sbatch / scancel; every call is a subprocess so tests can swap in a fake."""

    def submit(self, script: Path, cwd: Path, name: str, env: dict | None = None) -> str:
        cmd = ["sbatch", "--parsable", "-J", name]
        if env:
            cmd.append("--export=ALL," + ",".join(f"{k}={v}" for k, v in env.items()))
        out = subprocess.run(cmd + [str(script)], cwd=cwd, capture_output=True, text=True, env=slurm._env(), check=True).stdout.strip()
        job = out.split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"sbatch returned {out!r}")
        return job

    def queued(self, job: str) -> str | None:
        """PENDING / RUNNING / ... while the job is in the queue, None once it left."""
        r = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"], capture_output=True, text=True, env=slurm._env())
        s = r.stdout.strip()
        return s or None

    def final_state(self, job: str) -> str:
        r = subprocess.run(["sacct", "-j", job, "-n", "-X", "-o", "State"], capture_output=True, text=True, env=slurm._env())
        s = r.stdout.split()
        return s[0] if s else "UNKNOWN"

    def find(self, name: str) -> str | None:
        r = subprocess.run(["squeue", "-h", "-n", name, "-o", "%i"], capture_output=True, text=True, env=slurm._env()).stdout.split()
        if r:
            return r[0]
        r = subprocess.run(["sacct", "-n", "-X", "--name", name, "-S", "now-7days", "-o", "JobID"], capture_output=True, text=True, env=slurm._env()).stdout.split()
        return r[-1] if r else None

    def cancel(self, job: str) -> None:
        subprocess.run(["scancel", job], capture_output=True, text=True, env=slurm._env())


# ----------------------------------------------------------------------------------------------------------- batch
class Batch:
    def __init__(self, batch_file: Path, backend: Slurm | None = None):
        self.file = Path(batch_file).resolve()
        self.dir = self.file.parent
        self.cfg = json.loads(self.file.read_text())
        self.name = re.sub(r"[^A-Za-z0-9]+", "", self.cfg.get("batch", self.file.stem))[:16]
        self.proto = protocol_mod.load(self.cfg.get("protocol"), self.cfg.get("protocol_overrides"))
        self.res = {**DEFAULT_RES, **self.cfg.get("resources", {})}
        self.library = Path(self.cfg.get("library", ROOT / "data_new"))
        self.work_root = Path(self.cfg["work_root"]).resolve()
        self.profiles = Path(self.cfg["profiles"]) if self.cfg.get("profiles") else self.dir / "profiles"
        units = dict(self.cfg.get("units", {}))
        if self.cfg.get("plan"):
            plan = json.loads(Path(self.cfg["plan"]).read_text())
            units = {**plan["units"], **units}
        self.units = units
        self.slurm = backend or Slurm()
        for d in ("state", "drivers"):
            (self.dir / d).mkdir(exist_ok=True)

    # state ------------------------------------------------------------------------------------------------------
    def state_file(self, unit: str) -> Path:
        return self.dir / "state" / f"{tag(unit)}.json"

    def load(self, unit: str) -> dict:
        f = self.state_file(unit)
        if f.exists():
            return json.loads(f.read_text())
        return {"unit": unit, "state": "pending", "active": None, "jobs": [], "ladder_index": 0, "infra_retries": 0, "prepare_retries": 0,
                "attempt": 0, "message": "", "updated": now()}

    def save(self, st: dict) -> None:
        st["updated"] = now()
        f = self.state_file(st["unit"])
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False, default=str))
        os.replace(tmp, f)

    def event(self, unit: str, what: str, **kw) -> None:
        row = {"t": now(), "unit": unit, "event": what, **kw}
        with open(self.dir / "events.jsonl", "a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"[{row['t']}] {unit}: {what} {json.dumps(kw, ensure_ascii=False, default=str) if kw else ''}", flush=True)

    def work(self, unit: str) -> Path:
        return (self.work_root / unit).resolve()

    def job_name(self, unit: str, kind: str, n: int) -> str:
        h = hashlib.sha1(unit.encode()).hexdigest()[:6]
        return f"cfa-{self.name}-{h}-{kind}{n}"

    # submissions ------------------------------------------------------------------------------------------------
    def _driver_script(self, unit: str, kind: str, name: str) -> Path:
        r = self.res
        s = self.dir / "drivers" / f"{tag(unit)}_{kind}.slurm"
        s.write_text("\n".join([
            "#!/bin/bash", "#SBATCH --partition=CPU", f"#SBATCH --exclude={r['exclude']}" if r["exclude"] else "", f"#SBATCH --ntasks-per-node={r['driver_cores']}",
            f"#SBATCH --mem={r['driver_mem']}", f"#SBATCH --time={r['driver_time_limit']}", f"#SBATCH --output={self.dir / 'drivers'}/{tag(unit)}_{kind}_%j.out",
            "set -uo pipefail", "export PATH=/public/slurm/bin:$PATH", f"cd {ROOT}",
            f"export PYTHONPATH={ROOT} OMP_NUM_THREADS={r['driver_cores']} OPENBLAS_NUM_THREADS={r['driver_cores']} MKL_NUM_THREADS={r['driver_cores']}",
            "date -Is; hostname",
            f'{PY} -u -m cfd_auto.orchestrate _{kind} "{self.file}" "{unit}" --expect-fingerprint "$CFD_AUTO_FP"',
            "date -Is", ""]))
        return s

    def submit(self, st: dict, kind: str) -> None:
        n = sum(1 for j in st["jobs"] if j["kind"] == kind) + 1
        name = self.job_name(st["unit"], kind, n)
        st["active"] = {"kind": kind, "name": name, "job": None, "submitted": now(), "n": n}
        if kind == "cfd":
            fr = json.loads((self.work(st["unit"]) / schedule.FRAMES_FILE).read_text())
            st["active"]["schedule"] = fr["schedule"]["name"]; st["active"]["n_native"] = fr["schedule"]["n_native"]
        self.save(st)                                  # recorded before sbatch: a dying tick is recovered by job name
        if kind == "cfd":
            job = self.slurm.submit(self.work(st["unit"]) / "fluent.slurm", self.work(st["unit"]), name)
        else:
            job = self.slurm.submit(self._driver_script(st["unit"], kind, name), ROOT, name, env={"CFD_AUTO_FP": fingerprint(self.file)})
        st["active"]["job"] = job
        st["jobs"].append({k: st["active"].get(k) for k in ("kind", "name", "job", "submitted", "schedule")})
        st["state"] = {"prepare": "preparing", "cfd": "cfd", "post": "post"}[kind]
        self.save(st)
        self.event(st["unit"], f"submitted {kind}", job=job, name=name, schedule=st["active"].get("schedule"))

    # tick -------------------------------------------------------------------------------------------------------
    def counts(self, states: dict[str, dict]) -> dict[str, int]:
        c = {"prepare": 0, "cfd": 0, "post": 0}
        for st in states.values():
            if st.get("active"):
                c[st["active"]["kind"]] += 1
        return c

    def tick(self, dry_run: bool = False) -> dict[str, dict]:
        states = {u: self.load(u) for u in self.units}
        for u, st in states.items():
            try:
                self._advance_active(st, dry_run)
            except Exception as exc:     # one unit's problem must not stop the batch
                st["message"] = f"tick error: {type(exc).__name__}: {exc}"
                if not st.get("active") and st["state"] in ("preparing", "cfd", "post"):
                    st["state"] = "failed"          # an outcome handler died half-way: never leave a unit without a next step
                self.event(u, "tick error", error=st["message"], tb=traceback.format_exc(limit=3))
                if not dry_run:
                    self.save(st)
        cnt = self.counts(states)
        for u, st in states.items():
            if st.get("active"):
                continue
            kind = {"pending": "prepare", "ready": "cfd", "post_pending": "post"}.get(st["state"])
            if kind is None:
                continue
            if kind == "cfd" and self.units[u].get("hold_before_cfd") and not st.get("released"):
                st.update(state="held", message="prepared; held before CFD (spec hold_before_cfd) — `orchestrate release` to run")
                if not dry_run:
                    self.save(st); self.event(u, "held before CFD")
                continue
            limit = {"prepare": self.res["max_parallel_prepare"], "cfd": self.res["max_parallel_cfd"], "post": 1000}[kind]
            if cnt[kind] >= limit:
                continue
            if dry_run:
                print(f"would submit {kind} for {u}")
            else:
                self.submit(st, kind)
            cnt[kind] += 1
        if not dry_run:
            self._hooks(states)
        return states

    def _advance_active(self, st: dict, dry_run: bool) -> None:
        a = st.get("active")
        if not a:
            return
        if a.get("job") is None:                       # sbatch may or may not have happened before the last tick died
            found = self.slurm.find(a["name"])
            if found:
                a["job"] = found; st["jobs"].append({k: a.get(k) for k in ("kind", "name", "job", "submitted", "schedule")})
                self.event(st["unit"], "recovered job id", job=found)
            else:
                st["active"] = None
                st["state"] = {"prepare": "pending", "cfd": "ready", "post": "post_pending"}[a["kind"]]
                self.event(st["unit"], "submission not found; will resubmit", name=a["name"])
                if not dry_run:
                    self.save(st)
                return
        q = self.slurm.queued(a["job"])
        work = self.work(st["unit"])
        if q is not None:
            if a["kind"] == "cfd" and q == "RUNNING":
                h = runlog.classify(work, a["n_native"], self.proto["divergence"], a["job"])
                st["progress"] = {k: h.get(k) for k in ("status", "last_step", "n_native", "continuity_last")}
                if h["status"] in ("diverged", "diverging") and not a.get("cancelled"):
                    a["cancelled"] = h.get("reason")
                    self.event(st["unit"], "cancelling diverging run", job=a["job"], reason=h.get("reason"), step=h.get("last_step"))
                    if not dry_run:
                        self.slurm.cancel(a["job"])
            st["queue_state"] = q
            if not dry_run:
                self.save(st)
            return
        if dry_run:
            print(f"{st['unit']}: {a['kind']} job {a['job']} left the queue")
            return
        # NFS grace: files written on the compute node (result json, transcript moved into Global_conditions/) can be
        # invisible to this node for a while; deciding early could read a finished run as a crash and resubmit it, and
        # the resubmitted script would clear the good outputs
        if "left_queue" not in a:
            a["left_queue"] = time.time(); st["queue_state"] = "left"
            self.save(st)
            return
        if time.time() - a["left_queue"] < self.res.get("nfs_grace_s", 180):
            return
        getattr(self, f"_end_{a['kind']}")(st, work)
        self.save(st)

    # outcomes ---------------------------------------------------------------------------------------------------
    def _result(self, work: Path, kind: str, job: str) -> dict | None:
        f = work / "orchestrator" / f"{kind}_result.json"
        if not f.exists():
            return None
        r = json.loads(f.read_text())
        return r if str(r.get("job")) == str(job) else None

    def _end_prepare(self, st: dict, work: Path) -> None:
        a = st.pop("active"); st["active"] = None
        r = self._result(work, "prepare", a["job"])
        if r is None or r["status"] == "retry":
            st["prepare_retries"] += 1
            if st["prepare_retries"] > self.res["prepare_retries"]:
                st.update(state="failed", message=f"prepare job {a['job']} ended without a result {st['prepare_retries']} times" + (f": {r['message']}" if r else ""))
            else:
                st.update(state="pending", message=f"prepare job {a['job']} ended without a result; retrying" + (f" ({r['message']})" if r else ""))
        else:
            st.update(state={"ready": "ready", "needs_confirmation": "needs_confirmation", "blocked": "blocked"}.get(r["status"], "failed"), message=r.get("message", ""))
        self.event(st["unit"], f"prepare ended -> {st['state']}", job=a["job"], message=st["message"][:300])

    def _archive_attempt(self, work: Path, a: dict) -> Path:
        dst = work / "attempts" / f"{a['n']:02d}_{a.get('schedule')}_{a['job']}"
        dst.mkdir(parents=True, exist_ok=True)
        for p in (work / "2.jou", work / schedule.FRAMES_FILE):     # copies: the next attempt reads frames.json
            if p.exists():
                shutil.copy2(p, dst / p.name)
        for p in [work / "Global_conditions", *work.glob(f"Fluent_cfdauto_{a['job']}.*"), work / "run_attempts.log"]:
            if p.exists():
                shutil.move(str(p), str(dst / p.name))
        for d in ("ascii", "ascii_in"):
            shutil.rmtree(work / d, ignore_errors=True)
        for f in work.glob("*-[0-9][0-9][0-9][0-9]"):
            f.unlink()
        (work / "ascii").mkdir(exist_ok=True)
        return dst

    def _end_cfd(self, st: dict, work: Path) -> None:
        from cfd_auto import prepare
        a = st.pop("active"); st["active"] = None
        h = runlog.classify(work, a["n_native"], self.proto["divergence"], a["job"])
        ladder = self.proto["schedules"]["ladder"]
        st["last_run"] = {**{k: h.get(k) for k in ("status", "reason", "last_step", "n_native", "continuity_last", "transcript")}, "job": a["job"], "schedule": a.get("schedule")}
        if h["status"] == "completed":
            st.update(state="post_pending", message=f"CFD {a['job']} completed ({a.get('schedule')})")
        elif h["status"] in ("diverged", "diverging") or a.get("cancelled"):
            arch = self._archive_attempt(work, a)
            st["ladder_index"] += 1
            if st["ladder_index"] >= len(ladder):
                st.update(state="failed", message=f"diverged on every schedule of the ladder (last: {a.get('schedule')}: {h.get('reason') or a.get('cancelled')}); attempts in {arch}")
            else:
                nxt = ladder[st["ladder_index"]]
                prepare.rewrite_schedule(work, self.proto, nxt, self.res["cfd_cores"], self.job_name(st["unit"], "cfd", 0))
                st.update(state="ready", message=f"{a.get('schedule')} diverged at step {h.get('last_step')} ({h.get('reason') or a.get('cancelled')}) -> {nxt}")
        else:
            arch = self._archive_attempt(work, a)
            st["infra_retries"] += 1
            if st["infra_retries"] > self.proto["schedules"]["infra_retries"]:
                st.update(state="failed", message=f"CFD ended without finishing {st['infra_retries']} times (last: {h['status']} {h.get('reason', '')}); attempts in {arch}")
            else:
                st.update(state="ready", message=f"CFD {a['job']} {h['status']} at step {h.get('last_step')} ({self.slurm.final_state(a['job'])}); retrying {a.get('schedule')}")
        self.event(st["unit"], f"cfd ended -> {st['state']}", job=a["job"], run=h.get("status"), message=st["message"][:300])

    def _end_post(self, st: dict, work: Path) -> None:
        a = st.pop("active"); st["active"] = None
        r = self._result(work, "post", a["job"])
        if r is None or r["status"] == "retry":
            st["infra_retries"] += 1
            st.update(state="post_pending" if st["infra_retries"] <= self.proto["schedules"]["infra_retries"] else "failed",
                      message=f"post job {a['job']} ended without a result" + (f": {r['message']}" if r else ""))
        else:
            st.update(state=r["status"] if r["status"] in ("checked", "flagged") else "failed", message=r.get("message", ""), post=r)
        self.event(st["unit"], f"post ended -> {st['state']}", job=a["job"], message=st["message"][:300])

    # hooks ------------------------------------------------------------------------------------------------------
    def _hooks(self, states: dict[str, dict]) -> None:
        hooks = self.cfg.get("hooks") or []
        if not hooks or not all(st["state"] in TERMINAL for st in states.values()):
            return
        bf = self.dir / "state" / "_batch.json"
        bs = json.loads(bf.read_text()) if bf.exists() else {"hooks": {}}
        for h in hooks:
            hs = bs["hooks"].get(h["name"])
            if hs and hs.get("job"):
                if hs.get("state") in (None, "submitted") and self.slurm.queued(hs["job"]) is None:
                    hs["state"] = self.slurm.final_state(hs["job"]); self.event("_batch", f"hook {h['name']} ended", state=hs["state"])
                continue
            want = set(h.get("units", "checked").split("|"))
            ul = self.dir / f"units_{h['name']}.txt"
            ul.write_text("".join(f"{u}\n" for u, st in states.items() if st["state"] in want))
            name = f"cfa-{self.name}-hook-{h['name']}"
            job = self.slurm.submit(Path(h["script"]), ROOT, name, env={"CFD_AUTO_UNITS_FILE": ul, "CFD_AUTO_BATCH": self.file, **h.get("env", {})})
            bs["hooks"][h["name"]] = {"job": job, "state": "submitted", "units_file": str(ul), "submitted": now()}
            self.event("_batch", f"hook {h['name']} submitted", job=job)
        bf.write_text(json.dumps(bs, indent=1))

    # manual actions ---------------------------------------------------------------------------------------------
    def confirm(self, unit: str, keys: dict | None, accept: bool) -> None:
        st = self.load(unit)
        work = self.work(unit)
        if accept:
            nm = json.loads((work / "naming.json").read_text())
            keys = nm["methods"][nm["primary"]]["keys"]
        if not keys or sorted(keys.values()) != sorted(["inlet", *self.proto["openings"]["outlet_keys"]]):
            raise SystemExit(f"confirm: keys must name every opening once: {keys}")
        (work / "naming_confirmed.json").write_text(json.dumps({"keys": keys, "confirmed_at": now(), "by": os.environ.get("USER"), "accepted_automatic": accept}, indent=1))
        st.update(state="pending", active=None, message="naming confirmed", prepare_retries=0)
        self.save(st); self.event(unit, "naming confirmed", keys=keys)

    def retry(self, unit: str, stage: str) -> None:
        st = self.load(unit)
        if st.get("active") and self.slurm.queued(st["active"]["job"] or "0"):
            raise SystemExit(f"{unit}: job {st['active']['job']} still queued; cancel it first")
        st.update(active=None, state={"prepare": "pending", "cfd": "ready", "post": "post_pending"}[stage], message=f"manual retry from {stage}",
                  prepare_retries=0, infra_retries=0)
        if stage == "prepare":
            st["ladder_index"] = 0
        self.save(st); self.event(unit, f"retry from {stage}")

    def release(self, unit: str) -> None:
        st = self.load(unit)
        if st["state"] != "held":
            raise SystemExit(f"{unit} is {st['state']}, not held")
        st.update(state="ready", released=True, message="released for CFD")
        self.save(st); self.event(unit, "released")

    def cancel(self, unit: str) -> None:
        st = self.load(unit)
        if st.get("active") and st["active"].get("job"):
            self.slurm.cancel(st["active"]["job"])
        st.update(active=None, state="failed", message="cancelled by user")
        self.save(st); self.event(unit, "cancelled")

    def status(self) -> str:
        rows = []
        for u in self.units:
            st = self.load(u)
            a = st.get("active") or {}
            pr = st.get("progress") or {}
            prog = f"{pr.get('last_step')}/{pr.get('n_native')}" if a.get("kind") == "cfd" and pr else ""
            rows.append((u, st["state"], a.get("schedule") or (st.get("last_run") or {}).get("schedule") or "", a.get("job") or "", prog, st.get("message", "")[:90]))
        w = [max(len(str(r[i])) for r in rows + [("unit", "state", "schedule", "job", "step", "")]) for i in range(5)]
        out = ["  ".join(str(x).ljust(w[i]) for i, x in enumerate(("unit", "state", "schedule", "job", "step"))) + "  message"]
        out += ["  ".join(str(x).ljust(w[i]) for i, x in enumerate(r[:5])) + "  " + r[5] for r in rows]
        return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------- drivers
def _check_fingerprint(batch_file: Path, expect: str | None, wait_s: int = 300) -> str:
    fp = fingerprint(batch_file)
    t0 = time.time()
    while expect and fp != expect and time.time() - t0 < wait_s:
        time.sleep(20); fp = fingerprint(batch_file)
    print(f"cfd_auto fingerprint {fp} (expected {expect})", flush=True)
    if expect and fp != expect:
        raise RuntimeError(f"code fingerprint {fp} != {expect}: this node sees other code (NFS lag) or the code changed after submission")
    return fp


def _write_result(work: Path, kind: str, res: dict) -> None:
    (work / "orchestrator").mkdir(parents=True, exist_ok=True)
    res = {**res, "job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename, "t": now()}
    (work / "orchestrator" / f"{kind}_result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))


def driver_prepare(batch_file: Path, unit: str, expect: str | None) -> None:
    from cfd_auto import prepare
    b = Batch(batch_file)
    work = b.work(unit)
    try:
        fp = _check_fingerprint(b.file, expect)
    except RuntimeError as exc:
        _write_result(work, "prepare", {"status": "retry", "message": str(exc)}); raise SystemExit(1)
    try:
        rep = prepare.prepare_unit(unit, b.units[unit], b.proto, b.work_root, b.library, b.profiles, b.res["cfd_cores"], b.job_name(unit, "cfd", 0), fp)
        _write_result(work, "prepare", {"status": "ready", "message": f"ready; preflight warnings {rep['preflight']['warn']}", "preflight": rep["preflight"]})
    except prepare.Blocked as exc:
        _write_result(work, "prepare", {"status": "needs_confirmation" if exc.kind == "naming" else "blocked", "kind": exc.kind, "message": str(exc), "detail": exc.detail})
    except Exception as exc:
        _write_result(work, "prepare", {"status": "error", "message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        raise


def driver_post(batch_file: Path, unit: str, expect: str | None) -> None:
    from cfd_auto import compare, relabel, sanity
    b = Batch(batch_file)
    work = b.work(unit)
    try:
        fp = _check_fingerprint(b.file, expect)
    except RuntimeError as exc:
        _write_result(work, "post", {"status": "retry", "message": str(exc)}); raise SystemExit(1)
    try:
        rel = relabel.relabel(work)
        labels = protocol_mod.label_steps(b.proto)
        e = b.proto["exports"]
        layout = {}
        for sub in (e["wall_dir"], e["volume_dir"]):
            got = sorted(int(m.group(1)) for f in (work / sub).iterdir() if (m := re.search(r"-(\d{4})$", f.name)))
            layout[sub] = {"frames": len(got), "ok": got == labels}
        if not all(v["ok"] for v in layout.values()):
            raise RuntimeError(f"export layout: {layout} (expected labels {labels[0]}..{labels[-1]})")
        spec = b.units[unit]
        refs = [b.library / spec["template"]] if spec.get("template") else []
        coh = spec.get("cohort", unit.split("/")[0])
        gates = (b.proto.get("sanity_gates_by_cohort") or {}).get(coh) or b.proto["sanity_gates"]
        rep = sanity.check(work, refs, gates=gates)
        (work / "sanity.json").write_text(json.dumps(rep, indent=1, default=float))
        out = {"status": "checked" if rep["ok"] else "flagged", "flags": rep["flags"], "warnings": rep.get("warnings", []), "relabel": rel, "layout": layout,
               "message": ("sanity ok" if rep["ok"] else f"sanity flags: {rep['flags']}") + (f"; warnings {rep.get('warnings')}" if rep.get("warnings") else "")}
        if spec.get("validate_against_library"):
            res = compare.compare(b.library / unit, work, "v2", None, None, spec.get("compare_volume", True), "wall", spec.get("compare_ref_case", True))
            (work / "comparison.json").write_text(json.dumps(res, indent=1))
            out["comparison_all_pass"] = res["all_pass"]
        st = b.load(unit)
        manifest = {"unit": unit, "protocol": b.proto["name"], "code_fingerprint": fp, "spec": spec, "cfd_jobs": [j for j in st["jobs"] if j["kind"] == "cfd"],
                    "schedule": json.loads((work / schedule.FRAMES_FILE).read_text())["schedule"]["name"], "relabel": rel, "layout": layout,
                    "sanity": {"ok": rep["ok"], "flags": rep["flags"], "warnings": rep.get("warnings", [])}, "created": now()}
        for f in ("prepare_report.json",):
            if (work / f).exists():
                pr = json.loads((work / f).read_text())
                manifest.update({k: pr.get(k) for k in ("stl_sha256", "preflight", "final_quality", "case")})
        (work / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str))
        _write_result(work, "post", out)
    except Exception as exc:
        _write_result(work, "post", {"status": "error", "message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        raise


# ----------------------------------------------------------------------------------------------------------- CLI
def _locked(batch: Batch, name: str = ".orchestrate.lock", wait_s: float = 0):
    """Exclusive file lock: the tick lock (one tick or manual action at a time; ``run`` takes it per tick, manual commands
    wait for it) or the runner lock (one ``run`` loop per batch)."""
    fh = open(batch.dir / name, "w")
    t0 = time.time()
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.time() - t0 >= wait_s:
                fh.close()
                raise SystemExit(f"another orchestrator process holds {batch.dir / name}")
            time.sleep(1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="cfd_auto.orchestrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "tick", "status", "confirm", "retry", "release", "cancel", "_prepare", "_post"):
        p = sub.add_parser(c); p.add_argument("batch", type=Path)
        if c in ("confirm", "retry", "release", "cancel", "_prepare", "_post"):
            p.add_argument("unit")
        if c == "run":
            p.add_argument("--interval", type=int, default=120)
        if c == "tick":
            p.add_argument("--dry-run", action="store_true")
        if c == "confirm":
            p.add_argument("--keys", type=json.loads, default=None); p.add_argument("--accept", action="store_true")
        if c == "retry":
            p.add_argument("--stage", choices=["prepare", "cfd", "post"], default="prepare")
        if c in ("_prepare", "_post"):
            p.add_argument("--expect-fingerprint", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "_prepare":
        return driver_prepare(a.batch, a.unit, a.expect_fingerprint)
    if a.cmd == "_post":
        return driver_post(a.batch, a.unit, a.expect_fingerprint)
    b = Batch(a.batch)
    if a.cmd == "status":
        print(b.status()); return
    if a.cmd == "run":
        runner = _locked(b, ".orchestrate.run.lock")
        (b.dir / ".orchestrate.run.pid").write_text(f"{os.getpid()}\n")    # stop with: kill $(cat <batch dir>/.orchestrate.run.pid)
        try:
            while True:
                lock = _locked(b, wait_s=600)
                try:
                    b = Batch(a.batch)                 # re-read: the batch file may have gained units / changed resources
                    states = b.tick()
                finally:
                    lock.close()
                if all(st["state"] in TERMINAL for st in states.values()) and not any(st.get("active") for st in states.values()):
                    bf = b.dir / "state" / "_batch.json"
                    hooks = json.loads(bf.read_text())["hooks"] if bf.exists() else {}
                    if not (b.cfg.get("hooks") and (len(hooks) < len(b.cfg["hooks"]) or any(h.get("state") == "submitted" for h in hooks.values()))):
                        print(b.status()); break
                time.sleep(a.interval)
        finally:
            runner.close()
        return
    lock = _locked(b, wait_s=600)
    try:
        if a.cmd == "tick":
            b.tick(a.dry_run); print(b.status())
        elif a.cmd == "confirm":
            b.confirm(a.unit, a.keys, a.accept)
        elif a.cmd == "retry":
            b.retry(a.unit, a.stage)
        elif a.cmd == "release":
            b.release(a.unit)
        elif a.cmd == "cancel":
            b.cancel(a.unit)
    finally:
        lock.close()


if __name__ == "__main__":
    main(sys.argv[1:])
