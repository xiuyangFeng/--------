"""Persistent, owner-scoped jobs and the single inference worker.

The saved job is the source of truth. Queue entries include a version, so a
cancelled/retried entry cannot later start another copy of the same work.
"""
from __future__ import annotations

import base64
import binascii
import copy
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import threading
import time

from . import clock as _clock
from .errors import ResourceError, classify

LOG = logging.getLogger("wss_deploy.service")
# v0.14 (J8): one structured line per job state change (routed to server.log by ``cli serve``).
AUDIT = logging.getLogger("wss_deploy.audit")
JOB_ID_PATTERN = r"[A-Za-z0-9_-]{1,80}"
LABELS = {"out-le", "out-li", "out-re", "out-ri"}
FINAL = {"done", "failed", "cancelled", "interrupted"}
JOB_STATUSES = FINAL | {"queued", "running", "awaiting_input", "awaiting_confirmation",
                        "queued_A", "queued_B", "running_A", "running_B", "awaiting_outlets"}
CASE_METADATA = ("patient_id", "scan_label", "scan_date", "tags", "notes")
MAX_BATCH_ITEMS = 20
MAX_DELETE_ITEMS = 100
DELETING_PREFIX = ".deleting_"
DELETED_LOG = "deleted_jobs.jsonl"
TRASH_DIR = ".trash"
TRASH_DAYS = 30
TRASH_SCAN_SECONDS = 24 * 3600
REVIEW_STATUSES = ("unreviewed", "reviewed", "reopened")
FINDING_DECISIONS = ("confirmed", "rejected")
MAX_ANNOTATIONS = 50
MAX_ADDED_FINDINGS = 30
MAX_SNAPSHOTS = 12
MAX_NARRATIVE_CHARS = 4000
MAX_SNAPSHOT_BYTES = 6 * 1024 * 1024
SNAPSHOT_NAME = re.compile(r"[a-z0-9_-]{1,32}")
SNAPSHOT_FILE = re.compile(r"snapshot_[a-z0-9_-]{1,32}\.png")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FAMILY_OF_PROTOCOL = {"single_frame_wss": "wall", "single_frame_wss_cycle_multi": "wall", "single_frame_volume": "volume"}
# §19.2: the job record carries the cycle block verbatim (≈ 7 kB for M1) and the first findings.
CYCLE_RECORD_LIMIT = 16 * 1024
FINDINGS_TOP = 5
QUEUE_STATUSES = ("queued", "running", "awaiting_input", "awaiting_confirmation")
ETA_STATUSES = frozenset(QUEUE_STATUSES)
# v0.14 (J1/J9): job.json v2 no longer embeds the stage-A display geometry (``a.preview`` and
# ``a.proposal.preview_polylines``, ≈ 1.1 MB); ``stage_a.json`` in the job directory keeps it and
# ``geometry()`` / ``stage_a()`` read it lazily.  Records without ``schema_version`` are v1 (or older shapes).
JOB_SCHEMA_VERSION = "wss-deploy.job/v2"
STAGE_A_FILE = "stage_a.json"
# v0.14 (J2): mesh-only stage-B intermediates written by ``pipeline.precompute_geometry_cache`` (copied by rerun / reuse).
GEOMETRY_CACHE_DIR = "geometry_cache"
PRECOMPUTE_WAIT_S = 300.0          # stage B waits this long for a running precompute of its job, then cancels it
# v0.14 (J6): ``service stop|restart|upgrade --drain`` writes DRAIN_FILE naming the service pid; that process then
# starts no new work.  ``service upgrade`` queues post-start maintenance (analysis rebuilds + stale report refresh)
# in MAINTENANCE_FILE; the new service runs it in the background and writes MAINTENANCE_RESULT.
DRAIN_FILE = ".drain.json"
MAINTENANCE_FILE = ".maintenance.json"
MAINTENANCE_RESULT = "maintenance_result.json"
GENERIC_FAILURE = "计算未完成，请重试；如仍失败，请向维护者提供诊断编号。"
WORKER_COUNT = 2
# Server temp files (spooled uploads, bundles) and streamed upload staging live in ``<jobs_root>/.tmp``; like every
# dot-entry of the jobs root it is never a job.  ``<job>/.report_ui_stage_*`` are report_freshness staging dirs.
TMP_DIR = ".tmp"
UPLOAD_CHUNK_BYTES = 1024 * 1024


class JobError(ValueError):
    def __init__(self, message: str, status: int = 400, payload: dict | None = None):
        super().__init__(message)
        self.status = status
        # Extra top-level keys merged into the JSON error body (duplicate-upload choices, …).
        self.payload = payload or {}


def input_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def family_of_release(release: dict | None) -> str | None:
    """Result family implied by the bound release contract (known before stage B runs)."""
    contract = (release or {}).get("contract") if isinstance(release, dict) else None
    protocol = contract.get("protocol") if isinstance(contract, dict) else None
    return FAMILY_OF_PROTOCOL.get(protocol)


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def run_identity(*, input_sha256: str | None, release: dict | None,
                 mapping: dict | None, parameters: dict | None,
                 schema_version: str = "wss-deploy.run/v1") -> str | None:
    """Stable comparison key for runs; excludes owner and job timestamps."""
    if not input_sha256 or not release:
        return None
    payload = {"schema_version": schema_version, "input_sha256": str(input_sha256),
               "release": {"id": release.get("id") or release.get("release"),
                           "fingerprint": release.get("fingerprint")},
               "mapping": mapping or {}, "parameters": parameters or {}}
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict, *, compact: bool = False, durable: bool = True) -> None:
    """Replace a complete JSON document; never expose a partially written job.

    ``compact`` (job.json, v0.14) serialises with the C encoder and no indentation; ``durable=False`` keeps the
    atomic rename but skips both fsyncs (progress-only updates: a crash can at worst lose the latest phase text).
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        # v0.15: created 0600 (O_EXCL) whatever the umask — no window in which another account could open it.
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            if compact:
                stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            else:
                json.dump(value, stream, ensure_ascii=False, indent=1, allow_nan=False)
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(tmp, path)
        if durable:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        tmp.unlink(missing_ok=True)


def slim_stage_a(stage_a):
    """The stage-A snapshot as kept in job.json v2: everything except the display geometry.

    ``preview`` (display mesh) and ``proposal.preview_polylines`` stay in ``stage_a.json`` only.  Returns a new
    top-level dict (and a new ``proposal`` dict when one is trimmed); nested values are shared, not copied.
    """
    if not isinstance(stage_a, dict):
        return stage_a
    light = {key: value for key, value in stage_a.items() if key != "preview"}
    proposal = stage_a.get("proposal")
    if isinstance(proposal, dict) and "preview_polylines" in proposal:
        light["proposal"] = {key: value for key, value in proposal.items() if key != "preview_polylines"}
    return light


def _schema_rank(version) -> int:
    """``wss-deploy.job/vN`` → N; records written before the field existed are rank 0 (v1 or older shapes)."""
    match = re.search(r"/v(\d+)$", str(version or ""))
    return int(match.group(1)) if match else 0


def _migrate_v1(job: dict, context: dict) -> None:
    """Shapes written before job.json carried a schema version (pre-registry / pre-review / pre-metadata records)."""
    job.setdefault("version", 1)
    job.setdefault("params", {"units": "mm", "remove_fragments": False, "inlet": None})
    job.setdefault("compute", {"device": "auto", "seed_count": None, "threads": None})
    try:
        job.setdefault("created_ts", context["path"].stat().st_mtime)
    except OSError:
        job.setdefault("created_ts", time.time())
    job.setdefault("owner", context.get("legacy_owner"))
    for key in CASE_METADATA:
        job.setdefault(key, [] if key == "tags" else "")
    if "stage_a" in job and "a" not in job:
        job["a"] = job.pop("stage_a")


def _migrate_v2(job: dict, context: dict) -> None:
    """Drop the embedded display geometry when ``stage_a.json`` holds the identical value (J1).

    A value that is missing from, or different in, ``stage_a.json`` stays embedded: nothing is ever lost, and
    ``geometry()`` prefers an embedded value.  The file is read only when the record carries such a value.
    """
    a = job.get("a")
    if not isinstance(a, dict):
        return
    proposal = a.get("proposal") if isinstance(a.get("proposal"), dict) else {}
    if "preview" not in a and "preview_polylines" not in proposal:
        return
    stage_file = context["path"].parent / STAGE_A_FILE
    try:
        stored = json.loads(stage_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        stored = None
    stored = stored if isinstance(stored, dict) else {}
    stored_proposal = stored.get("proposal") if isinstance(stored.get("proposal"), dict) else {}
    light = dict(a)
    if "preview" in a and "preview" in stored and stored["preview"] == a["preview"]:
        light.pop("preview")
    if "preview_polylines" in proposal and "preview_polylines" in stored_proposal \
            and stored_proposal["preview_polylines"] == proposal["preview_polylines"]:
        light["proposal"] = {key: value for key, value in proposal.items() if key != "preview_polylines"}
    job["a"] = light


# J9: ordered schema migrations applied on load; each runs when the record is older than its version.
MIGRATIONS = [("wss-deploy.job/v1", _migrate_v1), ("wss-deploy.job/v2", _migrate_v2)]


def migrate_record(job: dict, *, path: Path, legacy_owner: str | None = None) -> list[str]:
    """Bring one loaded record to ``JOB_SCHEMA_VERSION`` in memory; returns the versions applied.

    The file is rewritten by the next save (lazy migration), so a read-only process changes nothing on disk.
    """
    context = {"path": Path(path), "legacy_owner": legacy_owner}
    current = _schema_rank(job.get("schema_version"))
    applied = []
    for version, migrate in MIGRATIONS:
        if _schema_rank(version) > current:
            migrate(job, context)
            applied.append(version)
    if current < _schema_rank(JOB_SCHEMA_VERSION):
        job["schema_version"] = JOB_SCHEMA_VERSION
    return applied


def _owner_label(owner) -> str | None:
    """User names as-is; random session owners (32-character url-safe tokens) only as a short hash in logs."""
    if not isinstance(owner, str) or not owner:
        return None
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,20}", owner):
        return owner
    return "h:" + hashlib.sha256(owner.encode("utf-8")).hexdigest()[:12]


def _date(timestamp: float) -> str:
    # O1: the deployment zone (clock: WSS_DEPLOY_TZ → service.json env.TZ → system), not the process TZ.
    return _clock.local(timestamp).isoformat(timespec="seconds")


def findings_top(summary: dict | None, limit: int = FINDINGS_TOP) -> list[dict]:
    """The headline findings of a result for the job record (§19.2), without the point index lists.

    Same choice as the one-pager's first page (one per kind first, rejected items dropped), so the
    workbench card, the one-pager and the report agree on what "重点发现" means."""
    findings = summary.get("findings") if isinstance(summary, dict) else None
    items = findings.get("items") if isinstance(findings, dict) else None
    try:
        from .onepager import top_findings
        chosen = top_findings(summary, limit) if items else []
    except Exception:  # never let a presentation helper break loading the job list
        chosen = (items or [])[:limit]
    return [{key: value for key, value in item.items() if key != "point_indices"} for item in chosen if isinstance(item, dict)]


def cycle_record(cycle) -> dict | None:
    """``summary.cycle`` for the job record: verbatim when small, otherwise without the per-branch tables."""
    if not isinstance(cycle, dict) or not cycle:
        return None
    if len(json.dumps(cycle, ensure_ascii=False, allow_nan=False)) <= CYCLE_RECORD_LIMIT:
        return copy.deepcopy(cycle)
    compact = copy.deepcopy(cycle)
    for block in list((compact.get("fields") or {}).values()) + [compact.get("stagnation")]:
        if isinstance(block, dict):
            block.pop("per_branch", None)
    compact["compact"] = True
    return compact


def release_short(release: dict | None) -> str | None:
    """Compact release name for lists: ``X5D_v51_5seed_20260916`` → ``X5D_v51``."""
    rid = (release.get("id") or release.get("release")) if isinstance(release, dict) else None
    if not rid:
        return None
    short = re.sub(r"_20\d{6}$", "", str(rid))
    short = re.sub(r"_\d+seeds?(?=_|$)", "", short)
    return short or str(rid)


def fresh_review() -> dict:
    """Review record of a result nobody has signed off yet (contract §8)."""
    return {"status": "unreviewed", "by": "", "at": "", "note": "", "version": None}


def _validate_mapping(job_dir: Path, mapping: dict) -> list[str]:
    from . import centerline as cl
    return cl.validate_mapping(cl.load_vessel_geom_atlas(job_dir / "centerline"), mapping)


class JobManager:
    def __init__(self, root: Path, *, release=None, registry=None, stage_a_fn=None, stage_b_fn=None,
                 mapping_validator=None, legacy_owner: str | None = None, clock=None, offline: bool = False,
                 precompute_fn=None):
        """``offline`` (v0.14, J3): a maintenance process (``cli jobs claim``) that must not act as the service —
        nothing is marked interrupted or re-queued, no trash / unfinished deletion is purged, and ``start()`` is
        refused.  ``precompute_fn`` overrides ``pipeline.precompute_geometry_cache`` (J2); managers built with
        injected stage functions (tests) precompute only when one is given."""
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.offline = bool(offline)
        self.release = release
        self.registry = registry
        self.stage_a_fn, self.stage_b_fn = stage_a_fn, stage_b_fn
        self.mapping_validator = mapping_validator or _validate_mapping
        # Injectable clock so the 30-day trash expiry can be tested without waiting.
        self.clock = clock or time.time
        self.jobs: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.tasks: queue.Queue = queue.Queue()
        # Stage B touches the legacy geometry capfit override and the model
        # runtime.  Keep it serialized while allowing two CPU-heavy Stage-A
        # preprocessing tasks to overlap with it.
        self._stage_b_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._workers: list[threading.Thread] = []
        # Live status streams (server-sent events): job id -> listener queues.
        self._subscribers: dict[str, list[queue.Queue]] = {}
        # Owner-level streams: every transition of the owner's jobs (completion notifications).
        self._owner_subscribers: dict[str, list[queue.Queue]] = {}
        # §21.2 estimate caches: history samples (keyed by the finished-job set), per-job feature hash and face count.
        self._eta_cache: dict = {}
        self._eta_hashes: dict[str, str | None] = {}
        self._eta_faces: dict[str, float] = {}
        self._eta_default_device: str | None = None
        self._eta_warm: set = set()
        # J2 background geometry precompute: one at a time, FIFO, per-job cancel events.
        self._precompute_fn = precompute_fn
        self._precompute_enabled = (precompute_fn is not None or (stage_a_fn is None and stage_b_fn is None)) \
            and os.environ.get("WSS_DEPLOY_PRECOMPUTE", "1").strip().lower() not in {"0", "false", "off", "no"}
        self._pc_cond = threading.Condition()
        self._pc_queue: list[str] = []
        self._pc_state: dict[str, str] = {}
        self._pc_cancel: dict[str, threading.Event] = {}
        self._pc_thread: threading.Thread | None = None
        # J6: the drain flag is a file (written by ``service … --drain``); cached for a second.
        self._drain_cache = (0.0, False)
        # J6: analysis rebuilds run inside the service after an upgrade; mutations of a job being rebuilt get a 409.
        self._maintenance_busy: set[str] = set()
        self._maintenance_thread: threading.Thread | None = None
        self._next_trash_scan = 0.0
        requeue: list[dict] = []
        self._load(legacy_owner, requeue=requeue)
        if not self.offline:
            self._requeue_after_restart(requeue)
            self.purge_expired()
            self._clear_stale_drain()

    def _save(self, job: dict, *, durable: bool = True) -> None:
        # J1: compact v2 record; the embedded stage-A display geometry is gone (see slim_stage_a).
        job.setdefault("schema_version", JOB_SCHEMA_VERSION)
        atomic_json(self.root / job["id"] / "job.json", job, compact=True, durable=durable)

    def _event(self, job: dict, action: str, *, version: bool = True, durable: bool | None = None,
               actor: str | None = None, **data) -> None:
        """Append an event, persist and publish it.

        ``version`` marks a state change (bumps the optimistic-concurrency version).  ``durable`` defaults to it:
        progress-only events (``version=False``) are written atomically without fsync (J1).  ``actor`` (the
        caller's owner / user name, J8) goes to the audit log only, never into the record returned by the API.
        """
        now = time.time()
        job["updated_at"] = _date(now)
        if version:
            job["version"] = job.get("version", 0) + 1
        job.setdefault("events", []).append({"at": _date(now), "action": action,
                                             "status": job["status"], "stage": job.get("stage"), **data})
        durable = version if durable is None else durable
        self._save(job, durable=durable)
        if durable:
            self._audit(job, action, actor=actor)
        self._publish(job, action)

    @classmethod
    def _audit(cls, job: dict, action: str, *, actor: str | None = None, **extra) -> None:
        """One structured ``wss_deploy.audit`` line per state change (J8): who, what, which release and device."""
        try:
            release = job.get("model_release") if isinstance(job.get("model_release"), dict) else {}
            compute = job.get("compute") if isinstance(job.get("compute"), dict) else {}
            summary = job.get("summary") if isinstance(job.get("summary"), dict) else {}
            record = {"job": job.get("id"), "action": action, "status": job.get("status"), "stage": job.get("stage"),
                      "version": job.get("version"), "owner": _owner_label(job.get("owner")),
                      "release": release.get("id"), "device": summary.get("device") or compute.get("device")}
            if actor is not None:
                record["actor"] = _owner_label(actor)
            if job.get("status") == "done" and action in {"finished", "restart_interrupted"} and summary.get("timing_s"):
                record["timing_s"] = summary.get("timing_s")
            if job.get("status") == "failed" and isinstance(job.get("error"), dict):
                record["error"] = {key: job["error"].get(key) for key in ("category", "retryable", "diagnostic_id")}
            record.update(extra)
            AUDIT.info("job %s", json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str))
        except Exception:  # noqa: BLE001 — logging must never break a state change
            LOG.exception("Audit line for %s failed", job.get("id"))

    @classmethod
    def _live_event(cls, job: dict, action: str) -> dict:
        review = job.get("review") if isinstance(job.get("review"), dict) else {}
        return {"job_id": job["id"], "version": job.get("version"), "status": job["status"], "stage": job.get("stage"),
                "phase": job.get("phase"), "detail": job.get("detail"), "action": action, "at": job.get("updated_at"),
                "final": job["status"] in FINAL, "case_id": job.get("case_id"), "family": cls._family(job),
                "review": review.get("status") or "unreviewed"}

    @staticmethod
    def _offer(listeners, event: dict) -> None:
        for listener in list(listeners or ()):
            try:
                listener.put_nowait(event)
            except queue.Full:
                # A stalled client simply misses intermediate progress; the next event carries the state.
                pass

    def _publish(self, job: dict, action: str) -> None:
        listeners = self._subscribers.get(job["id"])
        owner_listeners = self._owner_subscribers.get(job.get("owner") or "")
        if not listeners and not owner_listeners:
            return
        event = self._live_event(job, action)
        if job.get("status") in ETA_STATUSES:
            # §21.2: progress events carry the refreshed estimate so a page listening to the stream needs no poll.
            try:
                event["eta"] = self._eta(job, now=time.time())
            except Exception:  # noqa: BLE001
                LOG.exception("ETA for %s failed", job.get("id"))
        self._offer(listeners, event)
        self._offer(owner_listeners, event)

    def subscribe(self, job_id: str, owner: str, *, any_owner: bool = False) -> tuple[queue.Queue, dict]:
        """Register a live listener; returns the queue and the current light snapshot."""
        with self.lock:
            job = self._owned(job_id, owner, any_owner=any_owner)
            listener: queue.Queue = queue.Queue(maxsize=256)
            self._subscribers.setdefault(job_id, []).append(listener)
            snapshot = self._snapshot(job)
            snapshot["final"] = job["status"] in FINAL
            return listener, snapshot

    def unsubscribe(self, job_id: str, listener: queue.Queue) -> None:
        with self.lock:
            listeners = self._subscribers.get(job_id)
            if listeners and listener in listeners:
                listeners.remove(listener)
            if listeners is not None and not listeners:
                self._subscribers.pop(job_id, None)

    def subscribe_owner(self, owner: str) -> tuple[queue.Queue, dict]:
        """Owner-level stream (contract §11.5): every transition of this owner's jobs, no job snapshots."""
        with self.lock:
            listener: queue.Queue = queue.Queue(maxsize=512)
            self._owner_subscribers.setdefault(owner, []).append(listener)
            active = sum(1 for j in self.jobs.values() if j.get("owner") == owner and j["status"] not in FINAL)
            return listener, {"owner_jobs": sum(1 for j in self.jobs.values() if j.get("owner") == owner),
                              "active_jobs": active, "at": _date(time.time())}

    def unsubscribe_owner(self, owner: str, listener: queue.Queue) -> None:
        with self.lock:
            listeners = self._owner_subscribers.get(owner)
            if listeners and listener in listeners:
                listeners.remove(listener)
            if listeners is not None and not listeners:
                self._owner_subscribers.pop(owner, None)

    def _load(self, legacy_owner: str | None, *, requeue: list | None = None) -> None:
        # A deletion renames the directory first and removes it afterwards; anything still
        # carrying the prefix belongs to a deletion the previous process did not finish.
        if not self.offline:
            for leftover in self.root.glob(DELETING_PREFIX + "*"):
                if leftover.is_dir() and leftover.resolve().parent == self.root:
                    LOG.info("Purging unfinished deletion %s", leftover.name)
                    shutil.rmtree(leftover, ignore_errors=True)
        for path in sorted(self.root.glob("*/job.json")):
            try:
                if path.parent.name.startswith(".") or path.resolve().parent.parent != self.root:
                    continue
                job = self._restore_record(path, legacy_owner, requeue=requeue)
                self.jobs[job["id"]] = job
            except Exception:
                LOG.exception("Cannot restore job record %s", path)

    def _requeue_after_restart(self, jobs: list[dict]) -> None:
        """J7: jobs that were waiting in the queue when the previous process stopped go back into the queue
        (in their original order) instead of being marked interrupted; a running attempt stays interrupted."""
        for job in sorted(jobs, key=lambda item: (item.get("queued_ts") or 0, item["id"])):
            job.setdefault("queued_ts", time.time())
            job["cancel_requested"] = False
            job.update(phase="等待计算", detail="服务已重启，任务按原顺序继续排队。")
            self._event(job, "restart_requeued")
            self.tasks.put((job["id"], job["version"], job.get("stage") or "A"))
        if jobs:
            LOG.info("Re-queued %d job(s) that were waiting when the service stopped", len(jobs))

    def _restore_record(self, path: Path, legacy_owner: str | None, *, requeue: list | None = None) -> dict:
        """Load one persisted record, migrating old shapes; also used when restoring from the trash.

        ``requeue`` (service start only) collects ``queued`` jobs for :meth:`_requeue_after_restart`; without it a
        queued job is marked interrupted like a running one (restoring a queued job from the trash never starts it).
        """
        job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("id") != path.parent.name:
            raise ValueError("job directory/id mismatch")
        job.pop("stage_clock", None)   # §21.2: belongs to an attempt of a previous process
        migrate_record(job, path=path, legacy_owner=legacy_owner)
        if not isinstance(job.get("review"), dict) or job["review"].get("status") not in REVIEW_STATUSES:
            job["review"] = fresh_review()
        if isinstance(job.get("precompute"), dict) and job["precompute"].get("status") in {"pending", "running"}:
            job["precompute"] = dict(job["precompute"], status="interrupted")
        legacy_binding = self._recover_legacy_release(path.parent, job)
        if legacy_binding is not None:
            job["model_release"] = legacy_binding
            job.setdefault("legacy_release_migration", {"source": "historical_summary_or_report",
                                                           "manifest_sha256": legacy_binding.get("legacy_manifest_sha256")})
        old = job.get("status", "failed")
        job.setdefault("stage", "B" if old.endswith("_B") else "A")
        if old == "awaiting_outlets":
            job["status"] = "awaiting_confirmation"
        if self.offline:
            pass   # J3: a maintenance process leaves every status exactly as the service wrote it
        elif old == "queued" and requeue is not None and job.get("stage") in {"A", "B"}:
            requeue.append(job)
        elif old in {"queued", "running", "queued_A", "queued_B", "running_A", "running_B"}:
            job["status"] = "interrupted"
            job["phase"] = "服务重启，任务已中断"
            job["detail"] = "请重试以恢复计算；已完成的输入和中心线可以复用。"
            job["cancel_requested"] = False
            job.pop("started_ts", None)
            job["finished_ts"] = time.time()
            self._event(job, "restart_interrupted")
        self._backfill_summary(path.parent, job)
        if legacy_binding is not None:
            # This is a state migration, so persist it before the
            # worker can accept a retry.  The historical hash is
            # retained as evidence; the current registry fingerprint
            # is what future execution checks.
            job["version"] = job.get("version", 0) + 1
            job["updated_at"] = _date(time.time())
            job.setdefault("events", []).append({"at": job["updated_at"],
                "action": "legacy_release_bound", "status": job["status"],
                "stage": job.get("stage"), "release_id": legacy_binding.get("id")})
            self._save(job)
        return job

    @classmethod
    def _backfill_summary(cls, job_dir: Path, job: dict) -> None:
        """§19.2: finished records written before v0.12 lack ``cycle`` / ``findings_top`` (and older ones the
        morphology digest); copy them from ``summary.json`` in memory.  Values are copied, never recomputed."""
        summary = job.get("summary")
        if job.get("status") != "done" or not isinstance(summary, dict):
            return
        missing = [key for key in ("cycle", "findings_top", "morphology") if key not in summary]
        if not missing:
            return
        full = cls._json_file(job_dir / "summary.json")
        if "cycle" in missing:
            summary["cycle"] = cycle_record(full.get("cycle"))
        if "findings_top" in missing:
            summary["findings_top"] = findings_top(full)
        if "morphology" in missing and isinstance(full.get("morphology"), dict):
            from .morphology import digest as morphology_digest
            summary["morphology"] = morphology_digest(full.get("morphology"))

    @staticmethod
    def _json_file(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, UnicodeError, ValueError, TypeError):
            return {}

    @classmethod
    def _report_metadata(cls, job_dir: Path) -> dict:
        """Read the old embedded report metadata without executing HTML."""
        path = job_dir / "report.html"
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
            match = re.search(r"<!--WSS_META_START--><script id=\"wss-report-meta\" type=\"application/json\">(.*?)</script><!--WSS_META_END-->", text, re.S)
            if not match:
                return {}
            value = json.loads(match.group(1))
            return value if isinstance(value, dict) else {}
        except (OSError, UnicodeError, ValueError, TypeError):
            return {}

    def _recover_legacy_release(self, job_dir: Path, job: dict) -> dict | None:
        """Bind pre-registry jobs from explicit historical evidence only.

        Before the registry existed, summaries recorded ``release`` plus the
        MANIFEST hash as ``release_hash``.  Do not infer the default release:
        recover only when both the release id and a valid historical hash are
        present and the registry verifies the package bytes.
        """
        if self.registry is None or job.get("model_release"):
            return None
        summary = self._json_file(job_dir / "summary.json")
        embedded_summary = job.get("summary") if isinstance(job.get("summary"), dict) else {}
        manifest = self._json_file(job_dir / "run_manifest.json")
        report = self._report_metadata(job_dir)
        model_records = [embedded_summary.get("model_release"), summary.get("model_release"),
                         manifest.get("model_release"), report.get("model_release")]
        model_records = [item for item in model_records if isinstance(item, dict)]
        release_id = None
        for source in [*model_records, embedded_summary, summary, report]:
            for key in ("registry_id", "id", "release", "name"):
                value = source.get(key) if isinstance(source, dict) else None
                if isinstance(value, str) and value.strip():
                    release_id = value.strip()
                    break
            if release_id:
                break
        manifest_provenance = manifest.get("provenance") if isinstance(manifest.get("provenance"), dict) else {}
        manifest_sha256 = None
        for source in [*model_records, embedded_summary, summary, report, manifest_provenance]:
            if not isinstance(source, dict):
                continue
            for key in ("manifest_sha256", "release_hash", "release_sha256"):
                value = source.get(key)
                if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
                    manifest_sha256 = value
                    break
            if manifest_sha256:
                break
        release_json_sha256 = None
        for source in model_records:
            value = source.get("release_json_sha256") or source.get("release_json_hash")
            if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
                release_json_sha256 = value
                break
        if not release_id or not manifest_sha256:
            return None
        try:
            descriptor = self.registry.restore_legacy_binding(
                release_id, manifest_sha256=manifest_sha256,
                release_json_sha256=release_json_sha256)
        except Exception as exc:
            LOG.warning("Cannot bind historical release for %s: %s", job.get("id"), exc)
            return None
        public = descriptor.public()
        public["legacy_manifest_sha256"] = manifest_sha256.lower()
        if release_json_sha256:
            public["legacy_release_json_sha256"] = release_json_sha256.lower()
        return public

    def _owned(self, job_id: str, owner: str, *, any_owner: bool = False) -> dict:
        """``any_owner`` is the administrator's read-only view; mutations never pass it."""
        job = self.jobs.get(job_id)
        if not job or (not any_owner and job.get("owner") != owner):
            raise JobError("任务不存在或不属于当前会话。", 404)
        return job

    def _require_idle(self, job: dict) -> None:
        """J6: a job whose analysis is being rebuilt by the post-upgrade maintenance accepts no mutation."""
        if job.get("id") in self._maintenance_busy:
            raise JobError("该任务正在后台按新版分析重建（升级后自动执行），请稍后再操作。", 409)

    def _version(self, job: dict, payload: dict) -> None:
        self._require_idle(job)
        if type(payload.get("version")) is not int or payload["version"] != job["version"]:
            raise JobError("任务状态已更新，请刷新后再操作。", 409)
        if "stage" in payload and payload["stage"] != job.get("stage"):
            raise JobError("任务阶段已改变，请刷新后再操作。", 409)

    def _snapshot(self, job: dict, *, detail: bool = True) -> dict:
        now = time.time()
        compute = job.get("compute_seconds", 0.0)
        attempt = job.get("attempt_seconds", 0.0)
        if job["status"] == "running":
            running = max(0, now - job.get("started_ts", now))
            compute += running
            attempt += running
        queued = job.get("queue_seconds", 0.0)
        if job["status"] == "queued":
            queued += max(0, now - job.get("queued_ts", now))
        confirmation = job.get("confirmation_seconds", 0.0)
        if job["status"] in {"awaiting_input", "awaiting_confirmation"}:
            confirmation += max(0, now - job.get("awaiting_ts", now))
        public = {key: val for key, val in job.items() if key not in {
            "owner", "filename", "traceback", "started_ts", "created_ts", "queued_ts", "awaiting_ts", "stage_clock", "input_faces"}}
        if not detail:
            public = {key: public.get(key) for key in (
                "id", "case_id", "status", "stage", "version", "created_at", "updated_at", "phase", "detail", "error",
                "patient_id", "scan_label", "scan_date", "tags", "notes", "source_filename", "model_release",
                "source_job_id", "reused_from", "batch_id", "run_identity", "compute", "review")}
            public["finished_at"] = _date(job["finished_ts"]) if job.get("status") == "done" and isinstance(job.get("finished_ts"), (int, float)) else None
            public["family"] = self._family(job)
            public["input_sha256"] = self._input_sha256(job)
            public["reusable"] = self._reusable(job)
            public["annotations_count"] = len(((job.get("annotations") or {}).get("items") or []))
            public["findings_review_count"] = len(((job.get("findings_review") or {}).get("items") or {}))
            public["snapshots_count"] = len(((job.get("snapshots") or {}).get("items") or []))
            summary = job.get("summary") if isinstance(job.get("summary"), dict) else {}
            from .morphology import max_diameter_mm
            public["max_diameter_mm"] = max_diameter_mm(summary.get("morphology"))
            public.update(self._labels(job))
        elif isinstance(public.get("a"), dict):
            # The stage-A snapshot carries a display mesh and centreline polylines (megabytes).
            # They are served once by ``geometry()``; the status snapshot stays small so polling
            # clients and the worker's progress events never contend on a large deep copy.
            light = dict(public["a"])
            light.pop("preview", None)
            proposal = light.get("proposal")
            if isinstance(proposal, dict) and "preview_polylines" in proposal:
                proposal = dict(proposal)
                proposal.pop("preview_polylines", None)
                light["proposal"] = proposal
            light["geometry_endpoint"] = f"/api/jobs/{job['id']}/geometry"
            public["a"] = light
            # Keep the old name during the UI transition; both point to the
            # same immutable stage-A snapshot in the response.
            public.setdefault("stage_a", light)
        if detail:
            public.update(self._labels(job))
        public["elapsed"] = round(compute, 1)
        public["timing"] = {"compute_s": round(compute, 1), "attempt_s": round(attempt, 1), "queue_s": round(queued, 1),
                            "confirmation_s": round(confirmation, 1),
                            "wall_s": round(max(0, job.get("finished_ts", now) - job.get("created_ts", now)), 1)}
        # Long names are kept for the browser and older report consumers.
        public["timing"].update(queue_seconds=round(queued, 1),
                                confirmation_seconds=round(confirmation, 1),
                                compute_seconds=round(compute, 1))
        precompute = job.get("precompute") if isinstance(job.get("precompute"), dict) else {}
        if isinstance(precompute.get("seconds"), (int, float)):
            # J2: background geometry work done while the outlets were being confirmed (not in compute_s).
            public["timing"]["precompute_s"] = round(float(precompute["seconds"]), 1)
        waiting = sorted((j for j in self.jobs.values() if j["status"] == "queued"),
                         key=lambda j: j.get("queued_ts", 0))
        public["queue_position"] = next((i + 1 for i, j in enumerate(waiting) if j["id"] == job["id"]), None)
        if job["status"] in ETA_STATUSES:
            try:
                public["eta"] = self._eta(job, now=now)
            except Exception:  # noqa: BLE001 — an estimate must never break a status response
                LOG.exception("ETA for %s failed", job.get("id"))
        return copy.deepcopy(public)

    # ------------------------------------------------------------- §21.2 remaining-time estimate
    def _job_faces(self, job: dict) -> float | None:
        a = job.get("a") if isinstance(job.get("a"), dict) else {}
        faces = (a.get("input_check") or {}).get("faces") if isinstance(a.get("input_check"), dict) else None
        if isinstance(faces, (int, float)) and faces > 0:
            return float(faces)
        if isinstance(job.get("input_faces"), (int, float)) and job["input_faces"] > 0:
            return float(job["input_faces"])
        cached = self._eta_faces.get(job["id"])
        if cached is None:
            from .eta import faces_from_size
            try:
                cached = faces_from_size((self.root / job["id"] / (job.get("filename") or "input.stl")).stat().st_size) or 0
            except OSError:
                cached = 0
            self._eta_faces[job["id"]] = cached
        return float(cached) or None

    def _feature_hash(self, job: dict) -> str | None:
        """``summary.feature_contract.source_hash`` from the record, else (once) from summary.json's text."""
        summary = job.get("summary") if isinstance(job.get("summary"), dict) else {}
        contract = summary.get("feature_contract") if isinstance(summary.get("feature_contract"), dict) else {}
        if contract.get("source_hash"):
            return contract["source_hash"]
        if job["id"] not in self._eta_hashes:
            value = None
            try:
                text = (self.root / job["id"] / "summary.json").read_text(encoding="utf-8")
                match = re.search(r'"feature_contract":\s*\{[^{}]*?"source_hash":\s*"([0-9a-f]{64})"', text)
                value = match.group(1) if match else None
            except (OSError, UnicodeError):
                value = None
            self._eta_hashes[job["id"]] = value
        return self._eta_hashes[job["id"]]

    def _eta_samples(self) -> list[dict]:
        """History records of every finished job; rebuilt only when the set of finished jobs changes."""
        from .eta import history_sample
        done = [job for job in self.jobs.values() if job.get("status") == "done"]
        signature = (len(done), round(sum(float(job.get("finished_ts") or 0) for job in done), 3))
        if self._eta_cache.get("signature") == signature:
            return self._eta_cache["samples"]
        samples = []
        for job in done:
            summary = job.get("summary") if isinstance(job.get("summary"), dict) else {}
            a = job.get("a") if isinstance(job.get("a"), dict) else {}
            timing = {**(a.get("timing_s") or {}), **(summary.get("timing_s") or {})}
            from .eta import cache_reused_stages
            covered = self._precomputed_stages(job) | cache_reused_stages(summary.get("geometry_cache"))
            if covered:
                # J2: stage-B timings of reused (precomputed / copied) intermediates are ≈ 0 s; they must not drag
                # the per-stage history of full runs towards zero, so those stages of this run are left out.
                from .eta import TIMING_KEYS
                timing = {key: value for key, value in timing.items() if TIMING_KEYS.get(key) not in covered}
            sample = history_sample(family=self._family(job), faces=self._job_faces(job), timing=timing,
                                    source_hash=self._feature_hash(job), release_id=(job.get("model_release") or {}).get("id"),
                                    device=summary.get("device"), cold_start=bool(summary.get("eta_cold_start")))
            if sample:
                samples.append(sample)
        self._eta_cache = {"signature": signature, "samples": samples}
        return samples

    def _eta_device(self, job: dict) -> str:
        """Device the job's inference will use: the job's choice, else the service's, else whether a GPU is visible."""
        device = (job.get("compute") or {}).get("device") if isinstance(job.get("compute"), dict) else None
        if device in {"cpu", "cuda"}:
            return device
        device = getattr(self.registry, "device", None)
        if device in {"cpu", "cuda"}:
            return device
        if self._eta_default_device is None:
            import sys
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible is not None and visible.strip() in {"", "-1"}:
                self._eta_default_device = "cpu"
            elif "torch" in sys.modules:
                try:
                    self._eta_default_device = "cuda" if sys.modules["torch"].cuda.is_available() else "cpu"
                except Exception:  # noqa: BLE001
                    self._eta_default_device = "cuda"
            else:
                return "cuda"                                   # not cached: torch may be imported later
        return self._eta_default_device

    def _eta(self, job: dict, *, now: float, with_queue: bool = True) -> dict | None:
        from .eta import build_eta, current_feature_hash, expected_table, queue_wait, stage_seconds
        family = self._family(job) or "wall"
        faces = self._job_faces(job)
        release = job.get("model_release") if isinstance(job.get("model_release"), dict) else {}
        compute = job.get("compute") if isinstance(job.get("compute"), dict) else {}
        models = compute.get("seed_count") or release.get("models_count")
        table, basis, n_history = expected_table(family, faces, self._eta_samples(), source_hash=current_feature_hash(),
                                                 release_id=release.get("id"), device=self._eta_device(job),
                                                 models_count=models if isinstance(models, int) else None)
        a = job.get("a") if isinstance(job.get("a"), dict) else {}
        segment = job.get("stage") if job.get("stage") in {"A", "B"} else "A"
        if job["status"] in {"awaiting_input", "awaiting_confirmation"}:
            segment = "A" if job["status"] == "awaiting_input" else "B"
        queue_ahead = queue_ahead_s = None
        clock = job.get("stage_clock") if isinstance(job.get("stage_clock"), dict) else {}
        waiting = job["status"] == "running" and bool(clock.get("waiting"))
        if with_queue and job["status"] == "queued":
            mine = job.get("queued_ts", now)
            ahead = [other for other in self.jobs.values() if other is not job and (
                other["status"] == "running" or (other["status"] == "queued" and other.get("queued_ts", 0) < mine))]
            blocks = [block for block in (self._eta(other, now=now, with_queue=False) for other in ahead) if block]
            queue_ahead, queue_ahead_s = len(ahead), queue_wait(blocks)
        elif with_queue and waiting:
            # Taken by a worker but queued behind the stage-B job that holds the lock.
            ahead = [other for other in self.jobs.values() if other is not job and other["status"] == "running"
                     and other.get("stage") == "B" and not (other.get("stage_clock") or {}).get("waiting")]
            blocks = [block for block in (self._eta(other, now=now, with_queue=False) for other in ahead) if block]
            queue_ahead, queue_ahead_s = len(ahead), sum(float(b.get("segment_remaining_s") or 0.0) for b in blocks)
        covered = self._precomputed_stages(job)
        if covered:
            from .eta import apply_precompute
            table = apply_precompute(table, covered)
        out = build_eta(family=family, status=job["status"], segment=segment, faces=faces, table=table, basis=basis,
                        n_history=n_history, a_seconds=stage_seconds(a.get("timing_s")), clock=clock,
                        now=now, queue_ahead=queue_ahead, queue_ahead_s=queue_ahead_s)
        if out is not None and covered:
            out["precomputed"] = sorted(covered)
        if out is not None and waiting:
            out["waiting"] = True
        return out

    def _precomputed_stages(self, job: dict) -> set:
        """Stage keys whose work a finished background precompute already did for this job (J2)."""
        precompute = job.get("precompute") if isinstance(job.get("precompute"), dict) else {}
        if precompute.get("status") != "done":
            return set()
        from .eta import precompute_stages, stage_seconds, steps_to_stages
        covered = set(stage_seconds(precompute.get("timing_s") if isinstance(precompute.get("timing_s"), dict) else {}))
        covered |= steps_to_stages(precompute.get("steps"), self._family(job))
        return covered or set(precompute_stages(self._family(job)))

    @staticmethod
    def _labels(job: dict) -> dict:
        """Display labels the workbench shows as-is (§19.2): family text, short release name, cycle flag."""
        from .timeline import family_label
        summary = job.get("summary") if isinstance(job.get("summary"), dict) else {}
        return {"family_label": family_label(job), "release_short": release_short(job.get("model_release")),
                "has_cycle": bool(summary.get("cycle"))}

    def releases(self) -> list[dict]:
        if self.registry is None:
            return []
        return self.registry.list()

    def _known_releases(self) -> list[dict]:
        """Release descriptors for case cards: the registry, or the single injected release of a test manager."""
        if self.registry is not None:
            return self.registry.list()
        return [self._release_record()] if self.release is not None else []

    @staticmethod
    def _family(job: dict) -> str | None:
        """Result family: ``wall`` (WSS) or ``volume`` (pressure + velocity).

        A finished summary is authoritative; before stage B the bound release contract tells it.
        """
        summary = job.get("summary") if isinstance(job.get("summary"), dict) else None
        if summary:
            fields = summary.get("fields") if isinstance(summary.get("fields"), dict) else {}
            return "volume" if "velocity" in fields or "pressure" in fields else "wall"
        return family_of_release(job.get("model_release"))

    @staticmethod
    def _input_sha256(job: dict) -> str | None:
        value = job.get("input_sha256")
        if not value:
            a = job.get("a") if isinstance(job.get("a"), dict) else {}
            value = a.get("input_sha256")
        return str(value) if value else None

    @classmethod
    def _reusable(cls, job: dict) -> bool:
        """Geometry can be reused when stage A passed and the outlets were confirmed (B may be rerun)."""
        return bool(cls._gate(job.get("a")) and isinstance(job.get("mapping"), dict) and job.get("mapping"))

    @staticmethod
    def locked(job: dict) -> bool:
        """A signed-off result is frozen: no recomputation, override or deletion until reopened."""
        review = job.get("review") if isinstance(job.get("review"), dict) else {}
        return review.get("status") == "reviewed"

    def _require_unlocked(self, job: dict, action: str) -> None:
        if self.locked(job):
            raise JobError(f"该结果已审阅签字并锁定，不能{action}；请先「重新打开」并说明原因。", 409)

    def review(self, job_id: str, owner: str, payload: dict) -> dict:
        """Sign off (approve) or reopen a finished result.

        ``approve`` requires ``status == done`` and freezes the job; ``reopen`` needs a reason and
        lifts the lock.  The decision is copied into ``summary.json["review"]``, the embedded report
        metadata and the run manifest audit so an exported report carries its review state.
        """
        decision = payload.get("decision")
        if decision not in {"approve", "reopen"}:
            raise JobError("decision 必须是 approve 或 reopen。")
        reviewer = self._text(payload.get("reviewer"), "审阅人标识", 80)
        note = self._text(payload.get("note"), "审阅备注", 2000, multiline=True)
        if not reviewer:
            raise JobError("请填写审阅人的匿名标识。")
        with self.lock:
            job = self._owned(job_id, owner)
            self._version(job, payload)
            current = job.get("review") if isinstance(job.get("review"), dict) else fresh_review()
            if decision == "approve":
                if job.get("status") != "done":
                    raise JobError("只有已完成的结果才能审阅签字。", 409)
                if current.get("status") == "reviewed":
                    raise JobError("该结果已经审阅签字；如需修改请先重新打开。", 409)
                status, action = "reviewed", "review_approved"
            else:
                if current.get("status") != "reviewed":
                    raise JobError("该结果没有处于已审阅状态，无需重新打开。", 409)
                if not note:
                    raise JobError("重新打开必须说明原因。")
                status, action = "reopened", "review_reopened"
            record = {"status": status, "by": reviewer, "at": _date(time.time()), "note": note,
                      "version": job.get("version", 0) + 1}
            job.setdefault("review_history", []).append(dict(record, previous=current.get("status")))
            job["review"] = record
            if isinstance(job.get("summary"), dict):
                job["summary"]["review"] = dict(record)
            self._event(job, action, actor=owner, reviewer=reviewer, note=note[:200], review_status=status)
            self._persist_review(self.root / job_id, record, list(job.get("review_history", [])))
            return self._snapshot(job)

    @staticmethod
    def _persist_review(job_dir: Path, record: dict, history: list) -> None:
        """Copy the review decision into the portable result files when they exist."""
        summary_path = job_dir / "summary.json"
        if not summary_path.is_file():
            return
        try:
            meta = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            LOG.warning("Cannot read %s to record the review decision", summary_path)
            return
        if not isinstance(meta, dict):
            return
        meta["review"] = dict(record)
        audit = meta.get("audit") if isinstance(meta.get("audit"), dict) else {}
        audit["review"] = dict(record)
        audit["review_history"] = history
        meta["audit"] = audit
        atomic_json(summary_path, meta)
        report_path = job_dir / "report.html"
        if report_path.is_file():
            try:
                from .report import update_html_meta
                update_html_meta(report_path, meta)
            except (OSError, ValueError) as exc:
                LOG.warning("Report metadata of %s not refreshed with the review decision: %s", job_dir.name, exc)
        manifest_path = job_dir / "run_manifest.json"
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                outputs = list((existing.get("outputs") or {}).keys()) if isinstance(existing, dict) else []
                from .schema import write_run_manifest
                write_run_manifest(job_dir, meta, outputs=outputs)
            except (OSError, ValueError, TypeError) as exc:
                LOG.warning("Run manifest of %s not refreshed with the review decision: %s", job_dir.name, exc)

    # ------------------------------------------------------------- annotations (C8) / findings review (C16)
    @staticmethod
    def _xyz(value, label: str) -> list[float]:
        if not isinstance(value, list) or len(value) != 3 or not all(isinstance(v, (int, float)) and v == v for v in value):
            raise JobError(f"{label}的 xyz_mm 必须是三个有限数。")
        return [float(v) for v in value]

    def _sidecar_target(self, job_id: str, owner: str, payload: dict, action: str) -> dict:
        job = self._owned(job_id, owner)
        self._require_idle(job)
        if job.get("status") != "done":
            raise JobError(f"任务完成后才能{action}。", 409)
        self._require_unlocked(job, action)
        if "version" in payload and payload["version"] is not None:
            self._version(job, payload)
        return job

    def _persist_summary(self, job_dir: Path, apply, *, what: str) -> None:
        """Apply ``apply`` to summary.json and refresh the report's embedded metadata copy.

        Both files are optional: a job whose stage B never ran simply has nothing to mirror into.
        """
        summary_path = job_dir / "summary.json"
        meta = self._json_file(summary_path) if summary_path.is_file() else None
        if meta is None:
            return
        apply(meta)
        atomic_json(summary_path, meta)
        report_path = job_dir / "report.html"
        if report_path.is_file():
            try:
                from .report import update_html_meta
                update_html_meta(report_path, meta)
            except (OSError, ValueError) as exc:
                LOG.warning("Report metadata of %s not refreshed with %s: %s", job_dir.name, what, exc)

    def _persist_sidecar(self, job_dir: Path, name: str, document: dict, apply) -> None:
        """Write ``<name>`` and mirror it into summary.json + the embedded report metadata (offline copies)."""
        atomic_json(job_dir / name, document)
        self._persist_summary(job_dir, apply, what=name)

    def annotations(self, job_id: str, owner: str, payload: dict) -> dict:
        """Replace the wall/volume annotations pinned in the report (contract §11.7)."""
        raw = payload.get("items")
        if not isinstance(raw, list) or len(raw) > MAX_ANNOTATIONS:
            raise JobError(f"标注必须是最多 {MAX_ANNOTATIONS} 条的列表。")
        items, seen = [], set()
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise JobError("每条标注必须是对象。")
            ident = self._text(item.get("id"), "标注编号", 16) or f"A{index + 1}"
            if ident in seen:
                raise JobError("标注编号重复。")
            seen.add(ident)
            text = self._text(item.get("text"), "标注文字", 200, multiline=True)
            if not text:
                raise JobError("标注文字不能为空。")
            color = self._text(item.get("color"), "标注颜色", 16)
            if color and not re.fullmatch(r"#[0-9a-fA-F]{3,8}", color):
                raise JobError("标注颜色必须是 #rrggbb 形式。")
            entry = {"id": ident, "xyz_mm": self._xyz(item.get("xyz_mm"), "标注"), "text": text, "color": color or "#d97706",
                     "branch": self._text(item.get("branch"), "分支", 40),
                     "created_at": self._text(item.get("created_at"), "创建时间", 40) or _date(time.time())}
            for key in ("segment_id", "s_from_root_mm"):
                value = item.get(key)
                if value is not None:
                    if not isinstance(value, (int, float)) or value != value:
                        raise JobError(f"标注的 {key} 必须是数字。")
                    entry[key] = value
            items.append(entry)
        with self.lock:
            job = self._sidecar_target(job_id, owner, payload, "编辑标注")
            document = {"schema_version": "wss-deploy.annotations/v1", "items": items,
                        "updated_at": _date(time.time()), "updated_by": owner}
            job["annotations"] = document
            if isinstance(job.get("summary"), dict):
                job["summary"]["annotations"] = document
            self._event(job, "annotations_updated", actor=owner, count=len(items))
            self._persist_sidecar(self.root / job_id, "annotations.json", document,
                                  lambda meta: meta.__setitem__("annotations", document))
            return {"annotations": document, "version": job["version"]}

    def findings_review(self, job_id: str, owner: str, payload: dict) -> dict:
        """Reviewer decisions on the automatic findings plus manual additions (contract §11.10)."""
        raw_items = payload.get("items", {})
        if not isinstance(raw_items, dict) or len(raw_items) > 200:
            raise JobError("items 必须是以发现编号为键的对象。")
        items = {}
        for key, value in raw_items.items():
            ident = self._text(key, "发现编号", 16)
            if not ident or not isinstance(value, dict):
                raise JobError("每条发现判定必须是对象。")
            decision = value.get("decision")
            if decision is not None and decision not in FINDING_DECISIONS:
                raise JobError("decision 只能是 confirmed、rejected 或 null。")
            items[ident] = {"decision": decision, "note": self._text(value.get("note"), "发现备注", 500, multiline=True)}
        raw_added = payload.get("added", [])
        if not isinstance(raw_added, list) or len(raw_added) > MAX_ADDED_FINDINGS:
            raise JobError(f"手动发现最多 {MAX_ADDED_FINDINGS} 条。")
        added, seen = [], set()
        for index, item in enumerate(raw_added):
            if not isinstance(item, dict):
                raise JobError("每条手动发现必须是对象。")
            ident = self._text(item.get("id"), "发现编号", 16) or f"M{index + 1}"
            if ident in seen:
                raise JobError("手动发现编号重复。")
            seen.add(ident)
            text = self._text(item.get("text"), "发现文字", 500, multiline=True)
            if not text:
                raise JobError("手动发现的文字不能为空。")
            entry = {"id": ident, "xyz_mm": self._xyz(item.get("xyz_mm"), "手动发现"), "text": text, "kind": "manual",
                     "branch": self._text(item.get("branch"), "分支", 40),
                     "severity": item.get("severity") if item.get("severity") in {"attention", "note", "info"} else "note",
                     "created_at": self._text(item.get("created_at"), "创建时间", 40) or _date(time.time())}
            for key in ("segment_id", "s_from_root_mm", "value"):
                value = item.get(key)
                if value is not None:
                    if not isinstance(value, (int, float)) or value != value:
                        raise JobError(f"手动发现的 {key} 必须是数字。")
                    entry[key] = value
            units = self._text(item.get("units"), "单位", 16)
            if units:
                entry["units"] = units
            added.append(entry)
        with self.lock:
            job = self._sidecar_target(job_id, owner, payload, "编辑发现判定")
            document = {"schema_version": "wss-deploy.findings_review/v1", "items": items, "added": added,
                        "updated_at": _date(time.time()), "updated_by": owner}
            job["findings_review"] = document

            def apply(meta: dict) -> None:
                findings = meta.get("findings") if isinstance(meta.get("findings"), dict) else {}
                findings["review"] = document
                meta["findings"] = findings
            if isinstance(job.get("summary"), dict):
                apply(job["summary"])
            self._event(job, "findings_review_updated", actor=owner, decided=sum(1 for v in items.values() if v["decision"]), added=len(added))
            self._persist_sidecar(self.root / job_id, "findings_review.json", document, apply)
            return {"findings_review": document, "version": job["version"]}

    def narrative(self, job_id: str, owner: str, payload: dict) -> dict:
        """Replace or restore the automatic conclusion of a result (contract §17.2).

        The generated text stays in ``narrative.json`` under ``auto`` so an edit can always be
        compared with (or reverted to) what the numbers say; an empty string clears the edit.
        """
        from .narrative import build_narrative, merge_edit
        text = payload.get("text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise JobError("结论文字必须是字符串。")
        text = text.replace("\r\n", "\n").strip()
        if len(text) > MAX_NARRATIVE_CHARS:
            raise JobError(f"结论文字不能超过 {MAX_NARRATIVE_CHARS} 字符。")
        with self.lock:
            job = self._sidecar_target(job_id, owner, payload, "编辑结论")
            job_dir = self.root / job_id
            stored = self._json_file(job_dir / "narrative.json") or {}
            summary = self._json_file(job_dir / "summary.json") or {}
            auto = stored.get("auto") if isinstance(stored.get("auto"), dict) else None
            if auto is None:
                current = summary.get("narrative")
                auto = {key: value for key, value in current.items()
                        if key not in {"edited", "edited_by", "edited_at"}} if isinstance(current, dict) else None
            if auto is None:
                auto = build_narrative(summary)
            stamp = _date(time.time())
            block = merge_edit(auto, text, edited_by=owner, edited_at=stamp)
            document = {"schema_version": block.get("schema_version"), "auto": auto,
                        "edited": block["edited"], "edited_by": block["edited_by"],
                        "edited_at": block["edited_at"], "updated_at": stamp, "updated_by": owner}
            job["narrative"] = document

            def apply(meta: dict) -> None:
                current = meta.get("narrative") if isinstance(meta.get("narrative"), dict) else auto
                meta["narrative"] = merge_edit(current, text, edited_by=owner, edited_at=stamp)
            if isinstance(job.get("summary"), dict):
                apply(job["summary"])
            self._event(job, "narrative_updated", actor=owner, edited=bool(block["edited"]))
            self._persist_sidecar(job_dir, "narrative.json", document, apply)
            return {"narrative": document, "version": job["version"]}

    # ------------------------------------------------------------- one-page snapshots (§15.1) / metadata (§15.2)
    @classmethod
    def _snapshot_image(cls, item, seen: set) -> tuple[dict, bytes]:
        """Validate one posted image; returns the manifest entry and the decoded PNG bytes."""
        if not isinstance(item, dict):
            raise JobError("每张配图必须是对象。")
        name = item.get("name")
        if not isinstance(name, str) or not SNAPSHOT_NAME.fullmatch(name):
            raise JobError("配图名称只能是小写字母、数字、下划线或短横线，最多 32 个字符。")
        if name in seen:
            raise JobError("配图名称重复。")
        seen.add(name)
        raw = item.get("png_base64")
        if not isinstance(raw, str) or not raw:
            raise JobError("配图必须提供 png_base64。")
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            raise JobError("配图的 png_base64 不是有效的 base64。")
        if not data.startswith(PNG_MAGIC):
            raise JobError("配图必须是 PNG 图片。")
        if len(data) > MAX_SNAPSHOT_BYTES:
            raise JobError(f"单张配图解码后不能超过 {MAX_SNAPSHOT_BYTES // 1024 // 1024} MB。", 413)
        entry = {"name": name, "file": f"snapshot_{name}.png",
                 "caption": cls._text(item.get("caption"), "配图说明", 200),
                 "view": cls._text(item.get("view"), "视角", 32),
                 "bytes": len(data), "created_at": cls._text(item.get("created_at"), "创建时间", 40) or _date(time.time())}
        for key in ("width", "height"):
            value = item.get(key)
            if value is None:
                continue
            if type(value) is not int or not 1 <= value <= 20000:
                raise JobError(f"配图的 {key} 必须是 1 到 20000 的整数。")
            entry[key] = value
        return entry, data

    def snapshots(self, job_id: str, owner: str, payload: dict) -> dict:
        """Store the report's rendered views for the one-page report (contract §15.1).

        Pictures are illustrations of an existing result, so a signed-off job stays writable here;
        no number in ``summary.json`` is touched.
        """
        raw = payload.get("images")
        if not isinstance(raw, list) or not raw or len(raw) > MAX_SNAPSHOTS:
            raise JobError(f"配图必须是 1 到 {MAX_SNAPSHOTS} 张的列表。")
        replace = payload.get("replace", True)
        if type(replace) is not bool:
            raise JobError("replace 必须是布尔值。")
        seen: set = set()
        incoming = [self._snapshot_image(item, seen) for item in raw]
        with self.lock:
            job = self._owned(job_id, owner)
            self._require_idle(job)
            if job.get("status") != "done":
                raise JobError("任务完成后才能生成一页纸配图。", 409)
            if "version" in payload and payload["version"] is not None:
                self._version(job, payload)
            job_dir = self.root / job_id
            kept = [] if replace else [item for item in ((job.get("snapshots") or {}).get("items") or [])
                                       if isinstance(item, dict) and item.get("name") not in seen]
            items = [*kept, *(entry for entry, _ in incoming)]
            if len(items) > MAX_SNAPSHOTS:
                raise JobError(f"配图最多保留 {MAX_SNAPSHOTS} 张。")
            for entry, data in incoming:
                # Write to a temporary name first: a half-written PNG must never be served.
                tmp = job_dir / (entry["file"] + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(job_dir / entry["file"])
            document = {"schema_version": "wss-deploy.snapshots/v1", "items": items,
                        "updated_at": _date(time.time()), "updated_by": owner}
            atomic_json(job_dir / "snapshots.json", document)
            live = {item.get("file") for item in items}
            for stale in sorted(job_dir.glob("snapshot_*.png")):
                if SNAPSHOT_FILE.fullmatch(stale.name) and stale.name not in live:
                    stale.unlink(missing_ok=True)
            job["snapshots"] = document
            self._event(job, "snapshots_updated", actor=owner, count=len(items))
            return {"snapshots": document, "version": job["version"]}

    def update_metadata(self, job_id: str, owner: str, payload: dict) -> dict:
        """Correct the case identifiers of an existing task (contract §15.2).

        Metadata is independent of the computation, so any status is editable; only a signed-off
        result is frozen.  Provided keys are merged onto the current values and re-validated as a
        whole, so partial edits cannot bypass a check.
        """
        with self.lock:
            job = self._owned(job_id, owner)
            self._require_idle(job)
            self._require_unlocked(job, "修改元数据")
            if "version" in payload and payload["version"] is not None:
                self._version(job, payload)
            merged = {key: (payload[key] if key in payload else job.get(key, [] if key == "tags" else ""))
                      for key in CASE_METADATA}
            metadata = self._metadata(merged)
            case_id = job.get("case_id") or ""
            if "case_id" in payload:
                case_id = self._text(payload.get("case_id"), "病例编号", 160)
                if not case_id:
                    raise JobError("病例编号不能为空。")
            changed = sorted(key for key in (*CASE_METADATA, "case_id")
                             if key in payload and (metadata | {"case_id": case_id})[key] != job.get(key))
            job.update(metadata)
            job["case_id"] = case_id
            if isinstance(job.get("summary"), dict):
                job["summary"]["case_metadata"] = dict(metadata)
                job["summary"]["case_id"] = case_id

            def apply(meta: dict) -> None:
                meta["case_metadata"] = dict(metadata)
                meta["case_id"] = case_id
            self._event(job, "metadata_updated", actor=owner, changed=changed)
            self._persist_summary(self.root / job_id, apply, what="case_metadata")
            return self._snapshot(job)

    def _release_record(self, release_id: str | None = None) -> dict:
        if self.registry is not None:
            return self.registry.describe(release_id).public()
        obj = self.release
        if obj is None:
            # Unit-test/custom managers may inject only stage functions.  Keep
            # their historical behavior; the production service always
            # constructs a registry and therefore never takes this branch.
            return {"id": "legacy", "release": "legacy", "contract": {"protocol": "single_frame_wss"}}
        rid = getattr(obj, "registry_id", None) or getattr(obj, "name", None) or "legacy"
        fp = getattr(obj, "registry_fingerprint", None)
        return {"id": str(rid), "release": str(rid), **({"fingerprint": fp} if fp else {}),
                "contract": getattr(obj, "registry_contract", {"protocol": "single_frame_wss"})}

    def _release_for(self, job: dict, *, load: bool = True, device: str | None = None):
        """The bound release (verified fingerprint); ``device`` overrides the job's choice (CPU fallback, J7)."""
        if self.registry is not None:
            try:
                release_id = (job.get("model_release") or {}).get("id")
                if not release_id or not (job.get("model_release") or {}).get("fingerprint"):
                    raise JobError("历史任务缺少发布包指纹，不能安全重试。", 409)
                descriptor = self.registry.describe(release_id)
                if descriptor.fingerprint != job["model_release"]["fingerprint"]:
                    raise JobError("绑定的发布包指纹已变化，不能继续重试。", 409)
                if load:
                    compute = job.get("compute") or {}
                    return self.registry.load(release_id,
                                              device=device or compute.get("device"),
                                              seed_count=compute.get("seed_count"))
                return descriptor
            except JobError:
                raise
            except Exception as exc:
                raise JobError(f"绑定的发布包不可用：{exc}", 422)
        return self.release

    def get(self, job_id: str, owner: str, *, any_owner: bool = False) -> dict:
        with self.lock:
            return self._snapshot(self._owned(job_id, owner, any_owner=any_owner))

    def _stage_a_file(self, job_id: str, created_at) -> dict:
        """``stage_a.json`` of a job when it belongs to the same stage-A run as the record (``created_at``)."""
        try:
            stored = json.loads((self.root / job_id / STAGE_A_FILE).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return {}
        if not isinstance(stored, dict) or (created_at is not None and stored.get("created_at") != created_at):
            return {}
        return stored

    @staticmethod
    def _with_display_geometry(a: dict, stored: dict) -> dict:
        """``a`` plus the display geometry from ``stored`` where the record does not embed it (v2 records)."""
        full = dict(a)
        if "preview" not in full and "preview" in stored:
            full["preview"] = stored["preview"]
        proposal = full.get("proposal") if isinstance(full.get("proposal"), dict) else None
        stored_proposal = stored.get("proposal") if isinstance(stored.get("proposal"), dict) else {}
        if proposal is not None and "preview_polylines" not in proposal and "preview_polylines" in stored_proposal:
            full["proposal"] = {**proposal, "preview_polylines": stored_proposal["preview_polylines"]}
        return full

    def stage_a(self, job_id: str) -> dict | None:
        """The complete stage-A snapshot of a job (record + display geometry from ``stage_a.json``), a copy.

        v0.14 accessor for readers of the former ``job["a"]["preview"]``; None when the job has no stage A."""
        with self.lock:
            job = self.jobs.get(job_id)
            a = copy.deepcopy(job.get("a")) if job and isinstance(job.get("a"), dict) else None
        if a is None:
            return None
        return self._with_display_geometry(a, self._stage_a_file(job_id, a.get("created_at")))

    def geometry(self, job_id: str, owner: str, *, any_owner: bool = False) -> dict:
        """Display geometry of the stage-A snapshot (preview mesh, centreline polylines, endpoints).

        The worker replaces ``job["a"]`` wholesale when stage A re-runs, so the nested lists
        captured here are immutable and can be serialised outside the manager lock.  v2 records do
        not embed the preview; it is read from ``stage_a.json`` of the same stage-A run (J1).
        """
        with self.lock:
            job = self._owned(job_id, owner, any_owner=any_owner)
            version = job["version"]
            a = job.get("a") if isinstance(job.get("a"), dict) else {}
            proposal = a.get("proposal") if isinstance(a.get("proposal"), dict) else {}
            preview, polylines, endpoints = a.get("preview"), proposal.get("preview_polylines"), proposal.get("endpoints")
            created_at = a.get("created_at")
            has_a = bool(a)
        if has_a and (preview is None or polylines is None):
            stored = self._stage_a_file(job_id, created_at)
            stored_proposal = stored.get("proposal") if isinstance(stored.get("proposal"), dict) else {}
            preview = preview if preview is not None else stored.get("preview")
            polylines = polylines if polylines is not None else stored_proposal.get("preview_polylines")
        return {"job_id": job_id, "version": version, "stage_a_created_at": created_at,
                "preview": preview, "preview_polylines": polylines or [], "endpoints": endpoints or [],
                "available": preview is not None or bool(polylines)}

    def compare(self, left_id: str, right_id: str, owner: str) -> dict:
        """Compare two completed reports owned by the current session."""
        with self.lock:
            left = self._owned(left_id, owner)
            right = self._owned(right_id, owner)
            if left.get("status") != "done" or right.get("status") != "done":
                raise JobError("只有已完成任务才能比较。", 409)
            left_path = self.root / left_id / "summary.json"
            right_path = self.root / right_id / "summary.json"
            if not left_path.is_file() or not right_path.is_file():
                raise JobError("比较所需的 summary.json 不完整。", 409)
            from .comparison import compare_summary_files
            try:
                return compare_summary_files(left_path, right_path)
            except (OSError, ValueError, TypeError) as exc:
                raise JobError(f"结果比较失败：{exc}", 422)

    def list(self, owner: str, *, all_owners: bool = False) -> list[dict]:
        with self.lock:
            return [self._snapshot(j, detail=False) for j in sorted(self.jobs.values(),
                    key=lambda j: j.get("created_ts", 0), reverse=True) if all_owners or j.get("owner") == owner]

    @staticmethod
    def _matches(job: dict, *, q: str, status: str, patient_id: str, tag: str) -> bool:
        if status and job.get("status") != status:
            return False
        if patient_id and job.get("patient_id", "") != patient_id:
            return False
        tags = job.get("tags") or []
        if tag and tag not in {value.casefold() for value in tags}:
            return False
        if q:
            values = [job.get(key, "") for key in (
                "id", "case_id", "patient_id", "scan_label", "scan_date", "notes",
                "source_filename", "batch_id", "run_identity")]
            release = job.get("model_release") or {}
            values.extend((release.get("id", ""), release.get("release", ""), *tags))
            if not any(q in str(value).casefold() for value in values if value is not None):
                return False
        return True

    @classmethod
    def _query_args(cls, q, status, patient_id, tag, page, page_size) -> dict:
        q = cls._text(q, "检索词", 200).casefold()
        patient_id = cls._text(patient_id, "匿名患者编号", 80)
        tag = cls._text(tag, "标签", 40).casefold()
        status = cls._text(status, "状态", 40)
        if status and status not in JOB_STATUSES:
            raise JobError("任务状态筛选无效。")
        if type(page) is not int or page < 1:
            raise JobError("page 必须是正整数。")
        if type(page_size) is not int or not 1 <= page_size <= 100:
            raise JobError("page_size 必须是 1 到 100 的整数。")
        return {"q": q, "status": status, "patient_id": patient_id, "tag": tag, "page": page, "page_size": page_size}

    def query(self, owner: str, *, q: str = "", status: str = "", patient_id: str = "",
              tag: str = "", page: int = 1, page_size: int = 25, all_owners: bool = False) -> dict:
        """Search only this owner's records; pagination never changes list()."""
        args = self._query_args(q, status, patient_id, tag, page, page_size)
        page, page_size = args.pop("page"), args.pop("page_size")
        with self.lock:
            rows = [job for job in self.jobs.values()
                    if (all_owners or job.get("owner") == owner) and self._matches(job, **args)]
            rows.sort(key=lambda job: (job.get("created_ts", 0), job["id"]), reverse=True)
            start = (page - 1) * page_size
            return {"jobs": [self._snapshot(job, detail=False) for job in rows[start:start + page_size]],
                    "total": len(rows), "page": page, "page_size": page_size}

    def cases(self, owner: str, *, q: str = "", patient_id: str = "", tag: str = "",
              page: int = 1, page_size: int = 20, all_owners: bool = False) -> dict:
        """Case cards (contract §11.1): every run of one input geometry grouped under one card."""
        from .cases import group_cases
        args = self._query_args(q, "", patient_id, tag, page, page_size)
        page, page_size = args.pop("page"), args.pop("page_size")
        with self.lock:
            ordered = sorted((job for job in self.jobs.values() if all_owners or job.get("owner") == owner),
                             key=lambda job: (job.get("created_ts", 0), job["id"]), reverse=True)
            snapshots = [self._snapshot(job, detail=False) for job in ordered]
            matching = {job["id"] for job in self.jobs.values() if (all_owners or job.get("owner") == owner)
                        and self._matches(job, **args)}
        known = self._known_releases()
        cards = [card for card in group_cases(snapshots, known)
                 if any(run["job_id"] in matching for run in card["runs"])]
        start = (page - 1) * page_size
        return {"cases": cards[start:start + page_size], "total": len(cards), "page": page, "page_size": page_size,
                "releases": [{"id": r.get("id"), "family": family_of_release(r)} for r in known]}

    def count_owner(self, owner: str) -> int:
        with self.lock:
            return sum(1 for job in self.jobs.values() if job.get("owner") == owner)

    def queue_counts(self, owner: str | None = None) -> dict:
        """Jobs per active status: the whole service, or one owner's."""
        with self.lock:
            counts = {status: 0 for status in QUEUE_STATUSES}
            for job in self.jobs.values():
                if (owner is None or job.get("owner") == owner) and job.get("status") in counts:
                    counts[job["status"]] += 1
            return counts

    def worker_alive(self) -> bool:
        return any(worker.is_alive() for worker in (self._workers or ([self.thread] if self.thread else [])))

    def owner_of(self, job_id: str) -> str | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return job.get("owner") if job else None

    def done_job_dirs(self) -> list[Path]:
        with self.lock:
            return [self.root / job_id for job_id, job in self.jobs.items() if job.get("status") == "done"]

    def patient_timeline(self, patient_id, owner: str, *, all_owners: bool = False) -> dict:
        """Follow-up timeline of one patient (contract §19.3); owner-scoped unless the admin asks for all."""
        from .timeline import build_timeline, clean_patient_id
        try:
            patient = clean_patient_id(patient_id)
        except ValueError as exc:
            raise JobError(str(exc))
        keys = ("id", "status", "patient_id", "scan_date", "scan_label", "case_id", "created_at", "input_sha256",
                "model_release", "review")
        with self.lock:
            records = [{**{key: copy.deepcopy(job.get(key)) for key in keys},
                        "a": {"input_sha256": (job.get("a") or {}).get("input_sha256") if isinstance(job.get("a"), dict) else None}}
                       for job in self.jobs.values()
                       if (all_owners or job.get("owner") == owner) and job.get("status") == "done"
                       and str(job.get("patient_id") or "").strip() == patient]
        return build_timeline(patient, records, lambda job: self._json_file(self.root / job["id"] / "summary.json"))

    def claim_owner(self, old_owner: str | None, new_owner: str) -> dict:
        """Re-home every job of a session-scoped owner under a named user (C3 migration).

        ``old_owner=None`` moves the jobs that have no owner at all (created by ``cli run`` outside the service)."""
        if (old_owner is not None and (not isinstance(old_owner, str) or not old_owner)) or not isinstance(new_owner, str) or not new_owner:
            raise JobError("owner 与目标用户名不能为空。")
        if old_owner == new_owner:
            return {"claimed": 0, "job_ids": []}
        with self.lock:
            moved = []
            for job in self.jobs.values():
                if job.get("owner") == old_owner:
                    job["owner"] = new_owner
                    self._event(job, "owner_claimed", actor=new_owner, previous_owner=(old_owner or "")[:12], claimed_by=new_owner)
                    moved.append(job["id"])
        return {"claimed": len(moved), "job_ids": moved}

    def find_by_input(self, owner: str, sha: str) -> list[dict]:
        """Jobs of this owner that were computed from the same STL bytes, newest first (contract §11.2)."""
        with self.lock:
            rows = [job for job in self.jobs.values() if job.get("owner") == owner and self._input_sha256(job) == sha]
            rows.sort(key=lambda job: (job.get("created_ts", 0), job["id"]), reverse=True)
            return [{"job_id": job["id"], "case_id": job.get("case_id"), "status": job.get("status"),
                     "release_id": (job.get("model_release") or {}).get("id"), "family": self._family(job),
                     "created_at": job.get("created_at"), "review": (job.get("review") or {}).get("status", "unreviewed"),
                     "mapping_confirmed": bool(job.get("mapping")), "reusable": self._reusable(job)} for job in rows]

    def _enqueue(self, job: dict, stage: str, action: str, *, defer: bool = False, actor: str | None = None) -> None:
        now = time.time()
        if stage == "A":
            # Stage A rewrites the clean STL and stage_a.json: a background precompute of the old geometry must stop
            # (and no longer shortens the estimate).
            self._cancel_precompute(job["id"])
            job.pop("precompute", None)
        if job["status"] in {"awaiting_input", "awaiting_confirmation"}:
            job["confirmation_seconds"] = job.get("confirmation_seconds", 0) + max(0, now - job.get("awaiting_ts", now))
        job.update(status="queued", stage=stage, queued_ts=now, cancel_requested=False,
                   phase="等待计算", detail="任务将按顺序执行。", error=None,
                   # ``compute_seconds`` accumulates across retries; ``attempt_seconds`` is this attempt only.
                   attempt_seconds=0.0)
        job.pop("finished_ts", None)
        self._event(job, action, actor=actor)
        if not defer:
            self.tasks.put((job["id"], job["version"], stage))

    @staticmethod
    def _params(payload: dict) -> dict:
        units = payload.get("units", "mm")
        if units not in {"mm", "cm", "m", "auto"}:
            raise JobError("单位只能选择 mm、cm、m 或 auto。")
        remove = payload.get("remove_fragments", False)
        if type(remove) is not bool:
            raise JobError("remove_fragments 必须是布尔值。")
        return {"units": units, "remove_fragments": remove}

    @staticmethod
    def _compute(payload: dict) -> dict:
        """Validate per-job runtime settings without passing them to stage A."""
        device = payload.get("device", "auto")
        if device not in {"auto", "cpu", "cuda"}:
            raise JobError("计算设备只能选择 auto、cpu 或 cuda。")
        seed_count = payload.get("seed_count")
        if seed_count in (None, "", "all", "全部"):
            seed_count = None
        elif type(seed_count) is not int or not 1 <= seed_count <= 32:
            raise JobError("集成模型数必须是 1 到 32 的整数。")
        threads = payload.get("threads")
        if threads in (None, ""):
            threads = None
        elif type(threads) is not int or not 1 <= threads <= 64:
            raise JobError("CPU 线程数必须是 1 到 64 的整数。")
        return {"device": device, "seed_count": seed_count, "threads": threads}

    @staticmethod
    def _text(value, label: str, limit: int, *, multiline: bool = False) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise JobError(f"{label}必须是文本。")
        allowed = "\n\r\t" if multiline else ""
        if len(value) > limit or any((ord(char) < 32 or 127 <= ord(char) < 160) and char not in allowed for char in value):
            raise JobError(f"{label}最多 {limit} 个字符，不能含控制字符。")
        return value.strip()

    @classmethod
    def _metadata(cls, payload: dict) -> dict:
        """Keep caller-supplied anonymous identifiers separate from geometry."""
        metadata = {
            "patient_id": cls._text(payload.get("patient_id"), "匿名患者编号", 80),
            "scan_label": cls._text(payload.get("scan_label"), "扫描标签", 120),
            "scan_date": cls._text(payload.get("scan_date"), "扫描日期", 10),
            "notes": cls._text(payload.get("notes"), "备注", 2000, multiline=True),
        }
        if metadata["scan_date"]:
            try:
                date = dt.date.fromisoformat(metadata["scan_date"])
                if date.isoformat() != metadata["scan_date"]:
                    raise ValueError("non-canonical date")
            except ValueError:
                raise JobError("扫描日期必须是有效的 YYYY-MM-DD 日期。")
        tags = payload.get("tags")
        if tags is None:
            tags = []
        if not isinstance(tags, list) or len(tags) > 12:
            raise JobError("tags 必须是最多 12 项的标签列表。")
        metadata["tags"] = []
        seen = set()
        for value in tags:
            tag = cls._text(value, "标签", 40)
            if not tag:
                raise JobError("标签不能为空。")
            if tag.casefold() not in seen:
                metadata["tags"].append(tag)
                seen.add(tag.casefold())
        return metadata

    @staticmethod
    def _upload_checks(content, filename) -> str:
        import re
        if not isinstance(filename, str) or not filename.lower().endswith(".stl") or not isinstance(content, bytes) or not content:
            raise JobError("请上传非空 STL 文件。")
        return re.split(r"[/\\]", filename)[-1][:180]

    def _stage_upload(self, *, content_path=None, content_file=None) -> tuple[Path, str, int | None]:
        """Copy a streamed upload into ``<jobs_root>/.tmp`` in 1 MiB chunks, hashing (sha256) and counting faces on
        the way; returns ``(staged file, sha256, faces)``.  Never holds the whole STL in memory.

        ``content_file`` is a binary file-like object (e.g. the server's spooled temporary file); it is rewound
        first whenever it has ``seek`` (Python 3.10's SpooledTemporaryFile has no ``seekable()``).  The staged file is removed on any error, and an empty upload is refused like
        ``content=b""``."""
        from .eta import StlFaceCounter
        tmp_dir = self.root / TMP_DIR
        tmp_dir.mkdir(mode=0o700, exist_ok=True)
        staged = tmp_dir / f"upload_{secrets.token_hex(8)}.stl"
        digest, counter = hashlib.sha256(), StlFaceCounter()
        stream = open(content_path, "rb") if content_path is not None else content_file
        try:
            if content_path is None and callable(getattr(stream, "seek", None)):
                seekable = getattr(stream, "seekable", None)
                if not callable(seekable) or seekable():
                    stream.seek(0)
            with open(staged, "xb") as out:
                while True:
                    chunk = stream.read(UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise JobError("STL 文件必须以二进制方式读取。")
                    chunk = bytes(chunk)
                    digest.update(chunk)
                    counter.update(chunk)
                    out.write(chunk)
            if counter.size == 0:
                raise JobError("请上传非空 STL 文件。")
            return staged, digest.hexdigest(), counter.result()
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        finally:
            if content_path is not None:
                stream.close()

    def _duplicate_error(self, existing: list[dict], sha: str, *, reuse_unavailable: bool = False) -> JobError:
        reusable = next((row["job_id"] for row in existing if row["reusable"]), None)
        payload = {"duplicate": True, "input_sha256": sha, "existing": existing, "reusable": reusable}
        if reuse_unavailable:
            payload["reuse_unavailable"] = True
            return JobError("同一几何的已有任务没有可复用的中心线与出口确认，请选择强制重算。", 409, payload)
        return JobError("同一几何已有任务。", 409, payload)

    def create(self, owner: str, *, content: bytes | None = None, filename: str, case_id: str = "", release_id: str | None = None,
               patient_id: str = "", scan_label: str = "", scan_date: str = "", tags: list[str] | None = None,
               notes: str = "", on_duplicate: str = "force", _batch_id: str | None = None,
               content_path: str | os.PathLike | None = None, content_file=None, **params) -> dict:
        """Create a task; ``on_duplicate`` (contract §11.2) decides what happens when the same STL bytes were seen.

        ``force`` (the pre-C2 behaviour, also the default for direct callers) always creates; ``ask`` raises a
        409 carrying the existing runs; ``reuse`` clones centreline + confirmed outlets from the newest reusable
        run and only queues stage B.

        The STL comes as exactly one of ``content`` (bytes, unchanged behaviour), ``content_path`` (a file) or
        ``content_file`` (a binary file-like object, rewound first when it has ``seek``).  The last two are streamed in 1 MiB
        chunks through ``<jobs_root>/.tmp`` (sha256 while copying, then renamed into the job) — v0.14, so the
        server need not read a 128 MiB upload into memory.  All three run the same checks.
        """
        sources = sum(value is not None for value in (content, content_path, content_file))
        if sources != 1:
            raise JobError("请上传非空 STL 文件。" if sources == 0 else "STL 内容只能提供一种来源。")
        streamed = content is None
        safe_name = self._upload_checks(b"-" if streamed else content, filename)
        if on_duplicate not in {"ask", "reuse", "force"}:
            raise JobError("on_duplicate 只能是 ask、reuse 或 force。")
        case_id = self._text(case_id, "病例编号", 160)
        metadata = self._metadata({"patient_id": patient_id, "scan_label": scan_label,
                                   "scan_date": scan_date, "tags": tags, "notes": notes})
        inputs = self._params(params)
        compute = self._compute(params)
        model_release = self._release_record(release_id)
        available_models = model_release.get("models_count")
        if compute["seed_count"] is not None and isinstance(available_models, int) and compute["seed_count"] > available_models:
            raise JobError(f"集成模型数不能超过当前发布包的 {available_models} 个模型。")
        staged = faces = None
        if streamed:
            staged, sha, faces = self._stage_upload(content_path=content_path, content_file=content_file)
        else:
            sha = input_digest(content)
        try:
            if on_duplicate != "force":
                existing = self.find_by_input(owner, sha)
                if on_duplicate == "ask" and existing:
                    raise self._duplicate_error(existing, sha)
                if on_duplicate == "reuse":
                    source_id = next((row["job_id"] for row in existing if row["reusable"]), None)
                    if not source_id:
                        raise self._duplicate_error(existing, sha, reuse_unavailable=True)
                    return self._clone_for_stage_b(source_id, owner, model_release=model_release, compute=compute,
                                                   case_id=case_id.strip() or Path(safe_name).stem, metadata=metadata,
                                                   source_filename=safe_name, batch_id=_batch_id, reused=True, actor=owner)
            job_id = _clock.strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(6)
            job_dir = self.root / job_id
            job_dir.mkdir(mode=0o700)
            if streamed:
                os.replace(staged, job_dir / "input.stl")
                staged = None
            else:
                (job_dir / "input.stl").write_bytes(content)
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
        now = time.time()
        job = {"id": job_id, "owner": owner, "case_id": case_id.strip() or Path(safe_name).stem,
               "source_filename": safe_name, "filename": "input.stl", "input_sha256": sha, "status": "new", "stage": "A",
               "created_at": _date(now), "created_ts": now, "version": 0, "schema_version": JOB_SCHEMA_VERSION,
               "params": {**inputs, "inlet": None}, "compute": compute, "events": [], "review": fresh_review(), **metadata}
        if not streamed:
            from .eta import stl_faces
            faces = stl_faces(content)
        if faces:
            job["input_faces"] = faces   # §21.2: sizes the estimate before stage A has read the mesh
        if _batch_id is not None:
            job["batch_id"] = _batch_id
        job["model_release"] = model_release
        with self.lock:
            self.jobs[job_id] = job
            self._enqueue(job, "A", "created", actor=owner)
            return self._snapshot(job)

    def create_batch(self, owner: str, items: list[dict], **defaults) -> dict:
        """Create up to 20 independent tasks, preserving each item's outcome.

        Each item supplies filename plus one of content / content_path / content_file (see :meth:`create`)
        and may override the common create parameters. Invalid items do not prevent later items from being saved.
        Duplicates under ``on_duplicate="ask"`` are reported per item, not created.
        No patient metadata is written to the service log.
        """
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_BATCH_ITEMS:
            raise JobError(f"每批必须包含 1 到 {MAX_BATCH_ITEMS} 个 STL 文件。")
        allowed = {"content", "content_path", "content_file", "filename", "case_id", "release_id", "units", "remove_fragments",
                   "device", "seed_count", "threads", "on_duplicate", *CASE_METADATA}
        if set(defaults) - allowed:
            raise JobError("批量上传包含不支持的公共参数。")
        batch_id = "batch_" + _clock.strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(6)
        results, jobs, duplicates = [], [], 0
        for index, item in enumerate(items):
            try:
                if not isinstance(item, dict) or set(item) - allowed:
                    raise JobError("批量条目必须是有效的 STL 上传对象。")
                options = {**defaults, **item}
                if not any(options.get(key) is not None for key in ("content", "content_path", "content_file")) \
                        or "filename" not in options:
                    raise JobError("批量条目缺少 STL 文件内容或文件名。")
                job = self.create(owner, **options, _batch_id=batch_id)
                jobs.append(job)
                result = {"index": index, "job": job}
                if job.get("reused_from"):
                    result["reused_from"] = job["reused_from"]
                results.append(result)
            except JobError as error:
                if error.payload.get("duplicate") and not error.payload.get("reuse_unavailable"):
                    duplicates += 1
                    results.append({"index": index, **error.payload})
                else:
                    results.append({"index": index, "error": {"message": str(error), "status": error.status, **error.payload}})
            except ValueError as error:
                results.append({"index": index, "error": {"message": str(error), "status": getattr(error, "status", 422)}})
            except Exception:
                diagnostic_id = secrets.token_hex(6)
                LOG.error("Batch item could not be saved; diagnostic_id=%s", diagnostic_id)
                results.append({"index": index, "error": {
                    "message": "文件未能保存，请重试或提供诊断编号给维护者。", "status": 500,
                    "diagnostic_id": diagnostic_id}})
        return {"batch_id": batch_id, "jobs": jobs, "results": results, "created_count": len(jobs),
                "failed_count": len(items) - len(jobs) - duplicates, "duplicate_count": duplicates}

    def input(self, job_id: str, owner: str, payload: dict) -> dict:
        with self.lock:
            job = self._owned(job_id, owner)
            self._version(job, payload)
            input_recovery = (job["status"] in {"failed", "interrupted"} and
                              job.get("stage") == "A" and job.get("a") and
                              (job["a"].get("input_check") or {}).get("status") == "needs_confirmation")
            if job["status"] != "awaiting_input" and not input_recovery:
                raise JobError("当前任务不在输入确认阶段。", 409)
            if payload.get("acknowledged") is not True:
                raise JobError("请先确认尺寸、单位和表面修复选项。")
            job["params"].update(self._params(payload))
            job.pop("mapping", None)
            self._enqueue(job, "A", "input_confirmed", actor=owner)
            return self._snapshot(job)

    @staticmethod
    def _gate(a: dict | None) -> bool:
        if not a:
            return False
        ic, cl = a.get("input_check") or {}, a.get("centerline") or {}
        return ic.get("ok") is True and ic.get("status", "pass") == "pass" and cl.get("hard_pass") is True

    def _confidence_gate(self, a: dict | None, *, release=None) -> dict:
        """Evaluate the calibrated confidence contract for a Stage-A result."""
        from . import centerline as cl
        a = a or {}
        proposal = a.get("proposal") or {}
        gate = cl.evaluate_confidence_gate(
            proposal,
            orientation_source=(a.get("input_check") or {}).get("orientation_source", "unknown_stl"),
            release=self.release if release is None else release,
        )
        # Keep a copy in the immutable Stage-A snapshot.  The report and the
        # run manifest can then explain why an automatic route was or was not
        # taken, including the profile/release binding.
        proposal["confidence_gate"] = gate
        proposal["confidence_profile"] = gate.get("profile_id")
        proposal["confidence_is_calibrated"] = gate.get("calibration_status") == "validated"
        proposal["confidence_reasons"] = list(dict.fromkeys(
            list(proposal.get("confidence_reasons") or []) + list(gate.get("reasons") or [])))
        proposal["confirmation_required"] = not bool(gate.get("passed"))
        return gate

    def _auto_outlet_gate(self, a: dict | None) -> bool:
        """Return true only for a complete, explicitly high-confidence proposal.

        Legacy stage-A records do not contain the confidence fields and remain
        manual by default.  A high-confidence route still stores the proposed
        mapping in the job so the user can override it before a later rerun.
        """
        if not JobManager._gate(a):
            return False
        proposal = (a or {}).get("proposal") or {}
        confidence = proposal.get("confidence")
        gate = proposal.get("confidence_gate") or {}
        return (proposal.get("auto_ok") is True
                and proposal.get("confirmation_required") is False
                and gate.get("passed") is True
                and gate.get("calibration_status") == "validated"
                and isinstance(confidence, (int, float)) and confidence >= 0.95
                and isinstance(proposal.get("mapping"), dict)
                and len(proposal["mapping"]) == 4
                and set(proposal["mapping"].values()) == LABELS)

    @staticmethod
    def _mapping_payload(payload: dict) -> dict:
        mapping = payload.get("mapping")
        if not isinstance(mapping, dict) or len(mapping) != 4 or set(mapping.values()) != LABELS:
            raise JobError("四个出口必须分别且唯一对应左外、左内、右外、右内。")
        return {str(k): value for k, value in mapping.items()}

    def confirm(self, job_id: str, owner: str, payload: dict) -> dict:
        with self.lock:
            job = self._owned(job_id, owner)
            self._version(job, payload)
            # A high-confidence proposal can already have completed.  Keep a
            # deliberate override path so a reviewer can change its mapping
            # and rerun stage B without uploading the STL again.
            override_done = job["status"] == "done" and payload.get("override") is True
            if job["status"] != "awaiting_confirmation" and not override_done:
                raise JobError("当前任务不在出口确认阶段。", 409)
            if override_done:
                self._require_unlocked(job, "修改出口命名并重算")
            if payload.get("acknowledged") is not True:
                raise JobError("请确认出口命名及其对计算的影响。")
            if not self._gate(job.get("a")):
                raise JobError("输入或中心线未通过检查，不能继续计算。", 409)
            inlet = payload.get("inlet", job["params"].get("inlet"))
            if inlet is not None:
                available = {o.get("opening_id") for o in job["a"]["centerline"].get("openings", [])}
                if type(inlet) is not int or inlet not in available:
                    raise JobError("入口必须是当前几何中有效的开口编号。")
            if inlet != job["params"].get("inlet"):
                job["params"]["inlet"] = inlet
                job.pop("mapping", None)
                self._enqueue(job, "A", "inlet_changed", actor=owner)
                return self._snapshot(job)
            mapping = self._mapping_payload(payload)
            errors = self.mapping_validator(self.root / job_id, mapping)
            if errors:
                raise JobError("；".join(errors))
            if override_done and mapping == job.get("mapping"):
                return self._snapshot(job)
            job["mapping"] = mapping
            job.setdefault("mapping_history", []).append({"at": _date(time.time()),
                "suggested": (job["a"].get("proposal") or {}).get("mapping"), "confirmed": mapping,
                "inlet": inlet, "acknowledged": True,
                "source": "manual_override" if override_done else "manual_confirmation",
                "confidence_gate": (job["a"].get("proposal") or {}).get("confidence_gate", {})})
            if override_done:
                job.pop("summary", None)
            self._enqueue(job, "B", "outlets_confirmed", actor=owner)
            return self._snapshot(job)

    @staticmethod
    def deletable(job: dict) -> bool:
        """A job can be deleted unless a worker is executing it or its result is signed off."""
        return job.get("status") != "running" and not JobManager.locked(job)

    def delete(self, job_id: str, owner: str, payload: dict) -> dict:
        """Move a job and its directory (inputs, centreline, report, exports, history) into the trash.

        The record is detached under the lock and the directory renamed atomically, so no
        request or worker can observe a half-deleted job; the move into ``.trash`` then runs
        outside the lock.  The job stays restorable for ``TRASH_DAYS``; the provenance line in
        ``deleted_jobs.jsonl`` is written when it is finally purged.
        """
        with self.lock:
            job = self._owned(job_id, owner)
            self._version(job, payload)
            self._require_unlocked(job, "删除")
            if not self.deletable(job):
                raise JobError("任务正在计算，请先取消并等待其停止后再删除。", 409)
            record = self._detach_for_deletion(job, deleted_by=owner)
        self._finish_deletion(record)
        return {"deleted": True, "trashed": True, "id": job_id, "case_id": record["case_id"],
                "removed_bytes": record["removed_bytes"], "expires_at": record.get("expires_at")}

    def delete_many(self, owner: str, items: list) -> dict:
        """Delete up to MAX_DELETE_ITEMS jobs; each item is ``{"id", "version"}`` and fails independently."""
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_DELETE_ITEMS:
            raise JobError(f"每次最多删除 {MAX_DELETE_ITEMS} 个任务。")
        results, detached = [], []
        with self.lock:
            for item in items:
                job_id = item.get("id") if isinstance(item, dict) else None
                if not isinstance(job_id, str) or not re.fullmatch(JOB_ID_PATTERN, job_id):
                    results.append({"id": job_id, "error": {"message": "无效的任务编号。", "status": 400}})
                    continue
                try:
                    job = self._owned(job_id, owner)
                    self._version(job, item)
                    self._require_unlocked(job, "删除")
                    if not self.deletable(job):
                        raise JobError("任务正在计算，不能删除。", 409)
                    detached.append(self._detach_for_deletion(job, deleted_by=owner))
                    results.append({"id": job_id, "deleted": True, "trashed": True, "case_id": job.get("case_id"),
                                    "expires_at": detached[-1].get("expires_at")})
                except JobError as error:
                    results.append({"id": job_id, "error": {"message": str(error), "status": error.status}})
        for record in detached:
            self._finish_deletion(record)
        deleted = sum(1 for r in results if r.get("deleted"))
        return {"results": results, "deleted_count": deleted, "failed_count": len(results) - deleted}

    def _detach_for_deletion(self, job: dict, *, deleted_by: str = "") -> dict:
        """Under the lock: drop the record, close its live streams and rename its directory."""
        job_id = job["id"]
        job_dir = (self.root / job_id).resolve()
        staging = None
        if job_dir.parent == self.root and job_dir.is_dir():
            staging = self.root / f"{DELETING_PREFIX}{job_id}_{secrets.token_hex(4)}"
            os.rename(job_dir, staging)
        self.jobs.pop(job_id, None)
        self._cancel_precompute(job_id)
        self._audit(job, "deleted", actor=deleted_by or None)
        listeners = self._subscribers.pop(job_id, [])
        farewell = {"job_id": job_id, "version": job.get("version"), "status": "deleted", "stage": job.get("stage"),
                    "phase": "任务已删除", "detail": "", "action": "deleted", "at": _date(time.time()), "final": True,
                    "case_id": job.get("case_id"), "family": self._family(job),
                    "review": (job.get("review") or {}).get("status", "unreviewed")}
        self._offer(listeners, farewell)
        self._offer(self._owner_subscribers.get(job.get("owner") or ""), farewell)
        now = self.clock()
        return {"id": job_id, "case_id": job.get("case_id"), "status_before": job.get("status"),
                "created_at": job.get("created_at"), "run_identity": job.get("run_identity"),
                "model_release": (job.get("model_release") or {}).get("id"),
                "input_sha256": self._input_sha256(job), "source_job_id": job.get("source_job_id"),
                "staging": staging, "removed_bytes": 0, "deleted_by": deleted_by,
                # Light snapshot for the trash listing; the full job.json travels with the directory.
                "job_snapshot": copy.deepcopy({k: v for k, v in job.items() if k not in {"a", "summary", "events", "mapping_history", "review_history"}}),
                "deleted_at": _date(now), "expires_at": _date(now + TRASH_DAYS * 86400), "expires_ts": now + TRASH_DAYS * 86400}

    def _finish_deletion(self, record: dict) -> None:
        """Outside the lock: move the renamed directory into ``.trash`` with its ``trash.json`` sidecar."""
        staging = record.pop("staging", None)
        snapshot = record.pop("job_snapshot", None)
        trash_root = self.root / TRASH_DIR
        if staging is not None:
            try:
                record["removed_bytes"] = sum(f.stat().st_size for f in Path(staging).rglob("*") if f.is_file())
            except OSError:
                record["removed_bytes"] = 0
            trash_root.mkdir(mode=0o700, exist_ok=True)
            target = trash_root / record["id"]
            if target.exists():   # deleted, restored and deleted again: the older copy is superseded
                shutil.rmtree(target, ignore_errors=True)
            os.rename(staging, target)
            atomic_json(target / "trash.json", {"schema_version": "wss-deploy.trash/v1", "deleted_at": record["deleted_at"],
                                                "expires_at": record["expires_at"], "expires_ts": record["expires_ts"],
                                                "deleted_by": record.get("deleted_by", ""), "job_snapshot": snapshot or {},
                                                "provenance": {k: v for k, v in record.items() if k not in {"expires_ts"}}})
        LOG.info("Moved job %s (%s, %.1f MB) to the trash until %s", record["id"], record.get("status_before"),
                 record["removed_bytes"] / 1e6, record["expires_at"])

    # ------------------------------------------------------------------ trash (C4)
    def _trash_entries(self) -> list[tuple[Path, dict]]:
        entries = []
        for sidecar in sorted((self.root / TRASH_DIR).glob("*/trash.json")):
            folder = sidecar.parent
            if not re.fullmatch(JOB_ID_PATTERN, folder.name) or folder.resolve().parent != (self.root / TRASH_DIR).resolve():
                continue
            info = self._json_file(sidecar)
            if info.get("job_snapshot", {}).get("id") == folder.name:
                entries.append((folder, info))
        return entries

    def trash_list(self, owner: str, *, all_owners: bool = False) -> dict:
        now = self.clock()
        items = []
        for folder, info in self._trash_entries():
            snapshot = info.get("job_snapshot") or {}
            if not all_owners and snapshot.get("owner") != owner:
                continue
            expires_ts = float(info.get("expires_ts") or 0)
            items.append({"id": folder.name, "case_id": snapshot.get("case_id"), "status_before": snapshot.get("status"),
                          "release_id": (snapshot.get("model_release") or {}).get("id"), "family": self._family(snapshot),
                          "deleted_at": info.get("deleted_at"), "expires_at": info.get("expires_at"),
                          "days_left": max(0, int((expires_ts - now) // 86400)) if expires_ts else 0,
                          "patient_id": snapshot.get("patient_id", ""), "scan_label": snapshot.get("scan_label", "")})
        items.sort(key=lambda item: item.get("deleted_at") or "", reverse=True)
        return {"items": items}

    def _trashed(self, job_id: str, owner: str, *, any_owner: bool = False) -> tuple[Path, dict]:
        if not re.fullmatch(JOB_ID_PATTERN, job_id):
            raise JobError("无效的任务编号。")
        folder = self.root / TRASH_DIR / job_id
        info = self._json_file(folder / "trash.json") if folder.is_dir() else {}
        snapshot = info.get("job_snapshot") or {}
        if snapshot.get("id") != job_id or (not any_owner and snapshot.get("owner") != owner):
            raise JobError("回收站中没有该任务或它不属于当前会话。", 404)
        return folder, info

    def restore(self, job_id: str, owner: str, *, any_owner: bool = False) -> dict:
        """Move a trashed job back; the record keeps its pre-deletion state and version."""
        with self.lock:
            folder, info = self._trashed(job_id, owner, any_owner=any_owner)
            target = self.root / job_id
            if job_id in self.jobs or target.exists():
                raise JobError("同名任务已存在，不能恢复。", 409)
            os.rename(folder, target)
            (target / "trash.json").unlink(missing_ok=True)
            if not (target / "job.json").is_file():
                atomic_json(target / "job.json", info.get("job_snapshot") or {})
            job = self._restore_record(target / "job.json", None)
            if job.get("owner") is None:
                job["owner"] = (info.get("job_snapshot") or {}).get("owner")
            self.jobs[job_id] = job
            self._event(job, "restored", actor=owner, deleted_at=info.get("deleted_at"))
            return self._snapshot(job)

    def purge(self, job_id: str, owner: str, *, any_owner: bool = False, reason: str = "manual") -> dict:
        with self.lock:
            folder, info = self._trashed(job_id, owner, any_owner=any_owner)
            staging = self.root / f"{DELETING_PREFIX}{job_id}_{secrets.token_hex(4)}"
            os.rename(folder, staging)
        self._purge_directory(staging, info, reason)
        return {"purged": True, "id": job_id, "purge_reason": reason}

    def _purge_directory(self, staging: Path, info: dict, reason: str) -> None:
        """Outside the lock: remove the files for good and append the provenance line."""
        record = dict(info.get("provenance") or {})
        record.setdefault("id", (info.get("job_snapshot") or {}).get("id"))
        try:
            record["removed_bytes"] = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
        except OSError:
            record["removed_bytes"] = 0
        shutil.rmtree(staging, ignore_errors=True)
        if staging.exists():
            LOG.error("Job directory %s could not be fully removed; it will be purged at the next start", staging)
        record.update(deleted_at=info.get("deleted_at"), purged_at=_date(self.clock()), purge_reason=reason,
                      deleted_by=info.get("deleted_by", ""))
        self._audit({"id": record.get("id"), "status": "purged", "owner": (info.get("job_snapshot") or {}).get("owner"),
                     "model_release": (info.get("job_snapshot") or {}).get("model_release")}, "purged", reason=reason)
        try:
            # v0.15: the provenance log names cases; created 0600 whatever the umask.
            with os.fdopen(os.open(self.root / DELETED_LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        except (OSError, ValueError):
            LOG.exception("Cannot append to %s", DELETED_LOG)
        LOG.info("Purged job %s (%s, %.1f MB)", record.get("id"), reason, record["removed_bytes"] / 1e6)

    def maybe_purge_expired(self) -> list[str]:
        """Daily trash scan driven by the worker loop; the due check is cheap and only one worker runs the scan."""
        with self.lock:
            if self.clock() < self._next_trash_scan:
                return []
            self._next_trash_scan = self.clock() + TRASH_SCAN_SECONDS
        return self.purge_expired()

    def purge_expired(self) -> list[str]:
        """Remove trash entries past their expiry; runs at start-up and once a day from the worker loop."""
        now = self.clock()
        self._next_trash_scan = now + TRASH_SCAN_SECONDS
        purged = []
        for folder, info in self._trash_entries():
            if float(info.get("expires_ts") or 0) <= now:
                try:
                    self.purge(folder.name, "", any_owner=True, reason="expired")
                    purged.append(folder.name)
                except (JobError, OSError):
                    LOG.exception("Cannot purge expired trash entry %s", folder.name)
        return purged

    def cancel(self, job_id: str, owner: str, payload: dict) -> dict:
        with self.lock:
            job = self._owned(job_id, owner)
            self._version(job, payload)
            if job["status"] in FINAL or job.get("cancel_requested"):
                raise JobError("任务已经结束或正在取消。", 409)
            job["cancel_requested"] = True
            if job["status"] == "running":
                job.update(phase="正在取消", detail="当前步骤结束后停止；已完成的输入和中心线会保留。")
            else:
                now = time.time()
                if job["status"] == "queued":
                    job["queue_seconds"] = job.get("queue_seconds", 0) + max(0, now - job.get("queued_ts", now))
                if job["status"] in {"awaiting_input", "awaiting_confirmation"}:
                    job["confirmation_seconds"] = job.get("confirmation_seconds", 0) + max(0, now - job.get("awaiting_ts", now))
                job.update(status="cancelled", phase="已取消", detail="可以重试以恢复任务。", finished_ts=now)
            self._cancel_precompute(job_id)
            # ``attempt_aborted`` belongs to the worker: it is written when a *running* attempt is
            # actually torn down, so a job cancelled while queued keeps the historical event stream.
            self._event(job, "cancel_requested", actor=owner)
            return self._snapshot(job)

    def retry(self, job_id: str, owner: str, payload: dict) -> dict:
        with self.lock:
            job = self._owned(job_id, owner)
            self._version(job, payload)
            self._require_unlocked(job, "重试")
            if job["status"] not in {"failed", "cancelled", "interrupted"}:
                raise JobError("仅失败、取消或中断的任务可以重试。", 409)
            stage = "B" if job.get("stage") == "B" and job.get("mapping") and self._gate(job.get("a")) else "A"
            self._enqueue(job, stage, "retried", actor=owner)
            return self._snapshot(job)

    @staticmethod
    def _relativize_stage_a(stage_a: dict, old_dir: Path) -> dict:
        """Store job-local files by relative name so a copied stage-A snapshot never points at the source job.

        Records written before 2026-09-20 hold absolute paths; only the known file keys are rewritten,
        and only when they lie inside the source job directory.
        """
        old_dir = Path(old_dir).resolve()

        def portable(value):
            if not isinstance(value, str) or not value or not Path(value).is_absolute():
                return value
            try:
                return Path(value).resolve().relative_to(old_dir).as_posix()
            except (OSError, ValueError):
                return value

        if "stl" in stage_a:
            stage_a["stl"] = portable(stage_a.get("stl"))
        input_check = stage_a.get("input_check")
        if isinstance(input_check, dict):
            for key in ("source_stl", "clean_stl"):
                if key in input_check:
                    input_check[key] = portable(input_check.get(key))
        return stage_a

    def rerun(self, job_id: str, owner: str, payload: dict) -> dict:
        """Create a new task using an existing input/stage-A/mapping snapshot."""
        with self.lock:
            source = self._owned(job_id, owner)
            self._version(source, payload)
            if source.get("status") != "done" or not source.get("mapping") or not source.get("a"):
                raise JobError("只有已完成且确认出口的任务才能换模型重跑。", 409)
            model_release = self._release_record(payload.get("release_id"))
            return self._clone_for_stage_b(job_id, owner, model_release=model_release, actor=owner)

    def _clone_for_stage_b(self, job_id: str, owner: str, *, model_release: dict, compute: dict | None = None,
                           case_id: str | None = None, metadata: dict | None = None, source_filename: str | None = None,
                           batch_id: str | None = None, reused: bool = False, actor: str | None = None) -> dict:
        """Copy input + stage A + confirmed mapping of ``job_id`` into a new task that only runs stage B.

        Shared by ``rerun`` (same case, another release) and the C2 reuse path (a fresh upload of the same
        STL bytes; case metadata, compute settings and release come from the new upload form).
        """
        with self.lock:
            source = self._owned(job_id, owner)
            if not self._reusable(source):
                raise JobError("源任务的中心线或出口确认不可复用。", 409)
            new_id = _clock.strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(6)
            new_dir = self.root / new_id
            new_dir.mkdir(mode=0o700)
            old_dir = self.root / source["id"]
            # Copy only reusable inputs and geometry; reports/checkpoints from
            # the old run are never copied into the new result directory.
            for name in ("input.stl", "input_clean_mm.stl"):
                src = old_dir / name
                if src.is_file():
                    (new_dir / name).write_bytes(src.read_bytes())
            clean_ref = (source.get("a", {}).get("input_check") or {}).get("clean_stl")
            if clean_ref:
                src = Path(str(clean_ref))
                if not src.is_absolute():
                    src = old_dir / src
                if src.is_file() and src.resolve().parent == old_dir:
                    (new_dir / src.name).write_bytes(src.read_bytes())
            if (old_dir / "centerline").is_dir():
                shutil.copytree(old_dir / "centerline", new_dir / "centerline")
            cache_dir = self._geometry_cache_dir()
            if (old_dir / cache_dir).is_dir() and self._pc_state.get(source["id"]) != "running":
                # J2: the mesh-only stage-B intermediates are keyed by the clean-STL hash + parameters, so the new
                # run reuses them when nothing changed and recomputes otherwise.
                shutil.copytree(old_dir / cache_dir, new_dir / cache_dir)
            # J1: the record no longer embeds the display geometry; take it from the source's stage_a.json.
            full_a = self._with_display_geometry(copy.deepcopy(source["a"]),
                                                 self._stage_a_file(source["id"], source["a"].get("created_at")))
            stage_a = self._relativize_stage_a(full_a, old_dir)
            atomic_json(new_dir / "stage_a.json", stage_a)
            now = time.time()
            new_job = {"id": new_id, "owner": owner, "case_id": case_id or source.get("case_id", new_id),
                       "source_filename": source_filename or source.get("source_filename", "input.stl"), "filename": "input.stl",
                       "input_sha256": self._input_sha256(source),
                       "status": "new", "stage": "B", "created_at": _date(now), "created_ts": now,
                       "version": 0, "schema_version": JOB_SCHEMA_VERSION, "params": copy.deepcopy(source.get("params", {})),
                       "compute": copy.deepcopy(compute or source.get("compute", {"device": "auto", "seed_count": None, "threads": None})),
                       "events": [], "a": slim_stage_a(stage_a), "mapping": copy.deepcopy(source["mapping"]),
                       "mapping_history": copy.deepcopy(source.get("mapping_history", [])),
                       "model_release": model_release, "source_job_id": source["id"], "review": fresh_review()}
            if reused:
                new_job["reused_from"] = source["id"]
            # Seed subsets belong to a particular package.  A WSS package may
            # have five seeds while the volume package has three per field.
            if compute is None and model_release.get("id") != source.get("model_release", {}).get("id"):
                new_job["compute"]["seed_count"] = None
            for key in CASE_METADATA:
                default = [] if key == "tags" else ""
                new_job[key] = copy.deepcopy(metadata.get(key, default) if metadata is not None else source.get(key, default))
            if batch_id:
                new_job["batch_id"] = batch_id
            elif source.get("batch_id") and not reused:
                new_job["batch_id"] = source["batch_id"]
            self.jobs[new_id] = new_job
            automatic_source = bool((source.get("mapping_history") or []) and
                                    (source.get("mapping_history")[-1].get("source") == "automatic_high_confidence"))
            needs_review = False
            if automatic_source:
                from . import centerline as cl
                proposal = stage_a.get("proposal") or {}
                gate = cl.evaluate_confidence_gate(
                    proposal,
                    orientation_source=(stage_a.get("input_check") or {}).get("orientation_source", "unknown_stl"),
                    release=self.registry.describe(model_release.get("id")) if self.registry is not None else self.release)
                proposal["confidence_gate"] = gate
                proposal["confirmation_required"] = not bool(gate.get("passed"))
                needs_review = not bool(gate.get("passed"))
                atomic_json(new_dir / "stage_a.json", stage_a)
                new_job["a"] = slim_stage_a(stage_a)
            if needs_review:
                new_job["status"] = "awaiting_confirmation"
                new_job["stage"] = "A"
                new_job["awaiting_ts"] = now
                new_job["phase"] = "请确认出口"
                new_job["detail"] = "更换发布包后自动命名未满足当前发布包门控，请重新确认出口。"
                self._event(new_job, "rerun_created", actor=actor, source_job_id=source["id"])
                self._schedule_precompute(new_job)
            else:
                self._enqueue(new_job, "B", "rerun_created", defer=False, actor=actor)
            return self._snapshot(new_job)

    def _cancelled(self, job: dict) -> bool:
        with self.lock:
            return bool(job.get("cancel_requested") or self.stop_event.is_set())

    def _progress(self, job: dict, phase: str, detail: str = "") -> None:
        with self.lock:
            if self._cancelled(job):
                raise InterruptedError("任务已取消。")
            job.update(phase=phase, detail=detail)
            self._tick_stage(job, phase, detail)
            self._event(job, "progress", version=False, phase=phase, detail=detail)

    def _tick_stage(self, job: dict, phase: str, detail: str) -> None:
        """§21.2 stage clock: close the previous stage, start the one this progress call belongs to."""
        clock = job.get("stage_clock")
        if not isinstance(clock, dict):
            return
        from .eta import phase_stage
        key = phase_stage(self._family(job), phase, detail)
        if key is None or key == clock.get("current"):
            return
        now = time.time()
        previous = clock.get("current")
        if previous and isinstance(clock.get("current_started_ts"), (int, float)):
            clock.setdefault("done", {})[previous] = round(clock["done"].get(previous, 0.0) + max(0.0, now - clock["current_started_ts"]), 3)
        clock.update(current=key, current_started_ts=now)

    # ------------------------------------------------------------- v0.14 worker helpers (J2 / J6 / J7)
    @staticmethod
    def _error_record(error: BaseException, diagnostic_id: str) -> dict:
        """``job["error"]`` of a failed attempt (J7): typed errors keep category / retryable / admin detail;
        a plain ``ValueError`` shows its text and is not retryable; anything else is the generic retryable failure."""
        typed = classify(error)
        if typed is not None:
            return typed.to_record(diagnostic_id=diagnostic_id)
        if isinstance(error, ValueError):
            return {"message": str(error)[:800], "diagnostic_id": diagnostic_id, "category": "internal", "retryable": False}
        return {"message": GENERIC_FAILURE, "diagnostic_id": diagnostic_id, "category": "internal", "retryable": True,
                "admin_detail": f"{type(error).__name__}: {str(error)[:2000]}"}

    def _device_fallback(self, job: dict, device: str, typed) -> dict:
        """Record the switch to the CPU after a CUDA out-of-memory and free the GPU cache (J7)."""
        LOG.warning("Job %s: CUDA out of memory on %s; retrying stage B on the CPU", job.get("id"), device)
        import sys
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        record = {"from": device, "to": "cpu", "at": _date(time.time())}
        with self.lock:
            job["run_parameters"] = {**(job.get("run_parameters") or {}), "device_fallback": "cpu", "requested_device": device}
            job.update(phase="GPU 显存不足，改用 CPU 重新计算", detail="本次结果将在 CPU 上完成（数值等价，耗时更长）。")
            # Not a version bump: an open page must not get a 409 because the worker changed device.
            # ``reason`` is the user-facing text; the raw CUDA message goes under ``admin_detail``, which the HTTP
            # layer strips (at any depth) for non-admin sessions — job detail responses include ``events``.
            self._event(job, "device_fallback", version=False, durable=True, from_device=device, to_device="cpu",
                        reason=typed.user_message[:300],
                        **({"admin_detail": typed.admin_detail[:300]} if typed.admin_detail else {}))
        return record

    def _persist_run_note(self, job_dir: Path, key: str, value) -> None:
        """Write ``run_parameters[key]`` into summary.json (+ report metadata) and refresh the run manifest hashes."""
        def apply(meta: dict) -> None:
            parameters = meta.get("run_parameters") if isinstance(meta.get("run_parameters"), dict) else {}
            parameters[key] = value
            meta["run_parameters"] = parameters
        self._persist_summary(job_dir, apply, what=f"run_parameters.{key}")
        manifest_path = job_dir / "run_manifest.json"
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                outputs = list((existing.get("outputs") or {}).keys()) if isinstance(existing, dict) else []
                from .schema import write_run_manifest
                write_run_manifest(job_dir, self._json_file(job_dir / "summary.json"), outputs=outputs)
            except (OSError, ValueError, TypeError) as exc:
                LOG.warning("Run manifest of %s not refreshed with %s: %s", job_dir.name, key, exc)

    def _draining(self) -> bool:
        """True while ``service … --drain`` asks this process to start no new work (J6); checked at most once a second."""
        now = time.monotonic()
        checked, value = self._drain_cache
        if now - checked < 1.0:
            return value
        info = self._json_file(self.root / DRAIN_FILE) if (self.root / DRAIN_FILE).is_file() else {}
        value = bool(info) and info.get("pid") == os.getpid()
        self._drain_cache = (now, value)
        return value

    def _clear_stale_drain(self) -> None:
        """A drain file of another (stopped) process must never block this one."""
        path = self.root / DRAIN_FILE
        info = self._json_file(path) if path.is_file() else None
        if info is not None and info.get("pid") != os.getpid():
            path.unlink(missing_ok=True)

    def _requeue_for_drain(self, job: dict, stage: str) -> None:
        job.pop("started_ts", None)
        job.pop("stage_clock", None)
        job.update(status="queued", queued_ts=time.time(), phase="等待计算", detail="服务即将重启，任务会在重启后继续。")
        self._event(job, "drain_requeued")
        self.tasks.put((job["id"], job["version"], stage))

    # J2: background precompute of the mesh-only stage-B intermediates while the outlets are being confirmed.
    @staticmethod
    def _geometry_cache_dir() -> str:
        import sys
        module = sys.modules.get("wss_deploy.geometry_cache")
        return str(getattr(module, "CACHE_DIRNAME", None) or getattr(sys.modules.get("wss_deploy.pipeline"), "GEOMETRY_CACHE_DIR", None)
                   or GEOMETRY_CACHE_DIR)

    def _precompute_callable(self):
        if self._precompute_fn is not None:
            return self._precompute_fn
        from . import pipeline
        return getattr(pipeline, "precompute_geometry_cache", None)

    def _schedule_precompute(self, job: dict) -> None:
        """Queue a precompute for a job that just entered ``awaiting_confirmation`` (never fails the caller)."""
        if not self._precompute_enabled or self.offline or job.get("status") != "awaiting_confirmation":
            return
        try:
            with self._pc_cond:
                if job["id"] in self._pc_state:
                    return
                self._pc_state[job["id"]] = "pending"
                self._pc_cancel[job["id"]] = threading.Event()
                self._pc_queue.append(job["id"])
                if self._pc_thread is None or not self._pc_thread.is_alive():
                    self._pc_thread = threading.Thread(target=self._precompute_loop, name="wss-precompute", daemon=True)
                    self._pc_thread.start()
                self._pc_cond.notify_all()
            job["precompute"] = {"status": "pending", "queued_at": _date(time.time())}
        except Exception:  # noqa: BLE001
            LOG.exception("Cannot schedule the geometry precompute of %s", job.get("id"))

    def _cancel_precompute(self, job_id: str) -> None:
        with self._pc_cond:
            if self._pc_state.get(job_id) == "pending":
                self._pc_state[job_id] = "skipped"
            event = self._pc_cancel.get(job_id)
            if event is not None:
                event.set()
            self._pc_cond.notify_all()

    def _precompute_barrier(self, job: dict, stage: str) -> None:
        """Before a stage runs: a pending precompute of the job is skipped; a running one is cancelled (stage A,
        which rewrites the geometry) or awaited (stage B, which then reuses it; cancelled after PRECOMPUTE_WAIT_S)."""
        job_id = job["id"]
        with self._pc_cond:
            state = self._pc_state.get(job_id)
            if state == "pending":
                self._pc_state[job_id] = "skipped"
                return
            if state != "running":
                return
            if stage == "A":
                self._pc_cancel[job_id].set()
        if stage == "B":
            self._progress(job, "geometry", "等待后台几何预计算完成")
        deadline = time.monotonic() + PRECOMPUTE_WAIT_S
        with self._pc_cond:
            while self._pc_state.get(job_id) == "running":
                if time.monotonic() > deadline and job_id in self._pc_cancel:
                    self._pc_cancel[job_id].set()
                self._pc_cond.wait(0.5)

    def _precompute_loop(self) -> None:
        while True:
            with self._pc_cond:
                while not self._pc_queue and not self.stop_event.is_set():
                    self._pc_cond.wait(1.0)
                if self.stop_event.is_set():
                    for job_id in self._pc_queue:
                        self._pc_state.pop(job_id, None)
                        self._pc_cancel.pop(job_id, None)
                    self._pc_queue.clear()
                    self._pc_cond.notify_all()
                    return
                job_id = self._pc_queue[0]
                if self._pc_state.get(job_id) != "pending":
                    self._pc_queue.pop(0)
                    self._pc_state.pop(job_id, None)
                    self._pc_cancel.pop(job_id, None)
                    self._pc_cond.notify_all()
                    continue
                busy = self._stage_b_lock.locked()
                if not busy:
                    self._pc_queue.pop(0)
                    self._pc_state[job_id] = "running"
                    cancel = self._pc_cancel[job_id]
            if busy:
                time.sleep(0.5)           # lowest priority: never compete with a running stage B
                continue
            try:
                self._run_precompute(job_id, cancel)
            except Exception:  # noqa: BLE001 — a precompute must never take the service down
                LOG.exception("Geometry precompute of %s failed outside the pipeline", job_id)
            finally:
                with self._pc_cond:
                    self._pc_state.pop(job_id, None)
                    self._pc_cancel.pop(job_id, None)
                    self._pc_cond.notify_all()

    def _run_precompute(self, job_id: str, cancel: threading.Event) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.get("status") != "awaiting_confirmation" or cancel.is_set():
                return
            record = copy.deepcopy(job)
            job["precompute"] = {"status": "running", "started_at": _date(time.time())}
        started = time.perf_counter()
        status, result, detail = "done", None, None
        try:
            fn = self._precompute_callable()
            if fn is None:
                status = "unavailable"
            else:
                # The precompute needs only the family (contract) of the bound release, never its weights: the
                # verified descriptor, so the background thread cannot trigger a model load.
                release = self._release_for(record, load=False)
                result = fn(self.root / job_id, record, release=release, cancel_event=cancel)
                # The pipeline contract never raises: it reports ``ok`` / ``cancelled`` / ``skipped`` / ``error``.
                if isinstance(result, dict) and result.get("error"):
                    status, detail = "failed", str(result["error"])[:300]
                elif cancel.is_set() or (isinstance(result, dict) and result.get("cancelled")):
                    status = "cancelled"
                elif isinstance(result, dict) and result.get("skipped"):
                    status = "skipped"
                elif isinstance(result, dict) and result.get("ok") is False:
                    status = "incomplete"
        except InterruptedError:
            status = "cancelled"
        except Exception as exc:  # noqa: BLE001 — stage B simply recomputes
            status, detail = "failed", f"{type(exc).__name__}: {str(exc)[:300]}"
            LOG.warning("Geometry precompute of %s failed (stage B will recompute): %s", job_id, detail)
        seconds = round(time.perf_counter() - started, 2)
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            if status == "unavailable":
                job.pop("precompute", None)
                return
            entry = {"status": status, "seconds": seconds, "finished_at": _date(time.time())}
            if detail:
                entry["error"] = detail
            if isinstance(result, dict):
                for key, value in result.items():
                    if key == "timing_s" and isinstance(value, dict):
                        entry["timing_s"] = {k: v for k, v in value.items() if isinstance(v, (int, float))}
                    elif key == "steps" and isinstance(value, list):
                        entry["steps"] = [str(step) for step in value[:16]]
                    elif key != "error" and (isinstance(value, (str, int, float, bool)) or value is None):
                        entry.setdefault(key, value)
            job["precompute"] = entry
            self._event(job, "precompute_" + status, version=False, durable=False, seconds=seconds)

    def run_next(self, timeout: float = 0.1) -> bool:
        deferred_stage = None
        stage_lock_acquired = False
        aborted = False
        if self.clock() >= self._next_trash_scan:
            try:
                self.purge_expired()
            except Exception:
                LOG.exception("Trash expiry scan failed")
        if self._draining():
            # J6: ``--drain`` — take no new work; queued jobs stay queued and resume after the restart.
            time.sleep(min(max(float(timeout or 0.0), 0.0), 0.5))
            return False
        try:
            job_id, version, stage = self.tasks.get(timeout=timeout)
        except queue.Empty:
            return False
        try:
            with self.lock:
                job = self.jobs.get(job_id)
                if job is None or job["status"] != "queued" or job["version"] != version:
                    return True  # cancelled, superseded or deleted while queued
                now = time.time()
                job["queue_seconds"] = job.get("queue_seconds", 0) + max(0, now - job["queued_ts"])
                job.update(status="running", started_ts=now, phase="检查输入" if stage == "A" else "开始计算", detail="")
                # A stage-B attempt starts out "waiting" for the stage-B slot (set in the same critical section as
                # the status, so no reader sees a running stage-B job that is neither computing nor waiting).
                job["stage_clock"] = {"segment": stage, "attempt_started_ts": now, "current": None, "current_started_ts": None,
                                      "done": {}, **({"waiting": True} if stage == "B" else {})}
                self._event(job, "started")
            job_dir = self.root / job_id
            if stage == "B":
                # §21.2: a second worker may hold stage B; the clock says "waiting", not "computing".
                with self.lock:
                    if isinstance(job.get("stage_clock"), dict):
                        job["stage_clock"]["waiting"] = True
                self._stage_b_lock.acquire()
                stage_lock_acquired = True
                if self._draining():
                    # The drain began while this job waited for the stage-B slot: it goes back to the queue.
                    with self.lock:
                        self._requeue_for_drain(job, stage)
                    return True
                with self.lock:
                    if isinstance(job.get("stage_clock"), dict):
                        job["stage_clock"].update(waiting=False, attempt_started_ts=time.time())
            if self.stage_a_fn is None or self.stage_b_fn is None:
                from .pipeline import stage_a, stage_b
                a_fn, b_fn = self.stage_a_fn or stage_a, self.stage_b_fn or stage_b
            else:
                a_fn, b_fn = self.stage_a_fn, self.stage_b_fn
            callbacks = {"progress": lambda phase, detail="": self._progress(job, phase, detail),
                         "cancelled": lambda: self._cancelled(job)}
            try:
                # J2: stage A cancels a background precompute of this job; stage B waits for it (or skips it).
                self._precompute_barrier(job, stage)
                if stage == "A":
                    result = a_fn(job_dir / job["filename"], job_dir, **job["params"], **callbacks)
                    with self.lock:
                        # Confidence profiles are bound to the exact release
                        # selected at upload time, even if the service default
                        # changes before this queued job starts.
                        gate_release = self._release_for(job, load=False)
                        gate = self._confidence_gate(result, release=gate_release)
                        # stage_a() writes its immutable snapshot before the
                        # service knows which release is active.  Persist the
                        # release-bound gate immediately so a resumed job and
                        # its run manifest contain the same audit decision.
                        atomic_json(job_dir / "stage_a.json", result)
                        # J1: the record keeps the stage-A snapshot without its display geometry.
                        job["a"] = slim_stage_a(result)
                        if self._cancelled(job):
                            raise InterruptedError("任务已取消。")
                        input_check = result.get("input_check") or {}
                        input_status = input_check.get("status")
                        if input_status == "needs_confirmation":
                            job.update(status="awaiting_input", phase="请确认输入", detail="核对单位、尺寸和表面修复选项后继续。", awaiting_ts=time.time())
                        elif input_status == "fail":
                            errors = input_check.get("errors") or ["请检查 STL 文件、单位和开口几何。"]
                            job.update(
                                status="failed",
                                phase="输入检查失败",
                                detail="请修正 STL 后重试。",
                                finished_ts=time.time(),
                                error={"message": "输入检查未通过：" + "; ".join(str(item) for item in errors),
                                       "category": "input_geometry", "retryable": False},
                            )
                        elif input_status == "pass" and self._gate(result):
                            if self._auto_outlet_gate(result):
                                # Keep the exact proposal in the persisted job
                                # and enter stage B immediately.  It remains
                                # manually overridable through confirm(...,
                                # override=True) after completion.
                                job["mapping"] = dict(result["proposal"]["mapping"])
                                job.setdefault("mapping_history", []).append({
                                    "at": _date(time.time()),
                                    "suggested": dict(result["proposal"]["mapping"]),
                                    "confirmed": dict(result["proposal"]["mapping"]),
                                    "inlet": job["params"].get("inlet"),
                                    "acknowledged": False,
                                    "source": "automatic_high_confidence",
                                    "confidence": result["proposal"].get("confidence"),
                                    "confidence_gate": gate,
                                })
                                # The worker's final event below increments
                                # the job version.  Defer putting this queue
                                # item until after that event so it carries
                                # the final version and cannot be discarded as
                                # stale.
                                self._enqueue(job, "B", "outlets_auto_confirmed", defer=True)
                                deferred_stage = "B"
                            else:
                                reasons = "；".join(gate.get("reasons") or [])
                                job.update(status="awaiting_confirmation", phase="请确认出口", detail=(
                                    "自动命名未满足已校准的 95% 自动门控，请结合原始影像核对开口名称。"
                                    + (f" 原因：{reasons}" if reasons else "")), awaiting_ts=time.time())
                                self._schedule_precompute(job)
                        else:
                            raise ValueError("输入或中心线未通过检查，请检查 STL 和开口。")
                else:
                    if not self._gate(job.get("a")) or not job.get("mapping"):
                        raise ValueError("输入和出口尚未确认，不能继续计算。")
                    compute = job.get("compute") or {}
                    device = compute.get("device", "auto")
                    fallback = None
                    with self.lock:
                        job.pop("run_parameters", None)      # a fallback note belongs to one attempt only
                    try:
                        result = b_fn(job_dir, job["mapping"], self._release_for(job), confirmed=True,
                                      case_id=job["case_id"], device=device,
                                      seed_count=compute.get("seed_count"), threads=compute.get("threads"), **callbacks)
                    except Exception as error:  # noqa: BLE001 — only a CUDA OOM is handled here
                        typed = classify(error)
                        if not (isinstance(typed, ResourceError) and typed.retry_hint == "cpu") or device == "cpu":
                            raise
                        # J7: GPU out of memory → retry this stage B once on the CPU within the same job.
                        fallback = self._device_fallback(job, device, typed)
                        result = b_fn(job_dir, job["mapping"], self._release_for(job, device="cpu"), confirmed=True,
                                      case_id=job["case_id"], device="cpu",
                                      seed_count=compute.get("seed_count"), threads=compute.get("threads"), **callbacks)
                    with self.lock:
                        if self._cancelled(job):
                            raise InterruptedError("任务已取消。")
                        job["summary"] = {key: result.get(key) for key in (
                            "peak", "wss_field_pa", "timing_s", "device", "gpu", "release", "flags",
                            "schema_version", "model_release", "time_axis", "fields", "results", "run_manifest",
                            "surface_statistics", "reference_assessment", "case_metadata", "volume_statistics", "exports")}
                        for key in ("analysis_version", "deploy_version", "git_dirty", "geometry_cache"):
                            if result.get(key) is not None:
                                job["summary"][key] = result.get(key)
                        precompute = job.get("precompute") if isinstance(job.get("precompute"), dict) else {}
                        if precompute.get("status") == "done" and isinstance(precompute.get("seconds"), (int, float)):
                            # J2: background geometry time spent while the outlets were being confirmed.
                            job["summary"]["timing_s"] = {**(job["summary"].get("timing_s") or {}),
                                                          "precompute": round(float(precompute["seconds"]), 2)}
                        if fallback:
                            job["summary"]["device_fallback"] = fallback["to"]
                            job["summary"]["run_parameters"] = {**(result.get("run_parameters") or {}), "device_fallback": fallback["to"]}
                            self._persist_run_note(job_dir, "device_fallback", fallback["to"])
                        job["summary"]["quality"] = result.get("quality")
                        job["summary"]["audit"] = result.get("audit")
                        # §17: the cards need the conclusion and the headline diameter; the station
                        # series stay in summary.json so the job record does not grow by 150 kB a run.
                        from .morphology import digest as morphology_digest
                        job["summary"]["morphology"] = morphology_digest(result.get("morphology"))
                        job["summary"]["narrative"] = result.get("narrative")
                        # §19.2: headline cycle numbers and the first findings, so lists and detail cards
                        # need no summary.json read.
                        job["summary"]["cycle"] = cycle_record(result.get("cycle"))
                        job["summary"]["findings_top"] = findings_top(result)
                        # §21.2: the first run of a release on a device in this process carries a one-off warm-up;
                        # flag it so the estimate's inference history leaves it out.
                        warm_key = ((job.get("model_release") or {}).get("id"), result.get("device"))
                        if warm_key not in self._eta_warm:
                            self._eta_warm.add(warm_key)
                            job["summary"]["eta_cold_start"] = True
                        # §21.2: history only counts runs of the current feature program.
                        if isinstance(result.get("feature_contract"), dict):
                            job["summary"]["feature_contract"] = {key: result["feature_contract"].get(key)
                                                                  for key in ("version", "source_hash")}
                        job["run_identity"] = result.get("run_identity")
                        if job["run_identity"]:
                            job["summary"]["run_identity"] = job["run_identity"]
                        # A recomputed result has not been looked at by anyone yet.
                        job["review"] = fresh_review()
                        job["summary"]["review"] = fresh_review()
                        job.update(status="done", phase="计算完成", detail="报告和导出文件已准备好。", finished_ts=time.time())
            except InterruptedError:
                with self.lock:
                    aborted = True
                    job.update(status="interrupted" if self.stop_event.is_set() else "cancelled",
                               phase="任务已停止", detail="可以重试以恢复任务。", finished_ts=time.time())
            except Exception as error:
                diagnostic_id = secrets.token_hex(6)
                LOG.exception("Job %s failed at %s; diagnostic_id=%s", job_id, stage, diagnostic_id)
                with self.lock:
                    job.update(status="failed", phase="计算失败", detail="", finished_ts=time.time(),
                               error=self._error_record(error, diagnostic_id))
            finally:
                with self.lock:
                    elapsed = max(0, time.time() - job.pop("started_ts", time.time()))
                    job.pop("stage_clock", None)
                    job["compute_seconds"] = job.get("compute_seconds", 0) + elapsed
                    job["attempt_seconds"] = job.get("attempt_seconds", 0) + elapsed
                    if aborted:
                        # The wall clock the user actually waited for before cancelling (contract §15.5).
                        self._event(job, "attempt_aborted", version=False, attempt_s=round(job["attempt_seconds"], 1))
                    self._event(job, "finished")
                    if deferred_stage and job["status"] == "queued":
                        self.tasks.put((job["id"], job["version"], deferred_stage))
            return True
        finally:
            if stage_lock_acquired:
                self._stage_b_lock.release()
            self.tasks.task_done()

    def start(self) -> None:
        if self.offline:
            raise RuntimeError("an offline JobManager never runs workers")
        if self.thread and self.thread.is_alive():
            raise RuntimeError("worker already running")
        self.stop_event.clear()
        self._worker_failures = []
        def work():
            try:
                while not self.stop_event.is_set():
                    try:
                        self.run_next()
                        self.maybe_purge_expired()
                    except Exception:
                        LOG.exception("Worker failed outside pipeline; worker will continue")
            except BaseException as exc:  # noqa: BLE001 — O6: SystemExit / KeyboardInterrupt raised inside a library
                # The worker is not restarted (its job may be half-written); health() counts it dead, so /api/ready
                # answers 503 and ``service status`` says so.  One line with a diagnostic id goes to server.log.
                diagnostic_id = secrets.token_hex(6)
                self._worker_failures.append({"thread": threading.current_thread().name, "diagnostic_id": diagnostic_id,
                                              "error": type(exc).__name__, "at": _date(time.time())})
                LOG.critical("Worker thread %s died (%s); diagnostic_id=%s; /api/ready is now 503 — restart the service "
                             "(python -m wss_deploy.cli service restart)", threading.current_thread().name, type(exc).__name__,
                             diagnostic_id, exc_info=True)
        self._workers = [threading.Thread(target=work, name=f"wss-worker-{index + 1}", daemon=True)
                         for index in range(WORKER_COUNT)]
        for worker in self._workers:
            worker.start()
        self.thread = self._workers[0]
        self.start_maintenance()

    def close(self) -> None:
        self.stop_event.set()
        with self._pc_cond:
            for event in self._pc_cancel.values():
                event.set()
            self._pc_cond.notify_all()
        for worker in [*(self._workers or ([self.thread] if self.thread else [])), self._pc_thread, self._maintenance_thread]:
            if worker and worker.is_alive():
                worker.join(timeout=2)

    # ------------------------------------------------------------- J5 health
    _GPU_TTL_S = 30.0

    def health(self) -> dict:
        """Readiness of this process (contract: ``/api/health`` merges it, ``/api/ready`` answers 200 / 503).

        Hard checks: worker threads alive, a release loaded (or preload still running / disabled), jobs root
        writable, disk free above doctor's fail threshold, VMTK interpreter and vessel_geom present.  The GPU
        block is informational (CPU inference works) and cached for 30 s (``nvidia-smi``).
        """
        from .doctor import disk_thresholds
        from .paths import VESSEL_GEOM_DIR, VMTK_PYTHON
        DISK_FAIL_GB, DISK_WARN_GB = disk_thresholds()      # v0.15: WSS_DEPLOY_MIN_FREE_GB overrides both
        checks: dict = {}
        workers = list(self._workers or [])
        alive = sum(1 for worker in workers if worker.is_alive())
        checks["worker"] = {"ok": bool(workers) and alive == len(workers), "hard": True, "alive": alive,
                            "failures": list(getattr(self, "_worker_failures", None) or [])[-3:],
                            "expected": len(workers) or WORKER_COUNT, "started": bool(workers), "draining": self._draining(),
                            "queue": self.queue_counts(), "maintenance": bool(self._maintenance_thread and self._maintenance_thread.is_alive())}
        checks["releases"] = self._release_health()
        writable = self.root.is_dir() and os.access(self.root, os.W_OK | os.X_OK)
        checks["jobs_root"] = {"ok": bool(writable), "hard": True, "path": str(self.root)}
        try:
            free = round(shutil.disk_usage(self.root).free / 1024 ** 3, 1)
        except OSError:
            free = None
        checks["disk"] = {"ok": free is not None and free >= DISK_FAIL_GB, "hard": True, "free_gb": free,
                          "fail_below_gb": DISK_FAIL_GB, "warn": free is not None and free < DISK_WARN_GB}
        vmtk_ok = Path(VMTK_PYTHON).is_file()
        geom_ok = (Path(VESSEL_GEOM_DIR) / "vessel_geom" / "cli.py").is_file()
        checks["vmtk"] = {"ok": vmtk_ok and geom_ok, "hard": True, "python": vmtk_ok, "vessel_geom": geom_ok}
        gpu, cached_at = self._gpu_health()
        checks["gpu"] = {"ok": True, "hard": False, **gpu}
        ok = all(check.get("ok") for check in checks.values() if check.get("hard"))
        return {"ok": bool(ok), "checks": checks, "cached_at": cached_at}

    def _release_health(self) -> dict:
        registry = self.registry
        if registry is None:
            present = self.release is not None or self.stage_b_fn is not None
            return {"ok": present, "hard": True, "state": "injected" if present else "missing", "loaded": [], "known": []}
        try:
            known = [row.get("id") for row in registry.list()]
        except Exception:  # noqa: BLE001
            known = []
        cache = getattr(registry, "_cache", None) or {}
        try:
            loaded = sorted({key[0] for key in list(cache.keys()) if isinstance(key, tuple) and key})
        except Exception:  # noqa: BLE001
            loaded = []
        state_info = getattr(registry, "preload_state", None)
        preloading = any(t.name == "wss-preload" and t.is_alive() for t in threading.enumerate())
        disabled = os.environ.get("WSS_DEPLOY_PRELOAD", "1").strip().lower() in {"0", "false", "off", "no"}
        if isinstance(state_info, dict) and state_info.get("failed") and not loaded:
            state = "failed"
        elif loaded:
            state = "loaded"
        elif preloading or (isinstance(state_info, dict) and state_info.get("running")):
            state = "loading"
        elif disabled or not self._workers:
            state = "lazy"
        else:
            state = "failed"
        return {"ok": state != "failed", "hard": True, "state": state, "loaded": loaded, "known": known,
                "default": getattr(registry, "default_id", None), "device": getattr(registry, "device", None)}

    def _gpu_health(self) -> tuple[dict, str]:
        from .service import gpu_cache_time, gpu_summary
        gpu = gpu_summary()
        block = {"available": bool(gpu.get("available")), "name": gpu.get("name"), "count": gpu.get("count", 0),
                 "visible_devices": gpu.get("visible_devices"), "device": getattr(self.registry, "device", None)}
        devices = [{key: row.get(key) for key in ("index", "name", "memory_used_mb", "memory_total_mb")}
                   for row in gpu.get("devices") or []]
        if devices:
            block["devices"] = devices
        if gpu.get("note"):
            block["note"] = gpu["note"]
        return block, _date(gpu_cache_time() or time.time())

    # ------------------------------------------------------------- J6 post-upgrade maintenance
    def start_maintenance(self) -> bool:
        """Run a queued ``service upgrade`` maintenance (analysis rebuilds, stale report refresh) in the background."""
        if self.offline or not (self.root / MAINTENANCE_FILE).is_file():
            return False
        if self._maintenance_thread is not None and self._maintenance_thread.is_alive():
            return False
        self._maintenance_thread = threading.Thread(target=self._run_maintenance, name="wss-maintenance", daemon=True)
        self._maintenance_thread.start()
        return True

    def _run_maintenance(self) -> dict:
        path = self.root / MAINTENANCE_FILE
        request = self._json_file(path)
        started = time.time()
        from . import __version__
        report = {"schema_version": "wss-deploy.maintenance-result/v1", "request_id": request.get("request_id"),
                  # v0.15 (O3): when the upgrade was asked for and by which CLI version; ``service status`` shows it.
                  "requested_at": request.get("requested_at"), "requested_by": request.get("requested_by"), "version": __version__,
                  "started_at": _date(started), "pid": os.getpid(), "rebuilt": [], "rebuild_failed": [],
                  "refreshed": 0, "refresh_failed": [], "skipped": []}
        LOG.info("Post-upgrade maintenance: %d analysis rebuild(s), report refresh=%s",
                 len(request.get("rebuild") or []), bool(request.get("refresh_reports", True)))
        for job_id in list(dict.fromkeys(request.get("rebuild") or [])):
            if self.stop_event.is_set():
                break
            try:
                outcome = self.rebuild_analysis(job_id)
            except Exception as exc:  # noqa: BLE001 — one broken job must not stop the others
                LOG.exception("Post-upgrade rebuild of %s failed", job_id)
                report["rebuild_failed"].append({"id": job_id, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                continue
            (report["rebuilt"] if outcome == "rebuilt" else report["skipped"]).append(job_id if outcome == "rebuilt" else {"id": job_id, "reason": outcome})
        if request.get("refresh_reports", True) and not self.stop_event.is_set():
            from . import report_freshness as F
            try:
                stale = F.stale_jobs(self.root)
            except Exception:  # noqa: BLE001
                LOG.exception("Cannot list stale reports")
                stale = []
            for job_dir in stale:
                if self.stop_event.is_set():
                    break
                try:
                    with F._job_lock(job_dir):
                        with self.lock:
                            F.refresh(job_dir, source="upgrade")
                    report["refreshed"] += 1
                except Exception as exc:  # noqa: BLE001
                    LOG.exception("Post-upgrade report refresh of %s failed", job_dir.name)
                    report["refresh_failed"].append({"id": job_dir.name, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        report.update(finished_at=_date(time.time()), seconds=round(time.time() - started, 1),
                      complete=not self.stop_event.is_set())
        try:
            atomic_json(self.root / MAINTENANCE_RESULT, report)
            if report["complete"] and self._json_file(path).get("request_id") == request.get("request_id"):
                path.unlink(missing_ok=True)
        except OSError:
            LOG.exception("Cannot record the maintenance result")
        LOG.info("Post-upgrade maintenance finished: rebuilt %d, failed %d, reports refreshed %d, failed %d",
                 len(report["rebuilt"]), len(report["rebuild_failed"]), report["refreshed"], len(report["refresh_failed"]))
        return report

    def rebuild_analysis(self, job_id: str, *, rebuild_fn=None) -> str:
        """Full ``rebuild_report`` of one finished job inside the running service (J6).

        The job accepts no mutation meanwhile (409) and stage B is held off; afterwards the record is reloaded
        from disk (the rebuild rewrites summary.json, report.html, run_manifest.json and job.json) and a
        version-bumping event tells open pages to refresh.  Returns ``rebuilt`` or why it was skipped.
        """
        if not re.fullmatch(JOB_ID_PATTERN, str(job_id)):
            return "invalid_id"
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return "missing"
            if job.get("status") != "done":
                return "not_done"
            self._maintenance_busy.add(job_id)
        try:
            if rebuild_fn is None:
                from .rebuild_report import rebuild as rebuild_fn
            with self._stage_b_lock:
                rebuild_fn(self.root / job_id)
            with self.lock:
                fresh = self._restore_record(self.root / job_id / "job.json", None)
                if fresh.get("owner") is None:
                    fresh["owner"] = job.get("owner")
                self.jobs[job_id] = fresh
                self._event(fresh, "analysis_rebuilt")
            return "rebuilt"
        finally:
            with self.lock:
                self._maintenance_busy.discard(job_id)
