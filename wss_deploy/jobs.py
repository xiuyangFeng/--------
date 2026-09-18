"""Persistent, owner-scoped jobs and the single inference worker.

The saved job is the source of truth. Queue entries include a version, so a
cancelled/retried entry cannot later start another copy of the same work.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import os
from pathlib import Path
import queue
import secrets
import threading
import time

LOG = logging.getLogger("wss_deploy.service")
JOB_ID_PATTERN = r"[A-Za-z0-9_-]{1,80}"
LABELS = {"out-le", "out-li", "out-re", "out-ri"}
FINAL = {"done", "failed", "cancelled", "interrupted"}


class JobError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def atomic_json(path: Path, value: dict) -> None:
    """Replace a complete JSON document; never expose a partially written job."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as stream:
            os.chmod(tmp, 0o600)
            json.dump(value, stream, ensure_ascii=False, indent=1, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        tmp.unlink(missing_ok=True)


def _date(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def _validate_mapping(job_dir: Path, mapping: dict) -> list[str]:
    from . import centerline as cl
    return cl.validate_mapping(cl.load_vessel_geom_atlas(job_dir / "centerline"), mapping)


class JobManager:
    def __init__(self, root: Path, *, release=None, stage_a_fn=None, stage_b_fn=None,
                 mapping_validator=None, legacy_owner: str | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.release = release
        self.stage_a_fn, self.stage_b_fn = stage_a_fn, stage_b_fn
        self.mapping_validator = mapping_validator or _validate_mapping
        self.jobs: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.tasks: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._load(legacy_owner)

    def _save(self, job: dict) -> None:
        atomic_json(self.root / job["id"] / "job.json", job)

    def _event(self, job: dict, action: str, *, version: bool = True, **data) -> None:
        now = time.time()
        job["updated_at"] = _date(now)
        if version:
            job["version"] = job.get("version", 0) + 1
        job.setdefault("events", []).append({"at": _date(now), "action": action,
                                             "status": job["status"], "stage": job.get("stage"), **data})
        self._save(job)

    def _load(self, legacy_owner: str | None) -> None:
        for path in sorted(self.root.glob("*/job.json")):
            try:
                if path.resolve().parent.parent != self.root:
                    continue
                job = json.loads(path.read_text(encoding="utf-8"))
                if job.get("id") != path.parent.name:
                    raise ValueError("job directory/id mismatch")
                job.setdefault("version", 1)
                job.setdefault("params", {"units": "mm", "remove_fragments": False, "inlet": None})
                job.setdefault("created_ts", path.stat().st_mtime)
                job.setdefault("owner", legacy_owner)
                if "stage_a" in job and "a" not in job:
                    job["a"] = job.pop("stage_a")
                old = job.get("status", "failed")
                job.setdefault("stage", "B" if old.endswith("_B") else "A")
                if old == "awaiting_outlets":
                    job["status"] = "awaiting_confirmation"
                if old in {"queued", "running", "queued_A", "queued_B", "running_A", "running_B"}:
                    job["status"] = "interrupted"
                    job["phase"] = "服务重启，任务已中断"
                    job["detail"] = "请重试以恢复计算；已完成的输入和中心线可以复用。"
                    job["cancel_requested"] = False
                    job.pop("started_ts", None)
                    job["finished_ts"] = time.time()
                    self._event(job, "restart_interrupted")
                self.jobs[job["id"]] = job
            except Exception:
                LOG.exception("Cannot restore job record %s", path)

    def _owned(self, job_id: str, owner: str) -> dict:
        job = self.jobs.get(job_id)
        if not job or job.get("owner") != owner:
            raise JobError("任务不存在或不属于当前会话。", 404)
        return job

    def _version(self, job: dict, payload: dict) -> None:
        if type(payload.get("version")) is not int or payload["version"] != job["version"]:
            raise JobError("任务状态已更新，请刷新后再操作。", 409)
        if "stage" in payload and payload["stage"] != job.get("stage"):
            raise JobError("任务阶段已改变，请刷新后再操作。", 409)

    def _snapshot(self, job: dict, *, detail: bool = True) -> dict:
        now = time.time()
        compute = job.get("compute_seconds", 0.0)
        if job["status"] == "running":
            compute += max(0, now - job.get("started_ts", now))
        queued = job.get("queue_seconds", 0.0)
        if job["status"] == "queued":
            queued += max(0, now - job.get("queued_ts", now))
        confirmation = job.get("confirmation_seconds", 0.0)
        if job["status"] in {"awaiting_input", "awaiting_confirmation"}:
            confirmation += max(0, now - job.get("awaiting_ts", now))
        public = {key: val for key, val in job.items() if key not in {
            "owner", "filename", "traceback", "started_ts", "created_ts", "queued_ts", "awaiting_ts"}}
        if not detail:
            public = {key: public.get(key) for key in (
                "id", "case_id", "status", "stage", "version", "created_at", "updated_at", "phase", "detail", "error")}
        elif "a" in public:
            # Keep the old name during the UI transition; both point to the
            # same immutable stage-A snapshot in the response.
            public.setdefault("stage_a", public["a"])
        public["elapsed"] = round(compute, 1)
        public["timing"] = {"compute_s": round(compute, 1), "queue_s": round(queued, 1),
                            "confirmation_s": round(confirmation, 1),
                            "wall_s": round(max(0, job.get("finished_ts", now) - job.get("created_ts", now)), 1)}
        # Long names are kept for the browser and older report consumers.
        public["timing"].update(queue_seconds=round(queued, 1),
                                confirmation_seconds=round(confirmation, 1),
                                compute_seconds=round(compute, 1))
        waiting = sorted((j for j in self.jobs.values() if j["status"] == "queued"),
                         key=lambda j: j.get("queued_ts", 0))
        public["queue_position"] = next((i + 1 for i, j in enumerate(waiting) if j["id"] == job["id"]), None)
        return copy.deepcopy(public)

    def get(self, job_id: str, owner: str) -> dict:
        with self.lock:
            return self._snapshot(self._owned(job_id, owner))

    def list(self, owner: str) -> list[dict]:
        with self.lock:
            return [self._snapshot(j, detail=False) for j in sorted(self.jobs.values(),
                    key=lambda j: j.get("created_ts", 0), reverse=True) if j.get("owner") == owner]

    def _enqueue(self, job: dict, stage: str, action: str) -> None:
        now = time.time()
        if job["status"] in {"awaiting_input", "awaiting_confirmation"}:
            job["confirmation_seconds"] = job.get("confirmation_seconds", 0) + max(0, now - job.get("awaiting_ts", now))
        job.update(status="queued", stage=stage, queued_ts=now, cancel_requested=False,
                   phase="等待计算", detail="任务将按顺序执行。", error=None)
        job.pop("finished_ts", None)
        self._event(job, action)
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

    def create(self, owner: str, *, content: bytes, filename: str, case_id: str = "", **params) -> dict:
        import re
        if not filename.lower().endswith(".stl") or not content:
            raise JobError("请上传非空 STL 文件。")
        if len(case_id) > 160 or any(ord(c) < 32 for c in case_id):
            raise JobError("病例编号最多 160 个字符，不能含控制字符。")
        inputs = self._params(params)
        job_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(6)
        safe_name = re.split(r"[/\\]", filename)[-1][:180]
        job_dir = self.root / job_id
        job_dir.mkdir(mode=0o700)
        (job_dir / "input.stl").write_bytes(content)
        now = time.time()
        job = {"id": job_id, "owner": owner, "case_id": case_id.strip() or Path(safe_name).stem,
               "source_filename": safe_name, "filename": "input.stl", "status": "new", "stage": "A",
               "created_at": _date(now), "created_ts": now, "version": 0,
               "params": {**inputs, "inlet": None}, "events": []}
        with self.lock:
            self.jobs[job_id] = job
            self._enqueue(job, "A", "created")
            return self._snapshot(job)

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
            self._enqueue(job, "A", "input_confirmed")
            return self._snapshot(job)

    @staticmethod
    def _gate(a: dict | None) -> bool:
        if not a:
            return False
        ic, cl = a.get("input_check") or {}, a.get("centerline") or {}
        return ic.get("ok") is True and ic.get("status", "pass") == "pass" and cl.get("hard_pass") is True

    def confirm(self, job_id: str, owner: str, payload: dict) -> dict:
        with self.lock:
            job = self._owned(job_id, owner)
            self._version(job, payload)
            if job["status"] != "awaiting_confirmation":
                raise JobError("当前任务不在出口确认阶段。", 409)
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
                self._enqueue(job, "A", "inlet_changed")
                return self._snapshot(job)
            mapping = payload.get("mapping")
            if not isinstance(mapping, dict) or len(mapping) != 4 or set(mapping.values()) != LABELS:
                raise JobError("四个出口必须分别且唯一对应左外、左内、右外、右内。")
            mapping = {str(k): value for k, value in mapping.items()}
            errors = self.mapping_validator(self.root / job_id, mapping)
            if errors:
                raise JobError("；".join(errors))
            job["mapping"] = mapping
            job.setdefault("mapping_history", []).append({"at": _date(time.time()),
                "suggested": (job["a"].get("proposal") or {}).get("mapping"), "confirmed": mapping,
                "inlet": inlet, "acknowledged": True})
            self._enqueue(job, "B", "outlets_confirmed")
            return self._snapshot(job)

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
            self._event(job, "cancel_requested")
            return self._snapshot(job)

    def retry(self, job_id: str, owner: str, payload: dict) -> dict:
        with self.lock:
            job = self._owned(job_id, owner)
            self._version(job, payload)
            if job["status"] not in {"failed", "cancelled", "interrupted"}:
                raise JobError("仅失败、取消或中断的任务可以重试。", 409)
            stage = "B" if job.get("stage") == "B" and job.get("mapping") and self._gate(job.get("a")) else "A"
            self._enqueue(job, stage, "retried")
            return self._snapshot(job)

    def _cancelled(self, job: dict) -> bool:
        with self.lock:
            return bool(job.get("cancel_requested") or self.stop_event.is_set())

    def _progress(self, job: dict, phase: str, detail: str = "") -> None:
        with self.lock:
            if self._cancelled(job):
                raise InterruptedError("任务已取消。")
            job.update(phase=phase, detail=detail)
            self._event(job, "progress", version=False, phase=phase, detail=detail)

    def run_next(self, timeout: float = 0.1) -> bool:
        try:
            job_id, version, stage = self.tasks.get(timeout=timeout)
        except queue.Empty:
            return False
        try:
            with self.lock:
                job = self.jobs[job_id]
                if job["status"] != "queued" or job["version"] != version:
                    return True
                now = time.time()
                job["queue_seconds"] = job.get("queue_seconds", 0) + max(0, now - job["queued_ts"])
                job.update(status="running", started_ts=now, phase="检查输入" if stage == "A" else "开始计算", detail="")
                self._event(job, "started")
            job_dir = self.root / job_id
            if self.stage_a_fn is None or self.stage_b_fn is None:
                from .pipeline import stage_a, stage_b
                a_fn, b_fn = self.stage_a_fn or stage_a, self.stage_b_fn or stage_b
            else:
                a_fn, b_fn = self.stage_a_fn, self.stage_b_fn
            callbacks = {"progress": lambda phase, detail="": self._progress(job, phase, detail),
                         "cancelled": lambda: self._cancelled(job)}
            try:
                if stage == "A":
                    result = a_fn(job_dir / job["filename"], job_dir, **job["params"], **callbacks)
                    with self.lock:
                        job["a"] = result
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
                                error={"message": "输入检查未通过：" + "; ".join(str(item) for item in errors)},
                            )
                        elif input_status == "pass" and self._gate(result):
                            job.update(status="awaiting_confirmation", phase="请确认出口", detail="请结合原始影像核对开口名称。", awaiting_ts=time.time())
                        else:
                            raise ValueError("输入或中心线未通过检查，请检查 STL 和开口。")
                else:
                    if not self._gate(job.get("a")) or not job.get("mapping"):
                        raise ValueError("输入和出口尚未确认，不能继续计算。")
                    result = b_fn(job_dir, job["mapping"], self.release, confirmed=True, case_id=job["case_id"], **callbacks)
                    with self.lock:
                        if self._cancelled(job):
                            raise InterruptedError("任务已取消。")
                        job["summary"] = {key: result.get(key) for key in (
                            "peak", "wss_field_pa", "timing_s", "device", "gpu", "release", "flags")}
                        job.update(status="done", phase="计算完成", detail="报告和导出文件已准备好。", finished_ts=time.time())
            except InterruptedError:
                with self.lock:
                    job.update(status="interrupted" if self.stop_event.is_set() else "cancelled",
                               phase="任务已停止", detail="可以重试以恢复任务。", finished_ts=time.time())
            except Exception as error:
                diagnostic_id = secrets.token_hex(6)
                LOG.exception("Job %s failed at %s; diagnostic_id=%s", job_id, stage, diagnostic_id)
                with self.lock:
                    message = str(error)[:800] if isinstance(error, ValueError) else "计算未完成，请重试；如仍失败，请向维护者提供诊断编号。"
                    job.update(status="failed", phase="计算失败", detail="", finished_ts=time.time(),
                               error={"message": message, "diagnostic_id": diagnostic_id})
            finally:
                with self.lock:
                    job["compute_seconds"] = job.get("compute_seconds", 0) + max(0, time.time() - job.pop("started_ts", time.time()))
                    self._event(job, "finished")
            return True
        finally:
            self.tasks.task_done()

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            raise RuntimeError("worker already running")
        self.stop_event.clear()
        def work():
            while not self.stop_event.is_set():
                try:
                    self.run_next()
                except Exception:
                    LOG.exception("Worker failed outside pipeline; worker will continue")
        self.thread = threading.Thread(target=work, name="wss-single-worker", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
