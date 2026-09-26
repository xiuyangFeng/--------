"""Report template freshness (v0.12, contract ``ANALYSIS_CONTRACT.md`` §19.7).

A finished job's ``report.html`` embeds the viewer code of the release that wrote it.  When the report
templates change, the old pages keep the old viewer until someone runs ``rebuild_report --ui-only``.
This module fingerprints the presentation sources and refreshes a stale report on demand (the service's
report route) or in bulk (``python -m wss_deploy.cli reports refresh``).  The refresh itself is
:func:`wss_deploy.rebuild_report.refresh_ui_only`: embedded metadata and arrays stay byte-identical, only
the surrounding template and inlined viewer are replaced.

``<job>/report_ui.json`` records ``{"fingerprint", "refreshed_at", "source": "auto|cli"}``.

v0.14 (S8): the on-demand refresh renders into a staging copy inside the job directory *without* the job
manager's lock; the lock is taken only to check that the report and the template fingerprint did not change
meanwhile and to swap the new page in atomically (otherwise the staged page is discarded and re-checked).
"""
from __future__ import annotations

import contextlib
from . import clock as _clock
import hashlib
import json
import logging
import os
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

LOG = logging.getLogger("wss_deploy.service")
PACKAGE_DIR = Path(__file__).resolve().parent
UI_SOURCES = ("report.py", "volume_report.py", "static/report_common.js", "static/volume_viewer.js",
              "static/three.min.js", "static/OrbitControls.js", "static/glossary.json")
MARKER = "report_ui.json"
SCHEMA = "wss-deploy.report-ui/v1"
ENV_SWITCH = "WSS_DEPLOY_AUTO_REFRESH_REPORTS"
STAGE_PREFIX = ".report_ui_stage_"
STAGE_MAX_AGE_SECONDS = 3600
MAX_STAGE_ATTEMPTS = 3

_cache_lock = threading.Lock()
_cache: dict = {"stamp": None, "fingerprint": None}
_job_locks_guard = threading.Lock()
_job_locks: dict[str, threading.Lock] = {}
# job directory -> fingerprint whose refresh failed; not retried on every GET until the templates change.
_failed: dict[str, str] = {}


def _stamp(root: Path) -> tuple:
    out = []
    for name in UI_SOURCES:
        path = root / name
        try:
            st = path.stat()
            out.append((name, st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((name, None, None))
    return tuple(out)


def ui_fingerprint(root: Path | None = None) -> str:
    """sha256 over the presentation sources (name + bytes); cached until a source's mtime or size changes."""
    root = Path(root or PACKAGE_DIR)
    stamp = (str(root), _stamp(root))
    with _cache_lock:
        if _cache["stamp"] == stamp and _cache["fingerprint"]:
            return _cache["fingerprint"]
    digest = hashlib.sha256()
    for name in UI_SOURCES:
        digest.update(name.encode("utf-8") + b"\0")
        try:
            digest.update((root / name).read_bytes())
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    value = digest.hexdigest()
    with _cache_lock:
        _cache.update(stamp=stamp, fingerprint=value)
    return value


# Python modules that render reports.  The service imports them once; a later edit on disk is invisible to the
# running process, so an in-process refresh would pair the old template with the new viewer script and then
# mark the page current.  ``snapshot_loaded_code`` (called at service start) pins what is loaded.
PY_TEMPLATES = ("report.py", "volume_report.py", "rebuild_report.py")
_code_snapshot: dict = {}
_code_warned: set = set()


def _py_state(root: Path | None = None) -> dict:
    root = Path(root or PACKAGE_DIR)
    out = {}
    for name in PY_TEMPLATES:
        try:
            st = (root / name).stat()
            out[name] = (st.st_mtime_ns, st.st_size, hashlib.sha256((root / name).read_bytes()).hexdigest())
        except OSError:
            out[name] = (None, None, None)
    return out


def snapshot_loaded_code() -> dict:
    """Import the report template modules now and record their sources (service start)."""
    import importlib
    for module in ("report", "volume_report", "rebuild_report"):
        importlib.import_module(f"{__package__}.{module}")
    _code_snapshot.clear()
    _code_snapshot.update(_py_state())
    _code_warned.clear()
    return {name: value[2] for name, value in _code_snapshot.items()}


def code_changed() -> bool:
    """True when a report template module on disk differs from the one this process loaded at start."""
    if not _code_snapshot:
        return False
    for name, (mtime, size, digest) in _code_snapshot.items():
        try:
            st = (PACKAGE_DIR / name).stat()
        except OSError:
            return True
        if (st.st_mtime_ns, st.st_size) != (mtime, size) and hashlib.sha256((PACKAGE_DIR / name).read_bytes()).hexdigest() != digest:
            return True
    return False


def auto_enabled() -> bool:
    """``WSS_DEPLOY_AUTO_REFRESH_REPORTS=0`` switches the on-demand refresh off (default on)."""
    return os.environ.get(ENV_SWITCH, "1").strip().lower() not in {"0", "false", "off", "no"}


def read_marker(job_dir: Path) -> dict:
    try:
        value = json.loads((Path(job_dir) / MARKER).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def is_stale(job_dir: Path, fingerprint: str | None = None) -> bool:
    """True when the job has a report whose recorded template fingerprint differs from the current one."""
    job_dir = Path(job_dir)
    if not (job_dir / "report.html").is_file():
        return False
    return read_marker(job_dir).get("fingerprint") != (fingerprint or ui_fingerprint())


def _job_lock(job_dir: Path) -> threading.Lock:
    key = str(Path(job_dir).resolve())
    with _job_locks_guard:
        lock = _job_locks.get(key)
        if lock is None:
            lock = _job_locks[key] = threading.Lock()
        return lock


def write_marker(job_dir: Path, fingerprint: str, source: str) -> dict:
    from .jobs import atomic_json
    record = {"schema_version": SCHEMA, "fingerprint": fingerprint,
              "refreshed_at": _clock.now_iso(), "source": source}
    atomic_json(Path(job_dir) / MARKER, record)
    return record


def refresh(job_dir: Path, *, source: str = "cli", refresher: Callable | None = None) -> dict:
    """Refresh one report now (no staleness check) and write the marker; errors propagate."""
    job_dir = Path(job_dir)
    fingerprint = ui_fingerprint()
    if refresher is None:
        from .rebuild_report import refresh_ui_only as refresher
    result = refresher(job_dir)
    write_marker(job_dir, fingerprint, source)
    _failed.pop(str(job_dir.resolve()), None)
    return {"job": job_dir.name, "fingerprint": fingerprint, **(result if isinstance(result, dict) else {})}


def _report_state(path: Path) -> tuple | None:
    """Identity of the report bytes: (inode, size, mtime, sha256); None when absent."""
    try:
        st = path.stat()
        from .io_utils import file_sha256
        return st.st_ino, st.st_size, st.st_mtime_ns, file_sha256(path)
    except OSError:
        return None


def _refresh_manifest(job_dir: Path) -> None:
    """Update only ``run_manifest.outputs['report.html']`` (size + sha256) after a swap; nothing else changes."""
    from .io_utils import atomic_json, file_sha256
    path = Path(job_dir) / "run_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") if isinstance(manifest, dict) else None
    record = outputs.get("report.html") if isinstance(outputs, dict) else None
    if isinstance(record, dict):
        record["path"] = "report.html"
        record["present"] = True
        record["size_bytes"] = int((Path(job_dir) / "report.html").stat().st_size)
        record["sha256"] = file_sha256(Path(job_dir) / "report.html")
        atomic_json(path, manifest)


def _clear_stale_stages(job_dir: Path) -> None:
    cutoff = time.time() - STAGE_MAX_AGE_SECONDS
    for stage in Path(job_dir).glob(STAGE_PREFIX + "*"):
        try:
            if stage.is_dir() and stage.stat().st_mtime < cutoff:
                shutil.rmtree(stage, ignore_errors=True)
        except OSError:
            pass


def _has_embedded_arrays(report: Path) -> bool:
    try:
        text = report.read_bytes()
    except OSError:
        return False
    return b'id="wss-report-arrays"' in text or b'id="volume-arrays"' in text


def _staged_refresh(job_dir: Path, fingerprint: str, *, guard, source: str, refresher: Callable | None) -> str:
    """Render outside ``guard`` into a staging copy; under ``guard`` verify nothing moved, then swap.

    Returns ``refreshed``, ``fresh`` (someone else made it current) or ``changed`` (the report or the template
    changed while rendering; the staged page was discarded).  Rendering errors propagate.
    """
    report = job_dir / "report.html"
    if refresher is None:
        # A page without embedded arrays can never be refreshed (refresh_ui_only raises the same error); say so
        # without importing the renderer, whose first import (torch) costs seconds in a request thread.
        if not _has_embedded_arrays(report):
            raise ValueError("报告缺少嵌入数据：wss-report-arrays / volume-arrays")
        from .rebuild_report import refresh_ui_only as refresher
    before = _report_state(report)
    if before is None:
        return "fresh"
    # The staging directory carries the job's name so a refresher sees the same job name it always saw.
    root = job_dir / (STAGE_PREFIX + secrets.token_hex(6))
    stage = root / job_dir.name
    try:
        stage.mkdir(parents=True, mode=0o700)
        shutil.copy2(report, stage / "report.html")
        if _report_state(stage / "report.html")[3] != before[3]:
            return "changed"
        refresher(stage)
        staged = stage / "report.html"
        with (guard if guard is not None else contextlib.nullcontext()):
            if ui_fingerprint() != fingerprint or _report_state(report) != before:
                return "changed"
            if not staged.is_file():
                raise ValueError("UI 刷新没有生成报告")
            os.replace(staged, report)
            _refresh_manifest(job_dir)
            write_marker(job_dir, fingerprint, source)
        _failed.pop(str(job_dir.resolve()), None)
        return "refreshed"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def ensure_fresh(job_dir: Path, *, guard=None, source: str = "auto", refresher: Callable | None = None) -> str:
    """Refresh a stale report at most once per template version; never raises.

    Returns ``fresh`` (nothing to do), ``refreshed``, ``failed`` (the original report stays and the error
    is logged), ``disabled`` (environment switch), ``absent`` (no report.html) or ``code_changed`` (the
    template modules on disk changed after the service started: only ``service upgrade`` can render them).  Two concurrent requests
    for the same job wait on one per-job lock: the second finds the marker current and does nothing.
    ``guard`` (the job manager's lock in the service) is held only around the final check-and-swap: the page is
    rendered into a staging copy first, and when a concurrent review or metadata edit rewrote the report (or the
    templates changed) meanwhile, the staged page is discarded and the check starts again, so no edit is lost
    and the manager is never blocked for the seconds a render takes.
    """
    job_dir = Path(job_dir)
    if not auto_enabled():
        return "disabled"
    if not (job_dir / "report.html").is_file():
        return "absent"
    try:
        fingerprint = ui_fingerprint()
        if not is_stale(job_dir, fingerprint):
            return "fresh"
        if source == "auto" and code_changed():
            if fingerprint not in _code_warned:
                _code_warned.add(fingerprint)
                LOG.warning("报告模板的 Python 代码在服务启动后已更新，运行中的进程无法按新模板刷新；继续提供原报告。"
                            "请执行 python -m wss_deploy.cli service upgrade。")
            return "code_changed"
        key = str(job_dir.resolve())
        if _failed.get(key) == fingerprint:
            return "failed"
        with _job_lock(job_dir):
            _clear_stale_stages(job_dir)
            for _ in range(MAX_STAGE_ATTEMPTS):
                if not is_stale(job_dir, fingerprint):
                    return "fresh"
                try:
                    outcome = _staged_refresh(job_dir, fingerprint, guard=guard, source=source, refresher=refresher)
                except Exception:
                    _failed[key] = fingerprint
                    LOG.exception("报告模板自动刷新失败，继续提供原报告：%s", job_dir.name)
                    return "failed"
                if outcome == "refreshed":
                    LOG.info("报告模板已自动刷新：%s", job_dir.name)
                    return "refreshed"
                if outcome == "fresh":
                    return "fresh"
                fingerprint = ui_fingerprint()        # "changed": re-check against what is on disk now
                if source == "auto" and code_changed():
                    return "code_changed"
            LOG.warning("报告在自动刷新期间反复被修改，本次继续提供原报告：%s", job_dir.name)
            return "failed"
    except Exception:
        LOG.exception("报告模板新鲜度检查失败：%s", job_dir.name)
        return "failed"


def _job_status(job_dir: Path) -> str | None:
    try:
        value = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        return value.get("status") if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def done_jobs(jobs_root: Path) -> list[Path]:
    """Finished job directories with a report (read-only scan of ``*/job.json``)."""
    root = Path(jobs_root)
    out = []
    for path in sorted(root.glob("*/job.json")):
        job_dir = path.parent
        if job_dir.name.startswith(".") or not (job_dir / "report.html").is_file():
            continue
        if _job_status(job_dir) == "done":
            out.append(job_dir)
    return out


def stale_jobs(jobs_root: Path, job_dirs: Iterable[Path] | None = None) -> list[Path]:
    fingerprint = ui_fingerprint()
    candidates = done_jobs(jobs_root) if job_dirs is None else [Path(p) for p in job_dirs]
    return [job_dir for job_dir in candidates if is_stale(job_dir, fingerprint)]


__all__ = ["ENV_SWITCH", "MARKER", "PY_TEMPLATES", "UI_SOURCES", "auto_enabled", "code_changed", "snapshot_loaded_code", "done_jobs", "ensure_fresh", "is_stale", "read_marker",
           "refresh", "stale_jobs", "ui_fingerprint", "write_marker"]
