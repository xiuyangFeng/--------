"""Jobs-root disk hygiene (v0.15 O4): ``cli jobs du`` (read only) and ``cli jobs prune-cache`` (dry-run by default).

Only re-computable derivatives are ever pruned:

- ``<job>/geometry_cache/`` — content-keyed stage-B intermediates (clean mesh, resampling, point geometry,
  morphology); a missing entry is simply recomputed by the next stage B (Tier A: identical numbers);
- ``<jobs_root>/.tmp/`` — spooled uploads and bundle zips of finished requests (the service also clears files
  older than an hour when it starts).

Results, reports, inputs, ``history/`` and the trash are never touched here.  The trash empties itself: entries
older than 30 days are purged at start-up and once a day by the worker, each purge appending one line to
``<jobs_root>/deleted_jobs.jsonl`` (id, deleted/purged time, reason, bytes) plus ``Purged job …`` in ``server.log``
and a ``wss_deploy.audit`` line.  Jobs that are queued, running or waiting for input / outlet confirmation are
skipped (a background precompute may be writing their cache).  An executed prune appends one line to
``<jobs_root>/prune_cache.jsonl``.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

GEOMETRY_CACHE = "geometry_cache"
TMP_DIR = ".tmp"
TRASH_DIR = ".trash"
PRUNE_LOG = "prune_cache.jsonl"
ACTIVE_STATUSES = frozenset({"queued", "running", "awaiting_input", "awaiting_confirmation"})
LOG_NAMES = ("server.log", "access.log", "server.console.log")


def _tree(path: Path) -> dict:
    """Bytes / files / oldest and newest mtime below ``path`` (symlinks are counted, never followed)."""
    out = {"bytes": 0, "files": 0, "oldest": None, "newest": None}
    stack = [Path(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISDIR(st.st_mode):
                        stack.append(Path(entry.path))
                        continue
                    out["bytes"] += st.st_size
                    out["files"] += 1
                    out["oldest"] = st.st_mtime if out["oldest"] is None else min(out["oldest"], st.st_mtime)
                    out["newest"] = st.st_mtime if out["newest"] is None else max(out["newest"], st.st_mtime)
        except (NotADirectoryError, FileNotFoundError, PermissionError):
            if current == Path(path):
                try:
                    st = Path(path).lstat()
                except OSError:
                    return out
                if not stat.S_ISDIR(st.st_mode):
                    out.update(bytes=st.st_size, files=1, oldest=st.st_mtime, newest=st.st_mtime)
    return out


def _size(path: Path) -> int:
    try:
        st = Path(path).lstat()
    except OSError:
        return 0
    return st.st_size if stat.S_ISREG(st.st_mode) else 0


def _job_json(job_dir: Path) -> dict:
    try:
        value = json.loads((Path(job_dir) / "job.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def job_dirs(root: Path) -> list[Path]:
    """Job directories of ``root`` (``*/job.json``; dot-directories such as .tmp / .trash never count)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(path.parent for path in root.glob("*/job.json")
                  if not path.parent.name.startswith(".") and not path.parent.is_symlink())


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    from .clock import iso
    return iso(ts)


def usage(root: Path, *, now: float | None = None) -> dict:
    """Read-only size census of a jobs root: every job with its field.npz / report.html / geometry_cache / history /
    other split, plus the trash, ``.tmp``, the logs and the disk free space."""
    root = Path(root).resolve()
    now = time.time() if now is None else now
    jobs = []
    for job_dir in job_dirs(root):
        record = _job_json(job_dir)
        total = _tree(job_dir)
        cache = _tree(job_dir / GEOMETRY_CACHE)
        history = _tree(job_dir / "history")
        parts = {"field_npz": _size(job_dir / "field.npz"), "report_html": _size(job_dir / "report.html"),
                 "geometry_cache": cache["bytes"], "history": history["bytes"]}
        jobs.append({"id": job_dir.name, "case_id": record.get("case_id"), "status": record.get("status"),
                     "bytes": total["bytes"], "files": total["files"], **parts,
                     "other": max(0, total["bytes"] - sum(parts.values())),
                     "geometry_cache_files": cache["files"], "geometry_cache_oldest": _iso(cache["oldest"])})
    jobs.sort(key=lambda row: (-row["bytes"], row["id"]))
    trash_entries = []
    trash_root = root / TRASH_DIR
    if trash_root.is_dir():
        for sidecar in sorted(trash_root.glob("*/trash.json")):
            try:
                info = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError):
                info = {}
            trash_entries.append({"id": sidecar.parent.name, "bytes": _tree(sidecar.parent)["bytes"],
                                  "deleted_at": info.get("deleted_at"), "expires_at": info.get("expires_at"),
                                  "days_left": (max(0, int((float(info["expires_ts"]) - now) // 86400))
                                                if isinstance(info.get("expires_ts"), (int, float)) else None)})
    trash = _tree(trash_root)
    tmp = _tree(root / TMP_DIR)
    logs = {}
    for name in LOG_NAMES:
        files = sorted(root.glob(name + "*"))
        logs[name] = {"bytes": sum(_size(p) for p in files), "files": len(files)}
    whole = _tree(root)
    deleted_log = root / "deleted_jobs.jsonl"
    purges = 0
    if deleted_log.is_file():
        try:
            purges = sum(1 for line in deleted_log.read_text(encoding="utf-8").splitlines() if line.strip())
        except (OSError, UnicodeError):
            purges = None
    try:
        free = round(shutil.disk_usage(root).free / 1024 ** 3, 1)
    except OSError:
        free = None
    from .clock import now_iso
    return {"jobs_root": str(root), "generated_at": now_iso(), "jobs": jobs,
            "jobs_bytes": sum(row["bytes"] for row in jobs),
            "geometry_cache_bytes": sum(row["geometry_cache"] for row in jobs),
            "trash": {"bytes": trash["bytes"], "entries": trash_entries, "purge_log": str(deleted_log),
                      "purged_total": purges},
            "tmp": {"bytes": tmp["bytes"], "files": tmp["files"], "oldest": _iso(tmp["oldest"])},
            "logs": logs, "total_bytes": whole["bytes"], "disk_free_gb": free}


def human(n: int | float | None) -> str:
    if n is None:
        return "—"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _width(text: str) -> int:
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in str(text))


def _pad(text, width: int, right: bool = False) -> str:
    text = str(text)
    fill = " " * max(1, width - _width(text))
    return (fill[1:] + text + " ") if right else (text + fill)


def format_usage(info: dict, top: int | None = None) -> str:
    rows = info["jobs"][:top] if top else info["jobs"]
    lines = [_pad("任务号", 32) + _pad("病例", 16) + _pad("状态", 22) + _pad("合计", 10, True) + _pad("field.npz", 11, True)
             + _pad("report.html", 12, True) + _pad("geometry_cache", 15, True) + _pad("history", 10, True) + _pad("其他", 10, True)]
    for row in rows:
        lines.append(_pad(row["id"], 32) + _pad(str(row.get("case_id") or "")[:14], 16) + _pad(row.get("status") or "?", 22)
                     + _pad(human(row["bytes"]), 10, True) + _pad(human(row["field_npz"]), 11, True)
                     + _pad(human(row["report_html"]), 12, True) + _pad(human(row["geometry_cache"]), 15, True)
                     + _pad(human(row["history"]), 10, True) + _pad(human(row["other"]), 10, True))
    shown = f"（显示前 {len(rows)} 个）" if top and len(info["jobs"]) > len(rows) else ""
    trash = info["trash"]
    expiring = [e for e in trash["entries"] if e.get("days_left") is not None]
    soonest = min(expiring, key=lambda e: e["days_left"]) if expiring else None
    lines += ["",
              f"任务 {len(info['jobs'])} 个{shown}：合计 {human(info['jobs_bytes'])}，其中 geometry_cache {human(info['geometry_cache_bytes'])}（可重算，"
              "`jobs prune-cache` 可清）",
              f"回收站：{len(trash['entries'])} 个任务 {human(trash['bytes'])}"
              + (f"，最早 {soonest['days_left']} 天后自动清除（{soonest['id']}）" if soonest else "")
              + f"；清除记录 {trash['purge_log']}（已记 {trash['purged_total'] if trash['purged_total'] is not None else '?'} 条）",
              f".tmp：{info['tmp']['files']} 个文件 {human(info['tmp']['bytes'])}" + (f"，最老 {info['tmp']['oldest']}" if info["tmp"]["oldest"] else ""),
              "日志：" + "，".join(f"{name} {human(v['bytes'])}（{v['files']} 个文件）" for name, v in info["logs"].items() if v["files"]),
              f"任务目录总计 {human(info['total_bytes'])}；磁盘剩余 {info['disk_free_gb'] if info['disk_free_gb'] is not None else '—'} GB"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- prune-cache
def prune_plan(root: Path, older_than_days: float, *, now: float | None = None) -> dict:
    """Files that ``prune-cache`` would delete: ``geometry_cache`` entries of idle jobs and ``.tmp`` files whose
    mtime is older than ``older_than_days``.  Read only."""
    if older_than_days is None or float(older_than_days) < 0:
        raise ValueError("--older-than 需要非负的天数。")
    root = Path(root).resolve()
    now = time.time() if now is None else now
    cutoff = now - float(older_than_days) * 86400
    items, skipped = [], []
    for job_dir in job_dirs(root):
        cache = job_dir / GEOMETRY_CACHE
        if not cache.is_dir() or cache.is_symlink():
            continue
        status = _job_json(job_dir).get("status")
        if status in ACTIVE_STATUSES:
            skipped.append({"id": job_dir.name, "status": status, "bytes": _tree(cache)["bytes"]})
            continue
        for path in sorted(cache.rglob("*")):
            try:
                st = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode) and st.st_mtime < cutoff:
                items.append({"kind": "geometry_cache", "job": job_dir.name, "path": str(path.relative_to(root)),
                              "bytes": st.st_size, "age_days": round((now - st.st_mtime) / 86400, 1)})
    tmp = root / TMP_DIR
    if tmp.is_dir() and not tmp.is_symlink():
        for path in sorted(tmp.rglob("*")):
            try:
                st = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode) and st.st_mtime < cutoff:
                items.append({"kind": "tmp", "job": None, "path": str(path.relative_to(root)), "bytes": st.st_size,
                              "age_days": round((now - st.st_mtime) / 86400, 1)})
    return {"jobs_root": str(root), "older_than_days": float(older_than_days), "cutoff": _iso(cutoff), "items": items,
            "bytes": sum(item["bytes"] for item in items), "skipped_active": skipped}


def prune_execute(root: Path, plan: dict, *, by: str = "cli") -> dict:
    """Delete the planned files (each re-checked: still a regular file inside ``root`` and still older than the
    cutoff); empty ``geometry_cache`` directories are removed; one line goes to ``prune_cache.jsonl``."""
    root = Path(root).resolve()
    cutoff_ts = time.time() - float(plan["older_than_days"]) * 86400
    removed, failed, freed = 0, [], 0
    dirs = set()
    for item in plan["items"]:
        path = (root / item["path"])
        try:
            resolved_parent = path.parent.resolve()
            if root not in (resolved_parent, *resolved_parent.parents):
                raise OSError("路径超出任务目录")
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode) or st.st_mtime >= cutoff_ts:
                continue
            path.unlink()
            removed += 1
            freed += st.st_size
            if item["kind"] == "geometry_cache":
                dirs.add(path.parent)
        except FileNotFoundError:
            continue
        except OSError as exc:
            failed.append({"path": item["path"], "error": str(exc)})
    for folder in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        try:
            folder.rmdir()               # only when empty
        except OSError:
            pass
    from .clock import now_iso
    record = {"at": now_iso(), "by": by, "older_than_days": plan["older_than_days"], "files": removed,
              "bytes": freed, "failed": len(failed), "pid": os.getpid()}
    try:
        with open(root / PRUNE_LOG, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        print(f"警告：无法写入 {root / PRUNE_LOG}", file=sys.stderr)
    return {**record, "failed_items": failed}


def format_plan(plan: dict, *, limit: int = 40) -> str:
    items = plan["items"]
    lines = [f"将删除 {len(items)} 个文件，共 {human(plan['bytes'])}（早于 {plan['cutoff']}，即 {plan['older_than_days']:g} 天前）："]
    for item in items[:limit]:
        lines.append(f"  {item['path']}  {human(item['bytes'])}  {item['age_days']} 天")
    if len(items) > limit:
        lines.append(f"  … 另 {len(items) - limit} 个")
    for row in plan["skipped_active"]:
        lines.append(f"  跳过 {row['id']}（{row['status']}，geometry_cache {human(row['bytes'])}）：任务未结束")
    return "\n".join(lines)


__all__ = ["ACTIVE_STATUSES", "PRUNE_LOG", "format_plan", "format_usage", "human", "job_dirs", "prune_execute",
           "prune_plan", "usage"]
