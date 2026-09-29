"""Separate operations/support routes; the host server owns sessions and CSRF."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import heapq
import json
import logging
import os
from pathlib import Path
import re
import tempfile

from .jobs import JOB_STATUSES, JobError
from .operations import (ConflictError, JOB_ID, OperationsStore, REVIEWS, STATUSES, UNSET, _text,
                         event_fields, event_filters, event_timestamp)

HISTORY_LIMIT = 10000
ACTIVE_STATUSES = {"queued", "running", "queued_A", "queued_B", "running_A", "running_B"}
AUDIT = logging.getLogger("wss_deploy.audit")
LOG = logging.getLogger("wss_deploy.service")


def _store(handler):
    store = getattr(handler.server, "operations", None)
    if store is None:
        # Normal servers eagerly create the store; compatibility for lightweight
        # handler callers is protected by the manager's existing lock.
        with handler.server.manager.lock:
            store = getattr(handler.server, "operations", None)
            if store is None:
                store = OperationsStore(handler.server.manager.root)
                handler.server.operations = store
    return store


def _admin(handler, row):
    return bool(row and (row.get("role") == "admin" or not handler.server.sessions.shared))


def _actor(row):
    return row.get("username") or row.get("owner") or "system"


def _page(one, integer):
    try:
        page, page_size = integer("page", 1), integer("page_size", 25)
    except (ValueError, TypeError, JobError):
        raise JobError("page/page_size 必须是整数。")
    if type(page) is not int or type(page_size) is not int or not 1 <= page <= 1_000_000 or not 1 <= page_size <= 100:
        raise JobError("page 必须是 1 到 1000000，page_size 必须是 1 到 100。")
    return page, page_size


def _q(one, name, limit=200):
    try:
        return _text(one(name, ""), limit)
    except ValueError as exc:
        raise JobError(f"{name}: {exc}")


def _bad(exc):
    if isinstance(exc, ConflictError): return JobError(str(exc), 409)
    if isinstance(exc, (KeyError, FileNotFoundError)): return JobError("记录或原始文件不存在。", 404)
    return JobError(str(exc) or "请求参数无效。", 400)


def _timestamp(value):
    return event_timestamp(value) or 0.0


def _boolean_filter(one, name):
    raw = _q(one, name, 8)
    if raw not in {"", "true", "false"}: raise JobError(f"{name} 只能是 true 或 false。")
    return None if not raw else raw == "true"


def _event_args(one):
    try:
        return event_filters(**{key: _q(one, key, limit) for key, limit in (
            ("owner", 128), ("action", 80), ("q", 200), ("category", 20), ("severity", 20), ("since", 40), ("until", 40))})
    except ValueError as exc:
        raise JobError(str(exc))


def _event_sort_key(row):
    ident = row.get("id")
    stamp = event_timestamp(row.get("at"))
    return (stamp if stamp is not None else float("-inf"), isinstance(ident, int), ident if isinstance(ident, int) else 0, str(ident or ""))


def _expires_within(row, *, now, days):
    expires = event_timestamp(row.get("expires_at"))
    return expires is not None and expires <= now + days * 86400


def _active_records(manager, *, events=False):
    keys = ("id", "owner", "case_id", "source_filename", "status", "stage", "version", "created_at", "updated_at", "model_release", "error", "input_sha256")
    with manager.lock:
        records = []
        for job in manager.jobs.values():
            record = {key: copy.deepcopy(job.get(key)) for key in keys}
            a = job.get("a") if isinstance(job.get("a"), dict) else {}
            record["input_sha256"] = record.get("input_sha256") or a.get("input_sha256")
            if events: record["events"] = tuple(job.get("events") or [])
            records.append(record)
    return records


def _trash_records(manager, store):
    trash = Path(manager.root) / ".trash"
    if trash.is_symlink() or not trash.is_dir(): return
    for folder in trash.iterdir():
        if folder.is_symlink() or not JOB_ID.fullmatch(folder.name): continue
        try:
            yield store.read_trash_record(folder.name)
        except (OSError, ValueError, TypeError):
            continue


def _job_rows(manager, store, *, location="active", owner="", status="", q=""):
    if location == "active":
        records = [(job, None) for job in _active_records(manager)]
    else:
        records = ((info["job_snapshot"], info) for info in _trash_records(manager, store))
    rows = []
    for job, trash in records:
        if owner and job.get("owner") != owner: continue
        current = job.get("status") or ("deleted" if trash else "")
        if status and current != status: continue
        release = job.get("model_release") if isinstance(job.get("model_release"), dict) else {}
        values = [job.get(key, "") for key in ("id", "owner", "case_id", "source_filename", "status", "stage")]
        values.append(release.get("id") or release.get("release", ""))
        if q.casefold() not in " ".join(map(str, values)).casefold(): continue
        row = {key: job.get(key) for key in ("id", "owner", "case_id", "source_filename", "stage", "created_at", "error", "input_sha256")}
        row.update(status=current, version=job.get("version", 0), model_release=release.get("id") or release.get("release"),
                   updated_at=(trash or {}).get("deleted_at") or job.get("updated_at") or job.get("created_at"),
                   expires_at=(trash or {}).get("expires_at"))
        rows.append(row)
    rows.sort(key=lambda row: _timestamp(row.get("created_at")), reverse=True)
    return rows


def _tail_records(store):
    """At most 2 MiB / 10000 lines of the historical purge log, never read-all."""
    try:
        with store._open_relative(["deleted_jobs.jsonl"]) as stream:
            stream.seek(0, os.SEEK_END)
            start = max(0, stream.tell() - 2 * 1024 * 1024)
            stream.seek(start)
            data = stream.read(2 * 1024 * 1024)
        lines = data.splitlines()
        if start and lines: lines = lines[1:]
        for line in lines[-HISTORY_LIMIT:]:
            try: value = json.loads(line)
            except (ValueError, UnicodeError): continue
            if isinstance(value, dict): yield value
    except (OSError, ValueError):
        return


def _details(record):
    # Large historical payloads need not be mirrored in a monitoring response.
    result = {key: record[key] for key in ("stage", "version", "release", "device", "error", "reason", "purge_reason", "phase", "detail") if key in record}
    return result if len(json.dumps(result, ensure_ascii=False, default=str)) <= 4096 else {"truncated": True}


def _dynamic_events(manager, store, *, owner="", action="", q="", category="", severity="", since="", until="", page=1, page_size=25, with_meta=False):
    filters = event_filters(owner=owner, action=action, q=q, category=category, severity=severity, since=since, until=until)
    snapshot_at = datetime.fromtimestamp(store.clock(), timezone.utc).isoformat().replace("+00:00", "Z")
    persisted = store.recent_events(HISTORY_LIMIT + 1, **filters)
    since_ts = event_timestamp(filters["since"]) if filters["since"] else None
    until_ts = event_timestamp(filters["until"]) if filters["until"] else None
    # Job audit timestamps and event timestamps can straddle a second boundary.
    # Never invent a human actor for legacy state changes: older job.json files
    # deliberately omitted it, even when the owner happened to trigger them.
    signatures = {(row.get("job"), row.get("action"), int(_timestamp(row.get("at"))))
                  for row in persisted if row.get("source") == "job"}

    def already_recorded(job, action, at):
        second = int(_timestamp(at))
        return any((job, action, second + offset) in signatures for offset in (-1, 0, 1))

    def historical(job, event, source, index=""):
        at, action = event.get("at"), event.get("action")
        if already_recorded(job.get("id"), action, at): return None
        ident = hashlib.sha256(f"{source}|{job.get('id')}|{action}|{at}|{index}".encode()).hexdigest()[:16]
        return {"id": "historical-" + ident, "at": at, "action": action, "actor": event.get("actor") or "",
                "owner": job.get("owner"), "job": job.get("id"), "status": event.get("status") or job.get("status"),
                "source": source, "details": _details(event)}

    def candidates():
        yield from persisted
        for job in _active_records(manager, events=True):
            for index, event in enumerate(job["events"]):
                if isinstance(event, dict):
                    row = historical(job, event, "historical_job", index=index)
                    if row: yield row
        for info in _trash_records(manager, store):
            job = info["job_snapshot"]
            event = {"at": info.get("deleted_at"), "action": "deleted", "actor": info.get("deleted_by"), "status": "deleted"}
            row = historical(job, event, "historical_trash")
            if row: yield row
        for record in _tail_records(store):
            event = {"at": record.get("purged_at") or record.get("deleted_at"), "action": "purged", "status": "purged", "purge_reason": record.get("purge_reason")}
            # deleted_by identifies the earlier deletion, not necessarily the
            # later automatic purge, so the historical purge actor is unknown.
            row = historical({"id": record.get("id"), "owner": record.get("owner")}, event, "historical_purge")
            if row: yield row

    def matching():
        for record in candidates():
            row = event_fields(record)
            if filters["owner"] and row.get("owner") != filters["owner"]: continue
            if filters["action"] and row.get("action") != filters["action"]: continue
            if filters["category"] and row["category"] != filters["category"]: continue
            if filters["severity"] and row["severity"] != filters["severity"]: continue
            stamp = event_timestamp(row.get("at"))
            if since_ts is not None and (stamp is None or stamp < since_ts): continue
            if until_ts is not None and (stamp is None or stamp > until_ts): continue
            if filters["q"]:
                values = [str(row.get(key) or "") for key in ("action", "actor", "owner", "job", "status", "source", "at", "category", "severity")]
                values.append(json.dumps(row.get("details") or {}, ensure_ascii=False))
                if filters["q"].casefold() not in " ".join(values).casefold(): continue
            yield row

    rows = heapq.nlargest(HISTORY_LIMIT + 1, matching(), key=_event_sort_key)
    truncated = len(rows) > HISTORY_LIMIT
    rows = rows[:HISTORY_LIMIT]
    items = rows[(page - 1) * page_size:page * page_size]
    if with_meta:
        return {"items": items, "total": len(rows), "page": page, "page_size": page_size,
                "history_window_limit": HISTORY_LIMIT, "truncated": truncated, "filters": filters,
                "snapshot_at": snapshot_at, "latest_id": rows[0]["id"] if rows else None}
    return items, len(rows)



def _respond_page(handler, items, total, page, page_size):
    handler._json({"items": items, "total": total, "page": page, "page_size": page_size})
    return True


def handle_get(handler, row, path, one, integer):
    is_ops = path == "/api/ops" or path.startswith("/api/ops/")
    is_support = path == "/api/support/tickets" or path.startswith("/api/support/tickets/")
    if not (is_ops or is_support): return False
    if not row or not row.get("owner"): raise JobError("请先登录。", 401)
    if is_ops and not _admin(handler, row): raise JobError("只有管理员或本机运维员可以访问该页面。", 403)
    store, manager = _store(handler), handler.server.manager
    page, page_size = _page(one, integer)
    if path == "/api/ops/overview":
        jobs, trash = _job_rows(manager, store), _job_rows(manager, store, location="trash")
        users_store = getattr(handler.server.sessions, "users", None)
        users = users_store.cached() if users_store is not None else {}
        counts = {"jobs": len(jobs), "users": len(users), "active": sum(row["status"] in ACTIVE_STATUSES for row in jobs),
                  "failed": sum(row["status"] == "failed" for row in jobs), "trash": len(trash),
                  "trash_expiring": sum(_expires_within(item, now=store.clock(), days=7) for item in trash),
                  **store.todo_counts()}
        warnings = [{"code": code, "count": counts[key], "severity": severity} for key, code, severity in (
            ("failed", "failed_jobs", "error"), ("trash_expiring", "trash_expiring", "warning"),
            ("unassigned_tickets", "unassigned_tickets", "warning"), ("pending_review_assets", "pending_review_assets", "info"),
        ) if counts[key]]
        recent, _ = _dynamic_events(manager, store, page_size=20)
        handler._json({"counts": counts, "health": handler.server.health(row), "recent_events": recent, "warnings": warnings})
        return True
    if path == "/api/ops/jobs":
        location, status = _q(one, "location", 16) or "active", _q(one, "status", 64)
        if location not in {"active", "trash"}: raise JobError("location 只能是 active 或 trash。")
        if status and status not in JOB_STATUSES and not (location == "trash" and status == "deleted"): raise JobError("状态筛选无效。")
        rows = _job_rows(manager, store, location=location, owner=_q(one, "owner", 128), status=status, q=_q(one, "q"))
        days_raw = _q(one, "expires_within_days", 3)
        if days_raw:
            if location != "trash" or not days_raw.isdecimal() or not 1 <= int(days_raw) <= 365:
                raise JobError("expires_within_days 仅用于回收站，必须是 1 到 365。")
            now = store.clock()
            rows = [item for item in rows if _expires_within(item, now=now, days=int(days_raw))]
        return _respond_page(handler, rows[(page - 1) * page_size:page * page_size], len(rows), page, page_size)
    if path == "/api/ops/users":
        query, disabled = _q(one, "q").casefold(), _boolean_filter(one, "disabled")
        users_store = getattr(handler.server.sessions, "users", None)
        users = users_store.cached() if users_store is not None else {}
        counts = {}
        for job in _job_rows(manager, store):
            totals = counts.setdefault(job["owner"], {"jobs": 0, "failed": 0, "active": 0})
            totals["jobs"] += 1; totals["failed"] += job["status"] == "failed"; totals["active"] += job["status"] in ACTIVE_STATUSES
        items = [dict(users_store.public(name, value), **counts.get(name, {"jobs": 0, "failed": 0, "active": 0}))
                 for name, value in sorted(users.items()) if isinstance(value, dict)]
        if query: items = [item for item in items if query in (item["username"] + " " + item["display_name"]).casefold()]
        if disabled is not None: items = [item for item in items if item["disabled"] == disabled]
        # All accounts are normally small; pagination keeps larger installations
        # bounded while preserving the familiar items field.
        return _respond_page(handler, items[(page - 1) * page_size:page * page_size], len(items), page, page_size)
    if path in {"/api/ops/events", "/api/ops/events/export"}:
        filters = _event_args(one)
        if path.endswith("/export"):
            slots = handler.server.bundle_slots
            if not slots.acquire(blocking=False):
                raise JobError("正在生成其他下载，请稍后重试。", 429, {"retry_after": 2})
            try:
                feed = _dynamic_events(manager, store, **filters, page=1, page_size=HISTORY_LIMIT, with_meta=True)
                document = {"schema_version": "wss-deploy.events-export/v1", "exported_at": feed["snapshot_at"],
                            "snapshot_at": feed["snapshot_at"], "latest_id": feed["latest_id"],
                            "exported_count": len(feed["items"]), "truncated": feed["truncated"],
                            "history_window_limit": HISTORY_LIMIT, "filters": feed["filters"],
                            "filter_bounds": {"since": "inclusive", "until": "inclusive", "timezone": "UTC"},
                            "events": feed["items"]}
                # Stream JSON encoding to disk rather than constructing another
                # large in-memory string for up to 10,000 detailed events.
                with tempfile.TemporaryFile(dir=handler.server.temp_dir()) as export:
                    for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False).iterencode(document):
                        export.write(chunk.encode("utf-8"))
                    export.seek(0)
                    handler._send_open(export, None, "application/json; charset=utf-8", download="wss-operations-events.json")
                # Intentionally no export event: exporting must not mutate the
                # selected population or displace its oldest matching record.
                return True
            finally:
                slots.release()
        handler._json(_dynamic_events(manager, store, **filters, page=page, page_size=page_size, with_meta=True))
        return True
    if path in {"/api/ops/tickets", "/api/support/tickets"}:
        status = _q(one, "status", 30)
        if status and status not in STATUSES and not (is_ops and status == "unresolved"): raise JobError("工单状态无效。")
        owner = _q(one, "owner", 128) if is_ops else row["owner"]
        unassigned = _boolean_filter(one, "unassigned") if is_ops else None
        items, total = store.tickets(owner=owner, status=status, q=_q(one, "q"), page=page, page_size=page_size, unassigned=unassigned)
        return _respond_page(handler, items, total, page, page_size)
    if path == "/api/ops/assets":
        status = _q(one, "status", 30)
        if status and status not in REVIEWS: raise JobError("资产审阅状态无效。")
        items, total = store.assets(status=status, q=_q(one, "q"), page=page, page_size=page_size)
        return _respond_page(handler, items, total, page, page_size)
    match = re.fullmatch(r"/api/ops/assets/([0-9a-f]{64})/download", path)
    if match:
        # Share the server's existing ZIP budget.  Multiple browser downloads
        # must not bypass the limits applied to normal job bundle generation.
        slots = handler.server.bundle_slots
        if not slots.acquire(blocking=False):
            raise JobError("正在生成其他打包下载，请稍后重试。", 429, {"retry_after": 2})
        try:
            asset = store.asset(match[1])
            if asset is None: raise JobError("素材不存在。", 404)
            if asset["bytes"] > handler.server.max_bundle_bytes:
                raise JobError("素材超过当前服务的打包下载上限。", 413)
            # An unnamed temporary file is removed by the OS even if the
            # process dies or a client stalls halfway through the transfer.
            with tempfile.TemporaryFile(dir=handler.server.temp_dir()) as archive:
                try: store.write_asset_zip(match[1], archive)
                except (ValueError, KeyError, FileNotFoundError) as exc: raise _bad(exc)
                if os.fstat(archive.fileno()).st_size > handler.server.max_bundle_bytes:
                    raise JobError("导出包超过当前服务的打包下载上限。", 413)
                store.record_event({"action": "asset_downloaded", "actor": _actor(row), "source": "asset", "details": {"sha256": match[1]}})
                archive.seek(0)
                handler._send_open(archive, None, "application/zip", download=match[1] + ".zip")
                return True
        finally:
            slots.release()
    match = re.fullmatch(r"/api/support/tickets/([0-9a-f]{16})", path)
    if match:
        ticket = store.ticket(match[1])
        if ticket is None or ticket["owner"] != row["owner"]: raise JobError("工单不存在。", 404)
        handler._json({"ticket": ticket})
        return True
    return False


def _archive_job(manager, store, payload, *, audit_actor=None):
    job_id, location = payload.get("job_id"), payload.get("location", "active")
    try:
        store._source_parts(job_id, location)  # before constructing any path
        with manager.lock:
            if location == "active":
                source = manager.jobs.get(job_id)
                if source is None: raise KeyError("任务不存在。")
                job = {key: copy.deepcopy(source.get(key)) for key in ("id", "owner", "case_id", "model_release", "params", "mapping", "run_identity", "input_sha256")}
                a = source.get("a") if isinstance(source.get("a"), dict) else {}
                job["input_sha256"] = job.get("input_sha256") or a.get("input_sha256")
                job["input_check"] = copy.deepcopy(a.get("input_check")) if isinstance(a.get("input_check"), dict) else {}
            else:
                job = store.read_trash_record(job_id)["job_snapshot"]
            release = job.get("model_release") if isinstance(job.get("model_release"), dict) else {}
            asset = store.archive_asset(job_id=job_id, owner=job.get("owner") or "", case_id=job.get("case_id") or "",
                                        release_id=release.get("id"), location=location, note=payload.get("note", ""),
                                        source_job=job, expected_sha256=job.get("input_sha256"), audit_actor=audit_actor)
        return asset, job
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise _bad(exc)


def handle_post(handler, row, path):
    is_ops = path == "/api/ops" or path.startswith("/api/ops/")
    is_support = path == "/api/support/tickets" or path.startswith("/api/support/tickets/")
    if not (is_ops or is_support): return False
    if not row or not row.get("owner"): raise JobError("请先登录。", 401)
    if is_ops and not _admin(handler, row): raise JobError("只有管理员或本机运维员可以执行运维操作。", 403)
    store, manager = _store(handler), handler.server.manager
    payload = handler._payload()
    match = re.fullmatch(r"/api/ops/tickets/([0-9a-f]{16})", path)
    if match:
        try:
            ticket = store.update_ticket(match[1], version=payload.get("version"), status=payload.get("status"), priority=payload.get("priority"),
                                         assignee=payload.get("assignee", UNSET), message=payload.get("message", UNSET), actor=_actor(row),
                                         audit_actor=_actor(row))
        except (ValueError, KeyError) as exc: raise _bad(exc)
        handler._json({"ticket": ticket})
        return True
    if path == "/api/ops/assets":
        asset, _ = _archive_job(manager, store, payload, audit_actor=_actor(row))
        handler._json({"asset": asset}, 201)
        return True
    match = re.fullmatch(r"/api/ops/assets/([0-9a-f]{64})", path)
    if match:
        try: asset = store.review_asset(match[1], review_status=payload.get("review_status"), note=payload.get("note"), version=payload.get("version"), audit_actor=_actor(row))
        except (ValueError, KeyError) as exc: raise _bad(exc)
        handler._json({"asset": asset})
        return True
    match = re.fullmatch(r"/api/ops/users/([A-Za-z0-9_.-]{1,64})", path)
    if match:
        users, username = getattr(handler.server.sessions, "users", None), match[1]
        if users is None: raise JobError("用户管理不可用。", 503)
        disabled = payload.get("disabled")
        if type(disabled) is not bool: raise JobError("disabled 必须是布尔值。")
        if not isinstance(payload.get("reason", ""), str): raise JobError("原因必须是字符串。")
        try: reason = _text(payload.get("reason", ""), 1000)
        except ValueError as exc: raise _bad(exc)
        if disabled and username == row.get("username"): raise JobError("不能停用自己。", 409)
        with users.lock:
            data = users.load()
            if username not in data or not isinstance(data[username], dict): raise JobError("用户不存在。", 404)
            admins = users._enabled_admins(data)
            if disabled and username in admins and len(admins) == 1: raise JobError("不能停用唯一启用的管理员。", 409)
            if bool(data[username].get("disabled")) != disabled:
                data[username]["disabled"] = disabled
                users._bump(data[username]); users._save(data)
            public = users.public(username, data[username])
        audit = {"action": "user_disabled" if disabled else "user_enabled", "actor": _actor(row), "owner": username,
                 "source": "user", "details": {"reason": reason}}
        # users.json and SQLite cannot share a transaction.  Keep the durable
        # service-log fallback and report the account result truthfully even if
        # the independent operations journal is temporarily unavailable.
        AUDIT.info("http %s", json.dumps(audit, ensure_ascii=False, separators=(",", ":")))
        result = {"user": public}
        try:
            store.record_event(audit)
        except Exception:
            LOG.exception("Account state saved, but operations audit failed: action=%s user=%s actor=%s",
                          audit["action"], username, audit["actor"])
            result["warning"] = {"code": "audit_unavailable", "message": "账号状态已保存，但运维审计库写入失败；详情已记录到服务日志。"}
        handler._json(result)
        return True
    if path == "/api/support/tickets":
        job_id = payload.get("job_id")
        if job_id == "": job_id = None
        if job_id is not None:
            if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id): raise JobError("任务编号无效。")
            manager.get(job_id, row["owner"], any_owner=False)
        try: ticket = store.create_ticket(owner=row["owner"], title=payload.get("title"), description=payload.get("description"), job_id=job_id, priority=payload.get("priority", "normal"), audit_actor=_actor(row))
        except ValueError as exc: raise _bad(exc)
        handler._json({"ticket": ticket}, 201)
        return True
    match = re.fullmatch(r"/api/support/tickets/([0-9a-f]{16})", path)
    if match:
        ticket = store.ticket(match[1])
        if ticket is None or ticket["owner"] != row["owner"]: raise JobError("工单不存在。", 404)
        if not isinstance(payload.get("message"), str) or not payload["message"].strip(): raise JobError("请填写回复内容。")
        try: ticket = store.update_ticket(match[1], version=payload.get("version"), message=payload["message"], actor=_actor(row),
                                         audit_actor=_actor(row), audit_action="ticket_message")
        except (ValueError, KeyError) as exc: raise _bad(exc)
        handler._json({"ticket": ticket})
        return True
    return False
