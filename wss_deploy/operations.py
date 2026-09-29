"""Durable operations records and explicitly archived original STL inputs.

Uploads are never archived automatically.  The SQLite catalogue and private STL
copies live outside the job trash, so later trash purges do not remove assets.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import tempfile
import threading
import time
import zipfile

HEX64 = re.compile(r"[0-9a-f]{64}")
TICKET_ID = re.compile(r"[0-9a-f]{16}")
JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,80}")
STATUSES = {"open", "in_progress", "resolved", "closed"}
PRIORITIES = {"normal", "high", "urgent"}
REVIEWS = {"candidate", "approved", "excluded"}
MAX_MESSAGES = 100
MAX_SOURCES = 100
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
TEMP_MAX_AGE_SECONDS = 24 * 3600
UNSET = object()
LOG = logging.getLogger("wss_deploy.service")
EVENT_CATEGORIES = {"job", "auth", "ticket", "asset", "user"}
EVENT_SEVERITIES = {"info", "warning", "error"}
ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})")


def event_timestamp(value):
    """Normalise existing UTC, offset and legacy naive event timestamps."""
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None: stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


def event_category(action, source):
    action, source = str(action or ""), str(source or "")
    if action in {"login", "logout", "login_failed", "login_throttled"} or source == "auth": return "auth"
    if action.startswith("ticket_") or source in {"ticket", "support"}: return "ticket"
    if action.startswith("asset_") or source == "asset": return "asset"
    if action.startswith("user_") or action == "password_changed" or source == "user": return "user"
    return "job"


def event_severity(action, status):
    action, status = str(action or "").lower(), str(status or "").lower()
    # Job deletion audit rows retain the job's former status.  Deleting an
    # already-failed job is a warning action, not a new computation failure.
    if action in {
        "login_failed", "login_throttled", "delete", "deleted", "purge", "purged", "cancel", "cancelled", "cancel_requested",
        "interrupted", "restart_interrupted", "attempt_aborted", "user_disabled", "user_disable", "user_role", "user_role_changed",
        "review_reopened", "device_fallback",
    }: return "warning"
    if status in {"failed", "error"} or "failed" in action or "failure" in action or action.endswith("_error"):
        return "error"
    if status.isdecimal():
        if int(status) >= 500: return "error"
        if int(status) >= 400: return "warning"
    if status in {"cancelled", "interrupted", "excluded", "warning"}: return "warning"
    return "info"


def event_fields(record):
    return {**record, "category": event_category(record.get("action"), record.get("source")),
            "severity": event_severity(record.get("action"), record.get("status"))}


def event_filters(*, owner="", action="", q="", category="", severity="", since="", until=""):
    filters = {"owner": _text(owner, 128), "action": _text(action, 80), "q": _text(q, 200),
               "category": _text(category, 20), "severity": _text(severity, 20)}
    if filters["category"]: _choice(filters["category"], EVENT_CATEGORIES, "事件类别")
    if filters["severity"]: _choice(filters["severity"], EVENT_SEVERITIES, "事件级别")
    for key, raw in (("since", since), ("until", until)):
        value = _text(raw, 40)
        if value:
            if not ISO_TIME.fullmatch(value): raise ValueError(f"{key} 必须是含时区的 ISO 时间。")
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            except (ValueError, OverflowError):
                raise ValueError(f"{key} 时间无效。")
        filters[key] = value
    if filters["since"] and filters["until"] and event_timestamp(filters["since"]) > event_timestamp(filters["until"]):
        raise ValueError("since 不能晚于 until。")
    return filters


class ConflictError(ValueError):
    """The caller's version or expected input digest is no longer current."""


def _text(value, limit: int, *, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("文本字段必须是字符串。")
    value = value.strip()
    if len(value) > limit or "\0" in value:
        raise ValueError(f"文本字段不能超过 {limit} 字符或包含空字符。")
    return value


def _choice(value, allowed, label):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{label} 无效。")
    return value


def _pagination(page, page_size):
    if type(page) is not int or not 1 <= page <= 1_000_000:
        raise ValueError("page 必须是 1 到 1000000 的整数。")
    if type(page_size) is not int or not 1 <= page_size <= 100:
        raise ValueError("page_size 必须是 1 到 100 的整数。")


def _like(value):
    return "%" + value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


class OperationsStore:
    def __init__(self, root: str | os.PathLike, *, clock=None):
        self.root = Path(root).resolve()
        self.directory = self.root / ".operations"
        self.assets_dir = self.directory / "stl"
        self.clock = clock or time.time
        self.lock = threading.RLock()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for path in (self.directory, self.assets_dir):
            if path.is_symlink():
                raise ValueError("运维存储目录不能是符号链接。")
            path.mkdir(mode=0o700, exist_ok=True)
            os.chmod(path, 0o700)
        self.db_path = self.directory / "operations.sqlite3"
        # Create privately before sqlite opens it; O_NOFOLLOW also rejects an
        # existing symlink, including a dangling one.
        fd = os.open(self.db_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("运维数据库必须是普通文件。")
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        with self._db(write=True) as db:
            # A transaction plus a schema inspection makes this migration safe
            # when two server processes start against the same job root.
            for sql in (
                "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, action TEXT NOT NULL, actor TEXT, owner TEXT, job TEXT, status TEXT, source TEXT, details TEXT NOT NULL DEFAULT '{}')",
                "CREATE INDEX IF NOT EXISTS events_at ON events(at DESC)",
                "CREATE INDEX IF NOT EXISTS events_owner ON events(owner)",
                "CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL, job_id TEXT, status TEXT NOT NULL, priority TEXT NOT NULL, assignee TEXT, version INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, messages TEXT NOT NULL)",
                "CREATE INDEX IF NOT EXISTS tickets_updated ON tickets(updated_at DESC)",
                "CREATE TABLE IF NOT EXISTS assets (sha256 TEXT PRIMARY KEY, bytes INTEGER NOT NULL, created_at TEXT NOT NULL, review_status TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', version INTEGER NOT NULL DEFAULT 0)",
                "CREATE TABLE IF NOT EXISTS asset_sources (sha256 TEXT NOT NULL, job_id TEXT NOT NULL, owner TEXT NOT NULL, case_id TEXT, release_id TEXT, archived_at TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}', PRIMARY KEY (sha256,job_id), FOREIGN KEY(sha256) REFERENCES assets(sha256))",
            ):
                db.execute(sql)
            if "version" not in {row["name"] for row in db.execute("PRAGMA table_info(assets)")}:
                db.execute("ALTER TABLE assets ADD COLUMN version INTEGER NOT NULL DEFAULT 0")
            if "metadata" not in {row["name"] for row in db.execute("PRAGMA table_info(asset_sources)")}:
                db.execute("ALTER TABLE asset_sources ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
        self._clean_stale_temps()

    def _clean_stale_temps(self):
        """Recover disk after interrupted copies, never touch published assets.

        A 24-hour grace period also protects another process's active copy.
        The old named ZIP pattern covers upgrades from the first operations
        implementation; current downloads use OS-managed temporary files.
        """
        cutoff = self.clock() - TEMP_MAX_AGE_SECONDS
        for directory, pattern in ((self.assets_dir, ".stage-*"), (self.directory, "asset-*.zip")):
            try:
                for path in directory.glob(pattern):
                    try:
                        info = path.lstat()
                        if stat.S_ISREG(info.st_mode) and info.st_mtime < cutoff:
                            path.unlink()
                    except FileNotFoundError:
                        pass  # A concurrent successful copy already renamed it.
                    except OSError:
                        LOG.warning("Cannot remove stale operations temporary file %s", path)
            except OSError:
                LOG.warning("Cannot inspect operations temporary files in %s", directory)

    @contextmanager
    def _db(self, *, write=False):
        """Always close the connection; writers hold one SQLite transaction."""
        if self.directory.is_symlink() or self.db_path.is_symlink():
            raise ValueError("运维数据库路径不能是符号链接。")
        db = sqlite3.connect(self.db_path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.create_function("ops_category", 2, event_category, deterministic=True)
        db.create_function("ops_severity", 2, event_severity, deterministic=True)
        db.create_function("ops_timestamp", 1, event_timestamp, deterministic=True)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _now(self):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock()))

    @staticmethod
    def _json(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _event_row(row):
        value = dict(row)
        value["details"] = json.loads(value.get("details") or "{}")
        return event_fields(value)

    def record_event(self, record: dict) -> dict:
        with self.lock, self._db(write=True) as db:
            return self._record_event_db(db, record)

    def _record_event_db(self, db, record: dict) -> dict:
        """Append in the caller's transaction so a mutation and its audit agree."""
        if not isinstance(record, dict):
            raise ValueError("event 必须是对象。")
        action = _text(record.get("action"), 80)
        if not action:
            raise ValueError("event action 不能为空。")
        details = self._json(record.get("details") if isinstance(record.get("details"), dict) else {})
        if len(details.encode("utf-8")) > 16 * 1024:
            details = self._json({"truncated": True})
        values = (_text(record.get("at"), 64) or self._now(), action,
                  _text(record.get("actor"), 128), _text(record.get("owner"), 128),
                  _text(record.get("job"), 128), _text(str(record.get("status")), 64) if record.get("status") is not None else "",
                  _text(record.get("source"), 80), details)
        cur = db.execute("INSERT INTO events(at,action,actor,owner,job,status,source,details) VALUES(?,?,?,?,?,?,?,?)", values)
        return self._event_row(db.execute("SELECT * FROM events WHERE id=?", (cur.lastrowid,)).fetchone())

    @staticmethod
    def _event_query(**filters):
        filters = event_filters(**filters)
        where, args = [], []
        for column in ("owner", "action"):
            if filters[column]:
                where.append(column + "=?"); args.append(filters[column])
        if filters["q"]:
            columns = ("action", "actor", "owner", "job", "details", "status", "source", "at",
                       "ops_category(action,source)", "ops_severity(action,status)")
            where.append("(" + " OR ".join(column + " LIKE ? ESCAPE '\\'" for column in columns) + ")")
            args.extend([_like(filters["q"])] * len(columns))
        for name, expression in (("category", "ops_category(action,source)"), ("severity", "ops_severity(action,status)")):
            if filters[name]: where.append(expression + "=?"); args.append(filters[name])
        for name, comparison in (("since", ">="), ("until", "<=")):
            if filters[name]: where.append("ops_timestamp(at) " + comparison + " ?"); args.append(event_timestamp(filters[name]))
        return (" WHERE " + " AND ".join(where) if where else ""), args

    def events(self, *, owner="", action="", q="", category="", severity="", since="", until="", page=1, page_size=25):
        _pagination(page, page_size)
        clause, args = self._event_query(owner=owner, action=action, q=q, category=category, severity=severity, since=since, until=until)
        with self._db() as db:
            total = db.execute("SELECT COUNT(*) FROM events" + clause, args).fetchone()[0]
            rows = db.execute("SELECT * FROM events" + clause + " ORDER BY ops_timestamp(at) DESC,id DESC LIMIT ? OFFSET ?", args + [page_size, (page - 1) * page_size]).fetchall()
        return [self._event_row(row) for row in rows], total

    def recent_events(self, limit=20, *, owner="", action="", q="", category="", severity="", since="", until=""):
        """All filters precede the window; one extra row detects true truncation."""
        if type(limit) is not int or not 1 <= limit <= 10001:
            raise ValueError("事件读取上限必须是 1 到 10001。")
        clause, args = self._event_query(owner=owner, action=action, q=q, category=category, severity=severity, since=since, until=until)
        with self._db() as db:
            rows = db.execute("SELECT * FROM events" + clause + " ORDER BY ops_timestamp(at) DESC,id DESC LIMIT ?", args + [limit]).fetchall()
        return [self._event_row(row) for row in rows]

    def todo_counts(self):
        with self._db() as db:
            tickets = db.execute("SELECT COUNT(*) AS tickets_open, COALESCE(SUM(TRIM(COALESCE(assignee,''))=''),0) AS unassigned_tickets FROM tickets WHERE status IN ('open','in_progress')").fetchone()
            assets = db.execute("SELECT COUNT(*) AS assets, COALESCE(SUM(review_status='candidate'),0) AS pending_review_assets FROM assets").fetchone()
        return {**dict(tickets), **dict(assets)}

    @staticmethod
    def _ticket_row(row):
        if row is None:
            return None
        value = dict(row)
        value["messages"] = json.loads(value["messages"])
        return value

    def create_ticket(self, *, owner, title, description, job_id=None, priority="normal", audit_actor=None):
        title, description, owner = _text(title, 200), _text(description, 4000), _text(owner, 128)
        if not title or not description or not owner:
            raise ValueError("工单归属、标题和描述不能为空。")
        _choice(priority, PRIORITIES, "优先级")
        if job_id is not None and (not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id)):
            raise ValueError("任务编号无效。")
        now, ident = self._now(), secrets.token_hex(8)
        with self.lock, self._db(write=True) as db:
            db.execute("INSERT INTO tickets VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                       (ident, owner, title, description, job_id, "open", priority, None, 0, now, now, "[]"))
            if audit_actor is not None:
                self._record_event_db(db, {"action": "ticket_created", "actor": audit_actor, "owner": owner,
                                          "job": job_id, "status": "open", "source": "ticket",
                                          "details": {"ticket_id": ident, "priority": priority}})
            return self._ticket_row(db.execute("SELECT * FROM tickets WHERE id=?", (ident,)).fetchone())

    def ticket(self, ident):
        if not isinstance(ident, str) or not TICKET_ID.fullmatch(ident):
            return None
        with self._db() as db:
            return self._ticket_row(db.execute("SELECT * FROM tickets WHERE id=?", (ident,)).fetchone())

    def tickets(self, *, owner="", status="", q="", page=1, page_size=25, unassigned=None):
        _pagination(page, page_size)
        where, args = [], []
        for column, value in (("owner", _text(owner, 128)), ("status", _text(status, 30))):
            if value:
                if column == "status" and value == "unresolved":
                    where.append("status IN ('open','in_progress')")
                else:
                    if column == "status": _choice(value, STATUSES, "工单状态")
                    where.append(column + "=?"); args.append(value)
        if unassigned is not None:
            if type(unassigned) is not bool: raise ValueError("unassigned 必须是布尔值。")
            where.append("TRIM(COALESCE(assignee,''))" + ("=''" if unassigned else "<>''"))
            if unassigned: where.append("status IN ('open','in_progress')")
        q = _text(q, 200)
        if q:
            where.append("(" + " OR ".join(k + " LIKE ? ESCAPE '\\'" for k in ("id", "title", "description", "job_id", "assignee")) + ")")
            args.extend([_like(q)] * 5)
        clause = " WHERE " + " AND ".join(where) if where else ""
        with self._db() as db:
            total = db.execute("SELECT COUNT(*) FROM tickets" + clause, args).fetchone()[0]
            rows = db.execute("SELECT * FROM tickets" + clause + " ORDER BY updated_at DESC,id DESC LIMIT ? OFFSET ?", args + [page_size, (page - 1) * page_size]).fetchall()
            return [self._ticket_row(row) for row in rows], total

    def update_ticket(self, ident, *, version, status=None, priority=None, assignee=UNSET, message=UNSET, actor=None,
                      audit_actor=None, audit_action="ticket_updated"):
        if type(version) is not int or version < 0:
            raise ConflictError("工单版本无效，请刷新后重试。")
        if status is not None: _choice(status, STATUSES, "工单状态")
        if priority is not None: _choice(priority, PRIORITIES, "优先级")
        if assignee is not UNSET:
            if not isinstance(assignee, str): raise ValueError("处理人必须是字符串。")
            assignee = _text(assignee, 128)
        if message is not UNSET:
            if not isinstance(message, str): raise ValueError("回复必须是字符串。")
            message = _text(message, 4000)
            # The operations form sends an empty optional reply when only
            # changing status.  Empty text means no appended message.
        actor = _text(actor, 128, default="operator")
        _choice(audit_action, {"ticket_updated", "ticket_message"}, "工单审计动作")
        with self.lock, self._db(write=True) as db:
            current = self._ticket_row(db.execute("SELECT * FROM tickets WHERE id=?", (ident,)).fetchone())
            if current is None: raise KeyError("工单不存在。")
            if version != current["version"]: raise ConflictError("工单已更新，请刷新后重试。")
            messages = list(current["messages"])
            if message is not UNSET and message:
                if len(messages) >= MAX_MESSAGES:
                    raise ValueError(f"每个工单最多 {MAX_MESSAGES} 条回复，请新建后续工单。")
                messages.append({"actor": actor, "at": self._now(), "text": message})
            cur = db.execute("UPDATE tickets SET status=?,priority=?,assignee=?,version=version+1,updated_at=?,messages=? WHERE id=? AND version=?",
                             (status or current["status"], priority or current["priority"], current["assignee"] if assignee is UNSET else assignee,
                              self._now(), self._json(messages), ident, version))
            if cur.rowcount != 1: raise ConflictError("工单已更新，请刷新后重试。")
            if audit_actor is not None:
                self._record_event_db(db, {"action": audit_action, "actor": audit_actor, "owner": current["owner"],
                                          "job": current.get("job_id"), "status": status or current["status"],
                                          "source": "ticket", "details": {"ticket_id": ident}})
            return self._ticket_row(db.execute("SELECT * FROM tickets WHERE id=?", (ident,)).fetchone())

    @contextmanager
    def _open_relative(self, parts):
        """Open beneath the canonical root without following any symlink.

        Directory descriptors remain open while walking, preventing a rename or
        symlink substitution from redirecting a subsequent path component.
        """
        descriptors = []
        file_fd = None
        try:
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            descriptors.append(directory)
            for part in parts[:-1]:
                directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                descriptors.append(directory)
            file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise ValueError("只能读取普通文件。")
            with os.fdopen(file_fd, "rb") as stream:
                file_fd = None
                yield stream
        except OSError as exc:
            if exc.errno in {40, 20}:  # ELOOP / ENOTDIR from O_NOFOLLOW
                raise ValueError("归档路径不能包含符号链接。") from exc
            raise
        finally:
            if file_fd is not None: os.close(file_fd)
            for descriptor in reversed(descriptors): os.close(descriptor)

    @staticmethod
    def _source_parts(job_id, location):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise ValueError("任务编号无效。")
        _choice(location, {"active", "trash"}, "归档位置")
        return ([".trash"] if location == "trash" else []) + [job_id, "input.stl"]

    def read_trash_record(self, job_id):
        """Read a bounded, symlink-safe trash sidecar for an explicit job ID."""
        parts = self._source_parts(job_id, "trash")[:-1] + ["trash.json"]
        with self._open_relative(parts) as stream:
            body = stream.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise ValueError("回收站记录过大。")
        value = json.loads(body)
        job = value.get("job_snapshot") if isinstance(value, dict) else None
        if not isinstance(job, dict) or job.get("id") != job_id:
            raise ValueError("回收站记录与任务编号不匹配。")
        return value

    def archive_asset(self, *, job_id, owner, case_id="", release_id=None, location="active", note="", source_job=None,
                      expected_sha256=None, audit_actor=None):
        """Archive one original input; caller holds JobManager.lock during copy.

        Existing assets retain their note and review decision.  Additions of new
        provenance do not silently overwrite a concurrent operator's edit.
        """
        parts = self._source_parts(job_id, location)
        owner, case_id, release_id, note = _text(owner, 128), _text(case_id, 200), _text(release_id, 128), _text(note, 2000)
        if source_job is not None:
            if not isinstance(source_job, dict) or source_job.get("id") != job_id or (source_job.get("owner") or "") != owner:
                raise ValueError("任务来源记录不匹配。")
            expected_sha256 = expected_sha256 or source_job.get("input_sha256")
        if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not HEX64.fullmatch(expected_sha256)):
            raise ValueError("输入摘要无效。")
        if self.directory.is_symlink() or self.assets_dir.is_symlink() or self.assets_dir.resolve().parent != self.directory:
            raise ValueError("资产目录不能是符号链接。")
        fd, temp_name = tempfile.mkstemp(prefix=".stage-", dir=self.assets_dir)
        staged = Path(temp_name)
        digest, size = hashlib.sha256(), 0
        try:
            with os.fdopen(fd, "wb") as out, self._open_relative(parts) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk); size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES: raise ValueError("STL 超过 128 MiB 归档上限。")
                    out.write(chunk)
                out.flush(); os.fsync(out.fileno())
            if not size: raise ValueError("STL 为空，无法归档。")
            sha = digest.hexdigest()
            if expected_sha256 and expected_sha256 != sha:
                raise ConflictError("原始 STL 与任务记录摘要不一致，归档已停止。")
            metadata_source = dict(source_job or {})
            if not metadata_source.get("input_check"):
                try:
                    with self._open_relative(parts[:-1] + ["stage_a.json"]) as stream:
                        stage_bytes = stream.read(16 * 1024 * 1024 + 1)
                    stage = json.loads(stage_bytes) if len(stage_bytes) <= 16 * 1024 * 1024 else {}
                    if isinstance(stage, dict) and isinstance(stage.get("input_check"), dict):
                        metadata_source["input_check"] = stage["input_check"]
                except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
                    pass
            metadata = self._archive_metadata(metadata_source, sha)
            target = self.assets_dir / sha
            with self.lock, self._db(write=True) as db:
                if target.is_symlink(): raise ValueError("资产文件不能是符号链接。")
                publish = not target.exists()
                if not publish:
                    previous = hashlib.sha256()
                    with self._open_relative([".operations", "stl", sha]) as existing:
                        for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                            previous.update(chunk)
                    if previous.hexdigest() != sha: raise ConflictError("已有资产摘要不一致，请检查存储。")
                db.execute("INSERT OR IGNORE INTO assets(sha256,bytes,created_at,review_status,note,version) VALUES(?,?,?,?,?,0)",
                           (sha, size, self._now(), "candidate", note))
                db.execute("INSERT OR IGNORE INTO asset_sources(sha256,job_id,owner,case_id,release_id,archived_at,metadata) VALUES(?,?,?,?,?,?,?)",
                           (sha, job_id, owner, case_id, release_id, self._now(), self._json(metadata)))
                asset = self._asset_row(db, db.execute("SELECT * FROM assets WHERE sha256=?", (sha,)).fetchone())
                if audit_actor is not None:
                    self._record_event_db(db, {"action": "asset_archived", "actor": audit_actor, "owner": owner,
                                              "job": job_id, "status": asset["review_status"], "source": "asset",
                                              "details": {"sha256": sha, "location": location}})
                # Publish bytes only after all catalogue/audit writes succeeded;
                # they become visible together when the transaction commits.
                if publish:
                    os.replace(staged, target)
                    directory_fd = os.open(self.assets_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try: os.fsync(directory_fd)
                    finally: os.close(directory_fd)
                return asset
        finally:
            staged.unlink(missing_ok=True)

    @staticmethod
    def _archive_metadata(job, sha):
        """Keep only reproducibility fields, never patient/free-text metadata."""
        job = job if isinstance(job, dict) else {}
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        release = job.get("model_release") if isinstance(job.get("model_release"), dict) else {}
        input_check = job.get("input_check") if isinstance(job.get("input_check"), dict) else {}
        known_unit = params.get("units") if isinstance(params.get("units"), str) and params["units"] in {"mm", "cm", "m"} else "unknown"
        metadata = {"input_sha256": sha,
                    "input_check": {"selected_units": input_check.get("selected_units") or params.get("units") or "unknown",
                                    "resolved_units": input_check.get("resolved_units") or known_unit,
                                    "orientation_source": input_check.get("orientation_source") or "unknown_stl",
                                    **{key: input_check[key] for key in ("scale_factor", "unit_confidence", "unit_confirmation_required") if key in input_check}},
                    "params": {key: params[key] for key in ("units", "orientation") if key in params},
                    "model_release": {key: release[key] for key in ("id", "fingerprint", "contract") if key in release}}
        if job.get("run_identity"): metadata["run_identity"] = job["run_identity"]
        if isinstance(job.get("mapping"), dict): metadata["mapping"] = job["mapping"]
        # Bound a corrupt legacy record before it can expand every asset response.
        encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > 32 * 1024:
            raise ValueError("来源复现元数据过大。")
        return metadata

    def _asset_row(self, db, row):
        if row is None: return None
        value = dict(row)
        value["id"] = value["sha256"]
        value["sources_total"] = db.execute("SELECT COUNT(*) FROM asset_sources WHERE sha256=?", (value["sha256"],)).fetchone()[0]
        sources = db.execute("SELECT job_id,owner,case_id,release_id,archived_at,metadata FROM asset_sources WHERE sha256=? ORDER BY archived_at,job_id LIMIT ?", (value["sha256"], MAX_SOURCES)).fetchall()
        value["sources"] = [dict(source) for source in sources]
        for source in value["sources"]: source["metadata"] = json.loads(source["metadata"])
        return value

    def asset(self, sha):
        if not isinstance(sha, str) or not HEX64.fullmatch(sha): return None
        with self._db() as db:
            return self._asset_row(db, db.execute("SELECT * FROM assets WHERE sha256=?", (sha,)).fetchone())

    def assets(self, *, status="", q="", page=1, page_size=25):
        _pagination(page, page_size)
        where, args = [], []
        if status:
            _choice(status, REVIEWS, "资产审阅状态")
            where.append("review_status=?"); args.append(status)
        q = _text(q, 200)
        if q:
            where.append("(sha256 LIKE ? ESCAPE '\\' OR note LIKE ? ESCAPE '\\')"); args.extend([_like(q)] * 2)
        clause = " WHERE " + " AND ".join(where) if where else ""
        with self._db() as db:
            total = db.execute("SELECT COUNT(*) FROM assets" + clause, args).fetchone()[0]
            rows = db.execute("SELECT * FROM assets" + clause + " ORDER BY created_at DESC,sha256 LIMIT ? OFFSET ?", args + [page_size, (page - 1) * page_size]).fetchall()
            return [self._asset_row(db, row) for row in rows], total

    def review_asset(self, sha, *, review_status, note=None, version=None, audit_actor=None):
        _choice(review_status, REVIEWS, "资产审阅状态")
        if type(version) is not int or version < 0: raise ConflictError("资产版本无效，请刷新后重试。")
        if note is not None: note = _text(note, 2000)
        with self.lock, self._db(write=True) as db:
            current = db.execute("SELECT * FROM assets WHERE sha256=?", (sha,)).fetchone()
            if current is None: raise KeyError("资产不存在。")
            if version != current["version"]: raise ConflictError("资产已更新，请刷新后重试。")
            cur = db.execute("UPDATE assets SET review_status=?,note=?,version=version+1 WHERE sha256=? AND version=?",
                             (review_status, note if note is not None else current["note"], sha, version))
            if cur.rowcount != 1: raise ConflictError("资产已更新，请刷新后重试。")
            if audit_actor is not None:
                self._record_event_db(db, {"action": "asset_reviewed", "actor": audit_actor, "status": review_status,
                                          "source": "asset", "details": {"sha256": sha}})
            return self._asset_row(db, db.execute("SELECT * FROM assets WHERE sha256=?", (sha,)).fetchone())

    def write_asset_zip(self, sha, target):
        """Stream a verified input into a ZIP file, then write safe provenance.

        A failed hash check never returns a usable download.  Free-text notes,
        case labels, usernames and original filenames are excluded from export.
        The original STL bytes themselves are preserved unchanged.
        """
        value = self.asset(sha)
        if value is None: raise KeyError("资产不存在。")
        manifest = {"schema_version": "wss-deploy.asset/v1", "sha256": sha, "bytes": value["bytes"],
                    "created_at": value["created_at"], "review_status": value["review_status"],
                    "sources_total": value["sources_total"], "input_kind": "original_upload_unmodified",
                    "medical_deidentification_review": "not_performed",
                    "sources": [{"job_id": s["job_id"], "release_id": s.get("release_id"), "archived_at": s["archived_at"], "metadata": s.get("metadata", {})} for s in value["sources"]]}
        digest, size = hashlib.sha256(), 0
        try:
            with self._open_relative([".operations", "stl", sha]) as source, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                with archive.open("input.stl", "w", force_zip64=True) as out:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        size += len(chunk); digest.update(chunk); out.write(chunk)
                        if size > MAX_ARCHIVE_BYTES: raise ValueError("资产超过导出上限。")
                if digest.hexdigest() != sha or size != value["bytes"]:
                    raise ConflictError("资产摘要不一致，导出已停止。")
                archive.writestr("manifest.json", self._json(manifest))
        except BaseException:
            if isinstance(target, (str, os.PathLike)): Path(target).unlink(missing_ok=True)
            raise
        return value

    def asset_zip(self, sha) -> bytes:
        """Compatibility helper for small callers; HTTP uses write_asset_zip."""
        value = self.asset(sha)
        if value is None: raise KeyError("资产不存在。")
        if value["bytes"] > 8 * 1024 * 1024:
            raise ValueError("大文件请使用流式 ZIP 导出。")
        with tempfile.TemporaryFile() as stream:
            self.write_asset_zip(sha, stream)
            stream.seek(0)
            return stream.read()
