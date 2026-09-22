"""Local-first STL deployment service with persistent, owner-scoped tasks."""
from __future__ import annotations

from email.parser import BytesParser
from email.policy import default as email_default
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib, ipaddress, json, logging, os, re, secrets, socket, threading, time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import queue

from .jobs import FINAL, JOB_ID_PATTERN, JobError, JobManager, atomic_json
from .paths import PROJECT_ROOT, STATIC_DIR
from .registry import ReleaseRegistry, ReleaseError
from .users import UserStore, valid_username

LOG = logging.getLogger("wss_deploy.service")
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
MAX_BATCH_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024
MAX_PREFERENCES_BYTES = 16 * 1024
# One-page pictures (§15.1): up to 12 PNGs in one JSON body, so this route alone gets a larger cap.
MAX_SNAPSHOT_BYTES = 24 * 1024 * 1024
MAX_EXPORT_IDS = 200
MAX_EXPORT_ROWS = 500
COOKIE_NAME = "wss_session"
SESSION_SECONDS = 7 * 24 * 3600
SSE_KEEPALIVE_SECONDS = 15
SSE_MAX_SECONDS = 3600  # the browser reconnects transparently; bounds a forgotten tab's thread
STATIC_FILES = {"index.html", "app.js", "app.css", "three.min.js", "OrbitControls.js", "compare.html", "compare.js",
                "batch_export.js", "report_common.js", "workbench_core.js", "glossary.json"}
# Responses that may be embedded by our own pages (side-by-side comparison, one-page preview).
EMBEDDABLE_HTML = {"report.html", "onepage.html"}
OUTPUT_FILES = {"report.html", "summary.json", "run_manifest.json", "quality_audit.json", "wall_wss.vtp", "points_wss.csv", "field.npz",
                "volume_fields.vtp", "points_volume.csv", "wall_pressure.vtp", "streamlines.vtp",
                "annotations.json", "findings_review.json", "snapshots.json", "narrative.json"}
# One-page pictures are written by the report page itself; the whitelist stays strict (PNG only, fixed prefix).
SNAPSHOT_FILE = re.compile(r"snapshot_[a-z0-9_-]{1,32}\.png")
OWNER_KEY_PATTERN = r"[A-Za-z0-9_.-]{1,64}"


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost": return True
    try: return ipaddress.ip_address(host).is_loopback
    except ValueError: return False


def contained_file(root: Path, name: str, allowed: set[str]) -> Path | None:
    if name not in allowed: return None
    root = Path(root).resolve(); candidate = (root / name).resolve()
    return candidate if candidate.parent == root and candidate.is_file() else None


def output_file(job_dir: Path, name: str) -> Path | None:
    """A servable result file: the fixed whitelist plus the generated ``snapshot_<name>.png`` set."""
    allowed = OUTPUT_FILES | ({name} if SNAPSHOT_FILE.fullmatch(name) else set())
    return contained_file(job_dir, name, allowed)


def owner_key(owner: str) -> str:
    """File-name-safe key for per-owner files (user names as-is, random session owners hashed)."""
    return owner if re.fullmatch(OWNER_KEY_PATTERN, owner or "") else hashlib.sha256(str(owner).encode("utf-8")).hexdigest()[:32]


class PreferenceStore:
    """Per-owner JSON preferences (C6): whole-document replace, size-capped, one file per owner."""
    def __init__(self, root: Path):
        self.root = Path(root); self.lock = threading.Lock()
    def _path(self, owner: str) -> Path: return self.root / f"{owner_key(owner)}.json"
    def get(self, owner: str) -> dict:
        path = self._path(owner)
        if not path.is_file(): return {}
        try: value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError): return {}
        return value if isinstance(value, dict) else {}
    def put(self, owner: str, value: dict) -> dict:
        if not isinstance(value, dict): raise JobError("偏好必须是 JSON 对象。")
        if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > MAX_PREFERENCES_BYTES: raise JobError("偏好内容超过 16 KB。", 413)
        with self.lock:
            self.root.mkdir(mode=0o700, exist_ok=True); atomic_json(self._path(owner), value)
        return value


class SessionStore:
    def __init__(self, path: Path, *, shared: bool, token: str | None = None, legacy_owner: str | None = None, users: UserStore | None = None):
        if shared and not token and not (users and users.exists()):
            raise ValueError("共享访问需要设置 WSS_DEPLOY_TOKEN 或先用 `python -m wss_deploy.cli user add` 创建用户；单人使用请绑定 127.0.0.1。")
        if shared and legacy_owner: raise ValueError("WSS_DEPLOY_LEGACY_OWNER 仅允许在本地回环模式使用。")
        self.path, self.shared, self.token, self.legacy_owner, self.users = Path(path), shared, token, legacy_owner, users
        self.lock = threading.RLock(); self.sessions = {}
        if self.path.exists():
            try: self.sessions = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception: LOG.warning("Ignoring invalid sessions file")
        self.sessions = {k: v for k, v in self.sessions.items() if v.get("expires_at", 0) > time.time() and v.get("shared") is shared}

    def login_mode(self) -> str:
        """``none`` on loopback (login-free), ``password`` once users exist, ``token`` before that."""
        if not self.shared: return "none"
        return "password" if self.users is not None and self.users.exists() else "token"
    def legacy_token_allowed(self) -> bool:
        if not self.shared or not self.token: return False
        return self.login_mode() == "token" or os.environ.get("WSS_DEPLOY_ALLOW_LEGACY_TOKEN") == "1"

    def lookup(self, cookie: str | None):
        try:
            parsed = SimpleCookie(); parsed.load(cookie or ""); sid = parsed[COOKIE_NAME].value
        except (KeyError, ValueError): return None
        with self.lock:
            row = self.sessions.get(sid)
            if row and row["expires_at"] > time.time(): return sid, row.copy()
        return None

    def _new_row(self, owner: str, **extra) -> dict:
        return {"owner": owner, "csrf_token": secrets.token_urlsafe(32), "expires_at": time.time() + SESSION_SECONDS, "shared": self.shared,
                "username": None, "role": None, "display_name": None, "claimable_owners": [], **extra}

    def create(self, token: str | None = None, *, username=None, password=None, previous: dict | None = None, claimable_jobs=None):
        """Open a session: loopback → anonymous; shared → username/password (users.json) or the legacy token."""
        if not self.shared:
            row = self._new_row(self.legacy_owner or secrets.token_urlsafe(24))
        elif username is not None or password is not None:
            if self.users is None or self.login_mode() != "password": raise JobError("当前服务未启用用户名登录。", 401)
            user = self.users.verify(username, password)
            claimable = []
            if previous and previous.get("owner") and previous["owner"] != user["username"]:
                claimable = list(dict.fromkeys([*previous.get("claimable_owners", []), previous["owner"]]))
            row = self._new_row(user["username"], username=user["username"], role=user["role"], display_name=user["display_name"],
                                claimable_owners=claimable)
        else:
            if not self.legacy_token_allowed(): raise JobError("请使用用户名和口令登录。", 401)
            if not isinstance(token, str) or not secrets.compare_digest(token, self.token or ""): raise JobError("访问口令不正确。", 401)
            row = self._new_row(secrets.token_urlsafe(24), role="legacy")
        sid = secrets.token_urlsafe(32)
        with self.lock: self.sessions[sid] = row; atomic_json(self.path, self.sessions)
        return sid, row.copy()

    def destroy(self, sid: str) -> None:
        with self.lock:
            if self.sessions.pop(sid, None) is not None: atomic_json(self.path, self.sessions)

    def drop_claimable(self, sid: str, owner: str) -> None:
        with self.lock:
            row = self.sessions.get(sid)
            if row and owner in row.get("claimable_owners", []):
                row["claimable_owners"] = [o for o in row["claimable_owners"] if o != owner]; atomic_json(self.path, self.sessions)

    def describe(self, row: dict | None, *, manager=None) -> dict:
        """Public session facts for ``GET/POST /api/session``."""
        out = {"authenticated": bool(row), "shared": self.shared, "csrf_token": row["csrf_token"] if row else None,
               "max_upload_bytes": MAX_UPLOAD_BYTES, "login": self.login_mode(), "legacy_token_allowed": self.legacy_token_allowed(),
               "username": row.get("username") if row else None, "display_name": row.get("display_name") if row else None,
               "role": row.get("role") if row else None, "claimable_owners": list(row.get("claimable_owners", [])) if row else []}
        if row and manager is not None and out["claimable_owners"]:
            out["claimable"] = [{"owner": owner, "jobs": manager.count_owner(owner)} for owner in out["claimable_owners"]]
        return out


class ServiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, manager, sessions, preferences: PreferenceStore | None = None):
        if ":" in address[0]: self.address_family = socket.AF_INET6
        self.manager, self.sessions = manager, sessions
        self.preferences = preferences or PreferenceStore(Path(manager.root) / "preferences")
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: ServiceHTTPServer
    def setup(self): super().setup(); self.connection.settimeout(30)
    def log_message(self, fmt, *args): LOG.info("%s %s", self.client_address[0], fmt % args)

    def _send(self, code, body, ctype="application/json; charset=utf-8", *, cookie=None, download=None, headers=None, embeddable=False, clear_cookie=False):
        # ``embeddable`` responses (reports, one-page previews) may be framed by our own origin only;
        # everything else keeps ``frame-ancestors 'none'``.
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff"); self.send_header("X-Frame-Options", "SAMEORIGIN" if embeddable else "DENY"); self.send_header("Referrer-Policy", "no-referrer")
        for name, value in (headers or {}).items(): self.send_header(name, value)
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors " + ("'self'" if embeddable else "'none'"))
        if cookie: self.send_header("Set-Cookie", f"{COOKIE_NAME}={cookie}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_SECONDS}")
        if clear_cookie: self.send_header("Set-Cookie", f"{COOKIE_NAME}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
        if download: self.send_header("Content-Disposition", _disposition(download))
        self.end_headers(); self.wfile.write(body)
    def _json(self, obj, code=200, **kwargs): self._send(code, json.dumps(obj, ensure_ascii=False, allow_nan=False).encode(), **kwargs)

    def _path(self):
        try: path = unquote(urlsplit(self.path).path, errors="strict")
        except (ValueError, UnicodeError): raise JobError("无效的请求路径。", 400)
        if "\\" in path or "\0" in path or any(x in {".", ".."} for x in path.split("/")): raise JobError("请求路径不允许。", 404)
        return path
    def _host(self):
        host = self.headers.get("Host", "")
        try:
            parsed = urlsplit("http://" + host); valid = bool(parsed.hostname) and parsed.path == "" and parsed.username is None and parsed.password is None; port = parsed.port
        except ValueError: valid, port, parsed = False, None, None
        if not valid or any(c in host for c in "\r\n, /\\"): raise JobError("无效的服务地址。", 400)
        if not self.server.sessions.shared and (not is_loopback(parsed.hostname or "") or (port is not None and port != self.server.server_port)): raise JobError("本地服务仅接受回环地址访问。", 403)
        return host
    def _origin(self):
        host = self._host(); origin = self.headers.get("Origin")
        if origin is not None and origin not in {"http://" + host, "https://" + host}: raise JobError("请求来源与当前服务不一致。", 403)
        if self.headers.get("Sec-Fetch-Site") == "cross-site": raise JobError("不接受跨站请求。", 403)
    def _session(self, local_create=False):
        self._host(); found = self.server.sessions.lookup(self.headers.get("Cookie"))
        if found: return found[0], found[1], False
        if local_create and not self.server.sessions.shared:
            sid, row = self.server.sessions.create(); return sid, row, True
        return None, None, False
    def _require_session(self, mutation=False):
        _, row, _ = self._session()
        if not row: raise JobError("会话已失效，请刷新页面或重新输入访问口令。", 401)
        if mutation:
            self._origin(); csrf = self.headers.get("X-CSRF-Token", "")
            if not secrets.compare_digest(csrf, row["csrf_token"]): raise JobError("页面会话校验失败，请刷新页面后重试。", 403)
        return row
    @staticmethod
    def _admin(row: dict) -> bool: return row.get("role") == "admin"
    def _body(self, limit):
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1: raise JobError("请求必须指定唯一的 Content-Length。", 411)
        try: length = int(lengths[0])
        except ValueError: raise JobError("无效的请求长度。", 400)
        if length < 0 or length > limit: raise JobError(f"请求过大；STL 上传上限为 {MAX_UPLOAD_BYTES // 1024 // 1024} MiB。", 413)
        body = self.rfile.read(length)
        if len(body) != length: raise JobError("上传未完成，请重新上传。", 400)
        return body
    def _payload(self, limit=MAX_JSON_BYTES):
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json": raise JobError("请求必须使用 application/json。", 415)
        try: value = json.loads(self._body(limit) or b"{}")
        except (ValueError, UnicodeError): raise JobError("请求内容不是有效 JSON。", 400)
        if not isinstance(value, dict): raise JobError("请求内容必须是 JSON 对象。", 400)
        return value
    def _handle(self, method):
        try: {"GET": self._get, "POST": self._post, "PUT": self._put}[method]()
        except JobError as e:
            body = {"error": {"message": str(e)}}; body.update(e.payload); self._json(body, e.status)
        except (BrokenPipeError, ConnectionResetError, TimeoutError): LOG.info("Client disconnected")
        except Exception:
            did = secrets.token_hex(6); LOG.exception("Request failed; diagnostic_id=%s", did); self._json({"error": {"message": "服务暂时无法完成请求，请重试或提供诊断编号给维护者。", "diagnostic_id": did}}, 500)
    def do_GET(self): self._handle("GET")
    def do_POST(self): self._handle("POST")
    def do_PUT(self): self._handle("PUT")

    # ------------------------------------------------------------------ streams
    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
    def _emit(self, name: str, payload: dict) -> None:
        self.wfile.write(f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n".encode("utf-8"))
        self.wfile.flush()
    def _pump(self, listener: queue.Queue, *, stop_when_final: bool) -> None:
        deadline = time.time() + SSE_MAX_SECONDS
        while time.time() < deadline:
            try:
                event = listener.get(timeout=SSE_KEEPALIVE_SECONDS)
            except queue.Empty:
                self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
                continue
            self._emit("job", event)
            if stop_when_final and event.get("final"):
                return

    def _stream_events(self, job_id: str, owner: str, *, any_owner: bool = False) -> None:
        """Server-sent events: one snapshot, then every job event until the job is final or the client leaves.

        Replaces the 2 s polling of the full job record; the browser reconnects on its own if the
        stream is cut, and the page still polls as a fallback when EventSource is unavailable.
        """
        manager = self.server.manager
        listener, snapshot = manager.subscribe(job_id, owner, any_owner=any_owner)
        try:
            self._sse_headers()
            self._emit("snapshot", snapshot)
            if snapshot.get("final"):
                return
            self._pump(listener, stop_when_final=True)
        finally:
            manager.unsubscribe(job_id, listener)

    def _stream_owner_events(self, owner: str) -> None:
        """Owner-level stream (contract §11.5): ``hello`` then every transition of the owner's jobs."""
        manager = self.server.manager
        listener, hello = manager.subscribe_owner(owner)
        try:
            self._sse_headers()
            self._emit("hello", hello)
            self._pump(listener, stop_when_final=False)
        finally:
            manager.unsubscribe_owner(owner, listener)

    # ------------------------------------------------------------------ helpers
    def _job_dir(self, job_id: str) -> Path:
        manager = self.server.manager
        job_dir = (manager.root / job_id).resolve()
        if job_dir.parent != manager.root: raise JobError("文件不存在。", 404)
        return job_dir
    def _summary(self, job_id: str) -> dict:
        summary_file = output_file(self._job_dir(job_id), "summary.json")
        if not summary_file: raise JobError("缺少 summary.json。", 404)
        try: summary = json.loads(summary_file.read_text(encoding="utf-8"))
        except (OSError, ValueError): raise JobError("summary.json 无法读取。", 500)
        return summary if isinstance(summary, dict) else {}
    def _done_job(self, job_id: str, row: dict, *, what: str) -> dict:
        job = self.server.manager.get(job_id, row["owner"], any_owner=self._admin(row))
        if job["status"] != "done": raise JobError(f"任务完成后才能{what}。", 409)
        return job
    def _onepage(self, job_id: str, job: dict) -> str:
        from .onepager import render_onepage
        return render_onepage(self._summary(job_id), job, job_dir=self._job_dir(job_id))
    def _bundle(self, job_id: str, row: dict) -> tuple[dict, bytes]:
        from .bundle import job_bundle
        job = self._done_job(job_id, row, what="打包下载")
        try: onepage = self._onepage(job_id, job)
        except JobError: onepage = None
        return job, job_bundle(self._job_dir(job_id), job, OUTPUT_FILES, onepage_html=onepage)
    def _ids(self, raw, limit: int) -> list[str]:
        """Job ids from ``?ids=a,b`` or a JSON list; de-duplicated, order kept."""
        if isinstance(raw, str): raw = raw.split(",")
        if not isinstance(raw, list): raise JobError("ids 必须是逗号分隔的字符串或列表。")
        ids = list(dict.fromkeys(item.strip() for item in raw if isinstance(item, str) and item.strip()))
        if not ids: raise JobError("请提供至少一个任务编号。")
        if len(ids) > limit: raise JobError(f"一次最多处理 {limit} 个任务。")
        if any(not re.fullmatch(JOB_ID_PATTERN, item) for item in ids): raise JobError("任务编号无效。")
        return ids
    @staticmethod
    def _query(parsed_request):
        query = parse_qs(parsed_request.query, keep_blank_values=True)
        def one(name, default=""):
            return query.get(name, [default])[-1]
        def integer(name, default):
            raw = one(name, str(default))
            try: return int(raw)
            except (TypeError, ValueError): raise JobError(f"{name} 必须是整数。")
        return query, one, integer

    # ------------------------------------------------------------------ GET
    def _get(self):
        parsed_request = urlsplit(self.path)
        path = self._path(); self._host()
        sessions = self.server.sessions
        if path == "/api/session":
            sid, row, fresh = self._session(local_create=True)
            return self._json(sessions.describe(row, manager=self.server.manager), cookie=sid if fresh else None)
        if path in {"/", "/compare"} or path.startswith("/static/"):
            name = {"/": "index.html", "/compare": "compare.html"}.get(path) or path.removeprefix("/static/"); file = contained_file(STATIC_DIR, name, STATIC_FILES)
            if not file: raise JobError("文件不存在。", 404)
            sid, _, fresh = self._session(local_create=path in {"/", "/compare"}); ctype = {".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8"}.get(file.suffix, "application/octet-stream")
            return self._send(200, file.read_bytes(), ctype, cookie=sid if fresh else None)
        row = self._require_session(); manager = self.server.manager; admin = self._admin(row)
        query, one, integer = self._query(parsed_request)
        all_owners = admin and one("all", "0") == "1"
        if path == "/api/releases":
            return self._json({"releases": manager.releases()})
        if path == "/api/compare":
            raise JobError("结果比较需要使用 POST 请求。", 405)
        if path == "/api/events":
            return self._stream_owner_events(row["owner"])
        if path == "/api/preferences":
            return self._json({"preferences": self.server.preferences.get(row["owner"])})
        if path == "/api/trash":
            return self._json(manager.trash_list(row["owner"], all_owners=all_owners))
        if path == "/api/cases":
            return self._json(manager.cases(row["owner"], q=one("q"), patient_id=one("patient_id"), tag=one("tag"),
                                            page=integer("page", 1), page_size=integer("page_size", 20), all_owners=all_owners))
        if path == "/api/jobs/export":
            return self._export_table(row, one, all_owners=all_owners)
        if path == "/api/jobs":
            keys = {"q", "status", "patient_id", "tag", "page", "page_size"}
            if not (keys & set(query)):
                return self._json({"jobs": manager.list(row["owner"], all_owners=all_owners)})
            return self._json(manager.query(row["owner"], q=one("q"), status=one("status"),
                                            patient_id=one("patient_id"), tag=one("tag"),
                                            page=integer("page", 1), page_size=integer("page_size", 25), all_owners=all_owners))
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})", path)
        if match: return self._json(manager.get(match[1], row["owner"], any_owner=admin))
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/events", path)
        if match:
            return self._stream_events(match[1], row["owner"], any_owner=admin)
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/geometry", path)
        if match:
            # Display mesh + centreline polylines: fetched once per stage-A result, never polled.
            geometry = manager.geometry(match[1], row["owner"], any_owner=admin)
            etag = f'W/"{geometry["job_id"]}-{geometry["version"]}-{geometry.get("stage_a_created_at") or ""}"'
            if self.headers.get("If-None-Match") == etag:
                return self._send(304, b"", headers={"ETag": etag})
            return self._json(geometry, headers={"ETag": etag})
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/onepage", path)
        if match:
            # Rendered on demand from summary.json; nothing is written to the job directory.
            job = self._done_job(match[1], row, what="生成一页纸报告")
            return self._send(200, self._onepage(match[1], job).encode("utf-8"), "text/html; charset=utf-8", embeddable=True)
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/bundle\.zip", path)
        if match:
            from .bundle import bundle_name
            job, data = self._bundle(match[1], row)
            return self._send(200, data, "application/zip", download=bundle_name(job))
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/(?:files/([^/]+)|(report))", path); legacy = re.fullmatch(rf"/jobs/({JOB_ID_PATTERN})/([^/]+)", path)
        if match or legacy:
            job_id = (match or legacy)[1]; name = (match[2] or "report.html") if match else legacy[2]
            self._done_job(job_id, row, what="打开报告和导出文件")
            file = output_file(self._job_dir(job_id), name)
            if not file: raise JobError("文件不存在或未生成。", 404)
            ctype = {".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8", ".csv": "text/csv; charset=utf-8", ".vtp": "application/xml", ".png": "image/png"}.get(file.suffix, "application/octet-stream")
            return self._send(200, file.read_bytes(), ctype, download=name if file.suffix not in {".html", ".json", ".png"} else None, embeddable=name in EMBEDDABLE_HTML)
        raise JobError("接口不存在。", 404)

    def _export_table(self, row: dict, one, *, all_owners: bool = False) -> None:
        from .export_table import COLUMNS, build_rows, export_filename, population_blocks, to_csv, to_xlsx
        fmt = one("format", "csv").lower()
        if fmt not in {"csv", "xlsx", "json"}: raise JobError("format 只能是 csv、xlsx 或 json。")
        scope = one("scope", "ids") or "ids"
        if scope not in {"ids", "done"}: raise JobError("scope 只能是 ids 或 done。")
        manager = self.server.manager; records = []
        if fmt == "json" and scope == "done":
            # Cohort overview (§15.3): every finished job of this owner, newest first.
            jobs = [job for job in manager.list(row["owner"], all_owners=all_owners) if job.get("status") == "done"][:MAX_EXPORT_ROWS]
        else:
            jobs = [manager.get(job_id, row["owner"], any_owner=self._admin(row)) for job_id in self._ids(one("ids"), MAX_EXPORT_IDS)]
        for job in jobs:
            summary = None
            if job["status"] == "done":
                try: summary = self._summary(job["id"])
                except JobError: summary = None
            records.append((job, summary))
        rows = build_rows(records)
        if fmt == "json":
            population = population_blocks(getattr(manager, "registry", None), [r.get("release_id") for r in rows])
            return self._json({"columns": list(COLUMNS), "rows": rows, "population": population})
        data = to_csv(rows) if fmt == "csv" else to_xlsx(rows)
        ctype = "text/csv; charset=utf-8" if fmt == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        return self._send(200, data, ctype, download=export_filename(fmt))

    # ------------------------------------------------------------------ POST
    def _post(self):
        path = self._path(); self._host()
        sessions = self.server.sessions; manager = self.server.manager
        if path == "/api/session":
            self._origin(); payload = self._payload()
            _, previous, _ = self._session()
            if "username" in payload or "password" in payload:
                sid, row = sessions.create(username=payload.get("username"), password=payload.get("password"), previous=previous)
            else:
                sid, row = sessions.create(payload.get("token"))
            return self._json(sessions.describe(row, manager=manager), cookie=sid)
        if path == "/api/session/logout":
            self._origin(); sid, row, _ = self._session()
            if sid: sessions.destroy(sid)
            return self._json({"authenticated": False}, clear_cookie=True)
        row = self._require_session(mutation=True); admin = self._admin(row)
        if path == "/api/session/password":
            payload = self._payload()
            if not row.get("username") or sessions.users is None: raise JobError("当前会话不是用户名登录，不能改口令。", 409)
            sessions.users.change_password(row["username"], payload.get("old_password"), payload.get("new_password"))
            return self._json({"changed": True})
        if path == "/api/jobs/claim":
            payload = self._payload(); sid, _, _ = self._session()
            owner = payload.get("owner")
            if not isinstance(owner, str) or not owner: raise JobError("请提供要认领的旧会话 owner。")
            if not row.get("username"): raise JobError("请先用用户名登录再认领任务。", 409)
            if owner not in row.get("claimable_owners", []) and not admin: raise JobError("该 owner 不属于本会话可认领的范围。", 403)
            result = manager.claim_owner(owner, row["username"])
            if sid: sessions.drop_claimable(sid, owner)
            return self._json(result)
        if path == "/api/compare":
            payload = self._payload()
            left_id, right_id = payload.get("left_job_id"), payload.get("right_job_id")
            if not isinstance(left_id, str) or not isinstance(right_id, str):
                ids = payload.get("job_ids")
                if isinstance(ids, list) and len(ids) == 2 and all(isinstance(item, str) for item in ids):
                    left_id, right_id = ids
            if not left_id or not right_id:
                raise JobError("请提供两个已完成任务编号。")
            return self._json({"comparison": manager.compare(left_id, right_id, row["owner"])})
        if path == "/api/jobs/bundle":
            from .bundle import MAX_BUNDLE_JOBS, bundle_name, multi_bundle
            payload = self._payload(); ids = self._ids(payload.get("ids"), MAX_BUNDLE_JOBS)
            bundles = []
            for job_id in ids:
                job, data = self._bundle(job_id, row); bundles.append((bundle_name(job), data))
            return self._send(200, multi_bundle(bundles), "application/zip", download=f"wss_bundle_{time.strftime('%Y%m%d_%H%M')}.zip")
        if path in {"/api/jobs", "/api/jobs/batch", "/api/upload"}:
            return self._upload(row, batch=path == "/api/jobs/batch")
        if path == "/api/jobs/delete":
            payload = self._payload()
            items = payload.get("jobs")
            if not isinstance(items, list) or not items:
                raise JobError("请提供要删除的任务列表（id 与 version）。")
            return self._json(manager.delete_many(row["owner"], items))
        match = re.fullmatch(rf"/api/trash/({JOB_ID_PATTERN})/(restore|purge)", path)
        if match:
            self._payload()
            if match[2] == "restore": return self._json({"job": manager.restore(match[1], row["owner"], any_owner=admin)})
            return self._json(manager.purge(match[1], row["owner"], any_owner=admin))
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/snapshots", path)
        if match:
            # Pictures for the one-page report: a much larger body than any other JSON route.
            return self._json(manager.snapshots(match[1], row["owner"], self._payload(MAX_SNAPSHOT_BYTES)))
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/metadata", path)
        if match:
            return self._json({"job": manager.update_metadata(match[1], row["owner"], self._payload())})
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/(input|confirm|cancel|retry|rerun|delete|review)", path)
        if match:
            try:
                result = getattr(manager, match[2])(match[1], row["owner"], self._payload())
            except ReleaseError as exc:
                raise JobError(str(exc), 422)
            if match[2] == "delete":
                return self._json(result)
            return self._json({"job": result})
        raise JobError("接口不存在。", 404)

    # ------------------------------------------------------------------ PUT
    def _put(self):
        path = self._path(); self._host()
        row = self._require_session(mutation=True); manager = self.server.manager
        if path == "/api/preferences":
            return self._json({"preferences": self.server.preferences.put(row["owner"], self._payload(MAX_PREFERENCES_BYTES + 4096))})
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/(annotations|findings_review|narrative)", path)
        if match:
            return self._json(getattr(manager, match[2])(match[1], row["owner"], self._payload()))
        raise JobError("接口不存在。", 404)

    # ------------------------------------------------------------------ uploads
    def _upload(self, row: dict, *, batch: bool) -> None:
        manager = self.server.manager
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data;"): raise JobError("上传必须使用 multipart/form-data。", 415)
        body = self._body((MAX_BATCH_BYTES if batch else MAX_UPLOAD_BYTES) + 64 * 1024); message = BytesParser(policy=email_default).parsebytes(b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
        if not message.is_multipart(): raise JobError("上传表单格式无效。")
        fields, uploads, metadata_json = {}, [], None
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "stl":
                if not batch and uploads: raise JobError("每个任务只能上传一个 STL。")
                uploads.append((part.get_filename() or "input.stl", part.get_payload(decode=True)))
            elif name == "metadata_json":
                data = part.get_payload(decode=True) or b""
                if len(data) > 64 * 1024: raise JobError("批量元数据过长。")
                try: metadata_json = json.loads(data.decode("utf-8"))
                except (UnicodeError, ValueError): raise JobError("metadata_json 不是有效 JSON。")
            elif name in {"case_id", "units", "remove_fragments", "release_id", "patient_id",
                          "scan_label", "scan_date", "tags", "notes", "device", "seed_count", "threads", "on_duplicate"}:
                data = part.get_payload(decode=True) or b""
                # Identifiers stay small; notes intentionally allow the
                # same 2,000-character limit as JobManager._metadata.
                field_limit = 4096 if name == "notes" else 1024
                if len(data) > field_limit: raise JobError("上传表单字段过长。")
                try: fields[name] = data.decode("utf-8").strip()
                except UnicodeError: raise JobError("上传表单字段必须使用 UTF-8。")
        if not uploads or any(not content for _, content in uploads): raise JobError("请选择 STL 文件。")
        if any(len(content) > MAX_UPLOAD_BYTES for _, content in uploads): raise JobError("STL 超过 128 MiB 上传上限。", 413)
        remove = fields.get("remove_fragments", "false").lower()
        if remove not in {"true", "false", "1", "0", "on", "off"}: raise JobError("表面修复选项无效。")
        on_duplicate = fields.get("on_duplicate") or "ask"
        if on_duplicate not in {"ask", "reuse", "force"}: raise JobError("on_duplicate 只能是 ask、reuse 或 force。")
        # STL has no embedded unit metadata.  Omitted units therefore go
        # through the explicit input-confirmation page instead of being
        # silently treated as millimetres.
        tags = [item.strip() for item in fields.get("tags", "").split(",") if item.strip()]
        seed_count = fields.get("seed_count", "")
        if seed_count and seed_count.lower() in {"all", "全部"}:
            seed_count = None
        elif seed_count:
            try: seed_count = int(seed_count)
            except ValueError: raise JobError("集成模型数必须是整数或 all。")
        threads = fields.get("threads", "")
        if threads:
            try: threads = int(threads)
            except ValueError: raise JobError("CPU 线程数必须是整数。")
        if batch:
            if len(uploads) > 20: raise JobError("每批最多上传 20 个 STL 文件。")
            if metadata_json is None: metadata_json = [{} for _ in uploads]
            if not isinstance(metadata_json, list) or len(metadata_json) != len(uploads) or any(not isinstance(item, dict) for item in metadata_json):
                raise JobError("metadata_json 必须是与 STL 数量一致的对象列表。")
            defaults = {"case_id": fields.get("case_id", ""), "release_id": fields.get("release_id") or None,
                        "units": fields.get("units", "auto"), "remove_fragments": remove in {"true", "1", "on"},
                        "device": fields.get("device", "auto"), "seed_count": seed_count, "threads": threads,
                        "patient_id": fields.get("patient_id", ""), "scan_label": fields.get("scan_label", ""),
                        "scan_date": fields.get("scan_date", ""), "tags": tags, "notes": fields.get("notes", ""),
                        "on_duplicate": on_duplicate}
            items = []
            for (filename, content), item_meta in zip(uploads, metadata_json):
                item = {"filename": filename, "content": content, **item_meta}
                items.append(item)
            try:
                result = manager.create_batch(row["owner"], items, **defaults)
            except ReleaseError as exc:
                raise JobError(str(exc), 422)
            return self._json(result, 201)
        try:
            job = manager.create(row["owner"], content=uploads[0][1], filename=uploads[0][0],
                                 case_id=fields.get("case_id", ""), release_id=fields.get("release_id") or None,
                                 patient_id=fields.get("patient_id", ""), scan_label=fields.get("scan_label", ""),
                                 scan_date=fields.get("scan_date", ""), tags=tags, notes=fields.get("notes", ""),
                                 units=fields.get("units", "auto"), remove_fragments=remove in {"true", "1", "on"},
                                 device=fields.get("device", "auto"), seed_count=seed_count, threads=threads,
                                 on_duplicate=on_duplicate)
        except ReleaseError as exc:
            raise JobError(str(exc), 422)
        response = {"job": job, "job_id": job["id"]}
        if job.get("reused_from"): response["reused_from"] = job["reused_from"]
        return self._json(response, 201)


def _disposition(name: str) -> str:
    """ASCII fallback plus RFC 5987 UTF-8 form so Chinese case names survive as download names."""
    from urllib.parse import quote
    name = str(name).replace("\r", "").replace("\n", "")
    ascii_name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name) or "download"
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(name, safe="")}'


def serve(host: str = "127.0.0.1", port: int = 8765, jobs_root: Path | None = None, device: str = "auto", token: str | None = None) -> None:
    root = Path(jobs_root or PROJECT_ROOT / "outputs/wss_deploy_jobs").resolve(); root.mkdir(parents=True, exist_ok=True); shared = not is_loopback(host); legacy_owner = os.environ.get("WSS_DEPLOY_LEGACY_OWNER") or None
    users = UserStore(root / "users.json")
    sessions = SessionStore(root / ".sessions.json", shared=shared, token=token or os.environ.get("WSS_DEPLOY_TOKEN"), legacy_owner=legacy_owner, users=users)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    registry = ReleaseRegistry(device=device)
    manager = JobManager(root, registry=registry, legacy_owner=legacy_owner); httpd = ServiceHTTPServer((host, port), manager, sessions); manager.start(); LOG.info("Serving on http://%s:%s; jobs=%s; shared=%s; login=%s; releases=%s", host, port, root, shared, sessions.login_mode(), registry.root)
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally: httpd.server_close(); manager.close()
