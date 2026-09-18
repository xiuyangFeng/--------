"""Local-first STL deployment service with persistent, owner-scoped tasks."""
from __future__ import annotations

from email.parser import BytesParser
from email.policy import default as email_default
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress, json, logging, os, re, secrets, socket, threading, time
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .jobs import JOB_ID_PATTERN, JobError, JobManager, atomic_json
from .paths import PROJECT_ROOT, STATIC_DIR

LOG = logging.getLogger("wss_deploy.service")
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024
COOKIE_NAME = "wss_session"
SESSION_SECONDS = 7 * 24 * 3600
STATIC_FILES = {"index.html", "app.js", "app.css", "three.min.js", "OrbitControls.js"}
OUTPUT_FILES = {"report.html", "summary.json", "wall_wss.vtp", "points_wss.csv", "field.npz"}


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost": return True
    try: return ipaddress.ip_address(host).is_loopback
    except ValueError: return False


def contained_file(root: Path, name: str, allowed: set[str]) -> Path | None:
    if name not in allowed: return None
    root = Path(root).resolve(); candidate = (root / name).resolve()
    return candidate if candidate.parent == root and candidate.is_file() else None


class SessionStore:
    def __init__(self, path: Path, *, shared: bool, token: str | None = None, legacy_owner: str | None = None):
        if shared and not token: raise ValueError("共享访问需要设置 WSS_DEPLOY_TOKEN；单人使用请绑定 127.0.0.1。")
        if shared and legacy_owner: raise ValueError("WSS_DEPLOY_LEGACY_OWNER 仅允许在本地回环模式使用。")
        self.path, self.shared, self.token, self.legacy_owner = Path(path), shared, token, legacy_owner
        self.lock = threading.RLock(); self.sessions = {}
        if self.path.exists():
            try: self.sessions = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception: LOG.warning("Ignoring invalid sessions file")
        self.sessions = {k: v for k, v in self.sessions.items() if v.get("expires_at", 0) > time.time() and v.get("shared") is shared}

    def lookup(self, cookie: str | None):
        try:
            parsed = SimpleCookie(); parsed.load(cookie or ""); sid = parsed[COOKIE_NAME].value
        except (KeyError, ValueError): return None
        with self.lock:
            row = self.sessions.get(sid)
            if row and row["expires_at"] > time.time(): return sid, row.copy()
        return None

    def create(self, token: str | None = None):
        if self.shared and (not isinstance(token, str) or not secrets.compare_digest(token, self.token or "")): raise JobError("访问口令不正确。", 401)
        sid = secrets.token_urlsafe(32)
        row = {"owner": self.legacy_owner or secrets.token_urlsafe(24), "csrf_token": secrets.token_urlsafe(32), "expires_at": time.time() + SESSION_SECONDS, "shared": self.shared}
        with self.lock: self.sessions[sid] = row; atomic_json(self.path, self.sessions)
        return sid, row.copy()


class ServiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, manager, sessions):
        if ":" in address[0]: self.address_family = socket.AF_INET6
        self.manager, self.sessions = manager, sessions
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: ServiceHTTPServer
    def setup(self): super().setup(); self.connection.settimeout(30)
    def log_message(self, fmt, *args): LOG.info("%s %s", self.client_address[0], fmt % args)

    def _send(self, code, body, ctype="application/json; charset=utf-8", *, cookie=None, download=None):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff"); self.send_header("X-Frame-Options", "DENY"); self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if cookie: self.send_header("Set-Cookie", f"{COOKIE_NAME}={cookie}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_SECONDS}")
        if download: self.send_header("Content-Disposition", f'attachment; filename="{download}"')
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
    def _body(self, limit):
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1: raise JobError("请求必须指定唯一的 Content-Length。", 411)
        try: length = int(lengths[0])
        except ValueError: raise JobError("无效的请求长度。", 400)
        if length < 0 or length > limit: raise JobError(f"请求过大；STL 上传上限为 {MAX_UPLOAD_BYTES // 1024 // 1024} MiB。", 413)
        body = self.rfile.read(length)
        if len(body) != length: raise JobError("上传未完成，请重新上传。", 400)
        return body
    def _payload(self):
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json": raise JobError("请求必须使用 application/json。", 415)
        try: value = json.loads(self._body(MAX_JSON_BYTES) or b"{}")
        except (ValueError, UnicodeError): raise JobError("请求内容不是有效 JSON。", 400)
        if not isinstance(value, dict): raise JobError("请求内容必须是 JSON 对象。", 400)
        return value
    def _handle(self, method):
        try: self._get() if method == "GET" else self._post()
        except JobError as e: self._json({"error": {"message": str(e)}}, e.status)
        except (BrokenPipeError, ConnectionResetError, TimeoutError): LOG.info("Client disconnected")
        except Exception:
            did = secrets.token_hex(6); LOG.exception("Request failed; diagnostic_id=%s", did); self._json({"error": {"message": "服务暂时无法完成请求，请重试或提供诊断编号给维护者。", "diagnostic_id": did}}, 500)
    def do_GET(self): self._handle("GET")
    def do_POST(self): self._handle("POST")

    def _get(self):
        path = self._path(); self._host()
        if path == "/api/session":
            sid, row, fresh = self._session(local_create=True); return self._json({"authenticated": bool(row), "shared": self.server.sessions.shared, "csrf_token": row["csrf_token"] if row else None, "max_upload_bytes": MAX_UPLOAD_BYTES}, cookie=sid if fresh else None)
        if path == "/" or path.startswith("/static/"):
            name = "index.html" if path == "/" else path.removeprefix("/static/"); file = contained_file(STATIC_DIR, name, STATIC_FILES)
            if not file: raise JobError("文件不存在。", 404)
            sid, _, fresh = self._session(local_create=path == "/"); ctype = {".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8"}.get(file.suffix, "application/octet-stream")
            return self._send(200, file.read_bytes(), ctype, cookie=sid if fresh else None)
        row = self._require_session(); manager = self.server.manager
        if path == "/api/jobs": return self._json({"jobs": manager.list(row["owner"])})
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})", path)
        if match: return self._json(manager.get(match[1], row["owner"]))
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/(?:files/([^/]+)|(report))", path); legacy = re.fullmatch(rf"/jobs/({JOB_ID_PATTERN})/([^/]+)", path)
        if match or legacy:
            job_id = (match or legacy)[1]; name = (match[2] or "report.html") if match else legacy[2]; job = manager.get(job_id, row["owner"])
            if job["status"] != "done": raise JobError("任务完成后才能打开报告和导出文件。", 409)
            job_dir = (manager.root / job_id).resolve()
            if job_dir.parent != manager.root: raise JobError("文件不存在。", 404)
            file = contained_file(job_dir, name, OUTPUT_FILES)
            if not file: raise JobError("文件不存在或未生成。", 404)
            ctype = {".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8", ".csv": "text/csv; charset=utf-8", ".vtp": "application/xml"}.get(file.suffix, "application/octet-stream")
            return self._send(200, file.read_bytes(), ctype, download=name if file.suffix not in {".html", ".json"} else None)
        raise JobError("接口不存在。", 404)

    def _post(self):
        path = self._path(); self._host()
        if path == "/api/session":
            self._origin(); payload = self._payload(); sid, row = self.server.sessions.create(payload.get("token")); return self._json({"authenticated": True, "shared": self.server.sessions.shared, "csrf_token": row["csrf_token"]}, cookie=sid)
        row = self._require_session(mutation=True); manager = self.server.manager
        if path in {"/api/jobs", "/api/upload"}:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("multipart/form-data;"): raise JobError("上传必须使用 multipart/form-data。", 415)
            body = self._body(MAX_UPLOAD_BYTES + 64 * 1024); message = BytesParser(policy=email_default).parsebytes(b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
            if not message.is_multipart(): raise JobError("上传表单格式无效。")
            fields, upload = {}, None
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name == "stl":
                    if upload is not None: raise JobError("每个任务只能上传一个 STL。")
                    upload = (part.get_filename() or "input.stl", part.get_payload(decode=True))
                elif name in {"case_id", "units", "remove_fragments"}:
                    data = part.get_payload(decode=True) or b""
                    if len(data) > 1024: raise JobError("上传表单字段过长。")
                    try: fields[name] = data.decode("utf-8").strip()
                    except UnicodeError: raise JobError("上传表单字段必须使用 UTF-8。")
            if not upload or not upload[1]: raise JobError("请选择 STL 文件。")
            if len(upload[1]) > MAX_UPLOAD_BYTES: raise JobError("STL 超过 128 MiB 上传上限。", 413)
            remove = fields.get("remove_fragments", "false").lower()
            if remove not in {"true", "false", "1", "0", "on", "off"}: raise JobError("表面修复选项无效。")
            # STL has no embedded unit metadata.  Omitted units therefore go
            # through the explicit input-confirmation page instead of being
            # silently treated as millimetres.
            job = manager.create(row["owner"], content=upload[1], filename=upload[0], case_id=fields.get("case_id", ""), units=fields.get("units", "auto"), remove_fragments=remove in {"true", "1", "on"})
            return self._json({"job": job, "job_id": job["id"]}, 201)
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/(input|confirm|cancel|retry)", path)
        if match: return self._json({"job": getattr(manager, match[2])(match[1], row["owner"], self._payload())})
        raise JobError("接口不存在。", 404)


def serve(host: str = "127.0.0.1", port: int = 8765, jobs_root: Path | None = None, device: str = "auto", token: str | None = None) -> None:
    root = Path(jobs_root or PROJECT_ROOT / "outputs/wss_deploy_jobs").resolve(); root.mkdir(parents=True, exist_ok=True); shared = not is_loopback(host); legacy_owner = os.environ.get("WSS_DEPLOY_LEGACY_OWNER") or None
    sessions = SessionStore(root / ".sessions.json", shared=shared, token=token or os.environ.get("WSS_DEPLOY_TOKEN"), legacy_owner=legacy_owner)
    from .infer import Release
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"); LOG.info("Loading release"); release = Release(device=device)
    manager = JobManager(root, release=release, legacy_owner=legacy_owner); httpd = ServiceHTTPServer((host, port), manager, sessions); manager.start(); LOG.info("Serving on http://%s:%s; jobs=%s; shared=%s", host, port, root, shared)
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally: httpd.server_close(); manager.close()
