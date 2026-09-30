"""Local-first STL deployment service with persistent, owner-scoped tasks.

v0.14 hardening (security review): sessions carry the credential they were opened with and die with it
(token fingerprint / ``credential_generation``, idle timeout, runtime pruning); uploads stream to spooled
temporary files; bundles are built on disk; per-owner, per-session and per-IP limits; per-route CSP; gzip +
ETag for reports and static files; request lines go to the ``wss_deploy.access`` logger (path only, patient
ids redacted) and deletions / claims to ``wss_deploy.audit``.

v0.15 (pre-launch hardening, all network-layer switches opt-in): ``WSS_DEPLOY_TRUST_PROXY=1`` believes
``X-Forwarded-For / -Proto / -Host`` from a loopback peer only (client address for throttling, audit and access
logs; scheme for the Origin check) and makes a loopback-bound service require login; ``WSS_DEPLOY_COOKIE_SECURE``
(auto / 1 / 0) adds ``Secure`` to the session cookie and HSTS is sent only on requests judged https;
``WSS_DEPLOY_UMASK`` sets the process umask for ``serve``.  Log files are created 0600.  Login attempts use two
budgets (per client address, default 30 / min; per user name, 10 / min) and 429 answers carry ``retry_after``.
"""
from __future__ import annotations

from collections import OrderedDict
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64, copy, gzip, hashlib, inspect, ipaddress, json, logging, math, os, re, secrets, shutil, signal, socket, tempfile, threading, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import queue

from . import __version__
from .jobs import FINAL, JOB_ID_PATTERN, JobError, JobManager, _date, atomic_json
from .paths import PROJECT_ROOT, STATIC_DIR
from .registry import ReleaseRegistry, ReleaseError
from .users import UserStore, valid_username

LOG = logging.getLogger("wss_deploy.service")
ACCESS = logging.getLogger("wss_deploy.access")   # request lines; cli ``serve`` routes it to access.log
AUDIT = logging.getLogger("wss_deploy.audit")     # delete / restore / purge / claim with actor and job id
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
                "batch_export.js", "report_common.js", "volume_viewer.js", "workbench_core.js", "glossary.json",
                "ops.html", "ops.js", "ops.css", "support.html", "support.js"}
# Responses that may be embedded by our own pages (side-by-side comparison, one-page preview).
EMBEDDABLE_HTML = {"report.html", "onepage.html"}
_STATIC_BUILD: dict = {}
# Workspace v2 (WORKSPACE_V2_CONTRACT.md §1–§2): files under static/v2/ are served from the lists in its bundle.json;
# ``generated`` files (the example offline report) only through their own route.
V2_DIR = STATIC_DIR / "v2"
V2_SERVED_LISTS = ("scripts", "styles", "pages", "assets", "dev")
V2_EXAMPLE = "example_report.html"
V2_EXAMPLE_ROUTE = "/v2/example"
MAX_OFFLINE_REQUEST_BYTES = 512 * 1024
MAX_OFFLINE_BOOKMARKS = 200
_V2_BUNDLE: dict = {}
STATIC_TYPES = {".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8",
                ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".stl": "model/stl"}


def v2_bundle() -> dict:
    """``static/v2/bundle.json`` (cached by its stat signature); ``{}`` when it is missing or not a JSON object."""
    path = V2_DIR / "bundle.json"
    try:
        stat = path.stat()
    except OSError:
        return {}
    signature = (stat.st_size, stat.st_mtime_ns)
    if _V2_BUNDLE.get("key") != signature:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            LOG.warning("static/v2/bundle.json is not readable JSON; the v2 static whitelist is empty")
            value = {}
        _V2_BUNDLE.update(key=signature, value=value if isinstance(value, dict) else {})
    return _V2_BUNDLE.get("value") or {}


def v2_static_files() -> set[str]:
    """Flat file names ``/static/v2/<name>`` may serve: bundle.json's scripts, styles, pages, assets and dev lists."""
    bundle = v2_bundle()
    names = set()
    for key in V2_SERVED_LISTS:
        value = bundle.get(key)
        if isinstance(value, list):
            names.update(name for name in value if isinstance(name, str) and name and "/" not in name and "\\" not in name
                         and name not in {".", ".."})
    return names


def _build_files(scope: str) -> list[Path]:
    if scope == "v2":
        names = sorted(v2_static_files() | {"bundle.json"})
        legacy = v2_bundle().get("legacy_scripts") if isinstance(v2_bundle().get("legacy_scripts"), list) else []
        return [V2_DIR / name for name in names] + [STATIC_DIR / name for name in sorted(set(legacy) & STATIC_FILES)]
    return [STATIC_DIR / name for name in sorted(STATIC_FILES)]


def static_build_id(scope: str = "classic") -> str:
    """Short content digest of the workbench's static files (v0.15).  Logged-in pages compare it with the one they
    booted with, so a redeploy that changes only static files (same ``__version__``) still prompts a reload.
    Cached by the files' stat signature; empty when the directory cannot be read.

    ``scope="v2"`` digests the workspace v2 files (bundle.json, everything it serves and the legacy scripts it loads)
    separately, so a v2-only change does not ask users of the classic workbench to reload (and vice versa)."""
    try:
        files = [path for path in _build_files(scope) if path.is_file()]
        if not files:
            return ""
        signature = tuple((str(path), os.stat(path).st_size, os.stat(path).st_mtime_ns) for path in files)
        cached = _STATIC_BUILD.get(scope) or {}
        if cached.get("key") != signature:
            digest = hashlib.sha256()
            for path in files:
                digest.update(path.name.encode("utf-8")); digest.update(b"\0"); digest.update(path.read_bytes())
            _STATIC_BUILD[scope] = {"key": signature, "value": digest.hexdigest()[:12]}
        return _STATIC_BUILD[scope].get("value", "")
    except OSError:
        return ""
OUTPUT_FILES = {"report.html", "summary.json", "run_manifest.json", "quality_audit.json", "wall_wss.vtp", "points_wss.csv", "field.npz",
                "volume_fields.vtp", "points_volume.csv", "wall_pressure.vtp", "streamlines.vtp",
                "annotations.json", "findings_review.json", "snapshots.json", "narrative.json"}
# One-page pictures are written by the report page itself; the whitelist stays strict (PNG only, fixed prefix).
SNAPSHOT_FILE = re.compile(r"snapshot_[a-z0-9_-]{1,32}\.png")
OWNER_KEY_PATTERN = r"[A-Za-z0-9_.-]{1,64}"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"   # D8: the offset makes server.log unambiguous next to +08:00 job times
LOG_MAX_BYTES = 20 * 1024 * 1024
LOG_BACKUPS = 5
STALE_CACHE_SECONDS = 60


def _env_number(name: str, default, *, minimum=0, cast=int):
    try:
        value = cast(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        LOG.warning("Ignoring invalid %s", name)
        return default
    return value if value >= minimum else default


# S1 sessions: idle timeout on top of the absolute lifetime; ``last_seen`` is written at most once a minute.
SESSION_IDLE_ENV = "WSS_DEPLOY_SESSION_IDLE_HOURS"
LAST_SEEN_RESOLUTION = 60
SESSION_PRUNE_EVERY = 256          # lookups between runtime sweeps of expired rows …
SESSION_PRUNE_SECONDS = 600        # … or at least this often
PENDING_SESSION_SECONDS = 600      # loopback sessions never presented back are dropped unwritten
MAX_PENDING_SESSIONS = 1000
# S5 resource limits (environment overrides are read when the server starts).
SPOOL_BYTES = 8 * 1024 * 1024
UPLOAD_CHUNK = 256 * 1024
MAX_BATCH_FILES = 20
UPLOAD_SOURCE_KEYS = frozenset({"content", "content_path", "content_file"})   # never taken from client metadata_json
MAX_PART_HEADER_BYTES = 16 * 1024
DEFAULT_CONCURRENT_UPLOADS = 2
DEFAULT_CONCURRENT_BUNDLES = 1
DEFAULT_MAX_BUNDLE_BYTES = 2 * 1024 ** 3
BUNDLE_WAIT_SECONDS = 10
DEFAULT_ACTIVE_JOBS_PER_OWNER = 20
DEFAULT_SSE_PER_SESSION = 6
# v0.15: per client address 30 / min (a research group shares one NAT address; one mistyping colleague must not lock the
# others out) plus per user name 10 / min; users.py additionally locks a name after 5 wrong passwords in 60 s.
DEFAULT_LOGIN_PER_MINUTE = 30
DEFAULT_LOGIN_PER_USER_MINUTE = 10
THROTTLE_REPORT_SECONDS = 60       # one ``login_throttled`` audit line per key and minute (no log flooding)
MAX_THROTTLED_CLIENTS = 4096
# v0.15 reverse proxy / TLS readiness: every switch is off by default and read once when the server starts.
TRUST_PROXY_ENV = "WSS_DEPLOY_TRUST_PROXY"
COOKIE_SECURE_ENV = "WSS_DEPLOY_COOKIE_SECURE"
UMASK_ENV = "WSS_DEPLOY_UMASK"
HSTS_VALUE = "max-age=31536000"
TMP_DIR_NAME = ".tmp"
TMP_MAX_AGE_SECONDS = 3600
# S10 compression / caching.
GZIP_MIN_BYTES = 8 * 1024
GZIP_LEVEL = 6
DEFAULT_GZIP_CACHE_MB = 256
GZIP_TYPES = ("text/html", "application/javascript", "text/css", "application/json")
STATIC_CACHE = "private, max-age=0, must-revalidate"
REPORT_CACHE = "private, no-cache"
# S6 per-route Content-Security-Policy.  Workbench pages carry no inline script; the 3-D reports are
# self-contained pages with inline three.js + viewer (hashes computed from the served page would allow
# whatever it contains, and fixed hashes would break every older report), so only they keep
# 'unsafe-inline'.  The one-page report runs no script except its print button's handler.
ONEPAGE_HANDLER_HASH = "'sha256-" + base64.b64encode(hashlib.sha256(b"window.print()").digest()).decode("ascii") + "'"
CSP_SCRIPT = {"workbench": "script-src 'self'", "report": "script-src 'self' 'unsafe-inline'",
              "onepage": "script-src 'unsafe-hashes' " + ONEPAGE_HANDLER_HASH}
CSP_REST = "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'none'"
PATIENT_PATH = re.compile(r"^/api/patients/[^/]*")
_SSE_CLOSE = {"__close__": True}   # pushed into a stream's queue when a newer stream of the session replaces it


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost": return True
    try: return ipaddress.ip_address(host).is_loopback
    except ValueError: return False


def _loopback_ip(text) -> bool:
    """A literal loopback address (IPv4-mapped IPv6 included); host names never count."""
    try:
        address = ipaddress.ip_address(str(text).strip())
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped or address).is_loopback


def trust_proxy_enabled() -> bool:
    """``WSS_DEPLOY_TRUST_PROXY=1``: ``X-Forwarded-*`` from a loopback peer (the local reverse proxy) are believed."""
    return str(os.environ.get(TRUST_PROXY_ENV) or "").strip().lower() in {"1", "true", "yes", "on"}


def service_is_shared(host: str) -> bool:
    """Shared (login) mode: a non-loopback bind, or any bind behind a trusted reverse proxy (v0.15)."""
    return not is_loopback(host) or trust_proxy_enabled()


def cookie_secure_mode() -> str:
    """``WSS_DEPLOY_COOKIE_SECURE``: ``auto`` (default: Secure only on requests a trusted proxy marks https),
    ``always`` (1) or ``never`` (0)."""
    raw = str(os.environ.get(COOKIE_SECURE_ENV) or "auto").strip().lower()
    if raw in {"1", "true", "yes", "on", "always"}:
        return "always"
    if raw in {"0", "false", "no", "off", "never"}:
        return "never"
    if raw != "auto":
        LOG.warning("Ignoring invalid %s (use auto, 1 or 0)", COOKIE_SECURE_ENV)
    return "auto"


def forwarded_client(peer: str, values) -> str:
    """Client address behind a trusted proxy: the right-most ``X-Forwarded-For`` entry that is not a loopback hop.

    Entries further left were written by the client itself and are never believed; an unparsable hop ends the
    chain (the peer address is used).  ``values`` are all X-Forwarded-For header values in order."""
    hops = [item.strip() for value in values or [] for item in str(value).split(",") if item.strip()]
    for item in reversed(hops):
        try:
            address = ipaddress.ip_address(item)
        except ValueError:
            return peer
        if not _loopback_ip(item):
            return str(address)
    return peer


def forwarded_proto(values) -> str | None:
    """The scheme set by the nearest proxy (last ``X-Forwarded-Proto`` value), only ``http`` / ``https``."""
    items = [item.strip().lower() for value in values or [] for item in str(value).split(",") if item.strip()]
    return items[-1] if items and items[-1] in {"http", "https"} else None


def valid_host_value(host) -> bool:
    """``host[:port]`` as a Host / X-Forwarded-Host value (printable ASCII; no user info, path, list or query)."""
    host = str(host or "")
    if not host or any(ord(c) < 33 or ord(c) > 126 or c in ",/\\@?#" for c in host):
        return False
    try:
        parsed = urlsplit("http://" + host)
        parsed.port                       # ValueError for a malformed port
    except ValueError:
        return False
    return bool(parsed.hostname) and parsed.path == ""


def apply_umask_from_env() -> int | None:
    """``WSS_DEPLOY_UMASK`` (octal, e.g. ``077``) → ``os.umask``; unset = unchanged.  Returns the previous mask."""
    raw = str(os.environ.get(UMASK_ENV) or "").strip()
    if not raw:
        return None
    try:
        mask = int(raw, 8)
    except ValueError:
        mask = -1
    if not 0 <= mask <= 0o777:
        LOG.warning("Ignoring invalid %s (octal such as 077)", UMASK_ENV)
        return None
    return os.umask(mask)


class PrivateRotatingFileHandler(RotatingFileHandler):
    """Rotating log created 0600 whatever the umask (server.log carries the audit trail, access.log client
    addresses); an existing file of ours is narrowed to 0600 when it is opened."""
    def _open(self):
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except OSError:
            pass
        return os.fdopen(fd, "a", encoding=self.encoding, errors=self.errors)


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


def access_path(raw) -> str:
    """Request path for the access log: no query string, ``/api/patients/<id>/…`` redacted, printable only."""
    try:
        path = urlsplit(str(raw or "")).path or "/"
    except ValueError:
        return "<invalid>"
    path = PATIENT_PATH.sub("/api/patients/<id>", path)
    return re.sub(r"[^\x21-\x7e]", "?", path)[:512]


def redact_log_text(text) -> str:
    """Server-generated error lines (bad request line …) may quote the raw request: drop queries and patient ids."""
    text = re.sub(r"/api/patients/[^/\s'\"]*", "/api/patients/<id>", str(text))
    return re.sub(r"\?[^\s'\"]*", "", text)


def etag_matches(header: str | None, etag: str) -> bool:
    """Weak comparison of ``If-None-Match`` (a list or ``*``) with our ETag."""
    if not header:
        return False
    bare = etag[2:] if etag.startswith("W/") else etag
    for item in header.split(","):
        item = item.strip()
        if item == "*" or (item[2:] if item.startswith("W/") else item) == bare:
            return True
    return False


def accepts_gzip(header: str | None) -> bool:
    """``Accept-Encoding`` lists gzip (or ``*``) with a non-zero q value."""
    q_gzip = q_any = None
    for item in (header or "").split(","):
        name, _, params = item.strip().partition(";")
        name = name.strip().lower()
        match = re.search(r"q\s*=\s*([0-9.]+)", params)
        try: q = float(match[1]) if match else 1.0
        except ValueError: q = 0.0
        if name in {"gzip", "x-gzip"}:
            q_gzip = q if q_gzip is None else max(q_gzip, q)
        elif name == "*":
            q_any = q
    if q_gzip is not None:
        return q_gzip > 0
    return bool(q_any)


def strip_admin_detail(value):
    """Copy of a JSON value without any ``admin_detail`` key (typed pipeline errors, contract in errors.py)."""
    if isinstance(value, dict):
        return {key: strip_admin_detail(item) for key, item in value.items() if key != "admin_detail"}
    if isinstance(value, list):
        return [strip_admin_detail(item) for item in value]
    return value


def public_error_text(text) -> str:
    """Release errors can name server paths (release root, missing file): keep the last component only."""
    return re.sub(r"(?<![\w.])/(?:[^\s/:：，。;,'\"()（）]+/)+([^\s/:：，。;,'\"()（）]*)", lambda m: m[1] or "…", str(text))


def job_etag(stat) -> str:
    return f'W/"{stat.st_ino:x}-{stat.st_size:x}-{stat.st_mtime_ns:x}"'


class GzipCache:
    """Gzipped bytes of served files keyed by (path, inode, size, mtime), LRU-bounded by total compressed size."""
    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes)); self.lock = threading.Lock()
        self.items: OrderedDict = OrderedDict(); self.by_path: dict[str, tuple] = {}; self.total = 0
    def get(self, path: Path, stream, stat) -> bytes:
        key = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        with self.lock:
            hit = self.items.get(key)
            if hit is not None:
                self.items.move_to_end(key); return hit
        data = gzip.compress(stream.read(), GZIP_LEVEL, mtime=0)
        if len(data) <= self.max_bytes:
            with self.lock:
                old = self.by_path.pop(key[0], None)
                if old is not None and old in self.items: self.total -= len(self.items.pop(old))
                self.items[key] = data; self.by_path[key[0]] = key; self.total += len(data)
                while self.total > self.max_bytes and self.items:
                    evicted, blob = self.items.popitem(last=False); self.total -= len(blob)
                    if self.by_path.get(evicted[0]) == evicted: self.by_path.pop(evicted[0], None)
        return data


class LoginThrottle:
    """Per-client-IP token bucket for login attempts (both token and password logins)."""
    def __init__(self, per_minute: int, *, clock=time.monotonic, max_clients: int = MAX_THROTTLED_CLIENTS):
        self.capacity = float(max(1, per_minute)); self.rate = self.capacity / 60.0; self.clock = clock
        self.max_clients = max_clients; self.lock = threading.Lock(); self.buckets: OrderedDict = OrderedDict()
    def take(self, client: str) -> bool:
        now = self.clock()
        with self.lock:
            tokens, at = self.buckets.pop(client, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - at) * self.rate)
            allowed = tokens >= 1.0
            self.buckets[client] = (tokens - 1.0 if allowed else tokens, now)
            while len(self.buckets) > self.max_clients:
                self.buckets.popitem(last=False)
            return allowed
    def refund(self, client: str) -> None:
        """A successful login gives its token back: only failures use up the budget."""
        with self.lock:
            if client in self.buckets:
                tokens, at = self.buckets[client]; self.buckets[client] = (min(self.capacity, tokens + 1.0), at)
    def retry_after(self, client: str) -> int:
        """Whole seconds until ``client`` has a token again (0 = now)."""
        now = self.clock()
        with self.lock:
            tokens, at = self.buckets.get(client, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - at) * self.rate)
        return 0 if tokens >= 1.0 else max(1, math.ceil((1.0 - tokens) / self.rate))
    def first_refusal(self, client: str, window: float = THROTTLE_REPORT_SECONDS) -> bool:
        """True at most once per ``window`` per key: refusals are audited without flooding the log."""
        now = self.clock()
        with self.lock:
            reported = self.__dict__.setdefault("reported", OrderedDict())
            last = reported.pop(client, None)
            if last is not None and now - last < window:
                reported[client] = last
                return False
            reported[client] = now
            while len(reported) > self.max_clients:
                reported.popitem(last=False)
            return True


def disposition_params(value: str) -> dict:
    """Parameters of a part's Content-Disposition (quoted strings, RFC 5987 ``filename*``)."""
    params = {}
    for match in re.finditer(r';\s*([A-Za-z0-9_*.-]+)\s*=\s*("(?:[^"\\]|\\.)*"|[^;]*)', value):
        raw = match[2].strip()
        if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            raw = re.sub(r"\\(.)", r"\1", raw[1:-1])
        params.setdefault(match[1].lower(), raw)
    if "filename*" in params:
        charset, _, rest = params["filename*"].partition("'")
        _, _, encoded = rest.partition("'")
        try: params["filename"] = unquote(encoded, encoding=charset or "utf-8", errors="strict")
        except (LookupError, UnicodeError): pass
    return params


def stream_multipart(read, length: int, boundary: bytes, start_part) -> None:
    """Stream a ``multipart/form-data`` body of exactly ``length`` bytes without holding it in memory.

    ``start_part(name, filename)`` returns a ``write(bytes)`` callable (or None to discard the part); it may
    raise JobError to stop early.  At most one chunk plus one delimiter is buffered at any time.
    """
    delimiter = b"\r\n--" + boundary
    keep = len(delimiter) - 1
    remaining = length
    buffer = bytearray(b"\r\n")    # the opening boundary then matches the same delimiter as the others

    def fill() -> bool:
        nonlocal remaining
        if remaining <= 0:
            return False
        chunk = read(min(UPLOAD_CHUNK, remaining))
        if not chunk:
            raise JobError("上传未完成，请重新上传。", 400)
        remaining -= len(chunk); buffer.extend(chunk)
        return True

    def invalid():
        return JobError("上传表单格式无效。")

    while True:                                   # preamble
        index = buffer.find(delimiter)
        if index >= 0:
            del buffer[:index + len(delimiter)]; break
        if len(buffer) > keep: del buffer[:len(buffer) - keep]
        if not fill(): raise invalid()
    while True:
        while len(buffer) < 4 and fill():
            pass
        if buffer[:2] == b"--":                  # closing delimiter; the epilogue is read and ignored
            buffer.clear()
            while fill(): buffer.clear()
            return
        while True:                               # "\r\n" + headers + "\r\n\r\n"
            end = buffer.find(b"\r\n\r\n")
            if end >= 0: break
            if len(buffer) > MAX_PART_HEADER_BYTES: raise JobError("上传表单头部过长。")
            if not fill(): raise invalid()
        if not buffer.startswith(b"\r\n") and end != 0: raise invalid()
        block = bytes(buffer[2:end]) if end else b""
        del buffer[:end + 4]
        name = filename = None
        for line in block.decode("utf-8", "replace").split("\r\n"):
            key, _, value = line.partition(":")
            if key.strip().lower() == "content-disposition":
                params = disposition_params(value); name, filename = params.get("name"), params.get("filename")
        write = start_part(name, filename)
        while True:                               # body up to the next delimiter
            index = buffer.find(delimiter)
            if index >= 0:
                if index and write: write(bytes(buffer[:index]))
                del buffer[:index + len(delimiter)]; break
            if len(buffer) > keep:
                if write: write(bytes(buffer[:len(buffer) - keep]))
                del buffer[:len(buffer) - keep]
            if not fill(): raise invalid()


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
    """Cookie sessions (``.sessions.json``).

    S1 (v0.14): a shared-mode row records the credential it was opened with — ``token_fingerprint`` for a
    legacy token session, ``credential_generation`` (users.json) for a password session — and ``lookup``
    rejects it once that credential changed, the user was disabled or their role changed, or (legacy rows)
    the token login window closed.  Rows also end after ``WSS_DEPLOY_SESSION_IDLE_HOURS`` (default 12 h)
    without a request; expired rows are swept at runtime.  Loopback sessions are written only once the
    browser presents the cookie back, so cookie-less GETs cost no fsync.
    """
    def __init__(self, path: Path, *, shared: bool, token: str | None = None, legacy_owner: str | None = None, users: UserStore | None = None,
                 clock=None):
        if shared and not token and not (users and users.exists()):
            raise ValueError("共享访问需要设置 WSS_DEPLOY_TOKEN 或先用 `python -m wss_deploy.cli user add` 创建用户；单人使用请绑定 127.0.0.1。")
        if shared and legacy_owner: raise ValueError("WSS_DEPLOY_LEGACY_OWNER 仅允许在本地回环模式使用。")
        self.path, self.shared, self.token, self.legacy_owner, self.users = Path(path), shared, token, legacy_owner, users
        self.clock = clock or time.time
        self.idle_seconds = _env_number(SESSION_IDLE_ENV, 12.0, minimum=0.01, cast=float) * 3600
        self.lock = threading.RLock(); self.sessions = {}
        self._lookups = 0; self._last_prune = self.clock(); self._last_flush = 0.0; self._dirty = False
        if self.path.exists():
            try: self.sessions = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception: LOG.warning("Ignoring invalid sessions file")
        if not isinstance(self.sessions, dict): self.sessions = {}
        now = self.clock()
        self.sessions = {k: v for k, v in self.sessions.items()
                         if isinstance(v, dict) and v.get("expires_at", 0) > now and v.get("shared") is shared}
        for row in self.sessions.values():
            row.setdefault("last_seen", now)       # rows written before v0.14: the idle clock starts now

    def token_fingerprint(self) -> str | None:
        return hashlib.sha256(self.token.encode("utf-8")).hexdigest()[:16] if self.token else None

    def login_mode(self) -> str:
        """``none`` on loopback (login-free), ``password`` once users exist, ``token`` before that."""
        if not self.shared: return "none"
        return "password" if self.users is not None and self.users.exists() else "token"
    def legacy_token_allowed(self) -> bool:
        if not self.shared or not self.token: return False
        return self.login_mode() == "token" or os.environ.get("WSS_DEPLOY_ALLOW_LEGACY_TOKEN") == "1"

    # -- persistence ------------------------------------------------------------------------------
    def _persist(self) -> None:
        """Write every confirmed row (caller holds the lock)."""
        atomic_json(self.path, {sid: row for sid, row in self.sessions.items() if not row.get("pending")})
        self._dirty = False; self._last_flush = self.clock()

    def _touch(self, row: dict, now: float) -> None:
        if now - float(row.get("last_seen") or 0) >= LAST_SEEN_RESOLUTION:
            row["last_seen"] = now; self._dirty = True
        if self._dirty and now - self._last_flush >= LAST_SEEN_RESOLUTION:
            self._persist()

    def _problem(self, row: dict, now: float, *, for_claim: bool = False) -> str | None:
        """Why a row cannot authenticate: ``expired`` / ``revoked`` (both deleted) or ``disallowed`` (kept)."""
        if float(row.get("expires_at") or 0) <= now: return "expired"
        if now - float(row.get("last_seen") or now) > self.idle_seconds: return "expired"
        if row.get("pending") and now - float(row.get("created_at") or now) > PENDING_SESSION_SECONDS: return "expired"
        if not self.shared: return None
        if row.get("username"):
            try: current = self.users.credential(row["username"]) if self.users is not None else None
            except Exception:
                LOG.warning("users.json unreadable; refusing password sessions until it is fixed")
                return "disallowed"
            if (current is None or current["disabled"] or current["role"] != row.get("role")
                    or current["generation"] != row.get("credential_generation", 0)):
                return "revoked"
            return None
        if row.get("role") == "legacy":
            if not row.get("token_fingerprint") or row.get("token_fingerprint") != self.token_fingerprint(): return "revoked"
            if not for_claim and not self.legacy_token_allowed(): return "disallowed"
            return None
        return "revoked"

    def prune(self, now: float | None = None) -> int:
        """Drop expired / revoked rows now; returns how many went."""
        now = self.clock() if now is None else now
        with self.lock:
            dead = [sid for sid, row in self.sessions.items() if self._problem(row, now, for_claim=True) in {"expired", "revoked"}]
            written = any(not self.sessions[sid].get("pending") for sid in dead)
            for sid in dead: self.sessions.pop(sid, None)
            self._last_prune = now
            if written: self._persist()
            return len(dead)

    def lookup(self, cookie: str | None, *, for_claim: bool = False):
        """``(sid, row copy)`` of a live session, else None.

        ``for_claim`` (the login route only) also returns a legacy row whose token window has closed, so the
        legacy → named-user claim offer survives the switch to password login.
        """
        try:
            parsed = SimpleCookie(); parsed.load(cookie or ""); sid = parsed[COOKIE_NAME].value
        except (KeyError, ValueError): return None
        now = self.clock()
        with self.lock:
            self._lookups += 1
            if self._lookups % SESSION_PRUNE_EVERY == 0 or now - self._last_prune >= SESSION_PRUNE_SECONDS:
                self.prune(now)
            row = self.sessions.get(sid)
            if not row: return None
            problem = self._problem(row, now, for_claim=for_claim)
            if problem in {"expired", "revoked"}:
                self.sessions.pop(sid, None)
                if not row.get("pending"): self._persist()
                return None
            if problem: return None
            if row.pop("pending", None):
                row["last_seen"] = now; self._persist()     # the browser kept the cookie: now it is worth a write
            else:
                self._touch(row, now)
            return sid, row.copy()

    def _new_row(self, owner: str, **extra) -> dict:
        now = self.clock()
        return {"owner": owner, "csrf_token": secrets.token_urlsafe(32), "expires_at": now + SESSION_SECONDS, "shared": self.shared,
                "username": None, "role": None, "display_name": None, "claimable_owners": [], "created_at": now, "last_seen": now, **extra}

    def _registered(self, owner: str) -> bool:
        try: return self.users is not None and owner in self.users.cached()
        except Exception: return True

    def create(self, token: str | None = None, *, username=None, password=None, previous: dict | None = None, claimable_jobs=None):
        """Open a session: loopback → anonymous; shared → username/password (users.json) or the legacy token."""
        if not self.shared:
            row = self._new_row(self.legacy_owner or secrets.token_urlsafe(24), pending=True)
        elif username is not None or password is not None:
            if self.users is None or self.login_mode() != "password": raise JobError("当前服务未启用用户名登录。", 401)
            user, generation = self.users.authenticate(username, password)
            claimable = []
            # S2: only an anonymous (legacy token) session's jobs may follow the login; a named user's never do.
            if previous and previous.get("owner") and previous["owner"] != user["username"] and previous.get("username") is None:
                claimable = [owner for owner in dict.fromkeys([*previous.get("claimable_owners", []), previous["owner"]])
                             if not self._registered(owner)]
            row = self._new_row(user["username"], username=user["username"], role=user["role"], display_name=user["display_name"],
                                claimable_owners=claimable, credential_generation=generation)
        else:
            if not self.legacy_token_allowed(): raise JobError("请使用用户名和口令登录。", 401)
            if not isinstance(token, str) or not secrets.compare_digest(token.encode("utf-8", "surrogatepass"), (self.token or "").encode("utf-8")): raise JobError("访问口令不正确。", 401)
            row = self._new_row(secrets.token_urlsafe(24), role="legacy", token_fingerprint=self.token_fingerprint())
        sid = secrets.token_urlsafe(32)
        with self.lock:
            self.sessions[sid] = row
            if row.get("pending"):
                pending = [key for key, value in self.sessions.items() if value.get("pending")]
                for key in pending[:max(0, len(pending) - MAX_PENDING_SESSIONS)]: self.sessions.pop(key, None)
            else:
                self._persist()
        return sid, row.copy()

    def destroy(self, sid: str) -> None:
        with self.lock:
            row = self.sessions.pop(sid, None)
            if row is not None and not row.get("pending"): self._persist()

    def renew(self, sid: str):
        """After a user's own password change: replace this session by one bound to the new credential generation
        (same owner, role and CSRF token, so the open page keeps working); None when the user can no longer log in."""
        with self.lock:
            row = self.sessions.pop(sid, None)
            current = self.users.credential(row["username"]) if row and row.get("username") and self.users is not None else None
            if current is None or current["disabled"]:
                if row is not None: self._persist()
                return None
            fresh = dict(row, credential_generation=current["generation"], last_seen=self.clock())
            new_sid = secrets.token_urlsafe(32); self.sessions[new_sid] = fresh; self._persist()
            return new_sid, fresh.copy()

    def drop_claimable(self, sid: str, owner: str) -> None:
        with self.lock:
            row = self.sessions.get(sid)
            if row and owner in row.get("claimable_owners", []):
                row["claimable_owners"] = [o for o in row["claimable_owners"] if o != owner]; self._persist()

    def describe(self, row: dict | None, *, manager=None) -> dict:
        """Public session facts for ``GET/POST /api/session``."""
        out = {"authenticated": bool(row), "shared": self.shared, "csrf_token": row["csrf_token"] if row else None,
               "max_upload_bytes": MAX_UPLOAD_BYTES, "max_batch_bytes": MAX_BATCH_BYTES, "max_batch_files": MAX_BATCH_FILES,
               "login": self.login_mode(), "legacy_token_allowed": self.legacy_token_allowed(),
               "username": row.get("username") if row else None, "display_name": row.get("display_name") if row else None,
               "role": row.get("role") if row else None,
               # S2: rows written before v0.14 may list a registered user here; never offer those.
               "claimable_owners": [o for o in row.get("claimable_owners", []) if not self._registered(o)] if row else []}
        if row and manager is not None and out["claimable_owners"]:
            out["claimable"] = [{"owner": owner, "jobs": manager.count_owner(owner)} for owner in out["claimable_owners"]]
        return out


class ServiceHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, manager, sessions, preferences: PreferenceStore | None = None, *, operations=None):
        if ":" in address[0]: self.address_family = socket.AF_INET6
        self.manager, self.sessions = manager, sessions
        # The operations store is intentionally separate from job.json and the 30-day trash.  It contains
        # support tickets, a persistent audit journal and explicitly archived STL assets only.
        from .operations import OperationsStore
        self.operations = operations or OperationsStore(Path(manager.root))
        manager.audit_sink = self.operations.record_event
        self.preferences = preferences or PreferenceStore(Path(manager.root) / "preferences")
        self.started_ts = time.time()
        self._stale_cache: tuple[float, int | None] = (0.0, None)
        # S5 limits and S10 cache; environment overrides are read once, here.
        self.upload_slots = threading.BoundedSemaphore(_env_number("WSS_DEPLOY_MAX_CONCURRENT_UPLOADS", DEFAULT_CONCURRENT_UPLOADS, minimum=1))
        self.bundle_slots = threading.BoundedSemaphore(_env_number("WSS_DEPLOY_MAX_CONCURRENT_BUNDLES", DEFAULT_CONCURRENT_BUNDLES, minimum=1))
        self.max_bundle_bytes = _env_number("WSS_DEPLOY_MAX_BUNDLE_BYTES", DEFAULT_MAX_BUNDLE_BYTES, minimum=1)
        self.max_active_jobs = _env_number("WSS_DEPLOY_MAX_ACTIVE_JOBS_PER_OWNER", DEFAULT_ACTIVE_JOBS_PER_OWNER, minimum=1)
        self.max_sse_per_session = _env_number("WSS_DEPLOY_MAX_SSE_PER_SESSION", DEFAULT_SSE_PER_SESSION, minimum=1)
        self.login_throttle = LoginThrottle(_env_number("WSS_DEPLOY_LOGIN_RATE_PER_MIN", DEFAULT_LOGIN_PER_MINUTE, minimum=1))
        self.user_login_throttle = LoginThrottle(_env_number("WSS_DEPLOY_LOGIN_USER_RATE_PER_MIN", DEFAULT_LOGIN_PER_USER_MINUTE, minimum=1))
        # v0.15 reverse proxy / TLS readiness (opt-in; defaults keep v0.14 behaviour).
        self.trust_proxy = trust_proxy_enabled()
        self.cookie_secure = cookie_secure_mode()
        self.gzip_cache = GzipCache(_env_number("WSS_DEPLOY_GZIP_CACHE_MB", DEFAULT_GZIP_CACHE_MB, minimum=0) * 1024 * 1024)
        self.sse_lock = threading.Lock(); self.sse_streams: dict[str, list[dict]] = {}
        # Workspace v2: serialised (and gzipped) manifests by (job, ETag, encoding); the ETag covers every input.
        self.v2_bodies: OrderedDict = OrderedDict(); self.v2_bodies_lock = threading.Lock()
        self.tmp_dir = Path(manager.root) / TMP_DIR_NAME
        self._clean_tmp()
        super().__init__(address, Handler)

    def handle_error(self, request, client_address):
        """O6: an exception that escaped a request thread (e.g. an event stream after its headers were sent) goes to
        server.log with a diagnostic id instead of the console; only that connection ends."""
        LOG.exception("Request thread failed; diagnostic_id=%s", secrets.token_hex(6))

    def _clean_tmp(self) -> None:
        """Spooled uploads and bundle files live in ``<jobs root>/.tmp``; leftovers of a crashed process go."""
        try:
            self.tmp_dir.mkdir(mode=0o700, exist_ok=True)
            cutoff = time.time() - TMP_MAX_AGE_SECONDS
            for path in self.tmp_dir.iterdir():
                try:
                    if path.is_file() and path.stat().st_mtime < cutoff: path.unlink()
                except OSError: pass
        except OSError:
            LOG.warning("Cannot prepare %s; temporary files go to the system temp directory", self.tmp_dir)

    def temp_dir(self) -> str | None:
        return str(self.tmp_dir) if self.tmp_dir.is_dir() else None

    def sse_open(self, sid: str | None) -> dict:
        """Register a stream of this session; beyond the cap the session's oldest stream is told to close."""
        entry = {"closed": threading.Event(), "listener": None, "at": time.monotonic()}
        with self.sse_lock:
            streams = self.sse_streams.setdefault(sid or "", [])
            streams.append(entry)
            while len(streams) > self.max_sse_per_session:
                oldest = streams.pop(0); oldest["closed"].set()
                listener = oldest.get("listener")
                if listener is not None:
                    try: listener.put_nowait(_SSE_CLOSE)
                    except queue.Full: pass
        return entry

    def sse_close(self, sid: str | None, entry: dict) -> None:
        with self.sse_lock:
            streams = self.sse_streams.get(sid or "")
            if streams and entry in streams: streams.remove(entry)
            if streams is not None and not streams: self.sse_streams.pop(sid or "", None)

    def stale_reports(self) -> int | None:
        """Finished jobs whose report is not the current template (§19.7); recounted at most once a minute."""
        at, value = self._stale_cache
        if time.time() - at < STALE_CACHE_SECONDS:
            return value
        try:
            from .report_freshness import stale_jobs
            value = len(stale_jobs(self.manager.root, self.manager.done_job_dirs()))
        except Exception:
            LOG.exception("Cannot count stale reports")
            value = None
        self._stale_cache = (time.time(), value)
        return value

    def manager_health(self) -> dict | None:
        """``JobManager.health()`` (v0.14 contract) when the manager has it; a failing probe reports not-ok."""
        probe = getattr(self.manager, "health", None)
        if not callable(probe):
            return None
        try:
            value = probe()
        except Exception:
            LOG.exception("Health probe failed")
            return {"ok": False, "checks": {"probe": {"ok": False, "message": "健康检查执行失败，详见服务日志。"}}}
        return value if isinstance(value, dict) else {"ok": bool(value), "checks": {}}

    def health(self, row: dict | None) -> dict:
        """``GET /api/health`` (§19.6): ``ok`` + version only without a session; queue, worker, GPU, disk and the
        manager's checks with one.  ``ok`` is the manager's verdict when it provides ``health()``."""
        probe = self.manager_health()
        out = {"ok": bool(probe.get("ok")) if probe is not None else True, "version": __version__}
        if not row:
            return out
        out["ui_build"] = static_build_id()   # v0.15: lets an open page notice a static-only redeploy
        out["ui_build_v2"] = static_build_id("v2")   # the same signal for the workspace v2 files
        if probe is not None:
            checks = probe.get("checks") or {}
            if self.sessions.shared and row.get("role") != "admin" and isinstance(checks, dict):
                # v0.15: server paths are for administrators (like S9 error details).
                checks = {name: ({key: value for key, value in check.items() if key != "path"} if isinstance(check, dict) else check)
                          for name, check in checks.items()}
            out.update(checks=checks, cached_at=probe.get("cached_at"))
        from .service import gpu_summary
        manager = self.manager
        registry = getattr(manager, "registry", None)
        try:
            disk = round(shutil.disk_usage(manager.root).free / 1024 ** 3, 1)
        except OSError:
            disk = None
        gpu = gpu_summary()
        from .report_freshness import auto_enabled
        out.update(started_at=_date(self.started_ts),
                   uptime_s=round(time.time() - self.started_ts), queue=manager.queue_counts(),
                   queue_mine=manager.queue_counts(row.get("owner")), worker_alive=manager.worker_alive(),
                   gpu={"available": gpu.get("available", False), "name": gpu.get("name"), "count": gpu.get("count", 0),
                        "device": getattr(registry, "device", None)},
                   disk_free_gb=disk, default_release=getattr(registry, "default_id", None),
                   stale_reports=self.stale_reports(), auto_refresh_reports=auto_enabled(), report_code_changed=_report_code_changed(),
                   shared=self.sessions.shared, login=self.sessions.login_mode())
        return out


class Handler(BaseHTTPRequestHandler):
    server: ServiceHTTPServer
    server_version = "wss-deploy"
    sys_version = ""
    def setup(self): super().setup(); self.connection.settimeout(30)

    # ------------------------------------------------------------------ logging (S7)
    def log_message(self, fmt, *args):
        # Only server-generated error lines reach here (request lines go to the access logger).
        LOG.info("%s %s", self._client_ip(), redact_log_text(fmt % args))
    def send_response(self, code, message=None):
        """No ``Server:`` banner (S6); the status is kept for the access line written when the request ends."""
        self._status = code
        if not getattr(self, "_in_handle", False):
            self._access_line(code)
        self.send_response_only(code, message)
        self.send_header("Date", self.date_time_string())
    def _access_line(self, code=None) -> None:
        started = getattr(self, "_t0", None)
        took = f"{(time.monotonic() - started) * 1000:.0f}ms" if started is not None else "-"
        ACCESS.info("%s %s %s %s %s %s", self._client_ip(), getattr(self, "command", None) or "-",
                    access_path(getattr(self, "path", "")), code or getattr(self, "_status", None) or "-",
                    getattr(self, "_sent", 0), took)

    # ------------------------------------------------------------------ reverse proxy (v0.15, opt-in)
    def _header_values(self, name: str) -> list[str]:
        headers = getattr(self, "headers", None)
        try: return list(headers.get_all(name) or []) if headers is not None else []
        except Exception: return []
    def _peer_trusted(self) -> bool:
        """``X-Forwarded-*`` are believed only with WSS_DEPLOY_TRUST_PROXY=1 and a loopback TCP peer (the local proxy)."""
        return bool(getattr(self.server, "trust_proxy", False)) and _loopback_ip(self.client_address[0])
    def _client_ip(self) -> str:
        """Client address for throttling, audit and access logs (the TCP peer unless a trusted proxy forwarded it)."""
        peer = self.client_address[0]
        if not self._peer_trusted():
            return peer
        return forwarded_client(peer, self._header_values("X-Forwarded-For"))
    def _forwarded_proto(self) -> str | None:
        return forwarded_proto(self._header_values("X-Forwarded-Proto")) if self._peer_trusted() else None
    def _https(self) -> bool:
        """The browser talks https to us: only a trusted proxy can say so (this server itself speaks plain http)."""
        return self._forwarded_proto() == "https"
    def _cookie_secure(self) -> bool:
        mode = getattr(self.server, "cookie_secure", "auto")
        return mode == "always" or (mode == "auto" and self._https())
    def _allowed_origins(self, host: str) -> set[str]:
        """Origins accepted for mutations: the Host (and a trusted X-Forwarded-Host) with the scheme a trusted proxy
        reports; without a trusted proxy both http and https (v0.14 behaviour)."""
        hosts = {host}
        schemes = ("http", "https")
        if self._peer_trusted():
            forwarded = [item.strip() for value in self._header_values("X-Forwarded-Host") for item in value.split(",") if item.strip()]
            if forwarded and valid_host_value(forwarded[-1]):
                hosts.add(forwarded[-1])
            proto = self._forwarded_proto()
            if proto:
                schemes = (proto,)
        return {f"{scheme}://{name}" for scheme in schemes for name in hosts}

    # ------------------------------------------------------------------ responses
    def _gzip_ok(self) -> bool:
        return accepts_gzip(self.headers.get("Accept-Encoding"))
    def _write_headers(self, code, ctype, length, *, cookie=None, download=None, headers=None, embeddable=False, clear_cookie=False,
                       csp="workbench", cache="no-store", etag=None, encoding=None, vary=False):
        # ``embeddable`` responses (reports, one-page previews) may be framed by our own origin only;
        # everything else keeps ``frame-ancestors 'none'``.
        self.send_response(code)
        if ctype: self.send_header("Content-Type", ctype)
        if length is not None: self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        if etag: self.send_header("ETag", etag)
        if encoding: self.send_header("Content-Encoding", encoding)
        if vary: self.send_header("Vary", "Accept-Encoding")
        self.send_header("X-Content-Type-Options", "nosniff"); self.send_header("X-Frame-Options", "SAMEORIGIN" if embeddable else "DENY"); self.send_header("Referrer-Policy", "no-referrer")
        for name, value in (headers or {}).items(): self.send_header(name, value)
        self.send_header("Content-Security-Policy", f"default-src 'self'; {CSP_SCRIPT.get(csp, CSP_SCRIPT['workbench'])}; {CSP_REST}; frame-ancestors " + ("'self'" if embeddable else "'none'"))
        if self._https(): self.send_header("Strict-Transport-Security", HSTS_VALUE)   # v0.15: only behind a trusted https proxy
        secure = "; Secure" if (cookie or clear_cookie) and self._cookie_secure() else ""
        if cookie: self.send_header("Set-Cookie", f"{COOKIE_NAME}={cookie}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_SECONDS}{secure}")
        if clear_cookie: self.send_header("Set-Cookie", f"{COOKIE_NAME}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0{secure}")
        if download: self.send_header("Content-Disposition", _disposition(download))
        self.end_headers()
    def _not_modified(self, etag, cache, *, vary=False, embeddable=False, csp="workbench", cookie=None):
        # 304 carries no body and no Content-Length (a zero length could be stored over the cached one).
        self._write_headers(304, None, None, etag=etag, cache=cache, vary=vary, embeddable=embeddable, csp=csp, cookie=cookie)
    def _send(self, code, body, ctype="application/json; charset=utf-8", *, cookie=None, download=None, headers=None, embeddable=False, clear_cookie=False,
              csp="workbench", cache="no-store", etag=None, compress=False):
        if etag and code == 200 and etag_matches(self.headers.get("If-None-Match"), etag):
            return self._not_modified(etag, cache, vary=compress, embeddable=embeddable, csp=csp)
        encoding = None
        if compress and len(body) > GZIP_MIN_BYTES and self._gzip_ok():
            body = gzip.compress(body, GZIP_LEVEL, mtime=0); encoding = "gzip"
        self._write_headers(code, ctype, len(body), cookie=cookie, download=download, headers=headers, embeddable=embeddable,
                            clear_cookie=clear_cookie, csp=csp, cache=cache, etag=etag, encoding=encoding, vary=compress)
        self.wfile.write(body); self._sent = len(body)
    def _send_path(self, path: Path, ctype: str, *, download=None, embeddable=False, csp="workbench", cache="no-store", etag_prefix=None,
                   compress=False, cookie=None):
        """Serve a file from one open descriptor: ETag from its fstat, gzip from the cache, otherwise streamed."""
        with open(path, "rb") as stream:
            return self._send_open(stream, path, ctype, download=download, embeddable=embeddable, csp=csp, cache=cache,
                                   etag_prefix=etag_prefix, compress=compress, cookie=cookie)
    def _send_open(self, stream, path: Path | None, ctype: str, *, download=None, embeddable=False, csp="workbench", cache="no-store",
                   etag_prefix=None, compress=False, cookie=None):
        """Send an open binary file (``path`` keys the gzip cache; None = never cached, e.g. a bundle already unlinked)."""
        stat = os.fstat(stream.fileno())
        etag = job_etag(stat).replace('W/"', f'W/"{etag_prefix}', 1) if etag_prefix is not None else None
        if etag and etag_matches(self.headers.get("If-None-Match"), etag):
            return self._not_modified(etag, cache, vary=compress, embeddable=embeddable, csp=csp, cookie=cookie)
        common = dict(download=download, embeddable=embeddable, csp=csp, cache=cache, etag=etag, vary=compress, cookie=cookie)
        if compress and path is not None and stat.st_size > GZIP_MIN_BYTES and self._gzip_ok():
            body = self.server.gzip_cache.get(Path(path), stream, stat)
            self._write_headers(200, ctype, len(body), encoding="gzip", **common)
            self.wfile.write(body); self._sent = len(body)
            return
        self._write_headers(200, ctype, stat.st_size, **common)
        self._sent = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            self.wfile.write(block); self._sent += len(block)
    def _serialize(self, obj) -> bytes:
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
        if b'"admin_detail"' in body and not getattr(self, "_detail_ok", False):
            body = json.dumps(strip_admin_detail(obj), ensure_ascii=False, allow_nan=False).encode()   # S9
        return body
    def _json(self, obj, code=200, **kwargs):
        kwargs.setdefault("compress", True)
        self._send(code, self._serialize(obj), **kwargs)

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
        if origin is not None and origin not in self._allowed_origins(host): raise JobError("请求来源与当前服务不一致。", 403)
        if self.headers.get("Sec-Fetch-Site") == "cross-site": raise JobError("不接受跨站请求。", 403)
    def _session(self, local_create=False):
        self._host(); found = self.server.sessions.lookup(self.headers.get("Cookie"))
        if found:
            self._sid = found[0]; return found[0], found[1], False
        if local_create and not self.server.sessions.shared:
            sid, row = self.server.sessions.create(); self._sid = sid; return sid, row, True
        return None, None, False
    def _require_session(self, mutation=False):
        _, row, _ = self._session()
        if not row: raise JobError("会话已失效，请刷新页面或重新输入访问口令。", 401)
        # S9: typed-error ``admin_detail`` is for administrators (and the single local operator on loopback).
        self._detail_ok = not self.server.sessions.shared or self._admin(row)
        if mutation:
            self._origin(); csrf = self.headers.get("X-CSRF-Token", "")
            if not secrets.compare_digest(csrf.encode("utf-8", "surrogatepass"), str(row["csrf_token"]).encode("utf-8")): raise JobError("页面会话校验失败，请刷新页面后重试。", 403)
        return row
    @staticmethod
    def _admin(row: dict) -> bool: return row.get("role") == "admin"
    @staticmethod
    def _actor(row: dict) -> str:
        """Audit actor: the user name, else ``h:`` + a short hash of the anonymous owner (as the job manager logs it)."""
        if row.get("username"): return str(row["username"])
        return "h:" + hashlib.sha256(str(row.get("owner")).encode("utf-8")).hexdigest()[:12]
    def _audit(self, action: str, row: dict, job_ids, **extra) -> None:
        """S3: one ``wss_deploy.audit`` line per HTTP deletion / restore / purge / claim: who (session), from where, which jobs."""
        record = {"action": action, "actor": self._actor(row), "role": row.get("role") or ("local" if not self.server.sessions.shared else None),
                  "ip": self._client_ip(), "jobs": list(job_ids) if isinstance(job_ids, (list, tuple)) else [str(job_ids)], **extra}
        AUDIT.info("http %s", json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str))
        self._operations_audit(record, owner=row.get("owner"))
    def _operations_audit(self, record: dict, *, owner=None) -> None:
        try:
            self.server.operations.record_event({"at": _date(time.time()), "action": record["action"],
                "actor": record.get("actor") or record.get("username"), "owner": owner or record.get("username"),
                "job": (record.get("jobs") or [None])[0], "source": "http",
                "details": {key: value for key, value in record.items() if key not in {"action", "actor", "username"}}})
        except Exception:
            LOG.exception("Operations journal write failed for %s", record.get("action"))
    def _login_budget(self, client: str, username: str | None, *, method: str) -> None:
        """v0.15: one token from the client-address budget and, when a user name is given, one from that name's budget;
        a refusal answers 429 with ``retry_after`` (body and Retry-After header) and is audited once a minute per key."""
        ip_bucket, user_bucket = self.server.login_throttle, self.server.user_login_throttle
        if not ip_bucket.take(client):
            self._throttled(ip_bucket, client, scope="ip", username=username, method=method,
                            message="来自该网络地址的登录尝试过于频繁，请 {wait} 秒后再试。")
        if username and not user_bucket.take(username):
            ip_bucket.refund(client)          # this attempt never reached the password check
            self._throttled(user_bucket, username, scope="username", username=username, method=method,
                            message="该用户名的登录尝试过于频繁，请 {wait} 秒后再试。")
    def _throttled(self, bucket, key: str, *, scope: str, username, method: str, message: str):
        wait = bucket.retry_after(key) or 1
        if bucket.first_refusal(key):
            # Only registered names are logged (a password typed into the user-name field must not reach the log).
            name = username if username and self.server.sessions._registered(username) else ("<unknown>" if username else None)
            AUDIT.info("http %s", json.dumps({"action": "login_throttled", "scope": scope, "method": method, "username": name,
                                              "ip": self._client_ip(), "retry_after": wait}, ensure_ascii=False, separators=(",", ":")))
            self._operations_audit({"action": "login_throttled", "username": name, "scope": scope,
                                    "method": method, "ip": self._client_ip(), "retry_after": wait})
        raise JobError(message.format(wait=wait), 429, {"retry_after": wait})
    def _content_length(self, limit, message=None) -> int:
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1: raise JobError("请求必须指定唯一的 Content-Length。", 411)
        try: length = int(lengths[0])
        except ValueError: raise JobError("无效的请求长度。", 400)
        if length < 0 or length > limit: raise JobError(message or f"请求过大；STL 上传上限为 {MAX_UPLOAD_BYTES // 1024 // 1024} MiB。", 413)
        return length
    def _body(self, limit):
        length = self._content_length(limit)
        body = self.rfile.read(length)
        if len(body) != length: raise JobError("上传未完成，请重新上传。", 400)
        self._consumed = True
        return body
    def _payload(self, limit=MAX_JSON_BYTES):
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "application/json": raise JobError("请求必须使用 application/json。", 415)
        try: value = json.loads(self._body(limit) or b"{}")
        except (ValueError, UnicodeError, RecursionError): raise JobError("请求内容不是有效 JSON。", 400)   # v0.15: deep nesting is a 400, not a 500 + traceback
        if not isinstance(value, dict): raise JobError("请求内容必须是 JSON 对象。", 400)
        return value
    def _handle(self, method):
        self._t0 = time.monotonic(); self._status = None; self._sent = 0; self._sid = None; self._detail_ok = False
        self._consumed = False; self._in_handle = True
        try:
            try: {"GET": self._get, "POST": self._post, "PUT": self._put}[method]()
            except JobError as e:
                if not self._consumed and self.headers.get("Content-Length", "0").strip() not in {"", "0"}:
                    self.close_connection = True      # an unread request body must not be parsed as the next request
                body = {"error": {"message": str(e)}}; body.update(e.payload)
                retry = e.payload.get("retry_after") if e.status == 429 else None
                self._json(body, e.status, headers={"Retry-After": str(retry)} if type(retry) is int and retry > 0 else None)
            except (BrokenPipeError, ConnectionResetError, TimeoutError): LOG.info("Client disconnected")
            except Exception:
                did = secrets.token_hex(6); LOG.exception("Request failed; diagnostic_id=%s", did); self._json({"error": {"message": "服务暂时无法完成请求，请重试或提供诊断编号给维护者。", "diagnostic_id": did}}, 500)
        finally:
            self._in_handle = False
            self._access_line()
    def do_GET(self): self._handle("GET")
    def do_POST(self): self._handle("POST")
    def do_PUT(self): self._handle("PUT")

    # ------------------------------------------------------------------ streams
    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store"); self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Accel-Buffering", "no")
        if self._https(): self.send_header("Strict-Transport-Security", HSTS_VALUE)
        self.end_headers()
    def _emit(self, name: str, payload: dict) -> None:
        data = self._serialize(payload).decode("utf-8")
        self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode("utf-8"))
        self.wfile.flush()
    def _pump(self, listener: queue.Queue, *, stop_when_final: bool, closed: threading.Event | None = None) -> None:
        deadline = time.time() + SSE_MAX_SECONDS
        while time.time() < deadline:
            if closed is not None and closed.is_set():
                return
            try:
                event = listener.get(timeout=SSE_KEEPALIVE_SECONDS)
            except queue.Empty:
                self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
                continue
            if event is _SSE_CLOSE or (closed is not None and closed.is_set()):
                return
            self._emit("job", event)
            if stop_when_final and event.get("final"):
                return

    def _stream_events(self, job_id: str, owner: str, *, any_owner: bool = False) -> None:
        """Server-sent events: one snapshot, then every job event until the job is final or the client leaves.

        Replaces the 2 s polling of the full job record; the browser reconnects on its own if the
        stream is cut, and the page still polls as a fallback when EventSource is unavailable.
        At most ``WSS_DEPLOY_MAX_SSE_PER_SESSION`` streams per session (S5): a newer one ends the oldest.
        """
        manager = self.server.manager
        listener, snapshot = manager.subscribe(job_id, owner, any_owner=any_owner)
        entry = self.server.sse_open(self._sid); entry["listener"] = listener
        try:
            self._sse_headers()
            self._emit("snapshot", snapshot)
            if snapshot.get("final"):
                return
            self._pump(listener, stop_when_final=True, closed=entry["closed"])
        finally:
            self.server.sse_close(self._sid, entry)
            manager.unsubscribe(job_id, listener)

    def _stream_owner_events(self, owner: str) -> None:
        """Owner-level stream (contract §11.5): ``hello`` then every transition of the owner's jobs."""
        manager = self.server.manager
        listener, hello = manager.subscribe_owner(owner)
        entry = self.server.sse_open(self._sid); entry["listener"] = listener
        try:
            self._sse_headers()
            self._emit("hello", hello)
            self._pump(listener, stop_when_final=False, closed=entry["closed"])
        finally:
            self.server.sse_close(self._sid, entry)
            manager.unsubscribe_owner(owner, listener)

    # ------------------------------------------------------------------ helpers
    def _owner_names(self, jobs: list) -> None:
        """Administrator's all-users list (workspace v2 lane C, 「看全部用户」): ``owner_name`` = the registered user
        name that owns each job, '' for an anonymous (token / loopback) session owner.  The owner key itself is never sent."""
        manager, sessions = self.server.manager, self.server.sessions
        for job in jobs:
            owner = manager.owner_of(str(job.get("id") or "")) or ""
            job["owner_name"] = owner if owner and sessions._registered(owner) else ""

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
        kwargs = {}
        if _accepts(render_onepage, "timeline"):
            timeline = self._onepage_timeline(job_id, job)
            if timeline is not None:
                kwargs["timeline"] = timeline
        return render_onepage(self._summary(job_id), job, job_dir=self._job_dir(job_id), **kwargs)
    def _onepage_timeline(self, job_id: str, job: dict) -> dict | None:
        """§19.3: the patient's follow-up timeline (the job owner's jobs) when it has at least two scans."""
        patient = str(job.get("patient_id") or "").strip()
        if not patient:
            return None
        manager = self.server.manager
        try:
            timeline = manager.patient_timeline(patient, manager.owner_of(job_id) or "")
        except Exception:
            LOG.exception("Timeline for the one-page report of %s failed", job_id)
            return None
        return timeline if timeline.get("n_scans", 0) >= 2 else None
    def _template_editable(self, row: dict) -> bool:
        """The institution template is shared by everyone: loopback mode or an administrator may change it."""
        return not self.server.sessions.shared or self._admin(row)
    def _bundle_source(self, job_id: str, row: dict) -> tuple[dict, str | None, Path]:
        job = self._done_job(job_id, row, what="打包下载")
        try: onepage = self._onepage(job_id, job)
        except JobError: onepage = None
        return job, onepage, self._job_dir(job_id)
    def _send_bundle(self, ids: list[str], row: dict, *, download: str | None = None) -> None:
        """S5: zips are built in ``<jobs root>/.tmp`` (never in memory), one build at a time, capped in size,
        sent with Content-Length and deleted.  One id → that job's zip; several → an outer zip of per-job zips."""
        from .bundle import bundle_name, estimate_bytes, write_job_bundle, write_multi_bundle
        slots, cap = self.server.bundle_slots, self.server.max_bundle_bytes
        if not slots.acquire(timeout=BUNDLE_WAIT_SECONDS): raise JobError("正在生成其他打包下载，请稍后重试。", 429)
        paths: list[str] = []
        def temp(prefix):
            handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".zip", dir=self.server.temp_dir(), delete=False)
            paths.append(handle.name); return handle
        def too_big(size):
            return JobError(f"打包内容约 {size / 1024 ** 3:.2f} GiB，超过 {cap / 1024 ** 3:.2f} GiB 上限；请分批下载。", 413)
        try:
            sources = [self._bundle_source(job_id, row) for job_id in ids]
            estimate = sum(estimate_bytes(job_dir, OUTPUT_FILES) + len((onepage or "").encode("utf-8")) for _, onepage, job_dir in sources)
            if estimate > cap: raise too_big(estimate)
            if len(sources) == 1 and download is None:
                job, onepage, job_dir = sources[0]
                with temp("bundle_") as handle: write_job_bundle(handle, job_dir, job, OUTPUT_FILES, onepage_html=onepage)
                name = bundle_name(job)
            else:
                inner = []
                for job, onepage, job_dir in sources:
                    with temp("bundle_part_") as handle: write_job_bundle(handle, job_dir, job, OUTPUT_FILES, onepage_html=onepage)
                    inner.append((bundle_name(job), Path(handle.name)))
                    if sum(path.stat().st_size for _, path in inner) > cap: raise too_big(sum(path.stat().st_size for _, path in inner))
                with temp("bundle_") as handle: write_multi_bundle(handle, inner)
                for _, part in inner: part.unlink(missing_ok=True)
                name = download or "wss_bundle.zip"
            final = Path(paths[-1])
            if final.stat().st_size > cap: raise too_big(final.stat().st_size)
            with open(final, "rb") as stream:
                for path in paths:           # unlinked before sending: nothing is left behind if the client stalls or the process dies
                    try: os.unlink(path)
                    except OSError: pass
                self._send_open(stream, None, "application/zip", download=name)
        finally:
            for path in paths:
                try: os.unlink(path)
                except OSError: pass
            slots.release()
    def _check_active_jobs(self, owner: str, adding: int) -> None:
        """S5: queued + running jobs per owner are capped (``WSS_DEPLOY_MAX_ACTIVE_JOBS_PER_OWNER``, default 20)."""
        counts = self.server.manager.queue_counts(owner)
        active = int(counts.get("queued", 0)) + int(counts.get("running", 0))
        cap = self.server.max_active_jobs
        if active + adding > cap:
            raise JobError(f"排队和计算中的任务已达上限（{cap} 个，当前 {active} 个）；请等部分任务完成后再提交。", 429,
                           {"active_jobs": active, "max_active_jobs": cap})
    @staticmethod
    def _release_error(exc: Exception) -> JobError:
        """S9: release errors reach the client without server paths; the full text stays in the service log."""
        LOG.warning("Release error: %s", exc)
        return JobError(public_error_text(str(exc)), 422)
    def _ids(self, raw, limit: int) -> list[str]:
        """Job ids from ``?ids=a,b`` or a JSON list; de-duplicated, order kept."""
        if isinstance(raw, str): raw = raw.split(",")
        if not isinstance(raw, list): raise JobError("ids 必须是逗号分隔的字符串或列表。")
        ids = list(dict.fromkeys(item.strip() for item in raw if isinstance(item, str) and item.strip()))
        if not ids: raise JobError("请提供至少一个任务编号。")
        if len(ids) > limit: raise JobError(f"一次最多处理 {limit} 个任务。")
        if any(not re.fullmatch(JOB_ID_PATTERN, item) for item in ids): raise JobError("任务编号无效。")
        return ids
    # ------------------------------------------------------------------ workspace v2 (WORKSPACE_V2_CONTRACT.md §1–§2)
    def _v2_static(self, path: str) -> None:
        """``/v2/`` (the workspace page, session handling as ``/``), ``/static/v2/<name>`` (bundle.json whitelist,
        flat) and ``/v2/example`` (the generated example offline report, CSP ``report``, logged-in only)."""
        if path == V2_EXAMPLE_ROUTE:
            sid, row, fresh = self._session(local_create=True)
            if not row:
                raise JobError("会话已失效，请刷新页面或重新输入访问口令。", 401)
            file = contained_file(V2_DIR, V2_EXAMPLE, {V2_EXAMPLE})
            if not file:
                raise JobError("示例报告尚未生成。", 404)
            return self._send_path(file, "text/html; charset=utf-8", cookie=sid if fresh else None, csp="report",
                                   cache=STATIC_CACHE, etag_prefix="x", compress=True)
        page = path == "/v2/"
        name = "index.html" if page else path.removeprefix("/static/v2/")
        file = contained_file(V2_DIR, name, {"index.html"} if page else v2_static_files())
        if not file:
            raise JobError("文件不存在。", 404)
        sid, _, fresh = self._session(local_create=page)
        ctype = STATIC_TYPES.get(file.suffix.lower(), "application/octet-stream")
        return self._send_path(file, ctype, cookie=sid if fresh else None, cache=STATIC_CACHE, etag_prefix="s",
                               compress=ctype.split(";", 1)[0] in GZIP_TYPES)

    V2_RECORD_KEYS = ("id", "status", "stage", "phase", "detail", "version", "error", "case_id", "patient_id", "scan_label",
                      "scan_date", "created_at", "review", "companions", "companion_of", "narrative", "display_name",
                      "source_filename", "model_release", "run_identity", "mapping_history", "mapping")

    def _v2_record(self, job_id: str, row: dict) -> tuple[dict, str | None]:
        """The owner-checked job record fields the v2 data layer reads (administrators read any owner's job, as with
        ``GET /api/jobs/<id>``), without copying the whole record (events, stage A); plus the stored upload name."""
        manager = self.server.manager
        with manager.lock:
            job = manager._owned(job_id, row["owner"], any_owner=self._admin(row))
            record = copy.deepcopy({key: job.get(key) for key in self.V2_RECORD_KEYS if key in job})
            record["input_sha256"] = manager._input_sha256(job) if hasattr(manager, "_input_sha256") else job.get("input_sha256")
            filename = job.get("filename")
        return record, filename if isinstance(filename, str) else None

    @staticmethod
    def _v2_require_done(record: dict) -> None:
        if record.get("status") != "done":
            raise JobError("任务完成后才能读取结果数据。", 409, {"status": record.get("status")})

    def _v2_data(self, job_id: str):
        from . import v2_data
        try:
            return v2_data.load(self._job_dir(job_id))
        except v2_data.DataError as exc:
            raise JobError(str(exc), exc.status)

    def _v2_card(self, release_id):
        from . import v2_data
        registry = getattr(self.server.manager, "registry", None)
        return v2_data.load_card(release_id, v2_data.release_dir(registry, release_id) if registry is not None and release_id else None)

    def _v2_companions(self, record: dict, row: dict) -> list[dict]:
        """``companions`` of the record plus, for a companion task, the primary it was spawned from (owner-checked)."""
        out = [{**item, "role": "companion"} for item in (record.get("companions") or []) if isinstance(item, dict)]
        primary = record.get("companion_of")
        if isinstance(primary, str) and re.fullmatch(JOB_ID_PATTERN, primary):
            try:
                other, _ = self._v2_record(primary, row)
                out.insert(0, {"release_id": (other.get("model_release") or {}).get("id"), "job_id": primary, "role": "primary"})
            except JobError:
                pass
        return out

    def _v2_manifest(self, job_id: str, row: dict):
        from . import v2_data
        record, filename = self._v2_record(job_id, row)
        self._v2_require_done(record)
        data = self._v2_data(job_id)
        release = record.get("model_release") if isinstance(record.get("model_release"), dict) else {}
        card = self._v2_card(release.get("id") or release.get("release") or data.report.meta.get("release"))
        try:
            manifest = v2_data.build_manifest(data, record, card=card, companions=self._v2_companions(record, row))
        except v2_data.DataError as exc:
            raise JobError(str(exc), exc.status)
        return data, record, manifest, filename

    def _v2_get(self, path: str, row: dict) -> None:
        from . import v2_data
        manager = self.server.manager
        if path == "/api/v2/model-cards":
            registry = getattr(manager, "registry", None)
            ids = [item.get("id") for item in manager.releases()] if registry is not None else \
                [item.get("id") for item in getattr(manager, "_known_releases", lambda: [])()]
            return self._json({"cards": v2_data.all_cards(registry, ids)})
        match = re.fullmatch(rf"/api/v2/jobs/({JOB_ID_PATTERN})/manifest", path)
        if match:
            _, _, manifest, _ = self._v2_manifest(match[1], row)
            body = self._serialize(manifest)
            etag = v2_data.manifest_etag(manifest, body)
            if etag_matches(self.headers.get("If-None-Match"), etag):
                return self._not_modified(etag, REPORT_CACHE, vary=True)
            return self._v2_send_manifest(match[1], body, etag)
        match = re.fullmatch(rf"/api/v2/jobs/({JOB_ID_PATTERN})/arrays/([^/]+)", path)
        if match:
            record, _ = self._v2_record(match[1], row)
            self._v2_require_done(record)
            key = match[2]
            data = self._v2_data(match[1])
            spec = data.report.specs.get(key) if re.fullmatch(v2_data.ARRAY_KEY_PATTERN, key) else None
            if spec is None:
                raise JobError("数组不存在。", 404)
            etag = data.array_etag(key)
            if etag_matches(self.headers.get("If-None-Match"), etag):
                return self._not_modified(etag, REPORT_CACHE)
            try:
                body = data.report.raw(key)
            except v2_data.DataError as exc:
                raise JobError(str(exc), exc.status)
            # Raw little-endian bytes, never gzipped (contract §2); dtype and shape repeat the manifest entry.
            return self._send(200, body, "application/octet-stream", etag=etag, cache=REPORT_CACHE,
                              headers={"X-WSS-Dtype": spec.dtype, "X-WSS-Shape": ",".join(str(n) for n in spec.shape)})
        match = re.fullmatch(rf"/api/v2/jobs/({JOB_ID_PATTERN})/inputcheck", path)
        if match:
            record, filename = self._v2_record(match[1], row)
            stage_a = manager.stage_a(match[1]) if hasattr(manager, "stage_a") else None
            return self._json(v2_data.build_inputcheck(self._job_dir(match[1]), record, stage_a, input_filename=filename))
        raise JobError("接口不存在。", 404)

    V2_BODY_CACHE = 12

    def _v2_send_manifest(self, job_id: str, body: bytes, etag: str) -> None:
        """The serialised manifest, gzipped once per ETag (a three-head manifest is ~260 kB and gzip was three quarters
        of a warm request; the ETag digests the serialised body, so equal tags mean equal bytes)."""
        gz = self._gzip_ok()
        key = (job_id, etag, gz)
        with self.server.v2_bodies_lock:
            cached = self.server.v2_bodies.get(key)
            if cached is not None:
                self.server.v2_bodies.move_to_end(key)
        if cached is None:
            encoded = gz and len(body) > GZIP_MIN_BYTES
            cached = (gzip.compress(body, GZIP_LEVEL, mtime=0) if encoded else body, encoded)
            with self.server.v2_bodies_lock:
                self.server.v2_bodies[key] = cached
                while len(self.server.v2_bodies) > self.V2_BODY_CACHE:
                    self.server.v2_bodies.popitem(last=False)
        body, encoded = cached
        self._write_headers(200, "application/json; charset=utf-8", len(body), cache=REPORT_CACHE, etag=etag,
                            encoding="gzip" if encoded else None, vary=True)
        self.wfile.write(body); self._sent = len(body)

    def _v2_offline(self, job_id: str, row: dict) -> None:
        """``POST /api/v2/jobs/<id>/offline``: the single-file offline page as a download (contract §2, §6.3)."""
        from . import v2_data, v2_offline
        payload = self._payload(MAX_OFFLINE_REQUEST_BYTES)
        hide = payload.get("hide_name", False)
        if type(hide) is not bool:
            raise JobError("hide_name 必须是布尔值。")
        bookmarks = payload.get("bookmarks") if payload.get("bookmarks") is not None else []
        if not isinstance(bookmarks, list) or len(bookmarks) > MAX_OFFLINE_BOOKMARKS or any(not isinstance(item, dict) for item in bookmarks):
            raise JobError(f"bookmarks 必须是最多 {MAX_OFFLINE_BOOKMARKS} 个对象的列表。")
        view = payload.get("view")
        if view is not None and not isinstance(view, dict):
            raise JobError("view 必须是对象或 null。")
        try:
            json.dumps([bookmarks, view], allow_nan=False)
        except ValueError:
            raise JobError("书签或视图含有非有限数值。")
        data, record, manifest, filename = self._v2_manifest(job_id, row)
        slots = self.server.bundle_slots
        if not slots.acquire(timeout=BUNDLE_WAIT_SECONDS):
            raise JobError("正在生成其他下载，请稍后重试。", 429)
        try:
            page = v2_offline.build_offline_html(data, manifest, record={**record, "filename": filename}, hide_name=hide,
                                                 bookmarks=bookmarks, view=view)
        except v2_offline.OfflineBuildError as exc:
            LOG.warning("Offline package of %s not built: %s", job_id, exc)
            raise JobError(str(exc), 503)
        except v2_data.DataError as exc:
            raise JobError(str(exc), exc.status)
        finally:
            slots.release()
        body = page.encode("utf-8")
        self._audit("offline_export", row, [job_id], hide_name=hide, bytes=len(body))
        return self._send(200, body, "text/html; charset=utf-8", download=v2_offline.offline_filename(manifest, hide_name=hide),
                          csp="report", compress=True)

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
        if path in {"/api/health", "/api/ready"}:
            # S11: unauthenticated callers see ``ok`` + version only; /api/ready answers 503 while not ok.
            _, row, _ = self._session()
            if row: self._detail_ok = not sessions.shared or self._admin(row)
            health = self.server.health(row)
            if path == "/api/ready" and row and (not sessions.shared or self._admin(row)):
                from .service import server_summary   # O3: the ``service status`` facts, for administrators only
                health["summary"] = server_summary(self.server)
            return self._json(health, 200 if path == "/api/health" or health.get("ok") else 503)
        if path == "/v2":
            location = "/v2/" + ("?" + parsed_request.query if parsed_request.query else "")
            return self._send(301, b"", "text/plain; charset=utf-8", headers={"Location": location})
        if path in {"/v2/", V2_EXAMPLE_ROUTE} or path.startswith("/static/v2/"):
            return self._v2_static(path)
        if path in {"/", "/compare", "/ops", "/ops/", "/support", "/support/"} or path.startswith("/static/"):
            # Page shells contain no user data and provide their own login. Every operations API is admin-gated.
            name = {"/": "index.html", "/compare": "compare.html", "/ops": "ops.html", "/ops/": "ops.html",
                    "/support": "support.html", "/support/": "support.html"}.get(path) or path.removeprefix("/static/"); file = contained_file(STATIC_DIR, name, STATIC_FILES)
            if not file: raise JobError("文件不存在。", 404)
            sid, _, fresh = self._session(local_create=not path.startswith("/static/")); ctype = {".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8"}.get(file.suffix, "application/octet-stream")
            # S10: revalidated with ETag (304) and gzipped; S6: workbench pages allow no inline script.
            return self._send_path(file, ctype, cookie=sid if fresh else None, cache=STATIC_CACHE, etag_prefix="s",
                                   compress=ctype.split(";", 1)[0] in GZIP_TYPES)
        row = self._require_session(); manager = self.server.manager; admin = self._admin(row)
        query, one, integer = self._query(parsed_request)
        all_owners = admin and one("all", "0") == "1"
        # The console and support store are mounted before the legacy job routes.  They are deliberately
        # implemented outside the workbench API so its behaviour and owner scoping remain unchanged.
        if path.startswith("/api/ops/"):
            if sessions.shared and not admin:
                raise JobError("运维接口仅限管理员访问。", 403)
            from .operations_http import handle_get
            if handle_get(self, row, path, one, integer):
                return
        if path.startswith("/api/support/tickets"):
            from .operations_http import handle_get
            if handle_get(self, row, path, one, integer):
                return
        if path.startswith("/api/v2/"):
            return self._v2_get(path, row)
        if path == "/api/releases":
            return self._json({"releases": manager.releases()})
        if path == "/api/compare":
            raise JobError("结果比较需要使用 POST 请求。", 405)
        if path == "/api/events":
            return self._stream_owner_events(row["owner"])
        if path == "/api/preferences":
            return self._json({"preferences": self.server.preferences.get(row["owner"])})
        if path == "/api/report-template":
            from .report_template import load as load_template
            return self._json({"template": load_template(manager.root), "editable": self._template_editable(row)})
        match = re.fullmatch(r"/api/patients/([^/]+)/timeline", parsed_request.path)
        if match:
            try: patient = unquote(match[1], errors="strict")
            except (ValueError, UnicodeError): raise JobError("患者编号编码无效。")
            return self._json(manager.patient_timeline(patient, row["owner"], all_owners=all_owners))
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
                listing = {"jobs": manager.list(row["owner"], all_owners=all_owners)}
            else:
                listing = manager.query(row["owner"], q=one("q"), status=one("status"),
                                        patient_id=one("patient_id"), tag=one("tag"),
                                        page=integer("page", 1), page_size=integer("page_size", 25), all_owners=all_owners)
            if all_owners:
                self._owner_names(listing["jobs"])
            return self._json(listing)
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})", path)
        if match: return self._json(manager.get(match[1], row["owner"], any_owner=admin))
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/events", path)
        if match:
            return self._stream_events(match[1], row["owner"], any_owner=admin)
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/geometry", path)
        if match:
            # Display mesh + centreline polylines: fetched once per stage-A result, never polled.
            geometry = manager.geometry(match[1], row["owner"], any_owner=admin)
            # v0.15.8: the preview revision is part of the tag, so an upgraded (connected) preview replaces a cached scatter.
            etag = f'W/"{geometry["job_id"]}-{geometry["version"]}-{geometry.get("stage_a_created_at") or ""}-{geometry.get("preview_rev") or ""}"'
            if etag_matches(self.headers.get("If-None-Match"), etag):
                return self._not_modified(etag, REPORT_CACHE, vary=True)
            return self._json(geometry, etag=etag, cache=REPORT_CACHE)
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/onepage", path)
        if match:
            # Rendered on demand from summary.json; nothing is written to the job directory.  The ETag is the
            # hash of the rendered page, so a review, a template edit or a new follow-up scan all change it.
            job = self._done_job(match[1], row, what="生成一页纸报告")
            body = self._onepage(match[1], job).encode("utf-8")
            etag = f'W/"o-{hashlib.sha256(body).hexdigest()[:32]}"'
            return self._send(200, body, "text/html; charset=utf-8", embeddable=True, csp="onepage", cache=REPORT_CACHE, etag=etag, compress=True)
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/bundle\.zip", path)
        if match:
            return self._send_bundle([match[1]], row)
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/(?:files/([^/]+)|(report))", path); legacy = re.fullmatch(rf"/jobs/({JOB_ID_PATTERN})/([^/]+)", path)
        if match or legacy:
            job_id = (match or legacy)[1]; name = (match[2] or "report.html") if match else legacy[2]
            self._done_job(job_id, row, what="打开报告和导出文件")
            if name == "report.html":
                # §19.7: a report written with older templates is refreshed (UI only) before it is served;
                # on failure the original page is served and the error is logged.
                from .report_freshness import ensure_fresh
                if ensure_fresh(self._job_dir(job_id), guard=manager.lock) == "refreshed":
                    self.server._stale_cache = (0.0, None)
            file = output_file(self._job_dir(job_id), name)
            if not file: raise JobError("文件不存在或未生成。", 404)
            ctype = {".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8", ".csv": "text/csv; charset=utf-8", ".vtp": "application/xml", ".png": "image/png"}.get(file.suffix, "application/octet-stream")
            report = name == "report.html"
            return self._send_path(file, ctype, download=name if file.suffix not in {".html", ".json", ".png"} else None, embeddable=name in EMBEDDABLE_HTML,
                                   csp="report" if file.suffix == ".html" else "workbench", cache=REPORT_CACHE if report else "no-store",
                                   etag_prefix="r" if report else None, compress=file.suffix in {".html", ".json"})
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
            # S5 / v0.15: two budgets — per client address (WSS_DEPLOY_LOGIN_RATE_PER_MIN, 30 / min; token and password
            # logins share it) and per user name (WSS_DEPLOY_LOGIN_USER_RATE_PER_MIN, 10 / min); inside them users.py
            # locks a name after 5 wrong passwords in 60 s.  A successful login gives its tokens back.
            client = self._client_ip()
            method = "password" if "username" in payload or "password" in payload else "token"
            user_key = payload.get("username") if method == "password" and valid_username(payload.get("username")) else None
            if sessions.shared:
                self._login_budget(client, user_key, method=method)
            self._host(); found = sessions.lookup(self.headers.get("Cookie"), for_claim=True)
            previous = found[1] if found else None
            try:
                if method == "password":
                    sid, row = sessions.create(username=payload.get("username"), password=payload.get("password"), previous=previous)
                else:
                    sid, row = sessions.create(payload.get("token"))
            except JobError as exc:
                if sessions.shared:
                    # Only registered names are logged: a password typed into the user-name field must not reach the log.
                    typed = payload.get("username")
                    name = typed if valid_username(typed) and sessions._registered(typed) else ("<unknown>" if typed else None)
                    AUDIT.info("http %s", json.dumps({"action": "login_failed", "method": method, "username": name, "ip": client,
                                                      "status": exc.status}, ensure_ascii=False, separators=(",", ":")))
                    self._operations_audit({"action": "login_failed", "method": method, "username": name,
                                            "ip": client, "status": exc.status})
                raise
            if sessions.shared:
                self.server.login_throttle.refund(client)
                if user_key: self.server.user_login_throttle.refund(user_key)
                self._audit("login", row, [], method=method, claimable=len(row.get("claimable_owners") or []))
            return self._json(sessions.describe(row, manager=manager), cookie=sid)
        if path == "/api/session/logout":
            self._origin(); sid, row, _ = self._session()
            if sid: sessions.destroy(sid)
            if row and sessions.shared: self._audit("logout", row, [])
            return self._json({"authenticated": False}, clear_cookie=True)
        row = self._require_session(mutation=True); admin = self._admin(row)
        if path.startswith("/api/ops/"):
            if sessions.shared and not admin:
                raise JobError("运维接口仅限管理员访问。", 403)
            from .operations_http import handle_post
            if handle_post(self, row, path):
                return
        if path.startswith("/api/support/tickets"):
            from .operations_http import handle_post
            if handle_post(self, row, path):
                return
        match = re.fullmatch(rf"/api/v2/jobs/({JOB_ID_PATTERN})/offline", path)
        if match:
            return self._v2_offline(match[1], row)
        if path == "/api/session/password":
            payload = self._payload()
            if not row.get("username") or sessions.users is None: raise JobError("当前会话不是用户名登录，不能改口令。", 409)
            client = self._client_ip()
            self._login_budget(client, row["username"], method="password_change")
            sessions.users.change_password(row["username"], payload.get("old_password"), payload.get("new_password"))
            self.server.login_throttle.refund(client); self.server.user_login_throttle.refund(row["username"])
            # S1: the change bumps credential_generation, which ends every other session of this user; this
            # browser gets a renewed session (same CSRF token) bound to the new generation.
            renewed = sessions.renew(self._sid) if self._sid else None
            self._audit("password_changed", row, [])
            return self._json({"changed": True}, cookie=renewed[0] if renewed else None, clear_cookie=not renewed)
        if path == "/api/jobs/claim":
            payload = self._payload(); sid = self._sid
            owner = payload.get("owner")
            if not isinstance(owner, str) or not owner: raise JobError("请提供要认领的旧会话 owner。")
            if not row.get("username"): raise JobError("请先用用户名登录再认领任务。", 409)
            # S2: jobs of a registered user are never moved through the API (``cli jobs claim`` remains for that).
            if sessions._registered(owner): raise JobError("该 owner 是已注册用户，只能由维护者用命令行迁移。", 403)
            if owner not in row.get("claimable_owners", []) and not admin: raise JobError("该 owner 不属于本会话可认领的范围。", 403)
            result = manager.claim_owner(owner, row["username"])
            self._audit("claim", row, result.get("job_ids") or [], previous_owner="h:" + hashlib.sha256(owner.encode("utf-8")).hexdigest()[:12],
                        claimed=result.get("claimed"))
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
            from .bundle import MAX_BUNDLE_JOBS
            payload = self._payload(); ids = self._ids(payload.get("ids"), MAX_BUNDLE_JOBS)
            from .clock import strftime as _clock_strftime   # O1: deployment zone, not the process TZ
            return self._send_bundle(ids, row, download=f"wss_bundle_{_clock_strftime('%Y%m%d_%H%M')}.zip")
        if path in {"/api/jobs", "/api/jobs/batch", "/api/upload"}:
            return self._upload(row, batch=path == "/api/jobs/batch")
        if path == "/api/jobs/delete":
            payload = self._payload()
            items = payload.get("jobs")
            if not isinstance(items, list) or not items:
                raise JobError("请提供要删除的任务列表（id 与 version）。")
            result = manager.delete_many(row["owner"], items)
            deleted = [str(item.get("id")) for item in result.get("results", []) if isinstance(item, dict) and item.get("deleted")]
            self._audit("delete", row, deleted, requested=len(items), deleted=len(deleted))
            return self._json(result)
        match = re.fullmatch(rf"/api/trash/({JOB_ID_PATTERN})/(restore|purge)", path)
        if match:
            # S3: trash mutations act on the caller's own jobs only; the administrator's all-owner view is read-only.
            self._payload()
            if match[2] == "restore":
                job = manager.restore(match[1], row["owner"])
                self._audit("restore", row, [match[1]])
                return self._json({"job": job})
            result = manager.purge(match[1], row["owner"])
            self._audit("purge", row, [match[1]])
            return self._json(result)
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/snapshots", path)
        if match:
            # Pictures for the one-page report: a much larger body than any other JSON route.
            return self._json(manager.snapshots(match[1], row["owner"], self._payload(MAX_SNAPSHOT_BYTES)))
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/metadata", path)
        if match:
            return self._json({"job": manager.update_metadata(match[1], row["owner"], self._payload())})
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/(input|confirm|cancel|retry|rerun|delete|review)", path)
        if match:
            if match[2] in {"retry", "rerun"}:
                self._check_active_jobs(row["owner"], 1)
            try:
                result = getattr(manager, match[2])(match[1], row["owner"], self._payload())
            except ReleaseError as exc:
                raise self._release_error(exc)
            if match[2] == "delete":
                self._audit("delete", row, [match[1]])
                return self._json(result)
            return self._json({"job": result})
        raise JobError("接口不存在。", 404)

    # ------------------------------------------------------------------ PUT
    def _put(self):
        path = self._path(); self._host()
        row = self._require_session(mutation=True); manager = self.server.manager
        if path == "/api/preferences":
            return self._json({"preferences": self.server.preferences.put(row["owner"], self._payload(MAX_PREFERENCES_BYTES + 4096))})
        if path == "/api/report-template":
            from .report_template import load as load_template, save as save_template
            if not self._template_editable(row): raise JobError("只有本机回环模式或管理员可以修改报告模板。", 403)
            payload = self._payload()
            payload.pop("schema_version", None)
            try: saved = save_template(manager.root, {**load_template(manager.root), **payload})
            except ValueError as exc: raise JobError(str(exc))
            LOG.info("Report template updated by %s: %s", row.get("username") or "local", ", ".join(sorted(payload)) or "-")
            self._audit("template_updated", row, [], keys=sorted(payload))
            return self._json({"template": saved, "editable": True})
        match = re.fullmatch(rf"/api/jobs/({JOB_ID_PATTERN})/(annotations|findings_review|narrative)", path)
        if match:
            return self._json(getattr(manager, match[2])(match[1], row["owner"], self._payload()))
        raise JobError("接口不存在。", 404)

    # ------------------------------------------------------------------ uploads
    def _upload(self, row: dict, *, batch: bool) -> None:
        """S5: at most ``WSS_DEPLOY_MAX_CONCURRENT_UPLOADS`` uploads at once; the body streams part by part into
        spooled temporary files (≤ 8 MiB in memory each) instead of being buffered and copied twice."""
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data;"): raise JobError("上传必须使用 multipart/form-data。", 415)
        match = re.search(r'boundary=(?:"([^"]{1,70})"|([^;\s]{1,70}))', content_type, re.I)
        if not match: raise JobError("上传表单格式无效。")
        boundary = (match[1] or match[2]).encode("latin-1", "replace")
        length = self._content_length((MAX_BATCH_BYTES if batch else MAX_UPLOAD_BYTES) + 64 * 1024)
        slots = self.server.upload_slots
        if not slots.acquire(blocking=False):
            self._drain(length)
            raise JobError("当前正在接收其他上传，请稍后重试。", 429)
        spools: list = []
        try:
            try:
                self._check_active_jobs(row["owner"], 1)
            except JobError:
                self._drain(length); raise
            return self._upload_locked(row, batch=batch, spools=spools, boundary=boundary, length=length)
        finally:
            for spool in spools:
                try: spool.close()
                except OSError: pass
            slots.release()
    def _drain(self, length: int) -> None:
        """Read and discard a request body we refuse (429), so the browser receives the answer instead of a reset."""
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(UPLOAD_CHUNK, remaining))
            if not chunk: return
            remaining -= len(chunk)
        self._consumed = True
    def _upload_locked(self, row: dict, *, batch: bool, spools: list, boundary: bytes, length: int) -> None:
        manager = self.server.manager
        fields, uploads, raw_fields = {}, [], {}
        simple = {"case_id", "units", "remove_fragments", "release_id", "patient_id", "scan_label", "scan_date", "tags", "notes",
                  "device", "seed_count", "threads", "on_duplicate", "metadata_json", "companion_release_ids"}
        spool_dir = self.server.temp_dir()

        def start_part(name, filename):
            if name == "stl":
                if not batch and uploads: raise JobError("每个任务只能上传一个 STL。")
                if batch and len(uploads) >= MAX_BATCH_FILES: raise JobError(f"每批最多上传 {MAX_BATCH_FILES} 个 STL 文件。")
                spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_BYTES, dir=spool_dir); spools.append(spool)
                entry = [re.sub(r"[\x00-\x1f\x7f-\x9f]", "", filename or "") or "input.stl", spool, 0]; uploads.append(entry)
                def write(data):
                    entry[2] += len(data)
                    if entry[2] > MAX_UPLOAD_BYTES: raise JobError("STL 超过 128 MiB 上传上限。", 413)
                    spool.write(data)
                return write
            if name in simple:
                # Identifiers stay small; notes intentionally allow the same 2,000-character limit as
                # JobManager._metadata; the batch metadata document gets 64 KiB.
                limit = 64 * 1024 if name == "metadata_json" else 4096 if name == "notes" else 1024
                chunks = raw_fields[name] = []
                def write(data):
                    chunks.append(data)
                    if sum(len(c) for c in chunks) > limit: raise JobError("批量元数据过长。" if name == "metadata_json" else "上传表单字段过长。")
                return write
            return None

        consumed = [0]
        def read(size):
            data = self.rfile.read(size); consumed[0] += len(data); return data
        try:
            stream_multipart(read, length, boundary, start_part)
        except JobError:
            self._drain(length - consumed[0])     # answer a refused part (too large, too many files) cleanly
            raise
        self._consumed = True
        metadata_json = None
        for name, chunks in raw_fields.items():
            data = b"".join(chunks)
            if name == "metadata_json":
                try: metadata_json = json.loads(data.decode("utf-8"))
                except (UnicodeError, ValueError, RecursionError): raise JobError("metadata_json 不是有效 JSON。")
                continue
            try: fields[name] = data.decode("utf-8").strip()
            except UnicodeError: raise JobError("上传表单字段必须使用 UTF-8。")
        if not uploads or any(size == 0 for _, _, size in uploads): raise JobError("请选择 STL 文件。")
        self._check_active_jobs(row["owner"], len(uploads))
        # The spooled files go to the manager as file objects (``content_file``): it streams them into the job in
        # 1 MiB chunks, so no upload is ever read into memory as a whole.  They stay open until create returns.
        # Rewound here: Python 3.10's SpooledTemporaryFile has no ``seekable()``, so the manager would not rewind it.
        for _, spool, _ in uploads:
            spool.seek(0)
        uploads = [(filename, spool) for filename, spool, _ in uploads]
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
                        "on_duplicate": on_duplicate, "companion_release_ids": fields.get("companion_release_ids") or None}
            items = []
            for (filename, spool), item_meta in zip(uploads, metadata_json):
                # Client metadata must never name the STL source: ``content_path`` would read a server-side file.
                item_meta = {key: value for key, value in item_meta.items() if key not in UPLOAD_SOURCE_KEYS}
                item = {"filename": filename, **item_meta, "content_file": spool}
                items.append(item)
            try:
                result = manager.create_batch(row["owner"], items, **defaults)
            except ReleaseError as exc:
                raise self._release_error(exc)
            return self._json(result, 201)
        try:
            job = manager.create(row["owner"], content_file=uploads[0][1], filename=uploads[0][0],
                                 case_id=fields.get("case_id", ""), release_id=fields.get("release_id") or None,
                                 patient_id=fields.get("patient_id", ""), scan_label=fields.get("scan_label", ""),
                                 scan_date=fields.get("scan_date", ""), tags=tags, notes=fields.get("notes", ""),
                                 units=fields.get("units", "auto"), remove_fragments=remove in {"true", "1", "on"},
                                 device=fields.get("device", "auto"), seed_count=seed_count, threads=threads,
                                 on_duplicate=on_duplicate, companion_release_ids=fields.get("companion_release_ids") or None)
        except ReleaseError as exc:
            raise self._release_error(exc)
        response = {"job": job, "job_id": job["id"]}
        if job.get("reused_from"): response["reused_from"] = job["reused_from"]
        return self._json(response, 201)


def _accepts(fn, name: str) -> bool:
    """True when ``fn`` takes keyword ``name`` (contract §19.3: the one-pager gains ``timeline`` in W1's change)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def configure_logging(log_file: str | Path | None = None) -> None:
    """Timestamps with the UTC offset; ``log_file`` → rotating file (20 MB × 5) instead of stderr."""
    from .clock import LogFormatter   # O1: %(asctime)s in the deployment zone (WSS_DEPLOY_TZ → service.json env.TZ → system)
    formatter = LogFormatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = PrivateRotatingFileHandler(path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8")
    elif root.handlers:
        return
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root.addHandler(handler)


def _disposition(name: str) -> str:
    """ASCII fallback plus RFC 5987 UTF-8 form so Chinese case names survive as download names."""
    from urllib.parse import quote
    name = str(name).replace("\r", "").replace("\n", "")
    ascii_name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name) or "download"
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(name, safe="")}'


def _stop_on_sigterm(signum, _frame):
    # ``service stop`` sends SIGTERM: leave serve_forever through the normal shutdown path.
    raise KeyboardInterrupt(f"signal {signum}")


def _report_code_changed() -> bool:
    from .report_freshness import code_changed
    return code_changed()


def serve(host: str = "127.0.0.1", port: int = 8765, jobs_root: Path | None = None, device: str = "auto", token: str | None = None,
          log_file: str | Path | None = None) -> None:
    apply_umask_from_env()          # v0.15: before any file of this process is created
    configure_logging(log_file)
    # v0.15: behind a trusted reverse proxy every request arrives from loopback, so login-free loopback mode would
    # serve the whole network: WSS_DEPLOY_TRUST_PROXY=1 always means shared mode (users.json or a token required).
    trust_proxy = trust_proxy_enabled()
    root = Path(jobs_root or PROJECT_ROOT / "outputs/wss_deploy_jobs").resolve(); root.mkdir(parents=True, exist_ok=True); shared = service_is_shared(host); legacy_owner = os.environ.get("WSS_DEPLOY_LEGACY_OWNER") or None
    users = UserStore(root / "users.json")
    sessions = SessionStore(root / ".sessions.json", shared=shared, token=token or os.environ.get("WSS_DEPLOY_TOKEN"), legacy_owner=legacy_owner, users=users)
    registry = ReleaseRegistry(device=device)
    try:  # pin the report template code this process renders with (report_freshness.code_changed)
        from .report_freshness import snapshot_loaded_code
        snapshot_loaded_code()
    except Exception:
        LOG.exception("报告模板代码快照失败；运行中的自动刷新不做代码一致性检查")
    from .registry import preload_all
    threading.Thread(target=preload_all, args=(registry,), name="wss-preload", daemon=True).start()   # releases resident before the first job
    from .operations import OperationsStore
    operations = OperationsStore(root)
    manager = JobManager(root, registry=registry, legacy_owner=legacy_owner, audit_sink=operations.record_event)
    httpd = ServiceHTTPServer((host, port), manager, sessions, operations=operations); manager.start(); LOG.info("Serving wss_deploy %s on http://%s:%s; pid=%s; jobs=%s; shared=%s; login=%s; releases=%s; default=%s", __version__, host, port, os.getpid(), root, shared, sessions.login_mode(), registry.root, registry.default_id)
    if trust_proxy or cookie_secure_mode() != "auto" or os.environ.get(UMASK_ENV):
        LOG.info("Network options: trust_proxy=%s (loopback peers only); cookie_secure=%s; umask=%s", trust_proxy, cookie_secure_mode(),
                 os.environ.get(UMASK_ENV) or "unchanged")
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _stop_on_sigterm)
    try: httpd.serve_forever()
    except KeyboardInterrupt: LOG.info("Service stopping (pid=%s)", os.getpid())
    finally: httpd.server_close(); manager.close()
