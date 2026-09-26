"""Service management: ``python -m wss_deploy.cli service start|stop|restart|status|logs|token`` (contract §19.6).

Files in the jobs root:

- ``service.json``  – how to start the service (host / port / device / python / cwd / log file / ``env``).
- ``service.pid``   – ``{pid, started_at, host, port, argv, version, ...}`` of the process started here.
- ``.service_token`` – the shared-mode access token (mode 0600); never written to ``service.json`` or argv.
- ``server.log``    – rotating application log (``serve --log-file``); ``server.console.log`` keeps stdout/stderr.
- ``access.log``    – rotating request log (logger ``wss_deploy.access``, path only; v0.14).
- ``.service.lock`` – ``flock`` held by the running ``serve`` (single writer of the jobs root, v0.14); offline
  writers (``cli jobs claim``, ``cli reports refresh``) refuse while it is held unless ``--force``.
- ``.drain.json`` / ``.maintenance.json`` / ``maintenance_result.json`` – ``--drain`` and the post-upgrade
  maintenance the new service runs (see :func:`upgrade`).

A service started by hand (``nohup python -m wss_deploy.cli serve ...``) has no PID file.  ``status`` finds it by
scanning ``/proc/*/cmdline`` for ``wss_deploy.cli serve`` with the same ``--port``, and the first ``restart`` adopts it:
its argv, working directory and the ``WSS_DEPLOY_* / CUDA_VISIBLE_DEVICES / TZ`` environment are written to
``service.json`` and its token moves into ``.service_token``, so the restarted service keeps the same login.
Only processes whose command line matches are ever signalled.
"""
from __future__ import annotations

import errno
import fcntl
import ipaddress
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import __version__, clock
from .paths import PROJECT_ROOT

CONFIG_FILE = "service.json"
PID_FILE = "service.pid"
TOKEN_FILE = ".service_token"
LOG_FILE = "server.log"
ACCESS_LOG = "access.log"
LOCK_FILE = ".service.lock"
CONSOLE_FILE = "server.console.log"
CONFIG_SCHEMA = "wss-deploy.service/v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "TZ")
ENV_PREFIX = "WSS_DEPLOY_"
SECRET_ENV = ("WSS_DEPLOY_TOKEN", "WSS_DEPLOY_PASSWORD")
STOP_TIMEOUT_S = 15.0
HEALTH_TIMEOUT_S = 60.0
ACTIVE = ("running", "queued")
QUEUE_LABELS = (("running", "计算中"), ("queued", "排队"), ("awaiting_input", "待确认输入"), ("awaiting_confirmation", "待确认出口"))
DRAIN_DEFAULT_S = 600.0
MAINTENANCE_TIMEOUT_S = 900.0
PREFLIGHT_IMPORTS = "import wss_deploy.server, wss_deploy.jobs, wss_deploy.pipeline, wss_deploy.families, wss_deploy.report"
LOG_TAIL_LINES = 30


class ServiceError(RuntimeError):
    """A management action that cannot proceed; the message is shown to the user as-is."""


class ServiceLockError(ServiceError):
    """The jobs root is locked by another writer (``holder`` describes it)."""
    def __init__(self, message: str, holder: dict | None = None):
        super().__init__(message)
        self.holder = holder or {}


# ---------------------------------------------------------------------------------------------- single-writer lock (J3)
class ServiceLock:
    """An exclusive ``flock`` on ``<jobs_root>/.service.lock``; the kernel releases it when the process exits."""
    def __init__(self, path: Path, fd: int, record: dict):
        self.path, self.fd, self.record = Path(path), fd, record

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()


def _read_lock_record(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8") or "{}")
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def acquire_lock(root: Path, *, purpose: str, **info) -> ServiceLock:
    """Take the jobs-root writer lock without blocking; raises :class:`ServiceLockError` naming the holder."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_FILE
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
            raise
        holder = _read_lock_record(path)
        raise ServiceLockError(lock_message(root, holder), holder) from None
    record = {"pid": os.getpid(), "purpose": purpose, "started_at": _now(), "argv": redact_argv(sys.argv), **info}
    os.ftruncate(fd, 0)
    os.pwrite(fd, json.dumps(record, ensure_ascii=False).encode("utf-8"), 0)
    return ServiceLock(path, fd, record)


def lock_holder(root: Path) -> dict | None:
    """Who holds the writer lock of ``root`` (``{"held": True, pid, purpose, …}``), ``{"held": False, …}`` when the
    file exists but nobody holds it (a finished process; nothing to clean up), or None without a lock file."""
    path = Path(root) / LOCK_FILE
    if not path.is_file():
        return None
    record = _read_lock_record(path)
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return {"held": None, **record}
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        held = True
    else:
        held = False
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    out = {"held": held, **record}
    if held and record.get("pid"):
        out["alive"] = _proc_state(int(record["pid"])) not in {None, "Z", "X"}
    return out


def lock_message(root: Path, holder: dict) -> str:
    pid = holder.get("pid") or "未知"
    purpose = {"serve": "服务"}.get(holder.get("purpose"), holder.get("purpose") or "未知用途")
    since = f"，自 {holder['started_at']}" if holder.get("started_at") else ""
    return (f"任务目录 {root} 的写锁被进程 {pid}（{purpose}{since}）持有：同一任务目录同时只能有一个写入者。"
            "服务请用 `python -m wss_deploy.cli service stop` 停止；或换一个 --jobs-root。")


def offline_lock(root: Path, *, force: bool = False, purpose: str, out=print):
    """Writer lock for an offline command (``jobs claim`` / ``reports refresh``).

    Refuses with a clear message when a service (or another offline command) holds it; ``force`` proceeds without
    the lock after a warning (the running service may later overwrite what this command writes).  Returns the
    :class:`ServiceLock` to release, or None when forced."""
    try:
        return acquire_lock(root, purpose=purpose)
    except ServiceLockError as exc:
        holder = exc.holder
        if not force:
            what = "服务正在运行" if holder.get("purpose") == "serve" else "另一个维护命令正在运行"
            raise ServiceLockError(f"{what}（PID {holder.get('pid') or '未知'}，{holder.get('purpose') or '?'}）并持有任务目录 {Path(root)} 的写锁；"
                                   "离线命令会与它内存中的任务状态冲突。请先停止服务（`service stop`），"
                                   "或确认安全后加 --force。", holder) from None
        out(f"警告：写锁被 PID {holder.get('pid') or '未知'}（{holder.get('purpose') or '?'}）持有，--force 继续执行；"
            "运行中的服务可能覆盖本命令写入的内容。")
        return None


def redact_argv(argv) -> list:
    """argv with the value of ``--token`` replaced (service.json, adopt records, ``status --json``, lock file)."""
    out, hide = [], False
    for item in list(argv or []):
        item = str(item)
        if hide:
            out.append("***")
            hide = False
        elif item == "--token":
            out.append(item)
            hide = True
        elif item.startswith("--token="):
            out.append("--token=***")
        else:
            out.append(item)
    return out


def default_jobs_root() -> Path:
    return PROJECT_ROOT / "outputs/wss_deploy_jobs"


def _now() -> str:
    return clock.now_iso()          # O1: WSS_DEPLOY_TZ → service.json env.TZ → system zone (not the shell's TZ)


def _iso(ts: float | None) -> str | None:
    return clock.iso(ts) if ts else None


def is_loopback(host: str) -> bool:
    if (host or "").lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def kept_env(environ: dict) -> dict:
    """The environment a restart must carry over: ``WSS_DEPLOY_*``, ``CUDA_VISIBLE_DEVICES``, ``TZ`` (no secrets)."""
    return {key: value for key, value in sorted(environ.items())
            if (key.startswith(ENV_PREFIX) or key in ENV_KEYS) and key not in SECRET_ENV}


# ---------------------------------------------------------------------------------------------- files
def _json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _write_json(path: Path, value: dict, *, mode: int = 0o600) -> None:
    from .jobs import atomic_json
    atomic_json(path, value)
    os.chmod(path, mode)


def load_config(root: Path) -> dict:
    """``service.json`` merged onto defaults; an absent file gives the defaults (``exists`` False)."""
    root = Path(root)
    stored = _json(root / CONFIG_FILE)
    cfg = {"schema_version": CONFIG_SCHEMA, "host": DEFAULT_HOST, "port": DEFAULT_PORT, "device": "auto",
           "python": sys.executable, "cwd": str(PROJECT_ROOT), "log_file": str(root / LOG_FILE), "env": {}}
    for key in ("host", "port", "device", "python", "cwd", "log_file", "env", "adopted_from", "updated_at", "argv"):
        if key in stored:
            cfg[key] = stored[key]
    cfg["env"] = kept_env(cfg.get("env") if isinstance(cfg.get("env"), dict) else {})
    cfg["port"] = int(cfg["port"])
    cfg["exists"] = (root / CONFIG_FILE).is_file()
    return cfg


def save_config(root: Path, cfg: dict) -> dict:
    value = {key: cfg[key] for key in ("host", "port", "device", "python", "cwd", "log_file", "env", "adopted_from", "argv") if key in cfg}
    value.update(schema_version=CONFIG_SCHEMA, updated_at=_now())
    value["env"] = kept_env(value.get("env") or {})
    _write_json(Path(root) / CONFIG_FILE, value)
    return value


def read_token(root: Path) -> str | None:
    try:
        value = (Path(root) / TOKEN_FILE).read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def write_token(root: Path, token: str) -> Path:
    path = Path(root) / TOKEN_FILE
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, (token.strip() + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    return path


def read_pid_file(root: Path) -> dict:
    return _json(Path(root) / PID_FILE)


# ---------------------------------------------------------------------------------------------- processes
def _cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part] or None


def serve_options(argv: list[str] | None) -> dict | None:
    """Options of a ``wss_deploy.cli serve`` command line, or None when ``argv`` is not one.

    Matches argv *elements* (``python … -m wss_deploy.cli serve …`` or ``…/wss_deploy/cli.py serve``), so a shell
    whose ``-c`` string merely mentions the command is never mistaken for the service.
    """
    if not argv:
        return None
    start = None
    for index, item in enumerate(argv):
        if item == "-m" and argv[index + 1:index + 3] == ["wss_deploy.cli", "serve"]:
            start = index + 3
            break
        if item.endswith("wss_deploy/cli.py") and argv[index + 1:index + 2] == ["serve"]:
            start = index + 2
            break
    if start is None:
        return None
    options = {"host": DEFAULT_HOST, "port": DEFAULT_PORT, "jobs_root": None, "device": "auto", "token": None, "token_file": None, "log_file": None}
    names = {"--host": "host", "--port": "port", "--jobs-root": "jobs_root", "--device": "device", "--token": "token",
             "--token-file": "token_file", "--log-file": "log_file"}
    rest = argv[start:]
    index = 0
    while index < len(rest):
        item = rest[index]
        key, value = item, None
        if "=" in item and item.startswith("--"):
            key, value = item.split("=", 1)
        if key in names:
            if value is None:
                value = rest[index + 1] if index + 1 < len(rest) else None
                index += 1
            options[names[key]] = value
        index += 1
    try:
        options["port"] = int(options["port"])
    except (TypeError, ValueError):
        return None
    options["python"] = argv[0]
    return options


def _proc_started(pid: int) -> float | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rindex(")") + 2:].split()
        ticks = int(fields[19])
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime"))
        return boot + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def _proc_state(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rindex(")") + 2:].split()[0]
    except (OSError, ValueError, IndexError):
        return None


def process_env(pid: int) -> dict | None:
    """The full environment of one of our own processes (None when unreadable)."""
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    env = {}
    for part in raw.split(b"\0"):
        if b"=" in part:
            key, value = part.split(b"=", 1)
            env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def describe_process(pid: int) -> dict | None:
    """Facts about a live ``wss_deploy.cli serve`` process, or None when ``pid`` is not one."""
    argv = _cmdline(pid)
    options = serve_options(argv)
    if options is None or _proc_state(pid) in {None, "Z", "X"}:
        return None
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        cwd = None
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
    except OSError:
        uid = None
    jobs_root = options.get("jobs_root")
    if jobs_root:
        path = Path(jobs_root)
        if not path.is_absolute() and cwd:
            path = Path(cwd) / path
        jobs_root = str(path.resolve()) if cwd or path.is_absolute() else jobs_root
    else:
        jobs_root = str(default_jobs_root().resolve())
    try:
        log_target = os.readlink(f"/proc/{pid}/fd/1")
    except OSError:
        log_target = None
    started = _proc_started(pid)
    return {**options, "pid": pid, "argv": argv, "cwd": cwd, "own": uid == os.getuid(), "jobs_root": jobs_root,
            "started_ts": started, "started_at": _iso(started), "stdout": log_target}


def scan_processes(port: int | None = None) -> list[dict]:
    """Every live ``wss_deploy.cli serve`` process (optionally only those on ``port``), by PID."""
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        info = describe_process(int(entry.name))
        if info and (port is None or info["port"] == port):
            out.append(info)
    return sorted(out, key=lambda item: item["pid"])


def _same_root(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except (OSError, TypeError):
        return False


def locate(root: Path, port: int | None = None) -> dict | None:
    """The service of this jobs root: the PID file when it is valid, otherwise a scan by port.

    Adds ``managed`` (True when a valid ``service.pid`` names it) and ``root_matches``.
    """
    root = Path(root)
    record = read_pid_file(root)
    if record.get("pid"):
        info = describe_process(int(record["pid"]))
        if info and (port is None or info["port"] == port):
            return {**info, "managed": True, "pid_record": record, "root_matches": _same_root(info["jobs_root"], root)}
    if port is None:
        port = load_config(root)["port"]
    found = scan_processes(port)
    if not found:
        return None
    exact = [item for item in found if _same_root(item["jobs_root"], root)]
    info = (exact or found)[0]
    return {**info, "managed": False, "pid_record": record or None, "root_matches": bool(exact)}


# ---------------------------------------------------------------------------------------------- probes
def connect_host(host: str) -> str:
    if host in {"", "0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def http_opener():
    """urllib opener that never goes through an HTTP proxy (the service is on this machine / campus network)."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def probe_health(host: str, port: int, timeout: float = 3.0) -> dict | None:
    """``GET /api/health`` without a session: ``{"ok", "version"}``; an older service answers 401/404 (``legacy``)."""
    url = f"http://{connect_host(host)}:{port}/api/health"
    try:
        with http_opener().open(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
            value = json.loads(response.read() or b"{}")
            return value if isinstance(value, dict) else {"ok": False}
    except urllib.error.HTTPError as exc:
        return {"ok": True, "version": None, "legacy": True, "status": exc.code}
    except (OSError, ValueError):
        return None


def probe_ready(host: str, port: int, timeout: float = 3.0) -> dict | None:
    """``GET /api/ready`` (v0.14): 200 → ready, 503 → the service's failing checks; a service without the route
    (404 / 401) falls back to :func:`probe_health`.  None when nothing answers."""
    url = f"http://{connect_host(host)}:{port}/api/ready"
    try:
        with http_opener().open(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as response:
            value = json.loads(response.read() or b"{}")
            value = value if isinstance(value, dict) else {}
            value.setdefault("ok", True)
            return value
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            try:
                value = json.loads(exc.read() or b"{}")
            except ValueError:
                value = {}
            value = value if isinstance(value, dict) else {}
            value["ok"] = False
            return value
        return probe_health(host, port, timeout=timeout)
    except (OSError, ValueError):
        return None


# J5: ``nvidia-smi`` at most every 30 s per process (health, status and doctor share it).
GPU_CACHE_TTL_S = 30.0
_GPU_CACHE: dict = {}
_GPU_LOCK = threading.Lock()


def gpu_cache_time() -> float | None:
    """When the cached ``nvidia-smi`` rows were read (epoch seconds), None before the first read."""
    return _GPU_CACHE.get("at")


def _nvidia_rows() -> list[dict]:
    with _GPU_LOCK:
        return _nvidia_rows_locked()


def _nvidia_rows_locked() -> list[dict]:
    now = time.time()
    if _GPU_CACHE.get("at", 0) > now - GPU_CACHE_TTL_S:
        return _GPU_CACHE["rows"]
    rows = []
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            out = subprocess.run([exe, "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5, check=True).stdout
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    row = {"index": parts[0], "uuid": parts[1], "name": parts[2]}
                    if len(parts) >= 5:
                        try:
                            row.update(memory_used_mb=int(float(parts[3])), memory_total_mb=int(float(parts[4])))
                        except ValueError:
                            pass
                    rows.append(row)
        except (OSError, subprocess.SubprocessError):
            rows = []
    _GPU_CACHE.update(at=now, rows=rows)
    return rows


_UNSET = object()


def gpu_summary(visible=_UNSET) -> dict:
    """GPUs visible under ``CUDA_VISIBLE_DEVICES`` (this process's by default) via ``nvidia-smi``; torch is not imported."""
    if visible is _UNSET:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    base = {"visible_devices": visible}
    if visible is not None and visible.strip() in {"", "-1"}:
        return {**base, "available": False, "name": None, "count": 0, "names": [], "note": "CUDA_VISIBLE_DEVICES 为空：只用 CPU"}
    rows = _nvidia_rows()
    if visible:
        chosen = []
        for token in (t.strip() for t in visible.split(",") if t.strip()):
            match = next((r for r in rows if r["index"] == token or r["uuid"].startswith(token)), None)
            if match:
                chosen.append(match)
    else:
        chosen = rows
    return {**base, "available": bool(chosen), "name": chosen[0]["name"] if chosen else None, "count": len(chosen),
            "names": [r["name"] for r in chosen], "devices": chosen}


def queue_from_disk(jobs_root: Path) -> dict:
    """Active job counts read from ``*/job.json`` (read-only; used when no session can ask the service)."""
    counts = {key: 0 for key, _ in QUEUE_LABELS}
    for path in Path(jobs_root).glob("*/job.json"):
        if path.parent.name.startswith("."):
            continue
        status = _json(path).get("status")
        if status in counts:
            counts[status] += 1
    return counts


def disk_free_gb(path: Path) -> float | None:
    try:
        return round(shutil.disk_usage(path).free / 1024 ** 3, 1)
    except OSError:
        return None


def _port_busy(host: str, port: int) -> bool:
    try:
        with socket.create_connection((connect_host(host).strip("[]"), port), timeout=1):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------- actions
def status(root: Path, port: int | None = None) -> dict:
    """Read-only facts about the service of ``root`` (never writes, never signals)."""
    root = Path(root).resolve()
    cfg = load_config(root)
    port = port or cfg["port"]
    info = locate(root, port)
    record = read_pid_file(root)
    out = {"jobs_root": str(root), "port": port, "config_file": str(root / CONFIG_FILE) if cfg["exists"] else None,
           "running": bool(info), "version_cli": __version__, "lock": lock_holder(root)}
    if record.get("pid") and not (info and info.get("managed")):
        out["stale_pid_file"] = record.get("pid")
    token_file = read_token(root) is not None
    out.update(maintenance=maintenance_summary(root), last_start=record.get("started_at"), tz=_tz_brief(root))
    if not info:
        out.update(host=cfg["host"], address=f"http://{cfg['host']}:{port}/", log_file=cfg["log_file"], token_file=token_file,
                   queue=queue_from_disk(root), disk_free_gb=disk_free_gb(root), gpu=gpu_summary(cfg["env"].get("CUDA_VISIBLE_DEVICES")),
                   login=login_mode(root, cfg["host"], cfg.get("env")))
        if out.get("stale_pid_file"):
            # O6: service.pid exists but its process is gone — it exited without ``service stop`` (crash, OOM kill,
            # reboot).  Show the end of the log and the command that brings it back.
            log_path = Path(record.get("log_file") or cfg["log_file"] or root / LOG_FILE)
            tail = _tail(log_path, LOG_TAIL_LINES)
            console = Path(record.get("console_log") or root / CONSOLE_FILE)
            out.update(crashed=True, crash_log=str(log_path), log_tail=tail.splitlines(),
                       console_tail=_tail(console, 10).splitlines(), console_log=str(console),
                       start_command=start_command(root))
        return out
    env = process_env(info["pid"]) if info["own"] else None
    health = probe_health(info["host"], info["port"])
    now = time.time()
    log_file = info.get("log_file") or ((info.get("pid_record") or {}).get("log_file") if info["managed"] else None)
    out.update(pid=info["pid"], managed=info["managed"], own=info["own"], host=info["host"], address=f"http://{info['host']}:{info['port']}/",
               shared=not is_loopback(info["host"]), started_at=info["started_at"],
               uptime_s=round(now - info["started_ts"]) if info.get("started_ts") else None,
               healthy=bool(health and health.get("ok")), version=(health or {}).get("version"), legacy=bool((health or {}).get("legacy")),
               device=info.get("device"), argv=redact_argv(info["argv"]), cwd=info["cwd"], process_jobs_root=info["jobs_root"],
               root_matches=info["root_matches"], log_file=log_file, stdout=info.get("stdout"),
               queue=queue_from_disk(info["jobs_root"]), disk_free_gb=disk_free_gb(info["jobs_root"]),
               env=kept_env(env) if env is not None else None,
               token_in_env=bool(env and env.get("WSS_DEPLOY_TOKEN")) or bool(info.get("token")), token_file=token_file,
               gpu=gpu_summary((env or {}).get("CUDA_VISIBLE_DEVICES") if env is not None else None),
               login=login_mode(info["jobs_root"], info["host"], kept_env(env) if env is not None else cfg.get("env")),
               ready=None)
    if health and health.get("ok") and not health.get("legacy"):
        ready = probe_ready(info["host"], info["port"])
        out["ready"] = bool(ready and ready.get("ok"))
        out["healthy"] = out["ready"]
    return out


# ---------------------------------------------------------------------------------------------- O3 summary
LOGIN_LABELS = {"none": "本机回环免登录", "password": "用户名登录（users.json）", "token": "仅令牌登录"}


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def login_mode(root: Path, host: str, env: dict | None = None) -> str:
    """How the service of ``root`` bound to ``host`` authenticates: ``none`` (loopback, no ``WSS_DEPLOY_TRUST_PROXY``),
    ``password`` (users.json has an enabled user) or ``token`` — the same rule as ``SessionStore.login_mode``."""
    proxy = _truthy(os.environ.get("WSS_DEPLOY_TRUST_PROXY")) if env is None else _truthy((env or {}).get("WSS_DEPLOY_TRUST_PROXY"))
    if is_loopback(host) and not proxy:
        return "none"
    users = _json(Path(root) / "users.json")
    enabled = any(isinstance(row, dict) and not row.get("disabled") for row in users.values())
    return "password" if enabled else "token"


def maintenance_summary(root: Path) -> dict | None:
    """``maintenance_result.json`` of the last ``service upgrade`` (the new service writes it) plus whether a request
    is still queued; ``upgraded_at`` = when that upgrade was requested (older results: when maintenance started)."""
    from .jobs import MAINTENANCE_FILE, MAINTENANCE_RESULT
    result = _json(Path(root) / MAINTENANCE_RESULT)
    pending = (Path(root) / MAINTENANCE_FILE).is_file()
    if not result and not pending:
        return None
    count = lambda key: len(result.get(key) or []) if isinstance(result.get(key), list) else 0  # noqa: E731
    return {"upgraded_at": result.get("requested_at") or result.get("started_at"), "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at"), "seconds": result.get("seconds"), "version": result.get("version"),
            "requested_by": result.get("requested_by"), "complete": result.get("complete"),
            "rebuilt": count("rebuilt"), "rebuild_failed": count("rebuild_failed"), "refreshed": int(result.get("refreshed") or 0),
            "refresh_failed": count("refresh_failed"), "pending_request": pending} if result else \
        {"upgraded_at": None, "pending_request": True}


def _tz_brief(root: Path) -> dict:
    try:
        info = clock.describe(None if Path(root).resolve() == clock.jobs_root().resolve() else root)
        return {"name": info["name"], "offset": info["offset"], "source": info["source"], "errors": info["errors"]}
    except Exception:  # noqa: BLE001 — informational
        return {}


def install_crash_logging() -> None:
    """O6 (``serve``): an exception that ends any thread (precompute, maintenance, preload, a worker's
    ``BaseException``) is written to the application log (server.log) with a diagnostic id — by default Python
    prints it only to stderr (server.console.log).  Worker threads are not restarted: ``/api/ready`` answers 503
    while one is dead (``JobManager.health``); the precompute thread is restarted by the next outlet confirmation."""
    import logging
    log = logging.getLogger("wss_deploy.service")
    previous = threading.excepthook

    def hook(args):
        if issubclass(args.exc_type, SystemExit):
            return
        name = getattr(args.thread, "name", "?")
        try:
            log.critical("Thread %s died; diagnostic_id=%s", name, secrets.token_hex(6),
                         exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        except Exception:  # noqa: BLE001 — never lose the traceback
            previous(args)
    hook._wss_crash_logging = True
    if not getattr(threading.excepthook, "_wss_crash_logging", False):
        threading.excepthook = hook


def start_command(root: Path) -> str:
    extra = "" if _same_root(root, default_jobs_root()) else f" --jobs-root {Path(root)}"
    return f"python -m wss_deploy.cli service start{extra}"


def server_summary(httpd) -> dict:
    """``/api/ready`` for a logged-in administrator (O3): the ``service status`` facts of this process, from memory."""
    manager = httpd.manager
    root = Path(manager.root)
    host, port = httpd.server_address[:2]
    started = getattr(httpd, "started_ts", None)
    return {"version": __version__, "pid": os.getpid(), "started_at": _iso(started),
            "uptime_s": round(time.time() - started) if started else None,
            "bind": {"host": host, "port": port, "shared": bool(httpd.sessions.shared)},
            "login": httpd.sessions.login_mode(), "queue": manager.queue_counts(),
            "maintenance": maintenance_summary(root), "last_start": read_pid_file(root).get("started_at"),
            "disk_free_gb": disk_free_gb(root), "tz": _tz_brief(root)}


def _duration(seconds) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f"{days} 天 {hours} 小时"
    if hours:
        return f"{hours} 小时 {minutes} 分"
    return f"{minutes} 分 {seconds % 60} 秒" if minutes else f"{seconds} 秒"


def _gpu_text(gpu: dict | None) -> str:
    if not gpu:
        return "—"
    visible = gpu.get("visible_devices")
    scope = "CUDA_VISIBLE_DEVICES 未设置" if visible is None else f"CUDA_VISIBLE_DEVICES={visible!r}"
    if not gpu.get("available"):
        return f"不可用（{gpu.get('note') or scope}）"
    names = gpu.get("names") or [gpu.get("name")]
    label = names[0] if len(set(names)) == 1 else "、".join(names)
    return f"{gpu.get('count')} × {label}（{scope}）"


def format_status(info: dict) -> str:
    rows = []
    if not info["running"]:
        head = "服务状态：未运行"
        if info.get("stale_pid_file"):
            head += f"（service.pid 存在但进程 {info['stale_pid_file']} 已不在：服务没有经 service stop 就退出了——崩溃、被杀或机器重启）"
        rows += [("配置", info.get("config_file") or "（无 service.json，使用默认值）"), ("地址", info.get("address")),
                 ("登录", LOGIN_LABELS.get(info.get("login"), info.get("login") or "—")),
                 ("任务目录", info["jobs_root"]), ("日志", info.get("log_file"))]
    else:
        if info["managed"]:
            head = "服务状态：运行中（由 service 管理）"
        else:
            head = "服务状态：运行中（手工启动，未托管；首次 `service restart` 会接管：沿用其参数与环境并写入 service.json）"
        if not info.get("own"):
            head += "；进程属于其他用户，只能查看"
        if not info.get("healthy"):
            head += "；健康检查未响应"
        mode = "共享模式" if info.get("shared") else "本机回环"
        version = info.get("version") or ("旧版（无 /api/health）" if info.get("legacy") else "—")
        started = f"（启动于 {info['started_at']}）" if info.get("started_at") else ""
        env = info.get("env")
        env_text = "（无法读取）" if env is None else (", ".join(f"{k}={v}" for k, v in env.items()) or "（无 WSS_DEPLOY_* / CUDA_VISIBLE_DEVICES / TZ）")
        if info.get("token_in_env"):
            env_text += "；WSS_DEPLOY_TOKEN 已设置" + ("" if info.get("token_file") else "（接管时迁入 .service_token）")
        log = info.get("log_file")
        if not log:
            log = f"（无日志文件，标准输出 → {info.get('stdout') or '未知'}；接管后写入 {Path(info['jobs_root']) / LOG_FILE}）"
        if info.get("ready") is False and info.get("healthy") is False and "健康检查未响应" not in head:
            head += "；就绪检查 /api/ready 返回 503（工作线程 / 发布包 / 目录 / 磁盘 / VMTK 有一项不通过，见 service logs）"
        rows += [("PID", str(info["pid"])), ("地址", f"{info['address']}（{mode}）"), ("登录", LOGIN_LABELS.get(info.get("login"), info.get("login") or "—")),
                 ("运行时长", _duration(info.get("uptime_s")) + started),
                 ("版本", version), ("设备", info.get("device") or "auto"), ("GPU", _gpu_text(info.get("gpu"))),
                 ("任务目录", info.get("process_jobs_root") + ("" if info.get("root_matches") else "（与 --jobs-root 不一致）")),
                 ("日志", log), ("环境", env_text)]
    lock = info.get("lock")
    if lock is not None or info["running"]:
        if lock and lock.get("held"):
            text = f"PID {lock.get('pid')}（{'服务' if lock.get('purpose') == 'serve' else lock.get('purpose')}）持有"
        elif info["running"]:
            text = "未持有（运行中的是 v0.14 之前的进程：`service upgrade` 后生效）"
        else:
            text = "未持有"
        rows.append(("写锁", text))
    queue = info.get("queue") or {}
    rows.append(("队列", " · ".join(f"{label} {queue.get(key, 0)}" for key, label in QUEUE_LABELS)))
    free = info.get("disk_free_gb")
    rows.append(("磁盘", "—" if free is None else f"剩余 {free} GB" + ("（不足 20 GB）" if free < 20 else "")))
    maintenance = info.get("maintenance")
    if maintenance:
        text = f"{maintenance.get('upgraded_at') or '时间未知'}"
        if maintenance.get("finished_at"):
            text += (f"；后台维护完成于 {maintenance['finished_at']}：重建 {maintenance.get('rebuilt', 0)}（失败 {maintenance.get('rebuild_failed', 0)}），"
                     f"刷新报告 {maintenance.get('refreshed', 0)}（失败 {maintenance.get('refresh_failed', 0)}）")
            if maintenance.get("complete") is False:
                text += "，未全部完成（服务在维护中途停止）"
        if maintenance.get("pending_request"):
            text += "；还有排队的维护请求（下次启动执行）"
        rows.append(("最近升级", text))
    if info.get("last_start"):
        rows.append(("最近启动", str(info["last_start"])))
    tz = info.get("tz") or {}
    if tz.get("name"):
        rows.append(("时区", f"{tz['name']}（UTC{tz.get('offset')}，{ {'WSS_DEPLOY_TZ': '环境变量', 'service.json': 'service.json', 'pending': '本次 --env', 'system': '系统'}.get(tz.get('source'), tz.get('source')) }）"))
    if not info["running"]:
        rows.append(("GPU", _gpu_text(info.get("gpu"))))
    width = max(_width(label) for label, _ in rows)
    lines = [head] + [f"  {label}{' ' * (width - _width(label))}  {value}" for label, value in rows]
    if info.get("crashed"):
        tail = info.get("log_tail") or []
        lines += ["", f"{info.get('crash_log')} 最后 {len(tail)} 行：" if tail else f"{info.get('crash_log')} 没有内容。"]
        lines += [f"  {line}" for line in tail]
        if info.get("console_tail"):
            lines += [f"{info.get('console_log')}（标准输出 / 错误）最后 {len(info['console_tail'])} 行："]
            lines += [f"  {line}" for line in info["console_tail"]]
        lines += ["", f"重新启动：{info.get('start_command')}（排队的任务会自动续排，计算中的标为「中断」可在网页重试）"]
    return "\n".join(lines)


def _width(text: str) -> int:
    """Terminal columns: CJK characters take two."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1 for ch in text)


def active_jobs(jobs_root: Path) -> dict:
    counts = queue_from_disk(jobs_root)
    return {key: counts.get(key, 0) for key in ACTIVE}


def _alive(pid: int) -> bool:
    try:
        # Reap our own child (tests start and stop in one process); harmless for other processes.
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    except OSError:
        pass
    return _proc_state(pid) not in {None, "Z", "X"}


def _signal_if_service(pid: int, sig: int) -> bool:
    """Send ``sig`` only while ``pid`` still runs ``wss_deploy.cli serve`` (guards against PID reuse)."""
    if describe_process(pid) is None:
        return False
    os.kill(pid, sig)
    return True


def drain(root: Path, pid: int, *, timeout: float = DRAIN_DEFAULT_S, out=print, poll: float = 1.0) -> dict:
    """J6: ask service ``pid`` to start no new work (``.drain.json``) and wait up to ``timeout`` s for running jobs.

    Queued jobs stay queued (the next service re-queues them); the drain file is removed by :func:`stop`.
    """
    from .jobs import DRAIN_FILE, atomic_json
    root = Path(root).resolve()
    atomic_json(root / DRAIN_FILE, {"pid": int(pid), "at": _now(), "timeout_s": float(timeout), "by": redact_argv(sys.argv)[1:4]})
    started = time.time()
    running = active_jobs(root)["running"]
    out(f"排空中：服务（PID {pid}）不再开始新的计算；等待 {running} 个正在计算的任务结束（最多 {timeout:.0f} 秒）…" if running
        else f"排空中：服务（PID {pid}）不再开始新的计算；当前没有正在计算的任务。")
    last_report = started
    while running and time.time() - started < timeout and _alive(pid):
        time.sleep(poll)
        running = active_jobs(root)["running"]
        if running and time.time() - last_report >= 15:
            out(f"  仍有 {running} 个任务在计算（已等待 {time.time() - started:.0f} 秒）…")
            last_report = time.time()
    waited = round(time.time() - started, 1)
    if running:
        out(f"排空超时：{waited:.0f} 秒后仍有 {running} 个任务在计算，停止后它们会标记为「中断」，可在网页上「重试」。")
    else:
        out(f"排空完成（{waited:.0f} 秒）。")
    return {"drained": not running, "waited_s": waited, "running": running}


def stop(root: Path, port: int | None = None, *, timeout: float = STOP_TIMEOUT_S, out=print, drain_s: float | None = None) -> dict:
    """SIGTERM → wait ``timeout`` → SIGKILL, only for the matching serve process; removes our PID file.

    ``drain_s`` (``--drain``): first stop starting new work and wait that long for running jobs (J6)."""
    from .jobs import DRAIN_FILE
    root = Path(root).resolve()
    info = locate(root, port or load_config(root)["port"])
    if not info:
        (root / PID_FILE).unlink(missing_ok=True)
        out("服务没有在运行。")
        return {"stopped": False, "running": False}
    if not info["own"]:
        raise ServiceError(f"进程 {info['pid']} 属于其他用户，不能停止。")
    if not info["root_matches"]:
        raise ServiceError(f"端口 {info['port']} 上的服务使用任务目录 {info['jobs_root']}，与 {root} 不一致；请用 --jobs-root 指向它。")
    drained = None
    if drain_s is not None:
        drained = drain(root, info["pid"], timeout=drain_s, out=out)
    active = active_jobs(info["jobs_root"])
    if active["running"] or active["queued"]:
        out(f"注意：有 {active['running']} 个任务正在计算、{active['queued']} 个在排队；排队的任务在服务重启后自动继续，"
            "正在计算的会标记为「中断」，可在网页上点「重试」（已完成的输入与中心线会复用）。")
    pid = info["pid"]
    out(f"正在停止服务（PID {pid}）…")
    _signal_if_service(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline and _alive(pid):
        time.sleep(0.2)
    killed = False
    if _alive(pid):
        out(f"{timeout:.0f} 秒内未退出，强制结束（SIGKILL）。")
        killed = _signal_if_service(pid, signal.SIGKILL)
        deadline = time.time() + 5
        while time.time() < deadline and _alive(pid):
            time.sleep(0.2)
    if _alive(pid):
        raise ServiceError(f"进程 {pid} 仍未退出。")
    record = read_pid_file(root)
    if record.get("pid") == pid:
        (root / PID_FILE).unlink(missing_ok=True)
    (root / DRAIN_FILE).unlink(missing_ok=True)
    out("服务已停止。")
    return {"stopped": True, "pid": pid, "killed": killed, "active": active, "drain": drained}


def adopt(root: Path, info: dict, *, out=print) -> dict:
    """Turn a hand-started process into ``service.json`` (+ ``.service_token``); returns the new config."""
    root = Path(root).resolve()
    if not info.get("own"):
        raise ServiceError(f"进程 {info['pid']} 属于其他用户，不能接管。")
    env = process_env(info["pid"]) or {}
    token = env.get("WSS_DEPLOY_TOKEN") or info.get("token")
    if not token and info.get("token_file"):
        token_path = Path(info["token_file"])
        if not token_path.is_absolute() and info.get("cwd"):
            token_path = Path(info["cwd"]) / token_path
        try:
            token = token_path.read_text(encoding="utf-8").strip() or None
        except OSError:
            token = None
    argv = redact_argv(info["argv"])        # J10: a --token value never lands in service.json
    cfg = load_config(root)
    cfg.update(host=info["host"], port=info["port"], device=info.get("device") or "auto", python=info.get("python") or sys.executable,
               cwd=info.get("cwd") or str(PROJECT_ROOT), env=kept_env(env), argv=argv,
               log_file=info.get("log_file") or cfg.get("log_file") or str(root / LOG_FILE),
               adopted_from={"pid": info["pid"], "started_at": info.get("started_at"), "argv": argv, "at": _now()})
    if token:
        previous = read_token(root)
        write_token(root, token)
        if previous and previous != token:
            out("已用运行中服务的令牌覆盖 .service_token（原文件中的令牌与进程不一致）。")
        else:
            out(f"令牌已从运行中进程的环境迁入 {root / TOKEN_FILE}（0600）。")
    save_config(root, cfg)
    kept = ", ".join(f"{k}={v}" for k, v in cfg["env"].items()) or "无"
    out(f"已接管手工启动的服务（PID {info['pid']}）：{info['host']}:{info['port']}，设备 {cfg['device']}，沿用环境 {kept}；写入 {root / CONFIG_FILE}。")
    return load_config(root)


def _child_env(cfg: dict, root: Path) -> dict:
    env = {key: value for key, value in os.environ.items()
           if not (key.startswith(ENV_PREFIX) or key in ENV_KEYS)}
    env.update(kept_env(cfg.get("env") or {}))
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if not is_loopback(cfg["host"]):
        token = read_token(root)
        if not token:
            token = secrets.token_urlsafe(32)
            write_token(root, token)
        env["WSS_DEPLOY_TOKEN"] = token
    return env


def _tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        with open(path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 64 * 1024))
            data = stream.read().decode("utf-8", "replace")
        return "\n".join(data.splitlines()[-lines:])
    except OSError:
        return ""


def wait_healthy(host: str, port: int, *, pid: int | None = None, timeout: float = HEALTH_TIMEOUT_S,
                 detail: dict | None = None) -> dict | None:
    """Wait until ``/api/ready`` answers 200 (v0.14; older services: ``/api/health``); None on timeout / exit.

    ``detail`` (a dict) receives the last answer under ``last`` so a failure can name the failing checks."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        health = probe_ready(host, port, timeout=2)
        if detail is not None and health is not None:
            detail["last"] = health
        if health and health.get("ok"):
            return health
        if pid is not None and not _alive(pid):
            return None
        time.sleep(0.5)
    return None


def failing_checks(answer: dict | None) -> list[str]:
    """Names of the hard checks a ``/api/ready`` answer reports as failing (for messages)."""
    checks = (answer or {}).get("checks") if isinstance(answer, dict) else None
    if not isinstance(checks, dict):
        return []
    return [name for name, check in checks.items() if isinstance(check, dict) and check.get("ok") is False]


def apply_overrides(cfg: dict, *, host=None, port=None, device=None, env_items=None, unset_env=None, python=None, log_file=None) -> dict:
    """Command-line overrides for start/restart; ``--env KEY=VALUE`` (``KEY=`` keeps an empty value, e.g. CPU only),
    ``--unset-env KEY`` drops a saved variable."""
    if host:
        cfg["host"] = host
    if port:
        cfg["port"] = int(port)
    if device:
        cfg["device"] = device
    if python:
        cfg["python"] = python
    if log_file:
        cfg["log_file"] = log_file
    for item in env_items or []:
        if "=" not in item:
            raise ServiceError(f"--env 需要 KEY=VALUE 形式：{item}")
        key, value = item.split("=", 1)
        if key in SECRET_ENV:
            raise ServiceError("令牌不能写进 service.json；共享模式令牌在 .service_token（`service token`）。")
        if not (key.startswith(ENV_PREFIX) or key in ENV_KEYS):
            raise ServiceError(f"只保存 WSS_DEPLOY_*、CUDA_VISIBLE_DEVICES、TZ：{key}")
        cfg.setdefault("env", {})[key] = value
    for key in unset_env or []:
        cfg.setdefault("env", {}).pop(key, None)
    return cfg


def start(root: Path, *, timeout: float = HEALTH_TIMEOUT_S, out=print, **overrides) -> dict:
    """Start the service detached (new session), write ``service.pid`` and wait for ``/api/health``."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = apply_overrides(load_config(root), **overrides)
    running = locate(root, cfg["port"])
    if running:
        raise ServiceError(f"端口 {cfg['port']} 上已有服务在运行（PID {running['pid']}）；需要重启请用 `service restart`。")
    if _port_busy(cfg["host"], cfg["port"]):
        raise ServiceError(f"端口 {cfg['port']} 已被其他程序占用。")
    log_file = Path(cfg.get("log_file") or root / LOG_FILE)
    argv = [cfg.get("python") or sys.executable, "-u", "-m", "wss_deploy.cli", "serve", "--host", cfg["host"], "--port", str(cfg["port"]),
            "--jobs-root", str(root), "--device", cfg.get("device") or "auto", "--log-file", str(log_file)]
    # O1: saved *before* the child starts, so the child (and any CLI command) already reads the new env.TZ.
    cfg["argv"] = argv
    save_config(root, cfg)
    env = _child_env(cfg, root)
    console_path = root / CONSOLE_FILE
    # v0.15: stdout / stderr may carry tracebacks — created 0600 whatever the umask (an existing file is narrowed).
    with os.fdopen(os.open(console_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), "ab") as console:
        os.fchmod(console.fileno(), 0o600)
        console.write(f"\n===== {_now()} service start: {' '.join(argv)}\n".encode("utf-8"))
        console.flush()
        proc = subprocess.Popen(argv, cwd=cfg.get("cwd") or str(PROJECT_ROOT), env=env, stdin=subprocess.DEVNULL,
                                stdout=console, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    started = time.time()
    record = {"pid": proc.pid, "started_at": _iso(started), "host": cfg["host"], "port": cfg["port"], "argv": argv,
              "version": __version__, "jobs_root": str(root), "cwd": cfg.get("cwd"), "log_file": str(log_file),
              "console_log": str(console_path)}
    _write_json(root / PID_FILE, record, mode=0o644)
    out(f"服务启动中（PID {proc.pid}），等待就绪检查（/api/ready）…")
    detail: dict = {}
    health = wait_healthy(cfg["host"], cfg["port"], pid=proc.pid, timeout=timeout, detail=detail)
    if not health:
        tail = _tail(log_file, LOG_TAIL_LINES) or _tail(console_path, LOG_TAIL_LINES)
        failing = failing_checks(detail.get("last"))
        reason = f"未通过的检查：{'、'.join(failing)}。" if failing else ""
        if _alive(proc.pid):
            out(f"{timeout:.0f} 秒内就绪检查未通过；进程仍在运行（PID {proc.pid}）。{reason}\n{log_file} 最后 {LOG_TAIL_LINES} 行：\n{tail}")
        else:
            (root / PID_FILE).unlink(missing_ok=True)
            out(f"服务启动失败，进程已退出。{reason}\n{log_file} 最后 {LOG_TAIL_LINES} 行：\n{tail}\n控制台输出：{console_path}")
        raise ServiceError("新服务未能就绪：" + (reason or "请根据上面的日志修正后执行 `python -m wss_deploy.cli service start`。"))
    address = f"http://{cfg['host']}:{cfg['port']}/"
    out(f"服务已就绪：{address}（版本 {health.get('version')}，PID {proc.pid}，日志 {log_file}）")
    if not is_loopback(cfg["host"]):
        out("共享模式：令牌见 `python -m wss_deploy.cli service token`" + ("；已启用用户名登录（users.json）。" if (root / "users.json").is_file() else "。"))
    return {**record, "health": health, "address": address}


def restart(root: Path, port: int | None = None, *, timeout: float = HEALTH_TIMEOUT_S, stop_timeout: float = STOP_TIMEOUT_S,
            out=print, drain_s: float | None = None, **overrides) -> dict:
    """stop + start; a hand-started process is adopted first (argv, env, token) so nothing is lost.

    ``--env CUDA_VISIBLE_DEVICES=1`` pins the GPU (saved in service.json); ``--drain`` waits for running jobs."""
    root = Path(root).resolve()
    cfg = load_config(root)
    info = locate(root, port or overrides.get("port") or cfg["port"])
    if info:
        if not info["root_matches"]:
            raise ServiceError(f"端口 {info['port']} 上的服务使用任务目录 {info['jobs_root']}，与 {root} 不一致；请用 --jobs-root 指向它。")
        if not info["managed"] or not cfg["exists"]:
            adopt(root, info, out=out)
        stop(root, info["port"], timeout=stop_timeout, out=out, drain_s=drain_s)
    elif port and not overrides.get("port"):
        overrides["port"] = port
    return start(root, timeout=timeout, out=out, **overrides)


def _has_server_paths(value) -> bool:
    """True when any string inside ``value`` mentions the project root or the home directory (see schema.redact_paths)."""
    from .schema import redact_paths
    try:
        return redact_paths(value) != value
    except Exception:
        return False


def needs_rebuild(summary: dict, current: str | None = None) -> bool:
    """J9: a summary written by an older analysis needs a full ``rebuild_report`` (not just a UI refresh).

    ``summary.analysis_version`` (v0.14+) is compared with :data:`schema.ANALYSIS_VERSION`; records written before
    the field existed fall back to the key probes: TAWSS/OSI results without the v0.12 cycle findings or without
    the v0.13 derived RRT / ECAP blocks."""
    if not isinstance(summary, dict) or not summary:
        return False
    if current is None:
        from .schema import ANALYSIS_VERSION as current
    version = summary.get("analysis_version")
    if isinstance(version, str) and version:
        return version < current
    # v0.15.1: records written before the field existed (all pre-v0.14 jobs) still carry the server's absolute
    # paths in ``model_release``; a rebuild rewrites them redacted, so such a summary always qualifies.
    if _has_server_paths(summary.get("model_release")):
        return True
    findings = summary.get("findings") if isinstance(summary.get("findings"), dict) else {}
    cycle = summary.get("cycle") if isinstance(summary.get("cycle"), dict) else {}
    fields = cycle.get("fields") if isinstance(cycle.get("fields"), dict) else {}
    no_derived = "tawss" in fields and "osi" in fields and not ("rrt" in fields and "ecap" in fields)
    return bool(cycle and ("cycle_criteria" not in findings or no_derived))


def pending_rebuilds(root: Path) -> list[str]:
    """Finished jobs whose ``summary.json`` predates the analysis of the current code (see :func:`needs_rebuild`)."""
    out = []
    for job_dir in sorted(Path(root).iterdir()) if Path(root).is_dir() else []:
        if job_dir.name.startswith("."):      # .tmp, .trash, .deleting_* … are never jobs
            continue
        summary = _json(job_dir / "summary.json") if (job_dir / "job.json").is_file() else {}
        if not summary or _json(job_dir / "job.json").get("status") != "done":
            continue
        if needs_rebuild(summary):
            out.append(job_dir.name)
    return out


def _preflight_env(cfg: dict) -> dict:
    env = {key: value for key, value in os.environ.items() if not (key.startswith(ENV_PREFIX) or key in ENV_KEYS)}
    env.update(kept_env(cfg.get("env") or {}))
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if (cfg.get("env") or {}).get("TZ") and not env.get(clock.ENV_TZ):
        env[clock.ENV_TZ] = cfg["env"]["TZ"]      # O1: doctor resolves the TZ the upgrade is about to save
    return env


def preflight(root: Path, cfg: dict | None = None, *, out=print, timeout: float = 600.0) -> dict:
    """J6: check the code on disk *before* stopping anything — the service modules import in a fresh interpreter
    (same python / cwd / saved environment as the service) and ``doctor`` (without the service checks) has no ✗.
    Raises :class:`ServiceError` on failure; the running service is untouched."""
    root = Path(root).resolve()
    cfg = cfg or load_config(root)
    python, cwd, env = cfg.get("python") or sys.executable, cfg.get("cwd") or str(PROJECT_ROOT), _preflight_env(cfg)
    out("升级预检：导入新代码的服务模块 …")
    try:
        proc = subprocess.run([python, "-c", PREFLIGHT_IMPORTS], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ServiceError(f"升级预检失败：无法运行 {python}：{exc}。服务保持运行，未做任何改动。") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-LOG_TAIL_LINES:])
        raise ServiceError(f"升级预检失败：新代码的服务模块无法导入。服务保持运行，未做任何改动。\n{tail}")
    out("升级预检：doctor（不含服务检查）…")
    try:
        argv = [python, "-m", "wss_deploy.cli", "doctor", "--json", "--skip", "service", "--jobs-root", str(root)]
        if cfg.get("host"):
            argv += ["--host", str(cfg["host"])]      # v0.15: the exposure check judges the bind address being started
        proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ServiceError(f"升级预检失败：doctor 未能运行：{exc}。服务保持运行，未做任何改动。") from exc
    try:
        rows = json.loads(proc.stdout).get("checks") or []
    except (ValueError, AttributeError):
        rows = []
    failed = [row for row in rows if isinstance(row, dict) and row.get("status") == "fail"]
    if proc.returncode != 0 or failed:
        lines = [f"  ✗ {row.get('label')}：{row.get('detail')}" for row in failed] or \
                ["\n".join((proc.stderr or proc.stdout).strip().splitlines()[-LOG_TAIL_LINES:])]
        raise ServiceError("升级预检失败：doctor 报告错误。服务保持运行，未做任何改动。\n" + "\n".join(lines))
    warned = sum(1 for row in rows if isinstance(row, dict) and row.get("status") == "warn")
    out(f"升级预检通过（doctor {len(rows)} 项，⚠ {warned}）。")
    return {"checks": len(rows), "warn": warned}


def queue_maintenance(root: Path, rebuild: list[str], *, refresh_reports: bool = True) -> dict:
    """Ask the next service start to rebuild ``rebuild`` and refresh stale reports (``.maintenance.json``)."""
    from .jobs import MAINTENANCE_FILE, atomic_json
    root = Path(root).resolve()
    previous = _json(root / MAINTENANCE_FILE)
    targets = list(dict.fromkeys([*(previous.get("rebuild") or []), *rebuild]))    # an unfinished request is merged
    request = {"schema_version": "wss-deploy.maintenance/v1", "request_id": secrets.token_hex(8), "requested_at": _now(),
               "requested_by": __version__, "rebuild": targets, "refresh_reports": bool(refresh_reports or previous.get("refresh_reports"))}
    atomic_json(root / MAINTENANCE_FILE, request)
    return request


def wait_maintenance(root: Path, request_id: str, *, timeout: float = MAINTENANCE_TIMEOUT_S, out=print, poll: float = 1.0) -> dict | None:
    """The new service's maintenance result for ``request_id`` (None when it did not finish within ``timeout``)."""
    from .jobs import MAINTENANCE_RESULT
    path = Path(root) / MAINTENANCE_RESULT
    deadline, last = time.time() + timeout, time.time()
    while time.time() < deadline:
        result = _json(path)
        if result.get("request_id") == request_id and result.get("finished_at"):
            return result
        if time.time() - last >= 15:
            out("  新服务正在后台重建分析 / 刷新报告 …")
            last = time.time()
        time.sleep(poll)
    return None


def upgrade(root: Path, port: int | None = None, *, rebuild=(), auto_rebuild: bool = True, timeout: float = HEALTH_TIMEOUT_S,
            stop_timeout: float = STOP_TIMEOUT_S, out=print, drain_s: float | None = None, preflight_check: bool = True,
            maintenance_timeout: float = MAINTENANCE_TIMEOUT_S, **overrides) -> dict:
    """Put the code on disk into service in one step (v0.14 order):

    1. preflight — the new code imports and ``doctor`` passes (else abort; the old service keeps running);
    2. adopt / (``--drain``) / stop the running service;
    3. queue the maintenance — full rebuild of the named jobs plus :func:`pending_rebuilds`, then every stale
       report — in ``.maintenance.json``;
    4. start and wait for ``/api/ready``; the *new* service (holding the jobs-root lock) runs the maintenance in
       the background with each rebuilt job locked against edits, and this command waits for its result.

    The pre-v0.14 order (rebuild while the service was down) needed the jobs root without a writer; with the
    single-writer lock only the service writes, and the rebuild runs the new code by construction."""
    root = Path(root).resolve()
    cfg = load_config(root)
    if preflight_check:
        # v0.15: the preflight judges the configuration that is about to start (``--host`` / ``--env`` overrides
        # included), so e.g. ``--host 0.0.0.0`` without an enabled admin is refused before anything stops.
        effective = apply_overrides(json.loads(json.dumps(cfg, default=str)),
                                    **{k: v for k, v in overrides.items() if k in {"host", "port", "device", "env_items", "unset_env", "python"}})
        preflight(root, effective, out=out)
    info = locate(root, port or overrides.get("port") or cfg["port"])
    if info:
        if not info["root_matches"]:
            raise ServiceError(f"端口 {info['port']} 上的服务使用任务目录 {info['jobs_root']}，与 {root} 不一致；请用 --jobs-root 指向它。")
        if not info["managed"] or not cfg["exists"]:
            adopt(root, info, out=out)
        stop(root, info["port"], timeout=stop_timeout, out=out, drain_s=drain_s)
        if not overrides.get("port"):
            overrides["port"] = info["port"]
    report = {"rebuilt": [], "rebuild_failed": [], "refreshed": 0, "refresh_failed": [], "maintenance": None}
    targets = list(dict.fromkeys(list(rebuild or []) + (pending_rebuilds(root) if auto_rebuild else [])))
    request = queue_maintenance(root, targets)
    out(f"已排入新服务的后台维护：完整重建 {len(request['rebuild'])} 个任务，刷新过期报告。")
    try:
        started = start(root, timeout=timeout, out=out, **overrides)
    except ServiceError:
        out(f"新服务未就绪：旧服务已停止，维护请求保留在 {root / '.maintenance.json'}，下次成功启动时执行。")
        raise
    result = wait_maintenance(root, request["request_id"], timeout=maintenance_timeout, out=out)
    if result is None:
        report["maintenance"] = "pending"
        out(f"后台维护在 {maintenance_timeout:.0f} 秒内未完成，服务会继续执行；结果见 {root / 'maintenance_result.json'}。")
        return {**started, "upgrade": report}
    report.update(maintenance="done", rebuilt=list(result.get("rebuilt") or []),
                  rebuild_failed=[item.get("id") if isinstance(item, dict) else item for item in result.get("rebuild_failed") or []],
                  refreshed=int(result.get("refreshed") or 0),
                  refresh_failed=[item.get("id") if isinstance(item, dict) else item for item in result.get("refresh_failed") or []])
    for job_id in report["rebuilt"]:
        out(f"✓ 已完整重建 {job_id}（重算发现 / 沿程 / 结论，预测数值不变）")
    for item in result.get("rebuild_failed") or []:
        out(f"✗ 重建 {item.get('id') if isinstance(item, dict) else item} 失败：{(item or {}).get('error') if isinstance(item, dict) else ''}")
    out(f"报告模板：刷新 {report['refreshed']} 个" + (f"，失败 {len(report['refresh_failed'])} 个" if report["refresh_failed"] else "") + "。")
    return {**started, "upgrade": report}


def logs(root: Path, *, lines: int = 80, follow: bool = False, out=print) -> int:
    root = Path(root).resolve()
    record = read_pid_file(root)
    path = Path(record.get("log_file") or load_config(root).get("log_file") or root / LOG_FILE)
    if not path.is_file():
        out(f"日志文件不存在：{path}（由 `service start` 启动后生成）。")
        return 1
    out(_tail(path, lines))
    if not follow:
        return 0
    try:
        stream = open(path, "r", encoding="utf-8", errors="replace")
        stream.seek(0, os.SEEK_END)
        inode = os.fstat(stream.fileno()).st_ino
        while True:
            chunk = stream.read()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                continue
            time.sleep(1)
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_ino != inode or st.st_size < stream.tell():   # rotated
                stream.close()
                stream = open(path, "r", encoding="utf-8", errors="replace")
                inode = os.fstat(stream.fileno()).st_ino
    except KeyboardInterrupt:
        return 0


def token(root: Path, *, rotate: bool = False, out=print) -> int:
    root = Path(root).resolve()
    if rotate:
        value = secrets.token_urlsafe(32)
        write_token(root, value)
        out(value)
        print("已生成新令牌；运行中的服务仍使用旧令牌，`service restart` 后生效。", file=sys.stderr)
        return 0
    value = read_token(root)
    if value:
        out(value)
        return 0
    info = locate(root)
    if info and not info["managed"] and (process_env(info["pid"]) or {}).get("WSS_DEPLOY_TOKEN"):
        out("令牌仍在手工启动进程的环境里；执行一次 `service restart` 会把它迁入 .service_token。")
    else:
        out("尚未生成令牌（共享模式首次 `service start` 时生成；本机回环模式不需要令牌）。")
    return 1


# ---------------------------------------------------------------------------------------------- HTTP client
MAX_BATCH_FILES = 20
MAX_BATCH_BYTES = 200 * 1024 * 1024   # below the service's 256 MiB batch body limit


class ServiceClient:
    """Minimal standard-library client of a running service (``cli submit`` / ``cli jobs list``).

    Keeps the session cookie and CSRF token itself; never uses an HTTP proxy.
    """
    def __init__(self, base_url: str, *, timeout: float = 120.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.cookie: str | None = None
        self.csrf: str | None = None
        self.session: dict = {}
        self.opener = http_opener()

    def request(self, method: str, path: str, *, body: bytes | None = None, content_type: str | None = None,
                json_body=None) -> tuple[int, object]:
        headers = {"Accept": "application/json"}
        if json_body is not None:
            body, content_type = json.dumps(json_body, ensure_ascii=False).encode("utf-8"), "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        if self.cookie:
            headers["Cookie"] = self.cookie
        if self.csrf and method != "GET":
            headers["X-CSRF-Token"] = self.csrf
        request = urllib.request.Request(self.base + path, data=body, method=method, headers=headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                status, raw, cookie = response.status, response.read(), response.headers.get("Set-Cookie")
        except urllib.error.HTTPError as exc:
            status, raw, cookie = exc.code, exc.read(), exc.headers.get("Set-Cookie") if exc.headers else None
        except (OSError, ValueError) as exc:
            raise ServiceError(f"无法连接服务 {self.base}：{exc}") from exc
        if cookie and cookie.startswith("wss_session="):
            value = cookie.split(";", 1)[0]
            self.cookie = None if value == "wss_session=" else value
        try:
            payload = json.loads(raw or b"{}")
        except ValueError:
            payload = raw
        return status, payload

    @staticmethod
    def _error(payload, fallback: str) -> str:
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            return str(payload["error"].get("message") or fallback)
        return fallback

    def connect(self, *, user: str | None = None, password: str | None = None, token: str | None = None) -> dict:
        """Open a session: loopback needs nothing; shared mode logs in by user name or by the service token."""
        status, payload = self.request("GET", "/api/session")
        if status != 200 or not isinstance(payload, dict):
            raise ServiceError(self._error(payload, f"读取会话失败（HTTP {status}）"))
        if payload.get("authenticated"):
            self.session, self.csrf = payload, payload.get("csrf_token")
            return payload
        if user:
            body = {"username": user, "password": password or ""}
        elif token:
            body = {"token": token}
        else:
            raise ServiceError("共享模式需要登录：用 --user 名字（口令取 WSS_DEPLOY_PASSWORD 或终端输入），"
                               "或设置 WSS_DEPLOY_TOKEN / 在服务机器上用 --jobs-root 读取 .service_token。")
        status, payload = self.request("POST", "/api/session", json_body=body)
        if status != 200 or not isinstance(payload, dict) or not payload.get("authenticated"):
            raise ServiceError(self._error(payload, f"登录失败（HTTP {status}）"))
        self.session, self.csrf = payload, payload.get("csrf_token")
        return payload

    @staticmethod
    def multipart(files: list[tuple[str, bytes]], fields: dict) -> tuple[bytes, str]:
        boundary = "wssdeploy" + secrets.token_hex(12)
        chunks = []
        for name, value in fields.items():
            if value is None or value == "":
                continue
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
                          + str(value).encode("utf-8") + b"\r\n")
        for filename, content in files:
            safe = filename.replace('"', "_").replace("\r", "_").replace("\n", "_")
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="stl"; filename="{safe}"\r\n'
                          f"Content-Type: application/octet-stream\r\n\r\n".encode("utf-8") + content + b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"

    def submit(self, paths: list[Path], fields: dict) -> list[dict]:
        """Upload STL files in batches (≤ 20 files, ≤ 200 MiB); returns one result per file in input order."""
        batches, current, size = [], [], 0
        for path in paths:
            n = Path(path).stat().st_size
            if current and (len(current) >= MAX_BATCH_FILES or size + n > MAX_BATCH_BYTES):
                batches.append(current)
                current, size = [], 0
            current.append(Path(path))
            size += n
        if current:
            batches.append(current)
        results = []
        for batch in batches:
            body, ctype = self.multipart([(p.name, p.read_bytes()) for p in batch],
                                         {**fields, "metadata_json": json.dumps([{} for _ in batch])})
            status, payload = self.request("POST", "/api/jobs/batch", body=body, content_type=ctype)
            if status != 201 or not isinstance(payload, dict):
                message = self._error(payload, f"提交失败（HTTP {status}）")
                results += [{"file": str(p), "error": {"message": message, "status": status}} for p in batch]
                continue
            for item in payload.get("results") or []:
                index = item.get("index")
                entry = dict(item)
                entry["file"] = str(batch[index]) if isinstance(index, int) and 0 <= index < len(batch) else None
                results.append(entry)
        return results

    def jobs(self, *, status: str | None = None, all_owners: bool = False, page_size: int = 100) -> dict:
        query = f"/api/jobs?page_size={int(page_size)}" + (f"&status={urllib.request.quote(status)}" if status else "") + ("&all=1" if all_owners else "")
        code, payload = self.request("GET", query)
        if code != 200 or not isinstance(payload, dict):
            raise ServiceError(self._error(payload, f"读取任务列表失败（HTTP {code}）"))
        return payload


__all__ = ["ServiceClient", "ServiceError", "ServiceLock", "ServiceLockError", "acquire_lock", "adopt", "default_jobs_root", "drain",
           "format_status", "gpu_cache_time", "gpu_summary", "load_config", "locate", "lock_holder", "logs", "needs_rebuild",
           "offline_lock", "pending_rebuilds", "preflight", "probe_health", "probe_ready", "queue_maintenance", "read_token",
           "redact_argv", "restart", "scan_processes", "serve_options", "start", "status", "stop", "token", "upgrade",
           "wait_healthy", "wait_maintenance"]
