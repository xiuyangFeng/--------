"""One clock for every timestamp the deployment writes (v0.15, 上线前可运维加固 O1).

The zone is resolved in this order (first usable value wins):

1. ``WSS_DEPLOY_TZ`` in the environment of this process;
2. ``env.TZ`` of ``<jobs_root>/service.json`` (saved by ``service start|restart|upgrade --env TZ=…``).  The
   service child and every offline CLI command read the same file, so the service and the CLI agree;
3. the system zone (``/etc/localtime``) — deliberately *not* the ``TZ`` of the calling shell: ``service start``
   never passes an unsaved ``TZ`` to the service, so honouring the shell would make the CLI disagree with it.

``now_iso()`` / ``iso(ts)`` give ISO-8601 with the UTC offset at second resolution (``2026-09-26T14:00:00+08:00``);
``strftime(fmt)`` formats the same instant (job ids keep ``YYYYmmdd_HHMMSS``); :class:`LogFormatter` stamps log
records in the same zone (``server.log`` / ``access.log``).  An invalid name at one level falls through to the next
and is reported by :func:`resolve` (``errors``) so ``doctor`` can say so.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
from pathlib import Path

try:  # Python ≥ 3.9
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover — the deployment runs 3.10
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception

ENV_TZ = "WSS_DEPLOY_TZ"
CONFIG_FILE = "service.json"
LOCALTIME = Path("/etc/localtime")
TIMEZONE_FILE = Path("/etc/timezone")

_LOCK = threading.Lock()
_STATE: dict = {"root": None, "pending": None}
_CACHE: dict = {"key": None, "value": None}


def default_jobs_root() -> Path:
    from .paths import PROJECT_ROOT
    return PROJECT_ROOT / "outputs/wss_deploy_jobs"


def configure(jobs_root=None, *, tz: str | None = None) -> None:
    """Read ``service.json`` of ``jobs_root`` from now on (default: the production jobs root, read only).

    ``tz`` is a ``--env TZ=…`` the running command is about to save there: it ranks with ``service.json``."""
    with _LOCK:
        _STATE["root"] = Path(jobs_root) if jobs_root else None
        _STATE["pending"] = (tz or None)
        _CACHE.update(key=None, value=None)


def jobs_root() -> Path:
    return _STATE["root"] or default_jobs_root()


def _zone(name) -> tuple[dt.tzinfo | None, str | None]:
    """``(tzinfo, None)`` for a usable zone name (``Area/City``, ``UTC``, ``:Area/City``, a zoneinfo file path)."""
    text = str(name or "").strip()
    if text.startswith(":"):
        text = text[1:]
    if not text:
        return None, "空值"
    if ZoneInfo is None:
        return None, "zoneinfo 不可用"
    try:
        if text.startswith("/"):
            with open(text, "rb") as stream:
                return ZoneInfo.from_file(stream, key=text), None
        return ZoneInfo(text), None
    except (ZoneInfoNotFoundError, ValueError, OSError, IsADirectoryError) as exc:
        return None, f"无法识别的时区 {text!r}（{type(exc).__name__}）"


def system_zone() -> tuple[str, dt.tzinfo]:
    """The zone of this machine: ``/etc/localtime`` (by its zoneinfo name when it is a link), else ``/etc/timezone``,
    else the fixed offset this process sees."""
    try:
        target = os.path.realpath(LOCALTIME)
        if "zoneinfo/" in target:
            name = target.split("zoneinfo/", 1)[1]
            zone, _ = _zone(name)
            if zone is not None:
                return name, zone
    except OSError:
        pass
    try:
        name = TIMEZONE_FILE.read_text(encoding="utf-8").strip()
        zone, _ = _zone(name)
        if zone is not None:
            return name, zone
    except OSError:
        pass
    if ZoneInfo is not None and LOCALTIME.is_file():
        try:
            with open(LOCALTIME, "rb") as stream:
                return "localtime", ZoneInfo.from_file(stream, key="localtime")
        except (OSError, ValueError):
            pass
    local = dt.datetime.now().astimezone().tzinfo
    return (local.tzname(None) if local else "UTC") or "UTC", local or dt.timezone.utc


def _config_tz(root: Path) -> tuple[str | None, tuple | None]:
    path = Path(root) / CONFIG_FILE
    try:
        st = path.stat()
    except OSError:
        return None, None
    stamp = (st.st_ino, st.st_size, st.st_mtime_ns)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None, stamp
    env = value.get("env") if isinstance(value, dict) else None
    tz = env.get("TZ") if isinstance(env, dict) else None
    return (str(tz) if tz not in (None, "") else None), stamp


def resolve(root=None) -> dict:
    """``{name, source, tzinfo, errors, system}``; ``source`` is ``WSS_DEPLOY_TZ`` / ``pending`` / ``service.json`` /
    ``system``.  ``root`` (default: the configured jobs root) is read without changing the configuration.  Cached
    until the environment variable, the root or its ``service.json`` changes."""
    configured = root is None
    root = jobs_root() if configured else Path(root)
    pending = _STATE["pending"] if configured else None
    env_value = os.environ.get(ENV_TZ) or None
    try:
        st = (Path(root) / CONFIG_FILE).stat()
        stamp = (st.st_ino, st.st_size, st.st_mtime_ns)
    except OSError:
        stamp = None
    key = (env_value, str(root), stamp, pending)
    cached = _CACHE.get("value")
    if _CACHE.get("key") == key and cached is not None:
        return cached
    errors = []
    candidates = [(ENV_TZ, env_value)]
    if pending:
        candidates.append(("pending", pending))
    config_value, _ = _config_tz(root)
    candidates.append(("service.json", config_value))
    chosen = None
    for source, value in candidates:
        if not value:
            continue
        zone, error = _zone(value)
        if zone is not None:
            chosen = {"name": str(value).lstrip(":"), "source": source, "tzinfo": zone}
            break
        errors.append(f"{source}: {error}")
    system_name, system_tz = system_zone()
    if chosen is None:
        chosen = {"name": system_name, "source": "system", "tzinfo": system_tz}
    value = {**chosen, "errors": errors, "system": system_name, "jobs_root": str(root)}
    if configured:
        with _LOCK:
            _CACHE.update(key=key, value=value)
    return value


def tz() -> dt.tzinfo:
    return resolve()["tzinfo"]


def now_local() -> dt.datetime:
    """The current instant as an aware datetime in the deployment zone."""
    return dt.datetime.now(tz())


def local(timestamp: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(float(timestamp), tz())


def now_iso() -> str:
    """``2026-09-26T14:00:00+08:00`` (seconds, with offset) in the deployment zone."""
    return now_local().isoformat(timespec="seconds")


def iso(timestamp: float | None) -> str | None:
    """Epoch seconds → ISO-8601 with offset in the deployment zone; None stays None."""
    if timestamp is None:
        return None
    return local(timestamp).isoformat(timespec="seconds")


def strftime(fmt: str, timestamp: float | None = None) -> str:
    """``fmt`` of now (or of ``timestamp``) in the deployment zone — job ids, file names."""
    return (now_local() if timestamp is None else local(timestamp)).strftime(fmt)


def describe(root=None) -> dict:
    """Facts for ``doctor`` / ``service status``: the resolved zone, where it came from, the current offset, the
    system zone and the ``TZ`` of this process (the shell), with ``consistent`` = same UTC offset now."""
    info = resolve(root)
    now = dt.datetime.now(dt.timezone.utc)
    offset = now.astimezone(info["tzinfo"]).isoformat(timespec="seconds")[-6:]
    system_offset = now.astimezone(system_zone()[1]).isoformat(timespec="seconds")[-6:]
    return {"name": info["name"], "source": info["source"], "offset": offset, "system": info["system"],
            "system_offset": system_offset, "process_tz": os.environ.get("TZ"), "errors": list(info["errors"]),
            "consistent": offset == system_offset, "now": now.astimezone(info["tzinfo"]).isoformat(timespec="seconds")}


class LogFormatter(logging.Formatter):
    """``logging.Formatter`` whose ``%(asctime)s`` is in the deployment zone (``%z`` → ``+0800``)."""

    def formatTime(self, record, datefmt=None):  # noqa: N802 — logging API
        stamp = local(record.created)
        if datefmt:
            return stamp.strftime(datefmt)
        return stamp.strftime("%Y-%m-%d %H:%M:%S") + f",{int(record.msecs):03d}"


__all__ = ["ENV_TZ", "LogFormatter", "configure", "describe", "iso", "jobs_root", "local", "now_iso", "now_local",
           "resolve", "strftime", "system_zone", "tz"]
