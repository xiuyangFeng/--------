from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
READ_ONLY_ROOTS = tuple(
    (ROOT / name).resolve()
    for name in ("data_new", "data_wss_min", "pipeline_wss_min", "training_wss_min")
)
ALLOWED_WRITE_ROOTS = tuple(
    (ROOT / name).resolve()
    for name in ("wss_pinn", "data_wss_pinn", "outputs/wss_pinn")
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def guard_write_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved = resolved.resolve()
    if any(is_relative_to(resolved, root) for root in READ_ONLY_ROOTS):
        raise ValueError(f"refusing write inside read-only upstream: {resolved}")
    if not any(is_relative_to(resolved, root) for root in ALLOWED_WRITE_ROOTS):
        raise ValueError(f"write path is outside WSS-PINN roots: {resolved}")
    return resolved


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, target)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return target


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        process = subprocess.run(
            ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
        )
        return process.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
        status = run("status", "--short")
        return {
            "commit": commit,
            "branch": branch,
            "dirty": bool(status),
            "status_short": status.splitlines(),
        }
    except Exception as exc:  # pragma: no cover - defensive metadata path
        return {"commit": "unknown", "branch": "unknown", "dirty": True, "error": str(exc)}

