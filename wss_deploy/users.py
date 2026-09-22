"""Named users for the shared service (C3): scrypt password hashes, roles, lock-out on repeated failures.

``users.json`` lives next to the jobs (mode 0600).  Passwords never reach the log; the CLI reads them
from the terminal.  The store is only consulted in shared (non-loopback) mode; the local workbench
stays login-free.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

from .jobs import JobError, _date, atomic_json

USERNAME_PATTERN = r"[A-Za-z0-9_.-]{1,64}"
MIN_PASSWORD_LENGTH = 8
MAX_FAILURES = 5
FAILURE_WINDOW_SECONDS = 60
SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_LEN = 2 ** 14, 8, 1, 32


def valid_username(name) -> bool:
    return isinstance(name, str) and re.fullmatch(USERNAME_PATTERN, name) is not None


def hash_password(password: str, salt: bytes | None = None) -> str:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise JobError(f"口令至少 {MIN_PASSWORD_LENGTH} 个字符。")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=SCRYPT_LEN)
    return "scrypt$" + base64.b64encode(salt).decode("ascii") + "$" + base64.b64encode(digest).decode("ascii")


def verify_password(stored: str, password: str) -> bool:
    try:
        scheme, salt, digest = str(stored).split("$", 2)
        if scheme != "scrypt" or not isinstance(password, str):
            return False
        salt_bytes, expected = base64.b64decode(salt), base64.b64decode(digest)
    except (ValueError, TypeError):
        return False
    candidate = hashlib.scrypt(password.encode("utf-8"), salt=salt_bytes, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=len(expected))
    return secrets.compare_digest(candidate, expected)


class UserStore:
    def __init__(self, path: Path, *, clock=None):
        self.path = Path(path)
        self.clock = clock or time.time
        self.lock = threading.RLock()
        self._failures: dict[str, list[float]] = {}

    # -- persistence -----------------------------------------------------------------
    def load(self) -> dict:
        if not self.path.is_file():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise JobError("users.json 无法读取。", 500)
        return value if isinstance(value, dict) else {}

    def _save(self, users: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, users)
        os.chmod(self.path, 0o600)

    def exists(self) -> bool:
        """Password login is active as soon as one enabled user exists."""
        return any(not row.get("disabled") for row in self.load().values() if isinstance(row, dict))

    def names(self) -> list[str]:
        return sorted(self.load())

    # -- administration (CLI) ----------------------------------------------------------
    def add(self, username: str, password: str, *, admin: bool = False, display_name: str = "") -> dict:
        if not valid_username(username):
            raise JobError("用户名只能包含字母、数字、下划线、点和连字符（1–64 位）。")
        with self.lock:
            users = self.load()
            if username in users:
                raise JobError("用户已存在。", 409)
            users[username] = {"password": hash_password(password), "role": "admin" if admin else "user",
                               "created_at": _date(self.clock()), "display_name": display_name or username, "disabled": False}
            self._save(users)
            return self.public(username, users[username])

    def set_password(self, username: str, password: str) -> None:
        with self.lock:
            users = self.load()
            if username not in users:
                raise JobError("用户不存在。", 404)
            users[username]["password"] = hash_password(password)
            users[username]["password_changed_at"] = _date(self.clock())
            self._save(users)

    def disable(self, username: str) -> None:
        with self.lock:
            users = self.load()
            if username not in users:
                raise JobError("用户不存在。", 404)
            users[username]["disabled"] = True
            self._save(users)

    @staticmethod
    def public(username: str, row: dict) -> dict:
        return {"username": username, "role": row.get("role") or "user", "display_name": row.get("display_name") or username,
                "disabled": bool(row.get("disabled")), "created_at": row.get("created_at")}

    # -- authentication ----------------------------------------------------------------
    def _recent_failures(self, username: str) -> list[float]:
        now = self.clock()
        recent = [t for t in self._failures.get(username, []) if now - t < FAILURE_WINDOW_SECONDS]
        self._failures[username] = recent
        return recent

    def verify(self, username, password) -> dict:
        """Return the public user row or raise 401 (bad credentials / disabled) or 429 (locked out)."""
        if not valid_username(username) or not isinstance(password, str):
            raise JobError("用户名或口令不正确。", 401)
        with self.lock:
            if len(self._recent_failures(username)) >= MAX_FAILURES:
                raise JobError("口令错误次数过多，请一分钟后再试。", 429)
            row = self.load().get(username)
            ok = isinstance(row, dict) and not row.get("disabled") and verify_password(row.get("password", ""), password)
            if not ok:
                self._failures.setdefault(username, []).append(self.clock())
                raise JobError("用户名或口令不正确。", 401)
            self._failures.pop(username, None)
            return self.public(username, row)

    def change_password(self, username: str, old_password, new_password) -> None:
        self.verify(username, old_password)
        self.set_password(username, new_password)


__all__ = ["MAX_FAILURES", "FAILURE_WINDOW_SECONDS", "MIN_PASSWORD_LENGTH", "USERNAME_PATTERN", "UserStore",
           "hash_password", "valid_username", "verify_password"]
