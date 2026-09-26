"""Named users for the shared service (C3): scrypt password hashes, roles, lock-out on repeated failures.

``users.json`` lives next to the jobs (mode 0600).  Passwords never reach the log; the CLI reads them
from the terminal.  The store is only consulted in shared (non-loopback) mode; the local workbench
stays login-free.

v0.14: every user row carries an integer ``credential_generation`` (absent = 0 for older files).  It is
bumped by a password change, ``disable`` and a role change; the service stores the generation in each
session and rejects a session whose generation no longer matches, so sessions die with their credentials.
Unknown user names cost one scrypt like real ones (no timing oracle) and the failure table is bounded.

v0.15 (account hygiene): new passwords (``add`` / ``set_password`` / API change) are refused when they are all
digits, equal the user name, repeat one character or are on a short list of common passwords
(:func:`password_problem`; no expiry, no composition rules).  ``disable`` / ``set_role`` take ``keep_admin=True``
(the CLI passes it unless ``--force``) to refuse leaving the store without an enabled administrator.  A lock-out
answer carries ``retry_after`` (seconds).
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path

from .jobs import JobError, _date, atomic_json

USERNAME_PATTERN = r"[A-Za-z0-9_.-]{1,64}"
MIN_PASSWORD_LENGTH = 8
MAX_FAILURES = 5
FAILURE_WINDOW_SECONDS = 60
MAX_TRACKED_FAILURES = 1000   # user names with recent failures kept in memory (LRU)
ROLES = ("user", "admin")
SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_LEN = 2 ** 14, 8, 1, 32
# v0.15: refused as new passwords (compared case-insensitively); ≤ 30 entries on purpose — a floor, not a policy.
COMMON_PASSWORDS = frozenset({
    "password", "password1", "password123", "passw0rd", "p@ssw0rd", "qwertyui", "qwertyuiop", "qwerty123", "qwer1234",
    "1qaz2wsx", "1q2w3e4r", "zaq12wsx", "asdfghjk", "abc12345", "abcd1234", "a1234567", "aa123456", "admin123", "admin1234",
    "admin@123", "admin888", "administrator", "root1234", "test1234", "iloveyou", "letmein1", "welcome1", "changeme",
    "woaini1314", "5201314a"})   # all-digit passwords (12345678, 88888888 …) are refused by rule, not listed
_DUMMY_HASH: list[str] = []   # computed once: verified against for unknown names so timing does not reveal existence


def _dummy_hash() -> str:
    if not _DUMMY_HASH:
        _DUMMY_HASH.append(hash_password(secrets.token_urlsafe(16)))
    return _DUMMY_HASH[0]


def generation_of(row) -> int:
    """``credential_generation`` of a user row; rows written before v0.14 count as 0."""
    value = row.get("credential_generation") if isinstance(row, dict) else None
    return value if type(value) is int and value >= 0 else 0


def valid_username(name) -> bool:
    return isinstance(name, str) and re.fullmatch(USERNAME_PATTERN, name) is not None


def password_problem(password, username=None) -> str | None:
    """Why ``password`` may not be set for ``username`` (v0.15), or None.  Length first (as before), then a small floor:
    all digits, equal to the user name, one repeated character, or a common password."""
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        return f"口令至少 {MIN_PASSWORD_LENGTH} 个字符。"
    folded = password.casefold()
    if password.isdigit():
        return "口令不能全是数字。"
    if isinstance(username, str) and username and folded == username.casefold():
        return "口令不能与用户名相同。"
    if len(set(folded)) == 1:
        return "口令不能由同一个字符重复组成。"
    if folded in COMMON_PASSWORDS:
        return "口令过于常见，请换一个。"
    return None


def check_new_password(password, username=None) -> None:
    problem = password_problem(password, username)
    if problem:
        raise JobError(problem)


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
        self._failures: OrderedDict[str, list[float]] = OrderedDict()
        self._snapshot: tuple[tuple | None, dict] = (None, {})

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

    def cached(self) -> dict:
        """``load()`` memoised on the file's (inode, size, mtime); the session check reads it on every request."""
        try:
            st = self.path.stat()
            key = (st.st_ino, st.st_size, st.st_mtime_ns)
        except OSError:
            return {}
        with self.lock:
            if self._snapshot[0] == key:
                return self._snapshot[1]
            value = self.load()
            self._snapshot = (key, value)
            return value

    def credential(self, username) -> dict | None:
        """``{"generation", "role", "disabled"}`` of one user from the cached file, or None when unknown."""
        row = self.cached().get(username) if isinstance(username, str) else None
        if not isinstance(row, dict):
            return None
        return {"generation": generation_of(row), "role": row.get("role") or "user", "disabled": bool(row.get("disabled"))}

    def exists(self) -> bool:
        """Password login is active as soon as one enabled user exists."""
        return any(not row.get("disabled") for row in self.load().values() if isinstance(row, dict))

    def names(self) -> list[str]:
        return sorted(self.load())

    # -- administration (CLI) ----------------------------------------------------------
    def add(self, username: str, password: str, *, admin: bool = False, display_name: str = "") -> dict:
        if not valid_username(username):
            raise JobError("用户名只能包含字母、数字、下划线、点和连字符（1–64 位）。")
        check_new_password(password, username)
        with self.lock:
            users = self.load()
            if username in users:
                raise JobError("用户已存在。", 409)
            users[username] = {"password": hash_password(password), "role": "admin" if admin else "user",
                               "created_at": _date(self.clock()), "display_name": display_name or username, "disabled": False,
                               "credential_generation": 0}
            self._save(users)
            return self.public(username, users[username])

    def set_password(self, username: str, password: str) -> None:
        check_new_password(password, username)
        with self.lock:
            users = self.load()
            if username not in users:
                raise JobError("用户不存在。", 404)
            users[username]["password"] = hash_password(password)
            users[username]["password_changed_at"] = _date(self.clock())
            self._bump(users[username])
            self._save(users)

    @staticmethod
    def _enabled_admins(users: dict) -> list[str]:
        return sorted(name for name, row in users.items()
                      if isinstance(row, dict) and not row.get("disabled") and (row.get("role") or "user") == "admin")

    def enabled_admins(self) -> list[str]:
        return self._enabled_admins(self.load())

    def _guard_last_admin(self, users: dict, username: str, *, disable: bool = False, role: str | None = None) -> None:
        """v0.15: refuse disabling / demoting the last enabled administrator while other enabled accounts remain
        (they would be left without anyone who can see all tasks or edit the template).  A store whose only account
        is that administrator is exempt: nobody is stranded and ``user role <名> --admin`` restores it."""
        admins = self._enabled_admins(users)
        others = [name for name, row in users.items() if name != username and isinstance(row, dict) and not row.get("disabled")]
        if username in admins and len(admins) == 1 and others and (disable or (role is not None and role != "admin")):
            raise JobError(f"{username} 是唯一启用的管理员；请先用 `user add <名> --admin` 或 `user role <名> --admin` "
                           "设置另一位管理员（确需如此可加 --force）。", 409)

    def disable(self, username: str, *, keep_admin: bool = False) -> None:
        with self.lock:
            users = self.load()
            if username not in users:
                raise JobError("用户不存在。", 404)
            if keep_admin:
                self._guard_last_admin(users, username, disable=True)
            users[username]["disabled"] = True
            self._bump(users[username])
            self._save(users)

    def set_role(self, username: str, role: str, *, keep_admin: bool = False) -> None:
        """Make a user ``admin`` or ``user``; open sessions of that user end (their role is fixed at login)."""
        if role not in ROLES:
            raise JobError("角色只能是 user 或 admin。")
        with self.lock:
            users = self.load()
            if username not in users:
                raise JobError("用户不存在。", 404)
            if (users[username].get("role") or "user") == role:
                return
            if keep_admin:
                self._guard_last_admin(users, username, role=role)
            users[username]["role"] = role
            self._bump(users[username])
            self._save(users)

    @staticmethod
    def _bump(row: dict) -> None:
        row["credential_generation"] = generation_of(row) + 1

    @staticmethod
    def public(username: str, row: dict) -> dict:
        return {"username": username, "role": row.get("role") or "user", "display_name": row.get("display_name") or username,
                "disabled": bool(row.get("disabled")), "created_at": row.get("created_at")}

    # -- authentication ----------------------------------------------------------------
    def _recent_failures(self, username: str) -> list[float]:
        now = self.clock()
        recent = [t for t in self._failures.get(username, []) if now - t < FAILURE_WINDOW_SECONDS]
        if recent:
            self._failures[username] = recent
            self._failures.move_to_end(username)
        else:
            self._failures.pop(username, None)
        return recent

    def lock_seconds_left(self, username: str, recent: list[float] | None = None) -> int:
        """Whole seconds until ``username`` may try again (0 when it is not locked)."""
        recent = self._recent_failures(username) if recent is None else recent
        if len(recent) < MAX_FAILURES:
            return 0
        # the lock ends when enough of the recent failures have left the window
        release = sorted(recent)[len(recent) - MAX_FAILURES] + FAILURE_WINDOW_SECONDS
        return max(1, math.ceil(release - self.clock()))

    def _record_failure(self, username: str) -> None:
        self._failures.setdefault(username, []).append(self.clock())
        self._failures.move_to_end(username)
        while len(self._failures) > MAX_TRACKED_FAILURES:
            self._failures.popitem(last=False)

    def authenticate(self, username, password) -> tuple[dict, int]:
        """``verify`` plus the user's ``credential_generation`` read under the same lock (for the session row)."""
        if not valid_username(username) or not isinstance(password, str):
            raise JobError("用户名或口令不正确。", 401)
        with self.lock:
            recent = self._recent_failures(username)
            if len(recent) >= MAX_FAILURES:
                wait = self.lock_seconds_left(username, recent)
                raise JobError(f"口令错误次数过多，该用户名暂时锁定，请 {wait} 秒后再试。", 429, {"retry_after": wait})
            row = self.load().get(username)
            known = isinstance(row, dict)
            # Unknown and disabled names still pay one scrypt, so response time does not tell them apart.
            matched = verify_password(row.get("password", "") if known else _dummy_hash(), password)
            if not (known and matched and not row.get("disabled")):
                self._record_failure(username)
                raise JobError("用户名或口令不正确。", 401)
            self._failures.pop(username, None)
            return self.public(username, row), generation_of(row)

    def verify(self, username, password) -> dict:
        """Return the public user row or raise 401 (bad credentials / disabled) or 429 (locked out)."""
        return self.authenticate(username, password)[0]

    def change_password(self, username: str, old_password, new_password) -> None:
        self.verify(username, old_password)
        self.set_password(username, new_password)


__all__ = ["COMMON_PASSWORDS", "MAX_FAILURES", "FAILURE_WINDOW_SECONDS", "MAX_TRACKED_FAILURES", "MIN_PASSWORD_LENGTH", "ROLES",
           "USERNAME_PATTERN", "UserStore", "check_new_password", "generation_of", "hash_password", "password_problem",
           "valid_username", "verify_password"]
