"""C3: scrypt users, lock-out, username sessions, claiming session-scoped jobs, admin read-only view, legacy token window."""
from __future__ import annotations

import json
import os

import pytest

from wss_deploy.cli import main as cli_main
from wss_deploy.jobs import JobError
from wss_deploy.users import FAILURE_WINDOW_SECONDS, MAX_FAILURES, UserStore, hash_password, verify_password
from tests._c_helpers import Service, call, finished, manager


class Clock:
    def __init__(self, now=1_800_000_000.0): self.now = now
    def __call__(self): return self.now


def test_hash_verify_and_store_roundtrip(tmp_path):
    stored = hash_password("correct horse")
    assert stored.startswith("scrypt$") and verify_password(stored, "correct horse") and not verify_password(stored, "wrong horse")
    with pytest.raises(JobError):
        hash_password("short")
    clock = Clock()
    store = UserStore(tmp_path / "users.json", clock=clock)
    assert store.exists() is False
    row = store.add("alice", "alice-secret", admin=True, display_name="Alice")
    assert row == {"username": "alice", "role": "admin", "display_name": "Alice", "disabled": False, "created_at": row["created_at"]}
    assert oct(os.stat(tmp_path / "users.json").st_mode & 0o777) == "0o600" and store.exists() is True
    with pytest.raises(JobError):
        store.add("alice", "another-secret")
    with pytest.raises(JobError):
        store.add("bad name!", "another-secret")
    assert store.verify("alice", "alice-secret")["role"] == "admin"
    for _ in range(MAX_FAILURES):
        with pytest.raises(JobError) as info:
            store.verify("alice", "nope-nope")
        assert info.value.status == 401
    with pytest.raises(JobError) as info:
        store.verify("alice", "alice-secret")
    assert info.value.status == 429
    clock.now += FAILURE_WINDOW_SECONDS + 1
    assert store.verify("alice", "alice-secret")["username"] == "alice"
    store.change_password("alice", "alice-secret", "alice-newer-secret")
    assert store.verify("alice", "alice-newer-secret")
    store.disable("alice")
    with pytest.raises(JobError):
        store.verify("alice", "alice-newer-secret")
    assert store.exists() is False


def _users(tmp_path):
    store = UserStore(tmp_path / "users.json")
    store.add("alice", "alice-secret")
    store.add("root", "root-secret", admin=True)
    return store


def test_password_login_claim_and_admin_view(tmp_path):
    users = _users(tmp_path)
    service = Service(tmp_path, shared=True, token="legacy-token", users=users)
    api = service.api()
    try:
        status, payload, _ = call(api, "GET", "/api/session", {})
        assert status == 200 and payload["authenticated"] is False and payload["login"] == "password" and payload["legacy_token_allowed"] is False
        # the old token no longer opens a session once users exist
        status, payload, _, _ = service.login(api, {"token": "legacy-token"})
        assert status == 401
        # an old token session (simulated: opened before users.json existed) owns jobs; logging in with a
        # username offers to claim them even though the token window has closed since (v0.14 S1/S2)
        sid = "legacy-session-id"
        service.sessions.sessions[sid] = service.sessions._new_row("old-session-owner", role="legacy",
                                                                   token_fingerprint=service.sessions.token_fingerprint())
        finished(service.manager, owner="old-session-owner", case_id="OLD")
        api.request("POST", "/api/session", body=json.dumps({"username": "alice", "password": "alice-secret"}),
                    headers={"Content-Type": "application/json", "Cookie": f"wss_session={sid}"})
        response = api.getresponse(); payload = json.loads(response.read())
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        headers = {"Cookie": cookie, "X-CSRF-Token": payload["csrf_token"], "Content-Type": "application/json"}
        assert response.status == 200 and payload["username"] == "alice" and payload["role"] == "user"
        assert payload["claimable"] == [{"owner": "old-session-owner", "jobs": 1}]
        status, listing, _ = call(api, "GET", "/api/jobs", {"Cookie": cookie})
        assert listing["jobs"] == []
        status, payload, _ = call(api, "POST", "/api/jobs/claim", headers, {"owner": "not-mine"})
        assert status == 403
        status, payload, _ = call(api, "POST", "/api/jobs/claim", headers, {"owner": "old-session-owner"})
        assert status == 200 and payload["claimed"] == 1
        status, listing, _ = call(api, "GET", "/api/jobs", {"Cookie": cookie})
        assert [job["case_id"] for job in listing["jobs"]] == ["OLD"]
        status, payload, _ = call(api, "GET", "/api/session", {"Cookie": cookie})
        assert payload["claimable_owners"] == []
        # wrong passwords lock the account for a minute
        for _ in range(MAX_FAILURES):
            status, _, _, _ = service.login(api, {"username": "alice", "password": "wrong-secret"})
            assert status == 401
        status, _, _, _ = service.login(api, {"username": "alice", "password": "alice-secret"})
        assert status == 429
        # admin sees everything read-only, cannot mutate someone else's job
        status, payload, _, admin_headers = service.login(api, {"username": "root", "password": "root-secret"})
        assert status == 200 and payload["role"] == "admin"
        job_id = listing["jobs"][0]["id"]
        status, everything, _ = call(api, "GET", "/api/jobs?all=1", {"Cookie": admin_headers["Cookie"]})
        assert [job["id"] for job in everything["jobs"]] == [job_id]
        status, _, _ = call(api, "GET", "/api/jobs", {"Cookie": admin_headers["Cookie"]})
        assert _ .status == 200 and call(api, "GET", "/api/jobs", {"Cookie": admin_headers["Cookie"]})[1]["jobs"] == []
        status, _, _ = call(api, "GET", f"/api/jobs/{job_id}/report", {"Cookie": admin_headers["Cookie"]})
        assert status == 200
        status, _, _ = call(api, "GET", "/api/cases?all=1", {"Cookie": admin_headers["Cookie"]})
        assert status == 200 and _ is not None
        status, payload, _ = call(api, "POST", f"/api/jobs/{job_id}/review", admin_headers, {"version": 1, "decision": "approve", "reviewer": "root"})
        assert status == 404
        # password change and logout
        status, payload, _ = call(api, "POST", "/api/session/password", admin_headers, {"old_password": "root-secret", "new_password": "root-secret-2"})
        assert status == 200 and payload["changed"] is True
        status, payload, response = call(api, "POST", "/api/session/logout", admin_headers, {})
        assert status == 200 and payload["authenticated"] is False and "Max-Age=0" in response.getheader("Set-Cookie")
        status, _, _ = call(api, "GET", "/api/jobs", {"Cookie": admin_headers["Cookie"]})
        assert status == 401
    finally:
        api.close(); service.close()


def test_legacy_token_window_and_loopback_stay_login_free(tmp_path, monkeypatch):
    # no users.json yet: the token still works and the session reports the migration state
    service = Service(tmp_path / "token", shared=True, token="legacy-token", users=UserStore(tmp_path / "token" / "users.json"))
    api = service.api()
    try:
        status, payload, cookie, headers = service.login(api, {"token": "legacy-token"})
        assert status == 200 and payload["login"] == "token" and payload["role"] == "legacy" and payload["legacy_token_allowed"] is True
        status, _, _, _ = service.login(api, {"username": "alice", "password": "alice-secret"})
        assert status == 401
    finally:
        api.close(); service.close()
    users = _users(tmp_path / "flag")
    monkeypatch.setenv("WSS_DEPLOY_ALLOW_LEGACY_TOKEN", "1")
    service = Service(tmp_path / "flag", shared=True, token="legacy-token", users=users)
    api = service.api()
    try:
        status, payload, _, _ = service.login(api, {"token": "legacy-token"})
        assert status == 200 and payload["legacy_token_allowed"] is True and payload["login"] == "password"
    finally:
        api.close(); service.close()
    monkeypatch.delenv("WSS_DEPLOY_ALLOW_LEGACY_TOKEN")
    service = Service(tmp_path / "local", users=_users(tmp_path / "local"))
    api = service.api()
    try:
        status, payload, _ = call(api, "GET", "/api/session", {})
        assert status == 200 and payload["authenticated"] is True and payload["login"] == "none" and payload["role"] is None
    finally:
        api.close(); service.close()


def test_cli_user_and_claim_commands(tmp_path, monkeypatch, capsys):
    root = tmp_path / "jobs"
    monkeypatch.setenv("WSS_DEPLOY_PASSWORD", "cli-secret-1")
    monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": staticmethod(lambda: False)})())
    assert cli_main(["user", "add", "carol", "--admin", "--jobs-root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["role"] == "admin"
    with pytest.raises(SystemExit):
        cli_main(["user", "add", "carol", "--jobs-root", str(root)])
    assert cli_main(["user", "list", "--jobs-root", str(root)]) == 0 and "carol" in capsys.readouterr().out
    mgr = manager(root)
    finished(mgr, owner="sess-1", case_id="A")
    mgr.close()
    assert cli_main(["jobs", "claim", "--owner", "sess-1", "--user", "carol", "--jobs-root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["claimed"] == 1
    assert [job["case_id"] for job in manager(root).list("carol")] == ["A"]
