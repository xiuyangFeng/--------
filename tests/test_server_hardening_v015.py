"""v0.15 pre-launch hardening: reverse-proxy / TLS switches (opt-in), file permissions under any umask,
password floor and last-admin guard, two-bucket login throttling with ``retry_after``, audit coverage,
health without server paths for non-administrators, and small request-parsing fixes."""
from __future__ import annotations

import email.message
import json
import logging
import os
import stat
import time
import types
from pathlib import Path

import pytest

from wss_deploy import server as S
from wss_deploy.cli import main as cli_main
from wss_deploy.jobs import JobError, atomic_json
from wss_deploy.server import LoginThrottle
from wss_deploy.users import COMMON_PASSWORDS, UserStore, password_problem
from tests._c_helpers import Service, call, finished, manager, multipart


def _users(root: Path) -> UserStore:
    store = UserStore(root / "users.json")
    store.add("alice", "alice-secret")
    store.add("bob", "bob-secret-1")
    store.add("root", "root-secret", admin=True)
    return store


def _login(api, body, headers=None):
    api.request("POST", "/api/session", body=json.dumps(body), headers={"Content-Type": "application/json", **(headers or {})})
    response = api.getresponse(); payload = json.loads(response.read())
    raw_cookie = response.getheader("Set-Cookie") or ""
    cookie = raw_cookie.split(";", 1)[0]
    session = {"Cookie": cookie, "X-CSRF-Token": payload.get("csrf_token") or "", "Content-Type": "application/json"}
    return response, payload, raw_cookie, session


def _audit(caplog, prefix="http "):
    return [json.loads(r.getMessage()[len(prefix):]) for r in caplog.records
            if r.name == "wss_deploy.audit" and r.getMessage().startswith(prefix)]


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# ------------------------------------------------------------------------------------------ reverse proxy / TLS
def test_forwarded_headers_are_ignored_by_default(tmp_path, monkeypatch, caplog):
    for name in (S.TRUST_PROXY_ENV, S.COOKIE_SECURE_ENV):
        monkeypatch.delenv(name, raising=False)
    service = Service(tmp_path, shared=True, users=_users(tmp_path))
    assert service.server.trust_proxy is False and service.server.cookie_secure == "auto"
    service.server.login_throttle = LoginThrottle(2)
    api = service.api()
    host = f"127.0.0.1:{service.port}"
    forged = {"X-Forwarded-For": "6.6.6.6", "X-Forwarded-Proto": "https", "X-Forwarded-Host": "evil.example"}
    try:
        with caplog.at_level(logging.INFO, logger="wss_deploy.audit"):
            response, payload, raw_cookie, session = _login(api, {"username": "alice", "password": "alice-secret"}, forged)
        assert response.status == 200
        assert "Secure" not in raw_cookie and response.getheader("Strict-Transport-Security") is None
        assert [line["ip"] for line in _audit(caplog) if line["action"] == "login"] == ["127.0.0.1"]
        # v0.14 Origin rule unchanged: http and https origins of the Host are both accepted, a forwarded host is not
        for origin, expected in ((f"http://{host}", 200), (f"https://{host}", 200), ("https://evil.example", 403)):
            status, _, _ = call(api, "PUT", "/api/preferences", {**session, **forged, "Origin": origin}, {"a": 1})
            assert status == expected, origin
        # a forged X-Forwarded-For does not buy a fresh throttle bucket
        for index in range(2):
            response, _, _, _ = _login(api, {"username": "alice", "password": "wrong-secret"}, {"X-Forwarded-For": f"9.9.9.{index}"})
            assert response.status == 401
        response, payload, _, _ = _login(api, {"username": "bob", "password": "bob-secret-1"}, {"X-Forwarded-For": "8.8.8.8"})
        assert response.status == 429 and payload["retry_after"] >= 1 and response.getheader("Retry-After") == str(payload["retry_after"])
    finally:
        api.close(); service.close()


def test_trusted_proxy_forwards_client_address_scheme_and_secure_cookie(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(S.TRUST_PROXY_ENV, "1")
    monkeypatch.delenv(S.COOKIE_SECURE_ENV, raising=False)
    service = Service(tmp_path, shared=True, users=_users(tmp_path))
    assert service.server.trust_proxy is True
    service.server.login_throttle = LoginThrottle(1)
    api = service.api()
    host = f"127.0.0.1:{service.port}"
    https = {"X-Forwarded-For": "6.6.6.6, 10.1.2.3", "X-Forwarded-Proto": "https", "X-Forwarded-Host": "wss.lab:8443"}
    access = logging.getLogger("wss_deploy.access")
    access.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="wss_deploy.audit"), caplog.at_level(logging.INFO, logger="wss_deploy.access"):
            response, payload, raw_cookie, session = _login(api, {"username": "alice", "password": "alice-secret"}, https)
            assert response.status == 200
            assert raw_cookie.endswith("; Secure") and response.getheader("Strict-Transport-Security") == "max-age=31536000"
            # the right-most non-loopback hop is the client; the client-written 6.6.6.6 is not believed
            assert [line["ip"] for line in _audit(caplog) if line["action"] == "login"] == ["10.1.2.3"]
            # Origin: only the scheme the proxy reports, for the Host or the forwarded host
            for origin, expected in (("https://wss.lab:8443", 200), (f"https://{host}", 200), ("http://wss.lab:8443", 403),
                                     (f"http://{host}", 403), ("https://other.lab", 403)):
                status, _, response = call(api, "PUT", "/api/preferences", {**session, **https, "Origin": origin}, {"a": 1})
                assert status == expected, origin
                assert response.getheader("Strict-Transport-Security") == "max-age=31536000"
            # without X-Forwarded-Proto (e.g. a local CLI client) both schemes stay accepted, no HSTS, no Secure
            status, _, response = call(api, "PUT", "/api/preferences", {**session, "Origin": f"http://{host}"}, {"a": 1})
            assert status == 200 and response.getheader("Strict-Transport-Security") is None
            # plain http through the proxy: no HSTS and (auto) no Secure
            response, _, raw_cookie, _ = _login(api, {"username": "bob", "password": "bob-secret-1"},
                                                {"X-Forwarded-For": "10.9.9.9", "X-Forwarded-Proto": "http"})
            assert response.status == 200 and "Secure" not in raw_cookie and response.getheader("Strict-Transport-Security") is None
            # per-address throttling uses the forwarded address: one bucket per client, not one for the proxy
            assert _login(api, {"username": "alice", "password": "wrong-1"}, {"X-Forwarded-For": "10.0.0.1"})[0].status == 401
            assert _login(api, {"username": "alice", "password": "wrong-2"}, {"X-Forwarded-For": "10.0.0.1"})[0].status == 429
            assert _login(api, {"username": "alice", "password": "wrong-3"}, {"X-Forwarded-For": "10.0.0.2"})[0].status == 401
            deadline = time.time() + 5
            while time.time() < deadline and not any("10.0.0.2" in r.getMessage() for r in caplog.records if r.name == "wss_deploy.access"):
                time.sleep(0.05)
        lines = [r.getMessage() for r in caplog.records if r.name == "wss_deploy.access"]
        assert any(line.startswith("10.1.2.3 POST /api/session 200") for line in lines)
        assert any(line.startswith("10.0.0.2 POST /api/session 401") for line in lines)
        assert not any("6.6.6.6" in line for line in lines)
    finally:
        access.removeHandler(caplog.handler)
        api.close(); service.close()


def test_cookie_secure_switch_and_hsts_only_on_https(tmp_path, monkeypatch):
    monkeypatch.delenv(S.TRUST_PROXY_ENV, raising=False)
    monkeypatch.setenv(S.COOKIE_SECURE_ENV, "1")
    service = Service(tmp_path / "always", shared=True, users=_users(tmp_path / "always"))
    api = service.api()
    try:
        response, _, raw_cookie, session = _login(api, {"username": "alice", "password": "alice-secret"})
        assert response.status == 200 and raw_cookie.endswith("; Secure") and response.getheader("Strict-Transport-Security") is None
        status, _, response = call(api, "POST", "/api/session/logout", session, {})
        assert status == 200 and "Max-Age=0; Secure" in response.getheader("Set-Cookie")
    finally:
        api.close(); service.close()
    monkeypatch.setenv(S.TRUST_PROXY_ENV, "1")
    monkeypatch.setenv(S.COOKIE_SECURE_ENV, "0")
    service = Service(tmp_path / "never", shared=True, users=_users(tmp_path / "never"))
    api = service.api()
    try:
        response, _, raw_cookie, _ = _login(api, {"username": "alice", "password": "alice-secret"}, {"X-Forwarded-Proto": "https"})
        assert response.status == 200 and "Secure" not in raw_cookie
        assert response.getheader("Strict-Transport-Security") == "max-age=31536000"      # HSTS follows https, not the cookie switch
    finally:
        api.close(); service.close()
    monkeypatch.setenv(S.COOKIE_SECURE_ENV, "bogus")
    assert S.cookie_secure_mode() == "auto"


def test_proxy_trust_needs_a_loopback_peer_and_implies_shared_mode(monkeypatch):
    handler = S.Handler.__new__(S.Handler)
    handler.server = types.SimpleNamespace(trust_proxy=True, cookie_secure="auto")
    handler.headers = email.message.Message()
    handler.headers["X-Forwarded-For"] = "10.1.2.3"
    handler.headers["X-Forwarded-Proto"] = "https"
    handler.client_address = ("192.168.5.7", 50000)                 # not the local proxy
    assert handler._client_ip() == "192.168.5.7" and handler._https() is False and handler._cookie_secure() is False
    handler.client_address = ("::ffff:127.0.0.1", 50000)
    assert handler._client_ip() == "10.1.2.3" and handler._https() is True and handler._cookie_secure() is True
    handler.server.trust_proxy = False
    assert handler._client_ip() == "::ffff:127.0.0.1" and handler._https() is False
    assert S.forwarded_client("127.0.0.1", ["1.1.1.1, 10.0.0.5", "127.0.0.1"]) == "10.0.0.5"
    assert S.forwarded_client("127.0.0.1", ["10.0.0.5, not-an-ip"]) == "127.0.0.1"
    assert S.forwarded_proto(["http, https"]) == "https" and S.forwarded_proto(["gopher"]) is None
    assert S.valid_host_value("wss.lab:8443") and not S.valid_host_value("a b") and not S.valid_host_value("u@h")
    monkeypatch.delenv(S.TRUST_PROXY_ENV, raising=False)
    assert S.service_is_shared("127.0.0.1") is False and S.service_is_shared("0.0.0.0") is True
    monkeypatch.setenv(S.TRUST_PROXY_ENV, "1")
    assert S.service_is_shared("127.0.0.1") is True


# ------------------------------------------------------------------------------------------ files and umask
def test_secret_and_log_files_are_0600_under_any_umask(tmp_path, monkeypatch):
    from wss_deploy.service import write_token
    previous = os.umask(0)
    try:
        atomic_json(tmp_path / "doc.json", {"a": 1})
        users = _users(tmp_path)
        sessions = S.SessionStore(tmp_path / ".sessions.json", shared=True, users=users)
        sessions.create(username="alice", password="alice-secret")
        write_token(tmp_path, "t0ken-value")
        log = tmp_path / "server.log"
        handler = S.PrivateRotatingFileHandler(log, maxBytes=200, backupCount=2, encoding="utf-8")
        for index in range(20):
            handler.emit(logging.LogRecord("wss_deploy.audit", logging.INFO, __file__, 1, "line %s " + "x" * 40, (index,), None))
        handler.close()
        old = tmp_path / "access.log"
        old.write_text("old\n"); os.chmod(old, 0o664)
        S.PrivateRotatingFileHandler(old, maxBytes=0, encoding="utf-8").close()
        mgr = manager(tmp_path / "jobs")
        job = finished(mgr, owner="alice", case_id="CASE")
        mgr.delete(job["id"], "alice", {"version": job["version"]})
        mgr.purge(job["id"], "alice")
        mgr.close()
        files = [tmp_path / "doc.json", tmp_path / "users.json", tmp_path / ".sessions.json", tmp_path / ".service_token",
                 log, tmp_path / "server.log.1", old, tmp_path / "jobs" / "deleted_jobs.jsonl"]
        assert {path.name: oct(_mode(path)) for path in files} == {path.name: "0o600" for path in files}
    finally:
        os.umask(previous)


def test_umask_switch(monkeypatch):
    monkeypatch.delenv(S.UMASK_ENV, raising=False)
    assert S.apply_umask_from_env() is None
    previous = os.umask(0o022); os.umask(previous)
    try:
        for bad in ("abc", "1777", "-1"):
            monkeypatch.setenv(S.UMASK_ENV, bad)
            assert S.apply_umask_from_env() is None
        monkeypatch.setenv(S.UMASK_ENV, "077")
        assert S.apply_umask_from_env() == previous
        current = os.umask(0o077)
        assert current == 0o077
    finally:
        os.umask(previous)


# ------------------------------------------------------------------------------------------ passwords and admins
def test_weak_passwords_are_refused_everywhere(tmp_path, monkeypatch):
    assert len(COMMON_PASSWORDS) <= 30
    store = UserStore(tmp_path / "users.json")
    for password, user in (("12345678", "alice"), ("Alice-Long", "alice-long"), ("zzzzzzzz", "alice"), ("Admin123", "alice"),
                           ("PassWord", "alice"), ("short", "alice")):
        assert password_problem(password, user), password
        with pytest.raises(JobError):
            store.add(user, password)
    assert password_problem("correct horse", "alice") is None and password_problem("alice-secret", "alice") is None
    store.add("alice", "alice-secret")
    with pytest.raises(JobError) as info:
        store.set_password("alice", "88888888")
    assert "数字" in str(info.value)
    # the API change goes through the same check and keeps the old password
    service = Service(tmp_path / "svc", shared=True, users=_users(tmp_path / "svc"))
    api = service.api()
    try:
        _, _, _, session = _login(api, {"username": "alice", "password": "alice-secret"})
        status, body, _ = call(api, "POST", "/api/session/password", session, {"old_password": "alice-secret", "new_password": "alice"})
        assert status == 400 and "口令" in body["error"]["message"]
        status, body, _ = call(api, "POST", "/api/session/password", session, {"old_password": "alice-secret", "new_password": "password1"})
        assert status == 400 and "常见" in body["error"]["message"]
        assert _login(api, {"username": "alice", "password": "alice-secret"})[0].status == 200
    finally:
        api.close(); service.close()
    # CLI: WSS_DEPLOY_PASSWORD is checked too
    monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": staticmethod(lambda: False)})())
    monkeypatch.setenv("WSS_DEPLOY_PASSWORD", "12341234")
    with pytest.raises(SystemExit) as info:
        cli_main(["user", "add", "erin", "--jobs-root", str(tmp_path / "cli")])
    assert "数字" in str(info.value)


def test_last_enabled_admin_cannot_be_disabled_or_demoted(tmp_path, monkeypatch, capsys):
    store = _users(tmp_path)
    with pytest.raises(JobError) as info:
        store.disable("root", keep_admin=True)
    assert info.value.status == 409 and "唯一启用的管理员" in str(info.value)
    with pytest.raises(JobError):
        store.set_role("root", "user", keep_admin=True)
    assert store.credential("root") == {"generation": 0, "role": "admin", "disabled": False}
    store.set_role("alice", "admin", keep_admin=True)
    store.set_role("root", "user", keep_admin=True)                 # another admin exists now
    root = tmp_path / "cli"
    monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": staticmethod(lambda: False)})())
    monkeypatch.setenv("WSS_DEPLOY_PASSWORD", "cli-secret-1")
    assert cli_main(["user", "add", "carol", "--admin", "--jobs-root", str(root)]) == 0
    assert cli_main(["user", "add", "dave", "--jobs-root", str(root)]) == 0
    for argv in (["user", "disable", "carol"], ["user", "role", "carol", "--user"]):
        with pytest.raises(SystemExit) as info:
            cli_main([*argv, "--jobs-root", str(root)])
        assert "--force" in str(info.value)
    assert UserStore(root / "users.json").enabled_admins() == ["carol"]
    assert cli_main(["user", "disable", "carol", "--force", "--jobs-root", str(root)]) == 0
    assert UserStore(root / "users.json").enabled_admins() == []
    # a store whose only account is the admin stays manageable without --force (nobody is stranded)
    solo = UserStore(tmp_path / "solo.json"); solo.add("sole", "sole-secret", admin=True)
    solo.set_role("sole", "user", keep_admin=True)


# ------------------------------------------------------------------------------------------ login budgets
def test_login_budgets_per_address_and_per_user_with_retry_after(tmp_path, caplog):
    assert S.DEFAULT_LOGIN_PER_MINUTE == 30 and S.DEFAULT_LOGIN_PER_USER_MINUTE == 10
    service = Service(tmp_path, shared=True, users=_users(tmp_path))
    assert service.server.login_throttle.capacity == 30 and service.server.user_login_throttle.capacity == 10
    service.server.user_login_throttle = LoginThrottle(3)
    api = service.api()
    try:
        with caplog.at_level(logging.INFO, logger="wss_deploy.audit"):
            for _ in range(3):
                assert _login(api, {"username": "alice", "password": "wrong-secret"})[0].status == 401
            for _ in range(2):
                response, payload, _, _ = _login(api, {"username": "alice", "password": "alice-secret"})
                assert response.status == 429 and "该用户名" in payload["error"]["message"]
                assert 1 <= payload["retry_after"] <= 60 and response.getheader("Retry-After") == str(payload["retry_after"])
            # a colleague behind the same address is not affected, and the refusals did not use the address budget
            assert _login(api, {"username": "bob", "password": "bob-secret-1"})[0].status == 200
            assert service.server.login_throttle.buckets["127.0.0.1"][0] >= 30 - 3 - 1e-6
            throttled = [line for line in _audit(caplog) if line["action"] == "login_throttled"]
            assert throttled == [{"action": "login_throttled", "scope": "username", "method": "password", "username": "alice",
                                  "ip": "127.0.0.1", "retry_after": throttled[0]["retry_after"]}]        # once per key and minute
        # users.py lock-out (5 wrong passwords in 60 s) answers with the seconds left
        service.server.user_login_throttle = LoginThrottle(100)
        for _ in range(5):
            assert _login(api, {"username": "bob", "password": "wrong-secret"})[0].status == 401
        response, payload, _, _ = _login(api, {"username": "bob", "password": "bob-secret-1"})
        assert response.status == 429 and "锁定" in payload["error"]["message"] and "秒" in payload["error"]["message"]
        assert 1 <= payload["retry_after"] <= 60 and response.getheader("Retry-After") == str(payload["retry_after"])
        # the address budget
        service.server.login_throttle = LoginThrottle(1)
        assert _login(api, {"username": "root", "password": "wrong-secret"})[0].status == 401
        response, payload, _, _ = _login(api, {"username": "root", "password": "root-secret"})
        assert response.status == 429 and "网络地址" in payload["error"]["message"] and payload["retry_after"] >= 1
    finally:
        api.close(); service.close()
    clock = types.SimpleNamespace(now=0.0)
    bucket = LoginThrottle(2, clock=lambda: clock.now)
    assert bucket.retry_after("a") == 0 and bucket.take("a") and bucket.take("a") and not bucket.take("a")
    assert bucket.retry_after("a") == 30
    clock.now += 29.5
    assert bucket.retry_after("a") == 1
    assert bucket.first_refusal("a") and not bucket.first_refusal("a")
    clock.now += 61
    assert bucket.first_refusal("a")


# ------------------------------------------------------------------------------------------ audit coverage
def test_audit_covers_logout_template_review_metadata_without_patient_ids(tmp_path, caplog):
    service = Service(tmp_path, shared=True, users=_users(tmp_path))
    api = service.api()
    try:
        with caplog.at_level(logging.INFO, logger="wss_deploy.audit"):
            _, _, _, alice = _login(api, {"username": "alice", "password": "alice-secret"})
            _, _, _, admin = _login(api, {"username": "root", "password": "root-secret"})
            job = finished(service.manager, owner="alice", case_id="CASE-A")
            status, body, _ = call(api, "POST", f"/api/jobs/{job['id']}/metadata", alice, {"version": job["version"], "patient_id": "PATIENT-XYZ"})
            assert status == 200, body
            status, body, _ = call(api, "POST", f"/api/jobs/{job['id']}/review", alice,
                                   {"version": body["job"]["version"], "decision": "approve", "reviewer": "alice"})
            assert status == 200, body
            status, _, _ = call(api, "PUT", "/api/report-template", admin, {"institution": "示例医院"})
            assert status == 200
            assert call(api, "POST", "/api/session/logout", alice, {})[0] == 200
        http = _audit(caplog)
        assert ("template_updated", "root") in {(line["action"], line["actor"]) for line in http}
        assert [line["keys"] for line in http if line["action"] == "template_updated"] == [["institution"]]
        assert ("logout", "alice") in {(line["action"], line["actor"]) for line in http}
        jobs = _audit(caplog, "job ")
        assert {(line["action"], line.get("actor"), line["job"]) for line in jobs} >= {
            ("metadata_updated", "alice", job["id"]), ("review_approved", "alice", job["id"])}
        assert "PATIENT-XYZ" not in "\n".join(r.getMessage() for r in caplog.records if r.name == "wss_deploy.audit")
    finally:
        api.close(); service.close()


def test_cli_user_management_and_claims_are_audited_in_the_service_log(tmp_path, monkeypatch):
    root = tmp_path / "jobs"
    monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": staticmethod(lambda: False)})())
    monkeypatch.setenv("WSS_DEPLOY_PASSWORD", "cli-secret-1")
    previous = os.umask(0)
    try:
        assert cli_main(["user", "add", "carol", "--admin", "--jobs-root", str(root)]) == 0
        assert cli_main(["user", "add", "dave", "--jobs-root", str(root)]) == 0
        monkeypatch.setenv("WSS_DEPLOY_PASSWORD", "cli-secret-2")
        assert cli_main(["user", "passwd", "dave", "--jobs-root", str(root)]) == 0
        assert cli_main(["user", "role", "dave", "--admin", "--jobs-root", str(root)]) == 0
        assert cli_main(["user", "disable", "dave", "--jobs-root", str(root)]) == 0
        mgr = manager(root)
        job = finished(mgr, owner="old-session-owner", case_id="OLD")
        mgr.close()
        assert cli_main(["jobs", "claim", "--owner", "old-session-owner", "--user", "carol", "--jobs-root", str(root)]) == 0
    finally:
        os.umask(previous)
    log = root / "server.log"
    assert _mode(log) == 0o600
    text = log.read_text(encoding="utf-8")
    assert "cli-secret" not in text and "old-session-owner" not in text
    cli_lines = [json.loads(line.split(" wss_deploy.audit cli ", 1)[1]) for line in text.splitlines() if " wss_deploy.audit cli " in line]
    assert [(line["action"], line.get("target") or line.get("user")) for line in cli_lines] == [
        ("user_add", "carol"), ("user_add", "dave"), ("user_passwd", "dave"), ("user_role", "dave"), ("user_disable", "dave"),
        ("jobs_claim", "carol")]
    assert all(line["actor"].startswith("cli:") for line in cli_lines)
    assert cli_lines[-1]["jobs"] == [job["id"]] and cli_lines[-1]["previous_owner"].startswith("h:")
    assert f'"job":"{job["id"]}","action":"owner_claimed"' in text


# ------------------------------------------------------------------------------------------ health and parsing
def test_health_details_hide_server_paths_from_non_admins(tmp_path):
    service = Service(tmp_path, shared=True, users=_users(tmp_path))
    service.manager.health = lambda: {"ok": False, "checks": {"jobs_root": {"ok": True, "hard": True, "path": "/srv/secret/jobs"},
                                                               "disk": {"ok": False, "free_gb": 1.5}}, "cached_at": "x"}
    api = service.api()
    try:
        for path in ("/api/health", "/api/ready"):
            status, payload, _ = call(api, "GET", path, {})
            assert payload == {"ok": False, "version": S.__version__} and status == (200 if path == "/api/health" else 503)
        _, _, _, alice = _login(api, {"username": "alice", "password": "alice-secret"})
        status, payload, _ = call(api, "GET", "/api/ready", {"Cookie": alice["Cookie"]})
        assert status == 503 and payload["checks"]["jobs_root"] == {"ok": True, "hard": True} and payload["checks"]["disk"]["free_gb"] == 1.5
        assert "/srv/secret" not in json.dumps(payload)
        _, _, _, admin = _login(api, {"username": "root", "password": "root-secret"})
        status, payload, _ = call(api, "GET", "/api/health", {"Cookie": admin["Cookie"]})
        assert payload["checks"]["jobs_root"]["path"] == "/srv/secret/jobs"
    finally:
        api.close(); service.close()


def test_non_ascii_tokens_are_rejected_not_crashed(tmp_path):
    service = Service(tmp_path, shared=True, token="t0ken-value")
    api = service.api()
    try:
        response, payload, _, session = _login(api, {"token": "tökén"})
        assert response.status == 401
        _, _, _, session = _login(api, {"token": "t0ken-value"})
        status, _, _ = call(api, "PUT", "/api/preferences", {**session, "X-CSRF-Token": "é" * 10}, {"a": 1})
        assert status == 403
    finally:
        api.close(); service.close()


def test_upload_file_names_lose_control_characters(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        _, headers, _ = service.session(api)
        headers = {k: v for k, v in headers.items() if k != "Content-Type"}
        body, ctype = multipart("hardening-boundary", [("stl", b"solid x\nendsolid x\n", "case\x01\x1b[31mA.stl"), ("units", "mm", None)])
        api.request("POST", "/api/jobs", body=body, headers={**headers, "Content-Type": ctype})
        response = api.getresponse(); payload = json.loads(response.read())
        assert response.status == 201, payload
        assert payload["job"]["source_filename"] == "case[31mA.stl" and payload["job"]["case_id"] == "case[31mA"
    finally:
        api.close(); service.close()


def test_deeply_nested_json_is_a_400_without_a_traceback(tmp_path, caplog):
    service = Service(tmp_path, shared=True, users=_users(tmp_path))
    api = service.api()
    try:
        with caplog.at_level(logging.INFO):
            api.request("POST", "/api/session", body="[" * 60000, headers={"Content-Type": "application/json"})
            response = api.getresponse(); payload = json.loads(response.read())
        assert response.status == 400 and "JSON" in payload["error"]["message"]
        assert not any(record.levelno >= logging.ERROR for record in caplog.records)
    finally:
        api.close(); service.close()


def test_disabling_the_last_user_warns_about_the_token_fallback(tmp_path, monkeypatch, capsys):
    root = tmp_path / "jobs"
    monkeypatch.setattr("sys.stdin", type("S", (), {"isatty": staticmethod(lambda: False)})())
    monkeypatch.setenv("WSS_DEPLOY_PASSWORD", "cli-secret-1")
    assert cli_main(["user", "add", "solo", "--admin", "--jobs-root", str(root)]) == 0
    capsys.readouterr()
    assert cli_main(["user", "disable", "solo", "--jobs-root", str(root)]) == 0
    assert "令牌登录" in capsys.readouterr().err
