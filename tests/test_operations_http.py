"""HTTP contract tests for the separate operations console.

The service is always a short-lived, shared-mode instance backed by ``tmp_path``.
These tests intentionally exercise the permission and retention boundaries which
are easy to lose when the operations page evolves independently of the workbench.
"""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from wss_deploy.users import UserStore
from tests._c_helpers import Service, call, finished


def _users(root):
    users = UserStore(root / "users.json")
    users.add("alice", "alice-secret")
    users.add("bob", "bob-secret")
    users.add("root", "root-secret", admin=True)
    return users


@pytest.fixture
def service(tmp_path):
    users = _users(tmp_path)
    svc = Service(tmp_path, shared=True, token="legacy-token", users=users)
    try:
        yield svc
    finally:
        svc.close()


def _login(svc, api, username, password):
    status, payload, cookie, headers = svc.login(api, {"username": username, "password": password})
    assert status == 200, payload
    return cookie, headers


def _json_headers(headers, *, csrf=True, origin=None):
    result = dict(headers)
    result["Content-Type"] = "application/json"
    if not csrf:
        result.pop("X-CSRF-Token", None)
    if origin is not None:
        result["Origin"] = origin
    return result


def _archive_job(svc, job_id, headers, *, location="active", note=""):
    api = svc.api()
    try:
        return call(api, "POST", "/api/ops/assets", _json_headers(headers),
                    {"job_id": job_id, "location": location, "note": note})
    finally:
        api.close()


def test_ops_auth_csrf_origin_and_owner_scoping(service):
    api = service.api()
    try:
        # No session is a normal API authentication failure, while a named user
        # has a session but is still forbidden from the operations namespace.
        for endpoint in ("/api/ops/overview", "/api/ops/jobs", "/api/ops/users",
                         "/api/ops/events", "/api/ops/tickets", "/api/ops/assets"):
            assert call(api, "GET", endpoint, {})[0] == 401
        _, alice = _login(service, api, "alice", "alice-secret")
        _, bob = _login(service, api, "bob", "bob-secret")
        _, root = _login(service, api, "root", "root-secret")
        for endpoint in ("/api/ops/overview", "/api/ops/jobs", "/api/ops/users",
                         "/api/ops/events", "/api/ops/tickets", "/api/ops/assets"):
            assert call(api, "GET", endpoint, alice)[0] == 403
            assert call(api, "GET", endpoint, bob)[0] == 403
        # The shell is public so it can render its own login panel; the data
        # endpoints above remain the actual authorization boundary.
        assert call(api, "GET", "/ops", {"Cookie": alice["Cookie"]})[0] == 200
        assert call(api, "GET", "/api/ops/overview", root)[0] == 200

        # Mutations require both the synchroniser token and a same-origin
        # Origin header.  An invalid Origin must not be a CORS bypass.
        body = {"title": "Need help", "description": "upload stalled"}
        no_csrf = _json_headers(alice, csrf=False)
        assert call(api, "POST", "/api/support/tickets", no_csrf, body)[0] == 403
        evil_origin = _json_headers(alice, origin="https://evil.invalid")
        assert call(api, "POST", "/api/support/tickets", evil_origin, body)[0] == 403

        # Jobs and tickets remain owner-scoped for normal users.  Admin gets an
        # explicit cross-owner operations view.
        a_job = finished(service.manager, owner="alice", case_id="ALICE")
        b_job = finished(service.manager, owner="bob", case_id="BOB")
        status, alice_jobs, _ = call(api, "GET", "/api/support/tickets", {"Cookie": alice["Cookie"]})
        assert status == 200 and alice_jobs["items"] == []
        status, all_jobs, _ = call(api, "GET", "/api/ops/jobs?location=active", {"Cookie": root["Cookie"]})
        assert status == 200 and {x["id"] for x in all_jobs["items"]} >= {a_job["id"], b_job["id"]}
        status, only_bob, _ = call(api, "GET", "/api/ops/jobs?location=active&owner=bob", {"Cookie": root["Cookie"]})
        assert status == 200 and {x["owner"] for x in only_bob["items"]} == {"bob"}
    finally:
        api.close()


def test_support_ticket_lifecycle_is_isolated_and_optimistic(service):
    api = service.api()
    try:
        _, alice = _login(service, api, "alice", "alice-secret")
        _, bob = _login(service, api, "bob", "bob-secret")
        _, root = _login(service, api, "root", "root-secret")
        job = finished(service.manager, owner="alice", case_id="ticket-case")
        status, created, _ = call(api, "POST", "/api/support/tickets", alice,
                                  {"title": "Input issue", "description": "Please inspect", "job_id": job["id"]})
        assert status == 201
        ticket = created["ticket"]
        ident, version = ticket["id"], ticket["version"]
        assert ticket["owner"] == "alice" and ticket["job_id"] == job["id"]

        status, hidden, _ = call(api, "GET", "/api/support/tickets", {"Cookie": bob["Cookie"]})
        assert status == 200 and all(row["id"] != ident for row in hidden["items"])
        assert call(api, "GET", f"/api/support/tickets/{ident}", {"Cookie": bob["Cookie"]})[0] == 404
        status, reply, _ = call(api, "POST", f"/api/support/tickets/{ident}", alice,
                                {"version": version, "message": "More details"})
        assert status == 200 and reply["ticket"]["version"] == version + 1
        assert call(api, "POST", f"/api/support/tickets/{ident}", alice,
                    {"version": version, "message": "stale"})[0] == 409

        status, handled, _ = call(api, "POST", f"/api/ops/tickets/{ident}", root,
                                  {"version": version + 1, "status": "in_progress", "assignee": "root", "message": "triaged"})
        assert status == 200 and handled["ticket"]["status"] == "in_progress"
        assert call(api, "POST", f"/api/ops/tickets/{ident}", root,
                    {"version": version + 1, "status": "resolved"})[0] == 409
        status, listed, _ = call(api, "GET", "/api/ops/tickets", {"Cookie": root["Cookie"]})
        assert status == 200 and any(row["id"] == ident for row in listed["items"])
    finally:
        api.close()


def test_support_ignores_forged_owner_and_status_and_rejects_other_job(service):
    api = service.api()
    try:
        _, alice = _login(service, api, "alice", "alice-secret")
        bob_job = finished(service.manager, owner="bob", case_id="BOB_PRIVATE")
        status, forged, _ = call(api, "POST", "/api/support/tickets", alice,
                                 {"owner": "bob", "status": "resolved", "title": "Forged", "description": "keep owner"})
        assert status == 201
        assert forged["ticket"]["owner"] == "alice" and forged["ticket"]["status"] == "open"
        status, _, _ = call(api, "POST", "/api/support/tickets", alice,
                            {"title": "Cross owner", "description": "not allowed", "job_id": bob_job["id"]})
        assert status == 404
    finally:
        api.close()


def test_active_and_trashed_assets_are_deduplicated_and_survive_purge(service):
    api = service.api()
    try:
        _, alice = _login(service, api, "alice", "alice-secret")
        _, root = _login(service, api, "root", "root-secret")
        content = b"solid retained-stl\nendsolid retained-stl\n"
        active = finished(service.manager, owner="alice", case_id="CASE_ACTIVE", content=content)
        trashed = finished(service.manager, owner="alice", case_id="CASE_TRASH", content=content)
        # Explicitly archive the live source first.
        status, first, _ = _archive_job(service, active["id"], root)
        assert status == 201
        asset = first["asset"]
        sha = asset["sha256"]
        assert asset["sources"][0]["job_id"] == active["id"]

        # Delete -> trash, then archive the same bytes again.  The asset is one
        # content-addressed record but retains both source jobs.
        assert call(api, "POST", f"/api/jobs/{trashed['id']}/delete", alice, {"version": trashed["version"]})[0] == 200
        status, second, _ = _archive_job(service, trashed["id"], root, location="trash")
        assert status == 201 and second["asset"]["sha256"] == sha
        assert {x["job_id"] for x in second["asset"]["sources"]} == {active["id"], trashed["id"]}

        # The owner may purge the trashed source; the explicitly archived bytes
        # remain downloadable and the public manifest contains no owner, case,
        # or original upload filename.
        assert call(api, "POST", f"/api/trash/{trashed['id']}/purge", alice, {})[0] == 200
        status, response_asset, response = call(api, "GET", f"/api/ops/assets/{sha}/download", {"Cookie": root["Cookie"]})
        assert status == 200 and response.getheader("Content-Type", "").startswith("application/zip")
        with zipfile.ZipFile(io.BytesIO(response_asset if isinstance(response_asset, bytes) else b"")) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            encoded = json.dumps(manifest, ensure_ascii=False)
            assert set(archive.namelist()) == {"input.stl", "manifest.json"}
            assert "alice" not in encoded and "CASE_" not in encoded and "source_filename" not in encoded
        status, assets, _ = call(api, "GET", "/api/ops/assets", {"Cookie": root["Cookie"]})
        assert status == 200 and any(row["sha256"] == sha for row in assets["items"])
    finally:
        api.close()


def test_asset_review_uses_optimistic_version_conflicts(service):
    api = service.api()
    try:
        _, root = _login(service, api, "root", "root-secret")
        job = finished(service.manager, owner="alice", case_id="REVIEW_CASE")
        status, archived, _ = _archive_job(service, job["id"], root)
        assert status == 201
        sha = archived["asset"]["sha256"]
        status, reviewed, _ = call(api, "POST", f"/api/ops/assets/{sha}", root,
                                   {"version": 0, "review_status": "approved", "note": "keep"})
        assert status == 200 and reviewed["asset"]["version"] == 1
        status, _, _ = call(api, "POST", f"/api/ops/assets/{sha}", root,
                            {"version": 0, "review_status": "excluded"})
        assert status == 409
    finally:
        api.close()


def test_ops_events_include_job_ticket_asset_and_delete(service):
    api = service.api()
    try:
        _, alice = _login(service, api, "alice", "alice-secret")
        _, root = _login(service, api, "root", "root-secret")
        job = finished(service.manager, owner="alice", case_id="EVENT_CASE")
        status, created, _ = call(api, "POST", "/api/support/tickets", alice,
                                  {"title": "Event ticket", "description": "audit me", "job_id": job["id"]})
        assert status == 201
        status, archived, _ = _archive_job(service, job["id"], root)
        assert status == 201
        assert call(api, "POST", f"/api/jobs/{job['id']}/delete", alice, {"version": job["version"]})[0] == 200
        status, events, _ = call(api, "GET", "/api/ops/events?page_size=100", {"Cookie": root["Cookie"]})
        assert status == 200
        actions = {event["action"] for event in events["items"]}
        assert {"created", "deleted", "ticket_created", "asset_archived"} <= actions
    finally:
        api.close()


def test_audit_sink_survives_delete_purge_and_service_rebuild(service, tmp_path):
    api = service.api()
    job_id = None
    try:
        _, alice = _login(service, api, "alice", "alice-secret")
        job = finished(service.manager, owner="alice", case_id="PERSIST_CASE")
        job_id = job["id"]
        assert call(api, "POST", f"/api/jobs/{job_id}/delete", alice, {"version": job["version"]})[0] == 200
        assert call(api, "POST", f"/api/trash/{job_id}/purge", alice, {})[0] == 200
    finally:
        api.close()
        service.close()
    # A new HTTP server and JobManager must read the operations SQLite journal,
    # even though the source job and its trash directory are gone.
    rebuilt = Service(tmp_path, shared=True, token="legacy-token", users=UserStore(tmp_path / "users.json"))
    rebuilt_api = rebuilt.api()
    try:
        _, root = _login(rebuilt, rebuilt_api, "root", "root-secret")
        status, events, _ = call(rebuilt_api, "GET", "/api/ops/events?owner=alice&page_size=100", {"Cookie": root["Cookie"]})
        assert status == 200
        rows = [event for event in events["items"] if event.get("job") == job_id]
        assert {event["action"] for event in rows} >= {"created", "finished", "deleted", "purged"}
        assert all(event.get("owner") == "alice" for event in rows)
    finally:
        rebuilt_api.close()
        rebuilt.close()


def test_login_audit_and_user_listing_never_expose_password_or_session_data(service):
    api = service.api()
    try:
        status, _, _, _ = service.login(api, {"username": "alice", "password": "wrong-secret-that-must-not-leak"})
        assert status == 401
        _, root = _login(service, api, "root", "root-secret")
        status, users, _ = call(api, "GET", "/api/ops/users", {"Cookie": root["Cookie"]})
        assert status == 200
        status, events, _ = call(api, "GET", "/api/ops/events?page_size=100", {"Cookie": root["Cookie"]})
        assert status == 200
        encoded = json.dumps({"users": users, "events": events}, ensure_ascii=False)
        for secret in ("wrong-secret-that-must-not-leak", "root-secret", "alice-secret", "password_hash", "csrf_token", "wss_session"):
            assert secret not in encoded
        assert all(not any("password" in key.casefold() for key in row) for row in users["items"])
    finally:
        api.close()


def test_user_disable_revokes_session_and_preserves_last_admin(service):
    api = service.api()
    try:
        _, bob = _login(service, api, "bob", "bob-secret")
        _, root = _login(service, api, "root", "root-secret")
        status, disabled, _ = call(api, "POST", "/api/ops/users/bob", root, {"disabled": True, "reason": "offboarding"})
        assert status == 200 and disabled["user"]["disabled"] is True
        assert call(api, "GET", "/api/support/tickets", {"Cookie": bob["Cookie"]})[0] == 401
        # The endpoint must not allow an administrator to remove the only
        # currently authenticated administrator.
        status, _, _ = call(api, "POST", "/api/ops/users/root", root, {"disabled": True})
        assert status == 409
        assert service.sessions.users.credential("root")["disabled"] is False
    finally:
        api.close()


def test_malicious_paths_symlinks_and_bad_payloads_are_typed_errors(service, tmp_path):
    api = service.api()
    try:
        _, root = _login(service, api, "root", "root-secret")
        job = finished(service.manager, owner="alice", case_id="SYMLINK_CASE")
        outside = tmp_path / "outside.stl"
        outside.write_bytes(b"outside")
        source = service.manager.root / job["id"] / "input.stl"
        source.unlink()
        source.symlink_to(outside)
        assert _archive_job(service, job["id"], root)[0] in {400, 404}
        assert call(api, "POST", "/api/ops/assets", root, {"job_id": "../escape", "location": "active"})[0] in {400, 404}
        assert call(api, "POST", "/api/ops/assets", root, {"job_id": job["id"], "location": "../../etc"})[0] in {400, 404}
        assert call(api, "GET", "/api/ops/jobs?page=abc", {"Cookie": root["Cookie"]})[0] == 400
        assert call(api, "POST", "/api/support/tickets", root, {"title": [], "description": {}})[0] in {400, 403}
        assert call(api, "POST", "/api/ops/users/bob", root, {"disabled": "yes"})[0] == 400
    finally:
        api.close()


def test_asset_download_shares_bundle_budget_and_releases_failed_requests(service):
    import threading

    api = service.api()
    try:
        _, root = _login(service, api, "root", "root-secret")
        content = b"solid download\nendsolid download\n"
        job = finished(service.manager, owner="alice", case_id="DOWNLOAD", content=content)
        status, archived, _ = _archive_job(service, job["id"], root)
        assert status == 201
        sha = archived["asset"]["sha256"]
        endpoint = f"/api/ops/assets/{sha}/download"
        slots = service.server.bundle_slots = threading.BoundedSemaphore(1)
        assert slots.acquire(blocking=False)
        try:
            status, _, response = call(api, "GET", endpoint, root)
            assert status == 429 and response.getheader("Retry-After") == "2"
        finally:
            slots.release()
        service.server.max_bundle_bytes = 1
        assert call(api, "GET", endpoint, root)[0] == 413
        assert slots.acquire(blocking=False)
        slots.release()
        service.server.max_bundle_bytes = 1024 * 1024
        stored = service.server.operations.assets_dir / sha
        stored.write_bytes(b"corrupt")
        assert call(api, "GET", endpoint, root)[0] == 409
        assert slots.acquire(blocking=False)
        slots.release()
        stored.write_bytes(content)
        status, raw, _ = call(api, "GET", endpoint, root)
        assert status == 200
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            assert archive.read("input.stl") == content
        assert not list(service.server.operations.directory.glob("asset-*.zip"))
        assert not list(service.server.tmp_dir.glob("asset-*.zip"))
    finally:
        api.close()


def test_audit_failure_rolls_back_ticket_but_reports_saved_account_truthfully(service, monkeypatch, caplog):
    import logging
    import sqlite3

    api = service.api()
    try:
        _, root = _login(service, api, "root", "root-secret")
        _, bob = _login(service, api, "bob", "bob-secret")
        store = service.server.operations
        def unavailable(*_args, **_kwargs):
            raise sqlite3.OperationalError("injected unavailable audit")
        monkeypatch.setattr(store, "_record_event_db", unavailable)
        status, _, _ = call(api, "POST", "/api/support/tickets", root, {"title": "rollback", "description": "not committed"})
        assert status == 500
        assert store.tickets()[1] == 0
        caplog.set_level(logging.INFO, logger="wss_deploy.audit")
        status, saved, _ = call(api, "POST", "/api/ops/users/bob", root, {"disabled": True, "reason": "test outage"})
        assert status == 200 and saved["user"]["disabled"] is True
        assert saved["warning"]["code"] == "audit_unavailable"
        assert service.sessions.users.credential("bob")["disabled"] is True
        assert call(api, "GET", "/api/support/tickets", bob)[0] == 401
        assert any('"action":"user_disabled"' in record.getMessage() and '"owner":"bob"' in record.getMessage()
                   for record in caplog.records if record.name == "wss_deploy.audit")
        status, saved, _ = call(api, "POST", "/api/ops/users/bob", root, {"disabled": False})
        assert status == 200 and saved["user"]["disabled"] is False
        assert saved["warning"]["code"] == "audit_unavailable"
        # Re-enabling never revives credentials issued before the disable.
        assert call(api, "GET", "/api/support/tickets", bob)[0] == 401
    finally:
        api.close()
