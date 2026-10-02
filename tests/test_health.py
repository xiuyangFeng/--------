"""§19.6: GET /api/health — version only without a session; queue, worker, GPU, disk and stale reports with one."""
from __future__ import annotations

import wss_deploy
from wss_deploy.users import UserStore
from tests._c_helpers import Service, call, finished


def test_version_is_0_15():
    assert wss_deploy.__version__ == "0.16.4"


def test_health_without_session_is_minimal_and_with_session_is_detailed(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        status, payload, response = call(api, "GET", "/api/health", {})
        assert status == 200 and payload == {"ok": payload["ok"], "version": wss_deploy.__version__}
        assert response.getheader("Set-Cookie") is None          # health never creates a session
        cookie, headers, owner = service.session(api)
        finished(service.manager, owner=owner, case_id="A")
        service.manager.create(owner, content=b"queued", filename="q.stl")
        service.manager.create("someone-else", content=b"other", filename="o.stl")
        status, payload, _ = call(api, "GET", "/api/health", {"Cookie": cookie})
        assert status == 200 and isinstance(payload["ok"], bool) and payload["version"] == wss_deploy.__version__
        for key in ("started_at", "uptime_s", "queue", "worker_alive", "gpu", "disk_free_gb", "default_release", "stale_reports"):
            assert key in payload, key
        assert set(payload["queue"]) == {"queued", "running", "awaiting_input", "awaiting_confirmation"}
        assert payload["queue"]["queued"] == 2 and payload["queue_mine"]["queued"] == 1
        assert payload["worker_alive"] is False                  # the test manager never starts its workers
        assert set(payload["gpu"]) >= {"available", "name"}
        assert isinstance(payload["disk_free_gb"], float) and payload["uptime_s"] >= 0
        assert payload["stale_reports"] == 1                     # the finished job has a report without report_ui.json
        assert payload["started_at"][:4].isdigit() and ("+" in payload["started_at"][10:] or "-" in payload["started_at"][10:])
    finally:
        api.close(); service.close()


def test_health_in_shared_mode_needs_a_session_for_details(tmp_path):
    service = Service(tmp_path, shared=True, token="t0ken-value", users=UserStore(tmp_path / "users.json"))
    api = service.api()
    try:
        status, payload, _ = call(api, "GET", "/api/health", {})
        assert status == 200 and payload == {"ok": payload["ok"], "version": wss_deploy.__version__}
        status, _, cookie, _ = service.login(api, {"token": "t0ken-value"})
        assert status == 200
        status, payload, _ = call(api, "GET", "/api/health", {"Cookie": cookie})
        assert status == 200 and payload["shared"] is True and payload["login"] == "token" and "queue" in payload
    finally:
        api.close(); service.close()
