"""Event feed filters, exports and actionable operations counters."""
from datetime import datetime, timezone
import json
from urllib.parse import urlencode

import pytest

from tests._c_helpers import Service, call
from wss_deploy.users import UserStore


@pytest.fixture
def ops(tmp_path):
    users = UserStore(tmp_path / "users.json")
    users.add("admin", "admin-feed-secret", admin=True, display_name="Operator")
    users.add("bob", "bob-feed-secret", display_name="运营 张三")
    service = Service(tmp_path, shared=True, users=users)
    api = service.api()
    assert (login := service.login(api, {"username": "admin", "password": "admin-feed-secret"}))[0] == 200
    assert (other := service.login(api, {"username": "bob", "password": "bob-feed-secret"}))[0] == 200
    try:
        yield service, api, login[3], other[3]
    finally:
        api.close(); service.close()


def get(api, headers, path="/api/ops/events", **filters):
    return call(api, "GET", path + ("?" + urlencode(filters) if filters else ""), headers)


def append(store, **record):
    return store.record_event({"owner": "quiet", "action": "created", "source": "job", "status": "done",
                               "at": "2026-09-29T01:00:00Z", **record})


def bulk(store, count, *, owner="export", at="2026-09-29T01:00:00Z"):
    with store._db(write=True) as db:
        db.executemany("INSERT INTO events(at,action,actor,owner,job,status,source,details) VALUES(?,?,?,?,?,?,?,?)",
                       ((at, "created", "test", owner, "job", "done", "job", "{}") for _ in range(count)))


def test_combined_filters_time_boundaries_and_classification(ops):
    service, api, admin, _ = ops
    store = service.server.operations
    target = append(store, action="failed", status="failed", details={"error": "needle"})
    append(store, action="failed", status="failed", at="2026-09-29T00:59:59.999999Z", details={"error": "needle"})
    append(store, action="failed", status="failed", at="2026-09-29T01:00:00.000001Z", details={"error": "needle"})
    append(store, action="failed", status="failed", owner="other", details={"error": "needle"})
    append(store, action="started", status="running", at="2026-09-29T09:00:00+08:00")
    append(store, action="login_failed", source="http", status="401")
    append(store, action="user_disabled", source="user", status="")
    append(store, action="ticket_created", source="ticket", status="open")
    append(store, action="asset_reviewed", source="asset", status="approved")
    filters = dict(owner="quiet", action="failed", q="needle", category="job", severity="error",
                   since="2026-09-29T09:00:00+08:00", until="2026-09-29T01:00:00Z")
    status, result, _ = get(api, admin, **filters)
    assert status == 200 and [row["id"] for row in result["items"]] == [target["id"]]
    assert result["items"][0]["category"] == "job" and result["items"][0]["severity"] == "error"
    assert result["filters"]["since"] == "2026-09-29T01:00:00Z"
    assert result["snapshot_at"].endswith("Z") and result["latest_id"] == target["id"]
    assert result["truncated"] is False
    status, exported, response = get(api, admin, "/api/ops/events/export", **filters)
    assert status == 200 and response.getheader("Content-Type").startswith("application/json")
    assert "attachment" in response.getheader("Content-Disposition")
    assert exported["events"] == result["items"] and exported["filters"] == result["filters"]
    assert exported["exported_count"] == 1 and exported["filter_bounds"]["until"] == "inclusive"
    _, all_rows, _ = get(api, admin, owner="quiet", page_size=100)
    by_action = {row["action"]: row for row in all_rows["items"]}
    assert (by_action["login_failed"]["category"], by_action["login_failed"]["severity"]) == ("auth", "warning")
    assert (by_action["user_disabled"]["category"], by_action["user_disabled"]["severity"]) == ("user", "warning")
    assert (by_action["ticket_created"]["category"], by_action["ticket_created"]["severity"]) == ("ticket", "info")
    assert (by_action["asset_reviewed"]["category"], by_action["asset_reviewed"]["severity"]) == ("asset", "info")


@pytest.mark.parametrize("filters", [
    {"category": "all"}, {"severity": "critical"}, {"since": "2026-09-29"},
    {"until": "2026-09-29T01:00:00"}, {"since": "2026-02-30T01:00:00Z"},
    {"since": "2026-09-29T01:00:00+25:00"},
    {"since": "2026-09-30T00:00:00Z", "until": "2026-09-29T00:00:00Z"},
])
def test_invalid_event_filters_are_rejected_on_list_and_export(ops, filters):
    _, api, admin, _ = ops
    assert get(api, admin, **filters)[0] == 400
    assert get(api, admin, "/api/ops/events/export", **filters)[0] == 400


def test_new_filters_precede_limit_for_persistent_and_legacy_history(ops):
    service, api, admin, _ = ops
    store = service.server.operations
    target = append(store, at="2020-01-01T00:00:00Z", action="failed", status="failed", details={"error": "needle"})
    bulk(store, 10005, owner="quiet")
    with service.manager.lock:
        service.manager.jobs["legacy"] = {"id": "legacy", "owner": "quiet", "status": "done", "events": [
            {"at": "2020-01-01T00:00:00Z", "action": "failed", "status": "failed", "error": "needle"},
            *({"at": "2026-09-29T01:00:00Z", "action": "progress", "status": "running"} for _ in range(10005)),
        ]}
    filters = dict(owner="quiet", category="job", severity="error", q="needle", since="2019-01-01T00:00:00Z", until="2021-01-01T00:00:00Z")
    status, result, _ = get(api, admin, **filters)
    assert status == 200 and result["total"] == 2 and result["truncated"] is False
    assert target["id"] in {row["id"] for row in result["items"]}
    legacy = next(row for row in result["items"] if row["source"] == "historical_job")
    assert legacy["actor"] == "" and legacy["severity"] == "error"
    # An append does not renumber the first historical event's stable identity.
    with service.manager.lock:
        service.manager.jobs["legacy"]["events"].append({"at": "2026-09-29T02:00:00Z", "action": "progress"})
    _, again, _ = get(api, admin, **filters)
    assert [row["id"] for row in again["items"]] == [row["id"] for row in result["items"]]


def test_export_limit_true_truncation_stable_sort_and_no_self_event(ops):
    service, api, admin, _ = ops
    store = service.server.operations
    bulk(store, 10000)
    before = store.events()[1]
    status, exact, _ = get(api, admin, "/api/ops/events/export", owner="export")
    assert status == 200 and exact["exported_count"] == 10000 and exact["truncated"] is False
    assert len(exact["events"]) == 10000 and store.events()[1] == before
    ids = [row["id"] for row in exact["events"]]
    assert ids == sorted(ids, reverse=True)
    newest = append(store, owner="export")
    status, clipped, _ = get(api, admin, "/api/ops/events/export", owner="export")
    assert status == 200 and clipped["exported_count"] == 10000 and clipped["truncated"] is True
    assert clipped["latest_id"] == newest["id"] and clipped["events"][0]["id"] == newest["id"]
    assert store.events()[1] == before + 1
    _, page1, _ = get(api, admin, owner="export", page=1, page_size=2)
    _, page2, _ = get(api, admin, owner="export", page=2, page_size=2)
    assert [row["id"] for row in page1["items"] + page2["items"]] == [row["id"] for row in clipped["events"][:4]]


def test_export_requires_admin_and_live_session(ops):
    service, api, admin, bob = ops
    assert get(api, {}, "/api/ops/events/export")[0] == 401
    assert get(api, bob, "/api/ops/events/export")[0] == 403
    sid = service.sessions.lookup(admin["Cookie"])[0]
    service.sessions.destroy(sid)
    assert get(api, admin, "/api/ops/events/export")[0] == 401


def test_overview_todos_match_shortcuts_and_user_filters(ops):
    service, api, admin, _ = ops
    store = service.server.operations
    now = datetime(2026, 9, 29, tzinfo=timezone.utc).timestamp()
    store.clock = lambda: now
    for index, delta in enumerate((-1, 7 * 86400, 7 * 86400 + 1)):
        folder = service.manager.root / ".trash" / f"trash{index}"; folder.mkdir(parents=True)
        (folder / "trash.json").write_text(json.dumps({"job_snapshot": {"id": folder.name, "owner": "bob", "status": "done"},
            "expires_at": datetime.fromtimestamp(now + delta, timezone.utc).isoformat()}))
    for index, (status, assignee) in enumerate((("open", ""), ("in_progress", ""), ("open", "admin"), ("closed", ""))):
        ticket = store.create_ticket(owner="bob", title=f"ticket {index}", description="check")
        store.update_ticket(ticket["id"], version=0, status=status, assignee=assignee)
    for index in range(2):
        folder = service.manager.root / f"asset{index}"; folder.mkdir(); (folder / "input.stl").write_bytes(f"STL {index}".encode())
        asset = store.archive_asset(job_id=folder.name, owner="bob")
        if index: store.review_asset(asset["sha256"], review_status="approved", version=0)
    service.sessions.users.disable("bob", keep_admin=True)
    status, overview, _ = get(api, admin, "/api/ops/overview")
    assert status == 200
    assert overview["counts"]["pending_review_assets"] == 1
    assert overview["counts"]["trash_expiring"] == 2
    assert overview["counts"]["unassigned_tickets"] == 2
    assert overview["counts"]["tickets_open"] == 3
    assert {row["code"] for row in overview["warnings"]} >= {"pending_review_assets", "trash_expiring", "unassigned_tickets"}
    _, tickets, _ = get(api, admin, "/api/ops/tickets", unassigned="true")
    _, unresolved, _ = get(api, admin, "/api/ops/tickets", status="unresolved")
    _, trash, _ = get(api, admin, "/api/ops/jobs", location="trash", expires_within_days=7)
    assert tickets["total"] == 2 and unresolved["total"] == 3 and trash["total"] == 2
    _, users, _ = get(api, admin, "/api/ops/users", q="张三", disabled="true")
    assert [row["username"] for row in users["items"]] == ["bob"]
    _, users, _ = get(api, admin, "/api/ops/users", q="operator", disabled="false")
    assert [row["username"] for row in users["items"]] == ["admin"]
    assert get(api, admin, "/api/ops/users", disabled="yes")[0] == 400
    assert get(api, admin, "/api/ops/tickets", unassigned="yes")[0] == 400
    assert get(api, admin, "/api/ops/jobs", location="active", expires_within_days=7)[0] == 400


def test_local_operator_users_list_without_named_accounts(tmp_path):
    import threading
    from types import SimpleNamespace
    from wss_deploy.operations import OperationsStore, event_severity
    from wss_deploy.operations_http import handle_get

    class Handler:
        def __init__(self):
            self.server = SimpleNamespace(operations=OperationsStore(tmp_path), sessions=SimpleNamespace(shared=False, users=None),
                                          manager=SimpleNamespace(root=tmp_path, jobs={}, lock=threading.RLock()))
            self.result = None
        def _json(self, value): self.result = value
    handler = Handler()
    assert handle_get(handler, {"owner": "local-session"}, "/api/ops/users", lambda key, default="": default,
                      lambda key, default: default)
    assert handler.result["items"] == [] and handler.result["total"] == 0
    assert event_severity("deleted", "failed") == "warning"
