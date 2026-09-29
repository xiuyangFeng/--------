import json
from pathlib import Path
import zipfile
import io

from wss_deploy.operations import OperationsStore


def test_event_ticket_and_asset_roundtrip(tmp_path):
    root = Path(tmp_path)
    job = root / "job-1"; job.mkdir()
    (job / "input.stl").write_bytes(b"solid x\nendsolid x\n")
    store = OperationsStore(root)
    event = store.record_event({"action": "submitted", "owner": "alice", "job": "job-1", "details": {"ok": True}})
    assert event["action"] == "submitted"
    ticket = store.create_ticket(owner="alice", title="Help", description="A problem")
    assert len(ticket["id"]) == 16
    assert store.update_ticket(ticket["id"], version=0, message="seen", actor="alice")["version"] == 1
    asset = store.archive_asset(job_id="job-1", owner="alice", case_id="c", release_id="r")
    assert len(asset["sha256"]) == 64 and asset["sources"][0]["job_id"] == "job-1"
    data = store.asset_zip(asset["sha256"])
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert set(archive.namelist()) == {"input.stl", "manifest.json"}
        manifest = json.loads(archive.read("manifest.json"))
        assert "owner" not in json.dumps(manifest)


def test_symlink_stl_refused(tmp_path):
    root = Path(tmp_path); job = root / "job"; job.mkdir()
    outside = root / "outside.stl"; outside.write_bytes(b"x")
    (job / "input.stl").symlink_to(outside)
    store = OperationsStore(root)
    try:
        store.archive_asset(job_id="job", owner="a")
    except (ValueError, FileNotFoundError):
        pass
    else:
        raise AssertionError("symlink must not be archived")


def test_versions_are_atomic_across_store_instances(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from wss_deploy.operations import ConflictError

    first, second = OperationsStore(tmp_path), OperationsStore(tmp_path)
    ticket = first.create_ticket(owner="alice", title="race", description="concurrency")
    barrier = threading.Barrier(2)

    def reply(store, message):
        barrier.wait()
        try:
            return store.update_ticket(ticket["id"], version=0, message=message)["version"]
        except ConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(reply, first, "A"), pool.submit(reply, second, "B")
        assert sorted([str(a.result()), str(b.result())]) == ["1", "conflict"]
    assert len(first.ticket(ticket["id"])["messages"]) == 1

    job = tmp_path / "job"; job.mkdir(); (job / "input.stl").write_bytes(b"STL")
    asset = first.archive_asset(job_id="job", owner="alice")
    barrier = threading.Barrier(2)

    def review(store, status):
        barrier.wait()
        try:
            return store.review_asset(asset["sha256"], version=0, review_status=status)["version"]
        except ConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(review, first, "approved"), pool.submit(review, second, "excluded")
        assert sorted([str(a.result()), str(b.result())]) == ["1", "conflict"]


def test_archive_rejects_parent_symlinks_traversal_and_digest_mismatch(tmp_path):
    import pytest
    from wss_deploy.operations import ConflictError

    root = tmp_path / "jobs"; root.mkdir()
    store = OperationsStore(root)
    outside = tmp_path / "outside"; outside.mkdir(); (outside / "input.stl").write_bytes(b"secret")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (root / ".trash").symlink_to(tmp_path, target_is_directory=True)
    for job_id, location in (("linked", "active"), ("outside", "trash"), ("..", "active"), ("../outside", "active")):
        with pytest.raises((ValueError, FileNotFoundError)):
            store.archive_asset(job_id=job_id, owner="alice", location=location)
    job = root / "job"; job.mkdir(); (job / "input.stl").write_bytes(b"STL")
    with pytest.raises(ConflictError):
        store.archive_asset(job_id="job", owner="alice", expected_sha256="0" * 64)
    assert store.assets()[1] == 0
    assert not list(store.assets_dir.iterdir())


def test_asset_metadata_note_dedup_and_streamed_corruption_check(tmp_path):
    import hashlib
    import pytest
    from wss_deploy.operations import ConflictError

    content = b"solid patient-header\nendsolid patient-header\n"
    sha = hashlib.sha256(content).hexdigest()
    job = tmp_path / "job"; job.mkdir(); (job / "input.stl").write_bytes(content)
    store = OperationsStore(tmp_path)
    source = {"id": "job", "owner": "alice", "patient_id": "secret-patient", "source_filename": "secret.stl", "input_sha256": sha,
              "params": {"units": "mm", "orientation": "LPS", "private": "hidden"},
              "model_release": {"id": "release-1", "fingerprint": "abc", "contract": {"protocol": "single_frame_wss"}},
              "mapping": {"inlet": "in"}, "run_identity": "run-hash"}
    asset = store.archive_asset(job_id="job", owner="alice", note="patient private note", source_job=source)
    second = store.archive_asset(job_id="job", owner="alice", note="must not overwrite", source_job=source)
    assert second["note"] == asset["note"] and second["version"] == 0 and second["sources_total"] == 1
    destination = tmp_path / "asset.zip"
    store.write_asset_zip(sha, destination)
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        encoded = json.dumps(manifest)
        assert "secret" not in encoded and "alice" not in encoded and "patient private" not in encoded
        assert manifest["sources"][0]["metadata"]["params"] == {"units": "mm", "orientation": "LPS"}
        assert manifest["medical_deidentification_review"] == "not_performed"
        assert archive.read("input.stl") == content
    (store.assets_dir / sha).write_bytes(b"corrupt")
    with pytest.raises(ConflictError):
        store.write_asset_zip(sha, destination)
    assert not destination.exists()


def test_tickets_validate_optional_fields_and_bound_messages(tmp_path):
    import pytest
    from wss_deploy.operations import ConflictError, MAX_MESSAGES

    store = OperationsStore(tmp_path)
    ticket = store.create_ticket(owner="alice", title="help", description="details")
    for values in ({"assignee": {}}, {"message": []}, {"message": None}, {"priority": []}):
        with pytest.raises(ValueError):
            store.update_ticket(ticket["id"], version=0, **values)
    with pytest.raises(ConflictError):
        store.update_ticket(ticket["id"], version=True)
    ticket = store.update_ticket(ticket["id"], version=0, status="in_progress")
    for _ in range(MAX_MESSAGES):
        ticket = store.update_ticket(ticket["id"], version=ticket["version"], message="more")
    with pytest.raises(ValueError):
        store.update_ticket(ticket["id"], version=ticket["version"], message="overflow")
    # Status changes remain possible after the reply limit.
    assert store.update_ticket(ticket["id"], version=ticket["version"], status="resolved")["status"] == "resolved"


def test_database_private_and_filtered_event_limit(tmp_path):
    import os
    store = OperationsStore(tmp_path)
    assert os.stat(store.db_path).st_mode & 0o777 == 0o600
    store.record_event({"owner": "quiet", "action": "created"})
    for _ in range(105):
        store.record_event({"owner": "busy", "action": "created"})
    assert len(store.recent_events(10000)) == 106
    assert [row["owner"] for row in store.recent_events(1, owner="quiet")] == ["quiet"]
    assert len(store.recent_events(10, q="quiet")) == 1
    assert len(store.events(q="quiet")[0]) == 1


def test_support_scope_is_local_session_specific_and_dates_normalize(tmp_path):
    import threading
    from types import SimpleNamespace
    from wss_deploy.jobs import JobError
    from wss_deploy.operations_http import handle_get, _dynamic_events
    import pytest

    store = OperationsStore(tmp_path)
    alice = store.create_ticket(owner="local-a", title="A", description="private")
    store.create_ticket(owner="local-b", title="B", description="private")
    manager = SimpleNamespace(root=tmp_path, lock=threading.RLock(), jobs={})
    server = SimpleNamespace(manager=manager, operations=store, sessions=SimpleNamespace(shared=False))
    class Handler:
        def __init__(self): self.server = server; self.result = None
        def _json(self, value): self.result = value
    handler = Handler()
    one = lambda key, default="": default
    integer = lambda key, default: default
    assert handle_get(handler, {"owner": "local-b"}, "/api/support/tickets", one, integer)
    assert {ticket["owner"] for ticket in handler.result["items"]} == {"local-b"}
    with pytest.raises(JobError) as error:
        handle_get(handler, {"owner": "local-b"}, "/api/support/tickets/" + alice["id"], one, integer)
    assert error.value.status == 404

    store.record_event({"action": "older", "at": "2026-09-29T09:00:00+08:00"})
    store.record_event({"action": "newer", "at": "2026-09-29T02:00:00Z"})
    manager.jobs["job"] = {"id": "job", "owner": "alice", "events": [{"action": "legacy", "at": "2026-09-29T02:30:00Z"}]}
    rows, _ = _dynamic_events(manager, store)
    assert [row["action"] for row in rows[:3]] == ["legacy", "newer", "older"]
    assert rows[0]["actor"] == "" and rows[0]["source"] == "historical_job"
    store.record_event({"job": "job", "action": "legacy", "owner": "alice", "actor": "system", "source": "job", "at": "2026-09-29T10:30:00+08:00"})
    rows, _ = _dynamic_events(manager, store)
    assert len([row for row in rows if row["action"] == "legacy"]) == 1


def test_sqlite_connections_are_closed(tmp_path, monkeypatch):
    import sqlite3
    import pytest
    from wss_deploy import operations
    connect = sqlite3.connect
    opened = []
    def tracked(*args, **kwargs):
        connection = connect(*args, **kwargs)
        opened.append(connection)
        return connection
    monkeypatch.setattr(operations.sqlite3, "connect", tracked)
    store = OperationsStore(tmp_path)
    store.record_event({"action": "created"})
    store.recent_events()
    store.create_ticket(owner="a", title="b", description="c")
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_restart_cleans_only_old_operations_temporary_files(tmp_path):
    import os
    import time
    from wss_deploy.operations import TEMP_MAX_AGE_SECONDS

    store = OperationsStore(tmp_path)
    old_copy = store.assets_dir / ".stage-abandoned"
    new_copy = store.assets_dir / ".stage-active"
    old_export = store.directory / "asset-abandoned.zip"
    published = store.assets_dir / ("a" * 64)
    unrelated = store.directory / "operator-notes.txt"
    outside = tmp_path / "outside.txt"
    for path in (old_copy, new_copy, old_export, published, unrelated, outside):
        path.write_bytes(b"retained")
    old = time.time() - TEMP_MAX_AGE_SECONDS - 60
    for path in (old_copy, old_export, published, unrelated, outside):
        os.utime(path, (old, old))
    linked = store.assets_dir / ".stage-link"
    linked.symlink_to(outside)
    OperationsStore(tmp_path)
    assert not old_copy.exists() and not old_export.exists()
    assert all(path.exists() for path in (new_copy, published, unrelated, outside))
    assert linked.is_symlink()


def test_stopped_store_backup_restores_tickets_audit_and_retained_stl(tmp_path):
    import shutil

    original, restored = tmp_path / "original", tmp_path / "restored"
    store = OperationsStore(original)
    job = original / "job"; job.mkdir(); (job / "input.stl").write_bytes(b"solid backup\nendsolid backup\n")
    ticket = store.create_ticket(owner="alice", title="restore", description="backup verification")
    store.update_ticket(ticket["id"], version=0, status="in_progress", message="retained reply", actor="admin")
    store.record_event({"action": "backup_probe", "owner": "alice", "job": "job"})
    asset = store.archive_asset(job_id="job", owner="alice", note="keep")
    store.review_asset(asset["sha256"], version=0, review_status="approved")
    # No connections or mutations remain active: this exercises the documented
    # stopped-service copy, not an unsafe copy of a live SQLite database.
    restored.mkdir()
    shutil.copytree(store.directory, restored / ".operations")
    copy = OperationsStore(restored)
    assert copy.ticket(ticket["id"])["messages"][0]["text"] == "retained reply"
    assert copy.recent_events()[0]["action"] == "backup_probe"
    assert copy.asset(asset["sha256"])["review_status"] == "approved"
    assert not (restored / "job").exists()
    with zipfile.ZipFile(io.BytesIO(copy.asset_zip(asset["sha256"]))) as archive:
        assert archive.read("input.stl") == (job / "input.stl").read_bytes()


def test_mutation_and_audit_roll_back_together_on_journal_failure(tmp_path, monkeypatch):
    import sqlite3
    import pytest

    store = OperationsStore(tmp_path)
    ticket = store.create_ticket(owner="alice", title="existing", description="before failure")
    for job_id, content in (("existing", b"old STL"), ("new", b"new STL")):
        folder = tmp_path / job_id; folder.mkdir(); (folder / "input.stl").write_bytes(content)
    asset = store.archive_asset(job_id="existing", owner="alice")
    append = store._record_event_db

    def fail_after_audit_insert(db, record):
        append(db, record)
        raise sqlite3.OperationalError("injected audit write failure")

    monkeypatch.setattr(store, "_record_event_db", fail_after_audit_insert)
    with pytest.raises(sqlite3.OperationalError):
        store.create_ticket(owner="alice", title="new", description="must roll back", audit_actor="alice")
    with pytest.raises(sqlite3.OperationalError):
        store.update_ticket(ticket["id"], version=0, status="resolved", message="must roll back", audit_actor="root")
    with pytest.raises(sqlite3.OperationalError):
        store.archive_asset(job_id="new", owner="alice", audit_actor="root")
    with pytest.raises(sqlite3.OperationalError):
        store.review_asset(asset["sha256"], version=0, review_status="approved", audit_actor="root")
    assert store.tickets()[1] == 1
    assert store.ticket(ticket["id"])["status"] == "open"
    assert store.ticket(ticket["id"])["messages"] == []
    assert store.ticket(ticket["id"])["version"] == 0
    assert store.assets()[1] == 1
    assert store.asset(asset["sha256"])["review_status"] == "candidate"
    assert store.asset(asset["sha256"])["version"] == 0
    assert store.recent_events() == []
    assert [path.name for path in store.assets_dir.iterdir()] == [asset["sha256"]]
    monkeypatch.setattr(store, "_record_event_db", append)
    saved = store.create_ticket(owner="alice", title="retry", description="one result", audit_actor="alice")
    assert store.tickets()[1] == 2
    assert store.recent_events()[0]["details"]["ticket_id"] == saved["id"]
