"""Light job snapshots, the separate geometry view and live event subscriptions."""
from __future__ import annotations

import queue

from wss_deploy.jobs import JobManager


def _stage_a_stub(*_args, **_kwargs):
    return {
        "stage": "A", "created_at": "2026-09-20 10:00:00", "stl": "input.stl", "input_sha256": "a" * 64,
        "input_check": {"ok": True, "status": "pass", "clean_stl": "input_clean_mm.stl", "orientation_source": "unknown_stl"},
        "centerline": {"hard_pass": True, "openings": [{"opening_id": 0, "radius_mm": 9.0, "role": "inlet"}]},
        "preview": {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]] * 2000, "faces": [[0, 1, 2]] * 2000, "display_only": True},
        "proposal": {"auto_ok": True, "confirmation_required": True, "confidence": 0.5, "mapping": {"3": "out-le", "4": "out-li", "5": "out-re", "6": "out-ri"},
                     "endpoints": [{"segment_id": 3, "kind": "outlet", "auto_name": "out-le"}],
                     "preview_polylines": [{"segment_id": 0, "xyz": [[0, 0, 0], [1, 1, 1]] * 500}],
                     "flags": []},
        "timing_s": {"ingest": 0.1, "centerline": 1.0},
    }


def _manager(tmp_path):
    manager = JobManager(tmp_path, stage_a_fn=_stage_a_stub, stage_b_fn=lambda *_a, **_k: {},
                         mapping_validator=lambda *_a, **_k: [])
    return manager


def test_snapshot_is_light_and_geometry_is_served_separately(tmp_path):
    manager = _manager(tmp_path)
    job = manager.create("owner", content=b"solid", filename="scan.stl")
    assert manager.run_next(timeout=0.5)
    snapshot = manager.get(job["id"], "owner")
    assert snapshot["status"] == "awaiting_confirmation"
    assert "preview" not in snapshot["a"]
    assert "preview_polylines" not in snapshot["a"]["proposal"]
    assert snapshot["a"]["proposal"]["endpoints"]  # small fields stay
    assert snapshot["a"]["geometry_endpoint"].endswith("/geometry")
    geometry = manager.geometry(job["id"], "owner")
    assert geometry["available"] and len(geometry["preview"]["faces"]) == 2000
    assert len(geometry["preview_polylines"]) == 1 and geometry["version"] == snapshot["version"]


def test_live_events_follow_the_job_and_stop_when_final(tmp_path):
    manager = _manager(tmp_path)
    job = manager.create("owner", content=b"solid", filename="scan.stl")
    listener, snapshot = manager.subscribe(job["id"], "owner")
    assert snapshot["status"] == "queued" and snapshot["final"] is False
    assert manager.run_next(timeout=0.5)
    actions = []
    while True:
        try:
            actions.append(listener.get_nowait())
        except queue.Empty:
            break
    assert [event["action"] for event in actions][:1] == ["started"]
    assert actions[-1]["status"] == "awaiting_confirmation" and actions[-1]["final"] is False
    manager.cancel(job["id"], "owner", {"version": manager.get(job["id"], "owner")["version"]})
    final = listener.get_nowait()
    assert final["action"] == "cancel_requested" and final["final"] is True and final["status"] == "cancelled"
    manager.unsubscribe(job["id"], listener)
    assert job["id"] not in manager._subscribers
