"""v0.15.7: one upload, several releases — companion tasks cloned once the outlets are confirmed."""
import json
from pathlib import Path

import pytest

from wss_deploy import centerline as CL
from wss_deploy.jobs import JobError, JobManager
from wss_deploy.registry import ReleaseError, ReleaseRegistry

MAPPING = {"1": "out-le", "2": "out-li", "3": "out-re", "4": "out-ri"}


def _release(root: Path, rid: str):
    path = root / rid
    path.mkdir()
    (path / "release.json").write_text(json.dumps({"release": rid, "target": "wall shear stress magnitude (Pa)",
                                                     "source_runs": ["a"],
                                                     "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21}],
                                                     "fields": {"wss": {"units": "Pa", "location": "wall", "kind": "scalar"}}}),
                                          encoding="utf-8")
    return path


class _Registry(ReleaseRegistry):
    def load(self, *args, **kwargs):
        return object()


def _manager(tmp_path, *, releases=("m1", "vol", "x5d"), auto=False):
    for rid in releases:
        _release(tmp_path, rid)
    seen = []

    def stage_a(path, job_dir, **kwargs):
        (Path(job_dir) / "centerline").mkdir(exist_ok=True)
        return {"stage": "A", "created_at": "2026-09-27T12:00:00+08:00", "input_sha256": "a" * 64,
                "input_check": {"status": "pass", "ok": True, "orientation_source": "unknown_stl"},
                "centerline": {"hard_pass": True},
                "proposal": {"auto_ok": auto, "confidence": 0.99 if auto else 0.5, "mapping": dict(MAPPING), "flags": []},
                "timing_s": {}}

    def stage_b(job_dir, mapping, release, **kwargs):
        seen.append(Path(job_dir).name)
        (Path(job_dir) / "summary.json").write_text("{}")
        return {"timing_s": {}, "run_identity": "identity"}

    manager = JobManager(tmp_path / "jobs", registry=_Registry(tmp_path), stage_a_fn=stage_a, stage_b_fn=stage_b,
                         mapping_validator=lambda *_: [])
    return manager, seen


def _confirm(manager, job_id, **extra):
    current = manager.get(job_id, "owner")
    return manager.confirm(job_id, "owner", {"version": current["version"], "stage": "A", "mapping": dict(MAPPING),
                                             "acknowledged": True, **extra})


def test_manual_confirmation_spawns_the_companion_once_and_both_tasks_finish(tmp_path):
    manager, seen = _manager(tmp_path)
    job = manager.create("owner", content=b"stl", filename="LV.stl", case_id="LV", patient_id="P1",
                         release_id="m1", companion_release_ids="vol")
    assert job["companions"] == [{"release_id": "vol", "job_id": None}]
    assert manager.run_next()                                     # stage A → awaiting confirmation
    assert manager.get(job["id"], "owner")["status"] == "awaiting_confirmation"
    assert len(manager.list("owner")) == 1                        # nothing spawned before the outlets are confirmed
    _confirm(manager, job["id"])
    primary = manager.get(job["id"], "owner")
    companion_id = primary["companions"][0]["job_id"]
    companion = manager.get(companion_id, "owner")
    assert companion["model_release"]["id"] == "vol" and companion["companion_of"] == job["id"]
    assert companion["source_job_id"] == job["id"] and companion["mapping"] == MAPPING
    assert companion["case_id"] == "LV" and companion["patient_id"] == "P1"
    assert companion["status"] == "queued" and companion["stage"] == "B"
    assert "companion_created" in [e["action"] for e in primary["events"]]
    # the companion event does not bump the primary's version, so its queued stage-B item stays valid
    assert manager.run_next() and manager.run_next()
    assert manager.get(job["id"], "owner")["status"] == "done" and manager.get(companion_id, "owner")["status"] == "done"
    assert sorted(seen) == sorted([job["id"], companion_id])
    # a later mapping override reruns the primary but never spawns a second companion
    done = manager.get(job["id"], "owner")
    manager.confirm(job["id"], "owner", {"version": done["version"], "stage": done["stage"], "acknowledged": True, "override": True,
                                         "mapping": {"1": "out-li", "2": "out-le", "3": "out-re", "4": "out-ri"}})
    assert len(manager.list("owner")) == 2
    assert manager.get(job["id"], "owner")["companions"][0]["job_id"] == companion_id
    # list snapshots carry the links for the case cards
    rows = {row["id"]: row for row in manager.list("owner")}
    assert rows[companion_id]["companion_of"] == job["id"] and rows[job["id"]]["companions"][0]["job_id"] == companion_id


def test_automatic_confirmation_spawns_the_companion_with_its_own_gate(tmp_path, monkeypatch):
    manager, _ = _manager(tmp_path, auto=True)
    manager._auto_outlet_gate = lambda a: True
    manager._confidence_gate = lambda a, release=None: {"passed": True, "reasons": []}
    monkeypatch.setattr(CL, "evaluate_confidence_gate", lambda *a, **k: {"passed": False, "reasons": ["体场发布包未校准"]})
    job = manager.create("owner", content=b"stl", filename="LV.stl", release_id="m1", companion_release_ids=["vol"])
    assert manager.run_next()
    primary = manager.get(job["id"], "owner")
    assert primary["status"] == "queued" and primary["stage"] == "B"
    assert primary["mapping_history"][-1]["source"] == "automatic_high_confidence"
    companion = manager.get(primary["companions"][0]["job_id"], "owner")
    # the companion re-evaluates the automatic gate for its own release: here it fails, so it asks for review
    assert companion["status"] == "awaiting_confirmation" and companion["companion_of"] == job["id"]
    assert manager.run_next()
    assert manager.get(job["id"], "owner")["status"] == "done"


def test_companion_list_is_validated_at_upload(tmp_path):
    manager, _ = _manager(tmp_path)
    with pytest.raises(JobError):
        manager.create("owner", content=b"stl", filename="a.stl", release_id="m1", companion_release_ids="m1")
    with pytest.raises(JobError):
        manager.create("owner", content=b"stl", filename="a.stl", release_id="m1", companion_release_ids="vol,vol")
    with pytest.raises(JobError):
        manager.create("owner", content=b"stl", filename="a.stl", release_id="m1", companion_release_ids=["vol", "x5d", "m1"])
    with pytest.raises(ReleaseError):
        manager.create("owner", content=b"stl", filename="a.stl", release_id="m1", companion_release_ids="missing")
    assert manager.list("owner") == []
    plain = manager.create("owner", content=b"stl", filename="a.stl", release_id="m1", companion_release_ids="")
    assert "companions" not in plain or not plain["companions"]


def test_a_companion_that_cannot_be_created_is_recorded_and_never_blocks_the_primary(tmp_path):
    manager, _ = _manager(tmp_path)
    job = manager.create("owner", content=b"stl", filename="LV.stl", release_id="m1", companion_release_ids="vol")
    manager.run_next()
    (tmp_path / "vol" / "release.json").unlink()                 # the package disappears before confirmation
    manager.registry = _Registry(tmp_path)
    _confirm(manager, job["id"])
    primary = manager.get(job["id"], "owner")
    assert primary["companions"][0]["job_id"] is None and primary["companions"][0]["error"]
    assert "companion_failed" in [e["action"] for e in primary["events"]]
    assert manager.run_next() and manager.get(job["id"], "owner")["status"] == "done"
    assert len(manager.list("owner")) == 1


def test_batch_upload_passes_companions_to_every_item(tmp_path):
    manager, _ = _manager(tmp_path)
    result = manager.create_batch("owner", [{"content": b"a", "filename": "a.stl"}, {"content": b"b", "filename": "b.stl"}],
                                  release_id="m1", companion_release_ids="vol")
    ids = [row["job"]["id"] for row in result["results"] if row.get("job")]
    assert len(ids) == 2
    assert all(manager.get(i, "owner")["companions"] == [{"release_id": "vol", "job_id": None}] for i in ids)
