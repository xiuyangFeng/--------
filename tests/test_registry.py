import json
from pathlib import Path

import pytest

from wss_deploy.jobs import JobManager, run_identity
from wss_deploy.registry import ReleaseError, ReleaseRegistry


def _release(root: Path, rid: str, target="wall shear stress magnitude (Pa)"):
    path = root / rid
    path.mkdir()
    (path / "release.json").write_text(json.dumps({"release": rid, "target": target,
                                                     "source_runs": ["a"],
                                                     "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21}],
                                                     "fields": {"wss": {"units": "Pa", "location": "wall", "kind": "scalar"}}}), encoding="utf-8")
    return path


def test_registry_lists_safe_single_frame_releases_and_rejects_other_target(tmp_path):
    _release(tmp_path, "good-v1")
    _release(tmp_path, "bad", target="velocity vector at peak")
    registry = ReleaseRegistry(tmp_path)
    assert [row["id"] for row in registry.list()] == ["good-v1"]
    assert registry.describe().id == "good-v1"
    with pytest.raises(ReleaseError):
        registry.describe("bad")


def test_run_identity_is_order_independent():
    kwargs = {"input_sha256": "a" * 64,
             "release": {"id": "r", "fingerprint": "f"},
             "mapping": {"2": "out-li", "1": "out-le"},
             "parameters": {"spacing_mm": 0.5}}
    assert run_identity(**kwargs) == run_identity(**{**kwargs,
        "mapping": {"1": "out-le", "2": "out-li"},
        "parameters": {"spacing_mm": 0.5}})


def test_upload_binds_release_without_loading_it(tmp_path):
    _release(tmp_path, "r1")
    loaded = []

    class Registry(ReleaseRegistry):
        def load(self, *args, **kwargs):
            loaded.append(args)
            return object()

    registry = Registry(tmp_path)
    manager = JobManager(tmp_path / "jobs", registry=registry,
                         stage_a_fn=lambda *args, **kwargs: {}, stage_b_fn=lambda *args, **kwargs: {})
    job = manager.create("owner", content=b"solid", filename="case.stl", release_id="r1")
    assert job["model_release"]["id"] == "r1"
    assert loaded == []


def test_rerun_checks_owner_and_version_and_keeps_source_result(tmp_path):
    _release(tmp_path, "r1")
    _release(tmp_path, "r2")
    class Registry(ReleaseRegistry):
        def load(self, *args, **kwargs):
            return object()
    def stage_a(path, job_dir, **kwargs):
        (Path(job_dir) / "centerline").mkdir(exist_ok=True)
        return {"stage": "A", "input_sha256": "a" * 64,
                "input_check": {"status": "pass", "ok": True, "orientation_source": "unknown_stl"},
                "centerline": {"hard_pass": True},
                "proposal": {"auto_ok": False, "mapping": {"1": "out-le", "2": "out-li", "3": "out-re", "4": "out-ri"}},
                "timing_s": {}}
    def stage_b(job_dir, mapping, release, **kwargs):
        (Path(job_dir) / "summary.json").write_text('{"old": true}')
        return {"release": "r1", "timing_s": {}, "run_identity": "identity"}
    manager = JobManager(tmp_path / "jobs", registry=Registry(tmp_path), stage_a_fn=stage_a,
                         stage_b_fn=stage_b, mapping_validator=lambda *_: [])
    source = manager.create("owner", content=b"stl", filename="case.stl")
    manager.run_next(); current = manager.get(source["id"], "owner")
    manager.confirm(source["id"], "owner", {"version": current["version"], "stage": "A",
        "mapping": current["a"]["proposal"]["mapping"], "acknowledged": True})
    manager.run_next(); done = manager.get(source["id"], "owner")
    with pytest.raises(Exception):
        manager.rerun(source["id"], "other", {"version": done["version"], "release_id": "r2"})
    with pytest.raises(Exception):
        manager.rerun(source["id"], "owner", {"version": done["version"] - 1, "release_id": "r2"})
    rerun = manager.rerun(source["id"], "owner", {"version": done["version"], "release_id": "r2"})
    assert rerun["source_job_id"] == source["id"] and rerun["model_release"]["id"] == "r2"
    assert json.loads((tmp_path / "jobs" / source["id"] / "summary.json").read_text())["old"] is True


def test_changed_release_metadata_is_rejected_after_discovery(tmp_path):
    path = _release(tmp_path, "r1")
    registry = ReleaseRegistry(tmp_path)
    original = registry.describe("r1").fingerprint
    payload = json.loads((path / "release.json").read_text())
    payload["git_commit"] = "changed"
    (path / "release.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReleaseError):
        registry.describe("r1")
    # A fresh process/registry discovers the changed identity separately.
    fresh = ReleaseRegistry(tmp_path)
    assert fresh.describe("r1").fingerprint != original


def test_model_cache_isolated_by_device_and_seed_subset(tmp_path, monkeypatch):
    _release(tmp_path, "r1")
    monkeypatch.setattr("wss_deploy.registry._verify_package", lambda record: None)
    calls = []
    def loader(path, *, device="auto", seed_count=None):
        obj = type("Loaded", (), {})()
        obj.name = path.name
        calls.append((device, seed_count))
        return obj
    registry = ReleaseRegistry(tmp_path, loader=loader)
    first = registry.load("r1", device="cpu", seed_count=1)
    assert registry.load("r1", device="cpu", seed_count=1) is first
    assert registry.load("r1", device="cpu", seed_count=3) is not first
    assert registry.load("r1", device="cuda", seed_count=1) is not first
    assert calls == [("cpu", 1), ("cpu", 3), ("cuda", 1)]
