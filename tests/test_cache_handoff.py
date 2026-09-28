"""v0.15.11 (P0-2): a companion cloned while its parent's precompute runs still inherits the committed geometry cache."""
from __future__ import annotations

import json
import os
import stat
import threading
import time
from pathlib import Path

import numpy as np

from wss_deploy import geometry_cache as GC
from wss_deploy.cache_handoff import copy_committed
from wss_deploy.jobs import JobManager
from wss_deploy.registry import ReleaseRegistry

MAPPING = {"1": "out-le", "2": "out-li", "3": "out-re", "4": "out-ri"}
KEY = "0123456789abcdef" * 4


def _entry(cache_dir: Path, kind: str = "resample", key: str = KEY, value: float = 1.0) -> Path:
    cache = GC.GeometryCache(cache_dir.parent, active=True)
    assert cache.save(kind, key, {"points": np.full((4, 3), value)})
    return cache.path(kind, key)


def _jobs(tmp_path):
    root = tmp_path / "jobs"
    for name in ("src", "dst"):
        (root / name).mkdir(parents=True)
    return root


# ------------------------------------------------------------------------------------------ copy_committed
def test_copies_committed_entries_as_private_files_that_the_cache_reads(tmp_path):
    root = _jobs(tmp_path)
    source = _entry(root / "src" / GC.CACHE_DIRNAME)
    (root / "src" / GC.CACHE_DIRNAME / ".resample-abc.npz.tmp").write_bytes(b"partial")   # an uncommitted write
    (root / "src" / GC.CACHE_DIRNAME / "preview-0123.json").write_text("{}")             # not a geometry entry
    (root / "src" / GC.CACHE_DIRNAME / ("nested-" + "a" * 40 + ".npz")).mkdir()            # not a regular file
    assert copy_committed(root, "src", "dst") == 1
    copied = root / "dst" / GC.CACHE_DIRNAME / source.name
    assert sorted(p.name for p in copied.parent.iterdir()) == [source.name]
    assert copied.read_bytes() == source.read_bytes() and not os.path.samefile(copied, source)
    assert stat.S_IMODE(copied.stat().st_mode) == 0o600
    hit = GC.GeometryCache(root / "dst", active=True).load("resample", KEY)
    assert hit is not None and np.array_equal(hit["points"], np.full((4, 3), 1.0))


def test_never_replaces_an_entry_the_target_already_has(tmp_path):
    root = _jobs(tmp_path)
    _entry(root / "src" / GC.CACHE_DIRNAME, value=1.0)
    mine = _entry(root / "dst" / GC.CACHE_DIRNAME, value=2.0)
    before = mine.read_bytes()
    assert copy_committed(root, "src", "dst") == 0
    assert mine.read_bytes() == before


def test_symlinks_and_hard_links_are_not_followed(tmp_path):
    root = _jobs(tmp_path)
    outside = _entry(tmp_path / "elsewhere" / GC.CACHE_DIRNAME)
    source_dir = root / "src" / GC.CACHE_DIRNAME
    source_dir.mkdir()
    (source_dir / outside.name).symlink_to(outside)                         # entry that points outside the job
    os.link(outside, source_dir / ("mesh-" + "b" * 40 + ".npz"))             # entry shared with another file
    assert copy_committed(root, "src", "dst") == 0
    assert not list((root / "dst" / GC.CACHE_DIRNAME).iterdir())
    # a cache directory that is itself a link is refused as a whole
    (root / "other").mkdir()
    (root / "other" / GC.CACHE_DIRNAME).symlink_to(outside.parent, target_is_directory=True)
    assert copy_committed(root, "other", "dst") == 0


def test_invalid_ids_disabled_cache_and_revoked_validity_copy_nothing(tmp_path, monkeypatch):
    root = _jobs(tmp_path)
    _entry(root / "src" / GC.CACHE_DIRNAME)
    assert copy_committed(root, "src", "src") == 0
    assert copy_committed(root, "../src", "dst") == 0
    assert copy_committed(root, "src", "missing") == 0
    assert copy_committed(root, "src", "dst", valid=lambda: False) == 0
    monkeypatch.setenv("WSS_DEPLOY_GEOMETRY_CACHE", "0")
    assert copy_committed(root, "src", "dst") == 0
    assert not (root / "dst" / GC.CACHE_DIRNAME).exists()


def test_validity_revoked_mid_copy_publishes_nothing(tmp_path):
    root = _jobs(tmp_path)
    _entry(root / "src" / GC.CACHE_DIRNAME)
    calls = []

    def valid():
        calls.append(1)
        return len(calls) < 3                                               # revoked while the file is copied

    assert copy_committed(root, "src", "dst", valid=valid) == 0
    assert not list((root / "dst" / GC.CACHE_DIRNAME).iterdir())             # no entry and no temporary left


# ------------------------------------------------------------------------------------------ job manager timing
class _Registry(ReleaseRegistry):
    def load(self, *args, **kwargs):
        return object()


def _release(root: Path, rid: str) -> None:
    path = root / rid
    path.mkdir()
    (path / "release.json").write_text(json.dumps({"release": rid, "target": "wall shear stress magnitude (Pa)",
                                                     "source_runs": ["a"],
                                                     "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21}],
                                                     "fields": {"wss": {"units": "Pa", "location": "wall", "kind": "scalar"}}}),
                                          encoding="utf-8")


def _stage_a(path, job_dir, **kwargs):
    (Path(job_dir) / "centerline").mkdir(exist_ok=True)
    return {"stage": "A", "created_at": "2026-09-28T12:00:00+08:00", "input_sha256": "a" * 64,
            "input_check": {"status": "pass", "ok": True, "orientation_source": "unknown_stl"},
            "centerline": {"hard_pass": True},
            "proposal": {"auto_ok": False, "confidence": 0.5, "mapping": dict(MAPPING), "flags": []},
            "timing_s": {}}


def test_companion_cloned_during_the_parent_precompute_inherits_its_entries_at_stage_b(tmp_path):
    for rid in ("m1", "vol"):
        _release(tmp_path, rid)
    started, finish = threading.Event(), threading.Event()
    seen = {}

    def precompute(job_dir, job, *, release, cancel_event=None):
        started.set()
        finish.wait(10)
        _entry(Path(job_dir) / GC.CACHE_DIRNAME)                              # committed as the precompute ends
        return {"key": "k"}

    def stage_b(job_dir, mapping, release, **kwargs):
        seen[Path(job_dir).name] = sorted(p.name for p in (Path(job_dir) / GC.CACHE_DIRNAME).glob("*.npz"))
        (Path(job_dir) / "summary.json").write_text("{}")
        return {"timing_s": {}, "run_identity": "identity"}

    manager = JobManager(tmp_path / "jobs", registry=_Registry(tmp_path), stage_a_fn=_stage_a, stage_b_fn=stage_b,
                         mapping_validator=lambda *_: [], precompute_fn=precompute)
    try:
        job = manager.create("owner", content=b"stl", filename="LV.stl", case_id="LV", release_id="m1",
                             companion_release_ids="vol")
        assert manager.run_next(timeout=0.5)
        assert started.wait(5)
        current = manager.get(job["id"], "owner")
        manager.confirm(job["id"], "owner", {"version": current["version"], "stage": "A", "mapping": dict(MAPPING),
                                             "acknowledged": True})
        companion_id = manager.get(job["id"], "owner")["companions"][0]["job_id"]
        # cloned while the parent's precompute was running: nothing could be copied yet
        assert not (tmp_path / "jobs" / companion_id / GC.CACHE_DIRNAME).exists()
        # the companion reaches stage B while the precompute is still running: it waits, then inherits
        waiter = threading.Thread(target=manager._inherit_geometry_cache, args=(manager.jobs[companion_id],))
        waiter.start()
        time.sleep(0.5)
        assert waiter.is_alive() and "等待共享几何预计算" in manager.get(companion_id, "owner")["detail"]
        finish.set()
        waiter.join(10)
        assert not waiter.is_alive()
        entry = "resample-" + KEY[:40] + ".npz"
        assert (tmp_path / "jobs" / companion_id / GC.CACHE_DIRNAME / entry).is_file()
        assert manager.run_next(timeout=1) and manager.run_next(timeout=1)
        assert seen[job["id"]] == [entry] and seen[companion_id] == [entry]
        assert manager.get(companion_id, "owner")["status"] == "done"
    finally:
        finish.set()
        manager.close()


def test_stage_b_of_a_plain_job_or_rerun_after_the_precompute_needs_no_wait(tmp_path):
    for rid in ("m1", "vol"):
        _release(tmp_path, rid)
    manager = JobManager(tmp_path / "jobs", registry=_Registry(tmp_path), stage_a_fn=_stage_a,
                         stage_b_fn=lambda job_dir, *a, **k: {"timing_s": {}, "run_identity": "identity"},
                         mapping_validator=lambda *_: [])
    try:
        job = manager.create("owner", content=b"stl", filename="LV.stl", case_id="LV", release_id="m1")
        start = time.monotonic()
        manager._inherit_geometry_cache(manager.jobs[job["id"]])                # no source: returns at once
        assert time.monotonic() - start < 0.2
        assert not (tmp_path / "jobs" / job["id"] / GC.CACHE_DIRNAME).exists()
    finally:
        manager.close()
