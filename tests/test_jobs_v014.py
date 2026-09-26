"""v0.14 job manager: slim job.json v2 (J1), background geometry precompute (J2), offline managers (J3),
health (J5), typed errors / CPU fallback / restart re-queue (J7), audit lines (J8) and ordered migrations (J9)."""
from __future__ import annotations

import json
import logging
import os
import threading
import time

import pytest

from wss_deploy import jobs as J
from wss_deploy.errors import InputGeometryError, ResourceError, ToolchainError
from wss_deploy.jobs import JOB_SCHEMA_VERSION, JobError, JobManager, slim_stage_a
from tests._c_helpers import FakeRelease, MAPPING, finished, stage_b_stub

PREVIEW = {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]] * 500, "faces": [[0, 1, 2]] * 500, "display_only": True}
POLYLINES = [{"segment_id": 0, "xyz": [[0, 0, 0], [1, 1, 1]] * 100}]


def stage_a_full(*_args, **_kwargs):
    return {"stage": "A", "created_at": "2026-09-24 10:00:00", "stl": "input.stl", "input_sha256": "a" * 64,
            "input_check": {"ok": True, "status": "pass", "clean_stl": "input_clean_mm.stl", "orientation_source": "unknown_stl", "faces": 1000},
            "centerline": {"hard_pass": True, "openings": [{"opening_id": 0, "radius_mm": 9.0, "role": "inlet"}]},
            "preview": PREVIEW,
            "proposal": {"auto_ok": True, "confirmation_required": True, "confidence": 0.5, "mapping": dict(MAPPING),
                         "endpoints": [{"segment_id": 3}], "preview_polylines": POLYLINES, "flags": []},
            "timing_s": {"ingest": 0.1, "centerline": 1.0}}


def make(tmp_path, **kwargs):
    kwargs.setdefault("stage_a_fn", stage_a_full)
    kwargs.setdefault("stage_b_fn", stage_b_stub)
    return JobManager(tmp_path, release=FakeRelease(), mapping_validator=lambda *_a, **_k: [], **kwargs)


def awaiting(mgr, owner="owner"):
    job = mgr.create(owner, content=b"solid", filename="scan.stl")
    assert mgr.run_next(timeout=0.5)
    snap = mgr.get(job["id"], owner)
    assert snap["status"] == "awaiting_confirmation"
    return snap


# ------------------------------------------------------------------------------------------ J1 slim job.json
def test_v2_record_has_no_display_geometry_but_geometry_and_stage_a_still_serve_it(tmp_path):
    mgr = make(tmp_path)
    snap = awaiting(mgr)
    raw = (tmp_path / snap["id"] / "job.json").read_text(encoding="utf-8")
    saved = json.loads(raw)
    assert saved["schema_version"] == JOB_SCHEMA_VERSION
    assert "preview" not in saved["a"] and "preview_polylines" not in saved["a"]["proposal"]
    assert saved["a"]["proposal"]["endpoints"] and saved["a"]["input_check"]["faces"] == 1000   # small fields stay
    assert "\n" not in raw                                   # compact JSON
    assert "preview" not in mgr.jobs[snap["id"]]["a"]        # and not in memory either
    stage_file = json.loads((tmp_path / snap["id"] / "stage_a.json").read_text(encoding="utf-8"))
    assert stage_file["preview"] == PREVIEW
    geometry = mgr.geometry(snap["id"], "owner")
    assert geometry["available"] and geometry["preview"] == PREVIEW and geometry["preview_polylines"] == POLYLINES
    full = mgr.stage_a(snap["id"])
    assert full["preview"] == PREVIEW and full["proposal"]["preview_polylines"] == POLYLINES
    assert "preview" not in mgr.jobs[snap["id"]]["a"]        # the accessor returns a copy


def _write_v1(tmp_path, job_id="20260101_000000_v1", *, stage_file_preview=PREVIEW, status="done"):
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    a = stage_a_full()
    record = {"id": job_id, "owner": "owner", "case_id": "c", "status": status, "stage": "B", "version": 5, "events": [],
              "created_at": "2026-01-01T00:00:00+08:00", "created_ts": 1.0, "params": {"units": "mm"}, "a": a,
              "mapping": dict(MAPPING), "model_release": {"id": "REL_A", "fingerprint": "f" * 64}}
    (job_dir / "job.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    stored = dict(a)
    if stage_file_preview is None:
        stored.pop("preview")
    else:
        stored["preview"] = stage_file_preview
    (job_dir / "stage_a.json").write_text(json.dumps(stored), encoding="utf-8")
    return job_dir


def test_v1_record_with_embedded_preview_loads_slims_in_memory_and_migrates_on_next_save(tmp_path):
    job_dir = _write_v1(tmp_path)
    before = (job_dir / "job.json").read_bytes()
    mgr = make(tmp_path)
    job = mgr.jobs[job_dir.name]
    assert job["schema_version"] == JOB_SCHEMA_VERSION and "preview" not in job["a"]
    assert "preview_polylines" not in job["a"]["proposal"]
    assert (job_dir / "job.json").read_bytes() == before           # lazy: loading alone writes nothing
    assert mgr.geometry(job_dir.name, "owner")["preview"] == PREVIEW
    listed = mgr.list("owner")
    assert listed[0]["id"] == job_dir.name and mgr.get(job_dir.name, "owner")["status"] == "done"
    mgr.update_metadata(job_dir.name, "owner", {"notes": "x"})     # any save migrates the file
    saved = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == JOB_SCHEMA_VERSION and "preview" not in saved["a"] and saved["notes"] == "x"
    assert len((job_dir / "job.json").read_bytes()) < len(before) / 3


def test_v1_preview_that_differs_from_stage_a_json_stays_embedded(tmp_path):
    job_dir = _write_v1(tmp_path, stage_file_preview=None)
    mgr = make(tmp_path)
    assert mgr.jobs[job_dir.name]["a"]["preview"] == PREVIEW      # nothing is lost
    assert "preview_polylines" not in mgr.jobs[job_dir.name]["a"]["proposal"]
    assert mgr.geometry(job_dir.name, "owner")["preview"] == PREVIEW


def test_oldest_shape_with_stage_a_key_still_migrates(tmp_path):
    job_dir = tmp_path / "20260917_232710_old"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({"id": job_dir.name, "case_id": "c", "status": "done", "mapping": dict(MAPPING),
                                                  "stage_a": {"input_sha256": "b" * 64, "input_check": {"ok": True}}}), encoding="utf-8")
    mgr = make(tmp_path, legacy_owner="legacy")
    job = mgr.jobs[job_dir.name]
    assert job["a"]["input_sha256"] == "b" * 64 and "stage_a" not in job and job["owner"] == "legacy"
    assert job["params"]["units"] == "mm" and job["compute"]["device"] == "auto" and job["tags"] == []
    assert job["schema_version"] == JOB_SCHEMA_VERSION and job["version"] == 1


def test_migration_table_is_ordered_and_skips_current_records(tmp_path):
    assert [version for version, _ in J.MIGRATIONS] == ["wss-deploy.job/v1", "wss-deploy.job/v2"]
    path = tmp_path / "x" / "job.json"
    path.parent.mkdir()
    path.write_text("{}")
    assert J.migrate_record({"schema_version": JOB_SCHEMA_VERSION}, path=path) == []
    assert J.migrate_record({}, path=path) == ["wss-deploy.job/v1", "wss-deploy.job/v2"]


def test_progress_events_skip_fsync_and_state_changes_keep_it(tmp_path, monkeypatch):
    mgr = make(tmp_path)
    job = mgr.create("owner", content=b"solid", filename="scan.stl")
    calls = []
    real = os.fsync
    monkeypatch.setattr(J.os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    record = mgr.jobs[job["id"]]
    with mgr.lock:
        mgr._event(record, "progress", version=False, phase="几何特征")
    assert calls == []
    saved = json.loads((tmp_path / job["id"] / "job.json").read_text(encoding="utf-8"))
    assert saved["events"][-1]["phase"] == "几何特征"                 # still written (atomically)
    with mgr.lock:
        mgr._event(record, "metadata_updated")
    assert len(calls) == 2                                          # file + directory


def test_rerun_copies_full_stage_a_and_geometry_cache(tmp_path):
    mgr = make(tmp_path)
    snap = awaiting(mgr)
    (tmp_path / snap["id"] / J.GEOMETRY_CACHE_DIR).mkdir()
    (tmp_path / snap["id"] / J.GEOMETRY_CACHE_DIR / "key.json").write_text("{}")
    confirmed = mgr.confirm(snap["id"], "owner", {"version": snap["version"], "acknowledged": True, "mapping": dict(MAPPING)})
    assert mgr.run_next(timeout=0.5)
    done = mgr.get(snap["id"], "owner")
    assert done["status"] == "done", confirmed
    rerun = mgr.rerun(snap["id"], "owner", {"version": done["version"]})
    new_dir = tmp_path / rerun["id"]
    assert json.loads((new_dir / "stage_a.json").read_text(encoding="utf-8"))["preview"] == PREVIEW
    assert (new_dir / J.GEOMETRY_CACHE_DIR / "key.json").is_file()
    assert "preview" not in json.loads((new_dir / "job.json").read_text(encoding="utf-8"))["a"]


# ------------------------------------------------------------------------------------------ J2 precompute
def test_precompute_runs_during_confirmation_and_stage_b_waits_for_it(tmp_path):
    started, release_it, seen = threading.Event(), threading.Event(), {}

    def precompute(job_dir, job, *, release, cancel_event=None):
        seen.update(job_dir=job_dir, job_id=job["id"], release=release, cancel=cancel_event)
        started.set()
        release_it.wait(5)
        return {"key": "k1", "timing_s": {"smooth_resample": 1.0, "features": 0.5}}

    order = []

    def stage_b(job_dir, mapping, release, **kwargs):
        order.append("B")
        return stage_b_stub(job_dir, mapping, release, **kwargs)

    mgr = make(tmp_path, precompute_fn=precompute, stage_b_fn=stage_b)
    try:
        snap = awaiting(mgr)
        assert started.wait(5) and seen["job_id"] == snap["id"] and isinstance(seen["cancel"], threading.Event)
        confirmed = mgr.confirm(snap["id"], "owner", {"version": snap["version"], "acknowledged": True, "mapping": dict(MAPPING)})
        assert confirmed["version"] == snap["version"] + 1        # the precompute never bumped the version
        worker = threading.Thread(target=mgr.run_next, kwargs={"timeout": 1})
        worker.start()
        time.sleep(0.5)
        assert order == []                                          # stage B waits for the running precompute
        release_it.set()
        worker.join(10)
        done = mgr.get(snap["id"], "owner")
        assert order == ["B"] and done["status"] == "done"
        record = mgr.jobs[snap["id"]]
        assert record["precompute"]["status"] == "done" and record["precompute"]["key"] == "k1"
        assert "precompute" in record["summary"]["timing_s"] and "precompute_s" in done["timing"]
        assert any(event["action"] == "precompute_done" for event in record["events"])
    finally:
        release_it.set()
        mgr.close()


def test_precompute_is_skipped_when_stage_b_starts_first_and_cancelled_by_cancel(tmp_path):
    calls = []

    def precompute(job_dir, job, *, release, cancel_event=None):
        calls.append(job["id"])
        return {}

    mgr = make(tmp_path, precompute_fn=precompute)
    try:
        mgr._stage_b_lock.acquire()                                  # a stage B elsewhere: the precompute waits
        snap = awaiting(mgr)
        other = awaiting(mgr)
        time.sleep(0.6)
        assert calls == [] and mgr._pc_state.get(snap["id"]) == "pending"
        mgr.confirm(snap["id"], "owner", {"version": snap["version"], "acknowledged": True, "mapping": dict(MAPPING)})
        mgr._precompute_barrier(mgr.jobs[snap["id"]], "B")            # stage B reached its job first
        assert mgr._pc_state.get(snap["id"]) == "skipped"
        mgr.cancel(other["id"], "owner", {"version": other["version"]})
        assert mgr._pc_state.get(other["id"]) == "skipped"
        mgr._stage_b_lock.release()
        assert mgr.run_next(timeout=0.5) and mgr.get(snap["id"], "owner")["status"] == "done"
        deadline = time.time() + 3
        while mgr._pc_state and time.time() < deadline:
            time.sleep(0.05)
        assert calls == [] and not mgr._pc_state                      # neither precompute ever ran
    finally:
        mgr.close()


def test_precompute_failure_never_fails_the_job(tmp_path):
    def broken(*_a, **_k):
        raise RuntimeError("cache exploded")

    mgr = make(tmp_path, precompute_fn=broken)
    try:
        snap = awaiting(mgr)
        deadline = time.time() + 5
        while mgr.jobs[snap["id"]].get("precompute", {}).get("status") not in {"failed"} and time.time() < deadline:
            time.sleep(0.05)
        assert mgr.jobs[snap["id"]]["precompute"]["status"] == "failed"
        latest = mgr.get(snap["id"], "owner")
        assert latest["status"] == "awaiting_confirmation" and latest["version"] == snap["version"]
        mgr.confirm(snap["id"], "owner", {"version": latest["version"], "acknowledged": True, "mapping": dict(MAPPING)})
        assert mgr.run_next(timeout=0.5) and mgr.get(snap["id"], "owner")["status"] == "done"
    finally:
        mgr.close()


def test_stub_managers_do_not_precompute_by_default(tmp_path):
    mgr = make(tmp_path)
    assert mgr._precompute_enabled is False
    snap = awaiting(mgr)
    assert "precompute" not in mgr.jobs[snap["id"]]


# ------------------------------------------------------------------------------------------ J3 offline manager
def test_offline_manager_marks_nothing_and_purges_nothing(tmp_path):
    job_dir = _write_v1(tmp_path, "20260101_000000_run", status="running")
    queued_dir = _write_v1(tmp_path, "20260101_000001_q", status="queued")
    leftover = tmp_path / (J.DELETING_PREFIX + "x_1")
    leftover.mkdir()
    before = {d: (d / "job.json").read_bytes() for d in (job_dir, queued_dir)}
    mgr = JobManager(tmp_path, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {}, offline=True)
    assert mgr.jobs[job_dir.name]["status"] == "running" and mgr.jobs[queued_dir.name]["status"] == "queued"
    assert mgr.tasks.empty() and leftover.is_dir()
    assert {d: (d / "job.json").read_bytes() for d in (job_dir, queued_dir)} == before
    with pytest.raises(RuntimeError):
        mgr.start()
    result = mgr.claim_owner("owner", "alice")
    assert result["claimed"] == 2
    saved = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert saved["owner"] == "alice" and saved["status"] == "running"


# ------------------------------------------------------------------------------------------ J7 errors / restart
def _failing_b(error):
    def stage_b(*_a, **_k):
        raise error
    return stage_b


def _run_to_failure(tmp_path, error):
    mgr = make(tmp_path, stage_b_fn=_failing_b(error))
    snap = awaiting(mgr)
    mgr.confirm(snap["id"], "owner", {"version": snap["version"], "acknowledged": True, "mapping": dict(MAPPING)})
    assert mgr.run_next(timeout=0.5)
    return mgr, mgr.get(snap["id"], "owner")


@pytest.mark.parametrize("error, category, retryable, message", [
    (InputGeometryError("STL 有 3 个连通片"), "input_geometry", False, "STL 有 3 个连通片"),
    (ToolchainError("中心线提取失败。", admin_detail="vmtk stderr tail"), "toolchain", False, "中心线提取失败。"),
    (ValueError("输入或中心线未通过检查"), "internal", False, "输入或中心线未通过检查"),
    (RuntimeError("boom"), "internal", True, J.GENERIC_FAILURE),
])
def test_failed_job_carries_a_typed_error_record(tmp_path, error, category, retryable, message):
    _, job = _run_to_failure(tmp_path, error)
    assert job["status"] == "failed"
    record = job["error"]
    assert record["category"] == category and record["retryable"] is retryable and record["message"] == message
    assert len(record["diagnostic_id"]) == 12
    if isinstance(error, ToolchainError):
        assert record["admin_detail"] == "vmtk stderr tail"


def test_cuda_oom_retries_stage_b_once_on_the_cpu(tmp_path):
    devices = []

    def stage_b(job_dir, mapping, release, *, device="auto", **kwargs):
        devices.append(device)
        if device != "cpu":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        (job_dir / "summary.json").write_text(json.dumps({"run_parameters": {"device": "cpu"}}), encoding="utf-8")
        (job_dir / "run_manifest.json").write_text(json.dumps({"outputs": {"summary.json": {}}}), encoding="utf-8")
        return {**stage_b_stub(job_dir, mapping, release), "device": "cpu"}

    mgr = make(tmp_path, stage_b_fn=stage_b)
    snap = awaiting(mgr)
    mgr.confirm(snap["id"], "owner", {"version": snap["version"], "acknowledged": True, "mapping": dict(MAPPING)})
    assert mgr.run_next(timeout=0.5)
    job = mgr.get(snap["id"], "owner")
    assert devices == ["auto", "cpu"] and job["status"] == "done"
    record = mgr.jobs[snap["id"]]
    assert record["run_parameters"]["device_fallback"] == "cpu" and record["summary"]["device_fallback"] == "cpu"
    event = next(event for event in record["events"] if event["action"] == "device_fallback")
    assert event["reason"] == "GPU 显存不足，已改用 CPU 重试。" and "CUDA out of memory" in event["admin_detail"]
    from wss_deploy.server import strip_admin_detail                   # what a non-admin session receives
    assert "CUDA out of memory" not in json.dumps(strip_admin_detail(mgr.get(snap["id"], "owner")), ensure_ascii=False)
    summary = json.loads((tmp_path / snap["id"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_parameters"]["device_fallback"] == "cpu"
    manifest = json.loads((tmp_path / snap["id"] / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["outputs"]["summary.json"]["present"] is True       # hashes refreshed after the note


def test_cuda_oom_on_the_cpu_fallback_is_not_retried_again(tmp_path):
    _, job = _run_to_failure(tmp_path, ResourceError("GPU 显存不足。", retry_hint="cpu"))
    # the stub raises on every call: the CPU retry fails too and the job reports the typed resource error
    assert job["status"] == "failed" and job["error"]["category"] == "resource" and job["error"]["retryable"] is True


def test_restart_requeues_queued_jobs_and_interrupts_running_ones(tmp_path):
    mgr = make(tmp_path)
    first = mgr.create("owner", content=b"one", filename="one.stl")
    second = mgr.create("owner", content=b"two", filename="two.stl")
    running = mgr.create("owner", content=b"three", filename="three.stl")
    with mgr.lock:
        record = mgr.jobs[running["id"]]
        record.update(status="running", started_ts=time.time())
        mgr._event(record, "started")
    reloaded = make(tmp_path)
    assert reloaded.jobs[running["id"]]["status"] == "interrupted"
    for job in (first, second):
        record = reloaded.jobs[job["id"]]
        assert record["status"] == "queued" and record["events"][-1]["action"] == "restart_requeued"
    queued = [reloaded.tasks.get_nowait() for _ in range(reloaded.tasks.qsize())]
    assert [item[0] for item in queued] == [first["id"], second["id"]]          # original order
    assert all(item[1] == reloaded.jobs[item[0]]["version"] for item in queued)
    for item in queued:
        reloaded.tasks.put(item)
    assert reloaded.run_next(timeout=0.5)
    assert reloaded.get(first["id"], "owner")["status"] == "awaiting_confirmation"


def test_restored_queued_job_is_still_interrupted(tmp_path):
    mgr = make(tmp_path)
    queued = mgr.create("owner", content=b"q", filename="q.stl")
    mgr.delete(queued["id"], "owner", {"version": queued["version"]})
    assert mgr.restore(queued["id"], "owner")["status"] == "interrupted"


# ------------------------------------------------------------------------------------------ J5 health
def test_health_reports_hard_checks_and_caches_gpu(tmp_path, monkeypatch):
    from wss_deploy import service as S
    calls = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)          # an empty value would skip nvidia-smi entirely
    monkeypatch.setattr(S, "_GPU_CACHE", {})
    monkeypatch.setattr(S.shutil, "which", lambda name: calls.append(name) or None)
    mgr = make(tmp_path)
    health = mgr.health()
    assert set(health["checks"]) >= {"worker", "releases", "disk", "jobs_root", "gpu"}
    assert health["checks"]["worker"]["ok"] is False and health["ok"] is False      # workers never started
    assert health["checks"]["jobs_root"]["ok"] and health["checks"]["releases"]["ok"]
    assert health["checks"]["gpu"]["hard"] is False and health["cached_at"][:4].isdigit()
    mgr.start()
    try:
        health = mgr.health()
        assert health["checks"]["worker"]["ok"] and health["checks"]["worker"]["alive"] == 2
        assert health["ok"] is health["checks"]["vmtk"]["ok"] and health["checks"]["disk"]["ok"]
        mgr.health()
        assert calls.count("nvidia-smi") == 1                                     # 30 s cache
    finally:
        mgr.close()


def test_health_fails_when_the_jobs_root_is_not_writable(tmp_path, monkeypatch):
    mgr = make(tmp_path)
    mgr.start()
    try:
        monkeypatch.setattr(J.os, "access", lambda *_a, **_k: False)
        health = mgr.health()
        assert health["checks"]["jobs_root"]["ok"] is False and health["ok"] is False
    finally:
        mgr.close()


# ------------------------------------------------------------------------------------------ J6 drain / maintenance
def test_drain_file_for_this_process_stops_new_work(tmp_path):
    mgr = make(tmp_path)
    job = mgr.create("owner", content=b"solid", filename="scan.stl")
    J.atomic_json(tmp_path / J.DRAIN_FILE, {"pid": os.getpid()})
    mgr._drain_cache = (0.0, False)
    assert mgr.run_next(timeout=0.1) is False and mgr.get(job["id"], "owner")["status"] == "queued"
    assert mgr.health()["checks"]["worker"]["draining"] is True
    (tmp_path / J.DRAIN_FILE).unlink()
    mgr._drain_cache = (0.0, False)
    assert mgr.run_next(timeout=0.5) and mgr.get(job["id"], "owner")["status"] == "awaiting_confirmation"


def test_stale_drain_file_of_another_process_is_removed_at_start(tmp_path):
    J.atomic_json(tmp_path / J.DRAIN_FILE, {"pid": 1})
    mgr = make(tmp_path)
    assert not (tmp_path / J.DRAIN_FILE).exists() and mgr._draining() is False


def test_maintenance_rebuilds_inside_the_service_and_locks_the_job(tmp_path):
    mgr = make(tmp_path)
    job = finished(mgr, case_id="A")
    seen = {}

    def rebuild(job_dir):
        with mgr.lock:
            record = mgr.jobs[job_dir.name]
        with pytest.raises(JobError) as info:
            mgr.update_metadata(job_dir.name, "owner", {"notes": "during"})
        seen["status"] = info.value.status
        data = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        data["summary"]["narrative"] = {"text": "rebuilt"}
        (job_dir / "job.json").write_text(json.dumps(data), encoding="utf-8")
        return record

    assert mgr.rebuild_analysis(job["id"], rebuild_fn=rebuild) == "rebuilt"
    assert seen["status"] == 409
    fresh = mgr.get(job["id"], "owner")
    assert fresh["summary"]["narrative"] == {"text": "rebuilt"} and fresh["version"] == job["version"] + 1
    assert mgr.jobs[job["id"]]["events"][-1]["action"] == "analysis_rebuilt"
    mgr.update_metadata(job["id"], "owner", {"notes": "after"})              # unlocked again
    assert mgr.rebuild_analysis("missing_job", rebuild_fn=rebuild) == "missing"


def test_queued_maintenance_runs_after_start_and_reports(tmp_path, monkeypatch):
    mgr = make(tmp_path)
    job = finished(mgr, case_id="A")
    import wss_deploy.rebuild_report as R
    import wss_deploy.report_freshness as F
    monkeypatch.setattr(R, "rebuild", lambda job_dir: None)
    monkeypatch.setattr(F, "stale_jobs", lambda root: [tmp_path / job["id"]])
    refreshed = []
    monkeypatch.setattr(F, "refresh", lambda job_dir, source: refreshed.append((job_dir.name, source)))
    J.atomic_json(tmp_path / J.MAINTENANCE_FILE, {"request_id": "r1", "rebuild": [job["id"], "nope"], "refresh_reports": True})
    mgr.start()
    try:
        mgr._maintenance_thread.join(10)
        result = json.loads((tmp_path / J.MAINTENANCE_RESULT).read_text(encoding="utf-8"))
        assert result["request_id"] == "r1" and result["rebuilt"] == [job["id"]] and result["refreshed"] == 1
        assert refreshed == [(job["id"], "upgrade")] and not (tmp_path / J.MAINTENANCE_FILE).exists()
    finally:
        mgr.close()


# ------------------------------------------------------------------------------------------ J8 audit
def test_state_changes_write_one_audit_line_without_raw_session_owner(tmp_path, caplog):
    mgr = make(tmp_path)
    owner = "Xy3_" + "q" * 28                                  # a 32-character session owner
    with caplog.at_level(logging.INFO, logger="wss_deploy.audit"):
        job = mgr.create(owner, content=b"solid", filename="scan.stl")
        with mgr.lock:
            mgr._event(mgr.jobs[job["id"]], "progress", version=False, phase="x")
        mgr.cancel(job["id"], owner, {"version": job["version"]})
    lines = [json.loads(r.getMessage().split(" ", 1)[1]) for r in caplog.records if r.name == "wss_deploy.audit"]
    assert [line["action"] for line in lines] == ["created", "cancel_requested"]
    assert all(owner not in json.dumps(line) for line in lines) and lines[1]["actor"].startswith("h:")
    assert lines[1]["status"] == "cancelled" and lines[1]["release"] == "REL_A"
    assert all("actor" not in event for event in mgr.jobs[job["id"]]["events"])     # never in the API record


def test_slim_stage_a_shares_nested_values():
    full = stage_a_full()
    light = slim_stage_a(full)
    assert "preview" not in light and "preview_polylines" not in light["proposal"]
    assert full["preview"] is PREVIEW and "preview_polylines" in full["proposal"]
    assert light["input_check"] is full["input_check"]


@pytest.mark.parametrize("result, status", [
    ({"ok": True, "steps": ["mesh", "resample", "point_geometry", "morphology"], "seconds": 9.0}, "done"),
    ({"ok": False, "error": "ValueError: bad mesh"}, "failed"),
    ({"ok": False, "cancelled": True, "steps": ["mesh"]}, "cancelled"),
    ({"ok": False, "skipped": "disabled"}, "skipped"),
])
def test_precompute_result_contract_maps_to_the_job_status_and_eta(tmp_path, result, status):
    mgr = make(tmp_path, precompute_fn=lambda *_a, **_k: dict(result))
    try:
        snap = awaiting(mgr)
        deadline = time.time() + 5
        while mgr.jobs[snap["id"]].get("precompute", {}).get("status") in {None, "pending", "running"} and time.time() < deadline:
            time.sleep(0.05)
        record = mgr.jobs[snap["id"]]["precompute"]
        assert record["status"] == status and "error" not in record or status == "failed"
        eta = mgr.get(snap["id"], "owner")["eta"]
        if status == "done":
            assert record["steps"][-1] == "morphology" and eta["precomputed"] == ["features", "morphology", "smooth_resample"]
        else:
            assert "precomputed" not in eta
    finally:
        mgr.close()


def test_a_job_waiting_for_the_stage_b_slot_goes_back_to_the_queue_when_a_drain_starts(tmp_path):
    mgr = make(tmp_path)
    snap = awaiting(mgr)
    mgr.confirm(snap["id"], "owner", {"version": snap["version"], "acknowledged": True, "mapping": dict(MAPPING)})
    mgr._stage_b_lock.acquire()                                   # another stage B holds the slot
    worker = threading.Thread(target=mgr.run_next, kwargs={"timeout": 1})
    worker.start()
    deadline = time.time() + 5
    while mgr.jobs[snap["id"]]["status"] != "running" and time.time() < deadline:
        time.sleep(0.02)
    assert mgr.get(snap["id"], "owner")["eta"]["waiting"] is True
    J.atomic_json(tmp_path / J.DRAIN_FILE, {"pid": os.getpid()})
    mgr._drain_cache = (0.0, False)
    mgr._stage_b_lock.release()
    worker.join(5)
    job = mgr.get(snap["id"], "owner")
    assert job["status"] == "queued" and mgr.jobs[snap["id"]]["events"][-1]["action"] == "drain_requeued"
    assert "summary" not in mgr.jobs[snap["id"]] and not mgr._stage_b_lock.locked()
    (tmp_path / J.DRAIN_FILE).unlink()
    mgr._drain_cache = (0.0, False)
    assert mgr.run_next(timeout=0.5) and mgr.get(snap["id"], "owner")["status"] == "done"


# ------------------------------------------------------------------------------------------ streamed uploads
def _binary_stl(n_faces: int) -> bytes:
    import struct
    body = b"".join(struct.pack("<12fH", *([0.0] * 12), 0) for _ in range(n_faces))
    return b"\0" * 80 + n_faces.to_bytes(4, "little") + body


@pytest.mark.parametrize("kind", ["path", "file"])
def test_streamed_upload_matches_the_bytes_path(tmp_path, kind, monkeypatch):
    import io
    monkeypatch.setattr(J, "UPLOAD_CHUNK_BYTES", 7)               # many chunks, words split across boundaries
    ascii_stl = b"solid x\n" + b"facet normal 0 0 0\n outer loop\n vertex 0 0 0\n endloop\nendfacet\n" * 5 + b"endsolid x\n"
    for payload, faces in ((_binary_stl(3), 3), (ascii_stl, 5)):
        by_bytes = make(tmp_path / f"bytes_{faces}")
        reference = by_bytes.create("owner", content=payload, filename="scan.stl")
        mgr = make(tmp_path / f"{kind}_{faces}")
        if kind == "path":
            source = tmp_path / f"upload_{faces}.stl"
            source.write_bytes(payload)
            job = mgr.create("owner", content_path=source, filename="scan.stl")
        else:
            import tempfile
            # Python 3.10's SpooledTemporaryFile has no seekable(); after writing it sits at its end and is
            # passed on without a rewind — the manager must rewind it itself.
            spool = tempfile.SpooledTemporaryFile(max_size=64)
            spool.write(payload)
            job = mgr.create("owner", content_file=spool, filename="scan.stl")
            spool.close()
        record, ref = mgr.jobs[job["id"]], by_bytes.jobs[reference["id"]]
        assert record["input_sha256"] == ref["input_sha256"] and record["input_faces"] == ref["input_faces"] == faces
        assert (mgr.root / job["id"] / "input.stl").read_bytes() == payload
        assert not any((mgr.root / J.TMP_DIR).iterdir())            # nothing left in .tmp


def test_streamed_upload_errors_leave_no_staging_file(tmp_path):
    import io
    mgr = make(tmp_path)
    empty = tmp_path / "empty.stl"
    empty.write_bytes(b"")
    with pytest.raises(JobError, match="非空"):
        mgr.create("owner", content_path=empty, filename="scan.stl")
    with pytest.raises(JobError, match="非空"):
        mgr.create("owner", content_file=io.BytesIO(b"solid"), filename="scan.txt")
    with pytest.raises(JobError, match="一种来源"):
        mgr.create("owner", content=b"solid", content_file=io.BytesIO(b"solid"), filename="scan.stl")
    with pytest.raises(JobError, match="非空"):
        mgr.create("owner", filename="scan.stl")
    with pytest.raises(JobError, match="二进制"):
        mgr.create("owner", content_file=io.StringIO("solid"), filename="scan.stl")
    first = mgr.create("owner", content_file=io.BytesIO(b"solid same"), filename="a.stl")
    with pytest.raises(JobError) as info:                          # duplicate "ask": refused after hashing
        mgr.create("owner", content_file=io.BytesIO(b"solid same"), filename="b.stl", on_duplicate="ask")
    assert info.value.payload["duplicate"] and info.value.payload["existing"][0]["job_id"] == first["id"]
    assert not any((tmp_path / J.TMP_DIR).iterdir()) and len(mgr.jobs) == 1


def test_batch_accepts_streamed_items(tmp_path):
    import io
    mgr = make(tmp_path)
    source = tmp_path / "one.stl"
    source.write_bytes(b"solid one")
    result = mgr.create_batch("owner", [{"content_path": source, "filename": "one.stl"},
                                        {"content_file": io.BytesIO(b"solid two"), "filename": "two.stl"},
                                        {"content": b"solid three", "filename": "three.stl"},
                                        {"filename": "none.stl"}])
    assert result["created_count"] == 3 and result["failed_count"] == 1
    shas = {job["input_sha256"] for job in result["jobs"]}
    assert shas == {J.input_digest(b"solid one"), J.input_digest(b"solid two"), J.input_digest(b"solid three")}


# ------------------------------------------------------------------------------------------ expected dot entries
def test_tmp_and_report_staging_dirs_are_never_jobs_or_findings(tmp_path, caplog):
    from wss_deploy import doctor as D
    from wss_deploy import report_freshness as F
    from wss_deploy import service as S
    mgr = make(tmp_path)
    job = finished(mgr, case_id="A")
    stray_tmp = tmp_path / J.TMP_DIR
    stray_tmp.mkdir(exist_ok=True)
    (stray_tmp / "job.json").write_text("{not json")                      # server temp files may look like anything
    (stray_tmp / "spool_x").write_bytes(b"x")
    stage = tmp_path / job["id"] / ".report_ui_stage_x"
    stage.mkdir()
    (stage / "report.html").write_text("<html></html>")
    (stage / "job.json").write_text("{}")
    with caplog.at_level(logging.WARNING):
        reloaded = make(tmp_path)
    assert list(reloaded.jobs) == [job["id"]] and not [r for r in caplog.records if "Cannot restore" in r.getMessage()]
    assert [row["id"] for row in reloaded.list("owner")] == [job["id"]]
    assert reloaded.cases("owner")["total"] == 1 and reloaded.trash_list("owner")["items"] == []
    assert reloaded.purge_expired() == [] and stray_tmp.is_dir() and stage.is_dir()
    assert S.pending_rebuilds(tmp_path) == [] and sum(S.queue_from_disk(tmp_path).values()) == 0
    assert [d.name for d in F.done_jobs(tmp_path)] == [job["id"]]
    # v0.15: the ``cache`` group reports .tmp by name on purpose (derived files, never jobs), so it is left out here
    rows = D.run_checks(tmp_path, skip={"torch", "vmtk", "releases", "features", "glossary", "service", "git", "cache"})
    assert not [row for row in rows if row["status"] == "fail"]
    assert all(".tmp" not in row["detail"] and ".report_ui_stage" not in row["detail"] for row in rows)
    rerun = reloaded.rerun(job["id"], "owner", {"version": reloaded.get(job["id"], "owner")["version"]})
    assert not (tmp_path / rerun["id"] / ".report_ui_stage_x").exists()      # staging is never copied into a rerun


def test_precompute_never_asks_for_loaded_weights(tmp_path):
    """The precompute needs only the release family: ``_release_for(..., load=False)`` (the verified descriptor)."""
    done = threading.Event()
    seen = {}

    def precompute(job_dir, job, *, release, cancel_event=None):
        seen["release"] = release
        done.set()
        return {"ok": True, "steps": ["mesh"]}

    mgr = make(tmp_path, precompute_fn=precompute)
    calls = []
    real = mgr._release_for
    mgr._release_for = lambda job, **kwargs: calls.append(dict(kwargs)) or real(job, **kwargs)
    try:
        awaiting(mgr)
        assert done.wait(5) and seen["release"] is mgr.release
        assert calls and all(call.get("load") is False for call in calls)       # stage-A gate and precompute alike
    finally:
        mgr.close()


def test_claim_owner_none_moves_ownerless_cli_jobs(tmp_path):
    """v0.14.1: ``claim_owner(None, user)`` re-homes jobs written by ``cli run`` (no owner); other owners are untouched."""
    from tests._c_helpers import finished
    from wss_deploy.jobs import JobError, JobManager
    mgr = JobManager(tmp_path, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {}, offline=True)
    a = finished(mgr, owner="sess-a", case_id="A")
    b = finished(mgr, owner="sess-b", case_id="B")
    with mgr.lock:
        mgr.jobs[b["id"]]["owner"] = None
    assert mgr.count_owner("admin") == 0
    result = mgr.claim_owner(None, "admin")
    assert result["claimed"] == 1 and result["job_ids"] == [b["id"]]
    assert mgr.jobs[b["id"]]["owner"] == "admin" and mgr.jobs[a["id"]]["owner"] == "sess-a"
    assert mgr.jobs[b["id"]]["events"][-1]["action"] == "owner_claimed"
    with pytest.raises(JobError):
        mgr.claim_owner("", "admin")
    with pytest.raises(JobError):
        mgr.claim_owner(None, "")
