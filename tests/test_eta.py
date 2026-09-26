"""§21.2: stage progress and remaining-time estimate — history medians (same family, current feature program),
default-table interpolation with its floor, running remainder (never negative), queue wait, awaiting_confirmation,
and old records without an estimate."""
from __future__ import annotations

import json
import threading
import time

import pytest

from wss_deploy import eta as E
from wss_deploy.jobs import JobManager
from tests._c_helpers import MAPPING, finished, manager, stage_a_stub

LV, LIU = 23184, 273227
HASH = "b" * 64


def _sample(family="wall", faces=LV, scale=1.0, source_hash=HASH, release_id="REL_A"):
    timing = {"ingest": 0.1 * scale, "centerline": 3.0 * scale, "smooth_resample": 4.0 * scale, "features": 3.0 * scale,
              "inference_5_models": 6.0 * scale, "metrics_and_interpolation": 0.2 * scale, "morphology": 1.5 * scale,
              "export": 1.0 * scale, "total": 99.0}
    if family == "volume":
        timing = {"ingest": 0.1, "centerline": 3.0, "smooth_resample": 4.0, "volume_features": 9.0, "inference_volume": 1.0,
                  "streamlines_and_interpolation": 3.0, "morphology": 1.5, "export": 0.5}
    return E.history_sample(family=family, faces=faces, timing=timing, source_hash=source_hash, release_id=release_id)


# ------------------------------------------------------------------------------------------ pure functions
def test_default_table_interpolates_linearly_and_is_floored():
    assert E.default_seconds("centerline", "wall", LV) == pytest.approx(3.4)
    assert E.default_seconds("centerline", "wall", LIU) == pytest.approx(65.0)
    mid = (LV + LIU) / 2
    assert E.default_seconds("morphology", "wall", mid) == pytest.approx((1.7 + 14.0) / 2)
    assert E.default_seconds("centerline", "wall", 1000) == pytest.approx(3.4 / 2)            # floor = half the small case
    assert E.default_seconds("smooth_resample", "wall", 1000) == pytest.approx(4.2 - 8.6 * 22184 / 250043)
    assert E.default_seconds("inference", "wall", LV) == 7.0 and E.default_seconds("inference", "volume", LIU) == 3.2
    assert E.default_seconds("inference", "volume", 5_000_000) == 3.2
    assert E.default_seconds("ingest", "wall", None) == 0.2                                   # unknown size → small case
    assert E.default_seconds("centerline", "wall", 2 * LIU) > 65.0                          # extrapolates upwards


def test_history_uses_only_same_family_and_current_feature_program():
    samples = [_sample(scale=1.0), _sample(scale=2.0), _sample(scale=3.0),
               _sample(scale=50.0, source_hash="a" * 64),                                   # older feature program
               _sample("volume"), _sample("volume"), _sample("volume")]
    table, basis, n = E.expected_table("wall", LV, samples, source_hash=HASH, release_id="REL_A")
    assert basis == "history" and n == 3
    assert table["smooth_resample"] == pytest.approx(8.0)                                    # median of 4 / 8 / 12 s
    assert table["inference"] == pytest.approx(12.0) and "volume_features" not in table
    # normalised by size: the same seconds-per-10k-faces on a case ten times larger
    big, _, _ = E.expected_table("wall", 10 * LV, samples, source_hash=HASH)
    assert big["smooth_resample"] == pytest.approx(80.0)
    assert big["centerline"] == pytest.approx(6.0 * 10 ** 1.4)                               # VMTK is super-linear
    # fewer than three matching jobs → default table
    table, basis, n = E.expected_table("wall", LV, samples[:2], source_hash=HASH)
    assert basis == "default" and n == 2 and table["centerline"] == pytest.approx(3.4)
    table, basis, n = E.expected_table("wall", LV, samples, source_hash="c" * 64)
    assert basis == "default" and n == 0
    table, basis, n = E.expected_table("volume", LV, samples, source_hash=HASH)
    assert basis == "history" and n == 3 and table["volume_features"] == pytest.approx(9.0) and table["streamlines"] == pytest.approx(3.0)


def test_inference_prefers_the_same_release_when_enough_runs_exist():
    samples = [_sample(scale=1.0, release_id="X5D")] * 3 + [_sample(scale=0.5, release_id="M1")] * 3
    x5d, _, _ = E.expected_table("wall", LV, samples, source_hash=HASH, release_id="X5D")
    m1, _, _ = E.expected_table("wall", LV, samples, source_hash=HASH, release_id="M1")
    assert x5d["inference"] == pytest.approx(6.0) and m1["inference"] == pytest.approx(3.0)
    assert x5d["smooth_resample"] == m1["smooth_resample"]                                   # other stages pool the family


def test_phase_and_timing_key_mapping():
    assert E.phase_stage("wall", "geometry") == "smooth_resample"
    assert E.phase_stage("volume", "features") == "volume_features" and E.phase_stage("wall", "features") == "features"
    assert E.phase_stage("wall", "metrics", "正在汇总预测点云统计") == "metrics"
    assert E.phase_stage("wall", "metrics", "正在测量沿程管腔截面与瘤体形态") == "morphology"
    assert E.phase_stage("volume", "metrics", "正在积分体内稳态流线并汇总压力与速度") == "streamlines"
    assert E.phase_stage("volume", "metrics", "正在测量沿程管腔截面与瘤体形态") == "morphology"
    assert E.phase_stage("wall", "未知") is None
    seconds = E.stage_seconds({"inference_5_models": 4.6, "metrics_and_interpolation": 0.2, "streamlines_and_interpolation": 2.4,
                               "metrics_and_export": 1.0, "export": 0.5, "total": 40, "bogus": 3})
    assert seconds == {"inference": 4.6, "metrics": 0.2, "streamlines": 2.4, "export": 1.5}
    assert E.stages_for("volume")[2:] == E.B_STAGES["volume"] and E.stages_for(None) == E.A_STAGES + E.B_STAGES["wall"]


def _table(family="wall"):
    table, _, _ = E.expected_table(family, LV, [], source_hash=None)
    return table


def test_running_remaining_uses_actuals_and_never_goes_negative():
    table = _table()
    now = 1000.0
    clock = {"segment": "B", "current": "features", "current_started_ts": now - 1.0, "done": {"smooth_resample": 4.5}}
    out = E.build_eta(family="wall", status="running", segment="B", faces=LV, table=table, basis="default", n_history=0,
                      a_seconds={"ingest": 0.1, "centerline": 3.3}, clock=clock, now=now)
    rows = {row["key"]: row for row in out["stages"]}
    assert [row["key"] for row in out["stages"]] == list(E.A_STAGES + E.B_STAGES["wall"])
    assert rows["ingest"]["state"] == rows["centerline"]["state"] == rows["smooth_resample"]["state"] == "done"
    assert rows["centerline"]["elapsed_s"] == 3.3 and rows["smooth_resample"]["elapsed_s"] == 4.5
    assert rows["features"]["state"] == "running" and rows["features"]["elapsed_s"] == 1.0
    pending = sum(table[k] for k in ("inference", "metrics", "morphology", "export"))
    assert out["remaining_s"] == pytest.approx(round(table["features"] - 1.0 + pending, 1), abs=0.05)
    assert out["current"] == "features" and out["segment"] == "B" and out["segment_remaining_s"] == out["remaining_s"]
    assert out["basis"] == "default" and out["faces"] == LV and out["updated_at"]
    # overdue: a shrinking, strictly positive tail
    tails = []
    for elapsed in (table["features"] * 0.9, table["features"] * 2, table["features"] * 10):
        clock = {"segment": "B", "current": "features", "current_started_ts": now - elapsed, "done": {}}
        out = E.build_eta(family="wall", status="running", segment="B", faces=LV, table=table, basis="default", n_history=0,
                          clock=clock, now=now)
        tails.append(out["stages"][3].get("overdue", False))
        assert out["remaining_s"] >= pending - 0.05
    assert tails == [False, True, True]
    values = [E.stage_remaining(5.0, t) for t in (0, 2, 4, 4.5, 5, 10, 100, 1e6)]
    assert values[0] == 5.0 and values[2] == pytest.approx(1.0) and all(v > 0 for v in values)
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_running_before_first_progress_and_stage_a():
    table = _table()
    out = E.build_eta(family="wall", status="running", segment="B", faces=LV, table=table, basis="default", n_history=0,
                      clock={"segment": "B", "current": None, "done": {}}, now=10.0)
    assert out["current"] == "smooth_resample" and out["stages"][2]["elapsed_s"] == 0.0
    assert out["remaining_s"] == pytest.approx(out["b_total_s"], abs=0.05)
    out = E.build_eta(family="volume", status="running", segment="A", faces=LV, table=_table("volume"), basis="default", n_history=0,
                      clock={"segment": "A", "current": "centerline", "current_started_ts": 9.0, "done": {"ingest": 0.2}}, now=10.0)
    rows = {row["key"]: row for row in out["stages"]}
    assert rows["ingest"]["state"] == "done" and rows["centerline"]["state"] == "running" and rows["streamlines"]["state"] == "pending"
    assert out["segment_remaining_s"] == pytest.approx(3.4 - 1.0, abs=0.05)
    assert out["remaining_s"] == pytest.approx(3.4 - 1.0 + out["b_total_s"], abs=0.1)


def test_awaiting_confirmation_gives_the_stage_b_total_and_final_states_nothing():
    table = _table("volume")
    out = E.build_eta(family="volume", status="awaiting_confirmation", segment="B", faces=LV, table=table, basis="default",
                      n_history=0, a_seconds={"ingest": 0.1, "centerline": 3.3}, now=5.0)
    b_total = round(sum(table[k] for k in E.B_STAGES["volume"]), 1)
    assert out["remaining_s"] == b_total == out["b_total_s"] and out["current"] is None
    assert [row["state"] for row in out["stages"]] == ["done", "done"] + ["pending"] * 6
    assert out["stages"][1]["elapsed_s"] == 3.3
    out = E.build_eta(family="wall", status="awaiting_input", segment="A", faces=LV, table=_table(), basis="default", n_history=0,
                      a_seconds={"ingest": 0.1}, now=5.0)
    assert out["stages"][0]["state"] == "done" and out["stages"][1]["state"] == "pending"
    for status in ("done", "failed", "cancelled", "interrupted"):
        assert E.build_eta(family="wall", status=status, segment="B", faces=LV, table=_table(), basis="default", n_history=0, now=1.0) is None


def test_queue_wait_and_stl_faces():
    assert E.queue_wait([]) == 0.0
    assert E.queue_wait([{"segment": "B", "segment_remaining_s": 20}, {"segment": "A", "segment_remaining_s": 10}]) == 20.0
    assert E.queue_wait([{"segment": "A", "segment_remaining_s": 30}, {"segment": "A", "segment_remaining_s": 10}]) == 20.0
    binary = b"\0" * 80 + (3).to_bytes(4, "little") + b"\0" * 150
    assert E.stl_faces(binary) == 3 and E.faces_from_size(len(binary)) == 3
    assert E.stl_faces(b"solid x\n" + b"facet normal 0 0 1\nendfacet\n" * 4 + b"endsolid x\n" + b" " * 40) == 4
    assert E.stl_faces(b"solid") is None and E.faces_from_size(0) is None


# ------------------------------------------------------------------------------------------ job manager
def test_manager_history_from_finished_jobs_with_the_current_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "current_feature_hash", lambda: HASH)
    mgr = manager(tmp_path)
    for index in range(4):
        job = finished(mgr, case_id=f"C{index}", content=f"geom-{index}".encode())
        with mgr.lock:
            record = mgr.jobs[job["id"]]
            record["a"]["input_check"]["faces"] = LV
            record["a"]["timing_s"] = {"ingest": 0.1, "centerline": 3.0}
            record["summary"]["timing_s"] = {"smooth_resample": 4.0 + index, "features": 3.0, "inference_5_models": 6.0,
                                             "metrics_and_interpolation": 0.2, "morphology": 1.5, "export": 1.0}
            # three runs carry the hash in the record, one only in summary.json, one is from an older program
            if index < 2:
                record["summary"]["feature_contract"] = {"source_hash": HASH}
            elif index == 2:
                path = mgr.root / job["id"] / "summary.json"
                full = json.loads(path.read_text())
                full["feature_contract"] = {"version": "v", "source_hash": HASH, "recorded_hash": HASH}
                path.write_text(json.dumps(full, indent=1))
            else:
                record["summary"]["feature_contract"] = {"source_hash": "a" * 64}
    queued = mgr.create("owner", content=b"\0" * 80 + LV.to_bytes(4, "little") + b"\0" * (50 * LV), filename="new.stl")
    eta = queued["eta"]
    assert eta["basis"] == "history" and eta["n_history"] == 3 and eta["faces"] == LV and eta["status"] == "queued"
    rows = {row["key"]: row for row in eta["stages"]}
    assert rows["smooth_resample"]["expected_s"] == 5.0                                     # median of 4 / 5 / 6 s
    assert eta["queue_ahead"] == 0 and eta["queue_ahead_s"] == 0.0
    assert all(row["state"] == "pending" for row in eta["stages"])
    light = next(row for row in mgr.list("owner") if row["id"] == queued["id"])
    assert light["eta"]["remaining_s"] == eta["remaining_s"]
    assert all("eta" not in row for row in mgr.list("owner") if row["status"] == "done")
    mgr.close()


def test_manager_tracks_stages_while_running_and_queue_ahead(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "current_feature_hash", lambda: None)
    seen, release = {}, threading.Event()

    def stage_b(job_dir, mapping, rel, *, progress, cancelled, **_kwargs):
        progress("geometry", "正在平滑表面并重采样")
        time.sleep(0.05)
        progress("features", "正在构建 27 维几何特征")
        time.sleep(0.05)
        progress("inference", "正在运行五模型集成")
        job_id = job_dir.name
        seen["running"] = mgr.get(job_id, "owner")["eta"]
        seen["queued"] = mgr.get(other["id"], "owner")["eta"]
        progress("metrics", "正在汇总预测点云统计")
        progress("metrics", "正在测量沿程管腔截面与瘤体形态")
        progress("export", "正在写入 CSV、VTP 和 HTML 报告")
        return {"peak": {"p99_pa": 1.0}, "fields": {"wss": {}}, "timing_s": {"smooth_resample": 0.05, "features": 0.05},
                "run_identity": "r" * 64, "feature_contract": {"version": "v", "source_hash": HASH, "recorded_hash": HASH, "matches_record": True}}

    mgr = manager(tmp_path, stage_b=stage_b)
    job = mgr.create("owner", content=b"stl-one", filename="one.stl")
    mgr.run_next()
    waiting = mgr.get(job["id"], "owner")
    assert waiting["status"] == "awaiting_confirmation" and waiting["eta"]["remaining_s"] == waiting["eta"]["b_total_s"]
    assert [row["state"] for row in waiting["eta"]["stages"][:2]] == ["done", "done"]
    assert waiting["eta"]["stages"][1]["elapsed_s"] == 1.0                                  # stage-A actual from timing_s
    mgr.confirm(job["id"], "owner", {"version": waiting["version"], "mapping": dict(MAPPING), "acknowledged": True})
    other = mgr.create("owner", content=b"stl-two", filename="two.stl")
    mgr.run_next()                                                                         # runs stage B of the first job
    running = seen["running"]
    rows = {row["key"]: row for row in running["stages"]}
    assert running["status"] == "running" and running["current"] == "inference" and running["segment"] == "B"
    assert rows["smooth_resample"]["state"] == rows["features"]["state"] == "done" and rows["smooth_resample"]["elapsed_s"] >= 0.0
    assert rows["inference"]["state"] == "running" and rows["metrics"]["state"] == "pending"
    assert running["remaining_s"] > 0
    queued = seen["queued"]
    assert queued["status"] == "queued" and queued["queue_ahead"] == 1 and queued["queue_ahead_s"] == pytest.approx(running["segment_remaining_s"], abs=0.2)
    done = mgr.get(job["id"], "owner")
    assert done["status"] == "done" and "eta" not in done and "stage_clock" not in done
    assert done["summary"]["feature_contract"] == {"version": "v", "source_hash": HASH}
    assert done["summary"]["eta_cold_start"] is True                                   # first run of REL_A in this process
    record = json.loads((mgr.root / job["id"] / "job.json").read_text())
    assert "stage_clock" not in record
    mgr.close()


def test_old_records_load_without_estimate_errors(tmp_path):
    job_dir = tmp_path / "20260101_old"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({
        "id": "20260101_old", "owner": "owner", "status": "running", "stage": "B", "version": 3, "events": [],
        "created_at": "2026-01-01T00:00:00+08:00", "stage_clock": {"segment": "B", "current": "features", "current_started_ts": 1.0}}))
    done_dir = tmp_path / "20260101_done"
    done_dir.mkdir()
    (done_dir / "job.json").write_text(json.dumps({"id": "20260101_done", "owner": "owner", "status": "done", "stage": "B", "version": 2,
                                                   "events": [], "summary": {"peak": {"p99_pa": 1.0}}}))
    (done_dir / "summary.json").write_text("{not json")
    mgr = JobManager(tmp_path, stage_a_fn=stage_a_stub, stage_b_fn=lambda *_a, **_k: {})
    old = mgr.get("20260101_old", "owner")
    assert old["status"] == "interrupted" and "eta" not in old and "stage_clock" not in mgr.jobs["20260101_old"]
    assert "eta" not in mgr.get("20260101_done", "owner")
    queued = mgr.create("owner", content=b"tiny", filename="t.stl")                         # no face count → default, small case
    assert queued["eta"]["faces"] is None and queued["eta"]["basis"] == "default" and queued["eta"]["n_history"] == 0
    assert queued["eta"]["stages"][1]["expected_s"] == 3.4
    mgr.close()


def test_second_stage_b_job_reports_waiting_behind_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "current_feature_hash", lambda: None)
    gate, entered = threading.Event(), threading.Event()

    def stage_b(job_dir, mapping, rel, *, progress, cancelled, **_kwargs):
        progress("geometry", "正在平滑表面并重采样")
        entered.set()
        gate.wait(10)
        return {"peak": {"p99_pa": 1.0}, "fields": {"wss": {}}, "timing_s": {}, "run_identity": "r" * 64}

    mgr = manager(tmp_path, stage_b=stage_b)
    ids = [mgr.create("owner", content=name.encode(), filename=f"{name}.stl")["id"] for name in ("one", "two")]
    mgr.run_next(); mgr.run_next()                                                       # stage A of both
    for job_id in ids:
        waiting = mgr.get(job_id, "owner")
        mgr.confirm(job_id, "owner", {"version": waiting["version"], "mapping": dict(MAPPING), "acknowledged": True})
    mgr.start()
    try:
        assert entered.wait(10)
        deadline = time.time() + 10
        while time.time() < deadline and not all(mgr.get(i, "owner")["status"] == "running" for i in ids):
            time.sleep(0.05)
        first, second = (mgr.get(i, "owner")["eta"] for i in ids)
        if first.get("waiting"):
            first, second = second, first
        assert not first.get("waiting") and first["current"] == "smooth_resample"
        assert second["waiting"] is True and second["queue_ahead"] == 1
        assert second["queue_ahead_s"] == pytest.approx(first["segment_remaining_s"], abs=0.2)
        assert second["stages"][2]["elapsed_s"] == 0.0
    finally:
        gate.set()
        deadline = time.time() + 10
        while time.time() < deadline and not all(mgr.get(i, "owner")["status"] == "done" for i in ids):
            time.sleep(0.05)
        mgr.close()
    assert all("eta" not in mgr.get(i, "owner") for i in ids)


def test_inference_history_needs_same_release_and_device_else_scaled_default():
    samples = [_sample(scale=1.0, release_id="X5D"), _sample(scale=1.0, release_id="X5D"), _sample(scale=0.5, release_id="M1")]
    table, basis, _ = E.expected_table("wall", LV, samples, source_hash=HASH, release_id="M1", models_count=3)
    assert basis == "history" and table["inference"] == pytest.approx(7.0 * 3 / 5)            # < 3 M1 runs → scaled default
    assert table["smooth_resample"] == pytest.approx(4.0)                                    # other stages still pool the family
    gpu = [dict(s, device="cuda") for s in [_sample(scale=1.0, release_id="X5D")] * 3]
    cpu = [dict(s, device="cpu") for s in [_sample(scale=10.0, release_id="X5D")] * 3]
    on_gpu, _, _ = E.expected_table("wall", LV, gpu + cpu, source_hash=HASH, release_id="X5D", device="cuda")
    on_cpu, _, _ = E.expected_table("wall", LV, gpu + cpu, source_hash=HASH, release_id="X5D", device="cpu")
    assert on_gpu["inference"] == pytest.approx(6.0) and on_cpu["inference"] == pytest.approx(60.0)
    assert E.default_seconds("inference", "wall", LV, models_count=3) == pytest.approx(4.2)
    assert E.default_seconds("inference", "volume", LV, models_count=3) == 3.2               # volume table is per package


def test_cold_start_runs_are_left_out_of_the_inference_history():
    sample = E.history_sample(family="wall", faces=LV, timing={"smooth_resample": 4.0, "inference_5_models": 66.0},
                              source_hash=HASH, cold_start=True)
    assert sample["seconds"] == {"smooth_resample": 4.0}


def test_precompute_shrinks_the_covered_stage_b_stages():
    """v0.14 (J2): a finished background precompute leaves a 10 % residual of the stages it fully covers; the
    point geometry is only ⅔ of ``features``, which keeps ⅓ plus the residual of the rest (40 %)."""
    table = {"smooth_resample": 4.0, "features": 3.0, "inference": 7.0, "morphology": 2.0}
    covered = E.steps_to_stages(["mesh", "resample", "point_geometry", "morphology"], "wall")
    assert covered == {"smooth_resample", "features", "morphology"}
    assert E.steps_to_stages(["mesh", "point_geometry"], "volume") == {"smooth_resample"}
    out = E.apply_precompute(table, covered)
    assert out == pytest.approx({"smooth_resample": 0.4, "features": 1.2, "inference": 7.0, "morphology": 0.2})
    assert E.apply_precompute(table, covered, coverage={}) == pytest.approx(
        {"smooth_resample": 0.4, "features": 0.3, "inference": 7.0, "morphology": 0.2})
    assert E.precompute_stages("volume") == ("smooth_resample", "morphology")


def test_streamed_face_counter_agrees_with_stl_faces():
    """v0.14 streamed uploads: counting faces chunk by chunk gives exactly :func:`stl_faces`."""
    import random
    rng = random.Random(7)
    samples = [b"", b"solid", b"\0" * 80 + (2).to_bytes(4, "little") + b"\1" * 100, b"\0" * 80 + (9).to_bytes(4, "little") + b"x" * 30,
               b"solid a\n" + b"facet\nendfacet\n" * 40 + b"endsolid\n", bytes(rng.getrandbits(8) for _ in range(500)) + b"endfacetendfacet"]
    for data in samples:
        for size in (1, 3, 7, 8, 9, 64, 10_000):
            counter = E.StlFaceCounter()
            for start in range(0, len(data), size):
                counter.update(data[start:start + size])
            assert counter.result() == E.stl_faces(data), (data[:20], size)


def test_cache_assisted_runs_leave_their_reused_stages_out_of_the_history(tmp_path, monkeypatch):
    """Runs whose ``summary.geometry_cache.reused`` is non-empty report ≈ 0 s for the reused stages; a later
    job without a precompute must still get the full-run history for them."""
    monkeypatch.setattr(E, "current_feature_hash", lambda: HASH)
    assert E.cache_reused_stages({"reused": ["mesh", "resample", "pointgeom", "morph", "morphvol", "other"]}) == \
        {"smooth_resample", "features", "morphology"}
    assert E.cache_reused_stages(None) == set() and E.cache_reused_stages({"reused": []}) == set()
    mgr = manager(tmp_path)
    full = {"smooth_resample": 4.0, "features": 3.0, "inference_5_models": 7.0, "morphology": 2.0, "export": 1.0}
    hit = {"smooth_resample": 0.0, "features": 1.0, "inference_5_models": 7.0, "morphology": 0.01, "export": 1.0}
    for index, (timing, cache) in enumerate([(full, None)] * 3 + [(hit, {"reused": ["mesh", "resample", "pointgeom", "morph"]})] * 4):
        job = mgr.create("owner", content=f"solid {index}".encode(), filename=f"c{index}.stl")
        with mgr.lock:
            record = mgr.jobs[job["id"]]
            summary = {"fields": {"wss": {}}, "timing_s": timing, "feature_contract": {"source_hash": HASH}, "device": "cuda"}
            if cache:
                summary["geometry_cache"] = cache
            record.update(status="done", stage="B", summary=summary, finished_ts=1000.0 + index,
                          a={"input_check": {"faces": LV}, "timing_s": {"ingest": 0.1, "centerline": 3.0}})
    fresh = mgr.create("owner", content=b"solid new", filename="new.stl")
    mgr.jobs[fresh["id"]]["input_faces"] = LV
    stages = {row["key"]: row["expected_s"] for row in mgr.get(fresh["id"], "owner")["eta"]["stages"]}
    assert stages["smooth_resample"] == pytest.approx(4.0, abs=0.05) and stages["morphology"] == pytest.approx(2.0, abs=0.05)
    assert stages["features"] == pytest.approx(3.0, abs=0.05)                     # not dragged towards the cache hits
    mgr.close()
