"""v0.14 pipeline/compute speed-ups: CPU thread default, ensemble input memo, geometry cache, atomic writes,
typed errors, shared vertex interpolation, release verification cache.  Every optimisation must return
exactly what the plain computation returns (Tier A) — the tests compare with ``np.array_equal``."""
from __future__ import annotations

import json
import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_morphology import _atlas, _tube  # noqa: E402  (synthetic tube + atlas shared with the morphology tests)

from wss_deploy import geometry as G, geometry_cache as GC, input_memo, morphology as MO, pipeline as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RELEASES = ROOT / "outputs" / "wss_deploy_release"
GOLDEN = ROOT / "outputs" / "wss_deploy_golden" / "20260920_baseline"


# ----------------------------------------------------------------------------- P1 CPU threads
def test_cpu_thread_default_cap_and_override(monkeypatch):
    monkeypatch.delenv("WSS_DEPLOY_CPU_THREADS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 192)
    assert P.default_cpu_threads() == 32
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert P.default_cpu_threads() == 8
    monkeypatch.setenv("WSS_DEPLOY_CPU_THREADS", "12")
    assert P.default_cpu_threads() == 12
    monkeypatch.setenv("WSS_DEPLOY_CPU_THREADS", "zero")          # invalid → default
    assert P.default_cpu_threads() == 8
    monkeypatch.delenv("WSS_DEPLOY_CPU_THREADS")
    assert P.inference_threads("cpu", None) == 8                # CPU without an explicit value → default
    assert P.inference_threads("cpu", 3) == 3 and P.inference_threads("cuda", 5) == 5
    assert P.inference_threads("cuda", None) is None             # GPU path untouched


def test_torch_threads_are_set_for_the_block_and_restored():
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    seen = []
    with P.torch_threads(2):
        seen.append(torch.get_num_threads())
    with P.torch_threads(None):
        seen.append(torch.get_num_threads())
    assert seen == [2, before] and torch.get_num_threads() == before


# ----------------------------------------------------------------------------- P2 input memo
def _fake_dataset(monkeypatch):
    calls = []

    def build_features(case, idx, input_features, feat_stats, pos_override=None, frame_index=None):
        calls.append(("features", len(idx)))
        return np.asarray(case["pos"])[idx].astype(np.float32) * float(feat_stats["scale"])

    def build_query_patch(case, query_idx, input_features, feat_stats, nsample=16, frame_index=None,
                          rotation=None, radius_mm=0.0, radius_k=0, offset_norm="mm"):
        calls.append(("patch", len(query_idx)))
        return {"features": np.repeat(np.asarray(case["pos"])[query_idx, None, :], nsample, axis=1).astype(np.float32),
                "relative_mm": np.zeros((len(query_idx), nsample, 3), np.float32)}

    def sample_support_indices(case, cfg, seed, *, n_points, sampling, stream, **legacy):
        calls.append(("support", n_points))
        return np.sort(np.random.default_rng(seed).choice(len(case["pos"]), n_points, replace=False))

    module = types.ModuleType("fake_dataset_v014")
    module.build_features, module.build_query_patch, module.sample_support_indices = (
        build_features, build_query_patch, sample_support_indices)
    monkeypatch.setitem(sys.modules, "fake_dataset_v014", module)
    monkeypatch.setattr(input_memo, "_TARGET_MODULE", "fake_dataset_v014")
    return module, calls


def test_input_memo_reuses_only_identical_calls_and_hands_out_copies(monkeypatch):
    module, calls = _fake_dataset(monkeypatch)
    case = {"pos": np.random.default_rng(0).normal(size=(200, 3))}
    stats = {"scale": 2.0}
    idx = np.arange(0, 200, 3)
    with input_memo.shared_inputs() as counts:
        first = module.build_features(case, idx, ["x"], stats)
        first[:] = -1.0                                            # a caller mutating its result ...
        again = module.build_features(case, idx.copy(), ("x",), {"scale": 2.0})   # equal content, new objects
        other = module.build_features(case, idx[:-1], ["x"], stats)               # different rows
        patch_a = module.build_query_patch(case, idx, ["x"], stats, nsample=4)
        patch_b = module.build_query_patch(case, idx, ["x"], stats, nsample=4)
        same_content = module.build_features(dict(case), idx, ["x"], stats)       # equal content → reused
        changed = dict(case, pos=case["pos"] + 1e-12)
        foreign = module.build_features(changed, idx, ["x"], stats)                # different content → computed
    assert np.array_equal(again, case["pos"][idx].astype(np.float32) * 2.0)      # ... never reaches the next member
    assert again is not first and len(other) == len(idx) - 1
    assert all(np.array_equal(patch_a[k], patch_b[k]) and patch_a[k] is not patch_b[k] for k in patch_a)
    assert calls == [("features", 67), ("features", 66), ("patch", 67), ("features", 67)]
    assert counts == {"calls": 7, "reused": 3} and np.array_equal(same_content, again)
    assert np.array_equal(foreign, changed["pos"][idx].astype(np.float32) * 2.0)


def test_input_memo_passes_through_outside_scope_other_threads_and_when_off(monkeypatch):
    module, calls = _fake_dataset(monkeypatch)
    case = {"pos": np.zeros((50, 3))}
    idx = np.arange(10)
    input_memo.install()
    module.build_features(case, idx, ["x"], {"scale": 1.0}); module.build_features(case, idx, ["x"], {"scale": 1.0})
    assert len(calls) == 2
    with input_memo.shared_inputs():
        module.build_features(case, idx, ["x"], {"scale": 1.0})
        worker = threading.Thread(target=lambda: module.build_features(case, idx, ["x"], {"scale": 1.0}))
        worker.start(); worker.join()
        module.build_features(case, idx, ["x"], {"scale": 1.0})
    assert len(calls) == 4                                        # scope call + other thread; the repeat reused
    monkeypatch.setenv("WSS_DEPLOY_PATCH_MEMO", "0")
    with input_memo.shared_inputs() as counts:
        module.build_features(case, idx, ["x"], {"scale": 1.0}); module.build_features(case, idx, ["x"], {"scale": 1.0})
    assert len(calls) == 6 and counts == {"calls": 0, "reused": 0}


def test_input_memo_budget_caps_memory(monkeypatch):
    module, calls = _fake_dataset(monkeypatch)
    monkeypatch.setenv("WSS_DEPLOY_PATCH_MEMO_MB", "0")
    case = {"pos": np.ones((30, 3))}
    with input_memo.shared_inputs() as counts:
        module.build_features(case, np.arange(5), ["x"], {"scale": 1.0})
        module.build_features(case, np.arange(5), ["x"], {"scale": 1.0})
    assert len(calls) == 2 and counts["reused"] == 0             # nothing stored, everything computed


def _release_or_skip(name):
    pytest.importorskip("torch")
    if not (RELEASES / name / "release.json").is_file():
        pytest.skip(f"release {name} not available")
    from wss_deploy.infer import Release
    return Release(RELEASES / name, device="cpu")


def test_memo_and_warm_up_leave_real_ensemble_predictions_bit_identical(monkeypatch):
    """CPU, fixed threads: memo on/off and a preceding warm-up give np.array_equal predictions."""
    import torch
    release = _release_or_skip("M1_3head_3seed_20260922")
    from wss_deploy.infer import synthetic_case
    case_a, case_b = synthetic_case(release, n_points=6000, seed=1), synthetic_case(release, n_points=6000, seed=1)
    # v0.15.11: prepared device inputs bypass input_memo; it is still the evaluator path's memo, tested here.
    monkeypatch.setenv("WSS_DEPLOY_PREPARED_INPUTS", "0")
    with P.torch_threads(4):
        monkeypatch.setenv("WSS_DEPLOY_PATCH_MEMO", "0"); monkeypatch.setenv("WSS_DEPLOY_KNN_MEMO", "0")
        plain = release.predict(case_a)
        monkeypatch.setenv("WSS_DEPLOY_PATCH_MEMO", "1"); monkeypatch.setenv("WSS_DEPLOY_KNN_MEMO", "1")
        assert release.warm_up() is not None
        memo = release.predict(case_b)
    assert memo["input_reuse"]["reused"] > 0 and memo["knn_reuse"]["reused"] > 0
    for key in ("wss_pa", "seed_pred_pa", "tawss_pa", "osi"):
        assert np.array_equal(plain[key], memo[key]), key
    assert torch.get_num_threads() >= 1


def test_preload_runs_the_warm_up_and_never_fails(monkeypatch):
    from wss_deploy import registry as R
    events = []

    class Loaded:
        def __init__(self, fail):
            self.fail = fail

        def warm_up(self):
            events.append("warm")
            if self.fail:
                raise RuntimeError("no GPU")
            return 0.5

    reg = R.ReleaseRegistry.__new__(R.ReleaseRegistry)
    reg.default_id, reg.cache_size = "A", 2
    reg.list = lambda: [{"id": "A"}, {"id": "B"}]
    reg.load = lambda rid: Loaded(fail=(rid == "B"))
    timings = R.preload_all(reg)
    assert set(timings) == {"A", "B"} and events == ["warm", "warm"]
    # a failed warm-up is reported per release but the preload itself is done (both releases loaded)
    state = reg.preload_state
    assert state["status"] == "done" and state["running"] is False and state["loaded"] == ["A", "B"]
    assert set(state["failed"]) == {"B"} and "预热" in state["failed"]["B"]
    monkeypatch.setenv("WSS_DEPLOY_WARMUP", "0")
    events.clear(); R.preload_all(reg)
    assert events == [] and reg.preload_state["failed"] == {}

    def broken(rid):
        raise RuntimeError(f"cannot load {rid}")

    reg.load = broken
    R.preload_all(reg)
    assert reg.preload_state["status"] == "failed" and set(reg.preload_state["failed"]) == {"A", "B"}
    monkeypatch.setenv("WSS_DEPLOY_PRELOAD", "0")
    assert R.preload_all(reg) == {} and reg.preload_state["status"] == "disabled"


def test_preload_state_is_visible_while_running():
    from wss_deploy import registry as R
    seen = []
    reg = R.ReleaseRegistry.__new__(R.ReleaseRegistry)
    reg.default_id, reg.cache_size = "A", 1
    reg.list = lambda: [{"id": "A"}]

    def load(rid):
        seen.append(dict(reg.preload_state))
        return object()

    reg.load = load
    R.preload_all(reg)
    assert seen[0]["status"] == "running" and seen[0]["running"] is True
    assert reg.preload_state == {"status": "done", "running": False, "loaded": ["A"], "failed": {}}


# ----------------------------------------------------------------------------- P3 geometry cache
def test_geometry_cache_roundtrip_key_mismatch_corruption_and_switch(tmp_path, monkeypatch):
    cache = GC.GeometryCache(tmp_path)
    key = GC.key_of("x", np.arange(3), 1.5, "a")
    assert key != GC.key_of("x", np.arange(3, dtype=np.int32), 1.5, "a")      # dtype is part of the key
    assert cache.load("kind", key) is None
    assert cache.save("kind", key, {"a": np.arange(4.0), "doc": GC.json_array({"v": [0.1, None, float("nan")]})})
    hit = GC.GeometryCache(tmp_path).load("kind", key)
    assert np.array_equal(hit["a"], np.arange(4.0))
    doc = GC.from_json_array(hit["doc"])
    assert doc["v"][0] == 0.1 and doc["v"][1] is None and np.isnan(doc["v"][2])
    path = cache.path("kind", key)
    other = GC.key_of("y")
    path.rename(cache.path("kind", other))                        # a file whose stored key does not match
    reader = GC.GeometryCache(tmp_path)
    assert reader.load("kind", other) is None and reader.stats["errors"]
    cache.path("kind", other).write_bytes(b"not a zip")
    assert GC.GeometryCache(tmp_path).load("kind", other) is None
    assert not list((tmp_path / "geometry_cache").glob(".*tmp"))  # no temporary files left behind
    monkeypatch.setenv("WSS_DEPLOY_GEOMETRY_CACHE", "0")
    off = GC.GeometryCache(tmp_path)
    assert not off.active and off.save("kind", key, {"a": np.zeros(1)}) is False and off.load("kind", key) is None


def _synthetic_job(tmp_path, monkeypatch):
    from wss_features.stl import write_binary_stl
    vertices, faces = _tube(0, 120.0, 8.0, 0.0, 0.0)
    write_binary_stl(tmp_path / "input_clean_mm.stl", vertices, faces)
    (tmp_path / "stage_a.json").write_text(json.dumps({"stage": "A", "input_sha256": "ab" * 32,
                                                       "input_check": {"ok": True, "clean_stl": "input_clean_mm.stl"}}))
    monkeypatch.setattr(P.CL, "load_vessel_geom_atlas", lambda *_: _atlas())
    return tmp_path


def test_precompute_then_every_cached_step_equals_the_plain_computation(tmp_path, monkeypatch):
    from wss_features.cloud import patch_atlas_end_radius
    from wss_features.stl import load_stl
    job = _synthetic_job(tmp_path, monkeypatch)
    # An old job record may embed a stale stage-A copy (absolute path of a deleted directory); the precompute
    # must read stage_a.json exactly like stage B does.
    stale = {"stage": "A", "input_sha256": "cd" * 32, "input_check": {"ok": True, "clean_stl": "/nonexistent/x.stl"}}
    result = P.precompute_geometry_cache(job, {"params": {}, "a": stale}, release=None)
    assert result["ok"] and result["steps"] == ["mesh", "resample", "point_geometry", "morphology"]
    kinds = {p.name.split("-", 1)[0] for p in (job / "geometry_cache").glob("*.npz")}
    assert kinds == {"mesh", "resample", "pointgeom", "morph", "morphvol"}
    # plain computations
    v_plain, f_plain = load_stl(job / "input_clean_mm.stl")
    seed = G.stable_sampling_seed("ab" * 32)
    s_plain, p_plain = G.smooth_and_resample(np.asarray(v_plain, np.float64), np.asarray(f_plain, np.int64), seed=seed)
    atlas_plain = _atlas(); patch_atlas_end_radius(atlas_plain, p_plain)
    geo_plain = G.point_geometry(p_plain, atlas_plain)
    # cached computations (fresh cache objects: every step must be a hit)
    cache = GC.GeometryCache(job)
    v, f = P._clean_mesh(job, job / "input_clean_mm.stl", cache)
    assert np.array_equal(v, v_plain) and np.array_equal(f, f_plain) and f.dtype == f_plain.dtype
    s, p = P._resampled(np.asarray(v, np.float64), np.asarray(f, np.int64), smooth_mm=1.0, spacing_mm=0.5, seed=seed, cache=cache)
    assert np.array_equal(s, s_plain) and np.array_equal(p, p_plain)
    atlas = _atlas(); patch_atlas_end_radius(atlas, p)
    geo = P._point_geometry_provider(cache)(p, atlas)
    assert np.array_equal(geo[0], geo_plain[0]) and np.array_equal(geo[1], geo_plain[1])
    for block, block_plain in ((geo[2], geo_plain[2]), (geo[3], geo_plain[3])):
        assert list(block) == list(block_plain)                     # same keys in the same order
        for key in block:
            assert np.array_equal(block[key], block_plain[key]) and type(block[key]) is type(block_plain[key])
    names = {"0": "主动脉"}
    morph_plain = MO.compute(np.asarray(v_plain, np.float64), np.asarray(f_plain, np.int64), atlas_plain, branch_names=names)
    morph = MO.compute(np.asarray(v, np.float64), np.asarray(f, np.int64), atlas, branch_names=names, section_cache=cache)
    assert json.dumps(morph, sort_keys=True) == json.dumps(morph_plain, sort_keys=True)
    assert "morph" in cache.stats["hits"] and "morphvol" in cache.stats["hits"]
    assert set(cache.stats["hits"]) >= {"mesh", "resample", "pointgeom"} and not cache.stats["writes"]


def test_cache_misses_when_any_input_changes(tmp_path, monkeypatch):
    job = _synthetic_job(tmp_path, monkeypatch)
    cache = GC.GeometryCache(job)
    v, f = P._clean_mesh(job, job / "input_clean_mm.stl", cache)
    v, f = np.asarray(v, np.float64), np.asarray(f, np.int64)
    P._resampled(v, f, smooth_mm=1.0, spacing_mm=0.5, seed=7, cache=cache)
    probe = GC.GeometryCache(job)
    P._resampled(v, f, smooth_mm=1.0, spacing_mm=0.5, seed=8, cache=probe)          # other seed
    P._resampled(v, f, smooth_mm=0.8, spacing_mm=0.5, seed=7, cache=probe)          # other smoothing
    moved = v.copy(); moved[0, 0] += 1e-9
    P._resampled(moved, f, smooth_mm=1.0, spacing_mm=0.5, seed=7, cache=probe)      # other mesh
    assert probe.stats["hits"] == [] and probe.stats["misses"] == ["resample"] * 3
    P._resampled(v, f, smooth_mm=1.0, spacing_mm=0.5, seed=7, cache=probe)
    assert probe.stats["hits"] == ["resample"]
    # the stale entries are pruned by the next stage B's bookkeeping
    keeper = GC.GeometryCache(job)
    P._resampled(v, f, smooth_mm=1.0, spacing_mm=0.5, seed=7, cache=keeper)
    P._clean_mesh(job, job / "input_clean_mm.stl", keeper)
    P.prune_geometry_cache(keeper)
    assert sorted(p.name.split("-", 1)[0] for p in (job / "geometry_cache").glob("*.npz")) == ["mesh", "resample"]


def test_precompute_never_raises_and_honours_cancel(tmp_path, monkeypatch):
    job = _synthetic_job(tmp_path, monkeypatch)
    event = threading.Event(); event.set()
    stopped = P.precompute_geometry_cache(job, {}, release=None, cancel_event=event)
    assert stopped["cancelled"] and stopped["steps"] == ["mesh"] and not stopped["ok"]
    (job / "stage_a.json").write_text("{broken")
    broken = P.precompute_geometry_cache(job, {}, release=None)
    assert broken["ok"] is False and "error" in broken
    monkeypatch.setenv("WSS_DEPLOY_GEOMETRY_CACHE", "0")
    assert P.precompute_geometry_cache(job, {}, release=None)["skipped"] == "disabled"


def test_stage_b_waits_for_a_running_precompute_of_the_same_job(tmp_path):
    lock = GC.job_lock(tmp_path)
    assert lock is GC.job_lock(tmp_path / ".") and lock is not GC.job_lock(tmp_path / "other")


@pytest.mark.skipif(os.environ.get("WSS_DEPLOY_SLOW_TESTS") != "1" or not GOLDEN.is_dir(),
                    reason="set WSS_DEPLOY_SLOW_TESTS=1 (real job, CPU, ~1 min)")
def test_real_job_stage_b_with_and_without_cache_is_bit_identical(tmp_path, monkeypatch):
    import shutil
    from wss_deploy.registry import ReleaseRegistry
    src = GOLDEN / "20260920_144135_673ccd0e36b1"
    outputs = {}
    for variant in ("plain", "cached"):
        dst = tmp_path / variant
        dst.mkdir()
        for name in ("input.stl", "input_clean_mm.stl", "stage_a.json", "job.json"):
            shutil.copy2(src / name, dst / name)
        shutil.copytree(src / "centerline", dst / "centerline")
        a = json.loads((dst / "stage_a.json").read_text(encoding="utf-8"))
        a["input_check"]["clean_stl"] = str(dst / "input_clean_mm.stl")
        (dst / "stage_a.json").write_text(json.dumps(a), encoding="utf-8")
        job = json.loads((dst / "job.json").read_text(encoding="utf-8"))
        monkeypatch.setenv("WSS_DEPLOY_GEOMETRY_CACHE", "1" if variant == "cached" else "0")
        release = ReleaseRegistry(None, device="cpu").load(job["model_release"]["id"], device="cpu")
        if variant == "cached":
            assert P.precompute_geometry_cache(dst, job, release=release)["ok"]
        meta = P.stage_b(dst, job["mapping"], release, confirmed=True, device="cpu", threads=8)
        z = np.load(dst / "field.npz")
        outputs[variant] = ({k: z[k] for k in z.files}, meta, (dst / "points_wss.csv").read_bytes())
    plain, cached = outputs["plain"], outputs["cached"]
    assert plain[0].keys() == cached[0].keys() and all(np.array_equal(plain[0][k], cached[0][k]) for k in plain[0])
    assert plain[2] == cached[2]
    assert json.dumps(plain[1]["morphology"], sort_keys=True) == json.dumps(cached[1]["morphology"], sort_keys=True)
    assert cached[1]["timing_s"]["smooth_resample"] < plain[1]["timing_s"]["smooth_resample"]


# ----------------------------------------------------------------------------- P4 atomic writes / typed errors
def test_atomic_output_replaces_on_success_and_keeps_the_old_file_on_failure(tmp_path):
    target = tmp_path / "summary.json"
    target.write_text("old")
    with pytest.raises(RuntimeError):
        with P.atomic_output(target) as tmp:
            tmp.write_text("half")
            raise RuntimeError("disk")
    assert target.read_text() == "old" and sorted(p.name for p in tmp_path.iterdir()) == ["summary.json"]
    P.write_text_atomic(target, "new")
    assert target.read_text() == "new" and sorted(p.name for p in tmp_path.iterdir()) == ["summary.json"]
    with P.atomic_output(tmp_path / "wall.vtp") as tmp:
        assert tmp.suffix == ".vtp" and tmp.name.startswith(".")      # writers choose the format by suffix
    assert not (tmp_path / "wall.vtp").exists()                        # a writer that produced nothing
    P.save_npz_atomic(tmp_path / "field.npz", a=np.arange(3))
    assert np.array_equal(np.load(tmp_path / "field.npz")["a"], np.arange(3))


def test_inference_oom_becomes_a_retryable_resource_error_other_errors_pass_through():
    from wss_deploy.errors import ResourceError

    class Boom:
        def __init__(self, exc):
            self.exc = exc

        def predict(self, case):
            raise self.exc

    with pytest.raises(ResourceError) as error:
        P.run_inference(Boom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")), {})
    assert error.value.retryable and error.value.retry_hint == "cpu" and "CUDA" in error.value.admin_detail
    with pytest.raises(KeyError):
        P.run_inference(Boom(KeyError("pos")), {})


def test_unreadable_stl_is_a_typed_input_error_that_is_still_a_value_error(tmp_path):
    from wss_deploy.errors import InputGeometryError
    from wss_deploy.ingest import ingest
    bad = tmp_path / "bad.stl"
    bad.write_bytes(b"solid x\n" + b"\0" * 200)
    with pytest.raises(InputGeometryError) as error:
        ingest(bad, tmp_path / "out")
    assert isinstance(error.value, ValueError) and error.value.category == "input_geometry"
    assert error.value.retryable is False and "STL" in str(error.value)


# ----------------------------------------------------------------------------- P5 small wins
def test_shared_vertex_interpolation_is_bit_identical_to_the_report_helpers():
    from wss_deploy import report as R
    rng = np.random.default_rng(3)
    pts = rng.uniform(-10, 10, (400, 3))
    vertices = np.concatenate([rng.uniform(-10, 10, (150, 3)), rng.uniform(30, 40, (5, 3))])   # 5 uncovered
    interp = P.VertexInterpolation(pts, vertices)
    for values in (rng.normal(size=400).astype(np.float32), rng.uniform(0, 5, 400)):
        expected = R.interpolate_to_vertices(pts, values, vertices)
        got = interp.values(values)
        assert got.dtype == expected.dtype and np.array_equal(got, expected, equal_nan=True)
    labels = rng.integers(0, 7, 400).astype(np.int16)
    assert np.array_equal(interp.labels(labels), R.nearest_label(pts, labels, vertices))
    with pytest.raises(ValueError):
        interp.values(np.zeros(3))
    few = P.VertexInterpolation(pts[:3], vertices)                  # k > number of points
    assert np.array_equal(few.values(np.arange(3.0)), R.interpolate_to_vertices(pts[:3], np.arange(3.0), vertices), equal_nan=True)


def test_release_verification_is_cached_but_a_changed_file_still_fails(tmp_path, monkeypatch):
    import hashlib
    from wss_deploy import registry as R
    root = tmp_path / "rel"
    (root / "models" / "m").mkdir(parents=True); (root / "rules").mkdir()
    (root / "release.json").write_text("{}")
    (root / "models" / "m" / "ckpt_best.pt").write_bytes(b"weights")
    body = "".join(f"{hashlib.sha256((root / rel).read_bytes()).hexdigest()}  {rel}\n" for rel in ("models/m/ckpt_best.pt",))
    (root / "MANIFEST.sha256").write_text(body)
    record = types.SimpleNamespace(path=root, fingerprint=R._release_fingerprint(root / "release.json", root / "MANIFEST.sha256"),
                                   info={}, contract={}, id="rel")
    calls = []

    def verify(rec):
        calls.append(rec.id)
        manifest = (rec.path / "MANIFEST.sha256").read_text().split()
        if hashlib.sha256((rec.path / manifest[1]).read_bytes()).hexdigest() != manifest[0]:
            raise R.ReleaseError("发布包文件校验失败")

    monkeypatch.setattr(R, "_verify_package", verify)
    monkeypatch.setattr(R, "_VERIFIED", {})
    R._verify_package_cached(record); R._verify_package_cached(record)
    assert calls == ["rel"]                                          # second call: stat signature unchanged
    (root / "models" / "m" / "ckpt_best.pt").write_bytes(b"tampered")
    with pytest.raises(R.ReleaseError):
        R._verify_package_cached(record)
    with pytest.raises(R.ReleaseError):                              # failures are never cached
        R._verify_package_cached(record)
    assert calls == ["rel", "rel", "rel"]
    (root / "models" / "m" / "ckpt_best.pt").write_bytes(b"weights")
    monkeypatch.setenv("WSS_DEPLOY_RELEASE_VERIFY_TTL", "0")          # TTL expired → full verification again
    R._verify_package_cached(record); R._verify_package_cached(record)
    assert calls == ["rel"] * 5


def test_stage_a_preview_reuses_the_cached_clean_mesh(tmp_path, monkeypatch):
    job = _synthetic_job(tmp_path, monkeypatch)
    from wss_features import stl as stl_module
    reads = []
    original = stl_module.load_stl
    monkeypatch.setattr(stl_module, "load_stl", lambda path: reads.append(Path(path).name) or original(path))
    first = P._clean_mesh(job, job / "input_clean_mm.stl", GC.GeometryCache(job))
    second = P._clean_mesh(job, job / "input_clean_mm.stl", GC.GeometryCache(job))
    assert reads == ["input_clean_mm.stl"]                           # read once, then served from the cache
    assert all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(first, second))
    preview = P._preview_surface(job / "input_clean_mm.stl", mesh=second)
    assert preview == P._preview_surface(job / "input_clean_mm.stl")


def test_a_cache_entry_of_another_layout_is_recomputed_not_an_error(tmp_path, monkeypatch):
    job = _synthetic_job(tmp_path, monkeypatch)
    cache = GC.GeometryCache(job)
    v, f = P._clean_mesh(job, job / "input_clean_mm.stl", cache)
    s, pts = P._resampled(np.asarray(v, np.float64), np.asarray(f, np.int64), smooth_mm=1.0, spacing_mm=0.5, seed=1, cache=cache)
    atlas = _atlas()
    expected = G.point_geometry(pts, atlas)
    key = GC.key_of("point-geometry", P._GEOMETRY_CODE, GC.feature_program_hash(), np.asarray(pts),
                    np.asarray(atlas.table), np.asarray(atlas.tree_rows), list(atlas.columns))
    cache.save("pointgeom", key, {"normals": np.zeros(3)})             # right key, wrong layout
    got = P._point_geometry_provider(GC.GeometryCache(job))(pts, atlas)
    assert np.array_equal(got[0], expected[0]) and np.array_equal(got[1], expected[1])
    mesh_key = GC.key_of("clean-mesh", GC.feature_program_hash(), GC.file_sha256(job / "input_clean_mm.stl"))
    cache.save("mesh", mesh_key, {"vertices": np.zeros((1, 3))})       # no faces
    again = P._clean_mesh(job, job / "input_clean_mm.stl", GC.GeometryCache(job))
    assert np.array_equal(again[0], v) and np.array_equal(again[1], f)


def test_rebuild_reads_each_npz_array_once_and_caches_the_surface_variation(tmp_path, monkeypatch):
    import wss_features.cloud as cloud
    from wss_features.cloud import pca_normals
    from wss_deploy import rebuild_report as RB
    rng = np.random.default_rng(5)
    np.savez_compressed(tmp_path / "field.npz", segment_id=np.arange(10), pts=rng.normal(size=(10, 3)))
    arrays = RB._NpzArrays(tmp_path / "field.npz")
    assert arrays["segment_id"] is arrays["segment_id"] and set(arrays.files) == {"segment_id", "pts"}
    vertices, faces = _tube(0, 60.0, 8.0, 0.0, 0.0)
    pts = vertices[::3] + rng.normal(scale=0.05, size=vertices[::3].shape)
    atlas = _atlas()
    expected = pca_normals(pts, atlas)[1]
    calls = []
    monkeypatch.setattr(cloud, "pca_normals", lambda *a, **k: calls.append(1) or pca_normals(*a, **k))
    first = RB._surface_variation(tmp_path, pts, atlas)
    second = RB._surface_variation(tmp_path, pts, atlas)
    assert np.array_equal(first, expected) and np.array_equal(second, expected) and len(calls) == 1
    assert any(p.name.startswith("rebuildvar-") for p in (tmp_path / "geometry_cache").iterdir())


def test_inference_sections_run_one_at_a_time_and_set_threads_inside(monkeypatch):
    """v0.16.1: stage B overlaps, the model section does not; the CPU thread count is set inside it."""
    import threading
    import time
    import torch
    active, peak, seen, lock = [0], [0], [], threading.Lock()

    class Slow:
        def predict(self, case):
            with lock:
                active[0] += 1; peak[0] = max(peak[0], active[0])
            seen.append(torch.get_num_threads())
            time.sleep(0.1)
            with lock:
                active[0] -= 1
            return {"ok": case}

    before = torch.get_num_threads()
    threads = [threading.Thread(target=P.run_inference, args=(Slow(), i), kwargs={"threads": 2}) for i in range(3)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert peak[0] == 1 and seen == [2, 2, 2] and torch.get_num_threads() == before
