"""v0.15.11 (P0-4): the confirmed PF6/VF6 geometry is reused exactly from ``geometry_cache/volume_case-*.npz``."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from wss_deploy import geometry_cache as GC, volume_cache as VC
from wss_deploy.pipeline import prune_geometry_cache


class _Atlas:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.table = rng.normal(size=(40, 5))
        self.columns = ["x_mm", "y_mm", "z_mm", "radius_mm", "segment_id"]
        self.segments = [{"segment_id": 0, "outlet_name": "", "ends_at_leaf": True}]
        self.semantic_of_segment = {0: 1}
        self.frame_n = rng.normal(size=(40, 3))
        self.frame_b = rng.normal(size=(40, 3))
        self.tree_rows = np.arange(40)
        self.tree = cKDTree(self.table[:, :3])
        self.provenance = {"step_mm": 0.5}


def _result(seed=0):
    rng = np.random.default_rng(seed)
    case = {"pos": rng.normal(size=(30, 3)).astype(np.float32), "wall_coords_raw": rng.normal(size=(30, 3)),
            "n_wall": 12, "coord_scale_scalar": 3.25, "target_normalization": "global_stats",
            "query_groups": [np.arange(12), np.arange(12, 30)], "bundle_path": "", "flag": True}
    aux = {"geom": {"segment_id": np.zeros(30, np.int16), "wall_mask": np.arange(30) < 12},
           "diag": {"murray_shares": {"0": 1.0}, "spacing_mm": np.float64(0.5), "count": np.int64(7),
                    "caps": ({"label": "inlet", "radius_mm": 9.0},), "missing": None, "nan": float("nan"),
                    "by_segment": {0: 3, 1: 4}},
            "closed_surface": {"vertices": rng.normal(size=(8, 3)), "faces": np.arange(24).reshape(8, 3)}}
    return case, aux


def _same(a, b):
    assert type(a) is type(b)
    if isinstance(a, (np.ndarray, np.generic)):
        assert a.dtype == b.dtype and np.shape(a) == np.shape(b) and np.asarray(a).tobytes() == np.asarray(b).tobytes()
    elif isinstance(a, dict):
        assert list(a) == list(b)
        for key in a:
            _same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    elif isinstance(a, float) and np.isnan(a):
        assert np.isnan(b)
    else:
        assert a == b


def test_pack_round_trip_keeps_types_keys_and_bytes():
    value = _result()
    _same(VC._unpack(VC._pack(value)), value)


def test_unsupported_values_are_rejected_not_stringified():
    with pytest.raises(TypeError):
        VC._pack(({"surface": object()}, {}))
    with pytest.raises(TypeError):
        VC._pack(({"obj": np.array([object()], dtype=object)}, {}))


@pytest.fixture
def counted(monkeypatch):
    calls = []

    def build(wall, vertices, faces, atlas, input_features, **kwargs):
        calls.append(kwargs)
        return _result(kwargs["seed"])

    monkeypatch.setattr(VC.VG, "build_volume_case", build)
    return calls


def _call(cache, *, seed=3, mapping=None, atlas=None):
    wall = np.arange(36, dtype=np.float64).reshape(12, 3)
    return VC.build_volume_case_cached(wall, wall + 1, np.arange(12).reshape(4, 3), atlas or _Atlas(), ["rho", "theta_sin"],
                                       cache=cache, mapping=mapping or {"1": "out-le"}, case_name="input-x",
                                       n_internal=18, seed=seed)


def test_second_run_is_a_hit_with_the_exact_result(tmp_path, counted):
    first = _call(GC.GeometryCache(tmp_path, active=True))
    cache = GC.GeometryCache(tmp_path, active=True)
    second = _call(cache)
    assert len(counted) == 1 and cache.stats["hits"] == ["volume_case"] and not cache.stats["errors"]
    _same(second, first)
    _same(second, _result(3))
    assert [p.name.split("-")[0] for p in (tmp_path / GC.CACHE_DIRNAME).glob("*.npz")] == ["volume_case"]


@pytest.mark.parametrize("change", ["seed", "mapping", "atlas"])
def test_any_input_change_is_a_miss(tmp_path, counted, change):
    _call(GC.GeometryCache(tmp_path, active=True))
    kwargs = {"seed": {"seed": 4}, "mapping": {"mapping": {"1": "out-li"}}, "atlas": {"atlas": _Atlas(seed=1)}}[change]
    cache = GC.GeometryCache(tmp_path, active=True)
    _call(cache, **kwargs)
    assert len(counted) == 2 and cache.stats["misses"] == ["volume_case"]


def test_damaged_payload_is_a_miss_and_recomputed(tmp_path, counted):
    _call(GC.GeometryCache(tmp_path, active=True))
    path = next((tmp_path / GC.CACHE_DIRNAME).glob("volume_case-*.npz"))
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["array_0"] = arrays["array_0"] + 1                              # storage valid, content altered
    np.savez(path, **arrays)
    cache = GC.GeometryCache(tmp_path, active=True)
    result = _call(cache)
    assert len(counted) == 2 and cache.stats["hits"] == [] and cache.stats["misses"] == ["volume_case"]
    assert any("payload" in error for error in cache.stats["errors"])
    _same(result, _result(3))


def test_unkeyable_inputs_and_inactive_cache_compute_directly(tmp_path, counted):
    atlas = _Atlas()
    atlas.provenance = {"path": object()}                                   # cannot be keyed exactly
    cache = GC.GeometryCache(tmp_path, active=True)
    _same(_call(cache, atlas=atlas), _result(3))
    assert cache.stats["misses"] == ["volume_case"] and any("key" in error for error in cache.stats["errors"])
    _call(None)
    _call(GC.GeometryCache(tmp_path, active=False))
    assert len(counted) == 3 and not list((tmp_path / GC.CACHE_DIRNAME).glob("*.npz"))


def test_stale_volume_entries_are_pruned_with_the_other_stage_b_kinds(tmp_path, counted):
    _call(GC.GeometryCache(tmp_path, active=True), seed=1)
    cache = GC.GeometryCache(tmp_path, active=True)
    _call(cache, seed=2)
    assert len(list((tmp_path / GC.CACHE_DIRNAME).glob("volume_case-*.npz"))) == 2
    prune_geometry_cache(cache)
    remaining = list((tmp_path / GC.CACHE_DIRNAME).glob("volume_case-*.npz"))
    assert [p.name for p in remaining] == sorted(cache.used)
