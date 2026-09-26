"""v0.14 morphology speed-ups must not change a single bit: cluster-culled plane test, list-based loop
chaining, and the station-table / lumen-volume cache."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_morphology import NAMES, _atlas, _tube  # noqa: E402

from wss_deploy import geometry_cache as GC, morphology as MO  # noqa: E402


def _three_tubes():
    parts = [_tube(0, 120.0, 8.0, 0.0, 0.0), _tube(1, 30.0, 5.0, -14.0, 120.0), _tube(2, 30.0, 5.0, 14.0, 120.0)]
    vertices, faces, offset = [], [], 0
    for v, f in parts:
        vertices.append(v); faces.append(f + offset); offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def _planes(vertices, n=200, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    out = []
    for i in range(n):
        normal = rng.normal(size=3)
        if i % 5 == 0:
            normal = np.array([0.0, 0.0, 1.0])                  # axis-aligned: planes through vertex rings
        origin = vertices[rng.integers(len(vertices))] if i % 3 == 0 else rng.uniform(lo - 5, hi + 5)
        out.append(MO.plane_frame(origin, normal))
    out.append(MO.plane_frame(hi + 100.0, [1.0, 0.0, 0.0]))       # misses the mesh entirely
    return out


def test_culled_plane_test_returns_the_full_scan_segments_exactly(monkeypatch):
    vertices, faces = _three_tubes()
    mesh = MO.MeshSections(vertices, faces)
    for plane in _planes(vertices):
        monkeypatch.setenv("WSS_DEPLOY_MORPH_PREFILTER", "0")
        full = mesh.plane_contour(plane)
        monkeypatch.setenv("WSS_DEPLOY_MORPH_PREFILTER", "1")
        culled = mesh.plane_contour(plane)
        assert full[0].shape == culled[0].shape and np.array_equal(full[0], culled[0]) and np.array_equal(full[1], culled[1])


def test_sections_and_the_whole_morphology_block_are_identical_with_and_without_the_prefilter(monkeypatch):
    vertices, faces = _three_tubes()
    atlas = _atlas()
    results = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("WSS_DEPLOY_MORPH_PREFILTER", flag)
        mesh = MO.MeshSections(vertices, faces)
        sections = [mesh.section(p["origin"], p["normal"], max_dist=12.0) for p in _planes(vertices, n=60, seed=4)]
        block = MO.compute(vertices, faces, atlas, branch_names=NAMES)
        results[flag] = (json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                      for k, v in s.items() if k != "plane"} for s in sections], sort_keys=True, default=str),
                         json.dumps(block, sort_keys=True))
    assert results["0"] == results["1"]


def _contour_loops_reference(keys):
    """The pre-v0.14 implementation (numpy scalar indexing), kept here as the equivalence oracle."""
    keys = np.asarray(keys, dtype=np.int64)
    n = len(keys)
    by_key = {}
    for i in range(n):
        for k in (int(keys[i, 0]), int(keys[i, 1])):
            by_key.setdefault(k, []).append(i)
    used = bytearray(n)
    loops = []

    def walk(start, exit_key, stop_key, out):
        seg, key = start, exit_key
        for _ in range(n + 1):
            if key == stop_key and seg != start:
                return True
            nxt = None
            for j in by_key.get(key, ()):
                if not used[j]:
                    nxt = j
                    break
            if nxt is None:
                return key == stop_key and seg != start
            used[nxt] = 1
            out.append(nxt)
            seg = nxt
            k0, k1 = int(keys[nxt, 0]), int(keys[nxt, 1])
            key = k1 if k0 == key else k0
        return False

    for s0 in range(n):
        if used[s0]:
            continue
        used[s0] = 1
        members = [s0]
        a, b = int(keys[s0, 0]), int(keys[s0, 1])
        closed = walk(s0, b, a, members)
        if not closed:
            walk(s0, a, b, members)
        loops.append({"members": members, "closed": closed})
    return loops


def test_list_based_loop_chaining_matches_the_reference_walk():
    rng = np.random.default_rng(7)
    vertices, faces = _three_tubes()
    mesh = MO.MeshSections(vertices, faces)
    for plane in _planes(vertices, n=80, seed=9):
        _, keys = mesh.plane_contour(plane)
        assert MO.contour_loops(keys) == _contour_loops_reference(keys)
    for _ in range(50):                                            # arbitrary soups: open chains, shared keys
        keys = rng.integers(0, 12, size=(int(rng.integers(0, 25)), 2))
        assert MO.contour_loops(keys) == _contour_loops_reference(keys)
    assert MO.contour_loops(np.zeros((0, 2), dtype=np.int64)) == []


def test_station_cache_hits_are_the_computed_tables(tmp_path):
    vertices, faces = _three_tubes()
    atlas = _atlas()
    plain = MO.compute(vertices, faces, atlas, branch_names=NAMES)
    cold = GC.GeometryCache(tmp_path)
    first = MO.compute(vertices, faces, atlas, branch_names=NAMES, section_cache=cold)
    warm = GC.GeometryCache(tmp_path)
    second = MO.compute(vertices, faces, atlas, branch_names=NAMES, section_cache=warm)
    dump = lambda block: json.dumps(block, sort_keys=True)
    assert dump(plain) == dump(first) == dump(second)
    assert cold.stats["hits"] == [] and set(cold.stats["writes"]) == {"morph", "morphvol"}
    assert warm.stats["writes"] == [] and warm.stats["misses"].count("morph") == 0 and "morphvol" in warm.stats["hits"]
    # an entry that kept the ring polygons also serves a request without them
    mesh = MO.MeshSections(vertices, faces)
    line = MO._polyline(atlas, 0)
    stations = MO._station_positions(line, 1.0, skip_start=True, margin_mm=MO.OPENING_MARGIN_MM)
    with_rings = MO.cached_branch_stations(mesh, line, stations, station_mm=1.0, keep_polygons=True, cache=GC.GeometryCache(tmp_path))
    probe = GC.GeometryCache(tmp_path)
    without = MO.cached_branch_stations(mesh, line, stations, station_mm=1.0, keep_polygons=False, cache=probe)
    assert "_polygons" not in without and probe.stats["writes"] == []
    assert json.dumps(without, sort_keys=True) == json.dumps({k: v for k, v in with_rings.items() if k != "_polygons"}, sort_keys=True)
    reference = MO.branch_stations(mesh, line, stations, station_mm=1.0, keep_polygons=True)
    assert all((a is None and b is None) or np.array_equal(a, b) for a, b in zip(with_rings["_polygons"], reference["_polygons"]))
    # a different atlas radius (e.g. the end-radius patch) is a different key
    atlas2 = _atlas(radii=(8.5, 5.0, 5.0))
    probe2 = GC.GeometryCache(tmp_path)
    MO.compute(vertices, faces, atlas2, branch_names=NAMES, section_cache=probe2)
    assert "morph" in probe2.stats["writes"]


def test_precompute_sections_fills_what_compute_reads(tmp_path):
    vertices, faces = _three_tubes()
    atlas = _atlas()
    MO.precompute_sections(vertices, faces, atlas, GC.GeometryCache(tmp_path))
    cache = GC.GeometryCache(tmp_path)
    block = MO.compute(vertices, faces, atlas, branch_names=NAMES, section_cache=cache)
    assert cache.stats["writes"] == [] and json.dumps(block, sort_keys=True) == json.dumps(
        MO.compute(vertices, faces, atlas, branch_names=NAMES), sort_keys=True)
