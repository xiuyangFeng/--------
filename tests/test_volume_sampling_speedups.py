"""v0.15.11 volume geometry speed-ups must return exactly what v0.15.10 returned (Tier A)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_volume_geometry import tube  # noqa: E402,F401  (synthetic closed tube + atlas fixture)

from wss_deploy import streamlines, volume_geometry as VG  # noqa: E402


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint32])
def test_unique_edges_equals_row_unique(dtype):
    rng = np.random.default_rng(3)
    faces = rng.integers(0, 500, size=(4000, 3)).astype(dtype)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    expected, counts = np.unique(edges, axis=0, return_counts=True)
    got, got_counts = VG._unique_edges(edges)
    assert got.dtype == expected.dtype and np.array_equal(got, expected)
    assert got_counts.dtype == counts.dtype and np.array_equal(got_counts, counts)


def test_unique_edges_falls_back_for_empty_or_negative_input():
    empty = np.empty((0, 2), np.int64)
    assert VG._unique_edges(empty)[0].shape == (0, 2)
    negative = np.array([[-1, 2], [-1, 2], [0, 1]])
    expected = np.unique(negative, axis=0, return_counts=True)
    got = VG._unique_edges(negative)
    assert np.array_equal(got[0], expected[0]) and np.array_equal(got[1], expected[1])


def test_certified_sampling_equals_the_plain_exact_test(tube, monkeypatch):
    wall, vertices, faces, atlas = tube
    calls = {"vtk": 0}
    original = VG.make_inside_test

    def counting(v, f):
        exact = original(v, f)

        def contains(q):
            calls["vtk"] += len(q)
            return exact(q)

        return contains

    monkeypatch.setattr(VG, "make_inside_test", counting)
    certified_points, certified = VG.sample_internal_points(wall, vertices, faces, atlas, n_internal=300, seed=29)
    certified_calls = calls["vtk"]
    monkeypatch.setattr(streamlines, "ball_certified_inside", lambda inside, *a, **k: inside)
    calls["vtk"] = 0
    plain_points, plain = VG.sample_internal_points(wall, vertices, faces, atlas, n_internal=300, seed=29)
    assert certified_points.tobytes() == plain_points.tobytes()
    assert certified["diag"] == plain["diag"]
    for key in ("vertices", "faces"):
        assert np.array_equal(certified["closed_surface"][key], plain["closed_surface"][key])
    assert certified_calls < calls["vtk"]                       # the certificate spared exact-test calls
