"""v0.15.11 streamline reuse: the accepted end point's interpolated field starts the next step.

Tier A: the new integrator must return the same bytes as the v0.15.10 loop (frozen verbatim below)
and must hand ``inside`` the very same call sequence, because the VTK ray test draws its ray
directions from VTK's global random sequence.  Synthetic tube and velocity field, never CFD data.
"""
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from wss_deploy.streamlines import MIN_LINE_POINTS, ball_certified_inside, integrate_streamlines

COVERAGE_K, COVERAGE_FACTOR = 8, 1.3


def _frozen_sampler(points, velocity, allowed, k, inside):
    tree = cKDTree(points)

    def sample(query):
        if not len(query):
            return np.empty((0, 3)), np.empty(0), np.empty(0, bool)
        d, ix = tree.query(query, k=k)
        d = d.reshape(len(query), k); ix = ix.reshape(len(query), k)
        limit = allowed[ix[:, 0]]
        weights = 1. / np.maximum(d, 1e-6)**2
        weights[d > limit[:, None]] = 0
        den = weights.sum(1)
        vectors = (velocity[ix] * weights[:, :, None]).sum(1) / np.maximum(den[:, None], 1e-30)
        speed = np.linalg.norm(vectors, axis=1)
        valid = np.asarray(inside(query), bool) & (d[:, 0] <= limit) & (speed > 1e-5)
        return vectors / np.maximum(speed[:, None], 1e-30), speed, valid

    return tree, sample


def _frozen_integrate(points, velocity, seeds, inside, *, step_mm=0.6,
                      max_steps=650, max_distance_mm=None, min_points=MIN_LINE_POINTS):
    """wss_deploy.streamlines.integrate_streamlines as of v0.15.10 (validation elided: same inputs only)."""
    points = np.asarray(points, float); velocity = np.asarray(velocity, float)
    seeds = np.asarray(seeds, float).reshape(-1, 3)
    tree = cKDTree(points)
    k = min(COVERAGE_K, len(points))
    if max_distance_mm is None:
        local = tree.query(points, k=k)[0][:, -1] if k > 1 else np.full(len(points), 1.0)
        allowed = np.maximum(COVERAGE_FACTOR * local, 0.5)
        coverage_note = {"mode": "local_knn_radius", "k": int(k), "factor": COVERAGE_FACTOR,
                         "allowed_mm_p10_p50_p90": [float(x) for x in np.percentile(allowed, [10, 50, 90])]}
    else:
        allowed = np.full(len(points), float(max_distance_mm))
        coverage_note = {"mode": "absolute", "max_distance_mm": float(max_distance_mm)}
    _, sample = _frozen_sampler(points, velocity, allowed, k, inside)

    paths = []
    for sign in (-1, 1):
        current = seeds.copy()
        records = [[p.copy()] for p in current]
        speeds = [[] for _ in current]
        active = np.arange(len(current))
        for _ in range(max_steps):
            if not len(active): break
            direction, speed, valid = sample(current[active])
            ids = active[valid]
            if not len(ids): break
            for row, val in zip(ids, speed[valid]):
                if not speeds[row]: speeds[row].append(float(val))
            midpoint = current[ids] + sign * .5 * step_mm * direction[valid]
            mid_direction, _, mid_valid = sample(midpoint)
            ids = ids[mid_valid]
            candidate = current[ids] + sign * step_mm * mid_direction[mid_valid]
            _, end_speed, end_valid = sample(candidate)
            active = ids[end_valid]
            for row, point, val in zip(active, candidate[end_valid], end_speed[end_valid]):
                records[row].append(point.copy()); speeds[row].append(float(val))
            current[active] = candidate[end_valid]
        paths.append((records, speeds))
    result = []
    for row in range(len(seeds)):
        left, ls = paths[0][0][row], paths[0][1][row]
        right, rs = paths[1][0][row], paths[1][1][row]
        if not ls or not rs: continue
        xyz = np.asarray(left[:0:-1] + right, np.float32)
        speed = np.asarray(ls[:0:-1] + rs, np.float32)
        if len(xyz) >= max(5, int(min_points)):
            result.append({"points": xyz, "speed_m_s": speed})
    return result, {"method": "steady_midpoint_local_idw", "step_mm": step_mm,
                    "max_steps_each_direction": max_steps, "coverage": coverage_note,
                    "coverage_distance_mm": coverage_note.get("max_distance_mm"),
                    "seed_count": len(seeds), "line_count": len(result),
                    "time_dependent": False, "domain": "closed_lumen"}


RADIUS, HALF = 5.0, 20.0                       # closed cylinder along x


@pytest.fixture(scope="module")
def tube():
    import pyvista as pv
    from wss_deploy.volume_geometry import make_inside_test
    surface = pv.Cylinder(radius=RADIUS, height=2 * HALF, resolution=48).triangulate()
    vertices, faces = np.asarray(surface.points), np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    rng = np.random.default_rng(3)
    raw = np.c_[rng.uniform(-HALF, HALF, 9000), rng.uniform(-RADIUS, RADIUS, (9000, 2))]
    raw = raw[np.hypot(raw[:, 1], raw[:, 2]) < RADIUS * 0.97]
    raw = raw[(raw[:, 0] < 4.0) | (raw[:, 0] > 7.0)]          # an unsampled slab: coverage stops lines there
    samples = raw[:3000]
    y, z = samples[:, 1], samples[:, 2]
    r2 = (y * y + z * z) / RADIUS**2
    # Poiseuille axial flow + swirl + a transverse drift that pushes lines into the wall, and a dead zone.
    velocity = np.c_[0.6 * (1 - r2), -0.25 * z + 0.08, 0.25 * y + 0.05 * np.sin(samples[:, 0])]
    velocity[np.linalg.norm(samples - [-12.0, 2.0, 0.0], axis=1) < 2.0] = 0.0
    inside = ball_certified_inside(make_inside_test(vertices, faces), samples, vertices, faces)
    seeds = np.r_[np.c_[rng.uniform(-HALF, HALF, 70), rng.uniform(-RADIUS, RADIUS, (70, 2)) * 0.9],
                  [[-12.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 9.0], [30.0, 0.0, 0.0], [5.5, 0.0, 0.0]]]
    return samples, velocity, seeds, inside


def _run(fn, *args, rng_seed, **kwargs):
    import vtk
    samples, velocity, seeds, inside = args
    calls = []

    def recording(q):
        calls.append(np.asarray(q, np.float64).tobytes())
        return inside(q)

    vtk.vtkMath.RandomSeed(rng_seed)
    lines, info = fn(samples, velocity, seeds, recording, **kwargs)
    return lines, info, calls, vtk.vtkMath.Random()


def _assert_same(old, new):
    (lo, io, co, ro), (ln, in_, cn, rn) = old, new
    assert json.dumps(io, sort_keys=True) == json.dumps(in_, sort_keys=True)
    assert len(lo) == len(ln)
    for a, b in zip(lo, ln):
        assert set(a) == set(b) == {"points", "speed_m_s"}
        for key in a:
            assert a[key].dtype == b[key].dtype == np.float32 and a[key].shape == b[key].shape
            assert a[key].tobytes() == b[key].tobytes()
    assert co == cn                    # same inside() arguments, same order
    assert ro == rn                    # VTK's global random sequence left at the same position


@pytest.mark.parametrize("kwargs", [
    {},                                                        # production defaults
    {"step_mm": 0.35, "max_steps": 120},                       # step cap reached for many lines
    {"max_distance_mm": 0.8, "max_steps": 400},                # absolute coverage (gap test mode)
    {"min_points": 40, "max_steps": 60},                       # short lines dropped
])
def test_reuse_is_byte_identical_to_the_frozen_loop(tube, kwargs):
    old = _run(_frozen_integrate, *tube, rng_seed=11, **kwargs)
    new = _run(integrate_streamlines, *tube, rng_seed=11, **kwargs)
    assert old[0] and len(old[2]) > 10          # the case actually integrates and stops for several reasons
    _assert_same(old, new)


def test_reuse_edge_cases_match(tube):
    samples, velocity, seeds, inside = tube
    for case in ((samples, velocity * 0, seeds[:5], inside),                   # stagnant everywhere
                 (samples, velocity, np.empty((0, 3)), inside),                # no seeds
                 (samples, velocity, np.array([[30., 0, 0], [0, 9, 0]]), inside)):   # seeds outside the lumen
        _assert_same(_run(_frozen_integrate, *case, rng_seed=5), _run(integrate_streamlines, *case, rng_seed=5))


def test_idw_sample_is_row_wise_pure(tube):
    """The premise of the reuse: a point's interpolated values do not depend on the rest of its batch."""
    samples, velocity, seeds, _ = tube
    tree = cKDTree(samples)
    allowed = np.maximum(COVERAGE_FACTOR * tree.query(samples, k=COVERAGE_K)[0][:, -1], 0.5)
    _, sample = _frozen_sampler(samples, np.asarray(velocity, float), allowed, COVERAGE_K,
                                lambda q: np.ones(len(q), bool))
    rng = np.random.default_rng(8)
    query = np.c_[rng.uniform(-HALF, HALF, 257), rng.uniform(-RADIUS, RADIUS, (257, 2))]
    batch = sample(query)
    subset = np.flatnonzero(rng.random(len(query)) < 0.4)[::-1]
    for rows in [subset, *[[i] for i in range(0, len(query), 17)]]:
        part = sample(query[rows])
        for whole, piece in zip(batch, part):
            assert whole[rows].tobytes() == piece.tobytes()
