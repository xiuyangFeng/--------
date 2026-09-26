"""Steady streamlines from world-frame interior velocity samples (display only)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

COVERAGE_K = 8            # neighbours used both for IDW and for the local coverage radius
COVERAGE_FACTOR = 1.3     # a query is supported while its nearest sample is within this × that sample's k-NN radius
MIN_LINE_POINTS = 8       # stubs shorter than this are not worth drawing


def ball_certified_inside(inside, samples, vertices, faces, *, margin: float = 0.999):
    """Wrap an exact inside test with a certificate that skips it for points that are provably inside.

    Every interior sample ``s`` lies inside the closed lumen, and the open ball around ``s`` whose radius
    is its distance to the closed surface cannot cross that surface, so any query strictly within such a
    ball is inside too.  Only queries outside every certified ball (near the wall or an opening) are
    handed to ``inside`` — the VTK ray test stays the authority for all of them.  Streamline integration
    asks for ~10^6 mostly deep-lumen points; this removes most of the per-point VTK calls (v0.12.2).
    """
    import pyvista as pv
    samples = np.asarray(samples, dtype=np.float64)
    mesh = pv.PolyData(np.asarray(vertices, dtype=np.float64),
                       np.hstack([np.full((len(faces), 1), 3), np.asarray(faces, dtype=np.int64)]).ravel())
    clearance = np.abs(np.asarray(pv.PolyData(samples).compute_implicit_distance(mesh)["implicit_distance"], dtype=np.float64))
    clearance = clearance * float(margin)
    tree = cKDTree(samples)

    def contains(query):
        q = np.asarray(query, dtype=np.float64).reshape(-1, 3)
        if not len(q):
            return np.zeros(0, dtype=bool)
        d, ix = tree.query(q, k=1)
        sure = d < clearance[ix]
        out = np.zeros(len(q), dtype=bool)
        out[sure] = True
        rest = ~sure
        if rest.any():
            out[rest] = np.asarray(inside(q[rest]), dtype=bool)
        return out

    return contains


def integrate_streamlines(points, velocity, seeds, inside, *, step_mm=0.6,
                          max_steps=650, max_distance_mm=None, min_points=MIN_LINE_POINTS):
    """Bidirectional midpoint integration; stop at wall, gaps, or stagnation.

    Arc-length stepping uses millimetres while speeds remain m/s. This produces
    streamlines of one frozen vector field, never particle paths over time.
    Local inverse-distance interpolation never supplies values outside the
    closed lumen or beyond the recorded sample-coverage distance.

    Coverage is local: the interior samples are denser in narrow branches than
    in an aneurysm sac, so a single global cut-off either bridges gaps in the
    branches or truncates every line in the sac.  With ``max_distance_mm=None``
    a query is supported while its nearest sample lies within
    ``COVERAGE_FACTOR`` times that sample's own k-NN radius; an explicit
    ``max_distance_mm`` is an absolute cut-off (used by the gap test).
    """
    points = np.asarray(points, float); velocity = np.asarray(velocity, float)
    seeds = np.asarray(seeds, float).reshape(-1, 3)
    if points.ndim != 2 or points.shape[1] != 3 or velocity.shape != points.shape or len(points) < 2:
        raise ValueError("Streamlines require aligned interior point and vector arrays")
    if not np.isfinite(points).all() or not np.isfinite(velocity).all() or not np.isfinite(seeds).all():
        raise ValueError("Streamline inputs must be finite")
    if not np.isfinite(step_mm) or step_mm <= 0 or type(max_steps) is not int or max_steps < 1:
        raise ValueError("Streamline step and iteration count must be positive")
    tree = cKDTree(points)
    k = min(COVERAGE_K, len(points))
    if max_distance_mm is None:
        # k-NN radius of every sample = the local sampling scale around it.
        local = tree.query(points, k=k)[0][:, -1] if k > 1 else np.full(len(points), 1.0)
        allowed = np.maximum(COVERAGE_FACTOR * local, 0.5)
        coverage_note = {"mode": "local_knn_radius", "k": int(k), "factor": COVERAGE_FACTOR,
                         "allowed_mm_p10_p50_p90": [float(x) for x in np.percentile(allowed, [10, 50, 90])]}
    else:
        if not np.isfinite(max_distance_mm) or max_distance_mm <= 0:
            raise ValueError("Streamline coverage distance must be positive")
        allowed = np.full(len(points), float(max_distance_mm))
        coverage_note = {"mode": "absolute", "max_distance_mm": float(max_distance_mm)}

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


def centerline_seeds(atlas, *, station_mm: float = 12.0, rings=((0.35, 6), (0.72, 10)),
                     max_seeds: int = 420):
    """Seed rings along every branch: one station every ``station_mm`` of arc length.

    Each station carries the axis point plus concentric rings at fractions of the
    local inscribed radius; long segments therefore get many more seeds than the
    three fixed stations used before, and the sac of an aneurysm is seeded along
    its whole length instead of at 20/55/80 % only.  The total is capped by
    thinning stations uniformly so the report size stays bounded.
    """
    segment = atlas.col("segment_id").astype(int)
    sample_index = atlas.col("sample_index")
    radius_all = atlas.col("radius_mm")
    stations = []
    for sid in np.unique(segment):
        rows = np.flatnonzero(segment == sid)
        rows = rows[np.argsort(sample_index[rows])]
        xyz = atlas.xyz[rows]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))] if len(rows) > 1 else np.zeros(1)
        length = float(arc[-1])
        count = max(3, int(np.floor(length / station_mm)) + 1)
        # keep away from the very ends (caps / junction centres)
        for frac in np.linspace(0.08, 0.92, count):
            target = frac * length
            j = int(np.clip(np.searchsorted(arc, target), 0, len(rows) - 1))
            stations.append(rows[j])
    stations = np.asarray(sorted(set(int(r) for r in stations)))
    per_station = 1 + sum(n for _, n in rings)
    if len(stations) * per_station > max_seeds:
        keep = max(1, max_seeds // per_station)
        stations = stations[np.linspace(0, len(stations) - 1, keep).round().astype(int)]
    seeds = []
    for row in stations:
        center = atlas.xyz[row]; radius = float(radius_all[row])
        n_vec, b_vec = atlas.frame_n[row], atlas.frame_b[row]
        seeds.append(center)
        for frac, count in rings:
            for angle in np.linspace(0, 2 * np.pi, count, endpoint=False):
                seeds.append(center + frac * radius * (np.cos(angle) * n_vec + np.sin(angle) * b_vec))
    return np.asarray(seeds, dtype=np.float64).reshape(-1, 3)


def volume_seeds(points, *, n: int = 120, seed: int = 0):
    """Farthest-point subsample of the interior predictions: covers sacs and recirculation zones.

    Streamlines seeded only from centreline rings follow the main jet; a few seeds
    spread evenly through the sampled volume also start lines inside slow,
    recirculating regions so the sac is not left empty.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points) or n <= 0:
        return np.empty((0, 3))
    rng = np.random.default_rng(int(seed))
    n = min(int(n), len(points))
    chosen = [int(rng.integers(len(points)))]
    distance = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(points - points[nxt], axis=1))
    return points[chosen]


def thin_lines(lines, *, max_points: int = 80000):
    """Subsample line vertices uniformly so the embedded report stays small.

    Integration steps are 0.6 mm; keeping every second or third vertex changes
    nothing visible once the viewer draws smooth tubes through the points.
    """
    total = sum(len(line["points"]) for line in lines)
    if total <= max_points or not lines:
        return lines, 1
    stride = int(np.ceil(total / max_points))
    thinned = []
    for line in lines:
        keep = np.r_[np.arange(0, len(line["points"]) - 1, stride), len(line["points"]) - 1]
        thinned.append({"points": line["points"][keep], "speed_m_s": line["speed_m_s"][keep]})
    return thinned, stride


__all__ = ["centerline_seeds", "integrate_streamlines", "thin_lines", "volume_seeds"]
