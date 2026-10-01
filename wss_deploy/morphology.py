"""Lumen morphology from the wall mesh: per-station cross-sections, aneurysm sac / neck, volumes.

Contract: ``wss_deploy/ANALYSIS_CONTRACT.md`` §17.1.  Every number here is derived from the clean
STL and the confirmed centreline atlas; no prediction value is read or changed and nothing is
written to ``field.npz``.  The cross-section algorithm is a numpy port of the report shared library
(``static/report_common.js``: ``planeContour`` → ``contourLoops`` → ``selectLoop`` → ``closeChain``
→ ``loopPolygon`` → ``sectionMetrics``) and keeps the same semantics so a station measured here and
a diameter measured by hand in the 3-D report agree.
"""
from __future__ import annotations

import math
import os
from typing import Any, Mapping, Sequence

import numpy as np

MORPHOLOGY_SCHEMA = "wss-deploy.morphology/v1"
STATION_MM = 1.0
OPENING_MARGIN_MM = 2.0      # stations closer than this to a branch opening would cut the cap
SAC_FACTOR = 1.5             # AAA definition: ≥ 1.5 × the normal (reference) diameter
NECK_FACTOR = 1.2            # proximal neck: still close to the reference diameter
REFERENCE_PERCENTILE = 10.0  # robust "normal calibre" estimate of the aorta
MIN_MAX_DIST_MM = 8.0        # selectLoop search radius = max(4 × local radius, this)
MAX_DIST_RADII = 4.0
FERET_ANGLES = 90            # min Feret is the smallest width over this many orientations
OBLIQUITY_FACTOR = 1.6       # min-Feret / inscribed-diameter above which the plane is grazing the wall
ELONGATION_FACTOR = 2.2      # max-Feret / min-Feret above which the contour is a sliver, not a lumen
SOLIDITY_FLOOR = 0.8         # polygon area / convex-hull area below which the contour is a keyhole
REORIENT_TILTS_DEG = (15.0, 30.0, 45.0)   # cone searched around the tangent for a suspect station
REORIENT_AZIMUTHS = 8
MAX_POLYGON_WORLD = 200      # points kept for the ring drawn by the reports
# v0.14 exact spatial prefilter (``MeshSections``): faces are grouped into Morton-ordered clusters of
# CLUSTER_FACES with a bounding sphere; a plane only side-tests the faces of clusters its slab reaches.
CLUSTER_FACES = 64
BRANCH_ORDER = ("主动脉", "左髂总", "左髂外", "左髂内", "右髂总", "右髂外", "右髂内")
AORTA_NAME = "主动脉"


def prefilter_enabled() -> bool:
    return os.environ.get("WSS_DEPLOY_MORPH_PREFILTER", "1").strip().lower() not in {"0", "false", "off", "no"}


METHOD = {
    "section": "中心线每 1 mm 一站，以局部切线为法向取壁面网格交线的本地闭合环（穿开口时直线封口）",
    "max_diameter": "环上最大 Feret 直径",
    "equivalent_diameter": "2·sqrt(面积/π)",
    "reference_diameter": "主动脉等效直径的第 10 百分位（正常管径的稳健估计）",
    "sac": "等效直径 ≥ 1.5 × 参考直径的连续区段（AAA 常用定义；按管腔判定，不含附壁血栓与管壁）",
    "neck": "入口到瘤体起点之间、等效直径 < 1.2 × 参考直径的近端区段（按管腔判定）",
    "lumen": "输入是管腔面：所有直径、长度和体积都是管腔的，不含附壁血栓与管壁，通常小于 CT 报告的瘤体直径",
    "volume": "开口用扇形封盖后散度定理求全腔体积；瘤体体积 = 瘤体区段截面面积沿弧长积分",
    "inscribed_diameter": "中心线该处最大内切球直径（2 × atlas radius_mm），用于判断截面是否斜切",
    "obliquity": "截面最小宽度 > 1.6 × 内切直径时判为斜切（多见于瘤体肩部与分叉近端），其直径偏大",
    "reliability": "可疑站 = 斜切、或最大/最小 Feret > 2.2（细长切片）、或面积/凸包面积 < 0.8（钥匙孔形，多见于分叉处切到母血管）",
    "reorientation": "可疑站在同一中心线点上绕切线 15°/30°/45° × 8 方位共 24 个候选法向重切，取闭合截面面积最小者（最接近垂直），原值留在 raw_max_diameter_mm",
    "statistics": "参考直径、管腔最大直径、瘤体、瘤颈与分支直径统计只用可靠站（闭合且重定向后仍不可疑）",
}


# ----------------------------------------------------------------------------- small helpers
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _finite(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else np.array([0.0, 0.0, 1.0])


def plane_frame(origin, normal) -> dict:
    """Same frame as ``report_common.planeFrame``: u = ref × n, v = n × u (ref = z unless n ≈ z)."""
    n = _unit(normal)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(ref, n))
    v = _unit(np.cross(n, u))
    return {"origin": np.asarray(origin, dtype=np.float64).reshape(3), "normal": n, "u": u, "v": v}


class MeshSections:
    """Pre-processed wall mesh: per-face vertex indices reused by every plane.

    ``section(origin, normal, max_dist)`` returns the local lumen contour of one station.
    """

    def __init__(self, vertices, faces):
        self.vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64).reshape(-1, 3))
        self.faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int64).reshape(-1, 3))
        if len(self.vertices) == 0 or len(self.faces) == 0:
            raise ValueError("morphology needs a non-empty triangle mesh")
        self.n_vertices = int(len(self.vertices))
        # Per-face vertex indices, gathered once: a plane test is then one matrix-vector product,
        # one gather and two compares over the face table.
        self._faces_flat = np.ascontiguousarray(self.faces.reshape(-1))
        # Exactly one of a crossed triangle's three cyclic edges does not cross; this maps that
        # edge's index to the two that do, in the JS order (ab, bc, ca).
        self._other_edges = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        self._index = None
        self._digest = None

    def digest(self) -> str:
        """SHA256 of the mesh as used here (float64 vertices, int64 faces): part of every section cache key."""
        if self._digest is None:
            from .geometry_cache import key_of
            self._digest = key_of("morphology-mesh", self.vertices, self.faces)
        return self._digest

    # ---------------------------------------------------------------- v0.14 spatial index
    def _spatial_index(self) -> dict:
        """Morton-ordered face clusters (CLUSTER_FACES each) with bounding spheres, built once per mesh.

        Only used to skip side tests whose outcome is already known; see :meth:`plane_contour`.
        """
        if self._index is not None:
            return self._index
        V, F = self.vertices, self.faces
        tri = V[F]                                                     # (F, 3, 3)
        centroid = tri.mean(axis=1)
        lo, hi = V.min(axis=0), V.max(axis=0)
        extent = float(max(np.max(hi - lo), 1e-9))
        cells = np.clip(((centroid - lo) / extent * 1023.0).astype(np.int64), 0, 1023)
        code = np.zeros(len(F), dtype=np.int64)
        for bit in range(10):
            for axis in range(3):
                code |= ((cells[:, axis] >> bit) & 1) << (3 * bit + axis)
        order = np.argsort(code, kind="stable")
        n_clusters = -(-len(F) // CLUSTER_FACES)
        padded = np.concatenate([order, np.full(n_clusters * CLUSTER_FACES - len(F), order[-1])])
        members = padded.reshape(n_clusters, CLUSTER_FACES)
        points = tri[members].reshape(n_clusters, CLUSTER_FACES * 3, 3)
        center = 0.5 * (points.min(axis=1) + points.max(axis=1))
        radius = np.sqrt(((points - center[:, None, :]) ** 2).sum(axis=2)).max(axis=1)
        # Tolerance far above the float64 rounding of a plane distance (~1e-13 of the coordinates) and
        # far below any geometric scale: a cluster whose centre is farther than radius + tol from the
        # plane has every vertex strictly on one side, whatever the rounding of V @ n.
        tol = 1e-6 * (1.0 + extent + float(np.abs(V).max()))
        self._index = {"n_faces": len(F), "members": members, "center": center, "radius": radius + tol}
        return self._index

    # ---------------------------------------------------------------- plane ∩ mesh
    def plane_contour(self, plane: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Crossing segments of one plane, in-plane (x0, y0, x1, y1) plus their two mesh-edge keys.

        Port of ``planeContour`` with ``maxRadius = Infinity``: a triangle contributes when its
        three vertices are not all on the same side of the plane, and then exactly two of its
        edges cross.

        v0.14: the per-vertex signed distances are computed exactly as before over all vertices, but
        the per-face side test only visits the Morton clusters whose bounding sphere reaches the plane
        (a cluster farther than its radius + tolerance has every vertex strictly on one side, so none
        of its faces can cross).  The crossing faces, their order and every coordinate are unchanged.
        ``WSS_DEPLOY_MORPH_PREFILTER=0`` tests every face as before.
        """
        V, o = self.vertices, plane["origin"]
        n, u, v = plane["normal"], plane["u"], plane["v"]
        d = V @ n - float(o @ n)
        side = np.asarray(d >= 0.0)
        if prefilter_enabled():
            index = self._spatial_index()
            near = np.abs(index["center"] @ n - float(o @ n)) <= index["radius"]
            mask = np.zeros(index["n_faces"], dtype=bool)
            mask[index["members"][near]] = True
            candidates = np.flatnonzero(mask)                         # ascending face order, as before
            face_side = side[self.faces[candidates]]
            first = face_side[:, 0]
            crossing = (face_side[:, 1] != first) | (face_side[:, 2] != first)
            hit = candidates[crossing]
            sh = face_side[crossing]
        else:
            face_side = side[self._faces_flat].reshape(-1, 3)
            first = face_side[:, 0]
            hit = np.flatnonzero((face_side[:, 1] != first) | (face_side[:, 2] != first))
            sh = face_side[hit]
        if len(hit) == 0:
            return np.zeros((0, 4), dtype=np.float64), np.zeros((0, 2), dtype=np.int64)
        cross = sh != sh[:, [1, 2, 0]]                                  # edges (0,1), (1,2), (2,0)
        order = self._other_edges[np.argmin(cross, axis=1)]             # the two crossed edges
        faces_hit = self.faces[hit]
        rows = np.arange(len(hit), dtype=np.int64)[:, None]
        a = faces_hit[rows, order]
        b = faces_hit[rows, (order + 1) % 3]
        da, db = d[a], d[b]
        denom = da - db
        t = np.divide(da, denom, out=np.zeros_like(da), where=denom != 0.0)
        q = V[a] + (V[b] - V[a]) * t[..., None] - o                     # (M, 2, 3)
        x = q @ u
        y = q @ v
        segs = np.stack([x[:, 0], y[:, 0], x[:, 1], y[:, 1]], axis=1)
        keys = np.minimum(a, b).astype(np.int64) * self.n_vertices + np.maximum(a, b).astype(np.int64)
        return segs, keys

    def section(self, origin, normal, *, max_dist: float | None = None) -> dict:
        """One cross-section: local lumen polygon, its metrics and the world-space ring."""
        plane = plane_frame(origin, normal)
        segs, keys = self.plane_contour(plane)
        empty = {"plane": plane, "polygon": np.zeros((0, 2)), "polygon_world": np.zeros((0, 3)),
                 "metrics": None, "closed": False, "synthetic": False, "open": False, "found": False,
                 "dmin": None, "n_segments": 0}
        if len(segs) == 0:
            return empty
        limit = float(max_dist) if max_dist and math.isfinite(max_dist) and max_dist > 0 else math.inf
        loops = contour_loops(keys)
        best = select_loop(loops, segs, limit)
        if best is None:
            return empty
        contour = segs[best["members"]]
        contour_keys = keys[best["members"]]
        synthetic, is_open = False, False
        if not best["closed"]:
            closed = close_chain(contour, contour_keys)
            if closed is None:
                is_open = True
            else:
                contour, synthetic = closed, True
        polygon = loop_polygon(contour)
        metrics = None if is_open else section_metrics(polygon)
        world = (plane["origin"][None, :] + polygon[:, 0:1] * plane["u"][None, :]
                 + polygon[:, 1:2] * plane["v"][None, :]) if len(polygon) else np.zeros((0, 3))
        return {"plane": plane, "polygon": polygon, "polygon_world": world, "metrics": metrics,
                "closed": bool(best["closed"] or synthetic), "synthetic": synthetic, "open": is_open,
                "found": True, "dmin": best["dmin"], "n_segments": int(len(contour))}


# ----------------------------------------------------------------------------- loops / polygon / metrics
def contour_loops(keys: np.ndarray) -> list[dict]:
    """Chain crossing segments that share a mesh edge into loops (port of ``contourLoops``)."""
    keys = np.asarray(keys, dtype=np.int64)
    n = len(keys)
    # v0.14: the keys as Python ints once (numpy scalar indexing dominated the walk); same algorithm,
    # same insertion order, same members in the same order.
    pairs = keys.reshape(-1, 2).tolist() if n else []
    by_key: dict[int, list[int]] = {}
    for i, (k0, k1) in enumerate(pairs):
        by_key.setdefault(k0, []).append(i)
        by_key.setdefault(k1, []).append(i)
    used = bytearray(n)
    loops: list[dict] = []

    def walk(start: int, exit_key: int, stop_key: int, out: list[int]) -> bool:
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
            k0, k1 = pairs[nxt]
            key = k1 if k0 == key else k0
        return False

    for s0 in range(n):
        if used[s0]:
            continue
        used[s0] = 1
        members = [s0]
        a, b = pairs[s0]
        closed = walk(s0, b, a, members)
        if not closed:
            walk(s0, a, b, members)
        loops.append({"members": members, "closed": closed})
    return loops


def select_loop(loops: Sequence[Mapping[str, Any]], segs: np.ndarray, max_dist: float) -> dict | None:
    """The loop describing the lumen at the plane origin (port of ``selectLoop``, origin = (0, 0)).

    Preference: a closed loop containing the origin (smallest bounding box = innermost), else an
    open chain wrapping more than 180° around it, else the nearest loop within ``max_dist``.
    """
    limit = float(max_dist) if math.isfinite(max_dist) and max_dist > 0 else math.inf
    best = None
    multiple = len(loops) > 1

    def rank(cand: Mapping[str, Any]) -> int:
        return 3 if cand["inside"] else (2 if cand["wraps"] else 1)

    for loop in loops:
        members = loop["members"]
        if len(members) < 3 and multiple:
            continue
        ss = segs[members]
        x0, y0, x1, y1 = ss[:, 0], ss[:, 1], ss[:, 2], ss[:, 3]
        straddle = (y0 <= 0.0) != (y1 <= 0.0)
        crossings = 0
        if straddle.any():
            dy = (y1 - y0)[straddle]
            xi = x0[straddle] + (x1 - x0)[straddle] * np.divide(-y0[straddle], dy, out=np.zeros_like(dy), where=dy != 0.0)
            crossings = int(np.count_nonzero(xi > 0.0))
        dmin = float(min(np.hypot(x0, y0).min(), np.hypot(x1, y1).min()))
        minx = float(min(x0.min(), x1.min())); maxx = float(max(x0.max(), x1.max()))
        miny = float(min(y0.min(), y1.min())); maxy = float(max(y0.max(), y1.max()))
        span = 2.0 * math.pi
        if not loop["closed"]:
            angles = np.sort(np.arctan2(0.5 * (y0 + y1), 0.5 * (x0 + x1)))
            if len(angles):
                nxt = np.concatenate([angles[1:], [angles[0] + 2.0 * math.pi]])
                span = 2.0 * math.pi - float(np.max(nxt - angles))
        inside = bool(loop["closed"] and crossings % 2 == 1)
        if not inside and dmin > limit:
            continue
        cand = {"members": members, "inside": inside, "dmin": dmin, "closed": bool(loop["closed"]),
                "area": (maxx - minx) * (maxy - miny), "span": span,
                "wraps": bool(not loop["closed"] and span > math.pi)}
        if best is None:
            best = cand
            continue
        if rank(cand) > rank(best):
            best = cand
        elif rank(cand) == rank(best):
            if cand["inside"]:
                better = cand["area"] < best["area"]
            elif cand["wraps"]:
                better = (cand["span"] > best["span"] + 1e-6
                          or (abs(cand["span"] - best["span"]) <= 1e-6 and cand["dmin"] < best["dmin"]))
            else:
                better = cand["dmin"] < best["dmin"]
            if better:
                best = cand
    return best


def close_chain(segs: np.ndarray, keys: np.ndarray) -> np.ndarray | None:
    """Close an open chain with one straight segment when the gap is ≤ 0.6 × the chain length."""
    counts: dict[int, int] = {}
    for i in range(len(keys)):
        for k in (int(keys[i, 0]), int(keys[i, 1])):
            counts[k] = counts.get(k, 0) + 1
    ends = []
    for i in range(len(keys)):
        if counts.get(int(keys[i, 0])) == 1:
            ends.append((float(segs[i, 0]), float(segs[i, 1])))
        if counts.get(int(keys[i, 1])) == 1:
            ends.append((float(segs[i, 2]), float(segs[i, 3])))
    if len(ends) != 2:
        return None
    length = float(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]).sum())
    gap = math.hypot(ends[1][0] - ends[0][0], ends[1][1] - ends[0][1])
    if not gap <= max(0.6 * length, 1e-6):
        return None
    extra = np.array([[ends[0][0], ends[0][1], ends[1][0], ends[1][1]]], dtype=np.float64)
    return np.concatenate([segs, extra], axis=0)


NEIGHBOUR_CANDIDATES = 6   # nearest endpoints pre-listed per endpoint; a full scan covers the rest


def loop_polygon(segs: np.ndarray) -> np.ndarray:
    """Ordered in-plane polygon from a loop's segment soup (port of ``loopPolygon``).

    Segments are chained by repeatedly taking the nearest unused endpoint to the current tail.
    Endpoints are interleaved as ``[seg0.p0, seg0.p1, seg1.p0, …]`` — exactly the JS scan order —
    and a KD-tree lists each endpoint's nearest few neighbours once, so a step is a short scan of
    that list (the first still-unused entry is the global nearest unused endpoint, because every
    endpoint outside the list is farther than all of them).  When the whole list is used up the
    step falls back to a full scan, which keeps the result identical to the plain O(n²) walk.
    """
    from scipy.spatial import cKDTree
    segs = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
    n = len(segs)
    if n == 0:
        return np.zeros((0, 2))
    ext = max(1.0, float(np.max(np.abs(segs))))
    limit = max(ext * 1e-5, ext * 0.05)
    ends = np.empty((2 * n, 2), dtype=np.float64)
    ends[0::2] = segs[:, 0:2]
    ends[1::2] = segs[:, 2:4]
    width = min(2 * n, NEIGHBOUR_CANDIDATES)
    distance, neighbour = cKDTree(ends).query(ends, k=width)
    distance = np.atleast_2d(distance).reshape(2 * n, width).tolist()
    neighbour = np.atleast_2d(neighbour).reshape(2 * n, width).tolist()
    alive = bytearray(b"\x01" * (2 * n))
    alive[0] = alive[1] = 0
    remaining = n - 1
    points = [ends[0].copy(), ends[1].copy()]
    tail_index, tail = 1, ends[1].copy()
    while remaining > 0:
        best, best_distance = -1, math.inf
        for candidate, gap in zip(neighbour[tail_index], distance[tail_index]):
            if alive[candidate]:
                best, best_distance = candidate, gap
                break
        if best < 0:
            delta = ends - tail
            row = np.where(np.frombuffer(alive, dtype=np.uint8).astype(bool),
                           delta[:, 0] ** 2 + delta[:, 1] ** 2, np.inf)
            best = int(np.argmin(row))
            best_distance = math.sqrt(float(row[best]))
        if best_distance > limit:
            break
        j = best // 2
        tail_index = 2 * j + (0 if best % 2 else 1)   # matched p1 -> continue from p0, and vice versa
        tail = ends[tail_index].copy()
        points.append(tail)
        alive[2 * j] = alive[2 * j + 1] = 0
        remaining -= 1
    if len(points) > 2 and math.hypot(points[0][0] - tail[0], points[0][1] - tail[1]) <= limit:
        points.pop()
    return np.asarray(points, dtype=np.float64)


def section_metrics(polygon: np.ndarray) -> dict | None:
    """Area, perimeter, centroid, max/min Feret and equivalent diameter of a closed polygon."""
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    n = len(poly)
    if n < 3:
        return None
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    w = x * yn - xn * y
    a2 = float(w.sum())
    area = abs(a2) / 2.0
    if a2 != 0.0:
        cx = float(((x + xn) * w).sum() / (3.0 * a2))
        cy = float(((y + yn) * w).sum() / (3.0 * a2))
    else:
        cx, cy = float(x.mean()), float(y.mean())
    perimeter = float(np.hypot(xn - x, yn - y).sum())
    hull, solidity = poly, None
    try:
        from scipy.spatial import ConvexHull
        shell = ConvexHull(poly)
        hull = poly[shell.vertices]                    # max Feret is attained on the hull
        # ``volume`` of a 2-D hull is its area; a lumen fills its hull, a keyhole does not.
        solidity = float(area / shell.volume) if shell.volume > 0 else 0.0
    except Exception:
        hull, solidity = poly, 0.0 if n >= 3 else None
    diff = hull[:, None, :] - hull[None, :, :]
    dmax = float(np.sqrt((diff * diff).sum(axis=2)).max())
    angles = np.arange(FERET_ANGLES) * (math.pi / FERET_ANGLES)
    proj = np.stack([np.cos(angles), np.sin(angles)], axis=0)      # (2, K)
    widths = poly @ proj
    dmin = float((widths.max(axis=0) - widths.min(axis=0)).min())
    return {"area_mm2": area, "perimeter_mm": perimeter, "centroid": [cx, cy],
            "max_diameter_mm": dmax, "min_diameter_mm": dmin,
            "equivalent_diameter_mm": 2.0 * math.sqrt(area / math.pi),
            "circularity": (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0,
            "solidity": solidity, "n_points": int(n)}


# ----------------------------------------------------------------------------- lumen volume
def _boundary_loops(faces: np.ndarray) -> list[list[int]] | None:
    """Vertex loops of the open boundary (edges used by exactly one face); None when not chainable."""
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    keyed = np.sort(edges, axis=1)
    uniq, inverse, counts = np.unique(keyed, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    if counts.max(initial=0) > 2:
        return None
    single = counts == 1
    if not single.any():
        return []
    border = edges[single[inverse]]
    adjacency: dict[int, list[int]] = {}
    for a, b in border.tolist():
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if any(len(v) != 2 for v in adjacency.values()):
        return None
    seen: set[int] = set()
    loops: list[list[int]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        loop, node, previous = [start], start, None
        seen.add(start)
        while True:
            nxt = [v for v in adjacency[node] if v != previous]
            if not nxt:
                return None
            step = nxt[0]
            if step == start:
                break
            if step in seen:
                return None
            seen.add(step)
            loop.append(step)
            previous, node = node, step
            if len(loop) > len(adjacency):
                return None
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def lumen_volume_ml(vertices, faces) -> tuple[float | None, str | None]:
    """Enclosed volume of the wall mesh after fan-capping every opening at its loop centroid.

    Returns ``(millilitres, note)``; ``note`` explains why the value is ``None``.
    """
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    try:
        loops = _boundary_loops(faces)
    except (MemoryError, ValueError):
        loops = None
    if loops is None:
        return None, "壁面网格的开口边界不可闭合（非流形或边界断开），未给出全腔体积"
    tri = vertices[faces]
    volume = float(np.einsum("ij,ij->i", np.cross(tri[:, 0], tri[:, 1]), tri[:, 2]).sum()) / 6.0
    for loop in loops:
        ring = vertices[np.asarray(loop, dtype=np.int64)]
        centre = ring.mean(axis=0)
        # The existing face supplies the directed edge a→b on the boundary; the cap supplies b→a.
        b = ring
        a = np.roll(ring, -1, axis=0)
        volume += float(np.einsum("ij,ij->i", np.cross(b, a), np.broadcast_to(centre, b.shape)).sum()) / 6.0
    return abs(volume) / 1000.0, None


# ----------------------------------------------------------------------------- stations
def _branch_name(branch_names: Mapping[str, str] | None, sid: int) -> str:
    name = (branch_names or {}).get(str(int(sid)))
    return str(name) if isinstance(name, str) and name else f"分支 {int(sid)}"


def _ordered_segments(atlas, branch_names: Mapping[str, str] | None) -> list[dict]:
    def key(segment):
        name = _branch_name(branch_names, int(segment["segment_id"]))
        order = BRANCH_ORDER.index(name) if name in BRANCH_ORDER else len(BRANCH_ORDER)
        return (order, int(segment["segment_id"]))
    return sorted(list(atlas.segments), key=key)


def _polyline(atlas, sid: int) -> dict | None:
    """Atlas rows of one segment in path order plus the arrays the stations need."""
    seg = np.asarray(atlas.col("segment_id")).astype(int)
    rows = np.flatnonzero(seg == int(sid))
    if len(rows) < 2:
        return None
    rows = rows[np.argsort(np.asarray(atlas.col("sample_index"))[rows], kind="stable")]
    s_local = np.asarray(atlas.col("s_local_mm"), dtype=np.float64)[rows]
    order = np.argsort(s_local, kind="stable")
    rows, s_local = rows[order], s_local[order]
    keep = np.concatenate([[True], np.diff(s_local) > 1e-9])
    rows, s_local = rows[keep], s_local[keep]
    if len(rows) < 2:
        return None
    return {"rows": rows, "s_local": s_local,
            "s_root": np.asarray(atlas.col("s_from_root_mm"), dtype=np.float64)[rows],
            "radius": np.asarray(atlas.col("radius_mm"), dtype=np.float64)[rows],
            "xyz": np.asarray(atlas.xyz, dtype=np.float64)[rows]}


def _interp_xyz(line: Mapping[str, Any], s: float) -> np.ndarray:
    s_local, xyz = line["s_local"], line["xyz"]
    return np.array([np.interp(s, s_local, xyz[:, k]) for k in range(3)])


def _station_positions(line: Mapping[str, Any], station_mm: float, *, skip_start: bool,
                       margin_mm: float) -> np.ndarray:
    s_local = line["s_local"]
    lo = float(s_local[0]) + (margin_mm if skip_start else 0.0)
    hi = float(s_local[-1]) - margin_mm
    if hi < lo:
        return np.zeros(0)
    first = math.ceil(lo / station_mm - 1e-9)
    last = math.floor(hi / station_mm + 1e-9)
    if last < first:
        return np.zeros(0)
    return np.arange(first, last + 1, dtype=np.float64) * station_mm


def station_flags(metrics: Mapping[str, Any] | None, inscribed_diameter: float) -> dict:
    """Why a closed section cannot be trusted as a lumen cross-section.

    ``oblique``  the plane grazes the wall: its smallest width far exceeds the centreline's
                 maximal inscribed sphere (aneurysm shoulders, just distal to a bifurcation).
    ``elongated`` a sliver rather than a lumen: max Feret is more than 2.2 × min Feret.
    ``low_solidity`` a keyhole: the contour does not fill its convex hull, which happens when the
                 plane catches the parent lumen through the junction as well as the branch.
    """
    if not isinstance(metrics, Mapping):
        return {"oblique": False, "elongated": False, "low_solidity": False, "suspect": False}
    dmax = _finite(metrics.get("max_diameter_mm")) or 0.0
    dmin = _finite(metrics.get("min_diameter_mm")) or 0.0
    solidity = _finite(metrics.get("solidity"))
    oblique = bool(inscribed_diameter > 0 and dmin > OBLIQUITY_FACTOR * inscribed_diameter)
    elongated = bool(dmin > 0 and dmax / dmin > ELONGATION_FACTOR)
    low_solidity = bool(solidity is not None and solidity < SOLIDITY_FLOOR)
    return {"oblique": oblique, "elongated": elongated, "low_solidity": low_solidity,
            "suspect": bool(oblique or elongated or low_solidity)}


def _cone_normals(tangent: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """(tilt°, normal) candidates on a cone around ``tangent``; the tangent itself is tilt 0."""
    axis = _unit(tangent)
    u = _unit(np.cross(axis, np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])))
    v = _unit(np.cross(axis, u))
    out = [(0.0, axis)]
    for tilt in REORIENT_TILTS_DEG:
        angle = math.radians(tilt)
        rim = math.sin(angle)
        for k in range(REORIENT_AZIMUTHS):
            phi = 2.0 * math.pi * k / REORIENT_AZIMUTHS
            out.append((tilt, _unit(math.cos(angle) * axis + rim * (math.cos(phi) * u + math.sin(phi) * v))))
    return out


def reorient_section(mesh: "MeshSections", origin, tangent, *, max_dist: float,
                     baseline: Mapping[str, Any] | None = None) -> tuple[dict, float]:
    """Re-cut a suspect station: the closed candidate of smallest area is the most perpendicular one.

    A plane that grazes the wall always produces a larger section than the locally perpendicular
    one, so minimising the closed section area over a cone of normals recovers the true calibre.
    Only suspect stations pay for this (24 extra plane cuts).
    """
    best, best_tilt = None, 0.0
    best_area = math.inf
    if baseline is not None and baseline.get("metrics"):
        best, best_tilt = baseline, 0.0
        best_area = float(baseline["metrics"]["area_mm2"])
    for tilt, normal in _cone_normals(tangent):
        if tilt == 0.0 and baseline is not None:
            continue
        candidate = mesh.section(origin, normal, max_dist=max_dist)
        metrics = candidate["metrics"] if candidate["closed"] else None
        if metrics is None:
            continue
        area = float(metrics["area_mm2"])
        if area < best_area:
            best, best_tilt, best_area = candidate, float(tilt), area
    return best, best_tilt


def branch_stations(mesh: MeshSections, line: Mapping[str, Any], stations: np.ndarray, *,
                    station_mm: float, keep_polygons: bool = False) -> dict:
    """Cross-section every station of one branch; the tangent is a central difference of the polyline.

    A station whose contour looks unreliable (see :func:`station_flags`) is re-cut over a cone of
    normals and keeps the smallest closed section; ``reliable`` marks the stations the statistics
    may use, ``raw_max_diameter_mm`` preserves what the plain tangent plane measured.
    """
    out: dict[str, list] = {"s_local_mm": [], "s_from_root_mm": [], "max_diameter_mm": [], "min_diameter_mm": [],
                            "equivalent_diameter_mm": [], "area_mm2": [], "closed": [], "xyz_mm": [],
                            "inscribed_diameter_mm": [], "reliable": [], "reoriented": [], "tilt_deg": [],
                            "raw_max_diameter_mm": []}
    s_local, radius, s_root = line["s_local"], line["radius"], line["s_root"]
    half = float(station_mm)

    def one(s):
        origin = _interp_xyz(line, float(s))
        back = _interp_xyz(line, max(float(s) - half, float(s_local[0])))
        ahead = _interp_xyz(line, min(float(s) + half, float(s_local[-1])))
        tangent = ahead - back
        if float(np.linalg.norm(tangent)) <= 1e-12:
            tangent = line["xyz"][-1] - line["xyz"][0]
        r = float(np.interp(float(s), s_local, radius))
        inscribed = 2.0 * r
        max_dist = max(MAX_DIST_RADII * r, MIN_MAX_DIST_MM)
        section = mesh.section(origin, tangent, max_dist=max_dist)
        metrics = section["metrics"] if section["closed"] else None
        raw_max = _finite(metrics["max_diameter_mm"]) if metrics else None
        flags = station_flags(metrics, inscribed)
        tilt = 0.0
        if flags["suspect"]:
            section, tilt = reorient_section(mesh, origin, tangent, max_dist=max_dist, baseline=section)
            metrics = section["metrics"] if section["closed"] else None
            flags = station_flags(metrics, inscribed)
        return s, origin, inscribed, metrics, flags, tilt, raw_max, section

    # Serial on purpose: a thread pool over stations was measured slower (13.8 s → 21.5 s on a 273k-face case),
    # the contour walking is pure Python and holds the GIL.
    results = [one(s) for s in stations]
    polygons: list = []
    for s, origin, inscribed, metrics, flags, tilt, raw_max, section in results:
        out["s_local_mm"].append(float(s))
        out["s_from_root_mm"].append(float(np.interp(float(s), s_local, s_root)))
        out["xyz_mm"].append([float(v) for v in origin])
        out["inscribed_diameter_mm"].append(inscribed)
        out["closed"].append(bool(metrics is not None))
        out["reliable"].append(bool(metrics is not None and not flags["suspect"]))
        out["reoriented"].append(bool(tilt > 0.0))
        out["tilt_deg"].append(float(tilt))
        out["raw_max_diameter_mm"].append(raw_max)
        out["max_diameter_mm"].append(_finite(metrics["max_diameter_mm"]) if metrics else None)
        out["min_diameter_mm"].append(_finite(metrics["min_diameter_mm"]) if metrics else None)
        out["equivalent_diameter_mm"].append(_finite(metrics["equivalent_diameter_mm"]) if metrics else None)
        out["area_mm2"].append(_finite(metrics["area_mm2"]) if metrics else None)
        polygons.append(section["polygon_world"] if keep_polygons and metrics else None)
    if keep_polygons:
        out["_polygons"] = polygons
    return out


def _array(values) -> np.ndarray:
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _oblique_mask(table: Mapping[str, Any]) -> np.ndarray:
    """Stations whose plane is clearly not perpendicular to the local lumen.

    A perpendicular section can never be much narrower than the centreline's maximal inscribed
    sphere, and is rarely much wider: when the smallest width of the contour exceeds
    ``OBLIQUITY_FACTOR × 2 × radius`` the plane is grazing the wall (typically at an aneurysm
    shoulder or just distal to a bifurcation) and its diameters overstate the lumen.
    """
    dmin = _array(table["min_diameter_mm"])
    inscribed = _array(table["inscribed_diameter_mm"])
    with np.errstate(invalid="ignore"):
        return np.asarray(np.isfinite(dmin) & (inscribed > 0) & (dmin > OBLIQUITY_FACTOR * inscribed))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges of the contiguous True runs of ``mask``."""
    runs, start = [], None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _subsample_polygon(polygon, limit: int = MAX_POLYGON_WORLD) -> list[list[float]]:
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 3)
    if len(poly) == 0:
        return []
    if len(poly) > limit:
        poly = poly[np.linspace(0, len(poly) - 1, limit).round().astype(np.int64)]
    return [[round(float(c), 3) for c in row] for row in poly]


# ----------------------------------------------------------------------------- v0.14 section cache
def _code_hash() -> str:
    """Source hash of this module (memoised by geometry_cache; first taken when the module is imported)."""
    from .geometry_cache import source_hash
    return source_hash("wss_deploy.morphology")


def _section_key(mesh: MeshSections, line: Mapping[str, Any], stations: np.ndarray, *, station_mm: float,
                 keep_polygons: bool) -> str:
    """Everything :func:`branch_stations` reads: the mesh, the segment polyline, the stations, the code."""
    from .geometry_cache import key_of
    return key_of("morphology-stations", _code_hash(), mesh.digest(),
                  *(np.asarray(line[k]) for k in ("rows", "s_local", "s_root", "radius", "xyz")),
                  np.asarray(stations, dtype=np.float64), float(station_mm), bool(keep_polygons))


def _encode_table(table: Mapping[str, Any]) -> dict:
    from .geometry_cache import json_array
    polygons = table.get("_polygons")
    arrays = {"table": json_array({k: v for k, v in table.items() if k != "_polygons"})}
    if polygons is not None:
        lengths = np.asarray([-1 if p is None else len(p) for p in polygons], dtype=np.int64)
        parts = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in polygons if p is not None]
        arrays["poly_len"] = lengths
        arrays["poly_xyz"] = np.concatenate(parts) if parts else np.zeros((0, 3))
    return arrays


def _decode_table(arrays: Mapping[str, np.ndarray], keep_polygons: bool) -> dict:
    from .geometry_cache import from_json_array
    table = from_json_array(arrays["table"])
    if keep_polygons:
        lengths, xyz, at, polygons = arrays["poly_len"], arrays["poly_xyz"], 0, []
        for length in lengths.tolist():
            if length < 0:
                polygons.append(None)
            else:
                polygons.append(np.array(xyz[at:at + length], dtype=np.float64)); at += length
        table["_polygons"] = polygons
    return table


def cached_branch_stations(mesh: MeshSections, line: Mapping[str, Any], stations: np.ndarray, *,
                           station_mm: float, keep_polygons: bool = False, cache=None) -> dict:
    """:func:`branch_stations` through the job's geometry cache (``geometry_cache.GeometryCache`` or None).

    The key covers every input of the station computation and the morphology source, so a hit is the
    table the computation would return; an entry that kept the ring polygons also serves a request
    without them.  JSON keeps the Python floats of the table exact.
    """
    if cache is None or not getattr(cache, "active", False):
        return branch_stations(mesh, line, stations, station_mm=station_mm, keep_polygons=keep_polygons)
    for with_polygons in ((False, True) if not keep_polygons else (True,)):
        key = _section_key(mesh, line, stations, station_mm=station_mm, keep_polygons=with_polygons)
        hit = cache.load("morph", key)
        if hit is not None and (not with_polygons or "poly_len" in hit):
            try:
                return _decode_table(hit, keep_polygons)
            except Exception:   # an entry of another layout is a miss, never an error
                pass
    table = branch_stations(mesh, line, stations, station_mm=station_mm, keep_polygons=keep_polygons)
    cache.save("morph", _section_key(mesh, line, stations, station_mm=station_mm, keep_polygons=keep_polygons),
               _encode_table(table), compress=True)
    return table


def cached_lumen_volume(mesh: MeshSections, cache=None) -> tuple[float | None, str | None]:
    """:func:`lumen_volume_ml` of the mesh through the geometry cache (keyed by the mesh and the code)."""
    if cache is None or not getattr(cache, "active", False):
        return lumen_volume_ml(mesh.vertices, mesh.faces)
    from .geometry_cache import from_json_array, json_array, key_of
    key = key_of("morphology-volume", _code_hash(), mesh.digest())
    hit = cache.load("morphvol", key)
    if hit is not None:
        try:
            value = from_json_array(hit["value"])
            return value[0], value[1]
        except Exception:   # an entry of another layout is a miss, never an error
            pass
    volume, note = lumen_volume_ml(mesh.vertices, mesh.faces)
    cache.save("morphvol", key, {"value": json_array([volume, note])})
    return volume, note


def precompute_sections(vertices, faces, atlas, cache, *, station_mm: float = STATION_MM, cancelled=None) -> dict:
    """Fill the geometry cache with every segment's station table (ring polygons for the root segment,
    which is the aorta for every mapping) and the lumen volume; mapping-independent."""
    mesh = MeshSections(vertices, faces)
    done = 0
    for segment in list(atlas.segments):
        if cancelled is not None and cancelled():
            break
        line = _polyline(atlas, int(segment["segment_id"]))
        if line is None:
            continue
        root = bool(segment.get("starts_at_root"))
        stations = _station_positions(line, station_mm, skip_start=root, margin_mm=OPENING_MARGIN_MM)
        cached_branch_stations(mesh, line, stations, station_mm=station_mm, keep_polygons=root, cache=cache)
        done += 1
    if cancelled is None or not cancelled():
        cached_lumen_volume(mesh, cache)
    return {"segments": done}


# ----------------------------------------------------------------------------- §17.1 entry point
def compute(vertices, faces, atlas, *, branch_names: Mapping[str, str] | None = None,
            per_branch: Mapping[str, Any] | None = None, cloud: Mapping[str, Any] | None = None,
            interior: Mapping[str, Any] | None = None, findings: Mapping[str, Any] | None = None,
            thresholds: Sequence[float] | None = None, station_mm: float = STATION_MM,
            section_cache=None) -> dict:
    """``summary["morphology"]`` for either family (contract §17.1).

    ``cloud`` (wall family) is ``{"segment_id": .., "wss": ..}`` of the prediction cloud;
    ``interior`` (volume family) is ``{"segment_id": .., "speed": ..}`` of the interior points;
    ``findings`` supplies the per-branch ΔP of the volume family.  All are optional: a missing
    source leaves the corresponding ``branches[]`` columns ``null``.  ``section_cache`` (v0.14) is the
    job's ``GeometryCache``: station tables and the lumen volume are read from / written to it.
    """
    if not math.isfinite(station_mm) or station_mm <= 0:
        raise ValueError("station_mm must be positive")
    mesh = MeshSections(vertices, faces)
    branch_names = {str(k): v for k, v in (branch_names or {}).items()}
    low, high = (float(thresholds[0]), float(thresholds[1])) if thresholds else (0.4, 4.0)
    notes: list[str] = []
    aorta_sid = None
    for sid, name in branch_names.items():
        if name == AORTA_NAME:
            aorta_sid = int(sid)
            break
    segments = _ordered_segments(atlas, branch_names)
    if aorta_sid is None and segments:
        aorta_sid = int(segments[0]["segment_id"])
        notes.append("未找到名为「主动脉」的分支，按第一条分支统计瘤体形态")

    drops = {}
    for item in ((findings or {}).get("items") or []):
        if isinstance(item, Mapping) and item.get("kind") == "pressure_drop":
            drops[str(item.get("branch"))] = _finite(item.get("value"))

    cloud_seg = np.asarray(cloud["segment_id"]).astype(int) if cloud else None
    cloud_wss = np.asarray(cloud["wss"], dtype=np.float64) if cloud else None
    int_seg = np.asarray(interior["segment_id"]).astype(int) if interior else None
    int_speed = np.asarray(interior["speed"], dtype=np.float64) if interior else None

    branches, aorta_block = [], None
    for segment in segments:
        sid = int(segment["segment_id"])
        name = _branch_name(branch_names, sid)
        line = _polyline(atlas, sid)
        if line is None:
            continue
        is_aorta = sid == aorta_sid
        stations_s = _station_positions(line, station_mm,
                                        skip_start=bool(segment.get("starts_at_root")),
                                        margin_mm=OPENING_MARGIN_MM)
        table = cached_branch_stations(mesh, line, stations_s, station_mm=station_mm, keep_polygons=is_aorta,
                                       cache=section_cache)
        polygons = table.pop("_polygons", None)
        dmax = _array(table["max_diameter_mm"])
        equivalent = _array(table["equivalent_diameter_mm"])
        closed = np.asarray(table["closed"], dtype=bool)
        reliable = np.asarray(table["reliable"], dtype=bool)
        reoriented = np.asarray(table["reoriented"], dtype=bool)
        # Diameter statistics only ever look at reliable stations: an oblique, sliver or keyhole
        # section overstates the calibre by tens of millimetres at shoulders and bifurcations.
        usable = reliable if reliable.any() else closed
        chord = float(np.linalg.norm(line["xyz"][-1] - line["xyz"][0]))
        arc = float(np.linalg.norm(np.diff(line["xyz"], axis=0), axis=1).sum())
        stat = lambda fn, values: _finite(fn(values[usable])) if usable.any() else None
        entry = {"segment_id": sid, "name": name, "length_mm": _finite(segment.get("length_mm")) or arc,
                 "tortuosity": _finite(arc / chord) if chord > 0 else None,
                 "diameter_min_mm": stat(np.nanmin, dmax), "diameter_max_mm": stat(np.nanmax, dmax),
                 "diameter_mean_mm": stat(np.nanmean, dmax),
                 "area_mean_mm2": stat(np.nanmean, _array(table["area_mm2"])),
                 "n_stations": int(len(stations_s)), "n_closed": int(closed.sum()),
                 "n_reliable": int(reliable.sum()), "n_reoriented": int(reoriented.sum()),
                 "n_excluded": int((closed & ~reliable).sum()),
                 "wss_p99_pa": None, "wss_mean_pa": None, "area_frac_low": None, "area_frac_high": None,
                 "delta_p_pa": drops.get(name), "speed_mean_m_s": None, "speed_max_m_s": None,
                 "n_oblique": int(_oblique_mask(table).sum()),
                 "stations": {key: table[key] for key in ("s_from_root_mm", "s_local_mm", "max_diameter_mm",
                                                          "min_diameter_mm", "equivalent_diameter_mm",
                                                          "area_mm2", "closed", "reliable", "reoriented",
                                                          "tilt_deg", "raw_max_diameter_mm", "xyz_mm",
                                                          "inscribed_diameter_mm")}}
        if entry["n_reoriented"] or entry["n_excluded"]:
            notes.append(f"{name}：{entry['n_reoriented']} 站重新定向，{entry['n_excluded']} 站截面不可靠未计入直径统计"
                         f"（共 {entry['n_stations']} 站）")
        if not reliable.any() and closed.any():
            notes.append(f"{name}：没有可靠截面，直径统计退回全部闭合截面，数值可能偏大")
        if cloud_seg is not None and cloud_wss is not None:
            pick = cloud_seg == sid
            if pick.any():
                values = cloud_wss[pick]
                entry.update(wss_p99_pa=_finite(np.quantile(values, 0.99)), wss_mean_pa=_finite(values.mean()),
                             area_frac_low=_finite(np.mean(values < low)), area_frac_high=_finite(np.mean(values > high)))
        elif per_branch and isinstance(per_branch.get(name), Mapping):
            row = per_branch[name]
            entry.update(wss_p99_pa=_finite(row.get("wss_p99_pa")), wss_mean_pa=_finite(row.get("wss_mean_pa")),
                         area_frac_low=_finite(row.get("frac_low")), area_frac_high=_finite(row.get("frac_high")))
        if int_seg is not None and int_speed is not None:
            pick = int_seg == sid
            if pick.any():
                entry.update(speed_mean_m_s=_finite(int_speed[pick].mean()), speed_max_m_s=_finite(int_speed[pick].max()))
        branches.append(entry)
        if is_aorta:
            aorta_block = _aorta_block(entry, table, dmax, equivalent, usable, polygons,
                                       station_mm=station_mm, notes=notes)

    volume_ml, volume_note = cached_lumen_volume(mesh, section_cache)
    if volume_note:
        notes.append(volume_note)
    return {"schema_version": MORPHOLOGY_SCHEMA, "station_mm": float(station_mm), "method": dict(METHOD),
            "aorta": aorta_block, "lumen_volume_ml": volume_ml,
            "lumen_volume_method": "capped_mesh_divergence" if volume_ml is not None else None,
            "branches": branches, "notes": notes}


def _aorta_block(entry: Mapping[str, Any], table: Mapping[str, Any], dmax: np.ndarray, equivalent: np.ndarray,
                 usable: np.ndarray, polygons, *, station_mm: float, notes: list[str]) -> dict:
    """Reference diameter, the largest station, the sac and the proximal neck of the aorta.

    ``usable`` are the stations the statistics may read: the reliable ones, or — when the aorta has
    none — every closed station (the caller has already recorded that fallback in ``notes``).
    """
    block = {"segment_id": entry["segment_id"], "name": entry["name"], "length_mm": entry["length_mm"],
             "stations": dict(entry["stations"]), "reference_diameter_mm": None, "max": None,
             "sac": {"present": False}, "neck": {"present": False},
             "n_reliable": entry.get("n_reliable"), "n_reoriented": entry.get("n_reoriented"),
             "n_excluded": entry.get("n_excluded")}
    if not usable.any():
        notes.append("主动脉没有可用的闭合截面，未给出瘤体形态")
        return block
    s_root = _array(table["s_from_root_mm"])
    s_local = _array(table["s_local_mm"])
    area = _array(table["area_mm2"])
    # Every series is masked to the usable stations before a statistic is taken.
    dmax = np.where(usable, dmax, np.nan)
    equivalent = np.where(usable, equivalent, np.nan)
    reference = float(np.nanpercentile(equivalent, REFERENCE_PERCENTILE))
    block["reference_diameter_mm"] = _finite(reference)
    imax = int(np.nanargmax(dmax))
    inscribed = _array(table["inscribed_diameter_mm"])
    dmin_series = _array(table["min_diameter_mm"])
    oblique = _oblique_mask(table)
    block["max"] = {"max_diameter_mm": _finite(dmax[imax]), "equivalent_diameter_mm": _finite(equivalent[imax]),
                    "min_diameter_mm": _finite(dmin_series[imax]),
                    "area_mm2": _finite(area[imax]), "s_from_root_mm": _finite(s_root[imax]),
                    "s_local_mm": _finite(s_local[imax]),
                    "xyz_mm": [float(v) for v in table["xyz_mm"][imax]],
                    "distance_from_inlet_mm": _finite(s_local[imax]), "station_index": imax,
                    # Obliquity check: a plane that is not locally perpendicular cuts a long slab, so
                    # its smallest width far exceeds the centreline's inscribed diameter there.
                    "inscribed_diameter_mm": _finite(inscribed[imax]),
                    "obliquity_ratio": _finite(dmin_series[imax] / inscribed[imax]) if inscribed[imax] > 0 else None,
                    "oblique": bool(oblique[imax]),
                    "polygon_world": _subsample_polygon(polygons[imax]) if polygons and polygons[imax] is not None else []}
    if oblique[imax]:
        notes.append(f"管腔最大直径站（距入口 {s_local[imax]:.0f} mm）的截面明显斜切："
                     f"截面最小宽度 {dmin_series[imax]:.1f} mm 远大于中心线内切直径 {inscribed[imax]:.1f} mm，"
                     "该处管腔最大直径可能高估，请在工作区三维视图里用截面环核对")
    threshold = SAC_FACTOR * reference
    wide = usable & np.isfinite(equivalent) & (equivalent >= threshold)
    runs = _runs(wide)
    run = next((r for r in runs if r[0] <= imax <= r[1]), None)
    if run is None and runs:
        run = max(runs, key=lambda r: r[1] - r[0])
    if run is None:
        notes.append(f"管腔无瘤样扩张（主动脉管腔最大等效直径 {np.nanmax(equivalent):.1f} mm < {SAC_FACTOR:g} × 参考直径 {reference:.1f} mm），"
                     "不能据此排除动脉瘤")
        return block
    lo, hi = run
    span = slice(lo, hi + 1)
    s_sac, a_sac = s_root[span], area[span]
    good = np.isfinite(s_sac) & np.isfinite(a_sac)
    sac_volume = float(_trapezoid(a_sac[good], s_sac[good])) / 1000.0 if good.sum() >= 2 else None
    block["sac"] = {"present": True, "s_start_mm": _finite(s_root[lo]), "s_end_mm": _finite(s_root[hi]),
                    "length_mm": _finite(s_root[hi] - s_root[lo]), "volume_ml": _finite(sac_volume),
                    "max_diameter_mm": _finite(np.nanmax(dmax[span])),
                    "equivalent_diameter_max_mm": _finite(np.nanmax(equivalent[span])),
                    "threshold_mm": _finite(threshold), "n_stations": int(hi - lo + 1)}
    # Proximal neck: the run of near-reference calibre immediately below the sac.  Between the neck
    # and the sac there is usually a shoulder of a few stations that is neither (< 1.5 × but ≥ 1.2 ×
    # the reference), so the search starts at the last qualifying station below the sac rather than
    # requiring the neck to touch ``lo`` — otherwise a real neck is reported as absent.
    narrow = usable & np.isfinite(equivalent) & (equivalent < NECK_FACTOR * reference)
    below = np.flatnonzero(narrow[:lo])
    if len(below):
        neck_hi = int(below[-1])
        neck_lo = neck_hi
        while neck_lo > 0 and narrow[neck_lo - 1]:
            neck_lo -= 1
        neck = equivalent[neck_lo:neck_hi + 1]
        block["neck"] = {"present": True, "s_start_mm": _finite(s_root[neck_lo]), "s_end_mm": _finite(s_root[neck_hi]),
                         "length_mm": _finite((neck_hi - neck_lo + 1) * station_mm),
                         "gap_to_sac_mm": _finite(s_root[lo] - s_root[neck_hi]),
                         "diameter_mean_mm": _finite(np.nanmean(neck)), "diameter_min_mm": _finite(np.nanmin(neck)),
                         "diameter_max_mm": _finite(np.nanmax(neck)), "threshold_mm": _finite(NECK_FACTOR * reference),
                         "n_stations": int(neck_hi - neck_lo + 1)}
    else:
        notes.append(f"瘤体起点近端没有等效直径 < {NECK_FACTOR:g} × 参考直径（{NECK_FACTOR * reference:.1f} mm）的区段，未给出瘤颈")
    return block


# ----------------------------------------------------------------------------- report trimming
REPORT_STATION_KEYS = ("s_from_root_mm", "max_diameter_mm", "equivalent_diameter_mm", "reliable")


def _trim_stations(stations: Any) -> dict:
    stations = stations if isinstance(stations, Mapping) else {}
    return {key: list(stations.get(key) or []) for key in REPORT_STATION_KEYS}


def digest(morphology: Mapping[str, Any] | None) -> dict | None:
    """Station-free copy for the job record and the workbench cards (a few hundred bytes).

    Everything a card shows — the largest section, the sac, the neck, the reference diameter and
    the volumes — without the per-station series or the ring polygon.
    """
    if not isinstance(morphology, Mapping):
        return None
    aorta = morphology.get("aorta")
    out = {key: value for key, value in morphology.items() if key not in {"aorta", "branches", "method"}}
    if isinstance(aorta, Mapping):
        largest = aorta.get("max")
        out["aorta"] = {**{k: v for k, v in aorta.items() if k not in {"stations", "max"}},
                        "max": ({k: v for k, v in largest.items() if k != "polygon_world"}
                                if isinstance(largest, Mapping) else None)}
    else:
        out["aorta"] = None
    out["branches"] = [{k: v for k, v in branch.items() if k != "stations"}
                       for branch in (morphology.get("branches") or []) if isinstance(branch, Mapping)]
    return out


def max_diameter_mm(morphology: Mapping[str, Any] | None) -> float | None:
    """The headline diameter of a stored (full or digested) morphology block, or None."""
    aorta = morphology.get("aorta") if isinstance(morphology, Mapping) else None
    largest = aorta.get("max") if isinstance(aorta, Mapping) else None
    return _finite(largest.get("max_diameter_mm")) if isinstance(largest, Mapping) else None


def trim_for_report(morphology: Mapping[str, Any] | None) -> dict | None:
    """Report-sized copy: only the three station series the curves need, plus the max-diameter ring.

    ``summary.json`` keeps the full stations (``xyz_mm``, ``closed``, ``area_mm2``); the embedded
    report metadata would otherwise carry several hundred kilobytes of duplicated geometry.
    """
    if not isinstance(morphology, Mapping):
        return None
    out = {key: value for key, value in morphology.items() if key not in {"aorta", "branches"}}
    aorta = morphology.get("aorta")
    if isinstance(aorta, Mapping):
        trimmed = dict(aorta)
        trimmed["stations"] = _trim_stations(aorta.get("stations"))
        out["aorta"] = trimmed
    else:
        out["aorta"] = None
    out["branches"] = [{**{k: v for k, v in branch.items() if k != "stations"},
                        "stations": _trim_stations(branch.get("stations"))}
                       for branch in (morphology.get("branches") or []) if isinstance(branch, Mapping)]
    return out


# Take the source hash now, while the file on disk is the code this process has just imported.
try:
    _code_hash()
except Exception:  # pragma: no cover - an unreadable source only means cache keys are computed later
    pass

__all__ = ["AORTA_NAME", "MORPHOLOGY_SCHEMA", "METHOD", "MeshSections", "STATION_MM", "branch_stations",
           "cached_branch_stations", "cached_lumen_volume", "precompute_sections",
           "close_chain", "compute", "contour_loops", "digest", "loop_polygon", "lumen_volume_ml",
           "max_diameter_mm", "plane_frame", "reorient_section", "section_metrics", "select_loop",
           "station_flags", "trim_for_report"]
