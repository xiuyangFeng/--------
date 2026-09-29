"""Wall STL -> closed boundary mesh: planar caps on every opening, named, written as a Fluent ASCII boundary mesh.

The library meshes keep the STL facets as the anatomy wall (QIN_SI_FU: 28068 wall faces / 14112 nodes == the STL), so the
wall triangles are passed through untouched; only the caps are generated. Caps reuse the loop nodes of the wall, hence
the surface is conformal and watertight by construction.
"""
from __future__ import annotations

import dataclasses
import gzip
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import Delaunay, cKDTree

MM_TO_M = 1e-3
FACE_TYPE = {"wall": 3, "velocity-inlet": 10, "pressure-outlet": 5, "interior": 2, "internal": 15}  # internal: two-sided boundary-mesh zone (Fluent Meshing), read by the solver as interior


@dataclasses.dataclass
class Cap:
    name: str
    bc_type: str
    loop: np.ndarray            # ordered wall-node ids of the opening
    faces: np.ndarray           # (k, 3) node ids (wall nodes + new interior nodes), oriented outward
    centre_mm: np.ndarray
    normal: np.ndarray          # outward unit normal
    area_mm2: float
    planarity_mm: float         # max |distance| of the loop nodes to the fitted plane


@dataclasses.dataclass
class ClosedSurface:
    points_mm: np.ndarray       # (N, 3) wall nodes followed by the cap interior nodes
    wall_faces: np.ndarray      # (M, 3) oriented outward
    caps: list[Cap]
    n_wall_nodes: int


def read_stl(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Merged points (mm) and triangles, keeping the STL vertex order (pyvista merges duplicates exactly)."""
    import pyvista as pv
    mesh = pv.read(str(path)).clean(tolerance=0.0)
    faces = mesh.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError(f"{path}: non-triangular facets")
    return np.asarray(mesh.points, dtype=float), faces[:, 1:].astype(np.int64)


def _seg_tri(p0, p1, a, b, c, eps=1e-12):
    """Vectorised proper segment/triangle intersection (Moller-Trumbore; touching at vertices/edges excluded)."""
    d = p1 - p0; e1 = b - a; e2 = c - a
    h = np.cross(d, e2); det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > eps
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = p0 - a; u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1); v = inv * np.einsum("ij,ij->i", d, q); t = inv * np.einsum("ij,ij->i", e2, q)
    tol = 1e-9
    return ok & (u > tol) & (v > tol) & (u + v < 1 - tol) & (t > tol) & (t < 1 - tol)


def self_intersections(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Pairs of triangles that properly intersect (candidates within one max edge length, vertex-sharing pairs skipped)."""
    tri = points[faces]
    cen = tri.mean(1)
    r = 1.01 * float(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max())
    pairs = cKDTree(cen).query_pairs(r, output_type="ndarray")
    if not len(pairs):
        return pairs
    A, B = faces[pairs[:, 0]], faces[pairs[:, 1]]
    share = (A[:, :, None] == B[:, None, :]).any((1, 2))
    pairs, A, B = pairs[~share], A[~share], B[~share]
    hit = np.zeros(len(pairs), bool)
    for X, Y in ((A, B), (B, A)):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            hit |= _seg_tri(points[X[:, i]], points[X[:, j]], points[Y[:, 0]], points[Y[:, 1]], points[Y[:, 2]])
    return pairs[hit]


REPAIR_MAX_MOVE_MM = 0.5     # default limit; a caller may raise it for one case with a recorded reason (cfd_auto.recover)


def repair_self_intersections(points: np.ndarray, faces: np.ndarray, fixed: np.ndarray, rings: int = 2, max_iter: int = 50, max_move_mm: float | None = None) -> tuple[np.ndarray, dict]:
    """Local Laplacian smoothing of the vertices around intersecting triangles (``rings`` neighbourhood; ``fixed`` vertices,
    e.g. opening rims, never move) until no intersection remains. Raises if a vertex would move more than ``max_move_mm``
    (default ``REPAIR_MAX_MOVE_MM``)."""
    max_move_mm = REPAIR_MAX_MOVE_MM if max_move_mm is None else max_move_mm
    P = points.copy()
    nbr = [set() for _ in range(len(P))]
    for a, b, c in faces:
        nbr[a].update((b, c)); nbr[b].update((a, c)); nbr[c].update((a, b))
    pairs = self_intersections(P, faces)
    report = {"initial_pairs": int(len(pairs)), "locations_mm": P[faces[pairs[:, 0]]].mean(1).round(2).tolist()[:10] if len(pairs) else []}
    it = 0
    while len(pairs) and it < max_iter:
        it += 1
        region = set(np.unique(faces[pairs.ravel()]).tolist())
        for _ in range(rings):
            region |= {n for v in list(region) for n in nbr[v]}
        region = np.array(sorted(region - set(np.flatnonzero(fixed).tolist())))
        for _ in range(5):
            P[region] = 0.5 * P[region] + 0.5 * np.array([P[list(nbr[v])].mean(0) for v in region])
        pairs = self_intersections(P, faces)
    move = float(np.linalg.norm(P - points, axis=1).max())
    report.update(iterations=it, remaining_pairs=int(len(pairs)), moved_vertices=int(np.sum(np.linalg.norm(P - points, axis=1) > 1e-9)), max_move_mm=round(move, 4))
    if len(pairs) or move > max_move_mm:
        raise RuntimeError(f"self-intersection repair failed: {report}")
    return P, report


def boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    """Ordered node loops of the free edges (each edge used by one triangle); raises on non-manifold edges."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError(f"{int(np.sum(counts > 2))} non-manifold edges")
    free = uniq[counts == 1]
    nbr: dict[int, list[int]] = {}
    for a, b in free:
        nbr.setdefault(int(a), []).append(int(b)); nbr.setdefault(int(b), []).append(int(a))
    if any(len(v) != 2 for v in nbr.values()):
        raise ValueError("free edges do not form simple loops (pinched opening)")
    loops, seen = [], set()
    for start in nbr:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = nbr[cur][0] if nbr[cur][0] != prev else nbr[cur][1]
            if nxt == start:
                break
            loop.append(nxt); seen.add(nxt); prev, cur = cur, nxt
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def orient_outward(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Consistently orient a closed triangulation with outward normals (signed volume > 0)."""
    import pyvista as pv
    poly = pv.PolyData(points, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    fixed = poly.compute_normals(cell_normals=True, point_normals=False, consistent_normals=True, auto_orient_normals=True, split_vertices=False, non_manifold_traversal=False)
    out = fixed.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    if len(out) != len(faces) or fixed.n_points != len(points):
        raise RuntimeError("orientation changed the triangulation")
    v = points[out]
    if np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() <= 0:
        out = out[:, ::-1]
    return out


def _plane(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c)
    return c, vt[0], vt[1]  # centre and in-plane axes (normal = vt[2])


def cap_loop(points: np.ndarray, loop: np.ndarray, first_new_id: int, spacing_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Planar Delaunay cap of one opening: interior nodes on a triangular lattice (spacing = median loop edge), boundary
    edges of the loop must all be recovered. Returns (new interior points, triangles in global ids, stats)."""
    P = points[loop]
    c, ex, ey = _plane(P)
    normal = np.cross(ex, ey)
    uv = np.c_[(P - c) @ ex, (P - c) @ ey]
    edge = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    h = float(np.median(edge)) * spacing_scale
    lo, hi = uv.min(0) - h, uv.max(0) + h
    xs = np.arange(lo[0], hi[0], h); ys = np.arange(lo[1], hi[1], h * np.sqrt(3) / 2)
    gx, gy = np.meshgrid(xs, ys)
    gx = gx + (np.arange(len(ys))[:, None] % 2) * h / 2
    grid = np.c_[gx.ravel(), gy.ravel()]
    poly = MplPath(uv)
    inside = poly.contains_points(grid)
    d, _ = cKDTree(uv).query(grid)
    # distance to the polygon edges, not only its vertices
    seg_a, seg_b = uv, np.roll(uv, -1, 0)
    def seg_dist(q):
        ab = seg_b - seg_a; t = np.clip(((q[:, None, :] - seg_a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
        return np.linalg.norm(q[:, None, :] - (seg_a + t[..., None] * ab), axis=-1).min(1)
    keep = inside & (d > 0.6 * h)
    grid = grid[keep]
    if len(grid):
        grid = grid[seg_dist(grid) > 0.6 * h]
    all_uv = np.vstack([uv, grid])
    tri = Delaunay(all_uv).simplices
    cen = all_uv[tri].mean(1)
    tri = tri[poly.contains_points(cen)]
    # every loop edge must be a triangle edge (constrained boundary recovered)
    n = len(loop)
    tri_edges = {tuple(sorted(e)) for t in tri for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    missing = [(i, (i + 1) % n) for i in range(n) if tuple(sorted((i, (i + 1) % n))) not in tri_edges]
    if missing:
        raise RuntimeError(f"cap: {len(missing)} loop edges not recovered by the Delaunay triangulation")
    a = all_uv[tri]
    area2d = 0.5 * np.abs(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]))
    poly_area = 0.5 * abs(np.dot(uv[:, 0], np.roll(uv[:, 1], -1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], -1)))
    if abs(area2d.sum() - poly_area) > 1e-6 * poly_area + 1e-9:
        raise RuntimeError(f"cap area {area2d.sum():.4f} != polygon area {poly_area:.4f}")
    L = np.linalg.norm(a - np.roll(a, -1, 1), axis=-1)
    min_angle = []
    for k in range(3):
        b_, c_, a_ = L[:, k], L[:, (k + 1) % 3], L[:, (k + 2) % 3]
        min_angle.append(np.degrees(np.arccos(np.clip((b_ ** 2 + c_ ** 2 - a_ ** 2) / (2 * b_ * c_), -1, 1))))
    min_angle = np.min(np.stack(min_angle, 1), 1)
    new_pts = c + grid[:, :1] * ex + grid[:, 1:] * ey
    ids = np.concatenate([loop, first_new_id + np.arange(len(grid))])
    stats = {"interior_nodes": int(len(grid)), "triangles": int(len(tri)), "spacing_mm": h, "min_angle_deg_p1": float(np.percentile(min_angle, 1)),
             "min_angle_deg_min": float(min_angle.min()), "area_mm2": float(poly_area), "planarity_mm": float(np.abs((P - c) @ normal).max())}
    return new_pts, ids[tri], stats


def close_surface(stl: str | Path, names: dict[int, tuple[str, str]] | None = None) -> tuple[ClosedSurface, list[dict]]:
    """Cap every opening. ``names`` maps loop index -> (zone name, bc type); default names are cap-0.. (pressure-outlet)."""
    pts, wall = read_stl(stl)
    loops = boundary_loops(wall)
    rim = np.zeros(len(pts), bool); rim[np.concatenate(loops)] = True
    pts, repair = repair_self_intersections(pts, wall, rim)
    all_pts, caps_raw, stats = [pts], [], []
    nxt = len(pts)
    for i, loop in enumerate(loops):
        new, tri, st = cap_loop(pts, loop, nxt)
        all_pts.append(new); nxt += len(new); caps_raw.append((loop, tri)); stats.append(st)
    points = np.vstack(all_pts)
    faces = np.vstack([wall] + [t for _, t in caps_raw])
    oriented = orient_outward(points, faces)
    # split back (orientation keeps face order)
    wall_o = oriented[: len(wall)]
    caps, off = [], len(wall)
    for i, ((loop, tri), st) in enumerate(zip(caps_raw, stats)):
        f = oriented[off: off + len(tri)]; off += len(tri)
        v = points[f]; an = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]).sum(0); nrm = an / np.linalg.norm(an)
        name, bc = (names or {}).get(i, (f"cap-{i}", "pressure-outlet"))
        caps.append(Cap(name, bc, loop, f, points[loop].mean(0), nrm, st["area_mm2"], st["planarity_mm"]))
        st.update(name=name, bc_type=bc, loop_nodes=int(len(loop)), centre_mm=points[loop].mean(0).round(3).tolist(), normal=nrm.round(5).tolist())
    surf = ClosedSurface(points, wall_o, caps, len(pts))
    surf.repair = repair
    return surf, stats


def name_caps_by_reference(surf_loops_centres_mm: np.ndarray, reference: list[dict], tol_mm: float = 2.0) -> dict[int, tuple[str, str]]:
    """Match cap centres to a reference case's openings ({name, bc_type, centre_mm}); one-to-one within ``tol_mm``."""
    ref_c = np.array([r["centre_mm"] for r in reference])
    d = np.linalg.norm(surf_loops_centres_mm[:, None] - ref_c[None], axis=-1)
    j = d.argmin(1)
    if len(set(j.tolist())) != len(j) or np.any(d[np.arange(len(j)), j] > tol_mm):
        raise ValueError(f"opening/reference match is not one-to-one within {tol_mm} mm: {d.round(2).tolist()}")
    return {i: (reference[k]["name"], reference[k]["bc_type"]) for i, k in enumerate(j.tolist())}


@dataclasses.dataclass
class Extension:
    opening: str                # cap name (becomes the distal boundary zone)
    length_mm: float
    wall_zone: str              # e.g. wall1
    interface_zone: str         # e.g. in+ (the original cap, now internal)


def extend_surface(surf: ClosedSurface, extensions: list[Extension], wall_name: str = "wall") -> tuple[np.ndarray, list[tuple[str, str, np.ndarray]], list[dict]]:
    """Straight flow extensions: each cap's rim is swept along its outward normal (opening-shaped tube, as the
    library's SpaceClaim extensions: constant cross-section, library lengths), the swept cap closes the tube and keeps the
    boundary name; the original cap becomes the internal interface zone. Axial spacing = median rim edge length.
    Returns (points_mm, zones [(name, type, faces)], stats)."""
    pts = [surf.points_mm]
    nxt = len(surf.points_mm)
    zones: list[tuple[str, str, np.ndarray]] = [(wall_name, "wall", surf.wall_faces)]
    caps = {c.name: c for c in surf.caps}
    stats = []
    for ext in extensions:
        cap = caps[ext.opening]
        loop = cap.loop; m = len(loop); n = cap.normal
        rim = surf.points_mm[loop]
        h = float(np.median(np.linalg.norm(np.roll(rim, -1, 0) - rim, axis=1)))
        N = max(1, int(round(ext.length_mm / h)))
        s = np.linspace(0.0, ext.length_mm, N + 1)
        rings = [loop]
        for j in range(1, N + 1):
            new = rim + s[j] * n
            pts.append(new); rings.append(nxt + np.arange(m)); nxt += m
        side = []
        for j in range(N):
            a, b = rings[j], rings[j + 1]
            a1, b1 = np.roll(a, -1), np.roll(b, -1)
            if j % 2 == 0:
                side += [np.c_[a, a1, b1], np.c_[a, b1, b]]
            else:
                side += [np.c_[a, a1, b], np.c_[a1, b1, b]]
        side = np.vstack(side)
        # orient the tube wall outward (away from the axis)
        allp = np.vstack(pts)
        v = allp[side]; nrm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]); cen = v.mean(1)
        axis_pt = rim.mean(0) + ((cen - rim.mean(0)) @ n)[:, None] * n
        if np.mean(np.einsum("ij,ij->i", nrm, cen - axis_pt) > 0) < 0.5:
            side = side[:, ::-1]
        # distal cap = the original cap translated; rim ids -> last ring, cap interior ids -> new translated nodes
        cap_ids = np.unique(cap.faces)
        interior_ids = np.setdiff1d(cap_ids, loop)
        remap = dict(zip(loop.tolist(), rings[-1].tolist()))
        new_int = surf.points_mm[interior_ids] + ext.length_mm * n
        pts.append(new_int); remap.update(zip(interior_ids.tolist(), (nxt + np.arange(len(interior_ids))).tolist())); nxt += len(interior_ids)
        distal = np.vectorize(remap.__getitem__)(cap.faces)
        zones += [(ext.wall_zone, "wall", side), (ext.interface_zone, "internal", cap.faces), (cap.name, cap.bc_type, distal)]
        stats.append({"opening": cap.name, "length_mm": ext.length_mm, "axial_segments": N, "axial_spacing_mm": round(ext.length_mm / N, 4),
                      "rim_edge_mm": round(h, 4), "wall_faces": int(len(side)), "wall_zone": ext.wall_zone, "interface_zone": ext.interface_zone})
    return np.vstack(pts), zones, stats


def check_regions(points_mm: np.ndarray, zones: list[tuple[str, str, np.ndarray]], extensions: list[Extension], wall_name: str = "wall") -> dict:
    """Every region (anatomy; each extension) must be a closed manifold; extension tubes must not collide with the anatomy
    wall or with each other away from their own rim (first axial segment excluded)."""
    import pyvista as pv
    zf = {name: f for name, _, f in zones}
    poly = lambda f: pv.PolyData(points_mm, np.hstack([np.full((len(f), 1), 3), f]).ravel()).clean()
    out = {}
    caps_iface = [zf[e.interface_zone] for e in extensions]
    anatomy = poly(np.vstack([zf[wall_name]] + caps_iface))
    out["anatomy"] = {"open_edges": int(anatomy.n_open_edges), "manifold": bool(anatomy.is_manifold), "volume_mm3": round(float(anatomy.volume), 2)}
    tubes = {}
    for e in extensions:
        reg = poly(np.vstack([zf[e.wall_zone], zf[e.interface_zone][:, ::-1], zf[e.opening]]))  # interface seen from the extension
        out[e.opening] = {"open_edges": int(reg.n_open_edges), "manifold": bool(reg.is_manifold), "volume_mm3": round(float(reg.volume), 2)}
        side = zf[e.wall_zone]
        rim_nodes = set(np.unique(zf[e.interface_zone]).tolist())
        far = ~np.any(np.isin(side, list(rim_nodes)), axis=1)
        # drop the first ring of faces next to the rim as well (they touch the anatomy wall by construction)
        ring1 = set(np.unique(side[~far]).tolist())
        far &= ~np.any(np.isin(side, list(ring1)), axis=1)
        tubes[e.opening] = poly(side[far])
    wall = poly(zf[wall_name])
    collisions = {}
    names = list(tubes)
    for i, a in enumerate(names):
        _, n_wall = tubes[a].collision(wall, contact_mode=0)
        collisions[f"{a}|{wall_name}"] = int(n_wall)
        for b in names[i + 1:]:
            _, n_ab = tubes[a].collision(tubes[b], contact_mode=0)
            collisions[f"{a}|{b}"] = int(n_ab)
    out["collisions"] = collisions
    out["ok"] = all(v["open_edges"] == 0 and v["manifold"] for k, v in out.items() if isinstance(v, dict) and "open_edges" in v) and not any(collisions.values())
    return out


def write_fluent_boundary_mesh(surf: ClosedSurface, path: str | Path, wall_name: str = "wall") -> dict:
    """Fluent ASCII boundary mesh of the capped anatomy (one wall zone + one zone per cap)."""
    zones = [(wall_name, "wall", surf.wall_faces)] + [(c.name, c.bc_type, c.faces) for c in surf.caps]
    return write_fluent_zones(surf.points_mm, zones, path)


def write_fluent_zones(points_mm: np.ndarray, zones: list[tuple[str, str, np.ndarray]], path: str | Path) -> dict:
    """Fluent ASCII (hex-indexed) boundary mesh in metres, cells c0 = c1 = 0."""
    pts = points_mm * MM_TO_M
    n_nodes = len(pts); n_faces = sum(len(f) for *_, f in zones)
    out = [f'(0 "cfd_auto boundary mesh: {len(zones)} face zones")', "(2 3)", f"(10 (0 1 {n_nodes:x} 0 3))", f"(10 (1 1 {n_nodes:x} 1 3)(",
           *(f"{x:.12e} {y:.12e} {z:.12e}" for x, y, z in pts), "))", f"(13 (0 1 {n_faces:x} 0))"]
    first, zone_rows = 1, []
    for zid, (name, bc, f) in enumerate(zones, start=3):
        last = first + len(f) - 1
        out.append(f"(13 ({zid:x} {first:x} {last:x} {FACE_TYPE[bc]:x} 3)(")
        out.extend(f"{a + 1:x} {b + 1:x} {c + 1:x} 0 0" for a, b, c in f)
        out.append("))")
        zone_rows.append({"zone_id": zid, "name": name, "type": bc, "faces": int(len(f))})
        first = last + 1
    for z in zone_rows:
        out.append(f"(45 ({z['zone_id']} {z['type']} {z['name']})())")
    text = "\n".join(out) + "\n"
    path = Path(path)
    with (gzip.open(path, "wt") if path.suffix == ".gz" else open(path, "w")) as fh:
        fh.write(text)
    return {"path": str(path), "nodes": n_nodes, "faces": n_faces, "zones": zone_rows}
