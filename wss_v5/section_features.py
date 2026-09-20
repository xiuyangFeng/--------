"""Geometry-only cross sections for a review candidate, never a training loader.

Reference sections intersect an anatomical triangle surface. Point-cloud
sections use a deliberately limited star-shaped radial reconstruction; its
coverage/multimodality diagnostics are retained instead of silently filling a
convex hull across branches. All coordinates/areas are mm/mm².
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = unit(normal)
    hint = np.eye(3)[np.argmin(np.abs(t))]
    e1 = unit(hint - np.dot(hint, t) * t)
    return e1, np.cross(t, e1)


def contains_origin(p: np.ndarray) -> bool:
    """Even/odd test including an origin on an edge, for a simple 2D polygon."""
    p = np.asarray(p, float)
    q = np.roll(p, -1, axis=0)
    edge = q - p
    a = np.clip(-np.sum(p * edge, axis=1) / np.maximum(np.sum(edge * edge, axis=1), 1e-20), 0, 1)
    if np.min(np.linalg.norm(p + a[:, None] * edge, axis=1)) < 1e-8:
        return True
    hit = (p[:, 1] > 0) != (q[:, 1] > 0)
    den = q[:, 1] - p[:, 1]
    safe_den = np.where(np.abs(den) > 1e-20, den, 1.0)
    x = p[:, 0] - p[:, 1] * (q[:, 0] - p[:, 0]) / safe_den
    return bool(np.sum(hit & (x > 0)) % 2)


def polygon_metrics(xyz: np.ndarray, center: np.ndarray, normal: np.ndarray) -> dict:
    xyz = np.asarray(xyz, float)
    if len(xyz) > 1 and np.linalg.norm(xyz[0] - xyz[-1]) < 1e-7:
        xyz = xyz[:-1]
    if len(xyz) < 3:
        return {"valid": False, "reason": "fewer_than_three_vertices"}
    e1, e2 = frame(normal)
    p = np.column_stack(((xyz - center) @ e1, (xyz - center) @ e2))
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    signed_area = np.sum(cross) / 2
    area = float(abs(signed_area))
    if area <= 1e-8:
        return {"valid": False, "reason": "zero_area"}
    centroid_2d = np.sum((p + q) * cross[:, None], axis=0) / (6 * signed_area)
    perimeter = float(np.linalg.norm(q - p, axis=1).sum())
    req = np.sqrt(area / np.pi)
    return {
        "valid": True, "reason": "ok", "area_mm2": area,
        "perimeter_mm": perimeter, "req_mm": float(req),
        "roundness": float(4 * np.pi * area / max(perimeter ** 2, 1e-20)),
        "eccentricity": float(np.linalg.norm(centroid_2d) / req),
        "centroid_mm": np.asarray(center) + centroid_2d[0] * e1 + centroid_2d[1] * e2,
        "contains_center": contains_origin(p), "polygon_xyz_mm": xyz,
    }


def polyline_plane_intersection_groups(xyz: np.ndarray, s: np.ndarray, center: np.ndarray,
                                       normal: np.ndarray, *, plane_tol_mm: float = 1e-5,
                                       merge_s_tol_mm: float = 1e-4) -> list[dict]:
    """Continuous edge/plane intersections, merged by along-polyline identity.

    Adjacent edges hitting the same vertex form one crossing. Consecutive
    nearly coplanar edges form one interval, not many apparent crossings.
    Distant portions are never merged just because their 3D points coincide.
    """
    xyz, s = np.asarray(xyz, float), np.asarray(s, float)
    if len(xyz) < 2:
        return []
    distance = (xyz - center) @ unit(normal)
    near = np.abs(distance) <= plane_tol_mm
    candidates = np.flatnonzero(near[:-1] | near[1:] | ((distance[:-1] < 0) != (distance[1:] < 0)))
    intervals = []
    for i in candidates:
        if near[i] and near[i + 1]:
            intervals.append((s[i], s[i + 1], xyz[i:i + 2], True))
        elif near[i]:
            intervals.append((s[i], s[i], xyz[i:i + 1], False))
        elif near[i + 1]:
            intervals.append((s[i + 1], s[i + 1], xyz[i + 1:i + 2], False))
        else:
            alpha = distance[i] / (distance[i] - distance[i + 1])
            sj = s[i] + alpha * (s[i + 1] - s[i])
            p = xyz[i] + alpha * (xyz[i + 1] - xyz[i])
            intervals.append((sj, sj, p[None], False))
    groups = []
    for lo, hi, pts, coplanar in intervals:
        if groups and lo <= groups[-1]["s_max_mm"] + merge_s_tol_mm:
            groups[-1]["s_max_mm"] = max(groups[-1]["s_max_mm"], float(hi))
            groups[-1]["points_xyz_mm"].extend(pts.tolist())
            groups[-1]["coplanar_interval"] |= coplanar
        else:
            groups.append({"s_min_mm": float(lo), "s_max_mm": float(hi), "points_xyz_mm": pts.tolist(), "coplanar_interval": bool(coplanar)})
    return groups


def _line_intersects_polygon(a: np.ndarray, b: np.ndarray, polygon: np.ndarray) -> bool:
    if contains_origin(polygon - a) or contains_origin(polygon - b):
        return True
    edge = np.roll(polygon, -1, axis=0) - polygon
    v = b - a
    cross = lambda x, y: x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    den = cross(edge, v)
    offset = a - polygon
    ordinary = np.abs(den) > 1e-12
    safe = np.where(ordinary, den, 1.)
    t = cross(offset, v) / safe
    u = cross(offset, edge) / safe
    proper = ordinary & (t >= -1e-9) & (t <= 1 + 1e-9) & (u >= -1e-9) & (u <= 1 + 1e-9)
    collinear = ~ordinary & (np.abs(cross(offset, v)) <= 1e-10)
    edge_end = polygon + edge
    overlap = np.all(np.maximum(np.minimum(a, b), np.minimum(polygon, edge_end)) <= np.minimum(np.maximum(a, b), np.maximum(polygon, edge_end)) + 1e-9, axis=1)
    return bool(np.any(proper | (collinear & overlap)))


def same_segment_nonlocal_crossings(polygon: np.ndarray, center: np.ndarray, normal: np.ndarray,
                                    xyz: np.ndarray, s: np.ndarray, local_s: float, *,
                                    nonlocal_gap_mm: float = 5., plane_tol_mm: float = 1e-5) -> dict:
    """Find extra nonlocal crossings of the same branch inside this contour."""
    groups = polyline_plane_intersection_groups(xyz, s, center, normal, plane_tol_mm=plane_tol_mm)
    e1, e2 = frame(normal)
    poly2 = np.column_stack(((polygon - center) @ e1, (polygon - center) @ e2))
    inside = []
    for group in groups:
        pts = np.asarray(group["points_xyz_mm"])
        p2 = np.column_stack(((pts - center) @ e1, (pts - center) @ e2))
        hit = any(contains_origin(poly2 - p) for p in p2)
        if not hit and len(p2) > 1:
            hit = any(_line_intersects_polygon(a, b, poly2) for a, b in zip(p2[:-1], p2[1:]))
        if hit:
            inside.append(group)
    if not inside:
        return {"inside_crossing_groups": [], "extra_nonlocal_groups": [], "local_group_index": None}
    distance = [max(g["s_min_mm"] - local_s, local_s - g["s_max_mm"], 0.) for g in inside]
    local = int(np.argmin(distance))
    # A long near-coplanar interval remains one group; it is not silently
    # counted as repeated crossings just because many samples lie in plane.
    extra = [g for j, g in enumerate(inside) if j != local and distance[j] > nonlocal_gap_mm]
    return {"inside_crossing_groups": inside, "extra_nonlocal_groups": extra, "local_group_index": local}


def _boundary_cycles(triangles: np.ndarray) -> tuple[list[np.ndarray], dict]:
    """Exact mesh-edge cycles; never merge nearby but distinct source vertices."""
    triangles = np.asarray(triangles, np.int64)
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                   triangles[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    adjacency = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    info = {"boundary_edges": int(len(boundary)), "nonmanifold_edges": int(np.sum(counts > 2)),
            "boundary_branch_vertices": int(sum(len(v) != 2 for v in adjacency.values()))}
    if info["nonmanifold_edges"] or info["boundary_branch_vertices"]:
        return [], info
    remaining = set(adjacency)
    cycles = []
    while remaining:
        start = min(remaining)
        cycle, previous, current = [start], None, start
        while True:
            nxt = next(v for v in adjacency[current] if v != previous)
            if nxt == start:
                break
            cycle.append(nxt)
            previous, current = current, nxt
        remaining.difference_update(cycle)
        cycles.append(np.asarray(cycle, np.int64))
    info["boundary_cycles"] = len(cycles)
    return cycles, info


def _verified_fan_center(polygon: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Prove a centroid fan has positive triangles and a single winding.

    A mildly nonplanar CFD rim can defeat planar ear clipping. Projecting it
    only for this proof permits an exact-border, nonplanar fan, without moving
    a source point. Concave non-star rims and self-crossing loops do not pass.
    """
    polygon = np.asarray(polygon, float)
    center = polygon.mean(axis=0)
    _, _, vh = np.linalg.svd(polygon - center, full_matrices=False)
    normal = vh[-1]
    metrics = polygon_metrics(polygon, center, normal)
    if not metrics.get("valid") or not metrics.get("contains_center"):
        return None, {"fan_verified": False, "reason": "invalid_projected_polygon"}
    center = metrics["centroid_mm"]
    e1, e2 = frame(normal)
    p = np.column_stack(((polygon - center) @ e1, (polygon - center) @ e2))
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
    scale = max(float(np.max(np.sum(p * p, axis=1))), 1.)
    eps = 64 * np.finfo(float).eps * scale
    same_orientation = bool(np.all(cross > eps) or np.all(cross < -eps))
    winding = float(np.sum(np.arctan2(cross, np.sum(p * q, axis=1))) / (2 * np.pi))
    verified = same_orientation and abs(abs(winding) - 1.) < 1e-8
    return (center if verified else None), {
        "fan_verified": verified, "reason": "single_winding_positive_fan" if verified else "fan_not_proven",
        "projected_area_mm2": metrics["area_mm2"], "winding": winding,
        "plane_residual_max_mm": float(np.max(np.abs((polygon - center) @ normal))),
        "minimum_projected_double_triangle_area_mm2": float(np.min(np.abs(cross))),
    }


def _remove_opposite_triangle_fins(triangles: np.ndarray) -> tuple[np.ndarray, dict]:
    """Remove only exact opposite triangle pairs attached as zero-thickness fins.

    Ordinary duplicate faces and unrelated nonmanifold defects are retained.
    Each removed pair must change its three edge incidences from 4/2 to 2/0,
    preserving every real opening boundary edge and every surviving wall face.
    """
    triangles = np.asarray(triangles, np.int64)
    _, group, counts = np.unique(np.sort(triangles, axis=1), axis=0, return_inverse=True, return_counts=True)
    pairs = np.flatnonzero(counts == 2)
    removed, proof = [], []
    if len(pairs):
        edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        unique, incidence = np.unique(edges, axis=0, return_counts=True)
        edge_counts = {tuple(edge): int(n) for edge, n in zip(unique, incidence)}
        for group_id in pairs:
            rows = np.flatnonzero(group == group_id)
            a, b = triangles[rows]
            opposite = any(np.array_equal(np.roll(a[::-1], k), b) for k in range(3))
            ee = np.sort(np.array([[a[0], a[1]], [a[1], a[2]], [a[2], a[0]]]), axis=1)
            nn = np.array([edge_counts[tuple(edge)] for edge in ee])
            if opposite and len(np.unique(a)) == 3 and np.any(nn == 4) and np.all(np.isin(nn, [2, 4])):
                removed.extend(rows.tolist())
                proof.append({"source_triangle_rows": rows.tolist(), "triangle_node_rows": a.tolist(),
                              "edges": ee.tolist(), "edge_incidence_before": nn.tolist(),
                              "edge_incidence_after": (nn - 2).tolist(),
                              "exact_opposite_vertex_order": True, "zero_thickness": True})
    keep = np.ones(len(triangles), bool)
    keep[removed] = False
    return triangles[keep], {"removed_triangle_rows": sorted(removed), "removed_triangles": len(removed),
                             "pairs": proof, "frozen_source_modified": False}


def _fan_cap_outward_offset(point, polygon, center, normal, fan_center):
    """Signed offset from the actual 3D fan facet under a projected point."""
    e1, e2 = frame(normal)
    project = lambda a: np.column_stack(((np.atleast_2d(a) - center) @ e1, (np.atleast_2d(a) - center) @ e2))
    a, b = project(fan_center)[0], project(polygon)
    c, p = np.roll(b, -1, axis=0), project(point)[0]
    v, w, q = b - a, c - a, p - a
    cross = lambda x, y: x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    den = cross(v, w)
    u, t = cross(q, w) / den, cross(v, q) / den
    hit = (u >= -1e-9) & (t >= -1e-9) & (u + t <= 1 + 1e-9)
    if not hit.any():
        return None
    z, z0 = (polygon - center) @ normal, (fan_center - center) @ normal
    values = (1 - u - t) * z0 + u * z + t * np.roll(z, -1)
    return float((point - center) @ normal - np.max(values[hit]))


def _fan_cap_segment_crossing(a, b, polygon, fan_center):
    """Intersect a centerline edge with actual 3D fan triangles (Moller-Trumbore)."""
    t0 = np.broadcast_to(fan_center, polygon.shape)
    direction, e1, e2 = b - a, polygon - t0, np.roll(polygon, -1, axis=0) - t0
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    good = np.abs(det) > 1e-14
    inv = np.zeros(len(det))
    inv[good] = 1 / det[good]
    d = a - t0
    u = np.einsum("ij,ij->i", d, p) * inv
    q = np.cross(d, e1)
    v, t = q @ direction * inv, np.einsum("ij,ij->i", e2, q) * inv
    good &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t >= -1e-9) & (t <= 1 + 1e-9)
    ts = t[good]
    return float(np.median(ts)) if len(ts) and np.ptp(ts) < 1e-7 else None


def opening_overrun_domain(xyz: np.ndarray, sid: np.ndarray, s: np.ndarray,
                           segments: list[dict], openings: list[dict], *,
                           inside: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Identify contiguous atlas tails beyond their own measured openings.

    A point must lie strictly outward of the entire rim's plane-residual band
    or a verified 3D cap facet, with its projection inside the opening polygon.
    The latter also requires the independently closed-surface audit to agree
    that the point is outside; a plane tolerance is never increased.
    Only an endpoint-connected run on the root/terminal branch is classified.
    Interior wall escapes are never removed from the anatomical inside audit.
    Returned bounds preserve the original atlas ``s`` coordinate; callers can
    gate section sampling and wall interpolation without relabelling arrays.
    """
    xyz, sid, s = np.asarray(xyz, float), np.asarray(sid), np.asarray(s, float)
    inside = None if inside is None else np.asarray(inside, bool)
    excluded = np.zeros(len(xyz), bool)
    bounds = {int(seg["segment_id"]): [float(seg["start_s_mm"]), float(seg["end_s_mm"])] for seg in segments}
    evidence, unresolved = [], []
    by_segment = {int(seg["segment_id"]): seg for seg in segments}
    for opening in openings:
        segment_id = int(opening["segment_id"])
        segment = by_segment.get(segment_id)
        if segment is None or not opening.get("valid"):
            continue
        is_inlet = opening.get("label_provisional") == "inlet" and segment["parent_id"] == -1
        is_outlet = not segment["children"] and opening.get("label_provisional") != "inlet"
        if not (is_inlet or is_outlet):
            continue
        rows = np.flatnonzero(sid == segment_id)
        rows = rows[np.argsort(s[rows])]
        if is_outlet:
            rows = rows[::-1]
        if len(rows) < 2:
            continue
        polygon = np.asarray(opening["polygon_xyz_mm"], float)
        center = np.asarray(opening["center_mm"], float)
        normal = unit(np.asarray(opening["normal"], float))
        e1, e2 = frame(normal)
        residual = float(np.max(np.abs((polygon - center) @ normal)))
        axial = (xyz[rows] - center) @ normal
        fan_center, fan_proof = _verified_fan_center(polygon)
        matched = []
        for j, row in enumerate(rows):
            if axial[j] <= residual + 1e-7:
                if fan_center is None or inside is None or inside[row]:
                    break
                cap_offset = _fan_cap_outward_offset(xyz[row], polygon, center, normal, fan_center)
                if cap_offset is None or cap_offset <= 1e-7:
                    break
            p2 = np.column_stack(((polygon - xyz[row]) @ e1, (polygon - xyz[row]) @ e2))
            if not contains_origin(p2):
                break
            matched.append(int(row))
        if not matched:
            continue
        # A fully external or sideways escaping branch cannot be certified as
        # a harmless terminal extension by an opening projection alone.
        next_row = int(rows[len(matched)]) if len(matched) < len(rows) else None
        if next_row is None or axial[len(matched)] > residual + 1e-7 or (inside is not None and not inside[next_row]):
            unresolved.append({"segment_id": segment_id, "opening": opening.get("label_provisional"),
                               "reason": "no_verified_interior_sample_after_opening_extension", "atlas_rows": matched})
            continue
        last = matched[-1]
        d0, d1 = float((xyz[last] - center) @ normal), float((xyz[next_row] - center) @ normal)
        alpha = float(np.clip(d0 / max(d0 - d1, 1e-20), 0., 1.))
        crossing_method = "fitted_rim_plane"
        if fan_center is not None:
            exact_alpha = _fan_cap_segment_crossing(xyz[last], xyz[next_row], polygon, fan_center)
            if exact_alpha is None:
                unresolved.append({"segment_id": segment_id, "opening": opening.get("label_provisional"),
                                   "reason": "no_unique_actual_cap_crossing", "atlas_rows": matched})
                continue
            alpha, crossing_method = exact_alpha, "verified_3d_cap_triangle_intersection"
        crossing_s = float(s[last] + alpha * (s[next_row] - s[last]))
        crossing = xyz[last] + alpha * (xyz[next_row] - xyz[last])
        excluded[matched] = True
        bounds[segment_id][0 if is_inlet else 1] = crossing_s
        evidence.append({"segment_id": segment_id, "opening": opening.get("label_provisional"),
                         "endpoint": "start" if is_inlet else "end", "atlas_rows": matched,
                         "outward_offsets_mm": [float((xyz[row] - center) @ normal) for row in matched],
                         "actual_cap_outward_offsets_mm": None if fan_center is None else [
                             _fan_cap_outward_offset(xyz[row], polygon, center, normal, fan_center) for row in matched],
                         "rim_plane_residual_max_mm": residual, "crossing_s_mm": crossing_s,
                         "crossing_xyz_mm": crossing.tolist(), "first_interior_atlas_row": next_row,
                         "projection_inside_measured_opening": True, "endpoint_contiguous": True,
                         "crossing_method": crossing_method, "fan_proof": fan_proof})
    return excluded, {"method": "endpoint_connected_outward_extrusion_of_own_measured_opening",
                      "excluded_atlas_rows": np.flatnonzero(excluded).tolist(),
                      "excluded_points": int(excluded.sum()), "segment_s_bounds_mm": bounds,
                      "extensions": evidence, "unresolved_extensions": unresolved,
                      "source_centerline_coordinates_changed": False,
                      "inside_tolerance_relaxed": False}


class SurfaceSlicer:
    """VTK plane cutter with connected closed contour selection by containment."""

    def __init__(self, xyz: np.ndarray, triangles: np.ndarray):
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        self.vtk = vtk
        # Keep geometric warnings/errors, avoid one INFO line per plane cut.
        vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_WARNING)
        self.surface = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.asarray(xyz, float), deep=True))
        self.surface.SetPoints(points)
        triangles, self.surface_cleanup_audit = _remove_opposite_triangle_fins(triangles)
        cells = vtk.vtkCellArray()
        packed = np.column_stack((np.full(len(triangles), 3), triangles)).astype(np.int64).ravel()
        cells.SetCells(len(triangles), numpy_to_vtkIdTypeArray(packed, deep=True))
        self.surface.SetPolys(cells)
        self.plane = vtk.vtkPlane()
        self.cutter = vtk.vtkPlaneCutter()
        self.cutter.SetInputData(self.surface)
        self.cutter.SetPlane(self.plane)
        self.cutter.BuildTreeOn()
        self.cutter.ComputeNormalsOff()

    def _lines(self, poly) -> tuple[list[np.ndarray], np.ndarray]:
        from vtk.util.numpy_support import vtk_to_numpy
        if poly.GetNumberOfPoints() == 0:
            return [], np.empty((0, 2, 3))
        clean = self.vtk.vtkCleanPolyData()
        clean.SetInputData(poly)
        clean.SetTolerance(1e-8)
        clean.Update()
        stripper = self.vtk.vtkStripper()
        stripper.SetInputConnection(clean.GetOutputPort())
        stripper.JoinContiguousSegmentsOn()
        stripper.Update()
        out = stripper.GetOutput()
        pts = vtk_to_numpy(out.GetPoints().GetData())
        ids = self.vtk.vtkIdList()
        lines = out.GetLines()
        lines.InitTraversal()
        paths, all_edges = [], []
        while lines.GetNextCell(ids):
            p = pts[[ids.GetId(i) for i in range(ids.GetNumberOfIds())]]
            if len(p) > 1:
                paths.append(p)
                all_edges.extend(np.stack((p[:-1], p[1:]), axis=1))
        return paths, np.asarray(all_edges).reshape(-1, 2, 3)

    def slice(self, center: np.ndarray, normal: np.ndarray) -> dict:
        self.plane.SetOrigin(*center)
        self.plane.SetNormal(*unit(normal))
        self.cutter.Update()
        paths, edges = self._lines(self.cutter.GetOutput())
        candidates = []
        closed_count = 0
        for p in paths:
            if len(p) < 4 or np.linalg.norm(p[0] - p[-1]) > 1e-5:
                continue
            closed_count += 1
            m = polygon_metrics(p, center, normal)
            if m.get("valid") and m.get("contains_center"):
                candidates.append(m)
        out = {"valid": False, "reason": "no_closed_contour_containing_center", "polygon_xyz_mm": np.empty((0, 3))}
        if len(candidates) == 1:
            out = candidates[0]
        elif len(candidates) > 1:
            out = min(candidates, key=lambda d: d["area_mm2"])
            out = {**out, "valid": False, "reason": "multiple_contours_contain_center"}
        out.update(intersection_segments_xyz_mm=edges, contour_count=len(paths), closed_contour_count=closed_count)
        return out

    def contours(self, center: np.ndarray, normal: np.ndarray) -> list[dict]:
        """Return every closed contour on a plane, including nonlocal loops.

        Unlike :meth:`slice`, this intentionally does not select the loop
        containing the plane origin. It is used by bifurcation diagnostics to
        test containment of multiple centerline daughter points.
        """
        self.plane.SetOrigin(*np.asarray(center, float))
        self.plane.SetNormal(*unit(normal))
        self.cutter.Update()
        paths, _ = self._lines(self.cutter.GetOutput())
        out = []
        for p in paths:
            if len(p) >= 4 and np.linalg.norm(p[0] - p[-1]) <= 1e-5:
                m = polygon_metrics(p, center, normal)
                if m.get("valid"):
                    out.append(m)
        return out

    def boundary_loops(self) -> list[np.ndarray]:
        edges = self.vtk.vtkFeatureEdges()
        edges.SetInputData(self.surface)
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.ManifoldEdgesOff()
        edges.Update()
        paths, _ = self._lines(edges.GetOutput())
        return [p[:-1] for p in paths if len(p) >= 4 and np.linalg.norm(p[0] - p[-1]) < 1e-5]

    def enclosed(self, points: np.ndarray) -> tuple[np.ndarray, dict]:
        """Geometry audit with opening caps; no flow/label-dependent masks."""
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy, numpy_to_vtkIdTypeArray
        fill = self.vtk.vtkFillHolesFilter()
        fill.SetInputData(self.surface)
        fill.SetHoleSize(1e6)
        fill.Update()
        def residual_edges(surface):
            edges = self.vtk.vtkFeatureEdges()
            edges.SetInputData(surface)
            edges.BoundaryEdgesOn()
            edges.NonManifoldEdgesOn()
            edges.FeatureEdgesOff()
            edges.ManifoldEdgesOff()
            edges.Update()
            return edges.GetOutput().GetNumberOfCells()

        closed = fill.GetOutput()
        raw_residual = residual = residual_edges(closed)
        cap_audit = {"method": "vtkFillHoles", "raw_remaining_boundary_or_nonmanifold_edges": int(raw_residual),
                     "source_points_moved": False, "source_triangles_changed": False,
                     "slicer_zero_thickness_fin_cleanup": self.surface_cleanup_audit}
        if raw_residual:
            # vtkFillHoles can leave an otherwise valid nonplanar opening
            # partly capped. Rebuild only the audit caps, with exact original
            # edge IDs and a verifiable fan where planar ear clipping fails.
            xyz = vtk_to_numpy(self.surface.GetPoints().GetData())
            triangles = vtk_to_numpy(self.surface.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
            cycles, source_audit = _boundary_cycles(triangles)
            cap_audit["source_edge_audit"] = source_audit
            cap_audit["opening_caps"] = []
            extra_points, extra_triangles = [], []
            complete = bool(cycles)
            for cycle in cycles:
                center, proof = _verified_fan_center(xyz[cycle])
                cap_audit["opening_caps"].append({"vertices": len(cycle), **proof})
                if center is None:
                    complete = False
                    break
                index = len(xyz) + len(extra_points)
                extra_points.append(center)
                extra_triangles.extend((int(a), int(b), index) for a, b in zip(cycle, np.roll(cycle, -1)))
            if complete:
                candidate = self.vtk.vtkPolyData()
                cap_points = self.vtk.vtkPoints()
                cap_points.SetData(numpy_to_vtk(np.concatenate([xyz, np.asarray(extra_points)]), deep=True))
                candidate.SetPoints(cap_points)
                all_triangles = np.concatenate([triangles, np.asarray(extra_triangles, np.int64)])
                cells = self.vtk.vtkCellArray()
                packed = np.column_stack([np.full(len(all_triangles), 3), all_triangles]).astype(np.int64).ravel()
                cells.SetCells(len(all_triangles), numpy_to_vtkIdTypeArray(packed, deep=True))
                candidate.SetPolys(cells)
                candidate_residual = residual_edges(candidate)
                cap_audit["fan_remaining_boundary_or_nonmanifold_edges"] = int(candidate_residual)
                if candidate_residual == 0:
                    closed, residual = candidate, 0
                    cap_audit.update(method="verified_boundary_centroid_fans", cap_triangles=len(extra_triangles))
        pdata = self.vtk.vtkPolyData()
        vp = self.vtk.vtkPoints()
        vp.SetData(numpy_to_vtk(np.asarray(points, float), deep=True))
        pdata.SetPoints(vp)
        enclosed = self.vtk.vtkSelectEnclosedPoints()
        enclosed.SetInputData(pdata)
        enclosed.SetSurfaceData(closed)
        enclosed.SetTolerance(1e-7)
        enclosed.Update()
        inside = vtk_to_numpy(enclosed.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)
        return inside, {"closed_reference_valid": residual == 0, "remaining_boundary_or_nonmanifold_edges": residual,
                        "caps": cap_audit["method"] + "_for_inside_audit_only", "cap_audit": cap_audit}


def pointcloud_section(points: np.ndarray, center: np.ndarray, normal: np.ndarray,
                       radius_hint: float, spacing_mm: float, *, bins: int = 36) -> dict:
    """Bare-cloud slab + radial medians; only a star-convex candidate.

    Missing sectors and multi-valued rays are failures, not silently assumed
    circles. Slab thickness is fixed between density variants by caller.
    """
    normal = unit(normal)
    e1, e2 = frame(normal)
    d = np.asarray(points, float) - center
    axial = d @ normal
    thickness = max(0.75, min(2.0, 2.0 * spacing_mm))
    d = d[np.abs(axial) <= thickness]
    p = np.column_stack((d @ e1, d @ e2))
    r = np.linalg.norm(p, axis=1)
    # Generous geometry-only guard against an adjacent distant vessel.
    keep = (r > 1e-5) & (r < max(4 * radius_hint, 10.0))
    p, r = p[keep], r[keep]
    out = {"valid": False, "reason": "insufficient_slab_points", "polygon_xyz_mm": np.empty((0, 3)),
           "slab_halfwidth_mm": thickness, "slab_points": int(len(r)), "method": "star_radial_median_36"}
    if len(r) < 24:
        return out
    theta = np.mod(np.arctan2(p[:, 1], p[:, 0]), 2 * np.pi)
    bi = np.floor(theta / (2 * np.pi) * bins).astype(int).clip(0, bins - 1)
    med = np.full(bins, np.nan)
    multi = 0
    for j in range(bins):
        v = r[bi == j]
        if len(v):
            med[j] = np.median(v)
        if len(v) >= 4 and np.percentile(v, 90) / max(np.percentile(v, 10), 1e-6) > 1.6:
            multi += 1
    known = np.flatnonzero(np.isfinite(med))
    gaps = np.diff(np.r_[known, known[0] + bins])
    out.update(occupied_sectors=int(len(known)), max_gap_sectors=int(gaps.max()), multimodal_sectors=multi)
    if len(known) < bins // 2 or gaps.max() > 4:
        out["reason"] = "angular_coverage_gap"
        return out
    rr = np.interp(np.arange(bins), np.r_[known[-1] - bins, known, known[0] + bins],
                   np.r_[med[known[-1]], med[known], med[known[0]]])
    aa = (np.arange(bins) + 0.5) * (2 * np.pi / bins)
    polygon = center + (rr * np.cos(aa))[:, None] * e1 + (rr * np.sin(aa))[:, None] * e2
    metrics = polygon_metrics(polygon, center, normal)
    out.update(metrics)
    if multi:
        out.update(valid=False, reason="multi_radial_modes_or_nonstar_or_other_branch")
    out["applicability"] = "star_convex_about_centerline_unproven_for_nonstar_shapes"
    return out


def project_wall(points: np.ndarray, xyz: np.ndarray, sid: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    """Continuous closest point on each branch polyline; ambiguity across branches.

    A midpoint KD tree finds eight local candidate edges per branch. Atlas
    sampling is approximately uniform at 0.5 mm; this is a local search, not a
    proof of the globally closest edge for arbitrary pathological polylines.
    """
    n = len(points)
    best = np.full(n, np.inf)
    second = np.full(n, np.inf)
    chosen = np.full(n, -1, np.int16)
    ss = np.full(n, np.nan)
    for seg in np.unique(sid):
        rows = np.flatnonzero(sid == seg)
        rows = rows[np.argsort(s[rows])]
        if len(rows) < 2:
            continue
        a, b = xyz[rows[:-1]], xyz[rows[1:]]
        delta = b - a
        length2 = np.sum(delta * delta, axis=1)
        tree = cKDTree((a + b) / 2)
        k = min(8, len(a))
        for start in range(0, n, 4096):
            stop = min(n, start + 4096)
            p = points[start:stop]
            ix = tree.query(p, k=k)[1].reshape(len(p), k)
            alpha = np.clip(np.sum((p[:, None] - a[ix]) * delta[ix], axis=2) / np.maximum(length2[ix], 1e-20), 0, 1)
            dist2 = np.sum((p[:, None] - a[ix] - alpha[:, :, None] * delta[ix]) ** 2, axis=2)
            j = np.argmin(dist2, axis=1)
            local = np.arange(len(p))
            dist = np.sqrt(dist2[local, j])
            edge = ix[local, j]
            loc_s = s[rows[edge]] + alpha[local, j] * (s[rows[edge + 1]] - s[rows[edge]])
            win = dist < best[start:stop]
            second[start:stop] = np.where(win, best[start:stop], np.minimum(second[start:stop], dist))
            best[start:stop] = np.minimum(best[start:stop], dist)
            chosen[start:stop] = np.where(win, seg, chosen[start:stop])
            ss[start:stop] = np.where(win, loc_s, ss[start:stop])
    return {"segment_id": chosen, "s_local_mm": ss, "distance_mm": best,
            "second_branch_distance_mm": second, "ambiguous": second <= 1.2 * np.maximum(best, 1e-8)}


def log_area_slopes(s: np.ndarray, area: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """d(log A)/ds, no differentiation across missing runs or branches."""
    out = np.full(len(s), np.nan)
    idx = np.flatnonzero(valid & np.isfinite(area) & (area > 0))
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(run) >= 3:
            out[run] = np.gradient(np.log(area[run]), s[run], edge_order=2)
    return out
