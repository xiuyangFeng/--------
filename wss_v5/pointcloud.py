"""Deployment-side geometry program: wall point cloud + centerline atlas -> normals, caps, inside test, internal queries.

Nothing in this module reads CFD data.  The pilot gate feeds the CFD wall nodes
in as a bare point cloud (coordinates only) and judges the result against the
true cell zones; that judging code lives in ``build_case.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from . import contract as C
from .centerline_features import Atlas, map_points


@dataclass
class OrientedCloud:
    points_mm: np.ndarray
    normals_out: np.ndarray
    areas_mm2: np.ndarray
    is_cap: np.ndarray
    spacing_mm: float
    tree: cKDTree


def median_spacing(points: np.ndarray) -> float:
    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))


def pca_normals(points: np.ndarray, atlas: Atlas, k: int = C.PCA_NORMAL_K) -> tuple[np.ndarray, np.ndarray]:
    """Local-PCA normals with globally consistent outward orientation.

    1. unoriented normals from the k-NN covariance;
    2. provisional sign = away from the nearest centerline sample (good on
       tubular segments, unreliable on folded aneurysm lobes);
    3. orientation propagation along a minimum spanning tree of the k-NN graph
       (Hoppe et al. 1992) so that neighbouring normals agree, with the global
       sign of each connected component fixed by the centerline majority.

    Returns ``(normals_out, surface_variation)``; ``pca_normals.last_info``
    holds propagation diagnostics.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree

    pts = np.asarray(points, dtype=np.float64)
    n_pts = len(pts)
    tree = cKDTree(pts)
    k = min(k, n_pts)
    _, nn = tree.query(pts, k=k)
    neigh = pts[nn]  # (N, k, 3)
    centred = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centred, centred) / max(k - 1, 1)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    normals = evecs[:, :, 0].copy()
    variation = evals[:, 0] / np.maximum(evals.sum(axis=1), 1.0e-18)
    # provisional orientation from the centerline
    _, nearest = atlas.tree.query(pts, k=1)
    centre = atlas.xyz[atlas.tree_rows[np.asarray(nearest).reshape(-1)]]
    outward = pts - centre
    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1.0e-12)
    heuristic_dot = np.einsum("ij,ij->i", normals, outward)
    normals[heuristic_dot < 0] *= -1.0
    heuristic_dot = np.abs(heuristic_dot)
    # MST propagation on the k-NN graph.  Edge weight = (1 - |n_i.n_j|) + 2 * "across-gap" term, where the
    # across-gap term is |d.n| / |d| for the connecting vector d: ~0 for two points on one smooth surface,
    # ~1 for points on two facing walls (touching iliac branches, folded sacs) so the tree avoids such edges.
    kk = min(10, k)
    rows = np.repeat(np.arange(n_pts), kk - 1)
    cols = nn[:, 1:kk].reshape(-1)
    d = pts[cols] - pts[rows]
    dn = np.maximum(np.linalg.norm(d, axis=1), 1.0e-12)
    across = 0.5 * (np.abs(np.einsum("ij,ij->i", d, normals[rows])) + np.abs(np.einsum("ij,ij->i", d, normals[cols]))) / dn
    weight = 1.0 - np.abs(np.einsum("ij,ij->i", normals[rows], normals[cols])) + 2.0 * across + 1.0e-6
    graph = coo_matrix((weight, (rows, cols)), shape=(n_pts, n_pts)).tocsr()
    graph = graph.maximum(graph.T)
    mst = minimum_spanning_tree(graph).tocsr()
    und = (mst + mst.T).tocsr()
    n_comp, comp = connected_components(und, directed=False)
    flipped = 0
    oriented = normals.copy()
    for c in range(n_comp):
        members = np.flatnonzero(comp == c)
        if len(members) == 1:
            continue
        root = members[np.argmax(heuristic_dot[members])]
        order, pred = breadth_first_order(und, root, directed=False, return_predecessors=True)
        for node in order[1:]:
            parent = pred[node]
            if oriented[parent] @ oriented[node] < 0.0:
                oriented[node] *= -1.0
        # global sign of the component: agree with the centerline majority
        agree = np.einsum("ij,ij->i", oriented[members], outward[members])
        if np.sum(agree) < 0.0:
            oriented[members] *= -1.0
        flipped += int(np.sum(np.einsum("ij,ij->i", oriented[members], normals[members]) < 0.0))
    # restoration: where the centerline heuristic is confident (tubular wall at ~ the inscribed radius with the
    # normal clearly pointing away from the axis) it is right by construction; undo propagation flips there.
    radius = atlas.col("radius_mm")[atlas.tree_rows[np.asarray(nearest).reshape(-1)]]
    rho = np.linalg.norm(pts - centre, axis=1) / np.maximum(radius, 1.0e-6)
    confident = (heuristic_dot > 0.8) & (rho > 0.7) & (rho < 1.3)
    disagree = confident & (np.einsum("ij,ij->i", oriented, outward) < 0.0)
    oriented[disagree] *= -1.0
    restored = int(disagree.sum())
    pca_normals.last_info = {"components": int(n_comp), "flipped_vs_centerline_heuristic": flipped,
                             "flipped_fraction": flipped / max(n_pts, 1), "restored_to_heuristic": restored,
                             "confident_fraction": float(np.mean(confident))}
    return oriented, variation


def _disk_points(center: np.ndarray, outward: np.ndarray, radius: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Sunflower sampling of a disk; returns points and equal area weights."""
    n = max(int(np.pi * radius * radius / max(spacing * spacing, 1.0e-12)), 8)
    i = np.arange(n) + 0.5
    r = radius * np.sqrt(i / n)
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    t = outward / max(float(np.linalg.norm(outward)), 1.0e-12)
    hint = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = hint - (hint @ t) * t
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    pts = center[None, :] + r[:, None] * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)
    return pts, np.full(n, np.pi * radius * radius / n)


def _rim_plane(center: np.ndarray, outward: np.ndarray, radius: float, wall_points_mm: np.ndarray, tree: cKDTree) -> dict[str, Any] | None:
    """Fit the opening plane to the rim of the wall cloud around a centerline endpoint.

    Wall points near the endpoint are binned by azimuth; the most distal point of
    each sector is a rim sample; a least-squares plane through the rim samples
    gives the cap centre/normal (handles oblique cuts).  Deployment-legal.
    """
    idx = tree.query_ball_point(center, r=3.0 * radius)
    if len(idx) < 24:
        return None
    rel = wall_points_mm[idx] - center
    axial = rel @ outward
    lateral = rel - axial[:, None] * outward[None, :]
    lat = np.linalg.norm(lateral, axis=1)
    keep = (lat <= 1.6 * radius) & (axial >= -2.5 * radius)
    if keep.sum() < 24:
        return None
    hint = np.array([1.0, 0.0, 0.0]) if abs(outward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = hint - (hint @ outward) * outward
    u /= np.linalg.norm(u)
    v = np.cross(outward, u)
    ang = np.arctan2(lateral[keep] @ v, lateral[keep] @ u)
    sector = np.floor((ang + np.pi) / (2.0 * np.pi) * 24).astype(int) % 24
    ax_keep = axial[keep]
    pts_keep = wall_points_mm[idx][keep]
    rim = []
    for sct in range(24):
        members = np.flatnonzero(sector == sct)
        if len(members):
            rim.append(pts_keep[members[np.argmax(ax_keep[members])]])
    if len(rim) < 8:
        return None
    rim = np.asarray(rim)
    c_fit = rim.mean(axis=0)
    _, _, vt = np.linalg.svd(rim - c_fit, full_matrices=False)
    n_fit = vt[-1]
    if n_fit @ outward < 0:
        n_fit = -n_fit
    tilt = float(np.degrees(np.arccos(np.clip(n_fit @ outward, -1.0, 1.0))))
    if tilt > 40.0:  # degenerate fit: fall back to the tangent direction through the rim centroid
        n_fit = outward
    in_plane = rim - c_fit
    in_plane -= (in_plane @ n_fit)[:, None] * n_fit[None, :]
    r_fit = float(np.percentile(np.linalg.norm(in_plane, axis=1), 90))
    return {"center": c_fit, "normal": n_fit, "radius": max(r_fit, 0.8 * radius), "rim_samples": int(len(rim)), "tilt_deg": tilt}


def virtual_caps(atlas: Atlas, spacing: float, wall_points_mm: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Disk caps closing the openings.

    With a wall cloud, each cap is the least-squares plane through the rim of
    the cloud around the centerline endpoint (``_rim_plane``); otherwise the
    atlas endpoint, tangent and radius are used directly.
    """
    caps = []
    tree = cKDTree(wall_points_mm) if wall_points_mm is not None else None
    for end in atlas.endpoints():
        center = np.asarray(end["center_mm"], dtype=np.float64)
        outward = np.asarray(end["outward"], dtype=np.float64)
        radius = float(end["radius_mm"])
        fit = _rim_plane(center, outward, radius, wall_points_mm, tree) if tree is not None else None
        if fit is not None:
            cap_center, cap_normal, cap_radius = fit["center"], fit["normal"], fit["radius"]
            source = "rim_plane_fit"
        else:
            cap_center, cap_normal, cap_radius, source = center, outward, radius, "atlas_endpoint"
        pts, areas = _disk_points(cap_center, cap_normal, cap_radius, spacing * C.CAP_POINT_SPACING_FACTOR)
        caps.append({**end, "center_mm": cap_center, "outward": cap_normal, "radius_mm": cap_radius, "atlas_endpoint_mm": center,
                     "atlas_radius_mm": radius, "rim_snap_mm": float((cap_center - center) @ outward), "cap_source": source,
                     "rim_fit": {k: v for k, v in (fit or {}).items() if k in ("rim_samples", "tilt_deg")},
                     "points_mm": pts, "areas_mm2": areas})
    return caps


def local_areas(points: np.ndarray, k: int = 6) -> np.ndarray:
    """Density-based nominal area per point: (mean distance to the k nearest neighbours)^2."""
    d, _ = cKDTree(points).query(points, k=k + 1)
    return np.mean(d[:, 1:], axis=1) ** 2


def calibration_points(atlas: Atlas) -> np.ndarray:
    """Centerline samples deep inside the lumen (away from openings and junctions)."""
    rows = atlas.tree_rows
    radius = atlas.col("radius_mm")[rows]
    ok = (atlas.col("dist_to_endpoint_mm")[rows] > 2.0 * radius) & (atlas.col("dist_to_junction_mm")[rows] > radius)
    if ok.sum() < 20:
        ok = atlas.col("endpoint_mask")[rows] < 0.5
    return atlas.xyz[rows[ok]]


def build_oriented_cloud(points_mm: np.ndarray, atlas: Atlas, *, normals_out: np.ndarray | None = None,
                         areas_mm2: np.ndarray | None = None, calibrate: bool = True) -> tuple[OrientedCloud, dict[str, Any]]:
    """Assemble wall points (+ estimated normals/areas when absent) and virtual caps.

    When areas are estimated, they are rescaled so that the generalised winding
    number at interior centerline samples has median 1 (self-calibration that
    uses only the cloud and the centerline).
    """
    pts = np.asarray(points_mm, dtype=np.float64)
    spacing = median_spacing(pts)
    info: dict[str, Any] = {"spacing_mm": spacing, "normals_source": "given" if normals_out is not None else "pca+centerline"}
    if normals_out is None:
        normals_out, variation = pca_normals(pts, atlas)
        info["surface_variation_median"] = float(np.median(variation))
        info["normal_orientation"] = getattr(pca_normals, "last_info", {})
    estimated_areas = areas_mm2 is None
    if estimated_areas:
        areas_mm2 = local_areas(pts)
        info["areas_source"] = "local_knn_density"
    caps = virtual_caps(atlas, spacing, pts)
    cap_pts = np.concatenate([c["points_mm"] for c in caps])
    cap_nrm = np.concatenate([np.repeat(c["outward"][None, :], len(c["points_mm"]), axis=0) for c in caps])
    cap_area = np.concatenate([c["areas_mm2"] for c in caps])
    all_pts = np.concatenate([pts, cap_pts])
    all_nrm = np.concatenate([np.asarray(normals_out, dtype=np.float64), cap_nrm])
    all_area = np.concatenate([np.asarray(areas_mm2, dtype=np.float64), cap_area])
    is_cap = np.concatenate([np.zeros(len(pts), bool), np.ones(len(cap_pts), bool)])
    info["caps"] = [{"label": c["label"], "center_mm": np.asarray(c["center_mm"]).tolist(), "outward": np.asarray(c["outward"]).tolist(),
                     "radius_mm": float(c["radius_mm"]), "atlas_radius_mm": float(c["atlas_radius_mm"]), "rim_snap_mm": c["rim_snap_mm"],
                     "cap_source": c["cap_source"], "rim_fit": c["rim_fit"], "points": int(len(c["points_mm"]))} for c in caps]
    cloud = OrientedCloud(all_pts, all_nrm, all_area, is_cap, spacing, cKDTree(all_pts))
    if estimated_areas and calibrate:
        calib = calibration_points(atlas)
        w = winding_number(calib, cloud)
        factor = 1.0 / max(float(np.median(w)), 1.0e-3)
        cloud.areas_mm2[~is_cap] *= factor
        w2 = winding_number(calib, cloud)
        info["winding_calibration"] = {"points": int(len(calib)), "median_before": float(np.median(w)), "area_scale": factor,
                                       "median_after": float(np.median(w2)), "p10_after": float(np.percentile(w2, 10)), "p90_after": float(np.percentile(w2, 90))}
    info["wall_area_estimate_mm2"] = float(cloud.areas_mm2[~is_cap].sum())
    return cloud, info


def inside_score_vote(query_mm: np.ndarray, cloud: OrientedCloud, k: int = C.INSIDE_VOTE_K) -> np.ndarray:
    """Signed nearest-neighbour side score: positive inside (behind the outward normals)."""
    q = np.asarray(query_mm, dtype=np.float64)
    d, nn = cloud.tree.query(q, k=k)
    d = np.atleast_2d(d).reshape(len(q), k)
    nn = np.atleast_2d(nn).reshape(len(q), k)
    p = cloud.points_mm[nn]
    n = cloud.normals_out[nn]
    rel = q[:, None, :] - p
    side = -np.einsum("ijk,ijk->ij", rel, n) / np.maximum(d, 1.0e-9)
    w = 1.0 / np.maximum(d, 0.25 * cloud.spacing_mm) ** 2
    return np.einsum("ij,ij->i", w, side) / w.sum(axis=1)


def winding_number(query_mm: np.ndarray, cloud: OrientedCloud, chunk: int = 128) -> np.ndarray:
    """Generalised winding number of an oriented point cloud (Barill et al. 2018), brute force."""
    q = np.asarray(query_mm, dtype=np.float32)
    p = cloud.points_mm.astype(np.float32)
    an = (cloud.normals_out * cloud.areas_mm2[:, None]).astype(np.float32)  # area-weighted normals
    out = np.empty(len(q), dtype=np.float64)
    for start in range(0, len(q), chunk):
        block = q[start:start + chunk]
        rel = p[None, :, :] - block[:, None, :]  # (c, M, 3)
        r2 = np.einsum("cmk,cmk->cm", rel, rel)
        num = np.einsum("cmk,mk->cm", rel, an)
        out[start:start + chunk] = (num / np.maximum(r2, 1.0e-8) ** 1.5).sum(axis=1, dtype=np.float64) / (4.0 * np.pi)
    return out


def inside_hybrid(query_mm: np.ndarray, cloud: OrientedCloud, *, confident: float = 0.85, cap_margin_spacing: float = 4.0,
                  caps: list[dict[str, Any]] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Vote for confidently inside/outside points far from the openings; winding number for the rest."""
    q = np.asarray(query_mm, dtype=np.float64)
    score = inside_score_vote(q, cloud)
    near_cap = np.zeros(len(q), dtype=bool)
    cap_pts = cloud.points_mm[cloud.is_cap]
    if len(cap_pts):
        d_cap, _ = cKDTree(cap_pts).query(q, k=1)
        near_cap = d_cap <= cap_margin_spacing * cloud.spacing_mm
    decided_in = (score > confident) & ~near_cap
    decided_out = (score < -confident) & ~near_cap
    undecided = ~(decided_in | decided_out)
    inside = decided_in.copy()
    if np.any(undecided):
        w = winding_number(q[undecided], cloud)
        inside[undecided] = w > 0.5
    return inside, {"winding_fraction": float(np.mean(undecided)), "near_cap_fraction": float(np.mean(near_cap)),
                    "vote_confident_in": float(np.mean(decided_in)), "vote_confident_out": float(np.mean(decided_out))}


def generate_internal_queries(atlas: Atlas, cloud: OrientedCloud, *, n_target: int, seed: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    """Candidate points from a centerline tube sweep plus a bounding-box fill, filtered by the hybrid inside test."""
    rng = np.random.default_rng(seed)
    rows = atlas.tree_rows
    xyz = atlas.xyz[rows]
    tan = atlas.tangent[rows]
    n_vec = atlas.frame_n[rows]
    b_vec = atlas.frame_b[rows]
    radius = atlas.col("radius_mm")[rows] * (1.0 + C.TUBE_MARGIN)
    step = float(np.median(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    # tube candidates proportional to local cross-section
    weights = radius ** 2
    n_tube = int(n_target * 0.7)
    pick = rng.choice(len(rows), size=n_tube, p=weights / weights.sum())
    u = np.sqrt(rng.random(n_tube))
    phi = rng.random(n_tube) * 2.0 * np.pi
    axial = (rng.random(n_tube) - 0.5) * step
    r = radius[pick] * u
    tube = xyz[pick] + r[:, None] * (np.cos(phi)[:, None] * n_vec[pick] + np.sin(phi)[:, None] * b_vec[pick]) + axial[:, None] * tan[pick]
    # bounding-box fill near the wall cloud (covers aneurysm sacs the tube sweep misses)
    wall = cloud.points_mm[~cloud.is_cap]
    lo, hi = wall.min(axis=0), wall.max(axis=0)
    n_box = n_target - n_tube
    box = lo + rng.random((n_box * 3, 3)) * (hi - lo)
    d, _ = cloud.tree.query(box, k=1)
    box = box[d <= np.percentile(radius, 95)][:n_box]
    cand = np.concatenate([tube, box])
    inside, diag = inside_hybrid(cand, cloud)
    kept = cand[inside]
    return kept, {"candidates": int(len(cand)), "tube": int(len(tube)), "box": int(len(box)), "kept": int(len(kept)), **diag}
