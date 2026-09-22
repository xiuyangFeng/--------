"""STL-only PF6/VF6 support/query geometry, with explicit lumen containment.

The first rows are wall support points and subsequent rows are interior queries.
Features follow ``volume_view v1.1``: wall PCA normals become transverse atlas
radial vectors in the interior, and distances are to the *wall triangles*, not
to the artificial inlet/outlet caps.  This module never loads field labels.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from wss_features.atlas import Atlas, map_points
from wss_features.cloud import build_oriented_cloud, generate_internal_queries, pca_normals
from wss_features.flowref import Tree as _Tree, murray_shares as _murray_shares
from wss_features.frame import anatomical_frame

GEOMETRY_VERSION = "stl-volume-queries/v3"
MIN_EXACT_UNIFORM_FRACTION = 0.30
# The adaptive component is deliberately small enough to leave an exact
# volume-uniform component in every case.  A one-percent floor is useful for
# the 20k deployment default (140 points for a seven-segment atlas), while the
# absolute floor keeps small smoke cases from silently dropping a branch.
MIN_ADAPTIVE_BRANCH_FRACTION = 0.01
MIN_ADAPTIVE_BRANCH_POINTS = 16


def _surface(vertices: np.ndarray, faces: np.ndarray):
    import pyvista as pv

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("lumen vertices must be finite [N,3] coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("lumen faces must be nonempty [M,3] triangles")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("lumen triangle indices are outside the vertex array")
    return pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces))).clean()


def _boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("体场采样要求流形表面；当前 STL 含非流形边。")
    boundary = edges[counts == 1]
    if not len(boundary):
        return []
    vertices, degree = np.unique(boundary, return_counts=True)
    if np.any(degree != 2):
        raise ValueError("体场采样无法封闭非环状切口，请修复 STL 边界。")
    neighbors = {int(v): [] for v in vertices}
    for a, b in boundary:
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))
    remaining = set(neighbors)
    loops = []
    while remaining:
        start = min(remaining)
        current, previous = start, -1
        loop = []
        while True:
            loop.append(current)
            remaining.remove(current)
            following = next(v for v in neighbors[current] if v != previous)
            previous, current = current, following
            if current == start:
                break
            if current not in remaining:
                raise ValueError("STL boundary does not form disjoint simple loops")
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def close_lumen(vertices: np.ndarray, faces: np.ndarray, atlas: Atlas | None = None):
    """Cap actual boundary polygons, retaining concave cross-section boundaries.

    Caps are used only for inside/outside decisions, never as velocity queries
    or wall support.  When an atlas is supplied, every opening must match an
    anatomical endpoint.  Unknown wall holes cannot silently become caps.
    Returns a closed PyVista triangle surface and a JSON-safe audit record.
    """
    mesh = _surface(vertices, faces)
    triangles = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    loops = _boundary_loops(triangles)
    caps = []
    for loop in loops:
        points = mesh.points[loop]
        center = points.mean(axis=0)
        centered = points - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        radius = float(np.sqrt(np.linalg.norm(np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)) / (2 * np.pi)))
        rms = float(np.sqrt(np.mean((centered @ vt[-1]) ** 2)))
        if radius <= 1e-8 or rms > max(0.25 * radius, 0.05):
            raise ValueError("STL 切口无法可靠地作平面封口，请检查切口形状。")
        caps.append({"center_mm": center.tolist(), "radius_mm": radius, "planarity_rms_mm": rms,
                     "boundary_vertices": len(loop)})
    if atlas is not None and caps:
        ends = atlas.endpoints()
        if len(caps) != len(ends):
            raise ValueError(f"STL 开口数 {len(caps)} 与中心线端点数 {len(ends)} 不一致，不能可靠采样体内点。")
        distances = np.linalg.norm(np.asarray([c["center_mm"] for c in caps])[:, None] -
                                   np.asarray([e["center_mm"] for e in ends])[None], axis=2)
        rr, cc = linear_sum_assignment(distances)
        for i, j in zip(rr, cc):
            allowed = max(3.0 * float(ends[j]["radius_mm"]), 5.0)
            if distances[i, j] > allowed:
                raise ValueError("STL 的开放边界不在对应中心线端点附近，请检查切口与中心线。")
            caps[i].update(label=ends[j]["label"], endpoint_distance_mm=float(distances[i, j]))
    if loops:
        # vtkFillHolesFilter triangulates the actual boundary polygon.  A disk
        # or convex hull would incorrectly fill space outside a concave lumen.
        mesh = mesh.fill_holes(float(np.linalg.norm(np.ptp(mesh.points, axis=0))) * 2.0).triangulate().clean()
    remaining = _boundary_loops(np.asarray(mesh.faces).reshape(-1, 4)[:, 1:])
    if remaining:
        raise ValueError("STL 切口封闭失败，体场计算已停止以避免在血管外预测。")
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False, consistent_normals=True,
                                auto_orient_normals=True, split_vertices=False, inplace=False)
    return mesh, {"method": "triangle_boundary_polygon_caps", "cap_count": len(caps), "caps": caps,
                  "closed_triangle_count": int(mesh.n_cells), "self_intersections": "not_evaluated"}


def make_inside_test(vertices: np.ndarray, faces: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Create a reusable exact triangle ray test for an already closed surface.

    The signed nearest-normal heuristic alone leaks through folds and narrow
    concave branches; VTK's closed-surface ray voting is the final authority.
    """
    import vtk

    mesh = _surface(vertices, faces)
    if _boundary_loops(np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]):
        raise ValueError("inside test requires a closed triangle surface")
    selector = vtk.vtkSelectEnclosedPoints()
    selector.SetTolerance(1e-9)
    selector.Initialize(mesh)

    def contains(points: np.ndarray) -> np.ndarray:
        q = np.asarray(points, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
            raise ValueError("query points must be finite [N,3] coordinates")
        if not len(q):
            return np.zeros(0, dtype=bool)
        return np.fromiter((bool(selector.IsInsideSurface(point)) for point in q), dtype=bool, count=len(q))

    return contains


def _wall_distances(points: np.ndarray, wall_mesh) -> np.ndarray:
    import pyvista as pv

    # Matches the training volume_static/dist_to_wall_mm contract: closest
    # distance to actual wall triangles, excluding artificial end caps.
    return np.abs(np.asarray(pv.PolyData(points).compute_implicit_distance(wall_mesh)["implicit_distance"], dtype=np.float64))


def _branch_profiles(atlas: Atlas) -> dict[int, dict[str, float]]:
    """Return length/radius profiles used by adaptive query allocation.

    The atlas tree rows are used instead of all rows because child-junction
    duplicates would otherwise give the parent an artificial length.  The
    profile is geometry-only and does not use CFD labels or model outputs.
    """
    segment = atlas.col("segment_id").astype(int)
    xyz = atlas.xyz
    radius = np.asarray(atlas.col("radius_mm"), dtype=np.float64)
    profiles: dict[int, dict[str, float]] = {}
    for raw in atlas.segments:
        sid = int(raw["segment_id"])
        rows = atlas.tree_rows[segment[atlas.tree_rows] == sid]
        if not len(rows):
            continue
        order = np.argsort(atlas.col("sample_index")[rows])
        rows = rows[order]
        length = float(np.linalg.norm(np.diff(xyz[rows], axis=0), axis=1).sum()) if len(rows) > 1 else 0.0
        r = float(np.median(np.clip(radius[rows], 1.0e-6, None)))
        # Length / sqrt(radius) retains a volume-related length term while
        # giving narrow branches a controlled boost over radius**2 sampling.
        profiles[sid] = {"length_mm": max(length, r), "radius_mm": r,
                         "weight": max(length, r) / np.sqrt(r)}
    return profiles


def _radius_weighted_branch_quotas(n_points: int, profiles: dict[int, dict[str, float]],
                                   *, min_fraction: float = MIN_ADAPTIVE_BRANCH_FRACTION,
                                   min_points: int = MIN_ADAPTIVE_BRANCH_POINTS) -> tuple[dict[int, int], int]:
    """Allocate adaptive points by branch with a narrow-branch floor.

    ``n_points`` is the non-uniform/adaptive budget.  The returned minimum is
    reduced only when the budget cannot give every atlas segment a point; this
    makes that limitation explicit in diagnostics instead of silently
    over-allocating beyond the fixed model input size.
    """
    ids = sorted(int(sid) for sid in profiles)
    if not ids or n_points <= 0:
        return {sid: 0 for sid in ids}, 0
    floor = min(int(max(min_points, np.ceil(min_fraction * n_points))), n_points // len(ids))
    weights = np.asarray([max(float(profiles[sid]["weight"]), 1.0e-12) for sid in ids])
    remaining = n_points - floor * len(ids)
    raw = remaining * weights / weights.sum()
    quotas = np.full(len(ids), floor, dtype=np.int64) + np.floor(raw).astype(np.int64)
    leftover = int(remaining - np.floor(raw).sum())
    if leftover:
        order = np.argsort(-(raw - np.floor(raw)), kind="stable")
        quotas[order[:leftover]] += 1
    return {sid: int(value) for sid, value in zip(ids, quotas)}, int(floor)


def _segment_tube_candidates(atlas: Atlas, segment_id: int, n_candidates: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Generate targeted tube proposals for one segment.

    These are only fallback proposals for a branch whose global sweep had too
    few points.  They still go through the exact closed-surface filter before
    being used, so the targeted path cannot admit points outside the STL.
    """
    segment = atlas.col("segment_id").astype(int)
    rows = atlas.tree_rows[segment[atlas.tree_rows] == int(segment_id)]
    if not len(rows) or n_candidates <= 0:
        return np.empty((0, 3), dtype=np.float64)
    order = np.argsort(atlas.col("sample_index")[rows])
    rows = rows[order]
    xyz, tangent = atlas.xyz[rows], atlas.tangent[rows]
    normal, binormal = atlas.frame_n[rows], atlas.frame_b[rows]
    radius = np.clip(atlas.col("radius_mm")[rows].astype(np.float64), 1.0e-6, None)
    probability = radius ** 2
    probability /= probability.sum()
    pick = rng.choice(len(rows), size=int(n_candidates), p=probability)
    # Stay just inside the atlas radius; containment against the real STL is
    # still authoritative, and the small margin avoids over-proposing caps.
    radial = 0.96 * radius[pick] * np.sqrt(rng.random(len(pick)))
    phi = 2.0 * np.pi * rng.random(len(pick))
    return xyz[pick] + radial[:, None] * (np.cos(phi)[:, None] * normal[pick] +
                                          np.sin(phi)[:, None] * binormal[pick])


def _select_branch_stratified(candidates: np.ndarray, atlas: Atlas, target: int,
                              contains: Callable[[np.ndarray], np.ndarray],
                              rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Select the adaptive component with branch floors and fallback proposals."""
    profiles = _branch_profiles(atlas)
    quotas, branch_floor = _radius_weighted_branch_quotas(target, profiles)
    if not quotas or target <= 0:
        return np.empty((0, 3), dtype=np.float64), {"quotas": quotas, "selected": {},
                                                      "candidates": {}, "branch_min_points": branch_floor,
                                                      "supplemented": {}}

    points = np.asarray(candidates, dtype=np.float64)
    labels = map_points(points, atlas)["segment_id"].astype(int) if len(points) else np.empty(0, dtype=int)
    pools = {sid: points[labels == sid] for sid in quotas}
    candidate_counts = {sid: int(len(pool)) for sid, pool in pools.items()}
    supplemented: dict[int, int] = {}
    # A targeted pool is generated only for deficits.  Oversampling by eight
    # keeps the exact filter from turning a narrow outlet into a zero-pool
    # branch while bounding work for normal cases.
    for sid, quota in quotas.items():
        deficit = max(0, quota - len(pools[sid]))
        if not deficit:
            supplemented[sid] = 0
            continue
        accepted = np.empty((0, 3), dtype=np.float64)
        for _ in range(4):
            proposal_count = min(max(256, 8 * max(deficit - len(accepted), 1)), 50000)
            proposal = _segment_tube_candidates(atlas, sid, proposal_count, rng)
            if not len(proposal):
                break
            proposal = proposal[contains(proposal)]
            if len(proposal):
                branch = map_points(proposal, atlas)["segment_id"].astype(int)
                accepted = np.concatenate([accepted, proposal[branch == sid]])
            if len(accepted) >= deficit:
                break
        if len(accepted):
            pools[sid] = np.concatenate([pools[sid], accepted])
        supplemented[sid] = int(len(accepted))

    selected_parts = []
    selected: dict[int, int] = {}
    for sid, quota in quotas.items():
        pool = pools[sid]
        take = min(int(quota), len(pool))
        if take:
            selected_parts.append(pool[rng.choice(len(pool), take, replace=False)])
        selected[sid] = take
    selected_points = np.concatenate(selected_parts) if selected_parts else np.empty((0, 3), dtype=np.float64)
    return selected_points, {"quotas": quotas, "selected": selected,
                             "candidates": candidate_counts, "supplemented": supplemented,
                             "branch_min_points": branch_floor,
                             "min_satisfied": bool(all(selected[sid] >= branch_floor for sid in quotas)),
                             "profiles": profiles,
                             "radius_weighting": "length_mm / sqrt(median_radius_mm)"}


def sample_internal_points(wall_points: np.ndarray, vertices: np.ndarray, faces: np.ndarray,
                           atlas: Atlas, *, n_internal: int = 20000, seed: int = 0,
                           wall_normals: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Generate reproducible interior queries; reject every exterior candidate."""
    if isinstance(n_internal, bool) or not isinstance(n_internal, (int, np.integer)) or not 1 <= n_internal <= 200000:
        raise ValueError("n_internal must be an integer between 1 and 200000")
    wall = np.asarray(wall_points, dtype=np.float64)
    closed, closure = close_lumen(vertices, faces, atlas)
    closed_faces = np.asarray(closed.faces).reshape(-1, 4)[:, 1:]
    contains = make_inside_test(closed.points, closed_faces)
    cloud, cloud_info = build_oriented_cloud(wall, atlas, normals_out=wall_normals)
    candidates, sweep = generate_internal_queries(atlas, cloud, n_target=max(2 * n_internal, 1000), seed=int(seed))
    accepted_sweep = candidates[contains(candidates)]
    rejected = len(candidates) - len(accepted_sweep)
    rng = np.random.default_rng(int(seed) + 104729)
    lo, hi = np.min(closed.points, axis=0), np.max(closed.points, axis=0)
    # The hybrid point-cloud test can incorrectly reject a folded sac.  Always
    # include a separate exact-only uniform component, even when the tube sweep
    # already provides enough candidates; exact filtering cannot recover points
    # that an earlier heuristic discarded.
    minimum_uniform = int(np.ceil(n_internal * MIN_EXACT_UNIFORM_FRACTION))
    adaptive_target = n_internal - minimum_uniform
    adaptive_selected, adaptive_diag = _select_branch_stratified(
        accepted_sweep, atlas, adaptive_target, contains, rng)
    # If an STL/atlas mismatch leaves a branch with too few exact candidates,
    # let the independent exact-uniform component fill the remainder.  This
    # preserves the requested input size and raises the uniform fraction rather
    # than fabricating out-of-lumen branch points.
    sweep_count = len(adaptive_selected)
    uniform_target = n_internal - sweep_count
    accepted_uniform = np.empty((0, 3), dtype=np.float64)
    uniform_candidates = 0
    for _ in range(12):
        if len(accepted_uniform) >= uniform_target:
            break
        # Exact rejection sampling supplements tube proposals in aneurysm sacs.
        count = min(max(8 * (uniform_target - len(accepted_uniform)), 2048), 250000)
        proposal = rng.uniform(lo, hi, size=(count, 3))
        uniform_candidates += count
        accepted_uniform = np.concatenate([accepted_uniform, proposal[contains(proposal)]])
    if len(accepted_uniform) < uniform_target:
        raise ValueError(f"血管内精确均匀采样仅得到 {len(accepted_uniform)} / {uniform_target} 个点，请检查 STL 与中心线。")
    uniform_selected = accepted_uniform[rng.choice(len(accepted_uniform), uniform_target, replace=False)]
    adaptive_branch_counts = {str(k): int(v) for k, v in adaptive_diag.get("selected", {}).items()}
    uniform_branch_counts = {}
    if len(uniform_selected):
        labels = map_points(uniform_selected, atlas)["segment_id"].astype(int)
        uniform_branch_counts = {str(sid): int(np.sum(labels == sid)) for sid in np.unique(labels)}
    total_branch_counts = dict(uniform_branch_counts)
    for sid, count in adaptive_branch_counts.items():
        total_branch_counts[sid] = total_branch_counts.get(sid, 0) + count
    points = np.concatenate([adaptive_selected, uniform_selected])
    points = points[rng.permutation(len(points))]
    return points, {"closed_surface": {"vertices": np.asarray(closed.points), "faces": closed_faces},
                    "diag": {"geometry_version": GEOMETRY_VERSION, "closure": closure,
                             "internal_sampling": {"seed": int(seed), "n_internal": n_internal,
                                                   "sweep": sweep, "exact_filter_rejected": rejected,
                                                   "uniform_candidates": uniform_candidates,
                                                   "sweep_selected": sweep_count,
                                                   "exact_uniform_selected": uniform_target,
                                                   "exact_uniform_fraction": uniform_target / n_internal,
                                                   "minimum_exact_uniform_fraction": MIN_EXACT_UNIFORM_FRACTION,
                                                   "adaptive_target": adaptive_target,
                                                   "adaptive_selected": sweep_count,
                                                   "adaptive_branch_counts": adaptive_branch_counts,
                                                   "uniform_branch_counts": uniform_branch_counts,
                                                   "branch_counts": total_branch_counts,
                                                   "branch_stratification": adaptive_diag,
                                                   "distribution": "branch_radius_weighted_and_exact_uniform_mixture",
                                                   "containment": "closed_triangle_ray_test"},
                             "spacing_mm": float(cloud_info["spacing_mm"])}}


def build_volume_case(wall_points: np.ndarray, vertices: np.ndarray, faces: np.ndarray,
                      atlas: Atlas, input_features: list[str] | tuple[str, ...], *,
                      target: str = "pressure_velocity", case_name: str = "case",
                      n_internal: int = 20000, seed: int = 0) -> tuple[dict, dict]:
    """Build PF6/VF6 geometry without CFD, bundle or truth-derived statistics.

    ``pressure_mixed``/``pressure_velocity`` query wall + interior; ``velocity``
    queries only interior.  Frozen feature and target statistics remain the
    responsibility of the verified model release.  Pressure's learned gauge
    is relative to the case's volume-mean pressure, never absolute pressure.
    """
    if target not in {"pressure_mixed", "velocity", "pressure_velocity"}:
        raise ValueError(f"unsupported volume target: {target!r}")
    if any(int(value) < 0 for value in atlas.semantic_of_segment.values()):
        raise ValueError("体场特征需要已确认出口命名的中心线，请先确认出口。")
    wall = np.asarray(wall_points, dtype=np.float64)
    if wall.ndim != 2 or wall.shape[1] != 3 or len(wall) < 8 or not np.isfinite(wall).all():
        raise ValueError("volume support needs at least eight finite wall points")
    normals, _ = pca_normals(wall, atlas)
    interior, extra = sample_internal_points(wall, vertices, faces, atlas, n_internal=n_internal,
                                             seed=seed, wall_normals=normals)
    pts = np.concatenate([wall, interior])
    n_wall, n = len(wall), len(pts)
    feats = map_points(pts, atlas)
    frame = anatomical_frame(atlas.table, list(atlas.columns), {str(k): v for k, v in atlas.semantic_of_segment.items()})
    rotation = frame["rotation"]
    aligned = (pts - frame["origin_mm"]) @ rotation.T
    scale = max(float(np.abs(aligned[:n_wall]).max()), 1e-9)
    s_max = max(float(np.max(feats["s_from_root_mm"][:n_wall])), 1e-9)
    radius = np.clip(feats["radius_mm"].astype(np.float64), 1e-3, None)
    rows = feats["atlas_row"].astype(np.int64)
    tangent = atlas.tangent[rows]
    dvec = pts[n_wall:] - atlas.xyz[rows[n_wall:]]
    tan = tangent[n_wall:]
    dvec -= np.sum(dvec * tan, axis=1, keepdims=True) * tan
    norm = np.linalg.norm(dvec, axis=1, keepdims=True)
    radial = np.where(norm > 1e-6, dvec / np.clip(norm, 1e-12, None), 0.0)
    directions = np.concatenate([normals, radial])
    directions_aligned = directions @ rotation.T
    tree = _Tree(atlas.table, list(atlas.columns), atlas.segments, atlas.frame_n, atlas.frame_b)
    shares = _murray_shares(tree)
    log_q = np.array([np.log(max(shares[int(sid)], 1e-6)) for sid in feats["segment_id"]])
    wall_mesh = _surface(vertices, faces)
    dist = np.concatenate([np.zeros(n_wall), _wall_distances(interior, wall_mesh)])
    case = dict(cohort="deploy", case=case_name, unit_id=f"deploy/{case_name}",
                pos=(aligned / scale).astype(np.float32), wall_coords_raw=pts,
                frame_rotation=rotation, frame_origin_mm=frame["origin_mm"],
                coord_scale=np.full(n, scale, np.float32), coord_scale_scalar=scale,
                abscissa_norm=(feats["s_from_root_mm"] / s_max).astype(np.float32),
                local_radius=radius.astype(np.float32), log_local_radius=np.log(radius).astype(np.float32),
                curvature=feats["curvature_per_mm"].astype(np.float32),
                rho=feats["rho"].astype(np.float32), theta_sin=np.sin(feats["theta_rad"]).astype(np.float32),
                theta_cos=np.cos(feats["theta_rad"]).astype(np.float32),
                dist_to_wall_mm=dist.astype(np.float32), log_q_branch_murray=log_q.astype(np.float32),
                log_tau0_murray=(log_q - 3 * np.log(radius)).astype(np.float32),
                support_pool=np.arange(n_wall), query_pool=np.arange(n_wall if target == "velocity" else 0, n),
                n_wall=n_wall, point_kind=np.concatenate([np.zeros(n_wall, np.int8), np.ones(len(interior), np.int8)]),
                target_normalization="global_stats", peak_step=1162, bundle_path="",
                volume_view_version="v5_volume_view_v1.1", geometry_version=GEOMETRY_VERSION)
    for key in ("dr_ds", "dist_to_junction_mm", "dist_to_endpoint_mm", "end_zone"):
        case[key] = feats[key].astype(np.float32)
    case["radius_gradient"] = case["dr_ds"]
    case["_wall_segment_id"] = feats["segment_id"].astype(np.int64)
    for i, key in enumerate(("nx_aligned", "ny_aligned", "nz_aligned")):
        case[key] = directions_aligned[:, i].astype(np.float32)
    case["local_geometry"] = np.column_stack((directions_aligned, tangent @ rotation.T, radius, np.full(n, scale))).astype(np.float32)
    case["section"] = np.column_stack((feats["segment_id"], np.clip(feats["s_local_mm"], 0, None))).astype(np.float32)
    # Evaluation allocates predictions from these shapes; they are placeholders
    # only and must never be presented as model predictions or CFD labels.
    shape = (n, 3) if target == "velocity" else (n,)
    case["y_raw"], case["y_norm"] = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    if target != "velocity":
        case["query_groups"] = [np.arange(n_wall), np.arange(n_wall, n)]
    missing = [f for f in input_features if f not in case and f not in {"x", "y", "z"}]
    if missing:
        raise ValueError(f"unsupported volume features (wall curvature cannot be extended to interior): {missing}")
    for name in input_features:
        if name in case and not np.isfinite(case[name]).all():
            raise ValueError(f"non-finite volume feature: {name}")
    extra["geom"] = {key: feats[key].astype(np.float32) for key in (
        "s_from_root_mm", "s_local_mm", "theta_rad", "radius_mm", "dist_to_junction_mm")}
    extra["geom"].update(segment_id=feats["segment_id"].astype(np.int16),
                          semantic_id=feats["semantic_id"].astype(np.int16), normals=directions.astype(np.float32),
                          frame_rotation=rotation, frame_origin_mm=frame["origin_mm"],
                          point_kind=case["point_kind"], wall_mask=case["point_kind"] == 0)
    extra["internal_points"] = interior
    extra["diag"].update(n_points=n, n_wall=n_wall, n_internal=len(interior), scale_mm=scale,
                          frame_x_source=frame["x_source"], murray_shares={str(k): float(v) for k, v in shares.items()},
                          pressure_reference="predicted relative pressure; training gauge = volume mean",
                          velocity_basis="atlas aligned; world row vector = aligned @ frame_rotation",
                          wall_distance="exact original wall triangle distance; artificial caps excluded",
                          junction_ambiguous_fraction=float(np.mean(feats["junction_ambiguous"])))
    return case, extra
