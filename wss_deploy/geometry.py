"""Stage 2: Taubin smoothing -> 0.5 mm resampling -> the 27 X5D_v51 inputs, from (STL, atlas) only. No CFD, no bundle.

All geometry comes from the frozen ``wss_features`` program; nothing here imports the training packages
or mutates module-level state.  The cap-area split rule is an explicit input taken from the model release.
"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import wss_features.cloud as _cloud
from wss_features.atlas import Atlas, map_points
from wss_features.cloud import build_oriented_cloud, median_spacing, patch_atlas_end_radius, pca_normals
from wss_features.curvature import COARSE_K, FINE_K, FINE_MAP, principal_curvatures, unit_rows as _unit_rows
from wss_features.flowref import Tree as _Tree, compute_point_features, load_capfit_rule
from wss_features.frame import anatomical_frame
from wss_features.sampling import sample_surface, taubin_smooth


def _caps_helper():
    """``wss_features.cloud.virtual_caps`` when it has the ``(atlas, spacing, wall_points_mm)`` call that
    ``build_oriented_cloud`` uses, else ``None`` (then :func:`cap_records` builds the full oriented cloud)."""
    fn = getattr(_cloud, "virtual_caps", None)
    try:
        params = list(inspect.signature(fn).parameters) if callable(fn) else []
    except (TypeError, ValueError):
        params = []
    return fn if params[:3] == ["atlas", "spacing", "wall_points_mm"] else None


_VIRTUAL_CAPS = _caps_helper()


def stable_sampling_seed(input_sha256: str) -> int:
    """Make repeated runs of the same STL deterministic across job directories."""
    return int.from_bytes(hashlib.sha256(str(input_sha256).encode()).digest()[:8], "little") % (2**31 - 1)


def smooth_and_resample(vertices: np.ndarray, faces: np.ndarray, *, smooth_mm: float = 1.0, spacing_mm: float = 0.5, seed: int = 20260915) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    if smooth_mm and smooth_mm > 0:
        v = taubin_smooth(v, smooth_mm)
    pts = sample_surface(v, np.asarray(faces), spacing_mm, seed)
    return v, pts


def point_geometry(pts: np.ndarray, atlas: Atlas) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Mapping-independent per-point geometry of the resampled cloud: unit PCA normals, surface variation,
    fine (k=32) and coarse (k=128) principal curvatures.

    Pure function of the points and the (end-radius-patched) atlas geometry; ``build_case`` calls it
    unchanged, and the v0.14 geometry cache stores its result under a key over exactly these inputs.
    """
    normals, variation = pca_normals(pts, atlas)
    normals = _unit_rows(np.asarray(normals, dtype=np.float64))
    tree = cKDTree(pts)
    max_k = min(COARSE_K, len(pts) - 1)
    _, nb = tree.query(pts, k=max_k + 1, workers=-1); nb = nb[:, 1:]
    fine = principal_curvatures(pts, normals, nb[:, : min(FINE_K, max_k)])
    coarse = principal_curvatures(pts, normals, nb)
    return normals, variation, fine, coarse


def cap_records(pts: np.ndarray, atlas: Atlas, normals: np.ndarray) -> tuple[list[dict], float | None]:
    """``build_oriented_cloud(pts, atlas, normals_out=normals, calibrate=False)[1]["caps"]`` without the cloud.

    With the normals given and no calibration, the records only read the float64 points, their median spacing
    and the atlas endpoints: ``virtual_caps(atlas, median_spacing(p), p)`` plus the same record fields.  The
    cloud's local areas, concatenation and KD-tree were never read by :func:`build_case` (2026-09-28, Tier A:
    bit-identical, tests/test_caps_only.py).  The second value is that median spacing when ``pts`` was already
    the float64 array (``median_spacing(pts)`` is then the same call on the same array), else ``None``.
    No cap helper, or no opening (where the full cloud raises): the full oriented cloud, unchanged.
    """
    if _VIRTUAL_CAPS is not None:
        wall = np.asarray(pts, dtype=np.float64)  # build_oriented_cloud's own cast
        spacing = median_spacing(wall)
        caps = _VIRTUAL_CAPS(atlas, spacing, wall)
        if caps:
            return ([{"label": c["label"], "center_mm": np.asarray(c["center_mm"]).tolist(), "outward": np.asarray(c["outward"]).tolist(),
                      "radius_mm": float(c["radius_mm"]), "atlas_radius_mm": float(c["atlas_radius_mm"]), "rim_snap_mm": c["rim_snap_mm"],
                      "cap_source": c["cap_source"], "rim_fit": c["rim_fit"], "points": int(len(c["points_mm"]))} for c in caps],
                    spacing if wall is pts else None)
    return build_oriented_cloud(pts, atlas, normals_out=normals, calibrate=False)[1]["caps"], None


def build_case(pts: np.ndarray, atlas: Atlas, input_features: list[str], case_name: str = "case",
               capfit_rule_path: str | Path | None = None, capfit_rule: dict | None = None,
               point_geometry_provider=None) -> tuple[dict, dict]:
    """Truth-free port of deployment_stl_simulation.build_deployment_case (same programs, same order).

    Exactly one of ``capfit_rule`` (``{"a", "b"}``) or ``capfit_rule_path`` must be given; the rule
    ships inside the model release and is never read from a training-side default location.

    ``point_geometry_provider(pts, atlas)`` (v0.14, optional) returns the :func:`point_geometry`
    tuple — from the job's geometry cache when its key matches — and is called at the point where
    that geometry is needed (after the end-radius patch).  ``None`` computes it here.
    """
    if capfit_rule is None:
        if capfit_rule_path is None:
            raise ValueError("build_case requires the release cap-area split rule (capfit_rule or capfit_rule_path)")
        capfit_rule = load_capfit_rule(Path(capfit_rule_path))
    end_patch = patch_atlas_end_radius(atlas, pts)  # 2026-09-23: escaped-sphere endpoint radius -> held at the measured opening (same rule as the V5 build)
    table, columns, segments = atlas.table, list(atlas.columns), atlas.segments
    feats = map_points(pts, atlas)
    frame = anatomical_frame(table, columns, {str(k): v for k, v in atlas.semantic_of_segment.items()})
    aligned = (pts - frame["origin_mm"]) @ frame["rotation"].T
    scale = float(np.abs(aligned).max()) or 1.0
    # Normals, variation and curvatures only read the points and the patched atlas geometry (pure).
    normals, variation, fine, coarse = (point_geometry if point_geometry_provider is None
                                        else point_geometry_provider)(pts, atlas)
    caps, spacing = cap_records(pts, atlas, normals)  # only the oriented cloud's caps were ever read here
    cap_labels = [c["label"] for c in caps]
    cap_radius = np.array([c["radius_mm"] for c in caps], dtype=np.float64)
    atlas_row = feats["atlas_row"].astype(np.int64)
    tangent = atlas.tangent[atlas_row]
    ftree = _Tree(table, columns, segments, atlas.frame_n, atlas.frame_b)
    radius_pt = np.clip(feats["radius_mm"].astype(np.float64), 1e-3, None)
    flow, extra = compute_point_features(ftree, cap_labels, cap_radius, atlas_row, feats["segment_id"], feats["semantic_id"],
                                         feats["s_local_mm"], feats["s_from_root_mm"], feats["theta_rad"], radius_pt, feats["dist_to_junction_mm"],
                                         capfit_rule=capfit_rule)
    s_max = float(np.max(feats["s_from_root_mm"])); n = len(pts)
    case = dict(
        cohort="deploy", case=case_name, unit_id=f"deploy/{case_name}",
        pos=(aligned / scale).astype(np.float32), wall_coords_raw=pts.astype(np.float64),
        y_raw=np.zeros(n, np.float32), y_norm=np.zeros(n, np.float32), target_normalization="global_stats",
        abscissa_norm=(feats["s_from_root_mm"] / s_max).astype(np.float32), local_radius=radius_pt.astype(np.float32),
        log_local_radius=np.log(radius_pt).astype(np.float32), curvature=feats["curvature_per_mm"].astype(np.float32),
        coord_scale=np.full(n, scale, dtype=np.float32), coord_scale_scalar=scale, radius_gradient=feats["dr_ds"].astype(np.float32),
        rho=feats["rho"].astype(np.float32), theta_sin=np.sin(feats["theta_rad"]).astype(np.float32), theta_cos=np.cos(feats["theta_rad"]).astype(np.float32),
        dr_ds=feats["dr_ds"].astype(np.float32), dist_to_junction_mm=feats["dist_to_junction_mm"].astype(np.float32),
        dist_to_endpoint_mm=feats["dist_to_endpoint_mm"].astype(np.float32), end_zone=feats["end_zone"].astype(np.float32),
        _wall_segment_id=feats["segment_id"].astype(np.int64), peak_step=1162, bundle_path="",
    )
    aligned_normals = normals @ frame["rotation"].T
    case["nx_aligned"], case["ny_aligned"], case["nz_aligned"] = (aligned_normals[:, i].astype(np.float32) for i in range(3))
    for name in input_features:
        if name in FINE_MAP:
            case[name] = fine[FINE_MAP[name]].astype(np.float32)
        elif name.endswith("_c") and name[:-2] in FINE_MAP:
            case[name] = coarse[FINE_MAP[name[:-2]]].astype(np.float32)
        elif name == "tn_dot":
            case[name] = np.einsum("nd,nd->n", tangent, normals).astype(np.float32)
        elif f"wall_{name}" in flow:
            case[name] = np.asarray(flow[f"wall_{name}"], dtype=np.float32)
    missing = [f for f in input_features if f not in case and f not in ("x", "y", "z")]
    if missing:
        raise KeyError(f"could not build deployment features {missing}")
    tangent_aligned = _unit_rows(tangent @ frame["rotation"].T)
    case["local_geometry"] = np.column_stack((aligned_normals, tangent_aligned, radius_pt, np.full(n, scale))).astype(np.float32)
    case["section"] = np.column_stack((feats["segment_id"].astype(np.float64), np.clip(feats["s_local_mm"], 0.0, None))).astype(np.float32)
    diag = {"n_points": n, "spacing_mm": float(median_spacing(pts) if spacing is None else spacing), "scale_mm": scale, "frame_x_source": frame["x_source"],
            "junction_ambiguous_fraction": float(np.mean(feats["junction_ambiguous"])),
            "caps": {c["label"]: {"radius_mm": float(c["radius_mm"]), "atlas_radius_mm": float(c["atlas_radius_mm"])} for c in caps},
            "murray_shares": {str(k): float(v) for k, v in extra["murray_shares"].items()},
            "capfit_shares": {str(k): float(v) for k, v in extra["capfit_shares"].items()},
            "fine_neighbourhood_mm_median": float(np.median(fine["_neighbourhood_mm"])), "coarse_neighbourhood_mm_median": float(np.median(coarse["_neighbourhood_mm"])),
            "surface_variation_median": float(np.median(variation))}
    geom = {"segment_id": feats["segment_id"].astype(np.int16), "s_from_root_mm": feats["s_from_root_mm"].astype(np.float32), "s_local_mm": feats["s_local_mm"].astype(np.float32),
            "theta_rad": feats["theta_rad"].astype(np.float32), "radius_mm": feats["radius_mm"].astype(np.float32), "dist_to_junction_mm": feats["dist_to_junction_mm"].astype(np.float32),
            "normals": normals.astype(np.float32), "frame_rotation": frame["rotation"], "frame_origin_mm": frame["origin_mm"],
            # Per-point PCA surface variation (smallest eigenvalue share); the report's "rough surface" trust bit uses it.
            "surface_variation": np.asarray(variation, dtype=np.float32)}
    diag["atlas_end_radius_patch"] = end_patch
    return case, {"diag": diag, "geom": geom}
