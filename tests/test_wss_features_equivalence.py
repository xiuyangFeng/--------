"""Bit-for-bit equivalence of the frozen ``wss_features`` program with the training implementations.

Runs only when the training packages are importable and a finished deployment job (with its
centreline and clean STL) is available; set ``WSS_DEPLOY_EQUIV_JOB`` to choose the job directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _job_dir() -> Path | None:
    configured = os.environ.get("WSS_DEPLOY_EQUIV_JOB")
    if configured:
        path = Path(configured)
        return path if (path / "centerline" / "atlas.npz").is_file() else None
    root = ROOT / "outputs" / "wss_deploy_jobs"
    candidates = []
    for job in sorted(root.glob("*/job.json"), reverse=True):
        record = json.loads(job.read_text(encoding="utf-8"))
        if record.get("status") == "done" and (job.parent / "centerline" / "atlas.npz").is_file() \
                and (job.parent / "input_clean_mm.stl").is_file():
            candidates.append(job.parent)
    return candidates[0] if candidates else None


training = pytest.importorskip("wss_v5.centerline_features", reason="training packages not importable")
JOB = _job_dir()
pytestmark = pytest.mark.skipif(JOB is None, reason="no finished deployment job with a centreline available")


def _same(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    if a.dtype.kind in "fc":
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: NaN pattern differs"
        assert np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]), f"{name}: values differ (max |d| = {np.nanmax(np.abs(a - b))})"
    else:
        assert np.array_equal(a, b), f"{name}: values differ"


@pytest.fixture(scope="module")
def data():
    import wss_v5.centerline_features as ref_atlas
    from training_wss_min.surface import load_stl as ref_load_stl
    from training_wss_min.tools.deployment_geometry_sensitivity import taubin_smooth as ref_taubin
    from training_wss_min.tools.deployment_stl_simulation import sample_surface as ref_sample
    from wss_features import atlas as new_atlas, sampling, stl
    cl = JOB / "centerline"
    ref = ref_atlas.load_atlas(cl / "atlas.npz", cl / "atlas_summary.json")
    new = new_atlas.load_atlas(cl / "atlas.npz", cl / "atlas_summary.json")
    v_ref, f_ref = ref_load_stl(JOB / "input_clean_mm.stl")
    v_new, f_new = stl.load_stl(JOB / "input_clean_mm.stl")
    _same(v_ref, v_new, "load_stl vertices"); _same(f_ref, f_new, "load_stl faces")
    smooth_ref = ref_taubin(v_ref, 1.0)
    smooth_new = sampling.taubin_smooth(v_new, 1.0)
    _same(smooth_ref, smooth_new, "taubin_smooth")
    pts_ref = ref_sample(smooth_ref, f_ref, 0.5, 12345)
    pts_new = sampling.sample_surface(smooth_new, f_new, 0.5, 12345)
    _same(pts_ref, pts_new, "sample_surface")
    return {"ref_atlas": ref, "new_atlas": new, "pts": pts_new, "vertices": v_new, "faces": f_new}


def test_atlas_and_frames(data):
    ref, new = data["ref_atlas"], data["new_atlas"]
    _same(ref.table, new.table, "atlas table")
    assert ref.columns == new.columns
    _same(ref.frame_n, new.frame_n, "frame_n"); _same(ref.frame_b, new.frame_b, "frame_b")
    _same(ref.tree_rows, new.tree_rows, "tree_rows")
    assert ref.semantic_of_segment == new.semantic_of_segment
    assert [e["label"] for e in ref.endpoints()] == [e["label"] for e in new.endpoints()]


def test_map_points_and_anatomical_frame(data):
    import wss_v5.centerline_features as ref_atlas
    from wss_v5.views.wss_min_view import anatomical_frame as ref_frame
    from wss_features import atlas as new_atlas
    from wss_features.frame import anatomical_frame as new_frame
    a, b = ref_atlas.map_points(data["pts"], data["ref_atlas"]), new_atlas.map_points(data["pts"], data["new_atlas"])
    assert set(a) == set(b)
    for key in a:
        _same(a[key], b[key], f"map_points[{key}]")
    sem = {str(k): v for k, v in data["new_atlas"].semantic_of_segment.items()}
    fr, fn = ref_frame(data["ref_atlas"].table, list(data["ref_atlas"].columns), sem), new_frame(data["new_atlas"].table, list(data["new_atlas"].columns), sem)
    _same(fr["origin_mm"], fn["origin_mm"], "frame origin"); _same(fr["rotation"], fn["rotation"], "frame rotation")
    assert fr["x_source"] == fn["x_source"]


def test_normals_caps_and_curvatures(data):
    from scipy.spatial import cKDTree
    from wss_v5.pointcloud import build_oriented_cloud as ref_cloud, pca_normals as ref_normals
    from wss_v5.views.wall_geom_v2 import COARSE_K, FINE_K, _unit_rows, principal_curvatures as ref_curv
    from wss_features.cloud import build_oriented_cloud as new_cloud, pca_normals as new_normals
    from wss_features.curvature import principal_curvatures as new_curv, unit_rows
    pts = data["pts"]
    n_ref, var_ref = ref_normals(pts, data["ref_atlas"])
    n_new, var_new = new_normals(pts, data["new_atlas"])
    _same(n_ref, n_new, "pca_normals"); _same(var_ref, var_new, "surface variation")
    assert ref_normals.last_info == new_normals.last_info
    normals = unit_rows(np.asarray(n_new, dtype=np.float64))
    _same(_unit_rows(np.asarray(n_ref, dtype=np.float64)), normals, "unit_rows")
    _, info_ref = ref_cloud(pts, data["ref_atlas"], normals_out=normals, calibrate=False)
    _, info_new = new_cloud(pts, data["new_atlas"], normals_out=normals, calibrate=False)
    assert json.dumps(info_ref, sort_keys=True, default=str) == json.dumps(info_new, sort_keys=True, default=str)
    max_k = min(COARSE_K, len(pts) - 1)
    _, nb = cKDTree(pts).query(pts, k=max_k + 1, workers=-1)
    nb = nb[:, 1:]
    for k in (min(FINE_K, max_k), max_k):
        cr, cn = ref_curv(pts, normals, nb[:, :k]), new_curv(pts, normals, nb[:, :k])
        assert set(cr) == set(cn)
        for key in cr:
            if key == "_clipped_fraction":
                assert cr[key] == cn[key]
            else:
                _same(cr[key], cn[key], f"curvature[{key}] k={k}")


def test_flowref_features_and_shares(data):
    import wss_v5.centerline_features as ref_atlas
    from wss_v5.pointcloud import build_oriented_cloud as ref_cloud, pca_normals as ref_normals
    from wss_v5.views import wall_flowref_v1 as WF
    from wss_features.cloud import build_oriented_cloud as new_cloud
    from wss_features.flowref import Tree, compute_point_features, load_capfit_rule, murray_shares
    from wss_deploy.paths import RELEASE_DIR
    rule_path = RELEASE_DIR / "rules" / "flow_split_rule_train136.json"
    if not rule_path.is_file():
        pytest.skip("release cap-fit rule not available")
    pts = data["pts"]
    atlas_r, atlas_n = data["ref_atlas"], data["new_atlas"]
    normals, _ = ref_normals(pts, atlas_r)
    _, info_r = ref_cloud(pts, atlas_r, normals_out=normals, calibrate=False)
    _, info_n = new_cloud(pts, atlas_n, normals_out=normals, calibrate=False)
    cap_labels = [c["label"] for c in info_r["caps"]]
    cap_radius = np.array([c["radius_mm"] for c in info_r["caps"]], dtype=np.float64)
    feats = ref_atlas.map_points(pts, atlas_r)
    radius_pt = np.clip(feats["radius_mm"].astype(np.float64), 1e-3, None)
    args = (feats["atlas_row"].astype(np.int64), feats["segment_id"], feats["semantic_id"], feats["s_local_mm"],
            feats["s_from_root_mm"], feats["theta_rad"], radius_pt, feats["dist_to_junction_mm"])
    tree_r = WF._Tree(atlas_r.table, list(atlas_r.columns), atlas_r.segments, atlas_r.frame_n, atlas_r.frame_b)
    tree_n = Tree(atlas_n.table, list(atlas_n.columns), atlas_n.segments, atlas_n.frame_n, atlas_n.frame_b)
    _same(tree_r.psi_bend, tree_n.psi_bend, "psi_bend"); _same(tree_r.torsion, tree_n.torsion, "torsion")
    assert WF._murray_shares(tree_r) == murray_shares(tree_n)
    previous = WF._CAPFIT_RULE_OVERRIDE
    WF._CAPFIT_RULE_OVERRIDE = rule_path
    try:
        payload_r, extra_r = WF.compute_point_features(tree_r, cap_labels, cap_radius, *args)
    finally:
        WF._CAPFIT_RULE_OVERRIDE = previous
    payload_n, extra_n = compute_point_features(tree_n, cap_labels, cap_radius, *args, capfit_rule=load_capfit_rule(rule_path))
    assert set(payload_r) == set(payload_n)
    for key in payload_r:
        _same(payload_r[key], payload_n[key], f"flowref[{key}]")
    assert json.dumps(extra_r, sort_keys=True, default=str) == json.dumps(extra_n, sort_keys=True, default=str)


def test_internal_queries(data):
    from wss_v5.pointcloud import build_oriented_cloud as ref_cloud, generate_internal_queries as ref_queries, pca_normals as ref_normals
    from wss_features.cloud import build_oriented_cloud as new_cloud, generate_internal_queries as new_queries
    pts = data["pts"]
    normals, _ = ref_normals(pts, data["ref_atlas"])
    cloud_r, _ = ref_cloud(pts, data["ref_atlas"], normals_out=normals)
    cloud_n, _ = new_cloud(pts, data["new_atlas"], normals_out=normals)
    _same(cloud_r.areas_mm2, cloud_n.areas_mm2, "calibrated areas")
    q_r, d_r = ref_queries(data["ref_atlas"], cloud_r, n_target=4000, seed=7)
    q_n, d_n = new_queries(data["new_atlas"], cloud_n, n_target=4000, seed=7)
    _same(q_r, q_n, "internal queries")
    assert d_r == d_n
