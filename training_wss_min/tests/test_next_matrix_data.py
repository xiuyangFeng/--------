import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from training_wss_min.dataset import (
    compute_feature_stats,
    build_features,
    geom_stratified_sample,
    sample_indices,
)
from training_wss_min.config import DataConfig


def _case(n=120):
    x = np.linspace(-1, 1, n, dtype=np.float32)
    return {
        "pos": np.c_[x, np.zeros_like(x), np.zeros_like(x)],
        "abscissa_norm": (x + 1) / 2,
        "local_radius": np.linspace(2, 8, n, dtype=np.float32),
        "curvature": np.sin(np.arange(n)),
        "coord_scale_scalar": 100.0,
    }


def test_geom_stratified_is_deterministic_and_covers_extremes():
    c = _case()
    a = geom_stratified_sample(c, 24, 7, (4, 2, 2))
    b = geom_stratified_sample(c, 24, 7, (4, 2, 2))
    assert np.array_equal(a, b)
    assert len(np.unique(a)) == 24
    assert c["abscissa_norm"][a].min() < 0.2
    assert c["abscissa_norm"][a].max() > 0.8


def test_sample_indices_geom_stratified_default_compatibility():
    c = _case()
    cfg = DataConfig(wall_n_points=20, sampling="geom_stratified", stratified_bins=(2, 2, 2))
    out = sample_indices(c, cfg, seed=3)
    assert out.shape == (20,)
    assert len(np.unique(out)) == 20


def test_fixed_mm_and_nondimensional_channels_are_train_standardized():
    c = _case()
    names = ("radius_over_mm_5", "radius_over_mm_10", "radius_over_coord_scale", "log_radius_over_coord_scale")
    stats = compute_feature_stats([c], names)
    feat = build_features(c, np.arange(len(c["pos"])), names, stats)
    assert feat.shape == (len(c["pos"]), 4)
    assert np.all(np.isfinite(feat))
    assert np.allclose(feat.mean(0), 0.0, atol=1e-5)



def test_spatial_sampler_keeps_sparse_region_and_changes_with_seed():
    # Source rows are sorted and density is strongly unequal across two regions.
    rng = np.random.default_rng(1)
    dense = rng.uniform([0, 0, 0], [1, 1, 0], size=(4000, 3))
    sparse = rng.uniform([5, 0, 0], [6, 1, 0], size=(200, 3))
    case = _case(4200)
    case["pos"] = np.vstack((dense, sparse)).astype(np.float32)
    case["dist_to_junction_mm"] = np.linalg.norm(case["pos"] - [5.5, .5, 0], axis=1)
    a = geom_stratified_sample(case, 200, 71)
    b = geom_stratified_sample(case, 200, 72)
    assert len(np.unique(a)) == 200
    assert np.count_nonzero(a >= 4000) >= 45  # far above uniform-by-vertex expectation (~10)
    assert not np.array_equal(a, b)
    assert np.any(a < 1000) and np.any((a > 3000) & (a < 4000))


def test_full_wall_patch_is_chunk_invariant_and_ignores_labels():
    from training_wss_min.dataset import build_query_patch
    case = _case(120)
    indices = np.array([60, 1, 110, 27])
    names = ("x", "y", "z", "local_radius")
    stats = compute_feature_stats([case], names)
    whole = build_query_patch(case, indices, names, stats, nsample=16)
    parts = [build_query_patch(case, ids, names, stats, nsample=16) for ids in np.array_split(indices, 2)]
    for key in whole:
        np.testing.assert_array_equal(whole[key], np.concatenate([p[key] for p in parts]))
    assert whole["features"].shape == (4, 16, 4)
    assert whole["relative_mm"].shape == (4, 16, 3)
    np.testing.assert_allclose(whole["relative_mm"][:, 0], 0, atol=1e-6)
    # 1-hop complete-wall spacing is 200/119 mm, regardless of sparse support.
    assert np.isclose(np.linalg.norm(whole["relative_mm"][0, 1]), 200 / 119, atol=1e-4)
    case["y_raw"] = np.full(120, np.nan)
    case["y_norm"] = np.full(120, -1e10)
    again = build_query_patch(case, indices, names, stats, nsample=16)
    np.testing.assert_array_equal(whole["features"], again["features"])


def test_geometry_uses_frozen_node_identity_and_aligned_tangent(tmp_path):
    from training_wss_min.next_geometry import case_geometry
    root = tmp_path / "view"
    path = root / "AG" / "fast" / "case" / "bundle.npz"
    path.parent.mkdir(parents=True)
    source = tmp_path / "frozen.h5"
    rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    raw = np.array([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    pos = (raw @ rot.T).astype(np.float32) / 10
    # Frozen source has an extra dropped row and a different row order.
    with h5py.File(source, "w") as h:
        h.create_dataset("wall_static/node_id_cas", data=[30, 99, 10, 20])
        h.create_dataset("wall_static/xyz_mm", data=[raw[2], [99, 0, 0], raw[0], raw[1]])
        h.create_dataset("wall_static/atlas_row", data=[2, 0, 0, 1])
        h.create_dataset("geometry/atlas_table", data=np.eye(3))
        h["geometry"].attrs["atlas_columns"] = json.dumps(["tangent_x", "tangent_y", "tangent_z"])
    (root / "view_manifest.json").write_text(json.dumps({"reports": [{"canonical_id": "AG/fast/case", "source": {"case_h5": str(source)}}]}))
    np.savez(path, wall_node_id_cas=[10, 20, 30], wall_coords_raw=raw,
             wall_coords_norm=pos, transform_rotation=rot,
             wall_normal_pca_aligned=np.tile([0, 0, 1.], (3, 1)),
             wall_local_radius=[2., 3., 4.], coord_scale=10.)
    case = {"bundle_path": str(path), "pos": pos}
    geometry = case_geometry(case)
    assert geometry.shape == (3, 8)
    np.testing.assert_allclose(geometry[:, 3:6], np.eye(3) @ rot.T)
    np.testing.assert_array_equal(geometry[:, 6], [2, 3, 4])
    np.testing.assert_array_equal(geometry[:, 7], 10)


def test_dataset_collates_optional_geometry_and_patch():
    from training_wss_min.dataset import WSSMinDataset, collate, local_model_context
    case = _case(120)
    case.update(case="synthetic", cohort="AG/fast", unit_id="AG/fast/synthetic",
                y_norm=np.zeros(120, dtype=np.float32), y_raw=np.ones(120, dtype=np.float32))
    case["local_geometry"] = np.tile(np.array([0, 0, 1, 1, 0, 0, 3, 100], dtype=np.float32), (120, 1))
    cfg = DataConfig(wall_n_points=20, support_n_points=20, query_n_points=10,
                     query_mode="independent", input_features=("x", "y", "z"))
    cfg.local_geometry = True
    cfg.query_patch_nsample = 16
    dataset = WSSMinDataset([case], cfg, {}, training=False)
    item = dataset[0]
    batch = collate([item, item])
    assert batch["support_geometry"].shape == (40, 8)
    assert batch["query_geometry"].shape == (20, 8)
    assert batch["query_patch"]["features"].shape == (20, 16, 3)
    context = local_model_context(batch, "cpu")
    assert set(context) == {"support_geometry", "query_geometry", "query_patch"}
    assert local_model_context({}, "cpu") == {}


def test_inlet_reference_uses_true_area_and_rejects_inconsistent_source():
    from training_wss_min.tools.audit_next_matrix_geometry import inlet_feature_from_arrays
    vectors = np.array([[0, 0, 2e-4], [0, 0, 3e-4]])
    output = inlet_feature_from_arrays(vectors, 1e-4, 5e-4)
    assert output["inlet_velocity_nominal_m_s"] == pytest.approx(.2)
    with pytest.raises(ValueError, match="area disagrees"):
        inlet_feature_from_arrays(vectors, 1e-4, 6e-4)
    with pytest.raises(ValueError, match="orientation"):
        inlet_feature_from_arrays(vectors * [[1, 1, 1], [1, 1, -1]], 1e-4, 5e-4)
