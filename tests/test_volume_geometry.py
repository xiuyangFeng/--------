"""Deployment volume geometry tests use synthetic shapes, never CFD fields."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from wss_deploy import volume_geometry as VG
from wss_v5.centerline_features import Atlas, map_points


def _extruded_polygon(polygon, length=10.0):
    polygon = np.asarray(polygon, dtype=float)
    count = len(polygon)
    vertices = np.concatenate([np.column_stack([polygon, np.zeros(count)]),
                               np.column_stack([polygon, np.full(count, length)])])
    faces = []
    for i in range(count):
        j = (i + 1) % count
        faces.extend([[i, j, j + count], [i, j + count, i + count]])
    return vertices, np.asarray(faces)


@pytest.fixture
def tube():
    count = 32
    columns = ["segment_id", "sample_index", "x_mm", "y_mm", "z_mm", "tangent_x", "tangent_y", "tangent_z",
               "radius_mm", "s_local_mm", "s_from_root_mm", "curvature_per_mm", "dr_ds", "dist_to_junction_mm",
               "dist_to_endpoint_mm", "junction_mask", "endpoint_mask", "end_zone"]
    table = np.zeros((count, len(columns)))
    def put(name, value):
        table[:, columns.index(name)] = value
    z = np.linspace(10., 0., count)
    put("sample_index", np.arange(count)); put("z_mm", z); put("tangent_z", -1.)
    put("radius_mm", 2.); put("s_local_mm", 10 - z); put("s_from_root_mm", 10 - z)
    put("dist_to_junction_mm", 10.); put("dist_to_endpoint_mm", np.minimum(z, 10 - z))
    put("endpoint_mask", (z == 0) | (z == 10)); put("end_zone", (z < 1) | (z > 9))
    xyz = np.column_stack([np.zeros(count), np.zeros(count), z])
    atlas = Atlas(table, columns, [{"segment_id": 0, "parent_id": -1, "s_offset_mm": 0., "starts_at_root": True,
                                    "ends_at_leaf": True, "outlet_name": "out-le"}],
                  {0: 0}, np.tile([1., 0., 0.], (count, 1)), np.tile([0., -1., 0.], (count, 1)),
                  np.arange(count), cKDTree(xyz), {})
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    polygon = 2 * np.column_stack([np.cos(angles), np.sin(angles)])
    vertices, faces = _extruded_polygon(polygon)
    # Uniform wall points on triangle faces (not synthetic end cap points).
    wall = np.concatenate([np.column_stack([polygon, np.full(len(polygon), height)])
                           for height in np.linspace(.2, 9.8, 16)])
    return wall, vertices, faces, atlas


def test_concave_lumen_caps_do_not_fill_exterior_notch():
    vertices, faces = _extruded_polygon([[0., 0.], [3., 0.], [3., 1.], [1., 1.], [1., 3.], [0., 3.]])
    closed, audit = VG.close_lumen(vertices, faces)
    inside = VG.make_inside_test(closed.points, closed.faces.reshape(-1, 4)[:, 1:])
    assert audit["cap_count"] == 2
    assert inside(np.array([[.5, .5, 5.], [2., .5, 5.], [.5, 2., 5.], [2., 2., 5.], [.5, .5, 11.]])).tolist() == [True, True, True, False, False]
    assert inside(np.empty((0, 3))).shape == (0,)


def test_unclosed_and_nonmanifold_meshes_fail():
    vertices, faces = _extruded_polygon([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    with pytest.raises(ValueError, match="closed"):
        VG.make_inside_test(vertices, faces)
    with pytest.raises(ValueError, match="非流形"):
        VG.close_lumen(vertices, np.concatenate([faces, faces[:1]]))


def test_unknown_opening_is_not_silently_capped(tube):
    _, vertices, faces, atlas = tube
    # Two valid tube openings plus an unrelated open fragment cannot be
    # interpreted as three anatomical endpoints.
    count = len(vertices)
    vertices = np.concatenate([vertices, [[10., 0., 5.], [11., 0., 5.], [10., 1., 5.]]])
    faces = np.concatenate([faces, [[count, count + 1, count + 2]]])
    with pytest.raises(ValueError, match="开口数"):
        VG.close_lumen(vertices, faces, atlas)


def test_volume_features_match_wall_and_interior_training_contract(tube):
    from training_wss_min.dataset import build_features

    wall, vertices, faces, atlas = tube
    names = ["x", "y", "z", "abscissa_norm", "local_radius", "curvature", "log_local_radius", "rho",
             "theta_sin", "theta_cos", "dr_ds", "dist_to_junction_mm", "dist_to_endpoint_mm", "end_zone",
             "nx_aligned", "ny_aligned", "nz_aligned", "dist_to_wall_mm", "log_q_branch_murray", "log_tau0_murray"]
    case, extra = VG.build_volume_case(wall, vertices, faces, atlas, names, n_internal=150, seed=41)
    n_wall = len(wall)
    assert case["pos"].shape == (n_wall + 150, 3)
    assert np.array_equal(case["support_pool"], np.arange(n_wall))
    assert np.array_equal(case["query_pool"], np.arange(n_wall + 150))
    assert np.array_equal(case["query_groups"][1], np.arange(n_wall, n_wall + 150))
    assert np.array_equal(case["point_kind"] == 0, extra["geom"]["wall_mask"])
    assert case["bundle_path"] == ""
    assert "p_ref_pa" not in case  # An unknown physical pressure reference cannot be fabricated.
    pts = case["wall_coords_raw"]
    transformed = (pts - case["frame_origin_mm"]) @ case["frame_rotation"].T
    np.testing.assert_allclose(case["pos"] * case["coord_scale_scalar"], transformed, atol=1e-6)
    expected_scale = np.abs((wall - case["frame_origin_mm"]) @ case["frame_rotation"].T).max()
    assert case["coord_scale_scalar"] == pytest.approx(expected_scale)
    f = map_points(pts, atlas)
    np.testing.assert_allclose(case["abscissa_norm"], f["s_from_root_mm"] / f["s_from_root_mm"][:n_wall].max())
    np.testing.assert_allclose(case["log_tau0_murray"], -3 * np.log(case["local_radius"]), atol=1e-6)
    radial = pts[n_wall:].copy(); radial[:, 2] = 0
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    got = np.column_stack([case[k][n_wall:] for k in ("nx_aligned", "ny_aligned", "nz_aligned")])
    np.testing.assert_allclose(got, radial @ case["frame_rotation"].T, atol=1e-6)
    assert np.all(case["dist_to_wall_mm"][:n_wall] == 0)
    assert np.all(case["dist_to_wall_mm"][n_wall:] > 0)
    # Exact wall-triangle distance differs from the nearest sparse wall point
    # and is unaffected by distance to inlet/outlet artificial caps.
    normals = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 40, endpoint=False) + np.pi/40),
                               np.sin(np.linspace(0, 2*np.pi, 40, endpoint=False) + np.pi/40)])
    expected_distance = 2 * np.cos(np.pi / 40) - np.max(pts[n_wall:, :2] @ normals.T, axis=1)
    np.testing.assert_allclose(case["dist_to_wall_mm"][n_wall:], expected_distance, atol=1e-6)
    feature_stats = {name: {"mean": 0., "std": 1.} for name in names}
    features = build_features(case, np.arange(len(pts)), tuple(names), feature_stats)
    assert features.shape == (len(pts), 20)
    assert np.isfinite(features).all()
    inside = VG.make_inside_test(**extra["closed_surface"])
    assert inside(pts[n_wall:]).all()
    # Restore predicted aligned velocity to world coordinates with row R.
    world_v = np.array([[.3, -.6, 1.1]])
    np.testing.assert_allclose((world_v @ case["frame_rotation"].T) @ case["frame_rotation"], world_v)


def test_velocity_queries_are_interior_and_sampling_is_reproducible(tube):
    wall, vertices, faces, atlas = tube
    a, ae = VG.build_volume_case(wall, vertices, faces, atlas, ["x", "dist_to_wall_mm"],
                                 target="velocity", n_internal=100, seed=11)
    b, _ = VG.build_volume_case(wall, vertices, faces, atlas, ["x"], target="pressure_mixed", n_internal=100, seed=11)
    np.testing.assert_array_equal(a["wall_coords_raw"], b["wall_coords_raw"])
    assert np.all(a["query_pool"] >= len(wall))
    assert "query_groups" not in a
    assert a["y_raw"].shape == (len(wall) + 100, 3)
    assert ae["diag"]["internal_sampling"]["containment"] == "closed_triangle_ray_test"


def test_reject_surface_only_features_for_internal_queries(tube):
    with pytest.raises(ValueError, match="wall curvature"):
        VG.build_volume_case(*tube, ["curv_gauss"], n_internal=8, seed=5)


def test_exact_uniform_queries_are_kept_even_when_sweep_is_already_full(tube, monkeypatch):
    # Model a pathological hybrid test that only accepts a narrow centerline
    # region.  Supplying many sweep candidates must not disable independent
    # exact-only exploration of the rest of the closed lumen.
    def sweep(*_args, **_kwargs):
        return np.tile([0., .01, 5.], (1000, 1)), {"kept": 1000}
    monkeypatch.setattr(VG, "generate_internal_queries", sweep)
    points, extra = VG.sample_internal_points(*tube, n_internal=100, seed=17)
    diagnostic = extra["diag"]["internal_sampling"]
    assert diagnostic["sweep_selected"] == 70
    assert diagnostic["exact_uniform_selected"] == 30
    assert diagnostic["exact_uniform_fraction"] >= .30
    assert diagnostic["uniform_candidates"] > 0
    assert np.count_nonzero(np.linalg.norm(points[:, :2], axis=1) > .5) >= 20
    assert VG.make_inside_test(**extra["closed_surface"])(points).all()
    assert extra["diag"]["geometry_version"] == "stl-volume-queries/v3"


def test_radius_weighted_branch_quota_gives_narrow_branch_a_floor():
    # A broad parent would dominate radius-squared sampling.  The adaptive
    # allocator keeps a deterministic floor for the narrow branch and boosts
    # it with the inverse-sqrt radius term before the floor is applied.
    profiles = {
        0: {"length_mm": 100.0, "radius_mm": 3.0, "weight": 100.0 / np.sqrt(3.0)},
        1: {"length_mm": 12.0, "radius_mm": 0.45, "weight": 12.0 / np.sqrt(0.45)},
    }
    quotas, floor = VG._radius_weighted_branch_quotas(1400, profiles)
    assert floor == 16
    assert quotas[1] >= floor
    assert quotas[1] > 1400 * (12.0 * 0.45 ** 2) / (100.0 * 3.0 ** 2 + 12.0 * 0.45 ** 2)
    assert sum(quotas.values()) == 1400


def test_branch_sampling_diag_records_adaptive_and_uniform_components(tube):
    wall, vertices, faces, atlas = tube
    _, extra = VG.sample_internal_points(wall, vertices, faces, atlas, n_internal=100, seed=23)
    sampling = extra["diag"]["internal_sampling"]
    strat = sampling["branch_stratification"]
    assert sampling["adaptive_target"] == 70
    assert sampling["adaptive_selected"] == 70
    assert sampling["exact_uniform_fraction"] >= 0.30
    assert strat["radius_weighting"] == "length_mm / sqrt(median_radius_mm)"
    assert strat["min_satisfied"] is True
