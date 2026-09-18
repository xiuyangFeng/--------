import numpy as np
import pytest

from wss_deploy import metrics
from wss_deploy.report import build_html, branch_unroll_data, interpolate_to_vertices, update_html_meta


def test_interpolation_marks_points_outside_gaussian_support_as_nan():
    pts = np.array([[0., 0., 0.], [1., 0., 0.]])
    got = interpolate_to_vertices(pts, np.array([2., 4.]), np.array([[0., 0., 0.], [10., 0., 0.]]), max_dist_mm=.2)
    assert got[0] == pytest.approx(2.)
    assert np.isnan(got[1])


def test_branch_unroll_data_keeps_overlapping_s_ranges_separate():
    cloud = {'segment': np.array([1, 1, 2, 2]), 's_from_root_mm': np.array([0., 10., 0., 10.]),
             'theta_rad': np.array([-.5, .5, -.5, .5])}
    got = branch_unroll_data(cloud)
    assert [x['segment_id'] for x in got] == [1, 2]
    assert all(x['s_min_mm'] == 0 and x['s_max_mm'] == 10 for x in got)


def test_compute_guards_nonfinite_prediction_field():
    class Atlas:
        segments = [{'segment_id': 1, 'starts_at_root': True}]
    pts = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    geom = {k: np.array(v) for k, v in {'segment_id': [1]*4, 's_from_root_mm': [0, 1, 2, 3],
        'dist_to_junction_mm': [0, 1, 2, 3], 'radius_mm': [1, 1, 1, 1]}.items()}
    with pytest.raises(ValueError, match='finite'):
        metrics.compute(pts, np.array([1., 2., np.nan, 4.]), geom, Atlas(), {'spacing_mm': 1.}, 10.)


def test_compute_protocol_keeps_p99_and_max_definitions_distinct():
    class Atlas:
        segments = [{'segment_id': 1, 'starts_at_root': True, 'length_mm': 3.}]
        columns = ['segment_id', 'x_mm', 'y_mm', 'z_mm', 'radius_mm', 'curvature_per_mm']
        table = np.array([[1, 0, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0], [1, 2, 0, 0, 1, 0]], dtype=float)

    pts = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    geom = {'segment_id': np.ones(4), 's_from_root_mm': np.arange(4.), 'dist_to_junction_mm': np.arange(4.), 'radius_mm': np.ones(4)}
    out = metrics.compute(pts, np.array([1., 2., 3., 100.]), geom, Atlas(), {'spacing_mm': 1.}, 10.)
    assert out['peak']['max_pa'] == 100.
    assert out['peak']['p99_pa'] < out['peak']['max_pa']
    assert out['statistics_protocol']['support'] == 'prediction_point_cloud'
    assert out['statistics_protocol']['area_label'] == '估计面积'


def test_surface_metrics_uses_triangle_area_and_reports_coverage():
    # Two triangles with areas 1 and 3, unlike point-count weighting.
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0],
                         [0, 0, 0], [0, 3, 0], [2, 0, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    values = np.array([1., 1., 1., 9., 9., 9.])
    out = metrics.surface_metrics(vertices, faces, values, top_fraction=.25)
    assert out['protocol'] == 'gaussian_interpolated_wall_surface_area_weighted'
    assert out['total_area_mm2'] == pytest.approx(4.)
    assert out['effective_area_mm2'] == pytest.approx(4.)
    assert out['covered_area_fraction'] == pytest.approx(1.)
    assert out['p99_pa'] == pytest.approx(9.)
    assert out['top_area_mean_pa'] == pytest.approx(9.)


def test_surface_metrics_partial_coverage_is_not_extrapolated():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                         [0., 0., 0.], [0., 2., 0.], [2., 0., 0.]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    values = np.array([2., 2., 2., np.nan, 4., 4.])
    out = metrics.surface_metrics(vertices, faces, values)
    assert out['n_valid_faces'] == 1
    assert out['n_partial_faces'] == 1
    # Valid first triangle is 0.5 mm² out of total 2.5 mm².
    assert out['covered_area_fraction'] == pytest.approx(.2)
    with pytest.raises(ValueError, match='no fully covered'):
        metrics.surface_metrics(vertices, faces, np.full(6, np.nan))


def test_html_metadata_is_script_safe_and_update_preserves_markers(tmp_path):
    out = tmp_path / 'report.html'
    pts = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
    meta = {'case_id': '</script><script>alert(1)</script>', 'release': 'test', 'endpoints': [], 'branch_names': {},
            'per_branch': {}, 'geometry': {}, 'input_check': {}, 'cloud': {}, 'timing_s': {},
            'wss_field_pa': {'p99': 2., 'p99_pa': 2., 'mean': 1., 'median': 1., 'thresholds_pa': [.4, 4., 7.],
                'area_frac_low': 0., 'area_low_mm2': 0., 'area_frac_high': 0., 'area_high_mm2': 0., 'area_frac_very_high': 0.},
            'peak': {'p99_pa': 2., 'max_pa': 3., 'xyz_mm': [0, 0, 0], 'branch': 'x', 's_from_inlet_mm': 0.,
                'dist_to_junction_mm': 0., 'local_radius_mm': 1., 'top5_clusters_ge20': 0, 'top5_cluster_sizes': []}}
    mesh = {'vertices': pts, 'faces': np.array([[0, 1, 1]], dtype=np.uint32), 'wss': np.array([1., 2.], dtype=np.float32), 'segment': np.array([1, 1], dtype=np.int32)}
    cloud = {'pts': pts, 'wss': np.array([1., 2.], dtype=np.float32), 'segment': np.array([1, 1], dtype=np.int32),
             's_from_root_mm': np.array([0., 1.], dtype=np.float32), 'theta_rad': np.array([0., 1.], dtype=np.float32),
             'radius_mm': np.array([1., 1.], dtype=np.float32), 'dist_to_junction_mm': np.array([0., 1.], dtype=np.float32)}
    build_html(out, meta, mesh, cloud, {'xyz': pts, 'radius_mm': np.array([1., 1.], dtype=np.float32),
        'edges': np.array([[0, 1]], dtype=np.uint32), 'segment': np.array([1, 1], dtype=np.int32)})
    html = out.read_text()
    assert '</script><script>alert(1)</script>' not in html
    assert 'WSS_META_START' in html
    update_html_meta(out, {**meta, 'case_id': 'updated'})
    assert '"case_id": "updated"' in out.read_text()
