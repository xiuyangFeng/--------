"""Analytical geometry checks; no patient processing or flow labels."""
import numpy as np

from wss_v5.section_features import (SurfaceSlicer, log_area_slopes, pointcloud_section,
                                    polygon_metrics, polyline_plane_intersection_groups, project_wall, same_segment_nonlocal_crossings,
                                    _boundary_cycles, _verified_fan_center, opening_overrun_domain,
                                    _remove_opposite_triangle_fins)
from wss_v5.geometry_candidate import branch_context, interpolate_valid_field, section_geometry_gate, topology_semantics


def tube(radius=2., length=20., n=120):
    angle = np.arange(n) * 2 * np.pi / n
    ring = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros(n)))
    xyz = np.r_[ring, ring + [0, 0, length]]
    tri = []
    for i in range(n):
        j = (i + 1) % n
        tri.extend([[i, j, n + j], [i, n + j, n + i]])
    return xyz, np.asarray(tri)


def test_reference_tube_area_and_opening_loops():
    xyz, triangles = tube()
    slicer = SurfaceSlicer(xyz, triangles)
    section = slicer.slice(np.array([0., 0., 10.]), np.array([0., 0., 1.]))
    assert section['valid'], section['reason']
    assert abs(section['area_mm2'] / (4 * np.pi) - 1) < .001
    assert len(slicer.boundary_loops()) == 2
    assert not slicer.slice(np.array([0., 0., 21.]), np.array([0., 0., 1.]))['valid']
    enclosed, info = slicer.enclosed(np.array([[0., 0., 10.], [3., 0., 10.]]))
    assert info['closed_reference_valid']
    assert enclosed.tolist() == [True, False]


def test_nonplanar_opening_cap_repair_preserves_wall_and_outside_points():
    # A deterministic irregular rim reproduces vtkFillHoles' incomplete cap.
    rng = np.random.default_rng(0)
    n = 52
    angle = np.sort(rng.random(n) * 2 * np.pi)
    radius = 2 + rng.normal(0, .15, n)
    ring = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), rng.normal(0, .02, n)))
    xyz = np.r_[ring, ring + [0, 0, 20]]
    triangles = np.asarray([[i, (i + 1) % n, n + (i + 1) % n] for i in range(n)] +
                           [[i, n + (i + 1) % n, n + i] for i in range(n)])
    slicer = SurfaceSlicer(xyz, triangles)
    inside, audit = slicer.enclosed(np.array([[0., 0., 10.], [0., 0., -1.], [3., 0., 10.], [0., 0., 21.]]))
    assert inside.tolist() == [True, False, False, False]
    assert audit['closed_reference_valid']
    assert audit['cap_audit']['raw_remaining_boundary_or_nonmanifold_edges'] > 0
    assert audit['cap_audit']['method'] == 'verified_boundary_centroid_fans'
    assert all(c['fan_verified'] for c in audit['cap_audit']['opening_caps'])
    assert slicer.surface.GetNumberOfPoints() == len(xyz)
    assert slicer.surface.GetNumberOfPolys() == len(triangles)


def test_fan_proof_rejects_nonstar_and_repeated_winding():
    # Concave U-shaped polygon: a centroid fan would incorrectly fill its gap.
    nonstar = np.array([[0, 0, 0], [3, 0, 0], [3, 3, 0], [2, 3, 0],
                       [2, 1, 0], [1, 1, 0], [1, 3, 0], [0, 3, 0]], float)
    assert _verified_fan_center(nonstar)[0] is None
    angle = np.arange(5) * 4 * np.pi / 5
    crossing_star = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(5)))
    assert _verified_fan_center(crossing_star)[0] is None


def test_boundary_cycles_do_not_repair_nonmanifold_source():
    xyz, triangles = tube(n=12)
    cycles, audit = _boundary_cycles(np.r_[triangles, triangles[:1]])
    assert not cycles and audit['nonmanifold_edges'] > 0
    _, enclosed = SurfaceSlicer(xyz, np.r_[triangles, triangles[:1]]).enclosed(np.array([[0., 0., 10.]]))
    assert not enclosed['closed_reference_valid']


def test_exact_opposite_fin_removal_preserves_real_wall_triangles():
    xyz, triangles = tube(n=12)
    a, b = triangles[0, [0, 2]]
    fin = np.array([[a, b, len(xyz)], [b, a, len(xyz)]])
    cleaned, evidence = _remove_opposite_triangle_fins(np.r_[triangles, fin])
    assert np.array_equal(cleaned, triangles)
    assert evidence['removed_triangle_rows'] == [len(triangles), len(triangles) + 1]
    assert sorted(evidence['pairs'][0]['edge_incidence_before']) == [2, 2, 4]
    assert sorted(evidence['pairs'][0]['edge_incidence_after']) == [0, 0, 2]
    # Two equally oriented copies are not certified as a zero-thickness fin.
    unaltered, evidence = _remove_opposite_triangle_fins(np.r_[triangles, fin[[0, 0]]])
    assert len(unaltered) == len(triangles) + 2 and not evidence['removed_triangle_rows']


def test_opening_overrun_domain_excludes_only_verified_endpoint_tails():
    xyz = np.array([[0, 0, -2], [0, 0, -1], [0, 0, .5], [3, 0, 3], [0, 0, 8],
                    [0, 0, 12], [0, 0, 18], [0, 0, 21], [0, 0, 22]], float)
    sid = np.array([0] * 5 + [1] * 4)
    s = np.array([0, 1, 2.5, 5, 10, 0, 6, 9, 10], float)
    inside = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0], bool)
    angle = np.arange(32) * 2 * np.pi / 32
    ring = np.column_stack((2 * np.cos(angle), 2 * np.sin(angle), np.zeros(32)))
    segments = [{'segment_id': 0, 'parent_id': -1, 'children': [1, 2], 'start_s_mm': 0, 'end_s_mm': 10},
                {'segment_id': 1, 'parent_id': 0, 'children': [], 'start_s_mm': 0, 'end_s_mm': 10}]
    openings = [{'segment_id': 0, 'valid': True, 'label_provisional': 'inlet', 'center_mm': [0, 0, 0],
                 'normal': [0, 0, -1], 'polygon_xyz_mm': ring},
                {'segment_id': 1, 'valid': True, 'label_provisional': 'out-le', 'center_mm': [0, 0, 20],
                 'normal': [0, 0, 1], 'polygon_xyz_mm': ring + [0, 0, 20]}]
    excluded, evidence = opening_overrun_domain(xyz, sid, s, segments, openings, inside=inside)
    assert np.flatnonzero(excluded).tolist() == [0, 1, 7, 8]
    assert not excluded[3] and not inside[3]  # A real wall escape stays visible.
    assert evidence['segment_s_bounds_mm'] == {0: [2., 10.], 1: [0., 8.]}
    assert not evidence['unresolved_extensions']
    # A tail leaving sideways is not an opening extension.
    xyz[-1, 0] = 3
    excluded, _ = opening_overrun_domain(xyz, sid, s, segments, openings, inside=inside)
    assert not excluded[7:].any()


def test_opening_overrun_requires_interior_return_and_actual_cap_crossing():
    angle = np.arange(32) * 2 * np.pi / 32
    ring = np.column_stack((2 * np.cos(angle), 2 * np.sin(angle), .1 * np.cos(2 * angle)))
    segments = [{'segment_id': 0, 'parent_id': -1, 'children': [1, 2], 'start_s_mm': 0., 'end_s_mm': 2.}]
    opening = [{'segment_id': 0, 'valid': True, 'label_provisional': 'inlet', 'center_mm': [0, 0, 0],
                'normal': [0, 0, -1], 'polygon_xyz_mm': ring}]
    xyz = np.array([[0, 0, -1], [0, 0, -.01], [0, 0, 1]], float)
    excluded, evidence = opening_overrun_domain(xyz, np.zeros(3, int), np.arange(3.), segments, opening,
                                                 inside=np.zeros(3, bool))
    assert not excluded.any() and evidence['unresolved_extensions']
    # A point inside the rim's plane band needs exact facet evidence and
    # agreement with the independent inside audit, not a larger tolerance.
    excluded, evidence = opening_overrun_domain(xyz, np.zeros(3, int), np.arange(3.), segments, opening,
                                                 inside=np.array([0, 0, 1], bool))
    assert excluded.tolist() == [True, True, False]
    extension = evidence['extensions'][0]
    assert extension['crossing_method'] == 'verified_3d_cap_triangle_intersection'
    assert np.allclose(extension['crossing_xyz_mm'], [0, 0, 0], atol=1e-12)
    assert np.all(np.array(extension['actual_cap_outward_offsets_mm']) > 0)


def test_ellipse_centroid_and_area_are_not_inscribed_radius():
    a = np.linspace(0, 2 * np.pi, 721)[:-1]
    points = np.column_stack((4 * np.cos(a) + 1, 2 * np.sin(a), np.zeros(len(a))))
    m = polygon_metrics(points, np.zeros(3), np.array([0, 0, 1]))
    assert m['valid'] and m['contains_center']
    assert abs(m['area_mm2'] / (8 * np.pi) - 1) < 1e-4
    assert np.allclose(m['centroid_mm'], [1, 0, 0], atol=1e-9)
    assert abs(m['eccentricity'] - 1 / np.sqrt(8)) < 1e-4


def test_pointcloud_density_and_missing_sector_detection():
    a = np.arange(360) * 2 * np.pi / 360
    z = np.linspace(-1, 1, 7)
    pts = np.vstack([np.column_stack((3 * np.cos(a), 3 * np.sin(a), np.full(len(a), zz))) for zz in z])
    full = pointcloud_section(pts, np.zeros(3), np.array([0, 0, 1]), 3., .3)
    half = pointcloud_section(pts[::2], np.zeros(3), np.array([0, 0, 1]), 3., .3)
    assert full['valid'] and half['valid']
    assert abs(full['area_mm2'] / (9 * np.pi) - 1) < .01
    assert abs(full['area_mm2'] / half['area_mm2'] - 1) < 1e-12
    missing = pointcloud_section(pts[pts[:, 0] < 0], np.zeros(3), np.array([0, 0, 1]), 3., .3)
    assert not missing['valid'] and missing['reason'] == 'angular_coverage_gap'


def test_continuous_projection_and_gap_safe_interpolation():
    xyz = np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 2.], [10., 0., 0.], [10., 0., 2.]])
    sid = np.array([0, 0, 0, 1, 1])
    s = np.array([0., 1., 2., 0., 2.])
    q = np.array([[1., 0., .999], [1., 0., 1.001], [5., 0., 1.]])
    m = project_wall(q, xyz, sid, s)
    assert np.allclose(m['s_local_mm'][:2], [.999, 1.001])
    assert m['ambiguous'].tolist() == [False, False, True]
    val, valid = interpolate_valid_field(np.array([.5, 1.5, 2.5]), np.arange(4.), np.array([1., 2., 3., 4.]), np.array([1, 1, 0, 1], bool))
    assert valid.tolist() == [True, False, False]
    assert val[0] == 1.5 and np.isnan(val[1:]).all()


def test_log_area_derivative_does_not_bridge_mask():
    s = np.arange(7.)
    out = log_area_slopes(s, np.exp(s), np.array([1, 1, 1, 0, 1, 1, 1], bool))
    assert np.allclose(out[[0, 1, 2, 4, 5, 6]], 1)
    assert np.isnan(out[3])


def test_cross_side_cfd_names_do_not_become_valid_anatomy():
    segs = []
    parents = [-1, 0, 0, 1, 1, 2, 2]
    names = ['', '', '', 'out-le', 'out-ri', 'out-li', 'out-re']
    for i in range(7):
        segs.append({'segment_id': i, 'parent_id': parents[i], 'outlet_name': names[i]})
    xyz = np.array([[i, 0, j] for i in range(7) for j in range(2)], float)
    rows, audit = topology_semantics(segs, xyz, np.repeat(np.arange(7), 2), np.tile([0., 1.], 7))
    assert audit['valid']
    assert audit['mixed_side_cia_segments'] == [1, 2]
    assert all(r['anatomy_label'] == 'unknown' and not r['semantic_valid'] for r in rows if r['segment_id'] != 0)
    assert rows[3]['cfd_zone_label'] == 'out-le'


def test_upstream_context_stops_at_invalid_sections():
    s = np.arange(7.)
    area = np.array([2., 1., 3., .1, 4., 2., 5.])
    valid = np.array([1, 1, 1, 0, 1, 1, 1], bool)
    c = branch_context(s, area, valid, 10.)
    assert c['upstream_min_area_ratio'][2] == 1 / 3
    assert np.isnan(c['upstream_min_area_ratio'][4])
    assert c['upstream_min_area_ratio'][6] == 2 / 5
    assert c['upstream_min_distance_mm'][6] == 1


def test_context_plateau_chooses_nearest_in_both_directions():
    c = branch_context(np.arange(11.), np.full(11, 10.), np.ones(11, bool), 4.)
    assert c['upstream_min_distance_mm'][5] == c['downstream_min_distance_mm'][5] == 1


def test_generic_seven_segment_binary_tree_is_not_sufficient_anatomy():
    parents = [-1, 0, 0, 2, 2, 4, 4]
    segs = [{'segment_id': i, 'parent_id': p, 'outlet_name': 'out-le'} for i, p in enumerate(parents)]
    xyz = np.array([[i, 0, j] for i in range(7) for j in range(2)], float)
    _, audit = topology_semantics(segs, xyz, np.repeat(np.arange(7), 2), np.tile([0., 1.], 7))
    assert audit['generic_tree_valid']
    assert not audit['valid'] and not audit['expected_aortoiliac_pattern_valid']


def test_perturbation_gate_preserves_raw_contour_and_rejects_near_junction():
    angle = np.arange(120) * 2 * np.pi / 120
    polygon = np.column_stack((np.cos(angle), np.sin(angle), np.full(len(angle), 1.)))
    raw = polygon_metrics(polygon, np.array([0., 0., 1.]), np.array([0., 0., 1.]))
    seg = {'segment_id': 1, 'parent_id': 0, 'children': [2, 3], 'start_s_mm': 0., 'end_s_mm': 20.}
    cfg = {'junction_exclusion_mm': 2.5, 'endpoint_exclusion_mm': 1.}
    result = section_geometry_gate(raw, np.array([0., 0., 1.]), np.array([0., 0., 1.]), seg, 1., np.array([[0., 0., 1.]]), np.array([1]), cfg)
    assert result['raw_contour_valid'] and not result['pipeline_valid'] and not result['valid']
    assert result['pipeline_reason'] == 'junction_exclusion'


def test_continuous_plane_vertex_crossing_is_one_group():
    xyz = np.array([[0., 0., -2.], [0., 0., 0.], [0., 0., 2.]])
    groups = polyline_plane_intersection_groups(xyz, np.array([0., 2., 4.]), np.zeros(3), np.array([0., 0., 1.]))
    assert len(groups) == 1 and groups[0]['s_min_mm'] == groups[0]['s_max_mm'] == 2.


def test_returning_polyline_crossing_is_rejected_only_inside_selected_contour():
    xyz = np.array([[0., 0., -2.], [0., 0., 0.], [0., 0., 2.], [10., 0., 2.], [10., 0., -2.]])
    ss = np.array([0., 2., 4., 14., 18.])
    angle = np.arange(120) * 2 * np.pi / 120
    large = np.column_stack((20 * np.cos(angle), 20 * np.sin(angle), np.zeros(len(angle))))
    small = large / 10
    args = (np.zeros(3), np.array([0., 0., 1.]), xyz, ss, 2.)
    hit = same_segment_nonlocal_crossings(large, *args)
    assert len(hit['inside_crossing_groups']) == 2 and len(hit['extra_nonlocal_groups']) == 1
    assert hit['extra_nonlocal_groups'][0]['s_min_mm'] == 16.
    miss = same_segment_nonlocal_crossings(small, *args)
    assert len(miss['inside_crossing_groups']) == 1 and not miss['extra_nonlocal_groups']
    seg = {'segment_id': 0, 'parent_id': -1, 'children': [], 'start_s_mm': 0., 'end_s_mm': 18.}
    cfg = {'endpoint_exclusion_mm': 1., 'junction_exclusion_mm': 2.5, 'same_segment_nonlocal_s_gap_mm': 5., 'plane_intersection_tolerance_mm': 1e-5}
    raw = polygon_metrics(large, np.zeros(3), np.array([0., 0., 1.]))
    result = section_geometry_gate(raw, np.zeros(3), np.array([0., 0., 1.]), seg, 2., xyz, np.zeros(len(xyz), int), cfg, ss)
    assert result['raw_contour_valid'] and not result['pipeline_valid']
    assert result['same_segment_nonlocal_crossing_count'] == 1
    assert result['pipeline_reason'] == 'section_contains_same_segment_nonlocal_plane_crossing'


def test_nearly_coplanar_run_is_one_interval_not_repeated_crossings():
    xyz = np.array([[0., 0., -2.], [0., 0., -1e-6], [1., 0., 1e-6], [2., 0., -1e-6], [2., 0., 2.]])
    ss = np.array([0., 2., 3., 4., 6.])
    groups = polyline_plane_intersection_groups(xyz, ss, np.zeros(3), np.array([0., 0., 1.]), plane_tol_mm=1e-5)
    assert len(groups) == 1 and groups[0]['coplanar_interval']
    assert groups[0]['s_min_mm'] == 2. and groups[0]['s_max_mm'] == 4.
