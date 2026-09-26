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


def test_report_schema_metadata_backfills_legacy_wss_as_one_frame():
    from wss_deploy.report import _report_schema_metadata

    out = _report_schema_metadata({'wss_field_pa': {}, 'peak': {}})
    assert list(out['fields']) == ['wss']
    assert out['fields']['wss']['kind'] == 'scalar'
    assert out['fields']['wss']['units'] == 'Pa'
    assert out['fields']['wss']['location'] == 'wall'
    assert out['fields']['wss']['array_key'] == 'wss_pa'
    assert len(out['time_axis']) == 1


def test_report_keeps_unbacked_frames_and_fields_non_switchable(tmp_path):
    """A metadata-only multi-frame/velocity declaration must not fake arrays."""
    pts = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
    meta = {
        'case_id': 'multi', 'release': 'test', 'endpoints': [], 'branch_names': {},
        'per_branch': {}, 'geometry': {}, 'input_check': {}, 'cloud': {}, 'timing_s': {},
        'fields': {
            'wss': {'id': 'wss', 'label': 'WSS', 'units': 'Pa', 'location': 'wall',
                    'kind': 'scalar', 'components': 1, 'array_key': 'wss_pa',
                    'axis_order': ['time', 'point'], 'time_indices': [0, 1]},
            'velocity': {'id': 'velocity', 'label': '速度', 'units': 'm/s', 'location': 'wall',
                         'kind': 'vector', 'components': 3, 'array_key': 'velocity',
                         'axis_order': ['time', 'point', 'component'], 'time_indices': [0, 1]},
        },
        'time_axis': [{'index': 0, 'label': 't0'}, {'index': 1, 'label': 't1'}],
        'wss_field_pa': {'p99': 2., 'p99_pa': 2., 'mean': 1., 'median': 1.,
                         'thresholds_pa': [.4, 4., 7.], 'area_frac_low': 0.,
                         'area_low_mm2': 0., 'area_frac_high': 0., 'area_high_mm2': 0.,
                         'area_frac_very_high': 0.},
        'peak': {'p99_pa': 2., 'max_pa': 3., 'xyz_mm': [0, 0, 0], 'branch': 'x',
                 's_from_inlet_mm': 0., 'dist_to_junction_mm': 0., 'local_radius_mm': 1.,
                 'top5_clusters_ge20': 0, 'top5_cluster_sizes': []},
    }
    mesh = {'vertices': pts, 'faces': np.array([[0, 1, 1]], dtype=np.uint32),
            'wss': np.array([1., 2.], dtype=np.float32), 'segment': np.array([1, 1], dtype=np.int32)}
    cloud = {'pts': pts, 'wss': np.array([1., 2.], dtype=np.float32), 'segment': np.array([1, 1], dtype=np.int32),
             's_from_root_mm': np.array([0., 1.], dtype=np.float32), 'theta_rad': np.array([0., 1.], dtype=np.float32),
             'radius_mm': np.array([1., 1.], dtype=np.float32), 'dist_to_junction_mm': np.array([0., 1.], dtype=np.float32)}
    out = tmp_path / 'report.html'
    build_html(out, meta, mesh, cloud, {'xyz': pts, 'radius_mm': np.array([1., 1.], dtype=np.float32),
                                        'edges': np.array([[0, 1]], dtype=np.uint32), 'segment': np.array([1, 1], dtype=np.int32)})
    html = out.read_text()
    assert '<select id="frame-select" aria-describedby="frame-status" disabled>' in html
    assert 'select.disabled=true' in html
    assert "d.kind==='scalar'" in html and "d.units==='Pa'" in html
    assert '没有速度或压力数据' not in html


# ---------------------------------------------------------------------------
# Viewer helpers (WssReportCore) are exercised in Node; the page itself is smoke
# tested with a fake document so the no-WebGL path and optional-panel logic run.
# ---------------------------------------------------------------------------
import base64
import json
import re
import shutil
import subprocess
from pathlib import Path


def _node(script: str, tmp_path: Path, *extra_files: tuple[str, str]) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is required for the report viewer tests")
    from wss_deploy.paths import STATIC_DIR
    from wss_deploy.report import REPORT_CORE_JS
    core = tmp_path / "core.js"
    core.write_text(REPORT_CORE_JS, encoding="utf-8")
    # The page embeds the shared library right before its own script; the stub must load it the same way.
    (tmp_path / "common.js").write_text((STATIC_DIR / "report_common.js").read_text(encoding="utf-8"), encoding="utf-8")
    for name, content in extra_files:
        (tmp_path / name).write_text(content, encoding="utf-8")
    program = "const core=require(" + json.dumps(str(core)) + ");const DIR=" + json.dumps(str(tmp_path)) + ";\n" + script
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr[-3000:]   # show the page error, not the whole program
    return json.loads(result.stdout)


def _minimal_report_inputs():
    pts = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], dtype=np.float32)
    meta = {'case_id': 'legacy', 'release': 'rel', 'endpoints': [], 'branch_names': {'1': '主动脉'},
            'per_branch': {'主动脉': {'segment_id': 1, 'n_points': 4, 'area_mm2': 10., 'wss_mean_pa': 2., 'wss_p99_pa': 3., 'wss_max_pa': 4., 'frac_low': 0., 'frac_high': 0.}},
            'geometry': {}, 'input_check': {'area_mm2': 10.}, 'cloud': {'spacing_mm': .5}, 'timing_s': {},
            'wss_field_pa': {'p99': 3., 'p99_pa': 3., 'mean': 2., 'median': 2., 'thresholds_pa': [.4, 4., 7.],
                             'area_frac_low': 0., 'area_low_mm2': 0., 'area_frac_high': 0., 'area_high_mm2': 0., 'area_frac_very_high': 0.},
            'peak': {'p99_pa': 3., 'max_pa': 4., 'xyz_mm': [1, 1, 0], 'branch': '主动脉', 's_from_inlet_mm': 3.,
                     'dist_to_junction_mm': 0., 'local_radius_mm': 1., 'top5_clusters_ge20': 0, 'top5_cluster_sizes': []}}
    mesh = {'vertices': pts, 'faces': np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32),
            'wss': np.array([1., 2., 3., 4.], dtype=np.float32), 'segment': np.ones(4, dtype=np.int32)}
    cloud = {'pts': pts, 'wss': np.array([1., 2., 3., 4.], dtype=np.float32), 'segment': np.ones(4, dtype=np.int32),
             's_from_root_mm': np.array([0., 1., 2., 3.], dtype=np.float32), 'theta_rad': np.zeros(4, dtype=np.float32),
             'radius_mm': np.ones(4, dtype=np.float32), 'dist_to_junction_mm': np.array([0., 1., 2., 3.], dtype=np.float32)}
    centerline = {'xyz': pts[:2], 'radius_mm': np.ones(2, dtype=np.float32), 'edges': np.array([[0, 1]], dtype=np.uint32), 'segment': np.ones(2, dtype=np.int32)}
    return meta, mesh, cloud, centerline


def _analysis_meta(meta):
    return {**meta, 'run_identity': 'r' * 64,
            'frame_transform': {'source': 'x', 'direction_source': 'unknown_stl', 'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'origin_mm': [0, 0, 0]},
            'profiles': {'schema_version': 'wss-deploy.profiles/v1', 'bin_mm': 2.0, 'branches': [
                {'segment_id': 1, 'name': '主动脉', 'parent_id': -1, 's_local_mm': [1., 3.], 's_from_root_mm': [1., 3.], 'radius_mm': [1., 1.],
                 'wss': {'mean_pa': [1.5, 3.5], 'p99_pa': [2., 4.], 'min_pa': [1., 3.], 'n': [2, 2]}}]},
            'findings': {'schema_version': 'wss-deploy.findings/v1', 'items': [
                {'id': 'F1', 'kind': 'max_wss', 'label': '全场最大 WSS', 'branch': '主动脉', 'segment_id': 1, 'value': 4., 'units': 'Pa',
                 'xyz_mm': [1, 1, 0], 's_from_root_mm': 3., 'extent_mm': 1., 'area_mm2': None, 'n_points': 1, 'rank': 1, 'severity': 'note',
                 'definition': '全场最大值预测点', 'point_indices': [3]}]},
            'trust': {'schema_version': 'wss-deploy.trust/v1', 'bits': {'1': 'interpolation_uncovered', '2': 'rough_surface', '4': 'geometry_out_of_range'},
                      'fractions': {'interpolation_uncovered': 0., 'rough_surface': .25, 'geometry_out_of_range': 0.},
                      'sources': [{'bit': 2, 'label': '表面粗糙', 'rule': 'PCA surface_variation > 0.02'}]},
            'review': {'status': 'reviewed', 'by': 'reviewer-01', 'at': '2026-09-20T20:00:00+08:00', 'note': '', 'version': 3}}


def _labels_meta(meta):
    """§17: three ranked findings, the trimmed morphology stations, an aorta maximum and a narrative."""
    base = _analysis_meta(meta)
    items = list(base['findings']['items']) + [
        {'id': 'F2', 'kind': 'high_wss_cluster', 'label': '主动脉高 WSS 区', 'branch': '主动脉', 'segment_id': 1,
         'value': 21.9, 'units': 'Pa', 'xyz_mm': [0, 1, 0], 's_from_root_mm': 2., 'extent_mm': 4., 'area_mm2': 310.,
         'n_points': 12, 'rank': 2, 'severity': 'attention', 'definition': 'WSS ≥ 空间 p99 的连通簇'},
        {'id': 'F3', 'kind': 'low_wss_cluster', 'label': '主动脉低 WSS 区', 'branch': '主动脉', 'segment_id': 1,
         'value': .2, 'units': 'Pa', 'xyz_mm': [1, 0, 0], 's_from_root_mm': 1., 'extent_mm': 4., 'area_mm2': 120.,
         'n_points': 8, 'rank': 3, 'severity': 'note', 'definition': 'WSS < 0.4 Pa 的连通簇'},
    ]
    ring = [[.5, .5, 0.], [1.5, .5, 0.], [1.5, 1.5, 0.], [.5, 1.5, 0.]]
    return {**base,
            'findings': {**base['findings'], 'items': items},
            'morphology': {'schema_version': 'wss-deploy.morphology/v1', 'station_mm': 1.0,
                           'aorta': {'segment_id': 1, 'name': '主动脉', 'length_mm': 3.,
                                     'reference_diameter_mm': 1.8,
                                     'max': {'max_diameter_mm': 63.4, 'equivalent_diameter_mm': 58.9,
                                             'min_diameter_mm': 52.0, 'area_mm2': 2725.0, 's_from_root_mm': 2.,
                                             'xyz_mm': [1, 1, 0], 'distance_from_inlet_mm': 120.0,
                                             'polygon_world': ring},
                                     'sac': {'present': False}, 'neck': {'present': False}},
                           'lumen_volume_ml': 158.2,
                           'branches': [{'segment_id': 1, 'name': '主动脉', 'length_mm': 3., 'tortuosity': 1.06,
                                         'stations': {'s_from_root_mm': [1., 2., 3.],
                                                      'max_diameter_mm': [20., 63.4, 24.],
                                                      'equivalent_diameter_mm': [18., 58.9, 22.]}}],
                           'notes': ['无瘤样扩张']},
            'narrative': {'schema_version': 'wss-deploy.narrative/v1', 'generated_at': '2026-09-22T09:00:00+08:00',
                          'zh': ['主动脉最大直径 63.4 mm，位于入口下 120 mm。', '低 WSS 区占壁面 34%。'],
                          'en': ['Maximum aortic diameter 63.4 mm, 120 mm below the inlet.'],
                          'edited': None, 'edited_by': None, 'edited_at': None}}


def test_core_marching_triangles_crosses_edges_and_skips_nonfinite(tmp_path):
    out = _node("""
      const V=new Float32Array([0,0,0, 1,0,0, 0,1,0]),F=new Uint32Array([0,1,2]);
      console.log(JSON.stringify({seg:Array.from(core.marchingTriangles(V,F,new Float32Array([0,1,2]),.5)),
        nan:core.marchingTriangles(V,F,new Float32Array([0,NaN,2]),.5).length, outside:core.marchingTriangles(V,F,new Float32Array([0,1,2]),5).length}));
    """, tmp_path)
    assert out["seg"] == pytest.approx([.5, 0, 0, 0, .25, 0])
    assert out["nan"] == 0 and out["outside"] == 0


def test_core_view_state_hash_roundtrip_with_unicode(tmp_path):
    out = _node("""
      const s={schema_version:'wss-deploy.view/v1',family:'wall',camera:{position:[1,2,3],target:[0,0,0],up:[0,0,1]},units:'dyn',note:'左髂内 · 复现'};
      const h=core.viewHash(s);
      console.log(JSON.stringify({same:JSON.stringify(core.parseViewHash(h))===JSON.stringify(s),prefix:h.slice(0,6),safe:/^#view=[A-Za-z0-9_-]+$/.test(h),none:core.parseViewHash('#foo=1')}));
    """, tmp_path)
    assert out == {"same": True, "prefix": "#view=", "safe": True, "none": None}


def test_core_standard_views_and_anatomical_camera_roundtrip(tmp_path):
    out = _node("""
      const I=[[1,0,0],[0,1,0],[0,0,1]],v=core.standardViews(I,[0,0,0],10);
      const R=[[0,1,0],[-1,0,0],[0,0,1]],o=[5,5,5],cam={position:[10,20,30],target:[1,2,3],up:[0,0,1]};
      const a=core.cameraToAligned(cam,R,o),b=core.cameraFromAligned(a,R,o);
      console.log(JSON.stringify({front:v.front,left:v.left,top:v.top,aligned:a,back:b,valid:[core.validFrame({rotation:I,origin_mm:[0,0,0]}),core.validFrame({rotation:[[1,0]],origin_mm:[0,0,0]}),core.validFrame(null)]}));
    """, tmp_path)
    assert out["front"] == {"position": [0, -10, 0], "target": [0, 0, 0], "up": [0, 0, 1]}
    assert out["left"]["position"] == [10, 0, 0] and out["top"] == {"position": [0, 0, 10], "target": [0, 0, 0], "up": [0, -1, 0]}
    assert out["aligned"]["position"] == [15, -5, 25]
    assert out["back"] == {"position": [10, 20, 30], "target": [1, 2, 3], "up": [0, 0, 1]}
    assert out["valid"] == [True, False, False]


def test_core_threshold_areas_units_and_scales(tmp_path):
    out = _node("""
      console.log(JSON.stringify({areas:core.thresholdAreas([.1,.5,5,8],[.4,4,7],100),masked:core.thresholdAreas([.1,.5,5,8],[.4,4,7],100,[1,1,0,0]).frac_low,
        units:[core.toUnit(1,'dyn'),core.fromUnit(10,'dyn'),core.toUnit(2,'nope'),core.unitInfo('dyn').label],
        valid:[core.validThresholds([.4,4,7]),core.validThresholds([4,.4,7]),core.validThresholds(['1','2','3'])],
        norm:[core.normalize(5,0,10,false),core.normalize(.05,0,10,true),core.normalize(10,0,10,true)],
        quant:[core.quantize(.1,4),core.quantize(.99,4),core.quantize(.3,0)],edges:core.bandEdges(0,10,4,false),top:core.topThreshold([1,2,3,4,5,6,7,8,9,10],10)}));
    """, tmp_path)
    a = out["areas"]
    assert (a["n"], a["frac_low"], a["frac_high"], a["frac_very_high"]) == (4, .25, .5, .25)
    assert (a["area_low_mm2"], a["area_high_mm2"], a["area_very_high_mm2"]) == (25, 50, 25)
    assert out["masked"] == .5
    assert out["units"] == [10, 1, 2, "dyn/cm²"]
    assert out["valid"] == [[.4, 4, 7], None, [1, 2, 3]]
    assert out["norm"] == pytest.approx([.5, 0, 1])
    assert out["quant"] == pytest.approx([.125, .875, .3])
    assert out["edges"] == [0, 2.5, 5, 7.5, 10] and out["top"] == 9


def test_core_nearest_index_and_region_helpers(tmp_path):
    out = _node("""
      const pts=new Float32Array([0,0,0, 10,0,0, 0,10,0, 10,10,10]),q=new Float32Array([.1,0,0, 9,1,0, 8,9,9, 100,100,100]);
      console.log(JSON.stringify({near:Array.from(core.nearestIndex(pts,q,2)),auto:Array.from(core.nearestIndex(pts,q)),
        sphere:Array.from(core.sphereMask(pts,[0,0,0],10.5)),range:Array.from(core.rangeMask([0,5,10,15],[1,1,2,1],1,4,16)),any:Array.from(core.rangeMask([0,5,10,15],[1,1,2,1],null,4,16)),
        idx:core.maskIndices([0,1,1,0]),stats:core.stats([1,2,3,100,NaN],[0,1,2,3,4])}));
    """, tmp_path)
    assert out["near"] == [0, 1, 3, 3] and out["auto"] == [0, 1, 3, 3]
    assert out["sphere"] == [1, 1, 1, 0] and out["range"] == [0, 1, 0, 1] and out["any"] == [0, 1, 1, 1]
    assert out["idx"] == [1, 2]
    assert out["stats"]["n"] == 4 and out["stats"]["mean"] == 26.5 and out["stats"]["max"] == 100


def test_build_html_embeds_optional_trust_and_rejects_bad_shape(tmp_path):
    meta, mesh, cloud, centerline = _minimal_report_inputs()
    out = tmp_path / "r.html"
    build_html(out, meta, mesh, cloud, centerline)
    html = out.read_text(encoding="utf-8")
    assert '"mt":' not in html
    for marker in ('id="menu-display"', 'id="menu-findings"', 'id="menu-profiles"', 'id="menu-region"', 'id="menu-view"', 'id="menu-stats"',
                   'id="menu-measure"', 'id="menu-annot"', 'id="menu-presets"', 'id="branch-vis"', 'id="probe-card"', 'id="exp-png"',
                   'id="footer"', 'WssReportCore', 'WssReportCommon', 'id="std-views"', 'id="cmap"', 'id="bands"', 'id="units"', 'id="contours"', 'id="trust"'):
        assert marker in html
    build_html(out, meta, {**mesh, 'trust': np.array([0, 2, 0, 4], dtype=np.uint8)}, cloud, centerline)
    assert '"mt":' in out.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="one bitmask value per vertex"):
        build_html(out, meta, {**mesh, 'trust': np.array([0, 2], dtype=np.uint8)}, cloud, centerline)


def _main_script() -> str:
    """The page's own inline script: everything but the four embedded library placeholders."""
    from wss_deploy.report import TEMPLATE
    blocks = [b for b in re.findall(r"<script>(.*?)</script>", TEMPLATE, flags=re.S)
              if b not in ("__THREE__", "__ORBIT__", "__CORE__", "__COMMON__")]
    assert len(blocks) == 1
    return blocks[0]


def test_inline_report_scripts_parse_in_node(tmp_path):
    if not shutil.which("node"):
        pytest.skip("Node is required")
    from wss_deploy.paths import STATIC_DIR
    from wss_deploy.report import REPORT_CORE_JS
    for name, code in (("core.js", REPORT_CORE_JS), ("main.js", _main_script()),
                       ("common.js", (STATIC_DIR / "report_common.js").read_text(encoding="utf-8"))):
        path = tmp_path / name
        path.write_text(code, encoding="utf-8")
        subprocess.run(["node", "--check", str(path)], check=True, capture_output=True)


# The DOM stub: enough of an Element/document/window to run the page head-less.  ``THREE`` is either a
# renderer that throws (the no-WebGL fallback) or the small fake below, which records what the page builds.
_STUB_DOM = r"""
const fs=require('fs');
class Element{constructor(id){this.id=id;this.children=[];this.events={};this.hidden=false;this.disabled=false;this.open=false;this.checked=false;this.value='';this._text='';this._html='';this.style={};this.dataset={};this.files=null;
  this.clientWidth=800;this.clientHeight=600;this.width=800;this.height=600;this.parent=null;
  this.classList={_s:new Set(),toggle(c,f){if(f===undefined)f=!this._s.has(c);f?this._s.add(c):this._s.delete(c);return f;},add(c){this._s.add(c);},remove(c){this._s.delete(c);},contains(c){return this._s.has(c);}};}
 set textContent(v){this._text=String(v);} get textContent(){return this._text;} set innerHTML(v){this._html=String(v);} get innerHTML(){return this._html;}
 set className(v){this.classList._s=new Set(String(v).split(/\s+/).filter(Boolean));} get className(){return Array.from(this.classList._s).join(' ');}
 appendChild(x){x.parent=this;this.children.push(x);return x;} append(...xs){xs.forEach(x=>this.appendChild(x));} replaceChildren(){this.children=[];}
 remove(){if(this.parent){const i=this.parent.children.indexOf(this);if(i>=0)this.parent.children.splice(i,1);this.parent=null;}}
 addEventListener(n,f){(this.events[n]=this.events[n]||[]).push(f);} setAttribute(k,v){this[k]=v;} getAttribute(k){return this[k];}
 querySelectorAll(){return [];} querySelector(){return null;} closest(){return null;} click(){} getContext(){return null;} getBoundingClientRect(){return {left:0,top:0,width:800,height:600};} toDataURL(){return 'data:image/png;base64,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA';}}
const els={};
const docListeners={};
global.document={getElementById:id=>els[id]||(els[id]=new Element(id)),createElement:t=>new Element(t),querySelectorAll:()=>[],body:new Element('body'),
  addEventListener:(n,f)=>{(docListeners[n]=docListeners[n]||[]).push(f);},removeEventListener:(n,f)=>{docListeners[n]=(docListeners[n]||[]).filter(x=>x!==f);}};
// A keydown on the report document (what installShortcuts listens to); returns the event so tests can see preventDefault.
const key=(k,o)=>{const ev=Object.assign({key:k,target:{tagName:'BODY'},defaultPrevented:false,preventDefault(){this.defaultPrevented=true;}},o||{});(docListeners.keydown||[]).forEach(f=>f(ev));return ev;};
els['wss-report-meta']=new Element('m');els['wss-report-meta']._text=fs.readFileSync(DIR+'/meta.json','utf8');
els['wss-report-arrays']=new Element('a');els['wss-report-arrays']._text=fs.readFileSync(DIR+'/arrays.json','utf8');
els['wss-glossary']=new Element('gl');els['wss-glossary']._text=fs.readFileSync(DIR+'/glossary.json','utf8');
"""

_DOMLESS = _STUB_DOM + r"""
global.window=global;global.THREE={WebGLRenderer:class{constructor(){throw new Error('No WebGL');}}};global.localStorage={getItem:()=>null,setItem:()=>{}};
global.location={hash:'',href:'file:///r.html',protocol:'file:',pathname:'/r.html'};global.history={replaceState(){}};global.navigator={};global.performance={now:()=>0};global.requestAnimationFrame=()=>0;global.addEventListener=()=>{};
const errors=[];console.error=(...a)=>errors.push(a.map(String).join(' '));
require(DIR+'/main.js');
const g=id=>document.getElementById(id);
console.log(JSON.stringify({panel:g('panel')._html.length,footer:g('footer')._html,findings:g('findings-list').children.length,findingsNote:g('findings-note')._text,
  profOptions:g('prof-branch').children.length,trustDisabled:g('trust').disabled,trustHint:g('trust-hint')._text,thr:[g('thr0').value,g('thr1').value,g('thr2').value],
  branches:g('branch').children.length,fallback:g('view').children.length,measureHidden:g('menu-measure').hidden,errors}));
"""

# A fake three.js: object graph, buffers and a renderer whose canvas yields a data URL.  It never draws.
_FAKE_THREE = r"""
class V3{constructor(x,y,z){this.x=+x||0;this.y=+y||0;this.z=+z||0;}
 set(x,y,z){this.x=+x||0;this.y=+y||0;this.z=+z||0;return this;} setScalar(v){return this.set(v,v,v);}
 clone(){return new V3(this.x,this.y,this.z);} copy(v){return this.set(v.x,v.y,v.z);}
 sub(v){return this.set(this.x-v.x,this.y-v.y,this.z-v.z);} normalize(){const l=Math.hypot(this.x,this.y,this.z)||1;return this.set(this.x/l,this.y/l,this.z/l);}
 getComponent(i){return [this.x,this.y,this.z][i];} toArray(){return [this.x,this.y,this.z];} project(){return this;}
 distanceToSquared(v){return (this.x-v.x)**2+(this.y-v.y)**2+(this.z-v.z)**2;}
 lerpVectors(a,b,t){return this.set(a.x+(b.x-a.x)*t,a.y+(b.y-a.y)*t,a.z+(b.z-a.z)*t);}}
class Obj{constructor(){this.position=new V3();this.scale=new V3(1,1,1);this.up=new V3(0,1,0);this.children=[];this.visible=true;this.renderOrder=0;}
 add(o){this.children.push(o);return this;} remove(o){const i=this.children.indexOf(o);if(i>=0)this.children.splice(i,1);return this;}}
class Geo{constructor(){this.attributes={};this.index=null;} setAttribute(n,a){this.attributes[n]=a;return this;} getAttribute(n){return this.attributes[n];}
 setIndex(i){this.index=i;return this;} computeVertexNormals(){} dispose(){}}
class Attr{constructor(array,itemSize){this.array=array;this.itemSize=itemSize;this.needsUpdate=false;}}
class Mat{constructor(o){Object.assign(this,o||{});} dispose(){}}
class Holder extends Obj{constructor(g,m){super();this.geometry=g;this.material=m;}}
global.THREE={
 WebGLRenderer:class{constructor(){this.domElement=document.createElement('canvas');this._r=1;this.renders=0;}
  setPixelRatio(r){this._r=r;} getPixelRatio(){return this._r;} setSize(w,h){this.domElement.width=w;this.domElement.height=h;}
  setClearColor(){} getClearColor(t){return t;} getClearAlpha(){return 1;} render(){this.renders++;}},
 Scene:class extends Obj{}, Color:class{constructor(c){this.value=c;}},
 PerspectiveCamera:class extends Obj{constructor(f,a,n,fa){super();this.fov=f;this.aspect=a;this.near=n;this.far=fa;}updateProjectionMatrix(){}},
 OrbitControls:class{constructor(){this.target=new V3();this.enableDamping=false;}update(){}},
 HemisphereLight:class extends Obj{}, DirectionalLight:class extends Obj{},
 Plane:class{constructor(n,c){this.normal=n;this.constant=c;}distanceToPoint(){return 1;}},
 BufferGeometry:Geo, SphereGeometry:Geo, BufferAttribute:Attr, Float32BufferAttribute:Attr,
 MeshLambertMaterial:Mat, MeshBasicMaterial:Mat, PointsMaterial:Mat, LineBasicMaterial:Mat,
 Mesh:Holder, Points:Holder, LineSegments:Holder, Line:Holder, LineLoop:Holder, Group:class extends Obj{}, Object3D:Obj,
 Raycaster:class{constructor(){this.params={Points:{threshold:1}};}setFromCamera(){}intersectObject(){return [];}},
 Vector2:class{set(){}}, Vector3:V3, DoubleSide:2};
"""

_DOMLESS_GL = _STUB_DOM + _FAKE_THREE + r"""
global.window=global;
global.localStorage={_d:{},getItem(k){return Object.prototype.hasOwnProperty.call(this._d,k)?this._d[k]:null;},setItem(k,v){this._d[k]=String(v);}};
global.location={hash:'',href:'http://x/report.html',protocol:'http:',origin:'http://x',pathname:'/report.html'};
global.history={replaceState(){}};global.navigator={};global.performance={now:()=>0};global.requestAnimationFrame=()=>0;
const listeners={};global.addEventListener=(n,f)=>{(listeners[n]=listeners[n]||[]).push(f);};
const posted=[];global.parent={postMessage:(m,o)=>posted.push([m,o])};
const errors=[];console.error=(...a)=>errors.push(a.map(String).join(' '));
require(DIR+'/main.js');
const g=id=>document.getElementById(id);
const T=window.__wssTest;
const fire=(name,ev)=>(listeners[name]||[]).forEach(f=>f(ev));
__SCRIPT__
"""


_ONLINE_FETCH = r"""
const calls=[];
global.fetch=(url,init)=>{const u=String(url),o=init||{},h=o.headers||{};
  calls.push({url:u,method:o.method||'GET',token:h['X-CSRF-Token']||null,body:o.body?JSON.parse(o.body):null});
  const ok=b=>Promise.resolve({ok:true,status:200,json:()=>Promise.resolve(b)});
  if(/\/api\/session$/.test(u))return ok({csrf_token:'tok-1'});
  if(/\/snapshots$/.test(u))return ok({snapshots:{items:[{name:'front'},{name:'left'},{name:'top'},{name:'current'}]},version:7});
  return Promise.resolve({ok:false,status:404,json:()=>Promise.resolve(null)});};
"""

_DOMLESS_GL_ONLINE = (_DOMLESS_GL
                      .replace("global.location={hash:'',href:'http://x/report.html',protocol:'http:',origin:'http://x',pathname:'/report.html'};",
                               "global.location={hash:'',href:'http://x/api/jobs/j1/report',protocol:'http:',origin:'http://x',pathname:'/api/jobs/j1/report'};")
                      .replace("require(DIR+'/main.js');", _ONLINE_FETCH + "require(DIR+'/main.js');"))


def _report_parts(tmp_path, meta, parts=None, embed_derived=False):
    from wss_deploy.report import TEMPLATE  # noqa: F401  (main script comes from _main_script)
    _, mesh, cloud, centerline = parts or _minimal_report_inputs()
    out = tmp_path / "report.html"
    trust = np.zeros(len(mesh['vertices']), dtype=np.uint8)
    trust[1] = 2  # one rough vertex so the overlay has something to legend
    build_html(out, meta, {**mesh, 'trust': trust}, cloud, centerline, embed_derived=embed_derived)
    html = out.read_text(encoding="utf-8")
    grab = lambda tag: re.search(r'<script id="%s" type="application/json">(.*?)</script>' % tag, html, re.S).group(1)
    return html, (("meta.json", grab("wss-report-meta")), ("arrays.json", grab("wss-report-arrays")),
                  ("glossary.json", grab("wss-glossary")), ("main.js", _main_script()))


def _domless(tmp_path, meta):
    _, files = _report_parts(tmp_path, meta)
    return _node("require(DIR+'/core.js');require(DIR+'/common.js');" + _DOMLESS, tmp_path, *files)


def _domless_gl(tmp_path, meta, script, parts=None, stub=None, embed_derived=False):
    _, files = _report_parts(tmp_path, meta, parts, embed_derived=embed_derived)
    program = "require(DIR+'/core.js');require(DIR+'/common.js');" + (stub or _DOMLESS_GL).replace("__SCRIPT__", script)
    return _node(program, tmp_path, *files)


def _tube_report_inputs():
    """A straight tube along +z (r = 3 mm) so the centreline-normal section is a real circle."""
    na, nz = 24, 7
    ang = np.arange(na) * 2 * np.pi / na
    zs = np.linspace(0., 30., nz)
    verts = np.array([[3 * np.cos(a), 3 * np.sin(a), z] for z in zs for a in ang], dtype=np.float32)
    faces = []
    for iz in range(nz - 1):
        for ia in range(na):
            a0, a1 = iz * na + ia, iz * na + (ia + 1) % na
            faces += [[a0, a1, a0 + na], [a1, a1 + na, a0 + na]]
    faces = np.array(faces, dtype=np.uint32)
    n = len(verts)
    wss = np.linspace(1., 4., n, dtype=np.float32)
    seg = np.ones(n, dtype=np.int32)
    s = np.ascontiguousarray(verts[:, 2], dtype=np.float32)
    meta = {'case_id': 'tube', 'release': 'rel', 'endpoints': [], 'branch_names': {'1': '主动脉'},
            'per_branch': {'主动脉': {'segment_id': 1, 'n_points': n, 'area_mm2': 100., 'wss_mean_pa': 2., 'wss_p99_pa': 3., 'wss_max_pa': 4., 'frac_low': 0., 'frac_high': 0.}},
            'geometry': {}, 'input_check': {'area_mm2': 100.}, 'cloud': {'spacing_mm': .8}, 'timing_s': {},
            'wss_field_pa': {'p99': 3., 'p99_pa': 3., 'mean': 2., 'median': 2., 'thresholds_pa': [.4, 4., 7.],
                             'area_frac_low': 0., 'area_low_mm2': 0., 'area_frac_high': 0., 'area_high_mm2': 0., 'area_frac_very_high': 0.},
            'peak': {'p99_pa': 3., 'max_pa': 4., 'xyz_mm': [3, 0, 12.5], 'branch': '主动脉', 's_from_inlet_mm': 12.5,
                     'dist_to_junction_mm': 0., 'local_radius_mm': 3., 'top5_clusters_ge20': 0, 'top5_cluster_sizes': []}}
    mesh = {'vertices': verts, 'faces': faces, 'wss': wss, 'segment': seg}
    cloud = {'pts': verts, 'wss': wss, 'segment': seg, 's_from_root_mm': s,
             'theta_rad': np.ascontiguousarray(np.tile(ang, nz), dtype=np.float32),
             'radius_mm': np.full(n, 3., dtype=np.float32), 'dist_to_junction_mm': s}
    centerline = {'xyz': np.array([[0., 0., z] for z in zs], dtype=np.float32),
                  'radius_mm': np.full(nz, 3., dtype=np.float32),
                  'edges': np.array([[i, i + 1] for i in range(nz - 1)], dtype=np.uint32),
                  'segment': np.ones(nz, dtype=np.int32)}
    return meta, mesh, cloud, centerline


def test_domless_page_hides_analysis_panels_for_legacy_summary(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless(tmp_path, meta)
    assert out["errors"] == ["WSS viewer: Error: No WebGL"] and out["fallback"] == 1
    assert out["findings"] == 0 and "未包含发现列表" in out["findingsNote"]
    # The trust array alone is not enough: without meta.trust (legend/rules) the overlay stays disabled.
    assert out["profOptions"] == 0 and out["trustDisabled"] is True and "未含可信区域" in out["trustHint"]
    assert out["thr"] == ["0.4", "4", "7"] and out["branches"] == 1
    # v0.15: an unsigned page says 待审阅 like the workbench (the service writes the review into the page on every decision)
    assert "待审阅" in out["footer"] and "未记录审阅状态" not in out["footer"] and out["panel"] > 1000
    # §19.9: identities and hashes left the footer for the 「技术信息」 dialog.
    assert "技术信息" in out["footer"] and 'data-gloss="run_identity"' not in out["footer"]
    # Menus that need the 3D scene are hidden in the no-WebGL fallback, statistics stay.
    assert out["measureHidden"] is True


def test_domless_page_renders_findings_profiles_trust_and_review(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless(tmp_path, _analysis_meta(meta))
    assert out["findings"] == 1 and "点击定位" in out["findingsNote"]
    assert out["profOptions"] == 1
    assert out["trustDisabled"] is False and out["trustHint"] == ""
    assert "已审阅" in out["footer"] and "reviewer-01" in out["footer"]


def test_viewer_announces_itself_and_installs_the_message_handlers(tmp_path):
    """C13: the batch-export parent waits for wss-view:ready before applying a state."""
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const ready=posted.filter(p=>p[0].type==='wss-view:ready');
      fire('message',{origin:'http://x',data:{type:'wss-view:apply-state',state:{bands:6},request_id:'r1'}});
      setTimeout(()=>console.log(JSON.stringify({errors,ready:ready.map(p=>[p[0].family,p[0].run_identity,p[0].case_id,p[0].webgl,p[1]]),
        messageHandlers:(listeners.message||[]).length,applied:posted.filter(p=>p[0].type==='wss-view:applied').map(p=>p[0].request_id),
        bands:T.state().bands})),20);
    """)
    assert out["errors"] == []
    assert out["ready"] == [["wall", "r" * 64, "legacy", True, "http://x"]]
    # One handler is the §6 camera link, the other the §12.6 protocol from the shared library.
    assert out["messageHandlers"] == 2 and out["applied"] == ["r1"] and out["bands"] == 6


def test_no_webgl_fallback_still_reports_ready_without_a_scene(tmp_path):
    meta, *_ = _minimal_report_inputs()
    _, files = _report_parts(tmp_path, _analysis_meta(meta))
    out = _node("require(DIR+'/core.js');require(DIR+'/common.js');" + _STUB_DOM + r"""
      global.window=global;global.THREE={WebGLRenderer:class{constructor(){throw new Error('No WebGL');}}};
      global.localStorage={getItem:()=>null,setItem:()=>{}};
      global.location={hash:'',href:'http://x/report.html',protocol:'http:',origin:'http://x',pathname:'/report.html'};
      global.history={replaceState(){}};global.navigator={};global.performance={now:()=>0};global.requestAnimationFrame=()=>0;
      global.addEventListener=()=>{};const posted=[];global.parent={postMessage:(m,o)=>posted.push([m,o])};
      console.error=()=>{};require(DIR+'/main.js');
      console.log(JSON.stringify({ready:posted.filter(p=>p[0].type==='wss-view:ready').map(p=>[p[0].family,p[0].webgl])}));
    """, tmp_path, *files)
    assert out["ready"] == [["wall", False]]


def test_branch_visibility_is_display_only_and_lands_in_the_view_state(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const before=T.state();T.hide(1,true);const hidden=T.state();T.hide(1,false);
      console.log(JSON.stringify({errors,before:before.branches_hidden,hidden:hidden.branches_hidden,back:T.state().branches_hidden,
        noteShown:g('branch-vis-note').hidden===false,boxes:g('branch-vis').children.length,
        p99:before.range.max===hidden.range.max}));
    """)
    assert out["errors"] == []
    assert out["before"] == [] and out["hidden"] == [1] and out["back"] == []
    assert out["noteShown"] is True and out["boxes"] == 1
    assert out["p99"] is True  # hiding a branch never touches the statistics


def test_builtin_presets_apply_without_error_and_record_their_name(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const names=T.presets().map(p=>p.name),applied=[];
      for(const n of names){applied.push([n,T.applyPreset(n),T.state().preset_name]);}
      const missing=T.applyPreset('没有这个预设');
      console.log(JSON.stringify({errors,names,applied,missing,note:g('preset-note')._text,builtinButtons:g('preset-builtin').children.length}));
    """)
    assert out["errors"] == []
    assert out["names"] == ["瘤囊低 WSS 区", "髂分叉热点", "主动脉沿程", "临床视图"]
    assert out["applied"] == [[n, True, n] for n in out["names"]]
    assert out["missing"] is False and out["builtinButtons"] == 4


def test_measurement_picks_produce_arc_distance_and_diameter_entries(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      T.mode('distance');T.pick([0,0,0]);const half=T.state().measurements.length;T.pick([1,0,0]);
      T.mode('distance');T.mode('arc');T.pick([0,0,0]);T.pick([1,0,0]);
      T.mode('arc');T.mode('diameter');T.pick([.5,.2,0]);
      console.log(JSON.stringify({errors,half,list:T.state().measurements.map(m=>[m.kind,m.id,+m.value_mm.toFixed(3),m.branch]),
        rows:g('measure-list').children.length,status:g('measure-status')._text,labels:g('labels').children.length}));
    """)
    assert out["errors"] == []
    assert out["half"] == 0  # one pick of a two-point mode is not a measurement yet
    kinds = {m[0]: m for m in out["list"]}
    assert kinds["distance"][2] == pytest.approx(1.0) and kinds["distance"][1] == "D1"
    assert kinds["arc"][2] == pytest.approx(1.0) and kinds["arc"][3] == "主动脉"
    assert kinds["diameter"][2] == pytest.approx(2.0)  # 2 x the 1 mm inscribed radius
    assert out["rows"] == 3 and "管径" in out["status"]


def test_probe_annotation_and_manual_finding_round_trip_through_the_view_state(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      T.pick([1,1,0]);const card=g('probe-card').hidden;
      g('probe-card').children[1].children[0].onclick();        // 「记录」
      T.text('瘤囊后壁');g('annot-add').onclick();T.pick([0,1,0]);
      T.text('可疑龛影');g('finding-add').onclick();T.pick([1,0,0]);
      const s=T.state();
      const restored=(()=>{T.mode('');return T.apply(JSON.parse(JSON.stringify(s)));})();
      console.log(JSON.stringify({errors,card,probes:s.probe_log.map(r=>[r.id,r.branch,r.values.wss_pa]),
        annots:s.annotations.items.map(a=>[a.id,a.text,a.branch]),review:s.findings_review.added.map(a=>[a.id,a.text,a.kind]),
        findings:g('findings-list').children.length,restored,after:T.lists().ANNOTS.length,
        annotStatus:g('annot-status')._text,saveDisabled:g('annot-save').disabled,keys:Object.keys(s).sort()}));
    """)
    assert out["errors"] == []
    assert out["card"] is False
    assert out["probes"] == [["P1", "主动脉", 4.0]]
    assert out["annots"] == [["A1", "瘤囊后壁", "主动脉"]]
    assert out["review"] == [["M1", "可疑龛影", "manual"]]
    assert out["findings"] == 2 and out["restored"] is True and out["after"] == 1
    # Offline single files are read-only towards the service; edits still travel in the view state.
    assert out["saveDisabled"] is True and "离线" in out["annotStatus"]
    for key in ("measurements", "annotations", "probe_log", "branches_hidden", "preset_name", "lang", "export", "findings_review"):
        assert key in out["keys"]


def test_findings_review_decisions_dim_and_reorder_without_a_server(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      T.text('人工项');g('finding-add').onclick();T.pick([1,0,0]);
      const tools=id=>g('findings-list').children[id].children[1];
      tools(0).children[1].onclick();                            // 驳回 the automatic finding
      const order=g('findings-list').children.map(w=>w.children[0]._html.includes('全场最大')?'auto':'manual');
      const rejected=g('findings-list').children.map(w=>w.classList.contains('rejected'));
      console.log(JSON.stringify({errors,order,rejected,review:T.state().findings_review.items,note:g('findings-review-note')._text}));
    """)
    assert out["errors"] == []
    assert out["order"] == ["manual", "auto"]  # rejected items sink to the end
    assert out["rejected"] == [False, True]
    assert out["review"]["F1"]["decision"] == "rejected"
    assert "离线只读" in out["note"]


def test_glossary_is_embedded_and_the_popover_renders_a_term(tmp_path):
    meta, mesh, cloud, centerline = _minimal_report_inputs()
    out_path = tmp_path / "g.html"
    build_html(out_path, meta, mesh, cloud, centerline)
    html = out_path.read_text(encoding="utf-8")
    assert '<script id="wss-glossary" type="application/json">' in html
    assert '"p99"' in html and 'data-gloss="p99"' in html and 'data-gloss="run_identity"' in html
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const shown=T.gloss.show('p99',null),box=T.gloss.element;
      const missing=T.gloss.show('nope',null);
      console.log(JSON.stringify({errors,shown,missing,term:box.children.map(c=>c._text),hidden:box.hidden,
        text:(()=>{T.gloss.hide();return T.gloss.element.hidden;})()}));
    """)
    assert out["errors"] == [] and out["shown"] is True and out["missing"] is True
    assert out["term"][0] == "空间 p99" and "99" in out["term"][1]
    assert out["hidden"] is False and out["text"] is True


def test_view_state_round_trips_the_v11_keys_and_tolerates_their_absence(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      T.mode('distance');T.pick([0,0,0]);T.pick([1,0,0]);T.hide(1,true);
      const s=T.state();s.lang='en';s.preset_name='我的视图';
      T.apply(s);const back=T.state();
      const legacy=T.apply({schema_version:'wss-deploy.view/v1',family:'wall',mode:'wss',colormap:'turbo',bands:4,units:'dyn'});
      console.log(JSON.stringify({errors,schema:back.schema_version,lang:back.lang,preset:back.preset_name,
        meas:back.measurements.length,hidden:back.branches_hidden,exportKeys:Object.keys(back.export).sort(),
        legacy,afterLegacy:{meas:T.state().measurements.length,cmap:T.state().colormap,units:T.state().units}}));
    """)
    assert out["errors"] == []
    assert out["schema"] == "wss-deploy.view/v1" and out["lang"] == "en" and out["preset"] == "我的视图"
    assert out["meas"] == 1 and out["hidden"] == [1]
    assert out["exportKeys"] == ["background", "colorbar", "lang", "scale", "ui"]
    # A v1.0 state without the new keys applies and leaves the existing measurements alone.
    assert out["legacy"] is True and out["afterLegacy"] == {"meas": 1, "cmap": "turbo", "units": "dyn"}


def test_diameter_measurement_cuts_a_real_cross_section_on_a_tube(tmp_path):
    """§15.14: the pick is projected onto the centreline, the tangent becomes the section normal."""
    meta, mesh, cloud, centerline = _tube_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      T.mode('diameter');T.pick([3,0,12.5]);
      const m=T.state().measurements[0];
      console.log(JSON.stringify({errors,m:{kind:m.kind,method:m.method,value:m.value_mm,max:m.max_diameter_mm,min:m.min_diameter_mm,
        eq:m.equivalent_diameter_mm,area:m.area_mm2,branch:m.branch,poly:(m.polygon_world||[]).length,zs:(m.polygon_world||[]).map(q=>q[2])},
        label:m.label,en:T.label(m,'en'),status:g('measure-status')._text,
        redrawn:(()=>{T.apply(JSON.parse(JSON.stringify(T.state())));return T.state().measurements[0].polygon_world.length;})()}));
    """, parts=(meta, mesh, cloud, centerline))
    assert out["errors"] == []
    m = out["m"]
    assert m["method"] == "contour" and m["kind"] == "diameter" and m["branch"] == "主动脉"
    # A 24-gon inscribed in r = 3 mm: max Feret 6.00, min (width) 5.95, area 27.95 mm².
    assert m["max"] == pytest.approx(6.0, abs=.02) and m["min"] == pytest.approx(5.95, abs=.05)
    assert m["area"] == pytest.approx(27.95, abs=.3) and m["eq"] == pytest.approx(5.97, abs=.05)
    assert m["value"] == pytest.approx(m["eq"])
    assert m["poly"] >= 20 and all(z == pytest.approx(12.5, abs=1e-3) for z in m["zs"])
    assert "最大直径" in out["label"] and "面积" in out["label"] and "mm²" in out["label"]
    assert out["en"].startswith("Max diameter") and "Aorta" in out["en"]
    assert out["status"] == out["label"]
    assert out["redrawn"] == m["poly"]  # the polygon travels in the view state and redraws on apply
    from wss_deploy.report import TEMPLATE
    assert "等效直径" in TEMPLATE and "内切半径 × 2 并注明" in TEMPLATE  # the hint explains both branches


def test_diameter_falls_back_to_the_inscribed_radius_without_a_closed_contour(tmp_path):
    """A flat patch has no closed lumen loop: 2 × inscribed radius, labelled as such."""
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      T.mode('diameter');T.pick([.5,.2,0]);
      const m=T.state().measurements[0];
      console.log(JSON.stringify({errors,method:m.method,value:m.value_mm,max:m.max_diameter_mm,eq:m.equivalent_diameter_mm,
        area:m.area_mm2,poly:m.polygon_world===undefined,label:m.label,en:T.label(m,'en')}));
    """)
    assert out["errors"] == [] and out["method"] == "inscribed"
    assert out["value"] == pytest.approx(2.0) and out["max"] == pytest.approx(2.0) and out["eq"] == pytest.approx(2.0)
    assert out["area"] == pytest.approx(np.pi, abs=1e-3) and out["poly"] is True
    assert "内切半径×2" in out["label"] and "2 × inscribed radius" in out["en"]


def test_get_state_answers_the_parent_and_apply_state_leaves_the_camera_alone(tmp_path):
    """§15.6: get-state replies with the full v1.1 state; a display subset never moves the camera."""
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const before=T.camera();
      fire('message',{origin:'http://evil',data:{type:'wss-view:get-state',request_id:'nope'}});
      fire('message',{origin:'http://x',data:{type:'wss-view:get-state',request_id:'q1'}});
      fire('message',{origin:'http://x',data:{type:'wss-view:apply-state',request_id:'s1',
        state:{colormap:'turbo',bands:6,log:true,range:{mode:'fixed',max:12},units:'dyn',thresholds_pa:[.5,5,8],opacity:.6,overlay:{contours:true},mode:'wss'}}});
      setTimeout(()=>{const st=posted.filter(p=>p[0].type==='wss-view:state');
        console.log(JSON.stringify({errors,replies:st.map(p=>[p[0].request_id,p[0].family,p[1],p[0].state.schema_version]),
          keys:st.length?Object.keys(st[0][0].state).sort():[],
          foreign:posted.some(p=>p[0].request_id==='nope'),
          applied:posted.filter(p=>p[0].type==='wss-view:applied').map(p=>p[0].request_id),
          camera:[before,T.camera()],after:{cmap:T.state().colormap,bands:T.state().bands,units:T.state().units,max:T.state().range.max}}));},20);
    """)
    assert out["errors"] == []
    assert out["replies"] == [["q1", "wall", "http://x", "wss-deploy.view/v1"]]
    assert out["foreign"] is False and out["applied"] == ["s1"]
    for key in ("camera", "measurements", "montage", "export", "thresholds_pa"):
        assert key in out["keys"]
    assert out["camera"][0] == out["camera"][1]  # the display subset carried no camera
    assert out["after"] == {"cmap": "turbo", "bands": 6, "units": "dyn", "max": 12}


def test_one_page_figures_post_four_named_snapshots(tmp_path):
    """§15.8: three standard views plus the current one, 2×, white, colour bar composited."""
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      T.snapshots().then(res=>{const post=calls.filter(c=>/\\/snapshots$/.test(c.url));
        console.log(JSON.stringify({errors,version:res&&res.version,n:post.length,
          url:post[0]&&post[0].url,method:post[0]&&post[0].method,token:post[0]&&post[0].token,
          replace:post[0]&&post[0].body.replace,
          images:post[0]?post[0].body.images.map(i=>[i.name,i.view,i.caption,i.width,i.height,typeof i.png_base64,i.png_base64.startsWith('data:')]):[],
          note:g('snap-note').children.map(c=>[c.textContent,c.href||'']),hidden:g('snap-onepage').hidden}));},
        err=>console.log(JSON.stringify({errors,fail:String(err)})));
    """, stub=_DOMLESS_GL_ONLINE)
    assert out["errors"] == [] and out["version"] == 7
    assert out["n"] == 1 and out["url"] == "http://x/api/jobs/j1/snapshots" and out["method"] == "POST"
    assert out["token"] == "tok-1" and out["replace"] is True
    assert [i[0] for i in out["images"]] == ["front", "left", "top", "current"]
    assert [i[1] for i in out["images"]] == ["front", "left", "top", "current"]
    assert out["images"][0][2] == "前视 · WSS · Pa" and out["images"][3][2] == "当前视角 · WSS · Pa"
    assert all(i[3] == 1600 and i[4] == 1200 and i[5] == "string" and i[6] is False for i in out["images"])
    assert out["note"] == [["已生成 4 张 · ", ""], ["打开一页纸", "onepage"]] and out["hidden"] is False


def test_one_page_button_is_hidden_on_an_offline_single_file(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      T.snapshots().then(r=>console.log(JSON.stringify({errors,r,hidden:g('snap-onepage').hidden,note:g('snap-note')._text})));
    """)
    assert out["errors"] == [] and out["r"] is None and out["hidden"] is True and "离线" in out["note"]


def test_multi_view_montage_uses_the_export_settings_and_fails_soft_without_canvas(tmp_path):
    """§15.11: the default selection lands in the view state; node has no Image, so it must not throw."""
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const before=T.camera(),boxes=g('montage-views').children.length;
      T.montage().then(ok=>{
        const s=T.state();s.montage={views:['front','current'],columns:4};T.apply(s);
        console.log(JSON.stringify({errors,ok,boxes,state:T.state().montage,note:g('montage-note')._text,
          camera:[before,T.camera()],cols:g('montage-cols').value}));});
    """)
    assert out["errors"] == [] and out["ok"] is False
    assert out["boxes"] == 7
    assert out["state"] == {"views": ["front", "current"], "columns": 4} and out["cols"] == "4"
    assert "失败" in out["note"] and out["camera"][0] == out["camera"][1]


def test_profile_curves_export_two_svgs_with_thresholds(tmp_path):
    """§15.12: one SVG per physical quantity (WSS and radius) for the current branch."""
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const list=T.profileSVGs();g('prof-svg').onclick();
      console.log(JSON.stringify({errors,keys:list.map(x=>x.key),names:list.map(x=>x.filename),
        wss:{branch:list[0].svg.includes('主动脉'),path:list[0].svg.includes('<path'),svg:list[0].svg.startsWith('<svg'),
             mean:list[0].svg.includes('均值'),p99:list[0].svg.includes('p99'),min:list[0].svg.includes('最低'),
             thr:list[0].svg.includes('0.4 Pa')&&list[0].svg.includes('7.0 Pa'),x:list[0].svg.includes('距入口弧长 (mm)')},
        radius:{path:list[1].svg.includes('<path'),label:list[1].svg.includes('半径')},note:g('prof-note')._text}));
    """)
    assert out["errors"] == []
    assert out["keys"] == ["wss", "radius"]
    assert out["names"] == ["legacy_profile_主动脉_wss.svg", "legacy_profile_主动脉_radius.svg"]
    assert all(out["wss"].values()), out["wss"]
    assert all(out["radius"].values())
    assert "已导出" in out["note"]


def test_wall_template_has_compact_controls():
    from wss_deploy.report import TEMPLATE
    for marker in ('id="menu-toggle"', 'id="menu-tab"', 'id="menu-close"', 'id="layout-toggle"', 'class="menu-head"', '.compact aside.menu', "wss-report-compact"):
        assert marker in TEMPLATE, marker


# ---------------------------------------------------------------------------
# §17.3 automatic labels, maximum-diameter marker and §17.2 narrative block
# ---------------------------------------------------------------------------

_LABEL_HELPERS = """
      const cls=c=>g('labels').children.filter(d=>d.classList.contains(c));
      const texts=c=>cls(c).map(d=>d.textContent);
"""


def test_auto_labels_pin_findings_and_branch_names_and_round_trip(tmp_path):
    """labels:{findings:2, branches:true} pins two finding chips plus one per visible branch."""
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _labels_meta(meta), _LABEL_HELPERS + """
      const start=T.state().labels;
      T.apply(Object.assign(T.state(),{labels:{findings:2,branches:true}}));
      const on={labels:T.state().labels,f:texts('flabel'),b:texts('blabel'),
        sev:cls('flabel').map(d=>d.className),kinds:T.autoLabels().map(c=>c.kind)};
      const clicked=(()=>{cls('flabel')[0].onclick();return T.state().highlight.branch;})();
      const legacy=T.apply({schema_version:'wss-deploy.view/v1',family:'wall',mode:'wss',colormap:'turbo'});
      const kept=T.state().labels;
      const hidden=(()=>{T.hide(1,true);const n=[cls('flabel').length,cls('blabel').length];T.hide(1,false);return n;})();
      T.apply(Object.assign(T.state(),{labels:{findings:0,branches:false}}));
      console.log(JSON.stringify({errors,start,on,clicked,legacy,kept,hidden,
        off:[cls('flabel').length,cls('blabel').length],select:g('lbl-findings').value,box:g('lbl-branches').checked}));
    """)
    assert out["errors"] == []
    # Default is off; the key is always present in the captured state.
    assert out["start"] == {"findings": 0, "branches": False, "max_diameter": True}
    assert out["on"]["labels"] == {"findings": 2, "branches": True, "max_diameter": True}
    assert out["on"]["f"] == ["F1 最大 WSS 4.00 Pa", "F2 高 WSS 21.9 Pa"]  # top two by rank; §19.1 formatValue
    assert out["on"]["b"] == ["主动脉"]  # one centreline group in the minimal report
    assert out["on"]["sev"] == ["flabel note", "flabel attention"]
    assert out["on"]["kinds"] == ["flabel", "flabel", "blabel", "maxd"]
    assert out["clicked"] == -1  # clicking a label activates the finding (no branch filter change)
    # A v1.0 state without the key keeps the labels that are showing.
    assert out["legacy"] is True and out["kept"] == out["on"]["labels"]
    assert out["hidden"] == [0, 0]  # C10: hiding the branch hides its labels
    assert out["off"] == [0, 0] and out["select"] == "0" and out["box"] is False


def test_auto_labels_skip_findings_the_reviewer_rejected(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _labels_meta(meta), _LABEL_HELPERS + """
      T.apply(Object.assign(T.state(),{labels:{findings:3,branches:false}}));
      const before=texts('flabel');
      const row=g('findings-list').children.find(w=>w.children[0]._html.includes('全场最大'));
      row.children[1].children[1].onclick();                       // 驳回 F1
      const after=texts('flabel');
      console.log(JSON.stringify({errors,before,after,decision:T.state().findings_review.items.F1.decision}));
    """)
    assert out["errors"] == []
    assert len(out["before"]) == 3 and out["before"][0].startswith("F1 ")
    assert out["decision"] == "rejected"
    assert [t.split(" ")[0] for t in out["after"]] == ["F2", "F3"]


def test_clinical_view_preset_turns_the_labels_on(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _labels_meta(meta), _LABEL_HELPERS + """
      const ok=T.applyPreset('临床视图'),s=T.state();
      console.log(JSON.stringify({errors,ok,name:s.preset_name,labels:s.labels,mode:s.mode,cmap:s.colormap,
        thr:s.thresholds_pa,f:cls('flabel').length,b:cls('blabel').length,select:g('lbl-findings').value,box:g('lbl-branches').checked}));
    """)
    assert out["errors"] == [] and out["ok"] is True and out["name"] == "临床视图"
    assert out["labels"] == {"findings": 5, "branches": True, "max_diameter": True}
    assert out["mode"] == "wss" and out["cmap"] == "rainbow" and out["thr"] == [.4, 4, 7]
    assert out["f"] == 3 and out["b"] == 1  # only three findings exist
    assert out["select"] == "5" and out["box"] is True


def test_max_diameter_marker_row_ring_and_profile_series(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _labels_meta(meta), _LABEL_HELPERS + """
      const row=g('maxd-text')._text,hidden=[g('maxd-row').hidden,g('maxd-toggle').hidden];
      const chip=T.autoLabels().find(c=>c.kind==='maxd'),label=texts('maxd');
      g('maxd-fly').onclick();const tip=g('tip-text')._text;   // flyTo tweens the camera; the readout proves the jump
      const svg=T.profileSVGs();
      const offRing=(()=>{g('maxd-ring').checked=false;g('maxd-ring').onchange({target:g('maxd-ring')});
        return [T.state().labels.max_diameter,cls('maxd').length,T.autoLabels().length];})();
      console.log(JSON.stringify({errors,row,hidden,chip,label,
        tip,offRing,
        svg:{keys:svg.map(x=>x.key),radius:svg[1].svg,names:svg.map(x=>x.filename)}}));
    """)
    assert out["errors"] == []
    assert out["hidden"] == [False, False]
    assert "最大直径 63.4 mm" in out["row"] and "等效 58.9 mm" in out["row"] and "入口下 120 mm" in out["row"]
    assert out["chip"] == {"kind": "maxd", "text": "最大直径 63.4 mm", "ring": 4}
    assert out["label"] == ["最大直径 63.4 mm"]
    assert out["tip"] == "最大直径站 63.4 mm · 距入口 120 mm"  # 「飞到」 ran and reported the station
    assert out["svg"]["keys"] == ["wss", "radius"] and out["svg"]["names"][1].endswith("_radius.svg")
    assert "最大直径" in out["svg"]["radius"] and "等效直径" in out["svg"]["radius"] and "半径" in out["svg"]["radius"]
    assert out["offRing"] == [False, 0, 0]  # the checkbox is part of the view state


def test_narrative_block_renders_and_stays_away_for_legacy_meta(tmp_path):
    meta, *_ = _minimal_report_inputs()
    script = """
      const panel=T.panel();
      console.log(JSON.stringify({errors,card:panel.includes('id="narrative-card"'),head:panel.includes('结论（参考）'),
        gloss:panel.includes('data-gloss="narrative"'),edited:panel.includes('审阅人已修改'),
        first:panel.indexOf('结论（参考）'),fields:panel.indexOf('结果字段与模型'),
        sentence:panel.includes('主动脉最大直径 63.4 mm，位于入口下 120 mm。')}));
    """
    auto = _domless_gl(tmp_path, _labels_meta(meta), script)
    assert auto["errors"] == [] and auto["card"] is True and auto["head"] is True
    assert auto["gloss"] is True and auto["edited"] is False and auto["sentence"] is True
    assert 0 <= auto["first"] < auto["fields"]  # the block sits at the top of 统计与口径

    edited_meta = _labels_meta(meta)
    edited_meta["narrative"] = {**edited_meta["narrative"], "edited": "术者复核：瘤体最大 63 mm。",
                                "edited_by": "reviewer-01", "edited_at": "2026-09-22T10:00:00+08:00"}
    edited = _domless_gl(tmp_path, edited_meta, script)
    assert edited["errors"] == [] and edited["edited"] is True and edited["sentence"] is False

    legacy = _domless_gl(tmp_path, _analysis_meta(meta), script)
    assert legacy["errors"] == [] and legacy["card"] is False and legacy["head"] is False
    assert legacy["first"] == -1


def test_auto_labels_are_composited_into_exports_like_annotation_chips():
    from wss_deploy.report import TEMPLATE
    # The export compositor draws the same chips it pins in 3D (AUTO_CHIPS) plus the diameter ring.
    # §21.4: the chips are laid out again at the export resolution (same priorities) with leaders for moved chips.
    assert "for(const c of AUTO_CHIPS)if(Array.isArray(c.ring)" in TEMPLATE
    assert "lay=exportLabelLayout(W,H,k,t=>({w:ctx.measureText(t).width+2*cpad,h:chh}))" in TEMPLATE
    assert "leaderSegment(L.x,L.y,L.w,L.h,L.ax,L.ay)" in TEMPLATE
    assert "composeExport(r.dataURL,r.width,r.height,Object.assign({},o,{colorbar:'none'}),lang)" in TEMPLATE
    for marker in ('id="lbl-findings"', 'id="lbl-branches"', 'id="maxd-row"', 'id="maxd-ring"', 'id="maxd-fly"', 'data-gloss="narrative"'):
        assert marker in TEMPLATE, marker


# ---------------------------------------------------------------------------
# v0.12 (§19): fitted default view, rainbow default, slim footer + 技术信息, human frame text, M1 field
# switch (segmented control / F key), stagnation overlay, cycle finding kinds and profiles, warning
# banner, closable hint and keyboard shortcuts.  All driven through the stub DOM like the tests above.
# ---------------------------------------------------------------------------

def _m1_parts(profiles_cycle=True):
    """The minimal report as a three-head (M1) release: TAWSS / OSI arrays, descriptors and summary.cycle.

    Vertex / point 0 is the only stagnant one (TAWSS 0.2 < 0.4 and OSI 0.25 > 0.1).
    """
    from wss_deploy import cycle_fields as CF
    meta, mesh, cloud, centerline = _minimal_report_inputs()
    tawss = np.array([.2, .5, 1.5, 3.], dtype=np.float32)
    osi = np.array([.25, .05, .3, .12], dtype=np.float32)
    extra = {"tawss_pa": tawss, "osi": osi}
    base = _analysis_meta(meta)
    profiles = json.loads(json.dumps(base["profiles"]))
    if profiles_cycle:
        profiles["branches"][0].update(tawss={"mean_pa": [.35, 2.2], "min_pa": [.2, 1.5]}, osi={"mean": [.15, .2], "p90": [.25, .3]})
    m1 = {**base, "profiles": profiles,
          "fields": {"wss": {"id": "wss", "label": "壁面切应力", "units": "Pa", "location": "wall", "kind": "scalar", "components": 1,
                             "array_key": "wss_pa", "axis_order": ["point"], "time_indices": [0]},
                     "tawss": CF.descriptor("tawss", tawss), "osi": CF.descriptor("osi", osi)},
          "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21, "label": "peak_systole"}],
          "cycle": CF.cycle_block({"tawss": {"values": tawss}, "osi": {"values": osi}}, np.ones(4, dtype=np.int32), {1: "主动脉"}, 10.),
          "model_release": {"registry_id": "M1_3head_3seed_20260922", "name": "M1_3head_3seed_20260922", "model_family": "M1_3head", "models": [1, 2, 3]},
          "feature_contract": {"version": "wss-features/1.0", "source_hash": "f" * 64},
          "created_at": "2026-09-22T17:37:53+08:00", "audit": {"report_rebuilt_at": "2026-09-23T01:02:03+08:00"}}
    return m1, ({**mesh, "extra": extra}, {**cloud, "extra": extra}, centerline)


def _m1_gl(tmp_path, script, meta=None, stub=None, profiles_cycle=True):
    m1, (mesh, cloud, centerline) = _m1_parts(profiles_cycle)
    return _domless_gl(tmp_path, meta or m1, script, parts=(meta or m1, mesh, cloud, centerline), stub=stub)


_FIT = """
      const A=JSON.parse(fs.readFileSync(DIR+'/arrays.json','utf8')),B=Buffer.from(A.mv,'base64');
      const MVx=new Float32Array(B.buffer.slice(B.byteOffset,B.byteOffset+B.byteLength));
      const fit=(dir,up,aspect)=>WssReportCommon.fitView(MVx,{dir,up,fov:35,aspect,margin:1.12});
      const close=(a,b)=>a.every((v,k)=>Math.abs(v-b[k])<1e-6);
      const same=(cam,f)=>close(cam.position,f.position)&&close(cam.target,f.target);
"""


def test_default_view_is_the_fitted_anatomical_front_view_and_refits_on_resize(tmp_path):
    """U1: initial view = standardViews(R).front direction + fitView; reset / 0 / 1–6 re-fit; resize keeps it filled."""
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), _FIT + """
      const start=T.camera(),front=fit([0,1,0],[0,0,1],800/600),upStart=T.controlsUp();
      g('view').clientWidth=400;T.resize();const narrow=T.camera(),frontNarrow=fit([0,1,0],[0,0,1],400/600);
      key('5');T.settle();const upTop=T.controlsUp();
      key('3');T.settle();const left=T.camera(),fitLeft=fit([-1,0,0],[0,0,1],400/600),leftName=T.fitName();
      T.apply(Object.assign(T.state(),{camera:{position:[5,5,5],target:[0,0,0],up:[0,0,1]}}));const manualName=T.fitName();
      g('view').clientWidth=800;T.resize();const kept=T.camera();
      g('reset-camera').onclick();T.settle();const reset=T.camera();
      console.log(JSON.stringify({errors,startOk:same(start,front),narrowOk:same(narrow,frontNarrow),leftOk:same(left,fitLeft),leftName,manualName,upStart,upTop,
        kept:kept.position,resetOk:same(reset,front),fitName:T.fitName(),hint:g('std-note')._text,
        dist:[Math.hypot(...start.position.map((v,k)=>v-start.target[k])),front.distance]}));
    """)
    assert out["errors"] == []
    assert out["startOk"] and out["narrowOk"] and out["leftOk"] and out["resetOk"]
    assert out["leftName"] == "left" and out["manualName"] is None and out["fitName"] == "front"
    assert out["kept"] == [5, 5, 5]  # a hand-placed camera is not re-fitted on resize
    assert out["dist"][0] == pytest.approx(out["dist"][1])
    assert "撑满视口" in out["hint"]
    # OrbitControls is rebuilt about the anatomical up axis (head up in front view, posterior up in the top view).
    assert out["upStart"] == [0, 0, 1] and out["upTop"] == [0, -1, 0]


def test_default_view_falls_back_to_world_z_without_a_frame_and_a_link_camera_wins(tmp_path):
    meta, *_ = _minimal_report_inputs()
    legacy = {k: v for k, v in _analysis_meta(meta).items() if k != "frame_transform"}
    out = _domless_gl(tmp_path, legacy, _FIT + """
      console.log(JSON.stringify({errors,ok:same(T.camera(),fit([0,0,-1],[0,1,0],800/600)),stdDisabled:true}));
    """)
    assert out["errors"] == [] and out["ok"] is True
    # An explicit #view= camera is applied after the default and stays (no fit on later resizes).
    hashed = _DOMLESS_GL.replace("global.location={hash:''", "global.location={hash:core.viewHash({camera:{position:[9,8,7],target:[1,1,0],up:[0,0,1]}})")
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const a=T.camera();T.resize();
      console.log(JSON.stringify({errors,a,b:T.camera(),fit:T.fitName(),note:g('view-note')._text}));
    """, stub=hashed)
    assert out["errors"] == [] and out["a"]["position"] == [9, 8, 7] and out["b"]["position"] == [9, 8, 7]
    assert out["fit"] is None and "链接" in out["note"]


def test_colormap_defaults_to_rainbow_but_a_saved_choice_wins(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), "console.log(JSON.stringify({errors,cmap:T.state().colormap,sel:g('cmap').value}));")
    assert out == {"errors": [], "cmap": "rainbow", "sel": "rainbow"}
    saved = _DOMLESS_GL.replace("global.localStorage={_d:{},", "global.localStorage={_d:{'wss-report-cmap':'viridis'},")
    out = _domless_gl(tmp_path, _analysis_meta(meta), "console.log(JSON.stringify({errors,cmap:T.state().colormap}));", stub=saved)
    assert out == {"errors": [], "cmap": "viridis"}
    from wss_deploy.report import TEMPLATE
    assert '<option value="rainbow">彩虹（默认）</option><option value="viridis">Viridis</option>' in TEMPLATE


def test_term_buttons_are_pinned_to_a_circle():
    """U2: the generic button rules (min-height 34 px, padding, radius 8 px) stretched 「?」 into a tall ellipse."""
    from wss_deploy.report import TEMPLATE
    rule = re.search(r"\.gloss,button\.gloss\{([^}]*)\}", TEMPLATE).group(1)
    for part in ("width:18px!important", "height:18px!important", "min-height:0!important", "padding:0!important", "border-radius:50%!important"):
        assert part in rule, part


def test_slim_footer_and_technical_information_dialog(tmp_path):
    """U2 / U3: footer = review · short release · local time; ids, hashes and the raw frame live in 技术信息."""
    out = _m1_gl(tmp_path, """
      const footer=g('footer')._html,pop=g('tech-pop');pop.hidden=true;const before=pop.hidden;   // the stub does not parse the hidden attribute
      g('tech-btn').onclick();const html=pop._html,open=pop.hidden;
      g('tech-close').onclick();const closed=pop.hidden;
      g('tech-btn').onclick();key('Escape');const esc=pop.hidden;
      console.log(JSON.stringify({errors,footer,before,open,closed,esc,html,
        frame:[g('frame-select').children.map(o=>o.textContent),g('frame-status')._text],subtitle:g('subtitle')._text}));
    """)
    assert out["errors"] == []
    f = out["footer"]
    assert "已审阅" in f and "reviewer-01" in f and "<b title=\"M1_3head_3seed_20260922\">M1_3head</b>" in f
    assert "技术信息" in f and "r" * 12 not in f and "f" * 12 not in f and 'data-gloss="feature_contract"' not in f and "T17:37" not in f
    assert out["before"] is True and out["open"] is False and out["closed"] is True and out["esc"] is True
    h = out["html"]
    for part in ("r" * 64, "f" * 64, "wss-features/1.0", "peak_systole", "step 1162", "2026-09-23T01:02:03+08:00", "M1_3head · 3 个模型",
                 'data-gloss="run_identity"', "0-79 of 81"):
        assert part in h, part
    # The time-frame control speaks human: no raw label or solver step outside the dialog.
    assert out["frame"][0] == ["收缩期峰值帧（约 0.21 s）"] and "peak_systole" not in out["frame"][1] and "1162" not in out["frame"][1]
    # v0.13: an M1 page written before RRT / ECAP existed derives them in the viewer from its TAWSS / OSI arrays.
    assert out["subtitle"] == "legacy · M1_3head · WSS + TAWSS + OSI + RRT + ECAP"


def test_field_segmented_control_menu_tabs_and_f_key_share_one_switch(tmp_path):
    """§19.9: 峰值 WSS / TAWSS / OSI above the viewport, synced with the menu tabs; F cycles; the view state carries it."""
    out = _m1_gl(tmp_path, """
      const seg=g('field-seg'),tabs=()=>g('field-tabs').children.filter(t=>t.classList.contains('active')).map(t=>t.dataset.field);
      const segOn=()=>seg.children.filter(b=>b.className==='on').map(b=>b.dataset.field);
      const snap=()=>({field:T.field(),seg:segOn(),tabs:tabs(),cbar:g('cbar-unit')._text,frame:g('frame-select').children.map(o=>o.textContent),
        colors:T.colors(),panelFirst:T.panel().indexOf('id="cycle-card"')<T.panel().indexOf('结果字段与模型')});
      const s0=snap(),labels=seg.children.map(b=>b.textContent),hidden=seg.hidden,hasSeg=g('view').classList.contains('has-seg');
      seg.children[1].onclick();const s1=snap();
      const e=key('f');const s2=snap();key('F');const s3=snap();
      g('field-tabs').children.find(t=>t.dataset.field==='osi').onclick();const s4=snap();
      const st=T.state();T.apply(Object.assign(T.state(),{field:'tawss'}));const s5=T.field();T.apply(Object.assign(T.state(),{field:'velocity'}));
      console.log(JSON.stringify({errors,labels,hidden,hasSeg,s0,s1,s2,s3,s4,state:st.field,s5,s6:T.field(),prevented:e.defaultPrevented,
        help:T.shortcuts().bindings.map(b=>b.label)}));
    """)
    assert out["errors"] == []
    assert out["labels"] == ["峰值 WSS", "TAWSS", "OSI", "RRT", "ECAP"] and out["hidden"] is False and out["hasSeg"] is True
    s0, s1, s2, s3, s4 = (out[k] for k in ("s0", "s1", "s2", "s3", "s4"))
    assert (s0["field"], s0["seg"], s0["tabs"], s0["cbar"]) == ("wss", ["wss"], ["wss"], "WSS · Pa")
    assert s0["frame"] == ["收缩期峰值帧（约 0.21 s）"] and s0["panelFirst"] is False
    assert (s1["field"], s1["seg"], s1["tabs"], s1["cbar"]) == ("tawss", ["tawss"], ["tawss"], "TAWSS · Pa")
    assert s1["frame"] == ["单周期积分量（0.8 s）"] and s1["panelFirst"] is True and s1["colors"] != s0["colors"]
    assert (s2["field"], s2["cbar"]) == ("osi", "OSI") and out["prevented"] is True   # F: TAWSS → OSI
    assert (s3["field"], s3["cbar"]) == ("rrt", "RRT · Pa⁻¹")                      # v0.13: OSI → RRT (derived)
    assert (s4["field"], s4["seg"]) == ("osi", ["osi"])                             # menu tab drives the same switch
    assert out["state"] == "osi" and out["s5"] == "tawss" and out["s6"] == "tawss"  # unknown field ignored
    assert any(label.startswith("循环字段") for label in out["help"])


def test_single_field_reports_have_no_segmented_control_and_no_f_key(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const e=key('f');
      console.log(JSON.stringify({errors,hidden:g('field-seg').hidden,n:g('field-seg').children.length,field:T.field(),prevented:e.defaultPrevented,
        stagRow:g('stag-row').hidden,help:T.shortcuts().bindings.map(b=>b.keys.join('/'))}));
    """)
    assert out["errors"] == [] and out["hidden"] is True and out["n"] == 0 and out["field"] == "wss"
    assert out["prevented"] is False and out["stagRow"] is True
    assert out["help"] == ["1", "2", "3", "4", "5", "6", "0/r", "l", "p", "s", "?/shift+/"]


def test_stagnation_overlay_hatches_stagnant_vertices_under_any_field(tmp_path):
    """§19.2: TAWSS < 0.4 ∧ OSI > 0.1 vertices get the magenta / pale hatch; legend row and view-state key follow."""
    out = _m1_gl(tmp_path, """
      const c0=T.colors(),row=g('stag-row').hidden;
      g('stag').checked=true;g('stag').onchange({target:g('stag')});
      const c1=T.colors(),legend=g('legend-note')._html,s=T.stag(),state=T.state().overlay;
      const panel=T.panel();key('f');const c2=T.colors(),stillOn=T.stag().on;
      T.apply(Object.assign(T.state(),{overlay:{trust:false,contours:false,stagnation:false}}));
      console.log(JSON.stringify({errors,row,mask:s.mask,on:s.on,state,legend,off:T.stag().on,stillOn,
        v0:[c0.slice(0,3),c1.slice(0,3)],v1:[c0.slice(3,6),c1.slice(3,6)],c2:c2.slice(0,3),panel}));
    """)
    assert out["errors"] == [] and out["row"] is False
    assert out["mask"] == [1, 0, 0, 0] and out["on"] is True and out["state"]["stagnation"] is True
    assert out["v0"][0] != out["v0"][1] and out["v1"][0] == out["v1"][1]   # only the stagnant vertex changes
    assert "滞留区" in out["legend"] and "sw hatch" in out["legend"]
    assert out["stillOn"] is True and out["off"] is False
    # 统计：the cycle card reads summary.cycle — mean, p99 and the threshold area fractions of the active field.
    assert 'id="cycle-card"' in out["panel"] and "滞留区" in out["panel"] and "TAWSS 均值" in out["panel"]


def test_cycle_card_uses_the_active_field_thresholds(tmp_path):
    out = _m1_gl(tmp_path, """
      g('field-seg').children[1].onclick();const t=T.panel();key('f');const o=T.panel();
      console.log(JSON.stringify({errors,t,o}));
    """)
    assert out["errors"] == []
    t, o = out["t"], out["o"]
    for part in ("TAWSS · 单周期积分量（0.8 s）", "低 TAWSS &lt; 0.4 Pa", "&gt; 4 Pa", "&gt; 7 Pa", "p99", "主动脉"):
        assert part in t, part
    for part in ("OSI · 单周期积分量（0.8 s）", "OSI &gt; 0.1", "OSI &gt; 0.2", "OSI &gt; 0.3"):
        assert part in o, part


def test_cycle_finding_kinds_render_with_units_labels_and_fly_to(tmp_path):
    """W1's stagnation_cluster (cm²) / high_osi_cluster (dimensionless) and an unknown kind all render."""
    m1, _ = _m1_parts()
    items = list(m1["findings"]["items"]) + [
        {"id": "F2", "kind": "stagnation_cluster", "label": "主动脉滞留区", "branch": "主动脉", "segment_id": 1, "value": 1.55, "units": "cm²",
         "xyz_mm": [0, 0, 0], "s_from_root_mm": 0., "extent_mm": 3., "area_mm2": 155., "n_points": 1, "rank": 2, "severity": "attention",
         "tawss_mean_pa": .21, "osi_mean": .19, "definition": "TAWSS < 0.4 Pa 且 OSI > 0.1 的连通簇", "point_indices": [0]},
        {"id": "F3", "kind": "high_osi_cluster", "label": "主动脉高 OSI 区", "branch": "主动脉", "segment_id": 1, "value": .394, "units": "1",
         "xyz_mm": [0, 1, 0], "s_from_root_mm": 1., "extent_mm": 3., "area_mm2": 12., "n_points": 1, "rank": 3, "severity": "info",
         "osi_mean": .33, "definition": "OSI > 0.3 的连通簇", "point_indices": [2]},
        {"id": "F4", "kind": "mystery_kind", "value": 2, "units": "", "xyz_mm": [1, 0, 0], "rank": 4, "severity": "weird"}]
    m1 = {**m1, "findings": {**m1["findings"], "items": items}}
    out = _m1_gl(tmp_path, _LABEL_HELPERS + """
      const rows=g('findings-list').children.map(w=>w.children[0]._html);
      T.apply(Object.assign(T.state(),{labels:{findings:10,branches:false}}));const chips=texts('flabel');
      window.__selectFinding(1);const tip=g('tip-text')._text;
      console.log(JSON.stringify({errors,rows,chips,tip}));
    """, meta=m1)
    assert out["errors"] == []
    stag, osi, odd = out["rows"][1], out["rows"][2], out["rows"][3]
    assert "1.55 cm²" in stag and "155" not in stag.replace("1.55", "") and "TAWSS 均值 0.210 Pa" in stag and "OSI 均值 0.190" in stag and "关注" in stag
    assert "最大 OSI 0.394" in osi and "0.394 1" not in osi and "参考" in osi and "几何" not in osi
    assert "mystery_kind" in odd
    assert out["chips"][:3] == ["F1 最大 WSS 4.00 Pa", "F2 滞留区 1.55 cm²", "F3 高 OSI 0.394"]
    assert out["tip"].startswith("主动脉滞留区 · 1.55 cm²")


def test_profiles_follow_the_cycle_field_and_say_when_it_is_missing(tmp_path):
    script = """
      g('menu-profiles').open=true;const read=()=>{window.__drawProfiles();return {legend:g('prof-legend')._html,svg:T.profileSVGs().map(x=>[x.key,x.filename,x.svg])};};
      const w=read();g('field-seg').children[1].onclick();const t=read();key('f');const o=read();
      console.log(JSON.stringify({errors,w,t,o}));
    """
    out = _m1_gl(tmp_path, script)
    assert out["errors"] == []
    assert out["w"]["svg"][0][0] == "wss" and "p99" in out["w"]["legend"]
    tk, tf, tsvg = out["t"]["svg"][0]
    assert tk == "tawss" and tf.endswith("_tawss.svg") and "TAWSS" in tsvg and "0.4 Pa" in tsvg and "低 TAWSS 阈值" in out["t"]["legend"]
    ok, of, osvg = out["o"]["svg"][0]
    assert ok == "osi" and "OSI 0.1" in osvg and "p90" in osvg and "灰虚线：OSI 0.1" in out["o"]["legend"]
    missing = _m1_gl(tmp_path, script, profiles_cycle=False)
    assert missing["errors"] == []
    assert missing["t"]["svg"][0][0] == "wss" and "不含 TAWSS" in missing["t"]["legend"]


def test_warning_banner_for_geometry_out_of_range_and_non_good_quality(tmp_path):
    """B4: one closable line on top of the viewport; 「详情」 opens 统计与口径 at the reference card."""
    meta, *_ = _minimal_report_inputs()
    base = _analysis_meta(meta)
    bad = {**base, "reference_assessment": {"status": "review", "checks": [
               {"path": "geometry.主动脉.length_mm", "units": "mm", "value": 401., "min": 120., "max": 380., "status": "review"},
               {"path": "geometry.左髂总.radius_median_mm", "units": "mm", "value": 3.1, "min": 3.5, "max": 9., "status": "review"},
               {"path": "cloud.variation", "status": "pass"}], "reasons": ["部分几何测量超出当前发布包声明的参考范围，请复核。"],
               "population": {"status": "pass", "percentile": 40.}},
           "quality": {"level": "review", "label": "存在不确定性，建议复核", "reasons": []}}
    out = _domless_gl(tmp_path, bad, """
      const b=g('warn-banner'),shown=!b.hidden,text=b.children[0].textContent,cls=g('view').classList.contains('has-banner');
      b.children[1].onclick();const open=g('menu-stats').open;
      b.children[2].onclick();
      console.log(JSON.stringify({errors,shown,text,cls,open,after:[g('warn-banner').hidden,g('view').classList.contains('has-banner')],panel:T.panel()}));
    """)
    assert out["errors"] == [] and out["shown"] is True and out["cls"] is True and out["open"] is True
    assert "2 项超出发布包参考范围（主动脉长度 401 mm，参考 120–380 mm 等）" in out["text"] and "存在不确定性" in out["text"]
    assert out["text"].startswith("注意：") and out["text"].endswith("预测可信度可能下降，请结合详情复核。")
    assert out["after"] == [True, False]
    assert 'id="ref-card"' in out["panel"] and 'id="quality-card"' in out["panel"]
    clean = _domless_gl(tmp_path, base, "console.log(JSON.stringify({errors,hidden:g('warn-banner').hidden}));")
    assert clean == {"errors": [], "hidden": True}


def test_hint_is_short_closable_and_remembered_while_readouts_still_show(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _labels_meta(meta), """
      const first=[g('tip-text')._text,g('tip').hidden];
      g('tip-close').onclick();const closed=[g('tip').hidden,localStorage.getItem('wss-report-tip-off')];
      g('maxd-fly').onclick();const data=[g('tip').hidden,g('tip-text')._text];
      g('tab-stl').onclick&&0;
      console.log(JSON.stringify({errors,first,closed,data}));
    """)
    assert out["errors"] == []
    assert out["first"][1] is False and len(out["first"][0]) <= 30 and "Gaussian" not in out["first"][0]
    assert out["closed"] == [True, "1"]
    assert out["data"] == [False, "最大直径站 63.4 mm · 距入口 120 mm"]  # real readouts still appear
    remembered = _DOMLESS_GL.replace("global.localStorage={_d:{},", "global.localStorage={_d:{'wss-report-tip-off':'1'},")
    out = _domless_gl(tmp_path, _labels_meta(meta), "console.log(JSON.stringify({errors,hidden:g('tip').hidden}));", stub=remembered)
    assert out == {"errors": [], "hidden": True}
    from wss_deploy.report import TEMPLATE
    assert "■ 灰色：插值未覆盖" not in TEMPLATE and 'class="sw hatch"' in TEMPLATE  # compact swatch legend


def test_keyboard_shortcuts_views_labels_probe_help_and_typing_guard(tmp_path):
    """A9: 1–6 / 0 / R, L, P, S and ? through installShortcuts; nothing fires while typing in an input."""
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _labels_meta(meta), _FIT + """
      key('5');T.settle();const top=same(T.camera(),fit([0,0,-1],[0,-1,0],800/600));
      key('r');T.settle();const back=same(T.camera(),fit([0,1,0],[0,0,1],800/600));
      key('l');const on=T.state().labels;key('l');const off=T.state().labels;
      g('probe-card').hidden=true;key('p');const probe=g('tip-text')._text;
      let saved=0;g('save-image').onclick=()=>{saved++;};key('s');
      key('?');const help=document.body.children.filter(c=>c.className==='wss-shortcuts').length;key('Escape');
      const after=document.body.children.filter(c=>c.className==='wss-shortcuts').length;
      const typed=key('1',{target:{tagName:'INPUT',type:'text'}});const ctrl=key('1',{ctrlKey:true});
      console.log(JSON.stringify({errors,top,back,on,off,probe,saved,help,after,typed:typed.defaultPrevented,ctrl:ctrl.defaultPrevented}));
    """)
    assert out["errors"] == []
    assert out["top"] is True and out["back"] is True
    assert out["on"]["findings"] == 5 and out["on"]["branches"] is True and out["off"]["findings"] == 0 and out["off"]["branches"] is False
    assert "按 P" in out["probe"] and out["saved"] == 1
    assert out["help"] == 1 and out["after"] == 0
    assert out["typed"] is False and out["ctrl"] is False


def test_colour_bar_ticks_and_readouts_use_formatvalue(tmp_path):
    """U3: 0.348 / 0.174 / 0 instead of toFixed(1) (which printed 0.3 / 0.2 / 0.0)."""
    out = _m1_gl(tmp_path, """
      key('f');key('f');const osi=g('cbar')._html,desc=g('scale-description')._text;g('field-seg').children[0].onclick();
      T.pick([1,1,0]);const probe=g('probe-card').children[0].textContent;
      console.log(JSON.stringify({errors,osi,probe,desc}));
    """)
    assert out["errors"] == []
    # v0.13 fine ticks: nice steps from 0 upwards, the exact p99 end in three significant digits, no scientific notation.
    ticks = re.findall(r"<span[^>]*>([^<]+)</span>", out["osi"])
    assert ticks[0] == "0" and len(ticks) >= 5 and "e" not in "".join(ticks)
    values = [float(t) for t in ticks]
    assert values == sorted(values) and len(ticks[-1].replace("0.", "").lstrip("0")) == 3   # three significant digits at the top
    steps = {round(b - a, 6) for a, b in zip(values[:-2], values[1:-1])}
    assert len(steps) == 1                                                                  # evenly spaced nice ticks
    # OSI thresholds inside the 0 – p99 range (0.1, 0.2; 0.3 lies just above this fixture's p99) are marked on the bar.
    assert out["osi"].count('class="thr"') == 2 and ">0.1</em>" in out["osi"] and ">0.2</em>" in out["osi"]
    assert "0.0 –" not in out["desc"] and "OSI · 0 – " + ticks[-1] in out["desc"]
    assert out["probe"].startswith("探针 · 峰值 WSS 4.00 Pa · TAWSS 3.00 Pa · OSI 0.120")


# ---------------------------------------------------------------------------
# v0.12.2 §21.4: automatic labels are de-overlapped with WssReportCommon.declutterLabels
# ---------------------------------------------------------------------------

def _crowded_labels_meta(meta):
    """Three findings, the maximum-diameter station and the branch midpoint all at the viewport centre."""
    base = _labels_meta(meta)
    items = [dict(f) for f in base["findings"]["items"]]
    for f, xyz, sev in zip(items, ([0, 0, 0], [.004, 0, 0], [0, .004, 0]), ("note", "attention", "info")):
        f.update(xyz_mm=xyz, severity=sev)
    morph = json.loads(json.dumps(base["morphology"]))
    morph["aorta"]["max"]["xyz_mm"] = [0, 0, 0]
    return {**base, "findings": {**base["findings"], "items": items}, "morphology": morph}


_LAYOUT_HELPERS = """
      const auto=L=>L.filter(x=>['flabel','blabel','maxd'].includes(x.kind));
      const overlaps=L=>{const v=L.filter(x=>!x.hidden);let n=0;for(let i=0;i<v.length;i++)for(let j=i+1;j<v.length;j++){const a=v[i],b=v[j];
        if(Math.min(a.x+a.w/2,b.x+b.w/2)-Math.max(a.x-a.w/2,b.x-b.w/2)>0.5&&Math.min(a.y+a.h/2,b.y+b.h/2)-Math.max(a.y-a.h/2,b.y-b.h/2)>0.5)n++;}return n;};
      const brief=L=>auto(L).map(x=>[x.kind,x.text.split(' ')[0],x.moved,x.hidden]);
"""


def test_label_layout_separates_crowded_labels_by_priority_and_draws_leaders(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _crowded_labels_meta(meta), _LAYOUT_HELPERS + """
      g('view').clientWidth=1200;T.resize();
      T.apply(Object.assign(T.state(),{labels:{findings:5,branches:true}}));
      const L=T.layoutLabels(),lines=(T.leaders().match(/<line /g)||[]).length;
      const divs=g('labels').children.filter(d=>d.classList.contains('flabel')||d.classList.contains('blabel')||d.classList.contains('maxd'));
      console.log(JSON.stringify({errors,brief:brief(L),overlaps:overlaps(auto(L)),lines,leaders:L.filter(x=>x.leader).length,
        tops:divs.map(d=>[d.textContent.split(' ')[0],d.style.left,d.style.top,d.style.display]),
        anchor:auto(L).map(x=>[Math.round(x.ax),Math.round(x.ay)])}));
    """)
    assert out["errors"] == []
    kinds = {b[1] if b[0] == "flabel" else b[0]: b for b in out["brief"]}
    # attention (F2) keeps its spot; the note / info findings, the diameter and the branch name make way.
    assert kinds["F2"][2] is False and kinds["F1"][2] is True and kinds["F3"][2] is True
    assert kinds["maxd"][2] is True and kinds["blabel"][2] is True
    assert not any(b[3] for b in out["brief"])            # wide viewport: nothing hidden
    assert out["overlaps"] == 0
    assert out["lines"] == out["leaders"] >= 4             # one 1 px leader per moved chip
    assert all(abs(a[0] - 600) <= 3 and abs(a[1] - 300) <= 3 for a in out["anchor"])  # leaders return to the (near-)shared anchor
    assert all(t[1].endswith("px") and t[2].endswith("px") and t[3] == "" for t in out["tops"])


def test_label_layout_hides_overflow_in_compact_or_narrow_views(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _crowded_labels_meta(meta), _LAYOUT_HELPERS + """
      T.apply(Object.assign(T.state(),{labels:{findings:5,branches:true}}));
      g('view').clientWidth=260;g('view').clientHeight=70;T.resize();const narrow=T.layoutLabels();
      g('view').clientWidth=1200;g('view').clientHeight=600;T.resize();const wide=T.layoutLabels();
      document.body.classList.add('compact');g('view').clientWidth=260;g('view').clientHeight=70;const compact=T.layoutLabels();
      const hiddenDivs=g('labels').children.filter(d=>d.style.display==='none').map(d=>d.textContent.split(' ')[0]);
      console.log(JSON.stringify({errors,narrow:brief(narrow),wide:brief(wide),compact:brief(compact),hiddenDivs,overlaps:overlaps(auto(narrow))}));
    """)
    assert out["errors"] == []
    hidden = [b for b in out["narrow"] if b[3]]
    assert hidden and out["overlaps"] == 0
    # The lowest priorities go first: the branch name before the diameter before any finding; F2 (attention) stays.
    order = {"blabel": 0, "maxd": 1}
    assert any(b[0] == "blabel" for b in hidden)
    assert not any(b[1] == "F2" and b[3] for b in out["narrow"])
    assert all(order.get(b[0], 2) <= max(order.get(h[0], 2) for h in hidden) or not b[3] for b in out["narrow"])
    assert not any(b[3] for b in out["wide"])
    assert any(b[3] for b in out["compact"]) and "主动脉" in out["hiddenDivs"]


def test_label_layout_reruns_only_when_the_camera_moves(tmp_path):
    """requestAnimationFrame-throttled: frame() compares a camera / size / visibility signature."""
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _crowded_labels_meta(meta), """
      T.apply(Object.assign(T.state(),{labels:{findings:3,branches:false}}));
      T.frame();const a=T.labelRuns();T.frame();T.frame();const still=T.labelRuns();
      T.apply(Object.assign(T.state(),{camera:{position:[5,6,7],target:[0,0,0],up:[0,0,1]}}));T.frame();const moved=T.labelRuns();
      T.frame();const again=T.labelRuns();
      g('view').clientWidth=640;T.resize();T.frame();const resized=T.labelRuns();
      console.log(JSON.stringify({errors,a,still,moved,again,resized}));
    """)
    assert out["errors"] == []
    assert out["a"] >= 1 and out["still"] == out["a"]
    assert out["moved"] == out["still"] + 1 and out["again"] == out["moved"]
    assert out["resized"] == out["again"] + 1


def test_export_label_layout_is_recomputed_at_the_target_resolution(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _crowded_labels_meta(meta), _LAYOUT_HELPERS + """
      g('view').clientWidth=1200;T.resize();
      T.apply(Object.assign(T.state(),{labels:{findings:5,branches:true}}));
      const live=T.layoutLabels(),x2=T.exportLayout(2400,1200,2);
      console.log(JSON.stringify({errors,live:brief(live).map(b=>[b[0],b[1],b[2]]),x2:x2.map(x=>[x.kind,x.text.split(' ')[0],x.moved]),
        overlaps:overlaps(x2),anchor:x2.map(x=>[Math.round(x.ax),Math.round(x.ay)]),hidden:x2.filter(x=>x.hidden).length,
        h:x2.map(x=>x.h)}));
    """)
    assert out["errors"] == []
    # Same participants, same priorities: the attention finding stays, the rest move, at the 2× resolution.
    sort = lambda rows: sorted(rows, key=lambda r: (r[0], r[1]))
    assert sort(out["x2"]) == sort(out["live"])
    assert out["overlaps"] == 0 and out["hidden"] == 0
    assert all(abs(a[0] - 1200) <= 5 and abs(a[1] - 600) <= 5 for a in out["anchor"]) and all(h == 38 for h in out["h"])


# ---------------------------------------------------------------------------
# v0.13: RRT / ECAP (derived from TAWSS / OSI), per-field editable thresholds, fine colour bar and the
# four-tab menu (查看 / 测量 / 审阅 / 导出).
# ---------------------------------------------------------------------------

def test_rrt_ecap_are_derived_in_the_viewer_for_an_older_three_head_page(tmp_path):
    """Point 0: TAWSS 0.2 Pa, OSI 0.25 → RRT 1/(0.5·0.2) = 10 Pa⁻¹, ECAP 1.25 Pa⁻¹; thresholds follow the field."""
    out = _m1_gl(tmp_path, """
      const seg=g('field-seg');seg.children[3].onclick();
      const rrt={cbar:g('cbar-unit')._text,panel:T.panel(),thr:[g('thr0').value,g('thr1').value,g('thr2').value],unit:g('thr-unit')._text,
        label:g('thr-label-text')._text,frame:g('frame-status')._text,quick:g('quick-stats')._html,bar:g('cbar')._html,sc:g('schema-controls').hidden,
        tabText:g('field-tabs').children.map(t=>t.textContent)};
      T.pick([0,0,0]);const probe=g('probe-card').children[0].textContent;
      g('thr0').value='8';g('thr0').onchange();const edited={panel:T.panel(),state:T.state().field_thresholds,quick:g('quick-stats')._html};
      seg.children[4].onclick();const ecap={cbar:g('cbar-unit')._text,thr:[g('thr0').value,g('thr1').value,g('thr2').value],panel:T.panel()};
      seg.children[0].onclick();const wss=[g('thr0').value,g('thr1').value,g('thr2').value,g('thr-label-text')._text];
      T.apply(Object.assign(T.state(),{field_thresholds:{}}));const cleared=T.state().field_thresholds;
      T.apply(Object.assign(T.state(),{field:'rrt',field_thresholds:{rrt:[2,3,4],bogus:[1,2,3],ecap:[3,2,1]}}));
      console.log(JSON.stringify({errors,rrt,probe,edited,ecap,wss,cleared,restored:T.state().field_thresholds,field:T.field()}));
    """)
    assert out["errors"] == []
    r = out["rrt"]
    assert r["cbar"] == "RRT · Pa⁻¹" and r["thr"] == ["5", "10", "20"] and r["unit"] == "Pa⁻¹" and r["label"] == "RRT 阈值"
    assert "RRT · 由 TAWSS / OSI 派生（0.8 s 周期）" in r["panel"] and "RRT &gt; 5 Pa⁻¹" in r["panel"] and "RRT = 1 / [(1 − 2·OSI)·TAWSS]" in r["panel"]
    assert "逐点计算的派生指标" in r["frame"] and "<b>RRT</b>" in r["quick"] and "完整统计" in r["quick"]
    assert r["sc"] is True and "相对滞留时间 RRT · Pa⁻¹" in r["tabText"]              # menu copy hidden behind the segmented control
    assert r["bar"].count('class="thr"') == 1 and ">5</em>" in r["bar"]              # 10 / 20 Pa⁻¹ lie above this fixture's p99
    assert "峰值 WSS 1.00 Pa" in out["probe"] and "RRT 10.0 Pa⁻¹" in out["probe"] and "ECAP 1.25 Pa⁻¹" in out["probe"]
    # Editing a threshold recomputes the fractions from the arrays and lands in the view state (base units).
    assert "RRT &gt; 8 Pa⁻¹" in out["edited"]["panel"] and out["edited"]["state"] == {"rrt": [8, 10, 20]} and "&gt; 8 Pa⁻¹" in out["edited"]["quick"]
    assert out["ecap"]["cbar"] == "ECAP · Pa⁻¹" and out["ecap"]["thr"] == ["1.4", "2.8", "4.2"] and "ECAP &gt; 1.4 Pa⁻¹" in out["ecap"]["panel"]
    assert "腹主动脉瘤文献" in out["ecap"]["panel"]
    assert out["wss"] == ["0.4", "4", "7", "阈值"]                                  # WSS keeps its own thresholds
    assert out["cleared"] == {} and out["restored"] == {"rrt": [2, 3, 4]} and out["field"] == "rrt"   # invalid entries dropped


def test_summary_rrt_descriptors_are_used_when_the_page_carries_them(tmp_path):
    """A v0.13 page embeds rrt_per_pa / ecap_per_pa and summary.cycle blocks; the viewer uses them as they are."""
    from wss_deploy import cycle_fields as CF
    m1, (mesh, cloud, centerline) = _m1_parts()
    tawss, osi = cloud["extra"]["tawss_pa"].astype(float), cloud["extra"]["osi"].astype(float)
    extra = CF.with_derived({"tawss": {"values": tawss}, "osi": {"values": osi}})
    d = CF.derive_indices(tawss, osi)
    fields = {**m1["fields"], "rrt": CF.descriptor("rrt", d["rrt"]), "ecap": CF.descriptor("ecap", d["ecap"])}
    meta = {**m1, "fields": fields, "cycle": CF.cycle_block(extra, np.ones(4, dtype=np.int32), {1: "主动脉"}, 10.)}
    xe = {**mesh["extra"], "rrt_per_pa": (d["rrt"] * 2).astype(np.float32), "ecap_per_pa": d["ecap"].astype(np.float32)}
    xc = {**cloud["extra"], "rrt_per_pa": (d["rrt"] * 2).astype(np.float32), "ecap_per_pa": d["ecap"].astype(np.float32)}
    out = _domless_gl(tmp_path, meta, """
      g('field-seg').children[3].onclick();T.pick([0,0,0]);
      console.log(JSON.stringify({errors,labels:g('field-seg').children.map(b=>b.textContent),probe:g('probe-card').children[0].textContent,panel:T.panel()}));
    """, parts=(meta, {**mesh, "extra": xe}, {**cloud, "extra": xc}, centerline), embed_derived=True)   # the v0.13 layout
    assert out["errors"] == [] and out["labels"] == ["峰值 WSS", "TAWSS", "OSI", "RRT", "ECAP"]
    assert "RRT 20.0 Pa⁻¹" in out["probe"]                                          # the embedded array, not a recomputation
    assert "RRT · 由 TAWSS / OSI 派生" in out["panel"]


def test_menu_is_four_tabs_and_programmatic_opens_follow_their_tab(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const panes=()=>['view','measure','review','export'].filter(k=>!g('pane-'+k).hidden);
      const s0={panes:panes(),on:g('mtab-view').classList.contains('on'),sel:g('mtab-view')['aria-selected'],display:g('menu-display').open,badge:[g('mtab-review-count')._text,g('mtab-review-count').hidden]};
      g('mtab-measure').onclick();const s1={panes:panes(),measure:g('menu-measure').open};
      g('mtab-export').onclick();const s2={panes:panes(),view:g('menu-view').open};
      T.mode('distance');const s3={panes:panes(),measure:g('menu-measure').open};T.mode('');
      T.apply(Object.assign(T.state(),{ui:{menu:'presets'}}));const s4={panes:panes(),presets:g('menu-presets').open,exp:g('menu-view').open};
      T.apply(Object.assign(T.state(),{ui:{tab:'review'}}));const s5={panes:panes(),findings:g('menu-findings').open};
      const sc=g('schema-controls').hidden;
      console.log(JSON.stringify({errors,s0,s1,s2,s3,s4,s5,sc,hidden:['view','measure','review','export'].map(k=>g('mtab-'+k).hidden)}));
    """)
    assert out["errors"] == []
    # The 审阅 badge counts undecided 「关注」 findings only; this fixture's single finding is a 「提示」.
    assert out["s0"] == {"panes": ["view"], "on": True, "sel": "true", "display": True, "badge": ["", True]}
    assert out["s1"] == {"panes": ["measure"], "measure": True}                    # first card opens with its tab
    assert out["s2"] == {"panes": ["export"], "view": True}
    assert out["s3"] == {"panes": ["measure"], "measure": True}                    # a measurement mode brings its tab forward
    assert out["s4"] == {"panes": ["view"], "presets": True, "exp": True}          # v0.14: 预设 lives under 查看; 导出 keeps its card
    assert out["s5"] == {"panes": ["review"], "findings": True}
    assert out["hidden"] == [False, False, False, False]
    assert out["sc"] is False                                                        # single field: the menu keeps its field label


def test_without_webgl_only_the_view_tab_is_left(tmp_path):
    meta, *_ = _minimal_report_inputs()
    _, files = _report_parts(tmp_path, _analysis_meta(meta))
    out = _node("require(DIR+'/core.js');require(DIR+'/common.js');" + _STUB_DOM + r"""
      global.window=global;global.THREE={WebGLRenderer:class{constructor(){throw new Error('No WebGL');}}};
      global.localStorage={getItem:()=>null,setItem:()=>{}};
      global.location={hash:'',href:'http://x/report.html',protocol:'http:',origin:'http://x',pathname:'/report.html'};
      global.history={replaceState(){}};global.navigator={};global.performance={now:()=>0};global.requestAnimationFrame=()=>0;
      global.addEventListener=()=>{};global.parent={postMessage:()=>{}};console.error=()=>{};require(DIR+'/main.js');
      const g=id=>document.getElementById(id);
      console.log(JSON.stringify({tabs:['view','measure','review','export'].map(k=>g('mtab-'+k).hidden),stats:g('menu-stats').hidden,view:g('pane-view').hidden}));
    """, tmp_path, *files)
    assert out == {"tabs": [False, True, True, True], "stats": False, "view": False}


def test_colour_bar_has_band_edge_ticks_and_wss_threshold_marks(tmp_path):
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const cont=g('cbar')._html;
      T.apply(Object.assign(T.state(),{bands:20}));const b20=g('cbar')._html,bands=T.state().bands;
      T.apply(Object.assign(T.state(),{bands:7}));const bad=T.state().bands;
      console.log(JSON.stringify({errors,cont,b20,bands,bad}));
    """)
    assert out["errors"] == []
    labels = lambda h: re.findall(r"<span[^>]*>([^<]+)</span>", h)
    assert labels(out["cont"])[0] == "0" and labels(out["cont"])[-1] == "3.00" and len(labels(out["cont"])) >= 6
    assert out["cont"].count('class="tk minor"') >= 4
    # p99 3 Pa: only the 0.4 Pa low threshold lies inside the bar (4 / 7 Pa are above it).
    assert out["cont"].count('class="thr"') == 1 and ">0.4</em>" in out["cont"] and "低阈值 0.4 Pa" in out["cont"]
    assert out["bands"] == 20 and out["b20"].count('class="tk') == 21 and len(labels(out["b20"])) == 11
    assert out["bad"] == 0                                                          # not one of the offered steps


def test_menu_tables_scroll_inside_their_card_instead_of_widening_it(tmp_path):
    """v0.13.1: with wide system fonts a table's natural width stretched its card past the clipped menu edge.
    Every statistics / probe table now sits in a .tscroll box and the card grid track may shrink."""
    from wss_deploy.report import TEMPLATE
    assert "details.menu-card>.body{grid-template-columns:minmax(0,1fr)}" in TEMPLATE
    assert ".tscroll{overflow-x:auto" in TEMPLATE and ".tscroll table{width:100%;min-width:max-content}" in TEMPLATE
    assert ".tscroll th:first-child,.tscroll td:first-child{position:sticky;left:0" in TEMPLATE
    out = _m1_gl(tmp_path, """
      T.pick([0,0,0]);g('probe-card').children[1].children[0].onclick();
      const panel=T.panel(),probe=g('probe-log').children.map(c=>[c.className,(c.children[0]||{}).className]);
      console.log(JSON.stringify({errors,tables:(panel.match(/<table/g)||[]).length,wrapped:(panel.match(/<div class="tscroll"><table/g)||[]).length,probe}));
    """)
    assert out["errors"] == [] and out["tables"] >= 3 and out["wrapped"] == out["tables"]
    assert out["probe"] == [["tscroll", "probe-table"]]


# ---------------------------------------------------------------------------
# v0.14 (F10): RRT / ECAP are declared but not embedded (the page derives them with the cycle_fields formula),
# branch ids are uint8 when they fit, the file is written atomically; old int32 / embedded pages still load.
# ---------------------------------------------------------------------------

def _derived_m1_parts():
    from wss_deploy import cycle_fields as CF
    m1, (mesh, cloud, centerline) = _m1_parts()
    tawss, osi = cloud["extra"]["tawss_pa"], cloud["extra"]["osi"]
    extra = CF.with_derived({"tawss": {"values": tawss}, "osi": {"values": osi}})
    fields = {**m1["fields"], "rrt": CF.descriptor("rrt", extra["rrt"]["values"]), "ecap": CF.descriptor("ecap", extra["ecap"]["values"])}
    points, vertices = CF.derived_display_arrays(fields, extra, {"tawss_pa": mesh["extra"]["tawss_pa"], "osi": mesh["extra"]["osi"]})
    meta = {**m1, "fields": fields}
    return meta, {**mesh, "extra": {**mesh["extra"], **vertices}}, {**cloud, "extra": {**cloud["extra"], **points}}, centerline, points, vertices


def test_v014_payload_leaves_out_derived_fields_and_the_viewer_rebuilds_them_bit_identically(tmp_path):
    meta, mesh, cloud, centerline, points, vertices = _derived_m1_parts()
    html, files = _report_parts(tmp_path, meta, parts=(meta, mesh, cloud, centerline))
    arrays = json.loads(dict(files)["arrays.json"])
    assert set(arrays["xf"]) == {"tawss_pa", "osi"}                                   # RRT / ECAP not embedded
    assert arrays["dt"] == {"ms": "u8", "ps": "u8", "cs": "u8"}
    assert len(base64.b64decode(arrays["ps"])) == len(cloud["pts"])                   # one byte per branch id
    assert not list(tmp_path.glob("*.tmp"))                                            # atomic write leaves nothing behind
    out = _domless_gl(tmp_path, meta, """
      console.log(JSON.stringify({errors,supported:T.supported(),seg:T.segTypes(),rrtP:T.values('rrt','p'),rrtM:T.values('rrt','m'),
        ecapP:T.values('ecap','p'),ecapM:T.values('ecap','m')}));
    """, parts=(meta, mesh, cloud, centerline))
    assert out["errors"] == [] and out["supported"] == ["wss", "tawss", "osi", "rrt", "ecap"]
    assert out["seg"] == ["Uint8Array", "Uint8Array", "Uint8Array"]
    f32 = lambda v: np.asarray(v, dtype=np.float64).astype(np.float32)
    # float32 values from the page equal the Python-derived arrays bit for bit (NaN stays NaN)
    for key, which, ref in (("rrtP", "p", points["rrt_per_pa"]), ("ecapP", "p", points["ecap_per_pa"]),
                            ("rrtM", "m", vertices["rrt_per_pa"]), ("ecapM", "m", vertices["ecap_per_pa"])):
        got = f32([np.nan if x is None else x for x in out[key]])
        assert np.array_equal(got, np.asarray(ref, dtype=np.float32), equal_nan=True), (key, got, ref)


def test_v014_embed_derived_keeps_the_v013_layout_and_large_branch_ids_stay_int32(tmp_path):
    meta, mesh, cloud, centerline, *_ = _derived_m1_parts()
    _, files = _report_parts(tmp_path, meta, parts=(meta, mesh, cloud, centerline), embed_derived=True)
    assert set(json.loads(dict(files)["arrays.json"])["xf"]) == {"tawss_pa", "osi", "rrt_per_pa", "ecap_per_pa"}
    big = {**cloud, "segment": np.full(len(cloud["pts"]), 300, dtype=np.int32)}
    _, files = _report_parts(tmp_path, meta, parts=(meta, mesh, big, centerline))
    arrays = json.loads(dict(files)["arrays.json"])
    assert arrays["dt"] == {"ms": "u8", "cs": "u8"}                                    # ps keeps int32
    assert np.array_equal(np.frombuffer(base64.b64decode(arrays["ps"]), dtype=np.int32), big["segment"])
    out = _domless_gl(tmp_path, meta, "console.log(JSON.stringify({errors,seg:T.segTypes()}));", parts=(meta, mesh, big, centerline))
    assert out == {"errors": [], "seg": ["Uint8Array", "Int32Array", "Uint8Array"]}


def test_v014_compare_range_is_field_scoped_shared_and_never_saved_as_the_users_choice(tmp_path):
    """F1: the compare page's fixed range applies to its field only, carries case_max, is not persisted, and a
    display change after load tells the parent page (wss-view:changed)."""
    out = _m1_gl(tmp_path, """
      const s0=T.state().range;
      T.apply(Object.assign(T.state(),{range:{mode:'fixed',min:0,max:9,field:'wss',shared:true}}));
      const s1=T.state().range,d1=g('scale-description')._text;
      g('field-seg').children[1].onclick();
      const s2=T.state().range,d2=g('scale-description')._text;
      g('highlight-pct').value='2';g('highlight-pct').oninput({target:g('highlight-pct')});
      const saved=JSON.parse(localStorage.getItem('wss-report-scale-v1'));
      setTimeout(()=>{g('cmap').value='turbo';g('cmap').onchange({target:g('cmap')});
        setTimeout(()=>console.log(JSON.stringify({errors,s0,s1,d1,s2,d2,saved,changed:posted.filter(p=>p[0].type==='wss-view:changed').map(p=>[p[0].family,p[1]])})),200);},20);
    """)
    assert out["errors"] == []
    assert out["s0"]["mode"] == "case" and out["s0"]["case_max"] == pytest.approx(out["s0"]["max"])
    assert out["s1"] == {"mode": "fixed", "min": 0, "max": 9, "field": "wss", "shared": True, "case_max": pytest.approx(out["s0"]["case_max"])}
    assert "并排比较共用上限" in out["d1"] and "0 – 9" in out["d1"]
    # TAWSS keeps its own p99 while the shared WSS limit stays recorded for WSS
    assert out["s2"]["field"] == "wss" and out["s2"]["max"] == 9 and "本例空间 p99" in out["d2"] and "TAWSS" in out["d2"]
    assert out["saved"]["mode"] == "case"                                            # the shared range is not the user's choice
    assert out["changed"] and all(c == ["wall", "http://x"] for c in out["changed"])


def test_v014_timing_table_marks_precomputed_stages(tmp_path):
    meta, *_ = _minimal_report_inputs()
    meta = {**_analysis_meta(meta), "timing_s": {"smooth_resample": 0.02, "features": 0.9, "inference_5_models": 6.8, "precompute": 11.2},
            "geometry_cache": {"enabled": True, "reused": ["mesh", "morph"], "computed": ["pointgeom"]}}
    out = _domless_gl(tmp_path, meta, "console.log(JSON.stringify({errors,panel:T.panel()}));")
    assert out["errors"] == []
    assert "<td>平滑与重采样</td><td>0.02 s · 已预计算</td>" in out["panel"] and "<td>构建特征</td><td>0.90 s</td>" in out["panel"]
    assert "<td>后台几何预计算（确认出口期间）</td><td>11.20 s</td>" in out["panel"]


def test_v015_colour_bar_has_a_text_alternative_that_follows_the_field(tmp_path):
    """Round 15 (a11y): the colour bar is a picture; it carries role=img and an aria-label with units, range and
    the threshold marks, repainted with the bar."""
    meta, *_ = _minimal_report_inputs()
    out = _domless_gl(tmp_path, _analysis_meta(meta), """
      const first={role:g('cbar').role,label:g('cbar')['aria-label']};
      T.apply(Object.assign(T.state(),{bands:10}));
      console.log(JSON.stringify({errors,first,after:g('cbar')['aria-label']}));
    """)
    assert out["errors"] == []
    assert out["first"]["role"] == "img"
    label = out["first"]["label"]
    assert label.startswith("色标 ") and "Pa" in label and "0 – 3.00" in label and "阈值 0.4" in label
    assert out["after"] and out["after"].startswith("色标 ")
