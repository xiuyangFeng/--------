import base64
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from wss_deploy.report import update_html_meta
from wss_deploy.volume_report import build_html


VIEWER = Path(__file__).resolve().parents[1] / "wss_deploy/static/volume_viewer.js"
COMMON = Path(__file__).resolve().parents[1] / "wss_deploy/static/report_common.js"


def sample():
    pts = np.array([[0., 0., 0.], [0., 0., 2.], [1., 0., 2.], [0., 0., 4.]], dtype=np.float32)
    mesh = {"vertices": pts, "faces": [[0, 1, 2], [1, 2, 3]], "pressure_pa": [1., 2., np.nan, 4.]}
    cloud = {"pts": pts, "pressure_pa": [1., 2., 3., 4.], "velocity_m_s": [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 2]],
             "is_wall": [1, 0, 0, 0], "segment": [1, 1, 1, 1]}
    center = {"xyz": [[0, 0, 0], [0, 0, 4]], "segment": [1, 1]}
    return mesh, cloud, center


def arrays_from_html(html):
    return json.loads(re.search(r'<script id="volume-arrays" type="application/json">(.*?)</script>', html).group(1))


def node_json(script):
    if not shutil.which("node"):
        pytest.skip("Node is required to exercise volume viewer numerical behavior")
    program = "const core=require(" + json.dumps(str(VIEWER)) + ");\n" + script
    result = subprocess.run(["node", "-e", program], check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_volume_export_embeds_real_fields_and_reuses_safe_metadata_update(tmp_path):
    mesh, cloud, center = sample()
    out = tmp_path / "volume.html"
    meta = {"case_id": "</script><script>alert(1)</script>", "pressure_reference": "出口参考压"}
    lines = [{"points": [[0, 0, 1], [0, 0, 2]], "speed_m_s": [1., 2.]}]
    build_html(out, meta, mesh, cloud, center, lines)
    html = out.read_text()
    assert "</script><script>alert(1)</script>" not in html
    assert "https://cdn" not in html
    data = arrays_from_html(html)
    assert data["has_segments"] is True
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(data["velocity_m_s"]), dtype=np.float32).reshape(-1, 3), cloud["velocity_m_s"])
    assert len(data["streamlines"]) == 1
    assert np.isnan(np.frombuffer(base64.b64decode(data["wall_pressure_pa"]), dtype=np.float32)[2])
    update_html_meta(out, {**meta, "timing_s": {"total": 1.5}, "fields": {"velocity": {"units": "m/s"}}})
    assert '"total": 1.5' in out.read_text()
    assert arrays_from_html(out.read_text()) == data


def test_pressure_only_export_does_not_fabricate_velocity_or_streamlines(tmp_path):
    mesh, cloud, center = sample()
    cloud.pop("velocity_m_s")
    out = tmp_path / "p.html"
    build_html(out, {}, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    assert "velocity_m_s" not in data
    assert data["streamlines"] == []
    with pytest.raises(ValueError, match="velocity vector"):
        build_html(out, {}, mesh, cloud, center, [{"points": [[0, 0, 0], [1, 1, 1]], "speed_m_s": [1, 1]}])


@pytest.mark.parametrize("mutate,match", [
    (lambda m, c, l: c.update(velocity_m_s=[[0, 0, 1]]), "match cloud.pts"),
    (lambda m, c, l: c.update(pressure_pa=[1, 2, np.nan, 4]), "finite scalar"),
    (lambda m, c, l: c.update(is_wall=[0, 0, 2, 0]), "Boolean"),
    (lambda m, c, l: m.update(faces=[[0, 1, 20]]), "integer point indices"),
    (lambda m, c, l: l.update(tangent=[[0, 0, 1]]), "match centerline.xyz"),
])
def test_volume_export_rejects_misaligned_or_nonfinite_predictions(tmp_path, mutate, match):
    mesh, cloud, center = sample()
    mutate(mesh, cloud, center)
    with pytest.raises(ValueError, match=match):
        build_html(tmp_path / "bad.html", {}, mesh, cloud, center)


def test_slab_filters_wall_and_branch_and_uses_physical_thickness():
    result = node_json("""
      const pts=[0,0,0, 0,0,.9, 0,0,1.1, 0,0,-.5, 1,0,0];
      const plane={origin:[0,0,0],normal:[0,0,5]};
      console.log(JSON.stringify({
        all:core.slabIndices(pts,[1,0,0,0,0],[1,1,1,2,1],plane,2),
        branch:core.slabIndices(pts,[1,0,0,0,0],[1,1,1,2,1],plane,2,1),
        thin:core.slabIndices(pts,[1,0,0,0,0],[1,1,1,2,1],plane,.4),
        inside:core.insideIndices(pts,[1,0,0,0,0])
      }));
    """)
    assert result == {"all": [1, 3, 4], "branch": [1, 4], "thin": [4], "inside": [1, 2, 3, 4]}


def test_module_partition_and_slice_gesture_helpers():
    result = node_json("""
      const segments=[2,1,2,1], walls=[0,0,1,0], points=[0,1,2,3];
      console.log(JSON.stringify({
        modules:core.moduleSummary(segments,walls),
        selected:core.moduleIndices([0,1,2,3],segments,1),
        move:core.interactionDelta(0,12,{}),
        lateral:core.interactionDelta(8,0,{shiftKey:true}),
        zoom:core.interactionDelta(0,12,{ctrlKey:true})
      }));
    """)
    assert result["modules"] == [
        {"segment": 1, "count": 2, "interior": 2, "wall": 0},
        {"segment": 2, "count": 2, "interior": 1, "wall": 1},
    ]
    assert result["selected"] == [1, 3]
    assert result["move"] == {"zoom": 0, "position": 12, "offsetU": 0}
    assert result["lateral"] == {"zoom": 0, "position": 0, "offsetU": 8}
    assert result["zoom"] == {"zoom": -12, "position": 0, "offsetU": 0}


def test_report_exposes_module_selector_and_slice_offsets(tmp_path):
    mesh, cloud, center = sample()
    out = tmp_path / "interactive.html"
    build_html(out, {}, mesh, cloud, center)
    html = out.read_text()
    data = arrays_from_html(html)
    assert data["modules"] == [{"segment": 1, "count": 4, "interior": 3, "wall": 1}]
    assert 'id="volume-module"' in html
    assert 'id="slice-offset-u"' in html and 'id="slice-offset-v"' in html
    assert "Shift 滚轮改厚度" in html and "拖动蓝色截面" in html


def test_automatic_planes_follow_arc_length_without_joining_branches():
    result = node_json("""
      const groups=core.groupCenterline([0,0,0, 0,0,1, 0,0,10, 100,0,0, 110,0,0],[1,1,1,2,2]);
      const planes=core.automaticPlanes(groups);
      console.log(JSON.stringify({planes, middle:core.centerlinePlane(groups[0],.5)}));
    """)
    assert result["middle"]["origin"] == [0, 0, 5]
    assert len(result["planes"]) == 12
    assert result["planes"][1]["origin"] == [0, 0, 2]
    assert result["planes"][7]["origin"] == [102, 0, 0]
    assert result["planes"][1]["normal"] == [0, 0, 1]
    assert result["planes"][7]["normal"] == [1, 0, 0]


def test_manual_rotation_changes_plane_and_preserves_orthogonal_projection():
    result = node_json("""
      const basis=core.rotatePlane([0,0,1],90,0);
      const indices=core.slabIndices([0,1,0, 0,0,1, 0,0,0],null,null,{origin:[0,0,0],normal:basis.normal},.1);
      console.log(JSON.stringify({basis,indices}));
    """)
    np.testing.assert_allclose(result["basis"]["normal"], [0, -1, 0], atol=1e-12)
    basis = np.array([result["basis"][key] for key in ["normal", "u", "v"]])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    assert result["indices"] == [1, 2]


def test_cut_sides_keep_points_on_selected_side_of_plane():
    result = node_json("""
      const pts=[-2,0,0, -0.1,0,0, 0.1,0,0, 2,0,0];
      console.log(JSON.stringify({positive:core.sideIndices([0,1,2,3],pts,{origin:[0,0,0],normal:[1,0,0]},1),negative:core.sideIndices([0,1,2,3],pts,{origin:[0,0,0],normal:[1,0,0]},-1)}));
    """)
    assert result == {"positive": [2, 3], "negative": [0, 1]}


def test_local_idw_interpolation_masks_extrapolation_and_missing_support():
    result = node_json("""
      const points=[[-1,-1],[1,-1],[1,1],[-1,1]], values=[0,2,4,2];
      const field=core.interpolateIDW(points,values,{bounds:[-2,2,-2,2],nx:3,ny:3,maxDistance:3,minNeighbors:4});
      const restricted=core.interpolateIDW(points,values,{bounds:[-2,2,-2,2],nx:3,ny:3,maxDistance:.5,minNeighbors:4});
      console.log(JSON.stringify({center:field.values[4],centerMask:field.mask[4],valid:field.validCount,outerMask:field.mask[0],restricted:restricted.validCount}));
    """)
    assert result["centerMask"] == 1
    assert result["center"] == pytest.approx(2.0)
    assert result["outerMask"] == 0
    assert result["restricted"] == 0


def test_centerline_edges_order_shuffled_atlas_before_automatic_planes():
    result = node_json("""
      const group=core.groupCenterline([0,0,10, 0,0,0, 0,0,2],[3,3,3],null,[1,2,2,0])[0];
      console.log(JSON.stringify(core.centerlinePlane(group,.4)));
    """)
    assert result["origin"] == [0, 0, 4]
    assert result["normal"] == [0, 0, 1]


def test_speed_uses_vector_magnitude_and_statistics_exclude_wall_samples():
    result = node_json("""
      const values=core.speedField([100,0,0, 3,4,0, 0,0,2]);
      const inside=core.insideIndices([0,0,0, 0,0,1, 0,0,2],[1,0,0]);
      console.log(JSON.stringify(core.statistics(values,inside)));
    """)
    assert result["count"] == 2
    assert result["mean"] == 3.5
    assert result["max"] == 5
    assert result["p99"] == pytest.approx(4.97)


def test_no_webgl_viewer_still_switches_fields_and_manual_automatic_slices(tmp_path):
    """Exercise DOM events and fallback, including field-specific mode availability."""
    mesh, cloud, center = sample()
    out = tmp_path / "report.html"
    build_html(out, {"case_id": "test", "pressure_reference": "relative outlet"}, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    script = r"""
      class Element {
        constructor(value=''){this.value=value;this.children=[];this.events={};this.hidden=false;this.disabled=false;this.textContent='';this.checked=false;this.width=560;this.height=520;}
        appendChild(x){this.children.push(x);if(this.value==='')this.value=x.value;return x;}
        append(...xs){xs.forEach(x=>this.appendChild(x));}
        replaceChildren(){this.children=[];}
        querySelector(selector){const value=selector.match(/value="([^"]+)"/)[1];return this.children.find(x=>x.value===value);}
        addEventListener(name,fn){this.events[name]=fn;}
        getContext(){return new Proxy({}, {get:()=>()=>{}});}
      }
      const els={}, defaults={'volume-field':'','volume-mode':'cloud','slice-basis':'centerline','slice-position':'50','slice-pitch':'0','slice-yaw':'0','slice-thickness':'2','volume-opacity':'.1'};
      global.document={getElementById:id=>els[id]||(els[id]=new Element(defaults[id]||'')),createElement:()=>new Element()};
      function setupOptions(id,values){for(const value of values){const el=new Element(value);document.getElementById(id).appendChild(el);}}
      setupOptions('volume-mode',['cloud','slice','wall','streamlines']);setupOptions('slice-basis',['centerline','x','y','z']);
      document.getElementById('wss-report-meta').textContent=JSON.stringify({case_id:'test',pressure_reference:'relative outlet'});
      document.getElementById('volume-arrays').textContent=JSON.stringify(DATA);
      global.THREE={WebGLRenderer:class {constructor(){throw new Error('No WebGL');}}};
      global.addEventListener=()=>{};global.requestAnimationFrame=fn=>fn();
      delete require.cache[require.resolve(VIEWER)];require(VIEWER);
      const initial={field:els['volume-field'].value,wallDisabled:els['volume-mode'].querySelector('option[value="wall"]').disabled,
        count:els['volume-statistics'].children[0].children[1].textContent,error:els['volume-error'].hidden};
      els['volume-field'].value='pressure';els['volume-field'].events.change();
      const pressure={wallDisabled:els['volume-mode'].querySelector('option[value="wall"]').disabled,vectorsDisabled:els['volume-vectors'].disabled};
      els['auto-presets'].value='2';els['slice-apply-auto'].events.click();
      const auto={mode:els['volume-mode'].value,position:els['slice-position'].value,branch:els['slice-branch'].value};
      els['slice-basis'].value='x';els['slice-basis'].events.change();els['slice-position'].value='100';els['slice-position'].events.input();
      console.log(JSON.stringify({initial,pressure,auto,manual:els['slice-details'].textContent}));
    """.replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    # v0.12: wall pressure exists in this report, so its mode stays available under velocity (the tab switches
    # the field itself); it used to be disabled whenever velocity was selected.
    assert result["initial"] == {"field": "velocity", "wallDisabled": False, "count": "3", "error": False}
    assert result["pressure"] == {"wallDisabled": False, "vectorsDisabled": True}
    assert result["auto"] == {"mode": "slice", "position": "40", "branch": "1"}
    assert "中心 [1.00, 0.00, 2.00]" in result["manual"]


def test_pick_planes_and_colormaps():
    result = node_json("""
      const groups=core.groupCenterline([0,0,0, 0,0,10, 0,0,20],[1,1,1],[0,0,1, 0,0,1, 0,0,1]);
      const near=core.nearestTangent(groups,[3,0,9]);
      const one=core.planeFromPicks([[3,0,9]],near.tangent);
      const two=core.planeFromPicks([[3,0,9],[-3,0,11]],near.tangent);
      const names=core.colormapNames().map(x=>x.id);
      const before=core.color(0,0,1); core.setColormap('bwr'); const bwr=core.color(0.4,0,1); core.setColormap('rainbow'); const after=core.color(1,0,1);
      console.log(JSON.stringify({near,one,two,names,before,bwr,after,css:core.colormapCSS('rainbow').slice(0,15)}));
    """)
    assert result["near"]["segment"] == 1 and result["near"]["tangent"] == [0, 0, 1]
    assert result["one"]["origin"] == [3, 0, 9] and result["one"]["normal"] == [0, 0, 1] and result["one"]["picks"] == 1
    two = result["two"]
    assert two["origin"] == [0, 0, 10] and two["picks"] == 2
    chord = np.array([-6, 0, 2]) / np.linalg.norm([-6, 0, 2])
    np.testing.assert_allclose(np.dot(two["normal"], chord), 0, atol=1e-9)        # plane contains the chord
    assert np.dot(two["normal"], [0, 0, 1]) > 0.9                                  # and stays close to the axis
    assert result["names"] == ["rainbow", "viridis", "turbo", "bwr"]   # §19.9: rainbow is the default again
    assert result["before"] == pytest.approx([0, 0, 143 / 255]) and result["after"] == pytest.approx([190 / 255, 0, 0])   # default = rainbow
    # v0.14 bwr = ColorBrewer RdBu-7 from the shared palette table: t = 0.4 sits 40 % of the way from stop 2 to the white centre
    assert result["bwr"] == pytest.approx([(209 + .4 * (247 - 209)) / 255, (229 + .4 * (247 - 229)) / 255, (240 + .4 * (247 - 240)) / 255])
    assert result["css"].startswith("linear-gradient")


def test_report_has_menu_tabs_and_pick_controls(tmp_path):
    mesh, cloud, center = sample()
    out = tmp_path / "menu.html"
    build_html(out, {}, mesh, cloud, center)
    html = out.read_text()
    for marker in ('id="mode-cloud"', 'id="mode-streamlines"', 'id="pick-toggle"', 'id="pick-clear"', 'id="colormap"',
                   'id="menu-display"', 'id="menu-slice"', 'id="menu-stats"', 'id="streamline-width"', 'option value="pick"'):
        assert marker in html


# ---------------------------------------------------------------- contract v1 additions (probe / findings / profiles / trust / view)
def test_build_html_embeds_optional_trust_and_probe_arrays_and_validates_alignment(tmp_path):
    mesh, cloud, center = sample()
    mesh["trust"] = np.array([1, 0, 4, 0], dtype=np.uint8)
    cloud["trust"] = np.array([0, 8, 0, 16], dtype=np.uint8)          # aligned to cloud.pts
    cloud["s_from_root_mm"] = [0, 10, 20, 30]
    cloud["radius_mm"] = np.array([0, 1.5, 1.5, 1.5], dtype=np.float32)
    out = tmp_path / "trust.html"
    build_html(out, {}, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(data["trust"]), dtype=np.uint8), [0, 8, 0, 16])
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(data["wall_trust"]), dtype=np.uint8), [1, 0, 4, 0])
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(data["s_from_root_mm"]), dtype=np.float32), [0, 10, 20, 30])
    assert "dist_to_wall_mm" not in data
    # interior-only arrays (contract: uint8[n_interior]) are expanded onto the wall rows
    cloud["trust"] = np.array([8, 0, 16], dtype=np.uint8)
    cloud["dist_to_wall_mm"] = [0.5, 0.7, 0.9]
    build_html(out, {}, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(data["trust"]), dtype=np.uint8), [0, 8, 0, 16])
    wall_dist = np.frombuffer(base64.b64decode(data["dist_to_wall_mm"]), dtype=np.float32)
    assert np.isnan(wall_dist[0]) and list(wall_dist[1:]) == [np.float32(0.5), np.float32(0.7), np.float32(0.9)]
    for bad, match in (({"trust": [0, 1]}, "uint8 trust"), ({"trust": [0, 1, 300, 0]}, "uint8 trust"),
                       ({"radius_mm": [1, 2]}, "finite scalar")):
        mesh2, cloud2, center2 = sample()
        cloud2.update(bad)
        with pytest.raises(ValueError, match=match):
            build_html(tmp_path / "bad.html", {}, mesh2, cloud2, center2)
    mesh3, cloud3, center3 = sample()
    mesh3["trust"] = [1, 2]
    with pytest.raises(ValueError, match="mesh.trust"):
        build_html(tmp_path / "bad.html", {}, mesh3, cloud3, center3)


def test_standard_views_units_bands_and_view_state_round_trip():
    result = node_json("""
      const frame=core.frameFromMeta({rotation:[[0,1,0],[0,0,1],[1,0,0]],origin_mm:[10,20,30],direction_source:'unknown_stl'});
      const front=core.standardCamera('front',frame,[10,20,30],100), top=core.standardCamera('top',frame,[10,20,30],100);
      const aligned=core.alignedFromWorld([10,21,30],frame), back=core.worldFromAligned([1,0,0],frame);
      const cam={position:[110,20,30],target:[10,20,30],up:[1,0,0]};
      const round=core.cameraFromAligned(core.cameraToAligned(cam,frame),frame);
      const state={schema_version:'wss-deploy.view/v1',family:'volume',camera:cam,field:'velocity',slice:{picks:[[1,2,3]]},units:'m/s'};
      const decoded=core.decodeView(core.encodeView(state));
      let bad=null;try{core.decodeView(core.encodeView({schema_version:'other'}));}catch(e){bad=e.message;}
      core.setBands(4);const banded=[core.color(0.1,0,1,'rainbow'),core.color(0.2,0,1,'rainbow'),core.color(0.3,0,1,'rainbow')];const css=(core.colormapCSS('rainbow').match(/%/g)||[]).length;core.setBands(0);
      console.log(JSON.stringify({front,top,aligned,back,round,decoded,bad,banded,css,
        mmHg:core.convertUnit(133.322,'pressure','mmHg'),cms:core.convertUnit(1.5,'velocity','cm/s'),same:core.convertUnit(2,'pressure','Pa'),
        missing:core.frameFromMeta({rotation:[[1,0]],origin_mm:[0,0,0]}),units:core.unitOptions('pressure'),invalid:core.frameFromMeta(null)}));
    """)
    # rows of R are aligned axes in world coordinates: x_aligned = world y, y_aligned = world z, z_aligned = world x
    assert result["front"]["position"] == [10, 20, -70] and result["front"]["up"] == [1, 0, 0]      # anterior = -y_aligned = -world z
    assert result["top"]["position"] == [110, 20, 30] and result["top"]["up"] == [0, 0, -1]         # superior = +z_aligned = +world x
    assert result["aligned"] == [1, 0, 0] and result["back"] == [10, 21, 30]
    np.testing.assert_allclose(result["round"]["position"], [110, 20, 30], atol=1e-9)
    assert result["decoded"]["camera"] == {"position": [110, 20, 30], "target": [10, 20, 30], "up": [1, 0, 0]}
    assert result["decoded"]["slice"]["picks"] == [[1, 2, 3]] and result["bad"]
    assert result["banded"][0] == result["banded"][1] and result["banded"][1] != result["banded"][2]   # 4 bands: 0.1/0.2 share a step
    assert result["css"] == 8                                                                          # stepped gradient: 2 stops per band
    assert result["mmHg"] == pytest.approx(1.0) and result["cms"] == pytest.approx(150.0) and result["same"] == 2
    assert result["missing"] is None and result["invalid"] is None and result["units"] == ["Pa", "mmHg"]


def test_profile_bins_region_statistics_probe_and_findings_helpers():
    result = node_json("""
      const groups=core.groupCenterline([0,0,0, 0,0,10, 0,0,20],[1,1,1],null);
      const pts=[0,0,1, 0,0,9, 0,0,19, 5,5,5];
      const segments=[1,1,1,2];
      const arc=Array.from(core.arcAlongBranch(groups,pts,segments,[0,1,2,3]));
      const region=core.regionIndices([0,1,2,3],segments,arc,1,5,20);
      const drop=core.regionPressureDrop([0,1,2],arc,[100,60,20],0,20,0.1);
      const probe=core.probeRecord(1,{pts,velocity:[0,0,0, 3,4,0, 0,0,0, 0,0,0],pressure:[1,2,3,4],segments,s:[0,9,19,0],radius:[1,2,3,4],trust:[0,8,0,0]});
      const sorted=core.findingsSorted([{id:'a',severity:'info',rank:1},{id:'b',severity:'attention',rank:2},{id:'c',severity:'note',rank:1},null]).map(x=>x.id);
      const labels=core.trustLabels({bits:{'1':'interpolation_uncovered','8':'low_sample_support'},fractions:{low_sample_support:0.25},sources:[{bit:8,label:'采样支撑弱',rule:'8 近邻半径 > 中位数×2'}]});
      console.log(JSON.stringify({bin:core.nearestBin([0,2,4,6],3.9),fraction:core.fractionForArc(groups[0],5),arc,region,drop,probe,sorted,labels,
        sphere:core.sphereIndices(pts,[0,1,2,3],[0,0,10],1.5),nearest:core.nearestPoint(pts,[0,1,2],[0,0,8])}));
    """)
    assert result["bin"] == 2 and result["fraction"] == pytest.approx(0.25)
    assert result["arc"][:3] == [0, 10, 20] and result["arc"][3] is None            # segment 2 has no centreline -> NaN
    assert result["region"] == [1, 2]
    assert result["drop"]["proximal"] == 100 and result["drop"]["distal"] == 20 and result["drop"]["drop"] == 80
    assert result["probe"]["speed"] == 5 and result["probe"]["pressure"] == 2 and result["probe"]["segment"] == 1
    assert result["probe"]["s"] == 9 and result["probe"]["radius"] == 2 and result["probe"]["trust"] == 8 and "distWall" not in result["probe"]
    assert result["sorted"] == ["b", "c", "a"]
    assert result["labels"] == [{"bit": 1, "label": "插值无支撑", "rule": "", "fraction": None},
                                {"bit": 8, "label": "采样支撑弱", "rule": "8 近邻半径 > 中位数×2", "fraction": 0.25}]
    assert result["sphere"] == [1] and result["nearest"] == {"index": 1, "distance": 1}


def test_report_template_has_menus_dock_footer_and_optional_panels(tmp_path):
    mesh, cloud, center = sample()
    out = tmp_path / "menus.html"
    build_html(out, {"review": {"status": "reviewed", "by": "R1"}, "feature_contract": {"source_hash": "abcdef0123456789"}}, mesh, cloud, center)
    html = out.read_text()
    for marker in ('id="menu-findings"', 'id="menu-profiles"', 'id="menu-view"', 'id="probe-card"', 'id="trust-legend"', 'id="findings-list"',
                   'id="profile-canvas"', 'id="region-smin"', 'id="view-front"', 'id="view-six"', 'id="view-link"', 'id="color-bands"',
                   'id="pressure-unit"', 'id="velocity-unit"', 'id="trust-overlay"', 'id="footer-review"', 'id="footer-feature"'):
        assert marker in html
    assert html.index('id="legend"') < html.index('id="probe-card"') < html.index('id="slice-panel"')   # dock order


def _stub_prelude(extra="", common=False):
    """Fake DOM for the no-WebGL path; ``extra`` runs before the viewer loads (location, parent, listeners).

    ``common`` loads the shared library first, exactly as the published page does (phase 2); without it
    the viewer must still work through its local fallbacks."""
    if common:
        extra = "require(" + json.dumps(str(COMMON)) + ");\n" + extra
    return r"""
      class Element {
        constructor(value=''){this.value=value;this.children=[];this.events={};this.hidden=false;this.disabled=false;this.textContent='';this.checked=false;this.width=560;this.height=520;this.open=false;}
        appendChild(x){this.children.push(x);if(this.value==='')this.value=x.value;return x;}
        append(...xs){xs.forEach(x=>this.appendChild(x));}
        replaceChildren(){this.children=[];}
        querySelector(selector){const value=selector.match(/value="([^"]+)"/)[1];return this.children.find(x=>x.value===value);}
        addEventListener(name,fn){this.events[name]=fn;}
        getContext(){return new Proxy({}, {get:()=>()=>{}});}
      }
      const els={}, defaults={'volume-field':'','volume-mode':'cloud','slice-basis':'centerline','slice-position':'50','slice-pitch':'0','slice-yaw':'0','slice-thickness':'2','volume-opacity':'.1'};
      global.document={getElementById:id=>els[id]||(els[id]=new Element(defaults[id]||'')),createElement:()=>new Element()};
      function setupOptions(id,values){for(const value of values){const el=new Element(value);document.getElementById(id).appendChild(el);}}
      setupOptions('volume-mode',['cloud','slice','wall','streamlines']);setupOptions('slice-basis',['centerline','pick','x','y','z']);
      document.getElementById('wss-report-meta').textContent=JSON.stringify(META);
      document.getElementById('volume-arrays').textContent=JSON.stringify(DATA);
      global.THREE={WebGLRenderer:class {constructor(){throw new Error('No WebGL');}}};
      global.addEventListener=()=>{};global.requestAnimationFrame=fn=>fn();
      EXTRA
      delete require.cache[require.resolve(VIEWER)];require(VIEWER);
    """.replace("EXTRA", extra)


def test_no_webgl_viewer_hides_optional_panels_without_analysis_data(tmp_path):
    """Old summaries (no profiles / findings / trust / review) still open; the new panels stay hidden."""
    mesh, cloud, center = sample()
    out = tmp_path / "legacy.html"
    build_html(out, {"case_id": "legacy"}, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    script = _stub_prelude() + r"""
      console.log(JSON.stringify({findingsHidden:els['menu-findings'].hidden,profilesHidden:els['menu-profiles'].hidden,trustRowHidden:els['trust-row'].hidden,
        probeHidden:els['probe-card'].hidden,error:els['volume-error'].hidden,count:els['volume-statistics'].children[0].children[1].textContent,
        footer:els['footer-review'].textContent,feature:els['footer-feature'].textContent,note:els['view-direction-note'].textContent,
        legend:els['legend-note'].textContent}));
    """
    script = script.replace("META", json.dumps({"case_id": "legacy"})).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    assert result["findingsHidden"] and result["profilesHidden"] and result["trustRowHidden"] and result["probeHidden"]
    assert result["error"] is False and result["count"] == "3"   # hidden=False: the no-WebGL notice is shown
    assert result["footer"] == "待审阅" and result["feature"] == "未记录"   # hashes moved into 技术信息; v0.15: unsigned = 待审阅
    assert "没有解剖坐标架" in result["note"] and "连续色标" in result["legend"]


def test_no_webgl_viewer_uses_findings_profiles_trust_units_and_view_state(tmp_path):
    mesh, cloud, center = sample()
    cloud["trust"] = [0, 8, 0, 0]
    center["tangent"] = [[0, 0, 1], [0, 0, 1]]
    meta = {
        "case_id": "rich", "run_identity": "abc", "branch_names": {"1": "主动脉"},
        "frame_transform": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "origin_mm": [0, 0, 0], "direction_source": "unknown_stl"},
        "review": {"status": "reviewed", "by": "R1", "at": "2026-09-20T20:00:00"}, "feature_contract": {"source_hash": "0123456789abcdef"},
        "trust": {"bits": {"8": "low_sample_support"}, "fractions": {"low_sample_support": 0.25}},
        "findings": {"items": [
            {"id": "F1", "kind": "pressure_drop", "label": "主动脉压降", "branch": "主动脉", "segment_id": 1, "value": 120.0, "units": "Pa", "xyz_mm": [0, 0, 2], "extent_mm": 3, "severity": "note", "definition": "近端10%与远端10%平均压差"},
            {"id": "F2", "kind": "max_speed", "label": "速度最大值", "branch": "主动脉", "segment_id": 1, "value": 2.0, "units": "m/s", "xyz_mm": [0, 0, 4], "extent_mm": 1, "severity": "attention", "point_indices": [3]},
        ]},
        "profiles": {"bin_mm": 2.0, "branches": [{"segment_id": 1, "name": "主动脉", "s_local_mm": [0, 2, 4], "s_from_root_mm": [0, 2, 4], "radius_mm": [1, 1, 1],
                     "volume": {"speed_mean_m_s": [0.5, 1.0, 2.0], "speed_max_m_s": [0.5, 1.0, 2.0], "pressure_mean_pa": [4, 3, 2], "pressure_min_pa": [4, 3, 2], "n": [1, 1, 1]}}]},
    }
    out = tmp_path / "rich.html"
    build_html(out, meta, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    script = _stub_prelude() + r"""
      const first={findingsHidden:els['menu-findings'].hidden,profilesHidden:els['menu-profiles'].hidden,trustRowHidden:els['trust-row'].hidden,
        findings:els['findings-list'].children.length,firstSev:els['findings-list'].children[0].children[0].textContent,
        footer:els['footer-review'].textContent,feature:els['footer-feature'].textContent,note:els['view-direction-note'].textContent};
      // pressure_drop finding: slice goes to the branch at 10 %
      els['findings-list'].children[1].events.click();
      const drop={basis:els['slice-basis'].value,branch:els['slice-branch'].value,position:els['slice-position'].value,mode:els['volume-mode'].value,hint:els['findings-hint'].textContent};
      // profile click at mid-width -> slice at half the arc length
      els['profile-canvas'].events.click({clientX:44+504*0.5});
      const profile={position:els['slice-position'].value,basis:els['slice-basis'].value};
      // units + bands
      els['velocity-unit'].value='cm/s';els['velocity-unit'].events.change();
      els['color-bands'].value='4';els['color-bands'].events.change();
      const units={legend:els['legend-title'].textContent,max:els['legend-max'].textContent,legendNote:els['legend-note'].textContent};
      // region stats on branch 1 over the whole arc
      const region=els['region-stats'].children.map(c=>c.textContent);
      // trust overlay toggles the legend
      els['trust-overlay'].checked=true;els['trust-overlay'].events.change();
      const trustShown=!els['trust-legend'].hidden, trustText=els['trust-legend-list'].children[0].textContent;
      console.log(JSON.stringify({first,drop,profile,units,region,trustShown,trustText}));
    """
    script = script.replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    first = result["first"]
    assert not first["findingsHidden"] and not first["profilesHidden"] and not first["trustRowHidden"]
    assert first["findings"] == 2 and first["firstSev"] == "关注"                     # attention sorts first
    assert first["footer"].startswith("已审阅 · R1") and first["feature"] == "0123456789abcdef"
    assert "请核对左右" in first["note"]
    assert result["drop"] == {"basis": "centerline", "branch": "1", "position": "10", "mode": "slice", "hint": result["drop"]["hint"]}
    assert "近端 10%" in result["drop"]["hint"]
    assert result["profile"] == {"position": "50.0", "basis": "centerline"}
    assert result["units"]["legend"] == "速度 · cm/s" and result["units"]["max"] == "200" and "4 段" in result["units"]["legendNote"]
    assert result["region"][0] == "体内点数" and result["region"][1] == "3"
    assert result["trustShown"] and result["trustText"].startswith("采样支撑弱 · 25.0%")


# ---------------------------------------------------------------- C-class phase 1 (contract §10–§13): glossary, branch hiding, probe log, findings review, export, messages
def _rich_meta():
    return {
        "case_id": "rich", "run_identity": "abc123def456ghi", "branch_names": {"1": "主动脉"},
        "frame_transform": {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "origin_mm": [0, 0, 0], "direction_source": "unknown_stl"},
        "findings": {"items": [
            {"id": "F1", "kind": "pressure_drop", "label": "主动脉压降", "branch": "主动脉", "segment_id": 1, "value": 120.0, "units": "Pa", "xyz_mm": [0, 0, 2], "extent_mm": 3, "severity": "note", "definition": "近端10%与远端10%平均压差"},
            {"id": "F2", "kind": "max_speed", "label": "速度最大值", "branch": "主动脉", "segment_id": 1, "value": 2.0, "units": "m/s", "xyz_mm": [0, 0, 4], "extent_mm": 1, "severity": "attention", "point_indices": [3]},
        ]},
    }


def test_template_embeds_glossary_and_shared_library_and_uses_known_gloss_keys(tmp_path, monkeypatch):
    from wss_deploy import volume_report
    from wss_deploy.glossary import GLOSSARY
    mesh, cloud, center = sample()
    out = tmp_path / "c.html"
    build_html(out, {"case_id": "g"}, mesh, cloud, center)
    html = out.read_text(encoding="utf-8")
    for leftover in ("__COMMON__", "__GLOSSARY__", "__VIEWER__", "__META__", "__ARRAYS__"):
        assert leftover not in html
    glossary = json.loads(re.search(r'<script id="wss-glossary" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert glossary["schema_version"] == "wss-deploy.glossary/v1" and set(glossary["terms"]) == set(GLOSSARY)
    used = set(re.findall(r'data-gloss="([a-z_]+)"', volume_report.TEMPLATE)) | set(re.findall(r"'(trust_[a-z_]+)'", VIEWER.read_text(encoding="utf-8")))
    assert used and used <= set(GLOSSARY)
    assert {"max_diameter", "narrative"} <= used                       # §17 buttons use the terms §17.1 defines
    for marker in ('id="menu-measure"', 'id="probe-log-body"', 'id="probe-record"', 'id="branch-visibility"', 'id="export-scale"', 'id="export-png"',
                   'id="finding-add"', 'id="findings-review-status"', 'id="gloss-pop"', 'id="footer-identity"', 'id="defaults-save"',
                   'id="measure-modes"', 'id="measure-list"', 'id="menu-annot"', 'id="annot-add"', 'id="menu-presets"', 'id="preset-builtin"', 'id="labels"'):
        assert marker in html
    assert "WssReportCommon" in html                      # the shared library is embedded verbatim
    # phase 2: the shared library is mandatory, a missing file must fail loudly instead of degrading the page
    monkeypatch.setattr(volume_report, "STATIC_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="report_common.js"):
        volume_report._common_js()
    (tmp_path / "report_common.js").write_text("/* lib */", encoding="utf-8")
    assert volume_report._common_js() == "/* lib */"
    assert not hasattr(volume_report, "COMMON_PENDING")


def test_core_branch_hiding_probe_log_export_and_review_helpers():
    result = node_json("""
      const faces=new Uint32Array([0,1,2, 2,3,4, 3,4,5]), labels=new Int32Array([1,1,2, 2,2,1]);
      const kept=Array.from(core.filterFaces(faces,labels,new Set([2])));
      const unchanged=core.filterFaces(faces,labels,new Set())===faces;
      const near=Array.from(core.nearestLabels(new Float32Array([0,0,0, 10,0,0, 0,10,0]),new Int32Array([7,8,9]),new Float32Array([1,0,0, 9,1,0, 0,9,0, 50,50,50])));
      const rows=[{id:'P1',xyz_mm:[1,2,3],branch:'左髂外',s_from_root_mm:12.5,radius_mm:2,values:{speed_m_s:0.5,u:0,v:0,w:0.5,pressure_pa:-12.25},created_at:'t'},{id:'P2',xyz_mm:[0,0,0],branch:'a,b',values:{}}];
      const tsv=core.probeToTSV(rows,'zh'), csv=core.probeToCSV(rows,'en');
      const svg=core.colorbarSVGLocal({colormap:'rainbow',min:0,max:10,bands:4,units:'Pa',title:'压力'});
      const review=core.normalizeReview({items:{F1:{decision:'rejected',note:'x'},F2:{decision:'bogus',note:''},F3:{decision:null,note:'keep'}},added:[{id:'M1',xyz_mm:[1,2,3],text:'t'},{id:'bad'}]});
      const ordered=core.findingsWithReview([{id:'F1',severity:'attention',rank:1},{id:'F2',severity:'note',rank:1}],review).map(x=>x.id);
      console.log(JSON.stringify({kept,unchanged,near,tsvHead:tsv.split('\\n')[0].split('\\t').slice(0,5),tsvRow:tsv.split('\\n')[1].split('\\t'),csvStartsBom:csv.charCodeAt(0)===0xfeff,csvLines:csv.split('\\r\\n').length,csvQuoted:csv.includes('"a,b"'),csvHeaderEn:csv.split('\\r\\n')[0].startsWith('\\ufeffID,x_mm'),
        svgOk:svg.startsWith('<svg')&&(svg.match(/<rect /g)||[]).length===5&&svg.includes('压力 (Pa)'),name:core.exportFilenameLocal({case_id:'LIU YU/MING',view:'front',field:'speed',scale:2}),
        review,ordered,en:[core.labelText('主动脉','en'),core.labelText('左髂内','en'),core.labelText('速度','zh')],gloss:core.TRUST_GLOSS[16]}));
    """)
    assert result["kept"] == [0, 1, 2, 3, 4, 5] and result["unchanged"] is True          # only the all-hidden face (2,3,4) disappears
    assert result["near"] == [7, 8, 9, 8] or result["near"][:3] == [7, 8, 9]
    assert result["tsvHead"] == ["编号", "x_mm", "y_mm", "z_mm", "分支"]
    assert result["tsvRow"][:8] == ["P1", "1", "2", "3", "左髂外", "12.5", "2", "0.5"] and result["tsvRow"][11] == "-12.25"
    assert result["csvStartsBom"] and result["csvLines"] == 4 and result["csvQuoted"] and result["csvHeaderEn"]
    assert result["svgOk"] and result["name"] == "LIU_YU_MING_front_speed_2x.png"
    assert result["review"]["items"] == {"F1": {"decision": "rejected", "note": "x"}, "F3": {"decision": None, "note": "keep"}}
    assert [a["id"] for a in result["review"]["added"]] == ["M1"] and result["review"]["added"][0]["kind"] == "manual"
    assert result["ordered"] == ["F2", "M1", "F1"]                                        # rejected items sink to the end
    assert result["en"] == ["Aorta", "Left IIA", "速度"] and result["gloss"] == "trust_near_opening"


def _hash_state(state):
    return base64.urlsafe_b64encode(json.dumps(state, ensure_ascii=False).encode("utf-8")).decode("ascii").rstrip("=")


def test_no_webgl_viewer_applies_v11_state_hides_branches_logs_probes_and_reviews_findings(tmp_path):
    mesh, cloud, center = sample()
    cloud["s_from_root_mm"] = [0, 2, 2, 4]
    meta = _rich_meta()
    out = tmp_path / "v11.html"
    build_html(out, meta, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    state = {"schema_version": "wss-deploy.view/v1", "family": "volume", "branches_hidden": [1], "lang": "en",
             "export": {"scale": 4, "background": "transparent", "colorbar": "svg", "ui": False},
             "probe_log": [{"id": "P1", "kind": "interior", "index": 1, "xyz_mm": [0, 0, 2], "branch": "主动脉", "segment_id": 1, "s_from_root_mm": 2, "radius_mm": 1.5,
                            "values": {"speed_m_s": 1, "u": 0, "v": 0, "w": 1, "pressure_pa": 2}, "created_at": "2026-09-21T00:00:00Z"}],
             "findings_review": {"items": {"F2": {"decision": "rejected", "note": "artifact"}},
                                 "added": [{"id": "M1", "xyz_mm": [0, 0, 3], "branch": "主动脉", "segment_id": 1, "s_from_root_mm": 3, "text": "瘤囊内低速", "kind": "manual"}]}}
    extra = "global.location={protocol:'file:',hash:'#view=" + _hash_state(state) + "'};global.fetch=undefined;"
    script = _stub_prelude(extra) + r"""
      const list=els['findings-list'].children, row=els['probe-log-body'].children[0];
      const before={branches:els['branch-visibility'].children.length,firstChecked:els['branch-visibility'].children[0].children[0].checked,rowHidden:els['branch-visibility-row'].hidden,
        exportScale:els['export-scale'].value,background:els['export-background'].value,colorbar:els['export-colorbar'].value,hideUi:els['export-hide-ui'].checked,lang:els['export-lang'].value,
        probes:els['probe-log-body'].children.length,count:els['probe-log-count'].textContent,probeId:row.children[0].textContent,probeBranch:row.children[2].textContent,
        findings:list.length,firstText:list[0].children[1].textContent,manualBadge:list[1].children[0].textContent,manualText:list[1].children[1].textContent,lastClass:list[2].className,
        rejectedNote:list[2].children[2].children[3].value};
      list[0].children[2].children[0].events.click();                       // confirm F1 (offline)
      const afterConfirm={status:els['findings-review-status'].textContent,firstClass:els['findings-list'].children[0].className};
      els['probe-log-export'].events.click();const notes={exported:els['probe-log-note'].textContent};els['probe-log-copy'].events.click();
      els['branch-visibility'].children[0].children[0].checked=true;els['branch-visibility'].children[0].children[0].events.change();
      els['view-link'].events.click();
      console.log(JSON.stringify({before,afterConfirm,notes,copyNote:els['probe-log-note'].textContent,hash:global.location.hash}));
    """
    script = script.replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    before = result["before"]
    assert before["branches"] == 1 and before["firstChecked"] is False and before["rowHidden"] is False
    assert (before["exportScale"], before["background"], before["colorbar"], before["hideUi"], before["lang"]) == ("4", "transparent", "svg", True, "en")
    assert before["probes"] == 1 and before["count"] == "1" and before["probeId"] == "P1" and before["probeBranch"] == "主动脉"
    assert before["findings"] == 3 and before["firstText"].startswith("主动脉压降") and before["manualBadge"] == "人工" and before["manualText"] == "瘤囊内低速 · 主动脉"
    assert "rejected" in before["lastClass"] and before["rejectedNote"] == "artifact"
    assert "离线只读" in result["afterConfirm"]["status"] and "confirmed" in result["afterConfirm"]["firstClass"]
    assert "已导出 1 条" in result["notes"]["exported"] and "剪贴板不可用" in result["copyNote"]
    encoded = result["hash"].split("view=", 1)[1]
    saved = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode("utf-8"))
    assert saved["schema_version"] == "wss-deploy.view/v1" and saved["branches_hidden"] == [] and saved["lang"] == "en"
    assert saved["export"] == {"scale": 4, "background": "transparent", "colorbar": "svg", "ui": False}
    assert [r["id"] for r in saved["probe_log"]] == ["P1"] and saved["probe_log"][0]["values"]["pressure_pa"] == 2
    assert saved["findings_review"]["items"]["F1"] == {"decision": "confirmed", "note": ""} and saved["findings_review"]["items"]["F2"]["decision"] == "rejected"
    assert [a["id"] for a in saved["findings_review"]["added"]] == ["M1"]
    assert saved["measurements"] == [] and saved["annotations"] == {"items": []} and saved["preset_name"] is None


def test_no_webgl_viewer_message_protocol_ready_apply_state_and_export(tmp_path):
    mesh, cloud, center = sample()
    out = tmp_path / "msg.html"
    build_html(out, _rich_meta(), mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    extra = r"""
      global.location={origin:'http://h',protocol:'http:',pathname:'/api/jobs/j1/report',hash:''};global.fetch=undefined;
      const listeners={};global.addEventListener=(n,f)=>{listeners[n]=f;};global.listeners=listeners;
      const posted=[];global.posted=posted;global.parent={postMessage:(m,o)=>posted.push({m,o})};
    """
    script = _stub_prelude(extra) + r"""
      const send=(origin,data)=>listeners.message({origin,data});
      const ready=posted[0];
      send('http://evil',{type:'wss-view:apply-state',state:{lang:'en'},request_id:'x'});
      const afterForeign=posted.length;
      send('http://h',{type:'wss-view:apply-state',state:{lang:'en',branches_hidden:[1]},request_id:'r1'});
      send('http://h',{type:'wss-view:apply-state',state:{preset_name:'瘤囊截面系列'},request_id:'r2'});
      send('http://h',{type:'wss-view:export',options:{scale:2},request_id:'r3'});
      send('http://h',{type:'wss-view:set-camera',camera:{position:[0,0,1],target:[0,0,0],up:[0,0,1]}});
      console.log(JSON.stringify({ready,afterForeign,rest:posted.slice(1).map(p=>({type:p.m.type,id:p.m.request_id,message:p.m.message||null,origin:p.o})),lang:els['export-lang'].value,hidden:!els['branch-visibility'].children[0].children[0].checked}));
    """
    script = script.replace("META", json.dumps(_rich_meta(), ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    assert result["ready"]["m"] == {"type": "wss-view:ready", "family": "volume", "run_identity": "abc123def456ghi", "case_id": "rich", "webgl": False}
    assert result["ready"]["o"] == "http://h" and result["afterForeign"] == 1                    # foreign origins are ignored
    assert result["rest"] == [
        {"type": "wss-view:applied", "id": "r1", "message": None, "origin": "http://h"},
        {"type": "wss-view:error", "id": "r2", "message": "预设需要共享库", "origin": "http://h"},
        {"type": "wss-view:error", "id": "r3", "message": "三维视图不可用，无法导出。", "origin": "http://h"},
    ]
    assert result["lang"] == "en" and result["hidden"] is True


# ---------------------------------------------------------------- C-class phase 2 (contract §12): shared library, measurements, annotations, presets
def tube_sample():
    """``sample()`` with a resolvable centreline: a straight 4 mm tube of inscribed radius 1.5 mm along +z."""
    mesh, cloud, center = sample()
    center["radius_mm"] = [1.5, 1.5]
    center["edges"] = [[0, 1]]
    center["tangent"] = [[0, 0, 1], [0, 0, 1]]
    return mesh, cloud, center


def test_centerline_radius_keeps_its_own_key_so_the_probe_radius_stays_per_point(tmp_path):
    mesh, cloud, center = tube_sample()
    cloud["radius_mm"] = [0.0, 1.1, 1.2, 1.3]                 # per prediction point
    out = tmp_path / "radii.html"
    build_html(out, {}, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    np.testing.assert_allclose(np.frombuffer(base64.b64decode(data["radius_mm"]), dtype=np.float32), [0.0, 1.1, 1.2, 1.3], rtol=1e-6)
    np.testing.assert_allclose(np.frombuffer(base64.b64decode(data["center_radius_mm"]), dtype=np.float32), [1.5, 1.5])


def test_no_webgl_viewer_measures_distance_arc_diameter_and_segment_on_a_straight_tube(tmp_path):
    """C7: the handlers are driven with world coordinates because picking needs WebGL in a browser."""
    mesh, cloud, center = tube_sample()
    meta = {"case_id": "tube", "branch_names": {"1": "主动脉"}}
    out = tmp_path / "measure.html"
    build_html(out, meta, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    script = _stub_prelude(common=True) + r"""
      const V=require(VIEWER);
      const made=[V.__test.measure('distance',[[0,0,1],[0,0,3]]),V.__test.measure('arc',[[0,0,1],[0,0,3]]),
                  V.__test.measure('diameter',[[0.4,0,2]]),V.__test.measure('segment',[[0,0,1],[0,0,3]])];
      const partial=(V.__test.setMeasureMode('distance'),V.__test.measurePick([0,0,1]));
      const pending=els['measure-note'].textContent;
      const saved=V.__test.capture().measurements;
      console.log(JSON.stringify({groups:V.__test.centerlineGroups(),
        made:made.map(m=>m&&{id:m.id,kind:m.kind,value:m.value_mm,branch:m.branch,label:m.label,sid:m.segment_id}),
        partial,pending,rows:els['measure-list'].children.length,savedIds:saved.map(m=>m.id),
        en:V.__test.measureLabel(made[1],'en'),tsv:V.__test.measureTSV('zh').split('\n')[1]}));
    """
    script = script.replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    assert result["groups"] == 1                                        # one centreline branch from the shared builder
    kinds = {m["kind"]: m for m in result["made"]}
    assert [m["value"] for m in result["made"]] == [2.0, 2.0, 3.0, 2.0]  # distance / arc / 2 x 1.5 mm radius / segment
    assert all(m["value"] > 0 and m["branch"] == "主动脉" and m["sid"] == 1 for m in result["made"])
    assert [m["id"] for m in result["made"]] == ["D1", "A1", "R1", "S1"]
    assert kinds["diameter"]["label"] == "管径 3.0 mm（主动脉） · 内切半径 × 2"
    assert kinds["distance"]["label"].startswith("直线距离 2.0 mm")
    assert result["partial"] is None and "1/2" in result["pending"]      # the first pick of a pair only stages a point
    assert result["rows"] == 4 and result["savedIds"] == ["D1", "A1", "R1", "S1"]
    assert result["en"] == "Arc distance 2.0 mm (Aorta)"                 # shared English dictionary
    assert result["tsv"].split("\t")[:4] == ["D1", "distance", "2.000", "主动脉"]


def test_no_webgl_viewer_pins_annotations_offline_and_round_trips_them_through_the_view_state(tmp_path):
    """C8: the embedded copy loads offline, new pins land in the view state, applied state replaces them."""
    mesh, cloud, center = tube_sample()
    meta = {"case_id": "tube", "branch_names": {"1": "主动脉"},
            "annotations": {"schema_version": "wss-deploy.annotations/v1",
                            "items": [{"id": "A1", "xyz_mm": [0, 0, 2], "text": "内嵌标注", "branch": "主动脉", "segment_id": 1, "s_from_root_mm": 2}]}}
    out = tmp_path / "annot.html"
    build_html(out, meta, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    script = _stub_prelude(common=True) + r"""
      const V=require(VIEWER);
      const embedded={rows:els['annot-list'].children.length,status:els['annot-status'].textContent,ids:V.__test.annotations().map(a=>a.id)};
      const added=V.__test.addAnnotation([0,0,3],'瘤囊前壁血栓');
      const empty=V.__test.addAnnotation([0,0,3],'   ');
      const captured=V.__test.capture().annotations;
      V.__test.apply({annotations:{items:[{id:'A9',xyz_mm:[0,0,4],text:'来自视图状态'},{id:'bad',xyz_mm:[0,0]}]}});
      const applied=V.__test.annotations();
      const labels=V.__test.labelItems('zh').map(x=>x.kind);
      console.log(JSON.stringify({embedded,added:added&&{id:added.id,branch:added.branch,s:added.s_from_root_mm,text:added.text},empty,
        captured:captured.items.map(a=>[a.id,a.text]),applied:applied.map(a=>[a.id,a.text]),rows:els['annot-list'].children.length,
        status:els['annot-status'].textContent,labels}));
    """
    script = script.replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    assert result["embedded"]["ids"] == ["A1"] and result["embedded"]["rows"] == 1 and "离线只读（内嵌副本）" in result["embedded"]["status"]
    assert result["added"]["id"] == "A2" and result["added"]["branch"] == "主动脉" and result["added"]["text"] == "瘤囊前壁血栓"
    assert result["added"]["s"] is not None and result["empty"] is None            # empty text is refused
    assert result["captured"] == [["A1", "内嵌标注"], ["A2", "瘤囊前壁血栓"]]
    assert result["applied"] == [["A9", "来自视图状态"]] and result["rows"] == 1   # malformed entries are dropped
    assert "离线只读（内嵌副本）" in result["status"] and result["labels"] == ["annot"]


def test_no_webgl_viewer_applies_builtin_and_user_presets(tmp_path):
    """C9: builtin generators come from the shared library; user presets round-trip through localStorage offline."""
    mesh, cloud, center = tube_sample()
    meta = {"case_id": "tube", "branch_names": {"1": "主动脉"},
            "profiles": {"bin_mm": 2.0, "branches": [{"segment_id": 1, "name": "主动脉", "s_local_mm": [0, 2, 4], "s_from_root_mm": [0, 2, 4], "radius_mm": [1.5, 1.5, 1.5],
                         "volume": {"speed_mean_m_s": [0.5, 1, 2], "speed_max_m_s": [0.5, 1, 2], "pressure_mean_pa": [4, 3, 2], "pressure_min_pa": [4, 3, 2], "n": [1, 1, 1]}}]}}
    out = tmp_path / "presets.html"
    build_html(out, meta, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    extra = "global.localStorage={s:{},getItem(k){return this.s[k]===undefined?null:this.s[k];},setItem(k,v){this.s[k]=String(v);}};"
    script = _stub_prelude(extra, common=True) + r"""
      const V=require(VIEWER);
      const names=V.__test.presetNames(), buttons=els['preset-builtin'].children.length;
      const applied=names.map(n=>Boolean(V.__test.applyPreset(n)));
      V.__test.applyPreset('瘤囊截面系列');
      const afterCut={mode:els['volume-mode'].value,basis:els['slice-basis'].value,position:els['slice-position'].value,note:els['preset-note'].textContent};
      V.__test.applyPreset('沿程压降');
      const afterDrop={field:els['volume-field'].value,menuOpen:els['menu-profiles'].open,name:V.__test.capture().preset_name};
      document.getElementById('preset-name').value='我的视图';els['preset-save'].events.click();
      const stored=JSON.parse(global.localStorage.s['wss-report-presets:volume']);
      els['volume-field'].value='velocity';els['volume-field'].events.change();
      const reapplied=Boolean(V.__test.applyPreset('我的视图'));
      const missing=V.__test.applyPreset('不存在的预设');
      console.log(JSON.stringify({names,buttons,applied,afterCut,afterDrop,stored:stored.map(p=>p.name),
        userRows:els['preset-user'].children.length,reapplied,field:els['volume-field'].value,missing,note:els['preset-note'].textContent}));
    """
    script = script.replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    assert result["names"] == ["沿程压降", "流线全貌", "瘤囊截面系列", "临床视图"] and result["buttons"] == 4
    assert result["applied"] == [True, True, True, True]
    assert result["afterCut"] == {"mode": "slice", "basis": "centerline", "position": "40", "note": result["afterCut"]["note"]}
    assert "40% / 60% / 80%" in result["afterCut"]["note"]
    assert result["afterDrop"] == {"field": "pressure", "menuOpen": True, "name": "沿程压降"}
    assert result["stored"] == ["我的视图"] and result["userRows"] == 1
    assert result["reapplied"] and result["field"] == "pressure"          # the saved state restores the field
    assert result["missing"] is None and "没有该预设" in result["note"]


def test_no_webgl_viewer_applies_a_preset_over_the_message_protocol(tmp_path):
    """C9 + C13: ``apply-state {preset_name}`` is expanded against this case instead of failing."""
    mesh, cloud, center = tube_sample()
    meta = dict(_rich_meta(), branch_names={"1": "主动脉"})
    out = tmp_path / "preset_msg.html"
    build_html(out, meta, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    extra = r"""
      global.location={origin:'http://h',protocol:'http:',pathname:'/api/jobs/j1/report',hash:''};global.fetch=undefined;
      const listeners={};global.addEventListener=(n,f)=>{listeners[n]=f;};global.listeners=listeners;
      const posted=[];global.posted=posted;global.parent={postMessage:(m,o)=>posted.push({m,o})};
    """
    script = _stub_prelude(extra, common=True) + r"""
      const V=require(VIEWER);
      const send=(origin,data)=>listeners.message({origin,data});
      send('http://h',{type:'wss-view:apply-state',state:{preset_name:'瘤囊截面系列'},request_id:'p1'});
      const cut={mode:els['volume-mode'].value,position:els['slice-position'].value,name:V.__test.capture().preset_name};
      send('http://h',{type:'wss-view:apply-state',state:{preset_name:'流线全貌',opacity:0.3},request_id:'p2'});
      const over={opacity:els['volume-opacity'].value};
      send('http://h',{type:'wss-view:apply-state',state:{preset_name:'没有这个'},request_id:'p3'});
      console.log(JSON.stringify({rest:posted.slice(1).map(p=>({type:p.m.type,id:p.m.request_id,message:p.m.message||null})),cut,over}));
    """
    script = script.replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    assert result["rest"] == [
        {"type": "wss-view:applied", "id": "p1", "message": None},
        {"type": "wss-view:applied", "id": "p2", "message": None},
        {"type": "wss-view:error", "id": "p3", "message": "没有该预设：没有这个"},
    ]
    assert result["cut"] == {"mode": "slice", "position": "40", "name": "瘤囊截面系列"}
    assert result["over"]["opacity"] == "0.3"          # explicit keys win over the preset's 0.08


def test_no_webgl_viewer_english_labels_and_svg_colorbar_come_from_the_shared_library(tmp_path):
    """C12: legend / colour-bar captions and the standalone SVG go through WssReportCommon."""
    mesh, cloud, center = tube_sample()
    meta = {"case_id": "tube", "branch_names": {"1": "主动脉"}}
    out = tmp_path / "labels.html"
    build_html(out, meta, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    script = _stub_prelude(common=True) + r"""
      const V=require(VIEWER);
      els['volume-field'].value='pressure';els['volume-field'].events.change();
      const zh={legend:els['legend-title'].textContent,svg:V.__test.colorbarSVG('zh')};
      els['export-lang'].value='en';els['export-lang'].events.change();
      const en={legend:els['legend-title'].textContent,svg:V.__test.colorbarSVG('en'),
        field:V.__test.fieldLabel('velocity','en'),branch:V.__test.branchLabel('左髂外','en'),view:V.__test.viewLabel('top','en'),viewZh:V.__test.viewLabel('top','zh')};
      console.log(JSON.stringify({zh:{legend:zh.legend,title:zh.svg.includes('相对压力'),units:zh.svg.includes('(Pa)'),svg:zh.svg.slice(0,4),rects:(zh.svg.match(/<rect/g)||[]).length},
        en:{legend:en.legend,title:en.svg.includes('Relative pressure'),units:en.svg.includes('(Pa)'),field:en.field,branch:en.branch,view:en.view,viewZh:en.viewZh}}));
    """
    script = script.replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    assert result["zh"]["legend"] == "相对压力 · Pa" and result["zh"]["title"] and result["zh"]["units"] and result["zh"]["svg"] == "<svg"
    assert result["zh"]["rects"] >= 2                                  # gradient bar plus its frame
    assert result["en"]["legend"] == "Relative pressure · Pa" and result["en"]["title"] and result["en"]["units"]
    assert result["en"]["field"] == "Speed" and result["en"]["branch"] == "Left EIA"
    assert result["en"]["view"] == "Top" and result["en"]["viewZh"] == "上"


def test_export_composites_measurement_and_annotation_labels_and_shared_probe_columns(tmp_path):
    """C12 / C13: labels ride along with the colour-bar overlay; probe export uses the shared columns."""
    source = VIEWER.read_text(encoding="utf-8")
    export = source[source.index("function renderExport("):source.index("function exportPNG(")]
    assert "drawOverlayLabels(ctx,canvas.width,canvas.height,k,o.lang)" in export and "o.ui!==false" in export
    assert "if(o.colorbar==='svg')message.colorbar_svg=colorbarSvgCurrent(o.lang);" in source   # exported message carries the SVG
    mesh, cloud, center = tube_sample()
    meta = {"case_id": "tube", "branch_names": {"1": "主动脉"}}
    out = tmp_path / "export.html"
    build_html(out, meta, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    script = _stub_prelude(common=True) + r"""
      const V=require(VIEWER);
      V.__test.measure('distance',[[0,0,1],[0,0,3]]);V.__test.addAnnotation([0,0,2],'瘤囊');
      const items=V.__test.labelItems('en').map(x=>({kind:x.kind,text:x.text,lifted:x.xyz[2]>x.anchor[2]}));
      const calls=[];const ctx={fillText:(...a)=>calls.push(a),save(){},restore(){},fillRect(){},strokeRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},measureText:()=>({width:40})};
      V.__test.drawOverlayLabels(ctx,600,400,2,'en');                    // no camera without WebGL: must be a no-op, never a throw
      const tsv=V.__test.probeTSV([{id:'P1',xyz_mm:[0,0,2],branch:'主动脉',segment_id:1,s_from_root_mm:2,radius_mm:1.5,values:{speed_m_s:1,pressure_pa:2}}],'zh');
      console.log(JSON.stringify({items,drawn:calls.length,head:tsv.split('\n')[0].split('\t'),row:tsv.split('\n')[1].split('\t')}));
    """
    script = script.replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    assert [x["kind"] for x in result["items"]] == ["meas", "annot"]
    assert result["items"][0]["text"] == "Distance 2.0 mm (Aorta)" and result["items"][1]["text"] == "瘤囊"
    assert result["items"][1]["lifted"] is True and result["drawn"] == 0
    assert result["head"][:8] == ["编号", "x_mm", "y_mm", "z_mm", "分支", "分支编号", "距入口弧长_mm", "半径_mm"]   # shared probe columns
    assert result["row"][:6] == ["P1", "0", "0", "2", "主动脉", "1"]


def test_no_webgl_viewer_loads_and_saves_annotations_and_user_presets_online(tmp_path):
    """C8 + C9 online paths: GET files/annotations.json, PUT with the CSRF token, 409 = review lock, presets from preferences."""
    mesh, cloud, center = tube_sample()
    meta = {"case_id": "tube", "branch_names": {"1": "主动脉"}}
    out = tmp_path / "online.html"
    build_html(out, meta, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    extra = r"""
      global.location={origin:'http://h',protocol:'http:',pathname:'/api/jobs/j1/report',hash:''};
      const calls=[];global.calls=calls;
      global.fetch=async(url,opts)=>{
        const u=String(url),method=(opts&&opts.method)||'GET';calls.push([u,method,opts&&opts.headers&&opts.headers['X-CSRF-Token']||null]);
        if(u.endsWith('api/session'))return {ok:true,status:200,json:async()=>({csrf_token:'tok-1'})};
        if(u.endsWith('files/annotations.json'))return {ok:true,status:200,json:async()=>({items:[{id:'A1',xyz_mm:[0,0,2],text:'服务端标注'}]})};
        if(u.endsWith('api/preferences'))return {ok:true,status:200,json:async()=>({preferences:{presets:{wall:[{name:'壁面预设'}],volume:[{name:'我的体场',state:{field:'pressure'}}]}}})};
        if(u.endsWith('annotations')&&method==='PUT')return {ok:false,status:409,json:async()=>({error:{message:'结果已审阅锁定'}})};
        return {ok:false,status:404,json:async()=>({})};
      };
    """
    script = _stub_prelude(extra, common=True) + r"""
      const V=require(VIEWER);
      setTimeout(()=>{
        const loaded=V.__test.annotations().map(a=>a.id);
        V.__test.addAnnotation([0,0,3],'新的标注');
        setTimeout(()=>{
          const refused=V.__test.addAnnotation([0,0,1],'锁定后不应写入');
          console.log(JSON.stringify({loaded,userPresets:V.__test.userPresets().map(p=>p.name),rows:els['preset-user'].children.length,
            status:els['annot-status'].textContent,refused,ids:V.__test.annotations().map(a=>a.id),
            put:calls.filter(c=>c[1]==='PUT').map(c=>[c[0].replace(/^.*\/api/,'/api'),c[2]]),
            got:calls.filter(c=>c[1]==='GET').map(c=>c[0].replace(/^.*\/api/,'/api'))}));
        },600);      // the save is debounced 400 ms; the 409 must land before we try again
      },40);
    """
    script = script.replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    result = node_json(script)
    assert result["loaded"] == ["A1"]                                          # server copy wins online
    assert result["userPresets"] == ["我的体场"] and result["rows"] == 1        # presets.volume from preferences
    assert "已审阅锁定，标注只读" in result["status"]
    assert result["refused"] is None and result["ids"] == ["A1", "A2"]          # the pin stays local, no further writes
    assert result["put"] == [["/api/jobs/j1/annotations", "tok-1"]]
    assert "/api/jobs/j1/files/annotations.json" in result["got"] and "/api/preferences" in result["got"]


def test_slice_gizmo_drag_math():
    """Dragging the blue plane: pixel deltas projected on the axis' screen direction, converted to slider units."""
    result = node_json("""
      console.log(JSON.stringify({
        along:core.dragAlong(10,0,[1,0],0.5),           // 10 px along a rightward axis at 0.5 mm/px → 5 mm
        against:core.dragAlong(0,10,[0,-1],0.5),        // dragging down against an upward axis → −5 mm
        diagonal:core.dragAlong(3,4,[3,4],1),           // full projection on the same direction → 5 mm
        degenerate:core.dragAlong(3,4,[0,0],1),
        pick:core.positionStep('pick',4,0,0),           // 0.5 mm per slider unit → 8 units
        centerline:core.positionStep('centerline',5,200,0),   // 5 mm of a 200 mm branch → 2.5 %
        axis:core.positionStep('z',10,0,50),            // 10 mm of a 50 mm extent → 20 %
        zero:core.positionStep('centerline',5,0,0)
      }));
    """)
    assert result["along"] == 5 and result["against"] == -5 and abs(result["diagonal"] - 5) < 1e-9 and result["degenerate"] == 0
    assert result["pick"] == 8 and result["centerline"] == 2.5 and result["axis"] == 20 and result["zero"] == 0


def test_volume_template_carries_slice_gizmo_toolbar():
    from wss_deploy.volume_report import TEMPLATE
    for marker in ('id="slice-tools"', 'id="drag-move"', 'id="drag-rotate"', 'id="drag-offset"', 'id="slice-reset-angle"',
                   'id="pick-toggle-2"', 'id="pick-toggle-menu"', 'id="pick-hint"', '点选定位截面'):
        assert marker in TEMPLATE, marker
    assert '开始点选' not in TEMPLATE


def test_slice_fill_uses_wall_contour_and_no_slip_boundary():
    """A tube cut by a plane: the contour is the lumen circle, cells inside it are filled, the wall pulls speed to 0."""
    result = node_json("""
      // closed-ish tube along z: 24 rings × 24 around, radius 10, z from -20 to 20 (open ends are irrelevant here)
      const R=10,NA=24,NZ=24,verts=[],faces=[];
      for(let iz=0;iz<NZ;iz++)for(let ia=0;ia<NA;ia++){const a=ia/NA*2*Math.PI,z=-20+40*iz/(NZ-1);verts.push(R*Math.cos(a),R*Math.sin(a),z);}
      for(let iz=0;iz<NZ-1;iz++)for(let ia=0;ia<NA;ia++){const a0=iz*NA+ia,a1=iz*NA+(ia+1)%NA,b0=a0+NA,b1=a1+NA;faces.push(a0,a1,b0,a1,b1,b0);}
      const plane={origin:[0,0,1.3],normal:[0,0,1],u:[1,0,0],v:[0,1,0]};
      const wallP=new Float32Array(verts.length/3).fill(5);
      const segs=core.planeContour(new Float32Array(verts),new Uint32Array(faces),plane,wallP,50);
      const radii=segs.map(s=>Math.hypot(s[0],s[1]));
      const inside=core.scanlineInside(segs,[-12,12,-12,12],24,24);
      let count=0;for(const m of inside)count+=m;
      const centre=inside[12*24+12],corner=inside[0];
      // three interior samples with speed 1 near the centre; the wall contributes speed 0
      const filled=core.fillSection([[0,0],[2,0],[0,2]],[1,1,1],segs.map(s=>[(s[0]+s[2])/2,(s[1]+s[3])/2,0]),[-12,12,-12,12],24,24,inside,{directRadius:3,cell:1.5});
      const at=(x,y)=>Math.floor((y+12)/1)*24+Math.floor((x+12)/1);
      console.log(JSON.stringify({nseg:segs.length,rmin:Math.min(...radii),rmax:Math.max(...radii),wallVal:segs[0][4],count,centre,corner,
        vCentre:filled.values[at(0.5,0.5)],vNearWall:filled.values[at(9.5,0.5)],lowCentre:filled.low[at(0.5,0.5)],lowNearWall:filled.low[at(9.5,0.5)],filled:filled.filled,directCells:filled.directCells}));
    """)
    assert result["nseg"] == 48 and abs(result["rmin"] - 10) < 0.2 and abs(result["rmax"] - 10) < 0.2 and result["wallVal"] == 5
    # π·10² ≈ 314 cells of 1 mm²; the scanline count must be close, the centre inside, the corner outside
    assert 290 <= result["count"] <= 330 and result["centre"] == 1 and result["corner"] == 0
    assert result["filled"] == result["count"] and 0 < result["directCells"] < result["filled"]
    assert result["vCentre"] > 0.9 and result["vNearWall"] < 0.2 and result["lowCentre"] == 0 and result["lowNearWall"] == 1


def test_volume_template_has_slice_fill_toggle():
    from wss_deploy.volume_report import TEMPLATE
    assert 'id="slice-fill"' in TEMPLATE and '补全截面' in TEMPLATE and '无滑移' in TEMPLATE


def test_slice_contour_keeps_only_the_local_loop():
    """Two parallel tubes cut by one plane: only the loop around the plane origin is used, the other is dropped."""
    result = node_json("""
      const verts=[],faces=[];const NA=24,NZ=6;
      function tube(cx,cy,R){const base=verts.length/3;for(let iz=0;iz<NZ;iz++)for(let ia=0;ia<NA;ia++){const a=ia/NA*2*Math.PI;verts.push(cx+R*Math.cos(a),cy+R*Math.sin(a),-5+10*iz/(NZ-1));}
        for(let iz=0;iz<NZ-1;iz++)for(let ia=0;ia<NA;ia++){const a0=base+iz*NA+ia,a1=base+iz*NA+(ia+1)%NA,b0=a0+NA,b1=a1+NA;faces.push(a0,a1,b0,a1,b1,b0);}}
      tube(0,0,10);tube(40,0,6);
      const plane={origin:[0,0,0.7],normal:[0,0,1],u:[1,0,0],v:[0,1,0]};
      const segs=core.planeContour(new Float32Array(verts),new Uint32Array(faces),plane,null,100);
      const loops=core.contourLoops(segs);
      const chosen=core.selectLoop(loops,segs,[0,0]);
      const far=core.selectLoop(loops,segs,[40,0]);
      const radii=chosen.segs.map(s=>Math.hypot(s[0],s[1]));
      console.log(JSON.stringify({nseg:segs.length,nloops:loops.length,closed:loops.every(l=>l.closed),sizes:loops.map(l=>l.segments.length).sort((a,b)=>a-b),
        chosenN:chosen.segs.length,chosenInside:chosen.inside,rmin:Math.min(...radii),rmax:Math.max(...radii),farN:far.segs.length,
        inLoop:core.pointInLoop(chosen.segs,3,3),outLoop:core.pointInLoop(chosen.segs,40,0),bar:[core.scaleBarLength(0.1,90),core.scaleBarLength(1,90),core.scaleBarLength(0.02,90)]}));
    """)
    assert result["nseg"] == 96 and result["nloops"] == 2 and result["closed"] is True and result["sizes"] == [48, 48]
    assert result["chosenN"] == 48 and result["chosenInside"] is True and abs(result["rmin"] - 10) < 0.2 and abs(result["rmax"] - 10) < 0.2
    assert result["farN"] == 48 and result["inLoop"] is True and result["outLoop"] is False
    assert result["bar"] == [10, 100, 2]


def test_volume_template_has_slice_zoom_view():
    from wss_deploy.volume_report import TEMPLATE
    for marker in ('id="slice-zoom"', 'id="slice-zoom-canvas"', 'id="slice-zoom-open"', 'id="slice-zoom-png"', 'id="slice-zoom-points"', 'id="slice-fill-note"', '导出 PNG'):
        assert marker in TEMPLATE, marker


def test_slice_loop_selection_ignores_stray_arcs_and_closes_openings():
    """A slit tube (open local chain) next to a far partial arc: the local chain must win even though the
    old ray-parity test called the far arc 'enclosing'; the chain is then closed through the opening."""
    result = node_json("""
      const verts=[],faces=[];const NA=24,NZ=6;
      function ring(cx,cy,R,skipCol){const base=verts.length/3;for(let iz=0;iz<NZ;iz++)for(let ia=0;ia<NA;ia++){const a=ia/NA*2*Math.PI;verts.push(cx+R*Math.cos(a),cy+R*Math.sin(a),-5+10*iz/(NZ-1));}
        for(let iz=0;iz<NZ-1;iz++)for(let ia=0;ia<NA;ia++){if(ia===skipCol)continue;const a0=base+iz*NA+ia,a1=base+iz*NA+(ia+1)%NA,b0=a0+NA,b1=a1+NA;faces.push(a0,a1,b0,a1,b1,b0);}}
      function arc(cx,cy,R,a0,a1){const base=verts.length/3;for(let iz=0;iz<NZ;iz++)for(let ia=0;ia<=NA;ia++){const a=a0+(a1-a0)*ia/NA;verts.push(cx+R*Math.cos(a),cy+R*Math.sin(a),-5+10*iz/(NZ-1));}
        for(let iz=0;iz<NZ-1;iz++)for(let ia=0;ia<NA;ia++){const p0=base+iz*(NA+1)+ia,p1=p0+1,q0=p0+NA+1,q1=p1+NA+1;faces.push(p0,p1,q0,p1,q1,q0);}}
      ring(0,0,10,5);                                   // welded local vessel with one missing column (a slit)
      arc(40,0,6,-Math.PI/3,Math.PI/3);                 // far partial arc facing the origin
      const plane={origin:[0,0,0.7],normal:[0,0,1],u:[1,0,0],v:[0,1,0]};
      const segs=core.planeContour(new Float32Array(verts),new Uint32Array(faces),plane,null,Infinity);
      const loops=core.contourLoops(segs);
      const chosen=core.selectLoop(loops,segs,[0,0]);
      const closed=core.closeChain(chosen.segs);
      const pruned=core.selectLoop(loops,segs,[0,0],5);   // a limit below the radius prunes every non-enclosing chain
      const radii=chosen.segs.map(s=>Math.hypot(s[0],s[1]));
      const inside=core.scanlineInside(closed,[-12,12,-12,12],24,24);let count=0;for(const m of inside)count+=m;
      console.log(JSON.stringify({nloops:loops.length,closedFlags:loops.map(l=>l.closed),chosenClosed:chosen.closed,chosenWraps:chosen.wraps,chosenInside:chosen.inside,
        rmin:Math.min(...radii),rmax:Math.max(...radii),synthetic:closed?closed[closed.length-1][5]:null,nAfter:closed?closed.length:null,nBefore:chosen.segs.length,count,prunedNull:pruned===null}));
    """)
    assert result["nloops"] == 2 and result["closedFlags"] == [False, False]
    assert result["chosenClosed"] is False and result["chosenWraps"] is True and result["chosenInside"] is False
    assert abs(result["rmin"] - 10) < 0.2 and abs(result["rmax"] - 10) < 0.2
    assert result["synthetic"] == -1 and result["nAfter"] == result["nBefore"] + 1 and result["nBefore"] == 46
    assert 290 <= result["count"] <= 330 and result["prunedNull"] is True



def test_compact_layout_switches_on_narrow_windows(tmp_path):
    """Below 1180 px (a compare-page iframe) the body gets the compact class, header buttons take short labels and
    the slice panel body collapses; a wide window keeps the full layout; the stored override forces either."""
    mesh, cloud, center = sample()
    out = tmp_path / "compact.html"
    build_html(out, {"case_id": "compact"}, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    script = _stub_prelude(extra=r"""
      global.innerWidth=WIDTH;global.localStorage={_v:STORED,getItem(k){return k==='wss-report-compact'?this._v:null;},setItem(k,v){if(k==='wss-report-compact')this._v=v;}};
      global.document.body={classList:{_s:new Set(),toggle(c,f){f?this._s.add(c):this._s.delete(c);return f;},contains(c){return this._s.has(c);}}};
    """) + r"""
      console.log(JSON.stringify({compact:global.document.body.classList.contains('compact'),pick:els['pick-toggle'].textContent,fit:els['volume-fit'].textContent,
        layout:els['layout-toggle'].textContent,sliceHidden:els['slice-panel-body'].hidden}));
    """
    script = script.replace("META", json.dumps({"case_id": "compact"})).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    narrow = node_json(script.replace("WIDTH", "975").replace("STORED", "null"))
    wide = node_json(script.replace("WIDTH", "1600").replace("STORED", "null"))
    forced_on = node_json(script.replace("WIDTH", "1600").replace("STORED", "'on'"))
    forced_off = node_json(script.replace("WIDTH", "975").replace("STORED", "'off'"))
    assert narrow == {"compact": True, "pick": "点选", "fit": "复位", "layout": "完整布局", "sliceHidden": True}
    assert wide == {"compact": False, "pick": "点选定位截面", "fit": "复位视角", "layout": "紧凑布局", "sliceHidden": False}
    assert forced_on["compact"] is True and forced_off["compact"] is False


def test_volume_template_has_compact_controls():
    from wss_deploy.volume_report import TEMPLATE
    for marker in ('id="menu-toggle"', 'id="menu-tab"', 'id="menu-close"', 'id="layout-toggle"', 'class="menu-head"', '.compact aside.menu'):
        assert marker in TEMPLATE, marker


# ---------------------------------------------------------------- §15 figures: slice CSV, montages, curve SVG, one-pager snapshots
def tube_case(n_around=24, n_rings=9, radius=10.0, half=20.0):
    """A straight tube along +z whose default centreline slice cuts a closed lumen circle of radius 10 mm."""
    verts = []
    for iz in range(n_rings):
        z = -half + 2 * half * iz / (n_rings - 1)
        for ia in range(n_around):
            a = 2 * np.pi * ia / n_around
            verts.append([radius * np.cos(a), radius * np.sin(a), z])
    faces = []
    for iz in range(n_rings - 1):
        for ia in range(n_around):
            a0, a1 = iz * n_around + ia, iz * n_around + (ia + 1) % n_around
            faces += [[a0, a1, a0 + n_around], [a1, a1 + n_around, a0 + n_around]]
    verts = np.asarray(verts, np.float32)
    mesh = {"vertices": verts, "faces": faces, "pressure_pa": np.full(len(verts), 5.0, np.float32)}
    pts = []
    for z in (-1.0, -0.5, 0.0, 0.5, 1.0, -6.0, 6.0):
        for r in (0.0, 3.0, 6.0):
            for k in range(1 if r == 0 else 8):
                a = 2 * np.pi * k / 8 + 0.37 * z          # distinct in-plane positions per z level
                pts.append([r * np.cos(a), r * np.sin(a), z])
    pts = np.asarray(pts, np.float32)
    vel = np.zeros((len(pts), 3), np.float32)
    vel[:, 2] = 1.0 - (np.hypot(pts[:, 0], pts[:, 1]) / radius) ** 2
    cloud = {"pts": pts, "pressure_pa": (10.0 - 0.2 * pts[:, 2]).astype(np.float32), "velocity_m_s": vel,
             "is_wall": np.zeros(len(pts), np.uint8), "segment": np.ones(len(pts), np.int32)}
    zs = np.linspace(-half, half, 5)
    center = {"xyz": [[0.0, 0.0, float(z)] for z in zs], "segment": [1] * 5, "radius_mm": [radius] * 5,
              "edges": [[i, i + 1] for i in range(4)], "tangent": [[0, 0, 1]] * 5}
    return mesh, cloud, center


def _tube_meta(frame=False, profiles=False):
    meta = {"case_id": "TUBE", "run_identity": "tube-run", "branch_names": {"1": "主动脉"}}
    if frame:
        meta["frame_transform"] = {"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "origin_mm": [0, 0, 0], "direction_source": "unknown_stl"}
    if profiles:
        meta["profiles"] = {"bin_mm": 2.0, "branches": [{
            "segment_id": 1, "name": "主动脉", "s_local_mm": [0, 10, 20, 30, 40], "s_from_root_mm": [0, 10, 20, 30, 40],
            "radius_mm": [10, 10, 10, 10, 10],
            "volume": {"speed_mean_m_s": [0.4, 0.5, 0.6, 0.5, 0.4], "speed_max_m_s": [0.8, 0.9, 1.0, 0.9, 0.8],
                       "pressure_mean_pa": [12, 10, 8, 6, 4], "pressure_min_pa": [10, 8, 6, 4, 2], "n": [3, 3, 3, 3, 3]}}]}
    return meta


def _figure_extra(image=True, extra=""):
    """Canvas / download plumbing on top of ``_stub_prelude``: data URLs, a body, and captured downloads."""
    js = r"""
      const downloads=[];global.downloads=downloads;
      global.waitFor=(ready,done,left=80)=>{if(ready()||left<=0)return done();setTimeout(()=>waitFor(ready,done,left-1),25);};
      const baseCreate=document.createElement;
      document.createElement=function(tag){
        const el=baseCreate(tag);
        el.style={};
        el.setAttribute=function(k,v){this[k]=v;};
        el.toDataURL=()=>'data:image/png;base64,AAAA';
        el.getContext=()=>new Proxy({measureText:()=>({width:12})},{get:(t,k)=>(k in t?t[k]:()=>{})});
        el.click=function(){downloads.push({name:this.download,href:String(this.href||'')});};
        el.remove=()=>{};
        return el;
      };
      document.body=baseCreate('body');
    """
    if image:
        js += r"""
      global.Image=class{constructor(){this.width=700;this.height=620;}set src(v){this._src=v;if(this.onload)setTimeout(()=>this.onload(),0);}get src(){return this._src;}};
        """
    return js + extra


def _run_tube(tmp_path, name, meta, body, extra, common=True):
    mesh, cloud, center = tube_case()
    out = tmp_path / name
    build_html(out, meta, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    script = _stub_prelude(extra, common=common) + body
    script = script.replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    return node_json(script)


def test_slice_grid_rows_and_series_fractions_are_pure_helpers():
    """§15.9 / §15.10 numerics: cell centres of the masked grid, and the station fractions along a branch."""
    result = node_json("""
      const grid={values:[1,NaN,3,4],mask:[1,1,1,0],low:[0,0,1,0],nx:2,ny:2};
      console.log(JSON.stringify({rows:core.sliceGridToRows(grid,[-1,1,-1,1],'Pa'),
        noUnits:core.sliceGridToRows(grid,[-1,1,-1,1]).map(r=>r[3]),
        empty:core.sliceGridToRows(null,[-1,1,-1,1],'Pa'),badBounds:core.sliceGridToRows(grid,[0,0,0,0],'Pa'),
        six:core.seriesFractions(6),four:core.seriesFractions(4),one:core.seriesFractions(1),fallback:core.seriesFractions('x')}));
    """)
    assert result["rows"] == [[-0.5, -0.5, 1, "Pa", 0], [-0.5, 0.5, 3, "Pa", 1]]
    assert result["noUnits"] == ["", ""] and result["empty"] == [] and result["badBounds"] == []
    assert result["six"] == [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    assert result["four"] == [0.05, 0.35, 0.65, 0.95] and result["one"] == [0.5] and result["fallback"] == result["six"]


def test_slice_zoom_exports_csv_and_the_statistics_carry_the_section_geometry(tmp_path):
    """§15.9 + §15.14 tail: the zoom grid becomes a CSV in raw field units, and the closed lumen outline
    adds area / max / equivalent diameter to the slice statistics and the zoom caption."""
    result = _run_tube(tmp_path, "csv.html", _tube_meta(), r"""
      const V=require(VIEWER);
      els['slice-fill'].checked=true;els['slice-fill'].events.change();      // the stub has no HTML defaults
      els['slice-zoom-open'].events.click();
      const stats=els['slice-statistics'].children.map(kv=>[kv.children[0].textContent,kv.children[1].textContent]);
      els['slice-zoom-csv'].events.click();
      const csv=decodeURIComponent(String(downloads[0].href).replace(/^data:text\/csv;charset=utf-8,/,'')).replace(/^﻿/,'');
      const lines=csv.split('\r\n').filter(l=>l.length);
      const comments=lines.filter(l=>l.startsWith('#')), body=lines.filter(l=>!l.startsWith('#'));
      console.log(JSON.stringify({name:downloads[0].name,status:els['slice-zoom-status'].textContent,
        comments,head:body[0],rows:body.length-1,sample:body[1],
        filled:body.slice(1).filter(l=>l.endsWith(',1')).length,direct:body.slice(1).filter(l=>l.endsWith(',0')).length,
        stats,caption:els['slice-zoom-caption'].textContent,section:V.__test.sliceSection()}));
    """, _figure_extra())
    assert result["name"] == "TUBE_slice_speed.csv"
    assert result["head"] == "x_mm,y_mm,value,units,filled_from_wall"
    assert result["rows"] > 1000 and result["filled"] > 0 and result["direct"] > 0
    assert result["sample"].split(",")[3] == "m/s"                       # raw field units, not the display unit
    joined = " | ".join(result["comments"])
    for fragment in ("病例 TUBE", "中心 xyz", "法向", "厚度 2.0 mm", "网格 ", "补全模式 壁面边界补全"):
        assert fragment in joined, fragment
    assert "已导出" in result["status"] and "TUBE_slice_speed.csv" in result["status"]
    labels = [row[0] for row in result["stats"]]
    assert labels == ["预测点数", "均值", "第 99 百分位", "最大值", "轮廓面积", "最大直径", "等效直径"]
    area = float(dict(result["stats"])["轮廓面积"].split()[0])
    assert 300 <= area <= 320                                             # π·10² with a 48-gon outline
    assert 19.0 <= float(dict(result["stats"])["最大直径"].split()[0]) <= 20.1
    assert "轮廓面积" in result["caption"] and "等效直径" in result["caption"]
    assert result["section"]["synthetic"] is False


def test_slice_series_montage_downloads_a_png_and_records_the_stations_in_the_view_state(tmp_path):
    """§15.10: three stations along the branch → one montage PNG, and slice.series in the view state."""
    result = _run_tube(tmp_path, "series.html", _tube_meta(), r"""
      const V=require(VIEWER);
      els['slice-fill'].checked=true;els['slice-fill'].events.change();
      document.getElementById('slice-series-count').value='3';
      els['slice-series-export'].events.click();
      waitFor(()=>downloads.length,()=>{
        const state=V.__test.capture();
        console.log(JSON.stringify({note:els['slice-series-note'].textContent,downloads:downloads.map(d=>d.name),
          series:state.slice.series,branch:els['slice-series-branch'].value,
          applied:(()=>{V.__test.apply({slice:{series:{segment_id:1,fractions:[0.1,0.9]}}});
            return {series:V.__test.sliceSeries(),count:els['slice-series-count'].value,position:els['slice-position'].value};})()}));
      });
    """, _figure_extra())
    assert result["note"].startswith("已导出 3 站") and "TUBE_slices_主动脉_speed.png" in result["note"]
    assert result["downloads"] == ["TUBE_slices_主动脉_speed.png"]
    assert result["series"] == {"segment_id": 1, "fractions": [0.05, 0.5, 0.95]} and result["branch"] == "1"
    # applyView accepts a recorded series without moving the current plane
    assert result["applied"]["series"] == {"segment_id": 1, "fractions": [0.1, 0.9]}
    assert result["applied"]["count"] == "2" and result["applied"]["position"] == "50"


def test_slice_series_montage_fails_soft_when_the_page_cannot_make_images(tmp_path):
    """§15.10 node path: no Image constructor → a status line, never an exception."""
    result = _run_tube(tmp_path, "series_soft.html", _tube_meta(), r"""
      els['slice-series-export'].events.click();
      waitFor(()=>/失败/.test(els['slice-series-note'].textContent),()=>console.log(JSON.stringify({note:els['slice-series-note'].textContent,downloads:downloads.length})));
    """, _figure_extra(image=False))
    assert result["note"].startswith("导出失败：") and result["downloads"] == 0


def test_multi_view_montage_defaults_export_and_view_state(tmp_path):
    """§15.11: 前 / 左 / 当前 / 截面 by default, the selection round-trips through the view state, and the
    montage falls back to the panels that are actually available without WebGL."""
    result = _run_tube(tmp_path, "montage.html", _tube_meta(frame=True), r"""
      const V=require(VIEWER);
      els['slice-fill'].checked=true;els['slice-fill'].events.change();
      const picked=V.__test.montageSelection(), state=V.__test.capture().montage;
      els['montage-export'].events.click();
      waitFor(()=>downloads.length,()=>{
        const after={note:els['montage-status'].textContent,downloads:downloads.map(d=>d.name)};
        V.__test.apply({montage:{views:['top','bottom'],columns:4}});
        console.log(JSON.stringify({picked,state,after,
          applied:V.__test.montageSelection(),columns:els['montage-columns'].value}));
      });
    """, _figure_extra())
    assert result["picked"] == ["front", "left", "current", "slice"]
    assert result["state"] == {"views": ["front", "left", "current", "slice"], "columns": 2}
    assert result["after"]["downloads"] == ["TUBE_montage_speed_1x.png"]
    assert "已导出 1 幅视角拼图" in result["after"]["note"] and "跳过" in result["after"]["note"]
    assert result["applied"] == ["top", "bottom"] and result["columns"] == "4"


def test_profile_curves_export_two_svgs_for_the_current_branch(tmp_path):
    """§15.12: one SVG for the selected quantity in the display unit, one for the radius."""
    result = _run_tube(tmp_path, "profile.html", _tube_meta(profiles=True), r"""
      const V=require(VIEWER);
      els['velocity-unit'].value='cm/s';els['velocity-unit'].events.change();
      const svgs=V.__test.profileSVGs();
      els['profile-svg'].events.click();
      console.log(JSON.stringify({names:Object.keys(svgs).map(k=>svgs[k].name),
        fieldSvg:svgs.field.svg.slice(0,4000),radiusSvg:svgs.radius.svg.slice(0,2000),
        paths:(svgs.field.svg.match(/<path/g)||[]).length,
        note:els['profile-svg-note'].textContent,downloads:downloads.map(d=>d.name)}));
    """, _figure_extra())
    assert result["names"] == ["TUBE_profile_主动脉_speed.svg", "TUBE_profile_主动脉_radius.svg"]
    assert result["downloads"] == result["names"]
    assert "主动脉" in result["fieldSvg"] and "<path" in result["fieldSvg"] and result["paths"] == 2
    assert "距入口弧长 (mm)" in result["fieldSvg"] and "cm/s" in result["fieldSvg"]   # the display unit follows the page
    assert "主动脉" in result["radiusSvg"] and "<path" in result["radiusSvg"] and "mm" in result["radiusSvg"]
    assert "已导出" in result["note"]


def test_onepage_snapshots_post_the_expected_body_and_offer_the_link(tmp_path):
    """§15.8: the online button uploads the figures it could render with the CSRF token and links the one-pager."""
    fetch_js = r"""
      global.location={origin:'http://h',protocol:'http:',pathname:'/api/jobs/j1/report',hash:''};
      const calls=[];global.calls=calls;
      global.fetch=async(url,opts)=>{
        const u=String(url),method=(opts&&opts.method)||'GET';
        calls.push({url:u,method,csrf:opts&&opts.headers&&opts.headers['X-CSRF-Token']||null,body:opts&&opts.body||null});
        if(u.endsWith('api/session'))return {ok:true,status:200,json:async()=>({csrf_token:'tok-1'})};
        if(u.endsWith('snapshots')&&method==='POST')return {ok:true,status:200,json:async()=>({snapshots:{items:[{name:'slice'}]},version:8})};
        return {ok:false,status:404,json:async()=>({})};
      };
    """
    result = _run_tube(tmp_path, "shots.html", _tube_meta(), r"""
      els['slice-fill'].checked=true;els['slice-fill'].events.change();
      els['slice-zoom-open'].events.click();                  // a slice is on screen → the extra figure is included
      els['onepage-shots'].events.click();
      waitFor(()=>calls.some(c=>c.method==='POST'),()=>{
        const post=calls.find(c=>c.method==='POST');
        const body=post?JSON.parse(post.body):null;
        const status=els['onepage-status'];
        console.log(JSON.stringify({url:post&&post.url,csrf:post&&post.csrf,replace:body&&body.replace,
          images:body&&body.images.map(i=>({name:i.name,view:i.view,width:i.width,height:i.height,caption:i.caption,png:i.png_base64})),
          text:status.textContent,link:status.children.map(c=>[c.textContent,c.href])}));
      });
    """, _figure_extra(extra=fetch_js))
    assert result["url"] == "/api/jobs/j1/snapshots" and result["csrf"] == "tok-1" and result["replace"] is True
    assert result["images"] == [{"name": "slice", "view": "slice", "width": 1400, "height": 1200,
                                 "caption": "截面 · 速度 · m/s", "png": "AAAA"}]
    assert "已生成 1 张一页纸配图" in result["text"]
    assert result["link"] == [["打开一页纸", "/api/jobs/j1/onepage"]]


def test_get_state_replies_with_the_full_view_state_and_a_display_subset_leaves_the_slice_alone(tmp_path):
    """§15.6: get-state → wss-view:state; apply-state with only the display subset never touches camera or plane."""
    fetch_js = r"""
      global.location={origin:'http://h',protocol:'http:',pathname:'/api/jobs/j1/report',hash:''};global.fetch=undefined;
      const listeners={};global.addEventListener=(n,f)=>{listeners[n]=f;};global.listeners=listeners;
      const posted=[];global.posted=posted;global.parent={postMessage:(m,o)=>posted.push(m)};
    """
    result = _run_tube(tmp_path, "getstate.html", _tube_meta(), r"""
      const send=(origin,data)=>listeners.message({origin,data});
      els['slice-position'].value='30';els['slice-thickness'].value='5';els['slice-position'].events.input();
      send('http://h',{type:'wss-view:get-state',request_id:'g1'});
      const reply=posted.find(m=>m.type==='wss-view:state');
      send('http://evil',{type:'wss-view:get-state',request_id:'g2'});
      const foreign=posted.filter(m=>m.type==='wss-view:state').length;
      send('http://h',{type:'wss-view:apply-state',request_id:'a1',state:{field:'pressure',mode:'cloud',colormap:'turbo',
        bands:6,log:false,range:{mode:'case',min:0,max:1},pressure_units:'mmHg',speed_units:'cm/s',opacity:0.4,overlay:{trust:true}}});
      console.log(JSON.stringify({reply:reply&&{type:reply.type,id:reply.request_id,family:reply.family,
          keys:Object.keys(reply.state).sort(),camera:reply.state.camera,units:[reply.state.pressure_units,reply.state.speed_units]},
        foreign,applied:posted.filter(m=>m.type==='wss-view:applied').map(m=>m.request_id),
        after:{field:els['volume-field'].value,colormap:els['colormap'].value,bands:els['color-bands'].value,
          pressureUnit:els['pressure-unit'].value,speedUnit:els['velocity-unit'].value,opacity:els['volume-opacity'].value,
          trust:els['trust-overlay'].checked,position:els['slice-position'].value,thickness:els['slice-thickness'].value}}));
    """, fetch_js, common=False)
    reply = result["reply"]
    assert reply["type"] == "wss-view:state" and reply["id"] == "g1" and reply["family"] == "volume"
    assert reply["camera"] is None and reply["units"] == ["Pa", "m/s"]
    for key in ("field", "mode", "colormap", "bands", "range", "slice", "montage", "pressure_units", "speed_units"):
        assert key in reply["keys"], key
    assert result["foreign"] == 1                                            # foreign origins are ignored
    assert result["applied"] == ["a1"]
    after = result["after"]
    assert after["field"] == "pressure" and after["colormap"] == "turbo" and after["bands"] == "6"
    assert after["pressureUnit"] == "mmHg" and after["speedUnit"] == "cm/s" and after["trust"] is True
    assert after["opacity"] == "0.4"
    assert after["position"] == "30" and after["thickness"] == "5"           # no slice key → the plane stays put


def test_volume_template_carries_the_section15_controls():
    from wss_deploy.volume_report import TEMPLATE
    for marker in ('id="slice-zoom-csv"', 'id="slice-zoom-status"', 'id="slice-series-branch"', 'id="slice-series-count"',
                   'id="slice-series-export"', 'id="slice-series-note"', 'id="montage-views"', 'id="mv-front"', 'id="mv-slice"',
                   'id="montage-columns"', 'id="montage-export"', 'id="montage-status"', 'id="onepage-shots"', 'id="onepage-status"',
                   'id="profile-svg"', 'id="profile-svg-note"', '导出 CSV', '导出系列拼图', '导出多视角拼图', '导出曲线 SVG', '生成一页纸配图'):
        assert marker in TEMPLATE, marker


# ---------------------------------------------------------------- §17 doctor-facing wave: auto labels, max diameter, narrative
def _morphology_meta():
    """Trimmed ``summary["morphology"]`` exactly as §17.1 / §17.3 hand it to the report."""
    return {
        "schema_version": "wss-deploy.morphology/v1", "station_mm": 1.0,
        "method": {"max_diameter": "环上最大 Feret 直径"},
        "aorta": {
            "segment_id": 1, "name": "主动脉", "length_mm": 40.0, "reference_diameter_mm": 18.2,
            "max": {"max_diameter_mm": 63.4, "equivalent_diameter_mm": 58.9, "min_diameter_mm": 52.0,
                    "area_mm2": 2725.0, "s_from_root_mm": 20.0, "distance_from_inlet_mm": 20.0,
                    "xyz_mm": [0.0, 0.0, 0.0],
                    "polygon_world": [[10, 0, 0], [0, 10, 0], [-10, 0, 0], [0, -10, 0]]},
            "sac": {"present": True, "s_start_mm": 10.0, "s_end_mm": 30.0, "length_mm": 20.0, "volume_ml": 96.3},
            "neck": {"present": True, "length_mm": 10.0, "diameter_mean_mm": 19.1}},
        "lumen_volume_ml": 158.2, "lumen_volume_method": "capped_mesh_divergence",
        "branches": [{"segment_id": 1, "name": "主动脉", "length_mm": 40.0, "tortuosity": 1.02,
                      "stations": {"s_from_root_mm": [0, 10, 20, 30, 40],
                                   "max_diameter_mm": [20.0, 30.0, 63.4, 30.0, 20.0],
                                   "equivalent_diameter_mm": [18.0, 28.0, 58.9, 28.0, 18.0]}}],
        "notes": ["瘤样扩张：最大等效直径 ≥ 1.5 × 参考直径"],
    }


def _narrative_meta(edited=None):
    doc = {"schema_version": "wss-deploy.narrative/v1", "generated_at": "2026-09-22T09:00:00+08:00",
           "zh": ["主动脉最大直径 63.4 mm（截面最大 Feret 直径），位于入口下 20 mm。",
                  "主动脉近远端压差 120 Pa，最大速度 2.0 m/s 位于主动脉。",
                  "以上为固定收缩期单帧预测的参考描述，非诊断结论。"],
           "en": ["Maximum aortic diameter 63.4 mm, 20 mm below the inlet.",
                  "Aortic pressure drop 120 Pa; maximum speed 2.0 m/s."],
           "edited": edited, "edited_by": "R1" if edited else None, "edited_at": None}
    return doc


def _labels_meta(**extra):
    meta = dict(_rich_meta(), case_id="TUBE", run_identity="tube-run")
    meta.update(extra)
    return meta


def test_automatic_labels_pin_findings_and_branches_and_round_trip_through_the_view_state(tmp_path):
    """§17.3: `labels:{findings,branches}` drives the 3-D chips; rejected findings and hidden branches lose theirs."""
    meta = _labels_meta()
    result = _run_tube(tmp_path, "labels17.html", meta, r"""
      const V=require(VIEWER);
      const initial={state:V.__test.labels(),kinds:V.__test.labelItems('zh').map(x=>x.kind)};
      V.__test.setLabels({findings:3,branches:true});
      const on=V.__test.labelItems('zh');
      const en=V.__test.labelItems('en').filter(x=>x.kind==='flabel'||x.kind==='blabel').map(x=>x.text);
      const captured=V.__test.capture().labels;
      V.__test.apply({labels:{findings:0,branches:false}});
      const off=V.__test.labelItems('zh').map(x=>x.kind);
      V.__test.apply({labels:captured});
      const back={state:V.__test.labels(),count:V.__test.labelItems('zh').length};
      V.__test.apply({});                                            // tolerant: no labels key keeps the chips
      const kept=V.__test.labels();
      V.__test.apply({findings_review:{items:{F2:{decision:'rejected'}}}});
      const rejected=V.__test.labelItems('zh').filter(x=>x.kind==='flabel').map(x=>x.text);
      V.__test.apply({findings_review:{items:{}},branches_hidden:[1]});
      const hidden=V.__test.labelItems('zh').map(x=>x.kind);
      console.log(JSON.stringify({initial,on:on.map(x=>({kind:x.kind,text:x.text,lifted:x.kind==='flabel'?x.xyz[2]>x.anchor[2]:null})),
        en,captured,off,back,kept,rejected,hidden,select:els['labels-findings'].value,branchBox:els['labels-branches'].checked}));
    """, "")
    assert result["initial"] == {"state": {"findings": 0, "branches": False, "max_diameter": True}, "kinds": []}
    kinds = [x["kind"] for x in result["on"]]
    assert kinds == ["flabel", "flabel", "blabel"]                     # both findings (attention first) + one branch
    assert [x["text"] for x in result["on"]] == ["F2 最大速度 2.00 m/s", "F1 压降 120 Pa", "主动脉"]
    assert result["on"][0]["lifted"] is True                           # the chip floats above its anchor
    assert result["en"] == ["F2 Max speed 2.00 m/s", "F1 Pressure drop 120 Pa", "Aorta"]
    assert result["captured"] == {"findings": 3, "branches": True, "max_diameter": True}
    assert result["off"] == [] and result["back"] == {"state": result["captured"], "count": 3}
    assert result["kept"] == result["captured"]                        # apply({}) leaves the labels alone
    assert result["rejected"] == ["F1 压降 120 Pa"]                    # a rejected finding is never labelled
    assert result["hidden"] == []                                      # C10: hidden branch → no chips at all
    assert result["select"] == "3" and result["branchBox"] is True


def test_automatic_labels_honour_a_count_the_menu_does_not_offer(tmp_path):
    """Tolerant apply: an out-of-menu count is kept and gets its own option instead of being dropped."""
    from wss_deploy.volume_report import TEMPLATE as TEMPLATE_HTML
    result = _run_tube(tmp_path, "labels_n.html", _labels_meta(), r"""
      const V=require(VIEWER);
      V.__test.apply({labels:{findings:1}});
      const one={count:V.__test.labelItems('zh').filter(x=>x.kind==='flabel').length,value:els['labels-findings'].value,
        options:els['labels-findings'].children.map(o=>o.value)};
      V.__test.apply({labels:{findings:'x'}});
      const bad=V.__test.labels().findings;
      V.__test.apply({labels:{findings:999}});
      const clamped=V.__test.labels().findings;
      console.log(JSON.stringify({one,bad,clamped}));
    """, "")
    assert result["one"]["count"] == 1 and result["one"]["value"] == "1"
    assert result["one"]["options"] == ["0", "1"]                       # the unknown count is appended once
    assert '<option value="3">前 3</option>' in TEMPLATE_HTML and '<option value="10">前 10</option>' in TEMPLATE_HTML
    assert result["bad"] == 0 and result["clamped"] == 50


def test_clinical_preset_turns_on_the_labels_and_the_velocity_point_cloud(tmp_path):
    """§17.3 「临床视图」 comes from the shared library; the report only has to honour the `labels` key."""
    result = _run_tube(tmp_path, "clinical.html", _labels_meta(), r"""
      const V=require(VIEWER);
      const names=V.__test.presetNames();
      const partial=V.__test.applyPreset('临床视图');
      console.log(JSON.stringify({names,partial:partial&&{mode:partial.mode,field:partial.field,labels:partial.labels},
        labels:V.__test.labels(),mode:els['volume-mode'].value,field:els['volume-field'].value,colormap:els['colormap'].value,
        chips:V.__test.labelItems('zh').map(x=>x.kind),name:V.__test.capture().preset_name}));
    """, "")
    assert "临床视图" in result["names"]
    assert result["partial"] == {"mode": "cloud", "field": "velocity", "labels": {"findings": 5, "branches": True}}
    assert result["labels"] == {"findings": 5, "branches": True, "max_diameter": True}
    assert result["mode"] == "cloud" and result["field"] == "velocity" and result["colormap"] == "rainbow"
    assert result["chips"] == ["flabel", "flabel", "blabel"] and result["name"] == "临床视图"


def test_max_diameter_marker_row_ring_label_and_profile_series(tmp_path):
    """§17.3: the 沿程 row, the closed ring, the station diameters on the radius sub-plot and in the SVG."""
    meta = _labels_meta(morphology=_morphology_meta(), **{k: v for k, v in _tube_meta(profiles=True).items() if k == "profiles"})
    result = _run_tube(tmp_path, "maxdia.html", meta, r"""
      const V=require(VIEWER);
      const row={hidden:els['profile-morphology'].hidden,text:els['max-diameter-text'].textContent,
        ringBox:els['labels-max-diameter'].checked,flyDisabled:els['max-diameter-fly'].disabled};
      const marker=V.__test.morphMax();
      const ring=V.__test.labelItems('zh').filter(x=>x.kind==='dlabel').map(x=>x.text);
      const en=V.__test.labelItems('en').filter(x=>x.kind==='dlabel').map(x=>x.text);
      const series=V.__test.morphSeries(0);
      const svgs=V.__test.profileSVGs();
      V.__test.apply({labels:{max_diameter:false}});
      const off={chips:V.__test.labelItems('zh').filter(x=>x.kind==='dlabel').length,box:els['labels-max-diameter'].checked};
      console.log(JSON.stringify({row,marker,ring,en,series,off,
        radius:svgs&&svgs.radius&&{name:svgs.radius.name,max:svgs.radius.svg.includes('最大直径'),
          equiv:svgs.radius.svg.includes('等效直径'),radius:svgs.radius.svg.includes('半径'),svg:svgs.radius.svg.slice(0,4)}}));
    """, "")
    assert result["row"]["hidden"] is False
    assert result["row"]["text"] == "最大直径 63.4 mm（等效 58.9 mm） · 入口下 20.0 mm"
    assert result["row"]["ringBox"] is True and result["row"]["flyDisabled"] is False
    assert result["marker"] == {"max_diameter_mm": 63.4, "ring": True, "xyz": [0, 0, 0]}
    assert result["ring"] == ["最大直径 63.4 mm"] and result["en"] == ["Max diameter 63.4 mm"]
    assert result["series"]["max"] == [20.0, 30.0, 63.4, 30.0, 20.0] and result["series"]["offset_mm"] == 0
    assert result["series"]["equiv"][2] == 58.9
    assert result["radius"]["svg"] == "<svg" and result["radius"]["max"] and result["radius"]["equiv"] and result["radius"]["radius"]
    assert result["radius"]["name"].endswith("_radius.svg")
    assert result["off"] == {"chips": 0, "box": False}


def test_max_diameter_row_and_ring_stay_hidden_without_morphology(tmp_path):
    """Old summaries (no morphology) open unchanged: the row is hidden and no ring label exists."""
    result = _run_tube(tmp_path, "nomorph.html", _labels_meta(**{k: v for k, v in _tube_meta(profiles=True).items() if k == "profiles"}), r"""
      const V=require(VIEWER);
      const svgs=V.__test.profileSVGs();
      console.log(JSON.stringify({hidden:els['profile-morphology'].hidden,marker:V.__test.morphMax(),
        series:V.__test.morphSeries(0),chips:V.__test.labelItems('zh').filter(x=>x.kind==='dlabel').length,
        radiusOnly:svgs&&svgs.radius?!svgs.radius.svg.includes('最大直径'):null}));
    """, "")
    assert result["hidden"] is True and result["marker"] is None and result["series"] is None
    assert result["chips"] == 0 and result["radiusOnly"] is True


def test_narrative_block_shows_the_auto_summary_or_the_reviewed_edit(tmp_path):
    """§17.2: read-only 「结论（参考）」 at the top of 统计与口径; `edited` wins and is marked as such."""
    auto = _run_tube(tmp_path, "narr_auto.html", _labels_meta(narrative=_narrative_meta()), r"""
      const V=require(VIEWER);
      const zh={hidden:els['narrative-card'].hidden,edited:els['narrative-edited'].hidden,
        lines:els['narrative-body'].children.map(p=>p.textContent)};
      els['export-lang'].value='en';els['export-lang'].events.change();
      const en=els['narrative-body'].children.map(p=>p.textContent);
      console.log(JSON.stringify({zh,en}));
    """, "")
    assert auto["zh"]["hidden"] is False and auto["zh"]["edited"] is True      # no reviewer edit → the badge stays hidden
    assert len(auto["zh"]["lines"]) == 3 and auto["zh"]["lines"][0].startswith("主动脉最大直径 63.4 mm")
    assert auto["en"] == ["Maximum aortic diameter 63.4 mm, 20 mm below the inlet.",
                          "Aortic pressure drop 120 Pa; maximum speed 2.0 m/s."]

    edited = _run_tube(tmp_path, "narr_edit.html", _labels_meta(narrative=_narrative_meta(edited="审阅后结论：瘤体稳定。\n建议随访。")), r"""
      require(VIEWER);
      console.log(JSON.stringify({hidden:els['narrative-card'].hidden,badge:els['narrative-edited'].hidden,
        lines:els['narrative-body'].children.map(p=>p.textContent)}));
    """, "")
    assert edited["hidden"] is False and edited["badge"] is False
    assert edited["lines"] == ["审阅后结论：瘤体稳定。", "建议随访。"]

    absent = _run_tube(tmp_path, "narr_none.html", _labels_meta(), r"""
      require(VIEWER);
      console.log(JSON.stringify({hidden:els['narrative-card'].hidden,
        lines:document.getElementById('narrative-body').children.length}));
    """, "")
    assert absent == {"hidden": True, "lines": 0}


def test_volume_template_and_viewer_carry_the_section17_controls_and_label_compositing():
    from wss_deploy.volume_report import TEMPLATE
    for marker in ('id="labels-findings"', 'id="labels-branches"', 'id="labels-max-diameter"', 'id="profile-morphology"',
                   'id="max-diameter-text"', 'id="max-diameter-fly"', 'id="narrative-card"', 'id="narrative-body"',
                   'id="narrative-edited"', 'data-gloss="narrative"', 'data-gloss="max_diameter"', '自动标注发现', '分支名',
                   '显示最大直径环', '飞到', '结论（参考）', '审阅人已修改', '#labels div.flabel', '#labels div.blabel'):
        assert marker in TEMPLATE, marker
    source = VIEWER.read_text(encoding="utf-8")
    # the chips ride the shared compositing path (PNG export, montage panels, one-pager snapshots)
    overlay = source[source.index("const OVERLAY_STYLE="):source.index("function drawOverlayLabels(")]
    for key in ("flabel_attention", "flabel_note", "flabel_info", "flabel_manual", "blabel", "dlabel"):
        assert key in overlay, key
    render = source[source.index("function layoutLabelsNow("):source.index("const OVERLAY_STYLE=")]
    assert "activateFinding(" in render and "labelClass(p.item)" in render           # clicking a chip opens the finding
    assert "const labelClass=item=>item.kind==='flabel'?`flabel ${item.severity||'note'}`" in source
    assert "THREE.LineLoop" in source and "depthTest:false" in source               # §17.3 closed ring


# ---------------------------------------------------------------- v0.12 (§19.9 / A9 / B4): default view, shortcuts, footer, banner, formatting
def _rot(axis, degrees):
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


# A tilted anatomical frame: rows of R are the patient axes (x = left, y = back, z = towards the inlet) in world space.
FRAME_R = _rot([0.3, -0.5, 0.8], 37.0)


def _frame_meta(**extra):
    meta = {"case_id": "TUBE", "run_identity": "0123456789abcdef-run", "branch_names": {"1": "主动脉"},
            "frame_transform": {"rotation": FRAME_R.tolist(), "origin_mm": [0, 0, 0], "direction_source": "unknown_stl"}}
    meta.update(extra)
    return meta


def _webgl_extra(extra=""):
    """A three.js stand-in so the viewer's WebGL start-up runs under Node: a real camera / controls / vector, every
    other object an inert proxy.  Document and window keydown / resize listeners are captured for dispatch."""
    return r"""
      const docEvents={},winEvents={};global.docEvents=docEvents;global.winEvents=winEvents;
      document.addEventListener=(n,f)=>{(docEvents[n]=docEvents[n]||[]).push(f);};
      document.removeEventListener=(n,f)=>{docEvents[n]=(docEvents[n]||[]).filter(g=>g!==f);};
      global.addEventListener=(n,f)=>{(winEvents[n]=winEvents[n]||[]).push(f);};
      global.requestAnimationFrame=()=>0;
      Element.prototype.setAttribute=function(k,v){this.attrs=this.attrs||{};this.attrs[k]=String(v);};
      Element.prototype.getAttribute=function(k){return this.attrs?this.attrs[k]:undefined;};
      Element.prototype.remove=function(){this.removed=true;};
      Object.defineProperty(Element.prototype,'style',{get(){return this._style||(this._style={});},set(v){this._style=v;},configurable:true});
      document.body=new Element();document.documentElement=document.body;
      document.getElementById('volume-view').clientWidth=900;document.getElementById('volume-view').clientHeight=600;
      Element.prototype.dispatchEvent=function(ev){const f=this.events[ev.type];if(f)f(ev);return true;};
      for(const [id,lo,hi] of [['slice-position',0,100],['slice-pitch',-85,85],['slice-yaw',-85,85],['slice-thickness',0.2,12],['slice-offset-u',-30,30],['slice-offset-v',-30,30]]){const el=document.getElementById(id);el.min=String(lo);el.max=String(hi);}
      document.getElementById('streamline-thin').checked=true;   // tube geometry needs real vertex counts
      document.getElementById('tech-info').hidden=true;           // as in the template
      const any=()=>new Proxy(function(){}, {get:(t,k)=>k===Symbol.toPrimitive?(()=>0):k==='length'?0:k==='then'?undefined:any(),apply:()=>any(),construct:()=>any(),set:()=>true});
      class V3{constructor(x=0,y=0,z=0){this.x=x;this.y=y;this.z=z;}set(x,y,z){this.x=x;this.y=y;this.z=z;return this;}toArray(){return [this.x,this.y,this.z];}
        clone(){return new V3(this.x,this.y,this.z);}project(){this.z=9;return this;}distanceTo(o){return Math.hypot(this.x-o.x,this.y-o.y,this.z-o.z);}}
      class Camera{constructor(fov,aspect){this.fov=fov;this.aspect=aspect;this.position=new V3();this.up=new V3(0,1,0);}updateProjectionMatrix(){}}
      class Controls{constructor(cam){this.object=cam;this.target=new V3();this.enabled=true;this.listeners={};this.upAtBuild=cam.up.toArray();Controls.built=(Controls.built||0)+1;global.__controls=this;}
        update(){}addEventListener(n,f){(this.listeners[n]=this.listeners[n]||[]).push(f);}dispose(){this.disposed=true;}fire(n){for(const f of this.listeners[n]||[])f({type:n});}}
      global.THREE=new Proxy({PerspectiveCamera:Camera,Vector3:V3,OrbitControls:Controls},{get:(t,k)=>k in t?t[k]:any()});
      global.key=(k,opts={})=>{const ev={key:k,target:opts.target||{tagName:'BODY'},shiftKey:!!opts.shift,ctrlKey:false,metaKey:false,altKey:false,defaultPrevented:false,preventDefault(){this.defaultPrevented=true;}};
        for(const f of (docEvents.keydown||[]).slice())f(ev);for(const f of (winEvents.keydown||[]).slice())f(ev);return ev;};
      global.fire=(n)=>{for(const f of (winEvents[n]||[]).slice())f({type:n});};
    """ + extra


def _fit_expected(mesh_vertices, name, aspect=1.5):
    """Reference camera for a standard view: WssReportCommon.fitView driven from Python-side expectations."""
    dirs = {"front": ([0, 1, 0], [0, 0, 1]), "left": ([-1, 0, 0], [0, 0, 1]), "top": ([0, 0, -1], [0, -1, 0])}
    d, u = dirs[name]
    world = lambda v: (np.asarray(v, float) @ FRAME_R).tolist()   # dirFromAligned: sum_k v_k R[k]
    return world(d), world(u)


def test_v012_pure_helpers_format_frame_release_time_and_warnings():
    result = node_json(r"""
      const out={
        fmt:[core.formatNumber(0.00177),core.formatNumber(1.572),core.formatNumber(17),core.formatNumber(15544.8),core.formatNumber(null),core.formatNumber(-1404.97)],
        frame:core.frameText({label:'peak_systole',step:1162,time_s:0.21}),frameEn:core.frameText({label:'peak_systole',time_s:0.21},'en'),
        frameUnknown:core.frameText({label:'whatever'}),frameNone:core.frameText(null),
        release:core.releaseShort('PF6_VF6_peak_3seed_20260920'),releaseX5D:core.releaseShort('X5D_v51_5seed_20260916'),releasePlain:core.releaseShort('custom'),
        offset:core.withOffset('2026-09-20 02:15:34','2026-09-20T02:13:18-07:00'),keep:core.withOffset('2026-09-20T17:42:39+08:00','x'),naive:core.withOffset('2026-09-20 17:42:39',''),
        label:[core.referenceLabel('geometry.主动脉.radius_max_mm'),core.referenceLabel('cloud.spacing_mm')],
        none:core.warningItems({reference_assessment:{status:'pass',checks:[{path:'cloud.spacing_mm',value:0.5,min:0.4,max:0.6,status:'pass'}]}}),
        items:core.warningItems({reference_assessment:{status:'review',checks:[
            {path:'geometry.主动脉.radius_max_mm',units:'mm',value:55.31,min:9.223,max:50.306,status:'review'},
            {path:'cloud.spacing_mm',units:'mm',value:0.5,min:0.4,max:0.6,status:'pass'}]},quality:{level:'review',label:'存在不确定性，建议复核',reasons:['3 个模型离散度偏高']}}),
      };
      out.text=core.warningText(out.items);out.empty=core.warningText([]);
      console.log(JSON.stringify(out));
    """)
    assert result["fmt"] == ["0.0018", "1.57", "17.0", "15545", "—", "-1405"]   # never 1.77e-3
    assert result["frame"] == "收缩期峰值帧（约 0.21 s）" and result["frameEn"] == "Peak-systolic frame (≈ 0.21 s)"
    assert result["frameUnknown"] == "固定预测时相" and result["frameNone"] == "固定预测时相"
    assert result["release"] == "PF6_VF6_peak" and result["releaseX5D"] == "X5D_v51" and result["releasePlain"] == "custom"   # the wall report's rule
    assert result["offset"] == "2026-09-20T02:15:34-07:00" and result["keep"] == "2026-09-20T17:42:39+08:00" and result["naive"] == "2026-09-20T17:42:39"
    assert result["label"] == ["主动脉最大半径", "点云点间距"]
    assert result["none"] == [] and result["empty"] == ""
    assert [x["kind"] for x in result["items"]] == ["reference", "quality"]
    assert result["items"][0]["text"] == "主动脉最大半径 55.3 mm（参考 9.22–50.3 mm）"
    assert result["text"] == "注意：输入几何有 1 项超出发布包参考范围（主动脉最大半径 55.3 mm，参考 9.22–50.3 mm）；模型集成质量：存在不确定性，建议复核。预测可信度可能下降，请结合详情复核。"


def test_v012_fitted_standard_views_use_the_anatomical_frame_and_fill_the_viewport():
    """fittedStandardCamera = WssReportCommon.fitView along the frame's front direction; every vertex is inside the
    frustum and the tightest one touches the 1.12 margin (the view fills the viewport)."""
    mesh, _, _ = tube_case()
    verts = (np.asarray(mesh["vertices"], float) + [3, -2, 5]).tolist()
    result = node_json("require(" + json.dumps(str(COMMON)) + ");\n" + r"""
      const frame=core.frameFromMeta({rotation:__ROT__,origin_mm:[0,0,0]});
      const pts=__VERTS__.flat();
      const out={};
      for(const name of ['front','left','top'])out[name]=core.fittedStandardCamera(name,frame,pts,{fov:42,aspect:1.5,margin:1.12});
      out.legacy=core.fittedStandardCamera('legacy',null,pts,{fov:42,aspect:1.5});
      out.noFit=core.fittedStandardCamera('front',frame,pts,{fitView:'none',center:[1,2,3],distance:10});
      delete globalThis.WssReportCommon.fitView;
      out.fallback=core.fittedStandardCamera('front',frame,pts,{center:[1,2,3],distance:10});
      console.log(JSON.stringify(out));
    """.replace("__ROT__", json.dumps(FRAME_R.tolist())).replace("__VERTS__", json.dumps(verts)))
    pts = np.asarray(verts)
    tan = np.tan(np.radians(21.0))
    for name in ("front", "left", "top"):
        cam = result[name]
        d_expect, u_expect = _fit_expected(pts, name)
        pos, tgt, up = (np.asarray(cam[k]) for k in ("position", "target", "up"))
        d = (tgt - pos) / np.linalg.norm(tgt - pos)
        np.testing.assert_allclose(d, d_expect, atol=1e-6)                          # camera → target = anatomical direction
        np.testing.assert_allclose(up, u_expect, atol=1e-6)
        r = np.cross(d, up)
        rel = pts - pos
        depth = rel @ d
        ndc_x = np.abs(rel @ r) / (depth * tan * 1.5)
        ndc_y = np.abs(rel @ up) / (depth * tan)
        worst = max(ndc_x.max(), ndc_y.max())
        assert worst <= 1 / 1.12 + 1e-6 and worst >= 1 / 1.12 - 0.02, (name, worst)   # inside, and filling to the margin
    front_d = np.asarray(result["front"]["target"]) - np.asarray(result["front"]["position"])
    assert np.dot(front_d / np.linalg.norm(front_d), FRAME_R[1]) > 0.999          # looks from the patient's front to the back
    assert result["legacy"]["up"] == [0, 0, 1]
    # without the shared library: the old fixed distance along the same direction
    fb = result["fallback"]
    np.testing.assert_allclose(np.asarray(fb["target"]) - np.asarray(fb["position"]), 10 * FRAME_R[1], atol=1e-6)


def test_v0154_camera_is_framed_to_the_area_the_right_dock_and_slice_toolbar_leave_free(tmp_path):
    """Compare-page halves: a tall right dock (legend + slice plan view) shifts the image centre left by half its width
    and the standard view is fitted to the free area; the slice toolbar in the top-left corner pushes it down.  A short
    dock (legend only) leaves the full viewport."""
    meta = _frame_meta()
    extra = _webgl_extra(r"""
      global.__cams=[];const baseCam=global.THREE.PerspectiveCamera;
      baseCam.prototype.setViewOffset=function(...a){this.viewOffset=a;};baseCam.prototype.clearViewOffset=function(){this.viewOffset=null;};
      const rect=(l,t,w,h)=>({left:l,top:t,right:l+w,bottom:t+h,width:w,height:h});
      const view=document.getElementById('volume-view');view.getBoundingClientRect=()=>rect(0,0,900,600);
      global.__legend={hidden:false,getBoundingClientRect:()=>rect(600,10,290,70)};
      global.__plan={hidden:false,getBoundingClientRect:()=>rect(600,90,290,480)};
      document.getElementById('right-dock').children=[__legend,__plan];
      const bar=document.getElementById('slice-tools');bar.getBoundingClientRect=()=>rect(10,10,280,40);
    """)
    body = r"""
      const V=require(VIEWER),T=V.__test,cam=()=>__controls.object;
      const dist=c=>Math.hypot(...c.position.map((v,k)=>v-c.target[k]));
      const out={};
      document.getElementById('slice-tools').hidden=false;fire('resize');out.both={offset:cam().viewOffset,d:dist(T.camera())};
      document.getElementById('slice-tools').hidden=true;fire('resize');out.dock={offset:cam().viewOffset,d:dist(T.camera())};
      __plan.hidden=true;fire('resize');out.short={offset:cam().viewOffset,d:dist(T.camera())};
      console.log(JSON.stringify(out));
    """
    result = _run_tube(tmp_path, "dock.html", meta, body, extra, common=True)
    assert result["both"]["offset"] == [900, 600, 150, -27, 900, 600]   # dock 300 px wide → +150; toolbar bottom 50 + 4 → −27
    assert result["dock"]["offset"] == [900, 600, 150, 0, 900, 600]
    assert result["short"]["offset"] is None                             # legend only: 70 px of 600 is not a strip
    # the free area is narrower / lower than the viewport, so the fitted camera stands further back
    assert result["both"]["d"] >= result["dock"]["d"] >= result["short"]["d"]
    assert result["both"]["d"] > result["short"]["d"] * 1.05


def test_v0155_unpinned_probe_is_one_line_of_key_readings_and_pinned_lists_all(tmp_path):
    """A hover (unpinned) probe marks the card ``brief`` and only the active quantity + branch rows as ``key`` (the CSS
    shows just those, so the slice plan view keeps its room); pinning drops ``brief`` and keeps every row."""
    meta = _frame_meta()
    extra = _webgl_extra(r"""
      Object.defineProperty(Element.prototype,'classList',{get(){return this._cl||(this._cl={_s:new Set(),
        toggle(c,f){const on=f===undefined?!this._s.has(c):!!f;if(on)this._s.add(c);else this._s.delete(c);return on;},contains(c){return this._s.has(c);}});},configurable:true});
    """)
    body = r"""
      const V=require(VIEWER),T=V.__test,g=id=>document.getElementById(id);
      const rows=()=>g('probe-body').children.map(kv=>({label:kv.children[0].textContent,key:/\bkey\b/.test(kv.className)}));
      document.getElementById('volume-field').value='velocity';
      T.setProbe('interior',0,false);const hover={brief:g('probe-card').classList.contains('brief'),hidden:g('probe-card').hidden,rows:rows(),note:g('probe-note').textContent};
      T.setProbe('interior',0,true);const pinned={brief:g('probe-card').classList.contains('brief'),rows:rows(),note:g('probe-note').textContent};
      console.log(JSON.stringify({hover,pinned}));
    """
    result = _run_tube(tmp_path, "probe.html", meta, body, extra, common=True)
    hover, pinned = result["hover"], result["pinned"]
    assert hover["brief"] is True and hover["hidden"] is False and "单击血管固定" in hover["note"]
    keys = [r["label"] for r in hover["rows"] if r["key"]]
    assert keys == ["速度大小", "分支"] and len(hover["rows"]) > len(keys)       # the other rows stay in the DOM, hidden by CSS
    assert pinned["brief"] is False and [r["label"] for r in pinned["rows"]] == [r["label"] for r in hover["rows"]]
    assert "已固定" in pinned["note"]


def test_v012_webgl_startup_frames_the_front_view_and_explicit_cameras_win(tmp_path):
    """Default view = fitted anatomical front view (inlet up); standard views and 0 / R refit; a resize refits
    until the user moves the camera; a view-state camera (#view= link) always wins; changing up rebuilds the controls."""
    meta = _frame_meta()
    body = r"""
      const V=require(VIEWER),T=V.__test,C=require(COMMON);
      const verts=V.__test.fittedCamera?null:null;
      const out={initial:T.camera(),auto:T.autoView(),built:THREE.OrbitControls.built,upAtBuild:__controls.upAtBuild};
      key('3');out.left=T.camera();out.autoLeft=T.autoView();
      key('5');out.top=T.camera();out.builtAfterTop=THREE.OrbitControls.built;out.topUpAtBuild=__controls.upAtBuild;
      key('r');out.reset=T.camera();out.autoReset=T.autoView();
      document.getElementById('volume-view').clientWidth=200;fire('resize');out.resized=T.camera();
      __controls.fire('start');out.autoAfterUser=T.autoView();
      document.getElementById('volume-view').clientWidth=900;fire('resize');out.notRefit=T.camera();
      console.log(JSON.stringify(out));
    """
    result = _run_tube(tmp_path, "fit.html", meta, body.replace("COMMON", json.dumps(str(COMMON))), _webgl_extra(), common=True)
    init = result["initial"]
    d = np.asarray(init["target"]) - np.asarray(init["position"])
    assert np.dot(d / np.linalg.norm(d), FRAME_R[1]) > 0.999 and np.allclose(init["up"], FRAME_R[2], atol=1e-6)
    assert result["auto"] == "front" and np.allclose(result["upAtBuild"], FRAME_R[2], atol=1e-6)   # controls built with the head-up axis
    dl = np.asarray(result["left"]["target"]) - np.asarray(result["left"]["position"])
    assert np.dot(dl / np.linalg.norm(dl), -FRAME_R[0]) > 0.999 and result["autoLeft"] == "left"
    assert result["builtAfterTop"] == result["built"] + 1 and np.allclose(result["topUpAtBuild"], -FRAME_R[1], atol=1e-6)
    assert result["autoReset"] == "front" and np.allclose(result["reset"]["position"], init["position"], atol=1e-6)
    # narrower viewport → the fitted distance grows (the view is refitted), then a user orbit stops the refits
    dist = lambda c: np.linalg.norm(np.asarray(c["position"]) - np.asarray(c["target"]))
    assert dist(result["resized"]) > dist(init) * 1.2
    assert result["autoAfterUser"] is None and np.allclose(result["notRefit"]["position"], result["resized"]["position"])

    state = {"schema_version": "wss-deploy.view/v1", "family": "volume", "camera": {"position": [100, 0, 0], "target": [0, 0, 0], "up": [0, 0, 1]}}
    hash_extra = _webgl_extra("global.location={hash:'#view=" + _hash_state(state) + "',protocol:'file:',pathname:'/x/report.html'};")
    body2 = r"""
      const V=require(VIEWER);console.log(JSON.stringify({cam:V.__test.camera(),auto:V.__test.autoView()}));
    """
    linked = _run_tube(tmp_path, "fit_hash.html", meta, body2, hash_extra, common=True)
    assert linked["cam"]["position"] == [100, 0, 0] and linked["cam"]["target"] == [0, 0, 0] and linked["auto"] is None


def test_v012_shortcuts_switch_fields_tabs_labels_probe_and_leave_slice_keys_alone(tmp_path):
    meta = _frame_meta(findings={"items": [{"id": "F1", "kind": "max_speed", "label": "最大速度", "branch": "主动脉", "segment_id": 1,
                                            "value": 1.0, "units": "m/s", "xyz_mm": [0, 0, 0], "severity": "attention"}]})
    mesh, cloud, center = tube_case()
    lines = [{"points": [[0, 0, -5], [0, 0, 5]], "speed_m_s": [1.0, 1.0]}]
    out = tmp_path / "keys.html"
    build_html(out, meta, mesh, cloud, center, lines)
    data = arrays_from_html(out.read_text())
    body = r"""
      const V=require(VIEWER),T=V.__test,mode=()=>els['volume-mode'].value,field=()=>els['volume-field'].value;
      const r={};
      r.bindings=T.shortcuts().bindings.map(b=>b.keys.join('/'));
      key('b');r.b=field();key('v');r.v=field();
      const cycle=[];for(let i=0;i<5;i++){key('t');cycle.push(mode()+':'+field());}r.cycle=cycle;
      r.labels0=T.labels();key('l');r.labels1=T.labels();key('l');r.labels2=T.labels();
      key('p');r.probeOff=T.probeEnabled();r.note=els['view-note'].textContent;key('p');r.probeOn=T.probeEnabled();
      key('x');r.pick=T.pickMode();key('x');r.pickOff=T.pickMode();
      const typing=key('b',{target:{tagName:'INPUT',type:'text'}});r.typing=[field(),typing.defaultPrevented];
      // slice page: arrows stay with the slice handler (not consumed by the shortcut layer)
      T.setMode('slice');const before=els['slice-position'].value;const arrow=key('ArrowUp');
      r.arrow={before,after:els['slice-position'].value,mode:mode()};
      const help=key('?');r.help={prevented:help.defaultPrevented,overlay:document.body.children.some(c=>c.className==='wss-shortcuts'&&!c.removed)};
      const esc=key('Escape');r.escClosed=!document.body.children.some(c=>c.className==='wss-shortcuts'&&!c.removed);
      r.sliceAfterEsc=els['slice-position'].value;
      // with the zoom view open nothing but ? / Esc fires
      T.sliceZoom(true);key('b');r.zoomBlocked=field();T.sliceZoom(false);
      console.log(JSON.stringify(r));
    """
    script = _stub_prelude(_webgl_extra(), common=True) + body
    script = script.replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    r = node_json(script)
    for keys in ("1", "2", "3", "4", "5", "6", "0/r", "v", "b", "t", "l", "p", "s", "x", "?/shift+/"):
        assert keys in r["bindings"], keys
    assert r["b"] == "pressure" and r["v"] == "velocity"
    assert r["cycle"] == ["slice:velocity", "wall:pressure", "streamlines:velocity", "cloud:velocity", "slice:velocity"]
    assert r["labels0"]["findings"] == 0 and r["labels1"] == {"findings": 5, "branches": True, "max_diameter": True}
    assert r["labels2"]["findings"] == 0 and r["labels2"]["branches"] is False
    assert r["probeOff"] is False and "探针已关闭" in r["note"] and r["probeOn"] is True
    assert r["pick"] is True and r["pickOff"] is False
    assert r["typing"] == ["velocity", False]                                    # typing in an input never triggers
    assert float(r["arrow"]["after"]) > float(r["arrow"]["before"]) and r["arrow"]["mode"] == "slice"
    assert r["help"] == {"prevented": True, "overlay": True} and r["escClosed"] is True
    assert r["sliceAfterEsc"] == r["arrow"]["after"]                               # Esc closed the help only
    assert r["zoomBlocked"] == "velocity"


def test_v012_wall_and_streamline_tabs_switch_the_field_or_explain_missing_data(tmp_path):
    mesh, cloud, center = tube_case()
    lines = [{"points": [[0, 0, -5], [0, 0, 5]], "speed_m_s": [1.0, 1.0]}]
    body = r"""
      const V=require(VIEWER),r={};
      r.initial={field:els['volume-field'].value,wall:els['mode-wall'].disabled,lines:els['mode-streamlines'].disabled};
      els['mode-wall'].events.click();r.wall={field:els['volume-field'].value,mode:els['volume-mode'].value};
      els['mode-streamlines'].events.click();r.lines={field:els['volume-field'].value,mode:els['volume-mode'].value};
      els['volume-field'].value='pressure';els['volume-field'].events.change();r.fallback=els['volume-mode'].value;
      r.titles={wall:(els['mode-wall'].attrs||{}).title,lines:(els['mode-streamlines'].attrs||{}).title};
      console.log(JSON.stringify(r));
    """
    extra = "Element.prototype.setAttribute=function(k,v){this.attrs=this.attrs||{};this.attrs[k]=String(v);};"
    out = tmp_path / "tabs.html"
    build_html(out, {"case_id": "tabs"}, mesh, cloud, center, lines)
    script = (_stub_prelude(extra) + body).replace("META", json.dumps({"case_id": "tabs"})).replace("DATA", json.dumps(arrays_from_html(out.read_text()))).replace("VIEWER", json.dumps(str(VIEWER)))
    r = node_json(script)
    assert r["initial"] == {"field": "velocity", "wall": False, "lines": False}     # the old report greyed 壁面压力 out here
    assert r["wall"] == {"field": "pressure", "mode": "wall"} and r["lines"] == {"field": "velocity", "mode": "streamlines"}
    assert r["fallback"] == "cloud" and "自动切到压力" in r["titles"]["wall"]
    # a pressure-only report without wall pressure or streamlines: both tabs stay disabled and say why
    mesh2 = {k: v for k, v in mesh.items() if k != "pressure_pa"}
    cloud2 = {k: v for k, v in cloud.items() if k != "velocity_m_s"}
    out2 = tmp_path / "tabs2.html"
    build_html(out2, {"case_id": "tabs"}, mesh2, cloud2, center)
    script2 = (_stub_prelude(extra) + body).replace("META", json.dumps({"case_id": "tabs"})).replace("DATA", json.dumps(arrays_from_html(out2.read_text()))).replace("VIEWER", json.dumps(str(VIEWER)))
    r2 = node_json(script2)
    assert r2["initial"]["wall"] is True and r2["initial"]["lines"] is True
    assert "没有壁面压力" in r2["titles"]["wall"] and "没有流线" in r2["titles"]["lines"]
    assert r2["wall"]["mode"] == "cloud" and r2["lines"]["mode"] == "cloud"


def test_v012_footer_keeps_review_release_time_and_moves_hashes_into_the_tech_popover(tmp_path):
    meta = _frame_meta(created_at="2026-09-20 02:15:34", release="PF6_VF6_peak_3seed_20260920",
                       model_release={"registry_id": "PF6_VF6_peak_3seed_20260920", "fingerprint": "e210ca93e4f3aaaa"},
                       feature_contract={"source_hash": "5512fad4c80b6968c7c0", "version": "wss-features/1.0"},
                       input_sha256="ad233228972ea2dae7e5", model_frame={"target": "peak_systole", "step": 1162, "time_s": 0.21, "label": "peak_systole"},
                       audit={"mapping_history": [{"at": "2026-09-20T02:13:18-07:00"}], "report_rebuilt_at": "2026-09-22T22:52:20+08:00"},
                       review={"status": "reviewed", "by": "R1", "at": "2026-09-22T21:05:00+08:00", "note": "已核对"})
    body = r"""
      const V=require(VIEWER),T=V.__test,r={};
      r.footer={review:els['footer-review'].textContent,release:els['footer-release'].textContent,time:els['footer-time'].textContent,
        releaseTitle:(els['footer-release'].attrs||{}).title,sub:els['volume-subtitle'].textContent};
      r.hiddenBefore=els['tech-info'].hidden;
      els['tech-info-toggle'].events.click({stopPropagation(){}});
      r.open={hidden:els['tech-info'].hidden,expanded:els['tech-info-toggle'].attrs['aria-expanded'],rows:els['tech-info-rows'].children.map(c=>c.textContent)};
      r.text=T.techInfoText();
      key('Escape');r.afterEsc=els['tech-info'].hidden;
      els['tech-info-menu'].events.click({stopPropagation(){}});r.fromMenu=els['tech-info'].hidden;
      console.log(JSON.stringify(r));
    """
    extra = "process.env.TZ='Asia/Shanghai';" + _webgl_extra()
    r = _run_tube(tmp_path, "footer.html", meta, body, extra, common=True)
    f = r["footer"]
    assert f["review"] == "已审阅 · R1 · 2026-09-22 21:05"
    assert f["release"] == "PF6_VF6_peak" and f["releaseTitle"] == "PF6_VF6_peak_3seed_20260920"
    assert f["time"] == "生成 2026-09-20 17:15"          # naive 02:15 written at −07:00 → Shanghai wall clock
    assert f["sub"] == "TUBE · PF6_VF6_peak · 收缩期峰值帧（约 0.21 s）" and "peak_systole" not in f["sub"]
    assert r["hiddenBefore"] is True and r["open"]["hidden"] is False and r["open"]["expanded"] == "true"
    rows = r["open"]["rows"]
    assert "wss-features/1.0 · 5512fad4c80b6968c7c0" in rows and "0123456789abcdef-run" in rows and "PF6_VF6_peak_3seed_20260920" in rows
    assert "peak_systole · step 1162 · 0.21 s" in rows and "ad233228972ea2dae7e5" in rows and "e210ca93e4f3aaaa" in rows
    assert "run_identity\t0123456789abcdef-run" in r["text"] and "报告重建\t2026-09-22 22:52:20（2026-09-22T22:52:20+08:00）" in r["text"]
    assert r["afterEsc"] is True and r["fromMenu"] is False


def test_v012_warning_banner_lists_out_of_range_geometry_and_closes(tmp_path):
    ra = {"status": "review", "note": "仅检查已声明的几何参考范围。", "checks": [
        {"path": "geometry.主动脉.radius_max_mm", "units": "mm", "value": 55.31, "min": 9.223, "max": 50.306, "status": "review"},
        {"path": "geometry.右髂内.length_mm", "units": "mm", "value": 121.4, "min": 31.149, "max": 114.839, "status": "review"},
        {"path": "cloud.spacing_mm", "units": "mm", "value": 0.5, "min": 0.4, "max": 0.6, "status": "pass"}]}
    body = r"""
      const V=require(VIEWER),r={};
      r.shown=!els['warn-banner'].hidden;r.text=els['warn-banner-text'].textContent;r.list=els['reference-list'].children.map(c=>c.textContent);
      r.card=!els['reference-card'].hidden;r.status=els['reference-status'].textContent;
      els['warn-banner-more'].events.click();r.menu=els['menu-stats'].open;
      els['warn-banner-close'].events.click();r.closed=els['warn-banner'].hidden;
      console.log(JSON.stringify(r));
    """
    r = _run_tube(tmp_path, "banner.html", _tube_meta() | {"reference_assessment": ra}, body, "", common=True)
    assert r["shown"] and r["text"] == "注意：输入几何有 2 项超出发布包参考范围（主动脉最大半径 55.3 mm，参考 9.22–50.3 mm 等）。预测可信度可能下降，请结合详情复核。"
    assert r["list"] == ["主动脉最大半径 55.3 mm（参考 9.22–50.3 mm）", "右髂内长度 121 mm（参考 31.1–115 mm）"]
    assert r["card"] and r["status"].startswith("部分几何测量超出") and r["menu"] is True and r["closed"] is True
    ok = _run_tube(tmp_path, "banner_ok.html", _tube_meta() | {"reference_assessment": {**ra, "status": "pass", "checks": ra["checks"][2:]}}, body, "", common=True)
    assert ok["shown"] is False and ok["list"] == [] and ok["status"].startswith("输入几何在本发布包声明的参考范围内")
    legacy = _run_tube(tmp_path, "banner_none.html", _tube_meta(), body, "", common=True)
    assert legacy["shown"] is False and legacy["card"] is False                  # old summaries: nothing to show


def test_v012_numbers_never_use_scientific_notation_in_legend_stats_slice_map_and_colorbars(tmp_path):
    mesh, cloud, center = tube_case()
    vel = np.zeros_like(cloud["velocity_m_s"])
    vel[:, 2] = np.linspace(0.00177, 1.572, len(vel))
    cloud["velocity_m_s"] = vel
    out = tmp_path / "numbers.html"
    build_html(out, {"case_id": "N"}, mesh, cloud, center)
    body = r"""
      const V=require(VIEWER),r={};
      r.legend=[els['legend-min'].textContent,els['legend-max'].textContent];
      r.stats=els['volume-statistics'].children.map(c=>c.children[1].textContent);
      r.svg=Array.from(V.__test.colorbarSVG('zh').matchAll(/<text class="tick"[^>]*>([^<]*)<\/text>/g)).map(m=>m[1]);
      V.__test.setMode('slice');
      r.texts=texts.slice();
      texts.length=0;V.__test.setSliceDisplay({mode:'global'});r.globalTexts=texts.slice();
      console.log(JSON.stringify(r));
    """
    extra = r"""
      const texts=[];global.texts=texts;
      const rec=new Proxy({measureText:()=>({width:10}),fillText:(t)=>texts.push(String(t))},{get:(t,k)=>k in t?t[k]:()=>{},set:()=>true});
      document.getElementById('slice-canvas').getContext=()=>rec;
    """
    script = (_stub_prelude(extra, common=True) + body).replace("META", json.dumps({"case_id": "N"})).replace("DATA", json.dumps(arrays_from_html(out.read_text()))).replace("VIEWER", json.dumps(str(VIEWER)))
    r = node_json(script)
    assert r["legend"] == ["0.0018", "1.57"]
    assert all("e-" not in s and "e+" not in s for s in r["stats"] + r["svg"] + r["texts"])
    assert r["svg"][0] == "0.0018" and r["svg"][-1] == "1.57"                    # the shared colour bar ticks are rewritten too
    assert all("e-" not in s and "e+" not in s for s in r["globalTexts"])
    assert "0.0018" in r["globalTexts"] and "1.57" in r["globalTexts"]              # slice map colour-bar ends (全局)


def test_v012_template_gloss_circles_one_line_subtitle_footer_and_banner_markup():
    from wss_deploy.volume_report import TEMPLATE
    assert "button.gloss:not(.chip){width:16px!important;height:16px!important" in TEMPLATE and "border-radius:50%!important" in TEMPLATE
    assert "固定时相 ?" not in TEMPLATE and '<div class="subline"><span id="volume-subtitle"' in TEMPLATE
    footer = TEMPLATE[TEMPLATE.index("<footer>"):TEMPLATE.index("</footer>")]
    assert 'id="footer-review"' in footer and 'id="footer-release"' in footer and 'id="footer-time"' in footer and 'id="tech-info-toggle"' in footer
    assert "footer-feature" not in footer and "footer-identity" not in footer          # hashes live in the popover
    pop = TEMPLATE[TEMPLATE.index('id="tech-info"'):]
    assert 'id="footer-feature"' in pop and 'id="footer-identity"' in pop
    assert 'id="warn-banner"' in TEMPLATE and 'id="warn-banner-close"' in TEMPLATE and 'id="reference-card"' in TEMPLATE
    assert 'id="shortcuts-help"' in TEMPLATE and 'id="tech-info-menu"' in TEMPLATE
    assert 'class="gloss chip" data-gloss="relative_pressure"' in TEMPLATE


# ---------------------------------------------------------------- 用户试用反馈 7: slice colour range, log scale, in-plane arrows, through-plane velocity
def test_f7_pure_scales_ranges_log_mapping_arrows_and_through_plane():
    result = node_json(r"""
      const ramp=Array.from({length:100},(_,i)=>i);
      const r={
        robust:core.robustRange(ramp),few:core.robustRange([1,2,3]),flat:core.robustRange(Array(20).fill(0.04)),withNaN:core.robustRange(ramp.concat([NaN,Infinity])),
        sym:core.symmetricRange(ramp.map(v=>v%2?-v:v)),zeros:core.symmetricRange(Array(20).fill(0)),
        endsLog:core.scaleEnds({min:0,max:1,log:true}),endsFloor:core.scaleEnds({min:0,max:0.1,log:true}),endsKeep:core.scaleEnds({min:0.05,max:1,log:true}),
        tLo:core.scaleT(0.001,{min:0,max:1,log:true}),tHi:core.scaleT(1,{min:0,max:1,log:true}),tMid:core.scaleT(Math.sqrt(0.005),{min:0,max:1,log:true}),
        tLin:core.scaleT(0.25,{min:0,max:1}),inv:core.scaleValueAt(0.5,{min:0,max:1,log:true}),
        bwrLow:core.scaleColor(-1,{min:-1,max:1,diverging:true}),bwrMid:core.scaleColor(0,{min:-1,max:1,diverging:true}),nan:core.scaleColor(NaN,{min:0,max:1}),
        normal:Array.from(core.throughPlane(new Float32Array([0,0,1, 0,0,-2, 1,0,0]),[0,1],[0,0,2])).map(v=>Number.isNaN(v)?null:v),
        inplane:core.inPlane([1,2,3],{u:[1,0,0],v:[0,1,0]}),
      };
      const items=[];for(let i=0;i<40;i++)for(let j=0;j<40;j++){const x=-1+2*(i+.5)/40,y=-1+2*(j+.5)/40;items.push({x,y,du:-y,dv:x});}
      const a120=core.arrowSamples(items,[-1,1,-1,1],120),a300=core.arrowSamples(items,[-1,1,-1,1],300);
      r.arrows={n120:a120.arrows.length,n300:a300.arrows.length,ref:a120.ref,
        consistent:a120.arrows.every(a=>Math.abs(a.du+a.y)<1e-12&&Math.abs(a.dv-a.x)<1e-12&&Math.abs(a.mag-Math.hypot(a.du,a.dv))<1e-12),
        cells:new Set(a120.arrows.map(a=>Math.floor((a.x+1)/0.2)+'_'+Math.floor((a.y+1)/0.2))).size};
      r.none=core.arrowSamples([],[-1,1,-1,1],120);
      console.log(JSON.stringify(r));
    """)
    assert result["robust"]["min"] == pytest.approx(1.98) and result["robust"]["max"] == pytest.approx(97.02) and result["robust"]["count"] == 100
    assert result["few"] is None and result["withNaN"]["count"] == 100
    assert result["flat"]["min"] < 0.04 < result["flat"]["max"]                            # lo < hi protection
    assert result["sym"]["min"] == pytest.approx(-97.02) and result["sym"]["max"] == pytest.approx(97.02) and result["sym"]["diverging"]
    assert result["zeros"] is None
    # log: lower end = max(min, max / 200, 1e-3)
    assert result["endsLog"] == pytest.approx([0.005, 1]) and result["endsFloor"] == pytest.approx([0.001, 0.1]) and result["endsKeep"] == pytest.approx([0.05, 1])
    assert result["tLo"] == 0 and result["tHi"] == 1 and result["tMid"] == pytest.approx(0.5) and result["tLin"] == pytest.approx(0.25)
    assert result["inv"] == pytest.approx(np.sqrt(0.005))
    assert result["bwrLow"] == pytest.approx([33 / 255, 102 / 255, 172 / 255])
    assert result["bwrMid"] == pytest.approx([247 / 255, 247 / 255, 247 / 255])                      # v0.14: a true white centre (RdBu-7)
    assert result["nan"] == pytest.approx([0.63, 0.68, 0.72])
    assert result["normal"] == [1.0, -2.0, None] and result["inplane"] == [1, 2]              # v · n̂ with the normal made unit
    a = result["arrows"]
    assert a["n120"] == 100 and a["n300"] == 289 and a["consistent"] and a["cells"] == 100    # ⌊√max⌋² cells, one arrow each
    assert a["ref"] == pytest.approx(np.percentile([np.hypot(x, y) for x in (-1 + 2 * (np.arange(40) + .5) / 40) for y in (-1 + 2 * (np.arange(40) + .5) / 40)], 95), rel=1e-6)
    assert result["none"] == {"arrows": [], "ref": 0}


def _swirl_tube(tmp_path, name, meta, lines=None):
    """The straight tube with a parabolic axial profile (+z, downstream) plus an in-plane swirl (−y, x) · 0.1."""
    mesh, cloud, center = tube_case()
    vel = cloud["velocity_m_s"].copy()
    vel[:, 0] = -0.1 * cloud["pts"][:, 1]
    vel[:, 1] = 0.1 * cloud["pts"][:, 0]
    cloud["velocity_m_s"] = vel
    out = tmp_path / name
    build_html(out, meta, mesh, cloud, center, lines)
    return arrays_from_html(out.read_text()), cloud


def test_f7_slice_range_modes_quantity_arrows_log_and_legend(tmp_path):
    meta = _tube_meta()
    data, cloud = _swirl_tube(tmp_path, "f7.html", meta)
    body = r"""
      const V=require(VIEWER),T=V.__test,r={};
      const legend=()=>({title:els['legend-title'].textContent,min:els['legend-min'].textContent,max:els['legend-max'].textContent,note:els['legend-note'].textContent});
      r.cloudLegend=legend();
      T.setMode('slice');
      r.section={scale:T.sliceScale(),legend:legend(),info:T.sliceInfo(),fill:els['slice-fill-note'].textContent};
      r.global=T.setSliceDisplay({mode:'global'}).scale;r.globalLegend=legend();
      r.manual=T.setSliceDisplay({mode:'manual',min:0.2,max:0.8}).scale;
      els['velocity-unit'].value='cm/s';els['velocity-unit'].events.change();
      r.manualCm=T.setSliceDisplay({min:30,max:70}).scale;r.manualBoxes=[els['slice-min'].value,els['slice-max'].value];
      els['velocity-unit'].value='m/s';els['velocity-unit'].events.change();
      T.setSliceDisplay({mode:'section'});
      r.normal={state:T.setSliceDisplay({quantity:'normal'}),info:T.sliceInfo(),legend:legend(),samples:T.sliceInfo().samples.map(s=>s.v)};
      T.setSliceDisplay({quantity:'speed'});
      r.noArrows={arrows:T.setSliceDisplay({arrows:false}).arrows,count:T.sliceInfo().arrows};T.setSliceDisplay({arrows:true});
      r.zoomArrows=(T.sliceZoom(true),T.zoomArrows());T.sliceZoom(false);
      r.log=T.setSliceDisplay({log:true});r.sliceLogLegend=legend();
      T.setMode('cloud');r.cloudLog={legend:legend(),scale:T.legendScale()};
      T.setSliceDisplay({log:false});
      // pressure: no log, no v·n, no arrows; the slice still adapts
      els['volume-field'].value='pressure';els['volume-field'].events.change();T.setMode('slice');
      r.pressure={scale:T.sliceScale(),info:T.sliceInfo(),quantityHidden:els['slice-quantity-row'].hidden,arrowsHidden:els['slice-arrows-row'].hidden,logDisabled:els['velocity-log'].disabled};
      console.log(JSON.stringify(r));
    """
    script = (_stub_prelude(_figure_extra(image=False), common=True) + body).replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    r = node_json(script)
    speed = np.hypot(np.hypot(cloud["velocity_m_s"][:, 0], cloud["velocity_m_s"][:, 1]), cloud["velocity_m_s"][:, 2])
    sec = r["section"]
    samples = np.array([s["v"] for s in sec["info"]["samples"]])
    assert sec["scale"]["source"] == "section" and len(samples) >= 8
    assert sec["scale"]["min"] == pytest.approx(np.percentile(samples, 2), rel=1e-5) and sec["scale"]["max"] == pytest.approx(np.percentile(samples, 98), rel=1e-5)
    assert "本截面自适应 · 全局" in sec["legend"]["note"] and sec["legend"]["title"] == "速度 · m/s"
    assert "箭头 = 面内速度方向（长度按本截面归一化）" in sec["fill"] and 0 < sec["info"]["arrows"] <= 120
    assert r["global"]["source"] == "global" and r["global"]["min"] == pytest.approx(speed.min(), rel=1e-5) and r["global"]["max"] == pytest.approx(speed.max(), rel=1e-5)
    assert "全局色标" in r["globalLegend"]["note"]
    assert r["manual"]["source"] == "manual" and r["manual"]["min"] == pytest.approx(0.2) and r["manual"]["max"] == pytest.approx(0.8)
    assert r["manualCm"]["min"] == pytest.approx(0.3) and r["manualCm"]["max"] == pytest.approx(0.7) and r["manualBoxes"] == ["30", "70"]   # boxes in display units
    # through-plane velocity: the centreline tangent is +z (downstream), the axial profile is +z → every sample ≥ 0
    n = r["normal"]
    assert n["info"]["quantity"] == "normal" and min(n["samples"]) >= 0 and max(n["samples"]) > 0.5
    assert n["state"]["scale"]["diverging"] and n["state"]["scale"]["min"] == pytest.approx(-n["state"]["scale"]["max"])
    assert n["legend"]["title"] == "穿面速度 · m/s" and "顺流为正、负值=回流" in n["legend"]["note"]
    assert r["noArrows"] == {"arrows": False, "count": 0} and 0 < r["zoomArrows"] <= 300
    # log: the slice keeps its own robust range; the whole-field legend starts at max(min, max / 200, 1e-3)
    assert r["log"]["log"] is True and "对数色标" in r["sliceLogLegend"]["note"]
    lo = max(speed.min(), speed.max() / 200, 1e-3)
    assert r["cloudLog"]["scale"]["log"] is True and float(r["cloudLog"]["legend"]["min"]) == pytest.approx(lo, rel=5e-3)
    p = r["pressure"]
    assert p["scale"]["source"] == "section" and not p["scale"].get("log") and not p["scale"].get("diverging")
    assert p["info"]["quantity"] == "pressure" and p["info"]["arrows"] == 0 and p["quantityHidden"] and p["arrowsHidden"] and p["logDisabled"] is False


def test_f7_view_state_round_trip_legacy_replay_csv_header_and_shared_series_range(tmp_path):
    meta = _tube_meta()
    data, _ = _swirl_tube(tmp_path, "f7state.html", meta)
    body = r"""
      const V=require(VIEWER),T=V.__test,r={};
      T.setMode('slice');T.setSliceDisplay({mode:'manual',min:0.1,max:0.9,quantity:'normal',arrows:false,log:true});
      const s=T.capture();r.captured={log:s.log,color_range:s.slice.color_range,quantity:s.slice.quantity,arrows:s.slice.arrows};
      T.setSliceDisplay({mode:'section',quantity:'speed',arrows:true,log:false});
      T.apply(s);r.reapplied=T.setSliceDisplay({});
      // a complete older state (no color_range / quantity / arrows) replays on the whole-field range, no arrows
      const old=JSON.parse(JSON.stringify(s));delete old.slice.color_range;delete old.slice.quantity;delete old.slice.arrows;old.log=false;
      T.apply(old);r.legacy=T.setSliceDisplay({});
      // a partial subset (compare page) without the new keys changes nothing
      T.setSliceDisplay({mode:'manual',quantity:'normal',arrows:true});T.apply({slice:{thickness:3}});r.partial=T.setSliceDisplay({});
      T.setSliceDisplay({mode:'section',quantity:'normal'});T.sliceZoom(true);
      r.csv=T.sliceCSV();r.csv={name:r.csv.name,head:r.csv.csv.split('\n').filter(l=>l.startsWith('#')).join('\n')};T.sliceZoom(false);
      T.setSliceDisplay({quantity:'speed'});
      document.getElementById('slice-series-count').value='3';
      T.exportSliceSeries().then(out=>{r.series={range:out&&out.range,note:els['slice-series-note'].textContent,downloads:downloads.map(d=>d.name)};console.log(JSON.stringify(r));});
    """
    script = (_stub_prelude(_figure_extra(), common=True) + body).replace("META", json.dumps(meta, ensure_ascii=False)).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
    r = node_json(script)
    assert r["captured"]["log"] is True and r["captured"]["quantity"] == "normal" and r["captured"]["arrows"] is False
    assert r["captured"]["color_range"]["mode"] == "manual" and r["captured"]["color_range"]["min"] == pytest.approx(0.1) and r["captured"]["color_range"]["max"] == pytest.approx(0.9)
    re = r["reapplied"]
    assert re["mode"] == "manual" and re["quantity"] == "normal" and re["arrows"] is False and re["log"] is True
    assert re["scale"]["min"] == pytest.approx(0.1) and re["scale"]["max"] == pytest.approx(0.9) and re["scale"]["source"] == "manual"
    assert r["legacy"]["mode"] == "global" and r["legacy"]["quantity"] == "speed" and r["legacy"]["arrows"] is False and r["legacy"]["log"] is False
    assert r["partial"]["mode"] == "manual" and r["partial"]["quantity"] == "normal" and r["partial"]["arrows"] is True
    assert r["csv"]["name"] == "TUBE_slice_through_plane.csv"
    assert "穿面速度" in r["csv"]["head"] and "顺流为正、负值=回流" in r["csv"]["head"] and "显示色标" in r["csv"]["head"] and "本截面自适应" in r["csv"]["head"]
    s = r["series"]
    assert s["range"]["shared"] is True and s["range"]["source"] == "section" and s["downloads"] == ["TUBE_slices_主动脉_speed.png"]
    assert "系列共用" in s["note"]


def test_f7_template_carries_the_slice_colour_controls():
    from wss_deploy.volume_report import TEMPLATE
    for marker in ('id="slice-range-mode"', 'id="slice-zoom-range"', 'id="slice-min"', 'id="slice-max"', 'id="slice-quantity"', 'id="slice-zoom-quantity"',
                   'id="slice-arrows"', 'id="slice-zoom-arrows"', 'id="velocity-log"', '<option value="section">本截面</option>', '穿面速度（沿法向为正）', '面内流向箭头'):
        assert marker in TEMPLATE, marker
    body = TEMPLATE[TEMPLATE.index('id="slice-panel-body"'):]
    assert body.index('id="slice-controls"') < body.index('id="slice-canvas"')            # controls sit on top of the map


# ---------------------------------------------------------------- §21.4 (v0.12.2): automatic labels never overlap
def _boxes_overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > 0.5 and min(a[3], b[3]) - max(a[1], b[1]) > 0.5


def test_label_plan_orders_by_priority_keeps_user_labels_and_ends_leaders_on_the_box():
    result = node_json("require(" + json.dumps(str(COMMON)) + ");\n" + r"""
      const C=globalThis.WssReportCommon;
      const kinds=[['blabel'],['flabel','info'],['dlabel'],['flabel','attention'],['annot'],['flabel','note']];
      const items=kinds.map(([kind,severity],i)=>({kind,severity,text:'L'+i,xyz:[0,0,0],anchor:[0,0,0]}));
      const project=()=>({x:200,y:150});
      const size=()=>({w:60,h:20});
      const plan=core.planLabels(items,{project,size,declutter:C.declutterLabels,bounds:{width:400,height:300}});
      const tight=core.planLabels(items,{project,size,declutter:C.declutterLabels,bounds:{width:70,height:30},hideOverflow:true});
      const none=core.planLabels(items,{project,size});
      console.log(JSON.stringify({prio:items.map(core.labelPriority),plan,tight:tight.map(p=>({kind:p.item.kind,hidden:p.hidden})),none:none.map(p=>p.moved),
        end:core.leaderEnd({x:100,y:100,w:40,h:20,anchor:{x:0,y:100}}),endIn:core.leaderEnd({x:100,y:100,w:40,h:20,anchor:{x:150,y:40}})}));
    """)
    assert result["prio"] == [30, 70, 50, 90, 100, 80]              # branch < info < max diameter < note < attention < user
    plan = result["plan"]
    boxes = [(p["x"] - p["w"] / 2, p["y"] - p["h"] / 2, p["x"] + p["w"] / 2, p["y"] + p["h"] / 2) for p in plan]
    assert not any(_boxes_overlap(boxes[i], boxes[j]) for i in range(6) for j in range(i + 1, 6))
    assert plan[4]["moved"] is False and all(p["moved"] for i, p in enumerate(plan) if i != 4)   # the annotation keeps its spot
    assert plan[3]["y"] > plan[0]["y"] - 1e9 and plan[0]["anchor"] == {"x": 200, "y": 150}
    assert all(p["y"] != 150 or p["moved"] for p in plan)                   # the label sits above its point (y − h/2 − gap)
    hidden = {p["kind"] + str(i): p["hidden"] for i, p in enumerate(result["tight"])}
    assert hidden["annot4"] is False and hidden["blabel0"] is True         # user labels are never hidden, branch names first
    assert result["none"] == [False] * 6                                    # without the shared library nothing moves
    assert result["end"] == {"x": 80, "y": 100} and result["endIn"] == {"x": 120, "y": 90}


def _label_meta(**extra):
    near = [[0.5 * k, 0.0, 10.0 + 0.3 * k] for k in range(6)]
    sev = ["info", "attention", "note", "info", "attention", "note"]
    meta = {"case_id": "TUBE", "run_identity": "tube-run", "branch_names": {"1": "主动脉"},
            "findings": {"items": [{"id": f"F{k + 1}", "kind": "low_speed_region", "label": f"低速区 {k + 1}", "branch": "主动脉", "segment_id": 1,
                                    "value": 0.01 * (k + 1), "units": "m/s", "xyz_mm": near[k], "severity": sev[k], "rank": k + 1} for k in range(6)]}}
    meta.update(extra)
    return meta


_ORTHO = r"""
      V3.prototype.project=function(){const x=this.x,z=this.z;this.x=x/30;this.y=z/30;this.z=0;return this;};
      global.__declutterCalls=0;
"""


def test_page_labels_are_decluttered_with_leaders_and_hide_overflow_when_narrow(tmp_path):
    meta = _label_meta()
    body = r"""
      const V=require(VIEWER),T=V.__test,C=globalThis.WssReportCommon,orig=C.declutterLabels;
      C.declutterLabels=(items,opts)=>{__declutterCalls++;global.__lastOpts=opts;return orig(items,opts);};
      T.setLabels({findings:5,branches:true});
      const wide=T.layoutLabels(),children=document.getElementById('labels').children;
      const chips=children.filter(c=>c.className!=='leader'),leaders=children.filter(c=>c.className==='leader');
      const r={calls:__declutterCalls,opts:{hide:__lastOpts.hideOverflow,bounds:__lastOpts.bounds},wide,chips:chips.map(c=>({cls:c.className,left:c.style.left,top:c.style.top})),
        leaders:leaders.map(l=>({w:parseFloat(l.style.width),t:l.style.transform}))};
      r.keySame=T.cameraKey();
      __controls.target.set(1,2,3);r.keyMoved=T.cameraKey()!==r.keySame;
      document.getElementById('volume-view').clientWidth=800;
      r.narrow=T.layoutLabels();r.narrowOpts=__lastOpts.hideOverflow;
      r.narrowChips=document.getElementById('labels').children.filter(c=>c.className!=='leader').length;
      console.log(JSON.stringify(r));
    """
    r = _run_tube(tmp_path, "labels.html", meta, body, _webgl_extra(_ORTHO), common=True)
    assert r["calls"] == 1 and r["opts"]["hide"] is False and r["opts"]["bounds"] == {"width": 900, "height": 600}
    wide = r["wide"]
    assert [x["kind"] for x in wide].count("flabel") == 5 and [x["kind"] for x in wide].count("blabel") == 1
    boxes = [(p["x"] - p["w"] / 2, p["y"] - p["h"] / 2, p["x"] + p["w"] / 2, p["y"] + p["h"] / 2) for p in wide if not p["hidden"]]
    assert not any(_boxes_overlap(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes)))
    first_attention = next(p for p in wide if p["severity"] == "attention")
    assert first_attention["moved"] is False                                 # the highest priority chip keeps its spot
    moved = [p for p in wide if p["moved"] and not p["hidden"]]
    # one leader per moved chip, except a chip whose box still covers its anchor (nothing to point at)
    assert moved and 1 <= len(r["leaders"]) <= len(moved) and all(l["w"] >= 2 and l["t"].startswith("rotate(") for l in r["leaders"])
    assert len(r["chips"]) == len([p for p in wide if not p["hidden"]])
    assert all(c["cls"].endswith(" moved") == p["moved"] for c, p in zip(r["chips"], [p for p in wide if not p["hidden"]]))
    assert r["keyMoved"] is True                                             # a camera change invalidates the layout
    assert r["narrowOpts"] is True                                           # < 900 px → hideOverflow
    assert all(not p["hidden"] for p in r["narrow"] if p["severity"] == "attention")
    assert r["narrowChips"] == len([p for p in r["narrow"] if not p["hidden"]])


def test_export_composites_the_same_decluttered_layout_with_leaders(tmp_path):
    meta = _label_meta()
    body = r"""
      const V=require(VIEWER),T=V.__test;
      T.setLabels({findings:5,branches:true});
      const rects=[],lines=[];let alpha=[];
      const ctx={save(){},restore(){},fillText(){},beginPath(){},stroke(){},strokeRect(){},measureText:t=>({width:String(t).length*7*2}),
        fillRect:(x,y,w,h)=>rects.push([x,y,x+w,y+h]),moveTo:(x,y)=>lines.push([x,y]),lineTo(){},set globalAlpha(v){alpha.push(v);},get globalAlpha(){return 1;}};
      const plan=T.drawOverlayLabels(ctx,1800,1200,2,'zh');
      console.log(JSON.stringify({n:plan.length,moved:plan.filter(p=>p.moved&&!p.hidden).length,rects,leaders:lines.length,alpha}));
    """
    r = _run_tube(tmp_path, "labels_export.html", meta, body, _webgl_extra(_ORTHO), common=True)
    assert r["n"] == 6 and len(r["rects"]) == 6
    assert not any(_boxes_overlap(r["rects"][i], r["rects"][j]) for i in range(6) for j in range(i + 1, 6))
    assert r["moved"] >= 1 and r["leaders"] >= r["moved"] and 0.6 in r["alpha"]   # moved chips get a translucent leader


def test_v015_served_volume_report_links_back_to_its_case_and_labels_the_colour_bar(tmp_path):
    """Round 15: 「← 工作台」 appears on a page the service serves (like the wall report) and points at #job=<id>;
    a file:// copy keeps it hidden.  The colour bar gets a text alternative."""
    mesh, cloud, center = sample()
    out = tmp_path / "served.html"
    build_html(out, {"case_id": "served"}, mesh, cloud, center)
    data = arrays_from_html(out.read_text())
    probe = r"""
      console.log(JSON.stringify({hidden:els['back-to-workbench'].hidden,href:els['back-to-workbench'].href||null,
        legend:(els['legend-bar'].attr||{})['aria-label']||null,role:(els['legend-bar'].attr||{}).role||null}));
    """
    attrs = "Element.prototype.setAttribute=function(k,v){(this.attr=this.attr||{})[k]=String(v);};"
    served = _stub_prelude(attrs + "global.location={origin:'http://h',protocol:'http:',pathname:'/api/jobs/j1/report',hash:''};global.fetch=async()=>({ok:false,status:404,json:async()=>({})});") + probe
    local = _stub_prelude(attrs + "global.location={protocol:'file:',pathname:'/x/report.html',hash:''};global.fetch=undefined;") + probe
    results = []
    for script in (served, local):
        script = script.replace("META", json.dumps({"case_id": "served"})).replace("DATA", json.dumps(data)).replace("VIEWER", json.dumps(str(VIEWER)))
        results.append(node_json(script))
    assert results[0]["hidden"] is False and results[0]["href"] == "/#job=j1"
    assert results[1]["href"] is None             # file:// copy: untouched (the markup keeps it hidden)
    assert '<a id="back-to-workbench" class="back-link" href="/" title="回到工作台（本病例）" hidden>' in (tmp_path / "served.html").read_text()
    assert results[0]["role"] == "img" and results[0]["legend"].startswith("色标 ") and " – " in results[0]["legend"]
    assert "back-to-workbench" in (tmp_path / "served.html").read_text()
