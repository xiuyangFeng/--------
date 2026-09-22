"""Three-head (peak WSS + TAWSS + OSI) release family: contract, package verification, inference, extra-field exports."""
import copy
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wss_deploy import cycle_fields as CF
from wss_deploy.families import CYCLE_FIELDS, WALL_CYCLE_MULTI, family_for_info
from wss_deploy.infer import Release
from wss_deploy.io_utils import file_sha256
from wss_deploy.registry import ReleaseError, ReleaseRegistry, _contract
from wss_deploy.report import build_html

SEEDS = (1234, 7, 2025)
FEATURES = ["x", "y", "z"]


def contract():
    return {"release": "cycle-demo", "target": "wss_cycle_multi", "model_family": "M1_3head",
            "time_axis": [{"index": 0, "step": 1162, "time_s": .21}],
            "fields": copy.deepcopy(CYCLE_FIELDS), "cycle": {"frames": "0-79", "period_s": 0.8},
            "models": [{"seed": s, "path": f"models/M1_s{s}"} for s in SEEDS]}


def stats():
    return {"method": "multi", "channels": ["wss", "tawss", "osi"], "eps": 1e-6, "floor": 0.05,
            "wss": {"method": "log_z", "eps": 1e-6, "floor": 0.0, "log": {"mean": 0.0, "std": 1.0}, "linear": {"mean": 1., "std": 1.}},
            "tawss": {"method": "log_z", "eps": 1e-6, "floor": 0.05, "log": {"mean": 0.0, "std": 1.0}, "linear": {"mean": 1., "std": 1.}},
            "osi": {"method": "logit_z", "eps": 0.0, "logit": {"mean": 0.0, "std": 1.0, "scale": 0.5, "clip": 1e-3}}}


def manifest(root):
    paths = [p for sub in ("models", "rules") for p in sorted((root / sub).rglob("*")) if p.is_file()]
    (root / "MANIFEST.sha256").write_text("".join(f"{file_sha256(p)}  {p.relative_to(root)}\n" for p in paths))


def package(root):
    info = contract()
    root.mkdir()
    (root / "release.json").write_text(json.dumps(info))
    for spec in info["models"]:
        run = root / spec["path"]
        run.mkdir(parents=True)
        cfg = {"data": {"target": "wss_cycle_multi", "timesteps": "peak", "target_normalization": "global_stats",
                        "required_frame_version": "v5_atlas_frame_v1", "input_features": FEATURES},
               "model": {"out_dim": 3}, "train": {"seed": spec["seed"]}, "eval": {}}
        for name, data in (("config", cfg), ("feature_stats", {}), ("wss_global_stats", stats()),
                           ("target_normalization", {"mode": "global_stats", "physical_recovery_enabled": True})):
            (run / f"{name}.json").write_text(json.dumps(data))
        (run / "ckpt_best.pt").write_bytes(b"mock checkpoint")
    (root / "rules").mkdir()
    (root / "rules/flow_split_rule_train136.json").write_text("{}")
    manifest(root)
    return root


def test_cycle_contract_is_explicit_and_fails_closed():
    good = _contract(contract())
    assert good["protocol"] == "single_frame_wss_cycle_multi" and good["family"] == "wall_cycle_multi_v1"
    assert good["seeds"] == list(SEEDS) and good["channels"] == ["wss", "tawss", "osi"]
    assert family_for_info(contract()) is WALL_CYCLE_MULTI
    assert family_for_info({"target": "wss"}).id == "wall_wss_v1"      # legacy WSS packages are untouched
    for mutate in (lambda c: c.pop("model_family"),
                   lambda c: c["fields"].pop("osi"),
                   lambda c: c["fields"]["osi"].__setitem__("units", "Pa"),
                   lambda c: c.pop("cycle"),
                   lambda c: c["cycle"].__setitem__("period_s", 1.0),
                   lambda c: c["time_axis"].append(c["time_axis"][0]),
                   lambda c: c["models"].append(dict(c["models"][0])),
                   lambda c: c["models"].__setitem__(0, {"seed": "1234", "path": "models/M1_s1234"}),
                   lambda c: c.__setitem__("models", [])):
        bad = contract(); mutate(bad)
        with pytest.raises(ReleaseError):
            _contract(bad)


def test_cycle_package_verification_and_seed_subsets(tmp_path):
    root = package(tmp_path / "cycle")
    registry = ReleaseRegistry(root, loader=lambda *a, **kw: SimpleNamespace())
    listing = registry.list()[0]
    assert listing["models_count"] == 3 and listing["weights_count"] == 3
    registry.load(device="cpu")
    cfg_path = root / "models/M1_s7/config.json"
    cfg = json.loads(cfg_path.read_text()); cfg["model"]["out_dim"] = 1
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ReleaseError, match="校验失败"):
        registry.load(device="cpu")
    manifest(root)
    with pytest.raises(ReleaseError, match="三头"):
        ReleaseRegistry(root, loader=lambda *a, **kw: SimpleNamespace()).load(device="cpu")
    cfg["model"]["out_dim"] = 3; cfg_path.write_text(json.dumps(cfg))
    stats_path = root / "models/M1_s7/wss_global_stats.json"
    broken = stats(); broken["osi"]["logit"]["scale"] = 1.0
    stats_path.write_text(json.dumps(broken)); manifest(root)
    with pytest.raises(ReleaseError, match="OSI"):
        ReleaseRegistry(root, loader=lambda *a, **kw: SimpleNamespace()).load(device="cpu")


def test_cycle_inference_restores_three_channels_and_clips_osi(tmp_path, monkeypatch):
    root = package(tmp_path / "cycle")
    calls = []

    def load(run, device, checkpoint):
        raw = json.loads((run / "config.json").read_text())
        cfg = SimpleNamespace(data=SimpleNamespace(**raw["data"]))
        return cfg, {}, SimpleNamespace(eval=lambda: None, value=raw["train"]["seed"]), {}

    def predict(model, case, features, stats, device, *, cfg, return_all_channels):
        assert return_all_channels is True and cfg.data.target == "wss_cycle_multi"
        calls.append(model.value)
        n = len(case["pos"])
        if model.value == 1234:
            return np.zeros((n, 3))                          # exp(0) = 1 Pa, OSI = 0.5 * sigmoid(0) = 0.25
        return np.tile([np.log(2.0), np.log(2.0), 10.0], (n, 1))  # 2 Pa, 2 Pa, OSI -> 0.5

    monkeypatch.setattr("wss_deploy.infer.E.load_model_from_run", load)
    monkeypatch.setattr("wss_deploy.infer.E.predict_case_norm", predict)
    monkeypatch.setattr("wss_deploy.infer.D.load_case", lambda *a, **kw: pytest.fail("CFD data read"))
    release = ReleaseRegistry(root).load(device="cpu", seed_count=2)
    result = release.predict({"pos": np.zeros((4, 3))})
    assert calls == [1234, 7]
    np.testing.assert_allclose(result["wss_pa"], 1.5, atol=1e-5)
    np.testing.assert_allclose(result["tawss_pa"], 1.5, atol=1e-5)
    np.testing.assert_allclose(result["osi"], (0.25 + 0.5 / (1 + np.exp(-10.0))) / 2, atol=1e-5)
    assert result["seed_pred_pa"].shape == (2, 4) and result["seed_tawss_pa"].shape == (2, 4) and result["seed_osi"].shape == (2, 4)
    assert (result["osi"] <= 0.5).all() and (result["seed_osi"] <= 0.5).all()
    assert set(result["extra_fields"]) == {"tawss", "osi"}
    np.testing.assert_allclose(result["extra_fields"]["tawss"]["values"], result["tawss_pa"])
    assert result["extra_fields"]["osi"]["seed_pred"].shape == (2, 4)
    assert result["cycle"]["period_s"] == 0.8
    with pytest.raises(ValueError, match="模型数量"):
        Release(root, device="cpu", seed_count=4)


def test_cycle_field_summaries_and_descriptor():
    rng = np.random.default_rng(0)
    seg = np.repeat([1, 2], 50)
    tawss = np.where(seg == 1, 0.2, 2.0) + rng.random(100) * 0.05
    osi = np.where(seg == 1, 0.3, 0.05)
    s = CF.scalar_field_summary("tawss", tawss, seg, {1: "root", 2: "left_cia"}, total_area_mm2=200.0)
    assert s["units"] == "Pa" and s["area_frac"]["low"] == pytest.approx(0.5) and s["area_mm2"]["low"] == pytest.approx(100.0)
    assert set(s["per_branch"]) == {"root", "left_cia"} and s["per_branch"]["root"]["frac_low"] == pytest.approx(1.0)
    o = CF.scalar_field_summary("osi", osi, seg, {"1": "root", "2": "left_cia"}, total_area_mm2=200.0)
    assert o["units"] == "1" and o["area_frac"]["above_t0"] == pytest.approx(0.5) and o["area_frac"]["above_t2"] == pytest.approx(0.0)
    st = CF.stagnation_summary(tawss, osi, seg, {1: "root", 2: "left_cia"}, 200.0)
    assert st["area_frac"] == pytest.approx(0.5) and st["per_branch"]["root"]["frac"] == pytest.approx(1.0) and st["per_branch"]["left_cia"]["frac"] == 0.0
    block = CF.cycle_block({"tawss": {"values": tawss}, "osi": {"values": osi}}, seg, {1: "root", 2: "left_cia"}, 200.0)
    assert set(block["fields"]) == {"tawss", "osi"} and "stagnation" in block and block["definition"]["period_s"] == 0.8
    d = CF.descriptor("osi", osi)
    assert d["array_key"] == "osi" and d["units"] == "1" and d["display"]["log_scale"] is False and d["display"]["range"] == [0.0, 0.5]
    assert CF.descriptor("tawss", tawss)["display"]["thresholds"] == [0.4, 4.0, 7.0]
    with pytest.raises(ValueError):
        CF.scalar_field_summary("tawss", tawss[:10], seg, {}, 200.0)


def _report_fixture(tmp_path):
    pts = np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)
    meta = {"case_id": "c", "release": "test", "endpoints": [], "branch_names": {}, "per_branch": {}, "geometry": {}, "input_check": {},
            "cloud": {}, "timing_s": {},
            "wss_field_pa": {"p99": 2., "p99_pa": 2., "mean": 1., "median": 1., "thresholds_pa": [.4, 4., 7.], "area_frac_low": 0., "area_low_mm2": 0.,
                             "area_frac_high": 0., "area_high_mm2": 0., "area_frac_very_high": 0.},
            "peak": {"p99_pa": 2., "max_pa": 3., "xyz_mm": [0, 0, 0], "branch": "x", "s_from_inlet_mm": 0., "dist_to_junction_mm": 0.,
                     "local_radius_mm": 1., "top5_clusters_ge20": 0, "top5_cluster_sizes": []},
            "fields": {"wss": {"id": "wss", "label": "壁面切应力", "units": "Pa", "location": "wall", "kind": "scalar", "components": 1,
                               "array_key": "wss_pa", "axis_order": ["point"], "time_indices": [0]},
                       "tawss": CF.descriptor("tawss", np.array([0.5, 1.5])), "osi": CF.descriptor("osi", np.array([0.1, 0.3]))},
            "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21, "label": "peak_systole"}]}
    mesh = {"vertices": pts, "faces": np.array([[0, 1, 1]], dtype=np.uint32), "wss": np.array([1., 2.], dtype=np.float32), "segment": np.array([1, 1], dtype=np.int32),
            "extra": {"tawss_pa": np.array([0.5, 1.5], dtype=np.float32), "osi": np.array([0.1, 0.3], dtype=np.float32)}}
    cloud = {"pts": pts, "wss": np.array([1., 2.], dtype=np.float32), "segment": np.array([1, 1], dtype=np.int32),
             "s_from_root_mm": np.array([0., 1.], dtype=np.float32), "theta_rad": np.array([0., 1.], dtype=np.float32),
             "radius_mm": np.array([1., 1.], dtype=np.float32), "dist_to_junction_mm": np.array([0., 1.], dtype=np.float32),
             "extra": {"tawss_pa": np.array([0.5, 1.5], dtype=np.float32), "osi": np.array([0.1, 0.3], dtype=np.float32)}}
    centerline = {"xyz": pts, "radius_mm": np.array([1., 1.], dtype=np.float32), "edges": np.array([[0, 1]], dtype=np.uint32), "segment": np.array([1, 1], dtype=np.int32)}
    return meta, mesh, cloud, centerline


def test_report_embeds_extra_fields_and_page_script_parses(tmp_path):
    out = tmp_path / "report.html"
    meta, mesh, cloud, centerline = _report_fixture(tmp_path)
    build_html(out, meta, mesh, cloud, centerline)
    html = out.read_text(encoding="utf-8")
    arrays = json.loads(html.split('id="wss-report-arrays"')[1].split(">", 1)[1].split("</script>")[0])
    assert set(arrays["xf"]) == {"tawss_pa", "osi"} and set(arrays["xf"]["osi"]) == {"m", "p"}
    assert "fieldArrays(activeField)" in html and "ARR.xf" in html
    bad_cloud = dict(cloud, extra={"tawss_pa": cloud["extra"]["tawss_pa"]})
    with pytest.raises(ValueError, match="same fields"):
        build_html(tmp_path / "bad.html", meta, mesh, bad_cloud, centerline)
    bad_mesh = dict(mesh, extra={**mesh["extra"], "osi": np.array([0.1], dtype=np.float32)})
    with pytest.raises(ValueError, match="one value per"):
        build_html(tmp_path / "bad2.html", meta, bad_mesh, cloud, centerline)
    if not shutil.which("node"):
        pytest.skip("node not available for the page-script syntax check")
    # Every inline script of the page must still parse after the field-switch change.
    scripts = [chunk.split("</script>")[0] for chunk in html.split("<script>")[1:]]
    assert scripts
    for index, script in enumerate(scripts):
        path = tmp_path / f"page_{index}.js"
        path.write_text(script, encoding="utf-8")
        subprocess.run(["node", "--check", str(path)], check=True, capture_output=True, text=True)


def test_report_field_tab_click_repaints_wall(tmp_path):
    """Clicking the TAWSS tab must switch the active field AND call the viewer's recolor hook.

    Regression: recolor() is declared inside initViewer(), so a top-level ``typeof recolor`` guard never
    fired and the tab only changed its highlight (seen by the user on 2026-09-22).  The page-script
    segment holding the schema helpers runs in node against a stub DOM.
    """
    if not shutil.which("node"):
        pytest.skip("node not available")
    out = tmp_path / "r.html"
    meta, mesh, cloud, centerline = _report_fixture(tmp_path)
    build_html(out, meta, mesh, cloud, centerline)
    html = out.read_text(encoding="utf-8")
    assert "recolorActive=recolor;" in html and "if(recolorActive)recolorActive();" in html
    meta_json = html.split('id="wss-report-meta"')[1].split(">", 1)[1].split("</script>")[0]
    arrays_json = html.split('id="wss-report-arrays"')[1].split(">", 1)[1].split("</script>")[0]
    page = [chunk.split("</script>")[0] for chunk in html.split("<script>")[1:] if "function renderSchemaControls" in chunk][0]
    start, end = page.index("function dec(b64,T)"), page.index("// ---- arrays (decoded once")
    segment = page[start:end]
    (tmp_path / "meta.json").write_text(meta_json, encoding="utf-8")
    (tmp_path / "arrays.json").write_text(arrays_json, encoding="utf-8")
    prelude = """
const fs=require('fs');
const META=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),ARR=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
function mk(tag){return {tagName:tag,className:'',dataset:{},textContent:'',title:'',type:'',disabled:false,onclick:null,value:'',children:[],
  classes:[],classList:{add(c){}},setAttribute(){},appendChild(c){this.children.push(c);},replaceChildren(){this.children=[];}};}
const ELS={};function el(id){return ELS[id]||(ELS[id]=mk('div'));}
const document={createElement:mk};
var MW=new Float32Array(2),PW=new Float32Array(2),PEAK={p99_pa:1,max_pa:2};
"""
    epilogue = """
let calls=0;recolorActive=()=>{calls++;};
renderSchemaControls();
const before=el('field-tabs').children.map(t=>[t.dataset.field,t.tagName,!!t.onclick]);
const tab=el('field-tabs').children.find(t=>t.dataset.field==='tawss');
if(!tab||tab.tagName!=='button'||!tab.onclick)throw new Error('tawss tab not switchable: '+JSON.stringify(before));
tab.onclick();
const F=fieldArrays(activeField);
tab.onclick();  // clicking the active tab again is a no-op
console.log(JSON.stringify({calls,activeField,before,m_len:F.m.length,p_len:F.p.length,units:F.units,isPa:F.isPa,log:F.log,p99:F.p99,first_m:F.m[0],first_p:F.p[0]}));
"""
    script = tmp_path / "tabs.js"
    script.write_text(prelude + segment + epilogue, encoding="utf-8")
    run = subprocess.run(["node", str(script), str(tmp_path / "meta.json"), str(tmp_path / "arrays.json")], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    got = json.loads(run.stdout.strip().splitlines()[-1])
    assert got["calls"] == 1 and got["activeField"] == "tawss"
    assert sorted(f for f, _, _ in got["before"]) == ["osi", "tawss", "wss"] and all(tag == "button" and has for _, tag, has in got["before"])
    assert got["m_len"] == len(mesh["extra"]["tawss_pa"]) and got["p_len"] == len(cloud["extra"]["tawss_pa"])
    assert got["units"] == "Pa" and got["isPa"] and got["log"] is True and got["p99"] == pytest.approx(float(np.quantile(cloud["extra"]["tawss_pa"], 0.99)))
    assert got["first_m"] == pytest.approx(float(mesh["extra"]["tawss_pa"][0]), rel=1e-6)
    assert got["first_p"] == pytest.approx(float(cloud["extra"]["tawss_pa"][0]), rel=1e-6)
