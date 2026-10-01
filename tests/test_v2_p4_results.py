"""Final migration round, lane B (results): tests.

* W55 ② — ``ws_legacy.classicToV2`` maps the classic iso-lines, highlight of the highest x %, wall cut and velocity log
  scale into the link's ``x.display`` (ws_display's link state) when the page's display module takes it; the classic
  encoders (report.py ``viewHash``, ``VolumeViewerCore.encodeView``) produce the input; opened in the workspace, the
  classic link switches the same options on.
* C+1 — in compare mode the right-hand result's warnings (reference range / population / ensemble quality,
  ``ns.notice.model``) show in the compare tab and as a badge on the right viewport; no dispersion numbers.
* W13 / W27 — the overview's marker locations (branch, distance from inlet, distance to the junction, local radius)
  equal the classic report's ``fieldPeak`` cards; a three-head result shows the whole-field WSS p99 when WSS is shown.
* V37 — the overview's volume statistics table equals the classic volume page's ``setStats`` rows
  (``VolumeViewerCore.statistics`` over the interior points).
* W58 / W59 — Ctrl+P goes to the print-view page; a print started from the browser menu prints a snapshot of the
  3-D view instead of the blank WebGL canvas, removed again after printing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from wss_deploy.paths import STATIC_DIR

V2 = STATIC_DIR / "v2"
LEGACY = V2 / "ws_legacy.js"
OWN = ["ws_legacy.js", "ws_notice.js", "ws_compare.js", "ws_overview.js", "ws_figure.js"]


def _need_node():
    if not shutil.which("node"):
        pytest.skip("Node is required for the lane B tests")


def _node(script: str, **extra) -> dict:
    _need_node()
    pre = "globalThis.atob=s=>Buffer.from(s,'base64').toString('binary');globalThis.btoa=s=>Buffer.from(s,'binary').toString('base64');\n"
    for path in extra.values():
        pre += "require(" + json.dumps(str(path)) + ");\n"
    pre += "const L=require(" + json.dumps(str(LEGACY)) + ");\nconst out=x=>console.log(JSON.stringify(x));\n"
    result = subprocess.run(["node", "-e", pre + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"],
                            text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_own_scripts_parse():
    _need_node()
    for name in OWN:
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)


# ----------------------------------------------------------------------------------------------- W55 ②
def _wall_core(tmp_path: Path) -> Path:
    from wss_deploy.report import REPORT_CORE_JS
    path = tmp_path / "wall_core.js"
    path.write_text(REPORT_CORE_JS, encoding="utf-8")
    return path


def _states():
    from tests.test_v2_p3_s7 import VOLUME_STATE, WALL_STATE
    return WALL_STATE, VOLUME_STATE


def test_classic_iso_lines_highlight_cut_and_log_go_into_the_display_link_state(tmp_path):
    wall_state, volume_state = _states()
    changed = dict(wall_state, overlay={"trust": False, "contours": True, "stagnation": False}, slice={"clip": 0.4},
                   highlight={"branch": -1, "top_pct": 2.5, "top": True, "peak": True, "feature": "wss"})
    out = _node("""
      const C=globalThis.WssReportCore, V=globalThis.VolumeViewerCore, S=@@wall@@, X=@@changed@@, VS=@@vol@@;
      const wctx={family:'wall', fields:['wss','tawss','osi'], logFields:['wss','tawss'], branches:[0,1,2,3], xDisplay:true};
      const vctx={family:'volume', fields:['speed','pressure','wall_pressure'], logFields:[], branches:[0,2,3], xDisplay:true};
      const viaHash=st=>L.decodeClassicView(L.parseAddress('?job=J', C.viewHash(st)).view);
      const on=L.classicToV2(viaHash(X), wctx), off=L.classicToV2(viaHash(X), Object.assign({}, wctx, {xDisplay:false}));
      const def=L.classicToV2(viaHash(S), wctx);
      const odd=[L.convertDisplayExt({overlay:{contours:true}, highlight:{top_pct:40}, slice:{clip:-3}}, 'wall').x,
                 L.convertDisplayExt({highlight:{top:1, top_pct:'x'}, slice:{clip:1.5}}, 'wall').x,
                 L.convertDisplayExt({overlay:{trust:true}}, 'wall').x, L.convertDisplayExt({}, 'volume').x];
      const vol=L.classicToV2(L.decodeClassicView(V.encodeView(Object.assign({}, VS, {log:true, mode:'cloud'}))), vctx);
      const volOff=L.classicToV2(Object.assign({}, VS, {log:true, mode:'cloud'}), Object.assign({}, vctx, {xDisplay:false}));
      const volPressure=L.classicToV2(Object.assign({}, VS, {log:false, field:'pressure', mode:'cloud'}), vctx).state.x;
      out({on, off:off.dropped, def:def.state.x, defDropped:def.dropped, odd, vol:vol.state.x, volDropped:vol.dropped, volOff:volOff.dropped, volPressure});
    """.replace("@@wall@@", json.dumps(wall_state, ensure_ascii=False)).replace("@@changed@@", json.dumps(changed, ensure_ascii=False))
       .replace("@@vol@@", json.dumps(volume_state, ensure_ascii=False)),
       core=_wall_core(tmp_path), common=STATIC_DIR / "report_common.js", vv=STATIC_DIR / "volume_viewer.js")
    assert out["on"]["state"]["x"] == {"display": {"v": 1, "contours": True, "top": True, "topPct": 2.5, "clip": 0.4}}
    assert out["on"]["dropped"] == []                                                     # nothing named as not carried
    assert out["off"] == ["等值线", "剖切", "高亮最高区域"]                                 # older readers: named as before
    # the classic defaults reproduce as they looked: no lines, no highlight (1 %), no cut
    assert out["def"] == {"display": {"v": 1, "contours": False, "top": False, "topPct": 1, "clip": None}} and out["defDropped"] == []
    # read as the classic applyViewState read them: !!top, top_pct clamped to 0.5–10, clip clamped to 0–1 (1 = no cut)
    assert out["odd"] == [{"v": 1, "contours": True, "top": False, "topPct": 10, "clip": 0}, {"v": 1, "top": True, "clip": None},
                          {"v": 1, "contours": False}, None]
    assert out["vol"] == {"display": {"v": 1, "speedLog": True}} and "对数色标" not in out["volDropped"]
    assert out["volOff"][0] == "对数色标"
    assert out["volPressure"] == {"display": {"v": 1, "speedLog": False}}


def test_the_page_context_knows_the_display_link_state():
    out = _node("""
      const ns=globalThis.WSSV2;
      const before=L.pageContext({}, {result:{family:'wall'}, fields:[{id:'wss'}]}).xDisplay;
      ns.ext.push({id:'display', linkState(){ return {v:1}; }, applyLinkState(){ return true; }});
      const after=L.pageContext({}, {result:{family:'wall'}, fields:[{id:'wss'}]}).xDisplay;
      out({before, after, other:L.linkExtension('nope')});
    """)
    assert out == {"before": False, "after": True, "other": False}


def _workspace():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ws_harness_p4b", Path(__file__).with_name("test_v2_workspace_js.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _harness(scenario: str, files: list[str], *, before: str = "", offline: bool = False) -> dict:
    _need_node()
    ws = _workspace()
    program = (ws._HARNESS.replace("__WS__", json.dumps([str(V2 / f) for f in files])).replace("__OFFLINE__", "true" if offline else "false")
               + before + "\n(async () => {\n" + scenario + "\n})().catch(e => { errors.push('scenario: ' + (e && e.stack || e)); done({}); });\n")
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise AssertionError(result.stderr[-4000:])
    lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
    assert lines, result.stdout[-2000:] + result.stderr[-2000:]
    return json.loads(lines[-1])


# The harness's stub viewer has no result(); ws_display keys its per-result session (bands, thresholds, wall cut) on it.
VIEWER_RESULT = ("\nconst _create = ns.viewer.create;\n"
                 "ns.viewer.create = (c, o) => { const v = _create(c, o); let r = null; const sr = v.setResult;\n"
                 "  v.setResult = x => sr(x).then(y => { r = x; return y; }); v.result = () => r; return v; };\n")


def _files(*extra: str, after: str = "ws_lens.js") -> list[str]:
    ws = _workspace()
    files = [f for f in ws.WS_FILES if f not in {"ws_shell.js", "ws_main.js"}]
    at = files.index(after) + 1
    files[at:at] = list(extra)
    return ["../report_common.js", "../volume_viewer.js"] + files + ["ws_figure.js", "ws_shell.js", "ws_legacy.js", "ws_main.js"]


def test_a_classic_link_switches_the_display_options_on_in_the_workspace():
    from wss_deploy.report import REPORT_CORE_JS
    wall_state, _ = _states()
    state = dict(wall_state, overlay={"trust": True, "contours": True, "stagnation": False}, slice={"clip": 0.35},
                 highlight={"branch": -1, "top_pct": 3, "top": True, "peak": True, "feature": "wss"})
    before = (VIEWER_RESULT + "\nglobal.WSSV2_LEGACY_AUTOREWRITE = true;\n" + REPORT_CORE_JS + "\n"
              "global.location.search = '?job=A'; global.location.hash = WssReportCore.viewHash(" + json.dumps(state, ensure_ascii=False) + ");\n")
    out = _harness("""
      await boot();
      await wait(500);
      const D = ns.display, o = D.prefs().opts, v = shellState().viewerA;
      done({opts: {contours: o.contours, top: o.top, topPct: o.topPct}, clip: D.session && D.session(shellState().cur.runIdentity) ? D.session(shellState().cur.runIdentity).clip : null,
            toast: textOf(byId('ws-toast'))});
    """, _files("core_colormap.js", "core_contour.js", "ws_display.js"), before=before)
    assert out["errors"] == [], out["errors"]
    assert out["opts"] == {"contours": True, "top": True, "topPct": 3}
    assert out["clip"] == 0.35
    assert "已按旧版链接打开" in out["toast"]
    for word in ("等值线", "剖切", "高亮最高区域"):
        assert word not in out["toast"]


# ----------------------------------------------------------------------------------------------- C+1
_WARN = r"""
  const B = id => canned['/api/v2/jobs/' + id + '/manifest'].body;
  const warn = id => { const b = B(id); b.analysis.reference_assessment = {status: 'review', note: '仅检查已声明的几何参考范围。',
      checks: [{path: 'geometry.左髂总.length_mm', units: 'mm', value: 177.82, min: 21.116, max: 141.016, status: 'review'}], population: {status: 'review', reasons: ['参照队列不足']}};
    b.analysis.quality = {level: 'review', label: '存在不确定性，建议复核', reasons: ['3 个模型离散度超过常规范围，建议复核热点和分支']};
    b.provenance = Object.assign({}, b.provenance, {n_models: 3}); };
  const btn = (root, text) => walk(root, e => e.tagName === 'BUTTON' && textOf(e).trim() === text)[0];
"""


def test_compare_shows_the_right_hand_warnings_in_the_tab_and_on_its_viewport():
    out = _harness(_WARN + r"""
      warn('B');
      await boot();
      await hashTo('#/job/A', 200);
      const topBefore = textOf(byClass(app(), 'ws-notices')[0]);
      await hashTo('#/job/A?v=compare&cmp=B', 300);
      const S = shellState(), body = byClass(app(), 'insp-body')[0];
      const warns = byClass(body, 'cmp-warn').map(e => ({side: e.dataset.side, target: e.dataset.target, text: textOf(e)}));
      const badgeB = byClass(S.els.vpB.status, 'vp-notice')[0], badgeA = byClass(S.els.vpA.status, 'vp-notice')[0];
      const expected = ns.notice.model(S.cur.compare.manifest).text;
      fire(badgeB, 'click', {stopPropagation() {}}); await wait(20);
      const dlg = textOf(byId('ws-dialog')); if (byId('ws-dialog')) byId('ws-dialog').close();
      fire(btn(byClass(body, 'cmp-warn')[0], '详情'), 'click'); await wait(20);
      const dlg2 = textOf(byId('ws-dialog')); if (byId('ws-dialog')) byId('ws-dialog').close();
      // the right side keeps its badge whatever tab is open; exiting the comparison removes it
      fire(walk(app(), e => e.dataset && e.dataset.tab === 'overview')[0], 'click'); await wait(20);
      const stillThere = byClass(S.els.vpB.status, 'vp-notice').length, tabNow = S.tab, warnsNow = byClass(byClass(app(), 'insp-body')[0], 'cmp-warn').length;
      await hashTo('#/job/A', 200);
      done({topBefore, warns, badge: badgeB ? {text: textOf(badgeB), title: badgeB.title} : null, badgeA: Boolean(badgeA), expected, dlg, dlg2, stillThere, tabNow, warnsNow,
            after: byClass(S.els.vpB.status, 'vp-notice').length, compare: Boolean(S.cur.compare)});
    """, _files("ws_notice.js"))
    assert out["errors"] == [], out["errors"]
    assert out["topBefore"] == ""                                               # A itself has nothing to say
    assert len(out["warns"]) == 1 and out["warns"][0]["side"] == "right"
    w = out["warns"][0]["text"]
    assert w.startswith("右侧结果 注意：输入几何有 1 项超出发布包参考范围（左髂总长度 178 mm，参考 21.1–141 mm）；人群参照需要复核；模型集成质量：存在不确定性，建议复核。")
    assert out["expected"] in w                                                 # the classic banner sentence, ns.notice.model
    assert out["badge"]["text"] == "需复核" and out["badge"]["title"] == "右侧结果：" + out["expected"]
    assert out["badgeA"] is False
    for d in (out["dlg"], out["dlg2"]):
        assert "参考范围与质量" in d and "左髂总长度 178 mm（参考 21.1–141 mm）" in d and "3 个模型一致性" in d
        assert "逐点离散度只保存在质量审计文件里" in d                           # no dispersion numbers
    assert out["tabNow"] == "overview" and out["warnsNow"] == 0 and out["stillThere"] == 1
    assert out["after"] == 0 and out["compare"] is False


def test_side_warnings_model_both_sides_and_none():
    out = _harness(_WARN + r"""
      warn('A');
      await boot();
      const C = ns.compare, mA = B('A'), mB = B('B');
      done({both: C.sideWarnings(mA, mA).map(w => [w.side, w.label, w.target, w.items.length]), left: C.sideWarnings(mA, mB).map(w => w.side),
            none: C.sideWarnings(mB, mB), missing: C.sideWarnings(null, undefined)});
    """, _files("ws_notice.js"))
    assert out["errors"] == [], out["errors"]
    assert out["both"] == [["left", "左侧", "reference", 3], ["right", "右侧", "reference", 3]]
    assert out["left"] == ["left"] and out["none"] == [] and out["missing"] == []


# ----------------------------------------------------------------------------------------------- W13 / W27 / V37
JOB_ROOTS = [Path(__file__).resolve().parents[1] / "outputs" / "wss_deploy_jobs", Path(__file__).resolve().parents[1] / "outputs" / "wss_deploy_preview_jobs"]
CYCLE_JOB, PEAK_JOB, VOLUME_JOB = "20260929_230442_a7273f1670e4", "20260920_144135_673ccd0e36b1", "20260929_230450_d1428792b846"
PURE = [STATIC_DIR / "report_common.js", STATIC_DIR / "volume_viewer.js", V2 / "core_util.js", V2 / "ws_ui.js", V2 / "ws_icons.js", V2 / "ws_overview.js"]


def _job(job_id: str, keys=()) -> tuple[dict, dict, dict]:
    import base64

    from wss_deploy import v2_data
    for root in JOB_ROOTS:
        d = root / job_id
        if (d / "report.html").is_file() and (d / "job.json").is_file():
            data = v2_data.load(d)
            manifest = v2_data.build_manifest(data, json.loads((d / "job.json").read_text(encoding="utf-8")))
            arrays = {}
            for key in keys:
                spec = data.report.specs.get(key)
                if spec is not None:
                    arrays[key] = {"dtype": spec.dtype, "b64": base64.b64encode(data.report.array(key).tobytes()).decode("ascii")}
            return manifest, arrays, data.report.meta
    pytest.skip(f"job {job_id} is not in this working copy")


def _pure(script: str, data: dict, tmp_path: Path, pre: str = "") -> dict:
    _need_node()
    blob = tmp_path / "data.json"
    blob.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    program = ("globalThis.document={getElementById:()=>null};\n" + "".join("require(" + json.dumps(str(f)) + ");\n" for f in PURE) +
               "const ns=globalThis.WSSV2, OV=ns.overview, U=ns.ui, C=globalThis.WssReportCommon, VV=globalThis.VolumeViewerCore;\n"
               "const out=x=>console.log(JSON.stringify(x));\n"
               "const DATA=JSON.parse(require('fs').readFileSync(" + json.dumps(str(blob)) + ",'utf8'));\n"
               "const TYPES={float32:Float32Array,uint32:Uint32Array,int32:Int32Array,uint8:Uint8Array};\n"
               "const ARR={};for(const [k,a] of Object.entries(DATA.arrays||{})){const b=Buffer.from(a.b64,'base64');ARR[k]=new TYPES[a.dtype](b.buffer.slice(b.byteOffset,b.byteOffset+b.length));}\n"
               + pre + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});")
    path = tmp_path / "prog.js"
    path.write_text(program, encoding="utf-8")
    result = subprocess.run(["node", str(path)], text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def _cut(src: str, start: str, end: str) -> str:
    a = src.index(start)
    return src[a:src.index(end, a)]


def test_marker_locations_equal_the_classic_report(tmp_path):
    """W13: branch, distance from the inlet, distance to the junction and local radius of every wall field's maximum (and
    TAWSS's minimum) equal the classic report's fieldPeak rows (its source cut out of report.py and run on the same
    embedded arrays); the value reads the same through the workspace's number format."""
    src = (Path(__file__).resolve().parents[1] / "wss_deploy" / "report.py").read_text(encoding="utf-8")
    keys = ["f.wss.r", "f.tawss.r", "f.osi.r", "f.rrt.r", "f.ecap.r", "pv", "ps", "p_s", "p_dj", "p_r"]
    manifest, arrays, meta = _job(CYCLE_JOB, keys)
    classic = ("function fmt(x,d=2){return x===null||x===undefined||!Number.isFinite(+x)?'—':(+x).toFixed(d);}\n"
               + _cut(src, "const DERIVED_DEF=", "(function deriveInViewer(){") + "\n"
               + "const BRANCH_NAMES=DATA.branch_names;\n" + _cut(src, "const branchName=sid=>", "\n") + "\n"
               "const PEAK=DATA.peak,PV=ARR.pv,PS=ARR.ps,P_S=ARR.p_s,P_DJ=ARR.p_dj,P_R=ARR.p_r;\n"
               # the classic viewer's derive step (deriveInViewer calc) on the embedded TAWSS / OSI point arrays
               "const calc=(f,ta,oa)=>{const out=new Float32Array(ta.length);for(let i=0;i<ta.length;i++){const t=ta[i],o=oa[i];out[i]=Number.isFinite(t)&&Number.isFinite(o)?f(Math.max(t,.01),Math.min(Math.max(o,0),.5)):NaN;}return out;};\n"
               "const XP={wss:ARR['f.wss.r'],tawss:ARR['f.tawss.r'],osi:ARR['f.osi.r']};XP.rrt=calc(DERIVED_DEF.rrt.f,XP.tawss,XP.osi);XP.ecap=calc(DERIVED_DEF.ecap.f,XP.tawss,XP.osi);\n"
               "function fieldArrays(id){return {id,p:XP[id]};}\n"
               + _cut(src, "const PEAK_CACHE={}", "const HAS_STAG=") + "\n")
    out = _pure("""
      const M=DATA.manifest, rows={};
      for (const fid of ['wss','tawss','osi','rrt','ecap']) {
        const v2=OV.extremesModel(M, fid, k=>ARR[k]||null).map(e=>({kind:e.kind, value:e.value, where:OV.extremeWhere(e), text:U.sig(e.value)}));
        const cl=[fieldPeak(fid), fid==='tawss'?fieldPeak(fid,true):null].filter(Boolean).map((pk,i)=>({kind:i?'min':'max', value:pk.value,
          where:`${pk.branch}，距入口 ${fmt(pk.s,0)} mm；距分叉 ${fmt(pk.dj,0)} mm；局部半径 ${fmt(pk.r,1)} mm`, text:C.formatValue(pk.value).replace('-','−')}));
        rows[fid]={v2, cl, keys:OV.extremeKeys(M, fid)};
      }
      out(rows);
    """, {"manifest": manifest, "arrays": arrays, "branch_names": meta.get("branch_names"), "peak": meta.get("peak")}, tmp_path, pre=classic)
    for fid, row in out.items():
        assert row["v2"], fid
        assert [r["kind"] for r in row["v2"]] == [r["kind"] for r in row["cl"]], fid
        for a, b in zip(row["v2"], row["cl"]):
            assert a["where"] == b["where"], (fid, a, b)
            assert a["value"] == pytest.approx(b["value"], rel=1e-6), fid
            assert a["text"] == b["text"], fid
    assert out["wss"]["v2"][0]["where"] == "左髂内，距入口 263 mm；距分叉 14 mm；局部半径 2.6 mm" and out["wss"]["keys"] == []   # analysis.peak
    assert [r["kind"] for r in out["tawss"]["v2"]] == ["max", "min"]                                               # the white marker too
    assert "f.osi.r" in out["osi"]["keys"] and "p_dj" in out["osi"]["keys"]


def test_key_numbers_follow_the_field_and_show_the_wss_p99(tmp_path):
    """W27: a three-head result shows the TAWSS numbers, and while its peak-frame WSS is on screen the whole-field WSS
    p99 (analysis.peak.p99_pa, the classic p99 card) with the low / high fractions; a single-frame result is unchanged."""
    cycle, _, _ = _job(CYCLE_JOB)
    peak, _, _ = _job(PEAK_JOB)
    out = _pure("""
      const k=(m,f)=>OV.kpiModel(m,f).map(x=>[x.key, x.value]);
      out({def:k(DATA.cycle), tawss:k(DATA.cycle,'tawss'), osi:k(DATA.cycle,'osi'), wss:k(DATA.cycle,'wss'), peak:k(DATA.peak), peakWss:k(DATA.peak,'wss')});
    """, {"cycle": cycle, "peak": peak}, tmp_path)
    assert [x[0] for x in out["def"]] == ["tawss_mean", "tawss_low", "stagnation", "max_diameter"]
    assert out["tawss"] == out["def"] and out["osi"] == out["def"]
    wss = dict(out["wss"])
    assert list(wss) == ["wss_p99", "wss_low", "wss_high", "max_diameter"]
    assert wss["wss_p99"] == cycle["analysis"]["peak"]["p99_pa"] == pytest.approx(17.006174, rel=1e-6)
    assert wss["wss_low"] == cycle["fields"][0]["statistics"]["area_frac_low"]
    assert out["peak"] == out["peakWss"] and out["peak"][0] == ["wss_p99", peak["analysis"]["peak"]["p99_pa"]]


def test_volume_statistics_equal_the_classic_volume_page(tmp_path):
    """V37: point count, mean, p99 and maximum of speed and relative pressure equal the classic volume page's setStats
    rows (VolumeViewerCore.statistics over the interior points, speedField of the embedded velocity) as formatted there."""
    manifest, arrays, _ = _job(VOLUME_JOB, ["vpts", "vis_wall", "f.velocity.r", "f.pressure.r"])
    out = _pure("""
      const M=DATA.manifest, inside=VV.insideIndices(ARR.vpts, ARR.vis_wall);
      const speed=VV.speedField(ARR['f.velocity.r']), classic={speed:VV.statistics(speed, inside), pressure:VV.statistics(ARR['f.pressure.r'], inside)};
      const fmt=v=>C.formatValue(v);
      const rows=OV.volumeStatsModel(M).rows.map(r=>({field:r.field, label:r.label, units:r.units, count:String(r.count), mean:fmt(r.mean), p99:fmt(r.p99), max:fmt(r.max), sig:[U.sig(r.mean),U.sig(r.p99),U.sig(r.max)]}));
      const cl=Object.fromEntries(Object.entries(classic).map(([k,s])=>[k,{count:String(s.count), mean:fmt(s.mean), p99:fmt(s.p99), max:fmt(s.max)}]));
      out({rows, cl, wall:OV.volumeStatsModel({result:{family:'wall'}}), none:OV.volumeStatsModel({result:{family:'volume'}, analysis:{}, fields:[]})});
    """, {"manifest": manifest, "arrays": arrays}, tmp_path)
    rows = {r["field"]: r for r in out["rows"]}
    assert list(rows) == ["speed", "pressure"] and rows["speed"]["label"] == "速度" and rows["pressure"]["label"] == "相对压力"
    for field in ("speed", "pressure"):
        for key in ("count", "mean", "p99", "max"):
            assert rows[field][key] == out["cl"][field][key], (field, key)
        assert [x.replace("−", "-") for x in rows[field]["sig"]] == [out["cl"][field][k] for k in ("mean", "p99", "max")]   # what the table shows
    assert rows["speed"]["count"] == "20000"
    assert out["wall"] is None and out["none"] is None


_TABLES = r"""
const TABLES = {
  pv: new Float32Array([0, 0, 0, 1, 0, 10, 2, 0, 20, 3, 0, 30]), ps: new Uint8Array([0, 0, 2, 2]),
  p_s: new Float32Array([5.4, 12.6, 40.2, 61.7]), p_dj: new Float32Array([30.1, 22.5, 3.44, 18.0]), p_r: new Float32Array([9.84, 9.5, 4.26, 4.0]),
  'f.tawss.r': new Float32Array([0.3, 0.05, 2.5, NaN]), 'f.osi.r': new Float32Array([0.1, 0.2, 0.3, 0.3])};
const preloads = [];
const _load = ns.data.loadResult;
ns.data.loadResult = (source, o) => _load(source, o).then(r => {
  const loaded = new Set(), arr = r.array;
  r.has = k => TABLES[k] ? loaded.has(k) : true;
  r.preload = async keys => { (keys || []).forEach(k => loaded.add(k)); preloads.push((keys || []).slice()); return r; };
  r.array = k => TABLES[k] || arr(k);
  return r;
});
const patchCycle = id => {
  const m = canned['/api/v2/jobs/' + id + '/manifest'].body;
  m.geometry.points = {xyz: 'pv', segment: 'ps', s_from_root_mm: 'p_s', dist_to_junction_mm: 'p_dj', radius_mm: 'p_r'};
  Object.keys(TABLES).forEach(k => { m.arrays[k] = {dtype: 'float32', shape: [TABLES[k].length]}; });
  m.analysis.peak = {p99_pa: 17.006, max_pa: 52.6, branch: '左髂内', s_from_inlet_mm: 263.43, dist_to_junction_mm: 14.43, local_radius_mm: 2.61, xyz_mm: [1, 2, 3]};
  m.fields[0].statistics = {p99: 17.006, area_frac_low: 0.408, area_frac_high: 0.166};
  m.fields[1].statistics = {mean: 0.66, area_frac: {low: 0.6}};
  m.analysis.cycle = {stagnation: {area_frac: 0.31, area_mm2: 12000}};
};
"""


def test_overview_shows_marker_locations_key_numbers_by_field_and_flies_to_the_marker():
    out = _harness(_TABLES + r"""
      patchCycle('A');
      await boot();
      await hashTo('#/job/A?f=tawss', 300);
      const S = shellState(), ov = () => byClass(app(), 'insp-body')[0];
      const rows = () => byClass(ov(), 'kpi-where-row').map(e => ({kind: e.dataset.kind, text: textOf(e), title: e.title}));
      const kpis = () => byClass(ov(), 'kpi-label').map(textOf);
      const tawss = {rows: rows(), kpis: kpis(), preloads: preloads.slice()};
      vcalls.length = 0;
      fire(byClass(ov(), 'kpi-where-row')[0], 'click'); await wait(30);
      const flew = vcalls.filter(c => c.endsWith(':setCamera')).length;
      await hashTo('#/job/A?f=wss', 300);
      const wss = {field: S.cur.field, rows: rows(), kpis: kpis()};
      await hashTo('#/job/A?f=osi', 300);
      const osi = {rows: rows(), kpis: kpis()};
      done({tawss, flew, wss, osi});
    """, _files())
    assert out["errors"] == [], out["errors"]
    t = out["tawss"]
    assert [r["kind"] for r in t["rows"]] == ["max", "min"]
    assert t["rows"][0]["text"] == "最大 2.50 Pa左髂总，距入口 40 mm；距分叉 3 mm；局部半径 4.3 mm"
    assert t["rows"][1]["text"] == "最小 0.0500 Pa主动脉，距入口 13 mm；距分叉 23 mm；局部半径 9.5 mm"
    assert t["rows"][0]["title"].startswith("黄色标记：TAWSS 全场最大值预测点") and t["rows"][1]["title"].startswith("白色标记：TAWSS 全场最小值预测点")
    assert t["kpis"][:2] == ["TAWSS 均值", "低剪切占比"]
    assert any("f.tawss.r" in p and "p_dj" in p for p in t["preloads"])          # loaded once, then drawn again
    assert out["flew"] == 1
    w = out["wss"]
    assert w["field"] == "wss" and w["kpis"][:3] == ["WSS p99", "低 WSS 占比", "高 WSS 占比"]          # W27
    assert w["rows"] == [{"kind": "max", "text": "最大 52.6 Pa左髂内，距入口 263 mm；距分叉 14 mm；局部半径 2.6 mm",
                          "title": "黄色标记：WSS 全场最大值预测点；点一下转到这里"}]
    assert out["osi"]["kpis"][:2] == ["TAWSS 均值", "低剪切占比"] and [r["kind"] for r in out["osi"]["rows"]] == ["max"]


def test_overview_volume_statistics_table():
    out = _harness(r"""
      const m = canned['/api/v2/jobs/A/manifest'].body;
      m.result.family = 'volume';
      m.fields = [{id: 'pressure', short_label: '压力', units: 'Pa', kind: 'scalar', temporal: 'frame', tier: 'model', display: {}, statistics: {count: 20000, mean: -303.87, p99: 99.1, max: 130.56}},
                  {id: 'speed', short_label: '速度', units: 'm/s', kind: 'scalar', temporal: 'frame', tier: 'model', display: {p99: 1.37}}];
      m.analysis.zones = null;
      m.analysis.volume_statistics = {speed_m_s: {count: 20000, mean: 0.4221, p99: 1.3719, max: 1.5715}};
      await boot();
      await hashTo('#/job/A', 300);
      const tbl = byClass(byClass(app(), 'insp-body')[0], 'tbl-vstats')[0];
      done({head: walk(tbl, e => e.tagName === 'TH').map(textOf), cells: walk(tbl, e => e.tagName === 'TR').slice(1).map(tr => walk(tr, e => e.tagName === 'TD').map(textOf)),
            where: byClass(app(), 'kpi-where').length});
    """, _files())
    assert out["errors"] == [], out["errors"]
    assert out["head"] == ["物理量", "点数", "均值", "p99", "最大"]
    assert out["cells"] == [["速度 m/s", "20000", "0.422", "1.37", "1.57"], ["相对压力 Pa", "20000", "−304", "99.1", "131"]]
    assert out["where"] == 0


# ----------------------------------------------------------------------------------------------- W58 / W59
def test_ctrl_p_goes_to_the_print_view_and_the_browser_print_gets_a_snapshot():
    _need_node()
    program = r"""
      const winEv = {}, docEv = {};
      class El { constructor(t) { this.tagName = t.toUpperCase(); this.children = []; this.parentNode = null; this.attrs = {}; this.className = ''; this.width = 0; this.height = 0; this.drawn = []; }
        insertBefore(x, ref) { const i = ref ? this.children.indexOf(ref) : -1; x.parentNode = this; if (i < 0) this.children.push(x); else this.children.splice(i, 0, x); return x; }
        removeChild(x) { this.children.splice(this.children.indexOf(x), 1); x.parentNode = null; return x; }
        appendChild(x) { x.parentNode = this; this.children.push(x); return x; }
        setAttribute(k, v) { this.attrs[k] = String(v); }
        get nextSibling() { const p = this.parentNode; return p ? p.children[p.children.indexOf(this) + 1] || null : null; }
        getContext(kind) { const self = this; return kind === '2d' ? {drawImage(src) { self.drawn.push(src); }} : null; } }
      const cls = new Set();
      globalThis.document = {body: {classList: {add: c => cls.add(c), remove: c => cls.delete(c), contains: c => cls.has(c)}}, createElement: t => new El(t),
        addEventListener: (n, f, cap) => { (docEv[n] = docEv[n] || []).push([f, cap]); }};
      globalThis.addEventListener = (n, f) => { (winEv[n] = winEv[n] || []).push(f); };
      globalThis.WSSV2 = {ui: {toast() {}}};
      const F = require(@@fig@@);
      const wrap = new El('div'), gl = new El('canvas'), labels = new El('div'); gl.width = 800; gl.height = 600; wrap.appendChild(gl); wrap.appendChild(labels);
      const renders = [];
      const viewerA = {canvasElement: gl, renderNow() { renders.push('A'); }, renderPose() {}, isDisposed: () => false};
      const viewerB = {canvasElement: new El('canvas'), renderNow() { renders.push('B'); }};      // hidden: 0 × 0
      let cur = {manifest: {result: {family: 'wall'}}};
      const api = {cur: () => cur, viewer: () => viewerA, state: () => ({viewerA, viewerB}), hideName: () => true};
      F._setApi(api);
      const printed = [];
      F.printView = (a, o) => { printed.push(o.hideName); return Promise.resolve(); };
      const key = (k, extra) => { const e = Object.assign({key: k, prevented: false, preventDefault() { this.prevented = true; }}, extra || {}); docEv.keydown.forEach(([f]) => f(e)); return e.prevented; };
      const r = {capture: docEv.keydown.map(x => x[1]), ctrlP: key('p', {ctrlKey: true}), metaP: key('P', {metaKey: true}), plainP: key('p'), shiftP: key('p', {ctrlKey: true, shiftKey: true})};
      await new Promise(res => setTimeout(res, 10));
      r.printed = printed.slice();
      cur = null; r.home = key('p', {ctrlKey: true}); cur = {manifest: {result: {family: 'wall'}}};
      // the browser's menu: beforeprint lays a copy over the WebGL canvas (drawn now, same task), afterprint removes it
      winEv.beforeprint.forEach(f => f());
      const snap = wrap.children[1];
      r.before = {order: wrap.children.map(c => c.tagName + ':' + c.className), drawn: snap.drawn.length && snap.drawn[0] === gl, size: [snap.width, snap.height], renders: renders.slice(),
        hidden: snap.attrs['aria-hidden'], count: F.printSnaps().length};
      winEv.afterprint.forEach(f => f());
      r.after = {order: wrap.children.map(c => c.tagName), count: F.printSnaps().length};
      cls.add('ws-printing'); winEv.beforeprint.forEach(f => f()); r.duringPrintView = wrap.children.length; cls.delete('ws-printing');
      console.log(JSON.stringify(r));
    """.replace("@@fig@@", json.dumps(str(V2 / "ws_figure.js")))
    result = subprocess.run(["node", "-e", "(async()=>{" + program + "})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr[-3000:]
    r = json.loads(result.stdout.strip().splitlines()[-1])
    assert r["capture"] == [True]                                         # before the shell's own key handling
    assert r["ctrlP"] is True and r["metaP"] is True and r["plainP"] is False and r["shiftP"] is False
    assert r["printed"] == [True, True]                                   # the print-view page, presentation name hidden
    assert r["home"] is False                                             # no result open: the browser prints
    b = r["before"]
    assert b["order"] == ["CANVAS:", "CANVAS:ws-print-snap", "DIV:"]     # over the WebGL canvas, under the label chips
    assert b["drawn"] is True and b["size"] == [800, 600] and b["renders"] == ["A"] and b["hidden"] == "true" and b["count"] == 1
    assert r["after"] == {"order": ["CANVAS", "DIV"], "count": 0}
    assert r["duringPrintView"] == 2                                      # printView's own A4 page needs no copy


def test_print_snapshot_is_print_only_css():
    css = (V2 / "v2_export.css").read_text(encoding="utf-8")
    assert ".ws-print-snap { display: none; }" in css
    tail = css[css.index("the browser's own print"):]
    assert "@media print" in tail and "position: absolute; inset: 0;" in tail
