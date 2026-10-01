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
