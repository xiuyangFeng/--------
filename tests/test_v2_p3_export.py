"""Node tests for phase 3 lane 5 (PHASE3_LANES.md §3 第 5 路): exports, links and comparison of the v2 workspace.

Covers: the reproducible link's display state (d) and extension states (x) next to the lane A state, old v:1 links
left alone; the zip writer; the 拼图 page's 「分成单张」 output; the single-file downloads by family (#93); the
comparison's identity lines (#112), scalar table from POST /api/compare (C5) and searchable picker (#98); the batch
figures module ws_batch.js (#56): plan, settings, links, field / window choice, the display provider for the offscreen
viewer, the run (files, folders, manifest.json, failures, stop, clean-up) and the dialog.

WebGL and 2-D canvases are not available in Node: the pixels (batch figures, six views as files) are checked in the
sandbox browser (lane report wss_deploy/lanes/p3_export.md).
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import zipfile

import pytest

from wss_deploy.paths import STATIC_DIR

V2 = STATIC_DIR / "v2"
FILES = [STATIC_DIR / "three.min.js", STATIC_DIR / "report_common.js", V2 / "core_util.js", V2 / "core_colormap.js", V2 / "core_colorbar.js",
         V2 / "core_viewer.js", V2 / "ws_icons.js", V2 / "ws_ui.js", V2 / "ws_store.js", V2 / "ws_display.js", V2 / "ws_bookmarks.js",
         V2 / "ws_compare.js", V2 / "ws_export.js", V2 / "ws_figure.js", V2 / "ws_batch.js", V2 / "ws_shell.js"]

# Stub DOM (no innerHTML, no querySelectorAll): enough for ws_ui.h / fill / button / select / section / dialog.
DOM = r"""
class TextNode { constructor(t) { this.nodeValue = String(t); this.parentNode = null; } get textContent() { return this.nodeValue; } }
class Element {
  constructor(tag) {
    this.tagName = String(tag || 'div').toUpperCase(); this.children = []; this.parentNode = null; this.events = {}; this.attrs = {};
    this.style = {}; this.dataset = {}; this.hidden = false; this.disabled = false; this.checked = false; this.value = ''; this.open = false;
    this._text = ''; this.className = ''; this.id = ''; this.title = ''; this.type = '';
    const self = this, set = () => new Set(String(self.className).split(/\s+/).filter(Boolean));
    this.classList = {add: (...c) => { const s = set(); c.forEach(x => s.add(x)); self.className = [...s].join(' '); },
      remove: (...c) => { const s = set(); c.forEach(x => s.delete(x)); self.className = [...s].join(' '); },
      toggle: (c, f) => { const s = set(); if (f === undefined) f = !s.has(c); f ? s.add(c) : s.delete(c); self.className = [...s].join(' '); return f; },
      contains: c => set().has(c)};
  }
  get ownerDocument() { return globalThis.document; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  appendChild(x) { if (x.parentNode) x.parentNode.removeChild(x); x.parentNode = this; this.children.push(x); return x; }
  append(...xs) { xs.forEach(x => this.appendChild(typeof x === 'object' ? x : new TextNode(x))); }
  removeChild(x) { const i = this.children.indexOf(x); if (i >= 0) this.children.splice(i, 1); x.parentNode = null; return x; }
  replaceChildren(...xs) { this.children.forEach(c => { c.parentNode = null; }); this.children = []; this._text = ''; xs.forEach(x => this.appendChild(x)); }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  addEventListener(n, f) { (this.events[n] = this.events[n] || []).push(f); }
  removeEventListener(n, f) { const l = this.events[n] || []; const i = l.indexOf(f); if (i >= 0) l.splice(i, 1); }
  setAttribute(k, v) { this.attrs[k] = String(v); } getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; } removeAttribute(k) { delete this.attrs[k]; }
  focus() {} blur() {} select() {} click() { fire(this, 'click'); } showModal() { this.open = true; } close() { this.open = false; fire(this, 'close'); }
  getContext() { return null; }
}
const body = new Element('body');
const walk = (el, pred, out = []) => { if (!el || typeof el !== 'object') return out; if (el.tagName && pred(el)) out.push(el); for (const c of (el.children || [])) walk(c, pred, out); return out; };
const byId = id => walk(body, e => e.id === id)[0] || null;
global.document = {body, activeElement: body, getElementById: byId, createElement: t => new Element(t), createElementNS: (n, t) => new Element(t), createTextNode: t => new TextNode(t),
  addEventListener() {}, removeEventListener() {}};
global.window = global;
global.addEventListener = () => {}; global.removeEventListener = () => {};
global.location = {hash: '', href: 'http://127.0.0.1:8765/v2/'};
global.history = {replaceState(s, t, url) { global.location.hash = String(url).slice(String(url).indexOf('#')); }};
global.localStorage = {_m: new Map(), getItem(k) { return this._m.has(k) ? this._m.get(k) : null; }, setItem(k, v) { this._m.set(k, String(v)); }, removeItem(k) { this._m.delete(k); }};
global.matchMedia = () => ({matches: false});
const fire = (el, name, extra) => { for (const f of [...(el.events[name] || [])]) f(Object.assign({preventDefault() {}, stopPropagation() {}, target: el}, extra || {})); };
const textOf = el => el ? el.textContent : '';
const byText = (root, t) => walk(root, e => e.tagName === 'BUTTON' && textOf(e) === t)[0] || null;
const byClass = (root, c) => walk(root, e => String(e.className || '').split(/\s+/).includes(c));
const wait = ms => new Promise(r => setTimeout(r, ms));
"""

# A fake shell API (as lane A's): records what the figure module asks the shell and the viewer to do.
FAKE_API = r"""
function fakeApi(opts) {
  opts = opts || {};
  const calls = [];
  const viewer = {getState: () => ({camera: {position: [1, 2, 3], target: [0, 0, 0], up: [0, 0, 1], fov: 30}, cursor: null, selection: null, branches: null,
      display: opts.display || undefined}),
    applyState: (s) => { calls.push('applyState:' + Object.keys(s).sort().join(',') + (s.display ? ':' + JSON.stringify(s.display) : '')); return Promise.resolve(s); },
    hasAnatomicalFrame: () => true, renderPose: () => Promise.resolve({}), setLighting: l => calls.push('setLighting:' + l)};
  const fields = opts.fields || [{id: 'wss', units: 'Pa', display: {thresholds: [0.4, 4, 7]}}, {id: 'tawss', units: 'Pa', display: {thresholds: [0.4, 4, 7]}},
    {id: 'osi', units: '1', display: {thresholds: [0.1, 0.2, 0.3]}}, {id: 'velocity', kind: 'vector', units: 'm/s'}];
  const cur = {jobId: 'J', runIdentity: 'run-1234567890abcdef-long', field: opts.field || 'tawss', window: 'adaptive',
    manifest: {job: {id: 'J', display_name: 'CASE'}, result: {family: opts.family || 'wall', run_identity: 'run-1234567890abcdef-long'}, fields: fields},
    result: {manifest: {}}, slice: null, cursor: null, job: {}};
  const S = {layers: {outline: false, centerline: false, points: false, trust: false, streamlines: false, wall: true, interior: false}};
  const api = {calls, cur: () => cur, viewer: () => viewer, state: () => S, offline: () => false, hideName: () => false,
    parseHash: ns.shell.parseHash, buildHash: ns.shell.buildHash, fieldById: (m, id) => (m.fields || []).find(f => f.id === id) || null,
    applyField: (f, w) => { calls.push('applyField:' + f + ':' + JSON.stringify(w)); cur.field = f; cur.window = w; return true; },
    setLayers: () => calls.push('setLayers'), saveViewSoon: () => calls.push('save'), drawLabels: () => calls.push('drawLabels'), renderToolbar: () => calls.push('renderToolbar'),
    toggleCursor: () => calls.push('toggleCursor'), toggleSlice: () => calls.push('toggleSlice'), exitSlice: () => calls.push('exitSlice'), openExport: () => calls.push('openExport'),
    jobs: () => opts.jobs || [], cards: () => ({}),
    ui: () => ns.ui, h: ns.ui.h};
  return api;
}
"""


def _node(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is required for the lane 5 tests")
    prelude = DOM + "".join("require(" + json.dumps(str(f)) + ");\n" for f in FILES)
    prelude += ("const ns=globalThis.WSSV2, F=ns.figure, X=ns.exporter, B=ns.batch, CMP=ns.compare, D=ns.display, ST=ns.store;"
                "const out=x=>console.log(JSON.stringify(x));\n")
    program = prelude + FAKE_API + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------------------------------------------
def test_scripts_parse_bundle_and_offline():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    for name in ("ws_batch.js", "ws_figure.js", "ws_export.js", "ws_compare.js", "ws_bookmarks.js"):
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    s = bundle["scripts"]
    assert s.index("ws_figure.js") < s.index("ws_batch.js") < s.index("ws_shell.js")
    assert "ws_batch.js" in bundle["offline_exclude"]            # batch figures need the service
    assert "ws_figure.js" not in bundle["offline_exclude"]       # six views as files work in the offline report too
    css = (V2 / "v2_export.css").read_text(encoding="utf-8")
    assert ".wsb-log" in css and ".tbl-scalar" in css and ".pick-search" in css


def test_zip_store_is_a_valid_zip():
    out = _node("""
      const enc = s => new TextEncoder().encode(s);
      const zip = F.zipStore([{name: 'a/图 1.png', bytes: enc('PNG-1')}, {name: 'a/图 1.png', bytes: enc('PNG-2')}, {name: '/m.json', bytes: '{"x":1}'}]);
      out({b64: Buffer.from(zip).toString('base64'), crc: F.crc32(enc('123456789')).toString(16)});
    """)
    assert out["crc"] == "cbf43926"                                  # the standard CRC-32 check value
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(out["b64"])))
    assert z.testzip() is None
    assert z.namelist() == ["a/图 1.png", "a/图 1_2.png", "m.json"]
    assert z.read("a/图 1.png") == b"PNG-1" and z.read("a/图 1_2.png") == b"PNG-2" and json.loads(z.read("m.json")) == {"x": 1}
    assert all(i.flag_bits & 0x800 for i in z.infolist())          # UTF-8 names


# ---------------------------------------------------------------------------------------------------------------
# Reproducible link: the display (W55)
def test_link_state_carries_the_effective_display():
    out = _node("""
      ST.setPrefs({cmap: 'viridis', lighting: 'flat', labels: {branches: true, findings: 5, maxd: false, annotations: true}});
      const P = D.prefs(); P.defaults.osi = {bands: 8, thresholds: [0.05, 0.15, 0.25]}; P.units.pressure = 'mmHg'; P.layers.stagnation = true; D.savePrefs(P);
      const api = fakeApi({display: {v: 1, bands: {tawss: 6}, thresholds: {tawss: [0.3, 3, 6]}}});
      const st = F.viewState(api);
      const back = F.decodeState(F.encodeState(st));
      const vol = F.viewState(fakeApi({family: 'volume', fields: [{id: 'pressure', units: 'Pa', display: {}}, {id: 'speed', units: 'm/s', display: {}}]}));
      out({st, same: JSON.stringify(back) === JSON.stringify(st), volD: vol.d, wallVolume: 'volume' in st.d});
    """)
    st = out["st"]
    assert st["v"] == 1 and st["f"] == "tawss" and out["same"]       # still the v:1 format, read by older readers
    d = st["d"]
    assert d["cmap"] == "viridis" and d["light"] == "flat" and d["labels"] == {"branches": True, "findings": 5, "maxd": False, "annotations": True}
    assert d["bands"] == {"wss": 0, "tawss": 6, "osi": 8}             # this result's choice, else 「设为默认」, else continuous
    assert d["thr"] == {"wss": [0.4, 4, 7], "tawss": [0.3, 3, 6], "osi": [0.05, 0.15, 0.25]}
    # merged with lane 4: wall stresses have a display unit too (Pa / dyn/cm²)
    assert d["units"] == {"pressure": "mmHg", "velocity": "m/s", "wss": "Pa"} and d["layers"]["stagnation"] is True
    assert out["wallVolume"] is False and out["volD"]["volume"]["opacity"] == 0.1
    assert out["volD"]["bands"] == {"pressure": 0, "speed": 0} and out["volD"]["thr"] == {}


def test_old_link_leaves_the_display_alone_and_new_link_applies_it():
    out = _node("""
      const prefs0 = JSON.stringify([ST.prefs(), D.prefs()]);
      // an old (lane A) link: no d — field and window only
      const a = fakeApi({});
      const notesOld = await F.applyView(a, {v: 1, f: 'osi', w: 'adaptive'});
      const oldSame = JSON.stringify([ST.prefs(), D.prefs()]) === prefs0;
      // a link with the display: preferences as if chosen in the menus, the result's bands and thresholds, field drawn again
      const b = fakeApi({});
      const notes = await F.applyView(b, {v: 1, f: 'tawss', w: 'adaptive', d: {cmap: 'turbo', light: 'flat', labels: {branches: true, findings: 3},
        units: {pressure: 'mmHg', bogus: 'x'}, layers: {stagnation: true, peaks: false}, volume: {opacity: 0.3},
        bands: {tawss: 6}, thr: {tawss: [0.3, 3, 6]}}});
      const c = fakeApi({});
      const notesBad = await F.applyView(c, {v: 1, d: {cmap: 'bogus', units: {pressure: 'psi'}}});
      out({notesOld, oldCalls: a.calls, oldSame, notes, calls: b.calls, prefs: ST.prefs(), dp: D.prefs(), notesBad, badCalls: c.calls});
    """)
    assert out["oldSame"] and out["notesOld"] == [] and out["oldCalls"] == ['applyField:osi:"adaptive"', "save"]
    calls = out["calls"]
    disp = [c for c in calls if c.startswith("applyState:display")]
    assert disp == ['applyState:display:{"v":1,"bands":{"tawss":6},"thresholds":{"tawss":[0.3,3,6]}}']
    assert calls.index(disp[0]) < calls.index('applyField:tawss:"adaptive"')      # the field is drawn again with them
    assert "drawLabels" in calls and "setLighting:flat" in calls
    p, dp = out["prefs"], out["dp"]
    assert p["cmap"] == "turbo" and p["lighting"] == "flat" and p["labels"]["branches"] is True and p["labels"]["findings"] == 3
    assert dp["units"]["pressure"] == "mmHg" and dp["layers"]["stagnation"] is True and dp["layers"]["peaks"] is False and dp["volume"]["opacity"] == 0.3
    assert any(n.startswith("已按链接设置") and "色表" in n and "单位" in n for n in out["notes"])
    assert "这里不能用单位 x" in out["notes"]
    assert "这里没有色表 bogus" in out["notesBad"] and "这里不能用单位 psi" in out["notesBad"]
    assert not any(c.startswith("applyField") for c in out["badCalls"])        # nothing to redraw


def test_extension_link_states():
    out = _node("""
      const seen = [];
      (ns.ext = ns.ext || []).push({id: 'contour', linkState: () => ({on: true, levels: [1, 2]}), applyLinkState: (s, api) => { seen.push(s); }});
      ns.ext.push({id: 'quiet', linkState: () => undefined});
      const st = F.viewState(fakeApi({}));
      const notes = await F.applyView(fakeApi({}), {v: 1, x: {contour: {on: false}, gone: {a: 1}}});
      out({x: st.x, seen, notes});
    """)
    # merged with lane 4: its real display extension rides along under x.display; the test stub keeps its own slot
    assert out["x"]["contour"] == {"on": True, "levels": [1, 2]} and set(out["x"]) <= {"contour", "display"}
    assert out["seen"] == [{"on": False}]
    assert out["notes"] == ["链接里有这里没有的显示设置（gone），已跳过"]


# ---------------------------------------------------------------------------------------------------------------
# Six views as separate files (W51), downloads by family (#93)
def test_views_output_option_on_the_montage_page_and_tools():
    out = _node("""
      const api = fakeApi({}); F._setApi(api);
      const el = new Element('div');
      const first = F.viewsOutput();
      F.pane('montage', el, {state: {hideName: false}});
      const before = {buttons: walk(el, e => e.tagName === 'BUTTON').map(textOf), rows: byClass(el, 'exp-k').map(textOf)};
      fire(byText(el, '分成单张'), 'click');
      const after = {buttons: walk(el, e => e.tagName === 'BUTTON').map(textOf), rows: byClass(el, 'exp-k').map(textOf), stored: F.viewsOutput()};
      const ext = ns.ext.find(x => x.id === 'figure');
      const six = walk(ext.tools(api)[0], e => e.tagName === 'BUTTON' && textOf(e) === '六视角')[0];
      out({first, before, after, sixTitle: six.title, figOpts: Object.keys(F.options()).sort()});
    """)
    assert out["first"] == "montage"
    assert "导出拼图 PNG" in out["before"]["buttons"] and "列数" in out["before"]["rows"] and "输出" in out["before"]["rows"]
    assert "导出单张 zip" in out["after"]["buttons"] and "列数" not in out["after"]["rows"] and out["after"]["stored"] == "files"
    assert "zip" in out["sixTitle"]
    assert out["figOpts"] == ["background", "colorbar", "columns", "crop", "labels", "lang", "scale", "title", "views"]   # lane A's options untouched


def test_single_file_downloads_by_family():
    out = _node("""
      const api = fakeApi({}); F._setApi(api);
      ns.api = {urls: {table: () => '/t', file: (j, n) => '/f/' + n, bundle: () => '/b', onepage: () => '/o'}, offline: () => Promise.resolve({})};
      const links = ctx => { X.pendingTab = 'data'; X.open(ctx); const r = walk(byClass(byId('ws-dialog'), 'exp-links')[0], e => e.tagName === 'A').map(a => textOf(a) + '=' + a.attrs.href); ns.ui.dialog.close('done'); return r; };
      const vol = {job: {id: 'V', summary: {exports: {vtp: true, csv: true, wall_pressure: true, streamlines: true}}}, manifest: {job: {id: 'V'}, result: {family: 'volume'}}, bookmarks: []};
      const volNoLines = {job: {id: 'V', summary: {exports: {vtp: true, csv: true, wall_pressure: true}}}, manifest: {job: {id: 'V'}, result: {family: 'volume'}}, bookmarks: []};
      const wall = {job: {id: 'W', summary: {exports: {vtp: true, csv: true}}}, manifest: {job: {id: 'W'}, result: {family: 'wall'}}, bookmarks: []};
      out({vol: links(vol), volNoLines: links(volNoLines), wall: links(wall),
        old: X.dataFiles('wall', {}).map(f => f.file), off: X.dataFiles('wall', {csv: false}).map(f => f.file)});
    """)
    assert out["vol"] == ["统计表 CSV=/t", "体场 VTP=/f/volume_fields.vtp", "壁面压力 VTP=/f/wall_pressure.vtp", "体场 CSV=/f/points_volume.csv",
                          "流线 VTP=/f/streamlines.vtp", "全部文件 zip=/b"]
    assert "流线 VTP=/f/streamlines.vtp" not in out["volNoLines"] and len(out["volNoLines"]) == 5
    assert out["wall"] == ["统计表 CSV=/t", "壁面 VTP=/f/wall_wss.vtp", "点云 CSV=/f/points_wss.csv", "全部文件 zip=/b"]
    assert out["old"] == ["wall_wss.vtp", "points_wss.csv"] and out["off"] == ["wall_wss.vtp"]    # older summaries: the classic always-on links


# ---------------------------------------------------------------------------------------------------------------
# Comparison: identity (#112), scalar table (C5), picker (#98)
CMP_RESPONSE = {"comparison": {"compatible": True, "kind": "cross_case", "reasons": [], "rows": [
    {"id": "peak.p99_pa", "label": "Peak p99", "scope": "peak", "units": "Pa", "left": 11.092094648608903, "right": 10.56651860133792, "delta": -0.5255760472709827},
    {"id": "per_branch.主动脉.p99_pa", "label": "主动脉 p99", "scope": "per_branch", "units": "Pa", "left": 4.537895989401944, "right": 5.211165054141943, "delta": 0.6732690647399986, "branch": "主动脉"},
    {"id": "cycle.osi.mean", "label": "OSI mean", "scope": "cycle", "units": "1", "left": 0.1597321782728351, "right": 0.15995192242654013, "delta": 0.00021974415370504263},
    {"id": "cycle.stagnation.area_frac", "label": "Stagnation area fraction", "scope": "cycle", "units": "1", "left": 0.24843450684953558, "right": 0.1109667725728953, "delta": -0.13746773427664027},
    {"id": "cycle.rrt.mean_per_pa", "label": "RRT mean", "scope": "cycle", "units": "1/Pa", "left": None, "right": None, "delta": None}]}}


def test_scalar_model_identity_and_conditions():
    out = _node("""
      const resp = %s;
      const m = CMP.scalarModel(resp.comparison);
      const bad = CMP.scalarModel({compatible: false, kind: 'same_input', reasons: ['left: missing statistics_protocol', 'WSS thresholds_pa differ', 'something new'],
        rows: [{id: 'peak.max_pa', units: 'Pa', left: 27.5, right: 25.7, delta: null}], cycle: {note: '只有左侧有周期量（TAWSS / OSI），差值留空'}});
      const mf = (o) => ({job: Object.assign({id: 'A', display_name: 'P-DEMO-01', patient_id: 'P-DEMO-01', case_id: 'DEMO_A', scan_label: '第一次扫描', scan_date: '2024-03-15',
        review: {status: 'reviewed'}}, o || {}), mapping: {statistics_protocol: {field: 'f', top5_connectivity_radius_mm: 0.98}}, fields: [{id: 'wss', units: 'Pa', display: {thresholds: [0.4, 4, 7]}}]});
      const id = CMP.identity(mf()), hid = CMP.identity(mf(), true), pat = CMP.identity(mf({display_name: '张三', patient_id: 'P-1'}));
      const B2 = mf(); B2.mapping.statistics_protocol.top5_connectivity_radius_mm = 0.99;
      const C2 = mf(); C2.mapping.statistics_protocol.field = 'other';
      const verdict = (a, b) => CMP.conditions(a, b, 'wss', 'wss').find(r => r.key === 'statistics').verdict;
      out({m, bad, id, hid, pat, sameStat: verdict(mf(), B2), diffStat: verdict(mf(), C2)});
    """ % json.dumps(CMP_RESPONSE))
    rows = out["m"]["rows"]
    assert [r["label"] for r in rows] == ["峰值 p99", "主动脉 p99", "OSI 均值", "滞留区面积占比"]     # service order; empty rows dropped
    assert rows[0]["left"] == "11.1 Pa" and rows[0]["right"] == "10.6 Pa" and rows[0]["delta"] == "−0.526 Pa"
    assert rows[1]["delta"] == "+0.673 Pa" and rows[2]["left"] == "0.160" and rows[2]["delta"] == "+0.000220"
    assert rows[3]["left"] == "25%" and rows[3]["delta"] == "−14 个百分点"
    assert out["m"]["kind"] == "跨病例" and out["m"]["compatible"] is True and out["m"]["reasons"] == []
    assert out["bad"]["reasons"] == ["左侧：缺少统计定义", "WSS 阈值不同", "something new"] and out["bad"]["rows"][0]["delta"] == "—"
    assert out["bad"]["cycleNote"].startswith("只有左侧有周期量")
    assert out["id"]["text"] == "病例 DEMO_A · 第一次扫描 · 2024-03-15" and out["id"]["review"]["status"] == "reviewed"
    assert out["hid"]["text"] == "第一次扫描 · 2024-03-15"
    assert out["pat"]["text"].startswith("患者 P-1 · 病例 DEMO_A")
    assert out["sameStat"] == "一致" and out["diffStat"] == "不一致"          # per-geometry keys ignored, as /api/compare does


def test_compare_panel_scalar_table_and_identity():
    out = _node("""
      const resp = %s, reqs = [];
      let fail = false;
      ns.api = {request: (url, o) => { reqs.push([url, o.method, JSON.stringify(o.body)]); return fail ? Promise.reject(new Error('接口不存在。')) : Promise.resolve(JSON.parse(JSON.stringify(resp))); }};
      const man = (id, name, scan) => ({job: {id, display_name: name, patient_id: 'P-9', case_id: id + '_CASE', scan_label: scan, scan_date: '2025-03-20', review: {status: id === 'A' ? 'reviewed' : 'unreviewed'}},
        result: {display_name: '周期指标', release_id: 'R'}, mapping: {}, fields: [{id: 'tawss', units: 'Pa', display: {}}]});
      const ctx = {left: {manifest: man('A', 'LEFT', '术前'), field: 'tawss', name: 'LEFT', result: '周期指标'}, right: {manifest: man('B', 'RIGHT', '术后'), field: 'tawss', name: 'RIGHT', result: '周期指标'}, mode: 'each'};
      const el = new Element('div');
      CMP.render(el, ctx); await wait(20);
      const text = textOf(el), ids = byClass(el, 'cmp-id').map(textOf), heads = walk(byClass(el, 'tbl-scalar')[0], e => e.tagName === 'TH').map(textOf);
      CMP.render(el, ctx); await wait(20);                         // cached: one request for the pair
      const hidden = new Element('div');
      CMP.render(hidden, Object.assign({}, ctx, {left: Object.assign({}, ctx.left, {name: '病例'})})); await wait(20);
      CMP._clearScalars(); fail = true;
      const failed = new Element('div'); CMP.render(failed, ctx); await wait(20);
      out({text, ids, heads, reqs, hiddenIds: byClass(hidden, 'cmp-id').map(textOf), hiddenNames: byClass(hidden, 'cmp-name').map(textOf), failed: textOf(byClass(failed, 'cmp-scalars')[0])});
    """ % json.dumps(CMP_RESPONSE))
    assert out["reqs"][0] == ["/api/compare", "POST", '{"left_job_id":"A","right_job_id":"B"}'] and len(out["reqs"]) == 2   # 1 + the failing one
    assert "标量对照" in out["text"] and "峰值 p99" in out["text"] and "跨病例 · 口径一致" in out["text"]
    assert out["heads"] == ["指标", "左", "右", "右 − 左"]
    assert out["ids"][0] == "已复核患者 P-9 · 病例 A_CASE · 术前 · 2025-03-20" and out["ids"][1].startswith("未复核")
    assert out["hiddenIds"][0] == "已复核术前 · 2025-03-20" and out["hiddenNames"] == ["病例", "病例"]
    assert out["failed"] == "标量对照不可用：接口不存在。"


def test_picker_has_every_result_and_a_search():
    out = _node("""
      const jobs = [];
      for (let i = 0; i < 30; i++) jobs.push({id: 'J' + i, status: 'done', case_id: 'CASE_' + i, patient_id: 'P' + i, created_at: '2026-09-0' + (i % 9 + 1) + 'T00:00:00Z'});
      jobs.push({id: 'K1', status: 'done', case_id: 'DEMO', patient_id: 'P-DEMO', scan_label: '第二次扫描', tags: ['随访']});
      jobs.push({id: 'K2', status: 'failed', case_id: 'DEMO_BAD'});
      const all = CMP.candidates(jobs, {id: 'J0'});
      const q = CMP.candidates(jobs, {id: 'J0'}, 'demo 第二');
      const tag = CMP.candidates(jobs, {id: 'J0'}, '随访').others.map(j => j.id);
      let picked = null;
      CMP.pickDialog({jobs, current: {id: 'J0'}, cards: {}, onPick: id => { picked = id; }});
      const dlg = byId('ws-dialog'), search = byClass(dlg, 'pick-search')[0];
      const n0 = byClass(dlg, 'pick').length, c0 = textOf(byClass(dlg, 'pick-count')[0]);
      search.value = 'case_1'; fire(search, 'input');
      const n1 = byClass(dlg, 'pick').length, c1 = textOf(byClass(dlg, 'pick-count')[0]);
      search.value = 'nothing-here'; fire(search, 'input');
      const empty = textOf(byClass(dlg, 'pick-groups')[0]);
      search.value = 'k1'; fire(search, 'input');
      fire(byClass(dlg, 'pick')[0], 'click');
      out({all: all.others.length, q: q.others.map(j => j.id), tag, n0, c0, n1, c1, empty, picked});
    """)
    assert out["all"] == 30                                     # no 20-result cap (29 others + K1)
    assert out["q"] == ["K1"] and out["tag"] == ["K1"]
    assert out["n0"] == 30 and out["c0"] == "30 个"
    assert out["n1"] == 11 and out["c1"] == "11 / 30"           # CASE_1, CASE_10..19
    assert "没有匹配" in out["empty"] and out["picked"] == "K1"


# ---------------------------------------------------------------------------------------------------------------
# Batch figures (#56)
JOBS = r"""
const cyc = {model_release: {id: 'M1', contract: {protocol: 'single_frame_wss_cycle_multi', fields: {tawss: {}}}}};
const vol = {model_release: {id: 'PF', contract: {protocol: 'single_frame_volume', fields: {velocity: {}}}}};
const JOBS = [Object.assign({id: 'A', status: 'done', case_id: 'CASE_A'}, cyc), Object.assign({id: 'V', status: 'done', case_id: 'CASE_A'}, vol),
  {id: 'W', status: 'done', case_id: 'CASE_W', model_release: {id: 'X5D', contract: {protocol: 'single_frame_wss', fields: {wss: {}}}}},
  {id: 'Q', status: 'awaiting_confirmation', case_id: 'CASE_Q'}, {id: 'A', status: 'done'}];
"""


def test_batch_plan_settings_and_links():
    out = _node(JOBS + """
      const p = B.plan(JOBS);
      const s0 = B.settings();
      const s = B.sanitize({views: ['left', 'anterior', 'bogus', 'left'], layout: 'montage', svg: true, wall: {field: 'tawss', scale: 'fixed', lo: '0', hi: 2}, volume: {field: 'nope', scale: 'odd'}});
      const errs = B.checkSettings(B.sanitize({views: [], wall: {field: 'auto', scale: 'fixed'}, volume: {field: 'speed', scale: 'fixed', lo: 3, hi: 1}}), {wall: 1, volume: 1});
      const st = {v: 1, f: 'tawss', w: {range: [0, 2]}, d: {bands: {tawss: 6}, cmap: 'viridis'}, L: {outline: true}};
      const enc = F.encodeState(st);
      const links = ['http://h/v2/#/job/A?f=tawss&view=' + enc, '#/job/A?view=' + enc, enc, 'not a link', 'http://h/v2/#/job/A?view=@@@'].map(t => B.parseLink(t));
      const lp = B.linkPlan(st), set1 = B.applyLinkToSettings(B.sanitize({}), lp);
      const set2 = B.applyLinkToSettings(B.sanitize({}), B.linkPlan({v: 1, f: 'speed', w: 'adaptive-linear'}));
      const set3 = B.applyLinkToSettings(B.sanitize({}), B.linkPlan({v: 1, f: 'tawss', w: 'low'}));
      out({p: {sel: p.selected.map(j => j.id), skipped: p.skipped, fam: p.families, cycle: p.cycle}, s0, s, errs,
        links: links.map(r => [Boolean(r.state), r.error]), lp, set1: set1.wall, set2: set2.volume, set3: set3.wall, other: B.linkPlan({v: 1, f: 'mystery'}).family});
    """)
    p = out["p"]
    assert p["sel"] == ["A", "V", "W"] and p["fam"] == {"wall": 2, "volume": 1} and p["cycle"] is True
    assert p["skipped"] == [{"id": "Q", "name": "CASE_Q", "message": "未完成（待确认出口）"}]
    assert out["s0"] == {"views": ["anterior"], "layout": "files", "svg": False, "wall": {"field": "auto", "scale": "adaptive", "lo": None, "hi": None},
                         "volume": {"field": "auto", "scale": "adaptive", "lo": None, "hi": None}}
    s = out["s"]
    assert s["views"] == ["anterior", "left"] and s["layout"] == "montage" and s["svg"] is True
    assert s["wall"] == {"field": "tawss", "scale": "fixed", "lo": 0, "hi": 2} and s["volume"]["field"] == "auto" and s["volume"]["scale"] == "adaptive"
    assert out["errs"] == ["至少选一个视角", "壁面：固定范围要先选一个字段", "体场：固定范围要两个数，下限小于上限"]
    assert [x[0] for x in out["links"]] == [True, True, True, False, False]
    assert out["links"][3][1].startswith("这不是复现链接") and out["links"][4][1] == "这不是复现链接（地址里没有 view=）。"
    assert out["lp"]["family"] == "wall" and out["lp"]["d"]["bands"] == {"tawss": 6}
    assert out["set1"] == {"field": "tawss", "scale": "fixed", "lo": 0, "hi": 2}
    assert out["set2"]["field"] == "speed" and out["set2"]["scale"] == "adaptive-linear"
    assert out["set3"]["scale"] == "link" and out["other"] is None


def test_batch_field_window_choice_and_words():
    out = _node("""
      const wall = {result: {family: 'wall'}, fields: [{id: 'wss', units: 'Pa', display: {p99: 16.6, log_scale: true}}, {id: 'tawss', units: 'Pa', display: {p99: 4.36, log_scale: true}, windows: [{id: 'low', label: '低剪切窗', range: [0, 1]}]}]};
      const x5d = {result: {family: 'wall'}, fields: [{id: 'wss', units: 'Pa', display: {p99: 16.6}}]};
      const c = (m, set, link) => B.choose(m, Object.assign({field: 'auto', scale: 'adaptive', lo: null, hi: null}, set), link);
      const P = D.prefs(); P.units.pressure = 'mmHg'; D.savePrefs(P);
      const pf = {id: 'pressure', units: 'Pa', display: {range: [-1400, 131]}};
      out({auto: c(wall, {}), autoX5d: c(x5d, {}), fixed: c(wall, {field: 'tawss', scale: 'fixed', lo: 0, hi: 2}), missing: c(x5d, {field: 'tawss', scale: 'fixed', lo: 0, hi: 2}),
        linear: c(wall, {field: 'wss', scale: 'adaptive-linear'}), linkWin: c(wall, {field: 'tawss', scale: 'link'}, {window: 'low'}), linkMiss: c(x5d, {field: 'wss', scale: 'link'}, {window: 'low'}),
        wVol: B.windowLabel(pf, 'adaptive', 'volume'), wWall: B.windowLabel(pf, 'adaptive', 'wall'), wFixed: B.windowLabel(wall.fields[1], {range: [0, 2]}, 'wall'),
        wNamed: B.windowLabel(wall.fields[1], 'low', 'wall'), spec: [B.scaleSpec('adaptive-linear', 'turbo'), B.scaleSpec({range: [0, 2]}, 'rainbow')]});
    """)
    assert out["auto"] == {"field": "tawss", "window": "adaptive", "notes": []} and out["autoX5d"]["field"] == "wss"
    assert out["fixed"] == {"field": "tawss", "window": {"range": [0, 2]}, "notes": []}
    assert out["missing"]["field"] == "wss" and out["missing"]["window"] == "adaptive"
    assert out["missing"]["notes"] == ["没有 TAWSS，用 WSS", "固定范围只用于 TAWSS，这里用本例自适应"]
    assert out["linear"]["window"] == "adaptive-linear" and out["linkWin"]["window"] == "low"
    assert out["linkMiss"]["notes"] == ["没有链接里的色标窗，用本例自适应"]
    assert out["wVol"].startswith("本例自适应") and "mmHg" in out["wVol"] and "Pa" not in out["wVol"].replace("mmHg", "")
    assert out["wWall"] == "本例自适应 -1400 至 131 Pa"
    assert out["wFixed"] == "固定范围 0–2 Pa" and out["wNamed"] == "低剪切窗 0–1 Pa"
    assert out["spec"] == [{"window": "adaptive", "log": False, "bands": None, "cmap": "turbo"}, {"window": {"range": [0, 2]}, "log": None, "bands": None, "cmap": "rainbow"}]


def test_batch_display_provider_for_the_offscreen_viewer():
    out = _node("""
      const base = ns.viewer.displayProvider, other = {}, mine = {};
      const P = D.prefs(); P.defaults.tawss = {bands: 4, thresholds: [0.2, 2, 5]}; P.units.pressure = 'mmHg'; D.savePrefs(P);
      const field = {id: 'tawss', units: 'Pa', display: {thresholds: [0.4, 4, 7]}};
      const result = {family: 'wall', manifest: {fields: [field], analysis: {}}, field: id => id === 'tawss' ? field : null, declared: () => false, has: () => false};
      const hold = {viewer: mine, result, family: 'wall', field: 'tawss', link: null};
      const off = B.installProvider(hold), p = ns.viewer.displayProvider;
      const byDefaults = {bands: p.bands(mine, 'tawss'), thr: p.thresholds(mine, 'tawss'), flags: p.flags(mine), state: p.state(mine)};
      hold.link = {d: {bands: {tawss: 6}, thr: {tawss: [0.3, 3, 6]}, units: {pressure: 'Pa'}, layers: {stl: true, stagnation: false}}};
      const byLink = {bands: p.bands(mine, 'tawss'), thr: p.thresholds(mine, 'tawss'), stl: p.flags(mine).stl};
      const volHold = {viewer: mine, result: {family: 'volume', manifest: {}}, family: 'volume', field: 'pressure', link: null};
      const pf = {id: 'pressure', units: 'Pa'};
      hold.family = 'volume'; hold.link = null;
      const units = p.units(mine, pf);
      const others = {bands: p.bands(other, 'tawss'), units: p.units(other, pf)};
      off();
      out({byDefaults, byLink, units, others, restored: ns.viewer.displayProvider === base, wrapped: p !== base});
    """)
    d = out["byDefaults"]
    assert d["bands"] == 4 and d["thr"] == [0.2, 2, 5] and d["flags"]["stl"] is False and d["flags"]["vectors"] is False and d["state"] is None
    assert out["byLink"] == {"bands": 6, "thr": [0.3, 3, 6], "stl": False}     # the input STL never covers a batch figure
    assert out["units"]["units"] == "mmHg" and abs(out["units"]["factor"] - 1 / 133.322) < 1e-9
    assert out["others"] == {"bands": None, "units": None}                     # the stage's viewers: the original provider answers
    assert out["wrapped"] and out["restored"]


RUN_STUBS = r"""
// offscreen viewer and data stubs: the run's orchestration (files, folders, manifest, failures, stop, clean-up) without WebGL
const created = [];
ns.viewer.webglAvailable = () => true;
ns.viewer.create = (host, o) => {
  const v = {calls: [], host, resize() {}, setResult(r) { v.calls.push('setResult:' + r.manifest.job.id); return Promise.resolve(r); },
    setField(f, spec) { v.calls.push('setField:' + f + ':' + JSON.stringify(spec)); return Promise.resolve(f); }, setLighting(l) { v.calls.push('light:' + l); },
    setLayers(l) { v.calls.push('layers'); }, fit() {}, hasAnatomicalFrame: () => true,
    colorbarInfo: () => ({fieldId: 'x', label: 'X', units: 'Pa', scale: null}), dispose() { v.disposed = true; }};
  created.push(v); return v;
};
const manifests = {
  A: {job: {id: 'A', display_name: '病例 A/甲'}, result: {family: 'wall'}, fields: [{id: 'wss', units: 'Pa', display: {}}, {id: 'tawss', units: 'Pa', display: {}}], geometry: {}},
  V: {job: {id: 'V', display_name: 'CASE_V'}, result: {family: 'volume'}, fields: [{id: 'pressure', units: 'Pa', display: {}}, {id: 'speed', units: 'm/s', display: {}}], geometry: {}},
  W: {job: {id: 'W', display_name: 'CASE_W'}, result: {family: 'wall'}, fields: [{id: 'wss', units: 'Pa', display: {}}], geometry: {}}
};
ns.api = {dataFetch: () => null, request: () => Promise.reject(new Error('no'))};
ns.data = {createOnlineSource: id => ({id}), loadResult: src => src.id === 'BAD' ? Promise.reject(new Error('manifest request failed (HTTP 404)')) :
  Promise.resolve({manifest: manifests[src.id], family: manifests[src.id].result.family, runIdentity: 'r-' + src.id, declared: () => false, has: () => true, preload: () => Promise.resolve(),
    field: id => manifests[src.id].fields.find(f => f.id === id) || null})};
const seenViews = [];
F.viewFiles = (api, o) => { seenViews.push([api.cur().jobId, api.cur().field, JSON.stringify(api.cur().window), o.views.join(','), o.hideName]);
  return Promise.resolve({files: o.views.map(v => ({name: F.fileName(api, {hideName: o.hideName}, v, 'png', o.scale), bytes: new TextEncoder().encode('png ' + v)})), skipped: [], scale: o.scale}); };
F.montage = (api, o) => Promise.resolve({canvas: {}, count: o.views.length, skipped: [], scale: o.scale});
F.toBlob = () => Promise.resolve({});
F.blobBytes = () => Promise.resolve(new TextEncoder().encode('montage'));
F.colorbarSVG = () => '<svg/>';
const unzip = zip => Buffer.from(zip).toString('base64');
"""


def test_batch_run_files_folders_manifest_failures_and_stop():
    out = _node(RUN_STUBS + """
      const base = ns.viewer.displayProvider;
      const style = {scale: 2, background: 'white', lang: 'zh', colorbar: true, title: true, labels: false, crop: true, hideName: false};
      const settings = B.sanitize({views: ['anterior', 'left'], wall: {field: 'tawss', scale: 'fixed', lo: 0, hi: 2}});
      const events = [];
      const r1 = await B.run([{id: 'A'}, {id: 'V'}, {id: 'W'}, {id: 'BAD'}], {settings, style, link: null},
        {onStart: (j, i, n) => events.push('start ' + j.id + ' ' + i + '/' + n), onDone: (j, i, n, r) => events.push('done ' + j.id + ' ' + r.ok)});
      const after1 = {disposed: created.map(v => Boolean(v.disposed)), hosts: walk(body, e => String(e.className).includes('wsb-offscreen')).length, provider: ns.viewer.displayProvider === base,
        calls: created[0].calls};
      // montage + colour bar SVG + hidden names, and a stop after the first result
      let n = 0;
      const r2 = await B.run([{id: 'A'}, {id: 'W'}, {id: 'V'}], {settings: B.sanitize({views: ['anterior', 'posterior'], layout: 'montage', svg: true}), style: Object.assign({}, style, {hideName: true}), link: null},
        {stopped: () => n > 0, onDone: () => { n += 1; }});
      out({z1: unzip(r1.zip), count1: r1.count, failed1: r1.failed, events, after1, seenViews, z2: unzip(r2.zip), count2: r2.count, failed2: r2.failed});
    """)
    z1 = zipfile.ZipFile(io.BytesIO(base64.b64decode(out["z1"])))
    names = z1.namelist()
    assert names == ["病例_A_甲_A/病例_A_甲_front_tawss_2x.png", "病例_A_甲_A/病例_A_甲_left_tawss_2x.png",
                     "CASE_V_V/CASE_V_front_speed_2x.png", "CASE_V_V/CASE_V_left_speed_2x.png",
                     "CASE_W_W/CASE_W_front_wss_2x.png", "CASE_W_W/CASE_W_left_wss_2x.png", "manifest.json"]
    man = json.loads(z1.read("manifest.json"))
    assert man["schema_version"] == "wss-deploy.batch_export/v2" and man["frame_css_px"] == [900, 900]
    res = {r["id"]: r for r in man["results"]}
    assert res["A"]["field"] == "tawss" and res["A"]["window"] == {"range": [0, 2]} and res["A"]["case"] == "病例 A/甲"
    assert res["W"]["field"] == "wss" and res["W"]["window"] == "adaptive" and res["W"]["notes"] == ["没有 TAWSS，用 WSS", "固定范围只用于 TAWSS，这里用本例自适应"]
    assert res["V"]["field"] == "speed" and res["BAD"] == {"id": "BAD", "ok": False, "message": "manifest request failed (HTTP 404)"}
    assert out["count1"] == 3 and len(out["failed1"]) == 1
    assert out["events"][:2] == ["start A 0/4", "done A true"] and out["events"][-1] == "done BAD false"
    a = out["after1"]
    assert a["disposed"] == [True] and a["hosts"] == 0 and a["provider"] is True      # the offscreen viewer is gone, the provider is back
    assert a["calls"][:2] == ["setResult:A", "light:soft"] and a["calls"][2].startswith('setField:tawss:{"window":{"range":[0,2]}')
    z2 = zipfile.ZipFile(io.BytesIO(base64.b64decode(out["z2"])))
    assert z2.namelist() == ["case_A/case_montage_tawss_2x.png", "case_A/case_colorbar_tawss_1x.svg", "manifest.json"]
    man2 = json.loads(z2.read("manifest.json"))
    assert [r.get("message") for r in man2["results"]] == [None, "已停止", "已停止"] and "case" not in man2["results"][0]
    assert out["count2"] == 1 and len(out["failed2"]) == 2


def test_batch_dialog_counts_link_and_validation():
    out = _node(JOBS + RUN_STUBS + """
      const shellApi = fakeApi({jobs: JOBS});
      const dlg = B.open(['A', 'V', 'W', 'Q'], shellApi);
      await wait(20);
      const el = byId('ws-dialog');
      const head = textOf(byClass(el, 'wsb-head')[0]), log = byClass(el, 'wsb-li').map(textOf), rows = byClass(el, 'exp-k').map(textOf);
      const selects = () => walk(el, e => e.tagName === 'SELECT').map(s => s.value);
      const before = selects();
      const link = byClass(el, 'wsb-link')[0];
      link.value = 'http://h/v2/#/job/A?view=' + F.encodeState({v: 1, f: 'tawss', w: {range: [0, 2]}, d: {bands: {tawss: 6}, cmap: 'viridis'}});
      fire(link, 'change');
      const afterLink = {selects: selects(), nums: byClass(el, 'wsb-num').map(i => i.value), note: textOf(byClass(el, 'wsb-link-note')[0])};
      link.value = 'nonsense'; fire(link, 'change');
      const badNote = textOf(byClass(el, 'wsb-link-note')[0]);
      // no view chosen → the run does not start
      walk(el, e => e.tagName === 'BUTTON' && textOf(e) === '前')[0].click();
      byText(el, '开始出图').click(); await wait(10);
      const status = textOf(byClass(el, 'wsb-status')[0]);
      const cur = byText(el, '用当前显示');                          // the shell has a result open: its display can be taken
      cur.click();
      const fromCur = {note: textOf(byClass(el, 'wsb-link-note')[0]), selects: selects()};
      ns.ui.dialog.close('cancel');
      out({head, log, rows, before, afterLink, badNote, status, fromCur, created: created.length, stored: B.settings().views});
    """)
    assert out["head"].startswith("3 个结果壁面 2体场 1跳过 1")
    assert out["log"] == ["CASE_Q：跳过 · 未完成（待确认出口）"]
    assert out["rows"][:3] == ["壁面", "体场", "链接"] and "视角" in out["rows"] and "输出" in out["rows"] and "倍率" in out["rows"]
    assert out["before"] == ["auto", "adaptive", "auto", "adaptive"]
    assert out["afterLink"]["selects"] == ["tawss", "fixed", "auto", "adaptive"] and out["afterLink"]["nums"] == ["0", "2"]
    assert out["afterLink"]["note"] == "按链接：壁面 · TAWSS · 固定 0–2 Pa · 6 段 · viridis"
    assert out["badNote"] == "这不是复现链接（地址里没有 view=）。"
    assert out["status"] == "至少选一个视角。" and out["created"] == 0          # nothing rendered
    assert out["fromCur"]["note"].startswith("按当前显示：壁面 · TAWSS · 本例自适应") and out["fromCur"]["selects"][:2] == ["tawss", "adaptive"]
    assert out["stored"] == []
