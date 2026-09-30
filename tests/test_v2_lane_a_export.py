"""Node tests for lane A (second phase S5a): figure exports of the v2 workspace (ws_figure.js, ws_export.js, the section
in ws_bookmarks.js, the lane A block of core_viewer.js).

WebGL and 2-D canvases are not available in Node: the pixels (renders, crops, montages, the print page) are checked in
the sandbox browser (lane report).  Here: options, the English words, the colour bar SVG (the stage's own drawing, not
re-computed), the classic file names, the reproducible link (encoding, the shell's parseHash left untouched, the order
in which a view is applied), the one-pager plan and size limits, the classic standard views, bookmarks carrying the
section, the extension hooks and the export dialog's pages.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from wss_deploy.paths import STATIC_DIR, PROJECT_ROOT

V2 = STATIC_DIR / "v2"
FILES = [STATIC_DIR / "three.min.js", STATIC_DIR / "report_common.js", V2 / "core_util.js", V2 / "core_colormap.js", V2 / "core_colorbar.js",
         V2 / "core_viewer.js", V2 / "ws_icons.js", V2 / "ws_ui.js", V2 / "ws_store.js", V2 / "ws_bookmarks.js", V2 / "ws_export.js",
         V2 / "ws_figure.js", V2 / "ws_shell.js"]

# A small stub DOM: enough for ws_ui.h / fill / button / section / dialog and the icons (no innerHTML, no querySelectorAll).
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
const winEvents = {};
global.addEventListener = (n, f) => { (winEvents[n] = winEvents[n] || []).push(f); };
global.removeEventListener = () => {};
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


def _node(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("Node is required for the lane A tests")
    prelude = DOM + "".join("require(" + json.dumps(str(f)) + ");\n" for f in FILES)
    prelude += "const ns=globalThis.WSSV2, F=ns.figure, X=ns.exporter, B=ns.bookmarks, C=globalThis.WssReportCommon; const out=x=>console.log(JSON.stringify(x));\n"
    program = prelude + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_scripts_parse_and_are_bundled():
    if not shutil.which("node"):
        pytest.skip("Node is required")
    for name in ("ws_figure.js", "ws_export.js", "ws_bookmarks.js", "core_viewer.js", "ws_shell.js"):
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    s = bundle["scripts"]
    assert s.index("ws_bookmarks.js") < s.index("ws_export.js") < s.index("ws_figure.js") < s.index("ws_shell.js")
    assert "v2_export.css" in bundle["styles"]
    for name in ("ws_figure.js", "ws_export.js", "ws_bookmarks.js"):
        assert name not in bundle["offline_exclude"]          # the figure tools work in the offline report too
    css = (V2 / "v2_export.css").read_text(encoding="utf-8")
    assert "@media print" in css and "body.ws-printing > *:not(.ws-print)" in css


def test_lane_a_viewer_block_and_exports():
    src = (V2 / "core_viewer.js").read_text(encoding="utf-8")
    block = src[src.index("// ==== lane A ===="):src.index("// ==== end lane A ====")]
    assert "function renderPose" in block and "function exportPose" in block
    exports = src[src.index("// lane A exports"):src.index("// lane E exports")]
    assert "exportPose: exportPose, renderPose: renderPose" in exports
    shell = (V2 / "ws_shell.js").read_text(encoding="utf-8")
    lane = shell[shell.index("      // lane A\n"):shell.index("      // lane B\n")]
    assert "toggleCursor" in lane and "toggleSlice" in lane and "exitSlice" in lane and "cursorMoved" in lane


def test_options_are_sanitised_and_kept():
    out = _node("""
      const d = F.options();
      F.setOptions({scale: 3, background: 'pink', lang: 'en', title: false, views: ['anterior', 'bogus', 'slice', 'slice'], columns: 9});
      const a = F.options();
      out({d, a, raw: JSON.parse(localStorage.getItem('wssv2:export'))});
    """)
    assert out["d"] == {"scale": 2, "background": "white", "lang": "zh", "colorbar": True, "title": True, "labels": True, "crop": True,
                        "views": ["anterior", "posterior", "left", "right", "superior", "inferior"], "columns": 3}
    assert out["a"]["scale"] == 2 and out["a"]["background"] == "white" and out["a"]["lang"] == "en" and out["a"]["title"] is False
    assert out["a"]["views"] == ["anterior", "slice"] and out["a"]["columns"] == 3
    assert out["raw"]["lang"] == "en"


def test_english_words():
    out = _node("""
      out({w: F.tr('本例自适应（对数） 0–4.36 Pa', 'en'), b: F.tr('左髂总', 'en'), f: F.tr('F1 高 WSS 区 48.6 Pa', 'en'), p: F.tr('最低压力', 'en'),
        n: F.tr('灰柱：采样点分布：76426 点', 'en'), s: F.tr('截面 · 速度大小', 'en'), win: F.tr('低剪切窗 0–1 Pa（暂定）', 'en'), zh: F.tr('本例自适应', 'zh'),
        ascii: F.tr('TAWSS', 'en'), minus: F.tr('−3.2 至 4 Pa', 'en'), res: F.tr('周期指标 TAWSS · OSI', 'en')});
    """)
    assert out["w"] == "Case-adaptive (log) 0–4.36 Pa"
    assert out["b"] == "Left CIA" and out["f"] == "F1 high-WSS region 48.6 Pa" and out["p"] == "Min pressure"
    assert out["n"] == "Grey bars: sample-point distribution: 76426 points"
    assert out["s"] == "Section · Speed" and out["win"] == "Low-shear window 0–1 Pa (provisional)"
    assert out["zh"] == "本例自适应" and out["ascii"] == "TAWSS" and out["minus"] == "−3.2 to 4 Pa"
    assert out["res"] == "Cycle metrics TAWSS · OSI"


def test_colorbar_svg_is_the_stage_drawing():
    out = _node("""
      const sc = ns.colormap.scale({range: [0.05, 4.36], log: true, cmap: 'rainbow'});
      const info = {fieldId: 'tawss', label: 'TAWSS', fullLabel: '周期平均壁面切应力', units: 'Pa', scale: sc, thresholds: [0.4, 4, 7], histogram: null,
        windowLabel: {id: 'adaptive', label: '本例自适应', text: '本例自适应 0.05–4.36 Pa'}, missingFraction: 0};
      const white = F.themeOf('white');
      const zh = F.colorbarSVG(info, {lang: 'zh', theme: white, width: 190, graphicHeight: 300});
      const same = ns.colorbar.toSVG(info, {width: 190, graphicHeight: 300, background: '#ffffff'});
      const en = F.colorbarSVG(info, {lang: 'en', theme: white});
      const dark = F.colorbarSVG(info, {lang: 'zh', theme: Object.assign({}, white, {dark: true, fill: '#1a1e25'})});
      const clear = F.colorbarSVG(info, {lang: 'zh', theme: F.themeOf('transparent')});
      const texts = s => (s.match(/<text[^>]*>[^<]*<\\/text>/g) || []).map(t => t.replace(/<[^>]+>/g, ''));
      const ticks = s => texts(s).filter(t => /^[−\\d.]+$/.test(t));
      out({equal: zh === same, zhTicks: ticks(zh), enTicks: ticks(en), enText: texts(en), cjk: /[\\u4e00-\\u9fff]/.test(texts(en).join(' ')),
        width: +/width="(\\d+)"/.exec(en)[1], dark: dark.includes('#eef2f6') && !dark.includes('fill="#1b2430"'), clear: !/<rect width="\\d+" height="\\d+" fill=/.test(clear)});
    """)
    assert out["equal"] is True                               # zh + white: exactly the kernel's toSVG — numbers untouched
    assert out["zhTicks"] == out["enTicks"] and "0.4" in out["zhTicks"] and "4" in out["zhTicks"]
    assert not out["cjk"] and "Case-adaptive · log" in out["enText"]
    assert out["width"] > 190                                  # the English note lines get room
    assert out["dark"] and out["clear"]


def test_file_names_follow_the_classic_export_filename():
    out = _node("""
      const api = {cur: () => ({field: 'tawss', manifest: {job: {display_name: 'LV_GUO_YOU'}}})};
      out({fig: F.fileName(api, {}, 'current', 'png', 2), mon: F.fileName(api, {}, 'montage', 'png', 4), bar: F.fileName(api, {}, 'colorbar', 'svg', 1),
        front: F.fileName(api, {}, 'anterior', 'png', 2), hidden: F.fileName(api, {hideName: true}, 'current', 'png', 1),
        slice: F.fileName(api, {info: {fieldId: 'slice'}}, 'current', 'png', 2), classic: C.exportFilename({case_id: 'LV_GUO_YOU', view: 'front', field: 'tawss', scale: 2})});
    """)
    assert out["fig"] == "LV_GUO_YOU_custom_tawss_2x.png" and out["mon"] == "LV_GUO_YOU_montage_tawss_4x.png"
    assert out["bar"] == "LV_GUO_YOU_colorbar_tawss_1x.svg" and out["front"] == out["classic"] == "LV_GUO_YOU_front_tawss_2x.png"
    assert out["hidden"] == "case_custom_tawss_1x.png" and out["slice"] == "LV_GUO_YOU_custom_slice_2x.png"


def test_figure_layout():
    out = _node("""
      out({full: F.figureLayout({k: 2, imgW: 1000, imgH: 800, barW: 380, barH: 700, title: true, footer: true}),
        bare: F.figureLayout({k: 1, imgW: 500, imgH: 400, barW: 0, barH: 0, title: false, footer: false}),
        tallBar: F.figureLayout({k: 1, imgW: 300, imgH: 200, barW: 190, barH: 380, title: false, footer: false})});
    """)
    f = out["full"]
    assert f["W"] == 1000 + 380 + 36 and f["H"] == 128 + 800 + 60 and f["img"] == {"x": 0, "y": 128} and f["bar"] == {"x": 1000, "y": 164}
    assert out["bare"]["W"] == 500 and out["bare"]["H"] == 400 and out["bare"]["bar"] is None
    assert out["tallBar"]["H"] == 380 + 36                     # a bar taller than the image sets the height


def test_link_state_round_trip_and_the_shell_route_is_untouched():
    out = _node("""
      const st = {v: 1, run: 'abc', f: 'osi', w: {range: [0.1, 0.5]}, cam: {p: [1.25, -2, 3], t: [0, 0, 0], u: [0, 0, 1], fov: 30}, cu: {s: 2, m: 20.5},
        sel: null, br: [0, 1], L: {outline: true}, sl: {basis: 'pick', pick: {origin: [1, 2, 3], normal: [0, 0, 1], picks: 1}, thickness: 3, note: '截面 ①'}};
      const enc = F.encodeState(st), dec = F.decodeState(enc);
      const hash = '#/job/J_1?f=osi&view=' + enc;
      const r = ns.shell.parseHash(hash);
      let bad = null; try { F.decodeState(F.encodeState({v: 2})); } catch (e) { bad = e.message; }
      out({same: JSON.stringify(dec) === JSON.stringify(st), urlSafe: /^[A-Za-z0-9_-]+$/.test(enc), r, param: F.linkParam(hash) === enc, bad,
        w1: F.withoutView('#/job/J?f=osi&view=abc'), w2: F.withoutView('#/job/J?view=abc&f=osi'), w3: F.withoutView('#/job/J?view=abc'), w4: F.withoutView('#/job/J?f=a&view=x&q=b')});
    """)
    assert out["same"] and out["urlSafe"] and out["param"]
    assert out["r"] == {"jobId": "J_1", "v": None, "f": "osi", "cmp": None, "q": None, "bm": None}   # view= is ignored by parseHash
    assert out["bad"] == "invalid view state"
    assert out["w1"] == "#/job/J?f=osi" and out["w2"] == "#/job/J?f=osi" and out["w3"] == "#/job/J" and out["w4"] == "#/job/J?f=a&q=b"


# A fake shell API: records what the figure module asks the shell and the viewer to do.
FAKE_API = r"""
function fakeApi(opts) {
  opts = opts || {};
  const calls = [];
  const viewer = {getState: () => ({camera: {position: [1.234567, 2, 3], target: [0, 0, 0], up: [0, 0, 1], fov: 30}, cursor: opts.cursor || null, selection: 7, branches: null}),
    applyState: (s) => { calls.push('applyState:' + Object.keys(s).sort().join(',')); return Promise.resolve(s); }, hasAnatomicalFrame: () => true,
    renderPose: () => Promise.resolve({})};
  const slice = opts.slice ? {state: () => ({basis: 'centerline', segment: 0, fraction: 0.123456789, thickness: 2}), set: s => calls.push('slice.set:' + s.fraction), comp: () => ({})} : null;
  const cur = {jobId: 'J', runIdentity: 'run-1234567890abcdef-long', field: 'wss', window: 'adaptive', manifest: {fields: [{id: 'wss'}, {id: 'osi'}]},
    result: {manifest: {}}, slice: slice, cursor: opts.cursor ? {set: (a, b) => calls.push('cursor.set:' + a + ':' + b)} : null, job: {}};
  const S = {layers: {outline: false, centerline: false, points: false, trust: false, streamlines: false, wall: true, interior: false}};
  const api = {calls, cur: () => cur, viewer: () => viewer, state: () => S, offline: () => false, hideName: () => false,
    parseHash: ns.shell.parseHash, buildHash: ns.shell.buildHash, fieldById: (m, id) => (m.fields || []).find(f => f.id === id) || null,
    applyField: (f, w) => { calls.push('applyField:' + f + ':' + JSON.stringify(w)); cur.field = f; cur.window = w; return true; },
    setLayers: () => calls.push('setLayers'), saveViewSoon: () => calls.push('save'), cursorMoved: () => calls.push('cursorMoved'),
    toggleCursor: () => { calls.push('toggleCursor'); if (cur.cursor) cur.cursor = null; else setTimeout(() => { cur.cursor = {set: (a, b) => calls.push('cursor.set:' + a + ':' + b)}; }, 30); },
    toggleSlice: () => { calls.push('toggleSlice:' + JSON.stringify(cur.sliceState && cur.sliceState.fraction)); setTimeout(() => { cur.slice = {state: () => cur.sliceState, set: s => calls.push('slice.set:' + s.fraction), comp: () => ({})}; }, 30); },
    exitSlice: () => { calls.push('exitSlice'); cur.slice = null; }, openExport: () => calls.push('openExport'),
    ui: () => ns.ui, h: ns.ui.h};
  return api;
}
"""


def test_view_state_and_link():
    out = _node(FAKE_API + """
      const api = fakeApi({slice: true, cursor: {segmentId: 2, s_mm: 20.5}});
      api.cur().cursor = {state: () => ({})};
      location.hash = '#/job/J?f=wss&q=low&bm=b1';
      const st = F.viewState(api), url = F.linkURL(api);
      out({st, url, back: F.decodeState(F.linkParam(url))});
    """)
    st = out["st"]
    assert st["v"] == 1 and st["run"] == "run-1234567890ab" and st["f"] == "wss" and st["w"] == "adaptive"
    assert st["cam"] == {"p": [1.2346, 2, 3], "t": [0, 0, 0], "u": [0, 0, 1], "fov": 30}
    assert st["cu"] == {"s": 2, "m": 20.5} and st["sel"] == 7 and st["br"] is None
    assert st["sl"]["fraction"] == 0.1235 and st["L"]["wall"] is True
    assert out["url"].startswith("http://127.0.0.1:8765/v2/#/job/J?f=wss&view=")   # the question and the bookmark are dropped
    assert out["back"] == st


def test_apply_view_order_section_then_cursor_then_camera():
    out = _node(FAKE_API + """
      ns.slice = {supported: () => true};
      const api = fakeApi({});
      const st = {v: 1, run: 'other', f: 'osi', w: 'high', L: {outline: true}, cam: {p: [5, 5, 5], t: [0, 0, 0], u: [0, 0, 1], fov: 30}, cu: {s: 3, m: 12},
        sel: 4, br: [0, 2], sl: {basis: 'centerline', segment: 0, fraction: 0.62, thickness: 3}};
      const notes = await F.applyView(api, st);
      const closing = fakeApi({slice: true, cursor: true});
      await F.applyView(closing, {v: 1, sl: null, cu: null});
      out({calls: api.calls, notes, layers: api.state().layers.outline, sel: api.cur().selection, closing: closing.calls});
    """)
    c = out["calls"]
    order = [c.index("applyField:osi:\"high\""), c.index("setLayers"), c.index("toggleSlice:0.62"), c.index("slice.set:0.62"),
             c.index("toggleCursor"), c.index("cursor.set:3:12"), c.index("applyState:branches,camera,selection")]
    assert order == sorted(order), c
    assert out["notes"] == ["链接来自这份结果的另一次计算，已按可用项恢复"]
    assert out["layers"] is True and out["sel"] == 4
    assert out["closing"][:2] == ["exitSlice", "toggleCursor"]


def test_bookmarks_carry_the_section():
    out = _node(FAKE_API + """
      const api = fakeApi({slice: true});
      ns.slice = {supported: () => true};
      F._setApi(api);
      const ctx = {jobId: 'J', runIdentity: 'run-1234567890abcdef-long', arraysVersion: 'av', fields: ['wss'], field: 'wss', state: {camera: {position: [1, 2, 3], target: [0, 0, 0], up: [0, 0, 1]}}};
      const withSlice = B.make(ctx, {name: 'a'});
      const other = B.make(Object.assign({}, ctx, {runIdentity: 'another'}), {name: 'b'});
      api.cur().slice = null;
      const closed = B.make(ctx, {name: 'c'});
      ns.slice = {supported: () => false};
      const wall = B.make(ctx, {name: 'd'});
      // restore from the panel: the shell's restore first, then the section
      ns.slice = {supported: () => true};
      const seen = [];
      const realAfter = F.afterBookmark;
      F.afterBookmark = bm => { seen.push('after:' + (bm.slice ? bm.slice.fraction : bm.slice)); return Promise.resolve(); };
      B.add(ctx, withSlice); B.add(ctx, closed);
      const el = new Element('div');
      B.render(el, Object.assign({}, ctx, {onRestore: bm => seen.push('restore:' + bm.name), fieldLabel: x => x}));
      walk(el, e => e.tagName === 'BUTTON' && textOf(e) === '打开').forEach(b => fire(b, 'click'));
      F.afterBookmark = realAfter;
      // afterBookmark: opens the section, then puts the bookmark's camera back
      const api2 = fakeApi({}); F._setApi(api2);
      await F.afterBookmark(withSlice);
      out({withSlice: withSlice.slice, other: 'slice' in other, closed: closed.slice, wall: 'slice' in wall, seen, api2: api2.calls});
    """)
    assert out["withSlice"] == {"basis": "centerline", "segment": 0, "fraction": 0.1235, "thickness": 2}
    assert out["other"] is False and out["closed"] is None and out["wall"] is False
    assert out["seen"] == ["restore:a", "after:0.1235", "restore:c", "after:null"]
    assert out["api2"][0] == "toggleSlice:0.1235" and out["api2"][-2:] == ["applyState:camera", "save"]


def test_onepage_plan_and_limits():
    out = _node(FAKE_API + """
      const api = fakeApi({slice: true});
      const withFrame = F.onepagePlan(api);
      api.viewer().hasAnatomicalFrame = () => false; api.cur().slice = null;
      const bare = F.onepagePlan(api);
      const big = 'A'.repeat(Math.ceil(6.1 * 1024 * 1024 / 3) * 4);
      out({withFrame, bare, ok: F.checkSizes([{png_base64: 'AAAA'}]), big: F.checkSizes([{png_base64: big}])});
    """)
    assert out["withFrame"] == [["front", "anterior"], ["left", "left"], ["top", "superior"], ["current", "current"], ["slice", "slice"]]
    assert out["bare"] == [["current", "current"]]
    assert out["ok"] is None and "6 MB" in out["big"]


def test_standard_views_match_the_classic_report():
    src = (PROJECT_ROOT / "wss_deploy" / "report.py").read_text(encoding="utf-8")
    classic = re.search(r"function standardViews\(R,center,distance\)\{.*?\n\}", src, re.S).group(0)
    out = _node("const standardViews = (function(){ " + classic + "; return standardViews; })();\n" + """
      const a = 0.7, b = -0.4, ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
      const R = [[ca, -sa * cb, sa * sb], [sa, ca * cb, -ca * sb], [0, sb, cb]];
      const frame = {rotation: R, origin_mm: [0, 0, 0]};
      const cls = standardViews(R, [0, 0, 0], 1), rows = {};
      Object.keys(F.CLASSIC).forEach(v2 => {
        const name = F.CLASSIC[v2]; if (!cls[name]) return;
        const d = ns.viewer.viewDirection(v2, frame), c = cls[name];
        const cdir = c.target.map((t, k) => t - c.position[k]);
        rows[v2 + '→' + name] = Math.max(...d.dir.map((x, k) => Math.abs(x - cdir[k])), ...d.up.map((x, k) => Math.abs(x - c.up[k])));
      });
      out(rows);
    """)
    assert set(out) == {"anterior→front", "posterior→back", "left→left", "right→right", "superior→top", "inferior→bottom"}
    assert max(out.values()) < 1e-12


def test_extension_hooks_and_tools_section():
    out = _node(FAKE_API + """
      const ext = (ns.ext || []).find(x => x && x.id === 'figure');
      const api = fakeApi({});
      const secs = ext.tools(api);
      const labels = walk(secs[0], e => e.tagName === 'BUTTON').map(textOf);
      fire(byText(secs[0], '导图…'), 'click');
      const icons = ['fig-link', 'fig-montage', 'fig-colorbar'].map(n => ns.icons.has(n));
      out({hooks: Object.keys(ext).filter(k => typeof ext[k] === 'function').sort(), title: textOf(byClass(secs[0], 'sec-title')[0]), labels,
        pending: X.pendingTab, calls: api.calls, icons});
    """)
    assert out["hooks"] == ["onClose", "onResult", "tools"]
    assert out["title"] == "出图"
    assert out["labels"] == ["导图…", "六视角", "色标 SVG", "复现链接", "一页纸配图…", "打印"]
    assert out["pending"] == "figure" and out["calls"] == ["openExport"]
    assert out["icons"] == [True, True, True]


def test_export_dialog_pages():
    out = _node(FAKE_API + """
      const api = fakeApi({}); F._setApi(api);
      ns.api = {urls: {table: () => '/t', file: (j, n) => '/f/' + n, bundle: () => '/b', onepage: () => '/o'}, offline: () => Promise.resolve({})};
      const ctx = {job: {id: 'J'}, manifest: {job: {id: 'J', display_name: 'CASE'}, result: {family: 'wall'}}, displayName: 'CASE', bookmarks: []};
      X.pendingTab = 'data';
      X.open(ctx);
      const dlg = byId('ws-dialog');
      const tabs = walk(dlg, e => e.dataset && e.dataset.tab).map(textOf);
      const on = walk(dlg, e => e.dataset && e.dataset.tab && e.classList.contains('on')).map(textOf);
      const dataText = textOf(byClass(dlg, 'exp-pane')[0]);
      const foot = walk(byClass(dlg, 'dlg-foot')[0], e => e.tagName === 'BUTTON').map(textOf);
      ns.ui.dialog.close('done');
      X.open(Object.assign({}, ctx, {offline: true}));
      const offTabs = walk(byId('ws-dialog'), e => e.dataset && e.dataset.tab).map(textOf);
      ns.ui.dialog.close('done');
      // without the figure module (a trimmed bundle): the first-phase 汇报图 only
      const keep = ns.figure; ns.figure = undefined;
      X.open(Object.assign({}, ctx, {offline: true}));
      const bare = {tabs: walk(byId('ws-dialog'), e => e.dataset && e.dataset.tab).map(textOf), text: textOf(byId('ws-dialog'))};
      ns.figure = keep; ns.ui.dialog.close('done');
      out({tabs, on, dataText, foot, offTabs, bare});
    """)
    assert out["tabs"] == ["图片", "拼图", "一页纸", "数据"] and out["on"] == ["数据"]
    assert "统计表 CSV" in out["dataText"] and "下载离线报告" in out["dataText"]
    assert out["foot"] == ["复现链接", "关闭"]
    assert out["offTabs"] == ["图片", "拼图"]
    assert out["bare"]["tabs"] == ["图片"] and "下载 PNG" in out["bare"]["text"]
