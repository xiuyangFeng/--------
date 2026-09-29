"""Node tests for the v2 workspace shell and panels (WORKSPACE_V2_CONTRACT.md §6.3–§6.5, §8.3).

The ``static/v2/ws_*.js`` modules run against a stub DOM, a canned service (``fetch``) and stub viewer-kernel
modules (``ns.util / ns.data / ns.viewer / ns.colorbar / ns.orientation / ns.cursor``), so the workspace is tested
without WebGL and independently of the kernel (which has its own tests).  Every scenario runs in its own Node
process: boot → act → print one JSON line with what the page shows and which requests went out.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from wss_deploy.paths import STATIC_DIR

V2 = STATIC_DIR / "v2"
WS_FILES = ["ws_icons.js", "ws_ui.js", "ws_store.js", "ws_api.js", "ws_rail.js", "ws_overview.js", "ws_lens.js",
            "ws_bookmarks.js", "ws_questions.js", "ws_compare.js", "ws_upload.js", "ws_input.js", "ws_export.js",
            "ws_shell.js", "ws_main.js"]
OFFLINE_EXCLUDE = {"ws_api.js", "ws_rail.js", "ws_upload.js", "ws_input.js", "ws_compare.js"}


def _need_node():
    if not shutil.which("node"):
        pytest.skip("Node is required for the workspace tests")


# ---------------------------------------------------------------------------------------------------------------
# Harness: stub DOM + canned service + stub kernel.  Scenario code is appended and must print one JSON line.
# ---------------------------------------------------------------------------------------------------------------
_HARNESS = r"""
'use strict';
const WS = __WS__;
const OFFLINE = __OFFLINE__;
const errors = [], calls = [], vcalls = [], opened = [];
console.error = (...a) => errors.push(a.map(String).join(' '));
process.on('unhandledRejection', e => errors.push('unhandled: ' + (e && e.stack || e)));
process.on('uncaughtException', e => errors.push('uncaught: ' + (e && e.stack || e)));

// ---------------------------------------------------------------- stub DOM
class TextNode { constructor(t) { this.nodeValue = String(t); this.parentNode = null; } get textContent() { return this.nodeValue; } }
class Element {
  constructor(tag) {
    this.tagName = String(tag || 'div').toUpperCase(); this.children = []; this.parentNode = null; this.events = {}; this.attrs = {};
    this.style = {}; this.dataset = {}; this.hidden = false; this.disabled = false; this.checked = false; this.value = ''; this.open = false;
    this._text = ''; this.className = ''; this.id = ''; this.files = []; this.scrollTop = 0; this.title = ''; this.type = '';
    const set = new Set(), self = this;
    this.classList = {add: (...c) => { c.forEach(x => set.add(x)); self.className = [...set].join(' '); },
      remove: (...c) => { c.forEach(x => set.delete(x)); self.className = [...set].join(' '); },
      toggle: (c, f) => { if (f === undefined) f = !set.has(c); f ? set.add(c) : set.delete(c); self.className = [...set].join(' '); return f; },
      contains: c => set.has(c) || String(self.className).split(/\s+/).includes(c)};
  }
  set textContent(v) { this._text = String(v); this.children.forEach(c => { c.parentNode = null; }); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  appendChild(x) { if (x === null || x === undefined || typeof x !== 'object') throw new TypeError('appendChild: Argument 1 is not an object.'); if (x.parentNode) x.parentNode.removeChild(x); x.parentNode = this; this.children.push(x); return x; }
  append(...xs) { xs.forEach(x => this.appendChild(typeof x === 'object' ? x : new TextNode(x))); }
  removeChild(x) { const i = this.children.indexOf(x); if (i >= 0) this.children.splice(i, 1); x.parentNode = null; return x; }
  replaceChildren(...xs) {
    xs.forEach(x => { if (x === null || typeof x !== 'object') throw new TypeError('replaceChildren got a non-node: ' + String(x)); });
    this.children.forEach(c => { c.parentNode = null; }); this.children = []; this._text = ''; xs.forEach(x => this.appendChild(x));
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  addEventListener(n, f) { (this.events[n] = this.events[n] || []).push(f); }
  removeEventListener(n, f) { const l = this.events[n] || []; const i = l.indexOf(f); if (i >= 0) l.splice(i, 1); }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'id') this.id = String(v); }
  getAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  hasAttribute(k) { return k in this.attrs; }
  contains(x) { while (x) { if (x === this) return true; x = x.parentNode; } return false; }
  closest(sel) { let e = this; while (e && e.tagName) { if (sel.includes('dialog') && e.tagName === 'DIALOG' && e.open) return e; e = e.parentNode; } return null; }
  focus() { global.document.activeElement = this; } blur() {} click() { fire(this, 'click'); }
  showModal() { this.open = true; } close() { this.open = false; fire(this, 'close'); }
  getBoundingClientRect() { return {left: 10, right: 40, top: 10, bottom: 40, width: 30, height: 30}; }
  getContext() { return null; } scrollIntoView() {}
  get firstChild() { return this.children[0] || null; }
  get isConnected() { let e = this; while (e.parentNode) e = e.parentNode; return e === global.document.body; }
}
const body = new Element('body');
const byId = id => { const walk = e => { if (!e || !e.children) return null; if (e.id === id) return e; for (const c of e.children) { const r = walk(c); if (r) return r; } return null; }; return walk(body); };
const docEvents = {};
global.document = {readyState: 'complete', body, activeElement: body,
  getElementById: byId,
  createElement: t => new Element(t), createElementNS: (ns, t) => new Element(t), createTextNode: t => new TextNode(t),
  addEventListener: (n, f) => { (docEvents[n] = docEvents[n] || []).push(f); }, removeEventListener() {}};
const winEvents = {};
global.window = global;
global.addEventListener = (n, f) => { (winEvents[n] = winEvents[n] || []).push(f); };
global.removeEventListener = () => {};
global.location = {hash: '', href: 'http://127.0.0.1/v2/', origin: 'http://127.0.0.1', pathname: '/v2/', reload() { calls.push('RELOAD'); }};
global.history = {replaceState(s, t, url) { global.location.hash = String(url).slice(String(url).indexOf('#')); calls.push('REPLACE ' + url); }};
global.localStorage = {_m: new Map(), getItem(k) { return this._m.has(k) ? this._m.get(k) : null; }, setItem(k, v) { this._m.set(k, String(v)); }, removeItem(k) { this._m.delete(k); }};
global.matchMedia = () => ({matches: false, addEventListener() {}});
global.requestAnimationFrame = f => setTimeout(f, 0);
global.open = url => { opened.push(String(url)); return null; };
global.EventSource = class { constructor(url) { calls.push('SSE ' + url); } addEventListener() {} close() {} };
global.innerWidth = 1440;
const fire = (el, name, extra) => { if (!el) throw new Error('no element for ' + name); for (const f of [...(el.events[name] || [])]) f(Object.assign({preventDefault() {}, stopPropagation() {}, target: el, currentTarget: el}, extra || {})); };
const walk = (el, pred, out = []) => { if (!el || typeof el !== 'object') return out; if (el.tagName && pred(el)) out.push(el); for (const c of (el.children || [])) walk(c, pred, out); return out; };
const byClass = (root, cls) => walk(root, e => String(e.className || '').split(/\s+/).includes(cls));
const byText = (root, text) => walk(root, e => e.textContent === text)[0] || null;
const textOf = el => el ? el.textContent : '';
const wait = ms => new Promise(r => setTimeout(r, ms));
const hashTo = async (h, ms) => { global.location.hash = h; for (const f of winEvents.hashchange || []) f({}); await wait(ms === undefined ? 60 : ms); };
const key = async (k, target, extra) => { const ev = Object.assign({key: k, target: target || body, shiftKey: false, preventDefault() { this.prevented = true; }}, extra || {}); for (const f of docEvents.keydown || []) f(ev); await wait(20); return ev; };
const app = () => byId('ws-app');

// ---------------------------------------------------------------- canned service
const canned = {};
const delays = {};
const Resp = (status, bodyObj) => ({ok: status >= 200 && status < 300, status, json: async () => JSON.parse(JSON.stringify(bodyObj)), text: async () => JSON.stringify(bodyObj),
  blob: async () => ({size: 1}), arrayBuffer: async () => new ArrayBuffer(0), headers: {get: () => null}});
global.fetch = async (url, opts) => {
  const path = String(url).split('?')[0];
  const method = opts && opts.method || 'GET';
  calls.push(method + ' ' + path);
  if (delays[path]) await wait(delays[path]);
  const entry = canned[method + ' ' + path] || canned[path];
  if (!entry) return Resp(404, {error: {message: '接口不存在。'}});
  if (typeof entry === 'function') { const r = entry(opts); return Resp(r.status || 200, r.body); }
  return Resp(entry.status || 200, entry.body);
};

// ---------------------------------------------------------------- stub kernel (§6.2 signatures)
const ns = global.WSSV2 = {};
ns.util = {env: {offline: OFFLINE}, fmtSig: v => String(v)};
ns.env = ns.util.env;
function makeResult(m) {
  return {manifest: m, runIdentity: m.result && m.result.run_identity, family: m.result && m.result.family, dataVersion: m.data_version,
    jobId: m.job && m.job.id, preload: async () => null, declared: k => Boolean(m.arrays && m.arrays[k]), has: () => true,
    array: k => { if (k === 'mt') return new Uint8Array([0, 2, 0]); throw new Error('not loaded ' + k); },
    field: id => (m.fields || []).find(f => f.id === id) || null, fields: () => (m.fields || []).slice(), branches: () => (m.geometry && m.geometry.branches) || []};
}
ns.data = {
  createOnlineSource(jobId, o) {
    return {jobId, manifest: () => o.fetch('/api/v2/jobs/' + jobId + '/manifest', {credentials: 'same-origin'}).then(res => {
      if (!res.ok) { const e = new Error('manifest request failed (HTTP ' + res.status + ')'); e.name = 'DataError'; e.status = res.status; throw e; }
      return res.json();
    })};
  },
  createEmbeddedSource(doc) { return {jobId: null, manifest: async () => JSON.parse(doc.getElementById('wssv2-manifest').textContent)}; },
  loadResult(source, o) { return source.manifest(o).then(makeResult); }
};
let viewerSeq = 0;
ns.viewer = {
  webglAvailable: () => true,
  version: 'v2.0.0',
  link: (a, b) => { vcalls.push('link'); return () => vcalls.push('unlink'); },
  create(container, o) {
    const id = ++viewerSeq, handlers = {};
    let field = null, camera = {position: [0, -300, 0], target: [0, 0, 0], up: [0, 0, 1], fov: 30}, result = null;
    const v = {
      id,
      on(n, f) { (handlers[n] = handlers[n] || []).push(f); return () => {}; },
      emit(n, e) { (handlers[n] || []).forEach(f => f(e)); },
      setResult(r) { vcalls.push(id + ':setResult:' + (r && r.manifest.job.id)); return wait(r && r.manifest.__viewerDelay || 0).then(() => { result = r; field = null; v.emit('change', {what: 'result'}); return r; }); },
      setField(f, spec) { vcalls.push(id + ':setField:' + f + ':' + JSON.stringify(spec.window)); field = f; v.emit('change', {what: 'field', fieldId: f}); return Promise.resolve(f); },
      field: () => field,
      setLighting(l) { vcalls.push(id + ':setLighting:' + l); }, setLayers(l) { vcalls.push(id + ':setLayers'); },
      fit() { vcalls.push(id + ':fit'); }, resize() {}, render() {},
      getState() { return {field, scale: {window: 'adaptive'}, camera, lighting: 'flat', layers: {outline: true}, cursor: null, selection: null, branches: null, time_index: 0}; },
      applyState(s, o) { vcalls.push(id + ':applyState'); if (s && s.field) field = s.field; return Promise.resolve(s); },
      getCamera: () => camera, setCamera(c) { camera = c; vcalls.push(id + ':setCamera'); },
      select(i) { vcalls.push(id + ':select:' + i); }, highlight(ix) { vcalls.push(id + ':highlight:' + (ix ? ix.length : 'null')); },
      setMarkers(ms) { vcalls.push(id + ':markers:' + ms.length); }, setCursor(c) { vcalls.push(id + ':cursor:' + (c ? c.segmentId + '@' + c.s_mm : 'null')); },
      colorbarInfo: () => field ? {label: field, units: 'Pa', scale: null, windowLabel: null} : null,
      snapshot: async () => ({size: 1}), dispose() { vcalls.push(id + ':dispose'); }
    };
    vcalls.push(id + ':create:' + (o && o.kind));
    return v;
  }
};
const cbInfos = [];
ns.colorbar = {create: el => ({update: info => cbInfos.push(info && info.label), dispose() {}})};
ns.orientation = {create: () => ({update() {}, dispose() {}})};
ns.cursor = {requiredArrays: () => [], create(r) {
  let cur = null; const hs = [];
  return {available: true, on: (n, f) => hs.push(f), set(sid, s) { cur = {segmentId: sid, s_mm: s}; hs.forEach(f => f(cur)); return cur; },
    step(d) { cur = {segmentId: cur.segmentId, s_mm: Math.max(0, cur.s_mm + d)}; hs.forEach(f => f(cur)); return {state: cur, atEnd: false, children: []}; },
    state: () => cur, readout: f => ({segmentId: cur && cur.segmentId, s_mm: cur && cur.s_mm, bin: [10, 12], stats: {mean_pa: 1.23, n: 40}, definition: '2 mm 分箱，箱内预测点等权统计'})};
}};

// ---------------------------------------------------------------- fixtures
const FIELD = (id, extra) => Object.assign({id, label: id.toUpperCase(), short_label: id.toUpperCase(), units: id === 'osi' ? '1' : 'Pa', location: 'wall', kind: 'scalar', components: 1,
  temporal: id === 'wss' ? 'frame' : 'cycle_summary', time_index: id === 'wss' ? 0 : null, source: 'prediction', tier: 'model', derived_from: [],
  arrays: {display: 'f.' + id + '.d', read: 'f.' + id + '.r'}, display: {thresholds: [0.4, 4, 7], threshold_direction: 'below', p99: 4.36}, windows: [], validation: null}, extra || {});
function manifestFor(id, opts) {
  opts = opts || {};
  const m = {schema: 'wss-deploy.v2-manifest/v1', data_version: 'dv-' + id, arrays_version: 'av-' + id,
    job: {id, status: 'done', display_name: opts.name || ('CASE_' + id), case_id: 'CASE_' + id, patient_id: opts.patient || '', scan_label: '', scan_date: '', created_at: '2026-09-29T08:00:00+08:00',
      input_sha256: opts.sha || ('sha-' + id), review: {status: 'unreviewed', by: '', at: '', note: '', version: null}, companions: opts.companions || [], narrative_edited: false},
    result: {run_identity: 'run-' + id, family: opts.family || 'wall', release_id: opts.release || 'M1_3head_3seed_20260922', display_name: opts.resultName || '周期指标 TAWSS · OSI',
      analysis_version: '2026-09-30', deploy_version: '0.15.0', model_frame: {target: 'peak_systole', time_s: 0.21}},
    units: {length: 'mm'}, frame: {orientation: {left_right: 'inferred', superior_inferior: 'derived', anterior_posterior: 'inferred'}, direction_source: 'unknown_stl'},
    time: {mode: 'single_frame', axis: [{index: 0, time_s: 0.21, label: 'peak_systole'}], cycle: null},
    geometry: {display_mesh: {vertices: 'mv', faces: 'mf', segment: 'ms', trust: 'mt'}, branches: [{id: 0, key: 'root', name: '主动脉', parent: -1, length_mm: 211}, {id: 2, key: 'left_cia', name: '左髂总', parent: 0, length_mm: 38}], openings: []},
    fields: opts.fields || [FIELD('wss', {windows: [{id: 'low', label: '低值窗', range: [0, 1], provisional: true}], validation: {holdout_n: 34, r2_pa: 0.74, extra: '逐例均值 0.77'}}),
      FIELD('tawss', {windows: [{id: 'low', label: '低剪切窗', range: [0, 1], provisional: true}, {id: 'standard', label: '常规窗', range: [0, 4], provisional: true}], validation: {holdout_n: 34, r2_pa: 0.74, extra: '一致性系数 0.93'}}),
      FIELD('osi', {display: {thresholds: [0.1, 0.2, 0.3], threshold_direction: 'above', range: [0, 0.5]}}),
      FIELD('rrt', {tier: 'derived', source: 'derived', derived_from: ['tawss', 'osi'], units: '1/Pa'})],
    arrays: {mv: {dtype: 'float32', shape: [3, 3]}, mt: {dtype: 'uint8', shape: [3]}},
    mapping: {display_interpolation: {method: 'Gaussian', sigma_mm: 0.5, max_dist_mm: 1.5}, statistics_protocol: {weighting: 'points'}},
    analysis: {
      narrative: {zh: ['主动脉管腔最大直径 72.2 mm。', '周期平均 TAWSS 均值 0.66 Pa。', '以上为参考描述，非诊断结论。'], en: [], edited: null},
      findings: {items: [
        {id: 'F1', kind: 'high_wss_cluster', label: '左髂内高 WSS 区', branch: '左髂内', segment_id: 4, value: 52.6, units: 'Pa', xyz_mm: [1, 2, 3], s_from_root_mm: 263, extent_mm: 9, area_mm2: 171, n_points: 383, severity: 'note', grading: 'no_reference', point_indices: [1, 2, 3], contains_global_max: true, rank: 1, definition: 'WSS ≥ p99'},
        {id: 'F2', kind: 'high_wss_cluster', label: '右髂外高 WSS 区', branch: '右髂外', value: 36.2, units: 'Pa', xyz_mm: [1, 2, 3], severity: 'note', rank: 2},
        {id: 'F3', kind: 'low_wss_cluster', label: '主动脉低 WSS 区', branch: '主动脉', value: 0.249, units: 'Pa', xyz_mm: [4, 5, 6], area_mm2: 13749, severity: 'attention', rank: 3},
        {id: 'F4', kind: 'stagnation_cluster', label: '主动脉滞留区', branch: '主动脉', value: 122, units: 'cm²', xyz_mm: [4, 5, 6], area_mm2: 12162, severity: 'attention', rank: 4},
        {id: 'F5', kind: 'max_diameter', label: '主动脉最大直径', branch: '主动脉', value: 72.2, units: 'mm', xyz_mm: [4, 5, 6], severity: 'info', source: 'morphology', rank: 5}]},
      zones: opts.zones === false ? null : {schema_version: 'wss-deploy.zones/v1', definition: '按中心线分支与弧长划分解剖分区，区内预测点等权统计', primary_fields: ['tawss', 'osi', 'wss'],
        zones: [{id: 'neck', label: '近端瘤颈', segment_id: 0, branch_key: 'root', s_range_mm: [48, 119], n_points: 9000, area_mm2: 4000, fields: {tawss: {mean: 0.412, frac_low: 0.52}, osi: {mean: 0.121, frac_high: 0.4}, wss: {mean: 1.2}}},
          {id: 'sac', label: '瘤体', segment_id: 0, branch_key: 'root', s_range_mm: [122, 209], n_points: 30000, area_mm2: 15000, fields: {tawss: {mean: 0.25, frac_low: 0.83}, osi: {mean: 0.2}}}],
        pairs: [{id: 'cia', label: '髂总', left: 'left_cia', right: 'right_cia', fields: {tawss: {left_mean: 0.759, right_mean: 0.675, ratio_left_over_right: 1.124}}}]},
      morphology: {aorta: {reference_diameter_mm: 19.0, max: {max_diameter_mm: 72.24, s_from_root_mm: 176}, sac: {present: true, length_mm: 87, volume_ml: 225.5}, neck: {present: true, length_mm: 72, diameter_mean_mm: 20.04}, n_reoriented: 0, n_excluded: 0}, lumen_volume_ml: 347.5},
      per_branch: {}, trust: {fractions: {interpolation_uncovered: 0, rough_surface: 0.0057, geometry_out_of_range: 0}}, reference_assessment: {status: 'pass', population: {status: 'unknown'}},
      profiles: {bin_mm: 2, branches: []}
    },
    model_card: {display_name: opts.resultName || '周期指标 TAWSS · OSI', short_name: '周期指标', version_date: '2026-09-22', purpose: '一次预测峰值 WSS、TAWSS、OSI',
      training: {n_train: 136, cohorts: ['AAA', 'AG', 'ILO'], centers: null}, protocol: ['所有病例用同一条入口流量波形，不是该患者实测。'],
      validation: {holdout_n: 34, fields: {wss: {r2_pa: 0.74, extra: '逐例均值 0.77'}, tawss: {r2_pa: 0.74, extra: '一致性系数 0.93'}, rrt: {r2_pa: null, summary: '没有单独验证'}}},
      field_tiers: {rrt: 'derived'}, usage: {tawss: '高低分布可信，绝对值有误差'}, weaknesses: ['小口径髂支的高值容易被低估'], not_applicable: ['开口数不是 5 的输入'], caveat: '研究用途，不是诊断'},
    provenance: {release_hash: 'abc', timing_s: {total: 8}}, reserved: {time_series: null}
  };
  if (opts.legacy) {
    m.fields = [{id: 'wss', label: '壁面切应力', units: 'Pa', arrays: {display: 'f.wss.d', read: 'f.wss.r'}}];
    m.analysis = {narrative: null, findings: null, zones: null};
    m.model_card = null; delete m.arrays_version; m.mapping = {};
  }
  return m;
}
const jobRecord = (id, extra) => Object.assign({id, case_id: 'CASE_' + id, display_name: 'CASE_' + id, status: 'done', version: 3, review: {status: 'unreviewed'},
  model_release: {id: 'M1_3head_3seed_20260922', contract: {protocol: 'single_frame_wss_cycle_multi', fields: {wss: {}, tawss: {}, osi: {}}}}, family: 'wall',
  created_at: '2026-09-29T08:00:00+08:00', input_sha256: 'sha-' + id, run_identity: 'run-' + id, timing: {compute_s: 8}}, extra || {});
function serve(id, jobExtra, mOpts) {
  canned['/api/jobs/' + id] = {body: {job: jobRecord(id, jobExtra)}};
  canned['/api/v2/jobs/' + id + '/manifest'] = {body: manifestFor(id, mOpts)};
}
canned['/api/session'] = {body: {authenticated: true, login: 'password', csrf_token: 'csrf-1', username: 'admin', role: 'admin'}};
canned['/api/v2/model-cards'] = {body: {cards: {M1_3head_3seed_20260922: {display_name: '周期指标 TAWSS · OSI', short_name: '周期指标', validation: {holdout_n: 34, fields: {tawss: {r2_pa: 0.74}, osi: {r2_pa: 0.56}}}},
  PF6_VF6_peak_3seed_20260920: {display_name: '体内压力与速度', short_name: '体场', validation: {fields: {}, note: '尚未建立'}}}}};
canned['/api/releases'] = {body: {releases: [{id: 'M1_3head_3seed_20260922', default: true, contract: {protocol: 'single_frame_wss_cycle_multi', fields: {tawss: {}, osi: {}}}},
  {id: 'PF6_VF6_peak_3seed_20260920', contract: {protocol: 'single_frame_volume', fields: {velocity: {}}}}]}};
canned['/api/jobs'] = {body: {jobs: [jobRecord('A'), jobRecord('B', {case_id: 'CASE_B', display_name: 'CASE_B'}), jobRecord('C', {status: 'awaiting_confirmation', review: {status: 'unreviewed'}}),
  jobRecord('D', {status: 'failed', error: {message: '开口数 7 ≠ 5'}})], total: 4, page: 1, page_size: 100}};
serve('A'); serve('B', {}, {name: 'CASE_B'});

function boot() {
  for (const f of WS) require(f);
  return wait(120);
}
function shellState() { return ns.shell.state(); }
function visibleText() { return textOf(app()); }
const done = extra => { console.log(JSON.stringify(Object.assign({errors, calls, vcalls, opened}, extra || {}))); };
"""


def _run(scenario: str, *, offline: bool = False, files=None) -> dict:
    _need_node()
    files = files or [f for f in WS_FILES if not (offline and f in OFFLINE_EXCLUDE)]
    program = (_HARNESS.replace("__WS__", json.dumps([str(V2 / f) for f in files])).replace("__OFFLINE__", "true" if offline else "false")
               + "\n(async () => {\n" + scenario + "\n})().catch(e => { errors.push('scenario: ' + (e && e.stack || e)); done({}); });\n")
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise AssertionError(result.stderr[-4000:])
    lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
    assert lines, result.stdout[-2000:] + result.stderr[-2000:]
    return json.loads(lines[-1])


def test_every_workspace_file_parses_and_index_loads_bundle_order():
    _need_node()
    for name in WS_FILES:
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    html = (V2 / "index.html").read_text(encoding="utf-8")
    import re
    srcs = re.findall(r'<script src="([^"]+)"', html)
    expected = ["/static/" + n for n in bundle["legacy_scripts"]] + ["/static/v2/" + n for n in bundle["scripts"]]
    assert srcs == expected                              # load order = bundle.json
    assert "<script>" not in html and not re.search(r"<script(?![^>]*\ssrc=)", html)   # CSP: no inline script
    assert set(WS_FILES) <= set(bundle["scripts"])


def test_login_form_then_list():
    out = _run(r"""
      canned['/api/session'] = {body: {authenticated: false, login: 'password', csrf_token: 'c0'}};
      canned['POST /api/session'] = () => { canned['/api/session'] = {body: {authenticated: true, login: 'password', csrf_token: 'csrf-2', username: 'admin', role: 'admin'}}; return {body: {authenticated: true, login: 'password', csrf_token: 'csrf-2', username: 'admin', role: 'admin'}}; };
      await boot();
      const user = walk(app(), e => e.attrs['aria-label'] === '用户名')[0], pass = walk(app(), e => e.attrs['aria-label'] === '口令')[0];
      const loginShown = Boolean(user && pass), railBefore = byClass(app(), 'ws-rail').length;
      user.value = 'admin'; pass.value = 'secret';
      fire(walk(app(), e => e.tagName === 'FORM')[0], 'submit');
      await wait(150);
      done({loginShown, railBefore, rail: byClass(app(), 'rail-result').map(textOf), todo: byClass(app(), 'todo-row').length, home: textOf(byClass(app(), 'todo-head')[0])});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["loginShown"] is True and out["railBefore"] == 0
    assert "POST /api/session" in out["calls"] and "GET /api/jobs" in out["calls"]
    assert len(out["rail"]) == 4 and out["todo"] >= 3 and out["home"].startswith("待办")


def test_open_result_overview_and_no_internal_words():
    out = _run(r"""
      await boot();
      await hashTo('#/job/A', 150);
      const insp = byClass(app(), 'ws-inspector')[0];
      const secs = byClass(insp, 'sec-title').map(textOf);
      const fields = walk(app(), e => e.dataset && e.dataset.field).map(e => e.dataset.field);
      const zoneRows = walk(byClass(insp, 'tbl-zones')[0], e => e.tagName === 'TR').length;
      const findings = byClass(insp, 'finding').map(textOf);
      const txt = visibleText();
      // click the first zone value → evidence lens
      const val = byClass(insp, 'val-link')[0]; fire(val, 'click'); await wait(30);
      const lensSteps = byClass(app(), 'lens-step').map(e => e.dataset.step);
      const lensText = textOf(byClass(app(), 'lens')[0]);
      done({secs, fields, zoneRows, findings, research: (txt.match(/研究用途，非诊断/g) || []).length, internal: /M1_3head|run-A|Feret|Gaussian/.test(txt),
        header: textOf(byClass(app(), 'insp-name')[0]), status: textOf(byClass(app(), 'vp-status')[0]), lensSteps, lensText, cb: cbInfos.slice(-1)[0], cur: shellState().cur.field});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["header"] == "CASE_A"
    assert out["secs"][:5] == ["结论", "分区", "左右对比 · TAWSS 均值", "形态", "发现"]
    assert out["fields"] == ["wss", "tawss", "osi", "rrt"] and out["cur"] == "tawss"
    assert out["zoneRows"] == 3                          # header + two zones
    assert len(out["findings"]) == 4                     # one per kind (two high clusters → one)
    assert any("含全场最大值" in f and "提示 · 无队列参照" in f for f in out["findings"])
    assert out["research"] == 1 and out["internal"] is False
    assert "周期指标 TAWSS · OSI" in out["status"] and "未复核" in out["status"] and "计算 8 秒" in out["status"]
    assert out["lensSteps"] == ["result", "where", "value", "method", "support", "consistency", "population"]
    assert "近端瘤颈" in out["lensText"] and "R² 0.74" in out["lensText"] and "暂无同口径参照" in out["lensText"]
    assert out["cb"] == "tawss"
    assert "GET /api/v2/jobs/A/manifest" in out["calls"]


def test_fast_switch_drops_the_slow_answer():
    out = _run(r"""
      delays['/api/v2/jobs/A/manifest'] = 250;
      await boot();
      global.location.hash = '#/job/A'; for (const f of winEvents.hashchange) f({});
      await wait(20);
      await hashTo('#/job/B', 500);
      const sr = vcalls.filter(c => c.includes(':setResult:'));
      done({cur: shellState().cur.jobId, header: textOf(byClass(app(), 'insp-name')[0]), top: textOf(byClass(app(), 'top-case')[0]), setResults: sr,
        current: byClass(app(), 'rail-result').filter(e => e.className.includes('current')).map(e => e.dataset.jobId)});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["cur"] == "B" and out["header"] == "CASE_B" and out["top"] == "CASE_B"
    assert out["setResults"] == ["1:setResult:B"]        # A's late manifest never reached the viewer
    assert out["current"] == ["B"]


def test_legacy_job_without_zones_or_fields_does_not_break():
    out = _run(r"""
      serve('L', {model_release: {id: 'X5D_v51_5seed_20260916'}}, {legacy: true, name: 'OLD_CASE'});
      await boot();
      await hashTo('#/job/L', 150);
      const insp = byClass(app(), 'ws-inspector')[0];
      const secs = byClass(insp, 'sec-title').map(textOf);
      shellState().viewerA.emit('pick', {pointIndex: 7, vertexIndex: 1, xyz: [1, 2, 3], segmentId: 0, s_from_root_mm: 12, value: 3.4});
      await wait(30);
      const lensText = textOf(byClass(app(), 'lens')[0]);
      const tabs = byClass(app(), 'tab').map(textOf);
      fire(walk(app(), e => e.dataset && e.dataset.tab === 'bookmarks')[0], 'click'); await wait(20);
      const bmEmpty = textOf(byClass(app(), 'bm-list')[0]);
      done({secs, lensText, tabs, bmEmpty, header: textOf(byClass(app(), 'insp-name')[0])});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["header"] == "OLD_CASE"
    assert "分区" not in out["secs"] and "发现" not in out["secs"]      # missing blocks are hidden, not zero
    assert "3.40 Pa" in out["lensText"] and "表面粗糙" in out["lensText"]   # trust bit 2 at vertex 1
    assert "未知" in out["lensText"]                                        # no validation → written as unknown
    assert out["tabs"] == ["概览", "读数", "书签", "工具"]
    assert "还没有书签" in out["bmEmpty"]


def test_unfinished_job_goes_to_the_input_page():
    out = _run(r"""
      canned['/api/v2/jobs/R/manifest'] = {status: 409, body: {error: 'not done', status: 'running'}};
      canned['/api/jobs/R'] = {body: {job: jobRecord('R')}};                  // record says done, manifest says 409 (race)
      canned['/api/jobs/C'] = {body: {job: jobRecord('C', {status: 'awaiting_confirmation', a: {proposal: {confidence: 0.884, mapping: {'3': 'out-le', '4': 'out-li', '5': 'out-ri', '6': 'out-re'},
        confidence_reasons: ['右侧髂内/髂外区分置信度 88%'], endpoints: [{segment_id: 0, kind: 'inlet', radius_mm: 13.2, center_mm: [0, 0, 0]},
        {segment_id: 3, kind: 'outlet', radius_mm: 2.8, center_mm: [1, 0, 0]}, {segment_id: 4, kind: 'outlet', radius_mm: 2.8, center_mm: [2, 0, 0]},
        {segment_id: 5, kind: 'outlet', radius_mm: 3.0, center_mm: [3, 0, 0]}, {segment_id: 6, kind: 'outlet', radius_mm: 3.5, center_mm: [4, 0, 0]}]}}})}};
      canned['POST /api/jobs/C/confirm'] = o => ({body: {job: jobRecord('C', {status: 'queued', version: 4})}, sent: o});
      canned['/api/jobs/D'] = {body: {job: jobRecord('D', {status: 'failed', error: {message: '开口数 7 ≠ 5'}})}};
      canned['/api/v2/jobs/D/inputcheck'] = {body: {status: 'failed', error: {message: '开口数 7 ≠ 5'}, input_check: null, mesh: null,
        openings: [9, 3, 3.2, 2.9, 3.1, 0.6, 0.8].map((r, i) => ({opening_index: i, equivalent_radius_mm: r, center_mm: [i, 0, 0]}))}};
      await boot();
      await hashTo('#/job/R', 150);
      const r = {mode: shellState().mode, title: textOf(byClass(app(), 'tb-title')[0])};
      await hashTo('#/job/C', 200);
      const selects = walk(byClass(app(), 'tbl-outlets')[0], e => e.tagName === 'SELECT').map(e => e.value);
      const ack = walk(app(), e => e.tagName === 'INPUT' && e.type === 'checkbox')[0];
      ack.checked = true; fire(ack, 'change'); await wait(10);
      const swapBtn = byText(app(), '左右互换'); fire(swapBtn.parentNode && swapBtn.tagName !== 'BUTTON' ? swapBtn.parentNode : swapBtn, 'click'); await wait(10);
      const swapped = walk(byClass(app(), 'tbl-outlets')[0], e => e.tagName === 'SELECT').map(e => e.value);
      const ack2 = walk(app(), e => e.tagName === 'INPUT' && e.type === 'checkbox')[0]; ack2.checked = true; fire(ack2, 'change'); await wait(10);
      const go = walk(app(), e => e.tagName === 'BUTTON' && textOf(e) === '确认出口并开始预测')[0];
      fire(go, 'click'); await wait(60);
      await hashTo('#/job/D', 200);
      const openings = byClass(app(), 'tbl-openings')[0];
      done({r, selects, swapped, confirm: calls.includes('POST /api/jobs/C/confirm'), failText: textOf(app()),
        stubs: walk(openings, e => e.className === 'warn-text').length});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["r"] == {"mode": "input", "title": "已完成"} or out["r"]["mode"] == "input"
    assert out["selects"] == ["out-le", "out-li", "out-ri", "out-re"]
    assert out["swapped"] == ["out-re", "out-ri", "out-li", "out-le"]
    assert out["confirm"] is True
    assert "开口数 7 ≠ 5" in out["failText"] and "报错对照表" in out["failText"]
    assert out["stubs"] == 2                                   # the two smallest of seven openings


def test_offline_mode_boots_without_service():
    out = _run(r"""
      const m = manifestFor('A');
      const bm = {schema: 'wssv2.bookmark/1', id: 'b1', name: '瘤颈低剪切', note: '看这里', run_identity: 'run-A', arrays_version: 'av-A', field: 'tawss', state: {camera: null}};
      const s1 = new Element('script'); s1.id = 'wssv2-manifest'; s1.textContent = JSON.stringify(m); body.appendChild(s1);
      const s2 = new Element('script'); s2.id = 'wssv2-offline'; s2.textContent = JSON.stringify({bookmarks: [bm], hide_name: true, exported_at: '2026-09-30T09:00:00+08:00', view: {}}); body.appendChild(s2);
      const appEl = new Element('div'); appEl.id = 'ws-app'; body.appendChild(appEl);
      calls.length = 0;
      await boot();
      await wait(100);
      const tabs = byClass(app(), 'tab').map(textOf);
      fire(walk(app(), e => e.dataset && e.dataset.tab === 'bookmarks')[0], 'click'); await wait(20);
      done({fetches: calls.filter(c => /^(GET|POST|PUT) /.test(c)), tabs, top: textOf(byClass(app(), 'top-note')[0]), name: textOf(byClass(app(), 'insp-name')[0]),
        rail: byClass(app(), 'ws-rail').length, switcher: byClass(app(), 'seg-results').length + byClass(app(), 'result-pick').length,
        bms: byClass(app(), 'bm-name').map(textOf), txt: visibleText()});
    """, offline=True)
    assert out["errors"] == [], out["errors"]
    assert out["fetches"] == []                                    # no service needed
    assert out["tabs"] == ["概览", "读数", "书签"] and out["rail"] == 0 and out["switcher"] == 0
    assert out["top"].startswith("离线报告 · 导出于 2026-09-30") and "导出时复核状态：未复核" in out["top"]
    assert out["name"] == "病例（名称已隐藏）" and "CASE_A" not in out["txt"]
    assert out["bms"] == ["瘤颈低剪切"]


def test_bookmarks_export_import_and_incompatible():
    out = _run(r"""
      await boot();
      await hashTo('#/job/A', 150);
      const B = ns.bookmarks;
      const ctx = {runIdentity: 'run-A', arraysVersion: 'av-A', dataVersion: 'dv-A', fields: ['wss', 'tawss']};
      const mine = B.make(Object.assign({}, ctx, {jobId: 'A', field: 'tawss', window: 'low', state: {field: 'tawss', camera: {position: [1, 2, 3]}}}), {name: '一', note: 'n'});
      B.add(ctx, mine);
      const exported = JSON.parse(B.exportJson('run-A'));
      const reviewedLater = B.compatible(mine, Object.assign({}, ctx, {dataVersion: 'dv-A2'}));      // review changed data_version
      const rebuilt = Object.assign({}, mine, {id: 'b-old', arrays_version: 'av-OLD'});
      const foreign = Object.assign({}, mine, {id: 'b-x', run_identity: 'run-Z'});
      const res = B.importJson(JSON.stringify({schema: 'wssv2.bookmarks/1', items: [mine, rebuilt, foreign, {junk: 1}]}), ctx);
      const bad = B.importJson('not json', ctx);
      fire(walk(app(), e => e.dataset && e.dataset.tab === 'bookmarks')[0], 'click'); await wait(20);
      const rows = byClass(app(), 'bm').map(e => ({name: textOf(byClass(e, 'bm-name')[0]), why: textOf(byClass(e, 'bm-why')[0])}));
      // restore the compatible one
      const openBtn = walk(byClass(app(), 'bm')[0], e => e.tagName === 'BUTTON' && textOf(e) === '打开')[0]; fire(openBtn, 'click'); await wait(30);
      done({exported: exported.items.length, schema: exported.schema, reviewedLater, res, bad, rows, foreignStored: B.list('run-Z').length,
        restored: vcalls.filter(c => c.endsWith(':applyState')).length, field: shellState().cur.field, win: shellState().cur.window});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["exported"] == 1 and out["schema"] == "wssv2.bookmarks/1"
    assert out["reviewedLater"]["ok"] is True                      # compatibility = run_identity + arrays_version
    assert out["res"]["added"] == 1 and out["res"]["stale"] == 1 and out["res"]["foreign"] == 1 and out["res"]["errors"]
    assert out["bad"]["errors"] == ["文件不是 JSON。"]
    assert [r["why"] for r in out["rows"]] == ["", "不能套用：结果数据已重建（数组版本不同）"]
    assert out["foreignStored"] == 1
    assert out["restored"] >= 1 and out["field"] == "tawss" and out["win"] == "low"


def test_keyboard_respects_inputs_and_dialogs():
    out = _run(r"""
      await boot();
      await hashTo('#/job/A', 150);
      const input = new Element('input'); input.type = 'text';
      await key('1', input);
      const afterInput = shellState().cur.field;
      await key('1');
      const afterBody = shellState().cur.field;
      await key('9');                                              // no ninth field: ignored
      const after9 = shellState().cur.field;
      await key('b', new Element('textarea'));
      const dialogAfterTextarea = byId('ws-dialog') ? byId('ws-dialog').open : false;
      const ctrl = await key('l', body, {ctrlKey: true});
      await key('?');
      const helpOpen = byId('ws-dialog').open && textOf(byId('ws-dialog')).includes('快捷键');
      await key('2');                                              // dialog open: ignored
      const underDialog = shellState().cur.field;
      byId('ws-dialog').close(); await wait(10);
      await key('g'); await wait(20);
      const cursorOn = Boolean(shellState().cur.cursor);
      await key('ArrowRight', body, {shiftKey: true});
      const cursorPos = shellState().cur.cursor.state().s_mm;
      await key('Escape'); await wait(10);
      await key(']'); await wait(20);
      done({afterInput, afterBody, after9, dialogAfterTextarea, ctrlPrevented: Boolean(ctrl.prevented), helpOpen, underDialog, cursorOn, cursorPos,
        cursorOff: shellState().cur.cursor === null, finding: shellState().cur.findingSel,
        markers: vcalls.filter(c => c.includes(':markers:')).slice(-1)[0]});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["afterInput"] == "tawss" and out["afterBody"] == "wss" and out["after9"] == "wss"
    assert out["dialogAfterTextarea"] is False and out["ctrlPrevented"] is False
    assert out["helpOpen"] is True and out["underDialog"] == "wss"
    assert out["cursorOn"] is True and out["cursorPos"] == 110.5          # 211 / 2 capped at 120, then +5 mm
    assert out["cursorOff"] is True
    assert out["finding"] == "F4" and out["markers"].endswith(":markers:1")   # heaviest finding first


def test_questions_split_compare_conditions_and_undo():
    out = _run(r"""
      serve('V', {model_release: {id: 'PF6_VF6_peak_3seed_20260920', contract: {protocol: 'single_frame_volume', fields: {velocity: {}}}}, family: 'volume'},
        {family: 'volume', resultName: '体内压力与速度', release: 'PF6_VF6_peak_3seed_20260920', sha: 'sha-A',
         fields: [{id: 'pressure', short_label: '压力', units: 'Pa', kind: 'scalar', temporal: 'frame', time_index: 0, tier: 'model', display: {}},
                  {id: 'speed', short_label: '速度', units: 'm/s', kind: 'scalar', temporal: 'frame', time_index: 0, tier: 'model', display: {p99: 1.37}},
                  {id: 'velocity', kind: 'vector', units: 'm/s'}]});
      await boot();
      await hashTo('#/job/A', 150);
      await hashTo('#/job/A?q=low', 80);
      const low = {field: shellState().cur.field, win: shellState().cur.window, bar: textOf(byClass(app(), 'ws-qbar')[0]), hash: global.location.hash};
      fire(walk(byClass(app(), 'ws-qbar')[0], e => e.tagName === 'BUTTON' && textOf(e) === '撤销')[0], 'click'); await wait(30);
      const undone = {field: shellState().cur.field, win: shellState().cur.window, hidden: byClass(app(), 'ws-qbar')[0].hidden};
      await hashTo('#/job/A?q=pair', 120);
      const pair = {split: shellState().cur.split, created: vcalls.filter(c => c.includes(':create:')).length, linked: vcalls.includes('link')};
      await key('Escape'); await wait(20);
      await hashTo('#/job/A?v=compare&cmp=V', 200);
      const rows = ns.compare.conditions(shellState().cur.manifest, shellState().cur.compare.manifest, 'tawss', shellState().cur.compare.field);
      done({low, undone, pair, cmpField: shellState().cur.compare.field, mode: shellState().cur.compare.mode, sync: shellState().cur.compare.sync,
        rows: rows.map(r => [r.key, r.verdict]), tab: textOf(byClass(app(), 'tab').filter(e => e.className.includes('on'))[0]),
        cmpText: textOf(byClass(app(), 'insp-body')[0])});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["low"]["field"] == "tawss" and out["low"]["win"] == "low" and "低剪切窗" in out["low"]["bar"] and "q=low" in out["low"]["hash"]
    assert out["undone"]["hidden"] is True and out["undone"]["win"] == "adaptive"
    assert out["pair"]["split"]["field"] == "osi" and out["pair"]["linked"] is True
    assert out["cmpField"] == "speed" and out["mode"] == "each"          # different field → no same-scale mode
    assert out["sync"] is True                                           # same input sha → views synced
    assert dict(out["rows"])["field"] == "不一致" and dict(out["rows"])["geometry"] == "一致"
    assert out["tab"] == "比较" and "不做逐点差值" in out["cmpText"]


# ---------------------------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------------------------
def test_pure_helpers():
    out = _run(r"""
      for (const f of WS) require(f);
      const S = ns.shell, U = ns.ui, R = ns.rail, O = ns.overview, UP = ns.upload, IN = ns.input;
      const hashes = [S.parseHash('#/job/20260929_230442_a7273f1670e4?v=compare&f=tawss&cmp=x_1&q=low&bm=b12'), S.parseHash('#/'), S.parseHash('#/job/../x'), S.parseHash('#/job/a?v=evil&f=<x>')];
      const built = S.buildHash({jobId: 'a_1', f: 'osi', q: 'pair'});
      const nums = [U.sig(52.60006), U.sig(0.24853), U.sig(121.6), U.sig(13749), U.sig(9.996), U.sig(null), U.pct(0.408), U.pct(0.0057), U.pct(0), U.num(17.006, 'Pa'), U.num(0.357, '1'), U.trim(0.4)];
      const tree = R.model([
        {id: 'j1', case_id: 'LV', status: 'done', input_sha256: 's1', created_at: '2026-09-29T08:00:00+08:00', review: {status: 'reviewed'}},
        {id: 'j2', case_id: 'LV', status: 'done', input_sha256: 's1', created_at: '2026-09-20T08:00:00+08:00'},
        {id: 'j3', case_id: 'X', patient_id: 'P-1', status: 'failed', input_sha256: 's2', created_at: '2026-09-18T08:00:00+08:00'},
        {id: 'j4', case_id: 'Y', patient_id: 'P-1', status: 'running', input_sha256: 's3', created_at: '2026-09-19T08:00:00+08:00', scan_date: '2025-03-01'}], {});
      const filtered = R.model([{id: 'a', case_id: 'Q', status: 'failed'}, {id: 'b', case_id: 'Q', status: 'done'}], {status: 'failed'}).length;
      const m = {analysis: {findings: {items: [{id: 'F1', kind: 'high_wss_cluster', severity: 'note', rank: 1}, {id: 'F2', kind: 'high_wss_cluster', severity: 'attention', rank: 2},
        {id: 'F3', kind: 'low_wss_cluster', severity: 'attention', rank: 3}, {id: 'F4', kind: 'max_diameter', severity: 'info', source: 'morphology', rank: 4}], review: {items: {F1: {decision: 'confirmed'}}}}}};
      const fm = O.findingsModel(m);
      const zc = O.zoneColumns({primary_fields: ['wss'], zones: [{fields: {wss: {mean: 1}}}]}).map(c => c.label);
      const names = ['LV_GUO_YOU.stl', 'lv guo you.STL', 'P-001.stl', 'case_12.stl', 'Zhang.stl'].map(UP.looksLikeName);
      const hints = IN.openingHints([{opening_index: 0, equivalent_radius_mm: 12, center_mm: [0, 0, 0]}, {opening_index: 1, equivalent_radius_mm: 3, center_mm: [0, 0, 0]},
        {opening_index: 2, equivalent_radius_mm: 3.1, center_mm: [0, 0, 0]}, {opening_index: 3, equivalent_radius_mm: 2.9, center_mm: [0, 0, 0]}, {opening_index: 4, equivalent_radius_mm: 0.7, center_mm: [0, 0, 0]}]).map(o => o.small);
      const swapped = IN.swap({'3': 'out-le', '4': 'out-li', '5': 'out-re', '6': 'out-ri'}, 'lr');
      const valid = [IN.validMapping(swapped, ['3', '4', '5', '6']), IN.validMapping({'3': 'out-le', '4': 'out-le', '5': 'out-re', '6': 'out-ri'}, ['3', '4', '5', '6'])];
      const prefs = ns.store.sanitizePrefs({tier: 'full', lighting: 'disco', cmap: 'viridis', tab: 'compare', rail: false});
      const nar = O.narrativeModel({analysis: {narrative: {zh: ['一。', '以上为参考描述，非诊断结论。'], edited: null}}});
      const edited = O.narrativeModel({analysis: {narrative: {zh: ['一。'], edited: '人工写的。', edited_by: '医生A'}}});
      const choices = UP.choices([{id: 'M1', default: true, contract: {protocol: 'single_frame_wss_cycle_multi'}}, {id: 'PF6', contract: {protocol: 'single_frame_volume'}}],
        {M1: {display_name: '周期指标 TAWSS · OSI', short_name: '周期指标', validation: {holdout_n: 34, fields: {tawss: {r2_pa: 0.7391}, osi: {r2_pa: 0.5648}}}}});
      const follow = O.followupModel({scans: [{date: '2025-03-01', scan_label: '基线', geometry: {max_diameter_mm: 52.1, sac_volume_ml: 96.3}, models: {M1: {tawss_mean_pa: 0.71}}},
        {date: '2026-03-01', geometry: {max_diameter_mm: 55.3}, models: {M1: {tawss_mean_pa: 0.66}}}], growth: {max_diameter_mm: {per_year: 3.2}}, notes: ['n']}, 'M1');
      const noFollow = O.followupModel({scans: [{date: '2026-03-01'}]}, 'M1');
      done({follow, noFollow, hashes, built, nums, tree: tree.map(c => [c.name, c.scans.map(s => [s.label, s.jobs.map(j => j.id)])]), order: R.order(tree), filtered,
        top: fm.top.map(i => i.id), undecided: fm.undecided.map(i => i.id), zc, names, hints, swapped, valid, prefs, nar, edited,
        choices: choices.map(c => [c.value, c.label, c.detail])});
    """)
    assert out["errors"] == [], out["errors"]
    h = out["hashes"]
    assert h[0] == {"jobId": "20260929_230442_a7273f1670e4", "v": "compare", "f": "tawss", "cmp": "x_1", "q": "low", "bm": "b12"}
    assert h[1]["jobId"] is None and h[2]["jobId"] is None and h[3] == {"jobId": "a", "v": None, "f": None, "cmp": None, "q": None, "bm": None}
    assert out["built"] == "#/job/a_1?f=osi&q=pair"
    assert out["nums"] == ["52.6", "0.249", "122", "13700", "10.0", "—", "41%", "0.6%", "0%", "17.0 Pa", "0.357", "0.4"]
    assert out["tree"][0][0] == "LV" and out["tree"][0][1] == [["首次上传 2026-09-20", ["j1", "j2"]]]
    # a filled patient number is the display name (contract §3) and groups both scans, newest first
    assert out["tree"][1][0] == "P-1" and [s[1] for s in out["tree"][1][1]] == [["j4"], ["j3"]]
    assert out["order"] == ["j1", "j2", "j4", "j3"] and out["filtered"] == 1
    assert out["top"] == ["F3", "F2", "F4"] and out["undecided"] == ["F2", "F3", "F4"]   # same severity: low side first
    assert out["zc"] == ["WSS 均值", "低 WSS 占比", "WSS p99"]
    assert out["names"] == [True, True, False, False, False]
    assert out["hints"] == [False, False, False, False, True]
    assert out["swapped"] == {"3": "out-re", "4": "out-ri", "5": "out-le", "6": "out-li"} and out["valid"] == [True, False]
    assert out["prefs"]["tier"] == "full" and out["prefs"]["lighting"] == "flat" and out["prefs"]["tab"] == "overview" and out["prefs"]["rail"] is False
    assert out["nar"] == {"edited": False, "sentences": ["一。"]}             # the disclaimer lives once, at the inspector bottom
    assert out["edited"]["edited"] is True and out["edited"]["sentences"] == ["人工写的。"]
    assert out["choices"][0] == ["M1", "周期指标 TAWSS · OSI", "34 例留出，与 CFD 的 R²：TAWSS 0.74，OSI 0.56"]
    assert out["choices"][1][1] == "体内压力与速度" and out["choices"][2][0] == "M1+PF6"
    assert not any("M1_" in c[1] or "PF6" in c[1] for c in out["choices"])     # never a release code on screen
    # U17: two scans of one patient → follow-up rows with this release's cycle metric and the growth per year
    assert out["noFollow"] is None
    assert out["follow"]["metricLabel"] == "TAWSS 均值" and [r["diameter"] for r in out["follow"]["rows"]] == [52.1, 55.3]
    assert out["follow"]["growth"] == {"per_year": 3.2}
