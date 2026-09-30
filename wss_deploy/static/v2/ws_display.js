/* WSS workspace v2 — Lane E (S6e): display options (PHASE2_LANES.md §5 E).
 * Colour scale: bands and observation thresholds per field, 「设为默认」 per field; layers: input STL, the field maximum
 * (TAWSS also its minimum) as a persistent marker, the stagnation hatch, a reset-view button (key 0); volume: wall
 * opacity, streamline density and width, velocity arrows on the interior points, display units (Pa / mmHg,
 * m/s / cm/s) for the colour bar, the probe card and the section page; compare: 以左为准 / 以右为准 and the same-scale
 * mode sharing bands and thresholds.  Everything here changes what is drawn, never a number: thresholds only move the
 * lines on the colour bar (and the point-equal fractions next to them, the classic definition), units multiply by the
 * classic factors (VolumeViewerCore.convertUnit), statistics keep the release's definitions.
 * The viewer kernel asks this module through ns.viewer.displayProvider (core_viewer.js, lane E block); the shell
 * through its extension registry (layers menu, toolbar, keys, result hooks); the colour bar through its adjust button. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.display = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };

  var PREFS_KEY = 'wssv2:display:1';
  var BAND_STEPS = [0, 4, 6, 8, 10, 12, 16, 20];                       // classic 色带分段
  var UNITS = { pressure: { 'Pa': 1, 'mmHg': 1 / 133.322 }, velocity: { 'm/s': 1, 'cm/s': 100 } };   // classic volume_viewer UNITS
  var DENSITY = [{ v: 1, label: '全部' }, { v: 2, label: '一半' }, { v: 3, label: '三分之一' }, { v: 5, label: '五分之一' }];
  var MIN_FIELDS = { tawss: true };                                     // classic: TAWSS also marks its lowest point
  var ABOVE = { osi: true, rrt: true, ecap: true };
  var DEFAULTS = { schema: 'wssv2.display/1', defaults: {}, units: { pressure: 'Pa', velocity: 'm/s' },
    layers: { stl: false, peaks: true, stagnation: false, vectors: false }, volume: { opacity: 0.1, density: 1, width: 1, thin: true } };
  var ID_RE = /^[A-Za-z0-9_.-]{1,40}$/;

  // ------------------------------------------------------------------ pure helpers (Node-tested)
  // Classic validThresholds: three increasing non-negative numbers.
  function validThresholds(t) {
    t = (Array.isArray(t) ? t : []).map(function (x) { return x === '' || x === null || x === undefined ? NaN : Number(x); });
    return t.length === 3 && t.every(Number.isFinite) && t[0] >= 0 && t[0] < t[1] && t[1] < t[2] ? t : null;
  }
  function sameThresholds(a, b) {
    return Array.isArray(a) && Array.isArray(b) && a.length === 3 && b.length === 3 && a.every(function (x, k) { return Math.abs(x - b[k]) <= 1e-9 * Math.max(1, Math.abs(b[k])); });
  }
  // Factor of a display unit (value × factor); the classic core's own conversion when it is loaded.
  function unitFactor(kind, unit) {
    var table = UNITS[kind] || {};
    if (!Object.prototype.hasOwnProperty.call(table, unit)) return 1;
    var VC = root.VolumeViewerCore;
    if (VC && typeof VC.convertUnit === 'function') { var f = VC.convertUnit(1, kind, unit); if (Number.isFinite(f) && f > 0) return f; }
    return table[unit];
  }
  function unitOptions(kind) { return Object.keys(UNITS[kind] || {}); }
  // Which display unit applies to a field: volume pressures (Pa) and speeds (m/s) only; wall shear stays in Pa.
  function unitKind(field, family) {
    if (family !== 'volume' || !field) return null;
    if (field.units === 'Pa') return 'pressure';
    if (field.units === 'm/s') return 'velocity';
    return null;
  }
  function isAbove(field) {
    var d = field && field.display && field.display.threshold_direction;
    return d ? d === 'above' : Boolean(field && ABOVE[field.id]);
  }
  // Point-equal fractions at three thresholds (classic thrFracs / thresholdAreas): below-fields count x < t0, x > t1,
  // x > t2; above-fields (OSI, RRT, ECAP) x > t0, x > t1, x > t2.  NaN is not counted.
  function thresholdFractions(values, t, above) {
    var n = 0, a = 0, b = 0, c = 0;
    for (var i = 0; i < values.length; i++) {
      var x = values[i];
      if (!Number.isFinite(x)) continue;
      n++;
      if (above ? x > t[0] : x < t[0]) a++;
      if (x > t[1]) b++;
      if (x > t[2]) c++;
    }
    return { n: n, fractions: n ? [a / n, b / n, c / n] : [0, 0, 0] };
  }
  function pointKeys(m) { var p = (m && m.geometry && m.geometry.points) || {}; return { xyz: p.xyz || null, segment: p.segment || null }; }
  function arr(result, key) { try { return key && result.declared(key) && result.has(key) ? result.array(key) : null; } catch (_) { return null; } }
  // Persistent markers of a wall field (classic fieldPeak): peak WSS at the summary's point (metrics.py), any other field
  // at its first largest finite point value (np.argmax order); TAWSS also at its first smallest (np.argmin order).
  function peakMarkers(result, fieldId) {
    if (!result || result.family !== 'wall' || !fieldId) return [];
    var m = result.manifest, f = result.field(fieldId);
    if (!f) return [];
    var k = pointKeys(m), PV = arr(result, k.xyz), PS = arr(result, k.segment), v = null;
    try { v = f.arrays && f.arrays.read && result.has(f.arrays.read) ? result.fieldArray(fieldId, 'read') : null; } catch (_) { v = null; }
    var at = function (i, kind) {
      return i < 0 || !PV || 3 * i + 2 >= PV.length ? null : { kind: kind, index: i, value: v[i], xyz: [PV[3 * i], PV[3 * i + 1], PV[3 * i + 2]], segmentId: PS ? Number(PS[i]) : null };
    };
    var out = [], peak = m && m.analysis && m.analysis.peak;
    if (fieldId === 'wss' && peak && Array.isArray(peak.xyz_mm) && peak.xyz_mm.length === 3 && peak.xyz_mm.every(function (x) { return Number.isFinite(+x); }) && Number.isFinite(+peak.max_pa)) {
      out.push({ kind: 'max', index: null, value: Number(peak.max_pa), xyz: peak.xyz_mm.map(Number), segmentId: Number.isFinite(+peak.segment_id) ? +peak.segment_id : null });
    } else if (v) {
      var best = -1;
      for (var i = 0; i < v.length; i++) { var x = v[i]; if (Number.isFinite(x) && (best < 0 || x > v[best])) best = i; }
      var mx = at(best, 'max'); if (mx) out.push(mx);
    }
    if (MIN_FIELDS[fieldId] && v) {
      var lo = -1;
      for (var j = 0; j < v.length; j++) { var y = v[j]; if (Number.isFinite(y) && (lo < 0 || y < v[lo])) lo = j; }
      var mn = at(lo, 'min'); if (mn) out.push(mn);
    }
    return out;
  }
  function stagnationCriteria(m) {
    var c = (m && m.analysis && m.analysis.cycle && m.analysis.cycle.stagnation && m.analysis.cycle.stagnation.criteria) || {};
    return { tawss_lt_pa: Number.isFinite(+c.tawss_lt_pa) ? +c.tawss_lt_pa : 0.4, osi_gt: Number.isFinite(+c.osi_gt) ? +c.osi_gt : 0.1 };
  }
  function fieldOf(m, id) { return ((m && m.fields) || []).filter(function (f) { return f && f.id === id; })[0] || null; }
  function hasStagnation(m) {
    var t = fieldOf(m, 'tawss'), o = fieldOf(m, 'osi');
    return Boolean(t && o && t.arrays && o.arrays && t.arrays.display && o.arrays.display);
  }
  function velocityKey(m) { var f = fieldOf(m, 'velocity'); return f && f.arrays && typeof f.arrays.read === 'string' ? f.arrays.read : null; }
  function sanitizePrefs(raw) {
    var p = JSON.parse(JSON.stringify(DEFAULTS));
    if (!raw || typeof raw !== 'object') return p;
    if (raw.defaults && typeof raw.defaults === 'object') Object.keys(raw.defaults).forEach(function (id) {
      var d = raw.defaults[id];
      if (!ID_RE.test(id) || !d || typeof d !== 'object') return;
      var e = {};
      if (BAND_STEPS.indexOf(Number(d.bands)) >= 0 && Number(d.bands) > 0) e.bands = Number(d.bands);
      var t = validThresholds(d.thresholds); if (t) e.thresholds = t;
      if (Object.keys(e).length) p.defaults[id] = e;
    });
    var u = raw.units || {};
    if (UNITS.pressure[u.pressure]) p.units.pressure = u.pressure;
    if (UNITS.velocity[u.velocity]) p.units.velocity = u.velocity;
    var L = raw.layers || {};
    Object.keys(p.layers).forEach(function (k) { if (typeof L[k] === 'boolean') p.layers[k] = L[k]; });
    var V = raw.volume || {};
    if (V.opacity !== null && V.opacity !== '' && Number.isFinite(+V.opacity) && +V.opacity >= 0 && +V.opacity <= 0.5) p.volume.opacity = +V.opacity;
    if ([1, 2, 3, 5].indexOf(+V.density) >= 0) p.volume.density = +V.density;
    if (Number.isFinite(+V.width) && +V.width >= 0.4 && +V.width <= 3) p.volume.width = +V.width;
    if (typeof V.thin === 'boolean') p.volume.thin = V.thin;
    return p;
  }
  function sanitizeSessionState(d) {
    var out = { bands: {}, thresholds: {} };
    if (!d || typeof d !== 'object') return out;
    Object.keys(d.bands || {}).forEach(function (id) { var b = Number(d.bands[id]); if (ID_RE.test(id) && BAND_STEPS.indexOf(b) >= 0) out.bands[id] = b; });
    Object.keys(d.thresholds || {}).forEach(function (id) { var t = validThresholds(d.thresholds[id]); if (ID_RE.test(id) && t) out.thresholds[id] = t; });
    return out;
  }

  // ------------------------------------------------------------------ preferences and per-result sessions
  var prefsCache = null;
  function prefs() {
    if (!prefsCache) prefsCache = sanitizePrefs(ns.store && typeof ns.store.read === 'function' ? ns.store.read(PREFS_KEY) : null);
    return prefsCache;
  }
  function savePrefs(p) { prefsCache = sanitizePrefs(p); if (ns.store && typeof ns.store.write === 'function') ns.store.write(PREFS_KEY, prefsCache); return prefsCache; }
  // Per result (run identity): the bands and thresholds chosen for each field in this page session.  They travel with
  // the viewer state (getState().display), so the reading position and bookmarks bring them back.
  var sessions = {};
  function session(rid) { var k = String(rid || '-'); if (!sessions[k]) sessions[k] = { bands: {}, thresholds: {} }; return sessions[k]; }
  function ridOf(result) { return result ? (result.runIdentity || (result.manifest && result.manifest.result && result.manifest.result.run_identity) || result.jobId || '-') : null; }
  function effBands(s, id) { var d = prefs().defaults[id]; return s.bands[id] !== undefined ? s.bands[id] : (d && d.bands !== undefined ? d.bands : 0); }
  function effThresholds(s, id) { var d = prefs().defaults[id]; return s.thresholds[id] || (d && d.thresholds) || null; }
  function reset() { prefsCache = null; sessions = {}; closePop(); }

  // ------------------------------------------------------------------ the shell, its viewers and sides
  var API = null;
  function shell() { return ns.shell && typeof ns.shell.state === 'function' ? ns.shell.state() : null; }
  function sideOf(v) { var S = shell(); if (!S || !v) return null; return v === S.viewerA ? 'a' : v === S.viewerB ? 'b' : null; }
  function resultOf(v) { try { return v && typeof v.result === 'function' ? v.result() : null; } catch (_) { return null; } }
  function compareSame() { var S = shell(); return Boolean(S && S.cur && S.cur.compare && S.cur.compare.mode === 'same'); }
  // The session whose bands / thresholds a viewer shows: its own result's, except the right side of a same-scale
  // comparison, which follows the left (one scale, one set of bands and thresholds).
  function sessionFor(v, own) {
    var side = sideOf(v); if (!side) return null;
    var r = resultOf(v); if (!r) return null;
    var S = shell();
    if (!own && side === 'b' && compareSame() && S.viewerA && resultOf(S.viewerA)) r = resultOf(S.viewerA);
    return session(ridOf(r));
  }
  function viewers() { var S = shell(); return S ? [S.viewerA, S.viewerB].filter(Boolean) : []; }
  function refreshAll() {
    viewers().forEach(function (v) { if (typeof v.refreshDisplay === 'function') { try { v.refreshDisplay(); } catch (e) { if (root.console) root.console.error(e); } } });
    var S = shell();
    if (S && S.cur && S.cur.slice) { try { S.cur.slice.refresh(); } catch (_) {} }
  }
  function preload(result, keys) {
    if (!result || typeof result.preload !== 'function') return Promise.resolve(null);
    var need = keys.filter(function (k) { return k && result.declared(k) && !result.has(k); });
    return need.length ? result.preload(need) : Promise.resolve(result);
  }
  // Arrays a display option needs beyond the current field: velocity for the arrows, the TAWSS / OSI display arrays for
  // the stagnation hatch (the viewer loads those in the background too; asking again costs nothing).
  function ensureArrays() {
    var P = prefs(), jobs = [];
    viewers().forEach(function (v) {
      var r = resultOf(v); if (!r) return;
      var keys = [];
      if (P.layers.vectors && r.family === 'volume') keys.push(velocityKey(r.manifest));
      if (P.layers.stagnation && r.family === 'wall' && hasStagnation(r.manifest)) ['tawss', 'osi'].forEach(function (id) { var f = r.field(id); if (f && f.arrays) keys.push(f.arrays.display); });
      if (keys.length) jobs.push(preload(r, keys));
    });
    return Promise.all(jobs);
  }

  // ------------------------------------------------------------------ the provider the viewer kernel asks
  function fmtNum(v) { return ns.util && ns.util.fmtSig ? ns.util.fmtSig(v) : String(+(+v).toPrecision(3)); }
  function fmtCut(v) { return ns.util && ns.util.fmtTrim ? ns.util.fmtTrim(v) : String(+(+v).toPrecision(3)); }   // thresholds: 0.4 not 0.400
  function unitText(u) { return ns.util && ns.util.unitText ? ns.util.unitText(u) : (u || ''); }
  var provider = {
    bands: function (v, fieldId) { var s = sessionFor(v); return s ? effBands(s, fieldId) : null; },
    thresholds: function (v, fieldId) { var s = sessionFor(v); return s ? effThresholds(s, fieldId) : null; },
    units: function (v, field) {
      if (!sideOf(v)) return null;
      var r = resultOf(v), kind = unitKind(field, r && r.family);
      if (!kind) return null;
      var u = prefs().units[kind], f = unitFactor(kind, u);
      return f === 1 ? null : { units: u, factor: f };
    },
    flags: function (v) {
      if (!sideOf(v)) return {};
      var P = prefs(), r = resultOf(v), m = r && r.manifest;
      return { stl: P.layers.stl, stagnation: P.layers.stagnation && hasStagnation(m) ? stagnationCriteria(m) : null,
        opacity: P.volume.opacity, density: P.volume.density, width: P.volume.width, thin: P.volume.thin, vectors: P.layers.vectors };
    },
    markers: function (v) {
      if (!sideOf(v) || !prefs().layers.peaks || prefs().layers.stl) return [];
      var r = resultOf(v), fid = v.field && v.field(), f = r && r.field(fid);
      return peakMarkers(r, fid).map(function (p) {
        p.text = (p.kind === 'min' ? '最小 ' : '最大 ') + fmtNum(p.value) + (f && unitText(f.units) ? ' ' + unitText(f.units) : '');
        return p;
      });
    },
    state: function (v) {
      var s = sessionFor(v, true);
      return s && (Object.keys(s.bands).length || Object.keys(s.thresholds).length) ? { v: 1, bands: Object.assign({}, s.bands), thresholds: Object.assign({}, s.thresholds) } : null;
    },
    restore: function (v, d) {
      var s = sessionFor(v, true); if (!s) return;
      var c = sanitizeSessionState(d);
      s.bands = c.bands; s.thresholds = c.thresholds;
    }
  };
  if (ns.viewer && typeof ns.viewer === 'object') ns.viewer.displayProvider = provider;

  // Display unit of a physical quantity ('pressure' | 'velocity') for the section page and the probe card.
  function displayUnit(kind) {
    var u = prefs().units[kind];
    if (!UNITS[kind] || !u) return null;
    return { units: u, factor: unitFactor(kind, u) };
  }
  // A number of a volume result in its display unit, for the overview, the lens, the along-vessel curves and the colour
  // window labels: {value, units}.  Wall shear stress and every other quantity come back unchanged.
  function toDisplay(value, units, family) {
    var kind = unitKind({ units: units }, family), d = kind ? displayUnit(kind) : null;
    if (!d || !(d.factor > 0) || d.factor === 1 || value === null || value === undefined || !Number.isFinite(+value)) return { value: value, units: units };
    return { value: +value * d.factor, units: d.units };
  }

  // ------------------------------------------------------------------ floating panels (HUD style, over the stage)
  var pop = null;
  function closePop() {
    if (!pop) return;
    var p = pop; pop = null;
    if (p.el.parentNode) p.el.parentNode.removeChild(p.el);
    var doc = root.document;
    if (doc && doc.removeEventListener) { doc.removeEventListener('pointerdown', p.outside, true); doc.removeEventListener('keydown', p.esc, true); }
  }
  function openPop(kind, title, anchorRect, place, build) {
    closePop();
    var S = shell(), doc = root.document, h = ui().h;
    if (!S || !doc) return null;
    var body = h('div', { 'class': 'wsd-body' });
    var el = h('div', { 'class': 'wsd-pop wsd-pop-' + kind, role: 'dialog', 'aria-label': title },
      h('div', { 'class': 'wsd-head' }, h('span', { 'class': 'wsd-title', text: title }), ui().iconButton('close', '关闭', closePop, { cls: 'wsd-close' })), body);
    var host = (S.els && S.els.app) || doc.body;
    host.appendChild(el);
    var p = { kind: kind, el: el, body: body, build: build,
      outside: function (e) {
        var t = e && e.target;
        if (t && el.contains && el.contains(t)) return;
        if (t && t.closest && t.closest('.wssv2-cb-adjust, .ws-menu')) return;
        closePop();
      },
      esc: function (e) { if (e && e.key === 'Escape') { if (e.stopPropagation) e.stopPropagation(); if (e.preventDefault) e.preventDefault(); closePop(); } } };
    pop = p;
    if (doc.addEventListener) { doc.addEventListener('pointerdown', p.outside, true); doc.addEventListener('keydown', p.esc, true); }
    build(body);
    place(el, anchorRect || null);
    return p;
  }
  function rectOf(el) { try { return el && el.getBoundingClientRect ? el.getBoundingClientRect() : null; } catch (_) { return null; } }
  // Colour-scale panel: to the left of the colour bar, top aligned; in a narrow viewport (two side by side) where that
  // would cover the tool strip, under the colour bar instead, right aligned with it.
  var POP_W = 272, STRIP = 64;
  function placeByBar(vpRect) {
    return function (el, r) {
      var vw = root.innerWidth || 1280, vh = root.innerHeight || 800;
      if (!r) { el.style.right = '140px'; el.style.top = '120px'; return; }
      if (!vpRect || r.left - vpRect.left - STRIP >= POP_W + 10) {
        el.style.right = Math.max(8, Math.round(vw - r.left + 10)) + 'px';
        el.style.top = Math.max(56, Math.round(r.top - 6)) + 'px';
        return;
      }
      var hgt = el.offsetHeight || 200;
      el.style.right = Math.max(8, Math.round(vw - r.right)) + 'px';
      el.style.top = Math.max(56, Math.min(Math.round(r.bottom + 6), vh - hgt - 8)) + 'px';
    };
  }
  function placeRightOf(el, r) {  // volume panel: to the right of the layers button
    if (!r) { el.style.left = '64px'; el.style.top = '160px'; return; }
    el.style.left = Math.round(r.right + 10) + 'px';
    el.style.top = Math.max(56, Math.round(r.top - 6)) + 'px';
  }
  function row(label, control, tip) {
    var h = ui().h;
    return h('div', { 'class': 'wsd-row' }, h('span', { 'class': 'wsd-k', text: label }), control, tip ? ui().infoTip(tip) : null);
  }
  function saveSoon() { if (API && typeof API.saveViewSoon === 'function') { try { API.saveViewSoon(); } catch (_) {} } }

  // ---- colour scale (from the colour bar's adjust button)
  function viewerForLegend(el) {
    var S = shell(); if (!S || !S.els) return null;
    if (S.els.vpB && S.els.vpB.legend === el && S.viewerB) return S.viewerB;
    return S.viewerA || null;
  }
  function setBands(v, fieldId, n) {
    var s = sessionFor(v); if (!s) return;
    s.bands[fieldId] = BAND_STEPS.indexOf(n) >= 0 ? n : 0;
    refreshAll(); saveSoon();
  }
  function setThresholds(v, fieldId, t, field) {
    var s = sessionFor(v); if (!s) return;
    var base = validThresholds(field && field.display && field.display.thresholds), def = prefs().defaults[fieldId];
    if (!t || sameThresholds(t, (def && def.thresholds) || base)) delete s.thresholds[fieldId]; else s.thresholds[fieldId] = t;
    refreshAll(); saveSoon();
  }
  function setUnit(kind, u) {
    var P = prefs(); if (!UNITS[kind] || !UNITS[kind][u]) return;
    P.units[kind] = u; savePrefs(P);
    refreshAll();
    if (!API) return;
    if (typeof API.renderToolbar === 'function') API.renderToolbar();   // the colour-window labels carry the unit
    if (typeof API.drawLabels === 'function') API.drawLabels();         // automatic finding labels in the viewer
    if (['reading', 'slice', 'overview', 'along'].indexOf(API.currentTab()) >= 0) API.renderInspector();
  }
  function openScale(legendEl, info) {
    var v = viewerForLegend(legendEl), r = resultOf(v);
    if (!v || !r || !info || !info.fieldId || info.fieldId === 'slice') return;
    if (pop && pop.kind === 'scale' && pop.legend === legendEl) { closePop(); return; }
    var fieldId = info.fieldId, field = r.field(fieldId);
    var vp = legendEl && typeof legendEl.closest === 'function' ? legendEl.closest('.vp') : null;
    var p = openPop('scale', (field && (field.short_label || field.label)) || fieldId, rectOf(legendEl), placeByBar(rectOf(vp)), function (body) { buildScale(body, v, fieldId); });
    if (p) p.legend = legendEl;
  }
  function buildScale(body, v, fieldId) {
    var h = ui().h, r = resultOf(v), field = r && r.field(fieldId), s = sessionFor(v);
    if (!field || !s) { ui().fill(body, ui().note('这个视口没有结果。')); return; }
    var parts = [];
    var bsel = ui().select(BAND_STEPS.map(function (n) { return { value: String(n), label: n ? n + ' 段' : '连续' }; }), String(effBands(s, fieldId)),
      function (x) { setBands(v, fieldId, Number(x)); }, { 'aria-label': '色带分段' });
    parts.push(row('分段', bsel));
    var base = validThresholds(field.display && field.display.thresholds);
    if (base) {
      var t = effThresholds(s, fieldId) || base, above = isAbove(field), units = unitText(field.units);
      var inputs = t.map(function (x, k) {
        var inp = h('input', { type: 'number', min: '0', step: String(field.id === 'osi' ? 0.05 : k === 0 ? 0.1 : 0.5), value: String(+(+x).toPrecision(6)), 'class': 'wsd-num', 'aria-label': '阈值 ' + (k + 1) });
        inp.addEventListener('change', function () {
          var next = validThresholds(inputs.map(function (i) { return i.value; }));
          if (!next) { ui().toast('阈值要三个递增的非负数。', { kind: 'error' }); buildScale(body, v, fieldId); return; }
          setThresholds(v, fieldId, next, field); buildScale(body, v, fieldId);
        });
        return inp;
      });
      var custom = !sameThresholds(t, base);
      parts.push(row('阈值', h('span', { 'class': 'wsd-thr' }, inputs, units ? h('span', { 'class': 'wsd-unit', text: units }) : null,
        custom ? ui().iconButton('refresh', '恢复发布包阈值 ' + base.map(fmtCut).join(' / '), function () { setThresholds(v, fieldId, null, field); buildScale(body, v, fieldId); }, { cls: 'wsd-mini' }) : null),
        '只改色条上的粗线和下面的占比；概览、发现和导出的统计仍按发布包阈值。'));
      var values = null;
      try { values = r.fieldArray(fieldId, 'read'); } catch (_) { values = null; }
      if (values) {
        var fr = thresholdFractions(values, t, above).fractions, sign = [above ? '>' : '<', '>', '>'];
        parts.push(h('div', { 'class': 'wsd-fracs', title: '预测点等权占比（与经典报告同一口径）' }, t.map(function (x, k) {
          return h('span', { 'class': 'wsd-frac' }, h('span', { 'class': 'wsd-frac-k', text: sign[k] + ' ' + fmtCut(x) }), h('b', { text: ui().pct(fr[k]) }));
        })));
      }
    }
    var kind = unitKind(field, r.family);
    if (kind) {
      var usel = ui().select(unitOptions(kind).map(function (u) { return { value: u, label: u }; }), prefs().units[kind], function (u) { setUnit(kind, u); }, { 'aria-label': '单位' });
      parts.push(row('单位', usel, '只换显示：色条、探针读数和截面页一起换；导出的数据仍是 ' + (kind === 'pressure' ? 'Pa' : 'm/s') + '。'));
    }
    var def = prefs().defaults[fieldId];
    parts.push(h('div', { 'class': 'wsd-foot' },
      ui().button('设为默认', function () { setDefault(fieldId, effBands(s, fieldId), s.thresholds[fieldId] || (def && def.thresholds) || null); buildScale(body, v, fieldId); }, { cls: 'btn-sm' }),
      ui().infoTip('按字段记住分段和阈值；以后打开的结果都这样显示。'),
      h('span', { 'class': 'sec-fill' }),
      def ? ui().button('清除默认', function () { clearDefault(fieldId); buildScale(body, v, fieldId); }, { kind: 'link', cls: 'btn-sm' }) : null));
    ui().fill(body, parts);
  }
  function setDefault(fieldId, bands, thresholds) {
    var P = prefs(), e = {};
    if (bands) e.bands = bands;
    var t = validThresholds(thresholds); if (t) e.thresholds = t;
    if (Object.keys(e).length) P.defaults[fieldId] = e; else delete P.defaults[fieldId];
    savePrefs(P);
    ui().toast('已设为默认：以后打开的结果按它显示。', { kind: 'ok', ms: 2500 });
  }
  function clearDefault(fieldId) { var P = prefs(); delete P.defaults[fieldId]; savePrefs(P); refreshAll(); }

  // ---- volume: wall opacity and streamlines (from the layers menu)
  function layersButton() {
    var S = shell(); if (!S || !S.els || !S.els.toolbar) return null;
    var stack = [S.els.toolbar];
    while (stack.length) {
      var e = stack.pop();
      if (e && e.getAttribute && e.getAttribute('aria-label') === '图层、色表与背景') return e;
      var kids = (e && e.children) || [];
      for (var i = 0; i < kids.length; i++) stack.push(kids[i]);
    }
    return null;
  }
  function setVolume(patch, showLines) {
    var P = prefs(); Object.assign(P.volume, patch); savePrefs(P);
    if (showLines && API) { var L = API.state().layers; if (L && !L.streamlines) { L.streamlines = true; API.setLayers(); } }
    refreshAll();
  }
  function openVolume() { openPop('volume', '外壁与流线', rectOf(layersButton()), placeRightOf, buildVolume); }
  function buildVolume(body) {
    var h = ui().h, V = prefs().volume;
    var op = h('input', { type: 'range', min: '0', max: '0.5', step: '0.01', value: String(V.opacity), 'aria-label': '外壁不透明度' });
    var opv = h('output', { 'class': 'wsd-val', text: V.opacity.toFixed(2) });
    op.addEventListener('input', function () { opv.textContent = (+op.value).toFixed(2); setVolume({ opacity: +op.value }); });
    var thin = h('input', { type: 'checkbox', checked: V.thin, 'aria-label': '细线' });
    var wd = h('input', { type: 'range', min: '0.4', max: '3', step: '0.1', value: String(V.width), disabled: V.thin, 'aria-label': '流线粗细' });
    var wdv = h('output', { 'class': 'wsd-val' + (V.thin ? ' off' : ''), text: (+V.width).toFixed(1) + '×' });
    thin.addEventListener('change', function () { wd.disabled = thin.checked; wdv.classList.toggle('off', thin.checked); setVolume({ thin: thin.checked }, true); });
    wd.addEventListener('input', function () { wdv.textContent = (+wd.value).toFixed(1) + '×'; setVolume({ width: +wd.value }, true); });
    var dens = ui().select(DENSITY.map(function (d) { return { value: String(d.v), label: d.label }; }), String(V.density), function (x) { setVolume({ density: Number(x) }, true); }, { 'aria-label': '流线密度' });
    ui().fill(body, [
      row('外壁', h('span', { 'class': 'wsd-range' }, op, opv), '外壁不透明度；经典体场报告默认 0.10。'),
      row('流线', h('span', { 'class': 'wsd-range' }, h('label', { 'class': 'wsd-check' }, thin, h('span', { text: '细线' })), wd, wdv), '不勾细线时画成管子：半径 = 模型对角线 / 900 × 倍数（经典体场报告的流线）。'),
      row('密度', dens)]);
  }

  // ------------------------------------------------------------------ layers menu, toolbar, keys
  function toggleLayer(key) {
    var P = prefs(); P.layers[key] = !P.layers[key]; savePrefs(P);
    refreshAll();
    ensureArrays().then(function () { refreshAll(); }, function (e) { ui().toast('读取数据失败：' + (e && e.message || e), { kind: 'error' }); });
  }
  function layerItems(api) {
    API = api;
    var cur = api.cur(); if (!cur || !cur.manifest || !api.viewer()) return [];
    var m = cur.manifest, vol = m.result && m.result.family === 'volume', P = prefs();
    var items = [{ label: '输入 STL', checked: P.layers.stl, run: function () { toggleLayer('stl'); } }];
    if (!vol) {
      items.push({ label: MIN_FIELDS[cur.field] ? '最大 / 最小值标记' : '最大值标记', checked: P.layers.peaks, run: function () { toggleLayer('peaks'); } });
      if (hasStagnation(m)) { var c = stagnationCriteria(m); items.push({ label: '滞留区斜纹', checked: P.layers.stagnation, run: function () { toggleLayer('stagnation'); }, hint: 'TAWSS < ' + fmtCut(c.tawss_lt_pa) + ' 且 OSI > ' + fmtCut(c.osi_gt) }); }
    } else {
      if (velocityKey(m)) items.push({ label: '速度箭头', checked: P.layers.vectors, run: function () { toggleLayer('vectors'); }, hint: cur.field === 'speed' ? null : '速度场时显示' });
      items.push({ label: '外壁与流线…', run: openVolume });
    }
    return items;
  }
  function resetView() {
    var S = shell(); if (!S || !S.viewerA || !S.cur || !S.cur.result) return false;
    try { S.viewerA.standardView('anterior'); } catch (_) {}
    if (S.viewerB && !S.unlink) { try { S.viewerB.standardView('anterior'); } catch (_) {} }
    return true;
  }
  if (ns.icons && typeof ns.icons.register === 'function' && ns.icons.P) {
    ns.icons.register('home', [ns.icons.P('M2.25 7.25 8 2.5l5.75 4.75'), ns.icons.P('M3.75 6.25v7.25h8.5V6.25'), ns.icons.P('M6.5 13.5V9.75h3v3.75')]);
  }
  if (ns.colorbar && typeof ns.colorbar.setAdjustHandler === 'function') ns.colorbar.setAdjustHandler(openScale);

  (ns.ext = ns.ext || []).push({
    id: 'display',
    toolbar: function (api) {
      API = api;
      if (!api.viewer() || !api.cur() || !api.cur().result) return [];
      return [ui().iconButton('home', '复位视角：前视并撑满（0）', resetView)];
    },
    layers: function (api) { return layerItems(api); },
    key: function (e, api) {
      API = api;
      if (e && e.key === '0' && !e.shiftKey) return resetView();
      return false;
    },
    onResult: function (api) {
      API = api; closePop();
      var P = prefs();
      if (P.layers.vectors || P.layers.stagnation) ensureArrays().then(refreshAll, function () {});
    },
    onField: function (api) { API = api; if (pop && pop.kind === 'scale') closePop(); },
    onClose: function (api) { API = api; closePop(); }
  });

  // ------------------------------------------------------------------ compare (ws_compare.js buttons)
  // 以左为准 / 以右为准 (classic push-left / push-right): the field and the colour window go through the shell (its
  // compare branch keeps both sides on one field), bands and thresholds of that field are copied between the two
  // results' sessions; layers, units and the volume options are shared by both sides already.
  function pushSide(from) {
    var S = shell(), cur = S && S.cur, c = cur && cur.compare;
    if (!c || !API) return false;
    var fid = from === 'a' ? cur.field : c.field, win = from === 'a' ? cur.window : c.window;
    if (!fieldOf(from === 'a' ? c.manifest : cur.manifest, fid)) { ui().toast((from === 'a' ? '右' : '左') + '侧没有这个字段，不能照搬。', { kind: 'info' }); return false; }
    var ra = ridOf(cur.result), rb = ridOf(c.result), src = session(from === 'a' ? ra : rb), dst = session(from === 'a' ? rb : ra);
    if (src.bands[fid] !== undefined) dst.bands[fid] = src.bands[fid]; else delete dst.bands[fid];
    if (src.thresholds[fid]) dst.thresholds[fid] = src.thresholds[fid].slice(); else delete dst.thresholds[fid];
    API.applyField(fid, win);
    refreshAll();
    ui().toast(from === 'a' ? '右侧已按左侧显示：字段、色标窗、分段、阈值。' : '左侧已按右侧显示：字段、色标窗、分段、阈值。', { kind: 'ok', ms: 3000 });
    return true;
  }

  // Leaving the same-scale mode: the right side keeps the bands and thresholds it was showing (the left's).
  function keepShared() {
    var S = shell(), cur = S && S.cur, c = cur && cur.compare;
    if (!c || !cur.result || !c.result) return false;
    var src = session(ridOf(cur.result)), dst = session(ridOf(c.result)), fid = c.field;
    if (src.bands[fid] !== undefined) dst.bands[fid] = src.bands[fid];
    if (src.thresholds[fid]) dst.thresholds[fid] = src.thresholds[fid].slice();
    return true;
  }

  return {
    provider: provider, prefs: prefs, savePrefs: savePrefs, sanitizePrefs: sanitizePrefs, session: session, reset: reset, displayUnit: displayUnit, toDisplay: toDisplay,
    validThresholds: validThresholds, thresholdFractions: thresholdFractions, peakMarkers: peakMarkers, stagnationCriteria: stagnationCriteria,
    hasStagnation: hasStagnation, unitFactor: unitFactor, unitKind: unitKind, unitOptions: unitOptions, sanitizeSessionState: sanitizeSessionState,
    openScale: openScale, openVolume: openVolume, closePop: closePop, isOpen: function () { return pop ? pop.kind : null; },
    setBands: setBands, setThresholds: setThresholds, setUnit: setUnit, setDefault: setDefault, clearDefault: clearDefault, toggleLayer: toggleLayer,
    resetView: resetView, pushSide: pushSide, keepShared: keepShared, refreshAll: refreshAll, BAND_STEPS: BAND_STEPS.slice(), UNITS: UNITS
  };
});
