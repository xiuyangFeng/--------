/* WSS workspace v2 — Lane E (S6e): display options (PHASE2_LANES.md §5 E).
 * Colour scale: bands and observation thresholds per field, 「设为默认」 per field; layers: input STL, the field maximum
 * (TAWSS also its minimum) as a persistent marker, the stagnation hatch, a reset-view button (key 0); volume: wall
 * opacity, streamline density and width, velocity arrows on the interior points, display units (Pa / mmHg,
 * m/s / cm/s) for the colour bar, the probe card and the section page; compare: 以左为准 / 以右为准 and the same-scale
 * mode sharing bands and thresholds.  Everything here changes what is drawn, never a number: thresholds only move the
 * lines on the colour bar (and the point-equal fractions next to them, the classic definition), units multiply by the
 * classic factors (VolumeViewerCore.convertUnit), statistics keep the release's definitions.
 * The viewer kernel asks this module through ns.viewer.displayProvider (core_viewer.js, lane E block); the shell
 * through its extension registry (layers menu, toolbar, keys, result hooks); the colour bar through its adjust button.
 * Phase 3 lane 4 (PHASE3_LANES.md §3 第 4 路) adds: WSS in dyn/cm² (classic CORE.UNITS, × 10), the velocity log scale
 * (classic floor max(lower end, upper end / 200, 0.001 m/s)), a typed fixed upper limit remembered per field, iso-lines at
 * the three thresholds, the highest-x % highlight, the Z cut of wall results, the trust legend, the hover-readout switch
 * (P) and the classic presets / default settings stored in the server preferences (/api/preferences), read in on request. */
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
  var UNITS = { pressure: { 'Pa': 1, 'mmHg': 1 / 133.322 }, velocity: { 'm/s': 1, 'cm/s': 100 },   // classic volume_viewer UNITS
    wss: { 'Pa': 1, 'dyn/cm²': 10 } };                                                          // classic wall report CORE.UNITS (P3 lane 4)
  var DENSITY = [{ v: 1, label: '全部' }, { v: 2, label: '一半' }, { v: 3, label: '三分之一' }, { v: 5, label: '五分之一' }];
  var MIN_FIELDS = { tawss: true };                                     // classic: TAWSS also marks its lowest point
  var ABOVE = { osi: true, rrt: true, ecap: true };
  var DEFAULTS = { schema: 'wssv2.display/1', defaults: {}, units: { pressure: 'Pa', velocity: 'm/s', wss: 'Pa' },
    layers: { stl: false, peaks: true, stagnation: false, vectors: false }, volume: { opacity: 0.1, density: 1, width: 1, thin: true } };
  var ID_RE = /^[A-Za-z0-9_.-]{1,40}$/;
  // Phase 3 lane 4 options, kept apart from the lane E keys (their stored shape is unchanged).
  var OPTS = { contours: false, top: false, topPct: 1, hover: true, speedLog: false, fixed: {}, imported: {} };
  var TOP_PCT = { min: 0.5, max: 10, step: 0.5 };                     // classic highlight-pct slider
  function cleanPct(x) { x = Number(x); return Number.isFinite(x) ? Math.max(TOP_PCT.min, Math.min(TOP_PCT.max, Math.round(x / TOP_PCT.step) * TOP_PCT.step)) : 1; }
  function sanitizeOpts(o) {
    var p = JSON.parse(JSON.stringify(OPTS));
    if (!o || typeof o !== 'object') return p;
    ['contours', 'top', 'hover', 'speedLog'].forEach(function (k) { if (typeof o[k] === 'boolean') p[k] = o[k]; });
    if (o.topPct !== undefined && o.topPct !== null && o.topPct !== '') p.topPct = cleanPct(o.topPct);
    Object.keys(o.fixed || {}).forEach(function (id) { var v = Number(o.fixed[id]); if (ID_RE.test(id) && Number.isFinite(v) && v > 0) p.fixed[id] = v; });
    Object.keys(o.imported || {}).forEach(function (k) { if ((k === 'wall' || k === 'volume') && typeof o.imported[k] === 'string') p.imported[k] = o.imported[k].slice(0, 200); });
    return p;
  }

  // ------------------------------------------------------------------ pure helpers (Node-tested)
  // Classic validThresholds: three increasing non-negative numbers.
  function validThresholds(t) {
    t = (Array.isArray(t) ? t : []).map(function (x) { return x === '' || x === null || x === undefined ? NaN : Number(x); });
    return t.length === 3 && t.every(Number.isFinite) && t[0] >= 0 && t[0] < t[1] && t[1] < t[2] ? t : null;
  }
  function sameThresholds(a, b) {
    return Array.isArray(a) && Array.isArray(b) && a.length === 3 && b.length === 3 && a.every(function (x, k) { return Math.abs(x - b[k]) <= 1e-9 * Math.max(1, Math.abs(b[k])); });
  }
  // Factor of a display unit (value × factor); the classic core's own conversion when it is loaded (pressure, velocity).
  function unitFactor(kind, unit) {
    var table = UNITS[kind] || {};
    if (!Object.prototype.hasOwnProperty.call(table, unit)) return 1;
    var VC = root.VolumeViewerCore;
    if (kind !== 'wss' && VC && typeof VC.convertUnit === 'function') { var f = VC.convertUnit(1, kind, unit); if (Number.isFinite(f) && f > 0) return f; }
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
  // Phase 3 lane 4: the display kind of any number — the volume kinds above, and wall stresses in Pa (WSS, TAWSS; classic
  // isPa: RRT / ECAP in 1/Pa and OSI stay as they are).  A number without a family (a finding) counts as a wall number
  // unless the caller says volume (ws_overview passes 'volume' for the volume finding kinds).
  function displayKind(field, family) {
    var k = unitKind(field, family);
    if (k) return k;
    return family !== 'volume' && field && field.units === 'Pa' ? 'wss' : null;
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
    p.opts = sanitizeOpts(raw && typeof raw === 'object' ? raw.opts : null);   // P3 lane 4
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
    if (UNITS.wss[u.wss]) p.units.wss = u.wss;                          // P3 lane 4
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
    var c = Number(d.clip);                                             // P3 lane 4: the wall Z cut, 0 ≤ t < 1 (1 = no cut)
    if (d.clip !== null && d.clip !== undefined && d.clip !== '' && Number.isFinite(c) && c >= 0 && c < 1) out.clip = +c.toFixed(4);
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
  function reset() { prefsCache = null; sessions = {}; closePop(); serverPrefs = { doc: null, error: null, pending: null }; lastField = null; }

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
  function unitOf(kind) { return prefs().units[kind]; }
  var provider = {
    bands: function (v, fieldId) { var s = sessionFor(v); return s ? effBands(s, fieldId) : null; },
    thresholds: function (v, fieldId) { var s = sessionFor(v); return s ? effThresholds(s, fieldId) : null; },
    units: function (v, field) {
      if (!sideOf(v)) return null;
      var r = resultOf(v), kind = displayKind(field, r && r.family);
      if (!kind) return null;
      var u = unitOf(kind), f = unitFactor(kind, u);
      return f === 1 ? null : { units: u, factor: f };
    },
    // P3 lane 4 (wall results; the viewer's P3 lane 4 block): iso-lines, highest-x % highlight, Z cut
    contours: function (v) { return Boolean(sideOf(v) && prefs().opts.contours); },
    top: function (v) { var o = prefs().opts; return sideOf(v) && o.top ? o.topPct : null; },
    clip: function (v) { var s = sessionFor(v, true); return s && s.clip !== undefined ? s.clip : null; },
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
        var q = toDisplay(p.value, f && f.units, 'wall');
        p.text = (p.kind === 'min' ? '最小 ' : '最大 ') + fmtNum(q.value) + (unitText(q.units) ? ' ' + unitText(q.units) : '');
        return p;
      });
    },
    state: function (v) {
      var s = sessionFor(v, true);
      if (!s || !(Object.keys(s.bands).length || Object.keys(s.thresholds).length || s.clip !== undefined)) return null;
      var out = { v: 1, bands: Object.assign({}, s.bands), thresholds: Object.assign({}, s.thresholds) };
      if (s.clip !== undefined) out.clip = s.clip;
      return out;
    },
    restore: function (v, d) {
      var s = sessionFor(v, true); if (!s) return;
      var c = sanitizeSessionState(d);
      s.bands = c.bands; s.thresholds = c.thresholds;
      if (c.clip !== undefined) s.clip = c.clip; else delete s.clip;
    }
  };
  if (ns.viewer && typeof ns.viewer === 'object') ns.viewer.displayProvider = provider;

  // Display unit of a physical quantity ('pressure' | 'velocity' | 'wss') for the section page and the probe card.
  function displayUnit(kind) {
    var u = UNITS[kind] ? unitOf(kind) : null;
    if (!UNITS[kind] || !u) return null;
    return { units: u, factor: unitFactor(kind, u) };
  }
  // A number in its display unit, for the overview, the lens, the along-vessel curves and the colour window labels:
  // {value, units}.  Volume pressures and speeds follow Pa / mmHg, m/s / cm/s; wall stresses in Pa follow Pa / dyn/cm²
  // (P3 lane 4, default Pa); every other quantity comes back unchanged.
  function toDisplay(value, units, family) {
    var kind = displayKind({ units: units }, family), d = kind ? displayUnit(kind) : null;
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
  var POP_W = 296, STRIP = 64;
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
    rerender(['reading', 'slice', 'overview', 'along', 'region']);
  }
  // After a display choice that shows up in text: the colour-window labels (toolbar), the automatic finding labels in the
  // viewer and the inspector pages that print numbers.
  function rerender(tabs) {
    if (!API) return;
    if (typeof API.renderToolbar === 'function') API.renderToolbar();
    if (typeof API.drawLabels === 'function') API.drawLabels();
    if (typeof API.currentTab === 'function' && tabs.indexOf(API.currentTab()) >= 0) API.renderInspector();
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
    var kind = displayKind(field, r.family), k = kind ? unitFactor(kind, unitOf(kind)) : 1, shownUnits = unitText(kind ? unitOf(kind) : field.units);
    var bsel = ui().select(BAND_STEPS.map(function (n) { return { value: String(n), label: n ? n + ' 段' : '连续' }; }), String(effBands(s, fieldId)),
      function (x) { setBands(v, fieldId, Number(x)); }, { 'aria-label': '色带分段' });
    parts.push(row('分段', bsel));
    var base = validThresholds(field.display && field.display.thresholds);
    if (base) {
      // typed and shown in the display unit (classic initThresholdInputs), kept in the stored unit
      var t = effThresholds(s, fieldId) || base, above = isAbove(field);
      var inputs = t.map(function (x, j) {
        var inp = h('input', { type: 'number', min: '0', step: String((field.id === 'osi' ? 0.05 : j === 0 ? 0.1 : 0.5) * k), value: String(+(+x * k).toPrecision(6)), 'class': 'wsd-num', 'aria-label': '阈值 ' + (j + 1) });
        inp.addEventListener('change', function () {
          var next = validThresholds(inputs.map(function (i) { return i.value === '' ? '' : Number(i.value) / k; }));
          if (!next) { ui().toast('阈值要三个递增的非负数。', { kind: 'error' }); buildScale(body, v, fieldId); return; }
          next = next.map(function (x0) { return +x0.toPrecision(10); });
          setThresholds(v, fieldId, next, field); buildScale(body, v, fieldId);
        });
        return inp;
      });
      var custom = !sameThresholds(t, base);
      parts.push(row('阈值', h('span', { 'class': 'wsd-thr' }, inputs, shownUnits ? h('span', { 'class': 'wsd-unit', text: shownUnits }) : null,
        custom ? ui().iconButton('refresh', '恢复发布包阈值 ' + base.map(function (x) { return fmtCut(x * k); }).join(' / '), function () { setThresholds(v, fieldId, null, field); buildScale(body, v, fieldId); }, { cls: 'wsd-mini' }) : null),
        '只改色条上的粗线、等值线和下面的占比；概览、发现和导出的统计仍按发布包阈值。'));
      var values = null;
      try { values = r.fieldArray(fieldId, 'read'); } catch (_) { values = null; }
      if (values) {
        var fr = thresholdFractions(values, t, above).fractions, sign = [above ? '>' : '<', '>', '>'];
        parts.push(h('div', { 'class': 'wsd-fracs', title: '预测点等权占比（与经典报告同一口径）' }, t.map(function (x, j) {
          return h('span', { 'class': 'wsd-frac' }, h('span', { 'class': 'wsd-frac-k', text: sign[j] + ' ' + fmtCut(x * k) }), h('b', { text: ui().pct(fr[j]) }));
        })));
      }
    }
    if (kind) {
      var usel = ui().select(unitOptions(kind).map(function (u) { return { value: u, label: u }; }), unitOf(kind), function (u) { setUnit(kind, u); buildScale(body, v, fieldId); }, { 'aria-label': '单位' });
      parts.push(row('单位', usel, kind === 'wss' ? '只换显示：色条、概览、发现、探针和悬停读数一起换；1 Pa = 10 dyn/cm²；导出的数据仍是 Pa。'
        : '只换显示：色条、探针读数和截面页一起换；导出的数据仍是 ' + (kind === 'pressure' ? 'Pa' : 'm/s') + '。'));
    }
    if (r.family === 'volume' && fieldId === 'speed') {   // classic 速度对数色标 (V7)
      var lg = h('input', { type: 'checkbox', checked: prefs().opts.speedLog, 'aria-label': '对数色标' });
      lg.addEventListener('change', function () { setOpts({ speedLog: lg.checked }); rerender(['slice']); });
      parts.push(row('对数', h('label', { 'class': 'wsd-check' }, lg, h('span', { text: '速度对数色标' })),
        '下限 = max(范围下限, 上限 / 200, 0.001 m/s)，与经典体场报告相同；截面的速度色标也跟着换。只改颜色。'));
    }
    var fx = fixedRow(v, r, field, k, shownUnits);
    if (fx) parts.push(fx);
    var def = prefs().defaults[fieldId];
    parts.push(h('div', { 'class': 'wsd-foot' },
      ui().button('设为默认', function () { setDefault(fieldId, effBands(s, fieldId), s.thresholds[fieldId] || (def && def.thresholds) || null); buildScale(body, v, fieldId); }, { cls: 'btn-sm' }),
      ui().infoTip('按字段记住分段和阈值；以后打开的结果都这样显示。'),
      h('span', { 'class': 'sec-fill' }),
      def ? ui().button('清除默认', function () { clearDefault(fieldId); buildScale(body, v, fieldId); }, { kind: 'link', cls: 'btn-sm' }) : null));
    ui().fill(body, parts);
  }
  // Classic 固定上限 (W6): the colour range 0 … a typed upper limit (display unit in, stored unit kept) for fields without
  // negative values, on the main viewport outside a comparison.  「所有结果」 remembers it for this field: results opened
  // later start on it (classic 固定上限 · 跨报告保留).  Clearing the box goes back to 本例自适应.
  function fixedRow(v, r, field, k, shownUnits) {
    var S = shell(), cur = S && S.cur;
    if (!API || sideOf(v) !== 'a' || !cur || cur.compare || cur.field !== field.id) return null;
    var st = field.statistics || {}, dmin = Number(st.min);
    if (Number.isFinite(dmin) && dmin < 0) return null;
    var h = ui().h, win = cur.window, now = win && typeof win === 'object' && Array.isArray(win.range) && win.range[0] === 0 ? win.range[1] : null;
    var remembered = prefs().opts.fixed[field.id];
    var cb = v.colorbarInfo ? v.colorbarInfo() : null, hi = cb && cb.scale ? cb.scale.range[1] : null;
    var inp = h('input', { type: 'number', min: '0', step: 'any', 'class': 'wsd-num wsd-num-wide', value: now !== null ? String(+(now * k).toPrecision(6)) : '',
      placeholder: hi !== null ? String(+(hi * k).toPrecision(3)) : '', 'aria-label': '固定上限' });
    var keep = h('input', { type: 'checkbox', checked: remembered !== undefined, 'aria-label': '所有结果都用这个上限' });
    function apply() {
      var raw = inp.value === '' ? null : Number(inp.value) / k;
      var o = prefs().opts, fixed = Object.assign({}, o.fixed);
      if (raw === null) { delete fixed[field.id]; setOpts({ fixed: fixed }); API.applyField(field.id, 'adaptive'); return; }
      if (!(Number.isFinite(raw) && raw > 0)) { ui().toast('上限要大于 0。', { kind: 'error' }); return; }
      raw = +raw.toPrecision(10);
      if (keep.checked) fixed[field.id] = raw; else delete fixed[field.id];
      setOpts({ fixed: fixed });
      API.applyField(field.id, { range: [0, raw] });
    }
    inp.addEventListener('change', apply);
    keep.addEventListener('change', function () {
      var o = prefs().opts, fixed = Object.assign({}, o.fixed), raw = inp.value === '' ? null : Number(inp.value) / k;
      if (keep.checked && raw > 0) fixed[field.id] = +raw.toPrecision(10); else delete fixed[field.id];
      setOpts({ fixed: fixed });
    });
    return row('上限', h('span', { 'class': 'wsd-thr' }, inp, shownUnits ? h('span', { 'class': 'wsd-unit', text: shownUnits }) : null,
        h('label', { 'class': 'wsd-check' }, keep, h('span', { text: '所有结果' }))),
      '色标固定为 0 至这个数（经典「固定上限」）；空着就是本例自适应。勾「所有结果」后，以后打开的结果这个字段也从它开始。');
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

  // ------------------------------------------------------------------ phase 3 lane 4: options, panels, legend, classic settings
  // mode: undefined → refresh every viewer (colours, bars, overlays); 'p3' → only the iso-lines / highlight / cut; 'quiet' → nothing.
  function setOpts(patch, mode) {
    var P = prefs(); P.opts = sanitizeOpts(Object.assign({}, P.opts, patch || {})); savePrefs(P);
    if (mode === 'p3') viewers().forEach(function (v) { if (typeof v.refreshP3 === 'function') { try { v.refreshP3(); } catch (e) { if (root.console) root.console.error(e); } } });
    else if (mode !== 'quiet') refreshAll();
    return P.opts;
  }
  // Classic 速度对数色标 through the colour-scale resolver: the adaptive window of the speed field opens on a log scale.
  if (ns.colormap && typeof ns.colormap.setLogHint === 'function') ns.colormap.setLogHint(function (field) {
    return field && field.id === 'speed' && prefs().opts.speedLog ? true : null;
  });
  function viewerA() { var S = shell(); return S ? S.viewerA : null; }
  function fieldOfViewer(v) { var r = resultOf(v), fid = v && typeof v.field === 'function' ? v.field() : null; return r && fid ? r.field(fid) : null; }

  // ---- highest-x % highlight (classic 高亮最高 x%, W16)
  function openTop() { openPop('top', '高亮最高值', rectOf(layersButton()), placeRightOf, buildTop); }
  function topText(v) {
    var d = v && typeof v.p3Display === 'function' ? v.p3Display().top : null, f = fieldOfViewer(v);
    if (!d) return prefs().opts.top ? '这个字段没有预测点读数。' : '';
    var q = toDisplay(d.threshold, f && f.units, 'wall');
    return '≥ ' + fmtNum(q.value) + (unitText(q.units) ? ' ' + unitText(q.units) : '') + ' · ' + d.count + ' 个预测点';
  }
  function buildTop(body) {
    var h = ui().h, o = prefs().opts, v = viewerA();
    var on = h('input', { type: 'checkbox', checked: o.top, 'aria-label': '显示高亮' });
    var sl = h('input', { type: 'range', min: String(TOP_PCT.min), max: String(TOP_PCT.max), step: String(TOP_PCT.step), value: String(o.topPct), 'aria-label': '高亮比例' });
    var val = h('output', { 'class': 'wsd-val', text: fmtCut(o.topPct) + '%' });
    var info = h('div', { 'class': 'wsd-note', text: topText(v) });
    on.addEventListener('change', function () { setOpts({ top: on.checked }, 'p3'); info.textContent = topText(v); });
    sl.addEventListener('input', function () { val.textContent = fmtCut(+sl.value) + '%'; on.checked = true; setOpts({ topPct: +sl.value, top: true }, 'p3'); info.textContent = topText(v); });
    ui().fill(body, [row('显示', h('label', { 'class': 'wsd-check' }, on, h('span', { text: '品红点' }))),
      row('比例', h('span', { 'class': 'wsd-range' }, sl, val), '当前字段预测点值排在最高 x% 的点（与经典报告同一算法：排序后取第 ⌈(1 − x%)·n⌉ 个值为门槛）。隐藏的分支不画。'), info]);
  }

  // ---- Z cut of wall results (classic 剖切高度, W17): per result, travels with the view state
  function openClip() { openPop('clip', '剖切', rectOf(layersButton()), placeRightOf, buildClip); }
  function clipText(v) {
    var z = v && typeof v.p3Display === 'function' ? v.p3Display().clipZ : null;
    return z === null || z === undefined ? '不剖切' : 'Z ≤ ' + Math.round(z) + ' mm';
  }
  function setClip(v, t) {
    var s = sessionFor(v, true); if (!s) return;
    if (t === null || !(t < 1)) delete s.clip; else s.clip = +Math.max(0, t).toFixed(4);
    viewers().forEach(function (w) { if (resultOf(w) === resultOf(v) && typeof w.refreshClip === 'function') { try { w.refreshClip(); } catch (_) {} } });
    saveSoon();
  }
  function buildClip(body) {
    var h = ui().h, v = viewerA(), s = sessionFor(v, true), t = s && s.clip !== undefined ? s.clip : 1;
    var sl = h('input', { type: 'range', min: '0', max: '100', step: '1', value: String(Math.round(t * 100)), 'aria-label': '剖切高度' });
    var val = h('output', { 'class': 'wsd-val wsd-val-wide', text: clipText(v) });
    sl.addEventListener('input', function () { setClip(v, +sl.value >= 100 ? null : +sl.value / 100); val.textContent = clipText(v); });
    ui().fill(body, [row('高度', h('span', { 'class': 'wsd-range' }, sl, val), '沿 STL 的 Z 轴只留下切面以下的部分（经典「剖切高度」）；悬停、探针和测量只读留下的部分。只改显示。'),
      h('div', { 'class': 'wsd-foot' }, h('span', { 'class': 'sec-fill' }), ui().button('不剖切', function () { sl.value = '100'; setClip(v, null); val.textContent = clipText(v); }, { kind: 'link', cls: 'btn-sm' }))]);
  }

  // ---- trust legend (classic renderTrustLegend, W11 / V12): shown in the viewport while 可信度标记 is on
  var TRUST_LABELS = { 1: '插值无支撑', 2: '表面粗糙', 4: '几何越界', 8: '采样支撑弱', 16: '邻近切口' };
  function trustRows(m) {
    var t = (m && m.analysis && m.analysis.trust) || {}, src = Array.isArray(t.sources) ? t.sources : [];
    var bits = t.bits && typeof t.bits === 'object' ? t.bits : ((m && m.mapping && m.mapping.trust_bits) || {}), fr = t.fractions || {};
    return Object.keys(bits).map(Number).filter(Number.isFinite).sort(function (a, b) { return a - b; }).map(function (bit) {
      var name = String(bits[bit] || ''), s = src.filter(function (x) { return Number(x && x.bit) === bit; })[0] || {};
      var f = Number(fr[name]);
      return { bit: bit, name: name, label: s.label || TRUST_LABELS[bit] || name || ('位 ' + bit), rule: s.rule || '', fraction: fr[name] !== undefined && fr[name] !== null && Number.isFinite(f) ? f : null,
        support: s.support || (bit >= 8 ? 'interior_points' : 'vertices') };
    });
  }
  var legends = {};
  function updateLegend(side) {
    var S = shell(); if (!S || !S.els) return;
    var vp = side === 'b' ? S.els.vpB : S.els.vpA, v = side === 'b' ? S.viewerB : S.viewerA, host = vp && vp.legend && vp.legend.parentNode, el = legends[side];
    var on = false, r = resultOf(v);
    try { on = Boolean(v && r && v.getState().layers.trust); } catch (_) { on = false; }
    var rows = on ? trustRows(r.manifest) : [];
    if (!rows.length || !host) { if (el) el.hidden = true; return; }
    var h = ui().h, vol = r.family === 'volume', t = (r.manifest.analysis && r.manifest.analysis.trust) || {};
    if (!el || el.parentNode !== host) { if (el && el.parentNode) el.parentNode.removeChild(el); el = h('div', { 'class': 'wsd-trust', role: 'note', 'aria-label': '可信度标记图例' }); host.appendChild(el); legends[side] = el; }
    el.hidden = false;
    var rules = rows.map(function (x) { return x.label + '：' + (x.rule || '—'); }).concat(t.note ? [t.note] : []).join('\n');
    ui().fill(el, [h('div', { 'class': 'wsd-trust-head' }, h('span', { text: '可信度标记' }), ui().infoTip(rules)),
      rows.map(function (x) {
        var interior = vol && x.support === 'interior_points';
        return h('div', { 'class': 'wsd-trust-row', title: x.rule || '' }, h('i', { 'class': 'wsd-sw ' + (interior ? 'wsd-sw-grey' : 'wsd-sw-hatch'), 'aria-hidden': 'true' }),
          h('span', { 'class': 'wsd-trust-k', text: x.label }), h('b', { text: x.fraction === null ? '—' : (x.fraction * 100).toFixed(1) + '%' }));
      })]);
  }
  function updateLegends() { updateLegend('a'); updateLegend('b'); }
  function wireViewer(v) {
    if (!v || v.__wsdP3 || typeof v.on !== 'function') return;
    v.__wsdP3 = true;
    v.on('change', function (e) { if (e && (e.what === 'layers' || e.what === 'result')) updateLegend(sideOf(v) || 'a'); });
  }

  // ---- reproduction links (lane 5's x.<ext id>): {v: 1, contours, top, topPct, speedLog, clip}; an old or partial state only
  // changes what it names, anything malformed is ignored.
  function linkState(api) {
    var cur = api && api.cur && api.cur(), v = api && api.viewer && api.viewer();
    if (!cur || !cur.manifest) return undefined;
    var o = prefs().opts, s = v ? sessionFor(v, true) : null, out = { v: 1, contours: o.contours, top: o.top, topPct: o.topPct, speedLog: o.speedLog };
    out.clip = s && s.clip !== undefined ? s.clip : null;
    return out;
  }
  function applyLinkState(state, api) {
    if (!state || typeof state !== 'object' || Array.isArray(state)) return false;
    var patch = {};
    ['contours', 'top', 'speedLog'].forEach(function (k) { if (typeof state[k] === 'boolean') patch[k] = state[k]; });
    if (state.topPct !== undefined && state.topPct !== null && Number.isFinite(Number(state.topPct))) patch.topPct = cleanPct(state.topPct);
    var v = api && api.viewer && api.viewer(), clipSet = false;
    if (v && 'clip' in state) {
      var c = state.clip === null ? null : Number(state.clip);
      if (c === null || (Number.isFinite(c) && c >= 0)) { var sess = sessionFor(v, true); if (sess) { if (c === null || c >= 1) delete sess.clip; else sess.clip = +c.toFixed(4); clipSet = true; } }
    }
    if (!Object.keys(patch).length && !clipSet) return false;
    setOpts(patch);                                     // refreshes every viewer (overlays, cut, scales)
    rerender(['slice', 'overview']);
    return true;
  }

  // ---- remembered fixed upper limit: a field opened on 本例自适应 starts on the remembered range
  var lastField = null;
  function applyRemembered(api, opening) {
    var cur = api && api.cur && api.cur();
    if (!cur || !cur.manifest || cur.compare) { lastField = cur ? cur.field : null; return; }
    var fid = cur.field, switched = opening || fid !== lastField, max = prefs().opts.fixed[fid];
    lastField = fid;
    if (!switched || !(max > 0) || cur.window !== 'adaptive') return;
    Promise.resolve().then(function () {
      var c2 = api.cur();
      if (c2 === cur && c2.field === fid && c2.window === 'adaptive' && !c2.compare) api.applyField(fid, { range: [0, max] });
    });
  }

  // ---- classic presets and default settings stored in the server preferences (W22, W23, V44)
  // The classic reports wrote /api/preferences: presets.wall / presets.volume = [{name, state (wss-deploy.view/v1), created_at}]
  // and report_defaults.wall = {colormap, bands, units, thresholds_pa, opacity, log, lang}, report_defaults.volume =
  // {colormap, bands, pressure_units, speed_units, opacity, lang}.  They are read here and applied on request.
  var serverPrefs = { doc: null, error: null, pending: null };
  function loadServerPrefs(force) {
    if (!API || typeof API.api !== 'function' || !API.api() || (typeof API.offline === 'function' && API.offline())) return Promise.resolve(null);
    if (serverPrefs.pending) return serverPrefs.pending;
    if (serverPrefs.doc && !force) return Promise.resolve(serverPrefs.doc);
    serverPrefs.pending = Promise.resolve(API.api().request('/api/preferences')).then(function (j) {
      serverPrefs.pending = null; serverPrefs.error = null;
      serverPrefs.doc = j && j.preferences && typeof j.preferences === 'object' ? j.preferences : {};
      return serverPrefs.doc;
    }, function (e) { serverPrefs.pending = null; serverPrefs.error = e; throw e; });
    return serverPrefs.pending;
  }
  var CMAPS = ['rainbow', 'viridis', 'turbo', 'bwr'];
  var WALL_IDS = ['wss', 'tawss', 'osi', 'rrt', 'ecap'], VOLUME_IDS = ['speed', 'pressure', 'wall_pressure'];
  function familyIds(fam) {
    var cur = API && API.cur && API.cur(), m = cur && cur.manifest;
    var ids = m && m.result && m.result.family === fam ? (m.fields || []).map(function (f) { return f && f.id; }).filter(Boolean) : [];
    return ids.length ? ids : (fam === 'volume' ? VOLUME_IDS : WALL_IDS).slice();
  }
  // What a classic default-settings object would change here, as {patch: fn(), text: [..], skipped: [..]}.
  function defaultsPlan(fam, d) {
    var text = [], skipped = [], jobs = [];
    if (!d || typeof d !== 'object') return { text: text, skipped: skipped, run: function () {} };
    if (CMAPS.indexOf(d.colormap) >= 0) { text.push('色表 ' + (ns.colormap && ns.colormap.label ? ns.colormap.label(d.colormap) : d.colormap)); jobs.push(function () { if (API && API.store) API.store().setPrefs({ cmap: d.colormap }); }); }
    var b = Number(d.bands);
    if (d.bands !== undefined && d.bands !== null && BAND_STEPS.indexOf(b) >= 0) {
      text.push(b ? b + ' 段' : '连续');
      jobs.push(function (P) { familyIds(fam).forEach(function (id) { if (!ID_RE.test(id)) return; var e = Object.assign({}, P.defaults[id] || {}); if (b) e.bands = b; else delete e.bands; if (Object.keys(e).length) P.defaults[id] = e; else delete P.defaults[id]; }); });
    }
    if (fam === 'wall') {
      if (d.units === 'dyn' || d.units === 'Pa') { var wu = d.units === 'dyn' ? 'dyn/cm²' : 'Pa'; text.push('WSS 单位 ' + wu); jobs.push(function (P) { P.units.wss = wu; }); }
      var t = validThresholds(d.thresholds_pa);
      if (t) { text.push('WSS 阈值 ' + t.map(fmtCut).join(' / ') + ' Pa'); jobs.push(function (P) { P.defaults.wss = Object.assign({}, P.defaults.wss || {}, { thresholds: t }); }); }
      if (d.opacity !== undefined && d.opacity !== null && Number(d.opacity) !== 1) skipped.push('壁面透明度');
      if (d.log === true) skipped.push('对数色标（新工作区的 WSS、TAWSS 本来就按对数显示）');
    } else {
      if (UNITS.pressure[d.pressure_units]) { text.push('压力 ' + d.pressure_units); jobs.push(function (P) { P.units.pressure = d.pressure_units; }); }
      if (UNITS.velocity[d.speed_units]) { text.push('速度 ' + d.speed_units); jobs.push(function (P) { P.units.velocity = d.speed_units; }); }
      var op = Number(d.opacity);
      if (d.opacity !== undefined && d.opacity !== null && d.opacity !== '' && Number.isFinite(op) && op >= 0 && op <= 0.5) { text.push('外壁 ' + op.toFixed(2)); jobs.push(function (P) { P.volume.opacity = op; }); }
    }
    if (d.lang === 'en') skipped.push('英文界面');
    return { text: text, skipped: skipped, run: function () {
      var P = prefs(); jobs.forEach(function (fn) { fn(P); }); savePrefs(P);
      var cur = API && API.cur && API.cur();
      if (cur && cur.field) API.applyField(cur.field, cur.window);
      refreshAll(); rerender(['reading', 'slice', 'overview', 'along', 'region']);
    } };
  }
  function signature(d) { try { return JSON.stringify(d).slice(0, 200); } catch (_) { return ''; } }
  function applyDefaults(fam, d, api) {
    if (api) API = api;
    var plan = defaultsPlan(fam, d);
    if (!plan.text.length) { ui().toast('服务端的默认口径里没有新工作区能用的设置。', { kind: 'info' }); return false; }
    plan.run();
    var imp = Object.assign({}, prefs().opts.imported); imp[fam] = signature(d); setOpts({ imported: imp }, 'quiet');
    ui().toast('已套用：' + plan.text.join('、') + '。' + (plan.skipped.length ? '没带：' + plan.skipped.join('、') + '。' : ''), { kind: 'ok', ms: 5000 });
    return true;
  }
  // A classic view state (wss-deploy.view/v1) on the current result.  The camera is taken only from a preset saved on this
  // same result; everything the workspace cannot show is listed in the notice.
  function applyPreset(p, api) {
    if (api) API = api;
    var cur = API && API.cur && API.cur(), v = viewerA(), s = p && p.state;
    if (!cur || !cur.manifest || !v || !s || typeof s !== 'object') return false;
    var m = cur.manifest, fam = m.result && m.result.family, done = [], skipped = [], P = prefs(), sess = sessionFor(v, true);
    if (s.family && s.family !== fam) { ui().toast('这是' + (s.family === 'volume' ? '体场' : '壁面') + '报告的预设，这份结果用不了。', { kind: 'info' }); return false; }
    var has = function (id) { return (m.fields || []).some(function (f) { return f && f.id === id; }); };
    var fid = cur.field, want = null;
    if (fam === 'volume') want = s.mode === 'wall' ? 'wall_pressure' : s.field === 'velocity' ? 'speed' : s.field === 'pressure' ? 'pressure' : null;
    else want = typeof s.field === 'string' ? s.field : null;
    if (want && has(want)) fid = want; else if (want) skipped.push('字段 ' + want);
    if (CMAPS.indexOf(s.colormap) >= 0 && API.store) { API.store().setPrefs({ cmap: s.colormap }); done.push('色表'); }
    var b = Number(s.bands);
    if (s.bands !== undefined && s.bands !== null && BAND_STEPS.indexOf(b) >= 0 && sess) { sess.bands[fid] = b; done.push('分段'); }
    if (fam === 'wall') {
      if (s.units === 'dyn' || s.units === 'Pa') { P.units.wss = s.units === 'dyn' ? 'dyn/cm²' : 'Pa'; done.push('单位'); }
      var t = validThresholds(s.thresholds_pa);
      if (t && sess && has('wss')) sess.thresholds.wss = t;
      Object.keys(s.field_thresholds || {}).forEach(function (id) { var tv = validThresholds(s.field_thresholds[id]); if (tv && sess && id !== 'wss' && has(id)) sess.thresholds[id] = tv; });
      if (t || Object.keys(s.field_thresholds || {}).length) done.push('阈值');
      var ov = s.overlay || {};
      if (typeof ov.contours === 'boolean') P.opts.contours = ov.contours;
      if (typeof ov.stagnation === 'boolean') P.layers.stagnation = ov.stagnation && hasStagnation(m);
      var hl = s.highlight || {};
      if (typeof hl.top === 'boolean') { P.opts.top = hl.top; if (Number.isFinite(+hl.top_pct)) P.opts.topPct = cleanPct(hl.top_pct); }
      var clip = s.slice && Number(s.slice.clip);
      if (sess && Number.isFinite(clip)) { if (clip < 1) sess.clip = +Math.max(0, clip).toFixed(4); else delete sess.clip; }
      if (hl.branch !== undefined && Number(hl.branch) >= 0) skipped.push('淡化分支');
      if (hl.feature && hl.feature !== 'wss') skipped.push('点云特征着色');
    } else {
      if (UNITS.pressure[s.pressure_units]) P.units.pressure = s.pressure_units;
      if (UNITS.velocity[s.speed_units]) P.units.velocity = s.speed_units;
      if (typeof s.log === 'boolean') P.opts.speedLog = s.log;
      var op = Number(s.opacity);
      if (s.opacity !== undefined && s.opacity !== null && Number.isFinite(op) && op >= 0 && op <= 0.5) P.volume.opacity = op;
      var sl = s.streamlines || {};
      if ([1, 2, 3, 5].indexOf(Number(sl.density)) >= 0) P.volume.density = Number(sl.density);
      if (Number.isFinite(+sl.width) && +sl.width >= 0.4 && +sl.width <= 3) P.volume.width = +sl.width;
      if (typeof sl.thin === 'boolean') P.volume.thin = sl.thin;
      if (typeof s.vectors === 'boolean') P.layers.vectors = s.vectors;
      if (s.mode === 'slice') skipped.push('截面');
    }
    savePrefs(P);
    // layers of the shell (trust, streamlines) and the automatic labels
    var L = API.state && API.state().layers, lbl = s.labels || {};
    if (L) {
      if (s.overlay && typeof s.overlay.trust === 'boolean') L.trust = s.overlay.trust;
      if (fam === 'volume' && s.mode === 'streamlines') L.streamlines = true;
      API.setLayers();
    }
    if (API.store && (lbl.findings !== undefined || lbl.branches !== undefined || lbl.max_diameter !== undefined)) {
      var cl = Object.assign({}, API.store().prefs().labels), n = Number(lbl.findings);
      if ([0, 3, 5, 10].indexOf(n) >= 0) cl.findings = n;
      if (typeof lbl.branches === 'boolean') cl.branches = lbl.branches;
      if (typeof lbl.max_diameter === 'boolean') cl.maxd = lbl.max_diameter;
      API.store().setPrefs({ labels: cl });
      if (typeof API.drawLabels === 'function') API.drawLabels();
      done.push('自动标注');
    }
    var win = 'adaptive', r = s.range || {};
    if (r.mode === 'fixed' && Number(r.max) > 0 && (!r.field || r.field === fid)) win = { range: [0, Number(r.max)] };
    else if (s.log === false && fam === 'wall' && ns.colormap && ns.colormap.logHint(API.fieldById(m, fid))) win = 'adaptive-linear';
    API.applyField(fid, win);
    if (Array.isArray(s.branches_hidden) && s.branches_hidden.length) {
      var all = ((m.geometry && m.geometry.branches) || []).map(function (x) { return Number(x.id); }), hide = s.branches_hidden.map(Number);
      var keep = all.filter(function (id) { return hide.indexOf(id) < 0; });
      try { v.setBranchVisibility(keep.length === all.length || !keep.length ? null : keep); } catch (_) {}
    }
    var sameRun = s.run_identity && s.run_identity === cur.runIdentity;
    if (s.camera && sameRun) { try { v.setCamera(s.camera, { animate: true }); } catch (_) {} }
    else if (s.camera) skipped.push('视角（预设存自另一份结果）');
    ['measurements', 'probe_log'].forEach(function (k) { if (Array.isArray(s[k]) && s[k].length) skipped.push(k === 'measurements' ? '测量' : '探针记录'); });
    if (s.lang === 'en') skipped.push('英文');
    refreshAll(); updateLegends(); saveSoon();
    rerender(['reading', 'slice', 'overview', 'along', 'region']);
    ui().toast('已应用预设「' + (p.name || '') + '」。' + (skipped.length ? '没带：' + skipped.join('、') + '。' : ''), { kind: 'ok', ms: 6000 });
    return true;
  }
  function classicSection(api) {
    API = api;
    var cur = api.cur();
    if (!cur || !cur.manifest || api.offline()) return [];
    var h = api.h, U = api.ui(), fam = cur.manifest.result && cur.manifest.result.family === 'volume' ? 'volume' : 'wall';
    var body = h('div', { 'class': 'wsd-classic' }, U.note('正在读服务端偏好…'));
    function render(doc) {
      var d = doc && doc.report_defaults && doc.report_defaults[fam], list = doc && doc.presets && Array.isArray(doc.presets[fam]) ? doc.presets[fam].filter(function (x) { return x && typeof x === 'object' && typeof x.name === 'string' && x.state; }) : [];
      var parts = [];
      if (d && typeof d === 'object') {
        var plan = defaultsPlan(fam, d), used = prefs().opts.imported[fam] === signature(d);
        parts.push(h('div', { 'class': 'wsd-classic-row' }, h('span', { 'class': 'wsd-classic-k', text: '默认口径' }),
          h('span', { 'class': 'wsd-classic-v', text: plan.text.length ? plan.text.join(' · ') : '没有可用的设置' }),
          plan.text.length ? U.button(used ? '已套用' : '套用', function () { if (applyDefaults(fam, d)) render(doc); }, { cls: 'btn-sm', title: '写进这台电脑的显示设置（分段、阈值、单位、色表），以后打开的结果都这样显示' }) : null));
      }
      list.forEach(function (p) {
        parts.push(h('div', { 'class': 'wsd-classic-row' }, h('span', { 'class': 'wsd-classic-k', text: '预设' }),
          h('span', { 'class': 'wsd-classic-v', text: p.name + (p.created_at ? ' · ' + String(p.created_at).slice(0, 10) : '') }),
          U.button('应用', function () { applyPreset(p); }, { cls: 'btn-sm', title: '按这个预设显示当前结果' })));
      });
      if (!parts.length) parts.push(U.note('服务端没有经典报告存的' + (fam === 'volume' ? '体场' : '壁面') + '预设和默认口径。'));
      U.fill(body, parts);
    }
    loadServerPrefs().then(render, function (e) { U.fill(body, U.note('读不到服务端偏好：' + (e && e.message || e))); });
    return [U.section('经典设置', { tag: U.infoTip('经典报告的「预设」和「设为默认口径」存在服务端偏好里。这里读出来：默认口径可以套用到这台电脑的显示设置；预设可以按它显示当前结果。') }, body)];
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
    // P3 lane 4: iso-lines, highest-x % highlight and the Z cut (wall results), the hover readout (every result)
    var o = P.opts;
    if (!vol) {
      var f = api.fieldById ? api.fieldById(m, cur.field) : fieldOf(m, cur.field), v = api.viewer(), s = sessionFor(v, true);
      var th = f && provider.thresholds(v, f.id) || validThresholds(f && f.display && f.display.thresholds);
      var q = th ? th.map(function (x) { return toDisplay(x, f.units, 'wall'); }) : null;
      items.push({ label: '等值线', checked: o.contours, disabled: !th, run: function () { setOpts({ contours: !prefs().opts.contours }, 'p3'); },
        hint: q ? q.map(function (x) { return fmtCut(x.value); }).join(' / ') + (unitText(q[0].units) ? ' ' + unitText(q[0].units) : '') : '这个字段没有阈值' });
      items.push({ label: '高亮最高 ' + fmtCut(o.topPct) + '%…', checked: o.top, run: openTop });
      items.push({ label: '剖切…', checked: Boolean(s && s.clip !== undefined), run: openClip, hint: s && s.clip !== undefined ? clipText(v) : null });
    }
    items.push({ label: '悬停读数', checked: o.hover, run: toggleHover, hint: 'P' });
    return items;
  }
  function toggleHover() {
    var on = !prefs().opts.hover;
    setOpts({ hover: on }, 'quiet');
    if (!on && ns.probe && typeof ns.probe.hideHover === 'function') ns.probe.hideHover();
    ui().toast(on ? '悬停读数：开（P 关闭）' : '悬停读数：关（P 打开）', { kind: 'info', ms: 2000 });
    return true;
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
      var S0 = shell(); if (S0) { wireViewer(S0.viewerA); wireViewer(S0.viewerB); }   // P3 lane 4: viewer B appears with compare / split
      if (!api.viewer() || !api.cur() || !api.cur().result) return [];
      return [ui().iconButton('home', '复位视角：前视并撑满（0）', resetView)];
    },
    layers: function (api) { return layerItems(api); },
    key: function (e, api) {
      API = api;
      if (e && e.key === '0' && !e.shiftKey) return resetView();
      if (e && (e.key === 'p' || e.key === 'P') && api.cur() && api.cur().result) return toggleHover();   // P3 lane 4 (classic P)
      return false;
    },
    tools: function (api) { return classicSection(api); },
    // P3 lane 4 × lane 5 (复现链接): what the link must carry beyond lane 5's d — iso-lines, the highlight, the velocity log
    // scale and this result's wall cut.  The WSS unit travels in d.units (prefs().units.wss), the fixed upper limit in w.
    linkState: function (api) { API = api; return linkState(api); },
    applyLinkState: function (state, api) { API = api; return applyLinkState(state, api); },
    onResult: function (api) {
      API = api; closePop();
      var P = prefs(), S = shell();
      if (P.layers.vectors || P.layers.stagnation) ensureArrays().then(refreshAll, function () {});
      if (S) { wireViewer(S.viewerA); wireViewer(S.viewerB); }
      updateLegends();
      applyRemembered(api, true);
    },
    onField: function (api) { API = api; if (pop && pop.kind === 'scale') closePop(); applyRemembered(api, false); },
    onClose: function (api) { API = api; closePop(); Object.keys(legends).forEach(function (k) { if (legends[k]) legends[k].hidden = true; }); }
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
    resetView: resetView, pushSide: pushSide, keepShared: keepShared, refreshAll: refreshAll, BAND_STEPS: BAND_STEPS.slice(), UNITS: UNITS,
    // phase 3 lane 4
    displayKind: displayKind, sanitizeOpts: sanitizeOpts, setOpts: setOpts, openTop: openTop, openClip: openClip, setClip: setClip, toggleHover: toggleHover,
    trustRows: trustRows, updateLegends: updateLegends, defaultsPlan: defaultsPlan, applyDefaults: applyDefaults, applyPreset: applyPreset,
    loadServerPrefs: loadServerPrefs, applyRemembered: applyRemembered, layersButton: layersButton, linkState: linkState, applyLinkState: applyLinkState, openPanel: function (kind, title, anchorEl, build) { return openPop(kind, title, rectOf(anchorEl), placeRightOf, build); }
  };
});
