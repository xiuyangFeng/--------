/* WSSV2 core · colormap (contract §6.2, §7; discussion V7, U15).
   Colour scales over a value range (adaptive / named window / fixed), optional log and discrete bands, the
   missing colour for NaN, and the sample-point histogram shown beside the colour bar.  Palette stops come from
   report_common.js (WssReportCommon.PALETTES): rainbow is bit-identical to the classic reports. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.colormap = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var MISSING_HEX = '#a7adb3';   // --missing (§7)
  var DEFAULT_LOG_FLOOR = 0.05;   // the classic wall report's log floor (Pa); used when the range starts at ≤ 0
  var NAMES = ['rainbow', 'turbo', 'viridis', 'bwr'];
  var LABELS = { rainbow: '彩虹', turbo: 'Turbo', viridis: 'Viridis', bwr: '蓝白红' };

  function common() {
    var c = root.WssReportCommon;
    if (!c || typeof c.paletteRGB !== 'function') throw new Error('report_common.js (WssReportCommon) must be loaded before core_colormap.js is used');
    return c;
  }
  function util() { return ns.util; }

  function hexToRgb(h) {
    var m = /^#?([0-9a-f]{6})$/i.exec(String(h || ''));
    if (!m) return null;
    var v = parseInt(m[1], 16);
    return [(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255];
  }
  function rgbToHex(c) {
    return '#' + c.map(function (x) { var v = Math.round(Math.max(0, Math.min(1, x)) * 255); return (v < 16 ? '0' : '') + v.toString(16); }).join('');
  }
  function names() { return NAMES.slice(); }
  function label(name) { return LABELS[name] || name; }
  function known(name) { return NAMES.indexOf(name) >= 0; }
  // t in 0..1 (clamped) → [r, g, b] in 0..1, exactly WssReportCommon.paletteRGB (linear between stops).
  function rgb(name, t) { return common().paletteRGB(known(name) ? name : 'rainbow', t); }

  // Whether a log scale is meaningful for this range / field minimum.  Log cannot show values ≤ 0 honestly:
  // a range or a field that goes below zero (relative pressure) is refused with a reason.
  function logAllowed(o) {
    o = o || {};
    var lo = Number(o.range ? o.range[0] : o.min), hi = Number(o.range ? o.range[1] : o.max), fieldMin = Number(o.fieldMin);
    if (Number.isFinite(fieldMin) && fieldMin < 0) return { ok: false, reason: '该量有负值（跨零），对数色标不适用' };
    if (Number.isFinite(lo) && lo < 0) return { ok: false, reason: '色标范围含负值，对数色标不适用' };
    if (Number.isFinite(hi) && hi <= 0) return { ok: false, reason: '色标上限不为正，对数色标不适用' };
    return { ok: true, reason: null };
  }

  function logFloor(lo, hi, floor) {
    if (floor > 0 && floor < hi) return floor;
    if (lo > 0) return lo;
    var f = DEFAULT_LOG_FLOOR;
    if (!(f < hi / 2)) f = hi / 100;
    return f;
  }

  // spec = {range:[lo, hi], log, bands, cmap, missing, floor, units}
  function scale(spec) {
    spec = spec || {};
    var r = Array.isArray(spec.range) ? spec.range : [0, 1];
    var lo = Number(r[0]), hi = Number(r[1]);
    if (!Number.isFinite(lo)) lo = 0;
    if (!Number.isFinite(hi)) hi = lo + 1;
    if (hi < lo) { var t0 = lo; lo = hi; hi = t0; }
    if (!(hi > lo)) hi = lo + Math.max(Math.abs(lo) * 1e-6, 1e-9);
    var cmap = known(spec.cmap) ? spec.cmap : 'rainbow';
    var bands = Math.max(0, Math.min(64, Math.floor(Number(spec.bands) || 0)));
    if (bands === 1) bands = 0;
    var logRejected = null, log = spec.log === true;
    if (log) {
      var ok = logAllowed({ range: [lo, hi], fieldMin: spec.fieldMin });
      if (!ok.ok) { log = false; logRejected = ok.reason; }
    }
    var floor = log ? logFloor(lo, hi, Number(spec.floor)) : null;
    var l0 = log ? Math.log(Math.max(lo, floor)) : 0, l1 = log ? Math.log(Math.max(hi, floor * 1.001)) : 0;
    var missing = Array.isArray(spec.missing) ? spec.missing.slice(0, 3) : (hexToRgb(spec.missing) || hexToRgb(MISSING_HEX));
    var units = spec.units === undefined ? null : spec.units;

    // Position on the bar, unclamped (< 0 below range, > 1 above); NaN for missing values.
    function norm(v) {
      if (v === null || v === undefined) return NaN;
      v = +v;
      if (!Number.isFinite(v)) return NaN;
      if (log) return (Math.log(Math.max(v, floor * 1e-6)) - l0) / ((l1 - l0) || 1);
      return (v - lo) / (hi - lo);
    }
    function quantize(t) {
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      if (!bands) return t;
      return (Math.min(bands - 1, Math.floor(t * bands)) + 0.5) / bands;
    }
    function colorT(t) { return rgb(cmap, quantize(t)); }
    // Values outside the range take the end colours (the colour bar says so); NaN → the missing colour.
    function color(v) {
      var t = norm(v);
      if (t !== t) return missing.slice();
      return colorT(t);
    }
    // Bulk path: writes rgb triples into out (Float32Array) at 3·i and returns the number of missing values.
    // Same interpolation formula as WssReportCommon.paletteRGB (bit-identical colours), without allocations.
    // mask/dim: entries with mask[i] falsy are blended toward dim = [r, g, b, keep].
    function fill(values, out, mask, dim) {
      var stops = common().PALETTES[cmap].stops, ns1 = stops.length - 1, n = values.length, nMissing = 0;
      for (var i = 0; i < n; i++) {
        var t = norm(values[i]), r, g, b;
        if (t !== t) { r = missing[0]; g = missing[1]; b = missing[2]; nMissing++; }
        else {
          t = quantize(t);
          var x = t * ns1, k = Math.min(ns1 - 1, Math.floor(x)), u = x - k, s0 = stops[k], s1 = stops[k + 1];
          r = (s0[0] + (s1[0] - s0[0]) * u) / 255; g = (s0[1] + (s1[1] - s0[1]) * u) / 255; b = (s0[2] + (s1[2] - s0[2]) * u) / 255;
        }
        if (mask && !mask[i] && dim) { var q = dim[3]; r = r * q + dim[0] * (1 - q); g = g * q + dim[1] * (1 - q); b = b * q + dim[2] * (1 - q); }
        out[3 * i] = r; out[3 * i + 1] = g; out[3 * i + 2] = b;
      }
      return nMissing;
    }
    function valueAt(t) {
      t = t < 0 ? 0 : t > 1 ? 1 : +t || 0;
      return log ? Math.exp(l0 + t * (l1 - l0)) : lo + t * (hi - lo);
    }
    // [{v, f (0 bottom … 1 top), label, major}] — nice 1/2/2.5/5 steps plus the exact ends (report_common),
    // or the band edges when banded.
    function ticks(n) {
      var c = common(), fmt = util() ? (spec.trim ? util().fmtTrim : util().fmtSig) : function (x) { return String(x); };
      var raw = c.colorbarTicks({ min: log ? floor : lo, max: hi, log: log, bands: bands, floor: floor || undefined, target: n > 0 ? n : 6, minGap: 0.07 });
      if (log && lo < floor) raw.forEach(function (tk) { tk.f = norm(tk.v); });
      return raw.map(function (tk) {
        return { v: tk.v, f: Math.max(0, Math.min(1, tk.f)), major: !!tk.major, label: tk.nice ? String(+(+tk.v).toPrecision(6)) : fmt(tk.v) };
      });
    }
    function describe() {
      var U = util();
      var parts = [U ? U.fmtRange(log ? floor : lo, hi, units, { trim: !!spec.trim, trimLo: log || !!spec.trim }) : (lo + '–' + hi)];
      if (log) parts.push('对数');
      if (bands) parts.push(bands + ' 段');
      return {
        range: [lo, hi], shown: [log ? floor : lo, hi], log: log, logRejected: logRejected, floor: floor, bands: bands,
        cmap: cmap, cmapLabel: label(cmap), missing: rgbToHex(missing), units: units,
        text: parts.join(' · ')
      };
    }
    return {
      range: [lo, hi], log: log, bands: bands, cmap: cmap, floor: floor, logRejected: logRejected, missing: missing, units: units,
      norm: norm, color: color, colorT: colorT, fill: fill, valueAt: valueAt, ticks: ticks, describe: describe
    };
  }

  // Histogram of the finite values.  Bins are equal in value (or in log value when log); values outside the
  // range count in nBelow / nAbove, NaN in nMissing.  n = number of finite values (the denominator shown).
  function histogram(values, opts) {
    opts = opts || {};
    var bins = Math.max(1, Math.min(512, Math.floor(opts.bins) || 40));
    var mask = opts.mask || null, U = util();
    var r = Array.isArray(opts.range) ? opts.range : null;
    var lo, hi;
    if (r) { lo = +r[0]; hi = +r[1]; }
    else { var fr = U.finiteRange(values, mask); lo = fr.min; hi = fr.max; }
    var log = opts.log === true && hi > 0;
    var floor = log ? logFloor(lo, hi, Number(opts.floor)) : null;
    if (log && lo < floor) lo = floor;
    if (!(hi > lo)) hi = (Number.isFinite(lo) ? lo : 0) + 1e-9;
    var l0 = log ? Math.log(lo) : lo, l1 = log ? Math.log(hi) : hi, w = (l1 - l0) / bins;
    var counts = new Array(bins).fill(0), n = 0, nMissing = 0, nBelow = 0, nAbove = 0;
    for (var i = 0; i < values.length; i++) {
      if (mask && !mask[i]) continue;
      var v = values[i];
      if (!Number.isFinite(v)) { nMissing++; continue; }
      n++;
      if (v < lo) { nBelow++; continue; }
      if (v > hi) { nAbove++; continue; }
      var x = log ? Math.log(v) : v, k = Math.floor((x - l0) / w + 1e-9);   // values on an edge go to the upper bin despite rounding
      if (k >= bins) k = bins - 1; if (k < 0) k = 0;
      counts[k]++;
    }
    var edges = [];
    for (var j = 0; j <= bins; j++) edges.push(log ? Math.exp(l0 + j * w) : l0 + j * w);
    return { edges: edges, counts: counts, n: n, nMissing: nMissing, nBelow: nBelow, nAbove: nAbove, log: log, range: [lo, hi] };
  }

  // ---------------------------------------------------------------- field-level scale resolution
  // Resolves a viewer scaleSpec {window:'adaptive'|<window id>|{range:[a,b]}, log:null|bool, bands, cmap} for a
  // manifest field.  stats = {p99, p1, min, max} of the field's own values (read array, else display).
  //   adaptive: 0 … p99 when the field has no negative values (the classic wall report's default range),
  //             p1 … p99 when it does (relative pressure: 本例自适应, no fixed window).
  //   named:    field.windows[id].range (provisional windows are flagged).
  //   fixed:    the given range.
  // log = null → the field default: the adaptive window follows the release's display hint
  //   (field.display.log_scale, e.g. TAWSS / RRT / ECAP — the classic report shows them on a log
  //   scale, otherwise a sac full of low values paints as one colour); named and fixed windows stay linear.
  // Whether the adaptive window of a field opens on a log scale: the release's display hint when it states one;
  // otherwise the wall shear family (WSS, TAWSS, RRT, ECAP), whose values span two decades so a linear 0–p99 paints
  // a whole sac in one colour.  OSI, pressures and speeds stay linear.
  var LOG_FIELDS = { wss: true, tawss: true, rrt: true, ecap: true };
  function logHint(field) {
    var d = field && field.display;
    if (d && d.log_scale === true) return true;
    if (d && d.log_scale === false) return false;
    return Boolean(field && LOG_FIELDS[field.id]);
  }
  function resolve(field, spec, stats) {
    field = field || {};
    spec = spec || {};
    stats = stats || {};
    var U = util();
    var N = U ? U.num : function (x) { return x === null || x === undefined ? NaN : +x; };
    var fieldMin = Number.isFinite(N(stats.min)) ? N(stats.min) : null;
    var crossesZero = fieldMin !== null && fieldMin < 0;
    var win = spec.window === undefined || spec.window === null ? 'adaptive' : spec.window;
    var windows = Array.isArray(field.windows) ? field.windows : [];
    var range, info;
    if (win && typeof win === 'object' && Array.isArray(win.range) && win.range.length === 2 && win.range.every(function (x) { return Number.isFinite(N(x)); }) && +win.range[1] !== +win.range[0]) {
      range = [Math.min(+win.range[0], +win.range[1]), Math.max(+win.range[0], +win.range[1])];
      info = { kind: 'fixed', id: 'fixed', label: '固定范围', provisional: false };
    } else if (typeof win === 'string' && win !== 'adaptive') {
      var w = null;
      for (var i = 0; i < windows.length; i++) if (windows[i] && windows[i].id === win) w = windows[i];
      if (w && Array.isArray(w.range) && w.range.length === 2 && Number.isFinite(N(w.range[0])) && Number.isFinite(N(w.range[1])) && +w.range[1] > +w.range[0]) {
        range = [+w.range[0], +w.range[1]];
        info = { kind: 'named', id: w.id, label: String(w.label || w.id), provisional: w.provisional !== false };
      }
    }
    if (!range) {
      var p99 = N(stats.p99), disp = field.display || {};
      if (!Number.isFinite(p99)) p99 = N(disp.p99);
      var hiA = Number.isFinite(p99) ? p99 : (Number.isFinite(N(stats.max)) ? N(stats.max) : 1);
      var loA = crossesZero ? (Number.isFinite(N(stats.p1)) ? N(stats.p1) : fieldMin) : 0;
      if (!(hiA > loA)) hiA = loA + Math.max(Math.abs(loA) * 0.01, 0.01);
      range = [loA, hiA];
      info = { kind: 'adaptive', id: 'adaptive', label: '本例自适应', provisional: false,
        note: crossesZero ? '本例 p1–p99' : '0 至本例 p99' };
      if (typeof win === 'string' && win !== 'adaptive') info.fallback = '窗「' + win + '」不适用于该量，已用本例自适应';
    }
    var wantLog = spec.log === true ||
      (spec.log !== false && (spec.log === null || spec.log === undefined) && info.kind === 'adaptive' && !crossesZero && logHint(field));
    var sc = scale({ range: range, log: wantLog, bands: spec.bands, cmap: spec.cmap, units: field.units, fieldMin: fieldMin, floor: spec.floor, trim: info.kind !== 'adaptive' });
    info.text = info.label + (info.provisional ? '（暂定）' : '') + ' ' + (U ? U.fmtRange(sc.log ? sc.floor : sc.range[0], sc.range[1], field.units, { trim: info.kind !== 'adaptive', trimLo: sc.log || info.kind !== 'adaptive' }) : '');
    return {
      scale: sc, window: info, crossesZero: crossesZero,
      spec: { window: info.kind === 'fixed' ? { range: range.slice() } : info.id, log: sc.log, bands: sc.bands, cmap: sc.cmap }
    };
  }

  return {
    MISSING_HEX: MISSING_HEX, names: names, label: label, rgb: rgb, hexToRgb: hexToRgb, rgbToHex: rgbToHex,
    scale: scale, histogram: histogram, logAllowed: logAllowed, resolve: resolve, logHint: logHint
  };
});
