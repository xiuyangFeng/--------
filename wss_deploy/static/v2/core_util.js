/* WSSV2 core · util (contract WORKSPACE_V2_CONTRACT.md §6.1 / §6.2 / §7).
   Number formatting, a tiny event emitter, rAF throttling and a uniform-grid nearest-point index.
   Classic script: attaches WSSV2.util, require()-able under Node, touches no DOM while loading. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.util = mod;
  ns.env = mod.env;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  // ------------------------------------------------------------------ numbers (§7: three significant digits)
  function clamp(v, lo, hi) { v = +v; return v < lo ? lo : v > hi ? hi : v; }
  // Number or NaN: null / undefined / '' / booleans are missing, not 0 (Number(null) === 0 is a trap).
  function num(v) {
    if (v === null || v === undefined || v === '' || typeof v === 'boolean') return NaN;
    var x = Number(v);
    return Number.isFinite(x) ? x : NaN;
  }

  // Three significant digits, never scientific notation: 17.006 → "17.0", 0.00177 → "0.00177",
  // 1.572 → "1.57", 9.996 → "10.0", 15544.8 → "15545" (integers are never cut to fewer digits).
  // Missing (null / NaN / ±Infinity) → "—": a missing value is never printed as 0 (§1.6 invariant 3).
  function fmtSig(v, sig) {
    sig = sig > 0 ? Math.floor(sig) : 3;
    if (v === null || v === undefined || v === '' || typeof v === 'boolean') return '—';
    v = Number(v);
    if (!Number.isFinite(v)) return '—';
    if (v === 0) return '0';
    var a = Math.abs(v);
    if (a >= Math.pow(10, sig - 1) && a >= 1) {
      var r = Math.round(v);
      if (Math.abs(r) >= Math.pow(10, sig)) return String(r);
    }
    var mag = Math.floor(Math.log10(a));
    var dec = Math.max(0, Math.min(10, sig - 1 - mag));
    var s = v.toFixed(dec);
    // Rounding may carry into the next decade (9.996 → "10.00"): drop one decimal then.
    if (dec > 0 && Math.abs(+s) >= Math.pow(10, mag + 1)) s = v.toFixed(dec - 1);
    if (/^-0(\.0*)?$/.test(s)) s = s.slice(1);
    if (+s === 0) {   // too small for 10 decimals: keep the sign of life instead of a false zero
      return (v < 0 ? '-' : '') + '<' + '0.' + new Array(10).join('0') + '1';
    }
    return s;
  }

  // Percentages are whole numbers; below 1 % one decimal (§7).  0 < p < 0.05 % → "<0.1%"; 99 % ≤ p < 100 %
  // never rounds up to a false "100%".
  function fmtPct(frac) {
    if (frac === null || frac === undefined || frac === '') return '—';
    var f = Number(frac);
    if (!Number.isFinite(f)) return '—';
    var p = f * 100, a = Math.abs(p), sign = p < 0 ? '-' : '';
    if (a === 0) return '0%';
    if (a < 0.05) return sign + '<0.1%';
    if (a < 1) { var t = a.toFixed(1); return sign + (t === '1.0' ? '1' : t) + '%'; }
    if (a >= 99 && a < 100) return sign + Math.floor(a) + '%';
    return sign + Math.round(a) + '%';
  }

  // Display form of a unit: "1" (dimensionless, OSI) prints nothing, 1/Pa prints Pa⁻¹ (same as the classic report).
  function unitText(units) {
    if (units === null || units === undefined) return '';
    var u = String(units);
    if (u === '1' || u === '') return '';
    if (u === '1/Pa') return 'Pa⁻¹';
    return u;
  }
  function fmtValue(v, units, sig) {
    var s = fmtSig(v, sig), u = unitText(units);
    return s === '—' || !u ? s : s + ' ' + u;
  }
  // Same digits as fmtSig with trailing zeros dropped (1.00 → "1", 0.0500 → "0.05"): for values that are round by
  // construction (window bounds, nice ticks, thresholds), not for measured values.
  function fmtTrim(v, sig) {
    var s = fmtSig(v, sig);
    if (s.indexOf('.') < 0 || s.charAt(0) === '<' || s === '—') return s;
    return s.replace(/0+$/, '').replace(/\.$/, '');
  }
  // "0–4.36 Pa"; with a negative end "−1400 至 130 Pa" (a dash between signed numbers reads as a minus).
  function fmtRange(lo, hi, units, opts) {
    var u = unitText(units), f = opts && opts.trim ? fmtTrim : fmtSig, fl = opts && (opts.trim || opts.trimLo) ? fmtTrim : fmtSig;
    var sep = (Number(lo) < 0 || Number(hi) < 0) ? ' 至 ' : '–';
    return fl(lo) + sep + f(hi) + (u ? ' ' + u : '');
  }

  function escapeHtml(x) {
    return String(x === null || x === undefined ? '' : x).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  // ------------------------------------------------------------------ events
  function emitter() {
    var map = Object.create(null);
    var api = {
      on: function (name, fn) {
        if (typeof fn !== 'function') return function () {};
        (map[name] = map[name] || []).push(fn);
        return function () { api.off(name, fn); };
      },
      off: function (name, fn) {
        var a = map[name]; if (!a) return;
        var i = a.indexOf(fn); if (i >= 0) a.splice(i, 1);
      },
      emit: function (name, payload) {
        var a = map[name]; if (!a || !a.length) return 0;
        var list = a.slice(), n = 0;
        for (var i = 0; i < list.length; i++) {
          try { list[i](payload); n++; } catch (err) {
            if (name !== 'error' && map.error && map.error.length) api.emit('error', err);
            else if (typeof console !== 'undefined') console.error(err);
          }
        }
        return n;
      },
      count: function (name) { return map[name] ? map[name].length : 0; },
      clear: function () { map = Object.create(null); }
    };
    return api;
  }

  // At most one call per animation frame with the latest arguments; .cancel() drops a pending call.
  function rafThrottle(fn) {
    var pending = 0, lastArgs = null, useRaf = typeof root.requestAnimationFrame === 'function';
    function run() { pending = 0; var a = lastArgs; lastArgs = null; fn.apply(null, a || []); }
    function throttled() {
      lastArgs = Array.prototype.slice.call(arguments);
      if (pending) return;
      pending = useRaf ? root.requestAnimationFrame(run) : setTimeout(run, 16);
      if (!pending) pending = -1;
    }
    throttled.cancel = function () {
      if (pending && pending !== -1) { if (useRaf) root.cancelAnimationFrame(pending); else clearTimeout(pending); }
      pending = 0; lastArgs = null;
    };
    throttled.flush = function () { if (pending) { throttled.cancel(); } };
    return throttled;
  }

  function now() {
    return root.performance && typeof root.performance.now === 'function' ? root.performance.now() : Date.now();
  }

  // ------------------------------------------------------------------ statistics helpers
  // Linear-interpolated quantile of the finite values (numpy "linear"), NaN when there are none.
  function quantile(values, q, mask) {
    var v = [];
    for (var i = 0; i < values.length; i++) {
      if (mask && !mask[i]) continue;
      var x = values[i]; if (Number.isFinite(x)) v.push(x);
    }
    if (!v.length) return NaN;
    var arr = Float64Array.from(v).sort();
    var at = clamp(q, 0, 1) * (arr.length - 1), lo = Math.floor(at), hi = Math.ceil(at);
    return arr[lo] + (arr[hi] - arr[lo]) * (at - lo);
  }
  function finiteRange(values, mask) {
    var lo = Infinity, hi = -Infinity, n = 0;
    for (var i = 0; i < values.length; i++) {
      if (mask && !mask[i]) continue;
      var x = values[i]; if (!Number.isFinite(x)) continue;
      n++; if (x < lo) lo = x; if (x > hi) hi = x;
    }
    return n ? { min: lo, max: hi, n: n } : { min: NaN, max: NaN, n: 0 };
  }

  // ------------------------------------------------------------------ nearest-point index
  // Uniform grid over flat xyz points.  nearest(x, y, z, {maxDist, accept(i)}) → index or -1.  Searches rings
  // of cells outward and stops once the best distance is inside the searched shell, so a query costs a few
  // cells instead of a full scan.
  function gridIndex(points, opts) {
    opts = opts || {};
    var n = Math.floor(points.length / 3);
    var mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity], k, i;
    for (i = 0; i < n; i++) for (k = 0; k < 3; k++) {
      var v = points[3 * i + k]; if (!Number.isFinite(v)) continue;
      if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v;
    }
    if (!n || !Number.isFinite(mn[0])) {
      return { n: 0, cell: 1, nearest: function () { return -1; }, nearestMany: function (q) { return new Int32Array(Math.floor(q.length / 3)).fill(-1); } };
    }
    var ext = [0, 1, 2].map(function (j) { return Math.max(mx[j] - mn[j], 1e-6); });
    var cell = opts.cell > 0 ? opts.cell : Math.max(Math.cbrt(ext[0] * ext[1] * ext[2] / n) * 2, 1e-3);
    var dims = [0, 1, 2].map(function (j) { return Math.min(1024, Math.floor(ext[j] / cell) + 1); });
    function cellOf(x, j) { var c = Math.floor((x - mn[j]) / cell); return c < 0 ? 0 : c >= dims[j] ? dims[j] - 1 : c; }
    // Counting sort into CSR arrays (cells → point ids); no per-cell JS arrays.
    var nc = dims[0] * dims[1] * dims[2], cellId = new Int32Array(n), start = new Int32Array(nc + 1), order = new Int32Array(n);
    for (i = 0; i < n; i++) {
      var x = points[3 * i], y = points[3 * i + 1], z = points[3 * i + 2];
      if (!(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z))) { cellId[i] = -1; continue; }
      var c = cellOf(x, 0) + dims[0] * (cellOf(y, 1) + dims[1] * cellOf(z, 2));
      cellId[i] = c; start[c + 1]++;
    }
    for (i = 0; i < nc; i++) start[i + 1] += start[i];
    var fill = start.slice(0, nc);
    for (i = 0; i < n; i++) if (cellId[i] >= 0) order[fill[cellId[i]]++] = i;
    var maxRing = Math.max(dims[0], dims[1], dims[2]);
    function nearest(x, y, z, o) {
      var maxDist = o && o.maxDist > 0 ? o.maxDist : Infinity, accept = o && typeof o.accept === 'function' ? o.accept : null;
      var cx = cellOf(x, 0), cy = cellOf(y, 1), cz = cellOf(z, 2), best = -1, bd = Infinity;
      var ringLimit = Number.isFinite(maxDist) ? Math.min(maxRing, Math.ceil(maxDist / cell) + 1) : maxRing;
      for (var r = 0; r <= ringLimit; r++) {
        for (var ix = cx - r; ix <= cx + r; ix++) {
          if (ix < 0 || ix >= dims[0]) continue;
          for (var iy = cy - r; iy <= cy + r; iy++) {
            if (iy < 0 || iy >= dims[1]) continue;
            var edgeXY = ix === cx - r || ix === cx + r || iy === cy - r || iy === cy + r;
            for (var iz = cz - r; iz <= cz + r; iz++) {
              if (iz < 0 || iz >= dims[2]) continue;
              if (!edgeXY && iz !== cz - r && iz !== cz + r) continue;
              var c = ix + dims[0] * (iy + dims[1] * iz);
              for (var p = start[c], e = start[c + 1]; p < e; p++) {
                var j = order[p];
                if (accept && !accept(j)) continue;
                var dx = points[3 * j] - x, dy = points[3 * j + 1] - y, dz = points[3 * j + 2] - z, d = dx * dx + dy * dy + dz * dz;
                if (d < bd) { bd = d; best = j; }
              }
            }
          }
        }
        if (best >= 0 && Math.sqrt(bd) <= r * cell) break;
      }
      if (best >= 0 && Math.sqrt(bd) > maxDist) return -1;
      return best;
    }
    function nearestMany(q, o) {
      var m = Math.floor(q.length / 3), out = new Int32Array(m);
      for (var j = 0; j < m; j++) out[j] = nearest(q[3 * j], q[3 * j + 1], q[3 * j + 2], o);
      return out;
    }
    return { n: n, cell: cell, dims: dims, nearest: nearest, nearestMany: nearestMany };
  }

  // ------------------------------------------------------------------ environment
  // Evaluated on access, never at load (modules must not touch the DOM while loading, §6.1).
  var env = {};
  Object.defineProperty(env, 'offline', {
    enumerable: true,
    get: function () {
      try { return typeof root.document !== 'undefined' && !!root.document && !!root.document.getElementById('wssv2-manifest'); }
      catch (_) { return false; }
    }
  });

  return {
    fmtSig: fmtSig, fmtTrim: fmtTrim, fmtPct: fmtPct, fmtValue: fmtValue, fmtRange: fmtRange, unitText: unitText, escapeHtml: escapeHtml,
    emitter: emitter, rafThrottle: rafThrottle, clamp: clamp, num: num, now: now,
    quantile: quantile, finiteRange: finiteRange, gridIndex: gridIndex, env: env
  };
});
