/* WSS workspace v2 — phase 3 lane 4 (display): the classic wall report's numerical helpers for iso-lines and the
 * highest-x % highlight, ported from report.py REPORT_CORE_JS (WssReportCore) so they survive S7:
 *   marchingTriangles(vertices, faces, values, level)  one iso-line level on a triangle mesh → Float32Array of segment ends
 *   topThreshold(values, pct)                          the value at the classic rank ⌈(1 − pct / 100) · n⌉ − 1 of the finite values
 * Both are copied expression for expression (same order of operations, float32 output), so the iso-lines and the
 * highlighted points are the classic page's own.  Everything here is display only; no stored number changes. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.contour = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  // report.py REPORT_CORE_JS marchingTriangles: a triangle whose three values are finite and straddle the level gives one
  // segment between its two crossing edges (edge order ab, bc, ca; a vertex exactly on the level counts as above).
  function marchingTriangles(vertices, faces, values, level) {
    var out = [];
    var cross = function (a, b) {
      var va = values[a], vb = values[b];
      if ((va >= level) === (vb >= level)) return null;
      var s = (level - va) / (vb - va);
      return [vertices[3 * a] + (vertices[3 * b] - vertices[3 * a]) * s, vertices[3 * a + 1] + (vertices[3 * b + 1] - vertices[3 * a + 1]) * s, vertices[3 * a + 2] + (vertices[3 * b + 2] - vertices[3 * a + 2]) * s];
    };
    for (var t = 0; t < faces.length; t += 3) {
      var a = faces[t], b = faces[t + 1], c = faces[t + 2];
      if (!(Number.isFinite(values[a]) && Number.isFinite(values[b]) && Number.isFinite(values[c]))) continue;
      var p = [cross(a, b), cross(b, c), cross(c, a)].filter(Boolean);
      if (p.length === 2) out.push(p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2]);
    }
    return new Float32Array(out);
  }
  // Several levels in one buffer, level after level (classic rebuildContours).  Levels that are not finite are skipped.
  function segments(vertices, faces, values, levels) {
    var parts = [], total = 0;
    (Array.isArray(levels) ? levels : []).forEach(function (lv) {
      var x = Number(lv);
      if (!Number.isFinite(x)) return;
      var s = marchingTriangles(vertices, faces, values, x);
      parts.push(s); total += s.length;
    });
    var all = new Float32Array(total), at = 0;
    parts.forEach(function (p) { all.set(p, at); at += p.length; });
    return all;
  }
  // report.py REPORT_CORE_JS topThreshold.
  function topThreshold(values, pct) {
    var vals = [];
    for (var i = 0; i < values.length; i++) if (Number.isFinite(values[i])) vals.push(values[i]);
    vals.sort(function (a, b) { return a - b; });
    if (!vals.length) return 0;
    var rank = Math.max(0, Math.min(vals.length - 1, Math.ceil((1 - pct / 100) * vals.length) - 1));
    return vals[rank];
  }
  // The highlighted points (classic rebuildTop): finite values ≥ the threshold, optionally only where keep(i) is true.
  function topIndices(values, pct, keep) {
    var thr = topThreshold(values, pct), idx = [];
    for (var i = 0; i < values.length; i++) if (Number.isFinite(values[i]) && values[i] >= thr && (!keep || keep(i))) idx.push(i);
    return { threshold: thr, indices: idx };
  }
  // Classic 剖切高度: t ∈ [0, 1] of the model's z range (the display mesh's bounds); t ≥ 1 is no cut.  Returns the level
  // z₀ (the part with z ≤ z₀ is kept), or null.
  function clipLevel(bounds, t) {
    if (t === null || t === undefined || t === '') return null;
    t = Number(t);
    if (!bounds || !Array.isArray(bounds.min) || !Array.isArray(bounds.max) || !Number.isFinite(t) || t >= 1) return null;
    t = Math.max(0, t);
    return bounds.min[2] + (bounds.max[2] - bounds.min[2]) * t;
  }
  var TOP_STEPS = { min: 0.5, max: 10, step: 0.5 };   // classic highlight-pct slider
  function cleanPct(p) { var x = Number(p); return Number.isFinite(x) ? Math.max(TOP_STEPS.min, Math.min(TOP_STEPS.max, Math.round(x / TOP_STEPS.step) * TOP_STEPS.step)) : 1; }

  return { marchingTriangles: marchingTriangles, segments: segments, topThreshold: topThreshold, topIndices: topIndices, clipLevel: clipLevel,
    cleanPct: cleanPct, TOP_STEPS: TOP_STEPS };
});
