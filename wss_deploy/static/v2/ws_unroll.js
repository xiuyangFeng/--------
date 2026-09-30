/* WSS workspace v2 — branch unrolled map (second phase lane D, S6d).
 * Arc length s × circumferential angle θ of the wall prediction points (manifest geometry.points: s_from_root_mm /
 * theta_rad / segment), one map per centreline branch, each cell the largest value of the points that fall in it:
 * the classic report's drawUnroll definition (report.py) —
 *   x = ⌊(s − s_min) / (s_max − s_min) · (W − 1)⌋,  y = ⌊(1 − (θ + π) / 2π) · (H − 1)⌋,  s range = the branch's own points.
 * The number of cells follows the number of points (a few per cell on average, at most one per screen pixel), so a
 * small branch reads as a filled map instead of scattered dots; the classic page used one cell per canvas pixel.
 * Colours come from the viewer's current scale, so the map reads like the vessel.  Pure parts are Node-testable. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.unroll = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var TWO_PI = 2 * Math.PI;

  function keysOf(m) {
    var p = (m && m.geometry && m.geometry.points) || {};
    return {xyz: p.xyz || null, segment: p.segment || null, s: p.s_from_root_mm || null, theta: p.theta_rad || null};
  }
  // Wall results whose manifest declares the points with their arc length and angle.
  function supported(result) {
    if (!result || !result.manifest || result.family === 'volume') return false;
    var k = keysOf(result.manifest);
    return Boolean(k.xyz && k.segment && k.s && k.theta && result.declared(k.xyz) && result.declared(k.segment) && result.declared(k.s) && result.declared(k.theta));
  }
  function readKey(result, fieldId) {
    var f = result && result.field ? result.field(fieldId) : null;
    return f && f.arrays && typeof f.arrays.read === 'string' && (Number(f.components) || 1) === 1 ? f.arrays.read : null;
  }
  function requiredArrays(result, fieldId) {
    var k = keysOf(result.manifest);
    var out = [k.xyz, k.segment, k.s, k.theta];
    var r = fieldId ? readKey(result, fieldId) : null;
    if (r) out.push(r);
    return out.filter(function (x) { return typeof x === 'string' && result.declared(x); });
  }

  // One entry per segment present in the points (ascending id, like numpy.unique in report.branch_unroll_data).
  // nS = distinct arc lengths (0.01 mm): the points come in rings, so a map never needs more columns than rings.
  function branches(seg, s, manifest) {
    var acc = {}, rings = {};
    for (var i = 0; i < seg.length; i++) {
      var id = seg[i], v = s[i];
      if (!Number.isFinite(v)) continue;
      var b = acc[id];
      if (!b) { b = acc[id] = {segmentId: Number(id), n: 0, nS: 0, sMin: Infinity, sMax: -Infinity}; rings[id] = {}; }
      b.n++;
      var key = Math.round(v * 100);
      if (!rings[id][key]) { rings[id][key] = 1; b.nS++; }
      if (v < b.sMin) b.sMin = v;
      if (v > b.sMax) b.sMax = v;
    }
    var names = {};
    ((manifest && manifest.geometry && manifest.geometry.branches) || []).forEach(function (b) { if (b) names[Number(b.id)] = b.name; });
    return Object.keys(acc).map(function (k) { return acc[k]; }).sort(function (a, b) { return a.segmentId - b.segmentId; })
      .map(function (b) { b.name = names[b.segmentId] || ('分支 ' + b.segmentId); return b; });
  }
  // Cells: about `perCell` points each, never finer than the screen pixels (maxW × maxH), at least 8 × 8.  `aspect` =
  // columns / rows; the caller passes the branch's length over its circumference so cells are about square on the wall
  // (a short branch stretched across the strip would otherwise get columns finer than its point rings).
  function gridSize(n, maxW, maxH, perCell, aspect) {
    maxW = Math.max(1, Math.round(maxW)); maxH = Math.max(1, Math.round(maxH));
    var cells = Math.max(1, n / (perCell || 2));
    aspect = aspect > 0 && isFinite(aspect) ? aspect : maxW / maxH;
    var H = Math.round(Math.sqrt(cells / aspect)), W = Math.round(H * aspect);
    H = Math.max(8, Math.min(maxH, H)); W = Math.max(8, Math.min(maxW, W));
    return {width: W, height: H};
  }
  // o = {s, theta, segment, values, segmentId, sMin, sMax, width, height} → {max (Float32, −∞ = empty), arg (point index), n, nMissing}
  function grid(o) {
    var W = Math.max(1, o.width | 0), H = Math.max(1, o.height | 0), total = W * H;
    var max = new Float32Array(total), arg = new Int32Array(total);
    max.fill(-Infinity); arg.fill(-1);
    var s = o.s, th = o.theta, seg = o.segment, val = o.values, sid = o.segmentId;
    var s0 = +o.sMin, span = (+o.sMax - s0) || 1, n = 0, nMissing = 0;
    for (var i = 0; i < seg.length; i++) {
      if (seg[i] !== sid) continue;
      var v = val[i];
      if (!Number.isFinite(v)) { nMissing++; continue; }   // a missing value never paints a cell (the classic Math.max would turn it NaN)
      var x = Math.max(0, Math.min(W - 1, Math.floor((s[i] - s0) / span * (W - 1))));
      var y = Math.max(0, Math.min(H - 1, Math.floor((1 - (th[i] + Math.PI) / TWO_PI) * (H - 1))));
      var k = y * W + x;
      n++;
      if (v > max[k]) { max[k] = v; arg[k] = i; }
    }
    return {width: W, height: H, max: max, arg: arg, n: n, nMissing: nMissing, sMin: s0, sMax: +o.sMax, segmentId: sid};
  }
  // Cell of a position inside the drawn map (fractions 0..1 across and down), or the nearest filled cell within `reach` cells.
  function pickAt(g, fx, fy, reach) {
    if (!g) return null;
    var cx = Math.max(0, Math.min(g.width - 1, Math.floor(fx * g.width))), cy = Math.max(0, Math.min(g.height - 1, Math.floor(fy * g.height)));
    var best = -1, bd = Infinity, r = reach === undefined ? 2 : reach;
    for (var dy = -r; dy <= r; dy++) {
      for (var dx = -r; dx <= r; dx++) {
        var x = cx + dx, y = cy + dy;
        if (x < 0 || y < 0 || x >= g.width || y >= g.height) continue;
        var k = y * g.width + x;
        if (g.arg[k] < 0) continue;
        var d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = k; }
      }
    }
    if (best < 0) return null;
    return {cell: best, index: g.arg[best], value: g.max[best], x: best % g.width, y: Math.floor(best / g.width)};
  }
  // Where a point (s, θ) lands: fractions across / down, same rule as grid().
  function fractionOf(g, s, theta) {
    var span = (g.sMax - g.sMin) || 1;
    return {fx: Math.max(0, Math.min(1, (s - g.sMin) / span)), fy: Math.max(0, Math.min(1, 1 - (theta + Math.PI) / TWO_PI))};
  }
  function sAt(g, fx) { return g.sMin + Math.max(0, Math.min(1, fx)) * (g.sMax - g.sMin); }
  function thetaAt(g, fy) { return (1 - Math.max(0, Math.min(1, fy))) * TWO_PI - Math.PI; }
  function degrees(theta) { return theta * 180 / Math.PI; }

  // Paint the grid into a canvas of the grid's own size (the page scales it with CSS, pixelated); empty cells stay transparent.
  function paint(canvas, g, scale) {
    if (!canvas || !g) return false;
    canvas.width = g.width; canvas.height = g.height;
    var ctx = canvas.getContext ? canvas.getContext('2d') : null;
    if (!ctx || typeof ctx.createImageData !== 'function') return false;
    var img = ctx.createImageData(g.width, g.height), d = img.data;
    for (var k = 0; k < g.max.length; k++) {
      var v = g.max[k];
      if (!(v > -Infinity)) continue;
      var c = scale ? scale.color(v) : [0.5, 0.5, 0.5];
      d[4 * k] = Math.round(c[0] * 255); d[4 * k + 1] = Math.round(c[1] * 255); d[4 * k + 2] = Math.round(c[2] * 255); d[4 * k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return true;
  }

  // ------------------------------------------------------------------ figure: all branches, labelled, with a colour bar (PNG export)
  // rows = [{name, sMin, sMax, grid}], o = {title, fieldLabel, units, scale, width, rowHeight, doc}
  function figure(rows, o) {
    o = o || {};
    var doc = o.doc || root.document;
    if (!doc || !rows.length) return null;
    var W = o.width || 1600, rowH = o.rowHeight || 150, left = 150, right = 40, top = 70, gap = 46, barH = 18, font = 22;
    var H = top + rows.length * (rowH + gap) + 130;
    var cv = doc.createElement('canvas');
    cv.width = W; cv.height = H;
    var ctx = cv.getContext ? cv.getContext('2d') : null;
    if (!ctx) return null;
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#101828'; ctx.font = '600 ' + (font + 4) + 'px system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif';
    ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
    ctx.fillText(o.title || '', left, 40);
    var plotW = W - left - right;
    rows.forEach(function (r, i) {
      var y0 = top + i * (rowH + gap);
      var img = doc.createElement('canvas');
      if (paint(img, r.grid, o.scale)) {
        ctx.fillStyle = '#eef1f4'; ctx.fillRect(left, y0, plotW, rowH);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(img, left, y0, plotW, rowH);
      }
      ctx.strokeStyle = '#c6ccd2'; ctx.lineWidth = 1; ctx.strokeRect(left + 0.5, y0 + 0.5, plotW - 1, rowH - 1);
      ctx.font = '600 ' + font + 'px system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif'; ctx.fillStyle = '#101828'; ctx.textAlign = 'right';
      ctx.fillText(r.name, left - 14, y0 + rowH / 2 + 8);
      ctx.font = (font - 6) + 'px system-ui, -apple-system, sans-serif'; ctx.fillStyle = '#667085';
      ctx.fillText('+180°', left - 6, y0 + 14); ctx.fillText('−180°', left - 6, y0 + rowH - 2);
      ctx.textAlign = 'left'; ctx.fillText(fmtMm(r.sMin) + ' mm', left, y0 + rowH + 20);
      ctx.textAlign = 'right'; ctx.fillText(fmtMm(r.sMax) + ' mm', left + plotW, y0 + rowH + 20);
    });
    // horizontal colour bar with the scale's own ticks
    var sc = o.scale, by = H - 80;
    if (sc && typeof sc.colorT === 'function') {
      var bw = Math.min(700, plotW);
      for (var x = 0; x < bw; x++) { var c = sc.colorT(x / (bw - 1)); ctx.fillStyle = 'rgb(' + Math.round(c[0] * 255) + ',' + Math.round(c[1] * 255) + ',' + Math.round(c[2] * 255) + ')'; ctx.fillRect(left + x, by, 1, barH); }
      ctx.strokeStyle = '#98a2b3'; ctx.strokeRect(left + 0.5, by + 0.5, bw - 1, barH - 1);
      ctx.font = (font - 5) + 'px system-ui, -apple-system, sans-serif'; ctx.fillStyle = '#344054'; ctx.textAlign = 'center';
      (typeof sc.ticks === 'function' ? sc.ticks(6) : []).forEach(function (t) { var tx = left + t.f * (bw - 1); ctx.fillRect(tx, by + barH, 1, 5); ctx.fillText(t.label, tx, by + barH + 24); });
      ctx.textAlign = 'left'; ctx.fillText((o.fieldLabel || '') + (o.units ? ' · ' + o.units : ''), left + bw + 20, by + barH - 2);
    }
    ctx.font = (font - 6) + 'px system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif'; ctx.fillStyle = '#667085'; ctx.textAlign = 'left';
    ctx.fillText(o.note || '横轴：距入口弧长；纵轴：周向角；每格取落入该格的预测点最大值；空白 = 没有预测点。', left, H - 14);
    return cv;
  }
  function fmtMm(v) { var n = Number(v); return Number.isFinite(n) ? (Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(1)) : '—'; }

  return {supported: supported, requiredArrays: requiredArrays, readKey: readKey, keysOf: keysOf, branches: branches, gridSize: gridSize, grid: grid,
    pickAt: pickAt, fractionOf: fractionOf, sAt: sAt, thetaAt: thetaAt, degrees: degrees, paint: paint, figure: figure};
});
