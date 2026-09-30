/* WSS workspace v2 — volume cross-sections (second phase S1, scope §11).
 * A plane through a volume result: on the centreline (branch + position, perpendicular to it) or through one or two
 * picked wall points, tilted (pitch / yaw) and shifted in plane — the classic volume report's plane definition.
 * Every number comes from the classic report's numerical core (volume_viewer.js → VolumeViewerCore): the slab of
 * prediction points, the wall contour, the filled map (Barnes interior field + ≤ 0.5 mm wall layer, v0.15.12), the
 * point statistics and the area integrals (Voronoi cells clipped to the contour).  The same plane therefore gives the
 * same map and the same readings in both pages.  This module adds the plane state, the 3-D frame with the map drawn
 * on the plane, the inspector panel and the zoom view.  Display only; nothing is written back. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.slice = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var LIMIT = { thickness: [0.2, 12], angle: [-85, 85], offset: [-30, 30], shift: [-25, 25] };
  var GRID = 120, ZOOM_GRID = 240, TEX = 512;
  var LOW_FADE = 0.28;                       // classic cell fallback: cells filled from the wall fade towards white
  var FRAME_HEX = 0x5bb8cc, ACTIVE_HEX = 0xf2a33a;
  var SOURCE_LABEL = { section: '本截面 p2–p98', global: '与三维同', manual: '手动', fallback: '本截面样本不足，用三维色标' };
  var QUANTITIES = [{ id: 'speed', label: '速度' }, { id: 'normal', label: '穿面速度' }, { id: 'pressure', label: '压力' }];

  function core() {
    var c = root.VolumeViewerCore;
    if (!c || typeof c.sliceMapCore !== 'function') throw new Error('volume_viewer.js（VolumeViewerCore）没有加载，截面不可用');
    return c;
  }
  function common() { return root.WssReportCommon || null; }
  function U() { return ns.util; }
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
  function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function mul(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function norm(a) { return Math.hypot(a[0], a[1], a[2]); }
  function point(A, i) { return [A[3 * i], A[3 * i + 1], A[3 * i + 2]]; }
  function fmt(v) { return U() ? U().fmtSig(v) : String(+(+v).toPrecision(3)); }

  // ------------------------------------------------------------------ data
  function fieldKey(m, id, kind) {
    var f = ((m && m.fields) || []).filter(function (x) { return x && x.id === id; })[0];
    return f && f.arrays && typeof f.arrays[kind] === 'string' ? f.arrays[kind] : null;
  }
  function keys(m) {
    var g = (m && m.geometry) || {}, dm = g.display_mesh || {}, p = g.points || {}, cl = g.centerline || {};
    return {
      pts: p.xyz || null, walls: p.is_wall || null, segs: p.segment || null, V: dm.vertices || null, F: dm.faces || null,
      cv: cl.xyz || null, cs: cl.segment || null, ct: cl.tangent || null, ce: cl.edges || null, cr: cl.radius_mm || null,
      speed: fieldKey(m, 'speed', 'read'), pressure: fieldKey(m, 'pressure', 'read'), velocity: fieldKey(m, 'velocity', 'read'),
      wallP: fieldKey(m, 'wall_pressure', 'display')
    };
  }
  function declared(result, key) { return typeof key === 'string' && (!result.declared || result.declared(key)); }
  function supported(result) {
    if (!result || !result.manifest || (result.family || (result.manifest.result || {}).family) !== 'volume') return false;
    var k = keys(result.manifest);
    return [k.pts, k.V, k.F, k.cv].every(function (x) { return declared(result, x); }) &&
      [k.velocity, k.speed, k.pressure].some(function (x) { return declared(result, x); });
  }
  function requiredArrays(result) {
    var k = keys(result.manifest);
    return Object.keys(k).map(function (n) { return k[n]; }).filter(function (x) { return declared(result, x); });
  }
  // Arrays, centreline groups (classic grouping, exported tangents), bounds and branch names of one result.
  function model(result) {
    var m = result.manifest, k = keys(m), A = {}, c = core();
    Object.keys(k).forEach(function (n) { A[n] = declared(result, k[n]) && result.has(k[n]) ? result.array(k[n]) : null; });
    if (!A.pts || !A.V || !A.F) throw new Error('这份体场结果缺少预测点或壁面网格');
    // |v| exactly as the classic report computes it (Math.hypot → float32); the server's speed is the fallback.
    A.speedCalc = A.velocity ? c.speedField(A.velocity) : A.speed;
    var groups = A.cv && A.cv.length ? c.groupCenterline(A.cv, A.cs, A.ct, A.ce) : [];
    var mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (var i = 0; i < A.V.length; i++) { var j = i % 3; if (A.V[i] < mn[j]) mn[j] = A.V[i]; if (A.V[i] > mx[j]) mx[j] = A.V[i]; }
    var names = {};
    ((m.geometry && m.geometry.branches) || []).forEach(function (b) { if (b && b.name) names[b.id] = b.name; });
    return { result: result, manifest: m, A: A, groups: groups, names: names, diag: Math.max(norm(sub(mx, mn)), 1) };
  }
  function groupOf(M, sid) { for (var i = 0; i < M.groups.length; i++) if (M.groups[i].segment === Number(sid)) return M.groups[i]; return null; }
  function arcOf(g) { return g && g.arc.length ? g.arc[g.arc.length - 1] : 0; }
  function branchName(M, sid) { return sid === null || sid === undefined ? '—' : (M.names[sid] || ('分支 ' + sid)); }
  function branchList(M) {
    return M.groups.filter(function (g) { return arcOf(g) > 0; }).map(function (g) { return { id: g.segment, name: branchName(M, g.segment), length_mm: arcOf(g) }; });
  }
  // Frame size (classic gizmoSide): 8 × the inscribed radius of the nearest centreline sample, 24 mm … half the diagonal.
  function gizmoSide(M, p) {
    var best = Infinity, row = null;
    M.groups.forEach(function (g) { for (var i = 0; i < g.points.length; i++) { var d = norm(sub(g.points[i], p)); if (d < best) { best = d; row = g.rows[i]; } } });
    var r = row !== null && M.A.cr && Number.isFinite(M.A.cr[row]) ? M.A.cr[row] : M.diag / 40;
    return clamp(r * 8, 24, M.diag * 0.5);
  }

  // ------------------------------------------------------------------ plane state
  // hint: {xyz (a point to start at: the aneurysm's largest section, the cursor), field}
  function defaultState(M, hint) {
    hint = hint || {};
    var A = M.A;
    var st = { basis: 'centerline', segment: null, fraction: 0.5, pick: null, picks: [], shift: 0, pitch: 0, yaw: 0, offU: 0, offV: 0,
      thickness: 2, quantity: 'speed', range: 'section', manual: { min: null, max: null }, fill: true, arrows: true };
    var near = Array.isArray(hint.xyz) && M.groups.length ? core().nearestTangent(M.groups, hint.xyz) : null;
    var g = near ? groupOf(M, near.segment) : null;
    if (g && arcOf(g) > 0) { st.segment = g.segment; st.fraction = clamp(near.arc / arcOf(g), 0, 1); }
    else {
      var longest = null;
      M.groups.forEach(function (x) { if (!longest || arcOf(x) > arcOf(longest)) longest = x; });
      if (longest) st.segment = longest.segment;
    }
    var hasSpeed = Boolean(A.speedCalc);
    st.quantity = (hint.field === 'pressure' || hint.field === 'wall_pressure' || !hasSpeed) && A.pressure ? 'pressure' : 'speed';
    return st;
  }
  // The classic currentPlane(): base plane (centreline or picks), pitch / yaw about its own axes, then the in-plane offset.
  function planeOf(st, M) {
    var c = core(), base;
    if (st.basis === 'pick' && st.pick) base = { origin: add(st.pick.origin, mul(st.pick.normal, st.shift || 0)), normal: st.pick.normal.slice(), segment: null };
    else {
      var g = groupOf(M, st.segment) || M.groups[0];
      if (!g) return null;
      base = c.centerlinePlane(g, st.fraction);
    }
    var r = c.rotatePlane(base.normal, st.pitch || 0, st.yaw || 0);
    var plane = { origin: base.origin.slice(), normal: r.normal, u: r.u, v: r.v, segment: base.segment === undefined ? null : base.segment };
    plane.origin = add(plane.origin, add(mul(plane.u, st.offU || 0), mul(plane.v, st.offV || 0)));
    return plane;
  }
  function effectiveQuantity(st, A) {
    var q = st.quantity;
    if (q === 'normal' && !A.velocity) q = 'speed';
    if (q !== 'pressure' && !A.speedCalc) q = 'pressure';
    if (q === 'pressure' && !A.pressure) q = A.speedCalc ? 'speed' : null;
    return q;
  }
  // Colour range of the map: 本截面 = p2–p98 of this section's samples (±p98 |v·n| when signed, blue-white-red),
  // 与三维同 = the viewer's scale of that field, 手动 = typed bounds (raw units).  Only colours depend on it.
  function scaleFor(st, quantity, samples, o) {
    o = o || {};
    var c = core(), signed = quantity === 'normal', cmap = o.cmap || 'rainbow';
    var out = function (r, source) { return { min: r.min, max: r.max, log: Boolean(r.log) && !signed, diverging: signed, source: source, map: cmap, count: r.count || null }; };
    var g = null;
    if (typeof o.global === 'function') { try { g = o.global(quantity); } catch (_) { g = null; } }
    if (st.range === 'manual') {
      var lo = Number(st.manual && st.manual.min), hi = Number(st.manual && st.manual.max);
      if (st.manual && st.manual.min !== null && st.manual.max !== null && Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) return out({ min: lo, max: hi }, 'manual');
    }
    if (st.range === 'global' && g) return out(g, 'global');
    var r = signed ? c.symmetricRange(samples) : c.robustRange(samples);
    if (r) return out(r, 'section');
    return out(g || { min: 0, max: 1 }, 'fallback');
  }
  // Closed outline → area, largest chord, equivalent diameter (classic computeSliceSection).
  function sectionOf(data) {
    var cc = common();
    if (!cc || typeof cc.loopPolygon !== 'function' || typeof cc.sectionMetrics !== 'function') return null;
    if (!data || !data.loop || data.loop.open || !Array.isArray(data.contour) || data.contour.length < 3) return null;
    try {
      var poly = cc.loopPolygon(data.contour), m = poly && poly.length >= 3 ? cc.sectionMetrics(poly) : null;
      return m && Number.isFinite(m.area_mm2) && m.area_mm2 > 0 ? Object.assign({}, m, { synthetic: Boolean(data.loop.synthetic) }) : null;
    } catch (_) { return null; }
  }
  // Area integrals over the outline (classic sectionIntegralAt, on this plane): every interior point of the slab inside
  // the outline stands for its Voronoi cell clipped to the lumen (64 × 64 grid); flow Q = mean(v·n) × area.
  function integrate(M, plane, contour, thickness, area) {
    var c = core(), A = M.A, xy = [], idx = [];
    c.slabIndices(A.pts, A.walls, A.segs, plane, thickness).forEach(function (i) {
      var q = sub(point(A.pts, i), plane.origin), x = dot(q, plane.u), y = dot(q, plane.v);
      if (c.pointInLoop(contour, x, y)) { xy.push([x, y]); idx.push(i); }
    });
    var quantities = {};
    if (A.velocity) {
      quantities.speed = idx.map(function (i) { return A.speedCalc[i]; });
      quantities.normal = idx.map(function (i) { return dot(point(A.velocity, i), plane.normal); });
    }
    if (A.pressure) quantities.pressure = idx.map(function (i) { return A.pressure[i]; });
    var r = c.sectionIntegral(contour, xy, quantities, { grid: 64 }), q = r ? r.quantities : {};
    var mean = function (k) { return q[k] && Number.isFinite(q[k].mean) ? q[k].mean : null; };
    return { integrated: Boolean(r), area_mm2: area, samples: idx.length, speed_mean_m_s: mean('speed'), normal_mean_m_s: mean('normal'),
      flow_ml_s: mean('normal') === null ? null : mean('normal') * area, pressure_mean_pa: mean('pressure'), direct_fraction: r ? r.supported : null };   // m/s × mm² = mL/s
  }
  // Everything the panel, the 3-D map and the zoom view show for one state.  o = {cmap, global(quantity)}.
  function compute(st, M, o) {
    var c = core(), A = M.A, plane = planeOf(st, M);
    if (!plane) return null;
    var quantity = effectiveQuantity(st, A);
    if (!quantity) return null;
    var field = quantity === 'pressure' ? 'pressure' : 'velocity';
    // the centreline basis keeps the slab on its branch (classic behaviour); picked planes take every branch
    var filter = st.basis !== 'pick' && A.segs && plane.segment !== null ? plane.segment : null;
    var indices = c.slabIndices(A.pts, A.walls, A.segs, plane, st.thickness, filter);
    var values = quantity === 'normal' ? c.throughPlane(A.velocity, indices, plane.normal) : field === 'velocity' ? A.speedCalc : A.pressure;
    var side = gizmoSide(M, plane.origin);
    var opts = { fill: true, isVelocityField: field === 'velocity', vertices: A.V, faces: A.F, wallValues: A.wallP, loopRadius: Math.max(side * 0.5, 8) };
    var outline = c.sliceMapCore(A.pts, indices, values, plane, opts);
    var data = st.fill === false ? c.sliceMapCore(A.pts, indices, values, plane, Object.assign({}, opts, { fill: false })) : outline;
    var samples = data.finite.map(function (x) { return x.v; });
    var range = scaleFor(st, quantity, samples, o);
    var section = sectionOf(outline);
    var integral = section ? integrate(M, plane, outline.contour, st.thickness, section.area_mm2) : null;
    return {
      plane: plane, quantity: quantity, field: field, indices: indices, values: values, data: data, range: range, side: side,
      stats: c.statistics(field === 'velocity' ? A.speedCalc : A.pressure, indices),   // classic: points equal, on the raw field
      section: section, integral: integral, grid: c.sliceGrid(data, GRID, GRID), gridN: GRID, branchFilter: filter,
      outlineState: !outline.loop ? 'none' : outline.loop.open ? 'open' : outline.loop.synthetic ? 'synthetic' : 'closed'
    };
  }

  // ------------------------------------------------------------------ 2-D map (classic renderSliceMap, less the chrome)
  // o = {background, pad, grid, points, arrows, velocity, maxArrows, arrowWidth, lineWidth, outline, scaleBar, font, colorbar, caption}
  function paint(canvas, comp, o) {
    o = o || {};
    var ctx = canvas && canvas.getContext ? canvas.getContext('2d') : null;
    if (!ctx || !comp) return null;
    var c = core(), W = canvas.width, H = canvas.height, data = comp.data, b = data.bounds;
    var pad = o.pad || { left: 8, top: 8, right: 8, bottom: 8 };
    var plot = { left: pad.left, top: pad.top, width: W - pad.left - pad.right, height: H - pad.top - pad.bottom };
    var xmin = b[0], xmax = b[1], ymin = b[2], ymax = b[3];
    var scale = Math.min(plot.width / (xmax - xmin), plot.height / (ymax - ymin));
    var cx = plot.left + plot.width / 2 - (xmin + xmax) / 2 * scale, cy = plot.top + plot.height / 2 + (ymin + ymax) / 2 * scale;
    var toX = function (x) { return cx + x * scale; }, toY = function (y) { return cy - y * scale; };
    ctx.clearRect(0, 0, W, H);
    if (o.background) { ctx.fillStyle = o.background; ctx.fillRect(0, 0, W, H); }
    var grid = o.grid || comp.grid, nx = grid.nx || comp.gridN, ny = grid.ny || comp.gridN;
    var painted = Boolean(grid.stats && grid.display) && c.paintSection(ctx, W, H, grid, data.contour, b, comp.range, scale, cx, cy);
    if (!painted) {
      var cw = (xmax - xmin) / nx * scale, ch = (ymax - ymin) / ny * scale;
      for (var iy = 0; iy < ny; iy++) for (var ix = 0; ix < nx; ix++) {
        var at = iy * nx + ix;
        if (!grid.mask || !grid.mask[at] || !Number.isFinite(grid.values[at])) continue;
        var col = c.scaleColor(grid.values[at], comp.range);
        if (grid.low && grid.low[at]) col = col.map(function (v) { return v + (1 - v) * LOW_FADE; });
        ctx.fillStyle = 'rgb(' + col.map(function (v) { return Math.round(v * 255); }).join(',') + ')';
        ctx.fillRect(toX(xmin + ix * (xmax - xmin) / nx), toY(ymin + (iy + 1) * (ymax - ymin) / ny), cw + 1, ch + 1);
      }
    }
    if (data.contour.length && o.outline !== false) {
      ctx.strokeStyle = o.outline || '#33475b'; ctx.lineWidth = o.lineWidth || 1.4; ctx.beginPath();
      data.contour.forEach(function (sg) { if (sg[5] >= 0) { ctx.moveTo(toX(sg[0]), toY(sg[1])); ctx.lineTo(toX(sg[2]), toY(sg[3])); } });
      ctx.stroke();
      if (ctx.setLineDash) {
        ctx.setLineDash([6, 4]); ctx.beginPath();
        data.contour.forEach(function (sg) { if (sg[5] < 0) { ctx.moveTo(toX(sg[0]), toY(sg[1])); ctx.lineTo(toX(sg[2]), toY(sg[3])); } });
        ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.lineWidth = 1;
    }
    if (o.points) {
      var r0 = o.pointRadius || 2.7;
      data.finite.forEach(function (x) {
        var pc = c.scaleColor(comp.values[x.i], comp.range);
        ctx.fillStyle = 'rgb(' + pc.map(function (v) { return Math.round(v * 255); }).join(',') + ')'; ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(toX(x.p[0]), toY(x.p[1]), r0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      });
    }
    // In-plane flow: the samples' velocity projected on (u, v), one per grid cell, length ∝ in-plane speed / the section's p95.
    var arrowCount = 0;
    if (o.arrows && o.velocity && comp.field === 'velocity' && data.finite.length) {
      var items = data.finite.map(function (x) { var pv = c.inPlane(point(o.velocity, x.i), comp.plane); return { x: x.p[0], y: x.p[1], du: pv[0], dv: pv[1] }; });
      var as = c.arrowSamples(items, b, o.maxArrows || 120), maxLen = plot.width * 0.06, lw = o.arrowWidth || 1.4;
      if (as.ref > 0) {
        var segs = [];
        as.arrows.forEach(function (a) {
          var len = Math.min(1, a.mag / as.ref) * maxLen;
          if (len < 1.5) return;
          var ux = a.du / a.mag, uy = -a.dv / a.mag, x0 = toX(a.x) - ux * len / 2, y0 = toY(a.y) - uy * len / 2, x1 = x0 + ux * len, y1 = y0 + uy * len;
          var hh = Math.max(3, len * 0.35), cs = Math.cos(0.45), sn = Math.sin(0.45);
          segs.push([x0, y0, x1, y1], [x1, y1, x1 - hh * (ux * cs - uy * sn), y1 - hh * (uy * cs + ux * sn)], [x1, y1, x1 - hh * (ux * cs + uy * sn), y1 - hh * (uy * cs - ux * sn)]);
          arrowCount++;
        });
        [['#ffffff', lw + 2], ['#16283a', lw]].forEach(function (s) {
          ctx.strokeStyle = s[0]; ctx.lineWidth = s[1]; ctx.lineCap = 'round'; ctx.beginPath();
          segs.forEach(function (q) { ctx.moveTo(q[0], q[1]); ctx.lineTo(q[2], q[3]); });
          ctx.stroke();
        });
        ctx.lineWidth = 1;
      }
    }
    var font = o.font || 12;
    ctx.font = font + 'px system-ui, sans-serif';
    if (o.scaleBar) {
      var L = c.scaleBarLength(1 / scale, Math.min(120 * (font / 12), plot.width * 0.22)), px = L * scale, sx = plot.left + 8, sy = plot.top + plot.height - 10;
      ctx.strokeStyle = o.ink || '#33475b'; ctx.fillStyle = o.ink || '#33475b'; ctx.lineWidth = Math.max(2, font / 6);
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + px, sy); ctx.moveTo(sx, sy - 4); ctx.lineTo(sx, sy + 4); ctx.moveTo(sx + px, sy - 4); ctx.lineTo(sx + px, sy + 4); ctx.stroke();
      ctx.lineWidth = 1; ctx.textAlign = 'left'; ctx.fillText(L + ' mm', sx, sy - 6);
    }
    if (o.colorbar) drawColorbar(ctx, W, H, pad, comp, o, font);
    if (o.caption) { ctx.fillStyle = '#536a80'; ctx.textAlign = 'left'; ctx.fillText(o.caption, pad.left, H - pad.bottom + Math.round(font * 1.4), Math.max(80, W * 0.62)); }
    if (!data.finite.length && o.emptyText !== false) { ctx.textAlign = 'center'; ctx.fillStyle = '#8093a2'; ctx.fillText('此厚度内没有体内预测点', W / 2, H / 2); ctx.textAlign = 'left'; }
    return { plot: plot, bounds: b, scale: scale, cx: cx, cy: cy, grid: grid, nx: nx, ny: ny, arrows: arrowCount };
  }
  function drawColorbar(ctx, W, H, pad, comp, o, font) {
    var c = core(), range = comp.range, barW = Math.max(120, Math.round(W * 0.26)), barH = Math.max(10, Math.round(font * 0.9));
    var barX = W - pad.right - barW, barY = H - pad.bottom + Math.round(font * 2.6);
    for (var i = 0; i < barW; i++) {
      var col = c.colorAtT(i / Math.max(1, barW - 1), range);
      ctx.fillStyle = 'rgb(' + col.map(function (v) { return Math.round(v * 255); }).join(',') + ')';
      ctx.fillRect(barX + i, barY, 1, barH);
    }
    ctx.strokeStyle = '#8093a2'; ctx.strokeRect(barX, barY, barW, barH); ctx.fillStyle = '#536a80';
    var ends = c.scaleEnds(range);
    ctx.textAlign = 'left'; ctx.fillText(fmt(ends[0]), barX, barY - 3);
    ctx.textAlign = 'right'; ctx.fillText(fmt(ends[1]), barX + barW, barY - 3);
    if (range.diverging) { ctx.textAlign = 'center'; ctx.fillText(fmt(c.scaleValueAt(0.5, range)), barX + barW / 2, barY - 3); }
    ctx.textAlign = 'center'; ctx.fillText(quantityLabel(comp.quantity) + ' · ' + unitsOf(comp.quantity), barX + barW / 2, barY + barH + font + 2);
    ctx.textAlign = 'left';
  }
  // The value the pixel at canvas (px, py) was painted with (bilinear on the display grid), or null.
  function readAt(map, comp, px, py) {
    if (!map || !comp) return null;
    var b = map.bounds, wx = (px - map.cx) / map.scale, wy = (map.cy - py) / map.scale;
    var ix = Math.floor((wx - b[0]) / (b[1] - b[0]) * map.nx), iy = Math.floor((wy - b[2]) / (b[3] - b[2]) * map.ny);
    if (ix < 0 || iy < 0 || ix >= map.nx || iy >= map.ny) return null;
    var g = map.grid, at = iy * map.nx + ix;
    if (!g.mask || !g.mask[at]) return null;
    var v = g.display ? core().bilinearGrid(g.display, map.nx, map.ny, (wx - b[0]) / (b[1] - b[0]) * map.nx - 0.5, (wy - b[2]) / (b[3] - b[2]) * map.ny - 0.5) : g.values[at];
    return Number.isFinite(v) ? { x_mm: wx, y_mm: wy, value: v, fromWall: Boolean(g.low && g.low[at]) } : null;
  }
  function quantityLabel(q) { return q === 'normal' ? '穿面速度' : q === 'pressure' ? '相对压力' : '速度大小'; }
  function unitsOf(q) { return q === 'pressure' ? 'Pa' : 'm/s'; }

  // ------------------------------------------------------------------ 3-D: the frame and the map on the plane
  function overlay(viewer, THREE) {
    var group = new THREE.Group(); group.name = 'slice';
    var quadMat = new THREE.MeshBasicMaterial({ color: FRAME_HEX, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false });
    var quad = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), quadMat); quad.renderOrder = 4;
    var edgeMat = new THREE.LineBasicMaterial({ color: FRAME_HEX, transparent: true, opacity: 0.6, depthTest: false });
    var edge = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]].map(function (q) { return new THREE.Vector3(q[0], q[1], q[2]); })), edgeMat);
    edge.renderOrder = 9;
    var arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 1, FRAME_HEX, 0.2, 0.1);
    var frame = new THREE.Group(); frame.add(quad); frame.add(edge); frame.add(arrow); group.add(frame);
    var doc = root.document, canvas = doc && doc.createElement ? doc.createElement('canvas') : null, tex = null, map = null;
    if (canvas && canvas.getContext) {
      canvas.width = TEX; canvas.height = TEX;
      tex = new THREE.CanvasTexture(canvas);
      map = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
      map.renderOrder = 2; map.matrixAutoUpdate = false; group.add(map);
    }
    var parent = null;
    function attach() { var p = viewer.toolOverlay(); if (p !== parent || !group.parent) { p.add(group); parent = p; } }
    function update(comp, M) {
      attach();
      if (!comp) { group.visible = false; viewer.render(); return; }
      group.visible = true;
      var P = comp.plane, n = new THREE.Vector3(P.normal[0], P.normal[1], P.normal[2]);
      frame.position.set(P.origin[0], P.origin[1], P.origin[2]);
      frame.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
      quad.scale.set(comp.side, comp.side, 1); edge.scale.set(comp.side, comp.side, 1);
      // the normal arrow follows the lumen, not the (larger) grab frame
      var bb = comp.data.bounds, ext = clamp(Math.max(bb[1] - bb[0], bb[3] - bb[2]) * 0.5, 6, comp.side * 0.55);
      arrow.setLength(ext, ext * 0.22, ext * 0.12);
      if (map) {
        var b = comp.data.bounds, bw = b[1] - b[0], bh = b[3] - b[2], mx = (b[0] + b[1]) / 2, my = (b[2] + b[3]) / 2;
        paint(canvas, comp, { pad: { left: 0, top: 0, right: 0, bottom: 0 }, outline: 'rgba(255,255,255,0.92)', lineWidth: 3, emptyText: false });
        tex.needsUpdate = true;
        var u = new THREE.Vector3(P.u[0], P.u[1], P.u[2]).multiplyScalar(bw), v = new THREE.Vector3(P.v[0], P.v[1], P.v[2]).multiplyScalar(bh);
        var ctr = add(P.origin, add(mul(P.u, mx), mul(P.v, my)));
        map.matrix.makeBasis(u, v, n); map.matrix.setPosition(ctr[0], ctr[1], ctr[2]);
        map.matrixWorldNeedsUpdate = true;
        map.visible = comp.data.finite.length > 0 || comp.data.contour.length > 0;
      }
      viewer.render();
    }
    function style(active) {
      quadMat.opacity = active ? 0.2 : 0.08;
      edgeMat.color.setHex(active ? ACTIVE_HEX : FRAME_HEX); edgeMat.opacity = active ? 0.95 : 0.6;
      arrow.setColor(new THREE.Color(active ? ACTIVE_HEX : FRAME_HEX));
      viewer.render();
    }
    function hit(clientX, clientY) {
      if (!group.visible || !group.parent) return false;
      var ray = viewer.rayAt(clientX, clientY);
      return Boolean(ray && ray.intersectObject(quad, false).length);
    }
    function dispose() {
      if (group.parent) group.parent.remove(group);
      [quad.geometry, edge.geometry, quadMat, edgeMat].forEach(function (x) { x.dispose(); });
      if (map) { map.geometry.dispose(); map.material.dispose(); tex.dispose(); }
      arrow.traverse(function (o) { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
      viewer.render();
    }
    return { update: update, style: style, hit: hit, dispose: dispose, group: group };
  }

  // ------------------------------------------------------------------ session (one result, one viewer)
  // opts = {hint, state, cmap(), global(quantity), onChange(what), onNote(text)}
  function create(viewer, result, opts) {
    opts = opts || {};
    var M = model(result), THREE = root.THREE;
    var st = opts.state ? Object.assign(defaultState(M, opts.hint), JSON.parse(JSON.stringify(opts.state))) : defaultState(M, opts.hint);
    if (st.basis === 'pick' && !st.pick) st.basis = 'centerline';
    if (!groupOf(M, st.segment) && st.basis !== 'pick') st = defaultState(M, opts.hint);
    var comp = null, picking = false, hover = false, dragging = false, raf = 0, disposed = false, panels = [], pickNote = '', pickedAt = 0;
    var ov = THREE && viewer && typeof viewer.toolOverlay === 'function' ? overlay(viewer, THREE) : null;
    var detach = viewer && viewer.canvasElement ? bind() : function () {};

    function emit(what) { if (typeof opts.onChange === 'function') { try { opts.onChange(what); } catch (_) {} } }
    function recompute() {
      raf = 0;
      if (disposed) return;
      try { comp = compute(st, M, { cmap: typeof opts.cmap === 'function' ? opts.cmap() : 'rainbow', global: opts.global }); }
      catch (e) { comp = null; if (opts.onNote) opts.onNote('截面计算失败：' + (e && e.message || e)); }
      if (ov) ov.update(comp, M);
      panels = panels.filter(function (p) { return p.alive(); });
      panels.forEach(function (p) { p.update(); });
      emit('slice');
    }
    function schedule() {
      if (disposed || raf) return;
      var rq = root.requestAnimationFrame;
      if (typeof rq === 'function') raf = rq(recompute); else { raf = -1; recompute(); }
    }
    function set(partial) { Object.assign(st, partial || {}); schedule(); }
    function moveAlong(dMm) {
      if (!dMm) return;
      if (st.basis === 'pick' && st.pick) st.shift = clamp((st.shift || 0) + dMm, LIMIT.shift[0], LIMIT.shift[1]);
      else { var g = groupOf(M, st.segment); var L = arcOf(g); if (L > 0) st.fraction = clamp(st.fraction + dMm / L, 0, 1); }
      schedule();
    }
    function rotate(dYaw, dPitch) {
      st.yaw = clamp((st.yaw || 0) + (dYaw || 0), LIMIT.angle[0], LIMIT.angle[1]);
      st.pitch = clamp((st.pitch || 0) + (dPitch || 0), LIMIT.angle[0], LIMIT.angle[1]);
      schedule();
    }
    function nudgeThickness(d) { st.thickness = +clamp((st.thickness || 2) + d, LIMIT.thickness[0], LIMIT.thickness[1]).toFixed(2); schedule(); }
    function screenDir(origin, axis) {
      var a = viewer.projectPoint(origin), b = viewer.projectPoint(add(origin, axis));
      return [b.x - a.x, b.y - a.y];
    }
    // Drag on the frame: move along the normal (default), rotate (Shift) or slide in the plane (Alt) — classic applyPlaneDrag.
    function drag(kind, dx, dy) {
      var c = core(), P = comp ? comp.plane : planeOf(st, M);
      if (!P) return;
      if (kind === 'rotate') { rotate(dx * 0.3, -dy * 0.3); return; }
      var px = viewer.mmPerPixelAt(P.origin);
      if (kind === 'offset') {
        st.offU = clamp((st.offU || 0) + c.dragAlong(dx, dy, screenDir(P.origin, P.u), px), LIMIT.offset[0], LIMIT.offset[1]);
        st.offV = clamp((st.offV || 0) + c.dragAlong(dx, dy, screenDir(P.origin, P.v), px), LIMIT.offset[0], LIMIT.offset[1]);
        schedule(); return;
      }
      moveAlong(c.dragAlong(dx, dy, screenDir(P.origin, P.normal), px));
    }
    function setPicking(on) {
      picking = Boolean(on);
      if (picking) { st.picks = []; pickNote = '点血管壁放第 1 点：截面垂直于那里的中心线'; }
      else pickNote = '';
      if (ov) ov.style(false);
      emit('picking');
      panels.forEach(function (p) { if (p.alive()) p.update(); });
    }
    function addPick(xyz) {
      var c = core();
      if (!Array.isArray(st.picks) || st.picks.length >= 2) st.picks = [];
      st.picks.push(xyz.slice());
      var near = c.nearestTangent(M.groups, xyz) || { tangent: [0, 0, 1], segment: null };
      var pl = c.planeFromPicks(st.picks, near.tangent);
      Object.assign(st, { basis: 'pick', pick: { origin: pl.origin, normal: pl.normal, picks: pl.picks, chord_mm: pl.chord_mm || null, segment: near.segment },
        shift: 0, pitch: 0, yaw: 0, offU: 0, offV: 0 });
      if (st.picks.length >= 2) { picking = false; pickNote = ''; }
      else pickNote = '再点一点，截面就过这两点；Esc 结束点选';
      schedule(); emit('picking');
    }
    // Back on the centreline at the station nearest to the current plane origin.
    function toCenterline() {
      var P = comp ? comp.plane : planeOf(st, M), near = P ? core().nearestTangent(M.groups, P.origin) : null, g = near ? groupOf(M, near.segment) : null;
      Object.assign(st, { basis: 'centerline', pick: null, picks: [], shift: 0, pitch: 0, yaw: 0, offU: 0, offV: 0 });
      if (g && arcOf(g) > 0) { st.segment = g.segment; st.fraction = clamp(near.arc / arcOf(g), 0, 1); }
      schedule();
    }
    function reset() { Object.assign(st, { pitch: 0, yaw: 0, offU: 0, offV: 0, shift: 0 }); schedule(); }
    // Turns the view to 45° above the section's upstream face (the normal points downstream on the centreline),
    // from the side the camera is on now, and comes closer when the vessel is far larger than the section.
    function look(animate) {
      if (!viewer || !comp || typeof viewer.getCamera !== 'function' || typeof viewer.setCamera !== 'function') return false;
      var cam = viewer.getCamera(), P = comp.plane, n = P.normal;
      if (!cam || !cam.position || !cam.target) return false;
      var off = sub(cam.position, cam.target), d0 = norm(off) || 300;
      var side = sub(off, mul(n, dot(off, n))), sl = norm(side), sgn = dot(off, n) > 0.2 * d0 ? 1 : -1;
      var s1 = sl > 1e-6 ? mul(side, 1 / sl) : P.u, a = Math.PI / 4;
      var dir = add(mul(s1, Math.cos(a)), mul(n, sgn * Math.sin(a))), b = comp.data.bounds;
      var d = clamp(Math.max(b[1] - b[0], b[3] - b[2]) * 4, 120, Math.max(d0, 120));
      viewer.setCamera({ position: add(P.origin, mul(dir, d)), target: P.origin.slice(), up: cam.up }, { animate: animate !== false });
      return true;
    }
    function key(e) {
      if (!e || e.ctrlKey || e.metaKey || e.altKey) return false;
      var k = e.key, step = e.shiftKey ? 5 : 1, deg = e.shiftKey ? 10 : 2;
      if (k === 'Escape') { if (picking) { setPicking(false); return true; } return false; }
      if (k === 'ArrowUp') moveAlong(step);
      else if (k === 'ArrowDown') moveAlong(-step);
      else if (k === 'ArrowLeft') rotate(-deg, 0);
      else if (k === 'ArrowRight') rotate(deg, 0);
      else if (k === 'PageUp') rotate(0, deg);
      else if (k === 'PageDown') rotate(0, -deg);
      else if (k === '[') nudgeThickness(-0.4);
      else if (k === ']') nudgeThickness(0.4);
      else return false;
      return true;
    }
    // Pointer handling on the viewer's canvas (capture phase, like the classic gizmo): a press on the frame drags the
    // plane instead of orbiting; the wheel over the frame moves it (Shift: thickness); in point mode a click places a point.
    function bind() {
      var cv = viewer.canvasElement, drag0 = null, press = null, lastHover = 0;
      function onDown(e) {
        if (e.button !== 0 || disposed) return;
        press = { x: e.clientX, y: e.clientY, id: e.pointerId };
        if (picking || !ov || !ov.hit(e.clientX, e.clientY)) return;
        drag0 = { x: e.clientX, y: e.clientY, id: e.pointerId, kind: e.shiftKey ? 'rotate' : e.altKey ? 'offset' : 'move' };
        dragging = true;
        viewer.setControlsEnabled(false);
        try { if (cv.setPointerCapture) cv.setPointerCapture(e.pointerId); } catch (_) {}
        e.preventDefault(); e.stopImmediatePropagation();
        ov.style(true); cv.style.cursor = 'grabbing';
      }
      function onMove(e) {
        if (drag0 && drag0.id === e.pointerId) {
          var dx = e.clientX - drag0.x, dy = e.clientY - drag0.y;
          drag0.x = e.clientX; drag0.y = e.clientY;
          drag(drag0.kind, dx, dy);
          e.preventDefault(); e.stopImmediatePropagation();
          return;
        }
        if (e.buttons || picking || !ov) return;
        var now = Date.now();
        if (now - lastHover < 50) return;
        lastHover = now;
        var on = ov.hit(e.clientX, e.clientY);
        if (on !== hover) { hover = on; ov.style(on); cv.style.cursor = on ? 'grab' : ''; }
      }
      function onUp(e) {
        if (drag0 && drag0.id === e.pointerId) {
          drag0 = null; dragging = false;
          viewer.setControlsEnabled(true);
          if (ov) ov.style(hover);
          cv.style.cursor = hover ? 'grab' : '';
        }
        var click = press && press.id === e.pointerId && Math.hypot(e.clientX - press.x, e.clientY - press.y) < 6;
        press = null;
        if (click && picking) {
          pickedAt = Date.now();   // the viewer's own click (a reading) follows this one: the shell skips it
          var hit = viewer.surfaceAt(e.clientX, e.clientY);
          if (hit && hit.xyz) addPick(hit.xyz);
          else if (opts.onNote) opts.onNote('没有点到血管壁。请点在半透明的血管上。');
        }
      }
      function onWheel(e) {
        if (picking || disposed || !ov) return;
        if (!hover && !e.altKey && !ov.hit(e.clientX, e.clientY)) return;
        e.preventDefault(); e.stopImmediatePropagation();
        if (e.shiftKey) { nudgeThickness(e.deltaY > 0 ? 0.4 : -0.4); return; }
        moveAlong((e.deltaY > 0 ? -1 : 1) * (e.ctrlKey || e.metaKey ? 5 : 1));
      }
      var wheelOpts = { passive: false, capture: true };
      cv.addEventListener('pointerdown', onDown, true);
      cv.addEventListener('pointermove', onMove, true);
      cv.addEventListener('pointerup', onUp, true);
      cv.addEventListener('pointercancel', onUp, true);
      cv.addEventListener('wheel', onWheel, wheelOpts);
      return function () {
        cv.removeEventListener('pointerdown', onDown, true);
        cv.removeEventListener('pointermove', onMove, true);
        cv.removeEventListener('pointerup', onUp, true);
        cv.removeEventListener('pointercancel', onUp, true);
        cv.removeEventListener('wheel', onWheel, wheelOpts);
        cv.style.cursor = '';
        if (dragging) viewer.setControlsEnabled(true);
      };
    }
    // The stage colour bar while the section is shown: the section's own scale.
    function colorbarInfo() {
      if (!comp || !ns.colormap) return null;
      var r = comp.range, ends = core().scaleEnds(r);
      var sc = ns.colormap.scale({ range: ends, log: Boolean(r.log), cmap: r.diverging ? 'bwr' : r.map, units: unitsOf(comp.quantity) });
      var src = SOURCE_LABEL[r.source] || '';
      return { fieldId: 'slice', label: quantityLabel(comp.quantity), fullLabel: '截面 · ' + quantityLabel(comp.quantity), units: unitsOf(comp.quantity), scale: sc,
        histogram: null, thresholds: [], windowLabel: { id: 'slice', kind: r.source, label: '截面色标：' + src + (comp.quantity === 'normal' ? '；顺流为正' : ''), text: src } };
    }
    function hudText() {
      if (picking) return pickNote;
      if (!comp) return '';
      var s = st.basis === 'pick' ? (st.picks.length >= 2 || (st.pick && st.pick.picks === 2) ? '两点斜截面' : '选点处垂直截面') + (st.shift ? ' · 偏移 ' + fmt(st.shift) + ' mm' : '')
        : branchName(M, st.segment) + ' ' + fmt(st.fraction * arcOf(groupOf(M, st.segment))) + ' mm';
      return s + ' · 厚 ' + fmt(st.thickness) + ' mm · 拖动截面框或 ↑↓ 移动，Shift 拖动旋转';
    }
    function dispose() {
      if (disposed) return;
      disposed = true;
      if (raf > 0 && root.cancelAnimationFrame) root.cancelAnimationFrame(raf);
      detach();
      if (ov) ov.dispose();
      panels = [];
    }
    var session = {
      model: M,
      state: function () { return JSON.parse(JSON.stringify(st)); },
      comp: function () { return comp; },
      set: set, moveAlong: moveAlong, rotate: rotate, nudgeThickness: nudgeThickness, drag: drag, reset: reset, toCenterline: toCenterline, look: look,
      picking: function () { return picking; }, setPicking: setPicking, addPick: addPick,
      clickTaken: function () { return picking || Date.now() - pickedAt < 500; },
      dragging: function () { return dragging; },
      key: key, colorbarInfo: colorbarInfo, hudText: hudText, branches: function () { return branchList(M); },
      branchName: function (sid) { return branchName(M, sid); }, arcOf: function (sid) { return arcOf(groupOf(M, sid)); },
      addPanel: function (p) { panels.push(p); }, refresh: schedule, recomputeNow: recompute,
      dispose: dispose, disposed: function () { return disposed; }
    };
    recompute();
    return session;
  }

  // ------------------------------------------------------------------ inspector panel
  // ctx = {ui, onClose, onZoom}
  function panel(body, session, ctx) {
    var ui = ctx.ui, h = ui.h, doc = body.ownerDocument || root.document;
    var dpr = Math.min(2, Math.max(1, root.devicePixelRatio || 1));
    var canvas = h('canvas', { 'class': 'slice-canvas', width: Math.round(320 * dpr), height: Math.round(300 * dpr), 'aria-label': '截面图：悬停读数' });
    var read = h('div', { 'class': 'slice-read' });
    var mapBox = h('div', { 'class': 'slice-map' }, canvas, read);
    var pos = h('div', { 'class': 'slice-pos' });
    var qseg = h('div', { 'class': 'seg seg-slice', role: 'tablist', 'aria-label': '截面物理量' });
    var kpis = h('div', { 'class': 'kpis slice-kpis' });
    var rows = h('div', { 'class': 'slice-rows' });
    var ctrls = h('div', { 'class': 'slice-ctrls' });
    var outlineNote = h('p', { 'class': 'slice-note' });
    var lastMap = null, built = false;
    var pickBtn = ui.button('点选', function () { session.setPicking(!session.picking()); }, { cls: 'btn-sm', title: '在血管壁上点 1 点（垂直中心线）或 2 点（过两点的斜截面）' });
    var head = ui.section('截面', { cls: 'sec-slice', actions: [
      pickBtn,
      ui.iconButton('eye', '看截面：视角转到截面斜上方', function () { session.look(true); }),
      ui.iconButton('refresh', '回正：取消倾斜和偏移', function () { session.reset(); }),
      ui.iconButton('expand', '放大截面图（可导出 PNG / CSV）', function () { if (ctx.onZoom) ctx.onZoom(); }),
      ui.iconButton('close', '关闭截面（S）', function () { if (ctx.onClose) ctx.onClose(); })] },
      pos, mapBox, qseg, kpis, rows, ctrls, outlineNote);
    ui.fill(body, head);

    canvas.addEventListener('pointermove', function (e) {
      var comp = session.comp();
      if (!lastMap || !comp || !canvas.getBoundingClientRect) return;
      var rect = canvas.getBoundingClientRect();
      var r = readAt(lastMap, comp, (e.clientX - rect.left) * canvas.width / Math.max(1, rect.width), (e.clientY - rect.top) * canvas.height / Math.max(1, rect.height));
      read.textContent = r ? fmt(r.value) + ' ' + unitsOf(comp.quantity) + (r.fromWall ? ' · 壁面补全' : '') + ' · (' + r.x_mm.toFixed(1) + ', ' + r.y_mm.toFixed(1) + ') mm' : '';
    });
    canvas.addEventListener('pointerleave', function () { read.textContent = ''; });

    function positionRow(st) {
      if (st.basis === 'pick') {
        var what = st.pick && st.pick.picks === 2 ? '两点斜截面' : '选点处垂直截面';
        return [h('span', { 'class': 'slice-where', text: what }),
          ui.iconButton('chevron-left', '沿法向 −1 mm（↓）', function () { session.moveAlong(-1); }),
          h('span', { 'class': 'slice-s', text: (st.shift >= 0 ? '+' : '') + fmt(st.shift || 0) + ' mm' }),
          ui.iconButton('chevron-right', '沿法向 +1 mm（↑）', function () { session.moveAlong(1); }),
          ui.button('回到中心线', function () { session.toCenterline(); }, { kind: 'link', cls: 'btn-sm' })];
      }
      var list = session.branches();
      var sel = ui.select(list.map(function (b) { return { value: String(b.id), label: b.name }; }), String(st.segment), function (v) { session.set({ segment: Number(v), fraction: 0.5 }); }, { 'aria-label': '分支' });
      var L = session.arcOf(st.segment);
      return [sel,
        ui.iconButton('chevron-left', '向近端 1 mm（↓）', function () { session.moveAlong(-1); }),
        h('span', { 'class': 'slice-s', text: fmt(st.fraction * L) + ' / ' + fmt(L) + ' mm' }),
        ui.iconButton('chevron-right', '向远端 1 mm（↑）', function () { session.moveAlong(1); })];
    }
    function kpi(label, value, units, sub) {
      return h('div', { 'class': 'kpi kpi-static' }, h('span', { 'class': 'kpi-label', text: label }),
        h('span', { 'class': 'kpi-value' }, h('span', { 'class': 'kpi-num', text: value === null || value === undefined ? '—' : fmt(value) }), units ? h('span', { 'class': 'kpi-unit', text: units }) : null),
        sub ? h('span', { 'class': 'kpi-sub', text: sub }) : null);
    }
    function row(label, value, tip) {
      return h('div', { 'class': 'slice-row' }, h('span', { 'class': 'slice-row-k', text: label }), h('b', { text: value }), tip && ui.infoTip ? ui.infoTip(tip) : null);
    }
    function controls(st, comp) {
      var th = h('input', { type: 'range', min: String(LIMIT.thickness[0]), max: String(LIMIT.thickness[1]), step: '0.2', value: String(st.thickness), 'aria-label': '截面厚度' });
      var thv = h('span', { 'class': 'slice-thv', text: fmt(st.thickness) + ' mm' });
      th.addEventListener('input', function () { thv.textContent = fmt(+th.value) + ' mm'; session.set({ thickness: +th.value }); });
      var rsel = ui.select([{ value: 'section', label: '本截面自适应' }, { value: 'global', label: '与三维同' }, { value: 'manual', label: '手动' }], st.range, function (v) {
        var patch = { range: v };
        if (v === 'manual' && (st.manual.min === null || st.manual.max === null) && comp) { var e = core().scaleEnds(comp.range); patch.manual = { min: +e[0].toPrecision(4), max: +e[1].toPrecision(4) }; }
        session.set(patch); build();
      }, { 'aria-label': '截面色标' });
      var manual = null;
      if (st.range === 'manual') {
        var mk = function (k) {
          var inp = h('input', { type: 'number', step: 'any', value: st.manual[k] === null ? '' : String(st.manual[k]), 'aria-label': k === 'min' ? '下限' : '上限', 'class': 'slice-num' });
          inp.addEventListener('change', function () { var m = { min: st.manual.min, max: st.manual.max }; m[k] = inp.value === '' ? null : Number(inp.value); session.set({ manual: m }); });
          return inp;
        };
        manual = h('span', { 'class': 'slice-manual' }, mk('min'), h('span', { text: '–' }), mk('max'), h('span', { 'class': 'muted', text: unitsOf(comp ? comp.quantity : st.quantity) }));
      }
      var check = function (keyName, label, disabled) {
        var cb = h('input', { type: 'checkbox', checked: st[keyName] !== false, disabled: Boolean(disabled) });
        cb.addEventListener('change', function () { var p = {}; p[keyName] = cb.checked; session.set(p); });
        return h('label', { 'class': 'slice-check' + (disabled ? ' off' : '') }, cb, h('span', { text: label }));
      };
      var isVel = comp && comp.field === 'velocity';
      return [h('label', { 'class': 'slice-ctl' }, h('span', { 'class': 'slice-ctl-k', text: '厚度' }), th, thv),
        h('label', { 'class': 'slice-ctl' }, h('span', { 'class': 'slice-ctl-k', text: '色标' }), rsel, manual),
        h('div', { 'class': 'slice-ctl slice-checks' }, check('fill', '补全到管壁'), check('arrows', '面内流向箭头', !isVel))];
    }
    // Structure is rebuilt when the state's shape changes (basis, range mode, quantity); numbers every update.
    var shapeKey = '';
    function build() {
      var st = session.state(), comp = session.comp();
      ui.fill(pos, positionRow(st));
      ui.fill(qseg, QUANTITIES.map(function (q) {
        var A = session.model.A, ok = q.id === 'pressure' ? Boolean(A.pressure) : q.id === 'normal' ? Boolean(A.velocity) : Boolean(A.speedCalc);
        var on = comp ? comp.quantity === q.id : st.quantity === q.id;
        var b = h('button', { type: 'button', role: 'tab', 'class': 'seg-btn' + (on ? ' on' : ''), 'aria-selected': String(on), disabled: !ok, text: q.label,
          title: q.id === 'normal' ? '速度在截面法向上的分量，顺流为正、负值是回流' : null });
        b.addEventListener('click', function () { session.set({ quantity: q.id }); if (ctx.onQuantity) ctx.onQuantity(q.id); });
        return b;
      }));
      ui.fill(ctrls, controls(st, comp));
      shapeKey = [st.basis, st.range, comp && comp.quantity, st.segment, st.pick && st.pick.picks].join('|');
      built = true;
    }
    function update() {
      var st = session.state(), comp = session.comp();
      var key = [st.basis, st.range, comp && comp.quantity, st.segment, st.pick && st.pick.picks].join('|');
      if (!built || key !== shapeKey) build();
      else { var s = pos.querySelector ? pos.querySelector('.slice-s') : null; if (s) s.textContent = st.basis === 'pick' ? (st.shift >= 0 ? '+' : '') + fmt(st.shift || 0) + ' mm' : fmt(st.fraction * session.arcOf(st.segment)) + ' / ' + fmt(session.arcOf(st.segment)) + ' mm';
        else ui.fill(pos, positionRow(st)); }
      if (pickBtn.classList) pickBtn.classList.toggle('on', session.picking());
      pickBtn.setAttribute('aria-pressed', String(session.picking()));
      if (!comp) { ui.fill(kpis, null); ui.fill(rows, ui.note('这里算不出截面。换一个位置试试。')); lastMap = null; return; }
      lastMap = paint(canvas, comp, { background: null, outline: '#33475b', lineWidth: 1.6 * dpr, arrows: st.arrows !== false, velocity: session.model.A.velocity,
        maxArrows: 90, arrowWidth: 1.3 * dpr, scaleBar: true, font: 11 * dpr, ink: '#536a80' });
      var sec = comp.section, it = comp.integral;
      var main = comp.quantity === 'pressure' ? { label: '面积平均压力', v: it && it.pressure_mean_pa, u: 'Pa' }
        : comp.quantity === 'normal' ? { label: '平均穿面速度', v: it && it.normal_mean_m_s, u: 'm/s' } : { label: '面积平均速度', v: it && it.speed_mean_m_s, u: 'm/s' };
      ui.fill(kpis, [kpi('截面面积', sec && sec.area_mm2, 'mm²'), kpi('等效直径', sec && sec.equivalent_diameter_mm, 'mm'),
        kpi(main.label, main.v, main.u), it && it.flow_ml_s !== null ? kpi('流量 Q', it.flow_ml_s, 'mL/s', '顺流为正') : kpi('最大径', sec && sec.max_diameter_mm, 'mm')]);
      var S = comp.stats, list = [];
      if (it && it.flow_ml_s !== null) list.push(row('最大径', sec ? fmt(sec.max_diameter_mm) + ' mm' : '—', '中心线垂直截面上管腔的最长径（倾斜时为此平面上的最长径）'));
      if (it) list.push(row('面积加权', it.samples + ' 个体内点' + (it.direct_fraction !== null ? ' · 邻点直接支撑 ' + Math.round(100 * it.direct_fraction) + '%' : ''),
        '每个体内点代表离它最近的那部分截面面积（截到管腔轮廓为止），再按面积加权平均；流量 = 平均穿面速度 × 截面面积。'));
      list.push(row('按点统计', S.count ? '均值 ' + fmt(S.mean) + ' · p99 ' + fmt(S.p99) + ' · 最大 ' + fmt(S.max) + ' ' + (comp.field === 'velocity' ? 'm/s' : 'Pa') : '厚度内没有点',
        '厚度内全部体内预测点等权统计（' + S.count + ' 点），与经典报告截面页的统计相同。' + (comp.field === 'velocity' ? '这里统计的是速度大小。' : '')));
      ui.fill(rows, list);
      var o = comp.outlineState;
      outlineNote.textContent = o === 'synthetic' ? '截面经过血管开口，轮廓缺口按直线封闭（虚线）。' : o === 'open' ? '轮廓不闭合，不算面积和积分。' : o === 'none' ? '这里没有切到管壁轮廓（截面可能越过了血管末端）。' : '';
      outlineNote.hidden = !outlineNote.textContent;
    }
    var handle = { update: update, alive: function () { return Boolean(canvas.isConnected === undefined ? body.contains && body.contains(canvas) : canvas.isConnected) && !session.disposed(); } };
    session.addPanel(handle);
    build(); update();
    return handle;
  }

  // ------------------------------------------------------------------ zoom view (figure-quality map, PNG / CSV)
  // ctx = {ui, caseName, fileBase}
  function openZoom(session, ctx) {
    var ui = ctx.ui, h = ui.h;
    var comp = session.comp();
    if (!comp) { ui.toast('这里没有可放大的截面。', { kind: 'info' }); return null; }
    var st = session.state();
    var canvas = h('canvas', { 'class': 'slice-zoom-canvas', width: 1400, height: 1200 });
    var read = h('p', { 'class': 'slice-zoom-read muted' });
    var points = true, zmap = null, zgrid = null;
    function caption() {
      var P = comp.plane, sec = comp.section;
      return '中心 [' + P.origin.map(function (v) { return v.toFixed(1); }).join(', ') + '] mm · 法向 [' + P.normal.map(function (v) { return v.toFixed(3); }).join(', ') + '] · 厚度 ' + fmt(st.thickness) + ' mm' +
        (sec ? ' · 面积 ' + fmt(sec.area_mm2) + ' mm² · 等效直径 ' + fmt(sec.equivalent_diameter_mm) + ' mm' : '') + ' · 色标 ' + (SOURCE_LABEL[comp.range.source] || '');
    }
    function draw() {
      zgrid = zgrid || core().sliceGrid(comp.data, ZOOM_GRID, ZOOM_GRID);
      zmap = paint(canvas, comp, { background: '#ffffff', grid: zgrid, points: points, arrows: st.arrows !== false, velocity: session.model.A.velocity, maxArrows: 300, arrowWidth: 2.4,
        lineWidth: 2.2, pointRadius: 4.5, pad: { left: 40, top: 30, right: 40, bottom: 120 }, font: 22, scaleBar: true, colorbar: true, caption: caption() });
    }
    canvas.addEventListener('pointermove', function (e) {
      if (!zmap || !canvas.getBoundingClientRect) return;
      var rect = canvas.getBoundingClientRect();
      var r = readAt(zmap, comp, (e.clientX - rect.left) * canvas.width / Math.max(1, rect.width), (e.clientY - rect.top) * canvas.height / Math.max(1, rect.height));
      read.textContent = r ? '面内坐标 (' + r.x_mm.toFixed(1) + ', ' + r.y_mm.toFixed(1) + ') mm · ' + fmt(r.value) + ' ' + unitsOf(comp.quantity) + (r.fromWall ? ' · 壁面边界补全值' : '') : '';
    });
    var base = (ctx.fileBase || 'slice') + '_slice_' + comp.quantity;
    var pts = h('input', { type: 'checkbox', checked: true });
    pts.addEventListener('change', function () { points = pts.checked; draw(); });
    var png = ui.button('导出 PNG', function () {
      if (!canvas.toBlob) return;
      canvas.toBlob(function (blob) { if (blob) ui.downloadBlob(blob, base + '.png'); }, 'image/png');
    }, { icon: 'download' });
    var csv = ui.button('导出 CSV', function () {
      var cc = common(), rows = core().sliceGridToRows(zgrid, zmap.bounds, unitsOf(comp.quantity));
      if (!cc || typeof cc.tableToCSV !== 'function' || !rows.length) { ui.toast('这个截面没有可导出的格点。', { kind: 'info' }); return; }
      var P = comp.plane, filled = Boolean(comp.data.fillOn && comp.data.loop && !comp.data.loop.open);
      var text = cc.tableToCSV(['x_mm', 'y_mm', 'value', 'units', 'filled_from_wall'], rows, [
        '病例 ' + (ctx.caseName || ''), '物理量 ' + quantityLabel(comp.quantity) + '（原始单位 ' + unitsOf(comp.quantity) + '）' + (comp.quantity === 'normal' ? '；速度 · 截面法向，顺流为正' : ''),
        '中心 xyz [' + P.origin.map(function (v) { return v.toFixed(3); }).join(', ') + '] mm', '法向 [' + P.normal.map(function (v) { return v.toFixed(4); }).join(', ') + ']',
        '厚度 ' + (+st.thickness).toFixed(1) + ' mm', '网格 ' + zgrid.nx + '×' + zgrid.ny,
        '补全模式 ' + (filled ? '壁面边界补全（filled_from_wall=1 为主要由壁面边界补出的格子）' : '严格模式（只有预测点支撑的格子）'),
        'x_mm / y_mm 为截面平面内以截面中心为原点的格心坐标（u 轴、v 轴）。']);
      if (typeof root.Blob === 'function') ui.downloadBlob(new root.Blob([text], { type: 'text/csv;charset=utf-8' }), base + '.csv');
    }, { icon: 'download' });
    var dlg = ui.dialog.open({ title: '截面 · ' + quantityLabel(comp.quantity) + ' · ' + unitsOf(comp.quantity), wide: true, cls: 'dlg-slice',
      body: [canvas, read], actions: [h('label', { 'class': 'slice-check' }, pts, h('span', { text: '显示预测点' })), h('span', { 'class': 'sec-fill' }), csv, png] });
    draw();
    return dlg;
  }

  return {
    supported: supported, requiredArrays: requiredArrays, keys: keys, model: model, defaultState: defaultState, planeOf: planeOf,
    compute: compute, scaleFor: scaleFor, sectionOf: sectionOf, integrate: integrate, paint: paint, readAt: readAt, gizmoSide: gizmoSide,
    create: create, panel: panel, openZoom: openZoom, quantityLabel: quantityLabel, unitsOf: unitsOf, LIMIT: LIMIT, GRID: GRID
  };
});
