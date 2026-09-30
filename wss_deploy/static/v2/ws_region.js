/* WSS workspace v2 — custom wall regions (second phase S3, scope §11).
 * Wall results, as the classic wall report's 区域统计: a stretch of one branch between two arc positions from the
 * root (mm of the prediction points' s_from_root, the classic sliders) or a sphere around a clicked wall point.  The
 * prediction points inside give the statistics of the colouring field — equal-weight mean, p99, max
 * (VolumeViewerCore.statistics = the classic CORE.stats); the wall shows the region and dims the rest (a vertex
 * belongs to a stretch when its branch matches and its nearest prediction point's arc lies inside, to a sphere when
 * it lies inside).  The area is the display mesh's own area inside the region (the classic page showed the points'
 * share of the input surface instead).  Display only. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.region = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var MARK_HEX = 0x00c2d8;
  function VC() {
    var c = root.VolumeViewerCore;
    if (!c || typeof c.statistics !== 'function') throw new Error('volume_viewer.js（VolumeViewerCore）没有加载，区域统计不可用');
    return c;
  }
  function U() { return ns.util; }
  function fmt(v) { return v === null || v === undefined || !Number.isFinite(+v) ? '—' : (U() ? U().fmtSig(+v) : String(+(+v).toPrecision(3))); }
  function declared(result, key) { return typeof key === 'string' && (!result.declared || result.declared(key)); }

  function keys(result) {
    var g = (result.manifest || {}).geometry || {}, dm = g.display_mesh || {}, p = g.points || {};
    return { V: dm.vertices, F: dm.faces, MS: dm.segment, P: p.xyz, PS: p.segment, PSR: p.s_from_root_mm };
  }
  function supported(result) {
    if (!result || (result.family || ((result.manifest || {}).result || {}).family) !== 'wall') return false;
    var k = keys(result);
    return [k.V, k.F, k.P, k.PS, k.PSR].every(function (x) { return declared(result, x); });
  }
  function requiredArrays(result) {
    var k = keys(result);
    return Object.keys(k).map(function (n) { return k[n]; }).filter(function (x) { return declared(result, x); });
  }
  function model(result) {
    if (result.__region) return result.__region;
    var k = keys(result), A = {};
    Object.keys(k).forEach(function (n) { A[n] = declared(result, k[n]) && result.has(k[n]) ? result.array(k[n]) : null; });
    if (A.F && !(A.F instanceof Uint32Array)) A.F = Uint32Array.from(A.F);
    var names = {};
    (((result.manifest || {}).geometry || {}).branches || []).forEach(function (b) { if (b && b.name) names[b.id] = b.name; });
    // the classic unroll_branches: per branch, the arc range of its prediction points
    var br = {};
    for (var i = 0; i < A.PS.length; i++) {
      var sid = Number(A.PS[i]), s = A.PSR[i];
      if (!Number.isFinite(s)) continue;
      var b = br[sid] || (br[sid] = { id: sid, name: names[sid] || ('分支 ' + sid), s_min: Infinity, s_max: -Infinity });
      if (s < b.s_min) b.s_min = s; if (s > b.s_max) b.s_max = s;
    }
    var M = { A: A, names: names, branches: Object.keys(br).map(function (x) { return br[x]; }).sort(function (a, b) { return a.id - b.id; }), vs: null, grid: null };
    result.__region = M;
    return M;
  }
  // Arc of each display vertex = that of its nearest prediction point (exact, every point: the classic near()).
  function vertexArc(M) {
    if (M.vs) return M.vs;
    var A = M.A;
    M.grid = M.grid || U().gridIndex(A.P);
    var near = M.grid.nearestMany(A.V), vs = new Float32Array(near.length);
    for (var v = 0; v < near.length; v++) vs[v] = near[v] >= 0 ? A.PSR[near[v]] : NaN;
    M.vs = vs; M.near = near;
    return vs;
  }
  function meshArea(A, mask) {
    var s = 0, V = A.V, F = A.F;
    for (var t = 0; t < F.length; t += 3) {
      var a = F[t], b = F[t + 1], c = F[t + 2];
      if (!(mask[a] && mask[b] && mask[c])) continue;
      var ux = V[3 * b] - V[3 * a], uy = V[3 * b + 1] - V[3 * a + 1], uz = V[3 * b + 2] - V[3 * a + 2], wx = V[3 * c] - V[3 * a], wy = V[3 * c + 1] - V[3 * a + 1], wz = V[3 * c + 2] - V[3 * a + 2];
      s += 0.5 * Math.hypot(uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx);
    }
    return s;
  }
  // reg = {mode: 'range', segment, s0, s1} | {mode: 'sphere', center, radius}; values = the field's read array.
  function compute(result, reg, values) {
    var M = model(result), A = M.A, idx = [], nV = A.V.length / 3, mask = new Uint8Array(nV), label, v, i;
    if (reg.mode === 'sphere') {
      if (!Array.isArray(reg.center)) return null;
      var c = reg.center, r = Number(reg.radius) || 8, r2 = r * r;
      for (i = 0; i < A.PS.length; i++) { var dx = A.P[3 * i] - c[0], dy = A.P[3 * i + 1] - c[1], dz = A.P[3 * i + 2] - c[2]; if (dx * dx + dy * dy + dz * dz <= r2) idx.push(i); }
      for (v = 0; v < nV; v++) { var ex = A.V[3 * v] - c[0], ey = A.V[3 * v + 1] - c[1], ez = A.V[3 * v + 2] - c[2]; if (ex * ex + ey * ey + ez * ez <= r2) mask[v] = 1; }
      label = '球形区域 r = ' + fmt(r) + ' mm';
    } else {
      var b = M.branches.filter(function (x) { return x.id === Number(reg.segment); })[0];
      if (!b) return null;
      var lo = Math.min(reg.s0, reg.s1), hi = Math.max(reg.s0, reg.s1), sid = b.id;
      for (i = 0; i < A.PS.length; i++) if (Number(A.PS[i]) === sid && A.PSR[i] >= lo && A.PSR[i] <= hi) idx.push(i);   // the classic rangeMask
      var vs = vertexArc(M);
      for (v = 0; v < nV; v++) if (A.MS && Number(A.MS[v]) === sid && vs[v] >= lo && vs[v] <= hi) mask[v] = 1;
      label = b.name + ' 距入口 ' + fmt(lo) + '–' + fmt(hi) + ' mm';
    }
    var st = values ? VC().statistics(values, idx) : { count: 0, mean: null, p99: null, max: null };
    return { label: label, indices: idx, vertexMask: mask, stats: st, area_mm2: meshArea(A, mask), points: idx.length };
  }

  // ------------------------------------------------------------------ session
  // opts = {field() → field id, values(fieldId) → read array, onChange(what), onNote(text)}
  function create(viewer, result, opts) {
    opts = opts || {};
    var M = model(result), b0 = M.branches[0] || null;
    var reg = { mode: 'range', segment: b0 ? b0.id : null, s0: b0 ? b0.s_min : 0, s1: b0 ? b0.s_max : 0, center: null, radius: 8 };
    var res = null, picking = false, pickedAt = 0, disposed = false, marker = null, raf = 0;
    function emit(w) { if (typeof opts.onChange === 'function') { try { opts.onChange(w); } catch (_) {} } }
    function recompute() {
      raf = 0;
      if (disposed) return;
      var fid = opts.field ? opts.field() : null;
      try { res = compute(result, reg, fid && opts.values ? opts.values(fid) : null); } catch (e) { res = null; if (opts.onNote) opts.onNote('区域统计失败：' + (e && e.message || e)); }
      if (viewer && typeof viewer.highlight === 'function') { try { viewer.highlight(res ? [] : null, res ? { vertexMask: res.vertexMask } : null); } catch (_) {} }
      drawMarker();
      emit('region');
    }
    function schedule() { if (disposed || raf) return; var rq = root.requestAnimationFrame; if (typeof rq === 'function') raf = rq(recompute); else { raf = -1; recompute(); } }
    function set(partial) {
      Object.assign(reg, partial || {});
      if (partial && partial.segment !== undefined) { var b = M.branches.filter(function (x) { return x.id === Number(reg.segment); })[0]; if (b && partial.s0 === undefined) { reg.s0 = b.s_min; reg.s1 = b.s_max; } }
      schedule();
    }
    function drawMarker() {
      var THREE = root.THREE;
      if (marker) { if (marker.parent) marker.parent.remove(marker); marker.geometry.dispose(); marker.material.dispose(); marker = null; }
      if (!viewer || !THREE || typeof viewer.toolOverlay !== 'function') return;
      if (reg.mode === 'sphere' && Array.isArray(reg.center)) {
        marker = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 20), new THREE.MeshBasicMaterial({ color: MARK_HEX, transparent: true, opacity: 0.22, depthWrite: false }));
        marker.position.set(reg.center[0], reg.center[1], reg.center[2]); marker.scale.setScalar(Number(reg.radius) || 8); marker.renderOrder = 9;
        viewer.toolOverlay().add(marker);
      }
      viewer.render();
    }
    var cv = viewer && viewer.canvasElement, press = null;
    function onDown(e) { if (e.button === 0) press = { x: e.clientX, y: e.clientY, id: e.pointerId }; }
    function onUp(e) {
      var click = press && press.id === e.pointerId && Math.hypot(e.clientX - press.x, e.clientY - press.y) < 6;
      press = null;
      if (!click || !picking || disposed) return;
      pickedAt = Date.now();
      var hit = viewer.surfaceAt(e.clientX, e.clientY);
      if (hit && hit.xyz) { picking = false; set({ mode: 'sphere', center: hit.xyz.map(function (x) { return +x.toFixed(3); }) }); emit('picking'); }
      else if (opts.onNote) opts.onNote('没有点到管壁。');
    }
    if (cv) { cv.addEventListener('pointerdown', onDown, true); cv.addEventListener('pointerup', onUp, true); }
    function dispose() {
      if (disposed) return;
      disposed = true;
      if (raf > 0 && root.cancelAnimationFrame) root.cancelAnimationFrame(raf);
      if (cv) { cv.removeEventListener('pointerdown', onDown, true); cv.removeEventListener('pointerup', onUp, true); }
      if (viewer && typeof viewer.highlight === 'function') { try { viewer.highlight(null); } catch (_) {} }
      if (marker) { if (marker.parent) marker.parent.remove(marker); marker.geometry.dispose(); marker.material.dispose(); marker = null; }
      if (viewer && viewer.render) viewer.render();
    }
    recompute();
    return {
      state: function () { return JSON.parse(JSON.stringify(reg)); }, result: function () { return res; }, branches: function () { return M.branches.slice(); },
      set: set, refresh: schedule, recomputeNow: recompute,
      picking: function () { return picking; }, setPicking: function (on) { picking = Boolean(on); emit('picking'); },
      clickTaken: function () { return picking || Date.now() - pickedAt < 500; }, dispose: dispose,
      // P3 lane 4: the stored units of the coloured field, for the display unit of the numbers
      rawUnits: function () { var fid = opts.field ? opts.field() : null, f = fid && result.field ? result.field(fid) : null; return { units: f ? f.units : null, family: result.family || 'wall' }; }
    };
  }

  // ------------------------------------------------------------------ inspector page
  // ctx = {ui, fieldLabel(), units(), onClose}.  Controls are rebuilt when the shape changes (mode, branch, centre,
  // picking); the numbers on every update, so a slider being dragged stays in place.
  function panel(body, session, ctx) {
    var ui = ctx.ui, h = ui.h, ctrlBox = h('div', { 'class': 'region-ctrls' }), outBox = h('div', { 'class': 'region-out' }), shape = '';
    function kpi(label, value, units, sub) {
      return h('div', { 'class': 'kpi kpi-static' }, h('span', { 'class': 'kpi-label', text: label }),
        h('span', { 'class': 'kpi-value' }, h('span', { 'class': 'kpi-num', text: value === null || value === undefined ? '—' : fmt(value) }), units ? h('span', { 'class': 'kpi-unit', text: units }) : null),
        sub ? h('span', { 'class': 'kpi-sub', text: sub }) : null);
    }
    function build() {
      var st = session.state();
      var modeSeg = h('div', { 'class': 'seg seg-slice' }, [{ id: 'range', label: '分支上一段' }, { id: 'sphere', label: '球形区域' }].map(function (m) {
        var on = st.mode === m.id, b = h('button', { type: 'button', 'class': 'seg-btn' + (on ? ' on' : ''), 'aria-pressed': String(on), text: m.label });
        b.addEventListener('click', function () { session.set({ mode: m.id }); if (m.id === 'sphere' && !st.center) session.setPicking(true); update(); });
        return b;
      }));
      var controls;
      if (st.mode === 'range') {
        var list = session.branches(), b = list.filter(function (x) { return x.id === st.segment; })[0] || list[0];
        var sel = ui.select(list.map(function (x) { return { value: String(x.id), label: x.name }; }), String(st.segment), function (v) { session.set({ segment: Number(v) }); update(); }, { 'aria-label': '分支' });
        var slider = function (key, label) {
          var val = h('span', { 'class': 'slice-thv', text: fmt(st[key]) + ' mm' });
          var inp = h('input', { type: 'range', min: String(b ? b.s_min : 0), max: String(b ? b.s_max : 0), step: '0.5', value: String(st[key]), 'aria-label': label });
          inp.addEventListener('input', function () { val.textContent = fmt(+inp.value) + ' mm'; var p = {}; p[key] = Number(inp.value); session.set(p); });
          return h('label', { 'class': 'slice-ctl' }, h('span', { 'class': 'slice-ctl-k', text: label }), inp, val);
        };
        controls = [h('label', { 'class': 'slice-ctl' }, h('span', { 'class': 'slice-ctl-k', text: '分支' }), sel), slider('s0', '起点'), slider('s1', '终点')];
      } else {
        var rv = h('span', { 'class': 'slice-thv', text: fmt(st.radius) + ' mm' });
        var rin = h('input', { type: 'range', min: '1', max: '40', step: '0.5', value: String(st.radius), 'aria-label': '半径' });
        rin.addEventListener('input', function () { rv.textContent = fmt(+rin.value) + ' mm'; session.set({ radius: Number(rin.value) }); });
        var pickBtn = ui.button(session.picking() ? '在管壁上点…' : (st.center ? '重新选中心' : '点管壁选中心'), function () { session.setPicking(!session.picking()); update(); }, { cls: 'btn-sm' + (session.picking() ? ' on' : '') });
        controls = [h('div', { 'class': 'slice-ctl' }, pickBtn, h('span', { 'class': 'muted', text: st.center ? '中心 (' + st.center.map(function (x) { return x.toFixed(1); }).join(', ') + ') mm' : '还没选中心' })),
          h('label', { 'class': 'slice-ctl' }, h('span', { 'class': 'slice-ctl-k', text: '半径' }), rin, rv)];
      }
      ui.fill(ctrlBox, modeSeg, controls);
    }
    function update() {
      var st = session.state(), key = [st.mode, st.segment, st.center && st.center.join(','), session.picking()].join('|');
      if (key !== shape) { shape = key; build(); }
      var R = session.result(), units = ctx.units ? ctx.units() : '';
      // P3 lane 4: stresses in the display unit of ws_display (Pa / dyn/cm²); the statistics stay the stored ones
      var raw = session.rawUnits ? session.rawUnits() : null, cv = ns.display && typeof ns.display.toDisplay === 'function' ? ns.display.toDisplay : null;
      var q1 = raw && raw.units && cv ? cv(1, raw.units, raw.family) : null, k = q1 && Number.isFinite(+q1.value) && +q1.value > 0 ? +q1.value : 1;
      if (q1 && k !== 1) units = ui.unitText ? ui.unitText(q1.units) : q1.units;
      var sc = function (x) { return x === null || x === undefined || !Number.isFinite(+x) ? x : +x * k; };
      ui.fill(outBox, R ? [h('p', { 'class': 'region-label', text: R.label }),
        h('div', { 'class': 'kpis' }, kpi((ctx.fieldLabel ? ctx.fieldLabel() : '') + ' 均值', sc(R.stats.mean), units), kpi('p99', sc(R.stats.p99), units), kpi('最大', sc(R.stats.max), units),
          kpi('区域面积', R.area_mm2 / 100, 'cm²', R.stats.count + ' 个预测点')),
        ui.note('均值、p99、最大按区域里的预测点等权统计（与经典报告相同），跟着当前字段换；面积是显示网格落在区域里的面积。')]
        : ui.note(st.mode === 'sphere' ? '在管壁上点一下选区域中心。' : '这条分支没有可用的预测点。'));
    }
    ui.fill(body, ui.section('区域统计', { cls: 'sec-region', actions: [ui.iconButton('close', '关闭区域统计', ctx.onClose)] }, ctrlBox, outBox));
    update();
    return { update: update };
  }

  return { supported: supported, requiredArrays: requiredArrays, model: model, compute: compute, vertexArc: vertexArc, create: create, panel: panel };
});
