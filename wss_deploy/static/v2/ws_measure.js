/* WSS workspace v2 — measurement tool (second phase S3, scope §11).
 * Four modes, as the classic wall report: 距离 (two wall points, straight line), 弧长 (two points projected onto the
 * centreline tree, summed along it), 管径 (one point: the lumen outline of the section perpendicular to the centreline
 * there — largest / smallest / equivalent diameter and area; 2 × the inscribed radius when the plane closes no
 * outline) and 分段 (two points on one branch).  Clicks take the raw hit on the wall mesh.  Every number comes from
 * WssReportCommon (straightDistance, projectToCenterline, arcDistance, centerlineTangent, crossSection,
 * localDiameter) on the classic centreline groups, so a pair of points measures the same as in the classic report.
 * Measurements are kept per result in this browser.  Display only. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.measure = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var NEED = { distance: 2, arc: 2, diameter: 1, segment: 2 };
  var PREFIX = { distance: 'D', arc: 'C', diameter: 'R', segment: 'S' };
  var MODES = [{ id: 'distance', label: '距离' }, { id: 'arc', label: '弧长' }, { id: 'diameter', label: '管径' }, { id: 'segment', label: '分段' }];
  var HINT = { distance: '在管壁上点两点：两点间直线距离。', arc: '在管壁上点两点：投到中心线上，沿中心线树求弧长。', diameter: '在管壁上点一点：按那里的中心线方向切出真实截面，给管腔直径。', segment: '在同一分支上点两点：沿这条分支的长度。' };
  var LINE_HEX = 0x3fd0e0, PENDING_HEX = 0x3ccf7a;
  var STORE = 'wssv2:measure:';

  function C() {
    var c = root.WssReportCommon;
    if (!c || typeof c.arcDistance !== 'function') throw new Error('report_common.js（WssReportCommon）没有加载，测量不可用');
    return c;
  }
  function U() { return ns.util; }
  function fmt1(v) { return Number.isFinite(+v) ? (+v).toFixed(1) : '—'; }
  function r3(x) { return Number.isFinite(+x) ? +(+x).toFixed(3) : null; }

  function mesh(result) {
    var dm = ((result.manifest || {}).geometry || {}).display_mesh || {}, V = result.has(dm.vertices) ? result.array(dm.vertices) : null, F = result.has(dm.faces) ? result.array(dm.faces) : null;
    return { V: V, F: F ? (F instanceof Uint32Array ? F : Uint32Array.from(F)) : null };
  }
  function requiredArrays(result) {
    var dm = ((result.manifest || {}).geometry || {}).display_mesh || {};
    return (ns.probe ? ns.probe.requiredArrays(result) : []).concat([dm.vertices, dm.faces]).filter(function (k, i, a) { return typeof k === 'string' && a.indexOf(k) === i && (!result.declared || result.declared(k)); });
  }

  // ------------------------------------------------------------------ the classic computations
  function arcPolyline(groups, a) {
    if (!a || !a.same_branch || !a.a || !a.b) return null;
    var g = groups[a.a.group_index]; if (!g) return null;
    var first = a.a.s <= a.b.s ? a.a : a.b, last = first === a.a ? a.b : a.a, out = [first.xyz.slice()];
    for (var j = 0; j < g.s.length; j++) if (g.s[j] > first.s && g.s[j] < last.s) out.push([g.xyz[3 * j], g.xyz[3 * j + 1], g.xyz[3 * j + 2]]);
    out.push(last.xyz.slice());
    return out;
  }
  // A real lumen outline when the perpendicular plane closes one; otherwise 2 × the inscribed radius (marked).
  function diameterAt(groups, M, p) {
    var c = C(), pr = c.projectToCenterline(groups, p);
    if (!pr) return null;
    var r = Number(pr.radius_mm) || 0, tangent = c.centerlineTangent(groups, pr), lj = c.localDiameter(groups, p);
    var base = { branch: pr.name, segment_id: pr.segment_id, radius_mm: r, xyz: pr.xyz, dist_to_junction_mm: lj ? lj.dist_to_junction_mm : null };
    if (tangent && M.F && M.F.length) {
      var cs = null;
      try { cs = c.crossSection({ vertices: M.V, faces: M.F, origin: pr.xyz, normal: tangent, maxDist: Math.max(4 * r, 1) }); } catch (_) { cs = null; }
      var m = cs && cs.found && !cs.open ? cs.metrics : null;
      if (m && m.area_mm2 > 0 && Number.isFinite(m.equivalent_diameter_mm)) {
        return Object.assign(base, { method: 'contour', max_diameter_mm: m.max_diameter_mm, min_diameter_mm: m.min_diameter_mm, equivalent_diameter_mm: m.equivalent_diameter_mm,
          area_mm2: m.area_mm2, synthetic: Boolean(cs.synthetic), polygon_world: (cs.polygon_world || []).map(function (q) { return [+q[0].toFixed(3), +q[1].toFixed(3), +q[2].toFixed(3)]; }) });
      }
    }
    var d = 2 * r;
    return Object.assign(base, { method: 'inscribed', max_diameter_mm: d, min_diameter_mm: d, equivalent_diameter_mm: d, area_mm2: Math.PI * r * r, synthetic: false, polygon_world: null });
  }
  function diameterLabel(m) {
    var where = m.branch ? ' · ' + m.branch : '';
    if (m.method !== 'contour') return '管径 ≈ ' + fmt1(m.value_mm) + ' mm（内切半径×2）' + where;
    return '管腔最大直径 ' + fmt1(m.max_diameter_mm) + ' / 最小 ' + fmt1(m.min_diameter_mm) + ' / 等效 ' + fmt1(m.equivalent_diameter_mm) + ' mm · 面积 ' + fmt1(m.area_mm2) + ' mm²' + where + (m.synthetic ? '（轮廓有缺口，已闭合）' : '');
  }
  // The classic buildMeasurement: returns {item} or {error}.
  function build(result, kind, pts, items) {
    var c = C(), groups = ns.probe.groups(result), M = mesh(result), id = c.newId(PREFIX[kind] || 'M', items || []), at = new Date().toISOString();
    var name = function (sid) { return (ns.probe.model(result).names[sid]) || ''; };
    if (!groups.length) return { error: '这份结果没有可用的中心线，不能测量。' };
    if (kind === 'diameter') {
      var d = diameterAt(groups, M, pts[0]);
      if (!d) return { error: '这里没有可用的中心线，算不了管径。' };
      var m = { id: id, kind: kind, points: pts, value_mm: r3(d.equivalent_diameter_mm), branch: d.branch, segment_id: d.segment_id, created_at: at, method: d.method,
        max_diameter_mm: r3(d.max_diameter_mm), min_diameter_mm: r3(d.min_diameter_mm), equivalent_diameter_mm: r3(d.equivalent_diameter_mm), area_mm2: r3(d.area_mm2), radius_mm: r3(d.radius_mm) };
      if (d.synthetic) m.synthetic = true;
      if (Array.isArray(d.polygon_world) && d.polygon_world.length > 2) m.polygon_world = d.polygon_world;
      m.label = diameterLabel(m) + (Number.isFinite(d.dist_to_junction_mm) ? ' · 距分叉 ' + fmt1(d.dist_to_junction_mm) + ' mm' : '');
      return { item: m };
    }
    if (kind === 'distance') {
      var v = c.straightDistance(pts[0], pts[1]), pr = c.projectToCenterline(groups, pts[0]);
      return { item: { id: id, kind: kind, points: pts, value_mm: v, branch: pr ? pr.name : '', segment_id: pr ? pr.segment_id : null, created_at: at, label: '直线距离 ' + fmt1(v) + ' mm' + (pr ? '（' + pr.name + '）' : '') } };
    }
    var a = c.arcDistance(groups, pts[0], pts[1]);
    if (!a || !Number.isFinite(a.value_mm)) return { error: '两点不在同一棵中心线树上，求不了弧长。' };
    if (kind === 'segment' && !a.same_branch) return { error: '分段长度要两点在同一分支上；请重新取点，或改用「弧长」。' };
    return { item: { id: id, kind: kind, points: pts, value_mm: a.value_mm, branch: a.a ? a.a.name : '', segment_id: a.a ? a.a.segment_id : null, path: (a.path || []).slice(), same_branch: Boolean(a.same_branch),
      path_xyz: arcPolyline(groups, a), created_at: at,
      label: (kind === 'segment' ? '分段长度 ' : '弧长距离 ') + fmt1(a.value_mm) + ' mm · ' + (a.same_branch ? '同分支 ' + (a.a ? a.a.name : '') : '跨分支经 ' + (a.path || []).map(name).join(' → ')) } };
  }
  // The classic volume report's copy-all (the wall report copied one row at a time).
  function tsv(items) {
    var head = ['id', 'kind', 'value_mm', 'branch', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2'].join('\t');
    return [head].concat((items || []).map(function (m) {
      var a = (m.points || [])[0] || [], b = (m.points || [])[1] || [];
      return [m.id, m.kind, Number(m.value_mm).toFixed(3), m.branch || '', a[0], a[1], a[2], b[0], b[1], b[2]].map(function (v) { return v === undefined || v === null ? '' : String(v); }).join('\t');
    })).join('\n');
  }
  function load(runIdentity) { try { var raw = runIdentity && root.localStorage && root.localStorage.getItem(STORE + runIdentity); var v = raw ? JSON.parse(raw) : []; return Array.isArray(v) ? v : []; } catch (_) { return []; } }
  function save(runIdentity, items) { try { if (runIdentity && root.localStorage) { if (items.length) root.localStorage.setItem(STORE + runIdentity, JSON.stringify(items)); else root.localStorage.removeItem(STORE + runIdentity); } } catch (_) {} }

  // ------------------------------------------------------------------ session (one result, one viewer)
  // opts = {runIdentity, onChange(what), onNote(text)}
  function create(viewer, result, opts) {
    opts = opts || {};
    var items = load(opts.runIdentity), mode = null, pending = [], status = '', disposed = false, pickedAt = 0, group = null;
    function emit(what) { if (typeof opts.onChange === 'function') { try { opts.onChange(what); } catch (_) {} } }
    function persist() { save(opts.runIdentity, items); }
    function setMode(m) {
      mode = m && NEED[m] ? (mode === m ? null : m) : null;
      pending = [];
      status = mode ? HINT[mode] : '';
      draw(); emit('mode');
    }
    function addPoint(p) {
      if (!mode) return;
      pending.push(p.map(function (v) { return +v.toFixed(3); }));
      var need = NEED[mode];
      if (pending.length < need) { status = '已取 ' + pending.length + ' / ' + need + ' 点，继续在管壁上点。'; draw(); emit('pending'); return; }
      var pts = pending.slice(0, need); pending = [];
      var r = null;
      try { r = build(result, mode, pts, items); } catch (e) { r = { error: '测量失败：' + (e && e.message || e) }; }
      if (r.error) { status = r.error; draw(); emit('status'); return; }
      items.push(r.item); persist();
      status = r.item.label;
      draw(); emit('items');
    }
    function remove(id) { items = items.filter(function (m) { return m.id !== id; }); persist(); draw(); emit('items'); }
    function clearAll() { items = []; pending = []; persist(); status = '已清空测量。'; draw(); emit('items'); }
    function cancelPending() { if (pending.length) { pending = []; status = mode ? HINT[mode] : ''; draw(); emit('pending'); return true; } if (mode) { setMode(null); return true; } return false; }
    // ---- 3-D: lines (the arc path along the centreline), points, the diameter outline, a label per measurement
    function draw() {
      var THREE = root.THREE;
      if (!viewer || !THREE || typeof viewer.toolOverlay !== 'function') return;
      if (group) { if (group.parent) group.parent.remove(group); group.traverse(function (o) { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); }); group = null; }
      group = new THREE.Group(); group.name = 'measure';
      var lineMat = new THREE.LineBasicMaterial({ color: LINE_HEX, depthTest: false, transparent: true, opacity: 0.95 });
      var dot = function (p, hex) {
        var g = new THREE.Group(); g.position.set(p[0], p[1], p[2]); g.userData.screenPx = 3.5;
        var halo = new THREE.Mesh(new THREE.SphereGeometry(1.5, 12, 8), new THREE.MeshBasicMaterial({ color: 0x0b1118, depthTest: false })); halo.renderOrder = 14;
        var core = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), new THREE.MeshBasicMaterial({ color: hex, depthTest: false })); core.renderOrder = 15;
        g.add(halo); g.add(core); group.add(g);
      };
      var poly = function (pts, closed) {
        if (!pts || pts.length < 2) return;
        var list = closed ? pts.concat([pts[0]]) : pts, pos = new Float32Array((list.length - 1) * 6);
        for (var i = 0; i + 1 < list.length; i++) { pos.set(list[i], 6 * i); pos.set(list[i + 1], 6 * i + 3); }
        var gg = new THREE.BufferGeometry(); gg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        var o = new THREE.LineSegments(gg, lineMat); o.renderOrder = 13; group.add(o);
      };
      var labels = [];
      items.forEach(function (m) {
        var pts = m.points || [];
        if (m.kind === 'diameter') { if (m.polygon_world) poly(m.polygon_world, true); }
        else poly(m.path_xyz && m.path_xyz.length > 1 ? m.path_xyz : pts, false);
        pts.forEach(function (p) { dot(p, LINE_HEX); });
        var path = m.kind === 'diameter' ? pts : (m.path_xyz && m.path_xyz.length > 1 ? m.path_xyz : pts), mid = path[Math.floor((path.length - 1) / 2)];
        if (path.length === 2) mid = [(path[0][0] + path[1][0]) / 2, (path[0][1] + path[1][1]) / 2, (path[0][2] + path[1][2]) / 2];
        labels.push({ id: m.id, xyz: mid, text: m.id + ' · ' + fmt1(m.value_mm) + ' mm' });
      });
      pending.forEach(function (p) { dot(p, PENDING_HEX); });
      viewer.toolOverlay().add(group);
      if (typeof viewer.setToolLabels === 'function') viewer.setToolLabels('measure', labels);
      viewer.render();
    }
    // clicks in a measuring mode place points (the shell skips them as readings)
    var cv = viewer && viewer.canvasElement, press = null;
    function onDown(e) { if (e.button === 0) press = { x: e.clientX, y: e.clientY, id: e.pointerId }; }
    function onUp(e) {
      var click = press && press.id === e.pointerId && Math.hypot(e.clientX - press.x, e.clientY - press.y) < 6;
      press = null;
      if (!click || !mode || disposed) return;
      pickedAt = Date.now();
      var hit = viewer.surfaceAt(e.clientX, e.clientY);
      if (hit && hit.xyz) addPoint(hit.xyz);
      else if (opts.onNote) opts.onNote('没有点到管壁。');
    }
    if (cv) { cv.addEventListener('pointerdown', onDown, true); cv.addEventListener('pointerup', onUp, true); }
    function dispose() {
      if (disposed) return;
      disposed = true;
      if (cv) { cv.removeEventListener('pointerdown', onDown, true); cv.removeEventListener('pointerup', onUp, true); }
      if (group) { if (group.parent) group.parent.remove(group); group.traverse(function (o) { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); }); group = null; }
      if (viewer && typeof viewer.setToolLabels === 'function') viewer.setToolLabels('measure', []);
      if (viewer && viewer.render) viewer.render();
    }
    draw();
    return {
      mode: function () { return mode; }, setMode: setMode, addPoint: addPoint, pending: function () { return pending.slice(); },
      items: function () { return items.slice(); }, remove: remove, clear: clearAll, cancelPending: cancelPending, status: function () { return status; },
      clickTaken: function () { return Boolean(mode) || Date.now() - pickedAt < 500; }, tsv: function () { return tsv(items); },
      redraw: draw, dispose: dispose
    };
  }

  // ------------------------------------------------------------------ inspector page
  // ctx = {ui, onClose, onFly(xyz), onCopy(text)}
  function panel(body, session, ctx) {
    var ui = ctx.ui, h = ui.h, cur = session.mode();
    var seg = h('div', { 'class': 'seg seg-slice', role: 'tablist', 'aria-label': '测量方式' }, MODES.map(function (m) {
      var on = cur === m.id, b = h('button', { type: 'button', role: 'tab', 'class': 'seg-btn' + (on ? ' on' : ''), 'aria-selected': String(on), text: m.label, title: HINT[m.id] });
      b.addEventListener('click', function () { session.setMode(m.id); });
      return b;
    }));
    var st = session.status();
    var list = session.items();
    var rows = list.map(function (m) {
      return h('div', { 'class': 'meas-row' }, h('b', { 'class': 'meas-id', text: m.id }), h('span', { 'class': 'meas-text', text: m.label }),
        h('span', { 'class': 'meas-acts' },
          ui.iconButton('eye', '转到这里', function () { var p = m.points && m.points[0]; if (p && ctx.onFly) ctx.onFly(m.points.length > 1 ? [0, 1, 2].map(function (k) { return (m.points[0][k] + m.points[1][k]) / 2; }) : p); }),
          ui.iconButton('copy', '复制这一条', function () { if (ctx.onCopy) ctx.onCopy(m.id + '\t' + m.label + '\t' + (+m.value_mm).toFixed(3) + '\tmm'); }),
          ui.iconButton('close', '删除', function () { session.remove(m.id); })));
    });
    ui.fill(body, ui.section('测量', { cls: 'sec-measure', actions: [
        list.length ? ui.button('复制全部', function () { if (ctx.onCopy) ctx.onCopy(session.tsv()); }, { kind: 'link', cls: 'btn-sm' }) : null,
        list.length ? ui.button('清空', function () { session.clear(); }, { kind: 'link', cls: 'btn-sm' }) : null,
        ui.iconButton('close', '关闭测量（M）', ctx.onClose)] },
      seg, h('p', { 'class': 'meas-status' + (cur ? ' on' : ''), text: st || '先选一种测量，再在管壁上点。' }),
      rows.length ? h('div', { 'class': 'meas-list' }, rows) : ui.note('还没有测量。结果只存在这台电脑的浏览器里。')));
  }

  return { create: create, panel: panel, build: build, diameterAt: diameterAt, arcPolyline: arcPolyline, tsv: tsv, requiredArrays: requiredArrays, NEED: NEED, MODES: MODES };
});
