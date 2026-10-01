/* WSS workspace v2 — probe card and probe log (second phase S3, scope §11).
 * A click on the vessel pins a probe.  Wall results: the nearest prediction point to the raw hit (exact, every
 * point), and the section perpendicular to the centreline at the hit — each field averaged around the lumen outline
 * (∮ f dl / ∮ dl, every wall element takes its nearest prediction point) with the ring's min → max
 * (WssReportCommon.sectionMeans, as the classic wall report).  Volume results: the point's values and the area
 * integrals of that section (WssReportCommon.stationSection + the classic sectionIntegral through ws_slice).
 * 「记录」 keeps a row with the classic row schema; copy TSV / export CSV use WssReportCommon.probeToTSV / CSV, so the
 * files are the same as the classic reports'.  The log is kept per result in this browser.  Display only.
 * Phase 3 lane 4: the hover readout (classic #tip / probeAt, W41 / V17) — a small chip next to the pointer while it rests
 * on the vessel, switched with P or the layers menu — and wall stresses in the display unit (Pa / dyn/cm²) on the card. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.probe = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var ROW_KEY = { wss: 'wss_pa', tawss: 'tawss_pa', osi: 'osi', rrt: 'rrt_per_pa', ecap: 'ecap_per_pa' };   // classic array keys
  var SECTION_HEX = 0x111111;
  var STORE = 'wssv2:probes:';

  function C() {
    var c = root.WssReportCommon;
    if (!c || typeof c.sectionMeans !== 'function') throw new Error('report_common.js（WssReportCommon）没有加载，探针截面不可用');
    return c;
  }
  function U() { return ns.util; }
  function fmt(v) { return v === null || v === undefined || !Number.isFinite(+v) ? '—' : (U() ? U().fmtSig(+v) : String(+(+v).toPrecision(3))); }
  function fx(v, d) { return Number.isFinite(+v) ? +(+v).toFixed(d) : null; }
  function declared(result, key) { return typeof key === 'string' && (!result.declared || result.declared(key)); }
  function arr(result, key) { return declared(result, key) && result.has(key) ? result.array(key) : null; }
  function family(result) { return result && (result.family || (result.manifest && result.manifest.result && result.manifest.result.family)) || 'wall'; }

  // ------------------------------------------------------------------ data
  function keys(result) {
    var m = result.manifest || {}, g = m.geometry || {}, dm = g.display_mesh || {}, p = g.points || {}, cl = g.centerline || {};
    var fk = function (id, kind) { var f = (m.fields || []).filter(function (x) { return x && x.id === id; })[0]; return f && f.arrays && typeof f.arrays[kind] === 'string' ? f.arrays[kind] : null; };
    var out = { V: dm.vertices, F: dm.faces, P: p.xyz, PS: p.segment, PSR: p.s_from_root_mm, PR: p.radius_mm, PW: p.is_wall, PDW: p.dist_to_wall_mm, PT: p.trust,
      cv: cl.xyz, cr: cl.radius_mm, ce: cl.edges, cs: cl.segment, velocity: fk('velocity', 'read'), pressure: fk('pressure', 'read'), wallP: fk('wall_pressure', 'display') };
    wallFields(result).forEach(function (f) { out['f:' + f.id] = f.arrays.read; });
    return out;
  }
  // Wall scalar fields with a value per prediction point, in manifest order.
  function wallFields(result) {
    return ((result.manifest && result.manifest.fields) || []).filter(function (f) {
      return f && (f.location === 'wall' || family(result) === 'wall') && f.kind === 'scalar' && (Number(f.components) || 1) === 1 && f.arrays && typeof f.arrays.read === 'string' && declared(result, f.arrays.read);
    });
  }
  function requiredArrays(result) {
    var k = keys(result);
    return Object.keys(k).map(function (n) { return k[n]; }).filter(function (x) { return declared(result, x); });
  }
  function names(result) {
    var out = {};
    (((result.manifest || {}).geometry || {}).branches || []).forEach(function (b) { if (b && b.name) out[b.id] = b.name; });
    return out;
  }
  // The classic reports' centreline groups (WssReportCommon, no parent override, no length scaling) — not the cursor's.
  function groups(result) {
    if (result.__probeGroups) return result.__probeGroups;
    var k = keys(result), cv = arr(result, k.cv), g = [];
    if (cv && cv.length) { try { g = C().buildCenterlineGroups({ xyz: cv, radius: arr(result, k.cr), edges: arr(result, k.ce), segment: arr(result, k.cs) }, names(result)) || []; } catch (_) { g = []; } }
    result.__probeGroups = g;
    return g;
  }
  function model(result) {
    if (result.__probe) return result.__probe;
    var k = keys(result), P = arr(result, k.P), V = arr(result, k.V), F0 = arr(result, k.F);
    var X = { result: result, k: k, P: P, V: V, F: F0 ? (F0 instanceof Uint32Array ? F0 : Uint32Array.from(F0)) : null, names: names(result), groups: groups(result) };
    X.grid = P ? U().gridIndex(P) : null;
    var mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    if (V) for (var i = 0; i < V.length; i++) { var j = i % 3; if (V[i] < mn[j]) mn[j] = V[i]; if (V[i] > mx[j]) mx[j] = V[i]; }
    X.diag = V ? Math.max(Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]), 1) : 100;
    result.__probe = X;
    return X;
  }
  function branchName(X, sid) { return sid === null || sid === undefined || !Number.isFinite(+sid) ? '—' : (X.names[sid] || ('分支 ' + sid)); }

  // ------------------------------------------------------------------ probes
  // e: the viewer's pick ({pointIndex, vertexIndex, hitXyz, xyz, …}).  Returns null when nothing can be read there.
  function pick(result, e, opts) {
    if (!e) return null;
    return family(result) === 'volume' ? volumeProbe(result, e, opts || {}) : wallProbe(result, e);
  }
  function wallProbe(result, e) {
    var X = model(result), hit = (e.hitXyz || e.xyz || []).slice(0, 3);
    if (!X.grid || hit.length < 3) return null;
    var i = X.grid.nearest(hit[0], hit[1], hit[2]);   // exact, over every point (the classic lockProbe)
    if (!(i >= 0)) return null;
    var fields = {};
    wallFields(result).forEach(function (f) { fields[f.id] = result.array(f.arrays.read); });
    var sec = null;
    if (X.groups.length && X.V && X.F && X.F.length) {
      try { sec = C().sectionMeans({ groups: X.groups, point: hit, vertices: X.V, faces: X.F, fields: fields, sample: function (q) { return X.grid.nearestMany(q); } }); } catch (_) { sec = null; }
    }
    return { kind: 'wall', family: 'wall', index: i, hit: hit, section: sec, fields: fields };
  }
  // Volume: an interior prediction point (its record) or a wall vertex (wall pressure), and the section integrals at it.
  function volumeProbe(result, e, opts) {
    var X = model(result), k = X.k, pos, kind, index;
    if (e.pointIndex !== null && e.pointIndex !== undefined && e.pointIndex >= 0 && X.P) { kind = 'interior'; index = e.pointIndex; pos = [X.P[3 * index], X.P[3 * index + 1], X.P[3 * index + 2]]; }
    else if (e.vertexIndex !== null && e.vertexIndex !== undefined && e.vertexIndex >= 0 && X.V) { kind = 'wall'; index = e.vertexIndex; pos = [X.V[3 * index], X.V[3 * index + 1], X.V[3 * index + 2]]; }
    else return null;
    var thickness = Number(opts.thickness) > 0 ? Number(opts.thickness) : 2;
    return { kind: kind, family: 'volume', index: index, hit: pos, section: sectionIntegralAt(result, pos, thickness) };
  }
  // The classic volume report's sectionIntegralAt: the section perpendicular to the centreline at p, integrated over the
  // lumen with the slab thickness of the section tool (2 mm by default).
  function sectionIntegralAt(result, p, thickness) {
    var X = model(result), c = C();
    if (!X.groups.length) return { found: false, reason: 'no_centerline' };
    var st = c.stationSection({ groups: X.groups, point: p, vertices: X.V, faces: X.F, wallValues: arr(result, X.k.wallP) });
    st.thickness_mm = thickness;
    if (!st.found) return st;
    var cs = st.section, m = cs.metrics;
    if (cs.open || !m || !(m.area_mm2 > 0) || !ns.slice || typeof ns.slice.integrate !== 'function') return Object.assign(st, { integrated: false });
    var M = result.__sliceModel || (result.__sliceModel = ns.slice.model(result));
    var it = ns.slice.integrate(M, cs.plane, cs.segs, thickness, m.area_mm2);
    return Object.assign(st, { integrated: it.integrated, area_mm2: m.area_mm2, equivalent_diameter_mm: m.equivalent_diameter_mm, samples: it.samples,
      speed_mean_m_s: it.speed_mean_m_s, normal_mean_m_s: it.normal_mean_m_s, flow_ml_s: it.flow_ml_s, pressure_mean_pa: it.pressure_mean_pa, direct_fraction: it.direct_fraction });
  }
  // The classic probeRecord for an interior point.
  function record(result, i) {
    var X = model(result), k = X.k, P = X.P, vel = arr(result, k.velocity), pr = arr(result, k.pressure), seg = arr(result, k.PS);
    var rec = { index: i, position: [P[3 * i], P[3 * i + 1], P[3 * i + 2]] };
    if (vel) { rec.velocity = [vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]]; rec.speed = Math.hypot(rec.velocity[0], rec.velocity[1], rec.velocity[2]); }
    if (pr && Number.isFinite(pr[i])) rec.pressure = pr[i];
    if (seg) rec.segment = Number(seg[i]);
    [['s', k.PSR], ['radius', k.PR], ['distWall', k.PDW], ['trust', k.PT]].forEach(function (q) { var a = arr(result, q[1]); if (a && Number.isFinite(a[i])) rec[q[0]] = a[i]; });
    return rec;
  }

  // ------------------------------------------------------------------ log rows (classic schemas)
  function nextId(rows) { return 'P' + ((rows || []).reduce(function (m, r) { return Math.max(m, Number(String(r.id || '').replace(/^P/, '')) || 0); }, 0) + 1); }
  function rowKey(f) { return f.array_key || ROW_KEY[f.id] || f.id; }
  function wallRow(result, pb, rows) {
    var X = model(result), k = X.k, i = pb.index, PS = arr(result, k.PS), PSR = arr(result, k.PSR), PR = arr(result, k.PR), values = {};
    var list = wallFields(result).slice().sort(function (a, b) { return (b.id === 'wss') - (a.id === 'wss'); });
    list.forEach(function (f) { var v = pb.fields[f.id] ? pb.fields[f.id][i] : NaN; if (Number.isFinite(v)) values[rowKey(f)] = +v.toFixed(4); });
    var sec = pb.section;
    if (sec && sec.found) {
      list.forEach(function (f) { var mm = sec.means[f.id]; if (mm && Number.isFinite(mm.mean)) values['section_' + rowKey(f)] = +mm.mean.toFixed(4); });
      values.section_s_from_root_mm = +sec.s_from_root_mm.toFixed(2); values.section_perimeter_mm = +sec.wall_length_mm.toFixed(2);
    }
    return { id: nextId(rows), xyz_mm: [0, 1, 2].map(function (q) { return +X.P[3 * i + q].toFixed(3); }), branch: PS ? branchName(X, PS[i]) : '', segment_id: PS ? Number(PS[i]) : null,
      s_from_root_mm: PSR ? fx(PSR[i], 2) : null, radius_mm: PR ? fx(PR[i], 3) : null, values: values, created_at: new Date().toISOString() };
  }
  function sectionValues(values, st) {
    if (!st || !st.found || !st.integrated) return values;
    var put = function (key, v, d) { if (v !== null && v !== undefined && Number.isFinite(v)) values[key] = +v.toFixed(d); };
    put('section_s_from_root_mm', st.s_from_root_mm, 2); put('section_area_mm2', st.area_mm2, 2); put('section_speed_mean_m_s', st.speed_mean_m_s, 4);
    put('section_normal_mean_m_s', st.normal_mean_m_s, 4); put('section_flow_ml_s', st.flow_ml_s, 3); put('section_pressure_mean_pa', st.pressure_mean_pa, 2);
    return values;
  }
  function volumeRow(result, pb, rows) {
    var X = model(result), values = {};
    if (pb.kind === 'interior') {
      var rec = record(result, pb.index);
      if (rec.speed !== undefined) { values.speed_m_s = rec.speed; values.u = rec.velocity[0]; values.v = rec.velocity[1]; values.w = rec.velocity[2]; }
      if (rec.pressure !== undefined) values.pressure_pa = rec.pressure;
      if (rec.distWall !== undefined) values.dist_to_wall_mm = rec.distWall;
      sectionValues(values, pb.section);
      return { id: nextId(rows), kind: 'interior', index: pb.index, xyz_mm: rec.position.map(function (v) { return +v.toFixed(3); }), branch: rec.segment !== undefined ? branchName(X, rec.segment) : '',
        segment_id: rec.segment !== undefined ? rec.segment : null, s_from_root_mm: rec.s !== undefined ? +rec.s.toFixed(2) : null, radius_mm: rec.radius !== undefined ? +rec.radius.toFixed(3) : null,
        values: values, created_at: new Date().toISOString() };
    }
    var wp = arr(result, X.k.wallP), v = pb.index, st = pb.section;
    if (wp && Number.isFinite(wp[v])) values.wall_pressure_pa = wp[v];
    sectionValues(values, st);
    // the classic page wrote the branch-local arc of the nearest centreline sample here; this is the section's
    // arc from the root (the key says so)
    return { id: nextId(rows), kind: 'wall', index: v, xyz_mm: pb.hit.map(function (x) { return +x.toFixed(3); }), branch: st && st.found ? branchName(X, st.segment_id) : '',
      segment_id: st && st.found ? st.segment_id : null, s_from_root_mm: st && st.found ? fx(st.s_from_root_mm, 2) : null, radius_mm: null, values: values, created_at: new Date().toISOString() };
  }
  function row(result, pb, rows) { return pb.family === 'volume' ? volumeRow(result, pb, rows) : wallRow(result, pb, rows); }

  // ------------------------------------------------------------------ the log (per result, this browser)
  function loadLog(runIdentity) {
    if (!runIdentity) return [];
    try { var raw = root.localStorage && root.localStorage.getItem(STORE + runIdentity); var v = raw ? JSON.parse(raw) : []; return Array.isArray(v) ? v : []; } catch (_) { return []; }
  }
  function saveLog(runIdentity, rows) {
    if (!runIdentity) return;
    try { if (root.localStorage) { if (rows.length) root.localStorage.setItem(STORE + runIdentity, JSON.stringify(rows)); else root.localStorage.removeItem(STORE + runIdentity); } } catch (_) {}
  }
  function tsv(rows) { return C().probeToTSV(rows, 'zh'); }
  function csv(rows) { return C().probeToCSV(rows, 'zh'); }

  // ------------------------------------------------------------------ 3-D: the section ring at the pinned probe
  function draw(viewer, result, pb) {
    clear(viewer);
    var THREE = root.THREE;
    if (!viewer || !THREE || typeof viewer.toolOverlay !== 'function' || !pb) return;
    var X = model(result), sec = pb.section, ring = sec && sec.found ? (sec.polygon_world || (sec.section && sec.section.polygon_world) || []) : [];
    ring = ring.filter(function (q) { return Array.isArray(q) && q.length === 3; });
    var g = new THREE.Group(); g.name = 'probe-section';
    if (ring.length > 2) {
      var local = Number(sec.radius_mm) || Infinity, r = Math.max(0.15, Math.min(X.diag * 0.002, 0.15 * local));
      var open = Boolean(sec.open || (sec.section && sec.section.open));
      var curve = new THREE.CatmullRomCurve3(ring.map(function (q) { return new THREE.Vector3(q[0], q[1], q[2]); }), !open);
      [[r, SECTION_HEX, 12], [r * 0.45, 0xffffff, 13]].forEach(function (t) {
        var mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.min(600, 2 * ring.length), t[0], 6, !open), new THREE.MeshBasicMaterial({ color: t[1], depthTest: false }));
        mesh.renderOrder = t[2]; g.add(mesh);
      });
      var o = sec.origin || (sec.projection && sec.projection.xyz);
      if (Array.isArray(o)) {
        var R = Math.max(0.5, Math.min(X.diag * 0.006, 0.4 * local));
        [[SECTION_HEX, R, 12], [0xffffff, R * 0.5, 13]].forEach(function (t) { var s = new THREE.Mesh(new THREE.SphereGeometry(t[1], 14, 10), new THREE.MeshBasicMaterial({ color: t[0], depthTest: false })); s.position.set(o[0], o[1], o[2]); s.renderOrder = t[2]; g.add(s); });
        var lg = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(pb.hit[0], pb.hit[1], pb.hit[2]), new THREE.Vector3(o[0], o[1], o[2])]);
        var line = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: SECTION_HEX, depthTest: false })); line.renderOrder = 12; g.add(line);
      }
    }
    viewer.toolOverlay().add(g);
    viewer.__probeGroup = g;
    viewer.render();
  }
  function clear(viewer) {
    var g = viewer && viewer.__probeGroup;
    if (!g) return;
    if (g.parent) g.parent.remove(g);
    g.traverse(function (o) { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    viewer.__probeGroup = null;
    if (viewer.render) viewer.render();
  }

  // ------------------------------------------------------------------ the card
  // ctx = {ui, manifest, field (active id), onField(id), onRecord(), onClose(), onSlice(plane) (volume only)}
  function sectionNote(X, sec) {
    if (!sec) return '这里没有可用的中心线或壁面网格，取不到截面。';
    if (!sec.found) return sec.reason === 'no_contour' ? '这里的垂直截面没有切到管腔轮廓。' : '这里没有可用的中心线。';
    var s = '截面垂直于中心线，过' + branchName(X, sec.segment_id) + '距入口 ' + fmt(sec.s_from_root_mm) + ' mm 处';
    if (sec.wall_length_mm !== undefined) s += ' · 周长 ' + fmt(sec.wall_length_mm) + ' mm';
    if (sec.open) s += ' · 轮廓不闭合，只平均切到的壁面';
    else if (sec.synthetic) s += ' · 经过开口，缺口 ' + fmt(sec.gap_mm) + ' mm 不计';
    return s + '。';
  }
  function rangeBar(h, x, m) {
    if (!m || !Number.isFinite(m.min) || !Number.isFinite(m.max) || !(m.max > m.min)) return null;
    var pos = function (v) { return Math.max(0, Math.min(100, (v - m.min) / (m.max - m.min) * 100)).toFixed(1) + '%'; };
    var out = Number.isFinite(x) && (x < m.min || x > m.max);
    var bar = h('span', { 'class': 'pc-range', title: '环上 ' + fmt(m.min) + ' – ' + fmt(m.max) + '；竖线 = 截面均值，圆点 = 此点' + (out ? '（此点在环上范围之外）' : '') },
      h('i', { 'class': 'pc-mean', style: 'left:' + pos(m.mean) }), Number.isFinite(x) ? h('i', { 'class': 'pc-pt' + (out ? ' out' : ''), style: 'left:' + pos(Math.max(m.min, Math.min(m.max, x))) }) : null);
    return h('span', { 'class': 'pc-range-wrap' }, h('span', { 'class': 'pc-lo', text: fmt(m.min) }), bar, h('span', { 'class': 'pc-hi', text: fmt(m.max) }));
  }
  // Display units of the volume readings (lane E, ws_display: Pa / mmHg, m/s / cm/s); the record and exports stay raw.
  function qv(x, kind) {
    var d = ns.display && typeof ns.display.displayUnit === 'function' ? ns.display.displayUnit(kind) : null;
    var u = d && Number.isFinite(+d.factor) && +d.factor > 0 ? d : { units: kind === 'pressure' ? 'Pa' : 'm/s', factor: 1 };
    var one = function (v) { return fmt(Number.isFinite(+v) ? +v * u.factor : v); };
    return (Array.isArray(x) ? x.map(one).join(', ') : one(x)) + ' ' + u.units;
  }
  function card(result, pb, ctx) {
    var ui = ctx.ui, h = ui.h, X = model(result), k = X.k;
    var acts = [ui.button('记录', ctx.onRecord, { cls: 'btn-sm', title: '把此点和截面读数加入探针记录' })];
    if (pb.family === 'volume' && pb.section && pb.section.found && ctx.onSlice) acts.push(ui.button('看截面', function () { ctx.onSlice({ origin: pb.section.origin.slice(), normal: pb.section.tangent.slice() }); }, { cls: 'btn-sm', title: '在截面工具里打开这个截面' }));
    acts.push(ui.iconButton('close', '关闭探针', ctx.onClose));
    var head, body = [], sec = pb.section;
    if (pb.family === 'wall') {
      var PS = arr(result, k.PS), PSR = arr(result, k.PSR), PR = arr(result, k.PR), i = pb.index;
      head = [PS ? branchName(X, PS[i]) : '—', PSR ? '距入口 ' + fmt(PSR[i]) + ' mm' : null, PR ? '半径 ' + fmt(PR[i]) + ' mm' : null].filter(Boolean).join(' · ');
      var ok = Boolean(sec && sec.found);
      var rows = wallFields(result).map(function (f) {
        // P3 lane 4: stresses in Pa follow the display unit (× k), the record keeps Pa
        var q1 = conv(1, f.units, 'wall'), kk = Number.isFinite(+q1.value) && +q1.value > 0 ? +q1.value : 1;
        var x0 = pb.fields[f.id] ? pb.fields[f.id][i] : NaN, m0 = ok ? sec.means[f.id] : null, units = ui.unitText ? ui.unitText(q1.units) : (q1.units || '');
        var x = Number.isFinite(x0) ? x0 * kk : x0, m = m0 && kk !== 1 ? { mean: m0.mean * kk, min: m0.min * kk, max: m0.max * kk } : m0;
        var tr = h('tr', { 'class': f.id === ctx.field ? 'on' : null, title: '按 ' + (f.short_label || f.label || f.id) + ' 着色' },
          h('th', null, f.short_label || f.label || f.id, units ? h('small', { text: units }) : null), h('td', { text: fmt(x) }),
          ok ? h('td', { 'class': 'pc-sec', text: fmt(m && m.mean) }) : null, ok ? h('td', null, rangeBar(h, x, m)) : null);
        tr.addEventListener('click', function () { if (ctx.onField && f.id !== ctx.field) ctx.onField(f.id); });
        return tr;
      });
      body.push(h('table', { 'class': 'pc-table' },
        h('thead', null, h('tr', null, h('th'), h('th', { text: '此点', title: '离点击处最近的预测点' }),
          ok ? h('th', { text: '截面均值', title: '截面环上逐段取最近的预测点，按长度加权平均（∮f dl / 周长）' }) : null,
          ok ? h('th', { text: '环上范围', title: '截面环上最低 → 最高；竖线 = 截面均值，圆点 = 此点' }) : null)),
        h('tbody', null, rows)));
    } else {
      var kv = [];
      if (pb.kind === 'interior') {
        var rec = record(result, pb.index);
        head = [rec.segment !== undefined ? branchName(X, rec.segment) : '—', rec.s !== undefined ? '距入口 ' + fmt(rec.s) + ' mm' : null, rec.radius !== undefined ? '半径 ' + fmt(rec.radius) + ' mm' : null].filter(Boolean).join(' · ');
        if (rec.speed !== undefined) kv.push(['速度大小', qv(rec.speed, 'velocity')], ['速度分量', qv(rec.velocity, 'velocity')]);
        if (rec.pressure !== undefined) kv.push(['相对压力', qv(rec.pressure, 'pressure')]);
        if (rec.distWall !== undefined) kv.push(['到壁距离', fmt(rec.distWall) + ' mm']);
      } else {
        var wp = arr(result, k.wallP);
        head = sec && sec.found ? branchName(X, sec.segment_id) + ' · 壁面' : '壁面';
        kv.push(['壁面压力', wp && Number.isFinite(wp[pb.index]) ? qv(wp[pb.index], 'pressure') : '无插值支撑']);
      }
      if (sec && sec.found && sec.integrated) {
        kv.push(['截面面积', fmt(sec.area_mm2) + ' mm² · 等效直径 ' + fmt(sec.equivalent_diameter_mm) + ' mm']);
        if (sec.speed_mean_m_s !== null) kv.push(['面积平均速度', qv(sec.speed_mean_m_s, 'velocity')]);
        if (sec.normal_mean_m_s !== null) kv.push(['平均穿面速度', qv(sec.normal_mean_m_s, 'velocity') + '（顺流为正）'], ['截面流量 Q', fmt(sec.flow_ml_s) + ' mL/s']);
        if (sec.pressure_mean_pa !== null) kv.push(['面积平均压力', qv(sec.pressure_mean_pa, 'pressure')]);
      }
      body.push(h('div', { 'class': 'pc-kv' }, kv.map(function (r) { return h('div', { 'class': 'slice-row' }, h('span', { 'class': 'slice-row-k', text: r[0] }), h('b', { text: r[1] })); })));
    }
    var note = sectionNote(X, sec);
    if (pb.family === 'volume' && sec && sec.found) note += sec.integrated ? ' 厚 ' + fmt(sec.thickness_mm) + ' mm 内 ' + sec.samples + ' 个体内点，各代表离它最近的截面面积。' : ' 轮廓不闭合，不做面积积分。';
    var foot = h('div', { 'class': 'pc-foot' }, h('span', { text: '此点 (' + pb.hit.map(function (v) { return v.toFixed(1); }).join(', ') + ') mm' }), ui.infoTip ? ui.infoTip(note) : null);
    return h('section', { 'class': 'sec sec-probe' }, h('div', { 'class': 'sec-head' }, h('h3', { 'class': 'sec-title', text: '探针' }), h('span', { 'class': 'pc-where', text: head }), h('span', { 'class': 'sec-fill' }), acts),
      body, (sec && sec.found) || pb.family === 'wall' ? foot : null);
  }
  // ctx = {ui, rows, fieldId, onDelete(id), onClear(), onCopy(), onCsv()}
  function logSection(result, ctx) {
    var ui = ctx.ui, h = ui.h, rows = ctx.rows || [];
    if (!rows.length) return null;
    var fam = family(result), mainKey, secKey, mainLabel;
    if (fam === 'volume') { mainKey = 'speed_m_s'; secKey = 'section_flow_ml_s'; mainLabel = ['速度 m/s', 'Q mL/s']; }
    else {
      var f = wallFields(result).filter(function (x) { return x.id === ctx.fieldId; })[0] || wallFields(result)[0];
      mainKey = f ? rowKey(f) : 'wss_pa'; secKey = 'section_' + mainKey; mainLabel = [(f ? (f.short_label || f.id) : 'WSS') + ' 此点', '截面均值'];
    }
    var table = h('table', { 'class': 'tbl probe-log' },
      h('thead', null, h('tr', null, ['编号', '分支', 's mm', mainLabel[0], mainLabel[1], ''].map(function (t) { return h('th', { text: t }); }))),
      h('tbody', null, rows.map(function (r) {
        var v = r.values || {}, del = ui.iconButton('close', '删除 ' + r.id, function () { ctx.onDelete(r.id); }, { cls: 'btn-xs' });
        var main = v[mainKey] !== undefined ? v[mainKey] : (fam === 'volume' ? v.pressure_pa : undefined);
        return h('tr', null, h('td', { text: r.id }), h('td', { text: r.branch || '—' }), h('td', { 'class': 'num', text: fmt(r.s_from_root_mm) }),
          h('td', { 'class': 'num', text: fmt(main) }), h('td', { 'class': 'num', text: fmt(v[secKey]) }), h('td', null, del));
      })));
    return ui.section('探针记录', { tag: h('span', { 'class': 'badge', text: String(rows.length) }), actions: [
      ui.button('复制 TSV', ctx.onCopy, { kind: 'link', cls: 'btn-sm' }), ui.button('导出 CSV', ctx.onCsv, { kind: 'link', cls: 'btn-sm' }), ui.button('清空', ctx.onClear, { kind: 'link', cls: 'btn-sm' })] },
      h('div', { 'class': 'tscroll' }, table), ui.note('表里只列一两个数；复制或导出的文件含全部字段和截面读数，格式与经典报告相同。'));
  }

  // ------------------------------------------------------------------ hover readout (phase 3 lane 4: W41, V17)
  // A number in its display unit (ws_display: Pa / dyn/cm², Pa / mmHg, m/s / cm/s), {value, units}.
  function conv(v, units, fam) { var cv = ns.display && typeof ns.display.toDisplay === 'function' ? ns.display.toDisplay : null; return cv ? cv(v, units, fam) : { value: v, units: units }; }
  function qtext(q) { var U2 = ns.ui; return U2 && typeof U2.num === 'function' ? U2.num(q.value, q.units) : fmt(q.value) + (q.units ? ' ' + q.units : ''); }
  var TRUST_LABELS = { 1: '插值无支撑', 2: '表面粗糙', 4: '几何越界', 8: '采样支撑弱', 16: '邻近切口' };
  function trustText(m, bits, mask) {
    bits = Number(bits) & (mask || 31);
    if (!bits) return null;
    var src = (m && m.analysis && m.analysis.trust && Array.isArray(m.analysis.trust.sources)) ? m.analysis.trust.sources : [], out = [];
    [1, 2, 4, 8, 16].forEach(function (b) {
      if (!(bits & b)) return;
      var s0 = src.filter(function (x) { return Number(x && x.bit) === b; })[0];
      out.push((s0 && s0.label) || TRUST_LABELS[b]);
    });
    return out.length ? '可信提示：' + out.join('、') : null;
  }
  function fieldLabel(f, id) { return f ? (f.short_label || f.label || f.id) : id; }
  // The lines of the readout for a viewer pick e on result with field fieldId: [{text, kind: 'value'|'where'|'trust'|'xyz'}],
  // or null when there is nothing to read there.  Pure (tests).
  function hoverModel(result, e, fieldId) {
    if (!result || !e || e.marker !== undefined) return null;
    var m = result.manifest || {}, f = result.field ? result.field(fieldId) : null, fam = family(result), X = model(result), lines = [];
    var where = function (sid, s) { var t = [sid !== null && sid !== undefined && Number.isFinite(+sid) ? branchName(X, sid) : null, s !== null && s !== undefined && Number.isFinite(+s) ? '距入口 ' + fmt(s) + ' mm' : null].filter(Boolean).join(' · '); return t ? { text: t, kind: 'where' } : null; };
    if (fam === 'wall') {
      var disp = null, vi = e.vertexIndex;
      try { disp = result.fieldArray(fieldId, 'display'); } catch (_) { disp = null; }
      if (!disp || !(vi >= 0)) return null;
      var ok = Number.isFinite(disp[vi]) && (!Array.isArray(e.face) || e.face.every(function (q) { return Number.isFinite(disp[q]); }));   // classic: a partly unsupported triangle has no value
      lines.push(ok ? { text: fieldLabel(f, fieldId) + ' ' + qtext(conv(disp[vi], f && f.units, 'wall')), kind: 'value', note: '壁面插值' } : { text: '插值覆盖范围外：无读值', kind: 'value' });
      lines.push(where(e.segmentId, e.s_from_root_mm));
      lines.push(trustText(m, e.trust, 7) ? { text: trustText(m, e.trust, 7), kind: 'trust' } : null);
      if (X.V && 3 * vi + 2 < X.V.length) lines.push({ text: '(' + [0, 1, 2].map(function (c) { return X.V[3 * vi + c].toFixed(1); }).join(', ') + ') mm', kind: 'xyz' });
    } else if (e.pointIndex !== null && e.pointIndex !== undefined && e.pointIndex >= 0) {
      var i = e.pointIndex, vals = e.values || {}, ids = Object.keys(vals);
      ids.sort(function (a, b) { return (b === fieldId) - (a === fieldId); });
      ids.forEach(function (id) {
        var ff = result.field(id), x = vals[id];
        if (!ff || !Number.isFinite(x)) return;
        lines.push({ text: (id === 'pressure' ? '相对压力' : fieldLabel(ff, id)) + ' ' + qtext(conv(x, ff.units, 'volume')), kind: id === fieldId ? 'value' : 'also' });
      });
      if (!lines.length) return null;
      lines.push(where(e.segmentId, e.s_from_root_mm));
      var PT = arr(result, X.k.PT);
      if (PT && i < PT.length && trustText(m, PT[i], 28)) lines.push({ text: trustText(m, PT[i], 28), kind: 'trust' });
    } else if (e.vertexIndex !== null && e.vertexIndex !== undefined && e.vertexIndex >= 0) {
      if (!Number.isFinite(e.value)) lines.push({ text: '壁面压力：无插值支撑', kind: 'value' });
      else lines.push({ text: fieldLabel(f, fieldId) + ' ' + qtext(conv(e.value, f && f.units, 'volume')), kind: 'value' });
      lines.push(trustText(m, e.trust, 7) ? { text: trustText(m, e.trust, 7), kind: 'trust' } : null);
    } else return null;
    return lines.filter(Boolean);
  }
  function hoverOn() { try { return !(ns.display && ns.display.prefs && ns.display.prefs().opts && ns.display.prefs().opts.hover === false); } catch (_) { return true; } }
  var hovers = {};
  function shellState() { return ns.shell && typeof ns.shell.state === 'function' ? ns.shell.state() : null; }
  function sideOfViewer(v) { var S = shellState(); return S && v ? (v === S.viewerA ? 'a' : v === S.viewerB ? 'b' : null) : null; }
  function hideHover(side) { Object.keys(hovers).forEach(function (k) { if (!side || k === side) { var el = hovers[k]; if (el) el.hidden = true; } }); }
  function showHover(v, e) {
    var side = sideOfViewer(v), S = shellState();
    if (!side || !S || !S.els) return;
    if (!e || !e.screen || !hoverOn()) { hideHover(side); return; }
    var lines = null;
    try { lines = hoverModel(v.result(), e, v.field()); } catch (_) { lines = null; }
    if (!lines || !lines.length) { hideHover(side); return; }
    var vp = side === 'b' ? S.els.vpB : S.els.vpA, host = vp && vp.legend && vp.legend.parentNode, U2 = ns.ui;
    if (!host || !U2) return;
    var el = hovers[side];
    if (!el || el.parentNode !== host) { if (el && el.parentNode) el.parentNode.removeChild(el); el = U2.h('div', { 'class': 'wsp-hover', role: 'status', 'aria-live': 'off' }); host.appendChild(el); hovers[side] = el; }
    U2.fill(el, lines.map(function (l) { return U2.h('div', { 'class': 'wsp-hover-' + l.kind, title: l.note || null, text: l.text }); }));
    el.hidden = false;
    var W = host.clientWidth || 800, H = host.clientHeight || 600, w = el.offsetWidth || 160, hh = el.offsetHeight || 60;
    var x = e.screen.x + 14, y = e.screen.y + 16;
    if (x + w > W - 6) x = Math.max(6, e.screen.x - w - 14);
    if (y + hh > H - 6) y = Math.max(6, e.screen.y - hh - 12);
    el.style.transform = 'translate(' + Math.round(x) + 'px,' + Math.round(y) + 'px)';
  }
  function wireHover(v) {
    if (!v || v.__wspHover || typeof v.on !== 'function') return;
    v.__wspHover = true;
    v.on('hover', function (e) { showHover(v, e); });
  }
  (ns.ext = ns.ext || []).push({
    id: 'hover',
    onResult: function (api) { var S = api.state(); hideHover(); if (S) { wireHover(S.viewerA); wireHover(S.viewerB); } },
    toolbar: function (api) { var S = api.state(); if (S) { wireHover(S.viewerA); wireHover(S.viewerB); } return []; },   // viewer B appears with compare / split
    onField: function () { hideHover(); },
    onClose: function () { hideHover(); }
  });

  return {
    requiredArrays: requiredArrays, groups: groups, model: model, wallFields: wallFields, pick: pick, sectionIntegralAt: sectionIntegralAt, record: record,
    row: row, wallRow: wallRow, volumeRow: volumeRow, nextId: nextId, loadLog: loadLog, saveLog: saveLog, tsv: tsv, csv: csv,
    draw: draw, clear: clear, card: card, logSection: logSection, ROW_KEY: ROW_KEY,
    // phase 3 lane 4
    hoverModel: hoverModel, showHover: showHover, hideHover: function () { hideHover(); }, trustText: trustText
  };
});
