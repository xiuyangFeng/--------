/* WSS workspace v2 — annotations and automatic labels (second phase S4, scope §11).
 * Annotations are the classic document (PUT /api/jobs/<id>/annotations, wss-deploy.annotations/v1): ≤ 50 pins, each
 * {id A<k>, xyz_mm, text ≤ 200, color, branch, segment_id, s_from_root_mm, created_at}; the classic wall and volume
 * reports read and write the same file.  Pins are drawn whenever the result has any (a dot, a short leader and
 * 「A1 · text」); the 标注 tool places, edits and deletes them; saves are debounced and carry the job version.
 * Automatic labels (layers menu): branch names at mid-branch, the first N findings that are not rejected, and the
 * largest lumen section ring of the aorta.  Display only. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.annot = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var LIMIT = { items: 50, text: 200 };
  var PIN_HEX = 0xe0a84a, RING_HEX = 0xb07ce8, COLOR = '#d97706';
  function C() { return root.WssReportCommon || null; }
  function U() { return ns.util; }
  function fmt(v) { return v === null || v === undefined || !Number.isFinite(+v) ? '—' : (U() ? U().fmtSig(+v) : String(+(+v).toPrecision(3))); }

  // ------------------------------------------------------------------ the document
  function itemsOf(manifest) {
    var a = manifest && manifest.analysis && manifest.analysis.annotations;
    var list = a && Array.isArray(a.items) ? a.items : (Array.isArray(a) ? a : []);
    return list.filter(function (x) { return x && Array.isArray(x.xyz_mm) && x.xyz_mm.length === 3 && x.xyz_mm.every(function (v) { return Number.isFinite(+v); }); }).slice(0, LIMIT.items);
  }
  function add(items, result, xyz, text) {
    if (items.length >= LIMIT.items) return { error: '标注最多 ' + LIMIT.items + ' 条。' };
    text = String(text || '').trim().slice(0, LIMIT.text);
    if (!text) return { error: '请写标注文字。' };
    var c = C(), groups = ns.probe ? ns.probe.groups(result) : [], pr = null, arc = null;
    if (c && groups.length) { try { pr = c.projectToCenterline(groups, xyz); arc = c.arcFromRoot(groups, xyz); } catch (_) { pr = null; } }
    var a = { id: c ? c.newId('A', items) : 'A' + (items.length + 1), xyz_mm: xyz.map(function (v) { return +(+v).toFixed(3); }), text: text, color: COLOR, created_at: new Date().toISOString() };
    if (pr) { a.branch = String(pr.name || '').slice(0, 40); a.segment_id = pr.segment_id; }
    if (arc && Number.isFinite(arc.s_from_root_mm)) a.s_from_root_mm = +arc.s_from_root_mm.toFixed(2);
    return { items: items.concat([a]), item: a };
  }
  function edit(items, id, text) {
    text = String(text || '').trim().slice(0, LIMIT.text);
    if (!text) return items.filter(function (a) { return a.id !== id; });   // the classic rule: emptied text deletes
    return items.map(function (a) { return a.id === id ? Object.assign({}, a, { text: text }) : a; });
  }
  function remove(items, id) { return items.filter(function (a) { return a.id !== id; }); }

  // ------------------------------------------------------------------ drawing
  function diagOf(result) {
    var X = ns.probe ? ns.probe.model(result) : null;
    return X && X.diag ? X.diag : 200;
  }
  function clearGroup(viewer, key) {
    var g = viewer && viewer['__' + key];
    if (!g) return;
    if (g.parent) g.parent.remove(g);
    g.traverse(function (o) { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    viewer['__' + key] = null;
  }
  // Pins: a dot on the wall, a short leader upwards and the label at its tip.
  function drawPins(viewer, result, items) {
    var THREE = root.THREE;
    clearGroup(viewer, 'annotGroup');
    if (typeof viewer.setToolLabels === 'function') viewer.setToolLabels('annot', []);
    if (!THREE || !viewer || typeof viewer.toolOverlay !== 'function') return;
    if (!items || !items.length) { viewer.render(); return; }
    var g = new THREE.Group(); g.name = 'annotations';
    var lift = Math.max(2, diagOf(result) * 0.04), labels = [];
    var up = [0, 0, 1];
    var fr = viewer.frame ? viewer.frame() : null;
    if (fr && Array.isArray(fr.rotation) && Array.isArray(fr.rotation[2])) up = fr.rotation[2].slice(0, 3);   // the patient's superior axis
    var lineMat = new THREE.LineBasicMaterial({ color: PIN_HEX, depthTest: false });
    items.forEach(function (a) {
      var p = a.xyz_mm, tip = [p[0] + up[0] * lift, p[1] + up[1] * lift, p[2] + up[2] * lift];
      var dot = new THREE.Group(); dot.position.set(p[0], p[1], p[2]); dot.userData.screenPx = 4;
      var halo = new THREE.Mesh(new THREE.SphereGeometry(1.45, 12, 8), new THREE.MeshBasicMaterial({ color: 0x0b1118, depthTest: false })); halo.renderOrder = 14;
      var core = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), new THREE.MeshBasicMaterial({ color: PIN_HEX, depthTest: false })); core.renderOrder = 15;
      dot.add(halo); dot.add(core); g.add(dot);
      var lg = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(p[0], p[1], p[2]), new THREE.Vector3(tip[0], tip[1], tip[2])]);
      var line = new THREE.Line(lg, lineMat); line.renderOrder = 13; g.add(line);
      labels.push({ xyz: tip, text: a.id + ' · ' + (a.text.length > 24 ? a.text.slice(0, 23) + '…' : a.text) });
    });
    viewer.toolOverlay().add(g);
    viewer.__annotGroup = g;
    if (typeof viewer.setToolLabels === 'function') viewer.setToolLabels('annot', labels);
    viewer.render();
  }
  // Automatic labels.  opts = {branches, findings (N), maxd, findingsList (ordered, rejected already removed)}
  function drawAuto(viewer, result, manifest, opts) {
    var THREE = root.THREE;
    clearGroup(viewer, 'autoGroup');
    if (!viewer || typeof viewer.setToolLabels !== 'function') return;
    var labels = [];
    opts = opts || {};
    if (opts.branches && ns.probe) {
      ns.probe.groups(result).forEach(function (g) {
        if (!g || !g.s || !g.s.length || !(g.length_mm > 0)) return;
        var half = g.length_mm / 2, j = 0, bd = Infinity;
        for (var i = 0; i < g.s.length; i++) { var d = Math.abs(g.s[i] - half); if (d < bd) { bd = d; j = i; } }   // the classic branchMidpoint
        labels.push({ xyz: [g.xyz[3 * j], g.xyz[3 * j + 1], g.xyz[3 * j + 2]], text: g.name || ('分支 ' + g.segment_id) });
      });
    }
    var list = (opts.findingsList || []).filter(function (it) { return Array.isArray(it.xyz_mm) && it.xyz_mm.length === 3; }).slice(0, Math.max(0, opts.findings || 0));
    list.forEach(function (it) {
      var text = it.manual ? it.id + ' ' + (it.text.length > 20 ? it.text.slice(0, 19) + '…' : it.text) : it.id + ' ' + (ns.overview ? ns.overview.kindLabel(it) + ' ' + ns.overview.findingValue(it) : '');
      labels.push({ xyz: it.xyz_mm, text: text });
    });
    var mm = manifest && manifest.analysis && manifest.analysis.morphology && manifest.analysis.morphology.aorta && manifest.analysis.morphology.aorta.max;
    if (opts.maxd && mm && THREE && typeof viewer.toolOverlay === 'function') {
      var ring = Array.isArray(mm.polygon_world) ? mm.polygon_world.filter(function (q) { return Array.isArray(q) && q.length === 3; }) : [];
      var g2 = new THREE.Group(); g2.name = 'auto-maxd';
      if (ring.length > 2) {
        var loop = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(ring.map(function (q) { return new THREE.Vector3(q[0], q[1], q[2]); })),
          new THREE.LineBasicMaterial({ color: RING_HEX, depthTest: false, transparent: true, opacity: 0.95 }));
        loop.renderOrder = 11; g2.add(loop);
      }
      viewer.toolOverlay().add(g2); viewer.__autoGroup = g2;
      if (Array.isArray(mm.xyz_mm)) labels.push({ xyz: ring.length ? ring[0] : mm.xyz_mm, text: '管腔最大直径 ' + fmt(mm.max_diameter_mm) + ' mm' });
    }
    viewer.setToolLabels('auto', labels);
    viewer.render();
  }
  function clearAll(viewer) {
    if (!viewer) return;
    clearGroup(viewer, 'annotGroup'); clearGroup(viewer, 'autoGroup');
    if (typeof viewer.setToolLabels === 'function') { viewer.setToolLabels('annot', []); viewer.setToolLabels('auto', []); }
  }

  // ------------------------------------------------------------------ saving (same shape as the findings saver)
  // opts = {put(payload) → Promise<{annotations, version}>, version(), onSaved(items, version), onError(err), delay}
  function saver(opts) {
    var timer = null, pending = null, busy = false;
    function flush() {
      timer = null;
      if (!pending || busy) return;
      var items = pending; pending = null; busy = true;
      var payload = { items: items }, v = opts.version ? opts.version() : undefined;
      if (v !== undefined && v !== null) payload.version = v;
      Promise.resolve(opts.put(payload)).then(function (r) {
        busy = false;
        if (opts.onSaved) opts.onSaved(r && r.annotations && Array.isArray(r.annotations.items) ? r.annotations.items : items, r && r.version);
        if (pending) schedule();
      }, function (e) { busy = false; pending = null; if (opts.onError) opts.onError(e); });
    }
    function schedule() { if (timer) clearTimeout(timer); timer = setTimeout(flush, opts.delay === undefined ? 600 : opts.delay); }
    return { save: function (items) { pending = items; schedule(); }, busy: function () { return busy || Boolean(pending); } };
  }

  // ------------------------------------------------------------------ the 标注 page
  // ctx = {ui, items, editable, lockedText, placing, onPlace(), onEdit(id, text), onRemove(id), onFly(xyz), onClose()}
  function panel(body, ctx) {
    var ui = ctx.ui, h = ui.h, items = ctx.items || [];
    var place = ui.button(ctx.placing ? '在管壁上点一处…' : '在管壁上钉标注', ctx.onPlace, { kind: ctx.placing ? null : 'primary', cls: 'btn-sm' + (ctx.placing ? ' on' : ''), disabled: !ctx.editable || items.length >= LIMIT.items });
    var rows = items.map(function (a) {
      var inp = h('input', { type: 'text', 'class': 'annot-text', maxLength: LIMIT.text, value: a.text, disabled: !ctx.editable, 'aria-label': a.id + ' 文字' });
      inp.addEventListener('change', function () { ctx.onEdit(a.id, inp.value); });
      return h('div', { 'class': 'annot-row' }, h('b', { 'class': 'meas-id', text: a.id }), h('div', { 'class': 'annot-main' }, inp,
          h('span', { 'class': 'muted annot-where', text: [a.branch, Number.isFinite(+a.s_from_root_mm) ? '距入口 ' + fmt(a.s_from_root_mm) + ' mm' : null].filter(Boolean).join(' · ') })),
        h('span', { 'class': 'meas-acts' }, ui.iconButton('eye', '转到这里', function () { ctx.onFly(a.xyz_mm); }),
          ctx.editable ? ui.iconButton('close', '删除', function () { ctx.onRemove(a.id); }) : null));
    });
    ui.fill(body, ui.section('标注', { cls: 'sec-annot', tag: items.length ? h('span', { 'class': 'badge', text: String(items.length) }) : null,
        actions: [ui.iconButton('close', '关闭标注', ctx.onClose)] },
      h('div', { 'class': 'slice-ctl' }, place, h('span', { 'class': 'muted', text: ctx.editable ? '最多 ' + LIMIT.items + ' 条，每条 ≤ ' + LIMIT.text + ' 字；自动保存到服务' : (ctx.lockedText || '只读') })),
      rows.length ? h('div', { 'class': 'annot-list' }, rows) : ui.note('还没有标注。标注存在服务上，经典报告里也能看到。')));
  }
  // Text for a new pin.  cb(text).
  function addDialog(ui, where, cb) {
    var h = ui.h, inp = h('input', { type: 'text', 'class': 'annot-text', maxLength: LIMIT.text, placeholder: '标注文字（≤ 200 字）', 'aria-label': '标注文字' });
    var ok = ui.button('钉上', function () { var t = inp.value.trim(); if (!t) { inp.focus(); return; } dlg.close('done'); cb(t); }, { kind: 'primary' });
    inp.addEventListener('keydown', function (e) { if (e.key === 'Enter') ok.click(); });
    var dlg = ui.dialog.open({ title: '新标注', body: [h('p', { 'class': 'muted', text: where }), inp], actions: [ui.button('取消', function () { dlg.close('cancel'); }), ok], focus: inp });
    return dlg;
  }

  return { LIMIT: LIMIT, itemsOf: itemsOf, add: add, edit: edit, remove: remove, drawPins: drawPins, drawAuto: drawAuto, clearAll: clearAll, saver: saver, panel: panel, addDialog: addDialog };
});
