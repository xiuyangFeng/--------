/* WSS workspace v2 — findings review (second phase S4, scope §11).
 * The review document is the classic one (PUT /api/jobs/<id>/findings_review, wss-deploy.findings_review/v1):
 * items {finding id: {decision: confirmed | rejected | null, note ≤ 500}} and added [manual findings ≤ 30].  The
 * classic wall and volume reports read and write the same document, so a decision made here shows there and back.
 * Rules kept from the classic pages: an entry with neither decision nor note is dropped (volume report), a manual
 * finding's id is M<k> not used by any finding, its branch / arc come from the classic centreline projection.
 * Saves are debounced and carry the job version (a 409 — locked by the technical review, or changed elsewhere —
 * reloads).  Nothing is editable offline or once the result is locked. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.review = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var LIMIT = { note: 500, text: 500, added: 30, items: 200 };
  var DECISIONS = ['confirmed', 'rejected'];

  // ------------------------------------------------------------------ the document
  function docOf(manifest) {
    var f = manifest && manifest.analysis && manifest.analysis.findings;
    var r = f && f.review && typeof f.review === 'object' ? f.review : {};
    return { items: r.items && typeof r.items === 'object' ? r.items : {}, added: Array.isArray(r.added) ? r.added : [], legacy: Array.isArray(r.legacy) ? r.legacy : [] };
  }
  function clone(doc) { return { items: JSON.parse(JSON.stringify(doc.items || {})), added: JSON.parse(JSON.stringify(doc.added || [])) }; }
  function entry(doc, id) { var e = doc.items && doc.items[id]; return e && typeof e === 'object' ? e : null; }
  function decisionOf(doc, id) { var e = entry(doc, id); return e && DECISIONS.indexOf(e.decision) >= 0 ? e.decision : null; }
  function noteOf(doc, id) { var e = entry(doc, id); return e && typeof e.note === 'string' ? e.note : ''; }
  function tidy(doc, id) {
    var e = doc.items[id];
    if (e && !(DECISIONS.indexOf(e.decision) >= 0) && !(typeof e.note === 'string' && e.note.trim())) delete doc.items[id];
    return doc;
  }
  function setDecision(doc, id, decision) {
    var d = clone(doc), e = d.items[id] || {};
    d.items[id] = { decision: DECISIONS.indexOf(decision) >= 0 ? decision : null, note: typeof e.note === 'string' ? e.note : '' };
    return tidy(d, id);
  }
  function setNote(doc, id, note) {
    var d = clone(doc), e = d.items[id] || {};
    d.items[id] = { decision: DECISIONS.indexOf(e.decision) >= 0 ? e.decision : null, note: String(note || '').slice(0, LIMIT.note) };
    return tidy(d, id);
  }
  // 「其余按自动结果确认」: undecided automatic findings become confirmed, their notes kept.
  function confirmRest(doc, ids) {
    var d = clone(doc);
    ids.forEach(function (id) { var e = d.items[id] || {}; d.items[id] = { decision: 'confirmed', note: typeof e.note === 'string' ? e.note : '' }; });
    return d;
  }
  function nextManualId(doc, items) {
    var used = {};
    (items || []).forEach(function (it) { if (it && it.id) used[it.id] = 1; });
    (doc.added || []).forEach(function (a) { if (a && a.id) used[a.id] = 1; });
    var k = 1; while (used['M' + k]) k++;
    return 'M' + k;
  }
  // A manual finding at a wall point (the classic addManualFinding, wall flavour: branch and arc from the centreline tree).
  function addManual(doc, result, items, xyz, text, severity) {
    var d = clone(doc);
    if (d.added.length >= LIMIT.added) return { error: '人工发现最多 ' + LIMIT.added + ' 条。' };
    text = String(text || '').trim().slice(0, LIMIT.text);
    if (!text) return { error: '请写一句说明。' };
    var c = root.WssReportCommon, groups = ns.probe ? ns.probe.groups(result) : [], pr = null, arc = null;
    if (c && groups.length) { try { pr = c.projectToCenterline(groups, xyz); arc = c.arcFromRoot(groups, xyz); } catch (_) { pr = null; } }
    var a = { id: nextManualId(d, items), xyz_mm: xyz.map(function (v) { return +(+v).toFixed(3); }), text: text, kind: 'manual', severity: severity === 'attention' ? 'attention' : 'note',
      created_at: new Date().toISOString() };
    if (pr) { a.branch = String(pr.name || '').slice(0, 40); a.segment_id = pr.segment_id; }
    if (arc && Number.isFinite(arc.s_from_root_mm)) a.s_from_root_mm = +arc.s_from_root_mm.toFixed(2);
    d.added.push(a);
    return { doc: d, item: a };
  }
  function removeManual(doc, id) {
    var d = clone(doc);
    d.added = d.added.filter(function (a) { return a.id !== id; });
    delete d.items[id];
    return d;
  }
  // A manual finding as a finding object for lists, markers and the lens.
  function manualItem(a) {
    return { id: a.id, kind: 'manual', manual: true, label: a.text, text: a.text, severity: a.severity || 'note', xyz_mm: a.xyz_mm, branch: a.branch || '',
      segment_id: a.segment_id, s_from_root_mm: a.s_from_root_mm, extent_mm: 10, created_at: a.created_at, definition: '审阅人手工标记的发现。' };
  }

  // ------------------------------------------------------------------ saving
  // opts = {put(payload) → Promise<{findings_review, version}>, version() → n, onSaved(doc, version), onError(err), delay}
  function saver(opts) {
    var timer = null, pending = null, busy = false, state = 'idle';
    function flush() {
      timer = null;
      if (!pending || busy) return;
      var doc = pending; pending = null; busy = true; state = 'saving';
      var payload = { items: doc.items, added: doc.added };
      var v = opts.version ? opts.version() : undefined;
      if (v !== undefined && v !== null) payload.version = v;
      Promise.resolve(opts.put(payload)).then(function (r) {
        busy = false; state = 'saved';
        if (opts.onSaved) opts.onSaved(r && r.findings_review ? r.findings_review : doc, r && r.version);
        if (pending) schedule();
      }, function (e) {
        busy = false; state = 'error'; pending = null;
        if (opts.onError) opts.onError(e);
      });
    }
    function schedule() { if (timer) clearTimeout(timer); timer = setTimeout(flush, opts.delay === undefined ? 600 : opts.delay); }
    return {
      save: function (doc, now) { pending = doc; state = 'pending'; if (now) { if (timer) clearTimeout(timer); flush(); } else schedule(); },
      state: function () { return state; }, busy: function () { return busy || Boolean(pending); }
    };
  }

  // ------------------------------------------------------------------ UI
  // The decision bar above a selected finding's evidence.  ctx = {ui, item, doc, editable, lockedText, onDecision(d), onNote(t), onRemove(), status}
  function bar(ctx) {
    var ui = ctx.ui, h = ui.h, it = ctx.item, id = it.id, dec = decisionOf(ctx.doc, id), note = noteOf(ctx.doc, id);
    var opts = [{ id: 'confirmed', label: '确认', icon: 'check' }, { id: 'rejected', label: '驳回', icon: 'close' }, { id: null, label: '未判定' }];
    var seg = h('div', { 'class': 'seg seg-slice seg-decision', role: 'group', 'aria-label': '判定' }, opts.map(function (o) {
      var on = dec === o.id, b = h('button', { type: 'button', 'class': 'seg-btn dec-' + (o.id || 'none') + (on ? ' on' : ''), 'aria-pressed': String(on), disabled: !ctx.editable },
        o.icon ? ui.icon(o.icon) : null, h('span', { text: o.label }));
      b.addEventListener('click', function () { if (!on && ctx.onDecision) ctx.onDecision(o.id); });
      return b;
    }));
    var input = h('textarea', { 'class': 'dec-note', rows: 2, maxLength: LIMIT.note, placeholder: ctx.editable ? '备注（可选，≤ 500 字）' : '', disabled: !ctx.editable, 'aria-label': '备注' });
    input.value = note;
    input.addEventListener('change', function () { if (ctx.onNote) ctx.onNote(input.value); });
    var foot = h('div', { 'class': 'dec-foot' },
      h('span', { 'class': 'muted', text: !ctx.editable ? (ctx.lockedText || '只读') : ctx.status || (it.manual ? '人工发现 · 判定和备注自动保存' : '判定和备注自动保存') }),
      h('span', { 'class': 'sec-fill' }),
      it.manual && ctx.editable && ctx.onRemove ? ui.button('删除这条人工发现', ctx.onRemove, { kind: 'link', cls: 'btn-sm' }) : null);
    return h('section', { 'class': 'sec sec-decision' }, h('div', { 'class': 'sec-head' }, h('h3', { 'class': 'sec-title', text: '判定' }), it.manual ? h('span', { 'class': 'tag tag-manual', text: '人工' }) : null), seg, input, foot);
  }
  // Text and severity for a new manual finding.  cb({text, severity}).
  function addDialog(ui, where, cb) {
    var h = ui.h, sev = 'note';
    var ta = h('textarea', { 'class': 'dec-note', rows: 3, maxLength: LIMIT.text, placeholder: '这里看到了什么（≤ 500 字）', 'aria-label': '说明' });
    var seg = h('div', { 'class': 'seg seg-slice' }, [{ id: 'note', label: '提示' }, { id: 'attention', label: '关注' }].map(function (o) {
      var b = h('button', { type: 'button', 'class': 'seg-btn' + (o.id === sev ? ' on' : ''), text: o.label });
      b.addEventListener('click', function () { sev = o.id; Array.prototype.forEach.call(seg.children, function (x) { x.classList.toggle('on', x === b); }); });
      return b;
    }));
    var ok = ui.button('加入发现', function () { var t = ta.value.trim(); if (!t) { ta.focus(); return; } dlg.close('done'); cb({ text: t, severity: sev }); }, { kind: 'primary' });
    var dlg = ui.dialog.open({ title: '新增人工发现', body: [h('p', { 'class': 'muted', text: where }), ta, h('div', { 'class': 'slice-ctl' }, h('span', { 'class': 'slice-ctl-k', text: '程度' }), seg)],
      actions: [ui.button('取消', function () { dlg.close('cancel'); }), ok], focus: ta });
    return dlg;
  }

  // ------------------------------------------------------------------ P3 lane 3 (#82): facts above the sign-off
  // The classic workbench's reviewChecklist (app.js) item by item: outlet naming source and confidence, result quality,
  // geometry reference range, and 「需要注意」 = the title of WssWorkbenchCore.alertModel (ported here; that file goes
  // with the classic workbench).  Quality words only — never the ensemble's dispersion numbers.
  function obj(v) { return v && typeof v === 'object' && !Array.isArray(v) ? v : {}; }
  function numOrNull(v) { if (v === null || v === undefined || v === '' || typeof v === 'boolean') return null; var n = Number(v); return Number.isFinite(n) ? n : null; }
  function fmt(v, d) { var n = numOrNull(v); return n === null ? '—' : n.toFixed(d === undefined ? 1 : d); }
  function alertTitle(manifest) {
    var a = obj(manifest && manifest.analysis), ref = obj(a.reference_assessment), q = obj(a.quality);
    var outside = (Array.isArray(ref.checks) ? ref.checks : []).filter(function (c) { return c && typeof c === 'object' && c.status === 'review'; });
    var refCount = outside.length;
    if (ref.status === 'review' && !outside.length) refCount = (Array.isArray(ref.reasons) ? ref.reasons : []).filter(function (r) { return typeof r === 'string' && r.trim(); }).length;
    var qualityAlert = Boolean(q.level && q.level !== 'good');
    var parts = [];
    if (refCount) parts.push(outside.length ? outside.length + ' 项几何测量超出模型训练范围' : '几何测量超出模型训练范围');
    if (qualityAlert) parts.push(q.level === 'poor' ? '模型集成离散度较高' : '模型集成存在不确定性');
    return parts.length ? parts.join('，') + '，结果需要复核。' : null;
  }
  function checklistModel(manifest, job) {
    var a = obj(manifest && manifest.analysis), items = [];
    var hist = job && Array.isArray(job.mapping_history) ? job.mapping_history : [];
    var mapping = hist.length ? hist[hist.length - 1] : null;
    items.push(['出口命名', mapping && mapping.source === 'automatic_high_confidence' ? '自动确认（置信度 ' + fmt(Number(mapping.confidence) * 100, 1) + '%）' : mapping ? '已人工确认' : '—']);
    var q = a.quality && typeof a.quality === 'object' && Object.keys(a.quality).length ? a.quality : null;
    if (q) items.push(['结果质量', q.level === 'good' || q.label === '模型集成稳定' ? '多模型一致（一致不代表准确）' : (q.label || q.level || '未评估')]);
    var ref = obj(a.reference_assessment);
    items.push(['几何参考范围', ref.status === 'pass' ? '在参考范围内' : ref.status === 'review' ? '有超出范围的测量，请看结果页提示' : '当前发布包未配置']);
    var alert = alertTitle(manifest);
    if (alert) items.push(['需要注意', alert]);
    return items;
  }
  // ui = ns.ui → the facts as a definition list (the sign-off dialog's first block)
  function checklist(ui, manifest, job) {
    var h = ui.h, items = checklistModel(manifest, job), dl = h('dl', { 'class': 'review-facts' });
    items.forEach(function (it) {
      var warn = it[0] === '需要注意' || (it[0] === '几何参考范围' && /超出/.test(it[1])) || (it[0] === '结果质量' && !/多模型一致/.test(it[1]));
      dl.appendChild(h('dt', { text: it[0] }));
      dl.appendChild(h('dd', { 'class': warn ? 'warn-text' : null, text: it[1] }));
    });
    return dl;
  }

  return { LIMIT: LIMIT, docOf: docOf, decisionOf: decisionOf, noteOf: noteOf, setDecision: setDecision, setNote: setNote, confirmRest: confirmRest,
    addManual: addManual, removeManual: removeManual, manualItem: manualItem, nextManualId: nextManualId, saver: saver, bar: bar, addDialog: addDialog,
    alertTitle: alertTitle, checklistModel: checklistModel, checklist: checklist };
});
