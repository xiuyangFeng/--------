/* WSS workspace v2 — evidence bookmarks D10 (contract §6.3 ws_bookmarks.js).
 * A bookmark records the result identity, data and viewer versions, field, window, camera, cursor, selection and
 * time index.  Stored per result in this browser; import / export as JSON; carried into the offline export.
 * A bookmark from another result or another data version is never applied — it is shown as incompatible. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.bookmarks = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };
  var store = function () { return ns.store; };
  var SCHEMA = 'wssv2.bookmark/1', FILE_SCHEMA = 'wssv2.bookmarks/1';

  function uid() { return 'b' + Date.now().toString(36) + Math.floor(Math.random() * 1e6).toString(36); }
  function list(runIdentity) { return store().readBookmarks(runIdentity); }
  function save(runIdentity, items) { return store().writeBookmarks(runIdentity, items); }

  // ctx: {jobId, runIdentity, arraysVersion, dataVersion, viewerVersion, field, window, windowLabel, layout, state}
  function make(ctx, meta) {
    meta = meta || {};
    var st = ctx.state || {};
    return {schema: SCHEMA, id: uid(), name: String(meta.name || '').slice(0, 80) || '未命名书签', note: String(meta.note || '').slice(0, 300),
      created_at: new Date().toISOString(), job_id: ctx.jobId || null, run_identity: ctx.runIdentity || null,
      arrays_version: ctx.arraysVersion || null, data_version: ctx.dataVersion || null,
      viewer_version: ctx.viewerVersion || null, field: ctx.field || st.field || null, window: ctx.window || null, window_label: ctx.windowLabel || '',
      camera: st.camera || null, cursor: st.cursor || null, selection: st.selection === undefined ? null : st.selection,
      time_index: st.time_index === undefined ? 0 : st.time_index, layout: ctx.layout || null, state: st};
  }
  function compatible(bm, ctx) {
    var reasons = [];
    if (!bm || bm.schema !== SCHEMA) reasons.push('格式不认识');
    else {
      // Compatibility = same result identity + same array contract.  data_version also moves when only the
      // review or the conclusion changes, so it is recorded but never blocks a bookmark.
      if (bm.run_identity !== ctx.runIdentity) reasons.push('属于另一份结果');
      else if (bm.arrays_version && ctx.arraysVersion && bm.arrays_version !== ctx.arraysVersion) reasons.push('结果数据已重建（数组版本不同）');
      if (bm.field && ctx.fields && ctx.fields.indexOf(bm.field) < 0) reasons.push('这份结果没有字段 ' + bm.field);
    }
    return {ok: reasons.length === 0, reasons: reasons};
  }
  function add(ctx, bm) { var items = list(ctx.runIdentity); items.push(bm); save(ctx.runIdentity, items); return items; }
  function remove(runIdentity, id) { var items = list(runIdentity).filter(function (b) { return b.id !== id; }); save(runIdentity, items); return items; }
  function move(runIdentity, id, delta) {
    var items = list(runIdentity);
    var i = items.findIndex(function (b) { return b.id === id; });
    var j = i + delta;
    if (i < 0 || j < 0 || j >= items.length) return items;
    var tmp = items[i]; items[i] = items[j]; items[j] = tmp;
    save(runIdentity, items);
    return items;
  }
  function exportJson(runIdentity, items) {
    return JSON.stringify({schema: FILE_SCHEMA, run_identity: runIdentity, exported_at: new Date().toISOString(), items: items || list(runIdentity)}, null, 2);
  }
  // Returns {added, foreign, stale, errors}.  Foreign bookmarks are kept under their own result, never re-targeted.
  function importJson(text, ctx) {
    var out = {added: 0, foreign: 0, stale: 0, errors: []};
    var doc;
    try { doc = JSON.parse(text); } catch (_) { out.errors.push('文件不是 JSON。'); return out; }
    var items = Array.isArray(doc) ? doc : (doc && Array.isArray(doc.items) ? doc.items : null);
    if (!items) { out.errors.push('文件里没有书签列表。'); return out; }
    var byRun = {};
    items.forEach(function (bm) {
      if (!bm || typeof bm !== 'object' || bm.schema !== SCHEMA || !bm.run_identity) { out.errors.push('跳过一条格式不认识的书签。'); return; }
      (byRun[bm.run_identity] = byRun[bm.run_identity] || []).push(bm);
    });
    Object.keys(byRun).forEach(function (run) {
      var existing = list(run);
      var ids = {}; existing.forEach(function (b) { ids[b.id] = true; });
      byRun[run].forEach(function (bm) {
        if (ids[bm.id]) return;
        existing.push(bm); ids[bm.id] = true;
        if (run === ctx.runIdentity) {
          out.added += 1;
          if (bm.arrays_version && ctx.arraysVersion && bm.arrays_version !== ctx.arraysVersion) out.stale += 1;
        } else out.foreign += 1;
      });
      save(run, existing);
    });
    return out;
  }
  // Offline package: bookmarks embedded at export time join the local list once.
  function seed(runIdentity, items) {
    if (!runIdentity || !Array.isArray(items) || !items.length) return;
    var existing = list(runIdentity);
    var ids = {}; existing.forEach(function (b) { ids[b.id] = true; });
    var changed = false;
    items.forEach(function (bm) { if (bm && bm.id && !ids[bm.id] && bm.run_identity === runIdentity) { existing.push(bm); changed = true; } });
    if (changed) save(runIdentity, existing);
  }

  // ------------------------------------------------------------------ panel
  // ctx: {runIdentity, arraysVersion, fields, fieldLabel(id), onSave(), onRestore(bm), onChanged()}
  function render(el, ctx) {
    var h = ui().h;
    var items = list(ctx.runIdentity);
    var file = h('input', {type: 'file', accept: '.json,application/json', 'class': 'visually-hidden', 'aria-label': '选择书签文件'});
    file.addEventListener('change', function () {
      var f = file.files && file.files[0];
      if (!f || typeof f.text !== 'function') return;
      f.text().then(function (text) {
        var r = importJson(text, ctx);
        var parts = [];
        if (r.added) parts.push('导入 ' + r.added + ' 条');
        if (r.stale) parts.push(r.stale + ' 条来自重建前的数据，只列出不套用');
        if (r.foreign) parts.push(r.foreign + ' 条属于其他结果，已按原结果保存，打开那份结果时可用');
        if (r.errors.length) parts.push(r.errors[0]);
        ui().toast(parts.length ? parts.join('；') + '。' : '文件里没有新书签。', {kind: r.errors.length && !r.added ? 'error' : 'info'});
        render(el, ctx);
      });
    });
    var head = h('div', {'class': 'bm-head'},
      ui().button('保存当前视图', function () { if (ctx.onSave) ctx.onSave(); }, {icon: 'bookmark', title: '快捷键 B'}),
      h('span', {'class': 'sec-fill'}),
      ui().button('导入', function () { file.click(); }, {kind: 'link', cls: 'btn-sm'}),
      ui().button('导出', function () {
        var blob = new root.Blob([exportJson(ctx.runIdentity, items)], {type: 'application/json'});
        ui().downloadBlob(blob, 'WSS_书签_' + (ctx.fileTag || 'result') + '.json');
      }, {kind: 'link', cls: 'btn-sm', disabled: !items.length}), file);
    var body = h('div', {'class': 'bm-list'});
    if (!items.length) body.appendChild(ui().empty('还没有书签。找到要讲的位置后按 B 或点「保存当前视图」，写一句观察；书签只存在这台电脑的浏览器里，可导出后发给别人。'));
    items.forEach(function (bm, i) {
      var c = compatible(bm, ctx);
      var meta = [ctx.fieldLabel ? ctx.fieldLabel(bm.field) : bm.field, bm.window_label, ui().time(bm.created_at)].filter(Boolean).join(' · ');
      var row = h('div', {'class': 'bm' + (c.ok ? '' : ' bm-bad'), dataset: {bookmarkId: bm.id}},
        h('div', {'class': 'bm-name', text: bm.name}),
        bm.note ? h('div', {'class': 'bm-note', text: bm.note}) : null,
        h('div', {'class': 'bm-meta', text: meta}),
        c.ok ? null : h('div', {'class': 'bm-why', text: '不能套用：' + c.reasons.join('；')}),
        h('div', {'class': 'bm-actions'},
          ui().button('打开', function () { if (ctx.onRestore) ctx.onRestore(bm); }, {cls: 'btn-sm', disabled: !c.ok}),
          ui().button('上移', function () { move(ctx.runIdentity, bm.id, -1); render(el, ctx); }, {kind: 'link', cls: 'btn-sm', disabled: i === 0}),
          ui().button('下移', function () { move(ctx.runIdentity, bm.id, 1); render(el, ctx); }, {kind: 'link', cls: 'btn-sm', disabled: i === items.length - 1}),
          ui().button('删除', function () { remove(ctx.runIdentity, bm.id); render(el, ctx); if (ctx.onChanged) ctx.onChanged(); }, {kind: 'link', cls: 'btn-sm'})));
      body.appendChild(row);
    });
    ui().fill(el, head, body, ui().note(ctx.offline ? '书签不是复核签字；在这份离线报告里新存的书签只留在这台电脑的浏览器里。' : '书签不是复核签字；只保存在本机，导出离线报告时会一起带上。'));
  }
  // Save dialog: name + one sentence.
  function saveDialog(ctx, onDone) {
    var h = ui().h;
    var name = h('input', {type: 'text', maxLength: 80, value: ctx.defaultName || '', 'aria-label': '书签名称'});
    var note = h('textarea', {rows: 2, maxLength: 300, placeholder: '一句观察或疑问（可空）', 'aria-label': '备注'});
    var ok = ui().button('保存', function () {
      var bm = make(ctx, {name: name.value.trim() || ctx.defaultName, note: note.value.trim()});
      add(ctx, bm);
      ui().dialog.close('done');
      if (onDone) onDone(bm);
    }, {kind: 'primary'});
    ui().dialog.open({title: '保存书签', body: [h('label', {'class': 'fld'}, h('span', {text: '名称'}), name), h('label', {'class': 'fld'}, h('span', {text: '备注'}), note),
      ui().note('记录：' + [ctx.fieldLabel ? ctx.fieldLabel(ctx.field) : ctx.field, ctx.windowLabel, '当前视角', ctx.hasCursor ? '游标位置' : '', ctx.hasSelection ? '选中点' : ''].filter(Boolean).join('、') + '。')],
      actions: [ui().button('取消', function () { ui().dialog.close('cancel'); }), ok], focus: name});
  }

  return {SCHEMA: SCHEMA, list: list, make: make, compatible: compatible, add: add, remove: remove, move: move,
    exportJson: exportJson, importJson: importJson, seed: seed, render: render, saveDialog: saveDialog};
});
