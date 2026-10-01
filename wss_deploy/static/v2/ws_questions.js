/* WSS workspace v2 — open by question D8 (contract §6.3 ws_questions.js).
 * Three questions; each only changes the display (field, window, layout, markers), says what it applied,
 * and can be undone back to free browsing.  Nothing is recomputed and the summary is not changed. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.questions = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };

  function hasField(m, id) { return ((m && m.fields) || []).some(function (f) { return f && f.id === id; }); }
  function field(m, id) { return ((m && m.fields) || []).filter(function (f) { return f && f.id === id; })[0] || null; }
  function lowWindow(f) {
    var ws = (f && f.windows) || [];
    var w = ws.filter(function (x) { return x && (x.id === 'low' || /低/.test(x.label || '')); })[0];
    if (w) return {id: w.id, label: w.label + ' ' + ui().range(w.range[0], w.range[1], f.units) + (w.provisional ? '（暂定）' : ''), spec: w.id};
    return {id: 'range-0-1', label: '低值范围 ' + ui().range(0, 1, f && f.units) + '（暂定）', spec: {range: [0, 1]}};
  }
  function lowFindings(m, fieldId) {
    var items = (m && m.analysis && m.analysis.findings && m.analysis.findings.items) || [];
    var kinds = fieldId === 'tawss' ? ['low_tawss_cluster', 'stagnation_cluster'] : ['low_wss_cluster'];
    return items.filter(function (it) { return it && kinds.indexOf(it.kind) >= 0; });
  }
  var QUESTIONS = [
    {id: 'low', label: '低值区域在哪里', available: function (m) { return hasField(m, 'tawss') || hasField(m, 'wss'); }},
    {id: 'pair', label: '同一位置的 TAWSS 与 OSI', available: function (m) { return hasField(m, 'tawss') && hasField(m, 'osi'); }},
    {id: 'compare', label: '比较两次结果', available: function (m, env) { return !(env && env.offline) && Boolean(ns.compare); }}
  ];
  function list(m, env) { return QUESTIONS.filter(function (q) { return q.available(m, env); }).map(function (q) { return {id: q.id, label: q.label}; }); }

  // plan(): what the question will apply — pure, so the description and the action cannot drift apart.
  function plan(id, m) {
    if (id === 'low') {
      var fid = hasField(m, 'tawss') ? 'tawss' : 'wss';
      var f = field(m, fid);
      var w = lowWindow(f);
      var items = lowFindings(m, fid);
      return {id: id, label: '低值区域在哪里', field: fid, window: w.spec, split: null, findings: items,
        description: (f && (f.short_label || f.label) || fid.toUpperCase()) + ' · ' + w.label + ' · ' + (items.length ? '低值相关发现 ' + items.length + ' 处已标出' : '没有低值相关发现')};
    }
    if (id === 'pair') {
      return {id: id, label: '同一位置的 TAWSS 与 OSI', field: 'tawss', window: 'adaptive', split: {field: 'osi', window: 'adaptive', link: true}, findings: [],
        description: '左 TAWSS，右 OSI · 两图视角同步，各用自己的色标（数值不能跨图比较）'};
    }
    if (id === 'compare') return {id: id, label: '比较两次结果', compare: true, description: '先选第二个结果，再核对比较条件'};
    return null;
  }
  // shell: {snapshot(), restore(s), setField(id, window), setSplit(spec|null), showFindings(items), openCompare()}
  function apply(id, shell, m) {
    var p = plan(id, m);
    if (!p) return null;
    if (p.compare) { shell.openCompare(); return null; }
    var before = shell.snapshot();
    shell.setSplit(p.split);
    shell.setField(p.field, p.window);
    shell.showFindings(p.findings);
    return {id: id, plan: p, before: before};
  }
  function undo(active, shell) { if (active && active.before) shell.restore(active.before); }

  function bar(el, active, handlers) {
    var h = ui().h;
    if (!active) { el.hidden = true; el.replaceChildren(); return; }
    el.hidden = false;
    ui().fill(el, h('span', {'class': 'q-label', text: '按问题查看'}), h('strong', {text: active.plan.label}), h('span', {'class': 'q-desc', text: active.plan.description}),
      h('span', {'class': 'sec-fill'}),
      ui().button('撤销', handlers.onUndo, {cls: 'btn-sm', title: '恢复到按问题查看之前的视图'}),
      ui().button('保留视图，退出', handlers.onKeep, {kind: 'link', cls: 'btn-sm', title: '回到自由浏览，保留当前显示'}));
  }

  return {list: list, plan: plan, apply: apply, undo: undo, bar: bar, lowWindow: lowWindow, lowFindings: lowFindings};
});
