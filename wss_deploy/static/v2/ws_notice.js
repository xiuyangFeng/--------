/* WSS workspace v2 — phase 3 lane 3 (info & safety): the warning bar on top of a result (W65 / V45) and its details.
 *
 * Trigger and wording are the classic pages' (report.py bannerInfo, volume_viewer.js warningItems / warningText):
 *   · geometry outside the release's declared reference range (analysis.reference_assessment: a check with status
 *     'review', or the whole assessment in 'review'),
 *   · the population reference needs review (reference_assessment.population.status === 'review'),
 *   · the ensemble quality is not 'good' (analysis.quality.level; label and reasons only — the per-point spread lives in
 *     quality_audit.json and is never read here, user ruling: no dispersion numbers).
 * Numbers are formatted with WssReportCommon.formatValue, the function both classic pages use.  The bar sits in the
 * shell's notice host (shellApi.noticeHost()), shows in the offline report too (the manifest is embedded), 「详情」
 * opens the reference / quality dialog, 「×」 closes it for this opening of the result (classic: 仅本次打开).
 * Also asks ws_lens for the glossary (term tips) once the workspace starts.
 * Final round lane B (C+1): badge(manifest, {side}) — the same warning as a 「需复核」 badge on a viewport of a comparison
 * (the shell's status line puts it on each side); ws_compare lists each side's sentence in the 比较 tab. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.notice = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };
  function obj(v) { return v && typeof v === 'object' && !Array.isArray(v) ? v : {}; }

  // ------------------------------------------------------------------ pure: the classic banner model
  // The wall report's field names (report.py REF_FIELDS); the volume page's table is a subset of it.
  var REF_FIELDS = {length_mm: '长度', radius_min_mm: '最小半径', radius_median_mm: '中位半径', radius_max_mm: '最大半径', max_diameter_mm: '最大内切直径',
    tortuosity: '迂曲度', spacing_mm: '点间距', surface_variation_median: '表面变化度', variation: '表面变化度'};
  function referenceLabel(path) {
    var p = String(path || '');
    if (p.indexOf('cloud.') === 0) return '点云' + (REF_FIELDS[p.slice(6)] || p.slice(6));
    if (p.indexOf('geometry.') === 0) {
      var rest = p.slice(9), at = rest.lastIndexOf('.');
      if (at > 0) { var f = rest.slice(at + 1); return rest.slice(0, at) + (REF_FIELDS[f] || f); }
    }
    return p;
  }
  function fq(v) {
    var c = root.WssReportCommon;
    if (c && typeof c.formatValue === 'function') { try { return c.formatValue(v); } catch (_) {} }
    return v === null || v === undefined || !isFinite(Number(v)) ? '—' : String(Number(v));
  }
  function unitsOf(c) { return c && c.units && c.units !== '1' ? ' ' + c.units : ''; }
  function isSet(v) { return v !== null && v !== undefined; }
  function reviewChecks(ra) { return (Array.isArray(ra.checks) ? ra.checks : []).filter(function (c) { return c && c.status === 'review'; }); }
  // Every geometry check outside the declared range, as the volume page lists them: 「<名称> <值>（参考 <下>–<上>）」.
  function referenceItems(ra) {
    return reviewChecks(obj(ra)).map(function (c) {
      var u = unitsOf(c), what = referenceLabel(c.path) + ' ' + fq(c.value) + u, range = fq(c.min) + '–' + fq(c.max) + u;
      return {kind: 'reference', path: c.path, text: what + '（参考 ' + range + '）', what: what, range: range};
    });
  }
  // → null, or {text (the classic sentence), items [{kind, text}], target: 'reference' | 'quality'}
  function model(manifest) {
    var a = obj(manifest && manifest.analysis), ref = obj(a.reference_assessment), pop = obj(ref.population), q = obj(a.quality);
    var out = reviewChecks(ref);
    var msgs = [], items = referenceItems(ref), target = '';
    if (out.length || ref.status === 'review') {
      var c = out[0], u = unitsOf(c);
      msgs.push('输入几何有 ' + (out.length || '部分') + ' 项超出发布包参考范围' + (c ? '（' + referenceLabel(c.path) + (isSet(c.value) ? ' ' + fq(c.value) + u : '') +
        (isSet(c.min) && isSet(c.max) ? '，参考 ' + fq(c.min) + '–' + fq(c.max) + u : '') + (out.length > 1 ? ' 等' : '') + '）' : ''));
      if (!out.length) items.push({kind: 'reference', text: '部分几何测量超出参考范围'});
      target = 'reference';
    }
    if (pop.status === 'review') { msgs.push('人群参照需要复核'); items.push({kind: 'population', text: '人群参照需要复核'}); target = target || 'reference'; }
    if (q.level && q.level !== 'good') { var qt = '模型集成质量：' + (q.label || q.level); msgs.push(qt); items.push({kind: 'quality', text: qt}); target = target || 'quality'; }
    if (!msgs.length) return null;
    return {text: '注意：' + msgs.join('；') + '。预测可信度可能下降，请结合详情复核。', items: items, target: target};
  }

  // ------------------------------------------------------------------ pure: the details (classic ref-card, quality-card)
  var GEO_STATUS = {pass: '输入几何在本发布包声明的参考范围内。', review: '部分几何测量超出本发布包声明的参考范围，请复核。',
    unknown: '部分几何测量缺失或本发布包未声明参考范围，无法完成范围检查。'};
  var QUALITY_GOOD = '多模型一致（一致不代表准确）';
  function modelCount(manifest) {
    var pv = obj(manifest && manifest.provenance), n = Number(pv.n_models);
    return Number.isFinite(n) && n > 0 ? Math.round(n) : null;
  }
  // classic: 「五模型」 only when there are five (an unknown count keeps the historical wording)
  function ensembleWord(n) { return n === 5 || n === null || n === undefined ? '五模型' : n + ' 个模型'; }
  function qualityModel(manifest) {
    var q = obj(manifest && manifest.analysis && manifest.analysis.quality);
    if (!q.level) return null;
    var good = q.level === 'good' || q.label === '模型集成稳定';
    return {level: q.level, good: good, text: good ? QUALITY_GOOD : (q.label || q.level), title: ensembleWord(modelCount(manifest)) + '一致性',
      reasons: (Array.isArray(q.reasons) ? q.reasons : []).filter(function (t) { return typeof t === 'string' && t.trim(); })};
  }
  // The classic wall report's ref-card sentences (geometry / population) plus the volume page's status line.
  function referenceModel(manifest) {
    var ra = obj(manifest && manifest.analysis && manifest.analysis.reference_assessment), pop = obj(ra.population);
    if (!ra.status && !pop.status) return null;
    var pct = pop.percentile === null || pop.percentile === undefined ? NaN : Number(pop.percentile);
    return {status: ra.status || 'unknown', popStatus: pop.status || 'unknown',
      geometry: ra.status === 'pass' ? '在已声明几何参考范围内' : ra.status === 'review' ? '超出已声明几何参考范围，请复核' : '未配置几何参考范围',
      population: pop.status === 'pass' ? '已计算同协议交叉验证折外经验分位 ' + (Number.isFinite(pct) ? pct.toFixed(1) : '—') + '%' : pop.status === 'review' ? '人群参照需要复核' : '未配置可核验的人群参照',
      percentile: pop.status === 'pass' && Number.isFinite(pct) ? pct : null, referenceCount: pop.reference_count || null,
      statusLine: (GEO_STATUS[ra.status] || '参考范围检查状态未知。') + (ra.note ? ' ' + ra.note : ''),
      items: referenceItems(ra), reviewCount: reviewChecks(ra).length, reasons: (Array.isArray(ra.reasons) ? ra.reasons : []).filter(Boolean),
      popReasons: (Array.isArray(pop.reasons) ? pop.reasons : []).filter(Boolean),
      note: ra.note || pop.note || '参考范围用于复核，不代表校准概率或临床风险。', popNote: pop.note || '', hasNote: Boolean(ra.note || pop.note)};
  }

  // ------------------------------------------------------------------ rendering
  function term(key, fallback) { return ns.lens && ns.lens.termText ? ns.lens.termText(key, fallback) : fallback; }
  function list(texts) { var h = ui().h; return h('ul', {'class': 'notice-list'}, texts.map(function (t) { return h('li', {text: t}); })); }
  function detailsBody(manifest) {
    var h = ui().h, parts = [], rm = referenceModel(manifest), qm = qualityModel(manifest);
    if (rm) {
      parts.push(h('h4', {'class': 'notice-h'}, h('span', {text: '几何参考范围'}), ui().dot(rm.status === 'pass' ? 'ok' : rm.status === 'review' ? 'warn' : 'idle',
        rm.status === 'pass' ? '通过' : rm.status === 'review' ? '需复核' : '未检查')));
      parts.push(h('p', {text: rm.statusLine}));
      if (rm.items.length) parts.push(list(rm.items.map(function (x) { return x.text; })));
      else if (rm.status === 'review' && rm.reasons.length) parts.push(list(rm.reasons));
      parts.push(h('h4', {'class': 'notice-h'}, h('span', {text: '人群参照'}), ui().dot(rm.popStatus === 'pass' ? 'ok' : rm.popStatus === 'review' ? 'warn' : 'idle',
        rm.popStatus === 'pass' ? '已计算' : rm.popStatus === 'review' ? '需复核' : '未配置'), ui().infoTip(term('population_percentile', '本例 p99 在参照队列折外预测分布中的位置；参照来自模型预测，不是 CFD。'))));
      parts.push(h('p', {text: rm.population + (rm.referenceCount && rm.percentile !== null ? '（' + rm.referenceCount + ' 例）' : '')}));
      if (rm.popStatus === 'review' && rm.popReasons.length) parts.push(list(rm.popReasons));
      if (rm.popNote) parts.push(ui().note(rm.popNote));
    }
    if (qm) {
      parts.push(h('h4', {'class': 'notice-h'}, h('span', {text: qm.title}), ui().dot(qm.good ? 'ok' : 'warn', qm.text), ui().infoTip(term('quality_grade', '多个模型之间的一致程度；一致不代表准确。'))));
      if (qm.reasons.length) parts.push(list(qm.reasons));
      parts.push(ui().note('逐点离散度只保存在质量审计文件里，这里不显示，避免被误读为预测概率。'));
    }
    if (rm && !rm.hasNote) parts.push(ui().note(rm.note));   // the status line / population note already carry it otherwise
    if (!parts.length) parts.push(ui().note('这份结果没有参考范围检查和质量分级。'));
    return parts;
  }
  function openDetails(manifest) {
    if (!ui() || !ui().dialog) return null;
    return ui().dialog.open({title: '参考范围与质量', cls: 'dlg-notice', body: detailsBody(manifest),
      actions: [ui().button('关闭', function () { ui().dialog.close('done'); })]});
  }
  // The bar itself; opts = {onDetails(), onClose()}.  Returns null when nothing needs saying.
  function bar(manifest, opts) {
    var md = model(manifest);
    if (!md) return null;
    opts = opts || {};
    var h = ui().h;
    var text = h('span', {'class': 'ws-notice-text', text: md.text, title: md.items.map(function (x) { return x.text; }).join('\n')});
    var more = ui().button('详情', function () { if (opts.onDetails) opts.onDetails(); else openDetails(manifest); }, {kind: 'link', cls: 'btn-sm ws-notice-more'});
    var close = ui().iconButton('close', '关闭提示（仅本次打开）', function () { if (opts.onClose) opts.onClose(); }, {cls: 'ws-notice-close'});
    return h('div', {'class': 'ws-notice', role: 'alert', dataset: {target: md.target}},
      h('span', {'class': 'ws-notice-icon', 'aria-hidden': 'true'}, ui().icon ? ui().icon('warning', {size: 15}) : '!'), text, more, close);
  }
  // P4 (C+1): the same warning as a small badge on a viewport of a comparison (left or right), so each side's warning
  // stays in sight whatever tab is open; 「需复核」 with the classic sentence as its tip, a click opens that side's details.
  // Returns null when that result has nothing to say.  opts = {side: '左' | '右'}.
  function badge(manifest, opts) {
    var md = model(manifest);
    if (!md) return null;
    opts = opts || {};
    var h = ui().h, who = opts.side ? opts.side + '侧结果' : '这份结果';
    var b = h('button', {type: 'button', 'class': 'vp-notice', title: who + '：' + md.text, 'aria-label': who + '需复核：' + md.text, dataset: {target: md.target}},
      ui().icon ? ui().icon('warning', {size: 13}) : null, h('span', {text: '需复核'}));
    b.addEventListener('click', function (e) { if (e && e.stopPropagation) e.stopPropagation(); openDetails(manifest); });
    return b;
  }
  // Fill the shell's host for the current result (or empty it).
  function show(api) {
    var host = api && api.noticeHost ? api.noticeHost() : null;
    if (!host) return null;
    var cur = api.cur();
    if (!cur || !cur.manifest || cur.noticeClosed) { host.replaceChildren(); return null; }
    var el = null;
    try {
      el = bar(cur.manifest, {onDetails: function () { openDetails(cur.manifest); },
        onClose: function () { cur.noticeClosed = true; host.replaceChildren(); }});
    } catch (e) { if (root.console && root.console.error) root.console.error('notice: ' + (e && e.stack || e)); el = null; }
    if (el) host.replaceChildren(el); else host.replaceChildren();
    return el;
  }

  // ------------------------------------------------------------------ registration
  var glossaryAsked = false;
  function glossary(api) {
    if (glossaryAsked || !ns.lens || !ns.lens.loadGlossary) return;
    glossaryAsked = true;
    Promise.resolve(ns.lens.loadGlossary({offline: api.offline()})).then(function (terms) {
      // tips already on screen were written with the built-in text: once the glossary is here, redraw them
      if (terms && Object.keys(terms).length && api.cur() && api.cur().manifest) { try { api.renderInspector(); } catch (_) {} }
    }, function () {});
  }
  var EXT = {
    id: 'notice',
    onStart: function (api) { glossary(api); },
    onResult: function (api) { glossary(api); show(api); },
    onClose: function (api) { var host = api.noticeHost ? api.noticeHost() : null; if (host) host.replaceChildren(); }
  };
  (ns.ext = ns.ext || []).push(EXT);

  return {model: model, referenceLabel: referenceLabel, referenceItems: referenceItems, referenceModel: referenceModel, qualityModel: qualityModel,
    modelCount: modelCount, ensembleWord: ensembleWord, detailsBody: detailsBody, openDetails: openDetails, bar: bar, show: show, ext: EXT, QUALITY_GOOD: QUALITY_GOOD,
    badge: badge};
});
