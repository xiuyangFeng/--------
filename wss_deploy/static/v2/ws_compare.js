/* WSS workspace v2 — compare D11 + two viewports D3 (contract §6.3 ws_compare.js).
 * Before two results are read side by side, the conditions are listed one by one (field, units, time, model,
 * geometry, statistics, orientation) with 一致 / 不一致 / 未知.  Same-scale mode only for compatible fields;
 * camera sync on by default only for the same input geometry.  No point-by-point difference.
 * Second phase lane E: 以左为准 / 以右为准 (classic push-left / push-right, through ws_display.pushSide) and, in the
 * same-scale mode, one set of colour bands and thresholds for both sides (ws_display follows the mode).
 * Phase 3 lane 5: both sides' identity (patient, scan, date, review; classic compare.js renderIdentity, #112), the
 * scalar table of POST /api/compare — left, right, right − left of the declared-compatible statistics, the reasons when
 * they are not (classic renderComparison, C5) — and a searchable picker without the 20-result cap (#98). */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.compare = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };

  function fieldOf(m, id) { return ((m && m.fields) || []).filter(function (f) { return f && f.id === id; })[0] || null; }
  function canon(v) {
    if (v === null || v === undefined) return null;
    if (Array.isArray(v)) return '[' + v.map(canon).join(',') + ']';
    if (typeof v === 'object') return '{' + Object.keys(v).sort().map(function (k) { return k + ':' + canon(v[k]); }).join(',') + '}';
    return String(v);
  }
  function timeText(m, f) {
    if (!f) return null;
    if (f.temporal === 'cycle_summary') return '一个心动周期的汇总';
    var t = null;
    var axis = m && m.time && m.time.axis;
    if (Array.isArray(axis) && axis.length) { var a = axis[f.time_index || 0] || axis[0]; t = a && a.time_s; }
    if ((t === null || t === undefined) && m && m.result && m.result.model_frame) t = m.result.model_frame.time_s;
    return '收缩期峰值帧' + (t !== null && t !== undefined ? '（' + ui().trim(t) + ' s）' : '');
  }
  function verdict(a, b) {
    if (a === null || a === undefined || b === null || b === undefined || a === '' || b === '') return '未知';
    return a === b ? '一致' : '不一致';
  }
  function sameGeometry(mA, mB) {
    var a = mA && mA.job && mA.job.input_sha256, b = mB && mB.job && mB.job.input_sha256;
    if (!a || !b) return null;
    return a === b;
  }
  // Pure: the condition table.
  function conditions(mA, mB, fidA, fidB) {
    var fA = fieldOf(mA, fidA), fB = fieldOf(mB, fidB);
    var rows = [];
    var label = function (f, id) { return f ? (f.short_label || f.label || f.id) : (id ? id + '（不存在）' : null); };
    rows.push({key: 'field', label: '字段', left: label(fA, fidA), right: label(fB, fidB), verdict: fA && fB ? (fA.id === fB.id ? '一致' : '不一致') : '未知'});
    rows.push({key: 'units', label: '单位', left: fA ? (ui().unitText(fA.units) || '无量纲') : null, right: fB ? (ui().unitText(fB.units) || '无量纲') : null, verdict: fA && fB ? verdict(fA.units, fB.units) : '未知'});
    var tA = timeText(mA, fA), tB = timeText(mB, fB);
    rows.push({key: 'time', label: '时相 / 周期', left: tA, right: tB, verdict: verdict(tA, tB)});
    var rA = mA && mA.result || {}, rB = mB && mB.result || {};
    var nameA = rA.display_name ? rA.display_name + (mA.model_card && mA.model_card.version_date ? ' · ' + mA.model_card.version_date : '') : null;
    var nameB = rB.display_name ? rB.display_name + (mB.model_card && mB.model_card.version_date ? ' · ' + mB.model_card.version_date : '') : null;
    rows.push({key: 'model', label: '模型与协议', left: nameA, right: nameB, verdict: verdict(rA.release_id || null, rB.release_id || null)});
    var g = sameGeometry(mA, mB);
    rows.push({key: 'geometry', label: '几何', left: g === null ? null : (g ? '同一输入' : '本侧输入'), right: g === null ? null : (g ? '同一输入' : '另一份输入'), verdict: g === null ? '未知' : (g ? '一致' : '不一致')});
    // per-geometry values are not part of the definition (comparison.py _VOLATILE_PROTOCOL_KEYS): same verdict as /api/compare
    var sA = protocolOf(mA), sB = protocolOf(mB);
    var hasA = sA && Object.keys(sA).length, hasB = sB && Object.keys(sB).length;
    var thA = fA && fA.display && fA.display.thresholds, thB = fB && fB.display && fB.display.thresholds;
    rows.push({key: 'statistics', label: '统计定义', left: hasA ? '预测点等权' + (thA ? '，阈值 ' + thA.map(ui().trim).join(' / ') : '') : null,
      right: hasB ? '预测点等权' + (thB ? '，阈值 ' + thB.map(ui().trim).join(' / ') : '') : null,
      verdict: hasA && hasB ? (canon(sA) === canon(sB) && canon(thA) === canon(thB) ? '一致' : '不一致') : '未知'});
    var oA = mA && mA.frame && mA.frame.orientation && mA.frame.orientation.left_right, oB = mB && mB.frame && mB.frame.orientation && mB.frame.orientation.left_right;
    var ot = function (o) { return o === 'confirmed' ? '左右已确认' : o === 'inferred' ? '左右为推断' : null; };
    rows.push({key: 'orientation', label: '方向', left: ot(oA), right: ot(oB), verdict: oA === 'confirmed' && oB === 'confirmed' ? '一致' : '未知'});
    return rows;
  }
  var VOLATILE_PROTOCOL_KEYS = ['top5_connectivity_radius_mm', 'connectivity_radius_mm'];
  function protocolOf(m) {
    var p = m && m.mapping && m.mapping.statistics_protocol;
    if (!p || typeof p !== 'object') return p;
    var out = {};
    Object.keys(p).forEach(function (k) { if (VOLATILE_PROTOCOL_KEYS.indexOf(k) < 0) out[k] = p[k]; });
    return out;
  }
  function sameScaleAllowed(rows) {
    var need = ['field', 'units', 'time'];
    var bad = rows.filter(function (r) { return need.indexOf(r.key) >= 0 && r.verdict !== '一致'; });
    return {ok: bad.length === 0, reasons: bad.map(function (r) { return r.label + r.verdict; })};
  }
  // Common range for same-scale mode, from the declared display range / p99 of both fields.
  function commonRange(mA, mB, fidA, fidB) {
    var fA = fieldOf(mA, fidA), fB = fieldOf(mB, fidB);
    if (!fA || !fB) return null;
    var span = function (f) {
      var d = f.display || {};
      if (Array.isArray(d.range) && d.range.length === 2) return d.range;
      var hi = d.p99 !== undefined ? d.p99 : (f.statistics && f.statistics.p99);
      return hi === undefined || hi === null ? null : [0, hi];
    };
    var a = span(fA), b = span(fB);
    if (!a || !b) return null;
    return [Math.min(a[0], b[0]), Math.max(a[1], b[1])];
  }

  // ------------------------------------------------------------------ identity of a side (#112)
  // Words only from the manifest's job block (the same record the classic page read through /api/jobs/<id>).  With the
  // name hidden (presentation mode) the patient and case codes are left out as well.
  function identity(m, hidden) {
    var j = (m && m.job) || {}, parts = [];
    var shown = function (v) { return typeof v === 'string' && v.trim() && v !== j.display_name; };
    if (!hidden && shown(j.patient_id)) parts.push('患者 ' + j.patient_id);
    if (!hidden && shown(j.case_id) && j.case_id !== j.patient_id) parts.push('病例 ' + j.case_id);
    if (typeof j.scan_label === 'string' && j.scan_label.trim()) parts.push(j.scan_label.trim());
    if (j.scan_date) parts.push(ui().time(j.scan_date, false));
    return {text: parts.join(' · '), review: j.review || null};
  }

  // ------------------------------------------------------------------ scalar table (C5): POST /api/compare
  var ROW_ZH = {'peak.p99_pa': '峰值 p99', 'peak.max_pa': '峰值最大', 'wss_field.mean_pa': '壁面均值',
    'cycle.tawss.mean_pa': 'TAWSS 均值', 'cycle.tawss.p99_pa': 'TAWSS p99', 'cycle.osi.mean': 'OSI 均值', 'cycle.rrt.mean_per_pa': 'RRT 均值',
    'cycle.ecap.mean_per_pa': 'ECAP 均值', 'cycle.stagnation.area_frac': '滞留区面积占比'};
  var REASON_ZH = [[/^missing input_sha256/, '缺少输入指纹，不能判断是否同一输入'], [/^same input requires mapping/, '同一输入但缺少映射声明'],
    [/^mapping differs for the same input/, '同一输入但映射参数不同'], [/^WSS field declarations differ/, 'WSS 字段声明不同'], [/^time_axis differs/, '时相不同'],
    [/^statistics_protocol differs/, '统计定义不同'], [/^WSS thresholds_pa differ/, 'WSS 阈值不同'], [/^run_parameters smooth_mm\/spacing_mm differ/, '预处理参数（平滑、间距）不同'],
    [/^missing time_axis/, '缺少时相声明'], [/^missing fields declaration for WSS|^missing WSS field descriptor/, '没有 WSS 字段声明'],
    [/^WSS field must be a scalar wall field/, '不是壁面 WSS 标量'], [/^missing statistics_protocol/, '缺少统计定义'], [/^missing WSS thresholds_pa|^WSS thresholds_pa must/, 'WSS 阈值缺失或无效'],
    [/^missing run_parameters|^missing or invalid run_parameters/, '缺少预处理参数']];
  function reasonText(r) {
    var s = String(r || ''), side = '';
    var m = /^(left|right): (.*)$/.exec(s);
    if (m) { side = m[1] === 'left' ? '左侧' : '右侧'; s = m[2]; }
    for (var i = 0; i < REASON_ZH.length; i++) if (REASON_ZH[i][0].test(s)) { s = REASON_ZH[i][1]; break; }
    return (side ? side + '：' : '') + s;
  }
  function rowLabel(r) {
    if (ROW_ZH[r.id]) return ROW_ZH[r.id];
    if (r.scope === 'per_branch' && r.branch) return r.branch + ' p99';
    return r.label || r.id || '';
  }
  function isFrac(r) { return r.id === 'cycle.stagnation.area_frac'; }
  function valueText(r, v) {
    if (v === null || v === undefined || !isFinite(Number(v))) return '—';
    if (isFrac(r)) return ui().pct(v);
    return ui().num(v, r.units === '1' ? null : r.units);
  }
  function deltaText(r, v) {
    if (v === null || v === undefined || !isFinite(Number(v))) return '—';
    var n = Number(v), sign = n > 0 ? '+' : n < 0 ? '−' : '';
    if (isFrac(r)) return sign + ui().pct(Math.abs(n)).replace('%', ' 个百分点');
    return sign + ui().num(Math.abs(n), r.units === '1' ? null : r.units);
  }
  // Pure: the table model (rows with a value on at least one side, in the service's order) and the words above it.
  function scalarModel(cmp) {
    cmp = cmp || {};
    var rows = (Array.isArray(cmp.rows) ? cmp.rows : []).filter(function (r) { return r && (isFinite(Number(r.left)) && r.left !== null || isFinite(Number(r.right)) && r.right !== null); })
      .map(function (r) { return {id: r.id, label: rowLabel(r), left: valueText(r, r.left), right: valueText(r, r.right), delta: deltaText(r, r.delta), scope: r.scope || '', raw: r}; });
    var reasons = (Array.isArray(cmp.reasons) ? cmp.reasons : []).map(reasonText).filter(function (t, i, a) { return t && a.indexOf(t) === i; });
    var kind = cmp.kind === 'same_input' ? '同一输入' : cmp.kind === 'cross_case' ? '跨病例' : '';
    var cycleNote = cmp.cycle && cmp.cycle.note ? String(cmp.cycle.note) : '';
    return {rows: rows, reasons: reasons, compatible: cmp.compatible === true, kind: kind, cycleNote: cycleNote};
  }
  var scalarCache = {};
  function fetchScalars(leftId, rightId) {
    var key = leftId + '|' + rightId;
    if (scalarCache[key]) return scalarCache[key];
    var A = ns.api;
    if (!A || typeof A.request !== 'function') return Promise.reject(new Error('离线报告不能比较'));
    var p = A.request('/api/compare', {method: 'POST', body: {left_job_id: leftId, right_job_id: rightId}}).then(function (r) { return (r && r.comparison) || r; });
    scalarCache[key] = p;
    p.catch(function () { if (scalarCache[key] === p) delete scalarCache[key]; });
    return p;
  }
  function scalarSection(ctx) {
    var h = ui().h, body = h('div', {'class': 'cmp-scalars'}, h('p', {'class': 'muted cmp-wait', text: '正在读取…'}));
    var ids = [ctx.left.jobId || (ctx.left.manifest && ctx.left.manifest.job && ctx.left.manifest.job.id), ctx.right.jobId || (ctx.right.manifest && ctx.right.manifest.job && ctx.right.manifest.job.id)];
    var sec = ui().section('标量对照', {tag: ui().infoTip('服务端 /api/compare：只列两侧都声明一致口径的标量统计（峰值、分支 p99、周期量），差值 = 右 − 左；口径不一致时差值留空并写明原因。不做逐点差值。')}, body);
    if (!ids[0] || !ids[1]) { ui().fill(body, ui().note('缺少任务编号，不能对照。')); return sec; }
    fetchScalars(ids[0], ids[1]).then(function (cmp) { fillScalars(body, scalarModel(cmp)); },
      function (e) { ui().fill(body, ui().note('标量对照不可用：' + (e && e.message || e), 'warn')); });
    return sec;
  }
  function fillScalars(body, model) {
    var h = ui().h, parts = [];
    if (model.reasons.length) parts.push(ui().note('口径不一致，差值留空：' + model.reasons.join('；') + '。', 'warn'));
    else if (model.kind) parts.push(h('p', {'class': 'muted cmp-kind', text: model.kind + ' · 口径一致'}));
    if (model.cycleNote && !model.reasons.length) parts.push(ui().note(model.cycleNote + '。', 'warn'));
    if (!model.rows.length) parts.push(ui().note('这两份结果没有可对照的标量。'));
    else parts.push(ui().table([{key: 'label', label: '指标'}, {key: 'left', label: '左', num: true}, {key: 'right', label: '右', num: true}, {key: 'delta', label: '右 − 左', num: true}],
      model.rows, {cls: 'tbl-cmp tbl-scalar'}));
    ui().fill(body, parts);
  }

  // ------------------------------------------------------------------ inspector panel
  function render(el, ctx) {
    var h = ui().h;
    var rows = conditions(ctx.left.manifest, ctx.right.manifest, ctx.left.field, ctx.right.field);
    var allow = sameScaleAllowed(rows);
    var geo = sameGeometry(ctx.left.manifest, ctx.right.manifest);
    // the name is hidden on the left (presentation mode): leave the codes out on both sides
    var hidden = Boolean(ctx.hideName) || Boolean(ctx.left.manifest && ctx.left.manifest.job && ctx.left.manifest.job.display_name && ctx.left.name !== ctx.left.manifest.job.display_name);
    var who = function (side, x) {
      var id = identity(x.manifest, hidden);
      return h('div', {'class': 'cmp-who'}, h('span', {'class': 'cmp-side', text: side}), h('span', {'class': 'cmp-name', text: hidden ? '病例' : x.name}), h('span', {'class': 'muted', text: x.result}),
        h('div', {'class': 'cmp-id'}, ui().reviewDot(id.review), id.text ? h('span', {'class': 'cmp-id-text', text: id.text}) : null));
    };
    var table = ui().table([
      {key: 'label', label: '条件'}, {key: 'left', label: '左'}, {key: 'right', label: '右'},
      {key: 'verdict', label: '判定', render: function (r) { return ui().dot(r.verdict === '一致' ? 'ok' : r.verdict === '不一致' ? 'warn' : 'idle', r.verdict); }}], rows, {cls: 'tbl-cmp'});
    var modeName = 'cmp-mode-' + Math.floor(Math.random() * 1e6);
    var radio = function (value, label, disabled, title) {
      var input = h('input', {type: 'radio', name: modeName, value: value, checked: ctx.mode === value, disabled: Boolean(disabled)});
      input.addEventListener('change', function () {
        if (!input.checked || !ctx.onMode) return;
        if (value === 'each' && ctx.mode === 'same' && ns.display && ns.display.keepShared) ns.display.keepShared();   // the right side keeps what it showed
        ctx.onMode(value);
      });
      return h('label', {'class': 'radio' + (disabled ? ' disabled' : ''), title: title || null}, input, h('span', {text: label}));
    };
    var modes = h('div', {'class': 'cmp-modes', role: 'radiogroup', 'aria-label': '比较方式'},
      radio('same', '同尺度看幅值', !allow.ok, allow.ok ? '两侧用同一个色标范围、分段和阈值' : '条件不一致，不能同尺度：' + allow.reasons.join('、')),
      radio('each', '各自增强细节'));
    var D = ns.display && typeof ns.display.pushSide === 'function' ? ns.display : null;
    var push = D ? h('div', {'class': 'cmp-push'},
      ui().button('以左为准', function () { D.pushSide('a'); }, {cls: 'btn-sm', title: '把左侧的字段、色标窗、分段和阈值用到右侧'}),
      ui().button('以右为准', function () { D.pushSide('b'); }, {cls: 'btn-sm', title: '把右侧的字段、色标窗、分段和阈值用到左侧'}),
      ui().infoTip ? ui().infoTip('只照搬显示方式，不改数值。图层、单位和外壁透明度两侧本来就共用。') : null) : null;
    var sync = h('input', {type: 'checkbox', checked: Boolean(ctx.sync)});
    sync.addEventListener('change', function () { if (ctx.onSync) ctx.onSync(sync.checked); });
    var syncRow = h('label', {'class': 'check'}, sync, h('span', {text: '同步视角'}));
    var parts = [
      h('div', {'class': 'cmp-pair'}, who('左', ctx.left), who('右', ctx.right)),
      ui().section('比较条件', {}, table),
      ctx.scalars === false ? null : scalarSection(ctx),
      ui().section('怎么看', {}, modes, ctx.mode === 'each' ? ui().note('两侧色标范围不同，同一种颜色不代表同一个数。', 'warn') : ui().note('两侧用同一个色标范围：' + (ctx.rangeText || '—') + (D ? '；分段和阈值也一起改' : '') + '。'), push),
      ui().section('视角', {}, syncRow, geo ? ui().note('同一输入几何：视角同步后，同一屏幕位置是同一处血管。') : ui().note('不同输入：只同步视角，不代表位置对应。', 'warn')),
      ui().note('不做逐点差值：两份结果的预测点不一一对应。'),
      h('div', {'class': 'sec-actions'}, ctx.onSwap ? ui().button('左右互换', ctx.onSwap, {cls: 'btn-sm'}) : null, ctx.onClose ? ui().button('退出比较', ctx.onClose, {cls: 'btn-sm'}) : null)
    ];
    ui().fill(el, parts);
    return {rows: rows, allow: allow};
  }

  // ------------------------------------------------------------------ choose the second result
  // #98: every finished result (no cap), narrowed by the search words (all must match: name, patient, case, scan label,
  // scan date, tags, result name, task id).
  function matches(job, query, cards) {
    var words = String(query || '').trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return true;
    var hay = [ui().displayName(job), job.patient_id, job.case_id, job.scan_label, job.scan_date, job.id, ui().resultName(job, cards)]
      .concat(Array.isArray(job.tags) ? job.tags : []).filter(Boolean).join(' ').toLowerCase();
    return words.every(function (w) { return hay.indexOf(w) >= 0; });
  }
  function candidates(jobs, current, query, cards) {
    current = current || {};
    var done = (jobs || []).filter(function (j) { return j && j.status === 'done' && j.id !== current.id && matches(j, query, cards); });
    var sameInput = [], samePatient = [], others = [];
    done.forEach(function (j) {
      if (current.input_sha256 && j.input_sha256 === current.input_sha256) sameInput.push(j);
      else if (current.patient_id && j.patient_id === current.patient_id) samePatient.push(j);
      else others.push(j);
    });
    return {sameInput: sameInput, samePatient: samePatient, others: others};
  }
  function pickDialog(ctx) {
    var h = ui().h;
    var all = candidates(ctx.jobs, ctx.current || {}, '', ctx.cards);
    var total = all.sameInput.length + all.samePatient.length + all.others.length;
    var results = h('div', {'class': 'pick-groups'});
    var count = h('span', {'class': 'muted pick-count'});
    var search = h('input', {type: 'search', 'class': 'pick-search', placeholder: '搜索病例、患者、扫描、标签', 'aria-label': '搜索要比较的结果', autocomplete: 'off'});
    function draw() {
      var c = candidates(ctx.jobs, ctx.current || {}, search.value, ctx.cards);
      var groups = [['同一输入的其他结果', c.sameInput], ['同一患者的其他扫描', c.samePatient], ['其他已完成结果', c.others]];
      var body = [], n = 0;
      groups.forEach(function (g) {
        if (!g[1].length) return;
        n += g[1].length;
        body.push(h('h4', {text: g[0] + ' · ' + g[1].length}));
        var list = h('div', {'class': 'pick-list'});
        g[1].forEach(function (j) {
          var b = h('button', {type: 'button', 'class': 'pick', dataset: {jobId: j.id}},
            h('span', {'class': 'pick-name', text: ui().displayName(j)}), h('span', {'class': 'pick-result', text: ui().resultName(j, ctx.cards)}),
            h('span', {'class': 'pick-scan', text: j.scan_label || ''}),
            h('span', {'class': 'pick-time', text: ui().time(j.scan_date || j.created_at, false)}));
          b.addEventListener('click', function () { ui().dialog.close('done'); if (ctx.onPick) ctx.onPick(j.id); });
          list.appendChild(b);
        });
        body.push(list);
      });
      if (!body.length) body.push(ui().empty(total ? '没有匹配「' + search.value.trim() + '」的结果。' : '没有其他已完成的结果可以比较。'));
      count.textContent = search.value.trim() ? n + ' / ' + total : total + ' 个';
      ui().fill(results, body);
    }
    search.addEventListener('input', draw);
    draw();
    var head = total > 0 ? h('div', {'class': 'pick-head'}, search, count) : null;
    ui().dialog.open({title: '选择要比较的结果', body: [head, results], actions: [ui().button('取消', function () { ui().dialog.close('cancel'); })], wide: true, cls: 'dlg-pick', focus: total > 0 ? search : null});
  }

  return {conditions: conditions, sameScaleAllowed: sameScaleAllowed, sameGeometry: sameGeometry, commonRange: commonRange, render: render, candidates: candidates, pickDialog: pickDialog, timeText: timeText,
    // P3 lane 5
    identity: identity, matches: matches, scalarModel: scalarModel, reasonText: reasonText, fetchScalars: fetchScalars, _clearScalars: function () { scalarCache = {}; }};
});
