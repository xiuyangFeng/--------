/* WSS workspace v2 — compare D11 + two viewports D3 (contract §6.3 ws_compare.js).
 * Before two results are read side by side, the conditions are listed one by one (field, units, time, model,
 * geometry, statistics, orientation) with 一致 / 不一致 / 未知.  Same-scale mode only for compatible fields;
 * camera sync on by default only for the same input geometry.  No point-by-point difference.
 * Second phase lane E: 以左为准 / 以右为准 (classic push-left / push-right, through ws_display.pushSide) and, in the
 * same-scale mode, one set of colour bands and thresholds for both sides (ws_display follows the mode). */
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
    var sA = mA && mA.mapping && mA.mapping.statistics_protocol, sB = mB && mB.mapping && mB.mapping.statistics_protocol;
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

  // ------------------------------------------------------------------ inspector panel
  function render(el, ctx) {
    var h = ui().h;
    var rows = conditions(ctx.left.manifest, ctx.right.manifest, ctx.left.field, ctx.right.field);
    var allow = sameScaleAllowed(rows);
    var geo = sameGeometry(ctx.left.manifest, ctx.right.manifest);
    var who = function (side, x) {
      return h('div', {'class': 'cmp-who'}, h('span', {'class': 'cmp-side', text: side}), h('span', {'class': 'cmp-name', text: x.name}), h('span', {'class': 'muted', text: x.result}));
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
      ui().section('怎么看', {}, modes, ctx.mode === 'each' ? ui().note('两侧色标范围不同，同一种颜色不代表同一个数。', 'warn') : ui().note('两侧用同一个色标范围：' + (ctx.rangeText || '—') + (D ? '；分段和阈值也一起改' : '') + '。'), push),
      ui().section('视角', {}, syncRow, geo ? ui().note('同一输入几何：视角同步后，同一屏幕位置是同一处血管。') : ui().note('不同输入：只同步视角，不代表位置对应。', 'warn')),
      ui().note('不做逐点差值：两份结果的预测点不一一对应。'),
      h('div', {'class': 'sec-actions'}, ctx.onSwap ? ui().button('左右互换', ctx.onSwap, {cls: 'btn-sm'}) : null, ctx.onClose ? ui().button('退出比较', ctx.onClose, {cls: 'btn-sm'}) : null)
    ];
    ui().fill(el, parts);
    return {rows: rows, allow: allow};
  }

  // ------------------------------------------------------------------ choose the second result
  function candidates(jobs, current) {
    var done = (jobs || []).filter(function (j) { return j && j.status === 'done' && j.id !== current.id; });
    var sameInput = [], samePatient = [], others = [];
    done.forEach(function (j) {
      if (current.input_sha256 && j.input_sha256 === current.input_sha256) sameInput.push(j);
      else if (current.patient_id && j.patient_id === current.patient_id) samePatient.push(j);
      else others.push(j);
    });
    return {sameInput: sameInput, samePatient: samePatient, others: others.slice(0, 20)};
  }
  function pickDialog(ctx) {
    var h = ui().h;
    var c = candidates(ctx.jobs, ctx.current || {});
    var groups = [['同一输入的其他结果', c.sameInput], ['同一患者的其他扫描', c.samePatient], ['其他已完成结果', c.others]];
    var body = [];
    groups.forEach(function (g) {
      if (!g[1].length) return;
      body.push(h('h4', {text: g[0]}));
      var list = h('div', {'class': 'pick-list'});
      g[1].forEach(function (j) {
        var b = h('button', {type: 'button', 'class': 'pick', dataset: {jobId: j.id}},
          h('span', {'class': 'pick-name', text: ui().displayName(j)}), h('span', {'class': 'pick-result', text: ui().resultName(j, ctx.cards)}),
          h('span', {'class': 'pick-time', text: ui().time(j.scan_date || j.created_at, false)}));
        b.addEventListener('click', function () { ui().dialog.close('done'); if (ctx.onPick) ctx.onPick(j.id); });
        list.appendChild(b);
      });
      body.push(list);
    });
    if (!body.length) body.push(ui().empty('没有其他已完成的结果可以比较。'));
    ui().dialog.open({title: '选择要比较的结果', body: body, actions: [ui().button('取消', function () { ui().dialog.close('cancel'); })], wide: true});
  }

  return {conditions: conditions, sameScaleAllowed: sameScaleAllowed, sameGeometry: sameGeometry, commonRange: commonRange, render: render, candidates: candidates, pickDialog: pickDialog, timeText: timeText};
});
