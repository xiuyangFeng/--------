/* WSS workspace v2 — evidence lens D9 (contract §6.3 ws_lens.js).
 * One chain per value: 结果 → 位置 / 区域 → 值（来源档）→ 映射与聚合方法 → 支撑（覆盖 / 几何参照 / 采样支撑，分开写）
 * → 留出集一致性 → 人群位置（只在同口径参照存在时）。 Unknown is written as unknown; no probability is invented. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.lens = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };
  var ov = function () { return ns.overview; };

  var TRUST_BITS = {1: '插值未覆盖', 2: '表面粗糙', 4: '几何超出参考范围', 8: '采样支撑弱', 16: '靠近开口'};
  function analysis(m) { return (m && m.analysis) || {}; }
  function branchName(m, segmentId) {
    var list = (m && m.geometry && m.geometry.branches) || [];
    for (var i = 0; i < list.length; i++) if (list[i] && list[i].id === segmentId) return list[i].name;
    return segmentId === undefined || segmentId === null ? '' : '分支 ' + segmentId;
  }
  function fieldLabel(m, id) {
    var f = ov().fieldOf(m, id);
    if (f) return f.short_label || f.label || id;
    return {wss: 'WSS', tawss: 'TAWSS', osi: 'OSI', rrt: 'RRT', ecap: 'ECAP', speed: '速度', pressure: '相对压力', stagnation: '滞留区'}[id] || id || '';
  }
  function validationFor(m, fieldId) {
    var f = ov().fieldOf(m, fieldId);
    if (f && f.validation) return f.validation;
    var card = m && m.model_card;
    var v = card && card.validation && card.validation.fields && card.validation.fields[fieldId];
    if (v) return Object.assign({holdout_n: card.validation.holdout_n}, v);
    return null;
  }
  function xyzText(xyz) {
    if (!xyz || xyz.length < 3) return '';
    return '(' + [0, 1, 2].map(function (i) { return ui().sig(xyz[i]); }).join(', ') + ') mm';
  }
  function trustRows(m, ref, ctx) {
    var t = analysis(m).trust || {};
    var fr = t.fractions || {};
    var rows = [];
    var ra = analysis(m).reference_assessment || {};
    var geoRef = ra.status === 'pass' ? '在几何参考范围内' : ra.status === 'review' ? '有分支超出几何参考范围，需复核' : '未配置几何参考范围';
    if (ref.kind === 'point' || ref.kind === 'finding' && ref.vertexBits !== undefined) {
      var bits = ref.vertexBits;
      if (bits === undefined && ctx && ctx.trustAt && ref.vertexIndex !== undefined && ref.vertexIndex !== null) bits = ctx.trustAt(ref.vertexIndex);
      if (bits === null || bits === undefined) rows.push(['该处标记', '未知（这份结果没有逐顶点可信标记）']);
      else {
        var flags = Object.keys(TRUST_BITS).filter(function (b) { return (bits & Number(b)) !== 0; }).map(function (b) { return TRUST_BITS[b]; });
        rows.push(['该处标记', flags.length ? flags.join('、') : '没有已知问题']);
      }
      rows.push(['几何参照', geoRef]);
      return rows;
    }
    rows.push(['覆盖', fr.interpolation_uncovered !== undefined ? '显示插值未覆盖 ' + ui().pct(fr.interpolation_uncovered) + ' 的壁面' : '未知']);
    rows.push(['几何参照', fr.geometry_out_of_range !== undefined && fr.geometry_out_of_range > 0 ? geoRef + '（' + ui().pct(fr.geometry_out_of_range) + ' 的壁面）' : geoRef]);
    var support = [];
    if (fr.rough_surface !== undefined) support.push('表面粗糙 ' + ui().pct(fr.rough_surface));
    if (fr.low_sample_support !== undefined) support.push('采样支撑弱 ' + ui().pct(fr.low_sample_support));
    rows.push(['采样支撑', support.length ? support.join('，') : '未知']);
    return rows;
  }
  function consistencyRow(m, fieldId, tier) {
    if (tier === 'geometry') return '几何测量，不经过模型，不适用。';
    if (tier === 'derived') {
      var f = ov().fieldOf(m, fieldId);
      var from = f && f.derived_from && f.derived_from.length ? f.derived_from.map(function (id) { return fieldLabel(m, id); }).join('、') : '预测量';
      return '派生量，由 ' + from + ' 计算，没有单独验证。';
    }
    var v = validationFor(m, fieldId);
    var n = v && v.holdout_n ? v.holdout_n + ' 例留出：' : '';
    var hasR2 = v && v.r2_pa !== undefined && v.r2_pa !== null;
    if (v && v.summary) return n + v.summary + (hasR2 ? '。R² 为 1 表示和 CFD 完全一致。' : '。');
    if (!hasR2) return v && v.note ? v.note : '未知（发布记录里没有这个量的留出集数字）。';
    var extra = [];
    if (v.extra) extra.push(v.extra);
    else if (v.ccc !== undefined && v.ccc !== null) extra.push('一致性系数 ' + ui().sig(v.ccc, 2));
    return n + 'Pa 口径 R² ' + ui().sig(v.r2_pa, 2) + (extra.length ? '；' + extra.join('；') : '') + '。R² 为 1 表示和 CFD 完全一致。';
  }
  function populationRow(m, fieldId, ref) {
    var ra = analysis(m).reference_assessment || {};
    var pop = ra.population || null;
    var sameMetric = pop && pop.status === 'pass' && (fieldId === 'wss') && (ref.stat === 'p99' || ref.kind === 'point' || ref.kind === 'finding');
    if (!sameMetric) return '暂无同口径参照。';
    var n = pop.reference_count || pop.n || pop.case_count;
    return '本例空间 p99 位于本研究队列' + (n ? '（' + n + ' 例模型预测）' : '') + '第 ' + ui().sig(pop.percentile, 2) + ' 百分位。这是模型预测分布里的位置，不是临床正常范围或风险。';
  }

  // Pure: ref + manifest → [{step, label, rows:[[k, v]], tier}]
  function chain(ref, m, ctx) {
    ctx = ctx || {};
    m = m || {};
    var steps = [];
    var job = m.job || {};
    var res = m.result || {};
    var fieldId = ref.field || ref.fieldId || null;
    var tier = ref.kind === 'morph' ? 'geometry' : ref.kind === 'finding' && ov().isGeometryFinding(ref.item) ? 'geometry' : (ref.tier || (fieldId ? ov().fieldTier(m, fieldId) : 'model'));
    if (ref.kind === 'finding' && ref.item && /stagnation/.test(ref.item.kind || '')) tier = 'derived';
    var phase = '—';
    var itemField = ref.kind === 'finding' && ref.item ? findingField(ref.item) : null;
    if ((fieldId && isCycle(m, fieldId)) || (itemField && isCycle(m, itemField)) || (ref.item && /stagnation/.test(ref.item.kind || ''))) {
      var period = m.time && m.time.cycle && m.time.cycle.period_s;
      phase = '一个心动周期' + (period ? '（' + ui().trim(period) + ' s）' : '') + '的汇总';
    } else if (res.model_frame && res.model_frame.target === 'peak_systole') {
      phase = '收缩期峰值帧' + (res.model_frame.time_s !== undefined ? '（' + ui().trim(res.model_frame.time_s) + ' s）' : '');
    }
    if (tier === 'geometry') phase = '与时相无关';
    steps.push({step: 'result', label: '结果', rows: [['结果', res.display_name || '—'], ['病例', ctx.hideName ? '（已隐藏）' : (job.display_name || '—')], ['时相', phase]]});
    var where = [];
    var value = [];
    var method = [];
    var f = fieldId ? ov().fieldOf(m, fieldId) : null;
    var fl = fieldId ? fieldLabel(m, fieldId) : '';
    if (ref.kind === 'point') {
      where.push(['位置', branchName(m, ref.segmentId) + (ref.s_from_root_mm !== undefined && ref.s_from_root_mm !== null ? '，距主动脉入口 ' + ui().num(ref.s_from_root_mm, 'mm') + '（沿中心线）' : '')]);
      if (ref.xyz) where.push(['坐标', xyzText(ref.xyz)]);
      value.push([fl, isNaN(Number(ref.value)) || ref.value === null ? '缺失（该点没有预测值）' : ui().num(ref.value, f && f.units)]);
      if (ref.values && typeof ref.values === 'object') {
        Object.keys(ref.values).forEach(function (id) {
          if (id === fieldId) return;
          var v = ref.values[id], of = ov().fieldOf(m, id);
          if (!of || of.kind === 'vector') return;
          var when = f && of.temporal && f.temporal && of.temporal !== f.temporal ? (of.temporal === 'frame' ? '峰值帧' : '周期') : '';
          var note = [when, of.tier === 'derived' ? '派生' : ''].filter(Boolean).join('，');
          value.push([fieldLabel(m, id), (v === null || v === undefined || isNaN(Number(v)) ? '缺失' : ui().num(v, of.units)) + (note ? '（' + note + '）' : '')]);
        });
      }
      if (ref.valueSource === 'display') method.push(['读数', '这个量只有显示网格上的值，读数取最近的显示顶点，不从屏幕颜色反推。']);
      else method.push(['读数', '最近预测点的值' + (ref.distance_mm !== undefined && ref.distance_mm !== null ? '（距点击处 ' + ui().num(ref.distance_mm, 'mm') + '）' : '') + '，不从屏幕颜色反推。']);
      method.push(['显示', '壁面颜色是显示用插值，只影响外观。']);
    } else if (ref.kind === 'zone') {
      var z = ref.zone || {};
      var s = Array.isArray(z.s_range_mm) ? '弧长 ' + ui().trim(z.s_range_mm[0]) + '–' + ui().trim(z.s_range_mm[1]) + ' mm' : '';
      where.push(['区域', (z.label || z.id || '') + (s ? '，' + s : '')]);
      if (z.n_points !== undefined) where.push(['预测点', String(Math.round(z.n_points)) + ' 个']);
      if (z.area_mm2 !== undefined) where.push(['面积（估计）', ui().num(z.area_mm2 / 100, 'cm²')]);
      value.push([ref.label || fl, ref.pct ? ui().pct(ref.value) : ui().num(ref.value, ref.units)]);
      if (ref.threshold !== undefined && ref.threshold !== null) value.push(['阈值', (ref.stat === 'frac_low' ? '< ' : '> ') + ui().num(ref.threshold, f && f.units, {trim: true}) + '（观察阈值，不是临床界值）']);
      var zdef = analysis(m).zones && analysis(m).zones.definition;
      method.push(['聚合', zdef || '区内预测点等权统计；占比是点占比。']);
      if (!/面积/.test(zdef || '')) method.push(['面积', '点占比 × 输入壁面面积，是估计值。']);
    } else if (ref.kind === 'pair') {
      where.push(['区域', (ref.pair && ref.pair.label || '') + ' · ' + (ref.side === 'left' ? '左侧' : '右侧')]);
      value.push([fl + ' 均值', ui().num(ref.value, ref.units)]);
      method.push(['聚合', '该侧分区内预测点等权平均；比值 = 左均值 ÷ 右均值。']);
    } else if (ref.kind === 'branch') {
      where.push(['区域', ref.name || '']);
      value.push([fl + ' 均值', ui().num(ref.value, ref.units)]);
      method.push(['聚合', '分支内预测点等权平均。']);
    } else if (ref.kind === 'morph') {
      where.push(['位置', ref.label === '管腔最大直径' && analysis(m).morphology && analysis(m).morphology.aorta && analysis(m).morphology.aorta.max
        ? '主动脉，入口下 ' + ui().num(analysis(m).morphology.aorta.max.s_from_root_mm, 'mm') : '主动脉']);
      value.push([ref.label, ui().num(ref.value, ref.units)]);
      method.push(['测量', ref.definition || '由输入表面的截面计算。']);
      var a = analysis(m).morphology && analysis(m).morphology.aorta;
      if (a && (a.n_reoriented !== undefined || a.n_excluded !== undefined)) method.push(['截面可靠性', (a.n_reoriented || 0) + ' 站重新定向，' + (a.n_excluded || 0) + ' 站不可靠未计入']);
    } else if (ref.kind === 'finding') {
      var it = ref.item || {};
      where.push(['位置', (it.branch || '') + (it.s_from_root_mm !== undefined ? '，距主动脉入口 ' + ui().num(it.s_from_root_mm, 'mm') : '')]);
      if (it.area_mm2 !== undefined && it.area_mm2 !== null) where.push(['面积（估计）', ui().num(it.area_mm2 / 100, 'cm²')]);
      if (it.n_points) where.push(['预测点', String(Math.round(it.n_points)) + ' 个']);
      value.push([ov().kindLabel(it), ov().findingValue(it)]);
      var sev = ov().severityText(it);
      if (sev) value.push(['自动分级', sev.label + (it.grading === 'no_reference' ? '（无队列参照，不定级）' : '')]);
      if (it.contains_global_max) value.push(['说明', '包含全场最大值点']);
      method.push(['判定', it.definition || '—']);
      fieldId = fieldId || findingField(it);
    } else if (ref.kind === 'cursor') {
      var r = ref.readout || {};
      where.push(['位置', branchName(m, r.segmentId) + (r.s_mm !== undefined ? '，分支内 ' + ui().num(r.s_mm, 'mm') : '')]);
      if (Array.isArray(r.bin)) where.push(['分箱', ui().trim(r.bin[0]) + '–' + ui().trim(r.bin[1]) + ' mm']);
      Object.keys(r.stats || {}).forEach(function (k) { value.push([statLabel(k), typeof r.stats[k] === 'number' ? ui().sig(r.stats[k]) : String(r.stats[k])]); });
      method.push(['聚合', r.definition || '沿中心线每 2 mm 一箱，箱内预测点统计。']);
    } else if (ref.kind === 'stat') {
      value.push([ref.label || fl, ref.text || ui().num(ref.value, ref.units)]);
      method.push(['聚合', '体内采样点等权统计。' + (fieldId === 'pressure' ? '压力是相对压，只用于比较差值。' : '')]);
    }
    if (where.length) steps.push({step: 'where', label: '位置 / 区域', rows: where});
    steps.push({step: 'value', label: '值', rows: value, tier: tier});
    if (f && f.definition && ref.kind !== 'morph') method.push(['定义', f.definition]);
    steps.push({step: 'method', label: '映射与聚合', rows: method});
    steps.push({step: 'support', label: '支撑', rows: tier === 'geometry' ? [['来源', '直接从输入表面测量，不经过模型。']] : trustRows(m, ref, ctx)});
    steps.push({step: 'consistency', label: '留出集一致性', rows: [['', consistencyRow(m, fieldId, tier)]]});
    if (tier !== 'geometry') steps.push({step: 'population', label: '人群位置', rows: [['', populationRow(m, fieldId, ref)]]});
    return steps;
  }
  function findingField(it) {
    var k = (it && it.kind) || '';
    return /osi/.test(k) ? 'osi' : /tawss/.test(k) ? 'tawss' : /wss/.test(k) ? 'wss' : /speed/.test(k) ? 'speed' : /pressure/.test(k) ? 'pressure' : null;
  }
  function isCycle(m, fieldId) { var f = ov().fieldOf(m, fieldId); return Boolean(f && f.temporal === 'cycle_summary'); }
  function statLabel(k) {
    return {mean: '均值', mean_pa: '均值', p99: 'p99', p99_pa: 'p99', min: '最小', min_pa: '最小', max: '最大', n: '预测点', radius_mm: '中心线半径', speed_mean_m_s: '速度均值', speed_max_m_s: '最大速度', pressure_mean_pa: '相对压力均值', pressure_min_pa: '相对压力最小'}[k] || k;
  }

  function render(el, ref, ctx) {
    var h = ui().h;
    ctx = ctx || {};
    if (!ref) {
      el.replaceChildren(ui().empty('在血管上点一个位置，或点概览里的任一数字，这里写出它从哪里来、怎么算、能信多少。'));
      return;
    }
    var steps = chain(ref, ctx.manifest, ctx);
    var list = h('ol', {'class': 'lens'});
    steps.forEach(function (s) {
      var head = h('div', {'class': 'lens-step-head'}, h('span', {'class': 'lens-label', text: s.label}), s.tier ? ui().tierTag(s.tier) : null);
      var body = h('dl', {'class': 'lens-rows'});
      s.rows.forEach(function (r) { if (r[0]) body.appendChild(h('dt', {text: r[0]})); body.appendChild(h('dd', {'class': r[0] ? '' : 'wide', text: r[1] === undefined || r[1] === null || r[1] === '' ? '未知' : String(r[1])})); });
      list.appendChild(h('li', {'class': 'lens-step', dataset: {step: s.step}}, head, body));
    });
    ui().fill(el, list, ctx.onClear ? h('div', {'class': 'sec-actions'}, ui().button('清除', ctx.onClear, {kind: 'link', cls: 'btn-sm'})) : null);
  }

  return {chain: chain, render: render, TRUST_BITS: TRUST_BITS, branchName: branchName, fieldLabel: fieldLabel};
});
