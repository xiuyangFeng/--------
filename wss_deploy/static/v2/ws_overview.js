/* WSS workspace v2 — overview inspector (contract §6.3 ws_overview.js).
 * Order: identity + result switch + status → conclusion → zones (U10) → left / right → morphology → findings
 * (one per kind) → follow-up (U17) → model card.  Every number opens the evidence lens (D9). */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.overview = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };

  // ------------------------------------------------------------------ pure helpers (Node-testable)
  function analysis(manifest) { return (manifest && manifest.analysis) || {}; }
  function fieldOf(manifest, id) {
    var list = (manifest && manifest.fields) || [];
    for (var i = 0; i < list.length; i++) if (list[i] && list[i].id === id) return list[i];
    return null;
  }
  function fieldTier(manifest, id) {
    var f = fieldOf(manifest, id);
    if (f && f.tier) return f.tier;
    if (id === 'stagnation' || id === 'rrt' || id === 'ecap') return 'derived';
    return 'model';
  }
  function thresholdOf(manifest, id, side) {
    var f = fieldOf(manifest, id);
    var t = f && f.display && f.display.thresholds;
    if (Array.isArray(t) && t.length) return side === 'high' ? (id === 'osi' ? t[0] : t[1]) : t[0];
    var d = DEFAULT_THRESHOLDS[id];
    return d ? d[side] : null;
  }
  var DEFAULT_THRESHOLDS = {wss: {low: 0.4, high: 4}, tawss: {low: 0.4, high: 4}, osi: {low: null, high: 0.1}};
  var ZONE_COLUMNS = {
    tawss: [{stat: 'mean', label: 'TAWSS 均值', units: 'Pa'}, {stat: 'frac_low', label: '低剪切占比', pct: true, side: 'low'}],
    osi: [{stat: 'mean', label: 'OSI 均值', units: '1'}, {stat: 'frac_high', label: '高 OSI 占比', pct: true, side: 'high'}],
    wss: [{stat: 'mean', label: 'WSS 均值', units: 'Pa'}, {stat: 'frac_low', label: '低 WSS 占比', pct: true, side: 'low'}, {stat: 'p99', label: 'WSS p99', units: 'Pa'}]
  };
  function zoneColumns(zones) {
    if (!zones || !Array.isArray(zones.zones)) return [];
    var present = {};
    zones.zones.forEach(function (z) { Object.keys((z && z.fields) || {}).forEach(function (f) { present[f] = true; }); });
    var primary = Array.isArray(zones.primary_fields) && zones.primary_fields.length ? zones.primary_fields : Object.keys(present);
    var cols = [];
    primary.forEach(function (f) { if (present[f]) (ZONE_COLUMNS[f] || [{stat: 'mean', label: f + ' 均值'}]).forEach(function (c) { cols.push(Object.assign({field: f}, c)); }); });
    return cols.slice(0, 3);
  }
  function zonesModel(manifest) {
    var zones = analysis(manifest).zones;
    if (!zones || !Array.isArray(zones.zones) || !zones.zones.length) return null;
    var cols = zoneColumns(zones);
    if (!cols.length) return null;
    var rows = zones.zones.filter(Boolean).map(function (z) {
      return {zone: z, cells: cols.map(function (c) { var f = (z.fields || {})[c.field] || {}; return {col: c, value: f[c.stat]}; })};
    });
    return {columns: cols, rows: rows, definition: zones.definition || '', pairs: pairsModel(zones)};
  }
  function pairsModel(zones) {
    if (!zones || !Array.isArray(zones.pairs) || !zones.pairs.length) return null;
    var primary = (zones.primary_fields || []).filter(function (f) { return zones.pairs.some(function (p) { return p && p.fields && p.fields[f]; }); })[0];
    if (!primary) primary = Object.keys((zones.pairs[0] && zones.pairs[0].fields) || {})[0];
    if (!primary) return null;
    return {field: primary, rows: zones.pairs.filter(Boolean).map(function (p) {
      var f = (p.fields || {})[primary] || {};
      return {pair: p, left: f.left_mean, right: f.right_mean, ratio: f.ratio_left_over_right};
    })};
  }
  // Legacy wall result without zones: per-branch statistics in manifest branch order.
  function branchModel(manifest) {
    var per = analysis(manifest).per_branch;
    if (!per || typeof per !== 'object' || !Object.keys(per).length) return null;
    var names = ((manifest.geometry && manifest.geometry.branches) || []).map(function (b) { return b.name; });
    Object.keys(per).forEach(function (n) { if (names.indexOf(n) < 0) names.push(n); });
    var rows = names.filter(function (n) { return per[n]; }).map(function (n) { return {name: n, stats: per[n]}; });
    if (!rows.length || !rows.some(function (r) { return r.stats.wss_mean_pa !== undefined || r.stats.wss_p99_pa !== undefined; })) return null;
    return rows;
  }
  function morphologyModel(manifest) {
    var m = analysis(manifest).morphology;
    var a = m && m.aorta;
    if (!a) return null;
    var rows = [];
    var max = a.max || {};
    if (max.max_diameter_mm !== undefined) rows.push({key: 'max_diameter', label: '管腔最大直径', value: max.max_diameter_mm, units: 'mm',
      sub: max.s_from_root_mm !== undefined ? '入口下 ' + ui().sig(max.s_from_root_mm) + ' mm' : '', definition: '中心线每 1 mm 一站取截面，环上最大 Feret 直径；只含管腔，不含附壁血栓与管壁，通常小于 CT 报告的瘤体直径。'});
    var sac = a.sac || {};
    if (sac.present) rows.push({key: 'sac', label: '瘤体', value: sac.length_mm, units: 'mm', prefix: '长 ',
      sub: sac.volume_ml !== undefined ? '体积 ' + ui().num(sac.volume_ml, 'mL') : '', definition: '等效直径 ≥ 1.5 × 参考直径的连续区段；按管腔判定。'});
    else if (a.sac) rows.push({key: 'sac', label: '瘤体', text: '未见管腔瘤样扩张', definition: '等效直径没有达到 1.5 × 参考直径；按管腔判定，不能据此排除动脉瘤。'});
    var neck = a.neck || {};
    if (neck.present) rows.push({key: 'neck', label: '近端瘤颈', value: neck.length_mm, units: 'mm', prefix: '长 ',
      sub: neck.diameter_mean_mm !== undefined ? '平均直径 ' + ui().num(neck.diameter_mean_mm, 'mm') : '', definition: '入口到瘤体起点之间、等效直径 < 1.2 × 参考直径的近端区段；按管腔判定。'});
    if (a.reference_diameter_mm !== undefined) rows.push({key: 'reference_diameter', label: '参考直径', value: a.reference_diameter_mm, units: 'mm', definition: '主动脉等效直径的第 10 百分位，用来估计正常管径。'});
    if (m.lumen_volume_ml !== undefined) rows.push({key: 'lumen_volume', label: '管腔体积', value: m.lumen_volume_ml, units: 'mL', definition: '开口封盖后按散度定理求全腔体积（管腔）。'});
    return rows.length ? rows : null;
  }
  var SEVERITY = {attention: 3, note: 2, info: 1};
  var KIND_LABEL = {high_wss_cluster: '高 WSS 区', low_wss_cluster: '低 WSS 区', max_wss: '全场最大 WSS', max_diameter: '管腔最大直径', min_radius: '最小半径',
    stagnation_cluster: '滞留区', high_osi_cluster: '高 OSI 区', low_tawss_cluster: '低 TAWSS 区', max_speed: '最大速度', min_pressure: '最低相对压力',
    pressure_drop: '沿程压降', low_speed_region: '低速区'};
  var KIND_ORDER = ['low_tawss_cluster', 'stagnation_cluster', 'high_osi_cluster', 'low_wss_cluster', 'high_wss_cluster', 'max_wss', 'max_speed', 'min_pressure', 'pressure_drop', 'low_speed_region', 'max_diameter', 'min_radius'];
  function findingsModel(manifest) {
    var f = analysis(manifest).findings;
    var items = (f && Array.isArray(f.items)) ? f.items.filter(function (it) { return it && typeof it === 'object'; }) : [];
    var review = (f && f.review && f.review.items) || {};
    var byKind = {};
    items.forEach(function (it, i) {
      var k = it.kind || 'other';
      var cur = byKind[k];
      var rank = it.rank !== undefined ? it.rank : i + 1;
      if (!cur || (SEVERITY[it.severity] || 0) > (SEVERITY[cur.severity] || 0) || ((SEVERITY[it.severity] || 0) === (SEVERITY[cur.severity] || 0) && rank < (cur.rank !== undefined ? cur.rank : 1e9))) byKind[k] = it;
    });
    var top = Object.keys(byKind).map(function (k) { return byKind[k]; });
    top.sort(function (a, b) {
      var s = (SEVERITY[b.severity] || 0) - (SEVERITY[a.severity] || 0);
      if (s) return s;
      var ia = KIND_ORDER.indexOf(a.kind), ib = KIND_ORDER.indexOf(b.kind);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    var undecided = items.filter(function (it) { return it.id && !(review[it.id] && review[it.id].decision); });
    return {items: items, top: top, review: review, undecided: undecided, total: items.length};
  }
  function isGeometryFinding(item) { return Boolean(item && (item.source === 'morphology' || item.kind === 'max_diameter' || item.kind === 'min_radius')); }
  function kindLabel(item) { return KIND_LABEL[item && item.kind] || (item && item.label) || '发现'; }
  function severityText(item) {
    if (!item) return null;
    if (item.severity === 'attention') return {tone: 'warn', label: '关注'};
    if (item.severity === 'note') return {tone: 'idle', label: item.grading === 'no_reference' ? '提示 · 无队列参照' : '提示'};
    return null;
  }
  function findingValue(item) {
    if (!item) return '—';
    if (item.units === 'cm²' || item.units === 'cm2') return ui().num(item.value, 'cm²');
    return ui().num(item.value, item.units);
  }
  var DISCLAIMER = /非诊断|not a diagnosis|for reference only/i;
  function narrativeModel(manifest) {
    var n = analysis(manifest).narrative;
    if (!n || typeof n !== 'object') return null;
    var edited = typeof n.edited === 'string' && n.edited.trim() ? n.edited.trim() : null;
    if (edited) return {edited: true, sentences: edited.split(/\n+/).filter(function (s) { return s.trim(); }), by: n.edited_by || '', at: n.edited_at || ''};
    var zh = Array.isArray(n.zh) ? n.zh.filter(function (s) { return typeof s === 'string' && s.trim() && !DISCLAIMER.test(s); }) : [];
    return zh.length ? {edited: false, sentences: zh} : null;
  }
  function followupModel(timeline, releaseId) {
    if (!timeline || !Array.isArray(timeline.scans) || timeline.scans.length < 2) return null;
    var metricKey = null, metricLabel = '', metricUnits = '';
    var candidates = [['tawss_mean_pa', 'TAWSS 均值', 'Pa'], ['wss_p99_pa', 'WSS p99', 'Pa'], ['speed_p99_m_s', '速度 p99', 'm/s']];
    for (var i = 0; i < candidates.length && !metricKey; i++) {
      var key = candidates[i][0];
      if (timeline.scans.some(function (s) { return s.models && s.models[releaseId] && s.models[releaseId][key] !== undefined; })) { metricKey = key; metricLabel = candidates[i][1]; metricUnits = candidates[i][2]; }
    }
    var rows = timeline.scans.map(function (s) {
      var g = s.geometry || {};
      var m = (s.models && s.models[releaseId]) || {};
      return {date: s.date || '', label: s.scan_label || '', diameter: g.max_diameter_mm, sacVolume: g.sac_volume_ml, metric: metricKey ? m[metricKey] : undefined};
    });
    var growth = timeline.growth && timeline.growth.max_diameter_mm;
    return {rows: rows, metricLabel: metricLabel, metricUnits: metricUnits, growth: growth || null, notes: timeline.notes || []};
  }

  // ------------------------------------------------------------------ rendering
  function header(el, ctx) {
    var h = ui().h;
    var m = ctx.manifest || {};
    var job = ctx.job || m.job || {};
    var name = ctx.hideName ? '病例（名称已隐藏）' : (m.job && m.job.display_name) || ui().displayName(job);
    var scan = [];
    var mj = m.job || {};
    if (mj.scan_date || job.scan_date) scan.push('扫描 ' + (mj.scan_date || job.scan_date));
    else if (mj.created_at || job.created_at) scan.push('上传 ' + ui().time(mj.created_at || job.created_at, false));
    if (mj.scan_label || job.scan_label) scan.push(mj.scan_label || job.scan_label);
    if (!ctx.hideName && (mj.patient_id || job.patient_id) && (mj.patient_id || job.patient_id) !== name) scan.push('患者 ' + (mj.patient_id || job.patient_id));
    var box = h('div', {'class': 'insp-id'}, h('div', {'class': 'insp-name', text: name}), scan.length ? h('div', {'class': 'insp-scan', text: scan.join(' · ')}) : null);
    var sib = ctx.offline ? [] : (ctx.siblings || []);
    if (sib.length === 2) {
      var seg = h('div', {'class': 'seg seg-results', role: 'tablist', 'aria-label': '同一输入的结果'});
      sib.forEach(function (s) {
        var b = h('button', {type: 'button', 'class': 'seg-btn' + (s.current ? ' on' : ''), role: 'tab', 'aria-selected': String(Boolean(s.current)), text: s.name, title: s.title || s.name});
        if (!s.current && ctx.onOpenResult) b.addEventListener('click', function () { ctx.onOpenResult(s.jobId); });
        seg.appendChild(b);
      });
      box.appendChild(seg);
    } else if (sib.length > 2) {
      var cur = sib.filter(function (s) { return s.current; })[0];
      var sel = ui().select(sib.map(function (s) { return {value: s.jobId, label: s.name}; }), cur ? cur.jobId : null,
        function (v) { if (ctx.onOpenResult && (!cur || v !== cur.jobId)) ctx.onOpenResult(v); }, {'aria-label': '同一输入的结果', 'class': 'result-select'});
      box.appendChild(h('label', {'class': 'result-pick'}, h('span', {'class': 'muted', text: '结果'}), sel, h('span', {'class': 'muted', text: '共 ' + sib.length + ' 个'})));
    } else if (m.result) {
      box.appendChild(h('div', {'class': 'insp-result', text: m.result.display_name || ui().resultName(job, ctx.cards)}));
    }
    var row = h('div', {'class': 'insp-status'});
    if (ctx.offline) {
      row.appendChild(ui().dot(ui().reviewInfo(mj.review).tone, '导出时' + ui().reviewInfo(mj.review).label));
    } else {
      row.appendChild(ui().statusDot(job.status || mj.status || 'done'));
      row.appendChild(ui().reviewDot(job.review || mj.review));
      if (ctx.onReview) row.appendChild(ui().button(ui().reviewInfo(job.review || mj.review).key === 'reviewed' ? '复核记录' : '技术复核', ctx.onReview, {cls: 'btn-sm'}));
    }
    box.appendChild(row);
    el.replaceChildren(box);
    return box;
  }

  function render(el, ctx) {
    var h = ui().h;
    var m = ctx.manifest || {};
    var lens = ctx.onLens || null;
    var out = [];
    // 2. conclusion
    var nar = narrativeModel(m);
    if (nar) {
      var tag = h('span', {'class': 'sec-note', text: nar.edited ? '人工编辑' + (nar.by ? ' · ' + nar.by : '') : '自动生成'});
      var actions = ctx.onEditNarrative ? ui().button('编辑', ctx.onEditNarrative, {kind: 'link', cls: 'btn-sm'}) : null;
      out.push(ui().section('结论', {tag: tag, actions: actions, cls: 'sec-conclusion'}, h('ul', {'class': 'narr'}, nar.sentences.map(function (s) { return h('li', {text: s}); }))));
    }
    // 3. zones (U10) and 4. left / right
    var zm = zonesModel(m);
    if (zm) {
      var tiers = {}; zm.columns.forEach(function (c) { tiers[fieldTier(m, c.field)] = true; });
      var cols = [{key: 'label', label: '分区', render: function (r) { return r.zone.label || r.zone.id; }}].concat(zm.columns.map(function (c, i) {
        var thr = c.side ? thresholdOf(m, c.field, c.side) : null;
        return {key: 'c' + i, label: c.label, num: true, render: function (r) {
          var cell = r.cells[i];
          var text = c.pct ? ui().pct(cell.value) : ui().num(cell.value, c.units);
          return ui().evidence(text, lens && cell.value !== undefined && cell.value !== null ? function () { lens({kind: 'zone', zone: r.zone, field: c.field, stat: c.stat, value: cell.value, units: c.pct ? null : c.units, pct: Boolean(c.pct), label: c.label, threshold: thr}); } : null);
        }};
      }));
      var sec = ui().section('分区', {tag: tagsFor(tiers)}, ui().table(cols, zm.rows, {cls: 'tbl-zones'}), ui().note('区内预测点等权统计；占比是点占比，面积为估计。'));
      out.push(sec);
      if (zm.pairs && zm.pairs.rows.length) {
        var pf = zm.pairs.field, pu = (fieldOf(m, pf) || {}).units || (pf === 'osi' ? '1' : 'Pa');
        var fl = (fieldOf(m, pf) || {}).short_label || pf.toUpperCase();
        var pcols = [
          {key: 'label', label: '', render: function (r) { return r.pair.label || r.pair.id; }},
          {key: 'l', label: '左', num: true, render: function (r) { return ui().evidence(ui().num(r.left, pu), lens && r.left !== undefined ? function () { lens({kind: 'pair', pair: r.pair, field: pf, side: 'left', value: r.left, units: pu}); } : null); }},
          {key: 'r', label: '右', num: true, render: function (r) { return ui().evidence(ui().num(r.right, pu), lens && r.right !== undefined ? function () { lens({kind: 'pair', pair: r.pair, field: pf, side: 'right', value: r.right, units: pu}); } : null); }},
          {key: 'q', label: '左 / 右', num: true, render: function (r) { return r.ratio === null || r.ratio === undefined ? '—' : ui().sig(r.ratio); }}
        ];
        out.push(ui().section('左右对比 · ' + fl + ' 均值', {tag: ui().tierTag(fieldTier(m, pf))}, ui().table(pcols, zm.pairs.rows, {cls: 'tbl-pairs'})));
      }
    } else if (m.result && m.result.family !== 'volume') {
      var bm = branchModel(m);
      if (bm) {
        var bcols = [{key: 'name', label: '分支', render: function (r) { return r.name; }},
          {key: 'mean', label: 'WSS 均值', num: true, render: function (r) { return ui().evidence(ui().num(r.stats.wss_mean_pa, 'Pa'), lens ? function () { lens({kind: 'branch', name: r.name, field: 'wss', stat: 'mean', value: r.stats.wss_mean_pa, units: 'Pa', stats: r.stats}); } : null); }},
          {key: 'low', label: '低 WSS 占比', num: true, render: function (r) { return ui().pct(r.stats.frac_low); }},
          {key: 'p99', label: 'WSS p99', num: true, render: function (r) { return ui().num(r.stats.wss_p99_pa, 'Pa'); }}];
        out.push(ui().section('分支', {tag: ui().tierTag('model')}, ui().table(bcols, bm, {cls: 'tbl-zones'}), ui().note('这份结果没有分区统计，按中心线分支列出；区内预测点等权。')));
      }
    }
    // volume family: interior statistics
    var vs = analysis(m).volume_statistics;
    if (vs && typeof vs === 'object') {
      var vrows = [];
      if (vs.speed_m_s) vrows.push({label: '体内速度 p99', value: vs.speed_m_s.p99, units: 'm/s', field: 'speed', stat: 'p99'}, {label: '最大速度', value: vs.speed_m_s.max, units: 'm/s', field: 'speed', stat: 'max'});
      if (vs.pressure_interior_pa) vrows.push({label: '相对压力范围', text: ui().sig(vs.pressure_interior_pa.min) + ' – ' + ui().num(vs.pressure_interior_pa.max, 'Pa'), field: 'pressure', stat: 'range', value: vs.pressure_interior_pa.max, units: 'Pa'});
      if (vrows.length) {
        var vcols = [{key: 'label', label: '', render: function (r) { return r.label; }},
          {key: 'v', label: '', num: true, render: function (r) { return ui().evidence(r.text || ui().num(r.value, r.units), lens ? function () { lens({kind: 'stat', field: r.field, stat: r.stat, value: r.value, units: r.units, label: r.label, text: r.text}); } : null); }}];
        out.push(ui().section('体场', {tag: ui().tierTag('model')}, ui().table(vcols, vrows, {cls: 'tbl-kv tbl-nohead'}), ui().note('压力是相对压，只用于比较同一结果内的差值。')));
      }
    }
    // 5. morphology (geometry tier)
    var mo = morphologyModel(m);
    if (mo) {
      var mcols = [{key: 'label', label: '', render: function (r) { return r.label; }},
        {key: 'v', label: '', num: true, render: function (r) {
          if (r.text) return h('span', {'class': 'muted', text: r.text});
          return ui().evidence((r.prefix || '') + ui().num(r.value, r.units), lens ? function () { lens({kind: 'morph', key: r.key, label: r.label, value: r.value, units: r.units, definition: r.definition}); } : null);
        }},
        {key: 's', label: '', cls: 'sub', blank: true, render: function (r) { return r.sub || ''; }}];
      out.push(ui().section('形态', {tag: ui().tierTag('geometry')}, ui().table(mcols, mo, {cls: 'tbl-kv tbl-nohead'}), ui().note('直径、长度和体积都是管腔的，不含附壁血栓与管壁。')));
    }
    // 6. findings: the heaviest one per kind
    var fm = findingsModel(m);
    if (fm.total) {
      var expanded = Boolean(ctx.findingsExpanded);
      var list = expanded ? fm.items : fm.top;
      var rows = h('div', {'class': 'findings'});
      list.forEach(function (it) {
        var sev = severityText(it);
        var decision = fm.review[it.id] && fm.review[it.id].decision;
        var row = h('button', {type: 'button', 'class': 'finding' + (ctx.selectedFinding === it.id ? ' on' : ''), dataset: {findingId: it.id || ''}, title: it.definition || ''},
          h('span', {'class': 'f-kind', text: kindLabel(it)}),
          h('span', {'class': 'f-where', text: it.branch || ''}),
          h('span', {'class': 'f-val', text: findingValue(it)}),
          sev ? ui().dot(sev.tone, sev.label, 'f-sev') : h('span', {'class': 'f-sev muted', text: isGeometryFinding(it) ? '几何' : ''}),
          decision ? h('span', {'class': 'f-dec', text: decision === 'confirmed' ? '已确认' : decision === 'rejected' ? '已驳回' : ''}) : null,
          it.contains_global_max ? h('span', {'class': 'f-flag', text: '含全场最大值'}) : null);
        row.addEventListener('click', function () { if (ctx.onFinding) ctx.onFinding(it); });
        rows.appendChild(row);
      });
      var more = fm.total > fm.top.length ? ui().button(expanded ? '只看每类最重的一条' : '全部 ' + fm.total + ' 条', function () { if (ctx.onToggleFindings) ctx.onToggleFindings(); }, {kind: 'link', cls: 'btn-sm'}) : null;
      var confirmRest = null;
      if (ctx.onConfirmRest && fm.undecided.length) confirmRest = ui().button((fm.undecided.length === fm.total ? '全部 ' : '其余 ') + fm.undecided.length + ' 条按自动结果确认', function () { ctx.onConfirmRest(fm); }, {cls: 'btn-sm', disabled: Boolean(ctx.locked), title: ctx.locked ? '已复核锁定；重新打开后才能修改' : '未判定的发现一律记为「确认」，已判定的不变'});
      out.push(ui().section('发现', {actions: more}, rows, confirmRest ? h('div', {'class': 'sec-actions'}, confirmRest) : null,
        ui().note('自动发现只是候选位置；「关注」「提示」不是临床分级。')));
    }
    // 7. follow-up (U17)
    var fu = followupModel(ctx.timeline, m.result && m.result.release_id);
    if (fu) {
      var fcols = [{key: 'date', label: '扫描', render: function (r) { return r.date + (r.label ? ' ' + r.label : ''); }},
        {key: 'd', label: '管腔最大直径', num: true, render: function (r) { return ui().num(r.diameter, 'mm'); }},
        {key: 'v', label: '瘤体体积', num: true, render: function (r) { return ui().num(r.sacVolume, 'mL'); }}];
      if (fu.metricLabel) fcols.push({key: 'm', label: fu.metricLabel, num: true, render: function (r) { return ui().num(r.metric, fu.metricUnits); }});
      var growth = fu.growth && fu.growth.per_year !== undefined ? ui().note('管腔最大直径年增长 ' + (fu.growth.per_year > 0 ? '+' : '') + ui().sig(fu.growth.per_year) + ' mm/年（首末两次扫描）') : null;
      out.push(ui().section('随访', {tag: ui().tierTag('geometry')}, ui().table(fcols, fu.rows, {cls: 'tbl-follow'}), growth,
        fu.notes.length ? ui().note(fu.notes[0]) : null));
    }
    // 8. model card + the only research-use sentence of the page
    var card = m.model_card || (ctx.cards && m.result && ctx.cards[m.result.release_id]) || null;
    var mc = h('div', {'class': 'model-line'},
      h('span', {'class': 'model-name', text: (card && card.display_name) || (m.result && m.result.display_name) || '模型'}),
      card && card.version_date ? h('span', {'class': 'muted', text: card.version_date}) : null,
      h('span', {'class': 'sec-fill'}),
      ctx.onModelCard ? ui().button('模型说明', ctx.onModelCard, {kind: 'link', cls: 'btn-sm'}) : null);
    out.push(h('section', {'class': 'sec sec-model'}, mc, h('p', {'class': 'research', text: '研究用途，非诊断。'})));
    ui().fill(el, out);
    return {findings: fm, zones: zm};
  }
  function tagsFor(tiers) {
    var keys = Object.keys(tiers);
    if (!keys.length) return null;
    var h = ui().h;
    return h('span', {'class': 'tier-group'}, keys.map(function (k) { return ui().tierTag(k); }));
  }

  // ------------------------------------------------------------------ model card dialog (U7)
  var FIELD_NAMES = {stagnation: '滞留区', max_diameter: '管腔最大直径', speed: '速度', pressure: '相对压力', velocity: '速度', wall_pressure: '壁面压力'};
  function modelCardBody(card, manifest) {
    var h = ui().h;
    if (!card) return [ui().note('这个结果还没有模型说明。验证数字只能来自发布包记录，暂时空缺时不写。')];
    var parts = [];
    if (card.purpose) parts.push(h('p', {'class': 'mc-purpose', text: card.purpose}));
    var tr = card.training || {};
    var trainBits = [];
    if (tr.n_train) trainBits.push(tr.n_train + ' 例训练');
    if (Array.isArray(tr.cohorts) && tr.cohorts.length) trainBits.push(tr.cohorts.join('、'));
    trainBits.push(tr.centers ? (tr.centers === 1 ? '单中心' : tr.centers + ' 个中心') : '中心数未记录');
    if (tr.n_holdout) trainBits.push(tr.n_holdout + ' 例留出');
    if (tr.n_train) parts.push(h('p', {text: '训练数据：' + trainBits.join('，') + '。'}));
    if (Array.isArray(card.protocol) && card.protocol.length) parts.push(h('h4', {text: '血流条件与假设'}), h('ul', {}, card.protocol.map(function (t) { return h('li', {text: t}); })));
    var val = card.validation || {};
    var fields = val.fields || {};
    var labels = card.field_labels || {};
    var order = Object.keys(labels).filter(function (k) { return fields[k] || labels[k]; });
    Object.keys(fields).forEach(function (k) { if (order.indexOf(k) < 0) order.push(k); });
    parts.push(h('h4', {text: '与 CFD 的一致性' + (val.holdout_n ? '（' + val.holdout_n + ' 例留出）' : '')}));
    if (order.length) {
      var rows = order.map(function (fid) {
        var f = fields[fid] || {};
        var mf = fieldOf(manifest, fid);
        var tier = (card.field_tiers && card.field_tiers[fid]) || (mf && mf.tier) || 'model';
        var summary = f.summary || (f.r2_pa !== null && f.r2_pa !== undefined ? 'R² ' + ui().sig(f.r2_pa, 2) : '');
        return {name: labels[fid] || (mf && mf.short_label) || FIELD_NAMES[fid] || fid.toUpperCase(), tier: tier, summary: summary, usage: (card.usage && card.usage[fid]) || ''};
      });
      parts.push(ui().table([
        {key: 'name', label: '量'}, {key: 'tier', label: '来源', render: function (r) { return ui().tierTag(r.tier); }},
        {key: 'summary', label: '一致性'}, {key: 'usage', label: '怎么用', blank: true}], rows, {cls: 'tbl-card'}));
      parts.push(ui().note((val.r2_note || 'R² 为 1 表示和 CFD 完全一致。') + ' 派生量没有单独验证。'));
      if (val.note) parts.push(ui().note(val.note));
    } else {
      parts.push(ui().note(val.note || '发布记录里没有这个模型的验证数字。'));
    }
    if (Array.isArray(card.weaknesses) && card.weaknesses.length) parts.push(h('h4', {text: '已知薄弱处'}), h('ul', {}, card.weaknesses.map(function (t) { return h('li', {text: t}); })));
    if (Array.isArray(card.not_applicable) && card.not_applicable.length) parts.push(h('h4', {text: '不适用'}), h('ul', {}, card.not_applicable.map(function (t) { return h('li', {text: t}); })));
    return parts;
  }

  return {render: render, header: header, modelCardBody: modelCardBody,
    zonesModel: zonesModel, zoneColumns: zoneColumns, pairsModel: pairsModel, branchModel: branchModel, morphologyModel: morphologyModel,
    findingsModel: findingsModel, narrativeModel: narrativeModel, followupModel: followupModel, fieldOf: fieldOf, fieldTier: fieldTier,
    kindLabel: kindLabel, isGeometryFinding: isGeometryFinding, findingValue: findingValue, severityText: severityText, thresholdOf: thresholdOf};
});
