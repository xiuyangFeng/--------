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
  // Automatic findings plus the reviewer's manual ones (review.added); rejected ones go last (classic reports) and
  // never stand for their kind in the short list.
  function findingsModel(manifest) {
    var f = analysis(manifest).findings;
    var auto = (f && Array.isArray(f.items)) ? f.items.filter(function (it) { return it && typeof it === 'object'; }) : [];
    var review = (f && f.review && f.review.items) || {};
    var added = (f && f.review && Array.isArray(f.review.added) ? f.review.added : []).filter(function (a) { return a && a.id && Array.isArray(a.xyz_mm); });
    var manual = added.map(function (a) { return ns.review ? ns.review.manualItem(a) : Object.assign({ manual: true, kind: 'manual', label: a.text }, a); });
    var rejected = function (it) { return Boolean(review[it.id] && review[it.id].decision === 'rejected'); };
    var all = auto.concat(manual);
    var items = all.filter(function (it) { return !rejected(it); }).concat(all.filter(rejected));
    var byKind = {};
    auto.filter(function (it) { return !rejected(it); }).forEach(function (it, i) {
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
    // the reviewer's own findings follow the automatic 关注 ones, so the short list keeps them in view
    var attn = top.filter(function (it) { return it.severity === 'attention'; });
    top = attn.concat(manual.filter(function (it) { return !rejected(it); }), top.filter(function (it) { return it.severity !== 'attention'; }));
    var undecided = auto.filter(function (it) { return it.id && !(review[it.id] && review[it.id].decision); });
    return {items: items, top: top, review: review, undecided: undecided, total: items.length, manual: manual.length,
      rejected: all.filter(rejected).length, legacy: f && f.review && Array.isArray(f.review.legacy) ? f.review.legacy : []};
  }
  function isGeometryFinding(item) { return Boolean(item && (item.source === 'morphology' || item.kind === 'max_diameter' || item.kind === 'min_radius')); }
  function kindLabel(item) { return item && item.manual ? '人工' : (KIND_LABEL[item && item.kind] || (item && item.label) || '发现'); }
  function severityText(item) {
    if (!item) return null;
    if (item.severity === 'attention') return {tone: 'warn', label: '关注'};
    if (item.severity === 'note') return {tone: 'idle', label: item.grading === 'no_reference' ? '提示 · 无队列参照' : '提示'};
    return null;
  }
  function findingValue(item) {
    if (!item) return '—';
    if (item.manual) return item.text && item.text.length > 14 ? item.text.slice(0, 13) + '…' : (item.text || '');
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

  // ------------------------------------------------------------------ pure models for the 2026-09-30 look
  function statsOf(manifest, id) { var f = fieldOf(manifest, id); return (f && f.statistics) || {}; }
  // Up to four key numbers, each tied to a field (colour chip) or a fraction (bar), plus the lumen diameter.
  function kpiModel(manifest) {
    var a = analysis(manifest), out = [];
    var family = manifest && manifest.result && manifest.result.family;
    var morph = a.morphology && a.morphology.aorta && a.morphology.aorta.max;
    var diam = morph && morph.max_diameter_mm !== undefined ? {key: 'max_diameter', label: '管腔最大直径', value: morph.max_diameter_mm, units: 'mm', tier: 'geometry',
      kind: 'morph', definition: '中心线垂直截面上管腔的最长径；不含附壁血栓与管壁，通常小于 CT 报告的瘤体直径。'} : null;
    if (family === 'volume') {
      var vs = a.volume_statistics || {}, sp = vs.speed_m_s || statsOf(manifest, 'speed'), pr = vs.pressure_interior_pa || statsOf(manifest, 'pressure');
      if (sp && sp.p99 !== undefined) out.push({key: 'speed_p99', label: '体内速度 p99', value: sp.p99, units: 'm/s', tier: 'model', field: 'speed', stat: 'p99', chip: true});
      if (sp && sp.max !== undefined) out.push({key: 'speed_max', label: '最大速度', value: sp.max, units: 'm/s', tier: 'model', field: 'speed', stat: 'max', chip: true});
      if (pr && pr.min !== undefined && pr.max !== undefined) out.push({key: 'pressure_range', label: '相对压力跨度', value: pr.max - pr.min, units: 'Pa', tier: 'model', field: 'pressure', stat: 'range',
        definition: '体内最高与最低相对压力之差；压力是相对当前帧体积平均的相对压，不是血压。'});
    } else if (fieldOf(manifest, 'tawss')) {
      var t = statsOf(manifest, 'tawss'), cyc = a.cycle || {}, stag = cyc.stagnation || {};
      var tl = (t.area_frac && t.area_frac.low !== undefined) ? t.area_frac.low : (cyc.fields && cyc.fields.tawss && cyc.fields.tawss.area_frac ? cyc.fields.tawss.area_frac.low : undefined);
      if (t.mean !== undefined) out.push({key: 'tawss_mean', label: 'TAWSS 均值', value: t.mean, units: 'Pa', tier: 'model', field: 'tawss', stat: 'mean', chip: true});
      if (tl !== undefined) out.push({key: 'tawss_low', label: '低剪切占比', value: tl, pct: true, tier: 'model', field: 'tawss', stat: 'frac_low',
        definition: 'TAWSS 低于 ' + (thresholdOf(manifest, 'tawss', 'low') || 0.4) + ' Pa 的壁面占比（点占比估计）。'});
      if (stag.area_frac !== undefined) out.push({key: 'stagnation', label: '滞留区', value: stag.area_frac, pct: true, tier: 'derived', field: 'stagnation', stat: 'area_frac',
        sub: stag.area_mm2 !== undefined ? ui().num(stag.area_mm2 / 100, 'cm²') : '', definition: (stag.definition || 'TAWSS < 0.4 Pa 且 OSI > 0.1') + '；由预测的 TAWSS 与 OSI 算出，没有单独验证。'});
    } else if (fieldOf(manifest, 'wss')) {
      var w = statsOf(manifest, 'wss'), pk = a.peak || {};
      var p99 = pk.p99_pa !== undefined ? pk.p99_pa : w.p99;
      if (p99 !== undefined) out.push({key: 'wss_p99', label: 'WSS p99', value: p99, units: 'Pa', tier: 'model', field: 'wss', stat: 'p99', chip: true,
        definition: '收缩期峰值帧预测点的空间第 99 百分位；不是时间最大值。'});
      if (w.area_frac_low !== undefined) out.push({key: 'wss_low', label: '低 WSS 占比', value: w.area_frac_low, pct: true, tier: 'model', field: 'wss', stat: 'frac_low'});
      if (w.area_frac_high !== undefined) out.push({key: 'wss_high', label: '高 WSS 占比', value: w.area_frac_high, pct: true, tier: 'model', field: 'wss', stat: 'frac_high'});
    }
    if (diam) out.push(diam);
    return out.slice(0, 4);
  }
  // Metrics the zone map can colour by, per result family (first one is the default).
  function zoneMetrics(manifest) {
    if (fieldOf(manifest, 'tawss')) return [
      {id: 'tawss_mean', field: 'tawss', stat: 'mean', label: 'TAWSS 均值', units: 'Pa'},
      {id: 'tawss_low', field: 'tawss', stat: 'frac_low', label: '低剪切占比', pct: true},
      {id: 'osi_mean', field: 'osi', stat: 'mean', label: 'OSI 均值', units: '1'}];
    if (fieldOf(manifest, 'wss')) return [
      {id: 'wss_mean', field: 'wss', stat: 'mean', label: 'WSS 均值', units: 'Pa'},
      {id: 'wss_low', field: 'wss', stat: 'frac_low', label: '低 WSS 占比', pct: true},
      {id: 'wss_p99', field: 'wss', stat: 'p99', label: 'WSS p99', units: 'Pa'}];
    return [];
  }
  // Zones for the map: the analysis-layer zones, or (older results) the per-branch statistics mapped onto the
  // same ids so the picture still works; the aorta is then one piece.
  var BRANCH_ZONE = {root: 'aorta', left_cia: 'left_cia', right_cia: 'right_cia', 'out-le': 'left_eia', 'out-li': 'left_iia', 'out-re': 'right_eia', 'out-ri': 'right_iia'};
  function mapZones(manifest) {
    var z = analysis(manifest).zones;
    if (z && Array.isArray(z.zones) && z.zones.length) return {source: 'zones', zones: z.zones.filter(Boolean), pairs: z.pairs || [], definition: z.definition || ''};
    var per = analysis(manifest).per_branch, br = (manifest && manifest.geometry && manifest.geometry.branches) || [];
    if (!per || !br.length) return null;
    var zones = [];
    br.forEach(function (b) {
      var id = BRANCH_ZONE[b.key], st = per[b.name];
      if (!id || !st) return;
      zones.push({id: id, label: b.name, segment_id: b.id, fields: {wss: {mean: st.wss_mean_pa, p99: st.wss_p99_pa, frac_low: st.frac_low, frac_high: st.frac_high}}});
    });
    return zones.length ? {source: 'branches', zones: zones, pairs: [], definition: '按中心线分支的预测点等权统计（这份结果没有分区统计）。'} : null;
  }
  function zoneValue(zone, metric) { var f = zone && zone.fields && zone.fields[metric.field]; var v = f ? f[metric.stat] : undefined; return v === null || v === undefined || !Number.isFinite(Number(v)) ? null : Number(v); }
  // Sequential colour for fractions (0 → pale, 1 → deep red); the fields' own values use the vessel's scale.
  function fracColor(f) {
    var x = Math.max(0, Math.min(1, Number(f) || 0));
    var stops = [[0, [236, 240, 244]], [0.35, [252, 196, 107]], [0.7, [238, 120, 52]], [1, [185, 40, 38]]];
    for (var i = 1; i < stops.length; i++) if (x <= stops[i][0]) {
      var a = stops[i - 1], b = stops[i], t = (x - a[0]) / (b[0] - a[0] || 1);
      var c = [0, 1, 2].map(function (k) { return Math.round(a[1][k] + (b[1][k] - a[1][k]) * t); });
      return '#' + c.map(function (v) { return (v < 16 ? '0' : '') + v.toString(16); }).join('');
    }
    return '#b92826';
  }

  // Schematic of the aorto-iliac tree in the anterior view (patient right on the left of the screen).
  var SHAPES = {
    aorta_proximal: {d: 'M150 22 L150 60', w: 20, lx: 174, ly: 45},
    neck: {d: 'M150 66 L150 104', w: 16, lx: 174, ly: 89},
    sac: {ellipse: [150, 146, 34, 36], lx: 194, ly: 150},
    aorta_distal: {d: 'M150 188 L150 208', w: 18, lx: 174, ly: 202},
    aorta: {d: 'M150 22 L150 208', w: 22, lx: 178, ly: 118},
    right_cia: {d: 'M143 216 L108 252', w: 12, lx: 114, ly: 226, anchor: 'end'},
    left_cia: {d: 'M157 216 L192 252', w: 12, lx: 186, ly: 226},
    right_eia: {d: 'M103 260 L72 300', w: 9, lx: 68, ly: 318, anchor: 'end'},
    right_iia: {d: 'M111 261 L121 298', w: 7, lx: 124, ly: 318, anchor: 'middle'},
    left_eia: {d: 'M197 260 L228 300', w: 9, lx: 232, ly: 318},
    left_iia: {d: 'M189 261 L179 298', w: 7, lx: 176, ly: 318, anchor: 'middle'}
  };
  function svgNode(tag, attrs, kids) {
    var doc = root.document;
    var el = doc && typeof doc.createElementNS === 'function' ? doc.createElementNS('http://www.w3.org/2000/svg', tag) : null;
    if (!el) return null;
    Object.keys(attrs || {}).forEach(function (k) { if (attrs[k] !== null && attrs[k] !== undefined && el.setAttribute) el.setAttribute(k, String(attrs[k])); });
    (kids || []).forEach(function (c) { if (c) el.appendChild(typeof c === 'string' ? doc.createTextNode(c) : c); });
    return el;
  }
  // Built with DOM nodes (no markup parsing).  handlers: {enter(zone), leave(), click(zone)}.
  function zoneMapEl(zones, metric, colorOf, fmt, handlers) {
    var byId = {};
    zones.forEach(function (z) { byId[z.id] = z; });
    var hasSac = Boolean(byId.sac || byId.neck || byId.aorta_proximal);
    var order = (hasSac ? ['aorta_proximal', 'neck', 'sac', 'aorta_distal'] : ['aorta']).concat(['right_cia', 'left_cia', 'right_eia', 'right_iia', 'left_eia', 'left_iia']);
    var svg = svgNode('svg', {'class': 'zmap-svg', viewBox: '0 0 300 326', width: '100%', role: 'group', 'aria-label': '分区示意图（前面观）'});
    if (!svg) return null;
    order.forEach(function (id) {   // under-stroke: a thin dark edge around every piece
      var sh = SHAPES[id];
      if (!sh || !byId[id]) return;
      svg.appendChild(sh.ellipse ? svgNode('ellipse', {cx: sh.ellipse[0], cy: sh.ellipse[1], rx: sh.ellipse[2] + 1.5, ry: sh.ellipse[3] + 1.5, 'class': 'zmap-edge'})
        : svgNode('path', {d: sh.d, 'stroke-width': sh.w + 3, 'class': 'zmap-edge'}));
    });
    order.forEach(function (id) {
      var sh = SHAPES[id], z = byId[id];
      if (!sh || !z) return;
      var v = zoneValue(z, metric), col = v === null ? '#a7adb3' : (colorOf(v) || '#a7adb3');
      var title = (z.label || id) + '：' + (v === null ? '无数据' : fmt(v));
      var shape = sh.ellipse ? svgNode('ellipse', {cx: sh.ellipse[0], cy: sh.ellipse[1], rx: sh.ellipse[2], ry: sh.ellipse[3], fill: col, 'class': 'zmap-fill'})
        : svgNode('path', {d: sh.d, stroke: col, 'stroke-width': sh.w, 'class': 'zmap-stroke'});
      var g = svgNode('g', {'class': 'zmap-zone', 'data-zone': id, tabindex: 0, role: 'button', 'aria-label': title}, [svgNode('title', {}, [title]), shape,
        svgNode('text', {x: sh.lx, y: sh.ly, 'text-anchor': sh.anchor || 'start', 'class': 'zmap-val'}, [v === null ? '—' : fmt(v)])]);
      if (handlers && g.addEventListener) {
        g.addEventListener('mouseenter', function () { handlers.enter(z); });
        g.addEventListener('focus', function () { handlers.enter(z); });
        g.addEventListener('mouseleave', function () { handlers.leave(); });
        g.addEventListener('blur', function () { handlers.leave(); });
        g.addEventListener('click', function () { handlers.click(z); });
      }
      svg.appendChild(g);
    });
    svg.appendChild(svgNode('text', {x: 20, y: 18, 'class': 'zmap-side'}, ['右']));
    svg.appendChild(svgNode('text', {x: 280, y: 18, 'text-anchor': 'end', 'class': 'zmap-side'}, ['左']));
    return svg;
  }

  // ------------------------------------------------------------------ rendering
  function header(el, ctx) {
    var h = ui().h;
    var m = ctx.manifest || {};
    var job = ctx.job || m.job || {};
    var mj = m.job || {};
    var name = ctx.hideName ? '病例（名称已隐藏）' : (m.job && m.job.display_name) || ui().displayName(job);
    var meta = [];
    if (mj.scan_date || job.scan_date) meta.push((mj.scan_date || job.scan_date) + (mj.scan_label || job.scan_label ? ' ' + (mj.scan_label || job.scan_label) : ''));
    else if (mj.created_at || job.created_at) meta.push(ui().time(mj.created_at || job.created_at, false));
    var box = h('div', {'class': 'insp-id'}, h('div', {'class': 'insp-name', text: name}), meta.length ? h('div', {'class': 'insp-scan', text: meta.join(' · ')}) : null);
    var sib = ctx.offline ? [] : (ctx.siblings || []);
    var resultEl;
    if (sib.length > 1) {
      var cur = sib.filter(function (s) { return s.current; })[0];
      resultEl = ui().select(sib.map(function (s) { return {value: s.jobId, label: s.name}; }), cur ? cur.jobId : null,
        function (v) { if (ctx.onOpenResult && (!cur || v !== cur.jobId)) ctx.onOpenResult(v); }, {'aria-label': '同一输入的结果', 'class': 'result-select'});
    } else if (m.result) {
      resultEl = h('span', {'class': 'insp-result', text: m.result.display_name || ui().resultName(job, ctx.cards)});
    }
    var row = h('div', {'class': 'insp-status'});
    if (resultEl) row.appendChild(resultEl);
    row.appendChild(h('span', {'class': 'sec-fill'}));
    if (ctx.offline) {
      row.appendChild(ui().dot(ui().reviewInfo(mj.review).tone, '导出时' + ui().reviewInfo(mj.review).label));
    } else {
      var rv = ui().reviewInfo(job.review || mj.review);
      var t = Number(ctx.computeSeconds);
      var st = ui().statusDot(job.status || mj.status || 'done');
      if (Number.isFinite(t) && t > 0) st.title = '计算 ' + ui().duration(t) + '（不含排队与人工确认）';
      row.appendChild(st);
      if (ctx.onReview) row.appendChild(h('button', {type: 'button', 'class': 'chip-btn' + (rv.key === 'reviewed' ? ' is-done' : ''), text: rv.key === 'reviewed' ? '已复核' : '技术复核', onclick: ctx.onReview,
        title: rv.key === 'reviewed' ? '查看复核记录' : '核对输入、出口命名和质量后签字'}));
      else row.appendChild(ui().reviewDot(job.review || mj.review));
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
    // key numbers
    var kpis = kpiModel(m);
    if (kpis.length) {
      var grid = h('div', {'class': 'kpis'});
      kpis.forEach(function (k) {
        var valText = k.pct ? ui().pct(k.value) : ui().num(k.value, null);
        var unit = k.pct ? '' : ui().unitText ? ui().unitText(k.units) : (k.units || '');
        var visual = null;
        if (k.pct) visual = h('span', {'class': 'kpi-bar'}, h('span', {'class': 'kpi-bar-fill', style: 'width:' + Math.max(0, Math.min(100, k.value * 100)).toFixed(1) + '%;background:' + fracColor(k.value)}));
        else if (k.chip && ctx.fieldScale) { var fs = ctx.fieldScale(k.field); var c = fs && fs.color(k.value); if (c) visual = h('span', {'class': 'kpi-chip', style: 'background:' + c, title: '在当前色标上的颜色'}); }
        var tile = h('button', {type: 'button', 'class': 'kpi kpi-' + k.tier, title: ui().tierHelp ? ui().tierHelp(k.tier) : ''},
          h('span', {'class': 'kpi-label', text: k.label}),
          h('span', {'class': 'kpi-value'}, visual && !k.pct ? visual : null, h('span', {'class': 'kpi-num', text: valText}), unit ? h('span', {'class': 'kpi-unit', text: unit}) : null),
          k.pct ? visual : (k.sub ? h('span', {'class': 'kpi-sub', text: k.sub}) : null),
          k.pct && k.sub ? h('span', {'class': 'kpi-sub', text: k.sub}) : null);
        if (lens) tile.addEventListener('click', function () {
          if (k.kind === 'morph') lens({kind: 'morph', key: k.key, label: k.label, value: k.value, units: k.units, definition: k.definition});
          else lens({kind: 'stat', field: k.field, stat: k.stat, value: k.value, units: k.pct ? null : k.units, pct: Boolean(k.pct), label: k.label, text: k.pct ? ui().pct(k.value) : null, definition: k.definition});
        });
        grid.appendChild(tile);
      });
      out.push(h('section', {'class': 'sec sec-kpis'}, grid));
    }
    // zone map
    var zs = mapZones(m), metrics = zoneMetrics(m);
    if (zs && metrics.length) {
      var metric = metrics.filter(function (x) { return x.id === ctx.zoneMetric; })[0] || metrics[0];
      if (!zs.zones.some(function (z) { return zoneValue(z, metric) !== null; })) metric = metrics.filter(function (x) { return zs.zones.some(function (z) { return zoneValue(z, x) !== null; }); })[0] || metric;
      var fsz = metric.pct ? null : (ctx.fieldScale ? ctx.fieldScale(metric.field) : null);
      var colorOf = metric.pct ? fracColor : function (v) { return fsz ? fsz.color(v) : '#8a94a3'; };
      var fmt = metric.pct ? function (v) { return ui().pct(v); } : function (v) { return ui().sig(v); };
      var pills = h('div', {'class': 'pills', role: 'tablist', 'aria-label': '分区着色'});
      metrics.forEach(function (x) {
        var b = h('button', {type: 'button', role: 'tab', 'class': 'pill' + (x.id === metric.id ? ' on' : ''), 'aria-selected': String(x.id === metric.id), text: x.label});
        b.addEventListener('click', function () { if (ctx.onZoneMetric) ctx.onZoneMetric(x.id); });
        pills.appendChild(b);
      });
      var mapBox = h('div', {'class': 'zmap'});
      var svgEl = zoneMapEl(zs.zones, metric, colorOf, fmt, {
        enter: function (z) { if (ctx.onZone) ctx.onZone(z); },
        leave: function () { if (ctx.onZone) ctx.onZone(null); },
        click: function (z) {
          var v = zoneValue(z, metric);
          if (lens && v !== null) lens({kind: 'zone', zone: z, field: metric.field, stat: metric.stat, value: v, units: metric.pct ? null : metric.units, pct: Boolean(metric.pct), label: metric.label,
            threshold: metric.stat === 'frac_low' ? thresholdOf(m, metric.field, 'low') : null});
        }
      });
      if (svgEl) mapBox.appendChild(svgEl);
      var unitNote = metric.pct ? '' : ui().unitText ? ui().unitText(metric.units) : '';
      var tipText = (zs.definition || '') + '悬停一个分区，血管上会标出它；点一下看这个数的来源。前面观：屏幕左侧是患者右侧。';
      var morph = morphLine(m);
      var tableToggle = ui().button(ctx.zoneTable ? '收起表格' : '表格', function () { if (ctx.onZoneTable) ctx.onZoneTable(); }, {kind: 'link', cls: 'btn-sm'});
      var head = h('div', {'class': 'sec-head'}, h('h3', {'class': 'sec-title', text: '分区'}), unitNote ? h('span', {'class': 'sec-unit', text: unitNote}) : null, ui().infoTip(tipText), h('span', {'class': 'sec-fill'}), tableToggle);
      var sec = h('section', {'class': 'sec sec-zmap'}, head, pills, mapBox, morph ? h('div', {'class': 'zmap-morph', text: morph}) : null);
      if (ctx.zoneTable) { var zt = zoneTable(m, lens); if (zt) sec.appendChild(zt); }
      out.push(sec);
    } else if (m.result && m.result.family === 'volume') {
      var mo = morphLine(m);
      if (mo) out.push(h('section', {'class': 'sec'}, h('div', {'class': 'sec-head'}, h('h3', {'class': 'sec-title', text: '形态'}), ui().infoTip('直径、长度和体积都是管腔的，不含附壁血栓与管壁。')), h('div', {'class': 'zmap-morph', text: mo})));
    }
    // findings: compact, one per kind
    var fm = findingsModel(m);
    if (fm.total) {
      var expanded = Boolean(ctx.findingsExpanded);
      var list = expanded ? fm.items : fm.top.slice(0, 5);
      var rows = h('div', {'class': 'findings'});
      list.forEach(function (it) {
        var sev = severityText(it);
        var decision = fm.review[it.id] && fm.review[it.id].decision;
        var tone = sev ? sev.tone : 'idle';
        var row = h('button', {type: 'button', 'class': 'finding tone-' + tone + (ctx.selectedFinding === it.id ? ' on' : '') + (decision ? ' dec-' + decision : '') + (it.manual ? ' manual' : ''), dataset: {findingId: it.id || ''},
          title: [sev ? sev.label : (isGeometryFinding(it) ? '几何' : ''), it.contains_global_max ? '含全场最大值' : '', decision === 'confirmed' ? '已确认' : decision === 'rejected' ? '已驳回' : '', it.definition || ''].filter(Boolean).join(' · ')},
          h('span', {'class': 'f-mark'}),
          h('span', {'class': 'f-kind', text: kindLabel(it)}),
          h('span', {'class': 'f-where', text: it.branch || ''}),
          h('span', {'class': 'f-val', text: findingValue(it)}),
          decision === 'confirmed' ? h('span', {'class': 'f-dec', title: '已确认'}, ui().icon('check', {size: 14})) : decision === 'rejected' ? h('span', {'class': 'f-dec', title: '已驳回'}, ui().icon('close', {size: 14})) : h('span', {'class': 'f-dec'}));
        row.addEventListener('click', function () { if (ctx.onFinding) ctx.onFinding(it); });
        rows.appendChild(row);
      });
      var more = fm.total > list.length || expanded ? ui().button(expanded ? '收起' : '全部 ' + fm.total, function () { if (ctx.onToggleFindings) ctx.onToggleFindings(); }, {kind: 'link', cls: 'btn-sm'}) : null;
      var confirmRest = null;
      if (ctx.onConfirmRest && fm.undecided.length && ctx.tier === 'full') confirmRest = ui().button((fm.undecided.length === fm.total ? '全部 ' : '其余 ') + fm.undecided.length + ' 条按自动结果确认', function () { ctx.onConfirmRest(fm); }, {kind: 'link', cls: 'btn-sm', disabled: Boolean(ctx.locked)});
      var addBtn = ctx.onAddFinding ? ui().iconButton('plus', ctx.locked ? '已复核锁定，不能新增' : '新增人工发现：在管壁上点一处', ctx.onAddFinding, {cls: 'btn-xs'}) : null;
      if (addBtn && ctx.locked) addBtn.disabled = true;
      var fhead = h('div', {'class': 'sec-head'}, h('h3', {'class': 'sec-title', text: '发现'}), h('span', {'class': 'sec-count', text: String(fm.total)}),
        ui().infoTip('自动发现只是候选位置；「关注」「提示」不是临床分级。橙色点 = 关注，空心点 = 提示。点一条可以确认、驳回或写备注；驳回的排到最后。'), h('span', {'class': 'sec-fill'}), addBtn, more);
      var legacy = fm.legacy.length && ctx.onLegacy ? ui().button(fm.legacy.length + ' 条规则更新前的判定', function () { ctx.onLegacy(fm.legacy); }, {kind: 'link', cls: 'btn-sm'}) : null;
      out.push(h('section', {'class': 'sec sec-findings'}, fhead, rows, confirmRest || legacy ? h('div', {'class': 'sec-actions'}, confirmRest, legacy) : null));
    } else if (ctx.onAddFinding && !ctx.locked) {
      out.push(h('section', {'class': 'sec sec-findings'}, h('div', {'class': 'sec-head'}, h('h3', {'class': 'sec-title', text: '发现'}), h('span', {'class': 'sec-fill'}),
        ui().iconButton('plus', '新增人工发现：在管壁上点一处', ctx.onAddFinding, {cls: 'btn-xs'})), ui().note('没有自动发现。')));
    }
    // conclusion: collapsed to three lines
    var nar = narrativeModel(m);
    if (nar) {
      var open = Boolean(ctx.narrativeOpen);
      var edit = ctx.onEditNarrative ? ui().button('编辑', ctx.onEditNarrative, {kind: 'link', cls: 'btn-sm'}) : null;
      var toggle = ui().button(open ? '收起' : '展开', function () { if (ctx.onToggleNarrative) ctx.onToggleNarrative(); }, {kind: 'link', cls: 'btn-sm'});
      var nhead = h('div', {'class': 'sec-head'}, h('h3', {'class': 'sec-title', text: '结论'}), nar.edited ? h('span', {'class': 'sec-count', text: '人工编辑'}) : null,
        ui().infoTip(nar.edited ? '人工编辑的结论' + (nar.by ? '（' + nar.by + '）' : '') : '根据形态和统计量自动生成，供书写报告参考。'), h('span', {'class': 'sec-fill'}), edit, toggle);
      out.push(h('section', {'class': 'sec sec-conclusion' + (open ? ' open' : '')}, nhead, h('div', {'class': 'narr'}, nar.sentences.map(function (t) { return h('p', {text: t}); }))));
    }
    // follow-up (U17)
    var fu = followupModel(ctx.timeline, m.result && m.result.release_id);
    if (fu) {
      var fcols = [{key: 'date', label: '扫描', render: function (r) { return r.date || r.label; }},
        {key: 'd', label: '管腔最大直径 mm', num: true, render: function (r) { return ui().num(r.diameter, null); }}];
      if (fu.metricLabel) fcols.push({key: 'm', label: fu.metricLabel + (fu.metricUnits && fu.metricUnits !== '1' ? ' ' + fu.metricUnits : ''), num: true, render: function (r) { return ui().num(r.metric, null); }});
      var g = fu.growth && fu.growth.per_year !== undefined ? h('div', {'class': 'fu-growth'}, h('span', {'class': 'kpi-num', text: (fu.growth.per_year > 0 ? '+' : '') + ui().sig(fu.growth.per_year)}), h('span', {'class': 'kpi-unit', text: 'mm/年'}), h('span', {'class': 'muted', text: '管腔最大直径'})) : null;
      out.push(h('section', {'class': 'sec sec-follow'}, h('div', {'class': 'sec-head'}, h('h3', {'class': 'sec-title', text: '随访'}), h('span', {'class': 'sec-count', text: fu.rows.length + ' 次扫描'}), ui().infoTip((fu.notes[0] || '') + ' 年增长率按首末两次扫描计算。')),
        g, ui().table(fcols, fu.rows, {cls: 'tbl-follow'})));
    }
    // model + the only research-use line
    var card = m.model_card || (ctx.cards && m.result && ctx.cards[m.result.release_id]) || null;
    out.push(h('footer', {'class': 'insp-foot'},
      h('span', {text: ((card && card.display_name) || (m.result && m.result.display_name) || '模型') + (card && card.version_date ? ' · ' + card.version_date : '')}),
      ctx.onModelCard ? ui().button('模型说明', ctx.onModelCard, {kind: 'link', cls: 'btn-sm'}) : null,
      h('span', {'class': 'sec-fill'}), h('span', {'class': 'research', text: '研究用途，非诊断'})));
    ui().fill(el, out);
    return {findings: fm, zones: zonesModel(m), kpis: kpis};
  }
  function morphLine(m) {
    var a = analysis(m).morphology && analysis(m).morphology.aorta;
    if (!a) return '';
    var bits = [];
    var r0 = function (v) { return Number.isFinite(Number(v)) ? String(Math.round(Number(v))) : '—'; };
    if (a.sac && a.sac.present) bits.push('瘤体 ' + r0(a.sac.length_mm) + ' mm · ' + r0(a.sac.volume_ml) + ' mL');
    else if (a.sac) bits.push('管腔未见瘤样扩张');
    if (a.neck && a.neck.present) bits.push('瘤颈 ' + r0(a.neck.length_mm) + ' mm · Ø' + r0(a.neck.diameter_mean_mm) + ' mm');
    return bits.join('　');
  }
  // The full zone table (and left / right pairs) behind the 「表格」 link.
  function zoneTable(m, lens) {
    var h = ui().h;
    var zm = zonesModel(m);
    if (!zm) return null;
    var cols = [{key: 'label', label: '分区', render: function (r) { return r.zone.label || r.zone.id; }}].concat(zm.columns.map(function (c, i) {
      return {key: 'c' + i, label: c.label + (c.units && !c.pct && c.units !== '1' ? ' ' + c.units : ''), num: true, render: function (r) {
        var cell = r.cells[i];
        var text = c.pct ? ui().pct(cell.value) : ui().num(cell.value, null);
        return ui().evidence(text, lens && cell.value !== undefined && cell.value !== null ? function () { lens({kind: 'zone', zone: r.zone, field: c.field, stat: c.stat, value: cell.value, units: c.pct ? null : c.units, pct: Boolean(c.pct), label: c.label}); } : null);
      }};
    }));
    var kids = [ui().table(cols, zm.rows, {cls: 'tbl-zones'})];
    if (zm.pairs && zm.pairs.rows.length) {
      var pf = zm.pairs.field;
      kids.push(ui().table([
        {key: 'label', label: '左右对比', render: function (r) { return r.pair.label || r.pair.id; }},
        {key: 'l', label: '左', num: true, render: function (r) { return ui().num(r.left, null); }},
        {key: 'r', label: '右', num: true, render: function (r) { return ui().num(r.right, null); }},
        {key: 'q', label: '左/右', num: true, render: function (r) { return r.ratio === null || r.ratio === undefined ? '—' : ui().sig(r.ratio); }}], zm.pairs.rows, {cls: 'tbl-pairs'}));
    }
    return h('div', {'class': 'zmap-table'}, kids);
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

  return {render: render, header: header, modelCardBody: modelCardBody, kpiModel: kpiModel, zoneMetrics: zoneMetrics, mapZones: mapZones, zoneMapEl: zoneMapEl, fracColor: fracColor, morphLine: morphLine,
    zonesModel: zonesModel, zoneColumns: zoneColumns, pairsModel: pairsModel, branchModel: branchModel, morphologyModel: morphologyModel,
    findingsModel: findingsModel, narrativeModel: narrativeModel, followupModel: followupModel, fieldOf: fieldOf, fieldTier: fieldTier,
    kindLabel: kindLabel, isGeometryFinding: isGeometryFinding, findingValue: findingValue, severityText: severityText, thresholdOf: thresholdOf};
});
