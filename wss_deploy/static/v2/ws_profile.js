/* WSS workspace v2 — along-branch profile curves (second phase lane D, S6d).
 * The 2 mm bins of manifest.analysis.profiles (the statistics the cursor reads) drawn against the arc length from
 * the inlet, for the coloured field: WSS p99 / mean, TAWSS mean / minimum, OSI · RRT · ECAP p90 / mean, speed mean /
 * maximum, pressure mean / minimum — the series and thresholds of the classic reports (report.py profSpec, the volume
 * report's profile).  The radius strip is the bins' centreline radius.  SVG export goes through
 * WssReportCommon.profileSVG with the classic profileSVGs series (field + radius / lumen diameters).
 * Charts are built with createElementNS (no markup parsing); pure parts are Node-testable. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.profile = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var SVGNS = 'http://www.w3.org/2000/svg';
  var common = function () { return root.WssReportCommon || null; };

  function profiles(m) { var p = m && m.analysis && m.analysis.profiles; return p && Array.isArray(p.branches) ? p : null; }
  function branchList(m) { var p = profiles(m); return p ? p.branches.filter(function (b) { return b && Array.isArray(b.s_from_root_mm); }) : []; }
  function branchOf(m, segmentId) { return branchList(m).filter(function (b) { return Number(b.segment_id) === Number(segmentId); })[0] || null; }
  function has(m, key) { return branchList(m).some(function (b) { return b && b[key] && typeof b[key] === 'object'; }); }
  function arr(a) { return Array.isArray(a) ? a.map(function (v) { return v === null || v === undefined ? NaN : Number(v); }) : []; }
  function thresholds(m, id) {
    var f = ((m && m.fields) || []).filter(function (x) { return x && x.id === id; })[0];
    var t = f && f.display && f.display.thresholds;
    return Array.isArray(t) && t.length ? t.map(Number) : null;
  }
  function stagCriteria(m) {
    var c = (m && m.analysis && m.analysis.cycle && m.analysis.cycle.stagnation && m.analysis.cycle.stagnation.criteria) || {};
    return {t: Number.isFinite(+c.tawss_lt_pa) ? +c.tawss_lt_pa : 0.4, o: Number.isFinite(+c.osi_gt) ? +c.osi_gt : 0.1};
  }
  function wssTitle() { var c = common(); return (c && c.englishLabel ? c.englishLabel('wss_short', 'zh') : 'WSS') + ' · Pa'; }

  // What the curve shows for the coloured field: {id, label, units, primary, secondary, band, hlines, missing}.
  // primary / secondary: {key, name, get(branch) → array}; the band spans the two.
  function spec(m, fieldId) {
    var sp = rawSpec(m, fieldId), cv = ns.display && ns.display.toDisplay;
    if (!sp || !cv || !(m && m.result && m.result.family === 'volume')) return sp;
    // volume curves in the display unit of the colour-scale panel (ws_display: Pa / mmHg, m/s / cm/s)
    var k = cv(1, sp.units, 'volume');
    if (k.units === sp.units) return sp;
    var scaled = function (s) { return {key: s.key, name: s.name, get: function (b) { return s.get(b).map(function (v) { return v * k.value; }); }}; };
    return Object.assign({}, sp, {units: k.units, primary: scaled(sp.primary), secondary: scaled(sp.secondary)});
  }
  function rawSpec(m, fieldId) {
    var volume = m && m.result && m.result.family === 'volume';
    if (volume) {
      var q = fieldId === 'pressure' || fieldId === 'wall_pressure' ? 'pressure' : 'speed';
      var v = function (b) { return (b && b.volume) || {}; };
      if (q === 'pressure' && has(m, 'volume') && branchList(m).some(function (b) { return v(b).pressure_mean_pa; })) {
        return {id: 'pressure', label: '相对压力', units: 'Pa', primary: {key: 'mean', name: '截面平均', get: function (b) { return arr(v(b).pressure_mean_pa); }},
          secondary: {key: 'min', name: '最低', get: function (b) { return arr(v(b).pressure_min_pa); }}, hlines: [], note: '相对压力，只用于比较差值'};
      }
      if (has(m, 'volume')) return {id: 'speed', label: '速度', units: 'm/s', primary: {key: 'mean', name: '截面平均', get: function (b) { return arr(v(b).speed_mean_m_s); }},
        secondary: {key: 'max', name: '最大', get: function (b) { return arr(v(b).speed_max_m_s); }}, hlines: []};
      return null;
    }
    var g = function (id) { return function (b) { return (b && b[id]) || {}; }; };
    var st = stagCriteria(m);
    if (fieldId === 'tawss' && has(m, 'tawss')) return {id: 'tawss', label: 'TAWSS', units: 'Pa',
      primary: {key: 'mean', name: '均值', get: function (b) { return arr(g('tawss')(b).mean_pa); }}, secondary: {key: 'min', name: '最低', get: function (b) { return arr(g('tawss')(b).min_pa); }},
      hlines: [{v: st.t, name: '低 TAWSS'}]};
    if (fieldId === 'osi' && has(m, 'osi')) return {id: 'osi', label: 'OSI', units: '1',
      primary: {key: 'p90', name: 'p90', get: function (b) { return arr(g('osi')(b).p90); }}, secondary: {key: 'mean', name: '均值', get: function (b) { return arr(g('osi')(b).mean); }},
      hlines: [{v: st.o, name: 'OSI'}]};
    if ((fieldId === 'rrt' || fieldId === 'ecap') && has(m, fieldId)) {
      var t = thresholds(m, fieldId), t0 = t ? t[0] : (fieldId === 'rrt' ? 5 : 1.4), sh = fieldId.toUpperCase();
      return {id: fieldId, label: sh, units: '1/Pa', primary: {key: 'p90', name: 'p90', get: function (b) { return arr(g(fieldId)(b).p90); }},
        secondary: {key: 'mean', name: '均值', get: function (b) { return arr(g(fieldId)(b).mean); }}, hlines: [{v: t0, name: sh}]};
    }
    if (!has(m, 'wss')) return null;
    var missing = fieldId && fieldId !== 'wss' ? fieldId : null;
    return {id: 'wss', label: 'WSS', units: 'Pa', primary: {key: 'p99', name: 'p99', get: function (b) { return arr(g('wss')(b).p99_pa); }},
      secondary: {key: 'mean', name: '均值', get: function (b) { return arr(g('wss')(b).mean_pa); }}, hlines: [], missing: missing};
  }

  // Drawn data of one branch: x = arc length from the inlet; bins without points (n = 0) are gaps.
  function model(m, fieldId, segmentId, xRange) {
    var b = branchOf(m, segmentId), sp = spec(m, fieldId);
    if (!b || !sp) return null;
    var x = arr(b.s_from_root_mm), xl = arr(b.s_local_mm), n = arr((b.wss && b.wss.n) || (b.volume && b.volume.n) || []);
    var p = sp.primary.get(b), s = sp.secondary.get(b), r = arr(b.radius_mm);
    var keep = function (i) { return !n.length || n[i] > 0; };
    for (var i = 0; i < x.length; i++) if (!keep(i)) { p[i] = NaN; s[i] = NaN; }
    var lo = Infinity, hi = -Infinity;
    x.forEach(function (v) { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
    if (xRange && Number.isFinite(xRange[0]) && Number.isFinite(xRange[1]) && xRange[1] > xRange[0]) { lo = xRange[0]; hi = xRange[1]; }
    return {branch: b, segmentId: Number(b.segment_id), name: b.name || ('分支 ' + b.segment_id), spec: sp, x: x, xLocal: xl, n: n, primary: p, secondary: s, radius: r,
      xRange: [lo, hi], binMm: (profiles(m) && +profiles(m).bin_mm) || 2, definition: (profiles(m) && profiles(m).definition) || ''};
  }
  // Nearest bin to an arc length (inside the drawn range); −1 when the branch has none.
  function binAt(md, s) {
    var best = -1, bd = Infinity;
    for (var i = 0; i < md.x.length; i++) { var d = Math.abs(md.x[i] - s); if (Number.isFinite(d) && d < bd) { bd = d; best = i; } }
    return bd <= md.binMm ? best : -1;
  }
  function yRange(values, hlines, zero) {
    var lo = Infinity, hi = -Infinity;
    values.forEach(function (list) { list.forEach(function (v) { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }); });
    (hlines || []).forEach(function (h) { if (Number.isFinite(+h.v)) { hi = Math.max(hi, +h.v * 1.08); lo = Math.min(lo, +h.v); } });
    if (!Number.isFinite(lo)) return [0, 1];
    if (zero !== false && lo > 0) lo = 0;
    if (!(hi > lo)) hi = lo + (Math.abs(lo) || 1);
    return [lo, hi + (hi - lo) * 0.04];
  }
  function niceTicks(lo, hi, n) {
    var span = hi - lo;
    if (!(span > 0)) return [lo];
    var raw = span / Math.max(1, n || 4), mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    var step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * mag, out = [];
    for (var v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(10));
    return out;
  }
  function tickText(v) { var a = Math.abs(v); return a >= 100 ? v.toFixed(0) : a >= 10 ? String(+v.toFixed(1)) : String(+v.toFixed(2)); }

  // Morphology stations of the branch sampled onto the bins (classic computeStationSeries): max / equivalent lumen diameter.
  function stationSeries(m, b, which) {
    var mo = m && m.analysis && m.analysis.morphology;
    var list = mo && Array.isArray(mo.branches) ? mo.branches : [];
    var mb = list.filter(function (x) { return x && Number(x.segment_id) === Number(b && b.segment_id); })[0];
    var st = mb && mb.stations;
    if (!st || !Array.isArray(st.s_from_root_mm) || !st.s_from_root_mm.length) return null;
    var S = st.s_from_root_mm.map(Number), vals = arr(which === 'max' ? st.max_diameter_mm : st.equivalent_diameter_mm);
    if (!vals.length) return null;
    var bin = Math.max(Number((profiles(m) || {}).bin_mm) || 2, 1);
    var xs = arr(b.s_from_root_mm || b.s_local_mm), out = xs.map(function () { return NaN; }), any = false;
    xs.forEach(function (x, i) {
      if (!Number.isFinite(x)) return;
      var bj = -1, bd = Infinity;
      for (var j = 0; j < S.length; j++) { var d = Math.abs(S[j] - x); if (d < bd) { bd = d; bj = j; } }
      if (bj >= 0 && bd <= bin && Number.isFinite(vals[bj])) { out[i] = vals[bj]; any = true; }
    });
    return any ? out : null;
  }

  // Classic profileSVGs (report.py): [{key, filename, svg}] for the field and the radius / diameters of one branch.
  function exportSVGs(m, fieldId, segmentId, o) {
    o = o || {};
    var c = common(), b = branchOf(m, segmentId), sp = spec(m, fieldId);
    if (!c || typeof c.profileSVG !== 'function' || !b || !sp) return [];
    var caseName = o.caseName || 'case', name = b.name || ('分支 ' + b.segment_id), xs = arr(b.s_from_root_mm);
    var xf = xs.filter(Number.isFinite), series = [];
    var hline = function (label, th, color) { if (xf.length > 1) series.push({name: label, x: [Math.min.apply(null, xf), Math.max.apply(null, xf)], y: [th, th], color: color, dash: '5 5', width: 1.2}); };
    var fmt1 = function (v) { return Number(v).toFixed(1); }, trim3 = function (v) { return String(+(+v).toFixed(3)); };
    if (sp.id === 'wss') {
      var w = b.wss || {};
      series.push({name: '均值', x: xs, y: arr(w.mean_pa), color: '#176caa'}, {name: 'p99', x: xs, y: arr(w.p99_pa), color: '#c0392b'}, {name: '最低', x: xs, y: arr(w.min_pa), color: '#3f8f6b', dash: '7 5'});
      (thresholds(m, 'wss') || [0.4, 4, 7]).slice(0, 3).forEach(function (th, i) { hline(['低 ', '高 ', '极高 '][i] + fmt1(th) + ' Pa', th, ['#8aa0b5', '#d9963c', '#b03a2e'][i]); });
    } else if (sp.id === 'tawss') {
      var t = b.tawss || {};
      series.push({name: '均值', x: xs, y: arr(t.mean_pa), color: '#176caa'}, {name: '最低', x: xs, y: arr(t.min_pa), color: '#3f8f6b', dash: '7 5'});
      hline('低 TAWSS ' + trim3(sp.hlines[0].v) + ' Pa', sp.hlines[0].v, '#8aa0b5');
    } else if (sp.id === 'speed' || sp.id === 'pressure') {
      series.push({name: sp.primary.name, x: xs, y: sp.primary.get(b), color: '#176caa'}, {name: sp.secondary.name, x: xs, y: sp.secondary.get(b), color: '#c0392b', dash: '7 5'});
    } else {
      var d = b[sp.id] || {};
      series.push({name: '均值', x: xs, y: arr(d.mean), color: '#176caa'}, {name: 'p90', x: xs, y: arr(d.p90), color: '#c0392b'});
      hline(sp.label + ' ' + trim3(sp.hlines[0].v) + (sp.units === '1/Pa' ? ' Pa⁻¹' : ''), sp.hlines[0].v, '#8aa0b5');
    }
    var kept = series.filter(function (s) { return s.y.some(Number.isFinite); });
    var yTitle = sp.id === 'wss' ? wssTitle() : sp.id === 'osi' ? 'OSI' : sp.units === '1/Pa' ? sp.label + ' · Pa⁻¹' : sp.label + ' · ' + sp.units;
    var xLabel = '距入口弧长 (mm)', safe = c.safeName || function (x) { return String(x); };
    var stem = safe(caseName) + '_profile_' + safe(name);
    var out = [{key: sp.id, branch: name, filename: stem + '_' + sp.id + '.svg', svg: c.profileSVG({series: kept, xLabel: xLabel, yLabel: yTitle, title: caseName + ' · ' + name + ' · ' + yTitle})}];
    var radius = arr(b.radius_mm), rs = [];
    if (radius.some(Number.isFinite)) rs.push({name: '半径', x: xs, y: radius, color: '#5a4fa3'});
    var dmax = stationSeries(m, b, 'max'), deq = stationSeries(m, b, 'eq');
    if (dmax) rs.push({name: '管腔最大直径', x: xs, y: dmax, color: '#8e44ad', dash: '6 5'});
    if (deq) rs.push({name: '等效直径', x: xs, y: deq, color: '#c39bd3', dash: '2 4'});
    var rTitle = dmax || deq ? '半径 / 直径' : '半径';
    if (rs.length) out.push({key: 'radius', branch: name, filename: stem + '_radius.svg', svg: c.profileSVG({series: rs, xLabel: xLabel, yLabel: rTitle + ' (mm)', title: caseName + ' · ' + name + ' · ' + rTitle})});
    return out;
  }

  // ------------------------------------------------------------------ interactive chart (DOM, createElementNS)
  function el(tag, attrs, kids) {
    var doc = root.document, e = doc && typeof doc.createElementNS === 'function' ? doc.createElementNS(SVGNS, tag) : null;
    if (!e) return null;
    Object.keys(attrs || {}).forEach(function (k) { if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, String(attrs[k])); });
    (kids || []).forEach(function (k) { if (k) e.appendChild(typeof k === 'string' ? doc.createTextNode(k) : k); });
    return e;
  }
  function pathOf(xs, ys, X, Y) {
    var d = '', pen = false;
    for (var i = 0; i < xs.length; i++) {
      var x = xs[i], y = ys[i];
      if (!Number.isFinite(x) || !Number.isFinite(y)) { pen = false; continue; }
      d += (pen ? 'L' : 'M') + X(x).toFixed(1) + ' ' + Y(y).toFixed(1);
      pen = true;
    }
    return d;
  }
  function bandOf(xs, a, b, X, Y) {
    var d = '', run = [];
    var flush = function () {
      if (run.length > 1) {
        d += 'M' + run.map(function (i) { return X(xs[i]).toFixed(1) + ' ' + Y(a[i]).toFixed(1); }).join('L');
        d += 'L' + run.slice().reverse().map(function (i) { return X(xs[i]).toFixed(1) + ' ' + Y(b[i]).toFixed(1); }).join('L') + 'Z';
      }
      run = [];
    };
    for (var i = 0; i < xs.length; i++) { if (Number.isFinite(xs[i]) && Number.isFinite(a[i]) && Number.isFinite(b[i])) run.push(i); else flush(); }
    flush();
    return d;
  }
  // o = {width, height, xRange, x, lines:[{y, cls, dash}], band:[a, b], hlines:[{v, name}], yTicks, zero, yFmt, cls}
  // → {el, X, Y, invX, setMark(name, s|null)}
  function chart(o) {
    var W = Math.max(60, o.width || 288), H = Math.max(30, o.height || 120), top = o.top === undefined ? 6 : o.top, bottom = o.bottom === undefined ? 4 : o.bottom;
    var xr = o.xRange, yr = o.yRange || yRange(o.lines.map(function (l) { return l.y; }), o.hlines, o.zero);
    var X = function (v) { return (v - xr[0]) / ((xr[1] - xr[0]) || 1) * W; };
    var Y = function (v) { return top + (1 - (v - yr[0]) / ((yr[1] - yr[0]) || 1)) * (H - top - bottom); };
    var svg = el('svg', {'class': 'along-svg ' + (o.cls || ''), width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': o.label || ''});
    if (!svg) return null;
    var ticks = niceTicks(yr[0], yr[1], o.yTicks || 3);
    ticks.forEach(function (t) {
      var y = Y(t);
      if (y < top - 0.5 || y > H - bottom + 0.5) return;
      svg.appendChild(el('line', {x1: 0, x2: W, y1: y.toFixed(1), y2: y.toFixed(1), 'class': 'along-grid'}));
      svg.appendChild(el('text', {x: -6, y: (y + 4).toFixed(1), 'text-anchor': 'end', 'class': 'along-tick'}, [(o.yFmt || tickText)(t)]));
    });
    if (o.band) { var bd = bandOf(o.x, o.band[0], o.band[1], X, Y); if (bd) svg.appendChild(el('path', {d: bd, 'class': 'along-band'})); }
    (o.hlines || []).forEach(function (hl) {
      if (!Number.isFinite(+hl.v)) return;
      var y = Y(+hl.v);
      svg.appendChild(el('line', {x1: 0, x2: W, y1: y.toFixed(1), y2: y.toFixed(1), 'class': 'along-thr'}, [el('title', {}, [(hl.name || '') + ' ' + tickText(+hl.v)])]));
    });
    o.lines.forEach(function (l) { var d = pathOf(o.x, l.y, X, Y); if (d) svg.appendChild(el('path', {d: d, 'class': 'along-line ' + (l.cls || '')})); });
    var marks = {};
    var mark = function (name) {
      if (!marks[name]) { marks[name] = el('line', {x1: 0, x2: 0, y1: 0, y2: H, 'class': 'along-mark along-mark-' + name, visibility: 'hidden'}); svg.appendChild(marks[name]); }
      return marks[name];
    };
    return {el: svg, X: X, Y: Y, W: W, H: H, yRange: yr,
      invX: function (px) { return xr[0] + Math.max(0, Math.min(1, px / W)) * (xr[1] - xr[0]); },
      setMark: function (name, s) {
        var ln = mark(name);
        if (s === null || s === undefined || !Number.isFinite(+s) || +s < xr[0] - 1e-6 || +s > xr[1] + 1e-6) { ln.setAttribute('visibility', 'hidden'); return; }
        var x = Math.max(0.5, Math.min(W - 0.5, X(+s))).toFixed(1);
        ln.setAttribute('x1', x); ln.setAttribute('x2', x); ln.setAttribute('visibility', 'visible');
      }};
  }

  return {spec: spec, model: model, binAt: binAt, branchList: branchList, branchOf: branchOf, stationSeries: stationSeries, exportSVGs: exportSVGs,
    chart: chart, yRange: yRange, niceTicks: niceTicks, tickText: tickText, stagCriteria: stagCriteria};
});
