/* WSSV2 core · along-vessel cursor (contract §6.2; discussion D1).
   Stations are (branch id, arc length inside that branch in mm) on the centreline arrays of the manifest
   (cv / cs / ce / cr): each branch ordered proximal → distal by its edges, arc length from the branch start.
   step() walks along the current branch and stops at its ends (it never jumps branches silently): at the
   distal end it reports the child branches, at the proximal end the parent.
   Arc length unit: the branch lengths of the manifest (geometry.branches / profiles, measured on the full
   centreline).  The exported polyline is a subsample and runs ~0.5 % shorter, so polyline arc is scaled per
   branch to the manifest length; s_from_root uses the profiles' own offsets.  readout() returns the 2 mm bin
   of manifest.analysis.profiles that contains the station — the same statistics as the profile curves; no
   new statistic is computed here. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.cursor = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  function common() {
    var c = root.WssReportCommon;
    if (!c || typeof c.buildCenterlineGroups !== 'function') throw new Error('report_common.js (WssReportCommon) must be loaded before the cursor is used');
    return c;
  }
  function centerlineKeys(m) {
    var c = m && m.geometry && m.geometry.centerline || {};
    return { xyz: c.xyz || null, radius: c.radius_mm || null, edges: c.edges || null, segment: c.segment || null };
  }
  function requiredArrays(result) {
    var k = centerlineKeys(result.manifest);
    return [k.xyz, k.radius, k.edges, k.segment].filter(function (x) { return typeof x === 'string' && result.declared(x); });
  }
  function arr(result, key) { return key && result.declared(key) ? result.array(key) : null; }

  // Branch metadata from the manifest (id, name, parent); profiles supply names/parents when branches are absent.
  function branchTable(m) {
    var out = {}, list = m && m.geometry && Array.isArray(m.geometry.branches) ? m.geometry.branches : [];
    list.forEach(function (b) { if (b && Number.isFinite(+b.id)) out[+b.id] = { id: +b.id, name: b.name || null, key: b.key || null, parent: b.parent !== null && b.parent !== undefined && Number.isFinite(+b.parent) && +b.parent >= 0 ? +b.parent : null, length_mm: b.length_mm }; });
    var prof = m && m.analysis && m.analysis.profiles && Array.isArray(m.analysis.profiles.branches) ? m.analysis.profiles.branches : [];
    prof.forEach(function (b) {
      var sid = +b.segment_id; if (!Number.isFinite(sid)) return;
      if (!out[sid]) out[sid] = { id: sid, name: b.name || null, key: null, parent: b.parent_id !== null && b.parent_id !== undefined && Number.isFinite(+b.parent_id) && +b.parent_id >= 0 ? +b.parent_id : null, length_mm: b.length_mm };
    });
    return out;
  }

  // Groups from raw arrays: {segment_id, name, xyz (Float32 3k), radius, s (Float32 k), length_mm, parent_id, parent_s, children}.
  function buildGroups(cl, table) {
    var names = {}, parents = {};
    Object.keys(table).forEach(function (k) { if (table[k].name) names[k] = table[k].name; if (table[k].parent !== null) parents[k] = table[k].parent; });
    var groups = common().buildCenterlineGroups({ xyz: cl.xyz, radius: cl.radius, edges: cl.edges, segment: cl.segment }, names, { parents: parents });
    // The manifest's tree wins over geometric inference (roots stay roots).
    var byId = {};
    groups.forEach(function (g) { byId[g.segment_id] = g; });
    groups.forEach(function (g) {
      var t = table[g.segment_id];
      if (t && t.parent === null && Object.keys(table).length) { g.parent_id = null; g.parent_s = 0; }
    });
    groups.forEach(function (g) { g.children = []; });
    groups.forEach(function (g) { if (g.parent_id !== null && byId[g.parent_id]) byId[g.parent_id].children.push(g.segment_id); });
    groups.forEach(function (g) { g.children.sort(function (a, b) { return a - b; }); });
    return groups;
  }

  // g.scale = manifest length / polyline length; g.rootOffset = s_from_root at the branch start (manifest units).
  function stationOn(g, s, byId) {
    var k = g.s.length;
    if (!k) return null;
    var L = g.length_mm * g.scale;
    var sm = Math.max(0, Math.min(L, +s || 0));
    var sc = sm / g.scale;
    var j = 0;
    while (j < k - 2 && g.s[j + 1] < sc) j++;
    var s0 = g.s[j], s1 = k > 1 ? g.s[j + 1] : s0, t = s1 > s0 ? (sc - s0) / (s1 - s0) : 0;
    if (k === 1) t = 0;
    var jb = Math.min(k - 1, j + 1);
    var xyz = [0, 1, 2].map(function (c) { return g.xyz[3 * j + c] + (g.xyz[3 * jb + c] - g.xyz[3 * j + c]) * t; });
    var radius = g.radius[j] + (g.radius[jb] - g.radius[j]) * t;
    // tangent: chord over the neighbouring samples (downstream, like the classic reports)
    var a = Math.max(0, j - 1), b = Math.min(k - 1, jb + 1);
    var v = [g.xyz[3 * b] - g.xyz[3 * a], g.xyz[3 * b + 1] - g.xyz[3 * a + 1], g.xyz[3 * b + 2] - g.xyz[3 * a + 2]], n = Math.hypot(v[0], v[1], v[2]);
    var tangent = n > 1e-9 ? [v[0] / n, v[1] / n, v[2] / n] : null;
    var eps = 1e-6;
    return {
      segmentId: g.segment_id, name: g.name, s_mm: sm, length_mm: L, xyz: xyz, tangent: tangent, radius_mm: radius,
      s_from_root_mm: g.rootOffset + sm, atStart: sm <= eps, atEnd: sm >= L - eps, parent: g.parent_id, children: g.children.slice()
    };
  }
  // Per-branch length scale and root offset (manifest units).  Offsets come from the profiles
  // (s_from_root_mm − s_local_mm, median) when present, else from the tree (parent offset + junction arc).
  function calibrate(groups, table, profById) {
    var byId = {};
    groups.forEach(function (g) { byId[g.segment_id] = g; });
    groups.forEach(function (g) {
      var t = table[g.segment_id], pb = profById[g.segment_id];
      var L = t && Number.isFinite(+t.length_mm) && +t.length_mm > 0 ? +t.length_mm : (pb && +pb.length_mm > 0 ? +pb.length_mm : null);
      g.scale = L && g.length_mm > 0 ? L / g.length_mm : 1;
      if (!(g.scale > 0.8 && g.scale < 1.25)) g.scale = 1;   // implausible: keep the polyline measure
      g.rootOffset = null;
      if (pb && Array.isArray(pb.s_local_mm) && Array.isArray(pb.s_from_root_mm) && pb.s_local_mm.length === pb.s_from_root_mm.length && pb.s_local_mm.length) {
        var d = pb.s_local_mm.map(function (x, i) { return +pb.s_from_root_mm[i] - +x; }).filter(Number.isFinite).sort(function (a, b) { return a - b; });
        if (d.length) g.rootOffset = d[d.length >> 1];
      }
    });
    function offset(g, seen) {
      if (g.rootOffset !== null) return g.rootOffset;
      if (g.parent_id === null || !byId[g.parent_id] || seen[g.segment_id]) return (g.rootOffset = 0);
      seen[g.segment_id] = 1;
      var p = byId[g.parent_id];
      g.rootOffset = offset(p, seen) + g.parent_s * p.scale;
      return g.rootOffset;
    }
    groups.forEach(function (g) { offset(g, {}); });
  }

  // Profile field dict for a field id, normalised keys (mean_pa → mean, speed_max_m_s → max).
  var SUFFIX = /_(pa|per_pa|m_s|mm|1)$/;
  function profileStats(branch, fieldId) {
    if (!branch) return null;
    var dict = null, prefix = '';
    if (branch[fieldId] && typeof branch[fieldId] === 'object') dict = branch[fieldId];
    else if (branch.volume && typeof branch.volume === 'object' && (fieldId === 'speed' || fieldId === 'pressure')) { dict = branch.volume; prefix = fieldId + '_'; }
    if (!dict) return null;
    var out = {};
    Object.keys(dict).forEach(function (k) {
      if (!Array.isArray(dict[k])) return;
      if (prefix && k.indexOf(prefix) !== 0 && k !== 'n') return;
      var name = (prefix ? k.slice(k.indexOf(prefix) === 0 ? prefix.length : 0) : k).replace(SUFFIX, '');
      out[name] = { key: k, values: dict[k] };
    });
    return out;
  }
  function binIndex(centers, s, width) {
    var best = -1, bd = Infinity;
    for (var i = 0; i < centers.length; i++) {
      var c = +centers[i], d = Math.abs(s - c);
      if (d <= width / 2 + 1e-9) { if (s < c + width / 2 || i === centers.length - 1) return i; }
      if (d < bd) { bd = d; best = i; }
    }
    return bd <= width ? best : -1;
  }

  function create(result) {
    if (!result || !result.manifest) throw new Error('cursor needs a loaded result');
    var m = result.manifest, util = ns.util, ev = util.emitter();
    var keys = centerlineKeys(m);
    var cl = { xyz: arr(result, keys.xyz), radius: arr(result, keys.radius), edges: arr(result, keys.edges), segment: arr(result, keys.segment) };
    var table = branchTable(m);
    var groups = cl.xyz && cl.xyz.length >= 3 ? buildGroups(cl, table) : [];
    var byId = {};
    groups.forEach(function (g) { byId[g.segment_id] = g; });
    var profiles = m.analysis && m.analysis.profiles && typeof m.analysis.profiles === 'object' ? m.analysis.profiles : null;
    var profById = {};
    if (profiles && Array.isArray(profiles.branches)) profiles.branches.forEach(function (b) { profById[+b.segment_id] = b; });
    calibrate(groups, table, profById);
    var binW = profiles && +profiles.bin_mm > 0 ? +profiles.bin_mm : 2;
    var cur = null;

    function station(segmentId, s) { var g = byId[+segmentId]; return g ? stationOn(g, s, byId) : null; }
    function state() { return cur ? Object.assign({}, cur, { xyz: cur.xyz.slice(), tangent: cur.tangent ? cur.tangent.slice() : null, children: cur.children.slice() }) : null; }
    function set(segmentId, s) {
      var st = station(segmentId, s);
      cur = st;
      ev.emit('change', state());
      return state();
    }
    // Moves along the current branch only.  At the distal end: atEnd + the child branches to choose from
    // (the caller decides; the cursor never switches branch by itself).  At the proximal end: atStart + parent.
    function step(delta) {
      if (!cur) return { state: null, atEnd: false, atStart: false, children: [], parent: null, moved: false };
      var g = byId[cur.segmentId], st = stationOn(g, cur.s_mm + (+delta || 0), byId);
      var moved = Math.abs(st.s_mm - cur.s_mm) > 1e-9;
      cur = st;
      if (moved) ev.emit('change', state());
      return { state: state(), atEnd: st.atEnd, atStart: st.atStart, children: st.atEnd ? st.children.slice() : [], parent: st.atStart ? st.parent : null, moved: moved };
    }
    function readout(fieldId) {
      var f = result.field(fieldId);
      var base = { fieldId: fieldId, segmentId: cur ? cur.segmentId : null, s_mm: cur ? cur.s_mm : null, bin: null, binIndex: -1, stats: null, n: null, units: f ? f.units || '' : '', available: false };
      if (!cur) return Object.assign(base, { definition: '没有游标站位' });
      var b = profById[cur.segmentId];
      var nameTxt = '「' + (cur.name || ('分支 ' + cur.segmentId)) + '」';
      if (!profiles || !b) return Object.assign(base, { definition: '该结果没有沿程分箱统计（profiles），不在游标处另算统计' });
      var centers = Array.isArray(b.s_local_mm) ? b.s_local_mm : [];
      var k = binIndex(centers, cur.s_mm, binW);
      var ps = profileStats(b, fieldId);
      if (k < 0) return Object.assign(base, { definition: nameTxt + '在弧长 ' + util.fmtSig(cur.s_mm) + ' mm 处没有分箱' });
      var c = +centers[k], bin = [Math.max(0, c - binW / 2), c + binW / 2];
      var nArr = (b.wss && b.wss.n) || (b.volume && b.volume.n) || (ps && ps.n && ps.n.values) || null;
      var n = nArr && Number.isFinite(+nArr[k]) ? +nArr[k] : null;
      var stats = null;
      if (ps) {
        stats = {};
        Object.keys(ps).forEach(function (name) { if (name === 'n') return; var v = ps[name].values[k]; stats[name] = v === null || v === undefined || !Number.isFinite(+v) ? null : +v; });
      }
      var fam = result.family === 'volume' ? '体内预测点' : '壁面预测点';
      var def = nameTxt + '分支内弧长 ' + util.fmtTrim(bin[0]) + '–' + util.fmtTrim(bin[1]) + ' mm 的 ' + util.fmtTrim(binW) + ' mm 分箱：箱内全部' + fam + '等权统计' +
        (n !== null ? '（' + n + ' 点）' : '') + '；与沿程曲线同一口径；弧长从本分支起点沿中心线量起。';
      if (!ps) def = nameTxt + '的沿程分箱里没有「' + (f && (f.short_label || f.label) || fieldId) + '」，不在游标处另算统计';
      return Object.assign(base, { bin: bin, binIndex: k, stats: stats, n: n, available: !!ps && n !== 0, definition: def, branchName: cur.name, s_from_root_mm: cur.s_from_root_mm });
    }
    function branches() {
      return groups.map(function (g) { return { segmentId: g.segment_id, name: g.name, length_mm: g.length_mm * g.scale, parent: g.parent_id, children: g.children.slice(), s_from_root_start_mm: g.rootOffset, key: table[g.segment_id] ? table[g.segment_id].key : null }; });
    }
    // Projects a world point (e.g. a pick) onto the nearest centreline branch: {segmentId, s_mm, dist_mm}.
    function stationFromPoint(p) {
      if (!groups.length || !p) return null;
      var pr = common().projectToCenterline(groups, p);
      return pr ? { segmentId: pr.segment_id, s_mm: pr.s * (byId[pr.segment_id] ? byId[pr.segment_id].scale : 1), dist_mm: pr.dist_mm } : null;
    }
    return {
      available: groups.length > 0,
      binWidth: binW,
      set: set, step: step, state: state, readout: readout, station: station, branches: branches, stationFromPoint: stationFromPoint,
      clear: function () { cur = null; ev.emit('change', null); },
      on: ev.on, off: ev.off
    };
  }

  return { create: create, requiredArrays: requiredArrays, profileStats: profileStats, binIndex: binIndex };
});
