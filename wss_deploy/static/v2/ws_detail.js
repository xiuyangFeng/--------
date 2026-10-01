/* WSS workspace v2 — Lane D (S6d): result details.
 *  · inspector tab 「沿程」: the profile curves of the coloured field (manifest.analysis.profiles, ws_profile.js), the
 *    centreline radius and the branch unrolled map (ws_unroll.js) on one arc-length axis; linked to the along-vessel
 *    cursor (G), hover readout, click → cursor / bin highlight / fly to the point; 「放大」 = all branches, PNG / SVG export;
 *  · 工具 tab sections: case information (tags, notes), outlet names with recompute (confirm {override: true}), the
 *    compute process (stage timings, event log), the input check of a finished job, delete to the trash (with undo);
 *  · keys N (upload), / (search), O (one-pager).
 * Everything goes through the shell's extension registry (ws_shell.js 「extensions」); numbers come from the job record
 * and the manifest, the same sources as the classic workbench / report.
 * P4 lane A: 计算过程 gets the classic centreline rows (#102) and, with 输入检查, serves the input page too (#78); 出口命名
 * shows the confidence caveat (W30) and the naming suggestion's flags (W35). */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.detail = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };
  var API = null;                      // the shell's helper object (shellApi), kept from the last hook call
  function remember(api) { if (api) API = api; return api; }
  function shell() { return API; }
  function num(v) { return typeof v === 'number' && isFinite(v) ? v : null; }
  function pick() { for (var i = 0; i < arguments.length; i++) { var v = num(arguments[i]); if (v !== null) return v; } return null; }

  // ------------------------------------------------------------------ pure: compute process
  var TIMER_NAMES = {ingest: '输入检查', centerline: '中心线提取', smooth_resample: '平滑与重采样', features: '几何特征', inference_5_models: '模型预测', metrics: '指标统计',
    interpolation: '壁面插值', metrics_and_export: '统计与导出', export: '文件导出', report: '报告', precompute: '后台几何预计算', morphology: '形态测量',
    metrics_and_interpolation: '统计与插值', volume_features: '体内采样与特征', inference_volume: '模型推理', inference: '模型推理', streamlines_and_interpolation: '流线与插值'};
  var CACHE_KIND_STAGES = {mesh: 'smooth_resample', resample: 'smooth_resample', pointgeom: 'features', morph: 'morphology', morphvol: 'morphology'};
  var TIMING_KEY_STAGES = {smooth_resample: 'smooth_resample', features: 'features', volume_features: 'volume_features', morphology: 'morphology'};
  // Stage timings of the pipeline (summary.timing_s), the job's own clock (compute / attempt / queue / confirmation) —
  // classic processCard + WssWorkbenchCore.computeTiming / cachedStages.
  function timingModel(job) {
    job = job || {};
    var summary = job.summary || {}, a = job.a || job.stage_a || {}, t = summary.timing_s || a.timing_s || {}, jt = job.timing || {};
    var reused = summary.geometry_cache && Array.isArray(summary.geometry_cache.reused) ? summary.geometry_cache.reused : [];
    var cached = {};
    reused.forEach(function (k) { var st = CACHE_KIND_STAGES[String(k)]; if (st) cached[st] = true; });
    var stages = [];
    Object.keys(t).forEach(function (k) {
      if (k === 'total' || k === 'compute_total' || k === 'precompute' || num(t[k]) === null) return;   // precompute ran during the confirmation: not a pipeline stage
      stages.push({key: k, label: TIMER_NAMES[k] || k, s: t[k], cached: Boolean(TIMING_KEY_STAGES[k] && cached[TIMING_KEY_STAGES[k]])});
    });
    var sum = stages.reduce(function (acc, s) { return acc + s.s; }, 0);
    var recorded = [num(jt.compute_s), num(jt.compute_seconds)].filter(function (v) { return v !== null; });
    var positive = recorded.filter(function (v) { return v > 0; })[0];
    var compute = positive !== undefined ? positive : pick(t.total, recorded[0], job.compute_seconds);
    var attempt = num(jt.attempt_s);
    return {stages: stages, pipeline: pick(t.total, t.compute_total, stages.length ? sum : null), compute: compute, attempt: attempt,
      split: attempt !== null && attempt > 0 && compute !== null && Math.abs(attempt - compute) > 0.05,
      queue: pick(jt.queue_s, jt.queue_seconds, job.queue_seconds, job.elapsed && job.elapsed.queue),
      confirmation: pick(jt.confirmation_s, jt.confirmation_seconds, job.confirmation_seconds, job.elapsed && job.elapsed.confirmation),
      precompute: pick(jt.precompute_s, t.precompute), wall: num(jt.wall_s), device: summary.device || null, gpu: summary.gpu || null};
  }
  var EVENT_TEXT = {
    created: '已创建', started: '开始计算', finished: '计算结束', progress: '进度更新', cancel_requested: '请求取消', attempt_aborted: '本次尝试中止',
    rerun_created: '已换模型重跑', restored: '已从回收站恢复', deleted: '已删除', owner_claimed: '已认领', metadata_updated: '病例信息已修改',
    narrative_updated: '结论已修改', annotations_updated: '标注已保存', findings_review_updated: '发现判定已保存', snapshots_updated: '一页纸配图已更新',
    legacy_release_bound: '已绑定发布包', restart_interrupted: '服务重启时中断', restart_requeued: '服务重启后自动续排',
    drain_requeued: '升级排空时重新排队', device_fallback: '显存不足，已改用 CPU 重算', analysis_rebuilt: '升级后重建分析',
    precompute_started: '后台几何预计算开始', precompute_done: '后台几何预计算完成', precompute_skipped: '后台几何预计算已跳过（无需计算）',
    precompute_cancelled: '后台几何预计算已取消', precompute_failed: '后台几何预计算未完成（阶段 B 自行计算）',
    precompute_incomplete: '后台几何预计算未全部完成（阶段 B 补算）', precompute_interrupted: '后台几何预计算被中断',
    // job actions the classic table does not name
    input_confirmed: '输入已确认', inlet_changed: '入口已更改，重新提取中心线', outlets_confirmed: '出口已确认', outlets_auto_confirmed: '出口已自动确认',
    retried: '已重试', companion_created: '已建立同时预测的结果', companion_failed: '同时预测的结果没有建立', review_approved: '技术复核通过', review_reopened: '已重新打开'};
  function eventText(action) { var k = String(action === null || action === undefined ? '' : action); return Object.prototype.hasOwnProperty.call(EVENT_TEXT, k) ? EVENT_TEXT[k] : k; }
  // The job's own record, newest last, progress ticks left out (classic processCard).
  function eventRows(job, limit) {
    var list = (job && Array.isArray(job.events) ? job.events : []).filter(function (e) { return e && typeof e === 'object' && e.action && e.action !== 'progress'; });
    return list.slice(-(limit || 12)).map(function (e) {
      var extra = e.action === 'device_fallback' && e.from_device ? '（' + e.from_device + ' → ' + (e.to_device || 'cpu') + '）' : e.stage && (e.action === 'started' || e.action === 'finished') ? ' · 阶段 ' + e.stage : '';
      return {at: e.at || '', action: e.action, text: eventText(e.action) + extra};
    });
  }

  // ------------------------------------------------------------------ pure: case information (classic metadataDialog / WssWorkbenchCore)
  function parseTags(text) {
    var out = [];
    String(text === null || text === undefined ? '' : text).split(/[,，、\n]/).forEach(function (p) { var t = p.trim(); if (t && out.indexOf(t) < 0) out.push(t); });
    return out;
  }
  function tagsText(tags) { return Array.isArray(tags) ? tags.filter(function (t) { return typeof t === 'string' && t.trim(); }).join(', ') : (typeof tags === 'string' ? tags : ''); }
  function metadataPayload(form, version) {
    var text = function (v) { return String(v === null || v === undefined ? '' : v).trim(); };
    var p = {version: version, case_id: text(form.case_id), patient_id: text(form.patient_id), scan_label: text(form.scan_label), scan_date: text(form.scan_date),
      tags: parseTags(form.tags), notes: text(form.notes)};
    if (version === undefined || version === null) delete p.version;
    return p;
  }
  function identifierIssue(value, opts) {
    opts = opts || {};
    var label = opts.label || '编号', max = opts.max || 160, text = String(value === null || value === undefined ? '' : value);
    if (!text.trim()) return null;
    if (/[\u0000-\u001f\u007f-\u009f]/.test(text)) return {level: 'error', text: label + '含有换行、制表符等不可见字符（常见于从表格复制），请删掉后重新输入。'};
    if (text.trim().length > max) return {level: 'error', text: label + '最多 ' + max + ' 个字符（当前 ' + text.trim().length + ' 个）。'};
    if (/^[一-龥·]{2,4}$/.test(text.trim())) return {level: 'warn', text: '「' + text.trim() + '」看起来像真实姓名，请改用匿名编号。'};
    return null;
  }
  // Checks of the form before it is sent; null when the payload can go.
  function metadataError(payload, job) {
    if (!payload.case_id && job && job.case_id) return '病例名称不能为空。';
    if (payload.tags.length > 12) return '最多 12 个标签。';
    var long = payload.tags.filter(function (t) { return t.length > 40; })[0];
    if (long) return '标签「' + long.slice(0, 12) + '…」超过 40 个字符。';
    if (payload.scan_date && !/^\d{4}-\d{2}-\d{2}$/.test(payload.scan_date)) return '扫描日期要写成 YYYY-MM-DD。';
    if (payload.notes.length > 2000) return '备注最多 2000 个字符。';
    return null;
  }

  // ------------------------------------------------------------------ pure: outlet names (classic outletCard, override path)
  var OUTLET_NAMES = {inlet: '主动脉入口', 'out-le': '左髂外', 'out-li': '左髂内', 'out-re': '右髂外', 'out-ri': '右髂内'};
  var OUTLET_COLORS = {inlet: '#2569aa', 'out-le': '#0e6b5c', 'out-li': '#6f9f3c', 'out-re': '#c07a2c', 'out-ri': '#8f5ca3'};
  var SWAPS = {lr: {'out-le': 'out-re', 'out-li': 'out-ri', 'out-re': 'out-le', 'out-ri': 'out-li'}, left: {'out-le': 'out-li', 'out-li': 'out-le'}, right: {'out-re': 'out-ri', 'out-ri': 'out-re'}};
  function swapMapping(mapping, kind) {
    var pairs = SWAPS[kind] || {}, out = {};
    Object.keys(mapping || {}).forEach(function (k) { out[k] = pairs[mapping[k]] || mapping[k]; });
    return out;
  }
  function validMapping(mapping, ids) {
    var names = ids.map(function (id) { return mapping[id]; }), seen = {};
    names.forEach(function (n) { seen[n] = true; });
    return names.length === 4 && Object.keys(seen).length === 4 && names.every(function (n) { return n && n !== 'inlet' && OUTLET_NAMES[n]; });
  }
  function sameMapping(a, b) {
    var ka = Object.keys(a || {}), kb = Object.keys(b || {});
    return ka.length === kb.length && ka.every(function (k) { return String(a[k]) === String((b || {})[k]); });
  }
  function outletModel(job) {
    var a = (job && (job.a || job.stage_a)) || {}, p = a.proposal || {};
    var eps = (Array.isArray(p.endpoints) ? p.endpoints : []).filter(function (e) { return e && Array.isArray(e.center_mm); });
    var hist = job && Array.isArray(job.mapping_history) ? job.mapping_history : [];
    var last = hist.length ? hist[hist.length - 1] : null;
    var source = last ? ({automatic_high_confidence: '自动确认', manual_confirmation: '人工确认', manual_override: '人工修改后重算'}[last.source] || '已确认') : (job && job.mapping ? '已确认' : '未确认');
    return {endpoints: eps.map(function (e) { return {sid: String(e.segment_id), kind: e.kind, radius: e.radius_mm, center: e.center_mm}; }),
      mapping: Object.assign({}, (job && job.mapping) || p.mapping || {}), proposal: Object.assign({}, p.mapping || {}),
      confidence: typeof p.confidence === 'number' ? p.confidence : null, gate: p.confidence_gate && typeof p.confidence_gate === 'object' ? p.confidence_gate : null,
      source: source, available: eps.length > 0 && Boolean(p.mapping)};
  }
  // P4 lane A (W30): the naming confidence is a geometric proxy, not a probability — the classic report's gate card
  // (report.py: validated profile or not) in the classic workbench's words (workbench_core.humanOutletReason).
  function confidenceCaveat(gate) {
    return gate && gate.calibration_status === 'validated' ? '已绑定独立校准 profile。' : '把握度由几何形态估算，不是经过临床数据校准的概率。';
  }
  // P4 lane A (W35): the naming suggestion's own warnings — manifest analysis.flags (= summary flags, the classic
  // report's 「需关注的输入信息」), the summary and the stage-A proposal — in the classic workbench's plain words
  // (ws_input.humanOutletReason, ported from workbench_core), each once.
  function outletFlags(manifest, job) {
    var a = (job && (job.a || job.stage_a)) || {}, p = a.proposal || {};
    var an = manifest && manifest.analysis, sm = job && job.summary;
    var raw = [].concat(an && Array.isArray(an.flags) ? an.flags : [], sm && Array.isArray(sm.flags) ? sm.flags : [], Array.isArray(p.flags) ? p.flags : []);
    var human = ns.input && typeof ns.input.humanOutletReason === 'function' ? ns.input.humanOutletReason : function (t) { return String(t || '').trim(); };
    var out = [];
    raw.forEach(function (t) { if (typeof t !== 'string') return; var x = human(t); if (x && out.indexOf(x) < 0) out.push(x); });
    return out;
  }

  // ------------------------------------------------------------------ pure: input check of a finished job
  var CHECK_WORDS = {pass: '通过', pass_with_limits: '通过，有未评估项', review: '需要复核', fail: '不通过', not_checked: '未检查', unknown: '未知'};
  var CHECK_TONE = {pass: 'ok', pass_with_limits: 'ok', review: 'warn', fail: 'error', not_checked: 'idle', unknown: 'idle'};
  function inputModel(ic, openings) {
    if (!ic || typeof ic !== 'object') return null;
    var q = ic.quality || {};
    var units = ic.selected_units === 'auto' ? (ic.resolved_units || ic.suggested_units || ic.unit || '') : (ic.selected_units || ic.unit || '');
    var ops = (Array.isArray(openings) && openings.length ? openings : (q.opening_geometry || [])).map(function (o, i) {
      var r = num(o.radius_mm) !== null ? o.radius_mm : o.equivalent_radius_mm;
      return {i: i + 1, radius: num(r), center: Array.isArray(o.center_mm) ? o.center_mm : null, role: o.role || null};
    });
    var checks = (Array.isArray(q.checks) ? q.checks : []).map(function (c) { return {label: c.label || c.key, status: c.status || 'unknown', note: c.note || ''}; });
    var rank = {fail: 0, review: 1, unknown: 2, not_checked: 3, pass_with_limits: 4, pass: 5};
    checks.sort(function (a, b) { return (rank[a.status] === undefined ? 6 : rank[a.status]) - (rank[b.status] === undefined ? 6 : rank[b.status]); });
    var counts = {};
    checks.forEach(function (c) { counts[c.status] = (counts[c.status] || 0) + 1; });
    return {units: units, auto: ic.selected_units === 'auto', confidence: num(ic.unit_confidence), scale: num(ic.scale_factor), bbox: Array.isArray(ic.bbox_size_mm) ? ic.bbox_size_mm : null,
      areaCm2: num(ic.area_mm2) !== null ? ic.area_mm2 / 100 : null, faces: num(ic.faces), components: num(ic.components), openings: ops,
      grade: q.grade || null, gradeLabel: q.grade_label || CHECK_WORDS[q.grade] || '', checks: checks, counts: counts,
      flags: [].concat(ic.errors || [], ic.flags || []).filter(function (t, i, a) { return typeof t === 'string' && t && a.indexOf(t) === i; })};
  }

  // ------------------------------------------------------------------ P3 lane 3: technical information (W62 / V46)
  // The classic 技术信息 popover rows (report.py renderTech, volume_viewer.js techRows) plus the workspace's own
  // versions, from the manifest only — the same in every tier and in the offline report.  Times: the classic rule
  // (a naive created_at borrows the offset of an ISO stamp of the same run, VolumeViewerCore.withOffset) and
  // WssReportCommon.localTime, followed by the stored text.
  function stampText(raw, hint) {
    if (!raw) return null;
    var V = root.VolumeViewerCore, C = root.WssReportCommon, iso = raw;
    if (V && typeof V.withOffset === 'function') { try { iso = V.withOffset(raw, hint || '') || raw; } catch (_) { iso = raw; } }
    var local = C && typeof C.localTime === 'function' ? C.localTime(iso, {seconds: true}) : String(iso);
    return local + '（' + raw + '）';
  }
  var DIRECTION_TEXT = {unknown_stl: '按解剖坐标架推断（STL 无患者方向）'};
  function techRows(m, job) {
    m = m || {};
    var pv = m.provenance || {}, mr = pv.model_release || {}, res = m.result || {}, fc = pv.feature_contract || {}, mj = m.job || {};
    var t = m.time || {}, ax = (Array.isArray(t.axis) && t.axis[0]) || {}, mf = res.model_frame || {}, cy = t.cycle, di = (m.mapping && m.mapping.display_interpolation) || {};
    var set = function (v) { return v !== null && v !== undefined && v !== ''; };
    var n = num(pv.n_models);
    var frameTime = set(ax.time_s) ? ax.time_s : mf.time_s;
    var rows = [
      ['发布包', mr.registry_id || mr.name || mr.release || res.release_id],
      ['发布包哈希', pv.release_hash || pv.release_fingerprint],
      ['模型族', [pv.model_family, n !== null ? n + ' 个模型' : null].filter(Boolean).join(' · ')],
      ['特征合同', [fc.version, fc.source_hash].filter(Boolean).join(' · ')],
      ['运行身份', res.run_identity],
      ['输入 SHA256', mj.input_sha256 || (job && job.input_sha256)],
      ['模型帧', [ax.label || mf.label || mf.target, set(ax.step) ? 'step ' + ax.step : null, set(frameTime) ? frameTime + ' s' : null].filter(Boolean).join(' · ')],
      ['周期定义', cy ? [cy.frames, set(cy.period_s) ? '周期 ' + cy.period_s + ' s' : null].filter(Boolean).join(' · ') : null],
      ['生成时间', stampText(pv.created_at, pv.time_hint)],
      ['报告重建', stampText(pv.report_rebuilt_at)],
      ['摘要版本', pv.report_schema],
      ['数据版本', m.data_version], ['分析版本', res.analysis_version], ['部署版本', res.deploy_version], ['代码', pv.git_describe],
      ['显示插值', di.method ? di.method + (set(di.sigma_mm) ? '，σ ' + di.sigma_mm + ' mm' : '') + (set(di.max_dist_mm) ? '，最大距离 ' + di.max_dist_mm + ' mm' : '') : null],
      ['计算设备', [pv.device, pv.gpu && pv.gpu !== pv.device ? pv.gpu : null].filter(Boolean).join(' · ')],
      ['坐标架方向来源', m.frame ? (DIRECTION_TEXT[m.frame.direction_source] || m.frame.direction_source) : null]];
    return rows.filter(function (r) { return set(r[1]); }).map(function (r) { return [r[0], String(r[1])]; });
  }
  var TECH_TERMS = {'发布包': 'release', '特征合同': 'feature_contract', '运行身份': 'run_identity'};
  // ctx = {manifest, job, offline, fileUrl(name) | null}
  function techDialog(ctx) {
    ctx = ctx || {};
    var h = ui().h, rows = techRows(ctx.manifest, ctx.job), status = h('span', {'class': 'muted tech-status'});
    var term = function (k) { return ns.lens && ns.lens.termText ? ns.lens.termText(TECH_TERMS[k], '') : ''; };
    var tbl = ui().table([{key: 'k', label: '', render: function (r) { var tip = TECH_TERMS[r.k] ? term(r.k) : ''; return tip ? h('span', {}, r.k, ' ', ui().infoTip(tip)) : r.k; }},
      {key: 'v', label: '', render: function (r) { return /\s/.test(r.v) && !/^[0-9a-f]{16,}/.test(r.v) ? r.v : h('span', {'class': 'mono', text: r.v}); }}],
      rows.map(function (r) { return {k: r[0], v: r[1]}; }), {cls: 'tbl-kv tbl-nohead tbl-tech'});
    var text = rows.map(function (r) { return r[0] + '\t' + r[1]; }).join('\n');
    var copy = ui().button('复制', function () {
      var ok = function () { status.textContent = '已复制到剪贴板。'; }, fail = function () { status.textContent = '剪贴板不可用。'; };
      try { if (root.navigator && root.navigator.clipboard && root.navigator.clipboard.writeText) { root.navigator.clipboard.writeText(text).then(ok, fail); return; } } catch (_) {}
      fail();
    }, {icon: 'copy'});
    var links = !ctx.offline && ctx.fileUrl ? [ui().link('完整统计 JSON', ctx.fileUrl('summary.json'), {newTab: true}), ui().link('运行清单', ctx.fileUrl('run_manifest.json'), {newTab: true})] : [];
    return ui().dialog.open({title: '技术信息', cls: 'dlg-tech', body: [tbl, ui().note('标识与哈希可以全选复制。')],
      actions: [copy, status].concat(links, [h('span', {'class': 'sec-fill'}), ui().button('关闭', function () { ui().dialog.close('done'); })])});
  }

  // ------------------------------------------------------------------ P3 lane 3: sections at a centreline station (V15, V41)
  function sliceOk(api, segmentId) {
    var cur = api && api.cur && api.cur();
    if (!cur || !cur.result || !cur.manifest || !ns.slice || !ns.slice.supported(cur.result) || !api.viewer || !api.viewer() || !api.openSliceAt) return false;
    if (cur.compare || cur.split) return false;
    if (segmentId === undefined || segmentId === null) return true;
    return ((cur.manifest.geometry && cur.manifest.geometry.branches) || []).some(function (b) { return Number(b.id) === Number(segmentId); });
  }
  function canSliceAt(segmentId) { return sliceOk(shell(), segmentId); }
  // The section tool at a fraction of a branch's centreline (classic slice-basis 'centerline'), through the shell's
  // openSliceAt (a {segment, fraction} station): opens the tool when it is closed, shows its tab.
  function sliceStation(api, segmentId, fraction) {
    api = remember(api) || shell();
    if (!sliceOk(api, segmentId)) return false;
    api.openSliceAt({segment: Number(segmentId), fraction: Math.max(0, Math.min(1, Number(fraction) || 0))});
    return true;
  }
  // Classic activateFinding (volume_viewer.js): a branch pressure drop puts the section at its proximal 10 %.
  function pressureDropSlice(item, api, fraction) {
    api = remember(api) || shell();
    if (!item || item.kind !== 'pressure_drop' || item.segment_id === undefined || item.segment_id === null) return false;
    var f = fraction === undefined ? 0.1 : fraction;
    if (!sliceStation(api, item.segment_id, f)) return false;
    var name = item.branch || (ns.lens ? ns.lens.branchName(api.cur().manifest, item.segment_id) : '');
    ui().toast(f < 0.5 ? '压降定义：' + name + ' 近端 10% 与远端 10% 弧长段的平均压差。截面已放在近端 10%；把位置滑到 90% 查看远端。'
      : '截面已放在' + name + '弧长的 90%（压降的远端段）。', {kind: 'info', ms: 9000});
    return true;
  }

  // ------------------------------------------------------------------ 沿程 tab (profiles + radius + unrolled map)
  function alongAvailable(cur) {
    if (!cur || !cur.result || !cur.manifest) return false;
    return Boolean((ns.profile && ns.profile.branchList(cur.manifest).length) || (ns.unroll && ns.unroll.supported(cur.result)));
  }
  function alongState(cur) { if (!cur.along) cur.along = {seg: null, pin: null, pick: null, view: null, ub: null}; return cur.along; }
  function scaleFor(api, cur) {
    var v = api.viewer();
    try { var info = v && v.colorbarInfo ? v.colorbarInfo() : null; if (info && info.fieldId === cur.field && info.scale) return info.scale; } catch (_) {}
    var f = ((cur.manifest && cur.manifest.fields) || []).filter(function (x) { return x && x.id === cur.field; })[0];
    if (!f || !ns.colormap) return null;
    var st = f.statistics || {}, d = f.display || {}, w = cur.window;
    var spec = {window: w === 'adaptive-linear' || !w ? 'adaptive' : w, log: w === 'adaptive-linear' ? false : null, bands: null, cmap: (api.store().prefs() || {}).cmap};
    try { return ns.colormap.resolve(f, spec, {p99: d.p99 !== undefined && d.p99 !== null ? d.p99 : st.p99, min: st.min, max: st.max}).scale; } catch (_) { return null; }
  }
  function fieldShort(m, id) { var f = ((m && m.fields) || []).filter(function (x) { return x && x.id === id; })[0]; return f ? (f.short_label || f.label || id) : (id || ''); }
  function fieldUnits(m, id) { var f = ((m && m.fields) || []).filter(function (x) { return x && x.id === id; })[0]; return f ? f.units : ''; }
  function unitOf(u) { return ui().unitText ? ui().unitText(u) : (u || ''); }
  var CELL_POINTS = 4;   // points per unrolled-map cell on average (the classic page used one cell per canvas pixel)
  var FLY_EXTENT = 30;   // flyTo keeps ~6 × this distance (capped by the current one): the point with some vessel around it
  function onScreen(v, xyz) {
    if (!v || typeof v.projectPoint !== 'function') return true;
    try {
      var p = v.projectPoint(xyz), cv = v.canvasElement, w = cv ? cv.clientWidth : 0, hh = cv ? cv.clientHeight : 0;
      return Boolean(p && p.inFront !== false && (!w || (p.x > 0 && p.x < w && p.y > 0 && p.y < hh)));
    } catch (_) { return true; }
  }
  // Length over circumference of a branch (median centreline radius of its profile bins): the unrolled map's cell shape.
  function wallAspect(m, b) {
    var pb = ns.profile && ns.profile.branchOf(m, b.segmentId);
    var r = pb && Array.isArray(pb.radius_mm) ? pb.radius_mm.map(Number).filter(function (x) { return x > 0; }).sort(function (x, y) { return x - y; }) : [];
    if (!r.length) return null;
    return (b.sMax - b.sMin) / (2 * Math.PI * r[r.length >> 1]);
  }
  function dpr() { return Math.min(2, Math.max(1, Number(root.devicePixelRatio) || 1)); }
  function wire(api, cur) {
    var c = cur.cursor;
    if (c && typeof c.on === 'function' && !c.__alongWired) {
      c.__alongWired = true;
      c.on('change', function () {
        var sh = shell();
        if (!sh || sh.cur() !== cur || sh.currentTab() !== 'along' || cur.cursor !== c) return;
        var st = null; try { st = c.state(); } catch (_) {}
        var A = alongState(cur);
        if (st && st.segmentId !== A.seg) { A.seg = st.segmentId; A.pin = A.pick = null; sh.renderInspector(); return; }
        A.pin = null;
        if (A.view) A.view.sync();
      });
    }
    var v = api.viewer();
    if (v && typeof v.on === 'function' && !v.__alongWired) {
      v.__alongWired = true;
      v.on('change', function (e) {
        var sh = shell(), cc = sh && sh.cur();
        if (!e || e.what !== 'field' || !cc || sh.currentTab() !== 'along' || !cc.along || !cc.along.view) return;
        cc.along.view.repaint();
      });
    }
  }
  function unrollBranches(cur) {
    var A = alongState(cur), U = ns.unroll, res = cur.result;
    if (A.ub) return A.ub;
    var k = U.keysOf(cur.manifest);
    A.ub = U.branches(res.array(k.segment), res.array(k.s), cur.manifest);
    return A.ub;
  }
  function renderAlong(body, api) {
    var cur = api.cur(), h = ui().h, U = ns.unroll, P = ns.profile;
    var res = cur.result, m = cur.manifest, A = alongState(cur);
    var unrollOk = Boolean(U && U.supported(res));
    var need = unrollOk ? U.requiredArrays(res, cur.field).filter(function (k) { return !res.has(k); }) : [];
    if (!unrollOk && U) {   // the bin highlight on volume results needs the points' segment and arc length
      var pk = U.keysOf(m);
      [pk.segment, pk.s].forEach(function (k) { if (k && res.declared(k) && !res.has(k)) need.push(k); });
    }
    if (need.length) {
      ui().fill(body, ui().empty('正在读取预测点…'));
      res.preload(need).then(function () { var sh = shell(); if (sh && sh.cur() === cur && sh.currentTab() === 'along') sh.renderInspector(); },
        function (e) { var sh = shell(); if (sh && sh.cur() === cur && !(e && e.name === 'AbortError')) ui().toast('沿程数据读取失败：' + (e && e.message || e), {kind: 'error'}); });
      return;
    }
    wire(api, cur);
    var ub = unrollOk ? unrollBranches(cur) : [];
    var ubById = {};
    ub.forEach(function (b) { ubById[b.segmentId] = b; });
    var pbIds = {};
    (P ? P.branchList(m) : []).forEach(function (b) { pbIds[Number(b.segment_id)] = b; });
    var list = ((m.geometry && m.geometry.branches) || []).map(function (b) { return {id: Number(b.id), name: b.name || ('分支 ' + b.id)}; })
      .filter(function (b) { return ubById[b.id] || pbIds[b.id]; });
    Object.keys(pbIds).forEach(function (id) { if (!list.some(function (b) { return b.id === Number(id); })) list.push({id: Number(id), name: pbIds[id].name || ('分支 ' + id)}); });
    ub.forEach(function (b) { if (!list.some(function (x) { return x.id === b.segmentId; })) list.push({id: b.segmentId, name: b.name}); });
    if (!list.length) { ui().fill(body, ui().empty('这份结果没有沿程分箱，也没有展开图需要的预测点坐标。')); return; }
    var cst = null;
    try { cst = cur.cursor ? cur.cursor.state() : null; } catch (_) { cst = null; }
    if (cst && list.some(function (b) { return b.id === cst.segmentId; }) && A.seg === null) A.seg = cst.segmentId;
    if (A.seg === null || !list.some(function (b) { return b.id === A.seg; })) A.seg = list[0].id;
    var seg = A.seg, U0 = ubById[seg] || null;
    var md = P ? P.model(m, cur.field, seg, U0 ? [U0.sMin, U0.sMax] : null) : null;
    var xr = U0 ? [U0.sMin, U0.sMax] : md ? md.xRange : null;
    var width = (body.clientWidth ? body.clientWidth - 36 : 324) - 40;
    var pw = Math.max(160, Math.round(width));
    var views = {};

    // head: branch, expand, export
    var sel = ui().select(list.map(function (b) { return {value: String(b.id), label: b.name}; }), String(seg), function (v) { A.seg = Number(v); A.pin = A.pick = null; clearMarks(api); api.renderInspector(); },
      {'aria-label': '分支', 'class': 'along-branch'});
    var expandBtn = unrollOk ? ui().iconButton('expand', '全部分支的展开图', function () { openUnrollDialog(api, cur); }) : null;
    var dl = ui().iconButton('download', '导出沿程曲线 SVG / 展开图 PNG', null);
    dl.setAttribute('aria-haspopup', 'menu');
    dl.addEventListener('click', function () {
      ui().menu(dl, [md ? {label: '沿程曲线 SVG（本分支）', run: function () { exportProfile(api, cur, seg); }} : null,
        unrollOk ? {label: '展开图 PNG（全部分支）', run: function () { exportUnroll(api, cur); }} : null]);
    });
    var tip = '沿中心线每 ' + (md ? md.binMm : 2) + ' mm 一箱的预测点统计（与游标同一口径）；横轴是距主动脉入口的弧长。' + (unrollOk ? '展开图：纵轴是绕中心线的周向角，每格取落入的预测点最大值，颜色与三维同一色标。' : '') +
      '游标打开时点曲线移动游标；否则点曲线在血管上标出该箱，点展开图飞到那个点。';
    var head = h('div', {'class': 'along-head'}, sel, ui().infoTip(tip), h('span', {'class': 'sec-fill'}), expandBtn, dl);

    // strips
    var plot = h('div', {'class': 'along-plot'});
    var marks = [];
    if (md && xr) {
      var sp = md.spec, u = unitOf(sp.units);
      var legend = h('span', {'class': 'along-legend'}, h('i', {'class': 'lg lg-main'}), h('span', {text: sp.primary.name}), h('i', {'class': 'lg lg-sub'}), h('span', {text: sp.secondary.name}),
        sp.hlines.length ? h('i', {'class': 'lg lg-thr'}) : null, sp.hlines.length ? h('span', {text: ui().trim(sp.hlines[0].v)}) : null);
      var main = P.chart({width: pw, height: 118, xRange: xr, x: md.x, lines: [{y: md.primary, cls: 'main'}, {y: md.secondary, cls: 'sub'}], band: [md.primary, md.secondary],
        hlines: sp.hlines, cls: 'along-main', label: '沿程 ' + sp.label, zero: sp.id !== 'pressure', yTicks: 3});
      var cap = h('div', {'class': 'along-cap'}, h('span', {'class': 'along-cap-name', text: sp.label}), u ? h('span', {'class': 'sec-unit', text: u}) : null, h('span', {'class': 'sec-fill'}), legend);
      var note = sp.missing ? ui().note('沿程数据里没有 ' + fieldShort(m, sp.missing) + '，这里是峰值 WSS。') : null;
      if (main) { plot.appendChild(cap); if (note) plot.appendChild(note); plot.appendChild(h('div', {'class': 'along-strip'}, main.el)); marks.push(main); views.main = main; }
      if (md.radius.some(Number.isFinite)) {
        var rc = P.chart({width: pw, height: 42, xRange: xr, x: md.x, lines: [{y: md.radius, cls: 'rad'}], cls: 'along-rad', label: '中心线半径', yTicks: 2, top: 4});
        if (rc) { plot.appendChild(h('div', {'class': 'along-cap'}, h('span', {'class': 'along-cap-name', text: '半径'}), h('span', {'class': 'sec-unit', text: 'mm'}))); plot.appendChild(h('div', {'class': 'along-strip'}, rc.el)); marks.push(rc); views.rad = rc; }
      }
    }
    var mapBox = null, canvas = null, grid = null, mapMark = null, mapHover = null;
    if (unrollOk && U0) {
      var readKey = U.readKey(res, cur.field);
      if (readKey && res.has(readKey)) {
        var k = U.keysOf(m), sz = U.gridSize(U0.n, Math.min(pw * dpr(), U0.nS || Infinity), 84 * dpr(), CELL_POINTS, wallAspect(m, U0));
        grid = U.grid({s: res.array(k.s), theta: res.array(k.theta), segment: res.array(k.segment), values: res.array(readKey), segmentId: seg, sMin: U0.sMin, sMax: U0.sMax, width: sz.width, height: sz.height});
        canvas = h('canvas', {'class': 'along-map', width: grid.width, height: grid.height, 'aria-label': '展开图：' + U0.name});
        mapMark = h('span', {'class': 'along-map-mark', hidden: true});
        mapHover = h('span', {'class': 'along-map-mark hover', hidden: true});
        var dot = h('span', {'class': 'along-map-dot', hidden: true});
        mapBox = h('div', {'class': 'along-mapbox'}, canvas, mapMark, mapHover, dot,
          h('span', {'class': 'along-theta t-top', text: '+180°'}), h('span', {'class': 'along-theta t-mid', text: '0°'}), h('span', {'class': 'along-theta t-bot', text: '−180°'}));
        var fu = fieldUnits(m, cur.field);
        plot.appendChild(h('div', {'class': 'along-cap'}, h('span', {'class': 'along-cap-name', text: '展开图'}), h('span', {'class': 'sec-unit', text: fieldShort(m, cur.field) + (fu && fu !== '1' ? ' · ' + unitOf(fu) : '')}),
          h('span', {'class': 'sec-fill'}), h('span', {'class': 'along-cap-note', text: grid.width + ' × ' + grid.height + ' 格'})));
        plot.appendChild(h('div', {'class': 'along-strip'}, mapBox));
        views.dot = dot;
      } else plot.appendChild(ui().note('当前字段没有逐点数值，展开图不可用。'));
    }
    if (xr) {
      var axis = h('div', {'class': 'along-xaxis'});
      (P ? P.niceTicks(xr[0], xr[1], 4) : []).forEach(function (t) {
        var f = (t - xr[0]) / ((xr[1] - xr[0]) || 1);
        if (f < -0.001 || f > 1.001) return;
        axis.appendChild(h('span', {'class': 'along-xt', style: 'left:' + (f * 100).toFixed(2) + '%', text: P.tickText(t)}));
      });
      axis.appendChild(h('span', {'class': 'along-xunit', text: 'mm'}));
      plot.appendChild(axis);
    }
    var read = h('div', {'class': 'along-read'});

    // ---- readout, marks and interaction
    var S = function (key) { try { return key ? res.array(key) : null; } catch (_) { return null; } };
    var pk2 = U ? U.keysOf(m) : {};
    function binText(i) {
      var sp = md.spec;
      return md.name + ' · 距入口 ' + ui().sig(md.x[i]) + ' mm · ' + sp.primary.name + ' ' + ui().sig(md.primary[i]) + ' · ' + sp.secondary.name + ' ' + ui().num(md.secondary[i], sp.units) +
        (Number.isFinite(md.radius[i]) ? ' · 半径 ' + ui().sig(md.radius[i]) + ' mm' : '');
    }
    function pointText(i) {
      var s = S(pk2.s), th = S(pk2.theta), v = res.valueAt(cur.field, i);
      return '距入口 ' + ui().sig(s[i]) + ' mm · θ ' + (th[i] >= 0 ? '+' : '−') + Math.round(Math.abs(U.degrees(th[i]))) + '° · ' + fieldShort(m, cur.field) + ' ' + ui().num(v, fieldUnits(m, cur.field));
    }
    function setRead(text, actions, muted) { ui().fill(read, h('span', {'class': 'along-read-text' + (muted ? ' muted' : ''), text: text}), actions && actions.length ? h('span', {'class': 'along-read-acts'}, actions) : null); }
    function restRead() {
      if (A.pick && A.pick.seg === seg && grid) {
        setRead(pointText(A.pick.index), [ui().button('来源', function () { lensPoint(A.pick.index); }, {kind: 'link', cls: 'btn-sm'}), ui().button('清除', clearAll, {kind: 'link', cls: 'btn-sm'})]);
        return;
      }
      if (A.pin && A.pin.seg === seg && md) {
        setRead(binText(A.pin.i), [ui().button('来源', function () { lensBin(A.pin.i); }, {kind: 'link', cls: 'btn-sm'}), ui().button('清除', clearAll, {kind: 'link', cls: 'btn-sm'})]);
        return;
      }
      var c = null; try { c = cur.cursor ? cur.cursor.state() : null; } catch (_) {}
      if (c && c.segmentId === seg && md) {
        var i = P.binAt(md, c.s_from_root_mm);
        if (i >= 0) { setRead('游标 · ' + binText(i), [ui().button('来源', function () { lensBin(i); }, {kind: 'link', cls: 'btn-sm'})]); return; }
      }
      var ss = sliceS();
      if (ss !== null && md) { setRead('截面 · ' + md.name + ' 距入口 ' + ui().sig(ss) + ' mm；点曲线把截面移过去。', null, true); return; }
      setRead((sliceOn() ? '点曲线把截面移到那里' : cur.cursor ? '点曲线移动游标' : '悬停读数；点曲线标出该箱') + (grid ? '，点展开图飞到该点。' : '。'), null, true);
    }
    // P3 lane 3 (V41): with the section open a click on the curves moves the section there (classic sliceAtArc:
    // centreline basis, fraction = branch-local arc / the branch's centreline length, angles and offsets reset);
    // the section's station is drawn on the curves.  Arc lengths: curves are from the root, the section local.
    function sliceOn() { var sl = cur.slice; return Boolean(sl && !(sl.disposed && sl.disposed()) && md); }
    function localOffset() {
      for (var q = 0; q < md.x.length; q++) if (Number.isFinite(md.x[q]) && Number.isFinite(md.xLocal[q])) return md.x[q] - md.xLocal[q];
      return 0;
    }
    function sliceS() {
      if (!sliceOn()) return null;
      var st = null; try { st = cur.slice.state(); } catch (_) { return null; }
      if (!st || st.basis === 'pick' || Number(st.segment) !== Number(seg)) return null;
      var L = cur.slice.arcOf(seg);
      return L > 0 ? st.fraction * L + localOffset() : null;
    }
    function moveSlice(s) {
      var L = cur.slice.arcOf(seg);
      if (!(L > 0)) { ui().toast('这条分支没有中心线，截面不能放在这里。', {kind: 'info'}); return; }
      var fraction = Math.max(0, Math.min(1, (s - localOffset()) / L));
      cur.slice.set({basis: 'centerline', segment: Number(seg), fraction: fraction, pick: null, picks: [], shift: 0, pitch: 0, yaw: 0, offU: 0, offV: 0});
      if (A.pin) clearHighlight(api);
      A.pin = null; A.pick = null;
      sync();
      // the classic page kept the camera; here the section may land outside the view: then turn to it once computed
      var sl = cur.slice, v = api.viewer();
      setTimeout(function () {
        try { var c = sl && !sl.disposed() && sl.comp ? sl.comp() : null; if (c && c.plane && !onScreen(v, c.plane.origin)) sl.look(true); } catch (_) {}
      }, 120);
    }
    function setMapMark(elm, s) {
      if (!elm || !grid) return;
      if (s === null || s === undefined || !Number.isFinite(+s) || s < grid.sMin || s > grid.sMax) { elm.hidden = true; return; }
      elm.hidden = false; elm.style.left = ((s - grid.sMin) / ((grid.sMax - grid.sMin) || 1) * 100).toFixed(2) + '%';
    }
    function sync() {
      var c = null; try { c = cur.cursor ? cur.cursor.state() : null; } catch (_) {}
      var cs = c && c.segmentId === seg ? c.s_from_root_mm : null;
      var ps = A.pin && A.pin.seg === seg && md ? md.x[A.pin.i] : null;
      var sls = sliceS();
      marks.forEach(function (ch) { ch.setMark('cursor', cs); ch.setMark('pin', ps); ch.setMark('slice', sls); });
      setMapMark(mapMark, cs !== null ? cs : ps);
      var dot = views.dot;
      if (dot && grid) {
        if (A.pick && A.pick.seg === seg) {
          var s = S(pk2.s), th = S(pk2.theta), fr = U.fractionOf(grid, s[A.pick.index], th[A.pick.index]);
          dot.hidden = false; dot.style.left = (fr.fx * 100).toFixed(2) + '%'; dot.style.top = (fr.fy * 100).toFixed(2) + '%';
        } else dot.hidden = true;
      }
      restRead();
    }
    function hover(s, pointIndex) {
      marks.forEach(function (ch) { ch.setMark('hover', s); });
      setMapMark(mapHover, s);
      if (s === null) { restRead(); return; }
      if (pointIndex !== undefined && pointIndex !== null) { setRead(pointText(pointIndex), null); return; }
      if (md) { var i = P.binAt(md, s); if (i >= 0) setRead(binText(i), null); }
    }
    function clearAll() {
      A.pin = null; A.pick = null;
      clearMarks(api);
      sync();
    }
    function lensBin(i) {
      var sp = md.spec, stats = {};
      stats[sp.primary.key] = md.primary[i]; stats[sp.secondary.key] = md.secondary[i];
      if (Number.isFinite(md.radius[i])) stats.radius_mm = md.radius[i];
      if (md.n.length) stats.n = md.n[i];
      var lo = Math.max(0, md.xLocal[i] - md.binMm / 2), hi = md.xLocal[i] + md.binMm / 2;
      if (api.openLens) api.openLens({kind: 'cursor', field: sp.id, readout: {segmentId: seg, s_mm: md.xLocal[i], s_from_root_mm: md.x[i], bin: [lo, hi], stats: stats,
        definition: '「' + md.name + '」分支内弧长 ' + ui().trim(lo) + '–' + ui().trim(hi) + ' mm 的 ' + md.binMm + ' mm 分箱：箱内全部预测点等权统计；与游标、沿程曲线同一口径。'}});
    }
    function lensPoint(i) {
      var pv = S(pk2.xyz), s = S(pk2.s), th = S(pk2.theta), values = {};
      ((m.fields) || []).forEach(function (f) { if (f && f.kind !== 'vector' && f.arrays && f.arrays.read && res.has(f.arrays.read)) values[f.id] = res.valueAt(f.id, i); });
      if (api.openLens) api.openLens({kind: 'point', field: cur.field, pointIndex: i, xyz: [pv[3 * i], pv[3 * i + 1], pv[3 * i + 2]], segmentId: seg, s_from_root_mm: s[i], theta_rad: th[i],
        value: res.valueAt(cur.field, i), values: values});
    }
    function clickBin(s) {
      if (!md) return;
      if (sliceOn()) { moveSlice(s); return; }
      var i = P.binAt(md, s);
      if (i < 0) return;
      if (cur.cursor) { try { cur.cursor.set(seg, md.xLocal[i]); } catch (_) {} A.pin = null; A.pick = null; sync(); return; }
      A.pick = null; A.pin = {seg: seg, i: i};
      var v = api.viewer(), segA = S(pk2.segment), sA = S(pk2.s), idx = [];
      if (segA && sA) for (var k2 = 0; k2 < segA.length; k2++) if (segA[k2] === seg && Math.abs(sA[k2] - md.x[i]) <= md.binMm) idx.push(k2);
      try { if (v) { v.select(null); v.highlight(idx.length ? idx : null, {}); } } catch (_) {}
      var pv = S(pk2.xyz);
      if (pv && idx.length) {   // bring the bin into view when it is off screen (the classic page only marked it)
        var c3 = [0, 0, 0];
        idx.forEach(function (q) { c3[0] += pv[3 * q]; c3[1] += pv[3 * q + 1]; c3[2] += pv[3 * q + 2]; });
        c3 = c3.map(function (x) { return x / idx.length; });
        if (!onScreen(v, c3)) api.flyTo(c3, FLY_EXTENT);
      }
      sync();
    }
    function clickMap(fx, fy) {
      var p = U.pickAt(grid, fx, fy, 2);
      if (!p) return;
      A.pin = null; A.pick = {seg: seg, index: p.index};
      var v = api.viewer(), pv = S(pk2.xyz);
      try { if (v) { v.highlight(null); v.select(p.index); } } catch (_) {}
      if (pv) api.flyTo([pv[3 * p.index], pv[3 * p.index + 1], pv[3 * p.index + 2]], FLY_EXTENT);
      sync();
    }
    function fracIn(el, e) {
      var r = el.getBoundingClientRect ? el.getBoundingClientRect() : null;
      if (!r || !r.width) return null;
      return {fx: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), fy: Math.max(0, Math.min(1, (e.clientY - r.top) / r.height))};
    }
    marks.forEach(function (ch) {
      ch.el.addEventListener('pointermove', function (e) { var f = fracIn(ch.el, e); if (f) hover(ch.invX(f.fx * ch.W)); });
      ch.el.addEventListener('pointerleave', function () { hover(null); });
      ch.el.addEventListener('click', function (e) { var f = fracIn(ch.el, e); if (f) clickBin(ch.invX(f.fx * ch.W)); });
    });
    if (mapBox && grid) {
      mapBox.addEventListener('pointermove', function (e) {
        var f = fracIn(canvas, e); if (!f) return;
        var p = U.pickAt(grid, f.fx, f.fy, 1);
        hover(U.sAt(grid, f.fx), p ? p.index : null);
      });
      mapBox.addEventListener('pointerleave', function () { hover(null); });
      mapBox.addEventListener('click', function (e) { var f = fracIn(canvas, e); if (f) clickMap(f.fx, f.fy); });
    }
    A.view = {
      sync: sync, grid: grid, model: md, clickBin: clickBin, clickMap: clickMap,
      repaint: function () { if (canvas && grid) U.paint(canvas, grid, scaleFor(api, cur)); }
    };
    ui().fill(body, h('section', {'class': 'sec sec-along'}, head, plot, read));
    A.view.repaint();
    sync();
    // the section moved elsewhere (keys, drag, its own panel): its line follows while this view is the one on screen
    var view = A.view;
    if (cur.slice && typeof cur.slice.addPanel === 'function' && md) {
      cur.slice.addPanel({alive: function () { var sh = shell(); return Boolean(sh && sh.cur() === cur && A.view === view && sh.currentTab() === 'along'); },
        update: function () { sync(); }});
    }
  }
  function clearHighlight(api) { var v = api && api.viewer(); try { if (v) v.highlight(null); } catch (_) {} }
  function clearMarks(api) {
    var v = api && api.viewer();
    try { if (v) { v.highlight(null); v.select(null); } } catch (_) {}
  }
  function fileBase(api) { return api.fileBase ? api.fileBase() : 'case'; }
  function exportProfile(api, cur, seg) {
    var hidden = api.hideName && api.hideName();
    var list = ns.profile.exportSVGs(cur.manifest, cur.field, seg, {caseName: hidden ? '病例' : ((cur.manifest.job && cur.manifest.job.display_name) || 'case')});
    if (!list.length || typeof root.Blob !== 'function') { ui().toast('这条分支没有可导出的沿程曲线。', {kind: 'info'}); return; }
    list.forEach(function (it) { ui().downloadBlob(new root.Blob([it.svg], {type: 'image/svg+xml;charset=utf-8'}), it.filename); });
    ui().toast('已导出「' + list[0].branch + '」的 ' + list.length + ' 张 SVG。', {kind: 'ok', ms: 3000});
  }
  function unrollRows(api, cur, maxW, rowH) {
    var U = ns.unroll, res = cur.result, m = cur.manifest, k = U.keysOf(m), key = U.readKey(res, cur.field);
    if (!key || !res.has(key)) return [];
    return unrollBranches(cur).map(function (b) {
      var sz = U.gridSize(b.n, Math.min(maxW, b.nS || Infinity), rowH, CELL_POINTS, wallAspect(m, b));
      return {name: b.name, segmentId: b.segmentId, sMin: b.sMin, sMax: b.sMax,
        grid: U.grid({s: res.array(k.s), theta: res.array(k.theta), segment: res.array(k.segment), values: res.array(key), segmentId: b.segmentId, sMin: b.sMin, sMax: b.sMax, width: sz.width, height: sz.height})};
    });
  }
  function exportUnroll(api, cur) {
    var m = cur.manifest, rows = unrollRows(api, cur, 1410, 150);
    var name = api.hideName && api.hideName() ? '病例' : ((m.job && m.job.display_name) || '');
    var cv = rows.length ? ns.unroll.figure(rows, {title: [name, fieldShort(m, cur.field), '分支展开图'].filter(Boolean).join(' · '), fieldLabel: fieldShort(m, cur.field),
      units: unitOf(fieldUnits(m, cur.field)), scale: scaleFor(api, cur)}) : null;
    if (!cv || !cv.toBlob) { ui().toast('这台浏览器不能生成 PNG。', {kind: 'error'}); return; }
    cv.toBlob(function (blob) { if (blob) ui().downloadBlob(blob, fileBase(api) + '_unroll_' + cur.field + '.png'); }, 'image/png');
  }
  // 「放大」: every branch, like the classic 分支展开图 card; a click closes the dialog and flies to the point.
  function openUnrollDialog(api, cur) {
    var h = ui().h, U = ns.unroll, m = cur.manifest, res = cur.result, k = U.keysOf(m);
    var rows = unrollRows(api, cur, 700 * dpr(), 64 * dpr());
    if (!rows.length) { ui().toast('当前字段没有逐点数值，展开图不可用。', {kind: 'info'}); return; }
    var scale = scaleFor(api, cur);
    var read = h('p', {'class': 'unroll-read muted', text: '悬停读数；点一下关闭并飞到该点。'});
    var s = res.array(k.s), th = res.array(k.theta), pv = res.array(k.xyz);
    var body = rows.map(function (r) {
      var cv = h('canvas', {'class': 'unroll-canvas', width: r.grid.width, height: r.grid.height});
      U.paint(cv, r.grid, scale);
      cv.addEventListener('pointermove', function (e) {
        var rc = cv.getBoundingClientRect(); if (!rc || !rc.width) return;
        var p = U.pickAt(r.grid, (e.clientX - rc.left) / rc.width, (e.clientY - rc.top) / rc.height, 1);
        read.textContent = p ? r.name + ' · 距入口 ' + ui().sig(s[p.index]) + ' mm · θ ' + (th[p.index] >= 0 ? '+' : '−') + Math.round(Math.abs(U.degrees(th[p.index]))) + '° · ' + fieldShort(m, cur.field) + ' ' + ui().num(p.value, fieldUnits(m, cur.field)) : r.name;
      });
      cv.addEventListener('click', function (e) {
        var rc = cv.getBoundingClientRect(); if (!rc || !rc.width) return;
        var p = U.pickAt(r.grid, (e.clientX - rc.left) / rc.width, (e.clientY - rc.top) / rc.height, 2);
        if (!p) return;
        ui().dialog.close('done');
        var A = alongState(cur);
        A.seg = r.segmentId; A.pin = null; A.pick = {seg: r.segmentId, index: p.index};
        var v = api.viewer();
        try { if (v) { v.highlight(null); v.select(p.index); } } catch (_) {}
        api.flyTo([pv[3 * p.index], pv[3 * p.index + 1], pv[3 * p.index + 2]], FLY_EXTENT);
        if (api.currentTab() === 'along') api.renderInspector(); else api.setTab('along');
      });
      return h('div', {'class': 'unroll-row'}, h('div', {'class': 'unroll-label'}, h('b', {text: r.name}), h('span', {'class': 'muted', text: ui().sig(r.sMin) + '–' + ui().sig(r.sMax) + ' mm'})), cv);
    });
    var bar = null;
    if (scale && typeof scale.colorT === 'function') {
      var bc = h('canvas', {'class': 'unroll-bar', width: 240, height: 1});
      var ctx = bc.getContext ? bc.getContext('2d') : null;
      if (ctx && ctx.createImageData) {
        var img = ctx.createImageData(240, 1);
        for (var x = 0; x < 240; x++) { var c = scale.colorT(x / 239); img.data[4 * x] = Math.round(c[0] * 255); img.data[4 * x + 1] = Math.round(c[1] * 255); img.data[4 * x + 2] = Math.round(c[2] * 255); img.data[4 * x + 3] = 255; }
        ctx.putImageData(img, 0, 0);
      }
      var d = scale.describe ? scale.describe() : null;
      bar = h('div', {'class': 'unroll-legend'}, h('span', {text: d ? ui().sig(d.shown[0]) : ''}), bc, h('span', {text: d ? ui().sig(d.shown[1]) + ' ' + unitOf(fieldUnits(m, cur.field)) : ''}),
        d && d.log ? h('span', {'class': 'muted', text: '对数'}) : null);
    }
    ui().dialog.open({title: '分支展开图 · ' + fieldShort(m, cur.field), wide: true, cls: 'dlg-unroll',
      body: [h('div', {'class': 'unroll-rows'}, body), read],
      actions: [bar, h('span', {'class': 'sec-fill'}), ui().button('导出 PNG', function () { exportUnroll(api, cur); }, {icon: 'download'}), ui().button('关闭', function () { ui().dialog.close('done'); })]});
  }

  // ------------------------------------------------------------------ 工具 tab sections
  function locked(api) { return Boolean(api.isLocked && api.isLocked()); }
  var LOCK_TITLE = '已复核锁定；需要修改请先在「技术复核」里重新打开';
  function kvTable(rows) {
    return ui().table([{key: 'k', label: ''}, {key: 'v', label: '', render: function (r) { return r.v; }}], rows, {cls: 'tbl-kv tbl-nohead tbl-detail'});
  }
  function caseSection(api) {
    var h = ui().h, cur = api.cur(), j = cur.job || {};
    var tags = Array.isArray(j.tags) ? j.tags : [];
    var rows = [];
    if (j.patient_id) rows.push({k: '患者编号', v: j.patient_id});
    if (j.scan_date || j.scan_label) rows.push({k: '扫描', v: [j.scan_date, j.scan_label].filter(Boolean).join(' · ')});
    if (tags.length) rows.push({k: '标签', v: h('span', {'class': 'tag-list'}, tags.map(function (t) { return h('span', {'class': 'tag', text: t}); }))});
    if (j.notes) rows.push({k: '备注', v: h('span', {'class': 'notes-text', text: j.notes})});
    var edit = ui().button('编辑…', function () { metadataDialog(api); }, {kind: 'link', cls: 'btn-sm', disabled: locked(api), title: locked(api) ? LOCK_TITLE : null});
    return ui().section('病例信息', {actions: edit, cls: 'sec-case'}, rows.length ? kvTable(rows) : ui().note('还没有填写患者编号、扫描日期和标签。随访需要患者编号和扫描日期。'));
  }
  function metadataDialog(api) {
    api = remember(api) || shell();
    var h = ui().h, cur = api && api.cur(), j = (cur && cur.job) || {};
    if (!cur || !cur.jobId) return;
    var mk = function (label, key, attrs, value) {
      var area = attrs && attrs.area;
      var input = area ? h('textarea', {'aria-label': label, rows: 3, maxLength: 2000}) : h('input', Object.assign({type: 'text', 'aria-label': label}, attrs || {}));
      input.value = value !== undefined ? value : (j[key] || '');
      var hint = h('span', {'class': 'fld-hint', hidden: true});
      return {key: key, input: input, hint: hint, label: label, el: h('label', {'class': 'fld'}, h('span', {text: label}), input, hint)};
    };
    var f = {case_id: mk('病例名称', 'case_id', {maxLength: 160, autocomplete: 'off'}), patient_id: mk('患者编号', 'patient_id', {maxLength: 80, autocomplete: 'off', placeholder: '匿名编号，随访用'}),
      scan_label: mk('扫描标签', 'scan_label', {maxLength: 120, autocomplete: 'off', placeholder: '例如 术前、第一次随访'}), scan_date: mk('扫描日期', 'scan_date', {type: 'date'}),
      tags: mk('标签', 'tags', {maxLength: 1024, autocomplete: 'off', placeholder: '逗号分隔，例如 AAA, 随访'}, tagsText(j.tags)), notes: mk('备注', 'notes', {area: true})};
    var limits = {case_id: 160, patient_id: 80, scan_label: 120};
    Object.keys(limits).forEach(function (key) {
      var x = f[key];
      x.input.addEventListener('input', function () {
        var issue = identifierIssue(x.input.value, {label: x.label, max: limits[key]});
        x.hint.hidden = !issue; x.hint.textContent = issue ? issue.text : ''; x.hint.className = 'fld-hint' + (issue && issue.level === 'error' ? ' err' : '');
      });
    });
    var status = h('p', {'class': 'note note-error', hidden: true});
    var fail = function (text, input) { status.hidden = false; status.textContent = text; if (input) { input.setAttribute('aria-invalid', 'true'); try { input.focus(); } catch (_) {} } };
    var save = ui().button('保存', function () {
      status.hidden = true;
      var keys = Object.keys(limits);
      for (var i = 0; i < keys.length; i++) {
        var x = f[keys[i]], issue = identifierIssue(x.input.value, {label: x.label, max: limits[keys[i]]});
        if (issue && issue.level === 'error') { fail(issue.text, x.input); return; }
        x.input.removeAttribute('aria-invalid');
      }
      var payload = metadataPayload({case_id: f.case_id.input.value, patient_id: f.patient_id.input.value, scan_label: f.scan_label.input.value, scan_date: f.scan_date.input.value,
        tags: f.tags.input.value, notes: f.notes.input.value}, j.version);
      var err = metadataError(payload, j);
      if (err) { fail(err, err.indexOf('病例名称') === 0 ? f.case_id.input : err.indexOf('标签') >= 0 ? f.tags.input : err.indexOf('扫描日期') === 0 ? f.scan_date.input : null); return; }
      if (!payload.case_id) delete payload.case_id;
      save.disabled = true;
      api.api().metadata(cur.jobId, payload).then(function (r) {
        ui().dialog.close('done');
        var job = r && (r.job || r);
        if (job && job.id === cur.jobId) cur.job = Object.assign({}, cur.job || {}, job);
        ui().toast('病例信息已保存。', {kind: 'ok', ms: 3000});
        api.scheduleRefresh();
        if (api.cur() === cur) api.reloadCurrent();
      }, function (e) {
        save.disabled = false;
        if (e && e.status === 409) { ui().dialog.close('cancel'); api.conflictOr(e, '没有保存'); return; }
        fail((e && e.message) || '没有保存。');
      });
    }, {kind: 'primary'});
    ui().dialog.open({title: '编辑病例信息', cls: 'dlg-meta', body: [h('div', {'class': 'form-grid'}, f.case_id.el, f.patient_id.el, f.scan_label.el, f.scan_date.el), f.tags.el, f.notes.el,
      ui().note('只改检索和显示用的信息，不影响计算结果。病例名称来自文件名时可能是姓名，请改用匿名编号。'), status],
      actions: [ui().button('取消', function () { ui().dialog.close('cancel'); }), save], focus: f.case_id.input});
    return f;
  }

  function outletLabels(api, model, mapping) {
    var v = api && api.viewer();
    if (!v || typeof v.setToolLabels !== 'function') return;
    try {
      v.setToolLabels('outlets', model ? model.endpoints.map(function (e) {
        var name = e.kind === 'inlet' ? 'inlet' : mapping[e.sid];
        return {xyz: e.center, text: '#' + e.sid + ' ' + (OUTLET_NAMES[name] || '未命名')};
      }) : []);
    } catch (_) {}
  }
  // W35: the naming suggestion's warnings under the outlet names (nothing when there are none).
  function outletFlagList(flags) {
    var h = ui().h;
    if (!flags || !flags.length) return null;
    return h('div', {'class': 'outlet-flags'}, h('span', {'class': 'muted outlet-flags-head', text: '命名时的核对提示'}),
      h('ul', {'class': 'reasons'}, flags.map(function (t) { return h('li', {text: t}); })));
  }
  function outletSection(api) {
    var h = ui().h, cur = api.cur(), job = cur.job || {}, model = outletModel(job);
    var outlets = model.endpoints.filter(function (e) { return e.kind !== 'inlet'; });
    var ids = outlets.map(function (e) { return e.sid; });
    var ed = cur.outletEdit || null;
    var flags = outletFlags(cur.manifest, job);
    var chip = function (sid, name) { return h('span', {'class': 'outlet-chip'}, h('i', {style: 'background:' + (OUTLET_COLORS[name] || '#98a2b3')}), h('span', {text: '#' + sid + ' ' + (OUTLET_NAMES[name] || '未命名')})); };
    var tip = ui().infoTip('STL 不带患者方向，出口名称决定左右；名称错了，左右分支的结果会对调。修改后按新命名重新计算这个结果，原结果被替换；同时预测的其他结果不会自动重算。');
    if (!ed) {
      if (!model.available) return ui().section('出口命名', {tag: tip}, ui().note('这份结果没有出口命名建议的记录，不能在这里修改。'), outletFlagList(flags));
      var go = ui().button('修改并重算…', function () { cur.outletEdit = {mapping: Object.assign({}, model.mapping), ack: false}; outletLabels(api, model, cur.outletEdit.mapping); api.renderInspector(); },
        {kind: 'link', cls: 'btn-sm', disabled: locked(api) || job.status !== 'done', title: locked(api) ? LOCK_TITLE : null});
      return ui().section('出口命名', {tag: tip, actions: go, cls: 'sec-outlets'},
        h('div', {'class': 'outlet-chips'}, outlets.map(function (e) { return chip(e.sid, model.mapping[e.sid]); })),
        h('p', {'class': 'note outlet-source'}, h('span', {text: model.source + (model.confidence !== null ? ' · 自动命名把握度 ' + Math.round(model.confidence * 100) + '%' : '')}),
          model.confidence !== null ? ui().infoTip(confidenceCaveat(model.gate)) : null),
        outletFlagList(flags));
    }
    var mapping = ed.mapping;
    var ack = h('input', {type: 'checkbox', checked: Boolean(ed.ack)});
    var changed = !sameMapping(mapping, model.mapping);
    var ok = validMapping(mapping, ids);
    var submit = ui().button('采用并重算', function () {
      submit.disabled = true;
      api.api().confirm(cur.jobId, {mapping: Object.assign({}, mapping), acknowledged: true, override: true, stage: 'B', version: job.version}).then(function () {
        cur.outletEdit = null; outletLabels(api, null);
        ui().toast('已按新的出口命名开始重算。', {kind: 'ok'});
        api.scheduleRefresh(); api.reloadCurrent();
      }, function (e) { submit.disabled = false; api.conflictOr(e, '没有开始重算'); });
    }, {kind: 'primary', cls: 'btn-sm', disabled: !ed.ack || !ok || !changed});
    ack.addEventListener('change', function () { ed.ack = ack.checked; submit.disabled = !ack.checked || !ok || !changed; });
    var set = function (next) { cur.outletEdit = {mapping: next, ack: false}; outletLabels(api, model, next); api.renderInspector(); };
    var rows = model.endpoints.map(function (e) {
      var cell = e.kind === 'inlet' ? h('span', {text: OUTLET_NAMES.inlet}) : ui().select([{value: '', label: '请选择'}].concat(['out-le', 'out-li', 'out-re', 'out-ri'].map(function (k) { return {value: k, label: OUTLET_NAMES[k]}; })),
        mapping[e.sid] || '', function (v) { var next = Object.assign({}, mapping); next[e.sid] = v; set(next); }, {'aria-label': '出口 ' + e.sid + ' 的名称'});
      return {sid: e.sid, kind: e.kind, radius: e.radius, name: cell, center: e.center};
    });
    var tbl = ui().table([
      {key: 'sid', label: '端点', render: function (r) { var b = h('button', {type: 'button', 'class': 'val-link', text: '#' + r.sid + (r.kind === 'inlet' ? ' 入口' : ''), title: '在三维里看这个端点'}); b.addEventListener('click', function () { if (r.center) api.flyTo(r.center, 14); }); return b; }},
      {key: 'radius', label: '半径', num: true, render: function (r) { return ui().num(r.radius, 'mm'); }},
      {key: 'name', label: '名称', render: function (r) { return r.name; }}], rows, {cls: 'tbl-outlets'});
    var sw = function (label, kind, title) { return ui().button(label, function () { set(swapMapping(mapping, kind)); }, {cls: 'btn-sm', title: title}); };
    var diff = Object.keys(mapping).filter(function (k) { return mapping[k] !== model.mapping[k]; });
    return ui().section('出口命名', {tag: h('span', {'class': 'sec-tagrow'}, tip, h('span', {'class': 'sec-note', text: '修改中'})), cls: 'sec-outlets editing'},
      tbl, outletFlagList(flags),
      h('div', {'class': 'swap-row'}, sw('左右互换', 'lr', '左右两侧整体互换'), sw('左侧内 / 外', 'left', '左侧髂内与髂外互换'), sw('右侧内 / 外', 'right', '右侧髂内与髂外互换'),
        ui().button('恢复', function () { set(Object.assign({}, model.mapping)); }, {kind: 'link', cls: 'btn-sm', title: '恢复当前结果使用的命名'})),
      !ok ? ui().note('四个出口名称各用一次，同一髂总下的两个出口属于同侧。', 'warn') : diff.length ? h('p', {'class': 'note', text: '已改 ' + diff.length + ' 个：' + diff.map(function (k) { return '#' + k + ' → ' + (OUTLET_NAMES[mapping[k]] || '未命名'); }).join('；')}) : ui().note('命名没有变化。'),
      h('label', {'class': 'check'}, ack, h('span', {text: '我已对照原始影像核对入口和四个出口的名称'})),
      h('div', {'class': 'sec-actions'}, submit, ui().button('取消', function () { cur.outletEdit = null; outletLabels(api, null); api.renderInspector(); }, {cls: 'btn-sm'})));
  }

  var STAGE_SHADES = ['#0e6e8a', '#3f8fa6', '#79b3c3', '#a9cfd9', '#5a6b7d', '#8795a4', '#b7c0ca', '#d5dbe1'];
  // P4 lane A (#102): the classic processCard rows 「中心线检查 通过 / 未通过」「中心线端点 / 分叉」 (stage-A centreline
  // hard_pass, topology.endpoints / junctions).  An empty record (the step never ran) shows nothing.
  function centerlineRows(job) {
    var a = (job && (job.a || job.stage_a)) || {}, c = a.centerline;
    if (!c || typeof c !== 'object') c = job && job.summary && job.summary.centerline;
    if (!c || typeof c !== 'object' || (c.hard_pass === undefined && !(c.topology && typeof c.topology === 'object'))) return [];
    var t = c.topology && typeof c.topology === 'object' ? c.topology : {};
    var v = function (x) { return x === null || x === undefined ? '—' : String(x); };
    return [{k: '中心线检查', v: c.hard_pass ? '通过' : '未通过', ok: Boolean(c.hard_pass)}, {k: '中心线端点 / 分叉', v: v(t.endpoints) + ' / ' + v(t.junctions)}];
  }
  function processSection(api) { var cur = api.cur(); return processSectionFor(cur.job || {}, cur); }
  // ``holder`` keeps the open / closed state of the event list (the result's ``cur`` on the tools tab, the input
  // page's own state on unfinished jobs, P4 lane A #78).
  function processSectionFor(job, holder, opts) {
    var h = ui().h, cur = holder || {}, t = timingModel(job);
    opts = opts || {};
    var parts = [];
    if (t.stages.length) {
      var total = t.stages.reduce(function (a, s) { return a + s.s; }, 0) || 1;
      parts.push(h('div', {'class': 'stage-bar', role: 'img', 'aria-label': '各阶段耗时'}, t.stages.map(function (s, i) {
        return h('span', {'class': 'stage-seg', style: 'flex-grow:' + Math.max(s.s / total * 100, 0.4).toFixed(3) + ';background:' + STAGE_SHADES[i % STAGE_SHADES.length], title: s.label + ' ' + ui().sig(s.s) + ' 秒' + (s.cached ? ' · 已预计算' : '')});
      })));
      parts.push(h('div', {'class': 'stage-legend'}, t.stages.map(function (s, i) {
        return h('span', {'class': 'stage-item' + (s.cached ? ' cached' : ''), title: s.label + (s.cached ? '：确认出口期间已在后台算好' : '')}, h('i', {style: 'background:' + STAGE_SHADES[i % STAGE_SHADES.length]}),
          h('span', {'class': 'stage-name', text: s.label}), h('span', {'class': 'stage-s', text: (s.s < 0.05 ? '<0.1' : ui().sig(s.s, 2)) + ' 秒'}));
      })));
    }
    var line = [];
    if (t.split) line.push('本次尝试 ' + ui().duration(t.attempt));
    if (t.queue !== null) line.push('排队 ' + ui().duration(t.queue));
    if (t.confirmation !== null) line.push('人工确认 ' + ui().duration(t.confirmation));
    if (t.precompute !== null && t.precompute > 0) line.push('后台预计算 ' + ui().duration(t.precompute) + '（不计入）');
    if (t.wall !== null) line.push('任务历时 ' + ui().duration(t.wall));
    if (t.device) line.push('设备 ' + t.device + (t.gpu ? ' · ' + t.gpu : ''));
    if (line.length) parts.push(h('p', {'class': 'note', text: line.join(' · ')}));
    var cl = centerlineRows(job);
    if (cl.length) parts.push(kvTable(cl.map(function (r) { return {k: r.k, v: r.ok === undefined ? r.v : ui().dot(r.ok ? 'ok' : 'error', r.v)}; })));
    var evs = eventRows(job, 12);
    if (evs.length) {
      var list = h('ol', {'class': 'event-list', hidden: !cur.eventsOpen}, evs.map(function (e) { return h('li', {}, h('span', {'class': 'ev-time', text: ui().time(e.at)}), h('span', {text: e.text})); }));
      var label = function () { return (cur.eventsOpen ? '收起运行记录' : '运行记录') + ' · ' + evs.length; };
      var tg = ui().button(label(), function () { cur.eventsOpen = !cur.eventsOpen; list.hidden = !cur.eventsOpen; ui().fill(tg, h('span', {text: label()})); }, {kind: 'link', cls: 'btn-sm'});
      parts.push(h('div', {'class': 'sec-actions'}, tg), list);
    }
    if (!parts.length) {
      if (opts.quiet) return null;
      parts.push(ui().note('这份结果没有计时记录。'));
    }
    var headNote = t.compute !== null ? h('span', {'class': 'sec-note', text: '计算 ' + ui().duration(t.compute)}) : null;
    return ui().section('计算过程', {tag: h('span', {'class': 'sec-tagrow'}, ui().infoTip('条形 = 流水线各阶段' + (t.pipeline !== null ? '（合计 ' + ui().sig(t.pipeline) + ' 秒）' : '') + '；计算耗时不含排队与人工确认；图例里的浅色斜纹 = 确认出口期间已在后台算好。'), headNote), cls: 'sec-process'}, parts);
  }
  function openingLabels(api, ops, on) {
    var v = api && api.viewer();
    if (!v || typeof v.setToolLabels !== 'function') return;
    try { v.setToolLabels('openings', on ? ops.filter(function (o) { return o.center; }).map(function (o) { return {xyz: o.center, text: '开口 ' + o.i + ' · r ' + ui().num(o.radius, 'mm')}; }) : []); } catch (_) {}
  }
  function inputSection(api) {
    var cur = api.cur(), job = cur.job || {}, a = job.a || job.stage_a || {};
    var ic = a.input_check || (cur.inputcheck && cur.inputcheck.input_check) || null;
    var openings = (a.centerline && a.centerline.openings) || (cur.inputcheck && cur.inputcheck.openings) || [];
    if (!inputModel(ic, openings)) {
      // older records keep the input check only in the stage-A snapshot: the v2 input-check route reads it for any status
      if (!cur.inputcheckLoading && !cur.inputcheck && api.api && api.api().inputcheck) {
        cur.inputcheckLoading = true;
        api.api().inputcheck(cur.jobId).then(function (r) { cur.inputcheck = r || {}; cur.inputcheckLoading = false; if (api.cur() === cur && api.currentTab() === 'tools') api.renderInspector(); },
          function () { cur.inputcheck = {}; cur.inputcheckLoading = false; if (api.cur() === cur && api.currentTab() === 'tools') api.renderInspector(); });
      }
      return ui().section('输入检查', {}, ui().note(cur.inputcheckLoading ? '正在读取…' : '这份结果没有输入检查记录。'));
    }
    var im = inputModel(ic, openings);
    var showOps = im.openings.some(function (o) { return o.center; }) ? ui().button(cur.openingsShown ? '隐藏开口' : '在三维标出开口', function () {
      cur.openingsShown = !cur.openingsShown; openingLabels(api, im.openings, cur.openingsShown); api.renderInspector();
    }, {kind: 'link', cls: 'btn-sm'}) : null;
    return inputSectionFor(ic, openings, cur, {actions: showOps});
  }
  // The 输入检查 section from a stage-A input check; ``holder`` keeps the check list open / closed (P4 lane A #78: the
  // input page of a queued / running / waiting job shows the same section).  null without an input check.
  function inputSectionFor(ic, openings, holder, opts) {
    var h = ui().h, cur = holder || {}, im = inputModel(ic, openings);
    opts = opts || {};
    if (!im) return null;
    var facts = [];
    if (im.units) facts.push('单位 ' + im.units + (im.auto ? '（自动' + (im.confidence !== null ? '，把握度 ' + ui().pct(im.confidence) : '') + '）' : '') + (im.scale !== null && im.scale !== 1 ? ' · × ' + ui().trim(im.scale) : ''));
    if (im.bbox) facts.push('外包 ' + im.bbox.map(function (x) { return ui().sig(x); }).join(' × ') + ' mm');
    if (im.areaCm2 !== null) facts.push('壁面 ' + ui().num(im.areaCm2, 'cm²'));
    if (im.faces !== null) facts.push(ui().sig(im.faces) + ' 面');
    if (im.openings.length) facts.push(im.openings.length + ' 个开口');
    var counts = Object.keys(im.counts).map(function (k) { return im.counts[k] + ' 项' + (CHECK_WORDS[k] || k); }).join(' · ');
    var list = h('ul', {'class': 'checks', hidden: !cur.checksOpen}, im.checks.map(function (c) {
      return h('li', {}, ui().dot(CHECK_TONE[c.status] || 'idle', CHECK_WORDS[c.status] || c.status), h('span', {'class': 'check-label', text: c.label, title: c.note}), c.note ? h('span', {'class': 'muted', text: c.note}) : null);
    }));
    var tlabel = function () { return cur.checksOpen ? '收起' : '全部 ' + im.checks.length + ' 项'; };
    var tg = im.checks.length ? ui().button(tlabel(), function () { cur.checksOpen = !cur.checksOpen; list.hidden = !cur.checksOpen; ui().fill(tg, h('span', {text: tlabel()})); }, {kind: 'link', cls: 'btn-sm'}) : null;
    var tone = {pass: 'ok', pass_with_limits: 'ok', review: 'warn', fail: 'error'}[im.grade] || 'idle';
    return ui().section('输入检查', {actions: opts.actions || null, cls: 'sec-input'},
      im.gradeLabel ? h('div', {'class': 'input-grade'}, ui().dot(tone, im.gradeLabel)) : null,
      facts.length ? h('p', {'class': 'input-facts', text: facts.join(' · ')}) : null,
      im.flags.length ? h('ul', {'class': 'reasons'}, im.flags.map(function (t) { return h('li', {text: t}); })) : null,
      counts ? h('div', {'class': 'sec-actions'}, h('span', {'class': 'muted', text: counts}), h('span', {'class': 'sec-fill'}), tg) : null, list);
  }
  function deleteSection(api) {
    var h = ui().h, cur = api.cur(), job = cur.job || {};
    var running = /^running/.test(job.status || '');
    var btn = ui().button('删除任务…', function () { deleteJob(api); }, {cls: 'btn-sm btn-delete', disabled: locked(api) || running,
      title: locked(api) ? LOCK_TITLE : running ? '计算中的任务不能删除' : null});
    return ui().section('删除', {cls: 'sec-delete'}, h('div', {'class': 'sec-actions'}, btn, h('span', {'class': 'muted', text: '进回收站，30 天内可恢复'})));
  }
  function deleteJob(api) {
    api = remember(api) || shell();
    var cur = api && api.cur();
    if (!cur || !cur.jobId) return null;
    var job = cur.job || {}, id = cur.jobId;
    var name = (cur.manifest && cur.manifest.job && cur.manifest.job.display_name) || job.display_name || job.case_id || id;
    var result = (cur.manifest && cur.manifest.result && cur.manifest.result.display_name) || '';
    return ui().confirm('删除这个结果？', ['「' + name + (result ? ' · ' + result : '') + '」的输入、报告和导出文件会移入回收站，30 天内可以恢复，之后自动清除。'], {confirmLabel: '移入回收站', danger: true}).then(function (ok) {
      if (!ok) return null;
      return api.api().request('/api/jobs/' + encodeURIComponent(id) + '/delete', {method: 'POST', body: {version: job.version}}).then(function (r) {
        var trashed = !r || r.trashed !== false;
        api.refreshJobs();
        if (api.cur() && api.cur().jobId === id) api.go(null);
        ui().toast(trashed ? '已移入回收站。' : '已删除。', {kind: 'ok', actions: trashed ? [{label: '撤销', run: function () { restoreJob(api, id); }}] : []});
        return r;
      }, function (e) { api.conflictOr(e, '没有删除'); return null; });
    });
  }
  function restoreJob(api, id) {
    return api.api().request('/api/trash/' + encodeURIComponent(id) + '/restore', {method: 'POST', body: {}}).then(function () {
      ui().toast('已恢复。', {kind: 'ok', ms: 3000});
      api.refreshJobs(); api.go(id);
    }, function (e) { ui().toast('没有恢复：' + (e && e.message || e), {kind: 'error'}); });
  }
  function toolsSections(api) {
    var cur = api.cur();
    if (api.offline() || !cur || !cur.jobId || !cur.manifest) return [];
    if (!cur.outletEdit) outletLabels(api, null);
    var out = [];
    [caseSection, outletSection, processSection, inputSection, deleteSection].forEach(function (fn) {
      try { var s = fn(api); if (s) out.push(s); } catch (e) { if (root.console && root.console.error) root.console.error('detail tools: ' + (e && e.stack || e)); }
    });
    return out;
  }

  // ------------------------------------------------------------------ keys (N / / / O)
  function onKey(e, api) {
    var k = e && e.key;
    if (k === 'n' || k === 'N') {
      if (api.offline() || !api.openUpload) return false;
      api.openUpload(); return true;
    }
    if (k === '/') {
      if (api.offline() || !api.focusSearch) return false;
      return api.focusSearch() !== false;
    }
    if (k === 'o' || k === 'O') {
      var cur = api.cur();
      if (api.offline() || !cur || !cur.manifest || !cur.jobId) return false;
      root.open(api.api().urls.onepage(cur.jobId), '_blank', 'noopener');
      return true;
    }
    return false;
  }

  // ------------------------------------------------------------------ registration
  var EXT = {
    id: 'detail',
    tabs: function (api) {
      remember(api);
      var cur = api.cur(), st = api.state ? api.state() : null;
      // leaving the tab (no currentTab() here: it asks this hook): the faded wall of a bin highlight goes too
      var tab = st ? (st.tab || (api.store().prefs() || {}).tab) : null;
      if (cur && cur.along && cur.along.pin && tab !== 'along') { cur.along.pin = null; clearHighlight(api); }
      return alongAvailable(cur) ? [{id: 'along', label: '沿程'}] : [];
    },
    renderTab: function (tabId, body, api) {
      remember(api);
      if (tabId !== 'along') return false;
      if (!alongAvailable(api.cur())) { ui().fill(body, ui().empty('这份结果没有沿程数据。')); return true; }
      renderAlong(body, api);
      return true;
    },
    tools: function (api) { remember(api); return toolsSections(api); },
    key: function (e, api) { remember(api); return onKey(e, api); },
    onResult: function (api) { remember(api); },
    onClose: function (api) {
      remember(api);
      var cur = api.cur();
      if (cur) { cur.outletEdit = null; cur.openingsShown = false; if (cur.along) cur.along.view = null; }
      outletLabels(api, null); openingLabels(api, [], false);
    },
    onField: function (api) {
      remember(api);
      var cur = api.cur();
      if (cur && cur.along) cur.along.pick = null;
      if (api.currentTab() === 'along') api.renderInspector();
    }
  };
  (ns.ext = ns.ext || []).push(EXT);

  return {shell: shell, ext: EXT, metadataDialog: metadataDialog, deleteJob: deleteJob, restoreJob: restoreJob, renderAlong: renderAlong, alongAvailable: alongAvailable,
    timingModel: timingModel, eventRows: eventRows, eventText: eventText, parseTags: parseTags, tagsText: tagsText, metadataPayload: metadataPayload,
    identifierIssue: identifierIssue, metadataError: metadataError, swapMapping: swapMapping, validMapping: validMapping, sameMapping: sameMapping,
    outletModel: outletModel, inputModel: inputModel, onKey: onKey, toolsSections: toolsSections, OUTLET_NAMES: OUTLET_NAMES,
    // P4 lane A
    confidenceCaveat: confidenceCaveat, outletFlags: outletFlags, centerlineRows: centerlineRows, processSectionFor: processSectionFor, inputSectionFor: inputSectionFor,
    // P3 lane 3
    techRows: techRows, techDialog: techDialog, stampText: stampText, canSliceAt: canSliceAt, sliceStation: sliceStation, pressureDropSlice: pressureDropSlice};
});
