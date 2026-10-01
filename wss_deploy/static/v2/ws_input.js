/* WSS workspace v2 — input page (contract §6.3 ws_input.js): progress, unit check, outlet confirmation in the
 * main viewport (O4, same /confirm payload and swap semantics as the classic workbench), and the failure page
 * with every opening drawn and numbered in 3-D (U2).  Hints only; nothing is repaired automatically.
 *
 * Phase 3 lane 1 (PHASE3_LANES.md §3, audit rows #65 #68–#70 #72 #74–#77) brings the rest of the classic job page
 * (app.js renderJob / cancelJob / etaWidget / inputCard / outletCard / offerNext): stage table and live remaining
 * time (the rule of workbench_core.etaView, ported because that file leaves with S7), cancel while waiting for a
 * person, retry of failed jobs with the typed error card (category, CPU hint, diagnostic id, admin-only detail),
 * input facts, re-picking the inlet (/confirm {inlet}), outlet details (plain-words reasons, 3-D pick, changes,
 * remaining time), 「下一例」 after confirming or signing, and where a job came from (companions / reuse / rerun).
 * Every request is the classic one with the classic payload.
 * P4 lane A: the outlet viewer draws the centreline (#73, classic createViewer); 「输入检查与计算过程」 on queued / running /
 * waiting pages (#78, ws_detail's sections with the centreline rows #102); the naming confidence's caveat (W30) and the
 * summary flags among the outlet reasons (W35). */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.input = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };
  var api = function () { return ns.api; };

  var NAMES = {inlet: '主动脉入口', 'out-le': '左髂外', 'out-li': '左髂内', 'out-re': '右髂外', 'out-ri': '右髂内'};
  var COLORS = {inlet: '#2569aa', 'out-le': '#0e6b5c', 'out-li': '#6f9f3c', 'out-re': '#c07a2c', 'out-ri': '#8f5ca3', none: '#5f6b76', small: '#9c3326', pick: '#0e6e8a'};
  var SWAPS = {
    lr: {'out-le': 'out-re', 'out-li': 'out-ri', 'out-re': 'out-le', 'out-ri': 'out-li'},
    left: {'out-le': 'out-li', 'out-li': 'out-le'},
    right: {'out-re': 'out-ri', 'out-ri': 'out-re'}
  };
  function swap(mapping, kind) {
    var pairs = SWAPS[kind] || {};
    var out = {};
    Object.keys(mapping || {}).forEach(function (k) { out[k] = pairs[mapping[k]] || mapping[k]; });
    return out;
  }
  function validMapping(mapping, outletIds) {
    var names = outletIds.map(function (id) { return mapping[id]; });
    var uniq = {};
    names.forEach(function (n) { uniq[n] = true; });
    return names.length === 4 && Object.keys(uniq).length === 4 && names.every(function (n) { return n && n !== 'inlet' && NAMES[n]; });
  }
  // Openings from any source → [{id, center, radius}]
  function normalizeOpenings(list) {
    return (Array.isArray(list) ? list : []).map(function (o, i) {
      if (!o) return null;
      var id = o.opening_id !== undefined ? o.opening_id : o.opening_index !== undefined ? o.opening_index : o.id !== undefined ? o.id : i;
      var r = o.radius_mm !== undefined ? o.radius_mm : o.equivalent_radius_mm !== undefined ? o.equivalent_radius_mm : o.r_mm;
      var c = o.center_mm || o.center || o.centroid_mm || null;
      return {id: id, radius: Number(r), center: c, role: o.role || null};
    }).filter(function (o) { return o && Array.isArray(o.center) && o.center.length >= 3; });
  }
  // U2: flag openings that are clearly smaller than the rest; with more than five openings the extra smallest ones.
  function openingHints(openings) {
    var list = normalizeOpenings(openings);
    var radii = list.map(function (o) { return o.radius; }).filter(function (r) { return isFinite(r); }).sort(function (a, b) { return a - b; });
    var median = radii.length ? radii[Math.floor(radii.length / 2)] : null;
    var extra = Math.max(0, list.length - 5);
    var smallest = list.slice().filter(function (o) { return isFinite(o.radius); }).sort(function (a, b) { return a.radius - b.radius; }).slice(0, extra).map(function (o) { return o.id; });
    return list.map(function (o) {
      var small = isFinite(o.radius) && (o.radius < 1 || (median && o.radius < 0.35 * median) || smallest.indexOf(o.id) >= 0);
      return {id: o.id, radius: o.radius, center: o.center, small: Boolean(small)};
    });
  }
  function b64ToTyped(b64, Ctor) {
    if (!b64 || typeof b64 !== 'string' || typeof root.atob !== 'function') return null;
    var bin = root.atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    if (bytes.length % Ctor.BYTES_PER_ELEMENT) return null;
    return new Ctor(bytes.buffer);
  }
  function meshFrom(inputcheck, geometry) {
    if (inputcheck && inputcheck.mesh && inputcheck.mesh.vertices) {
      var v = b64ToTyped(inputcheck.mesh.vertices, Float32Array), f = b64ToTyped(inputcheck.mesh.faces, Uint32Array);
      if (v && f) return {vertices: v, faces: f};
    }
    var p = geometry && geometry.preview;
    if (p && Array.isArray(p.vertices) && Array.isArray(p.faces) && p.vertices.length) {
      var vv = new Float32Array(p.vertices.length * 3), ff = new Uint32Array(p.faces.length * 3);
      p.vertices.forEach(function (x, i) { vv[3 * i] = x[0]; vv[3 * i + 1] = x[1]; vv[3 * i + 2] = x[2]; });
      p.faces.forEach(function (x, i) { ff[3 * i] = x[0]; ff[3 * i + 1] = x[1]; ff[3 * i + 2] = x[2]; });
      return {vertices: vv, faces: ff};
    }
    return null;
  }
  // P4 lane A (#73): centreline polylines of the outlet viewer, as the classic createViewer: the /geometry payload's
  // preview_polylines, else the stage-A proposal's; each line = its finite xyz points (two or more).
  function polylinesFrom(geometry, proposal) {
    var g = geometry && Array.isArray(geometry.preview_polylines) && geometry.preview_polylines.length ? geometry.preview_polylines : null;
    var src = g || (proposal && Array.isArray(proposal.preview_polylines) ? proposal.preview_polylines : []);
    var fin = function (v) { return typeof v === 'number' && isFinite(v); };
    var ok = function (p) { return Array.isArray(p) && p.length >= 3 && fin(p[0]) && fin(p[1]) && fin(p[2]); };
    return src.map(function (line) {
      return line && Array.isArray(line.xyz) ? line.xyz.filter(ok).map(function (p) { return [p[0], p[1], p[2]]; }) : [];
    }).filter(function (pts) { return pts.length > 1; });
  }

  // ------------------------------------------------------------------ pure: status rules (classic app.js renderJob / cancelJob)
  var ACTIVE = ['queued', 'queued_A', 'queued_B', 'running', 'running_A', 'running_B'];
  var AWAITING = ['awaiting_input', 'awaiting_confirmation', 'awaiting_outlets'];
  var RECOVER = ['failed', 'cancelled', 'interrupted'];
  function numOrNull(value) {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
    var n = Number(value);
    return isFinite(n) ? n : null;
  }
  function isLocked(job) { return Boolean(job && job.review && job.review.status === 'reviewed'); }
  // The service refuses a second cancel (jobs.py cancel): a job already asked to stop shows no button.
  function canCancel(job) {
    return Boolean(job) && !job.cancel_requested && job.status !== 'cancelling' && (ACTIVE.indexOf(job.status) >= 0 || AWAITING.indexOf(job.status) >= 0);
  }
  function cancelLines(job) {
    var running = /^running/.test((job && job.status) || '');
    return [running ? '正在执行的步骤会在安全停止点结束。' : '任务会停止，不再排队或等待确认。',
      '已上传的 STL、中心线和已确认的出口都会保留；之后点「重试」可以从可用的阶段继续，不必重新上传。'];
  }
  // /input accepts a failed or interrupted stage-A job whose check still waits for the units (jobs.py input_recovery).
  function inputRecovery(job) {
    var ic = job && (job.a || job.stage_a) && (job.a || job.stage_a).input_check;
    return Boolean(job && (job.status === 'failed' || job.status === 'interrupted') && ic && ic.status === 'needs_confirmation');
  }

  // ------------------------------------------------------------------ pure: typed error card (classic ERROR_TITLES / ERROR_HINTS, v0.14 F9)
  var ERROR_TITLES = {input_geometry: '输入几何无法计算', toolchain: '计算工具出错', resource: '计算资源不足', internal: '本次计算未完成'};
  var ERROR_HINTS = {
    input_geometry: '问题出在上传的 STL 本身，重试不会改变结果：请按提示修正几何后重新上传。',
    toolchain: '中心线提取等计算工具没有正常完成。可以稍后重试；若仍失败，请把诊断编号发给维护者。',
    resource: '显存、内存或磁盘暂时不足，通常稍后重试即可完成。',
    internal: '服务内部错误。请把诊断编号发给维护者。'};
  function errorModel(job, fallback) {
    job = job || {};
    var status = job.status || '';
    var raw = job.error || fallback || null;
    var rec = raw && typeof raw === 'object' ? raw : null;
    var message = rec ? String(rec.message || '') : (typeof raw === 'string' ? raw : '');
    if (!message && status !== 'cancelled') message = typeof job.detail === 'string' && job.detail ? job.detail : '计算没有完成。';
    var noRetry = Boolean(rec && rec.retryable === false);
    var title = status === 'interrupted' ? '任务中断，可恢复' : status === 'cancelled' ? '已取消' : (ERROR_TITLES[rec && rec.category] || '本次计算未完成');
    var hint = ERROR_HINTS[rec && rec.category] || (status === 'cancelled' ? '任务已停止。恢复时会重新检查输入和可复用的中间结果。' : '检查下方输入信息后重试。已有的有效中心线与特征按服务端校验结果复用。');
    return {title: title, message: message, hint: hint, category: (rec && rec.category) || null, cpu: Boolean(rec && rec.retry_hint === 'cpu'),
      diagnostic: rec && rec.diagnostic_id ? String(rec.diagnostic_id) : null, adminDetail: rec && rec.admin_detail ? String(rec.admin_detail) : null,
      retryable: !noRetry, locked: isLocked(job), canRetry: RECOVER.indexOf(status) >= 0 && !isLocked(job) && !noRetry,
      retryLabel: status === 'interrupted' ? '恢复任务' : '重试',
      noRetryText: noRetry ? '这个错误重试不会得到不同的结果：请修正输入后重新上传，或把诊断编号发给维护者。' : null};
  }

  // ------------------------------------------------------------------ pure: remaining time (the rule of workbench_core.etaView, §21.2 / §21.3)
  // "几秒" below 10 s, 5-second steps below a minute, then "约 N 分 M 秒" (whole minutes from 10 min on).
  function formatDuration(seconds) {
    var s = numOrNull(seconds);
    if (s === null) return '';
    if (s < 10) return '几秒';
    if (s < 57.5) return '约 ' + Math.max(10, Math.round(s / 5) * 5) + ' 秒';
    if (s >= 570) return '约 ' + Math.round(s / 60) + ' 分钟';
    var total = Math.round(s / 5) * 5, minutes = Math.floor(total / 60), rest = total % 60;
    return rest ? '约 ' + minutes + ' 分 ' + rest + ' 秒' : '约 ' + minutes + ' 分钟';
  }
  // Same rule as eta.stage_remaining: expected − elapsed until 80 %, then a shrinking 20 % tail, never negative.
  function stageRemaining(expected, elapsed) {
    var e = Math.max(0, numOrNull(expected) || 0), t = Math.max(0, numOrNull(elapsed) || 0), knee = 0.8 * e;
    if (t <= knee || e <= 0) return Math.max(0, e - t);
    return 0.2 * e * (knee / t);
  }
  function secondsText(value) { var v = numOrNull(value); return v === null ? '—' : (v < 10 ? v.toFixed(1) : String(Math.round(v))) + ' 秒'; }
  // ``ageS`` = seconds since the snapshot arrived: the running stage keeps advancing between server updates (browser clock).
  function etaView(eta, opts) {
    if (!eta || typeof eta !== 'object' || !Array.isArray(eta.stages) || !eta.stages.length) return null;
    opts = opts || {};
    var age = Math.max(0, numOrNull(opts.ageS) || 0);
    var minPct = numOrNull(opts.minPct) !== null ? numOrNull(opts.minPct) : 2.5;
    var pre = {};
    (Array.isArray(eta.precomputed) ? eta.precomputed : []).forEach(function (k) { pre[String(k)] = true; });
    var rows = eta.stages.filter(function (s) { return s && typeof s === 'object'; }).map(function (stage) {
      var expected = Math.max(0, numOrNull(stage.expected_s) || 0);
      var state = ['done', 'running', 'pending'].indexOf(stage.state) >= 0 ? stage.state : 'pending';
      var elapsed = numOrNull(stage.elapsed_s);
      if (state === 'running') elapsed = (elapsed || 0) + age;
      return {key: String(stage.key || ''), label: String(stage.label || stage.key || '阶段'), expected: expected, elapsed: elapsed, state: state,
        overdue: state === 'running' && elapsed > expected && expected > 0, precomputed: Boolean(pre[String(stage.key || '')])};
    });
    // ``waiting``: the job holds a worker but waits for the previous stage-B job (serialised) → shown as queued.
    var rawStatus = eta.status || (rows.some(function (r) { return r.state === 'running'; }) ? 'running' : null);
    var waiting = eta.waiting === true && rawStatus === 'running';
    var status = waiting ? 'queued' : rawStatus;
    var total = rows.reduce(function (s, r) { return s + r.expected; }, 0);
    var widths = rows.map(function (r) { return total > 0 ? r.expected / total * 100 : 100 / rows.length; }).map(function (w) { return Math.max(minPct, w); });
    var scale = 100 / widths.reduce(function (s, w) { return s + w; }, 0);
    var filled = 0, currentIndex = -1;
    var segments = rows.map(function (row, index) {
      var width = +(widths[index] * scale).toFixed(3);
      var fill = row.state === 'done' ? 100 : 0;
      if (row.state === 'running' && !waiting) { currentIndex = index; fill = row.expected > 0 ? Math.min(96, row.elapsed / row.expected * 100) : 50; }
      filled += width * fill / 100;
      var title = row.state === 'done'
        ? row.label + '：已完成' + (row.elapsed !== null ? '，用时 ' + secondsText(row.elapsed) : '') + '（预计 ' + secondsText(row.expected) + '）'
        : row.state === 'running'
          ? row.label + '：进行中，已用 ' + secondsText(row.elapsed) + ' / 预计 ' + secondsText(row.expected) + (row.overdue ? '，比预计慢' : '')
          : row.label + '：预计 ' + secondsText(row.expected);
      return {key: row.key, label: row.label, expected: row.expected, elapsed: row.elapsed, overdue: row.overdue, precomputed: row.precomputed,
        state: waiting && row.state === 'running' ? 'pending' : row.state, width: width, fill: +fill.toFixed(1), title: row.precomputed ? title + '（已预计算）' : title};
    });
    var remaining;
    if (status === 'running' && !waiting) remaining = rows.reduce(function (s, r) { return s + (r.state === 'running' ? stageRemaining(r.expected, r.elapsed) : r.state === 'pending' ? r.expected : 0); }, 0);
    else remaining = numOrNull(eta.remaining_s);
    var segmentRemaining = numOrNull(eta.segment_remaining_s);
    var queueAhead = numOrNull(eta.queue_ahead), queueWait = numOrNull(eta.queue_ahead_s);
    var n = rows.length, current = currentIndex >= 0 ? segments[currentIndex] : null;
    var headline = '', sub = '';
    var manual = eta.segment === 'A' ? '，不含人工确认出口' : '';
    if (status === 'queued') {
      var wait = queueWait === null ? null : Math.max(0, queueWait - age);
      headline = queueAhead ? '前面 ' + queueAhead + ' 个，' + (wait === null ? '稍后' : formatDuration(wait) + '后') + '开始' : waiting ? '等前一个任务算完后开始' : '即将开始';
      sub = remaining !== null ? '预计' + formatDuration((wait || 0) + remaining) + '后出结果（含排队' + manual + '）' : '';
    } else if (status === 'running') {
      var overdue = currentIndex >= 0 && segments[currentIndex].overdue;
      headline = overdue && remaining < 10 ? '即将完成' : '预计还需' + formatDuration(remaining) + (manual ? '（' + manual.slice(1) + '）' : '');
      sub = current ? '正在：' + current.label + '（第 ' + (currentIndex + 1) + ' / ' + n + ' 步）' + (current.overdue ? ' · 比预计慢' : '') : '';
    } else if (status === 'awaiting_confirmation') {
      headline = remaining !== null ? '确认后' + formatDuration(remaining) + '出结果' : '';
    } else if (status === 'awaiting_input') {
      var a = segmentRemaining !== null ? segmentRemaining : remaining;
      headline = a !== null ? '确认后' + formatDuration(a) + '完成中心线提取' : '';
    }
    var history = numOrNull(eta.n_history), faces = numOrNull(eta.faces);
    var basis = eta.basis === 'history' ? '按 ' + (history || 0) + ' 例同类历史耗时估计' : '按默认耗时估计（同类历史少于 3 例）';
    return {status: status, waiting: waiting, segments: segments, currentIndex: currentIndex, remaining: remaining, progress: Math.round(Math.min(100, filled)),
      headline: headline, sub: sub, basis: faces ? basis + ' · 输入 ' + Math.round(faces).toLocaleString('en-US') + ' 个面片' : basis, queueAhead: queueAhead, queueWait: queueWait};
  }

  // ------------------------------------------------------------------ pure: outlet reasons in plain words (workbench_core.outletReview, C3)
  var OUTLET_REASONS = [
    [/^(左|右)侧髂内\s*\/\s*髂外区分置信度\s*([\d.]+)%/, function (m) { return m[1] + '侧髂内与髂外的区分把握度只有 ' + Math.round(Number(m[2])) + '%，请重点核对' + m[1] + '侧的两个出口。'; }],
    [/几何置信度代理\s*([\d.]+)%\s*低于门槛\s*([\d.]+)%/, function (m) { return '整体命名把握度 ' + Math.round(Number(m[1])) + '%，低于自动放行要求的 ' + Math.round(Number(m[2])) + '%。'; }],
    [/代理值|未校准/, function () { return '把握度由几何形态估算，不是经过临床数据校准的概率。'; }],
    [/髂内动脉朝向腹侧|左右解剖朝向检查不一致/, function () { return '按坐标判断的左右与按解剖朝向（髂内动脉向后）判断的不一致：STL 可能来自另一种坐标约定或是镜像，请对照原始影像核对左右。'; }],
    [/交叉判定/, function () { return 'STL 不带患者方位信息，左右由坐标方向与解剖朝向两种方法交叉判定。'; }],
    [/患者方向|世界\s*XYZ|左右语义/, function () { return 'STL 文件不带患者方位信息，左右需要对照原始影像确认。'; }],
    [/计算方法与 profile|与 profile 一致的左右/, function () { return '出口命名的算法版本与校准时不同，这一例请人工确认。'; }],
    [/命名建议自身标记为需要确认/, function () { return '自动命名在至少一侧的内 / 外区分上不够确定。'; }],
    [/profile|联合正确率/, function () { return '当前模型版本尚未完成出口命名的独立验证，因此每例都请人工确认。'; }]
  ];
  function humanOutletReason(text) {
    var raw = String(text || '').trim();
    if (!raw) return '';
    for (var i = 0; i < OUTLET_REASONS.length; i++) { var m = OUTLET_REASONS[i][0].exec(raw); if (m) return OUTLET_REASONS[i][1](m); }
    return raw;
  }
  function outletReview(proposal, jobDetail, fallbackNote) {
    var p = proposal && typeof proposal === 'object' ? proposal : {};
    var gate = p.confidence_gate && typeof p.confidence_gate === 'object' ? p.confidence_gate : {};
    var detailReasons = typeof jobDetail === 'string' && jobDetail.indexOf('原因：') >= 0 ? jobDetail.split('原因：').slice(1).join('').split(/[；;]/) : [];
    var raw = [].concat(p.confidence_reasons || [], gate.reasons || [], p.flags || [], p.direction_note ? [p.direction_note] : [], detailReasons);
    var reasons = [];
    raw.forEach(function (t) { var human = humanOutletReason(t); if (human && reasons.indexOf(human) < 0) reasons.push(human); });
    if (!reasons.length && fallbackNote) reasons.push(humanOutletReason(fallbackNote));
    var confidence = numOrNull(p.confidence);
    var required = p.confirmation_required !== false;
    var pct = confidence === null ? null : Math.round(confidence * 100);
    var headline = pct === null ? '请结合原始影像核对出口命名' : required ? '自动命名置信度 ' + pct + '%，需要人工核对' : '自动命名置信度 ' + pct + '%，已自动通过';
    return {headline: headline, confidence: confidence, required: required, reasons: reasons};
  }
  // P4 lane A (W30): the caveat beside 「自动命名置信度 x%」 — ws_detail's rule (classic report gate card), else the
  // classic workbench sentence for a proxy value.
  function confidenceCaveat(gate) {
    var D = ns.detail;
    return D && typeof D.confidenceCaveat === 'function' ? D.confidenceCaveat(gate) : humanOutletReason('代理值');
  }
  function mappingChanges(mapping, proposal) {
    return Object.keys(mapping || {}).filter(function (k) { return mapping[k] !== (proposal || {})[k]; }).sort(function (a, b) { return Number(a) - Number(b); });
  }

  // ------------------------------------------------------------------ pure: input facts (classic app.js inputFacts / boxSize)
  function boxSize(ic, raw) {
    var direct = raw ? ic.raw_bbox_size : ic.bbox_size_mm;
    if (Array.isArray(direct)) return direct;
    var box = raw ? ic.raw_bbox : ic.bbox_mm;
    if (Array.isArray(box) && Array.isArray(box[0])) return box[1].map(function (v, i) { return v - box[0][i]; });
    return Array.isArray(box) && box.length === 3 ? box : null;
  }
  function inputFacts(ic) {
    ic = ic && typeof ic === 'object' ? ic : {};
    var dims = function (s) { return s && s.length === 3 ? s.map(function (v) { return ui().sig(v); }).join(' × ') : '—'; };
    var removed = Number(ic.removed_area_fraction || 0);
    var mm = dims(boxSize(ic, false));
    var area = numOrNull(ic.area_mm2), verts = numOrNull(ic.vertices);
    return [
      {key: 'raw', label: '原始尺寸', value: dims(boxSize(ic, true)), note: 'STL 原坐标值 X × Y × Z'},
      {key: 'mm', label: '换算后', value: mm === '—' ? mm : mm + ' mm', note: '请与原始影像的尺度核对'},
      {key: 'openings', label: '开口数', value: ic.openings === undefined || ic.openings === null ? '—' : String(ic.openings), note: '要求 1 个入口和 4 个出口', warn: numOrNull(ic.openings) !== null && Number(ic.openings) !== 5},
      {key: 'area', label: '壁面面积', value: (area === null ? '—' : ui().num(area / 100, 'cm²')) + (verts === null ? '' : ' · ' + Math.round(verts) + ' 个顶点')},
      {key: 'components', label: '连通片', value: ic.components === undefined || ic.components === null ? '—' : String(ic.components), note: '确认没有遗漏血管分支'},
      {key: 'removed', label: '拟删除面积', value: (removed * 100).toFixed(3) + '%', note: removed > 0 ? '相对于输入壁面总面积；超过 1% 需要修复原始几何' : '无需删除孤立片'}
    ];
  }
  // Classic inputCard reference card: the geometry range and the population position of the release, when the job has them.
  function referenceText(summary) {
    var ref = summary && summary.reference_assessment;
    if (!ref || typeof ref !== 'object') return null;
    var pop = ref.population || {};
    if (!ref.status && !pop.status) return null;
    var geo = ref.status === 'pass' ? '在已声明几何参考范围内' : ref.status === 'review' ? '超出已声明范围，请复核' : '未配置几何参考范围';
    var p = pop.status === 'pass' && numOrNull(pop.percentile) !== null ? '同协议经验分位 ' + Number(pop.percentile).toFixed(1) + '%' : '未配置可核验的人群参照';
    return {text: '几何：' + geo + ' · 人群：' + p, review: ref.status === 'review'};
  }

  // ------------------------------------------------------------------ pure: provenance (classic renderJob job-origin lines, #77)
  function provenance(job, cards, jobs) {
    job = job || {};
    var list = Array.isArray(jobs) ? jobs : [];
    var nameOf = function (id, releaseId) {
      var j = list.filter(function (x) { return x && x.id === id; })[0];
      if (j) return ui().resultName(j, cards);
      return releaseId ? ui().resultName({model_release: {id: releaseId}}, cards) : '另一份结果';
    };
    var out = [];
    if (job.companion_of) out.push({kind: 'companion_of', text: '与「' + nameOf(job.companion_of) + '」同时上传，沿用它的中心线和出口确认', jobId: job.companion_of, action: '打开'});
    (Array.isArray(job.companions) ? job.companions : []).filter(function (c) { return c && c.release_id; }).forEach(function (c) {
      var name = nameOf(c.job_id, c.release_id);
      if (c.job_id) out.push({kind: 'companion', text: '同时预测「' + name + '」', jobId: c.job_id, action: '打开'});
      else out.push({kind: 'companion', text: '同时预测「' + name + '」：' + (c.error ? '未能创建（' + c.error + '）' : '确认出口后自动创建'), warn: Boolean(c.error)});
    });
    if (!job.companion_of) {
      if (job.reused_from) out.push({kind: 'reused', text: '复用已有结果的中心线和出口确认，只重新预测', jobId: job.reused_from, action: '打开来源'});
      else if (job.source_job_id) out.push({kind: 'rerun', text: '换模型重跑，沿用来源结果的中心线和出口确认', jobId: job.source_job_id, action: '打开来源'});
    }
    return out;
  }
  function provenanceBlock(lines, onOpen) {
    var h = ui().h;
    if (!lines || !lines.length) return null;
    return h('ul', {'class': 'prov-list'}, lines.map(function (l) {
      return h('li', {'class': l.warn ? 'warn-text' : null}, h('span', {text: l.text}),
        l.jobId && onOpen ? ui().button(l.action || '打开', function () { onOpen(l.jobId); }, {kind: 'link', cls: 'btn-sm'}) : null);
    }));
  }

  // ------------------------------------------------------------------ 「下一例」 after confirming outlets or signing (classic offerNext, A7)
  var CONFIRM_STATUSES = ['awaiting_confirmation', 'awaiting_outlets', 'awaiting_input'];
  function timeMs(value) { var ms = Date.parse(value || ''); return isFinite(ms) ? ms : null; }
  // Oldest waiting first: a batch is worked through in upload order, whatever the list's sort (workbench_core.nextJob).
  function nextJob(jobs, currentId, kind) {
    var pred = kind === 'review'
      ? function (j) { return j.status === 'done' && !(j.review && j.review.status === 'reviewed'); }
      : function (j) { return CONFIRM_STATUSES.indexOf(j.status) >= 0; };
    var list = (jobs || []).filter(function (j) { return j && typeof j === 'object' && j.id && j.id !== currentId && pred(j); });
    list.sort(function (a, b) { return ((timeMs(a.created_at) || 0) - (timeMs(b.created_at) || 0)) || String(a.id).localeCompare(String(b.id)); });
    return list[0] || null;
  }
  function offerNext(kind, doneId, opts) {
    opts = opts || {};
    var go = opts.go || (ns.shell && ns.shell.go);
    var fallback = opts.jobs || (ns.shell && ns.shell.state && ns.shell.state() && ns.shell.state().jobs) || [];
    var all = ns.admin && typeof ns.admin.viewAll === 'function' && ns.admin.viewAll();
    var done = kind === 'review' ? '技术复核已记录，结果已锁定。' : '出口已确认，开始预测。';
    return Promise.resolve(api().jobs({page: 1, all: all ? 1 : null})).then(function (r) {
      return Array.isArray(r) ? r : (r && Array.isArray(r.jobs) ? r.jobs : fallback);
    }, function () { return fallback; }).then(function (jobs) {
      var next = nextJob(jobs, doneId, kind);
      if (!next) { ui().toast(done + (kind === 'review' ? '没有其他未复核的结果。' : '没有其他等待确认的任务。'), {kind: 'ok'}); return null; }
      ui().toast(done + '还有' + (kind === 'review' ? '未复核的结果' : '等待确认的任务') + '：' + ui().displayName(next) + '。', {kind: 'ok',
        actions: go ? [{label: kind === 'review' ? '下一例待复核' : '处理下一例', run: function () { go(next.id); }}] : []});
      return next;
    });
  }
  // The technical review is signed in the shell's dialog (lane 3 may move it): watching the service call catches an
  // approval wherever it is made, without touching that dialog.  Same response, same promise; the offer follows the
  // dialog's own confirmation.
  function watchReview() {
    var A = ns.api;
    if (!A || typeof A.review !== 'function' || A.review.__p3Next) return false;
    var orig = A.review;
    var wrapped = function (id, payload) {
      var p = orig.apply(A, arguments);
      if (!payload || payload.decision !== 'approve' || !p || typeof p.then !== 'function') return p;
      return p.then(function (r) {
        setTimeout(function () { try { offerNext('review', id); } catch (_) {} }, 0);
        return r;
      });
    };
    wrapped.__p3Next = true;
    A.review = wrapped;
    return true;
  }

  // ------------------------------------------------------------------ small 3-D view: mesh + numbered markers (clickable)
  function meshView(container) {
    var T = root.THREE;
    if (!T || !T.WebGLRenderer || !container) return null;
    var renderer;
    try { renderer = new T.WebGLRenderer({antialias: true, alpha: true, preserveDrawingBuffer: true}); } catch (_) { return null; }
    var disposables = [];
    renderer.setPixelRatio(Math.min(root.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);   // the stage behind paints the background
    var canvas = renderer.domElement;
    canvas.className = 'mv-canvas';
    container.appendChild(canvas);
    var scene = new T.Scene();
    var camera = new T.PerspectiveCamera(35, 1, 0.1, 10000);
    scene.add(new T.HemisphereLight(0xffffff, 0x8a93a3, 0.6));
    var key = new T.DirectionalLight(0xffffff, 0.62);
    camera.add(key); key.position.set(0.3, 0.6, 1);
    scene.add(camera);
    var group = new T.Group(); scene.add(group);
    var lineGroup = new T.Group(); scene.add(lineGroup);
    var markerGroup = new T.Group(); scene.add(markerGroup);
    var controls = T.OrbitControls ? new T.OrbitControls(camera, canvas) : null;
    var radius = 100, frame = 0, home = null, pickCb = null, targets = [];
    function draw() { frame = 0; renderer.render(scene, camera); }
    function request() { if (!frame) frame = (root.requestAnimationFrame || setTimeout)(draw); }
    if (controls) { controls.enableDamping = false; controls.addEventListener('change', request); }
    function size() {
      var w = container.clientWidth || 600, h = container.clientHeight || 400;
      renderer.setSize(w, h, false); canvas.style.width = '100%'; canvas.style.height = '100%';
      camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix(); request();
    }
    var ro = typeof root.ResizeObserver === 'function' ? new root.ResizeObserver(size) : null;
    if (ro) ro.observe(container); else root.addEventListener('resize', size);
    function applyHome() {
      if (!home) return;
      camera.position.copy(home.position); camera.up.set(0, 0, 1);
      if (controls) controls.target.copy(home.target);
      camera.lookAt(home.target); camera.updateProjectionMatrix();
      if (controls) controls.update();
      request();
    }
    // Framing on a bounding sphere (the wall's, or the centreline's when the job has no display mesh).
    function frameOn(s) {
      radius = s.radius || 100;
      var vfov = camera.fov * Math.PI / 180, hfov = 2 * Math.atan(Math.tan(vfov / 2) * (camera.aspect || 1));
      var dist = radius / Math.sin(Math.min(vfov, hfov) / 2) * 1.02;
      camera.near = radius / 100; camera.far = radius * 20;
      home = {position: new T.Vector3(s.center.x, s.center.y - dist, s.center.z), target: s.center.clone()};
      applyHome();
    }
    var hasMesh = false;
    function setMesh(mesh) {
      while (group.children.length) group.remove(group.children[0]);
      hasMesh = Boolean(mesh);
      if (!mesh) { request(); return; }
      var g = new T.BufferGeometry();
      g.setAttribute('position', new T.BufferAttribute(mesh.vertices, 3));
      g.setIndex(new T.BufferAttribute(mesh.faces, 1));
      g.computeVertexNormals(); g.computeBoundingSphere();
      var mat = new T.MeshPhongMaterial({color: 0xdfe4ea, specular: 0x333333, shininess: 30, side: T.DoubleSide});
      disposables.push(g, mat);
      group.add(new T.Mesh(g, mat));
      frameOn(g.boundingSphere);
    }
    // P4 lane A (#73): the centreline polylines ([[x, y, z], …] each) over the wall, colour and see-through drawing of
    // the classic outlet viewer (0x567891, opacity 0.9, no depth test); under the endpoint markers.
    function setLines(lines) {
      while (lineGroup.children.length) lineGroup.remove(lineGroup.children[0]);
      var all = [];
      (lines || []).forEach(function (pts) {
        if (!Array.isArray(pts) || pts.length < 2) return;
        var vecs = pts.map(function (p) { return new T.Vector3(p[0], p[1], p[2]); });
        var g = new T.BufferGeometry().setFromPoints(vecs);
        var mat = new T.LineBasicMaterial({color: 0x567891, transparent: true, opacity: 0.9, depthTest: false});
        disposables.push(g, mat);
        var line = new T.Line(g, mat);
        line.renderOrder = 3;
        lineGroup.add(line);
        all = all.concat(vecs);
      });
      if (!hasMesh && all.length && T.Sphere) frameOn(new T.Sphere().setFromPoints(all));
      request();
      return lineGroup.children.length;
    }
    function pill(text, color) {
      var c = root.document.createElement('canvas');
      var ctx = c.getContext && c.getContext('2d');
      if (!ctx) return null;
      var font = '600 28px system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif';
      ctx.font = font;
      c.width = Math.ceil(ctx.measureText(text).width + 28); c.height = 44;
      ctx.font = font;
      ctx.fillStyle = '#ffffff'; ctx.strokeStyle = color; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.rect(1.5, 1.5, c.width - 3, c.height - 3); ctx.fill(); ctx.stroke();
      ctx.fillStyle = color; ctx.textBaseline = 'middle'; ctx.fillText(text, 14, c.height / 2 + 1);
      var tex = new T.CanvasTexture(c);
      var mat = new T.SpriteMaterial({map: tex, depthTest: false, depthWrite: false, transparent: true});
      disposables.push(tex, mat);
      var sp = new T.Sprite(mat);
      var hWorld = radius * 0.06;
      sp.scale.set(hWorld * c.width / c.height, hWorld, 1);
      sp.renderOrder = 5;
      return sp;
    }
    function setMarkers(markers) {
      while (markerGroup.children.length) markerGroup.remove(markerGroup.children[0]);
      targets = [];
      (markers || []).forEach(function (m, idx) {
        if (!m || !m.center) return;
        var color = m.color || COLORS.none;
        var geo = new T.SphereGeometry(Math.max(radius * 0.012, Math.min(radius * 0.03, (m.radius || 2) * 0.6)), 16, 12);
        var mat = new T.MeshBasicMaterial({color: new T.Color(color), depthTest: false, transparent: true, opacity: m.selected ? 1 : 0.9});
        disposables.push(geo, mat);
        var ball = new T.Mesh(geo, mat); ball.position.set(m.center[0], m.center[1], m.center[2]); ball.renderOrder = 4;
        if (m.selected) ball.scale.setScalar(1.5);
        ball.userData.pickId = m.id;
        markerGroup.add(ball);
        if (m.id !== undefined && m.id !== null) targets.push(ball);
        var sp = pill(m.label, color);
        // Labels alternate above / below their point so neighbouring outlets do not cover each other.
        if (sp) {
          sp.position.set(m.center[0], m.center[1], m.center[2] + (idx % 2 ? -1 : 1) * radius * 0.07);
          if (m.selected) sp.scale.multiplyScalar(1.2);
          sp.userData.pickId = m.id;
          markerGroup.add(sp);
          if (m.id !== undefined && m.id !== null) targets.push(sp);
        }
      });
      canvas.style.cursor = pickCb && targets.length ? 'pointer' : '';
      request();
    }
    // A click (not a drag) on a marker or its label → onPick(id), as the classic outlet viewer's endpoint spheres.
    var down = null;
    function onDown(ev) { down = [ev.clientX, ev.clientY]; }
    function onUp(ev) {
      if (!down || !pickCb || !targets.length || !T.Raycaster) { down = null; return; }
      var moved = Math.abs(ev.clientX - down[0]) + Math.abs(ev.clientY - down[1]);
      down = null;
      if (moved > 6) return;
      var r = canvas.getBoundingClientRect();
      var mouse = new T.Vector2((ev.clientX - r.left) / Math.max(r.width, 1) * 2 - 1, -(ev.clientY - r.top) / Math.max(r.height, 1) * 2 + 1);
      var ray = new T.Raycaster();
      ray.setFromCamera(mouse, camera);
      var hit = ray.intersectObjects(targets, false)[0];
      if (hit && hit.object && hit.object.userData) pickCb(hit.object.userData.pickId);
    }
    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointerup', onUp);
    // Client coordinates of a marker (for a scripted click in the browser walk-through).
    function markerScreen(id) {
      var ball = null;
      markerGroup.children.forEach(function (o) { if (!ball && o.isMesh && o.userData.pickId === id) ball = o; });
      if (!ball) return null;
      var v = ball.position.clone().project(camera), r = canvas.getBoundingClientRect();
      return {x: r.left + (v.x + 1) / 2 * r.width, y: r.top + (1 - v.y) / 2 * r.height};
    }
    size();
    return {setMesh: setMesh, setLines: setLines, lineCount: function () { return lineGroup.children.length; }, setMarkers: setMarkers, render: request, canvas: canvas, markerScreen: markerScreen,
      onPick: function (cb) { pickCb = typeof cb === 'function' ? cb : null; canvas.style.cursor = pickCb && targets.length ? 'pointer' : ''; },
      reset: applyHome,
      snapshot: function () { try { renderer.render(scene, camera); return canvas.toDataURL('image/png'); } catch (_) { return null; } },
      dispose: function () {
        if (ro) ro.disconnect(); else root.removeEventListener('resize', size);
        canvas.removeEventListener('pointerdown', onDown); canvas.removeEventListener('pointerup', onUp);
        if (controls) controls.dispose();
        disposables.forEach(function (d) { try { d.dispose(); } catch (_) {} });
        try { renderer.dispose(); if (renderer.forceContextLoss) renderer.forceContextLoss(); } catch (_) {}
        if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
      }};
  }

  // ------------------------------------------------------------------ panels
  function checkList(ic) {
    var h = ui().h;
    var q = ic && ic.quality;
    if (!q || !Array.isArray(q.checks) || !q.checks.length) return null;
    var words = {pass: '通过', pass_with_limits: '通过，有未评估项', review: '需要复核', fail: '不通过', not_checked: '未检查'};
    var tone = {pass: 'ok', pass_with_limits: 'ok', review: 'warn', fail: 'error', not_checked: 'idle'};
    var rank = {fail: 0, review: 1, not_checked: 2, pass_with_limits: 3, pass: 4};
    var checks = q.checks.slice().sort(function (a, b) { return (rank[a.status] === undefined ? 5 : rank[a.status]) - (rank[b.status] === undefined ? 5 : rank[b.status]); });
    return h('ul', {'class': 'checks'}, checks.map(function (c) {
      return h('li', {}, ui().dot(tone[c.status] || 'idle', words[c.status] || c.status || ''), h('span', {'class': 'check-label', text: c.label || c.key || ''}), c.note ? h('span', {'class': 'muted', text: c.note}) : null);
    }));
  }
  function factsTable(ic) {
    var h = ui().h;
    return ui().table([{key: 'label', label: ''}, {key: 'value', label: '', render: function (r) {
      return h('span', {'class': r.warn ? 'warn-text' : null, title: r.note || null, text: r.value});
    }}], inputFacts(ic), {cls: 'tbl-kv tbl-nohead tbl-facts'});
  }
  function copyTextTo(text) {
    var nav = root.navigator;
    if (nav && nav.clipboard && typeof nav.clipboard.writeText === 'function') {
      return nav.clipboard.writeText(text).then(function () { ui().toast('已复制。', {kind: 'ok', ms: 2000}); }, function () { ui().toast('浏览器不允许复制，请手动选中。', {kind: 'error'}); });
    }
    ui().toast('浏览器不允许复制，请手动选中。', {kind: 'error'});
    return Promise.resolve();
  }

  // ctx: {job, stage, inspector, cards, onChanged(job), onUpload(), onOpen(jobId)}
  function create(ctx) {
    var h = ui().h;
    var st = {job: ctx.job, etaAt: Date.now(), view: null, mesh: null, inputcheck: null, geometry: null, mapping: null, selected: null, disposed: false,
      inletMode: false, inletChoice: '', reasonsOpen: false, etaOpen: false, factsOpen: false, busy: false, eta: null, etaNote: null, rows: {}, ticker: null, poll: null,
      lines: [], procOpen: false, eventsOpen: false, checksOpen: false};
    var stageBox = h('div', {'class': 'input-stage'});
    var viewHost = h('div', {'class': 'input-view'});
    var stageMsg = h('div', {'class': 'input-stage-msg'});
    var stageTools = h('div', {'class': 'input-stage-tools', hidden: true},
      ui().iconButton('refresh', '复位视角', function () { if (st.view) st.view.reset(); }),
      ui().iconButton('snapshot', '保存截图', function () { saveShot(); }));
    stageBox.appendChild(viewHost); stageBox.appendChild(stageMsg); stageBox.appendChild(stageTools);
    ctx.stage.replaceChildren(stageBox);
    var panel = h('div', {'class': 'input-panel'});
    ctx.inspector.replaceChildren(panel);
    var go = function (id) { if (ctx.onOpen) ctx.onOpen(id); else if (ns.shell && ns.shell.go) ns.shell.go(id); };
    var jobsList = function () { return (ns.shell && ns.shell.state && ns.shell.state() && ns.shell.state().jobs) || []; };

    function ensureView() {
      if (st.view || st.disposed) return st.view;
      st.view = meshView(viewHost);
      if (!st.view) stageMsg.textContent = '这台设备的浏览器无法显示三维（WebGL 不可用）。右侧的检查结果和操作仍可使用。';
      else st.view.onPick(onPick);
      return st.view;
    }
    function saveShot() {
      var url = st.view && st.view.snapshot();
      if (!url) { ui().toast('这个浏览器没有保存三维截图，请用系统截图。', {kind: 'error'}); return; }
      var a = h('a', {href: url, download: 'wss-' + (st.job.status === 'awaiting_confirmation' ? 'outlets' : 'input') + '-' + st.job.id + '.png'});
      root.document.body.appendChild(a); a.click(); a.remove();
    }
    function loadGeometry() {
      var job = st.job;
      var needs = /^(awaiting|failed|interrupted|cancelled)/.test(job.status || '');
      if (!needs) return Promise.resolve();
      var tasks = [];
      tasks.push(api().inputcheck(job.id).then(function (r) { st.inputcheck = r; }, function () { st.inputcheck = null; }));
      if (job.status === 'awaiting_confirmation' || job.status === 'awaiting_outlets' || job.status === 'failed' || job.status === 'interrupted') tasks.push(api().geometry(job.id).then(function (r) { st.geometry = r; }, function () { st.geometry = null; }));
      return Promise.all(tasks).then(function () {
        if (st.disposed) return;
        st.mesh = meshFrom(st.inputcheck, st.geometry);
        // #73: the outlet viewer draws the centreline the outlets were named on (classic createViewer)
        var outletPage = st.job.status === 'awaiting_confirmation' || st.job.status === 'awaiting_outlets';
        st.lines = outletPage ? polylinesFrom(st.geometry, stageA().proposal) : [];
        var v = ensureView();
        if (v) { v.setMesh(st.mesh); v.setLines(st.lines); stageMsg.textContent = st.mesh || st.lines.length ? '' : '没有可显示的网格。'; }
        stageTools.hidden = !(v && (st.mesh || st.lines.length));
        drawMarkers();
      });
    }
    function stageA() { return st.job.a || st.job.stage_a || {}; }
    function endpoints() {
      var p = stageA().proposal || {};
      var eps = (st.geometry && st.geometry.endpoints && st.geometry.endpoints.length ? st.geometry.endpoints : p.endpoints) || [];
      return eps.filter(function (e) { return e && Array.isArray(e.center_mm); });
    }
    // Openings the centreline step found (opening_id is what /confirm {inlet} takes); the current inlet is marked.
    function inletOpenings() {
      var list = (stageA().centerline && stageA().centerline.openings) || (st.inputcheck && st.inputcheck.openings) || [];
      var chosen = st.job.params && st.job.params.inlet;
      return list.filter(function (o) { return o && o.opening_id !== undefined && o.opening_id !== null; }).map(function (o) {
        var id = Number(o.opening_id);
        return {id: id, radius: numOrNull(o.radius_mm), center: Array.isArray(o.center_mm) ? o.center_mm : null,
          current: chosen !== null && chosen !== undefined ? Number(chosen) === id : o.role === 'inlet'};
      }).sort(function (a, b) { return a.id - b.id; });
    }
    function drawMarkers() {
      if (!st.view) return;
      var job = st.job;
      var markers = [];
      if ((job.status === 'awaiting_confirmation' || job.status === 'awaiting_outlets') && st.inletMode) {
        inletOpenings().forEach(function (o) {
          var picked = String(o.id) === String(st.inletChoice);
          markers.push({id: 'o' + o.id, center: o.center, radius: o.radius, selected: picked, color: picked ? COLORS.pick : o.current ? COLORS.inlet : COLORS.none,
            label: '开口 ' + (o.id + 1) + (o.current ? ' · 当前入口' : '')});
        });
      } else if (job.status === 'awaiting_confirmation' || job.status === 'awaiting_outlets') {
        endpoints().forEach(function (e) {
          var sid = String(e.segment_id);
          var name = e.kind === 'inlet' ? 'inlet' : (st.mapping && st.mapping[sid]) || null;
          markers.push({id: 'e' + sid, center: e.center_mm, radius: e.radius_mm, color: COLORS[name] || COLORS.none, selected: st.selected === sid,
            label: '#' + sid + ' ' + (name ? NAMES[name] : '未命名')});
        });
      } else {
        var src = (st.inputcheck && st.inputcheck.openings && st.inputcheck.openings.length) ? st.inputcheck.openings
          : (st.inputcheck && st.inputcheck.input_check && st.inputcheck.input_check.quality && st.inputcheck.input_check.quality.opening_geometry)
          || (((job.a || {}).input_check || {}).quality || {}).opening_geometry || ((job.a || {}).centerline || {}).openings || [];
        openingHints(src).forEach(function (o, i) {
          markers.push({center: o.center, radius: o.radius, color: o.small ? COLORS.small : COLORS.inlet,
            label: '开口 ' + (i + 1) + ' · r ' + ui().num(o.radius, 'mm')});
        });
      }
      st.view.setMarkers(markers);
    }
    function onPick(id) {
      if (id === undefined || id === null) return;
      var s = String(id);
      if (s.charAt(0) === 'o' && st.inletMode) {
        st.inletChoice = s.slice(1);
        if (st.inletSelect) st.inletSelect.value = st.inletChoice;
        if (st.inletSync) st.inletSync(); else drawMarkers();
      } else if (s.charAt(0) === 'e') selectEndpoint(s.slice(1));
    }
    function selectEndpoint(sid) {
      st.selected = sid;
      Object.keys(st.rows).forEach(function (k) { if (st.rows[k] && st.rows[k].classList) st.rows[k].classList.toggle('sel', k === sid); });
      var row = st.rows[sid];
      if (row && row.scrollIntoView) { try { row.scrollIntoView({block: 'nearest'}); } catch (_) {} }
      drawMarkers();
    }

    // ---- follow-up polling: the shell polls only jobs that were active when the page opened; a retry (or a cancel
    // that has not stopped yet) turns a finished job active again, so this page follows it until the shell does.
    function shellPolls() { var S = ns.shell && ns.shell.state && ns.shell.state(); return Boolean(S && S.pollTimer); }
    function stopPoll() { if (st.poll) { clearInterval(st.poll); st.poll = null; } }
    function ensurePoll() {
      var active = /^(queued|running|cancelling|awaiting)/.test(st.job.status || '');
      if (!active || st.disposed || shellPolls()) { stopPoll(); return; }
      if (st.poll) return;
      st.poll = setInterval(function () {
        if (st.disposed || shellPolls()) { stopPoll(); return; }
        var id = st.job.id;
        api().job(id).then(function (r) {
          var j = r && (r.job || r);
          if (st.disposed || !j || j.id !== st.job.id) return;
          var moved = j.status !== st.job.status || j.version !== st.job.version;
          if (moved && ctx.onChanged) ctx.onChanged(j); else view.update(j);
        }, function () {});
      }, 3000);
      if (st.poll && st.poll.unref) st.poll.unref();
    }

    // ---- remaining time: one 1 s ticker advances the running stage on the browser clock between polls
    function stopTicker() { if (st.ticker) { clearInterval(st.ticker); st.ticker = null; } }
    function startTicker() {
      stopTicker();
      if (!st.job.eta || !/^(queued|running)/.test(st.job.status || '')) return;
      st.ticker = setInterval(function () { if (st.disposed) { stopTicker(); return; } if (st.eta) st.eta.paint(); }, 1000);
      if (st.ticker && st.ticker.unref) st.ticker.unref();
    }
    function etaWidget() {
      var box = h('div', {'class': 'eta'});
      var headRow = h('div', {'class': 'eta-headrow'});
      var headline = h('p', {'class': 'eta-head'});
      var basisTip = h('span', {'class': 'eta-basis'});
      var sub = h('p', {'class': 'muted eta-sub'});
      var bar = h('div', {'class': 'eta-bar', role: 'progressbar', 'aria-label': '计算进度', 'aria-valuemin': '0', 'aria-valuemax': '100'});
      var tbody = h('tbody');
      var table = h('table', {'class': 'tbl tbl-eta', hidden: !st.etaOpen},
        h('thead', {}, h('tr', {}, ['阶段', '预计', '已用', '状态'].map(function (t, i) { return h('th', {'class': i && i < 3 ? 'num' : null, scope: 'col', text: t}); }))), tbody);
      var toggle = ui().button('各阶段耗时', function () { st.etaOpen = !st.etaOpen; table.hidden = !st.etaOpen; ui().setPressed(toggle, st.etaOpen); paint(); }, {kind: 'link', cls: 'btn-sm eta-toggle'});
      ui().setPressed(toggle, st.etaOpen);
      var extra = h('p', {'class': 'muted eta-extra'});
      headRow.appendChild(headline); headRow.appendChild(basisTip);
      box.appendChild(headRow); box.appendChild(sub); box.appendChild(bar); box.appendChild(extra);
      box.appendChild(h('div', {'class': 'eta-foot'}, toggle)); box.appendChild(table);
      var keys = '', cells = [];
      function paint() {
        var job = st.job;
        var view = etaView(job.eta, {ageS: Math.max(0, (Date.now() - st.etaAt) / 1000)});
        var live = Boolean(view && (view.status === 'queued' || view.status === 'running') && !job.cancel_requested);
        bar.hidden = !live; toggle.hidden = !live; table.hidden = !live || !st.etaOpen;
        if (!live) {
          headline.textContent = job.phase || ui().statusInfo(job.status).label;
          ui().fill(basisTip);
          sub.textContent = typeof job.detail === 'string' ? job.detail : '';
          sub.hidden = !sub.textContent;
          extra.textContent = job.queue_position ? '队列第 ' + job.queue_position + ' 位' : '';
          extra.hidden = !extra.textContent;
          return;
        }
        headline.textContent = view.headline;
        ui().fill(basisTip, ui().infoTip(view.basis));
        sub.textContent = view.sub; sub.hidden = !view.sub;
        extra.textContent = ''; extra.hidden = true;
        var signature = view.segments.map(function (s) { return s.key; }).join(',');
        if (signature !== keys) {
          keys = signature; cells = [];
          bar.replaceChildren();
          view.segments.forEach(function () {
            var fill = h('span', {'class': 'eta-fill'});
            var cell = h('span', {'class': 'eta-seg'}, fill);
            cells.push({cell: cell, fill: fill}); bar.appendChild(cell);
          });
        }
        view.segments.forEach(function (s, i) {
          var c = cells[i];
          c.cell.className = 'eta-seg ' + s.state + (s.overdue ? ' overdue' : '') + (s.precomputed ? ' precomputed' : '');
          c.cell.style.flexBasis = s.width + '%'; c.fill.style.width = s.fill + '%';
          c.cell.title = s.title;
        });
        bar.setAttribute('aria-valuenow', String(view.progress));
        bar.setAttribute('aria-valuetext', [view.headline, view.sub].filter(Boolean).join('，'));
        if (st.etaOpen) {
          tbody.replaceChildren();
          view.segments.forEach(function (s) {
            var word = s.state === 'done' ? '已完成' : s.state === 'running' ? (s.overdue ? '比预计慢' : '进行中') : '等待';
            tbody.appendChild(h('tr', {'class': (s.state === 'running' ? 'on' : '') + (s.overdue ? ' overdue' : '') || null},
              h('td', {text: s.precomputed ? s.label + '（已预计算）' : s.label}), h('td', {'class': 'num', text: secondsText(s.expected)}),
              h('td', {'class': 'num', text: s.elapsed === null || s.elapsed === undefined ? '—' : secondsText(s.elapsed)}),
              h('td', {'class': s.overdue ? 'warn-text' : null, text: word})));
          });
        }
      }
      paint();
      return {el: box, paint: paint};
    }
    // 「确认后约 N 秒出结果」 next to a confirm button (classic etaNote).
    function etaNoteEl() {
      var el = h('span', {'class': 'eta-note muted', role: 'status'});
      var paint = function () {
        var view = etaView(st.job.eta, {ageS: 0});
        var text = view && (view.status === 'awaiting_confirmation' || view.status === 'awaiting_input') ? view.headline : '';
        el.textContent = text; el.hidden = !text; el.title = view ? view.basis : '';
      };
      paint();
      return {el: el, paint: paint};
    }

    function renderPanel() {
      var job = st.job;
      var status = job.status || '';
      st.eta = null; st.etaNote = null; st.rows = {}; st.inletSelect = null; st.inletSync = null;
      var cancel = canCancel(job) ? ui().button('取消任务', function () { cancelJob(); }, {kind: 'link', cls: 'btn-sm btn-cancel'}) : null;
      var head = h('div', {'class': 'input-head'}, h('div', {'class': 'insp-name', text: ui().displayName(job)}),
        h('div', {'class': 'insp-scan', text: ui().resultName(job, ctx.cards)}),
        h('div', {'class': 'insp-status'}, ui().statusDot(status), h('span', {'class': 'sec-fill'}), cancel));
      var body = [];
      var prov = provenanceBlock(provenance(job, ctx.cards, jobsList()), go);
      if (prov) body.push(h('div', {'class': 'input-prov'}, prov));
      if (/^(queued|running|cancelling)/.test(status)) {
        st.eta = etaWidget();
        body.push(ui().section('进度', {}, st.eta.el));
        stageMsg.textContent = '计算完成后这里显示结果。';
        stageTools.hidden = true;
      } else if (status === 'awaiting_input') {
        body.push(unitsCard(job));
      } else if (status === 'awaiting_confirmation' || status === 'awaiting_outlets') {
        body.push(outletCard(job));
      } else if (status === 'done') {
        // The record says done but the result files are not readable yet (a race with the worker): no error card.
        body.push(ui().section('已完成', {}, ui().note('结果文件还在准备，稍后再打开。'),
          h('div', {'class': 'sec-actions'}, ui().button('打开结果', function () { if (ns.shell && ns.shell.go) ns.shell.go(job.id); else go(job.id); }, {kind: 'primary', cls: 'btn-sm'}))));
      } else {
        body.push(failureCard(job));
        if (inputRecovery(job)) body.push(unitsCard(job));
      }
      var proc = processCard(job);
      if (proc) body.push(proc);
      ui().fill(panel, head, h('div', {'class': 'input-body'}, body));
      syncTitle();
      startTicker();
      ensurePoll();
    }
    // The viewport title pill is written once when the page opens; keep its status word in step (confirm → 计算中).
    function syncTitle() {
      var S = ns.shell && ns.shell.state && ns.shell.state();
      var tb = S && S.els && S.els.toolbar;
      if (!tb || !tb.children) return;
      Array.prototype.forEach.call(tb.children, function (c) { if (c && c.className === 'tb-title') c.textContent = ui().statusInfo(st.job.status).label; });
    }
    function act(promise) {
      st.busy = true;
      return promise.then(function (r) {
        st.busy = false;
        var job = r && (r.job || r);
        if (job && job.id && ctx.onChanged) ctx.onChanged(job);
        return job;
      }, function (error) {
        st.busy = false;
        if (error.status === 409) {
          ui().toast('任务状态已经变了，已读回最新状态。', {kind: 'info'});
          api().job(st.job.id).then(function (r) { if (ctx.onChanged) ctx.onChanged(r.job || r); });
        } else ui().toast(error.message, {kind: 'error'});
        return null;
      });
    }
    function cancelJob() {
      var job = st.job;
      return ui().confirm('取消任务 · ' + ui().displayName(job), cancelLines(job), {confirmLabel: '取消任务', cancelLabel: '不取消', danger: true}).then(function (ok) {
        if (!ok || st.disposed || st.job.id !== job.id) return null;
        return act(api().cancel(job.id, {version: st.job.version})).then(function (j) { if (j) ui().toast('任务已取消。之后可以点「重试」继续。', {kind: 'ok'}); return j; });
      });
    }
    // P4 lane A (#78 / #102): 「输入检查与计算过程」 behind one toggle, as the classic processCard (shown whenever stage A
    // has an input check, except while the units wait for a person) — the same 输入检查 / 计算过程 sections as the tools
    // tab of a finished result (ws_detail), with the centreline check rows.  A failed page already lists its input facts
    // and checks, so it adds the process only.
    function processCard(job) {
      var D = ns.detail, status = job.status || '';
      if (!D || typeof D.processSectionFor !== 'function' || status === 'awaiting_input' || status === 'done') return null;
      var a = stageA(), ic = (st.inputcheck && st.inputcheck.input_check) || a.input_check || null;
      if (!ic) return null;
      var failed = RECOVER.indexOf(status) >= 0;
      var openings = (a.centerline && a.centerline.openings) || (st.inputcheck && st.inputcheck.openings) || [];
      var parts = [failed ? null : D.inputSectionFor(ic, openings, st), D.processSectionFor(job, st, {quiet: true})].filter(Boolean);
      if (!parts.length) return null;
      var more = h('div', {'class': 'proc-more', hidden: !st.procOpen}, parts);
      var btn = ui().button(failed ? '计算过程' : '输入检查与计算过程', function () {
        st.procOpen = !st.procOpen; more.hidden = !st.procOpen; ui().setPressed(btn, st.procOpen);
      }, {kind: 'link', cls: 'btn-sm'});
      ui().setPressed(btn, st.procOpen);
      return h('div', {'class': 'input-proc'}, h('div', {'class': 'sec-actions'}, btn), more);
    }
    function unitsCard(job) {
      var ic = (st.inputcheck && st.inputcheck.input_check) || (job.a || {}).input_check || {};
      var raw = boxSize(ic, true);
      var suggested = Array.isArray(ic.suggested_units) ? ic.suggested_units[0] : ic.suggested_units;
      var flags = [].concat(ic.errors || [], ic.flags || []).filter(function (t, i, a) { return typeof t === 'string' && t && a.indexOf(t) === i; });
      var fail = ic.status === 'fail';
      var ref = referenceText(job.summary);
      var parts = [ui().note('STL 不记录单位。单位错一档，血管尺寸差 10 倍，结果不能用。请对照影像里的血管直径核对。'), factsTable(ic)];
      flags.forEach(function (t) { parts.push(ui().note(t, fail ? 'error' : 'warn')); });
      if (ref) parts.push(h('p', {'class': 'note' + (ref.review ? ' note-warn' : ''), text: ref.text}));
      if (fail) {
        parts.push(ui().note('这份几何没有达到计算门限，不能通过单位确认继续。请按上面的提示修复原始 STL 后重新上传。', 'error'));
        parts.push(checkList(ic));
        parts.push(h('div', {'class': 'sec-actions'}, ctx.onUpload ? ui().button('修改后重新上传', ctx.onUpload, {kind: 'primary', cls: 'btn-sm'}) : null,
          ui().link('输入说明', '/static/v2/help_input.html', {newTab: true})));
        return ui().section('核对单位与尺寸', {}, parts);
      }
      var units = ui().select([{value: 'mm', label: '毫米 mm'}, {value: 'cm', label: '厘米 cm'}, {value: 'm', label: '米 m'}],
        [ic.selected_units, (job.params || {}).units, typeof suggested === 'string' ? suggested : suggested && suggested.unit].filter(function (u) { return u === 'mm' || u === 'cm' || u === 'm'; })[0] || 'mm');
      var preview = h('p', {'class': 'muted', 'aria-live': 'polite'});
      var ack = h('input', {type: 'checkbox'});
      var frag = (ic.components || 1) > 1 ? h('input', {type: 'checkbox', checked: (job.params || {}).remove_fragments === true}) : null;
      var goBtn = ui().button('确认并继续', function () {
        goBtn.disabled = true;
        act(api().input(job.id, {units: units.value, remove_fragments: Boolean(frag && frag.checked), acknowledged: true, version: job.version})).then(function (j) { if (!j) goBtn.disabled = !ack.checked; });
      }, {kind: 'primary', disabled: true});
      var upd = function () { var s = {mm: 1, cm: 10, m: 1000}[units.value]; preview.textContent = raw ? '确认后 ' + raw.map(function (x) { return ui().sig(x * s); }).join(' × ') + ' mm（× ' + s + '）' : '确认后重新检查尺寸。'; ack.checked = false; goBtn.disabled = true; };
      units.addEventListener('change', upd);
      if (frag) frag.addEventListener('change', function () { ack.checked = false; goBtn.disabled = true; });
      ack.addEventListener('change', function () { goBtn.disabled = !ack.checked; });
      upd();
      st.etaNote = etaNoteEl();
      var removedPct = (Number(ic.removed_area_fraction || 0) * 100).toFixed(3) + '%';
      parts.push(h('label', {'class': 'fld'}, h('span', {text: 'STL 的原始单位'}), units), preview,
        frag ? h('label', {'class': 'check'}, frag, h('span', {text: '删除孤立碎片（面积 ' + removedPct + '），保留主要壁面；已核对碎片不是缺失的分支'})) : null,
        checkList(ic),
        h('label', {'class': 'check'}, ack, h('span', {text: '我已核对单位和换算后的尺寸'})),
        h('div', {'class': 'sec-actions'}, goBtn, st.etaNote.el));
      return h('div', {}, ui().section('核对单位与尺寸', {tag: ui().infoTip('服务端会再次检查全部输入门限；删除比例超过 1% 的壁面需要修复原始几何后重新上传。')}, parts));
    }
    function outletCard(job) {
      var a = stageA();
      var p = a.proposal || {};
      if (!st.mapping) st.mapping = Object.assign({}, job.mapping || p.mapping || {});
      var eps = endpoints();
      var outlets = eps.filter(function (e) { return e.kind !== 'inlet'; }).map(function (e) { return String(e.segment_id); });
      var review = outletReview(p, job.detail, p.direction_note || (a.input_check || {}).direction_note);
      // W35: a summary's top-level flags (the naming suggestion's warnings, normally the proposal's own) join the reasons
      ((job.summary && Array.isArray(job.summary.flags)) ? job.summary.flags : []).forEach(function (t) {
        var x = humanOutletReason(t);
        if (x && review.reasons.indexOf(x) < 0) review.reasons.push(x);
      });
      var ack = h('input', {type: 'checkbox'});
      var goBtn = ui().button('确认出口并开始预测', function () {
        goBtn.disabled = true;
        act(api().confirm(job.id, {mapping: Object.assign({}, st.mapping), acknowledged: true, version: job.version})).then(function (j) {
          if (j) offerNext('confirm', job.id, {go: go});
          else refresh();
        });
      }, {kind: 'primary', disabled: true});
      var ruleNote = ui().note('四个出口名称各用一次，同一侧的两个出口属于同侧。', 'warn');
      var changed = h('p', {'class': 'muted mapping-changes', 'aria-live': 'polite'});
      function refresh() {
        var ok = validMapping(st.mapping, outlets);
        goBtn.disabled = !ack.checked || !ok || st.busy; ruleNote.hidden = ok;
        var diff = mappingChanges(st.mapping, p.mapping);
        changed.textContent = diff.length ? '已修改 ' + diff.length + ' 个出口：' + diff.map(function (id) { return '#' + id + ' → ' + (NAMES[st.mapping[id]] || '未命名'); }).join('；') : '';
        changed.hidden = !diff.length;
        drawMarkers();
      }
      var rows = eps.map(function (e) {
        var sid = String(e.segment_id);
        var nameCell;
        if (e.kind === 'inlet') nameCell = h('span', {text: NAMES.inlet});
        else {
          nameCell = ui().select([{value: '', label: '请选择'}].concat(['out-le', 'out-li', 'out-re', 'out-ri'].map(function (k) { return {value: k, label: NAMES[k]}; })),
            st.mapping[sid] || '', function (v) { st.mapping[sid] = v; ack.checked = false; selectEndpoint(sid); refresh(); }, {'aria-label': '出口 ' + sid + ' 的名称'});
        }
        var c = Array.isArray(e.center_mm) ? e.center_mm.map(function (v) { return Number(v).toFixed(1); }).join(', ') : '';
        return {sid: sid, kind: e.kind, radius: e.radius_mm, name: nameCell, coords: c, _cls: st.selected === sid ? 'sel' : null};
      });
      var tbl = ui().table([
        {key: 'sid', label: '端点', render: function (r) { var b = h('button', {type: 'button', 'class': 'val-link', text: '#' + r.sid + (r.kind === 'inlet' ? ' 入口' : '')}); b.addEventListener('click', function () { selectEndpoint(r.sid); }); return b; }},
        {key: 'radius', label: '半径', num: true, render: function (r) { return ui().num(r.radius, 'mm'); }},
        {key: 'name', label: '名称', render: function (r) { return r.name; }}], rows, {cls: 'tbl-outlets',
        onRow: function (r, tr) { st.rows[r.sid] = tr; if (r.coords) tr.title = '世界坐标 ' + r.coords + ' mm'; }});
      var sw = function (label, kind, title) { return ui().button(label, function () { st.mapping = swap(st.mapping, kind); ack.checked = false; renderPanel(); }, {cls: 'btn-sm', title: title}); };
      ack.addEventListener('change', refresh);
      setTimeout(refresh, 0);
      // Reasons in plain words: the first one (normally the side to check) stays visible, the rest open on demand.
      var reasons = review.reasons;
      var more = reasons.length > 1 ? h('ul', {'class': 'reasons', hidden: !st.reasonsOpen}, reasons.slice(1).map(function (t) { return h('li', {text: t}); })) : null;
      var moreBtn = more ? ui().button('更多原因 · ' + (reasons.length - 1), function () { st.reasonsOpen = !st.reasonsOpen; more.hidden = !st.reasonsOpen; ui().setPressed(moreBtn, st.reasonsOpen); }, {kind: 'link', cls: 'btn-sm'}) : null;
      if (moreBtn) ui().setPressed(moreBtn, st.reasonsOpen);
      st.etaNote = etaNoteEl();
      // W30: the percentage in the headline is a geometric proxy — said first in the ⓘ beside it
      var tag = h('span', {'class': 'sec-tagrow'}, h('span', {'class': 'sec-note', text: review.headline}),
        ui().infoTip((review.confidence !== null ? confidenceCaveat(p.confidence_gate) + ' ' : '') +
          'STL 不带患者方向，出口名称决定模型用的左右坐标；名称错了，左右分支的结果会对调。请对照原始影像核对。三维里的 X / Y / Z 是 STL 世界坐标，不是患者方位。'));
      return h('div', {}, ui().section('确认出口', {tag: tag},
        reasons.length ? ui().note(reasons[0], review.required ? 'warn' : null) : null,
        moreBtn ? h('div', {'class': 'reasons-more'}, moreBtn) : null, more,
        tbl,
        h('div', {'class': 'swap-row'}, sw('左右互换', 'lr', '左右两侧整体互换'), sw('左侧内 / 外', 'left', '左侧髂内与髂外互换'), sw('右侧内 / 外', 'right', '右侧髂内与髂外互换'),
          ui().button('恢复建议', function () { st.mapping = Object.assign({}, p.mapping || {}); ack.checked = false; renderPanel(); }, {kind: 'link', cls: 'btn-sm'})),
        changed, ruleNote,
        h('label', {'class': 'check'}, ack, h('span', {text: '我已结合原始影像核对入口和四个出口的名称'})),
        h('div', {'class': 'sec-actions'}, goBtn, st.etaNote.el)), inletSection(job));
    }
    // #74: the wrong opening was taken as the inlet → pick another one (list or 3-D) and extract the centreline again.
    function inletSection(job) {
      var ops = inletOpenings();
      if (!ops.length) return null;
      var current = ops.filter(function (o) { return o.current; })[0];
      var sel = ui().select([{value: '', label: '选择作为入口的开口'}].concat(ops.map(function (o) {
        return {value: String(o.id), label: '开口 ' + (o.id + 1) + ' · r ' + ui().num(o.radius, 'mm') + (o.current ? ' · 当前入口' : '')};
      })), st.inletChoice, null, {'aria-label': '重新选择入口'});
      st.inletSelect = sel;
      var redo = ui().button('重提中心线', function () {
        var id = Number(sel.value);
        if (sel.value === '' || !isFinite(id)) return;
        redo.disabled = true;
        act(api().confirm(job.id, {mapping: Object.assign({}, st.mapping || {}), inlet: id, acknowledged: true, version: job.version})).then(function (j) {
          if (j) { st.inletMode = false; st.inletChoice = ''; ui().toast('已改用开口 ' + (id + 1) + ' 作为入口，正在重新提取中心线。', {kind: 'ok'}); }
          else redo.disabled = false;
        });
      }, {cls: 'btn-sm', disabled: true});
      var sync = function () { st.inletChoice = sel.value; redo.disabled = sel.value === '' || Boolean(current && String(current.id) === sel.value); drawMarkers(); };
      sel.addEventListener('change', sync);
      st.inletSync = sync;
      var body = h('div', {'class': 'inlet-body', hidden: !st.inletMode},
        h('div', {'class': 'inlet-row'}, sel, redo),
        ui().note('也可以在三维里点开口。改入口后重新提取中心线，之后要重新确认出口。'));
      var toggle = ui().button('入口选错了？', function () {
        st.inletMode = !st.inletMode; body.hidden = !st.inletMode; ui().setPressed(toggle, st.inletMode);
        if (!st.inletMode) { st.inletChoice = ''; sel.value = ''; }
        sync();
      }, {kind: 'link', cls: 'btn-sm'});
      ui().setPressed(toggle, st.inletMode);
      if (st.inletMode) redo.disabled = sel.value === '' || Boolean(current && String(current.id) === sel.value);
      return ui().section('入口', {cls: 'sec-inlet', actions: toggle, tag: current ? h('span', {'class': 'sec-note', text: '开口 ' + (current.id + 1) + ' · r ' + ui().num(current.radius, 'mm')}) : null}, body);
    }
    function failureCard(job) {
      var ic = (st.inputcheck && st.inputcheck.input_check) || (job.a || {}).input_check || null;
      var em = errorModel(job, st.inputcheck && st.inputcheck.error);
      var hints = openingHints((st.inputcheck && st.inputcheck.openings && st.inputcheck.openings.length ? st.inputcheck.openings : null)
        || (ic && ic.quality && ic.quality.opening_geometry) || ((job.a || {}).centerline || {}).openings || []);
      var small = hints.filter(function (o) { return o.small; });
      var parts = [];
      if (em.message) parts.push(ui().note(em.message, job.status === 'cancelled' ? null : 'error'));
      if (em.cpu) parts.push(ui().note('重试时改用 CPU 计算（较慢），不必重新上传。'));
      if (em.noRetryText) parts.push(ui().note(em.noRetryText));
      if (em.diagnostic) {
        parts.push(h('p', {'class': 'diag'}, h('span', {'class': 'muted', text: '诊断编号 '}), h('span', {'class': 'mono', text: em.diagnostic}),
          ui().iconButton('copy', '复制诊断编号', function () { copyTextTo(em.diagnostic); }, {cls: 'btn-copy'})));
      }
      // admin_detail reaches the page only for administrators (server.py strips it otherwise).
      if (em.adminDetail) parts.push(h('details', {'class': 'admin-detail'}, h('summary', {text: '技术细节'}), h('pre', {text: em.adminDetail})));
      // Actions first (retry is the way out of most failures), then the evidence: openings, checks, input facts.
      var actions = [];
      if (em.canRetry) {
        var retryBtn = ui().button(em.retryLabel, function () {
          retryBtn.disabled = true;
          act(api().retry(job.id, {version: job.version})).then(function (j) {
            if (j) ui().toast(job.status === 'interrupted' ? '已恢复，从可用的阶段继续。' : '已重新排队，从可用的阶段继续。', {kind: 'ok'});
            else retryBtn.disabled = false;
          });
        }, {kind: 'primary', cls: 'btn-sm'});
        actions.push(retryBtn);
      }
      if (ctx.onUpload) actions.push(ui().button('修改后重新上传', ctx.onUpload, {kind: em.canRetry ? null : 'primary', cls: 'btn-sm'}));
      if (em.locked) parts.push(ui().note('已复核锁定，重新打开后才能重试。'));
      parts.push(h('div', {'class': 'sec-actions'}, actions, ui().link('报错对照表', '/static/v2/help_errors.html', {newTab: true}), ui().link('输入说明', '/static/v2/help_input.html', {newTab: true})));
      // A cancelled job with five normal openings has nothing to show here.
      if (hints.length && (job.status !== 'cancelled' || hints.length !== 5 || small.length)) {
        parts.push(ui().table([{key: 'i', label: '开口', render: function (r) { return String(r.i); }}, {key: 'r', label: '等效半径', num: true, render: function (r) { return ui().num(r.radius, 'mm'); }},
          {key: 'n', label: '', blank: true, render: function (r) { return r.small ? h('span', {'class': 'warn-text', text: '可能是没封闭的分支残端'}) : ''; }}],
          hints.map(function (o, i) { return Object.assign({i: i + 1}, o); }), {cls: 'tbl-openings'}));
        parts.push(ui().note('需要恰好五个开口：主动脉入口和四个髂动脉出口。' + (hints.length !== 5 ? '这份输入有 ' + hints.length + ' 个。' : '') + (small.length ? '红色标记的开口明显偏小，请在分割软件里封闭后重新导出。' : '')));
      }
      parts.push(checkList(ic));
      if (ic && !inputRecovery(job)) {
        var facts = h('div', {'class': 'facts-more', hidden: !st.factsOpen}, factsTable(ic));
        var fbtn = ui().button('输入尺寸与面积', function () { st.factsOpen = !st.factsOpen; facts.hidden = !st.factsOpen; ui().setPressed(fbtn, st.factsOpen); }, {kind: 'link', cls: 'btn-sm'});
        ui().setPressed(fbtn, st.factsOpen);
        parts.push(h('div', {'class': 'sec-actions'}, fbtn), facts);
      }
      return ui().section(em.title, {tag: ui().infoTip(em.hint), cls: 'sec-failure'}, parts);
    }

    renderPanel();
    loadGeometry().then(function () { if (!st.disposed) renderPanel(); });
    var view = {
      update: function (job) {
        var changed = !st.job || job.status !== st.job.status || job.version !== st.job.version;
        st.job = job; st.etaAt = Date.now();
        if (!changed) {
          // Same state, newer estimate: repaint the progress only (inputs and choices on the panel stay as they are).
          if (st.eta) st.eta.paint();
          if (st.etaNote) st.etaNote.paint();
          if (!st.ticker) startTicker();
          return;
        }
        if (job.status !== 'awaiting_confirmation' && job.status !== 'awaiting_outlets') { st.mapping = null; st.inletMode = false; st.inletChoice = ''; }
        renderPanel(); loadGeometry().then(function () { if (!st.disposed) renderPanel(); });
      },
      dispose: function () { st.disposed = true; stopTicker(); stopPoll(); if (st.view) st.view.dispose(); st.view = null; },
      pick: onPick,   // what a click on a 3-D marker does ('e<segment>' endpoint, 'o<opening>' inlet candidate)
      markerAt: function (id) { return st.view ? st.view.markerScreen(id) : null; },
      state: st
    };
    return view;
  }

  // ------------------------------------------------------------------ finished results: where the result came from (#77, tools page)
  (ns.ext = ns.ext || []).push({
    id: 'input-provenance',
    tools: function (sh) {
      var cur = sh && sh.cur && sh.cur();
      if (!cur || !cur.job || (sh.offline && sh.offline())) return null;
      var lines = provenance(cur.job, sh.cards ? sh.cards() : {}, sh.jobs ? sh.jobs() : []);
      if (!lines.length) return null;
      return ui().section('来源', {cls: 'sec-prov'}, provenanceBlock(lines, function (id) { sh.go(id); }));
    }
  });
  watchReview();

  return {create: create, swap: swap, validMapping: validMapping, openingHints: openingHints, normalizeOpenings: normalizeOpenings, meshFrom: meshFrom, NAMES: NAMES,
    etaView: etaView, formatDuration: formatDuration, stageRemaining: stageRemaining, errorModel: errorModel, canCancel: canCancel, cancelLines: cancelLines,
    humanOutletReason: humanOutletReason, outletReview: outletReview, mappingChanges: mappingChanges, inputFacts: inputFacts, boxSize: boxSize, referenceText: referenceText,
    provenance: provenance, nextJob: nextJob, offerNext: offerNext, watchReview: watchReview, inputRecovery: inputRecovery,
    // P4 lane A
    polylinesFrom: polylinesFrom, confidenceCaveat: confidenceCaveat};
});
