/* Workbench helpers without DOM or network dependencies (unread bookkeeping, preference merging,
   review grouping, case-card display helpers, metadata payloads, the side-by-side display-state
   subset and echo guard, and the cohort overview's binning / filtering / sorting).  Loaded by the
   workbench and comparison pages as a plain script and required by tests/test_workbench_js.py in Node. */
(function (root) {
  'use strict';
  const FINAL = ['done', 'failed', 'cancelled', 'interrupted'];
  const NOTIFY_STATUSES = [...FINAL, 'awaiting_confirmation', 'awaiting_input'];
  const STATUS_TEXT = {done: '已完成', failed: '失败', cancelled: '已取消', interrupted: '已中断', awaiting_confirmation: '待确认出口', awaiting_input: '待确认输入'};
  const FAMILY_TEXT = {wall: '壁面 WSS', volume: '压力 + 速度体场'};
  const BUILTIN_PRESETS = {
    wall: ['瘤囊低 WSS 区', '髂分叉热点', '主动脉沿程'],
    volume: ['沿程压降', '流线全貌', '瘤囊截面系列'],
  };

  // ---- unread set: entries are "job_id:version" strings kept in localStorage ----
  function unreadParse(raw) {
    let list = [];
    try { list = JSON.parse(raw || '[]'); } catch (_) { list = []; }
    if (!Array.isArray(list)) list = [];
    return new Set(list.filter(item => typeof item === 'string' && item.includes(':')));
  }
  function unreadSerialize(set) { return JSON.stringify([...set]); }
  function unreadKey(jobId, version) { return `${jobId}:${version === undefined || version === null ? 0 : version}`; }
  function unreadAdd(set, jobId, version) {
    if (!jobId) return false;
    const key = unreadKey(jobId, version);
    if (set.has(key)) return false;
    for (const item of [...set]) if (item.slice(0, item.lastIndexOf(':')) === String(jobId)) set.delete(item);
    set.add(key);
    return true;
  }
  function unreadRemove(set, jobId) {
    let removed = false;
    for (const item of [...set]) if (item.slice(0, item.lastIndexOf(':')) === String(jobId)) { set.delete(item); removed = true; }
    return removed;
  }
  function unreadHas(set, jobId) {
    for (const item of set) if (item.slice(0, item.lastIndexOf(':')) === String(jobId)) return true;
    return false;
  }
  function unreadJobs(set) {
    const ids = new Set();
    for (const item of set) ids.add(item.slice(0, item.lastIndexOf(':')));
    return [...ids];
  }
  function unreadCount(set) { return unreadJobs(set).length; }
  function unreadPrune(set, existingIds) {
    // Keep only entries whose job still exists (an owner-level list refresh knows the full set).
    if (!existingIds) return 0;
    const keep = new Set(existingIds.map(String));
    let dropped = 0;
    for (const item of [...set]) if (!keep.has(item.slice(0, item.lastIndexOf(':')))) { set.delete(item); dropped++; }
    return dropped;
  }
  function titleWithUnread(base, count) { return count > 0 ? `(${count}) ${base}` : base; }

  // ---- owner-level event stream ----
  function shouldNotify(event, lastStatusByJob) {
    // A transition is worth a notification when it lands on a terminal state or a manual-confirmation
    // state, and only once per (job, status) pair — progress events never notify.
    if (!event || !event.job_id || event.action === 'progress') return false;
    const status = event.status;
    if (!NOTIFY_STATUSES.includes(status)) { if (lastStatusByJob) lastStatusByJob[event.job_id] = status; return false; }
    const previous = lastStatusByJob ? lastStatusByJob[event.job_id] : undefined;
    if (lastStatusByJob) lastStatusByJob[event.job_id] = status;
    return previous !== status;
  }
  function notificationText(event) {
    const name = event.case_id || event.job_id;
    const status = STATUS_TEXT[event.status] || event.status;
    const detail = event.status === 'failed' ? '请打开任务查看原因。' : event.status === 'done' ? '报告与导出文件已就绪。' : event.status === 'awaiting_confirmation' ? '请核对出口命名。' : event.status === 'awaiting_input' ? '请确认单位与尺寸。' : '';
    return {title: `${name} · ${status}`, body: [event.family ? FAMILY_TEXT[event.family] : '', detail].filter(Boolean).join(' · ')};
  }

  // ---- preferences ----
  function isObject(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }
  function mergePreferences(base, patch) {
    // Section-wise merge: top-level objects merge one level deep, everything else is replaced.
    const out = isObject(base) ? {...base} : {};
    for (const [key, value] of Object.entries(isObject(patch) ? patch : {})) {
      if (isObject(value) && isObject(out[key])) out[key] = {...out[key], ...value};
      else out[key] = value;
    }
    if (!out.schema_version) out.schema_version = 'wss-deploy.preferences/v1';
    return out;
  }
  function uploadPreferencesFrom(form) {
    return {
      units: form.units || 'auto', release_id: form.release_id || '', device: form.device || 'auto',
      seed_count: form.seed_count || 'all', threads: form.threads || '', remember_patient: Boolean(form.remember_patient),
      last_patient: {patient_id: form.patient_id || '', scan_label: form.scan_label || '', scan_date: form.scan_date || '', tags: form.tags || ''},
    };
  }

  // ---- review grouping (C18) ----
  function isPendingReview(job) { return job && job.status === 'done' && !(job.review && job.review.status === 'reviewed'); }
  function isReviewed(job) { return job && job.status === 'done' && job.review && job.review.status === 'reviewed'; }
  function splitByReview(jobs) {
    const pending = [], reviewed = [], other = [];
    for (const job of jobs || []) {
      if (isPendingReview(job)) pending.push(job);
      else if (isReviewed(job)) reviewed.push(job);
      else other.push(job);
    }
    return {pending, reviewed, other};
  }

  // ---- case cards (C1) ----
  function familyOfRelease(releaseId, releases) {
    for (const item of releases || []) if ((item.id || item.release) === releaseId) return item.family || null;
    return null;
  }
  // §17.4: the card chip reads the maximum diameter straight off the light run snapshot; older runs
  // (and older services) carry no such key, and then the chip is simply not drawn.
  function latestMaxDiameter(runs) {
    for (const run of Array.isArray(runs) ? runs : []) {
      if (!isObject(run) || run.status !== 'done') continue;
      const value = numberOrNull(run.max_diameter_mm);
      if (value !== null) return value;
    }
    return null;
  }
  function caseCardModel(card, releases) {
    const runs = Array.isArray(card.runs) ? card.runs : [];
    const latest = isObject(card.latest) ? card.latest : {};
    const byFamily = {};
    for (const [releaseId, jobId] of Object.entries(latest)) {
      const run = runs.find(item => item.job_id === jobId);
      const family = (run && run.family) || familyOfRelease(releaseId, releases);
      if (family && !byFamily[family]) byFamily[family] = jobId;
    }
    const wall = byFamily.wall || null, volume = byFamily.volume || null;
    return {
      // C7 (§5.4): the service's display name (patient id when filled, else the case name); older services
      // send none, and then the case names are shown as before.
      title: (typeof card.display_name === 'string' && card.display_name.trim()) || (card.case_ids || []).join(' / ') || '未命名病例',
      caseIds: (card.case_ids || []).join(' / '),
      subtitle: [card.patient_id && `患者 ${card.patient_id}`, card.scan_label, card.scan_date].filter(Boolean).join(' · '),
      tags: Array.isArray(card.tags) ? card.tags : [],
      runs, pendingReview: Number(card.pending_review || 0),
      compare: wall && volume ? {left: wall, right: volume} : null,
      missing: (card.missing_releases || []).filter(id => typeof id === 'string'),
      reusable: card.reusable_job_id || null,
      doneJobIds: runs.filter(run => run.status === 'done').map(run => run.job_id),
      latestAt: card.latest_at || '',
      maxDiameterMm: latestMaxDiameter(runs),
    };
  }
  function releaseLabel(releaseId, family) {
    const text = FAMILY_TEXT[family];
    return text ? `${text} · ${releaseId}` : releaseId;
  }

  // ---- automatic narrative (§17.2) ----
  function narrativeModel(narrative, lang) {
    // `edited` (a reviewer's text) wins over the generated sentences; a summary without either —
    // an older job, or a service that does not generate one — yields null and no card is drawn.
    if (!isObject(narrative)) return null;
    const raw = narrative[lang === 'en' ? 'en' : 'zh'];
    const sentences = (Array.isArray(raw) ? raw : []).filter(item => typeof item === 'string' && item.trim()).map(item => item.trim());
    const editedRaw = typeof narrative.edited === 'string' ? narrative.edited.trim() : '';
    const edited = editedRaw || null;
    if (!edited && !sentences.length) return null;
    return {
      sentences, edited,
      editedBy: (typeof narrative.edited_by === 'string' && narrative.edited_by.trim()) || '',
      editedAt: (typeof narrative.edited_at === 'string' && narrative.edited_at.trim()) || '',
      text: edited || sentences.join('\n'),
    };
  }

  // ---- exports (C13/C14/C15) ----
  function exportUrl(ids, format) {
    const clean = (ids || []).filter(id => /^[A-Za-z0-9_-]{1,80}$/.test(String(id)));
    return `/api/jobs/export?ids=${clean.map(encodeURIComponent).join(',')}&format=${format === 'xlsx' ? 'xlsx' : 'csv'}`;
  }
  function filenameFromDisposition(header, fallback) {
    const match = /filename\*=UTF-8''([^;]+)/i.exec(header || '') || /filename="?([^";]+)"?/i.exec(header || '');
    if (!match) return fallback;
    try { return decodeURIComponent(match[1]); } catch (_) { return match[1]; }
  }
  function stateFamily(state) {
    if (!isObject(state)) return null;
    if (state.family === 'wall' || state.family === 'volume') return state.family;
    if (state.preset_name) { for (const [family, names] of Object.entries(BUILTIN_PRESETS)) if (names.includes(state.preset_name)) return family; }
    return null;
  }
  function planBatchExport(jobs, state) {
    // Wall states apply to wall jobs only (and vice versa); a state without a family applies everywhere.
    const family = stateFamily(state);
    const selected = [], skipped = [];
    for (const job of jobs || []) {
      if (job.status !== 'done') { skipped.push({id: job.id, case_id: job.case_id, message: '任务未完成'}); continue; }
      if (family && job.family && job.family !== family) { skipped.push({id: job.id, case_id: job.case_id, message: `视图状态属于${FAMILY_TEXT[family]}，与该任务的${FAMILY_TEXT[job.family]}不匹配`}); continue; }
      selected.push(job);
    }
    return {selected, skipped, family};
  }
  function presetState(name) {
    const family = stateFamily({preset_name: name});
    if (!family) return null;
    return {schema_version: 'wss-deploy.view/v1', family, preset_name: name};
  }
  function parseViewStateJSON(text) {
    let value;
    try { value = JSON.parse(text); } catch (_) { throw new Error('视图状态不是有效 JSON。'); }
    if (!isObject(value)) throw new Error('视图状态必须是 JSON 对象。');
    if (value.schema_version && !String(value.schema_version).startsWith('wss-deploy.view/')) throw new Error('这不是报告导出的视图状态文件。');
    return value;
  }

  // ---- metadata editing (§15.2) ----
  function parseTags(text) {
    // Free-text tag field → list: comma (half or full width) and the Chinese enumeration comma all separate.
    const parts = String(text === null || text === undefined ? '' : text).split(/[,，、\n]/);
    const out = [];
    for (const part of parts) {
      const tag = part.trim();
      if (tag && !out.includes(tag)) out.push(tag);
    }
    return out;
  }
  function tagsText(tags) {
    if (Array.isArray(tags)) return tags.filter(tag => typeof tag === 'string' && tag.trim()).join(', ');
    return typeof tags === 'string' ? tags : '';
  }
  function metadataPayload(form, version) {
    const text = value => String(value === null || value === undefined ? '' : value).trim();
    const payload = {version, case_id: text(form.case_id), patient_id: text(form.patient_id), scan_label: text(form.scan_label),
      scan_date: text(form.scan_date), tags: parseTags(form.tags), notes: text(form.notes)};
    if (version === undefined || version === null) delete payload.version;
    return payload;
  }

  // ---- batch rerun (§15.4) ----
  // Mirrors the service's own precondition for `POST /api/jobs/<id>/rerun`: a finished task with a
  // confirmed outlet mapping and a reusable stage-A snapshot.
  function rerunSkipReason(job) {
    if (!isObject(job)) return '任务不存在';
    if (job.status !== 'done') return '任务未完成';
    if (!isObject(job.mapping) || !Object.keys(job.mapping).length) return '尚未确认出口命名';
    if (!isObject(job.a) && !isObject(job.stage_a)) return '缺少可复用的中心线结果';
    return null;
  }

  // ---- compute time: this attempt vs. cumulative (§15.5) ----
  function computeTiming(timing, fallbackTotal) {
    const source = isObject(timing) ? timing : {};
    const num = value => (typeof value === 'number' && isFinite(value) ? value : null);
    // Records written before compute time was tracked (2026-09-17) report 0 s: prefer the summary total then.
    const recorded = [num(source.compute_s), num(source.compute_seconds)];
    const positive = recorded.find(value => value !== null && value > 0);
    const total = positive !== undefined ? positive : [num(fallbackTotal), ...recorded].find(value => value !== null);
    const attempt = num(source.attempt_s);
    const resolved = total === undefined ? null : total;
    // A zero attempt (e.g. a record touched after it finished) says nothing useful; show the total only.
    return {attempt, total: resolved, split: attempt !== null && attempt > 0 && resolved !== null && Math.abs(attempt - resolved) > 0.05};
  }

  // ---- side-by-side display sync (§15.6 / §15.7) ----
  const DISPLAY_KEYS = ['field', 'mode', 'colormap', 'bands', 'log', 'range', 'units', 'pressure_units', 'speed_units', 'thresholds_pa', 'opacity', 'overlay', 'slice'];
  const CROSS_FAMILY_DISPLAY_KEYS = ['colormap', 'bands', 'log', 'opacity'];
  function compareDisplaySubset(state, sameFamily) {
    // Only the display half of a view state travels between frames; the camera never does.
    // Across families (wall vs volume) the field-specific keys are meaningless, so only the shared ones go.
    const source = isObject(state) ? state : {};
    const keys = sameFamily === false ? CROSS_FAMILY_DISPLAY_KEYS : DISPLAY_KEYS;
    const out = {};
    for (const key of keys) if (Object.prototype.hasOwnProperty.call(source, key) && source[key] !== undefined) out[key] = source[key];
    return out;
  }
  // v0.14 (F1): same colour = same value on the compare page.  Given the two frames' view states, return the one
  // colour range both should use: a fixed upper limit = the larger of the two case p99s of the coloured field
  // (``range.case_max``; an older page without it contributes its current ``range.max``).  A fixed range the
  // user typed on both sides (no ``shared`` flag) and already equal is left alone.  Mixed families, volume
  // pages (per-case ranges only) or different coloured fields cannot share one bar: a mismatch text says so.
  const FIELD_SHORT_TEXT = {wss: '峰值 WSS', tawss: 'TAWSS', osi: 'OSI', rrt: 'RRT', ecap: 'ECAP', pressure: '压力', velocity: '速度'};
  function sharedDisplayRange(left, right, leftFamily, rightFamily) {
    const L = isObject(left) ? left : {}, R = isObject(right) ? right : {};
    const fl = leftFamily || L.family || null, fr = rightFamily || R.family || null;
    if (fl && fr && fl !== fr) return {mismatch: 'family', text: '两侧结果类型不同（壁面 / 体场），色标各自独立，颜色不能直接对比。'};
    if (fl === 'volume' || fr === 'volume') return {mismatch: 'volume', text: '体场报告的色标按各自本例范围显示，两侧颜色不能直接对比。'};
    const fieldL = typeof L.field === 'string' ? L.field : 'wss', fieldR = typeof R.field === 'string' ? R.field : 'wss';
    if (fieldL !== fieldR) return {mismatch: 'field', text: `两侧着色字段不同（左 ${FIELD_SHORT_TEXT[fieldL] || fieldL} / 右 ${FIELD_SHORT_TEXT[fieldR] || fieldR}），色标不能直接对比。`};
    const lr = isObject(L.range) ? L.range : {}, rr = isObject(R.range) ? R.range : {};
    const positive = value => { const n = Number(value); return Number.isFinite(n) && n > 0 ? n : null; };
    const userFixed = r => r.mode === 'fixed' && !r.shared && positive(r.max) !== null;
    if (userFixed(lr) && userFixed(rr) && Math.abs(lr.max - rr.max) <= 1e-9 * Math.max(1, Math.abs(lr.max))) return {same: true, max: Number(lr.max), field: fieldL};
    const contribution = r => userFixed(r) ? positive(r.max) : (positive(r.case_max) ?? positive(r.max));
    const a = contribution(lr), b = contribution(rr);
    if (a === null && b === null) return {mismatch: 'range', text: '读不到两侧的色标范围，颜色不能直接对比。'};
    const max = Math.max(a ?? 0, b ?? 0);
    const unchanged = [lr, rr].every(r => r.mode === 'fixed' && r.shared && Math.abs(Number(r.max) - max) <= 1e-9 * Math.max(1, max) && (r.field || fieldL) === fieldL);
    return {range: {mode: 'fixed', min: 0, max, field: fieldL, shared: true}, max, field: fieldL, unchanged};
  }
  // ---- v0.14 job event actions (jobs.py ``_event``) → short Chinese labels; an unknown action shows as written ----
  const EVENT_TEXT = {
    created: '已创建', started: '开始计算', finished: '计算结束', progress: '进度更新', cancel_requested: '请求取消', attempt_aborted: '本次尝试中止',
    rerun_created: '已换发布包重跑', restored: '已从回收站恢复', deleted: '已删除', owner_claimed: '已认领', metadata_updated: '病例信息已修改',
    narrative_updated: '结论已修改', annotations_updated: '标注已保存', findings_review_updated: '发现判定已保存', snapshots_updated: '一页纸配图已更新',
    legacy_release_bound: '已绑定发布包', restart_interrupted: '服务重启时中断', restart_requeued: '服务重启后自动续排',
    drain_requeued: '升级排空时重新排队', device_fallback: '显存不足，已改用 CPU 重算', analysis_rebuilt: '升级后重建分析',
    precompute_started: '后台几何预计算开始', precompute_done: '后台几何预计算完成', precompute_skipped: '后台几何预计算已跳过（无需计算）',
    precompute_cancelled: '后台几何预计算已取消', precompute_failed: '后台几何预计算未完成（阶段 B 自行计算）',
    precompute_incomplete: '后台几何预计算未全部完成（阶段 B 补算）', precompute_interrupted: '后台几何预计算被中断'};
  function eventText(action) {
    const key = String(action === null || action === undefined ? '' : action);
    return Object.prototype.hasOwnProperty.call(EVENT_TEXT, key) ? EVENT_TEXT[key] : key;
  }
  // ---- v0.14 geometry cache: which ``summary.timing_s`` rows were served from the background precompute ----
  // Same mapping as eta.CACHE_KIND_STAGES (cache entry kind → stage) and eta.TIMING_KEYS (timing key → stage).
  const CACHE_KIND_STAGES = {mesh: 'smooth_resample', resample: 'smooth_resample', pointgeom: 'features', morph: 'morphology', morphvol: 'morphology'};
  const TIMING_KEY_STAGES = {smooth_resample: 'smooth_resample', features: 'features', volume_features: 'volume_features', morphology: 'morphology'};
  function cachedStages(geometryCache) {
    const reused = isObject(geometryCache) && Array.isArray(geometryCache.reused) ? geometryCache.reused : [];
    return new Set(reused.map(kind => CACHE_KIND_STAGES[String(kind)]).filter(Boolean));
  }
  function timingPrecomputed(key, stages) { const stage = TIMING_KEY_STAGES[String(key)]; return Boolean(stage && stages && stages.has(stage)); }
  // ---- v0.14 service limits (429 / 413): the server's message, or the per-owner cap with its numbers ----
  function limitErrorText(status, body, message) {
    const b = isObject(body) ? body : {};
    const cap = numberOrNull(b.max_active_jobs), active = numberOrNull(b.active_jobs);
    if (cap !== null) return `排队与计算中的任务已达 ${cap} 个上限${active !== null ? `（当前 ${active} 个）` : ''}，请等部分任务完成后再提交。`;
    const text = typeof message === 'string' ? message.trim() : '';
    if (text) return text;
    if (Number(status) === 429) return '服务正忙（同时处理的请求已达上限），请稍后重试。';
    if (Number(status) === 413) return '请求内容超过服务上限。';
    return '请求未完成，请稍后重试。';
  }
  class EchoGuard {
    // Ignore messages coming back from a frame we just pushed into, so an `applied` reply cannot
    // bounce the state back and forth between the two reports.
    constructor(windowMs, now) { this.windowMs = Number(windowMs) > 0 ? Number(windowMs) : 400; this.now = typeof now === 'function' ? now : () => Date.now(); this.stamps = {}; }
    mark(key) { this.stamps[key] = this.now(); }
    blocked(key) { const at = this.stamps[key]; return at !== undefined && this.now() - at < this.windowMs; }
    clear(key) { if (key === undefined) this.stamps = {}; else delete this.stamps[key]; }
  }

  // ---- cohort overview (§15.13) ----
  function numberOrNull(value) {
    if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }
  function binEdges(values, count) {
    const numbers = [];
    for (const value of values || []) { const number = numberOrNull(value); if (number !== null) numbers.push(number); }
    const bins = Math.max(1, Math.floor(Number(count)) || 10);
    if (!numbers.length) return [];
    let min = numbers[0], max = numbers[0];
    for (const number of numbers) { if (number < min) min = number; if (number > max) max = number; }
    if (!(max > min)) { const pad = Math.abs(min) > 0 ? Math.abs(min) * 0.05 : 0.5; min -= pad; max += pad; }
    const step = (max - min) / bins, edges = [];
    for (let i = 0; i <= bins; i++) edges.push(min + step * i);
    edges[bins] = max;
    return edges;
  }
  function histogram(values, edges) {
    const bounds = Array.isArray(edges) ? edges : [];
    const counts = new Array(Math.max(0, bounds.length - 1)).fill(0);
    if (!counts.length) return counts;
    const last = bounds.length - 1;
    for (const value of values || []) {
      const number = numberOrNull(value);
      if (number === null || number < bounds[0] || number > bounds[last]) continue;
      let index = 0;
      while (index < counts.length - 1 && number >= bounds[index + 1]) index++;
      counts[index]++;
    }
    return counts;
  }
  function cohortFilter(rows, filters) {
    const f = isObject(filters) ? filters : {};
    const atLeast = (value, minimum, scale) => {
      if (minimum === null) return true;
      const number = numberOrNull(value);
      return number !== null && number * (scale || 1) >= minimum;
    };
    const p99 = numberOrNull(f.p99_min), high = numberOrNull(f.high_min), veryHigh = numberOrNull(f.very_high_min), speed = numberOrNull(f.speed_min);
    const diameter = numberOrNull(f.diameter_min);          // §17.4: mm, filled for both families
    return (rows || []).filter(row => {
      if (!isObject(row)) return false;
      if (f.family && row.family !== f.family) return false;
      if (f.release_id && row.release_id !== f.release_id) return false;
      if (f.review_status && (row.review_status || 'unreviewed') !== f.review_status) return false;
      // Area fractions are stored as 0–1 and filtered in percent.
      return atLeast(row.wss_p99_pa, p99) && atLeast(row.area_frac_high, high, 100) && atLeast(row.area_frac_very_high, veryHigh, 100)
        && atLeast(row.speed_p99_m_s, speed) && atLeast(row.max_diameter_mm, diameter);
    });
  }
  function cohortSort(rows, key, direction) {
    const sign = direction === 'desc' ? -1 : 1;
    const blank = value => value === null || value === undefined || value === '' || (typeof value === 'number' && !isFinite(value));
    return [...(rows || [])].sort((a, b) => {
      const left = isObject(a) ? a[key] : null, right = isObject(b) ? b[key] : null;
      if (blank(left) && blank(right)) return 0;
      if (blank(left)) return 1;            // missing values always sink, in both directions
      if (blank(right)) return -1;
      const ln = numberOrNull(left), rn = numberOrNull(right);
      if (ln !== null && rn !== null) return (ln - rn) * sign;
      return String(left).localeCompare(String(right)) * sign;
    });
  }
  function populationValues(population, rows, releaseId) {
    // The reference histogram belongs to one release; prefer the filtered release, else the most common one.
    const source = isObject(population) ? population : {};
    const pick = id => {
      const entry = source[id];
      if (!isObject(entry) || !Array.isArray(entry.values_pa)) return null;
      const values = entry.values_pa.map(numberOrNull).filter(value => value !== null);
      return values.length ? {release_id: id, values, case_count: numberOrNull(entry.case_count) ?? values.length, reference_set: entry.reference_set || '', metric: entry.metric || 'p99_pa'} : null;
    };
    if (releaseId) return pick(releaseId);
    const counts = new Map();
    for (const row of rows || []) if (isObject(row) && row.release_id) counts.set(row.release_id, (counts.get(row.release_id) || 0) + 1);
    const ordered = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(entry => entry[0]);
    for (const id of [...ordered, ...Object.keys(source)]) { const found = pick(id); if (found) return found; }
    return null;
  }
  function cohortReleases(rows) {
    const seen = [];
    for (const row of rows || []) if (isObject(row) && row.release_id && !seen.includes(row.release_id)) seen.push(row.release_id);
    return seen.sort();
  }

  // =============================================================================================
  // v0.12 (contract §19): routing, work groups, overview, release kinds, cycle cards, alerts,
  // morphology reliability, outlet-review wording, patient timeline.  All pure; app.js renders.
  // =============================================================================================

  // Numbers without scientific notation (same rule as report_common.formatValue, kept local so the
  // core stays dependency-free under node).
  function formatNumber(value, digits, maxDecimals) {
    const number = numberOrNull(value);
    if (number === null) return '—';
    const sig = digits || 3, maxDec = maxDecimals === undefined ? 4 : maxDecimals, abs = Math.abs(number);
    if (abs === 0) return '0';
    if (abs >= Math.pow(10, sig)) return String(Math.round(number));
    if (abs < Math.pow(10, -maxDec)) return '0';
    return number.toFixed(Math.max(0, Math.min(maxDec, sig - 1 - Math.floor(Math.log10(abs)))));
  }
  function percentText(fraction, decimals) {
    const number = numberOrNull(fraction);
    return number === null ? '—' : `${(number * 100).toFixed(decimals === undefined ? 0 : decimals)}%`;
  }
  const timeMs = value => { const ms = Date.parse(value || ''); return Number.isFinite(ms) ? ms : null; };

  // ---- routing: #job=<id> (§19.9) ----
  const JOB_ID = /^[A-Za-z0-9_-]{1,80}$/;
  function parseJobHash(hash) {
    const text = String(hash || '').replace(/^#/, '');
    for (const part of text.split('&')) {
      const eq = part.indexOf('=');
      if (eq < 0 || part.slice(0, eq) !== 'job') continue;
      let id;
      try { id = decodeURIComponent(part.slice(eq + 1)); } catch (_) { return null; }
      return JOB_ID.test(id) ? id : null;
    }
    return null;
  }
  function jobHash(id) { return JOB_ID.test(String(id || '')) ? `#job=${encodeURIComponent(String(id))}` : ''; }

  // ---- work groups: 需要处理 → 计算中与排队 → 待审阅 → 已审阅 (→ 已取消) ----
  const ACTION_STATUSES = ['awaiting_input', 'awaiting_confirmation', 'awaiting_outlets', 'failed', 'interrupted'];
  const ACTIVE_STATUSES = ['queued', 'queued_A', 'queued_B', 'running', 'running_A', 'running_B', 'cancelling'];
  const CONFIRM_STATUSES = ['awaiting_confirmation', 'awaiting_outlets', 'awaiting_input'];
  const WORK_GROUPS = [
    {key: 'action', label: '需要处理', open: true, tone: 'action'},
    {key: 'active', label: '计算中与排队', open: true, tone: 'active'},
    {key: 'pending', label: '待审阅', open: true, tone: 'pending'},
    {key: 'reviewed', label: '已审阅', open: false, tone: ''},
    {key: 'other', label: '已取消', open: false, tone: ''},
  ];
  function workGroupOf(job) {
    if (!isObject(job)) return 'other';
    if (ACTION_STATUSES.includes(job.status)) return 'action';
    if (ACTIVE_STATUSES.includes(job.status)) return 'active';
    if (isPendingReview(job)) return 'pending';
    if (isReviewed(job)) return 'reviewed';
    return 'other';
  }
  function workGroups(jobs) {
    const out = {action: [], active: [], pending: [], reviewed: [], other: []};
    for (const job of jobs || []) out[workGroupOf(job)].push(job);
    return out;
  }
  // Oldest waiting first: a batch is worked through in upload order, whatever the list's sort.
  function nextJob(jobs, currentId, predicate) {
    const list = (jobs || []).filter(job => isObject(job) && job.id && job.id !== currentId && predicate(job));
    list.sort((a, b) => (timeMs(a.created_at) ?? 0) - (timeMs(b.created_at) ?? 0) || String(a.id).localeCompare(String(b.id)));
    return list[0] || null;
  }
  const nextToConfirm = (jobs, currentId) => nextJob(jobs, currentId, job => CONFIRM_STATUSES.includes(job.status));
  const nextToReview = (jobs, currentId) => nextJob(jobs, currentId, isPendingReview);
  function stepId(ids, currentId, delta) {
    const list = (ids || []).filter(Boolean);
    if (!list.length) return null;
    const index = list.indexOf(currentId);
    if (index < 0) return delta > 0 ? list[0] : list[list.length - 1];
    return list[Math.max(0, Math.min(list.length - 1, index + delta))];
  }

  // ---- overview / quick filters ----
  const QUICK_FILTERS = {action: '需要处理', active: '计算中与排队', pending: '待审阅', week: '近 7 天完成'};
  // updated_at moves on every metadata edit or review; the creation time is the stable proxy for a finish.
  const finishedMs = job => timeMs(job.finished_at) ?? timeMs(job.created_at) ?? timeMs(job.updated_at);
  function quickFilter(jobs, key, nowMs) {
    const now = Number.isFinite(nowMs) ? nowMs : Date.now();
    const list = (jobs || []).filter(isObject);
    if (key === 'action' || key === 'active' || key === 'pending') return list.filter(job => workGroupOf(job) === key);
    if (key === 'week') return list.filter(job => job.status === 'done' && (finishedMs(job) ?? -Infinity) >= now - 7 * 86400000);
    return list;
  }
  function overviewModel(jobs, nowMs) {
    const counts = {};
    for (const key of Object.keys(QUICK_FILTERS)) counts[key] = quickFilter(jobs, key, nowMs).length;
    const recent = (jobs || []).filter(job => isObject(job) && job.status === 'done')
      .sort((a, b) => (finishedMs(b) ?? 0) - (finishedMs(a) ?? 0)).slice(0, 5);
    return {counts, recent, total: (jobs || []).length};
  }

  // ---- release kinds: WSS / WSS + TAWSS + OSI / 压力 + 速度体场 ----
  const KIND_TEXT = {wall: 'WSS', cycle: 'WSS + TAWSS + OSI', volume: '压力 + 速度体场'};
  function releaseKind(release, family) {
    const r = isObject(release) ? release : {};
    const c = isObject(r.contract) ? r.contract : {};
    const fields = isObject(c.fields) ? c.fields : {};
    const id = String(r.id || r.release || r.registry_id || r.name || (typeof release === 'string' ? release : '') || '');
    if (fields.tawss || fields.osi || c.protocol === 'single_frame_wss_cycle_multi' || c.target === 'wss_cycle_multi') return 'cycle';
    if (fields.velocity || fields.pressure || c.protocol === 'single_frame_volume' || c.target === 'pressure_velocity') return 'volume';
    if (fields.wss || c.protocol === 'single_frame_wss') return 'wall';
    if (family === 'volume') return 'volume';
    if (/3head|cycle|tawss/i.test(id)) return 'cycle';
    if (/PF6|VF6|volume/i.test(id)) return 'volume';
    if (family === 'wall' || id) return 'wall';
    return null;
  }
  function summaryKind(summary, fallback) {
    const s = isObject(summary) ? summary : {};
    const fields = isObject(s.fields) ? s.fields : {};
    if (isObject(s.cycle) || fields.tawss || fields.osi) return 'cycle';
    if (fields.velocity || fields.pressure || isObject(s.volume_statistics) && Object.keys(s.volume_statistics).length && !s.peak) return 'volume';
    return fallback || 'wall';
  }

  // ---- M1 cycle quantities (§19.2) ----
  function cycleSummary(cycle) {
    if (!isObject(cycle) || !isObject(cycle.fields)) return null;
    const t = isObject(cycle.fields.tawss) ? cycle.fields.tawss : {}, o = isObject(cycle.fields.osi) ? cycle.fields.osi : {};
    const st = isObject(cycle.stagnation) ? cycle.stagnation : {};
    const tf = isObject(t.area_frac) ? t.area_frac : {}, of = isObject(o.area_frac) ? o.area_frac : {};
    const tth = Array.isArray(t.thresholds) ? t.thresholds : [0.4, 4, 7], oth = Array.isArray(o.thresholds) ? o.thresholds : [0.1, 0.2, 0.3];
    let main = null;
    for (const [name, row] of Object.entries(isObject(st.per_branch) ? st.per_branch : {})) {
      const area = numberOrNull(isObject(row) ? row.area_mm2 : null);
      if (area !== null && (!main || area > main.area_mm2)) main = {name, area_mm2: area};
    }
    const area = numberOrNull(st.area_mm2);
    const out = {
      tawss: {mean: numberOrNull(t.mean), p99: numberOrNull(t.p99), lowFrac: numberOrNull(tf.low), lowThreshold: numberOrNull(tth[0]) ?? 0.4},
      osi: {mean: numberOrNull(o.mean), highFrac: numberOrNull(of.above_t0), veryHighFrac: numberOrNull(of.above_t2),
        t0: numberOrNull(oth[0]) ?? 0.1, t2: numberOrNull(oth[2]) ?? 0.3},
      stagnation: {frac: numberOrNull(st.area_frac), areaMm2: area, areaCm2: area === null ? null : area / 100, mainBranch: main ? main.name : null},
      periodS: numberOrNull(isObject(cycle.definition) ? cycle.definition.period_s : null),
    };
    // v0.13 derived indices (1/Pa); absent in older summaries → null, and the cards stay away.
    for (const [key, fallback] of [['rrt', 5], ['ecap', 1.4]]) {
      const d = isObject(cycle.fields[key]) ? cycle.fields[key] : null, df = d && isObject(d.area_frac) ? d.area_frac : {};
      out[key] = d && numberOrNull(d.mean) !== null ? {mean: numberOrNull(d.mean), p99: numberOrNull(d.p99), highFrac: numberOrNull(df.above_t0),
        t0: numberOrNull(Array.isArray(d.thresholds) ? d.thresholds[0] : null) ?? fallback} : null;
    }
    if (out.tawss.mean === null && out.osi.mean === null && out.stagnation.frac === null) return null;
    return out;
  }

  // ---- ensemble size and the quality wording (§19.2: "五模型" only when there are five) ----
  function modelCount(summary, job) {
    const s = isObject(summary) ? summary : {}, mr = isObject(s.model_release) ? s.model_release : {};
    const audit = isObject(s.audit) && isObject(s.audit.quality_audit) ? s.audit.quality_audit : {};
    const lengths = [mr.loaded_weights, mr.models, mr.weights].filter(Array.isArray).map(list => list.length).filter(n => n > 0);
    const candidates = [numberOrNull(audit.seed_count), ...lengths,
      numberOrNull(isObject(job) && isObject(job.compute) ? job.compute.seed_count : null),
      numberOrNull(isObject(job) && isObject(job.model_release) ? job.model_release.models_count : null)];
    const found = candidates.find(value => value !== null && value > 0);
    return found === undefined ? null : Math.round(found);
  }
  function ensembleWord(count) { const n = numberOrNull(count); return n === 5 ? '五模型' : n ? `${n} 个模型` : '模型'; }
  function qualityNote(quality, count) {
    const n = numberOrNull(count);
    const reasons = (isObject(quality) && Array.isArray(quality.reasons) ? quality.reasons : []).filter(text => typeof text === 'string' && text.trim());
    if (reasons.length) return reasons.map(text => (n && n !== 5 ? text.replace(/五模型/g, `${n} 个模型`) : text)).join('；');
    return `${ensembleWord(n)}集成离散度在当前分流阈值内。逐点标准差保存在质量审计文件中。`;
  }

  // ---- B4 alert banner: geometry outside the release's declared range, or quality not good ----
  const GEOMETRY_FIELD = {length_mm: '长度', radius_min_mm: '最小半径', radius_median_mm: '中位半径', radius_max_mm: '最大半径',
    spacing_mm: '点云间距', variation: '表面粗糙度', surface_variation_median: '表面粗糙度'};
  function referencePathLabel(path) {
    const parts = String(path || '').split('.');
    if (parts[0] === 'geometry' && parts.length >= 3) return `${parts[1]}${GEOMETRY_FIELD[parts[2]] || parts[2]}`;
    if (parts[0] === 'cloud' && parts[1]) return GEOMETRY_FIELD[parts[1]] || parts[1];
    return String(path || '');
  }
  const trimNumber = value => formatNumber(value).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
  function alertModel(summary) {
    const s = isObject(summary) ? summary : {};
    const ref = isObject(s.reference_assessment) ? s.reference_assessment : {};
    const items = [];
    const outside = (Array.isArray(ref.checks) ? ref.checks : []).filter(check => isObject(check) && check.status === 'review');
    for (const check of outside) {
      const units = check.units ? ` ${check.units}` : '';
      const value = numberOrNull(check.value), low = numberOrNull(check.min), high = numberOrNull(check.max);
      const side = value !== null && high !== null && value > high ? '高于' : value !== null && low !== null && value < low ? '低于' : '超出';
      items.push({kind: 'reference', path: check.path, text: `${referencePathLabel(check.path)} ${trimNumber(value)}${units}，${side}模型训练范围 ${trimNumber(low)}–${trimNumber(high)}${units}`});
    }
    if (ref.status === 'review' && !outside.length) {
      for (const reason of Array.isArray(ref.reasons) ? ref.reasons : []) if (typeof reason === 'string' && reason.trim()) items.push({kind: 'reference', text: reason.trim()});
    }
    const q = isObject(s.quality) ? s.quality : null;
    const qualityAlert = Boolean(q && q.level && q.level !== 'good');
    if (qualityAlert) {
      const reasons = (Array.isArray(q.reasons) ? q.reasons : []).filter(text => typeof text === 'string' && text.trim());
      items.push({kind: 'quality', level: q.level, text: `${q.label || '集成质量需要复核'}${reasons.length ? `：${reasons.join('；')}` : ''}`});
    }
    if (!items.length) return null;
    const parts = [];
    const refCount = items.filter(item => item.kind === 'reference').length;
    if (refCount) parts.push(outside.length ? `${outside.length} 项几何测量超出模型训练范围` : '几何测量超出模型训练范围');
    if (qualityAlert) parts.push(q.level === 'poor' ? '模型集成离散度较高' : '模型集成存在不确定性');
    return {title: `${parts.join('，')}，结果需要复核。`, items, severity: qualityAlert && q.level === 'poor' ? 'serious' : 'warning'};
  }

  // ---- morphology reliability, one sentence + per-branch details (§19.5 wording) ----
  const RELIABILITY_NOTE = /站重新定向|截面不可靠/;
  function reliabilitySummary(morphology) {
    if (!isObject(morphology)) return null;
    const notes = (Array.isArray(morphology.notes) ? morphology.notes : []).filter(text => typeof text === 'string' && text.trim()).map(text => text.trim());
    const other = notes.filter(text => !RELIABILITY_NOTE.test(text));
    const rows = [];
    const aorta = isObject(morphology.aorta) ? morphology.aorta : null;
    const push = (name, part) => {
      if (!isObject(part)) return;
      const reoriented = numberOrNull(part.n_reoriented), excluded = numberOrNull(part.n_excluded);
      if (reoriented === null && excluded === null) return;
      rows.push({name: name || '分支', reoriented: reoriented || 0, excluded: excluded || 0, total: numberOrNull(part.n_stations) ?? numberOrNull(part.n_reliable)});
    };
    if (aorta) push(aorta.name || '主动脉', aorta);
    for (const branch of Array.isArray(morphology.branches) ? morphology.branches : []) if (isObject(branch) && branch.segment_id !== (aorta && aorta.segment_id)) push(branch.name, branch);
    let reoriented = 0, excluded = 0, known = rows.length > 0;
    if (known) for (const row of rows) { reoriented += row.reoriented; excluded += row.excluded; }
    else {
      for (const text of notes) {
        const match = /(\d+)\s*站重新定向[，,、\s]*(\d+)\s*站/.exec(text);
        if (match) { known = true; reoriented += Number(match[1]); excluded += Number(match[2]); }
      }
    }
    const details = rows.length
      ? rows.filter(row => row.reoriented || row.excluded).map(row => `${row.name}：${row.reoriented} 站重定向，${row.excluded} 站不可靠${row.total !== null ? `（共 ${row.total} 站）` : ''}`)
      : notes.filter(text => RELIABILITY_NOTE.test(text));
    return {known, reoriented, excluded, text: known ? `截面可靠性：${reoriented} 站重定向，${excluded} 站不可靠（已排除）` : null, details, other};
  }

  // ---- outlet confirmation: one status sentence and the reasons in plain words (C3) ----
  const OUTLET_REASONS = [
    [/^(左|右)侧髂内\s*\/\s*髂外区分置信度\s*([\d.]+)%/, m => `${m[1]}侧髂内与髂外的区分把握度只有 ${Math.round(Number(m[2]))}%，请重点核对${m[1]}侧的两个出口。`],
    [/几何置信度代理\s*([\d.]+)%\s*低于门槛\s*([\d.]+)%/, m => `整体命名把握度 ${Math.round(Number(m[1]))}%，低于自动放行要求的 ${Math.round(Number(m[2]))}%。`],
    [/代理值|未校准/, () => '把握度由几何形态估算，不是经过临床数据校准的概率。'],
    [/患者方向|世界\s*XYZ|左右语义/, () => 'STL 文件不带患者方位信息，左右需要对照原始影像确认。'],
    [/命名建议自身标记为需要确认/, () => '自动命名在至少一侧的内 / 外区分上不够确定。'],
    [/profile|联合正确率/, () => '当前模型版本尚未完成出口命名的独立验证，因此每例都请人工确认。'],
  ];
  function humanOutletReason(text) {
    const raw = String(text || '').trim();
    if (!raw) return '';
    for (const [pattern, render] of OUTLET_REASONS) { const match = pattern.exec(raw); if (match) return render(match); }
    return raw;
  }
  function outletReview(proposal, jobDetail) {
    const p = isObject(proposal) ? proposal : {};
    const gate = isObject(p.confidence_gate) ? p.confidence_gate : {};
    const detailReasons = typeof jobDetail === 'string' && jobDetail.includes('原因：') ? jobDetail.split('原因：').slice(1).join('').split(/[；;]/) : [];
    const raw = [].concat(p.confidence_reasons || [], gate.reasons || [], p.flags || [], p.direction_note ? [p.direction_note] : [], detailReasons);
    const reasons = [];
    for (const text of raw) { const human = humanOutletReason(text); if (human && !reasons.includes(human)) reasons.push(human); }
    const confidence = numberOrNull(p.confidence);
    const required = p.confirmation_required !== false;
    const pct = confidence === null ? null : Math.round(confidence * 100);
    const headline = pct === null ? '请结合原始影像核对出口命名' : required ? `自动命名置信度 ${pct}%，需要人工核对` : `自动命名置信度 ${pct}%，已自动通过`;
    return {headline, confidence, required, reasons};
  }

  // ---- detail-page findings (G4): the service already orders findings_top like the one-pager; the
  // workbench keeps that order, never filters by kind, and only labels each row ----
  const GEOMETRY_FINDINGS = ['max_diameter', 'min_radius'];
  function findingChip(item) {
    const f = isObject(item) ? item : {};
    if (GEOMETRY_FINDINGS.includes(f.kind)) return {text: '几何', tone: 'geometry'};
    if (f.severity === 'attention') return {text: '需关注', tone: 'attention'};
    if (f.severity === 'info') return {text: '参考', tone: 'info'};
    return {text: '记录', tone: 'note'};
  }
  function findingsForCard(items, limit) {
    return (Array.isArray(items) ? items : []).filter(item => isObject(item) && item.label).slice(0, limit === undefined ? 3 : limit);
  }
  const findingUnits = item => (isObject(item) && item.units && item.units !== '1' ? ` ${item.units}` : '');

  // ---- stage progress and remaining time (§21.2 / §21.3) ----
  // "几秒" below 10 s, 5-second steps below a minute, then "约 N 分 M 秒" (whole minutes from 10 min on).
  function formatDuration(seconds) {
    const s = numberOrNull(seconds);
    if (s === null) return '';
    if (s < 10) return '几秒';
    if (s < 57.5) return `约 ${Math.max(10, Math.round(s / 5) * 5)} 秒`;
    if (s >= 570) return `约 ${Math.round(s / 60)} 分钟`;
    const total = Math.round(s / 5) * 5, minutes = Math.floor(total / 60), rest = total % 60;
    return rest ? `约 ${minutes} 分 ${rest} 秒` : `约 ${minutes} 分钟`;
  }
  // Same rule as eta.stage_remaining: expected − elapsed until 80 %, then a shrinking 20 % tail, never negative.
  function stageRemaining(expected, elapsed) {
    const e = Math.max(0, numberOrNull(expected) || 0), t = Math.max(0, numberOrNull(elapsed) || 0), knee = 0.8 * e;
    if (t <= knee || e <= 0) return Math.max(0, e - t);
    return 0.2 * e * (knee / t);
  }
  const secondsText = value => { const v = numberOrNull(value); return v === null ? '—' : `${v < 10 ? v.toFixed(1) : Math.round(v)} 秒`; };
  // ``ageS`` = seconds since the snapshot arrived: the running stage keeps advancing between server updates
  // (measured on the browser's own clock, so a skewed server clock does not matter).
  function etaView(eta, opts) {
    if (!isObject(eta) || !Array.isArray(eta.stages) || !eta.stages.length) return null;
    const age = Math.max(0, numberOrNull(isObject(opts) ? opts.ageS : null) || 0);
    const minPct = isObject(opts) && numberOrNull(opts.minPct) !== null ? numberOrNull(opts.minPct) : 2.5;
    // v0.14: ``eta.precomputed`` = stages whose mesh-only work the background precompute already did while the
    // outlets were being confirmed; they are marked 「已预计算」 so their near-zero time does not look like a fault.
    const precomputed = new Set(Array.isArray(eta.precomputed) ? eta.precomputed.map(String) : []);
    const rows = eta.stages.filter(isObject).map(stage => {
      const expected = Math.max(0, numberOrNull(stage.expected_s) || 0);
      const state = ['done', 'running', 'pending'].includes(stage.state) ? stage.state : 'pending';
      let elapsed = numberOrNull(stage.elapsed_s);
      if (state === 'running') elapsed = (elapsed || 0) + age;
      return {key: String(stage.key || ''), label: String(stage.label || stage.key || '阶段'), expected, elapsed, state,
        overdue: state === 'running' && elapsed > expected && expected > 0, precomputed: precomputed.has(String(stage.key || ''))};
    });
    // ``waiting``: the job holds a worker but waits for the previous stage-B job (serialised) → shown as queued.
    const rawStatus = eta.status || (rows.some(row => row.state === 'running') ? 'running' : null);
    const waiting = eta.waiting === true && rawStatus === 'running';
    const status = waiting ? 'queued' : rawStatus;
    const total = rows.reduce((sum, row) => sum + row.expected, 0);
    let widths = rows.map(row => (total > 0 ? row.expected / total * 100 : 100 / rows.length));
    widths = widths.map(width => Math.max(minPct, width));
    const scale = 100 / widths.reduce((sum, width) => sum + width, 0);
    let filled = 0, currentIndex = -1;
    const segments = rows.map((row, index) => {
      const width = +(widths[index] * scale).toFixed(3);
      let fill = row.state === 'done' ? 100 : 0;
      if (row.state === 'running' && !waiting) { currentIndex = index; fill = row.expected > 0 ? Math.min(96, row.elapsed / row.expected * 100) : 50; }
      filled += width * fill / 100;
      const title = row.state === 'done'
        ? `${row.label}：已完成${row.elapsed !== null ? `，用时 ${secondsText(row.elapsed)}` : ''}（预计 ${secondsText(row.expected)}）`
        : row.state === 'running'
          ? `${row.label}：进行中，已用 ${secondsText(row.elapsed)} / 预计 ${secondsText(row.expected)}${row.overdue ? '，比预计慢' : ''}`
          : `${row.label}：预计 ${secondsText(row.expected)}`;
      // A job held by the stage-B lock has not started its stage yet: draw it as pending.
      return {...row, state: waiting && row.state === 'running' ? 'pending' : row.state, width, fill: +fill.toFixed(1), title: row.precomputed ? `${title}（已预计算）` : title};
    });
    let remaining;
    if (status === 'running' && !waiting) remaining = rows.reduce((sum, row) => sum + (row.state === 'running' ? stageRemaining(row.expected, row.elapsed) : row.state === 'pending' ? row.expected : 0), 0);
    else remaining = numberOrNull(eta.remaining_s);
    const segmentRemaining = numberOrNull(eta.segment_remaining_s);
    const queueAhead = numberOrNull(eta.queue_ahead), queueWait = numberOrNull(eta.queue_ahead_s);
    const n = rows.length, current = currentIndex >= 0 ? segments[currentIndex] : null;
    let headline = '', sub = '';
    const manual = eta.segment === 'A' ? '，不含人工确认出口' : '';
    if (status === 'queued') {
      const wait = queueWait === null ? null : Math.max(0, queueWait - age);
      headline = queueAhead ? `前面 ${queueAhead} 个，${wait === null ? '稍后' : `${formatDuration(wait)}后`}开始` : waiting ? '等前一个任务算完后开始' : '即将开始';
      // Waiting and computing add up: the queue estimate plus this job's own work.
      sub = remaining !== null ? `预计${formatDuration((wait || 0) + remaining)}后出结果（含排队${manual}）` : '';
    } else if (status === 'running') {
      const overdue = currentIndex >= 0 && segments[currentIndex].overdue;
      headline = overdue && remaining < 10 ? '即将完成' : `预计还需${formatDuration(remaining)}${manual ? `（${manual.slice(1)}）` : ''}`;
      sub = current ? `正在：${current.label}（第 ${currentIndex + 1} / ${n} 步）${current.overdue ? ' · 比预计慢' : ''}` : '';
    } else if (status === 'awaiting_confirmation') {
      headline = remaining !== null ? `确认后${formatDuration(remaining)}出结果` : '';
    } else if (status === 'awaiting_input') {
      const a = segmentRemaining !== null ? segmentRemaining : remaining;
      headline = a !== null ? `确认后${formatDuration(a)}完成中心线提取` : '';
    }
    const history = numberOrNull(eta.n_history), faces = numberOrNull(eta.faces);
    const basis = eta.basis === 'history' ? `按 ${history || 0} 例同类历史耗时估计` : '按默认耗时估计（同类历史少于 3 例）';
    return {status, waiting, segments, currentIndex, remaining, progress: Math.round(Math.min(100, filled)), headline, sub,
      basis: faces ? `${basis} · 输入 ${Math.round(faces).toLocaleString('en-US')} 个面片` : basis,
      queueAhead, queueWait};
  }
  // One short line for a list row: a thin bar (running only) and the remaining time.
  function etaRowSummary(eta, opts) {
    const view = etaView(eta, opts);
    if (!view) return null;
    if (view.status === 'running') return {pct: view.progress, text: view.headline === '即将完成' ? '即将完成' : `还需${formatDuration(view.remaining)}`, state: 'running'};
    if (view.status === 'queued') return {pct: null, text: view.queueAhead ? `前面 ${view.queueAhead} 个` : isObject(eta) && eta.waiting ? '等待前一个任务' : '即将开始', state: 'queued'};
    if (view.status === 'awaiting_confirmation' && view.remaining !== null) return {pct: null, text: `确认后${formatDuration(view.remaining)}`, state: 'waiting'};
    return null;
  }

  // ---- patient timeline (§19.3) ----
  const TIMELINE_METRICS = {
    max_diameter_mm: {label: '管腔最大直径', units: 'mm', group: 'geometry'},
    sac_volume_ml: {label: '瘤体体积', units: 'mL', group: 'geometry'},
    wss_p99_pa: {label: 'WSS p99', units: 'Pa', group: 'model'},
    speed_p99_m_s: {label: '速度 p99', units: 'm/s', group: 'model'},
  };
  const KIND_COLOR = {wall: '#176caa', cycle: '#0f8f6b', volume: '#c2630f'};
  const mainMetricOf = kind => (kind === 'volume' ? 'speed_p99_m_s' : 'wss_p99_pa');
  function timelineModel(data, preferredRelease) {
    if (!isObject(data)) return null;
    const scans = (Array.isArray(data.scans) ? data.scans : []).filter(isObject);
    const releases = [];
    const addRelease = (id, label, family) => {
      if (!id || releases.some(item => item.id === id)) return;
      const kind = releaseKind(id, family);
      releases.push({id, label: label || KIND_TEXT[kind] || id, kind, color: KIND_COLOR[kind] || '#176caa', metric: mainMetricOf(kind)});
    };
    for (const scan of scans) {
      for (const job of Array.isArray(scan.jobs) ? scan.jobs : []) if (isObject(job)) addRelease(job.release_id, job.release_label, job.family);
      for (const id of Object.keys(isObject(scan.models) ? scan.models : {})) addRelease(id, null, null);
    }
    // Same-kind releases would share a colour; give later ones a darker/lighter variant of the fixed order.
    const used = {};
    for (const release of releases) { used[release.kind] = (used[release.kind] || 0) + 1; if (used[release.kind] > 1) release.color = ['#0b4f80', '#5b9fd0', '#8a4a0c'][(used[release.kind] - 2) % 3]; }
    const rows = scans.map((scan, index) => {
      const geometry = isObject(scan.geometry) ? scan.geometry : {}, models = isObject(scan.models) ? scan.models : {};
      const jobs = (Array.isArray(scan.jobs) ? scan.jobs : []).filter(job => isObject(job) && job.job_id);
      const compareJob = jobs.find(job => job.release_id === preferredRelease) || jobs[0] || null;
      const metrics = {};
      for (const release of releases) metrics[release.id] = numberOrNull(isObject(models[release.id]) ? models[release.id][release.metric] : null);
      return {index, sha: scan.input_sha256 || String(index), date: scan.date || '', dateSource: scan.date_source || '', label: scan.scan_label || '',
        caseId: scan.case_id || '', maxDiameter: numberOrNull(geometry.max_diameter_mm), sacVolume: numberOrNull(geometry.sac_volume_ml),
        metrics, jobs, compareJobId: compareJob ? compareJob.job_id : null};
    });
    const pointsFor = pick => rows.map(row => ({date: row.date, value: pick(row), sha: row.sha})).filter(point => point.value !== null);
    const charts = [];
    const chart = (key, lines) => {
      const usable = lines.filter(line => line.points.length >= 2);
      if (usable.length) charts.push({key, ...TIMELINE_METRICS[key], lines: usable});
    };
    chart('max_diameter_mm', [{id: 'geometry', label: '管腔最大直径', color: '#18324b', points: pointsFor(row => row.maxDiameter)}]);
    chart('sac_volume_ml', [{id: 'geometry', label: '瘤体体积', color: '#18324b', points: pointsFor(row => row.sacVolume)}]);
    for (const key of ['wss_p99_pa', 'speed_p99_m_s']) {
      chart(key, releases.filter(release => release.metric === key).map(release => ({id: release.id, label: release.label, color: release.color, points: pointsFor(row => row.metrics[release.id])})));
    }
    const growthOf = block => {
      if (!isObject(block)) return null;
      const perYear = numberOrNull(block.per_year);
      if (perYear === null) return null;
      return {perYear, delta: numberOrNull(block.delta), days: numberOrNull(block.days),
        from: isObject(block.from) ? block.from.date || '' : '', to: isObject(block.to) ? block.to.date || '' : ''};
    };
    const growth = isObject(data.growth) ? data.growth : {}, recent = isObject(data.growth_recent) ? data.growth_recent : {};
    return {
      patientId: data.patient_id || '', nScans: numberOrNull(data.n_scans) ?? scans.length, rows, releases, charts,
      growth: {diameter: growthOf(growth.max_diameter_mm), volume: growthOf(growth.sac_volume_ml),
        diameterRecent: growthOf(recent.max_diameter_mm), volumeRecent: growthOf(recent.sac_volume_ml)},
      notes: (Array.isArray(data.notes) ? data.notes : []).filter(text => typeof text === 'string' && text.trim()),
    };
  }
  // Pixel geometry for one small-multiple line chart (one y-axis per chart, time-scaled x).
  function sparkGeometry(lines, opts) {
    const o = isObject(opts) ? opts : {};
    const width = o.width || 300, height = o.height || 110, pad = Object.assign({left: 8, right: 8, top: 14, bottom: 20}, o.pad || {});
    const all = [];
    for (const line of lines || []) for (const point of line.points || []) { const value = numberOrNull(point.value); if (value !== null) all.push({ms: timeMs(point.date), value}); }
    if (!all.length) return null;
    const dated = all.every(point => point.ms !== null);
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const point of all) { if (dated) { xMin = Math.min(xMin, point.ms); xMax = Math.max(xMax, point.ms); } yMin = Math.min(yMin, point.value); yMax = Math.max(yMax, point.value); }
    if (!(yMax > yMin)) { const padY = Math.abs(yMin) > 0 ? Math.abs(yMin) * 0.05 : 1; yMin -= padY; yMax += padY; }
    else { const padY = (yMax - yMin) * 0.12; yMin -= padY; yMax += padY; }
    const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
    const out = [];
    for (const line of lines || []) {
      const points = (line.points || []).map((point, index) => ({point, index, value: numberOrNull(point.value), ms: timeMs(point.date)})).filter(item => item.value !== null);
      const count = Math.max(1, (lines[0] && lines[0].points ? lines[0].points.length : points.length) - 1);
      const mapped = points.map(item => {
        const fx = dated ? (xMax > xMin ? (item.ms - xMin) / (xMax - xMin) : 0.5) : item.index / count;
        return {x: +(pad.left + fx * plotW).toFixed(2), y: +(pad.top + (1 - (item.value - yMin) / (yMax - yMin)) * plotH).toFixed(2), value: item.value, date: item.point.date || ''};
      });
      out.push({id: line.id, label: line.label, color: line.color, points: mapped, d: mapped.map((p, i) => `${i ? 'L' : 'M'}${p.x} ${p.y}`).join(' ')});
    }
    return {width, height, pad, yMin, yMax, dated, lines: out, baseline: height - pad.bottom};
  }

  // ---- v0.15 round 15 (front-end quick wins): waits, upload checks, identifier hints ----
  // Elapsed / waiting time for people: seconds → minutes → hours → days ("已等待 11460 分" was unreadable).
  function waitText(seconds) {
    const s = numberOrNull(seconds);
    if (s === null) return '—';
    const t = Math.max(0, Math.round(s));
    if (t < 60) return `${t} 秒`;
    if (t < 3600) return `${Math.floor(t / 60)} 分 ${t % 60} 秒`;
    if (t < 86400) { const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60); return m ? `${h} 小时 ${m} 分` : `${h} 小时`; }
    const d = Math.floor(t / 86400), h = Math.floor(t % 86400 / 3600);
    return h ? `${d} 天 ${h} 小时` : `${d} 天`;
  }
  function formatBytes(bytes) {
    const n = numberOrNull(bytes);
    if (n === null || n < 0) return '—';
    if (n < 1024) return `${Math.round(n)} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10 * 1024 ? 1 : 0)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1048576).toFixed(n < 10 * 1048576 ? 1 : 0)} MB`;
    return `${(n / 1073741824).toFixed(1)} GB`;
  }
  // Browser-side check of the chosen files before anything is sent (the service repeats every check).
  // Returns {rows:[{index,name,size,ok,reason,warn}], valid:[File], invalid:number}.
  const UPLOAD_MAX_BYTES = 128 * 1024 * 1024;
  function checkUploadFiles(files, opts) {
    const o = isObject(opts) ? opts : {};
    const maxBytes = numberOrNull(o.maxBytes) || UPLOAD_MAX_BYTES;
    const list = Array.from(files || []), seen = new Map(), rows = [];
    list.forEach((file, index) => {
      const name = String(file && file.name || ''), size = numberOrNull(file && file.size);
      let reason = '', warn = '';
      if (!/\.stl$/i.test(name)) reason = '不是 .stl 文件，不会上传';
      else if (!size) reason = '文件是空的（0 字节），不会上传';
      else if (size > maxBytes) reason = `超过单个文件上限 ${formatBytes(maxBytes)}，不会上传；请先在建模软件中抽稀网格`;
      else if (seen.has(name.toLowerCase())) warn = `与第 ${seen.get(name.toLowerCase()) + 1} 个文件同名，两例会得到相同的病例编号`;
      else { seen.set(name.toLowerCase(), index); if (o.nameHint !== false && nameLikeFilename(name)) warn = NAME_LIKE_TEXT; }
      rows.push({index, name, size, ok: !reason, reason, warn});
    });
    return {rows, valid: rows.filter(row => row.ok).map(row => list[row.index]), invalid: rows.filter(row => !row.ok).length};
  }
  // Batch requests: at most ``maxFiles`` files and ``maxBytes`` bytes per request (the service answers 413 above
  // 256 MB per batch); a single file larger than the byte budget still gets a request of its own.
  function uploadChunks(files, opts) {
    const o = isObject(opts) ? opts : {};
    const maxFiles = Math.max(1, numberOrNull(o.maxFiles) || 20), maxBytes = numberOrNull(o.maxBytes) || 240 * 1024 * 1024;
    const chunks = []; let current = [], bytes = 0;
    for (const file of Array.from(files || [])) {
      const size = Math.max(0, numberOrNull(file && file.size) || 0);
      if (current.length && (current.length >= maxFiles || bytes + size > maxBytes)) { chunks.push(current); current = []; bytes = 0; }
      current.push(file); bytes += size;
    }
    if (current.length) chunks.push(current);
    return chunks;
  }
  // Case / patient identifiers: the service refuses control characters and over-long text; a bare 2–4 character
  // Chinese name is only a warning (identifiers must be anonymous).  → null | {level:'error'|'warn', text}
  function identifierIssue(value, opts) {
    const o = isObject(opts) ? opts : {};
    const label = o.label || '编号', max = numberOrNull(o.max) || 160;
    const text = String(value === null || value === undefined ? '' : value);
    if (!text.trim()) return null;
    // eslint-disable-next-line no-control-regex
    if (/[\u0000-\u001f\u007f-\u009f]/.test(text)) return {level: 'error', text: `${label}含有换行、制表符等不可见字符（常见于从表格复制），请删掉后重新输入。`};
    if (text.trim().length > max) return {level: 'error', text: `${label}最多 ${max} 个字符（当前 ${text.trim().length} 个）。`};
    if (/^[一-龥·]{2,4}$/.test(text.trim())) return {level: 'warn', text: `「${text.trim()}」看起来像真实姓名，请改用匿名编号。`};
    return null;
  }
  // C7 (§5.4): a file name made only of two to four letter groups (LV_GUO_YOU, Zhang San) may be a pinyin name.
  // Only a reminder; the upload is never blocked.
  const NAME_LIKE = /^[A-Za-z]+(?:[_ -][A-Za-z]+){1,3}$/;
  const NAME_LIKE_TEXT = '文件名可能是姓名，建议填写患者编号';
  function nameLikeFilename(name) {
    const stem = String(name === null || name === undefined ? '' : name).replace(/\.stl$/i, '').trim();
    return NAME_LIKE.test(stem);
  }
  // C7: the name shown for a job — the service's display_name (patient id when filled, else the case name);
  // older services send none, and then the case name is used.
  function displayName(job, fallback) {
    const j = isObject(job) ? job : {};
    const name = typeof j.display_name === 'string' ? j.display_name.trim() : '';
    return name || (typeof j.case_id === 'string' && j.case_id.trim()) || (fallback === undefined ? '未命名病例' : fallback);
  }
  // C5 (§5.4): the stored quality.label stays as it is (golden regression); only the wording shown changes.
  const QUALITY_GOOD_TEXT = '多模型一致（一致不代表准确）';
  function qualityDisplayLabel(quality) {
    const q = isObject(quality) ? quality : {};
    if (q.level === 'good' || q.label === '模型集成稳定') return QUALITY_GOOD_TEXT;
    return q.label || q.level || '未评估';
  }

  const core = {FINAL, NOTIFY_STATUSES, FAMILY_TEXT, BUILTIN_PRESETS, unreadParse, unreadSerialize, unreadKey, unreadAdd, unreadRemove, unreadHas, unreadJobs, unreadCount, unreadPrune, titleWithUnread,
    shouldNotify, notificationText, mergePreferences, uploadPreferencesFrom, isPendingReview, isReviewed, splitByReview, familyOfRelease, caseCardModel, latestMaxDiameter, narrativeModel, releaseLabel,
    exportUrl, filenameFromDisposition, stateFamily, planBatchExport, presetState, parseViewStateJSON,
    parseTags, tagsText, metadataPayload, rerunSkipReason, computeTiming,
    DISPLAY_KEYS, CROSS_FAMILY_DISPLAY_KEYS, compareDisplaySubset, sharedDisplayRange, EchoGuard, EVENT_TEXT, eventText, limitErrorText, cachedStages, timingPrecomputed,
    binEdges, histogram, cohortFilter, cohortSort, populationValues, cohortReleases, numberOrNull,
    formatNumber, trimNumber, percentText, parseJobHash, jobHash, ACTION_STATUSES, ACTIVE_STATUSES, CONFIRM_STATUSES, WORK_GROUPS, workGroupOf, workGroups,
    nextToConfirm, nextToReview, stepId, QUICK_FILTERS, quickFilter, overviewModel, KIND_TEXT, KIND_COLOR, releaseKind, summaryKind,
    cycleSummary, modelCount, ensembleWord, qualityNote, referencePathLabel, alertModel, reliabilitySummary, humanOutletReason, outletReview,
    TIMELINE_METRICS, timelineModel, sparkGeometry, GEOMETRY_FINDINGS, findingChip, findingsForCard, findingUnits,
    formatDuration, stageRemaining, etaView, etaRowSummary,
    waitText, formatBytes, checkUploadFiles, uploadChunks, identifierIssue, UPLOAD_MAX_BYTES,
    nameLikeFilename, NAME_LIKE_TEXT, displayName, qualityDisplayLabel, QUALITY_GOOD_TEXT};
  root.WssWorkbenchCore = core;
  if (typeof module !== 'undefined' && module.exports) module.exports = core;
})(typeof globalThis !== 'undefined' ? globalThis : this);
