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
      title: (card.case_ids || []).join(' / ') || '匿名病例',
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
    const total = [num(source.compute_s), num(source.compute_seconds), num(fallbackTotal)].find(value => value !== null);
    const attempt = num(source.attempt_s);
    const resolved = total === undefined ? null : total;
    return {attempt, total: resolved, split: attempt !== null && resolved !== null && Math.abs(attempt - resolved) > 0.05};
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

  const core = {FINAL, NOTIFY_STATUSES, FAMILY_TEXT, BUILTIN_PRESETS, unreadParse, unreadSerialize, unreadKey, unreadAdd, unreadRemove, unreadHas, unreadJobs, unreadCount, unreadPrune, titleWithUnread,
    shouldNotify, notificationText, mergePreferences, uploadPreferencesFrom, isPendingReview, isReviewed, splitByReview, familyOfRelease, caseCardModel, latestMaxDiameter, narrativeModel, releaseLabel,
    exportUrl, filenameFromDisposition, stateFamily, planBatchExport, presetState, parseViewStateJSON,
    parseTags, tagsText, metadataPayload, rerunSkipReason, computeTiming,
    DISPLAY_KEYS, CROSS_FAMILY_DISPLAY_KEYS, compareDisplaySubset, EchoGuard,
    binEdges, histogram, cohortFilter, cohortSort, populationValues, cohortReleases, numberOrNull};
  root.WssWorkbenchCore = core;
  if (typeof module !== 'undefined' && module.exports) module.exports = core;
})(typeof globalThis !== 'undefined' ? globalThis : this);
