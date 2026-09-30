/* WSS workspace v2 — Lane C (S6c): workbench.
 * Task list (#/tasks: sort, exact filters, server search, paging, multi-select → trash / rerun / summary table / zip),
 * trash (#/trash), cohort overview (#/cohort: sortable key numbers + one histogram), service state (health, reconnect
 * banner, 「服务已更新」), password change, the administrator's 「看全部用户」, finished-job notices with unread marks,
 * and the account preferences the upload dialog remembers.  Same endpoints and documents as the classic workbench
 * (app.js / workbench_core.js); numbers come from the service as stored.  Excluded from offline packages. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.admin = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };
  var api = function () { return ns.api; };
  var store = function () { return ns.store; };
  function h() { return ns.ui.h.apply(null, arguments); }

  if (ns.icons && ns.icons.register) {
    var P = ns.icons.P;
    ns.icons.register('chevron-up', [P('M3.5 10 8 5.5 12.5 10')]);
    ns.icons.register('trash', [P('M2.75 4.25h10.5'), P('M6.25 4.25V2.75h3.5v1.5'), P('M4 4.25l.6 9a1 1 0 0 0 1 .95h4.8a1 1 0 0 0 1-.95l.6-9')]);
    ns.icons.register('list', [P('M5.5 4h8'), P('M5.5 8h8'), P('M5.5 12h8'), ns.icons.C(2.75, 4, 0.8, true), ns.icons.C(2.75, 8, 0.8, true), ns.icons.C(2.75, 12, 0.8, true)]);
    ns.icons.register('chart', [P('M2.25 13.75h11.5'), P('M4 13.75V9'), P('M7 13.75V4.5'), P('M10 13.75V7'), P('M13 13.75V10.5')]);
  }

  // Service limits (classic workbench: DELETE_LIMIT / EXPORT_LIMIT / BUNDLE_LIMIT).
  var LIMITS = {delete: 100, export: 200, bundle: 50};
  var KEYS = {viewAll: 'wssv2:viewall', unread: 'wssv2:unread', tasks: 'wssv2:tasks', prefs: 'wssv2:account-prefs'};
  var FINAL = ['done', 'failed', 'cancelled', 'interrupted'];
  var NOTIFY = FINAL.concat(['awaiting_confirmation', 'awaiting_input', 'awaiting_outlets']);
  var QUIET_ACTIONS = /^(restored|owner_claimed|analysis_rebuilt|review_|[a-z_]+_updated$)/;
  var HEALTH_EVERY_MS = 30000, DOWN_GRACE_MS = 2500, DOWN_POLL_MS = 5000;

  var A = {sh: null, started: false, viewAll: false, unread: null, lastStatus: {}, conn: 'up', downTimer: null, downPoll: null, downSince: 0,
    boot: null, notified: null, health: null, healthTimer: null, prefs: null, prefsServer: null, baseTitle: '', banners: {}, mounted: null, els: null,
    tasks: {q: '', status: '', patient: '', tag: '', quick: '', sort: {key: 'created', dir: 'desc'}, page: 1, size: 50, selected: {},
      server: {key: null, at: 0, jobs: [], total: 0, busy: false, error: null, stale: false}},
    trash: {items: null, busy: false, error: null},
    cohort: {data: null, busy: false, error: null, stale: false, family: '', release: '', review: '', sort: {key: 'created_at', dir: 'desc'}, metric: '', bin: null}};

  // ------------------------------------------------------------------ small helpers
  function session() { return (A.sh && A.sh.session && A.sh.session()) || (store() && store().state.session) || {}; }
  function isAdmin() { return session().role === 'admin'; }
  function me() { return session().username || null; }
  function viewAll() { return isAdmin() && A.viewAll; }
  function jobs() { return (A.sh && A.sh.jobs()) || (store() && store().state.jobs) || []; }
  function cards() { return (A.sh && A.sh.cards()) || {}; }
  function enc(v) { return encodeURIComponent(String(v)); }
  function query(params) {
    var parts = [];
    Object.keys(params || {}).forEach(function (k) { var v = params[k]; if (v === undefined || v === null || v === '') return; parts.push(enc(k) + '=' + enc(v)); });
    return parts.length ? '?' + parts.join('&') : '';
  }
  function go(id) { if (A.sh) A.sh.go(id); else root.location.hash = '#/job/' + enc(id); }
  function refreshJobs() { return A.sh ? Promise.resolve(A.sh.refreshJobs()) : Promise.resolve(); }
  function nowMs() { return Date.now(); }
  function timeMs(v) { var t = Date.parse(v || ''); return isNaN(t) ? null : t; }
  function isRunning(job) { return /^running/.test((job && job.status) || '') || (job && job.status) === 'cancelling'; }
  function locked(job) { return Boolean(job && job.review && job.review.status === 'reviewed'); }
  // In the all-users view another user's task can be read, not changed (the service's rule).
  function own(job) { return !viewAll() || Boolean(job && job.owner_name && job.owner_name === me()); }
  function deletable(job) { return Boolean(job) && !isRunning(job) && !locked(job) && own(job); }
  function bucket(job) { return ns.rail && ns.rail.bucket ? ns.rail.bucket(job) : (job && job.status) || 'other'; }
  function sameDay(iso, now) {
    var t = timeMs(iso); if (t === null) return false;
    var a = new Date(t), b = new Date(now === undefined ? nowMs() : now);
    return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
  }
  function statusLabel(job) {
    var b = bucket(job);
    if (b === 'reviewed') return {tone: 'ok', label: '已复核'};
    if (b === 'unreviewed') return {tone: 'idle', label: '待复核'};
    var s = ui().statusInfo(job && job.status);
    return {tone: s.tone, label: s.label};
  }
  function shortTime(iso) {
    var t = ui().time(iso);
    if (!t) return '';
    return t.slice(0, 4) === String(new Date(nowMs()).getFullYear()) ? t.slice(5) : t.slice(0, 10);
  }
  function download(url, opts) { return api().download(url, opts).then(function (r) { ui().downloadBlob(r.blob, r.filename); return r; }); }
  function errText(e) { return (e && e.message) || String(e || ''); }
  function isObj(v) { return v !== null && typeof v === 'object' && !Array.isArray(v); }

  // ------------------------------------------------------------------ account preferences (/api/preferences, classic C6)
  // Section-wise merge, as workbench_core.mergePreferences: the document is shared with the classic workbench.
  function mergePrefs(base, patch) {
    var out = isObj(base) ? Object.assign({}, base) : {};
    Object.keys(isObj(patch) ? patch : {}).forEach(function (k) {
      var v = patch[k];
      out[k] = isObj(v) && isObj(out[k]) ? Object.assign({}, out[k], v) : v;
    });
    if (!out.schema_version) out.schema_version = 'wss-deploy.preferences/v1';
    return out;
  }
  function loadPrefs() {
    return api().request('/api/preferences').then(function (r) { A.prefs = mergePrefs({}, (r && r.preferences) || {}); A.prefsServer = true; },
      function () { A.prefsServer = false; A.prefs = mergePrefs({}, store().read(KEYS.prefs) || {}); });
  }
  function prefs() { return A.prefs || {}; }
  function savePrefs(patch) {
    A.prefs = mergePrefs(A.prefs || {}, patch);
    store().write(KEYS.prefs, A.prefs);
    if (A.prefsServer !== true) return Promise.resolve(A.prefs);   // never overwrite a document that could not be read
    return api().request('/api/preferences', {method: 'PUT', body: A.prefs}).then(function (r) {
      if (r && r.preferences) A.prefs = mergePrefs({}, r.preferences);
      return A.prefs;
    }, function (e) { if (e && (e.status === 404 || e.status === 405)) A.prefsServer = false; return A.prefs; });
  }

  // ------------------------------------------------------------------ unread marks ("job_id:version", classic C5)
  function unreadLoad() {
    var raw = store() ? store().read(KEYS.unread) : null;
    A.unread = {};
    (Array.isArray(raw) ? raw : []).forEach(function (s) { if (typeof s === 'string' && s.lastIndexOf(':') > 0) A.unread[s] = true; });
  }
  function unreadId(item) { return item.slice(0, item.lastIndexOf(':')); }
  function unreadItems() { if (!A.unread) unreadLoad(); return Object.keys(A.unread); }
  function isUnread(id) { return unreadItems().some(function (s) { return unreadId(s) === String(id); }); }
  function unreadIds() { var seen = {}; unreadItems().forEach(function (s) { seen[unreadId(s)] = true; }); return Object.keys(seen); }
  function unreadAdd(id, version) {
    if (!id) return false;
    var key = id + ':' + (version === undefined || version === null ? 0 : version);
    unreadItems();
    if (A.unread[key]) return false;
    Object.keys(A.unread).forEach(function (s) { if (unreadId(s) === String(id)) delete A.unread[s]; });
    A.unread[key] = true; persistUnread(); return true;
  }
  function markRead(id) {
    var hit = false;
    unreadItems().forEach(function (s) { if (unreadId(s) === String(id)) { delete A.unread[s]; hit = true; } });
    if (hit) { persistUnread(); repaintMarks(); }
    return hit;
  }
  function persistUnread() { if (store()) store().write(KEYS.unread, Object.keys(A.unread || {})); updateTitle(); }
  function pruneUnread(list) {
    if (viewAll()) return;   // an all-users list is not this owner's set of jobs
    var keep = {}; (list || []).forEach(function (j) { if (j && j.id) keep[j.id] = true; });
    var dropped = false;
    unreadItems().forEach(function (s) { if (!keep[unreadId(s)]) { delete A.unread[s]; dropped = true; } });
    if (dropped) persistUnread();
  }
  function updateTitle() {
    if (!root.document) return;
    if (!A.baseTitle) A.baseTitle = String(root.document.title || 'WSS 工作区').replace(/^\(\d+\)\s*/, '');
    var n = unreadIds().length;
    root.document.title = n > 0 ? '(' + n + ') ' + A.baseTitle : A.baseTitle;
  }
  function repaintMarks() {
    var S = ns.shell && ns.shell.state && ns.shell.state();
    if (S && S.rail && S.rail.render) S.rail.render();
    if (A.mounted === 'tasks') updateTasks();
  }

  // ------------------------------------------------------------------ notices (owner event stream)
  // Once per (job, status): a finished task, a failure, or a task waiting for a person (workbench_core.shouldNotify).
  // The status a job had in the loaded list counts as already known, so a review or an edit of a finished task (the
  // stream repeats status 「done」) is not announced as a new result.
  function shouldNotify(ev, known) {
    if (!ev || !ev.job_id || ev.action === 'progress') return false;
    var status = ev.status;
    if (!Object.prototype.hasOwnProperty.call(A.lastStatus, ev.job_id)) {
      var j = (known || jobs()).filter(function (x) { return x && x.id === ev.job_id; })[0];
      if (j) A.lastStatus[ev.job_id] = j.status;
    }
    // Edits, reviews and a restore from the trash repeat the job's status; they are not news.
    if (NOTIFY.indexOf(status) < 0 || QUIET_ACTIONS.test(ev.action || '')) { A.lastStatus[ev.job_id] = status; return false; }
    var prev = A.lastStatus[ev.job_id];
    A.lastStatus[ev.job_id] = status;
    return prev !== status;
  }
  var NOTICE_TEXT = {done: '已完成', failed: '失败', cancelled: '已取消', interrupted: '已中断', awaiting_confirmation: '待确认出口', awaiting_outlets: '待确认出口', awaiting_input: '待核对输入'};
  function nameOf(id, fallback) {
    var j = jobs().filter(function (x) { return x.id === id; })[0];
    return j ? ui().displayName(j) : (fallback || id);
  }
  function currentJobId() { var S = ns.shell && ns.shell.state && ns.shell.state(); return S && S.cur ? S.cur.jobId : null; }
  function onEvent(ev) {
    if (!ev || !ev.job_id) return;
    if (ev.action !== 'progress') A.tasks.server.stale = true;
    if (ev.action !== 'progress' && (ev.status === 'done' || ev.status === 'deleted' || /^review/.test(ev.action || ''))) A.cohort.stale = true;
    if (ev.status === 'deleted') A.trash.items = null;
    if (!shouldNotify(ev)) return;
    if (ev.status === 'cancelled') return;           // the person who cancelled already knows
    var hidden = Boolean(root.document && root.document.hidden);
    var isCur = ev.job_id === currentJobId();
    if (!isCur || hidden) unreadAdd(ev.job_id, ev.version);
    var name = nameOf(ev.job_id, ev.case_id), label = NOTICE_TEXT[ev.status] || ev.status;
    if (hidden) desktopNotice(ev, name + ' · ' + label);
    if (isCur && !hidden) return;
    ui().toast(name + ' · ' + label, {kind: ev.status === 'failed' || ev.status === 'interrupted' ? 'error' : ev.status === 'done' ? 'ok' : 'info', ms: 10000,
      actions: [{label: ev.status === 'done' ? '打开' : '查看', run: function () { go(ev.job_id); }}]});
  }
  function notificationsOn() { var n = prefs().notifications; return !(n && n.enabled === false); }
  function desktopNotice(ev, title) {
    if (!notificationsOn() || typeof root.Notification !== 'function' || root.Notification.permission !== 'granted') return;
    try {
      var n = new root.Notification(title, {body: ev.status === 'done' ? '结果已就绪。' : ev.status === 'failed' ? '打开查看原因。' : '需要你确认。', tag: 'wss-job-' + ev.job_id});
      n.onclick = function () { try { root.focus(); } catch (_) {} go(ev.job_id); n.close(); };
    } catch (_) {}
  }
  function toggleNotifications() {
    var on = !notificationsOn();
    savePrefs({notifications: {enabled: on}});
    var N = typeof root.Notification === 'function' ? root.Notification : null;
    if (on && N && N.permission === 'default') { try { N.requestPermission(); } catch (_) {} }
    if (on && N && N.permission === 'denied') ui().toast('浏览器禁止了本站通知：请在地址栏的站点设置里允许。', {kind: 'info'});
    else ui().toast(on ? '页面在后台时，任务完成会有桌面提醒。' : '已关闭桌面提醒。', {kind: 'ok', ms: 3000});
  }

  // ------------------------------------------------------------------ service state: banners, reconnect, version
  function bannerHost() {
    var d = root.document, el = d.getElementById('wsc-banner');
    if (el) return el;
    el = h('div', {id: 'wsc-banner', 'class': 'wsc-banner', role: 'status', 'aria-live': 'polite', hidden: true});
    if (d.body) d.body.appendChild(el);
    return el;
  }
  function showBanner(kind, text, tone, actions, closable) { A.banners[kind] = {text: text, tone: tone || 'info', actions: actions || [], closable: Boolean(closable)}; paintBanners(); }
  function hideBanner(kind) { if (A.banners[kind]) { delete A.banners[kind]; paintBanners(); } }
  function paintBanners() {
    if (!root.document) return;
    var host = bannerHost();
    var kinds = Object.keys(A.banners);
    host.hidden = kinds.length === 0;
    ui().fill(host, kinds.map(function (k) {
      var b = A.banners[k];
      return h('div', {'class': 'wsc-banner-row tone-' + b.tone, dataset: {kind: k}},
        h('span', {'class': 'wsc-banner-dot', 'aria-hidden': 'true'}), h('span', {'class': 'wsc-banner-text', text: b.text}),
        b.actions.map(function (a) { return ui().button(a.label, a.run, {cls: 'btn-sm'}); }),
        b.closable ? ui().iconButton('close', '关闭', function () { hideBanner(k); }, {cls: 'wsc-banner-close'}) : null);
    }));
  }
  function onConnection(st) {
    if (st === 'up') {
      var wasDown = A.conn === 'down';
      A.conn = 'up';
      if (A.downTimer) { clearTimeout(A.downTimer); A.downTimer = null; }
      stopDownPoll();
      if (A.banners.conn) { hideBanner('conn'); ui().toast('连接已恢复。', {kind: 'ok', ms: 2500}); }
      if (wasDown) { A.tasks.server.stale = true; refreshJobs(); checkHealth(); }
      return;
    }
    if (A.conn === 'down') return;
    A.conn = 'down'; A.downSince = nowMs();
    if (A.downTimer) clearTimeout(A.downTimer);
    A.downTimer = setTimeout(function () {
      A.downTimer = null;
      if (A.conn !== 'down') return;
      showBanner('conn', '连接中断，正在重连…', 'warn');
      startDownPoll();
    }, DOWN_GRACE_MS);
    if (A.downTimer && A.downTimer.unref) A.downTimer.unref();
  }
  // While the stream is down the service is probed; once it answers and the stream has not come back by itself (a
  // browser gives up on a stream that answered with an error), the stream is reopened.
  function startDownPoll() {
    if (A.downPoll) return;
    A.downPoll = setInterval(function () {
      if (A.conn !== 'down') { stopDownPoll(); return; }
      healthFetch().then(function (hl) {
        if (!hl || A.conn !== 'down') return;
        checkVersion(hl.version, hl.ui_build_v2);
        if (nowMs() - A.downSince > DOWN_POLL_MS && A.sh && A.sh.reopenEvents) A.sh.reopenEvents();
      });
    }, DOWN_POLL_MS);
    if (A.downPoll && A.downPoll.unref) A.downPoll.unref();
  }
  function stopDownPoll() { if (A.downPoll) { clearInterval(A.downPoll); A.downPoll = null; } }
  // /api/health through fetch (not api().request): a logged-out answer is not a session expiry.
  function healthFetch() {
    if (typeof root.fetch !== 'function') return Promise.resolve(null);
    return Promise.resolve(root.fetch('/api/health', {credentials: 'same-origin', headers: {Accept: 'application/json'}})).then(function (r) {
      return r && r.ok ? r.json() : null;
    }).catch(function () { return null; });
  }
  function checkHealth() {
    return healthFetch().then(function (hl) {
      if (!hl) return null;
      A.health = hl;
      checkVersion(hl.version, hl.ui_build_v2);
      return hl;
    });
  }
  // The page remembers the service version and the digest of the workspace files it booted with; after an upgrade
  // (or a static-only redeploy) it says so once and offers a reload (the hash keeps the open case).
  function checkVersion(version, build) {
    if (!version || typeof version !== 'string') return;
    var hasBuild = typeof build === 'string' && build.length > 0;
    if (!A.boot) { A.boot = {version: version, build: hasBuild ? build : null}; return; }
    if (hasBuild && !A.boot.build && version === A.boot.version) { A.boot.build = build; return; }
    var changed = version !== A.boot.version || Boolean(hasBuild && A.boot.build && build !== A.boot.build);
    if (!changed) return;
    var key = version + '/' + (hasBuild ? build : '');
    if (A.notified === key) return;
    A.notified = key;
    showBanner('update', version !== A.boot.version ? '服务已更新到 v' + version + '。刷新页面后使用新版本。' : '页面文件已更新。刷新页面后使用新版本。', 'info',
      [{label: '刷新页面', run: function () { root.location.reload(); }}], true);
  }
  function healthRows(hl) {
    if (!hl) return [];
    var q = hl.queue || {}, rows = [];
    rows.push(['实时更新', A.conn === 'down' ? '中断，正在重连' : '已连接']);
    rows.push(['版本', hl.version ? 'v' + hl.version : '—']);
    if (hl.uptime_s !== undefined) rows.push(['已运行', ui().duration(hl.uptime_s) + (hl.started_at ? '（' + ui().time(hl.started_at) + ' 启动）' : '')]);
    if (hl.queue) rows.push(['队列', '计算中 ' + (q.running || 0) + ' · 排队 ' + (q.queued || 0) + ' · 待确认 ' + ((q.awaiting_input || 0) + (q.awaiting_confirmation || 0))]);
    if (hl.worker_alive !== undefined) rows.push(['计算线程', hl.worker_alive ? '正常' : '未运行']);
    if (hl.gpu) rows.push(['GPU', hl.gpu.available ? (hl.gpu.name || '可用') : '不可用（用 CPU 计算）']);
    if (hl.disk_free_gb !== undefined && hl.disk_free_gb !== null) rows.push(['磁盘剩余', ui().num(hl.disk_free_gb, 'GB')]);
    if (hl.default_release) { var c = cards()[hl.default_release]; rows.push(['默认模型', (c && c.display_name) || ui().resultName({model_release: {id: hl.default_release}}, cards())]); }
    if (hl.stale_reports !== undefined && hl.stale_reports !== null) rows.push(['待刷新报告', String(hl.stale_reports)]);
    return rows;
  }
  function healthDialog() {
    var body = h('div', {'class': 'wsc-health'}, h('p', {'class': 'note', text: '正在读取…'}));
    var paint = function (hl) {
      var ok = hl && hl.ok !== false;
      ui().fill(body,
        h('div', {'class': 'wsc-health-head'}, ui().dot(!hl ? 'error' : ok && A.conn !== 'down' ? 'ok' : 'warn', !hl ? '连不上服务' : ok ? (A.conn === 'down' ? '实时更新中断' : '服务正常') : '服务报告异常'),
          h('span', {'class': 'sec-fill'}), ui().iconButton('refresh', '刷新', function () { checkHealth().then(paint); })),
        hl ? ui().table([{key: 'k', label: ''}, {key: 'v', label: ''}], healthRows(hl).map(function (r) { return {k: r[0], v: r[1]}; }), {cls: 'tbl-kv tbl-nohead'}) : null);
    };
    ui().dialog.open({title: '服务状态', body: [body], actions: [ui().button('关闭', function () { ui().dialog.close('done'); })]});
    checkHealth().then(paint);
  }
  function passwordDialog() {
    var mk = function (label, auto, ph) { return h('input', {type: 'password', autocomplete: auto, 'aria-label': label, placeholder: ph || ''}); };
    var oldF = mk('当前口令', 'current-password'), newF = mk('新口令', 'new-password', '至少 8 个字符'), again = mk('再输一次新口令', 'new-password');
    var err = h('p', {'class': 'note note-error', role: 'alert', hidden: true});
    var fail = function (text, field) { err.hidden = false; err.textContent = text; if (field) { try { field.focus(); } catch (_) {} } };
    var submit = ui().button('保存', function () {
      err.hidden = true;
      if (newF.value.length < 8) { fail('新口令至少 8 个字符。', newF); return; }
      if (newF.value !== again.value) { fail('两次输入的新口令不一致。', again); return; }
      submit.disabled = true;
      api().request('/api/session/password', {method: 'POST', body: {old_password: oldF.value, new_password: newF.value}}).then(function () {
        ui().dialog.close('done'); ui().toast('口令已更新。其他浏览器里的登录已退出。', {kind: 'ok'});
      }, function (e) {
        submit.disabled = false;
        fail(e.status === 401 ? '当前口令不正确（区分大小写）。' : e.status === 429 ? '尝试过于频繁，请一分钟后再试。' : errText(e), e.status === 401 ? oldF : null);
      });
    }, {kind: 'primary'});
    ui().dialog.open({title: '修改口令', cls: 'dlg-narrow', body: [
      h('label', {'class': 'fld'}, h('span', {text: '当前口令'}), oldF), h('label', {'class': 'fld'}, h('span', {text: '新口令'}), newF),
      h('label', {'class': 'fld'}, h('span', {text: '再输一次'}), again), err],
      actions: [ui().button('取消', function () { ui().dialog.close('cancel'); }), submit], focus: oldF});
  }
  function toggleViewAll() {
    A.viewAll = !A.viewAll;
    store().write(KEYS.viewAll, A.viewAll);
    A.tasks.selected = {}; A.tasks.page = 1; A.tasks.server.key = null; A.cohort.data = null; A.cohort.bin = null;
    ui().toast(A.viewAll ? '显示全部用户的任务。别人的任务只能查看。' : '只显示我的任务。', {kind: 'ok', ms: 3000});
    return refreshJobs().then(function () { if (A.mounted === 'cohort') loadCohort(true); });
  }
  function menuItems() {
    var s = session(), items = [];
    items.push({label: '服务状态…', hint: A.conn === 'down' ? '重连中' : '', run: healthDialog});
    if (s.username && s.login === 'password') items.push({label: '修改口令…', run: passwordDialog});
    if (isAdmin()) items.push({label: '看全部用户', checked: Boolean(A.viewAll), run: toggleViewAll});
    items.push({label: '完成时桌面提醒', checked: notificationsOn(), run: toggleNotifications});
    return items;
  }

  // ------------------------------------------------------------------ lifecycle
  function start(shellApi) {
    A.sh = shellApi || A.sh;
    if (A.started) return;
    A.started = true;
    if (!A.unread) unreadLoad();
    A.baseTitle = String((root.document && root.document.title) || 'WSS 工作区').replace(/^\(\d+\)\s*/, '');
    updateTitle();
    loadPrefs();
    checkHealth();
    A.healthTimer = setInterval(function () { if (!(root.document && root.document.hidden)) checkHealth(); }, HEALTH_EVERY_MS);
    if (A.healthTimer && A.healthTimer.unref) A.healthTimer.unref();
    if (root.addEventListener) root.addEventListener('hashchange', function () {
      var id = openedJob(); if (id) markRead(id);
      var pg = /^#\/([a-z][a-z0-9-]{1,30})\/?$/.exec(root.location.hash || '');
      if (!pg || PAGES.indexOf(pg[1]) < 0) leftPage();
    });
    if (root.document && root.document.addEventListener) root.document.addEventListener('visibilitychange', function () {
      if (root.document.hidden) return;
      var id = openedJob(); if (id) markRead(id);
      checkHealth();
    });
    var first = openedJob(); if (first) markRead(first);
    if (ns.upload && ns.upload.installDrop) ns.upload.installDrop(uploadCtx);
  }
  function openedJob() { var m = /^#\/job\/([A-Za-z0-9_.-]{1,80})/.exec((root.location && root.location.hash) || ''); return m ? m[1] : null; }
  function uploadCtx() {
    var s = A.sh;
    return {releases: s ? s.releases() : [], cards: cards(), onCreated: function (id) { if (s) s.scheduleRefresh(); if (id) go(id); }, onOpen: function (id) { go(id); },
      onRefresh: function () { if (s) s.scheduleRefresh(); }};
  }
  // After an in-place re-login (ws_main): the stream, the list and the service facts are read again.
  function resumed() {
    A.tasks.server.key = null; A.trash.items = null; A.cohort.stale = true;
    if (A.sh && A.sh.reopenEvents) A.sh.reopenEvents();
    refreshJobs(); checkHealth(); loadPrefs();
  }
  // The shell's list request (refreshJobs): every job of this owner (or of every user), unpaged — the home, the rail
  // and the task list work over it; the task list uses the service's search and paging for its own queries.
  function loadJobs() {
    if (!A.started) { A.viewAll = (store() ? store().read(KEYS.viewAll) : null) === true; if (!A.unread) unreadLoad(); }
    var all = viewAll();
    return api().request('/api/jobs' + (all ? '?all=1' : ''), {timeout: 60000}).then(function (r) {
      var list = Array.isArray(r) ? r : ((r && r.jobs) || []);
      pruneUnread(list);
      return {jobs: list};
    });
  }

  // ------------------------------------------------------------------ home pages: shared head
  function head(active, count, right) {
    return ns.rail && ns.rail.homeHead ? ns.rail.homeHead(active, count, right) : h('div', {'class': 'home-head'}, count, h('span', {'class': 'sec-fill'}), right);
  }

  // ------------------------------------------------------------------ #/tasks — flat task list
  var STATUS_FILTERS = [
    {value: '', label: '全部状态'},
    {value: 'confirm', label: '待确认', bucket: 'confirm'},
    {value: 'active', label: '计算中', bucket: 'running'},
    {value: 'pending', label: '待复核', bucket: 'unreviewed', server: 'done'},
    {value: 'reviewed', label: '已复核', bucket: 'reviewed', server: 'done'},
    {value: 'failed', label: '失败', bucket: 'failed'},
    {value: 'cancelled', label: '已取消', bucket: 'cancelled', server: 'cancelled'}];
  var QUICK = {today: '今日', todo: '待处理', attn: '需要处理'};
  function statusSpec(v) { return STATUS_FILTERS.filter(function (s) { return s.value === v; })[0] || STATUS_FILTERS[0]; }
  function quickMatch(job, key, now) {
    var b = bucket(job);
    if (key === 'today') return sameDay(job.created_at, now);
    if (key === 'todo') return b === 'confirm' || b === 'unreviewed';
    if (key === 'attn') return b === 'confirm' || b === 'unreviewed' || b === 'failed' || b === 'running';
    return true;
  }
  function needsServer(t) { return Boolean(String(t.q || '').trim() || String(t.patient || '').trim() || String(t.tag || '').trim()); }
  function serverParams(t) {
    return {q: String(t.q || '').trim(), patient_id: String(t.patient || '').trim(), tag: String(t.tag || '').trim(), status: statusSpec(t.status).server || '', all: viewAll() ? 1 : null};
  }
  // Every page of a server query (the service pages by ≤ 100) → one list.
  function fetchAllPages(params) {
    var out = [], page = 1;
    function next() {
      return api().request('/api/jobs' + query(Object.assign({}, params, {page: page, page_size: 100})), {timeout: 60000}).then(function (r) {
        var list = (r && r.jobs) || [];
        out = out.concat(list);
        var total = Number(r && r.total) || out.length;
        if (out.length < total && list.length && page < 100) { page += 1; return next(); }
        return {jobs: out, total: total};
      });
    }
    return next();
  }
  function fetchServer() {
    var t = A.tasks, s = t.server, params = serverParams(t), key = JSON.stringify(params);
    if (s.busy && s.key === key) return;
    s.key = key; s.busy = true; s.error = null; s.stale = false;
    fetchAllPages(params).then(function (r) {
      if (s.key !== key) return;
      s.jobs = r.jobs; s.total = r.total; s.at = nowMs(); s.busy = false;
      if (A.mounted === 'tasks') updateTasks();
    }, function (e) {
      if (s.key !== key) return;
      s.busy = false; s.error = errText(e); s.jobs = []; s.at = nowMs();
      if (A.mounted === 'tasks') updateTasks();
    });
  }
  var SORTS = {
    name: function (j) { return ui().displayName(j).toLowerCase(); },
    patient: function (j) { return String(j.patient_id || '').toLowerCase(); },
    result: function (j) { return ui().resultName(j, cards(), {short: true}); },
    status: function (j) { return ['confirm', 'failed', 'running', 'unreviewed', 'reviewed', 'cancelled', 'other'].indexOf(bucket(j)); },
    created: function (j) { return timeMs(j.created_at) || 0; },
    owner: function (j) { return String(j.owner_name || ''); }
  };
  // Pure: jobs + task state → {all (filtered, sorted), rows (this page), total, page, pages, size}
  function taskModel(list, t, now) {
    var spec = statusSpec(t.status);
    var rows = (list || []).filter(function (j) {
      if (!j || !j.id) return false;
      if (spec.bucket && bucket(j) !== spec.bucket) return false;
      if (t.quick && !quickMatch(j, t.quick, now)) return false;
      return true;
    });
    var key = t.sort && SORTS[t.sort.key] ? t.sort.key : 'created', sign = t.sort && t.sort.dir === 'asc' ? 1 : -1, f = SORTS[key];
    rows = rows.map(function (j, i) { return {j: j, v: f(j), i: i}; }).sort(function (a, b) {
      var blankA = a.v === '' || a.v === null, blankB = b.v === '' || b.v === null;
      if (blankA && !blankB) return 1;               // blanks sink in both directions
      if (blankB && !blankA) return -1;
      if (a.v < b.v) return -sign;
      if (a.v > b.v) return sign;
      return a.i - b.i;
    }).map(function (x) { return x.j; });
    var size = Math.max(1, t.size || 50), pages = Math.max(1, Math.ceil(rows.length / size));
    var page = Math.min(Math.max(1, t.page || 1), pages);
    return {all: rows, rows: rows.slice((page - 1) * size, page * size), total: rows.length, page: page, pages: pages, size: size};
  }
  function saveTaskPrefs() { store().write(KEYS.tasks, {sort: A.tasks.sort, size: A.tasks.size}); }
  function loadTaskPrefs() {
    var p = store().read(KEYS.tasks);
    if (p && p.sort && SORTS[p.sort.key]) A.tasks.sort = {key: p.sort.key, dir: p.sort.dir === 'asc' ? 'asc' : 'desc'};
    if (p && [25, 50, 100].indexOf(p.size) >= 0) A.tasks.size = p.size;
  }
  // Set filters from elsewhere (the home tiles) and open the page.
  function openTasks(filters) {
    var t = A.tasks;
    t.q = ''; t.patient = ''; t.tag = ''; t.status = ''; t.quick = ''; t.page = 1;
    Object.keys(filters || {}).forEach(function (k) { if (Object.prototype.hasOwnProperty.call(t, k)) t[k] = filters[k]; });
    syncInputs();
    if (root.location.hash === '#/tasks') { if (A.mounted === 'tasks') updateTasks(); }
    else root.location.hash = '#/tasks';
  }
  function syncInputs() {
    var E = A.els, t = A.tasks;
    if (!E || E.kind !== 'tasks') return;
    E.search.value = t.q; E.patient.value = t.patient; E.tag.value = t.tag; E.status.value = t.status;
  }
  function mountTasks(el) {
    var t = A.tasks;
    if (!(A.els && A.els.kind === 'tasks' && A.els.wrap.parentNode === el)) {
      loadTaskPrefs();
      var E = A.els = {kind: 'tasks'};
      var debounce = null;
      var onFilter = function (delay) {
        if (debounce) clearTimeout(debounce);
        debounce = setTimeout(function () {
          debounce = null;
          if (t.q === E.search.value && t.patient === E.patient.value && t.tag === E.tag.value) return;
          t.q = E.search.value; t.patient = E.patient.value; t.tag = E.tag.value; t.page = 1;
          updateTasks();
        }, delay);
      };
      E.search = h('input', {type: 'search', 'class': 'home-search', placeholder: '搜索编号、备注、文件名', 'aria-label': '搜索任务', value: t.q});
      E.search.addEventListener('input', function () { onFilter(320); });
      E.patient = h('input', {type: 'text', 'class': 'wsc-input', placeholder: '患者编号', 'aria-label': '患者编号（精确）', list: 'wsc-patients', value: t.patient, autocomplete: 'off'});
      E.tag = h('input', {type: 'text', 'class': 'wsc-input wsc-input-sm', placeholder: '标签', 'aria-label': '标签（精确）', list: 'wsc-tags', value: t.tag, autocomplete: 'off'});
      [E.patient, E.tag].forEach(function (inp) { inp.addEventListener('input', function () { onFilter(420); }); inp.addEventListener('change', function () { onFilter(0); }); });
      E.status = ui().select(STATUS_FILTERS, t.status, function (v) { t.status = v; t.page = 1; updateTasks(); }, {'class': 'wsc-select', 'aria-label': '任务状态'});
      E.size = ui().select([{value: '25', label: '每页 25'}, {value: '50', label: '每页 50'}, {value: '100', label: '每页 100'}], String(t.size), function (v) { t.size = Number(v); t.page = 1; saveTaskPrefs(); updateTasks(); },
        {'class': 'wsc-select', 'aria-label': '每页条数'});
      E.patients = h('datalist', {id: 'wsc-patients'}); E.tags = h('datalist', {id: 'wsc-tags'});
      E.quick = h('span', {'class': 'wsc-chips'});
      E.count = h('span', {'class': 'sec-count'});
      E.head = head('tasks', E.count, h('label', {'class': 'home-search-wrap'}, ui().icon('search'), E.search));
      E.filters = h('div', {'class': 'wsc-filters'}, E.status, E.patient, E.tag, E.quick, h('span', {'class': 'sec-fill'}),
        ui().infoTip('搜索在服务端进行：编号、患者、扫描标签与日期、备注、文件名、标签都会匹配。患者编号和标签是精确筛选。'), E.size, E.patients, E.tags);
      E.table = h('div', {'class': 'wsc-tablewrap'});
      E.pager = h('div', {'class': 'wsc-pager'});
      E.selbar = h('div', {'class': 'wsc-selbar', hidden: true});
      E.wrap = h('div', {'class': 'home wsc-page wsc-tasks'}, E.head, E.filters, E.table, E.pager, E.selbar);
      el.replaceChildren(E.wrap);
    }
    A.mounted = 'tasks';
    syncInputs();
    updateTasks();
  }
  function updateTasks() {
    var E = A.els, t = A.tasks;
    if (!E || E.kind !== 'tasks') return;
    var server = needsServer(t);
    var list = jobs();
    if (server) {
      var key = JSON.stringify(serverParams(t));
      if (t.server.key !== key || (t.server.stale && !t.server.busy)) fetchServer();
      list = t.server.key === key ? t.server.jobs : [];
    }
    var m = taskModel(list, t, nowMs());
    t.page = m.page;
    // the selection follows the fresh records; vanished tasks drop out
    var byId = {}; list.forEach(function (j) { byId[j.id] = j; });
    Object.keys(t.selected).forEach(function (id) { if (byId[id]) t.selected[id] = byId[id]; else delete t.selected[id]; });
    E.count.textContent = String(m.total);
    var pids = {}, tags = {};
    jobs().forEach(function (j) { if (j.patient_id) pids[j.patient_id] = true; (j.tags || []).forEach(function (g) { if (g) tags[g] = true; }); });
    ui().fill(E.patients, Object.keys(pids).sort().slice(0, 200).map(function (p) { return h('option', {value: p}); }));
    ui().fill(E.tags, Object.keys(tags).sort().slice(0, 200).map(function (g) { return h('option', {value: g}); }));
    ui().fill(E.quick, t.quick ? h('span', {'class': 'wsc-chip'}, h('span', {text: QUICK[t.quick] || t.quick}),
      ui().iconButton('close', '清除筛选', function () { t.quick = ''; t.page = 1; updateTasks(); }, {cls: 'wsc-chip-x'})) : null);
    E.wrap.classList.toggle('is-loading', Boolean(server && t.server.busy));
    renderTaskTable(E.table, m, server);
    renderPager(E.pager, m);
    renderSelbar(E.selbar);
  }
  function sortButton(label, active, dir, onClick, title) {
    var b = h('button', {type: 'button', 'class': 'wsc-sort' + (active ? ' on' : ''), 'aria-label': '按' + label + '排序', title: title || null},
      h('span', {text: label}), active ? ui().icon(dir === 'asc' ? 'chevron-up' : 'chevron-down', {size: 12}) : null);
    b.addEventListener('click', onClick);
    return b;
  }
  function sortHead(label, key, cls) {
    var t = A.tasks, active = t.sort.key === key;
    return h('th', {scope: 'col', 'class': cls || null, 'aria-sort': active ? (t.sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'},
      sortButton(label, active, t.sort.dir, function () {
        t.sort = active ? {key: key, dir: t.sort.dir === 'asc' ? 'desc' : 'asc'} : {key: key, dir: key === 'created' ? 'desc' : 'asc'};
        t.page = 1; saveTaskPrefs(); updateTasks();
      }));
  }
  function renderTaskTable(host, m, server) {
    var t = A.tasks, all = viewAll();
    var pageIds = m.rows.map(function (j) { return j.id; });
    var master = h('input', {type: 'checkbox', 'aria-label': '选择本页全部', checked: pageIds.length > 0 && pageIds.every(function (id) { return t.selected[id]; })});
    var syncMaster = function () { master.checked = pageIds.length > 0 && pageIds.every(function (id) { return t.selected[id]; }); };
    master.addEventListener('change', function () {
      m.rows.forEach(function (j) { if (master.checked) t.selected[j.id] = j; else delete t.selected[j.id]; });
      updateTasks();
    });
    var headRow = h('tr', {}, h('th', {'class': 'wsc-cb', scope: 'col'}, master), sortHead('病例', 'name'), sortHead('患者 / 扫描', 'patient', 'wsc-hide-sm'), sortHead('结果', 'result'),
      sortHead('状态', 'status'), h('th', {scope: 'col', 'class': 'wsc-hide-sm', text: '标签'}), sortHead('创建', 'created', 'num'), all ? sortHead('属主', 'owner') : null);
    var body = h('tbody');
    m.rows.forEach(function (j) {
      var st = statusLabel(j), mine = own(j), shown = ui().displayName(j);
      var cb = h('input', {type: 'checkbox', 'aria-label': '选择 ' + shown, checked: Boolean(t.selected[j.id])});
      var tr = null;
      cb.addEventListener('change', function () { if (cb.checked) t.selected[j.id] = j; else delete t.selected[j.id]; tr.classList.toggle('sel', cb.checked); renderSelbar(A.els.selbar); syncMaster(); });
      cb.addEventListener('click', function (e) { if (e && e.stopPropagation) e.stopPropagation(); });
      var name = h('a', {'class': 'wsc-name', href: '#/job/' + enc(j.id), title: shown + (j.case_id && j.case_id !== shown ? ' · 病例 ' + j.case_id : '')},
        isUnread(j.id) ? h('span', {'class': 'wsc-unread', title: '有新状态', 'aria-label': '未读'}) : null, h('span', {text: shown}));
      name.addEventListener('click', function (e) { if (e && e.stopPropagation) e.stopPropagation(); });
      var tags = (j.tags || []).filter(Boolean);
      var who = [j.patient_id && j.patient_id !== shown ? j.patient_id : '', j.scan_label || ''].filter(Boolean).join(' · ');
      tr = h('tr', {'class': 'wsc-row' + (t.selected[j.id] ? ' sel' : '') + (mine ? '' : ' not-mine'), dataset: {jobId: j.id}},
        h('td', {'class': 'wsc-cb'}, cb),
        h('td', {'class': 'wsc-td-name'}, name),
        h('td', {'class': 'wsc-hide-sm wsc-muted', text: who}),
        h('td', {text: ui().resultName(j, cards(), {short: true}), title: ui().resultName(j, cards())}),
        h('td', {}, ui().dot(st.tone, st.label)),
        h('td', {'class': 'wsc-hide-sm wsc-tags', text: tags.map(function (g) { return '#' + g; }).join(' ')}),
        h('td', {'class': 'num wsc-muted', text: shortTime(j.created_at), title: ui().time(j.created_at)}),
        all ? h('td', {'class': 'wsc-muted', text: j.owner_name || '旧会话'}) : null);
      tr.addEventListener('click', function (e) { if (e && e.target && e.target.tagName === 'INPUT') return; go(j.id); });
      body.appendChild(tr);
    });
    var table = h('table', {'class': 'tbl wsc-table'}, h('thead', {}, headRow), body);
    var empty = null;
    if (!m.rows.length) {
      var s = t.server;
      empty = ui().empty(server && s.busy ? '正在查找…' : server && s.error ? '查找失败：' + s.error : jobs().length ? '没有符合条件的任务。清空搜索或换一个筛选。' : '还没有任务。点顶栏「上传 STL」，或把 STL 拖进页面。');
    }
    ui().fill(host, table, empty);
  }
  function renderPager(host, m) {
    var t = A.tasks;
    if (!m.total) { host.replaceChildren(); return; }
    var from = (m.page - 1) * m.size + 1, to = Math.min(m.total, m.page * m.size);
    var prev = ui().iconButton('chevron-left', '上一页', function () { t.page = m.page - 1; updateTasks(); }, {cls: 'wsc-pg'});
    var next = ui().iconButton('chevron-right', '下一页', function () { t.page = m.page + 1; updateTasks(); }, {cls: 'wsc-pg'});
    prev.disabled = m.page <= 1; next.disabled = m.page >= m.pages;
    ui().fill(host, h('span', {'class': 'wsc-muted wsc-range', text: from + '–' + to + ' / ' + m.total}), h('span', {'class': 'sec-fill'}),
      m.pages > 1 ? [prev, h('span', {'class': 'wsc-pg-text', text: m.page + ' / ' + m.pages}), next] : null);
  }
  function selectedJobs() { var t = A.tasks; return Object.keys(t.selected).map(function (id) { return t.selected[id]; }); }
  function renderSelbar(host) {
    var sel = selectedJobs();
    host.hidden = sel.length === 0;
    if (!sel.length) { host.replaceChildren(); return; }
    var done = sel.filter(function (j) { return j.status === 'done'; });
    var mineDone = done.filter(own), removable = sel.filter(deletable);
    var b = function (label, fn, n, why, opts) {
      var x = ui().button(label + (n && n !== sel.length ? '（' + n + '）' : ''), fn, Object.assign({cls: 'btn-sm'}, opts || {}));
      x.disabled = !n; if (!n && why) x.title = why;
      return x;
    };
    ui().fill(host,
      h('span', {'class': 'wsc-sel-count', text: '已选 ' + sel.length}),
      b('删除', function () { deleteJobs(removable); }, removable.length, '计算中、已复核锁定或别人的任务不能删除', {icon: 'trash'}),
      b('重跑…', function () { rerunDialog(mineDone); }, mineDone.length, '只对自己已完成的任务可用', {icon: 'refresh'}),
      h('span', {'class': 'wsc-sel-sep'}),
      b('汇总表 CSV', function () { exportSummary(done, 'csv'); }, done.length, '只对已完成的任务可用', {icon: 'download'}),
      b('Excel', function () { exportSummary(done, 'xlsx'); }, done.length, '只对已完成的任务可用'),
      b('打包 zip', function () { bundleJobs(done); }, done.length, '只对已完成的任务可用'),
      h('span', {'class': 'sec-fill'}),
      ui().button('取消选择', function () { A.tasks.selected = {}; updateTasks(); }, {kind: 'link', cls: 'btn-sm'}));
  }

  // ------------------------------------------------------------------ batch actions
  function namesLine(list) {
    var names = list.slice(0, 4).map(function (j) { return ui().displayName(j); }).join('、');
    return names + (list.length > 4 ? ' 等 ' + list.length + ' 个' : '');
  }
  function deleteJobs(list) {
    list = (list || []).filter(deletable);
    if (!list.length) return Promise.resolve();
    if (list.length > LIMITS.delete) { ui().toast('一次最多删除 ' + LIMITS.delete + ' 个任务，请分批。', {kind: 'error'}); return Promise.resolve(); }
    return ui().confirm('删除 ' + list.length + ' 个任务', [namesLine(list) + '。', '连同输入、中心线、报告和导出一起移入回收站，30 天内可以恢复。'], {confirmLabel: '移入回收站', danger: true}).then(function (yes) {
      if (!yes) return null;
      return api().request('/api/jobs/delete', {method: 'POST', body: {jobs: list.map(function (j) { return {id: j.id, version: j.version}; })}, timeout: 120000}).then(function (r) {
        var deleted = [], failures = [];
        ((r && r.results) || []).forEach(function (it) {
          if (it.deleted) deleted.push(it.id);
          else failures.push(nameOf(it.id) + '：' + ((it.error && it.error.message) || '未删除'));
        });
        deleted.forEach(function (id) { delete A.tasks.selected[id]; markRead(id); });
        A.tasks.server.stale = true; A.trash.items = null; A.cohort.stale = true;
        var undo = deleted.length ? [{label: '撤销', run: function () { undoDelete(deleted); }}] : [];
        if (failures.length) ui().toast('已移入回收站 ' + deleted.length + ' 个；未删除：' + failures.join('；'), {kind: 'error', actions: undo});
        else ui().toast('已移入回收站 ' + deleted.length + ' 个任务。', {kind: 'ok', actions: undo});
        return refreshJobs();
      }, function (e) { ui().toast('删除失败：' + errText(e), {kind: 'error'}); });
    });
  }
  function undoDelete(ids) {
    var restored = 0, failures = [], chain = Promise.resolve();
    ids.forEach(function (id) {
      chain = chain.then(function () {
        return api().request('/api/trash/' + enc(id) + '/restore', {method: 'POST', body: {}}).then(function () { restored += 1; }, function (e) { failures.push(errText(e)); });
      });
    });
    return chain.then(function () {
      A.tasks.server.stale = true; A.trash.items = null; A.cohort.stale = true;
      refreshJobs();
      if (A.mounted === 'trash') loadTrash(true);
      ui().toast(failures.length ? '已恢复 ' + restored + ' 个；未恢复：' + failures.join('；') : '已撤销，恢复 ' + restored + ' 个任务。', {kind: failures.length ? 'error' : 'ok'});
    });
  }
  // workbench_core.rerunSkipReason: a finished task with confirmed outlets and a reusable stage-A snapshot.
  function rerunSkipReason(job) {
    if (!isObj(job)) return '任务不存在';
    if (job.status !== 'done') return '任务未完成';
    if (!isObj(job.mapping) || !Object.keys(job.mapping).length) return '还没有确认出口';
    if (!isObj(job.a) && !isObj(job.stage_a)) return '缺少可复用的中心线结果';
    return null;
  }
  function releaseChoices() {
    var rel = (A.sh && A.sh.releases()) || [];
    return rel.map(function (r) {
      var id = r.id || r.release, c = cards()[id];
      return {value: id, label: ((c && c.display_name) || ui().resultName({model_release: r}, cards())) + (r.default ? '（默认）' : ''), isDefault: Boolean(r.default)};
    }).filter(function (o) { return o.value; });
  }
  function rerunDialog(list) {
    list = (list || []).filter(function (j) { return j.status === 'done' && own(j); });
    if (!list.length) return;
    var opts = releaseChoices();
    if (!opts.length) { ui().toast('服务上没有可用的模型。', {kind: 'error'}); return; }
    var pick = ui().select(opts, (opts.filter(function (o) { return o.isDefault; })[0] || opts[0]).value, null, {'aria-label': '用哪个模型重跑'});
    var log = h('ul', {'class': 'wsc-log'});
    var progress = h('p', {'class': 'note', 'aria-live': 'polite', text: list.length === 1 ? '沿用这份输入的中心线和出口确认，只重新预测；原结果保留。' : '逐个新建任务，沿用各自的中心线和出口确认；原结果保留。'});
    var cancel = ui().button('取消', function () { ui().dialog.close('cancel'); });
    var startBtn = ui().button('开始', function () {
      startBtn.disabled = true; pick.disabled = true;
      var releaseId = pick.value, created = [], skipped = 0, failed = 0, chain = Promise.resolve();
      list.forEach(function (job, i) {
        chain = chain.then(function () {
          var line = h('li', {text: ui().displayName(job) + '：读取…'}); log.appendChild(line);
          progress.textContent = '正在提交 ' + (i + 1) + ' / ' + list.length;
          return api().job(job.id).then(function (r) {
            var full = (r && r.job) || r;
            var why = rerunSkipReason(full);
            if (why) { skipped += 1; line.textContent = ui().displayName(job) + '：跳过（' + why + '）'; line.className = 'bad'; return null; }
            return api().rerun(job.id, {version: full.version, release_id: releaseId}).then(function (x) {
              var fresh = (x && x.job) || x; created.push(fresh && fresh.id);
              line.textContent = ui().displayName(job) + '：已建立新任务';
            });
          }).catch(function (e) { failed += 1; line.textContent = ui().displayName(job) + '：失败 · ' + errText(e); line.className = 'bad'; });
        });
      });
      chain.then(function () {
        progress.textContent = '完成：新建 ' + created.length + ' 个' + (skipped ? '，跳过 ' + skipped + ' 个' : '') + (failed ? '，失败 ' + failed + ' 个' : '') + '。';
        cancel.replaceChildren(h('span', {text: '关闭'}));
        A.tasks.server.stale = true; refreshJobs();
      });
    }, {kind: 'primary'});
    ui().dialog.open({title: list.length === 1 ? '换模型重跑' : '重跑 ' + list.length + ' 个任务', body: [
      progress, h('label', {'class': 'fld'}, h('span', {text: '模型'}), pick), log], actions: [cancel, startBtn]});
  }
  function exportSummary(list, format) {
    var ids = (list || []).filter(function (j) { return j.status === 'done'; }).map(function (j) { return j.id; });
    if (!ids.length) return Promise.resolve();
    if (ids.length > LIMITS.export) { ui().toast('一次最多导出 ' + LIMITS.export + ' 个任务的汇总表。', {kind: 'error'}); return Promise.resolve(); }
    var fmt = format === 'xlsx' ? 'xlsx' : 'csv';
    return download('/api/jobs/export?ids=' + ids.map(enc).join(',') + '&format=' + fmt, {filename: 'wss_summary.' + fmt}).then(function () {
      ui().toast('已导出 ' + ids.length + ' 个任务的汇总表。', {kind: 'ok', ms: 3000});
    }, function (e) { ui().toast('导出失败：' + errText(e), {kind: 'error'}); });
  }
  function bundleJobs(list) {
    list = (list || []).filter(function (j) { return j.status === 'done'; });
    if (!list.length) return Promise.resolve();
    if (list.length > LIMITS.bundle) { ui().toast('一次最多打包 ' + LIMITS.bundle + ' 个任务。', {kind: 'error'}); return Promise.resolve(); }
    ui().toast('正在打包 ' + list.length + ' 个任务…', {kind: 'info', ms: 0});
    var p = list.length === 1 ? download('/api/jobs/' + enc(list[0].id) + '/bundle.zip', {filename: (list[0].case_id || 'case') + '_' + list[0].id + '.zip'})
      : download('/api/jobs/bundle', {method: 'POST', body: {ids: list.map(function (j) { return j.id; })}, filename: 'wss_bundle.zip'});
    return p.then(function () { ui().toast('已打包 ' + list.length + ' 个任务的全部文件。', {kind: 'ok', ms: 3000}); },
      function (e) { ui().toast('打包失败：' + errText(e), {kind: 'error'}); });
  }

  // ------------------------------------------------------------------ #/trash
  function loadTrash(force) {
    var T = A.trash;
    if (T.busy || (T.items && !force)) return;
    T.busy = true; T.error = null;
    api().request('/api/trash').then(function (r) {
      T.items = (r && Array.isArray(r.items)) ? r.items : []; T.busy = false;
      if (A.mounted === 'trash') renderTrash();
    }, function (e) {
      T.busy = false; T.items = []; T.error = e && e.status === 404 ? '这个服务版本没有回收站。' : errText(e);
      if (A.mounted === 'trash') renderTrash();
    });
  }
  function mountTrash(el) {
    if (!(A.els && A.els.kind === 'trash' && A.els.wrap.parentNode === el)) {
      var E = A.els = {kind: 'trash'};
      E.count = h('span', {'class': 'sec-count'});
      E.note = h('span', {'class': 'wsc-head-note'});
      E.head = head('trash', E.count, E.note);
      E.body = h('div', {'class': 'wsc-tablewrap'});
      E.wrap = h('div', {'class': 'home wsc-page wsc-trash'}, E.head, E.body);
      el.replaceChildren(E.wrap);
      A.trash.items = null;
    }
    A.mounted = 'trash';
    ui().fill(A.els.note, ui().infoTip('删除的任务在这里保留 30 天，到期自动彻底清除。' + (viewAll() ? '这里只列你自己删除的任务。' : '')),
      ui().iconButton('refresh', '刷新', function () { loadTrash(true); }));
    if (!A.trash.items) loadTrash(true);
    renderTrash();
  }
  function renderTrash() {
    var E = A.els, T = A.trash;
    if (!E || E.kind !== 'trash') return;
    var items = T.items || [];
    E.count.textContent = T.items ? String(items.length) : '';
    if (!T.items) { ui().fill(E.body, ui().empty('正在读取…')); return; }
    if (T.error) { ui().fill(E.body, ui().empty(T.error)); return; }
    if (!items.length) { ui().fill(E.body, ui().empty('回收站是空的。')); return; }
    var rows = items.map(function (it) {
      var restore = ui().button('恢复', function () { restoreTrash(it, restore); }, {cls: 'btn-sm'});
      var purge = ui().button('彻底删除', function () { purgeTrash(it, purge); }, {kind: 'flat', cls: 'btn-sm wsc-danger'});
      return {name: it.patient_id || it.case_id || it.id, result: ui().resultName({model_release: {id: it.release_id}, family: it.family}, cards(), {short: true}),
        before: ui().statusInfo(it.status_before), at: it.deleted_at, left: it.days_left, restore: restore, purge: purge, it: it, _cls: 'wsc-row'};
    });
    ui().fill(E.body, ui().table([
      {key: 'name', label: '病例', render: function (r) { return h('span', {'class': 'wsc-name-plain', text: r.name, title: r.it.case_id && r.it.case_id !== r.name ? '病例 ' + r.it.case_id : r.name}); }},
      {key: 'result', label: '结果'},
      {key: 'before', label: '删除前', render: function (r) { return ui().dot(r.before.tone, r.before.label); }},
      {key: 'at', label: '删除于', num: true, render: function (r) { return h('span', {'class': 'wsc-muted', text: shortTime(r.at), title: ui().time(r.at)}); }},
      {key: 'left', label: '剩余', num: true, render: function (r) { var n = Number(r.left); return isFinite(n) && r.left !== null ? Math.max(0, Math.ceil(n)) + ' 天' : '—'; }},
      {key: 'act', label: '', cls: 'wsc-actions', blank: true, render: function (r) { return [r.restore, r.purge]; }}], rows, {cls: 'wsc-table'}));
  }
  function restoreTrash(it, btn) {
    btn.disabled = true;
    return api().request('/api/trash/' + enc(it.id) + '/restore', {method: 'POST', body: {}}).then(function (r) {
      var job = (r && r.job) || {};
      ui().toast('已恢复 ' + (it.patient_id || it.case_id || it.id) + '。', {kind: 'ok', actions: job.id ? [{label: '打开', run: function () { go(job.id); }}] : []});
      A.tasks.server.stale = true; A.cohort.stale = true;
      refreshJobs(); loadTrash(true);
    }, function (e) { btn.disabled = false; ui().toast('没有恢复：' + errText(e), {kind: 'error'}); });
  }
  function purgeTrash(it, btn) {
    return ui().confirm('彻底删除', ['彻底删除 ' + (it.patient_id || it.case_id || it.id) + ' 及其全部文件？删除后无法恢复。'], {confirmLabel: '彻底删除', danger: true}).then(function (yes) {
      if (!yes) return null;
      btn.disabled = true;
      return api().request('/api/trash/' + enc(it.id) + '/purge', {method: 'POST', body: {}}).then(function () {
        ui().toast('已彻底删除。', {kind: 'ok', ms: 3000}); loadTrash(true);
      }, function (e) { btn.disabled = false; ui().toast('没有删除：' + errText(e), {kind: 'error'}); });
    });
  }

  // ------------------------------------------------------------------ #/cohort — key numbers of finished tasks
  // Values come from GET /api/jobs/export?format=json&scope=done (the rows of the summary table, copied from
  // summary.json as stored); nothing is recomputed here.
  var COHORT_COLS = [
    {key: 'name', label: '病例', text: true},
    {key: 'result', label: '结果', text: true},
    {key: 'max_diameter_mm', label: '管腔最大直径', units: 'mm', tip: '中心线站位截面的最大直径，不含附壁血栓与管壁'},
    {key: 'wss_p99_pa', label: 'WSS p99', units: 'Pa'},
    {key: 'area_frac_high', label: '高 WSS 占比', pct: true},
    {key: 'population_percentile', label: '人群分位', tip: '在同口径人群参照中的百分位（只有带人群参照的模型有）'},
    {key: 'tawss_mean_pa', label: 'TAWSS 均值', units: 'Pa'},
    {key: 'tawss_low_frac', label: '低 TAWSS 占比', pct: true},
    {key: 'osi_high_frac', label: '高 OSI 占比', pct: true},
    {key: 'speed_p99_m_s', label: '速度 p99', units: 'm/s'},
    {key: 'review', label: '复核', text: true}];
  function num(v) { return v === null || v === undefined || v === '' || typeof v === 'boolean' || !isFinite(Number(v)) ? null : Number(v); }
  function cohortText(row, col) {
    if (col.key === 'name') return row.patient_id || row.case_id || row.job_id || '—';
    if (col.key === 'result') { var c = cards()[row.release_id]; return (c && (c.short_name || c.display_name)) || ui().resultName({model_release: {id: row.release_id}, family: row.family}, cards(), {short: true}); }
    if (col.key === 'review') return row.review_status === 'reviewed' ? '已复核' : row.review_status === 'reopened' ? '已重新打开' : '未复核';
    return null;
  }
  function cohortValue(row, col) { return col.text ? cohortText(row, col) : num(row[col.key]); }
  function cohortCell(row, col) {
    if (col.text) return cohortText(row, col);
    var v = num(row[col.key]);
    if (v === null) return '—';
    if (col.pct) return ui().pct(v);
    return ui().sig(v);                      // the unit is in the column head
  }
  // Pure: rows + filters → filtered, sorted rows (missing values sink, as workbench_core.cohortSort).
  function cohortModel(rows, c) {
    var list = (rows || []).filter(function (r) {
      if (!r) return false;
      if (c.family && r.family !== c.family) return false;
      if (c.release && r.release_id !== c.release) return false;
      if (c.review && (r.review_status || 'unreviewed') !== c.review) return false;
      if (c.bin) {
        var v = num(r[c.bin.metric]);
        if (v === null || v < c.bin.lo || v > c.bin.hi || (v === c.bin.hi && !c.bin.last)) return false;
      }
      return true;
    });
    var sort = c.sort || {key: 'created_at', dir: 'desc'};
    var col = COHORT_COLS.filter(function (x) { return x.key === sort.key; })[0];
    var val = col ? function (r) { return cohortValue(r, col); } : function (r) { return timeMs(r.created_at) || 0; };
    var sign = sort.dir === 'asc' ? 1 : -1;
    return list.map(function (r, i) { return {r: r, v: val(r), i: i}; }).sort(function (a, b) {
      var ba = a.v === null || a.v === undefined || a.v === '', bb = b.v === null || b.v === undefined || b.v === '';
      if (ba && bb) return a.i - b.i;
      if (ba) return 1;
      if (bb) return -1;
      if (typeof a.v === 'number' && typeof b.v === 'number') return (a.v - b.v) * sign || a.i - b.i;
      return String(a.v).localeCompare(String(b.v), 'zh') * sign || a.i - b.i;
    }).map(function (x) { return x.r; });
  }
  // workbench_core.binEdges / histogram: the same edges and counting rule as the classic cohort panel.
  function binEdges(values, count) {
    var nums = (values || []).map(num).filter(function (v) { return v !== null; });
    if (!nums.length) return [];
    var min = Math.min.apply(null, nums), max = Math.max.apply(null, nums), bins = Math.max(1, Math.floor(count) || 10);
    if (!(max > min)) { var pad = Math.abs(min) > 0 ? Math.abs(min) * 0.05 : 0.5; min -= pad; max += pad; }
    var step = (max - min) / bins, edges = [];
    for (var i = 0; i <= bins; i++) edges.push(min + step * i);
    edges[bins] = max;
    return edges;
  }
  function histogram(values, edges) {
    var counts = [];
    for (var i = 0; i < Math.max(0, (edges || []).length - 1); i++) counts.push(0);
    if (!counts.length) return counts;
    (values || []).forEach(function (x) {
      var v = num(x);
      if (v === null || v < edges[0] || v > edges[edges.length - 1]) return;
      var k = 0; while (k < counts.length - 1 && v >= edges[k + 1]) k++;
      counts[k]++;
    });
    return counts;
  }
  // The population reference belongs to one release: the filtered release, else the most common one (workbench_core.populationValues).
  function populationValues(population, rows, releaseId) {
    var src = isObj(population) ? population : {};
    var pick = function (id) {
      var e = src[id];
      if (!isObj(e) || !Array.isArray(e.values_pa)) return null;
      var vals = e.values_pa.map(num).filter(function (v) { return v !== null; });
      return vals.length ? {release_id: id, values: vals, case_count: num(e.case_count) || vals.length} : null;
    };
    if (releaseId) return pick(releaseId);
    var counts = {};
    (rows || []).forEach(function (r) { if (r && r.release_id) counts[r.release_id] = (counts[r.release_id] || 0) + 1; });
    var order = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; }).concat(Object.keys(src));
    for (var i = 0; i < order.length; i++) { var f = pick(order[i]); if (f) return f; }
    return null;
  }
  function loadCohort(force) {
    var C = A.cohort;
    if (C.busy || (C.data && !force && !C.stale)) return;
    C.busy = true; C.error = null; C.stale = false;
    api().request('/api/jobs/export' + query({format: 'json', scope: 'done', all: viewAll() ? 1 : null}), {timeout: 60000}).then(function (r) {
      C.busy = false;
      C.data = {rows: (r && Array.isArray(r.rows)) ? r.rows : [], population: (r && r.population) || {}};
      if (A.mounted === 'cohort') renderCohort();
    }, function (e) {
      C.busy = false; C.data = {rows: [], population: {}}; C.error = e && e.status === 404 ? '这个服务版本没有队列数据。' : errText(e);
      if (A.mounted === 'cohort') renderCohort();
    });
  }
  function mountCohort(el) {
    var C = A.cohort;
    if (!(A.els && A.els.kind === 'cohort' && A.els.wrap.parentNode === el)) {
      var E = A.els = {kind: 'cohort'};
      E.count = h('span', {'class': 'sec-count'});
      E.head = head('cohort', E.count, h('span', {'class': 'wsc-head-note'}, ui().infoTip('已完成任务的关键数字，来自汇总表（与导出的 CSV 同源）。筛选作用于下面的卡片、图和表；点分布图的柱子只看那一段，点散点打开那个结果。'),
        ui().iconButton('refresh', '刷新', function () { loadCohort(true); })));
      E.filters = h('div', {'class': 'wsc-filters'});
      E.kpis = h('div', {'class': 'wsc-kpis'});
      E.chart = h('div', {'class': 'wsc-charts'});
      E.body = h('div', {'class': 'wsc-tablewrap'});
      E.wrap = h('div', {'class': 'home wsc-page wsc-cohort'}, E.head, E.filters, E.kpis, E.chart, E.body);
      el.replaceChildren(E.wrap);
    }
    A.mounted = 'cohort';
    watchResize();
    if (!C.data || C.stale) loadCohort(true);
    renderCohort();
  }
  // The charts are drawn in pixels for the width they get; a window resize redraws them.
  var resizeTimer = null, resizeBound = false;
  function watchResize() {
    if (resizeBound || typeof root.addEventListener !== 'function') return;
    resizeBound = true;
    root.addEventListener('resize', function () {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () { if (A.mounted === 'cohort' && A.cohort.data) renderCohort(); }, 160);
    });
  }
  function metricCols(rows) { return COHORT_COLS.filter(function (c) { return !c.text && rows.some(function (r) { return num(r[c.key]) !== null; }); }); }
  function renderCohort() {
    var E = A.els, C = A.cohort;
    if (!E || E.kind !== 'cohort') return;
    if (!C.data) { E.count.textContent = ''; E.filters.replaceChildren(); E.kpis.replaceChildren(); E.chart.replaceChildren(); ui().fill(E.body, ui().empty('正在读取…')); return; }
    var all = C.data.rows;
    var releases = {}; all.forEach(function (r) { if (r.release_id) releases[r.release_id] = true; });
    var relOpts = [{value: '', label: '全部模型'}].concat(Object.keys(releases).sort().map(function (id) { var c = cards()[id]; return {value: id, label: (c && c.display_name) || ui().resultName({model_release: {id: id}}, cards())}; }));
    if (C.release && !releases[C.release]) C.release = '';
    var fam = ui().select([{value: '', label: '全部结果'}, {value: 'wall', label: '壁面'}, {value: 'volume', label: '体场'}], C.family, function (v) { C.family = v; C.bin = null; renderCohort(); }, {'class': 'wsc-select', 'aria-label': '结果类型'});
    var rel = ui().select(relOpts, C.release, function (v) { C.release = v; C.bin = null; renderCohort(); }, {'class': 'wsc-select', 'aria-label': '模型'});
    var rev = ui().select([{value: '', label: '全部复核状态'}, {value: 'unreviewed', label: '未复核'}, {value: 'reviewed', label: '已复核'}], C.review, function (v) { C.review = v; renderCohort(); }, {'class': 'wsc-select', 'aria-label': '复核状态'});
    var rowsNoBin = cohortModel(all, Object.assign({}, C, {bin: null}));
    var rows = cohortModel(all, C);
    E.count.textContent = rows.length === all.length ? String(all.length) : rows.length + ' / ' + all.length;
    var picked = rows.filter(function (r) { return r.job_id; }).map(function (r) { return {id: r.job_id, status: 'done'}; });
    // the metric the two charts show sits with the other filters (one row scopes everything below it)
    var metrics = metricCols(rowsNoBin);
    if (metrics.length && !metrics.some(function (c) { return c.key === C.metric; })) C.metric = (metrics.filter(function (c) { return c.key === 'wss_p99_pa'; })[0] || metrics[0]).key;
    var col = metrics.filter(function (c) { return c.key === C.metric; })[0] || null;
    var pickMetric = metrics.length ? h('label', {'class': 'wsc-field'}, h('span', {text: '指标'}),
      ui().select(metrics.map(function (c) { return {value: c.key, label: c.label}; }), C.metric, function (v) { C.metric = v; C.bin = null; renderCohort(); }, {'class': 'wsc-select', 'aria-label': '图上的量'})) : null;
    ui().fill(E.filters, fam, rel, rev, pickMetric ? h('span', {'class': 'wsc-sep', 'aria-hidden': 'true'}) : null, pickMetric,
      C.bin ? h('span', {'class': 'wsc-chip'}, h('span', {text: C.bin.label}), ui().iconButton('close', '清除', function () { C.bin = null; renderCohort(); }, {cls: 'wsc-chip-x'})) : null,
      h('span', {'class': 'sec-fill'}),
      h('span', {'class': 'wsc-exports'},
        ui().button('CSV', function () { exportSummary(picked, 'csv'); }, {cls: 'btn-sm', icon: 'download', disabled: !picked.length}),
        ui().button('Excel', function () { exportSummary(picked, 'xlsx'); }, {cls: 'btn-sm', disabled: !picked.length})));
    if (C.error) { E.kpis.replaceChildren(); E.chart.replaceChildren(); ui().fill(E.body, ui().empty(C.error)); return; }
    if (!all.length) { E.kpis.replaceChildren(); E.chart.replaceChildren(); ui().fill(E.body, ui().empty('还没有已完成的任务。')); return; }
    renderKpis(E.kpis, cohortStats(rows));
    if (col) renderCharts(E.chart, rowsNoBin, col); else E.chart.replaceChildren();
    var shownCols = metricCols(rows.length ? rows : rowsNoBin);
    var cols = COHORT_COLS.filter(function (c) { return c.text || shownCols.indexOf(c) >= 0; });
    var thead = h('tr', {}, cols.map(function (c) {
      var active = C.sort.key === c.key;
      return h('th', {scope: 'col', 'class': c.text ? null : 'num', 'aria-sort': active ? (C.sort.dir === 'asc' ? 'ascending' : 'descending') : 'none'},
        sortButton(c.label + (c.units ? ' ' + ui().unitText(c.units) : ''), active, C.sort.dir, function () {
          C.sort = active ? {key: c.key, dir: C.sort.dir === 'asc' ? 'desc' : 'asc'} : {key: c.key, dir: c.text ? 'asc' : 'desc'}; renderCohort();
        }, c.tip));
    }));
    var body = h('tbody');
    rows.slice(0, 500).forEach(function (r) {
      var tr = h('tr', {'class': 'wsc-row', tabindex: '0', dataset: {jobId: r.job_id || ''}, title: '打开这个结果'}, cols.map(function (c) {
        return h('td', {'class': c.text ? (c.key === 'name' ? 'wsc-td-name' : null) : 'num', text: cohortCell(r, c)});
      }));
      if (r.job_id) {
        tr.addEventListener('click', function () { go(r.job_id); });
        tr.addEventListener('keydown', function (e) { if (e.key === 'Enter') go(r.job_id); });
      }
      body.appendChild(tr);
    });
    ui().fill(E.body, h('table', {'class': 'tbl wsc-table wsc-cohort-table'}, h('thead', {}, thead), body),
      rows.length > 500 ? ui().note('表格显示前 500 行；导出包含全部筛选结果（每次最多 ' + LIMITS.export + ' 个）。') : null,
      !rows.length ? ui().empty('没有符合条件的任务。') : null);
  }

  // ------------------------------------------------------------------ cohort cards (the filtered rows)
  function median(values) {
    var s = values.slice().sort(function (a, b) { return a - b; }), n = s.length;
    return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : null;
  }
  // Pure: what the four cards show.  Results by kind (most first), review progress, results with an 「关注」 finding
  // (rows without the column are not counted either way), and the lumen diameters.
  function quantile(sorted, q) {
    if (!sorted.length) return null;
    var p = (sorted.length - 1) * q, i = Math.floor(p), f = p - i;
    return i + 1 < sorted.length ? sorted[i] + (sorted[i + 1] - sorted[i]) * f : sorted[i];
  }
  function cohortStats(rows) {
    var byKind = {}, kinds = [];
    rows.forEach(function (r) { var k = cohortText(r, {key: 'result'}); if (!(k in byKind)) { byKind[k] = 0; kinds.push(k); } byKind[k] += 1; });
    kinds.sort(function (a, b) { return byKind[b] - byKind[a]; });
    var known = rows.filter(function (r) { return num(r.findings_attention) !== null; });
    var diam = rows.map(function (r) { return {r: r, v: num(r.max_diameter_mm)}; }).filter(function (x) { return x.v !== null; });
    var dv = diam.map(function (x) { return x.v; }), ds = dv.slice().sort(function (a, b) { return a - b; });
    return {n: rows.length, kinds: kinds.map(function (k) { return {label: k, n: byKind[k]}; }),
      reviewed: rows.filter(function (r) { return r.review_status === 'reviewed'; }).length,
      attention: known.filter(function (r) { return num(r.findings_attention) > 0; }).length, findingsKnown: known.length,
      diameters: diam, dMedian: median(dv), dMin: dv.length ? ds[0] : null, dMax: dv.length ? ds[ds.length - 1] : null, dQ1: quantile(ds, 0.25), dQ3: quantile(ds, 0.75)};
  }
  function rowName(r) { return cohortText(r, {key: 'name'}) + ' · ' + cohortText(r, {key: 'result'}); }
  function statCard(label, value, unit, body, tone) {
    return h('div', {'class': 'ov-card ov-static'},
      h('div', {'class': 'ov-top'}, tone ? h('span', {'class': 'ov-dot tone-' + tone, 'aria-hidden': 'true'}) : null, h('span', {'class': 'ov-label', text: label})),
      h('div', {'class': 'ov-value'}, h('span', {text: value}), unit ? h('span', {'class': 'ov-unit', text: unit}) : null), body);
  }
  function meter(frac, cls, label) {
    var f = Math.max(0, Math.min(1, frac || 0));
    return h('div', {'class': 'ov-meter ' + cls, role: 'img', 'aria-label': label}, f > 0 ? h('span', {'class': 'ov-meter-fill', style: 'width:' + (f * 100).toFixed(1) + '%'}) : null);
  }
  function renderKpis(host, st) {
    var n = st.n;
    // 1 results by kind: one bar per kind, one colour (the bars compare counts; the names are the labels)
    var top = st.kinds.slice(0, 3), rest = st.kinds.slice(3).reduce(function (s, k) { return s + k.n; }, 0), peak = Math.max.apply(null, [1].concat(st.kinds.map(function (k) { return k.n; })));
    var kinds = h('div', {'class': 'ov-bars'}, top.map(function (k) {
      return h('div', {'class': 'ov-bar-row'}, h('span', {'class': 'ov-bar-label', text: k.label}),
        h('span', {'class': 'ov-bar-track'}, h('span', {'class': 'ov-bar-fill', style: 'width:' + (k.n / peak * 100).toFixed(1) + '%'})), h('span', {'class': 'ov-bar-n', text: String(k.n)}));
    }), rest ? h('div', {'class': 'ov-sub', text: '其他 ' + rest}) : null);
    var c1 = statCard('已完成结果', String(n), null, n ? kinds : h('div', {'class': 'ov-sub', text: '没有符合条件的结果'}));
    // 2 review progress
    var c2 = statCard('已复核', n ? Math.round(st.reviewed / n * 100) + '%' : '—', null, h('div', {'class': 'ov-foot'},
      meter(n ? st.reviewed / n : 0, 'm-review', '已复核 ' + st.reviewed + ' / ' + n), h('div', {'class': 'ov-sub', text: st.reviewed + ' / ' + n + ' 个结果'})));
    // 3 results with an 「关注」 finding
    var k = st.findingsKnown;
    var c3 = statCard('有「关注」发现', k ? String(st.attention) : '—', null, h('div', {'class': 'ov-foot'},
      k ? meter(st.attention / k, 'm-attn', st.attention + ' / ' + k) : null,
      h('div', {'class': 'ov-sub', text: k ? st.attention + ' / ' + k + ' 个结果' : '这些结果没有发现记录'})), k && st.attention ? 'warn' : null);
    // 4 lumen diameter: median, and every result as a dot on the min–max strip
    var c4body = h('div', {'class': 'ov-foot'});
    var c4 = statCard('管腔最大直径 · 中位', st.dMedian === null ? '—' : ui().sig(st.dMedian), st.dMedian === null ? null : 'mm', c4body);
    host.replaceChildren(c1, c2, c3, c4);
    if (st.dMedian !== null) {
      var span = st.dMax - st.dMin, pos = function (v) { return (span > 0 ? (v - st.dMin) / span * 100 : 50).toFixed(2) + '%'; };
      var tip = ui().chartTip ? ui().chartTip(c4) : null;
      var iqr = h('span', {'class': 'ov-strip-iqr', style: 'left:' + pos(st.dQ1) + ';width:' + (span > 0 ? (st.dQ3 - st.dQ1) / span * 100 : 0).toFixed(2) + '%'});
      var strip = h('div', {'class': 'ov-strip', role: 'img', 'aria-label': '管腔最大直径 ' + st.diameters.length + ' 例，' + ui().sig(st.dMin) + '–' + ui().sig(st.dMax) + ' mm，四分位 ' + ui().sig(st.dQ1) + '–' + ui().sig(st.dQ3) + ' mm'},
        h('span', {'class': 'ov-strip-line'}), iqr);
      if (tip) tip.bind(iqr, ui().sig(st.dQ1) + '–' + ui().sig(st.dQ3) + ' mm', '中间一半（四分位）· 中位 ' + ui().sig(st.dMedian) + ' mm');
      // one dot per result while they stay apart; past that the quartile band carries the spread
      if (st.diameters.length <= 24) st.diameters.forEach(function (d) {
        var dot = h('span', {'class': 'ov-strip-dot', style: 'left:' + pos(d.v)});
        if (tip) tip.bind(dot, ui().sig(d.v) + ' mm', rowName(d.r));
        strip.appendChild(dot);
      });
      strip.appendChild(h('span', {'class': 'ov-strip-med', style: 'left:' + pos(st.dMedian)}));
      c4body.appendChild(strip);
      c4body.appendChild(h('div', {'class': 'ov-axis'}, h('span', {text: ui().sig(st.dMin)}), h('span', {text: st.diameters.length + ' 例'}), h('span', {text: ui().sig(st.dMax) + ' mm'})));
    }
  }

  // ------------------------------------------------------------------ cohort charts: distribution + scatter
  var SVGNS = 'http://www.w3.org/2000/svg';
  function svg(tag, attrs, kids) {
    var el = root.document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { if (attrs[k] !== null && attrs[k] !== undefined) el.setAttribute(k, String(attrs[k])); });
    (kids || []).forEach(function (c) { if (c) el.appendChild(c); });
    return el;
  }
  function svgText(x, y, text, attrs) { var t = svg('text', Object.assign({x: x, y: y}, attrs || {})); t.textContent = text; return t; }
  function histModel(values, refValues) {
    var edges = binEdges(values.concat(refValues || []), 16);
    return {edges: edges, counts: histogram(values, edges), ref: refValues ? histogram(refValues, edges) : null};
  }
  // Pure: round axis ticks (1 / 2 / 5 × 10^k, the step nearest to the asked count as d3 picks it) covering [lo, hi].
  function niceTicks(lo, hi, count) {
    if (!(hi > lo)) { var p = Math.abs(lo) * 0.1 || 1; lo -= p; hi += p; }
    var raw = (hi - lo) / Math.max(1, count || 4), mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), f = raw / mag;
    var step = (f >= 7.07 ? 10 : f >= 3.16 ? 5 : f >= 1.41 ? 2 : 1) * mag;
    var a = Math.floor(lo / step + 1e-9) * step, b = Math.ceil(hi / step - 1e-9) * step, ticks = [];
    for (var i = 0; a + i * step <= b + step * 1e-6; i++) ticks.push(+(a + i * step).toPrecision(12));
    var decimals = (String(+step.toPrecision(6)).split('.')[1] || '').length;
    return {lo: a, hi: b, step: step, ticks: ticks, decimals: decimals, fmt: function (v) { return Number(v).toFixed(decimals); }};
  }
  function inBin(r, bin) {
    var v = num(r[bin.metric]);
    return v !== null && v >= bin.lo && (v < bin.hi || (bin.last && v <= bin.hi));
  }
  function chartCard(cls, title, sub, right) {
    return h('section', {'class': 'wsc-card ' + cls},
      h('div', {'class': 'wsc-card-head'}, h('div', {'class': 'wsc-card-titles'}, h('h3', {'class': 'wsc-card-title', text: title}), sub ? h('span', {'class': 'wsc-card-sub', text: sub}) : null),
        h('span', {'class': 'sec-fill'}), right || null));
  }
  function plotWidth(el, fallback) { var w = Number(el && el.clientWidth); return Math.max(280, Math.round(isFinite(w) && w > 0 ? w : fallback)); }
  function renderCharts(host, rows, col) {
    var C = A.cohort;
    var values = rows.map(function (r) { return num(r[col.key]); }).filter(function (v) { return v !== null; });
    var pop = col.key === 'wss_p99_pa' ? populationValues(C.data.population, rows, C.release) : null;
    var unit = col.units && !col.pct ? ui().unitText(col.units) : '';
    var legend = pop ? h('div', {'class': 'wsc-legend'},
      h('span', {'class': 'wsc-key'}, h('span', {'class': 'wsc-swatch'}), h('span', {text: '这些任务 ' + values.length + ' 例'})),
      h('span', {'class': 'wsc-key'}, h('span', {'class': 'wsc-linekey'}), h('span', {text: '人群参照 ' + pop.case_count + ' 例'})),
      ui().infoTip('人群参照是模型发布时的同口径病例。纵轴是各自的占比，两组放在同一把尺上比较分布形状。')) : h('div', {'class': 'wsc-legend'}, h('span', {'class': 'wsc-muted', text: values.length + ' 例'}));
    var dist = chartCard('wsc-dist', '分布', col.label + (unit ? ' · ' + unit : ''), legend);
    var ycol = col.key === 'max_diameter_mm' ? metricCols(rows).filter(function (c) { return c.key !== 'max_diameter_mm'; })[0] || null : col;
    var hasX = rows.some(function (r) { return num(r.max_diameter_mm) !== null; });
    var scat = ycol && hasX ? chartCard('wsc-scatter', '与管腔最大直径', ycol.label + '，每个点一个结果') : null;
    host.classList.toggle('single', !scat);
    if (scat) host.replaceChildren(dist, scat); else host.replaceChildren(dist);
    drawDistribution(dist, rows, col, values, pop);
    if (scat) drawScatter(scat, rows, ycol);
  }
  function drawDistribution(card, rows, col, values, pop) {
    var C = A.cohort;
    var hm = histModel(values, pop ? pop.values : null), edges = hm.edges, counts = hm.counts, ref = hm.ref;
    var plot = h('div', {'class': 'wsc-plot'});
    card.appendChild(plot);
    var tip = ui().chartTip ? ui().chartTip(card) : null;
    var W = plotWidth(plot, 560), H = 236, pad = {l: 40, r: 10, t: 10, b: 28}, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    var n = values.length, refN = pop ? pop.values.length : 0;
    var share = counts.map(function (c) { return n ? c / n : 0; }), rshare = ref ? ref.map(function (c) { return refN ? c / refN : 0; }) : null;
    var yt = niceTicks(0, Math.max.apply(null, [0.05].concat(share, rshare || [])), 4);
    var Y = function (v) { return pad.t + ph - v / yt.hi * ph; };
    var e0 = edges[0], eN = edges[edges.length - 1], X = function (v) { return pad.l + (eN > e0 ? (v - e0) / (eN - e0) : 0.5) * pw; };
    var unit = col.units && !col.pct ? ' ' + ui().unitText(col.units) : '';
    var fmt = function (v) { return col.pct ? ui().pct(v) : ui().sig(v); };
    var kids = [];
    yt.ticks.forEach(function (t) {
      kids.push(svg('line', {x1: pad.l, x2: W - pad.r, y1: Math.round(Y(t)) + 0.5, y2: Math.round(Y(t)) + 0.5, 'class': t === 0 ? 'wsc-axis' : 'wsc-grid'}));
      kids.push(svgText(pad.l - 8, Y(t) + 4, Math.round(t * 100) + '%', {'class': 'wsc-tick', 'text-anchor': 'end'}));
    });
    var xt = niceTicks(e0, eN, Math.max(3, Math.floor(pw / 90)));
    xt.ticks.forEach(function (t) {
      if (t < e0 - 1e-9 || t > eN + 1e-9) return;
      kids.push(svgText(X(t), H - 8, col.pct ? ui().pct(t) : xt.fmt(t), {'class': 'wsc-tick', 'text-anchor': 'middle'}));
    });
    var slot = pw / Math.max(1, counts.length), bw = Math.min(24, Math.max(3, slot - 2));
    if (rshare) {   // the reference first, as a 10 % wash under a 2 px step line; the bars sit on top
      var area = ['M' + pad.l + ',' + Y(0)];
      rshare.forEach(function (v, i) { var x0 = pad.l + i * slot, x1 = x0 + slot, y = Y(v); area.push('L' + x0 + ',' + y, 'L' + x1 + ',' + y); });
      area.push('L' + (pad.l + pw) + ',' + Y(0), 'Z');
      kids.push(svg('path', {d: area.join(''), 'class': 'wsc-ref-area'}));
    }
    counts.forEach(function (c, i) {
      var lo = edges[i], hi = edges[i + 1], x = pad.l + i * slot + (slot - bw) / 2;
      var hgt = c ? Math.max(2, share[i] / yt.hi * ph) : 0, y = pad.t + ph - hgt, r = Math.min(4, hgt, bw / 2);
      var active = Boolean(C.bin && C.bin.metric === col.key && Math.abs(C.bin.lo - lo) < 1e-12);
      var d = 'M' + x + ',' + (pad.t + ph) + 'V' + (y + r) + 'Q' + x + ',' + y + ' ' + (x + r) + ',' + y + 'H' + (x + bw - r) + 'Q' + (x + bw) + ',' + y + ' ' + (x + bw) + ',' + (y + r) + 'V' + (pad.t + ph) + 'Z';
      var range = fmt(lo) + '–' + fmt(hi) + unit;
      var valueText = c + ' 例 · ' + Math.round(share[i] * 100) + '%';
      var label = range + (rshare ? ' · 人群 ' + Math.round(rshare[i] * 100) + '%' : '');
      var hit = svg('rect', {x: pad.l + i * slot, y: pad.t, width: slot, height: ph, 'class': 'wsc-hit', tabindex: c ? '0' : null, role: c ? 'button' : null, 'aria-label': range + '：' + c + ' 例'});
      var choose = function () {
        if (!c) return;
        C.bin = active ? null : {metric: col.key, lo: lo, hi: hi, last: i === counts.length - 1, label: col.label + ' ' + range};
        renderCohort();
      };
      hit.addEventListener('click', choose);
      hit.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { if (e.preventDefault) e.preventDefault(); choose(); } });
      if (tip) tip.bind(hit, valueText, label);
      kids.push(svg('g', {'class': 'wsc-bin' + (active ? ' on' : '') + (C.bin && !active ? ' dim' : '')}, [c ? svg('path', {d: d, 'class': 'wsc-bar'}) : null, hit]));
    });
    if (rshare) {
      var pts = [];
      rshare.forEach(function (v, i) { var y = Y(v); pts.push((pad.l + i * slot) + ',' + y, (pad.l + (i + 1) * slot) + ',' + y); });
      kids.push(svg('polyline', {points: pts.join(' '), 'class': 'wsc-ref'}));
    }
    plot.appendChild(svg('svg', {width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, 'class': 'wsc-hist', role: 'img', 'aria-label': col.label + ' 分布，' + n + ' 例'}, kids));
  }
  function drawScatter(card, rows, ycol) {
    var C = A.cohort;
    var pts = rows.map(function (r) { return {r: r, x: num(r.max_diameter_mm), y: num(r[ycol.key])}; }).filter(function (p) { return p.x !== null && p.y !== null; });
    var plot = h('div', {'class': 'wsc-plot'});
    card.appendChild(plot);
    if (pts.length < 2) { plot.appendChild(ui().note('至少要有两个结果同时有「管腔最大直径」和「' + ycol.label + '」才画这张图。')); return; }
    var tip = ui().chartTip ? ui().chartTip(card) : null;
    var W = plotWidth(plot, 420), H = 236, pad = {l: 46, r: 14, t: 10, b: 28}, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    var ext = function (vals) {
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), m = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
      return [Math.max(0, lo - m), hi + m];
    };
    var xe = ext(pts.map(function (p) { return p.x; })), ye = ext(pts.map(function (p) { return p.y; }));
    var xt = niceTicks(xe[0], xe[1], Math.max(3, Math.floor(pw / 90))), yt = niceTicks(ye[0], ye[1], 4);
    var X = function (v) { return pad.l + (v - xt.lo) / (xt.hi - xt.lo) * pw; }, Y = function (v) { return pad.t + ph - (v - yt.lo) / (yt.hi - yt.lo) * ph; };
    var fy = function (v) { return ycol.pct ? ui().pct(v) : ui().sig(v); };
    var yunit = ycol.units && !ycol.pct ? ' ' + ui().unitText(ycol.units) : '';
    var kids = [];
    yt.ticks.forEach(function (t, i) {
      kids.push(svg('line', {x1: pad.l, x2: W - pad.r, y1: Math.round(Y(t)) + 0.5, y2: Math.round(Y(t)) + 0.5, 'class': i === 0 ? 'wsc-axis' : 'wsc-grid'}));
      kids.push(svgText(pad.l - 8, Y(t) + 4, ycol.pct ? ui().pct(t) : yt.fmt(t), {'class': 'wsc-tick', 'text-anchor': 'end'}));
    });
    xt.ticks.forEach(function (t, i) {
      var last = i === xt.ticks.length - 1;
      kids.push(svgText(X(t), H - 8, xt.fmt(t) + (last ? ' mm' : ''), {'class': 'wsc-tick', 'text-anchor': last ? 'end' : 'middle'}));
    });
    // points in the chosen distribution bin stay blue, the rest step back; the hit area is 24 px
    pts.sort(function (a, b) { return b.x - a.x; }).forEach(function (p) {
      var sel = C.bin ? inBin(p.r, C.bin) : null;
      var hit = svg('circle', {cx: X(p.x), cy: Y(p.y), r: 12, 'class': 'wsc-pt-hit', tabindex: p.r.job_id ? '0' : null, role: p.r.job_id ? 'link' : null,
        'aria-label': rowName(p.r) + '：' + fy(p.y) + yunit + '，' + ui().sig(p.x) + ' mm'});
      if (p.r.job_id) {
        hit.addEventListener('click', function () { go(p.r.job_id); });
        hit.addEventListener('keydown', function (e) { if (e.key === 'Enter') go(p.r.job_id); });
      }
      if (tip) tip.bind(hit, fy(p.y) + yunit + ' · ' + ui().sig(p.x) + ' mm', rowName(p.r));
      kids.push(svg('g', {'class': 'wsc-pt' + (sel === true ? ' on' : sel === false ? ' dim' : '')}, [svg('circle', {cx: X(p.x), cy: Y(p.y), r: 4.5, 'class': 'wsc-dot'}), hit]));
    });
    plot.appendChild(svg('svg', {width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, 'class': 'wsc-scat', role: 'img', 'aria-label': ycol.label + ' 与管腔最大直径，' + pts.length + ' 个结果'}, kids));
  }

  // ------------------------------------------------------------------ pages (#/tasks, #/trash, #/cohort)
  var PAGES = ['tasks', 'trash', 'cohort'];
  function renderPage(name, el) {
    if (!el) return false;
    if (name === 'tasks') { mountTasks(el); return true; }
    if (name === 'trash') { mountTrash(el); return true; }
    if (name === 'cohort') { mountCohort(el); return true; }
    A.mounted = null;
    return false;
  }
  function leftPage() { A.mounted = null; }

  // ------------------------------------------------------------------ extension registration (ws_shell.js 「extensions」)
  (ns.ext = ns.ext || []).push({
    id: 'admin',
    onStart: function (shellApi) { start(shellApi); },
    onEvent: function (ev, shellApi) { A.sh = shellApi || A.sh; onEvent(ev); },
    onConnection: function (st, shellApi) { A.sh = shellApi || A.sh; onConnection(st); },
    userMenu: function (shellApi) { A.sh = shellApi || A.sh; return menuItems(); },
    page: function (name, container, shellApi) { A.sh = shellApi || A.sh; return renderPage(name, container); },
    onResult: function (shellApi) { A.sh = shellApi || A.sh; leftPage(); var c = shellApi && shellApi.cur(); if (c && c.jobId) markRead(c.jobId); }
  });

  return {
    // used by the shell, the rail, the upload dialog and ws_main
    loadJobs: loadJobs, renderPage: renderPage, leftPage: leftPage, shellApi: function () { return A.sh; }, isUnread: isUnread, markRead: markRead, unreadIds: unreadIds, viewAll: viewAll,
    prefs: prefs, savePrefs: savePrefs, openTasks: openTasks, resumed: resumed, start: start, onEvent: onEvent, onConnection: onConnection, checkVersion: checkVersion, checkHealth: checkHealth,
    // pure helpers (tests)
    taskModel: taskModel, cohortModel: cohortModel, binEdges: binEdges, histogram: histogram, histModel: histModel, populationValues: populationValues, cohortStats: cohortStats, niceTicks: niceTicks,
    rerunSkipReason: rerunSkipReason, mergePrefs: mergePrefs, quickMatch: quickMatch, shouldNotify: shouldNotify, STATUS_FILTERS: STATUS_FILTERS, _state: A
  };
});
