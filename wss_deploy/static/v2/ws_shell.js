/* WSS workspace v2 — shell (contract §6.3 ws_shell.js): layout, hash routing, basic / full tiers (U22), keyboard
 * (§6.5), status line (O5), time-bar placeholder (E9), offline shell.  Panels live in their own modules; the
 * viewer kernel (ns.viewer / ns.data / ns.cursor / ns.colorbar / ns.orientation) is called through its §6.2 API. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.shell = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };
  var store = function () { return ns.store; };
  var api = function () { return ns.api; };
  var VIEWS = ['wall', 'volume', 'compare', 'input'];
  var CLASSIC_TOOLS = '六视角、出版级导图、分支展开图';

  // ------------------------------------------------------------------ extensions (second phase, parallel lanes)
  // Other modules add toolbar buttons, inspector tabs, menu items, sections of the 工具 tab, keys, pages (#/name) and
  // result hooks without editing this file:
  //   (ns.ext = ns.ext || []).push({id, toolbar(api) → [buttons], tabs(api) → [{id, label}], renderTab(tabId, body, api) → true,
  //     layers(api) / userMenu(api) → [menu items], tools(api) → [sections], key(event, api) → true, page(name, container, api) → true,
  //     onResult(api), onClose(api), onField(api)})
  // `api` (shellApi below) is the last argument of every hook.  A hook that handles the call returns true; errors are
  // caught and logged, so one extension cannot break the page.
  function exts() { return Array.isArray(ns.ext) ? ns.ext : []; }
  function extCall(hook) {
    var args = Array.prototype.slice.call(arguments, 1).concat([shellApi()]), out = [];
    exts().forEach(function (x) {
      if (!x || typeof x[hook] !== 'function') return;
      try { var r = x[hook].apply(x, args); if (Array.isArray(r)) out = out.concat(r.filter(Boolean)); else if (r && r !== true) out.push(r); }
      catch (e) { if (root.console && root.console.error) root.console.error('extension ' + (x.id || '?') + '.' + hook + ': ' + (e && e.stack || e)); }
    });
    return out;
  }
  function extHandled(hook) {
    var args = Array.prototype.slice.call(arguments, 1).concat([shellApi()]), done = false;
    exts().forEach(function (x) {
      if (done || !x || typeof x[hook] !== 'function') return;
      try { done = x[hook].apply(x, args) === true; }
      catch (e) { if (root.console && root.console.error) root.console.error('extension ' + (x.id || '?') + '.' + hook + ': ' + (e && e.stack || e)); }
    });
    return done;
  }
  // Layers menu: the extension items follow a separator; user menu: they sit after its own separator and before one.
  function extMenuItems(hook) { var items = extCall(hook); if (!items.length) return []; return hook === 'userMenu' ? items.concat([{separator: true}]) : [{separator: true}].concat(items); }
  var SHELL_API = null;
  function shellApi() {
    if (SHELL_API) return SHELL_API;
    SHELL_API = {
      state: function () { return S; }, cur: function () { return S && S.cur; }, viewer: function () { return S && S.viewerA; }, offline: function () { return Boolean(S && S.offline); },
      ui: ui, api: api, store: store, h: h,
      renderInspector: function () { renderInspector(); }, renderToolbar: function () { renderToolbar(); }, renderHome: function () { renderHome(); }, renderTop: function () { renderTop(); },
      setTab: function (id) { setTab(id); }, currentTab: function () { return currentTab(); }, showMode: function (m) { showMode(m); },
      applyField: function (id, win) { return applyField(id, win); }, updateColorbar: function () { updateColorbar('a'); }, setLayers: function () { setLayers(); },
      flyTo: function (xyz, extent) { flyTo(xyz, extent); }, selectFinding: function (it) { selectFinding(it); }, drawLabels: function () { drawLabels(); },
      pickOnce: function (msg, cb) { pickOnce(msg, cb); }, cancelPickOnce: function () { cancelPickOnce(); },
      conflictOr: function (e, what) { conflictOr(e, what); }, reloadCurrent: function () { reloadCurrent(); }, go: function (id, extra) { go(id, extra); },
      refreshJobs: function () { refreshJobs(); }, scheduleRefresh: function () { scheduleRefresh(); }, openExport: function () { openExport(); }, openSliceAt: function (p) { openSliceAt(p); },
      saveViewSoon: function () { saveViewSoon(); }, hideName: function () { return hideName(); }, fieldById: fieldById, fieldName: fieldName, visibleFields: visibleFields,
      isLocked: function () { return Boolean(S.cur && S.cur.manifest && isLocked(S.cur)); }, editable: function () { return Boolean(S.cur && S.cur.manifest && editable(S.cur)); },
      withCenterline: function (fn) { if (S.cur && S.cur.result) withCenterline(S.cur, fn); }, copyText: function (t, done) { copyText(t, done); },
      fileBase: function () { return S.cur && S.cur.manifest ? fileBase(S.cur) : 'case'; }, stageMessage: function (t, err, keep) { stageMessage(t, err, keep); },
      parseHash: parseHash, buildHash: buildHash, replaceRoute: function (patch) { replaceRoute(patch); }, session: function () { return S && S.session; },
      jobs: function () { return (S && S.jobs) || []; }, cards: function () { return (S && S.cards) || {}; }, releases: function () { return (S && S.releases) || []; },
      // lane A
      // lane B
      // lane C
      // lane D
      // lane E
      _: null
    };
    return SHELL_API;
  }

  // ------------------------------------------------------------------ routing (pure)
  function parseHash(hash) {
    var raw = String(hash || '').replace(/^#/, '');
    var q = raw.indexOf('?');
    var path = q >= 0 ? raw.slice(0, q) : raw;
    var query = q >= 0 ? raw.slice(q + 1) : '';
    var out = {jobId: null, v: null, f: null, cmp: null, q: null, bm: null};
    var m = /^\/job\/([A-Za-z0-9_.-]{1,80})\/?$/.exec(path);
    if (m) out.jobId = m[1];
    query.split('&').forEach(function (part) {
      if (!part) return;
      var i = part.indexOf('=');
      var k = decodeURIComponent(i >= 0 ? part.slice(0, i) : part), v = i >= 0 ? decodeURIComponent(part.slice(i + 1)) : '';
      if (k === 'v' && VIEWS.indexOf(v) >= 0) out.v = v;
      else if (k === 'f' && /^[A-Za-z0-9_.-]{1,40}$/.test(v)) out.f = v;
      else if (k === 'cmp' && /^[A-Za-z0-9_.-]{1,80}$/.test(v)) out.cmp = v;
      else if (k === 'q' && /^[a-z]{1,20}$/.test(v)) out.q = v;
      else if (k === 'bm' && /^[A-Za-z0-9_-]{1,40}$/.test(v)) out.bm = v;
    });
    return out;
  }
  function buildHash(r) {
    if (!r || !r.jobId) return '#/';
    var parts = [];
    ['v', 'f', 'cmp', 'q', 'bm'].forEach(function (k) { if (r[k]) parts.push(k + '=' + encodeURIComponent(r[k])); });
    return '#/job/' + encodeURIComponent(r.jobId) + (parts.length ? '?' + parts.join('&') : '');
  }
  function visibleFields(m) { return ((m && m.fields) || []).filter(function (f) { return f && f.id && f.kind !== 'vector'; }); }
  function defaultField(m) {
    var ids = visibleFields(m).map(function (f) { return f.id; });
    var pref = m && m.result && m.result.family === 'volume' ? ['speed', 'pressure', 'wall_pressure'] : ['tawss', 'wss'];
    for (var i = 0; i < pref.length; i++) if (ids.indexOf(pref[i]) >= 0) return pref[i];
    return ids[0] || null;
  }
  function fieldById(m, id) { return visibleFields(m).filter(function (f) { return f.id === id; })[0] || null; }
  function fieldName(m, id) { var f = fieldById(m, id) || ((m && m.fields) || []).filter(function (x) { return x && x.id === id; })[0]; return f ? (f.short_label || f.label || f.id) : (id || ''); }
  function windowOptions(f) {
    var logHint = Boolean(ns.colormap && ns.colormap.logHint ? ns.colormap.logHint(f) : (f && f.display && f.display.log_scale === true));
    var opts = [{value: 'adaptive', label: logHint ? '本例自适应（对数）' : '本例自适应'}];
    if (logHint) opts.push({value: 'adaptive-linear', label: '本例自适应（线性）'});
    ((f && f.windows) || []).forEach(function (w) {
      if (!w || !w.id || !Array.isArray(w.range)) return;
      opts.push({value: w.id, label: w.label + ' ' + ui().range(w.range[0], w.range[1], f.units) + (w.provisional ? '（暂定）' : '')});
    });
    return opts;
  }
  function windowLabel(f, spec) {
    if (!spec || spec === 'adaptive' || spec === 'adaptive-linear') {
      var d = f && f.display || {};
      var hi = Array.isArray(d.range) ? d.range[1] : d.p99;
      var lo = Array.isArray(d.range) ? d.range[0] : 0;
      var mode = spec === 'adaptive-linear' ? '（线性）' : ((ns.colormap && ns.colormap.logHint ? ns.colormap.logHint(f) : d.log_scale === true) ? '（对数）' : '');
      return '本例自适应' + mode + (hi !== undefined && hi !== null ? ' ' + ui().range(lo, hi, f && f.units) : '');
    }
    if (typeof spec === 'object' && Array.isArray(spec.range)) return '固定范围 ' + ui().range(spec.range[0], spec.range[1], f && f.units);
    var w = ((f && f.windows) || []).filter(function (x) { return x && x.id === spec; })[0];
    return w ? w.label + ' ' + ui().range(w.range[0], w.range[1], f.units) + (w.provisional ? '（暂定）' : '') : String(spec);
  }

  // ------------------------------------------------------------------ the shell instance
  var S = null;
  function h() { return ui().h.apply(null, arguments); }

  function newState(opts) {
    return {offline: Boolean(opts && opts.offline), offlineMeta: (opts && opts.offlineMeta) || null, session: (opts && opts.session) || null,
      jobs: [], cards: {}, releases: [], cur: null, els: {}, viewerA: null, viewerB: null, colorbarA: null, colorbarB: null, orientA: null, orientB: null,
      unlink: null, inputView: null, pollTimer: null, closeEvents: null, abort: null, cmpSeq: 0, listTimer: null, lensRef: null, saveTimer: null,
      layers: {centerline: false, points: false, trust: false, outline: false, streamlines: true, wall: true, interior: true}};
  }

  // ------------------------------------------------------------------ layout
  function buildLayout(appEl) {
    var E = S.els;
    E.app = appEl;
    appEl.className = 'ws-app' + (S.offline ? ' is-offline' : '');
    E.topCase = h('span', {'class': 'top-case'});
    E.topNote = h('span', {'class': 'top-note'});
    E.topRight = h('div', {'class': 'top-right'});
    var brand = h('span', {'class': 'brand'}, ns.icons ? ns.icons.icon('logo', {size: 18}) : null, h('span', {'class': 'mark', text: 'WSS'}));
    E.top = h('header', {'class': 'ws-top'}, h('div', {'class': 'top-left'}, brand, E.topCase, E.topNote), E.topRight);
    E.rail = h('aside', {'class': 'ws-rail', 'aria-label': '病例'});
    E.toolbar = h('div', {'class': 'ws-toolbar', role: 'toolbar', 'aria-label': '视图工具'});
    E.qbar = h('div', {'class': 'ws-qbar', hidden: true});
    E.vpA = viewportShell('a');
    E.vpB = viewportShell('b');
    E.grid = h('div', {'class': 'vp-grid'}, E.vpA.root, E.vpB.root);
    E.vpB.root.hidden = true;
    E.inputHost = h('div', {'class': 'ws-inputhost', hidden: true});
    E.stageMsg = h('div', {'class': 'stage-msg', hidden: true});
    E.stage = h('div', {'class': 'ws-stage'}, E.grid, E.inputHost, E.stageMsg, E.toolbar, E.qbar);
    E.timebar = h('div', {'class': 'ws-timebar', hidden: true});
    E.home = h('div', {'class': 'ws-home', hidden: true});
    E.main = h('main', {'class': 'ws-main'}, E.stage, E.timebar, E.home);
    E.inspHead = h('div', {'class': 'insp-head'});
    E.tabs = h('div', {'class': 'insp-tabs', role: 'tablist', 'aria-label': '检查器'});
    E.inspBody = h('div', {'class': 'insp-body'});
    E.insp = h('aside', {'class': 'ws-inspector', 'aria-label': '检查器'}, E.inspHead, E.tabs, E.inspBody);
    E.body = h('div', {'class': 'ws-body'}, S.offline ? null : E.rail, E.main, E.insp);
    E.login = h('section', {'class': 'ws-login', hidden: true});
    appEl.replaceChildren(E.top, E.body, E.login);
    applyPanels();
  }
  function viewportShell(side) {
    var canvas = h('div', {'class': 'vp-canvas'});
    var status = h('div', {'class': 'vp-status'});
    var orient = h('div', {'class': 'vp-orient'});
    var hud = h('div', {'class': 'vp-hud', hidden: true});
    var msg = h('div', {'class': 'vp-msg', hidden: true});
    var legend = h('div', {'class': 'vp-legend'});
    // Overlays on the canvas; the viewer's fit keeps the vessel clear of the status line (top), the colour bar
    // (right) and the orientation marker (bottom left) through its default insets.
    var mainBox = h('div', {'class': 'vp-main'}, canvas, status, legend, orient, hud, msg);
    return {root: h('div', {'class': 'vp vp-' + side}, mainBox), canvas: canvas, status: status, orient: orient, hud: hud, msg: msg, legend: legend};
  }
  function applyPanels() {
    var p = store().prefs();
    var narrow = root.matchMedia ? root.matchMedia('(max-width: 760px)').matches : false;
    var body = S.els.body;
    if (!body) return;
    body.classList.toggle('rail-off', S.offline || !p.rail || (narrow && !S.railOpenNarrow));
    body.classList.toggle('insp-off', !p.inspector);
    if (S.els.app) { S.els.app.classList.toggle('stage-dark', p.stage !== 'light'); S.els.app.classList.toggle('stage-light', p.stage === 'light'); }
    resizeViewers();
  }
  function resizeViewers() {
    [S.viewerA, S.viewerB].forEach(function (v) { if (v && v.resize) { try { v.resize(); } catch (_) {} } });
  }

  // ------------------------------------------------------------------ top bar
  function renderTop() {
    var E = S.els;
    var cur = S.cur;
    var name = cur && cur.manifest ? (hideName() ? '病例' : (cur.manifest.job && cur.manifest.job.display_name) || ui().displayName(cur.job)) : cur && cur.job ? ui().displayName(cur.job) : '';
    E.topCase.textContent = name;
    if (S.offline) {
      var meta = S.offlineMeta || {};
      var review = cur && cur.manifest && cur.manifest.job ? ui().reviewInfo(cur.manifest.job.review).label : '未知';
      E.topNote.textContent = '离线报告 · 导出于 ' + (meta.exported_at ? ui().time(meta.exported_at) : '未知时间') + ' · 导出时复核状态：' + review;
      E.topRight.replaceChildren();
      return;
    }
    E.topNote.textContent = '';
    var full = store().prefs().tier === 'full';
    var upload = ui().button('上传 STL', openUpload, {icon: 'upload', kind: 'primary', cls: 'top-upload'});
    var helpBtn = ui().iconButton('help', '帮助', null, {cls: 'top-icon'});
    helpBtn.setAttribute('aria-haspopup', 'menu');
    helpBtn.addEventListener('click', function () {
      ui().menu(helpBtn, [
        {label: '输入说明', href: '/static/v2/help_input.html', newTab: true},
        {label: '一页操作卡', href: '/static/v2/help_quickstart.html', newTab: true},
        {label: '报错对照表', href: '/static/v2/help_errors.html', newTab: true},
        {separator: true},
        {label: '快捷键', hint: '?', run: shortcutsDialog}
      ]);
    });
    var classicHref = cur && cur.jobId ? api().urls.classic(cur.jobId) : '/';
    var sess = S.session || {};
    var initial = String(sess.username || '本机').slice(0, 1).toUpperCase();
    var userBtn = h('button', {type: 'button', 'class': 'top-avatar', title: sess.username || '本机服务', 'aria-label': '账户与设置', text: initial});
    userBtn.setAttribute('aria-haspopup', 'menu');
    userBtn.addEventListener('click', function () {
      var stage = store().prefs().stage;
      ui().menu(userBtn, [
        {heading: sess.username ? sess.username + (sess.role === 'admin' ? ' · 管理员' : '') : '本机服务'},
        {label: '基本档', checked: !full, run: function () { setTier('basic'); }},
        {label: '完整档', checked: full, run: function () { setTier('full'); }},
        {separator: true},
        {label: '深色视口', checked: stage !== 'light', run: function () { setStage('dark'); }},
        {label: '浅色视口', checked: stage === 'light', run: function () { setStage('light'); }},
        {separator: true},
      ].concat(extMenuItems('userMenu'), [
        {label: '经典工作台', href: classicHref},
        sess.login && sess.login !== 'none' ? {separator: true} : null,
        sess.login && sess.login !== 'none' ? {label: '退出登录', run: logout} : null
      ]));
    });
    ui().fill(E.topRight, full ? h('span', {'class': 'top-tier', text: '完整档'}) : null, upload, helpBtn, userBtn);
  }
  function setStage(v) { store().setPrefs({stage: v === 'light' ? 'light' : 'dark'}); applyPanels(); [S.viewerA, S.viewerB].forEach(function (x) { if (x && x.renderNow) { try { x.renderNow(); } catch (_) {} } }); }
  function setTier(t) { store().setPrefs({tier: t}); renderTop(); if (S.rail) S.rail.setFull(t === 'full'); if (S.cur && S.cur.manifest) renderInspector(); }
  function hideName() { return Boolean(S.offline && S.offlineMeta && S.offlineMeta.hide_name); }

  // ------------------------------------------------------------------ list + rail + home
  function refreshJobs() {
    if (S.offline || !api()) return Promise.resolve();
    var all = S.session && S.session.role === 'admin' ? 1 : null;
    return api().jobs({all: all}).then(function (r) {
      S.jobs = Array.isArray(r) ? r : (r.jobs || []);
      store().state.jobs = S.jobs;
      if (S.rail) S.rail.setJobs(S.jobs);
      if (!S.cur) renderHome();
    }, function (e) { if (e.status !== 401) ui().toast('病例列表读取失败：' + e.message, {kind: 'error'}); });
  }
  function scheduleRefresh() {
    if (S.listTimer) return;
    S.listTimer = setTimeout(function () { S.listTimer = null; refreshJobs(); }, 800);
    if (S.listTimer && S.listTimer.unref) S.listTimer.unref();
  }
  function renderHome() {
    var E = S.els;
    showMode('home');
    if (ns.rail) ns.rail.todo(E.home, S.jobs, {cards: S.cards, onOpen: go, onUpload: openUpload, q: S.homeQuery || '', focusSearch: Boolean(S.homeFocus),
      fetchThumb: api() ? function (job) {
        if (job.status === 'done' && ns.data) return ns.data.loadResult(ns.data.createOnlineSource(job.id)).then(function (res) { return {kind: 'result', result: res}; });
        return api().geometry(job.id);
      } : null,
      onSearch: function (v) { S.homeQuery = v; S.homeFocus = true; renderHome(); S.homeFocus = false; }});
    else E.home.replaceChildren(ui().empty('离线报告没有病例列表。'));
    renderTop();
  }
  function showMode(mode) {
    var E = S.els;
    E.home.hidden = mode !== 'home';
    E.toolbar.hidden = mode === 'home';
    E.stage.hidden = mode === 'home';
    E.insp.hidden = mode === 'home';
    if (mode === 'home') { E.qbar.hidden = true; E.timebar.hidden = true; }
    E.grid.hidden = mode !== 'result';
    E.inputHost.hidden = mode !== 'input';
    S.els.body.classList.toggle('is-home', mode === 'home');
    S.mode = mode;
  }
  function go(jobId, extra) {
    var r = Object.assign({jobId: jobId}, extra || {});
    var target = buildHash(r);
    if (root.location.hash === target) route();
    else root.location.hash = target;
  }
  function replaceRoute(patch) {
    if (!S.cur) return;
    var r = parseHash(root.location.hash);
    if (r.jobId !== S.cur.jobId) r = {jobId: S.cur.jobId};
    Object.keys(patch).forEach(function (k) { r[k] = patch[k] || null; });
    var target = buildHash(r);
    if (root.location.hash === target) return;
    try { root.history.replaceState(null, '', target); } catch (_) { root.location.hash = target; }
  }
  function route() {
    var r = parseHash(root.location.hash);
    if (S.offline) return;
    var page = /^#\/([a-z][a-z0-9-]{1,30})\/?$/.exec(root.location.hash || '');   // #/trash, #/cohort … : a page of an extension
    if (!r.jobId && page && page[1] !== 'job') {
      closeCurrent(); showMode('home'); if (S.rail) S.rail.setCurrent(null);
      if (extHandled('page', page[1], S.els.home)) { renderTop(); return; }
    }
    if (!r.jobId) { closeCurrent(); renderHome(); if (S.rail) S.rail.setCurrent(null); return; }
    if (S.cur && S.cur.jobId === r.jobId && S.cur.result && r.v !== 'input') { applyRouteExtras(r); return; }
    openJob(r.jobId, r);
  }

  // ------------------------------------------------------------------ opening a result
  function closeCurrent() {
    if (S.cur) extCall('onClose');
    if (S.abort) { try { S.abort.abort(); } catch (_) {} S.abort = null; }
    stopPoll();
    if (S.inputView) { S.inputView.dispose(); S.inputView = null; }
    exitSplit(true); exitCompare(true);
    if (S.cur && S.cur.slice) { try { S.cur.slice.dispose(); } catch (_) {} S.cur.slice = null; S.els.vpA.hud.hidden = true; }
    if (S.cur && S.cur.measure) { try { S.cur.measure.dispose(); } catch (_) {} S.cur.measure = null; }
    if (S.cur && S.cur.region) { try { S.cur.region.dispose(); } catch (_) {} S.cur.region = null; }
    if (S.cur && S.cur.probe && ns.probe) { try { ns.probe.clear(S.viewerA); } catch (_) {} S.cur.probe = null; }
    cancelPickOnce();
    if (ns.annot && S.viewerA) { try { ns.annot.clearAll(S.viewerA); } catch (_) {} }
    if (S.cur && S.cur.cursor) { try { S.viewerA.setCursor(null); } catch (_) {} }
    S.cur = null; S.lensRef = null; S.question = null;
    ns.questions && ns.questions.bar(S.els.qbar, null, {});
  }
  function openJob(jobId, r) {
    var seq = store().nextSeq();
    closeCurrent();
    S.abort = typeof root.AbortController === 'function' ? new root.AbortController() : null;
    var signal = S.abort ? S.abort.signal : undefined;
    S.cur = {seq: seq, jobId: jobId, job: null, result: null, manifest: null};
    if (S.tab === 'reading' || S.tab === 'compare' || S.tab === 'slice' || S.tab === 'measure' || S.tab === 'region' || S.tab === 'annot') S.tab = null;
    S.lensRef = null;
    if (S.rail) S.rail.setCurrent(jobId);
    showMode('result');
    stageMessage('正在读取…');
    S.els.toolbar.replaceChildren(); S.els.inspHead.replaceChildren(); S.els.tabs.replaceChildren(); S.els.inspBody.replaceChildren();
    return api().job(jobId, {signal: signal}).then(function (res) {
      if (!store().isLatest(seq)) return null;
      var job = res.job || res;
      if (!job || job.id !== jobId) throw new api().ApiError('服务返回的任务与请求不一致。', 0);
      S.cur.job = job;
      renderTop();
      if (job.status !== 'done' || (r && r.v === 'input')) { showInput(job); return null; }
      return loadResult(jobId, seq, signal, r);
    }).catch(function (e) {
      if (!store().isLatest(seq)) return;
      if (e && e.status === 409 && S.cur && S.cur.job) { showInput(S.cur.job); return; }
      if (e && e.status === 401) return;
      stageMessage(e && e.status === 404 ? '这个结果不存在，或者不属于当前账号。' : '读取失败：' + (e && e.message || e), true);
    });
  }
  function loadResult(jobId, seq, signal, r) {
    if (!ns.data || !ns.viewer) { stageMessage('查看器没有加载。请刷新页面；仍不行时用经典报告。', true); return Promise.resolve(); }
    var source = ns.data.createOnlineSource(jobId, {fetch: api().dataFetch});
    return Promise.resolve(ns.data.loadResult(source, {signal: signal})).then(function (result) {
      if (!store().isLatest(seq)) return null;
      var m = result && result.manifest;
      if (!m || !m.job || m.job.id !== jobId) throw new api().ApiError('结果数据与所选病例不一致，已停止显示。', 0);
      var expected = S.cur.job && S.cur.job.run_identity;
      var got = result.runIdentity || (m.result && m.result.run_identity);
      if (expected && got && expected !== got) throw new api().ApiError('结果身份与任务记录不一致，已停止显示。', 0);
      return showResult(result, seq, r);
    }, function (e) {
      if (!store().isLatest(seq)) return null;
      var status = api().state.lastStatus[api().manifestUrl(jobId)];
      if ((e && e.status === 409) || status === 409) { showInput(S.cur.job); return null; }
      throw e;
    });
  }
  function ensureViewerA() {
    if (S.viewerA) return S.viewerA;
    if (!ns.viewer || (typeof ns.viewer.webglAvailable === 'function' && !ns.viewer.webglAvailable())) return null;
    try { S.viewerA = ns.viewer.create(S.els.vpA.canvas, {kind: 'main'}); } catch (e) { S.viewerA = null; return null; }
    wireViewer(S.viewerA, 'a');
    if (ns.colorbar) { try { S.colorbarA = ns.colorbar.create(S.els.vpA.legend); } catch (_) { S.colorbarA = null; } }
    if (ns.orientation) { try { S.orientA = ns.orientation.create(S.els.vpA.orient, S.viewerA); } catch (_) { S.orientA = null; } }
    return S.viewerA;
  }
  function wireViewer(v, side) {
    if (!v || typeof v.on !== 'function') return;
    v.on('pick', function (e) { if (!e) return; onPick(e, side); });
    v.on('camera', function () {
      var o = side === 'a' ? S.orientA : S.orientB;
      if (o && o.update) { try { o.update(); } catch (_) {} }
      if (side === 'a') saveViewSoon();
    });
    v.on('contextlost', function () { var vp = side === 'a' ? S.els.vpA : S.els.vpB; vp.msg.hidden = false; vp.msg.textContent = '三维显示中断，正在恢复。数字和发现不受影响。'; });
    v.on('contextrestored', function () { var vp = side === 'a' ? S.els.vpA : S.els.vpB; vp.msg.hidden = true; });
    v.on('change', function (e) { if (e && (e.what === 'field' || e.what === 'result' || e.what === 'branches')) updateColorbar(side); });
    v.on('error', function (e) { ui().toast('三维显示出错：' + (e && e.message || e), {kind: 'error'}); });
    v.on('marker', function (e) {   // a finding marker in 3-D: select that finding
      if (side !== 'a' || !e || !S.cur || !S.cur.manifest) return;
      var it = ns.overview.findingsModel(S.cur.manifest).items.filter(function (x) { return x.id === e.id; })[0];
      if (it) selectFinding(it);
    });
  }
  function showResult(result, seq, r) {
    var cur = S.cur;
    cur.result = result; cur.manifest = result.manifest;
    cur.runIdentity = result.runIdentity || (cur.manifest.result && cur.manifest.result.run_identity) || cur.jobId;
    cur.dataVersion = cur.manifest.data_version || null;
    cur.arraysVersion = cur.manifest.arrays_version || null;
    cur.findingsExpanded = false;
    showMode('result');
    renderTop();
    var saved = store().readView(cur.runIdentity);
    if (S.offline && S.offlineMeta && S.offlineMeta.view && S.offlineMeta.view.run_identity === cur.runIdentity && !saved) saved = S.offlineMeta.view;
    var fid = (r && r.f && fieldById(cur.manifest, r.f)) ? r.f : (saved && saved.field && fieldById(cur.manifest, saved.field)) ? saved.field : defaultField(cur.manifest);
    cur.field = fid;
    cur.window = saved && saved.field === fid && saved.window ? saved.window : 'adaptive';
    if (cur.window !== 'adaptive' && cur.window !== 'adaptive-linear' && typeof cur.window === 'string' && !((fieldById(cur.manifest, fid) || {}).windows || []).some(function (w) { return w.id === cur.window; })) cur.window = 'adaptive';
    renderToolbar(); renderInspector(); renderStatusLine(); renderTimebar();
    var v = ensureViewerA();
    if (!v) {
      stageMessage('这台设备的浏览器无法显示三维（WebGL 不可用）。右侧的数字、发现和导出仍可使用。', false, true);
      afterShown(r);
      return null;
    }
    stageMessage('正在准备三维…');
    return Promise.resolve(v.setResult(result)).then(function () {
      if (!store().isLatest(seq) || S.cur !== cur) return;
      stageMessage(null);
      applyField(cur.field, cur.window, {save: false});
      applyLighting();
      if (saved && saved.viewer) { try { v.applyState(Object.assign({}, saved.viewer, {field: cur.field}), {animate: false}); } catch (_) { safeFit(v); } }
      else safeFit(v);
      if (typeof v.getState === 'function') { try { var ls = v.getState().layers; if (ls) Object.keys(ls).forEach(function (k) { S.layers[k] = ls[k]; }); } catch (_) {} }
      afterShown(r);
    }, function (e) {
      if (!store().isLatest(seq)) return;
      stageMessage('三维显示失败：' + (e && e.message || e) + '。右侧的数字仍可使用。', true);
      afterShown(r);
    });
  }
  function afterShown(r) {
    var cur = S.cur;
    loadTimeline(cur);
    drawLabels();
    extCall('onResult');
    if (r) applyRouteExtras(r);
  }
  function safeFit(v) { try { v.fit(); } catch (_) {} }
  function applyRouteExtras(r) {
    var cur = S.cur;
    if (!cur || !cur.result) return;
    if (r.f && r.f !== cur.field && fieldById(cur.manifest, r.f)) applyField(r.f, 'adaptive');
    if (r.q && r.q !== (S.question && S.question.id) && ns.questions) askQuestion(r.q);
    if (r.cmp && (!cur.compare || cur.compare.jobId !== r.cmp) && ns.compare) enterCompare(r.cmp);
    if (!r.cmp && cur.compare) exitCompare();
    if (r.bm) {
      var bm = (ns.bookmarks ? ns.bookmarks.list(cur.runIdentity) : []).filter(function (b) { return b.id === r.bm; })[0];
      if (bm) restoreBookmark(bm); else ui().toast('链接里的书签不在这台电脑上。书签只保存在保存它的浏览器里。', {kind: 'info'});
    }
  }
  function loadTimeline(cur) {
    var pid = (cur.manifest && cur.manifest.job && cur.manifest.job.patient_id) || (cur.job && cur.job.patient_id);
    if (S.offline || !pid || !api()) return;
    api().timeline(pid, S.session && S.session.role === 'admin').then(function (t) {
      if (S.cur !== cur) return;
      cur.timeline = t;
      if (currentTab() === 'overview') renderInspector();
    }, function () {});
  }
  function stageMessage(text, isError, keepGrid) {
    var el = S.els.stageMsg;
    if (!text) { el.hidden = true; el.textContent = ''; el.className = 'stage-msg'; S.els.grid.classList.remove('dim'); return; }
    el.hidden = false; el.className = 'stage-msg' + (isError ? ' err' : '');
    ui().fill(el, h('p', {text: text}), isError && S.cur && S.cur.jobId && api() ? ui().link('在经典报告中打开', api().urls.report(S.cur.jobId), {newTab: true}) : null);
    S.els.grid.classList.toggle('dim', !keepGrid);
  }

  // ------------------------------------------------------------------ input page (non-finished jobs)
  function showInput(job) {
    showMode('input');
    ui().fill(S.els.toolbar, h('span', {'class': 'tb-title', text: ui().statusInfo(job.status).label}), h('span', {'class': 'sec-fill'}),
      api() ? ui().link('在经典工作台中处理', api().urls.classic(job.id), {newTab: true}) : null);
    S.els.inspHead.replaceChildren(); S.els.tabs.replaceChildren();
    S.els.qbar.hidden = true; S.els.timebar.hidden = true;
    stageMessage(null);
    if (!ns.input) { S.els.inspBody.replaceChildren(ui().empty('这个任务还没有结果。')); return; }
    if (S.inputView) S.inputView.dispose();
    S.inputView = ns.input.create({job: job, stage: S.els.inputHost, inspector: S.els.inspBody, cards: S.cards,
      onChanged: function (j) { if (!S.cur || S.cur.jobId !== j.id) return; S.cur.job = j; if (j.status === 'done') { stopPoll(); openJob(j.id, {}); } else if (S.inputView) S.inputView.update(j); scheduleRefresh(); },
      onUpload: openUpload, onOpen: go});
    startPoll(job);
    renderTop();
  }
  function startPoll(job) {
    stopPoll();
    if (!/^(queued|running|cancelling|awaiting)/.test(job.status || '')) return;
    var id = job.id;
    S.pollTimer = setInterval(function () {
      if (!S.cur || S.cur.jobId !== id || S.cur.result) { stopPoll(); return; }
      api().job(id).then(function (r) {
        var j = r.job || r;
        if (!S.cur || S.cur.jobId !== id || S.cur.result) return;
        var before = S.cur.job;
        S.cur.job = j;
        if (j.status === 'done') { stopPoll(); scheduleRefresh(); openJob(id, {}); return; }
        if (S.inputView && (!before || before.version !== j.version || before.status !== j.status || j.eta)) S.inputView.update(j);
      }, function () {});
    }, 3000);
    if (S.pollTimer && S.pollTimer.unref) S.pollTimer.unref();
  }
  function stopPoll() { if (S.pollTimer) { clearInterval(S.pollTimer); S.pollTimer = null; } }

  // ------------------------------------------------------------------ toolbar, fields, windows
  function renderToolbar() {
    var cur = S.cur;
    var E = S.els;
    if (!cur || !cur.manifest) return;
    var m = cur.manifest;
    var p = store().prefs();
    var kids = [];
    if (!S.offline && (E.body.classList.contains('rail-off'))) kids.push(ui().iconButton('panel-left', '展开病例栏', function () { toggleRail(true); }, {cls: 'tb-solo'}));
    var seg = h('div', {'class': 'seg seg-fields', role: 'tablist', 'aria-label': '字段'});
    visibleFields(m).forEach(function (f, i) {
      var b = h('button', {type: 'button', role: 'tab', 'class': 'seg-btn' + (f.id === cur.field ? ' on' : ''), 'aria-selected': String(f.id === cur.field),
        title: (f.label || f.id) + (i < 9 ? ' · 快捷键 ' + (i + 1) : ''), dataset: {field: f.id}, text: f.short_label || f.label || f.id});
      b.addEventListener('click', function () { applyField(f.id, 'adaptive'); });
      seg.appendChild(b);
    });
    kids.push(seg);
    var f = fieldById(m, cur.field);
    var opts = windowOptions(f);
    var sameScale = cur.compare && cur.compare.mode === 'same' && cur.compare.rangeText;
    if (sameScale) opts = [{value: '__same', label: '两侧同尺度 ' + cur.compare.rangeText}];
    else if (cur.window && typeof cur.window === 'object') opts.push({value: '__custom', label: windowLabel(f, cur.window)});
    var sel = ui().select(opts, sameScale ? '__same' : typeof cur.window === 'object' ? '__custom' : (cur.window || 'adaptive'), function (v) { if (v !== '__custom' && v !== '__same') applyField(cur.field, v); },
      {'class': 'win-select', 'aria-label': '色标窗', title: sameScale ? '比较时两侧共用这个范围；在「比较」页改看法' : '色标窗', disabled: Boolean(sameScale)});
    kids.push(h('label', {'class': 'tb-win'}, h('span', {'class': 'tb-label', text: '色标'}), sel));
    kids.push(h('span', {'class': 'sec-fill'}));
    if (ns.questions) {
      var qs = ns.questions.list(m, {offline: S.offline});
      if (qs.length) {
        var qb = ui().button('按问题看', null, {iconAfter: 'chevron-down', cls: 'btn-flat tb-questions'});
        qb.setAttribute('aria-haspopup', 'menu');
        qb.addEventListener('click', function () { ui().menu(qb, qs.map(function (q) { return {label: q.label, checked: S.question && S.question.id === q.id ? true : undefined, run: function () { askQuestion(q.id); }}; })); });
        kids.push(qb);
      }
    }
    var tools = [];
    var cursorBtn = ui().iconButton('probe', '沿血管游标（G）', toggleCursor, {pressed: Boolean(cur.cursor)});
    cursorBtn.disabled = !ns.cursor || !S.viewerA;
    tools.push(cursorBtn);
    if (ns.slice && cur.result && ns.slice.supported(cur.result)) {
      var sliceBtn = ui().iconButton('slice', cur.slice ? '关闭截面（S）' : '截面（S）：在血管里切一刀，看截面上的速度和压力', toggleSlice, {pressed: Boolean(cur.slice)});
      sliceBtn.disabled = !S.viewerA || Boolean(cur.compare || cur.split);
      tools.push(sliceBtn);
    }
    if (ns.measure && S.viewerA && toolSupported('measure')) {
      var measBtn = ui().iconButton('ruler', cur.measure ? '关闭测量（M）' : '测量（M）：距离、沿中心线弧长、管径', toggleMeasure, {pressed: Boolean(cur.measure)});
      measBtn.disabled = Boolean(cur.compare || cur.split);
      tools.push(measBtn);
    }
    if (ns.annot && S.viewerA && !S.offline) {
      var anBtn = ui().iconButton('pin', cur.annotTool ? '关闭标注' : '标注：在管壁上钉文字，存到服务', toggleAnnot, {pressed: Boolean(cur.annotTool)});
      anBtn.disabled = Boolean(cur.compare || cur.split);
      tools.push(anBtn);
    }
    if (ns.region && cur.result && ns.region.supported(cur.result)) {
      var regBtn = ui().iconButton('region', cur.region ? '关闭区域统计' : '区域统计：分支上一段或球形区域的均值、p99、最大', toggleRegion, {pressed: Boolean(cur.region)});
      regBtn.disabled = !S.viewerA || Boolean(cur.compare || cur.split);
      tools.push(regBtn);
    }
    extCall('toolbar').forEach(function (b) { tools.push(b); });
    tools.push(ui().iconButton('light', p.lighting === 'soft' ? '光照：柔和（L 切换为平涂，读色更准）' : '光照：平涂（L 切换为柔和光照）', toggleLighting, {pressed: p.lighting === 'soft'}));
    var layerBtn = ui().iconButton('layers', '图层、色表与背景', null);
    layerBtn.setAttribute('aria-haspopup', 'menu');   // the document click that closes menus lets this one open
    layerBtn.addEventListener('click', function () { layersMenu(layerBtn); });
    tools.push(layerBtn);
    tools.push(ui().iconButton('bookmark', '保存书签（B）', saveBookmark));
    if (!S.offline && ns.compare) tools.push(ui().iconButton('compare', cur.compare ? '退出比较' : '比较两次结果', function () { if (cur.compare) exitCompare(); else openCompare(); }, {pressed: Boolean(cur.compare)}));
    tools.push(ui().iconButton('download', '导出', openExport));
    kids.push(h('div', {'class': 'tb-group', role: 'group', 'aria-label': '工具'}, tools));
    if (!p.inspector) kids.push(ui().iconButton('panel-right', '展开检查器', function () { setInspector(true); }, {cls: 'tb-insp tb-solo'}));
    ui().fill(E.toolbar, kids);
  }
  function scaleSpec(windowSpec) {
    // 'adaptive-linear' is a workspace-only choice: the adaptive window with the log hint switched off.
    if (windowSpec === 'adaptive-linear') return {window: 'adaptive', log: false, bands: null, cmap: store().prefs().cmap};
    return {window: windowSpec || 'adaptive', log: null, bands: null, cmap: store().prefs().cmap};
  }
  function applyField(fieldId, windowSpec, opts) {
    var cur = S.cur;
    if (!cur || !cur.manifest || !fieldById(cur.manifest, fieldId)) return false;
    cur.field = fieldId; cur.window = windowSpec || 'adaptive';
    if (cur.compare) {
      var c = cur.compare;
      if (fieldById(c.manifest, fieldId)) { c.field = fieldId; c.window = cur.window; }
      var allowed = ns.compare.sameScaleAllowed(ns.compare.conditions(cur.manifest, c.manifest, cur.field, c.field)).ok;
      if (!allowed && c.mode === 'same') c.mode = 'each';
      applyCompareScale();
      if (currentTab() === 'compare') renderInspector();
    }
    else if (S.viewerA) { try { S.viewerA.setField(fieldId, scaleSpec(cur.window)); } catch (e) { ui().toast('切换字段失败：' + e.message, {kind: 'error'}); } }
    if (cur.slice) {
      var sq = cur.slice.state().quantity;
      var want = fieldId === 'pressure' || fieldId === 'wall_pressure' ? 'pressure' : (fieldId === 'speed' && sq === 'pressure' ? 'speed' : sq);
      if (want !== sq) cur.slice.set({quantity: want});
      else cur.slice.refresh();
    }
    if (cur.region) cur.region.refresh();
    extCall('onField');
    updateColorbar('a');
    renderToolbar(); renderStatusLine();
    if (currentTab() === 'reading' || currentTab() === 'overview') renderInspector();
    if (!opts || opts.save !== false) { replaceRoute({f: fieldId}); saveViewSoon(); }
    return true;
  }
  function updateColorbar(side) {
    var v = side === 'a' ? S.viewerA : S.viewerB, cb = side === 'a' ? S.colorbarA : S.colorbarB;
    if (!v || !cb || typeof v.colorbarInfo !== 'function') return;
    try {
      var info = side === 'a' && S.cur && S.cur.slice ? (S.cur.slice.colorbarInfo() || v.colorbarInfo()) : v.colorbarInfo();
      var m = side === 'a' ? S.cur.manifest : (S.cur.compare ? S.cur.compare.manifest : S.cur.manifest);
      var fid = side === 'a' ? S.cur.field : (S.cur.compare ? S.cur.compare.field : S.cur.split && S.cur.split.field);
      var win = side === 'a' ? S.cur.window : (S.cur.compare ? S.cur.compare.window : S.cur.split && S.cur.split.window);
      if (info && !info.windowLabel) info.windowLabel = windowLabel(fieldById(m, fid), win);
      cb.update(info);
    } catch (_) {}
  }
  function toggleLighting() {
    var next = store().prefs().lighting === 'soft' ? 'flat' : 'soft';
    store().setPrefs({lighting: next});
    applyLighting(); renderToolbar();
    ui().toast(next === 'soft' ? '柔和光照：用来看形状；读数以探针和数字为准。' : '平涂：颜色只由数值决定。', {kind: 'info', ms: 3000});
  }
  function applyLighting() {
    var l = store().prefs().lighting;
    [S.viewerA, S.viewerB].forEach(function (v) { if (v && v.setLighting) { try { v.setLighting(l); } catch (_) {} } });
  }
  function layersMenu(anchor) {
    var m = S.cur && S.cur.manifest;
    var volume = m && m.result && m.result.family === 'volume';
    var L = S.layers;
    var item = function (key, label) { return {label: label, checked: Boolean(L[key]), run: function () { L[key] = !L[key]; setLayers(); }}; };
    var cm = store().prefs().cmap;
    ui().menu(anchor, [
      {heading: '图层'},
      item('outline', '轮廓线'), item('centerline', '中心线'), item('points', '预测点'), item('trust', '可信度标记（斜纹）'),
      volume ? item('streamlines', '流线') : null, volume ? item('wall', '血管外壁') : null, volume ? item('interior', '体内点') : null,
      {label: '分支显隐…', run: function () { branchDialog(); }}
    ].concat(extMenuItems('layers'), [   // lane E: extension layers sit in the 图层 group (the menu runs past short screens)
      {separator: true}, {heading: '标注与自动标签'},
      {label: '标注', checked: Boolean(store().prefs().labels.annotations), run: function () { setLabelPref({annotations: !store().prefs().labels.annotations}); }},
      {label: '分支名', checked: Boolean(store().prefs().labels.branches), run: function () { setLabelPref({branches: !store().prefs().labels.branches}); }},
      {label: '发现标签（前 5 条）', checked: store().prefs().labels.findings > 0, run: function () { setLabelPref({findings: store().prefs().labels.findings > 0 ? 0 : 5}); }},
      {label: '管腔最大直径环', checked: Boolean(store().prefs().labels.maxd), run: function () { setLabelPref({maxd: !store().prefs().labels.maxd}); }},
      {separator: true}, {heading: '视口背景'},
      {label: '深色', checked: store().prefs().stage !== 'light', run: function () { setStage('dark'); }},
      {label: '浅色', checked: store().prefs().stage === 'light', run: function () { setStage('light'); }},
      {separator: true}, {heading: '色表'},
      {label: '彩虹（默认）', checked: cm === 'rainbow', run: function () { setCmap('rainbow'); }},
      {label: 'viridis', checked: cm === 'viridis', run: function () { setCmap('viridis'); }},
      {label: 'turbo', checked: cm === 'turbo', run: function () { setCmap('turbo'); }}
    ]));
  }
  // Branch visibility (classic 分支显隐 / 血管模块): display only, every statistic keeps all branches.
  function branchDialog() {
    var cur = S.cur, v = S.viewerA;
    if (!cur || !cur.manifest || !v) return;
    var list = (cur.manifest.geometry && cur.manifest.geometry.branches) || [];
    if (list.length < 2) { ui().toast('这份结果只有一条分支。', {kind: 'info'}); return; }
    var visible = null;
    try { visible = v.getState().branches; } catch (_) { visible = null; }
    var on = function (id) { return !visible || visible.indexOf(Number(id)) >= 0; };
    var apply = function () {
      var ids = boxes.filter(function (b) { return b.input.checked; }).map(function (b) { return b.id; });
      try { v.setBranchVisibility(ids.length === boxes.length ? null : ids); } catch (_) {}
      saveViewSoon();
    };
    var boxes = list.map(function (b) {
      var input = h('input', {type: 'checkbox', checked: on(b.id)});
      input.addEventListener('change', apply);
      return {id: Number(b.id), input: input, el: h('label', {'class': 'branch-row'}, input, h('span', {text: b.name || ('分支 ' + b.id)}), b.length_mm ? h('span', {'class': 'muted', text: ui().num(b.length_mm, 'mm')}) : null)};
    });
    ui().dialog.open({title: '分支显隐', body: [h('div', {'class': 'branch-list'}, boxes.map(function (b) { return b.el; })),
      ui().note('只影响三维显示；所有统计、探针和截面仍按全部分支计算。')],
      actions: [ui().button('全部显示', function () { boxes.forEach(function (b) { b.input.checked = true; }); apply(); }, {kind: 'link'}), h('span', {'class': 'sec-fill'}),
        ui().button('完成', function () { ui().dialog.close('done'); }, {kind: 'primary'})]});
  }
  function setLayers() {
    [S.viewerA, S.viewerB].forEach(function (v) {
      if (!v || !v.setLayers) return;
      // while a section is shown the interior points would hide it: the map on the plane stands for them
      var L = Object.assign({}, S.layers, v === S.viewerA && S.cur && S.cur.slice && S.cur.slice.cut() === 'none' ? {interior: false} : {});
      try { v.setLayers(L); } catch (_) {}
    });
  }
  function setCmap(c) {
    store().setPrefs({cmap: c});
    if (S.cur && S.cur.field) applyField(S.cur.field, S.cur.window);
    if (S.cur && S.cur.split) setSplitField(S.cur.split.field, S.cur.split.window);
  }

  // ------------------------------------------------------------------ status line (O5) and time bar (E9)
  function renderStatusLine() {
    var cur = S.cur;
    if (!cur || !cur.manifest) return;
    var m = cur.manifest;
    var job = cur.job || m.job || {};
    var parts = [];
    var twoCases = cur.compare && cur.compare.manifest.job.display_name !== m.job.display_name;
    if (cur.split || cur.compare) {
      // Two viewports: side, (case when the two cases differ), result, field; status stays in the inspector.
      parts.push(h('span', {'class': 'vp-side', text: '左'}));
      if (twoCases) parts.push(h('span', {'class': 'vp-case', text: hideName() ? '病例' : m.job.display_name}));
      if (cur.compare) parts.push(h('span', {'class': 'vp-result', text: (m.result && m.result.display_name) || ''}));
      parts.push(h('span', {'class': 'vp-field', text: fieldName(m, cur.field)}));
      if (cur.compare && cur.compare.mode === 'each') parts.push(h('span', {'class': 'vp-warn', text: '色标各自'}));
      ui().fill(S.els.vpA.status, parts);
    }
    if (cur.split) { S.els.vpB.status.replaceChildren(splitFieldSelect()); return; }
    if (cur.compare) {
      var cm = cur.compare.manifest;
      ui().fill(S.els.vpB.status, h('span', {'class': 'vp-side', text: '右'}), twoCases ? h('span', {'class': 'vp-case', text: cm.job.display_name}) : null,
        h('span', {'class': 'vp-result', text: cm.result.display_name || ''}), h('span', {'class': 'vp-field', text: fieldName(cm, cur.compare.field)}),
        cur.compare.mode === 'each' ? h('span', {'class': 'vp-warn', text: '色标各自'}) : null);
      return;
    }
    // single view: nothing on the stage — the top bar and the inspector carry result, status and review
    S.els.vpA.status.replaceChildren();
  }
  function splitFieldSelect() {
    var cur = S.cur;
    var opts = visibleFields(cur.manifest).map(function (f) { return {value: f.id, label: f.short_label || f.label || f.id}; });
    return h('label', {'class': 'vp-fieldpick'}, h('span', {'class': 'vp-side', text: '右'}), ui().select(opts, cur.split.field, function (v) { setSplitField(v, 'adaptive'); }, {'aria-label': '右侧字段'}));
  }
  function renderTimebar() {
    var m = S.cur && S.cur.manifest;
    var axis = (m && m.time && m.time.axis) || [];
    var el = S.els.timebar;
    if (axis.length <= 1) { el.hidden = true; el.replaceChildren(); return; }
    el.hidden = false;
    el.replaceChildren(ui().iconButton('play', '播放（尚未开放）', null, {cls: 'tb-play'}), h('span', {'class': 'muted', text: axis.length + ' 帧 · 当前 ' + (axis[0].label || '第 1 帧')}),
      h('span', {'class': 'muted', text: '入口流量用协议标准波形，不是该患者实测。'}));
    el.firstChild.disabled = true;
  }

  // ------------------------------------------------------------------ inspector
  function currentTab() {
    var t = S.tab || store().prefs().tab;
    var tabs = tabList();
    return tabs.some(function (x) { return x.id === t; }) ? t : 'overview';
  }
  function tabList() {
    var tabs = [{id: 'overview', label: '概览'}];
    if (S.cur && S.cur.slice) tabs.push({id: 'slice', label: '截面'});
    if (S.cur && S.cur.measure) tabs.push({id: 'measure', label: '测量'});
    if (S.cur && S.cur.region) tabs.push({id: 'region', label: '区域'});
    if (S.cur && S.cur.annotTool) tabs.push({id: 'annot', label: '标注'});
    extCall('tabs').forEach(function (t) { if (t && t.id && t.label) tabs.push(t); });
    tabs.push({id: 'reading', label: '读数'}, {id: 'bookmarks', label: '书签'});
    if (!S.offline) tabs.push({id: 'tools', label: '工具'});
    if (S.cur && S.cur.compare) tabs.push({id: 'compare', label: '比较'});
    return tabs;
  }
  function setInspector(open) { store().setPrefs({inspector: open}); applyPanels(); renderToolbar(); if (open) renderInspector(); }
  // The reading tab is a transient answer to one click; a new result opens on the overview again.
  function setTab(id) { S.tab = id; if (['compare', 'reading', 'slice', 'measure', 'region', 'annot'].indexOf(id) < 0) store().setPrefs({tab: id}); renderInspector(); }
  function renderInspector() {
    var cur = S.cur;
    var E = S.els;
    if (!cur || !cur.manifest) return;
    ns.overview.header(E.inspHead, overviewCtx());
    var tab = currentTab();
    ui().fill(E.tabs, tabList().map(function (t) {
      var b = h('button', {type: 'button', role: 'tab', 'class': 'tab' + (t.id === tab ? ' on' : ''), 'aria-selected': String(t.id === tab), text: t.label, dataset: {tab: t.id}});
      b.addEventListener('click', function () { setTab(t.id); });
      return b;
    }), h('span', {'class': 'sec-fill'}), ui().iconButton('chevron-right', '收起检查器', function () { setInspector(false); }, {cls: 'tab-collapse'}));
    var body = E.inspBody;
    if (tab === 'overview') ns.overview.render(body, overviewCtx());
    else if (extHandled('renderTab', tab, body)) { /* an extension's tab */ }
    else if (tab === 'slice') renderSlice(body);
    else if (tab === 'measure') renderMeasure(body);
    else if (tab === 'region') renderRegion(body);
    else if (tab === 'annot') renderAnnot(body);
    else if (tab === 'reading') renderReading(body);
    else if (tab === 'bookmarks') renderBookmarks(body);
    else if (tab === 'tools') renderTools(body);
    else if (tab === 'compare' && cur.compare && ns.compare) renderComparePanel(body);
    if (body.scrollTop !== undefined && S.lastTab !== tab) body.scrollTop = 0;
    S.lastTab = tab;
  }
  // Results of the same input (companions included), named by their model cards; a date is added when two
  // results share a name.  Offline packages carry one result only: no switch.
  function siblings() {
    var cur = S.cur;
    var m = cur.manifest;
    if (S.offline) return [];
    var sha = (m.job && m.job.input_sha256) || (cur.job && cur.job.input_sha256);
    var ids = {};
    var out = [];
    var add = function (id, name, created) { if (!id || ids[id]) return; ids[id] = true; out.push({jobId: id, name: name, created: created || '', current: id === cur.jobId}); };
    add(cur.jobId, (m.result && m.result.display_name) || ui().resultName(cur.job, S.cards), m.job && m.job.created_at);
    ((m.job && m.job.companions) || []).forEach(function (c) {
      if (!c || !c.job_id) return;
      var j = (S.jobs || []).filter(function (x) { return x.id === c.job_id; })[0];
      if (j && j.status !== 'done') return;
      add(c.job_id, ui().resultName(j || {model_release: {id: c.release_id}}, S.cards), j && j.created_at);
    });
    (S.jobs || []).forEach(function (j) { if (sha && j.input_sha256 === sha && j.status === 'done') add(j.id, ui().resultName(j, S.cards), j.created_at); });
    var count = {};
    out.forEach(function (s) { count[s.name] = (count[s.name] || 0) + 1; });
    out.forEach(function (s) { s.title = s.name + (s.created ? ' · ' + ui().time(s.created) : ''); if (count[s.name] > 1 && s.created) s.name += ' · ' + ui().time(s.created).slice(5, 10); });
    return out;
  }
  function overviewCtx() {
    var cur = S.cur;
    var locked = ui().reviewInfo((cur.job && cur.job.review) || cur.manifest.job.review).key === 'reviewed';
    var tsec = cur.job && cur.job.timing && (cur.job.timing.compute_s || cur.job.timing.compute_seconds);
    if (tsec === undefined || tsec === null) tsec = cur.job && cur.job.compute_seconds;
    if ((tsec === undefined || tsec === null) && cur.manifest.provenance && cur.manifest.provenance.timing_s) tsec = cur.manifest.provenance.timing_s.total;
    return {job: cur.job, manifest: cur.manifest, cards: S.cards, siblings: siblings(), timeline: cur.timeline || null, offline: S.offline, hideName: hideName(), computeSeconds: tsec,
      onZone: zoneHighlight, fieldScale: fieldScaleFor, zoneMetric: cur.zoneMetric, zoneTable: Boolean(cur.zoneTable), narrativeOpen: Boolean(cur.narrativeOpen),
      onZoneMetric: function (id) { cur.zoneMetric = id; renderInspector(); },
      onZoneTable: function () { cur.zoneTable = !cur.zoneTable; renderInspector(); },
      onToggleNarrative: function () { cur.narrativeOpen = !cur.narrativeOpen; renderInspector(); },
      tier: store().prefs().tier, locked: locked, selectedFinding: cur.findingSel, findingsExpanded: cur.findingsExpanded,
      onLens: openLens, onFinding: selectFinding, onOpenResult: function (id) { go(id); },
      onReview: S.offline ? null : reviewDialog, onEditNarrative: S.offline || locked ? null : narrativeDialog, onModelCard: modelCardDialog,
      onConfirmRest: S.offline ? null : confirmRest, onToggleFindings: function () { cur.findingsExpanded = !cur.findingsExpanded; renderInspector(); },
      onAddFinding: S.offline || !ns.review ? null : startAddFinding, onLegacy: legacyDialog};
  }
  function openLens(ref) {
    S.lensRef = ref;
    if (ref && ref.kind === 'finding' && ref.item && ref.item.id) S.cur.findingSel = ref.item.id;
    setTab('reading');
  }
  function renderReading(body) {
    var cur = S.cur;
    var parts = [];
    if (cur.cursor) parts.push(cursorPanel());
    if (ns.review && S.lensRef && S.lensRef.kind === 'finding' && S.lensRef.item && S.lensRef.item.id) parts.push(decisionBar(S.lensRef.item));
    var probeOn = Boolean(cur.probe && ns.probe && S.lensRef && S.lensRef.kind === 'point' && S.lensRef.side !== 'b');
    if (probeOn) { try { parts.push(ns.probe.card(cur.result, cur.probe, probeCtx())); } catch (e) { parts.push(ui().note('探针读数失败：' + (e && e.message || e))); } }
    var lensBox = h('div', {'class': 'lens-box' + (probeOn && !S.lensOpen ? ' folded' : '')});
    // A value picked in the right-hand viewport of a comparison belongs to the other result: read it against that manifest.
    var rightSide = Boolean(S.lensRef && S.lensRef.side === 'b' && cur.compare);
    var res = rightSide ? cur.compare.result : cur.result;
    ns.lens.render(lensBox, S.lensRef, {manifest: rightSide ? cur.compare.manifest : cur.manifest, hideName: hideName() && !rightSide,
      trustAt: function (vi) { return trustAt(res, vi); }, onClear: S.lensRef ? function () { unpinProbe(); S.lensRef = null; renderInspector(); } : null});
    if (rightSide) parts.push(ui().note('右侧结果：' + cur.compare.manifest.job.display_name + ' · ' + cur.compare.manifest.result.display_name));
    if (probeOn) parts.push(h('button', {type: 'button', 'class': 'lens-toggle', 'aria-expanded': String(Boolean(S.lensOpen)), onclick: function () { S.lensOpen = !S.lensOpen; renderInspector(); }},
      ui().icon(S.lensOpen ? 'chevron-down' : 'chevron-right'), h('span', {text: '这个数怎么来的'})));
    parts.push(lensBox);
    var log = probeLog(cur);
    if (log.length && ns.probe) parts.push(ns.probe.logSection(cur.result, {ui: ui(), rows: log, fieldId: cur.field,
      onDelete: function (id) { setProbeLog(cur, log.filter(function (r) { return r.id !== id; })); },
      onClear: function () { setProbeLog(cur, []); },
      onCopy: function () { copyText(ns.probe.tsv(probeLog(cur)), '已复制 ' + log.length + ' 行探针记录（TSV）。'); },
      onCsv: function () { if (typeof root.Blob === 'function') ui().downloadBlob(new root.Blob([ns.probe.csv(probeLog(cur))], {type: 'text/csv;charset=utf-8'}), fileBase(cur) + '_probes.csv'); }}));
    ui().fill(body, parts);
  }
  // ------------------------------------------------------------------ probe (S3): pinned on a click, section readings, log
  function pinProbe(e) {
    var cur = S.cur;
    if (!cur || !cur.result || !ns.probe || !e) return;
    try { ns.probe.clear(S.viewerA); } catch (_) {}
    cur.probe = null;
    var keys = ns.probe.requiredArrays(cur.result).filter(function (k) { return !cur.result.has(k); });
    var myCur = cur;
    Promise.resolve(keys.length ? cur.result.preload(keys) : null).then(function () {
      if (S.cur !== myCur) return;
      var thickness = myCur.slice ? myCur.slice.state().thickness : 2;
      myCur.probe = ns.probe.pick(myCur.result, e, {thickness: thickness});
      if (myCur.probe) ns.probe.draw(S.viewerA, myCur.result, myCur.probe);
      if (currentTab() === 'reading') renderInspector();
    }).catch(function (err) { if (S.cur === myCur) ui().toast('探针读数失败：' + (err && err.message || err), {kind: 'error'}); });
  }
  function unpinProbe() {
    var cur = S.cur;
    if (!cur) return;
    try { ns.probe.clear(S.viewerA); } catch (_) {}
    cur.probe = null;
  }
  function probeCtx() {
    var cur = S.cur;
    return {ui: ui(), manifest: cur.manifest, field: cur.field,
      onField: function (id) { applyField(id, 'adaptive'); },
      onRecord: function () { var log = probeLog(cur); var r = ns.probe.row(cur.result, cur.probe, log); setProbeLog(cur, log.concat([r])); ui().toast('已记录 ' + r.id + '（共 ' + (log.length + 1) + ' 行）。', {kind: 'ok', ms: 2500}); },
      onClose: function () { unpinProbe(); S.lensRef = null; renderInspector(); },
      onSlice: function (plane) { openSliceAt(plane); }};
  }
  function probeLog(cur) {
    if (!cur.probeLog) cur.probeLog = ns.probe ? ns.probe.loadLog(cur.runIdentity) : [];
    return cur.probeLog;
  }
  function setProbeLog(cur, rows) { cur.probeLog = rows; if (ns.probe) ns.probe.saveLog(cur.runIdentity, rows); renderInspector(); }
  function fileBase(cur) { var name = (cur.manifest.job && cur.manifest.job.display_name) || ''; return hideName() || !name ? 'case' : name; }
  function copyText(text, done) {
    var ok = function () { ui().toast(done || '已复制。', {kind: 'ok', ms: 2500}); };
    var fail = function () { ui().toast('浏览器不允许复制；可以用导出。', {kind: 'error'}); };
    try { if (root.navigator && root.navigator.clipboard && root.navigator.clipboard.writeText) { root.navigator.clipboard.writeText(text).then(ok, fail); return; } } catch (_) {}
    fail();
  }
  // The section tool opened on a given plane (the volume probe's 「看截面」).
  function openSliceAt(plane) {
    var cur = S.cur;
    if (!cur || !plane) return;
    var st = {basis: 'pick', pick: {origin: plane.origin, normal: plane.normal, picks: 1}, picks: [], shift: 0, pitch: 0, yaw: 0, offU: 0, offV: 0};
    if (cur.slice) { cur.slice.set(st); cur.slice.look(true); setTab('slice'); return; }
    cur.sliceState = Object.assign({}, cur.sliceState || {}, st);
    toggleSlice();
  }

  // ------------------------------------------------------------------ findings review, annotations, labels (S4)
  function isLocked(cur) { return ui().reviewInfo((cur.job && cur.job.review) || (cur.manifest.job && cur.manifest.job.review)).key === 'reviewed'; }
  function editable(cur) { return !S.offline && !isLocked(cur); }
  var LOCKED_TEXT = '已复核锁定：判定、人工发现和标注只读；需要修改请先「重新打开」。';
  function reviewDoc(cur) { return ns.review.docOf(cur.manifest); }
  function reviewSaver(cur) {
    if (!cur.reviewSaver) cur.reviewSaver = ns.review.saver({
      put: function (payload) { return api().findingsReview(cur.jobId, payload); },
      version: function () { return cur.job && cur.job.version; },
      onSaved: function (doc, version) {
        if (S.cur !== cur) return;
        cur.manifest.analysis.findings.review = doc;
        if (cur.job && version !== undefined) cur.job.version = version;
        cur.reviewStatus = '已保存';
        if (currentTab() === 'reading' || currentTab() === 'overview') renderInspector();
        drawLabels();
      },
      onError: function (e) { if (S.cur !== cur) return; cur.reviewStatus = null; conflictOr(e, '保存发现判定失败'); }
    });
    return cur.reviewSaver;
  }
  // Local update first (the list and the bar follow at once), then the debounced save.
  function applyReview(cur, doc) {
    var a = cur.manifest.analysis || (cur.manifest.analysis = {});
    var f = a.findings || (a.findings = {items: []});
    f.review = Object.assign({}, f.review || {}, {items: doc.items, added: doc.added});
    cur.reviewStatus = '保存中…';
    reviewSaver(cur).save({items: doc.items, added: doc.added});
    renderInspector();
    drawLabels();
  }
  function decisionBar(item) {
    var cur = S.cur, id = item.id;
    return ns.review.bar({ui: ui(), item: item, doc: reviewDoc(cur), editable: editable(cur), lockedText: S.offline ? '离线报告：只读' : LOCKED_TEXT, status: cur.reviewStatus,
      onDecision: function (d) { applyReview(cur, ns.review.setDecision(reviewDoc(cur), id, d)); },
      onNote: function (t) { applyReview(cur, ns.review.setNote(reviewDoc(cur), id, t)); },
      onRemove: item.manual ? function () {
        ui().confirm('删除人工发现 ' + id + '？', ['「' + (item.text || '') + '」'], {confirmLabel: '删除', danger: true}).then(function (ok) {
          if (!ok || S.cur !== cur) return;
          S.lensRef = null; cur.findingSel = null;
          try { S.viewerA.setMarkers([]); } catch (_) {}
          applyReview(cur, ns.review.removeManual(reviewDoc(cur), id));
        });
      } : null});
  }
  // One click on the wall, then cb(xyz).  Used by 新增人工发现 and by 钉标注.
  function pickOnce(message, cb) {
    var v = S.viewerA;
    if (!v || !v.canvasElement) return;
    cancelPickOnce();
    var cv = v.canvasElement, press = null;
    var down = function (e) { if (e.button === 0) press = {x: e.clientX, y: e.clientY, id: e.pointerId}; };
    var up = function (e) {
      var click = press && press.id === e.pointerId && Math.hypot(e.clientX - press.x, e.clientY - press.y) < 6;
      press = null;
      if (!click) return;
      S.pickTakenAt = Date.now();
      var hit = v.surfaceAt(e.clientX, e.clientY);
      if (!hit || !hit.xyz) { ui().toast('没有点到管壁。', {kind: 'info', ms: 2500}); return; }
      var fn = S.pickOnceCb;
      cancelPickOnce();
      fn(hit.xyz);
    };
    cv.addEventListener('pointerdown', down, true);
    cv.addEventListener('pointerup', up, true);
    S.pickOnce = {cv: cv, down: down, up: up}; S.pickOnceCb = cb;
    cv.style.cursor = 'crosshair';
    var hud = S.els.vpA.hud; hud.hidden = false; hud.textContent = message + ' · Esc 取消';
  }
  function cancelPickOnce() {
    if (!S.pickOnce) return;
    var p = S.pickOnce;
    p.cv.removeEventListener('pointerdown', p.down, true); p.cv.removeEventListener('pointerup', p.up, true);
    p.cv.style.cursor = '';
    S.pickOnce = null; S.pickOnceCb = null;
    if (S.cur && S.cur.slice) sliceChanged(S.cur); else S.els.vpA.hud.hidden = true;
    if (S.cur && S.cur.annotTool && currentTab() === 'annot') renderInspector();
  }
  function withCenterline(cur, fn) {
    var keys = ns.probe ? ns.probe.requiredArrays(cur.result).filter(function (k) { return !cur.result.has(k); }) : [];
    Promise.resolve(keys.length ? cur.result.preload(keys) : null).then(function () { if (S.cur === cur) fn(); }, function (e) { ui().toast('读取中心线失败：' + (e && e.message || e), {kind: 'error'}); });
  }
  function whereText(cur, xyz) {
    var c = root.WssReportCommon, groups = ns.probe ? ns.probe.groups(cur.result) : [], pr = null;
    try { pr = c && groups.length ? c.projectToCenterline(groups, xyz) : null; } catch (_) { pr = null; }
    return (pr ? pr.name + ' · ' : '') + '(' + xyz.map(function (v) { return v.toFixed(1); }).join(', ') + ') mm';
  }
  function startAddFinding() {
    var cur = S.cur;
    if (!cur || !cur.result || !editable(cur)) return;
    withCenterline(cur, function () {
      pickOnce('在管壁上点一处，新增一条人工发现', function (xyz) {
        ns.review.addDialog(ui(), whereText(cur, xyz), function (v) {
          var fm = ns.overview.findingsModel(cur.manifest);
          var r = ns.review.addManual(reviewDoc(cur), cur.result, fm.items, xyz, v.text, v.severity);
          if (r.error) { ui().toast(r.error, {kind: 'error'}); return; }
          applyReview(cur, r.doc);
          selectFinding(ns.review.manualItem(r.item));
        });
      });
    });
  }
  function legacyDialog(list) {
    var dec = {confirmed: '已确认', rejected: '已驳回'};
    ui().dialog.open({title: '规则更新前的判定', body: [ui().note('分析规则更新后没能对上新发现的旧判定。只读，服务端一直保留；一页纸附录里也有。'),
      h('div', {'class': 'legacy-list'}, list.map(function (x) {
        return h('div', {'class': 'meas-row'}, h('b', {'class': 'meas-id', text: x.id || ''}),
          h('span', {'class': 'meas-text', text: [x.label || x.kind || '', x.branch || '', x.value !== undefined ? ui().num(x.value, x.units) : '', dec[x.decision] || '', x.note || ''].filter(Boolean).join(' · ')}));
      }))]});
  }
  // Annotation pins and automatic labels, from the result and the prefs; redrawn after every change.
  function annotItems(cur) { if (!cur.annotItems) cur.annotItems = ns.annot.itemsOf(cur.manifest); return cur.annotItems; }
  function drawLabels() {
    var cur = S.cur, v = S.viewerA;
    if (!cur || !cur.result || !v || !ns.annot) return;
    var L = store().prefs().labels || {};
    withCenterline(cur, function () {
      try { ns.annot.drawPins(v, cur.result, L.annotations ? annotItems(cur) : []); } catch (_) {}
      var fm = ns.overview.findingsModel(cur.manifest), rv = fm.review;
      var list = fm.items.filter(function (it) { return !(rv[it.id] && rv[it.id].decision === 'rejected'); });
      try { ns.annot.drawAuto(v, cur.result, cur.manifest, {branches: L.branches, findings: L.findings, maxd: L.maxd, findingsList: list}); } catch (_) {}
    });
  }
  function setLabelPref(patch) {
    store().setPrefs({labels: Object.assign({}, store().prefs().labels, patch)});
    drawLabels();
  }
  function annotSaver(cur) {
    if (!cur.annotSaver) cur.annotSaver = ns.annot.saver({
      put: function (payload) { return api().annotations(cur.jobId, payload); },
      version: function () { return cur.job && cur.job.version; },
      onSaved: function (items, version) {
        if (S.cur !== cur) return;
        cur.annotItems = items;
        cur.manifest.analysis.annotations = Object.assign({}, cur.manifest.analysis.annotations || {}, {items: items});
        if (cur.job && version !== undefined) cur.job.version = version;
        drawLabels();
        if (currentTab() === 'annot') renderInspector();
      },
      onError: function (e) { if (S.cur !== cur) return; cur.annotItems = null; conflictOr(e, '保存标注失败'); }
    });
    return cur.annotSaver;
  }
  function setAnnots(cur, items) {
    cur.annotItems = items;
    annotSaver(cur).save(items);
    drawLabels();
    if (currentTab() === 'annot') renderInspector();
  }
  function toggleAnnot() {
    var cur = S.cur;
    if (!cur || !cur.result || !ns.annot) return;
    if (cur.annotTool) { cur.annotTool = false; cancelPickOnce(); if (S.tab === 'annot') S.tab = null; renderToolbar(); renderInspector(); return; }
    if (!store().prefs().labels.annotations) setLabelPref({annotations: true});
    cur.annotTool = true;
    setTab('annot'); renderToolbar();
  }
  function renderAnnot(body) {
    var cur = S.cur;
    if (!cur.annotTool) { ui().fill(body, ui().empty('标注已关闭。')); return; }
    ns.annot.panel(body, {ui: ui(), items: annotItems(cur), editable: editable(cur), lockedText: S.offline ? '离线报告：只读' : LOCKED_TEXT, placing: Boolean(S.pickOnce),
      onClose: toggleAnnot, onFly: function (xyz) { flyTo(xyz, 10); },
      onEdit: function (id, text) { setAnnots(cur, ns.annot.edit(annotItems(cur), id, text)); },
      onRemove: function (id) { setAnnots(cur, ns.annot.remove(annotItems(cur), id)); },
      onPlace: function () {
        if (S.pickOnce) { cancelPickOnce(); return; }
        withCenterline(cur, function () {
          pickOnce('在管壁上点一处放标注', function (xyz) {
            ns.annot.addDialog(ui(), whereText(cur, xyz), function (text) {
              var r = ns.annot.add(annotItems(cur), cur.result, xyz, text);
              if (r.error) { ui().toast(r.error, {kind: 'error'}); return; }
              setAnnots(cur, r.items);
            });
          });
          renderInspector();
        });
      }});
  }

  // ------------------------------------------------------------------ measurement and regions (S3)
  function toolSupported(name) {
    try { return S.viewerA && S.viewerA.supports().tools.indexOf(name) >= 0; } catch (_) { return false; }
  }
  function toggleMeasure() {
    var cur = S.cur;
    if (!cur || !cur.result || !ns.measure || !S.viewerA) return;
    if (cur.measure) { exitMeasure(); return; }
    if (cur.compare || cur.split) { ui().toast('测量在单视口里用。请先退出比较或并排。', {kind: 'info'}); return; }
    if (cur.region && cur.region.picking()) cur.region.setPicking(false);
    if (cur.slice && cur.slice.picking()) cur.slice.setPicking(false);
    var myCur = cur, keys = ns.measure.requiredArrays(cur.result).filter(function (k) { return !cur.result.has(k); });
    Promise.resolve(keys.length ? cur.result.preload(keys) : null).then(function () {
      if (S.cur !== myCur || myCur.measure) return;
      myCur.measure = ns.measure.create(S.viewerA, myCur.result, {runIdentity: myCur.runIdentity,
        onChange: function () { if (S.cur === myCur && currentTab() === 'measure') renderInspector(); },
        onNote: function (t) { ui().toast(t, {kind: 'info', ms: 3000}); }});
      if (!myCur.measure.items().length) myCur.measure.setMode('distance');
      setTab('measure'); renderToolbar();
    }).catch(function (e) { if (S.cur === myCur) ui().toast('测量不可用：' + (e && e.message || e), {kind: 'error'}); });
  }
  function exitMeasure() {
    var cur = S.cur;
    if (!cur || !cur.measure) return;
    try { cur.measure.dispose(); } catch (_) {}
    cur.measure = null;
    if (S.tab === 'measure') S.tab = null;
    renderToolbar(); renderInspector();
  }
  function renderMeasure(body) {
    var cur = S.cur;
    if (!cur.measure) { ui().fill(body, ui().empty('测量已关闭。')); return; }
    ns.measure.panel(body, cur.measure, {ui: ui(), onClose: exitMeasure, onFly: function (xyz) { flyTo(xyz, 20); },
      onCopy: function (t) { copyText(t, '已复制测量。'); }});
  }
  function toggleRegion() {
    var cur = S.cur;
    if (!cur || !cur.result || !ns.region || !S.viewerA) return;
    if (cur.region) { exitRegion(); return; }
    if (cur.compare || cur.split) { ui().toast('区域统计在单视口里用。请先退出比较或并排。', {kind: 'info'}); return; }
    if (cur.measure && cur.measure.mode()) cur.measure.setMode(null);
    var myCur = cur, keys = ns.region.requiredArrays(cur.result).filter(function (k) { return !cur.result.has(k); });
    Promise.resolve(keys.length ? cur.result.preload(keys) : null).then(function () {
      if (S.cur !== myCur || myCur.region) return;
      myCur.region = ns.region.create(S.viewerA, myCur.result, {
        field: function () { return myCur.field; },
        values: function (fid) { var f = fieldById(myCur.manifest, fid); return f && f.arrays && f.arrays.read && myCur.result.has(f.arrays.read) ? myCur.result.array(f.arrays.read) : null; },
        onChange: function (what) { if (S.cur !== myCur || currentTab() !== 'region') return; if (what === 'region' && myCur.regionPanel) myCur.regionPanel.update(); else renderInspector(); },
        onNote: function (t) { ui().toast(t, {kind: 'info', ms: 3000}); }});
      setTab('region'); renderToolbar();
    }).catch(function (e) { if (S.cur === myCur) ui().toast('区域统计不可用：' + (e && e.message || e), {kind: 'error'}); });
  }
  function exitRegion() {
    var cur = S.cur;
    if (!cur || !cur.region) return;
    try { cur.region.dispose(); } catch (_) {}
    cur.region = null; cur.regionPanel = null;
    if (S.tab === 'region') S.tab = null;
    renderToolbar(); renderInspector();
  }
  function renderRegion(body) {
    var cur = S.cur;
    if (!cur.region) { ui().fill(body, ui().empty('区域统计已关闭。')); return; }
    cur.regionPanel = ns.region.panel(body, cur.region, {ui: ui(), onClose: exitRegion,
      fieldLabel: function () { return fieldName(cur.manifest, cur.field); },
      units: function () { var f = fieldById(cur.manifest, cur.field); return f ? ui().unitText(f.units) : ''; }});
  }
  function trustAt(result, vertexIndex) {
    var m = result && result.manifest;
    var key = m && m.geometry && m.geometry.display_mesh && m.geometry.display_mesh.trust;
    if (!key || typeof result.array !== 'function') return null;
    try { var a = result.array(key); return a && vertexIndex < a.length ? a[vertexIndex] : null; } catch (_) { return null; }
  }
  // Colour of a value under the field's current scale (the vessel's colours), for the overview's zone map and chips.
  function fieldScaleFor(fieldId) {
    var cur = S.cur;
    if (!cur || !cur.manifest || !ns.colormap) return null;
    var f = (cur.manifest.fields || []).filter(function (x) { return x && x.id === fieldId; })[0];
    if (!f) return null;
    // Resolved the way the viewer resolves it (same window, log hint and colour table), from the manifest's own
    // statistics, so the colour never depends on which field the viewport happens to show at this moment.
    var st = f.statistics || {}, disp = f.display || {};
    var stats = {p99: disp.p99 !== undefined && disp.p99 !== null ? disp.p99 : st.p99, min: st.min, max: st.max};
    var spec = scaleSpec(cur.field === fieldId && !cur.compare ? cur.window : 'adaptive');
    var sc = null;
    try { sc = ns.colormap.resolve(f, spec, stats).scale; } catch (_) { sc = null; }
    if (!sc) return null;
    return {scale: sc, color: function (v) { var n = Number(v); return Number.isFinite(n) ? ns.colormap.rgbToHex(sc.color(n)) : null; }};
  }
  // Hovering a zone in the overview marks its prediction points on the vessel; null clears the mark.
  function zoneHighlight(zone) {
    var v = S.viewerA, cur = S.cur;
    if (!v || !cur || !cur.result || typeof v.highlight !== 'function') return;
    if (!zone) { try { v.highlight(null); } catch (_) {} return; }
    var seg = null, s = null;
    try { seg = cur.result.array('ps'); } catch (_) { seg = null; }
    try { s = cur.result.array('p_s'); } catch (_) { s = null; }
    if (!seg) return;
    var sid = zone.segment_id, rg = Array.isArray(zone.s_range_mm) ? zone.s_range_mm : null, idx = [];
    for (var i = 0; i < seg.length; i++) {
      if (seg[i] !== sid) continue;
      if (rg && s && !(s[i] >= rg[0] && s[i] <= rg[1])) continue;
      idx.push(i);
    }
    try { v.highlight(idx, {}); } catch (_) {}   // the rest of the wall fades; the zone keeps its true colours
  }
  function onPick(e, side) {
    var cur = S.cur;
    if (!cur || !cur.manifest) return;
    if (side !== 'b' && cur.slice && cur.slice.clickTaken()) return;   // the click placed a section point
    if (side !== 'b' && ((cur.measure && cur.measure.clickTaken()) || (cur.region && cur.region.clickTaken()))) return;   // a measuring / region point
    if (side !== 'b' && (S.pickOnce || Date.now() - (S.pickTakenAt || 0) < 500)) return;   // a manual finding / annotation point
    if (side !== 'b') pinProbe(e);
    var fid = side === 'b' ? (cur.compare ? cur.compare.field : cur.split && cur.split.field) : cur.field;
    if (side !== 'b') cur.selection = e.pointIndex === undefined ? null : e.pointIndex;
    var pv = side === 'b' ? S.viewerB : S.viewerA;
    if (pv && typeof pv.select === 'function' && e.pointIndex !== undefined) { try { pv.select(e.pointIndex); } catch (_) {} }
    S.lensRef = {kind: 'point', side: side, field: fid, pointIndex: e.pointIndex, vertexIndex: e.vertexIndex, xyz: e.xyz, segmentId: e.segmentId, s_from_root_mm: e.s_from_root_mm,
      value: e.value, valueSource: e.valueSource || null, values: e.values || null, distance_mm: e.distance_mm, vertexBits: typeof e.trust === 'number' ? e.trust : undefined};
    if (store().prefs().inspector) setTab('reading');
    saveViewSoon();
  }

  // ------------------------------------------------------------------ section (second phase S1)
  // Volume results: a plane through the vessel with the filled map on it; numbers from the classic report's core.
  function toggleSlice() {
    var cur = S.cur;
    if (!cur || !cur.result || !ns.slice || !S.viewerA) return;
    if (cur.slice) { exitSlice(); return; }
    if (cur.compare || cur.split) { ui().toast('截面在单视口里用。请先退出比较或并排。', {kind: 'info'}); return; }
    if (!ns.slice.supported(cur.result)) { ui().toast('这份结果没有体内预测点，不能看截面。', {kind: 'info'}); return; }
    if (cur.cursor) toggleCursor();
    var myCur = cur;
    var keys = ns.slice.requiredArrays(cur.result).filter(function (k) { return !cur.result.has(k); });
    Promise.resolve(keys.length ? cur.result.preload(keys) : null).then(function () {
      if (S.cur !== myCur || myCur.slice) return;
      if (myCur.field === 'wall_pressure' && fieldById(myCur.manifest, 'pressure')) applyField('pressure', 'adaptive');
      var a = myCur.manifest.analysis || {}, mm = a.morphology && a.morphology.aorta && a.morphology.aorta.max;
      myCur.slice = ns.slice.create(S.viewerA, myCur.result, {
        hint: {xyz: mm && Array.isArray(mm.xyz_mm) ? mm.xyz_mm : null, field: myCur.field}, state: myCur.sliceState || null,
        cmap: function () { return store().prefs().cmap; }, global: sliceGlobal,
        onChange: function (what) { sliceChanged(myCur, what); }, onNote: function (t) { ui().toast(t, {kind: 'info', ms: 4000}); }
      });
      setLayers();
      myCur.slice.look(true);
      setTab('slice'); renderToolbar(); sliceChanged(myCur);
    }).catch(function (e) { if (S.cur === myCur) ui().toast('截面不可用：' + (e && e.message || e), {kind: 'error'}); });
  }
  function exitSlice() {
    var cur = S.cur;
    if (!cur || !cur.slice) return;
    try { cur.sliceState = cur.slice.state(); } catch (_) {}
    try { cur.slice.dispose(); } catch (_) {}
    cur.slice = null;
    S.els.vpA.hud.hidden = true;
    setLayers(); updateColorbar('a');
    if (S.tab === 'slice') S.tab = null;
    renderToolbar(); renderInspector();
  }
  function sliceChanged(cur, what) {
    if (!cur || S.cur !== cur || !cur.slice) return;
    if (what === 'cut') setLayers();   // a cut shows the interior points again, clipped at the plane
    var t = cur.slice.hudText(), hud = S.els.vpA.hud;
    hud.hidden = !t; hud.textContent = t;
    updateColorbar('a');
  }
  // 「与三维同」: the viewer's scale of that field (±the larger end for the through-plane velocity).
  function sliceGlobal(q) {
    var fs = fieldScaleFor(q === 'pressure' ? 'pressure' : 'speed'), sc = fs && fs.scale;
    if (!sc || !Array.isArray(sc.range)) return null;
    if (q === 'normal') { var m = Math.max(Math.abs(sc.range[0]), Math.abs(sc.range[1])); return {min: -m, max: m}; }
    return {min: sc.log && sc.floor ? sc.floor : sc.range[0], max: sc.range[1], log: Boolean(sc.log)};
  }
  function sliceFileCtx(cur) {
    var name = (cur.manifest.job && cur.manifest.job.display_name) || '';
    return {ui: ui(), caseName: hideName() ? '' : name, fileBase: hideName() || !name ? 'case' : name};
  }
  function renderSlice(body) {
    var cur = S.cur;
    if (!cur.slice) { ui().fill(body, ui().empty('截面已关闭。')); return; }
    var s = cur.slice;
    ns.slice.panel(body, s, {ui: ui(), onClose: exitSlice,
      onZoom: function () { ns.slice.openZoom(s, sliceFileCtx(cur)); },
      onSeries: function (n) { ns.slice.openSeries(s, Object.assign(sliceFileCtx(cur), {count: n, onGo: function () { s.look(true); }})); },
      onQuantity: function (q) {
        var fid = q === 'pressure' ? 'pressure' : 'speed';
        if (cur.field !== fid && fieldById(cur.manifest, fid)) applyField(fid, 'adaptive');
      }});
  }

  // ------------------------------------------------------------------ cursor D1
  function toggleCursor() {
    var cur = S.cur;
    if (!cur || !cur.result || !ns.cursor || !S.viewerA) return;
    if (cur.cursor) {
      cur.cursor = null;
      try { S.viewerA.setCursor(null); } catch (_) {}
      if (S.viewerB && cur.split) { try { S.viewerB.setCursor(null); } catch (_) {} }
      S.els.vpA.hud.hidden = true;
      renderToolbar(); if (currentTab() === 'reading') renderInspector();
      return;
    }
    var branches = (cur.manifest.geometry && cur.manifest.geometry.branches) || [];
    var b0 = branches[0];
    if (!b0) { ui().toast('这份结果没有中心线分支，游标不可用。', {kind: 'error'}); return; }
    var keys = [];
    try { keys = typeof ns.cursor.requiredArrays === 'function' ? ns.cursor.requiredArrays(cur.result) : []; } catch (_) { keys = []; }
    var myCur = cur;
    Promise.resolve(keys.length && cur.result.preload ? cur.result.preload(keys) : null).then(function () {
      if (S.cur !== myCur || myCur.cursor) return;
      var c = ns.cursor.create(myCur.result);
      if (c.available === false) throw new Error('这份结果没有可用的中心线');
      myCur.cursor = c;
      if (typeof c.on === 'function') c.on('change', function () { cursorMoved(); });
      c.set(b0.id, Math.min((b0.length_mm || 0) / 2, 120));
      cursorMoved();
      setTab('reading'); renderToolbar();
    }).catch(function (e) { if (S.cur === myCur) ui().toast('游标不可用：' + (e && e.message || e), {kind: 'error'}); });
  }
  function cursorMoved() {
    var cur = S.cur;
    if (!cur || !cur.cursor) return;
    var st = null;
    try { st = cur.cursor.state(); } catch (_) { st = null; }
    if (st) {
      try { S.viewerA.setCursor(st); } catch (_) {}
      if (S.viewerB && cur.split) { try { S.viewerB.setCursor(st); } catch (_) {} }
    }
    var hud = S.els.vpA.hud;
    hud.hidden = !st;
    if (st) hud.textContent = ns.lens.branchName(cur.manifest, st.segmentId) + ' · ' + ui().num(st.s_mm, 'mm') + ' · 方向键移动 1 mm，Shift 5 mm';
    if (currentTab() === 'reading') renderInspector();
    saveViewSoon();
  }
  function stepCursor(delta) {
    var cur = S.cur;
    if (!cur || !cur.cursor) return;
    var r = null;
    try { r = cur.cursor.step(delta); } catch (_) { r = null; }
    cur.cursorEnd = r && r.atEnd ? r : null;
    cursorMoved();
  }
  function cursorPanel() {
    var cur = S.cur, c = cur.cursor;
    var st = null; try { st = c.state(); } catch (_) {}
    var branches = (cur.manifest.geometry && cur.manifest.geometry.branches) || [];
    var sel = ui().select(branches.map(function (b) { return {value: String(b.id), label: b.name || ('分支 ' + b.id)}; }), st ? String(st.segmentId) : '', function (v) {
      var b = branches.filter(function (x) { return String(x.id) === v; })[0];
      try { c.set(b.id, Math.min((b.length_mm || 0) / 2, 20)); } catch (_) {}
      cursorMoved();
    }, {'aria-label': '分支'});
    var readout = null;
    try { readout = c.readout(cur.field); } catch (_) { readout = null; }
    var rows = [];
    if (readout && readout.stats) Object.keys(readout.stats).forEach(function (k) { var v = readout.stats[k]; if (v !== null && v !== undefined) rows.push({k: k, v: v}); });
    var f = fieldById(cur.manifest, cur.field);
    var statName = {mean: '均值', mean_pa: '均值', p99: 'p99', p99_pa: 'p99', min: '最小', min_pa: '最小', max: '最大', n: '预测点', radius_mm: '中心线半径'};
    var tbl = rows.length ? ui().table([{key: 'k', label: '', render: function (r) { return statName[r.k] || r.k; }},
      {key: 'v', label: '', num: true, render: function (r) { return typeof r.v === 'number' ? (r.k === 'n' ? String(r.v) : r.k === 'radius_mm' ? ui().num(r.v, 'mm') : ui().num(r.v, f && f.units)) : String(r.v); }}], rows, {cls: 'tbl-kv tbl-nohead'})
      : ui().note(readout ? '这个分箱没有预测点。' : '这个字段没有沿程分箱统计。');
    var ends = cur.cursorEnd && Array.isArray(cur.cursorEnd.children) && cur.cursorEnd.children.length ? h('div', {'class': 'cursor-children'}, h('span', {'class': 'muted', text: '到分支端点了，进入：'}),
      cur.cursorEnd.children.map(function (id) { return ui().button(ns.lens.branchName(cur.manifest, id), function () { try { c.set(id, 0); } catch (_) {} cur.cursorEnd = null; cursorMoved(); }, {cls: 'btn-sm'}); })) : null;
    return ui().section('沿血管游标', {actions: ui().button('关闭', toggleCursor, {kind: 'link', cls: 'btn-sm'})},
      h('div', {'class': 'cursor-row'}, sel, ui().iconButton('chevron-left', '向近端 1 mm（←）', function () { stepCursor(-1); }), h('span', {'class': 'cursor-pos', text: st ? ui().num(st.s_mm, 'mm') : '—'}),
        ui().iconButton('chevron-right', '向远端 1 mm（→）', function () { stepCursor(1); })),
      ends,
      h('div', {'class': 'cursor-read'}, h('div', {'class': 'cursor-head'}, h('span', {text: fieldName(cur.manifest, cur.field) + ' · ' + (readout && Array.isArray(readout.bin) ? ui().trim(readout.bin[0]) + '–' + ui().trim(readout.bin[1]) + ' mm 分箱' : '当前分箱')}),
        readout ? ui().button('来源', function () { S.lensRef = {kind: 'cursor', field: cur.field, readout: readout}; renderInspector(); }, {kind: 'link', cls: 'btn-sm'}) : null), tbl),
      ui().note(readout && readout.definition ? readout.definition : '分支内弧长；沿中心线每 2 mm 一箱，箱内预测点统计。'));
  }

  // ------------------------------------------------------------------ findings
  function findingOrder() {
    var fm = ns.overview.findingsModel(S.cur.manifest);
    return S.cur.findingsExpanded ? fm.items : fm.top;
  }
  function selectFinding(item) {
    var cur = S.cur;
    if (!cur || !item) return;
    cur.findingSel = item.id;
    if (S.viewerA) {
      try { S.viewerA.setMarkers(item.xyz_mm ? [{id: item.id || 'f', xyz: item.xyz_mm, label: ns.overview.kindLabel(item), kind: item.kind}] : []); } catch (_) {}
      try { S.viewerA.highlight(Array.isArray(item.point_indices) && item.point_indices.length ? item.point_indices : null, {color: '#1d5d95'}); } catch (_) {}
      if (item.xyz_mm) flyTo(item.xyz_mm, item.extent_mm);
    }
    S.lensRef = {kind: 'finding', item: item};
    setTab('reading');
  }
  function stepFinding(delta) {
    var list = findingOrder();
    if (!list.length) return;
    var i = list.findIndex(function (it) { return it.id === S.cur.findingSel; });
    var next = list[(i < 0 ? (delta > 0 ? 0 : list.length - 1) : (i + delta + list.length) % list.length)];
    selectFinding(next);
  }
  function flyTo(xyz, extent) {
    var v = S.viewerA;
    if (!v || typeof v.getCamera !== 'function' || typeof v.setCamera !== 'function') return;
    try {
      var cam = v.getCamera();
      if (!cam || !cam.position || !cam.target) return;
      var d = [cam.position[0] - cam.target[0], cam.position[1] - cam.target[1], cam.position[2] - cam.target[2]];
      var len = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) || 1;
      var dist = Math.max(40, Math.min(len, (extent || 10) * 6));
      var pos = [xyz[0] + d[0] / len * dist, xyz[1] + d[1] / len * dist, xyz[2] + d[2] / len * dist];
      v.setCamera(Object.assign({}, cam, {position: pos, target: xyz.slice(0, 3)}), {animate: !reducedMotion()});
    } catch (_) {}
  }
  function reducedMotion() { try { return Boolean(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches); } catch (_) { return false; } }
  function showFindings(items) {
    if (!S.viewerA) return;
    try { S.viewerA.setMarkers((items || []).filter(function (it) { return it.xyz_mm; }).map(function (it) { return {id: it.id, xyz: it.xyz_mm, label: ns.overview.kindLabel(it) + ' ' + ns.overview.findingValue(it), kind: it.kind}; })); } catch (_) {}
    try { S.viewerA.highlight(null); } catch (_) {}
  }
  function confirmRest(fm) {
    var cur = S.cur;
    var f = cur.manifest.analysis.findings || {};
    var review = f.review || {};
    // the notes of undecided findings stay (they were wiped before S4)
    var items = ns.review ? ns.review.confirmRest({items: review.items || {}, added: review.added || []}, fm.undecided.map(function (it) { return it.id; })).items
      : Object.assign({}, review.items || {});
    api().findingsReview(cur.jobId, {items: items, added: review.added || [], version: cur.job && cur.job.version}).then(function (r) {
      if (S.cur !== cur) return;
      f.review = r.findings_review || {items: items, added: review.added || []};
      if (cur.job && r.version !== undefined) cur.job.version = r.version;
      ui().toast('已保存：其余 ' + fm.undecided.length + ' 条按自动结果确认。', {kind: 'ok'});
      renderInspector();
    }, function (e) { conflictOr(e, '保存发现判定失败'); });
  }
  function conflictOr(e, what) {
    if (e && e.status === 409) {
      ui().toast('记录已被更新（可能已复核锁定或在别处修改），已读回最新状态。', {kind: 'info'});
      reloadCurrent();
    } else ui().toast(what + '：' + (e && e.message || e), {kind: 'error'});
  }
  function reloadCurrent() { if (S.cur) openJob(S.cur.jobId, parseHash(root.location.hash)); }

  // ------------------------------------------------------------------ dialogs: review (U21), conclusion, model card
  function reviewDialog() {
    var cur = S.cur;
    var job = cur.job || {};
    var review = job.review || (cur.manifest.job && cur.manifest.job.review) || {};
    var info = ui().reviewInfo(review);
    var who = h('input', {type: 'text', maxLength: 80, value: (S.session && (S.session.display_name || S.session.username)) || '', 'aria-label': '复核人'});
    var note = h('textarea', {rows: 3, maxLength: 2000, 'aria-label': '备注'});
    var submit;
    if (info.key === 'reviewed') {
      submit = ui().button('重新打开', function () {
        if (!note.value.trim()) { note.focus(); ui().toast('重新打开要写原因。', {kind: 'error'}); return; }
        send({decision: 'reopen', reviewer: who.value.trim(), note: note.value.trim(), version: job.version});
      }, {kind: 'primary'});
      ui().dialog.open({title: '技术复核记录', body: [
        ui().table([{key: 'k', label: ''}, {key: 'v', label: ''}], [{k: '状态', v: '已复核，结果已锁定'}, {k: '复核人', v: review.by || '—'}, {k: '时间', v: ui().time(review.at) || '—'}, {k: '备注', v: review.note || '—'}], {cls: 'tbl-kv tbl-nohead'}),
        h('h4', {text: '重新打开'}), ui().note('重新打开后可以修改出口、结论和发现判定；原因会记入历史。'),
        h('label', {'class': 'fld'}, h('span', {text: '操作人'}), who), h('label', {'class': 'fld'}, h('span', {text: '原因'}), note)],
        actions: [ui().button('关闭', function () { ui().dialog.close('cancel'); }), submit]});
      return;
    }
    var checks = ['输入单位与尺寸已核对', '出口命名已对照影像核对', '质量提示和几何参照已看过'].map(function (t) { return h('input', {type: 'checkbox', 'aria-label': t}); });
    submit = ui().button('复核通过并锁定', function () {
      if (!who.value.trim()) { who.focus(); ui().toast('请填写复核人。', {kind: 'error'}); return; }
      send({decision: 'approve', reviewer: who.value.trim(), note: note.value.trim(), version: job.version});
    }, {kind: 'primary', disabled: true});
    checks.forEach(function (c) { c.addEventListener('change', function () { submit.disabled = !checks.every(function (x) { return x.checked; }); }); });
    ui().dialog.open({title: '技术复核', body: [
      ui().note('技术复核是操作者对输入、出口和质量的核对，不是临床签字。通过后结果锁定，结论和发现不能再改，直到重新打开。'),
      h('div', {'class': 'checks-form'}, checks.map(function (c, i) { return h('label', {'class': 'check'}, c, h('span', {text: c.getAttribute('aria-label')})); })),
      h('label', {'class': 'fld'}, h('span', {text: '复核人'}), who), h('label', {'class': 'fld'}, h('span', {text: '备注（可空）'}), note)],
      actions: [ui().button('取消', function () { ui().dialog.close('cancel'); }), submit]});
    function send(payload) {
      submit.disabled = true;
      api().review(cur.jobId, payload).then(function (r) {
        ui().dialog.close('done');
        if (S.cur !== cur) return;
        var j = r.job || r;
        cur.job = Object.assign({}, cur.job || {}, j);
        if (cur.manifest.job) cur.manifest.job.review = j.review || cur.manifest.job.review;
        ui().toast(payload.decision === 'approve' ? '技术复核已记录，结果已锁定。' : '已重新打开。', {kind: 'ok'});
        renderInspector(); renderStatusLine(); scheduleRefresh();
      }, function (e) { submit.disabled = false; ui().dialog.close('cancel'); conflictOr(e, '复核没有保存'); });
    }
  }
  function narrativeDialog() {
    var cur = S.cur;
    var nm = ns.overview.narrativeModel(cur.manifest);
    var text = h('textarea', {rows: 8, maxLength: 4000, 'aria-label': '结论'});
    text.value = nm ? nm.sentences.join('\n') : '';
    var save = function (value) {
      api().narrative(cur.jobId, {text: value, version: cur.job && cur.job.version}).then(function (r) {
        ui().dialog.close('done');
        if (S.cur !== cur) return;
        if (r.narrative) cur.manifest.analysis.narrative = r.narrative;
        if (cur.job && r.version !== undefined) cur.job.version = r.version;
        ui().toast(value ? '结论已保存，标为人工编辑。' : '已恢复自动结论。', {kind: 'ok'});
        renderInspector();
      }, function (e) { ui().dialog.close('cancel'); conflictOr(e, '结论没有保存'); });
    };
    ui().dialog.open({title: '编辑结论', body: [text, ui().note('保存后标为人工编辑；一页纸和离线报告用这份文字。')],
      actions: [ui().button('恢复自动', function () { save(''); }, {kind: 'link'}), ui().button('取消', function () { ui().dialog.close('cancel'); }), ui().button('保存', function () { save(text.value.trim()); }, {kind: 'primary'})], focus: text});
  }
  function modelCardDialog() {
    var cur = S.cur;
    var card = cur.manifest.model_card || S.cards[cur.manifest.result.release_id] || null;
    ui().dialog.open({title: '模型说明 · ' + ((card && card.display_name) || cur.manifest.result.display_name || ''), wide: true, body: ns.overview.modelCardBody(card, cur.manifest).concat([h('p', {'class': 'research', text: (card && card.caveat) || '研究用途，非诊断。'})]),
      actions: [ui().button('关闭', function () { ui().dialog.close('done'); })]});
  }
  function shortcutsDialog() {
    var rows = [['J / K', '下一例 / 上一例'], ['1–9', '切换字段（按工具栏顺序）'], ['← / →', '游标打开时沿血管移动 1 mm，Shift 5 mm'], ['[ / ]', '上一个 / 下一个发现'],
      ['L', '光照：平涂 / 柔和'], ['B', '保存书签'], ['G', '沿血管游标开关'], ['S', '截面开关（体场结果）'], ['M', '测量开关'],
      ['↑ / ↓', '截面打开时沿中心线（或法向）移动 1 mm，Shift 5 mm'], ['← / → · PgUp / PgDn', '截面打开时转动截面 2°，Shift 10°'], ['[ / ]（截面）', '截面打开时改厚度 0.4 mm'], ['Esc', '退出当前工具或关闭对话框'], ['?', '这张表']];
    ui().dialog.open({title: '快捷键', body: [ui().table([{key: 'k', label: '键'}, {key: 'v', label: '作用'}], rows.map(function (r) { return {k: r[0], v: r[1]}; }), {cls: 'tbl-keys'}),
      ui().note('在输入框和对话框里不响应；不占用浏览器自己的组合键。')], actions: [ui().button('关闭', function () { ui().dialog.close('done'); })]});
  }

  // ------------------------------------------------------------------ bookmarks D10
  function bookmarkCtx() {
    var cur = S.cur;
    var f = fieldById(cur.manifest, cur.field);
    var st = null;
    try { st = S.viewerA && S.viewerA.getState ? S.viewerA.getState() : null; } catch (_) { st = null; }
    return {jobId: cur.jobId, runIdentity: cur.runIdentity, arraysVersion: cur.arraysVersion, dataVersion: cur.dataVersion, viewerVersion: (ns.viewer && ns.viewer.version) || 'v2',
      field: cur.field, window: cur.window, windowLabel: windowLabel(f, cur.window), fields: ((cur.manifest.fields) || []).map(function (x) { return x.id; }),
      layout: cur.split ? {split: {field: cur.split.field, window: cur.split.window}} : null, state: st || {field: cur.field, time_index: 0},
      fieldLabel: function (id) { return fieldName(cur.manifest, id); }, hasCursor: Boolean(cur.cursor), hasSelection: cur.selection !== null && cur.selection !== undefined,
      defaultName: '书签 ' + ((ns.bookmarks.list(cur.runIdentity).length || 0) + 1) + ' · ' + fieldName(cur.manifest, cur.field),
      fileTag: ns.exporter ? ns.exporter.safeName(hideName() ? '病例' : cur.manifest.job.display_name) : 'result'};
  }
  function saveBookmark() {
    if (!S.cur || !S.cur.manifest || !ns.bookmarks) return;
    ns.bookmarks.saveDialog(bookmarkCtx(), function (bm) {
      ui().toast('已保存书签「' + bm.name + '」。', {kind: 'ok'});
      if (currentTab() === 'bookmarks') renderInspector();
    });
  }
  function renderBookmarks(body) {
    var ctx = bookmarkCtx();
    ctx.onSave = saveBookmark; ctx.onRestore = restoreBookmark; ctx.onChanged = function () {}; ctx.offline = S.offline;
    ns.bookmarks.render(body, ctx);
  }
  function restoreBookmark(bm) {
    var cur = S.cur;
    var c = ns.bookmarks.compatible(bm, bookmarkCtx());
    if (!c.ok) { ui().toast('书签不能用在这份结果上：' + c.reasons.join('；'), {kind: 'error'}); return; }
    if (bm.layout && bm.layout.split) setSplit({field: bm.layout.split.field, window: bm.layout.split.window, link: true}); else exitSplit();
    if (bm.field && bm.field !== cur.field) applyField(bm.field, bm.window || 'adaptive'); else if (bm.window) applyField(cur.field, bm.window);
    if (S.viewerA && bm.state) { try { S.viewerA.applyState(bm.state, {animate: !reducedMotion()}); } catch (_) {} }
    if (bm.cursor && ns.cursor && !cur.cursor) toggleCursor();
    if (bm.cursor && cur.cursor) { try { cur.cursor.set(bm.cursor.segmentId, bm.cursor.s_mm); cursorMoved(); } catch (_) {} }
    cur.selection = bm.selection;
    replaceRoute({bm: bm.id});
    ui().toast('已打开书签「' + bm.name + '」。', {kind: 'ok', ms: 3000});
  }
  function saveViewSoon() {
    if (S.saveTimer) clearTimeout(S.saveTimer);
    S.saveTimer = setTimeout(saveView, 600);
    if (S.saveTimer && S.saveTimer.unref) S.saveTimer.unref();
  }
  function saveView() {
    var cur = S.cur;
    if (!cur || !cur.runIdentity || !cur.result) return;
    var st = null;
    try { st = S.viewerA && S.viewerA.getState ? S.viewerA.getState() : null; } catch (_) { st = null; }
    // A comparison paints both sides on a shared range; that range belongs to the comparison, not to this result's
    // reading position (reopening the result alone must not come back on 「固定范围」).
    if (st && cur.compare) { delete st.scale; delete st.field; }
    store().writeView(cur.runIdentity, {field: cur.field, window: cur.window, viewer: st, split: cur.split ? {field: cur.split.field, window: cur.split.window} : null});
  }

  // ------------------------------------------------------------------ split (same result, two fields) and compare
  function ensureViewerB() {
    if (S.viewerB) return S.viewerB;
    if (!ns.viewer || !S.viewerA) return null;
    try { S.viewerB = ns.viewer.create(S.els.vpB.canvas, {kind: 'main'}); } catch (_) { S.viewerB = null; return null; }
    wireViewer(S.viewerB, 'b');
    if (ns.colorbar) { try { S.colorbarB = ns.colorbar.create(S.els.vpB.legend); } catch (_) { S.colorbarB = null; } }
    if (ns.orientation) { try { S.orientB = ns.orientation.create(S.els.vpB.orient, S.viewerB); } catch (_) { S.orientB = null; } }
    return S.viewerB;
  }
  function disposeViewerB() {
    if (S.unlink) { try { S.unlink(); } catch (_) {} S.unlink = null; }
    [S.colorbarB, S.orientB, S.viewerB].forEach(function (x) { if (x && x.dispose) { try { x.dispose(); } catch (_) {} } });
    S.viewerB = S.colorbarB = S.orientB = null;
    S.els.vpB.root.hidden = true; S.els.grid.classList.remove('split');
    S.els.vpB.status.replaceChildren(); S.els.vpB.canvas.replaceChildren(); S.els.vpB.legend.replaceChildren();
    resizeViewers();
  }
  function setSplit(spec) {
    var cur = S.cur;
    if (!spec) { exitSplit(); return; }
    if (cur.compare) exitCompare(true);
    if (cur.slice) exitSlice();
    var v = ensureViewerB();
    if (!v) { ui().toast('三维不可用，不能并排。', {kind: 'error'}); return; }
    cur.split = {field: spec.field, window: spec.window || 'adaptive', link: spec.link !== false};
    S.els.vpB.root.hidden = false; S.els.grid.classList.add('split');
    resizeViewers();
    var myCur = cur;
    Promise.resolve(v.setResult(cur.result)).then(function () {
      if (S.cur !== myCur || !myCur.split) return;
      setSplitField(myCur.split.field, myCur.split.window);
      applyLighting(); setLayers();
      if (myCur.split.link && ns.viewer.link && !S.unlink) { try { S.unlink = ns.viewer.link(S.viewerA, S.viewerB); } catch (_) {} }
      try { var st = S.viewerA.getState(); S.viewerB.applyState(Object.assign({}, st, {field: myCur.split.field}), {animate: false}); } catch (_) { safeFit(S.viewerB); }
      renderStatusLine(); renderToolbar();
    });
    renderStatusLine();
  }
  function setSplitField(fid, win) {
    var cur = S.cur;
    if (!cur || !cur.split || !S.viewerB) return;
    cur.split.field = fid; cur.split.window = win || 'adaptive';
    try { S.viewerB.setField(fid, scaleSpec(cur.split.window)); } catch (_) {}
    updateColorbar('b'); renderStatusLine(); saveViewSoon();
  }
  function exitSplit(silent) {
    var cur = S.cur;
    if (!cur || !cur.split) return;
    cur.split = null;
    disposeViewerB();
    if (!silent) { renderStatusLine(); saveViewSoon(); }
  }
  function openCompare() {
    if (!ns.compare || !S.cur) return;
    ns.compare.pickDialog({jobs: S.jobs, current: Object.assign({}, S.cur.job || {}, {input_sha256: S.cur.manifest.job.input_sha256, patient_id: S.cur.manifest.job.patient_id}),
      cards: S.cards, onPick: function (id) { replaceRoute({v: 'compare', cmp: id, q: null}); enterCompare(id); }});
  }
  function enterCompare(otherId) {
    var cur = S.cur;
    if (!cur || !cur.result || !ns.compare) return;
    if (cur.split) exitSplit(true);
    if (cur.slice) exitSlice();
    var n = ++S.cmpSeq;
    var jobB;
    api().job(otherId).then(function (r) {
      jobB = r.job || r;
      if (!jobB || jobB.status !== 'done') throw new api().ApiError('要比较的结果还没有完成。', 409);
      return ns.data.loadResult(ns.data.createOnlineSource(otherId, {fetch: api().dataFetch}), {});
    }).then(function (resB) {
      if (S.cur !== cur || n !== S.cmpSeq) return;
      var mB = resB.manifest;
      if (!mB || !mB.job || mB.job.id !== otherId) throw new api().ApiError('比较结果的数据与所选任务不一致。', 0);
      var fidB = fieldById(mB, cur.field) ? cur.field : defaultField(mB);
      if (!fieldById(mB, cur.field) && fieldById(mB, 'wss') && fieldById(cur.manifest, 'wss')) {
        var before = fieldName(cur.manifest, cur.field);
        fidB = 'wss'; cur.field = 'wss'; cur.window = 'adaptive';
        replaceRoute({f: 'wss'});
        ui().toast('另一个结果没有 ' + before + '，两侧都切到共有的 WSS。', {kind: 'info', ms: 5000});
      }
      var rows = ns.compare.conditions(cur.manifest, mB, cur.field, fidB);
      var allowed = ns.compare.sameScaleAllowed(rows).ok;
      var sameGeo = ns.compare.sameGeometry(cur.manifest, mB) === true;
      cur.compare = {jobId: otherId, job: jobB, result: resB, manifest: mB, field: fidB, window: 'adaptive', mode: allowed ? 'same' : 'each', sync: sameGeo};
      var rg0 = allowed ? ns.compare.commonRange(cur.manifest, mB, cur.field, fidB) : null;
      cur.compare.rangeText = rg0 ? ui().range(rg0[0], rg0[1], (fieldById(cur.manifest, cur.field) || {}).units) : '';
      var v = ensureViewerB();
      S.els.vpB.root.hidden = false; S.els.grid.classList.add('split');
      resizeViewers();
      S.tab = 'compare';
      renderToolbar(); renderInspector(); renderStatusLine();
      if (!v) return;
      return Promise.resolve(v.setResult(resB)).then(function () {
        if (S.cur !== cur || !cur.compare || n !== S.cmpSeq) return;
        applyLighting(); setLayers();
        applyCompareScale(); applySync();
        safeFit(v);
        if (cur.compare.sync) { try { var st = S.viewerA.getState(); v.applyState(Object.assign({}, st, {field: cur.compare.field}), {animate: false}); } catch (_) {} }
      });
    }).catch(function (e) {
      if (S.cur !== cur || n !== S.cmpSeq) return;
      ui().toast('不能比较：' + (e && e.message || e), {kind: 'error'});
      replaceRoute({v: null, cmp: null});
    });
  }
  function applyCompareScale() {
    var cur = S.cur;
    if (!cur || !cur.compare) return;
    var c = cur.compare;
    var specA = scaleSpec(cur.window), specB = scaleSpec(c.window);
    if (c.mode === 'same') {
      var rg = ns.compare.commonRange(cur.manifest, c.manifest, cur.field, c.field);
      if (rg) { specA = scaleSpec({range: rg}); specB = scaleSpec({range: rg}); c.rangeText = ui().range(rg[0], rg[1], (fieldById(cur.manifest, cur.field) || {}).units); }
      else { c.mode = 'each'; c.rangeText = ''; }
    }
    if (S.viewerA) { try { S.viewerA.setField(cur.field, specA); } catch (_) {} }
    if (S.viewerB) { try { S.viewerB.setField(c.field, specB); } catch (_) {} }
    updateColorbar('a'); updateColorbar('b');
    S.els.grid.classList.toggle('cmp-each', c.mode === 'each');
    renderToolbar(); renderStatusLine();
  }
  function applySync() {
    var cur = S.cur;
    if (S.unlink) { try { S.unlink(); } catch (_) {} S.unlink = null; }
    if (cur && cur.compare && cur.compare.sync && S.viewerA && S.viewerB && ns.viewer.link) { try { S.unlink = ns.viewer.link(S.viewerA, S.viewerB); } catch (_) {} }
  }
  function renderComparePanel(body) {
    var cur = S.cur, c = cur.compare;
    ns.compare.render(body, {
      left: {manifest: cur.manifest, field: cur.field, name: hideName() ? '病例' : cur.manifest.job.display_name, result: cur.manifest.result.display_name},
      right: {manifest: c.manifest, field: c.field, name: c.manifest.job.display_name, result: c.manifest.result.display_name},
      mode: c.mode, sync: c.sync, rangeText: c.rangeText,
      onMode: function (mode) { c.mode = mode; applyCompareScale(); renderInspector(); },
      onSync: function (on) { c.sync = on; applySync(); renderInspector(); },
      onSwap: function () { var other = c.jobId; var mine = cur.jobId; go(other, {v: 'compare', cmp: mine}); },
      onClose: function () { exitCompare(); }});
  }
  function exitCompare(silent) {
    var cur = S.cur;
    if (!cur || !cur.compare) return;
    cur.compare = null;
    S.cmpSeq += 1;
    disposeViewerB();
    S.els.grid.classList.remove('cmp-each');
    if (!silent) {
      if (S.tab === 'compare') S.tab = 'overview';
      if (S.viewerA) { try { S.viewerA.setField(cur.field, scaleSpec(cur.window)); } catch (_) {} updateColorbar('a'); }
      replaceRoute({v: null, cmp: null});
      renderToolbar(); renderInspector(); renderStatusLine();
    }
  }

  // ------------------------------------------------------------------ questions D8
  function questionShell() {
    return {
      snapshot: function () {
        var cur = S.cur;
        var st = null; try { st = S.viewerA ? S.viewerA.getState() : null; } catch (_) {}
        return {field: cur.field, window: cur.window, split: cur.split ? {field: cur.split.field, window: cur.split.window, link: cur.split.link} : null, viewer: st};
      },
      restore: function (s) {
        if (!s) return;
        if (s.split) setSplit(s.split); else exitSplit();
        applyField(s.field, s.window);
        showFindings([]);
        if (s.viewer && S.viewerA) { try { S.viewerA.applyState(s.viewer, {animate: false}); } catch (_) {} }
      },
      setField: function (id, win) { applyField(id, win); },
      setSplit: function (spec) { if (spec) setSplit(spec); else exitSplit(); },
      showFindings: showFindings,
      openCompare: openCompare
    };
  }
  function askQuestion(id) {
    var cur = S.cur;
    if (!cur || !cur.manifest || !ns.questions) return;
    if (S.question) { ns.questions.undo(S.question, questionShell()); S.question = null; }
    var active = ns.questions.apply(id, questionShell(), cur.manifest);
    S.question = active;
    replaceRoute({q: active ? id : null});
    ns.questions.bar(S.els.qbar, active, {
      onUndo: function () { ns.questions.undo(S.question, questionShell()); S.question = null; ns.questions.bar(S.els.qbar, null, {}); replaceRoute({q: null}); },
      onKeep: function () { S.question = null; ns.questions.bar(S.els.qbar, null, {}); replaceRoute({q: null}); }
    });
    resizeViewers();
  }

  // ------------------------------------------------------------------ tools tab, upload, export
  function renderTools(body) {
    var cur = S.cur;
    var full = store().prefs().tier === 'full';
    var jobId = cur.jobId;
    var parts = [];
    parts.push(ui().section('经典报告里的工具', {}, ui().note('这些工具还没有搬到新工作区：' + CLASSIC_TOOLS + '。'),
      h('div', {'class': 'sec-actions'}, ui().button('在经典报告中打开', function () { root.open(api().urls.report(jobId), '_blank', 'noopener'); }, {icon: 'external'}))));
    parts.push(ui().section('导出', {}, ui().note('汇报图、复核数据、离线报告、一页纸。'), h('div', {'class': 'sec-actions'}, ui().button('导出…', openExport, {icon: 'download'}))));
    if (full) {
      var m = cur.manifest, pv = m.provenance || {}, mp = m.mapping || {}, di = mp.display_interpolation || {};
      var rows = [
        ['发布包', m.result && m.result.release_id], ['结果身份', m.result && m.result.run_identity], ['数据版本', m.data_version], ['分析版本', m.result && m.result.analysis_version],
        ['部署版本', m.result && m.result.deploy_version], ['显示插值', di.method ? di.method + (di.sigma_mm !== undefined ? '，σ ' + di.sigma_mm + ' mm' : '') + (di.max_dist_mm !== undefined ? '，最大距离 ' + di.max_dist_mm + ' mm' : '') : null],
        ['方向来源', m.frame && m.frame.direction_source], ['发布包哈希', pv.release_hash], ['代码', pv.git_describe], ['输入 SHA256', m.job && m.job.input_sha256]];
      parts.push(ui().section('技术信息', {}, ui().table([{key: 'k', label: ''}, {key: 'v', label: '', render: function (r) { return /\s/.test(r.v) ? r.v : h('span', {'class': 'mono', text: r.v}); }}],
        rows.filter(function (r) { return r[1]; }).map(function (r) { return {k: r[0], v: String(r[1])}; }), {cls: 'tbl-kv tbl-nohead tbl-tech'}),
        h('div', {'class': 'sec-actions'}, ui().link('完整统计 JSON', api().urls.file(jobId, 'summary.json'), {newTab: true}), ui().link('运行清单', api().urls.file(jobId, 'run_manifest.json'), {newTab: true}))));
      var rel = (S.releases || []).filter(function (r) { return (r.id || r.release) !== (m.result && m.result.release_id); });
      if (rel.length) {
        var sel = ui().select(rel.map(function (r) { var id = r.id || r.release; var card = S.cards[id]; return {value: id, label: (card && card.display_name) || ui().resultName({model_release: r}, S.cards)}; }), null, null, {'aria-label': '换一个模型'});
        parts.push(ui().section('换模型重跑', {}, ui().note('沿用这份输入的中心线和出口确认，只重新预测。'), h('div', {'class': 'sec-actions'}, sel, ui().button('开始', function () {
          api().rerun(jobId, {version: cur.job && cur.job.version, release_id: sel.value}).then(function (r) { var j = r.job || r; ui().toast('已建立新任务。', {kind: 'ok'}); scheduleRefresh(); if (j && j.id) go(j.id); }, function (e) { conflictOr(e, '没有建立新任务'); });
        }, {cls: 'btn-sm'}))));
      }
      parts.push(ui().section('病例信息', {}, ui().note('改病例名称、患者编号、扫描日期；不影响计算。'), h('div', {'class': 'sec-actions'}, ui().button('编辑信息…', metadataDialog, {cls: 'btn-sm'}))));
    } else {
      parts.push(ui().note('换模型重跑、技术信息和完整统计在「完整」档。'));
    }
    parts = parts.concat(extCall('tools'));
    ui().fill(body, parts);
  }
  function metadataDialog() {
    var cur = S.cur;
    var j = cur.job || {};
    var mk = function (label, key, type) { var input = h('input', {type: type || 'text', value: j[key] || '', 'aria-label': label}); return {key: key, input: input, el: h('label', {'class': 'fld'}, h('span', {text: label}), input)}; };
    var fields = [mk('病例名称', 'case_id'), mk('患者编号', 'patient_id'), mk('扫描标签', 'scan_label'), mk('扫描日期', 'scan_date', 'date')];
    ui().dialog.open({title: '编辑病例信息', body: fields.map(function (f) { return f.el; }), actions: [ui().button('取消', function () { ui().dialog.close('cancel'); }), ui().button('保存', function () {
      var payload = {version: j.version};
      fields.forEach(function (f) { payload[f.key] = f.input.value.trim(); });
      api().metadata(cur.jobId, payload).then(function () { ui().dialog.close('done'); ui().toast('病例信息已保存。', {kind: 'ok'}); scheduleRefresh(); reloadCurrent(); }, function (e) { ui().dialog.close('cancel'); conflictOr(e, '没有保存'); });
    }, {kind: 'primary'})]});
  }
  function openUpload() {
    if (!ns.upload || S.offline) return;
    ns.upload.open({releases: S.releases, cards: S.cards, onCreated: function (id) { scheduleRefresh(); go(id); }, onOpen: function (id) { go(id); }});
  }
  function openExport() {
    var cur = S.cur;
    if (!cur || !cur.manifest || !ns.exporter) return;
    var f = fieldById(cur.manifest, cur.field);
    ns.exporter.open({job: cur.job, manifest: cur.manifest, viewer: S.viewerA, info: function () { return S.viewerA && S.viewerA.colorbarInfo ? S.viewerA.colorbarInfo() : null; },
      fieldId: cur.field, fieldLabel: fieldName(cur.manifest, cur.field), units: f && f.units, windowLabel: windowLabel(f, cur.window),
      displayName: cur.manifest.job.display_name, hideName: hideName(), offline: S.offline, full: store().prefs().tier === 'full',
      bookmarks: ns.bookmarks ? ns.bookmarks.list(cur.runIdentity) : [],
      viewState: function () { var st = null; try { st = S.viewerA.getState(); } catch (_) {} return {run_identity: cur.runIdentity, field: cur.field, window: cur.window, viewer: st}; }});
  }
  function toggleRail(open) {
    var narrow = root.matchMedia ? root.matchMedia('(max-width: 760px)').matches : false;
    if (narrow) S.railOpenNarrow = open; else store().setPrefs({rail: open});
    applyPanels(); renderToolbar();
  }
  function logout() {
    api().logout().then(function () { root.location.reload(); }, function () { root.location.reload(); });
  }

  // ------------------------------------------------------------------ keyboard (§6.5)
  function onKey(e) {
    if (!S || !e) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var key = e.key;
    if (key === 'Escape') {
      if (ui().menuOpen()) { ui().closeMenu(); return; }
      if (ui().dialog.isOpen()) return;          // the <dialog> handles its own Esc
      if (S.pickOnce) { cancelPickOnce(); return; }
      if (S.cur && S.cur.measure && S.cur.measure.cancelPending()) { if (currentTab() === 'measure') renderInspector(); return; }
      if (S.cur && S.cur.region && S.cur.region.picking()) { S.cur.region.setPicking(false); return; }
      if (S.cur && S.cur.slice) { if (S.cur.slice.picking()) S.cur.slice.setPicking(false); else exitSlice(); return; }
      if (S.cur && S.cur.cursor) { toggleCursor(); return; }
      if (S.question) { S.question = null; ns.questions.bar(S.els.qbar, null, {}); return; }
      if (S.cur && S.cur.compare) { exitCompare(); return; }
      if (S.cur && S.cur.split) { exitSplit(); return; }
      return;
    }
    if (ui().dialog.isOpen() || ui().isTyping(e.target)) return;
    if (extHandled('key', e)) { if (e.preventDefault) e.preventDefault(); return; }
    var done = true;
    var cur = S.cur;
    if (cur && cur.slice && cur.slice.key(e)) { if (e.preventDefault) e.preventDefault(); return; }
    if (key === 'm' || key === 'M') {
      if (!cur || !cur.result || !ns.measure || !toolSupported('measure')) return;
      toggleMeasure();
    } else if (key === 's' || key === 'S') {
      if (!cur || !cur.result || !ns.slice || !ns.slice.supported(cur.result)) return;
      toggleSlice();
    } else if (key === 'j' || key === 'J' || key === 'k' || key === 'K') {
      if (!S.rail) return;
      var order = S.rail.order();
      if (!order.length) return;
      var i = cur ? order.indexOf(cur.jobId) : -1;
      var next = key.toLowerCase() === 'j' ? order[Math.min(order.length - 1, i + 1)] : order[Math.max(0, i < 0 ? 0 : i - 1)];
      if (next && (!cur || next !== cur.jobId)) go(next);
    } else if (/^[1-9]$/.test(key)) {
      if (!cur || !cur.manifest) return;
      var fl = visibleFields(cur.manifest)[Number(key) - 1];
      if (!fl) return;
      applyField(fl.id, 'adaptive');
    } else if (key === 'ArrowLeft' || key === 'ArrowRight') {
      if (!cur || !cur.cursor) return;
      stepCursor((key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 5 : 1));
    } else if (key === '[' || key === ']') {
      if (!cur || !cur.manifest) return;
      stepFinding(key === ']' ? 1 : -1);
    } else if (key === 'l' || key === 'L') {
      if (!cur || !cur.manifest) return;
      toggleLighting();
    } else if (key === 'b' || key === 'B') {
      if (!cur || !cur.manifest) return;
      saveBookmark();
    } else if (key === 'g' || key === 'G') {
      if (!cur || !cur.manifest) return;
      toggleCursor();
    } else if (key === '?') {
      shortcutsDialog();
    } else done = false;
    if (done && e.preventDefault) e.preventDefault();
  }

  // ------------------------------------------------------------------ start
  function start(opts) {
    opts = opts || {};
    S = newState(opts);
    store().state.session = S.session;
    var appEl = opts.app || ui().host('ws-app', 'div', 'ws-app');
    buildLayout(appEl);
    renderTop();
    if (ns.rail) {
      S.rail = ns.rail.create(S.els.rail, {onOpen: function (id) { if (root.matchMedia && root.matchMedia('(max-width: 760px)').matches) { S.railOpenNarrow = false; applyPanels(); } go(id); }, onCollapse: function () { toggleRail(false); }});
      S.rail.setFull(store().prefs().tier === 'full');
    }
    root.document.addEventListener('keydown', onKey);
    root.addEventListener('hashchange', route);
    root.addEventListener('resize', function () { applyPanels(); });
    root.document.addEventListener('click', function (ev) {
      if (!ui().menuOpen()) return;
      var t = ev && ev.target;
      var menuEl = root.document.getElementById('ws-menu');
      if (t && menuEl && typeof menuEl.contains === 'function' && menuEl.contains(t)) return;
      if (t && typeof t.closest === 'function' && t.closest('[aria-haspopup="menu"]')) return;
      ui().closeMenu();
    });
    var cardsP = api().modelCards().then(function (r) { S.cards = (r && r.cards) || {}; store().state.cards = S.cards; }, function () { S.cards = {}; });
    var relP = api().releases().then(function (r) { S.releases = Array.isArray(r) ? r : (r.releases || []); }, function () { S.releases = []; });
    S.closeEvents = api().events(function (ev) {
      scheduleRefresh();
      if (S.cur && ev && ev.job_id === S.cur.jobId && !S.cur.result && ev.status === 'done') openJob(S.cur.jobId, {});
    });
    return Promise.all([cardsP, relP]).then(function () {
      if (S.rail) S.rail.setCards(S.cards);
      return refreshJobs();
    }).then(function () { route(); return S; });
  }
  function startOffline(opts) {
    opts = opts || {};
    S = newState({offline: true, offlineMeta: opts.offlineMeta || null});
    store().state.offline = true;
    var appEl = opts.app || ui().host('ws-app', 'div', 'ws-app');
    buildLayout(appEl);
    renderTop();
    root.document.addEventListener('keydown', onKey);
    root.addEventListener('resize', function () { applyPanels(); });
    var seq = store().nextSeq();
    S.cur = {seq: seq, jobId: null, job: null, result: null, manifest: null};
    showMode('result');
    stageMessage('正在读取离线数据…');
    if (!ns.data) { stageMessage('离线报告不完整：缺少查看器。', true); return Promise.resolve(S); }
    var source = ns.data.createEmbeddedSource(root.document);
    return Promise.resolve(ns.data.loadResult(source, {})).then(function (result) {
      var m = result.manifest;
      S.cur.jobId = m && m.job && m.job.id;
      S.cur.job = m && m.job ? {id: m.job.id, status: m.job.status, review: m.job.review, patient_id: m.job.patient_id} : null;
      var rid = result.runIdentity || (m.result && m.result.run_identity);
      if (ns.bookmarks && S.offlineMeta && Array.isArray(S.offlineMeta.bookmarks)) ns.bookmarks.seed(rid, S.offlineMeta.bookmarks);
      return showResult(result, seq, null);
    }).then(function () { return S; }, function (e) { stageMessage('离线数据读取失败：' + (e && e.message || e), true); return S; });
  }
  function showLogin(session, onDone) {
    var E = S ? S.els : null;
    var app = ui().host('ws-app', 'div', 'ws-app');
    var mode = session && session.login === 'password' ? 'password' : 'token';
    var user = h('input', {type: 'text', autocomplete: 'username', 'aria-label': '用户名', id: 'wsl-user'});
    var pass = h('input', {type: 'password', autocomplete: mode === 'password' ? 'current-password' : 'off', 'aria-label': mode === 'password' ? '口令' : '访问令牌', id: 'wsl-pass'});
    try { var remembered = root.localStorage && root.localStorage.getItem('wss-login-user'); if (remembered) user.value = remembered; } catch (_) {}
    var err = h('p', {'class': 'login-err', role: 'alert', hidden: true});
    var submit = h('button', {type: 'submit', 'class': 'btn btn-primary', text: '登录'});
    var form = h('form', {'class': 'login-form'},
      h('div', {'class': 'login-mark'}, h('span', {'class': 'mark', text: 'WSS'}), h('span', {'class': 'muted', text: '血流场预测 · 工作区'})),
      mode === 'password' ? h('label', {'class': 'fld'}, h('span', {text: '用户名'}), user) : null,
      h('label', {'class': 'fld'}, h('span', {text: mode === 'password' ? '口令' : '访问令牌'}), pass),
      err, submit,
      h('p', {'class': 'muted login-foot'}, '也可以回到 ', h('a', {href: '/', text: '经典工作台'}), '。'));
    form.addEventListener('submit', function (ev) {
      if (ev && ev.preventDefault) ev.preventDefault();
      submit.disabled = true; err.hidden = true;
      var body = mode === 'password' ? {username: user.value.trim(), password: pass.value} : {token: pass.value};
      api().login(body).then(function (s) {
        pass.value = '';
        try { if (mode === 'password' && root.localStorage) root.localStorage.setItem('wss-login-user', user.value.trim()); } catch (_) {}
        if (onDone) onDone(s);
      }, function (e) {
        submit.disabled = false; err.hidden = false;
        err.textContent = e.status === 401 ? '用户名或口令不正确（区分大小写）。' : e.status === 429 ? '尝试过于频繁，请一分钟后再试。' : e.message;
      });
    });
    var box = h('section', {'class': 'ws-login'}, form);
    app.className = 'ws-app is-login';
    app.replaceChildren(box);
    try { (mode === 'password' && !user.value ? user : pass).focus(); } catch (_) {}
    return {form: form, user: user, pass: pass, err: err, submit: submit, el: E};
  }

  function state() { return S; }
  return {start: start, startOffline: startOffline, showLogin: showLogin, parseHash: parseHash, buildHash: buildHash, defaultField: defaultField,
    visibleFields: visibleFields, windowOptions: windowOptions, windowLabel: windowLabel, state: state, route: route, go: go, onKey: onKey};
});
