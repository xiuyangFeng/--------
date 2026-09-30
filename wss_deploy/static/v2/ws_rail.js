/* WSS workspace v2 — case rail and the to-do home (contract §6.3 ws_rail.js).
 * Three layers: case (patient) → scan (one input geometry) → result (one release run).  Names, scans and result
 * titles come from the job records and model cards; no release code is shown.
 * Lane C (S6c): the home carries a head with the four home pages (病例 / 任务 / 队列 / 回收站, the last three live in
 * ws_admin.js), four counts (今日 / 进行中 / 待处理 / 失败), 待复核 in 「需要处理」, unread marks and, in the
 * administrator's all-users view, the owner of each case.
 * Phase 3 lane 2 (工作台): live remaining time on running rows (classic etaRowSummary, one shared 1 s ticker), the
 * 「最近完成」 list and the 「三步上手」 guide on the home, and on each case card the lumen diameter, the tags and
 * 「补跑缺少的结果」 (the releases this scan has no finished result of, rerun from its confirmed outlets). */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.rail = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };

  var FILTERS = [
    {value: '', label: '全部状态'},
    {value: 'confirm', label: '待确认'},
    {value: 'running', label: '计算中'},
    {value: 'failed', label: '失败'},
    {value: 'unreviewed', label: '未复核'},
    {value: 'reviewed', label: '已复核'}
  ];
  function bucket(job) {
    var s = job && job.status || '';
    if (s === 'awaiting_confirmation' || s === 'awaiting_input' || s === 'awaiting_outlets') return 'confirm';
    if (/^(queued|running|cancelling)/.test(s)) return 'running';
    if (s === 'failed' || s === 'interrupted') return 'failed';
    if (s === 'cancelled') return 'cancelled';
    if (s === 'done') return job.review && job.review.status === 'reviewed' ? 'reviewed' : 'unreviewed';
    return 'other';
  }
  function created(job) { var t = Date.parse(job && (job.created_at || job.updated_at) || ''); return isNaN(t) ? 0 : t; }
  function caseKey(job) {
    var owner = job.owner_name !== undefined ? 'o:' + (job.owner_name || '') + '|' : '';   // all-users view: one card per owner
    if (job.patient_id) return owner + 'p:' + String(job.patient_id).trim();
    return owner + 'c:' + ui().displayName(job);
  }
  function unread(id) { return Boolean(ns.admin && ns.admin.isUnread && ns.admin.isUnread(id)); }
  function scanKey(job) { return job.input_sha256 ? 's:' + job.input_sha256 : 'j:' + job.id; }
  function scanLabel(job) {
    var parts = [];
    if (job.scan_date) parts.push(job.scan_date);
    else if (job.created_at) parts.push(ui().time(job.created_at, false));
    if (job.scan_label) parts.push(job.scan_label);
    return parts.join(' · ');
  }
  function matches(job, q, cards) {
    if (!q) return true;
    var needle = String(q).trim().toLowerCase();
    if (!needle) return true;
    var hay = [ui().displayName(job), job.case_id, job.patient_id, job.scan_label, job.scan_date, job.id, ui().resultName(job, cards)]
      .concat(job.tags || []).filter(Boolean).join(' ').toLowerCase();
    return hay.indexOf(needle) >= 0;
  }
  // Pure: jobs → [{key, name, patientId, latest, scans:[{key, label, latest, jobs:[…]}]}]
  function model(jobs, opts) {
    opts = opts || {};
    var cards = opts.cards || {};
    var cases = {}, order = [];
    (jobs || []).forEach(function (job) {
      if (!job || !job.id) return;
      if (opts.status && bucket(job) !== opts.status) return;
      if (!matches(job, opts.q, cards)) return;
      var ck = caseKey(job);
      var c = cases[ck];
      if (!c) { c = cases[ck] = {key: ck, name: ui().displayName(job), patientId: job.patient_id || '', owner: job.owner_name, latest: 0, scans: {}, scanOrder: []}; order.push(ck); }
      var sk = scanKey(job);
      var s = c.scans[sk];
      if (!s) { s = c.scans[sk] = {key: sk, label: scanLabel(job), latest: 0, jobs: []}; c.scanOrder.push(sk); }
      s.jobs.push(job);
      var t = created(job);
      s.latest = Math.max(s.latest, t); c.latest = Math.max(c.latest, t);
      if (!s.label && scanLabel(job)) s.label = scanLabel(job);
    });
    return order.map(function (k) { return cases[k]; }).sort(function (a, b) { return b.latest - a.latest; }).map(function (c) {
      var scans = c.scanOrder.map(function (k) { return c.scans[k]; }).sort(function (a, b) { return b.latest - a.latest; });
      scans.forEach(function (s) {
        s.jobs.sort(function (a, b) { return created(b) - created(a); });
        // Scan without a date or label: the day it first came in.
        if (!s.jobs.some(function (j) { return j.scan_date || j.scan_label; })) {
          var first = s.jobs[s.jobs.length - 1];
          s.label = first && first.created_at ? '首次上传 ' + ui().time(first.created_at, false) : s.label;
        }
      });
      return {key: c.key, name: c.name, patientId: c.patientId, owner: c.owner, latest: c.latest, scans: scans};
    });
  }
  function order(tree) {
    var ids = [];
    (tree || []).forEach(function (c) { c.scans.forEach(function (s) { s.jobs.forEach(function (j) { ids.push(j.id); }); }); });
    return ids;
  }
  function rowStatus(job) {
    var b = bucket(job);
    if (b === 'reviewed') return {tone: 'ok', label: '已复核'};
    if (b === 'unreviewed') return {tone: 'ok', label: '已完成'};
    var s = ui().statusInfo(job.status);
    return {tone: s.tone, label: s.label};
  }

  // ------------------------------------------------------------------ P3 lane 2: remaining time (classic §21.2 / §21.3)
  // Same rules as workbench_core formatDuration / stageRemaining / etaView / etaRowSummary (ported: the classic file
  // leaves with S7).  Snapshots carry ``eta``; between list updates the running stage advances on the browser clock.
  function numOrNull(v) {
    if (v === null || v === undefined || v === '' || typeof v === 'boolean') return null;
    var n = Number(v);
    return isFinite(n) ? n : null;
  }
  function isObj(v) { return v !== null && typeof v === 'object' && !Array.isArray(v); }
  function formatDuration(seconds) {
    var s = numOrNull(seconds);
    if (s === null) return '';
    if (s < 10) return '几秒';
    if (s < 57.5) return '约 ' + Math.max(10, Math.round(s / 5) * 5) + ' 秒';
    if (s >= 570) return '约 ' + Math.round(s / 60) + ' 分钟';
    var total = Math.round(s / 5) * 5, minutes = Math.floor(total / 60), rest = total % 60;
    return rest ? '约 ' + minutes + ' 分 ' + rest + ' 秒' : '约 ' + minutes + ' 分钟';
  }
  function stageRemaining(expected, elapsed) {
    var e = Math.max(0, numOrNull(expected) || 0), t = Math.max(0, numOrNull(elapsed) || 0), knee = 0.8 * e;
    if (t <= knee || e <= 0) return Math.max(0, e - t);
    return 0.2 * e * (knee / t);
  }
  function etaView(eta, opts) {
    if (!isObj(eta) || !Array.isArray(eta.stages) || !eta.stages.length) return null;
    var age = Math.max(0, numOrNull(isObj(opts) ? opts.ageS : null) || 0);
    var minPct = isObj(opts) && numOrNull(opts.minPct) !== null ? numOrNull(opts.minPct) : 2.5;
    var pre = {};
    (Array.isArray(eta.precomputed) ? eta.precomputed : []).forEach(function (k) { pre[String(k)] = true; });
    var rows = eta.stages.filter(isObj).map(function (stage) {
      var expected = Math.max(0, numOrNull(stage.expected_s) || 0);
      var state = ['done', 'running', 'pending'].indexOf(stage.state) >= 0 ? stage.state : 'pending';
      var elapsed = numOrNull(stage.elapsed_s);
      if (state === 'running') elapsed = (elapsed || 0) + age;
      return {key: String(stage.key || ''), label: String(stage.label || stage.key || '阶段'), expected: expected, elapsed: elapsed, state: state,
        overdue: state === 'running' && elapsed > expected && expected > 0, precomputed: Boolean(pre[String(stage.key || '')])};
    });
    var rawStatus = eta.status || (rows.some(function (r) { return r.state === 'running'; }) ? 'running' : null);
    var waiting = eta.waiting === true && rawStatus === 'running';
    var status = waiting ? 'queued' : rawStatus;
    var total = rows.reduce(function (s, r) { return s + r.expected; }, 0);
    var widths = rows.map(function (r) { return total > 0 ? r.expected / total * 100 : 100 / rows.length; }).map(function (w) { return Math.max(minPct, w); });
    var scale = 100 / widths.reduce(function (s, w) { return s + w; }, 0);
    var filled = 0, currentIndex = -1;
    var segments = rows.map(function (r, i) {
      var width = +(widths[i] * scale).toFixed(3), fill = r.state === 'done' ? 100 : 0;
      if (r.state === 'running' && !waiting) { currentIndex = i; fill = r.expected > 0 ? Math.min(96, r.elapsed / r.expected * 100) : 50; }
      filled += width * fill / 100;
      return Object.assign({}, r, {state: waiting && r.state === 'running' ? 'pending' : r.state, width: width, fill: +fill.toFixed(1)});
    });
    var remaining;
    if (status === 'running' && !waiting) remaining = rows.reduce(function (s, r) { return s + (r.state === 'running' ? stageRemaining(r.expected, r.elapsed) : r.state === 'pending' ? r.expected : 0); }, 0);
    else remaining = numOrNull(eta.remaining_s);
    var segmentRemaining = numOrNull(eta.segment_remaining_s), queueAhead = numOrNull(eta.queue_ahead), queueWait = numOrNull(eta.queue_ahead_s);
    var headline = '', manual = eta.segment === 'A' ? '，不含人工确认出口' : '';
    if (status === 'queued') {
      var wait = queueWait === null ? null : Math.max(0, queueWait - age);
      headline = queueAhead ? '前面 ' + queueAhead + ' 个，' + (wait === null ? '稍后' : formatDuration(wait) + '后') + '开始' : waiting ? '等前一个任务算完后开始' : '即将开始';
    } else if (status === 'running') {
      var over = currentIndex >= 0 && segments[currentIndex].overdue;
      headline = over && remaining < 10 ? '即将完成' : '预计还需' + formatDuration(remaining) + (manual ? '（' + manual.slice(1) + '）' : '');
    } else if (status === 'awaiting_confirmation') {
      headline = remaining !== null ? '确认后' + formatDuration(remaining) + '出结果' : '';
    } else if (status === 'awaiting_input') {
      var a = segmentRemaining !== null ? segmentRemaining : remaining;
      headline = a !== null ? '确认后' + formatDuration(a) + '完成中心线提取' : '';
    }
    return {status: status, waiting: waiting, segments: segments, currentIndex: currentIndex, remaining: remaining, progress: Math.round(Math.min(100, filled)),
      headline: headline, queueAhead: queueAhead, queueWait: queueWait};
  }
  // One short line for a list row: a thin bar (running only) and the remaining time.
  function etaRowSummary(eta, opts) {
    var view = etaView(eta, opts);
    if (!view) return null;
    if (view.status === 'running') return {pct: view.progress, text: view.headline === '即将完成' ? '即将完成' : '还需' + formatDuration(view.remaining), state: 'running'};
    if (view.status === 'queued') return {pct: null, text: view.queueAhead ? '前面 ' + view.queueAhead + ' 个' : isObj(eta) && eta.waiting ? '等待前一个任务' : '即将开始', state: 'queued'};
    if (view.status === 'awaiting_confirmation' && view.remaining !== null) return {pct: null, text: '确认后' + formatDuration(view.remaining), state: 'waiting'};
    return null;
  }
  // Live readouts share one 1 s ticker; a readout whose element left the page drops out.
  var etaEntries = [], etaTimer = null;
  function etaTick() {
    etaEntries = etaEntries.filter(function (e) {
      if (e.host.isConnected === false) return false;
      try { e.paint(); return true; } catch (_) { return false; }
    });
    if (!etaEntries.length && etaTimer) { clearInterval(etaTimer); etaTimer = null; }
  }
  function etaTrack(entry) {
    etaEntries.push(entry);
    if (!etaTimer && typeof setInterval === 'function') { etaTimer = setInterval(etaTick, 1000); if (etaTimer && etaTimer.unref) etaTimer.unref(); }
  }
  // The job's remaining time as a small live readout (text, and a thin bar while running); null when the job has no
  // estimate to show (a finished job, an older service, an input check).  ``_etaAt`` is stamped when the list arrives.
  function etaBadge(job, opts) {
    if (!job || !isObj(job.eta)) return null;
    opts = opts || {};
    var h = ui().h, at = Date.now(), extra = opts.cls ? ' ' + opts.cls : '';
    var text = h('span', {'class': 'eta-txt'}), fill = h('span', {'class': 'eta-fill'});
    var bar = opts.bar === false ? null : h('span', {'class': 'eta-thin', 'aria-hidden': 'true'}, fill);
    var el = h('span', {'class': 'eta-live' + extra}, bar, text);
    var entry = {host: el, paint: function () {
      var s = etaRowSummary(job.eta, {ageS: Math.max(0, (Date.now() - (job._etaAt || at)) / 1000)});
      el.hidden = !s;
      if (!s) return;
      if (text.textContent !== s.text) text.textContent = s.text;
      el.className = 'eta-live eta-' + s.state + extra;
      if (bar) { bar.hidden = s.pct === null; if (s.pct !== null) fill.style.width = s.pct + '%'; }
    }};
    entry.paint();
    if (el.hidden) return null;
    etaTrack(entry);
    return {el: el, text: text};
  }

  // ------------------------------------------------------------------ P3 lane 2: case facts (classic caseCard)
  var FINAL = ['done', 'failed', 'cancelled', 'interrupted'];
  // Pure: the releases a scan has no finished result of, and the job to rerun them from — the newest finished job of
  // the scan whose centreline and confirmed outlets can be reused (classic /api/cases missing_releases +
  // reusable_job_id, with the rerun rule of the service: done + confirmed outlets).  A release with a job still on its
  // way (queued, running, waiting for a person) is not offered again.  ``canRun(job)`` limits the source (the
  // administrator's all-users view: only the user's own jobs).
  function missingResults(jobs, releases, opts) {
    opts = opts || {};
    var have = {}, busy = {};
    (jobs || []).forEach(function (j) {
      var rid = j ? ui().releaseIdOf(j) : '';
      if (!rid) return;
      if (j.status === 'done') have[rid] = true;
      else if (FINAL.indexOf(j.status) < 0) busy[rid] = true;
    });
    var source = (jobs || []).filter(function (j) { return j && j.status === 'done' && j.reusable === true && (!opts.canRun || opts.canRun(j)); })[0] || null;
    var missing = (releases || []).filter(function (r) { var id = r && (r.id || r.release); return id && !have[id] && !busy[id]; })
      .map(function (r) { return {releaseId: r.id || r.release, release: r}; });
    return {source: source, missing: source ? missing : [], unavailable: source ? [] : missing};
  }
  // Pure: what a case card adds under its name — the lumen diameter of the newest finished result of the latest scan
  // that has one (classic latestMaxDiameter), and the tags of every job of the case (first seen first).
  function caseFacts(c) {
    var scan = c && c.scans && c.scans[0], diameter = null, tags = [];
    ((scan && scan.jobs) || []).some(function (j) { var v = j.status === 'done' ? numOrNull(j.max_diameter_mm) : null; if (v !== null) { diameter = v; return true; } return false; });
    ((c && c.scans) || []).forEach(function (s) { s.jobs.forEach(function (j) { (j.tags || []).forEach(function (t) { if (t && tags.indexOf(t) < 0) tags.push(t); }); }); });
    return {diameter: diameter, tags: tags};
  }
  // Pure: the newest finished results (classic overviewModel.recent: by finish time, the creation time standing in).
  function finishedMs(job) {
    var list = [job.finished_at, job.created_at, job.updated_at];
    for (var i = 0; i < list.length; i++) { var t = Date.parse(list[i] || ''); if (!isNaN(t)) return t; }
    return 0;
  }
  function recentDone(jobs, n) {
    return (jobs || []).filter(function (j) { return j && j.status === 'done'; })
      .map(function (j, i) { return {j: j, t: finishedMs(j), i: i}; })
      .sort(function (a, b) { return b.t - a.t || a.i - b.i; }).slice(0, n === undefined ? 5 : n).map(function (x) { return x.j; });
  }

  function create(el, opts) {
    opts = opts || {};
    var h = ui().h;
    var st = {jobs: [], q: '', status: '', current: null, cards: {}, full: false};
    var search = h('input', {type: 'search', 'class': 'rail-search', placeholder: '搜索病例、患者编号', 'aria-label': '搜索病例'});
    var filter = ui().select(FILTERS, '', function (v) { st.status = v; render(); }, {'class': 'rail-filter', 'aria-label': '按状态筛选'});
    var list = h('div', {'class': 'rail-list', role: 'navigation', 'aria-label': '病例列表'});
    var foot = h('div', {'class': 'rail-foot'});
    var searchWrap = h('label', {'class': 'rail-search-wrap'}, ui().icon('search'), search);
    var head = h('div', {'class': 'rail-head'}, searchWrap, h('div', {'class': 'rail-head-row'}, filter,
      ui().iconButton('chevron-left', '收起病例栏', function () { if (opts.onCollapse) opts.onCollapse(); }, {cls: 'rail-collapse'})));
    el.replaceChildren(head, list, foot);
    var timer = null;
    search.addEventListener('input', function () { if (timer) clearTimeout(timer); timer = setTimeout(function () { st.q = search.value; render(); }, 120); });

    function render() {
      var tree = model(st.jobs, {q: st.q, status: st.status, cards: st.cards});
      list.replaceChildren();
      if (!tree.length) {
        list.appendChild(ui().empty(st.jobs.length ? '没有符合条件的病例。清空搜索或换一个状态。' : '还没有病例。点顶栏「上传」开始第一例。'));
      }
      tree.forEach(function (c) {
        var box = h('div', {'class': 'rail-case'});
        var title = h('div', {'class': 'rail-case-name'}, h('span', {text: c.name}));
        if (c.patientId && c.patientId !== c.name) title.appendChild(h('span', {'class': 'rail-pid', text: c.patientId}));
        if (c.owner !== undefined) title.appendChild(h('span', {'class': 'rail-pid rail-owner', text: c.owner || '旧会话', title: '属主'}));
        box.appendChild(title);
        c.scans.forEach(function (s) {
          if (c.scans.length > 1 || s.label) box.appendChild(h('div', {'class': 'rail-scan', text: s.label || '扫描'}));
          var names = {};
          s.jobs.forEach(function (job) { var n = ui().resultName(job, st.cards, {short: true}); names[n] = (names[n] || 0) + 1; });
          s.jobs.forEach(function (job) {
            var status = rowStatus(job);
            var rname = ui().resultName(job, st.cards);
            var sname = ui().resultName(job, st.cards, {short: true});
            var label = names[sname] > 1 && job.created_at ? sname + ' · ' + ui().time(job.created_at).slice(5) : sname;
            // P3 lane 2: a running (or queued) result shows its remaining time and a thin progress line
            var eta = FINAL.indexOf(job.status) < 0 ? etaBadge(job, {cls: 'rail-eta'}) : null;
            var row = h('button', {type: 'button', 'class': 'rail-result' + (job.id === st.current ? ' current' : '') + (eta ? ' has-eta' : ''), dataset: {jobId: job.id},
              'aria-current': job.id === st.current ? 'true' : null, title: rname + ' · ' + status.label + (job.created_at ? ' · ' + ui().time(job.created_at) : '')},
              unread(job.id) ? h('span', {'class': 'rail-unread', title: '有新状态', 'aria-label': '未读'}) : null,
              h('span', {'class': 'rail-result-name', text: label}),
              eta ? h('span', {'class': 'st st-' + status.tone + ' rail-st'}, h('span', {'class': 'st-dot', 'aria-hidden': 'true'}), eta.el) : ui().dot(status.tone, status.label, 'rail-st'));
            row.addEventListener('click', function () { if (ns.admin && ns.admin.markRead) ns.admin.markRead(job.id); if (opts.onOpen) opts.onOpen(job.id); });
            box.appendChild(row);
          });
        });
        list.appendChild(box);
      });
      // lane C: the other home pages, one click away from any case
      ui().fill(foot, ns.admin ? h('nav', {'class': 'rail-pages', 'aria-label': '工作台页面'}, PAGES.slice(1).map(function (p) {
        return h('a', {'class': 'rail-page', href: p.href}, ui().icon(p.icon, {size: 14}), h('span', {text: p.label}));
      })) : null);
      st.tree = tree;
    }
    return {
      el: el,
      setJobs: function (jobs) { st.jobs = Array.isArray(jobs) ? jobs.slice() : []; render(); },
      setCards: function (cards) { st.cards = cards || {}; render(); },
      setCurrent: function (id) { st.current = id || null; render(); },
      setFull: function (full) { st.full = Boolean(full); render(); },
      order: function () { return order(model(st.jobs, {q: st.q, status: st.status, cards: st.cards})); },
      focusSearch: function () { try { search.focus(); } catch (_) {} },
      render: render
    };
  }

  // ------------------------------------------------------------------ home pages (lane C): shared head
  var PAGES = [
    {id: '', label: '病例', href: '#/', icon: 'logo'},
    {id: 'tasks', label: '任务', href: '#/tasks', icon: 'list'},
    {id: 'cohort', label: '队列', href: '#/cohort', icon: 'chart'},
    {id: 'trash', label: '回收站', href: '#/trash', icon: 'trash'}];
  // The four home pages as big title tabs, the count of the active one, then the page's own controls on the right.
  // Without the workbench module (ns.admin) only the case gallery exists and the head is its plain title.
  function homeHead(active, count, right) {
    var h = ui().h;
    var pages = ns.admin ? PAGES : PAGES.slice(0, 1);
    var countEl = count && typeof count === 'object' ? count : h('span', {'class': 'sec-count', text: count === undefined || count === null ? '' : String(count)});
    var tabs = h('nav', {'class': 'home-tabs', 'aria-label': '工作台页面'}, pages.map(function (p) {
      var on = p.id === (active || '');
      return on ? h('h2', {'class': 'home-tab on', 'aria-current': 'page'}, p.label, countEl) : h('a', {'class': 'home-tab', href: p.href}, p.label);
    }));
    return h('div', {'class': 'home-head'}, tabs, h('span', {'class': 'sec-fill'}), right || null);
  }

  // ------------------------------------------------------------------ to-do home (no case selected)
  function todoModel(jobs) {
    var out = {confirm: [], failed: [], review: [], running: []};
    (jobs || []).forEach(function (job) {
      var b = bucket(job);
      if (b === 'confirm') out.confirm.push(job);
      else if (b === 'failed') out.failed.push(job);
      else if (b === 'unreviewed') out.review.push(job);
      else if (b === 'running') out.running.push(job);
    });
    Object.keys(out).forEach(function (k) { out[k].sort(function (a, b) { return created(b) - created(a); }); });
    return out;
  }
  // Pure: the four home counts.  今日 = created today (local day); 进行中 = queued / running; 待处理 = outlets or
  // input to confirm + finished and not yet reviewed; 失败 = failed or interrupted.
  function homeCounts(jobs, nowMs) {
    var now = new Date(nowMs === undefined ? Date.now() : nowMs);
    var out = {today: 0, active: 0, todo: 0, failed: 0};
    (jobs || []).forEach(function (job) {
      var t = created(job);
      if (t) { var d = new Date(t); if (d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate()) out.today += 1; }
      var b = bucket(job);
      if (b === 'running') out.active += 1;
      else if (b === 'confirm' || b === 'unreviewed') out.todo += 1;
      else if (b === 'failed') out.failed += 1;
    });
    return out;
  }
  // Pure: what the home overview draws — the four counts plus the last 14 local days of uploads (oldest first), the
  // running jobs, the split of 待处理 into outlets / input to confirm and results to review, and the latest failure.
  var DAYS = 14, WEEKDAY = '日一二三四五六';
  function dayKey(t) { var d = new Date(t); return d.getFullYear() + '-' + (d.getMonth() + 1) + '-' + d.getDate(); }
  function homeStats(jobs, nowMs) {
    var now = nowMs === undefined ? Date.now() : nowMs;
    var out = homeCounts(jobs, now), index = {}, days = [];
    var noon = new Date(now); noon.setHours(12, 0, 0, 0);
    for (var i = DAYS - 1; i >= 0; i--) {
      var d = new Date(noon.getTime() - i * 86400000);
      var e = {key: dayKey(d.getTime()), month: d.getMonth() + 1, day: d.getDate(), weekday: d.getDay(), n: 0, failed: 0, today: i === 0};
      index[e.key] = e; days.push(e);
    }
    out.days = days; out.running = []; out.queued = 0; out.confirm = 0; out.review = 0; out.lastFailed = null;
    (jobs || []).forEach(function (job) {
      var t = created(job), b = bucket(job), day = t ? index[dayKey(t)] : null;
      if (day) { day.n += 1; if (b === 'failed') day.failed += 1; }
      if (b === 'running') { if (/^queued/.test(job.status || '')) out.queued += 1; else out.running.push(job); }
      else if (b === 'confirm') out.confirm += 1;
      else if (b === 'unreviewed') out.review += 1;
      else if (b === 'failed' && (!out.lastFailed || t > created(out.lastFailed))) out.lastFailed = job;
    });
    out.running.sort(function (a, b) { return created(b) - created(a); });
    out.recent = days.reduce(function (sum, x) { return sum + x.n; }, 0);
    return out;
  }
  // The overview band: four cards across the full width, each opening the task list with its filter.  今日 carries the
  // last 14 days as columns (today in the data blue, the other days in the de-emphasis gray); 待处理 a split meter.
  function overview(n) {
    var h = ui().h;
    var card = function (cls, label, tone, value, body, filter, tip) {
      var el = h('div', {'class': 'ov-card ' + cls, role: 'button', tabindex: '0', title: tip},
        h('div', {'class': 'ov-top'}, tone ? h('span', {'class': 'ov-dot tone-' + tone, 'aria-hidden': 'true'}) : null, h('span', {'class': 'ov-label', text: label})),
        value !== null ? h('div', {'class': 'ov-value', text: String(value)}) : null, body);
      var open = function () { ns.admin.openTasks(filter); };
      el.addEventListener('click', open);
      el.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { if (e.preventDefault) e.preventDefault(); open(); } });
      return el;
    };
    // 今日: value + the 14-day columns
    var peak = Math.max.apply(null, [1].concat(n.days.map(function (d) { return d.n; })));
    var chart = h('div', {'class': 'ov-chart'});
    var cols = h('div', {'class': 'ov-cols', role: 'img', 'aria-label': '近 14 天每天上传的任务数：' + n.days.map(function (d) { return d.month + '月' + d.day + '日 ' + d.n; }).join('，')});
    chart.appendChild(cols);
    var tip = ui().chartTip ? ui().chartTip(chart) : null;
    n.days.forEach(function (d) {
      var col = h('span', {'class': 'ov-col' + (d.today ? ' today' : '')},
        h('span', {'class': 'ov-col-bar' + (d.n ? '' : ' zero'), style: 'height:' + (d.n ? Math.max(8, Math.round(d.n / peak * 100)) : 0) + '%'}));
      if (tip) tip.bind(col, d.n + ' 个', (d.today ? '今天' : d.month + '月' + d.day + '日') + ' 周' + WEEKDAY[d.weekday] + (d.failed ? ' · 失败 ' + d.failed : ''));
      cols.appendChild(col);
    });
    chart.appendChild(h('div', {'class': 'ov-axis'}, h('span', {text: n.days[0].month + '/' + n.days[0].day}), h('span', {text: '今天'})));
    var today = card('ov-wide', '今日上传', null, null, h('div', {'class': 'ov-split'},
      h('div', {'class': 'ov-main'}, h('div', {'class': 'ov-value', text: String(n.today)}), h('div', {'class': 'ov-sub', text: '近 14 天共 ' + n.recent + ' 个'})), chart),
      {quick: 'today'}, '今天上传的任务；柱子是近 14 天每天的数量');
    // 进行中: the running jobs by name
    var runBody = n.running.length || n.queued
      ? h('div', {'class': 'ov-list'}, n.running.slice(0, 2).map(function (j) {
          var eta = etaBadge(j, {bar: false, cls: 'ov-run-what'});   // P3 lane 2: the remaining time when the service estimates it
          return h('div', {'class': 'ov-run'}, h('span', {'class': 'ov-run-name', text: ui().displayName(j)}), eta ? eta.el : h('span', {'class': 'ov-run-what', text: j.phase || '计算中'}));
        }), n.running.length > 2 ? h('div', {'class': 'ov-sub', text: '另有 ' + (n.running.length - 2) + ' 个在算'}) : null,
          n.queued ? h('div', {'class': 'ov-sub', text: '排队 ' + n.queued + ' 个'}) : null)
      : h('div', {'class': 'ov-sub', text: '当前空闲'});
    var active = card('', '进行中', n.active ? 'busy' : null, n.active, runBody, {status: 'active'}, '排队或计算中');
    // 待处理: split meter (outlets / input to confirm, results to review)
    var total = n.confirm + n.review;
    var seg = function (cls, k) { return k ? h('span', {'class': 'ov-seg ' + cls, style: 'flex-grow:' + k}) : null; };
    var todoBody = h('div', {'class': 'ov-foot'},
      h('div', {'class': 'ov-meter' + (total ? '' : ' empty')}, seg('seg-confirm', n.confirm), seg('seg-review', n.review)),
      h('div', {'class': 'ov-keys'},
        h('span', {'class': 'ov-key'}, h('span', {'class': 'ov-swatch seg-confirm'}), h('span', {text: '待确认 ' + n.confirm})),
        h('span', {'class': 'ov-key'}, h('span', {'class': 'ov-swatch seg-review'}), h('span', {text: '待复核 ' + n.review}))));
    var todoCard = card('', '待处理', total ? 'warn' : null, n.todo, todoBody, {quick: 'todo'}, '待确认出口或输入，以及已完成待复核');
    // 失败: the latest failure
    var lf = n.lastFailed;
    var failBody = lf
      ? h('div', {'class': 'ov-list'}, h('div', {'class': 'ov-sub', text: '最近一次'}),
          h('div', {'class': 'ov-run'}, h('span', {'class': 'ov-run-name', text: ui().displayName(lf)}), h('span', {'class': 'ov-run-what', text: ui().time(lf.created_at, false).slice(5)})))
      : h('div', {'class': 'ov-sub', text: '没有失败的任务'});
    var failed = card('', '失败', n.failed ? 'error' : null, n.failed, failBody, {status: 'failed'}, '失败或中断');
    return h('section', {'class': 'home-kpis', 'aria-label': '概况'}, today, active, todoCard, failed);
  }
  // Home: a gallery of cases (vessel thumbnail, name, scans, one chip per result) under the counts and a short strip
  // of the jobs that need an action (outlets to confirm, failures, running, results to review).  Pure layout;
  // thumbnails come from ns.thumbs.  A page route (#/tasks …) is drawn by the workbench module instead.
  function chipTone(job) {
    var b = bucket(job);
    return {reviewed: 'ok', unreviewed: 'idle', confirm: 'warn', failed: 'error', running: 'busy'}[b] || 'idle';
  }
  var ATTN_MAX = 8, GALLERY_STEP = 48, galleryShown = GALLERY_STEP;
  // The shell redraws the home after every list refresh (renderHome → todo); on a page route (#/tasks, #/cohort …, or
  // another lane's page) the page's own extension redraws itself instead of the gallery replacing it.
  function pageHandled(name, el) {
    var sa = ns.admin && ns.admin.shellApi ? ns.admin.shellApi() : undefined;
    return (Array.isArray(ns.ext) ? ns.ext : []).some(function (x) {
      if (!x || typeof x.page !== 'function') return false;
      try { return x.page(name, el, sa) === true; }
      catch (e) { if (root.console && root.console.error) root.console.error('extension ' + (x.id || '?') + '.page: ' + (e && e.stack || e)); return false; }
    });
  }
  // Pure: up to ``max`` items, taking one from each non-empty group in turn, then shown in group order.
  function attnPick(groups, max) {
    var take = groups.map(function () { return 0; }), left = max, more = true;
    while (left > 0 && more) {
      more = false;
      for (var g = 0; g < groups.length && left > 0; g++) if (take[g] < groups[g].length) { take[g] += 1; left -= 1; more = true; }
    }
    return [].concat.apply([], groups.map(function (list, g) { return list.slice(0, take[g]); }));
  }
  // ------------------------------------------------------------------ P3 lane 2: 三步上手, 最近完成
  var RECENT_MAX = 5, GUIDE_KEY = 'wss-guide-closed';   // the classic workbench's key: closed there = closed here
  function guideClosed() { try { return Boolean(root.localStorage) && root.localStorage.getItem(GUIDE_KEY) === '1'; } catch (_) { return false; } }
  function setGuideClosed(on) {
    try { if (!root.localStorage) return; if (on) root.localStorage.setItem(GUIDE_KEY, '1'); else root.localStorage.removeItem(GUIDE_KEY); } catch (_) {}
  }
  var GUIDE = [
    ['上传 STL', '顶栏「上传 STL」，或把文件拖进页面；选单位和要算的结果。'],
    ['核对单位和出口', '需要时会停下来：对照影像核对尺寸和四个髂支出口的名称。'],
    ['看结果、复核、出报告', '读结论和关键数字，核对后签字复核，再导出一页纸。']];
  function guideCard(opts, empty, onClose) {
    var h = ui().h;
    var steps = h('ol', {'class': 'guide-steps'}, GUIDE.map(function (s, i) {
      return h('li', {'class': 'guide-step'}, h('span', {'class': 'guide-n', 'aria-hidden': 'true', text: String(i + 1)}),
        h('span', {'class': 'guide-text'}, h('strong', {text: s[0]}), h('span', {text: s[1]})));
    }));
    var actions = h('div', {'class': 'guide-actions'},
      empty && opts.onUpload ? ui().button('上传 STL', opts.onUpload, {icon: 'upload', kind: 'primary', cls: 'btn-sm'}) : null,
      ui().link('操作卡', '/static/v2/help_quickstart.html', {newTab: true}),
      ui().link('输入要求', '/static/v2/help_input.html', {newTab: true}),
      empty ? ui().link('示例报告', '/v2/example', {newTab: true}) : null);
    return h('section', {'class': 'home-sec home-guide', 'aria-label': '三步上手'},
      h('div', {'class': 'home-sub-row'}, h('h3', {'class': 'home-sub', text: '三步上手'}), empty ? h('span', {'class': 'guide-note', text: '还没有病例'}) : null,
        h('span', {'class': 'sec-fill'}), empty ? null : ui().iconButton('close', '收起（头像菜单里可以再打开）', onClose, {cls: 'guide-close'})),
      h('div', {'class': 'guide-card'}, steps, actions));
  }
  // The service's release list (the shell has it before the first home is drawn; the workbench module after start).
  function releasesNow() {
    var S = ns.shell && ns.shell.state ? ns.shell.state() : null;
    if (S && Array.isArray(S.releases)) return S.releases;
    var sa = ns.admin && ns.admin.shellApi ? ns.admin.shellApi() : null;
    return (sa && sa.releases && sa.releases()) || [];
  }
  function shortWhen(iso) {
    var t = Date.parse(iso || '');
    if (isNaN(t)) return '';
    var d = new Date(t), now = new Date(), pad = function (n) { return n < 10 ? '0' + n : String(n); };
    var hm = pad(d.getHours()) + ':' + pad(d.getMinutes());
    if (d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate()) return hm;
    return (d.getFullYear() === now.getFullYear() ? '' : d.getFullYear() + '-') + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) + ' ' + hm;
  }
  function todo(el, jobs, opts) {
    opts = opts || {};
    var h = ui().h;
    var page = /^#\/([a-z][a-z0-9-]{1,30})\/?$/.exec((root.location && root.location.hash) || '');
    if (page && page[1] !== 'job' && pageHandled(page[1], el)) return todoModel(jobs);
    if (ns.admin && ns.admin.leftPage) ns.admin.leftPage();
    var cards = opts.cards || {};
    var m = todoModel(jobs);
    var q = opts.q || '';
    var tree = model(jobs, {q: q, cards: cards});
    var wrap = h('div', {'class': 'home'});
    var search = h('input', {type: 'search', 'class': 'home-search', placeholder: '搜索病例、患者编号', 'aria-label': '搜索病例', value: q});
    var timer = null;
    search.addEventListener('input', function () { if (timer) clearTimeout(timer); timer = setTimeout(function () { if (opts.onSearch) opts.onSearch(search.value); }, 160); });
    wrap.appendChild(homeHead('', String(tree.length), h('label', {'class': 'home-search-wrap'}, ui().icon('search'), search)));
    // overview band (lane C counts): each card opens the task list with that filter
    if (ns.admin && ns.admin.openTasks && (jobs || []).length) wrap.appendChild(overview(homeStats(jobs)));
    // P3 lane 2: 「三步上手」 — open while the library is empty and until it is closed once (classic 'wss-guide-closed')
    var emptyLib = !(jobs || []).length;
    if (!q && (emptyLib || !guideClosed())) wrap.appendChild(guideCard(opts, emptyLib, function () { setGuideClosed(true); todo(el, jobs, opts); }));
    // needs an action (待复核 included); when there are more than fit, every kind keeps a place (round robin in this order)
    var groups = [m.confirm.map(function (j) { return {job: j, tone: 'warn', what: j.status === 'awaiting_input' ? '核对单位与尺寸' : '确认出口', action: '去确认'}; }),
      m.failed.map(function (j) { return {job: j, tone: 'error', what: '失败', action: '查看原因'}; }),
      m.running.map(function (j) { return {job: j, tone: 'busy', what: j.phase || '计算中', action: '查看进度'}; }),
      m.review.map(function (j) { return {job: j, tone: 'idle', what: '待复核', action: '去复核'}; })];
    var attn = [].concat.apply([], groups), shownAttn = attnPick(groups, ATTN_MAX), attnSec = null, recentSec = null;
    if (attn.length) {
      var strip = h('div', {'class': 'attn'});
      shownAttn.forEach(function (a) {
        var eta = a.tone === 'busy' || a.tone === 'warn' ? etaBadge(a.job, {bar: false}) : null;   // P3 lane 2: remaining time instead of the phase
        var card = h('button', {type: 'button', 'class': 'attn-card tone-' + a.tone, title: a.action},
          ui().dot(a.tone, '', 'attn-dot'),
          h('span', {'class': 'attn-text'}, h('span', {'class': 'attn-name'}, unread(a.job.id) ? h('span', {'class': 'rail-unread', 'aria-label': '未读'}) : null, h('span', {text: ui().displayName(a.job)})),
            eta && a.tone === 'busy' ? h('span', {'class': 'attn-what'}, eta.el, ' · ' + ui().resultName(a.job, cards, {short: true}))
              : h('span', {'class': 'attn-what', text: a.what + ' · ' + ui().resultName(a.job, cards, {short: true})})),
          h('span', {'class': 'attn-go', text: a.action}));
        card.addEventListener('click', function () { if (ns.admin && ns.admin.markRead) ns.admin.markRead(a.job.id); if (opts.onOpen) opts.onOpen(a.job.id); });
        strip.appendChild(card);
      });
      var more = attn.length > ATTN_MAX && ns.admin && ns.admin.openTasks
        ? ui().button('全部 ' + attn.length + ' 条', function () { ns.admin.openTasks({quick: 'attn'}); }, {kind: 'link', cls: 'btn-sm', iconAfter: 'chevron-right'}) : null;
      attnSec = h('section', {'class': 'home-sec'}, h('div', {'class': 'home-sub-row'}, h('h3', {'class': 'home-sub', text: '需要处理'}), h('span', {'class': 'sec-fill'}), more), strip);
    }
    // P3 lane 2: the newest finished results with their review state (classic 今日概览「最近完成」)
    var recent = q ? [] : recentDone(jobs, RECENT_MAX);
    if (recent.length) {
      var rlist = h('div', {'class': 'recent', role: 'list'});
      recent.forEach(function (j) {
        var rv = bucket(j) === 'reviewed' ? {tone: 'ok', label: '已复核'} : {tone: 'idle', label: '待复核'};
        var at = j.finished_at || j.created_at;
        var row = h('button', {type: 'button', 'class': 'recent-row', role: 'listitem', title: ui().resultName(j, cards) + (at ? ' · 完成于 ' + ui().time(at) : '')},
          h('span', {'class': 'recent-main'},
            h('span', {'class': 'recent-name'}, unread(j.id) ? h('span', {'class': 'rail-unread', 'aria-label': '未读'}) : null, h('span', {text: ui().displayName(j)})),
            h('span', {'class': 'recent-what', text: ui().resultName(j, cards, {short: true}) + (at ? ' · ' + shortWhen(at) : '')})),
          ui().dot(rv.tone, rv.label, 'recent-st'));
        row.addEventListener('click', function () { if (ns.admin && ns.admin.markRead) ns.admin.markRead(j.id); if (opts.onOpen) opts.onOpen(j.id); });
        rlist.appendChild(row);
      });
      recentSec = h('section', {'class': 'home-sec home-recent'}, h('div', {'class': 'home-sub-row'}, h('h3', {'class': 'home-sub', text: '最近完成'}), h('span', {'class': 'sec-fill'})), rlist);
    }
    // 需要处理 and 最近完成 side by side (stacked on narrow screens)
    if (attnSec && recentSec) wrap.appendChild(h('div', {'class': 'home-split'}, attnSec, recentSec));
    else if (attnSec || recentSec) wrap.appendChild(attnSec || recentSec);
    // gallery: the newest cases first; more on request (every card queues a thumbnail, so a long history is not all drawn at once)
    var grid = h('div', {'class': 'gallery'});
    var releases = releasesNow();
    var limit = q ? tree.length : Math.max(GALLERY_STEP, galleryShown);
    tree.slice(0, limit).forEach(function (c) {
      var all = [];
      c.scans.forEach(function (s) { s.jobs.forEach(function (j) { all.push(j); }); });
      var done = all.filter(function (j) { return j.status === 'done'; });
      var primary = done[0] || all[0];
      var thumbJob = all.filter(function (j) { return j.status !== 'failed'; })[0] || all[0];
      var thumb = h('div', {'class': 'case-thumb'});
      var card = h('article', {'class': 'case-card', tabindex: '0', role: 'button', 'aria-label': c.name});
      var scanText = c.scans.length > 1 ? c.scans.length + ' 次扫描 · ' + (c.scans[0].label || '') : (c.scans[0] && c.scans[0].label) || '';
      var chips = h('div', {'class': 'case-results'});
      c.scans[0].jobs.forEach(function (j) {
        var chip = h('button', {type: 'button', 'class': 'res-chip' + (unread(j.id) ? ' unread' : ''), title: ui().resultName(j, cards) + ' · ' + rowStatus(j).label + (unread(j.id) ? ' · 有新状态' : '')},
          h('span', {'class': 'res-dot tone-' + chipTone(j)}), h('span', {text: ui().resultName(j, cards, {short: true})}));
        chip.addEventListener('click', function (e) { e.stopPropagation(); if (ns.admin && ns.admin.markRead) ns.admin.markRead(j.id); if (opts.onOpen) opts.onOpen(j.id); });
        chips.appendChild(chip);
      });
      // P3 lane 2: 「补跑缺少的结果」 — one dashed chip per release this scan has no finished result of
      if (ns.admin && ns.admin.fillMissing) {
        var miss = missingResults(c.scans[0].jobs, releases, {canRun: ns.admin.canChange});
        miss.missing.forEach(function (x) {
          var short = ui().resultName({model_release: x.release}, cards, {short: true}), full = ui().resultName({model_release: x.release}, cards);
          var add = h('button', {type: 'button', 'class': 'res-chip res-add', title: '补跑「' + full + '」：沿用这次扫描已确认的中心线和出口', 'aria-label': '补跑' + full},
            ui().icon('plus', {size: 12}), h('span', {text: short}));
          add.addEventListener('click', function (e) { e.stopPropagation(); ns.admin.fillMissing(miss.source, x.releaseId, full); });
          add.addEventListener('keydown', function (e) { if (e && e.stopPropagation) e.stopPropagation(); });
          chips.appendChild(add);
        });
      }
      var facts = caseFacts(c);
      card.appendChild(thumb);
      card.appendChild(h('div', {'class': 'case-body'},
        h('div', {'class': 'case-name-row'}, h('div', {'class': 'case-name', text: c.name}),
          facts.diameter !== null ? h('span', {'class': 'case-dia', title: '管腔最大直径（最近一次扫描的完成结果）', text: ui().sig(facts.diameter) + ' mm'}) : null),
        h('div', {'class': 'case-meta', text: [c.patientId && c.patientId !== c.name ? c.patientId : '', scanText, c.owner !== undefined ? (c.owner || '旧会话') : ''].filter(Boolean).join(' · ')}),
        facts.tags.length ? h('div', {'class': 'case-tags', title: facts.tags.map(function (t) { return '#' + t; }).join(' '),
          text: facts.tags.slice(0, 4).map(function (t) { return '#' + t; }).join(' ') + (facts.tags.length > 4 ? ' …' : '')}) : null,
        chips));
      var open = function () { if (opts.onOpen && primary) opts.onOpen(primary.id); };
      card.addEventListener('click', open);
      card.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
      grid.appendChild(card);
      // thumbnail: a wall result reads best (cycle metrics first, then peak WSS); a volume result only when alone
      var rank = function (j) { var k = ui().kindOf ? ui().kindOf(j) : null; return k === 'cycle' ? 0 : k === 'wall' ? 1 : 2; };
      var tj = done.slice().sort(function (a, b) { return rank(a) - rank(b); })[0] || thumbJob;
      if (ns.thumbs && opts.fetchThumb && tj && tj.status !== 'failed') {
        var tkey = ns.thumbs.keyFor ? ns.thumbs.keyFor(tj) : tj.id;   // P3 lane 2: job + run identity (persistent cache)
        var cached = ns.thumbs.cached(tkey);
        var put = function (url) { if (url) thumb.replaceChildren(h('img', {src: url, alt: '', draggable: 'false'})); else thumb.classList.add('no-thumb'); };
        if (cached) put(cached);
        else ns.thumbs.request(tkey, function () { return opts.fetchThumb(tj); }, put);
      } else thumb.classList.add('no-thumb');
    });
    var moreCases = tree.length > limit ? ui().button('再显示 ' + Math.min(GALLERY_STEP, tree.length - limit) + ' 个病例（共 ' + tree.length + ' 个）', function () {
      galleryShown = limit + GALLERY_STEP; todo(el, jobs, opts);
    }, {cls: 'gallery-more'}) : null;
    if (tree.length) wrap.appendChild(h('section', {'class': 'home-sec'}, attn.length || recent.length ? h('h3', {'class': 'home-sub', text: '全部病例'}) : null, grid, moreCases));
    else if (!(emptyLib && !q)) wrap.appendChild(ui().empty(q ? '没有符合搜索的病例。' : '还没有病例。上传一份管腔 STL，或把 STL 拖进页面。', [
      opts.onUpload ? ui().button('上传 STL', opts.onUpload, {icon: 'upload', kind: 'primary'}) : null,
      ui().link('输入要求', '/static/v2/help_input.html', {newTab: true}),
      ui().link('示例报告', '/v2/example', {newTab: true})
    ].filter(Boolean)));
    el.replaceChildren(wrap);
    if (opts.focusSearch) { try { search.focus(); search.setSelectionRange(search.value.length, search.value.length); } catch (_) {} }
    return m;
  }

  return {create: create, model: model, order: order, todo: todo, todoModel: todoModel, bucket: bucket, FILTERS: FILTERS, homeHead: homeHead, homeCounts: homeCounts, homeStats: homeStats, attnPick: attnPick, PAGES: PAGES,
    // P3 lane 2
    etaView: etaView, etaRowSummary: etaRowSummary, formatDuration: formatDuration, stageRemaining: stageRemaining, etaBadge: etaBadge,
    missingResults: missingResults, caseFacts: caseFacts, recentDone: recentDone, guideClosed: guideClosed, setGuideClosed: setGuideClosed, GUIDE: GUIDE};
});
