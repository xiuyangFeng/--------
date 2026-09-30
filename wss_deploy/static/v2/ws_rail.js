/* WSS workspace v2 — case rail and the to-do home (contract §6.3 ws_rail.js).
 * Three layers: case (patient) → scan (one input geometry) → result (one release run).  Names, scans and result
 * titles come from the job records and model cards; no release code is shown.
 * Lane C (S6c): the home carries a head with the four home pages (病例 / 任务 / 队列 / 回收站, the last three live in
 * ws_admin.js), four counts (今日 / 进行中 / 待处理 / 失败), 待复核 in 「需要处理」, unread marks and, in the
 * administrator's all-users view, the owner of each case. */
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
            var row = h('button', {type: 'button', 'class': 'rail-result' + (job.id === st.current ? ' current' : ''), dataset: {jobId: job.id},
              'aria-current': job.id === st.current ? 'true' : null, title: rname + ' · ' + status.label + (job.created_at ? ' · ' + ui().time(job.created_at) : '')},
              unread(job.id) ? h('span', {'class': 'rail-unread', title: '有新状态', 'aria-label': '未读'}) : null,
              h('span', {'class': 'rail-result-name', text: label}),
              ui().dot(status.tone, status.label, 'rail-st'));
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
    // counts (lane C): each opens the task list with that filter
    if (ns.admin && ns.admin.openTasks && (jobs || []).length) {
      var n = homeCounts(jobs);
      var tile = function (label, value, tone, filter, tip) {
        var b = h('button', {type: 'button', 'class': 'kpi-tile' + (value && tone ? ' tone-' + tone : ''), title: tip},
          h('span', {'class': 'kpi-tile-value', text: String(value)}), h('span', {'class': 'kpi-tile-label', text: label}));
        b.addEventListener('click', function () { ns.admin.openTasks(filter); });
        return b;
      };
      wrap.appendChild(h('div', {'class': 'home-kpis'},
        tile('今日', n.today, '', {quick: 'today'}, '今天上传的任务'),
        tile('进行中', n.active, 'busy', {status: 'active'}, '排队或计算中'),
        tile('待处理', n.todo, 'warn', {quick: 'todo'}, '待确认出口或输入，以及已完成待复核'),
        tile('失败', n.failed, 'error', {status: 'failed'}, '失败或中断')));
    }
    // needs an action (待复核 included); when there are more than fit, every kind keeps a place (round robin in this order)
    var groups = [m.confirm.map(function (j) { return {job: j, tone: 'warn', what: j.status === 'awaiting_input' ? '核对单位与尺寸' : '确认出口', action: '去确认'}; }),
      m.failed.map(function (j) { return {job: j, tone: 'error', what: '失败', action: '查看原因'}; }),
      m.running.map(function (j) { return {job: j, tone: 'busy', what: j.phase || '计算中', action: '查看进度'}; }),
      m.review.map(function (j) { return {job: j, tone: 'idle', what: '待复核', action: '去复核'}; })];
    var attn = [].concat.apply([], groups), shownAttn = attnPick(groups, ATTN_MAX);
    if (attn.length) {
      var strip = h('div', {'class': 'attn'});
      shownAttn.forEach(function (a) {
        var card = h('button', {type: 'button', 'class': 'attn-card tone-' + a.tone, title: a.action},
          ui().dot(a.tone, '', 'attn-dot'),
          h('span', {'class': 'attn-text'}, h('span', {'class': 'attn-name'}, unread(a.job.id) ? h('span', {'class': 'rail-unread', 'aria-label': '未读'}) : null, h('span', {text: ui().displayName(a.job)})),
            h('span', {'class': 'attn-what', text: a.what + ' · ' + ui().resultName(a.job, cards, {short: true})})),
          h('span', {'class': 'attn-go', text: a.action}));
        card.addEventListener('click', function () { if (ns.admin && ns.admin.markRead) ns.admin.markRead(a.job.id); if (opts.onOpen) opts.onOpen(a.job.id); });
        strip.appendChild(card);
      });
      var more = attn.length > ATTN_MAX && ns.admin && ns.admin.openTasks
        ? ui().button('全部 ' + attn.length + ' 条', function () { ns.admin.openTasks({quick: 'attn'}); }, {kind: 'link', cls: 'btn-sm', iconAfter: 'chevron-right'}) : null;
      wrap.appendChild(h('section', {'class': 'home-sec'}, h('div', {'class': 'home-sub-row'}, h('h3', {'class': 'home-sub', text: '需要处理'}), h('span', {'class': 'sec-fill'}), more), strip));
    }
    // gallery: the newest cases first; more on request (every card queues a thumbnail, so a long history is not all drawn at once)
    var grid = h('div', {'class': 'gallery'});
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
      card.appendChild(thumb);
      card.appendChild(h('div', {'class': 'case-body'},
        h('div', {'class': 'case-name', text: c.name}),
        h('div', {'class': 'case-meta', text: [c.patientId && c.patientId !== c.name ? c.patientId : '', scanText, c.owner !== undefined ? (c.owner || '旧会话') : ''].filter(Boolean).join(' · ')}),
        chips));
      var open = function () { if (opts.onOpen && primary) opts.onOpen(primary.id); };
      card.addEventListener('click', open);
      card.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
      grid.appendChild(card);
      // thumbnail: a wall result reads best (cycle metrics first, then peak WSS); a volume result only when alone
      var rank = function (j) { var k = ui().kindOf ? ui().kindOf(j) : null; return k === 'cycle' ? 0 : k === 'wall' ? 1 : 2; };
      var tj = done.slice().sort(function (a, b) { return rank(a) - rank(b); })[0] || thumbJob;
      if (ns.thumbs && opts.fetchThumb && tj && tj.status !== 'failed') {
        var cached = ns.thumbs.cached(tj.id);
        var put = function (url) { if (url) thumb.replaceChildren(h('img', {src: url, alt: '', draggable: 'false'})); else thumb.classList.add('no-thumb'); };
        if (cached) put(cached);
        else ns.thumbs.request(tj.id, function () { return opts.fetchThumb(tj); }, put);
      } else thumb.classList.add('no-thumb');
    });
    var moreCases = tree.length > limit ? ui().button('再显示 ' + Math.min(GALLERY_STEP, tree.length - limit) + ' 个病例（共 ' + tree.length + ' 个）', function () {
      galleryShown = limit + GALLERY_STEP; todo(el, jobs, opts);
    }, {cls: 'gallery-more'}) : null;
    if (tree.length) wrap.appendChild(h('section', {'class': 'home-sec'}, attn.length ? h('h3', {'class': 'home-sub', text: '全部病例'}) : null, grid, moreCases));
    else wrap.appendChild(ui().empty(q ? '没有符合搜索的病例。' : '还没有病例。上传一份管腔 STL，或把 STL 拖进页面。', [
      opts.onUpload ? ui().button('上传 STL', opts.onUpload, {icon: 'upload', kind: 'primary'}) : null,
      ui().link('输入要求', '/static/v2/help_input.html', {newTab: true}),
      ui().link('示例报告', '/v2/example', {newTab: true})
    ].filter(Boolean)));
    el.replaceChildren(wrap);
    if (opts.focusSearch) { try { search.focus(); search.setSelectionRange(search.value.length, search.value.length); } catch (_) {} }
    return m;
  }

  return {create: create, model: model, order: order, todo: todo, todoModel: todoModel, bucket: bucket, FILTERS: FILTERS, homeHead: homeHead, homeCounts: homeCounts, attnPick: attnPick, PAGES: PAGES};
});
