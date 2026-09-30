/* WSS workspace v2 — case rail and the to-do home (contract §6.3 ws_rail.js).
 * Three layers: case (patient) → scan (one input geometry) → result (one release run).  Names, scans and result
 * titles come from the job records and model cards; no release code is shown. */
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
    if (job.patient_id) return 'p:' + String(job.patient_id).trim();
    return 'c:' + ui().displayName(job);
  }
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
      if (!c) { c = cases[ck] = {key: ck, name: ui().displayName(job), patientId: job.patient_id || '', latest: 0, scans: {}, scanOrder: []}; order.push(ck); }
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
      return {key: c.key, name: c.name, patientId: c.patientId, latest: c.latest, scans: scans};
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
              h('span', {'class': 'rail-result-name', text: label}),
              ui().dot(status.tone, status.label, 'rail-st'));
            row.addEventListener('click', function () { if (opts.onOpen) opts.onOpen(job.id); });
            box.appendChild(row);
          });
        });
        list.appendChild(box);
      });
      foot.replaceChildren();
      if (st.full) {
        foot.appendChild(h('p', {'class': 'rail-foot-text', text: '队列总览、回收站、批量上传仍在经典工作台。'}));
        foot.appendChild(ui().link('打开经典工作台', '/', {newTab: true}));
      }
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
  // Home: a gallery of cases (vessel thumbnail, name, scans, one chip per result) under a short strip of the jobs
  // that need an action (outlets to confirm, failures, running).  Pure layout; thumbnails come from ns.thumbs.
  function chipTone(job) {
    var b = bucket(job);
    return {reviewed: 'ok', unreviewed: 'idle', confirm: 'warn', failed: 'error', running: 'busy'}[b] || 'idle';
  }
  function todo(el, jobs, opts) {
    opts = opts || {};
    var h = ui().h;
    var cards = opts.cards || {};
    var m = todoModel(jobs);
    var q = opts.q || '';
    var tree = model(jobs, {q: q, cards: cards});
    var wrap = h('div', {'class': 'home'});
    var search = h('input', {type: 'search', 'class': 'home-search', placeholder: '搜索病例、患者编号', 'aria-label': '搜索病例', value: q});
    var timer = null;
    search.addEventListener('input', function () { if (timer) clearTimeout(timer); timer = setTimeout(function () { if (opts.onSearch) opts.onSearch(search.value); }, 160); });
    wrap.appendChild(h('div', {'class': 'home-head'}, h('h2', {text: '病例'}), h('span', {'class': 'sec-count', text: String(tree.length)}), h('span', {'class': 'sec-fill'}),
      h('label', {'class': 'home-search-wrap'}, ui().icon('search'), search)));
    // needs an action
    var attn = m.confirm.map(function (j) { return {job: j, tone: 'warn', what: j.status === 'awaiting_input' ? '核对单位与尺寸' : '确认出口', action: '去确认'}; })
      .concat(m.failed.map(function (j) { return {job: j, tone: 'error', what: '失败', action: '查看原因'}; }))
      .concat(m.running.map(function (j) { return {job: j, tone: 'busy', what: j.phase || '计算中', action: '查看进度'}; }));
    if (attn.length) {
      var strip = h('div', {'class': 'attn'});
      attn.slice(0, 8).forEach(function (a) {
        var card = h('button', {type: 'button', 'class': 'attn-card tone-' + a.tone, title: a.action},
          ui().dot(a.tone, '', 'attn-dot'),
          h('span', {'class': 'attn-text'}, h('span', {'class': 'attn-name', text: ui().displayName(a.job)}),
            h('span', {'class': 'attn-what', text: a.what + ' · ' + ui().resultName(a.job, cards, {short: true})})),
          h('span', {'class': 'attn-go', text: a.action}));
        card.addEventListener('click', function () { if (opts.onOpen) opts.onOpen(a.job.id); });
        strip.appendChild(card);
      });
      wrap.appendChild(h('section', {'class': 'home-sec'}, h('h3', {'class': 'home-sub', text: '需要处理'}), strip));
    }
    // gallery
    var grid = h('div', {'class': 'gallery'});
    tree.forEach(function (c) {
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
        var chip = h('button', {type: 'button', 'class': 'res-chip', title: ui().resultName(j, cards) + ' · ' + rowStatus(j).label},
          h('span', {'class': 'res-dot tone-' + chipTone(j)}), h('span', {text: ui().resultName(j, cards, {short: true})}));
        chip.addEventListener('click', function (e) { e.stopPropagation(); if (opts.onOpen) opts.onOpen(j.id); });
        chips.appendChild(chip);
      });
      card.appendChild(thumb);
      card.appendChild(h('div', {'class': 'case-body'},
        h('div', {'class': 'case-name', text: c.name}),
        h('div', {'class': 'case-meta', text: [c.patientId && c.patientId !== c.name ? c.patientId : '', scanText].filter(Boolean).join(' · ')}),
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
    if (tree.length) wrap.appendChild(h('section', {'class': 'home-sec'}, attn.length ? h('h3', {'class': 'home-sub', text: '全部病例'}) : null, grid));
    else wrap.appendChild(ui().empty(q ? '没有符合搜索的病例。' : '还没有病例。上传一份管腔 STL 开始第一例。', [
      opts.onUpload ? ui().button('上传 STL', opts.onUpload, {icon: 'upload', kind: 'primary'}) : null,
      ui().link('输入要求', '/static/v2/help_input.html', {newTab: true}),
      ui().link('示例报告', '/v2/example', {newTab: true})
    ].filter(Boolean)));
    el.replaceChildren(wrap);
    if (opts.focusSearch) { try { search.focus(); search.setSelectionRange(search.value.length, search.value.length); } catch (_) {} }
    return m;
  }

  return {create: create, model: model, order: order, todo: todo, todoModel: todoModel, bucket: bucket, FILTERS: FILTERS};
});
