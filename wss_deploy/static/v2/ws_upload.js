/* WSS workspace v2 — upload dialog (contract §6.3 ws_upload.js, U1 / U14 / U17 / C7).
 * Four sentences of input requirements with the help pages and the sample STL; 「想看什么」 lists model-card names
 * and one validation line (never a release code); patient number and scan date are marked 「随访需要」; a file
 * name that looks like a person's name gets a reminder.  Duplicate handling follows the existing service flow.
 * Lane C (S6c): several files at once (POST /api/jobs/batch in chunks, like the classic workbench), the remaining
 * metadata (scan label, tags, notes, 「记住患者」 in the account preferences), device / models / threads in the full
 * tier, and STL files dropped anywhere on the page. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.upload = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };
  var api = function () { return ns.api; };

  var NAME_RE = /^[A-Za-z]+(?:[_ -][A-Za-z]+){1,3}$/;
  function stem(filename) { return String(filename || '').replace(/^.*[\\/]/, '').replace(/\.stl$/i, ''); }
  function looksLikeName(filename) { return NAME_RE.test(stem(filename)); }
  var REQUIREMENTS = [
    '管腔面：分割出的血管内腔表面，不含附壁血栓和管壁。',
    '五个开口：主动脉入口和左右髂内、髂外四个出口切开，其余分支封闭。',
    '单位：毫米、厘米或米都可以；拿不准选「自动判断」，上传后会请你核对尺寸。',
    '一侧髂动脉闭塞（只有三个出口）的几何目前不支持。'
  ];
  // Service limits (server.py MAX_UPLOAD_BYTES / MAX_BATCH_FILES / MAX_BATCH_BYTES); the session reports the live ones.
  var DEFAULT_MAX_FILE = 128 * 1024 * 1024, DEFAULT_BATCH_FILES = 20, DEFAULT_BATCH_BYTES = 256 * 1024 * 1024;
  function releaseId(r) { return r && (r.id || r.release) || ''; }
  function releaseKind(r) {
    var c = (r && r.contract) || {}, f = c.fields || {};
    if (c.protocol === 'single_frame_volume' || f.velocity) return 'volume';
    if (c.protocol === 'single_frame_wss_cycle_multi' || f.tawss || f.osi) return 'cycle';
    return 'wall';
  }
  var FIELD_NAMES = {wss: '峰值 WSS', tawss: 'TAWSS', osi: 'OSI', pressure: '压力', speed: '速度', velocity: '速度'};
  function validationLine(card) {
    if (!card) return '暂无模型说明';
    var v = card.validation || {};
    var fields = v.fields || {};
    var parts = Object.keys(fields).filter(function (k) { return fields[k] && fields[k].r2_pa !== null && fields[k].r2_pa !== undefined; })
      .map(function (k) { return (FIELD_NAMES[k] || k.toUpperCase()) + ' ' + ui().sig(fields[k].r2_pa, 2); });
    if (!parts.length) {
      var anyNote = Object.keys(fields).map(function (k) { return fields[k] && fields[k].summary; }).concat([v.note, v.r2_note]).filter(Boolean).join(' ');
      return '暂无留出集一致性数字' + (/口径/.test(anyNote) ? '（口径尚未建立）' : '');
    }
    return (v.holdout_n ? v.holdout_n + ' 例留出，' : '') + '与 CFD 的 R²：' + parts.join('，');
  }
  // Pure: releases + cards → the 「想看什么」 choices (value = release id or ids joined by '+').
  function choices(releases, cards) {
    cards = cards || {};
    var list = (releases || []).filter(function (r) { return releaseId(r); });
    var kindName = {volume: '体内压力与速度', cycle: '周期指标 TAWSS · OSI', wall: '峰值 WSS'};
    var out = list.map(function (r) {
      var card = cards[releaseId(r)] || null;
      return {value: releaseId(r), label: (card && card.display_name) || r.display_name || kindName[releaseKind(r)], detail: validationLine(card),
        purpose: card && card.purpose || '', isDefault: Boolean(r.default), kind: releaseKind(r), models: Number(r.models_count || r.model_count || 0) || null};
    });
    var cycles = list.filter(function (r) { return releaseKind(r) === 'cycle'; }), vols = list.filter(function (r) { return releaseKind(r) === 'volume'; });
    cycles.forEach(function (c) {
      vols.forEach(function (v) {
        var cc = cards[releaseId(c)], vc = cards[releaseId(v)];
        out.push({value: releaseId(c) + '+' + releaseId(v), label: ((cc && cc.short_name) || '周期指标') + ' ＋ ' + ((vc && vc.short_name) || '体内压力与速度'),
          detail: '同时建两个任务，出口只确认一次', purpose: '', isDefault: false, kind: 'combo', models: null});
      });
    });
    return out;
  }

  // ------------------------------------------------------------------ pure file checks (workbench_core.checkUploadFiles / uploadChunks)
  function formatBytes(n) {
    n = Number(n);
    if (!isFinite(n) || n < 0) return '—';
    if (n < 1024) return Math.round(n) + ' B';
    if (n < 1048576) return (n / 1024).toFixed(n < 10240 ? 1 : 0) + ' KB';
    if (n < 1073741824) return (n / 1048576).toFixed(n < 10485760 ? 1 : 0) + ' MB';
    return (n / 1073741824).toFixed(1) + ' GB';
  }
  // Every chosen file is checked here first (the service checks again): extension, empty, size, a repeated name,
  // and the 「可能是姓名」 reminder unless a patient number is filled in.
  function checkFiles(files, opts) {
    opts = opts || {};
    var max = Number(opts.maxBytes) || DEFAULT_MAX_FILE, seen = {}, rows = [];
    Array.prototype.slice.call(files || []).forEach(function (f, i) {
      var name = String((f && f.name) || ''), size = Number(f && f.size) || 0, reason = '', warn = '';
      if (!/\.stl$/i.test(name)) reason = '不是 .stl 文件';
      else if (!size) reason = '空文件';
      else if (size > max) reason = '超过 ' + formatBytes(max) + ' 上限，先在建模软件里抽稀';
      else if (Object.prototype.hasOwnProperty.call(seen, name.toLowerCase())) warn = '与第 ' + (seen[name.toLowerCase()] + 1) + ' 个同名';
      else { seen[name.toLowerCase()] = i; if (opts.nameHint !== false && looksLikeName(name)) warn = '文件名可能是姓名'; }
      rows.push({index: i, name: name, size: size, ok: !reason, reason: reason, warn: warn});
    });
    return {rows: rows, valid: rows.filter(function (r) { return r.ok; }).map(function (r) { return files[r.index]; }), invalid: rows.filter(function (r) { return !r.ok; }).length};
  }
  function uploadChunks(files, opts) {
    opts = opts || {};
    var maxFiles = Math.max(1, Number(opts.maxFiles) || DEFAULT_BATCH_FILES), maxBytes = Number(opts.maxBytes) || 240 * 1024 * 1024;
    var chunks = [], cur = [], bytes = 0;
    Array.prototype.slice.call(files || []).forEach(function (f) {
      var size = Math.max(0, Number(f && f.size) || 0);
      if (cur.length && (cur.length >= maxFiles || bytes + size > maxBytes)) { chunks.push(cur); cur = []; bytes = 0; }
      cur.push(f); bytes += size;
    });
    if (cur.length) chunks.push(cur);
    return chunks;
  }
  // Case name of one file: a single file takes the typed name; several files take 「前缀-文件名」 (classic caseIdFor).
  function caseIdFor(base, file, many) {
    var s = stem(file && file.name);
    base = String(base || '').trim();
    return many ? (base ? base + '-' + s : s) : base;
  }
  function parseTags(text) {
    var out = [];
    String(text === null || text === undefined ? '' : text).split(/[,，、\n]/).forEach(function (p) { var t = p.trim(); if (t && out.indexOf(t) < 0) out.push(t); });
    return out;
  }

  // ------------------------------------------------------------------ dialog
  var current = null;   // the open dialog's controller (whole-page drop adds files to it)
  function open(ctx) {
    ctx = ctx || {};
    var h = ui().h;
    var full = Boolean(ns.store && ns.store.prefs && ns.store.prefs().tier === 'full');
    var session = (ns.store && ns.store.state && ns.store.state.session) || {};
    var up = (ns.admin && ns.admin.prefs && ns.admin.prefs().upload) || {};
    var st = {files: [], busy: false, results: []};
    var opts = choices(ctx.releases || [], ctx.cards || {});
    var chosen = ((up.release_id && opts.filter(function (o) { return o.value === up.release_id; })[0]) || opts.filter(function (o) { return o.isDefault; })[0] || opts[0] || {}).value || '';
    var file = h('input', {type: 'file', accept: '.stl', multiple: true, 'class': 'visually-hidden', id: 'wsu-file', 'aria-label': '选择 STL 文件'});
    var fileText = h('span', {'class': 'drop-text', text: '选择或拖入 STL，可以多选'});
    var drop = h('label', {'class': 'drop', 'for': 'wsu-file'}, ui().icon('upload'), fileText);
    var fileList = h('div', {'class': 'up-files', hidden: true});
    var nameWarn = h('p', {'class': 'note note-warn', hidden: true, text: '文件名可能是姓名。建议填写患者编号：界面、报告和导出会优先显示患者编号。'});
    var caseLabel = h('span', {text: '病例名称'});
    var caseId = h('input', {type: 'text', maxLength: 160, placeholder: '默认用文件名', 'aria-label': '病例名称'});
    var patient = h('input', {type: 'text', maxLength: 80, placeholder: '如 P-001', 'aria-label': '患者编号'});
    var scanDate = h('input', {type: 'date', 'aria-label': '扫描日期'});
    var scanLabel = h('input', {type: 'text', maxLength: 120, placeholder: '如 基线、随访 1', 'aria-label': '扫描标签'});
    var tags = h('input', {type: 'text', maxLength: 1024, placeholder: '逗号分隔，如 AAA, 随访', 'aria-label': '标签'});
    var notes = h('textarea', {rows: 2, maxLength: 2000, 'aria-label': '备注', placeholder: '可选'});
    var remember = h('input', {type: 'checkbox', 'aria-label': '记住患者编号、扫描信息和标签', checked: Boolean(up.remember_patient)});
    if (up.remember_patient && up.last_patient) {
      var last = up.last_patient;
      patient.value = last.patient_id || ''; scanLabel.value = last.scan_label || ''; scanDate.value = last.scan_date || ''; tags.value = last.tags || '';
    }
    var unitOpts = [{value: 'auto', label: '自动判断'}, {value: 'mm', label: '毫米 mm'}, {value: 'cm', label: '厘米 cm'}, {value: 'm', label: '米 m'}];
    var units = ui().select(unitOpts, unitOpts.some(function (o) { return o.value === up.units; }) ? up.units : 'auto', null, {'aria-label': '单位'});
    var device = ui().select([{value: 'auto', label: '自动'}, {value: 'cpu', label: 'CPU'}, {value: 'cuda', label: 'GPU'}], ['auto', 'cpu', 'cuda'].indexOf(up.device) >= 0 ? up.device : 'auto', null, {'aria-label': '计算设备'});
    var seeds = h('select', {'aria-label': '集成模型数'});
    var threads = h('input', {type: 'number', min: 1, max: 32, step: 1, placeholder: '默认', 'aria-label': 'CPU 线程数', value: up.threads ? String(up.threads) : ''});
    var progress = h('div', {'class': 'upload-progress', 'aria-live': 'polite', hidden: true});
    var radioName = 'wsu-release';
    var choiceList = h('div', {'class': 'choices', role: 'radiogroup', 'aria-label': '想看什么'});
    if (!opts.length) choiceList.appendChild(ui().note('服务上没有可用的模型，暂时不能上传。', 'error'));
    function fillSeeds() {
      var o = opts.filter(function (x) { return x.value === chosen; })[0] || {};
      var prev = seeds.value || (up.seed_count ? String(up.seed_count) : 'all');
      var list = [{value: 'all', label: o.kind === 'combo' ? '全部（各模型包）' : '全部' + (o.models ? '（' + o.models + ' 个）' : '')}];
      if (o.kind !== 'combo') [1, 3].forEach(function (n) { if (!o.models || n < o.models) list.push({value: String(n), label: n + ' 个'}); });
      ui().fill(seeds, list.map(function (x) { return h('option', {value: x.value, text: x.label}); }));
      seeds.value = list.some(function (x) { return x.value === prev; }) ? prev : 'all';
    }
    opts.forEach(function (o) {
      var input = h('input', {type: 'radio', name: radioName, value: o.value, checked: o.value === chosen});
      input.addEventListener('change', function () { if (input.checked) { chosen = o.value; fillSeeds(); } });
      choiceList.appendChild(h('label', {'class': 'choice'}, input, h('span', {'class': 'choice-text'}, h('span', {'class': 'choice-label', text: o.label}), h('span', {'class': 'choice-detail', text: o.detail}))));
    });
    fillSeeds();
    function limits() {
      return {maxBytes: session.max_upload_bytes || DEFAULT_MAX_FILE, maxFiles: session.max_batch_files || DEFAULT_BATCH_FILES,
        batchBytes: Math.floor((session.max_batch_bytes || DEFAULT_BATCH_BYTES) * 15 / 16)};
    }
    function check() { return checkFiles(st.files, {maxBytes: limits().maxBytes, nameHint: !patient.value.trim() && !(st.files.length === 1 && caseId.value.trim())}); }
    function paintFiles() {
      var c = check(), many = st.files.length > 1;
      caseLabel.textContent = many ? '名称前缀' : '病例名称';
      caseId.placeholder = many ? '可选：前缀-文件名' : (st.files[0] ? stem(st.files[0].name) : '默认用文件名');
      var total = st.files.reduce(function (s, f) { return s + (Number(f.size) || 0); }, 0);
      fileText.textContent = !st.files.length ? '选择或拖入 STL，可以多选' : st.files.length === 1 ? st.files[0].name + '（' + formatBytes(st.files[0].size) + '）' : st.files.length + ' 个文件（' + formatBytes(total) + '）';
      nameWarn.hidden = !(st.files.length === 1 && c.rows[0] && c.rows[0].warn === '文件名可能是姓名') && !(caseId.value && looksLikeName(caseId.value));
      fileList.hidden = st.files.length < 2 && !c.invalid;
      ui().fill(fileList, c.rows.map(function (r) {
        var res = st.results[r.index];
        return h('div', {'class': 'up-file' + (r.ok ? (r.warn ? ' warn' : '') : ' bad') + (res && res.state ? ' ' + res.state : ''), title: r.name},
          h('span', {'class': 'up-file-name', text: r.name}), h('span', {'class': 'up-file-size', text: formatBytes(r.size)}),
          h('span', {'class': 'up-file-state', text: res ? res.text : (r.reason || r.warn || '')}), res && res.actions ? h('span', {'class': 'up-file-act'}, res.actions) : null);
      }));
      submit.disabled = st.busy || !c.valid.length || !opts.length;
      return c;
    }
    function setFiles(list, append) {
      var arr = Array.prototype.slice.call(list || []).filter(Boolean);
      st.files = append ? st.files.concat(arr) : arr;
      st.results = [];
      progress.hidden = true;
      paintFiles();
    }
    function setFile(f) { setFiles(f ? [f] : []); }
    file.addEventListener('change', function () { setFiles(file.files); });
    drop.addEventListener('dragover', function (ev) { if (ev.preventDefault) ev.preventDefault(); if (ev.stopPropagation) ev.stopPropagation(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', function () { drop.classList.remove('over'); });
    drop.addEventListener('drop', function (ev) {
      if (ev.preventDefault) ev.preventDefault(); if (ev.stopPropagation) ev.stopPropagation(); drop.classList.remove('over');
      var fs = Array.prototype.slice.call((ev.dataTransfer && ev.dataTransfer.files) || []).filter(function (f) { return /\.stl$/i.test(f.name); });
      if (fs.length) setFiles(fs, st.files.length > 0 && !st.busy); else ui().toast('只接受 .stl 文件。', {kind: 'error'});
    });
    [caseId, patient].forEach(function (inp) { inp.addEventListener('input', function () { if (st.files.length) paintFiles(); else nameWarn.hidden = !(caseId.value && looksLikeName(caseId.value)); }); });

    var submit = ui().button('上传并检查', function () { send(); }, {kind: 'primary', disabled: true});
    var cancelBtn = ui().button('取消', function () { ui().dialog.close('cancel'); });
    function common() {
      var ids = String(chosen || '').split('+').filter(Boolean);
      var m = {patient_id: patient.value.trim(), scan_date: scanDate.value || '', scan_label: scanLabel.value.trim(), tags: parseTags(tags.value).join(','),
        notes: notes.value.trim(), units: units.value || 'auto', release_id: ids[0] || '', companion_release_ids: ids.slice(1).join(',')};
      if (full) { m.device = device.value || 'auto'; m.seed_count = seeds.value || 'all'; m.threads = String(threads.value || '').trim(); }
      return m;
    }
    // ``many``: a batch request; ``fromBatch``: one file of a batch sent again (it keeps the batch's 「前缀-文件名」).
    function formData(files, many, mode, fromBatch) {
      var data = new root.FormData();
      files.forEach(function (f) { data.append('stl', f); });
      var m = common();
      Object.keys(m).forEach(function (k) { if (m[k] !== '') data.append(k, m[k]); });
      if (many) data.append('metadata_json', JSON.stringify(files.map(function (f) { return {case_id: caseIdFor(caseId.value, f, true)}; })));
      else if (caseIdFor(caseId.value, files[0], Boolean(fromBatch))) data.append('case_id', caseIdFor(caseId.value, files[0], Boolean(fromBatch)));
      data.append('on_duplicate', mode || 'ask');
      return data;
    }
    function rememberPrefs() {
      if (!ns.admin || !ns.admin.savePrefs) return;
      var m = common();
      ns.admin.savePrefs({upload: {units: m.units, release_id: chosen, device: full ? m.device : (up.device || 'auto'), seed_count: full ? m.seed_count : (up.seed_count || 'all'),
        threads: full ? m.threads : (up.threads || ''), remember_patient: Boolean(remember.checked),
        last_patient: {patient_id: m.patient_id, scan_label: m.scan_label, scan_date: m.scan_date, tags: tags.value.trim()}}});
    }
    var sizeText = function (loaded, total) { return formatBytes(loaded) + ' / ' + formatBytes(total); };
    function say(text) { progress.hidden = false; progress.replaceChildren(h('span', {text: text})); }
    function sendOne(f, mode, label, fromBatch) {
      return api().upload(formData([f], false, mode, fromBatch), {onProgress: function (loaded, total) {
        say(loaded === null ? label + '：已上传，服务正在检查几何…' : label + '：' + Math.round(100 * loaded / (total || 1)) + '%（' + sizeText(loaded, total) + '）');
      }});
    }
    function send() {
      if (st.busy) return;
      var c = paintFiles(), files = c.valid;
      if (!files.length) return;
      st.busy = true; submit.disabled = true;
      var idx = c.rows.filter(function (r) { return r.ok; }).map(function (r) { return r.index; });
      st.results = [];
      if (files.length === 1) {
        say('正在上传 ' + files[0].name);
        sendOne(files[0], 'ask', '正在上传 ' + files[0].name).then(function (result) {
          st.busy = false; rememberPrefs();
          var job = result.job || result, id = job.id || result.job_id;
          ui().dialog.close('done');
          ui().toast('已建立任务' + (result.reused_from ? '（复用已确认的出口）' : '') + '，正在检查输入。', {kind: 'ok'});
          if (ctx.onCreated) ctx.onCreated(id);
        }, function (error) {
          st.busy = false; submit.disabled = false;
          if (error.status === 409 && error.body && error.body.duplicate) { duplicate(error.body, files[0]); return; }
          say(error.message + (error.status === 413 ? ' 可以先在建模软件里抽稀网格。' : ''));
        });
        return;
      }
      sendBatch(files, idx);
    }
    // Several files: one request per chunk (≤ max_batch_files and ≤ 15/16 of max_batch_bytes); each file's row shows
    // its outcome; duplicates are resolved one by one afterwards.
    function sendBatch(files, idx) {
      var L = limits(), created = [], failures = 0, dups = [], start = 0, chain = Promise.resolve();
      var mark = function (k, state, text, actions) { st.results[idx[k]] = {state: state, text: text, actions: actions || null}; paintFiles(); };
      uploadChunks(files, {maxFiles: L.maxFiles, maxBytes: L.batchBytes}).forEach(function (chunk) {
        chain = chain.then(function () {
          var first = start; start += chunk.length;
          chunk.forEach(function (_, i) { mark(first + i, 'busy', '上传中…'); });
          var label = '批量上传 ' + (first + 1) + '–' + (first + chunk.length) + ' / ' + files.length;
          say(label);
          return api().upload(formData(chunk, true, 'ask'), {url: '/api/jobs/batch', timeout: 900000, onProgress: function (loaded, total) {
            say(loaded === null ? label + '：已上传，服务正在检查几何…' : label + '：' + Math.round(100 * loaded / (total || 1)) + '%（' + sizeText(loaded, total) + '）');
          }}).then(function (r) {
            var seen = {};
            ((r && r.results) || []).forEach(function (it) {
              var k = first + Number(it.index); seen[it.index] = true;
              if (it.duplicate) { dups.push({k: k, body: it, file: chunk[it.index]}); mark(k, 'wait', '这份几何已经算过'); return; }
              var job = it.job || {};
              if (job.id) { created.push(job.id); mark(k, 'ok', '已建立任务' + (job.reused_from || it.reused_from ? '（复用出口确认）' : '')); }
              else { failures += 1; mark(k, 'bad', (it.error && it.error.message) || '没有建立任务'); }
            });
            chunk.forEach(function (_, i) { if (!seen[i]) { failures += 1; mark(first + i, 'bad', '服务没有返回这个文件的结果'); } });
          }, function (e) {
            chunk.forEach(function (_, i) { failures += 1; mark(first + i, 'bad', e.message + (e.status === 413 ? '（可以分批上传）' : '')); });
          });
        });
      });
      chain.then(function () {
        return dups.reduce(function (p, d) { return p.then(function () { return resolveDuplicate(d, mark, created); }); }, Promise.resolve());
      }).then(function () {
        st.busy = false; rememberPrefs();
        var skipped = files.length - created.length - failures;
        say('已建立 ' + created.length + ' 个任务' + (failures ? '，' + failures + ' 个失败' : '') + (skipped > 0 ? '，' + skipped + ' 个跳过' : '') + '。');
        if (created.length && ctx.onCreated && !failures) {
          ui().dialog.close('done');
          ui().toast('已建立 ' + created.length + ' 个任务，正在检查输入。', {kind: 'ok', actions: ns.admin && ns.admin.openTasks ? [{label: '看任务列表', run: function () { ns.admin.openTasks({quick: 'today'}); }}] : []});
          ctx.onCreated(created[created.length - 1]);
        } else {
          submit.disabled = true; cancelBtn.replaceChildren(h('span', {text: '关闭'}));
          if (created.length && ctx.onRefresh) ctx.onRefresh();
        }
      });
    }
    function resolveDuplicate(d, mark, created) {
      return new Promise(function (resolve) {
        var body = d.body, existing = Array.isArray(body.existing) ? body.existing : [], first = existing[0];
        var finish = function (text, state) { mark(d.k, state || 'skip', text); resolve(); };
        var act = function (mode) {
          mark(d.k, 'busy', mode === 'reuse' ? '只算所选模型…' : '重新计算…');
          sendOne(d.file, mode, d.file.name, true).then(function (r) { var j = r.job || r; created.push(j.id || r.job_id); finish('已建立任务', 'ok'); },
            function (e) { finish(e.body && e.body.reuse_unavailable ? '已有任务不能复用，请选重新计算' : e.message, 'bad'); });
        };
        var btn = function (label, fn) { return ui().button(label, fn, {kind: 'link', cls: 'btn-sm'}); };
        say('「' + d.file.name + '」已经算过：选择怎么处理。');
        mark(d.k, 'wait', '已经算过', [
          first ? btn('打开已有', function () { finish('已打开已有结果'); if (ctx.onOpen) ctx.onOpen(first.job_id); }) : null,
          body.reusable ? btn('只算所选模型', function () { act('reuse'); }) : null,
          btn('重新计算', function () { act('force'); }),
          btn('跳过', function () { finish('已跳过'); })]);
      });
    }
    function duplicate(body, f) {
      var existing = Array.isArray(body.existing) ? body.existing : [];
      var first = existing[0] || null;
      var rows = existing.slice(0, 6).map(function (e) {
        return h('li', {text: [e.case_id || '', ui().resultName({model_release: {id: e.release_id}, family: e.family}, ctx.cards), ui().statusInfo(e.status).label, ui().time(e.created_at)].filter(Boolean).join(' · ')});
      });
      progress.hidden = false;
      progress.replaceChildren(
        h('strong', {text: '这份几何已经算过。'}), h('ul', {'class': 'dup-list'}, rows),
        h('span', {'class': 'dup-actions'},
          first ? ui().button('打开已有结果', function () { ui().dialog.close('done'); if (ctx.onOpen) ctx.onOpen(first.job_id); }, {cls: 'btn-sm'}) : null,
          body.reusable ? ui().button('只算所选模型（沿用出口确认）', function () { resend(f, 'reuse'); }, {cls: 'btn-sm'}) : null,
          ui().button('重新计算', function () { resend(f, 'force'); }, {cls: 'btn-sm'})));
    }
    function resend(f, mode) {
      if (st.busy) return;
      st.busy = true; submit.disabled = true;
      sendOne(f, mode, '正在上传 ' + f.name).then(function (result) {
        st.busy = false; rememberPrefs();
        var job = result.job || result;
        ui().dialog.close('done');
        ui().toast('已建立任务' + (result.reused_from ? '（复用已确认的出口）' : '') + '，正在检查输入。', {kind: 'ok'});
        if (ctx.onCreated) ctx.onCreated(job.id || result.job_id);
      }, function (error) {
        st.busy = false; submit.disabled = false;
        say(error.body && error.body.reuse_unavailable ? '已有任务不能复用，请选「重新计算」。' : error.message);
      });
    }
    var help = h('div', {'class': 'req'},
      h('ol', {'class': 'req-list'}, REQUIREMENTS.map(function (t) { return h('li', {text: t}); })),
      h('div', {'class': 'req-links'}, ui().link('输入说明', '/static/v2/help_input.html', {newTab: true}), ui().link('一页操作卡', '/static/v2/help_quickstart.html', {newTab: true}),
        h('a', {'class': 'lnk', href: '/static/v2/example_aaa.stl', download: 'example_aaa.stl', text: '下载示例 STL'})));
    var grid = h('div', {'class': 'form-grid'},
      h('label', {'class': 'fld'}, h('span', {text: '单位'}), units),
      h('label', {'class': 'fld'}, caseLabel, caseId),
      h('label', {'class': 'fld'}, h('span', {}, '患者编号', h('em', {'class': 'fld-tag', text: '随访需要'})), patient),
      h('label', {'class': 'fld'}, h('span', {}, '扫描日期', h('em', {'class': 'fld-tag', text: '随访需要'})), scanDate));
    var more = h('details', {'class': 'up-more', open: Boolean(scanLabel.value || tags.value)},
      h('summary', {text: '扫描标签、标签、备注'}),
      h('div', {'class': 'form-grid'}, h('label', {'class': 'fld'}, h('span', {text: '扫描标签'}), scanLabel), h('label', {'class': 'fld'}, h('span', {text: '标签'}), tags)),
      h('label', {'class': 'fld'}, h('span', {text: '备注'}), notes));
    var compute = full ? h('details', {'class': 'up-more'}, h('summary', {text: '计算设置'}),
      h('div', {'class': 'form-grid form-grid-3'}, h('label', {'class': 'fld'}, h('span', {text: '设备'}), device), h('label', {'class': 'fld'}, h('span', {text: '集成模型数'}), seeds),
        h('label', {'class': 'fld'}, h('span', {text: 'CPU 线程'}), threads))) : null;
    ui().dialog.open({title: '上传 STL', wide: true, cls: 'dlg-upload', body: [
      help, file, drop, fileList, nameWarn,
      h('h4', {text: '想看什么'}), choiceList,
      grid,
      h('label', {'class': 'check up-remember'}, remember, h('span', {text: '记住患者编号、扫描信息和标签'}), ui().infoTip('同一患者的多次扫描填同一个患者编号和各自的扫描日期，概览里会出现随访表。勾选后下次上传自动填好。')),
      more, compute,
      progress],
      actions: [cancelBtn, submit],
      onClose: function () { if (current === ctl) current = null; }});
    if (Array.isArray(ctx.files) && ctx.files.length) setFiles(ctx.files);
    var ctl = {setFile: setFile, setFiles: setFiles, addFiles: function (fs) { if (!st.busy) setFiles(fs, true); }, send: send, choices: opts, busy: function () { return st.busy; }};
    current = ctl;
    return ctl;
  }

  // ------------------------------------------------------------------ whole-page drop (classic A7)
  var dropState = {installed: false, depth: 0, getCtx: null};
  function hasFiles(ev) {
    var types = ev && ev.dataTransfer && ev.dataTransfer.types;
    if (!types) return false;
    for (var i = 0; i < types.length; i++) if (types[i] === 'Files') return true;
    return false;
  }
  function overlay(on) {
    var d = root.document, el = d.getElementById('wsu-drop');
    if (!el && on) {
      el = ui().h('div', {id: 'wsu-drop', 'class': 'wsu-drop', 'aria-hidden': 'true'},
        ui().h('div', {'class': 'wsu-drop-box'}, ui().icon('upload', {size: 28}), ui().h('strong', {text: '松开，上传 STL'}), ui().h('span', {text: '可以一次放多个文件'})));
      if (d.body) d.body.appendChild(el);
    }
    if (el) el.hidden = !on;
  }
  function reloginOpen() { var r = root.document.getElementById('ws-relogin'); return Boolean(r && r.open); }
  function dropFiles(list) {
    var all = Array.prototype.slice.call(list || []);
    var stl = all.filter(function (f) { return /\.stl$/i.test(f.name || ''); });
    if (!stl.length) { ui().toast(all.length ? '只接受 .stl 文件。' : '没有读到拖入的文件。', {kind: 'error'}); return false; }
    if (current) { if (current.busy()) { ui().toast('正在上传，完成后再拖入。', {kind: 'info'}); return false; } current.addFiles(stl); }
    else if (ui().dialog.isOpen()) { ui().toast('先关闭当前对话框，再拖入 STL。', {kind: 'info'}); return false; }
    else open(Object.assign({}, dropState.getCtx ? dropState.getCtx() : {}, {files: stl}));
    if (stl.length < all.length) ui().toast('已忽略 ' + (all.length - stl.length) + ' 个非 STL 文件。', {kind: 'info', ms: 4000});
    return true;
  }
  function installDrop(getCtx) {
    dropState.getCtx = getCtx || dropState.getCtx;
    if (dropState.installed || !root.document || !root.document.addEventListener) return;
    dropState.installed = true;
    var d = root.document;
    var usable = function (ev) { return hasFiles(ev) && !(ns.env && ns.env.offline) && !reloginOpen(); };
    d.addEventListener('dragenter', function (ev) { if (!usable(ev)) return; if (ev.preventDefault) ev.preventDefault(); dropState.depth += 1; overlay(true); });
    d.addEventListener('dragover', function (ev) { if (!usable(ev)) return; if (ev.preventDefault) ev.preventDefault(); if (ev.dataTransfer) ev.dataTransfer.dropEffect = 'copy'; });
    d.addEventListener('dragleave', function (ev) { if (!hasFiles(ev)) return; dropState.depth = Math.max(0, dropState.depth - 1); if (!dropState.depth || !ev.relatedTarget) { dropState.depth = 0; overlay(false); } });
    d.addEventListener('drop', function (ev) {
      if (!hasFiles(ev)) return;
      if (ev.preventDefault) ev.preventDefault();
      dropState.depth = 0; overlay(false);
      if (!usable(ev)) return;
      dropFiles(ev.dataTransfer.files);
    });
  }

  return {open: open, choices: choices, looksLikeName: looksLikeName, validationLine: validationLine, REQUIREMENTS: REQUIREMENTS,
    checkFiles: checkFiles, uploadChunks: uploadChunks, caseIdFor: caseIdFor, parseTags: parseTags, formatBytes: formatBytes, installDrop: installDrop, dropFiles: dropFiles,
    current: function () { return current; }};
});
