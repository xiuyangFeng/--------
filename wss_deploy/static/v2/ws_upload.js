/* WSS workspace v2 — upload dialog (contract §6.3 ws_upload.js, U1 / U14 / U17 / C7).
 * Four sentences of input requirements with the help pages and the sample STL; 「想看什么」 lists model-card names
 * and one validation line (never a release code); patient number and scan date are marked 「随访需要」; a file
 * name that looks like a person's name gets a reminder.  Duplicate handling follows the existing service flow. */
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
        purpose: card && card.purpose || '', isDefault: Boolean(r.default), kind: releaseKind(r)};
    });
    var cycles = list.filter(function (r) { return releaseKind(r) === 'cycle'; }), vols = list.filter(function (r) { return releaseKind(r) === 'volume'; });
    cycles.forEach(function (c) {
      vols.forEach(function (v) {
        var cc = cards[releaseId(c)], vc = cards[releaseId(v)];
        out.push({value: releaseId(c) + '+' + releaseId(v), label: ((cc && cc.short_name) || '周期指标') + ' ＋ ' + ((vc && vc.short_name) || '体内压力与速度'),
          detail: '同时建两个任务，出口只确认一次', purpose: '', isDefault: false, kind: 'combo'});
      });
    });
    return out;
  }

  function open(ctx) {
    ctx = ctx || {};
    var h = ui().h;
    var st = {file: null, busy: false};
    var opts = choices(ctx.releases || [], ctx.cards || {});
    var chosen = (opts.filter(function (o) { return o.isDefault; })[0] || opts[0] || {}).value || '';
    var file = h('input', {type: 'file', accept: '.stl', 'class': 'visually-hidden', id: 'wsu-file', 'aria-label': '选择 STL 文件'});
    var fileText = h('span', {'class': 'drop-text', text: '选择或拖入一个 STL 文件'});
    var drop = h('label', {'class': 'drop', 'for': 'wsu-file'}, ui().icon('upload'), fileText);
    var nameWarn = h('p', {'class': 'note note-warn', hidden: true, text: '文件名可能是姓名。建议填写患者编号：界面、报告和导出会优先显示患者编号。'});
    var caseId = h('input', {type: 'text', maxLength: 160, placeholder: '默认用文件名', 'aria-label': '病例名称'});
    var patient = h('input', {type: 'text', maxLength: 80, placeholder: '如 P-001', 'aria-label': '患者编号'});
    var scanDate = h('input', {type: 'date', 'aria-label': '扫描日期'});
    var units = ui().select([{value: 'auto', label: '自动判断'}, {value: 'mm', label: '毫米 mm'}, {value: 'cm', label: '厘米 cm'}, {value: 'm', label: '米 m'}], 'auto', null, {'aria-label': '单位'});
    var progress = h('p', {'class': 'upload-progress', 'aria-live': 'polite', hidden: true});
    var radioName = 'wsu-release';
    var choiceList = h('div', {'class': 'choices', role: 'radiogroup', 'aria-label': '想看什么'});
    if (!opts.length) choiceList.appendChild(ui().note('服务上没有可用的模型，暂时不能上传。', 'error'));
    opts.forEach(function (o) {
      var input = h('input', {type: 'radio', name: radioName, value: o.value, checked: o.value === chosen});
      input.addEventListener('change', function () { if (input.checked) chosen = o.value; });
      choiceList.appendChild(h('label', {'class': 'choice'}, input, h('span', {'class': 'choice-text'}, h('span', {'class': 'choice-label', text: o.label}), h('span', {'class': 'choice-detail', text: o.detail}))));
    });
    function setFile(f) {
      st.file = f || null;
      var size = f && f.size ? (f.size >= 1048576 ? ui().sig(f.size / 1048576) + ' MB' : Math.max(1, Math.round(f.size / 1024)) + ' KB') : '';
      fileText.textContent = f ? f.name + (size ? '（' + size + '）' : '') : '选择或拖入一个 STL 文件';
      nameWarn.hidden = !(f && looksLikeName(f.name));
      if (f && !caseId.value) caseId.placeholder = stem(f.name);
      submit.disabled = !f || !opts.length;
    }
    file.addEventListener('change', function () { setFile(file.files && file.files[0]); });
    drop.addEventListener('dragover', function (ev) { if (ev.preventDefault) ev.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', function () { drop.classList.remove('over'); });
    drop.addEventListener('drop', function (ev) {
      if (ev.preventDefault) ev.preventDefault(); drop.classList.remove('over');
      var f = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
      if (f && /\.stl$/i.test(f.name)) setFile(f); else ui().toast('只接受一个 .stl 文件。', {kind: 'error'});
    });
    caseId.addEventListener('input', function () { nameWarn.hidden = !((st.file && looksLikeName(st.file.name) && !caseId.value) || (caseId.value && looksLikeName(caseId.value))); });

    var submit = ui().button('上传并检查', function () { send('ask'); }, {kind: 'primary', disabled: true});
    function formData(mode) {
      var ids = String(chosen || '').split('+').filter(Boolean);
      var data = new root.FormData();
      data.append('stl', st.file);
      if (caseId.value.trim()) data.append('case_id', caseId.value.trim());
      if (patient.value.trim()) data.append('patient_id', patient.value.trim());
      if (scanDate.value) data.append('scan_date', scanDate.value);
      data.append('units', units.value || 'auto');
      if (ids[0]) data.append('release_id', ids[0]);
      if (ids.length > 1) data.append('companion_release_ids', ids.slice(1).join(','));
      data.append('on_duplicate', mode);
      return data;
    }
    function send(mode) {
      if (!st.file || st.busy) return;
      st.busy = true; submit.disabled = true; progress.hidden = false; progress.textContent = '正在上传 ' + st.file.name;
      api().upload(formData(mode), {onProgress: function (loaded, total) {
        progress.textContent = loaded === null ? '已上传，服务正在检查几何…' : '正在上传 ' + Math.round(100 * loaded / (total || 1)) + '%';
      }}).then(function (result) {
        var job = result.job || result;
        var id = job.id || result.job_id;
        st.busy = false;
        ui().dialog.close('done');
        ui().toast('已建立任务' + (result.reused_from ? '（复用已确认的出口）' : '') + '，正在检查输入。', {kind: 'ok'});
        if (ctx.onCreated) ctx.onCreated(id);
      }, function (error) {
        st.busy = false; submit.disabled = false;
        if (error.status === 409 && error.body && error.body.duplicate) { duplicate(error.body); return; }
        progress.textContent = error.message + (error.status === 413 ? ' 可以先在建模软件里抽稀网格。' : '');
      });
    }
    function duplicate(body) {
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
          body.reusable ? ui().button('只算所选模型（沿用出口确认）', function () { send('reuse'); }, {cls: 'btn-sm'}) : null,
          ui().button('重新计算', function () { send('force'); }, {cls: 'btn-sm'})));
    }
    var help = h('div', {'class': 'req'},
      h('ol', {'class': 'req-list'}, REQUIREMENTS.map(function (t) { return h('li', {text: t}); })),
      h('div', {'class': 'req-links'}, ui().link('输入说明', '/static/v2/help_input.html', {newTab: true}), ui().link('一页操作卡', '/static/v2/help_quickstart.html', {newTab: true}),
        h('a', {'class': 'lnk', href: '/static/v2/example_aaa.stl', download: 'example_aaa.stl', text: '下载示例 STL'})));
    var grid = h('div', {'class': 'form-grid'},
      h('label', {'class': 'fld'}, h('span', {text: '单位'}), units),
      h('label', {'class': 'fld'}, h('span', {text: '病例名称'}), caseId),
      h('label', {'class': 'fld'}, h('span', {}, '患者编号', h('em', {'class': 'fld-tag', text: '随访需要'})), patient),
      h('label', {'class': 'fld'}, h('span', {}, '扫描日期', h('em', {'class': 'fld-tag', text: '随访需要'})), scanDate));
    ui().dialog.open({title: '上传 STL', wide: true, cls: 'dlg-upload', body: [
      help, file, drop, nameWarn,
      h('h4', {text: '想看什么'}), choiceList,
      grid,
      ui().note('同一患者的多次扫描填同一个患者编号和各自的扫描日期，概览里会出现随访表。'),
      progress],
      actions: [ui().button('取消', function () { ui().dialog.close('cancel'); }), submit]});
    return {setFile: setFile, send: send, choices: opts};
  }

  return {open: open, choices: choices, looksLikeName: looksLikeName, validationLine: validationLine, REQUIREMENTS: REQUIREMENTS};
});
