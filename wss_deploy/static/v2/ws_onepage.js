/* WSS workspace v2 — Lane B (S5b): the institution report template of the one-page report.
 * User menu → 「报告模板…」 opens the classic template dialog (app.js templateDialog) in the workspace look: the same
 * service calls (GET / PUT /api/report-template), the same payload {institution, department, report_title, footer_note,
 * signature_lines, show_glossary, appendix} and the same rules (report_template.validate), plus a small live sketch of
 * where each setting lands on the printed page.  The one-pager itself is rendered by the service (onepager.py).
 * Registered through the shell extension registry (ws_shell.js 「extensions」); offline it adds nothing. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.onepage = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var URL_TEMPLATE = '/api/report-template';
  // report_template.py: DEFAULTS and the text limits (signature lines: ≤ 4 items, ≤ 20 characters each).
  var DEFAULTS = {institution: '', department: '', report_title: '', footer_note: '', signature_lines: ['报告人', '审阅人'], show_glossary: 'used', appendix: true};
  var LIMITS = {institution: 120, department: 120, report_title: 80, footer_note: 400, lines: 4, line: 20};
  var GLOSSARY = [{value: 'used', label: '只列本页用到的术语'}, {value: 'all', label: '列出全部术语'}, {value: 'none', label: '不印术语表'}];
  var STATEMENT = '仅供研究参考，不作临床诊断依据';     // onepager.FOOTER_STATEMENT

  // ------------------------------------------------------------------ pure helpers (Node-tested)
  // The classic WB.parseTags: comma (half / full width), 、 and new lines separate; blanks and repeats are dropped.
  function parseLines(text) {
    var out = [];
    String(text === null || text === undefined ? '' : text).split(/[,，、\n]/).forEach(function (part) {
      var t = part.trim();
      if (t && out.indexOf(t) < 0) out.push(t);
    });
    return out;
  }
  function linesText(lines) { return Array.isArray(lines) ? lines.join('，') : ''; }
  // Form values → the classic PUT body (same keys, same trimming as app.js templateDialog).
  function payload(v) {
    return {institution: String(v.institution || '').trim(), department: String(v.department || '').trim(), report_title: String(v.report_title || '').trim(),
      footer_note: String(v.footer_note || '').trim(), signature_lines: parseLines(v.signature_lines),
      show_glossary: GLOSSARY.some(function (g) { return g.value === v.show_glossary; }) ? v.show_glossary : 'used', appendix: v.appendix !== false};
  }
  // Checked here with the service's wording so the dialog can say it before sending; the service validates again.
  function problem(body) {
    if (body.signature_lines.length > LIMITS.lines || body.signature_lines.some(function (x) { return x.length > LIMITS.line; })) return '签字栏最多 4 项，每项不超过 20 个字符。';
    if (body.footer_note.length > LIMITS.footer_note) return '页脚声明不能超过 ' + LIMITS.footer_note + ' 个字符。';
    return null;
  }
  function errorText(e) {
    if (!e) return '保存没有完成。';
    if (e.status === 403) return '没有修改权限：只有管理员可以修改。';
    if (e.status === 404) return '当前服务版本没有报告模板设置。';
    return e.message || '保存没有完成。';
  }

  // ------------------------------------------------------------------ the page sketch (where each setting prints)
  function sketch(h) {
    var el = {inst: h('div', {'class': 'otp-inst'}), title: h('div', {'class': 'otp-title'}), sign: h('div', {'class': 'otp-sign'}), foot: h('div', {'class': 'otp-foot'})};
    var body = h('div', {'class': 'otp-bars'}, h('span', {'class': 'otp-bar w90'}), h('span', {'class': 'otp-bar w70'}),
      h('span', {'class': 'otp-kpis'}, h('i'), h('i'), h('i'), h('i')), h('span', {'class': 'otp-shots'}, h('i'), h('i'), h('i')),
      h('span', {'class': 'otp-bar w80'}), h('span', {'class': 'otp-bar w60'}));
    el.back = h('div', {'class': 'otp-back'});
    var sheet = h('div', {'class': 'otp-sheet'}, h('div', {'class': 'otp-head'}, el.inst, el.title), body, el.sign, el.foot);
    el.root = h('div', {'class': 'otp-preview', 'aria-hidden': 'true'}, h('div', {'class': 'otp-stack'}, el.back, sheet), h('div', {'class': 'otp-cap', text: '预览 · 全部任务共用'}));
    return el;
  }
  function drawSketch(h, el, body) {
    var inst = [body.institution, body.department].filter(Boolean).join(' · ');
    el.inst.textContent = inst; el.inst.hidden = !inst;
    el.title.textContent = body.report_title || '按结果类型自动';
    el.title.classList.toggle('auto', !body.report_title);
    el.sign.replaceChildren();
    body.signature_lines.slice(0, LIMITS.lines).forEach(function (line) {
      el.sign.appendChild(h('span', {'class': 'otp-line'}, h('b', {text: line}), h('i'), h('b', {text: '日期'}), h('i', {'class': 'short'})));
    });
    el.sign.hidden = !body.signature_lines.length;
    el.foot.textContent = STATEMENT + (body.footer_note ? '；' + body.footer_note.split('\n').join(' ') : '');
    el.back.hidden = !body.appendix;
  }

  // ------------------------------------------------------------------ dialog
  function openTemplate(shell) {
    var ui = shell.ui();
    return shell.api().request(URL_TEMPLATE).then(function (data) { return showDialog(shell, data || {}); }, function (e) {
      ui.toast(e && e.status === 404 ? errorText(e) : ((e && e.message) || '读取报告模板没有完成。'), {kind: 'error'});
      return null;
    });
  }
  function showDialog(shell, data) {
    var ui = shell.ui(), api = shell.api(), h = ui.h;
    var t = Object.assign({}, DEFAULTS, data.template || {}), editable = data.editable !== false;
    var input = function (key, label, max, placeholder) {
      var el = h('input', {type: 'text', maxLength: max, value: t[key] || '', placeholder: placeholder, 'aria-label': label, autocomplete: 'off', disabled: !editable});
      return {el: el, field: h('label', {'class': 'fld'}, h('span', {text: label}), el)};
    };
    var inst = input('institution', '机构名称', LIMITS.institution, '例如 某某医院');
    var dept = input('department', '科室', LIMITS.department, '例如 血管外科');
    var title = input('report_title', '报告标题', LIMITS.report_title, '按结果类型自动');
    var sign = h('input', {type: 'text', maxLength: 200, value: linesText(t.signature_lines), placeholder: '留空则不印签字栏', 'aria-label': '签字栏', autocomplete: 'off', disabled: !editable});
    var foot = h('textarea', {rows: 2, maxLength: LIMITS.footer_note, 'aria-label': '页脚声明', disabled: !editable});
    foot.value = t.footer_note || '';
    var gloss = ui.select(GLOSSARY, GLOSSARY.some(function (g) { return g.value === t.show_glossary; }) ? t.show_glossary : 'used', null, {'aria-label': '术语表', disabled: !editable});
    var appx = h('input', {type: 'checkbox', checked: t.appendix !== false, 'aria-label': '打印附录页', disabled: !editable});
    var status = h('p', {'class': editable ? 'note' : 'note note-warn', role: 'status', text: editable ? '' : '只能查看：修改需要管理员。'});
    status.hidden = editable;
    var values = function () {
      return payload({institution: inst.el.value, department: dept.el.value, report_title: title.el.value, footer_note: foot.value,
        signature_lines: sign.value, show_glossary: gloss.value, appendix: appx.checked});
    };
    var pv = sketch(h);
    var redraw = function () { drawSketch(h, pv, values()); };
    [inst.el, dept.el, title.el, sign, foot].forEach(function (el) { el.addEventListener('input', redraw); });
    [gloss, appx].forEach(function (el) { el.addEventListener('change', redraw); });
    redraw();

    var form = h('div', {'class': 'otp-form'},
      h('div', {'class': 'otp-pair'}, inst.field, dept.field), title.field,
      h('label', {'class': 'fld'}, h('span', {'class': 'otp-lab'}, '签字栏', ui.infoTip('逗号分隔，最多 4 项；每项印成「姓名 ____ 日期 ____」。')), sign),
      h('label', {'class': 'fld'}, h('span', {'class': 'otp-lab'}, '页脚声明', ui.infoTip('接在「' + STATEMENT + '」之后。')), foot),
      h('div', {'class': 'otp-pair otp-opts'}, h('label', {'class': 'fld'}, h('span', {text: '术语表'}), gloss),
        h('div', {'class': 'otp-appx'}, h('label', {'class': 'check'}, appx, h('span', {text: '打印附录页'})),
          ui.infoTip('分支统计、发现详情、输入检查、局限性声明等，打印时另起一页。'))),
      status);

    var dlg = null;
    var save = ui.button('保存', function () {
      var body = values(), bad = problem(body);
      if (bad) { status.hidden = false; status.className = 'note note-error'; status.textContent = bad; return; }
      save.disabled = true; status.hidden = false; status.className = 'note'; status.textContent = '正在保存…';
      api.request(URL_TEMPLATE, {method: 'PUT', body: body}).then(function () {
        if (dlg) dlg.close('done');
        var cur = shell.cur(), jobId = cur && cur.manifest && cur.jobId;
        ui.toast('报告模板已保存，新打开的一页纸会用它。', {kind: 'info', actions: jobId && api.urls ? [{label: '打开一页纸', run: function () {
          if (typeof root.open === 'function') root.open(api.urls.onepage(jobId), '_blank', 'noopener');
        }}] : []});
      }, function (e) {
        save.disabled = false; status.hidden = false; status.className = 'note note-error'; status.textContent = errorText(e);
      });
    }, {kind: 'primary', disabled: !editable});
    var reset = editable ? ui.button('恢复默认', function () {
      inst.el.value = ''; dept.el.value = ''; title.el.value = ''; foot.value = ''; sign.value = linesText(DEFAULTS.signature_lines);
      gloss.value = DEFAULTS.show_glossary; appx.checked = DEFAULTS.appendix; redraw();
    }, {kind: 'link', title: '填回默认设置；保存后才生效'}) : null;
    dlg = ui.dialog.open({title: '报告模板', cls: 'dlg-onepage', body: [h('div', {'class': 'otp'}, form, pv.root)],
      actions: [reset, reset ? h('span', {'class': 'sec-fill'}) : null, ui.button(editable ? '取消' : '关闭', function () { ui.dialog.close('cancel'); }), editable ? save : null].filter(Boolean),
      focus: editable ? inst.el : null});
    return dlg;
  }

  // ------------------------------------------------------------------ shell hook (user menu, before 「经典工作台」)
  function menuItems(shell) {
    if (!shell || shell.offline() || !shell.api()) return [];
    return [{label: '报告模板…', run: function () { openTemplate(shell); }}];
  }
  (ns.ext = ns.ext || []).push({id: 'onepage', userMenu: menuItems});

  return {parseLines: parseLines, linesText: linesText, payload: payload, problem: problem, errorText: errorText, menuItems: menuItems,
    openTemplate: openTemplate, DEFAULTS: DEFAULTS};
});
