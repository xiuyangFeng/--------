/* WSS workspace v2 — DOM helpers (contract §6.3 ws_ui.js): h(), number formatting, status and tier labels,
 * dialog, menu, notice bar, tables.  Dynamic text always goes through textContent, never HTML. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.ui = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var doc = function () { return root.document; };

  // ------------------------------------------------------------------ element builder
  function h(tag, attrs) {
    var el = doc().createElement(tag);
    attrs = attrs || {};
    Object.keys(attrs).forEach(function (key) {
      var value = attrs[key];
      if (value === undefined || value === null || value === false) return;
      if (key === 'class') el.className = value;
      else if (key === 'text') el.textContent = String(value);
      else if (key === 'style' && typeof value === 'object') Object.keys(value).forEach(function (k) { el.style[k] = value[k]; });
      else if (key === 'dataset' && typeof value === 'object') Object.keys(value).forEach(function (k) { el.dataset[k] = String(value[k]); });
      else if (key.slice(0, 2) === 'on' && typeof value === 'function') el.addEventListener(key.slice(2), value);
      else if (key.indexOf('aria-') === 0 || key.indexOf('data-') === 0 || key === 'role' || key === 'for' || key === 'tabindex') el.setAttribute(key, String(value));
      else if (key in el) { try { el[key] = value; } catch (_) { el.setAttribute(key, String(value)); } }
      else el.setAttribute(key, String(value));
    });
    for (var i = 2; i < arguments.length; i++) appendKids(el, arguments[i]);
    return el;
  }
  function appendKids(el, kid) {
    if (kid === null || kid === undefined || kid === false || kid === true) return;
    if (Array.isArray(kid)) { kid.forEach(function (k) { appendKids(el, k); }); return; }
    if (typeof kid === 'object' && (kid.nodeType || kid.tagName || kid.nodeValue !== undefined)) { el.appendChild(kid); return; }
    el.appendChild(doc().createTextNode(String(kid)));
  }
  function clear(el) { if (el) el.replaceChildren(); return el; }
  // Replace the children; null / false are skipped and arrays are flattened (replaceChildren would print them).
  function fill(el) {
    if (!el) return el;
    el.replaceChildren();
    for (var i = 1; i < arguments.length; i++) appendKids(el, arguments[i]);
    return el;
  }
  function icon(name, opts) { return ns.icons ? ns.icons.icon(name, opts) : null; }

  // ------------------------------------------------------------------ numbers (§7: three significant figures, units)
  function sig(value, n) {
    n = n || 3;
    var v = Number(value);
    if (value === null || value === undefined || value === '' || !isFinite(v)) return '—';
    if (v === 0) return '0';
    var p = Number(v.toPrecision(n));
    if (p === 0) return '0';
    var d = Math.floor(Math.log10(Math.abs(p)));
    var decimals = Math.min(8, Math.max(0, n - 1 - d));
    return p.toFixed(decimals).replace('-', '−');   // typographic minus
  }
  // Thresholds and window bounds: 0.4 not 0.400.
  function trim(value) {
    var v = Number(value);
    if (value === null || value === undefined || !isFinite(v)) return '—';
    return String(Number(v.toPrecision(3)));
  }
  var UNIT_TEXT = {'1': '', '': '', 'Pa': 'Pa', '1/Pa': '1/Pa', 'm/s': 'm/s', 'mm': 'mm', 'mm2': 'mm²', 'mm^2': 'mm²', 'cm2': 'cm²', 'cm²': 'cm²', 'mL': 'mL', 'ml': 'mL'};
  function unitText(units) {
    if (units === null || units === undefined) return '';
    return Object.prototype.hasOwnProperty.call(UNIT_TEXT, units) ? UNIT_TEXT[units] : String(units);
  }
  function num(value, units, opts) {
    opts = opts || {};
    var text = opts.trim ? trim(value) : sig(value, opts.sig || 3);
    if (text === '—') return text;
    var u = unitText(units);
    return u ? text + ' ' + u : text;
  }
  function pct(frac) {
    var v = Number(frac);
    if (frac === null || frac === undefined || frac === '' || !isFinite(v)) return '—';
    var p = v * 100;
    if (p === 0) return '0%';
    if (Math.abs(p) < 1) return p.toFixed(1) + '%';
    return Math.round(p) + '%';
  }
  function range(lo, hi, units) {
    var u = unitText(units);
    // A dash between signed numbers reads as a minus sign (−1400–131): use 「至」 when the range goes below zero.
    var sep = Number(lo) < 0 || Number(hi) < 0 ? ' 至 ' : '–';
    return trim(lo) + sep + trim(hi) + (u ? ' ' + u : '');
  }
  function pad(n) { return n < 10 ? '0' + n : String(n); }
  function time(iso, withTime) {
    if (!iso) return '';
    var d = new Date(iso);
    if (isNaN(d.getTime())) return String(iso);
    var day = d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
    return withTime === false ? day : day + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
  }
  function duration(seconds) {
    var s = Number(seconds);
    if (seconds === null || seconds === undefined || !isFinite(s)) return '—';
    if (s < 60) return Math.max(1, Math.round(s)) + ' 秒';
    if (s < 3600) return Math.floor(s / 60) + ' 分 ' + Math.round(s % 60) + ' 秒';
    return Math.floor(s / 3600) + ' 小时 ' + Math.round(s % 3600 / 60) + ' 分';
  }

  // ------------------------------------------------------------------ status words (§7: text + 6 px dot)
  var STATUS = {
    queued: ['排队', 'idle'], queued_A: ['排队', 'idle'], queued_B: ['排队', 'idle'],
    running: ['计算中', 'busy'], running_A: ['计算中', 'busy'], running_B: ['计算中', 'busy'], cancelling: ['正在取消', 'busy'],
    awaiting_input: ['待核对输入', 'warn'], awaiting_confirmation: ['待确认出口', 'warn'], awaiting_outlets: ['待确认出口', 'warn'],
    done: ['已完成', 'ok'], failed: ['失败', 'error'], cancelled: ['已取消', 'idle'], interrupted: ['已中断', 'error']
  };
  function statusInfo(status) {
    var row = STATUS[status] || [status ? String(status) : '未知', 'idle'];
    return {key: status || 'unknown', label: row[0], tone: row[1]};
  }
  function reviewInfo(review) {
    var status = review && review.status;
    if (status === 'reviewed') return {key: 'reviewed', label: '已复核', tone: 'ok'};
    if (status === 'reopened') return {key: 'reopened', label: '已重新打开', tone: 'warn'};
    return {key: 'unreviewed', label: '未复核', tone: 'idle'};
  }
  function dot(tone, label, extra) {
    return h('span', {'class': 'st st-' + (tone || 'idle') + (extra ? ' ' + extra : '')}, h('span', {'class': 'st-dot', 'aria-hidden': 'true'}), h('span', {'class': 'st-text', text: label}));
  }
  function statusDot(status) { var s = statusInfo(status); return dot(s.tone, s.label); }
  function reviewDot(review) { var r = reviewInfo(review); return dot(r.tone, r.label); }
  var TIERS = {geometry: '几何', model: '模型', derived: '派生'};
  function tierTag(tier) {
    if (!TIERS[tier]) return null;
    return h('span', {'class': 'tier tier-' + tier, title: tierHelp(tier), text: TIERS[tier]});
  }
  function tierHelp(tier) {
    return {geometry: '从输入表面量出，不经过模型', model: '模型预测，一致性见模型说明', derived: '由预测量计算得到，没有单独验证'}[tier] || '';
  }

  // ------------------------------------------------------------------ identities (no release code in the reading text of a result;
  // the task list, the rerun choices and the result menus name the package — user request 2026-10-02, old and new packages
  // of one kind otherwise look the same)
  function releaseIdOf(job) {
    if (!job) return '';
    var r = job.model_release || {};
    return r.id || r.release || job.release_id || '';
  }
  function kindOf(job) {
    if (!job) return null;
    var r = job.model_release || {};
    var c = r.contract || {};
    var f = c.fields || {};
    if (c.protocol === 'single_frame_volume' || f.velocity || job.family === 'volume') return 'volume';
    if (c.protocol === 'single_frame_wss_cycle_multi' || f.tawss || f.osi || job.has_cycle) return 'cycle';
    if (c.protocol || f.wss) return 'wall';
    if (/^M1_/.test(releaseIdOf(job))) return 'cycle';
    return job.family === 'wall' ? 'wall' : (job.family || null);
  }
  var KIND_NAME = {volume: ['体内压力与速度', '体场'], cycle: ['周期指标 TAWSS · OSI', '周期指标'], wall: ['峰值 WSS', '峰值 WSS']};
  // The package a result (or a release record wrapped as {model_release: r}) came from: the release id without its seed
  // count and freeze date (as jobs.release_short) and the card's version date, e.g. 「M1cap_v52d · 10-02」.
  function packageName(job, cards) {
    var id = releaseIdOf(job);
    if (!id) return '';
    var short = String(id).replace(/_20\d{6}$/, '').replace(/_\d+seeds?(?=_|$)/, '') || String(id);
    var card = cards ? cards[id] : null;
    var date = card && typeof card.version_date === 'string' && /^\d{4}-\d{2}-\d{2}/.test(card.version_date) ? card.version_date.slice(5, 10) : '';
    return short + (date ? ' · ' + date : '');
  }
  function resultName(job, cards, opts) {
    var id = releaseIdOf(job);
    var card = cards && id ? cards[id] : null;
    var short = opts && opts.short;
    if (card) return short ? (card.short_name || card.display_name) : (card.display_name || card.short_name);
    var kind = kindOf(job);
    if (KIND_NAME[kind]) return KIND_NAME[kind][short ? 1 : 0];
    return '结果';
  }
  function displayName(job) {
    if (!job) return '';
    return job.display_name || job.patient_id || job.case_id || '未命名病例';
  }

  // ------------------------------------------------------------------ controls
  function button(label, onClick, opts) {
    opts = opts || {};
    var cls = 'btn' + (opts.kind ? ' btn-' + opts.kind : '') + (opts.cls ? ' ' + opts.cls : '');
    var b = h('button', {type: 'button', 'class': cls, title: opts.title || null, disabled: Boolean(opts.disabled)});
    if (opts.icon) { var ic = icon(opts.icon); if (ic) b.appendChild(ic); }
    if (label) b.appendChild(h('span', {text: label}));
    if (opts.iconAfter) { var ia = icon(opts.iconAfter, {size: 14}); if (ia) b.appendChild(ia); }
    if (onClick) b.addEventListener('click', function (ev) { if (!b.disabled) onClick(ev); });
    return b;
  }
  function iconButton(name, title, onClick, opts) {
    opts = opts || {};
    var b = h('button', {type: 'button', 'class': 'btn btn-icon' + (opts.cls ? ' ' + opts.cls : ''), title: title, 'aria-label': title,
      'aria-pressed': opts.pressed === undefined ? null : String(Boolean(opts.pressed))});
    var ic = icon(name); if (ic) b.appendChild(ic);
    if (onClick) b.addEventListener('click', function (ev) { if (!b.disabled) onClick(ev); });
    return b;
  }
  function setPressed(b, on) { if (b) { b.setAttribute('aria-pressed', String(Boolean(on))); if (b.classList) b.classList.toggle('on', Boolean(on)); } }
  function link(label, href, opts) {
    opts = opts || {};
    var a = h('a', {href: href, 'class': 'lnk' + (opts.cls ? ' ' + opts.cls : ''), target: opts.newTab ? '_blank' : null, rel: opts.newTab ? 'noopener' : null});
    a.appendChild(h('span', {text: label}));
    if (opts.newTab) { var ic = icon('external', {size: 12}); if (ic) a.appendChild(ic); }
    return a;
  }
  function select(options, value, onChange, attrs) {
    var s = h('select', attrs || {});
    options.forEach(function (o) { s.appendChild(h('option', {value: o.value, text: o.label, disabled: Boolean(o.disabled)})); });
    if (value !== undefined && value !== null) s.value = String(value);
    if (onChange) s.addEventListener('change', function () { onChange(s.value); });
    return s;
  }
  function section(title, opts) {
    opts = opts || {};
    var head = h('div', {'class': 'sec-head'}, h('h3', {'class': 'sec-title', text: title}), opts.tag || null,
      h('span', {'class': 'sec-fill'}), opts.actions || null);
    var sec = h('section', {'class': 'sec' + (opts.cls ? ' ' + opts.cls : '')}, head);
    for (var i = 1; i < arguments.length; i++) if (i > 1) appendKids(sec, arguments[i]);
    return sec;
  }
  function empty(text, actions) {
    return h('div', {'class': 'empty'}, h('p', {text: text}), actions && actions.length ? h('div', {'class': 'empty-actions'}, actions) : null);
  }
  function note(text, kind) { return h('p', {'class': 'note' + (kind ? ' note-' + kind : ''), text: text}); }
  // A small ⓘ that carries an explanation as a tooltip instead of a line of grey text.
  function infoTip(text) { return text ? h('span', {'class': 'info-tip', title: text, 'aria-label': text, tabindex: '0', role: 'note', text: 'i'}) : null; }
  // Rows: array of objects; columns: [{key,label,num,cls,render(row)}]; opts.onRow(row, tr)
  function table(columns, rows, opts) {
    opts = opts || {};
    var thead = h('thead', {}, h('tr', {}, columns.map(function (c) { return h('th', {'class': (c.num ? 'num' : '') + (c.cls ? ' ' + c.cls : ''), scope: 'col', text: c.label}); })));
    var tbody = h('tbody');
    rows.forEach(function (row) {
      var tr = h('tr', {'class': row._cls || null});
      columns.forEach(function (c) {
        var content = c.render ? c.render(row) : row[c.key];
        var td = h('td', {'class': (c.num ? 'num' : '') + (c.cls ? ' ' + c.cls : '')});
        appendKids(td, content === undefined || content === null || content === '' ? (c.blank ? '' : '—') : content);
        tr.appendChild(td);
      });
      if (opts.onRow) opts.onRow(row, tr);
      tbody.appendChild(tr);
    });
    return h('table', {'class': 'tbl' + (opts.cls ? ' ' + opts.cls : '')}, thead, tbody);
  }
  // A number the reader can click for its evidence chain (D9).
  function evidence(text, onClick, title) {
    if (!onClick) return h('span', {'class': 'val', text: text});
    return h('button', {type: 'button', 'class': 'val val-link', title: title || '查看这个数的来源', text: text, onclick: onClick});
  }

  // ------------------------------------------------------------------ hosts for floating layers
  function host(id, tag, cls) {
    var d = doc();
    var el = d.getElementById(id);
    if (el) return el;
    el = d.createElement(tag || 'div');
    el.id = id;
    if (cls) el.className = cls;
    if (d.body) d.body.appendChild(el);
    return el;
  }

  // ------------------------------------------------------------------ dialog
  var current = null;
  function dialogOpen() { return Boolean(current); }
  function openDialog(spec) {
    var dlg = host('ws-dialog', 'dialog', 'ws-dialog');
    if (current) closeDialog('replaced');
    var closeBtn = iconButton('close', '关闭', function () { closeDialog('cancel'); }, {cls: 'dlg-close'});
    var body = h('div', {'class': 'dlg-body'});
    appendKids(body, spec.body || []);
    var foot = spec.actions && spec.actions.length ? h('div', {'class': 'dlg-foot'}, spec.actions) : null;
    dlg.className = 'ws-dialog' + (spec.wide ? ' wide' : '') + (spec.cls ? ' ' + spec.cls : '');
    fill(dlg, h('div', {'class': 'dlg-head'}, h('h2', {id: 'ws-dialog-title', text: spec.title || ''}), closeBtn), body, foot);
    var before = doc().activeElement;
    current = {el: dlg, body: body, onClose: spec.onClose || null, returnFocus: before && before !== doc().body ? before : null};
    if (!dlg._wsBound) {
      dlg._wsBound = true;
      dlg.addEventListener('cancel', function (ev) { if (ev && ev.preventDefault) ev.preventDefault(); closeDialog('cancel'); });
      dlg.addEventListener('click', function (ev) { if (ev && ev.target === dlg) closeDialog('cancel'); });
      // Closed some other way (form method=dialog, script): forget it so keyboard shortcuts work again.
      dlg.addEventListener('close', function () { if (current && current.el === dlg && !current.closing) closeDialog('cancel'); });
    }
    if (typeof dlg.showModal === 'function') { try { if (!dlg.open) dlg.showModal(); } catch (_) { dlg.setAttribute('open', ''); } }
    else dlg.setAttribute('open', '');
    // Focus goes to the named control, else to the dialog itself (no focus ring on the close button).
    if (spec.focus && spec.focus.focus) { try { spec.focus.focus(); } catch (_) {} }
    else { dlg.setAttribute('tabindex', '-1'); try { dlg.focus(); } catch (_) {} }
    return {el: dlg, body: body, close: function (reason) { if (current && current.el === dlg) closeDialog(reason || 'done'); }};
  }
  function closeDialog(reason) {
    if (!current) return;
    var c = current; current = null;
    c.closing = true;
    try { if (typeof c.el.close === 'function' && c.el.open) c.el.close(); else c.el.removeAttribute('open'); } catch (_) {}
    c.el.replaceChildren();
    if (c.onClose) { try { c.onClose(reason || 'cancel'); } catch (_) {} }
    if (c.returnFocus && c.returnFocus.focus && reason !== 'replaced') { try { c.returnFocus.focus(); } catch (_) {} }
  }
  function confirm(title, lines, opts) {
    opts = opts || {};
    return new Promise(function (resolve) {
      var done = false;
      var finish = function (v) { if (done) return; done = true; resolve(v); };
      var ok = button(opts.confirmLabel || '确定', function () { finish(true); closeDialog('done'); }, {kind: opts.danger ? 'danger' : 'primary'});
      openDialog({title: title, body: (lines || []).map(function (t) { return h('p', {text: t}); }),
        actions: [button(opts.cancelLabel || '取消', function () { closeDialog('cancel'); }), ok],
        onClose: function () { finish(false); }, focus: ok});
    });
  }

  // ------------------------------------------------------------------ menu
  var menuState = null;
  function closeMenu() {
    if (!menuState) return;
    var m = menuState; menuState = null;
    m.el.hidden = true; m.el.replaceChildren();
    if (m.anchor) { m.anchor.setAttribute('aria-expanded', 'false'); }
  }
  // ``opts.at`` = {x, y} (client px): a context menu at the pointer instead of below the anchor; a second right-click
  // on the same anchor moves it rather than closing it.
  function menu(anchor, items, opts) {
    var at = opts && opts.at && isFinite(opts.at.x) && isFinite(opts.at.y) ? opts.at : null;
    var el = host('ws-menu', 'div', 'ws-menu');
    if (menuState && menuState.anchor === anchor && !at) { closeMenu(); return; }
    closeMenu();
    el.setAttribute('role', 'menu');
    el.replaceChildren();
    items.filter(Boolean).forEach(function (item) {
      if (item.separator) { el.appendChild(h('div', {'class': 'menu-sep', role: 'separator'})); return; }
      if (item.heading) { el.appendChild(h('div', {'class': 'menu-head', text: item.heading})); return; }
      var tag = item.href ? 'a' : 'button';
      var entry = h(tag, {'class': 'menu-item' + (item.checked ? ' checked' : ''), role: item.checked === undefined ? 'menuitem' : 'menuitemcheckbox',
        type: item.href ? null : 'button', href: item.href || null, target: item.href && item.newTab ? '_blank' : null, rel: item.href && item.newTab ? 'noopener' : null,
        'aria-checked': item.checked === undefined ? null : String(Boolean(item.checked)), disabled: item.href ? null : Boolean(item.disabled)});
      var mark = h('span', {'class': 'menu-mark', 'aria-hidden': 'true'});
      if (item.checked) { var ic = icon('check', {size: 14}); if (ic) mark.appendChild(ic); }
      entry.appendChild(mark);
      entry.appendChild(h('span', {'class': 'menu-label', text: item.label}));
      if (item.hint) entry.appendChild(h('span', {'class': 'menu-hint', text: item.hint}));
      entry.addEventListener('click', function (ev) {
        if (item.href) { closeMenu(); return; }
        if (ev && ev.preventDefault) ev.preventDefault();
        closeMenu(); if (item.run) item.run();
      });
      el.appendChild(entry);
    });
    el.hidden = false;
    menuState = {el: el, anchor: anchor};
    if (anchor) anchor.setAttribute('aria-expanded', 'true');
    var rect = anchor && typeof anchor.getBoundingClientRect === 'function' ? anchor.getBoundingClientRect() : null;
    if (at) {
      var vw0 = root.innerWidth || 1280, vh0 = root.innerHeight || 800, w0 = 240;
      el.style.left = Math.max(8, Math.min(at.x, vw0 - w0 - 8)) + 'px';
      el.style.maxHeight = Math.max(160, vh0 - 16) + 'px';
      var tall0 = el.offsetHeight || 0, top0 = at.y;
      if (tall0 && top0 + tall0 > vh0 - 8) top0 = Math.max(8, vh0 - 8 - tall0);
      el.style.top = top0 + 'px';
    } else if (rect) {
      var width = 240, vw = root.innerWidth || 1280;
      var left = Math.max(8, Math.min(rect.right - width, vw - width - 8));
      if (rect.left + width < vw - 8 && rect.left < vw / 2) left = Math.max(8, rect.left);
      el.style.left = left + 'px';
      // a long menu (图层) stays inside the window: it moves up as far as needed and scrolls when taller than the window
      var vh = root.innerHeight || 800, top = rect.bottom + 4;
      el.style.maxHeight = Math.max(160, vh - 16) + 'px';
      var tall = el.offsetHeight || 0;
      if (tall && top + tall > vh - 8) top = Math.max(8, vh - 8 - tall);
      el.style.top = top + 'px';
    }
    return el;
  }
  function menuOpen() { return Boolean(menuState); }

  // ------------------------------------------------------------------ notice bar
  var toastTimer = null;
  function hideToast() { var el = host('ws-toast', 'div', 'ws-toast'); el.hidden = true; el.replaceChildren(); if (toastTimer) { clearTimeout(toastTimer); toastTimer = null; } }
  function toast(message, opts) {
    opts = opts || {};
    var el = host('ws-toast', 'div', 'ws-toast');
    if (toastTimer) { clearTimeout(toastTimer); toastTimer = null; }
    var kind = opts.kind || 'info';
    el.className = 'ws-toast toast-' + kind;
    el.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    var actions = (opts.actions || []).map(function (a) { return button(a.label, function () { hideToast(); a.run(); }, {kind: 'link'}); });
    fill(el, h('span', {'class': 'toast-text', text: message}), actions, iconButton('close', '关闭提示', hideToast, {cls: 'toast-close'}));
    el.hidden = false;
    var ms = opts.ms !== undefined ? opts.ms : (kind === 'error' ? 0 : (actions.length ? 12000 : 6000));
    if (ms > 0) { toastTimer = setTimeout(hideToast, ms); if (toastTimer && toastTimer.unref) toastTimer.unref(); }
    return el;
  }

  function downloadBlob(blob, name) {
    var d = doc();
    var URLs = root.URL;
    if (!URLs || typeof URLs.createObjectURL !== 'function') return false;
    var href = URLs.createObjectURL(blob);
    var a = h('a', {href: href, download: name});
    d.body.appendChild(a); a.click(); if (a.remove) a.remove();
    setTimeout(function () { try { URLs.revokeObjectURL(href); } catch (_) {} }, 20000);
    return true;
  }
  function isTyping(target) {
    if (!target) return false;
    var tag = String(target.tagName || '').toUpperCase();
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
    if (target.isContentEditable) return true;
    return typeof target.closest === 'function' && Boolean(target.closest('dialog[open], [contenteditable="true"]'));
  }

  // ------------------------------------------------------------------ chart tooltip (home overview, cohort charts)
  // One tooltip per chart card (``host`` is positioned): the value leads, the label follows; text only.  Marks call
  // bind(mark, value, label) and get the same readout on hover and on keyboard focus.  It never gates a value: every
  // number it shows is also in a table (task list, cohort table).
  function chartTip(host) {
    var el = h('div', {'class': 'chart-tip', role: 'tooltip'});
    el.hidden = true;
    host.appendChild(el);
    function show(anchor, value, label) {
      fill(el, h('strong', {text: value}), label ? h('span', {text: label}) : null);
      el.hidden = false;
      var hr = host.getBoundingClientRect(), ar = anchor.getBoundingClientRect();
      var w = el.offsetWidth || 0, x = ar.left + ar.width / 2 - hr.left, y = ar.top - hr.top;
      if (w) x = Math.max(w / 2 + 4, Math.min(hr.width - w / 2 - 4, x));
      el.style.left = x + 'px';
      el.style.top = y + 'px';
    }
    function hide() { el.hidden = true; }
    function bind(mark, value, label) {
      var on = function () { show(mark, typeof value === 'function' ? value() : value, typeof label === 'function' ? label() : label); };
      mark.addEventListener('pointerenter', on); mark.addEventListener('focus', on);
      mark.addEventListener('pointerleave', hide); mark.addEventListener('blur', hide);
      return mark;
    }
    return {show: show, hide: hide, bind: bind, el: el};
  }

  var _infoTip = infoTip;
  return {h: h, clear: clear, fill: fill, icon: icon, appendKids: appendKids, chartTip: chartTip,
    sig: sig, trim: trim, num: num, pct: pct, range: range, unitText: unitText, time: time, duration: duration,
    statusInfo: statusInfo, reviewInfo: reviewInfo, dot: dot, statusDot: statusDot, reviewDot: reviewDot, tierTag: tierTag, tierHelp: tierHelp, TIERS: TIERS,
    releaseIdOf: releaseIdOf, kindOf: kindOf, resultName: resultName, packageName: packageName, displayName: displayName,
    button: button, iconButton: iconButton, setPressed: setPressed, link: link, select: select, section: section, empty: empty, note: note, table: table, evidence: evidence,
    host: host, dialog: {open: openDialog, close: closeDialog, isOpen: dialogOpen}, confirm: confirm, menu: menu, closeMenu: closeMenu, menuOpen: menuOpen,
    toast: toast, hideToast: hideToast, downloadBlob: downloadBlob, isTyping: isTyping, infoTip: _infoTip};
});
