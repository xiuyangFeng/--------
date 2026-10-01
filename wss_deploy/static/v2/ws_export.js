/* WSS workspace v2 — export by purpose O6 (contract §6.3 ws_export.js), second phase S5a (lane A).
 * One dialog, four pages:
 *   图片     publication figure of the current view: 1× / 2× / 4×, viewport / white / transparent background, Chinese or
 *           English words, colour bar / title / labels; colour bar as SVG; print the view (ns.figure)
 *   拼图     six standard views or a chosen set (+ current view, + section map) with one colour bar (ns.figure)
 *   一页纸   figures for the one-page report (classic snapshots format) and the one-pager itself
 *   数据     statistics CSV, the classic result card's single files by family (phase 3 lane 5: wall VTP / point CSV;
 *           volume VTP / wall-pressure VTP / volume CSV / streamlines VTP), zip, and the offline HTML
 *           (/api/v2/.../offline, bookmarks, name can be hidden)
 * The footer copies the reproducible link.  Without ns.figure (tests, a trimmed bundle) the 图片 page falls back to the
 * first-phase 汇报图: snapshot + colour bar + window name + display name. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.exporter = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };

  function stamp() { var d = new Date(); var p = function (n) { return n < 10 ? '0' + n : String(n); }; return d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()); }
  function safeName(s) { return String(s || '病例').replace(/[\\/:*?"<>|\s]+/g, '_').slice(0, 60); }
  function loadImage(blob) {
    return new Promise(function (resolve, reject) {
      if (typeof root.createImageBitmap === 'function') { root.createImageBitmap(blob).then(resolve, reject); return; }
      var url = root.URL.createObjectURL(blob);
      var img = new root.Image();
      img.onload = function () { resolve(img); setTimeout(function () { root.URL.revokeObjectURL(url); }, 1000); };
      img.onerror = reject; img.src = url;
    });
  }
  function svgImage(svg) {
    return new Promise(function (resolve, reject) {
      var url = root.URL.createObjectURL(new root.Blob([svg], {type: 'image/svg+xml'}));
      var img = new root.Image();
      img.onload = function () { resolve(img); setTimeout(function () { root.URL.revokeObjectURL(url); }, 1000); };
      img.onerror = function (e) { root.URL.revokeObjectURL(url); reject(e); };
      img.src = url;
    });
  }
  // First-phase 汇报图 (fallback): caption on top, the WebGL snapshot, the kernel's colour bar on the right, one
  // research line at the bottom.
  function composeFigure(opts) {
    var s = opts.scale || 2;
    var svg = null;
    try { svg = opts.info && ns.colorbar && typeof ns.colorbar.toSVG === 'function' ? ns.colorbar.toSVG(opts.info, {width: 190, graphicHeight: 300, background: '#ffffff'}) : null; } catch (_) { svg = null; }
    return Promise.all([loadImage(opts.snapshot), svg ? svgImage(svg).catch(function () { return null; }) : Promise.resolve(null)]).then(function (imgs) {
      var img = imgs[0], bar = imgs[1];
      s = Math.max(1, Math.min(4, Math.round(img.width / 800)));
      var legendW = bar ? Math.round(bar.width * s) + 32 * s : 0, headH = 62 * s, footH = 34 * s;
      var c = root.document.createElement('canvas');
      c.width = img.width + legendW; c.height = Math.max(img.height, bar ? bar.height * s + 40 * s : 0) + headH + footH;
      var g = c.getContext('2d');
      g.fillStyle = '#ffffff'; g.fillRect(0, 0, c.width, c.height);
      g.fillStyle = '#eceef0'; g.fillRect(0, headH, img.width, img.height);
      g.drawImage(img, 0, headH);
      var font = function (px, w) { return (w || 400) + ' ' + px * s + 'px system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif'; };
      g.textBaseline = 'top';
      g.fillStyle = '#1b2430'; g.font = font(18, 600); g.fillText(opts.title || '', 16 * s, 12 * s);
      g.fillStyle = '#465361'; g.font = font(13); g.fillText(opts.subtitle || '', 16 * s, 38 * s);
      if (bar) g.drawImage(bar, img.width + 16 * s, headH + 16 * s, bar.width * s, bar.height * s);
      g.fillStyle = '#77828e'; g.font = font(11);
      g.fillText(opts.footer || '', 16 * s, c.height - footH + 10 * s);
      return new Promise(function (resolve) { c.toBlob(resolve, 'image/png'); });
    });
  }
  function fallbackFigure(el, ctx) {
    var h = ui().h, m = ctx.manifest || {}, name = ctx.displayName || '病例';
    var hideFig = h('input', {type: 'checkbox', checked: Boolean(ctx.hideName)});
    var figBtn = ui().button('下载 PNG', function () {
      if (!ctx.viewer || typeof ctx.viewer.snapshot !== 'function') { ui().toast('三维视图不可用，不能出图。', {kind: 'error'}); return; }
      figBtn.disabled = true;
      var shown = hideFig.checked ? '病例' : name;
      var info = null; try { info = ctx.info ? ctx.info() : null; } catch (_) { info = null; }
      Promise.resolve(ctx.viewer.snapshot({scale: 2, transparent: false})).then(function (blob) {
        return composeFigure({snapshot: blob, info: info, scale: 2, units: ui().unitText(ctx.units),
          title: shown + ' · ' + ((m.result && m.result.display_name) || ''),
          subtitle: (ctx.fieldLabel || '') + (ctx.units && ui().unitText(ctx.units) ? '（' + ui().unitText(ctx.units) + '）' : '') + ' · 色标：' + (ctx.windowLabel || '本例自适应'),
          footer: '研究用途，非诊断 · 导出于 ' + ui().time(new Date().toISOString())});
      }).then(function (png) {
        ui().downloadBlob(png, 'WSS_' + safeName(shown) + '_' + (ctx.fieldId || 'field') + '_' + stamp() + '.png');
      }, function (e) { ui().toast('出图失败：' + (e && e.message || e), {kind: 'error'}); }).then(function () { figBtn.disabled = false; });
    }, {icon: 'download', cls: 'btn-sm'});
    ui().fill(el, h('div', {'class': 'exp-controls'}, h('label', {'class': 'check'}, hideFig, h('span', {text: '隐藏病例名'})), figBtn));
  }

  function dataPane(el, ctx, state) {
    var h = ui().h, api = ns.api, m = ctx.manifest || {};
    var jobId = (ctx.job && ctx.job.id) || (m.job && m.job.id);
    var name = ctx.displayName || '病例';
    var exportsMap = (ctx.job && ctx.job.summary && ctx.job.summary.exports) || {};
    var family = m.result && m.result.family === 'volume' ? 'volume' : 'wall';
    var links = [h('a', {'class': 'lnk', href: api.urls.table(jobId, 'csv'), download: '', text: '统计表 CSV'})];
    // P3 lane 5 (#93): the classic result card's single files, by family (summary.exports holds flags, not names)
    dataFiles(family, exportsMap).forEach(function (f) { links.push(h('a', {'class': 'lnk', href: api.urls.file(jobId, f.file), download: '', text: f.label})); });
    links.push(h('a', {'class': 'lnk', href: api.urls.bundle(jobId), download: '', text: '全部文件 zip'}));
    if (ctx.full) links.push(h('a', {'class': 'lnk', href: api.urls.file(jobId, 'summary.json'), target: '_blank', rel: 'noopener', text: '完整统计 JSON'}));
    var hideOff = h('input', {type: 'checkbox', checked: Boolean(state.hideName)});
    var bms = ctx.bookmarks || [];
    var withBm = h('input', {type: 'checkbox', checked: bms.length > 0, disabled: !bms.length});
    var offBtn = ui().button('下载离线报告', function () {
      offBtn.disabled = true;
      var view = null; try { view = ctx.viewState ? ctx.viewState() : null; } catch (_) { view = null; }
      api.offline(jobId, {hide_name: hideOff.checked, bookmarks: withBm.checked ? bms : [], view: view || {}}).then(function (r) {
        ui().downloadBlob(r.blob, r.filename || ('WSS_' + safeName(hideOff.checked ? '病例' : name) + '_' + stamp() + '.html'));
        ui().toast('离线报告已下载。', {kind: 'ok'});
      }, function (e) { ui().toast('离线报告没有生成：' + e.message, {kind: 'error'}); }).then(function () { offBtn.disabled = false; });
    }, {icon: 'download', cls: 'btn-sm'});
    ui().fill(el,
      block('复核数据', '原始统计和网格数据，保留全精度，给自己或同事复算。VTP 可在 ParaView 打开；CSV 是逐点数值。', [h('div', {'class': 'exp-links'}, links)]),
      block('离线阅读', '一个 HTML 文件，在没有服务的电脑上用浏览器直接打开；显示导出时的复核状态。', [
        h('label', {'class': 'check'}, hideOff, h('span', {text: '隐藏病例名'})),
        h('label', {'class': 'check'}, withBm, h('span', {text: bms.length ? '带上书签（' + bms.length + ' 条）' : '带上书签（还没有书签）'})), offBtn]));
  }
  // The classic workbench result card (app.js resultCard): wall VTP + point CSV; volume VTP, wall-pressure VTP, volume CSV,
  // and the streamlines VTP only when the run wrote it.  summary.exports = {vtp, csv, wall_pressure, streamlines, …}: flags.
  var DATA_FILES = {
    wall: [{file: 'wall_wss.vtp', label: '壁面 VTP', flag: 'vtp'}, {file: 'points_wss.csv', label: '点云 CSV', flag: 'csv'}],
    volume: [{file: 'volume_fields.vtp', label: '体场 VTP', flag: 'vtp'}, {file: 'wall_pressure.vtp', label: '壁面压力 VTP', flag: 'wall_pressure'},
      {file: 'points_volume.csv', label: '体场 CSV', flag: 'csv'}, {file: 'streamlines.vtp', label: '流线 VTP', flag: 'streamlines', only: true}]
  };
  function dataFiles(family, exportsMap) {
    var ex = exportsMap && typeof exportsMap === 'object' ? exportsMap : {};
    return (DATA_FILES[family] || DATA_FILES.wall).filter(function (f) { return f.only ? Boolean(ex[f.flag]) : ex[f.flag] !== false; });
  }
  function block(title, tip, controls) {
    var h = ui().h;
    return h('section', {'class': 'exp-block'}, h('h4', {}, h('span', {text: title}), ui().infoTip(tip)), h('div', {'class': 'exp-controls'}, controls));
  }

  // ctx: {job, manifest, viewer, info(), fieldId, fieldLabel, units, windowLabel, displayName, bookmarks:[], offline, hideName, full, viewState()}
  // mod.pendingTab (set by the 工具 tab's buttons) picks the first page.
  var lastTab = 'figure';
  function open(ctx) {
    var h = ui().h;
    var api = ns.api, m = ctx.manifest || {};
    var jobId = (ctx.job && ctx.job.id) || (m.job && m.job.id);
    var F = ns.figure && typeof ns.figure.pane === 'function' && ns.figure.available && ns.figure.available() ? ns.figure : null;
    var online = Boolean(!ctx.offline && api && jobId);
    var tabs = [{id: 'figure', label: '图片'}];
    if (F) tabs.push({id: 'montage', label: '拼图'});
    if (online && F) tabs.push({id: 'onepage', label: '一页纸'});
    if (online) tabs.push({id: 'data', label: '数据'});
    var want = mod.pendingTab || lastTab;
    mod.pendingTab = null;
    if (!tabs.some(function (t) { return t.id === want; })) want = 'figure';
    var state = {hideName: Boolean(ctx.hideName)};
    var fctx = {hideName: Boolean(ctx.hideName), offline: Boolean(ctx.offline), jobId: jobId, state: state};
    var bar = h('div', {'class': 'seg exp-tabs', role: 'tablist', 'aria-label': '导出用途'});
    var paneEl = h('div', {'class': 'exp-pane', role: 'tabpanel'});
    var buttons = {};
    function show(id) {
      lastTab = id;
      Object.keys(buttons).forEach(function (k) { var on = k === id; buttons[k].classList.toggle('on', on); buttons[k].setAttribute('aria-selected', String(on)); });
      if (F) F.disposePane();
      if (id === 'data') dataPane(paneEl, ctx, state);
      else if (F) F.pane(id, paneEl, fctx);
      else fallbackFigure(paneEl, ctx);
    }
    tabs.forEach(function (t) {
      var b = h('button', {type: 'button', role: 'tab', 'class': 'seg-btn', text: t.label, dataset: {tab: t.id}});
      b.addEventListener('click', function () { show(t.id); });
      buttons[t.id] = b; bar.appendChild(b);
    });
    var linkBox = h('input', {type: 'text', readOnly: true, 'class': 'exp-link-text', 'aria-label': '复现链接', hidden: true});
    var linkBtn = F ? ui().button('复现链接', function () {
      F.copyLink(null, function (url, copied) {
        linkBox.value = url; linkBox.hidden = false;
        try { linkBox.focus(); linkBox.select(); } catch (_) {}
        if (copied) ui().toast('已复制复现链接：打开它就回到同样的字段、色标、视角、游标和截面。', {kind: 'ok', ms: 4000});
      });
    }, {icon: 'fig-link', kind: 'link', title: '复制一个链接：打开后回到同样的字段、色标（色表、分段、阈值、单位）、标签、视角、游标和截面'}) : null;
    ui().dialog.open({title: '导出', wide: true, cls: 'dlg-export', body: [bar, paneEl],
      actions: [linkBtn, linkBox, h('span', {'class': 'sec-fill'}), ui().button('关闭', function () { ui().dialog.close('done'); })],
      onClose: function () { if (F) F.disposePane(); }});
    show(want);
  }

  var mod = {open: open, composeFigure: composeFigure, safeName: safeName, dataFiles: dataFiles, DATA_FILES: DATA_FILES, pendingTab: null};
  return mod;
});
