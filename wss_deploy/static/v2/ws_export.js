/* WSS workspace v2 — export by purpose O6 (contract §6.3 ws_export.js).
 *   汇报图   snapshot + colour bar + window name + display name (the name can be hidden)
 *   复核数据 the existing CSV / VTP / zip
 *   离线阅读 POST /api/v2/jobs/<id>/offline, bookmarks included, name can be hidden
 *   打印报告 the existing one-page report */
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
  // Compose the figure: caption on top, the WebGL snapshot, the kernel's own colour bar (ns.colorbar.toSVG, the same
  // drawing as on screen) on the right, one research line at the bottom.
  function composeFigure(opts) {
    var s = opts.scale || 2;   // re-derived from the snapshot below: text and colour bar follow its pixel size
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

  // ctx: {job, manifest, viewer, info(), fieldLabel, units, windowLabel, displayName, bookmarks:[], offline, hideName}
  function open(ctx) {
    var h = ui().h;
    var api = ns.api;
    var m = ctx.manifest || {};
    var jobId = (ctx.job && ctx.job.id) || (m.job && m.job.id);
    var name = ctx.displayName || '病例';
    var blocks = [];
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
    blocks.push(block('汇报图', '当前视口的截图，带色标、窗名和病例名，适合放进幻灯片。', [h('label', {'class': 'check'}, hideFig, h('span', {text: '隐藏病例名'})), figBtn]));
    if (!ctx.offline && api && jobId) {
      var exportsList = (ctx.job && ctx.job.summary && ctx.job.summary.exports) || {};
      var vtp = Object.keys(exportsList).map(function (k) { return exportsList[k]; }).filter(function (v) { return typeof v === 'string' && /\.vtp$/i.test(v); })[0] || (m.result && m.result.family === 'volume' ? null : 'wall_wss.vtp');
      var links = [h('a', {'class': 'lnk', href: api.urls.table(jobId, 'csv'), download: '', text: '统计表 CSV'})];
      if (vtp) links.push(h('a', {'class': 'lnk', href: api.urls.file(jobId, String(vtp).replace(/^.*\//, '')), download: '', text: '壁面数据 VTP'}));
      links.push(h('a', {'class': 'lnk', href: api.urls.bundle(jobId), download: '', text: '全部文件 zip'}));
      if (ctx.full) links.push(h('a', {'class': 'lnk', href: api.urls.file(jobId, 'summary.json'), target: '_blank', rel: 'noopener', text: '完整统计 JSON'}));
      blocks.push(block('复核数据', '原始统计和网格数据，保留全精度，给自己或同事复算。', [h('div', {'class': 'exp-links'}, links)]));
      var hideOff = h('input', {type: 'checkbox', checked: Boolean(ctx.hideName)});
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
      blocks.push(block('离线阅读', '一个 HTML 文件，在没有服务的电脑上用浏览器直接打开；显示导出时的复核状态。', [
        h('label', {'class': 'check'}, hideOff, h('span', {text: '隐藏病例名'})),
        h('label', {'class': 'check'}, withBm, h('span', {text: bms.length ? '带上书签（' + bms.length + ' 条）' : '带上书签（还没有书签）'})), offBtn]));
      blocks.push(block('打印报告', '一页纸报告，可直接打印或另存为 PDF。', [ui().button('打开一页纸', function () { root.open(api.urls.onepage(jobId), '_blank', 'noopener'); }, {icon: 'print', cls: 'btn-sm'})]));
    }
    ui().dialog.open({title: '导出', wide: true, body: blocks, actions: [ui().button('关闭', function () { ui().dialog.close('done'); })]});
  }
  function block(title, text, controls) {
    var h = ui().h;
    return h('section', {'class': 'exp-block'}, h('h4', {text: title}), h('p', {'class': 'muted', text: text}), h('div', {'class': 'exp-controls'}, controls));
  }

  return {open: open, composeFigure: composeFigure, safeName: safeName};
});
