/* WSS workspace v2 — Lane A (S5a): figures (PHASE2_LANES.md §5 A).
 *   出版级导图  the current view rendered offscreen at 1× / 2× / 4× (viewer.renderPose), background viewport / white /
 *              transparent, the stage colour bar (ns.colorbar.toSVG — the drawing shown on screen), title, and the 3-D
 *              label chips redrawn at the export resolution; Chinese or English words
 *   拼图        the six standard views or a chosen set (+ the current view, + the volume section map while the section
 *              tool is open), one shared colour bar — WssReportCommon.composeMontage, as in the classic reports
 *   色标 SVG    the stage colour bar as a standalone file
 *   复现链接    #/job/<id>?…&view=<base64url JSON>: field, window, camera, cursor, selection, shown branches, layers and
 *              the volume section; read back in onResult (the shell's parseHash is not changed)
 *   一页纸配图  POST /api/jobs/<id>/snapshots in the classic buildSnapshots / makeOnepageShots format
 *   打印        the current view on a print-only A4 page
 * Shell hooks through ns.ext (the 工具 tab section, onResult / onClose); the export dialog (ws_export.js) shows the panes.
 * Phase 3 lane 5 (PHASE3_LANES.md): the link also carries the display (d: colour map, lighting, labels, bands,
 * thresholds, units, display layers, volume options) and extension states (x, ns.ext linkState / applyLinkState), still
 * v:1 — an older link without d leaves the display alone; the chosen views as separate PNGs in a zip (拼图 page 「分成
 * 单张」); the store-mode zip writer; headless(ctx) for a result that is not on screen (ws_batch.js).
 * Final round lane B (W58 / W59): Ctrl / ⌘ + P on an open result goes to the print-view page (printView); a print started
 * from the browser's menu prints a 2-D copy of every 3-D viewport (beforeprint), removed again on afterprint. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.figure = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };
  var common = function () { return root.WssReportCommon || null; };
  var API = null;   // the shell's helper object: the last argument of every extension hook

  var FONT = 'system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
  var VIEWS = ['anterior', 'posterior', 'left', 'right', 'superior', 'inferior'];
  // Classic names (report.py CORE.VIEW_NAMES / volume_viewer STANDARD_VIEWS): file names and one-pager snapshot names.
  var CLASSIC = {anterior: 'front', posterior: 'back', left: 'left', right: 'right', superior: 'top', inferior: 'bottom', current: 'custom', slice: 'slice'};
  var VIEW_ZH = {anterior: '前视', posterior: '后视', left: '左视', right: '右视', superior: '头侧视', inferior: '足侧视', current: '当前视角', slice: '截面'};
  var VIEW_EN = {anterior: 'Anterior', posterior: 'Posterior', left: 'Left', right: 'Right', superior: 'Superior', inferior: 'Inferior', current: 'Current view', slice: 'Section'};
  var VIEW_SHORT = {anterior: '前', posterior: '后', left: '左', right: '右', superior: '头', inferior: '足'};
  var ONEPAGE_PLAN = [['front', 'anterior'], ['left', 'left'], ['top', 'superior']];
  var SNAPSHOT_MAX = 6 * 1024 * 1024, SNAPSHOT_TOTAL = 22 * 1024 * 1024;   // server: 6 MB per PNG, 24 MB per request

  // ------------------------------------------------------------------ icons (lane-specific names)
  if (ns.icons && typeof ns.icons.register === 'function') {
    var P = ns.icons.P, R = ns.icons.R;
    ns.icons.register('fig-link', [P('M6.75 9.25a2.75 2.75 0 0 0 3.9 0l2.1-2.1a2.75 2.75 0 0 0-3.9-3.9l-.9.9'), P('M9.25 6.75a2.75 2.75 0 0 0-3.9 0l-2.1 2.1a2.75 2.75 0 0 0 3.9 3.9l.9-.9')]);
    ns.icons.register('fig-montage', [R(1.75, 2.75, 4, 4.5, 0.75), R(6.25, 2.75, 4, 4.5, 0.75), R(1.75, 8.75, 4, 4.5, 0.75), R(6.25, 8.75, 4, 4.5, 0.75), R(12, 2.75, 2.25, 10.5, 0.75)]);
    ns.icons.register('fig-colorbar', [R(3.25, 1.75, 3.5, 12.5, 0.75), P('M6.75 4.5h2'), P('M6.75 8h2'), P('M6.75 11.5h2'), P('M11 4.5h2.5'), P('M11 8h2.5'), P('M11 11.5h2.5')]);
  }

  // ------------------------------------------------------------------ options (per browser, try/catch via ns.store)
  var PREF_KEY = 'wssv2:export';
  var DEFAULTS = {scale: 2, background: 'white', lang: 'zh', colorbar: true, title: true, labels: true, crop: true, views: VIEWS.slice(), columns: 3};
  function sanitize(raw) {
    var o = {};
    Object.keys(DEFAULTS).forEach(function (k) { o[k] = Array.isArray(DEFAULTS[k]) ? DEFAULTS[k].slice() : DEFAULTS[k]; });
    if (!raw || typeof raw !== 'object') return o;
    if ([1, 2, 4].indexOf(+raw.scale) >= 0) o.scale = +raw.scale;
    if (['viewport', 'white', 'transparent'].indexOf(raw.background) >= 0) o.background = raw.background;
    if (raw.lang === 'zh' || raw.lang === 'en') o.lang = raw.lang;
    ['colorbar', 'title', 'labels', 'crop'].forEach(function (k) { if (typeof raw[k] === 'boolean') o[k] = raw[k]; });
    if (Array.isArray(raw.views)) o.views = raw.views.filter(function (v, i, a) { return (VIEWS.indexOf(v) >= 0 || v === 'current' || v === 'slice') && a.indexOf(v) === i; });
    if ([2, 3, 4].indexOf(+raw.columns) >= 0) o.columns = +raw.columns;
    return o;
  }
  function readRaw() {
    try { if (ns.store && typeof ns.store.read === 'function') return ns.store.read(PREF_KEY); } catch (_) {}
    try { var s = root.localStorage && root.localStorage.getItem(PREF_KEY); return s ? JSON.parse(s) : null; } catch (_) { return null; }
  }
  function options() { return sanitize(readRaw()); }
  function setOptions(patch) {
    var next = sanitize(Object.assign({}, options(), patch || {}));
    try { if (ns.store && typeof ns.store.write === 'function') ns.store.write(PREF_KEY, next); else if (root.localStorage) root.localStorage.setItem(PREF_KEY, JSON.stringify(next)); } catch (_) {}
    return next;
  }
  // P3 lane 5: how the chosen views leave the 拼图 page — one montage, or one PNG per view in a zip (the classic
  // 「导出六视角」 wrote six files).  Kept apart from the figure options (their shape is part of lane A's contract).
  var OUT_KEY = 'wssv2:export:views';
  function viewsOutput() {
    var raw = null;
    try { raw = ns.store && typeof ns.store.read === 'function' ? ns.store.read(OUT_KEY) : JSON.parse((root.localStorage && root.localStorage.getItem(OUT_KEY)) || 'null'); } catch (_) { raw = null; }
    return raw && raw.output === 'files' ? 'files' : 'montage';
  }
  function setViewsOutput(v) {
    var val = {output: v === 'files' ? 'files' : 'montage'};
    try { if (ns.store && typeof ns.store.write === 'function') ns.store.write(OUT_KEY, val); else if (root.localStorage) root.localStorage.setItem(OUT_KEY, JSON.stringify(val)); } catch (_) {}
    return val.output;
  }

  // ------------------------------------------------------------------ zip (store mode, CRC-32; the classic batch_export.js writer)
  // files: [{name, bytes: Uint8Array | string, mtime?}] → Uint8Array.  Repeated names get _2, _3 …; names are UTF-8 (flag 0x0800).
  var CRC_TABLE = null;
  function crc32(bytes) {
    if (!CRC_TABLE) {
      CRC_TABLE = new Uint32Array(256);
      for (var n = 0; n < 256; n++) { var c = n; for (var k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; CRC_TABLE[n] = c >>> 0; }
    }
    var crc = 0xFFFFFFFF;
    for (var i = 0; i < bytes.length; i++) crc = CRC_TABLE[(crc ^ bytes[i]) & 0xFF] ^ (crc >>> 8);
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }
  function utf8Bytes(text) {
    if (typeof root.TextEncoder === 'function') return new root.TextEncoder().encode(String(text));
    var bin = unescape(encodeURIComponent(String(text))), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function dosDateTime(date) {
    var d = date instanceof Date && !isNaN(date.getTime()) ? date : new Date();
    var year = Math.min(Math.max(d.getFullYear(), 1980), 2107);
    return {time: ((d.getHours() & 31) << 11) | ((d.getMinutes() & 63) << 5) | ((d.getSeconds() >> 1) & 31),
      day: (((year - 1980) & 127) << 9) | (((d.getMonth() + 1) & 15) << 5) | (d.getDate() & 31)};
  }
  function toBytes(value) {
    if (value instanceof Uint8Array) return value;
    if (typeof value === 'string') return utf8Bytes(value);
    if (value && value.buffer instanceof ArrayBuffer) return new Uint8Array(value.buffer, value.byteOffset || 0, value.byteLength);
    if (value instanceof ArrayBuffer) return new Uint8Array(value);
    throw new Error('zip entry bytes must be a Uint8Array or string');
  }
  function zipStore(files) {
    var entries = [], offset = 0, seen = {};
    (files || []).forEach(function (file) {
      var name = String(file.name || '').replace(/\\/g, '/').replace(/^\/+/, '');
      if (!name) throw new Error('zip entry needs a name');
      var unique = name, n = 2;
      while (seen[unique]) { var dot = name.lastIndexOf('.'); unique = dot > 0 ? name.slice(0, dot) + '_' + n + name.slice(dot) : name + '_' + n; n++; }
      seen[unique] = true;
      var nameBytes = utf8Bytes(unique), bytes = toBytes(file.bytes), crc = crc32(bytes), dt = dosDateTime(file.mtime);
      var local = new Uint8Array(30 + nameBytes.length), v = new DataView(local.buffer);
      v.setUint32(0, 0x04034b50, true); v.setUint16(4, 20, true); v.setUint16(6, 0x0800, true); v.setUint16(8, 0, true);
      v.setUint16(10, dt.time, true); v.setUint16(12, dt.day, true); v.setUint32(14, crc, true);
      v.setUint32(18, bytes.length, true); v.setUint32(22, bytes.length, true); v.setUint16(26, nameBytes.length, true); v.setUint16(28, 0, true);
      local.set(nameBytes, 30);
      entries.push({local: local, bytes: bytes, nameBytes: nameBytes, crc: crc, time: dt.time, day: dt.day, offset: offset});
      offset += local.length + bytes.length;
    });
    var centralStart = offset, central = [];
    entries.forEach(function (e) {
      var rec = new Uint8Array(46 + e.nameBytes.length), v = new DataView(rec.buffer);
      v.setUint32(0, 0x02014b50, true); v.setUint16(4, 20, true); v.setUint16(6, 20, true); v.setUint16(8, 0x0800, true); v.setUint16(10, 0, true);
      v.setUint16(12, e.time, true); v.setUint16(14, e.day, true); v.setUint32(16, e.crc, true);
      v.setUint32(20, e.bytes.length, true); v.setUint32(24, e.bytes.length, true); v.setUint16(28, e.nameBytes.length, true);
      v.setUint16(30, 0, true); v.setUint16(32, 0, true); v.setUint16(34, 0, true); v.setUint16(36, 0, true); v.setUint32(38, 0, true); v.setUint32(42, e.offset, true);
      rec.set(e.nameBytes, 46);
      central.push(rec); offset += rec.length;
    });
    var end = new Uint8Array(22), ev = new DataView(end.buffer);
    ev.setUint32(0, 0x06054b50, true); ev.setUint16(4, 0, true); ev.setUint16(6, 0, true);
    ev.setUint16(8, entries.length, true); ev.setUint16(10, entries.length, true);
    ev.setUint32(12, offset - centralStart, true); ev.setUint32(16, centralStart, true); ev.setUint16(20, 0, true);
    var out = new Uint8Array(offset + 22), at = 0;
    entries.forEach(function (e) { out.set(e.local, at); at += e.local.length; out.set(e.bytes, at); at += e.bytes.length; });
    central.forEach(function (r) { out.set(r, at); at += r.length; });
    out.set(end, at);
    return out;
  }
  // A PNG blob (canvas.toBlob) as bytes for the zip.
  function blobBytes(blob) {
    if (blob && typeof blob.arrayBuffer === 'function') return blob.arrayBuffer().then(function (b) { return new Uint8Array(b); });
    return new Promise(function (resolve, reject) {
      var r = new root.FileReader();
      r.onload = function () { resolve(new Uint8Array(r.result)); };
      r.onerror = function () { reject(r.error || new Error('读取图片失败')); };
      r.readAsArrayBuffer(blob);
    });
  }

  // ------------------------------------------------------------------ words: Chinese → English for figures
  // Longest first.  Branch and finding names fall through to WssReportCommon.englishLabel (the classic table).
  var EN = [
    ['周期指标 TAWSS · OSI', 'Cycle metrics TAWSS · OSI'], ['体内压力与速度', 'Pressure and velocity'], ['峰值 WSS', 'Peak WSS'], ['周期指标', 'Cycle metrics'], ['体场', 'Volume field'],
    ['周期平均壁面切应力', 'Time-averaged WSS'], ['壁面切应力', 'Wall shear stress'], ['振荡剪切指数', 'Oscillatory shear index'], ['相对滞留时间', 'Relative residence time'],
    ['内皮细胞激活势', 'Endothelial cell activation potential'], ['壁面相对压力', 'Wall relative pressure'], ['壁面压力', 'Wall pressure'], ['相对压力', 'Relative pressure'],
    ['速度矢量', 'Velocity'], ['速度大小', 'Speed'], ['穿面速度', 'Through-plane velocity'], ['压力', 'Pressure'], ['速度', 'Speed'],
    ['本截面样本不足，用三维色标', 'too few samples here, 3-D scale'], ['截面色标：', 'Section scale: '], ['本截面 p2–p98', 'this section p2–p98'], ['与三维同', 'same as 3-D'],
    ['；顺流为正', '; downstream positive'], ['两侧同尺度', 'Same scale on both sides'], ['本例自适应', 'Case-adaptive'], ['固定范围', 'Fixed range'],
    ['低剪切窗', 'Low-shear window'], ['高振荡窗', 'High-oscillation window'], ['低值窗', 'Low window'], ['常规窗', 'Standard window'], ['全范围', 'Full range'],
    ['（对数）', ' (log)'], ['（线性）', ' (linear)'], ['（暂定）', ' (provisional)'], ['（仅显示的分支）', ' (shown branches only)'],
    ['灰柱：显示网格顶点分布', 'Grey bars: display-vertex distribution'], ['灰柱：采样点分布', 'Grey bars: sample-point distribution'], ['显示网格顶点分布', 'display-vertex distribution'], ['采样点分布', 'sample-point distribution'],
    ['超出范围按端色', 'Out of range: end colours'], ['高于上限的值按上端色', 'Values above the range take the top colour'], ['低于下限的值按下端色', 'Values below the range take the bottom colour'],
    ['高于上限', 'above'], ['低于下限', 'below'], ['缺失（灰）', 'Missing (grey) '], ['粗线：观察阈值', 'Bold lines: observation thresholds'], ['非临床界值', 'not clinical cut-offs'],
    ['最大 ', 'Max '], ['最小 ', 'Min '],   // P3 lane 5: the persistent maximum / minimum markers (ws_display)
    ['研究用途，非诊断', 'Research use only, not for diagnosis'], ['导出于', 'exported'], ['当前视角', 'Current view'], ['病例', 'Case'], ['截面', 'Section'], ['对数', 'log'], [' 至 ', ' to '],
    ['，', ', '], ['：', ': '], ['（', ' ('], ['）', ')'], ['、', ', '], ['。', '.']
  ];
  // Order: the classic table's longer names (branches, findings: 左髂总, 最低压力 …), then the figure words, then the
  // classic one- and two-character words (前, 分支 …) for whatever Chinese is left — so no pass cuts into another's words.
  function tr(text, lang) {
    var s = text === null || text === undefined ? '' : String(text);
    if (lang !== 'en' || !/[⺀-￿]/.test(s.replace(/[‐-−]/g, ''))) return s;
    s = s.replace(/(\d+) 点/g, '$1 points').replace(/(\d+) 段/g, '$1 bands');
    var c = common(), z = c && c.ZH2EN ? Object.keys(c.ZH2EN).sort(function (a, b) { return b.length - a.length; }) : [];
    var swap = function (from, to) { if (s.indexOf(from) >= 0) s = s.split(from).join(to); };
    z.forEach(function (zh) { if (zh.length >= 3) swap(zh, c.ZH2EN[zh] + ' '); });
    EN.forEach(function (p) { swap(p[0], p[1]); });
    z.forEach(function (zh) { if (zh.length < 3) swap(zh, c.ZH2EN[zh] + ' '); });
    return s.replace(/\s{2,}/g, ' ').replace(/\s+([,.;:)])/g, '$1').replace(/\(\s+/g, '(').trim();
  }

  // ------------------------------------------------------------------ themes
  // White / transparent figures use the light ink; the viewport background follows the stage (dark or light).
  var LIGHT = {dark: false, ink: '#1b2430', ink2: '#465361', ink3: '#77828e', line: '#c6ccd2', chip: 'rgba(255,255,255,0.92)', chipLine: '#c6ccd2'};
  var DARK = {dark: true, ink: '#eef2f6', ink2: '#b6bfca', ink3: '#8d98a5', line: 'rgba(255,255,255,0.28)', chip: 'rgba(22,26,32,0.88)', chipLine: 'rgba(255,255,255,0.24)'};
  function hexDark(hex) {
    var m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(String(hex || '').trim());
    if (!m) return false;
    return 0.2126 * parseInt(m[1], 16) + 0.7152 * parseInt(m[2], 16) + 0.0722 * parseInt(m[3], 16) < 128;
  }
  function stageHex(api) {
    try {
      var el = api && api.state && api.state() && api.state().els && api.state().els.app;
      var v = el && root.getComputedStyle ? String(root.getComputedStyle(el).getPropertyValue('--snapshot-bg') || '').trim() : '';
      return /^#[0-9a-f]{6}$/i.test(v) ? v : (ns.store && ns.store.prefs && ns.store.prefs().stage === 'light' ? '#ffffff' : '#1a1e25');
    } catch (_) { return '#1a1e25'; }
  }
  function themeOf(background, api) {
    if (background === 'transparent') return Object.assign({key: 'transparent', fill: null, render: 'transparent'}, LIGHT);
    if (background === 'viewport') { var hx = stageHex(api); return Object.assign({key: 'viewport', fill: hx, render: hx}, hexDark(hx) ? DARK : LIGHT); }
    return Object.assign({key: 'white', fill: '#ffffff', render: 'white'}, LIGHT);
  }

  // ------------------------------------------------------------------ colour bar (the stage's own drawing)
  function windowText(w) {
    if (!w) return '';
    if (typeof w === 'object') return (w.label || '') + (w.provisional && String(w.label || '').indexOf('暂定') < 0 ? '（暂定）' : '');
    return String(w);
  }
  var FIELD_EN = {wss: 'WSS', tawss: 'TAWSS', osi: 'OSI', rrt: 'RRT', ecap: 'ECAP', pressure: 'Relative pressure', speed: 'Speed', wall_pressure: 'Wall pressure', velocity: 'Velocity'};
  // info = the stage colour bar's info (ns.colorbar); o = {lang, theme, width, graphicHeight}
  function colorbarSVG(info, o) {
    o = o || {};
    if (!info || !info.scale || !ns.colorbar || typeof ns.colorbar.toSVG !== 'function') return '';
    var lang = o.lang === 'en' ? 'en' : 'zh', th = o.theme || LIGHT;
    var i2 = Object.assign({}, info);
    if (lang === 'en') {
      i2.label = FIELD_EN[info.fieldId] || tr(info.label, 'en');
      if (info.windowLabel && typeof info.windowLabel === 'object') i2.windowLabel = Object.assign({}, info.windowLabel, {label: tr(windowText(info.windowLabel), 'en'), provisional: false});
      else i2.windowLabel = tr(info.windowLabel, 'en');
    }
    var svg = ns.colorbar.toSVG(i2, {width: o.width || colorbarWidth(i2, lang), graphicHeight: o.graphicHeight || 300, background: th.fill || null});
    if (lang === 'en') svg = svg.replace(/>(\s*)([^<]*?[⺀-￿][^<]*?)(\s*)</g, function (all, a, text, b) { return '>' + a + escapeXml(tr(unescapeXml(text), 'en')) + b + '<'; });
    if (th.dark) svg = svg.split('#1b2430').join(DARK.ink).split('#465361').join(DARK.ink2).split('#77828e').join(DARK.ink3).split('#c6ccd2').join('#6b7682');
    return svg;
  }
  // Wide enough for the longest line under the bar (the note lines are single lines in toSVG): CJK ≈ 12 px, Latin ≈ 6.7 px.
  function textWidth(s) { var w = 0; String(s || '').split('').forEach(function (ch) { w += /[⺀-￿]/.test(ch) ? 12.2 : 6.8; }); return w; }
  function colorbarWidth(info, lang) {
    var lines = [];
    try { lines = typeof ns.colorbar.noteLines === 'function' ? ns.colorbar.noteLines(info).map(function (n) { return n.text + (n.sub ? '：' + n.sub : ''); }) : []; } catch (_) { lines = []; }
    lines.push(windowText(info.windowLabel) + ' · 对数');
    var w = Math.max.apply(null, lines.map(function (t) { return textWidth(tr(t, lang)); }).concat([150]));
    return Math.max(190, Math.min(420, Math.ceil(w + 10)));
  }
  function escapeXml(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function unescapeXml(s) { return String(s).replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&amp;/g, '&'); }
  function svgSize(svg) {
    var m = /<svg[^>]*\swidth="([\d.]+)"[^>]*\sheight="([\d.]+)"/.exec(svg || '');
    return m ? {w: +m[1], h: +m[2]} : {w: 190, h: 360};
  }
  function loadImage(src) {
    var c = common();
    if (c && typeof c.loadImage === 'function') return c.loadImage(src, root.document);
    return Promise.reject(new Error('image loader missing'));
  }
  // The SVG rasterised at k× on a canvas (crisp at 2× / 4×).
  function svgCanvas(svg, k) {
    var sz = svgSize(svg);
    return loadImage('data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg)).then(function (img) {
      var c = root.document.createElement('canvas');
      c.width = Math.round(sz.w * k); c.height = Math.round(sz.h * k);
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      return c;
    });
  }

  // ------------------------------------------------------------------ context
  function currentApi(api) { return api || API; }
  function stageInfo(api) {
    var S = api && api.state ? api.state() : null, info = null;
    try { info = S && S.colorbarA && typeof S.colorbarA.info === 'function' ? S.colorbarA.info() : null; } catch (_) { info = null; }
    if (!info) { try { var v = api.viewer(); info = v && v.colorbarInfo ? v.colorbarInfo() : null; } catch (_) { info = null; } }
    return info;
  }
  // Words of a figure: title (case · result), subtitle (field · units · window), footer (research line · time).
  function captions(api, o) {
    var cur = api.cur(), m = cur.manifest, lang = o.lang === 'en' ? 'en' : 'zh', info = o.info || stageInfo(api) || {};
    var hide = Boolean(o.hideName), name = hide || !(m.job && m.job.display_name) ? (lang === 'en' ? 'Case' : '病例') : m.job.display_name;
    var result = tr((m.result && m.result.display_name) || '', lang);
    var fid = info.fieldId || cur.field, f = api.fieldById ? api.fieldById(m, cur.field) : null;
    var slice = fid === 'slice';
    var long = !slice && info.fullLabel && info.label !== info.fullLabel ? info.fullLabel : '';
    var enShort = FIELD_EN[fid] || tr(info.label, 'en'), enLong = long ? tr(long, 'en') : '';
    var short = lang === 'en' ? (slice ? tr(info.fullLabel || info.label, 'en') : [enShort, enLong !== enShort ? enLong : ''].filter(Boolean).join(' '))
      : (long ? info.label + ' ' + long : (info.fullLabel || info.label || ''));
    var units = ui().unitText(info.units !== undefined ? info.units : (f && f.units));
    // the toolbar's words for the window (with its range); a section's own scale keeps the section wording.
    // api.windowLabel: a result that is not the one on screen (batch figures) names its window in its own units.
    var wlabel = typeof api.windowLabel === 'function' ? api.windowLabel : (ns.shell && typeof ns.shell.windowLabel === 'function' ? ns.shell.windowLabel : null);
    var wtext = !slice && f && wlabel && !(cur.compare && cur.compare.mode === 'same') ? wlabel(f, cur.window) : '';
    var win = tr(wtext || (info.windowLabel && info.windowLabel.text) || windowText(info.windowLabel), lang);
    var when = ui().time(new Date().toISOString());
    return {name: name, result: result, fieldShort: lang === 'en' ? (FIELD_EN[fid] || tr(info.label, 'en')) : (info.label || ''), units: units,
      title: [name, result].filter(Boolean).join(' · '),
      subtitle: [short + (units ? (lang === 'en' ? ' (' + units + ')' : '（' + units + '）') : ''), win].filter(Boolean).join(' · '),
      footer: lang === 'en' ? 'Research use only, not for diagnosis · exported ' + when : '研究用途，非诊断 · 导出于 ' + when};
  }
  function fieldKey(api, info) { var cur = api.cur(); return info && info.fieldId === 'slice' ? 'slice' : cur.field; }
  // Classic file names: <case>_<view>_<field>_<k>x.<ext> (WssReportCommon.exportFilename).
  function fileName(api, o, view, ext, k) {
    var cur = api.cur(), m = cur.manifest, c = common();
    var caseId = o.hideName || !(m.job && m.job.display_name) ? 'case' : m.job.display_name;
    var spec = {case_id: caseId, view: CLASSIC[view] || view, field: fieldKey(api, o.info), scale: k || 1, ext: ext || 'png'};
    if (c && typeof c.exportFilename === 'function') return c.exportFilename(spec);
    return [caseId, spec.view, spec.field, spec.scale + 'x'].join('_') + '.' + spec.ext;
  }

  // ------------------------------------------------------------------ rendering and composing
  function hasFrame(viewer) { try { return Boolean(viewer && viewer.hasAnatomicalFrame && viewer.hasAnatomicalFrame()); } catch (_) { return false; } }
  function renderView(viewer, view, o) {
    if (!viewer || typeof viewer.renderPose !== 'function') return Promise.reject(new Error('三维视图不可用'));
    var pose = null;
    if (view !== 'current') {
      pose = hasFrame(viewer) && viewer.exportPose ? viewer.exportPose(view) : null;
      if (!pose) return Promise.reject(new Error('没有解剖坐标架，不能取' + (VIEW_ZH[view] || view)));
    }
    return viewer.renderPose({camera: pose, scale: o.scale, background: o.theme.render, labels: o.labels !== false});
  }
  // Layout of a figure in export px (pure, tested): render at the left under the title, colour bar to the right.
  function figureLayout(o) {
    var k = o.k || 1, pad = 18 * k;
    var headH = o.title ? 64 * k : 0, footH = o.footer ? 30 * k : 0;
    var barCol = o.barW ? o.barW + pad : 0;
    var bodyH = Math.max(o.imgH, o.barH ? o.barH + 2 * pad : 0);
    return {W: Math.round(o.imgW + barCol), H: Math.round(headH + bodyH + footH), pad: pad, headH: headH, footH: footH,
      img: {x: 0, y: headH}, bar: o.barW ? {x: o.imgW, y: headH + pad} : null};
  }
  function roundRect(g, x, y, w, h, r) {
    g.beginPath();
    g.moveTo(x + r, y); g.lineTo(x + w - r, y); g.quadraticCurveTo(x + w, y, x + w, y + r);
    g.lineTo(x + w, y + h - r); g.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    g.lineTo(x + r, y + h); g.quadraticCurveTo(x, y + h, x, y + h - r);
    g.lineTo(x, y + r); g.quadraticCurveTo(x, y, x + r, y); g.closePath();
  }
  // The 3-D label chips at the export resolution: laid out again (WssReportCommon.declutterLabels, on the whole render,
  // as on screen) with a thin leader back to their point when moved — the classic composeExport does the same.
  // Marker chips (findings, branch names, annotations) in regular weight, tool labels (measurements) in semibold.
  function layoutChips(render, o) {
    var labels = render.labels || [];
    if (!labels.length || !root.document) return null;
    var k = render.scale || 1, padX = 7 * k, hh = 19 * k, g = root.document.createElement('canvas').getContext('2d');
    if (!g) return null;
    var items = labels.map(function (L) {
      var text = tr(L.text, o.lang), font = (L.kind === 'tool' ? '600 ' : '400 ') + 12 * k + 'px ' + FONT;
      g.font = font;
      return {x: L.x * k, y: L.y * k - 18 * k, w: g.measureText(text).width + 2 * padX, h: hh, text: text, font: font, ax: L.x * k, ay: L.y * k, priority: L.kind === 'tool' ? 1 : 0};
    });
    var c = common();
    var placed = c && typeof c.declutterLabels === 'function' ? c.declutterLabels(items, {maxShift: 140 * k, step: 8 * k, padding: 3 * k, bounds: {width: render.width, height: render.height}}) : items;
    return {k: k, items: items.map(function (it, i) { var p = placed[i] || it; return Object.assign({}, it, {px: p.x, py: p.y, moved: Boolean(p.moved), hidden: Boolean(p.hidden)}); })};
  }
  function chipsBox(lay) {
    if (!lay || !lay.items.length) return null;
    var b = {x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity};
    lay.items.forEach(function (it) {
      if (it.hidden) return;
      b.x0 = Math.min(b.x0, it.px - it.w / 2, it.ax); b.x1 = Math.max(b.x1, it.px + it.w / 2, it.ax);
      b.y0 = Math.min(b.y0, it.py - it.h / 2, it.ay); b.y1 = Math.max(b.y1, it.py + it.h / 2, it.ay);
    });
    return b.x1 > b.x0 ? b : null;
  }
  // box ∪ chips (+ margin), clamped to the render
  function withChips(box, lay, W, H) {
    var cb = chipsBox(lay);
    if (!box || !cb) return box;
    var m = 10 * lay.k, x0 = Math.max(0, Math.min(box.x, cb.x0 - m)), y0 = Math.max(0, Math.min(box.y, cb.y0 - m));
    var x1 = Math.min(W, Math.max(box.x + box.w, cb.x1 + m)), y1 = Math.min(H, Math.max(box.y + box.h, cb.y1 + m));
    return {x: Math.round(x0), y: Math.round(y0), w: Math.round(x1 - x0), h: Math.round(y1 - y0)};
  }
  function paintChips(g, lay, o) {
    if (!lay || !lay.items.length) return 0;
    var k = lay.k, th = o.theme, ox = o.ox || 0, oy = o.oy || 0;
    g.save();
    g.strokeStyle = th.ink2; g.fillStyle = th.ink2; g.lineWidth = Math.max(1, k); g.globalAlpha = 0.7;
    lay.items.forEach(function (it) {
      if (!it.moved || it.hidden) return;
      g.beginPath(); g.moveTo(ox + it.px, oy + it.py + it.h / 2); g.lineTo(ox + it.ax, oy + it.ay); g.stroke();
      g.beginPath(); g.arc(ox + it.ax, oy + it.ay, 2 * k, 0, 2 * Math.PI); g.fill();
    });
    g.globalAlpha = 1; g.textAlign = 'center'; g.textBaseline = 'middle';
    lay.items.forEach(function (it) {
      if (it.hidden) return;
      g.fillStyle = th.chip; g.strokeStyle = th.chipLine; g.lineWidth = Math.max(1, k);
      roundRect(g, ox + it.px - it.w / 2, oy + it.py - it.h / 2, it.w, it.h, 6 * k); g.fill(); g.stroke();
      g.font = it.font; g.fillStyle = th.ink; g.fillText(it.text, ox + it.px, oy + it.py + 0.5 * k);
    });
    g.restore();
    return lay.items.length;
  }
  function drawChips(g, render, o) { return paintChips(g, layoutChips(render, o), o); }
  // 裁边: the part of the render that is not background, plus a margin (render px); null when it cannot be read.
  function contentBox(img, th, k) {
    var c = root.document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    var g = c.getContext('2d'), d;
    g.drawImage(img, 0, 0);
    try { d = g.getImageData(0, 0, c.width, c.height).data; } catch (_) { return null; }
    var W = c.width, H = c.height, step = Math.max(1, Math.round(k)), clear = th.render === 'transparent';
    var br = d[0], bgc = d[1], bb = d[2], x0 = W, y0 = H, x1 = -1, y1 = -1;
    for (var y = 0; y < H; y += step) {
      for (var x = 0; x < W; x += step) {
        var i = (y * W + x) * 4;
        var on = clear ? d[i + 3] > 8 : Math.max(Math.abs(d[i] - br), Math.abs(d[i + 1] - bgc), Math.abs(d[i + 2] - bb)) > 10;
        if (!on) continue;
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
    if (x1 < 0) return null;
    var m = Math.round(28 * k);
    x0 = Math.max(0, x0 - m); y0 = Math.max(0, y0 - m); x1 = Math.min(W, x1 + step + m); y1 = Math.min(H, y1 + step + m);
    return {x: x0, y: y0, w: x1 - x0, h: y1 - y0};
  }
  function measure(font, text) {
    var c = root.document.createElement('canvas'), g = c.getContext('2d');
    g.font = font;
    return text ? g.measureText(text).width : 0;
  }
  // spec = {render, info (null = no colour bar), lang, theme, title, subtitle, footer (null = none), labels, crop, box (a given crop)}
  function composeFigure(spec) {
    var r = spec.render, k = r.scale || 1, th = spec.theme || LIGHT;
    var svg = spec.info ? colorbarSVG(spec.info, {lang: spec.lang, theme: th, graphicHeight: Math.max(220, Math.min(380, Math.round((r.css ? r.css.h : r.height / k) * 0.5)))}) : '';
    return Promise.all([loadImage(r.dataURL), svg ? svgCanvas(svg, k).catch(function () { return null; }) : Promise.resolve(null)]).then(function (res) {
      var img = res[0], bar = res[1];
      var lay = spec.labels ? (spec.layout || layoutChips(r, {lang: spec.lang})) : null;
      var box = spec.box || (spec.crop ? withChips(contentBox(img, th, k), lay, img.width, img.height) : null);
      if (!box) box = {x: 0, y: 0, w: img.width, h: img.height};
      var L = figureLayout({k: k, imgW: box.w, imgH: box.h, barW: bar ? bar.width : 0, barH: bar ? bar.height : 0, title: Boolean(spec.title), footer: Boolean(spec.footer)});
      // the words must fit: widen the frame (image and bar stay together at the left) when a title is longer
      var need = spec.title ? Math.max(measure('600 ' + 19 * k + 'px ' + FONT, spec.title), measure(13 * k + 'px ' + FONT, spec.subtitle), measure(11 * k + 'px ' + FONT, spec.footer)) + 2 * L.pad : 0;
      if (need > L.W) L.W = Math.round(need);
      var c = root.document.createElement('canvas');
      c.width = L.W; c.height = L.H;
      var g = c.getContext('2d');
      if (th.fill) { g.fillStyle = th.fill; g.fillRect(0, 0, L.W, L.H); }
      g.drawImage(img, box.x, box.y, box.w, box.h, L.img.x, L.img.y, box.w, box.h);
      if (lay) {
        g.save(); g.beginPath(); g.rect(L.img.x, L.img.y, box.w, box.h); g.clip();
        paintChips(g, lay, {theme: th, ox: L.img.x - box.x, oy: L.img.y - box.y});
        g.restore();
      }
      if (bar && L.bar) g.drawImage(bar, L.bar.x, L.bar.y);
      g.textBaseline = 'top'; g.textAlign = 'left';
      if (spec.title) {
        g.fillStyle = th.ink; g.font = '600 ' + 19 * k + 'px ' + FONT; g.fillText(spec.title, L.pad, 14 * k, L.W - 2 * L.pad);
        if (spec.subtitle) { g.fillStyle = th.ink2; g.font = 13 * k + 'px ' + FONT; g.fillText(spec.subtitle, L.pad, 40 * k, L.W - 2 * L.pad); }
      }
      if (spec.footer) { g.fillStyle = th.ink3; g.font = 11 * k + 'px ' + FONT; g.fillText(spec.footer, L.pad, L.H - L.footH + 9 * k, L.W - 2 * L.pad); }
      return c;
    });
  }
  // The current view as a figure with the given options.  o = options() + {hideName, scale}
  function figure(api, o) {
    api = currentApi(api);
    var viewer = api.viewer(), th = themeOf(o.background, api), info = stageInfo(api);
    var cap = captions(api, {lang: o.lang, hideName: o.hideName, info: info});
    // a standard view says which one it is (six views as files, batch figures)
    var sub = cap.subtitle + (o.view && o.view !== 'current' && VIEW_ZH[o.view] ? ' · ' + (o.lang === 'en' ? VIEW_EN : VIEW_ZH)[o.view] : '');
    return renderView(viewer, o.view || 'current', {scale: o.scale, theme: th, labels: o.labels}).then(function (r) {
      return composeFigure({render: r, info: o.colorbar ? info : null, lang: o.lang, theme: th, labels: o.labels, crop: o.crop,
        title: o.title ? cap.title : null, subtitle: o.title ? sub : null, footer: o.title ? cap.footer : null}).then(function (canvas) {
        return {canvas: canvas, render: r, captions: cap, info: info};
      });
    });
  }
  // A section map on a canvas of the panel size: the zoom view's drawing (ws_slice.paint, VolumeViewerCore numbers).
  function sliceCanvas(api, W, H, o) {
    var cur = api.cur(), s = cur && cur.slice;
    var comp = s && typeof s.comp === 'function' ? s.comp() : null;
    if (!comp || !ns.slice || typeof ns.slice.paint !== 'function' || !root.document) return null;
    var st = s.state(), k = o.k || 1, core = root.VolumeViewerCore;
    var c = root.document.createElement('canvas');
    c.width = Math.round(W); c.height = Math.round(H);
    var grid = null;
    try { grid = core && typeof core.sliceGrid === 'function' ? core.sliceGrid(comp.data, 240, 240) : null; } catch (_) { grid = null; }
    var bottom = o.withBar ? 120 * k / 2 : 30 * k;
    ns.slice.paint(c, comp, {background: o.transparent ? null : '#ffffff', grid: grid || undefined, points: false, arrows: st.arrows !== false,
      velocity: s.model && s.model.A ? s.model.A.velocity : null, maxArrows: 300, arrowWidth: 1.2 * k, lineWidth: 1.1 * k,
      pad: {left: 20 * k, top: 15 * k, right: 20 * k, bottom: bottom}, font: 11 * k, scaleBar: true, colorbar: Boolean(o.withBar), caption: o.caption || null});
    return c;
  }
  function sliceCaption(api, lang) {
    var s = api.cur().slice, comp = s && s.comp ? s.comp() : null;
    if (!comp) return '';
    var q = ns.slice.quantityLabel(comp.quantity), u = ns.slice.unitsOf(comp.quantity);
    return (lang === 'en' ? 'Section · ' + tr(q, 'en') : '截面 · ' + q) + ' · ' + u;
  }
  function sliceOpen(api) { var cur = api.cur(); return Boolean(cur && cur.slice && typeof cur.slice.comp === 'function' && cur.slice.comp()); }
  // Panel letter in the theme's chip (composeMontage's own letter box is always white).
  function letter(canvas, text, th, k) {
    var g = canvas.getContext('2d');
    g.save(); g.font = '700 ' + 13 * k + 'px ' + FONT;
    var w = g.measureText(text).width + 12 * k, h = 20 * k;
    g.fillStyle = th.chip; g.strokeStyle = th.chipLine; g.lineWidth = Math.max(1, k);
    roundRect(g, 8 * k, 8 * k, w, h, 5 * k); g.fill(); g.stroke();
    g.fillStyle = th.ink; g.textBaseline = 'middle'; g.textAlign = 'center'; g.fillText(text, 8 * k + w / 2, 8 * k + h / 2 + 0.5 * k);
    g.restore();
  }
  // One size for every panel (composeMontage lays out a uniform grid): with 裁边 the panels are cut to the largest
  // content box among them, each around its own content; the section map is then painted at that size.
  function commonCrop(panels, th, k) {
    var boxes = panels.map(function (p) { return contentBox(p.image, th, k); });
    if (boxes.some(function (b) { return !b; })) return panels;
    var W = Math.max.apply(null, boxes.map(function (b) { return b.w; })), H = Math.max.apply(null, boxes.map(function (b) { return b.h; }));
    return panels.map(function (p, i) {
      var b = boxes[i], src = p.image;
      var x = Math.round(Math.max(0, Math.min(src.width - W, b.x + b.w / 2 - W / 2))), y = Math.round(Math.max(0, Math.min(src.height - H, b.y + b.h / 2 - H / 2)));
      var c = root.document.createElement('canvas');
      c.width = Math.min(W, src.width); c.height = Math.min(H, src.height);
      c.getContext('2d').drawImage(src, x, y, c.width, c.height, 0, 0, c.width, c.height);
      return Object.assign({}, p, {image: c});
    });
  }
  // Montage: o = options() + {views, columns, hideName, onStep(i, n, label)}; resolves {canvas, count, skipped}.
  function montage(api, o) {
    api = currentApi(api);
    var viewer = api.viewer(), th = themeOf(o.background, api), info = stageInfo(api), lang = o.lang;
    var views = (o.views || []).filter(function (v) { return v === 'current' || v === 'slice' || VIEWS.indexOf(v) >= 0; });
    var panels = [], skipped = [], k = o.scale;
    var order = views.filter(function (v) { return v !== 'slice'; });
    var wantSlice = views.indexOf('slice') >= 0;
    var chain = Promise.resolve();
    order.forEach(function (view, i) {
      chain = chain.then(function () {
        if (o.onStep) o.onStep(i, order.length + (wantSlice ? 1 : 0), lang === 'en' ? VIEW_EN[view] : VIEW_ZH[view]);
        return renderView(viewer, view, {scale: o.scale, theme: th, labels: o.labels}).then(function (r) {
          return loadImage(r.dataURL).then(function (img) {
            var c = root.document.createElement('canvas');
            c.width = img.width; c.height = img.height;
            var g = c.getContext('2d');
            if (th.fill) { g.fillStyle = th.fill; g.fillRect(0, 0, c.width, c.height); }
            g.drawImage(img, 0, 0);
            if (o.labels) drawChips(g, r, {theme: th, lang: lang});
            k = r.scale;
            panels.push({view: view, image: c, caption: lang === 'en' ? VIEW_EN[view] : VIEW_ZH[view]});
          });
        }, function () { skipped.push(view); });
      });
    });
    return chain.then(function () {
      if (o.crop !== false && panels.length) panels = commonCrop(panels, th, k);
      if (wantSlice) {
        var W = panels.length ? panels[0].image.width : 700 * k, H = panels.length ? panels[0].image.height : 600 * k;
        var cv = sliceOpen(api) ? sliceCanvas(api, W, H, {k: k, transparent: th.key === 'transparent'}) : null;
        if (cv) panels.push({view: 'slice', image: cv, caption: sliceCaption(api, lang)}); else skipped.push('slice');
      }
      if (!panels.length) throw new Error('没有可用的视角');
      panels.forEach(function (p, i) { letter(p.image, String.fromCharCode(97 + i), th, k); });
      var cols = Math.max(1, Math.min(o.columns || 3, panels.length)), rows = Math.ceil(panels.length / cols);
      // words and colour bar grow with the montage (a figure is read at the size of the whole page)
      var grow = Math.max(1, Math.min(2, rows * panels[0].image.height / k / 900));
      var cap = captions(api, {lang: lang, hideName: o.hideName, info: info});
      var svg = o.colorbar && info ? colorbarSVG(info, {lang: lang, theme: th, graphicHeight: Math.max(220, Math.min(420, Math.round(panels[0].image.height / k * 0.55)))}) : '';
      return (svg ? svgCanvas(svg, k * grow).catch(function () { return null; }) : Promise.resolve(null)).then(function (bar) {
        var c = common();
        if (!c || typeof c.composeMontage !== 'function') throw new Error('拼图需要共享库');
        var canvas = c.composeMontage({panels: panels.map(function (p) { return {image: p.image, label: '', caption: p.caption}; }),
          columns: cols, colorbar: bar ? {image: bar} : null, document: root.document, gap: Math.round(12 * k), pad: Math.round(20 * k),
          background: th.fill || 'rgba(0,0,0,0)', ink: th.ink, muted: th.ink2, font: Math.round(13 * k * grow),
          title: o.title ? [cap.name, cap.subtitle].filter(Boolean).join(' · ') : ''});
        if (!canvas) throw new Error('当前环境无法合成拼图画布');
        return {canvas: canvas, count: panels.length, skipped: skipped, views: panels.map(function (p) { return p.view; }), scale: k};
      });
    });
  }
  function toBlob(canvas) {
    return new Promise(function (resolve, reject) {
      if (canvas && typeof canvas.toBlob === 'function') canvas.toBlob(function (b) { if (b) resolve(b); else reject(new Error('PNG 编码失败')); }, 'image/png');
      else reject(new Error('画布不可用'));
    });
  }

  // ------------------------------------------------------------------ downloads
  function download(blob, name) { return ui().downloadBlob(blob, name); }
  function downloadFigure(api, o) {
    api = currentApi(api);
    return figure(api, o).then(function (f) {
      return toBlob(f.canvas).then(function (blob) {
        var name = fileName(api, {hideName: o.hideName, info: f.info}, 'current', 'png', f.render.scale);
        download(blob, name);
        return {name: name, width: f.canvas.width, height: f.canvas.height, downgraded: f.render.downgraded, bytes: blob.size};
      });
    });
  }
  function downloadMontage(api, o) {
    api = currentApi(api);
    return montage(api, o).then(function (m) {
      return toBlob(m.canvas).then(function (blob) {
        var name = fileName(api, {hideName: o.hideName, info: stageInfo(api)}, 'montage', 'png', m.scale);
        download(blob, name);
        return {name: name, count: m.count, skipped: m.skipped, width: m.canvas.width, height: m.canvas.height, blob: blob};
      });
    });
  }
  function downloadColorbar(api, o) {
    api = currentApi(api);
    var info = stageInfo(api);
    var svg = colorbarSVG(info, {lang: o.lang, theme: themeOf(o.background, api)});
    if (!svg) return null;
    var name = fileName(api, {hideName: o.hideName, info: info}, 'colorbar', 'svg', 1);
    download(new root.Blob([svg], {type: 'image/svg+xml'}), name);
    return {name: name, svg: svg};
  }
  // P3 lane 5 (W51): the chosen views as separate PNG files — each one the 图片 page's figure (colour bar, title with the
  // view's name, labels, crop), the section map (when open and chosen) with its own colour bar — in one zip.
  // o = options() + {views, hideName, onStep(i, n, label)}; resolves {files: [{name, bytes, view, width, height}], skipped, scale, info}.
  function viewFiles(api, o) {
    api = currentApi(api);
    var views = (o.views || []).filter(function (v, i, a) { return (v === 'current' || v === 'slice' || VIEWS.indexOf(v) >= 0) && a.indexOf(v) === i; });
    var order = views.filter(function (v) { return v !== 'slice'; });
    var files = [], skipped = [], k = o.scale, info = stageInfo(api), lang = o.lang;
    var chain = Promise.resolve();
    order.forEach(function (view, i) {
      chain = chain.then(function () {
        if (o.onStep) o.onStep(i, views.length, lang === 'en' ? VIEW_EN[view] : VIEW_ZH[view]);
        return figure(api, Object.assign({}, o, {view: view})).then(function (f) {
          k = f.render.scale;
          return toBlob(f.canvas).then(blobBytes).then(function (bytes) {
            files.push({name: fileName(api, {hideName: o.hideName, info: f.info}, view, 'png', f.render.scale), bytes: bytes, view: view, width: f.canvas.width, height: f.canvas.height});
          });
        }, function () { skipped.push(view); });
      });
    });
    return chain.then(function () {
      if (views.indexOf('slice') < 0) return null;
      if (o.onStep) o.onStep(views.length - 1, views.length, lang === 'en' ? VIEW_EN.slice : VIEW_ZH.slice);
      var cv = sliceOpen(api) ? sliceCanvas(api, 700 * k, 600 * k, {k: k, withBar: true, transparent: o.background === 'transparent', caption: sliceCaption(api, lang)}) : null;
      if (!cv) { skipped.push('slice'); return null; }
      return toBlob(cv).then(blobBytes).then(function (bytes) {
        files.push({name: fileName(api, {hideName: o.hideName, info: null}, 'section', 'png', k), bytes: bytes, view: 'slice', width: cv.width, height: cv.height});
      });
    }).then(function () {
      if (!files.length) throw new Error('没有可用的视角');
      return {files: files, skipped: skipped, scale: k, info: info};
    });
  }
  function downloadViews(api, o) {
    api = currentApi(api);
    return viewFiles(api, o).then(function (r) {
      var zip = zipStore(r.files.map(function (f) { return {name: f.name, bytes: f.bytes}; }));
      var name = fileName(api, {hideName: o.hideName, info: null}, 'views', 'zip', r.scale);
      download(new root.Blob([zip], {type: 'application/zip'}), name);
      return {name: name, count: r.files.length, skipped: r.skipped, files: r.files.map(function (f) { return f.name; }), bytes: zip.length};
    });
  }

  // ------------------------------------------------------------------ reproducible link
  function round(x, d) {
    if (typeof x === 'number') return Number.isFinite(x) ? +x.toFixed(d === undefined ? 4 : d) : null;
    if (Array.isArray(x)) return x.map(function (v) { return round(v, d); });
    if (x && typeof x === 'object') { var o = {}; Object.keys(x).forEach(function (k) { o[k] = round(x[k], d); }); return o; }
    return x;
  }
  function utf8(s) {
    if (typeof root.TextEncoder === 'function') { var b = new root.TextEncoder().encode(s), out = ''; for (var i = 0; i < b.length; i++) out += String.fromCharCode(b[i]); return out; }
    return unescape(encodeURIComponent(s));
  }
  function fromUtf8(bin) {
    if (typeof root.TextDecoder === 'function') { var u = new Uint8Array(bin.length); for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new root.TextDecoder().decode(u); }
    return decodeURIComponent(escape(bin));
  }
  function encodeState(st) { return root.btoa(utf8(JSON.stringify(st))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''); }
  function decodeState(text) {
    var s = String(text || '').replace(/-/g, '+').replace(/_/g, '/');
    while (s.length % 4) s += '=';
    var st = JSON.parse(fromUtf8(root.atob(s)));
    if (!st || typeof st !== 'object' || Array.isArray(st) || st.v !== 1) throw new Error('invalid view state');
    return st;
  }
  var LAYER_KEYS = ['outline', 'centerline', 'points', 'trust', 'streamlines', 'wall', 'interior'];
  function viewState(api) {
    api = currentApi(api);
    var cur = api.cur(), v = api.viewer(), st = {};
    try { st = v && v.getState ? v.getState() : {}; } catch (_) { st = {}; }
    var L = {}, S = api.state();
    LAYER_KEYS.forEach(function (k) { if (S && S.layers && typeof S.layers[k] === 'boolean') L[k] = S.layers[k]; });
    var out = {v: 1, run: String(cur.runIdentity || '').slice(0, 16), f: cur.field, w: cur.window || 'adaptive', L: L};
    if (st.camera) out.cam = {p: round(st.camera.position), t: round(st.camera.target), u: round(st.camera.up), fov: round(st.camera.fov, 2)};
    out.cu = cur.cursor && st.cursor ? {s: st.cursor.segmentId, m: round(st.cursor.s_mm, 2)} : null;
    out.sel = st.selection === undefined ? null : st.selection;
    out.br = st.branches === undefined ? null : st.branches;
    out.sl = cur.slice ? round(cur.slice.state()) : null;
    var d = displayState(api, cur, st);
    if (d) out.d = d;
    var x = extensionStates(api);
    if (x) out.x = x;
    return out;
  }

  // ------------------------------------------------------------------ P3 lane 5 (W55): the display in the link
  // d = what the colour scale and the labels look like, so that the link shows the same picture in another browser:
  //   cmap    colour map (the layers menu 色表; store prefs)            light   'soft' | 'flat' (L)
  //   labels  {branches, findings, maxd, annotations} (the layers menu 标注与自动标签; store prefs)
  //   bands   {field id: n}          effective colour bands of every field of the result (0 = continuous)
  //   thr     {field id: [t0,t1,t2]} effective observation thresholds of every field that has them
  //   units   {kind: unit}           display units (ws_display: pressure Pa | mmHg, velocity m/s | cm/s, and any kind
  //                                  the display module adds later)
  //   layers  {stl, peaks, stagnation, vectors}          ws_display's layers   (and any key it adds later)
  //   volume  {opacity, density, width, thin}            volume results only
  // Effective = this result's own choice, else the field's 「设为默认」, else the release (thresholds) / 0 (bands): the
  // receiver sees these values even when its own defaults differ.  A link without d (older links) leaves the display
  // as the receiver has it.  x = {extension id: state} from ns.ext hooks linkState(api) / applyLinkState(state, api).
  function plain(o) { return o && typeof o === 'object' ? JSON.parse(JSON.stringify(o)) : null; }
  function displayState(api, cur, st) {
    var d = {}, P = null, D = ns.display;
    try { P = ns.store && typeof ns.store.prefs === 'function' ? ns.store.prefs() : null; } catch (_) { P = null; }
    if (P) {
      if (typeof P.cmap === 'string') d.cmap = P.cmap;
      if (typeof P.lighting === 'string') d.light = P.lighting;
      if (P.labels && typeof P.labels === 'object') d.labels = plain(P.labels);
    }
    if (D && typeof D.prefs === 'function') {
      var DP = null;
      try { DP = D.prefs(); } catch (_) { DP = null; }
      if (DP) {
        if (DP.units) d.units = plain(DP.units);
        if (DP.layers) d.layers = plain(DP.layers);
        if (DP.volume && cur.manifest && cur.manifest.result && cur.manifest.result.family === 'volume') d.volume = plain(DP.volume);
        var eff = effectiveScale(cur, st, DP.defaults || {});
        if (eff) { d.bands = eff.bands; d.thr = eff.thr; }
      }
    }
    return Object.keys(d).length ? round(d) : null;
  }
  function validThr(t) {
    var D = ns.display;
    if (D && typeof D.validThresholds === 'function') return D.validThresholds(t);
    t = Array.isArray(t) ? t.map(Number) : [];
    return t.length === 3 && t.every(Number.isFinite) && t[0] >= 0 && t[0] < t[1] && t[1] < t[2] ? t : null;
  }
  function shownFields(m) {
    var S = ns.shell;
    if (S && typeof S.visibleFields === 'function') return S.visibleFields(m);
    return ((m && m.fields) || []).filter(function (f) { return f && f.id && f.kind !== 'vector'; });
  }
  function effectiveScale(cur, st, defaults) {
    var m = cur && cur.manifest;
    if (!m) return null;
    var sess = (st && st.display) || {}, sb = sess.bands || {}, stt = sess.thresholds || {}, out = {bands: {}, thr: {}};
    shownFields(m).forEach(function (f) {
      var def = defaults[f.id] || {};
      out.bands[f.id] = sb[f.id] !== undefined ? +sb[f.id] : (def.bands !== undefined ? +def.bands : 0);
      var base = validThr(f.display && f.display.thresholds);
      if (base) out.thr[f.id] = (validThr(stt[f.id]) || validThr(def.thresholds) || base).slice();
    });
    return out;
  }
  function extensionStates(api) {
    var x = {};
    (Array.isArray(ns.ext) ? ns.ext : []).forEach(function (e) {
      if (!e || !e.id || typeof e.linkState !== 'function') return;
      try { var s = e.linkState(api); if (s !== undefined && s !== null) x[e.id] = plain(s); } catch (_) {}
    });
    return Object.keys(x).length ? x : null;
  }
  function sameJSON(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
  // Apply d: the browser's own preferences as if chosen in the menus (they stay chosen), this result's bands and
  // thresholds through the viewer (its display state).  Returns true when the field must be drawn again.
  function applyDisplay(api, cur, d, notes) {
    if (!d || typeof d !== 'object') return false;
    var St = ns.store, D = ns.display, changed = [], redraw = false;
    if (St && typeof St.prefs === 'function' && typeof St.setPrefs === 'function') {
      var P = St.prefs(), patch = {};
      // a colour map this workspace does not know would reset the preference to the default: leave it and say so
      var known = function (c) { try { return typeof St.sanitizePrefs !== 'function' || St.sanitizePrefs(Object.assign({}, P, {cmap: c})).cmap === c; } catch (_) { return false; } };
      if (typeof d.cmap === 'string' && d.cmap !== P.cmap) { if (known(d.cmap)) patch.cmap = d.cmap; else notes.push('这里没有色表 ' + d.cmap); }
      if ((d.light === 'soft' || d.light === 'flat') && d.light !== P.lighting) patch.lighting = d.light;
      if (d.labels && typeof d.labels === 'object') {
        var L = Object.assign({}, P.labels);
        ['branches', 'findings', 'maxd', 'annotations'].forEach(function (k) { if (d.labels[k] !== undefined) L[k] = d.labels[k]; });
        if (!sameJSON(L, P.labels)) patch.labels = L;
      }
      if (Object.keys(patch).length) {
        var next = St.setPrefs(patch);
        if (patch.cmap && next.cmap === d.cmap) { changed.push('色表'); redraw = true; }
        if (patch.lighting) {
          changed.push('光照');
          try { var v0 = api.viewer(); if (v0 && v0.setLighting) v0.setLighting(next.lighting); } catch (_) {}
          if (typeof api.renderToolbar === 'function') api.renderToolbar();
        }
        if (patch.labels) { changed.push('标签'); if (typeof api.drawLabels === 'function') api.drawLabels(); }
      }
    }
    if (D && typeof D.prefs === 'function') {
      var DP = function () { return D.prefs(); };   // read afresh: every setter below replaces the stored object
      if (d.units && typeof d.units === 'object' && typeof D.setUnit === 'function') {
        Object.keys(d.units).forEach(function (kind) {
          var u = d.units[kind];
          if (!DP().units || DP().units[kind] === u) return;
          var table = D.UNITS && D.UNITS[kind];
          if (!table || !Object.prototype.hasOwnProperty.call(table, u)) { notes.push('这里不能用单位 ' + u); return; }
          D.setUnit(kind, u); changed.push('单位');
        });
      }
      if (d.layers && typeof d.layers === 'object' && typeof D.toggleLayer === 'function') {
        Object.keys(d.layers).forEach(function (k) {
          var L = DP().layers;
          if (typeof d.layers[k] === 'boolean' && L && typeof L[k] === 'boolean' && L[k] !== d.layers[k]) { D.toggleLayer(k); changed.push('图层'); }
        });
      }
      if (d.volume && typeof d.volume === 'object' && DP().volume && typeof D.savePrefs === 'function') {
        var vol = Object.assign({}, DP().volume, d.volume);
        if (!sameJSON(vol, DP().volume)) { D.savePrefs(Object.assign({}, DP(), {volume: vol})); if (D.refreshAll) D.refreshAll(); changed.push('外壁与流线'); }
      }
    }
    if ((d.bands && typeof d.bands === 'object') || (d.thr && typeof d.thr === 'object')) {
      var v = api.viewer();
      if (v && typeof v.applyState === 'function') {
        try { v.applyState({display: {v: 1, bands: d.bands || {}, thresholds: d.thr || {}}}, {animate: false}); } catch (_) {}
        redraw = true;
      }
    }
    if (changed.length) notes.push('已按链接设置' + changed.filter(function (c, i, a) { return a.indexOf(c) === i; }).join('、'));
    return redraw;
  }
  function applyExtensions(api, x, notes) {
    if (!x || typeof x !== 'object') return Promise.resolve();
    var jobs = [];
    (Array.isArray(ns.ext) ? ns.ext : []).forEach(function (e) {
      if (!e || !e.id || x[e.id] === undefined || typeof e.applyLinkState !== 'function') return;
      try { jobs.push(Promise.resolve(e.applyLinkState(x[e.id], api)).catch(function () {})); } catch (_) {}
    });
    var unknown = Object.keys(x).filter(function (id) { return !(ns.ext || []).some(function (e) { return e && e.id === id && typeof e.applyLinkState === 'function'; }); });
    if (unknown.length) notes.push('链接里有这里没有的显示设置（' + unknown.join('、') + '），已跳过');
    return Promise.all(jobs);
  }
  function linkURL(api) {
    api = currentApi(api);
    var cur = api.cur(), r = api.parseHash(root.location.hash);
    if (r.jobId !== cur.jobId) r = {jobId: cur.jobId, v: null, f: null, cmp: null, q: null, bm: null};
    r = Object.assign({}, r, {f: cur.field, q: null, bm: null});
    var hash = api.buildHash(r);
    hash += (hash.indexOf('?') >= 0 ? '&' : '?') + 'view=' + encodeState(viewState(api));
    return String(root.location.href || '').split('#')[0] + hash;
  }
  function linkParam(hash) { var m = /[?&]view=([A-Za-z0-9_-]+)/.exec(String(hash || '')); return m ? m[1] : null; }
  function waitFor(test, ms) {
    return new Promise(function (resolve) {
      var t0 = Date.now();
      (function poll() { var v = test(); if (v || Date.now() - t0 > (ms || 4000)) resolve(Boolean(v)); else setTimeout(poll, 40); })();
    });
  }
  // Apply a view state: field and window, layers, then the section (async: its arrays), the cursor (async), and the
  // camera last (opening the section turns the view to it).  o.skipField: the bookmark restore has set the field.
  function applyView(api, st, o) {
    api = currentApi(api); o = o || {};
    var cur = api.cur();
    if (!cur || !cur.manifest || !st) return Promise.resolve(false);
    var notes = [];
    if (st.run && cur.runIdentity && String(cur.runIdentity).slice(0, st.run.length) !== st.run) notes.push('链接来自这份结果的另一次计算，已按可用项恢复');
    // P3 lane 5: the display first (colour map, bands, thresholds, units …), so the field is drawn once with it
    var redraw = st.d ? applyDisplay(api, cur, st.d, notes) : false;
    if (!o.skipField && st.f) {
      if (api.fieldById(cur.manifest, st.f)) { if (redraw || st.f !== cur.field || JSON.stringify(st.w || 'adaptive') !== JSON.stringify(cur.window)) api.applyField(st.f, st.w || 'adaptive'); }
      else { notes.push('这份结果没有字段 ' + st.f); if (redraw) api.applyField(cur.field, cur.window); }
    } else if (redraw) api.applyField(cur.field, cur.window);
    if (st.L && typeof st.L === 'object') {
      var S = api.state(), changed = false;
      LAYER_KEYS.forEach(function (k) { if (typeof st.L[k] === 'boolean' && S.layers && S.layers[k] !== st.L[k]) { S.layers[k] = st.L[k]; changed = true; } });
      if (changed) api.setLayers();
    }
    var chain = st.x ? applyExtensions(api, st.x, notes) : Promise.resolve();
    if (st.sl !== undefined) chain = chain.then(function () { return applySlice(api, cur, st.sl, notes); });
    if (st.cu !== undefined) chain = chain.then(function () { return applyCursor(api, cur, st.cu); });
    return chain.then(function () {
      if (api.cur() !== cur) return false;
      var v = api.viewer(), vs = {};
      if (st.cam && st.cam.p && st.cam.t) vs.camera = {position: st.cam.p, target: st.cam.t, up: st.cam.u || [0, 1, 0], fov: st.cam.fov};
      if (st.br !== undefined) vs.branches = st.br;
      if (st.sel !== undefined && st.sel !== null) vs.selection = st.sel;
      if (v && typeof v.applyState === 'function' && Object.keys(vs).length) { try { v.applyState(vs, {animate: false}); } catch (_) {} }
      if (st.sel !== undefined) cur.selection = st.sel;
      api.saveViewSoon();
      return notes;
    });
  }
  function applySlice(api, cur, sl, notes) {
    if (sl === null) { if (cur.slice) api.exitSlice(); return Promise.resolve(true); }
    if (!sl || typeof sl !== 'object') return Promise.resolve(false);
    if (!ns.slice || !cur.result || !ns.slice.supported(cur.result)) { notes.push('这份结果不能看截面'); return Promise.resolve(false); }
    if (cur.slice) { cur.slice.set(sl); return Promise.resolve(true); }
    if (cur.compare || cur.split) { notes.push('截面在单视口里用，比较或并排时没有恢复截面'); return Promise.resolve(false); }
    cur.sliceState = sl;
    api.toggleSlice();
    return waitFor(function () { return api.cur() !== cur || cur.slice; }, 8000).then(function () {
      if (api.cur() === cur && cur.slice) { cur.slice.set(sl); return true; }
      return false;
    });
  }
  function applyCursor(api, cur, cu) {
    if (cu === null) { if (cur.cursor) api.toggleCursor(); return Promise.resolve(true); }
    if (!cu || !Number.isFinite(+cu.s) || !Number.isFinite(+cu.m)) return Promise.resolve(false);
    var set = function () { try { cur.cursor.set(+cu.s, +cu.m); api.cursorMoved(); return true; } catch (_) { return false; } };
    if (cur.cursor) return Promise.resolve(set());
    api.toggleCursor();
    return waitFor(function () { return api.cur() !== cur || cur.cursor; }, 4000).then(function () { return api.cur() === cur && cur.cursor ? set() : false; });
  }
  function copyLink(api, done) {
    api = currentApi(api);
    var url;
    try { url = linkURL(api); } catch (e) { ui().toast('链接没有生成：' + (e && e.message || e), {kind: 'error'}); return null; }
    var ok = function () { ui().toast('已复制复现链接：打开它就回到同样的字段、色标、视角' + (api.cur().slice ? '和截面' : '') + '。', {kind: 'ok', ms: 4000}); if (done) done(url, true); };
    var fail = function () { if (done) done(url, false); else ui().toast('浏览器不允许复制；链接在导出对话框底部。', {kind: 'info'}); };
    try { if (root.navigator && root.navigator.clipboard && root.navigator.clipboard.writeText) { root.navigator.clipboard.writeText(url).then(ok, fail); return url; } } catch (_) {}
    fail();
    return url;
  }
  var hashBound = false;
  // The link is read once: its view= part leaves the address bar (the rest of the route stays), so a later reload of
  // the same result (after a save, a conflict) does not throw the reader back; the reading position is saved as usual.
  function withoutView(hash) { return String(hash || '').replace(/([?&])view=[A-Za-z0-9_-]*(&|$)/, function (all, a, b) { return b ? a : ''; }); }
  // hash: the address as it was when the result opened / the hash changed (the shell may rewrite it right after)
  function applyFromHash(api, hash) {
    hash = hash === undefined ? root.location.hash : hash;
    var cur = api.cur(), p = linkParam(hash);
    if (!cur || !cur.result || !p) return false;
    var r = api.parseHash(hash);
    if (r.jobId && r.jobId !== cur.jobId) return false;
    var st;
    try { st = decodeState(p); } catch (_) { ui().toast('链接里的视图状态无法读取，已按默认打开。', {kind: 'info'}); return false; }
    try { root.history.replaceState(null, '', withoutView(root.location.hash)); } catch (_) {}
    applyView(api, st).then(function (notes) {
      if (Array.isArray(notes) && notes.length) ui().toast(notes.join('；') + '。', {kind: 'info', ms: 6000});
    });
    return true;
  }
  function bindHash() {
    if (hashBound || typeof root.addEventListener !== 'function') return;
    hashBound = true;
    // A link pasted while its result is already open: the shell applies only its own route extras (no onResult),
    // so the view is applied here, after the shell's own hashchange handling.
    root.addEventListener('hashchange', function (e) {
      var url = e && typeof e.newURL === 'string' ? e.newURL : '', hash = url.indexOf('#') >= 0 ? url.slice(url.indexOf('#')) : root.location.hash;
      if (!linkParam(hash)) return;
      setTimeout(function () {
        if (!API) return;
        var cur = API.cur(), r = API.parseHash(hash);
        if (cur && cur.result && r.jobId === cur.jobId) applyFromHash(API, hash);
      }, 0);
    });
  }

  // ------------------------------------------------------------------ bookmarks: the section rides along
  function bookmarkExtra(runIdentity) {
    var api = API, cur = api && api.cur ? api.cur() : null;
    if (!cur || !cur.result || (runIdentity && cur.runIdentity !== runIdentity)) return undefined;
    if (!ns.slice || !ns.slice.supported || !ns.slice.supported(cur.result)) return undefined;
    return {slice: cur.slice ? round(cur.slice.state()) : null};
  }
  // After the shell restored a bookmark (field, window, viewer state, cursor): the section, then its camera again.
  function afterBookmark(bm) {
    var api = API;
    if (!api || !bm || bm.slice === undefined) return Promise.resolve(false);
    var cur = api.cur();
    if (!cur || !cur.result) return Promise.resolve(false);
    var cam = bm.state && bm.state.camera ? bm.state.camera : bm.camera;
    var st = {sl: bm.slice};
    if (bm.slice && bm.cursor) st.cu = {s: bm.cursor.segmentId, m: bm.cursor.s_mm};   // opening the section closes the cursor
    if (cam && cam.position) st.cam = {p: cam.position, t: cam.target, u: cam.up, fov: cam.fov};
    if (!bm.slice && !cur.slice) return Promise.resolve(true);
    return applyView(api, st, {skipField: true});
  }

  // ------------------------------------------------------------------ one-page figures (classic snapshots format)
  function b64(dataURL) { return String(dataURL || '').replace(/^data:image\/png;base64,/, ''); }
  function b64Bytes(s) { return Math.floor(String(s).length * 3 / 4); }
  // The classic set: front / left / top / current (+ the section when it is open), 2×, white, colour bar, Chinese.
  function onepagePlan(api) {
    var viewer = api.viewer(), plan = hasFrame(viewer) ? ONEPAGE_PLAN.slice() : [];
    plan = plan.concat([['current', 'current']]);
    if (sliceOpen(api)) plan.push(['slice', 'slice']);
    return plan;
  }
  function snapshotCaption(api, view, info) {
    if (view === 'slice') return sliceCaption(api, 'zh');
    var cap = captions(api, {lang: 'zh', info: info});
    return [VIEW_ZH[view], cap.fieldShort || '', cap.units].filter(Boolean).join(' · ');
  }
  // The 3-D figures share one crop size (the one-pager lays them out in a row), each around its own content.
  function onepageImages(api, o) {
    api = currentApi(api); o = o || {};
    var viewer = api.viewer(), th = themeOf('white', api), info = stageInfo(api), images = [], skipped = [], renders = [];
    var plan = onepagePlan(api), chain = Promise.resolve();
    plan.forEach(function (p, i) {
      if (p[1] === 'slice') return;
      chain = chain.then(function () {
        if (o.onStep) o.onStep(i, plan.length, VIEW_ZH[p[1]]);
        return renderView(viewer, p[1], {scale: 2, theme: th, labels: o.labels !== false}).then(function (r) {
          return loadImage(r.dataURL).then(function (img) {
            var lay = o.labels !== false ? layoutChips(r, {lang: 'zh'}) : null;
            renders.push({name: p[0], view: p[1], render: r, layout: lay, box: withChips(contentBox(img, th, r.scale), lay, img.width, img.height), w: img.width, h: img.height});
          });
        }, function () { skipped.push(p[1]); });
      });
    });
    return chain.then(function () {
      var boxed = renders.filter(function (x) { return x.box; }), W = 0, H = 0;
      boxed.forEach(function (x) { W = Math.max(W, x.box.w); H = Math.max(H, x.box.h); });
      var jobs = renders.map(function (x) {
        var box = null;
        if (boxed.length === renders.length && W && H) {
          var bw = Math.min(W, x.w), bh = Math.min(H, x.h);
          box = {x: Math.round(Math.max(0, Math.min(x.w - bw, x.box.x + x.box.w / 2 - bw / 2))), y: Math.round(Math.max(0, Math.min(x.h - bh, x.box.y + x.box.h / 2 - bh / 2))), w: bw, h: bh};
        }
        return composeFigure({render: x.render, info: info, lang: 'zh', theme: th, labels: o.labels !== false, layout: x.layout, title: null, subtitle: null, footer: null, box: box}).then(function (canvas) {
          images.push({name: x.name, view: x.name, png_base64: b64(canvas.toDataURL('image/png')), width: canvas.width, height: canvas.height, caption: snapshotCaption(api, x.view, info)});
        });
      });
      return Promise.all(jobs);
    }).then(function () {
      if (plan.some(function (p) { return p[1] === 'slice'; })) {
        if (o.onStep) o.onStep(plan.length - 1, plan.length, VIEW_ZH.slice);
        var cv = sliceCanvas(api, 1400, 1200, {k: 2, withBar: true, caption: null});
        if (cv) images.push({name: 'slice', view: 'slice', png_base64: b64(cv.toDataURL('image/png')), width: cv.width, height: cv.height, caption: snapshotCaption(api, 'slice', info)});
        else skipped.push('slice');
      }
      var rank = {front: 0, left: 1, top: 2, current: 3, slice: 4};
      images.sort(function (a, b) { return rank[a.name] - rank[b.name]; });
      return {images: images, skipped: skipped};
    });
  }
  function checkSizes(images) {
    var total = images.reduce(function (s, i) { return s + b64Bytes(i.png_base64); }, 0);
    if (images.some(function (i) { return b64Bytes(i.png_base64) > SNAPSHOT_MAX; }) || total > SNAPSHOT_TOTAL) return '配图过大（单张上限 6 MB、合计 24 MB）：请缩小窗口后重试。';
    return null;
  }
  function uploadOnepage(api, o) {
    api = currentApi(api); o = o || {};
    var cur = api.cur();
    if (api.offline() || !api.api()) return Promise.reject(new Error('离线报告不能写回服务端。'));
    return onepageImages(api, o).then(function (res) {
      if (!res.images.length) throw new Error('没有可用配图');
      var tooBig = checkSizes(res.images);
      if (tooBig) throw new Error(tooBig);
      if (o.onUpload) o.onUpload(res.images.length);
      return api.api().request('/api/jobs/' + encodeURIComponent(cur.jobId) + '/snapshots', {method: 'POST', body: {images: res.images, replace: true}, timeout: 120000}).then(function (out) {
        var n = out && out.snapshots && Array.isArray(out.snapshots.items) ? out.snapshots.items.length : res.images.length;
        if (cur.job) cur.job.snapshots = out && out.snapshots ? out.snapshots : cur.job.snapshots;
        return {count: n, images: res.images, skipped: res.skipped, document: out && out.snapshots};
      });
    });
  }

  // ------------------------------------------------------------------ print (a print-only A4 page)
  var printToken = 0;
  function printView(api, o) {
    api = currentApi(api); o = Object.assign({}, options(), o || {});
    var h = ui().h, doc = root.document;
    var spec = {scale: 2, background: 'white', lang: o.lang, colorbar: true, title: false, labels: o.labels, crop: true, hideName: o.hideName};
    return figure(api, spec).then(function (f) {
      var cap = captions(api, {lang: o.lang, hideName: o.hideName, info: f.info});
      var url = f.canvas.toDataURL('image/png');
      var host = ui().host('ws-print', 'div', 'ws-print');
      var land = f.canvas.width >= f.canvas.height;
      var img = h('img', {'class': 'wp-img', alt: cap.subtitle});
      ui().fill(host, h('style', {text: '@media print { @page { size: A4 ' + (land ? 'landscape' : 'portrait') + '; margin: 12mm; } }'}),
        h('div', {'class': 'wp-head'}, h('h1', {text: cap.title}), h('p', {text: cap.subtitle})), img,
        h('p', {'class': 'wp-foot', text: cap.footer}));
      host.className = 'ws-print' + (land ? ' wp-land' : ' wp-port');
      if (doc.body && doc.body.classList) doc.body.classList.add('ws-printing');
      var mine = ++printToken;
      var cleanup = function () { if (mine !== printToken) return; if (doc.body && doc.body.classList) doc.body.classList.remove('ws-printing'); host.replaceChildren(); };
      return new Promise(function (resolve) {
        var go = function () {
          var after = function () { if (typeof root.removeEventListener === 'function') root.removeEventListener('afterprint', after); if (!o.keep) cleanup(); resolve({width: f.canvas.width, height: f.canvas.height, title: cap.title}); };
          if (typeof root.addEventListener === 'function') root.addEventListener('afterprint', after);
          try { (mod._print || root.print).call(root); } catch (_) { after(); return; }
          // print() blocks in most browsers; where it returns at once, afterprint follows later — clean up after a while
          setTimeout(after, 60000);
        };
        img.onload = function () { setTimeout(go, 60); };
        img.onerror = function () { cleanup(); resolve(null); };
        img.src = url;
      });
    });
  }

  // ------------------------------------------------------------------ P4 lane B (W58 / W59): Ctrl+P and the browser's print
  // The viewports draw with WebGL without a kept drawing buffer (preserveDrawingBuffer false), so the browser's own print
  // would show a blank 3-D view.  Ctrl / ⌘ + P on an open result therefore goes to the print-view page (printView: the
  // current view, colour bar and caption on one A4 page — the 工具 tab's 打印).  A print started another way (the
  // browser's menu) gets a picture all the same: on beforeprint every shown viewport is drawn now and copied, in the same
  // task, into a 2-D canvas laid over its WebGL canvas (.ws-print-snap, shown only in print); afterprint removes them.
  var printSnaps = [];
  function isPrintKey(e) {
    return Boolean(e && !e.altKey && !e.shiftKey && (e.ctrlKey || e.metaKey) && (e.key === 'p' || e.key === 'P'));
  }
  function printKey(e) {
    if (!isPrintKey(e)) return false;
    var api = API, cur = api && typeof api.cur === 'function' ? api.cur() : null, doc = root.document;
    if (!cur || !cur.manifest) return false;                                            // home page: the browser's own print
    var v = api.viewer ? api.viewer() : null;
    if (!v || typeof v.renderPose !== 'function') return false;                         // no 3-D view to draw
    if (e.preventDefault) e.preventDefault();
    if (doc && doc.body && doc.body.classList && doc.body.classList.contains('ws-printing')) return true;   // already printing
    var o = options(); o.hideName = Boolean(api.hideName && api.hideName());
    Promise.resolve().then(function () { return mod.printView(api, o); }).catch(function (err) { failToast('打印失败', err); });
    return true;
  }
  function clearSnaps() {
    printSnaps.forEach(function (c) { if (c.parentNode) c.parentNode.removeChild(c); });
    printSnaps = [];
  }
  // Copies of the shown viewports; returns how many.  Not while printView's own page is printing.
  function snapViewports(api) {
    var doc = root.document;
    clearSnaps();
    if (!doc || !doc.body || (doc.body.classList && doc.body.classList.contains('ws-printing'))) return 0;
    var S = api && typeof api.state === 'function' ? api.state() : null;
    if (!S || !api.cur || !api.cur()) return 0;
    [S.viewerA, S.viewerB].forEach(function (v) {
      var gl = v && v.canvasElement;
      if (!gl || !gl.parentNode || typeof v.renderNow !== 'function' || !(gl.width > 0 && gl.height > 0)) return;
      if (typeof v.isDisposed === 'function' && v.isDisposed()) return;
      var copy = doc.createElement('canvas'), g = null;
      copy.width = gl.width; copy.height = gl.height;
      try { g = copy.getContext('2d'); } catch (_) { g = null; }
      if (!g) return;
      try { v.renderNow(); g.drawImage(gl, 0, 0); } catch (_) { return; }               // same task: the frame is still there
      copy.className = 'ws-print-snap';
      copy.setAttribute('aria-hidden', 'true');
      gl.parentNode.insertBefore(copy, gl.nextSibling);
      printSnaps.push(copy);
    });
    return printSnaps.length;
  }
  function bindPrint() {
    var doc = root.document;
    if (bindPrint.done || !doc || typeof root.addEventListener !== 'function' || typeof doc.addEventListener !== 'function') return;
    bindPrint.done = true;
    doc.addEventListener('keydown', function (e) { printKey(e); }, true);
    root.addEventListener('beforeprint', function () { snapViewports(API); });
    root.addEventListener('afterprint', function () { clearSnaps(); });
  }
  bindPrint();

  // ------------------------------------------------------------------ panes of the export dialog
  // ctx (from ws_export) = {hideName (forced in an offline copy with the name hidden), offline, jobId, state: {hideName}}
  var paneToken = 0;
  function segControl(items, value, onPick, label) {
    var h = ui().h, box = h('div', {'class': 'seg exp-seg', role: 'group', 'aria-label': label});
    items.forEach(function (it) {
      var b = h('button', {type: 'button', 'class': 'seg-btn' + (String(it.value) === String(value) ? ' on' : ''), 'aria-pressed': String(String(it.value) === String(value)), text: it.label, title: it.title || null});
      b.addEventListener('click', function () {
        for (var i = 0; i < box.children.length; i++) { box.children[i].classList.remove('on'); box.children[i].setAttribute('aria-pressed', 'false'); }
        b.classList.add('on'); b.setAttribute('aria-pressed', 'true');
        onPick(it.value);
      });
      box.appendChild(b);
    });
    return box;
  }
  function pill(label, on, onToggle, opts) {
    var h = ui().h;
    opts = opts || {};
    var b = h('button', {type: 'button', 'class': 'pill exp-pill' + (on ? ' on' : ''), 'aria-pressed': String(Boolean(on)), text: label, title: opts.title || null, disabled: Boolean(opts.disabled)});
    b.addEventListener('click', function () { if (b.disabled) return; var next = !b.classList.contains('on'); b.classList.toggle('on', next); b.setAttribute('aria-pressed', String(next)); onToggle(next); });
    return b;
  }
  function optRow(label, control, tip) {
    var h = ui().h;
    return h('div', {'class': 'exp-row'}, h('span', {'class': 'exp-k', text: label}), control, tip ? ui().infoTip(tip) : null);
  }
  function figureOpts(ctx) { var o = options(); o.hideName = Boolean(ctx.state.hideName); return o; }
  function commonRows(ctx, onChange) {
    var o = options(), h = ui().h;
    var set = function (patch) { setOptions(patch); onChange(); };
    var rows = [
      optRow('倍率', segControl([{value: 1, label: '1×'}, {value: 2, label: '2×'}, {value: 4, label: '4×'}], o.scale, function (v) { set({scale: +v}); }, '倍率'), '按视口大小的倍数渲染；显卡不支持时自动降到 1×。'),
      optRow('背景', segControl([{value: 'viewport', label: '视口'}, {value: 'white', label: '白'}, {value: 'transparent', label: '透明'}], o.background, function (v) { set({background: v}); }, '背景'),
        '视口：与当前视口同色（深色或浅色）；透明：PNG 透明底，文字按浅底配色。'),
      optRow('文字', segControl([{value: 'zh', label: '中文'}, {value: 'en', label: 'English'}], o.lang, function (v) { set({lang: v}); }, '文字'), '标题、色条和标签的语言；数字不变。'),
      optRow('包含', h('div', {'class': 'exp-pills'},
        pill('色条', o.colorbar, function (on) { set({colorbar: on}); }, {title: '右侧色条：与视口里的色条同一张图（范围、刻度、阈值线、分布）'}),
        pill('标题', o.title, function (on) { set({title: on}); }, {title: '上方病例名、结果、字段与色标窗；下方研究用途与导出时间'}),
        pill('标签', o.labels, function (on) { set({labels: on}); }, {title: '三维里显示的发现、标注、测量标签，按导出分辨率重新排布'}),
        pill('裁边', o.crop, function (on) { set({crop: on}); }, {title: '去掉血管周围的空白（视口里的视角不变，只裁掉背景）'}),
        pill('隐藏病例名', ctx.state.hideName, function (on) { ctx.state.hideName = on; onChange(); }, {disabled: Boolean(ctx.hideName), title: ctx.hideName ? '这份离线报告导出时已隐藏病例名' : '标题和文件名里用「病例」代替病例名'})))
    ];
    return rows;
  }
  function busy(btn, on) { if (btn) { btn.disabled = Boolean(on); btn.classList.toggle('is-busy', Boolean(on)); } }
  function failToast(prefix, e) { ui().toast(prefix + '：' + (e && e.message || e), {kind: 'error'}); }
  function paneFigure(el, ctx) {
    var h = ui().h, api = API, token = ++paneToken;
    var img = h('img', {'class': 'exp-preview-img', alt: '导出预览'});
    var size = h('span', {'class': 'exp-size'});
    var box = h('div', {'class': 'exp-preview'}, img, h('div', {'class': 'exp-preview-foot'}, size));
    var timer = null, seq = 0;
    function preview() {
      if (timer) clearTimeout(timer);
      timer = setTimeout(function () {
        timer = null;
        var my = ++seq, o = figureOpts(ctx);
        box.classList.add('is-loading');
        figure(api, Object.assign({}, o, {scale: 1})).then(function (f) {
          if (my !== seq || token !== paneToken) return;
          img.src = f.canvas.toDataURL('image/png');
          size.textContent = o.scale + '× · ' + f.canvas.width * o.scale + ' × ' + f.canvas.height * o.scale + ' px';
          box.classList.remove('is-loading');
        }, function (e) { if (my === seq) { box.classList.remove('is-loading'); size.textContent = '预览失败：' + (e && e.message || e); } });
      }, 120);
    }
    var png = ui().button('下载 PNG', function () {
      busy(png, true);
      downloadFigure(api, figureOpts(ctx)).then(function (r) {
        ui().toast('已导出 ' + r.width + ' × ' + r.height + ' PNG' + (r.downgraded ? '（显卡不支持该倍率，已降到 1×）' : '') + '。', {kind: 'ok', ms: 3500});
      }, function (e) { failToast('导出失败', e); }).then(function () { busy(png, false); });
    }, {icon: 'download', kind: 'primary'});
    var svg = ui().button('色标 SVG', function () {
      var r = downloadColorbar(api, figureOpts(ctx));
      if (!r) ui().toast('没有可导出的色条。', {kind: 'info'});
    }, {icon: 'fig-colorbar', title: '当前色条单独存成 SVG（矢量，可在排版软件里改字）'});
    var prt = ui().button('打印', function () {
      busy(prt, true);
      printView(api, figureOpts(ctx)).catch(function (e) { failToast('打印失败', e); }).then(function () { busy(prt, false); });
    }, {icon: 'print', title: '当前视图单独排成一页（A4），不含界面'});
    ui().fill(el, h('div', {'class': 'exp-figure'}, box,
      h('div', {'class': 'exp-opts'}, commonRows(ctx, preview), h('div', {'class': 'exp-actions'}, png, svg, prt))));
    preview();
    return {refresh: preview, dispose: function () { if (timer) clearTimeout(timer); seq++; }};
  }
  function paneMontage(el, ctx) {
    var h = ui().h, api = API, o = options(), viewer = api.viewer(), frame = hasFrame(viewer), sliceOn = sliceOpen(api);
    var picked = o.views.slice();
    if (!sliceOn) picked = picked.filter(function (v) { return v !== 'slice'; });
    var status = h('p', {'class': 'exp-status muted'});
    var thumb = h('div', {'class': 'exp-thumb'});
    var save = function () { setOptions({views: picked.filter(function (v) { return v !== 'slice' || sliceOn; })}); };
    var pills = VIEWS.map(function (v) {
      return pill(VIEW_SHORT[v], frame && picked.indexOf(v) >= 0, function (on) { picked = picked.filter(function (x) { return x !== v; }); if (on) picked.push(v); save(); }, {disabled: !frame, title: frame ? VIEW_ZH[v] : '没有解剖坐标架，只能拼当前视角'});
    });
    pills.push(pill('当前', picked.indexOf('current') >= 0, function (on) { picked = picked.filter(function (x) { return x !== 'current'; }); if (on) picked.push('current'); save(); }, {title: '当前视角'}));
    if (sliceOn) pills.push(pill('截面', picked.indexOf('slice') >= 0, function (on) { picked = picked.filter(function (x) { return x !== 'slice'; }); if (on) picked.push('slice'); save(); }, {title: '截面工具当前的截面图（与视口同色标）'}));
    var six = ui().button('六视角', function () {
      picked = VIEWS.slice(); save();
      pills.forEach(function (p, i) { var on = i < 6; p.classList.toggle('on', on); p.setAttribute('aria-pressed', String(on)); });
    }, {kind: 'link', cls: 'btn-sm', disabled: !frame, title: '前、后、左、右、头、足'});
    var out = viewsOutput(), files = out === 'files';
    var go = ui().button(files ? '导出单张 zip' : '导出拼图 PNG', function () {
      var views = VIEWS.concat(['current', 'slice']).filter(function (v) { return picked.indexOf(v) >= 0 && (frame || VIEWS.indexOf(v) < 0); });
      if (!views.length) { status.textContent = '至少选一个视角。'; return; }
      busy(go, true);
      var run = Object.assign(figureOpts(ctx), {views: views, columns: options().columns, onStep: function (i, n, label) { status.textContent = '正在渲染 ' + label + '（' + (i + 1) + ' / ' + n + '）…'; }});
      var skippedText = function (r) { return r.skipped.length ? ' · 跳过 ' + r.skipped.map(function (v) { return VIEW_ZH[v]; }).join('、') : ''; };
      if (files) {
        downloadViews(api, run).then(function (r) {
          status.textContent = '已导出 ' + r.count + ' 张 PNG（zip）' + skippedText(r);
          ui().fill(thumb);
        }, function (e) { status.textContent = ''; failToast('导出失败', e); }).then(function () { busy(go, false); });
        return;
      }
      downloadMontage(api, run).then(function (r) {
        status.textContent = '已导出 ' + r.count + ' 幅 · ' + r.width + ' × ' + r.height + ' px' + skippedText(r);
        if (root.URL && root.URL.createObjectURL) ui().fill(thumb, h('img', {src: root.URL.createObjectURL(r.blob), alt: '拼图', 'class': 'exp-thumb-img'}));
      }, function (e) { status.textContent = ''; failToast('拼图失败', e); }).then(function () { busy(go, false); });
    }, {icon: files ? 'download' : 'fig-montage', kind: 'primary'});
    var outSeg = segControl([{value: 'montage', label: '一张拼图'}, {value: 'files', label: '分成单张'}], out, function (v) {
      setViewsOutput(v); paneMontage(el, ctx);
    }, '输出');
    ui().fill(el, h('div', {'class': 'exp-montage'},
      optRow('视角', h('div', {'class': 'exp-pills'}, pills, six), '选几个视角拼成一张图；标准视角按解剖坐标架对准并撑满画幅，「当前」即视口里的视角。'),
      optRow('输出', outSeg, '分成单张：每个视角一张 PNG（色条、标题带视角名，同「图片」页样式），打成一个 zip；截面图单独一张。'),
      files ? null : optRow('列数', segControl([{value: 2, label: '2'}, {value: 3, label: '3'}, {value: 4, label: '4'}], o.columns, function (v) { setOptions({columns: +v}); }, '列数')),
      h('div', {'class': 'exp-sub'}, h('span', {'class': 'exp-k', text: '样式'}), h('span', {'class': 'muted', text: styleText()}),
        ui().infoTip(files ? '倍率、背景、文字、色条、标题、标签和裁边跟「图片」页一致。' : '倍率、背景、文字和标签跟「图片」页一致；全图共用一条色条，面板按 a、b、c 标号。')),
      h('div', {'class': 'exp-actions'}, go), status, thumb));
    return {dispose: function () {}};
  }
  function styleText() {
    var o = options();
    return [o.scale + '×', {viewport: '视口背景', white: '白底', transparent: '透明底'}[o.background], o.lang === 'en' ? 'English' : '中文', o.colorbar ? '色条' : '', o.title ? '标题' : '', o.labels ? '标签' : '', o.crop ? '裁边' : ''].filter(Boolean).join(' · ');
  }
  function paneOnepage(el, ctx) {
    var h = ui().h, api = API, cur = api.cur();
    var plan = onepagePlan(api).map(function (p) { return VIEW_ZH[p[1]]; });
    var status = h('p', {'class': 'exp-status muted'});
    var strip = h('div', {'class': 'exp-strip'});
    var fresh = false;
    function showStrip(items, fromData) {
      ui().fill(strip, items.map(function (it) {
        var src = fromData ? 'data:image/png;base64,' + it.png_base64 : api.api().urls.file(cur.jobId, it.file) + '?t=' + encodeURIComponent(it.created_at || '');
        return h('figure', {'class': 'exp-shot'}, h('img', {src: src, alt: it.caption || it.name}), h('figcaption', {text: it.caption || it.name}));
      }));
    }
    // What the one-pager shows now (snapshots.json is one of the job's servable files).
    if (cur && cur.job && cur.job.snapshots_count) {
      api.api().request(api.api().urls.file(cur.jobId, 'snapshots.json')).then(function (doc) {
        var items = doc && Array.isArray(doc.items) ? doc.items : [];
        if (fresh || !items.length) return;
        status.textContent = '一页纸里现有 ' + items.length + ' 张配图（' + ui().time(doc.updated_at) + '）。';
        showStrip(items, false);
      }, function () {});
    }
    var open = ui().button('打开一页纸', function () { root.open(api.api().urls.onepage(cur.jobId), '_blank', 'noopener'); }, {icon: 'external'});
    var go = ui().button('生成配图', function () {
      busy(go, true);
      uploadOnepage(api, {labels: options().labels, onStep: function (i, n, label) { status.textContent = '正在渲染 ' + label + '（' + (i + 1) + ' / ' + n + '）…'; },
        onUpload: function (n) { status.textContent = '正在上传 ' + n + ' 张配图…'; }}).then(function (r) {
        fresh = true;
        if (cur.job) cur.job.snapshots_count = r.count;
        status.textContent = '已生成 ' + r.count + ' 张配图' + (r.skipped.length ? '（' + r.skipped.map(function (v) { return VIEW_ZH[v]; }).join('、') + ' 不可用）' : '') + '。';
        showStrip(r.images, true);
        ui().toast('一页纸配图已更新。', {kind: 'ok', ms: 3000});
      }, function (e) { status.textContent = ''; failToast(e && e.status === 409 ? '配图没有写入（任务正在处理）' : '配图没有生成', e); }).then(function () { busy(go, false); });
    }, {icon: 'snapshot', kind: 'primary'});
    ui().fill(el, h('div', {'class': 'exp-onepage'},
      optRow('配图', h('div', {'class': 'exp-pills exp-plan'}, plan.map(function (t) { return h('span', {'class': 'exp-tag', text: t}); })),
        '一页纸配图：前视、左视、头侧视、当前视角（截面打开时加截面图），白底 2×、带色条；替换已有配图。复核锁定不影响配图。'),
      h('div', {'class': 'exp-actions'}, go, open), status, strip));
    return {dispose: function () {}};
  }
  var livePane = null;
  function pane(id, el, ctx) {
    if (livePane && livePane.dispose) { try { livePane.dispose(); } catch (_) {} }
    livePane = null;
    var api = API;
    if (!api || !api.cur() || !api.cur().manifest) { ui().fill(el, ui().empty('先打开一份结果。')); return null; }
    if ((id === 'figure' || id === 'montage' || id === 'onepage') && (!api.viewer() || typeof api.viewer().renderPose !== 'function')) {
      ui().fill(el, ui().empty('这台设备的浏览器不能显示三维（WebGL 不可用），不能出图。数据和离线报告仍可导出。'));
      return null;
    }
    if (id === 'figure') livePane = paneFigure(el, ctx);
    else if (id === 'montage') livePane = paneMontage(el, ctx);
    else if (id === 'onepage') livePane = paneOnepage(el, ctx);
    return livePane;
  }
  function disposePane() { if (livePane && livePane.dispose) { try { livePane.dispose(); } catch (_) {} } livePane = null; }
  function available() { return Boolean(API && API.cur && API.cur() && API.cur().manifest); }

  // ------------------------------------------------------------------ P3 lane 5: a result that is not on screen
  // What figure(), montage(), captions() and fileName() read from the shell, for ws_batch's offscreen viewer.
  // ctx = {jobId, manifest, result, viewer, field, window, info() (colour-bar info), hideName, windowLabel(field, spec)}
  function headless(ctx) {
    var cur = {jobId: ctx.jobId, manifest: ctx.manifest, result: ctx.result || null, field: ctx.field, window: ctx.window || 'adaptive',
      runIdentity: ctx.result && ctx.result.runIdentity, compare: null, split: null, slice: null, cursor: null};
    var shellState = null;
    try { shellState = ns.shell && typeof ns.shell.state === 'function' ? ns.shell.state() : null; } catch (_) { shellState = null; }
    return {
      cur: function () { return cur; }, viewer: function () { return ctx.viewer; }, offline: function () { return false; },
      state: function () { return {colorbarA: {info: function () { return typeof ctx.info === 'function' ? ctx.info() : null; }}, els: shellState && shellState.els, layers: {}}; },
      fieldById: function (m, id) { return shownFields(m).filter(function (f) { return f.id === id; })[0] || null; },
      hideName: function () { return Boolean(ctx.hideName); }, windowLabel: typeof ctx.windowLabel === 'function' ? ctx.windowLabel : null,
      ui: function () { return ns.ui; }, h: ns.ui && ns.ui.h
    };
  }

  // ------------------------------------------------------------------ shell hooks
  function openTab(tab) {
    if (!API) return;
    if (ns.exporter) ns.exporter.pendingTab = tab;
    API.openExport();
  }
  function quick(what) {
    var api = API;
    if (!api || !api.cur() || !api.cur().manifest) return;
    var o = options(); o.hideName = Boolean(api.hideName());
    if (what === 'six') {
      if (!hasFrame(api.viewer())) { ui().toast('没有解剖坐标架，不能取标准视角。', {kind: 'info'}); return; }
      ui().toast('正在渲染六视角…', {kind: 'info', ms: 0});
      if (viewsOutput() === 'files') {
        downloadViews(api, Object.assign(o, {views: VIEWS.slice()})).then(function (r) {
          ui().toast('已导出六视角 ' + r.count + ' 张 PNG（zip）。', {kind: 'ok', ms: 3500});
        }, function (e) { failToast('导出失败', e); });
        return;
      }
      downloadMontage(api, Object.assign(o, {views: VIEWS.slice(), columns: 3})).then(function (r) {
        ui().toast('已导出六视角拼图 ' + r.width + ' × ' + r.height + ' px。', {kind: 'ok', ms: 3500});
      }, function (e) { failToast('拼图失败', e); });
    } else if (what === 'colorbar') {
      if (!downloadColorbar(api, o)) ui().toast('没有可导出的色条。', {kind: 'info'});
    } else if (what === 'link') copyLink(api);
    else if (what === 'print') printView(api, o).catch(function (e) { failToast('打印失败', e); });
  }
  var ext = {
    id: 'figure',
    tools: function (api) {
      API = api;
      var cur = api.cur();
      if (!cur || !cur.manifest) return [];
      var U = api.ui(), h = api.h, v = api.viewer(), gl = Boolean(v && typeof v.renderPose === 'function');
      var b = function (label, icon, fn, title, off) { return U.button(label, fn, {icon: icon, cls: 'btn-sm', title: title, disabled: Boolean(off)}); };
      return [U.section('出图', {tag: U.infoTip('出版级 PNG、拼图、色标 SVG、一页纸配图和打印都按当前字段、色标窗与视角；复现链接把这些连同游标和截面写进一个地址。')},
        h('div', {'class': 'sec-actions fig-tools'},
          b('导图…', 'snapshot', function () { openTab('figure'); }, '出版级导图与自选拼图：倍率、背景、语言、色条、标题、标签', !gl),
          b('六视角', 'fig-montage', function () { quick('six'); }, viewsOutput() === 'files' ? '前后左右头足六张 PNG（zip），按上次的导图设置；在「拼图」页可改成一张拼图' : '前后左右头足六幅 + 共用色条，按上次的导图设置；在「拼图」页可改成分成单张', !gl || !hasFrame(v)),
          b('色标 SVG', 'fig-colorbar', function () { quick('colorbar'); }, '当前色条存成 SVG'),
          b('复现链接', 'fig-link', function () { quick('link'); }, '复制一个链接：打开后回到同样的字段、色标（色表、分段、阈值、单位）、标签、视角、游标和截面'),
          api.offline() ? null : b('一页纸配图…', 'snapshot', function () { openTab('onepage'); }, '生成一页纸里的配图', !gl),
          b('打印', 'print', function () { quick('print'); }, '当前视图单独排成一页打印', !gl)))];
    },
    onResult: function (api) {
      API = api;
      bindHash();
      var hash = root.location.hash, r = api.parseHash(hash);
      // after the shell's own route extras (field, question, comparison, bookmark), which run right after this hook
      setTimeout(function () {
        if (!API || API.cur() == null) return;
        if (linkParam(hash)) applyFromHash(API, hash);
        else if (r.bm && ns.bookmarks) {
          var bm = ns.bookmarks.list(API.cur().runIdentity).filter(function (b) { return b.id === r.bm; })[0];
          if (bm) afterBookmark(bm);
        }
      }, 0);
    },
    onClose: function () { disposePane(); }
  };
  (ns.ext = ns.ext || []).push(ext);

  var mod = {
    VIEWS: VIEWS, CLASSIC: CLASSIC, VIEW_ZH: VIEW_ZH, VIEW_EN: VIEW_EN, DEFAULTS: DEFAULTS, ONEPAGE_PLAN: ONEPAGE_PLAN,
    options: options, setOptions: setOptions, sanitize: sanitize, tr: tr, themeOf: themeOf, colorbarSVG: colorbarSVG, captions: captions, fileName: fileName,
    figureLayout: figureLayout, composeFigure: composeFigure, figure: figure, montage: montage, drawChips: drawChips, layoutChips: layoutChips, withChips: withChips,
    downloadFigure: downloadFigure, downloadMontage: downloadMontage, downloadColorbar: downloadColorbar,
    encodeState: encodeState, decodeState: decodeState, viewState: viewState, linkURL: linkURL, linkParam: linkParam, withoutView: withoutView, applyView: applyView, copyLink: copyLink,
    bookmarkExtra: bookmarkExtra, afterBookmark: afterBookmark,
    onepagePlan: onepagePlan, onepageImages: onepageImages, uploadOnepage: uploadOnepage, checkSizes: checkSizes, snapshotCaption: snapshotCaption,
    printView: printView, pane: pane, disposePane: disposePane, available: available, openTab: openTab, ext: ext,
    // P3 lane 5
    zipStore: zipStore, crc32: crc32, blobBytes: blobBytes, toBlob: toBlob, viewFiles: viewFiles, downloadViews: downloadViews, viewsOutput: viewsOutput, setViewsOutput: setViewsOutput,
    displayState: displayState, applyDisplay: applyDisplay, effectiveScale: effectiveScale, headless: headless, styleRows: commonRows, styleText: styleText, hasFrame: hasFrame,
    _setApi: function (a) { API = a; }, _api: function () { return API; }, _print: null,
    // P4 lane B (W58 / W59)
    printKey: printKey, isPrintKey: isPrintKey, snapViewports: snapViewports, clearSnaps: clearSnaps, printSnaps: function () { return printSnaps.slice(); }
  };
  return mod;
});
