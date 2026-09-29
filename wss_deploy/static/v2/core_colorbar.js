/* WSSV2 core · colour bar (contract §6.2, §7; discussion V7).
   Vertical bar with ticks, bold observation-threshold lines, the embedded sample-point histogram (labelled as
   such), the window name, end-colour arrows when values fall outside the range, and the missing fraction.
   create(el) renders HTML + inline SVG into el; toSVG(info) returns a standalone SVG for exports. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.colorbar = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var FONT = 'system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
  var INK = 'var(--ink, #1b2430)', INK2 = 'var(--ink-2, #465361)', INK3 = 'var(--ink-3, #77828e)', LINE2 = 'var(--line-2, #c6ccd2)';
  var INK_HEX = '#1b2430', INK2_HEX = '#465361', INK3_HEX = '#77828e', LINE2_HEX = '#c6ccd2';

  function U() { return ns.util; }
  function esc(x) { return U().escapeHtml(x); }
  function hex(c) { return ns.colormap.rgbToHex(c); }

  function windowText(info) {
    var w = info.windowLabel;
    if (w && typeof w === 'object') return (w.label || '') + (w.provisional ? '（暂定）' : '');
    return w ? String(w) : '';
  }
  function titleParts(info) {
    var u = U().unitText(info.units);
    return { title: String(info.label || ''), units: u };
  }
  // Lines shown under the bar (plain text + optional swatch colour).  Every number keeps its unit / percent.
  function noteLines(info) {
    var sc = info.scale || null, h = info.histogram || null, out = [];
    if (h && h.n + h.nMissing > 0) {
      var what = info.histogramSource === 'vertices' ? '显示网格顶点分布' : '采样点分布';
      out.push({ kind: 'hist', text: '灰柱：' + what, sub: h.n + ' 点' + (info.histogramScope === 'visible' ? '（仅显示的分支）' : '') });
    }
    if (h && h.n > 0 && (h.nAbove > 0 || h.nBelow > 0)) {
      var parts = [];
      if (h.nAbove > 0) parts.push('高于上限 ' + U().fmtPct(h.nAbove / h.n));
      if (h.nBelow > 0) parts.push('低于下限 ' + U().fmtPct(h.nBelow / h.n));
      out.push({ kind: 'clip', text: '超出范围按端色', sub: parts.join('，') });
    } else if (sc) {
      out.push({ kind: 'clip', text: '超出范围按端色' });
    }
    var mf = Number(info.missingFraction);
    if (Number.isFinite(mf) && mf > 0) out.push({ kind: 'missing', swatch: sc ? hex(sc.missing) : '#a7adb3', text: '缺失（灰）' + U().fmtPct(mf) });
    var th = thresholdMarks(info);
    if (th.length) out.push({ kind: 'thr', text: '粗线：观察阈值', sub: '非临床界值' });
    if (sc && sc.logRejected) out.push({ kind: 'log', text: sc.logRejected });
    return out;
  }
  function scaleTag(sc) {
    if (!sc) return '';
    var t = [];
    if (sc.log) t.push('对数');
    if (sc.bands) t.push(sc.bands + ' 段');
    return t.join(' · ');
  }
  function thresholdMarks(info) {
    var sc = info.scale; if (!sc || !Array.isArray(info.thresholds)) return [];
    var out = [];
    info.thresholds.forEach(function (t) {
      var v = Number(t && typeof t === 'object' ? t.v : t); if (!Number.isFinite(v)) return;
      var f = sc.norm(v); if (!(f >= -1e-9 && f <= 1 + 1e-9)) return;
      out.push({ v: v, f: Math.max(0, Math.min(1, f)) });
    });
    return out;
  }

  // The graphic part: histogram | bar | ticks.  W × H px; returns {svg, height}.
  function graphicSVG(info, W, H, opts) {
    opts = opts || {};
    var sc = info.scale, h = info.histogram, css = opts.css !== false;
    var ink = css ? INK : INK_HEX, ink2 = css ? INK2 : INK2_HEX, ink3 = css ? INK3 : INK3_HEX, line2 = css ? LINE2 : LINE2_HEX;
    var histW = W >= 120 ? 34 : 22, gap = 3, barW = 12, histX0 = 1, barX = histX0 + histW + gap, labelX = barX + barW + 7;
    var arrow = 7, top = arrow + 5, bottom = H - arrow - 5, barH = Math.max(40, bottom - top);
    var yOf = function (f) { return top + barH * (1 - f); };
    var s = [];
    s.push('<svg xmlns="http://www.w3.org/2000/svg" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" font-family=\'' + FONT + '\' font-size="12" style="font-variant-numeric:tabular-nums;display:block">');
    if (!sc) return s.concat(['</svg>']).join('');
    // histogram (sample points), bars grow leftwards from the colour bar
    if (h && h.counts && h.counts.length) {
      var mx = 0; h.counts.forEach(function (c) { if (c > mx) mx = c; });
      if (mx > 0) {
        var d = [];
        for (var k = 0; k < h.counts.length; k++) {
          if (!h.counts[k]) continue;
          var f0 = sc.norm(h.edges[k]), f1 = sc.norm(h.edges[k + 1]);
          if (!(f1 > 0) || !(f0 < 1)) continue;
          f0 = Math.max(0, f0); f1 = Math.min(1, f1);
          var y0 = yOf(f1), y1 = yOf(f0), len = Math.max(0.8, histW * h.counts[k] / mx);
          d.push('M' + (barX - 1).toFixed(1) + ' ' + y0.toFixed(2) + 'h' + (-len).toFixed(2) + 'V' + y1.toFixed(2) + 'h' + len.toFixed(2) + 'Z');
        }
        s.push('<path d="' + d.join('') + '" fill="' + ink3 + '" fill-opacity="0.5"><title>' + esc(info.histogramSource === 'vertices' ? '显示网格顶点分布' : '采样点分布') + '</title></path>');
      }
    }
    // colour bar: exact band blocks, or 64 thin slices (no gradient interpolation: the slices are the scale's own colours)
    var nSlices = sc.bands || 96;
    for (var i = 0; i < nSlices; i++) {
      var fa = i / nSlices, fb = (i + 1) / nSlices, c = hex(sc.colorT((fa + fb) / 2));
      s.push('<rect x="' + barX + '" y="' + yOf(fb).toFixed(2) + '" width="' + barW + '" height="' + (barH / nSlices + 0.35).toFixed(2) + '" fill="' + c + '" shape-rendering="crispEdges"/>');
    }
    s.push('<rect x="' + (barX - 0.5) + '" y="' + (top - 0.5) + '" width="' + (barW + 1) + '" height="' + (barH + 1) + '" fill="none" stroke="' + line2 + '"/>');
    // end-colour arrows when values fall outside the range
    var above = h && h.nAbove > 0, below = h && h.nBelow > 0;
    var cx = barX + barW / 2;
    if (above) s.push('<path d="M' + barX + ' ' + (top - 1) + 'L' + cx + ' ' + (top - 1 - arrow) + 'L' + (barX + barW) + ' ' + (top - 1) + 'Z" fill="' + hex(sc.colorT(1)) + '" stroke="' + line2 + '" stroke-width="0.8"><title>高于上限的值按上端色</title></path>');
    if (below) s.push('<path d="M' + barX + ' ' + (bottom + 1) + 'L' + cx + ' ' + (bottom + 1 + arrow) + 'L' + (barX + barW) + ' ' + (bottom + 1) + 'Z" fill="' + hex(sc.colorT(0)) + '" stroke="' + line2 + '" stroke-width="0.8"><title>低于下限的值按下端色</title></path>');
    // thresholds first (their labels win), then ticks that do not collide with them
    var marks = thresholdMarks(info), taken = [];
    marks.forEach(function (m) {
      var y = yOf(m.f);
      s.push('<line x1="' + (histX0) + '" y1="' + y.toFixed(2) + '" x2="' + (barX + barW + 4) + '" y2="' + y.toFixed(2) + '" stroke="' + ink + '" stroke-width="2"/>');
      if (taken.some(function (t) { return Math.abs(t - y) < 13; })) return;
      taken.push(y);
      s.push('<text x="' + labelX + '" y="' + (y + 4).toFixed(2) + '" fill="' + ink + '" font-weight="600">' + esc(U().fmtTrim(m.v)) + '</text>');
    });
    var ticks = sc.ticks(Math.max(3, Math.round(barH / 48)));
    ticks.forEach(function (t) {
      var y = yOf(t.f);
      if (!t.major) { s.push('<line x1="' + (barX + barW) + '" y1="' + y.toFixed(2) + '" x2="' + (barX + barW + 2.5) + '" y2="' + y.toFixed(2) + '" stroke="' + ink3 + '"/>'); return; }
      s.push('<line x1="' + (barX + barW) + '" y1="' + y.toFixed(2) + '" x2="' + (barX + barW + 4) + '" y2="' + y.toFixed(2) + '" stroke="' + ink2 + '"/>');
      if (taken.some(function (u) { return Math.abs(u - y) < 13; })) return;
      taken.push(y);
      s.push('<text x="' + labelX + '" y="' + (y + 4).toFixed(2) + '" fill="' + ink2 + '">' + esc(t.label) + '</text>');
    });
    s.push('</svg>');
    return s.join('');
  }

  function ariaText(info) {
    var sc = info.scale, tp = titleParts(info);
    if (!sc) return tp.title;
    var d = sc.describe();
    return '色标 ' + tp.title + '：' + U().fmtRange(d.shown[0], d.shown[1], info.units) + (d.log ? '，对数' : '') + '。' + windowText(info);
  }

  function create(el) {
    if (!el) throw new Error('colorbar needs an element');
    var doc = el.ownerDocument || root.document, last = null, disposed = false;
    var box = doc.createElement('div');
    box.className = 'wssv2-colorbar';
    box.setAttribute('role', 'img');
    box.style.cssText = 'font:13px/1.35 ' + FONT + ';color:' + INK + ';font-variant-numeric:tabular-nums;user-select:none;width:100%;';
    el.appendChild(box);
    function render(info) {
      if (disposed) return;
      last = info || null;
      if (!info || !info.scale) { box.innerHTML = ''; box.setAttribute('aria-label', '无色标'); return; }
      var W = Math.max(96, Math.round(el.clientWidth || 148));
      var Htot = Math.round(el.clientHeight || 340);
      var notes = noteLines(info), tp = titleParts(info), wt = windowText(info), tag = scaleTag(info.scale);
      // header ~ 38 px, notes ~ 17 px per line (they may wrap: budget 1.5 lines each)
      var lines = notes.reduce(function (a, n) { return a + (n.sub ? 2 : 1); }, 0);
      var gH = Math.max(120, Math.min(420, Htot - 44 - lines * 17 - notes.length * 3));
      var html = [];
      html.push('<div class="wssv2-cb-title" style="font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + esc(tp.title) +
        (tp.units ? ' <span style="font-weight:400;color:' + INK3 + ';font-size:12px">' + esc(tp.units) + '</span>' : '') + '</div>');
      html.push('<div class="wssv2-cb-window" style="font-size:12px;color:' + INK2 + ';margin-bottom:4px">' + esc(wt) + (tag ? '<span style="color:' + INK3 + '"> · ' + esc(tag) + '</span>' : '') + '</div>');
      html.push('<div class="wssv2-cb-graphic">' + graphicSVG(info, W, gH) + '</div>');
      html.push('<div class="wssv2-cb-notes" style="font-size:12px;color:' + INK3 + ';margin-top:2px;line-height:1.35">');
      notes.forEach(function (n) {
        html.push('<div class="wssv2-cb-note" data-kind="' + n.kind + '" style="margin-top:3px">' + (n.swatch ? '<span style="display:inline-block;width:10px;height:10px;background:' + n.swatch + ';border:1px solid ' + LINE2 + ';vertical-align:-1px;margin-right:4px"></span>' : '') + esc(n.text) +
          (n.sub ? '<br><span style="color:' + INK2 + '">' + esc(n.sub) + '</span>' : '') + '</div>');
      });
      html.push('</div>');
      box.innerHTML = html.join('');
      box.setAttribute('aria-label', ariaText(info));
    }
    return {
      element: box,
      update: render,
      refresh: function () { if (last) render(last); },
      info: function () { return last; },
      toSVG: function (o) { return last ? toSVG(last, o) : ''; },
      dispose: function () { disposed = true; if (box.parentNode) box.parentNode.removeChild(box); last = null; }
    };
  }

  // Standalone SVG (title, window, graphic, notes as single lines) for figure exports; hex colours, no CSS vars.
  function toSVG(info, o) {
    o = o || {};
    var W = o.width || 190, gH = o.graphicHeight || 300, lh = 16;
    var notes = noteLines(info), tp = titleParts(info), wt = windowText(info), tag = scaleTag(info.scale);
    var H = 40 + gH + 6 + notes.length * lh + 4;
    var g = graphicSVG(info, W, gH, { css: false }).replace(/^<svg[^>]*>/, '').replace(/<\/svg>$/, '');
    var s = ['<svg xmlns="http://www.w3.org/2000/svg" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" font-family=\'' + FONT + '\' font-size="12" style="font-variant-numeric:tabular-nums">'];
    if (o.background) s.push('<rect width="' + W + '" height="' + H + '" fill="' + o.background + '"/>');
    s.push('<text x="1" y="14" font-size="13" font-weight="600" fill="' + INK_HEX + '">' + esc(tp.title) + (tp.units ? ' <tspan font-weight="400" fill="' + INK3_HEX + '" font-size="12">' + esc(tp.units) + '</tspan>' : '') + '</text>');
    s.push('<text x="1" y="31" fill="' + INK2_HEX + '">' + esc(wt + (tag ? ' · ' + tag : '')) + '</text>');
    s.push('<g transform="translate(0 38)">' + g + '</g>');
    notes.forEach(function (n, i) { s.push('<text x="1" y="' + (38 + gH + 6 + (i + 1) * lh - 4) + '" fill="' + INK3_HEX + '">' + esc(n.text + (n.sub ? '：' + n.sub : '')) + '</text>'); });
    s.push('</svg>');
    return s.join('');
  }

  return { create: create, toSVG: toSVG, noteLines: noteLines, graphicSVG: graphicSVG };
});
