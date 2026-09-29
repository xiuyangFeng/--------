/* WSSV2 core · orientation marker (contract §6.2; discussion U16).
   A small triad drawn from the anatomical frame and the current camera: 左/右, 头/足, 前/后 at the projected
   axis ends.  Left/right is marked 「推断」 until the outlet naming is confirmed; front/back is dashed grey
   「推断」 unless the manifest says the STL carried patient coordinates.  The caption states which side the
   view is taken from ("从前方看"). */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.orientation = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var INK = 'var(--ink, #1b2430)', INK3 = 'var(--ink-3, #77828e)';
  var FONT = 'system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
  var FROM = { anterior: '从前方看', posterior: '从后方看', left: '从左侧看', right: '从右侧看', superior: '从头侧看', inferior: '从足侧看' };

  // Pure layout: frame (manifest.frame) + camera state → axis end positions (unit screen coords, y up) and styles.
  function layout(frame, camera) {
    var V = ns.viewer;
    var ax = V.anatomicalAxes(frame);
    if (!ax || !camera) return { available: false, axes: [], caption: ['无解剖坐标架，方位未知'] };
    var o = frame.orientation || {};
    var lrConfirmed = o.left_right === 'confirmed', apConfirmed = o.anterior_posterior === 'confirmed';
    var b = V.cameraBasis(camera);
    var dot = function (a, c) { return a[0] * c[0] + a[1] * c[1] + a[2] * c[2]; };
    var anterior = ax.posterior.map(function (x) { return -x; });
    var defs = [
      { key: 'lr', vec: ax.left, pos: '左', neg: '右', inferred: !lrConfirmed, dashed: false },
      { key: 'si', vec: ax.superior, pos: '头', neg: '足', inferred: false, dashed: false },
      { key: 'ap', vec: anterior, pos: '前', neg: '后', inferred: !apConfirmed, dashed: !apConfirmed }
    ];
    var axes = defs.map(function (d) {
      var x = dot(d.vec, b.right), y = dot(d.vec, b.up), z = dot(d.vec, b.forward), len = Math.hypot(x, y);
      return { key: d.key, x: x, y: y, depth: z, length: len, visible: len > 0.28, pos: d.pos, neg: d.neg, inferred: d.inferred, dashed: d.dashed };
    });
    // view side: the anatomical direction the camera sits on (opposite to forward)
    var cand = [['anterior', anterior], ['posterior', ax.posterior], ['left', ax.left], ['right', ax.left.map(function (x) { return -x; })], ['superior', ax.superior], ['inferior', ax.superior.map(function (x) { return -x; })]];
    var best = null, bv = -1;
    cand.forEach(function (c) { var v = -dot(c[1], b.forward); if (v > bv) { bv = v; best = c[0]; } });
    var caption = [];
    if (bv > 0.8) caption.push(FROM[best] + (best === 'anterior' || best === 'posterior' ? '' : ''));
    else caption.push('斜视角');
    var notes = [];
    if (!lrConfirmed) notes.push('左右为推断');
    if (!apConfirmed) notes.push('前后为推断');
    if (notes.length) caption.push(notes.join('，'));
    return { available: true, axes: axes, caption: caption, view: bv > 0.8 ? best : null };
  }

  function esc(x) { return ns.util.escapeHtml(x); }

  function svg(lay, size) {
    var S = size || 84, c = S / 2, R = S * 0.34, lab = S * 0.45;
    var out = ['<svg xmlns="http://www.w3.org/2000/svg" width="' + S + '" height="' + S + '" viewBox="0 0 ' + S + ' ' + S + '" font-family=\'' + FONT + '\' font-size="12" style="display:block;overflow:visible">'];
    // back-to-front so the axis pointing at the viewer is drawn last
    lay.axes.slice().sort(function (a, b) { return b.depth - a.depth; }).forEach(function (a) {
      var ux = a.length > 1e-6 ? a.x / a.length : 0, uy = a.length > 1e-6 ? a.y / a.length : 0, L = R * Math.min(1, a.length);
      var x1 = c + ux * L, y1 = c - uy * L, x0 = c - ux * L, y0 = c + uy * L;
      var col = a.dashed ? INK3 : INK;
      out.push('<line x1="' + x0.toFixed(1) + '" y1="' + y0.toFixed(1) + '" x2="' + x1.toFixed(1) + '" y2="' + y1.toFixed(1) + '" stroke="' + col + '" stroke-width="1.2"' + (a.dashed ? ' stroke-dasharray="3 2.5"' : '') + ' opacity="' + (a.visible ? 1 : 0.35) + '"/>');
      if (!a.visible) return;
      var d = lab * Math.min(1, a.length) + 2;
      [[1, a.pos], [-1, a.neg]].forEach(function (e) {
        var tx = c + e[0] * ux * d, ty = c - e[0] * uy * d + 4;
        out.push('<text x="' + tx.toFixed(1) + '" y="' + ty.toFixed(1) + '" text-anchor="middle" fill="' + col + '" font-weight="' + (a.dashed ? 400 : 600) + '"' +
          ' paint-order="stroke" stroke="var(--viewport, #eceef0)" stroke-width="3" stroke-linejoin="round">' + esc(e[1]) + '</text>');
      });
    });
    out.push('<circle cx="' + c + '" cy="' + c + '" r="1.6" fill="' + INK + '"/>');
    out.push('</svg>');
    return out.join('');
  }

  function create(el, viewer) {
    if (!el || !viewer) throw new Error('orientation needs an element and a viewer');
    var doc = el.ownerDocument || root.document, disposed = false;
    var box = doc.createElement('div');
    box.className = 'wssv2-orientation';
    box.style.cssText = 'font:12px/1.35 ' + FONT + ';color:' + INK + ';user-select:none;pointer-events:none;';
    box.setAttribute('role', 'img');
    el.appendChild(box);
    var pending = false, lastKey = '';
    function frameOf() { return viewer.frame ? viewer.frame() : null; }
    function update() {
      if (disposed) return;
      pending = false;
      var lay = layout(frameOf(), viewer.getCamera ? viewer.getCamera() : null);
      var key = JSON.stringify(lay.axes.map(function (a) { return [a.x.toFixed(3), a.y.toFixed(3), a.visible]; })) + lay.caption.join('|');
      if (key === lastKey) return;
      lastKey = key;
      if (!lay.available) {
        box.innerHTML = '<div style="color:' + INK3 + '">' + esc(lay.caption[0]) + '</div>';
        box.setAttribute('aria-label', lay.caption[0]);
        return;
      }
      box.innerHTML = svg(lay, 84) + '<div class="wssv2-orientation-caption" style="color:' + INK3 + ';margin-top:2px;white-space:nowrap">' + lay.caption.map(esc).join('<br>') + '</div>';
      box.setAttribute('aria-label', '方位：' + lay.caption.join('；'));
    }
    function schedule() {
      if (pending || disposed) return;
      pending = true;
      (root.requestAnimationFrame || function (f) { return setTimeout(f, 16); })(update);
    }
    var offs = [viewer.on('camera', schedule), viewer.on('ready', schedule)];
    update();
    return {
      element: box,
      update: function () { lastKey = ''; update(); },
      dispose: function () { disposed = true; offs.forEach(function (f) { f(); }); if (box.parentNode) box.parentNode.removeChild(box); }
    };
  }

  return { create: create, layout: layout, svg: svg };
});
