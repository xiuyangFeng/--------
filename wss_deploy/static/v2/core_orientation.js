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

  var INK = 'var(--hud-ink, #1b2430)', INK3 = 'var(--hud-ink-3, #77828e)';
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
    // screen-space basis of the three anatomical axes, for the orientation cube
    var proj = function (v) { return [dot(v, b.right), dot(v, b.up), dot(v, b.forward)]; };
    var cube = { left: proj(ax.left), superior: proj(ax.superior), anterior: proj(anterior), lrInferred: !lrConfirmed, apInferred: !apConfirmed };
    return { available: true, axes: axes, caption: caption, view: bv > 0.8 ? best : null, cube: cube };
  }

  function esc(x) { return ns.util.escapeHtml(x); }

  // Orientation cube: the six faces of a unit cube aligned with the anatomical axes, drawn back to front with the
  // faces turned towards the viewer labelled.  A face whose direction is only inferred gets a dashed edge and a
  // lighter label.  Each face carries data-view so a click can turn the camera to that standard view.
  var FACES = [
    { key: 'left', axis: 'left', sign: 1, label: '左', view: 'left', inferred: 'lr' },
    { key: 'right', axis: 'left', sign: -1, label: '右', view: 'right', inferred: 'lr' },
    { key: 'superior', axis: 'superior', sign: 1, label: '头', view: 'superior', inferred: null },
    { key: 'inferior', axis: 'superior', sign: -1, label: '足', view: 'inferior', inferred: null },
    { key: 'anterior', axis: 'anterior', sign: 1, label: '前', view: 'anterior', inferred: 'ap' },
    { key: 'posterior', axis: 'anterior', sign: -1, label: '后', view: 'posterior', inferred: 'ap' }
  ];
  function svg(lay, size) {
    var S = size || 76, c = S / 2, R = S * 0.25;
    var out = ['<svg xmlns="http://www.w3.org/2000/svg" width="' + S + '" height="' + S + '" viewBox="0 0 ' + S + ' ' + S + '" font-family=\'' + FONT + '\' style="display:block;overflow:visible">'];
    var cb = lay && lay.cube;
    if (!cb) { out.push('</svg>'); return out.join(''); }
    // A small constant tilt (yaw 22°, pitch 16°) so three faces always show and the cube reads as a solid; the
    // labelled face turned most towards the viewer is still the view the camera is taken from.
    var tilt = function (v) {
      var cy = Math.cos(0.38), sy = Math.sin(0.38), cp = Math.cos(0.28), sp = Math.sin(0.28);
      var x = v[0] * cy + v[2] * sy, z = -v[0] * sy + v[2] * cy, y = v[1];
      return [x, y * cp - z * sp, y * sp + z * cp];
    };
    var A = { left: tilt(cb.left), superior: tilt(cb.superior), anterior: tilt(cb.anterior) };
    var others = { left: ['superior', 'anterior'], superior: ['left', 'anterior'], anterior: ['left', 'superior'] };
    var faces = FACES.map(function (f) {
      var n = A[f.axis].map(function (x) { return x * f.sign; });
      var o = others[f.axis], u = A[o[0]], v = A[o[1]];
      var corners = [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(function (k) {
        return [n[0] + k[0] * u[0] + k[1] * v[0], n[1] + k[0] * u[1] + k[1] * v[1], n[2] + k[0] * u[2] + k[1] * v[2]];
      });
      var inferred = f.inferred === 'lr' ? cb.lrInferred : f.inferred === 'ap' ? cb.apInferred : false;
      return { f: f, n: n, corners: corners, facing: -n[2], inferred: inferred };
    }).filter(function (x) { return x.facing > 0.02; });
    faces.sort(function (a, b) { return a.facing - b.facing; });
    faces.forEach(function (x) {
      var pts = x.corners.map(function (p) { return (c + p[0] * R).toFixed(1) + ',' + (c - p[1] * R).toFixed(1); }).join(' ');
      var shade = (0.38 + 0.56 * x.facing).toFixed(2);
      out.push('<g class="oc-face" data-view="' + x.f.view + '" style="cursor:pointer"><title>' + esc('转到' + (FROM[x.f.view] || '').replace('看', '') + '视角' + (x.inferred ? '（方向为推断）' : '')) + '</title>' +
        '<polygon points="' + pts + '" fill="var(--hud-cube, #ffffff)" fill-opacity="' + shade + '" stroke="#5b6674" stroke-opacity="0.55" stroke-width="1"' +
        (x.inferred ? ' stroke-dasharray="3 2"' : '') + ' stroke-linejoin="round"/>');
      if (x.facing > 0.35) {
        var tx = c + x.n[0] * R, ty = c - x.n[1] * R;
        var fs = (9 + 5 * x.facing).toFixed(1);
        out.push('<text x="' + tx.toFixed(1) + '" y="' + (ty + fs * 0.36).toFixed(1) + '" text-anchor="middle" font-size="' + fs + '" font-weight="' + (x.inferred ? 400 : 600) + '" fill="' +
          (x.inferred ? '#6b7785' : '#1b2430') + '" style="pointer-events:none">' + esc(x.f.label) + '</text>');
      }
      out.push('</g>');
    });
    out.push('</svg>');
    return out.join('');
  }

  function create(el, viewer) {
    if (!el || !viewer) throw new Error('orientation needs an element and a viewer');
    var doc = el.ownerDocument || root.document, disposed = false;
    var box = doc.createElement('div');
    box.className = 'wssv2-orientation';
    box.style.cssText = 'font:12px/1.35 ' + FONT + ';color:' + INK + ';user-select:none;pointer-events:auto;';
    box.addEventListener('click', function (e) {
      var g = e.target && e.target.closest ? e.target.closest('[data-view]') : null;
      if (g && viewer.standardView) { try { viewer.standardView(g.getAttribute('data-view'), { animate: true }); } catch (_) {} }
    });
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
      box.innerHTML = svg(lay, 76);
      box.title = lay.caption.join('；');
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
