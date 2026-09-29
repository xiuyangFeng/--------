/* WSS workspace v2 — icon set (contract §6.3 ws_icons.js, §7 "不要用 Unicode 字符当图标").
 * 16 px grid, 1.5 px stroke, currentColor.  Each icon is a list of SVG primitives built with createElementNS,
 * so no markup string is parsed and the page keeps a strict CSP. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.icons = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var SVG = 'http://www.w3.org/2000/svg';
  var P = function (d) { return ['path', {d: d}]; };
  var C = function (cx, cy, r, fill) { return ['circle', fill ? {cx: cx, cy: cy, r: r, fill: 'currentColor', stroke: 'none'} : {cx: cx, cy: cy, r: r}]; };
  var R = function (x, y, w, h, rx) { return ['rect', {x: x, y: y, width: w, height: h, rx: rx || 0}]; };

  var ICONS = {
    upload: [P('M8 10.5V2.75'), P('M4.75 6 8 2.75 11.25 6'), P('M2.5 10.5v2a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-2')],
    search: [C(7, 7, 4.25), P('M10.25 10.25 13.75 13.75')],
    'chevron-left': [P('M10 3.5 5.5 8 10 12.5')],
    'chevron-right': [P('M6 3.5 10.5 8 6 12.5')],
    'chevron-down': [P('M3.5 6 8 10.5 12.5 6')],
    close: [P('M4 4l8 8'), P('M12 4l-8 8')],
    layers: [P('M8 2.25 1.75 5.5 8 8.75l6.25-3.25L8 2.25Z'), P('M1.75 8.25 8 11.5l6.25-3.25'), P('M1.75 11 8 14.25 14.25 11')],
    snapshot: [R(1.75, 4.5, 12.5, 9, 1.25), P('M5.5 4.5 6.5 2.5h3l1 2'), C(8, 9, 2.4)],
    bookmark: [P('M4.25 2.25h7.5v11.5L8 10.75l-3.75 3V2.25Z')],
    'bookmark-fill': [['path', {d: 'M4.25 2.25h7.5v11.5L8 10.75l-3.75 3V2.25Z', fill: 'currentColor'}]],
    compare: [R(1.75, 2.75, 5.25, 10.5, 0.75), R(9, 2.75, 5.25, 10.5, 0.75)],
    probe: [C(8, 8, 4.5), P('M8 1v2.5'), P('M8 12.5V15'), P('M1 8h2.5'), P('M12.5 8H15'), C(8, 8, 1, true)],
    ruler: [P('M1.75 11.25 11.25 1.75l3 3-9.5 9.5-3-3Z'), P('M4.75 8.25l1.5 1.5'), P('M7 6l1.5 1.5'), P('M9.25 3.75l1.5 1.5')],
    eye: [P('M1.5 8S4 3.75 8 3.75 14.5 8 14.5 8 12 12.25 8 12.25 1.5 8 1.5 8Z'), C(8, 8, 2)],
    'eye-off': [P('M1.5 8S4 3.75 8 3.75 14.5 8 14.5 8 12 12.25 8 12.25 1.5 8 1.5 8Z'), C(8, 8, 2), P('M2.5 2.5l11 11')],
    light: [C(8, 8, 2.75), P('M8 1.5v1.5'), P('M8 13v1.5'), P('M1.5 8H3'), P('M13 8h1.5'), P('M3.4 3.4l1.05 1.05'), P('M11.55 11.55l1.05 1.05'), P('M3.4 12.6l1.05-1.05'), P('M11.55 4.45l1.05-1.05')],
    download: [P('M8 2.5v8'), P('M4.75 7.25 8 10.5l3.25-3.25'), P('M2.5 13.5h11')],
    print: [P('M4 5.75V2.25h8v3.5'), R(1.75, 5.75, 12.5, 5.75, 1), R(4, 9.5, 8, 4.25)],
    external: [P('M9.25 2.25h4.5v4.5'), P('M13.75 2.25 7.5 8.5'), P('M11.75 9.5v3.25a1 1 0 0 1-1 1h-7.5a1 1 0 0 1-1-1v-7.5a1 1 0 0 1 1-1H6.5')],
    info: [C(8, 8, 6.25), P('M8 7.25v4.25'), C(8, 4.9, 0.8, true)],
    warning: [P('M8 2 14.5 13.75h-13L8 2Z'), P('M8 6.5v3.5'), C(8, 11.9, 0.8, true)],
    check: [P('M3 8.5 6.5 12 13 4.5')],
    refresh: [P('M13.5 8a5.5 5.5 0 1 1-1.61-3.89'), P('M13.5 2.25v3.5H10')],
    play: [P('M5 3v10l8-5-8-5Z')],
    pause: [P('M5.5 3v10'), P('M10.5 3v10')],
    user: [C(8, 5.25, 2.75), P('M2.5 14c.75-2.75 2.9-4.25 5.5-4.25s4.75 1.5 5.5 4.25')],
    settings: [P('M2 4.5h7'), P('M12.5 4.5H14'), C(10.75, 4.5, 1.6), P('M2 11.5h1.75'), P('M7.25 11.5H14'), C(5.5, 11.5, 1.6)],
    more: [C(3.5, 8, 1.1, true), C(8, 8, 1.1, true), C(12.5, 8, 1.1, true)]
  };

  function icon(name, opts) {
    opts = opts || {};
    var spec = ICONS[name];
    var doc = root.document;
    if (!spec || !doc || typeof doc.createElementNS !== 'function') return null;
    var size = String(opts.size || 16);
    var svg = doc.createElementNS(SVG, 'svg');
    var attrs = {width: size, height: size, viewBox: '0 0 16 16', fill: 'none', stroke: 'currentColor',
      'stroke-width': '1.5', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'class': 'ic ic-' + name, focusable: 'false'};
    Object.keys(attrs).forEach(function (k) { svg.setAttribute(k, attrs[k]); });
    if (opts.title) {
      svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', opts.title);
    } else svg.setAttribute('aria-hidden', 'true');
    spec.forEach(function (prim) {
      var el = doc.createElementNS(SVG, prim[0]);
      Object.keys(prim[1]).forEach(function (k) { el.setAttribute(k, String(prim[1][k])); });
      svg.appendChild(el);
    });
    return svg;
  }

  return {icon: icon, names: Object.keys(ICONS), has: function (name) { return Object.prototype.hasOwnProperty.call(ICONS, name); }};
});
