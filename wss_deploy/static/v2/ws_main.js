/* WSS workspace v2 — start-up (contract §6.3 ws_main.js): offline package → offline shell; otherwise
 * session → login form when needed → shell.  The only module that starts work on load. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.main = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var started = false, expired = false;

  function readJson(id) {
    var el = root.document.getElementById(id);
    if (!el) return null;
    try { return JSON.parse(el.textContent || 'null'); } catch (_) { return null; }
  }
  function isOffline() {
    if (ns.env && ns.env.offline === true) return true;
    return Boolean(root.document.getElementById('wssv2-manifest'));
  }
  function fatal(text) {
    var ui = ns.ui;
    var app = ui.host('ws-app', 'div', 'ws-app');
    app.className = 'ws-app is-login';
    app.replaceChildren(ui.h('section', {'class': 'ws-login'}, ui.h('div', {'class': 'login-form'},
      ui.h('div', {'class': 'login-mark'}, ui.h('span', {'class': 'mark', text: 'WSS'})),
      ui.h('p', {text: text}), ui.h('p', {}, ui.button('重试', function () { root.location.reload(); }), ' ', ui.link('经典工作台', '/')))));
  }
  function startApp(session) {
    ns.api.setCsrf(session.csrf_token || '');
    return ns.shell.start({session: session});
  }
  function onUnauthorized() {
    if (expired) return;
    expired = true;
    ns.api.session().then(function (s) { return s; }, function () { return {login: 'password'}; }).then(function (s) {
      ns.api.setCsrf(s && s.csrf_token || '');
      ns.shell.showLogin(s || {login: 'password'}, function () { root.location.reload(); });
      ns.ui.toast('登录已过期，请重新登录。登录后回到刚才的病例。', {kind: 'info'});
    });
  }
  function boot() {
    if (started) return Promise.resolve(null);
    started = true;
    var missing = ['ui', 'store', 'shell'].filter(function (k) { return !ns[k]; });
    if (missing.length) { if (root.console) root.console.error('WSSV2 missing modules: ' + missing.join(', ')); return Promise.resolve(null); }
    if (isOffline()) return ns.shell.startOffline({offlineMeta: readJson('wssv2-offline')});
    if (!ns.api) { fatal('页面文件不完整（缺少服务接口模块）。'); return Promise.resolve(null); }
    ns.api.state.onUnauthorized = onUnauthorized;
    return ns.api.session().then(function (s) {
      ns.api.setCsrf(s.csrf_token || '');
      if (!s.authenticated) {
        ns.shell.showLogin(s, function (s2) { startApp(s2); });
        return null;
      }
      return startApp(s);
    }, function (e) {
      fatal('暂时连不上服务：' + (e && e.message || e) + ' 已保存的结果不会丢失。');
      return null;
    });
  }
  function autostart() {
    if (root.WSSV2_NO_AUTOSTART || !root.document || typeof root.document.getElementById !== 'function') return;
    if (root.document.readyState === 'loading' && root.document.addEventListener) root.document.addEventListener('DOMContentLoaded', boot);
    else setTimeout(boot, 0);
  }
  autostart();
  return {boot: boot, isOffline: isOffline};
});
