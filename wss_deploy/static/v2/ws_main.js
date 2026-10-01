/* WSS workspace v2 — start-up (contract §6.3 ws_main.js): offline package → offline shell; otherwise
 * session → login form when needed → shell.  The only module that starts work on load.
 * Lane C (S6c): a session that expires while working is renewed in place — a login form opens over the page (its own
 * modal <dialog>, above any open dialog), so the open case, the scroll position and anything typed in a dialog stay;
 * the same user carries on without a reload, another user gets a fresh page (classic workbench v0.15 round 15). */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.main = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var started = false, expired = false, shellStarted = false, relogin = null;

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
      ui.h('p', {text: text}), ui.h('p', {}, ui.button('重试', function () { root.location.reload(); })))));
  }
  function startApp(session) {
    ns.api.setCsrf(session.csrf_token || '');
    shellStarted = true;
    return ns.shell.start({session: session});
  }
  // The session the shell started with; updated in place after a re-login (the shell and the store share the object).
  function currentSession() { return (ns.store && ns.store.state && ns.store.state.session) || null; }
  function onUnauthorized() {
    if (expired) return;
    expired = true;
    if (!shellStarted) {   // before the workspace exists: the plain login page, then a reload
      ns.api.session().then(function (s) { return s; }, function () { return {login: 'password'}; }).then(function (s) {
        ns.api.setCsrf(s && s.csrf_token || '');
        ns.shell.showLogin(s || {login: 'password'}, function () { root.location.reload(); });
        ns.ui.toast('登录已过期，请重新登录。登录后回到刚才的病例。', {kind: 'info'});
      });
      return;
    }
    ns.api.session().then(function (s) { return s || {}; }, function () { return {login: 'password'}; }).then(function (s) {
      if (s.authenticated) {   // another tab logged in meanwhile: carry on with that session (another user: a fresh page)
        var sess = currentSession();
        if (sess && sess.username && s.username !== sess.username) { root.location.reload(); return; }
        expired = false; ns.api.setCsrf(s.csrf_token || '');
        if (sess) Object.keys(s).forEach(function (k) { sess[k] = s[k]; });
        return;
      }
      openRelogin(s);
    });
  }
  function openRelogin(fresh) {
    var ui = ns.ui, h = ui.h;
    var before = currentSession() || {};
    var mode = (fresh && fresh.login) === 'password' || before.login === 'password' ? 'password' : 'token';
    var dlg = ui.host('ws-relogin', 'dialog', 'ws-dialog ws-relogin');
    var user = h('input', {type: 'text', autocomplete: 'username', 'aria-label': '用户名', value: before.username || ''});
    var noun = mode === 'password' ? '口令' : '访问令牌';
    var pass = h('input', {type: 'password', autocomplete: mode === 'password' ? 'current-password' : 'off', 'aria-label': noun});
    var err = h('p', {'class': 'login-err', role: 'alert', hidden: true});
    var submit = h('button', {type: 'submit', 'class': 'btn btn-primary', text: '登录'});
    // the same helps as the login page (ws_shell showLogin): show / hide, Caps Lock, the wait a 429 names, 5xx as maintenance
    var reveal = ui.iconButton('eye', '显示' + noun, function () {
      var show = pass.type === 'password';
      pass.type = show ? 'text' : 'password';
      ui.fill(reveal, ui.icon(show ? 'eye-off' : 'eye'));
      reveal.setAttribute('aria-label', (show ? '隐藏' : '显示') + noun); reveal.title = (show ? '隐藏' : '显示') + noun;
      reveal.setAttribute('aria-pressed', String(show));
      try { pass.focus(); } catch (_) {}
    }, {cls: 'login-reveal', pressed: false});
    var caps = h('p', {'class': 'login-caps', role: 'status', hidden: true, text: '大写锁定已打开，' + noun + '区分大小写。'});
    var capsCheck = function (ev) { caps.hidden = !(ev && typeof ev.getModifierState === 'function' && ev.getModifierState('CapsLock')); };
    pass.addEventListener('keydown', capsCheck); pass.addEventListener('keyup', capsCheck);
    pass.addEventListener('blur', function () { caps.hidden = true; });
    var waitTimer = null;
    var cooldown = function (seconds) {
      if (waitTimer) clearInterval(waitTimer);
      var until = Date.now() + Math.max(1, Math.round(seconds)) * 1000;
      var paint = function () {
        var left = Math.ceil((until - Date.now()) / 1000);
        if (left > 0) { submit.disabled = true; submit.textContent = left + ' 秒后可再试'; err.hidden = false; err.textContent = '尝试过于频繁。（' + left + ' 秒后可再试）'; return; }
        clearInterval(waitTimer); waitTimer = null; submit.disabled = false; submit.textContent = '登录'; err.hidden = true;
      };
      paint();
      waitTimer = setInterval(paint, 1000);
      if (waitTimer && waitTimer.unref) waitTimer.unref();
    };
    var form = h('form', {'class': 'relogin-form'},
      h('div', {'class': 'dlg-head'}, h('h2', {text: '登录已过期'})),
      h('div', {'class': 'dlg-body'},
        h('p', {'class': 'note', text: '页面和对话框里的内容都还在，登录后继续。'}),
        mode === 'password' ? h('label', {'class': 'fld'}, h('span', {text: '用户名'}), user) : null,
        h('label', {'class': 'fld'}, h('span', {text: noun}), h('span', {'class': 'login-secret'}, pass, reveal)),
        caps, err),
      h('div', {'class': 'dlg-foot'}, h('a', {'class': 'lnk relogin-other', href: '/v2/', text: '换个账号'}), h('span', {'class': 'sec-fill'}), submit));
    form.addEventListener('submit', function (ev) {
      if (ev && ev.preventDefault) ev.preventDefault();
      submit.disabled = true; err.hidden = true;
      var body = mode === 'password' ? {username: user.value.trim(), password: pass.value} : {token: pass.value};
      ns.api.login(body).then(function (s) {
        pass.value = '';
        if (before.username && s.username !== before.username) { root.location.reload(); return; }   // someone else: nothing of the old page stays
        ns.api.setCsrf(s.csrf_token || '');
        var sess = currentSession();
        if (sess) Object.keys(s).forEach(function (k) { sess[k] = s[k]; });
        closeRelogin();
        expired = false;
        if (ns.admin && ns.admin.resumed) ns.admin.resumed();
        ui.toast('已重新登录。刚才没有完成的操作，请再点一次。', {kind: 'ok'});
      }, function (e) {
        if (e.status === 429) { var after = Number(e.body && e.body.retry_after); cooldown(isFinite(after) && after > 0 ? Math.ceil(after) : 60); return; }
        submit.disabled = false; err.hidden = false;
        err.textContent = e.status === 401 ? (mode === 'password' ? '用户名或口令不正确（区分大小写）。' : '访问令牌不正确。')
          : e.status === 502 || e.status === 503 || e.status === 504 ? '服务正在启动或维护，请稍后再试。' : e.message;
        if (e.status === 401) { try { pass.focus(); if (pass.select) pass.select(); } catch (_) {} }
      });
    });
    ui.fill(dlg, form);
    if (!dlg._wsRelogin) {
      dlg._wsRelogin = true;
      dlg.addEventListener('cancel', function (ev) { if (ev && ev.preventDefault) ev.preventDefault(); });   // the page is unusable until logged in
    }
    try { if (typeof dlg.showModal === 'function') { if (!dlg.open) dlg.showModal(); } else dlg.setAttribute('open', ''); } catch (_) { dlg.setAttribute('open', ''); }
    relogin = {dlg: dlg, form: form, user: user, pass: pass, err: err, submit: submit};
    try { (mode === 'password' && !user.value ? user : pass).focus(); } catch (_) {}
    return relogin;
  }
  function closeRelogin() {
    if (!relogin) return;
    var dlg = relogin.dlg;
    relogin = null;
    try { if (typeof dlg.close === 'function' && dlg.open) dlg.close(); else dlg.removeAttribute('open'); } catch (_) {}
    dlg.replaceChildren();
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
      var st = e && e.status;
      fatal(st === 502 || st === 503 || st === 504 ? '服务正在启动或维护，请稍后再试。已保存的结果不会丢失。'
        : '暂时连不上服务：' + (e && e.message || e) + ' 已保存的结果不会丢失。');
      return null;
    });
  }
  function autostart() {
    if (root.WSSV2_NO_AUTOSTART || !root.document || typeof root.document.getElementById !== 'function') return;
    if (root.document.readyState === 'loading' && root.document.addEventListener) root.document.addEventListener('DOMContentLoaded', boot);
    else setTimeout(boot, 0);
  }
  autostart();
  return {boot: boot, isOffline: isOffline, reloginOpen: function () { return Boolean(relogin); }};
});
