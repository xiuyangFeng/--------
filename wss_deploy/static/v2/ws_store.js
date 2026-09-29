/* WSS workspace v2 — state contract A8 (contract §6.4).
 *   偏好      localStorage wssv2:prefs                 tier / lighting / colour map / panels
 *   阅读位置  localStorage wssv2:view:<run_identity>   viewer state, field, window; same result only
 *   书签      localStorage wssv2:bm:<run_identity>     (ws_bookmarks.js)
 *   业务事实  the service (status, review, findings decisions, edited conclusion)
 *   暂态      memory
 * Every storage access is wrapped: a private window or blocked storage must not break the page. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.store = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var PREFS_KEY = 'wssv2:prefs';
  var DEFAULTS = {schema: 'wssv2.prefs/1', tier: 'basic', lighting: 'flat', cmap: 'rainbow', rail: true, inspector: true, tab: 'overview'};
  var TIERS = ['basic', 'full'], LIGHTS = ['flat', 'soft'], CMAPS = ['rainbow', 'viridis', 'turbo'], TABS = ['overview', 'reading', 'bookmarks', 'tools', 'compare'];

  function storage() { try { return root.localStorage || null; } catch (_) { return null; } }
  function read(key) {
    try { var s = storage(); if (!s) return null; var raw = s.getItem(key); return raw === null || raw === undefined ? null : JSON.parse(raw); }
    catch (_) { return null; }
  }
  function write(key, value) {
    try { var s = storage(); if (!s) return false; s.setItem(key, JSON.stringify(value)); return true; }
    catch (_) { return false; }
  }
  function remove(key) { try { var s = storage(); if (s) s.removeItem(key); } catch (_) {} }

  function sanitizePrefs(raw) {
    var p = {};
    Object.keys(DEFAULTS).forEach(function (k) { p[k] = DEFAULTS[k]; });
    if (!raw || typeof raw !== 'object') return p;
    if (TIERS.indexOf(raw.tier) >= 0) p.tier = raw.tier;
    if (LIGHTS.indexOf(raw.lighting) >= 0) p.lighting = raw.lighting;
    if (CMAPS.indexOf(raw.cmap) >= 0) p.cmap = raw.cmap;
    if (typeof raw.rail === 'boolean') p.rail = raw.rail;
    if (typeof raw.inspector === 'boolean') p.inspector = raw.inspector;
    if (TABS.indexOf(raw.tab) >= 0 && raw.tab !== 'compare') p.tab = raw.tab;
    return p;
  }
  var prefsCache = null;
  function prefs() { if (!prefsCache) prefsCache = sanitizePrefs(read(PREFS_KEY)); return prefsCache; }
  function setPrefs(patch) {
    var next = sanitizePrefs(Object.assign({}, prefs(), patch || {}));
    prefsCache = next; write(PREFS_KEY, next); emit('prefs', next);
    return next;
  }

  // Reading position: attached to one result identity; never migrated to another result.
  function viewKey(runIdentity) { return 'wssv2:view:' + runIdentity; }
  function readView(runIdentity) {
    if (!runIdentity) return null;
    var v = read(viewKey(runIdentity));
    if (!v || typeof v !== 'object' || v.run_identity !== runIdentity) return null;
    return v;
  }
  function writeView(runIdentity, view) {
    if (!runIdentity || !view) return false;
    var value = Object.assign({}, view, {run_identity: runIdentity, saved_at: new Date().toISOString()});
    return write(viewKey(runIdentity), value);
  }
  function bookmarkKey(runIdentity) { return 'wssv2:bm:' + runIdentity; }
  function readBookmarks(runIdentity) {
    if (!runIdentity) return [];
    var v = read(bookmarkKey(runIdentity));
    return Array.isArray(v) ? v.filter(function (b) { return b && typeof b === 'object' && b.id; }) : [];
  }
  function writeBookmarks(runIdentity, list) { return runIdentity ? write(bookmarkKey(runIdentity), list || []) : false; }

  // Fast case switching: every open carries an increasing number; a late answer for an older open is dropped.
  var seq = 0;
  function nextSeq() { seq += 1; return seq; }
  function isLatest(n) { return n === seq; }
  function currentSeq() { return seq; }

  // Tiny event hub for panels that follow shared state (prefs, current result).
  var listeners = {};
  function on(name, fn) { (listeners[name] = listeners[name] || []).push(fn); return function () { off(name, fn); }; }
  function off(name, fn) { var l = listeners[name]; if (!l) return; var i = l.indexOf(fn); if (i >= 0) l.splice(i, 1); }
  function emit(name, payload) { (listeners[name] || []).slice().forEach(function (fn) { try { fn(payload); } catch (e) { if (root.console) root.console.error(e); } }); }

  // Shared memory state (business facts live on the server; this is the loaded copy).
  var state = {session: null, jobs: [], cards: {}, releases: [], current: null, offline: false};

  function reset() { prefsCache = null; seq = 0; listeners = {}; state = {session: null, jobs: [], cards: {}, releases: [], current: null, offline: false}; api.state = state; }

  var api = {read: read, write: write, remove: remove, prefs: prefs, setPrefs: setPrefs, sanitizePrefs: sanitizePrefs, DEFAULTS: DEFAULTS,
    readView: readView, writeView: writeView, readBookmarks: readBookmarks, writeBookmarks: writeBookmarks,
    nextSeq: nextSeq, isLatest: isLatest, currentSeq: currentSeq, on: on, off: off, emit: emit, state: state, reset: reset};
  return api;
});
