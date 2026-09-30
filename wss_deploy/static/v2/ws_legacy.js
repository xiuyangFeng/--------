/* WSS workspace v2 — phase 3 lane 6 (S7, PHASE3_LANES.md §3 item 2): classic addresses become workspace routes.
 *
 * The server sends every retired classic address here (server.py _classic_redirect):
 *   /               → /v2/            the browser keeps the fragment, e.g. the classic workbench's #job=<id>
 *   …/report[#view] → /v2/?job=<id>   the classic report; its #view=<base64url JSON> (wss-deploy.view/v1) rides along
 *   /compare?…      → /v2/#/job/A?v=compare&cmp=B   (a workspace route already, nothing to do here)
 * At boot (this file runs after ws_shell.js and before ws_main.js starts the page) the address is rewritten in place
 * to #/job/<id>[?f=<field>] with history.replaceState, before the shell reads its route.  A classic view state is
 * kept in memory and, once that result is open, converted with the result's manifest (classicToV2, pure) into the
 * workspace's reproducible-view state {v:1, …} (ws_figure.js) and applied through ws_figure's decode + applyView,
 * exactly like a workspace link.  Settings the workspace cannot carry are named in one notice.
 * Online only (bundle.json offline_exclude). */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.legacy = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var JOB_ID = /^[A-Za-z0-9_-]{1,80}$/;
  var CLASSIC_SCHEMA = 'wss-deploy.view/v1';
  var V2_FOV = 30;                                   // core_viewer.js PerspectiveCamera
  var CLASSIC_FOV = { wall: 35, volume: 42 };        // report.py / volume_viewer.js PerspectiveCamera
  var WALL_FIELDS = ['wss', 'tawss', 'osi', 'rrt', 'ecap'];
  var LOG_FIELDS = { wss: true, tawss: true, rrt: true, ecap: true };   // core_colormap.js logHint without a manifest hint
  var VOLUME_MODES = ['cloud', 'slice', 'wall', 'streamlines'];
  var SLICE_RANGES = ['section', 'global', 'manual'];

  // ------------------------------------------------------------------ pure: addresses
  function decodePart(s) { try { return decodeURIComponent(String(s).replace(/\+/g, ' ')); } catch (_) { return null; } }
  // ``?job=<id>`` of the redirect (the last one wins, as parse_qs does on the server).
  function jobFromSearch(search) {
    var out = null;
    String(search || '').replace(/^\?/, '').split('&').forEach(function (part) {
      var i = part.indexOf('=');
      if (i < 0 || part.slice(0, i) !== 'job') return;
      var v = decodePart(part.slice(i + 1));
      out = v !== null && JOB_ID.test(v) ? v : null;
    });
    return out;
  }
  // A classic fragment: ``job=<id>`` (workbench; the first job= part, as workbench_core.parseJobHash) and
  // ``view=<base64url>`` (report).  A workspace route (#/…) is not classic.
  function classicHash(hash) {
    var raw = String(hash || '').replace(/^#/, '');
    var out = { job: null, view: null };
    if (!raw || raw.charAt(0) === '/') return out;
    var jobSeen = false;
    raw.split('&').forEach(function (part) {
      var i = part.indexOf('=');
      if (i < 0) return;
      var k = part.slice(0, i), v = part.slice(i + 1);
      if (k === 'job' && !jobSeen) { jobSeen = true; var id = decodePart(v); out.job = id !== null && JOB_ID.test(id) ? id : null; }
      else if (k === 'view' && out.view === null && v) out.view = v;   // decoded later; unreadable text → a notice
    });
    return out;
  }
  // What the boot rewrite needs: {jobId, view (base64url text or null), keepHash (already a workspace route), classic}.
  function parseAddress(search, hash) {
    var fromQuery = jobFromSearch(search), h = classicHash(hash);
    var workspace = /^#\//.test(String(hash || ''));
    var jobId = fromQuery || (workspace ? null : h.job);
    var jobParam = /(^\?|&)job(=|&|$)/.test(String(search || ''));
    return { jobId: jobId, view: jobId && !workspace ? h.view : null, keepHash: workspace,
      classic: jobParam || Boolean(!workspace && (h.job || h.view)) };
  }

  // ------------------------------------------------------------------ pure: the classic view state
  function fromBase64Url(text) {
    var s = String(text || '').replace(/-/g, '+').replace(/_/g, '/');
    while (s.length % 4) s += '=';
    var bin = root.atob(s);
    if (typeof root.TextDecoder === 'function') {
      var u = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
      return new root.TextDecoder('utf-8', { fatal: true }).decode(u);
    }
    return decodeURIComponent(escape(bin));
  }
  // report.py decodeView / volume_viewer.js decodeView: a JSON object; a stated schema must be the classic v1.
  function decodeClassicView(text) {
    var st = JSON.parse(fromBase64Url(text));
    if (!st || typeof st !== 'object' || Array.isArray(st)) throw new Error('invalid classic view state');
    if (st.schema_version !== undefined && st.schema_version !== CLASSIC_SCHEMA) throw new Error('unknown classic view schema');
    return st;
  }

  function num(x) { var n = Number(x); return x === null || x === undefined || x === '' || typeof x === 'boolean' || !Number.isFinite(n) ? null : n; }
  function vec3(a) { return Array.isArray(a) && a.length === 3 && a.every(function (x) { return num(x) !== null; }) ? a.map(Number) : null; }
  function round(x, d) { return +Number(x).toFixed(d === undefined ? 4 : d); }
  function sameList(a, b) { return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every(function (x, i) { return Number(x) === Number(b[i]); }); }
  function familyOf(classic, ctx) {
    if (ctx && (ctx.family === 'wall' || ctx.family === 'volume')) return ctx.family;
    if (classic.family === 'wall' || classic.family === 'volume') return classic.family;
    return classic.field === 'velocity' || classic.field === 'pressure' || VOLUME_MODES.indexOf(classic.mode) > 0 ? 'volume' : 'wall';
  }

  // Camera {position, target, up} of the classic page → {p, t, u, fov}: the workspace keeps its own field of view and
  // moves along the line of sight so the same extent fills the view (distance × tan(classic / 2) / tan(v2 / 2)).
  function convertCamera(cam, family, ctx) {
    var p = vec3(cam && cam.position), t = vec3(cam && cam.target);
    if (!p || !t) return null;
    var u = vec3(cam.up) || [0, 1, 0];
    var from = CLASSIC_FOV[family] || CLASSIC_FOV.wall, to = (ctx && num(ctx.fov)) || V2_FOV;
    var k = Math.tan(from * Math.PI / 360) / Math.tan(to * Math.PI / 360);
    return { p: [0, 1, 2].map(function (i) { return round(t[i] + (p[i] - t[i]) * k); }), t: t.map(function (x) { return round(x); }),
      u: u.map(function (x) { return round(x); }), fov: to };
  }

  // The classic volume 横截面 → the workspace section state (ws_slice.js st).  basis 'pick' (its plane came from the
  // picked points and the centreline and is not stored) and x / y / z (no such basis in the workspace) fall back to
  // the centreline position.  Cut sides: volume_viewer applyCutPlanes keeps n·(x − o) ≥ 0 for cut-positive;
  // ws_slice 'pos' → core_viewer setClipPlane(side 1) keeps the same side of the same plane.
  function convertSlice(classic, drop) {
    var sl = classic.slice && typeof classic.slice === 'object' ? classic.slice : {};
    var out = { basis: 'centerline', pick: null, picks: [], shift: 0 };
    var fromCenterline = sl.basis === 'centerline' || sl.basis === undefined;
    if (sl.basis === 'pick') drop('点选截面');
    else if (sl.basis === 'x' || sl.basis === 'y' || sl.basis === 'z') drop('按坐标轴放的截面');
    var seg = num(sl.branch);
    out.segment = seg === null ? null : Math.round(seg);
    var pos = num(sl.position);
    out.fraction = pos === null || !fromCenterline ? 0.5 : Math.max(0, Math.min(1, pos / 100));
    out.pitch = fromCenterline && num(sl.pitch) !== null ? num(sl.pitch) : 0;
    out.yaw = fromCenterline && num(sl.yaw) !== null ? num(sl.yaw) : 0;
    out.offU = fromCenterline && num(sl.offset_u) !== null ? num(sl.offset_u) : 0;
    out.offV = fromCenterline && num(sl.offset_v) !== null ? num(sl.offset_v) : 0;
    if (num(sl.thickness) !== null && num(sl.thickness) > 0) out.thickness = num(sl.thickness);
    // v0.12 colour keys; a complete older state without them was drawn on the whole-field range, speed, no arrows.
    var legacy = Boolean(classic.schema_version);
    var cr = sl.color_range && typeof sl.color_range === 'object' ? sl.color_range : null;
    if (cr && SLICE_RANGES.indexOf(cr.mode) >= 0) out.range = cr.mode;
    else if (!cr && legacy) out.range = 'global';
    if (cr && num(cr.min) !== null && num(cr.max) !== null) out.manual = { min: num(cr.min), max: num(cr.max) };
    if (classic.field === 'pressure') out.quantity = 'pressure';
    else if (sl.quantity === 'speed' || sl.quantity === 'normal') out.quantity = sl.quantity;
    else if (legacy) out.quantity = 'speed';
    if (sl.arrows !== undefined) out.arrows = Boolean(sl.arrows);
    else if (legacy) out.arrows = false;
    if (sl.fill !== undefined) out.fill = sl.fill !== false;
    out.cut = sl.cut && sl.module === 'cut-positive' ? 'pos' : sl.cut && sl.module === 'cut-negative' ? 'neg' : 'none';
    return out;
  }

  // classicToV2(classic, ctx) → {state: {v:1, run, f, w, L, cam, br?, sl?}, dropped: [labels], hiddenBranches, family}
  //   ctx (optional, from the open result): {family, fields: [ids], branches: [ids], logFields: [ids], fov}.
  // Without ctx.branches the classic hidden branches cannot become the workspace's list of shown branches: ``br`` is
  // left out and ``hiddenBranches`` keeps them.  ``dropped`` names, in words, the non-default classic settings the
  // workspace state does not carry (PHASE3_LANES.md §3 lane 6 item 2; p3_audit_reports.md §F).
  function classicToV2(classic, ctx) {
    ctx = ctx || {};
    classic = classic && typeof classic === 'object' ? classic : {};
    var dropped = [];
    var drop = function (label) { if (dropped.indexOf(label) < 0) dropped.push(label); };
    var family = familyOf(classic, ctx);
    var st = { v: 1 };
    if (classic.run_identity) st.run = String(classic.run_identity).slice(0, 16);
    var mode = typeof classic.mode === 'string' ? classic.mode : null;
    var L = {};
    // ---- field and layers
    var field = null;
    if (family === 'volume') {
      field = classic.field === 'pressure' ? 'pressure' : classic.field === 'velocity' ? 'speed' : null;
      if (mode === 'wall') field = 'wall_pressure';
      if (mode === 'streamlines') { field = 'speed'; L.streamlines = true; L.interior = false; }
      else if (mode === 'cloud') { L.streamlines = false; L.interior = true; }
      else if (mode === 'slice') L.streamlines = false;
    } else {
      field = WALL_FIELDS.indexOf(classic.field) >= 0 ? classic.field : null;
      if (mode === 'cl') L.centerline = true;
      else if (mode === 'cloud') L.points = true;
      else if (mode === 'stl') drop('输入 STL 视图');
    }
    if (field && Array.isArray(ctx.fields) && ctx.fields.indexOf(field) < 0) field = null;
    if (field) st.f = field;
    var overlay = classic.overlay && typeof classic.overlay === 'object' ? classic.overlay : {};
    if (typeof overlay.trust === 'boolean') L.trust = overlay.trust;
    if (overlay.contours === true) drop('等值线');
    if (overlay.stagnation === true) drop('滞留区叠加');
    st.L = L;
    // ---- colour window (the classic range is in the field's own base unit, as the workspace windows are)
    var range = classic.range && typeof classic.range === 'object' ? classic.range : {};
    var logFields = Array.isArray(ctx.logFields) ? ctx.logFields : Object.keys(LOG_FIELDS);
    var logDefault = Boolean(field && logFields.indexOf(field) >= 0);
    var fixedMax = num(range.max);
    var fixedHere = range.mode === 'fixed' && fixedMax !== null && fixedMax > 0 && (!range.field || range.field === classic.field);
    if (fixedHere) {
      st.w = { range: [0, fixedMax] };
      if (classic.log === true && logDefault) drop('固定色标的对数刻度');
    } else {
      // the classic wall report opens linear (logScale false) unless the reader ticked 对数
      st.w = classic.log === false && logDefault ? 'adaptive-linear' : 'adaptive';
      if (classic.log === true && !logDefault && family === 'volume') drop('对数色标');
    }
    // ---- camera
    var cam = convertCamera(classic.camera, family, ctx);
    if (cam) st.cam = cam;
    // ---- shown branches
    var hidden = Array.isArray(classic.branches_hidden) ? classic.branches_hidden.map(Number).filter(Number.isFinite) : [];
    if (hidden.length && Array.isArray(ctx.branches)) {
      st.br = ctx.branches.map(Number).filter(function (id) { return hidden.indexOf(id) < 0; });
      if (!st.br.length) st.br = null;
    }
    // ---- volume section
    if (family === 'volume' && (mode === 'slice' || (classic.slice && classic.slice.cut))) st.sl = convertSlice(classic, drop);
    // ---- what the workspace state does not carry (non-default values only)
    if (classic.colormap && classic.colormap !== 'rainbow') drop('配色');
    if (num(classic.bands)) drop('分段色带');
    if (family === 'wall') {
      if (classic.units === 'dyn') drop('dyn/cm² 单位');
      if (Array.isArray(classic.thresholds_pa) && !sameList(classic.thresholds_pa, [0.4, 4, 7])) drop('阈值');
      if (classic.field_thresholds && typeof classic.field_thresholds === 'object' && Object.keys(classic.field_thresholds).length) drop('阈值');
      if (num(classic.opacity) !== null && num(classic.opacity) < 1) drop('壁面透明度');
      if (classic.slice && num(classic.slice.clip) !== null && num(classic.slice.clip) < 1) drop('剖切');
      var hl = classic.highlight && typeof classic.highlight === 'object' ? classic.highlight : {};
      if (hl.top === true) drop('高亮最高区域');
      if (num(hl.branch) !== null && num(hl.branch) >= 0) drop('高亮分支');
      if (mode === 'cloud' && hl.feature && hl.feature !== 'wss') drop('点云特征着色');
      if (hl.finding) drop('选中的发现');
    } else {
      var ub = classic.units_by_field && typeof classic.units_by_field === 'object' ? classic.units_by_field : {};
      if ((ub.pressure && ub.pressure !== 'Pa') || (ub.velocity && ub.velocity !== 'm/s')) drop('单位');
      if (num(classic.opacity) !== null && Math.abs(num(classic.opacity) - 0.1) > 1e-9) drop('外壁透明度');
      var sls = classic.streamlines && typeof classic.streamlines === 'object' ? classic.streamlines : {};
      if ((num(sls.width) !== null && num(sls.width) !== 1) || (sls.density !== undefined && String(sls.density) !== '1') || sls.thin === true) drop('流线粗细和密度');
      if (classic.vectors === true) drop('速度箭头');
      var vh = classic.highlight && typeof classic.highlight === 'object' ? classic.highlight : {};
      if (vh.region) drop('区域统计');
      if (vh.finding) drop('选中的发现');
    }
    var lb = classic.labels && typeof classic.labels === 'object' ? classic.labels : {};
    if (num(lb.findings) || lb.branches === true) drop('自动标注');
    if (Array.isArray(classic.measurements) && classic.measurements.length) drop('测量');
    if (Array.isArray(classic.probe_log) && classic.probe_log.length) drop('探针记录');
    if (classic.lang === 'en') drop('英文标注');
    return { state: st, dropped: dropped, hiddenBranches: hidden, family: family };
  }

  // The route the boot rewrite writes: #/job/<id>[?f=<field>] (the field, so the first paint is already right).
  function routeHash(jobId, field) {
    return '#/job/' + encodeURIComponent(jobId) + (field ? '?f=' + encodeURIComponent(field) : '');
  }
  // The search part without ``job`` (other parameters stay).
  function searchWithoutJob(search) {
    var rest = String(search || '').replace(/^\?/, '').split('&').filter(function (part) { return part && !/^job(=|$)/.test(part); });
    return rest.length ? '?' + rest.join('&') : '';
  }

  // ctx for classicToV2 from an open result's manifest.
  function manifestContext(manifest) {
    var m = manifest || {};
    var fields = (m.fields || []).filter(function (f) { return f && f.id && f.kind !== 'vector'; });
    var hint = ns.colormap && typeof ns.colormap.logHint === 'function' ? ns.colormap.logHint : function (f) { return Boolean(LOG_FIELDS[f.id]); };
    return {
      family: m.result && m.result.family,
      fields: fields.map(function (f) { return f.id; }),
      logFields: fields.filter(function (f) { try { return Boolean(hint(f)); } catch (_) { return false; } }).map(function (f) { return f.id; }),
      branches: ((m.geometry && m.geometry.branches) || []).map(function (b) { return Number(b && b.id); }).filter(Number.isFinite)
    };
  }

  // ------------------------------------------------------------------ boot rewrite (runs once, before the shell)
  var pending = null;   // {jobId, classic | null, unreadable}
  function rewrite(win) {
    win = win || root;
    var loc = win && win.location;
    if (!loc) return null;
    var a = parseAddress(loc.search, loc.hash);
    if (!a.classic) return null;
    var classic = null, unreadable = false, field = null;
    if (a.view) {
      try { classic = decodeClassicView(a.view); field = classicToV2(classic, {}).state.f || null; }
      catch (_) { classic = null; unreadable = true; }
    }
    var hash = a.keepHash ? loc.hash : a.jobId ? routeHash(a.jobId, field) : '#/';
    var url = (loc.pathname || '/v2/') + searchWithoutJob(loc.search) + hash;
    try { win.history.replaceState(null, '', url); } catch (_) { try { loc.hash = hash; } catch (_e) { /* the shell opens the home page */ } }
    pending = a.jobId && !a.keepHash && (classic || unreadable) ? { jobId: a.jobId, classic: classic, unreadable: unreadable } : null;
    return { url: url, pending: pending };
  }

  // ------------------------------------------------------------------ after the result opened
  function notice(api, text) {
    var U = api && api.ui ? (typeof api.ui === 'function' ? api.ui() : api.ui) : ns.ui;   // the shell API's ui is a getter
    if (!text || !U || typeof U.toast !== 'function') return;
    U.toast(text, { kind: 'info', ms: 9000 });
  }
  function applyPending(api) {
    var p = pending;
    pending = null;
    var cur = api && api.cur ? api.cur() : null;
    if (!p || !cur || cur.jobId !== p.jobId || !cur.manifest) return Promise.resolve(null);
    if (p.unreadable) { notice(api, '旧版链接里的视图无法读取，已按默认打开。'); return Promise.resolve(null); }
    var conv = classicToV2(p.classic, manifestContext(cur.manifest));
    var F = ns.figure, state = conv.state;
    // Through the workspace link decoder, so the state is read exactly as a pasted reproducible link would be.
    if (F && typeof F.encodeState === 'function' && typeof F.decodeState === 'function') {
      try { state = F.decodeState(F.encodeState(state)); } catch (_) { state = conv.state; }
    }
    var applied = F && typeof F.applyView === 'function' ? Promise.resolve(F.applyView(api, state)) : Promise.resolve([]);
    return applied.then(function (notes) {
      var parts = [];
      if (Array.isArray(notes) && notes.length) parts.push(notes.join('；'));
      if (conv.dropped.length) parts.push('旧版链接里的' + conv.dropped.join('、') + '没有带过来');
      notice(api, parts.length ? '已按旧版链接打开。' + parts.join('；') + '。' : null);
      return { state: state, dropped: conv.dropped, notes: Array.isArray(notes) ? notes : [] };
    }, function () { return null; });
  }

  // In the page: rewrite now.  Under Node (tests) only when asked, so requiring the module has no side effects.
  if (!(typeof module === 'object' && module.exports) || root.WSSV2_LEGACY_AUTOREWRITE) {
    try { rewrite(root); } catch (e) { if (root.console && root.console.error) root.console.error('ws_legacy: ' + (e && e.stack || e)); }
  }
  (ns.ext = ns.ext || []).push({
    id: 'legacy',
    onResult: function (api) { if (pending) applyPending(api); }
  });

  return {
    CLASSIC_SCHEMA: CLASSIC_SCHEMA, parseAddress: parseAddress, jobFromSearch: jobFromSearch, classicHash: classicHash,
    decodeClassicView: decodeClassicView, classicToV2: classicToV2, convertCamera: convertCamera, routeHash: routeHash,
    searchWithoutJob: searchWithoutJob, manifestContext: manifestContext, rewrite: rewrite, applyPending: applyPending,
    pending: function () { return pending; }
  };
});
