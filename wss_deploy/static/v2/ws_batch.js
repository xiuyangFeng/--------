/* WSS workspace v2 — phase 3 lane 5 (PHASE3_LANES.md §3 第 5 路): figures for many results at once — the classic
 * workbench 「跨病例批量出图」 (batch_export.js: a hidden classic report per job), rewritten on the v2 viewer.
 *
 * Entry: ns.batch.open(jobIds, shellApi) — lane 2's 「批量出图」 in the task list's selection bar calls it.
 * Each finished result is loaded like the workspace loads it (ns.data + the /api/v2 manifest and arrays), drawn on one
 * offscreen viewer (a hidden FRAME-sized viewport, the same frame for every result) and turned into figures by
 * ws_figure — the 图片 page's figure / the 拼图 page's montage: same colour bar, words, labels, crop and file names.
 * All files go into one zip (<case>_<job id>/… + manifest.json), like the classic batch.
 *
 * What is drawn, per family (壁面 / 体场): a field and a colour window — each result's own adaptive window or one fixed
 * range for all — the chosen standard views (one PNG per view, or one montage per result), and the 图片 page's style.
 * Bands, thresholds, units, colour map and labels follow this browser's settings (「设为默认」 per field), or a pasted
 * reproducible link (its field, window, colour map, bands, thresholds, units, labels and layers — never its camera:
 * other results have other geometry).  Numbers are never recomputed: the viewer, the colour bar and the file names are
 * the workspace's own. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.batch = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };
  var F = function () { return ns.figure; };

  var FRAME = {w: 900, h: 900};          // CSS px of the offscreen viewport
  var JOB_TIMEOUT_MS = 180000;
  var MAX_JOBS = 200;
  var SCHEMA = 'wss-deploy.batch_export/v2';
  var FIELDS = {
    wall: [{id: 'wss', label: 'WSS'}, {id: 'tawss', label: 'TAWSS', cycle: true}, {id: 'osi', label: 'OSI', cycle: true}, {id: 'rrt', label: 'RRT', cycle: true}, {id: 'ecap', label: 'ECAP', cycle: true}],
    volume: [{id: 'speed', label: '速度'}, {id: 'pressure', label: '压力'}, {id: 'wall_pressure', label: '壁面压力'}]
  };
  var UNITS = {wss: 'Pa', tawss: 'Pa', osi: '', rrt: '1/Pa', ecap: '1/Pa', speed: 'm/s', pressure: 'Pa', wall_pressure: 'Pa'};
  var FAMILY_TEXT = {wall: '壁面', volume: '体场'};
  var VIEWS = ['anterior', 'posterior', 'left', 'right', 'superior', 'inferior'];
  var VIEW_SHORT = {anterior: '前', posterior: '后', left: '左', right: '右', superior: '头', inferior: '足'};

  // ------------------------------------------------------------------ pure helpers (Node-tested)
  function familyOf(job) { return ui().kindOf(job) === 'volume' ? 'volume' : 'wall'; }   // cycle metrics are wall results
  function safeName(text) { return String(text || 'case').replace(/[\\/:*?"<>|\s]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 80) || 'case'; }
  function fieldLabel(id) { return (FIELDS.wall.concat(FIELDS.volume).filter(function (f) { return f.id === id; })[0] || {}).label || id; }
  // jobs: job records → {selected, skipped: [{id, name, message}], families: {wall, volume}, cycle}
  function plan(jobs) {
    var selected = [], skipped = [], fam = {wall: 0, volume: 0}, cycle = false, seen = {};
    (jobs || []).forEach(function (j) {
      if (!j || !j.id || seen[j.id]) return;
      seen[j.id] = true;
      if (j.status !== 'done') { skipped.push({id: j.id, name: ui().displayName(j), message: j._error || ('未完成（' + ui().statusInfo(j.status).label + '）')}); return; }
      if (selected.length >= MAX_JOBS) { skipped.push({id: j.id, name: ui().displayName(j), message: '一次最多 ' + MAX_JOBS + ' 个'}); return; }
      selected.push(j);
      fam[familyOf(j)] += 1;
      if (ui().kindOf(j) === 'cycle') cycle = true;
    });
    return {selected: selected, skipped: skipped, families: fam, cycle: cycle};
  }
  function fieldFamily(id) {
    if (FIELDS.wall.some(function (f) { return f.id === id; })) return 'wall';
    if (FIELDS.volume.some(function (f) { return f.id === id; }) || id === 'velocity') return 'volume';
    return null;
  }

  // Settings kept in this browser.  Per family: field ('auto' = each result's default), scale ('adaptive',
  // 'adaptive-linear', 'fixed' with lo / hi in the field's own units, or 'link' = the pasted link's named window).
  var PREF_KEY = 'wssv2:batch';
  var DEFAULT_VIEWS = ['anterior'];
  function num(v) { return v === '' || v === null || v === undefined || !isFinite(Number(v)) ? null : Number(v); }
  function sanitizeFamily(raw, fam) {
    var o = {field: 'auto', scale: 'adaptive', lo: null, hi: null};
    if (!raw || typeof raw !== 'object') return o;
    if (raw.field === 'auto' || FIELDS[fam].some(function (f) { return f.id === raw.field; })) o.field = raw.field;
    if (['adaptive', 'adaptive-linear', 'fixed', 'link'].indexOf(raw.scale) >= 0) o.scale = raw.scale;
    o.lo = num(raw.lo); o.hi = num(raw.hi);
    return o;
  }
  function sanitize(raw) {
    raw = raw && typeof raw === 'object' ? raw : {};
    var views = Array.isArray(raw.views) ? VIEWS.filter(function (v) { return raw.views.indexOf(v) >= 0; }) : DEFAULT_VIEWS.slice();
    return {views: views, layout: raw.layout === 'montage' ? 'montage' : 'files', svg: raw.svg === true,
      wall: sanitizeFamily(raw.wall, 'wall'), volume: sanitizeFamily(raw.volume, 'volume')};
  }
  function settings() { var raw = null; try { raw = ns.store && ns.store.read ? ns.store.read(PREF_KEY) : null; } catch (_) { raw = null; } return sanitize(raw); }
  function saveSettings(s) { var v = sanitize(s); try { if (ns.store && ns.store.write) ns.store.write(PREF_KEY, Object.assign({}, v, {wall: Object.assign({}, v.wall, v.wall.scale === 'link' ? {scale: 'adaptive'} : {}), volume: Object.assign({}, v.volume, v.volume.scale === 'link' ? {scale: 'adaptive'} : {})})); } catch (_) {} return v; }
  // Before a run: at least one view; a fixed range is two finite numbers, low < high, for one chosen field.
  function checkSettings(s, fams) {
    var errs = [];
    if (!s.views.length) errs.push('至少选一个视角');
    ['wall', 'volume'].forEach(function (fam) {
      if (!fams[fam]) return;
      var x = s[fam];
      if (x.scale === 'fixed') {
        if (x.field === 'auto') errs.push(FAMILY_TEXT[fam] + '：固定范围要先选一个字段');
        else if (x.lo === null || x.hi === null || !(x.lo < x.hi)) errs.push(FAMILY_TEXT[fam] + '：固定范围要两个数，下限小于上限');
      }
    });
    return errs;
  }

  // A reproducible link (the whole address, the hash or only its view= part) → {state, error}.
  function parseLink(text) {
    var t = String(text || '').trim();
    if (!t) return {state: null, error: null};
    var p = F() && F().linkParam ? F().linkParam(t) : null;
    if (!p && /^[A-Za-z0-9_-]{16,}$/.test(t)) p = t;
    if (!p) return {state: null, error: '这不是复现链接（地址里没有 view=）。'};
    try { return {state: F().decodeState(p), error: null}; } catch (_) { return {state: null, error: '链接里的视图状态读不出来。'}; }
  }
  // What a link sets: its family's field and window; the display (bands, thresholds, colour map, units, labels) for all.
  function linkPlan(st) {
    if (!st) return null;
    var fam = fieldFamily(st.f);
    return {family: fam, field: fam ? st.f : null, window: st.w || 'adaptive', d: st.d && typeof st.d === 'object' ? st.d : null, L: st.L && typeof st.L === 'object' ? st.L : null};
  }
  function applyLinkToSettings(s, lp) {
    if (!lp || !lp.family) return s;
    var x = s[lp.family];
    x.field = FIELDS[lp.family].some(function (f) { return f.id === lp.field; }) ? lp.field : 'auto';
    var w = lp.window;
    if (w && typeof w === 'object' && Array.isArray(w.range)) { x.scale = 'fixed'; x.lo = num(w.range[0]); x.hi = num(w.range[1]); }
    else if (w === 'adaptive-linear') x.scale = 'adaptive-linear';
    else if (typeof w === 'string' && w !== 'adaptive') x.scale = 'link';
    else x.scale = 'adaptive';
    return s;
  }
  // Which field and colour window a result gets → {field, window, notes}.
  function choose(manifest, set, link) {
    var notes = [], shown = ns.shell && ns.shell.visibleFields ? ns.shell.visibleFields(manifest) : (manifest.fields || []);
    var fields = shown.map(function (f) { return f.id; });
    var fid = set.field !== 'auto' && fields.indexOf(set.field) >= 0 ? set.field : null;
    if (!fid) {
      fid = ns.shell && ns.shell.defaultField ? ns.shell.defaultField(manifest) : fields[0];
      if (set.field !== 'auto') notes.push('没有 ' + fieldLabel(set.field) + '，用 ' + fieldLabel(fid));
    }
    var win = 'adaptive';
    if (set.scale === 'adaptive-linear') win = 'adaptive-linear';
    else if (set.scale === 'fixed' && fid === set.field && set.lo !== null && set.hi !== null) win = {range: [set.lo, set.hi]};
    else if (set.scale === 'fixed') notes.push('固定范围只用于 ' + fieldLabel(set.field) + '，这里用本例自适应');
    else if (set.scale === 'link' && link && typeof link.window === 'string') {
      var f = (manifest.fields || []).filter(function (x) { return x && x.id === fid; })[0];
      if (f && (f.windows || []).some(function (w) { return w && w.id === link.window; })) win = link.window;
      else notes.push('没有链接里的色标窗，用本例自适应');
    }
    return {field: fid, window: win, notes: notes};
  }
  // The window words in the result's own display units (the shell's windowLabel converts with the open result's family).
  function windowLabel(f, spec, family) {
    var cv = ns.display && ns.display.toDisplay;
    var rng = function (lo, hi) {
      if (!cv || !f) return ui().range(lo, hi, f && f.units);
      var a = cv(lo, f.units, family), b = cv(hi, f.units, family);
      return ui().range(a.value, b.value, b.units);
    };
    if (!spec || spec === 'adaptive' || spec === 'adaptive-linear') {
      var d = (f && f.display) || {}, hi = Array.isArray(d.range) ? d.range[1] : d.p99, lo = Array.isArray(d.range) ? d.range[0] : 0;
      var log = ns.colormap && ns.colormap.logHint ? ns.colormap.logHint(f) : d.log_scale === true;
      return '本例自适应' + (spec === 'adaptive-linear' ? '（线性）' : log ? '（对数）' : '') + (hi !== undefined && hi !== null ? ' ' + rng(lo, hi) : '');
    }
    if (typeof spec === 'object' && Array.isArray(spec.range)) return '固定范围 ' + rng(spec.range[0], spec.range[1]);
    var w = ((f && f.windows) || []).filter(function (x) { return x && x.id === spec; })[0];
    return w ? w.label + ' ' + rng(w.range[0], w.range[1]) + (w.provisional ? '（暂定）' : '') : String(spec);
  }
  function scaleSpec(win, cmap) {
    if (win === 'adaptive-linear') return {window: 'adaptive', log: false, bands: null, cmap: cmap};
    return {window: win || 'adaptive', log: null, bands: null, cmap: cmap};
  }

  // ------------------------------------------------------------------ the display for the offscreen viewer
  // ws_display answers only for the stage's two viewers.  While a batch runs its provider is wrapped: the offscreen
  // viewer gets this browser's display settings (or the link's) through ws_display's exported helpers; every other
  // viewer, and every method not listed here, is answered by the original provider.
  function linkD(hold) { return hold.link && hold.link.d ? hold.link.d : null; }
  function bandsFor(hold, fid) {
    var D = ns.display, d = linkD(hold), steps = (D && D.BAND_STEPS) || [0, 4, 6, 8, 10, 12, 16, 20];
    if (d && d.bands && d.bands[fid] !== undefined && steps.indexOf(Number(d.bands[fid])) >= 0) return Number(d.bands[fid]);
    var def = D && D.prefs ? (D.prefs().defaults || {})[fid] : null;
    return def && def.bands !== undefined ? Number(def.bands) : 0;
  }
  function thresholdsFor(hold, fid) {
    var D = ns.display, d = linkD(hold), valid = D && D.validThresholds ? D.validThresholds : function () { return null; };
    if (d && d.thr && valid(d.thr[fid])) return valid(d.thr[fid]);
    var def = D && D.prefs ? (D.prefs().defaults || {})[fid] : null;
    return def ? valid(def.thresholds) : null;
  }
  function unitFor(hold, kind) {
    var D = ns.display, d = linkD(hold), table = D && D.UNITS ? D.UNITS[kind] : null;
    if (d && d.units && table && Object.prototype.hasOwnProperty.call(table, d.units[kind])) return d.units[kind];
    return D && D.prefs ? (D.prefs().units || {})[kind] || null : null;
  }
  // Batch figures never show the input STL overlay or the velocity arrows (both hide the field on a still image).
  function layersFor(hold) {
    var D = ns.display, d = linkD(hold);
    return Object.assign({}, D && D.prefs ? D.prefs().layers : {}, d && d.layers ? d.layers : {}, {stl: false, vectors: false});
  }
  function installProvider(hold) {
    var base = ns.viewer && ns.viewer.displayProvider, D = ns.display;
    if (!base || typeof base !== 'object' || !D || typeof D.prefs !== 'function') return function () {};
    var over = Object.create(base);
    var own = function (v) { return Boolean(hold.viewer) && v === hold.viewer && Boolean(hold.result); };
    var wrap = function (name, fn) {
      over[name] = function (v, a, b) {
        if (!own(v)) return typeof base[name] === 'function' ? base[name](v, a, b) : null;
        return fn(a, b);
      };
    };
    wrap('bands', function (fid) { return bandsFor(hold, fid); });
    wrap('thresholds', function (fid) { return thresholdsFor(hold, fid); });
    wrap('units', function (field) {
      var kind = D.unitKind ? D.unitKind(field, hold.family) : null, u = kind ? unitFor(hold, kind) : null;
      var f = u && D.unitFactor ? D.unitFactor(kind, u) : 1;
      return u && f !== 1 ? {units: u, factor: f} : null;
    });
    wrap('flags', function () {
      var P = D.prefs(), L = layersFor(hold), V = Object.assign({}, P.volume || {}, (linkD(hold) || {}).volume || {}), m = hold.result.manifest;
      return {stl: false, stagnation: L.stagnation && D.hasStagnation && D.hasStagnation(m) ? D.stagnationCriteria(m) : null,
        opacity: V.opacity, density: V.density, width: V.width, thin: V.thin, vectors: false};
    });
    wrap('markers', function () {
      if (!layersFor(hold).peaks || !D.peakMarkers || hold.family !== 'wall') return [];
      var r = hold.result, f = r.field(hold.field), u = f ? ui().unitText(f.units) : '';
      return D.peakMarkers(r, hold.field).map(function (p) {
        p.text = (p.kind === 'min' ? '最小 ' : '最大 ') + (ns.util && ns.util.fmtSig ? ns.util.fmtSig(p.value) : String(+(+p.value).toPrecision(3))) + (u ? ' ' + u : '');
        return p;
      });
    });
    wrap('state', function () { return null; });
    wrap('restore', function () {});
    ns.viewer.displayProvider = over;
    return function () { if (ns.viewer && ns.viewer.displayProvider === over) ns.viewer.displayProvider = base; };
  }

  // ------------------------------------------------------------------ offscreen viewer
  function offscreen() {
    var doc = root.document;
    if (!ns.viewer || typeof ns.viewer.create !== 'function' || (ns.viewer.webglAvailable && !ns.viewer.webglAvailable())) throw new Error('这台设备的浏览器不能显示三维（WebGL 不可用），不能出图。');
    var host = doc.createElement('div');
    host.className = 'wsb-offscreen';
    host.setAttribute('aria-hidden', 'true');
    host.style.cssText = 'position:fixed;left:-20000px;top:0;width:' + FRAME.w + 'px;height:' + FRAME.h + 'px;pointer-events:none;overflow:hidden;';
    doc.body.appendChild(host);
    var v;
    try { v = ns.viewer.create(host, {kind: 'main'}); } catch (e) { if (host.parentNode) host.parentNode.removeChild(host); throw e; }
    try { v.resize(); } catch (_) {}
    return {host: host, viewer: v, dispose: function () {
      try { v.dispose(); } catch (_) {}
      if (host.parentNode) host.parentNode.removeChild(host);
    }};
  }
  function withTimeout(p, ms, onTimeout) {
    var timer = null;
    return Promise.race([p, new Promise(function (resolve, reject) {
      timer = setTimeout(function () { if (onTimeout) onTimeout(); reject(new Error('超过 ' + Math.round(ms / 1000) + ' 秒没有完成')); }, ms);
    })]).then(function (v) { clearTimeout(timer); return v; }, function (e) { clearTimeout(timer); throw e; });
  }
  function arrayKeys(obj, result) {
    return Object.keys(obj || {}).map(function (k) { return obj[k]; }).filter(function (k) { return typeof k === 'string' && k && result.declared(k) && !result.has(k); });
  }

  // One result → its files.  hold: the offscreen viewer and what the provider reads; o = {settings, style, link}.
  function renderJob(hold, job, o) {
    var A = ns.api;
    return Promise.resolve(ns.data.loadResult(ns.data.createOnlineSource(job.id, {fetch: A && A.dataFetch}), {})).then(function (result) {
      var m = result && result.manifest;
      if (!m || !m.job || m.job.id !== job.id) throw new Error('结果数据与所选任务不一致');
      var fam = result.family === 'volume' ? 'volume' : 'wall';
      // the link's field, window and layers belong to its own family; its colour map, bands, units and labels to all
      var link = o.link ? (o.link.family === fam ? o.link : {family: o.link.family, field: null, window: null, d: o.link.d, L: null}) : null;
      var pick = choose(m, o.settings[fam], link);
      if (!pick.field) throw new Error('没有可显示的字段');
      hold.result = result; hold.family = fam; hold.field = pick.field; hold.link = link;
      var v = hold.viewer, d = link && link.d ? link.d : {};
      var P = ns.store && ns.store.prefs ? ns.store.prefs() : {};
      var cmap = typeof d.cmap === 'string' ? d.cmap : P.cmap;
      var labels = Object.assign({}, P.labels || {}, d.labels || {});
      var layers = Object.assign({}, link && link.L ? link.L : {});
      var need = [];
      if (layers.streamlines && m.geometry && m.geometry.streamlines) need = need.concat(arrayKeys(m.geometry.streamlines, result));
      if (layersFor(hold).stagnation && ns.display && ns.display.hasStagnation && ns.display.hasStagnation(m)) {
        ['tawss', 'osi'].forEach(function (id) { var f = result.field(id); if (f && f.arrays && f.arrays.display && result.declared(f.arrays.display) && !result.has(f.arrays.display)) need.push(f.arrays.display); });
      }
      if (o.style.labels && ns.probe && ns.probe.requiredArrays) need = need.concat(ns.probe.requiredArrays(result).filter(function (k) { return !result.has(k); }));
      return Promise.resolve(need.length ? result.preload(need) : null).then(function () {
        return v.setResult(result);
      }).then(function () {
        try { v.resize(); } catch (_) {}
        if (v.setLighting) v.setLighting(d.light === 'soft' || d.light === 'flat' ? d.light : (P.lighting || 'soft'));
        if (Object.keys(layers).length && v.setLayers) v.setLayers(layers);
        return v.setField(pick.field, scaleSpec(pick.window, cmap));
      }).then(function () {
        if (o.style.labels && ns.annot) {
          try { ns.annot.drawPins(v, result, labels.annotations && ns.annot.itemsOf ? ns.annot.itemsOf(m) : []); } catch (_) {}
          try {
            var fm = ns.overview && ns.overview.findingsModel ? ns.overview.findingsModel(m) : {items: [], review: {}};
            var list = fm.items.filter(function (it) { return !(fm.review[it.id] && fm.review[it.id].decision === 'rejected'); });
            ns.annot.drawAuto(v, result, m, {branches: labels.branches, findings: labels.findings, maxd: labels.maxd, findingsList: list});
          } catch (_) {}
        }
        var f = (m.fields || []).filter(function (x) { return x && x.id === pick.field; })[0] || null;
        var info = function () {
          var i = null;
          try { i = v.colorbarInfo(); } catch (_) { i = null; }
          if (i && !i.windowLabel) i.windowLabel = windowLabel(f, pick.window, fam);
          return i;
        };
        var api = F().headless({jobId: job.id, manifest: m, result: result, viewer: v, field: pick.field, window: pick.window, info: info, hideName: o.style.hideName,
          windowLabel: function (ff, spec) { return windowLabel(ff, spec, fam); }});
        var frame = F().hasFrame(v), views = frame ? o.settings.views.slice() : ['current'];
        if (!frame) { pick.notes.push('没有解剖坐标架，用默认视角'); try { v.fit(); } catch (_) {} }
        var style = Object.assign({}, o.style, {views: views});
        var files = [];
        var step = o.settings.layout === 'montage' && views.length > 1 ?
          F().montage(api, Object.assign({}, style, {columns: F().options().columns})).then(function (mt) {
            return F().toBlob(mt.canvas).then(F().blobBytes).then(function (bytes) {
              files.push({name: F().fileName(api, {hideName: o.style.hideName, info: info()}, 'montage', 'png', mt.scale), bytes: bytes});
            });
          }) :
          F().viewFiles(api, style).then(function (r) { r.files.forEach(function (x) { files.push({name: x.name, bytes: x.bytes}); }); });
        return step.then(function () {
          if (o.settings.svg) {
            var svg = F().colorbarSVG(info(), {lang: o.style.lang, theme: F().themeOf(o.style.background, api)});
            if (svg) files.push({name: F().fileName(api, {hideName: o.style.hideName, info: info()}, 'colorbar', 'svg', 1), bytes: svg});
          }
          var name = o.style.hideName ? 'case' : (m.job.display_name || job.case_id || 'case');
          return {files: files, folder: safeName(name) + '_' + job.id, caseName: o.style.hideName ? null : name, field: pick.field, window: pick.window, notes: pick.notes, family: fam};
        });
      });
    });
  }

  // Run the whole batch.  jobs: finished job records; o = {settings, style, link}; hooks = {onStart(job, i, n),
  // onDone(job, i, n, row), stopped() → true to stop before the next result}.  Resolves {zip, results, count, failed, files}.
  function run(jobs, o, hooks) {
    hooks = hooks || {};
    var hold = {viewer: null, result: null, family: null, field: null, link: null};
    var screen = null, uninstall = function () {};
    try { screen = offscreen(); hold.viewer = screen.viewer; uninstall = installProvider(hold); }
    catch (e) { if (screen) screen.dispose(); return Promise.reject(e); }
    var zipFiles = [], results = [], chain = Promise.resolve(), n = jobs.length;
    jobs.forEach(function (job, i) {
      chain = chain.then(function () {
        if (hooks.stopped && hooks.stopped()) {
          var skip = {id: job.id, ok: false, message: '已停止'};
          results.push(skip);
          if (hooks.onDone) hooks.onDone(job, i, n, skip);
          return null;
        }
        if (hooks.onStart) hooks.onStart(job, i, n);
        var t0 = Date.now();
        return withTimeout(renderJob(hold, job, o), JOB_TIMEOUT_MS, function () {
          // a result that hangs: the next one gets a fresh viewer, so the late answer cannot draw into it
          uninstall(); screen.dispose(); hold.result = null;
          screen = offscreen(); hold.viewer = screen.viewer; uninstall = installProvider(hold);
        }).then(function (r) {
          r.files.forEach(function (f) { zipFiles.push({name: r.folder + '/' + f.name, bytes: f.bytes}); });
          var row = {id: job.id, ok: true, family: r.family, field: r.field, window: r.window, files: r.files.map(function (f) { return f.name; }),
            notes: r.notes, seconds: Math.round((Date.now() - t0) / 100) / 10};
          if (r.caseName) row.case = r.caseName;
          results.push(row);
          if (hooks.onDone) hooks.onDone(job, i, n, row);
        }, function (e) {
          var row = {id: job.id, ok: false, message: String(e && e.message || e)};
          results.push(row);
          if (hooks.onDone) hooks.onDone(job, i, n, row);
        });
      });
    });
    return chain.then(function () {
      uninstall(); screen.dispose(); hold.result = null;
      var manifest = {schema_version: SCHEMA, created_at: new Date().toISOString(), frame_css_px: [FRAME.w, FRAME.h],
        settings: o.settings, style: o.style, link: o.link ? {field: o.link.field, window: o.link.window, display: o.link.d || null, layers: o.link.L || null} : null, results: results};
      zipFiles.push({name: 'manifest.json', bytes: JSON.stringify(manifest, null, 1)});
      return {zip: F().zipStore(zipFiles), results: results, count: results.filter(function (r) { return r.ok; }).length,
        failed: results.filter(function (r) { return !r.ok; }), files: zipFiles.length - 1};
    }, function (e) { uninstall(); if (screen) screen.dispose(); throw e; });
  }

  // ------------------------------------------------------------------ dialog
  function stamp() { var d = new Date(), p = function (x) { return x < 10 ? '0' + x : String(x); }; return d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) + '_' + p(d.getHours()) + p(d.getMinutes()); }
  function segControl(items, value, onPick, label) {
    var h = ui().h, box = h('div', {'class': 'seg exp-seg', role: 'group', 'aria-label': label});
    items.forEach(function (it) {
      var on = String(it.value) === String(value);
      var b = h('button', {type: 'button', 'class': 'seg-btn' + (on ? ' on' : ''), 'aria-pressed': String(on), text: it.label, title: it.title || null});
      b.addEventListener('click', function () {
        for (var i = 0; i < box.children.length; i++) { box.children[i].classList.remove('on'); box.children[i].setAttribute('aria-pressed', 'false'); }
        b.classList.add('on'); b.setAttribute('aria-pressed', 'true');
        onPick(it.value);
      });
      box.appendChild(b);
    });
    return box;
  }
  function pill(label, on, onToggle, title) {
    var b = ui().h('button', {type: 'button', 'class': 'pill exp-pill' + (on ? ' on' : ''), 'aria-pressed': String(Boolean(on)), text: label, title: title || null});
    b.addEventListener('click', function () { var next = !b.classList.contains('on'); b.classList.toggle('on', next); b.setAttribute('aria-pressed', String(next)); onToggle(next); });
    return b;
  }
  function row(label, control, tip) {
    var h = ui().h;
    return h('div', {'class': 'exp-row'}, h('span', {'class': 'exp-k', text: label}), control, tip ? ui().infoTip(tip) : null);
  }
  function linkSummary(lp) {
    if (!lp) return '';
    var d = lp.d || {};
    if (!lp.family) return '链接的字段这里不认识，只用它的色表、分段、阈值、单位和标签。';
    var parts = [FAMILY_TEXT[lp.family], fieldLabel(lp.field)], w = lp.window;
    parts.push(w && typeof w === 'object' && Array.isArray(w.range) ? '固定 ' + ui().range(w.range[0], w.range[1], UNITS[lp.field]) : w === 'adaptive-linear' ? '自适应（线性）' : w && w !== 'adaptive' ? '色标窗 ' + w : '本例自适应');
    if (d.bands && d.bands[lp.field]) parts.push(d.bands[lp.field] + ' 段');
    if (d.cmap) parts.push(d.cmap);
    return parts.join(' · ');
  }

  var running = null;   // one batch at a time
  function stopRunning() { if (running) running.stop(); }
  function open(jobIds, shellApi) {
    var h = ui().h, S = shellApi || null;
    if (!F() || typeof F().headless !== 'function' || !ns.data || !ns.api) { ui().toast('批量出图需要在线工作区。', {kind: 'info'}); return null; }
    if (running) { ui().toast('上一批还在出图。', {kind: 'info'}); return null; }
    var ids = (Array.isArray(jobIds) ? jobIds : [jobIds]).filter(function (x, i, a) { return typeof x === 'string' && x && a.indexOf(x) === i; });
    var known = {};
    ((S && S.jobs ? S.jobs() : null) || (ns.store && ns.store.state && ns.store.state.jobs) || []).forEach(function (j) { if (j && j.id) known[j.id] = j; });
    var missing = ids.filter(function (id) { return !known[id]; });
    var body = h('div', {'class': 'wsb'}, h('p', {'class': 'muted', text: '正在读取任务…'}));
    var foot = h('div', {'class': 'wsb-foot'}, h('span', {'class': 'sec-fill'}), ui().button('关闭', function () { ui().dialog.close('cancel'); }));
    var dlg = ui().dialog.open({title: '批量出图', wide: true, cls: 'dlg-export dlg-batch', body: [body], actions: [foot], onClose: stopRunning});
    Promise.all(missing.map(function (id) {
      return ns.api.job(id).then(function (r) { known[id] = r.job || r; }, function (e) { known[id] = {id: id, status: 'missing', _error: '读不到（' + (e && e.message || e) + '）'}; });
    })).then(function () {
      if (!body.parentNode) return;   // closed meanwhile
      build(body, foot, ids.map(function (id) { return known[id]; }), S);
    });
    return dlg;
  }
  function build(body, foot, jobs, S) {
    var h = ui().h;
    var p = plan(jobs), s = settings(), link = null, stop = false;
    var fstate = {hideName: Boolean(S && S.hideName && S.hideName())};
    var fctx = {hideName: false, state: fstate};
    var cards = S && S.cards ? S.cards() : null;
    var head = h('div', {'class': 'wsb-head'},
      h('b', {text: p.selected.length + ' 个结果'}),
      p.families.wall ? h('span', {'class': 'muted', text: '壁面 ' + p.families.wall}) : null,
      p.families.volume ? h('span', {'class': 'muted', text: '体场 ' + p.families.volume}) : null,
      p.skipped.length ? h('span', {'class': 'wsb-skip', text: '跳过 ' + p.skipped.length, title: p.skipped.map(function (x) { return x.name + '：' + x.message; }).join('\n')}) : null,
      ui().infoTip('每个结果在后台单独渲染（同样 ' + FRAME.w + ' × ' + FRAME.h + ' 画幅），按下面的设置出图，全部打成一个 zip；zip 里的 manifest.json 记下设置和每例结果。'));
    var famRows = h('div', {'class': 'wsb-fams'});
    var linkIn = h('input', {type: 'text', 'class': 'wsb-link', placeholder: '粘贴复现链接（可选）', 'aria-label': '复现链接', autocomplete: 'off', spellcheck: 'false'});
    var linkNote = h('span', {'class': 'muted wsb-link-note'});
    function drawFamilies() {
      var rows = [];
      ['wall', 'volume'].forEach(function (fam) {
        if (!p.families[fam]) return;
        var x = s[fam];
        var opts = [{value: 'auto', label: '各例默认'}].concat(FIELDS[fam].filter(function (f) { return !f.cycle || p.cycle || x.field === f.id; }).map(function (f) { return {value: f.id, label: f.label}; }));
        var fieldSel = ui().select(opts, x.field, function (v) { x.field = v; if (v === 'auto' && x.scale === 'fixed') x.scale = 'adaptive'; saveSettings(s); drawFamilies(); }, {'aria-label': FAMILY_TEXT[fam] + '字段', 'class': 'wsb-sel'});
        var sc = [{value: 'adaptive', label: '本例自适应'}, {value: 'adaptive-linear', label: '自适应（线性）'}, {value: 'fixed', label: '固定范围', disabled: x.field === 'auto'}];
        if (link && link.family === fam && typeof link.window === 'string' && ['adaptive', 'adaptive-linear'].indexOf(link.window) < 0) sc.push({value: 'link', label: '链接的色标窗'});
        if (x.scale === 'link' && !sc.some(function (o) { return o.value === 'link'; })) x.scale = 'adaptive';
        var scaleSel = ui().select(sc, x.scale, function (v) { x.scale = v; saveSettings(s); drawFamilies(); }, {'aria-label': FAMILY_TEXT[fam] + '色标', 'class': 'wsb-sel'});
        var range = null;
        if (x.scale === 'fixed') {
          var mk = function (key, label) {
            var inp = h('input', {type: 'number', 'class': 'wsd-num wsb-num', step: 'any', value: x[key] === null ? '' : String(x[key]), 'aria-label': FAMILY_TEXT[fam] + label});
            inp.addEventListener('change', function () { x[key] = num(inp.value); saveSettings(s); });
            return inp;
          };
          var u = ui().unitText(UNITS[x.field]);
          range = h('span', {'class': 'wsb-range'}, mk('lo', '下限'), h('span', {text: '–'}), mk('hi', '上限'), u ? h('span', {'class': 'muted', text: u}) : null);
        }
        rows.push(row(FAMILY_TEXT[fam], h('span', {'class': 'wsb-fam'}, fieldSel, scaleSel, range),
          fam === 'wall' ? '「各例默认」：峰值 WSS 结果用 WSS，周期指标用 TAWSS。固定范围让各例同一颜色表示同一数值。分段和阈值按各字段的「设为默认」。' :
            '「各例默认」：速度。固定范围按基本单位（Pa、m/s）填；色条按你选的显示单位。'));
      });
      ui().fill(famRows, rows);
    }
    function useLink(lp, prefix) {
      link = lp;
      if (link) applyLinkToSettings(s, link);
      linkNote.classList.remove('wsb-bad');
      linkNote.textContent = link ? prefix + linkSummary(link) : '';
      drawFamilies();
    }
    function setLink(text) {
      var r = parseLink(text);
      if (r.error) { link = null; linkNote.textContent = r.error; linkNote.classList.add('wsb-bad'); drawFamilies(); return; }
      useLink(linkPlan(r.state), '按链接：');
    }
    linkIn.addEventListener('change', function () { setLink(linkIn.value); });
    linkIn.addEventListener('input', function () { if (!linkIn.value.trim()) setLink(''); });
    var curBtn = null;
    if (S && S.cur && S.cur() && S.cur().result && F().viewState) {
      curBtn = ui().button('用当前显示', function () {
        var st = null;
        try { st = F().viewState(S); } catch (_) { st = null; }
        if (!st) return;
        linkIn.value = '';
        useLink(linkPlan(st), '按当前显示：');
      }, {kind: 'link', cls: 'btn-sm', title: '取当前打开的结果的字段、色标、分段、阈值、色表、单位和标签'});
    }
    var viewPills = VIEWS.map(function (v) {
      return pill(VIEW_SHORT[v], s.views.indexOf(v) >= 0, function (on) {
        s.views = VIEWS.filter(function (x) { return x === v ? on : s.views.indexOf(x) >= 0; });
        saveSettings(s);
      }, F().VIEW_ZH[v]);
    });
    var six = ui().button('六视角', function () {
      s.views = VIEWS.slice(); saveSettings(s);
      viewPills.forEach(function (b) { b.classList.add('on'); b.setAttribute('aria-pressed', 'true'); });
    }, {kind: 'link', cls: 'btn-sm', title: '前、后、左、右、头、足'});
    var layout = segControl([{value: 'files', label: '每个视角一张'}, {value: 'montage', label: '每例一张拼图'}], s.layout, function (v) { s.layout = v; saveSettings(s); }, '输出');
    var svg = pill('色标 SVG', s.svg, function (on) { s.svg = on; saveSettings(s); }, '每例另存一份色条 SVG');
    var styleBox = h('div', {'class': 'wsb-style'});
    var drawStyle = function () { ui().fill(styleBox, F().styleRows(fctx, drawStyle)); };
    drawStyle();
    var progress = h('progress', {'class': 'wsb-progress', max: String(Math.max(1, p.selected.length)), value: '0'});
    progress.hidden = true;
    var log = h('ol', {'class': 'wsb-log'});
    var status = h('p', {'class': 'exp-status muted wsb-status'});
    var lastZip = null, lastName = '';
    var again = ui().button('再次下载', function () { if (lastZip) ui().downloadBlob(new root.Blob([lastZip], {type: 'application/zip'}), lastName); }, {kind: 'link', cls: 'btn-sm'});
    again.hidden = true;
    var start = ui().button('开始出图', go, {icon: 'download', kind: 'primary', disabled: !p.selected.length});
    var stopBtn = ui().button('停止', function () { stop = true; stopBtn.disabled = true; status.textContent = '这一例完成后停止…'; }, {cls: 'btn-sm'});
    stopBtn.hidden = true;
    var skippedItems = function () { return p.skipped.map(function (x) { return h('li', {'class': 'wsb-li is-skip', text: x.name + '：跳过 · ' + x.message}); }); };
    ui().fill(log, skippedItems());
    drawFamilies();
    var form = h('div', {'class': 'wsb-form'},
      h('h4', {'class': 'wsb-sub', text: '显示'}), famRows,
      row('链接', h('span', {'class': 'wsb-linkrow'}, linkIn, curBtn), '粘贴一个结果的复现链接：用它的字段、色标窗、分段、阈值、色表、单位、标签和图层；不用它的视角（每例几何不同，按下面的标准视角出图）。'),
      h('div', {'class': 'wsb-link-line'}, linkNote),
      h('h4', {'class': 'wsb-sub', text: '视角'}),
      row('视角', h('div', {'class': 'exp-pills'}, viewPills, six), '标准视角按每例自己的解剖坐标架对准并撑满画幅；没有坐标架的结果用默认视角出一张。'),
      row('输出', h('span', {'class': 'wsb-fam'}, layout, svg), '每例一张拼图：选中的视角拼在一起，共用一条色条（列数同「拼图」页）。'),
      h('h4', {'class': 'wsb-sub', text: '样式'}), styleBox);
    ui().fill(body, head, form, h('div', {'class': 'wsb-run'}, progress, status, log));
    ui().fill(foot, again, h('span', {'class': 'sec-fill'}), stopBtn, ui().button('关闭', function () { ui().dialog.close('cancel'); }), start);
    function go() {
      var errs = checkSettings(s, p.families);
      if (errs.length) { status.textContent = errs.join('；') + '。'; status.classList.add('wsb-bad'); return; }
      status.classList.remove('wsb-bad');
      var opts = F().options();
      var style = {scale: opts.scale, background: opts.background, lang: opts.lang, colorbar: opts.colorbar, title: opts.title, labels: opts.labels, crop: opts.crop, hideName: Boolean(fstate.hideName)};
      stop = false;
      start.disabled = true; stopBtn.hidden = false; stopBtn.disabled = false; again.hidden = true;
      form.classList.add('is-running');
      progress.hidden = false; progress.value = 0;
      ui().fill(log, skippedItems());
      var items = {}, t0 = Date.now();
      running = {stop: function () { stop = true; }};
      var nameOf = function (job, i) { return style.hideName ? '病例 ' + (i + 1) : ui().displayName(job); };
      run(p.selected, {settings: JSON.parse(JSON.stringify(s)), style: style, link: link}, {
        stopped: function () { return stop; },
        onStart: function (job, i, n) {
          items[job.id] = h('li', {'class': 'wsb-li is-busy', text: nameOf(job, i) + ' · ' + ui().resultName(job, cards, {short: true}) + '：渲染中…'});
          log.appendChild(items[job.id]);
          status.textContent = '第 ' + (i + 1) + ' / ' + n + ' 例…';
        },
        onDone: function (job, i, n, r) {
          var li = items[job.id];
          if (!li) { li = h('li'); log.appendChild(li); }
          li.className = 'wsb-li ' + (r.ok ? 'is-ok' : r.message === '已停止' ? 'is-skip' : 'is-bad');
          li.textContent = nameOf(job, i) + ' · ' + ui().resultName(job, cards, {short: true}) + '：' +
            (r.ok ? '已导出 ' + r.files.length + ' 个文件' + (r.notes && r.notes.length ? '（' + r.notes.join('；') + '）' : '') : r.message === '已停止' ? '未开始（已停止）' : '失败 · ' + r.message);
          progress.value = i + 1;
        }
      }).then(function (res) {
        running = null;
        if (!body.parentNode) { ui().toast('批量出图已停止' + (res.count ? '（已完成 ' + res.count + ' 例，没有下载）' : '') + '。', {kind: 'info'}); return; }
        if (res.count) {
          lastZip = res.zip; lastName = 'wss_batch_figures_' + stamp() + '.zip';
          ui().downloadBlob(new root.Blob([res.zip], {type: 'application/zip'}), lastName);
          again.hidden = false;
        }
        var failed = res.failed.filter(function (r) { return r.message !== '已停止'; }).length, stopped = res.failed.length - failed;
        status.textContent = (stopped ? '已停止：' : '完成：') + res.count + ' 例 · ' + res.files + ' 个文件' + (failed ? ' · ' + failed + ' 例失败' : '') + (stopped ? ' · ' + stopped + ' 例未开始' : '') +
          (p.skipped.length ? ' · 跳过 ' + p.skipped.length : '') + ' · ' + ui().duration((Date.now() - t0) / 1000) + (res.count ? '' : '；没有可下载的文件');
        status.classList.toggle('wsb-bad', !res.count);
      }, function (e) {
        running = null;
        status.textContent = '没有开始：' + (e && e.message || e);
        status.classList.add('wsb-bad');
      }).then(function () {
        start.disabled = false; stopBtn.hidden = true; form.classList.remove('is-running');
      });
    }
    return {go: go, body: body};
  }

  return {open: open, run: run, plan: plan, sanitize: sanitize, settings: settings, saveSettings: saveSettings, checkSettings: checkSettings,
    parseLink: parseLink, linkPlan: linkPlan, applyLinkToSettings: applyLinkToSettings, choose: choose, windowLabel: windowLabel, scaleSpec: scaleSpec,
    installProvider: installProvider, bandsFor: bandsFor, thresholdsFor: thresholdsFor, unitFor: unitFor, layersFor: layersFor, safeName: safeName, familyOf: familyOf,
    stop: stopRunning, isRunning: function () { return Boolean(running); }, FRAME: FRAME, FIELDS: FIELDS, SCHEMA: SCHEMA};
});
