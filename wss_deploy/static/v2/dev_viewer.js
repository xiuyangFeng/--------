/* Route-4 debug page for the viewer core (contract §6.2 "调试页").  Loads only the core: ?job=<id>&field=<id>
   online, or the embedded wssv2-manifest / wssv2-arrays offline.  Extra query keys for screenshots:
   window=<id|a,b> log=1 bands=<n> cmap=<name> light=soft layers=trust,points,centerline,streamlines,-outline
   view=<anterior|left|…> cursor=<segmentId>:<s_mm> dual=<fieldId> pick=1.  Timings go to window.__devPerf. */
(function (root) {
  'use strict';
  var perf = root.__devPerf = { t_script: performance.now(), marks: {}, field_switch_ms: [], pick_ms: [], errors: [] };
  function mark(name) { perf.marks[name] = +(performance.now()).toFixed(1); }
  var ns = root.WSSV2;
  var U = ns.util;
  var doc = root.document;
  var q = new URLSearchParams(root.location.search);

  function h(tag, attrs, kids) {
    var el = doc.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === 'style') el.style.cssText = attrs[k];
      else if (k === 'text') el.textContent = attrs[k];
      else if (k.slice(0, 2) === 'on') el.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== null && attrs[k] !== undefined && attrs[k] !== false) el.setAttribute(k, attrs[k] === true ? '' : attrs[k]);
    });
    (kids || []).forEach(function (c) { if (c !== null && c !== undefined) el.appendChild(typeof c === 'string' ? doc.createTextNode(c) : c); });
    return el;
  }
  var css = [
    '.dv{display:grid;grid-template-columns:232px 1fr 300px;grid-template-rows:40px 1fr;height:100vh}',
    '.dv-top{grid-column:1/4;display:flex;align-items:center;gap:12px;padding:0 14px;border-bottom:1px solid var(--line);background:var(--panel)}',
    '.dv-top b{font-size:15px}.dv-top .nm{font-size:18px;font-weight:600}.dv-top .st{color:var(--ink-3);font-size:12px}',
    '.dv-left,.dv-right{background:var(--panel);overflow:auto;padding:10px 12px;border-right:1px solid var(--line)}.dv-right{border-right:0;border-left:1px solid var(--line)}',
    '.dv h3{font-size:12px;color:var(--ink-3);font-weight:600;margin:12px 0 4px}.dv h3:first-child{margin-top:0}',
    '.dv select,.dv input[type=number]{height:28px;border:1px solid var(--line-2);border-radius:3px;background:#fff;color:var(--ink);font:inherit;padding:0 6px;width:100%;box-sizing:border-box}',
    '.dv button{white-space:nowrap;height:28px;border:1px solid var(--line-2);border-radius:3px;background:#fff;color:var(--ink);font:inherit;padding:0 8px;cursor:pointer}',
    '.dv button.on{border-color:var(--accent);color:var(--accent);background:var(--accent-weak)}',
    '.dv label.ck{display:flex;gap:6px;align-items:center;min-height:24px}.dv .row{display:flex;gap:4px;flex-wrap:wrap}.dv .note{color:var(--ink-3);font-size:12px}',
    '.dv-main{position:relative;display:flex;min-width:0}.dv-vp{position:relative;flex:1;min-width:0;border-right:1px solid var(--line)}.dv-vp:last-child{border-right:0}',
    '.dv-cb{position:absolute;right:10px;top:40px;width:148px;height:calc(100% - 80px);max-height:460px}',
    '.dv-or{position:absolute;left:12px;bottom:12px}.dv-status{position:absolute;left:12px;top:8px;font-size:12px;color:var(--ink-2)}',
    '.dv table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}.dv td{border-bottom:1px solid var(--line);height:26px;padding:0 4px}.dv td.n{text-align:right}',
    '.dv .def{font-size:12px;color:var(--ink-2);margin-top:6px}.dv .err{color:var(--error);background:var(--error-bg);padding:6px 8px;margin:8px 0}'
  ].join('\n');

  var ui = {}, viewer = null, viewer2 = null, unlink = null, result = null, cursor = null, colorbar = null, colorbar2 = null, orient = null, cursorOn = false;

  function layout() {
    var rootEl = doc.getElementById('dev-root') || doc.body.appendChild(h('div', { id: 'dev-root' }));
    rootEl.appendChild(h('style', { text: css }));
    ui.name = h('span', { class: 'nm', text: '…' });
    ui.state = h('span', { class: 'st', text: '加载中' });
    ui.left = h('div', { class: 'dv-left' });
    ui.right = h('div', { class: 'dv-right' });
    ui.vp1 = h('div', { class: 'dv-vp', id: 'vp1' });
    ui.cb1 = h('div', { class: 'dv-cb' });
    ui.or1 = h('div', { class: 'dv-or' });
    ui.status1 = h('div', { class: 'dv-status' });
    ui.main = h('div', { class: 'dv-main' }, [ui.vp1]);
    var dv = h('div', { class: 'dv' }, [h('div', { class: 'dv-top' }, [h('b', { text: 'WSS' }), ui.name, ui.state]), ui.left, ui.main, ui.right]);
    rootEl.appendChild(dv);
  }

  function fieldDesc(id) { return result ? result.field(id) : null; }
  function currentSpec() {
    var w = ui.window.value, spec = { window: w, log: ui.log.checked, bands: +ui.bands.value || 0, cmap: ui.cmap.value };
    if (w === 'fixed') spec.window = { range: [+ui.fixLo.value, +ui.fixHi.value] };
    return spec;
  }
  function refreshColorbar() {
    if (!viewer) return;
    var info = viewer.colorbarInfo();
    colorbar.update(info);
    if (info && info.scale && info.scale.logRejected) { ui.log.checked = false; ui.logNote.textContent = info.scale.logRejected; } else ui.logNote.textContent = '';
    var st = viewer.getState();
    ui.status1.textContent = (info ? info.fullLabel : '') + ' · ' + (st.lighting === 'soft' ? '柔和光照' : '平涂') + (info && info.windowLabel ? ' · ' + info.windowLabel.text : '');
    if (viewer2) { colorbar2.update(viewer2.colorbarInfo()); }
  }
  function fillWindows() {
    var f = fieldDesc(viewer.field()), sel = ui.window, cur = sel.value;
    sel.innerHTML = '';
    sel.appendChild(h('option', { value: 'adaptive', text: '本例自适应' }));
    (f && f.windows || []).forEach(function (w) { sel.appendChild(h('option', { value: w.id, text: w.label + (w.provisional ? '（暂定）' : '') })); });
    sel.appendChild(h('option', { value: 'fixed', text: '固定范围' }));
    var st = viewer.getState().scale.window;
    sel.value = typeof st === 'object' ? 'fixed' : st;
    if (!sel.value) sel.value = cur || 'adaptive';
    ui.fixRow.style.display = sel.value === 'fixed' ? '' : 'none';
  }
  function applyField(id, spec) {
    var t0 = performance.now();
    return viewer.setField(id, spec).then(function () {
      viewer.renderNow();
      perf.field_switch_ms.push(+(performance.now() - t0).toFixed(1));
      fillWindows(); refreshColorbar(); renderCursorReadout();
    });
  }

  function controls() {
    var L = ui.left;
    L.innerHTML = '';
    L.appendChild(h('h3', { text: '字段' }));
    ui.field = h('select', { onchange: function () { applyField(ui.field.value, { window: 'adaptive', log: false, bands: +ui.bands.value || 0, cmap: ui.cmap.value }); } });
    viewer.fields().forEach(function (id) { var f = fieldDesc(id); ui.field.appendChild(h('option', { value: id, text: (f.short_label || f.label || id) + ' · ' + ({ model: '模型', derived: '派生', geometry: '几何' }[f.tier] || '') })); });
    ui.field.value = viewer.field();
    L.appendChild(ui.field);
    L.appendChild(h('h3', { text: '色标窗' }));
    ui.window = h('select', { onchange: function () { ui.fixRow.style.display = ui.window.value === 'fixed' ? '' : 'none'; if (ui.window.value !== 'fixed') applyField(viewer.field(), currentSpec()); } });
    L.appendChild(ui.window);
    ui.fixLo = h('input', { type: 'number', value: '0', step: 'any' }); ui.fixHi = h('input', { type: 'number', value: '5', step: 'any' });
    ui.fixRow = h('div', { class: 'row', style: 'margin-top:4px;flex-wrap:nowrap' }, [ui.fixLo, ui.fixHi, h('button', { text: '应用', onclick: function () { applyField(viewer.field(), currentSpec()); } })]);
    L.appendChild(ui.fixRow);
    ui.log = h('input', { type: 'checkbox', onchange: function () { applyField(viewer.field(), currentSpec()); } });
    L.appendChild(h('label', { class: 'ck' }, [ui.log, '对数色标']));
    ui.logNote = h('div', { class: 'note' });
    L.appendChild(ui.logNote);
    ui.bands = h('select', { onchange: function () { applyField(viewer.field(), currentSpec()); } });
    [0, 4, 6, 8, 10].forEach(function (n) { ui.bands.appendChild(h('option', { value: String(n), text: n ? n + ' 段' : '连续' })); });
    ui.cmap = h('select', { onchange: function () { applyField(viewer.field(), currentSpec()); } });
    ns.colormap.names().forEach(function (n) { ui.cmap.appendChild(h('option', { value: n, text: ns.colormap.label(n) })); });
    L.appendChild(h('div', { class: 'row', style: 'flex-wrap:nowrap;margin-top:4px' }, [ui.bands, ui.cmap]));
    L.appendChild(h('h3', { text: '显示' }));
    ui.light = h('button', { text: '柔和光照', onclick: function () { var s = viewer.getState().lighting === 'soft' ? 'flat' : 'soft'; viewer.setLighting(s); if (viewer2) viewer2.setLighting(s); ui.light.classList.toggle('on', s === 'soft'); refreshColorbar(); } });
    L.appendChild(ui.light);
    var layers = viewer.getState().layers, names = { outline: '轮廓线', trust: '低支撑斜纹', points: '预测点', centerline: '中心线', wall: '血管壁', interior: '体内点', streamlines: '流线' };
    var tools = viewer.supports().tools;
    Object.keys(names).forEach(function (k) {
      if (!(k in layers)) return;
      if (result.family === 'wall' && (k === 'interior' || k === 'streamlines')) return;
      if (result.family === 'volume' && k === 'points') return;
      var cb = h('input', { type: 'checkbox', 'data-layer': k, onchange: function () { var o = {}; o[k] = cb.checked; viewer.setLayers(o); if (viewer2) viewer2.setLayers(o); } });
      cb.checked = !!layers[k];
      L.appendChild(h('label', { class: 'ck' }, [cb, names[k]]));
    });
    void tools;
    L.appendChild(h('h3', { text: '标准视角' }));
    var row = h('div', { class: 'row' });
    ns.viewer.VIEW_NAMES.forEach(function (n) { row.appendChild(h('button', { text: ns.viewer.VIEW_LABELS[n], title: n, onclick: function () { viewer.standardView(n); } })); });
    L.appendChild(row);
    if (!viewer.hasAnatomicalFrame()) L.appendChild(h('div', { class: 'note', text: '本结果没有解剖坐标架：视角按世界坐标。' }));
    L.appendChild(h('h3', { text: '沿血管游标' }));
    ui.cursorBtn = h('button', { text: '游标', onclick: function () { toggleCursor(!cursorOn); } });
    ui.cursorBranch = h('select', { onchange: function () { cursor.set(+ui.cursorBranch.value, 0); } });
    cursor.branches().forEach(function (b) { ui.cursorBranch.appendChild(h('option', { value: String(b.segmentId), text: b.name + ' · ' + U.fmtSig(b.length_mm) + ' mm' })); });
    L.appendChild(h('div', { class: 'row', style: 'flex-wrap:nowrap' }, [ui.cursorBtn, ui.cursorBranch]));
    L.appendChild(h('div', { class: 'row', style: 'margin-top:4px' }, [-5, -1, 1, 5].map(function (d) { return h('button', { text: (d > 0 ? '+' : '') + d + ' mm', onclick: function () { stepCursor(d); } }); })));
    ui.cursorNote = h('div', { class: 'note' });
    L.appendChild(ui.cursorNote);
    L.appendChild(h('h3', { text: '分支显示' }));
    var br = h('div');
    result.branches().forEach(function (b) {
      var cb = h('input', { type: 'checkbox', checked: true, onchange: function () {
        var ids = Array.prototype.slice.call(br.querySelectorAll('input')).filter(function (x) { return x.checked; }).map(function (x) { return +x.value; });
        viewer.setBranchVisibility(ids.length === result.branches().length ? null : ids); refreshColorbar();
      } });
      cb.value = String(b.id);
      br.appendChild(h('label', { class: 'ck' }, [cb, b.name]));
    });
    L.appendChild(br);
    L.appendChild(h('h3', { text: '双视口' }));
    ui.dual = h('select', { onchange: function () { setDual(ui.dual.value); } });
    ui.dual.appendChild(h('option', { value: '', text: '单视口' }));
    viewer.fields().forEach(function (id) { var f = fieldDesc(id); ui.dual.appendChild(h('option', { value: id, text: '并排：' + (f.short_label || id) })); });
    L.appendChild(ui.dual);
    L.appendChild(h('h3', { text: '资源' }));
    ui.res = h('div', { class: 'note' });
    L.appendChild(h('button', { text: '连续切换 20 次', onclick: function () { stress(20); } }));
    L.appendChild(ui.res);
    fillWindows();
  }

  function renderPick(p) {
    var R = ui.pick;
    if (!p) { R.innerHTML = '<div class="note">单击血管读数（读的是预测点的原值，不是颜色）。</div>'; return; }
    var f = fieldDesc(p.fieldId) || {}, rows = [];
    rows.push(['值', U.fmtValue(p.value, f.units) + (p.valueSource === 'display' ? '（显示网格顶点值）' : '')]);
    if (p.pointIndex !== null) rows.push(['预测点', '#' + p.pointIndex + (p.distance_mm != null ? ' · 距点击处 ' + U.fmtSig(p.distance_mm) + ' mm' : '')]);
    var b = p.segmentId !== null ? result.branch(p.segmentId) : null;
    rows.push(['分支', b ? b.name : '—']);
    rows.push(['距入口弧长', U.fmtValue(p.s_from_root_mm, 'mm')]);
    Object.keys(p.values || {}).forEach(function (id) { if (id === p.fieldId) return; var d = fieldDesc(id); rows.push([d.short_label || id, U.fmtValue(p.values[id], d.units)]); });
    if (p.trust) rows.push(['低支撑', '是（位 ' + p.trust + '）']);
    R.innerHTML = '<table>' + rows.map(function (r) { return '<tr><td>' + U.escapeHtml(r[0]) + '</td><td class="n">' + U.escapeHtml(r[1]) + '</td></tr>'; }).join('') + '</table>';
  }
  function renderCursorReadout() {
    var R = ui.cursorOut;
    if (!R) return;
    if (!cursorOn || !cursor.state()) { R.innerHTML = '<div class="note">游标关闭。按 G 或左栏「游标」打开，← / → 移动 1 mm（Shift 5 mm）。</div>'; return; }
    var st = cursor.state(), rd = cursor.readout(viewer.field()), f = fieldDesc(viewer.field()) || {};
    var rows = [['分支', st.name], ['分支内弧长', U.fmtValue(st.s_mm, 'mm') + ' / ' + U.fmtValue(st.length_mm, 'mm')], ['距入口弧长', U.fmtValue(st.s_from_root_mm, 'mm')], ['内切半径', U.fmtValue(st.radius_mm, 'mm')]];
    if (rd.stats) Object.keys(rd.stats).forEach(function (k) { rows.push([({ mean: '均值', p99: 'p99', min: '最小', max: '最大', p90: 'p90', median: '中位' }[k] || k), U.fmtValue(rd.stats[k], f.units)]); });
    R.innerHTML = '<table>' + rows.map(function (r) { return '<tr><td>' + U.escapeHtml(r[0]) + '</td><td class="n">' + U.escapeHtml(r[1]) + '</td></tr>'; }).join('') + '</table><div class="def">' + U.escapeHtml(rd.definition) + '</div>';
  }
  function toggleCursor(on) {
    cursorOn = !!on;
    ui.cursorBtn.classList.toggle('on', cursorOn);
    if (cursorOn) { if (!cursor.state()) cursor.set(+ui.cursorBranch.value, 40); viewer.setCursor({ segmentId: cursor.state().segmentId, s_mm: cursor.state().s_mm }); }
    else viewer.setCursor(null);
    renderCursorReadout();
  }
  function stepCursor(d) {
    if (!cursorOn) toggleCursor(true);
    var t0 = performance.now();
    var r = cursor.step(d);
    viewer.setCursor({ segmentId: r.state.segmentId, s_mm: r.state.s_mm });
    viewer.renderNow();
    perf.cursor_ms = (perf.cursor_ms || []).concat([+(performance.now() - t0).toFixed(1)]);
    ui.cursorNote.textContent = r.atEnd ? '已到本支远端。可选子支：' + (r.children.map(function (c) { var b = result.branch(c); return b ? b.name : c; }).join('、') || '无（出口）') : r.atStart ? '已到本支起点' + (r.parent !== null ? '，上一级：' + (result.branch(r.parent) || {}).name : '') : '';
    renderCursorReadout();
  }
  function setDual(fieldId) {
    if (viewer2) { if (unlink) unlink(); unlink = null; colorbar2.dispose(); viewer2.dispose(); viewer2 = null; ui.vp2.parentNode.removeChild(ui.vp2); }
    if (!fieldId) { viewer.resize(); return Promise.resolve(); }
    ui.vp2 = h('div', { class: 'dv-vp' });
    ui.main.appendChild(ui.vp2);
    var cb2 = h('div', { class: 'dv-cb' });
    viewer2 = ns.viewer.create(ui.vp2, { kind: 'main' });
    ui.vp2.appendChild(cb2);
    colorbar2 = ns.colorbar.create(cb2);
    viewer.resize();
    return viewer2.setResult(result).then(function () { return viewer2.setField(fieldId); }).then(function () {
      viewer2.applyState({ lighting: viewer.getState().lighting });
      unlink = ns.viewer.link(viewer, viewer2);
      colorbar2.update(viewer2.colorbarInfo());
    });
  }
  function stress(n) {
    var before = viewer.stats(), i = 0, t0 = performance.now();
    function next() {
      if (i >= n) {
        var after = viewer.stats();
        ui.res.textContent = '切换 ' + n + ' 次：' + (performance.now() - t0).toFixed(0) + ' ms；几何 ' + before.geometries + '→' + after.geometries + '，纹理 ' + before.textures + '→' + after.textures +
          (performance.memory ? '；JS 堆 ' + (performance.memory.usedJSHeapSize / 1048576).toFixed(0) + ' MB' : '');
        perf.stress = { n: n, before: before, after: after, ms: +(performance.now() - t0).toFixed(0) };
        return;
      }
      i++;
      viewer.setResult(result).then(function () { viewer.renderNow(); next(); });
    }
    next();
  }

  function right() {
    ui.right.appendChild(h('h3', { text: '读数' }));
    ui.pick = h('div');
    ui.right.appendChild(ui.pick);
    ui.right.appendChild(h('h3', { text: '游标处（2 mm 分箱）' }));
    ui.cursorOut = h('div');
    ui.right.appendChild(ui.cursorOut);
    ui.right.appendChild(h('h3', { text: '计时' }));
    ui.timing = h('div', { class: 'note' });
    ui.right.appendChild(ui.timing);
    renderPick(null); renderCursorReadout();
  }
  function showTiming() {
    var m = perf.marks;
    ui.timing.innerHTML = [['manifest', m.manifest], ['首帧可操作', m.first_frame], ['换场（最近）', perf.field_switch_ms[perf.field_switch_ms.length - 1]], ['拾取（最近）', perf.pick_ms[perf.pick_ms.length - 1]]]
      .map(function (r) { return U.escapeHtml(r[0]) + '：' + (r[1] === undefined ? '—' : r[1] + ' ms'); }).join('<br>');
  }

  function start() {
    layout();
    right();
    var source;
    try {
      if (doc.getElementById('wssv2-manifest')) source = ns.data.createEmbeddedSource(doc);
      else if (q.get('job')) source = ns.data.createOnlineSource(q.get('job'));
      else throw new Error('缺少 ?job=<任务 id>，页面里也没有内嵌数据');
    } catch (err) { fail(err); return; }
    if (!ns.viewer.webglAvailable()) { fail(new Error('浏览器不支持 WebGL，无法显示三维视图')); return; }
    mark('start');
    source.manifest().then(function () { mark('manifest'); return ns.data.loadResult(source); }).then(function (r) {
      result = r;
      var m = r.manifest;
      ui.name.textContent = m.job && m.job.display_name || '病例';
      ui.state.textContent = (m.result && m.result.display_name || '') + (ns.env.offline ? ' · 离线数据' : '');
      viewer = ns.viewer.create(ui.vp1, { kind: 'main' });
      ui.vp1.appendChild(ui.cb1); ui.vp1.appendChild(ui.or1); ui.vp1.appendChild(ui.status1);
      colorbar = ns.colorbar.create(ui.cb1);
      viewer.on('error', function (e) { perf.errors.push(String(e && e.message || e)); console.error(e); });
      viewer.on('pick', function (p) { if (p) perf.pick_ms.push(+p.ms.toFixed(2)); renderPick(p); viewer.select(p ? p.pointIndex : null); if (p && cursorOn) { var s = cursor.stationFromPoint(p.xyz); if (s) { cursor.set(s.segmentId, s.s_mm); viewer.setCursor(s); renderCursorReadout(); } } showTiming(); });
      viewer.on('change', function (e) { if (e && (e.what === 'branches')) refreshColorbar(); });
      return viewer.setResult(r);
    }).then(function () {
      viewer.renderNow();
      mark('first_frame');
      cursor = ns.cursor.create(result);
      orient = ns.orientation.create(ui.or1, viewer);
      var want = q.get('field');
      var spec = { window: q.get('window') || 'adaptive', log: q.get('log') === '1', bands: +(q.get('bands') || 0), cmap: q.get('cmap') || 'rainbow' };
      if (spec.window && spec.window.indexOf(',') > 0) spec.window = { range: spec.window.split(',').map(Number) };
      var fid = want && viewer.fields().indexOf(want) >= 0 ? want : viewer.field();
      return viewer.setField(fid, spec);
    }).then(function () {
      controls();
      ui.log.checked = viewer.getState().scale.log === true;
      ui.bands.value = String(viewer.getState().scale.bands || 0);
      ui.cmap.value = viewer.getState().scale.cmap;
      if (q.get('light') === 'soft') { viewer.setLighting('soft'); ui.light.classList.add('on'); }
      var ly = q.get('layers');
      if (ly) { var o = {}; ly.split(',').forEach(function (k) { if (k[0] === '-') o[k.slice(1)] = false; else o[k] = true; }); viewer.setLayers(o); Array.prototype.forEach.call(ui.left.querySelectorAll('[data-layer]'), function (cb) { cb.checked = !!viewer.getState().layers[cb.getAttribute('data-layer')]; }); }
      if (q.get('view')) viewer.standardView(q.get('view'), { animate: false });
      var cq = q.get('cursor');
      if (cq) { var p = cq.split(':'); ui.cursorBranch.value = p[0]; cursor.set(+p[0], +p[1]); toggleCursor(true); }
      refreshColorbar();
      root.addEventListener('keydown', function (e) {
        if (e.target && /input|select|textarea/i.test(e.target.tagName)) return;
        if (e.key === 'g' || e.key === 'G') { toggleCursor(!cursorOn); e.preventDefault(); }
        else if (cursorOn && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) { stepCursor((e.key === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 5 : 1)); e.preventDefault(); }
        else if (e.key === 'l' || e.key === 'L') ui.light.click();
      });
      var dual = q.get('dual');
      return dual ? setDual(dual).then(function () { ui.dual.value = dual; }) : null;
    }).then(function () {
      mark('ready');
      showTiming();
      root.__dev = { viewer: viewer, result: result, cursor: cursor, ns: ns, viewer2: function () { return viewer2; }, applyField: applyField, stepCursor: stepCursor, toggleCursor: toggleCursor, renderPick: renderPick, refreshColorbar: refreshColorbar };
      doc.body.setAttribute('data-ready', '1');
    }).catch(fail);
  }
  function fail(err) {
    perf.errors.push(String(err && err.message || err));
    var box = doc.getElementById('dev-root') || doc.body;
    box.appendChild(h('div', { class: 'err', text: '加载失败：' + (err && err.message || err) }));
    if (typeof console !== 'undefined') console.error(err);
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', start); else start();
})(typeof globalThis !== 'undefined' ? globalThis : this);
