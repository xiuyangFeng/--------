/* WSS workspace v2 — input page (contract §6.3 ws_input.js): progress, unit check, outlet confirmation in the
 * main viewport (O4, same /confirm payload and swap semantics as the classic workbench), and the failure page
 * with every opening drawn and numbered in 3-D (U2).  Hints only; nothing is repaired automatically. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.input = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var ui = function () { return ns.ui; };
  var api = function () { return ns.api; };

  var NAMES = {inlet: '主动脉入口', 'out-le': '左髂外', 'out-li': '左髂内', 'out-re': '右髂外', 'out-ri': '右髂内'};
  var COLORS = {inlet: '#2569aa', 'out-le': '#0e6b5c', 'out-li': '#6f9f3c', 'out-re': '#c07a2c', 'out-ri': '#8f5ca3', none: '#5f6b76', small: '#9c3326'};
  var SWAPS = {
    lr: {'out-le': 'out-re', 'out-li': 'out-ri', 'out-re': 'out-le', 'out-ri': 'out-li'},
    left: {'out-le': 'out-li', 'out-li': 'out-le'},
    right: {'out-re': 'out-ri', 'out-ri': 'out-re'}
  };
  function swap(mapping, kind) {
    var pairs = SWAPS[kind] || {};
    var out = {};
    Object.keys(mapping || {}).forEach(function (k) { out[k] = pairs[mapping[k]] || mapping[k]; });
    return out;
  }
  function validMapping(mapping, outletIds) {
    var names = outletIds.map(function (id) { return mapping[id]; });
    var uniq = {};
    names.forEach(function (n) { uniq[n] = true; });
    return names.length === 4 && Object.keys(uniq).length === 4 && names.every(function (n) { return n && n !== 'inlet' && NAMES[n]; });
  }
  // Openings from any source → [{id, center, radius}]
  function normalizeOpenings(list) {
    return (Array.isArray(list) ? list : []).map(function (o, i) {
      if (!o) return null;
      var id = o.opening_id !== undefined ? o.opening_id : o.opening_index !== undefined ? o.opening_index : o.id !== undefined ? o.id : i;
      var r = o.radius_mm !== undefined ? o.radius_mm : o.equivalent_radius_mm !== undefined ? o.equivalent_radius_mm : o.r_mm;
      var c = o.center_mm || o.center || o.centroid_mm || null;
      return {id: id, radius: Number(r), center: c, role: o.role || null};
    }).filter(function (o) { return o && Array.isArray(o.center) && o.center.length >= 3; });
  }
  // U2: flag openings that are clearly smaller than the rest; with more than five openings the extra smallest ones.
  function openingHints(openings) {
    var list = normalizeOpenings(openings);
    var radii = list.map(function (o) { return o.radius; }).filter(function (r) { return isFinite(r); }).sort(function (a, b) { return a - b; });
    var median = radii.length ? radii[Math.floor(radii.length / 2)] : null;
    var extra = Math.max(0, list.length - 5);
    var smallest = list.slice().filter(function (o) { return isFinite(o.radius); }).sort(function (a, b) { return a.radius - b.radius; }).slice(0, extra).map(function (o) { return o.id; });
    return list.map(function (o) {
      var small = isFinite(o.radius) && (o.radius < 1 || (median && o.radius < 0.35 * median) || smallest.indexOf(o.id) >= 0);
      return {id: o.id, radius: o.radius, center: o.center, small: Boolean(small)};
    });
  }
  function b64ToTyped(b64, Ctor) {
    if (!b64 || typeof b64 !== 'string' || typeof root.atob !== 'function') return null;
    var bin = root.atob(b64);
    var bytes = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    if (bytes.length % Ctor.BYTES_PER_ELEMENT) return null;
    return new Ctor(bytes.buffer);
  }
  function meshFrom(inputcheck, geometry) {
    if (inputcheck && inputcheck.mesh && inputcheck.mesh.vertices) {
      var v = b64ToTyped(inputcheck.mesh.vertices, Float32Array), f = b64ToTyped(inputcheck.mesh.faces, Uint32Array);
      if (v && f) return {vertices: v, faces: f};
    }
    var p = geometry && geometry.preview;
    if (p && Array.isArray(p.vertices) && Array.isArray(p.faces) && p.vertices.length) {
      var vv = new Float32Array(p.vertices.length * 3), ff = new Uint32Array(p.faces.length * 3);
      p.vertices.forEach(function (x, i) { vv[3 * i] = x[0]; vv[3 * i + 1] = x[1]; vv[3 * i + 2] = x[2]; });
      p.faces.forEach(function (x, i) { ff[3 * i] = x[0]; ff[3 * i + 1] = x[1]; ff[3 * i + 2] = x[2]; });
      return {vertices: vv, faces: ff};
    }
    return null;
  }

  // ------------------------------------------------------------------ small 3-D view: mesh + numbered markers
  function meshView(container) {
    var T = root.THREE;
    if (!T || !T.WebGLRenderer || !container) return null;
    var renderer;
    try { renderer = new T.WebGLRenderer({antialias: true, preserveDrawingBuffer: true}); } catch (_) { return null; }
    var disposables = [];
    renderer.setPixelRatio(Math.min(root.devicePixelRatio || 1, 2));
    renderer.setClearColor(0xeceef0, 1);
    var canvas = renderer.domElement;
    canvas.className = 'mv-canvas';
    container.appendChild(canvas);
    var scene = new T.Scene();
    var camera = new T.PerspectiveCamera(35, 1, 0.1, 10000);
    scene.add(new T.AmbientLight(0xffffff, 0.55));
    var key = new T.DirectionalLight(0xffffff, 0.6);
    camera.add(key); key.position.set(0.3, 0.6, 1);
    scene.add(camera);
    var group = new T.Group(); scene.add(group);
    var markerGroup = new T.Group(); scene.add(markerGroup);
    var controls = T.OrbitControls ? new T.OrbitControls(camera, canvas) : null;
    var radius = 100, frame = 0;
    function draw() { frame = 0; renderer.render(scene, camera); }
    function request() { if (!frame) frame = (root.requestAnimationFrame || setTimeout)(draw); }
    if (controls) { controls.enableDamping = false; controls.addEventListener('change', request); }
    function size() {
      var w = container.clientWidth || 600, h = container.clientHeight || 400;
      renderer.setSize(w, h, false); canvas.style.width = '100%'; canvas.style.height = '100%';
      camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix(); request();
    }
    var ro = typeof root.ResizeObserver === 'function' ? new root.ResizeObserver(size) : null;
    if (ro) ro.observe(container); else root.addEventListener('resize', size);
    function setMesh(mesh) {
      while (group.children.length) group.remove(group.children[0]);
      if (!mesh) { request(); return; }
      var g = new T.BufferGeometry();
      g.setAttribute('position', new T.BufferAttribute(mesh.vertices, 3));
      g.setIndex(new T.BufferAttribute(mesh.faces, 1));
      g.computeVertexNormals(); g.computeBoundingSphere();
      var mat = new T.MeshLambertMaterial({color: 0xd3d8dd, side: T.DoubleSide, transparent: true, opacity: 0.92});
      disposables.push(g, mat);
      group.add(new T.Mesh(g, mat));
      var s = g.boundingSphere;
      radius = s.radius || 100;
      if (controls) controls.target.copy(s.center);
      var vfov = camera.fov * Math.PI / 180, hfov = 2 * Math.atan(Math.tan(vfov / 2) * (camera.aspect || 1));
      var dist = radius / Math.sin(Math.min(vfov, hfov) / 2) * 1.02;
      camera.position.set(s.center.x, s.center.y - dist, s.center.z);
      camera.up.set(0, 0, 1); camera.near = radius / 100; camera.far = radius * 20; camera.updateProjectionMatrix();
      camera.lookAt(s.center);
      if (controls) controls.update();
      request();
    }
    function pill(text, color) {
      var c = root.document.createElement('canvas');
      var ctx = c.getContext && c.getContext('2d');
      if (!ctx) return null;
      var font = '600 28px system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif';
      ctx.font = font;
      c.width = Math.ceil(ctx.measureText(text).width + 28); c.height = 44;
      ctx.font = font;
      ctx.fillStyle = '#ffffff'; ctx.strokeStyle = color; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.rect(1.5, 1.5, c.width - 3, c.height - 3); ctx.fill(); ctx.stroke();
      ctx.fillStyle = color; ctx.textBaseline = 'middle'; ctx.fillText(text, 14, c.height / 2 + 1);
      var tex = new T.CanvasTexture(c);
      var mat = new T.SpriteMaterial({map: tex, depthTest: false, depthWrite: false, transparent: true});
      disposables.push(tex, mat);
      var sp = new T.Sprite(mat);
      var hWorld = radius * 0.06;
      sp.scale.set(hWorld * c.width / c.height, hWorld, 1);
      sp.renderOrder = 5;
      return sp;
    }
    function setMarkers(markers) {
      while (markerGroup.children.length) markerGroup.remove(markerGroup.children[0]);
      (markers || []).forEach(function (m, idx) {
        if (!m || !m.center) return;
        var color = m.color || COLORS.none;
        var geo = new T.SphereGeometry(Math.max(radius * 0.012, Math.min(radius * 0.03, (m.radius || 2) * 0.6)), 16, 12);
        var mat = new T.MeshBasicMaterial({color: new T.Color(color), depthTest: false, transparent: true, opacity: m.selected ? 1 : 0.9});
        disposables.push(geo, mat);
        var ball = new T.Mesh(geo, mat); ball.position.set(m.center[0], m.center[1], m.center[2]); ball.renderOrder = 4;
        markerGroup.add(ball);
        var sp = pill(m.label, color);
        // Labels alternate above / below their point so neighbouring outlets do not cover each other.
        if (sp) { sp.position.set(m.center[0], m.center[1], m.center[2] + (idx % 2 ? -1 : 1) * radius * 0.07); if (m.selected) sp.scale.multiplyScalar(1.2); markerGroup.add(sp); }
      });
      request();
    }
    size();
    return {setMesh: setMesh, setMarkers: setMarkers, render: request, canvas: canvas,
      dispose: function () {
        if (ro) ro.disconnect(); else root.removeEventListener('resize', size);
        if (controls) controls.dispose();
        disposables.forEach(function (d) { try { d.dispose(); } catch (_) {} });
        try { renderer.dispose(); if (renderer.forceContextLoss) renderer.forceContextLoss(); } catch (_) {}
        if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
      }};
  }

  // ------------------------------------------------------------------ panels
  function etaBlock(job) {
    var h = ui().h;
    var eta = job.eta || null;
    var box = h('div', {'class': 'eta'});
    var phase = job.phase || ui().statusInfo(job.status).label;
    if (eta && Array.isArray(eta.stages) && eta.stages.length) {
      var idx = eta.stages.findIndex(function (s) { return s.state === 'running'; });
      box.appendChild(h('p', {'class': 'eta-head', text: (eta.remaining_s !== undefined && eta.remaining_s !== null ? '预计还需约 ' + ui().duration(eta.remaining_s) : phase)}));
      if (idx >= 0) box.appendChild(h('p', {'class': 'muted', text: '正在：' + eta.stages[idx].label + '（第 ' + (idx + 1) + ' / ' + eta.stages.length + ' 步）'}));
      var bar = h('div', {'class': 'eta-bar', role: 'progressbar', 'aria-label': '计算进度'});
      eta.stages.forEach(function (s) { bar.appendChild(h('span', {'class': 'eta-seg ' + (s.state || 'pending'), title: s.label})); });
      box.appendChild(bar);
    } else {
      box.appendChild(h('p', {'class': 'eta-head', text: phase}));
      if (job.detail) box.appendChild(h('p', {'class': 'muted', text: job.detail}));
    }
    if (job.queue_position) box.appendChild(h('p', {'class': 'muted', text: '队列第 ' + job.queue_position + ' 位'}));
    return box;
  }
  function checkList(ic) {
    var h = ui().h;
    var q = ic && ic.quality;
    if (!q || !Array.isArray(q.checks) || !q.checks.length) return null;
    var words = {pass: '通过', pass_with_limits: '通过，有未评估项', review: '需要复核', fail: '不通过', not_checked: '未检查'};
    var tone = {pass: 'ok', pass_with_limits: 'ok', review: 'warn', fail: 'error', not_checked: 'idle'};
    var rank = {fail: 0, review: 1, not_checked: 2, pass_with_limits: 3, pass: 4};
    var checks = q.checks.slice().sort(function (a, b) { return (rank[a.status] === undefined ? 5 : rank[a.status]) - (rank[b.status] === undefined ? 5 : rank[b.status]); });
    return h('ul', {'class': 'checks'}, checks.map(function (c) {
      return h('li', {}, ui().dot(tone[c.status] || 'idle', words[c.status] || c.status || ''), h('span', {'class': 'check-label', text: c.label || c.key || ''}), c.note ? h('span', {'class': 'muted', text: c.note}) : null);
    }));
  }

  // ctx: {job, stage, inspector, cards, onChanged(job), onUpload(), onOpen(jobId)}
  function create(ctx) {
    var h = ui().h;
    var st = {job: ctx.job, view: null, mesh: null, inputcheck: null, geometry: null, mapping: null, selected: null, disposed: false, status: null};
    var stageBox = h('div', {'class': 'input-stage'});
    var viewHost = h('div', {'class': 'input-view'});
    var stageMsg = h('div', {'class': 'input-stage-msg'});
    stageBox.appendChild(viewHost); stageBox.appendChild(stageMsg);
    ctx.stage.replaceChildren(stageBox);
    var panel = h('div', {'class': 'input-panel'});
    ctx.inspector.replaceChildren(panel);

    function ensureView() {
      if (st.view || st.disposed) return st.view;
      st.view = meshView(viewHost);
      if (!st.view) stageMsg.textContent = '这台设备的浏览器无法显示三维（WebGL 不可用）。右侧的检查结果和操作仍可使用。';
      return st.view;
    }
    function loadGeometry() {
      var job = st.job;
      var needs = /^(awaiting|failed|interrupted|cancelled)/.test(job.status || '');
      if (!needs) return Promise.resolve();
      var tasks = [];
      tasks.push(api().inputcheck(job.id).then(function (r) { st.inputcheck = r; }, function () { st.inputcheck = null; }));
      if (job.status === 'awaiting_confirmation' || job.status === 'failed' || job.status === 'interrupted') tasks.push(api().geometry(job.id).then(function (r) { st.geometry = r; }, function () { st.geometry = null; }));
      return Promise.all(tasks).then(function () {
        if (st.disposed) return;
        st.mesh = meshFrom(st.inputcheck, st.geometry);
        var v = ensureView();
        if (v) { v.setMesh(st.mesh); stageMsg.textContent = st.mesh ? '' : '没有可显示的网格。'; }
        drawMarkers();
      });
    }
    function endpoints() {
      var a = st.job.a || st.job.stage_a || {};
      var p = a.proposal || {};
      var eps = (st.geometry && st.geometry.endpoints && st.geometry.endpoints.length ? st.geometry.endpoints : p.endpoints) || [];
      return eps.filter(function (e) { return e && Array.isArray(e.center_mm); });
    }
    function drawMarkers() {
      if (!st.view) return;
      var job = st.job;
      var markers = [];
      if (job.status === 'awaiting_confirmation') {
        endpoints().forEach(function (e) {
          var sid = String(e.segment_id);
          var name = e.kind === 'inlet' ? 'inlet' : (st.mapping && st.mapping[sid]) || null;
          markers.push({center: e.center_mm, radius: e.radius_mm, color: COLORS[name] || COLORS.none, selected: st.selected === sid,
            label: '#' + sid + ' ' + (name ? NAMES[name] : '未命名')});
        });
      } else {
        var src = (st.inputcheck && st.inputcheck.openings && st.inputcheck.openings.length) ? st.inputcheck.openings
          : (st.inputcheck && st.inputcheck.input_check && st.inputcheck.input_check.quality && st.inputcheck.input_check.quality.opening_geometry)
          || (((job.a || {}).input_check || {}).quality || {}).opening_geometry || ((job.a || {}).centerline || {}).openings || [];
        openingHints(src).forEach(function (o, i) {
          markers.push({center: o.center, radius: o.radius, color: o.small ? COLORS.small : COLORS.inlet,
            label: '开口 ' + (i + 1) + ' · r ' + ui().num(o.radius, 'mm')});
        });
      }
      st.view.setMarkers(markers);
    }

    function renderPanel() {
      var job = st.job;
      var status = job.status || '';
      var head = h('div', {'class': 'input-head'}, h('div', {'class': 'insp-name', text: ui().displayName(job)}),
        h('div', {'class': 'insp-scan', text: ui().resultName(job, ctx.cards)}), h('div', {'class': 'insp-status'}, ui().statusDot(status)));
      var body = [];
      if (/^(queued|running|cancelling)/.test(status)) {
        body.push(ui().section('进度', {}, etaBlock(job)));
        body.push(h('div', {'class': 'sec-actions'}, ui().button('取消计算', function () {
          ui().confirm('取消计算', ['取消后这次计算停止，已上传的文件保留，可以之后重试。'], {confirmLabel: '取消计算', cancelLabel: '继续计算', danger: true}).then(function (ok) {
            if (ok) act(api().cancel(job.id, {version: job.version}));
          });
        }, {cls: 'btn-sm', disabled: status === 'cancelling'})));
        stageMsg.textContent = '计算完成后这里显示结果。';
      } else if (status === 'awaiting_input') {
        body.push(unitsCard(job));
      } else if (status === 'awaiting_confirmation' || status === 'awaiting_outlets') {
        body.push(outletCard(job));
      } else {
        body.push(failureCard(job));
      }
      ui().fill(panel, head, h('div', {'class': 'input-body'}, body));
    }
    function act(promise) {
      return promise.then(function (r) {
        var job = r && (r.job || r);
        if (job && job.id && ctx.onChanged) ctx.onChanged(job);
        return job;
      }, function (error) {
        if (error.status === 409) {
          ui().toast('任务状态已经变了，已读回最新状态。', {kind: 'info'});
          api().job(st.job.id).then(function (r) { if (ctx.onChanged) ctx.onChanged(r.job || r); });
        } else ui().toast(error.message, {kind: 'error'});
      });
    }
    function unitsCard(job) {
      var ic = (st.inputcheck && st.inputcheck.input_check) || (job.a || {}).input_check || {};
      var raw = ic.raw_bbox_size || ic.bbox_size_mm || null;
      var units = ui().select([{value: 'mm', label: '毫米 mm'}, {value: 'cm', label: '厘米 cm'}, {value: 'm', label: '米 m'}],
        [ic.selected_units, (job.params || {}).units].filter(function (u) { return u === 'mm' || u === 'cm' || u === 'm'; })[0] || 'mm');
      var preview = h('p', {'class': 'muted'});
      var upd = function () { var s = {mm: 1, cm: 10, m: 1000}[units.value]; preview.textContent = raw ? '换算后外包尺寸 ' + raw.map(function (x) { return ui().sig(x * s); }).join(' × ') + ' mm' : '确认后重新检查尺寸。'; ack.checked = false; go.disabled = true; };
      var ack = h('input', {type: 'checkbox'});
      var frag = (ic.components || 1) > 1 ? h('input', {type: 'checkbox'}) : null;
      var go = ui().button('确认并继续', function () { act(api().input(job.id, {units: units.value, remove_fragments: Boolean(frag && frag.checked), acknowledged: true, version: job.version})); }, {kind: 'primary', disabled: true});
      units.addEventListener('change', upd);
      ack.addEventListener('change', function () { go.disabled = !ack.checked; });
      upd();
      return h('div', {}, ui().section('核对单位与尺寸', {}, ui().note('STL 不记录单位。单位错一档，血管尺寸差 10 倍，结果不能用。请对照影像里的血管直径核对。'),
        h('label', {'class': 'fld'}, h('span', {text: 'STL 的原始单位'}), units), preview,
        frag ? h('label', {'class': 'check'}, frag, h('span', {text: '删除孤立碎片，保留主要壁面（已核对碎片不是缺失的分支）'})) : null,
        checkList(ic),
        h('label', {'class': 'check'}, ack, h('span', {text: '我已核对单位和换算后的尺寸'})),
        h('div', {'class': 'sec-actions'}, go)));
    }
    function outletCard(job) {
      var a = job.a || job.stage_a || {};
      var p = a.proposal || {};
      if (!st.mapping) st.mapping = Object.assign({}, job.mapping || p.mapping || {});
      var eps = endpoints();
      var outlets = eps.filter(function (e) { return e.kind !== 'inlet'; }).map(function (e) { return String(e.segment_id); });
      var ack = h('input', {type: 'checkbox'});
      var go = ui().button('确认出口并开始预测', function () {
        act(api().confirm(job.id, {mapping: Object.assign({}, st.mapping), acknowledged: true, version: job.version}));
      }, {kind: 'primary', disabled: true});
      var ruleNote = ui().note('四个出口名称各用一次，同一侧的两个出口属于同侧。', 'warn');
      function refresh() {
        var ok = validMapping(st.mapping, outlets);
        go.disabled = !ack.checked || !ok; ruleNote.hidden = ok;
        drawMarkers();
      }
      var rows = eps.map(function (e) {
        var sid = String(e.segment_id);
        var nameCell;
        if (e.kind === 'inlet') nameCell = h('span', {text: NAMES.inlet});
        else {
          nameCell = ui().select([{value: '', label: '请选择'}].concat(['out-le', 'out-li', 'out-re', 'out-ri'].map(function (k) { return {value: k, label: NAMES[k]}; })),
            st.mapping[sid] || '', function (v) { st.mapping[sid] = v; ack.checked = false; st.selected = sid; refresh(); }, {'aria-label': '出口 ' + sid + ' 的名称'});
        }
        return {sid: sid, kind: e.kind, radius: e.radius_mm, name: nameCell};
      });
      var tbl = ui().table([
        {key: 'sid', label: '端点', render: function (r) { var b = h('button', {type: 'button', 'class': 'val-link', text: '#' + r.sid + (r.kind === 'inlet' ? ' 入口' : '')}); b.addEventListener('click', function () { st.selected = r.sid; drawMarkers(); }); return b; }},
        {key: 'radius', label: '半径', num: true, render: function (r) { return ui().num(r.radius, 'mm'); }},
        {key: 'name', label: '名称', render: function (r) { return r.name; }}], rows, {cls: 'tbl-outlets'});
      var sw = function (label, kind, title) { var b = ui().button(label, function () { st.mapping = swap(st.mapping, kind); ack.checked = false; renderPanel(); refresh(); }, {cls: 'btn-sm', title: title}); return b; };
      var reasons = (p.confidence_reasons || p.flags || []).filter(Boolean);
      var conf = typeof p.confidence === 'number' ? '自动命名把握度 ' + Math.round(p.confidence * 100) + '%' : '';
      ack.addEventListener('change', refresh);
      setTimeout(refresh, 0);
      return h('div', {}, ui().section('确认出口', {tag: conf ? h('span', {'class': 'sec-note', text: conf}) : null},
        ui().note('STL 不带患者方向，出口名称决定左右。名称错了，左右分支的结果会对调。请对照原始影像核对。'),
        reasons.length ? h('ul', {'class': 'reasons'}, reasons.slice(0, 4).map(function (t) { return h('li', {text: t}); })) : null,
        tbl,
        h('div', {'class': 'swap-row'}, sw('左右互换', 'lr', '左右两侧整体互换'), sw('左侧内 / 外', 'left', '左侧髂内与髂外互换'), sw('右侧内 / 外', 'right', '右侧髂内与髂外互换'),
          ui().button('恢复建议', function () { st.mapping = Object.assign({}, p.mapping || {}); ack.checked = false; renderPanel(); refresh(); }, {kind: 'link', cls: 'btn-sm'})),
        ruleNote,
        h('label', {'class': 'check'}, ack, h('span', {text: '我已结合原始影像核对入口和四个出口的名称'})),
        h('div', {'class': 'sec-actions'}, go)));
    }
    function failureCard(job) {
      var ic = (st.inputcheck && st.inputcheck.input_check) || (job.a || {}).input_check || null;
      var err = (st.inputcheck && st.inputcheck.error) || job.error;
      var message = err && (err.message || err) || job.detail || '计算没有完成。';
      var hints = openingHints((st.inputcheck && st.inputcheck.openings && st.inputcheck.openings.length ? st.inputcheck.openings : null)
        || (ic && ic.quality && ic.quality.opening_geometry) || ((job.a || {}).centerline || {}).openings || []);
      var small = hints.filter(function (o) { return o.small; });
      var parts = [ui().note(typeof message === 'string' ? message : '计算没有完成。', 'error')];
      if (hints.length) {
        parts.push(ui().table([{key: 'i', label: '开口', render: function (r) { return String(r.i); }}, {key: 'r', label: '等效半径', num: true, render: function (r) { return ui().num(r.radius, 'mm'); }},
          {key: 'n', label: '', blank: true, render: function (r) { return r.small ? h('span', {'class': 'warn-text', text: '可能是没封闭的分支残端'}) : ''; }}],
          hints.map(function (o, i) { return Object.assign({i: i + 1}, o); }), {cls: 'tbl-openings'}));
        parts.push(ui().note('需要恰好五个开口：主动脉入口和四个髂动脉出口。' + (hints.length !== 5 ? '这份输入有 ' + hints.length + ' 个。' : '') + (small.length ? '红色标记的开口明显偏小，请在分割软件里封闭后重新导出。' : '')));
      }
      parts.push(checkList(ic));
      var actions = [];
      if (job.status === 'interrupted' || job.status === 'cancelled') actions.push(ui().button('重试', function () { act(api().retry(job.id, {version: job.version})); }, {cls: 'btn-sm'}));
      if (ctx.onUpload) actions.push(ui().button('修改后重新上传', ctx.onUpload, {kind: 'primary', cls: 'btn-sm'}));
      parts.push(h('div', {'class': 'sec-actions'}, actions, ui().link('报错对照表', '/static/v2/help_errors.html', {newTab: true}), ui().link('输入说明', '/static/v2/help_input.html', {newTab: true})));
      return ui().section(ui().statusInfo(job.status).label, {}, parts);
    }

    renderPanel();
    loadGeometry().then(renderPanel);
    return {
      update: function (job) {
        var changed = !st.job || job.status !== st.job.status || job.version !== st.job.version;
        st.job = job;
        if (!changed) return;
        if (job.status !== 'awaiting_confirmation') st.mapping = null;
        renderPanel(); loadGeometry().then(renderPanel);
      },
      dispose: function () { st.disposed = true; if (st.view) st.view.dispose(); st.view = null; },
      state: st
    };
  }

  return {create: create, swap: swap, validMapping: validMapping, openingHints: openingHints, normalizeOpenings: normalizeOpenings, meshFrom: meshFrom, NAMES: NAMES};
});
