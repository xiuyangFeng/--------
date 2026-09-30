/* WSS workspace v2 — vessel thumbnails for the case gallery.
 * One shared off-screen renderer draws the stage-A preview surface (GET /api/jobs/<id>/geometry) of a job as a lit
 * grey vessel in the anterior view (inlet up, patient left on the right of the picture) and caches the PNG data URL.
 * Requests are queued and drawn one at a time so a long gallery never blocks the page.  Display only. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.thumbs = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';
  var W = 220, H = 260;
  var cache = {}, queue = [], busy = false, renderer = null, failed = false;

  function T() { return root.THREE; }
  function ensure() {
    if (renderer || failed) return renderer;
    var THREE = T();
    if (!THREE || typeof THREE.WebGLRenderer !== 'function' || !root.document) { failed = true; return null; }
    try {
      renderer = new THREE.WebGLRenderer({antialias: true, alpha: true, preserveDrawingBuffer: true});
      renderer.setPixelRatio(Math.min(2, Math.max(1, root.devicePixelRatio || 1)));
      renderer.setSize(W, H, false);
      renderer.setClearColor(0x000000, 0);
    } catch (_) { renderer = null; failed = true; }
    return renderer;
  }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function norm(a) { var l = Math.hypot(a[0], a[1], a[2]); return l > 1e-9 ? [a[0] / l, a[1] / l, a[2] / l] : null; }
  function mean(list) { var s = [0, 0, 0]; list.forEach(function (p) { s[0] += p[0]; s[1] += p[1]; s[2] += p[2]; }); return list.length ? [s[0] / list.length, s[1] / list.length, s[2] / list.length] : null; }
  function finite3(p) { return Array.isArray(p) && p.length >= 3 && isFinite(p[0]) && isFinite(p[1]) && isFinite(p[2]); }

  // Pure: preview geometry → display basis {origin, x (patient left), y (towards the inlet), z (anterior)}.
  function basis(geo) {
    var pv = (geo && geo.preview) || {}, verts = Array.isArray(pv.vertices) ? pv.vertices : [];
    var ends = ((geo && geo.endpoints) || []).filter(function (e) { return e && finite3(e.center_mm); });
    var inlet = ends.filter(function (e) { return e.kind === 'inlet'; })[0];
    var outs = ends.filter(function (e) { return e.kind !== 'inlet'; });
    var origin = inlet ? inlet.center_mm : (verts.length ? mean(verts.slice(0, 2000)) : [0, 0, 0]);
    var down = outs.length && inlet ? norm(sub(mean(outs.map(function (e) { return e.center_mm; })), origin)) : null;
    if (!down) down = [0, 0, -1];
    var name = function (e) { return String(e.auto_name || e.name || ''); };
    var L = outs.filter(function (e) { return /^out-l/.test(name(e)); }), R = outs.filter(function (e) { return /^out-r/.test(name(e)); });
    var left = L.length && R.length ? sub(mean(L.map(function (e) { return e.center_mm; })), mean(R.map(function (e) { return e.center_mm; }))) : [1, 0, 0];
    var d = dot(left, down);
    left = norm([left[0] - d * down[0], left[1] - d * down[1], left[2] - d * down[2]]) || [1, 0, 0];
    var up = [-down[0], -down[1], -down[2]];
    var anterior = norm(cross(left, up)) || [0, -1, 0];
    return {origin: origin, x: left, y: up, z: anterior};
  }

  // Shared scene set-up and framing: positions already in display coordinates (x = patient left, y = up,
  // z = anterior); an orthographic camera in front, framing the bounding box.
  function render(pos, idx, colors) {
    var THREE = T(), r = ensure();
    if (!r) return null;
    var n = pos.length / 3, mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (var i = 0; i < n; i++) for (var k = 0; k < 3; k++) { var v = pos[3 * i + k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; }
    if (!isFinite(mn[0])) return null;
    var g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    if (colors) g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    var mat = colors ? new THREE.MeshPhongMaterial({vertexColors: true, specular: 0x262626, shininess: 28, side: THREE.DoubleSide})
      : new THREE.MeshPhongMaterial({color: 0xd6dde5, specular: 0x3c3c3c, shininess: 36, side: THREE.DoubleSide});
    var scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x4a5563, colors ? 0.7 : 0.85));
    var key = new THREE.DirectionalLight(0xffffff, colors ? 0.6 : 0.75); key.position.set(0.5, 0.7, 1.2); scene.add(key);
    var rim = new THREE.DirectionalLight(0x9fd3e2, 0.25); rim.position.set(-1, 0.2, -0.8); scene.add(rim);
    scene.add(new THREE.Mesh(g, mat));
    var cx = (mn[0] + mx[0]) / 2, cy = (mn[1] + mx[1]) / 2, w = (mx[0] - mn[0]) * 1.14 || 1, h = (mx[1] - mn[1]) * 1.1 || 1;
    var aspect = W / H;
    if (w / h > aspect) h = w / aspect; else w = h * aspect;
    // orthographic frustum bounds are in camera space: centre the camera on the box instead
    var cam = new THREE.OrthographicCamera(-w / 2, w / 2, h / 2, -h / 2, 0.1, (mx[2] - mn[2]) + 400);
    cam.position.set(cx, cy, mx[2] + 200); cam.lookAt(cx, cy, mn[2]);
    r.render(scene, cam);
    var url = null;
    try { url = r.domElement.toDataURL('image/png'); } catch (_) { url = null; }
    g.dispose(); mat.dispose();
    return url;
  }
  // Grey vessel from the stage-A preview (jobs without a result yet).
  function draw(geo) {
    var pv = geo && geo.preview;
    if (!pv || !Array.isArray(pv.vertices) || !pv.vertices.length || !Array.isArray(pv.faces) || !pv.faces.length) return null;
    var B = basis(geo), n = pv.vertices.length, pos = new Float32Array(n * 3);
    for (var i = 0; i < n; i++) {
      var p = pv.vertices[i];
      if (!finite3(p)) continue;
      var q = sub(p, B.origin);
      pos[3 * i] = dot(q, B.x); pos[3 * i + 1] = dot(q, B.y); pos[3 * i + 2] = dot(q, B.z);
    }
    var idx = [];
    pv.faces.forEach(function (f) { if (Array.isArray(f) && f.length >= 3 && f[0] < n && f[1] < n && f[2] < n) idx.push(f[0], f[1], f[2]); });
    return idx.length ? render(pos, new Uint32Array(idx), null) : null;
  }
  // Coloured vessel from a finished result: its display mesh, its main field on the viewer's own scale, turned to
  // the anterior view through manifest.frame (aligned x = patient left, z = towards the inlet, y = posterior).
  var FIELD_ORDER = ['tawss', 'wss', 'wall_pressure'];
  function drawResult(res) {
    var m = res.manifest || {}, dm = (m.geometry && m.geometry.display_mesh) || {};
    var f = null;
    for (var i = 0; i < FIELD_ORDER.length && !f; i++) f = (m.fields || []).filter(function (x) { return x && x.id === FIELD_ORDER[i] && x.arrays && x.arrays.display; })[0] || null;
    var V = res.array(dm.vertices), F = res.array(dm.faces), D = f ? res.array(f.arrays.display) : null;
    var n = Math.floor(V.length / 3), pos = new Float32Array(n * 3);
    var fr = m.frame || {}, R = fr.rotation, o = fr.origin_mm || [0, 0, 0];
    for (i = 0; i < n; i++) {
      var p = [V[3 * i] - o[0], V[3 * i + 1] - o[1], V[3 * i + 2] - o[2]];
      if (Array.isArray(R) && R.length === 3) {
        var a = [dot(R[0], p), dot(R[1], p), dot(R[2], p)];
        pos[3 * i] = a[0]; pos[3 * i + 1] = a[2]; pos[3 * i + 2] = -a[1];
      } else { pos[3 * i] = p[0]; pos[3 * i + 1] = p[2]; pos[3 * i + 2] = -p[1]; }
    }
    var colors = null;
    if (D && ns.colormap) {
      var st = f.statistics || {}, disp = f.display || {};
      var sc = ns.colormap.resolve(f, {window: 'adaptive', log: null}, {p99: disp.p99 !== undefined && disp.p99 !== null ? disp.p99 : st.p99, min: st.min, max: st.max}).scale;
      colors = new Float32Array(n * 3);
      for (i = 0; i < n; i++) { var c = sc.color(D[i]); colors[3 * i] = c[0]; colors[3 * i + 1] = c[1]; colors[3 * i + 2] = c[2]; }
    }
    return render(pos, F instanceof Uint32Array ? F : Uint32Array.from(F), colors);
  }

  function pump() {
    if (busy || !queue.length) return;
    busy = true;
    var it = queue.shift();
    if (cache[it.key]) { finish(it, cache[it.key]); return; }
    Promise.resolve().then(function () { return it.fetch(); }).then(function (got) {
      if (got && got.kind === 'result') {
        var res = got.result, m = res.manifest || {}, dm = (m.geometry && m.geometry.display_mesh) || {};
        var keys = [dm.vertices, dm.faces];
        FIELD_ORDER.some(function (id) { var f = (m.fields || []).filter(function (x) { return x && x.id === id && x.arrays && x.arrays.display; })[0]; if (f) { keys.push(f.arrays.display); return true; } return false; });
        return res.preload(keys.filter(Boolean)).then(function () { return drawResult(res); });
      }
      return draw(got);
    }).then(function (url) {
      if (url) cache[it.key] = url;
      finish(it, url || null);
    }, function () { finish(it, null); });
  }
  function finish(it, url) {
    try { it.done(url); } catch (_) {}
    busy = false;
    (root.setTimeout || function (f) { f(); })(pump, 16);
  }
  // key: a stable id (the job id of the scan); fetch(): Promise<geometry>; done(url|null)
  function request(key, fetch, done) {
    if (cache[key]) { done(cache[key]); return; }
    queue.push({key: key, fetch: fetch, done: done});
    pump();
  }
  return {request: request, draw: draw, drawResult: drawResult, basis: basis, cached: function (key) { return cache[key] || null; }, size: {width: W, height: H}};
});
