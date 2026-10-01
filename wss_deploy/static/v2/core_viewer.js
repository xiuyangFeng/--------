/* WSSV2 core · viewer (contract §6.2, §7; discussion D1, D3, V6, V7, A1, A8, A10, E7).
   One three.js viewport: result → family adapter (wall / volume), field colouring through ns.colormap, flat
   shading with a silhouette outline by default and optional soft lighting, picking (ray → display mesh →
   nearest prediction point; the value is read from the read array, never from the colour), anatomical
   standard views from manifest.frame, the along-vessel cursor ring, markers, selection, render on demand,
   state get/apply, camera linking between viewports and full disposal.  three.js r128 (root.THREE). */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.viewer = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var VIEWPORT_HEX = '#eceef0', INK_HEX = '#1b2430', ACCENT_HEX = '#1d5d95';
  var VIEW_NAMES = ['anterior', 'posterior', 'left', 'right', 'superior', 'inferior'];
  var VIEW_LABELS = { anterior: '前', posterior: '后', left: '左', right: '右', superior: '头', inferior: '足' };
  // Free viewport area (CSS px) for fitting: status line on top, colour bar on the right; `corner` is the
  // orientation marker at the bottom left — the fit only gives way to it when the vessel would run into it.
  var DEFAULT_INSETS = { main: { top: 64, right: 116, bottom: 24, left: 64, corner: { width: 104, height: 104 } }, thumb: { top: 4, right: 4, bottom: 4, left: 4, corner: null } };

  function U() { return ns.util; }
  function T() { var t = root.THREE; if (!t) throw new Error('three.js (THREE) is not loaded'); return t; }

  // ---------------------------------------------------------------- pure helpers (Node-testable)
  function validFrame(fr) {
    return !!(fr && Array.isArray(fr.rotation) && fr.rotation.length === 3 && fr.rotation.every(function (r) { return Array.isArray(r) && r.length === 3 && r.every(Number.isFinite); }) &&
      Array.isArray(fr.origin_mm) && fr.origin_mm.length === 3 && fr.origin_mm.every(Number.isFinite));
  }
  // Anatomical axes in world coordinates: R rows (p_aligned = R · (p − origin)); x = patient left, y = posterior,
  // z = toward the inlet (superior).
  function anatomicalAxes(frame) {
    if (!validFrame(frame)) return null;
    var R = frame.rotation;
    return { left: R[0].slice(), posterior: R[1].slice(), superior: R[2].slice() };
  }
  // Viewing direction (camera → target) and screen-up for a standard view.  Without a frame the world axes
  // stand in (flagged by the caller: no anatomical meaning).
  function viewDirection(name, frame) {
    var ax = anatomicalAxes(frame) || { left: [1, 0, 0], posterior: [0, 0, -1], superior: [0, 1, 0] };
    var neg = function (v) { return [-v[0], -v[1], -v[2]]; };
    var ex = ax.left, ey = ax.posterior, ez = ax.superior;
    switch (name) {
      case 'anterior': return { dir: ey.slice(), up: ez.slice() };        // camera in front of the patient, looking back
      case 'posterior': return { dir: neg(ey), up: ez.slice() };
      case 'left': return { dir: neg(ex), up: ez.slice() };               // camera on the patient's left
      case 'right': return { dir: ex.slice(), up: ez.slice() };
      case 'superior': return { dir: neg(ez), up: neg(ey) };              // from the head, anterior at the top
      case 'inferior': return { dir: ez.slice(), up: neg(ey) };
      default: return null;
    }
  }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function unit(v) { var n = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / n, v[1] / n, v[2] / n]; }
  // Screen basis of a camera state: right, up (orthogonal), forward.
  function cameraBasis(cam) {
    var f = unit(sub(cam.target, cam.position)), u0 = cam.up || [0, 1, 0];
    var r = unit(cross(f, u0));
    var u = cross(r, f);
    return { right: r, up: u, forward: f };
  }
  // Fit with viewport insets (CSS px): the vessel fills the free area, not the part under colour bar / panels.
  // Projects sample points with a camera (position, target, up, fov) to CSS px; true when any falls in rect.
  function anyPointIn(pts, cam, W, H, fov, rect) {
    var d = unit(sub(cam.target, cam.position)), up = unit(cam.up), r = unit(cross(d, up)), u = cross(r, d);
    var t = Math.tan(fov * Math.PI / 360), a = W / H, n = Math.floor(pts.length / 3), step = Math.max(1, Math.floor(n / 4000));
    for (var i = 0; i < n; i += step) {
      var q = [pts[3 * i] - cam.position[0], pts[3 * i + 1] - cam.position[1], pts[3 * i + 2] - cam.position[2]];
      var z = dot(q, d); if (!(z > 0)) continue;
      var x = (dot(q, r) / (z * t * a) + 1) / 2 * W, y = (1 - dot(q, u) / (z * t)) / 2 * H;
      if (x >= rect.x0 && x <= rect.x1 && y >= rect.y0 && y <= rect.y1) return true;
    }
    return false;
  }
  function insetFit(pts, o) {
    var f = insetFit1(pts, o), ins = o.insets || {}, c = ins.corner;
    if (c && c.width > 0 && c.height > 0 && (+ins.bottom || 0) < c.height) {
      var H = Math.max(1, o.height);
      if (anyPointIn(pts, f, Math.max(1, o.width), H, o.fov, { x0: 0, x1: c.width, y0: H - c.height, y1: H })) {
        var ins2 = Object.assign({}, ins, { bottom: c.height });
        f = insetFit1(pts, Object.assign({}, o, { insets: ins2 }));
        f.cornerAvoided = true;
      }
    }
    return f;
  }
  function insetFit1(pts, o) {
    var C = root.WssReportCommon;
    var W = Math.max(1, o.width), H = Math.max(1, o.height), ins = o.insets || {};
    var l = Math.max(0, +ins.left || 0), r = Math.max(0, +ins.right || 0), t = Math.max(0, +ins.top || 0), b = Math.max(0, +ins.bottom || 0);
    var We = Math.max(40, W - l - r), He = Math.max(40, H - t - b);
    if (We >= W) { l = r = 0; We = W; }
    if (He >= H) { t = b = 0; He = H; }
    var f = C.fitView(pts, { dir: o.dir, up: o.up, fov: o.fov, aspect: We / He, margin: o.margin || 1.08 });
    var scale = H / He, D = f.distance * scale;
    var d = unit(o.dir), up = unit(f.up), right = unit(cross(d, up));
    var target = f.target.slice();
    var wpp = 2 * D * Math.tan(o.fov * Math.PI / 360) / H;
    var ox = (l - r) / 2, oy = (b - t) / 2;
    var shift = [-(right[0] * ox + up[0] * oy) * wpp, -(right[1] * ox + up[1] * oy) * wpp, -(right[2] * ox + up[2] * oy) * wpp];
    target = [target[0] + shift[0], target[1] + shift[1], target[2] + shift[2]];
    return { position: [target[0] - d[0] * D, target[1] - d[1] * D, target[2] - d[2] * D], target: target, up: up, distance: D };
  }
  function cleanCamera(c) {
    if (!c || !Array.isArray(c.position) || !Array.isArray(c.target)) return null;
    var ok = function (v) { return Array.isArray(v) && v.length === 3 && v.every(function (x) { return Number.isFinite(+x); }); };
    if (!ok(c.position) || !ok(c.target)) return null;
    var out = { position: c.position.map(Number), target: c.target.map(Number), up: ok(c.up) ? c.up.map(Number) : [0, 1, 0] };
    if (Number.isFinite(+c.fov) && +c.fov > 5 && +c.fov < 120) out.fov = +c.fov;
    return out;
  }
  // Validates a stored view state against what the current result offers (A8: never guess).  Returns a clean,
  // JSON-serialisable state with only the parts that apply; unknown fields / windows / branches are dropped.
  // ctx = {fields: [ids], windows: {fieldId: [ids]}, branches: [ids], family}.
  function sanitizeState(state, ctx) {
    ctx = ctx || {};
    var out = { time_index: 0 };
    if (!state || typeof state !== 'object') return out;
    var fields = ctx.fields || [];
    if (typeof state.field === 'string' && fields.indexOf(state.field) >= 0) out.field = state.field;
    var sc = state.scale && typeof state.scale === 'object' ? state.scale : null;
    if (sc && out.field) {
      var w = sc.window, wins = (ctx.windows && ctx.windows[out.field]) || [];
      var win = 'adaptive';
      if (w && typeof w === 'object' && Array.isArray(w.range) && w.range.length === 2 && w.range.every(function (x) { return typeof x === 'number' && Number.isFinite(x); }) && w.range[0] !== w.range[1]) win = { range: [Math.min(w.range[0], w.range[1]), Math.max(w.range[0], w.range[1])] };
      else if (typeof w === 'string' && (w === 'adaptive' || wins.indexOf(w) >= 0)) win = w;
      out.scale = {
        window: win,
        log: sc.log === true ? true : (sc.log === false ? false : null),
        bands: Number.isInteger(sc.bands) && sc.bands >= 0 && sc.bands <= 64 ? sc.bands : 0,
        cmap: typeof sc.cmap === 'string' && ['rainbow', 'turbo', 'viridis', 'bwr'].indexOf(sc.cmap) >= 0 ? sc.cmap : 'rainbow'
      };
    }
    var cam = cleanCamera(state.camera);
    if (cam) out.camera = cam;
    if (state.lighting === 'flat' || state.lighting === 'soft') out.lighting = state.lighting;
    if (state.layers && typeof state.layers === 'object') {
      var L = {};
      Object.keys(DEFAULT_LAYERS[ctx.family] || DEFAULT_LAYERS.wall).forEach(function (k) { if (typeof state.layers[k] === 'boolean') L[k] = state.layers[k]; });
      out.layers = L;
    }
    var br = ctx.branches || null;
    if (state.branches === null) out.branches = null;
    else if (Array.isArray(state.branches)) {
      var ids = state.branches.map(Number).filter(function (x) { return Number.isInteger(x) && (!br || br.indexOf(x) >= 0); });
      out.branches = ids.length ? ids : null;
    }
    if (state.cursor === null) out.cursor = null;
    else if (state.cursor && typeof state.cursor === 'object' && Number.isFinite(+state.cursor.segmentId) && Number.isFinite(+state.cursor.s_mm) && (!br || br.indexOf(+state.cursor.segmentId) >= 0)) out.cursor = { segmentId: +state.cursor.segmentId, s_mm: +state.cursor.s_mm };
    if (state.selection === null || (Number.isInteger(state.selection) && state.selection >= 0)) out.selection = state.selection;
    if (state.time_index !== undefined && state.time_index !== 0) out.ignored = ['time_index'];   // one frame only (E7)
    return out;
  }
  var DEFAULT_LAYERS = {
    wall: { outline: false, centerline: false, points: false, trust: false, wall: true, interior: false, streamlines: false },
    volume: { outline: true, centerline: false, points: false, trust: false, wall: true, interior: true, streamlines: false }
  };
  function normalizeLayers(family, cur, next) {
    var base = Object.assign({}, DEFAULT_LAYERS[family] || DEFAULT_LAYERS.wall, cur || {});
    if (next && typeof next === 'object') Object.keys(base).forEach(function (k) { if (typeof next[k] === 'boolean') base[k] = next[k]; });
    return base;
  }

  // ---------------------------------------------------------------- WebGL availability (evaluated on call)
  var glCache = null;
  function webglAvailable() {
    if (glCache !== null) return glCache;
    try {
      var doc = root.document;
      if (!doc || !doc.createElement) return (glCache = false);
      var c = doc.createElement('canvas');
      var gl = c.getContext('webgl2') || c.getContext('webgl') || c.getContext('experimental-webgl');
      glCache = !!gl;
      if (gl) { var ext = gl.getExtension('WEBGL_lose_context'); if (ext) ext.loseContext(); }
    } catch (_) { glCache = false; }
    return glCache;
  }

  // ---------------------------------------------------------------- shared graphics helpers (used by adapters)
  var gfx = {
    // Silhouette outline: the shared surface drawn again, displaced along its normals by a constant number of
    // screen pixels, only the faces turned away from the camera, pushed back in depth so the surface wins ties.
    outlineMaterial: function (o) {
      var THREE = T(); o = o || {};
      var mat = new THREE.ShaderMaterial({
        uniforms: {
          uColor: { value: new THREE.Color(o.color || INK_HEX) }, uOpacity: { value: o.opacity === undefined ? 1 : o.opacity },
          uWidth: { value: o.width || 1.5 }, uResolution: { value: new THREE.Vector2(800, 600) }, uSign: { value: o.sign === -1 ? -1 : 1 }
        },
        vertexShader: [
          'uniform float uWidth; uniform vec2 uResolution; uniform float uSign;',
          'void main(){',
          '  vec4 clip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
          '  vec3 n = normalize(normalMatrix * normal) * uSign;',
          '  vec2 dir = (projectionMatrix * vec4(n, 0.0)).xy;',
          '  float len = length(dir);',
          '  dir = len > 1e-6 ? dir / len : vec2(0.0);',
          '  clip.xy += dir * uWidth * 2.0 / uResolution * clip.w;',
          '  gl_Position = clip;',
          '}'].join('\n'),
        fragmentShader: 'uniform vec3 uColor; uniform float uOpacity; void main(){ gl_FragColor = vec4(uColor, uOpacity); }',
        side: o.sign === -1 ? THREE.FrontSide : THREE.BackSide,
        transparent: (o.opacity !== undefined && o.opacity < 1), depthWrite: !(o.opacity !== undefined && o.opacity < 1),
        polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 6
      });
      mat.userData.outline = true;
      return mat;
    },
    // Diagonal screen-space hatch over flagged vertices (attribute aFlag ∈ {0,1}); the colour underneath stays
    // readable between the stripes.
    hatchMaterial: function (o) {
      var THREE = T(); o = o || {};
      return new THREE.ShaderMaterial({
        uniforms: {
          uColor: { value: new THREE.Color(o.color || INK_HEX) }, uOpacity: { value: o.opacity === undefined ? 0.62 : o.opacity },
          uSpacing: { value: o.spacing || 7 }, uLine: { value: o.line || 2 }
        },
        vertexShader: 'attribute float aFlag; varying float vFlag; void main(){ vFlag = aFlag; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
        fragmentShader: 'uniform vec3 uColor; uniform float uOpacity; uniform float uSpacing; uniform float uLine; varying float vFlag;' +
          'void main(){ if (vFlag < 0.5) discard; float d = mod(gl_FragCoord.x + gl_FragCoord.y, uSpacing); if (d > uLine) discard; gl_FragColor = vec4(uColor, uOpacity); }',
        side: THREE.DoubleSide, transparent: true, depthWrite: false,
        polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4
      });
    },
    // Round point sprite as a DataTexture (no canvas / DOM needed).
    roundPointTexture: function () {
      var THREE = T(), n = 32, data = new Uint8Array(n * n * 4);
      for (var y = 0; y < n; y++) for (var x = 0; x < n; x++) {
        var dx = (x + 0.5) / n * 2 - 1, dy = (y + 0.5) / n * 2 - 1, r = Math.sqrt(dx * dx + dy * dy);
        var a = r <= 0.82 ? 255 : r >= 1 ? 0 : Math.round(255 * (1 - (r - 0.82) / 0.18));
        var i = 4 * (y * n + x); data[i] = data[i + 1] = data[i + 2] = 255; data[i + 3] = a;
      }
      var tex = new THREE.DataTexture(data, n, n, THREE.RGBAFormat);
      tex.needsUpdate = true;
      tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter;
      return tex;
    },
    // Surface orientation sign of an (open) triangle mesh: +1 when the winding gives outward normals.
    windingSign: function (V, F) {
      var n = Math.floor(V.length / 3), cx = 0, cy = 0, cz = 0, i;
      for (i = 0; i < n; i++) { cx += V[3 * i]; cy += V[3 * i + 1]; cz += V[3 * i + 2]; }
      cx /= n || 1; cy /= n || 1; cz /= n || 1;
      var vol = 0;
      for (i = 0; i + 2 < F.length; i += 3) {
        var a = F[i], b = F[i + 1], c = F[i + 2];
        var ax = V[3 * a] - cx, ay = V[3 * a + 1] - cy, az = V[3 * a + 2] - cz;
        var bx = V[3 * b] - cx, by = V[3 * b + 1] - cy, bz = V[3 * b + 2] - cz;
        var qx = V[3 * c] - cx, qy = V[3 * c + 1] - cy, qz = V[3 * c + 2] - cz;
        vol += ax * (by * qz - bz * qy) - ay * (bx * qz - bz * qx) + az * (bx * qy - by * qx);
      }
      return vol < 0 ? -1 : 1;
    },
    // Releases geometries, materials and textures below obj (each once).
    disposeObject: function (obj) {
      if (!obj) return;
      var seen = new Set();
      obj.traverse(function (o) {
        if (o.geometry && !seen.has(o.geometry)) { seen.add(o.geometry); o.geometry.dispose(); }
        var mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
        mats.forEach(function (m) {
          if (seen.has(m)) return; seen.add(m);
          ['map', 'alphaMap'].forEach(function (k) { if (m[k] && !seen.has(m[k])) { seen.add(m[k]); m[k].dispose(); } });
          m.dispose();
        });
      });
      if (obj.parent) obj.parent.remove(obj);
    },
    // Uniform scale for markers / rings: a fraction of the model size.
    VIEWPORT_HEX: VIEWPORT_HEX, INK_HEX: INK_HEX, ACCENT_HEX: ACCENT_HEX
  };

  function cssVar(el, name, fallback) {
    try {
      var v = root.getComputedStyle ? root.getComputedStyle(el).getPropertyValue(name) : '';
      v = String(v || '').trim();
      return /^#[0-9a-f]{6}$/i.test(v) ? v : fallback;
    } catch (_) { return fallback; }
  }
  function reducedMotion() {
    try { return !!(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches); } catch (_) { return false; }
  }
  function dataURLToBlob(url) {
    var m = /^data:([^;,]+)(;base64)?,(.*)$/.exec(url || '');
    if (!m) throw new Error('snapshot failed');
    var bin = m[2] ? root.atob(m[3]) : decodeURIComponent(m[3]), u = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new root.Blob([u], { type: m[1] });
  }

  // ---------------------------------------------------------------- viewer
  function create(container, opts) {
    if (!container) throw new Error('viewer needs a container element');
    opts = opts || {};
    var THREE = T(), util = U(), doc = container.ownerDocument || root.document;
    var kind = opts.kind === 'thumb' ? 'thumb' : 'main';
    var ev = util.emitter();
    var disposed = false, lost = false, visible = true, dirty = true, rafId = 0, tween = null, fitName = null;
    var insets = Object.assign({}, DEFAULT_INSETS[kind], opts.insets || {});
    var cleanups = [];
    function listen(target, type, fn, o) { target.addEventListener(type, fn, o); cleanups.push(function () { target.removeEventListener(type, fn, o); }); }

    // DOM
    var wrap = doc.createElement('div');
    wrap.className = 'wssv2-viewport wssv2-viewport-' + kind;
    wrap.style.cssText = 'position:relative;width:100%;height:100%;overflow:hidden;background:transparent;touch-action:none;';
    container.appendChild(wrap);
    var labels = doc.createElement('div');
    labels.className = 'wssv2-labels';
    labels.style.cssText = 'position:absolute;inset:0;pointer-events:none;overflow:hidden;';

    var renderer = new THREE.WebGLRenderer({ antialias: kind === 'main', alpha: true, premultipliedAlpha: false, preserveDrawingBuffer: false });
    renderer.setPixelRatio(kind === 'thumb' ? 1 : Math.min(root.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.localClippingEnabled = true;   // the section tool cuts the vessel at its plane (setClipPlane)
    var canvas = renderer.domElement;
    canvas.style.cssText = 'display:block;width:100%;height:100%;outline:none;';
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', '三维视图：拖动旋转，滚轮缩放，右键平移，单击读数');
    canvas.tabIndex = kind === 'main' ? 0 : -1;
    wrap.appendChild(canvas);
    wrap.appendChild(labels);

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(30, 1, 0.1, 5000);
    scene.add(camera);
    var ambient = new THREE.HemisphereLight(0xffffff, 0x8a93a3, 0.62);
    scene.add(ambient);
    var fill = new THREE.DirectionalLight(0xffffff, 0.16);
    fill.position.set(-0.4, -0.6, 0.2);
    camera.add(fill);
    var headlight = new THREE.DirectionalLight(0xffffff, 0.58);
    headlight.position.set(0.35, 0.5, 0);
    camera.add(headlight);
    camera.add(headlight.target);
    headlight.target.position.set(0, 0, -1);
    var content = new THREE.Group(); content.name = 'content'; scene.add(content);
    var overlay = new THREE.Group(); overlay.name = 'overlay'; scene.add(overlay);

    var controls = null, controlsUp = null;
    function ensureControls() {
      var u = camera.up.toArray();
      if (controls && controlsUp && u.every(function (v, k) { return Math.abs(v - controlsUp[k]) < 1e-6; })) return;
      var tgt = controls ? controls.target.clone() : null;
      if (controls) controls.dispose();
      controls = new THREE.OrbitControls(camera, canvas);
      controls.enableDamping = false;
      controls.rotateSpeed = 0.9;
      controls.zoomSpeed = 1.1;
      controls.screenSpacePanning = true;
      if (tgt) controls.target.copy(tgt);
      controls.addEventListener('change', onControlsChange);
      controls.addEventListener('start', function () { fitName = null; interacting = true; });
      controls.addEventListener('end', function () { interacting = false; });
      controlsUp = u;
    }
    var interacting = false;
    function onControlsChange() {
      requestRender();
      if (!applyingCamera) { fitName = null; emitCamera('user'); }
    }
    ensureControls();

    // state
    var S = {
      result: null, handle: null, adapter: null, token: 0,
      fieldId: null, spec: { window: 'adaptive', log: null, bands: 0, cmap: 'rainbow' }, resolved: null,
      lighting: 'soft', layers: null, cursor: null, cursorHelper: null, selection: null, branches: null,
      markers: [], highlight: null, stats: {}, hist: null, histKey: '', branchVersion: 0
    };
    var applyingCamera = false;
    var size = { w: 1, h: 1 };

    // ---- rendering on demand
    function requestRender() {
      if (disposed) return;
      dirty = true;
      if (rafId || !visible) return;
      rafId = (root.requestAnimationFrame || function (f) { return setTimeout(f, 16); })(frame);
    }
    function frame() {
      rafId = 0;
      if (disposed) return;
      if (!visible || lost) return;
      var moving = false;
      if (tween) {
        var u = Math.min(1, (util.now() - tween.t0) / tween.ms), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        applyingCamera = true;
        camera.position.lerpVectors(tween.p0, tween.p1, e);
        controls.target.lerpVectors(tween.q0, tween.q1, e);
        camera.up.copy(tween.u0).lerp(tween.u1, e).normalize();
        camera.lookAt(controls.target);
        if (u >= 1) { tween = null; ensureControls(); controls.update(); clipPlanes(); }
        applyingCamera = false;
        if (!tween) emitCamera(tweenSource); else moving = true;
      }
      updateOutlineResolution();
      updateScreenSized();
      applyClip();
      renderer.render(scene, camera);
      dirty = false;
      placeLabels();
      ev.emit('render', null);
      if (moving) requestRender();
    }
    var tweenSource = 'api', wtmp = null;
    // World size of one CSS pixel at distance d from the camera.
    function worldPerPixel(d) { return 2 * d * Math.tan(camera.fov * Math.PI / 360) / Math.max(1, size.h); }
    // Overlay objects flagged userData.screenPx keep a constant on-screen size; the cursor ring is rebuilt
    // when the zoom changed enough for its tube to look too thin or too thick.
    function updateScreenSized() {
      wtmp = wtmp || new THREE.Vector3();
      overlay.traverse(function (o) {
        if (!o.userData || !o.userData.screenPx) return;
        o.getWorldPosition(wtmp);
        o.scale.setScalar(o.userData.screenPx * worldPerPixel(camera.position.distanceTo(wtmp)));
      });
      if (cursorGroup && cursorGroup.userData.station) {
        var st = cursorGroup.userData.station, d = camera.position.distanceTo(wtmp.set(st.xyz[0], st.xyz[1], st.xyz[2]));
        if (!cursorGroup.userData.dist || Math.abs(Math.log(d / cursorGroup.userData.dist)) > 0.3) drawCursor();
      }
    }
    function updateOutlineResolution() {
      var pr = renderer.getPixelRatio();
      content.traverse(function (o) {
        var m = o.material;
        if (m && m.uniforms && m.uniforms.uResolution) m.uniforms.uResolution.value.set(size.w * pr, size.h * pr);
        if (m && m.uniforms && m.uniforms.uSpacing && m.userData.basePx) { m.uniforms.uSpacing.value = m.userData.basePx.spacing * pr; m.uniforms.uLine.value = m.userData.basePx.line * pr; }
        if (m && m.uniforms && m.uniforms.uWidth && m.userData.baseWidth) m.uniforms.uWidth.value = m.userData.baseWidth * pr;
      });
    }
    function resize() {
      if (disposed) return;
      var w = Math.max(1, Math.round(wrap.clientWidth || container.clientWidth || 1)), h = Math.max(1, Math.round(wrap.clientHeight || container.clientHeight || 1));
      if (w === size.w && h === size.h) return;
      size = { w: w, h: h };
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      if (fitName && S.handle) standardView(fitName, { animate: false, keepFit: true });
      requestRender();
    }
    if (typeof root.ResizeObserver === 'function') {
      var ro = new root.ResizeObserver(function () { resize(); });
      ro.observe(wrap);
      cleanups.push(function () { ro.disconnect(); });
    } else if (root.addEventListener) listen(root, 'resize', resize);
    if (typeof root.IntersectionObserver === 'function') {
      var io = new root.IntersectionObserver(function (entries) {
        var e = entries[entries.length - 1];
        visible = !!(e && e.isIntersecting);
        if (visible && dirty) requestRender();
      });
      io.observe(wrap);
      cleanups.push(function () { io.disconnect(); });
    }
    listen(canvas, 'webglcontextlost', function (e) { e.preventDefault(); lost = true; ev.emit('contextlost', null); });
    listen(canvas, 'webglcontextrestored', function () { lost = false; ev.emit('contextrestored', null); requestRender(); });

    // ---- camera
    function cameraState() {
      return { position: camera.position.toArray(), target: controls.target.toArray(), up: camera.up.toArray(), fov: camera.fov };
    }
    function emitCamera(source) { ev.emit('camera', { camera: cameraState(), source: source || 'api' }); }
    function setCamera(c, o) {
      o = o || {};
      c = cleanCamera(c);
      if (!c) return false;
      var source = o.source || 'api';
      if (c.fov && Math.abs(c.fov - camera.fov) > 1e-6) { camera.fov = c.fov; camera.updateProjectionMatrix(); }
      if (o.animate && !reducedMotion()) {
        tween = { t0: util.now(), ms: Math.min(300, o.ms || 280), p0: camera.position.clone(), q0: controls.target.clone(), u0: camera.up.clone(),
          p1: new THREE.Vector3().fromArray(c.position), q1: new THREE.Vector3().fromArray(c.target), u1: new THREE.Vector3().fromArray(c.up) };
        tweenSource = source;
        requestRender();
        return true;
      }
      tween = null;
      applyingCamera = true;
      camera.position.fromArray(c.position);
      camera.up.fromArray(c.up).normalize();
      ensureControls();
      controls.target.fromArray(c.target);
      camera.lookAt(controls.target);
      controls.update();
      applyingCamera = false;
      clipPlanes();
      requestRender();
      emitCamera(source);
      return true;
    }
    function clipPlanes() {
      var b = S.handle && S.handle.bounds;
      var sz = b ? b.size : 300;
      var dist = camera.position.distanceTo(controls.target);
      camera.near = Math.max(0.01, Math.min(sz * 0.002, dist * 0.01));
      camera.far = Math.max(sz * 20, dist * 4);
      camera.updateProjectionMatrix();
    }
    function frameOf() { var m = S.result && S.result.manifest; return m && validFrame(m.frame) ? m.frame : null; }
    function standardView(name, o) {
      o = o || {};
      if (!S.handle) return false;
      var d = viewDirection(name, frameOf());
      if (!d) return false;
      var pts = S.handle.fitPoints();
      var f = insetFit(pts, { dir: d.dir, up: d.up, fov: camera.fov, width: size.w, height: size.h, insets: insets, margin: o.margin || 1.06 });
      setCamera({ position: f.position, target: f.target, up: f.up }, { animate: !!o.animate, source: o.source || 'api' });
      fitName = name;
      return true;
    }
    function fit(o) { return standardView(fitName || 'anterior', Object.assign({ animate: false }, o || {})); }

    // ---- fields and scales
    function fieldStats(fieldId) {
      if (S.stats[fieldId]) return S.stats[fieldId];
      var r = S.result, f = r.field(fieldId), st = {};
      var arr = null;
      try { arr = r.fieldArray(fieldId, 'read') || r.fieldArray(fieldId, 'display'); } catch (_) { arr = null; }
      var comps = Number(f && f.components) || 1;
      var disp = f && f.display || {}, stat = f && f.statistics || {}, N = util.num;
      if (Number.isFinite(N(disp.p99))) st.p99 = N(disp.p99); else if (Number.isFinite(N(stat.p99))) st.p99 = N(stat.p99);
      if (Number.isFinite(N(stat.min))) st.min = N(stat.min);
      if (arr && comps === 1) {
        var fr = util.finiteRange(arr);
        if (!Number.isFinite(st.min)) st.min = fr.min;
        st.max = fr.max;
        if (!Number.isFinite(st.p99)) st.p99 = util.quantile(arr, 0.99);
        if (st.min < 0) st.p1 = util.quantile(arr, 0.01);
      }
      S.stats[fieldId] = st;
      return st;
    }
    function fieldIds() { return S.handle ? S.handle.fields.slice() : []; }
    function mergeSpec(spec, keepWindow) {
      var cur = S.spec, s = spec || {};
      return {
        window: s.window !== undefined && s.window !== null ? s.window : (keepWindow ? cur.window : 'adaptive'),
        log: typeof s.log === 'boolean' ? s.log : (s.log === null ? null : (keepWindow ? cur.log : null)),
        bands: s.bands !== undefined && s.bands !== null ? Math.max(0, Math.floor(+s.bands) || 0) : cur.bands,
        cmap: s.cmap && ns.colormap.names().indexOf(s.cmap) >= 0 ? s.cmap : cur.cmap
      };
    }
    function applyField(fieldId, spec) {
      var r = S.result, f = r.field(fieldId);
      var st = fieldStats(fieldId);
      var res = ns.colormap.resolve(f, spec, st);
      S.fieldId = fieldId;
      S.spec = { window: res.spec.window, log: spec.log === true ? res.scale.log : spec.log, bands: res.scale.bands, cmap: res.scale.cmap };
      S.resolved = res;
      S.handle.setField(fieldId, res.scale);
      S.hist = null; S.histKey = '';
      if (S.selection !== null) drawSelection();
      requestRender();
      ev.emit('change', { what: 'field', fieldId: fieldId });
    }
    function setField(fieldId, scaleSpec) {
      if (!S.handle) return Promise.reject(new Error('no result'));
      if (S.handle.fields.indexOf(fieldId) < 0) {
        var err = new Error('field not displayable here: ' + fieldId); err.code = 'field';
        return Promise.reject(err);
      }
      var spec = mergeSpec(scaleSpec, fieldId === S.fieldId && !scaleSpec);
      var keys = S.result.fieldKeys(fieldId).filter(function (k) { return !S.result.has(k); });
      if (!keys.length) { applyField(fieldId, spec); return Promise.resolve(fieldId); }
      var token = S.token;
      return S.result.preload(keys).then(function () {
        if (token !== S.token || disposed) return null;
        applyField(fieldId, spec);
        return fieldId;
      });
    }
    function histogram() {
      if (!S.handle || !S.fieldId || !S.resolved) return null;
      var sc = S.resolved.scale, key = S.fieldId + '|' + sc.range.join(',') + '|' + sc.log + '|' + S.branchVersion;
      if (S.hist && S.histKey === key) return S.hist;
      var hv = S.handle.histogramValues(S.fieldId);
      if (!hv || !hv.values) return null;
      var h = ns.colormap.histogram(hv.values, { bins: 48, range: [sc.log ? sc.floor : sc.range[0], sc.range[1]], log: sc.log, floor: sc.floor, mask: hv.mask || null });
      h.source = hv.source; h.scope = hv.scope || 'all';
      S.hist = h; S.histKey = key;
      return h;
    }
    function colorbarInfo() {
      if (!S.handle || !S.fieldId || !S.resolved) return null;
      var f = S.result.field(S.fieldId) || {}, d = f.display || {}, h = histogram();
      var total = h ? h.n + h.nMissing : 0;
      return {
        fieldId: S.fieldId, label: f.short_label || f.label || S.fieldId, fullLabel: f.label || S.fieldId, units: f.units || '',
        tier: f.tier || null, source: f.source || null,
        scale: S.resolved.scale, histogram: h, histogramSource: h ? h.source : null, histogramScope: h ? h.scope : null,
        thresholds: Array.isArray(d.thresholds) ? d.thresholds.slice() : [], thresholdDirection: d.threshold_direction || null,
        windowLabel: { id: S.resolved.window.id, kind: S.resolved.window.kind, label: S.resolved.window.label, provisional: !!S.resolved.window.provisional, text: S.resolved.window.text, note: S.resolved.window.note || null },
        missingFraction: total ? h.nMissing / total : 0
      };
    }

    // ---- layers, lighting, branches
    function setLighting(mode) {
      S.lighting = mode === 'soft' ? 'soft' : 'flat';
      if (S.handle) S.handle.setLighting(S.lighting);
      requestRender();
      ev.emit('change', { what: 'lighting' });
    }
    function setLayers(layers) {
      var fam = S.result ? S.result.family : 'wall';
      S.layers = normalizeLayers(fam, S.layers, layers);
      if (S.handle) S.layers = normalizeLayers(fam, S.layers, S.handle.setLayers(S.layers) || null);
      requestRender();
      ev.emit('change', { what: 'layers' });
    }
    function setBranchVisibility(ids) {
      S.branches = Array.isArray(ids) ? ids.map(Number).filter(Number.isFinite) : null;
      S.branchVersion++;
      if (S.handle) S.handle.setBranchVisibility(S.branches ? new Set(S.branches) : null);
      S.hist = null;
      if (S.cursor) drawCursor();
      requestRender();
      ev.emit('change', { what: 'branches' });
    }

    // ---- overlay: selection, highlight, markers, cursor
    var selGroup = null, cursorGroup = null, markerGroup = null, markerChips = [];
    function modelSize() { return S.handle && S.handle.bounds ? S.handle.bounds.size : 100; }
    function drawSelection() {
      if (selGroup) { gfx.disposeObject(selGroup); selGroup = null; }
      if (S.selection === null || !S.handle) return;
      var p = S.handle.pointXYZ(S.selection);
      if (!p) return;
      selGroup = new THREE.Group();
      var halo = new THREE.Mesh(new THREE.SphereGeometry(1.6, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
      var dotm = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: new THREE.Color(ACCENT_HEX), depthTest: false }));
      halo.renderOrder = 20; dotm.renderOrder = 21;
      selGroup.add(halo); selGroup.add(dotm);
      selGroup.position.set(p[0], p[1], p[2]);
      selGroup.userData.screenPx = 4;   // core radius in CSS px, whatever the zoom
      overlay.add(selGroup);
    }
    function select(i) {
      S.selection = i === null || i === undefined || !Number.isInteger(+i) || +i < 0 ? null : +i;
      drawSelection();
      requestRender();
      ev.emit('change', { what: 'selection' });
    }
    function highlight(indices, o) {
      S.highlight = indices ? { indices: indices, color: o && o.color || null, size: o && o.size || null, opacity: o && o.opacity || null, vertexMask: o && o.vertexMask || null } : null;
      if (S.handle) S.handle.highlight(S.highlight ? S.highlight.indices : null, S.highlight || {});
      requestRender();
    }
    function setMarkers(list) {
      S.markers = Array.isArray(list) ? list.filter(function (m) { return m && Array.isArray(m.xyz) && m.xyz.length === 3 && m.xyz.every(function (x) { return Number.isFinite(+x); }); }).map(function (m) {
        return { id: m.id, xyz: m.xyz.map(Number), label: m.label == null ? '' : String(m.label), kind: m.kind || 'marker' };
      }) : [];
      drawMarkers();
      requestRender();
    }
    function drawMarkers() {
      if (markerGroup) { gfx.disposeObject(markerGroup); markerGroup = null; }
      markerChips.forEach(function (c) { if (c.el.parentNode) c.el.parentNode.removeChild(c.el); });
      markerChips = [];
      if (!S.markers.length) return;
      markerGroup = new THREE.Group();
      var geo = new THREE.SphereGeometry(1, 14, 10), halo = new THREE.SphereGeometry(1.6, 14, 10);
      var mInk = new THREE.MeshBasicMaterial({ color: new THREE.Color(INK_HEX), depthTest: false });
      var mHalo = new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false });
      S.markers.forEach(function (m, i) {
        var g = new THREE.Group();
        var h = new THREE.Mesh(halo, mHalo), d = new THREE.Mesh(geo, mInk);
        h.renderOrder = 18; d.renderOrder = 19;
        d.userData.markerId = m.id; h.userData.markerId = m.id;
        g.add(h); g.add(d);
        g.position.set(m.xyz[0], m.xyz[1], m.xyz[2]);
        g.userData.screenPx = 3.5;
        markerGroup.add(g);
        if (m.label) {
          var chip = doc.createElement('button');
          chip.type = 'button';
          chip.className = 'wssv2-marker wssv2-marker-' + String(m.kind).replace(/[^a-z0-9_-]/gi, '');
          chip.textContent = m.label;
          chip.style.cssText = 'position:absolute;left:0;top:0;transform:translate(-50%,-100%);pointer-events:auto;cursor:pointer;white-space:nowrap;margin-top:-10px;' +
            'font:12px/1.3 system-ui,-apple-system,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif;font-variant-numeric:tabular-nums;' +
            'color:var(--hud-ink,#1b2430);background:var(--hud-chip,#fff);border:1px solid var(--hud-line,#c6ccd2);border-radius:6px;padding:2px 7px;';
          chip.addEventListener('click', function (e) { e.stopPropagation(); ev.emit('marker', { id: m.id }); });
          labels.appendChild(chip);
          markerChips.push({ el: chip, m: m, idx: i });
        }
      });
      overlay.add(markerGroup);
    }
    var vtmp = null;
    // Text labels of the tools (measurements): DOM chips placed with the marker chips, keyed by tool.
    var toolChips = {};
    function setToolLabels(key, list) {
      (toolChips[key] || []).forEach(function (c) { if (c.el.parentNode) c.el.parentNode.removeChild(c.el); });
      toolChips[key] = (Array.isArray(list) ? list : []).filter(function (it) { return it && Array.isArray(it.xyz) && it.xyz.length === 3 && it.xyz.every(function (x) { return Number.isFinite(+x); }); }).map(function (it) {
        var el = doc.createElement('span');
        el.className = 'wssv2-toollabel wssv2-toollabel-' + String(key).replace(/[^a-z0-9_-]/gi, '');
        el.textContent = it.text == null ? '' : String(it.text);
        el.style.cssText = 'position:absolute;left:0;top:0;pointer-events:none;white-space:nowrap;' +
          'font:600 12px/1.3 system-ui,-apple-system,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif;font-variant-numeric:tabular-nums;' +
          'color:var(--hud-ink,#1b2430);background:var(--hud-chip,#fff);border:1px solid var(--hud-line,#c6ccd2);border-radius:6px;padding:2px 7px;';
        labels.appendChild(el);
        return { el: el, m: { xyz: it.xyz.map(Number) } };
      });
      requestRender();
    }
    function allChips() { var out = markerChips.slice(); Object.keys(toolChips).forEach(function (k) { out = out.concat(toolChips[k]); }); return out; }
    function placeLabels() {
      var chips = allChips();
      if (!chips.length) return;
      vtmp = vtmp || new THREE.Vector3();
      var items = [];
      chips.forEach(function (c) {
        vtmp.set(c.m.xyz[0], c.m.xyz[1], c.m.xyz[2]).project(camera);
        var inFront = vtmp.z < 1 && vtmp.z > -1;
        var x = (vtmp.x + 1) / 2 * size.w, y = (1 - vtmp.y) / 2 * size.h;
        items.push({ x: x, y: y - 18, w: Math.max(24, c.el.offsetWidth || 60), h: 18, c: c, inFront: inFront });
      });
      var placed = root.WssReportCommon && root.WssReportCommon.declutterLabels ? root.WssReportCommon.declutterLabels(items, { maxShift: 60, bounds: { width: size.w, height: size.h } }) : items;
      items.forEach(function (it, k) {
        var p = placed[k] || it;
        it.c.el.style.display = it.inFront ? '' : 'none';
        it.c.el.style.transform = 'translate(' + Math.round(p.x - it.w / 2) + 'px,' + Math.round(p.y + 9 - 18) + 'px)';
      });
    }
    function cursorHelper() {
      if (!S.result) return null;
      if (!S.cursorHelper && ns.cursor) { try { S.cursorHelper = ns.cursor.create(S.result); } catch (err) { S.cursorHelper = null; ev.emit('error', err); } }
      return S.cursorHelper;
    }
    function drawCursor() {
      if (cursorGroup) { gfx.disposeObject(cursorGroup); cursorGroup = null; }
      if (!S.cursor || !S.handle) return;
      var ch = cursorHelper(); if (!ch) return;
      var st = ch.station(S.cursor.segmentId, S.cursor.s_mm);
      if (!st) return;
      var mesh = S.handle.sectionMesh ? S.handle.sectionMesh() : null;
      var poly = mesh && st.tangent ? ringOutline(mesh, st.xyz, st.tangent, st.radius_mm || 1) : null;
      var ringIsOutline = !!poly;
      if (!poly) {   // fallback: a circle around the tangent, just outside the wall so it stays visible
        var n = st.tangent || [0, 0, 1], a = Math.abs(n[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
        var u = unit(cross(a, n)), v = cross(n, u), rr = Math.max(0.5, circleRadius(mesh, st.xyz, n, st.radius_mm || 1));
        poly = [];
        for (var k = 0; k < 48; k++) { var t = k / 48 * Math.PI * 2; poly.push([st.xyz[0] + rr * (Math.cos(t) * u[0] + Math.sin(t) * v[0]), st.xyz[1] + rr * (Math.cos(t) * u[1] + Math.sin(t) * v[1]), st.xyz[2] + rr * (Math.cos(t) * u[2] + Math.sin(t) * v[2])]); }
      }
      var pts = poly.map(function (p) { return new THREE.Vector3(p[0], p[1], p[2]); });
      var curve = new THREE.CatmullRomCurve3(pts, true, 'centripetal');
      var dist = camera.position.distanceTo(new THREE.Vector3(st.xyz[0], st.xyz[1], st.xyz[2]));
      var tube = Math.max(0.05, 1.8 * worldPerPixel(dist)), seg = Math.min(400, Math.max(48, pts.length * 2));
      cursorGroup = new THREE.Group();
      var outer = new THREE.Mesh(new THREE.TubeGeometry(curve, seg, tube, 6, true), new THREE.MeshBasicMaterial({ color: new THREE.Color(INK_HEX) }));
      var inner = new THREE.Mesh(new THREE.TubeGeometry(curve, seg, tube * 0.45, 6, true), new THREE.MeshBasicMaterial({ color: 0xffffff }));
      var ghost = new THREE.Mesh(outer.geometry, new THREE.MeshBasicMaterial({ color: new THREE.Color(INK_HEX), transparent: true, opacity: 0.45, depthTest: false }));
      outer.renderOrder = 12; inner.renderOrder = 13; ghost.renderOrder = 11;
      inner.position.set(0, 0, 0);
      cursorGroup.add(ghost); cursorGroup.add(outer); cursorGroup.add(inner);
      cursorGroup.userData = { station: st, ring: ringIsOutline ? 'outline' : 'circle', dist: dist };
      overlay.add(cursorGroup);
    }
    // Local lumen outline at a station: the plane ∩ display mesh within 3 × the inscribed radius, the loop around
    // the station (closed, or an opening closed by a straight edge).  Near a bifurcation the plane can cut several
    // lumens into one large loop; anything reaching beyond 3 r is rejected and the ring falls back to a circle of
    // the inscribed radius.  The ring only marks the position; it is not a measurement.
    function ringOutline(mesh, origin, normal, r) {
      var C = root.WssReportCommon;
      if (!C || typeof C.planeContour !== 'function') return null;
      try {
        var lim = 3 * Math.max(r, 0.5), plane = C.planeFrame(origin, normal);
        var segs = C.planeContour(mesh.vertices, mesh.faces, plane, null, lim);
        if (!segs.length) return null;
        var loop = C.selectLoop(C.contourLoops(segs), segs, [0, 0], lim);
        if (!loop) return null;
        var contour = loop.segs;
        if (!loop.closed) { contour = C.closeChain(contour); if (!contour) return null; }
        else if (!loop.inside) return null;
        var poly2 = C.loopPolygon(contour);
        if (poly2.length < 3) return null;
        for (var i = 0; i < poly2.length; i++) if (Math.hypot(poly2[i][0], poly2[i][1]) > lim) return null;
        return poly2.map(function (q) { return [0, 1, 2].map(function (k) { return plane.origin[k] + plane.u[k] * q[0] + plane.v[k] * q[1]; }); });
      } catch (_) { return null; }
    }
    // Radius of the fallback ring: the median distance to display-mesh vertices within ±1 mm of the station plane
    // and within 2.5 × the inscribed radius (≈ the local lumen radius), + 4 %; never below the inscribed radius.
    function circleRadius(mesh, o, n, r) {
      var d = [], V = mesh ? mesh.vertices : null, lim = 2.5 * Math.max(r, 0.5);
      if (V) for (var i = 0; i + 2 < V.length; i += 3) {
        var dx = V[i] - o[0], dy = V[i + 1] - o[1], dz = V[i + 2] - o[2], h = dx * n[0] + dy * n[1] + dz * n[2];
        if (Math.abs(h) > 1) continue;
        var rad = Math.sqrt(Math.max(0, dx * dx + dy * dy + dz * dz - h * h));
        if (rad <= lim) d.push(rad);
      }
      if (d.length < 6) return r * 1.05;
      d.sort(function (x, y) { return x - y; });
      return Math.max(r, d[d.length >> 1]) * 1.04;
    }
    function setCursor(c) {
      if (!c || !Number.isFinite(+c.segmentId) || !Number.isFinite(+c.s_mm)) S.cursor = null;
      else {
        var ch = cursorHelper();
        var st = ch ? ch.station(+c.segmentId, +c.s_mm) : null;
        S.cursor = st ? { segmentId: st.segmentId, s_mm: st.s_mm } : null;
      }
      drawCursor();
      requestRender();
      ev.emit('change', { what: 'cursor', cursor: S.cursor ? Object.assign({}, S.cursor) : null });
      return S.cursor ? Object.assign({}, S.cursor) : null;
    }

    // ---- picking
    var raycaster = new THREE.Raycaster(), ndc = new THREE.Vector2();
    function rayAt(clientX, clientY) {
      var rect = canvas.getBoundingClientRect();
      var x = clientX - rect.left, y = clientY - rect.top;
      ndc.set(x / Math.max(1, rect.width) * 2 - 1, -(y / Math.max(1, rect.height)) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      return { x: x, y: y };
    }
    function pickAt(clientX, clientY) {
      if (!S.handle || lost) return null;
      var scr = rayAt(clientX, clientY);
      if (markerGroup) {
        var mh = raycaster.intersectObject(markerGroup, true);
        if (mh.length && mh[0].object.userData.markerId !== undefined) return { marker: mh[0].object.userData.markerId, screen: scr };
      }
      var t0 = util.now();
      var p = S.handle.pick(raycaster, S.fieldId);
      if (!p) return null;
      p.screen = scr;
      p.fieldId = S.fieldId;
      p.ms = util.now() - t0;
      return p;
    }
    var down = null;
    listen(canvas, 'pointerdown', function (e) { down = { x: e.clientX, y: e.clientY, t: util.now(), b: e.button }; });
    listen(canvas, 'pointerup', function (e) {
      if (!down) return;
      var d = Math.abs(e.clientX - down.x) + Math.abs(e.clientY - down.y), dt = util.now() - down.t, b = down.b;
      down = null;
      if (b !== 0 || d > 5 || dt > 700) return;
      var p = pickAt(e.clientX, e.clientY);
      if (p && p.marker !== undefined) { ev.emit('marker', { id: p.marker }); return; }
      ev.emit('pick', p);
    });
    var hoverT = util.rafThrottle(function (x, y) {
      if (disposed || interacting) return;
      var p = pickAt(x, y);
      ev.emit('hover', p && p.marker === undefined ? p : null);
    });
    if (kind === 'main') {
      listen(canvas, 'pointermove', function (e) { if (e.buttons) return; if (ev.count('hover')) hoverT(e.clientX, e.clientY); });
      listen(canvas, 'pointerleave', function () { hoverT.cancel(); if (ev.count('hover')) ev.emit('hover', null); });
    }

    // ---- tool hooks (second phase S1, the section tool): a group for a tool's own objects, a ray at client
    // coordinates, screen projection and scale, the orbit controls switch while a tool drags, the wall under the pointer.
    var toolGroup = null;
    function toolOverlay() { if (!toolGroup) { toolGroup = new THREE.Group(); toolGroup.name = 'tool'; overlay.add(toolGroup); } return toolGroup; }
    function toolRay(clientX, clientY) { if (lost) return null; rayAt(clientX, clientY); return raycaster; }
    function projectPoint(xyz) {
      var v = new THREE.Vector3(xyz[0], xyz[1], xyz[2]).project(camera);
      return { x: (v.x + 1) / 2 * size.w, y: (1 - v.y) / 2 * size.h, inFront: v.z > -1 && v.z < 1 };
    }
    function mmPerPixelAt(xyz) { return worldPerPixel(camera.position.distanceTo(new THREE.Vector3(xyz[0], xyz[1], xyz[2]))); }
    function surfaceAt(clientX, clientY) {
      if (!S.handle || lost || typeof S.handle.pickSurface !== 'function') return null;
      rayAt(clientX, clientY);
      return S.handle.pickSurface(raycaster);
    }
    function setControlsEnabled(on) { if (controls) controls.enabled = Boolean(on); }
    // One clipping plane on the result's own objects (not on the tool overlay): keeps the side where
    // side · normal · (p − origin) ≥ 0.  The plane object is reused, so moving it recompiles nothing.
    var clipPlane = null, clipList = null, clipSeen = false, pointPlane = null, pointList = null;
    function setClipPlane(p) {
      if (!p || !Array.isArray(p.normal) || !Array.isArray(p.origin)) { clipList = null; pointList = null; }
      else {
        var n = new THREE.Vector3(p.normal[0], p.normal[1], p.normal[2]).multiplyScalar(p.side < 0 ? -1 : 1).normalize();
        clipPlane = clipPlane || new THREE.Plane();
        clipPlane.setFromNormalAndCoplanarPoint(n, new THREE.Vector3(p.origin[0], p.origin[1], p.origin[2]));
        clipList = clipList || [clipPlane];
        // point sprites: the plane moved into the kept side by p.pointGap mm (their radius)
        pointPlane = pointPlane || new THREE.Plane();
        pointPlane.copy(clipPlane); pointPlane.constant -= Math.max(0, Number(p.pointGap) || 0);
        pointList = pointList || [pointPlane];
      }
      requestRender();
    }
    function applyClip() {
      if (!clipList && !clipSeen) return;
      content.traverse(function (o) {
        var mats = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
        var want = o.isPoints ? pointList : clipList;
        mats.forEach(function (m) { if (m.clippingPlanes !== want) { m.clippingPlanes = want; m.needsUpdate = true; } });
      });
      clipSeen = Boolean(clipList);
    }

    // ==== lane A ==== (second phase: figure exports; PHASE2_LANES.md §3)
    // A pose is {position, target, up, fov}.  exportPose(name) is a standard view fitted to the whole export frame
    // (small even margins, no HUD insets); renderPose(o) draws the scene from a pose (the live camera when none is
    // given) offscreen at o.scale × the viewport's CSS size.  The live camera is posed, rendered and put back within
    // the call — no 'camera' event, the fitted view and the controls are untouched, and the canvas is redrawn from the
    // live camera before the browser shows it.  Outline and hatch widths follow the export scale (as they follow the
    // device pixel ratio on screen).  The HTML label chips (finding markers, tool labels) are not in the WebGL image:
    // they come back projected for that pose in CSS px of the export frame, for the caller to draw on its 2-D canvas.
    var EXPORT_INSETS = { top: 14, right: 14, bottom: 14, left: 14, corner: null };
    function exportPose(name, o) {
      o = o || {};
      if (!S.handle) return null;
      var d = viewDirection(name, frameOf());
      if (!d) return null;
      var f = insetFit(S.handle.fitPoints(), { dir: d.dir, up: d.up, fov: camera.fov, width: o.width || size.w, height: o.height || size.h, insets: o.insets || EXPORT_INSETS, margin: o.margin || 1.04 });
      return { position: f.position, target: f.target, up: f.up, fov: camera.fov };
    }
    function exportWidths(k) {
      content.traverse(function (o) {
        var m = o.material;
        if (m && m.uniforms && m.uniforms.uResolution) m.uniforms.uResolution.value.set(size.w * k, size.h * k);
        if (m && m.uniforms && m.uniforms.uSpacing && m.userData.basePx) { m.uniforms.uSpacing.value = m.userData.basePx.spacing * k; m.uniforms.uLine.value = m.userData.basePx.line * k; }
        if (m && m.uniforms && m.uniforms.uWidth && m.userData.baseWidth) m.uniforms.uWidth.value = m.userData.baseWidth * k;
      });
    }
    function exportLabels() {
      var out = [];
      vtmp = vtmp || new THREE.Vector3();
      allChips().forEach(function (c) {
        var text = c.el && c.el.textContent ? String(c.el.textContent) : '';
        if (!text) return;
        vtmp.set(c.m.xyz[0], c.m.xyz[1], c.m.xyz[2]).project(camera);
        if (!(vtmp.z < 1 && vtmp.z > -1)) return;
        var cls = String(c.el.className || '');
        out.push({ x: (vtmp.x + 1) / 2 * size.w, y: (1 - vtmp.y) / 2 * size.h, text: text, kind: cls.indexOf('wssv2-toollabel') >= 0 ? 'tool' : 'marker', xyz: c.m.xyz.slice() });
      });
      return out;
    }
    function renderPose(o) {
      o = o || {};
      return new Promise(function (resolve, reject) {
        if (!S.handle || disposed) { reject(new Error('no result')); return; }
        var pose = o.camera ? cleanCamera(o.camera) : null;
        if (o.camera && !pose) { reject(new Error('invalid camera')); return; }
        var k = Math.max(1, Math.min(4, +o.scale || 2));
        var saved = { p: camera.position.clone(), q: camera.quaternion.clone(), u: camera.up.clone(), fov: camera.fov, near: camera.near, far: camera.far };
        var used = pose ? { position: pose.position.slice(), target: pose.target.slice(), up: pose.up.slice(), fov: pose.fov || camera.fov } : cameraState();
        var prevBg = scene.background, out = null, err = null;
        try {
          if (pose) {
            var tgt = new THREE.Vector3().fromArray(pose.target), sz = S.handle.bounds ? S.handle.bounds.size : 300;
            camera.position.fromArray(pose.position);
            camera.up.fromArray(pose.up).normalize();
            if (pose.fov) camera.fov = pose.fov;
            camera.lookAt(tgt);
            var dist = camera.position.distanceTo(tgt);
            camera.near = Math.max(0.01, Math.min(sz * 0.002, dist * 0.01));
            camera.far = Math.max(sz * 20, dist * 4);
            camera.updateProjectionMatrix();
            camera.updateMatrixWorld(true);
            updateScreenSized();
          }
          applyClip();
          exportWidths(k);
          var bg = o.background === 'transparent' ? null : (o.background === 'white' ? '#ffffff' : (typeof o.background === 'string' && /^#[0-9a-f]{6}$/i.test(o.background) ? o.background : cssVar(wrap, '--snapshot-bg', '#ffffff')));
          scene.background = bg ? new THREE.Color(bg) : null;
          var r = root.WssReportCommon.renderOffscreen({ renderer: renderer, scene: scene, camera: camera, THREE: THREE, scale: k, width: size.w, height: size.h, background: bg ? undefined : 'transparent' });
          out = { dataURL: r.dataURL, width: r.width, height: r.height, scale: r.scale, downgraded: !!r.downgraded, css: { w: size.w, h: size.h },
            camera: used, labels: o.labels === false ? [] : exportLabels() };
        } catch (e) { err = e; }
        finally {
          scene.background = prevBg;
          if (pose) {
            camera.position.copy(saved.p); camera.quaternion.copy(saved.q); camera.up.copy(saved.u);
            camera.fov = saved.fov; camera.near = saved.near; camera.far = saved.far;
            camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);
          }
          updateOutlineResolution();
          // redraw from the live camera now, so the posed frame never reaches the screen
          if (rafId) { (root.cancelAnimationFrame || clearTimeout)(rafId); rafId = 0; }
          dirty = true;
          try { frame(); } catch (_) { requestRender(); }
        }
        if (err) reject(err); else resolve(out);
      });
    }
    // ==== end lane A ====

    // ---- result lifecycle
    function clearResult() {
      if (S.handle) { try { S.handle.dispose(); } catch (err) { ev.emit('error', err); } }
      S.handle = null; S.adapter = null; S.cursorHelper = null; S.stats = {}; S.hist = null; S.histKey = '';
      S.cursor = null; S.selection = null; S.highlight = null; S.markers = [];
      [selGroup, cursorGroup, markerGroup, toolGroup].forEach(function (g) { if (g) gfx.disposeObject(g); });
      selGroup = cursorGroup = markerGroup = toolGroup = null;
      markerChips.forEach(function (c) { if (c.el.parentNode) c.el.parentNode.removeChild(c.el); });
      markerChips = [];
      Object.keys(toolChips).forEach(function (k) { toolChips[k].forEach(function (c) { if (c.el.parentNode) c.el.parentNode.removeChild(c.el); }); });
      toolChips = {};
      while (content.children.length) gfx.disposeObject(content.children[content.children.length - 1]);
    }
    function setResult(result) {
      if (disposed) return Promise.reject(new Error('viewer disposed'));
      var token = ++S.token;
      if (!result) { clearResult(); S.result = null; requestRender(); return Promise.resolve(null); }
      var adapter = ns.adapters && ns.adapters[result.family];
      if (!adapter) { var e0 = new Error('unsupported result family: ' + result.family); ev.emit('error', e0); return Promise.reject(e0); }
      var prevField = S.fieldId;
      var want = adapter.defaultField(result, prevField);
      var keys = adapter.requiredArrays(result, want);
      return result.preload(keys).then(function () {
        if (token !== S.token || disposed) return null;
        clearResult();
        S.result = result;
        S.adapter = adapter;
        S.layers = normalizeLayers(result.family, null, null);
        S.handle = adapter.build(result, { THREE: THREE, root: content, gfx: gfx, requestRender: requestRender, kind: kind, util: util, fov: camera.fov });
        S.handle.setLayers(S.layers);
        S.handle.setLighting(S.lighting);
        S.branches = null; S.branchVersion++;
        applyField(want, mergeSpec(null, false));
        size = { w: 0, h: 0 }; resize();
        standardView('anterior', { animate: false });
        requestRender();
        ev.emit('ready', { family: result.family, fields: S.handle.fields.slice(), runIdentity: result.runIdentity });
        ev.emit('change', { what: 'result' });
        // Remaining field arrays in the background, so later field switches are instant.
        var rest = [];
        S.handle.fields.forEach(function (fid) { result.fieldKeys(fid).forEach(function (k) { if (!result.has(k)) rest.push(k); }); });
        (adapter.optionalArrays ? adapter.optionalArrays(result) : []).forEach(function (k) { if (!result.has(k) && result.declared(k)) rest.push(k); });
        if (rest.length) result.preload(rest).then(function () {
          if (token === S.token && S.handle && S.handle.arraysLoaded) { S.handle.arraysLoaded(); requestRender(); }
        }, function (err) { if (token === S.token) ev.emit('error', err); });
        return result;
      }, function (err) { if (token === S.token) ev.emit('error', err); throw err; });
    }

    // ==== lane E ==== (second phase: display options; PHASE2_LANES.md §3)
    // Display options come from one provider the workspace installs (ns.viewer.displayProvider, ws_display.js); every
    // call receives this viewer first:
    //   bands(v, fieldId) → int | null            colour bands of that field (null: leave the caller's spec alone)
    //   thresholds(v, fieldId) → [3] | null       observation thresholds drawn on the colour bar (display only)
    //   units(v, field) → {units, factor} | null  display units of a field: numbers × factor, the data are untouched
    //   flags(v) → {…}                            adapter display flags (input STL, stagnation hatch, wall opacity,
    //                                             streamline density / width, velocity arrows)
    //   markers(v) → [{xyz, kind, text, segmentId}]   persistent markers (field maximum, TAWSS minimum)
    //   state(v) → {…} / restore(v, state)        the display part of getState / applyState
    // Without a provider (the kernel dev page, the kernel tests) everything here is a pass-through.  The viewer's own
    // functions are wrapped by reassigning their bindings, so internal callers (applyState → setField, setResult →
    // applyField) go through the wrappers as well; the exported object below picks the wrapped bindings up.
    function displayProvider() { var p = ns.viewer && ns.viewer.displayProvider; return p && typeof p === 'object' ? p : null; }
    function dpCall(name, a, b) {
      var p = displayProvider();
      if (!p || typeof p[name] !== 'function') return null;
      try { return p[name](viewer, a, b); } catch (err) { ev.emit('error', err); return null; }
    }
    function dpBands(fieldId) { var b = dpCall('bands', fieldId); return b === null || b === undefined || !Number.isFinite(+b) ? null : Math.max(0, Math.floor(+b)); }
    function pushDisplayFlags() {
      if (!S.handle || typeof S.handle.setDisplay !== 'function' || !displayProvider()) return;
      try { S.handle.setDisplay(dpCall('flags') || {}); } catch (err) { ev.emit('error', err); }
    }
    var peakGroup = null;
    function clearPeaks() {
      if (peakGroup) { gfx.disposeObject(peakGroup); peakGroup = null; }
      if (toolChips.peak) setToolLabels('peak', []);
    }
    // Persistent markers: an ink ring around a yellow (maximum) or white (minimum) dot, constant on-screen size,
    // drawn over the vessel like the finding markers; the value chip goes with the tool labels (decluttered).
    function drawPeaks() {
      clearPeaks();
      if (!S.handle || !displayProvider()) return;
      var list = (dpCall('markers') || []).filter(function (m) {
        return m && Array.isArray(m.xyz) && m.xyz.length === 3 && m.xyz.every(function (x) { return Number.isFinite(+x); }) &&
          !(S.branches && m.segmentId !== null && m.segmentId !== undefined && S.branches.indexOf(Number(m.segmentId)) < 0);
      });
      if (!list.length) { requestRender(); return; }
      peakGroup = new THREE.Group(); peakGroup.name = 'peaks';
      var geo = new THREE.SphereGeometry(1, 14, 10), halo = new THREE.SphereGeometry(1.7, 14, 10);
      // transparent (opacity 1) so they draw after the transparent overlays of the wall (trust / stagnation hatch)
      var mInk = new THREE.MeshBasicMaterial({ color: new THREE.Color(INK_HEX), depthTest: false, transparent: true });
      list.forEach(function (m) {
        var g = new THREE.Group();
        var h = new THREE.Mesh(halo, mInk), d = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: new THREE.Color(m.kind === 'min' ? '#ffffff' : '#ffd400'), depthTest: false, transparent: true }));
        h.renderOrder = 16; d.renderOrder = 17;
        g.add(h); g.add(d);
        g.position.set(+m.xyz[0], +m.xyz[1], +m.xyz[2]);
        g.userData.screenPx = 3.5;
        peakGroup.add(g);
      });
      overlay.add(peakGroup);
      setToolLabels('peak', list.filter(function (m) { return m.text; }).map(function (m) { return { xyz: m.xyz, text: m.text }; }));
    }
    // Re-applies everything the provider decides for the current result: adapter flags, the current field with its
    // bands (the colour bar follows through the 'change' event), markers.
    function refreshDisplay() {
      if (!S.handle) return;
      pushDisplayFlags();
      if (S.fieldId && S.resolved) {
        var b = dpBands(S.fieldId);
        applyField(S.fieldId, { window: S.spec.window, log: S.spec.log, bands: b === null ? S.spec.bands : b, cmap: S.spec.cmap });
      } else drawPeaks();
      requestRender();
    }
    var laneE = { setField: setField, applyField: applyField, colorbarInfo: colorbarInfo, getState: getState, applyState: applyState, setResult: setResult, setBranchVisibility: setBranchVisibility };
    setField = function (fieldId, scaleSpec) {
      var b = displayProvider() ? dpBands(fieldId) : null;
      if (b === null) return laneE.setField(fieldId, scaleSpec);
      var spec = scaleSpec ? Object.assign({}, scaleSpec) : (fieldId === S.fieldId ? { window: S.spec.window, log: S.spec.log, cmap: S.spec.cmap } : {});
      spec.bands = b;
      return laneE.setField(fieldId, spec);
    };
    applyField = function (fieldId, spec) { laneE.applyField(fieldId, spec); drawPeaks(); };
    colorbarInfo = function () {
      var info = laneE.colorbarInfo();
      if (!info || !displayProvider()) return info;
      var th = dpCall('thresholds', info.fieldId);
      if (Array.isArray(th)) info.thresholds = th.slice();
      var u = dpCall('units', S.result ? S.result.field(info.fieldId) : null);
      if (u && typeof u.units === 'string' && Number.isFinite(+u.factor) && +u.factor > 0 && +u.factor !== 1) { info.displayUnits = u.units; info.unitFactor = +u.factor; }
      var fl = dpCall('flags');
      if (fl && fl.stl) info.stl = true;
      return info;
    };
    getState = function () {
      var st = laneE.getState();
      var d = S.handle ? dpCall('state') : null;
      if (d && typeof d === 'object') st.display = d;
      return st;
    };
    applyState = function (state, o) {
      if (state && typeof state === 'object' && state.display && typeof state.display === 'object' && S.handle) { dpCall('restore', state.display); pushDisplayFlags(); }
      return laneE.applyState(state, o);
    };
    setResult = function (result) {
      clearPeaks();
      return laneE.setResult(result).then(function (r) {
        if (r && S.result === r && S.handle) { pushDisplayFlags(); drawPeaks(); requestRender(); }
        return r;
      });
    };
    setBranchVisibility = function (ids) { laneE.setBranchVisibility(ids); if (peakGroup || toolChips.peak) drawPeaks(); };
    // ==== end lane E ====

    // ==== P3 lane 4 ==== (phase 3: hover readout, iso-lines, highlight, clipping, trust greying; PHASE3_LANES.md §3)
    // More optional provider methods (ws_display.js), each asked with this viewer first, wall results only:
    //   contours(v, fieldId) → bool     iso-lines at the colour bar's three thresholds on the display mesh (classic 等值线:
    //                                   WssReportCore.marchingTriangles on the display values, ported to core_contour.js)
    //   top(v, fieldId) → pct | null    the prediction points at or above the classic topThreshold of the field (高亮最高 x%)
    //   clip(v) → t | null              keep z ≤ min z + t · (max z − min z) of the display mesh (classic 剖切高度, world Z)
    // Display only: the lines, points and the cut follow the field, thresholds, branches and input-STL mode; no number
    // changes.  The hover readout lives in ws_probe.js (it listens to the 'hover' event the viewer already emits).
    var CONTOUR_HEX = '#203049', TOP_HEX = '#f233bf';   // classic contour colour 0x203049; classic highlight rgb(.95, .2, .75)
    var contourObj = null, topObj = null, clipZ = null, overlayClipped = false, p3Top = null, p3Lines = 0;
    function p3Wall() { return Boolean(S.handle && S.result && S.result.family === 'wall' && S.fieldId); }
    function p3Stl() { var fl = displayProvider() ? dpCall('flags') : null; return Boolean(fl && fl.stl); }
    function p3Thresholds(fieldId) {
      var th = dpCall('thresholds', fieldId);
      if (Array.isArray(th)) return th;
      var f = S.result.field(fieldId), d = f && f.display;
      return d && Array.isArray(d.thresholds) ? d.thresholds : [];
    }
    function p3Array(key) { try { return typeof key === 'string' && S.result.has(key) ? S.result.array(key) : null; } catch (_) { return null; } }
    // Lines lie on the surface: drawn with the surface's depth, pulled 0.1 % of their distance toward the camera so the
    // wall never hides them where they lie on it; they take the viewer's clipping planes (the Z cut).
    function contourMaterial() {
      return new THREE.ShaderMaterial({
        clipping: true, uniforms: { uColor: { value: new THREE.Color(CONTOUR_HEX) } },
        vertexShader: '#include <clipping_planes_pars_vertex>\nvoid main(){ vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);\n#include <clipping_planes_vertex>\n gl_Position = projectionMatrix * vec4(mvPosition.xyz * 0.999, 1.0); }',
        fragmentShader: '#include <clipping_planes_pars_fragment>\nuniform vec3 uColor; void main(){\n#include <clipping_planes_fragment>\n gl_FragColor = vec4(uColor, 1.0); }'
      });
    }
    function drawContours() {
      if (contourObj) { gfx.disposeObject(contourObj); contourObj = null; }
      p3Lines = 0;
      if (!p3Wall() || !displayProvider() || !ns.contour || p3Stl() || !dpCall('contours', S.fieldId)) return;
      var values = null;
      try { values = S.result.fieldArray(S.fieldId, 'display'); } catch (_) { values = null; }
      var mesh = S.handle.sectionMesh ? S.handle.sectionMesh() : null;
      if (!values || !mesh || !mesh.vertices || !mesh.faces) return;
      var seg = ns.contour.segments(mesh.vertices, mesh.faces, values, p3Thresholds(S.fieldId));
      p3Lines = seg.length / 6;
      if (!seg.length) return;
      var g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(seg, 3));
      contourObj = new THREE.LineSegments(g, contourMaterial());
      contourObj.name = 'p3-contours'; contourObj.renderOrder = 5;
      contourObj.visible = !S.layers || S.layers.wall !== false;
      content.add(contourObj);
    }
    // Magenta points on the prediction points of the highest pct % (hidden branches left out), drawn over the vessel
    // like the classic highlight (depthTest off), at a constant screen size.
    function drawTop() {
      if (topObj) { gfx.disposeObject(topObj); topObj = null; }
      p3Top = null;
      if (!p3Wall() || !displayProvider() || !ns.contour || p3Stl()) return;
      var pct = dpCall('top', S.fieldId);
      if (!(Number(pct) > 0)) return;
      pct = ns.contour.cleanPct(pct);
      var read = null;
      try { read = S.result.fieldArray(S.fieldId, 'read'); } catch (_) { read = null; }
      var pk = (S.result.manifest.geometry && S.result.manifest.geometry.points) || {}, PV = p3Array(pk.xyz), PS = p3Array(pk.segment);
      if (!read || !PV || PV.length !== 3 * read.length) return;
      var shown = S.branches && PS ? S.branches : null;
      var r = ns.contour.topIndices(read, pct, shown ? function (i) { return shown.indexOf(Number(PS[i])) >= 0; } : null);
      p3Top = { pct: pct, threshold: r.threshold, count: r.indices.length };
      if (!r.indices.length) return;
      var pos = new Float32Array(3 * r.indices.length);
      r.indices.forEach(function (i, j) { pos[3 * j] = PV[3 * i]; pos[3 * j + 1] = PV[3 * i + 1]; pos[3 * j + 2] = PV[3 * i + 2]; });
      var g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      var mat = new THREE.PointsMaterial({ size: 4 * renderer.getPixelRatio(), sizeAttenuation: false, color: new THREE.Color(TOP_HEX), map: gfx.roundPointTexture(),
        alphaTest: 0.5, transparent: true, opacity: 0.95, depthTest: false });
      topObj = new THREE.Points(g, mat);
      topObj.name = 'p3-top'; topObj.renderOrder = 8;
      topObj.visible = !S.layers || S.layers.wall !== false;
      content.add(topObj);
    }
    function drawP3() { drawContours(); drawTop(); requestRender(); }
    // The wall Z cut through the viewer's one clipping plane (the section tool owns it on volume results, where this stays
    // off); the adapter's picking skips the removed part, overlay objects and label chips above the cut are hidden.
    function applyWallClip() {
      var wall = Boolean(S.handle && S.result && S.result.family === 'wall');
      var z = wall && displayProvider() && ns.contour ? ns.contour.clipLevel(S.handle.bounds, dpCall('clip')) : null;
      if (z === clipZ) return;
      clipZ = z;
      if (z === null) setClipPlane(null);
      else setClipPlane({ normal: [0, 0, -1], origin: [0, 0, z], side: 1, pointGap: 0 });
      if (S.handle && typeof S.handle.setPickClip === 'function') S.handle.setPickClip(z === null ? null : { normal: [0, 0, -1], origin: [0, 0, z] });
      requestRender();
    }
    var p3 = { applyField: applyField, setBranchVisibility: setBranchVisibility, setResult: setResult, refreshDisplay: refreshDisplay, setLayers: setLayers, applyClip: applyClip };
    applyClip = function () {
      p3.applyClip();
      var want = clipZ !== null && clipList ? clipList : null;
      if (!want && !overlayClipped) return;
      overlay.traverse(function (o) {
        (o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : []).forEach(function (m) { if (m.clippingPlanes !== want) { m.clippingPlanes = want; m.needsUpdate = true; } });
      });
      overlayClipped = Boolean(want);
    };
    ev.on('render', function () {
      if (clipZ === null) return;
      allChips().forEach(function (c) { if (c.m && c.m.xyz && c.m.xyz[2] > clipZ + 1e-6) c.el.style.display = 'none'; });
    });
    applyField = function (fieldId, spec) { p3.applyField(fieldId, spec); drawP3(); };
    setBranchVisibility = function (ids) { p3.setBranchVisibility(ids); drawP3(); };
    setLayers = function (layers) {
      p3.setLayers(layers);
      var on = !S.layers || S.layers.wall !== false;
      if (contourObj) contourObj.visible = on;
      if (topObj) topObj.visible = on;
    };
    setResult = function (result) {
      if (clipZ !== null) { clipZ = null; setClipPlane(null); }
      contourObj = null; topObj = null; p3Top = null; p3Lines = 0;   // the content group is emptied with the old result
      return p3.setResult(result).then(function (r) {
        if (r && S.result === r && S.handle) { applyWallClip(); drawP3(); }
        return r;
      });
    };
    refreshDisplay = function () {
      if (!S.handle) return;
      applyWallClip();
      p3.refreshDisplay();
      if (!(S.fieldId && S.resolved)) drawP3();
    };
    // ==== end P3 lane 4 ====

    // ---- state
    function getState() {
      return {
        field: S.fieldId,
        scale: { window: S.spec.window && typeof S.spec.window === 'object' ? { range: S.spec.window.range.slice() } : S.spec.window, log: S.spec.log, bands: S.spec.bands, cmap: S.spec.cmap },
        camera: cameraState(),
        lighting: S.lighting,
        layers: Object.assign({}, S.layers || {}),
        cursor: S.cursor ? { segmentId: S.cursor.segmentId, s_mm: S.cursor.s_mm } : null,
        selection: S.selection,
        branches: S.branches ? S.branches.slice() : null,
        time_index: 0
      };
    }
    function stateContext() {
      var wins = {};
      S.handle.fields.forEach(function (fid) { var f = S.result.field(fid); wins[fid] = (f && Array.isArray(f.windows) ? f.windows : []).map(function (w) { return w && w.id; }).filter(Boolean); });
      return { fields: S.handle.fields.slice(), windows: wins, branches: S.result.branches().map(function (b) { return Number(b.id); }), family: S.result.family };
    }
    function applyState(state, o) {
      o = o || {};
      if (!state || typeof state !== 'object') return Promise.resolve(false);
      if (!S.handle) return Promise.reject(new Error('no result'));
      var st = sanitizeState(state, stateContext());
      var jobs = [];
      if (st.field) jobs.push(setField(st.field, st.scale || { window: 'adaptive', log: null }));
      return Promise.all(jobs).then(function () {
        if (st.lighting) setLighting(st.lighting);
        if (st.layers) setLayers(st.layers);
        if (st.branches !== undefined) setBranchVisibility(st.branches);
        if (st.cursor !== undefined) setCursor(st.cursor);
        if (st.selection !== undefined) select(st.selection);
        // time_index: only frame 0 exists (E7 placeholder); other values are ignored.
        if (st.camera) { fitName = null; setCamera(st.camera, { animate: !!o.animate, source: 'api' }); }
        return st;
      });
    }

    function snapshot(o) {
      o = o || {};
      return new Promise(function (resolve, reject) {
        try {
          updateOutlineResolution();
          var prevBg = scene.background;
          if (!o.transparent) scene.background = new THREE.Color(o.background || cssVar(wrap, '--snapshot-bg', '#ffffff'));
          var r;
          try { r = root.WssReportCommon.renderOffscreen({ renderer: renderer, scene: scene, camera: camera, THREE: THREE, scale: Math.max(1, Math.min(4, +o.scale || 2)), width: size.w, height: size.h, background: o.transparent ? 'transparent' : undefined }); }
          finally { scene.background = prevBg; }
          updateOutlineResolution();
          requestRender();
          resolve(dataURLToBlob(r.dataURL));
        } catch (err) { reject(err); }
      });
    }

    function dispose() {
      if (disposed) return;
      clearResult();
      disposed = true;
      if (rafId) { (root.cancelAnimationFrame || clearTimeout)(rafId); rafId = 0; }
      hoverT.cancel();
      cleanups.forEach(function (f) { try { f(); } catch (_) {} });
      cleanups = [];
      if (controls) { controls.removeEventListener('change', onControlsChange); controls.dispose(); controls = null; }
      try { renderer.dispose(); renderer.forceContextLoss(); } catch (_) {}
      if (wrap.parentNode) wrap.parentNode.removeChild(wrap);
      ev.emit('disposed', null);
      ev.clear();
      S.result = null;
    }

    var viewer = {
      kind: kind,
      element: wrap,
      on: ev.on, off: ev.off,
      setResult: setResult,
      result: function () { return S.result; },
      fields: fieldIds,
      field: function () { return S.fieldId; },
      supports: function () { return S.adapter ? { fields: fieldIds(), tools: (S.adapter.supports.tools || []).slice(), layers: Object.keys(S.layers || {}) } : { fields: [], tools: [], layers: [] }; },
      setField: setField,
      setLighting: setLighting,
      setLayers: setLayers,
      setBranchVisibility: setBranchVisibility,
      fit: fit,
      standardView: function (name, o) { return standardView(name, Object.assign({ animate: true }, o || {})); },
      hasAnatomicalFrame: function () { return !!frameOf(); },
      frame: frameOf,
      select: select,
      highlight: highlight,
      setMarkers: setMarkers,
      setCursor: setCursor,
      cursorStation: function () {
        var ch = cursorHelper(), st = S.cursor && ch ? ch.station(S.cursor.segmentId, S.cursor.s_mm) : null;
        if (st && cursorGroup) st.ring = cursorGroup.userData.ring;   // 'outline' (lumen cut) or 'circle' (inscribed radius)
        return st;
      },
      getState: getState,
      applyState: applyState,
      getCamera: cameraState,
      setCamera: function (c, o) { fitName = null; return setCamera(c, o); },
      setInsets: function (ins) { insets = Object.assign({}, insets, ins || {}); if (fitName) standardView(fitName, { animate: false, keepFit: true }); },
      insets: function () { return Object.assign({}, insets); },
      colorbarInfo: colorbarInfo,
      snapshot: snapshot,
      pickAt: pickAt,
      canvasElement: canvas,
      toolOverlay: toolOverlay,
      rayAt: toolRay,
      projectPoint: projectPoint,
      mmPerPixelAt: mmPerPixelAt,
      surfaceAt: surfaceAt,
      setControlsEnabled: setControlsEnabled,
      setClipPlane: setClipPlane,
      setToolLabels: setToolLabels,
      // lane A exports
      exportPose: exportPose, renderPose: renderPose,
      resize: function () { size = { w: 0, h: 0 }; resize(); },
      render: function () { requestRender(); },
      renderNow: function () { if (rafId) { (root.cancelAnimationFrame || clearTimeout)(rafId); rafId = 0; } frame(); },
      stats: function () { var i = renderer.info; return { geometries: i.memory.geometries, textures: i.memory.textures, calls: i.render.calls, triangles: i.render.triangles, programs: (i.programs || []).length }; },
      // lane E exports
      refreshDisplay: refreshDisplay,
      peakMarkers: function () { return peakGroup ? peakGroup.children.map(function (g) { return g.position.toArray(); }) : []; },
      // P3 lane 4 exports
      p3Display: function () { return { contours: p3Lines, top: p3Top ? Object.assign({}, p3Top) : null, clipZ: clipZ }; },
      refreshP3: function () { if (S.handle) drawP3(); }, refreshClip: function () { if (S.handle) applyWallClip(); },
      dispose: dispose,
      isDisposed: function () { return disposed; }
    };
    return viewer;
  }

  // Camera sync between two viewers in world coordinates; events that came from the link are not echoed back.
  function link(a, b) {
    if (!a || !b || a === b) return function () {};
    var busy = false;
    function relay(to) {
      return function (e) {
        if (!e || e.source === 'link' || busy) return;
        busy = true;
        try { to.setCamera(e.camera, { source: 'link' }); } finally { busy = false; }
      };
    }
    var offA = a.on('camera', relay(b)), offB = b.on('camera', relay(a));
    try { b.setCamera(a.getCamera(), { source: 'link' }); } catch (_) {}
    return function unlink() { offA(); offB(); };
  }

  return {
    create: create, webglAvailable: webglAvailable, link: link, gfx: gfx,
    VIEW_NAMES: VIEW_NAMES.slice(), VIEW_LABELS: VIEW_LABELS,
    // pure helpers (tests, orientation widget)
    viewDirection: viewDirection, anatomicalAxes: anatomicalAxes, cameraBasis: cameraBasis, insetFit: insetFit, validFrame: validFrame,
    normalizeLayers: normalizeLayers, DEFAULT_LAYERS: DEFAULT_LAYERS, sanitizeState: sanitizeState, cleanCamera: cleanCamera
  };
});
