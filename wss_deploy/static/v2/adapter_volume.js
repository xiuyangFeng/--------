/* WSSV2 adapter · volume (contract §3 体场, §6.2).
   pressure / speed: interior prediction points coloured from the read array inside a translucent vessel
   (a rim-shaded glass surface so the outline stays readable); wall_pressure: the display mesh coloured per
   vertex (opaque, silhouette outline); streamlines: a line layer coloured by speed when speed is the active
   field, neutral otherwise (one colour bar at a time).  Picking: nearest interior point along the ray; on the
   coloured wall the hit vertex (wall_pressure has no read array: its display values are the data). */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.adapters = ns.adapters || {};
  ns.adapters.volume = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  function geo(m) { return m && m.geometry || {}; }
  function isPointField(f, result) {
    return f && f.kind === 'scalar' && (Number(f.components) || 1) === 1 && f.arrays && typeof f.arrays.read === 'string' && result.declared(f.arrays.read) &&
      !(typeof f.arrays.display === 'string' && f.arrays.display);
  }
  function isWallField(f, result) {
    return f && f.kind === 'scalar' && (Number(f.components) || 1) === 1 && f.arrays && typeof f.arrays.display === 'string' && result.declared(f.arrays.display);
  }
  function displayable(result) {
    return result.fields().filter(function (f) { return isPointField(f, result) || isWallField(f, result); }).map(function (f) { return f.id; });
  }
  function defaultField(result, prev) {
    var ids = displayable(result);
    if (prev && ids.indexOf(prev) >= 0) return prev;
    if (ids.indexOf('speed') >= 0) return 'speed';
    return ids[0] || null;
  }
  function keyOr(result, a, b) { return typeof a === 'string' && a && result.declared(a) ? a : (b && result.declared(b) ? b : null); }
  function keys(result) {
    var g = geo(result.manifest), dm = g.display_mesh || {}, p = g.points || {}, sl = g.streamlines || {};
    return {
      vertices: keyOr(result, dm.vertices), faces: keyOr(result, dm.faces), mtrust: keyOr(result, dm.trust),
      pxyz: keyOr(result, p.xyz, 'vpts'), pseg: keyOr(result, p.segment, 'vseg'), pwall: keyOr(result, p.is_wall, 'vis_wall'),
      ptrust: keyOr(result, p.trust, 'vtrust'), ps: keyOr(result, p.s_from_root_mm, 'v_s'),
      slxyz: keyOr(result, sl.xyz, 'sl.xyz'), slspeed: keyOr(result, sl.speed, 'sl.speed'), sloff: keyOr(result, sl.offsets, 'sl.offsets')
    };
  }
  function requiredArrays(result, fieldId) {
    var k = keys(result), out = [k.vertices, k.faces, k.mtrust, k.pxyz, k.pseg, k.pwall, k.ps].filter(Boolean);
    if (ns.cursor) ns.cursor.requiredArrays(result).forEach(function (x) { out.push(x); });
    if (fieldId) result.fieldKeys(fieldId).forEach(function (x) { if (result.declared(x)) out.push(x); });
    return out;
  }
  function optionalArrays(result) { var k = keys(result); return [k.slxyz, k.slspeed, k.sloff, k.ptrust].filter(Boolean); }
  function opt(result, key) { return key && result.has(key) ? result.array(key) : null; }
  function fail(msg) { throw (ns.data && ns.data.DataError ? ns.data.DataError(msg, { code: 'geometry' }) : new Error(msg)); }

  // Phase 3 lane 4 (classic 可信区域, V12): with the trust layer on, interior points flagged geometry out of range (4),
  // weak sample support (8) or near an opening (16) are desaturated exactly as the classic desaturate(c, 0.7):
  // c · 0.3 + 0.62 · 0.7.  Only colours change.
  var TRUST_INTERIOR = 4 | 8 | 16, TRUST_DIM = [0.62, 0.62, 0.62, 0.3];
  // The glass takes the viewer's clipping planes (the section tool's cut), hence the clipping chunks.  uGain scales its
  // opacity (lane E 外壁不透明度: the classic slider value / its default 0.1, so the default look is unchanged).
  var OPACITY_DEFAULT = 0.1;   // classic volume-opacity default (0 … 0.5)
  function glassMaterial(THREE) {
    return new THREE.ShaderMaterial({
      clipping: true,
      uniforms: { uColor: { value: new THREE.Color('#a9b6c4') }, uBase: { value: 0.07 }, uRim: { value: 0.55 }, uGain: { value: 1 } },   // mid grey: reads on a dark and on a light stage
      vertexShader: '#include <clipping_planes_pars_vertex>\nvarying vec3 vN; varying vec3 vV; void main(){ vec4 mvPosition = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mvPosition.xyz); gl_Position = projectionMatrix * mvPosition;\n#include <clipping_planes_vertex>\n}',
      fragmentShader: '#include <clipping_planes_pars_fragment>\nuniform vec3 uColor; uniform float uBase; uniform float uRim; uniform float uGain; varying vec3 vN; varying vec3 vV;' +
        'void main(){\n#include <clipping_planes_fragment>\n float f = 1.0 - abs(dot(normalize(vN), normalize(vV))); gl_FragColor = vec4(uColor, clamp((uBase + uRim * f * f * f) * uGain, 0.0, 0.9)); }',
      transparent: true, depthWrite: false, side: THREE.DoubleSide
    });
  }
  // Classic planeBasis (unit u ⟂ direction), for the arrow heads; VolumeViewerCore's when loaded.
  function sideOf(d) {
    var VC = root.VolumeViewerCore;
    if (VC && typeof VC.planeBasis === 'function') return VC.planeBasis(d).u;
    var a = Math.abs(d[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    var u = [a[1] * d[2] - a[2] * d[1], a[2] * d[0] - a[0] * d[2], a[0] * d[1] - a[1] * d[0]], n = Math.hypot(u[0], u[1], u[2]) || 1;
    return [u[0] / n, u[1] / n, u[2] / n];
  }
  // Velocity arrows on interior points (classic addVectors): at most 600 (stride ⌈n / 600⌉ over the shown points),
  // length diag × 0.025 × √(|v| / top of the speed scale), a two-stroke head, coloured by |v| on that scale.
  // Returns {positions, colors, count} (Float32Arrays of segment ends).
  function arrowSegments(P, VEL, indices, diag, scale) {
    var n = indices.length, stride = Math.max(1, Math.ceil(n / 600)), top = Math.max(scale.range[1], 1e-9), pos = [], col = [], count = 0;
    for (var j = 0; j < n; j += stride) {
      var i = indices[j], vx = VEL[3 * i], vy = VEL[3 * i + 1], vz = VEL[3 * i + 2], len = Math.sqrt(vx * vx + vy * vy + vz * vz);
      if (!(len >= 1e-8)) continue;
      var d = [vx / len, vy / len, vz / len], size = diag * 0.025 * Math.sqrt(len / top);
      var s = [P[3 * i], P[3 * i + 1], P[3 * i + 2]], e = [s[0] + d[0] * size, s[1] + d[1] * size, s[2] + d[2] * size];
      var u = sideOf(d), b = [e[0] - d[0] * size * 0.24, e[1] - d[1] * size * 0.24, e[2] - d[2] * size * 0.24], w = size * 0.1;
      var c = scale.color(len);
      [[s, e], [e, [b[0] + u[0] * w, b[1] + u[1] * w, b[2] + u[2] * w]], [e, [b[0] - u[0] * w, b[1] - u[1] * w, b[2] - u[2] * w]]].forEach(function (seg) {
        pos.push(seg[0][0], seg[0][1], seg[0][2], seg[1][0], seg[1][1], seg[1][2]); col.push(c[0], c[1], c[2], c[0], c[1], c[2]);
      });
      count++;
    }
    return { positions: new Float32Array(pos), colors: new Float32Array(col), count: count };
  }

  function build(result, ctx) {
    var THREE = ctx.THREE, gfx = ctx.gfx, util = ns.util, m = result.manifest, k = keys(result);
    if (!k.vertices || !k.faces || !k.pxyz) fail('volume result lacks display mesh or volume points');
    var V = result.array(k.vertices), F0 = result.array(k.faces), P = result.array(k.pxyz);
    var MT = opt(result, k.mtrust), PSEG = opt(result, k.pseg), PWALL = opt(result, k.pwall), PS = opt(result, k.ps);
    var nV = Math.floor(V.length / 3), nP = Math.floor(P.length / 3), i;
    var F = F0 instanceof Uint32Array ? F0 : Uint32Array.from(F0);
    for (i = 0; i < F.length; i++) if (F[i] >= nV) fail('display mesh face index out of range');
    if (PSEG && PSEG.length !== nP) fail('point segment length ≠ points');
    if (PWALL && PWALL.length !== nP) fail('point wall flag length ≠ points');
    if (MT && MT.length !== nV) fail('mesh trust length ≠ vertices');

    var group = new THREE.Group(); group.name = 'volume'; ctx.root.add(group);
    var mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (i = 0; i < V.length; i += 3) for (var c = 0; c < 3; c++) { var v = V[i + c]; if (v < mn[c]) mn[c] = v; if (v > mx[c]) mx[c] = v; }
    var size = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1);
    var diag = Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) || size;
    var bounds = { min: mn, max: mx, center: [0, 1, 2].map(function (j) { return (mn[j] + mx[j]) / 2; }), size: size };

    // wall surface: glass (context) or coloured (wall_pressure)
    var geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(V, 3));
    geom.setIndex(new THREE.BufferAttribute(F, 1));
    geom.computeVertexNormals();
    var wcolors = new Float32Array(3 * nV), wcolorAttr = new THREE.BufferAttribute(wcolors, 3);
    geom.setAttribute('color', wcolorAttr);
    var flag = new Float32Array(nV); if (MT) for (i = 0; i < nV; i++) flag[i] = MT[i] ? 1 : 0;
    geom.setAttribute('aFlag', new THREE.BufferAttribute(flag, 1));
    var glass = glassMaterial(THREE);
    var flat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    var soft = new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide, shininess: 26, specular: new THREE.Color(0x262626) });
    var wall = new THREE.Mesh(geom, glass); wall.name = 'volume-wall'; wall.renderOrder = 3; group.add(wall);
    var outlineMat = gfx.outlineMaterial({ sign: gfx.windingSign(V, F), width: 1.4, color: gfx.INK_HEX });
    outlineMat.userData.baseWidth = 1.4;
    var outline = new THREE.Mesh(geom, outlineMat); outline.name = 'volume-outline'; outline.visible = false; group.add(outline);
    var hatchMat = gfx.hatchMaterial({ color: gfx.INK_HEX, opacity: 0.6, spacing: 7, line: 2 });
    hatchMat.userData.basePx = { spacing: 7, line: 2 };
    var hatch = new THREE.Mesh(geom, hatchMat); hatch.visible = false; hatch.renderOrder = 4; group.add(hatch);

    // interior points
    var interior = [];
    for (i = 0; i < nP; i++) if (!PWALL || !PWALL[i]) interior.push(i);
    var interiorIdx = Uint32Array.from(interior), nI = interiorIdx.length;
    var ipos = new Float32Array(3 * nI);
    for (i = 0; i < nI; i++) { var q = interiorIdx[i]; ipos[3 * i] = P[3 * q]; ipos[3 * i + 1] = P[3 * q + 1]; ipos[3 * i + 2] = P[3 * q + 2]; }
    var pgeom = new THREE.BufferGeometry();
    pgeom.setAttribute('position', new THREE.BufferAttribute(ipos, 3));
    var pcolors = new Float32Array(3 * nI), pcolorAttr = new THREE.BufferAttribute(pcolors, 3);
    pgeom.setAttribute('color', pcolorAttr);
    var dotTex = gfx.roundPointTexture();
    // Point diameter in mm (world).  three.js r128 attenuates as size · (H/2) / depth, without the 1 / tan(fov/2)
    // of the true projection, so the material size is the world size divided by tan(fov/2).
    var pointSize = diag / 190, fovK = 1 / Math.tan((ctx.fov || 30) * Math.PI / 360);
    var pmat = new THREE.PointsMaterial({ size: pointSize * fovK, sizeAttenuation: true, vertexColors: true, map: dotTex, alphaTest: 0.5 });
    var cloud = new THREE.Points(pgeom, pmat); cloud.name = 'volume-points'; cloud.renderOrder = 1; group.add(cloud);
    var ivals = new Float32Array(nI);

    // streamlines (loaded in the background; built when available)
    var lines = null, lineColors = null, lineSpeed = null;
    var neutralLine = new THREE.LineBasicMaterial({ color: new THREE.Color('#465361'), transparent: true, opacity: 0.55 });
    var colouredLine = new THREE.LineBasicMaterial({ vertexColors: true });
    function buildStreamlines() {
      if (lines) return true;
      var X = opt(result, k.slxyz), S = opt(result, k.slspeed), O = opt(result, k.sloff);
      if (!X || !O) return false;
      var nL = Math.floor(X.length / 3);
      var lg = new THREE.BufferGeometry();
      lg.setAttribute('position', new THREE.BufferAttribute(X, 3));
      lineColors = new Float32Array(3 * nL);
      lg.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
      lineSpeed = S && S.length === nL ? S : null;
      lines = new THREE.LineSegments(lg, neutralLine); lines.name = 'volume-streamlines'; lines.renderOrder = 2; lines.visible = false;
      group.add(lines);
      applyLineFilter();
      return true;
    }
    // Lane E display: streamline density (every n-th line, classic stride 1 / 2 / 3 / 5) and width (classic lit tubes,
    // radius = diagonal / 900 × width; 细线 keeps the line layer), glass opacity, velocity arrows, input STL.
    var diagC = Math.max(Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]), 1);   // classic `diagonal`
    var dsp = { stl: false, opacity: OPACITY_DEFAULT, density: 1, width: 1, thin: true, vectors: false };
    var tubes = null, tubeKey = '', arrows = null, arrowKey = '';
    var tubeMat = new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 35, specular: new THREE.Color(0x333333) });
    // uncoloured tubes (another field is shown) stay in the background, like the neutral line layer
    var tubeGreyMat = new THREE.MeshPhongMaterial({ color: new THREE.Color('#8a94a6'), shininess: 20, transparent: true, opacity: 0.4, depthWrite: false });
    var arrowMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 });
    var stlMat = new THREE.MeshPhongMaterial({ color: new THREE.Color(0.72, 0.75, 0.8), side: THREE.DoubleSide, shininess: 20, specular: new THREE.Color(0x1f1f1f) });
    function lineKept(l, ls) { return l % dsp.density === 0 && !(ls && ls[l] >= 0 && hidden && hidden(ls[l])); }
    function disposeTubes() { if (tubes) { tubes.children.forEach(function (o) { o.geometry.dispose(); }); group.remove(tubes); tubes = null; tubeKey = ''; } }
    function buildTubes() {
      var X = opt(result, k.slxyz), O = opt(result, k.sloff);
      if (!X || !O) return;
      var key = [dsp.density, dsp.width, hidden ? branchVersion : 0].join('|');
      if (tubes && key === tubeKey) return;
      disposeTubes();
      var nL = Math.floor(X.length / 3), ls = hidden ? lineSegments() : null, radius = diagC / 900 * dsp.width;
      tubes = new THREE.Group(); tubes.name = 'volume-streamtubes'; tubes.renderOrder = 2;
      for (var l = 0; l + 1 < O.length; l++) {
        if (!lineKept(l, ls)) continue;
        var a = O[l], b = Math.min(O[l + 1], nL), n = b - a;
        if (n < 2) continue;
        var pts = [];
        for (var j = a; j < b; j++) pts.push(new THREE.Vector3(X[3 * j], X[3 * j + 1], X[3 * j + 2]));
        var segs = Math.max(2, Math.min(n - 1, 160)), radial = 5;
        var tg = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts, false, 'centripetal'), segs, radius, radial, false);
        tg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(tg.attributes.position.count * 3), 3));
        var mesh = new THREE.Mesh(tg, tubeMat);
        mesh.userData.line = { a: a, n: n, segs: segs, radial: radial };
        tubes.add(mesh);
      }
      group.add(tubes);
      tubeKey = key;
      colorTubes();
    }
    // Rings of a tube take the speed of the line vertex at the same fraction (classic); neutral grey unless the field is speed.
    function colorTubes() {
      if (!tubes) return;
      var coloured = Boolean(cur.fieldId === 'speed' && lineSpeed && cur.scale);
      tubes.children.forEach(function (mesh) {
        mesh.material = coloured ? tubeMat : tubeGreyMat;
        if (!coloured) return;
        var L = mesh.userData.line, col = mesh.geometry.attributes.color.array;
        for (var i = 0; i <= L.segs; i++) {
          var c = cur.scale.color(lineSpeed[L.a + Math.round(i / L.segs * (L.n - 1))]);
          for (var j = 0; j <= L.radial; j++) { var o = 3 * (i * (L.radial + 1) + j); col[o] = c[0]; col[o + 1] = c[1]; col[o + 2] = c[2]; }
        }
        mesh.geometry.attributes.color.needsUpdate = true;
      });
    }
    function disposeArrows() { if (arrows) { arrows.geometry.dispose(); group.remove(arrows); arrows = null; arrowKey = ''; } }
    function buildArrows() {
      var VEL = null, f = result.field('velocity');
      if (f && f.arrays && typeof f.arrays.read === 'string' && result.has(f.arrays.read)) VEL = result.array(f.arrays.read);
      if (!dsp.vectors || dsp.stl || !VEL || VEL.length !== 3 * nP || cur.fieldId !== 'speed' || !cur.scale) { disposeArrows(); return false; }   // also with the points hidden: arrows alone read better
      var key = [cur.scale.range.join(','), cur.scale.cmap, cur.scale.log, cur.scale.bands, hidden ? branchVersion : 0].join('|');
      if (arrows && key === arrowKey) return true;
      disposeArrows();
      var shown = [];
      for (var j = 0; j < nI; j++) { var q = interiorIdx[j]; if (!hidden || !pointMask || pointMask[q]) shown.push(q); }
      var seg = arrowSegments(P, VEL, shown, diagC, cur.scale);
      var ag = new THREE.BufferGeometry();
      ag.setAttribute('position', new THREE.BufferAttribute(seg.positions, 3));
      ag.setAttribute('color', new THREE.BufferAttribute(seg.colors, 3));
      arrows = new THREE.LineSegments(ag, arrowMat); arrows.name = 'volume-arrows'; arrows.renderOrder = 5; arrows.userData.count = seg.count;
      group.add(arrows);
      arrowKey = key;
      return true;
    }
    var branchVersion = 0;

    // centreline
    var clObj = null, clKeys = geo(m).centerline || {}, CV = opt(result, clKeys.xyz), CE = opt(result, clKeys.edges), CS = opt(result, clKeys.segment);
    if (CV && CE) {
      var cg = new THREE.BufferGeometry();
      cg.setAttribute('position', new THREE.BufferAttribute(CV, 3));
      cg.setIndex(new THREE.BufferAttribute(CE instanceof Uint32Array ? CE : Uint32Array.from(CE), 1));
      clObj = new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ color: new THREE.Color('#465361'), depthTest: false, transparent: true, opacity: 0.85 }));
      clObj.renderOrder = 6; clObj.visible = false; group.add(clObj);
    }

    var cur = { fieldId: null, scale: null, wallField: false, read: null, disp: null };
    var L = {}, lighting = 'flat', hidden = null, pointMask = null, hlObj = null, trustMask = null;
    // 1 = keep the colour, 0 = grey (interior order); null while the trust array is not loaded or nothing is flagged.
    function interiorTrustMask() {
      if (trustMask !== null) return trustMask || null;
      var PT = opt(result, k.ptrust);
      if (!PT || PT.length !== nP) return null;
      var mk = new Uint8Array(nI), any = false;
      for (var j = 0; j < nI; j++) { var bad = (PT[interiorIdx[j]] & TRUST_INTERIOR) !== 0; mk[j] = bad ? 0 : 1; if (bad) any = true; }
      trustMask = any ? mk : false;
      return trustMask || null;
    }
    function applyVisibility() {
      var wallOn = L.wall !== false || dsp.stl, opaque = cur.wallField || dsp.stl;
      wall.visible = wallOn;
      wall.material = dsp.stl ? stlMat : cur.wallField ? (lighting === 'soft' ? soft : flat) : glass;
      wall.renderOrder = opaque ? 0 : 3;
      glass.uniforms.uRim.value = L.outline ? 0.55 : 0.12;
      glass.uniforms.uGain.value = Math.max(0, dsp.opacity) / OPACITY_DEFAULT;
      outline.visible = wallOn && opaque && !!L.outline;
      hatch.visible = wallOn && cur.wallField && !dsp.stl && !!L.trust && !!MT;
      cloud.visible = !opaque && L.interior !== false;
      if (L.streamlines && !opaque) buildStreamlines();
      var tubesOn = Boolean(L.streamlines && !opaque && !dsp.thin);
      if (lines) {
        lines.visible = !!L.streamlines && !opaque && dsp.thin;
        var coloured = cur.fieldId === 'speed' && lineSpeed;
        lines.material = coloured ? colouredLine : neutralLine;
      }
      if (tubesOn && lines) buildTubes();
      if (tubes) tubes.visible = tubesOn;
      buildArrows();
      if (clObj) clObj.visible = !!L.centerline;
    }
    function recolor() {
      if (!cur.scale) return;
      if (cur.wallField) { cur.scale.fill(cur.disp, wcolors, null, null); wcolorAttr.needsUpdate = true; }
      else {
        for (var j = 0; j < nI; j++) ivals[j] = cur.read[interiorIdx[j]];
        var tm = L.trust ? interiorTrustMask() : null;
        cur.scale.fill(ivals, pcolors, tm, tm ? TRUST_DIM : null); pcolorAttr.needsUpdate = true;
      }
      if (lines && lineSpeed && cur.fieldId === 'speed') { cur.scale.fill(lineSpeed, lineColors, null, null); lines.geometry.attributes.color.needsUpdate = true; }
      colorTubes();
    }
    function setField(fieldId, scale) {
      var f = result.field(fieldId);
      var wallField = isWallField(f, result);
      var disp = wallField ? result.fieldArray(fieldId, 'display') : null, read = result.fieldArray(fieldId, 'read');
      if (wallField && disp.length !== nV) fail('field ' + fieldId + ': display array length ≠ mesh vertices');
      if (!wallField && (!read || read.length !== nP)) fail('field ' + fieldId + ': read array length ≠ volume points');
      cur = { fieldId: fieldId, scale: scale, wallField: wallField, read: read, disp: disp };
      recolor();
      applyVisibility();
    }
    function setLighting(mode) { lighting = mode === 'soft' ? 'soft' : 'flat'; applyVisibility(); }
    function setLayers(layers) {
      var trustBefore = Boolean(L.trust);
      L = Object.assign({}, layers);
      applyVisibility();
      if ((L.streamlines && lines && cur.fieldId === 'speed') || Boolean(L.trust) !== trustBefore) recolor();
      var hasLines = !!(k.slxyz && k.sloff);
      return { streamlines: !!L.streamlines && hasLines, trust: !!L.trust && (!!MT || !!k.ptrust), centerline: !!L.centerline && !!clObj, points: false };
    }
    // Hidden branches (classic volume report): the interior points of those branches, the wall faces whose three
    // vertices all belong to them (vertex branch = that of the nearest wall prediction point, else of the nearest
    // centreline sample) and the streamlines whose middle vertex is nearest an interior point of them.
    var vertexSeg = null, lineSeg = null;
    function vertexSegments() {
      var VC = root.VolumeViewerCore;
      if (vertexSeg || !VC || typeof VC.nearestLabels !== 'function') return vertexSeg;
      var wallIdx = [];
      if (PWALL && PSEG) for (var j = 0; j < nP; j++) if (PWALL[j]) wallIdx.push(j);
      if (wallIdx.length) {
        var src = new Float32Array(wallIdx.length * 3), lab = new Int32Array(wallIdx.length);
        wallIdx.forEach(function (q, k) { src[3 * k] = P[3 * q]; src[3 * k + 1] = P[3 * q + 1]; src[3 * k + 2] = P[3 * q + 2]; lab[k] = PSEG[q]; });
        vertexSeg = VC.nearestLabels(src, lab, V);
      } else if (CV && CS) vertexSeg = VC.nearestLabels(CV, CS, V);
      return vertexSeg;
    }
    function lineSegments() {
      if (lineSeg || !lines || !PSEG) return lineSeg;
      var X = opt(result, k.slxyz), O = opt(result, k.sloff), nL = X ? Math.floor(X.length / 3) : 0, g = util.gridIndex(ipos);
      lineSeg = [];
      for (var l = 0; l + 1 < O.length; l++) {
        var a = O[l], b = Math.min(O[l + 1], nL), n = b - a;
        if (n <= 0) { lineSeg.push(-1); continue; }
        var mid = a + Math.floor(n / 2), q = g.nearest(X[3 * mid], X[3 * mid + 1], X[3 * mid + 2]);
        lineSeg.push(q >= 0 ? Number(PSEG[interiorIdx[q]]) : -1);
      }
      return lineSeg;
    }
    function applyLineFilter() {
      if (!lines) return;
      var X = opt(result, k.slxyz), O = opt(result, k.sloff), nL = Math.floor(X.length / 3), idx = [], ls = hidden ? lineSegments() : null;
      for (var l = 0; l + 1 < O.length; l++) {
        if (!lineKept(l, ls)) continue;
        var a = O[l], b = Math.min(O[l + 1], nL); for (var j = a; j + 1 < b; j++) idx.push(j, j + 1);
      }
      lines.geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
    }
    function setBranchVisibility(set) {
      hidden = set && PSEG ? function (sid) { return !set.has(Number(sid)); } : null;
      branchVersion++;
      var vs = hidden ? vertexSegments() : null;
      if (!hidden || !vs) geom.setIndex(new THREE.BufferAttribute(F, 1));
      else {
        var kf = [];
        for (var t = 0; t < F.length; t += 3) { if (hidden(vs[F[t]]) && hidden(vs[F[t + 1]]) && hidden(vs[F[t + 2]])) continue; kf.push(F[t], F[t + 1], F[t + 2]); }
        geom.setIndex(new THREE.BufferAttribute(new Uint32Array(kf), 1));
      }
      applyLineFilter();
      if (!hidden) { pgeom.setIndex(null); pointMask = null; }
      else {
        var keep = []; pointMask = new Uint8Array(nP);
        for (var j = 0; j < nI; j++) { var q = interiorIdx[j]; if (!hidden(PSEG[q])) { keep.push(j); pointMask[q] = 1; } }
        pgeom.setIndex(new THREE.BufferAttribute(new Uint32Array(keep), 1));
      }
      pgeom.computeBoundingSphere();
      applyVisibility();
    }
    function pick(raycaster) {
      if (cur.wallField) {
        if (!wall.visible) return null;
        var hits = raycaster.intersectObject(wall, false);
        if (!hits.length) return null;
        var h = hits[0], p = h.point, best = -1, bd = Infinity;
        [h.face.a, h.face.b, h.face.c].forEach(function (vi) { var dx = V[3 * vi] - p.x, dy = V[3 * vi + 1] - p.y, dz = V[3 * vi + 2] - p.z, d = dx * dx + dy * dy + dz * dz; if (d < bd) { bd = d; best = vi; } });
        return { pointIndex: null, vertexIndex: best, xyz: [V[3 * best], V[3 * best + 1], V[3 * best + 2]], hitXyz: [p.x, p.y, p.z], segmentId: null, s_from_root_mm: null,
          value: Number.isFinite(cur.disp[best]) ? cur.disp[best] : NaN, valueSource: 'display', values: {}, trust: MT ? MT[best] : 0 };
      }
      if (!cloud.visible) return null;
      raycaster.params.Points = { threshold: pointSize * 0.6 };
      var ph = raycaster.intersectObject(cloud, false);
      if (!ph.length) return null;
      var hit = ph[0], qi = interiorIdx[hit.index];
      var values = {};
      result.fields().forEach(function (f) { if (isPointField(f, result) && result.has(f.arrays.read)) values[f.id] = result.valueAt(f.id, qi); });
      return {
        pointIndex: qi, vertexIndex: null, xyz: [P[3 * qi], P[3 * qi + 1], P[3 * qi + 2]], hitXyz: [hit.point.x, hit.point.y, hit.point.z],
        segmentId: PSEG ? Number(PSEG[qi]) : null, s_from_root_mm: PS && Number.isFinite(PS[qi]) ? PS[qi] : null,
        value: Number.isFinite(cur.read[qi]) ? cur.read[qi] : NaN, valueSource: 'read', values: values, vector: result.vectorAt('velocity', qi)
      };
    }
    // The wall surface under a ray whatever the field (the section tool places its points on the glass wall).
    function pickSurface(raycaster) {
      var hits = raycaster.intersectObject(wall, false);
      if (!hits.length) return null;
      var h = hits[0], p = h.point, best = h.face.a, bd = Infinity;
      [h.face.a, h.face.b, h.face.c].forEach(function (vi) { var dx = V[3 * vi] - p.x, dy = V[3 * vi + 1] - p.y, dz = V[3 * vi + 2] - p.z, d = dx * dx + dy * dy + dz * dz; if (d < bd) { bd = d; best = vi; } });
      return { xyz: [p.x, p.y, p.z], vertexIndex: best };
    }
    function highlight(indices, o) {
      if (hlObj) { gfx.disposeObject(hlObj); hlObj = null; }
      if (!indices || !indices.length) return;
      var pos = [];
      for (var j = 0; j < indices.length; j++) { var q = indices[j]; if (q >= 0 && q < nP) pos.push(P[3 * q], P[3 * q + 1], P[3 * q + 2]); }
      var g2 = new THREE.BufferGeometry(); g2.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
      // o.size: diameter relative to the interior points (1.8 by default), o.opacity < 1: see-through (a region)
      var op = o && Number(o.opacity) > 0 && Number(o.opacity) < 1 ? Number(o.opacity) : 1;
      hlObj = new THREE.Points(g2, new THREE.PointsMaterial({ color: new THREE.Color(o && o.color || gfx.ACCENT_HEX), size: pointSize * (o && Number(o.size) > 0 ? Number(o.size) : 1.8) * fovK,
        sizeAttenuation: true, map: dotTex, alphaTest: op < 1 ? 0.1 : 0.5, transparent: op < 1, opacity: op, depthWrite: op >= 1 }));
      hlObj.renderOrder = 7;
      group.add(hlObj);
    }
    // o = {stl, opacity (classic 0 … 0.5), density (1 / 2 / 3 / 5), width (0.4 … 3), thin, vectors}; returns what is shown.
    function setDisplay(o) {
      o = o || {};
      var d = Math.floor(Number(o.density)), w = Number(o.width), op = Number(o.opacity);
      var next = { stl: Boolean(o.stl), opacity: Number.isFinite(op) ? Math.max(0, Math.min(0.5, op)) : OPACITY_DEFAULT,
        density: [1, 2, 3, 5].indexOf(d) >= 0 ? d : 1, width: Number.isFinite(w) ? Math.max(0.4, Math.min(3, w)) : 1, thin: o.thin !== false, vectors: Boolean(o.vectors) };
      if (next.density !== dsp.density) { dsp.density = next.density; applyLineFilter(); }
      dsp = next;
      applyVisibility();
      return { stl: dsp.stl, arrows: arrows ? arrows.userData.count : 0, tubes: tubes && tubes.visible ? tubes.children.length : 0 };
    }
    function histogramValues(fieldId) {
      var f = result.field(fieldId);
      if (isWallField(f, result)) return { values: result.fieldArray(fieldId, 'display'), source: 'vertices', mask: null, scope: 'all' };
      var read = result.fieldArray(fieldId, 'read');
      if (!read) return null;
      var mask = pointMask;
      if (!mask && PWALL && nI < nP) { mask = new Uint8Array(nP); for (var j = 0; j < nI; j++) mask[interiorIdx[j]] = 1; }
      return { values: read, source: 'points', mask: mask, scope: pointMask ? 'visible' : 'all' };
    }
    return {
      family: 'volume',
      fields: displayable(result),
      bounds: bounds,
      setField: setField, setLighting: setLighting, setLayers: setLayers, setBranchVisibility: setBranchVisibility,
      pick: pick, pickSurface: pickSurface, highlight: highlight, histogramValues: histogramValues,
      setDisplay: setDisplay,
      displayState: function () { return { stl: dsp.stl, opacity: dsp.opacity, density: dsp.density, width: dsp.width, thin: dsp.thin, vectors: dsp.vectors,
        arrows: arrows ? arrows.userData.count : 0, tubes: tubes && tubes.visible ? tubes.children.length : 0 }; },
      fitPoints: function () { return V; },
      pointXYZ: function (q) { return q >= 0 && q < nP ? [P[3 * q], P[3 * q + 1], P[3 * q + 2]] : null; },
      sectionMesh: function () { return { vertices: V, faces: F }; },
      arraysLoaded: function () { if (L.streamlines) buildStreamlines(); applyVisibility(); if (L.streamlines || L.trust) recolor(); },
      trustGrey: function () { var tm = L.trust ? interiorTrustMask() : null; if (!tm) return 0; var n = 0; for (var j = 0; j < tm.length; j++) if (!tm[j]) n++; return n; },
      counts: { vertices: nV, faces: F.length / 3, points: nP, interior: nI },
      dispose: function () {
        gfx.disposeObject(group);
        [glass, flat, soft, neutralLine, colouredLine, tubeMat, tubeGreyMat, arrowMat, stlMat].forEach(function (mm) { mm.dispose(); });
        dotTex.dispose();
      }
    };
  }

  return {
    supports: { fields: ['interior', 'wall'], tools: ['pick', 'cursor', 'streamlines', 'outline', 'lighting', 'markers', 'highlight', 'branches', 'trust', 'slice', 'measure'] },
    displayable: displayable, defaultField: defaultField, requiredArrays: requiredArrays, optionalArrays: optionalArrays, build: build
  };
});
