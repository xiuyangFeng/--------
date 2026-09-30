/* WSSV2 adapter · wall (contract §6.2).
   Display mesh coloured per vertex from the field's display array (NaN → missing grey), flat by default with a
   silhouette outline, soft Lambert on request; trust hatch; prediction points; centreline.  Picking: ray →
   display mesh → nearest prediction point (uniform-grid index) → value from the field's read array. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.adapters = ns.adapters || {};
  ns.adapters.wall = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var HIGHLIGHT_DIM = [0.88, 0.89, 0.9, 0.3];   // non-highlighted vertices: 30 % colour, 70 % light grey

  function geo(m) { return m && m.geometry || {}; }
  function displayable(result) {
    return result.fields().filter(function (f) {
      return f && f.kind === 'scalar' && (Number(f.components) || 1) === 1 && f.arrays && typeof f.arrays.display === 'string' && f.arrays.display && result.declared(f.arrays.display);
    }).map(function (f) { return f.id; });
  }
  function defaultField(result, prev) {
    var ids = displayable(result);
    if (prev && ids.indexOf(prev) >= 0) return prev;
    return ids[0] || null;
  }
  function meshKeys(m) {
    var dm = geo(m).display_mesh || {}, pts = geo(m).points || {};
    return { vertices: dm.vertices, faces: dm.faces, segment: dm.segment, trust: dm.trust, pxyz: pts.xyz, pseg: pts.segment, ps: pts.s_from_root_mm };
  }
  function requiredArrays(result, fieldId) {
    var k = meshKeys(result.manifest), out = [];
    [k.vertices, k.faces, k.segment, k.trust, k.pxyz, k.pseg, k.ps].forEach(function (x) { if (typeof x === 'string' && x && result.declared(x)) out.push(x); });
    if (ns.cursor) ns.cursor.requiredArrays(result).forEach(function (x) { out.push(x); });
    if (fieldId) result.fieldKeys(fieldId).forEach(function (x) { if (result.declared(x)) out.push(x); });
    return out;
  }
  function opt(result, key) { return typeof key === 'string' && key && result.declared(key) && result.has(key) ? result.array(key) : null; }
  function fail(msg) { var e = ns.data && ns.data.DataError ? ns.data.DataError(msg, { code: 'geometry' }) : new Error(msg); throw e; }
  // Magenta hatch over the stagnation vertices (attribute aStag), stripes on the x − y diagonal of the screen (the
  // trust hatch runs on x + y).  uSpacing / uLine carry userData.basePx so the viewer scales them with the pixel ratio.
  var STAG_HEX = '#d91a8c';   // classic STAG_RGB (0.85, 0.10, 0.55)
  function stagnationMaterial(THREE) {
    var m = new THREE.ShaderMaterial({
      uniforms: { uColor: { value: new THREE.Color(STAG_HEX) }, uOpacity: { value: 0.85 }, uSpacing: { value: 6 }, uLine: { value: 2.2 } },
      vertexShader: 'attribute float aStag; varying float vStag; void main(){ vStag = aStag; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: 'uniform vec3 uColor; uniform float uOpacity; uniform float uSpacing; uniform float uLine; varying float vStag;' +
        'void main(){ if (vStag < 0.5) discard; float d = mod(gl_FragCoord.x - gl_FragCoord.y + 4096.0, uSpacing); if (d > uLine) discard; gl_FragColor = vec4(uColor, uOpacity); }',
      side: THREE.DoubleSide, transparent: true, depthWrite: false,
      polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4
    });
    m.userData.basePx = { spacing: 6, line: 2.2 };
    return m;
  }

  function build(result, ctx) {
    var THREE = ctx.THREE, gfx = ctx.gfx, util = ns.util, m = result.manifest;
    var k = meshKeys(m);
    if (!k.vertices || !k.faces || !k.pxyz) fail('wall result lacks display mesh or prediction points');
    var V = result.array(k.vertices), F0 = result.array(k.faces), PV = result.array(k.pxyz);
    var MS = opt(result, k.segment), MT = opt(result, k.trust), PS = opt(result, k.pseg), PSR = opt(result, k.ps);
    var nV = Math.floor(V.length / 3), nP = Math.floor(PV.length / 3);
    if (V.length !== 3 * nV || !nV) fail('display mesh vertices must be N×3');
    if (F0.length % 3) fail('display mesh faces must be M×3');
    var F = F0 instanceof Uint32Array ? F0 : Uint32Array.from(F0);
    for (var i = 0; i < F.length; i++) if (F[i] >= nV) fail('display mesh face index out of range');
    if (MS && MS.length !== nV) fail('display mesh segment length ≠ vertices');
    if (MT && MT.length !== nV) fail('display mesh trust length ≠ vertices');
    if (PS && PS.length !== nP) fail('point segment length ≠ points');

    var group = new THREE.Group();
    group.name = 'wall';
    ctx.root.add(group);

    // bounds
    var mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (i = 0; i < V.length; i += 3) for (var c = 0; c < 3; c++) { var v = V[i + c]; if (v < mn[c]) mn[c] = v; if (v > mx[c]) mx[c] = v; }
    var bounds = { min: mn, max: mx, center: [0, 1, 2].map(function (j) { return (mn[j] + mx[j]) / 2; }), size: Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1) };

    // surface
    var geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(V, 3));
    geom.setIndex(new THREE.BufferAttribute(F, 1));
    geom.computeVertexNormals();
    var colors = new Float32Array(3 * nV);
    var colorAttr = new THREE.BufferAttribute(colors, 3);
    geom.setAttribute('color', colorAttr);
    var flag = new Float32Array(nV);
    if (MT) for (i = 0; i < nV; i++) flag[i] = MT[i] ? 1 : 0;
    geom.setAttribute('aFlag', new THREE.BufferAttribute(flag, 1));
    var flat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    var soft = new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide, shininess: 26, specular: new THREE.Color(0x262626) });
    var mesh = new THREE.Mesh(geom, flat);
    mesh.name = 'wall-surface';
    group.add(mesh);
    var sign = gfx.windingSign(V, F);
    var outlineMat = gfx.outlineMaterial({ sign: sign, width: 1.4, color: gfx.INK_HEX });
    outlineMat.userData.baseWidth = 1.4;
    var outline = new THREE.Mesh(geom, outlineMat);
    outline.name = 'wall-outline';
    group.add(outline);
    var hatchMat = gfx.hatchMaterial({ color: gfx.INK_HEX, opacity: 0.6, spacing: 7, line: 2 });
    hatchMat.userData.basePx = { spacing: 7, line: 2 };
    var hatch = new THREE.Mesh(geom, hatchMat);
    hatch.name = 'wall-trust';
    hatch.renderOrder = 2;
    hatch.visible = false;
    group.add(hatch);

    // Lane E display options.  Input STL (classic 输入 STL view): the same display mesh in the classic neutral grey
    // (0.72, 0.75, 0.80), always lit so the shape reads; the field colours stay computed underneath.  Stagnation
    // (classic §19.2 overlay): display vertices with TAWSS < criterion and OSI > criterion carry a magenta hatch,
    // drawn in screen space like the trust hatch but on the other diagonal so the two stay distinguishable.
    var stlMat = new THREE.MeshPhongMaterial({ color: new THREE.Color(0.72, 0.75, 0.8), side: THREE.DoubleSide, shininess: 20, specular: new THREE.Color(0x1f1f1f) });
    var stag = new Float32Array(nV);
    geom.setAttribute('aStag', new THREE.BufferAttribute(stag, 1));
    var stagMat = stagnationMaterial(THREE);
    var stagMesh = new THREE.Mesh(geom, stagMat);
    stagMesh.name = 'wall-stagnation';
    stagMesh.renderOrder = 3;
    stagMesh.visible = false;
    group.add(stagMesh);
    var disp = { stl: false, stagnation: null }, stagState = 'off', lightMode = 'flat', layerState = {};

    // prediction points
    var pgeom = new THREE.BufferGeometry();
    pgeom.setAttribute('position', new THREE.BufferAttribute(PV, 3));
    var pcolors = new Float32Array(3 * nP);
    pgeom.setAttribute('color', new THREE.BufferAttribute(pcolors, 3));
    var dotTex = gfx.roundPointTexture();
    var pmat = new THREE.PointsMaterial({ size: 2, sizeAttenuation: false, vertexColors: true, map: dotTex, alphaTest: 0.5 });
    var pointsObj = new THREE.Points(pgeom, pmat);
    pointsObj.name = 'wall-points';
    pointsObj.visible = false;
    group.add(pointsObj);

    // centreline
    var clObj = null, clKeys = geo(m).centerline || {}, CV = opt(result, clKeys.xyz), CE = opt(result, clKeys.edges), CS = opt(result, clKeys.segment);
    if (CV && CE) {
      var cgeom = new THREE.BufferGeometry();
      cgeom.setAttribute('position', new THREE.BufferAttribute(CV, 3));
      cgeom.setIndex(new THREE.BufferAttribute(CE instanceof Uint32Array ? CE : Uint32Array.from(CE), 1));
      clObj = new THREE.LineSegments(cgeom, new THREE.LineBasicMaterial({ color: new THREE.Color('#465361'), depthTest: false, transparent: true, opacity: 0.85 }));
      clObj.renderOrder = 6;
      clObj.name = 'wall-centerline';
      clObj.visible = false;
      group.add(clObj);
    }

    // state
    var cur = { fieldId: null, scale: null, disp: null, read: null };
    var hidden = null, pointMask = null, vertexMask = null, visibleVerts = null;
    var hl = null, hlMask = null, grid = null, v2p = null;

    function pointGrid() {
      if (!grid) grid = util.gridIndex(PV, { cell: 1.5 });
      return grid;
    }
    function vertexToPoint() {
      if (!v2p) v2p = pointGrid().nearestMany(V, { maxDist: 6 });
      return v2p;
    }
    function recolor() {
      if (!cur.scale || !cur.disp) return;
      cur.scale.fill(cur.disp, colors, hlMask, hlMask ? HIGHLIGHT_DIM : null);
      colorAttr.needsUpdate = true;
      if (cur.read && cur.read.length === nP) { cur.scale.fill(cur.read, pcolors, null, null); pgeom.attributes.color.needsUpdate = true; }
      else { pcolors.fill(0.4); pgeom.attributes.color.needsUpdate = true; }
    }
    function setField(fieldId, scale) {
      var disp = result.fieldArray(fieldId, 'display'), read = result.fieldArray(fieldId, 'read');
      if (!disp || disp.length !== nV) fail('field ' + fieldId + ': display array length ≠ mesh vertices');
      if (read && read.length !== nP) fail('field ' + fieldId + ': read array length ≠ prediction points');
      cur = { fieldId: fieldId, scale: scale, disp: disp, read: read };
      recolor();
    }
    function setLighting(mode) { lightMode = mode === 'soft' ? 'soft' : 'flat'; mesh.material = disp.stl ? stlMat : (lightMode === 'soft' ? soft : flat); }
    function setLayers(L) {
      layerState = Object.assign({}, L);
      mesh.visible = L.wall !== false;
      outline.visible = !!L.outline && mesh.visible;
      hatch.visible = !!L.trust && !!MT && mesh.visible;
      stagMesh.visible = stagState === 'on' && mesh.visible && !disp.stl;
      pointsObj.visible = !!L.points;
      if (clObj) clObj.visible = !!L.centerline;
      return { trust: !!L.trust && !!MT, centerline: !!L.centerline && !!clObj, streamlines: false, interior: false };
    }
    // Stagnation mask on the display vertices from the TAWSS / OSI display arrays (classic stagnationMask('m')); a NaN
    // on either side is not stagnant.  'pending' until both arrays are loaded, 'unavailable' without them.
    function stagnationMask(c) {
      var ft = result.field('tawss'), fo = result.field('osi');
      var kt = ft && ft.arrays && ft.arrays.display, ko = fo && fo.arrays && fo.arrays.display;
      if (!kt || !ko || !result.declared(kt) || !result.declared(ko)) return 'unavailable';
      if (!result.has(kt) || !result.has(ko)) return 'pending';
      var T = result.array(kt), O = result.array(ko);
      if (T.length !== nV || O.length !== nV) return 'unavailable';
      var lt = Number.isFinite(+c.tawss_lt_pa) ? +c.tawss_lt_pa : 0.4, gt = Number.isFinite(+c.osi_gt) ? +c.osi_gt : 0.1;
      for (var q = 0; q < nV; q++) stag[q] = T[q] < lt && O[q] > gt ? 1 : 0;
      geom.attributes.aStag.needsUpdate = true;
      return 'on';
    }
    // o = {stl, stagnation: null | {tawss_lt_pa, osi_gt}}; returns what is shown.
    function setDisplay(o) {
      o = o || {};
      disp = { stl: Boolean(o.stl), stagnation: o.stagnation && typeof o.stagnation === 'object' ? o.stagnation : null };
      stagState = disp.stagnation ? stagnationMask(disp.stagnation) : 'off';
      setLighting(lightMode);
      setLayers(layerState);
      return { stl: disp.stl, stagnation: stagState };
    }
    function setBranchVisibility(set) {
      hidden = set ? function (sid) { return !set.has(Number(sid)); } : null;
      if (!hidden || !MS) {
        geom.setIndex(new THREE.BufferAttribute(F, 1));
        pointMask = null; vertexMask = null; visibleVerts = null;
        pgeom.setIndex(null);
        if (clObj) clObj.geometry.setIndex(new THREE.BufferAttribute(CE instanceof Uint32Array ? CE : Uint32Array.from(CE), 1));
      } else {
        vertexMask = new Uint8Array(nV);
        for (i = 0; i < nV; i++) vertexMask[i] = hidden(MS[i]) ? 0 : 1;
        var keep = [];
        for (i = 0; i < F.length; i += 3) if (vertexMask[F[i]] || vertexMask[F[i + 1]] || vertexMask[F[i + 2]]) keep.push(F[i], F[i + 1], F[i + 2]);
        geom.setIndex(new THREE.BufferAttribute(new Uint32Array(keep), 1));
        pointMask = new Uint8Array(nP);
        var pk = [];
        for (i = 0; i < nP; i++) if (!PS || !hidden(PS[i])) { pointMask[i] = 1; pk.push(i); }
        pgeom.setIndex(new THREE.BufferAttribute(new Uint32Array(pk), 1));
        if (clObj && CS) {
          var ek = [];
          for (i = 0; i + 1 < CE.length; i += 2) if (!hidden(CS[CE[i]]) && !hidden(CS[CE[i + 1]])) ek.push(CE[i], CE[i + 1]);
          clObj.geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(ek), 1));
        }
        var vv = [];
        for (i = 0; i < nV; i++) if (vertexMask[i]) vv.push(V[3 * i], V[3 * i + 1], V[3 * i + 2]);
        visibleVerts = new Float32Array(vv);
      }
      geom.index.needsUpdate = true;
      geom.computeBoundingSphere();
    }
    function pointValues(i) {
      var out = {};
      result.fields().forEach(function (f) {
        if (!f || !f.arrays || typeof f.arrays.read !== 'string' || !result.has(f.arrays.read) || (Number(f.components) || 1) !== 1) return;
        out[f.id] = result.valueAt(f.id, i);
      });
      return out;
    }
    // The wall under a ray, the raw hit (measurement and region centres; the classic report used hit.point).
    function pickSurface(raycaster) {
      if (!mesh.visible) return null;
      var hits = raycaster.intersectObject(mesh, false);
      if (!hits.length) return null;
      var h = hits[0], p = h.point, best = h.face.a, bd = Infinity;
      [h.face.a, h.face.b, h.face.c].forEach(function (vi) { var dx = V[3 * vi] - p.x, dy = V[3 * vi + 1] - p.y, dz = V[3 * vi + 2] - p.z, d = dx * dx + dy * dy + dz * dz; if (d < bd) { bd = d; best = vi; } });
      return { xyz: [p.x, p.y, p.z], vertexIndex: best };
    }
    function pick(raycaster) {
      if (!mesh.visible) return null;
      var hits = raycaster.intersectObject(mesh, false);
      if (!hits.length) return null;
      var h = hits[0], p = h.point, face = h.face;
      var best = -1, bd = Infinity;
      [face.a, face.b, face.c].forEach(function (vi) { var dx = V[3 * vi] - p.x, dy = V[3 * vi + 1] - p.y, dz = V[3 * vi + 2] - p.z, d = dx * dx + dy * dy + dz * dz; if (d < bd) { bd = d; best = vi; } });
      var pi = pointGrid().nearest(p.x, p.y, p.z, { maxDist: 6, accept: pointMask ? function (j) { return pointMask[j] === 1; } : null });
      var hit = [p.x, p.y, p.z];
      var out = {
        pointIndex: pi >= 0 ? pi : null, vertexIndex: best, hitXyz: hit,
        xyz: pi >= 0 ? [PV[3 * pi], PV[3 * pi + 1], PV[3 * pi + 2]] : hit,
        distance_mm: pi >= 0 ? Math.hypot(PV[3 * pi] - p.x, PV[3 * pi + 1] - p.y, PV[3 * pi + 2] - p.z) : null,
        segmentId: pi >= 0 && PS ? Number(PS[pi]) : (MS ? Number(MS[best]) : null),
        s_from_root_mm: pi >= 0 && PSR ? (Number.isFinite(PSR[pi]) ? PSR[pi] : null) : null,
        value: NaN, valueSource: null, values: pi >= 0 ? pointValues(pi) : {},
        trust: MT ? MT[best] : 0
      };
      if (cur.read && pi >= 0) { out.value = Number.isFinite(cur.read[pi]) ? cur.read[pi] : NaN; out.valueSource = 'read'; }
      else if (!cur.read && cur.disp) { out.value = Number.isFinite(cur.disp[best]) ? cur.disp[best] : NaN; out.valueSource = 'display'; }
      return out;
    }
    // o.vertexMask (Uint8Array per display vertex) marks the vertices directly (the region tool); otherwise the
    // highlighted prediction points are carried to the vertices by their nearest point.
    function highlight(indices, o) {
      if (!indices) { hl = null; hlMask = null; recolor(); return; }
      if (o && o.vertexMask && o.vertexMask.length === nV) {
        hlMask = o.vertexMask; hl = { set: null, color: o.color || null }; recolor();
        if (hl.color) { var c0 = new THREE.Color(hl.color); for (var t0 = 0; t0 < nV; t0++) if (hlMask[t0]) { colors[3 * t0] = colors[3 * t0] * 0.65 + c0.r * 0.35; colors[3 * t0 + 1] = colors[3 * t0 + 1] * 0.65 + c0.g * 0.35; colors[3 * t0 + 2] = colors[3 * t0 + 2] * 0.65 + c0.b * 0.35; } colorAttr.needsUpdate = true; }
        return;
      }
      var set = new Uint8Array(nP);
      for (var j = 0; j < indices.length; j++) { var q = indices[j]; if (q >= 0 && q < nP) set[q] = 1; }
      var map = vertexToPoint();
      hlMask = new Uint8Array(nV);
      for (var t = 0; t < nV; t++) hlMask[t] = map[t] >= 0 && set[map[t]] ? 1 : 0;
      hl = { set: set, color: o && o.color || null };
      recolor();
      if (hl.color) {   // tint the highlighted part toward the given colour
        var cc = new THREE.Color(hl.color);
        for (t = 0; t < nV; t++) if (hlMask[t]) { colors[3 * t] = colors[3 * t] * 0.65 + cc.r * 0.35; colors[3 * t + 1] = colors[3 * t + 1] * 0.65 + cc.g * 0.35; colors[3 * t + 2] = colors[3 * t + 2] * 0.65 + cc.b * 0.35; }
        colorAttr.needsUpdate = true;
      }
    }
    function histogramValues(fieldId) {
      var read = result.fieldArray(fieldId, 'read');
      if (read && read.length === nP) return { values: read, source: 'points', mask: pointMask, scope: pointMask ? 'visible' : 'all' };
      var disp = result.fieldArray(fieldId, 'display');
      return disp ? { values: disp, source: 'vertices', mask: vertexMask, scope: vertexMask ? 'visible' : 'all' } : null;
    }
    return {
      family: 'wall',
      fields: displayable(result),
      bounds: bounds,
      setField: setField, setLighting: setLighting, setLayers: setLayers, setBranchVisibility: setBranchVisibility,
      pick: pick, pickSurface: pickSurface, highlight: highlight, histogramValues: histogramValues,
      setDisplay: setDisplay,
      displayState: function () { return { stl: disp.stl, stagnation: stagState }; },
      arraysLoaded: function () { if (stagState === 'pending' && disp.stagnation) setDisplay(disp); },
      fitPoints: function () { return visibleVerts || V; },
      pointXYZ: function (q) { return q >= 0 && q < nP ? [PV[3 * q], PV[3 * q + 1], PV[3 * q + 2]] : null; },
      sectionMesh: function () { return { vertices: V, faces: geom.index.array }; },
      counts: { vertices: nV, faces: F.length / 3, points: nP },
      dispose: function () {
        gfx.disposeObject(group);
        soft.dispose(); flat.dispose(); stlMat.dispose();
        dotTex.dispose();
        grid = null; v2p = null;
      }
    };
  }

  return {
    supports: { fields: ['wall'], tools: ['pick', 'cursor', 'trust', 'centerline', 'points', 'outline', 'lighting', 'markers', 'highlight', 'branches', 'measure', 'region'] },
    displayable: displayable, defaultField: defaultField, requiredArrays: requiredArrays, build: build
  };
});
