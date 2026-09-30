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

  function glassMaterial(THREE) {
    return new THREE.ShaderMaterial({
      uniforms: { uColor: { value: new THREE.Color('#a9b6c4') }, uBase: { value: 0.07 }, uRim: { value: 0.55 } },   // mid grey: reads on a dark and on a light stage
      vertexShader: 'varying vec3 vN; varying vec3 vV; void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }',
      fragmentShader: 'uniform vec3 uColor; uniform float uBase; uniform float uRim; varying vec3 vN; varying vec3 vV;' +
        'void main(){ float f = 1.0 - abs(dot(normalize(vN), normalize(vV))); gl_FragColor = vec4(uColor, clamp(uBase + uRim * f * f * f, 0.0, 0.9)); }',
      transparent: true, depthWrite: false, side: THREE.DoubleSide
    });
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
      var nL = Math.floor(X.length / 3), idx = [];
      for (var l = 0; l + 1 < O.length; l++) { var a = O[l], b = Math.min(O[l + 1], nL); for (var j = a; j + 1 < b; j++) idx.push(j, j + 1); }
      var lg = new THREE.BufferGeometry();
      lg.setAttribute('position', new THREE.BufferAttribute(X, 3));
      lineColors = new Float32Array(3 * nL);
      lg.setAttribute('color', new THREE.BufferAttribute(lineColors, 3));
      lg.setIndex(new THREE.BufferAttribute(new Uint32Array(idx), 1));
      lineSpeed = S && S.length === nL ? S : null;
      lines = new THREE.LineSegments(lg, neutralLine); lines.name = 'volume-streamlines'; lines.renderOrder = 2; lines.visible = false;
      group.add(lines);
      return true;
    }

    // centreline
    var clObj = null, clKeys = geo(m).centerline || {}, CV = opt(result, clKeys.xyz), CE = opt(result, clKeys.edges);
    if (CV && CE) {
      var cg = new THREE.BufferGeometry();
      cg.setAttribute('position', new THREE.BufferAttribute(CV, 3));
      cg.setIndex(new THREE.BufferAttribute(CE instanceof Uint32Array ? CE : Uint32Array.from(CE), 1));
      clObj = new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ color: new THREE.Color('#465361'), depthTest: false, transparent: true, opacity: 0.85 }));
      clObj.renderOrder = 6; clObj.visible = false; group.add(clObj);
    }

    var cur = { fieldId: null, scale: null, wallField: false, read: null, disp: null };
    var L = {}, lighting = 'flat', hidden = null, pointMask = null, hlObj = null;
    function applyVisibility() {
      var wallOn = L.wall !== false;
      wall.visible = wallOn;
      wall.material = cur.wallField ? (lighting === 'soft' ? soft : flat) : glass;
      wall.renderOrder = cur.wallField ? 0 : 3;
      glass.uniforms.uRim.value = L.outline ? 0.55 : 0.12;
      outline.visible = wallOn && cur.wallField && !!L.outline;
      hatch.visible = wallOn && cur.wallField && !!L.trust && !!MT;
      cloud.visible = !cur.wallField && L.interior !== false;
      if (L.streamlines) buildStreamlines();
      if (lines) {
        lines.visible = !!L.streamlines && !cur.wallField;
        var coloured = cur.fieldId === 'speed' && lineSpeed;
        lines.material = coloured ? colouredLine : neutralLine;
      }
      if (clObj) clObj.visible = !!L.centerline;
    }
    function recolor() {
      if (!cur.scale) return;
      if (cur.wallField) { cur.scale.fill(cur.disp, wcolors, null, null); wcolorAttr.needsUpdate = true; }
      else {
        for (var j = 0; j < nI; j++) ivals[j] = cur.read[interiorIdx[j]];
        cur.scale.fill(ivals, pcolors, null, null); pcolorAttr.needsUpdate = true;
      }
      if (lines && lineSpeed && cur.fieldId === 'speed') { cur.scale.fill(lineSpeed, lineColors, null, null); lines.geometry.attributes.color.needsUpdate = true; }
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
      L = Object.assign({}, layers);
      applyVisibility();
      if (L.streamlines && lines && cur.fieldId === 'speed') recolor();
      var hasLines = !!(k.slxyz && k.sloff);
      return { streamlines: !!L.streamlines && hasLines, trust: !!L.trust && !!MT, centerline: !!L.centerline && !!clObj, points: false };
    }
    function setBranchVisibility(set) {
      hidden = set && PSEG ? function (sid) { return !set.has(Number(sid)); } : null;
      if (!hidden) { pgeom.setIndex(null); pointMask = null; }
      else {
        var keep = []; pointMask = new Uint8Array(nP);
        for (var j = 0; j < nI; j++) { var q = interiorIdx[j]; if (!hidden(PSEG[q])) { keep.push(j); pointMask[q] = 1; } }
        pgeom.setIndex(new THREE.BufferAttribute(new Uint32Array(keep), 1));
      }
      pgeom.computeBoundingSphere();
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
      hlObj = new THREE.Points(g2, new THREE.PointsMaterial({ color: new THREE.Color(o && o.color || gfx.ACCENT_HEX), size: pointSize * 1.8 * fovK, sizeAttenuation: true, map: dotTex, alphaTest: 0.5 }));
      hlObj.renderOrder = 7;
      group.add(hlObj);
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
      fitPoints: function () { return V; },
      pointXYZ: function (q) { return q >= 0 && q < nP ? [P[3 * q], P[3 * q + 1], P[3 * q + 2]] : null; },
      sectionMesh: function () { return { vertices: V, faces: F }; },
      arraysLoaded: function () { if (L.streamlines) { buildStreamlines(); applyVisibility(); recolor(); } },
      counts: { vertices: nV, faces: F.length / 3, points: nP, interior: nI },
      dispose: function () {
        gfx.disposeObject(group);
        [glass, flat, soft, neutralLine, colouredLine].forEach(function (mm) { mm.dispose(); });
        dotTex.dispose();
      }
    };
  }

  return {
    supports: { fields: ['interior', 'wall'], tools: ['pick', 'cursor', 'streamlines', 'outline', 'lighting', 'markers', 'highlight', 'branches', 'trust', 'slice'] },
    displayable: displayable, defaultField: defaultField, requiredArrays: requiredArrays, optionalArrays: optionalArrays, build: build
  };
});
