/* WSSV2 core · data (contract §2, §3, §6.2).
   A source yields the v2 manifest and raw little-endian arrays, either online (/api/v2/jobs/<id>/…) or from
   the offline package (<script id="wssv2-manifest"> + <script id="wssv2-arrays">).  Every array is decoded
   once and checked against its manifest entry (dtype, shape, byte count); a mismatch rejects with DataError.
   Nothing is padded or zero-filled; NaN stays NaN (a missing value, never 0). */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.data = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var SCHEMA = 'wss-deploy.v2-manifest/v1';
  var DTYPES = {
    float32: { T: typeof Float32Array !== 'undefined' ? Float32Array : null, size: 4 },
    float64: { T: typeof Float64Array !== 'undefined' ? Float64Array : null, size: 8 },
    int8: { T: Int8Array, size: 1 }, uint8: { T: Uint8Array, size: 1 },
    int16: { T: Int16Array, size: 2 }, uint16: { T: Uint16Array, size: 2 },
    int32: { T: Int32Array, size: 4 }, uint32: { T: Uint32Array, size: 4 }
  };
  var LITTLE = (function () { try { return new Uint8Array(new Uint16Array([1]).buffer)[0] === 1; } catch (_) { return true; } })();

  function DataError(message, info) {
    var err = new Error(message);
    err.name = 'DataError';
    info = info || {};
    for (var k in info) if (Object.prototype.hasOwnProperty.call(info, k)) err[k] = info[k];
    Object.setPrototypeOf(err, DataError.prototype);
    return err;
  }
  DataError.prototype = Object.create(Error.prototype);
  DataError.prototype.constructor = DataError;
  DataError.prototype.name = 'DataError';

  function isDataError(err) { return !!err && (err instanceof DataError || err.name === 'DataError'); }

  function abortError() {
    var e = new Error('aborted');
    e.name = 'AbortError';
    return e;
  }

  // ------------------------------------------------------------------ decoding and validation
  function atobAny(s) {
    if (typeof root.atob === 'function') return root.atob(s);
    if (typeof Buffer !== 'undefined') return Buffer.from(s, 'base64').toString('binary');
    throw DataError('no base64 decoder available', { code: 'no_base64' });
  }
  // Standard or URL-safe base64 → fresh ArrayBuffer (offset 0, so any typed view is aligned).
  function base64ToBuffer(b64) {
    if (typeof b64 !== 'string') throw DataError('embedded array is not a base64 string', { code: 'bad_embedded' });
    var s = b64.replace(/\s+/g, '').replace(/-/g, '+').replace(/_/g, '/');
    while (s.length % 4) s += '=';
    var bin;
    try { bin = atobAny(s); } catch (err) { throw DataError('embedded array is not valid base64', { code: 'bad_base64', cause: err }); }
    var u = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u.buffer;
  }

  function prod(shape) { var n = 1; for (var i = 0; i < shape.length; i++) n *= shape[i]; return n; }

  // Checks one manifest array entry for internal consistency: known dtype, integer shape, bytes = Π shape × size.
  function checkEntry(key, entry) {
    if (!entry || typeof entry !== 'object') throw DataError('array "' + key + '" is not declared in the manifest', { code: 'undeclared', key: key });
    var dt = DTYPES[entry.dtype];
    if (!dt || !dt.T) throw DataError('array "' + key + '": unsupported dtype ' + entry.dtype, { code: 'dtype', key: key });
    var shape = entry.shape;
    if (!Array.isArray(shape) || !shape.length || !shape.every(function (d) { return Number.isInteger(d) && d >= 0; }))
      throw DataError('array "' + key + '": invalid shape', { code: 'shape', key: key });
    var expect = prod(shape) * dt.size;
    if (entry.bytes !== undefined && entry.bytes !== null && Number(entry.bytes) !== expect)
      throw DataError('array "' + key + '": manifest bytes ' + entry.bytes + ' ≠ shape × ' + dt.size + ' = ' + expect, { code: 'bytes', key: key });
    return { dt: dt, expect: expect, shape: shape.slice() };
  }

  // ArrayBuffer (or view) → typed array of the declared dtype; the byte length must match exactly.
  function decodeArray(key, entry, buffer) {
    var info = checkEntry(key, entry);
    var buf = buffer;
    if (buf && buf.buffer instanceof ArrayBuffer && !(buf instanceof ArrayBuffer)) {
      buf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
    if (!(buf instanceof ArrayBuffer)) throw DataError('array "' + key + '": no data', { code: 'no_data', key: key });
    if (buf.byteLength !== info.expect)
      throw DataError('array "' + key + '": ' + buf.byteLength + ' bytes received, ' + info.expect + ' expected (' + entry.dtype + ' ' + JSON.stringify(info.shape) + ')',
        { code: 'length', key: key, received: buf.byteLength, expected: info.expect });
    if (!LITTLE && info.dt.size > 1) {   // arrays are little-endian on the wire (§2)
      var u = new Uint8Array(buf), s = info.dt.size;
      for (var i = 0; i < u.length; i += s) for (var a = i, b = i + s - 1; a < b; a++, b--) { var t = u[a]; u[a] = u[b]; u[b] = t; }
    }
    var out = new info.dt.T(buf);
    out.shape = info.shape;   // expando: [n] or [n, k]
    return out;
  }

  // ------------------------------------------------------------------ sources
  function jsonScript(doc, id) {
    var el = doc && doc.getElementById ? doc.getElementById(id) : null;
    if (!el) return undefined;
    try { return JSON.parse(el.textContent || ''); }
    catch (err) { throw DataError('embedded ' + id + ' is not valid JSON', { code: 'bad_embedded', cause: err }); }
  }

  function checkManifest(m) {
    if (!m || typeof m !== 'object') throw DataError('manifest missing', { code: 'manifest' });
    if (m.schema !== SCHEMA) throw DataError('unsupported manifest schema: ' + m.schema, { code: 'schema', schema: m.schema });
    if (!m.arrays || typeof m.arrays !== 'object') throw DataError('manifest has no arrays table', { code: 'manifest' });
    return m;
  }

  // Online: GET /api/v2/jobs/<id>/manifest and /arrays/<key> with the session cookie (same origin).
  function createOnlineSource(jobId, opts) {
    opts = opts || {};
    var fetchFn = opts.fetch || (typeof root.fetch === 'function' ? root.fetch.bind(root) : null);
    if (!fetchFn) throw DataError('fetch is not available', { code: 'no_fetch' });
    var base = (opts.base || '') + '/api/v2/jobs/' + encodeURIComponent(String(jobId));
    var manifestPromise = null;
    function httpError(res, what, key) {
      return Promise.resolve(res.text ? res.text() : '').catch(function () { return ''; }).then(function (text) {
        var body = null; try { body = JSON.parse(text); } catch (_) { body = text || null; }
        return DataError(what + ' request failed (HTTP ' + res.status + ')', { code: 'http', status: res.status, body: body, key: key || null });
      });
    }
    var source = {
      kind: 'online', jobId: String(jobId),
      manifest: function (o) {
        if (manifestPromise) return manifestPromise;
        manifestPromise = Promise.resolve(fetchFn(base + '/manifest', { credentials: 'same-origin', signal: o && o.signal, headers: { Accept: 'application/json' } }))
          .then(function (res) {
            if (!res || !res.ok) return httpError(res || { status: 0 }, 'manifest').then(function (e) { throw e; });
            return res.json();
          })
          .then(checkManifest)
          .catch(function (err) { manifestPromise = null; throw err; });
        return manifestPromise;
      },
      // Raw bytes of one array (ArrayBuffer).  The URL comes from the manifest entry when present.
      fetchArray: function (key, entry, o) {
        var url = entry && typeof entry.url === 'string' && entry.url ? entry.url : base + '/arrays/' + encodeURIComponent(key);
        return Promise.resolve(fetchFn(url, { credentials: 'same-origin', signal: o && o.signal }))
          .then(function (res) {
            if (!res || !res.ok) return httpError(res || { status: 0 }, 'array "' + key + '"', key).then(function (e) { throw e; });
            return res.arrayBuffer();
          });
      }
    };
    return source;
  }

  // Offline: the page carries the manifest and base64 arrays.  wssv2-arrays may be {key: b64} or {arrays: {key: b64}};
  // an entry may also be {b64: "..."}.
  function createEmbeddedSource(doc) {
    doc = doc || root.document;
    var cached = null, table = null;
    function arrays() {
      if (table) return table;
      var raw = jsonScript(doc, 'wssv2-arrays');
      if (raw === undefined) throw DataError('embedded arrays (wssv2-arrays) not found', { code: 'no_embedded' });
      table = raw && typeof raw === 'object' && raw.arrays && typeof raw.arrays === 'object' && !Array.isArray(raw.arrays) ? raw.arrays : raw;
      return table;
    }
    var source = {
      kind: 'embedded', jobId: null,
      manifest: function () {
        if (cached) return cached;
        cached = new Promise(function (resolve) {
          var m = jsonScript(doc, 'wssv2-manifest');
          if (m === undefined) throw DataError('embedded manifest (wssv2-manifest) not found', { code: 'no_embedded' });
          checkManifest(m);
          source.jobId = m.job && m.job.id ? String(m.job.id) : null;
          resolve(m);
        });
        cached.catch(function () { cached = null; });
        return cached;
      },
      fetchArray: function (key) {
        return new Promise(function (resolve) {
          var t = arrays(), v = t ? t[key] : undefined;
          if (v && typeof v === 'object' && typeof v.b64 === 'string') v = v.b64;
          if (typeof v !== 'string') throw DataError('embedded array "' + key + '" missing', { code: 'missing', key: key });
          resolve(base64ToBuffer(v));
        });
      },
      // wssv2-offline JSON (bookmarks, view, hide_name, exported_at) or null
      offline: function () { try { return jsonScript(doc, 'wssv2-offline') || null; } catch (_) { return null; } }
    };
    return source;
  }

  // ------------------------------------------------------------------ result
  function fieldList(m) { return Array.isArray(m.fields) ? m.fields : []; }

  function loadResult(source, opts) {
    opts = opts || {};
    var signal = opts.signal;
    if (!source || typeof source.manifest !== 'function') return Promise.reject(DataError('invalid source', { code: 'source' }));
    if (signal && signal.aborted) return Promise.reject(abortError());
    return source.manifest({ signal: signal }).then(function (m) {
      if (signal && signal.aborted) throw abortError();
      checkManifest(m);
      var store = Object.create(null), pending = Object.create(null);
      var fieldsById = Object.create(null);
      fieldList(m).forEach(function (f) { if (f && typeof f.id === 'string') fieldsById[f.id] = f; });
      var branches = Array.isArray(m.geometry && m.geometry.branches) ? m.geometry.branches : [];

      function load(key, o) {
        if (store[key]) return Promise.resolve(store[key]);
        if (pending[key]) return pending[key];
        var entry = m.arrays[key];
        var p;
        try { checkEntry(key, entry); } catch (err) { return Promise.reject(err); }
        p = Promise.resolve(source.fetchArray(key, entry, o)).then(function (buf) {
          var arr = decodeArray(key, entry, buf);
          store[key] = arr;
          delete pending[key];
          return arr;
        }, function (err) { delete pending[key]; throw err; });
        pending[key] = p;
        return p;
      }

      var result = {
        manifest: m,
        source: source,
        runIdentity: m.result && m.result.run_identity || null,
        family: m.result && m.result.family || null,
        dataVersion: m.data_version || null,
        jobId: m.job && m.job.id || source.jobId || null,
        preload: function (keys, o) {
          keys = (keys || []).filter(function (k, i, a) { return typeof k === 'string' && k && a.indexOf(k) === i; });
          var sig = o && o.signal || signal;
          return Promise.all(keys.map(function (k) { return load(k, { signal: sig }); })).then(function () {
            if (sig && sig.aborted) throw abortError();
            return result;
          });
        },
        has: function (key) { return !!store[key]; },
        declared: function (key) { return !!(key && m.arrays[key]); },
        array: function (key) {
          var a = store[key];
          if (!a) throw DataError('array "' + key + '" has not been preloaded', { code: 'not_loaded', key: key });
          return a;
        },
        fields: function () { return fieldList(m).slice(); },
        field: function (id) { return fieldsById[id] || null; },
        fieldKeys: function (id, which) {
          var f = fieldsById[id]; if (!f || !f.arrays) return [];
          var out = [];
          (which ? [which] : ['display', 'read']).forEach(function (w) { var k = f.arrays[w]; if (typeof k === 'string' && k) out.push(k); });
          return out;
        },
        fieldArray: function (id, which) {
          var f = fieldsById[id]; if (!f || !f.arrays) return null;
          var key = f.arrays[which === 'display' ? 'display' : 'read'];
          if (typeof key !== 'string' || !key) return null;
          return result.array(key);
        },
        branch: function (id) {
          var sid = Number(id);
          for (var i = 0; i < branches.length; i++) if (Number(branches[i].id) === sid) return branches[i];
          return null;
        },
        branches: function () { return branches.slice(); },
        // Read-array value at a prediction point; NaN when missing, out of range, not loaded or vector-valued.
        valueAt: function (fieldId, pointIndex) {
          var f = fieldsById[fieldId];
          if (!f || !f.arrays || typeof f.arrays.read !== 'string' || !store[f.arrays.read]) return NaN;
          var comps = Number(f.components) || 1;
          if (comps !== 1) return NaN;
          var a = store[f.arrays.read], i = Number(pointIndex);
          if (!Number.isInteger(i) || i < 0 || i >= a.length) return NaN;
          var v = a[i];
          return Number.isFinite(v) ? v : NaN;
        },
        vectorAt: function (fieldId, pointIndex) {
          var f = fieldsById[fieldId];
          if (!f || !f.arrays || typeof f.arrays.read !== 'string' || !store[f.arrays.read]) return null;
          var k = Number(f.components) || 1, a = store[f.arrays.read], i = Number(pointIndex);
          if (!Number.isInteger(i) || i < 0 || (i + 1) * k > a.length) return null;
          var out = []; for (var c = 0; c < k; c++) out.push(Number.isFinite(a[i * k + c]) ? a[i * k + c] : NaN);
          return out;
        }
      };
      return result;
    });
  }

  return {
    SCHEMA: SCHEMA, DTYPES: DTYPES, DataError: DataError, isDataError: isDataError,
    createOnlineSource: createOnlineSource, createEmbeddedSource: createEmbeddedSource, loadResult: loadResult,
    base64ToBuffer: base64ToBuffer, decodeArray: decodeArray, checkEntry: checkEntry, checkManifest: checkManifest
  };
});
