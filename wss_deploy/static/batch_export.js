/* Cross-case batch figure export (C13, contract §12.6 / §13).

   zipStore(files)  — store-mode zip writer with CRC-32 (no compression, no external library); DOM-free.
   run(options)     — loads each job's report in a hidden iframe, applies one view state, asks the report
                      to export a PNG through the wss-view message protocol and collects the results.
   The `env` option lets Node tests inject a fake document/window; the browser uses the real ones. */
(function (root) {
  'use strict';
  const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; table[n] = c >>> 0; }
    return table;
  })();
  function crc32(bytes) {
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < bytes.length; i++) crc = CRC_TABLE[(crc ^ bytes[i]) & 0xFF] ^ (crc >>> 8);
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }
  function utf8(text) { return new TextEncoder().encode(String(text)); }
  function dosDateTime(date) {
    const d = date instanceof Date && !Number.isNaN(date.getTime()) ? date : new Date();
    const year = Math.min(Math.max(d.getFullYear(), 1980), 2107);
    const time = ((d.getHours() & 31) << 11) | ((d.getMinutes() & 63) << 5) | ((d.getSeconds() >> 1) & 31);
    const day = (((year - 1980) & 127) << 9) | (((d.getMonth() + 1) & 15) << 5) | (d.getDate() & 31);
    return {time, day};
  }
  function toBytes(value) {
    if (value instanceof Uint8Array) return value;
    if (typeof value === 'string') return utf8(value);
    if (value && value.buffer instanceof ArrayBuffer) return new Uint8Array(value.buffer, value.byteOffset || 0, value.byteLength);
    if (value instanceof ArrayBuffer) return new Uint8Array(value);
    throw new Error('zip entry bytes must be a Uint8Array or string');
  }
  function zipStore(files) {
    const entries = [];
    let offset = 0;
    const seen = new Set();
    for (const file of files || []) {
      let name = String(file.name || '').replace(/\\/g, '/').replace(/^\/+/, '');
      if (!name) throw new Error('zip entry needs a name');
      let unique = name, n = 2;
      while (seen.has(unique)) { const dot = name.lastIndexOf('.'); unique = dot > 0 ? `${name.slice(0, dot)}_${n}${name.slice(dot)}` : `${name}_${n}`; n++; }
      seen.add(unique);
      const nameBytes = utf8(unique), bytes = toBytes(file.bytes), crc = crc32(bytes), {time, day} = dosDateTime(file.mtime);
      const local = new Uint8Array(30 + nameBytes.length);
      const view = new DataView(local.buffer);
      view.setUint32(0, 0x04034b50, true); view.setUint16(4, 20, true); view.setUint16(6, 0x0800, true); view.setUint16(8, 0, true);
      view.setUint16(10, time, true); view.setUint16(12, day, true); view.setUint32(14, crc, true);
      view.setUint32(18, bytes.length, true); view.setUint32(22, bytes.length, true); view.setUint16(26, nameBytes.length, true); view.setUint16(28, 0, true);
      local.set(nameBytes, 30);
      entries.push({local, bytes, nameBytes, crc, time, day, offset});
      offset += local.length + bytes.length;
    }
    const centralStart = offset;
    const central = [];
    for (const entry of entries) {
      const record = new Uint8Array(46 + entry.nameBytes.length);
      const view = new DataView(record.buffer);
      view.setUint32(0, 0x02014b50, true); view.setUint16(4, 20, true); view.setUint16(6, 20, true); view.setUint16(8, 0x0800, true); view.setUint16(10, 0, true);
      view.setUint16(12, entry.time, true); view.setUint16(14, entry.day, true); view.setUint32(16, entry.crc, true);
      view.setUint32(20, entry.bytes.length, true); view.setUint32(24, entry.bytes.length, true); view.setUint16(28, entry.nameBytes.length, true);
      view.setUint16(30, 0, true); view.setUint16(32, 0, true); view.setUint16(34, 0, true); view.setUint16(36, 0, true); view.setUint32(38, 0, true); view.setUint32(42, entry.offset, true);
      record.set(entry.nameBytes, 46);
      central.push(record); offset += record.length;
    }
    const end = new Uint8Array(22);
    const endView = new DataView(end.buffer);
    endView.setUint32(0, 0x06054b50, true); endView.setUint16(4, 0, true); endView.setUint16(6, 0, true);
    endView.setUint16(8, entries.length, true); endView.setUint16(10, entries.length, true);
    endView.setUint32(12, offset - centralStart, true); endView.setUint32(16, centralStart, true); endView.setUint16(20, 0, true);
    const out = new Uint8Array(offset + 22);
    let cursor = 0;
    for (const entry of entries) { out.set(entry.local, cursor); cursor += entry.local.length; out.set(entry.bytes, cursor); cursor += entry.bytes.length; }
    for (const record of central) { out.set(record, cursor); cursor += record.length; }
    out.set(end, cursor);
    return out;
  }
  function base64ToBytes(text) {
    const clean = String(text || '').replace(/^data:[^,]*,/, '').replace(/\s+/g, '');
    if (typeof atob === 'function') { const binary = atob(clean); const out = new Uint8Array(binary.length); for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i); return out; }
    return new Uint8Array(Buffer.from(clean, 'base64'));
  }
  function safeName(text) { return String(text || 'case').replace(/[\\/:*?"<>|\s]+/g, '_').slice(0, 80) || 'case'; }

  const DEFAULT_TIMEOUT_MS = 60000;
  function exportOne(job, state, options, env, timeoutMs) {
    // One hidden iframe per job: ready → apply-state → applied → export → exported.
    return new Promise(resolve => {
      const origin = env.origin;
      const frame = env.createFrame();
      let done = false, requestSeq = 0;
      const requests = {};
      const finish = result => {
        if (done) return; done = true;
        env.clearTimeout(timer); env.removeListener(onMessage);
        try { frame.remove(); } catch (_) {}
        resolve(result);
      };
      const timer = env.setTimeout(() => finish({ok: false, message: `等待报告响应超过 ${Math.round(timeoutMs / 1000)} 秒`}), timeoutMs);
      const post = (message, key) => {
        const requestId = `${job.id}-${++requestSeq}`;
        requests[key] = requestId;
        try { frame.contentWindow.postMessage({...message, request_id: requestId}, origin); }
        catch (error) { finish({ok: false, message: `无法向报告发送消息：${error.message}`}); }
      };
      const onMessage = event => {
        if (done || event.origin !== origin || event.source !== frame.contentWindow) return;
        const data = event.data;
        if (!data || typeof data.type !== 'string') return;
        if (data.type === 'wss-view:ready') {
          if (data.webgl === false) { finish({ok: false, message: '报告页没有 WebGL，无法出图'}); return; }
          if (job.family && data.family && data.family !== job.family) { finish({ok: false, message: `报告族 ${data.family} 与任务族 ${job.family} 不一致`}); return; }
          if (state) post({type: 'wss-view:apply-state', state}, 'apply');
          else post({type: 'wss-view:export', options}, 'export');
        } else if (data.type === 'wss-view:applied') {
          if (data.request_id && requests.apply && data.request_id !== requests.apply) return;
          post({type: 'wss-view:export', options}, 'export');
        } else if (data.type === 'wss-view:exported') {
          if (data.request_id && requests.export && data.request_id !== requests.export) return;
          let bytes;
          try { bytes = base64ToBytes(data.png_base64); } catch (error) { finish({ok: false, message: `PNG 数据无法解码：${error.message}`}); return; }
          const scale = options && options.scale ? `${options.scale}x` : '1x';
          const filename = data.filename || `${safeName(job.case_id)}_${job.id}_${scale}.png`;
          const files = [{name: filename, bytes}];
          if (data.colorbar_svg) files.push({name: filename.replace(/\.png$/i, '') + '_colorbar.svg', bytes: utf8(data.colorbar_svg)});
          finish({ok: true, files, downgraded: Boolean(data.downgraded), width: data.width, height: data.height});
        } else if (data.type === 'wss-view:error') {
          finish({ok: false, message: data.message || '报告页报告了错误'});
        }
      };
      env.addListener(onMessage);
      try { frame.src = env.reportUrl(job.id); }
      catch (error) { finish({ok: false, message: error.message}); }
    });
  }
  function browserEnv() {
    return {
      origin: root.location.origin,
      reportUrl: id => `/api/jobs/${encodeURIComponent(id)}/report`,
      createFrame() {
        const frame = root.document.createElement('iframe');
        frame.setAttribute('aria-hidden', 'true'); frame.setAttribute('title', '批量出图渲染中');
        frame.style.cssText = 'position:fixed;left:-20000px;top:0;width:1280px;height:800px;border:0;opacity:0.01;pointer-events:none';
        root.document.body.append(frame);
        return frame;
      },
      addListener: handler => root.addEventListener('message', handler),
      removeListener: handler => root.removeEventListener('message', handler),
      setTimeout: (fn, ms) => root.setTimeout(fn, ms),
      clearTimeout: id => root.clearTimeout(id),
    };
  }
  async function run({jobs, state, options, onProgress, timeoutMs, env}) {
    env = env || browserEnv();
    const list = Array.isArray(jobs) ? jobs : [];
    const files = [], failed = [], results = [];
    for (let index = 0; index < list.length; index++) {
      const job = list[index];
      if (typeof onProgress === 'function') onProgress({index, total: list.length, job, phase: 'start'});
      const result = await exportOne(job, state, options || {}, env, timeoutMs || DEFAULT_TIMEOUT_MS);
      if (result.ok) { for (const file of result.files) { file.name = `${safeName(job.case_id)}_${job.id}/${file.name}`; files.push(file); } results.push({id: job.id, ok: true, downgraded: result.downgraded}); }
      else { failed.push({id: job.id, case_id: job.case_id, message: result.message}); results.push({id: job.id, ok: false, message: result.message}); }
      if (typeof onProgress === 'function') onProgress({index, total: list.length, job, phase: 'done', ok: result.ok, message: result.message});
    }
    const manifest = {schema_version: 'wss-deploy.batch_export/v1', created_at: new Date().toISOString(), state: state || null, options: options || null, results};
    files.push({name: 'manifest.json', bytes: utf8(JSON.stringify(manifest, null, 1))});
    return {zip: zipStore(files), failed, count: list.length - failed.length};
  }
  const api = {crc32, zipStore, base64ToBytes, run, exportOne, safeName, DEFAULT_TIMEOUT_MS};
  root.WssBatchExport = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
