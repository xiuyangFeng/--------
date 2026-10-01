/* WSS workspace v2 — service calls (contract §6.3 ws_api.js).  Same session, CSRF header and owner rules as the
 * classic workbench (app.js request / sendUpload); nothing here touches the DOM. */
(function (root, factory) {
  'use strict';
  var ns = root.WSSV2 = root.WSSV2 || {};
  var mod = factory(ns, root);
  ns.api = mod;
  if (typeof module === 'object' && module.exports) module.exports = mod;
})(typeof globalThis !== 'undefined' ? globalThis : this, function (ns, root) {
  'use strict';

  var SESSION_CALLS = {'/api/session': true, '/api/session/password': true};
  var state = {csrf: '', onUnauthorized: null, lastStatus: {}};

  function ApiError(message, status, body) {
    var e = new Error(message); e.name = 'ApiError'; e.status = status || 0; e.body = body || null; return e;
  }
  function messageOf(body, status) {
    if (!body) return status ? '服务返回错误（HTTP ' + status + '）。' : '请求没有完成。';
    var e = body.error !== undefined ? body.error : body.message;
    if (e && typeof e === 'object') e = e.message;
    return e ? String(e) : '服务返回错误（HTTP ' + status + '）。';
  }
  function enc(id) { return encodeURIComponent(String(id)); }
  function unauthorized(url) { if (!SESSION_CALLS[url] && typeof state.onUnauthorized === 'function') { try { state.onUnauthorized(); } catch (_) {} } }

  function request(url, opts) {
    opts = opts || {};
    var method = opts.method || 'GET';
    var headers = {Accept: 'application/json'};
    var body = opts.body;
    if (method !== 'GET' && state.csrf) headers['X-CSRF-Token'] = state.csrf;
    if (body !== undefined && !(root.FormData && body instanceof root.FormData)) { headers['Content-Type'] = 'application/json'; body = JSON.stringify(body); }
    var controller = typeof root.AbortController === 'function' ? new root.AbortController() : null;
    var timer = null;
    if (controller && opts.timeout !== 0) { timer = setTimeout(function () { controller.abort(); }, opts.timeout || 30000); if (timer && timer.unref) timer.unref(); }
    if (opts.signal && controller && opts.signal.addEventListener) opts.signal.addEventListener('abort', function () { controller.abort(); });
    return Promise.resolve(root.fetch(url, {method: method, headers: headers, body: body, credentials: 'same-origin', signal: controller ? controller.signal : undefined}))
      .then(function (response) {
        if (timer) clearTimeout(timer);
        state.lastStatus[url.split('?')[0]] = response.status;
        return Promise.resolve(response.json ? response.json() : null).catch(function () { return null; }).then(function (data) {
          if (!response.ok) {
            if (response.status === 401) unauthorized(url.split('?')[0]);
            throw ApiError(messageOf(data, response.status), response.status, data);
          }
          if (data === null) throw ApiError('服务返回了无法读取的响应（HTTP ' + response.status + '）。', response.status, null);
          return data;
        });
      }, function (error) {
        if (timer) clearTimeout(timer);
        if (error && error.name === 'AbortError') throw ApiError(opts.signal && opts.signal.aborted ? '请求已取消。' : '请求超时，请稍后重试。', 0, null);
        throw ApiError('暂时无法连接服务。已保存的结果不会丢失，恢复连接后可继续。', 0, null);
      });
  }
  // Binary downloads keep a JSON error body readable instead of navigating away.
  function download(url, opts) {
    opts = opts || {};
    var method = opts.method || 'GET';
    var headers = {};
    var body = opts.body;
    if (method !== 'GET' && state.csrf) headers['X-CSRF-Token'] = state.csrf;
    if (body !== undefined) { headers['Content-Type'] = 'application/json'; body = JSON.stringify(body); }
    return Promise.resolve(root.fetch(url, {method: method, headers: headers, body: body, credentials: 'same-origin'})).then(function (response) {
      if (!response.ok) {
        return Promise.resolve(response.json ? response.json() : null).catch(function () { return null; }).then(function (data) {
          if (response.status === 401) unauthorized(url);
          throw ApiError(messageOf(data, response.status), response.status, data);
        });
      }
      return response.blob().then(function (blob) {
        var cd = response.headers && response.headers.get ? response.headers.get('Content-Disposition') : null;
        return {blob: blob, filename: filenameFrom(cd, opts.filename || 'download')};
      });
    }, function () { throw ApiError('暂时无法连接服务，下载没有开始。', 0, null); });
  }
  function filenameFrom(disposition, fallback) {
    if (!disposition) return fallback;
    var star = /filename\*\s*=\s*UTF-8''([^;]+)/i.exec(disposition);
    if (star) { try { return decodeURIComponent(star[1].trim().replace(/^"|"$/g, '')); } catch (_) {} }
    var plain = /filename\s*=\s*"?([^";]+)"?/i.exec(disposition);
    return plain ? plain[1].trim() : fallback;
  }
  // Fetch handed to ns.data.createOnlineSource: same-origin credentials, 401 handling, status bookkeeping.
  function dataFetch(url, opts) {
    var o = Object.assign({credentials: 'same-origin'}, opts || {});
    return Promise.resolve(root.fetch(url, o)).then(function (response) {
      state.lastStatus[String(url).split('?')[0]] = response.status;
      if (response.status === 401) unauthorized(String(url));
      return response;
    });
  }
  function query(params) {
    var parts = [];
    Object.keys(params || {}).forEach(function (k) {
      var v = params[k];
      if (v === undefined || v === null || v === '') return;
      parts.push(encodeURIComponent(k) + '=' + encodeURIComponent(String(v)));
    });
    return parts.length ? '?' + parts.join('&') : '';
  }

  // Uploads: XMLHttpRequest for byte progress when available; the error shape matches request().
  function upload(formData, opts) {
    opts = opts || {};
    var url = opts.url || '/api/jobs';
    if (typeof root.XMLHttpRequest === 'undefined' || !opts.onProgress) return request(url, {method: 'POST', body: formData, timeout: opts.timeout || 600000});
    return new Promise(function (resolve, reject) {
      var xhr = new root.XMLHttpRequest();
      xhr.open('POST', url); xhr.timeout = opts.timeout || 600000;
      xhr.setRequestHeader('Accept', 'application/json');
      if (state.csrf) xhr.setRequestHeader('X-CSRF-Token', state.csrf);
      if (xhr.upload) {
        xhr.upload.onprogress = function (ev) { if (ev.lengthComputable) opts.onProgress(ev.loaded, ev.total); };
        xhr.upload.onload = function () { opts.onProgress(null, null); };
      }
      xhr.onerror = function () { reject(ApiError('暂时无法连接服务。已保存的结果不会丢失，恢复连接后可继续。', 0, null)); };
      xhr.ontimeout = function () { reject(ApiError('上传超时。任务可能已经建立，请刷新列表确认。', 0, null)); };
      xhr.onload = function () {
        var data = null;
        try { data = JSON.parse(xhr.responseText); } catch (_) {}
        if (xhr.status >= 200 && xhr.status < 300 && data) { resolve(data); return; }
        if (xhr.status === 401) unauthorized(url);
        reject(ApiError(messageOf(data, xhr.status), xhr.status, data));
      };
      xhr.send(formData);
    });
  }

  function events(onEvent, onState) {
    if (typeof root.EventSource !== 'function') return function () {};
    var source;
    try { source = new root.EventSource('/api/events'); } catch (_) { return function () {}; }
    var handler = function (ev) { var data = null; try { data = JSON.parse(ev.data); } catch (_) {} if (data) onEvent(data); };
    source.addEventListener('job', handler);
    if (onState) { source.onerror = function () { onState('down'); }; source.onopen = function () { onState('up'); }; }
    return function () { try { source.close(); } catch (_) {} };
  }

  var jobPath = function (id, suffix) { return '/api/jobs/' + enc(id) + (suffix || ''); };
  var api = {
    state: state, ApiError: ApiError, request: request, download: download, dataFetch: dataFetch, filenameFrom: filenameFrom, upload: upload, events: events,
    setCsrf: function (token) { state.csrf = token || ''; },
    session: function () { return request('/api/session'); },
    login: function (body) { return request('/api/session', {method: 'POST', body: body}); },
    logout: function () { return request('/api/session/logout', {method: 'POST', body: {}}); },
    releases: function () { return request('/api/releases'); },
    modelCards: function () { return request('/api/v2/model-cards'); },
    jobs: function (params) { return request('/api/jobs' + query(Object.assign({page_size: 100}, params || {}))); },
    job: function (id, opts) { return request(jobPath(id), opts); },
    geometry: function (id) { return request(jobPath(id, '/geometry'), {timeout: 60000}); },
    inputcheck: function (id) { return request('/api/v2/jobs/' + enc(id) + '/inputcheck', {timeout: 60000}); },
    manifestUrl: function (id) { return '/api/v2/jobs/' + enc(id) + '/manifest'; },
    timeline: function (patientId, all) { return request('/api/patients/' + enc(patientId) + '/timeline' + (all ? '?all=1' : '')); },
    confirm: function (id, payload) { return request(jobPath(id, '/confirm'), {method: 'POST', body: payload}); },
    input: function (id, payload) { return request(jobPath(id, '/input'), {method: 'POST', body: payload}); },
    cancel: function (id, payload) { return request(jobPath(id, '/cancel'), {method: 'POST', body: payload || {}}); },
    retry: function (id, payload) { return request(jobPath(id, '/retry'), {method: 'POST', body: payload || {}}); },
    rerun: function (id, payload) { return request(jobPath(id, '/rerun'), {method: 'POST', body: payload}); },
    review: function (id, payload) { return request(jobPath(id, '/review'), {method: 'POST', body: payload}); },
    findingsReview: function (id, payload) { return request(jobPath(id, '/findings_review'), {method: 'PUT', body: payload}); },
    narrative: function (id, payload) { return request(jobPath(id, '/narrative'), {method: 'PUT', body: payload}); },
    annotations: function (id, payload) { return request(jobPath(id, '/annotations'), {method: 'PUT', body: payload}); },
    metadata: function (id, payload) { return request(jobPath(id, '/metadata'), {method: 'POST', body: payload}); },
    offline: function (id, body) { return download('/api/v2/jobs/' + enc(id) + '/offline', {method: 'POST', body: body, filename: 'WSS_病例.html'}); },
    bundle: function (id) { return download(jobPath(id, '/bundle.zip'), {filename: 'wss_' + id + '.zip'}); },
    urls: {
      onepage: function (id) { return jobPath(id, '/onepage'); },
      file: function (id, name) { return jobPath(id, '/files/' + encodeURIComponent(name)); },
      bundle: function (id) { return jobPath(id, '/bundle.zip'); },
      table: function (id, format) { return '/api/jobs/export?ids=' + enc(id) + '&format=' + (format || 'csv'); }
      // S7: the classic pages are gone (urls.report and urls.classic removed)
    }
  };
  return api;
});
