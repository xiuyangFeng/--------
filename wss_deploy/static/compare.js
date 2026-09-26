/* Side-by-side report comparison: two same-origin report iframes with camera linking (contract §6), display-state
   sync (§15.7) and, since v0.14, one shared colour range so that the same colour means the same value on both sides. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const node = (tag, attrs = {}, ...children) => {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === 'class') el.className = value;
      else if (key === 'text') el.textContent = String(value);
      else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
      else el.setAttribute(key, String(value));
    }
    for (const child of children.flat(Infinity)) if (child !== null && child !== undefined && child !== false) el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    return el;
  };
  const fmt = (value, digits = 2) => Number.isFinite(Number(value)) && value !== null && value !== undefined ? Number(value).toFixed(digits) : '—';
  // v0.14: a notice can be closed (×), like the workbench's.
  const clearNotice = () => { $('notice').hidden = true; };
  const notify = (message, error = true) => {
    const el = $('notice');
    el.replaceChildren(node('span', {class: 'notice-text', text: message}), node('button', {type: 'button', class: 'notice-close', text: '×', title: '关闭提示', 'aria-label': '关闭提示', onclick: clearNotice}));
    el.className = `notice${error ? ' error' : ''}`; el.hidden = false;
  };
  const setConnection = (text, online = false) => { $('connection').textContent = text; $('connection').className = `connection${online ? ' online' : ''}`; };
  const ID_PATTERN = /^[A-Za-z0-9_-]{1,80}$/;
  const WB = (typeof window !== 'undefined' && window.WssWorkbenchCore) || (typeof globalThis !== 'undefined' && globalThis.WssWorkbenchCore) || null;
  const state = {csrf: '', left: null, right: null, sync: true, applying: {left: 0, right: 0},
    displaySync: true, families: {left: null, right: null}, active: null, pending: null, requestId: 0, syncTimer: null, lastSync: 0,
    ready: {left: false, right: false}, unify: null, unifyTimer: null};

  async function request(url, {method = 'GET', body} = {}) {
    const headers = {'Accept': 'application/json'};
    if (method !== 'GET' && state.csrf) headers['X-CSRF-Token'] = state.csrf;
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    const response = await fetch(url, {method, headers, body: body === undefined ? undefined : JSON.stringify(body), credentials: 'same-origin'});
    let result;
    try { result = await response.json(); } catch (_) { throw new Error(`服务返回了无法读取的响应（HTTP ${response.status}）。`); }
    if (!response.ok) throw new Error(result.error?.message || result.message || '请求未完成。');
    return result;
  }
  function readQuery() {
    const params = new URLSearchParams(window.location.search);
    const left = params.get('left') || '', right = params.get('right') || '';
    if (!ID_PATTERN.test(left) || !ID_PATTERN.test(right)) throw new Error('缺少有效的 left / right 任务编号。');
    if (left === right) throw new Error('左右两侧必须是不同的任务。');
    return {left, right};
  }
  const reportUrl = id => `/api/jobs/${encodeURIComponent(id)}/report`;
  const identityText = job => {
    const release = job.model_release?.id || job.model_release?.release || '—';
    return [job.case_id && `病例 ${job.case_id}`, job.patient_id && `患者 ${job.patient_id}`, job.scan_label && `扫描 ${job.scan_label}`, job.scan_date && job.scan_date, `发布包 ${release}`, job.review?.status === 'reviewed' ? '已审阅签字' : null].filter(Boolean).join(' · ');
  };
  function renderIdentity(side, job) {
    const el = $(`identity-${side}`); el.replaceChildren(node('strong', {text: side === 'left' ? '左侧' : '右侧'}), node('span', {text: identityText(job)}));
    const release = job.model_release?.id || job.model_release?.release || '';
    $(`head-${side}`).textContent = [`${side === 'left' ? '左侧' : '右侧'} · ${job.case_id || job.id}`, job.patient_id && `患者 ${job.patient_id}`, job.scan_label, release, job.review?.status === 'reviewed' ? '已审阅签字' : null].filter(Boolean).join(' · ');
  }
  function renderComparison(comparison) {
    const host = $('compare-table'); host.replaceChildren();
    if (comparison.reasons?.length) host.append(node('div', {class: 'callout error', text: comparison.reasons.join('；')}));
    else host.append(node('p', {class: 'help', text: `${comparison.kind === 'same_input' ? '同一输入' : '跨病例'} · 仅比较声明一致的标量统计，不做点对点差值。`}));
    const rows = (comparison.rows || []).map(row => node('tr', {}, node('td', {text: row.label || row.id}), node('td', {text: fmt(row.left)}), node('td', {text: fmt(row.right)}), node('td', {text: fmt(row.delta)})));
    host.append(node('table', {}, node('thead', {}, node('tr', {}, ...['指标', '左侧', '右侧', '差值'].map(text => node('th', {text})))), node('tbody', {}, rows)));
  }
  // Camera relay.  A report posts `wss-view:camera` (anatomical-frame coordinates) whenever its camera moves;
  // we forward it to the other frame as `wss-view:set-camera`.  Report pages do not echo while applying a set
  // (contract §6); as a second guard we ignore camera events from a frame for a short window after we pushed
  // a camera into it, which prevents ping-pong even with a report that forgets to suppress.
  const ECHO_GUARD_MS = 150;
  function frameOf(source) {
    for (const side of ['left', 'right']) { const frame = $(`frame-${side}`); if (frame && frame.contentWindow === source) return side; }
    return null;
  }
  // Display-state relay (contract §15.6 / §15.7).  The side that last moved or applied is the source:
  // we ask it for its full view state, keep only the display keys, and push those to the other side.
  // Cameras never travel this way.  A push marks the target so its `applied` reply (and any camera it
  // emits while applying) cannot turn it into the source and bounce the state back.
  const DISPLAY_SYNC_MS = 400;
  const displayGuard = WB ? new WB.EchoGuard(DISPLAY_SYNC_MS, () => Date.now()) : null;
  const otherSide = side => (side === 'left' ? 'right' : 'left');
  const frameWindow = side => $(`frame-${side}`)?.contentWindow || null;
  function requestDisplayState(side) {
    if (!WB || !side) return;
    const target = frameWindow(side); if (!target) return;
    const requestId = `sync-${++state.requestId}`;
    state.pending = {request_id: requestId, from: side};
    target.postMessage({type: 'wss-view:get-state', request_id: requestId}, window.location.origin);
  }
  function scheduleDisplaySync() {
    if (state.syncTimer) return;                                     // at most one in flight
    const wait = Math.max(0, DISPLAY_SYNC_MS - (Date.now() - state.lastSync));
    state.syncTimer = setTimeout(() => {
      state.syncTimer = null; state.lastSync = Date.now();
      if (state.displaySync) requestDisplayState(state.active);
    }, wait);
  }
  function noteActivity(side) {
    if (!side || !state.displaySync || !WB) return;
    if (displayGuard && displayGuard.blocked(side)) return;          // this is our own push coming back
    state.active = side;
    scheduleDisplaySync();
  }
  function pushDisplayState(from, viewState, family) {
    if (!WB) return;
    const to = otherSide(from), target = frameWindow(to); if (!target) return;
    const fromFamily = family || state.families[from] || null, toFamily = state.families[to] || null;
    const sameFamily = !fromFamily || !toFamily || fromFamily === toFamily;
    const subset = WB.compareDisplaySubset(viewState, sameFamily);
    if (!Object.keys(subset).length) return;
    if (displayGuard) displayGuard.mark(to);
    target.postMessage({type: 'wss-view:apply-state', state: subset, request_id: `apply-${++state.requestId}`}, window.location.origin);
    scheduleUnify();                                                 // the pushed range may be the source's own p99
  }
  // ---- v0.14 (F1) shared colour range ----
  // Both frames report their state; WssWorkbenchCore.sharedDisplayRange picks one fixed upper limit (the larger
  // case p99 of the coloured field) that is pushed to both, so the two colour bars are identical.  When that is
  // impossible (different families / fields, volume pages) a short note under the toolbar says so.
  function setScaleNote(text, warn = false) {
    const el = $('scale-note'); if (!el) return;
    el.textContent = text || ''; el.hidden = !text; el.className = `compare-scale-note${warn ? ' warn' : ''}`;
  }
  function unifyRanges() {
    state.unifyTimer = null;
    if (!WB || !state.displaySync || !state.ready.left || !state.ready.right) return;
    const left = frameWindow('left'), right = frameWindow('right'); if (!left || !right) return;
    const id = ++state.requestId; state.unify = {id, states: {}};
    left.postMessage({type: 'wss-view:get-state', request_id: `unify-${id}-left`}, window.location.origin);
    right.postMessage({type: 'wss-view:get-state', request_id: `unify-${id}-right`}, window.location.origin);
  }
  function scheduleUnify(wait = 300) {
    if (state.unifyTimer) clearTimeout(state.unifyTimer);
    state.unifyTimer = setTimeout(unifyRanges, wait);
  }
  function finishUnify(pending) {
    state.unify = null;
    const {left, right} = pending.states;
    const plan = WB.sharedDisplayRange(left.state, right.state, left.family || state.families.left, right.family || state.families.right);
    if (plan.range) {
      if (!plan.unchanged) for (const side of ['left', 'right']) {
        const target = frameWindow(side); if (!target) continue;
        if (displayGuard) displayGuard.mark(side);
        target.postMessage({type: 'wss-view:apply-state', state: {range: plan.range}, request_id: `apply-${++state.requestId}`}, window.location.origin);
      }
      setScaleNote('两侧色标已统一：上限取两例 p99 的较大值，同一颜色表示同一数值。');
    } else if (plan.same) setScaleNote('两侧使用相同的固定色标上限。');
    else setScaleNote(plan.text, true);
  }
  window.addEventListener('message', event => {
    if (event.origin !== window.location.origin) return;
    const data = event.data;
    if (!data || typeof data.type !== 'string') return;
    const from = frameOf(event.source); if (!from) return;
    if (data.type === 'wss-view:ready') { state.families[from] = data.family || null; state.ready[from] = true; if (state.ready.left && state.ready.right) scheduleUnify(150); return; }
    // v0.14: a report announces display changes (field, colour map, thresholds …) without waiting for a camera move.
    if (data.type === 'wss-view:changed') { noteActivity(from); return; }
    if (data.type === 'wss-view:state') {
      const unify = state.unify;
      if (unify && data.request_id === `unify-${unify.id}-${from}`) {
        unify.states[from] = {state: data.state, family: data.family};
        if (unify.states.left && unify.states.right) finishUnify(unify);
        return;
      }
      if (!state.pending || data.request_id !== state.pending.request_id || from !== state.pending.from) return;
      state.pending = null;
      pushDisplayState(from, data.state, data.family);
      return;
    }
    if (data.type === 'wss-view:applied') { noteActivity(from); return; }
    if (data.type !== 'wss-view:camera' || !data.camera) return;
    noteActivity(from);
    if (!state.sync) return;
    if (Date.now() - state.applying[from] < ECHO_GUARD_MS) return;
    const to = otherSide(from);
    const target = frameWindow(to); if (!target) return;
    state.applying[to] = Date.now();
    target.postMessage({type: 'wss-view:set-camera', camera: data.camera, family: data.family}, window.location.origin);
  });
  $('sync-camera').addEventListener('change', () => { state.sync = $('sync-camera').checked; });
  $('sync-display').addEventListener('change', () => {
    state.displaySync = $('sync-display').checked;
    if (!state.displaySync) { setScaleNote('未同步显示口径：两侧色标各自独立，颜色不能直接对比。', true); return; }
    if (state.active) requestDisplayState(state.active);          // the rest of the display state first
    scheduleUnify(state.active ? 350 : 0);
  });
  $('push-left').addEventListener('click', () => { state.active = 'left'; requestDisplayState('left'); });
  $('push-right').addEventListener('click', () => { state.active = 'right'; requestDisplayState('right'); });
  $('swap-sides').addEventListener('click', () => {
    if (!state.left || !state.right) return;
    const params = new URLSearchParams({left: state.right, right: state.left});
    window.location.search = `?${params.toString()}`;
  });
  async function start() {
    let ids;
    try { ids = readQuery(); } catch (error) { notify(error.message); setConnection('参数无效'); return; }
    state.left = ids.left; state.right = ids.right;
    try {
      const session = await request('/api/session');
      if (!session.authenticated) { notify('会话未建立，请先在工作台登录后再打开比较页。'); setConnection('未登录'); return; }
      state.csrf = session.csrf_token || '';
      setConnection('服务已连接', true);
      const [left, right] = await Promise.all([request(`/api/jobs/${encodeURIComponent(ids.left)}`), request(`/api/jobs/${encodeURIComponent(ids.right)}`)]);
      renderIdentity('left', left.job || left); renderIdentity('right', right.job || right);
      for (const [side, job] of [['left', left.job || left], ['right', right.job || right]]) {
        // Seed the family so a display push before `wss-view:ready` already respects wall vs volume.
        const fields = job.summary?.fields;
        state.families[side] = job.family || (fields ? ((fields.velocity || fields.pressure) ? 'volume' : 'wall') : null);
        if (job.status !== 'done') { notify(`${side === 'left' ? '左侧' : '右侧'}任务尚未完成，报告不可用。`); continue; }
        $(`frame-${side}`).src = reportUrl(job.id);
      }
      try { const result = await request('/api/compare', {method: 'POST', body: {left_job_id: ids.left, right_job_id: ids.right}}); renderComparison(result.comparison || result); }
      catch (error) { $('compare-table').replaceChildren(node('div', {class: 'callout error', text: `标量比较不可用：${error.message}`})); }
    } catch (error) { notify(error.message); setConnection('连接失败'); }
  }
  start();
})();
