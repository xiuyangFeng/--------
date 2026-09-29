/* Owner-scoped support portal. The server enforces ownership; this page never requests the ops APIs. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {csrf: '', session: null, page: 1, total: 0, filters: {}, selected: null, dirty: false, busy: false, mutation: false, authenticating: false, pending: false, generation: 0, seq: 0};
  const labels = {open: '待处理', in_progress: '处理中', resolved: '已解决', closed: '已关闭', normal: '普通', high: '高', urgent: '紧急'};
  function node(tag, attrs = {}, ...children) { const el = document.createElement(tag); for (const [key, value] of Object.entries(attrs)) { if (value === undefined || value === null || value === false) continue; if (key === 'text') el.textContent = String(value); else if (key === 'class') el.className = value; else if (key.startsWith('on')) el.addEventListener(key.slice(2), value); else el.setAttribute(key, String(value)); } for (const child of children.flat(Infinity)) if (child !== undefined && child !== null && child !== false) el.append(child instanceof Node ? child : document.createTextNode(String(child))); return el; }
  const fmt = value => { if (!value) return '—'; const d = new Date(value); return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString('zh-CN', {hour12: false}); };
  const button = (text, onclick, cls = '') => node('button', {type: 'button', text, onclick, class: cls});
  function notice(message, error = true) { const el = $('support-notice'); el.hidden = !message; el.className = `notice${error ? ' error' : ''}`; el.setAttribute('role', error ? 'alert' : 'status'); const close = button('×', () => { el.hidden = true; }, 'notice-close'); close.setAttribute('aria-label', '关闭提示'); el.replaceChildren(node('span', {text: message}), close); }
  function mode() { const token = $('support-mode').value === 'token'; $('support-user-fields').hidden = token; $('support-token-wrap').hidden = !token; }
  function session(value) {
    const resetMode = !state.session || state.session.login !== value.login;
    const identity = item => `${Boolean(item?.authenticated)}:${item?.username || ''}:${item?.csrf_token || ''}`;
    if (identity(state.session) !== identity(value)) { state.generation += 1; state.seq += 1; state.selected = null; state.dirty = false; state.page = 1; state.total = 0; state.filters = {}; $('support-rows').replaceChildren(); $('support-create').reset(); $('support-filter').reset(); $('support-page').textContent = '尚未读取'; $('support-prev').disabled = true; $('support-next').disabled = true; $('support-password').value = ''; $('support-token').value = ''; renderDetail(null); $('support-refresh-state').textContent = value.authenticated ? '正在读取工单…' : '请登录后查看工单'; notice(''); }
    state.session = value; state.csrf = value.csrf_token || ''; $('support-login').hidden = value.authenticated; $('support-content').hidden = !value.authenticated; $('support-logout').hidden = !value.authenticated; $('support-actor').textContent = value.authenticated ? value.display_name || value.username || '本机会话' : '尚未登录'; if (resetMode || (value.login === 'password' && !value.legacy_token_allowed)) $('support-mode').value = value.login === 'token' ? 'token' : 'password'; $('support-mode-wrap').hidden = !(value.login === 'password' && value.legacy_token_allowed); mode();
  }
  async function request(url, {method = 'GET', body} = {}) {
    const headers = {Accept: 'application/json'}; if (body !== undefined) headers['Content-Type'] = 'application/json'; if (method !== 'GET' && state.csrf) headers['X-CSRF-Token'] = state.csrf;
    const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), 30000);
    try { const response = await fetch(url, {method, headers, body: body === undefined ? undefined : JSON.stringify(body), credentials: 'same-origin', signal: controller.signal}); let data; try { data = await response.json(); } catch (_) { throw new Error(`服务响应无法读取（HTTP ${response.status}）。`); } if (!response.ok) { const err = new Error(data.error?.message || data.message || `请求失败（HTTP ${response.status}）。`); err.status = response.status; if (err.status === 401 && url !== '/api/session') session({...state.session, authenticated: false}); throw err; } return data; }
    catch (error) { if (error.name === 'AbortError') throw new Error(method === 'GET' ? '读取超时，请检查连接后重试。' : '请求超时，服务端可能已收到操作。请先刷新工单核对结果，再决定是否重试。'); if (error instanceof TypeError) throw new Error(method === 'GET' ? '无法连接服务，稍后将自动重试。' : '连接中断，请先刷新工单核对操作结果，再决定是否重试。'); throw error; }
    finally { clearTimeout(timeout); }
  }
  async function mutate(control, callback) {
    if (state.mutation || state.authenticating) return; state.mutation = true;
    const container = control.closest('.ticket-detail') || control.closest('form');
    const controls = [control, $('support-logout'), ...(container?.querySelectorAll('button,input,select,textarea') || [])];
    const previous = new Map(controls.map(item => [item, item.disabled])); controls.forEach(item => { item.disabled = true; });
    const label = control.textContent; control.textContent = '处理中…';
    try { await callback(); }
    catch (error) { notice(error.status === 409 ? '工单有新进展，草稿已保留。请点击“读取最新回复（保留草稿）”，核对后再发送。' : error.message); }
    finally { state.mutation = false; previous.forEach((disabled, item) => { item.disabled = disabled; }); control.textContent = label; if (state.pending) { state.pending = false; refresh(true); } }
  }
  function renderDetail(ticket, draft = '') {
    const host = $('support-detail'); host.replaceChildren(); if (!ticket) return host.append(node('p', {class: 'help', text: '选择工单查看回复。'}));
    host.append(node('h3', {text: ticket.title}), node('p', {class: 'detail-meta', text: `${ticket.id} · ${labels[ticket.status] || ticket.status} · ${labels[ticket.priority] || ticket.priority} · ${fmt(ticket.created_at)}`}), ticket.job_id ? node('p', {class: 'detail-meta', text: `关联任务：${ticket.job_id}`}) : null, node('p', {text: ticket.description}));
    const messages = node('div', {class: 'ticket-messages'}); for (const item of ticket.messages || []) messages.append(node('article', {}, node('small', {text: `${item.actor} · ${fmt(item.at)}`}), node('p', {text: item.text}))); if (!ticket.messages?.length) messages.append(node('p', {class: 'help', text: '暂无回复，运维人员会在这里更新处理进度。'})); host.append(messages);
    const reply = node('textarea', {rows: 4, maxlength: 4000, placeholder: '补充信息或回复运维人员', 'aria-label': '回复内容'}); reply.value = draft;
    const editStatus = node('p', {class: 'help', role: 'status'}); const dirty = () => { state.dirty = Boolean(reply.value); editStatus.textContent = state.dirty ? '回复尚未发送，自动刷新会保留草稿。' : ''; }; reply.addEventListener('input', dirty); dirty();
    const save = button('发送回复', async () => { if (!reply.value.trim()) return notice('请填写回复内容。');
      await mutate(save, async () => { const generation = state.generation; const result = await request(`/api/support/tickets/${ticket.id}`, {method: 'POST', body: {version: ticket.version, message: reply.value.trim()}}); if (generation !== state.generation) return; state.selected = result.ticket; state.dirty = false; renderDetail(result.ticket); notice('回复已发送。', false); await refresh(true); });
    }, 'primary');
    const reload = button('读取最新回复（保留草稿）', () => mutate(reload, async () => { const generation = state.generation; const result = await request(`/api/support/tickets/${ticket.id}`); if (generation !== state.generation) return; state.selected = result.ticket; renderDetail(result.ticket, reply.value); notice('已读取最新回复并保留草稿，请核对后发送。', false); }));
    host.append(reply, editStatus, node('div', {class: 'ticket-actions'}, save, reload, button('放弃草稿', () => { state.dirty = false; renderDetail(state.selected); })));
  }
  function renderList(result) {
    const host = $('support-rows'); host.replaceChildren(); if (!result.items.length) host.append(node('tr', {}, node('td', {colspan: 3, class: 'ops-empty', text: '没有匹配的工单'})));
    for (const ticket of result.items) host.append(node('tr', {class: state.selected?.id === ticket.id ? 'is-selected' : ''}, node('td', {}, button(ticket.title, () => { if (state.dirty && state.selected?.id !== ticket.id) return notice('请先发送回复，或放弃当前草稿后切换工单。'); if (!state.dirty) { state.selected = ticket; renderDetail(ticket); } }, 'text-button'), node('div', {class: 'muted mono', text: ticket.id})), node('td', {}, node('span', {class: 'status info', text: labels[ticket.status] || ticket.status})), node('td', {class: 'muted', text: fmt(ticket.updated_at)})));
    const fresh = result.items.find(ticket => ticket.id === state.selected?.id); if (fresh && !state.dirty && !$('support-detail').contains(document.activeElement)) { state.selected = fresh; renderDetail(fresh); }
    state.total = result.total; const pages = Math.max(1, Math.ceil(result.total / 25)); $('support-page').textContent = `${state.page} / ${pages}（共 ${result.total} 条）`; $('support-prev').disabled = state.page <= 1; $('support-next').disabled = state.page >= pages;
  }
  async function refresh(manual = false) {
    if (state.busy || state.mutation || state.authenticating) { if (manual) { state.pending = true; state.seq += 1; } return; }
    state.busy = true;
    try {
      const generation = state.generation; const resultSession = await request('/api/session'); if (state.authenticating || generation !== state.generation) return;
      session(resultSession); if (!state.session.authenticated) return; const seq = ++state.seq;
      const query = new URLSearchParams({...state.filters, page: state.page, page_size: 25}); const result = await request(`/api/support/tickets?${query}`); if (seq !== state.seq) return;
      const lastPage = Math.max(1, Math.ceil(result.total / 25)); if (state.page > lastPage) { state.page = lastPage; state.pending = true; return; }
      if (!state.mutation && (manual || !$('support-rows').contains(document.activeElement))) renderList(result); $('support-refresh-state').textContent = `最近刷新：${fmt(new Date())}`;
    }
    catch (error) { notice(error.message); $('support-refresh-state').textContent = '刷新失败，请重试'; }
    finally { state.busy = false; if (state.pending && !state.authenticating && !state.mutation) { state.pending = false; refresh(true); } }
  }
  function bind() {
    $('support-mode').addEventListener('change', mode);
    $('support-login-form').addEventListener('submit', async event => { event.preventDefault(); if (state.authenticating) return; state.authenticating = true; const submit = $('support-login-submit'); submit.disabled = true; $('support-login-error').hidden = true; try { const body = $('support-mode').value === 'token' ? {token: $('support-token').value} : {username: $('support-username').value.trim(), password: $('support-password').value}; session(await request('/api/session', {method: 'POST', body})); $('support-password').value = ''; $('support-token').value = ''; notice(''); state.pending = true; } catch (error) { $('support-login-error').textContent = error.message; $('support-login-error').hidden = false; } finally { submit.disabled = false; state.authenticating = false; if (state.pending) { state.pending = false; refresh(true); } } });
    $('support-logout').addEventListener('click', async () => { if (state.mutation || state.authenticating) return; state.authenticating = true; $('support-logout').disabled = true; try { await request('/api/session/logout', {method: 'POST', body: {}}); session({...state.session, authenticated: false}); session(await request('/api/session')); } catch (error) { notice(error.message); } finally { state.authenticating = false; $('support-logout').disabled = false; } });
    $('support-create').addEventListener('submit', async event => { event.preventDefault(); const body = {title: $('support-title').value.trim(), description: $('support-description').value.trim(), priority: $('support-priority').value}; if ($('support-job').value.trim()) body.job_id = $('support-job').value.trim();
      if (!body.title || !body.description) return notice('请填写工单主题和问题描述。');
      await mutate($('support-create-submit'), async () => { const generation = state.generation; const result = await request('/api/support/tickets', {method: 'POST', body}); if (generation !== state.generation) return; event.target.reset(); if (!state.dirty) { state.selected = result.ticket; renderDetail(result.ticket); } notice(`工单已提交：${result.ticket.id}`, false); state.page = 1; await refresh(true); });
    });
    $('support-filter').addEventListener('submit', event => { event.preventDefault(); state.filters = {q: $('support-query').value.trim(), status: $('support-status').value}; state.page = 1; refresh(true); });
    $('support-refresh').addEventListener('click', () => refresh(true)); $('support-prev').addEventListener('click', () => { state.page = Math.max(1, state.page - 1); refresh(true); }); $('support-next').addEventListener('click', () => { state.page += 1; refresh(true); });
  }
  async function init() { bind(); try { session(await request('/api/session')); await refresh(); } catch (error) { notice(error.message); } setInterval(() => { if (!document.hidden && !state.mutation) refresh(); }, 10000); document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); }); }
  window.addEventListener('DOMContentLoaded', init);
})();
