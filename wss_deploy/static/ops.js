/* Independent administrator console. All rendered data is text; write requests carry the session CSRF token. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const newFeed = () => ({paused: false, pending: false, snapshot: '', historyUntil: '', latestKey: null, renderedKey: null, checkedAt: ''});
  const state = {session: null, csrf: '', allowed: false, busy: false, mutation: false, authenticating: false, refreshPending: false, selected: null, ticketDirty: false, assetDrafts: new Map(), generation: 0, view: 'overview', feed: newFeed(), jobSelection: new Map(), batchRunning: false,
    pages: {jobs: 1, tickets: 1, assets: 1, events: 1, users: 1}, totals: {}, filters: {jobs: {location: 'active'}, tickets: {}, assets: {}, users: {}, events: {page_size: 50}}, sequences: {}, data: {}};
  const labels = {queued: '排队中', running: '计算中', awaiting_input: '待补充输入', awaiting_confirmation: '待确认', awaiting_outlets: '待确认出口', done: '已完成', failed: '失败', interrupted: '已中断', cancelled: '已取消', open: '待处理', in_progress: '处理中', resolved: '已解决', closed: '已关闭', active: '可用', disabled: '已停用', candidate: '待审阅', approved: '可复用', excluded: '已排除', normal: '普通', high: '高', urgent: '紧急'};
  const actions = {created: '提交了任务', started: '开始计算', finished: '计算完成', failed: '计算失败', deleted: '删除了任务', delete: '删除了任务', restored: '恢复了任务', restore: '恢复了任务', purged: '永久清理了任务', purge: '永久清理了任务', login: '登录成功', logout: '退出登录', login_failed: '登录失败', login_throttled: '登录尝试受到限制', password_changed: '更新了口令', ticket_created: '提交了工单', ticket_updated: '更新了工单', ticket_message: '添加了工单回复', asset_archived: '归档了 STL', asset_reviewed: '更新了素材审阅', asset_downloaded: '下载了素材包', user_disabled: '暂停了账号访问', user_enabled: '恢复了账号访问', user_add: '创建了账号', user_passwd: '重置了账号口令', user_role: '调整了账号角色', user_role_changed: '调整了账号角色', user_disable: '暂停了账号访问', user_enable: '恢复了账号访问', jobs_claim: '批量认领了任务', cancel_requested: '请求取消任务', claim: '认领了任务', owner_claimed: '认领了任务', restart_requeued: '重启后重新排队', restart_interrupted: '任务因重启中断', drain_requeued: '任务重新排队', annotations_updated: '更新了标注', findings_review_updated: '更新了结果审阅', narrative_updated: '更新了报告说明', snapshots_updated: '更新了截图', metadata_updated: '更新了任务信息', companion_failed: '配套计算失败', companion_created: '创建了配套计算', rerun_created: '提交了重新计算', progress: '更新了计算进度', device_fallback: '切换了计算设备', attempt_aborted: '中止了本次计算', analysis_rebuilt: '更新了分析结果', template_updated: '更新了模板', cancelled: '取消了任务', interrupted: '任务已中断'};
  const categories = {job: '任务', auth: '登录与认证', ticket: '工单', asset: 'STL 素材', user: '用户管理'};
  const severities = {info: '一般', warning: '需关注', error: '错误'};
  function node(tag, attrs = {}, ...children) {
    const element = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === 'text') element.textContent = String(value);
      else if (key === 'class') element.className = value;
      else if (key.startsWith('on')) element.addEventListener(key.slice(2), value);
      else element.setAttribute(key, String(value));
    }
    for (const child of children.flat(Infinity)) if (child !== undefined && child !== null && child !== false) element.append(child instanceof Node ? child : document.createTextNode(String(child)));
    return element;
  }
  const str = value => value === undefined || value === null || value === '' ? '—' : String(value);
  const fmt = value => { if (!value) return '—'; const date = new Date(value); return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('zh-CN', {hour12: false}); };
  const count = value => Number.isFinite(Number(value)) ? Number(value).toLocaleString('zh-CN') : '—';
  const button = (text, handler, className = '') => node('button', {type: 'button', text, class: className, onclick: handler});
  const badge = value => node('span', {class: `status ${['done','resolved','approved','active'].includes(value) ? 'good' : ['failed','disabled','excluded','interrupted'].includes(value) ? 'bad' : 'info'}`, text: labels[value] || str(value)});
  function options(values, selected) { const select = node('select'); values.forEach(value => select.append(node('option', {value, text: labels[value] || value}))); select.value = selected || values[0]; return select; }
  function notice(message, error = true) { const el = $('ops-notice'); el.hidden = !message; el.className = `notice${error ? ' error' : ''}`; el.setAttribute('role', error ? 'alert' : 'status'); const close = button('×', () => { el.hidden = true; }, 'notice-close'); close.setAttribute('aria-label', '关闭提示'); el.replaceChildren(node('span', {text: message}), close); }
  function connection(message, online = false) { $('ops-connection').textContent = message; $('ops-connection').className = `connection${online ? ' online' : ''}`; $('ops-refresh-dot').className = `ops-refresh-dot${online ? ' online' : ''}`; }
  function showView(view) {
    if (!$(`view-${view}`)) return;
    state.view = view;
    if (location.hash !== `#${view}`) history.replaceState(null, '', `#${view}`);
    document.querySelectorAll('#ops-nav [data-view]').forEach(control => { if (control.dataset.view === view) control.setAttribute('aria-current', 'page'); else control.removeAttribute('aria-current'); });
    document.querySelectorAll('.ops-view').forEach(panel => { panel.hidden = panel.dataset.panel !== view; });
  }
  function feedStatus() {
    $('events-pause').textContent = state.feed.paused ? '恢复更新' : '暂停更新'; $('events-pause').setAttribute('aria-pressed', String(state.feed.paused));
    $('events-live-dot').className = `ops-refresh-dot${state.allowed && !state.feed.paused ? ' online' : ''}`;
    $('events-stream-status').textContent = !state.allowed ? '等待管理员登录' : `${state.feed.paused ? '已暂停展示更新，仍检查新动态' : '每 10 秒检查新记录'}${state.feed.historyUntil ? ' · 正在浏览历史记录' : ''}${state.feed.checkedAt ? ` · 最近检查 ${new Date(state.feed.checkedAt).toLocaleTimeString('zh-CN', {hour12: false})}` : ''}`;
    $('events-new').hidden = !state.feed.pending; $('events-nav-dot').hidden = !state.feed.pending;
    $('events-advanced').querySelector('summary').textContent = state.filters.events.since || state.filters.events.until || state.filters.events.action || Number(state.filters.events.page_size) !== 50 ? '更多筛选（已应用）：时间、操作代码与每页条数' : '更多筛选：时间、操作代码与每页条数';
  }
  function clearJobSelection() {
    state.jobSelection.clear(); $('jobs-batch-results').replaceChildren(); $('jobs-batch-results').hidden = true; $('jobs-batch-status').hidden = true; updateJobSelection();
  }
  function updateJobSelection() {
    const size = state.jobSelection.size; $('jobs-selection-count').textContent = size ? `已选择 ${size} 个任务` : '未选择任务';
    $('jobs-archive-selected').disabled = !size || state.batchRunning; $('jobs-clear-selection').disabled = !size || state.batchRunning;
    const controls = [...document.querySelectorAll('#jobs-rows .job-select')]; controls.forEach(control => { control.checked = state.jobSelection.has(control.dataset.selectionKey); control.disabled = state.batchRunning; });
    const checked = controls.filter(control => control.checked).length; $('jobs-select-page').checked = Boolean(controls.length && checked === controls.length); $('jobs-select-page').indeterminate = checked > 0 && checked < controls.length; $('jobs-select-page').disabled = !controls.length || state.batchRunning;
  }
  function applySession(session) {
    const resetLoginMode = !state.session || state.session.login !== session.login;
    const identity = value => `${Boolean(value?.authenticated)}:${value?.username || ''}:${value?.role || ''}:${value?.csrf_token || ''}`;
    if (identity(state.session) !== identity(session)) {
      state.generation += 1; state.selected = null; state.ticketDirty = false; state.data = {}; state.assetDrafts.clear(); state.feed = newFeed(); state.batchRunning = false; clearJobSelection();
      state.pages = {jobs: 1, tickets: 1, assets: 1, events: 1, users: 1}; state.totals = {};
      state.filters = {jobs: {location: 'active'}, tickets: {}, assets: {}, users: {}, events: {page_size: 50}};
      for (const name of ['jobs','tickets','events','assets','users']) { $(`${name}-rows`).replaceChildren(); $(`${name}-page`).textContent = '尚未读取'; $(`${name}-prev`).disabled = true; $(`${name}-next`).disabled = true; $(`${name}-filter`)?.reset(); }
      $('login-password').value = ''; $('login-token').value = '';
      renderTicketDetail(null); $('overview-cards').replaceChildren(node('p', {class: 'help', text: '正在读取概览…'})); $('overview-warnings').replaceChildren(); $('overview-warnings').hidden = true; $('events-window-note').hidden = true; $('ops-health').textContent = ''; $('health-facts')?.remove(); $('asset-job').value = ''; $('ops-last-refresh').textContent = '尚未刷新'; showView(['overview','events','jobs','tickets','assets','users'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'overview'); notice('');
    }
    state.session = session; state.csrf = session.csrf_token || ''; state.allowed = Boolean(session.authenticated && (!session.shared || session.role === 'admin'));
    $('ops-content').hidden = !state.allowed; $('ops-login').hidden = state.allowed; $('ops-logout').hidden = !session.authenticated; $('ops-refresh').disabled = !state.allowed;
    $('ops-actor').textContent = session.authenticated ? `${session.display_name || session.username || '本机运维员'}${session.role === 'admin' ? ' · 管理员' : ''}` : ''; $('ops-actor').hidden = !session.authenticated;
    if (resetLoginMode || (session.login === 'password' && !session.legacy_token_allowed)) $('login-mode').value = session.login === 'token' ? 'token' : 'password'; $('login-mode-wrap').hidden = !(session.login === 'password' && session.legacy_token_allowed);
    $('login-help').textContent = session.authenticated && !state.allowed ? '当前账号没有运维权限。请使用管理员账号登录；用户反馈可进入工单入口。' : '登录管理员账号后查看跨账号任务与审计信息。'; loginMode();
    connection(state.allowed ? '已登录' : '等待管理员登录', state.allowed); feedStatus();
  }
  function loginMode() { const token = $('login-mode').value === 'token'; $('login-user-fields').hidden = token; $('login-token-wrap').hidden = !token; }
  async function request(url, {method = 'GET', body} = {}) {
    const generation = state.generation;
    const headers = {Accept: 'application/json'}; if (body !== undefined) headers['Content-Type'] = 'application/json'; if (method !== 'GET' && state.csrf) headers['X-CSRF-Token'] = state.csrf;
    const abort = new AbortController(); const timer = setTimeout(() => abort.abort(), 30000);
    try {
      const response = await fetch(url, {method, headers, credentials: 'same-origin', body: body === undefined ? undefined : JSON.stringify(body), signal: abort.signal});
      let data; try { data = await response.json(); } catch (_) { throw new Error(`服务响应无法读取（HTTP ${response.status}）。`); }
      if (generation !== state.generation) { const error = new Error('会话已变更，已忽略旧请求。'); error.stale = true; throw error; }
      if (!response.ok) { const err = new Error(data.error?.message || data.message || `请求失败（HTTP ${response.status}）。`); err.status = response.status; if (response.status === 401 && url !== '/api/session') applySession({...state.session, authenticated: false}); if (response.status === 403 && url.startsWith('/api/ops/')) applySession({...state.session, authenticated: false}); throw err; } return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error(method === 'GET' ? '读取超时，请检查连接后重试。' : '请求超时，服务端可能已收到操作。请先刷新核对结果，再决定是否重试。');
      if (error instanceof TypeError) throw new Error(method === 'GET' ? '无法连接服务，稍后将自动重试。' : '连接中断，请先刷新核对操作结果，再决定是否重试。');
      throw error;
    } finally { clearTimeout(timer); }
  }
  function pager(section, result) {
    state.totals[section] = result.total; const size = section === 'events' ? Number(state.filters.events.page_size) : 25; const pages = Math.max(1, Math.ceil(result.total / size));
    $(`${section}-page`).textContent = `${state.pages[section]} / ${pages}（共 ${count(result.total)} 条）${section === 'events' && result.truncated ? ` · 仅最近 ${count(result.history_window_limit || 10000)} 条匹配事件，请缩小筛选范围` : ''}`; $(`${section}-prev`).disabled = state.pages[section] <= 1; $(`${section}-next`).disabled = state.pages[section] >= pages;
  }
  function empty(id, columns, message) { $(id).replaceChildren(node('tr', {}, node('td', {colspan: columns, class: 'ops-empty', text: message}))); }
  function jumpTo(section, filters = {}) {
    state.filters[section] = filters; state.pages[section] = 1; $(`${section}-filter`)?.reset();
    const fields = {jobs: {q: 'query', owner: 'user', status: 'status', location: 'location', expires_within_days: 'expiring'}, tickets: {q: 'query', owner: 'user', status: 'status', unassigned: 'unassigned'}, assets: {q: 'query', status: 'status'}, users: {q: 'query', disabled: 'disabled'}};
    for (const [key, suffix] of Object.entries(fields[section] || {})) { const field = $(`${section}-${suffix}`); if (field) field.value = filters[key] ?? (key === 'location' ? 'active' : ''); }
    if (section === 'jobs') clearJobSelection();
    showView(section); run(refreshSection(section, true));
  }
  function renderOverview(result) {
    const c = result.counts; const metrics = [['当前任务', c.jobs, `其中 ${count(c.active)} 条排队或计算中`, 'jobs', {location: 'active'}], ['失败任务', c.failed, '查看故障和原始提交', 'jobs', {location: 'active', status: 'failed'}], ['未结工单', c.tickets_open, '待处理及处理中', 'tickets', {status: 'unresolved'}], ['待分派工单', c.unassigned_tickets, '尚未指定处理人', 'tickets', {status: 'unresolved', unassigned: 'true'}], ['待审阅 STL', c.pending_review_assets, '审阅后决定是否复用', 'assets', {status: 'candidate'}], ['STL 素材', c.assets, '已归档几何与来源', 'assets', {}], ['7 天内到期', c.trash_expiring, '含回收站中已过期记录', 'jobs', {location: 'trash', expires_within_days: '7'}], ['用户账号', c.users, '查看访问状态', 'users', {}]];
    $('overview-cards').replaceChildren(...metrics.map(([title, value, note, section, filters]) => node('button', {type: 'button', class: 'overview-tile', 'data-shortcut': title, onclick: () => jumpTo(section, filters)}, node('span', {class: 'tile-label', text: title}), node('strong', {class: 'tile-value', text: count(value)}), node('span', {class: 'tile-note', text: note}))));
    const warnings = $('overview-warnings'); warnings.replaceChildren();
    if (Number(c.trash_expiring)) warnings.append(button(`${count(c.trash_expiring)} 个回收站任务将在 7 天内到期，查看并归档 STL`, () => jumpTo('jobs', {location: 'trash', expires_within_days: '7'}), 'ops-warning-link'));
    if (Number(c.unassigned_tickets)) warnings.append(button(`${count(c.unassigned_tickets)} 个未结工单等待分派`, () => jumpTo('tickets', {status: 'unresolved', unassigned: 'true'}), 'ops-warning-link'));
    warnings.hidden = !warnings.childElementCount;
    $('overview-updated').textContent = `统计时间：${fmt(new Date())}`;
    const health = result.health || {}; $('ops-health').textContent = `服务健康：${health.ok === true ? '正常' : health.ok === false ? '异常' : '未知'}${health.version ? ` · 版本 ${health.version}` : ''}`;
    const existing = $('health-facts'); if (existing) existing.querySelector('pre').textContent = JSON.stringify(health, null, 2);
    else $('ops-health').after(node('details', {id: 'health-facts', class: 'ops-health-details'}, node('summary', {text: '查看服务健康详情'}), node('pre', {text: JSON.stringify(health, null, 2)})));
  }
  function renderJobs(result) {
    const rows = result.items; const location = state.filters.jobs.location; const host = $('jobs-rows'); host.replaceChildren(); if (!rows.length) empty('jobs-rows', 7, '没有匹配的任务');
    rows.forEach(job => {
      const action = button('归档 STL', event => archive(job.id, location, event.currentTarget));
      const key = `${location}:${job.id}`;
      const selected = node('input', {type: 'checkbox', class: 'job-select', 'data-selection-key': key, 'data-job-id': job.id, 'aria-label': `选择任务 ${job.case_id || job.id}`}); selected.checked = state.jobSelection.has(key);
      selected.addEventListener('change', () => { if (selected.checked) state.jobSelection.set(key, {job_id: job.id, location, title: job.case_id || job.id}); else state.jobSelection.delete(key); updateJobSelection(); });
      let failure = null;
      if (job.error) {
        const message = typeof job.error === 'string' ? job.error : job.error.message || job.error.public_message || '任务执行失败';
        failure = node('details', {class: 'ops-job-error'}, node('summary', {text: '故障详情'}), node('p', {text: message}), job.error.diagnostic_id ? node('p', {class: 'mono', text: `诊断编号：${job.error.diagnostic_id}`}) : null);
      }
      host.append(node('tr', {'data-job-id': job.id}, node('td', {}, selected), node('td', {}, node('strong', {text: job.case_id || job.id}), node('div', {class: 'muted mono', text: job.id}), node('div', {class: 'muted', text: job.source_filename || ''}), node('div', {class: 'muted', text: `发布包：${str(job.model_release)}`}), failure), node('td', {text: str(job.owner)}), node('td', {}, badge(job.status)), node('td', {class: 'muted', text: fmt(job.created_at)}), node('td', {class: 'muted', text: fmt(job.expires_at)}), node('td', {class: 'asset-actions'}, action)));
    }); pager('jobs', result); updateJobSelection();
  }
  function renderTickets(result) {
    const rows = result.items; const host = $('tickets-rows'); host.replaceChildren(); if (!rows.length) empty('tickets-rows', 4, '没有匹配的工单');
    rows.forEach(ticket => host.append(node('tr', {class: ticket.id === state.selected?.id ? 'is-selected' : ''}, node('td', {}, button(ticket.title, () => {
      if (state.ticketDirty) { if (state.selected?.id !== ticket.id) notice('请先保存当前处理意见，或点击“放弃草稿”后切换工单。'); return; } state.selected = ticket; state.ticketDirty = false; renderTicketDetail(ticket); renderTickets(result);
    }, 'text-button'), node('div', {class: 'muted mono', text: ticket.id})), node('td', {text: ticket.owner}), node('td', {}, badge(ticket.status)), node('td', {class: 'muted', text: fmt(ticket.updated_at)}))));
    const fresh = rows.find(item => item.id === state.selected?.id); if (fresh && !state.ticketDirty && !$('ticket-detail').contains(document.activeElement)) { state.selected = fresh; renderTicketDetail(fresh); } pager('tickets', result);
  }
  function renderTicketDetail(ticket, draft = null) {
    const host = $('ticket-detail'); host.replaceChildren(); if (!ticket) return host.append(node('p', {class: 'help', text: '选择工单查看详情。'}));
    const status = options(['open','in_progress','resolved','closed'], draft?.status ?? ticket.status); status.setAttribute('aria-label','处理状态');
    const priority = options(['normal','high','urgent'], draft?.priority ?? ticket.priority); priority.setAttribute('aria-label','优先级');
    const assignee = node('input', {value: draft?.assignee ?? ticket.assignee ?? '', maxlength: 128, placeholder: '处理人', 'aria-label': '处理人'});
    const message = node('textarea', {rows: 3, maxlength: 4000, placeholder: '回复用户（可选）', 'aria-label': '回复用户'});
    message.value = draft?.message || '';
    const editStatus = node('p', {class: 'help', role: 'status'});
    const dirty = () => { state.ticketDirty = status.value !== ticket.status || priority.value !== ticket.priority || assignee.value !== (ticket.assignee || '') || Boolean(message.value); editStatus.textContent = state.ticketDirty ? '有未保存的处理意见，自动刷新会保留草稿。' : '处理意见已同步。'; };
    [status, priority, assignee, message].forEach(el => el.addEventListener('input', dirty)); dirty();
    const save = button('保存处理', async () => {
      const body = {version: ticket.version, status: status.value, priority: priority.value, assignee: assignee.value.trim()}; if (message.value.trim()) body.message = message.value.trim();
      await mutate(save, async () => { const result = await request(`/api/ops/tickets/${ticket.id}`, {method: 'POST', body}); state.selected = result.ticket; state.ticketDirty = false; renderTicketDetail(result.ticket); notice('工单已更新。', false); await refreshSection('tickets', true); });
    }, 'primary');
    host.append(node('h3', {text: ticket.title}), node('p', {class: 'detail-meta', text: `${ticket.id} · ${ticket.owner} · ${fmt(ticket.created_at)}${ticket.job_id ? ` · 任务 ${ticket.job_id}` : ''}`}), node('p', {text: ticket.description}));
    const reload = button('读取最新进展（保留草稿）', () => mutate(reload, async () => {
      const edits = {message: message.value};
      if (status.value !== ticket.status) edits.status = status.value;
      if (priority.value !== ticket.priority) edits.priority = priority.value;
      if (assignee.value !== (ticket.assignee || '')) edits.assignee = assignee.value;
      const result = await request(`/api/ops/tickets?${new URLSearchParams({q: ticket.id, page: 1, page_size: 25})}`);
      const current = result.items.find(item => item.id === ticket.id);
      if (!current) throw new Error('工单已不可用，请刷新列表。');
      state.selected = current; renderTicketDetail(current, edits); notice('已读取最新进展并保留草稿，请核对后保存。', false);
    }));
    const history = node('div', {class: 'ticket-messages'}); (ticket.messages || []).forEach(item => history.append(node('article', {}, node('small', {text: `${item.actor} · ${fmt(item.at)}`}), node('p', {text: item.text}))));
    if (!ticket.messages?.length) history.append(node('p', {class: 'help', text: '暂无回复。'}));
    host.append(history, node('div', {class: 'ticket-actions'}, status, priority, assignee), message, editStatus, node('div', {class: 'ticket-actions'}, save, reload, button('放弃草稿', () => { state.ticketDirty = false; renderTicketDetail(state.selected); })));
  }
  const eventKey = event => String(event.id ?? `${event.at}|${event.action}|${event.job}|${event.actor}|${event.source}`);
  const eventSignature = result => JSON.stringify(result.items.map(eventKey));
  function eventDay(value) {
    const date = new Date(value); if (Number.isNaN(date.getTime())) return '时间未记录';
    const today = new Date(); const yesterday = new Date(); yesterday.setDate(today.getDate() - 1);
    const day = date.toLocaleDateString('zh-CN'); return `${day === today.toLocaleDateString('zh-CN') ? '今天 · ' : day === yesterday.toLocaleDateString('zh-CN') ? '昨天 · ' : ''}${day}`;
  }
  function renderEvents(result) {
    const host = $('events-rows'); host.replaceChildren(); if (!result.items.length) host.append(node('p', {class: 'ops-empty', text: '没有匹配的事件，可调整账号、类别或时间范围。'}));
    let previousDay = ''; let group;
    result.items.forEach(event => {
      const day = eventDay(event.at);
      if (day !== previousDay) { group = node('section', {class: 'ops-event-day'}, node('h3', {text: day})); host.append(group); previousDay = day; }
      const category = categories[event.category] || '任务'; const severity = severities[event.severity] ? event.severity : 'info';
      const actor = event.actor === 'system' ? '系统' : event.actor || '操作者未记录';
      const shorten = (value, length = 12) => String(value).length > length ? `${String(value).slice(0, length)}…` : String(value);
      const owner = event.owner || event.details?.owner; const context = [actor]; if (owner && owner !== event.actor) context.push(`所属 ${owner}`);
      if (event.details?.case_id) context.push(`病例 ${shorten(event.details.case_id, 24)}`);
      else if (event.job || event.details?.job_id) context.push(`任务 ${shorten(event.job || event.details.job_id)}`);
      else if (event.details?.ticket_id) context.push(`工单 ${shorten(event.details.ticket_id)}`);
      else if (event.details?.sha256) context.push(`素材 ${shorten(event.details.sha256)}`);
      else if (event.details?.username && event.details.username !== owner) context.push(`账号 ${event.details.username}`);
      const summary = node('summary', {class: 'ops-event-summary'}, node('span', {class: `ops-event-indicator ${severity}`, 'aria-hidden': 'true'}), node('span', {class: 'ops-event-copy'}, node('strong', {class: 'ops-event-action', text: actions[event.action] || '记录了一项操作'}), node('span', {class: 'ops-event-actor', text: context.join(' · ')})), node('span', {class: 'ops-event-tags'}, node('span', {class: 'ops-event-category', text: category}), node('span', {class: `ops-event-severity ${severity}`, text: severities[severity]})), node('time', {datetime: event.at || '', class: 'ops-event-time', text: fmt(event.at)}));
      const metadata = node('dl', {class: 'ops-event-metadata'});
      const add = (label, value) => { if (value !== undefined && value !== null && value !== '') metadata.append(node('dt', {text: label}), node('dd', {text: typeof value === 'object' ? JSON.stringify(value) : value})); };
      add('发生时间', fmt(event.at)); add('操作账号', actor); add('操作代码', event.action); add('任务 ID', event.job || event.details?.job_id); add('所属账号', event.owner || event.details?.owner); add('日志来源', event.source); if (event.status && !event.details?.status) add('状态', labels[event.status] || event.status);
      const fieldLabels = {reason: '操作原因', message: '说明', error: '错误信息', filename: '文件名', source_filename: '原始文件名', case_id: '病例', ticket_id: '工单 ID', sha256: '素材指纹', status: '状态', previous_status: '原状态', review_status: '审阅状态', location: '任务位置', assignee: '处理人', username: '目标账号', target: '目标账号', role: '账号角色', claimed: '认领数量', note: '备注', title: '主题', text: '内容', count: '数量', diagnostic_id: '诊断编号'};
      for (const [key, label] of Object.entries(fieldLabels)) { const value = event.details?.[key]; add(label, ['status','previous_status','review_status'].includes(key) ? labels[value] || value : value); }
      const raw = node('details', {class: 'ops-event-raw event-raw'}, node('summary', {text: '查看原始 JSON'}), node('pre', {text: JSON.stringify(event, null, 2)}));
      const entry = node('details', {class: 'ops-event event-entry', 'data-event-id': eventKey(event), 'data-event-key': eventKey(event)}, summary, node('div', {class: 'ops-event-detail'}, metadata, raw));
      entry.addEventListener('toggle', feedStatus); group.append(entry);
    });
    state.feed.renderedKey = eventSignature(result); state.feed.snapshot = result.snapshot_at || new Date().toISOString();
    $('events-window-note').hidden = !result.truncated; $('events-window-note').textContent = result.truncated ? `当前结果受最近 ${count(result.history_window_limit || 10000)} 条匹配事件上限限制，请缩小账号或时间范围后查询、导出。` : '';
    pager('events', result); feedStatus();
  }
  function feedProtected() {
    const host = $('events-rows');
    return Boolean(state.feed.historyUntil || state.pages.events > 1 || host.querySelector('.event-entry[open]') || host.contains(document.activeElement) || (state.view === 'events' && host.getBoundingClientRect().top < 80));
  }
  async function refreshEvents(force = false) {
    if (!state.allowed) return;
    const seq = state.sequences.events = (state.sequences.events || 0) + 1; const generation = state.generation;
    const page = force ? state.pages.events : 1; const filters = {...state.filters.events};
    if (force && state.feed.historyUntil && (!filters.until || new Date(state.feed.historyUntil) < new Date(filters.until))) filters.until = state.feed.historyUntil;
    const params = new URLSearchParams({...filters, page});
    try {
      const result = await request(`/api/ops/events?${params}`);
      if (seq !== state.sequences.events || generation !== state.generation || !state.allowed) return;
      state.feed.checkedAt = new Date().toISOString();
      if (!force) {
        const signature = eventSignature(result); const changed = state.feed.latestKey !== null && signature !== state.feed.latestKey;
        state.feed.latestKey = signature;
        if (state.data.events && (feedProtected() || state.feed.paused || state.mutation)) { if (changed) state.feed.pending = true; feedStatus(); return; }
        if (state.data.events && state.feed.renderedKey === signature) { feedStatus(); return; }
      }
      const last = Math.max(1, Math.ceil(result.total / Number(state.filters.events.page_size)));
      if (page > last) { state.pages.events = last; return refreshEvents(true); }
      if (!state.feed.historyUntil && page === 1) { state.feed.latestKey = eventSignature(result); state.feed.pending = false; }
      state.data.events = result; renderEvents(result);
    } catch (error) { if (seq !== state.sequences.events || generation !== state.generation || error.stale) return; $('events-stream-status').textContent = `读取失败：${error.message}`; throw error; }
  }
  function latestEvents() { state.feed.historyUntil = ''; state.feed.pending = false; state.feed.paused = false; state.pages.events = 1; feedStatus(); run(refreshEvents(true)); }
  async function exportEvents() {
    const generation = state.generation; const control = $('events-export'); control.disabled = true;
    const {page_size, ...filters} = state.filters.events;
    try {
      const result = await request(`/api/ops/events/export?${new URLSearchParams(filters)}`);
      if (generation !== state.generation || !state.allowed) return;
      const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: 'application/json'}));
      const link = node('a', {href: url, download: `wss-events-${new Date().toISOString().replace(/[:.]/g, '-')}.json`}); document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      notice(`已导出 ${count(result.exported_count ?? result.events?.length)} 条事件${result.truncated ? '；达到导出上限，请缩小筛选范围以获得完整记录' : ''}。`, Boolean(result.truncated));
    } catch (error) { if (generation === state.generation && !error.stale) notice(error.message); }
    finally { control.disabled = false; }
  }
  function renderAssets(result) {
    const host = $('assets-rows'); const opened = new Set([...host.querySelectorAll('details[open][data-disclosure]')].map(item => item.dataset.disclosure)); host.replaceChildren(); if (!result.items.length) empty('assets-rows', 5, '没有匹配的素材');
    result.items.forEach(asset => {
      const draft = state.assetDrafts.get(asset.sha256);
      const review = options(['candidate','approved','excluded'], draft?.review_status || asset.review_status); review.setAttribute('aria-label', `素材 ${asset.sha256.slice(0,8)} 审阅状态`);
      const note = node('input', {value: draft?.note ?? asset.note ?? '', maxlength: 2000, placeholder: '审阅备注', 'aria-label': '审阅备注'});
      const remember = () => state.assetDrafts.set(asset.sha256, {review_status: review.value, note: note.value, version: draft?.version ?? asset.version});
      [review, note].forEach(control => control.addEventListener('input', remember));
      const save = button('保存', () => mutate(save, async () => { await request(`/api/ops/assets/${asset.sha256}`, {method: 'POST', body: {review_status: review.value, version: draft?.version ?? asset.version, note: note.value.trim()}}); state.assetDrafts.delete(asset.sha256); notice('素材审阅已保存。', false); await refreshSection('assets', true); }));
      const discard = button('放弃修改', () => { state.assetDrafts.delete(asset.sha256); run(refreshSection('assets', true)); });
      const reload = button('刷新版本', () => mutate(reload, async current => {
        const result = await request(`/api/ops/assets?${new URLSearchParams({q: asset.sha256, page: 1, page_size: 25})}`);
        const latest = result.items.find(item => item.sha256 === asset.sha256);
        if (!latest) throw new Error('素材已不可用，请刷新列表。');
        if (state.assetDrafts.has(asset.sha256)) state.assetDrafts.set(asset.sha256, {...state.assetDrafts.get(asset.sha256), version: latest.version});
        await refreshSection('assets', true); if (current()) notice(`已读取最新版本并保留草稿。服务端状态：${labels[latest.review_status]}；备注：${latest.note || '无'}。请核对后保存。`, false);
      }));
      const sources = asset.sources || [];
      const provenanceKey = `${asset.sha256}:sources`; const reviewKey = `${asset.sha256}:review`;
      const provenance = node('details', {class: 'ops-asset-provenance', 'data-disclosure': provenanceKey, open: opened.has(provenanceKey)}, node('summary', {text: asset.sources_total > sources.length ? `查看来源（显示前 ${sources.length} / 共 ${count(asset.sources_total)} 条）` : `查看 ${sources.length} 条来源`}), node('p', {class: 'mono', text: `SHA-256：${asset.sha256}`}), ...sources.map(source => node('div', {class: 'ops-asset-source'}, node('p', {text: `病例：${source.case_id || '未记录'} · 账号：${source.owner || '未记录'}`}), node('p', {class: 'mono', text: `任务：${source.job_id || '未记录'}`}), node('p', {text: `发布包：${source.release_id || '未记录'} · 归档：${fmt(source.archived_at)}`}))));
      const editor = node('details', {class: 'ops-asset-review', 'data-disclosure': reviewKey, open: opened.has(reviewKey) || Boolean(draft)}, node('summary', {text: draft ? '审阅素材（有草稿）' : '审阅素材'}), node('label', {}, '审阅状态', review), node('label', {}, '审阅备注', note), node('div', {class: 'asset-actions'}, save, reload, discard));
      host.append(node('tr', {'data-asset-id': asset.sha256}, node('td', {}, node('strong', {text: `${asset.sha256.slice(0,12)}…`}), provenance), node('td', {text: sources.map(source => source.owner).filter((value, index, all) => all.indexOf(value) === index).join('、') || '—'}), node('td', {class: 'muted', text: `${(asset.bytes / 1048576).toFixed(2)} MiB`}), node('td', {}, badge(asset.review_status), editor), node('td', {class: 'asset-actions'}, node('a', {href: `/api/ops/assets/${asset.sha256}/download`, text: '下载包'}))));
    }); pager('assets', result);
  }
  function renderUsers(result) {
    const host = $('users-rows'); host.replaceChildren(); if (!result.items.length) empty('users-rows', 6, Object.keys(state.filters.users).length ? '没有匹配的账号，请调整搜索或账号状态。' : state.session.shared ? '暂无已注册账号。' : '本机模式无需注册用户，可直接使用运维中心。');
    result.items.forEach(user => { const action = button(user.disabled ? '恢复访问' : '暂停访问', () => mutate(action, async () => { const result = await request(`/api/ops/users/${encodeURIComponent(user.username)}`, {method: 'POST', body: {disabled: !user.disabled}}); notice(result.warning?.message || '账号状态已更新。', Boolean(result.warning)); await refreshSection('users', true); })); action.disabled = user.username === state.session.username;
      host.append(node('tr', {}, node('td', {}, node('strong', {text: user.display_name}), node('div', {class: 'muted mono', text: user.username})), node('td', {text: user.role === 'admin' ? '管理员' : '用户'}), node('td', {class: 'muted', text: fmt(user.created_at)}), node('td', {text: `${count(user.jobs)} / ${count(user.failed)}`}), node('td', {}, badge(user.disabled ? 'disabled' : 'active')), node('td', {class: 'user-actions'}, action)));
    }); pager('users', result);
  }
  async function mutate(control, callback) {
    if (state.mutation || state.authenticating || !state.allowed) return; state.mutation = true; const generation = state.generation;
    const container = control.closest('.ticket-detail') || control.closest('tr');
    const controls = [control, $('ops-logout'), ...(container?.querySelectorAll('button,input,select,textarea') || [])];
    const previous = new Map(controls.map(item => [item, item.disabled])); controls.forEach(item => { item.disabled = true; });
    const label = control.textContent; control.textContent = '处理中…';
    try { await callback(() => generation === state.generation && state.allowed); } catch (error) { if (generation === state.generation && !error.stale) notice(error.status === 409 ? `记录已更新，草稿已保留。工单可“读取最新进展”，素材可“刷新版本”，核对后再保存。${error.message}` : error.message); }
    finally { state.mutation = false; previous.forEach((disabled, item) => { item.disabled = disabled; }); control.textContent = label; }
  }
  async function archive(jobId, location, control) { await mutate(control, async current => { await request('/api/ops/assets', {method: 'POST', body: {job_id: jobId, location}}); notice('原始 STL 已归档，素材按内容去重。', false); await refreshSection('assets', true); if (current()) await refreshSection('overview', true); }); }
  async function archiveSelected() {
    if (!state.jobSelection.size || state.batchRunning || state.mutation || !state.allowed) return;
    const selected = [...state.jobSelection.entries()].map(([key, item]) => ({key, ...item})); const generation = state.generation;
    const host = $('jobs-batch-results'); host.replaceChildren(); host.hidden = false; $('jobs-batch-status').hidden = false;
    await mutate($('jobs-archive-selected'), async () => {
      state.batchRunning = true; updateJobSelection(); let successes = 0; let failures = 0;
      const controls = [...$('jobs-filter').querySelectorAll('input,select,button'), $('jobs-prev'), $('jobs-next')]; const previous = new Map(controls.map(control => [control, control.disabled])); controls.forEach(control => { control.disabled = true; });
      try {
        for (const [index, item] of selected.entries()) {
          if (generation !== state.generation || !state.allowed) return;
          $('jobs-batch-status').textContent = `正在归档 ${index + 1} / ${selected.length}：${item.title}`;
          try {
            await request('/api/ops/assets', {method: 'POST', body: {job_id: item.job_id, location: item.location}});
            if (generation !== state.generation || !state.allowed) return;
            successes += 1; state.jobSelection.delete(item.key); host.append(node('li', {class: 'success', text: `${item.title}（${item.job_id}）：归档成功`}));
          } catch (error) {
            if (generation !== state.generation || error.stale) return;
            failures += 1; host.append(node('li', {class: 'failure', text: `${item.title}（${item.job_id}）：${error.message}`}));
            if (error.status === 401 || error.status === 403) break;
          }
        }
        if (generation !== state.generation) return;
        $('jobs-batch-status').textContent = `归档结束：成功 ${successes} 个，失败 ${failures} 个${failures ? '。失败项保留选择，可核对后重试。' : '。素材按内容去重。'}`;
        await Promise.allSettled([refreshSection('assets', true), refreshSection('overview', true)]);
      } finally { state.batchRunning = false; previous.forEach((disabled, control) => { control.disabled = disabled; }); if (generation === state.generation) updateJobSelection(); }
    });
    if (generation === state.generation) updateJobSelection();
  }
  const renderers = {overview: renderOverview, jobs: renderJobs, tickets: renderTickets, events: renderEvents, assets: renderAssets, users: renderUsers};
  async function refreshSection(section, force = false) {
    if (!state.allowed) return;
    if (section === 'events') return refreshEvents(force);
    const host = section === 'overview' ? $('overview-cards') : $(`${section}-rows`);
    if (!force && (state.mutation || host?.contains(document.activeElement) || (section === 'assets' && state.assetDrafts.size))) return;
    const seq = state.sequences[section] = (state.sequences[section] || 0) + 1; const generation = state.generation;
    const params = new URLSearchParams({...state.filters[section], ...(state.pages[section] ? {page: state.pages[section], page_size: section === 'events' ? state.filters.events.page_size : 25} : {})});
    try {
      const result = await request(`/api/ops/${section}?${params}`);
      if (seq !== state.sequences[section] || generation !== state.generation || !state.allowed) return;
      if (!force && (state.mutation || host?.contains(document.activeElement) || (section === 'assets' && state.assetDrafts.size))) return;
      if (state.pages[section]) { const last = Math.max(1, Math.ceil(result.total / Number(params.get('page_size')))); if (state.pages[section] > last) { state.pages[section] = last; return refreshSection(section, force); } }
      state.data[section] = result; renderers[section](result);
    }
    catch (error) { if (seq !== state.sequences[section] || generation !== state.generation) return; connection('数据刷新失败'); notice(`${section === 'overview' ? '概览' : '列表'}：${error.message}`); throw error; }
  }
  async function refreshAll(manual = false) {
    if (state.busy || state.authenticating) { if (manual) state.refreshPending = true; return; }
    state.busy = true; $('ops-refresh-dot').className = 'ops-refresh-dot busy';
    try {
      // The workbench and support portal share this cookie: notice account changes in another tab.
      const generation = state.generation; const session = await request('/api/session');
      if (state.authenticating || generation !== state.generation) return;
      applySession(session);
      if (!state.allowed) return;
      const activeGeneration = state.generation; const results = await Promise.allSettled(Object.keys(renderers).map(section => refreshSection(section)));
      if (!state.allowed || activeGeneration !== state.generation) return;
      const ok = results.every(result => result.status === 'fulfilled');
      connection(ok ? '服务在线' : '部分数据刷新失败', ok); $('ops-last-refresh').textContent = `${ok ? '最近刷新' : '最近尝试'}：${fmt(new Date())}`;
    } catch (error) { if (!error.stale) { connection('会话检查失败'); notice(error.message); } }
    finally { state.busy = false; if (state.refreshPending && !state.authenticating && !state.mutation) { state.refreshPending = false; refreshAll(true); } }
  }
  const run = promise => promise.catch(() => {});
  const compact = values => Object.fromEntries(Object.entries(values).filter(([, value]) => value !== '' && value !== undefined && value !== null));
  function applyFilters(section) {
    let filters;
    if (section === 'jobs') { filters = {q: $('jobs-query').value.trim(), owner: $('jobs-user').value.trim(), status: $('jobs-status').value, location: $('jobs-location').value, expires_within_days: $('jobs-location').value === 'trash' ? $('jobs-expiring').value : ''}; clearJobSelection(); }
    if (section === 'tickets') filters = {q: $('tickets-query').value.trim(), owner: $('tickets-user').value.trim(), status: $('tickets-status').value, unassigned: $('tickets-unassigned').value};
    if (section === 'assets') filters = {q: $('assets-query').value.trim(), status: $('assets-status').value};
    if (section === 'users') filters = {q: $('users-query').value.trim(), disabled: $('users-disabled').value};
    if (section === 'events') {
      const since = $('events-since').value; const until = $('events-until').value;
      if ((since && Number.isNaN(new Date(since).getTime())) || (until && Number.isNaN(new Date(until).getTime()))) return notice('请输入有效的起始和截止时间。');
      if (since && until && new Date(since) > new Date(until)) return notice('起始时间不能晚于截止时间。');
      filters = {q: $('events-query').value.trim(), owner: $('events-user').value.trim(), action: $('events-action').value.trim(), category: $('events-category').value, severity: $('events-severity').value, since: since ? new Date(since).toISOString() : '', until: until ? new Date(until).toISOString() : '', page_size: Number($('events-limit').value)};
      state.feed.historyUntil = ''; state.feed.pending = false; state.feed.latestKey = null; feedStatus();
    }
    state.filters[section] = compact(filters); state.pages[section] = 1; run(refreshSection(section, true));
  }
  function bind() {
    $('login-mode').addEventListener('change', loginMode); $('ops-login-form').addEventListener('submit', async event => { event.preventDefault(); if (state.authenticating) return; state.authenticating = true; const submit = $('login-submit'); submit.disabled = true; $('login-error').hidden = true; try { const body = $('login-mode').value === 'token' ? {token: $('login-token').value} : {username: $('login-user').value.trim(), password: $('login-password').value}; applySession(await request('/api/session', {method: 'POST', body})); $('login-password').value = ''; $('login-token').value = ''; if (state.allowed) { notice(''); state.refreshPending = true; } } catch (error) { $('login-error').textContent = error.message; $('login-error').hidden = false; } finally { submit.disabled = false; state.authenticating = false; if (state.refreshPending) { state.refreshPending = false; refreshAll(true); } } });
    $('ops-logout').addEventListener('click', async () => { if (state.mutation || state.authenticating) return; state.authenticating = true; $('ops-logout').disabled = true; try { await request('/api/session/logout', {method: 'POST', body: {}}); applySession({...state.session, authenticated: false}); applySession(await request('/api/session')); } catch (error) { notice(error.message); } finally { state.authenticating = false; $('ops-logout').disabled = false; } });
    document.querySelectorAll('#ops-nav [data-view]').forEach(control => control.addEventListener('click', () => showView(control.dataset.view)));
    window.addEventListener('hashchange', () => { const view = location.hash.slice(1); if (['overview','events','jobs','tickets','assets','users'].includes(view)) showView(view); });
    $('ops-refresh').addEventListener('click', () => refreshAll(true)); document.querySelectorAll('[data-refresh]').forEach(control => control.addEventListener('click', () => run(control.dataset.refresh === 'events' ? refreshEvents(false) : refreshSection(control.dataset.refresh, true))));
    for (const section of ['jobs','tickets','assets','events','users']) {
      $(`${section}-filter`).addEventListener('submit', event => { event.preventDefault(); applyFilters(section); });
      for (const [direction, step] of [['prev', -1], ['next', 1]]) $(`${section}-${direction}`).addEventListener('click', () => {
        if (section === 'events' && step > 0 && !state.feed.historyUntil) state.feed.historyUntil = state.feed.snapshot || new Date().toISOString();
        state.pages[section] = Math.max(1, state.pages[section] + step); run(refreshSection(section, true));
      });
    }
    $('events-pause').addEventListener('click', () => { state.feed.paused = !state.feed.paused; feedStatus(); if (!state.feed.paused) run(refreshEvents()); });
    $('events-latest').addEventListener('click', latestEvents); $('events-new').addEventListener('click', latestEvents);
    $('events-reset').addEventListener('click', () => { $('events-filter').reset(); applyFilters('events'); }); $('events-export').addEventListener('click', exportEvents);
    $('jobs-filter').addEventListener('input', clearJobSelection);
    $('jobs-location').addEventListener('change', () => { $('jobs-expiring').disabled = $('jobs-location').value !== 'trash'; if ($('jobs-location').value !== 'trash') $('jobs-expiring').value = ''; });
    $('jobs-select-page').addEventListener('change', event => { const location = state.filters.jobs.location; for (const job of state.data.jobs?.items || []) { const key = `${location}:${job.id}`; if (event.target.checked) state.jobSelection.set(key, {job_id: job.id, location, title: job.case_id || job.id}); else state.jobSelection.delete(key); } updateJobSelection(); });
    $('jobs-clear-selection').addEventListener('click', clearJobSelection); $('jobs-archive-selected').addEventListener('click', archiveSelected);
    $('asset-reclaim').addEventListener('submit', event => { event.preventDefault(); const job = $('asset-job').value.trim(); if (!job) return notice('请输入任务 ID。'); run(archive(job, $('asset-location').value, event.submitter)); });
  }
  async function init() { bind(); try { applySession(await request('/api/session')); await refreshAll(); } catch (error) { notice(error.message); } window.setInterval(() => { if (!document.hidden) refreshAll(); }, 10000); document.addEventListener('visibilitychange', () => { if (!document.hidden) refreshAll(); }); }
  window.addEventListener('DOMContentLoaded', init);
})();
