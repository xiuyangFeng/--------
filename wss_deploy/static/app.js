/* WSS local workbench. Dynamic data is inserted through textContent, never HTML. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const WB = window.WssWorkbenchCore;
  // §19.1 shared helpers (report_common.js).  The workbench still works if that script failed to load.
  const RC = window.WssReportCommon || null;
  const localTime = (iso, seconds = false) => { if (!iso) return ''; try { return RC?.localTime ? RC.localTime(iso,{seconds}) : String(iso); } catch (_) { return String(iso); } };
  const friendlyTime = iso => { if (!iso) return ''; try { return RC?.friendlyTime ? RC.friendlyTime(iso) : localTime(iso); } catch (_) { return String(iso); } };
  const formatValue = (value, opts) => RC?.formatValue ? RC.formatValue(value, opts) : WB.formatNumber(value);
  const trimValue = value => WB.trimNumber(value);   // thresholds and ratios: 0.4 not 0.400
  const CN = {inlet:'主动脉入口', 'out-le':'左髂外', 'out-li':'左髂内', 'out-re':'右髂外', 'out-ri':'右髂内'};
  // v0.14: 左髂外 (dark teal) and 左髂内 (light green) now differ clearly in lightness; light fills get dark text.
  const COLORS = {inlet:0x2569aa, 'out-le':0x0e6b5c, 'out-li':0x8cc152, 'out-re':0xd08838, 'out-ri':0xa16cb3};
  const STATUS = {queued:'排队中', queued_A:'排队中', queued_B:'排队中', running:'计算中', running_A:'正在提取中心线', running_B:'正在预测血流场', awaiting_input:'待确认输入', awaiting_confirmation:'待确认出口', awaiting_outlets:'待确认出口', done:'已完成', failed:'失败', cancelled:'已取消', interrupted:'已中断', cancelling:'正在取消'};
  const PHASE = {ingest:'检查壁面与尺寸', input:'检查壁面与尺寸', centerline:'提取中心线', naming:'生成出口建议', geometry:'构建几何特征', smooth_resample:'平滑与重采样', features:'构建几何特征', inference:'预测血流场', inference_5_models:'预测壁面 WSS', report:'生成报告与导出文件', export:'生成报告与导出文件', metrics:'汇总预测统计', metrics_and_export:'汇总统计并导出', queued:'等待计算资源', cancelled:'任务已取消', interrupted:'服务中断，等待恢复'};
  const TIMER_NAMES = {ingest:'输入检查', centerline:'中心线提取', smooth_resample:'平滑与重采样', features:'几何特征', inference_5_models:'五模型预测', metrics:'指标统计', interpolation:'壁面插值', metrics_and_export:'统计与导出', export:'文件导出', report:'HTML 报告', total:'计算总耗时',
    // v0.14
    precompute:'后台几何预计算（确认出口期间）', morphology:'形态测量', metrics_and_interpolation:'统计与插值', volume_features:'体内采样与特征', inference_volume:'模型推理', inference:'模型推理', streamlines_and_interpolation:'流线与插值'};
  const FAMILY_TEXT = WB.FAMILY_TEXT;
  const state = {csrf:'',authenticated:false,session:null,current:null,job:null,jobs:[],releases:[],renderKey:null,mapping:{},viewer:null,geometry:null,busy:false,polling:false,listBusy:false,listTick:0,sequence:0,history:{q:'',status:'',patient_id:'',tag:'',page:1,page_size:20,total:0},selectMode:false,selected:new Map(),
    listMode:'jobs',viewAll:false,cases:[],caseReleases:[],prefs:{},prefsServer:null,glossary:null,unread:new Set(),ownerEvents:null,lastStatus:{},trash:[],claimable:[],modalResolve:null,refreshTimer:null,baseTitle:document.title,
    groupsOpen:Object.fromEntries(WB.WORK_GROUPS.map(group => [group.key, group.open])),
    cohort:null,cohortBusy:false,cohortFiltered:[],cohortSort:{key:'wss_p99_pa',direction:'desc'},
    // v0.12: overview list (latest 100 tasks), quick filter, /api/health, summary.json fallbacks, shortcuts
    overview:null,overviewBusy:false,quickFilter:'',health:null,healthMissing:false,extras:new Map(),shortcuts:null,templateEditable:null};
  const BATCH_LIMIT = 20;
  const DELETE_LIMIT = 100;
  const EXPORT_LIMIT = 200;
  const BUNDLE_LIMIT = 50;
  const locked = job => Boolean(job && job.review && job.review.status === 'reviewed');
  const REVIEW_LABEL = {unreviewed:'未审阅', reviewed:'已审阅', reopened:'已重新打开'};
  const isRunning = job => Boolean(job) && /^running/.test(job.status || '');
  const deletable = job => job && !isRunning(job) && !locked(job);
  const node = (tag, attrs = {}, ...children) => {
    const el = document.createElement(tag);
    for (const [key,value] of Object.entries(attrs)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === 'class') el.className = value;
      else if (key === 'text') el.textContent = String(value);
      else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
      else if (key in el && !key.startsWith('aria-') && !key.startsWith('data-')) el[key] = value;
      else el.setAttribute(key, String(value));
    }
    for (const child of children.flat(Infinity)) if (child !== null && child !== undefined && child !== false) el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    return el;
  };
  const fmt = (value, digits = 1) => Number.isFinite(Number(value)) && value !== null && value !== undefined ? Number(value).toFixed(digits) : '—';
  // v0.15: hours and days once past an hour (a case waiting overnight read 「已等待 720 分 5 秒」).
  const duration = value => {
    if (!Number.isFinite(Number(value)) || value === null || value === undefined) return '—';
    return WB.waitText(value);
  };
  // Compute time (§15.5): a retried job reports this attempt separately from the cumulative total.
  const timeText = (job, fallback) => {
    const timing = WB.computeTiming(job && job.timing);
    if (timing.split) return `本次 ${duration(timing.attempt)} · 累计 ${duration(timing.total)}`;
    return duration(fallback === undefined ? timing.total : fallback);
  };
  const badge = status => node('span', {class:`badge ${Object.hasOwn(STATUS,status) ? status : ''}`, text:STATUS[status] || status || '等待'});
  const reviewBadge = job => locked(job) ? node('span',{class:'badge reviewed',text:'已审阅'}) : job.review?.status === 'reopened' ? node('span',{class:'badge reopened',text:'已重开'}) : (job.status === 'done' ? node('span',{class:'badge pending',text:'待审阅'}) : null);
  const button = (text, onclick, cls = '') => node('button', {type:'button',class:cls,text,onclick});
  const gloss = key => node('button',{type:'button',class:'gloss','data-gloss':key,'aria-label':'术语说明',title:'术语说明',text:'?'});
  const fact = (label,value,note,glossKey) => node('div',{class:'fact'},node('span',{class:'fact-label'},label,glossKey && gloss(glossKey)),node('strong',{class:'fact-value',text:value}),note && node('small',{class:'fact-note',text:note}));
  const help = text => node('p',{class:'help',text});
  const callout = (text,kind = '') => node('div',{class:`callout ${kind}`,text});
  // §17.4 maximum-diameter chip: `null` (an older job without morphology) renders nothing at all.
  const maxDiameterMm = summary => { const value = Number(summary?.morphology?.aorta?.max?.max_diameter_mm); return Number.isFinite(value) ? value : null; };
  const diameterChip = value => value === null || value === undefined ? null : node('span',{class:'chip diameter-chip',title:'中心线站位截面的最大 Feret 直径'},'最大直径 ',node('strong',{text:`${fmt(value,1)} mm`}));
  const getA = job => job.a || job.stage_a || {};
  const jobUrl = (job, suffix = '') => `/api/jobs/${encodeURIComponent(job.id)}${suffix}`;
  // v0.13: every notice carries a × and optional action buttons (e.g. 「撤销」 after a delete).  Success notices
  // disappear after 8 s (15 s when they offer an action) unless the pointer rests on them; errors stay until closed.
  let noticeTimer = null;
  const stopNoticeTimer = () => { if (noticeTimer) { clearTimeout(noticeTimer); noticeTimer = null; } };
  const clearNotice = () => { stopNoticeTimer(); $('notice').hidden = true; };
  const showNotice = (message, kind, actions = [], ms = 0) => {
    const el = $('notice'); stopNoticeTimer();
    const close = node('button',{type:'button',class:'notice-close',text:'×',title:'关闭提示','aria-label':'关闭提示',onclick:clearNotice});
    el.className = `notice${kind ? ' ' + kind : ''}${actions.length ? ' has-actions' : ''}`;
    el.replaceChildren(node('span',{class:'notice-text',text:message}),...actions.map(a => button(a.label,() => { clearNotice(); a.run(); },a.cls || 'primary')),close);
    el.hidden = false; el.onmouseenter = stopNoticeTimer;
    if (ms > 0) { noticeTimer = setTimeout(clearNotice, ms); if (noticeTimer && typeof noticeTimer.unref === 'function') noticeTimer.unref(); }   // unref: node test harness only
  };
  const notify = (message,error = true,actions = []) => showNotice(message,error ? 'error' : '',actions,error ? 0 : (actions.length ? 15000 : 8000));
  // A success notice with one follow-up action (「处理下一例」 / 「下一例待审阅」); it stays until used or closed.
  const notifyAction = (message, label, run) => showNotice(message,'notice-action',[{label,run}],0);
  // §19.1: lists show friendly times, details show local wall-clock; the full time is always in the tooltip.
  const timeNode = (iso, {friendly = true, prefix = '', cls = 'job-time'} = {}) => node('time',{class:cls,datetime:iso || '',title:iso ? localTime(iso,true) : '',text:iso ? `${prefix}${friendly ? friendlyTime(iso) : localTime(iso)}` : ''});
  // Release kind of a job or run: WSS / WSS + TAWSS + OSI / 压力 + 速度体场 (§19.2 chips).
  const releaseOf = id => state.releases.find(item => (item.id || item.release) === id) || state.caseReleases.find(item => (item.id || item.release) === id) || null;
  const jobKind = job => {
    const release = job?.model_release || {}, id = release.id || release.release || release.registry_id || '';
    const withContract = release.contract ? release : (releaseOf(id) || release);
    if (job?.summary && (job.summary.cycle || job.summary.fields)) return WB.summaryKind(job.summary, WB.releaseKind(withContract, job.family));
    if (job?.has_cycle === true) return 'cycle';          // W2 list flag (§19.2)
    if (!id && !job?.family) return null;
    return WB.releaseKind(withContract, job?.family) || (job?.family === 'volume' ? 'volume' : 'wall');
  };
  const kindChip = kind => kind ? node('span',{class:`chip kind-chip kind-${kind}`,text:WB.KIND_TEXT[kind] || kind}) : null;
  const withTitle = (el, title) => { if (el && title) el.title = `发布包 ${title}`; return el; };
  const glossIf = key => state.glossary && state.glossary[key] ? gloss(key) : null;
  const openReport = job => window.open(jobUrl(job,'/report'),'_blank','noopener');
  const openOnepage = job => window.open(jobUrl(job,'/onepage'),'_blank','noopener');
  const errorText = error => typeof error === 'string' ? error : error?.message || '请求未完成，请稍后重试。';
  const store = {
    get(key, fallback = null) { try { const raw = localStorage.getItem(key); return raw === null ? fallback : raw; } catch (_) { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch (_) {} },
  };
  const SESSION_CALLS = new Set(['/api/session','/api/session/password']);
  async function request(url, {method = 'GET', body, timeout = 30000} = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    const headers = {'Accept':'application/json'};
    if (method !== 'GET' && state.csrf) headers['X-CSRF-Token'] = state.csrf;
    if (body && !(body instanceof FormData)) { headers['Content-Type'] = 'application/json'; body = JSON.stringify(body); }
    let response;
    try { response = await fetch(url, {method,headers,body,credentials:'same-origin',signal:controller.signal}); }
    catch (error) { throw new Error(error.name === 'AbortError' ? '请求超时。任务可能已保存，请刷新任务列表确认。' : '暂时无法连接服务。已保存的任务不会因此丢失，恢复连接后可继续。'); }
    finally { clearTimeout(timer); }
    let result;
    try { result = await response.json(); } catch (_) { throw new Error(`服务返回了无法读取的响应（HTTP ${response.status}），请稍后刷新。`); }
    if (!response.ok) {
      // v0.14: 429 / 413 limits keep the service's own sentence; the per-owner cap is spelled out with its numbers.
      const message = [413,429].includes(response.status) ? WB.limitErrorText(response.status,result,errorText(result.error || result.message)) : errorText(result.error || result.message);
      const error = new Error(message); error.status = response.status; error.body = result;
      // v0.15: the login POST and 改口令 answer 401 for a wrong password — that is not an expired session.
      if (response.status === 401 && !SESSION_CALLS.has(url)) expireSession();
      throw error;
    }
    return result;
  }
  // Binary downloads (zip / csv / xlsx) go through fetch so a JSON error body is shown instead of navigating away.
  async function downloadFile(url, {method = 'GET', body, filename = 'download'} = {}) {
    const headers = {};
    if (method !== 'GET' && state.csrf) headers['X-CSRF-Token'] = state.csrf;
    if (body !== undefined) { headers['Content-Type'] = 'application/json'; body = JSON.stringify(body); }
    let response;
    try { response = await fetch(url, {method,headers,body,credentials:'same-origin'}); }
    catch (_) { throw new Error('暂时无法连接服务，下载未开始。'); }
    if (!response.ok) {
      let message = `HTTP ${response.status}`, data = null;
      try { data = await response.json(); message = errorText(data.error || data.message); } catch (_) {}
      if ([413,429].includes(response.status)) message = WB.limitErrorText(response.status,data,data ? message : '');
      const error = new Error(message); error.status = response.status; error.body = data; throw error;
    }
    const blob = await response.blob();
    saveBlob(blob, WB.filenameFromDisposition(response.headers.get('Content-Disposition'), filename));
  }
  function saveBlob(blob, name) {
    const href = URL.createObjectURL(blob);
    const a = node('a',{href,download:name}); document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(href), 20000);
  }
  function setConnection(text, online = false, warn = false) { $('connection').textContent = text; $('connection').className = `connection${online ? ' online' : ''}${warn && !online ? ' warn' : ''}`; }
  // v0.15: one wording for "connected" (startSession used to say 「共享服务 · 已连接 · admin」, pollJob 「服务已连接」).
  const connectedText = () => state.session?.shared ? `共享服务 · 已连接${state.session.username ? ` · ${state.session.username}` : ''}` : '本机服务 · 已连接';
  const markOnline = () => {
    if (!state.authenticated || state.relogin) return;
    // While the live stream is down the page falls back to timed refreshes; keep saying so (amber) until it reopens.
    if (state.streamDown) { if (!$('connection').classList?.contains('warn')) setConnection('实时更新中断 · 正在重连…',false,true); return; }
    if (!$('connection').classList?.contains('online')) setConnection(connectedText(),true);
  };

  // ---------------------------------------------------------------------------------------------
  // Service health (§19.6): the status dot's tooltip.  A service without /api/health stays silent.
  // ---------------------------------------------------------------------------------------------
  async function refreshHealth() {
    if (!state.authenticated || state.healthMissing) return;
    try {
      const response = await fetch('/api/health',{credentials:'same-origin',headers:{Accept:'application/json'}});
      if (response.status === 404) { state.healthMissing = true; state.health = null; return; }
      if (response.status === 503) { setConnection('服务正在启动或维护 · 自动重试',false,true); return; }
      if (!response.ok) return;
      state.health = await response.json();
      checkServiceVersion(state.health?.version, state.health?.ui_build);
      if ($('health-pop') && !$('health-pop').hidden) renderHealthPop();
      if (!state.current) renderOverviewStatus();
    } catch (_) {}
  }
  // v0.15: the page remembers the service version it started with; after an upgrade the old page keeps working
  // against new code, so say it once and offer a reload (the reload keeps #job=… and therefore the open case).
  // ``build`` (health.ui_build, logged-in polls only) is the digest of the static files: a redeploy that changed only
  // app.js / app.css keeps the version but still deserves the reload prompt.
  function checkServiceVersion(version, build) {
    if (!version || typeof version !== 'string') return;
    const hasBuild = typeof build === 'string' && build.length > 0;
    const key = version + (hasBuild ? '/' + build : '');
    if (!state.bootVersion) { state.bootVersion = key; return; }
    const [bootVersion, bootBuild] = state.bootVersion.split('/');
    if (hasBuild && !bootBuild && version === bootVersion) { state.bootVersion = key; return; }   // first logged-in poll learns the build
    if (!hasBuild && version === bootVersion) return;                                             // logged-out polls carry no build
    if (key === state.bootVersion || state.versionNotified === key) return;
    state.versionNotified = key;
    const text = version === bootVersion ? '服务的页面文件已更新。请刷新页面以使用新版本，当前病例会保持打开。'
      : `服务已更新到 v${version}（本页面是 v${bootVersion} 时打开的）。请刷新页面以使用新版本，当前病例会保持打开。`;
    showNotice(text,'notice-action',[{label:'刷新页面',run:() => location.reload()}],0);
  }
  const uptimeText = seconds => { const s = Number(seconds); if (!Number.isFinite(s)) return '—'; const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60); return d ? `${d} 天 ${h} 小时` : h ? `${h} 小时 ${m} 分` : `${m} 分钟`; };
  function healthRows(health) {
    if (!health) return [];
    const queue = health.queue || {};
    const rows = [['版本',health.version ? `v${health.version}` : '—']];
    if (health.started_at || health.uptime_s !== undefined) rows.push(['已运行',`${uptimeText(health.uptime_s)}${health.started_at ? `（${localTime(health.started_at)} 启动）` : ''}`]);
    if (health.queue) rows.push(['队列',`计算中 ${queue.running ?? 0} · 排队 ${queue.queued ?? 0} · 待确认 ${(queue.awaiting_input ?? 0) + (queue.awaiting_confirmation ?? 0)}`]);
    if (health.worker_alive !== undefined) rows.push(['计算线程',health.worker_alive ? '正常' : '未运行']);
    if (health.gpu) rows.push(['GPU',health.gpu.available ? (health.gpu.name || '可用') : '不可用（CPU 计算）']);
    if (health.disk_free_gb !== undefined && health.disk_free_gb !== null) rows.push(['磁盘剩余',`${formatValue(health.disk_free_gb)} GB`]);
    if (health.default_release) rows.push(['默认发布包',releaseDisplay(health.default_release)]);
    if (health.stale_reports !== undefined && health.stale_reports !== null) rows.push(['待刷新报告',String(health.stale_reports)]);
    return rows;
  }
  function renderHealthPop() {
    const pop = $('health-pop'); if (!pop) return;
    const rows = healthRows(state.health);
    pop.replaceChildren(node('strong',{text:$('connection').textContent || '服务状态'}),
      rows.length ? node('dl',{class:'health-list'},...rows.flatMap(([k,v]) => [node('dt',{text:k}),node('dd',{text:v})])) : node('p',{class:'help',text:'当前服务版本未提供状态详情。'}));
  }
  function showHealthPop() {
    const pop = $('health-pop'); if (!pop || !state.authenticated || (state.healthMissing && !state.health)) return;
    renderHealthPop(); pop.hidden = false;
    const anchor = $('connection'), rect = typeof anchor.getBoundingClientRect === 'function' ? anchor.getBoundingClientRect() : null;
    if (rect) { pop.style.top = `${rect.bottom + window.scrollY + 8}px`; pop.style.left = `${Math.max(12, rect.right + window.scrollX - 300)}px`; }
    refreshHealth();
  }
  const hideHealthPop = () => { if ($('health-pop')) $('health-pop').hidden = true; };
  for (const [name,fn] of [['mouseenter',showHealthPop],['focus',showHealthPop],['mouseleave',hideHealthPop],['blur',hideHealthPop]]) $('connection').addEventListener(name,fn);

  // ---------------------------------------------------------------------------------------------
  // "⋯" action menus (list rows, detail header, settings): one floating element, rebuilt per open.
  // ---------------------------------------------------------------------------------------------
  let menuAnchor = null;
  function closeActionMenu() {
    const menu = $('action-menu'); if (!menu || menu.hidden) return;
    menu.hidden = true; menu.replaceChildren();
    if (menuAnchor) { menuAnchor.setAttribute('aria-expanded','false'); const anchor = menuAnchor; menuAnchor = null; anchor.focus?.({preventScroll:true}); }
  }
  function openActionMenu(anchor, items) {
    const menu = $('action-menu'); if (!menu) return;
    if (menuAnchor === anchor && !menu.hidden) { closeActionMenu(); return; }
    closeActionMenu();
    const usable = items.filter(Boolean);
    menu.replaceChildren(...usable.map(item => {
      if (item.separator) return node('div',{class:'menu-separator',role:'separator'});
      const entry = button(item.label,() => { closeActionMenu(); item.run(); },`menu-item${item.danger ? ' danger-item' : ''}`);
      entry.setAttribute('role','menuitem'); if (item.title) entry.title = item.title; entry.disabled = Boolean(item.disabled);
      return entry;
    }));
    menu.hidden = false; menuAnchor = anchor; anchor.setAttribute('aria-expanded','true');
    const rect = typeof anchor.getBoundingClientRect === 'function' ? anchor.getBoundingClientRect() : null;
    if (rect) {
      const width = 220, left = Math.max(8, Math.min(rect.right + window.scrollX - width, window.scrollX + window.innerWidth - width - 8));
      menu.style.left = `${left}px`; menu.style.top = `${rect.bottom + window.scrollY + 4}px`;
      const height = menu.offsetHeight || 0;
      if (height && rect.bottom + height + 12 > window.innerHeight && rect.top > height + 12) menu.style.top = `${rect.top + window.scrollY - height - 4}px`;
    }
    menu.querySelector('.menu-item:not(:disabled)')?.focus({preventScroll:true});
  }
  const moreButton = (label, items) => {
    const more = node('button',{type:'button',class:'more-button','aria-haspopup':'menu','aria-expanded':'false','aria-label':label,title:label,text:'⋯'});
    more.addEventListener('click',event => { event.stopPropagation?.(); openActionMenu(more, typeof items === 'function' ? items() : items); });
    return more;
  };
  document.addEventListener('click',event => { const menu = $('action-menu'); if (!menu || menu.hidden) return; if (event.target instanceof Element && (menu.contains(event.target) || event.target.closest?.('.more-button,.settings-button'))) return; closeActionMenu(); });
  document.addEventListener('keydown',event => {
    const menu = $('action-menu'); if (!menu || menu.hidden) return;
    if (event.key === 'Escape') { event.preventDefault(); closeActionMenu(); return; }
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      const entries = [...menu.querySelectorAll('.menu-item:not(:disabled)')]; if (!entries.length) return;
      const index = entries.indexOf(document.activeElement);
      entries[(index + (event.key === 'ArrowDown' ? 1 : entries.length - 1)) % entries.length].focus(); event.preventDefault();
    }
  });
  window.addEventListener('resize',closeActionMenu);

  // ---------------------------------------------------------------------------------------------
  // Modal dialog (duplicate upload choice, password change, batch export)
  // ---------------------------------------------------------------------------------------------
  function openModal(title, content, actions = [], {wide = false} = {}) {
    const dialog = $('modal'), body = $('modal-body');
    // v0.15: remember the control that opened the dialog; focus returns there on close (Esc, 取消, backdrop).
    if (!dialog.open) { const active = document.activeElement; state.modalReturn = active && active !== document.body && typeof active.focus === 'function' ? active : null; }
    dialog.classList.toggle('wide',Boolean(wide));
    body.replaceChildren(...[node('h2',{id:'modal-title',text:title}),...content,actions.length ? node('div',{class:'actions modal-actions'},...actions) : null].filter(Boolean));
    if (typeof dialog.showModal === 'function') { if (!dialog.open) dialog.showModal(); }
    else dialog.setAttribute('open','');
    return body;
  }
  function closeModal() {
    const dialog = $('modal');
    if (typeof dialog.close === 'function' && dialog.open) dialog.close(); else dialog.removeAttribute('open');
  }
  // v0.14 (F9): the styled replacement for window.confirm.  Resolves true only on the confirm button; Esc, a click
  // on the backdrop or 「取消」 resolve false (the modal's close handler below delivers {action:'cancel'}).
  function confirmDialog(title, message, {confirmLabel = '确定', cancelLabel = '取消', danger = false} = {}) {
    return new Promise(resolve => {
      if (state.modalResolve) { const previous = state.modalResolve; state.modalResolve = null; previous({action:'cancel'}); }
      state.modalResolve = result => resolve(Boolean(result && result.action === 'confirm'));
      const finish = ok => { state.modalResolve = null; closeModal(); resolve(ok); };
      const ok = button(confirmLabel,() => finish(true),danger ? 'danger' : 'primary');
      openModal(title,String(message || '').split('\n').filter(Boolean).map(line => help(line)),[button(cancelLabel,() => finish(false)),ok]);
      try { ok.focus(); } catch (_) {}
    });
  }
  $('modal').addEventListener('close', () => {
    const resolve = state.modalResolve; state.modalResolve = null; $('modal-body').replaceChildren(); if (resolve) resolve({action:'cancel'});
    const back = state.modalReturn; state.modalReturn = null;
    if (back && back.isConnected !== false && !$('modal').open) { try { back.focus({preventScroll:true}); } catch (_) {} }
  });
  $('modal').addEventListener('click', event => { if (event.target === $('modal') && !state.modalLocked) closeModal(); });
  $('modal').addEventListener('cancel', event => { if (state.modalLocked) event.preventDefault(); });

  // ---------------------------------------------------------------------------------------------
  // Session, login modes, claim (C3)
  // ---------------------------------------------------------------------------------------------
  const loginMode = session => session?.login || (session?.shared ? 'token' : 'none');
  // v0.15 (round 15): a 401 while working no longer throws the page away.  The login form opens over the page
  // (a modal <dialog>, above any open dialog), the case, the scroll position and anything typed stay; after the
  // same user logs in again the page carries on without a reload.  A different user gets a fresh workbench.
  function expireSession() {
    if (state.relogin) return;
    const wasWorking = state.authenticated && !$('workspace').hidden;
    state.authenticated = false; state.sequence++;
    closeEvents(); closeOwnerEvents(); closeActionMenu(); hideHealthPop();
    renderLoginPanel(state.session || {});
    const dialog = $('relogin-dialog');
    if (wasWorking && dialog && typeof dialog.showModal === 'function') {
      state.relogin = {jobId: state.current, username: state.session?.username || null};
      $('relogin-note').hidden = false; $('login-error').hidden = true;
      if (state.relogin.username && !$('login-user-group').hidden) $('login-username').value = state.relogin.username;   // the same person, usually
      $('relogin-body').append($('login-panel')); $('login-panel').hidden = false;
      if (!dialog.open) dialog.showModal();
      setConnection('登录已过期 · 请重新登录',false,true);
      focusLogin();
      return;
    }
    state.releases = []; state.current = null;
    closeModal(); closeUploadDialog(); closeCohortOverview();
    $('login-panel').hidden = false; $('workspace').hidden = true; $('user-menu').hidden = true; $('new-prediction').hidden = true; $('settings-button').hidden = true;
    $('detail').setAttribute('aria-busy','false'); syncUploadAvailability();
    setConnection('登录已过期 · 请重新登录',false,true);
    focusLogin();
  }
  // Put the login panel back in the page and close the overlay (after a successful re-login).
  function closeRelogin() {
    const dialog = $('relogin-dialog');
    $('relogin-note').hidden = true;
    const home = $('workspace'); if (home && home.parentNode && typeof home.parentNode.insertBefore === 'function') home.parentNode.insertBefore($('login-panel'),home);
    if (dialog?.open && typeof dialog.close === 'function') dialog.close();
  }
  // Same user again: keep everything on the page and only resynchronise (events, list, the open case).
  async function resumeSession(session) {
    const previous = state.relogin; state.relogin = null;
    closeRelogin();
    if (!session || !session.authenticated || (previous && (previous.username || null) !== (session.username || null))) {
      // Another person logged in on this page: nothing of the previous user's page may stay on screen.
      closeModal(); closeUploadDialog(); state.viewer?.dispose(); state.viewer = null;
      state.current = null; state.job = null; state.jobs = []; state.selected.clear(); syncRoute(null,{replace:true});
      $('jobs-list').replaceChildren(); $('cases-list').replaceChildren();
      await startSession(); return;
    }
    applySession(session);
    setConnection(connectedText(),true);
    openOwnerEvents(); if (state.current) openEvents(state.current);
    await refreshJobs(); if (!state.authenticated) return;
    if (state.current) { state.renderKey = state.job ? `${state.job.id}/${state.job.version}/${state.job.status}` : null; await pollJob(); }
    else if (overviewVisible()) refreshOverview();
    refreshTrash(); refreshHealth();
    notify('已重新登录，页面内容保留。如果刚才的操作没有完成，请再点一次。',false);
  }
  const focusLogin = () => {
    const password = !$('login-user-group').hidden;
    const target = !password ? $('access-token') : ($('login-username').value.trim() ? $('login-password') : $('login-username'));
    try { target.focus(); } catch (_) {}
  };
  function renderLoginPanel(session) {
    const mode = loginMode(session);
    const password = mode === 'password', legacy = Boolean(session?.legacy_token_allowed);
    $('login-user-group').hidden = !password;
    $('login-token-group').hidden = password && !legacy;
    $('login-username').required = password; $('login-password').required = password && !legacy;
    $('access-token').required = !password;
    $('login-help').textContent = password
      ? (legacy ? '使用用户名和口令登录；迁移期也可留空用户名、只填旧访问令牌。同一用户在任何浏览器都能看到同一份任务列表。' : '使用用户名和口令登录；同一用户在任何浏览器都能看到同一份任务列表。')
      : '输入此服务的访问令牌。任务和报告仅对当前会话可见。';
    // v0.15: user-name mode says 登录; the remembered user name (never the password) is filled in.
    $('login-title').textContent = password ? '登录工作台' : '连接工作台';
    $('login-submit').textContent = password ? '登录' : '连接';
    if (password && !$('login-username').value) { const remembered = store.get('wss-login-user',''); if (remembered) $('login-username').value = remembered; }
    $('remember-username').checked = store.get('wss-login-remember','1') !== '0';
  }
  // Login form helpers: show / hide the secret, Caps Lock hint, inline errors, the 60 s wait after a 429.
  const showLoginError = text => { const el = $('login-error'); el.textContent = text || ''; el.hidden = !text; };
  for (const [buttonId, inputId, noun] of [['reveal-password','login-password','口令'],['reveal-token','access-token','令牌']]) {
    $(buttonId).addEventListener('click', () => {
      const input = $(inputId), show = input.type === 'password';
      input.type = show ? 'text' : 'password';
      $(buttonId).setAttribute('aria-pressed',String(show)); $(buttonId).textContent = show ? '隐藏' : '显示'; $(buttonId).setAttribute('aria-label',`${show ? '隐藏' : '显示'}${noun}`);
      try { input.focus(); } catch (_) {}
    });
  }
  const capsCheck = event => { const on = Boolean(event?.getModifierState?.('CapsLock')); $('caps-hint').hidden = !on; };
  for (const id of ['login-password','access-token']) { $(id).addEventListener('keydown',capsCheck); $(id).addEventListener('keyup',capsCheck); $(id).addEventListener('blur',() => { $('caps-hint').hidden = true; }); }
  const loginWaiting = () => Boolean(state.loginWaitUntil) && Date.now() < state.loginWaitUntil;
  for (const id of ['login-username','login-password','access-token']) $(id).addEventListener('input',() => { if (!loginWaiting()) showLoginError(''); });
  function loginCooldown(seconds, message) {
    const submit = $('login-submit'); clearInterval(state.loginTimer);
    state.loginWaitUntil = Date.now() + Math.max(1, Math.round(seconds)) * 1000;
    const label = $('login-user-group').hidden ? '连接' : '登录';
    const paint = () => {
      const left = Math.ceil((state.loginWaitUntil - Date.now()) / 1000);
      if (left > 0) { submit.disabled = true; submit.textContent = `${left} 秒后可再试`; showLoginError(`${message || '尝试过于频繁。'}（${left} 秒后可再试）`); return; }
      clearInterval(state.loginTimer); state.loginWaitUntil = 0; submit.disabled = false; submit.textContent = label; showLoginError('');
    };
    paint();
    state.loginTimer = setInterval(paint,1000);
    if (state.loginTimer && typeof state.loginTimer.unref === 'function') state.loginTimer.unref();   // node test harness only
  }
  function applySession(session, {claimable} = {}) {
    state.session = session; state.csrf = session.csrf_token || ''; state.authenticated = Boolean(session.authenticated);
    renderLoginPanel(session);
    $('login-panel').hidden = state.authenticated; $('workspace').hidden = !state.authenticated;
    $('new-prediction').hidden = !state.authenticated; $('settings-button').hidden = !state.authenticated;
    if (!state.authenticated) { state.releases = []; closeUploadDialog(); hideHealthPop(); closeActionMenu(); }
    syncUploadAvailability();
    const menu = $('user-menu');
    if (state.authenticated && session.username) {
      const name = $('user-name'); name.replaceChildren(node('span',{text:session.display_name || session.username}));
      if (session.role === 'admin') name.append(node('span',{class:'role',text:'管理员'}));
      menu.hidden = false;
    } else menu.hidden = true;
    const admin = state.authenticated && session.role === 'admin';
    $('view-all-wrap').hidden = !admin; if (!admin) { state.viewAll = false; $('view-all').checked = false; }
    // Old-session tasks the user may claim: from the login response (with counts) or the session row (owner list).
    let owners = Array.isArray(claimable) ? claimable : null;
    if (!owners) owners = (session.claimable_owners || []).map(owner => typeof owner === 'string' ? {owner,jobs:null} : owner);
    const dismissed = new Set(dismissedOwners());
    state.claimable = owners.filter(item => item && item.owner && !dismissed.has(item.owner) && item.jobs !== 0);
    const banner = $('claim-banner');
    if (state.authenticated && session.username && state.claimable.length) {
      const total = state.claimable.reduce((sum,item) => sum + (Number(item.jobs) || 0),0);
      $('claim-text').textContent = total ? `检测到之前会话创建的 ${total} 个任务，尚未归属到用户 ${session.username}。` : '检测到之前会话创建的任务，尚未归属到当前用户。';
      $('claim-button').textContent = total ? `认领本会话任务（${total} 个）` : '认领本会话任务';
      banner.hidden = false;
    } else banner.hidden = true;
  }
  async function startSession() {
    try {
      const session = await request('/api/session');
      applySession(session, {claimable:state.pendingClaimable}); state.pendingClaimable = null;
      if (!state.authenticated) focusLogin();
      if (state.authenticated) {
        setConnection(connectedText(),true);
        await loadPreferences(); if (!state.authenticated) return;
        await refreshReleases(); if (!state.authenticated) return;
        // Do not expose a submit path until the server has supplied a usable,
        // immutable model release. An empty/failed release list must remain
        // an explicit blocked state rather than producing a late 400 response.
        applyUploadPreferences(); syncUploadAvailability(); loadGlossary(); await refreshJobs();
        if (!state.authenticated) return;
        openOwnerEvents(); updateUnreadBadge(); refreshTrash(); refreshHealth();
        // §19.9 deep link: #job=<id> reopens that case after a reload; otherwise the overview.
        const routed = WB.parseJobHash(location.hash);
        if (routed) await selectJob(routed,{fromRoute:true});
        else if (!state.current) await showOverview();
      } else setConnection(loginMode(session) === 'password' ? '未登录 · 请先登录' : '等待连接');
    } catch (error) { if (error.status === 401) return; notify(error.message); setConnection('连接失败 · 请刷新重试'); }
  }
  $('login-form').addEventListener('submit', async event => {
    event.preventDefault(); if (loginWaiting()) return;
    const submit = $('login-submit'); submit.disabled = true; showLoginError('');
    const username = $('login-username').value.trim(), passwordMode = !$('login-user-group').hidden;
    if (passwordMode && !username && $('login-token-group').hidden) { showLoginError('请输入用户名。'); submit.disabled = false; $('login-username').focus(); return; }
    const body = passwordMode && username ? {username,password:$('login-password').value} : {token:$('access-token').value};
    let wait = false;
    try {
      const result = await request('/api/session',{method:'POST',body});
      $('access-token').value = ''; $('login-password').value = '';
      if (passwordMode && username) {
        store.set('wss-login-remember',$('remember-username').checked ? '1' : '0');
        if ($('remember-username').checked) store.set('wss-login-user',username); else { try { localStorage.removeItem('wss-login-user'); } catch (_) {} }
      }
      state.pendingClaimable = Array.isArray(result.claimable) ? result.claimable : null;
      if (state.relogin) await resumeSession(result); else { clearNotice(); await startSession(); }
    }
    catch (error) {
      if (error.status === 429) { const after = +(error.body && error.body.retry_after); wait = true; loginCooldown(Number.isFinite(after) && after > 0 ? Math.ceil(after) : 60,error.message || '尝试过于频繁。'); }
      else {
        showLoginError(error.status === 401 ? (error.message || '用户名或口令不正确。') + (passwordMode && username ? ' 口令区分大小写。' : '') : error.message);
        const field = passwordMode && username ? $('login-password') : !$('login-token-group').hidden ? $('access-token') : $('login-username');
        try { field.focus(); field.select?.(); } catch (_) {}
      }
    } finally { if (!wait) submit.disabled = false; }
  });
  $('relogin-dialog')?.addEventListener('cancel', event => event.preventDefault());   // the page is unusable until logged in
  $('logout').addEventListener('click', async () => {
    try { await request('/api/session/logout',{method:'POST',body:{}}); } catch (error) { if (error.status === 404) { notify('当前服务版本不支持退出登录；清除浏览器 Cookie 可结束会话。'); return; } if (error.status !== 401) notify(error.message); }
    closeEvents(); closeOwnerEvents(); state.authenticated = false; state.current = null; state.job = null; state.jobs = []; state.selected.clear();
    $('jobs-list').replaceChildren(); $('cases-list').replaceChildren(); $('user-menu').hidden = true; $('claim-banner').hidden = true;
    await startSession();
  });
  $('change-password').addEventListener('click', () => {
    const oldField = node('input',{type:'password',autocomplete:'current-password','aria-label':'当前口令',placeholder:'当前口令'});
    const newField = node('input',{type:'password',autocomplete:'new-password','aria-label':'新口令',placeholder:'新口令（至少 8 个字符）',minlength:8});
    const again = node('input',{type:'password',autocomplete:'new-password','aria-label':'重复新口令',placeholder:'再输入一次新口令'});
    const status = node('p',{class:'help'});
    const submit = button('保存新口令',async () => {
      if (newField.value.length < 8) { status.textContent = '新口令至少 8 个字符。'; newField.focus(); return; }
      if (newField.value !== again.value) { status.textContent = '两次输入的新口令不一致。'; again.focus(); return; }
      submit.disabled = true;
      try { await request('/api/session/password',{method:'POST',body:{old_password:oldField.value,new_password:newField.value}}); closeModal(); notify('口令已更新。',false); }
      catch (error) { status.textContent = error.status === 401 ? '当前口令不正确（区分大小写），请重新输入。' : error.message; submit.disabled = false; if (error.status === 401) { oldField.focus(); oldField.select?.(); } }
    },'primary');
    openModal('修改口令',[node('div',{class:'review-form'},oldField,newField,again),status],[button('取消',closeModal),submit]);
    oldField.focus();
  });
  async function claimJobs() {
    const owners = state.claimable.slice(); if (!owners.length) return;
    $('claim-button').disabled = true;
    try {
      let claimed = 0;
      for (const item of owners) { const result = await request('/api/jobs/claim',{method:'POST',body:{owner:item.owner}}); claimed += Number(result.claimed || 0); }
      notify(claimed ? `已认领 ${claimed} 个任务，它们现在归属于用户 ${state.session?.username || ''}。` : '没有需要认领的任务。',false);
      dismissClaim(); await refreshJobs(true);
    } catch (error) { notify(error.message); $('claim-button').disabled = false; }
  }
  const dismissedOwners = () => { try { const list = JSON.parse(store.get('wss-claim-dismissed','[]') || '[]'); return Array.isArray(list) ? list : []; } catch (_) { return []; } };
  function dismissClaim() {
    const dismissed = new Set(dismissedOwners());
    for (const item of state.claimable) dismissed.add(item.owner);
    store.set('wss-claim-dismissed',JSON.stringify([...dismissed].slice(-20)));
    state.claimable = []; $('claim-banner').hidden = true; $('claim-button').disabled = false;
  }
  $('claim-button').addEventListener('click',claimJobs);
  $('claim-dismiss').addEventListener('click',dismissClaim);
  $('view-all').addEventListener('change', async () => { state.viewAll = $('view-all').checked; state.history.page = 1; state.cohort = null; await refreshJobs(true); refreshTrash(); if ($('cohort-panel').open) loadCohort(true); });

  // ---------------------------------------------------------------------------------------------
  // Preferences (C6): server first, localStorage fallback; sections merge, never overwrite each other.
  // ---------------------------------------------------------------------------------------------
  async function loadPreferences() {
    let prefs = null;
    try { const result = await request('/api/preferences'); prefs = result.preferences || {}; state.prefsServer = true; }
    catch (_) { state.prefsServer = false; try { prefs = JSON.parse(store.get('wss-preferences','null') || 'null'); } catch (_) { prefs = null; } }
    state.prefs = WB.mergePreferences({}, prefs || {});
    $('notify-toggle').checked = state.prefs.notifications?.enabled !== false;
  }
  async function savePreferences(patch) {
    state.prefs = WB.mergePreferences(state.prefs, patch);
    store.set('wss-preferences', JSON.stringify(state.prefs));
    if (state.prefsServer === false) return;
    try { const result = await request('/api/preferences',{method:'PUT',body:state.prefs}); if (result.preferences) state.prefs = WB.mergePreferences({}, result.preferences); }
    catch (error) { if (error.status === 404 || error.status === 405) state.prefsServer = false; }
  }
  const setSelect = (id, value) => { const el = $(id); if (!el || value === undefined || value === null || value === '') return false; if ([...el.options].some(option => option.value === String(value))) { el.value = String(value); return true; } return false; };
  function applyUploadPreferences() {
    const up = state.prefs.upload || {};
    setSelect('upload-units', up.units);
    if (setSelect('upload-release', up.release_id)) $('upload-release').onchange?.();
    setSelect('upload-device', up.device); setSelect('upload-seeds', up.seed_count);
    if (up.threads !== undefined && up.threads !== null) $('upload-threads').value = String(up.threads);
    $('remember-patient').checked = Boolean(up.remember_patient);
    if (up.remember_patient && up.last_patient) {
      const last = up.last_patient;
      $('patient-id').value = last.patient_id || ''; $('scan-label').value = last.scan_label || ''; $('scan-date').value = last.scan_date || ''; $('case-tags').value = last.tags || '';
    }
  }
  $('remember-patient').addEventListener('change', () => { savePreferences({upload:{remember_patient:$('remember-patient').checked}}); });
  $('notify-toggle').addEventListener('change', () => {
    const enabled = $('notify-toggle').checked; savePreferences({notifications:{enabled}});
    if (enabled && typeof Notification !== 'undefined' && Notification.permission === 'default') Notification.requestPermission().catch(() => {});
    if (enabled && typeof Notification !== 'undefined' && Notification.permission === 'denied') notify('浏览器已禁止本站通知，请在地址栏的站点设置中允许通知。');
  });

  // The upload form lives in a dialog so the case list remains the first
  // thing visible on the workbench. IDs and form handlers stay unchanged.
  const uploadDialog = $('upload-dialog');
  const closeUploadDialog = () => { if (uploadDialog?.open && typeof uploadDialog.close === 'function') uploadDialog.close(); else uploadDialog?.removeAttribute('open'); };
  function openUploadDialog() {
    if (!state.authenticated) return false;
    closeActionMenu(); hideHealthPop();
    if (!uploadDialog.open) { if (typeof uploadDialog?.showModal === 'function') uploadDialog.showModal(); else uploadDialog?.setAttribute('open',''); }
    $('stl-file')?.focus();
    return true;
  }
  $('new-prediction')?.addEventListener('click', openUploadDialog);
  // Drag STL files anywhere on the page (A7): open the upload dialog and hand the files to the same
  // input + change handler a manual file choice goes through.
  const dragHasFiles = event => { const types = event.dataTransfer?.types; return Boolean(types) && [...types].includes('Files'); };
  let dragDepth = 0;
  const showDrop = on => { const overlay = $('drop-overlay'); if (overlay) overlay.hidden = !on; };
  document.addEventListener('dragenter', event => { if (!state.authenticated || !dragHasFiles(event)) return; event.preventDefault(); dragDepth++; showDrop(true); });
  document.addEventListener('dragover', event => { if (!state.authenticated || !dragHasFiles(event)) return; event.preventDefault(); if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy'; });
  document.addEventListener('dragleave', event => { if (!dragHasFiles(event)) return; dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth || !event.relatedTarget) { dragDepth = 0; showDrop(false); } });
  document.addEventListener('drop', event => {
    if (!dragHasFiles(event)) return;
    event.preventDefault(); dragDepth = 0; showDrop(false);
    if (!state.authenticated) return;
    if (state.busy) { notify('正在上传或处理其他操作，请稍后再拖入文件。'); return; }
    const files = [...(event.dataTransfer?.files || [])];
    const stl = files.filter(file => /\.stl$/i.test(file.name));
    if (!stl.length) { notify(files.length ? '只接受 .stl 文件，请拖入血管壁面 STL。' : '没有读取到拖入的文件。'); return; }
    openUploadDialog();
    try {
      const transfer = new DataTransfer(); for (const file of stl) transfer.items.add(file);
      $('stl-file').files = transfer.files;
    } catch (_) { notify('当前浏览器不支持拖放到上传框，请点击「选择血管壁面 STL」手动选择。'); return; }
    $('stl-file').dispatchEvent(new Event('change',{bubbles:true}));
    if (stl.length < files.length) notify(`已忽略 ${files.length - stl.length} 个非 STL 文件。`,false);
  });
  $('close-upload')?.addEventListener('click', closeUploadDialog);
  uploadDialog?.addEventListener('click', event => { if (event.target === uploadDialog) closeUploadDialog(); });
  uploadDialog?.addEventListener('cancel', closeUploadDialog);
  uploadDialog?.addEventListener('close', () => { if (state.authenticated && !$('workspace').hidden && !$('modal').open) $('new-prediction').focus({preventScroll:true}); });
  function uploadReady() {
    const ids = splitReleaseChoice($('upload-release').value);
    return state.authenticated && ids.length > 0 && ids.every(id => state.releases.some(release => (release.id || release.release) === id));
  }
  function syncUploadAvailability() { $('upload-button').disabled = state.busy || !uploadReady(); }

  // ---------------------------------------------------------------------------------------------
  // Glossary popovers (C17)
  // ---------------------------------------------------------------------------------------------
  async function loadGlossary() {
    if (state.glossary) return state.glossary;
    try { const response = await fetch('/static/glossary.json',{credentials:'same-origin'}); if (!response.ok) throw new Error(); const doc = await response.json(); state.glossary = doc.terms || {}; }
    catch (_) { state.glossary = {}; }
    return state.glossary;
  }
  document.addEventListener('click', async event => {
    const pop = $('gloss-popover');
    const target = event.target instanceof Element ? event.target.closest('[data-gloss]') : null;
    if (!target) { if (!pop.hidden && !pop.contains(event.target)) pop.hidden = true; return; }
    event.preventDefault();
    const key = target.getAttribute('data-gloss'), terms = await loadGlossary(), term = terms[key];
    pop.replaceChildren(...[node('strong',{text:term?.zh || key}),node('p',{text:term?.zh_desc || '术语表暂不可用。'}),term?.en ? node('small',{text:`${term.en} · ${term.en_desc || ''}`}) : null].filter(Boolean));
    const rect = target.getBoundingClientRect();
    pop.hidden = false;
    const width = Math.min(320, window.innerWidth - 24);
    pop.style.left = `${Math.max(12, Math.min(rect.left + window.scrollX, window.scrollX + window.innerWidth - width - 12))}px`;
    pop.style.top = `${rect.bottom + window.scrollY + 6}px`;
  });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') $('gloss-popover').hidden = true; });

  // ---------------------------------------------------------------------------------------------
  // Releases
  // ---------------------------------------------------------------------------------------------
  async function refreshReleases() {
    const select = $('upload-release'); if (!select) return;
    state.releases = []; select.disabled = true; select.onchange = null;
    select.replaceChildren(node('option',{value:'',text:'正在读取可用发布包…'})); syncUploadAvailability();
    try {
      const result = await request('/api/releases');
      if (!state.authenticated) return;
      const releases = Array.isArray(result) ? result : result.releases;
      state.releases = (Array.isArray(releases) ? releases : []).filter(release => release && typeof (release.id || release.release) === 'string' && (release.id || release.release).trim());
      select.replaceChildren();
      if (!state.releases.length) { select.append(node('option',{value:'',text:'没有可用发布包'})); select.disabled = true; $('upload-button').disabled = true; return; }
      for (const release of state.releases) {
        const label = releaseDisplay(release);
        select.append(node('option',{value:release.id || release.release,text:label,title:release.id || release.release}));
      }
      // v0.15.7: one upload → the three-head wall package and the volume package (two tasks, outlets confirmed once).
      for (const combo of releaseCombos(state.releases)) select.append(node('option',{value:combo.value,text:combo.label,title:combo.ids.join(' + ')}));
      const selected = state.releases.find(item => item.default) || state.releases[0];
      if (selected) select.value = selected.id || selected.release;
      select.disabled = state.releases.length <= 1;
      function updateReleaseSettings() {
        const ids = splitReleaseChoice(select.value), combo = ids.length > 1;
        const selected = state.releases.find(item => (item.id || item.release) === ids[0]);
        const fields = selected?.contract?.fields || {};
        const volume = Boolean(fields.velocity || selected?.contract?.protocol === 'single_frame_volume');
        const cycleWall = Boolean(fields.tawss || fields.osi || selected?.contract?.protocol === 'single_frame_wss_cycle_multi');
        const seedSelect = $('upload-seeds');
        const count = Number(selected?.models_count || selected?.model_count || 5);
        if (seedSelect && combo) seedSelect.replaceChildren(node('option',{value:'all',text:'全部模型（各发布包）'}));
        else if (seedSelect) {
          seedSelect.replaceChildren(node('option',{value:'all',text:`全部模型（${volume ? '每字段 ' : ''}${count}）`}));
          for (const n of [1,3,5]) if (n < count) seedSelect.append(node('option',{value:String(n),text:`${volume ? '每字段 ' : ''}${n} 个模型`}));
        }
        const helpText = $('release-help');
        if (helpText) helpText.textContent = combo ? '同时创建两个任务：峰值 WSS + TAWSS + OSI，以及压力 + 速度体场。出口只需确认一次，确认后体场任务自动排队，两个结果显示在同一病例下。' : volume ? '查看体内速度、流线与截面，压力可切换到壁面。固定收缩期单帧。' : cycleWall ? '查看峰值 WSS、周期平均 TAWSS 与 OSI，可在报告中切换字段。任务会固定绑定所选发布包。' : '查看壁面 WSS 热点与分支统计。任务会固定绑定所选发布包。';
        if ($('release-version')) $('release-version').textContent = `发布包版本：${combo ? ids.join(' + ') : selected?.id || selected?.release || '未指定'}`;
        syncUploadAvailability();
      }
      select.onchange = updateReleaseSettings;
      updateReleaseSettings();
    } catch (error) {
      state.releases = []; select.disabled = true; select.replaceChildren(node('option',{value:'',text:'发布包暂时不可用'})); syncUploadAvailability();
      const helpText = $('release-help'); if (helpText) helpText.textContent = `发布包列表读取失败：${error.message}`;
    }
  }
  const releaseFamily = releaseId => {
    const fromCases = WB.familyOfRelease(releaseId, state.caseReleases); if (fromCases) return fromCases;
    const release = state.releases.find(item => (item.id || item.release) === releaseId);
    if (!release) return null;
    const protocol = release.contract?.protocol;
    const fields = release.contract?.fields || {};
    if (protocol === 'single_frame_volume' || fields.velocity) return 'volume';
    if (protocol === 'single_frame_wss' || protocol === 'single_frame_wss_cycle_multi' || fields.wss || fields.tawss || fields.osi) return 'wall';
    return null;
  };
  // v0.15.7: an upload choice is one release id or several joined by '+' (primary first, then companions).
  const splitReleaseChoice = value => String(value || '').split('+').map(part => part.trim()).filter(Boolean);
  const releaseKindOfRecord = release => {
    const fields = release?.contract?.fields || {}, protocol = release?.contract?.protocol;
    if (protocol === 'single_frame_volume' || fields.velocity) return 'volume';
    if (fields.tawss || fields.osi || protocol === 'single_frame_wss_cycle_multi') return 'cycle';
    return 'wall';
  };
  // Short family name of a release for provenance lines ('压力 + 速度体场' …); unknown ids fall back to the display name.
  const releaseFamilyName = id => {
    const release = state.releases.find(item => (item.id || item.release) === id) || state.caseReleases.find(item => (item.id || item.release) === id);
    if (!release || !release.contract) return releaseDisplay(id);   // case-list records carry no contract
    return {volume:'压力 + 速度体场', cycle:'WSS + TAWSS + OSI', wall:'壁面 WSS'}[releaseKindOfRecord(release)];
  };
  // Combined choices offered at upload: every three-head wall package with every volume package (plain WSS is not
  // combined for now — it is expected to merge into the three-head model).
  const releaseCombos = releases => {
    const idOf = release => release.id || release.release, list = Array.isArray(releases) ? releases : [];
    const cycle = list.filter(release => releaseKindOfRecord(release) === 'cycle'), volume = list.filter(release => releaseKindOfRecord(release) === 'volume');
    return cycle.flatMap(c => volume.map(v => ({ids:[idOf(c), idOf(v)], value:`${idOf(c)}+${idOf(v)}`,
      label:`WSS + TAWSS + OSI ＋ 压力 + 速度体场（同时预测，两个任务）${cycle.length > 1 || volume.length > 1 ? ` · ${releaseDisplay(c)} / ${releaseDisplay(v)}` : ''}`})));
  };
  // Keep internal release IDs available as option values, but show a short,
  // task-oriented name wherever a person chooses or reviews a model package.
  const releaseDisplay = (releaseOrId, {includeId = false} = {}) => {
    const release = typeof releaseOrId === 'object' && releaseOrId ? releaseOrId : state.releases.find(item => (item.id || item.release) === releaseOrId) || state.caseReleases.find(item => (item.id || item.release) === releaseOrId);
    const id = String(release?.id || release?.release || releaseOrId || '').trim();
    if (!id) return '未指定发布包';
    const explicit = release?.label || release?.display_name || release?.title || release?.name;
    const protocol = release?.contract?.protocol;
    const fields = release?.contract?.fields || {};
    const cycleWall = Boolean(fields.tawss || fields.osi || protocol === 'single_frame_wss_cycle_multi');
    const family = protocol === 'single_frame_volume' || fields.velocity ? '压力 + 速度体场' : cycleWall ? 'WSS + TAWSS + OSI' : protocol === 'single_frame_wss' || fields.wss ? '壁面 WSS' : FAMILY_TEXT[releaseFamily(id)] || '';
    const count = Number(release?.models_count || release?.model_count || 0);
    const date = id.match(/20\d{6}/)?.[0];
    let label = explicit && explicit !== id ? explicit : family || id.replace(/_/g,' ');
    if (explicit === id || !explicit) {
      if (count) label += ` · ${count} 个模型`;
      if (date && !label.includes(date)) label += ` · ${date.slice(0,4)}-${date.slice(4,6)}-${date.slice(6)}`;
    }
    if (release?.default) label += ' · 默认';
    return includeId && !label.includes(id) ? `${label}（${id}）` : label;
  };

  // ---------------------------------------------------------------------------------------------
  // Upload with duplicate detection (C2)
  // ---------------------------------------------------------------------------------------------
  // v0.15: every chosen file is checked in the browser first (extension, empty, size limit, duplicate name) and
  // shows its size and the reason it will not be sent; only the valid ones are uploaded.  The service checks again.
  const uploadCheck = () => WB.checkUploadFiles([...($('stl-file').files || [])],{maxBytes:state.session?.max_upload_bytes});
  function renderUploadFiles() {
    const files = [...($('stl-file').files || [])], check = uploadCheck();
    $('file-name').textContent = files.length ? `${files.length} 个文件：${files.slice(0,2).map(file => file.name).join('、')}${files.length > 2 ? '…' : ''}` : '支持单个或多个 STL；上传后自动检查开口';
    const list = $('upload-files'); if (!list) return check;
    list.replaceChildren(); list.hidden = files.length === 0;
    for (const row of check.rows) list.append(node('div',{class:`upload-file-row${row.ok ? (row.warn ? ' warn' : '') : ' invalid'}`,title:row.name},
      node('span',{text:row.name}),node('span',{class:'upload-file-size',text:WB.formatBytes(row.size)}),node('span',{class:'upload-file-state',text:row.reason || row.warn || '待上传'})));
    if (check.invalid) list.append(node('p',{class:'upload-summary',role:'alert',text:check.valid.length ? `${check.invalid} 个文件不会上传（原因见上），将只上传其余 ${check.valid.length} 个。` : '没有可以上传的 STL 文件，请重新选择。'}));
    return check;
  }
  $('stl-file').addEventListener('change', renderUploadFiles);
  // Anonymous identifiers: control characters / length are errors (the service would refuse them), a bare
  // Chinese name is a warning.  Shown under the field while typing; errors block 「上传并检查」.
  const ID_FIELDS = [['case-id','case-id-issue','病例编号',160],['patient-id','patient-id-issue','患者编号',80]];
  function identifierIssues() {
    let blocking = null;
    for (const [inputId, issueId, label, max] of ID_FIELDS) {
      const input = $(inputId), out = $(issueId); if (!input || !out) continue;
      const issue = WB.identifierIssue(input.value,{label,max});
      out.hidden = !issue; out.textContent = issue ? issue.text : ''; out.className = `field-issue${issue?.level === 'warn' ? ' warn' : ''}`;
      input.classList?.toggle('has-issue',Boolean(issue && issue.level === 'error'));
      if (issue) input.setAttribute('aria-invalid',String(issue.level === 'error')); else input.removeAttribute('aria-invalid');
      if (issue?.level === 'error' && !blocking) blocking = {input,issue};
    }
    return blocking;
  }
  for (const [inputId] of ID_FIELDS) $(inputId)?.addEventListener('input',identifierIssues);
  // Uploads go through XMLHttpRequest when the browser has it, for a byte-level progress bar (fetch cannot report
  // upload progress); errors are shaped exactly like request()'s.  Without XHR (tests) it falls back to request().
  function sendUpload(url, data, {timeout = 120000, onProgress = null} = {}) {
    if (typeof XMLHttpRequest === 'undefined' || !onProgress) return request(url,{method:'POST',body:data,timeout});
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST',url); xhr.timeout = timeout;
      xhr.setRequestHeader('Accept','application/json'); if (state.csrf) xhr.setRequestHeader('X-CSRF-Token',state.csrf);
      if (xhr.upload) { xhr.upload.onprogress = event => { if (event.lengthComputable) onProgress(event.loaded,event.total); }; xhr.upload.onload = () => onProgress(null,null); }
      xhr.onerror = () => reject(new Error('暂时无法连接服务。已保存的任务不会因此丢失，恢复连接后可继续。'));
      xhr.ontimeout = () => reject(new Error('请求超时。任务可能已保存，请刷新任务列表确认。'));
      xhr.onload = () => {
        let result;
        try { result = JSON.parse(xhr.responseText); } catch (_) { reject(new Error(`服务返回了无法读取的响应（HTTP ${xhr.status}），请稍后刷新。`)); return; }
        if (xhr.status >= 200 && xhr.status < 300) { resolve(result); return; }
        const message = [413,429].includes(xhr.status) ? WB.limitErrorText(xhr.status,result,errorText(result.error || result.message)) : errorText(result.error || result.message);
        const error = new Error(message); error.status = xhr.status; error.body = result;
        if (xhr.status === 401) expireSession();
        reject(error);
      };
      xhr.send(data);
    });
  }
  const uploadErrorText = error => error.status === 413 ? `${error.message} 可以分批上传（每批不超过 240 MB），或先在建模软件中抽稀网格。` : error.message;
  function duplicateDialog(file, body) {
    // Resolves with {action: 'open'|'reuse'|'force'|'cancel', jobId?}
    return new Promise(resolve => {
      state.modalResolve = resolve;
      const existing = Array.isArray(body.existing) ? body.existing : [];
      const rows = existing.slice(0,8).map(item => node('tr',{},node('td',{text:item.case_id || '—'}),node('td',{text:item.release_id ? releaseDisplay(item.release_id) : '—'}),node('td',{},badge(item.status)),node('td',{text:REVIEW_LABEL[item.review] || item.review || '—'}),node('td',{class:'coordinates'},timeNode(item.created_at,{cls:''}))));
      const finish = result => { state.modalResolve = null; closeModal(); resolve(result); };
      const reusable = typeof body.reusable === 'string' && body.reusable ? body.reusable : null;
      const openTarget = existing[0]?.job_id || null;
      const open = button('打开已有任务',() => finish({action:'open',jobId:openTarget})); open.disabled = !openTarget;
      const reuse = button('只跑新发布包（复用中心线与出口确认）',() => finish({action:'reuse'}),'primary'); reuse.disabled = !reusable;
      const force = button('强制重算',() => finish({action:'force'}));
      openModal(`同一几何已有任务 · ${file.name}`,[
        help(`上传内容的 SHA256 与 ${existing.length} 个已有任务一致（${String(body.input_sha256 || '').slice(0,12)}…）。`),
        node('div',{class:'table-wrap'},node('table',{},node('thead',{},node('tr',{},...['病例','发布包','状态','审阅','创建时间'].map(text => node('th',{text})))),node('tbody',{},rows))),
        help(reusable ? `「只跑新发布包」会复用任务 ${reusable} 的中心线与已确认出口，只计算所选发布包，不再重复确认。` : '已有任务尚未完成中心线或出口确认，不能复用；只能打开已有任务或强制重算。'),
      ],[button('跳过此文件',() => finish({action:'cancel'})),open,reuse,force]);
    });
  }
  $('upload-form').addEventListener('submit', async event => {
    event.preventDefault(); if (state.busy) return; clearNotice();
    if (!uploadReady()) { syncUploadAvailability(); $('upload-progress').hidden = false; $('upload-progress-text').textContent = state.authenticated ? '请先读取并选择可用的模型发布包，再上传。' : '会话已过期，请重新连接。'; return; }
    const form = event.currentTarget; if (![...($('stl-file').files || [])].length) return;
    const check = renderUploadFiles(), files = check.valid, rowIndex = check.rows.filter(row => row.ok).map(row => row.index);
    if (!files.length) { $('upload-progress').hidden = false; $('upload-progress-text').textContent = '没有可以上传的 STL 文件：请看文件列表里每个文件的原因，重新选择后再上传。'; return; }
    const idProblem = identifierIssues();
    if (idProblem) { $('upload-progress').hidden = false; $('upload-progress-text').textContent = idProblem.issue.text; if (idProblem.input.id === 'patient-id') { const meta = idProblem.input.closest?.('details'); if (meta) meta.open = true; } try { idProblem.input.focus(); } catch (_) {} return; }
    state.busy = true; const submit = $('upload-button'); submit.disabled = true; submit.textContent = '正在上传…';
    const progress = $('upload-progress'), bar = $('upload-progress-bar'), progressText = $('upload-progress-text');
    if (progress) progress.hidden = false; if (bar) {bar.value = 0; bar.max = files.length;}
    const rows = [...($('upload-files')?.children || [])].filter(el => String(el.className || '').includes('upload-file-row')); const failures = [], created = [], opened = [];
    const releaseIds = splitReleaseChoice($('upload-release').value);
    const baseCase = $('case-id').value.trim(), metadata = {
      case_id: baseCase, patient_id: $('patient-id').value.trim(), scan_label: $('scan-label').value.trim(),
      scan_date: $('scan-date')?.value || '',
      tags: $('case-tags').value.trim(), notes: $('case-notes').value.trim(), units: $('upload-units').value,
      release_id: releaseIds[0] || '', companion_release_ids: releaseIds.slice(1).join(','), remove_fragments: 'false', device: $('upload-device').value,
      seed_count: $('upload-seeds').value, threads: $('upload-threads').value.trim()
    };
    try {
      const caseIdFor = file => { const stem = file.name.replace(/\.stl$/i,''); return files.length > 1 ? (baseCase ? `${baseCase}-${stem}` : stem) : baseCase; };
      const appendCommon = data => { for (const [key,value] of Object.entries(metadata)) if (key !== 'case_id' && value !== '') data.append(key, value); };
      const setRow = (index, text) => { const stateNode = rows[rowIndex[index]]?.querySelector('.upload-file-state'); if (stateNode) stateNode.textContent = text; if (bar) bar.value = Math.max(bar.value, index + 1); };
      // Byte progress of the request in flight (one file, or one batch): the bar shows files done + the fraction sent.
      const bytesProgress = (label, doneBefore, count = 1) => (loaded, total) => {
        if (loaded === null) { if (progressText) progressText.textContent = `${label} · 已上传，服务正在检查几何…`; return; }
        const fraction = total ? loaded / total : 0;
        if (bar) bar.value = Math.min(bar.max, doneBefore + fraction * count);
        if (progressText) progressText.textContent = `${label} · ${Math.round(fraction * 100)}%（${WB.formatBytes(loaded)} / ${WB.formatBytes(total)}）`;
      };
      const record = (index, jobId, error, reusedFrom) => {
        if (jobId) { created.push(jobId); setRow(index, (reusedFrom ? `已创建任务 ${jobId}（复用 ${reusedFrom}）` : `已创建任务 ${jobId}`) + (releaseIds.length > 1 ? '；体场任务将在出口确认后自动创建' : '')); }
        else { failures.push(`${files[index].name}：${error}`); setRow(index, `失败：${error}`); }
      };
      const uploadOne = async (index, mode) => {
        const data = new FormData(); data.append('stl', files[index]); if (caseIdFor(files[index])) data.append('case_id', caseIdFor(files[index])); appendCommon(data); data.append('on_duplicate', mode);
        const result = await sendUpload('/api/jobs',data,{timeout:600000,onProgress:bytesProgress(`正在上传 ${files[index].name}`,index)});
        const job = result.job || result, id = job.id || result.job_id;
        if (!id) throw new Error('服务未返回任务编号。');
        return {id, reusedFrom: result.reused_from || job.reused_from || null};
      };
      const resolveDuplicate = async (index, body) => {
        const choice = await duplicateDialog(files[index], body);
        if (choice.action === 'open') { if (choice.jobId) opened.push(choice.jobId); setRow(index, `已打开已有任务 ${choice.jobId || ''}`); return; }
        if (choice.action === 'cancel') { setRow(index, '已跳过（同几何已有任务）'); return; }
        try { const {id, reusedFrom} = await uploadOne(index, choice.action); record(index, id, null, reusedFrom); }
        catch (error) { record(index, null, error.body?.reuse_unavailable ? '已有任务不能复用，请选择强制重算。' : uploadErrorText(error)); }
      };
      if (files.length === 1) {
        if (progressText) progressText.textContent = `正在上传 ${files[0].name}`;
        try { const {id, reusedFrom} = await uploadOne(0, 'ask'); record(0, id, null, reusedFrom); }
        catch (error) { if (error.status === 409 && error.body?.duplicate) await resolveDuplicate(0, error.body); else record(0, null, uploadErrorText(error)); }
      } else {
        // One multipart request per batch of files (repeated `stl` parts + per-file metadata list); v0.15: batches
        // are also cut by size (≤ 240 MB) so a large selection never meets the service's 256 MB batch limit (413).
        const duplicates = [];
        let start = 0;
        for (const chunk of WB.uploadChunks(files,{maxFiles:state.session?.max_batch_files || BATCH_LIMIT,maxBytes:state.session?.max_batch_bytes ? Math.floor(state.session.max_batch_bytes * 15 / 16) : 240 * 1024 * 1024})) {
          chunk.forEach((_, i) => setRow(start + i, '上传中…'));
          if (progressText) progressText.textContent = `正在批量上传 ${start + 1}–${start + chunk.length} / ${files.length}`;
          const data = new FormData(); for (const file of chunk) data.append('stl', file);
          data.append('metadata_json', JSON.stringify(chunk.map(file => ({case_id: caseIdFor(file)})))); appendCommon(data); data.append('on_duplicate','ask');
          try {
            const result = await sendUpload('/api/jobs/batch',data,{timeout:900000,onProgress:bytesProgress(`正在批量上传 ${start + 1}–${start + chunk.length} / ${files.length}`,start,chunk.length)});
            const seen = new Set();
            for (const item of result.results || []) {
              seen.add(item.index);
              if (item.duplicate) { duplicates.push({index:start + item.index, body:item}); setRow(start + item.index, '同几何已有任务，等待选择…'); continue; }
              record(start + item.index, item.job?.id || null, item.error?.message || '未创建任务', item.job?.reused_from || item.reused_from);
            }
            chunk.forEach((_, i) => { if (!seen.has(i)) record(start + i, null, '服务未返回该文件的结果'); });
          } catch (error) { chunk.forEach((_, i) => record(start + i, null, uploadErrorText(error))); }
          start += chunk.length;
        }
        for (const item of duplicates) { if (progressText) progressText.textContent = `请选择如何处理 ${files[item.index].name}`; await resolveDuplicate(item.index, item.body); }
      }
      savePreferences({upload: WB.uploadPreferencesFrom({...metadata, release_id: $('upload-release').value, remember_patient: $('remember-patient').checked})});
      await refreshJobs(true);
      const focus = created[created.length - 1] || opened[opened.length - 1];
      if (focus) { if (!failures.length) closeUploadDialog(); await selectJob(focus); }
      const openedText = (opened.length ? `；打开已有 ${opened.length} 个` : '') + (check.invalid ? `；${check.invalid} 个文件未上传（不是 STL、空文件或超过大小上限）` : '');
      if (failures.length) notify(`已创建 ${created.length}/${files.length} 个任务${openedText}；失败：${failures.join('；')}`);
      else notify(`已创建 ${created.length} 个任务${openedText}。`,false);
    } finally { state.busy = false; syncUploadAvailability(); submit.textContent = failures.length ? '重新上传并检查' : '上传并检查'; if (progressText) progressText.textContent = failures.length ? `有 ${failures.length} 个文件未完成：${failures.join('；')}。请核对后重新上传；已创建的任务可在病例列表中查看。` : `已完成 ${created.length}/${files.length}`; }
  });

  // ---------------------------------------------------------------------------------------------
  // Job list: by task (with review groups, C18) or by case (C1)
  // ---------------------------------------------------------------------------------------------
  $('refresh-jobs').addEventListener('click', async () => { clearNotice(); await refreshJobs(true); await pollJob(true); refreshTrash(); if (!state.current && overviewVisible()) refreshOverview(); });
  const readHistoryFilters = () => ({q:$('jobs-query')?.value.trim() || '',status:$('jobs-status')?.value || '',patient_id:$('jobs-patient')?.value.trim() || '',tag:$('jobs-tag')?.value.trim() || '',page_size:Number($('jobs-page-size')?.value || 20)});
  $('jobs-filter')?.addEventListener('submit', async event => { event.preventDefault(); state.history = {...state.history,...readHistoryFilters(),page:1}; await refreshJobs(true); });
  $('reset-filter')?.addEventListener('click', async () => { ['jobs-query','jobs-patient','jobs-tag'].forEach(id => {if ($(id)) $(id).value = '';}); if ($('jobs-status')) $('jobs-status').value = ''; state.history = {...state.history,...readHistoryFilters(),page:1}; await refreshJobs(true); });
  $('jobs-prev')?.addEventListener('click', async () => { if (state.history.page > 1) {state.history.page -= 1; await refreshJobs(true);} });
  $('jobs-next')?.addEventListener('click', async () => { if (state.history.page * state.history.page_size < state.history.total) {state.history.page += 1; await refreshJobs(true);} });
  function setListMode(mode) {
    state.listMode = mode === 'cases' ? 'cases' : 'jobs'; store.set('wss-list-mode', state.listMode);
    $('list-mode-jobs').setAttribute('aria-pressed',String(state.listMode === 'jobs')); $('list-mode-cases').setAttribute('aria-pressed',String(state.listMode === 'cases'));
    $('jobs-list').hidden = state.listMode !== 'jobs'; $('cases-list').hidden = state.listMode !== 'cases';
    $('jobs-status').disabled = state.listMode === 'cases';
    if (state.listMode === 'cases' && state.selectMode) { state.selectMode = false; state.selected.clear(); }
    $('select-jobs').disabled = state.listMode === 'cases';
    updateSelectionBar();
  }
  $('list-mode-jobs').addEventListener('click', async () => { setListMode('jobs'); state.history.page = 1; await refreshJobs(true); });
  $('list-mode-cases').addEventListener('click', async () => { setListMode('cases'); state.history.page = 1; await refreshJobs(true); });
  function updatePagination() {
    const pageLabel = $('jobs-page');
    if (pageLabel) pageLabel.textContent = state.quickFilter
      ? `${state.history.total} 条（最近 100 个任务内）`
      : state.history.total ? `${(state.history.page - 1) * state.history.page_size + 1}–${Math.min(state.history.page * state.history.page_size,state.history.total)} / ${state.history.total}` : '0 条';
    if ($('jobs-prev')) $('jobs-prev').disabled = Boolean(state.quickFilter) || state.history.page <= 1;
    if ($('jobs-next')) $('jobs-next').disabled = Boolean(state.quickFilter) || state.history.page * state.history.page_size >= state.history.total;
  }
  // Overview cards filter the list (§C1): the latest 100 tasks are fetched and filtered here.
  function setQuickFilter(key) {
    state.quickFilter = key && WB.QUICK_FILTERS[key] ? key : '';
    if (state.quickFilter && state.listMode === 'cases') setListMode('jobs');
    state.history.page = 1; renderQuickFilter(); refreshJobs(true);
    if (!state.current && overviewVisible()) renderOverview();   // the pressed tile follows the filter
    if (state.quickFilter && window.matchMedia('(max-width: 760px)').matches) $('jobs-title').scrollIntoView({block:'start'});
  }
  function renderQuickFilter() {
    const host = $('quick-filter'); if (!host) return;
    host.hidden = !state.quickFilter;
    host.replaceChildren(...(state.quickFilter ? [node('span',{text:`筛选：${WB.QUICK_FILTERS[state.quickFilter]}`}),button('清除',() => setQuickFilter(''),'text-button')] : []));
  }
  const hasHistoryFilters = () => Boolean(state.history.q || state.history.status || state.history.patient_id || state.history.tag);
  async function refreshJobs(showError = false) {
    if (!state.authenticated || state.listBusy) return;
    state.listBusy = true;
    try {
      if (state.listMode === 'cases') { await refreshCases(showError); return; }
      const filters = {...state.history}; const pendingOnly = filters.status === 'pending_review';
      if (pendingOnly) filters.status = 'done';
      if (state.quickFilter) { filters.page = 1; filters.page_size = 100; filters.status = ''; }
      const query = new URLSearchParams(); for (const [key,value] of Object.entries(filters)) if (key !== 'total' && value !== '' && value != null) query.set(key,String(value));
      if (state.viewAll) query.set('all','1');
      const result = await request(`/api/jobs?${query.toString()}`); let jobs = stampEta(Array.isArray(result) ? result : result.jobs || []);
      if (state.quickFilter) { jobs = WB.quickFilter(jobs, state.quickFilter); state.history.total = jobs.length; state.history.page = 1; }
      else if (!Array.isArray(result)) { state.history.total = Number(result.total || jobs.length); state.history.page = Number(result.page || state.history.page); state.history.page_size = Number(result.page_size || state.history.page_size); } else { state.history.total = jobs.length; }
      if (pendingOnly && !state.quickFilter) jobs = jobs.filter(WB.isPendingReview);
      state.jobs = jobs;
      for (const id of [...state.selected.keys()]) { const fresh = state.jobs.find(job => job.id === id); if (fresh) state.selected.set(id, fresh); else state.selected.delete(id); }
      renderJobsList();
      updateSelectionBar(); updatePagination(); markOnline();
    } catch (error) { if (error.status === 401 || !state.authenticated) return; if (showError) notify(error.message); setConnection('连接暂时中断 · 自动重试'); }
    finally { state.listBusy = false; }
  }
  const compactDiameter = value => value === null || value === undefined ? null : node('span',{class:'chip diameter-mini',title:'最大直径（中心线站位截面的最大 Feret 直径）'},'⌀ ',node('strong',{text:`${fmt(value,1)} mm`}));
  const jobTags = job => (Array.isArray(job.tags) ? job.tags : String(job.tags || '').split(',')).map(tag => String(tag).trim()).filter(Boolean);
  const jobMenuItems = job => [
    job.status === 'done' && {label:'打开三维报告',run:() => openReport(job)},
    job.status === 'done' && {label:'一页纸报告',run:() => openOnepage(job)},
    {label:'编辑信息',run:() => editMetadata(job.id),disabled:locked(job),title:locked(job) ? '已审阅锁定，先重新打开才能修改' : ''},
    job.status === 'done' && {label:'用其他发布包重跑…',run:() => batchRerunDialog([job],'用其他发布包重跑')},
    job.patient_id && {label:'患者时间线',run:() => timelineDialog(job.patient_id,job)},
    {separator:true},
    {label:'删除…',danger:true,disabled:!deletable(job),title:isRunning(job) ? '计算中的任务不能删除' : locked(job) ? '已审阅锁定的任务不能删除' : '',run:() => deleteJobs([job])},
  ];
  function jobRow(job, ref = {job}) {
    const b = button('', () => selectJob(ref.job.id),'job-button'); b.setAttribute('aria-current',String(job.id === state.current)); b.dataset.jobId = job.id;
    const name = node('span',{class:'job-name'});
    if (WB.unreadHas(state.unread, job.id)) name.append(node('span',{class:'unread-dot','aria-label':'未读',title:'任务状态有更新'}));
    name.append(node('span',{class:'job-case',text:job.case_id || '匿名病例',title:job.case_id || ''}));   // v0.15: truncated → full name on hover
    // A finished task shows its review state (待审阅 / 已审阅); anything else shows its run status.
    const status = job.status === 'done' ? reviewBadge(job) : badge(job.eta?.waiting ? 'queued' : job.status);
    const who = [job.patient_id && `患者 ${job.patient_id}`,job.scan_label].filter(Boolean).join(' · ');
    const tags = jobTags(job);
    const eta = etaRow(job);
    b.append(node('span',{class:'job-title'},name,status),
      node('span',{class:'job-meta'},withTitle(kindChip(jobKind(job)),job.release_short || job.model_release?.id || ''),who && node('span',{class:'job-subtitle',text:who}),tags.length ? node('span',{class:'job-tags',text:tags.map(tag => `#${tag}`).join(' ')}) : null),
      node('span',{class:'job-foot'},timeNode(job.created_at),eta || compactDiameter(WB.numberOrNull(job.max_diameter_mm))));
    const row = node('li',{class:`job-row${state.selectMode && state.selected.has(job.id) ? ' selected' : ''}`}); row.dataset.rowId = job.id;
    if (state.selectMode) {
      const box = node('input',{type:'checkbox','aria-label':`选择 ${job.case_id || job.id}`,checked:state.selected.has(job.id),disabled:isRunning(job),title:isRunning(job) ? '计算中的任务不能选择' : ''});
      box.addEventListener('change',() => { if (box.checked) state.selected.set(ref.job.id,ref.job); else state.selected.delete(ref.job.id); row.classList?.toggle('selected',box.checked); updateSelectionBar(); });
      row.append(node('span',{class:'job-select'},box));
    }
    row.append(b,moreButton(`${job.case_id || job.id} 的更多操作`,() => jobMenuItems(ref.job)));
    row._wss = {ref, button:b, eta};
    return row;
  }
  // v0.14 (F4): the list is patched, not rebuilt.  A row is rebuilt only when something it shows changed (version,
  // status, review, identity, tags, release, relative time, selection, unread …); unchanged rows keep their DOM node,
  // so keyboard focus (↑ / ↓ navigation) and the scroll position survive a refresh.
  const rowCache = new Map(), groupEls = new Map();
  const rowKey = job => JSON.stringify([job.version ?? null, job.status || '', job.review?.status || '', Boolean(job.eta), Boolean(job.eta?.waiting), job.case_id || '', job.patient_id || '',
    job.scan_label || '', jobTags(job).join(','), job.release_short || job.model_release?.id || '', jobKind(job) || '', job.created_at || '', job.created_at ? friendlyTime(job.created_at) : '',
    job.max_diameter_mm ?? null, state.selectMode, state.selected.has(job.id), WB.unreadHas(state.unread, job.id)]);
  function cachedRow(job) {
    const key = rowKey(job), hit = rowCache.get(job.id);
    if (hit && hit.key === key) {
      const w = hit.el._wss; w.ref.job = job; w.button.setAttribute('aria-current',String(job.id === state.current));
      if (w.eta?._etaEntry) { w.eta._etaEntry.job = job; w.eta._etaEntry.paint(); }
      return hit.el;
    }
    const el = jobRow(job); rowCache.set(job.id,{key,el}); return el;
  }
  // Replace a parent's children only when the wanted sequence differs from what is there.
  function syncChildren(parent, wanted) {
    const current = Array.from(parent.children || []);
    if (current.length === wanted.length && current.every((el,i) => el === wanted[i])) return false;
    parent.replaceChildren(...wanted); return true;
  }
  function renderJobsList() {
    const list = $('jobs-list'); $('jobs-empty').hidden = state.jobs.length > 0;
    $('jobs-empty').textContent = state.quickFilter ? `没有「${WB.QUICK_FILTERS[state.quickFilter]}」的任务。` : hasHistoryFilters() ? '没有符合条件的任务。' : '暂无任务。点击「＋ 新建预测」或把 STL 文件拖到页面上开始。';
    // Remember what had keyboard focus inside the list and where it was scrolled.
    const active = document.activeElement, inside = Boolean(active && typeof list.contains === 'function' && list.contains(active));
    const focus = inside ? {id:active.closest?.('.job-row')?.dataset?.rowId || null, cls:String(active.className || '')} : null, scrollTop = list.scrollTop;
    // §19.9 order: 需要处理 → 计算中与排队 → 待审阅 → 已审阅 (collapsed) → 已取消 (collapsed).
    const groups = WB.workGroups(state.jobs), shown = [];
    for (const group of WB.WORK_GROUPS) {
      const jobs = groups[group.key]; if (!jobs.length) continue;
      let entry = groupEls.get(group.key);
      if (!entry) {
        const count = node('span',{class:'count'}), ul = node('ul',{});
        const details = node('details',{open:state.groupsOpen[group.key] !== false},node('summary',{class:group.tone},node('span',{text:group.label}),count),ul);
        details.addEventListener('toggle',() => { state.groupsOpen[group.key] = details.open; store.set('wss-groups-v2',JSON.stringify(state.groupsOpen)); });
        entry = {li:node('li',{class:`job-group group-${group.key}`},details),count,ul}; groupEls.set(group.key,entry);
      }
      if (entry.count.textContent !== String(jobs.length)) entry.count.textContent = String(jobs.length);
      syncChildren(entry.ul,jobs.map(cachedRow));
      shown.push(entry.li);
    }
    syncChildren(list,shown);
    const ids = new Set(state.jobs.map(job => job.id)); for (const id of [...rowCache.keys()]) if (!ids.has(id)) rowCache.delete(id);
    if (focus && document.activeElement !== active) {
      const row = focus.id ? rowCache.get(focus.id)?.el : null;
      const target = row ? ([...(row.querySelectorAll?.('button,input') || [])].find(el => String(el.className || '') === focus.cls) || row._wss?.button) : null;
      try { target?.focus({preventScroll:true}); } catch (_) {}
    }
    if (list.scrollTop !== scrollTop) list.scrollTop = scrollTop;
  }
  // Relative times ("3 分钟前") in list rows advance in place; no row is rebuilt for it.
  function refreshListTimes() {
    for (const el of document.querySelectorAll?.('#jobs-list time.job-time[datetime]') || []) { const iso = el.getAttribute('datetime'); const text = iso ? friendlyTime(iso) : ''; if (text && el.textContent !== text) el.textContent = text; }
  }
  function markCurrent() { for (const el of document.querySelectorAll('[data-job-id]')) el.setAttribute('aria-current',String(el.dataset.jobId === state.current)); for (const card of document.querySelectorAll('.case-card')) card.classList.toggle('current', Boolean(state.current) && card.querySelector(`[data-job-id="${CSS.escape(state.current || '')}"]`) !== null); }
  async function refreshCases(showError) {
    const query = new URLSearchParams(); for (const key of ['q','patient_id','tag','page','page_size']) { const value = state.history[key]; if (value !== '' && value != null) query.set(key,String(value)); }
    if (state.viewAll) query.set('all','1');
    const host = $('cases-list');
    try {
      const result = await request(`/api/cases?${query.toString()}`);
      state.cases = Array.isArray(result.cases) ? result.cases : []; state.caseReleases = Array.isArray(result.releases) ? result.releases : [];
      state.history.total = Number(result.total || state.cases.length); state.history.page = Number(result.page || state.history.page); state.history.page_size = Number(result.page_size || state.history.page_size);
      state.jobs = state.cases.flatMap(card => (card.runs || []).map(run => ({id:run.job_id, case_id:(card.case_ids || [])[0] || '', patient_id:card.patient_id || '', scan_label:card.scan_label || '', scan_date:card.scan_date || '', tags:card.tags || [], status:run.status, review:{status:run.review}, version:run.version, family:run.family, model_release:{id:run.release_id}, created_at:run.created_at, run_identity:run.run_identity})));
      $('jobs-empty').hidden = state.cases.length > 0;
      host.replaceChildren(...state.cases.map(caseCard)); markCurrent(); updatePagination();
    } catch (error) {
      if (error.status === 404) { host.replaceChildren(help('当前服务版本尚未提供病例卡视图，请切换到「按任务」。')); $('jobs-empty').hidden = true; }
      else if (error.status !== 401 && state.authenticated) { if (showError) notify(error.message); setConnection('连接暂时中断 · 自动重试'); }
    }
  }
  function caseCard(card) {
    const model = WB.caseCardModel(card, state.caseReleases.length ? state.caseReleases : state.releases.map(item => ({id:item.id || item.release, family:releaseFamily(item.id || item.release)})));
    const head = node('div',{class:'case-head'},node('strong',{text:model.title}));
    // §17.4: the chip comes from the latest finished run's light snapshot; no extra request is made.
    const chip = diameterChip(model.maxDiameterMm); if (chip) head.append(chip);
    if (model.pendingReview) head.append(node('span',{class:'badge pending',text:`待审阅 ${model.pendingReview}`}));
    const article = node('article',{class:'case-card'},head);
    if (model.subtitle) article.append(node('small',{class:'job-subtitle',text:model.subtitle}));
    if (model.tags.length) article.append(node('div',{class:'case-tags',text:`#${model.tags.join(' #')}`}));
    const runs = node('ul',{class:'case-runs'});
    for (const run of model.runs) {
      const chip = button('',() => selectJob(run.job_id),'run-chip'); chip.dataset.jobId = run.job_id; chip.setAttribute('aria-current',String(run.job_id === state.current));
      const pseudo = {status:run.status,review:{status:run.review}};
      const kind = WB.releaseKind(releaseOf(run.release_id) || run.release_id, run.family);
      chip.append(kindChip(kind) || node('span',{class:'run-family',text:FAMILY_TEXT[run.family] || '结果'}));
      chip.title = run.release_id ? releaseDisplay(run.release_id,{includeId:true}) : '';
      chip.append(run.status === 'done' ? reviewBadge(pseudo) : badge(run.status));
      chip.append(timeNode(run.created_at));
      if (WB.unreadHas(state.unread, run.job_id)) chip.prepend(node('span',{class:'unread-dot','aria-label':'未读'}));
      runs.append(node('li',{},chip));
    }
    article.append(runs);
    const actions = node('div',{class:'case-actions'});
    // §15.2: the card's identity fields are edited through the most recent run of this geometry.
    const latestRun = [...model.runs].sort((a,b) => String(b.created_at || '').localeCompare(String(a.created_at || '')))[0];
    if (latestRun?.job_id) { const edit = button('编辑',() => editMetadata(latestRun.job_id)); edit.title = `编辑病例信息（最近任务 ${latestRun.job_id}）`; actions.append(edit); }
    // §19.3: the patient's follow-up timeline (same anonymous patient id across scans).
    if (card.patient_id) actions.append(button('患者时间线',() => timelineDialog(card.patient_id,{model_release:{id:latestRun?.release_id || ''}})));
    if (model.compare) actions.append(button('并排打开 WSS + 体场',() => { const params = new URLSearchParams({left:model.compare.left,right:model.compare.right}); window.open(`/compare?${params.toString()}`,'_blank','noopener'); }));
    for (const releaseId of model.missing) {
      const label = FAMILY_TEXT[WB.familyOfRelease(releaseId, state.caseReleases) || releaseFamily(releaseId)] || releaseId;
      const rerun = button(`补跑 ${label}`,() => rerunFromCase(model.reusable, releaseId),'primary'); rerun.title = releaseId; rerun.disabled = !model.reusable || state.busy;
      actions.append(rerun);
    }
    if (model.doneJobIds.length) actions.append(button('导出该病例汇总',() => exportSummary(model.doneJobIds,'csv')));
    if (actions.children.length) article.append(actions);
    if (model.missing.length && !model.reusable) article.append(node('div',{class:'case-missing',text:'缺少已确认出口的任务，不能一键补跑；请打开任务完成出口确认。'}));
    return article;
  }
  async function rerunFromCase(sourceId, releaseId) {
    if (!sourceId || state.busy) return;
    state.busy = true; clearNotice();
    try {
      const source = await request(`/api/jobs/${encodeURIComponent(sourceId)}`); const job = source.job || source;
      const result = await request(jobUrl(job,'/rerun'),{method:'POST',body:{version:job.version,release_id:releaseId}});
      const created = result.job || result;
      notify(`已创建补跑任务 ${created.id}，复用中心线与出口确认。`,false);
      state.busy = false; await refreshJobs(true); if (created.id) await selectJob(created.id);
    } catch (error) { notify(error.message); }
    finally { state.busy = false; }
  }

  // ---------------------------------------------------------------------------------------------
  // Case metadata editing (§15.2): identity fields only, never numbers.
  // ---------------------------------------------------------------------------------------------
  async function editMetadata(target) {
    // A job object from the detail view already carries version and notes; a case-card run is just an id.
    let job = target;
    if (typeof target === 'string' || !target || target.version === undefined) {
      const id = typeof target === 'string' ? target : target?.id;
      if (!id) return;
      try { const result = await request(`/api/jobs/${encodeURIComponent(id)}`); job = result.job || result; }
      catch (error) { notify(error.message); return; }
    }
    metadataDialog(job);
  }
  function metadataDialog(job) {
    const field = (label, control, hint) => node('div',{class:'meta-field'},node('label',{text:label}),control,hint && help(hint));
    const caseId = node('input',{type:'text',maxlength:160,value:job.case_id || '','aria-label':'病例编号',autocomplete:'off'});
    const patient = node('input',{type:'text',maxlength:80,value:job.patient_id || '','aria-label':'患者编号',autocomplete:'off'});
    const scanLabel = node('input',{type:'text',maxlength:120,value:job.scan_label || '','aria-label':'扫描标签',autocomplete:'off'});
    const scanDate = node('input',{type:'date',value:job.scan_date || '','aria-label':'扫描日期'});
    const tags = node('input',{type:'text',maxlength:1024,value:WB.tagsText(job.tags),'aria-label':'检索标签',autocomplete:'off'});
    const notes = node('textarea',{rows:3,maxlength:2000,'aria-label':'备注'}); notes.value = job.notes || '';
    const status = node('p',{class:'help'});
    const submit = button('保存信息',async () => {
      // v0.15: the same identifier checks as the upload form, with focus on the field at fault
      for (const [input,label,max] of [[caseId,'病例编号',160],[patient,'患者编号',80],[scanLabel,'扫描标签',120]]) {
        const issue = WB.identifierIssue(input.value,{label,max});
        if (issue?.level === 'error') { status.textContent = issue.text; input.setAttribute('aria-invalid','true'); try { input.focus(); } catch (_) {} return; }
        input.removeAttribute('aria-invalid');
      }
      submit.disabled = true; status.textContent = '正在保存…';
      const body = WB.metadataPayload({case_id:caseId.value, patient_id:patient.value, scan_label:scanLabel.value, scan_date:scanDate.value, tags:tags.value, notes:notes.value}, job.version);
      // The service refuses to blank an existing case id; an anonymous task simply keeps none.
      if (!body.case_id) { if (job.case_id) { status.textContent = '病例编号不能为空。'; submit.disabled = false; caseId.focus(); return; } delete body.case_id; }
      if (body.tags.length > 12) { status.textContent = '最多 12 个检索标签。'; submit.disabled = false; return; }
      try {
        const result = await request(jobUrl(job,'/metadata'),{method:'POST',body});
        const updated = result.job || result;
        closeModal(); notify(`已更新任务 ${updated.id || job.id} 的病例信息。`,false);
        if ((updated.id || job.id) === state.current) { state.renderKey = null; await pollJob(true); }
        await refreshJobs(true);
      } catch (error) {
        status.textContent = error.status === 404 ? '当前服务版本尚未提供病例信息修改。'
          : error.status === 409 ? error.message
          : error.message;
        submit.disabled = false;
      }
    },'primary');
    openModal(`编辑病例信息 · ${job.case_id || job.id}`,[
      help('只修改检索与展示用的标识信息，不影响已计算的几何、预测数值和运行清单。请使用匿名编号。'),
      node('div',{class:'meta-form'},
        field('病例编号',caseId,'已有病例编号不能清空。'),
        field('患者编号',patient),
        field('扫描标签',scanLabel),
        field('扫描日期',scanDate),
        field('检索标签',tags,'逗号分隔，例如 AAA, 随访。'),
        field('备注',notes)),
      status,
    ],[button('取消',closeModal),submit]);
    caseId.focus();
  }

  // ---------------------------------------------------------------------------------------------
  // Batch rerun with another release (§15.4): sequential reruns over the selected finished jobs.
  // ---------------------------------------------------------------------------------------------
  function batchRerunDialog(jobs, title = '用发布包重跑所选任务') {
    if (!jobs.length) return;
    if (!state.releases.length) { notify('没有可用的发布包，无法批量重跑。'); return; }
    const select = node('select',{'aria-label':'重跑使用的发布包'},...state.releases.map(item => {
      const id = item.id || item.release;
      return node('option',{value:id,text:releaseDisplay(item)});
    }));
    const preferred = state.releases.find(item => item.default) || state.releases[0];
    if (preferred) select.value = preferred.id || preferred.release;
    const progress = node('p',{class:'help','aria-live':'polite',text:jobs.length === 1 ? `为 ${jobs[0].case_id || jobs[0].id} 新建一个任务，复用它的中心线与出口确认，原结果保留。` : `已选择 ${jobs.length} 个任务。只有已完成且已确认出口的任务会被重跑，其余会跳过并说明原因。`});
    const log = node('ul',{class:'batch-log'});
    const bar = node('progress',{max:jobs.length,value:0}); bar.hidden = true;
    const start = button('开始重跑',async () => {
      const releaseId = select.value;
      if (!releaseId) { progress.textContent = '请先选择发布包。'; return; }
      start.disabled = true; select.disabled = true; cancel.disabled = true; bar.hidden = false; bar.value = 0;
      state.busy = true; state.modalLocked = true; updateSelectionBar();
      const created = [], skipped = [], failed = [];
      log.replaceChildren();
      for (const [index, job] of jobs.entries()) {
        const name = job.case_id || job.id;
        progress.textContent = `正在提交 ${index + 1} / ${jobs.length}：${name}`;
        const line = node('li',{text:`${name}：读取任务…`}); log.append(line);
        try {
          const source = await request(`/api/jobs/${encodeURIComponent(job.id)}`);
          const full = source.job || source;
          const reason = WB.rerunSkipReason(full);
          if (reason) { skipped.push(job.id); line.textContent = `${name}：跳过（${reason}）`; line.classList.add('failed'); continue; }
          const result = await request(jobUrl(full,'/rerun'),{method:'POST',body:{version:full.version,release_id:releaseId}});
          const fresh = result.job || result;
          created.push(fresh.id || job.id); line.textContent = `${name}：已创建任务 ${fresh.id || '—'}`;
        } catch (error) { failed.push(job.id); line.textContent = `${name}：失败 · ${error.message}`; line.classList.add('failed'); }
        finally { bar.value = index + 1; }
      }
      progress.textContent = `完成：新建 ${created.length} 个任务${skipped.length ? `，跳过 ${skipped.length} 个` : ''}${failed.length ? `，失败 ${failed.length} 个` : ''}。新任务复用各自的中心线与出口确认。`;
      state.busy = false; state.modalLocked = false; cancel.textContent = '关闭'; cancel.disabled = false;
      await refreshJobs(true); updateSelectionBar();
    },'primary');
    const cancel = button('取消',closeModal);
    openModal(title,[
      progress,
      node('div',{class:'form-grid'},node('div',{},node('label',{text:'发布包'}),select),node('div',{},node('label',{text:'说明'}),help('逐个调用「换发布包重跑」，每例新建一个任务，原结果保留。已审阅锁定的任务也可以重跑。'))),
      bar, log,
    ],[cancel,start]);
  }

  // ---------------------------------------------------------------------------------------------
  // Live events: per-job stream (detail page) and owner-level stream with notifications (C5)
  // ---------------------------------------------------------------------------------------------
  function closeEvents() { if (state.events) { try { state.events.close(); } catch (_) {} } state.events = null; state.eventsLive = false; }
  function openEvents(id) {
    // Live status over server-sent events; the 2 s poll below only runs while no stream is live.
    closeEvents();
    if (typeof EventSource === 'undefined') return;
    try {
      const source = new EventSource(`/api/jobs/${encodeURIComponent(id)}/events`);
      state.events = source;
      const handle = event => {
        let data = null; try { data = JSON.parse(event.data); } catch (_) { return; }
        if (!data || data.job_id !== state.current) return;
        state.eventsLive = true;
        if (data.status === 'deleted') { clearDetail('该任务已在另一处被删除。'); refreshJobs(); return; }
        if (data.final) closeEvents();
        if (data.action && data.action !== 'progress') state.renderKey = null;
        pollJob(); if (data.action && data.action !== 'progress') refreshJobs();
      };
      source.addEventListener('snapshot', handle); source.addEventListener('job', handle);
      source.onerror = () => { state.eventsLive = false; };
    } catch (_) { closeEvents(); }
  }
  function closeOwnerEvents() { if (state.ownerEvents) { try { state.ownerEvents.close(); } catch (_) {} } state.ownerEvents = null; state.ownerLive = false; }
  function openOwnerEvents() {
    if (state.ownerEvents || typeof EventSource === 'undefined') return;
    try {
      const source = new EventSource('/api/events'); state.ownerEvents = source;
      // v0.14 (F4): while this stream is open the list follows its events; the 10 s timer refresh is only a fallback.
      source.onopen = () => { state.ownerLive = true; state.streamDown = false; markOnline(); };
      source.addEventListener('job', event => { state.ownerLive = true; let data = null; try { data = JSON.parse(event.data); } catch (_) { return; } handleOwnerEvent(data); });
      // v0.15: say that live updates stopped (the dot turns amber); the browser retries by itself, and a stream the
      // browser gave up on (readyState CLOSED) is reopened by the 2 s timer below.  Lists keep refreshing meanwhile.
      source.onerror = () => {
        state.ownerLive = false; if (source.readyState === EventSource.CLOSED) state.ownerEvents = null;
        if (state.authenticated && !state.relogin) { state.streamDown = true; setConnection('实时更新中断 · 正在重连…',false,true); }
      };
    } catch (_) { state.ownerEvents = null; }
  }
  function persistUnread() { store.set('wss-unread', WB.unreadSerialize(state.unread)); updateUnreadBadge(); }
  function updateUnreadBadge() {
    const count = WB.unreadCount(state.unread), el = $('unread-badge');
    el.hidden = count === 0; el.textContent = count ? String(count) : '';
    document.title = WB.titleWithUnread(state.baseTitle, count);
    for (const dot of document.querySelectorAll('.unread-dot')) { const owner = dot.closest('[data-job-id]'); if (owner && !WB.unreadHas(state.unread, owner.dataset.jobId)) dot.remove(); }
  }
  function scheduleListRefresh() { clearTimeout(state.refreshTimer); state.refreshTimer = setTimeout(() => refreshJobs(), 400); }
  function scheduleOverviewRefresh() { clearTimeout(state.overviewTimer); state.overviewTimer = setTimeout(() => refreshOverview(), 450); }
  function handleOwnerEvent(data) {
    if (!data || !data.job_id) return;
    if (WB.shouldNotify(data, state.lastStatus)) {
      if (data.job_id !== state.current || document.hidden) { WB.unreadAdd(state.unread, data.job_id, data.version); persistUnread(); }
      showNotification(data);
    }
    if (data.job_id === state.current && data.action !== 'progress') { state.renderKey = null; pollJob(); }
    if (data.action !== 'progress') scheduleListRefresh();
    // The overview and the cohort table change only when a job finishes / needs a person / disappears.
    if (data.action !== 'progress' && (WB.FINAL.includes(data.status) || /^awaiting_/.test(data.status || '') || data.status === 'deleted')) {
      if (!state.current && overviewVisible()) scheduleOverviewRefresh();
      if (data.status === 'done') { if ($('cohort-panel')?.open) loadCohort(true); else state.cohort = null; }
    }
  }
  function showNotification(data) {
    if (state.prefs.notifications?.enabled === false) return;
    if (typeof Notification === 'undefined' || Notification.permission !== 'granted') return;
    if (!document.hidden && data.job_id === state.current) return;
    const {title, body} = WB.notificationText(data);
    try {
      const notification = new Notification(title,{body,tag:`wss-job-${data.job_id}`});
      notification.onclick = () => { try { window.focus(); } catch (_) {} selectJob(data.job_id); notification.close(); };
    } catch (_) {}
  }
  const askNotificationPermission = () => {
    // First user gesture after login: ask once; later clicks stop re-checking.
    if (!state.authenticated || state.prefs.notifications?.enabled === false) return;
    if (typeof Notification !== 'undefined' && Notification.permission === 'default') Notification.requestPermission().catch(() => {});
    document.removeEventListener('click', askNotificationPermission);
  };
  document.addEventListener('click', askNotificationPermission);

  // ---------------------------------------------------------------------------------------------
  // Multi-select bar: delete (to trash), summary export (C14), bundle (C15), batch figures (C13)
  // ---------------------------------------------------------------------------------------------
  function updateSelectionBar() {
    const bar = $('selection-bar'); if (!bar) return;
    bar.hidden = !state.selectMode;
    $('select-jobs').setAttribute('aria-pressed',String(state.selectMode)); $('select-jobs').textContent = state.selectMode ? '完成' : '选择';
    const selected = [...state.selected.values()], done = selected.filter(job => job.status === 'done'), removable = selected.filter(deletable);
    $('selection-count').textContent = `已选 ${selected.length} 个${done.length ? ` · ${done.length} 个已完成` : ''}`;
    $('delete-selected').disabled = removable.length === 0 || state.busy; $('delete-selected').textContent = removable.length ? `删除选中 (${removable.length})` : '删除选中';
    for (const id of ['export-csv','export-xlsx','bundle-selected','batch-export','rerun-selected']) { $(id).disabled = done.length === 0 || state.busy; $(id).title = done.length === 0 && selected.length ? '只对已完成的任务可用：已选的任务里还没有已完成的' : ''; }
    $('delete-selected').title = removable.length === 0 && selected.length ? '计算中或已审阅锁定的任务不能删除' : '';
  }
  $('select-jobs')?.addEventListener('click', () => { state.selectMode = !state.selectMode; if (!state.selectMode) state.selected.clear(); refreshJobs(true); });
  $('select-finished')?.addEventListener('click', () => { for (const job of state.jobs) if (WB.FINAL.includes(job.status)) state.selected.set(job.id,job); refreshJobs(true); });
  $('delete-selected')?.addEventListener('click', () => { const jobs = [...state.selected.values()].filter(deletable); const lockedCount = [...state.selected.values()].filter(locked).length; if (lockedCount) notify(`${lockedCount} 个已审阅锁定的任务不会被删除。`); if (jobs.length) deleteJobs(jobs); });
  const selectedDone = () => [...state.selected.values()].filter(job => job.status === 'done');
  async function exportSummary(ids, format) {
    if (!ids.length) return;
    if (ids.length > EXPORT_LIMIT) { notify(`一次最多导出 ${EXPORT_LIMIT} 个任务的汇总表。`); return; }
    try { await downloadFile(WB.exportUrl(ids, format),{filename:`wss_summary.${format}`}); notify(`已导出 ${ids.length} 个任务的汇总表（${format}）。`,false); }
    catch (error) { notify(error.status === 404 ? '当前服务版本尚未提供汇总表导出。' : error.message); }
  }
  async function bundleJobs(jobs) {
    if (!jobs.length) return;
    if (jobs.length > BUNDLE_LIMIT) { notify(`一次最多打包 ${BUNDLE_LIMIT} 个任务。`); return; }
    try {
      if (jobs.length === 1) await downloadFile(jobUrl(jobs[0],'/bundle.zip'),{filename:`${jobs[0].case_id || 'case'}_${jobs[0].id}.zip`});
      else await downloadFile('/api/jobs/bundle',{method:'POST',body:{ids:jobs.map(job => job.id)},filename:'wss_bundle.zip'});
      notify(`已打包 ${jobs.length} 个任务的全部文件。`,false);
    } catch (error) { notify(error.status === 404 ? '当前服务版本尚未提供打包下载。' : error.message); }
  }
  $('export-csv').addEventListener('click',() => exportSummary(selectedDone().map(job => job.id),'csv'));
  $('export-xlsx').addEventListener('click',() => exportSummary(selectedDone().map(job => job.id),'xlsx'));
  $('bundle-selected').addEventListener('click',() => bundleJobs(selectedDone()));
  $('batch-export').addEventListener('click',() => batchExportDialog(selectedDone()));
  $('rerun-selected').addEventListener('click',() => batchRerunDialog(selectedDone()));
  function batchExportDialog(jobs) {
    if (!jobs.length) return;
    const Batch = window.WssBatchExport; if (!Batch) { notify('批量出图脚本未加载，请刷新页面。'); return; }
    const source = value => node('input',{type:'radio',name:'batch-source',value});
    const presetSelect = node('select',{'aria-label':'内置预设'});
    for (const [family, names] of Object.entries(WB.BUILTIN_PRESETS)) presetSelect.append(node('optgroup',{label:FAMILY_TEXT[family]},...names.map(name => node('option',{value:name,text:name}))));
    const stateText = node('textarea',{rows:4,placeholder:'粘贴报告「导出」菜单里「导出 JSON」得到的视图状态','aria-label':'视图状态 JSON'});
    const stateFile = node('input',{type:'file',accept:'.json,application/json','aria-label':'上传视图状态文件'});
    stateFile.addEventListener('change',async () => { const file = stateFile.files?.[0]; if (!file) return; try { stateText.value = await file.text(); useJson.checked = true; } catch (_) { notify('无法读取视图状态文件。'); } });
    const usePreset = source('preset'), useJson = source('json'); usePreset.checked = true;
    const scale = node('select',{'aria-label':'分辨率'},...[1,2,4].map(k => node('option',{value:String(k),text:`${k}×`})));
    const background = node('select',{'aria-label':'背景'},node('option',{value:'white',text:'白底'}),node('option',{value:'transparent',text:'透明'}),node('option',{value:'current',text:'当前背景'}));
    const colorbar = node('select',{'aria-label':'色标'},node('option',{value:'overlay',text:'叠加在图上'}),node('option',{value:'svg',text:'单独导出 SVG'}),node('option',{value:'none',text:'不含色标'}));
    const lang = node('select',{'aria-label':'标签语言'},node('option',{value:'zh',text:'中文'}),node('option',{value:'en',text:'English'}));
    scale.value = '2';
    const progress = node('progress',{max:jobs.length,value:0}); progress.hidden = true;
    const log = node('ul',{class:'batch-log'});
    const summary = help(`已选择 ${jobs.length} 个已完成任务。视图状态按族匹配：壁面状态只应用于壁面任务，体场状态只应用于体场任务。`);
    const start = button('开始出图',async () => {
      let viewState;
      try { viewState = usePreset.checked ? WB.presetState(presetSelect.value) : WB.parseViewStateJSON(stateText.value); }
      catch (error) { notify(error.message); return; }
      if (!viewState) { notify('请选择内置预设或粘贴视图状态。'); return; }
      const plan = WB.planBatchExport(jobs, viewState);
      log.replaceChildren(...plan.skipped.map(item => node('li',{class:'failed',text:`${item.case_id || item.id}：跳过（${item.message}）`})));
      if (!plan.selected.length) { notify('没有可以应用该视图状态的任务。'); return; }
      start.disabled = true; cancel.disabled = true; progress.hidden = false; progress.value = 0; state.busy = true; state.modalLocked = true; updateSelectionBar();
      const options = {scale:Number(scale.value),background:background.value,colorbar:colorbar.value,ui:false,lang:lang.value};
      try {
        const result = await Batch.run({jobs:plan.selected,state:viewState,options,onProgress:info => {
          if (info.phase === 'start') log.append(node('li',{text:`${info.job.case_id || info.job.id}：渲染中…`,id:`batch-${info.job.id}`}));
          else { const item = document.getElementById(`batch-${info.job.id}`); if (item) { item.textContent = `${info.job.case_id || info.job.id}：${info.ok ? '已导出' : `失败 · ${info.message}`}`; item.classList.toggle('failed',!info.ok); } progress.value = info.index + 1; }
        }});
        const stamp = new Date().toISOString().replace(/[-:]/g,'').slice(0,13).replace('T','_');
        if (result.count) saveBlob(new Blob([result.zip],{type:'application/zip'}),`wss_batch_export_${stamp}.zip`);
        summary.textContent = `完成：${result.count} 例已导出${result.failed.length ? `，${result.failed.length} 例失败` : ''}${plan.skipped.length ? `，${plan.skipped.length} 例跳过` : ''}。`;
      } catch (error) { notify(error.message); }
      finally { state.busy = false; state.modalLocked = false; updateSelectionBar(); cancel.textContent = '关闭'; cancel.disabled = false; }
    },'primary');
    const cancel = button('取消',closeModal);
    openModal('跨病例批量出图',[
      summary,
      node('label',{class:'radio-row'},usePreset,node('span',{text:'内置预设'})),presetSelect,
      node('label',{class:'radio-row'},useJson,node('span',{text:'报告导出的视图状态 JSON'})),stateText,stateFile,
      node('div',{class:'form-grid'},node('div',{},node('label',{text:'分辨率'}),scale),node('div',{},node('label',{text:'背景'}),background),node('div',{},node('label',{text:'色标'}),colorbar),node('div',{},node('label',{text:'标签语言'}),lang)),
      progress,log,
    ],[cancel,start]);
    stateText.addEventListener('input',() => { if (stateText.value.trim()) useJson.checked = true; });
  }

  // ---------------------------------------------------------------------------------------------
  // Delete to trash and trash panel (C4)
  // ---------------------------------------------------------------------------------------------
  function clearDetail(message) {
    state.viewer?.dispose(); state.viewer = null; closeEvents(); state.current = null; state.job = null; state.renderKey = null; state.sequence++;
    syncRoute(null,{replace:true}); markCurrent();
    showOverview().then(() => { if (message) notify(message,false); });
  }
  async function deleteJobs(jobs) {
    if (state.busy || !jobs.length) return;
    if (jobs.length > DELETE_LIMIT) { notify(`一次最多删除 ${DELETE_LIMIT} 个任务，请分批操作。`); return; }
    const names = jobs.slice(0,5).map(job => job.case_id || job.id).join('、') + (jobs.length > 5 ? ` 等 ${jobs.length} 个` : '');
    const message = `将删除 ${jobs.length} 个任务（${names}）及其全部文件：输入 STL、中心线、报告、导出结果与历史记录。\n任务会进入回收站，30 天内可恢复；到期后自动彻底清除。`;
    if (!await confirmDialog(`删除 ${jobs.length} 个任务`,message,{confirmLabel:'移入回收站',danger:true})) return;
    if (state.busy) return;
    state.busy = true; updateSelectionBar(); clearNotice();
    const wasCurrent = state.current;
    try {
      let deleted = [], failures = [], trashed = false;
      if (jobs.length === 1) {
        try { const result = await request(jobUrl(jobs[0],'/delete'),{method:'POST',body:{version:jobs[0].version}}); deleted.push(result.id || jobs[0].id); trashed = Boolean(result.trashed); }
        catch (error) { failures.push(`${jobs[0].case_id || jobs[0].id}：${error.message}`); }
      } else {
        const result = await request('/api/jobs/delete',{method:'POST',body:{jobs:jobs.map(job => ({id:job.id,version:job.version}))},timeout:120000});
        for (const item of result.results || []) {
          if (item.deleted) { deleted.push(item.id); if (item.trashed) trashed = true; }
          else { const job = jobs.find(j => j.id === item.id); failures.push(`${job?.case_id || item.id}：${item.error?.message || '未删除'}`); }
        }
      }
      for (const id of deleted) { state.selected.delete(id); WB.unreadRemove(state.unread, id); }
      persistUnread();
      if (deleted.includes(state.current)) clearDetail();
      const where = trashed ? '已移入回收站，30 天内可恢复' : '已删除';
      const undo = trashed && deleted.length ? [{label:'撤销',run:() => undoDelete(deleted.slice(),deleted.includes(wasCurrent) ? wasCurrent : null)}] : [];
      if (failures.length) notify(`${where} ${deleted.length} 个；未删除：${failures.join('；')}`,true,undo);
      else notify(`${where} ${deleted.length} 个任务。`,false,undo);
    } catch (error) { notify(error.message); }
    finally { state.busy = false; await refreshJobs(true); refreshTrash(); }
  }
  // 「撤销」 on the delete notice: restore exactly the tasks that call moved to the trash; reopen the one that was open.
  async function undoDelete(ids, reopen) {
    if (state.busy || !ids.length) return;
    state.busy = true; const restored = [], failures = [];
    try {
      for (const id of ids) {
        try { await request(`/api/trash/${encodeURIComponent(id)}/restore`,{method:'POST',body:{}}); restored.push(id); }
        catch (error) { failures.push(error.message); }
      }
    } finally { state.busy = false; }
    await refreshJobs(true); refreshTrash();
    if (reopen && restored.includes(reopen)) await selectJob(reopen);   // before the notice: opening a job clears notices
    if (failures.length) notify(`已恢复 ${restored.length} 个；未恢复：${failures.join('；')}`);
    else notify(`已撤销删除，恢复 ${restored.length} 个任务。`,false);
  }
  async function refreshTrash(showError = false) {
    if (!state.authenticated) return;
    const list = $('trash-list'), empty = $('trash-empty'), count = $('trash-count');
    try {
      const result = await request(`/api/trash${state.viewAll ? '?all=1' : ''}`);
      state.trash = Array.isArray(result.items) ? result.items : [];
      count.textContent = state.trash.length ? `· ${state.trash.length}` : '';
      list.replaceChildren(...state.trash.map(trashItem));
      empty.textContent = '回收站是空的。'; empty.hidden = state.trash.length > 0;
    } catch (error) {
      if (error.status === 404) { list.replaceChildren(); count.textContent = ''; empty.textContent = '当前服务版本未提供回收站。'; empty.hidden = false; }
      else if (showError) notify(error.message);
    }
  }
  function trashItem(item) {
    const meta = [item.family && FAMILY_TEXT[item.family], item.release_id && releaseDisplay(item.release_id), item.patient_id && `患者 ${item.patient_id}`, item.scan_label, `删除于 ${item.deleted_at ? friendlyTime(item.deleted_at) : '—'}`, Number.isFinite(Number(item.days_left)) ? `剩余 ${Math.max(0, Math.ceil(Number(item.days_left)))} 天` : null].filter(Boolean).join(' · ');
    const restore = button('恢复',async () => {
      restore.disabled = true;
      try { const result = await request(`/api/trash/${encodeURIComponent(item.id)}/restore`,{method:'POST',body:{}}); notify(`已恢复任务 ${item.case_id || item.id}。`,false); await refreshJobs(true); await refreshTrash(); const job = result.job || result; if (job?.id) await selectJob(job.id); }
      catch (error) { notify(error.message); restore.disabled = false; }
    },'primary');
    const purge = button('彻底删除',async () => {
      if (!await confirmDialog('彻底删除',`彻底删除任务 ${item.case_id || item.id} 及其全部文件？此操作无法恢复。`,{confirmLabel:'彻底删除',danger:true})) return;
      purge.disabled = true;
      try { await request(`/api/trash/${encodeURIComponent(item.id)}/purge`,{method:'POST',body:{}}); notify(`已彻底删除任务 ${item.case_id || item.id}。`,false); await refreshTrash(); }
      catch (error) { notify(error.message); purge.disabled = false; }
    },'danger');
    return node('li',{class:'trash-item'},node('strong',{},item.case_id || '匿名病例',' ',badge(item.status_before)),node('span',{class:'trash-meta',text:meta}),node('div',{class:'trash-actions'},restore,purge));
  }
  $('trash-panel').addEventListener('toggle',() => { if ($('trash-panel').open) refreshTrash(true); });
  $('trash-refresh').addEventListener('click',() => refreshTrash(true));

  // ---------------------------------------------------------------------------------------------
  // Cohort overview (§15.13): distributions and a filter table over every finished job of this owner.
  // Numbers come from `GET /api/jobs/export?format=json`; nothing is recomputed in the browser.
  // ---------------------------------------------------------------------------------------------
  const COHORT_COLUMNS = [
    {key:'case_id',label:'病例'},
    {key:'release_id',label:'发布包'},
    {key:'max_diameter_mm',label:'最大直径 mm',kind:'number',gloss:'max_diameter'},
    {key:'wss_p99_pa',label:'p99 Pa',kind:'number',gloss:'p99'},
    {key:'area_frac_high',label:'高占比 %',kind:'percent',gloss:'area_fraction'},
    {key:'population_percentile',label:'人群分位',kind:'number',gloss:'population_percentile'},
    {key:'review_status',label:'审阅',kind:'review'},
  ];
  const cohortValue = (row,column) => {
    const raw = row[column.key];
    if (column.kind === 'review') return REVIEW_LABEL[raw] || raw || '未审阅';
    if (column.kind === 'percent') return Number.isFinite(Number(raw)) && raw !== null ? fmt(Number(raw) * 100,2) : '—';
    if (column.kind === 'number') return fmt(raw,2);
    return raw === null || raw === undefined || raw === '' ? '—' : String(raw);
  };
  const cohortFilters = () => ({
    family:$('cohort-family').value, release_id:$('cohort-release').value, review_status:$('cohort-review').value,
    p99_min:$('cohort-p99').value, high_min:$('cohort-high').value, very_high_min:$('cohort-very-high').value, speed_min:$('cohort-speed').value,
    diameter_min:$('cohort-diameter').value,
  });
  function drawHistogram(canvas, values, reference) {
    // Two overlaid histograms on one set of bin edges; each is scaled to its own peak so a 20-case
    // selection stays readable next to a 136-case population reference.
    if (!canvas || typeof canvas.getContext !== 'function') return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    // v0.14 (F8): the bitmap follows the displayed width × devicePixelRatio (crisp on HiDPI screens); drawing stays in
    // CSS pixels.  The markup's width / height attributes (560 × 140) only fix the aspect ratio.
    if (!canvas.dataset.baseW) { canvas.dataset.baseW = String(Number(canvas.getAttribute?.('width')) || Number(canvas.width) || 560); canvas.dataset.baseH = String(Number(canvas.getAttribute?.('height')) || Number(canvas.height) || 140); }
    const baseW = Number(canvas.dataset.baseW) || 560, baseH = Number(canvas.dataset.baseH) || 140;
    const width = Math.max(120, Math.round(Number(canvas.clientWidth) || baseW)), height = Math.round(width * baseH / baseW);
    const ratio = Math.min(Math.max(window.devicePixelRatio || 1, 1), 3);
    if (canvas.width !== Math.round(width * ratio)) canvas.width = Math.round(width * ratio);
    if (canvas.height !== Math.round(height * ratio)) canvas.height = Math.round(height * ratio);
    if (typeof ctx.setTransform === 'function') ctx.setTransform(ratio,0,0,ratio,0,0);
    const pad = {left:6, right:6, top:8, bottom:18};
    ctx.clearRect(0,0,width,height);
    const values2 = (values || []).map(Number).filter(Number.isFinite);
    const reference2 = (reference || []).map(Number).filter(Number.isFinite);
    ctx.font = '12px system-ui, sans-serif'; ctx.textBaseline = 'alphabetic';
    if (!values2.length && !reference2.length) { ctx.fillStyle = '#8aa0b2'; ctx.fillText('没有可绘制的数据',pad.left,height / 2); return; }
    const edges = WB.binEdges(values2.concat(reference2),16);
    if (edges.length < 2) return;
    const counts = WB.histogram(values2,edges), refCounts = reference2.length ? WB.histogram(reference2,edges) : null;
    const plotW = width - pad.left - pad.right, plotH = height - pad.top - pad.bottom;
    const barW = plotW / counts.length;
    const peak = Math.max(1,...counts), refPeak = refCounts ? Math.max(1,...refCounts) : 1;
    if (refCounts) {
      ctx.fillStyle = 'rgba(116,150,175,.28)';
      refCounts.forEach((count,index) => { const h = (count / refPeak) * plotH; if (h > 0) ctx.fillRect(pad.left + index * barW, pad.top + plotH - h, Math.max(1,barW - 1), h); });
    }
    ctx.fillStyle = '#176caa';
    counts.forEach((count,index) => { const h = (count / peak) * plotH; if (h > 0) ctx.fillRect(pad.left + index * barW + barW * .18, pad.top + plotH - h, Math.max(1,barW * .64), h); });
    ctx.strokeStyle = '#d3dee7'; ctx.beginPath(); ctx.moveTo(pad.left,pad.top + plotH + .5); ctx.lineTo(pad.left + plotW,pad.top + plotH + .5); ctx.stroke();
    ctx.fillStyle = '#7b8fa1';
    ctx.fillText(fmt(edges[0],2),pad.left,height - 4);
    const maxText = fmt(edges[edges.length - 1],2);
    ctx.fillText(maxText,Math.max(pad.left,width - pad.right - ctx.measureText(maxText).width),height - 4);
  }
  function renderCohort() {
    const data = state.cohort;
    const status = $('cohort-status'), head = $('cohort-head'), body = $('cohort-rows'), count = $('cohort-count');
    if (!data) { head.replaceChildren(); body.replaceChildren(); count.textContent = ''; return; }
    const filters = cohortFilters();
    const filtered = WB.cohortFilter(data.rows,filters);
    const sorted = WB.cohortSort(filtered,state.cohortSort.key,state.cohortSort.direction);
    state.cohortFiltered = sorted;
    count.textContent = `· ${sorted.length} / ${data.rows.length}`;
    const wall = sorted.filter(row => row.family === 'wall'), volume = sorted.filter(row => row.family === 'volume');
    const population = WB.populationValues(data.population,wall.length ? wall : sorted,filters.release_id);
    drawHistogram($('cohort-hist-p99'),wall.map(row => row.wss_p99_pa),population ? population.values : null);
    drawHistogram($('cohort-hist-high'),wall.map(row => Number.isFinite(Number(row.area_frac_high)) && row.area_frac_high !== null ? Number(row.area_frac_high) * 100 : null),null);
    drawHistogram($('cohort-hist-speed'),volume.map(row => row.speed_p99_m_s),null);
    // §17.4: the maximum diameter is filled for both families, so its histogram spans every filtered row.
    const diameters = sorted.map(row => row.max_diameter_mm).filter(value => Number.isFinite(Number(value)) && value !== null);
    drawHistogram($('cohort-hist-diameter'),diameters,null);
    $('cohort-cap-p99').textContent = `壁面 p99 分布 · Pa · ${wall.length} 例${population ? `；浅色为人群参照 ${population.release_id}（${population.case_count} 例，各自按峰值归一化）` : ''}`;
    $('cohort-cap-high').textContent = `高 WSS 面积占比分布 · % · ${wall.length} 例`;
    $('cohort-cap-speed').textContent = `体场速度 p99 分布 · m/s · ${volume.length} 例`;
    $('cohort-cap-diameter').textContent = `最大直径分布 · mm · ${diameters.length} 例${diameters.length ? '' : '（当前结果没有形态数据）'}`;
    head.replaceChildren(...COHORT_COLUMNS.map(column => {
      const active = state.cohortSort.key === column.key;
      const sortButton = button(`${column.label}${active ? (state.cohortSort.direction === 'desc' ? ' ↓' : ' ↑') : ''}`,() => {
        state.cohortSort = active ? {key:column.key,direction:state.cohortSort.direction === 'desc' ? 'asc' : 'desc'} : {key:column.key,direction:column.kind ? 'desc' : 'asc'};
        renderCohort();
      },'sort-button');
      sortButton.setAttribute('aria-label',`按${column.label}排序`);
      return node('th',{scope:'col','aria-sort':active ? (state.cohortSort.direction === 'desc' ? 'descending' : 'ascending') : 'none'},sortButton,column.gloss && gloss(column.gloss));
    }));
    body.replaceChildren(...sorted.slice(0,200).map(row => {
      const tr = node('tr',{class:'cohort-row',tabindex:'0',title:`打开任务 ${row.job_id || ''}`,'aria-current':String(row.job_id === state.current)},...COHORT_COLUMNS.map(column => node('td',{class:column.kind ? 'cohort-num' : '',text:cohortValue(row,column)})));
      if (row.job_id) {
        tr.dataset.jobId = row.job_id;
        tr.addEventListener('click',() => selectJob(row.job_id));
        tr.addEventListener('keydown',event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault?.(); selectJob(row.job_id); } });
      }
      return tr;
    }));
    const empty = sorted.length === 0;
    $('cohort-export-csv').disabled = empty; $('cohort-export-xlsx').disabled = empty;
    status.textContent = empty
      ? (data.rows.length ? '没有符合当前筛选条件的任务。' : '本账户还没有已完成的任务。')
      : `共 ${data.rows.length} 个已完成任务，当前筛选出 ${sorted.length} 个${sorted.length > 200 ? '（表格只显示前 200 行，导出包含全部筛选结果）' : ''}。`;
  }
  async function loadCohort(force = false) {
    if (!state.authenticated || state.cohortBusy) return;
    if (state.cohort && !force) { renderCohort(); return; }
    state.cohortBusy = true;
    $('cohort-status').textContent = '正在读取队列数据…';
    try {
      const result = await request(`/api/jobs/export?format=json&scope=done${state.viewAll ? '&all=1' : ''}`,{timeout:60000});
      const rows = Array.isArray(result.rows) ? result.rows : [];
      state.cohort = {rows, population:result.population || {}, columns:Array.isArray(result.columns) ? result.columns : []};
      const select = $('cohort-release'), previous = select.value;
      select.replaceChildren(node('option',{value:'',text:'全部发布包'}),...WB.cohortReleases(rows).map(id => node('option',{value:id,text:releaseDisplay(id)})));
      if (previous && [...select.options].some(option => option.value === previous)) select.value = previous;
      renderCohort();
    } catch (error) {
      state.cohort = null; $('cohort-head').replaceChildren(); $('cohort-rows').replaceChildren(); $('cohort-count').textContent = '';
      $('cohort-export-csv').disabled = true; $('cohort-export-xlsx').disabled = true;
      $('cohort-status').textContent = error.status === 404 ? '后端未提供队列数据' : `队列数据读取失败：${error.message}`;
    } finally { state.cohortBusy = false; }
  }
  // Move the existing panel, preserving filters, sorting and event handlers.
  const cohortHome = $('cohort-panel').parentElement;
  function closeCohortOverview() {
    if ($('cohort-dialog').open) $('cohort-dialog').close();
  }
  $('cohort-expand').addEventListener('click',() => {
    $('cohort-panel').open = true;
    $('cohort-dialog-body').append($('cohort-panel'));
    $('cohort-dialog').showModal();
    if (state.cohort) renderCohort();   // the histograms follow the wider panel at full resolution
  });
  $('cohort-close').addEventListener('click',closeCohortOverview);
  $('cohort-dialog').addEventListener('click',event => { if (event.target === $('cohort-dialog')) closeCohortOverview(); });
  $('cohort-dialog').addEventListener('close',() => {
    cohortHome.append($('cohort-panel')); if (state.cohort) renderCohort();
    $('cohort-expand').focus();
  });
  $('cohort-panel').addEventListener('toggle',() => { if ($('cohort-panel').open) loadCohort(); });
  $('cohort-refresh').addEventListener('click',() => loadCohort(true));
  for (const id of ['cohort-family','cohort-release','cohort-review']) $(id).addEventListener('change',() => { if (state.cohort) renderCohort(); });
  for (const id of ['cohort-p99','cohort-high','cohort-very-high','cohort-speed','cohort-diameter']) $(id).addEventListener('input',() => { if (state.cohort) renderCohort(); });
  const cohortIds = () => (state.cohortFiltered || []).map(row => row.job_id).filter(Boolean);
  $('cohort-export-csv').addEventListener('click',() => exportSummary(cohortIds(),'csv'));
  $('cohort-export-xlsx').addEventListener('click',() => exportSummary(cohortIds(),'xlsx'));

  // ---------------------------------------------------------------------------------------------
  // Job detail
  // ---------------------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------------------
  // Routing (§19.9): #job=<id> follows the selection; reload and back / forward restore it.
  // ---------------------------------------------------------------------------------------------
  function syncRoute(id, {replace = false} = {}) {
    const target = id ? WB.jobHash(id) : '';
    if ((location.hash || '') === target) return;
    const url = `${location.pathname || '/'}${location.search || ''}${target}`;
    try { if (replace) history.replaceState(null,'',url); else history.pushState(null,'',url); } catch (_) {}
  }
  function onRoute() {
    if (!state.authenticated) return;
    const id = WB.parseJobHash(location.hash);
    if (id && id !== state.current) selectJob(id,{fromRoute:true});
    else if (!id && state.current) showOverview();
  }
  window.addEventListener('popstate',onRoute);
  window.addEventListener('hashchange',onRoute);

  // ---------------------------------------------------------------------------------------------
  // 今日概览 (C1): what needs doing now, what is computing, what waits for review, recent results.
  // ---------------------------------------------------------------------------------------------
  const overviewVisible = () => Boolean($('detail').querySelector?.('.overview-card')) || state.overviewShown === true;
  async function refreshOverview() {
    if (!state.authenticated || state.overviewBusy) return;
    state.overviewBusy = true;
    try {
      const query = new URLSearchParams({page:'1',page_size:'100'}); if (state.viewAll) query.set('all','1');
      const result = await request(`/api/jobs?${query.toString()}`);
      state.overview = {jobs:Array.isArray(result) ? result : result.jobs || [], total:Number(result.total || 0), at:Date.now()};
      if (!state.current && overviewVisible()) renderOverview();
    } catch (_) {} finally { state.overviewBusy = false; }
  }
  async function showOverview() {
    state.viewer?.dispose(); state.viewer = null; closeEvents();
    state.current = null; state.job = null; state.renderKey = null; state.sequence++;
    syncRoute(null,{replace:true}); markCurrent();
    state.overviewShown = true; renderOverview();
    $('detail').setAttribute('aria-busy','false');
    await refreshOverview();
  }
  function renderOverviewStatus() {
    const host = $('overview-status'); if (!host) return;
    const rows = healthRows(state.health);
    const online = $('connection').classList?.contains('online');
    host.replaceChildren(node('h3',{text:'服务状态'}),
      node('p',{class:`overview-connection${online ? ' online' : ''}`,text:$('connection').textContent || '正在连接服务…'}),
      rows.length ? node('dl',{class:'health-list'},...rows.flatMap(([k,v]) => [node('dt',{text:k}),node('dd',{text:v})])) : help(state.healthMissing ? '当前服务版本未提供状态详情。' : '正在读取服务状态…'));
  }
  function renderOverview() {
    const detail = $('detail');
    const jobs = state.overview?.jobs || [];
    const model = WB.overviewModel(jobs);
    const loaded = Boolean(state.overview);
    const signature = JSON.stringify([loaded,model.counts,model.recent.map(job => [job.id,job.version,job.review?.status,job.finished_at || job.created_at,job.case_id]),state.quickFilter,jobs.length === 0,new Date().getHours() < 6 ? 0 : new Date().getHours() < 12 ? 1 : new Date().getHours() < 18 ? 2 : 3]);
    if (signature === state.overviewKey && detail.querySelector?.('.overview-card')) { renderOverviewStatus(); return; }
    state.overviewKey = signature;
    const card = node('section',{class:'card overview-card','aria-labelledby':'overview-title'});
    const hour = new Date().getHours();
    const greeting = hour < 6 ? '夜深了' : hour < 12 ? '上午好' : hour < 18 ? '下午好' : '晚上好';
    card.append(node('div',{class:'overview-head'},node('div',{},node('p',{class:'eyebrow',text:'今日概览'}),node('h1',{id:'overview-title',text:`${greeting}，${model.counts.action ? `有 ${model.counts.action} 例需要处理` : model.counts.pending ? `有 ${model.counts.pending} 例待审阅` : '暂无待办'}`}))));
    const tiles = [
      ['action','需要处理','待确认输入 / 出口、失败或中断'],
      ['active','计算中与排队','完成后会通知'],
      ['pending','待审阅','已完成、尚未签字'],
      ['week','近 7 天完成','按完成时间统计'],
    ].map(([key,label,note]) => {
      const tile = button('',() => setQuickFilter(state.quickFilter === key ? '' : key),`overview-tile tile-${key}`);
      tile.setAttribute('aria-pressed',String(state.quickFilter === key));
      tile.append(node('span',{class:'tile-label',text:label}),node('strong',{class:'tile-value',text:loaded ? String(model.counts[key]) : '…'}),node('span',{class:'tile-note',text:note}));
      tile.title = `在左侧列表中只看「${label}」`;
      return tile;
    });
    card.append(node('div',{class:'overview-tiles'},tiles));
    const recent = node('section',{class:'overview-recent'},node('h3',{text:'最近完成'}));
    if (model.recent.length) {
      recent.append(node('ul',{class:'recent-list'},model.recent.map(job => {
        const row = button('',() => selectJob(job.id),'recent-item');
        row.append(node('span',{class:'recent-name',text:job.case_id || '匿名病例',title:job.case_id || ''}),kindChip(jobKind(job)),job.review?.status === 'reviewed' ? node('span',{class:'badge reviewed',text:'已审阅'}) : node('span',{class:'badge pending',text:'待审阅'}),timeNode(job.finished_at || job.created_at));
        return node('li',{},row);
      })));
    } else recent.append(help(loaded ? '还没有完成的任务。上传 STL 后，结果会出现在这里。' : '正在读取…'));
    // v0.15: 「三步上手」 — open for a new user (no task yet) and until it is closed once; 设置 menu reopens it.
    const guide = node('details',{class:'overview-guide',open:!jobs.length || store.get('wss-guide-closed','') !== '1'},node('summary',{text:'三步上手'}),guideSteps());
    guide.addEventListener('toggle',() => { if (!guide.open) store.set('wss-guide-closed','1'); });
    const side = node('section',{class:'overview-side'},guide,node('div',{id:'overview-status',class:'overview-status'}),
      node('div',{class:'overview-keys'},node('h3',{text:'常用快捷键'}),keyHints(),
        help('也可以把 STL 文件直接拖到页面上新建预测。')));
    card.append(node('div',{class:'overview-grid'},recent,side));
    detail.replaceChildren(card);
    renderOverviewStatus();
  }

  // First-use guide: the whole path in three steps (upload → check inputs and outlets → read, export, sign).
  const GUIDE_STEPS = [
    ['上传 STL','点右上角「＋ 新建预测」或把 STL 拖进页面，选好原始单位和模型发布包，点「上传并检查」。'],
    ['核对单位和出口','系统需要时会停下来请你确认：单位换算后的尺寸，以及三维视图里主动脉入口和四个髂动脉出口的名称。确认后开始计算，列表和详情页显示进度与剩余时间。'],
    ['看结果、导出、签字','算完会提示。打开「三维报告」查看、测量和导出图片，「一页纸」可打印；核对无误后点「审阅签字」锁定结果。'],
  ];
  const guideSteps = () => node('ol',{class:'guide-steps'},GUIDE_STEPS.map(([title,text]) => node('li',{},node('strong',{text:title}),node('span',{text}))));
  const keyHints = () => node('ul',{class:'key-hints'},...[['N','新建预测'],['/','搜索病例'],['J / K','下一例 / 上一例'],['Enter','打开三维报告'],['G','一页纸'],['?','全部快捷键']].map(([k,label]) => node('li',{},node('kbd',{text:k}),node('span',{text:label}))));
  function guideDialog() {
    const all = button('全部快捷键…',() => { closeModal(); state.shortcuts?.showHelp(); });
    openModal('三步上手',[guideSteps(),node('div',{class:'guide-keys'},node('h3',{text:'常用快捷键'}),keyHints()),
      help('也可以把 STL 文件直接拖到页面上新建预测。每个数值旁的「?」是术语说明。')],[all,button('知道了',closeModal,'primary')]);
  }
  async function selectJob(id, {fromRoute = false} = {}) {
    if (!state.authenticated) return;
    closeCohortOverview(); closeActionMenu();
    if (!fromRoute) syncRoute(id);
    else syncRoute(id,{replace:true});
    state.overviewShown = false;
    state.current = id; state.sequence++; state.renderKey = null; clearNotice();
    openEvents(id);
    if (WB.unreadRemove(state.unread, id)) persistUnread();
    markCurrent();
    state.viewer?.dispose(); state.viewer = null;
    $('detail').replaceChildren(node('section',{class:'card loading-card',role:'status'},node('span',{class:'spinner','aria-hidden':'true'}),node('p',{text:'正在读取病例…'})));
    $('detail').setAttribute('aria-busy','true');
    await pollJob(true);
    if (state.authenticated && state.current === id && !uploadDialog?.open && window.matchMedia('(max-width: 760px)').matches) {
      $('detail').focus({preventScroll:true});
      $('detail').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',block:'start'});
    }
  }
  async function pollJob(showError = false) {
    if (!state.current || !state.authenticated || state.polling) return;
    const id = state.current, sequence = state.sequence; state.polling = true;
    try {
      const result = await request(`/api/jobs/${encodeURIComponent(id)}`);
      if (id !== state.current || sequence !== state.sequence) return;
      const job = result.job || result; stampEta([job]); state.job = job;
      const key = `${job.id}/${job.version}/${job.status}`;
      if (key !== state.renderKey) { renderJob(job); state.renderKey = key; }
      else updateLiveStatus(job);
      markOnline();
    } catch (error) {
      // A reply for a job that is no longer open (e.g. just moved to the trash) must not replace the current notice.
      if (id !== state.current || sequence !== state.sequence) return;
      if (showError || error.status === 404) notify(error.message);
      if (id === state.current && sequence === state.sequence && $('detail').querySelector('.loading-card')) $('detail').replaceChildren(node('section',{class:'card'},node('h2',{text:'病例暂时无法读取'}),help(error.message),button('重新读取',() => selectJob(id),'primary')));
      if (state.authenticated) setConnection('连接暂时中断 · 自动重试');
    }
    finally {
      state.polling = false;
      if (state.current && (id !== state.current || sequence !== state.sequence)) pollJob(showError);
      else $('detail').setAttribute('aria-busy','false');
    }
  }
  async function mutate(job, suffix, payload = {}) {
    if (state.busy) return;
    state.busy = true; clearNotice();
    const controls = [...$('detail').querySelectorAll('button,input,select')];
    const previous = controls.map(el => el.disabled); controls.forEach(el => {el.disabled = true;});
    let ok = false;
    try {
      const result = await request(jobUrl(job,suffix),{method:'POST',body:{...payload,version:job.version}});
      const created = result.job || result;
      if (suffix === '/rerun' && created?.id && created.id !== job.id) { state.current = created.id; state.sequence++; state.renderKey = null; syncRoute(created.id); }
      state.renderKey = null; await pollJob(true); await refreshJobs();
      ok = true;
    } catch (error) {
      notify(error.status === 409 ? `任务状态已更新，请核对刷新后的内容再操作。${error.message}` : error.message);
      if (error.status === 409) { state.renderKey = null; await pollJob(true); }
    } finally { state.busy = false; controls.forEach((el,i) => {if (el.isConnected) el.disabled = previous[i];}); }
    return ok;
  }
  // v0.15: cancelling asks first and says how to get the task back (取消后可「重试任务」从可用阶段继续).
  async function cancelJob(job) {
    const running = /^running/.test(job.status || '');
    const message = `${running ? '正在执行的步骤会在安全停止点结束。' : '任务会停止，不再排队或等待确认。'}\n已上传的 STL、中心线和已确认的出口都会保留；之后在详情页点「重试任务」即可从可用的阶段继续，不必重新上传。`;
    if (!await confirmDialog(`取消任务 · ${job.case_id || job.id}`,message,{confirmLabel:'取消任务',cancelLabel:'不取消，继续',danger:true})) return false;
    return mutate(job,'/cancel');
  }
  // After confirming outlets / signing a review, offer the next case of the same kind (A7).
  async function offerNext(kind, doneId) {
    let jobs = state.jobs;
    try {
      const query = new URLSearchParams({page:'1',page_size:'100'}); if (state.viewAll) query.set('all','1');
      const result = await request(`/api/jobs?${query.toString()}`); jobs = Array.isArray(result) ? result : result.jobs || jobs;
    } catch (_) {}
    const next = kind === 'review' ? WB.nextToReview(jobs, doneId) : WB.nextToConfirm(jobs, doneId);
    const done = kind === 'review' ? '已签字通过并锁定。' : '出口已确认，开始预测。';
    if (!next) { notify(`${done}${kind === 'review' ? '没有其他待审阅的任务。' : '没有其他等待确认的任务。'}`,false); return; }
    notifyAction(`${done}还有${kind === 'review' ? '待审阅' : '等待确认'}的任务：${next.case_id || next.id}。`,kind === 'review' ? '下一例待审阅' : '处理下一例',() => selectJob(next.id));
  }
  async function compareWith(job, otherId) {
    try {
      const result = await request('/api/compare',{method:'POST',body:{left_job_id:job.id,right_job_id:otherId}});
      const comparison = result.comparison || result;
      const rows = (comparison.rows || []).map(row => node('tr',{},
        node('td',{text:row.label || row.id}), node('td',{text:fmt(row.left,2)}),
        node('td',{text:fmt(row.right,2)}), node('td',{text:fmt(row.delta,2)})));
      const identityText = (identity, fallbackJob) => {
        if (!identity && !fallbackJob) return '身份信息不可用';
        identity = identity || {};
        const release = identity.release?.id || identity.release?.release || identity.release?.name || fallbackJob?.model_release?.id || '—';
        const caseId = identity.case_id || fallbackJob?.case_id;
        const patientId = identity.patient_id || fallbackJob?.patient_id;
        const scanLabel = identity.scan_label || fallbackJob?.scan_label;
        return [caseId && `病例 ${caseId}`,patientId && `患者 ${patientId}`,scanLabel && `扫描 ${scanLabel}`,`发布包 ${release}`].filter(Boolean).join(' · ');
      };
      const otherJob = state.jobs.find(item => item.id === otherId);
      const card = node('section',{class:'card comparison-card'},
        node('h2',{text:comparison.compatible ? '结果比较' : '结果不可直接比较'}),
        comparison.reasons?.length ? callout(comparison.reasons.join('；'),'error') : help(`${comparison.kind === 'same_input' ? '同一输入' : '跨病例'} · 仅比较声明一致的标量统计，不做点对点差值。`),
        node('div',{class:'comparison-identities'},node('div',{class:'comparison-identity'},node('strong',{text:'左侧'}),node('span',{text:identityText(comparison.left,job)})),node('div',{class:'comparison-identity'},node('strong',{text:'右侧'}),node('span',{text:identityText(comparison.right,otherJob)}))),
        help('差值方向固定为：右侧 − 左侧；WSS 统计单位为 Pa。'),
        node('table',{},node('thead',{},node('tr',{},...['指标','左侧','右侧','差值'].map(text => node('th',{text})))),node('tbody',{},rows)));
      $('detail').append(card); card.scrollIntoView({behavior:'smooth',block:'start'});
    } catch (error) { notify(error.message); }
  }
  function progressIndex(job) {
    if (job.status === 'done') return 4;
    if (job.status === 'awaiting_input' || job.stage === 'input') return 1;
    if (job.status === 'awaiting_confirmation' || job.status === 'awaiting_outlets') return 2;
    if (job.stage === 'B') return 3;
    return getA(job).proposal ? 2 : 1;
  }
  const jobLink = (job, text, path, cls = '') => node('a',{class:`button ${cls}`,href:jobUrl(job,path),target:'_blank',rel:'noopener',text});
  function openOutletOverride(job) {
    const detail = $('detail');
    const existing = detail.querySelector?.('.override-outlet-card');
    if (existing) { existing.scrollIntoView({behavior:'smooth',block:'start'}); return; }
    const editor = outletCard(job,{override:true});
    detail.append(editor);
    editor.scrollIntoView?.({behavior:'smooth',block:'start'});
  }
  function detailMenuItems(job) {
    const done = job.status === 'done';
    return [
      done && {label:'用其他发布包重跑…',run:() => batchRerunDialog([job],'用其他发布包重跑')},
      done && {label:'打包下载 zip',run:() => bundleJobs([job])},
      {label:'编辑信息',run:() => editMetadata(job),disabled:locked(job),title:locked(job) ? '已审阅锁定，先重新打开才能修改' : ''},
      done && getA(job).proposal && {label:'检查 / 修改出口命名并重算',run:() => openOutletOverride(job),disabled:locked(job),title:locked(job) ? '已审阅锁定' : ''},
      done && {label:'查看完整统计 JSON',run:() => window.open(jobUrl(job,'/files/summary.json'),'_blank','noopener')},
      {separator:true},
      {label:'删除任务…',danger:true,disabled:!deletable(job),title:isRunning(job) ? '计算中的任务不能删除' : locked(job) ? '已审阅锁定的任务不能删除' : '',run:() => deleteJobs([job])},
    ];
  }
  // Review sign-off (G4): the existing form, opened from the header's 「审阅签字」 button.
  function reviewChecklist(job) {
    const summary = job.summary || {}, items = [];
    const mapping = [...(job.mapping_history || [])].reverse()[0];
    items.push(['出口命名',mapping?.source === 'automatic_high_confidence' ? `自动确认（置信度 ${fmt(Number(mapping.confidence) * 100,1)}%）` : mapping ? '已人工确认' : '—']);
    const quality = summary.quality;
    if (quality) items.push(['结果质量',quality.label || quality.level || '未评估']);
    const ref = summary.reference_assessment || {};
    items.push(['几何参考范围',ref.status === 'pass' ? '在参考范围内' : ref.status === 'review' ? '有超出范围的测量，请看结果页提示' : '当前发布包未配置']);
    const alert = WB.alertModel(summary);
    if (alert) items.push(['需要注意',alert.title]);
    return node('dl',{class:'review-checklist'},...items.flatMap(([k,v]) => [node('dt',{text:k}),node('dd',{text:v})]));
  }
  function reviewDialog(job) {
    const section = reviewSection(job,{onDone:async decision => { closeModal(); if (decision === 'approve') await offerNext('review',job.id); else notify('已重新打开，可再次修改出口命名或重算。',false); }});
    openModal(`审阅签字 · ${job.case_id || job.id}`,[help('签字前请核对下面几项；签字后结果锁定，报告与一页纸会记录审阅人和时间。'),reviewChecklist(job),section],[button('关闭',closeModal)]);
    section.querySelector?.('input')?.focus();
  }
  const ERROR_TITLES = {input_geometry:'输入几何无法计算',toolchain:'计算工具出错',resource:'计算资源不足',internal:'本次计算未完成'};
  const ERROR_HINTS = {
    input_geometry:'问题出在上传的 STL 本身，重试不会改变结果：请按提示修正几何后重新上传。',
    toolchain:'中心线提取等计算工具没有正常完成。可以稍后重试；若仍失败，请把诊断编号发给维护者。',
    resource:'显存、内存或磁盘暂时不足，通常稍后重试即可完成。',
    internal:'服务内部错误。请把诊断编号发给维护者。'};
  function renderJob(job) {
    state.viewer?.dispose(); state.viewer = null; closeActionMenu();
    const detail = $('detail'); detail.replaceChildren(); sweepThumbs();
    const done = job.status === 'done';
    // v0.14: a finished job shows only its review state (待审阅 / 已审阅 · 已锁定 / 已重新打开); 「已完成」 was redundant.
    const headBadges = node('span',{class:'job-badges'},done ? null : badge(job.eta?.waiting ? 'queued' : job.status));
    if (locked(job)) headBadges.append(node('span',{class:'badge reviewed',text:'已审阅 · 已锁定'}));
    else if (job.review?.status === 'reopened') headBadges.append(node('span',{class:'badge reopened',text:'已重新打开'}));
    else if (done) headBadges.append(node('span',{class:'badge pending',text:'待审阅'}));
    const tags = jobTags(job);
    const identity = node('p',{class:'job-id'},kindChip(jobKind(job)),
      ...[job.patient_id && `患者 ${job.patient_id}`,job.scan_label && `扫描 ${job.scan_label}`,job.scan_date && `扫描日期 ${job.scan_date}`].filter(Boolean).map(text => node('span',{class:'id-part',text})),
      tags.length ? node('span',{class:'job-tags',text:tags.map(tag => `#${tag}`).join(' ')}) : null,
      timeNode(job.created_at,{friendly:false,prefix:'创建 ',cls:'id-part'}),
      node('span',{class:'id-part job-code',title:'任务编号',text:`任务 ${job.id}`}));
    const actions = node('div',{class:'detail-actions'});
    if (done) {
      actions.append(jobLink(job,'打开三维报告','/report','primary'),jobLink(job,'一页纸','/onepage'));
      actions.append(button(locked(job) ? '已审阅 · 查看签字' : '审阅签字',() => reviewDialog(job),`review-button${locked(job) ? ' reviewed' : ''}`));
    }
    // Outlet confirmation needs the whole screen for the vessel: the task card collapses to one title row
    // (title · status · current step · cancel) plus one line of identity and live status.
    const confirming = ['awaiting_confirmation','awaiting_outlets'].includes(job.status) && Boolean(getA(job).proposal) && !job.error;
    if (confirming) {
      headBadges.append(node('span',{class:'step-chip',text:`步骤 ${progressIndex(job) + 1} / 5 · 确认出口`}));
      actions.append(button('取消任务',() => cancelJob(job),'danger'));
    }
    actions.append(moreButton('更多操作',() => detailMenuItems(job)));
    const heading = node('section',{class:`card detail-head${confirming ? ' compact-head' : ''}`},node('div',{class:'job-heading'},node('div',{class:'job-heading-main'},node('h1',{text:job.case_id || '匿名病例'}),headBadges),actions),identity);
    const back = button('← 返回病例列表',() => { const list = $('jobs-title'); list.scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',block:'start'}); $('jobs-query').focus({preventScroll:true}); },'text-button back-to-cases');
    heading.prepend(back);
    const companions = Array.isArray(job.companions) ? job.companions.filter(entry => entry && entry.release_id) : [];
    if (job.companion_of) heading.append(node('p',{class:'job-origin'},`与任务 ${job.companion_of} 同时上传的${releaseFamilyName(job.model_release?.id || '')}预测，复用其中心线与出口确认。 `,button('打开该任务',() => selectJob(job.companion_of),'link-button')));
    if (companions.length) heading.append(node('p',{class:'job-origin'},...companions.flatMap((entry,i) => {
      const name = releaseFamilyName(entry.release_id), sep = i ? '；' : '';
      if (entry.job_id) return [`${sep}同时预测 ${name}：任务 ${entry.job_id} `, button('打开',() => selectJob(entry.job_id),'link-button')];
      return [`${sep}同时预测 ${name}：${entry.error ? `未能创建（${entry.error}）` : '确认出口后自动创建'}`];
    })));
    if (job.companion_of) { /* provenance already shown above */ }
    else if (job.reused_from) heading.append(node('p',{class:'job-origin',text:`复用自任务 ${job.reused_from}：中心线与出口确认沿用，只重新计算所选发布包。`}));
    else if (job.source_job_id) heading.append(node('p',{class:'job-origin',text:`由任务 ${job.source_job_id} 换发布包重跑，复用其中心线与出口确认。`}));
    // While queued / running with an estimate, the stage bar below replaces the coarse five-step stepper.
    const staged = Boolean(job.eta) && ['queued','queued_A','queued_B','running','running_A','running_B'].includes(job.status);
    if (!done && !confirming && !staged) {
      const idx = progressIndex(job);
      heading.append(node('ol',{class:'stepper','aria-label':'预测步骤'},['上传 STL','确认输入','确认出口','预测血流场','查看结果'].map((label,i) => node('li',{class:i < idx ? 'complete' : i === idx ? 'current' : '',text:label,'aria-current':i === idx ? 'step' : null}))));
    }
    if (confirming) {
      identity.append(node('span',{class:'id-part live-part'},node('strong',{id:'job-phase',class:'screen-reader-only'}),node('span',{id:'job-live-detail'})));
      detail.append(heading); updateLiveStatus(job);
      detail.append(outletCard(job));
      if (getA(job).input_check) detail.append(processCard(job));
      return;
    }
    const statusLine = node('div',{class:`status-line${done ? ' compact' : ''}`});
    if (['running','running_A','running_B','queued','queued_A','queued_B','cancelling'].includes(job.status)) statusLine.append(node('span',{class:'spinner','aria-hidden':'true'}));
    statusLine.append(node('p',{},node('strong',{id:'job-phase'}),node('span',{id:'job-live-detail',class:'status-detail'})));
    if (['queued','queued_A','queued_B','running','running_A','running_B','awaiting_input','awaiting_confirmation','awaiting_outlets'].includes(job.status)) statusLine.append(button('取消任务',() => cancelJob(job),'danger'));
    heading.append(statusLine);
    if (job.eta && ['queued','queued_A','queued_B','running','running_A','running_B'].includes(job.status)) heading.append(etaWidget(job));
    detail.append(heading); updateLiveStatus(job);
    // v0.14 (F9): typed error records (wss_deploy/errors.py) — category wording, retryable, admin-only detail.
    const errorRecord = job.error && typeof job.error === 'object' ? job.error : null;
    const noRetry = Boolean(errorRecord && errorRecord.retryable === false);
    if (job.error) {
      const errorCard = node('section',{class:'card error-card'},node('h2',{text:job.status === 'interrupted' ? '任务中断，可恢复' : (ERROR_TITLES[errorRecord?.category] || '本次计算未完成')}),callout(errorText(job.error),'error'));
      const hint = ERROR_HINTS[errorRecord?.category];
      errorCard.append(help(hint || '检查下方输入信息后重试。已有的有效中心线与特征将按服务端校验结果复用。'));
      if (errorRecord?.retry_hint === 'cpu') errorCard.append(help('服务会改用 CPU 重新计算（较慢），无需重新上传。'));
      if (errorRecord?.diagnostic_id) errorCard.append(node('p',{class:'diagnostic',text:`诊断编号：${errorRecord.diagnostic_id}`}));
      // admin_detail reaches the page only for administrators (server.py strips it otherwise).
      if (errorRecord?.admin_detail) errorCard.append(node('details',{class:'admin-detail'},node('summary',{text:'技术细节'}),node('pre',{text:String(errorRecord.admin_detail)})));
      detail.append(errorCard);
    }
    if (['failed','cancelled','interrupted'].includes(job.status)) {
      const recovery = node('section',{class:'card'},node('h2',{text:job.status === 'interrupted' ? '恢复这个任务' : '继续处理'}),help(job.status === 'cancelled' ? '任务已停止。恢复时会重新检查输入和可复用的中间结果。' : noRetry ? '这个错误重试不会得到不同的结果：请修正输入后重新上传，或把诊断编号发给维护者。' : '修正输入或恢复任务后，将从可用的阶段继续。'));
      if (!locked(job) && !noRetry) recovery.append(node('div',{class:'actions'},button(job.status === 'interrupted' ? '恢复任务' : '重试任务',() => mutate(job,'/retry'),'primary'))); detail.append(recovery);
    }
    const a = getA(job);
    if (job.status === 'done') {
      // Completed cases show one lightweight synchronized vessel thumbnail inside
      // the result overview; the full interactive viewer remains in the report.
      detail.append(resultCard(job,narrativeCard(job)));
      if (job.patient_id) detail.append(timelineCard(job));
    }
    else if ((job.status === 'awaiting_confirmation' || job.status === 'awaiting_outlets') && a.proposal) detail.append(outletCard(job));
    else if (job.status === 'awaiting_input' || (['failed','interrupted'].includes(job.status) && a.input_check?.status === 'needs_confirmation')) detail.append(inputCard(job));
    else if (['running','running_A','running_B','queued','queued_A','queued_B'].includes(job.status)) {
      detail.append(node('section',{class:'card'},node('h2',{text:job.stage === 'B' ? '正在生成血流场预测报告' : '正在准备几何与中心线'}),node('p',{class:'muted',text:job.stage === 'B' ? '完成后将显示所选发布包的预测统计和可交互三维报告。' : '几何检查通过后提取中心线，并展示需要核对的入口和四个出口。'}),help('可以切换到其他任务；任务状态保存在服务端。取消计算后，正在执行的步骤会在安全停止点结束。')));
    }
    if (a.input_check && job.status !== 'awaiting_input') detail.append(processCard(job));
  }

  // ---------------------------------------------------------------------------------------------
  // Stage progress and remaining time (§21.2 / §21.3).  Snapshots carry ``eta``; between server updates the
  // running stage keeps advancing on the browser's clock (one shared 1 s ticker).  No ``eta`` (an older
  // service) → nothing is drawn and the page looks as before.
  // ---------------------------------------------------------------------------------------------
  const etaLive = new Set();
  const stampEta = (jobs, at = Date.now()) => { for (const job of jobs || []) if (job && job.eta) job._etaAt = at; return jobs; };
  const etaAge = entry => Math.max(0, (Date.now() - (entry.job?._etaAt || entry.at)) / 1000);
  function tickEta() {
    for (const entry of [...etaLive]) {
      if (entry.el.isConnected === false) { etaLive.delete(entry); continue; }
      try { entry.paint(); } catch (_) { etaLive.delete(entry); }
    }
  }
  function refreshEtaFor(job) {
    for (const entry of etaLive) if (entry.scope === 'detail' && entry.job?.id === job.id) { entry.job = job; entry.paint(); }
  }
  // Detail page: one segment per stage, width ∝ expected seconds; done = solid, current = filled by
  // elapsed / expected (animated), pending = light.  Hover (or the stage table) shows expected / actual seconds.
  function etaWidget(job) {
    const bar = node('div',{class:'eta-bar',role:'progressbar','aria-valuemin':'0','aria-valuemax':'100','aria-label':'计算进度'});
    const headline = node('strong',{class:'eta-headline'}), sub = node('span',{class:'eta-sub'});
    const table = node('tbody');
    const stages = node('details',{class:'eta-stages'},node('summary',{text:'各阶段耗时'}),node('div',{class:'table-wrap'},node('table',{},node('thead',{},node('tr',{},...['阶段','预计','实际 / 已用','状态'].map(text => node('th',{scope:'col',text})))),table)));
    const basis = node('small',{class:'eta-basis'});
    const host = node('div',{class:'eta-block'},node('p',{class:'eta-text'},headline,sub),bar,node('div',{class:'eta-foot'},basis,stages));
    let keys = '';
    const cells = [];
    const entry = {el:host,job,at:Date.now(),scope:'detail',paint() {
      const view = WB.etaView(entry.job?.eta,{ageS:etaAge(entry)});
      host.hidden = !view || !['queued','running'].includes(view.status);
      if (host.hidden) return;
      headline.textContent = view.headline; sub.textContent = view.sub;
      basis.textContent = view.basis;
      const signature = view.segments.map(segment => segment.key).join(',');
      if (signature !== keys) {
        keys = signature; cells.length = 0;
        bar.replaceChildren(...view.segments.map(segment => { const fill = node('span',{class:'eta-fill'}); const cell = node('span',{class:'eta-seg'},fill); cells.push({cell,fill}); return cell; }));
      }
      view.segments.forEach((segment,index) => {
        const {cell,fill} = cells[index];
        cell.className = `eta-seg ${segment.state}${segment.overdue ? ' overdue' : ''}${segment.precomputed ? ' precomputed' : ''}`;
        cell.style.flexBasis = `${segment.width}%`; fill.style.width = `${segment.fill}%`;
        cell.title = segment.title;
      });
      bar.setAttribute('aria-valuenow',String(view.progress));
      bar.setAttribute('aria-valuetext',[view.headline,view.sub].filter(Boolean).join('，'));
      if (stages.open || !table.children.length) {
        table.replaceChildren(...view.segments.map(segment => node('tr',{class:`${segment.state === 'running' ? 'selected' : ''}${segment.precomputed ? ' precomputed' : ''}`},
          node('td',{text:segment.precomputed ? `${segment.label}（已预计算）` : segment.label}),node('td',{class:'num',text:`${fmt(segment.expected,1)} 秒`}),
          node('td',{class:'num',text:segment.elapsed === null || segment.elapsed === undefined ? '—' : `${fmt(segment.elapsed,1)} 秒`}),
          node('td',{text:segment.state === 'done' ? '已完成' : segment.state === 'running' ? (segment.overdue ? '进行中 · 比预计慢' : '进行中') : '等待'}))));
      }
    }};
    stages.addEventListener('toggle',() => entry.paint());
    etaLive.add(entry); entry.paint();
    return host;
  }
  // 「确认后约 N 秒出结果」 next to a confirm button (outlet or input confirmation).
  function etaNote(job, cls = 'eta-note') {
    const el = node('p',{class:cls,role:'status'});
    const entry = {el,job,at:Date.now(),scope:'detail',paint() {
      const view = WB.etaView(entry.job?.eta,{ageS:0});
      const text = view && ['awaiting_confirmation','awaiting_input'].includes(view.status) ? view.headline : '';
      el.textContent = text; el.hidden = !text;
      if (view) el.title = view.basis;
    }};
    etaLive.add(entry); entry.paint();
    return el;
  }
  // List rows: a thin bar while running and a short remaining-time text.
  function etaRow(job) {
    if (!job.eta) return null;
    const text = node('span',{class:'eta-row-text'}), fill = node('span',{class:'eta-row-fill'}), track = node('span',{class:'eta-row-bar','aria-hidden':'true'},fill);
    const el = node('span',{class:'eta-row'},track,text);
    const entry = {el,job,at:Date.now(),scope:'row',paint() {
      const summary = WB.etaRowSummary(entry.job.eta,{ageS:etaAge(entry)});
      el.hidden = !summary; if (!summary) return;
      text.textContent = summary.text;
      track.hidden = summary.pct === null;
      if (summary.pct !== null) fill.style.width = `${summary.pct}%`;
      el.className = `eta-row ${summary.state}`;
    }};
    el._etaEntry = entry;
    etaLive.add(entry); entry.paint();
    return el;
  }
  function updateLiveStatus(job) {
    refreshEtaFor(job);
    // ``eta.waiting`` can flip without a new version: keep the header badge in step (排队中 ↔ 计算中).
    const headBadge = $('detail').querySelector?.('.detail-head .job-badges > .badge');
    if (headBadge && job.status !== 'done') { const shown = job.eta?.waiting ? 'queued' : job.status; headBadge.className = `badge ${Object.hasOwn(STATUS,shown) ? shown : ''}`; headBadge.textContent = STATUS[shown] || shown || '等待'; }
    if (!$('job-phase')) return;
    const phase = typeof job.phase === 'object' ? job.phase?.label || job.phase?.name : job.phase;
    $('job-phase').textContent = job.eta?.waiting ? '排队中 · 等前一个任务算完' : (['running','running_A','running_B'].includes(job.status) && (PHASE[phase] || phase)) || STATUS[job.status] || phase || '等待任务更新';
    const detail = [];
    // The confirmation gate's reasons are rewritten in the outlet panel; a finished job needs no echo.
    const quietDetail = ['done','awaiting_confirmation','awaiting_outlets'].includes(job.status);
    if (job.detail && !quietDetail) detail.push(typeof job.detail === 'string' ? job.detail : job.detail.message || '');
    if (job.status === 'done' && job.updated_at) detail.push(`更新于 ${localTime(job.updated_at)}`);
    if (['queued','queued_A','queued_B'].includes(job.status) && job.queue_position != null && !job.eta) detail.push(`队列位置 ${job.queue_position}`);
    const queued = ['queued','queued_A','queued_B'].includes(job.status), waitingConfirmation = ['awaiting_input','awaiting_confirmation','awaiting_outlets'].includes(job.status);
    const elapsed = queued ? (job.timing?.queue_s ?? job.timing?.queue_seconds) : waitingConfirmation ? (job.timing?.confirmation_s ?? job.timing?.confirmation_seconds) : (typeof job.elapsed === 'number' ? job.elapsed : job.elapsed?.total ?? job.elapsed_s);
    if (elapsed != null && !(job.status === 'done' && !(elapsed > 0))) detail.push(queued || waitingConfirmation ? `已等待 ${duration(elapsed)}` : job.status === 'done' ? `计算 ${timeText(job,elapsed)}` : `已计算 ${timeText(job,elapsed)}`);
    if (job.status === 'awaiting_input') detail.push('请确认原始单位、换算尺寸和几何处理');
    if (job.status === 'awaiting_confirmation' || job.status === 'awaiting_outlets') detail.push('请结合原始影像核对解剖方向');
    $('job-live-detail').textContent = detail.filter(Boolean).join(' · ');
  }
  function boxSize(ic, raw = false) {
    const direct = raw ? ic.raw_bbox_size : ic.bbox_size_mm;
    if (Array.isArray(direct)) return direct;
    const box = raw ? ic.raw_bbox : ic.bbox_mm;
    if (Array.isArray(box) && Array.isArray(box[0])) return box[1].map((v,i) => v - box[0][i]);
    return Array.isArray(box) && box.length === 3 ? box : null;
  }
  const dimensions = size => size?.length === 3 ? size.map(v => fmt(v)).join(' × ') : '—';
  function inputFacts(ic) {
    const removed = Number(ic.removed_area_fraction || 0);
    return node('div',{class:'check-grid'},fact('原始尺寸 · X × Y × Z',dimensions(boxSize(ic,true)),'STL 原坐标值'),fact('换算后尺寸 · mm',dimensions(boxSize(ic)),'请与原始影像的尺度核对'),fact('开口数',ic.openings ?? '—','要求 1 个入口和 4 个出口'),fact('壁面面积',`${fmt(Number(ic.area_mm2)/100)} cm²`,`${ic.vertices?.toLocaleString?.() || '—'} 个顶点`),fact('连通片',ic.components ?? '—','确认没有遗漏血管分支'),fact('拟删除 / 已删除面积',`${fmt(removed * 100,3)}%`,removed > 0 ? '相对于输入壁面总面积' : '无需删除孤立片'));
  }
  function inputCard(job) {
    const ic = getA(job).input_check || {}, params = job.params || {}, summary = job.summary || {};
    const card = node('section',{class:'card'},node('p',{class:'eyebrow',text:'步骤 02'}),node('h2',{text:'确认几何尺寸与处理方式'}));
    card.append(help('为什么要确认：STL 文件不记录单位，自动判断把握不足时需要你确认。单位错一档，血管尺寸就差 10 倍，预测不能用；请对照影像里的血管直径核对换算后的尺寸。'));
    card.append(inputFacts(ic));
    const quality = ic.quality;
    if (quality) {
      const labels = {pass:'通过',pass_with_limits:'通过但有未评估项',review:'需要复核',fail:'阻断',not_checked:'未检查',unknown:'未知'};
      const qualityCard = node('div',{class:'quality-card'},node('div',{class:'quality-head'},node('strong',{text:'输入质量评分卡'}),node('span',{class:`quality-status ${quality.grade || ''}`,text:quality.grade_label || labels[quality.grade] || '未评估'})));
      const grid = node('div',{class:'quality-grid'});
      for (const check of quality.checks || []) {
        const status = check.status || 'not_checked';
        grid.append(node('div',{class:'quality-check'},node('span',{class:`quality-dot ${status}`,'aria-hidden':'true'}),node('span',{},node('strong',{text:check.label || check.key}),node('small',{text:`${labels[status] || status} · ${check.note || ''}`}))));
      }
      qualityCard.append(grid);
      if ((quality.not_evaluated || []).length) qualityCard.append(node('small',{class:'quality-limit',text:`未评估项目：${quality.not_evaluated.join('、')}。质量评分卡不代表预测准确率。`}));
      card.append(qualityCard);
    }
    const reference = summary.reference_assessment || {}, population = reference.population || {};
    if (reference.status || population.status) {
      const geometryText = reference.status === 'pass' ? '在已声明几何参考范围内' : reference.status === 'review' ? '超出已声明范围，请复核' : '未配置几何参考范围';
      const populationText = population.status === 'pass' ? `同协议经验分位 ${fmt(population.percentile,1)}%` : '未配置可核验的人群参照';
      card.append(node('div',{class:'quality-card reference-card'},node('div',{class:'quality-head'},node('strong',{},'参考范围与人群位置',gloss('population_percentile')),node('span',{class:`badge ${reference.status === 'review' ? 'failed' : ''}`,text:reference.status === 'pass' ? '通过' : reference.status === 'review' ? '需复核' : '未配置'})),node('small',{text:`几何：${geometryText} · 人群：${populationText}`})));
    }
    const flags = [...(ic.errors || []),...(ic.flags || [])];
    for (const text of [...new Set(flags)]) card.append(callout(text,ic.status === 'fail' ? 'error' : 'warn'));
    if (ic.status === 'fail') {
      card.append(callout('这份几何未达到计算门限，当前不能通过单位确认继续。请按上面的提示修复原始 STL 后重新上传；如果认为单位判断错误，请保留原文件并重新选择实际单位。','error'));
      card.append(node('div',{class:'actions'},help('服务端不会对超过面积门限的孤立片或不合格拓扑强行预测。')));
      return card;
    }
    const units = node('select',{id:'confirm-units'},...['mm','cm','m'].map(unit => node('option',{value:unit,text:{mm:'毫米 mm',cm:'厘米 cm',m:'米 m'}[unit]})));
    const suggested = Array.isArray(ic.suggested_units) ? ic.suggested_units[0] : ic.suggested_units;
    const defaultUnit = [ic.selected_units,params.units,typeof suggested === 'string' ? suggested : suggested?.unit].find(value => ['mm','cm','m'].includes(value));
    units.value = defaultUnit || 'mm';
    const preview = node('p',{class:'unit-preview','aria-live':'polite'});
    const updateUnit = () => { const size = boxSize(ic,true), scale = {mm:1,cm:10,m:1000}[units.value]; preview.textContent = size ? `确认后：${dimensions(size.map(v => v * scale))} mm（× ${scale}）` : '确认后将重新检查几何尺寸。'; };
    units.addEventListener('change',updateUnit); updateUnit();
    card.append(node('div',{class:'form-grid'},node('div',{},node('label',{htmlFor:'confirm-units',text:'选择 STL 的实际原始单位'}),units,preview),node('div',{},node('label',{text:'尺寸判定'}),help('自动单位建议只用于辅助。毫米、厘米和米的换算将统一到 mm，再执行中心线和预测。'))));
    const fragments = node('input',{type:'checkbox',id:'remove-fragments',checked:params.remove_fragments === true});
    if ((ic.components || 1) > 1) card.append(node('label',{class:'ack',htmlFor:'remove-fragments'},fragments,node('span',{text:`我确认可删除孤立片，保留主要壁面（面积比例 ${fmt(Number(ic.removed_area_fraction || 0)*100,3)}%）。已核对这些片段不属于缺失的血管分支。`})),help('删除比例超过 1% 的壁面需要修复原始几何后重新上传。'));
    const acknowledge = node('input',{type:'checkbox',id:'input-ack'});
    card.append(node('label',{class:'ack',htmlFor:'input-ack'},acknowledge,node('span',{text:'我已核对原始单位和换算后的尺寸，并确认上面的几何处理方式。'})));
    const submit = button('确认输入并检查中心线',() => mutate(job,'/input',{units:units.value,remove_fragments:fragments.checked,acknowledged:true}),'primary');
    submit.disabled = true;
    acknowledge.addEventListener('change',() => {submit.disabled = !acknowledge.checked;});
    units.addEventListener('change',() => {acknowledge.checked = false; submit.disabled = true;});
    fragments.addEventListener('change',() => {acknowledge.checked = false; submit.disabled = true;});
    card.append(node('div',{class:'actions'},help('服务端会再次检查全部输入门限。'),etaNote(job),submit));
    return card;
  }
  function outletCard(job, options = {}) {
    const override = options.override === true;
    const a = getA(job), p = a.proposal;
    if (!a.preview && job.preview) a.preview = job.preview;
    state.mapping = {...(job.mapping || p.mapping)};
    const card = node('section',{class:`card outlet-card${override ? ' override-outlet-card' : ''}`},
      node('div',{class:'outlet-card-head'},node('h2',{},node('span',{class:'eyebrow',text:override ? '人工复核' : '步骤 03'}),override ? '检查或修改出口命名后重算' : '旋转壁面，确认每个出口')));
    // C3: one sentence of status, the reasons in plain words behind 「为什么需要核对」.
    const review = WB.outletReview(p, override ? '' : job.detail);
    const fallbackNote = p.direction_note || getA(job).input_check?.direction_note;
    if (!review.reasons.length && fallbackNote) review.reasons.push(WB.humanOutletReason(fallbackNote));
    const status = node('div',{class:`outlet-status ${override ? (review.required ? 'info' : 'ok') : review.required ? 'warn' : 'ok'}`,role:'status'},
      node('strong',{text:override ? (Number.isFinite(review.confidence) ? `当前结果按${review.required ? '人工确认' : '自动确认'}的命名计算（自动命名置信度 ${Math.round(review.confidence * 100)}%）` : '当前结果按已确认的命名计算') : review.headline}));
    const whyFirst = override ? null : node('p',{class:'why-first',text:'STL 不带患者方向，出口名称决定模型用的左右坐标；名称错了，左右分支的结果会对调。请对照原始影像核对，通常一分钟内完成。'});
    if (review.reasons.length) status.append(node('details',{class:'outlet-why'},node('summary',{text:override ? '命名时的核对提示' : '为什么需要核对'}),whyFirst,node('ul',{},review.reasons.map(text => node('li',{text}))),
      node('p',{class:'help',text:'图中的 X / Y / Z 是 STL 世界坐标，不是患者左 / 右 / 前 / 后方向。左右命名参与模型坐标的构建，改动左右后会按新的坐标重新预测。'})));
    const viewerHost = node('div',{class:'viewer outlet-viewer',id:'outlet-viewer','aria-label':'可旋转的血管壁面和中心线三维预览'});
    const reset = button('复位视角',() => state.viewer?.reset());
    const snapshot = button('保存截图',() => state.viewer?.snapshot());
    const stage = node('div',{class:'outlet-stage'},node('div',{class:'viewer-shell'},node('div',{class:'viewer-toolbar'},node('span',{text:'拖动旋转 · 滚轮缩放 · 右键平移 · 点击彩色端点定位',title:'壁面仅为显示抽稀，计算仍使用完整几何。端点标签颜色与命名同步。'}),node('div',{class:'viewer-buttons'},reset,snapshot)),viewerHost,node('div',{class:'viewer-hint',text:'壁面仅为显示抽稀，计算仍使用完整几何。端点标签颜色与命名同步。'})));
    // v0.15: the general *why* (STL has no patient orientation; the names set left / right) opens the same
    // 「为什么需要核对」 disclosure, so it costs no height on a 1440×900 screen where every outlet row must stay visible.
    if (!review.reasons.length && whyFirst) status.append(node('details',{class:'outlet-why'},node('summary',{text:'为什么需要核对'}),whyFirst));
    const panel = node('aside',{class:'outlet-panel','aria-label':'出口命名与确认'},status);
    const tbody = node('tbody'), rows = new Map(), selects = new Map();
    let selected = null;
    const selectEndpoint = sid => { selected = Number(sid); rows.forEach((row,key) => row.classList.toggle('selected',Number(key) === selected)); state.viewer?.select(selected); };
    for (const e of p.endpoints || []) {
      const sid = String(e.segment_id), dot = node('span',{class:'endpoint-dot'});
      const selectButton = button('',() => selectEndpoint(sid),'endpoint-button');
      selectButton.setAttribute('aria-label',`定位${e.kind === 'inlet' ? '入口' : '出口'} ${sid}`); selectButton.append(dot,`#${sid} ${e.kind === 'inlet' ? '入口' : '出口'}`);
      let name;
      if (e.kind === 'inlet') name = node('span',{text:CN.inlet});
      else {
        name = node('select',{'aria-label':`出口 ${sid} 的解剖名称`},node('option',{value:'',text:'请选择'}),...Object.entries(CN).filter(([key]) => key !== 'inlet').map(([key,label]) => node('option',{value:key,text:label})));
        name.value = state.mapping[sid] || ''; name.addEventListener('change',() => {state.mapping[sid] = name.value; selectEndpoint(sid); updateMapping();}); selects.set(sid,name);
      }
      const coordinates = (e.center_mm || []).map(v => fmt(v)).join(', ');
      const row = node('tr',{title:coordinates ? `世界坐标 ${coordinates} mm` : ''},node('td',{},selectButton),node('td',{class:'num',text:`${fmt(e.radius_mm)} mm`}),node('td',{},name),node('td',{class:'coordinates',text:coordinates}));
      rows.set(sid,row); row.dataset.segmentId = sid; dot.dataset.segmentId = sid; tbody.append(row);
    }
    panel.append(node('div',{class:'table-wrap outlet-table'},node('table',{},node('thead',{},node('tr',{},...['端点','半径','解剖命名','世界坐标 X, Y, Z（mm）'].map((text,i) => node('th',{scope:'col',text,class:i === 3 ? 'coordinates' : ''})))),tbody)));
    const changed = node('p',{class:'mapping-changes','aria-live':'polite'});
    const ack = node('input',{type:'checkbox',id:override ? 'outlet-ack-override' : 'outlet-ack'});
    const confirm = button(override ? '采用修改后的命名并重算' : '确认出口并开始预测',async () => {
      const ok = await mutate(job,'/confirm',{mapping:{...state.mapping},acknowledged:true,...(override ? {override:true,stage:'B'} : {})});
      if (ok && !override) await offerNext('confirm',job.id);
    },'primary');
    function validMapping() { const names = [...selects.keys()].map(id => state.mapping[id]); return names.length === 4 && new Set(names).size === 4 && names.every(name => name && Object.hasOwn(CN,name)); }
    const namingRule = help('四个出口名称各用一次，同一髂总下的两个出口须属于同侧。');
    function updateConfirm() { confirm.disabled = !ack.checked || !validMapping(); namingRule.hidden = validMapping(); }
    function updateMapping() {
      ack.checked = false; updateConfirm();
      selects.forEach((el,sid) => {el.value = state.mapping[sid] || '';});
      const differences = Object.keys(state.mapping).filter(key => state.mapping[key] !== p.mapping?.[key]);
      changed.textContent = differences.length ? `已修改 ${differences.length} 个出口：${differences.map(id => `#${id} → ${CN[state.mapping[id]] || '未命名'}`).join('；')}。确认后将采用当前命名计算。` : (override ? '当前结果使用自动命名建议。你可以修改后重新计算。' : '当前使用自动命名建议，仍需人工核对。');
      for (const e of p.endpoints || []) { const name = e.kind === 'inlet' ? 'inlet' : state.mapping[String(e.segment_id)]; const dot = rows.get(String(e.segment_id))?.querySelector('.endpoint-dot'); if (dot) dot.style.backgroundColor = `#${(COLORS[name] || 0x74889b).toString(16).padStart(6,'0')}`; }
      state.viewer?.update(state.mapping);
    }
    const swap = pairs => { for (const key of Object.keys(state.mapping)) if (pairs[state.mapping[key]]) state.mapping[key] = pairs[state.mapping[key]]; updateMapping(); };
    // The swap buttons, acknowledgement and confirm button stay pinned at the bottom of the side panel
    // (sticky) so they remain on screen even when the panel is taller than the viewport.
    const titled = (text, title, run) => { const b = button(text,run); b.title = title; return b; };
    const foot = node('div',{class:'outlet-panel-foot'},
      node('div',{class:'mapping-actions'},titled('左右互换','左右两侧的出口命名整体互换',() => swap({'out-le':'out-re','out-li':'out-ri','out-re':'out-le','out-ri':'out-li'})),titled('左侧内 / 外','左侧髂内与髂外互换',() => swap({'out-le':'out-li','out-li':'out-le'})),titled('右侧内 / 外','右侧髂内与髂外互换',() => swap({'out-re':'out-ri','out-ri':'out-re'})),titled('恢复建议','恢复自动命名建议',() => {state.mapping = {...p.mapping}; updateMapping();})),changed,
      node('label',{class:'ack',htmlFor:ack.id},ack,node('span',{text:override ? '我已核对修改后的患者方向、主动脉入口和四个出口命名。' : '我已结合原始影像核对患者方向、主动脉入口和四个髂内 / 髂外出口，命名正确。'})),
      node('div',{class:'actions outlet-confirm'},confirm,override ? null : etaNote(job),namingRule));
    ack.addEventListener('change',updateConfirm); updateMapping();
    const openings = a.centerline?.openings || [];
    if (openings.length) {
      const inlet = node('select',{'aria-label':'重新选择入口'},node('option',{value:'',text:'选择需要作为入口的开口'}),...openings.map(o => node('option',{value:String(o.opening_id),text:`开口 ${o.opening_id} · r ${fmt(o.radius_mm)} mm${o.role === 'inlet' ? ' · 当前入口' : ''}`})));
      const recompute = button('重提中心线',() => mutate(job,'/confirm',{mapping:{...state.mapping},inlet:Number(inlet.value),acknowledged:true})); recompute.disabled = true;
      inlet.addEventListener('change',() => {recompute.disabled = inlet.value === '';});
      panel.append(node('details',{class:'inlet-details'},node('summary',{text:'入口选错了？重新选择入口'}),help('入口变更会重新提取中心线。完成后需要再次确认所有出口命名。'),node('div',{class:'inlet-row'},inlet,recompute)));
    }
    panel.append(foot);
    card.append(node('div',{class:'outlet-layout'},stage,panel));
    // Mount after this card is attached, so the renderer receives its real width.  The display mesh
    // and centreline polylines are fetched once from the geometry endpoint instead of riding on every poll.
    requestAnimationFrame(async () => {
      if (!viewerHost.isConnected) return;
      const geometry = await loadGeometry(job);
      if (!viewerHost.isConnected) return;
      const view = {...a, preview: geometry?.preview || a.preview, proposal: {...(a.proposal || {}), preview_polylines: geometry?.preview_polylines?.length ? geometry.preview_polylines : (a.proposal?.preview_polylines || [])}};
      state.viewer = createViewer(viewerHost,view,selectEndpoint);
      if (!state.viewer) {reset.disabled = true; snapshot.disabled = true;} else state.viewer.update(state.mapping);
    });
    return card;
  }
  // Capsule label in the outlet's own colour with white text (C2 / C3): readable on the pale wall and
  // never a white box covering its neighbour.  Returned canvas is sized to the text.
  // Relative luminance of #rrggbb; fills above ~0.35 carry dark text so the label stays readable (WCAG-ish).
  const fillLuminance = hex => { const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || '')); if (!m) return 0; const v = parseInt(m[1],16); return [v >> 16 & 255,v >> 8 & 255,v & 255].map(c => { c /= 255; return c <= .03928 ? c / 12.92 : Math.pow((c + .055) / 1.055,2.4); }).reduce((sum,c,i) => sum + c * [.2126,.7152,.0722][i],0); };
  const textOn = fill => fillLuminance(fill) > .35 ? '#12301f' : '#fff';
  function pillCanvas(text, color, {fontPx = 30, padX = 16, height = 50} = {}) {
    const canvas = document.createElement('canvas'), context = canvas.getContext?.('2d');
    if (!context) return null;
    const font = `600 ${fontPx}px system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif`;
    context.font = font;
    canvas.width = Math.ceil(context.measureText(text).width + padX * 2); canvas.height = height;
    context.font = font;
    const w = canvas.width, h = canvas.height, r = h / 2;
    context.beginPath(); context.moveTo(r,1); context.lineTo(w - r,1); context.arc(w - r,r,r - 1,-Math.PI / 2,Math.PI / 2); context.lineTo(r,h - 1); context.arc(r,r,r - 1,Math.PI / 2,Math.PI * 1.5); context.closePath();
    context.fillStyle = color; context.fill(); context.lineWidth = 2; context.strokeStyle = 'rgba(255,255,255,.9)'; context.stroke();
    context.fillStyle = textOn(color); context.textAlign = 'center'; context.textBaseline = 'middle'; context.fillText(text,w / 2,h / 2 + 1);
    return canvas;
  }
  // v0.14 (F5): viewers draw on demand — a frame when the controls change, while damping still settles (update()
  // returns true), on resize or a scene change — instead of a continuous requestAnimationFrame loop.
  function renderOnDemand(draw, alive) {
    let id = 0;
    const request = () => { if (id || !alive()) return; id = requestAnimationFrame(() => { id = 0; if (alive() && draw()) request(); }) || 0; };
    return {request, cancel() { if (id) cancelAnimationFrame(id); id = 0; }};
  }
  // v0.14 (F3): pinned sprite labels are laid out in screen space with the shared declutterLabels.  Each label keeps
  // its preferred spot (label.userData.home, group-local); a label that would cover another one is moved to the
  // nearest free spot at the same depth.  Runs before every drawn frame, so orbiting re-evaluates the layout.
  function declutterSprites(T, sprites, camera, group, width, height) {
    if (!RC?.declutterLabels || !sprites.length || !(width > 0 && height > 0)) return;
    group.updateMatrixWorld(true); camera.updateMatrixWorld(true);
    const tanHalf = Math.tan(T.MathUtils.degToRad(camera.fov / 2)), forward = new T.Vector3(); camera.getWorldDirection(forward);
    const items = [], metas = [];
    for (const sprite of sprites) {
      const home = sprite.userData.home; if (!home) continue;
      const world = group.localToWorld(home.clone()), ndc = world.clone().project(camera);
      const depth = Math.max(world.clone().sub(camera.position).dot(forward), 1e-6), perPx = 2 * depth * tanHalf / height;
      const w = sprite.scale.x * group.scale.x / perPx, h = sprite.scale.y * group.scale.y / perPx;
      const sx = (ndc.x + 1) / 2 * width, sy = (1 - ndc.y) / 2 * height, cx = sprite.center.x, cy = sprite.center.y;
      items.push({x:sx + (.5 - cx) * w, y:sy - (.5 - cy) * h, w, h, priority:sprite.userData.priority || 0});
      metas.push({sprite, ndcZ:ndc.z, w, h, cx, cy});
    }
    const placed = RC.declutterLabels(items,{padding:2,maxShift:Math.max(40,height * .25),step:6,bounds:{width,height}});
    placed.forEach((spot, index) => {
      const {sprite, ndcZ, w, h, cx, cy} = metas[index];
      if (!spot.moved) { sprite.position.copy(sprite.userData.home); return; }
      const px = spot.x - (.5 - cx) * w, py = spot.y + (.5 - cy) * h;
      const world = new T.Vector3(px / width * 2 - 1, 1 - py / height * 2, ndcZ).unproject(camera);
      sprite.position.copy(group.worldToLocal(world));
    });
  }
  // Detached 3-D thumbnails release their WebGL context on the next sweep (render-on-demand has no loop to notice).
  const liveThumbs = new Set();
  function sweepThumbs() { for (const entry of [...liveThumbs]) if (!entry.canvas.isConnected) { liveThumbs.delete(entry); entry.cleanup(); } }
  function pillSprite(T, text, color, heightWorld, remember = value => value) {
    const canvas = pillCanvas(text,color); if (!canvas) return null;
    const texture = remember(new T.CanvasTexture(canvas));
    if ('colorSpace' in texture && T.SRGBColorSpace) texture.colorSpace = T.SRGBColorSpace;
    const sprite = new T.Sprite(remember(new T.SpriteMaterial({map:texture,transparent:true,depthTest:false,depthWrite:false})));
    sprite.scale.set(heightWorld * canvas.width / canvas.height,heightWorld,1); sprite.renderOrder = 6;
    return sprite;
  }
  const hexColor = value => `#${(value || 0x74889b).toString(16).padStart(6,'0')}`;
  async function loadGeometry(job) {
    const key = `${job.id}/${getA(job).created_at || ''}`;
    if (state.geometry?.key === key) return state.geometry.value;
    try {
      const value = await request(jobUrl(job,'/geometry'),{timeout:60000});
      state.geometry = {key,value};
      return value;
    } catch (error) { notify(`三维预览数据读取失败：${error.message}`); return null; }
  }
  // The result page uses a light-weight, read-only geometry sketch. The full WebGL
  // viewer remains in the outlet confirmation card and in the report; this thumbnail
  // gives a case its visual identity without adding another heavy viewer here. It is
  // fed by the same immutable /geometry payload used by the confirmation viewer.
  function drawVascularThumbnail(canvas, geometry, mapping = {}) {
    const raw = geometry?.preview || {};
    const vertices = Array.isArray(raw.vertices) ? raw.vertices : [];
    const faces = Array.isArray(raw.faces) ? raw.faces : [];
    const lines = Array.isArray(geometry?.preview_polylines) ? geometry.preview_polylines : [];
    const validVertices = vertices.map(point => Array.isArray(point) && point.length >= 3 && point.slice(0,3).every(Number.isFinite) ? point : null);
    const points = validVertices.filter(Boolean);
    const linePoints = lines.flatMap(line => Array.isArray(line?.xyz) ? line.xyz : []).filter(point => Array.isArray(point) && point.length >= 3 && point.slice(0,3).every(Number.isFinite));
    // A few archived stage-A payloads contain only the centreline PCA sketch and
    // omit the display mesh. Keep those usable instead of treating them as empty.
    const sketchPoints = lines.flatMap(line => Array.isArray(line?.xy) ? line.xy.map(point => [point[0], point[1], 0]) : []).filter(point => point.slice(0,2).every(Number.isFinite));
    const all = points.concat(linePoints, sketchPoints);
    if (!all.length) return false;
    const width = Math.max(220, Math.round(canvas.clientWidth || 420)), height = Math.max(130, Math.round(canvas.clientHeight || 190));
    const ratio = Math.min(window.devicePixelRatio || 1, 2); canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    const context = canvas.getContext('2d'); if (!context) return false;
    context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
    /*
     * The stage-A preview already carries a PCA centreline plane (`xy`).  A
     * generic "two largest world axes" projection looks different for every
     * scan and often puts the inlet on the side, hiding the four branches.
     * Build a stable anatomical view from that plane instead: inlet at the
     * top, the branch bundle below it, and left/right inferred from the
     * proposed outlet names.  Older jobs may lack `xy`; those retain the
     * previous world-axis fallback below.
     */
    const validXY = point => Array.isArray(point) && point.length >= 2 && point.slice(0,2).every(Number.isFinite);
    const lineXY = lines.flatMap(line => Array.isArray(line?.xy) ? line.xy : []).filter(validXY);
    // A rerun may have a manually corrected mapping while the immutable stage-A
    // geometry still carries the automatic name. Use the current job mapping for
    // colours and labels, without changing any geometry coordinates.
    const endpointData = (geometry?.endpoints || []).map(e => {
      const auto_name = e.kind === 'inlet' ? 'inlet' : (mapping[String(e.segment_id)] || e.auto_name || e.label || '');
      return {...e, auto_name, name_cn: e.kind === 'inlet' ? (e.name_cn || CN.inlet) : (CN[auto_name] || e.name_cn || '出口')};
    });
    const endpointXY = endpointData.filter(e => validXY(e?.xy));
    let projectionSpace = null;
    const hasAnatomicalPlane = lineXY.length >= 2 && endpointXY.length >= 2;
    let project2d = null;
    const pad = 32;
    if (hasAnatomicalPlane) {
      const inlet = endpointXY.find(e => e.kind === 'inlet') || endpointXY[0];
      const outlets = endpointXY.filter(e => e.kind !== 'inlet');
      const origin = [Number(inlet.xy[0]), Number(inlet.xy[1])];
      const branchMean = outlets.length ? outlets.reduce((sum,e) => [sum[0] + e.xy[0], sum[1] + e.xy[1]], [0,0]).map(v => v / outlets.length) : [origin[0] + 1, origin[1]];
      let vx = branchMean[0] - origin[0], vy = branchMean[1] - origin[1];
      const vnorm = Math.hypot(vx,vy) || 1; vx /= vnorm; vy /= vnorm;
      // h is the left/right screen axis. Its sign is corrected from names so
      // a mirrored PCA never turns left iliac branches into right branches.
      let hx = -vy, hy = vx;
      const q = point => [point[0] - origin[0], point[1] - origin[1]];
      const dotH = e => { const d = q(e.xy); return d[0] * hx + d[1] * hy; };
      const left = outlets.filter(e => /^out-l[ei]$/.test(e.auto_name || e.name || ''));
      const right = outlets.filter(e => /^out-r[ei]$/.test(e.auto_name || e.name || ''));
      if (left.length && right.length) {
        const leftMean = left.reduce((sum,e) => sum + dotH(e), 0) / left.length;
        const rightMean = right.reduce((sum,e) => sum + dotH(e), 0) / right.length;
        if (leftMean > rightMean) { hx = -hx; hy = -hy; }
      }
      const local = point => { const d = q(point); return [d[0] * hx + d[1] * hy, d[0] * vx + d[1] * vy]; };
      const localPoints = lineXY.concat(endpointXY.map(e => e.xy)).map(local);
      const lo = [Math.min(...localPoints.map(p => p[0])), Math.min(...localPoints.map(p => p[1]))];
      const hi = [Math.max(...localPoints.map(p => p[0])), Math.max(...localPoints.map(p => p[1]))];
      const ranges2 = [Math.max(hi[0] - lo[0], 1e-6), Math.max(hi[1] - lo[1], 1e-6)];
      // Stretch the two display axes independently. The card is a non-measurement
      // sketch; filling both axes makes the paired iliac branches readable at a glance.
      const scaleX = (width - 2*pad) / ranges2[0], scaleY = (height - 2*pad) / ranges2[1];
      project2d = point => { const p = local(point); return [pad + (p[0] - lo[0]) * scaleX, pad + (p[1] - lo[1]) * scaleY]; };
      projectionSpace = 'xy';
    }
    // Very old stage-A snapshots may not contain the PCA `xy` values. Recreate the
    // same semantic view directly from xyz so they do not fall back to arbitrary
    // world axes (which can put the inlet on the side and hide branches).
    const validXYZ = point => Array.isArray(point) && point.length >= 3 && point.slice(0,3).every(Number.isFinite);
    const lineXYZ = lines.flatMap(line => Array.isArray(line?.xyz) ? line.xyz : []).filter(validXYZ);
    const endpointXYZ = endpointData.filter(e => validXYZ(e?.center_mm));
    if (!project2d && lineXYZ.length >= 2 && endpointXYZ.length >= 2) {
      const inlet = endpointXYZ.find(e => e.kind === 'inlet') || endpointXYZ[0];
      const outlets = endpointXYZ.filter(e => e.kind !== 'inlet');
      const origin = inlet.center_mm.slice(0,3), mean = outlets.length ? outlets.reduce((sum,e) => sum.map((v,i) => v + e.center_mm[i]), [0,0,0]).map(v => v / outlets.length) : [origin[0],origin[1],origin[2]+1];
      const dot = (a,b) => a.reduce((sum,v,i) => sum + v*b[i],0), norm = a => Math.hypot(...a), sub = (a,b) => a.map((v,i) => v-b[i]), mul = (a,s) => a.map(v => v*s);
      const flow = (() => { const v=sub(mean,origin), n=norm(v); return n > 1e-8 ? mul(v,1/n) : [0,0,1]; })();
      const orth = v => sub(v,mul(flow,dot(v,flow)));
      let horizontal = null, widest = -1;
      for (let i=0;i<outlets.length;i++) for (let j=i+1;j<outlets.length;j++) { const v=orth(sub(outlets[j].center_mm,outlets[i].center_mm)), n=norm(v); if (n>widest) { widest=n; horizontal=v; } }
      if (!horizontal) horizontal = orth([1,0,0]);
      horizontal = (() => { const n=norm(horizontal)||1; return mul(horizontal,1/n); })();
      const left = outlets.filter(e => /^out-l[ei]$/.test(e.auto_name || '')), right = outlets.filter(e => /^out-r[ei]$/.test(e.auto_name || ''));
      if (left.length && right.length) { const avg = group => group.reduce((sum,e) => sum + dot(sub(e.center_mm,origin),horizontal),0) / group.length; if (avg(left) > avg(right)) horizontal=mul(horizontal,-1); }
      const local = point => { const d=sub(point,origin); return [dot(d,horizontal),dot(d,flow)]; };
      const allLocal=lineXYZ.concat(endpointXYZ.map(e=>e.center_mm)).map(local), lo=[Math.min(...allLocal.map(p=>p[0])),Math.min(...allLocal.map(p=>p[1]))], hi=[Math.max(...allLocal.map(p=>p[0])),Math.max(...allLocal.map(p=>p[1]))];
      const rx=Math.max(hi[0]-lo[0],1e-6), ry=Math.max(hi[1]-lo[1],1e-6), sx=(width-2*pad)/rx, sy=(height-2*pad)/ry;
      project2d = point => { const p=local(point); return [pad+(p[0]-lo[0])*sx,pad+(p[1]-lo[1])*sy]; };
      projectionSpace = 'xyz';
    }
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (const point of all) for (let axis = 0; axis < 3; axis++) { min[axis] = Math.min(min[axis], point[axis]); max[axis] = Math.max(max[axis], point[axis]); }
    const ranges = max.map((value, axis) => Math.max(value - min[axis], 1e-6));
    const axes = [0,1,2].sort((left,right) => ranges[right] - ranges[left]).slice(0,2);
    const span = Math.max(ranges[axes[0]], ranges[axes[1]], 1e-6);
    const worldProject = point => [pad + (point[axes[0]] - min[axes[0]]) / span * (width - 2*pad), height - pad - (point[axes[1]] - min[axes[1]]) / span * (height - 2*pad)];
    const project = point => project2d ? project2d(point) : worldProject(point);
    // Fit a tiny affine map from the mesh's 3-D vertices to the centreline
    // plane. This lets the translucent wall silhouette follow the same stable
    // inlet/branch orientation instead of disappearing when `xy` is present.
    let meshProject = worldProject;
    if (project2d && projectionSpace === 'xyz') meshProject = project2d;
    if (project2d && projectionSpace === 'xy') {
      const paired = [];
      for (const line of lines) {
        const xyz = Array.isArray(line?.xyz) ? line.xyz : [], xy = Array.isArray(line?.xy) ? line.xy : [];
        const count = Math.min(xyz.length, xy.length);
        for (let i = 0; i < count; i++) if (Array.isArray(xyz[i]) && xyz[i].length >= 3 && xyz[i].slice(0,3).every(Number.isFinite) && validXY(xy[i])) paired.push([xyz[i].slice(0,3), xy[i].slice(0,2)]);
      }
      if (paired.length >= 4) {
        const normal = Array.from({length:4}, () => Array(4).fill(0)), rhs = [Array(4).fill(0), Array(4).fill(0)];
        for (const [xyz,xy] of paired) { const row = [xyz[0],xyz[1],xyz[2],1]; for (let i = 0; i < 4; i++) { for (let j = 0; j < 4; j++) normal[i][j] += row[i] * row[j]; rhs[0][i] += row[i] * xy[0]; rhs[1][i] += row[i] * xy[1]; } }
        const solve = values => { const a = normal.map((row,i) => row.concat(values[i])); for (let col = 0; col < 4; col++) { let pivot = col; for (let row = col + 1; row < 4; row++) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row; if (Math.abs(a[pivot][col]) < 1e-9) return null; [a[col],a[pivot]] = [a[pivot],a[col]]; const div = a[col][col]; for (let j = col; j <= 4; j++) a[col][j] /= div; for (let row = 0; row < 4; row++) if (row !== col) { const factor = a[row][col]; for (let j = col; j <= 4; j++) a[row][j] -= factor * a[col][j]; } } return a.map(row => row[4]); };
        const ax = solve(rhs[0]), ay = solve(rhs[1]);
        if (ax && ay) meshProject = point => project2d([ax[0]*point[0] + ax[1]*point[1] + ax[2]*point[2] + ax[3], ay[0]*point[0] + ay[1]*point[1] + ay[2]*point[2] + ay[3]]);
      }
    }
    // Preserve original vertex indices because preview faces refer to the STL array.
    const projected = validVertices.map(point => point ? meshProject(point) : null);
    const gradient = context.createLinearGradient(0, 0, width, height); gradient.addColorStop(0, '#f8fbfd'); gradient.addColorStop(1, '#edf5fa');
    context.fillStyle = gradient; context.fillRect(0, 0, width, height);
    // Cap the face pass so a large STL stays responsive while preserving its silhouette.
    const stride = Math.max(1, Math.ceil(faces.length / 20000));
    context.lineJoin = 'round'; context.strokeStyle = 'rgba(63,116,145,.16)'; context.lineWidth = .55;
    for (let index = 0; index < faces.length && (!project2d || meshProject !== worldProject); index += stride) {
      const face = faces[index]; if (!Array.isArray(face) || face.length < 3) continue;
      const a = projected[Number(face[0])], b = projected[Number(face[1])], c = projected[Number(face[2])];
      if (!a || !b || !c) continue;
      context.beginPath(); context.moveTo(a[0],a[1]); context.lineTo(b[0],b[1]); context.lineTo(c[0],c[1]); context.closePath();
      context.fillStyle = project2d ? 'rgba(77,139,168,.08)' : 'rgba(77,139,168,.14)'; context.fill(); context.stroke();
    }
    // Give each branch a distinct colour and resolve shared segments through
    // their descendant leaves. The colour coding makes the four outlets
    // readable even when their centreline paths run close together.
    const endpointById = new Map(endpointData.map(e => [Number(e.segment_id), e]));
    const parentById = new Map(lines.map(line => [Number(line.segment_id), Number(line.parent_id)]));
    const childrenById = new Map();
    for (const line of lines) {
      const sid = Number(line.segment_id), parent = Number(line.parent_id);
      if (!Number.isFinite(sid) || !Number.isFinite(parent) || parent < 0) continue;
      if (!childrenById.has(parent)) childrenById.set(parent, []);
      childrenById.get(parent).push(sid);
    }
    // A common iliac segment has no endpoint of its own, so walking toward its
    // parent cannot identify its side. Resolve the terminal labels below it;
    // when all terminals belong to one side, keep that side colour on the
    // shared segment and switch back to the neutral trunk at the aortic split.
    const segmentEndpoint = sid => {
      const own = endpointById.get(Number(sid)); if (own) return own;
      const labels = [], seen = new Set(), stack = [...(childrenById.get(Number(sid)) || [])];
      while (stack.length && seen.size < 64) {
        const current = Number(stack.pop()); if (seen.has(current)) continue; seen.add(current);
        const endpoint = endpointById.get(current);
        if (endpoint && endpoint.kind !== 'inlet') labels.push(endpoint);
        else for (const child of childrenById.get(current) || []) stack.push(child);
      }
      if (!labels.length) return null;
      const side = new Set(labels.map(label => String(label.auto_name || '').slice(0, 5)).filter(name => name === 'out-l' || name === 'out-r'));
      if (side.size !== 1) return null;
      return {...labels[0], auto_name: side.has('out-l') ? 'out-li' : 'out-ri', name_cn: side.has('out-l') ? '左髂内' : '右髂内'};
    };
    const colour = label => { const key = label?.auto_name || label?.name; const rgb = COLORS[key] || COLORS.outlet; return `#${(rgb || 0x397a9c).toString(16).padStart(6,'0')}`; };
    const linePath = line => {
      // In the anatomical view, do not silently interpret an old 3-D line's
      // X/Y coordinates as the PCA plane; that produces a misleading branch.
      const xy = Array.isArray(line?.xy) && line.xy.length > 1 && line.xy.every(validXY) ? line.xy : null;
      const xyz = Array.isArray(line?.xyz) && line.xyz.length > 1 && line.xyz.every(validXYZ) ? line.xyz : null;
      if (project2d && projectionSpace === 'xy') return xy || [];
      if (project2d && projectionSpace === 'xyz') return xyz || [];
      const fallback = xyz || xy;
      return Array.isArray(fallback) ? fallback : [];
    };
    for (const line of lines) {
      const path = linePath(line); if (path.length < 2) continue;
      const endpoint = segmentEndpoint(line.segment_id);
      context.beginPath(); path.forEach((point,index) => { const p = project(point); if (index) context.lineTo(p[0],p[1]); else context.moveTo(p[0],p[1]); });
      context.strokeStyle = '#ffffff'; context.lineWidth = endpoint ? 4.6 : 3.2; context.stroke();
      context.strokeStyle = endpoint ? colour(endpoint) : '#4d7890'; context.lineWidth = endpoint ? 2.4 : 1.7; context.stroke();
    }
    const occupied = [];
    const labelText = point => point.kind === 'inlet' ? '入口' : (point.name_cn || CN[point.auto_name] || point.auto_name || '出口');
    const putLabel = (point, p, text, fill, preferredSide = 1) => {
      context.font = '600 10px system-ui,sans-serif'; const w = context.measureText(text).width + 10, h = 18;
      const candidates = [[preferredSide * 8,-h-3], [preferredSide * 8,5], [-preferredSide * (w+8),-h-3],[-preferredSide * (w+8),5],[-(w/2),-h-10]];
      let chosen = candidates[0];
      for (const candidate of candidates) { const x = p[0] + candidate[0], y = p[1] + candidate[1]; const box = [x,y,x+w,y+h]; if (box[0] >= 2 && box[1] >= 2 && box[2] <= width-2 && box[3] <= height-2 && !occupied.some(o => !(box[2] < o[0] || box[0] > o[2] || box[3] < o[1] || box[1] > o[3]))) { chosen = candidate; break; } }
      const x = p[0] + chosen[0], y = p[1] + chosen[1]; occupied.push([x,y,x+w,y+h]); context.fillStyle = fill; context.strokeStyle = 'rgba(255,255,255,.9)'; context.lineWidth = 1; context.beginPath(); if (typeof context.roundRect === 'function') context.roundRect(x,y,w,h,h/2); else context.rect(x,y,w,h); context.fill(); context.stroke(); context.fillStyle = textOn(fill); context.textAlign = 'left'; context.textBaseline = 'middle'; context.fillText(text,x+5,y+h/2);
    };
    for (const point of endpointData) {
      if (project2d && projectionSpace === 'xy' && !validXY(point?.xy)) continue;
      if (project2d && projectionSpace === 'xyz' && !validXYZ(point?.center_mm)) continue;
      const q = projectionSpace === 'xy' ? point.xy : (projectionSpace === 'xyz' ? point.center_mm : (validXYZ(point?.center_mm) ? point.center_mm : point.xy)); if (!Array.isArray(q) || q.length < 2) continue;
      const p = project(q); const fill = point.kind === 'inlet' ? '#2569aa' : colour(point); context.beginPath(); context.arc(p[0],p[1],4.2,0,Math.PI*2); context.fillStyle = fill; context.fill(); context.strokeStyle = '#fff'; context.lineWidth = 1.5; context.stroke();
      putLabel(point,p,labelText(point),fill,point.kind === 'inlet' ? 1 : ((point.auto_name || '').startsWith('out-l') ? -1 : 1));
    }
    context.fillStyle = 'rgba(32,48,73,.72)'; context.font = '12px system-ui,sans-serif'; context.textAlign = 'left'; context.textBaseline = 'alphabetic'; context.fillText(project2d ? '入口朝上 · 左右按命名' : '形态预览 · 非测量视图', 10, height - 8);
    return true;
  }
  // A read-only WebGL sketch for completed cases.  The confirmation viewer above is
  // intentionally feature-rich; this version only keeps the wall silhouette, the
  // centreline and four labelled endpoints so a result card stays quick to open.
  // It returns false when WebGL/Three is unavailable and the caller then uses the
  // stable 2-D anatomical sketch above.
  function renderVascularThumbnail3D(canvas, geometry, mapping = {}, host = canvas?.parentElement) {
    const T = window.THREE;
    if (!T || typeof T.WebGLRenderer !== 'function' || typeof T.OrbitControls !== 'function') return false;
    const raw = geometry?.preview || {};
    const vertices = Array.isArray(raw.vertices) ? raw.vertices.filter(p => Array.isArray(p) && p.length >= 3 && p.slice(0,3).every(Number.isFinite)) : [];
    const rawFaces = Array.isArray(raw.faces) ? raw.faces.filter(f => Array.isArray(f) && f.length >= 3 && f.slice(0,3).every(i => Number.isInteger(Number(i)) && Number(i) >= 0 && Number(i) < vertices.length)) : [];
    // v0.15.8: the preview is a connected ≤ 18 000-face surface; keep every face (dropping every other one left holes).
    const faceStride = Math.max(1, Math.ceil(rawFaces.length / 20000));
    const faces = rawFaces.filter((_, index) => index % faceStride === 0);
    const lines = Array.isArray(geometry?.preview_polylines) ? geometry.preview_polylines : [];
    const endpointData = (geometry?.endpoints || []).filter(Boolean).map(e => {
      const name = e.kind === 'inlet' ? 'inlet' : (mapping[String(e.segment_id)] || e.auto_name || e.label || '');
      return {...e, auto_name:name, name_cn:e.kind === 'inlet' ? (e.name_cn || CN.inlet) : (CN[name] || e.name_cn || '出口')};
    });
    const endpoints = endpointData.filter(e => Array.isArray(e.center_mm) && e.center_mm.length >= 3 && e.center_mm.slice(0,3).every(Number.isFinite));
    const validLine = line => Array.isArray(line?.xyz) && line.xyz.length > 1 && line.xyz.every(p => Array.isArray(p) && p.length >= 3 && p.slice(0,3).every(Number.isFinite));
    const usableLines = lines.filter(validLine);
    if (!vertices.length && !usableLines.length && !endpoints.length) return false;
    let renderer, controls, observer, frame, disposed = false;
    const disposables = new Set();
    const disposeItem = item => { try { item?.dispose?.(); } catch (_) {} };
    const remember = item => {if (item) disposables.add(item); return item;};
    try {
      // Measure the canvas itself. The parent also contains the caption, so using
      // its height would distort the camera aspect ratio and squeeze the vessel.
      const width = Math.max(canvas?.clientWidth || host?.clientWidth || 260, 180);
      const height = Math.max(canvas?.clientHeight || 300, 150);
      renderer = new T.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:false});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setClearColor(0xf4f9fc, 1);
      renderer.domElement.className = canvas.className || 'vascular-thumb-canvas';
      renderer.domElement.setAttribute('aria-label','可旋转的血管三维快速预览，点击查看完整报告');
      renderer.domElement.setAttribute('role','img');
      canvas.replaceWith(renderer.domElement);
      const viewCanvas = renderer.domElement;
      const scene = new T.Scene();
      const camera = new T.PerspectiveCamera(28, width / height, .01, 10000);
      const group = new T.Group(); scene.add(group);
      scene.add(new T.HemisphereLight(0xffffff,0x7895a7,1.05));
      const key = new T.DirectionalLight(0xffffff,.72); key.position.set(1.4,1.2,2.5); scene.add(key);
      const fill = new T.DirectionalLight(0xb9e0ed,.3); fill.position.set(-1.5,.4,-1.1); scene.add(fill);

      // Build a patient-facing basis.  The flow axis points from the aortic inlet
      // towards the outlet bundle; the left/right axis is corrected from the
      // anatomical names, so a mirrored PCA/world coordinate never flips labels.
      const v3 = p => new T.Vector3(Number(p[0]),Number(p[1]),Number(p[2]));
      const inlet = endpoints.find(e => e.kind === 'inlet') || endpoints[0];
      const outlets = endpoints.filter(e => e.kind !== 'inlet');
      const origin = inlet ? v3(inlet.center_mm) : (vertices.length ? v3(vertices[0]) : new T.Vector3());
      const branchMean = outlets.length ? outlets.reduce((sum,e) => sum.add(v3(e.center_mm)),new T.Vector3()).multiplyScalar(1 / outlets.length) : origin.clone().add(new T.Vector3(0,1,0));
      const flow = branchMean.clone().sub(origin).normalize(); if (flow.lengthSq() < 1e-8) flow.set(0,1,0);
      const left = outlets.filter(e => /^out-l[ei]$/.test(e.auto_name || ''));
      const right = outlets.filter(e => /^out-r[ei]$/.test(e.auto_name || ''));
      let leftAxis = left.length && right.length
        ? left.reduce((sum,e) => sum.add(v3(e.center_mm)),new T.Vector3()).multiplyScalar(1 / left.length).sub(right.reduce((sum,e) => sum.add(v3(e.center_mm)),new T.Vector3()).multiplyScalar(1 / right.length))
        : (outlets.length > 1 ? v3(outlets[0].center_mm).sub(v3(outlets[outlets.length - 1].center_mm)) : new T.Vector3(1,0,0));
      leftAxis.sub(flow.clone().multiplyScalar(leftAxis.dot(flow)));
      if (leftAxis.lengthSq() < 1e-8) leftAxis.set(1,0,0);
      leftAxis.normalize();
      // local X grows to the patient's right (screen right), while local Y is up.
      const rightAxis = leftAxis.clone().negate();
      const depthAxis = rightAxis.clone().cross(flow).normalize();
      const displayPoint = point => {
        const d = v3(point).sub(origin);
        return new T.Vector3(d.dot(rightAxis), -d.dot(flow), d.dot(depthAxis));
      };

      const allDisplay = [];
      const transformedVertices = vertices.map(point => { const q = displayPoint(point); allDisplay.push(q); return q; });
      const lineDisplay = usableLines.map(line => line.xyz.map(point => { const q = displayPoint(point); allDisplay.push(q); return q; }));
      const endpointDisplay = endpoints.map(e => { const q = displayPoint(e.center_mm); allDisplay.push(q); return q; });
      if (!allDisplay.length) throw new Error('没有可显示的三维坐标');

      if (transformedVertices.length && faces.length) {
        const pos = new Float32Array(transformedVertices.length * 3);
        transformedVertices.forEach((q,i) => pos.set([q.x,q.y,q.z],i*3));
        const index = [];
        // A preview is an identity sketch, not the calculation mesh.  Keep the
        // triangle budget bounded so a high-resolution STL cannot stall the
        // result page or consume a second large WebGL context.
        const faceStride = Math.max(1,Math.ceil(faces.length / 20000));
        for (let fi = 0; fi < faces.length; fi += faceStride) { const face = faces[fi]; index.push(Number(face[0]),Number(face[1]),Number(face[2])); }
        const meshGeometry = remember(new T.BufferGeometry());
        meshGeometry.setAttribute('position',new T.BufferAttribute(pos,3)); meshGeometry.setIndex(index); meshGeometry.computeVertexNormals();
        const material = remember(new T.MeshPhongMaterial({color:0x7faec2,transparent:true,opacity:.22,side:T.DoubleSide,depthWrite:false,shininess:32}));
        const mesh = new T.Mesh(meshGeometry,material); mesh.renderOrder = 0; group.add(mesh);
      }

      const children = new Map(), endpointById = new Map(endpoints.map(e => [Number(e.segment_id),e]));
      for (const line of usableLines) { const sid = Number(line.segment_id), parent = Number(line.parent_id); if (Number.isFinite(sid) && Number.isFinite(parent) && parent >= 0) {if (!children.has(parent)) children.set(parent,[]); children.get(parent).push(sid);} }
      const descendantEndpoint = sid => {
        const direct = endpointById.get(Number(sid)); if (direct) return direct;
        const found = [], seen = new Set(), stack = [...(children.get(Number(sid)) || [])];
        while (stack.length && seen.size < 64) { const child = Number(stack.pop()); if (seen.has(child)) continue; seen.add(child); const e = endpointById.get(child); if (e && e.kind !== 'inlet') found.push(e); else stack.push(...(children.get(child) || [])); }
        if (!found.length) return null;
        const sides = new Set(found.map(e => String(e.auto_name || '').slice(0,5)).filter(v => v === 'out-l' || v === 'out-r'));
        return sides.size === 1 ? found[0] : null;
      };
      const branchColor = endpoint => COLORS[endpoint?.auto_name] || 0x527a91;
      for (let i = 0; i < usableLines.length; i++) {
        const line = usableLines[i], vectors = lineDisplay[i];
        const g = remember(new T.BufferGeometry().setFromPoints(vectors));
        const material = remember(new T.LineBasicMaterial({color:branchColor(descendantEndpoint(line.segment_id)),transparent:true,opacity:.93,depthTest:false}));
        const object = new T.Line(g,material); object.renderOrder = 2; group.add(object);
      }
      const targets = [];
      const spanGuess = Math.max(...allDisplay.map(p => Math.max(Math.abs(p.x),Math.abs(p.y),Math.abs(p.z))),1);
      const endpointRadius = Math.max(spanGuess * .022, .55);
      for (let i = 0; i < endpoints.length; i++) {
        const e = endpoints[i], q = endpointDisplay[i];
        const material = remember(new T.MeshPhongMaterial({color:e.kind === 'inlet' ? COLORS.inlet : branchColor(e),depthTest:false}));
        const sphere = new T.Mesh(remember(new T.SphereGeometry(endpointRadius,12,8)),material); sphere.position.copy(q); sphere.renderOrder = 4; sphere.userData.endpoint = e; group.add(sphere); targets.push(sphere);
      }

      const bounds = new T.Box3().setFromPoints(allDisplay), center = bounds.getCenter(new T.Vector3()), size = bounds.getSize(new T.Vector3());
      const span = Math.max(size.x,size.y,size.z,1);
      // Slightly elongate the vertical anatomical axis in display space; this
      // preserves the corrected inlet-at-top composition without distorting data.
      group.scale.set(1,1.14,1); group.position.set(-center.x,-center.y * 1.14,-center.z);
      // C2: capsule labels in the outlet colour; left-side outlets hang to the left of their endpoint
      // and right-side ones to the right, so the two iliac pairs no longer cover each other.
      const thumbLabels = [];
      for (let i = 0; i < endpoints.length; i++) {
        const endpoint = endpoints[i], q = endpointDisplay[i], key = endpoint.kind === 'inlet' ? 'inlet' : endpoint.auto_name;
        const label = pillSprite(T,endpoint.kind === 'inlet' ? '入口' : (CN[key] || '出口'),hexColor(COLORS[key] || 0x527a91),span * .05,remember); if (!label) continue;
        label.scale.y /= 1.14;   // the group is stretched vertically; keep the capsule's proportions
        // Internal iliacs hang below their endpoint, external ones and the inlet sit beside it (outwards),
        // so the two labels of one side never cover each other or the neighbouring sphere.
        const side = String(key).startsWith('out-l') ? -1 : 1, inner = /^out-[lr]i$/.test(String(key));
        // hang below and inwards (towards the midline), away from the same side's external label
        if (inner) { label.center.set(side < 0 ? 0 : 1,1); label.position.copy(q).add(new T.Vector3(0,-endpointRadius * 1.9,0)); }
        else { label.center.set(side > 0 ? 0 : 1,.5); label.position.copy(q).add(new T.Vector3(side * endpointRadius * 1.9,0,0)); }
        label.userData.home = label.position.clone(); label.userData.priority = endpoint.kind === 'inlet' ? 2 : 1;
        group.add(label); thumbLabels.push(label);
      }
      camera.near = Math.max(span / 5000,.001); camera.far = span * 40;
      controls = new T.OrbitControls(camera,viewCanvas); controls.enableDamping = true; controls.dampingFactor = .08; controls.enablePan = false; controls.minDistance = span*.72; controls.maxDistance = span*7; controls.rotateSpeed = .55; controls.autoRotate = false;
      const fit = () => { const distance = span / (2 * Math.tan(T.MathUtils.degToRad(camera.fov/2))) * 1.08; camera.position.set(distance*.16,distance*.06,distance*1.25); camera.up.set(0,1,0); controls.target.set(0,0,0); controls.update(); camera.updateProjectionMatrix(); };
      const loop = renderOnDemand(() => {
        const moving = controls.update();
        declutterSprites(T,thumbLabels,camera,group,viewCanvas.clientWidth || width,viewCanvas.clientHeight || height);
        renderer.render(scene,camera); return moving;
      },() => !disposed && viewCanvas.isConnected !== false);
      controls.addEventListener?.('change',loop.request);
      const resize = () => { const w = Math.max(viewCanvas.clientWidth || host?.clientWidth || width,1), h = Math.max(viewCanvas.clientHeight || height,1); renderer.setSize(w,h,false); camera.aspect = w / h; camera.updateProjectionMatrix(); loop.request(); };
      resize(); fit();
      if (typeof ResizeObserver === 'function' && host) { observer = new ResizeObserver(resize); observer.observe(host); }
      let moved = false, pointerStart = null;
      const pointerDown = event => {pointerStart = [event.clientX,event.clientY]; moved = false;};
      const pointerMove = event => {if (pointerStart && Math.hypot(event.clientX-pointerStart[0],event.clientY-pointerStart[1]) > 5) {moved = true; controls.autoRotate = false;}};
      viewCanvas.addEventListener('pointerdown',pointerDown); viewCanvas.addEventListener('pointermove',pointerMove);
      const cleanup = () => { if (disposed) return; disposed = true; loop.cancel(); cancelAnimationFrame(frame); observer?.disconnect(); controls?.dispose(); viewCanvas.removeEventListener('pointerdown',pointerDown); viewCanvas.removeEventListener('pointermove',pointerMove); for (const item of disposables) disposeItem(item); renderer?.dispose(); renderer?.forceContextLoss?.(); };
      liveThumbs.add({canvas:viewCanvas,cleanup}); loop.request();
      // The parent button opens the full report on a click.  A drag remains in
      // the thumbnail, so orbiting the vessel does not accidentally navigate.
      viewCanvas.addEventListener('click',event => {if (moved) {event.preventDefault(); event.stopPropagation(); moved = false;}});
      return {cleanup};
    } catch (_) {
      disposed = true; cancelAnimationFrame(frame); observer?.disconnect(); controls?.dispose(); for (const item of disposables) disposeItem(item); renderer?.dispose?.(); renderer?.forceContextLoss?.();
      // If construction failed after replacing the original canvas, restore it
      // so the 2-D renderer can take over without leaving a blank WebGL element.
      if (renderer?.domElement && canvas && !canvas.isConnected && renderer.domElement.isConnected) renderer.domElement.replaceWith(canvas);
      return false;
    }
  }
  function vascularThumbnail(job) {
    const buttonHost = node('button',{type:'button',class:'vascular-thumb',title:'打开三维报告查看完整模型','aria-label':'打开三维报告查看血管三维模型'});
    const canvas = node('canvas',{class:'vascular-thumb-canvas','aria-hidden':'true'});
    const caption = node('span',{class:'vascular-thumb-caption'},node('strong',{text:'血管形态'}),node('small',{text:'点击查看完整三维报告'}));
    buttonHost.append(canvas,caption);
    const draw = async () => {
      try {
        const geometry = await loadGeometry(job);
        if (!buttonHost.isConnected) return;
        // Jobs from before the preview mesh existed (2026-09-17) carry only endpoints: a few dots are not a preview.
        const hasMesh = Boolean(geometry && geometry.preview && geometry.preview.vertices && geometry.preview.vertices.length);
        const lines = geometry && Array.isArray(geometry.preview_polylines) ? geometry.preview_polylines : [];
        const hasXyz = lines.some(line => line && Array.isArray(line.xyz) && line.xyz.length), hasXy = lines.some(line => line && Array.isArray(line.xy) && line.xy.length);
        if (!hasMesh && !hasXyz && !hasXy) throw new Error('no preview geometry');
        // Without a mesh or 3-D centreline the 3-D view would show bare endpoints; the 2-D sketch draws the centreline.
        const three = hasMesh || hasXyz ? renderVascularThumbnail3D(canvas,geometry,job.mapping || {},buttonHost) : null;
        if (three) caption.querySelector('small').textContent = '拖动旋转 · 点击打开报告';
        if (!three && !drawVascularThumbnail(canvas,geometry,job.mapping || {})) { buttonHost.classList.add('is-empty'); caption.replaceChildren(node('strong',{text:'暂无形态预览'}),node('small',{text:'可打开三维报告查看'})); }
      } catch (_) {
        if (buttonHost.isConnected) { buttonHost.classList.add('is-empty'); caption.replaceChildren(node('strong',{text:'暂无形态预览'}),node('small',{text:'可打开三维报告查看'})); }
      }
    };
    requestAnimationFrame(draw);
    buttonHost.addEventListener('click',() => { window.open(jobUrl(job,'/report'),'_blank','noopener'); });
    return buttonHost;
  }
  // -------------------------------------------------------------------------------------------
  // Aneurysm morphology (§17.1) and the automatic narrative (§17.2).  Both are new summary keys:
  // a job computed before them simply renders neither, no branch below may throw on the absence.
  // -------------------------------------------------------------------------------------------
  // Two-line fact: the headline value never shares a line with its companion (C2: "87.0 mm / 225 mL"
  // used to wrap mid-value); the secondary value sits on its own line under it.
  const factPair = (label, main, secondary, note, glossKey) => node('div',{class:'fact fact-pair'},node('span',{class:'fact-label'},label,glossKey && (typeof glossKey === 'string' ? gloss(glossKey) : glossKey)),node('strong',{class:'fact-value',text:main}),secondary ? node('span',{class:'fact-secondary',text:secondary}) : null,note && node('small',{class:'fact-note',text:note}));
  function morphologyBlock(summary) {
    const morphology = summary.morphology;
    if (!morphology || typeof morphology !== 'object') return null;
    const aorta = morphology.aorta || {}, max = aorta.max || {}, sac = aorta.sac || {}, neck = aorta.neck || {};
    const facts = [];
    const largest = Number(max.max_diameter_mm);
    if (Number.isFinite(largest)) {
      const position = Number(max.distance_from_inlet_mm ?? max.s_from_root_mm);
      facts.push(factPair('最大直径',`${fmt(largest,1)} mm`,Number.isFinite(Number(max.equivalent_diameter_mm)) ? `等效 ${fmt(max.equivalent_diameter_mm,1)} mm` : null,Number.isFinite(position) ? `入口下 ${fmt(position,0)} mm` : '位置见三维报告','max_diameter'));
    }
    if (sac.present) facts.push(factPair('瘤体长度 / 体积',`${fmt(sac.length_mm,1)} mm`,`${fmt(sac.volume_ml,0)} mL`,Number.isFinite(Number(sac.threshold_mm)) ? `判定阈值 ${fmt(sac.threshold_mm,1)} mm` : '截面面积沿弧长积分','aneurysm_sac'));
    if (neck.present) facts.push(factPair('瘤颈直径 / 长度',`${fmt(neck.diameter_mean_mm,1)} mm`,`长 ${fmt(neck.length_mm,1)} mm`,'近端瘤颈平均直径与长度','aneurysm_neck'));
    const lumen = Number(morphology.lumen_volume_ml);
    if (Number.isFinite(lumen)) facts.push(factPair('全腔体积',`${fmt(lumen,0)} mL`,null,'开口封盖后的管腔体积','lumen_volume'));
    if (!facts.length) return null;
    const reference = Number(aorta.reference_diameter_mm);
    const block = node('div',{class:'quality-card morphology-card'},
      node('div',{class:'quality-head'},node('strong',{text:'瘤体形态'}),Number.isFinite(reference) ? node('span',{class:'quality-status',text:`参考直径 ${fmt(reference,1)} mm`}) : null),
      node('div',{class:'check-grid morphology-grid'},...facts));
    // §19.5 wording: the per-branch reliability sentences collapse into one line with details on demand.
    const reliability = WB.reliabilitySummary(morphology);
    if (reliability?.text) {
      block.append(reliability.details.length
        ? node('details',{class:'reliability'},node('summary',{text:reliability.text}),node('ul',{class:'compact-list'},reliability.details.map(text => node('li',{text}))))
        : node('p',{class:'reliability-line',text:reliability.text}));
    } else if (reliability?.details.length) block.append(node('small',{text:reliability.details.join('；')}));
    const other = reliability?.other || [];
    block.append(node('small',{text:other.length ? other.join('；') : '直径取中心线每站截面的最大 Feret 直径，由壁面网格求交得到，不依赖预测数值。'}));
    return block;
  }
  async function saveNarrative(job, text, status) {
    // §17.2: an empty string restores the generated sentences; the service answers 409 when the
    // job moved on (or was locked) and its message is what the reviewer needs to read.
    try {
      await request(jobUrl(job,'/narrative'),{method:'PUT',body:{version:job.version,text}});
      closeModal();
      notify(text ? '已保存修改后的结论。' : '已恢复自动生成的结论。',false);
      state.renderKey = null; await pollJob(true); await refreshJobs();
      return true;
    } catch (error) {
      const message = error.status === 404 ? '当前服务版本尚未提供结论修改。' : error.message;
      if (status) status.textContent = message; else notify(message);
      return false;
    }
  }
  function narrativeDialog(job, model) {
    const area = node('textarea',{rows:8,maxlength:4000,'aria-label':'结论文字'}); area.value = model.text || '';
    const status = node('p',{class:'help'});
    const restore = button('恢复自动',async () => { restore.disabled = true; if (!await saveNarrative(job,'',status)) restore.disabled = false; });
    const submit = button('保存',async () => {
      const text = area.value.trim();
      if (!text) { status.textContent = '结论不能为空；要回到自动生成的文字请点「恢复自动」。'; return; }
      submit.disabled = true; status.textContent = '正在保存…';
      if (!await saveNarrative(job,text,status)) submit.disabled = false;
    },'primary');
    openModal(`编辑结论 · ${job.case_id || job.id}`,[
      help('修改后的文字会替代自动结论，并在报告、一页纸与打包文件中标注「审阅人已修改」。最多 4000 字符；请只写可核对的描述，不要写诊断结论。'),
      area, status,
    ],[button('取消',closeModal),restore,submit]);
    area.focus();
  }
  function narrativeCard(job) {
    const model = WB.narrativeModel(job.summary?.narrative);
    if (!model) return null;
    const head = node('div',{class:'quality-head'},node('strong',{},'结论（参考）',gloss('narrative')));
    if (model.edited) head.append(node('span',{class:'quality-status review',text:`审阅人已修改${model.editedBy ? ` · ${model.editedBy}` : ''}`}));
    const card = node('section',{class:'card narrative-card'},head);
    if (model.edited) card.append(node('div',{class:'narrative-text'},...model.edited.split(/\r?\n/).map(line => line.trim()).filter(Boolean).map(line => node('p',{text:line}))));
    else card.append(node('ul',{class:'narrative-list'},model.sentences.map(text => node('li',{text}))));
    const actions = node('div',{class:'actions narrative-actions'});
    if (locked(job)) actions.append(node('span',{class:'badge reviewed',text:'已锁定'}),help('任务已审阅签字并锁定，结论不能再修改；如需修改请先重新打开。'));
    else {
      actions.append(button('编辑结论',() => narrativeDialog(job,model)));
      if (model.edited) actions.append(button('恢复自动',() => saveNarrative(job,'')));
    }
    card.append(actions);
    card.append(help('根据本例的形态与统计量自动生成，供书写报告时参考，不是诊断结论。'));
    return card;
  }
  // Older job records carry no cycle block, top findings or morphology; summary.json has them (§19.2).
  async function loadSummaryExtras(job) {
    const key = `${job.id}/${job.version}`;
    if (state.extras.has(key)) return state.extras.get(key);
    const pending = (async () => {
      try {
        const response = await fetch(jobUrl(job,'/files/summary.json'),{credentials:'same-origin',headers:{Accept:'application/json'}});
        if (!response.ok) return {};
        const full = await response.json(), out = {};
        if (full && typeof full.cycle === 'object' && full.cycle) out.cycle = full.cycle;
        const items = full?.findings?.items;
        if (Array.isArray(items)) out.findings_top = items.slice(0,5).map(item => { const copy = {...item}; delete copy.point_indices; return copy; });
        for (const name of ['morphology','reference_assessment','quality']) if (full && full[name]) out[name] = full[name];
        return out;
      } catch (_) { return {}; }
    })();
    state.extras.set(key,pending);
    if (state.extras.size > 12) state.extras.delete(state.extras.keys().next().value);
    return pending;
  }
  const mergeSummary = (summary, extra) => { const out = {...summary}; for (const [key,value] of Object.entries(extra || {})) if (out[key] === undefined || out[key] === null) out[key] = value; return out; };
  // B4: a yellow banner when the geometry leaves the release's declared range or quality is not good.
  function alertBanner(summary) {
    const model = WB.alertModel(summary); if (!model) return null;
    const more = node('div',{class:'alert-details',hidden:true},node('ul',{},model.items.map(item => node('li',{text:item.text}))),
      help('超出范围表示训练数据里没有见过类似尺寸或形态，预测可能偏离；集成离散度高表示多个模型的结果分歧较大。请结合影像与临床判断。'));
    const toggle = button('查看详情',() => { more.hidden = !more.hidden; toggle.textContent = more.hidden ? '查看详情' : '收起'; toggle.setAttribute('aria-expanded',String(!more.hidden)); },'text-button');
    toggle.setAttribute('aria-expanded','false');
    return node('div',{class:`alert-banner ${model.severity}`,role:'alert'},node('div',{class:'alert-line'},node('span',{class:'alert-icon','aria-hidden':'true',text:'!'}),node('strong',{text:model.title}),toggle),more);
  }
  // M1 (§19.2): TAWSS, OSI and the stagnation zone, straight from summary.cycle.
  function cycleBlock(cycle) {
    const c = WB.cycleSummary(cycle); if (!c) return null;
    const pct = (value, decimals) => WB.percentText(value,decimals);
    const facts = [
      factPair('TAWSS 均值',`${formatValue(c.tawss.mean)} Pa`,c.tawss.lowFrac !== null ? `低 TAWSS（< ${trimValue(c.tawss.lowThreshold)} Pa）占 ${pct(c.tawss.lowFrac)}` : null,c.tawss.p99 !== null ? `p99 ${formatValue(c.tawss.p99)} Pa` : null,glossIf('tawss')),
      factPair('OSI 均值',formatValue(c.osi.mean),c.osi.highFrac !== null ? `OSI > ${trimValue(c.osi.t0)} 占 ${pct(c.osi.highFrac)}` : null,c.osi.veryHighFrac !== null ? `OSI > ${trimValue(c.osi.t2)} 占 ${pct(c.osi.veryHighFrac,1)}` : null,glossIf('osi')),
      factPair('滞留区占比',pct(c.stagnation.frac),c.stagnation.areaCm2 !== null ? `约 ${formatValue(c.stagnation.areaCm2)} cm²` : null,c.stagnation.mainBranch ? `主要位于${c.stagnation.mainBranch}` : null,glossIf('stagnation')),
      // v0.13: RRT / ECAP derived from TAWSS / OSI (Pa⁻¹), when the summary carries them.
      ...[['rrt','RRT 均值'],['ecap','ECAP 均值']].filter(([key]) => c[key]).map(([key,label]) => factPair(label,`${formatValue(c[key].mean)} Pa⁻¹`,
        c[key].highFrac !== null ? `> ${trimValue(c[key].t0)} Pa⁻¹ 占 ${pct(c[key].highFrac)}` : null,c[key].p99 !== null ? `p99 ${formatValue(c[key].p99)} Pa⁻¹` : null,glossIf(key))),
    ];
    return node('div',{class:'quality-card cycle-card'},
      node('div',{class:'quality-head'},node('strong',{text:'周期量'}),node('span',{class:'quality-status',text:`单周期${c.periodS ? ` ${trimValue(c.periodS)} s` : ''} · 直接回归预测`})),
      node('div',{class:'check-grid cycle-grid'},...facts),
      node('small',{text:'滞留区 = TAWSS < 0.4 Pa 且 OSI > 0.1 的壁面；面积为点占比 × 输入壁面面积的估计。' + (c.rrt || c.ecap ? 'RRT = 1/[(1−2·OSI)·TAWSS]、ECAP = OSI/TAWSS，由预测的 TAWSS 与 OSI 逐点计算。' : '')}));
  }
  // G4: the first three findings of the report's list (job.summary.findings_top, §19.2).
  // Array order is the service's (same rule as the one-pager); no re-sorting or kind filtering here.
  function findingsBlock(job, items) {
    const list = WB.findingsForCard(items,3);
    if (!list.length) return null;
    const rows = list.map((item,index) => {
      const chip = WB.findingChip(item);
      return node('li',{class:`finding-row severity-${item.severity || 'note'}`,title:item.definition || ''},
        node('span',{class:'finding-id',text:item.id || `F${index + 1}`}),
        node('span',{class:'finding-main'},node('strong',{text:item.label}),item.branch && !String(item.label).includes(item.branch) ? node('small',{text:item.branch}) : null),
        node('span',{class:'finding-value',text:`${formatValue(item.value)}${WB.findingUnits(item)}`}),
        node('span',{class:`finding-severity ${chip.tone}`,text:chip.text}));
    });
    return node('div',{class:'quality-card findings-card'},node('div',{class:'quality-head'},node('strong',{text:'重点发现'}),jobLink(job,'在三维报告中查看全部','/report','text-link')),node('ol',{class:'findings-list'},rows));
  }
  function resultCard(job,narrative = null) {
    const summary = job.summary || {}, peak = summary.peak || {}, times = summary.timing_s || {}, stats = summary.volume_statistics || {};
    const kind = jobKind(job), volume = kind === 'volume' || Boolean(summary.fields?.velocity);
    const heading = node('div',{class:'result-head'},node('h2',{text:volume ? '压力与速度体场' : kind === 'cycle' ? '峰值 WSS · TAWSS · OSI' : '峰值 WSS'}));
    const chip = diameterChip(maxDiameterMm(summary)); if (chip) heading.append(chip);
    const alertHost = node('div',{class:'alert-host'});
    const card = node('section',{class:'card result-card'},alertHost,node('p',{class:'eyebrow',text:'预测结果'}),heading);
    const link = (text,path,cls = '') => jobLink(job,text,path,cls);
    const automatic = [...(job.mapping_history || [])].reverse().find(item => item.source === 'automatic_high_confidence');
    if (automatic) card.append(callout(`出口命名已自动确认（最低置信度 ${fmt(Number(automatic.confidence) * 100,1)}%）。仍可在「⋯」菜单里人工复核并重算。`,'success'));
    const overviewMain = node('div',{class:'result-overview-main'});
    const overview = node('div',{class:'result-overview'},vascularThumbnail(job),overviewMain);
    let areaFractions = null;
    // Compute time: the total is the headline; a retried job's split goes on the note line (§15.5).
    const timing = WB.computeTiming(job.timing, times.total);
    const precomputeS = Number(job.timing?.precompute_s ?? times.precompute), precomputeNote = Number.isFinite(precomputeS) && precomputeS > 0 ? `；另有确认出口期间后台预计算 ${duration(precomputeS)}` : '';
    const timingFact = () => fact('计算耗时',duration(timing.total ?? times.total),(timing.split ? `本次 ${duration(timing.attempt)} · 累计 ${duration(timing.total)}，不含排队与人工确认` : '不包含排队与人工确认') + precomputeNote);
    if (volume) overviewMain.append(node('div',{class:'result-cards'},fact('体内速度 p99',`${fmt(stats.speed_m_s?.p99,3)} m/s`,'内部采样点速度模长','speed'),
      factPair('体内相对压力',`最高 ${fmt(stats.pressure_interior_pa?.max,0)} Pa`,`最低 ${fmt(stats.pressure_interior_pa?.min,0)} Pa`,'相对于该帧体积平均压力',gloss('relative_pressure')),timingFact()));
    else {
      overviewMain.append(node('div',{class:'result-cards'},fact('全场 p99',`${fmt(peak.p99_pa,2)} Pa`,'收缩期峰值帧，预测点云第 99 百分位','p99'),fact('全场最大值',`${fmt(peak.max_pa,2)} Pa`,peak.branch ? `最大值位置：${peak.branch}` : '位置见三维报告','max'),timingFact()));
      const field = summary.wss_field_pa || {};
      if (Number.isFinite(Number(field.area_frac_low)) || Number.isFinite(Number(field.area_frac_high))) {
        const t = Array.isArray(field.thresholds_pa) && field.thresholds_pa.length === 3 ? field.thresholds_pa : [0.4,4,7];
        areaFractions = node('div',{class:'result-fractions'},node('span',{},'峰值 WSS 面积占比（估计）',gloss('area_fraction')),node('span',{},`低 WSS < ${t[0]} Pa `,node('strong',{text:`${fmt(Number(field.area_frac_low) * 100,1)}%`})),node('span',{},`高 WSS > ${t[1]} Pa `,node('strong',{text:`${fmt(Number(field.area_frac_high) * 100,1)}%`})),node('span',{},`极高 > ${t[2]} Pa `,node('strong',{text:`${fmt(Number(field.area_frac_very_high) * 100,1)}%`})),node('span',{},'阈值',gloss('thresholds')));
      }
    }
    card.append(overview);
    const cycleHost = node('div',{class:'cycle-host'}), findingsHost = node('div',{class:'findings-host'}), morphologyHost = node('div',{class:'morphology-host'});
    // v0.14 (F3): the conclusion sits right under the headline numbers, on the first screen (it used to be inside
    // the collapsed 「更多结果、复核与操作」).
    if (narrative) overviewMain.append(narrative);
    overviewMain.append(cycleHost,findingsHost,morphologyHost);
    const fill = merged => {
      const banner = alertBanner(merged); alertHost.replaceChildren(...(banner ? [banner] : []));
      const cycle = cycleBlock(merged.cycle); cycleHost.replaceChildren(...(cycle ? [cycle] : []));
      const findings = findingsBlock(job,merged.findings_top); findingsHost.replaceChildren(...(findings ? [findings] : []));
      const morphology = morphologyBlock(merged); morphologyHost.replaceChildren(...(morphology ? [morphology] : []));
      if (!heading.querySelector?.('.diameter-chip') && !chip) { const late = diameterChip(maxDiameterMm(merged)); if (late) heading.append(late); }
    };
    fill(summary);
    // A record written before §19.2 lacks these blocks; read them once from summary.json.
    const missing = (kind === 'cycle' && !summary.cycle) || !Array.isArray(summary.findings_top) || !summary.morphology;
    if (missing) loadSummaryExtras(job).then(extra => { if (card.isConnected !== false && state.current === job.id && Object.keys(extra || {}).length) fill(mergeSummary(summary,extra)); });
    // Secondary review material stays available, but starts collapsed so the first screen keeps
    // the core metrics, morphology and report actions in view.
    const reviewLabel = locked(job) ? '已审阅' : job.review?.status === 'reopened' ? '已重新打开' : '待审阅';
    const supplemental = node('details',{class:'result-more'},node('summary',{},'更多结果、复核与操作',node('span',{class:'result-more-hint'},` · ${reviewLabel}`)));
    if (areaFractions) supplemental.append(areaFractions);
    const quality = summary.quality;
    if (quality) {
      const qualityCard = node('div',{class:'quality-card result-quality'},
        node('div',{class:'quality-head'},node('strong',{},'集成质量与复核提示',gloss('quality_grade')),node('span',{class:`quality-status ${quality.level || ''}`,text:quality.label || quality.level || '未评估'})));
      qualityCard.append(node('small',{text:WB.qualityNote(quality,WB.modelCount(summary,job))}));
      supplemental.append(qualityCard);
    }
    const reference = summary.reference_assessment || {}, population = reference.population || {};
    if (population.status || reference.status) {
      const populationText = population.status === 'pass' ? `折外人群分位 ${fmt(population.percentile,1)}%` : '未配置可核验的人群参照';
      const geometryText = reference.status === 'pass' ? '几何在参考范围内' : reference.status === 'review' ? '几何超出参考范围，请复核' : '未配置几何参考范围';
      supplemental.append(node('div',{class:'quality-card reference-card'},node('div',{class:'quality-head'},node('strong',{},'人群位置与几何范围',gloss('population_percentile')),node('span',{class:`badge ${reference.status === 'review' ? 'failed' : ''}`,text:reference.status === 'pass' ? '通过' : reference.status === 'review' ? '需复核' : '未配置'})),node('small',{text:`${populationText} · ${geometryText}`})));
    }
    const surface = summary.surface_statistics || summary.results?.statistics?.surface;
    if (surface && Number.isFinite(Number(surface.p99_pa))) {
      const coverage = Number(surface.covered_area_fraction);
      supplemental.append(node('div',{class:'quality-card surface-card'},node('div',{class:'quality-head'},node('strong',{},'壁面面积加权参考',gloss('area_weighted_p99')),node('span',{class:'badge',text:'补充口径'})),
        node('div',{class:'kv'},node('span',{class:'fact-label',text:'面积加权 p99'}),node('strong',{text:`${fmt(surface.p99_pa,2)} Pa`}),node('span',{class:'fact-label',text:'有效覆盖面积'}),node('strong',{text:Number.isFinite(coverage) ? `${fmt(coverage * 100,1)}%` : '—'})),
        node('small',{text:'主指标仍为预测点云 p99；面积口径仅统计 Gaussian 插值后完整有效三角面。'})));
    }
    supplemental.append(reviewStatusCard(job));
    overviewMain.append(supplemental);
    const actions = volume ? [link('下载体场 VTP','/files/volume_fields.vtp'),link('下载壁面压力 VTP','/files/wall_pressure.vtp'),link('下载体场 CSV','/files/points_volume.csv')] : [link('下载壁面 VTP','/files/wall_wss.vtp'),link('下载点云 CSV','/files/points_wss.csv')];
    if (volume && summary.exports?.streamlines) actions.push(link('下载流线 VTP','/files/streamlines.vtp'));
    actions.push(button('打包下载 zip',() => bundleJobs([job])));
    const boundRelease = summary.model_release?.registry_id || summary.model_release?.name || summary.model_release?.release || summary.release;
    const alternatives = state.releases.filter(item => (item.id || item.release) !== boundRelease);
    if (alternatives.length) {
      const rerunSelect = node('select',{class:'inline-select','aria-label':'选择重跑发布包'},...alternatives.map(item => node('option',{value:item.id || item.release,text:releaseDisplay(item)})));
      actions.push(rerunSelect,button('用发布包重跑',() => mutate(job,'/rerun',{release_id:rerunSelect.value})));
    }
    const family = volume ? 'volume' : 'wall';
    const comparable = state.jobs.filter(item => item.id !== job.id && item.status === 'done' && (item.family ? item.family === family : true));
    if (comparable.length) {
      const compareSelect = node('select',{class:'inline-select','aria-label':'选择比较任务'},...comparable.map(item => node('option',{value:item.id,text:`${item.case_id || '匿名病例'} · ${item.scan_label || friendlyTime(item.created_at) || item.id}`})));
      actions.push(compareSelect);
      if (!volume) actions.push(button('比较结果',() => compareWith(job,compareSelect.value)));
      actions.push(button('并排比较',() => { const params = new URLSearchParams({left:job.id,right:compareSelect.value}); window.open(`/compare?${params.toString()}`,'_blank','noopener'); }));
    }
    if (getA(job).proposal && !locked(job)) actions.push(button('检查 / 修改出口命名并重算',() => openOutletOverride(job)));
    supplemental.append(node('div',{class:'actions'},actions));
    supplemental.append(help(volume ? '速度查看体内点云、稳态流线及截面；压力可切换壁面和内部。截面是有限厚度采样切片，流线来自固定时间帧，压力不是绝对血压。' : kind === 'cycle' ? '报告可在峰值 WSS、周期平均 TAWSS 与 OSI 之间切换着色，查看热点、滞留区和分支统计。p99 是统计量，不对应单一解剖位置。' : '报告可旋转查看整段壁面、查看最大值位置和分支统计，并导出当前视角截图。p99 是统计量，不对应单一解剖位置。'));
    const fileLinks = [node('a',{href:jobUrl(job,'/files/summary.json'),target:'_blank',rel:'noopener',text:'查看完整统计与参数 JSON'})];
    if (summary.run_manifest) fileLinks.push(node('a',{href:jobUrl(job,'/files/run_manifest.json'),target:'_blank',rel:'noopener',text:'查看可复现运行清单'}));
    supplemental.append(node('div',{class:'file-links'},fileLinks));
    return card;
  }
  // Compact review state inside the collapsed section; the form itself opens from the header.
  function reviewStatusCard(job) {
    const review = job.review || {status:'unreviewed'};
    const text = review.status === 'reviewed' ? `${review.by || '—'} 于 ${review.at ? localTime(review.at) : '—'} 签字通过${review.note ? `：${review.note}` : ''}。结果已锁定。`
      : review.status === 'reopened' ? `${review.by || '—'} 于 ${review.at ? localTime(review.at) : '—'} 重新打开${review.note ? `：${review.note}` : ''}。`
      : '尚未签字。签字后结果锁定：不能重算、修改出口命名或删除。';
    return node('div',{class:`quality-card review-card ${review.status || 'unreviewed'}`},
      node('div',{class:'quality-head'},node('strong',{},'审阅签字',gloss('review_status')),node('span',{class:`quality-status review-${review.status || 'unreviewed'}`,text:REVIEW_LABEL[review.status] || '未审阅'})),
      node('small',{text}),node('div',{class:'actions'},button(review.status === 'reviewed' ? '查看签字 / 重新打开' : '审阅签字',() => reviewDialog(job),review.status === 'reviewed' ? '' : 'primary')));
  }
  function reviewSection(job, {onDone} = {}) {
    const review = job.review || {status:'unreviewed'};
    const wrap = node('div',{class:`quality-card review-card ${review.status || 'unreviewed'}`});
    const head = node('div',{class:'quality-head'},node('strong',{},'审阅签字',gloss('review_status')),node('span',{class:`quality-status review-${review.status || 'unreviewed'}`,text:REVIEW_LABEL[review.status] || '未审阅'}));
    wrap.append(head);
    if (review.status === 'reviewed') {
      wrap.append(node('small',{text:`${review.by || '—'} 于 ${review.at ? localTime(review.at) : '—'} 签字通过${review.note ? `：${review.note}` : ''}。结果已锁定：不能重算、修改出口命名或删除；换发布包重跑会新建任务。`}));
      const reviewer = node('input',{type:'text',maxlength:80,placeholder:'重新打开人的匿名标识','aria-label':'重新打开人'});
      const reason = node('textarea',{rows:2,maxlength:2000,placeholder:'重新打开的原因（必填）','aria-label':'重新打开原因'});
      const reopen = button('重新打开',async () => { if (await mutate(job,'/review',{decision:'reopen',reviewer:reviewer.value.trim(),note:reason.value.trim()})) onDone?.('reopen'); },'danger');
      reopen.disabled = true;
      const check = () => { reopen.disabled = !reviewer.value.trim() || !reason.value.trim(); };
      reviewer.addEventListener('input',check); reason.addEventListener('input',check);
      wrap.append(node('div',{class:'review-form'},reviewer,reason,node('div',{class:'actions'},help('重新打开后可再次修改出口命名或重算；原签字记录保留在审计中。'),reopen)));
    } else {
      if (review.status === 'reopened') wrap.append(node('small',{text:`${review.by || '—'} 于 ${review.at ? localTime(review.at) : '—'} 重新打开${review.note ? `：${review.note}` : ''}。`}));
      else wrap.append(node('small',{text:'签字后结果将被锁定：不能重算、修改出口命名或删除，报告与运行清单会记录审阅人和时间。'}));
      const reviewer = node('input',{type:'text',maxlength:80,placeholder:'审阅人匿名标识（必填）','aria-label':'审阅人',autocomplete:'off'});
      if (state.session?.username && !reviewer.value) reviewer.value = state.session.display_name || state.session.username;
      const note = node('textarea',{rows:2,maxlength:2000,placeholder:'审阅备注（可选）','aria-label':'审阅备注'});
      const ack = node('input',{type:'checkbox',id:`review-ack-${job.id}`});
      const approve = button('签字通过并锁定',async () => { if (await mutate(job,'/review',{decision:'approve',reviewer:reviewer.value.trim(),note:note.value.trim()})) onDone?.('approve'); },'primary');
      approve.disabled = true;
      const check = () => { approve.disabled = !(ack.checked && reviewer.value.trim()); };
      reviewer.addEventListener('input',check); ack.addEventListener('change',check);
      wrap.append(node('div',{class:'review-form'},reviewer,note,node('label',{class:'ack',htmlFor:`review-ack-${job.id}`},ack,node('span',{text:'我已核对出口命名、输入检查与结果质量提示，确认该结果可以签字。'})),node('div',{class:'actions'},approve)));
    }
    return wrap;
  }
  // ---------------------------------------------------------------------------------------------
  // Patient follow-up timeline (B2, contract §19.3).  One scan = one input geometry; geometry is
  // comparable across releases, model numbers only within one release (one line per release).
  // ---------------------------------------------------------------------------------------------
  async function fetchTimeline(patientId) {
    const response = await fetch(`/api/patients/${encodeURIComponent(patientId)}/timeline${state.viewAll ? '?all=1' : ''}`,{credentials:'same-origin',headers:{Accept:'application/json'}});
    if (!response.ok) { const error = new Error(`HTTP ${response.status}`); error.status = response.status; throw error; }
    return response.json();
  }
  const SVG_NS = 'http://www.w3.org/2000/svg';
  function svg(tag, attrs = {}, ...children) {
    const el = document.createElementNS(SVG_NS,tag);
    for (const [key,value] of Object.entries(attrs)) if (value !== undefined && value !== null) el.setAttribute(key,String(value));
    for (const child of children.flat()) if (child !== null && child !== undefined && child !== false) el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    return el;
  }
  // One small chart per quantity (one y-axis each); the table below is the exact-value view.
  function sparkChart(chart) {
    if (typeof document.createElementNS !== 'function') return null;
    const geo = WB.sparkGeometry(chart.lines,{width:300,height:124,pad:{left:12,right:12,top:20,bottom:22}});
    if (!geo) return null;
    const root = svg('svg',{viewBox:`0 0 ${geo.width} ${geo.height}`,class:'spark',role:'img','aria-label':`${chart.label}随扫描日期的变化`});
    root.append(svg('line',{x1:geo.pad.left,x2:geo.width - geo.pad.right,y1:geo.baseline + .5,y2:geo.baseline + .5,class:'spark-axis'}));
    const dates = geo.lines.flatMap(line => line.points.map(point => point.date)).filter(Boolean).sort();
    if (dates.length) {
      root.append(svg('text',{x:geo.pad.left,y:geo.height - 6,class:'spark-date','text-anchor':'start'},dates[0]));
      if (dates[dates.length - 1] !== dates[0]) root.append(svg('text',{x:geo.width - geo.pad.right,y:geo.height - 6,class:'spark-date','text-anchor':'end'},dates[dates.length - 1]));
    }
    for (const line of geo.lines) {
      root.append(svg('path',{d:line.d,fill:'none',stroke:line.color,'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'}));
      line.points.forEach((point,index) => {
        root.append(svg('circle',{cx:point.x,cy:point.y,r:4,fill:line.color,stroke:'#fff','stroke-width':2,class:'spark-dot'},svg('title',{},`${line.label} · ${point.date || '—'}：${formatValue(point.value)} ${chart.units}`)));
        // Direct labels: the last value of each line, plus the first one when there is a single line.
        const last = index === line.points.length - 1;
        if (last || (geo.lines.length === 1 && index === 0)) {
          const anchor = point.x > geo.width * .75 ? 'end' : point.x < geo.width * .25 ? 'start' : 'middle';
          root.append(svg('text',{x:point.x,y:Math.max(11,point.y - 8),class:'spark-value','text-anchor':anchor},formatValue(point.value)));
        }
      });
    }
    const legend = chart.lines.length > 1 ? node('div',{class:'spark-legend'},chart.lines.map(line => { const swatch = node('i',{'aria-hidden':'true'}); swatch.style.background = line.color; return node('span',{},swatch,line.label); })) : null;
    return node('figure',{class:'spark-figure'},node('figcaption',{},node('strong',{text:chart.label}),node('span',{text:` · ${chart.units}`})),root,legend);
  }
  function growthFact(label, growth, units, recent) {
    const sign = value => (Number(value) > 0 ? '+' : '');
    return node('div',{class:'growth-fact'},node('span',{class:'fact-label',text:label}),
      node('strong',{class:'growth-value',text:`${sign(growth.perYear)}${fmt(growth.perYear,1)} ${units}/年`}),
      node('small',{text:`${growth.from || '—'} → ${growth.to || '—'}${growth.days !== null ? `，${fmt(growth.days,0)} 天` : ''}${growth.delta !== null ? `，共 ${sign(growth.delta)}${fmt(growth.delta,1)} ${units}` : ''}`}),
      recent && (recent.from !== growth.from || recent.to !== growth.to) ? node('small',{text:`最近两次：${sign(recent.perYear)}${fmt(recent.perYear,1)} ${units}/年`}) : null);
  }
  function timelineTable(model, job) {
    const selected = [];
    const compare = button('并排比较所选两次扫描',() => {
      const rows = selected.map(sha => model.rows.find(row => row.sha === sha)).filter(Boolean).sort((a,b) => String(a.date).localeCompare(String(b.date)));
      if (rows.length !== 2) return;
      const params = new URLSearchParams({left:rows[0].compareJobId,right:rows[1].compareJobId});
      window.open(`/compare?${params.toString()}`,'_blank','noopener');
    },'primary');
    compare.disabled = true;
    const boxes = new Map();
    const sync = () => { compare.disabled = selected.length !== 2; for (const [sha,box] of boxes) box.checked = selected.includes(sha); };
    const head = ['比较','扫描日期','扫描','最大直径 mm','瘤体体积 mL',...model.releases.map(release => `${release.label} · ${WB.TIMELINE_METRICS[release.metric].label} ${WB.TIMELINE_METRICS[release.metric].units}`)];
    const body = model.rows.map(row => {
      const box = node('input',{type:'checkbox','aria-label':`选择 ${row.date || row.label || row.sha} 这次扫描`,disabled:!row.compareJobId});
      box.addEventListener('change',() => {
        const at = selected.indexOf(row.sha);
        if (box.checked && at < 0) { selected.push(row.sha); if (selected.length > 2) selected.shift(); }
        if (!box.checked && at >= 0) selected.splice(at,1);
        sync();
      });
      boxes.set(row.sha,box);
      const current = row.jobs.some(item => item.job_id === job?.id);
      return node('tr',{class:current ? 'current' : ''},
        node('td',{},box),
        node('td',{class:'num',title:row.dateSource === 'created_at' ? '未填写扫描日期，按任务创建日期排序' : ''},row.date || '—',row.dateSource === 'created_at' ? node('span',{class:'date-note',text:' *'}) : null),
        node('td',{text:[row.label,current ? '（当前）' : ''].filter(Boolean).join(' ') || '—'}),
        node('td',{class:'num',text:formatValue(row.maxDiameter)}),
        node('td',{class:'num',text:formatValue(row.sacVolume)}),
        ...model.releases.map(release => node('td',{class:'num',text:formatValue(row.metrics[release.id])})));
    });
    const hasCreated = model.rows.some(row => row.dateSource === 'created_at');
    return node('div',{class:'timeline-table'},
      node('div',{class:'table-wrap'},node('table',{},node('thead',{},node('tr',{},head.map(text => node('th',{scope:'col',text})))),node('tbody',{},body))),
      hasCreated ? help('* 未填写扫描日期，按任务创建日期排序，不参与年增长率计算。') : null,
      node('div',{class:'actions timeline-actions'},help('勾选两次扫描，并排打开三维报告（视角联动，不做点对点配准）。'),compare));
  }
  function renderTimeline(body, model, job) {
    if (!model || model.nScans < 2) {
      body.replaceChildren(node('p',{class:'timeline-empty',text:model && model.nScans === 1 ? '目前只有这一次扫描。' : '还没有该患者已完成的扫描。'}),
        help('上传随访扫描并填写相同的匿名患者编号和扫描日期后，这里会显示变化趋势。'));
      return;
    }
    const parts = [];
    const g = model.growth, growth = [];
    if (g.diameter) growth.push(growthFact('最大直径年增长',g.diameter,'mm',g.diameterRecent));
    if (g.volume) growth.push(growthFact('瘤体体积年增长',g.volume,'mL',g.volumeRecent));
    parts.push(node('p',{class:'timeline-summary',text:`共 ${model.nScans} 次扫描${model.rows[0]?.date ? `，${model.rows[0].date} 至 ${model.rows[model.rows.length - 1].date || '—'}` : ''}。`}));
    if (growth.length) parts.push(node('div',{class:'growth-row'},growth));
    const charts = model.charts.map(sparkChart).filter(Boolean);
    if (charts.length) parts.push(node('div',{class:'timeline-charts'},charts));
    parts.push(timelineTable(model,job));
    if (model.notes.length) parts.push(node('ul',{class:'compact-list timeline-notes'},model.notes.map(text => node('li',{text}))));
    body.replaceChildren(...parts);
  }
  function timelineCard(job, {inDialog = false} = {}) {
    const pid = job.patient_id;
    const body = node('div',{class:'timeline-body'},help('正在读取时间线…'));
    const card = node(inDialog ? 'div' : 'section',{class:`${inDialog ? '' : 'card '}timeline-card`,'aria-label':`患者 ${pid} 的时间线`},
      inDialog ? null : node('div',{class:'timeline-head'},node('p',{class:'eyebrow',text:'随访'}),node('h2',{text:`患者时间线 · ${pid}`})),body);
    const releaseId = job.model_release?.id || job.model_release?.release || '';
    (async () => {
      let data;
      try { data = await fetchTimeline(pid); }
      catch (error) {
        body.replaceChildren(node('p',{class:'timeline-empty',text:'时间线暂不可用。'}),help(error.status && error.status !== 404 ? `读取失败（${error.message}），稍后刷新重试。` : '当前服务版本尚未提供患者时间线。'));
        return;
      }
      renderTimeline(body,WB.timelineModel(data,releaseId),job);
    })();
    return card;
  }
  function timelineDialog(patientId, job = {}) {
    if (!patientId) return;
    openModal(`患者时间线 · ${patientId}`,[timelineCard({...job,patient_id:patientId},{inDialog:true})],[button('关闭',closeModal)],{wide:true});
  }
  function processCard(job) {
    const a = getA(job), ic = a.input_check || {}, summary = job.summary || {}, timing = summary.timing_s || a.timing_s || {};
    const details = node('details',{},node('summary',{text:'查看输入检查与计算过程'}),inputFacts(ic));
    const info = node('dl',{class:'timing-list'});
    const append = (label,value,glossKey) => {info.append(node('dt',{},label,glossKey && gloss(glossKey)),node('dd',{text:value}));};
    const displayUnit = ic.selected_units === 'auto' ? `${ic.resolved_units || ic.suggested_units || '—'}（自动）` : (ic.selected_units || ic.unit || '—');
    append('原始单位',displayUnit); append('换算倍数',`× ${ic.scale_factor || 1}`);
    if (Number.isFinite(Number(ic.unit_confidence))) append('单位自动判定置信度',`${fmt(Number(ic.unit_confidence) * 100,1)}%`);
    if (a.centerline) {append('中心线检查',a.centerline.hard_pass ? '通过' : '未通过'); append('中心线端点 / 分叉',`${a.centerline.topology?.endpoints ?? '—'} / ${a.centerline.topology?.junctions ?? '—'}`);}
    // v0.14: a stage served from the background geometry precompute shows ≈ 0 s; say so instead of looking broken.
    const cached = WB.cachedStages(summary.geometry_cache);
    for (const [key,value] of Object.entries(timing)) if (typeof value === 'number') append(TIMER_NAMES[key] || key,`${fmt(value,2)} 秒${WB.timingPrecomputed(key,cached) ? ' · 已预计算' : ''}`);
    const precomputeS = Number(job.timing?.precompute_s);
    if (typeof timing.precompute !== 'number' && Number.isFinite(precomputeS) && job.timing?.precompute_s != null) append(TIMER_NAMES.precompute,`${fmt(precomputeS,1)} 秒（不计入计算耗时）`);
    const attempt = WB.computeTiming(job.timing);
    if (attempt.split) append('本次尝试 / 累计计算',`本次 ${duration(attempt.attempt)} · 累计 ${duration(attempt.total)}`);
    const queue = job.timing?.queue_seconds ?? job.queue_seconds ?? job.elapsed?.queue;
    const manual = job.timing?.confirmation_seconds ?? job.confirmation_seconds ?? job.elapsed?.confirmation;
    if (queue != null) append('排队等待',duration(queue));
    if (manual != null) append('人工确认',duration(manual));
    if (summary.device) append('计算设备',`${summary.device}${summary.gpu ? ` · ${summary.gpu}` : ''}`);
    if (summary.model_release?.release || summary.model_release?.name) append('模型发布包',summary.model_release.release || summary.model_release.name,'release');
    if (summary.model_release?.fingerprint) append('发布指纹',String(summary.model_release.fingerprint).slice(0,16)+'…');
    if (job.input_sha256 || a.input_sha256) append('输入 SHA256',String(job.input_sha256 || a.input_sha256).slice(0,16)+'…');
    if (job.run_identity) append('运行身份',String(job.run_identity).slice(0,16)+'…','run_identity');
    if (Array.isArray(summary.time_axis) && summary.time_axis.length) append('时间帧',`${summary.time_axis.length} 帧 · ${summary.time_axis[0].label || 'frame_0'}`,'fixed_frame');
    details.append(info);
    // v0.14: the job's own event record (restarts, re-queues, CPU fallback, background precompute …), newest last.
    const events = (Array.isArray(job.events) ? job.events : []).filter(e => e && typeof e === 'object' && e.action && e.action !== 'progress').slice(-12);
    if (events.length) details.append(node('div',{class:'event-log'},node('strong',{text:'运行记录'}),node('ol',{class:'compact-list event-list'},
      events.map(e => node('li',{},timeNode(e.at,{friendly:false,cls:'event-time'}),node('span',{class:'event-text',text:` ${WB.eventText(e.action)}${e.action === 'device_fallback' && e.from_device ? `（${e.from_device} → ${e.to_device || 'cpu'}）` : ''}`}))))));
    if (ic.flags?.length) details.append(node('ul',{class:'compact-list'},ic.flags.map(text => node('li',{text}))));
    if (summary.release) details.append(help(`发布版本：${summary.release}`));
    return node('section',{class:'card process-card'},details);
  }
  function createViewer(host,a,onSelect) {
    const T = window.THREE;
    const fallback = message => host.replaceChildren(node('div',{class:'viewer-fallback'},node('strong',{text:'当前环境无法显示三维预览'}),node('p',{text:message}),node('p',{text:'端点名称、半径和世界坐标仍可在下方表格查看。请在支持 WebGL 的浏览器中检查壁面，或结合原始影像确认方向。'})));
    if (!T || !T.OrbitControls) {fallback('三维组件未成功加载，请刷新页面或检查本机静态文件。'); return null;}
    let renderer,controls,observer,frame,disposed = false;
    const disposables = new Set();
    try {
      renderer = new T.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true}); renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,2)); renderer.setClearColor(0xf6f9fc,1);
      host.append(renderer.domElement); renderer.domElement.setAttribute('aria-label','三维壁面预览，拖动旋转并点击端点');
      const scene = new T.Scene(), camera = new T.PerspectiveCamera(38,1,.1,10000), group = new T.Group(); scene.add(group);
      camera.up.set(0,0,1);   // before OrbitControls: it derives its orbit axis from camera.up at construction
      scene.add(new T.HemisphereLight(0xffffff,0x7895a7,1.15)); const light = new T.DirectionalLight(0xffffff,.55); light.position.set(200,300,300); scene.add(light);
      const points = [], targets = [], labels = [];
      const addDisposable = value => {disposables.add(value); return value;};
      const rawPreview = a.preview || a.input_check?.preview;
      // Keep interaction responsive on large uploads. The service may already send a
      // display mesh; this final guard never changes the geometry used for prediction.
      let preview = rawPreview;
      if (rawPreview?.vertices?.length && rawPreview?.faces?.length > 18000) {
        const faces = rawPreview.faces.slice(0,18000), used = new Set(faces.flat()), remap = new Map();
        const vertices = [...used].map((old,index) => { remap.set(old,index); return rawPreview.vertices[old]; });
        preview = {vertices,faces:faces.map(face => face.map(old => remap.get(old)))};
      }
      if (preview?.vertices?.length && preview?.faces?.length) {
        const geometry = addDisposable(new T.BufferGeometry()); geometry.setAttribute('position',new T.Float32BufferAttribute(preview.vertices.flat(),3)); geometry.setIndex(preview.faces.flat()); geometry.computeVertexNormals();
        group.add(new T.Mesh(geometry,addDisposable(new T.MeshPhongMaterial({color:0xa9c8d9,transparent:true,opacity:.30,side:T.DoubleSide,depthWrite:false,shininess:20}))));
        for (const point of preview.vertices) points.push(new T.Vector3(...point));
      }
      for (const line of a.proposal?.preview_polylines || []) {
        if (!line.xyz?.length) continue;
        const vectors = line.xyz.map(p => new T.Vector3(...p)); points.push(...vectors);
        const geometry = addDisposable(new T.BufferGeometry().setFromPoints(vectors)); group.add(new T.Line(geometry,addDisposable(new T.LineBasicMaterial({color:0x567891,transparent:true,opacity:.9,depthTest:false}))));
      }
      for (const e of a.proposal?.endpoints || []) if (e.center_mm?.length === 3) points.push(new T.Vector3(...e.center_mm));
      if (!points.length) throw new Error('这份任务没有可显示的三维坐标，请重新提取中心线。');
      const bounds = new T.Box3().setFromPoints(points), center = bounds.getCenter(new T.Vector3()), size = bounds.getSize(new T.Vector3());
      const span = Math.max(size.x,size.y,size.z,1); group.position.copy(center).multiplyScalar(-1);
      const makeLabel = (text,color) => {
        const canvas = document.createElement('canvas'); canvas.width = 512; canvas.height = 88;
        const context = canvas.getContext('2d'); if (!context) return null;
        context.fillStyle = 'rgba(255,255,255,.9)'; context.fillRect(0,0,512,88);
        context.font = '500 31px system-ui, sans-serif'; context.textAlign = 'center'; context.textBaseline = 'middle'; context.fillStyle = color; context.fillText(text,256,44);
        const texture = addDisposable(new T.CanvasTexture(canvas));
        const sprite = new T.Sprite(addDisposable(new T.SpriteMaterial({map:texture,depthTest:false,transparent:true}))); sprite.scale.set(span*.33,span*.057,1); return sprite;
      };
      for (const e of a.proposal?.endpoints || []) {
        if (!e.center_mm?.length) continue;
        const radius = Math.max(span*.013,Math.min(Number(e.radius_mm) * .7,span*.037));
        const sphere = new T.Mesh(addDisposable(new T.SphereGeometry(radius,20,14)),addDisposable(new T.MeshPhongMaterial({color:0x2569aa,depthTest:false})));
        sphere.position.set(...e.center_mm); sphere.renderOrder = 3; sphere.userData.endpoint = e; group.add(sphere); targets.push(sphere);
      }
      const axis = new T.AxesHelper(span*.16); axis.position.copy(bounds.min).add(new T.Vector3(-span*.06,-span*.06,-span*.06)); group.add(axis); disposables.add(axis.geometry); for (const material of Array.isArray(axis.material) ? axis.material : [axis.material]) disposables.add(material);
      for (const [name,offset,color] of [['X',[1,0,0],'#a54b45'],['Y',[0,1,0],'#397047'],['Z',[0,0,1],'#30679e']]) {
        const label = makeLabel(name,color); if (!label) continue; label.scale.multiplyScalar(.28); label.position.copy(axis.position).add(new T.Vector3(...offset).multiplyScalar(span*.18)); group.add(label);
      }
      controls = new T.OrbitControls(camera,renderer.domElement); controls.enableDamping = true; controls.dampingFactor = .09; controls.minDistance = span*.2; controls.maxDistance = span*12;
      camera.near = Math.max(span/1000,.001); camera.far = span*50;
      // Initial view and 「复位视角」: same viewing direction as before, framed with report_common.fitView so
      // the whole vessel fills the viewport (group is shifted by -center, so points are passed centred).
      const framePoints = points.map(point => [point.x - center.x,point.y - center.y,point.z - center.z]);
      const fit = () => {
        const aspect = Math.max(camera.aspect,.3);
        camera.up.set(0,0,1);
        if (RC?.fitView) {
          const view = RC.fitView(framePoints,{dir:[-.4,.85,-.8],up:[0,0,1],fov:camera.fov,aspect,margin:1.12,limit:6000});
          camera.position.set(...view.position); controls.target.set(...view.target);
          camera.far = Math.max(span * 50,view.distance * 4);
        } else {
          const distance = span / (2 * Math.tan(T.MathUtils.degToRad(camera.fov/2))) / Math.min(aspect,1);
          camera.position.set(distance*.4,-distance*.85,distance*.8); controls.target.set(0,0,0);
        }
        controls.update(); camera.updateProjectionMatrix();
      };
      const loop = renderOnDemand(() => {
        const moving = controls.update();
        declutterSprites(T,labels,camera,group,Math.max(host.clientWidth,1),Math.max(host.clientHeight,1));
        renderer.render(scene,camera); return moving;
      },() => !disposed);
      controls.addEventListener?.('change',loop.request);
      const resize = () => {const width = Math.max(host.clientWidth,1),height = Math.max(host.clientHeight,1); renderer.setSize(width,height,false); camera.aspect = width / height; camera.updateProjectionMatrix(); loop.request();};
      resize(); fit(); observer = new ResizeObserver(resize); observer.observe(host);
      const raycaster = new T.Raycaster(), mouse = new T.Vector2(); let pointerStart = null;
      const down = event => {pointerStart = [event.clientX,event.clientY];};
      const up = event => {if (!pointerStart || Math.hypot(event.clientX-pointerStart[0],event.clientY-pointerStart[1]) > 6) return; const r = renderer.domElement.getBoundingClientRect(); mouse.set((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1); raycaster.setFromCamera(mouse,camera); const hit = raycaster.intersectObjects(targets)[0]; if (hit) onSelect(hit.object.userData.endpoint.segment_id);};
      renderer.domElement.addEventListener('pointerdown',down); renderer.domElement.addEventListener('pointerup',up);
      loop.request();
      const result = {
        update(mapping) {
          for (const old of labels.splice(0)) {group.remove(old); const texture = old.material.map; texture?.dispose(); old.material.dispose(); disposables.delete(texture); disposables.delete(old.material);}
          // C3: capsule labels in the outlet colour, anchored beside the sphere (left of it for the
          // leftmost endpoints so neighbouring labels spread apart instead of stacking).
          const xs = targets.map(sphere => sphere.position.x), midX = xs.length ? (Math.min(...xs) + Math.max(...xs)) / 2 : 0;
          for (const sphere of targets) {
            const e = sphere.userData.endpoint, name = e.kind === 'inlet' ? 'inlet' : mapping[String(e.segment_id)], color = COLORS[name] || 0x74889b;
            sphere.material.color.setHex(color);
            const label = pillSprite(T,`#${e.segment_id} ${e.kind === 'inlet' ? '入口' : CN[name] || '未命名'}`,hexColor(color),span * .052,addDisposable);
            if (!label) continue;
            const side = e.kind === 'inlet' ? 1 : sphere.position.x < midX ? -1 : 1, radius = sphere.geometry?.parameters?.radius || span * .02;
            // Internal iliacs hang below the sphere (world +z is up in this viewer), the rest sit beside it.
            if (/^out-[lr]i$/.test(String(name))) { label.center.set(side < 0 ? 0 : 1,1); label.position.copy(sphere.position).add(new T.Vector3(0,0,-radius * 1.9)); }
            else { label.center.set(side > 0 ? 0 : 1,.5); label.position.copy(sphere.position).add(new T.Vector3(side * radius * 1.9,0,0)); }
            label.userData.home = label.position.clone(); label.userData.priority = e.kind === 'inlet' ? 2 : 1;
            group.add(label); labels.push(label);
          }
          loop.request();
        },
        select(sid) {for (const sphere of targets) sphere.scale.setScalar(Number(sphere.userData.endpoint.segment_id) === Number(sid) ? 1.5 : 1); loop.request();},
        reset() {fit(); loop.request();},
        snapshot() {try {renderer.render(scene,camera); const a = node('a',{href:renderer.domElement.toDataURL('image/png'),download:`wss-outlet-confirmation-${state.current}.png`}); document.body.append(a); a.click(); a.remove();} catch (_) {notify('当前浏览器无法保存三维截图，请使用系统截图工具。');}},
        dispose() {disposed = true; loop.cancel(); cancelAnimationFrame(frame); observer?.disconnect(); controls?.dispose(); renderer.domElement.removeEventListener('pointerdown',down); renderer.domElement.removeEventListener('pointerup',up); for (const item of disposables) item.dispose?.(); renderer.dispose(); renderer.forceContextLoss();}
      };
      return result;
    } catch (error) {disposed = true; cancelAnimationFrame(frame); observer?.disconnect(); controls?.dispose(); for (const item of disposables) item.dispose?.(); renderer?.dispose(); fallback(error.message || '浏览器未启用 WebGL。'); return null;}
  }

  // ---------------------------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------------------
  // Keyboard shortcuts (§19.9 工作台): / N J K ↑↓ Enter O G ?  — via report_common.installShortcuts.
  // ---------------------------------------------------------------------------------------------
  const anyDialogOpen = () => Boolean($('modal').open || uploadDialog?.open || $('cohort-dialog').open || !$('action-menu').hidden);
  const inList = target => Boolean(target?.closest?.('#jobs-list, #cases-list'));
  const interactive = target => Boolean(target?.closest?.('button, a, summary, input, select, textarea, [role="button"]'));
  function focusRow(id) {
    const row = [...document.querySelectorAll('#jobs-list [data-job-id], #cases-list [data-job-id]')].find(el => el.dataset.jobId === id);
    if (row) { row.focus({preventScroll:true}); row.scrollIntoView({block:'nearest'}); }
  }
  function stepCase(delta, focus = false) {
    const ids = [...new Set([...document.querySelectorAll('#jobs-list [data-job-id], #cases-list [data-job-id]')].filter(el => el.offsetParent !== null).map(el => el.dataset.jobId))];
    const id = WB.stepId(ids,state.current,delta);
    if (id && id !== state.current) { selectJob(id); if (focus) focusRow(id); }
  }
  function installWorkbenchShortcuts() {
    if (state.shortcuts || !RC?.installShortcuts) return;
    const idle = () => state.authenticated && !$('workspace').hidden && !anyDialogOpen();
    const current = () => (state.job && state.job.id === state.current ? state.job : null);
    state.shortcuts = RC.installShortcuts([
      {keys:['/'],label:'聚焦搜索',group:'列表',run:() => { $('jobs-query').focus(); $('jobs-query').select?.(); },when:idle},
      {keys:['n'],label:'新建预测（也可以把 STL 拖到页面上）',group:'列表',run:openUploadDialog,when:idle},
      {keys:['j'],label:'下一例',group:'列表',run:() => stepCase(1),when:idle},
      {keys:['k'],label:'上一例',group:'列表',run:() => stepCase(-1),when:idle},
      {keys:['ArrowDown'],label:'下一例（列表获得焦点时）',group:'列表',run:() => stepCase(1,true),when:event => idle() && inList(event.target)},
      {keys:['ArrowUp'],label:'上一例（列表获得焦点时）',group:'列表',run:() => stepCase(-1,true),when:event => idle() && inList(event.target)},
      {keys:['Enter','o'],label:'打开三维报告',group:'当前病例',run:() => openReport(current()),when:event => idle() && current()?.status === 'done' && (event.key !== 'Enter' || !interactive(event.target))},
      {keys:['g'],label:'打开一页纸报告',group:'当前病例',run:() => openOnepage(current()),when:() => idle() && current()?.status === 'done'},
    ],{title:'工作台快捷键',doc:document});
  }
  // ---------------------------------------------------------------------------------------------
  // Settings menu: shortcut help and the institution report template (§19.4, optional).
  // ---------------------------------------------------------------------------------------------
  async function templateDialog() {
    let data;
    try { data = await request('/api/report-template'); }
    catch (error) { notify(error.status === 404 ? '当前服务版本尚未提供报告模板设置。' : error.message); return; }
    const t = data.template || {}, editable = data.editable !== false;
    const text = (key, label, placeholder, max = 200) => { const input = node('input',{type:'text',maxlength:max,value:t[key] || '',placeholder,'aria-label':label,disabled:!editable,autocomplete:'off'}); return {key,input,field:node('div',{class:'meta-field'},node('label',{text:label}),input)}; };
    const fields = [text('institution','机构名称','例如 某某医院'),text('department','科室','例如 血管外科'),text('report_title','报告标题','留空时按结果类型自动生成')];
    const footer = node('textarea',{rows:2,maxlength:500,'aria-label':'页脚声明',disabled:!editable}); footer.value = t.footer_note || '';
    const signatures = node('input',{type:'text',maxlength:200,value:Array.isArray(t.signature_lines) ? t.signature_lines.join('，') : '','aria-label':'签字栏',disabled:!editable,placeholder:'例如 报告人，审阅人；留空则不印签字栏'});
    const glossaryMode = node('select',{'aria-label':'术语表',disabled:!editable},node('option',{value:'used',text:'只列本页用到的术语'}),node('option',{value:'all',text:'列出全部术语'}),node('option',{value:'none',text:'不印术语表'}));
    glossaryMode.value = ['used','all','none'].includes(t.show_glossary) ? t.show_glossary : 'used';
    const appendix = node('input',{type:'checkbox',id:'template-appendix',checked:t.appendix !== false,disabled:!editable});
    const status = node('p',{class:'help',text:editable ? '修改后新生成的一页纸会使用这些信息；已打印的纸质报告不受影响。' : '当前账户只能查看。修改需要在本机回环模式下或由管理员操作。'});
    const save = button('保存模板',async () => {
      save.disabled = true; status.textContent = '正在保存…';
      const body = Object.fromEntries(fields.map(item => [item.key,item.input.value.trim()]));
      body.footer_note = footer.value.trim(); body.signature_lines = WB.parseTags(signatures.value); body.show_glossary = glossaryMode.value; body.appendix = appendix.checked;
      try { await request('/api/report-template',{method:'PUT',body}); closeModal(); notify('报告模板已保存，新生成的一页纸会使用它。',false); }
      catch (error) { status.textContent = error.status === 403 ? '没有修改权限：只有本机回环模式或管理员可以修改。' : error.message; save.disabled = false; }
    },'primary');
    save.disabled = !editable;
    openModal('报告模板',[help('一页纸页眉、页脚、签字栏与附录的机构设置，对本服务全部任务生效。'),
      node('div',{class:'meta-form'},...fields.map(item => item.field),node('div',{class:'meta-field'},node('label',{text:'签字栏'}),signatures,help('逗号分隔，每项印成「姓名 ____ 日期 ____」。')),
        node('div',{class:'meta-field'},node('label',{text:'术语表'}),glossaryMode),node('div',{class:'meta-field wide-field'},node('label',{text:'页脚声明'}),footer)),
      node('label',{class:'ack',htmlFor:'template-appendix'},appendix,node('span',{text:'打印附录页（分支统计、发现详情、输入检查、限制声明等）'})),status],
      [button(editable ? '取消' : '关闭',closeModal),save]);
  }
  $('settings-button').addEventListener('click',event => {
    event.stopPropagation?.();
    openActionMenu($('settings-button'),[
      {label:'三步上手与快捷键',run:guideDialog},
      {label:'快捷键说明（?）',run:() => state.shortcuts?.showHelp()},
      {label:'报告模板…',run:templateDialog},
      {label:'队列总览',run:() => { $('cohort-panel').open = true; $('cohort-expand').click(); }},
      {label:'回收站',run:() => { $('trash-panel').open = true; $('trash-panel').scrollIntoView({block:'nearest'}); }},
    ]);
  });

  state.unread = WB.unreadParse(store.get('wss-unread','[]'));
  try { state.groupsOpen = {...state.groupsOpen, ...JSON.parse(store.get('wss-groups-v2','{}') || '{}')}; } catch (_) {}
  setListMode(store.get('wss-list-mode','jobs'));
  installWorkbenchShortcuts();
  window.addEventListener('beforeunload',() => state.viewer?.dispose());
  // Every 2 s: detail poll while its stream is down; every 10 s the list (and the overview) only while the owner
  // stream is down; relative times every minute in place; health every 30 s.
  setInterval(() => {sweepThumbs(); if (!document.hidden && state.authenticated) {if (!state.eventsLive) pollJob(); ++state.listTick;
    if (!state.ownerEvents && state.listTick % 5 === 0) openOwnerEvents();
    if (state.listTick % 5 === 0 && !state.ownerLive) { refreshJobs(); if (!state.current && overviewVisible()) refreshOverview(); }
    if (state.listTick % 30 === 0) refreshListTimes(); if (state.listTick % 15 === 0) refreshHealth();}},2000);
  setInterval(() => { if (!document.hidden && state.authenticated && etaLive.size) tickEta(); },1000);
  window.addEventListener('beforeunload',() => { closeEvents(); closeOwnerEvents(); });
  document.addEventListener('visibilitychange',() => {if (!document.hidden && state.authenticated) {pollJob(); refreshJobs(); if (!state.current && overviewVisible()) refreshOverview(); if (state.current && WB.unreadRemove(state.unread,state.current)) persistUnread(); if (!state.ownerEvents) openOwnerEvents();}});
  startSession();
})();
