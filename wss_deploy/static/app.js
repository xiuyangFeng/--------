/* WSS local workbench. Dynamic data is inserted through textContent, never HTML. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const WB = window.WssWorkbenchCore;
  const CN = {inlet:'主动脉入口', 'out-le':'左髂外', 'out-li':'左髂内', 'out-re':'右髂外', 'out-ri':'右髂内'};
  const COLORS = {inlet:0x2569aa, 'out-le':0x19837a, 'out-li':0x6a9b35, 'out-re':0xd08838, 'out-ri':0xa16cb3};
  const STATUS = {queued:'排队中', queued_A:'排队中', queued_B:'排队中', running:'计算中', running_A:'正在提取中心线', running_B:'正在预测血流场', awaiting_input:'待确认输入', awaiting_confirmation:'待确认出口', awaiting_outlets:'待确认出口', done:'已完成', failed:'失败', cancelled:'已取消', interrupted:'已中断', cancelling:'正在取消'};
  const PHASE = {ingest:'检查壁面与尺寸', input:'检查壁面与尺寸', centerline:'提取中心线', naming:'生成出口建议', geometry:'构建几何特征', smooth_resample:'平滑与重采样', features:'构建几何特征', inference:'预测血流场', inference_5_models:'预测壁面 WSS', report:'生成报告与导出文件', export:'生成报告与导出文件', metrics:'汇总预测统计', metrics_and_export:'汇总统计并导出', queued:'等待计算资源', cancelled:'任务已取消', interrupted:'服务中断，等待恢复'};
  const TIMER_NAMES = {ingest:'输入检查', centerline:'中心线提取', smooth_resample:'平滑与重采样', features:'几何特征', inference_5_models:'五模型预测', metrics:'指标统计', interpolation:'壁面插值', metrics_and_export:'统计与导出', export:'文件导出', report:'HTML 报告', total:'计算总耗时'};
  const FAMILY_TEXT = WB.FAMILY_TEXT;
  const state = {csrf:'',authenticated:false,session:null,current:null,job:null,jobs:[],releases:[],renderKey:null,mapping:{},viewer:null,geometry:null,busy:false,polling:false,listBusy:false,listTick:0,sequence:0,history:{q:'',status:'',patient_id:'',tag:'',page:1,page_size:20,total:0},selectMode:false,selected:new Map(),
    listMode:'jobs',viewAll:false,cases:[],caseReleases:[],prefs:{},prefsServer:null,glossary:null,unread:new Set(),ownerEvents:null,lastStatus:{},trash:[],claimable:[],modalResolve:null,refreshTimer:null,baseTitle:document.title,groupsOpen:{pending:true,other:true,reviewed:false},
    cohort:null,cohortBusy:false,cohortFiltered:[],cohortSort:{key:'wss_p99_pa',direction:'desc'}};
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
  const duration = value => {
    if (!Number.isFinite(Number(value)) || value === null || value === undefined) return '—';
    const s = Math.max(0, Math.round(Number(value)));
    return s < 60 ? `${s} 秒` : `${Math.floor(s / 60)} 分 ${s % 60} 秒`;
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
  const clearNotice = () => { $('notice').hidden = true; };
  const notify = (message,error = true) => { const el = $('notice'); el.textContent = message; el.className = `notice${error ? ' error' : ''}`; el.hidden = false; };
  const errorText = error => typeof error === 'string' ? error : error?.message || '请求未完成，请稍后重试。';
  const store = {
    get(key, fallback = null) { try { const raw = localStorage.getItem(key); return raw === null ? fallback : raw; } catch (_) { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch (_) {} },
  };
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
      const error = new Error(errorText(result.error || result.message)); error.status = response.status; error.body = result;
      if (response.status === 401) { state.authenticated = false; renderLoginPanel(state.session || {}); $('login-panel').hidden = false; $('workspace').hidden = true; setConnection('会话已过期，请重新连接'); }
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
      let message = `HTTP ${response.status}`;
      try { const data = await response.json(); message = errorText(data.error || data.message); } catch (_) {}
      const error = new Error(message); error.status = response.status; throw error;
    }
    const blob = await response.blob();
    saveBlob(blob, WB.filenameFromDisposition(response.headers.get('Content-Disposition'), filename));
  }
  function saveBlob(blob, name) {
    const href = URL.createObjectURL(blob);
    const a = node('a',{href,download:name}); document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(href), 20000);
  }
  function setConnection(text, online = false) { $('connection').textContent = text; $('connection').className = `connection${online ? ' online' : ''}`; }

  // ---------------------------------------------------------------------------------------------
  // Modal dialog (duplicate upload choice, password change, batch export)
  // ---------------------------------------------------------------------------------------------
  function openModal(title, content, actions = []) {
    const dialog = $('modal'), body = $('modal-body');
    body.replaceChildren(...[node('h2',{id:'modal-title',text:title}),...content,actions.length ? node('div',{class:'actions modal-actions'},...actions) : null].filter(Boolean));
    if (typeof dialog.showModal === 'function') { if (!dialog.open) dialog.showModal(); }
    else dialog.setAttribute('open','');
    return body;
  }
  function closeModal() {
    const dialog = $('modal');
    if (typeof dialog.close === 'function' && dialog.open) dialog.close(); else dialog.removeAttribute('open');
  }
  $('modal').addEventListener('close', () => { const resolve = state.modalResolve; state.modalResolve = null; $('modal-body').replaceChildren(); if (resolve) resolve({action:'cancel'}); });
  $('modal').addEventListener('click', event => { if (event.target === $('modal') && !state.modalLocked) closeModal(); });
  $('modal').addEventListener('cancel', event => { if (state.modalLocked) event.preventDefault(); });

  // ---------------------------------------------------------------------------------------------
  // Session, login modes, claim (C3)
  // ---------------------------------------------------------------------------------------------
  const loginMode = session => session?.login || (session?.shared ? 'token' : 'none');
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
  }
  function applySession(session, {claimable} = {}) {
    state.session = session; state.csrf = session.csrf_token || ''; state.authenticated = Boolean(session.authenticated);
    renderLoginPanel(session);
    $('login-panel').hidden = state.authenticated; $('workspace').hidden = !state.authenticated;
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
      if (state.authenticated) {
        setConnection(session.shared ? `共享服务 · 已连接${session.username ? ` · ${session.username}` : ''}` : '本机服务 · 已连接',true); $('upload-button').disabled = false;
        await loadPreferences(); await refreshReleases(); applyUploadPreferences(); await refreshJobs(); openOwnerEvents(); updateUnreadBadge(); refreshTrash();
      } else setConnection('等待连接');
    } catch (error) { notify(error.message); setConnection('连接失败 · 请刷新重试'); }
  }
  $('login-form').addEventListener('submit', async event => {
    event.preventDefault(); const submit = event.currentTarget.querySelector('button'); submit.disabled = true; clearNotice();
    const username = $('login-username').value.trim();
    const body = !$('login-user-group').hidden && username ? {username,password:$('login-password').value} : {token:$('access-token').value};
    try {
      const result = await request('/api/session',{method:'POST',body});
      $('access-token').value = ''; $('login-password').value = '';
      state.pendingClaimable = Array.isArray(result.claimable) ? result.claimable : null;
      await startSession();
    }
    catch (error) { notify(error.status === 429 ? '口令错误次数过多，请一分钟后再试。' : error.message); } finally { submit.disabled = false; }
  });
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
      if (newField.value.length < 8) { status.textContent = '新口令至少 8 个字符。'; return; }
      if (newField.value !== again.value) { status.textContent = '两次输入的新口令不一致。'; return; }
      submit.disabled = true;
      try { await request('/api/session/password',{method:'POST',body:{old_password:oldField.value,new_password:newField.value}}); closeModal(); notify('口令已更新。',false); }
      catch (error) { status.textContent = error.message; submit.disabled = false; }
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
    try {
      const result = await request('/api/releases'); state.releases = Array.isArray(result) ? result : result.releases || [];
      const select = $('upload-release'); if (!select) return;
      select.replaceChildren();
      if (!state.releases.length) { select.append(node('option',{value:'',text:'没有可用发布包'})); select.disabled = true; return; }
      for (const release of state.releases) {
        const label = `${release.id || release.release}${release.default ? ' · 默认' : ''}${release.models_count ? ` · ${release.models_count} 个模型 / 字段` : ''}`;
        select.append(node('option',{value:release.id || release.release,text:label}));
      }
      const selected = state.releases.find(item => item.default) || state.releases[0];
      if (selected) select.value = selected.id || selected.release;
      select.disabled = state.releases.length <= 1;
      function updateReleaseSettings() {
        const selected = state.releases.find(item => (item.id || item.release) === select.value);
        const volume = Boolean(selected?.contract?.fields?.velocity);
        const seedSelect = $('upload-seeds');
        const count = Number(selected?.models_count || selected?.model_count || 5);
        if (seedSelect) {
          seedSelect.replaceChildren(node('option',{value:'all',text:`全部模型（${volume ? '每字段 ' : ''}${count}）`}));
          for (const n of [1,3,5]) if (n < count) seedSelect.append(node('option',{value:String(n),text:`${volume ? '每字段 ' : ''}${n} 个模型`}));
        }
        const helpText = $('release-help');
        if (helpText) helpText.textContent = volume ? 'PF6 压力＋VF6 速度：体内、流线、自动和手动截面；压力可切换壁面。固定收缩期单帧。' : 'X5D 壁面 WSS：热点、分支统计。任务会固定绑定所选发布包。';
      }
      select.onchange = updateReleaseSettings;
      updateReleaseSettings();
    } catch (error) { const helpText = $('release-help'); if (helpText) helpText.textContent = `发布包列表读取失败：${error.message}`; }
  }
  const releaseFamily = releaseId => {
    const fromCases = WB.familyOfRelease(releaseId, state.caseReleases); if (fromCases) return fromCases;
    const release = state.releases.find(item => (item.id || item.release) === releaseId);
    if (!release) return null;
    const protocol = release.contract?.protocol;
    return protocol === 'single_frame_volume' || release.contract?.fields?.velocity ? 'volume' : protocol === 'single_frame_wss' || release.contract?.fields?.wss ? 'wall' : null;
  };

  // ---------------------------------------------------------------------------------------------
  // Upload with duplicate detection (C2)
  // ---------------------------------------------------------------------------------------------
  $('stl-file').addEventListener('change', () => {
    const files = [...$('stl-file').files];
    $('file-name').textContent = files.length ? `${files.length} 个 STL：${files.slice(0,2).map(file => file.name).join('、')}${files.length > 2 ? '…' : ''}` : '1 个入口 · 4 个髂动脉出口';
    const list = $('upload-files'); if (!list) return;
    list.replaceChildren(); list.hidden = files.length < 2;
    for (const file of files) list.append(node('div',{class:'upload-file-row'},node('span',{text:file.name}),node('span',{class:'upload-file-state',text:'待上传'})));
  });
  function duplicateDialog(file, body) {
    // Resolves with {action: 'open'|'reuse'|'force'|'cancel', jobId?}
    return new Promise(resolve => {
      state.modalResolve = resolve;
      const existing = Array.isArray(body.existing) ? body.existing : [];
      const rows = existing.slice(0,8).map(item => node('tr',{},node('td',{text:item.case_id || '—'}),node('td',{text:item.release_id ? `${FAMILY_TEXT[item.family] || ''} ${item.release_id}`.trim() : '—'}),node('td',{},badge(item.status)),node('td',{text:REVIEW_LABEL[item.review] || item.review || '—'}),node('td',{class:'coordinates',text:item.created_at || ''})));
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
    const form = event.currentTarget; const files = [...$('stl-file').files]; if (!files.length) return;
    const invalid = files.find(file => !/\.stl$/i.test(file.name) || !file.size);
    if (invalid) { notify(`文件 ${invalid.name} 不是有效的 STL。`); return; }
    state.busy = true; const submit = $('upload-button'); submit.disabled = true; submit.textContent = '正在上传…';
    const progress = $('upload-progress'), bar = $('upload-progress-bar'), progressText = $('upload-progress-text');
    if (progress) progress.hidden = false; if (bar) {bar.value = 0; bar.max = files.length;}
    const rows = [...($('upload-files')?.children || [])]; const failures = [], created = [], opened = [];
    const baseCase = $('case-id').value.trim(), metadata = {
      case_id: baseCase, patient_id: $('patient-id').value.trim(), scan_label: $('scan-label').value.trim(),
      scan_date: $('scan-date')?.value || '',
      tags: $('case-tags').value.trim(), notes: $('case-notes').value.trim(), units: $('upload-units').value,
      release_id: $('upload-release').value, remove_fragments: 'false', device: $('upload-device').value,
      seed_count: $('upload-seeds').value, threads: $('upload-threads').value.trim()
    };
    try {
      const caseIdFor = file => { const stem = file.name.replace(/\.stl$/i,''); return files.length > 1 ? (baseCase ? `${baseCase}-${stem}` : stem) : baseCase; };
      const appendCommon = data => { for (const [key,value] of Object.entries(metadata)) if (key !== 'case_id' && value !== '') data.append(key, value); };
      const setRow = (index, text) => { const stateNode = rows[index]?.querySelector('.upload-file-state'); if (stateNode) stateNode.textContent = text; if (bar) bar.value = Math.max(bar.value, index + 1); };
      const record = (index, jobId, error, reusedFrom) => {
        if (jobId) { created.push(jobId); setRow(index, reusedFrom ? `已创建任务 ${jobId}（复用 ${reusedFrom}）` : `已创建任务 ${jobId}`); }
        else { failures.push(`${files[index].name}：${error}`); setRow(index, `失败：${error}`); }
      };
      const uploadOne = async (index, mode) => {
        const data = new FormData(); data.append('stl', files[index]); if (caseIdFor(files[index])) data.append('case_id', caseIdFor(files[index])); appendCommon(data); data.append('on_duplicate', mode);
        const result = await request('/api/jobs',{method:'POST',body:data,timeout:120000});
        const job = result.job || result, id = job.id || result.job_id;
        if (!id) throw new Error('服务未返回任务编号。');
        return {id, reusedFrom: result.reused_from || job.reused_from || null};
      };
      const resolveDuplicate = async (index, body) => {
        const choice = await duplicateDialog(files[index], body);
        if (choice.action === 'open') { if (choice.jobId) opened.push(choice.jobId); setRow(index, `已打开已有任务 ${choice.jobId || ''}`); return; }
        if (choice.action === 'cancel') { setRow(index, '已跳过（同几何已有任务）'); return; }
        try { const {id, reusedFrom} = await uploadOne(index, choice.action); record(index, id, null, reusedFrom); }
        catch (error) { record(index, null, error.body?.reuse_unavailable ? '已有任务不能复用，请选择强制重算。' : error.message); }
      };
      if (files.length === 1) {
        if (progressText) progressText.textContent = `正在上传 ${files[0].name}`;
        try { const {id, reusedFrom} = await uploadOne(0, 'ask'); record(0, id, null, reusedFrom); }
        catch (error) { if (error.status === 409 && error.body?.duplicate) await resolveDuplicate(0, error.body); else record(0, null, error.message); }
      } else {
        // One multipart request per batch of files (repeated `stl` parts + per-file metadata list).
        const duplicates = [];
        for (let start = 0; start < files.length; start += BATCH_LIMIT) {
          const chunk = files.slice(start, start + BATCH_LIMIT);
          chunk.forEach((_, i) => setRow(start + i, '上传中…'));
          if (progressText) progressText.textContent = `正在批量上传 ${start + 1}–${start + chunk.length} / ${files.length}`;
          const data = new FormData(); for (const file of chunk) data.append('stl', file);
          data.append('metadata_json', JSON.stringify(chunk.map(file => ({case_id: caseIdFor(file)})))); appendCommon(data); data.append('on_duplicate','ask');
          try {
            const result = await request('/api/jobs/batch',{method:'POST',body:data,timeout:600000});
            const seen = new Set();
            for (const item of result.results || []) {
              seen.add(item.index);
              if (item.duplicate) { duplicates.push({index:start + item.index, body:item}); setRow(start + item.index, '同几何已有任务，等待选择…'); continue; }
              record(start + item.index, item.job?.id || null, item.error?.message || '未创建任务', item.job?.reused_from || item.reused_from);
            }
            chunk.forEach((_, i) => { if (!seen.has(i)) record(start + i, null, '服务未返回该文件的结果'); });
          } catch (error) { chunk.forEach((_, i) => record(start + i, null, error.message)); }
        }
        for (const item of duplicates) { if (progressText) progressText.textContent = `请选择如何处理 ${files[item.index].name}`; await resolveDuplicate(item.index, item.body); }
      }
      savePreferences({upload: WB.uploadPreferencesFrom({...metadata, remember_patient: $('remember-patient').checked})});
      await refreshJobs(true);
      const focus = created[created.length - 1] || opened[opened.length - 1];
      if (focus) { await selectJob(focus); if (window.matchMedia('(max-width:760px)').matches) $('detail').scrollIntoView({behavior:'smooth',block:'start'}); }
      const openedText = opened.length ? `；打开已有 ${opened.length} 个` : '';
      if (failures.length) notify(`已创建 ${created.length}/${files.length} 个任务${openedText}；失败：${failures.join('；')}`);
      else notify(`已创建 ${created.length} 个任务${openedText}。`,false);
    } finally { state.busy = false; submit.disabled = false; submit.textContent = '上传并检查'; if (progressText && !failures.length) progressText.textContent = `已完成 ${created.length}/${files.length}`; }
  });

  // ---------------------------------------------------------------------------------------------
  // Job list: by task (with review groups, C18) or by case (C1)
  // ---------------------------------------------------------------------------------------------
  $('refresh-jobs').addEventListener('click', async () => { clearNotice(); await refreshJobs(true); await pollJob(true); refreshTrash(); });
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
    const pageLabel = $('jobs-page'); if (pageLabel) pageLabel.textContent = state.history.total ? `${(state.history.page - 1) * state.history.page_size + 1}–${Math.min(state.history.page * state.history.page_size,state.history.total)} / ${state.history.total}` : '0 条';
    if ($('jobs-prev')) $('jobs-prev').disabled = state.history.page <= 1;
    if ($('jobs-next')) $('jobs-next').disabled = state.history.page * state.history.page_size >= state.history.total;
  }
  async function refreshJobs(showError = false) {
    if (!state.authenticated || state.listBusy) return;
    state.listBusy = true;
    try {
      if (state.listMode === 'cases') { await refreshCases(showError); return; }
      const filters = {...state.history}; const pendingOnly = filters.status === 'pending_review';
      if (pendingOnly) filters.status = 'done';
      const query = new URLSearchParams(); for (const [key,value] of Object.entries(filters)) if (key !== 'total' && value !== '' && value != null) query.set(key,String(value));
      if (state.viewAll) query.set('all','1');
      const result = await request(`/api/jobs?${query.toString()}`); let jobs = Array.isArray(result) ? result : result.jobs || [];
      if (!Array.isArray(result)) { state.history.total = Number(result.total || jobs.length); state.history.page = Number(result.page || state.history.page); state.history.page_size = Number(result.page_size || state.history.page_size); } else { state.history.total = jobs.length; }
      if (pendingOnly) jobs = jobs.filter(WB.isPendingReview);
      state.jobs = jobs;
      for (const id of [...state.selected.keys()]) { const fresh = state.jobs.find(job => job.id === id); if (fresh) state.selected.set(id, fresh); else state.selected.delete(id); }
      renderJobsList();
      updateSelectionBar(); updatePagination();
    } catch (error) { if (showError) notify(error.message); setConnection('连接暂时中断 · 自动重试'); }
    finally { state.listBusy = false; }
  }
  function jobRow(job) {
    const b = button('', () => selectJob(job.id),'job-button'); b.setAttribute('aria-current',String(job.id === state.current)); b.dataset.jobId = job.id;
    const subtitle = [job.patient_id && `患者 ${job.patient_id}`,job.scan_label,job.family && FAMILY_TEXT[job.family],job.tags && `#${Array.isArray(job.tags) ? job.tags.join(' #') : job.tags}`].filter(Boolean).join(' · ');
    const badges = node('span',{class:'job-badges'},badge(job.status));
    const review = reviewBadge(job); if (review && review.textContent !== '待审阅') badges.append(review);
    const name = node('span',{class:'job-name'});
    if (WB.unreadHas(state.unread, job.id)) name.append(node('span',{class:'unread-dot','aria-label':'未读',title:'任务状态有更新'}));
    name.append(node('span',{text:job.case_id || '匿名病例'}),subtitle && node('small',{class:'job-subtitle',text:subtitle}));
    b.append(node('span',{class:'job-title'},name,badges),node('span',{class:'job-time',text:job.created_at || job.id}));
    const row = node('li',{class:'job-row'});
    if (state.selectMode) {
      const box = node('input',{type:'checkbox','aria-label':`选择 ${job.case_id || job.id}`,checked:state.selected.has(job.id),disabled:isRunning(job)});
      box.addEventListener('change',() => { if (box.checked) state.selected.set(job.id,job); else state.selected.delete(job.id); updateSelectionBar(); });
      row.append(node('span',{class:'job-select'},box));
    }
    row.append(b);
    if (deletable(job)) { const del = button('删除',() => deleteJobs([job]),'job-delete danger'); del.setAttribute('aria-label',`删除任务 ${job.case_id || job.id}`); row.append(del); }
    return row;
  }
  function renderJobsList() {
    const list = $('jobs-list'); list.replaceChildren(); $('jobs-empty').hidden = state.jobs.length > 0;
    const groups = WB.splitByReview(state.jobs);
    if (!groups.pending.length && !groups.reviewed.length) { for (const job of state.jobs) list.append(jobRow(job)); return; }
    const section = (key, label, jobs, cls = '') => {
      if (!jobs.length) return;
      const details = node('details',{open:state.groupsOpen[key] !== false},node('summary',{class:cls},node('span',{text:label}),node('span',{class:'count',text:String(jobs.length)})),node('ul',{},jobs.map(jobRow)));
      details.addEventListener('toggle',() => { state.groupsOpen[key] = details.open; store.set('wss-groups',JSON.stringify(state.groupsOpen)); });
      list.append(node('li',{class:'job-group'},details));
    };
    section('pending','待审阅',groups.pending,'pending');
    section('other','进行中与未完成',groups.other);
    section('reviewed','已审阅',groups.reviewed);
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
      else { if (showError) notify(error.message); setConnection('连接暂时中断 · 自动重试'); }
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
      chip.append(node('span',{class:'run-family',text:FAMILY_TEXT[run.family] || '结果'}),node('span',{class:'run-release',text:run.release_id || '—'}),badge(run.status));
      const review = reviewBadge(pseudo); if (review) chip.append(review);
      chip.append(node('span',{class:'job-time',text:run.created_at || ''}));
      if (WB.unreadHas(state.unread, run.job_id)) chip.prepend(node('span',{class:'unread-dot','aria-label':'未读'}));
      runs.append(node('li',{},chip));
    }
    article.append(runs);
    const actions = node('div',{class:'case-actions'});
    // §15.2: the card's identity fields are edited through the most recent run of this geometry.
    const latestRun = [...model.runs].sort((a,b) => String(b.created_at || '').localeCompare(String(a.created_at || '')))[0];
    if (latestRun?.job_id) { const edit = button('编辑',() => editMetadata(latestRun.job_id)); edit.title = `编辑病例信息（最近任务 ${latestRun.job_id}）`; actions.append(edit); }
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
      submit.disabled = true; status.textContent = '正在保存…';
      const body = WB.metadataPayload({case_id:caseId.value, patient_id:patient.value, scan_label:scanLabel.value, scan_date:scanDate.value, tags:tags.value, notes:notes.value}, job.version);
      // The service refuses to blank an existing case id; an anonymous task simply keeps none.
      if (!body.case_id) { if (job.case_id) { status.textContent = '病例编号不能为空。'; submit.disabled = false; return; } delete body.case_id; }
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
  function batchRerunDialog(jobs) {
    if (!jobs.length) return;
    if (!state.releases.length) { notify('没有可用的发布包，无法批量重跑。'); return; }
    const select = node('select',{'aria-label':'重跑使用的发布包'},...state.releases.map(item => {
      const id = item.id || item.release;
      const family = FAMILY_TEXT[releaseFamily(id)];
      return node('option',{value:id,text:family ? `${id} · ${family}` : id});
    }));
    const preferred = state.releases.find(item => item.default) || state.releases[0];
    if (preferred) select.value = preferred.id || preferred.release;
    const progress = node('p',{class:'help','aria-live':'polite',text:`已选择 ${jobs.length} 个任务。只有已完成且已确认出口的任务会被重跑，其余会跳过并说明原因。`});
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
    openModal('用发布包重跑所选任务',[
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
  function closeOwnerEvents() { if (state.ownerEvents) { try { state.ownerEvents.close(); } catch (_) {} } state.ownerEvents = null; }
  function openOwnerEvents() {
    if (state.ownerEvents || typeof EventSource === 'undefined') return;
    try {
      const source = new EventSource('/api/events'); state.ownerEvents = source;
      source.addEventListener('job', event => { let data = null; try { data = JSON.parse(event.data); } catch (_) { return; } handleOwnerEvent(data); });
      source.onerror = () => { if (source.readyState === EventSource.CLOSED) state.ownerEvents = null; };
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
  function handleOwnerEvent(data) {
    if (!data || !data.job_id) return;
    if (WB.shouldNotify(data, state.lastStatus)) {
      if (data.job_id !== state.current || document.hidden) { WB.unreadAdd(state.unread, data.job_id, data.version); persistUnread(); }
      showNotification(data);
    }
    if (data.job_id === state.current && data.action !== 'progress') { state.renderKey = null; pollJob(); }
    if (data.action !== 'progress') scheduleListRefresh();
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
    for (const id of ['export-csv','export-xlsx','bundle-selected','batch-export','rerun-selected']) $(id).disabled = done.length === 0 || state.busy;
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
    const stateText = node('textarea',{rows:4,placeholder:'粘贴报告「视图」菜单导出的视图状态 JSON','aria-label':'视图状态 JSON'});
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
    $('detail').replaceChildren(node('div',{class:'empty-state card'},node('p',{class:'eyebrow',text:'任务已删除'}),node('h1',{text:message || '该任务已移入回收站。'}),node('p',{class:'help',text:'可以从左侧列表打开其他任务、上传新的 STL，或在回收站中恢复。'})));
  }
  async function deleteJobs(jobs) {
    if (state.busy || !jobs.length) return;
    if (jobs.length > DELETE_LIMIT) { notify(`一次最多删除 ${DELETE_LIMIT} 个任务，请分批操作。`); return; }
    const names = jobs.slice(0,5).map(job => job.case_id || job.id).join('、') + (jobs.length > 5 ? ` 等 ${jobs.length} 个` : '');
    const message = `将删除 ${jobs.length} 个任务（${names}）及其全部文件：输入 STL、中心线、报告、导出结果与历史记录。\n任务会进入回收站，30 天内可恢复；到期后自动彻底清除。确定继续？`;
    if (!window.confirm(message)) return;
    state.busy = true; updateSelectionBar(); clearNotice();
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
      if (deleted.includes(state.current)) clearDetail(trashed ? '该任务已移入回收站。' : undefined);
      const where = trashed ? '已移入回收站，30 天内可恢复' : '已删除';
      if (failures.length) notify(`${where} ${deleted.length} 个；未删除：${failures.join('；')}`);
      else notify(`${where} ${deleted.length} 个任务。`,false);
    } catch (error) { notify(error.message); }
    finally { state.busy = false; await refreshJobs(true); refreshTrash(); }
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
    const meta = [item.family && FAMILY_TEXT[item.family], item.release_id, item.patient_id && `患者 ${item.patient_id}`, item.scan_label, `删除于 ${item.deleted_at || '—'}`, Number.isFinite(Number(item.days_left)) ? `剩余 ${Math.max(0, Math.ceil(Number(item.days_left)))} 天` : null].filter(Boolean).join(' · ');
    const restore = button('恢复',async () => {
      restore.disabled = true;
      try { const result = await request(`/api/trash/${encodeURIComponent(item.id)}/restore`,{method:'POST',body:{}}); notify(`已恢复任务 ${item.case_id || item.id}。`,false); await refreshJobs(true); await refreshTrash(); const job = result.job || result; if (job?.id) await selectJob(job.id); }
      catch (error) { notify(error.message); restore.disabled = false; }
    },'primary');
    const purge = button('彻底删除',async () => {
      if (!window.confirm(`彻底删除任务 ${item.case_id || item.id} 及其全部文件？此操作无法恢复。`)) return;
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
    const width = Number(canvas.width) || 560, height = Number(canvas.height) || 140;
    const pad = {left:6, right:6, top:8, bottom:16};
    ctx.clearRect(0,0,width,height);
    const values2 = (values || []).map(Number).filter(Number.isFinite);
    const reference2 = (reference || []).map(Number).filter(Number.isFinite);
    ctx.font = '11px system-ui, sans-serif'; ctx.textBaseline = 'alphabetic';
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
      select.replaceChildren(node('option',{value:'',text:'全部发布包'}),...WB.cohortReleases(rows).map(id => node('option',{value:id,text:id})));
      if (previous && [...select.options].some(option => option.value === previous)) select.value = previous;
      renderCohort();
    } catch (error) {
      state.cohort = null; $('cohort-head').replaceChildren(); $('cohort-rows').replaceChildren(); $('cohort-count').textContent = '';
      $('cohort-export-csv').disabled = true; $('cohort-export-xlsx').disabled = true;
      $('cohort-status').textContent = error.status === 404 ? '后端未提供队列数据' : `队列数据读取失败：${error.message}`;
    } finally { state.cohortBusy = false; }
  }
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
  async function selectJob(id) {
    state.current = id; state.sequence++; state.renderKey = null; clearNotice();
    openEvents(id);
    if (WB.unreadRemove(state.unread, id)) persistUnread();
    markCurrent();
    $('detail').setAttribute('aria-busy','true');
    await pollJob(true);
  }
  async function pollJob(showError = false) {
    if (!state.current || !state.authenticated || state.polling) return;
    const id = state.current, sequence = state.sequence; state.polling = true;
    try {
      const result = await request(`/api/jobs/${encodeURIComponent(id)}`);
      if (id !== state.current || sequence !== state.sequence) return;
      const job = result.job || result; state.job = job;
      const key = `${job.id}/${job.version}/${job.status}`;
      if (key !== state.renderKey) { renderJob(job); state.renderKey = key; }
      else updateLiveStatus(job);
      setConnection('服务已连接',true);
    } catch (error) { if (showError || error.status === 404) notify(error.message); setConnection('连接暂时中断 · 自动重试'); }
    finally { state.polling = false; $('detail').setAttribute('aria-busy','false'); }
  }
  async function mutate(job, suffix, payload = {}) {
    if (state.busy) return;
    state.busy = true; clearNotice();
    const controls = [...$('detail').querySelectorAll('button,input,select')];
    const previous = controls.map(el => el.disabled); controls.forEach(el => {el.disabled = true;});
    try {
      const result = await request(jobUrl(job,suffix),{method:'POST',body:{...payload,version:job.version}});
      const created = result.job || result;
      if (suffix === '/rerun' && created?.id && created.id !== job.id) { state.current = created.id; state.sequence++; state.renderKey = null; }
      state.renderKey = null; await pollJob(true); await refreshJobs();
    } catch (error) {
      notify(error.status === 409 ? `任务状态已更新，请核对刷新后的内容再操作。${error.message}` : error.message);
      if (error.status === 409) { state.renderKey = null; await pollJob(true); }
    } finally { state.busy = false; controls.forEach((el,i) => {if (el.isConnected) el.disabled = previous[i];}); }
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
  function renderJob(job) {
    state.viewer?.dispose(); state.viewer = null;
    const detail = $('detail'); detail.replaceChildren();
    const headBadges = node('span',{class:'job-badges'},badge(job.status));
    if (locked(job)) headBadges.append(node('span',{class:'badge reviewed',text:'已审阅 · 已锁定'}));
    else if (job.review?.status === 'reopened') headBadges.append(node('span',{class:'badge reopened',text:'已重新打开'}));
    const identity = [job.patient_id && `患者 ${job.patient_id}`,job.scan_label,job.scan_date,(Array.isArray(job.tags) ? job.tags : String(job.tags || '').split(',').map(t => t.trim()).filter(Boolean)).map(tag => `#${tag}`).join(' ')].filter(Boolean).join(' · ');
    const idLine = node('p',{class:'job-id'},node('span',{text:`任务 ${job.id}${identity ? ` · ${identity}` : ''}`}),button('编辑信息',() => editMetadata(job),'text-button meta-edit'));
    const heading = node('section',{class:'card'},node('div',{class:'job-heading'},node('h1',{text:job.case_id || '匿名病例'}),headBadges),idLine);
    if (job.reused_from) heading.append(node('p',{class:'job-origin',text:`复用自任务 ${job.reused_from}：中心线与出口确认沿用，只重新计算所选发布包。`}));
    else if (job.source_job_id) heading.append(node('p',{class:'job-origin',text:`由任务 ${job.source_job_id} 换发布包重跑，复用其中心线与出口确认。`}));
    const idx = progressIndex(job);
    heading.append(node('ol',{class:'stepper','aria-label':'预测步骤'},['上传 STL','确认输入','确认出口','预测血流场','查看结果'].map((label,i) => node('li',{class:i < idx ? 'complete' : i === idx ? 'current' : '',text:label,'aria-current':i === idx ? 'step' : null}))));
    const statusLine = node('div',{class:'status-line'});
    if (['running','running_A','running_B','queued','queued_A','queued_B','cancelling'].includes(job.status)) statusLine.append(node('span',{class:'spinner','aria-hidden':'true'}));
    statusLine.append(node('p',{},node('strong',{id:'job-phase'}),node('span',{id:'job-live-detail',class:'status-detail'})));
    if (['queued','queued_A','queued_B','running','running_A','running_B','awaiting_input','awaiting_confirmation','awaiting_outlets'].includes(job.status)) statusLine.append(button('取消任务',() => mutate(job,'/cancel'),'danger'));
    if (deletable(job)) statusLine.append(button('删除任务',() => deleteJobs([job]),'danger'));
    heading.append(statusLine); detail.append(heading); updateLiveStatus(job);
    if (job.error) {
      const errorCard = node('section',{class:'card'},node('h2',{text:job.status === 'interrupted' ? '任务中断，可恢复' : '本次计算未完成'}),callout(errorText(job.error),'error'));
      if (job.error.diagnostic_id) errorCard.append(node('p',{class:'diagnostic',text:`诊断编号：${job.error.diagnostic_id}`}));
      errorCard.append(help('检查下方输入信息后重试。已有的有效中心线与特征将按服务端校验结果复用。')); detail.append(errorCard);
    }
    if (['failed','cancelled','interrupted'].includes(job.status)) {
      const recovery = node('section',{class:'card'},node('h2',{text:job.status === 'interrupted' ? '恢复这个任务' : '继续处理'}),help(job.status === 'cancelled' ? '任务已停止。恢复时会重新检查输入和可复用的中间结果。' : '修正输入或恢复任务后，将从可用的阶段继续。'));
      if (!locked(job)) recovery.append(node('div',{class:'actions'},button(job.status === 'interrupted' ? '恢复任务' : '重试任务',() => mutate(job,'/retry'),'primary'))); detail.append(recovery);
    }
    const a = getA(job);
    if (job.status === 'done') { const narrative = narrativeCard(job); if (narrative) detail.append(narrative); detail.append(resultCard(job)); }
    else if ((job.status === 'awaiting_confirmation' || job.status === 'awaiting_outlets') && a.proposal) detail.append(outletCard(job));
    else if (job.status === 'awaiting_input' || (['failed','interrupted'].includes(job.status) && a.input_check?.status === 'needs_confirmation')) detail.append(inputCard(job));
    else if (['running','running_A','running_B','queued','queued_A','queued_B'].includes(job.status)) {
      detail.append(node('section',{class:'card'},node('h2',{text:job.stage === 'B' ? '正在生成血流场预测报告' : '正在准备几何与中心线'}),node('p',{class:'muted',text:job.stage === 'B' ? '完成后将显示所选发布包的预测统计和可交互三维报告。' : '几何检查通过后提取中心线，并展示需要核对的入口和四个出口。'}),help('可以切换到其他任务；任务状态保存在服务端。取消计算后，正在执行的步骤会在安全停止点结束。')));
    }
    if (a.input_check && job.status !== 'awaiting_input') detail.append(processCard(job));
  }
  function updateLiveStatus(job) {
    if (!$('job-phase')) return;
    const phase = typeof job.phase === 'object' ? job.phase?.label || job.phase?.name : job.phase;
    $('job-phase').textContent = (['running','running_A','running_B'].includes(job.status) && (PHASE[phase] || phase)) || STATUS[job.status] || phase || '等待任务更新';
    const detail = [];
    if (job.detail) detail.push(typeof job.detail === 'string' ? job.detail : job.detail.message || '');
    if (['queued','queued_A','queued_B'].includes(job.status) && job.queue_position != null) detail.push(`队列位置 ${job.queue_position}`);
    const queued = ['queued','queued_A','queued_B'].includes(job.status), waitingConfirmation = ['awaiting_input','awaiting_confirmation','awaiting_outlets'].includes(job.status);
    const elapsed = queued ? (job.timing?.queue_s ?? job.timing?.queue_seconds) : waitingConfirmation ? (job.timing?.confirmation_s ?? job.timing?.confirmation_seconds) : (typeof job.elapsed === 'number' ? job.elapsed : job.elapsed?.total ?? job.elapsed_s);
    if (elapsed != null) detail.push(queued || waitingConfirmation ? `已等待 ${duration(elapsed)}` : `已计算 ${timeText(job,elapsed)}`);
    if (job.status === 'awaiting_input') detail.push('请确认原始单位、换算尺寸和几何处理');
    if (job.status === 'awaiting_confirmation' || job.status === 'awaiting_outlets') detail.push('请结合原始影像核对解剖方向');
    if (job.status === 'done') detail.push('报告与导出文件已就绪');
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
    card.append(node('div',{class:'actions'},help('服务端会再次检查全部输入门限。'),submit));
    return card;
  }
  function outletCard(job, options = {}) {
    const override = options.override === true;
    const a = getA(job), p = a.proposal;
    if (!a.preview && job.preview) a.preview = job.preview;
    state.mapping = {...(job.mapping || p.mapping)};
    const card = node('section',{class:`card${override ? ' override-outlet-card' : ''}`},node('p',{class:'eyebrow',text:override ? '人工复核' : '步骤 03'}),node('h2',{text:override ? '检查或修改出口命名后重算' : '旋转壁面，确认每个出口'}),node('p',{class:'muted',text:'点击三维端点或表格中的编号，定位对应开口。颜色与命名同步更新。'}));
    if (Number.isFinite(Number(p.confidence))) {
      card.append(callout(p.confirmation_required
        ? `自动命名最低置信度 ${fmt(Number(p.confidence) * 100,1)}%，低于 95%，需要人工核对。`
        : `自动命名最低置信度 ${fmt(Number(p.confidence) * 100,1)}%，已自动继续；如需修正，可在此人工修改并重算。`, p.confirmation_required ? 'warn' : 'success'));
    }
    const directionNote = p.direction_note || getA(job).input_check?.direction_note || '方向来源：STL 世界坐标。图中的 X / Y / Z 不是患者左 / 右 / 前 / 后方向；请结合原始影像确认左右及髂内、髂外。';
    card.append(callout(directionNote,'warn'));
    for (const flag of p.flags || []) card.append(callout(flag,'warn'));
    const viewerHost = node('div',{class:'viewer',id:'outlet-viewer','aria-label':'可旋转的血管壁面和中心线三维预览'});
    const reset = button('复位视角',() => state.viewer?.reset());
    const snapshot = button('保存截图',() => state.viewer?.snapshot());
    card.append(node('div',{class:'viewer-shell'},node('div',{class:'viewer-toolbar'},node('span',{text:'半透明壁面 · 中心线 · 开口'}),node('div',{class:'viewer-buttons'},reset,snapshot)),viewerHost,node('div',{class:'viewer-hint',text:'拖动旋转 · 滚轮缩放 · 右键平移 · 点击彩色端点定位。壁面仅为显示抽稀，计算仍使用完整几何。'})));
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
      const row = node('tr',{},node('td',{},selectButton),node('td',{text:`${fmt(e.radius_mm)} mm`}),node('td',{},name),node('td',{class:'coordinates',text:(e.center_mm || []).map(v => fmt(v)).join(', ')}));
      rows.set(sid,row); row.dataset.segmentId = sid; dot.dataset.segmentId = sid; tbody.append(row);
    }
    card.append(node('div',{class:'table-wrap'},node('table',{},node('thead',{},node('tr',{},...['端点','开口半径','解剖命名','世界坐标 X, Y, Z（mm）'].map(text => node('th',{scope:'col',text})))),tbody)));
    const changed = node('p',{class:'mapping-changes','aria-live':'polite'});
    const ack = node('input',{type:'checkbox',id:'outlet-ack'});
    const confirm = button(override ? '采用修改后的命名并重算' : '确认出口并开始预测',() => mutate(job,'/confirm',{mapping:{...state.mapping},acknowledged:true,...(override ? {override:true,stage:'B'} : {})}),'primary');
    function validMapping() { const names = [...selects.keys()].map(id => state.mapping[id]); return names.length === 4 && new Set(names).size === 4 && names.every(name => name && Object.hasOwn(CN,name)); }
    function updateConfirm() { confirm.disabled = !ack.checked || !validMapping(); }
    function updateMapping() {
      ack.checked = false; updateConfirm();
      selects.forEach((el,sid) => {el.value = state.mapping[sid] || '';});
      const differences = Object.keys(state.mapping).filter(key => state.mapping[key] !== p.mapping?.[key]);
      changed.textContent = differences.length ? `已修改 ${differences.length} 个出口：${differences.map(id => `#${id} → ${CN[state.mapping[id]] || '未命名'}`).join('；')}。确认后将采用当前命名计算。` : (override ? '当前结果使用自动命名建议。你可以修改后重新计算。' : '当前使用自动命名建议，仍需人工核对。');
      for (const e of p.endpoints || []) { const name = e.kind === 'inlet' ? 'inlet' : state.mapping[String(e.segment_id)]; const dot = rows.get(String(e.segment_id))?.querySelector('.endpoint-dot'); if (dot) dot.style.backgroundColor = `#${(COLORS[name] || 0x74889b).toString(16).padStart(6,'0')}`; }
      state.viewer?.update(state.mapping);
    }
    const swap = pairs => { for (const key of Object.keys(state.mapping)) if (pairs[state.mapping[key]]) state.mapping[key] = pairs[state.mapping[key]]; updateMapping(); };
    card.append(node('div',{class:'mapping-actions'},button('左右互换',() => swap({'out-le':'out-re','out-li':'out-ri','out-re':'out-le','out-ri':'out-li'})),button('左侧内 / 外互换',() => swap({'out-le':'out-li','out-li':'out-le'})),button('右侧内 / 外互换',() => swap({'out-re':'out-ri','out-ri':'out-re'})),button('恢复自动建议',() => {state.mapping = {...p.mapping}; updateMapping();})),changed);
    card.append(callout('左右命名参与模型坐标的构建。修改左右后，确认计算会按新的坐标和出口分配重新生成预测。'));
    card.append(node('label',{class:'ack',htmlFor:'outlet-ack'},ack,node('span',{text:override ? '我已核对修改后的患者方向、主动脉入口和四个出口命名。' : '我已结合原始影像核对患者方向、主动脉入口以及四个髂内 / 髂外出口，确认表格命名正确。'})));
    ack.addEventListener('change',updateConfirm); updateMapping();
    card.append(node('div',{class:'actions'},help('四个出口名称必须各使用一次，同一髂总下的两个出口须属于同侧。'),confirm));
    const openings = a.centerline?.openings || [];
    if (openings.length) {
      const inlet = node('select',{'aria-label':'重新选择入口'},node('option',{value:'',text:'选择需要作为入口的开口'}),...openings.map(o => node('option',{value:String(o.opening_id),text:`开口 ${o.opening_id} · r ${fmt(o.radius_mm)} mm${o.role === 'inlet' ? ' · 当前入口' : ''}`})));
      const recompute = button('重提中心线',() => mutate(job,'/confirm',{mapping:{...state.mapping},inlet:Number(inlet.value),acknowledged:true})); recompute.disabled = true;
      inlet.addEventListener('change',() => {recompute.disabled = inlet.value === '';});
      card.append(node('details',{},node('summary',{text:'入口选错了？重新选择入口'}),help('入口变更会重新提取中心线。完成后需要再次确认所有出口命名。'),node('div',{class:'inlet-row'},inlet,recompute)));
    }
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
  async function loadGeometry(job) {
    const key = `${job.id}/${getA(job).created_at || ''}`;
    if (state.geometry?.key === key) return state.geometry.value;
    try {
      const value = await request(jobUrl(job,'/geometry'),{timeout:60000});
      state.geometry = {key,value};
      return value;
    } catch (error) { notify(`三维预览数据读取失败：${error.message}`); return null; }
  }
  // -------------------------------------------------------------------------------------------
  // Aneurysm morphology (§17.1) and the automatic narrative (§17.2).  Both are new summary keys:
  // a job computed before them simply renders neither, no branch below may throw on the absence.
  // -------------------------------------------------------------------------------------------
  function morphologyBlock(summary) {
    const morphology = summary.morphology;
    if (!morphology || typeof morphology !== 'object') return null;
    const aorta = morphology.aorta || {}, max = aorta.max || {}, sac = aorta.sac || {}, neck = aorta.neck || {};
    const facts = [];
    const largest = Number(max.max_diameter_mm);
    if (Number.isFinite(largest)) {
      const position = Number(max.distance_from_inlet_mm ?? max.s_from_root_mm);
      facts.push(fact('最大直径',`${fmt(largest,1)} mm`,Number.isFinite(position) ? `入口下 ${fmt(position,0)} mm` : '位置见三维报告','max_diameter'));
    }
    if (sac.present) facts.push(fact('瘤体长度 / 体积',`${fmt(sac.length_mm,1)} mm / ${fmt(sac.volume_ml,0)} mL`,Number.isFinite(Number(sac.threshold_mm)) ? `判定阈值 ${fmt(sac.threshold_mm,1)} mm` : '截面面积沿弧长积分','aneurysm_sac'));
    if (neck.present) facts.push(fact('瘤颈直径 / 长度',`${fmt(neck.diameter_mean_mm,1)} mm / ${fmt(neck.length_mm,1)} mm`,'近端瘤颈平均直径与长度','aneurysm_neck'));
    const lumen = Number(morphology.lumen_volume_ml);
    if (Number.isFinite(lumen)) facts.push(fact('全腔体积',`${fmt(lumen,0)} mL`,'开口封盖后的管腔体积','lumen_volume'));
    if (!facts.length) return null;
    const reference = Number(aorta.reference_diameter_mm);
    const block = node('div',{class:'quality-card morphology-card'},
      node('div',{class:'quality-head'},node('strong',{text:'瘤体形态'}),Number.isFinite(reference) ? node('span',{class:'quality-status',text:`参考直径 ${fmt(reference,1)} mm`}) : null),
      node('div',{class:'check-grid morphology-grid'},...facts));
    const notes = (Array.isArray(morphology.notes) ? morphology.notes : []).filter(text => typeof text === 'string' && text.trim());
    block.append(node('small',{text:notes.length ? notes.join('；') : '直径取中心线每站截面的最大 Feret 直径，由壁面网格求交得到，不依赖预测数值。'}));
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
  function resultCard(job) {
    const summary = job.summary || {}, peak = summary.peak || {}, times = summary.timing_s || {}, volume = Boolean(summary.fields?.velocity), stats = summary.volume_statistics || {};
    const heading = node('div',{class:'result-head'},node('h2',{text:volume ? '压力与速度体场已就绪' : '壁面 WSS 结果已就绪'}));
    const chip = diameterChip(maxDiameterMm(summary)); if (chip) heading.append(chip);
    const card = node('section',{class:'card'},node('p',{class:'eyebrow',text:'预测完成'}),heading);
    const automatic = [...(job.mapping_history || [])].reverse().find(item => item.source === 'automatic_high_confidence');
    if (automatic) card.append(callout(`出口命名已自动确认（最低置信度 ${fmt(Number(automatic.confidence) * 100,1)}%）。仍可打开下方入口人工复核并重算。`,'success'));
    if (volume) card.append(node('div',{class:'result-cards'},fact('体内速度 p99',`${fmt(stats.speed_m_s?.p99,3)} m/s`,'内部采样点速度模长','speed'),fact('体内相对压力范围',`${fmt(stats.pressure_interior_pa?.min,1)} ～ ${fmt(stats.pressure_interior_pa?.max,1)} Pa`,'相对于该帧体积平均压力','relative_pressure'),fact('计算耗时',timeText(job,times.total),'不包含排队与人工确认')));
    else {
      card.append(node('div',{class:'result-cards'},fact('全场 p99',`${fmt(peak.p99_pa,2)} Pa`,'预测点云的第 99 百分位','p99'),fact('全场最大值',`${fmt(peak.max_pa,2)} Pa`,peak.branch ? `最大值位置：${peak.branch}` : '位置见三维报告','max'),fact('计算耗时',timeText(job,times.total),'不包含排队与人工确认')));
      const field = summary.wss_field_pa || {};
      if (Number.isFinite(Number(field.area_frac_low)) || Number.isFinite(Number(field.area_frac_high))) {
        const t = Array.isArray(field.thresholds_pa) && field.thresholds_pa.length === 3 ? field.thresholds_pa : [0.4,4,7];
        card.append(node('div',{class:'result-fractions'},node('span',{},'面积占比（估计）',gloss('area_fraction')),node('span',{},`低 WSS < ${t[0]} Pa `,node('strong',{text:`${fmt(Number(field.area_frac_low) * 100,1)}%`})),node('span',{},`高 WSS > ${t[1]} Pa `,node('strong',{text:`${fmt(Number(field.area_frac_high) * 100,1)}%`})),node('span',{},`极高 > ${t[2]} Pa `,node('strong',{text:`${fmt(Number(field.area_frac_very_high) * 100,1)}%`})),node('span',{},'阈值',gloss('thresholds'))));
      }
    }
    const morphology = morphologyBlock(summary); if (morphology) card.append(morphology);
    const quality = summary.quality;
    if (quality) {
      const qualityCard = node('div',{class:'quality-card result-quality'},
        node('div',{class:'quality-head'},node('strong',{},'结果质量与复核提示',gloss('quality_grade')),node('span',{class:`quality-status ${quality.level || ''}`,text:quality.label || quality.level || '未评估'})));
      const reasons = Array.isArray(quality.reasons) ? quality.reasons : [];
      qualityCard.append(node('small',{text:reasons.length ? reasons.join('；') : '五模型集成离散度在当前分流阈值内。逐点标准差保存在质量审计文件中。'}));
      card.append(qualityCard);
    }
    const reference = summary.reference_assessment || {}, population = reference.population || {};
    if (population.status || reference.status) {
      const populationText = population.status === 'pass' ? `折外人群分位 ${fmt(population.percentile,1)}%` : '未配置可核验的人群参照';
      const geometryText = reference.status === 'pass' ? '几何在参考范围内' : reference.status === 'review' ? '几何超出参考范围，请复核' : '未配置几何参考范围';
      card.append(node('div',{class:'quality-card reference-card'},node('div',{class:'quality-head'},node('strong',{},'人群位置与几何范围',gloss('population_percentile')),node('span',{class:`badge ${reference.status === 'review' ? 'failed' : ''}`,text:reference.status === 'pass' ? '通过' : reference.status === 'review' ? '需复核' : '未配置'})),node('small',{text:`${populationText} · ${geometryText}`})));
    }
    const surface = summary.surface_statistics || summary.results?.statistics?.surface;
    if (surface && Number.isFinite(Number(surface.p99_pa))) {
      const coverage = Number(surface.covered_area_fraction);
      card.append(node('div',{class:'quality-card surface-card'},node('div',{class:'quality-head'},node('strong',{},'壁面面积加权参考',gloss('area_weighted_p99')),node('span',{class:'badge',text:'补充口径'})),
        node('div',{class:'kv'},node('span',{class:'fact-label',text:'面积加权 p99'}),node('strong',{text:`${fmt(surface.p99_pa,2)} Pa`}),node('span',{class:'fact-label',text:'有效覆盖面积'}),node('strong',{text:Number.isFinite(coverage) ? `${fmt(coverage * 100,1)}%` : '—'})),
        node('small',{text:'主指标仍为预测点云 p99；面积口径仅统计 Gaussian 插值后完整有效三角面。'})));
    }
    card.append(reviewSection(job));
    const link = (text,path,cls = '') => node('a',{class:`button ${cls}`,href:jobUrl(job,path),target:'_blank',rel:'noopener',text});
    const actions = volume ? [link('打开体场与截面报告','/report','primary'),link('下载体场 VTP','/files/volume_fields.vtp'),link('下载壁面压力 VTP','/files/wall_pressure.vtp'),link('下载体场 CSV','/files/points_volume.csv')] : [link('打开三维报告','/report','primary'),link('下载壁面 VTP','/files/wall_wss.vtp'),link('下载点云 CSV','/files/points_wss.csv')];
    if (volume && summary.exports?.streamlines) actions.push(link('下载流线 VTP','/files/streamlines.vtp'));
    actions.push(link('一页纸报告','/onepage'));
    actions.push(button('打包下载 zip',() => bundleJobs([job])));
    const boundRelease = summary.model_release?.registry_id || summary.model_release?.name || summary.model_release?.release || summary.release;
    const alternatives = state.releases.filter(item => (item.id || item.release) !== boundRelease);
    if (alternatives.length) {
      const rerunSelect = node('select',{class:'inline-select','aria-label':'选择重跑发布包'},...alternatives.map(item => node('option',{value:item.id || item.release,text:item.id || item.release})));
      actions.push(rerunSelect,button('用发布包重跑',() => mutate(job,'/rerun',{release_id:rerunSelect.value})));
    }
    const family = volume ? 'volume' : 'wall';
    const comparable = state.jobs.filter(item => item.id !== job.id && item.status === 'done' && (item.family ? item.family === family : true));
    if (comparable.length) {
      const compareSelect = node('select',{class:'inline-select','aria-label':'选择比较任务'},...comparable.map(item => node('option',{value:item.id,text:`${item.case_id || '匿名病例'} · ${item.scan_label || item.id}`})));
      actions.push(compareSelect);
      if (!volume) actions.push(button('比较结果',() => compareWith(job,compareSelect.value)));
      actions.push(button('并排比较',() => { const params = new URLSearchParams({left:job.id,right:compareSelect.value}); window.open(`/compare?${params.toString()}`,'_blank','noopener'); }));
    }
    if (getA(job).proposal && !locked(job)) actions.push(button('检查 / 修改出口命名并重算',() => {
      if (card.parentElement?.querySelector('.override-outlet-card')) return;
      const editor = outletCard(job,{override:true});
      card.parentElement?.append(editor);
      editor.scrollIntoView({behavior:'smooth',block:'start'});
    }));
    card.append(node('div',{class:'actions'},actions));
    card.append(help(volume ? '速度查看体内点云、稳态流线及截面；压力可切换壁面和内部。截面是有限厚度采样切片，流线来自固定时间帧，压力不是绝对血压。' : '报告可旋转查看整段壁面、查看最大值位置和分支统计，并导出当前视角截图。p99 是统计量，不对应单一解剖位置。'));
    const fileLinks = [node('a',{href:jobUrl(job,'/files/summary.json'),target:'_blank',rel:'noopener',text:'查看完整统计与参数 JSON'})];
    if (summary.run_manifest) fileLinks.push(node('a',{href:jobUrl(job,'/files/run_manifest.json'),target:'_blank',rel:'noopener',text:'查看可复现运行清单'}));
    card.append(node('div',{class:'file-links'},fileLinks));
    return card;
  }
  function reviewSection(job) {
    const review = job.review || {status:'unreviewed'};
    const wrap = node('div',{class:`quality-card review-card ${review.status || 'unreviewed'}`});
    const head = node('div',{class:'quality-head'},node('strong',{},'审阅签字',gloss('review_status')),node('span',{class:`quality-status review-${review.status || 'unreviewed'}`,text:REVIEW_LABEL[review.status] || '未审阅'}));
    wrap.append(head);
    if (review.status === 'reviewed') {
      wrap.append(node('small',{text:`${review.by || '—'} 于 ${review.at || '—'} 签字通过${review.note ? `：${review.note}` : ''}。结果已锁定：不能重算、修改出口命名或删除；换发布包重跑会新建任务。`}));
      const reviewer = node('input',{type:'text',maxlength:80,placeholder:'重新打开人的匿名标识','aria-label':'重新打开人'});
      const reason = node('textarea',{rows:2,maxlength:2000,placeholder:'重新打开的原因（必填）','aria-label':'重新打开原因'});
      const reopen = button('重新打开',() => mutate(job,'/review',{decision:'reopen',reviewer:reviewer.value.trim(),note:reason.value.trim()}),'danger');
      reopen.disabled = true;
      const check = () => { reopen.disabled = !reviewer.value.trim() || !reason.value.trim(); };
      reviewer.addEventListener('input',check); reason.addEventListener('input',check);
      wrap.append(node('div',{class:'review-form'},reviewer,reason,node('div',{class:'actions'},help('重新打开后可再次修改出口命名或重算；原签字记录保留在审计中。'),reopen)));
    } else {
      if (review.status === 'reopened') wrap.append(node('small',{text:`${review.by || '—'} 于 ${review.at || '—'} 重新打开${review.note ? `：${review.note}` : ''}。`}));
      else wrap.append(node('small',{text:'签字后结果将被锁定：不能重算、修改出口命名或删除，报告与运行清单会记录审阅人和时间。'}));
      const reviewer = node('input',{type:'text',maxlength:80,placeholder:'审阅人匿名标识（必填）','aria-label':'审阅人',autocomplete:'off'});
      if (state.session?.username && !reviewer.value) reviewer.value = state.session.display_name || state.session.username;
      const note = node('textarea',{rows:2,maxlength:2000,placeholder:'审阅备注（可选）','aria-label':'审阅备注'});
      const ack = node('input',{type:'checkbox',id:`review-ack-${job.id}`});
      const approve = button('签字通过并锁定',() => mutate(job,'/review',{decision:'approve',reviewer:reviewer.value.trim(),note:note.value.trim()}),'primary');
      approve.disabled = true;
      const check = () => { approve.disabled = !(ack.checked && reviewer.value.trim()); };
      reviewer.addEventListener('input',check); ack.addEventListener('change',check);
      wrap.append(node('div',{class:'review-form'},reviewer,note,node('label',{class:'ack',htmlFor:`review-ack-${job.id}`},ack,node('span',{text:'我已核对出口命名、输入检查与结果质量提示，确认该结果可以签字。'})),node('div',{class:'actions'},approve)));
    }
    return wrap;
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
    for (const [key,value] of Object.entries(timing)) if (typeof value === 'number') append(TIMER_NAMES[key] || key,`${fmt(value,2)} 秒`);
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
    if (ic.flags?.length) details.append(node('ul',{class:'compact-list'},ic.flags.map(text => node('li',{text}))));
    if (summary.release) details.append(help(`发布版本：${summary.release}`));
    return node('section',{class:'card'},details);
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
      const fit = () => { const aspect = Math.max(camera.aspect,.3); const distance = span / (2 * Math.tan(T.MathUtils.degToRad(camera.fov/2))) / Math.min(aspect,1); camera.position.set(distance*.4,-distance*.85,distance*.8); camera.up.set(0,0,1); controls.target.set(0,0,0); controls.update(); camera.updateProjectionMatrix();};
      const resize = () => {const width = Math.max(host.clientWidth,1),height = Math.max(host.clientHeight,1); renderer.setSize(width,height,false); camera.aspect = width / height; camera.updateProjectionMatrix();};
      resize(); fit(); observer = new ResizeObserver(resize); observer.observe(host);
      const raycaster = new T.Raycaster(), mouse = new T.Vector2(); let pointerStart = null;
      const down = event => {pointerStart = [event.clientX,event.clientY];};
      const up = event => {if (!pointerStart || Math.hypot(event.clientX-pointerStart[0],event.clientY-pointerStart[1]) > 6) return; const r = renderer.domElement.getBoundingClientRect(); mouse.set((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1); raycaster.setFromCamera(mouse,camera); const hit = raycaster.intersectObjects(targets)[0]; if (hit) onSelect(hit.object.userData.endpoint.segment_id);};
      renderer.domElement.addEventListener('pointerdown',down); renderer.domElement.addEventListener('pointerup',up);
      const animate = () => {if (disposed) return; controls.update(); renderer.render(scene,camera); frame = requestAnimationFrame(animate);}; animate();
      const result = {
        update(mapping) {
          for (const old of labels.splice(0)) {group.remove(old); const texture = old.material.map; texture?.dispose(); old.material.dispose(); disposables.delete(texture); disposables.delete(old.material);}
          for (const sphere of targets) {const e = sphere.userData.endpoint, name = e.kind === 'inlet' ? 'inlet' : mapping[String(e.segment_id)], color = COLORS[name] || 0x74889b; sphere.material.color.setHex(color); const label = makeLabel(`#${e.segment_id} ${CN[name] || '未命名'}`,`#${color.toString(16).padStart(6,'0')}`); if (label) {label.position.copy(sphere.position).add(new T.Vector3(span*.07,0,span*.055)); group.add(label); labels.push(label);}}
        },
        select(sid) {for (const sphere of targets) sphere.scale.setScalar(Number(sphere.userData.endpoint.segment_id) === Number(sid) ? 1.5 : 1);},
        reset:fit,
        snapshot() {try {renderer.render(scene,camera); const a = node('a',{href:renderer.domElement.toDataURL('image/png'),download:`wss-outlet-confirmation-${state.current}.png`}); document.body.append(a); a.click(); a.remove();} catch (_) {notify('当前浏览器无法保存三维截图，请使用系统截图工具。');}},
        dispose() {disposed = true; cancelAnimationFrame(frame); observer?.disconnect(); controls?.dispose(); renderer.domElement.removeEventListener('pointerdown',down); renderer.domElement.removeEventListener('pointerup',up); for (const item of disposables) item.dispose?.(); renderer.dispose(); renderer.forceContextLoss();}
      };
      return result;
    } catch (error) {disposed = true; cancelAnimationFrame(frame); observer?.disconnect(); controls?.dispose(); for (const item of disposables) item.dispose?.(); renderer?.dispose(); fallback(error.message || '浏览器未启用 WebGL。'); return null;}
  }

  // ---------------------------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------------------------
  state.unread = WB.unreadParse(store.get('wss-unread','[]'));
  try { state.groupsOpen = {...state.groupsOpen, ...JSON.parse(store.get('wss-groups','{}') || '{}')}; } catch (_) {}
  setListMode(store.get('wss-list-mode','jobs'));
  window.addEventListener('beforeunload',() => state.viewer?.dispose());
  setInterval(() => {if (!document.hidden && state.authenticated) {if (!state.eventsLive) pollJob(); if (++state.listTick % 5 === 0) refreshJobs();}},2000);
  window.addEventListener('beforeunload',() => { closeEvents(); closeOwnerEvents(); });
  document.addEventListener('visibilitychange',() => {if (!document.hidden && state.authenticated) {pollJob(); refreshJobs(); if (state.current && WB.unreadRemove(state.unread,state.current)) persistUnread(); if (!state.ownerEvents) openOwnerEvents();}});
  startSession();
})();
