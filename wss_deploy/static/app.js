/* WSS local workbench. Dynamic data is inserted through textContent, never HTML. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const CN = {inlet:'主动脉入口', 'out-le':'左髂外', 'out-li':'左髂内', 'out-re':'右髂外', 'out-ri':'右髂内'};
  const COLORS = {inlet:0x2569aa, 'out-le':0x19837a, 'out-li':0x6a9b35, 'out-re':0xd08838, 'out-ri':0xa16cb3};
  const STATUS = {queued:'排队中', queued_A:'排队中', queued_B:'排队中', running:'计算中', running_A:'正在提取中心线', running_B:'正在计算 WSS', awaiting_input:'待确认输入', awaiting_confirmation:'待确认出口', awaiting_outlets:'待确认出口', done:'已完成', failed:'失败', cancelled:'已取消', interrupted:'已中断', cancelling:'正在取消'};
  const PHASE = {ingest:'检查壁面与尺寸', input:'检查壁面与尺寸', centerline:'提取中心线', naming:'生成出口建议', geometry:'构建几何特征', smooth_resample:'平滑与重采样', features:'构建几何特征', inference:'预测壁面 WSS', inference_5_models:'预测壁面 WSS', report:'生成报告与导出文件', export:'生成报告与导出文件', metrics:'汇总 WSS 统计', metrics_and_export:'汇总统计并导出', queued:'等待计算资源', cancelled:'任务已取消', interrupted:'服务中断，等待恢复'};
  const TIMER_NAMES = {ingest:'输入检查', centerline:'中心线提取', smooth_resample:'平滑与重采样', features:'几何特征', inference_5_models:'五模型预测', metrics:'指标统计', interpolation:'壁面插值', metrics_and_export:'统计与导出', export:'文件导出', report:'HTML 报告', total:'计算总耗时'};
  const state = {csrf:'',authenticated:false,current:null,job:null,jobs:[],renderKey:null,mapping:{},viewer:null,busy:false,polling:false,listBusy:false,listTick:0,sequence:0};
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
  const badge = status => node('span', {class:`badge ${Object.hasOwn(STATUS,status) ? status : ''}`, text:STATUS[status] || status || '等待'});
  const button = (text, onclick, cls = '') => node('button', {type:'button',class:cls,text,onclick});
  const fact = (label,value,note) => node('div',{class:'fact'},node('span',{class:'fact-label',text:label}),node('strong',{class:'fact-value',text:value}),note && node('small',{class:'fact-note',text:note}));
  const help = text => node('p',{class:'help',text});
  const callout = (text,kind = '') => node('div',{class:`callout ${kind}`,text});
  const getA = job => job.a || job.stage_a || {};
  const jobUrl = (job, suffix = '') => `/api/jobs/${encodeURIComponent(job.id)}${suffix}`;
  const clearNotice = () => { $('notice').hidden = true; };
  const notify = (message,error = true) => { const el = $('notice'); el.textContent = message; el.className = `notice${error ? ' error' : ''}`; el.hidden = false; };
  const errorText = error => typeof error === 'string' ? error : error?.message || '请求未完成，请稍后重试。';
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
      const error = new Error(errorText(result.error || result.message)); error.status = response.status;
      if (response.status === 401) { state.authenticated = false; $('login-panel').hidden = false; $('workspace').hidden = true; setConnection('会话已过期，请重新连接'); }
      throw error;
    }
    return result;
  }
  function setConnection(text, online = false) { $('connection').textContent = text; $('connection').className = `connection${online ? ' online' : ''}`; }
  async function startSession() {
    try {
      const session = await request('/api/session');
      state.csrf = session.csrf_token || ''; state.authenticated = Boolean(session.authenticated);
      $('login-panel').hidden = state.authenticated; $('workspace').hidden = !state.authenticated;
      if (state.authenticated) { setConnection(session.shared ? '共享服务 · 已连接' : '本机服务 · 已连接',true); $('upload-button').disabled = false; await refreshJobs(); }
      else setConnection('等待连接');
    } catch (error) { notify(error.message); setConnection('连接失败 · 请刷新重试'); }
  }
  $('login-form').addEventListener('submit', async event => {
    event.preventDefault(); const submit = event.currentTarget.querySelector('button'); submit.disabled = true; clearNotice();
    try { await request('/api/session',{method:'POST',body:{token:$('access-token').value}}); $('access-token').value = ''; await startSession(); }
    catch (error) { notify(error.message); } finally { submit.disabled = false; }
  });
  $('stl-file').addEventListener('change', () => { $('file-name').textContent = $('stl-file').files[0]?.name || '1 个入口 · 4 个髂动脉出口'; });
  $('upload-form').addEventListener('submit', async event => {
    event.preventDefault(); if (state.busy) return; clearNotice();
    const form = event.currentTarget; const file = $('stl-file').files[0]; if (!file) return;
    if (!/\.stl$/i.test(file.name)) { notify('请选择 .stl 格式的血管壁面文件。'); return; }
    if (!file.size) { notify('STL 文件为空，请重新选择。'); return; }
    state.busy = true; const submit = $('upload-button'); submit.disabled = true; submit.textContent = '正在上传…';
    try {
      const data = new FormData(form); data.set('remove_fragments','false');
      const result = await request('/api/jobs',{method:'POST',body:data,timeout:120000});
      const job = result.job || result; const id = job.id || result.job_id;
      if (!id) throw new Error('服务未返回任务编号，请刷新任务列表确认上传结果。');
      await selectJob(id); await refreshJobs();
      if (window.matchMedia('(max-width:760px)').matches) $('detail').scrollIntoView({behavior:'smooth',block:'start'});
    } catch (error) { notify(error.message); } finally { state.busy = false; submit.disabled = false; submit.textContent = '上传并检查'; }
  });
  $('refresh-jobs').addEventListener('click', async () => { clearNotice(); await refreshJobs(true); await pollJob(true); });
  async function refreshJobs(showError = false) {
    if (!state.authenticated || state.listBusy) return;
    state.listBusy = true;
    try {
      const result = await request('/api/jobs'); state.jobs = Array.isArray(result) ? result : result.jobs || [];
      const list = $('jobs-list'); list.replaceChildren(); $('jobs-empty').hidden = state.jobs.length > 0;
      for (const job of state.jobs) {
        const b = button('', () => selectJob(job.id),'job-button'); b.setAttribute('aria-current',String(job.id === state.current));
        b.append(node('span',{class:'job-title'},node('span',{class:'job-name',text:job.case_id || '匿名病例'}),badge(job.status)),node('span',{class:'job-time',text:job.created_at || job.id}));
        list.append(node('li',{},b));
      }
    } catch (error) { if (showError) notify(error.message); setConnection('连接暂时中断 · 自动重试'); }
    finally { state.listBusy = false; }
  }
  async function selectJob(id) {
    state.current = id; state.sequence++; state.renderKey = null; clearNotice();
    for (const [i,item] of [...$('jobs-list').children].entries()) item.querySelector('button')?.setAttribute('aria-current',String(state.jobs[i]?.id === id));
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
      await request(jobUrl(job,suffix),{method:'POST',body:{...payload,version:job.version}});
      state.renderKey = null; await pollJob(true); await refreshJobs();
    } catch (error) {
      notify(error.status === 409 ? `任务状态已更新，请核对刷新后的内容再操作。${error.message}` : error.message);
      if (error.status === 409) { state.renderKey = null; await pollJob(true); }
    } finally { state.busy = false; controls.forEach((el,i) => {if (el.isConnected) el.disabled = previous[i];}); }
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
    const heading = node('section',{class:'card'},node('div',{class:'job-heading'},node('h1',{text:job.case_id || '匿名病例'}),badge(job.status)),node('p',{class:'job-id',text:`任务 ${job.id}`}));
    const idx = progressIndex(job);
    heading.append(node('ol',{class:'stepper','aria-label':'预测步骤'},['上传 STL','确认输入','确认出口','计算 WSS','查看结果'].map((label,i) => node('li',{class:i < idx ? 'complete' : i === idx ? 'current' : '',text:label,'aria-current':i === idx ? 'step' : null}))));
    const statusLine = node('div',{class:'status-line'});
    if (['running','running_A','running_B','queued','queued_A','queued_B','cancelling'].includes(job.status)) statusLine.append(node('span',{class:'spinner','aria-hidden':'true'}));
    statusLine.append(node('p',{},node('strong',{id:'job-phase'}),node('span',{id:'job-live-detail',class:'status-detail'})));
    if (['queued','queued_A','queued_B','running','running_A','running_B','awaiting_input','awaiting_confirmation','awaiting_outlets'].includes(job.status)) statusLine.append(button('取消任务',() => mutate(job,'/cancel'),'danger'));
    heading.append(statusLine); detail.append(heading); updateLiveStatus(job);
    if (job.error) {
      const errorCard = node('section',{class:'card'},node('h2',{text:job.status === 'interrupted' ? '任务中断，可恢复' : '本次计算未完成'}),callout(errorText(job.error),'error'));
      if (job.error.diagnostic_id) errorCard.append(node('p',{class:'diagnostic',text:`诊断编号：${job.error.diagnostic_id}`}));
      errorCard.append(help('检查下方输入信息后重试。已有的有效中心线与特征将按服务端校验结果复用。')); detail.append(errorCard);
    }
    if (['failed','cancelled','interrupted'].includes(job.status)) {
      const recovery = node('section',{class:'card'},node('h2',{text:job.status === 'interrupted' ? '恢复这个任务' : '继续处理'}),help(job.status === 'cancelled' ? '任务已停止。恢复时会重新检查输入和可复用的中间结果。' : '修正输入或恢复任务后，将从可用的阶段继续。'));
      recovery.append(node('div',{class:'actions'},button(job.status === 'interrupted' ? '恢复任务' : '重试任务',() => mutate(job,'/retry'),'primary'))); detail.append(recovery);
    }
    const a = getA(job);
    if (job.status === 'done') detail.append(resultCard(job));
    else if ((job.status === 'awaiting_confirmation' || job.status === 'awaiting_outlets') && a.proposal) detail.append(outletCard(job));
    else if (job.status === 'awaiting_input' || (['failed','interrupted'].includes(job.status) && a.input_check?.status === 'needs_confirmation')) detail.append(inputCard(job));
    else if (['running','running_A','running_B','queued','queued_A','queued_B'].includes(job.status)) {
      detail.append(node('section',{class:'card'},node('h2',{text:job.stage === 'B' ? '正在生成壁面 WSS 报告' : '正在准备几何与中心线'}),node('p',{class:'muted',text:job.stage === 'B' ? '完成后将显示全场 p99、最大值位置和可交互的三维报告。' : '几何检查通过后提取中心线，并展示需要核对的入口和四个出口。'}),help('可以切换到其他任务；任务状态保存在服务端。取消计算后，正在执行的步骤会在安全停止点结束。')));
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
    if (elapsed != null) detail.push(`${queued || waitingConfirmation ? '已等待' : '已计算'} ${duration(elapsed)}`);
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
    const ic = getA(job).input_check || {}, params = job.params || {};
    const card = node('section',{class:'card'},node('p',{class:'eyebrow',text:'步骤 02'}),node('h2',{text:'确认几何尺寸与处理方式'}));
    card.append(inputFacts(ic));
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
    const confirm = button(override ? '采用修改后的命名并重算' : '确认出口并计算 WSS',() => mutate(job,'/confirm',{mapping:{...state.mapping},acknowledged:true,...(override ? {override:true,stage:'B'} : {})}),'primary');
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
    // Mount after this card is attached, so the renderer receives its real width.
    requestAnimationFrame(() => {
      if (!viewerHost.isConnected) return;
      state.viewer = createViewer(viewerHost,a,selectEndpoint);
      if (!state.viewer) {reset.disabled = true; snapshot.disabled = true;} else state.viewer.update(state.mapping);
    });
    return card;
  }
  function resultCard(job) {
    const summary = job.summary || {}, peak = summary.peak || {}, times = summary.timing_s || {};
    const card = node('section',{class:'card'},node('p',{class:'eyebrow',text:'预测完成'}),node('h2',{text:'壁面 WSS 结果已就绪'}));
    const automatic = [...(job.mapping_history || [])].reverse().find(item => item.source === 'automatic_high_confidence');
    if (automatic) card.append(callout(`出口命名已自动确认（最低置信度 ${fmt(Number(automatic.confidence) * 100,1)}%）。仍可打开下方入口人工复核并重算。`,'success'));
    card.append(node('div',{class:'result-cards'},fact('全场 p99',`${fmt(peak.p99_pa,2)} Pa`,'预测点云的第 99 百分位'),fact('全场最大值',`${fmt(peak.max_pa,2)} Pa`,peak.branch ? `最大值位置：${peak.branch}` : '位置见三维报告'),fact('计算耗时',duration(times.total),'不包含排队与人工确认')));
    const link = (text,path,cls = '') => node('a',{class:`button ${cls}`,href:jobUrl(job,path),target:'_blank',rel:'noopener',text});
    const actions = [link('打开三维报告','/report','primary'),link('下载壁面 VTP','/files/wall_wss.vtp'),link('下载点云 CSV','/files/points_wss.csv')];
    if (getA(job).proposal) actions.push(button('检查 / 修改出口命名并重算',() => {
      if (card.parentElement?.querySelector('.override-outlet-card')) return;
      const editor = outletCard(job,{override:true});
      card.parentElement?.append(editor);
      editor.scrollIntoView({behavior:'smooth',block:'start'});
    }));
    card.append(node('div',{class:'actions'},actions));
    card.append(help('报告可旋转查看整段壁面、查看最大值位置和分支统计，并导出当前视角截图。p99 是统计量，不对应单一解剖位置。'));
    card.append(node('div',{class:'file-links'},node('a',{href:jobUrl(job,'/files/summary.json'),target:'_blank',rel:'noopener',text:'查看完整统计与参数 JSON'})));
    return card;
  }
  function processCard(job) {
    const a = getA(job), ic = a.input_check || {}, summary = job.summary || {}, timing = summary.timing_s || a.timing_s || {};
    const details = node('details',{},node('summary',{text:'查看输入检查与计算过程'}),inputFacts(ic));
    const info = node('dl',{class:'timing-list'});
    const append = (label,value) => {info.append(node('dt',{text:label}),node('dd',{text:value}));};
    const displayUnit = ic.selected_units === 'auto' ? `${ic.resolved_units || ic.suggested_units || '—'}（自动）` : (ic.selected_units || ic.unit || '—');
    append('原始单位',displayUnit); append('换算倍数',`× ${ic.scale_factor || 1}`);
    if (Number.isFinite(Number(ic.unit_confidence))) append('单位自动判定置信度',`${fmt(Number(ic.unit_confidence) * 100,1)}%`);
    if (a.centerline) {append('中心线检查',a.centerline.hard_pass ? '通过' : '未通过'); append('中心线端点 / 分叉',`${a.centerline.topology?.endpoints ?? '—'} / ${a.centerline.topology?.junctions ?? '—'}`);}
    for (const [key,value] of Object.entries(timing)) if (typeof value === 'number') append(TIMER_NAMES[key] || key,`${fmt(value,2)} 秒`);
    const queue = job.timing?.queue_seconds ?? job.queue_seconds ?? job.elapsed?.queue;
    const manual = job.timing?.confirmation_seconds ?? job.confirmation_seconds ?? job.elapsed?.confirmation;
    if (queue != null) append('排队等待',duration(queue));
    if (manual != null) append('人工确认',duration(manual));
    if (summary.device) append('计算设备',`${summary.device}${summary.gpu ? ` · ${summary.gpu}` : ''}`);
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
  window.addEventListener('beforeunload',() => state.viewer?.dispose());
  setInterval(() => {if (!document.hidden && state.authenticated) {pollJob(); if (++state.listTick % 3 === 0) refreshJobs();}},2000);
  document.addEventListener('visibilitychange',() => {if (!document.hidden && state.authenticated) {pollJob(); refreshJobs();}});
  startSession();
})();
