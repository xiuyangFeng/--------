"""Phase-3 lane 1 (PHASE3_LANES.md §3 「第 1 路」): unfinished jobs on the v2 input page.

Audit rows (lanes/p3_audit_workbench.md): #65 stage table and live remaining time, #68 cancel while waiting for a
person, #69 retry of failed jobs, #70 typed error card, #72 input facts, #74 re-pick the inlet, #75 outlet details,
#76 「下一例」 after confirming or signing, #77 provenance.  Every request must be the classic one with the classic
payload (app.js mutate: ``{...payload, version}``).  The ported remaining-time / outlet-reason / next-case rules are
checked against values frozen from the classic ``workbench_core.js`` (2026-09-30), which leaves with S7.
"""
from __future__ import annotations

import json

from tests.test_v2_workspace_js import V2, _run

# Stage-A snapshot of the preview job 20260930_122126_e434ae6d6df2 (outlets to confirm), trimmed to what the page reads.
PROPOSAL = {
    "confidence": 0.8838613308054825, "confirmation_required": True,
    "confidence_reasons": ["右侧髂内/髂外区分置信度 88.4%（score 1.47），请人工核对", "几何间隔置信度只是未校准的代理值", "STL 未提供患者方向，左右语义必须人工核对",
                           "几何置信度代理 88.4% 低于门槛 95.0%", "命名建议自身标记为需要确认", "当前发布包没有经过独立标注集验证的命名置信度 profile"],
    "flags": ["右侧髂内/髂外区分置信度 88.4%（score 1.47），请人工核对"],
    "direction_note": "STL 未提供患者方向；世界 XYZ 不能自动解释为患者左右，请参考原始影像核对。",
    "mapping": {"3": "out-le", "4": "out-li", "5": "out-ri", "6": "out-re"},
    "endpoints": [{"segment_id": 0, "kind": "inlet", "radius_mm": 13.16, "center_mm": [13.7, -121.87, -131.54]},
                  {"segment_id": 3, "kind": "outlet", "radius_mm": 2.78, "center_mm": [-48.85, -147.54, -386.9]},
                  {"segment_id": 4, "kind": "outlet", "radius_mm": 2.85, "center_mm": [-31.13, -137.3, -386.17]},
                  {"segment_id": 5, "kind": "outlet", "radius_mm": 3.16, "center_mm": [26.41, -136.93, -382.88]},
                  {"segment_id": 6, "kind": "outlet", "radius_mm": 3.3, "center_mm": [36.43, -145.17, -380.1]}],
}
OPENINGS = [{"opening_id": 0, "radius_mm": 2.84, "center_mm": [-49.0, -147.4, -387.3], "role": "outlet"},
            {"opening_id": 1, "radius_mm": 2.85, "center_mm": [-31.1, -137.3, -386.2], "role": "outlet"},
            {"opening_id": 2, "radius_mm": 13.42, "center_mm": [13.4, -120.7, -130.9], "role": "inlet"},
            {"opening_id": 3, "radius_mm": 3.16, "center_mm": [26.4, -136.9, -382.9], "role": "outlet"},
            {"opening_id": 4, "radius_mm": 3.3, "center_mm": [36.4, -145.2, -380.1], "role": "outlet"}]
INPUT_CHECK = {"selected_units": "auto", "resolved_units": "mm", "suggested_units": "mm", "unit_confidence": 0.999, "scale_factor": 1.0,
               "raw_bbox_size": [96.67243576049805, 113.49871826171875, 260.7757873535156], "bbox_size_mm": [96.67243576049805, 113.49871826171875, 260.7757873535156],
               "vertices": 11669, "faces": 23184, "area_mm2": 34110.31940034909, "components": 1, "removed_area_fraction": 0.0, "openings": 5,
               "flags": [], "errors": [], "status": "pass", "ok": True}
ETA_CONFIRM = {"status": "awaiting_confirmation", "segment": "B", "basis": "history", "n_history": 4, "remaining_s": 31.2,
               "stages": [{"key": "centerline", "label": "中心线", "expected_s": 3.5, "state": "done", "elapsed_s": 3.5},
                          {"key": "infer", "label": "模型推理", "expected_s": 31.2, "state": "pending"}]}
ETA_RUNNING = {"status": "running", "segment": "B", "basis": "history", "n_history": 12, "faces": 23184, "remaining_s": 26, "precomputed": ["smooth_resample"],
               "stages": [{"key": "ingest", "label": "读取与检查", "expected_s": 0.4, "state": "done", "elapsed_s": 0.3},
                          {"key": "centerline", "label": "中心线", "expected_s": 3.5, "state": "done", "elapsed_s": 3.4},
                          {"key": "smooth_resample", "label": "平滑与重采样", "expected_s": 2.4, "state": "done", "elapsed_s": 0.1},
                          {"key": "features", "label": "点几何特征", "expected_s": 1.4, "state": "running", "elapsed_s": 2.0},
                          {"key": "infer", "label": "模型推理", "expected_s": 20, "state": "pending"},
                          {"key": "report", "label": "报告", "expected_s": 6, "state": "pending"}]}
ETA_QUEUED = {"status": "queued", "segment": "A", "basis": "default", "remaining_s": 49, "queue_ahead": 2, "queue_ahead_s": 95,
              "stages": [{"key": "ingest", "label": "读取与检查", "expected_s": 1, "state": "pending"}, {"key": "centerline", "label": "中心线", "expected_s": 8, "state": "pending"},
                         {"key": "infer", "label": "模型推理", "expected_s": 40, "state": "pending"}]}

FIXTURES = "const PROPOSAL = %s, OPENINGS = %s, IC = %s, ETA_CONFIRM = %s, ETA_RUNNING = %s, ETA_QUEUED = %s;\n" % tuple(
    json.dumps(x, ensure_ascii=False) for x in (PROPOSAL, OPENINGS, INPUT_CHECK, ETA_CONFIRM, ETA_RUNNING, ETA_QUEUED))
HELPERS = r"""
const buttons = (root, text) => walk(root, e => e.tagName === 'BUTTON' && textOf(e) === text);
const dialog = () => byId('ws-dialog');
const toast = () => byId('ws-toast');
const posts = path => calls.filter(c => c === 'POST ' + path).length;
const sent = {};
const capture = (path, reply) => { canned['POST ' + path] = o => { sent[path] = JSON.parse(o.body); return {body: {job: reply}}; }; };
const inspector = () => byClass(app(), 'input-panel')[0];
"""


def run(scenario: str) -> dict:
    return _run(FIXTURES + HELPERS + scenario)


def test_ported_rules_equal_the_classic_workbench_core():
    """etaView / outletReview / nextJob / formatDuration: values frozen from workbench_core.js on 2026-09-30."""
    out = run(r"""
      for (const f of WS) require(f);
      const IN = ns.input;
      const pick = v => v && {status: v.status, headline: v.headline, sub: v.sub, basis: v.basis, progress: v.progress,
        segments: v.segments.map(s => [s.key, s.state, s.width, s.fill, s.overdue, s.precomputed])};
      const waiting = {status: 'running', segment: 'B', basis: 'history', n_history: 3, waiting: true, remaining_s: 25,
        stages: [{key: 'infer', label: '模型推理', expected_s: 20, state: 'running', elapsed_s: 0}, {key: 'report', label: '报告', expected_s: 5, state: 'pending'}]};
      const late = {status: 'running', segment: 'B', basis: 'history', n_history: 9, remaining_s: 3,
        stages: [{key: 'infer', label: '模型推理', expected_s: 10, state: 'running', elapsed_s: 30}, {key: 'report', label: '报告', expected_s: 1, state: 'pending'}]};
      const input = {status: 'awaiting_input', segment: 'A', basis: 'history', n_history: 5, remaining_s: 80, segment_remaining_s: 4,
        stages: [{key: 'ingest', label: '读取与检查', expected_s: 0.4, state: 'done', elapsed_s: 0.4}, {key: 'centerline', label: '中心线', expected_s: 4, state: 'pending'}]};
      const eta = [pick(IN.etaView(ETA_RUNNING, {ageS: 0})), pick(IN.etaView(ETA_QUEUED, {ageS: 7.5})), pick(IN.etaView(ETA_CONFIRM, {ageS: 0})),
        pick(IN.etaView(input, {ageS: 0})), pick(IN.etaView(waiting, {ageS: 0})), pick(IN.etaView(late, {ageS: 12}))];
      const review = IN.outletReview(PROPOSAL, '等待确认出口。原因：左侧髂内/髂外区分置信度 70%；其他说明');
      const auto = IN.outletReview({confidence: 0.99, confirmation_required: false}, '');
      const jobs = [{id: 'a', status: 'awaiting_confirmation', created_at: '2026-09-30T10:00:00+08:00'}, {id: 'b', status: 'awaiting_input', created_at: '2026-09-30T09:00:00+08:00'},
        {id: 'c', status: 'done', created_at: '2026-09-29T09:00:00+08:00', review: {status: 'reviewed'}}, {id: 'd', status: 'done', created_at: '2026-09-30T08:00:00+08:00'},
        {id: 'e', status: 'done', created_at: '2026-09-30T08:00:00+08:00', review: {status: 'reopened'}}, {id: 'f', status: 'failed', created_at: '2026-09-28T08:00:00+08:00'}];
      const next = [IN.nextJob(jobs, 'b', 'confirm'), IN.nextJob(jobs, 'x', 'confirm'), IN.nextJob(jobs, 'd', 'review'), IN.nextJob(jobs, 'x', 'review'), IN.nextJob([], 'x', 'review')].map(j => j && j.id);
      const fmt = [3, 10, 23, 57.4, 57.6, 125, 569, 570, 3600].map(IN.formatDuration);
      done({eta, review, auto, next, fmt, none: IN.etaView(null), stage: [IN.stageRemaining(10, 5), IN.stageRemaining(10, 9), IN.stageRemaining(10, 40)]});
    """)
    assert out["errors"] == [], out["errors"]
    eta = out["eta"]
    # frozen from workbench_core.etaView (same inputs)
    assert eta[0] == {"status": "running", "headline": "预计还需约 25 秒", "sub": "正在：点几何特征（第 4 / 6 步） · 比预计慢",
                      "basis": "按 12 例同类历史耗时估计 · 输入 23,184 个面片", "progress": 24,
                      "segments": [["ingest", "done", 2.468, 100, False, False], ["centerline", "done", 10.251, 100, False, False],
                                   ["smooth_resample", "done", 7.029, 100, False, True], ["features", "running", 4.1, 96, True, False],
                                   ["infer", "pending", 58.578, 0, False, False], ["report", "pending", 17.573, 0, False, False]]}
    assert eta[1]["headline"] == "前面 2 个，约 1 分 30 秒后开始" and eta[1]["sub"] == "预计约 2 分 15 秒后出结果（含排队，不含人工确认出口）"
    assert eta[1]["basis"] == "按默认耗时估计（同类历史少于 3 例）" and [s[2] for s in eta[1]["segments"]] == [2.489, 16.252, 81.26]
    assert eta[2]["headline"] == "确认后约 30 秒出结果" and eta[2]["progress"] == 10
    assert eta[3]["headline"] == "确认后几秒完成中心线提取"
    assert eta[4]["status"] == "queued" and eta[4]["headline"] == "等前一个任务算完后开始" and eta[4]["segments"][0][1] == "pending"
    assert eta[5]["headline"] == "即将完成" and eta[5]["progress"] == 87 and eta[5]["segments"][0][4] is True
    assert out["none"] is None
    assert out["stage"][0] == 5 and abs(out["stage"][1] - 1.7777777777777777) < 1e-12 and abs(out["stage"][2] - 0.4) < 1e-12
    assert out["review"]["headline"] == "自动命名置信度 88%，需要人工核对" and out["review"]["required"] is True
    assert out["review"]["reasons"] == ["右侧髂内与髂外的区分把握度只有 88%，请重点核对右侧的两个出口。", "把握度由几何形态估算，不是经过临床数据校准的概率。",
                                        "STL 文件不带患者方位信息，左右需要对照原始影像确认。", "整体命名把握度 88%，低于自动放行要求的 95%。",
                                        "自动命名在至少一侧的内 / 外区分上不够确定。", "当前模型版本尚未完成出口命名的独立验证，因此每例都请人工确认。",
                                        "左侧髂内与髂外的区分把握度只有 70%，请重点核对左侧的两个出口。", "其他说明"]
    assert out["auto"] == {"headline": "自动命名置信度 99%，已自动通过", "confidence": 0.99, "required": False, "reasons": []}
    assert out["next"] == ["a", "b", "e", "d", None]            # oldest waiting first; reopened counts as not reviewed
    assert out["fmt"] == ["几秒", "约 10 秒", "约 25 秒", "约 55 秒", "约 1 分钟", "约 2 分 5 秒", "约 9 分 30 秒", "约 10 分钟", "约 60 分钟"]


def test_failed_retryable_error_card_retries_with_classic_payload():
    """#69 / #70: a failed job with a typed resource error — title, CPU hint, diagnostic id, admin detail, 「重试」 → /retry {version}."""
    out = run(r"""
      canned['/api/jobs/F'] = {body: {job: jobRecord('F', {status: 'failed', version: 7, a: {input_check: IC, centerline: {openings: OPENINGS}},
        error: {message: 'GPU 显存不足，已改用 CPU 重试。', category: 'resource', retryable: true, retry_hint: 'cpu', diagnostic_id: 'D-20260930-9', admin_detail: 'CUDA out of memory. Tried to allocate 2.00 GiB'}})}};
      canned['/api/v2/jobs/F/inputcheck'] = {body: {status: 'failed', input_check: IC, openings: OPENINGS, mesh: null}};
      canned['/api/jobs/F/geometry'] = {body: {}};
      capture('/api/jobs/F/retry', jobRecord('F', {status: 'queued', version: 8, eta: ETA_QUEUED}));
      await boot();
      await hashTo('#/job/F', 200);
      const p = inspector();
      const title = textOf(byClass(p, 'sec-title')[0]);
      const txt = textOf(p);
      const tip = (walk(p, e => String(e.className).includes('info-tip'))[0] || {}).title;
      const detail = walk(p, e => e.tagName === 'DETAILS')[0];
      const toolbar = textOf(byClass(app(), 'ws-toolbar')[0]);
      fire(buttons(p, '重试')[0], 'click'); await wait(80);
      const after = textOf(inspector()), toastText = textOf(toast());
      // the shell was not polling a failed job: the page follows the retried job itself until it settles again
      canned['/api/jobs/F'] = {body: {job: jobRecord('F', {status: 'failed', version: 10, error: {message: '开口数 6 ≠ 5', category: 'input_geometry', retryable: false}})}};
      await wait(3300);
      const settled = textOf(byClass(inspector(), 'sec-title')[0]);
      const polling = Boolean(shellState().inputView.state.poll);
      done({title, txt, tip, detail: detail && textOf(detail), toolbar, sent: sent['/api/jobs/F/retry'], mode: shellState().mode,
        after, toast: toastText, settled, polling});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["title"] == "计算资源不足"
    assert "GPU 显存不足，已改用 CPU 重试。" in out["txt"] and "改用 CPU 计算" in out["txt"]
    assert "诊断编号 D-20260930-9" in out["txt"]
    assert out["detail"] == "技术细节CUDA out of memory. Tried to allocate 2.00 GiB"
    assert out["tip"] == "显存、内存或磁盘暂时不足，通常稍后重试即可完成。"
    assert "经典工作台" not in out["toolbar"]                     # the classic hand-off link is gone (§3 item 5)
    assert out["sent"] == {"version": 7}                          # classic mutate(job, '/retry') payload
    assert out["mode"] == "input" and "前面 2 个" in out["after"] and "已重新排队" in out["toast"]
    assert out["settled"] == "输入几何无法计算" and out["polling"] is False


def test_failed_not_retryable_locked_and_interrupted():
    out = run(r"""
      canned['/api/jobs/G'] = {body: {job: jobRecord('G', {status: 'failed', a: {input_check: Object.assign({}, IC, {status: 'fail', openings: 6, errors: ['开口数 6 ≠ 5']})},
        error: {message: '输入检查未通过：开口数 6 ≠ 5；需要主动脉入口与四个髂动脉出口，且切口保持开放。', category: 'input_geometry', retryable: false}})}};
      canned['/api/v2/jobs/G/inputcheck'] = {body: {status: 'failed', input_check: null, mesh: null, openings: []}};
      canned['/api/jobs/L'] = {body: {job: jobRecord('L', {status: 'failed', review: {status: 'reviewed'}, error: {message: '计算未完成', category: 'internal', retryable: true, diagnostic_id: 'D-1'}})}};
      canned['/api/jobs/I'] = {body: {job: jobRecord('I', {status: 'interrupted', error: null, detail: '服务重启时任务中断。'})}};
      canned['/api/jobs/C2'] = {body: {job: jobRecord('C2', {status: 'cancelled', error: null, detail: '可以重试以恢复任务。'})}};
      await boot();
      await hashTo('#/job/G', 200);
      const g = {title: textOf(byClass(inspector(), 'sec-title')[0]), retry: buttons(inspector(), '重试').length, txt: textOf(inspector()),
        upload: walk(inspector(), e => e.tagName === 'BUTTON' && textOf(e) === '修改后重新上传' && String(e.className).includes('btn-primary')).length,
        facts: byClass(inspector(), 'tbl-facts').length};
      await hashTo('#/job/L', 200);
      const l = {retry: buttons(inspector(), '重试').length, txt: textOf(inspector()), admin: walk(inspector(), e => e.tagName === 'DETAILS').length};
      await hashTo('#/job/I', 200);
      const i = {title: textOf(byClass(inspector(), 'sec-title')[0]), restore: buttons(inspector(), '恢复任务').length, txt: textOf(inspector())};
      await hashTo('#/job/C2', 200);
      const c = {title: textOf(byClass(inspector(), 'sec-title')[0]), retry: buttons(inspector(), '重试').length, err: byClass(inspector(), 'note-error').length};
      done({g, l, i, c});
    """)
    assert out["errors"] == [], out["errors"]
    g = out["g"]
    assert g["title"] == "输入几何无法计算" and g["retry"] == 0 and g["upload"] == 1
    assert "重试不会得到不同的结果" in g["txt"] and g["facts"] == 1        # facts kept behind 「输入尺寸与面积」
    assert out["l"]["retry"] == 0 and "已复核锁定" in out["l"]["txt"] and "诊断编号 D-1" in out["l"]["txt"] and out["l"]["admin"] == 0
    assert out["i"]["title"] == "任务中断，可恢复" and out["i"]["restore"] == 1 and "服务重启时任务中断。" in out["i"]["txt"]
    assert out["c"]["title"] == "已取消" and out["c"]["retry"] == 1 and out["c"]["err"] == 0


def test_cancel_while_waiting_and_while_running():
    """#68: awaiting_input / awaiting_confirmation (and queued / running) can be cancelled; the dialog says what is kept."""
    out = run(r"""
      canned['/api/jobs/W'] = {body: {job: jobRecord('W', {status: 'awaiting_input', version: 4, a: {input_check: Object.assign({}, IC, {status: 'needs_confirmation'})}})}};
      canned['/api/v2/jobs/W/inputcheck'] = {body: {status: 'awaiting_input', input_check: Object.assign({}, IC, {status: 'needs_confirmation'}), mesh: null, openings: []}};
      capture('/api/jobs/W/cancel', jobRecord('W', {status: 'cancelled', version: 5, detail: '可以重试以恢复任务。'}));
      canned['/api/jobs/Q'] = {body: {job: jobRecord('Q', {status: 'awaiting_confirmation', version: 2, a: {proposal: PROPOSAL, centerline: {openings: OPENINGS}}, eta: ETA_CONFIRM})}};
      canned['/api/v2/jobs/Q/inputcheck'] = {body: {status: 'awaiting_confirmation', input_check: IC, mesh: null, openings: OPENINGS}};
      canned['/api/jobs/Q/geometry'] = {body: {endpoints: PROPOSAL.endpoints}};
      canned['/api/jobs/R2'] = {body: {job: jobRecord('R2', {status: 'running', version: 9, eta: ETA_RUNNING})}};
      canned['/api/jobs/X'] = {body: {job: jobRecord('X', {status: 'running', version: 9, cancel_requested: true, phase: '正在取消', detail: '当前步骤结束后停止；已完成的输入和中心线会保留。', eta: ETA_RUNNING})}};
      await boot();
      await hashTo('#/job/Q', 200);
      const q = buttons(inspector(), '取消任务').length;
      await hashTo('#/job/R2', 200);
      const r = buttons(inspector(), '取消任务').length;
      await hashTo('#/job/X', 200);
      const x = {n: buttons(inspector(), '取消任务').length, head: textOf(byClass(inspector(), 'eta-head')[0])};
      await hashTo('#/job/W', 200);
      fire(buttons(inspector(), '取消任务')[0], 'click'); await wait(20);
      const dlgText = textOf(dialog());
      fire(buttons(dialog(), '取消任务')[0], 'click'); await wait(80);
      done({q, r, x, dlgText, sent: sent['/api/jobs/W/cancel'], after: textOf(byClass(inspector(), 'sec-title')[0]), toast: textOf(toast())});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["q"] == 1 and out["r"] == 1
    assert out["x"] == {"n": 0, "head": "正在取消"}                 # already asked to stop: no second cancel
    assert "不再排队或等待确认" in out["dlgText"] and "不必重新上传" in out["dlgText"]
    assert out["sent"] == {"version": 4}                          # classic cancelJob → mutate(job, '/cancel')
    assert out["after"] == "已取消" and "任务已取消" in out["toast"]


def test_progress_stage_table_ticks_and_repaints_on_newer_estimate():
    """#65: headline + step, segments sized by expected seconds, 「各阶段耗时」 table with the overdue stage, ⓘ basis."""
    out = run(r"""
      canned['/api/jobs/R'] = {body: {job: jobRecord('R', {status: 'running', version: 9, eta: ETA_RUNNING})}};
      await boot();
      await hashTo('#/job/R', 200);
      const p = inspector();
      const head = textOf(byClass(p, 'eta-head')[0]), sub = textOf(byClass(p, 'eta-sub')[0]);
      const segs = byClass(p, 'eta-seg').map(e => [e.className, e.style.flexBasis, e.children[0].style.width]);
      const basis = (walk(p, e => String(e.className).includes('info-tip'))[0] || {}).title;
      const tableHidden = byClass(p, 'tbl-eta')[0].hidden;
      fire(buttons(p, '各阶段耗时')[0], 'click'); await wait(10);
      const rows = walk(byClass(p, 'tbl-eta')[0], e => e.tagName === 'TR').slice(1).map(tr => tr.children.map(textOf));
      // same version, newer estimate: only the progress repaints (the page is not rebuilt)
      const before = byClass(p, 'eta')[0];
      shellState().inputView.update(jobRecord('R', {status: 'running', version: 9, eta: Object.assign({}, ETA_RUNNING, {stages: ETA_RUNNING.stages.map(s => s.key === 'features' ? Object.assign({}, s, {state: 'done', elapsed_s: 2.1}) : s.key === 'infer' ? Object.assign({}, s, {state: 'running', elapsed_s: 1}) : s)})}));
      const same = byClass(inspector(), 'eta')[0] === before;
      done({head, sub, segs, basis, tableHidden, rows, same, head2: textOf(byClass(inspector(), 'eta-head')[0]), sub2: textOf(byClass(inspector(), 'eta-sub')[0]),
        ticker: Boolean(shellState().inputView.state.ticker)});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["head"].startswith("预计还需") and out["sub"].startswith("正在：点几何特征（第 4 / 6 步）") and "比预计慢" in out["sub"]
    assert out["segs"][2] == ["eta-seg done precomputed", "7.029%", "100%"]
    assert out["segs"][3][0] == "eta-seg running overdue" and out["segs"][3][2] == "96%"
    assert out["basis"] == "按 12 例同类历史耗时估计 · 输入 23,184 个面片"
    assert out["tableHidden"] is True
    assert out["rows"][2] == ["平滑与重采样（已预计算）", "2.4 秒", "0.1 秒", "已完成"]
    assert out["rows"][3][0] == "点几何特征" and out["rows"][3][3] == "比预计慢"
    assert out["rows"][4] == ["模型推理", "20 秒", "—", "等待"]
    assert out["same"] is True and out["sub2"].startswith("正在：模型推理（第 5 / 6 步）")
    assert out["ticker"] is True


def test_units_card_facts_flags_reference_and_fail_block():
    """#72: facts table (raw / converted size, openings, area + vertices, pieces, removed area), flags, reference, fail."""
    out = run(r"""
      const ic = Object.assign({}, IC, {status: 'needs_confirmation', selected_units: 'auto', suggested_units: 'cm', raw_bbox_size: [9.667, 11.36, 26.08], bbox_size_mm: null,
        components: 3, removed_area_fraction: 0.00123, flags: ['单位把握度不足，请确认。']});
      canned['/api/jobs/U'] = {body: {job: jobRecord('U', {status: 'awaiting_input', version: 3, params: {units: 'auto', remove_fragments: false}, a: {input_check: ic},
        summary: {reference_assessment: {status: 'review', population: {status: 'pass', percentile: 97.25}}}, eta: {status: 'awaiting_input', segment: 'A', basis: 'history', n_history: 5,
        remaining_s: 80, segment_remaining_s: 14, stages: [{key: 'ingest', label: '读取与检查', expected_s: 0.4, state: 'done'}, {key: 'centerline', label: '中心线', expected_s: 14, state: 'pending'}]}})}};
      canned['/api/v2/jobs/U/inputcheck'] = {body: {status: 'awaiting_input', input_check: ic, mesh: null, openings: []}};
      capture('/api/jobs/U/input', jobRecord('U', {status: 'queued', version: 4}));
      const bad = Object.assign({}, IC, {status: 'fail', openings: 6, errors: ['开口数 6 ≠ 5']});
      canned['/api/jobs/V'] = {body: {job: jobRecord('V', {status: 'awaiting_input', a: {input_check: bad}})}};
      canned['/api/v2/jobs/V/inputcheck'] = {body: {status: 'awaiting_input', input_check: bad, mesh: null, openings: []}};
      await boot();
      await hashTo('#/job/U', 200);
      const p = inspector();
      const facts = walk(byClass(p, 'tbl-facts')[0], e => e.tagName === 'TR').slice(1).map(tr => tr.children.map(textOf));
      const unit = walk(p, e => e.tagName === 'SELECT')[0].value;
      const preview = textOf(walk(p, e => e.attrs && e.attrs['aria-live'] === 'polite')[0]);
      const frag = textOf(byClass(p, 'check')[0]);
      const txt = textOf(p);
      const note = textOf(byClass(p, 'eta-note')[0]);
      const boxes = walk(p, e => e.tagName === 'INPUT' && e.type === 'checkbox');
      boxes[0].checked = true; fire(boxes[0], 'change');
      boxes[boxes.length - 1].checked = true; fire(boxes[boxes.length - 1], 'change');
      fire(buttons(p, '确认并继续')[0], 'click'); await wait(60);
      await hashTo('#/job/V', 200);
      const v = {txt: textOf(inspector()), go: buttons(inspector(), '确认并继续').length, select: walk(inspector(), e => e.tagName === 'SELECT').length};
      done({facts, unit, preview, frag, txt, note, sent: sent['/api/jobs/U/input'], v});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["facts"] == [["原始尺寸", "9.67 × 11.4 × 26.1"], ["换算后", "—"], ["开口数", "5"], ["壁面面积", "341 cm² · 11669 个顶点"],
                            ["连通片", "3"], ["拟删除面积", "0.123%"]]
    assert out["unit"] == "cm" and out["preview"] == "确认后 96.7 × 114 × 261 mm（× 10）"
    assert "0.123%" in out["frag"]
    assert "单位把握度不足，请确认。" in out["txt"] and "超出已声明范围，请复核" in out["txt"] and "同协议经验分位 97.3%" in out["txt"]
    assert out["note"] == "确认后约 15 秒完成中心线提取"
    assert out["sent"] == {"units": "cm", "remove_fragments": True, "acknowledged": True, "version": 3}
    assert "不能通过单位确认继续" in out["v"]["txt"] and "开口数 6 ≠ 5" in out["v"]["txt"] and out["v"]["go"] == 0 and out["v"]["select"] == 0


def test_outlet_details_confirm_and_next_case():
    """#75 / #76: plain-words reasons, 「已修改 N 个出口」, ETA next to the button, world coordinates, 3-D pick → row;
    confirm sends the classic payload, then 「处理下一例」 opens the oldest job still waiting for a person."""
    out = run(r"""
      canned['/api/jobs/Q'] = {body: {job: jobRecord('Q', {status: 'awaiting_confirmation', version: 2, created_at: '2026-09-30T12:00:00+08:00',
        a: {proposal: PROPOSAL, centerline: {openings: OPENINGS}, input_check: IC}, eta: ETA_CONFIRM})}};
      canned['/api/v2/jobs/Q/inputcheck'] = {body: {status: 'awaiting_confirmation', input_check: IC, mesh: null, openings: OPENINGS}};
      canned['/api/jobs/Q/geometry'] = {body: {endpoints: PROPOSAL.endpoints}};
      capture('/api/jobs/Q/confirm', jobRecord('Q', {status: 'queued', version: 3, eta: ETA_QUEUED}));
      canned['/api/jobs'] = {body: {jobs: [jobRecord('Q', {status: 'queued'}), jobRecord('N1', {status: 'awaiting_confirmation', display_name: 'CASE_NEXT', created_at: '2026-09-29T08:00:00+08:00'}),
        jobRecord('N2', {status: 'awaiting_input', created_at: '2026-09-30T08:00:00+08:00'}), jobRecord('A')], total: 4, page: 1, page_size: 100}};
      canned['/api/jobs/N1'] = {body: {job: jobRecord('N1', {status: 'awaiting_confirmation', a: {proposal: PROPOSAL}})}};
      await boot();
      await hashTo('#/job/Q', 250);
      const p = inspector();
      const tag = textOf(byClass(p, 'sec-note')[0]);
      const first = textOf(byClass(p, 'note-warn')[0]);
      const moreBtn = walk(p, e => e.tagName === 'BUTTON' && textOf(e).startsWith('更多原因'))[0];
      const moreHidden = byClass(p, 'reasons')[0].hidden;
      const coords = walk(byClass(p, 'tbl-outlets')[0], e => e.tagName === 'TR').slice(1).map(tr => tr.title);
      const eta = textOf(byClass(p, 'eta-note')[0]);
      // 3-D pick of endpoint #5 selects its row
      shellState().inputView.pick('e5'); await wait(10);
      const sel = walk(byClass(inspector(), 'tbl-outlets')[0], e => e.tagName === 'TR').slice(1).map(tr => String(tr.className).includes('sel'));
      fire(buttons(inspector(), '右侧内 / 外')[0], 'click'); await wait(20);
      const changed = textOf(byClass(inspector(), 'mapping-changes')[0]);
      const ack = walk(inspector(), e => e.tagName === 'INPUT' && e.type === 'checkbox')[0];
      ack.checked = true; fire(ack, 'change'); await wait(10);
      calls.length = 0;
      fire(buttons(inspector(), '确认出口并开始预测')[0], 'click'); await wait(150);
      const t = toast(), action = buttons(t, '处理下一例')[0];
      const toastText = textOf(t);
      fire(action, 'click'); await wait(150);
      done({tag, first, more: moreBtn && textOf(moreBtn), moreHidden, coords, eta, sel, changed, sent: sent['/api/jobs/Q/confirm'], toastText, hash: location.hash,
        listCall: calls.filter(c => c.startsWith('GET /api/jobs') && !c.includes('/api/jobs/')).length});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["tag"] == "自动命名置信度 88%，需要人工核对"
    assert out["first"] == "右侧髂内与髂外的区分把握度只有 88%，请重点核对右侧的两个出口。"
    assert out["more"] == "更多原因 · 5" and out["moreHidden"] is True
    assert out["coords"][1] == "世界坐标 -48.9, -147.5, -386.9 mm"
    assert out["eta"] == "确认后约 30 秒出结果"
    assert out["sel"] == [False, False, False, True, False]
    assert out["changed"] == "已修改 2 个出口：#5 → 右髂外；#6 → 右髂内"
    assert out["sent"] == {"mapping": {"3": "out-le", "4": "out-li", "5": "out-re", "6": "out-ri"}, "acknowledged": True, "version": 2}
    assert out["listCall"] >= 1
    assert out["toastText"].startswith("出口已确认，开始预测。还有等待确认的任务：CASE_NEXT。")
    assert out["hash"] == "#/job/N1"


def test_repick_inlet_from_list_or_3d_sends_inlet():
    """#74: 「入口选错了？」 → pick an opening (list or 3-D) → 「重提中心线」 = /confirm {mapping, inlet, acknowledged, version}."""
    out = run(r"""
      canned['/api/jobs/Q'] = {body: {job: jobRecord('Q', {status: 'awaiting_confirmation', version: 2, params: {inlet: null}, a: {proposal: PROPOSAL, centerline: {openings: OPENINGS}}})}};
      canned['/api/v2/jobs/Q/inputcheck'] = {body: {status: 'awaiting_confirmation', input_check: IC, mesh: null, openings: OPENINGS}};
      canned['/api/jobs/Q/geometry'] = {body: {endpoints: PROPOSAL.endpoints}};
      capture('/api/jobs/Q/confirm', jobRecord('Q', {status: 'queued', version: 3, params: {inlet: 3}}));
      await boot();
      await hashTo('#/job/Q', 250);
      const sec = () => byClass(inspector(), 'sec-inlet')[0];
      const tag = textOf(byClass(sec(), 'sec-note')[0]);
      const bodyHidden = byClass(sec(), 'inlet-body')[0].hidden;
      fire(buttons(sec(), '入口选错了？')[0], 'click'); await wait(10);
      const sel = walk(sec(), e => e.tagName === 'SELECT')[0];
      const options = sel.children.map(textOf);
      const redo = buttons(sec(), '重提中心线')[0];
      sel.value = '2'; fire(sel, 'change');
      const sameDisabled = redo.disabled;                          // the current inlet: nothing to redo
      shellState().inputView.pick('o3'); await wait(10);            // 3-D click on opening 4
      const picked = [sel.value, redo.disabled, shellState().inputView.state.inletChoice];
      fire(redo, 'click'); await wait(80);
      done({tag, bodyHidden, options, sameDisabled, picked, sent: sent['/api/jobs/Q/confirm'], toast: textOf(toast()), mode: shellState().mode});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["tag"] == "开口 3 · r 13.4 mm" and out["bodyHidden"] is True
    assert out["options"] == ["选择作为入口的开口", "开口 1 · r 2.84 mm", "开口 2 · r 2.85 mm", "开口 3 · r 13.4 mm · 当前入口", "开口 4 · r 3.16 mm", "开口 5 · r 3.30 mm"]
    assert out["sameDisabled"] is True
    assert out["picked"] == ["3", False, "3"]
    assert out["sent"] == {"mapping": {"3": "out-le", "4": "out-li", "5": "out-ri", "6": "out-re"}, "inlet": 3, "acknowledged": True, "version": 2}
    assert "已改用开口 4 作为入口" in out["toast"]


def test_provenance_on_input_page_and_tools_page():
    """#77: companions (pending / failed / created), companion_of, reused_from, source_job_id."""
    out = run(r"""
      canned['/api/jobs/P'] = {body: {job: jobRecord('P', {status: 'awaiting_confirmation', a: {proposal: PROPOSAL}, companions: [
        {release_id: 'PF6_VF6_peak_3seed_20260920', job_id: null}, {release_id: 'X_missing', job_id: null, error: '发布包不存在'}]})}};
      canned['/api/v2/jobs/P/inputcheck'] = {body: {status: 'awaiting_confirmation', input_check: IC, mesh: null, openings: []}};
      canned['/api/jobs/P/geometry'] = {body: {}};
      canned['/api/jobs/K'] = {body: {job: jobRecord('K', {status: 'queued', companion_of: 'A', eta: ETA_QUEUED})}};
      canned['/api/jobs/S'] = {body: {job: jobRecord('S', {status: 'queued', reused_from: 'B'})}};
      await boot();
      await hashTo('#/job/P', 200);
      const p = byClass(inspector(), 'prov-list')[0].children.map(li => [textOf(li), String(li.className)]);
      await hashTo('#/job/K', 200);
      const k = textOf(byClass(inspector(), 'prov-list')[0]);
      fire(buttons(byClass(inspector(), 'prov-list')[0], '打开')[0], 'click'); await wait(200);
      const hashK = location.hash;
      await hashTo('#/job/S', 200);
      const s = textOf(byClass(inspector(), 'prov-list')[0]);
      // a finished result made by a model rerun: the tools page says where it came from
      serve('RR', {source_job_id: 'A'});
      await hashTo('#/job/RR', 200);
      fire(walk(app(), e => e.dataset && e.dataset.tab === 'tools')[0], 'click'); await wait(30);
      const tools = byClass(app(), 'sec-prov').map(textOf);
      done({p, k, hashK, s, tools});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["p"][0][0] == "同时预测「体内压力与速度」：确认出口后自动创建"
    assert out["p"][1][0] == "同时预测「结果」：未能创建（发布包不存在）" and "warn-text" in out["p"][1][1]
    assert out["k"] == "与「周期指标 TAWSS · OSI」同时上传，沿用它的中心线和出口确认打开" and out["hashK"] == "#/job/A"
    assert out["s"].startswith("复用已有结果的中心线和出口确认，只重新预测")
    assert out["tools"] == ["来源换模型重跑，沿用来源结果的中心线和出口确认打开来源"]


def test_review_approval_offers_the_next_unreviewed_result():
    """#76 (signing): an approval through the service call offers 「下一例待复核」; a reopen does not."""
    out = run(r"""
      canned['POST /api/jobs/A/review'] = o => ({body: {job: jobRecord('A', {review: {status: JSON.parse(o.body).decision === 'approve' ? 'reviewed' : 'reopened'}})}});
      canned['/api/jobs'] = {body: {jobs: [jobRecord('A', {review: {status: 'reviewed'}}), jobRecord('B', {display_name: 'CASE_B2', created_at: '2026-09-28T08:00:00+08:00'})], total: 2}};
      await boot();
      const wrapped = ns.api.review.__p3Next === true && ns.input.watchReview() === false;   // idempotent
      calls.length = 0;
      await ns.api.review('A', {decision: 'reopen', reviewer: 'x', note: 'n', version: 3}); await wait(60);
      const afterReopen = calls.filter(c => c.startsWith('GET /api/jobs')).length;
      await ns.api.review('A', {decision: 'approve', reviewer: 'x', note: '', version: 3}); await wait(80);
      const t = toast();
      done({wrapped, afterReopen, text: textOf(t), action: buttons(t, '下一例待复核').length});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["wrapped"] is True and out["afterReopen"] == 0
    assert out["text"].startswith("技术复核已记录，结果已锁定。还有未复核的结果：CASE_B2。") and out["action"] == 1


def test_input_files_parse_and_are_bundled():
    import shutil
    import subprocess
    if not shutil.which("node"):
        import pytest
        pytest.skip("Node is required")
    subprocess.run(["node", "--check", str(V2 / "ws_input.js")], check=True, capture_output=True)
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    assert "ws_input.js" in bundle["scripts"] and "v2_input.css" in bundle["styles"] and "ws_input.js" in bundle["offline_exclude"]
    css = (V2 / "v2_input.css").read_text(encoding="utf-8")
    assert ".eta-fill" in css and ".admin-detail" in css and ".input-stage-tools" in css
    assert "经典工作台中处理" not in (V2 / "ws_shell.js").read_text(encoding="utf-8")
