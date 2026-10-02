"""Phase 3 lane 2 (工作台) of the workspace v2 (PHASE3_LANES.md §3 「第 2 路」).

Pure parts: the remaining-time rules ported from the classic workbench_core (etaView / etaRowSummary /
formatDuration / stageRemaining — fixed values, and the classic file itself while it still exists), the case facts
(「补跑缺少的结果」, lumen diameter, tags, 「最近完成」) and the queue's value ranges (classic cohortFilter).
Shell parts on the workspace harness (stub DOM, canned service): the login page (show / hide, Caps Lock, 记住用户名,
429 countdown, 503), the home (三步上手, 最近完成, case-card diameter / tags / 补跑, live remaining time in the rail and
the 进行中 card), the task rows (「⋯」 menu, remaining time, diameter, 批量出图), the queue (已重新打开, ranges), the
maintenance banner, and the persistent thumbnail cache (IndexedDB stub).
"""
from __future__ import annotations

import json
import subprocess

from tests.test_v2_workspace_js import _HARNESS, V2, _need_node
from wss_deploy.paths import STATIC_DIR

LANE_FILES = ["ws_rail.js", "ws_admin.js", "ws_thumbs.js", "ws_upload.js", "ws_shell.js"]
WS = ["ws_icons.js", "ws_ui.js", "ws_store.js", "ws_api.js", "ws_rail.js", "ws_admin.js", "ws_overview.js", "ws_lens.js", "ws_detail.js", "ws_unroll.js",
      "ws_profile.js", "ws_bookmarks.js", "ws_questions.js", "ws_compare.js", "ws_upload.js", "ws_input.js", "ws_export.js", "ws_batch.js", "ws_shell.js", "ws_main.js"]
# The classic workbench_core.js (866cf02, deleted in S7) on the inputs below: etaView (picked fields) and
# etaRowSummary for every ETAS case at age 0 / 20 s, formatDuration, cohortFilter.  P4 lane A (F6): the comparison
# used to run only while the file existed, so after S7 it silently stopped; the values are frozen here instead.
CLASSIC_ETA = {
    'running@0': {'view': {'status': 'running', 'progress': 47, 'headline': '预计还需约 1 分 20 秒', 'remaining': 80, 'cur': 1, 'segs': [['done', 20, 100, False], ['running', 66.667, 40, False], ['pending', 13.333, 0, False]]}, 'row': {'pct': 47, 'text': '还需约 1 分 20 秒', 'state': 'running'}},
    'running@20': {'view': {'status': 'running', 'progress': 60, 'headline': '预计还需约 1 分钟', 'remaining': 60, 'cur': 1, 'segs': [['done', 20, 100, False], ['running', 66.667, 60, False], ['pending', 13.333, 0, False]]}, 'row': {'pct': 60, 'text': '还需约 1 分钟', 'state': 'running'}},
    'queued@0': {'view': {'status': 'queued', 'progress': 0, 'headline': '前面 2 个，约 1 分 30 秒后开始', 'remaining': 130, 'cur': -1, 'segs': [['pending', 23.077, 0, False], ['pending', 76.923, 0, False]]}, 'row': {'pct': None, 'text': '前面 2 个', 'state': 'queued'}},
    'queued@20': {'view': {'status': 'queued', 'progress': 0, 'headline': '前面 2 个，约 1 分 10 秒后开始', 'remaining': 130, 'cur': -1, 'segs': [['pending', 23.077, 0, False], ['pending', 76.923, 0, False]]}, 'row': {'pct': None, 'text': '前面 2 个', 'state': 'queued'}},
    'waiting@0': {'view': {'status': 'queued', 'progress': 23, 'headline': '等前一个任务算完后开始', 'remaining': 100, 'cur': -1, 'segs': [['done', 23.077, 100, False], ['pending', 76.923, 0, False]]}, 'row': {'pct': None, 'text': '等待前一个任务', 'state': 'queued'}},
    'waiting@20': {'view': {'status': 'queued', 'progress': 23, 'headline': '等前一个任务算完后开始', 'remaining': 100, 'cur': -1, 'segs': [['done', 23.077, 100, False], ['pending', 76.923, 0, False]]}, 'row': {'pct': None, 'text': '等待前一个任务', 'state': 'queued'}},
    'confirm@0': {'view': {'status': 'awaiting_confirmation', 'progress': 33, 'headline': '确认后约 1 分 15 秒出结果', 'remaining': 75, 'cur': -1, 'segs': [['done', 33.333, 100, False], ['pending', 66.667, 0, False]]}, 'row': {'pct': None, 'text': '确认后约 1 分 15 秒', 'state': 'waiting'}},
    'confirm@20': {'view': {'status': 'awaiting_confirmation', 'progress': 33, 'headline': '确认后约 1 分 15 秒出结果', 'remaining': 75, 'cur': -1, 'segs': [['done', 33.333, 100, False], ['pending', 66.667, 0, False]]}, 'row': {'pct': None, 'text': '确认后约 1 分 15 秒', 'state': 'waiting'}},
    'overdue@0': {'view': {'status': 'running', 'progress': 99, 'headline': '即将完成', 'remaining': 0.533333, 'cur': 1, 'segs': [['done', 75, 100, False], ['running', 25, 96, True]]}, 'row': {'pct': 99, 'text': '即将完成', 'state': 'running'}},
    'overdue@20': {'view': {'status': 'running', 'progress': 99, 'headline': '即将完成', 'remaining': 0.32, 'cur': 1, 'segs': [['done', 75, 100, False], ['running', 25, 96, True]]}, 'row': {'pct': 99, 'text': '即将完成', 'state': 'running'}},
    'input@0': {'view': {'status': 'awaiting_input', 'progress': 0, 'headline': '确认后约 40 秒完成中心线提取', 'remaining': 200, 'cur': -1, 'segs': [['pending', 100, 0, False]]}, 'row': None},
    'input@20': {'view': {'status': 'awaiting_input', 'progress': 0, 'headline': '确认后约 40 秒完成中心线提取', 'remaining': 200, 'cur': -1, 'segs': [['pending', 100, 0, False]]}, 'row': None},
    'segA@0': {'view': {'status': 'running', 'progress': 6, 'headline': '预计还需约 2 分 30 秒（不含人工确认出口）', 'remaining': 150, 'cur': 0, 'segs': [['running', 37.5, 16.7, False], ['pending', 62.5, 0, False]]}, 'row': {'pct': 6, 'text': '还需约 2 分 30 秒', 'state': 'running'}},
    'segA@20': {'view': {'status': 'running', 'progress': 19, 'headline': '预计还需约 2 分 10 秒（不含人工确认出口）', 'remaining': 130, 'cur': 0, 'segs': [['running', 37.5, 50, False], ['pending', 62.5, 0, False]]}, 'row': {'pct': 19, 'text': '还需约 2 分 10 秒', 'state': 'running'}},
    'empty@0': {'view': None, 'row': None},
    'empty@20': {'view': None, 'row': None}}
CLASSIC_FMT = ['几秒', '约 10 秒', '约 45 秒', '约 1 分钟', '约 1 分钟', '约 2 分 5 秒', '约 9 分 30 秒', '约 10 分钟', '约 50 分钟', '', '']
CLASSIC_COHORT_LOWER = [['b', 'd'], ['b', 'd'], ['b', 'd'], ['c'], ['b', 'c'], ['b']]
CLASSIC_COHORT_REOPENED = ['b']

_EXTRA = r"""
const urls = [], bodies = {};
const baseFetch = global.fetch;
global.fetch = async (url, opts) => {
  urls.push((opts && opts.method || 'GET') + ' ' + String(url));
  if (opts && opts.body && typeof opts.body === 'string') bodies[(opts.method || 'GET') + ' ' + String(url).split('?')[0]] = JSON.parse(opts.body);
  return baseFetch(url, opts);
};
const stage = (key, e, el, state) => ({key, label: key.toUpperCase(), expected_s: e, elapsed_s: el, state});
const menuItems = () => walk(byId('ws-menu'), e => String(e.className).includes('menu-item'));
const menuItem = label => menuItems().filter(e => textOf(byClass(e, 'menu-label')[0]) === label)[0];
const button = (root, label) => walk(root, e => e.tagName === 'BUTTON' && textOf(e) === label)[0];
"""


def _run(scenario: str) -> dict:
    _need_node()
    files = [str(STATIC_DIR / "report_common.js")] + [str(V2 / f) for f in WS]
    program = (_HARNESS.replace("__WS__", json.dumps(files)).replace("__OFFLINE__", "false") + _EXTRA
               + "\n(async () => {\n" + scenario + "\n})().catch(e => { errors.push('scenario: ' + (e && e.stack || e)); done({}); });\n")
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise AssertionError(result.stderr[-4000:])
    lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
    assert lines, result.stdout[-2000:] + result.stderr[-2000:]
    return json.loads(lines[-1])


def _pure(script: str) -> dict:
    _need_node()
    prelude = ("global.WSSV2 = {};\n" + "".join("require(" + json.dumps(str(V2 / f)) + ");\n" for f in ["ws_icons.js", "ws_ui.js", "ws_rail.js", "ws_admin.js"])
               + "const ns = global.WSSV2, R = ns.rail, AD = ns.admin;\n"
               + "const out = x => console.log(JSON.stringify(x));\n")
    result = subprocess.run(["node", "-e", prelude + "(async () => {\n" + script + "\n})().catch(e => { console.error(e && e.stack || e); process.exit(1); });"],
                            text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_lane_files_parse():
    _need_node()
    for name in LANE_FILES:
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)


# ---------------------------------------------------------------------------------------------------------- pure
ETAS = r"""
const E = {
  running: {status: 'running', stages: [stage('a', 30, 30, 'done'), stage('b', 100, 40, 'running'), stage('c', 20, null, 'pending')], precomputed: ['c']},
  queued: {status: 'queued', stages: [stage('a', 30, null, 'pending'), stage('b', 100, null, 'pending')], queue_ahead: 2, queue_ahead_s: 90, remaining_s: 130},
  waiting: {status: 'running', waiting: true, stages: [stage('a', 30, 30, 'done'), stage('b', 100, 5, 'running')], remaining_s: 100},
  confirm: {status: 'awaiting_confirmation', stages: [stage('a', 30, 30, 'done'), stage('b', 60, null, 'pending')], remaining_s: 75},
  overdue: {status: 'running', stages: [stage('a', 30, 30, 'done'), stage('b', 10, 30, 'running')]},
  input: {status: 'awaiting_input', stages: [stage('a', 30, null, 'pending')], segment_remaining_s: 40, remaining_s: 200},
  segA: {status: 'running', segment: 'A', stages: [stage('a', 60, 10, 'running'), stage('b', 100, null, 'pending')]},
  empty: {status: 'running', stages: []}};
"""


def test_remaining_time_rules_match_the_classic_workbench():
    out = _pure("const stage = (key, e, el, state) => ({key, label: key.toUpperCase(), expected_s: e, elapsed_s: el, state});\n" + ETAS + r"""
      const pick = v => v && {status: v.status, progress: v.progress, headline: v.headline, remaining: v.remaining === null || v.remaining === undefined ? null : +v.remaining.toFixed(6),
        cur: v.currentIndex, segs: v.segments.map(s => [s.state, s.width, s.fill, s.overdue])};
      const rows = {}, heads = {}, views = {};
      for (const [k, e] of Object.entries(E)) for (const age of [0, 20]) {
        rows[k + '@' + age] = R.etaRowSummary(e, {ageS: age});
        const v = R.etaView(e, {ageS: age}); heads[k + '@' + age] = v && v.headline; views[k + '@' + age] = pick(v);
      }
      const secs = [3, 12, 44, 57.5, 61, 125, 569, 570, 3000, null, 'x'];
      out({rows, heads, views, fmt: secs.map(R.formatDuration),
        rem: [[100, 50], [100, 85], [100, 200], [0, 5]].map(([e, t]) => +R.stageRemaining(e, t).toFixed(6))});
    """)
    r = out["rows"]
    assert r["running@0"] == {"pct": 47, "text": "还需约 1 分 20 秒", "state": "running"}
    assert r["running@20"] == {"pct": 60, "text": "还需约 1 分钟", "state": "running"}          # the running stage advances on the browser clock
    assert r["queued@0"] == {"pct": None, "text": "前面 2 个", "state": "queued"}
    assert r["waiting@0"]["text"] == "等待前一个任务" and r["confirm@0"] == {"pct": None, "text": "确认后约 1 分 15 秒", "state": "waiting"}
    assert r["overdue@0"] == {"pct": 99, "text": "即将完成", "state": "running"}
    assert r["input@0"] is None and r["empty@0"] is None                                   # nothing drawn for these
    assert r["segA@20"] == {"pct": 19, "text": "还需约 2 分 10 秒", "state": "running"}
    assert out["heads"]["queued@20"] == "前面 2 个，约 1 分 10 秒后开始" and out["heads"]["segA@0"] == "预计还需约 2 分 30 秒（不含人工确认出口）"
    assert out["heads"]["input@0"] == "确认后约 40 秒完成中心线提取"
    assert out["fmt"] == ["几秒", "约 10 秒", "约 45 秒", "约 1 分钟", "约 1 分钟", "约 2 分 5 秒", "约 9 分 30 秒", "约 10 分钟", "约 50 分钟", "", ""]
    assert out["rem"] == [50, 18.823529, 8, 0]
    assert out["fmt"] == CLASSIC_FMT                                                         # the classic workbench_core, frozen
    assert sorted(out["views"]) == sorted(CLASSIC_ETA)
    for key, classic in CLASSIC_ETA.items():
        assert out["views"][key] == classic["view"], key
        assert out["rows"][key] == classic["row"], key


def test_case_facts_missing_results_recent_and_queue_ranges():
    out = _pure(r"""
      const rel = [{id: 'X5D', contract: {protocol: 'single_frame_wss'}}, {id: 'M1', default: true}, {id: 'PF6', contract: {protocol: 'single_frame_volume'}}];
      const J = (id, rid, extra) => Object.assign({id, status: 'done', model_release: {id: rid}, reusable: true, created_at: '2026-09-2' + id.length + 'T08:00:00+08:00'}, extra || {});
      // newest first, as the rail model sorts a scan
      const scan = [J('C3', 'PF6', {status: 'running'}), J('C2', 'M1', {reusable: false}), J('C1', 'M1', {owner_name: 'bob'}), J('C0', 'X5D', {status: 'failed', reusable: false})];
      const m1 = R.missingResults(scan, rel);
      const m2 = R.missingResults(scan, rel, {canRun: j => j.owner_name !== 'bob'});
      const m3 = R.missingResults([J('D1', 'M1', {status: 'awaiting_confirmation', reusable: false})], rel);
      const facts = R.caseFacts({scans: [{jobs: [J('E2', 'M1', {status: 'running', max_diameter_mm: 99}), J('E1', 'M1', {max_diameter_mm: '52.14', tags: ['AAA']})]},
        {jobs: [J('E0', 'X5D', {max_diameter_mm: 40, tags: ['随访', 'AAA']})]}]});
      const none = R.caseFacts({scans: [{jobs: [J('F1', 'M1')]}]});
      const recent = R.recentDone([J('a', 'M1', {finished_at: '2026-09-29T08:00:00Z'}), J('b', 'M1', {status: 'failed'}), J('c', 'M1', {finished_at: '2026-09-30T08:00:00Z'}),
        J('d', 'M1', {finished_at: null, created_at: '2026-09-29T09:00:00Z'})], 5).map(j => j.id);
      // queue ranges: lower bounds as the classic cohortFilter (area shares in percent there, fractions here)
      const rows = [
        {job_id: 'a', wss_p99_pa: 12, area_frac_high: 0.02, area_frac_very_high: 0.001, speed_p99_m_s: null, max_diameter_mm: 30, review_status: 'unreviewed'},
        {job_id: 'b', wss_p99_pa: 20, area_frac_high: 0.08, area_frac_very_high: 0.02, max_diameter_mm: 55, review_status: 'reopened'},
        {job_id: 'c', wss_p99_pa: null, speed_p99_m_s: 1.4, max_diameter_mm: 72, review_status: 'reviewed'},
        {job_id: 'd', wss_p99_pa: 31, area_frac_high: 0.15, area_frac_very_high: 0.05, max_diameter_mm: null}];
      const C = o => AD.cohortModel(rows, Object.assign({family: '', release: '', review: '', sort: {key: 'job_id', dir: 'asc'}, bin: null, ranges: {}}, o)).map(r => r.job_id).sort();
      const cases = [
        [{wss_p99_pa: {lo: 18}}, {p99_min: 18}], [{area_frac_high: {lo: 0.05}}, {high_min: 5}], [{area_frac_very_high: {lo: 0.02}}, {very_high_min: 2}],
        [{speed_p99_m_s: {lo: 1}}, {speed_min: 1}], [{max_diameter_mm: {lo: 50}}, {diameter_min: 50}], [{wss_p99_pa: {lo: 15}, max_diameter_mm: {lo: 40}}, {p99_min: 15, diameter_min: 40}]];
      const lower = cases.map(([rg]) => C({ranges: rg}));   // the classic cohortFilter arguments stay beside them (CLASSIC_COHORT_LOWER)
      const upper = [C({ranges: {wss_p99_pa: {lo: 15, hi: 25}}}), C({ranges: {max_diameter_mm: {hi: 55}}}), C({ranges: {wss_p99_pa: {lo: null, hi: null}}})];
      const reopened = [C({review: 'reopened'}), C({review: 'unreviewed'})];
      const cols = AD.COHORT_COLS.map(c => c.key);
      const col = k => AD.COHORT_COLS.filter(c => c.key === k)[0];
      const texts = [AD.rangeText(col('wss_p99_pa'), {lo: 18}), AD.rangeText(col('area_frac_high'), {lo: 0.05, hi: 0.2}), AD.rangeText(col('max_diameter_mm'), {hi: 55})];
      const bounds = [AD.parseBound('5', col('area_frac_high')), AD.parseBound(' 18.5 ', col('wss_p99_pa')), AD.parseBound('', col('wss_p99_pa')), AD.parseBound('x', col('wss_p99_pa'))];
      out({m1: {src: m1.source && m1.source.id, miss: m1.missing.map(x => x.releaseId)}, m2: {src: m2.source && m2.source.id, miss: m2.missing.map(x => x.releaseId)},
        m3: {src: m3.source, miss: m3.missing.length, un: m3.unavailable.map(x => x.releaseId)}, facts, none, recent, lower, upper, reopened, cols, texts, bounds});
    """)
    assert out["m1"] == {"src": "C1", "miss": ["X5D"]}           # C2 is not reusable, C0 failed; PF6 is on its way; M1 is there
    assert out["m2"] == {"src": None, "miss": []}                  # the only reusable source is someone else's
    assert out["m3"] == {"src": None, "miss": 0, "un": ["X5D", "PF6"]}   # outlets not confirmed yet: nothing offered
    assert out["facts"] == {"diameter": 52.14, "tags": ["AAA", "随访"]} and out["none"] == {"diameter": None, "tags": []}
    assert out["recent"] == ["c", "d", "a"]                         # by finish time, the creation time standing in
    assert out["lower"] == CLASSIC_COHORT_LOWER                                            # classic cohortFilter, frozen
    assert out["lower"] == [["b", "d"], ["b", "d"], ["b", "d"], ["c"], ["b", "c"], ["b"]]
    assert out["upper"] == [["b"], ["a", "b"], ["a", "b", "c", "d"]]
    assert out["reopened"][0] == CLASSIC_COHORT_REOPENED == ["b"] and out["reopened"][1] == ["a", "d"]
    assert out["cols"].index("area_frac_very_high") == out["cols"].index("area_frac_high") + 1
    assert out["texts"] == ["WSS p99 ≥ 18 Pa", "高 WSS 占比 5%–20%", "管腔最大直径 ≤ 55 mm"]
    assert out["bounds"] == [0.05, 18.5, None, None]


# ---------------------------------------------------------------------------------------------------------- shell
def test_login_page_reveal_caps_remember_cooldown_and_maintenance():
    out = _run(r"""
      canned['/api/session'] = {body: {authenticated: false, login: 'password', csrf_token: 'c0'}};
      localStorage.setItem('wss-login-user', 'doctor');
      let tries = 0;
      canned['POST /api/session'] = () => {
        tries += 1;
        if (tries === 1) return {status: 401, body: {error: {message: '用户名或口令不正确。'}}};
        if (tries === 2) return {status: 429, body: {error: {message: '尝试过于频繁'}, retry_after: 2}};
        if (tries === 3) return {status: 503, body: null};
        canned['/api/session'] = {body: {authenticated: true, login: 'password', csrf_token: 'c2', username: 'doctor', role: 'user'}};
        return {body: {authenticated: true, login: 'password', csrf_token: 'c2', username: 'doctor', role: 'user'}};
      };
      await boot();
      const user = byId('wsl-user'), pass = byId('wsl-pass'), keep = byId('wsl-remember'), form = walk(app(), e => e.tagName === 'FORM')[0];
      const submit = walk(form, e => e.tagName === 'BUTTON' && e.type === 'submit')[0];
      const start = {user: user.value, keep: keep.checked, classic: textOf(app()).includes('经典工作台'), links: walk(app(), e => e.tagName === 'A').length};
      const eye = walk(form, e => e.attrs['aria-label'] === '显示口令')[0];
      fire(eye, 'click'); const shown = {type: pass.type, label: eye.attrs['aria-label'], pressed: eye.attrs['aria-pressed']};
      fire(eye, 'click'); const hidden = pass.type;
      const caps = byId('wsl-caps');
      fire(pass, 'keydown', {getModifierState: k => k === 'CapsLock'}); const capsOn = !caps.hidden;
      fire(pass, 'blur'); const capsOff = caps.hidden;
      pass.value = 'bad'; fire(form, 'submit'); await wait(40);
      const e401 = textOf(byClass(form, 'login-err')[0]);
      fire(form, 'submit'); await wait(40);
      const cool = {disabled: submit.disabled, text: textOf(submit), err: textOf(byClass(form, 'login-err')[0])};
      fire(form, 'submit'); await wait(20); const triesWhileWaiting = tries;
      await wait(2300);
      const after = {disabled: submit.disabled, text: textOf(submit)};
      fire(form, 'submit'); await wait(40);
      const e503 = textOf(byClass(form, 'login-err')[0]);
      keep.checked = false; pass.value = 'good'; fire(form, 'submit'); await wait(150);
      done({start, shown, hidden, capsOn, capsOff, e401, cool, triesWhileWaiting, after, e503, stored: localStorage.getItem('wss-login-user'), remember: localStorage.getItem('wss-login-remember'),
        rail: byClass(app(), 'ws-rail').length});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["start"] == {"user": "doctor", "keep": True, "classic": False, "links": 0}          # no 「经典工作台」 footer
    assert out["shown"] == {"type": "text", "label": "隐藏口令", "pressed": "true"} and out["hidden"] == "password"
    assert out["capsOn"] is True and out["capsOff"] is True
    assert out["e401"].startswith("用户名或口令不正确")
    assert out["cool"]["disabled"] is True and out["cool"]["text"] == "2 秒后可再试" and "（2 秒后可再试）" in out["cool"]["err"]
    assert out["triesWhileWaiting"] == 2                                                          # no request while waiting
    assert out["after"] == {"disabled": False, "text": "登录"}
    assert out["e503"] == "服务正在启动或维护，请稍后再试。"
    assert out["stored"] is None and out["remember"] == "0" and out["rail"] == 1                 # not remembered; logged in


HOME_JOBS = r"""
const etaRun = {status: 'running', stages: [stage('a', 50, 10, 'running')]};
canned['/api/jobs'] = {body: {jobs: [
  jobRecord('A', {reusable: true, max_diameter_mm: 52.14, tags: ['AAA', '随访'], patient_id: 'P-1', display_name: 'P-1', finished_at: '2026-09-29T09:00:00+08:00'}),
  jobRecord('B', {review: {status: 'reviewed'}, display_name: 'CASE_B', patient_id: 'P-2', finished_at: '2026-09-29T10:00:00+08:00'}),
  jobRecord('R', {status: 'running', phase: '阶段 B', display_name: 'RUN_ME', patient_id: 'P-3', eta: etaRun})]}};
canned['/api/jobs/A'] = {body: {job: jobRecord('A', {version: 4, mapping: {'3': 'out-le'}, a: {}})}};
canned['POST /api/jobs/A/rerun'] = {body: {job: {id: 'N9', status: 'queued'}}};
"""


def test_home_guide_recent_case_facts_fill_missing_and_live_eta():
    out = _run(HOME_JOBS + r"""
      await boot();
      const guide = byClass(app(), 'home-guide')[0];
      const g = {steps: byClass(guide, 'guide-step').length, close: walk(guide, e => String(e.className).includes('guide-close')).length, upload: Boolean(button(guide, '上传 STL'))};
      const recent = textOf(byClass(byClass(app(), 'home-kpis')[0], 'ov-card')[4]);   // 10-01: the 最近完成 card (no list on the home)
      const cardOf = name => byClass(app(), 'case-card').filter(c => textOf(byClass(c, 'case-name')[0]) === name)[0];
      const A = cardOf('P-1'), Bc = cardOf('CASE_B');
      const facts = {dia: textOf(byClass(A, 'case-dia')[0]), tags: textOf(byClass(A, 'case-tags')[0]), addA: byClass(A, 'res-add').map(textOf), addB: byClass(Bc, 'res-add').length,
        diaB: byClass(Bc, 'case-dia').length};
      const band = textOf(byClass(byClass(app(), 'home-kpis')[0], 'ov-card')[1]);
      const railRow = byClass(app(), 'rail-result').filter(r => r.dataset.jobId === 'R')[0];
      const railEta = {cls: railRow.className, text: textOf(byClass(railRow, 'eta-txt')[0]), fill: byClass(railRow, 'eta-fill')[0].style.width};
      await wait(1300);
      const railLater = byClass(railRow, 'eta-fill')[0].style.width;
      // 补跑: the dashed chip → confirm → POST /rerun with the source's version and the missing release
      fire(byClass(A, 'res-add')[0], 'click'); await wait(20);
      const confirmText = textOf(byId('ws-dialog'));
      fire(button(byId('ws-dialog'), '开始'), 'click'); await wait(80);
      const rerunBody = bodies['POST /api/jobs/A/rerun'], toast = textOf(byId('ws-toast'));
      // closing the guide remembers it (classic key); the avatar menu opens it again
      fire(walk(guide, e => String(e.className).includes('guide-close'))[0], 'click'); await wait(20);
      const closed = {key: localStorage.getItem('wss-guide-closed'), guide: byClass(app(), 'home-guide').length};
      fire(byClass(app(), 'top-avatar')[0], 'click'); await wait(10);
      fire(menuItem('三步上手'), 'click'); await wait(40);
      done({g, recent, facts, band, railEta, railLater, confirmText, rerunBody, toast, closed, reopened: {key: localStorage.getItem('wss-guide-closed'), guide: byClass(app(), 'home-guide').length}});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["g"] == {"steps": 3, "close": 1, "upload": False}
    assert out["recent"].startswith("最近完成") and "最近一次" in out["recent"] and "CASE_B" in out["recent"]   # the newest finished result
    f = out["facts"]
    assert f["dia"] == "52.1 mm" and f["tags"] == "#AAA #随访" and f["addA"] == ["体场"] and f["addB"] == 0 and f["diaB"] == 0
    assert "RUN_ME" in out["band"] and "还需约 40 秒" in out["band"]                           # the 进行中 card reads the estimate
    assert "has-eta" in out["railEta"]["cls"] and out["railEta"]["text"] == "还需约 40 秒" and out["railEta"]["fill"] == "20%"
    assert out["railLater"] in ("22%", "23%")                                                  # the shared ticker advances it
    assert "体内压力与速度" in out["confirmText"] and "原有结果不变" in out["confirmText"]
    assert out["rerunBody"] == {"version": 4, "release_id": "PF6_VF6_peak_3seed_20260920"} and out["toast"].startswith("已建立补跑任务")
    assert out["closed"] == {"key": "1", "guide": 0} and out["reopened"] == {"key": None, "guide": 1}


def test_empty_library_opens_the_guide_with_upload():
    out = _run(r"""
      canned['/api/jobs'] = {body: {jobs: []}};
      localStorage.setItem('wss-guide-closed', '1');
      await boot();
      const guide = byClass(app(), 'home-guide')[0];
      done({guide: Boolean(guide), upload: Boolean(guide && button(guide, '上传 STL')), close: guide ? walk(guide, e => String(e.className).includes('guide-close')).length : -1,
        empty: byClass(byClass(app(), 'home')[0], 'empty').length, recent: byClass(app(), 'home-recent').length});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["guide"] is True and out["upload"] is True and out["close"] == 0 and out["empty"] == 0 and out["recent"] == 0


def test_task_rows_menu_edit_info_eta_diameter_and_batch_figures():
    out = _run(r"""
      canned['/api/jobs'] = {body: {jobs: [
        jobRecord('A', {max_diameter_mm: 52.14, patient_id: 'P-1', input_sha256: 'sha-1', created_at: '2026-09-29T09:00:00+08:00'}),
        jobRecord('A2', {patient_id: 'P-1', input_sha256: 'sha-9', created_at: '2026-09-29T08:00:00+08:00'}),
        jobRecord('R', {status: 'running', eta: {status: 'running', stages: [stage('a', 50, 10, 'running')]}, created_at: '2026-09-29T07:00:00+08:00'}),
        jobRecord('L', {review: {status: 'reviewed'}, created_at: '2026-09-29T06:00:00+08:00'})]}};
      canned['/api/jobs/A'] = {body: {job: jobRecord('A', {version: 5, patient_id: 'P-1', notes: '旧备注', tags: ['AAA']})}};
      canned['POST /api/jobs/A/metadata'] = {body: {job: jobRecord('A', {version: 6, patient_id: 'P-9'})}};
      const batchCalls = [];
      await boot();
      await hashTo('#/tasks', 150);
      const heads = walk(byClass(app(), 'wsc-table')[0], e => e.tagName === 'TH').map(textOf);
      const row = id => byClass(app(), 'wsc-row').filter(r => r.dataset.jobId === id)[0];
      const dia = textOf(row('A').children[6]);
      const eta = textOf(byClass(row('R'), 'wsc-td-status')[0]);
      const openMenu = id => { fire(byClass(row(id), 'wsc-more')[0], 'click'); return menuItems().map(e => [textOf(byClass(e, 'menu-label')[0]), Boolean(e.disabled)]); };
      const menuA = openMenu('A'); ns.ui.closeMenu();
      const menuL = openMenu('L'); ns.ui.closeMenu();
      const menuR = openMenu('R'); ns.ui.closeMenu();
      const hashBefore = location.hash;
      openMenu('A'); fire(menuItem('一页纸'), 'click'); await wait(10);
      // 编辑信息: the result page's own dialog, for this row's job
      openMenu('A'); fire(menuItem('编辑信息…'), 'click'); await wait(60);
      const D = byId('ws-dialog');
      const title = textOf(walk(D, e => e.tagName === 'H2')[0]);
      const field = l => walk(D, e => e.attrs['aria-label'] === l)[0];
      const values = [field('病例名称').value, field('患者编号').value, field('备注').value];
      field('患者编号').value = 'P-9';
      fire(button(D, '保存'), 'click'); await wait(80);
      const meta = bodies['POST /api/jobs/A/metadata'];
      const after = {open: D.open, cur: ns.detail.shell() && ns.detail.shell().cur ? ns.detail.shell().cur() : 'x', hash: location.hash};
      // 删除… asks first
      openMenu('A'); fire(menuItem('删除…'), 'click'); await wait(20);
      const del = textOf(byId('ws-dialog')); ns.ui.dialog.close('cancel');
      // 批量出图 only with lane 5's module; it gets the finished ones of the selection
      const all = walk(byClass(app(), 'wsc-table')[0], e => e.tagName === 'INPUT' && e.attrs['aria-label'] === '选择本页全部')[0];
      all.checked = true; fire(all, 'change'); await wait(20);
      const without = Boolean(button(byClass(app(), 'wsc-selbar')[0], '批量出图…'));
      ns.batch.open = (ids, api) => batchCalls.push([ids, typeof api.go]);
      all.checked = false; fire(all, 'change'); all.checked = true; fire(all, 'change'); await wait(20);
      const bbtn = walk(byClass(app(), 'wsc-selbar')[0], e => e.tagName === 'BUTTON' && textOf(e).startsWith('批量出图'))[0];
      fire(bbtn, 'click'); await wait(10);
      done({heads, dia, eta, menuA, menuL, menuR, opened, hashBefore, title, values, meta, after, del, without, bbtn: textOf(bbtn), batchCalls});
    """)
    assert out["errors"] == [], out["errors"]
    assert "直径 mm" in out["heads"] and out["dia"] == "52.1"
    assert out["eta"].startswith("计算中") and "还需约 40 秒" in out["eta"]
    assert out["menuA"] == [["打开", False], ["一页纸", False], ["编辑信息…", False], ["换模型重跑…", False], ["患者时间线", False], ["删除…", False]]
    assert ["编辑信息…", True] in out["menuL"] and ["删除…", True] in out["menuL"]              # signed-off: locked
    assert ["删除…", True] in out["menuR"] and all(label != "一页纸" for label, _ in out["menuR"])   # running
    assert out["opened"] == ["/api/jobs/A/onepage"] and out["hashBefore"] == "#/tasks"
    assert out["title"] == "编辑病例信息" and out["values"] == ["CASE_A", "P-1", "旧备注"]
    assert out["meta"]["version"] == 5 and out["meta"]["patient_id"] == "P-9" and out["meta"]["tags"] == ["AAA"]
    assert out["after"] == {"open": False, "cur": None, "hash": "#/tasks"}                    # the dialog let go of the row's job
    assert "删除 1 个任务" in out["del"]
    assert out["without"] is False and out["bbtn"].startswith("批量出图…")
    assert out["batchCalls"] == [[["A", "A2", "L"], "function"]]


def test_patient_timeline_from_a_row_opens_the_follow_up_section():
    out = _run(r"""
      canned['/api/jobs'] = {body: {jobs: [jobRecord('A', {patient_id: 'P-1', input_sha256: 'sha-1'}), jobRecord('B', {patient_id: 'P-1', input_sha256: 'sha-2'})]}};
      let scrolled = 0; const sec = {classList: {add() {}, remove() {}}, scrollIntoView() { scrolled += 1; }};
      document.querySelector = sel => sel.includes('sec-follow') ? sec : null;
      await boot();
      await hashTo('#/tasks', 150);
      fire(byClass(byClass(app(), 'wsc-row')[0], 'wsc-more')[0], 'click');
      fire(menuItem('患者时间线'), 'click');
      for (const f of winEvents.hashchange || []) f({});
      await wait(900);
      done({hash: location.hash, scrolled, tab: ns.shell.state().tab || null});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["hash"] == "#/job/A" and out["scrolled"] == 1 and out["tab"] in (None, "overview")


def test_queue_reopened_filter_and_value_ranges():
    out = _run(r"""
      canned['/api/jobs/export'] = {body: {columns: [], rows: [
        {job_id: 'A', case_id: 'CASE_A', release_id: 'X5D', family: 'wall', review_status: 'unreviewed', wss_p99_pa: 16.6, area_frac_high: 0.02, area_frac_very_high: 0.004, max_diameter_mm: 72.2, created_at: '2026-09-20T08:00:00+08:00'},
        {job_id: 'B', case_id: 'CASE_B', release_id: 'X5D', family: 'wall', review_status: 'reopened', wss_p99_pa: 20.7, area_frac_high: 0.09, area_frac_very_high: 0.02, max_diameter_mm: 38.6, created_at: '2026-09-18T08:00:00+08:00'},
        {job_id: 'C', case_id: 'CASE_C', release_id: 'X5D', family: 'wall', review_status: 'reviewed', wss_p99_pa: 25.1, area_frac_high: 0.12, max_diameter_mm: 30.2, created_at: '2026-09-17T08:00:00+08:00'}],
        population: {}}};
      await boot();
      await hashTo('#/cohort', 150);
      const count = () => textOf(byClass(app(), 'sec-count')[0]);
      const filters = () => byClass(app(), 'wsc-filters')[0];
      const sel = l => walk(filters(), e => e.attrs['aria-label'] === l)[0];
      const rev = sel('复核状态'); const revOpts = walk(rev, e => e.tagName === 'OPTION').map(textOf);
      rev.value = 'reopened'; fire(rev, 'change'); await wait(10);
      const reopened = count();
      sel('复核状态').value = ''; fire(sel('复核状态'), 'change'); await wait(10);
      const heads = walk(byClass(app(), 'wsc-cohort-table')[0], e => e.tagName === 'TH').map(textOf);
      // WSS p99 ≥ 18 Pa (the metric on show): typed bounds apply after a short pause
      const lo = walk(filters(), e => e.attrs['aria-label'] === '下限')[0];
      lo.value = '22'; fire(lo, 'input'); await wait(450);
      const p99 = {count: count(), unit: textOf(byClass(filters(), 'wsc-range-unit')[0]), dim: walk(app(), e => String(e.attrs.class || '').includes('wsc-bin') && String(e.attrs.class).includes('dim')).length,
        pts: walk(app(), e => String(e.attrs.class || '').startsWith('wsc-pt') && String(e.attrs.class).includes('dim')).length};
      // another metric: the p99 range stays as a chip; a percent bound on the high-WSS share
      const metric = sel('图上的量'); metric.value = 'area_frac_high'; fire(metric, 'change'); await wait(10);
      const chip = textOf(byClass(filters(), 'wsc-chip')[0]);
      const lo2 = walk(filters(), e => e.attrs['aria-label'] === '下限')[0];
      const cleared = lo2.value;
      lo2.value = '5'; fire(lo2, 'change'); await wait(20);
      const both = {count: count(), unit: textOf(byClass(filters(), 'wsc-range-unit')[0]), ranges: JSON.parse(JSON.stringify(ns.admin._state.cohort.ranges))};
      // the chip's close drops that range
      fire(walk(byClass(filters(), 'wsc-chip')[0], e => e.attrs['aria-label'] === '清除')[0], 'click'); await wait(10);
      done({revOpts, reopened, heads, p99, chip, cleared, both, last: count(), exportAll: urls.filter(u => u.includes('/api/jobs/export')).length});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["revOpts"] == ["全部复核状态", "未复核", "已复核", "已重新打开"] and out["reopened"] == "1 / 3"
    assert "极高 WSS 占比" in out["heads"]
    assert out["p99"]["count"] == "1 / 3" and out["p99"]["unit"] == "Pa" and out["p99"]["dim"] >= 1 and out["p99"]["pts"] == 2   # the charts keep the context
    assert out["chip"] == "WSS p99 ≥ 22 Pa" and out["cleared"] == ""
    assert out["both"]["count"] == "1 / 3" and out["both"]["unit"] == "%"
    assert out["both"]["ranges"] == {"wss_p99_pa": {"lo": 22, "hi": None}, "area_frac_high": {"lo": 0.05, "hi": None}}
    assert out["last"] == "2 / 3"                                                               # only the share bound is left


def test_maintenance_banner_on_503_then_recovery():
    out = _run(r"""
      canned['/api/health'] = {status: 503, body: null};
      await boot();
      await ns.admin.checkHealth(); await wait(20);
      const down = {hidden: byId('wsc-banner').hidden, text: textOf(byId('wsc-banner'))};
      canned['/api/health'] = {body: {ok: true, version: '0.16.0'}};
      await ns.admin.checkHealth(); await wait(40);
      done({down, up: byId('wsc-banner').hidden, toast: textOf(byId('ws-toast'))});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["down"] == {"hidden": False, "text": "服务正在启动或维护，自动重试…"}
    assert out["up"] is True and out["toast"].startswith("服务已恢复")


# ---------------------------------------------------------------------------------------------------------- thumbnails
_IDB = r"""
// A small in-memory IndexedDB: open (upgrade + success), one store with an 'at' index, get / put / count / cursor.
function fakeIDB() {
  const dbs = {};
  const later = f => setTimeout(f, 0);
  const req = () => ({result: undefined, onsuccess: null, onerror: null});
  function storeApi(rec) {
    return {
      get(key) { const r = req(); later(() => { r.result = rec.data.get(key); r.onsuccess && r.onsuccess(); }); return r; },
      put(value, key) { rec.data.set(key, value); const r = req(); later(() => r.onsuccess && r.onsuccess()); return r; },
      count() { const r = req(); later(() => { r.result = rec.data.size; r.onsuccess && r.onsuccess(); }); return r; },
      createIndex() {},
      index() {
        return {openCursor() {
          const r = req(); const keys = [...rec.data.keys()].sort((a, b) => rec.data.get(a).at - rec.data.get(b).at); let i = 0;
          const step = () => later(() => { const k = keys[i]; r.result = k === undefined ? null : {delete() { rec.data.delete(k); }, continue() { i += 1; step(); }}; r.onsuccess && r.onsuccess(); });
          step(); return r;
        }};
      }
    };
  }
  return {open(name) {
    const r = req();
    later(() => {
      let fresh = false;
      if (!dbs[name]) { dbs[name] = {stores: {}}; fresh = true; }
      const d = dbs[name];
      r.result = {objectStoreNames: {contains: n => Boolean(d.stores[n])}, createObjectStore(n) { d.stores[n] = {data: new Map()}; return storeApi(d.stores[n]); },
        transaction(n) { return {objectStore: () => storeApi(d.stores[n])}; }, close() {}};
      if (fresh && r.onupgradeneeded) r.onupgradeneeded();
      r.onsuccess && r.onsuccess();
    });
    return r;
  }, _dbs: dbs};
}
// A stand-in three.js: enough for ws_thumbs.draw to "render" and hand back a data URL.
class Any { constructor() { this.position = {set() {}}; this.domElement = {toDataURL: () => 'data:image/png;base64,QUJD'}; }
  setPixelRatio() {} setSize() {} setClearColor() {} render() {} setAttribute() {} setIndex() {} computeVertexNormals() {} dispose() {} add() {} lookAt() {} }
global.THREE = {WebGLRenderer: Any, BufferGeometry: Any, BufferAttribute: Any, MeshPhongMaterial: Any, Scene: Any, HemisphereLight: Any, DirectionalLight: Any, Mesh: Any,
  OrthographicCamera: Any, DoubleSide: 2};
global.document = {createElement: () => ({})};
global.WSSV2 = {};
"""


def test_thumbnails_persist_in_indexeddb_and_work_without_it():
    _need_node()
    program = _IDB + "require(" + json.dumps(str(V2 / "ws_thumbs.js")) + ");\n" + r"""
      const T = global.WSSV2.thumbs, wait = ms => new Promise(r => setTimeout(r, ms));
      const geo = {preview: {vertices: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces: [[0, 1, 2]]}, endpoints: []};
      const ask = (key, fetchLog) => new Promise(res => T.request(key, () => { fetchLog.push(key); return Promise.resolve(geo); }, url => res(url)));
      (async () => {
        const keys = [T.keyFor({id: 'J1', status: 'done', run_identity: 'r1', version: 3}), T.keyFor({id: 'J1', status: 'done', run_identity: 'r2', version: 3}),
          T.keyFor({id: 'J2', status: 'awaiting_confirmation', version: 7}), T.keyFor(null)];
        // no IndexedDB at all: drawn, memory only
        const log0 = [], u0 = await ask('K0', log0);
        // with IndexedDB: the first request draws and stores; after a "reload" (memory cleared) it comes back without drawing
        global.indexedDB = fakeIDB(); T._reset();
        const log1 = [], u1 = await ask('K1', log1); await wait(20);
        const rows = global.indexedDB._dbs['wssv2-thumbs'] ? global.indexedDB._dbs['wssv2-thumbs'].stores.thumbs.data.size : -1;
        T._reset();
        const log2 = [], u2 = await ask('K1', log2);
        // an IndexedDB that throws on open: still drawn
        global.indexedDB = {open() { throw new Error('SecurityError'); }}; T._reset();
        const log3 = [], u3 = await ask('K3', log3);
        console.log(JSON.stringify({keys, u0, log0, u1, log1, rows, u2, log2, u3, log3}));
      })().catch(e => { console.error(e && e.stack || e); process.exit(1); });
    """
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr[-3000:]
    out = json.loads(result.stdout.strip().splitlines()[-1])
    url = "data:image/png;base64,QUJD"
    assert out["keys"][0] == "J1|r:r1|d1" and out["keys"][1] != out["keys"][0] and out["keys"][2] == "J2|g:7|d1" and out["keys"][3] == ""
    assert out["u0"] == url and out["log0"] == ["K0"]
    assert out["u1"] == url and out["log1"] == ["K1"] and out["rows"] == 1
    assert out["u2"] == url and out["log2"] == []                       # from IndexedDB: no geometry fetch, no drawing
    assert out["u3"] == url and out["log3"] == ["K3"]


def test_package_names_and_the_result_right_click_menu():
    """2026-10-02 (user request): the task rows and every rerun choice name the package a result came from (old and new
    packages of one kind otherwise read the same); right-click on a result in the left rail or on a case card opens the
    row menu at the pointer with the package in its heading; deleting the open result from it moves to another result of
    the same scan."""
    out = _run(r"""
      canned['/api/v2/model-cards'].body.cards.M1_3head_3seed_20260922.version_date = '2026-09-22';
      const vol = {model_release: {id: 'PF6_VF6_peak_3seed_20260920', contract: {protocol: 'single_frame_volume', fields: {velocity: {}}}}, family: 'volume'};
      canned['/api/jobs'] = {body: {jobs: [jobRecord('A', {input_sha256: 'sha-1'}), jobRecord('V', Object.assign({input_sha256: 'sha-1'}, vol))]}};
      serve('A', {input_sha256: 'sha-1'}); serve('V', Object.assign({input_sha256: 'sha-1'}, vol));
      await boot();
      await hashTo('#/tasks', 150);
      const row = id => byClass(app(), 'wsc-row').filter(r => r.dataset.jobId === id)[0];
      const cells = [textOf(byClass(row('A'), 'wsc-td-result')[0]), textOf(byClass(row('V'), 'wsc-td-result')[0])];
      const pkgs = [textOf(byClass(row('A'), 'wsc-pkg')[0]), textOf(byClass(row('V'), 'wsc-pkg')[0])];
      fire(byClass(row('A'), 'wsc-more')[0], 'click'); fire(menuItem('换模型重跑…'), 'click'); await wait(20);
      const D = byId('ws-dialog');
      const options = walk(D, e => e.tagName === 'OPTION').map(textOf);
      const from = textOf(byClass(D, 'wsc-rerun-from')[0]);
      ns.ui.dialog.close('cancel');
      // the result page: right-click the rail row of A
      await hashTo('#/job/A', 200);
      const railA = byClass(app(), 'rail-result').filter(r => r.dataset.jobId === 'A')[0];
      fire(railA, 'contextmenu', {clientX: 120, clientY: 300});
      const M = byId('ws-menu');
      const head = byClass(M, 'menu-head').map(textOf);
      const items = menuItems().map(e => textOf(byClass(e, 'menu-label')[0]));
      const pos = [M.style.left, M.style.top, M.hidden];
      // a second right-click elsewhere moves the menu instead of closing it
      fire(railA, 'contextmenu', {clientX: 140, clientY: 320});
      const moved = [M.style.left, M.style.top, M.hidden];
      // the context-menu key opens it under the row
      ns.ui.closeMenu();
      fire(railA, 'keydown', {key: 'ContextMenu'});
      const byKey = !M.hidden && textOf(byClass(M, 'menu-head')[0]) === head[0];
      // delete from the menu: confirm, then the other result of the scan opens
      canned['POST /api/jobs/delete'] = {body: {results: [{id: 'A', deleted: true}]}};
      fire(menuItem('删除…'), 'click'); await wait(20);
      fire(button(byId('ws-dialog'), '移入回收站'), 'click'); await wait(120);
      done({cells, pkgs, options, from, head, items, pos, moved, byKey, hash: location.hash, title: railA.title || railA.attrs.title});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["pkgs"] == ["M1_3head · 09-22", "PF6_VF6_peak"] and out["cells"][0].startswith("周期指标")
    assert out["options"] == ["周期指标 TAWSS · OSI（默认，M1_3head · 09-22）", "体内压力与速度（PF6_VF6_peak）"], out["options"]
    assert out["from"] == "当前结果：周期指标 TAWSS · OSI（M1_3head · 09-22）"
    assert out["head"] == ["周期指标 TAWSS · OSI", "模型包 M1_3head · 09-22"]
    assert out["items"] == ["打开", "一页纸", "打包下载", "编辑信息…", "换模型重跑…", "删除…"], out["items"]
    assert out["pos"] == ["120px", "300px", False] and out["moved"] == ["140px", "320px", False] and out["byKey"] is True
    assert "模型包 M1_3head · 09-22" in out["title"] and "右键" in out["title"]
    assert out["hash"] == "#/job/V"
