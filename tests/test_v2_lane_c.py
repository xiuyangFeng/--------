"""Lane C (S6c, 工作台) of the workspace v2 second phase: task list, batch actions, trash, cohort overview, service
state, notices, in-place re-login, upload additions (PHASE2_LANES.md §5 C).

Node scenarios reuse the stub DOM / canned service harness of ``test_v2_workspace_js.py`` with ``ws_admin.js``
loaded in bundle order; the pure helpers are compared with the classic ``workbench_core.js`` where the classic
workbench has the same rule.  S7 retired that file; its outputs for these inputs are frozen in ``CLASSIC_CORE``
(computed with workbench_core.js at 866cf02 before it was deleted).  The one server change (``owner_name`` in the administrator's all-users list) is tested
against a live test server.
"""
from __future__ import annotations

import json
import subprocess

from tests._c_helpers import Service, call, finished
from tests.test_v2_workspace_js import _HARNESS, V2, _need_node
from wss_deploy.users import UserStore

# Classic ``workbench_core.js`` (866cf02, deleted in S7) on the inputs of test_pure_helpers_match_the_classic_workbench_core.
CLASSIC_CORE = {'edges': [3.2, 6.6, 10, 13.399999999999999, 16.8, 20.2, 23.599999999999998, 27, 30.4, 33.8, 37.2, 40.6, 44, 47.4, 50.800000000000004, 54.2, 57.6],
 'hist': [2, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
 'pop': {'release_id': 'X5D', 'values': [5, 6, 7, 8], 'case_count': 136},
 'rerun': [True, False, False, False, False],
 'merged': {'upload': {'units': 'cm', 'device': 'cpu'}, 'x': 1, 'notifications': {'enabled': False}, 'schema_version': 'wss-deploy.preferences/v1'},
 'chk': [[True, True], [False, False], [False, False], [False, False], [True, True], [True, False]],
 'validN': 3,
 'chunks': [2, 20, 20, 3],
 'tags': ['AAA', '随访', 'ILO']}
LANE_FILES = ["ws_rail.js", "ws_admin.js", "ws_upload.js", "ws_main.js"]
WS = ["ws_icons.js", "ws_ui.js", "ws_store.js", "ws_api.js", "ws_rail.js", "ws_admin.js", "ws_overview.js", "ws_lens.js",
      "ws_bookmarks.js", "ws_questions.js", "ws_compare.js", "ws_upload.js", "ws_input.js", "ws_export.js", "ws_shell.js", "ws_main.js"]

# Extra stubs for the lane: full-URL request log, downloads, a settable fetch for queries, a FormData reader.
_EXTRA = r"""
const urls = [], downloads = [], bodies = {};
const baseFetch = global.fetch;
let route = null;
global.fetch = async (url, opts) => {
  urls.push((opts && opts.method || 'GET') + ' ' + String(url));
  if (opts && opts.body && typeof opts.body === 'string') bodies[(opts.method || 'GET') + ' ' + String(url).split('?')[0]] = JSON.parse(opts.body);
  if (opts && opts.body && typeof FormData === 'function' && opts.body instanceof FormData) bodies[(opts.method || 'GET') + ' ' + String(url).split('?')[0]] = opts.body;
  if (route) { const r = route(String(url), opts || {}); if (r) { calls.push((opts && opts.method || 'GET') + ' ' + String(url).split('?')[0]); return Resp(r.status || 200, r.body); } }
  return baseFetch(url, opts);
};
const stubDownloads = () => { ns.ui.downloadBlob = (blob, name) => { downloads.push(name); return true; }; };
const syn = (i, extra) => jobRecord('S' + String(i).padStart(3, '0'), Object.assign({status: i % 3 ? 'failed' : 'done', case_id: 'SYN_' + i, display_name: 'SYN_' + i,
  created_at: new Date(Date.UTC(2026, 8, 1, 0, 0, 0) + i * 3600000).toISOString(), patient_id: i % 2 ? 'P-' + (i % 5) : '', tags: i % 4 ? [] : ['AAA']}, extra || {}));
"""


def _run(scenario: str) -> dict:
    _need_node()
    program = (_HARNESS.replace("__WS__", json.dumps([str(V2 / f) for f in WS])).replace("__OFFLINE__", "false") + _EXTRA
               + "\n(async () => {\n" + scenario + "\n})().catch(e => { errors.push('scenario: ' + (e && e.stack || e)); done({}); });\n")
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise AssertionError(result.stderr[-4000:])
    lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
    assert lines, result.stdout[-2000:] + result.stderr[-2000:]
    return json.loads(lines[-1])


def test_lane_files_parse_and_load_in_bundle_order():
    _need_node()
    for name in LANE_FILES:
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    scripts = bundle["scripts"]
    assert scripts.index("ws_rail.js") < scripts.index("ws_admin.js") < scripts.index("ws_shell.js")
    assert "ws_admin.js" in bundle["offline_exclude"] and "v2_workbench.css" in bundle["styles"]


def test_pure_helpers_match_the_classic_workbench_core():
    out = _run(r"""
      for (const f of WS) require(f);
      const WB = __CLASSIC__;   // frozen classic outputs (workbench_core.js was retired in S7)
      const AD = ns.admin, UP = ns.upload, R = ns.rail;
      const vals = [3.2, 7.1, 7.1, 12.5, null, 'x', 20.65, 5.21, 57.6];
      const edges = AD.binEdges(vals, 16), wbEdges = WB.edges;
      const hist = AD.histogram(vals, edges), wbHist = WB.hist;
      const pop = {X5D: {values_pa: [5, 6, 7, 8], case_count: 136}, M1: {values_pa: []}};
      const rows = [{release_id: 'X5D'}, {release_id: 'X5D'}, {release_id: 'M1'}];
      const popSame = JSON.stringify(AD.populationValues(pop, rows, '')) === JSON.stringify(WB.pop);
      const jobsForRerun = [{status: 'done', mapping: {'3': 'out-le'}, a: {}}, {status: 'done', mapping: {}}, {status: 'failed'}, {status: 'done', mapping: {'3': 'x'}}, null];
      const rerun = jobsForRerun.map((j, i) => [AD.rerunSkipReason(j) === null, WB.rerun[i]]);
      const merged = [AD.mergePrefs({upload: {units: 'mm', device: 'cpu'}, x: 1}, {upload: {units: 'cm'}, notifications: {enabled: false}}),
                      WB.merged];
      const files = [{name: 'LV_GUO_YOU.stl', size: 10}, {name: 'a.txt', size: 5}, {name: 'e.stl', size: 0}, {name: 'big.stl', size: 200 * 1048576}, {name: 'lv_guo_you.STL', size: 3}, {name: 'P-001.stl', size: 4}];
      const chk = UP.checkFiles(files, {});
      const chunks = UP.uploadChunks(Array.from({length: 45}, (_, i) => ({size: i < 3 ? 100 * 1048576 : 1})), {maxFiles: 20, maxBytes: 240 * 1048576}).map(c => c.length);
      const wbChunks = WB.chunks;
      const tags = [UP.parseTags('AAA, 随访，AAA、ILO'), WB.tags];
      const ids = [UP.caseIdFor('', {name: 'x.stl'}, false), UP.caseIdFor('CT', {name: 'x.stl'}, true), UP.caseIdFor('', {name: 'y.STL'}, true), UP.caseIdFor('K1', {name: 'y.stl'}, false)];
      // task list model: status buckets, quick filters, sort with blanks sinking, paging
      const list = [syn(1), syn(2), syn(3, {review: {status: 'reviewed'}}), syn(4, {status: 'awaiting_confirmation'}), syn(5, {status: 'running'}), syn(6), syn(7, {status: 'cancelled'})];
      const T = (o) => AD.taskModel(list, Object.assign({q: '', status: '', patient: '', tag: '', quick: '', sort: {key: 'created', dir: 'desc'}, page: 1, size: 50}, o), Date.UTC(2026, 8, 1, 5));
      const byStatus = {}; ['', 'confirm', 'active', 'pending', 'reviewed', 'failed', 'cancelled'].forEach(s => { byStatus[s || 'all'] = T({status: s}).rows.map(j => j.id); });
      const quick = {today: T({quick: 'today'}).total, todo: T({quick: 'todo'}).rows.map(j => j.id), attn: T({quick: 'attn'}).total};
      const byPatient = T({sort: {key: 'patient', dir: 'asc'}}).rows.map(j => j.patient_id || '-');
      const paged = T({size: 3, page: 3}); const clamp = T({size: 3, page: 99});
      // cohort model: filters, histogram bin (last bin closed), sort with blanks last in both directions
      const crow = (id, fam, rel, p99, rev) => ({job_id: id, family: fam, release_id: rel, wss_p99_pa: p99, review_status: rev, case_id: id});
      const crows = [crow('a', 'wall', 'X5D', 10, 'unreviewed'), crow('b', 'wall', 'M1', 20, 'reviewed'), crow('c', 'volume', 'PF6', null, 'unreviewed'), crow('d', 'wall', 'X5D', 30, 'unreviewed')];
      const C = (o) => AD.cohortModel(crows, Object.assign({family: '', release: '', review: '', sort: {key: 'wss_p99_pa', dir: 'desc'}, bin: null}, o)).map(r => r.job_id);
      const cohort = {desc: C({}), asc: C({sort: {key: 'wss_p99_pa', dir: 'asc'}}), wall: C({family: 'wall'}), rev: C({review: 'reviewed'}),
        lastBin: C({bin: {metric: 'wss_p99_pa', lo: 20, hi: 30, last: true}}), midBin: C({bin: {metric: 'wss_p99_pa', lo: 10, hi: 20, last: false}})};
      // notices: a status already in the loaded list is not new; a real transition is announced once
      const known = [{id: 'J1', status: 'done'}, {id: 'J2', status: 'running'}];
      const notes = [AD.shouldNotify({job_id: 'J1', status: 'done', action: 'review_approved'}, known), AD.shouldNotify({job_id: 'J2', status: 'running', action: 'progress'}, known),
        AD.shouldNotify({job_id: 'J2', status: 'done', action: 'finished'}, known), AD.shouldNotify({job_id: 'J2', status: 'done', action: 'review_approved'}, known),
        AD.shouldNotify({job_id: 'J3', status: 'failed', action: 'failed'}, known),
        AD.shouldNotify({job_id: 'J4', status: 'failed', action: 'restored'}, known), AD.shouldNotify({job_id: 'J5', status: 'done', action: 'metadata_updated'}, known)];
      const counts = R.homeCounts([syn(1, {created_at: '2026-09-30T08:00:00+08:00'}), syn(2, {status: 'running'}), syn(3, {status: 'done'}), syn(4, {status: 'awaiting_input'}), syn(5, {status: 'interrupted'})], Date.parse('2026-09-30T12:00:00+08:00'));
      const pick = R.attnPick([['c1'], ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'], [], ['r1', 'r2']], 5);
      done({edges: [edges, wbEdges], hist: [hist, wbHist], popSame, rerun, merged, chk: chk.rows.map(r => [r.ok, Boolean(r.warn)]), wbChk: WB.chk,
        validN: [chk.valid.length, WB.validN], chunks, wbChunks, tags, ids, byStatus, quick, byPatient, paged: [paged.page, paged.pages, paged.rows.length], clamp: clamp.page,
        cohort, notes, counts, pick});
    """.replace("__CLASSIC__", json.dumps(CLASSIC_CORE, ensure_ascii=False)))
    assert out["errors"] == [], out["errors"]
    assert out["edges"][0] == out["edges"][1] and out["hist"][0] == out["hist"][1] and sum(out["hist"][0]) == 7
    assert out["popSame"] is True
    assert all(a == b for a, b in out["rerun"])
    assert out["merged"][0] == out["merged"][1] and out["merged"][0]["upload"] == {"units": "cm", "device": "cpu"}
    assert out["chk"] == out["wbChk"] and out["validN"][0] == out["validN"][1] == 3
    assert out["chunks"] == out["wbChunks"] == [2, 20, 20, 3]
    assert out["tags"][0] == out["tags"][1] == ["AAA", "随访", "ILO"]
    assert out["ids"] == ["", "CT-x", "y", "K1"]
    s = out["byStatus"]
    assert len(s["all"]) == 7 and s["confirm"] == ["S004"] and s["active"] == ["S005"] and s["cancelled"] == ["S007"]
    assert s["reviewed"] == ["S003"] and s["pending"] == ["S006"] and s["failed"] == ["S002", "S001"]
    assert out["quick"]["todo"] == ["S006", "S004"] and out["quick"]["attn"] == 5 and out["quick"]["today"] == 7
    assert out["byPatient"] == ["P-0", "P-1", "P-2", "P-3", "-", "-", "-"]      # blanks sink
    assert out["paged"] == [3, 3, 1] and out["clamp"] == 3
    c = out["cohort"]
    assert c["desc"] == ["d", "b", "a", "c"] and c["asc"] == ["a", "b", "d", "c"]    # missing values last both ways
    assert c["wall"] == ["d", "b", "a"] and c["rev"] == ["b"] and c["lastBin"] == ["d", "b"] and c["midBin"] == ["a"]
    assert out["notes"] == [False, False, True, False, True, False, False]
    assert out["counts"] == {"today": 1, "active": 1, "todo": 2, "failed": 2}
    assert out["pick"] == ["c1", "f1", "f2", "r1", "r2"]


def test_home_tabs_counts_full_list_and_review_in_attention():
    out = _run(r"""
      canned['/api/jobs'] = {body: {jobs: [jobRecord('A'), jobRecord('B', {review: {status: 'reviewed'}}), jobRecord('C', {status: 'awaiting_confirmation'}), jobRecord('D', {status: 'failed'})]}};
      await boot();
      const head = byClass(app(), 'home-head')[0];
      done({tabs: byClass(head, 'home-tab').map(textOf), kpis: byClass(app(), 'ov-card').map(e => textOf(byClass(e, 'ov-value')[0]) + textOf(byClass(e, 'ov-label')[0])), attn: byClass(app(), 'attn-card').map(textOf),
        listUrls: urls.filter(u => u.startsWith('GET /api/jobs') && !u.includes('/api/jobs/')), pages: byClass(app(), 'rail-page').map(e => e.attrs.href || e.href)});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["tabs"][0].startswith("病例") and out["tabs"][1:] == ["任务", "队列", "回收站"]
    assert [t[-2:] if not t.endswith("进行中") else t for t in out["kpis"]] and len(out["kpis"]) == 4
    assert out["kpis"][2] == "2待处理" and out["kpis"][3] == "1失败"             # awaiting + unreviewed; failed
    assert any("待复核" in a for a in out["attn"]) and any("确认出口" in a for a in out["attn"])
    assert out["listUrls"] and all(u == "GET /api/jobs" for u in out["listUrls"])   # the whole list, not the first 100
    assert out["pages"] == ["#/tasks", "#/cohort", "#/trash"]


def test_task_list_server_search_paging_and_batch_actions():
    out = _run(r"""
      const many = Array.from({length: 130}, (_, i) => syn(i + 1));
      canned['/api/jobs'] = {body: {jobs: many}};
      const pagesSeen = [];
      route = (url, opts) => {
        const m = /^\/api\/jobs\?(.*)$/.exec(url); if (!m) return null;
        const q = Object.fromEntries(new URLSearchParams(m[1]));
        pagesSeen.push(q);
        const hits = many.filter(j => j.case_id.toLowerCase().includes((q.q || '').toLowerCase()));
        const page = Number(q.page), size = Number(q.page_size);
        return {body: {jobs: hits.slice((page - 1) * size, page * size), total: hits.length, page, page_size: size}};
      };
      canned['POST /api/jobs/delete'] = opts => ({body: {results: JSON.parse(opts.body).jobs.map(j => ({id: j.id, deleted: true, trashed: true}))}});
      canned['POST /api/trash/S001/restore'] = {body: {job: jobRecord('S001')}};
      canned['POST /api/trash/S002/restore'] = {body: {job: jobRecord('S002')}};
      canned['/api/jobs/export'] = {body: {}};
      await boot();
      stubDownloads();
      await hashTo('#/tasks', 150);
      const first = {count: textOf(byClass(app(), 'sec-count')[0]), rows: byClass(app(), 'wsc-row').length, pager: textOf(byClass(app(), 'wsc-pager')[0])};
      fire(walk(app(), e => e.attrs['aria-label'] === '下一页')[0], 'click'); await wait(20);
      fire(walk(app(), e => e.attrs['aria-label'] === '下一页')[0], 'click'); await wait(20);
      const last = {rows: byClass(app(), 'wsc-row').length, pager: textOf(byClass(app(), 'wsc-pager')[0])};
      // server search: every page of the query is fetched
      const search = walk(app(), e => e.attrs['aria-label'] === '搜索任务')[0];
      search.value = 'syn_1'; fire(search, 'input'); await wait(500);
      const found = {count: textOf(byClass(app(), 'sec-count')[0]), queries: pagesSeen.map(q => [q.q, q.page, q.page_size])};
      search.value = ''; fire(search, 'input'); await wait(450);
      // select two failed tasks on page 1 and delete, then undo
      const status = walk(app(), e => e.attrs['aria-label'] === '任务状态')[0]; status.value = 'failed'; fire(status, 'change'); await wait(20);
      const boxes = walk(byClass(app(), 'wsc-table')[0], e => e.tagName === 'INPUT' && e.attrs['aria-label'] && e.attrs['aria-label'].startsWith('选择 '));
      boxes[0].checked = true; fire(boxes[0], 'change'); boxes[1].checked = true; fire(boxes[1], 'change');
      const bar = textOf(byClass(app(), 'wsc-selbar')[0]);
      const delBtn = walk(byClass(app(), 'wsc-selbar')[0], e => e.tagName === 'BUTTON' && textOf(e).startsWith('删除'))[0];
      fire(delBtn, 'click'); await wait(20);
      const confirmText = textOf(byId('ws-dialog'));
      fire(walk(byId('ws-dialog'), e => e.tagName === 'BUTTON' && textOf(e) === '移入回收站')[0], 'click'); await wait(80);
      const delBody = bodies['POST /api/jobs/delete'];
      const toast = textOf(byId('ws-toast'));
      fire(walk(byId('ws-toast'), e => e.tagName === 'BUTTON' && textOf(e) === '撤销')[0], 'click'); await wait(80);
      // export of the done ones among a selection
      status.value = ''; fire(status, 'change'); await wait(20);
      const all = walk(byClass(app(), 'wsc-table')[0], e => e.tagName === 'INPUT' && e.attrs['aria-label'] === '选择本页全部')[0];
      all.checked = true; fire(all, 'change'); await wait(20);
      const csv = walk(byClass(app(), 'wsc-selbar')[0], e => e.tagName === 'BUTTON' && textOf(e).startsWith('汇总表 CSV'))[0];
      fire(csv, 'click'); await wait(60);
      done({first, last, found, bar, confirmText, delBody, toast, restores: calls.filter(c => c.includes('/restore')), exportUrl: urls.filter(u => u.includes('/api/jobs/export')), downloads, csvLabel: textOf(csv)});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["first"] == {"count": "130", "rows": 50, "pager": "1–50 / 1301 / 3"}
    assert out["last"]["rows"] == 30 and out["last"]["pager"].startswith("101–130 / 130")
    assert out["found"]["count"] == "42"                                          # SYN_1, SYN_10–19, SYN_100–130
    assert out["found"]["queries"] == [["syn_1", "1", "100"]]
    assert out["bar"].startswith("已选 2")
    assert "30 天内可以恢复" in out["confirmText"]
    assert [j["id"] for j in out["delBody"]["jobs"]] == ["S130", "S128"] and all("version" in j for j in out["delBody"]["jobs"])
    assert out["toast"].startswith("已移入回收站 2 个任务")
    assert out["restores"] == ["POST /api/trash/S130/restore", "POST /api/trash/S128/restore"]
    assert len(out["exportUrl"]) == 1 and "format=csv" in out["exportUrl"][0] and out["downloads"] == ["wss_summary.csv"]
    assert out["csvLabel"].startswith("汇总表 CSV（")                              # only the finished ones of the page


def test_search_fetches_every_page_over_100():
    out = _run(r"""
      const many = Array.from({length: 230}, (_, i) => syn(i + 1));
      canned['/api/jobs'] = {body: {jobs: many.slice(0, 5)}};
      const seen = [];
      route = (url) => {
        const m = /^\/api\/jobs\?(.*)$/.exec(url); if (!m) return null;
        const q = Object.fromEntries(new URLSearchParams(m[1])); seen.push(q.page + '/' + q.page_size + '/' + (q.tag || '') + '/' + (q.status || ''));
        const page = Number(q.page), size = Number(q.page_size);
        return {body: {jobs: many.slice((page - 1) * size, page * size), total: many.length, page, page_size: size}};
      };
      await boot();
      await hashTo('#/tasks', 100);
      const status = walk(app(), e => e.attrs['aria-label'] === '任务状态')[0]; status.value = 'pending'; fire(status, 'change');
      const tag = walk(app(), e => e.attrs['aria-label'] === '标签（精确）')[0]; tag.value = 'AAA'; fire(tag, 'change'); await wait(200);
      done({seen, count: textOf(byClass(app(), 'sec-count')[0])});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["seen"] == ["1/100/AAA/done", "2/100/AAA/done", "3/100/AAA/done"]
    assert out["count"] == "76"                          # the finished ones (every third) of all 230, all unreviewed


def test_trash_and_cohort_pages():
    out = _run(r"""
      canned['/api/trash'] = {body: {items: [{id: 'T1', case_id: 'OLD_1', status_before: 'done', release_id: 'M1_3head_3seed_20260922', family: 'wall', deleted_at: '2026-09-29T08:00:00+08:00', days_left: 28.2},
                                             {id: 'T2', case_id: 'OLD_2', patient_id: 'P-9', status_before: 'failed', deleted_at: '2026-09-28T08:00:00+08:00', days_left: 27}]}};
      canned['POST /api/trash/T1/restore'] = {body: {job: jobRecord('T1')}};
      canned['POST /api/trash/T2/purge'] = {body: {purged: true, id: 'T2'}};
      canned['/api/jobs/export'] = {body: {columns: [], rows: [
        {job_id: 'A', case_id: 'CASE_A', release_id: 'X5D', family: 'wall', review_status: 'unreviewed', wss_p99_pa: 16.6, max_diameter_mm: 72.2, population_percentile: 32.4, created_at: '2026-09-20T08:00:00+08:00'},
        {job_id: 'B', case_id: 'CASE_B', patient_id: 'P-2', release_id: 'X5D', family: 'wall', review_status: 'reviewed', wss_p99_pa: 20.7, max_diameter_mm: 38.6, created_at: '2026-09-18T08:00:00+08:00'},
        {job_id: 'V', case_id: 'CASE_V', release_id: 'PF6', family: 'volume', review_status: 'unreviewed', speed_p99_m_s: 1.37, max_diameter_mm: 72.2, created_at: '2026-09-21T08:00:00+08:00'}],
        population: {X5D: {values_pa: [10, 12, 14, 16, 18, 20, 22], case_count: 136}}}};
      await boot();
      await hashTo('#/trash', 150);
      const trashRows = byClass(app(), 'wsc-row').map(textOf);
      fire(walk(app(), e => e.tagName === 'BUTTON' && textOf(e) === '恢复')[0], 'click'); await wait(60);
      const restoreToast = textOf(byId('ws-toast'));
      const purge = walk(app(), e => e.tagName === 'BUTTON' && textOf(e) === '彻底删除');
      fire(purge[purge.length - 1], 'click'); await wait(20);
      const confirm = textOf(byId('ws-dialog'));
      fire(walk(byId('ws-dialog'), e => e.tagName === 'BUTTON' && textOf(e) === '彻底删除')[0], 'click'); await wait(60);
      await hashTo('#/cohort', 150);
      const heads = walk(byClass(app(), 'wsc-cohort-table')[0], e => e.tagName === 'TH').map(textOf);
      const rows = walk(byClass(app(), 'wsc-cohort-table')[0], e => e.tagName === 'TR' && e.className.includes('wsc-row')).map(r => r.children.map(textOf));
      const bars = walk(app(), e => String(e.attrs.class || '') === 'wsc-bar').length, ref = walk(app(), e => String(e.attrs.class || '') === 'wsc-ref').length;
      const legend = textOf(byClass(app(), 'wsc-legend')[0]);
      fire(walk(app(), e => e.attrs['aria-label'] === '按WSS p99 Pa排序')[0], 'click'); await wait(20);
      const sorted = walk(byClass(app(), 'wsc-cohort-table')[0], e => e.tagName === 'TR' && e.className.includes('wsc-row')).map(r => r.dataset.jobId);
      const hits = walk(app(), e => String(e.attrs.class || '') === 'wsc-hit' && e.attrs.role === 'button');
      fire(hits[hits.length - 1], 'click'); await wait(20);
      const binned = {count: textOf(byClass(app(), 'sec-count')[0]), chip: textOf(byClass(byClass(app(), 'wsc-filters')[0], 'wsc-chip')[0])};
      done({trashRows, restoreToast, confirm, calls: calls.filter(c => c.startsWith('POST /api/trash')), heads, rows, bars, ref, legend, sorted, binned,
        exportUrl: urls.filter(u => u.includes('/api/jobs/export'))});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["trashRows"][0].startswith("OLD_1") and "29 天" in out["trashRows"][0] and out["trashRows"][1].startswith("P-9")
    assert out["restoreToast"].startswith("已恢复 OLD_1")
    assert "无法恢复" in out["confirm"]
    assert out["calls"] == ["POST /api/trash/T1/restore", "POST /api/trash/T2/purge"]
    assert out["exportUrl"] == ["GET /api/jobs/export?format=json&scope=done"]
    assert out["heads"][:3] == ["病例", "结果", "管腔最大直径 mm"] and "速度 p99 m/s" in out["heads"] and "TAWSS 均值 Pa" not in out["heads"]   # empty columns hidden
    assert out["rows"][0][0] == "CASE_V" and out["rows"][0][2] == "72.2"          # newest first; the unit lives in the head
    assert out["ref"] == 1 and "人群参照 136 例" in out["legend"] and out["bars"] >= 1
    assert out["sorted"] == ["B", "A", "V"]                                       # p99 descending, missing last
    assert out["binned"]["count"] == "1 / 3" and out["binned"]["chip"].startswith("WSS p99")


def test_notices_unread_banner_and_view_all():
    out = _run(r"""
      canned['/api/jobs'] = {body: {jobs: [jobRecord('A'), jobRecord('R', {status: 'running'})]}};
      canned['/api/health'] = {body: {ok: true, version: '0.16.0', ui_build_v2: 'b1'}};
      let sse = null;
      global.EventSource = class { constructor(url) { calls.push('SSE ' + url); sse = this; this.l = {}; } addEventListener(n, f) { this.l[n] = f; } close() { calls.push('SSE close'); } };
      await boot();
      // a review of a finished task is not news; a running task that finishes is
      sse.l.job({data: JSON.stringify({job_id: 'A', status: 'done', action: 'review_approved', version: 5})}); await wait(20);
      const afterReview = {toast: byId('ws-toast') ? byId('ws-toast').hidden : true, unread: ns.admin.unreadIds()};
      sse.l.job({data: JSON.stringify({job_id: 'R', status: 'done', action: 'finished', version: 7, case_id: 'CASE_R'})}); await wait(900);
      const afterDone = {toast: textOf(byId('ws-toast')), unread: ns.admin.unreadIds(), title: document.title, railDots: byClass(byClass(app(), 'ws-rail')[0], 'rail-unread').length};
      await hashTo('#/job/R', 150);
      const afterOpen = {unread: ns.admin.unreadIds(), title: document.title};
      // connection: a short blip shows nothing, a real outage shows the banner; recovery hides it and says so
      sse.onerror(); await wait(100); sse.onopen(); await wait(50);
      const blip = byId('wsc-banner') ? byId('wsc-banner').hidden : true;
      sse.onerror(); await wait(2700);
      const down = {hidden: byId('wsc-banner').hidden, text: textOf(byId('wsc-banner'))};
      sse.onopen(); await wait(50);
      const up = {hidden: byId('wsc-banner').hidden, toast: textOf(byId('ws-toast'))};
      // a new service version after boot → the reload offer
      canned['/api/health'] = {body: {ok: true, version: '0.16.1', ui_build_v2: 'b2'}};
      await ns.admin.checkHealth(); await wait(20);
      const update = textOf(byId('wsc-banner'));
      // administrator: 看全部用户 → ?all=1, owner in the list
      canned['/api/jobs'] = opts => ({body: {jobs: [jobRecord('A', {owner_name: 'admin'}), jobRecord('Z', {owner_name: 'bob', case_id: 'BOB_1', display_name: 'BOB_1'})]}});
      const avatar = byClass(app(), 'top-avatar')[0]; fire(avatar, 'click'); await wait(10);
      const menu = walk(byId('ws-menu'), e => String(e.className).includes('menu-item')).map(textOf);
      fire(walk(byId('ws-menu'), e => String(e.className).includes('menu-item') && textOf(e) === '看全部用户')[0], 'click'); await wait(80);
      await hashTo('#/tasks', 100);
      const owners = byClass(app(), 'wsc-row').map(r => r.children[r.children.length - 1].textContent);
      const notMine = byClass(app(), 'not-mine').length;
      done({afterReview, afterDone, afterOpen, blip, down, up, update, menu, owners, notMine, listUrls: urls.filter(u => /^GET \/api\/jobs(\?|$)/.test(u))});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["afterReview"]["unread"] == []
    assert out["afterDone"]["toast"].startswith("CASE_R · 已完成") and out["afterDone"]["unread"] == ["R"]
    assert out["afterDone"]["title"] == "(1) WSS 工作区" and out["afterDone"]["railDots"] == 1
    assert out["afterOpen"] == {"unread": [], "title": "WSS 工作区"}
    assert out["blip"] is True
    assert out["down"] == {"hidden": False, "text": "连接中断，正在重连…"}
    assert out["up"]["hidden"] is True and out["up"]["toast"].startswith("连接已恢复")
    assert out["update"].startswith("服务已更新到 v0.16.1") and "刷新页面" in out["update"]
    assert {"服务状态…", "修改口令…", "看全部用户", "完成时桌面提醒"} <= set(out["menu"])
    assert out["listUrls"][0] == "GET /api/jobs" and out["listUrls"][-1] == "GET /api/jobs?all=1"
    assert sorted(out["owners"]) == ["admin", "bob"] and out["notMine"] == 1


def test_in_place_relogin_keeps_the_open_dialog():
    out = _run(r"""
      await boot();
      // an open dialog with typed content
      ns.upload.open({releases: shellState().releases, cards: shellState().cards});
      const patient = walk(byId('ws-dialog'), e => e.attrs['aria-label'] === '患者编号')[0];
      patient.value = 'P-KEEP';
      // the session expires: the next call answers 401
      canned['/api/session'] = {body: {authenticated: false, login: 'password', csrf_token: null}};
      canned['/api/trash'] = {status: 401, body: {error: {message: '请先登录。'}}};
      await ns.api.request('/api/trash').catch(() => null); await wait(60);
      const dlg = byId('ws-relogin');
      const shown = {open: dlg && dlg.open, user: walk(dlg, e => e.attrs['aria-label'] === '用户名')[0].value, upload: byId('ws-dialog').open};
      canned['POST /api/session'] = {body: {authenticated: true, login: 'password', csrf_token: 'csrf-NEW', username: 'admin', role: 'admin'}};
      walk(dlg, e => e.attrs['aria-label'] === '口令')[0].value = 'secret';
      fire(walk(dlg, e => e.tagName === 'FORM')[0], 'submit'); await wait(80);
      done({shown, after: {relogin: dlg.open, upload: byId('ws-dialog').open, patient: patient.value, csrf: ns.api.state.csrf, session: shellState().session.csrf_token,
        toast: textOf(byId('ws-toast'))}, reloads: calls.filter(c => c === 'RELOAD').length, sse: calls.filter(c => c.startsWith('SSE /api/events')).length});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["shown"] == {"open": True, "user": "admin", "upload": True}
    assert out["after"]["relogin"] is False and out["after"]["upload"] is True and out["after"]["patient"] == "P-KEEP"
    assert out["after"]["csrf"] == "csrf-NEW" and out["after"]["session"] == "csrf-NEW" and out["after"]["toast"].startswith("已重新登录")
    assert out["reloads"] == 0 and out["sse"] == 2                                  # the stream is reopened, the page is not reloaded


def test_upload_batch_metadata_full_tier_and_page_drop():
    out = _run(r"""
      canned['POST /api/jobs/batch'] = {body: {results: [{index: 0, job: {id: 'N1'}}, {index: 1, duplicate: true, existing: [{job_id: 'A', case_id: 'CASE_A'}], reusable: 'A'}]}};
      canned['POST /api/jobs'] = {body: {job: {id: 'N2'}}};
      canned['/api/preferences'] = {body: {preferences: {upload: {remember_patient: true, last_patient: {patient_id: 'P-7', scan_label: '基线', tags: 'AAA'}, device: 'cpu'}, other: {keep: 1}}}};
      await boot();
      await wait(50);
      ns.store.setPrefs({tier: 'full'});
      // STL files dropped anywhere on the page open the dialog with them (a non-STL file is left out)
      const f1 = new File([new Uint8Array(20)], 'CASE_ONE.stl'), f2 = new File([new Uint8Array(30)], 'case-two.stl'), txt = new File(['x'], 'notes.txt');
      const dt = {types: ['Files'], files: [f1, f2, txt]};
      for (const f of docEvents.dragenter || []) f({dataTransfer: dt, preventDefault() {}});
      const overlay = byId('wsu-drop') && !byId('wsu-drop').hidden;
      for (const f of docEvents.drop || []) f({dataTransfer: dt, preventDefault() {}});
      await wait(20);
      const D = byId('ws-dialog');
      const val = l => walk(D, e => e.attrs['aria-label'] === l)[0];
      const prefill = ['患者编号', '扫描标签', '标签'].map(l => val(l).value);
      const hasCompute = Boolean(val('计算设备'));
      const rows = byClass(D, 'up-file').map(textOf);
      val('病例名称').value = 'CT'; val('备注').value = '备注一'; val('CPU 线程数').value = '4';
      fire(walk(D, e => e.tagName === 'BUTTON' && textOf(e) === '上传并检查')[0], 'click'); await wait(120);
      const fd = bodies['POST /api/jobs/batch'];
      const sent = fd ? {stl: fd.getAll('stl').map(f => f.name), meta: JSON.parse(fd.get('metadata_json')), patient: fd.get('patient_id'), scan: fd.get('scan_label'), tags: fd.get('tags'),
        notes: fd.get('notes'), device: fd.get('device'), threads: fd.get('threads'), seeds: fd.get('seed_count'), dup: fd.get('on_duplicate')} : null;
      const dupRow = byClass(D, 'up-file').map(textOf);
      // the duplicate: 「只算所选模型」 re-sends that one file on its own with on_duplicate=reuse
      fire(walk(D, e => e.tagName === 'BUTTON' && textOf(e) === '只算所选模型')[0], 'click'); await wait(120);
      const fd2 = bodies['POST /api/jobs'];
      const resent = fd2 ? {stl: fd2.getAll('stl').map(f => f.name), caseId: fd2.get('case_id'), dup: fd2.get('on_duplicate')} : null;
      const prefsPut = bodies['PUT /api/preferences'];
      done({overlay, prefill, hasCompute, rows, sent, dupRow, resent, closed: !D.open, prefsPut, hash: global.location.hash});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["overlay"] is True and out["hasCompute"] is True
    assert out["prefill"] == ["P-7", "基线", "AAA"]                                 # 「记住患者」 from the account preferences
    assert len(out["rows"]) == 2 and out["rows"][0].startswith("CASE_ONE.stl")
    s = out["sent"]
    assert s["stl"] == ["CASE_ONE.stl", "case-two.stl"] and s["meta"] == [{"case_id": "CT-CASE_ONE"}, {"case_id": "CT-case-two"}]
    assert s["patient"] == "P-7" and s["scan"] == "基线" and s["tags"] == "AAA" and s["notes"] == "备注一"
    assert s["device"] == "cpu" and s["threads"] == "4" and s["seeds"] == "all" and s["dup"] == "ask"
    assert "已建立任务" in out["dupRow"][0] and "已经算过" in out["dupRow"][1]
    assert out["resent"] == {"stl": ["case-two.stl"], "caseId": "CT-case-two", "dup": "reuse"}
    assert out["closed"] is True and out["hash"].startswith("#/job/N2")
    p = out["prefsPut"]
    assert p["other"] == {"keep": 1} and p["upload"]["last_patient"]["patient_id"] == "P-7" and p["upload"]["threads"] == "4"


def test_admin_all_users_list_carries_owner_names(tmp_path):
    users = UserStore(tmp_path / "users.json")
    users.add("alice", "alice-secret-1")
    users.add("root", "root-secret-1", admin=True)
    service = Service(tmp_path, shared=True, token="legacy-token", users=users)
    api = service.api()
    try:
        finished(service.manager, owner="alice", case_id="ALICE_1")
        finished(service.manager, owner="anon-session-owner", case_id="ANON_1")
        finished(service.manager, owner="root", case_id="ROOT_1")
        _, payload, _, admin = service.login(api, {"username": "root", "password": "root-secret-1"})
        cookie = {"Cookie": admin["Cookie"]}
        status, listing, _ = call(api, "GET", "/api/jobs?all=1", cookie)
        assert status == 200
        names = {job["case_id"]: job["owner_name"] for job in listing["jobs"]}
        assert names == {"ALICE_1": "alice", "ANON_1": "", "ROOT_1": "root"}   # anonymous owners are never shown
        assert all("owner" not in job for job in listing["jobs"])
        status, paged, _ = call(api, "GET", "/api/jobs?all=1&q=alice&page=1&page_size=10", cookie)
        assert [(j["case_id"], j["owner_name"]) for j in paged["jobs"]] == [("ALICE_1", "alice")] and paged["total"] == 1
        status, own, _ = call(api, "GET", "/api/jobs", cookie)
        assert [j["case_id"] for j in own["jobs"]] == ["ROOT_1"] and "owner_name" not in own["jobs"][0]
        _, _, _, alice = service.login(api, {"username": "alice", "password": "alice-secret-1"})
        status, mine, _ = call(api, "GET", "/api/jobs?all=1", {"Cookie": alice["Cookie"]})   # not an administrator: all=1 is ignored
        assert [j["case_id"] for j in mine["jobs"]] == ["ALICE_1"] and "owner_name" not in mine["jobs"][0]
    finally:
        api.close(); service.close()


def test_a_list_refresh_keeps_any_extension_page():
    # renderHome runs after every list refresh; a page of another extension must not be replaced by the gallery
    out = _run(r"""
      (ns.ext = ns.ext || []).push({id: 'zz', page(name, el) { if (name !== 'zz') return false; el.replaceChildren(ns.ui.h('div', {'class': 'zz-page', text: 'ZZ ' + (++zzDraws)})); return true; }});
      let zzDraws = 0;
      await boot();
      await hashTo('#/zz', 100);
      const before = textOf(byClass(app(), 'zz-page')[0]);
      await ns.shell.state().rail && null;
      ns.admin.shellApi().refreshJobs(); await wait(100);
      done({before, after: textOf(byClass(app(), 'zz-page')[0]), gallery: byClass(app(), 'gallery').length});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["before"] == "ZZ 1" and out["after"] == "ZZ 2" and out["gallery"] == 0
