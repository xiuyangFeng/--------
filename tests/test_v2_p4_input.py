"""P4 lane A (final migration round): the input page, the upload dialog and the 工具 tab.

Re-audit rows (jobs/83844143 reaudit_workbench.md / reaudit_reports.md):

* #73  — the outlet-confirmation 3-D view draws the centreline polylines (``preview_polylines`` of /geometry, else the
  stage-A proposal's), colour and see-through drawing of the classic ``createViewer`` (0x567891, opacity 0.9, no depth test);
* #26  — the upload dialog checks case / patient / scan label / tags while typing with the result page's rule
  (``ws_detail.identifierIssue`` / ``metadataError``; a local copy when ws_detail.js is absent) and refuses to send;
* #102 / #78 — the process card has the classic rows 「中心线检查」「中心线端点 / 分叉」 (``hard_pass``, ``topology``), and
  queued / running / waiting pages show 「输入检查与计算过程」 (the tools-tab sections), failed pages the process;
* W30  — 「把握度 x%」 of the outlet naming carries the classic caveat in an ⓘ (not a probability / validated profile);
* W35  — the naming suggestion's warnings (manifest ``analysis.flags`` = summary flags) under 出口命名 and on the input page.
"""
from __future__ import annotations

import json
import subprocess

from tests.test_v2_p3_input import ETA_CONFIRM, ETA_RUNNING, INPUT_CHECK, OPENINGS, PROPOSAL
from tests.test_v2_workspace_js import _HARNESS, V2, _need_node
from wss_deploy.paths import STATIC_DIR

LANE_FILES = ["ws_input.js", "ws_upload.js", "ws_detail.js"]
WS = ["ws_icons.js", "ws_ui.js", "ws_store.js", "ws_api.js", "ws_rail.js", "ws_admin.js", "ws_overview.js", "ws_lens.js", "ws_detail.js", "ws_unroll.js",
      "ws_profile.js", "ws_bookmarks.js", "ws_questions.js", "ws_compare.js", "ws_upload.js", "ws_input.js", "ws_export.js", "ws_shell.js", "ws_main.js"]
NO_DETAIL = [f for f in WS if f not in ("ws_detail.js", "ws_unroll.js", "ws_profile.js")]

CENTERLINE = {"hard_pass": True, "attempt": 0, "openings": OPENINGS,
              "topology": {"nodes": 1149, "edges": 1148, "endpoints": 5, "junctions": 3, "max_degree": 3, "components": 1}}
POLYLINES = [{"segment_id": 0, "parent_id": -1, "xyz": [[13.4, -120.7, -130.9], [12.0, -125.0, -200.0], [10.0, -130.0, -260.0]]},
             {"segment_id": 3, "parent_id": 1, "xyz": [[10.0, -130.0, -260.0], [-20.0, -140.0, -330.0], [-48.85, -147.54, -386.9]]},
             {"segment_id": 9, "parent_id": 1, "xyz": [[1.0, 2.0, None], [1.0, 2.0, 3.0]]},          # one finite point left: dropped
             {"segment_id": 7, "parent_id": 1, "xy": [[0, 0], [1, 1]]}]                                # 2-D sketch only: dropped
EVENTS = [{"at": "2026-09-30T12:21:26+08:00", "action": "created"}, {"at": "2026-09-30T12:21:26+08:00", "action": "started", "stage": "A"},
          {"at": "2026-09-30T12:21:27+08:00", "action": "progress"}, {"at": "2026-09-30T12:21:30+08:00", "action": "finished", "stage": "A"},
          {"at": "2026-09-30T12:21:35+08:00", "action": "precompute_done"}]
GATE = {"passed": False, "threshold": 0.95, "proxy_confidence": 0.8838613308054825, "calibration_status": "missing",
        "reasons": ["几何置信度代理 88.4% 低于门槛 95.0%", "命名建议自身标记为需要确认"]}
CAVEAT = "把握度由几何形态估算，不是经过临床数据校准的概率。"

# A minimal three.js stand-in: enough for ws_input.meshView to build its scene; lines and their materials are recorded.
THREE_STUB = r"""
const made = {lines: [], lineMats: [], renders: 0};
class V3 { constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; } set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
  copy(v) { return this.set(v.x, v.y, v.z); } clone() { return new V3(this.x, this.y, this.z); } project() { return this; } }
class Obj { constructor() { this.children = []; this.position = new V3(); this.up = new V3(); this.userData = {};
    this.scale = {set() {}, setScalar() {}, multiplyScalar() {}}; }
  add(o) { this.children.push(o); return this; } remove(o) { const i = this.children.indexOf(o); if (i >= 0) this.children.splice(i, 1); return this; }
  lookAt() {} updateProjectionMatrix() {} }
class Mat { constructor(o) { Object.assign(this, o || {}); } dispose() {} }
class Geo { setAttribute() {} setIndex() {} computeVertexNormals() {} computeBoundingSphere() { this.boundingSphere = {radius: 120, center: new V3(0, -130, -250)}; }
  setFromPoints(p) { this.points = p; return this; } dispose() {} }
global.THREE = {
  Vector3: V3, Vector2: class { constructor(x, y) { this.x = x; this.y = y; } }, Scene: class extends Obj {}, Group: class extends Obj {},
  PerspectiveCamera: class extends Obj { constructor(fov) { super(); this.fov = fov; this.aspect = 1; } },
  HemisphereLight: class extends Obj {}, DirectionalLight: class extends Obj {}, Sprite: class extends Obj {},
  WebGLRenderer: class { constructor() { this.domElement = document.createElement('canvas'); } setPixelRatio() {} setClearColor() {} setSize() {}
    render() { made.renders += 1; } dispose() {} },
  BufferGeometry: Geo, BufferAttribute: class {}, SphereGeometry: Geo, MeshPhongMaterial: Mat, MeshBasicMaterial: Mat, SpriteMaterial: Mat, CanvasTexture: Mat,
  LineBasicMaterial: class extends Mat { constructor(o) { super(o); made.lineMats.push(this); } },
  Mesh: class extends Obj { constructor(g, m) { super(); this.geometry = g; this.material = m; this.isMesh = true; } },
  Line: class extends Obj { constructor(g, m) { super(); this.geometry = g; this.material = m; made.lines.push(this); } },
  Sphere: class { setFromPoints(p) { this.center = new V3(); this.radius = 100; this.n = p.length; return this; } },
  Color: class { constructor(c) { this.c = c; } }, DoubleSide: 2};
"""

FIXTURES = "const PROPOSAL = %s, OPENINGS = %s, IC = %s, ETA_CONFIRM = %s, ETA_RUNNING = %s, CL = %s, POLY = %s, EVENTS = %s, GATE = %s;\n" % tuple(
    json.dumps(x, ensure_ascii=False) for x in (PROPOSAL, OPENINGS, INPUT_CHECK, ETA_CONFIRM, ETA_RUNNING, CENTERLINE, POLYLINES, EVENTS, GATE))
HELPERS = r"""
const buttons = (root, text) => walk(root, e => e.tagName === 'BUTTON' && textOf(e) === text);
const inspector = () => byClass(app(), 'input-panel')[0];
const tips = root => walk(root, e => String(e.className).includes('info-tip')).map(e => e.title);
const kv = root => walk(root, e => e.tagName === 'TR').slice(1).map(tr => tr.children.map(textOf));   // without the (empty) head row
const urls = [], bodies = {};
const baseFetch = global.fetch;
global.fetch = async (url, opts) => {
  urls.push((opts && opts.method || 'GET') + ' ' + String(url));
  if (opts && opts.body && typeof opts.body !== 'string') bodies[(opts.method || 'GET') + ' ' + String(url).split('?')[0]] = opts.body;
  return baseFetch(url, opts);
};
"""


def _run(scenario: str, files=None, three: bool = False) -> dict:
    _need_node()
    files = [str(STATIC_DIR / "report_common.js")] + [str(V2 / f) for f in (files or WS)]
    program = (_HARNESS.replace("__WS__", json.dumps(files)).replace("__OFFLINE__", "false") + (THREE_STUB if three else "") + FIXTURES + HELPERS
               + "\n(async () => {\n" + scenario + "\n})().catch(e => { errors.push('scenario: ' + (e && e.stack || e)); done({}); });\n")
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise AssertionError(result.stderr[-4000:])
    lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
    assert lines, result.stdout[-2000:] + result.stderr[-2000:]
    return json.loads(lines[-1])


def test_lane_files_parse():
    _need_node()
    for name in LANE_FILES:
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)


# ---------------------------------------------------------------------------------------------------------- #73
def test_outlet_view_draws_the_centreline_like_the_classic_viewer():
    out = _run(r"""
      const job = (id, extra) => jobRecord(id, Object.assign({status: 'awaiting_confirmation', version: 2, a: {proposal: PROPOSAL, centerline: CL, input_check: IC}, eta: ETA_CONFIRM}, extra || {}));
      canned['/api/jobs/Q'] = {body: {job: job('Q')}};
      canned['/api/v2/jobs/Q/inputcheck'] = {body: {status: 'awaiting_confirmation', input_check: IC, mesh: null, openings: OPENINGS}};
      canned['/api/jobs/Q/geometry'] = {body: {endpoints: PROPOSAL.endpoints, preview_polylines: POLY, preview: null}};
      // no polylines in /geometry: the stage-A proposal's (an older record that still embeds them)
      canned['/api/jobs/P'] = {body: {job: job('P', {a: {proposal: Object.assign({}, PROPOSAL, {preview_polylines: POLY.slice(0, 1)}), centerline: CL, input_check: IC}})}};
      canned['/api/v2/jobs/P/inputcheck'] = {body: {status: 'awaiting_confirmation', input_check: IC, mesh: null, openings: OPENINGS}};
      canned['/api/jobs/P/geometry'] = {body: {endpoints: PROPOSAL.endpoints, preview_polylines: []}};
      // a failed job: openings only, no centreline
      canned['/api/jobs/F'] = {body: {job: jobRecord('F', {status: 'failed', a: {input_check: IC, centerline: {}}, error: {message: '计算未完成', category: 'internal'}})}};
      canned['/api/v2/jobs/F/inputcheck'] = {body: {status: 'failed', input_check: IC, mesh: null, openings: OPENINGS}};
      canned['/api/jobs/F/geometry'] = {body: {endpoints: [], preview_polylines: POLY}};
      await boot();
      await hashTo('#/job/Q', 250);
      const st = shellState().inputView.state;
      const q = {lines: st.lines.map(l => l.length), count: st.view.lineCount(), mats: made.lineMats.map(m => [m.color, m.opacity, m.depthTest, m.transparent]),
        order: made.lines.map(l => l.renderOrder), msg: textOf(byClass(app(), 'input-stage-msg')[0]), tools: byClass(app(), 'input-stage-tools')[0].hidden,
        first: made.lines[0].geometry.points.map(p => [p.x, p.y, p.z])};
      // inlet re-pick mode keeps the centreline (it shows which branch the inlet led to)
      fire(buttons(inspector(), '入口选错了？')[0], 'click'); await wait(20);
      const inletMode = shellState().inputView.state.view.lineCount();
      await hashTo('#/job/P', 250);
      const p = shellState().inputView.state.view.lineCount();
      await hashTo('#/job/F', 250);
      const f = shellState().inputView.state.view.lineCount();
      const pure = [ns.input.polylinesFrom(null, null), ns.input.polylinesFrom({preview_polylines: []}, {preview_polylines: [{xyz: [[0, 0, 0], [1, 1, 1]]}]}),
        ns.input.polylinesFrom({preview_polylines: [{xyz: [[0, 0, 0], [1, 1, 1]]}]}, {preview_polylines: [{xyz: [[5, 5, 5], [6, 6, 6]]}]})];
      done({q, inletMode, p, f, pure});
    """, three=True)
    assert out["errors"] == [], out["errors"]
    q = out["q"]
    assert q["lines"] == [3, 3] and q["count"] == 2                   # the broken and the 2-D-only lines are left out
    assert q["mats"][:2] == [[0x567891, 0.9, False, True]] * 2       # classic createViewer: LineBasicMaterial 0x567891, .9, depthTest false
    assert q["order"][:2] == [3, 3]                                   # over the wall, under the endpoint markers (4 / 5)
    assert q["first"] == POLYLINES[0]["xyz"]
    assert q["msg"] == "" and q["tools"] is False                     # no display mesh: framed on the centreline, tools shown
    assert out["inletMode"] == 2 and out["p"] == 1 and out["f"] == 0
    assert out["pure"] == [[], [[[0, 0, 0], [1, 1, 1]]], [[[0, 0, 0], [1, 1, 1]]]]


# ---------------------------------------------------------------------------------------------------------- #26
def test_upload_identifier_rule_is_the_result_pages():
    """Local copy (without ws_detail.js) and ws_detail.identifierIssue give the same verdict; tags as metadataError."""
    script = r"""
      await boot();
      const vals = [['', '病例名称', 160], ['P-001', '患者编号', 80], ['P-0\t01', '患者编号', 80], ['CT\n1', '病例名称', 160], ['x'.repeat(81), '患者编号', 80],
        ['  ' + 'y'.repeat(160) + '  ', '病例名称', 160], ['张三', '病例名称', 160], ['欧阳·娜', '患者编号', 80], ['基线\u0085', '扫描标签', 120]];
      const UP = ns.upload, D = ns.detail;
      const mine = vals.map(([v, label, max]) => UP.identifierIssue(v, {label, max}));
      const ref = D ? vals.map(([v, label, max]) => D.identifierIssue(v, {label, max})) : null;
      const tagCases = ['AAA, 随访', Array.from({length: 13}, (_, i) => 't' + i).join(','), 'ok,' + 'z'.repeat(41), 'A\tB, C', '随访，随访、AAA\nAAA'];
      const tags = tagCases.map(UP.tagsIssue);
      const tagRef = D ? tagCases.map(t => D.metadataError({case_id: 'x', tags: UP.parseTags(t), scan_date: '', notes: ''}, null)) : null;
      const fi = UP.fieldIssues({case_id: '张三', patient_id: 'P\t1', scan_label: 'x'.repeat(121), tags: 'a'}, {many: true});
      const many = UP.fieldIssues({case_id: 'C\t1'}, {many: true}).issues.case_id;
      done({mine, ref, tags, tagRef, fi: {keys: Object.keys(fi.issues).filter(k => fi.issues[k]).sort(), blocking: fi.blocking}, many});
    """
    with_detail = _run(script)
    without = _run(script, files=NO_DETAIL)
    for out in (with_detail, without):
        assert out["errors"] == [], out["errors"]
    assert without["ref"] is None and without["mine"] == with_detail["mine"] == with_detail["ref"]
    m = with_detail["mine"]
    assert m[0] is None and m[1] is None
    assert m[2] == {"level": "error", "text": "患者编号含有换行、制表符等不可见字符（常见于从表格复制），请删掉后重新输入。"}
    assert m[3]["level"] == "error" and m[3]["text"].startswith("病例名称含有换行")
    assert m[4] == {"level": "error", "text": "患者编号最多 80 个字符（当前 81 个）。"}
    assert m[5] is None                                                                       # trimmed length counts
    assert m[6] == {"level": "warn", "text": "「张三」看起来像真实姓名，请改用匿名编号。"} and m[7]["level"] == "warn"
    assert m[8]["level"] == "error"                                                           # U+0085 (C1 control) as the service
    t = with_detail["tags"]
    assert t[0] is None and t[1] == {"level": "error", "text": "最多 12 个标签。"}
    assert t[2] == {"level": "error", "text": "标签「zzzzzzzzzzzz…」超过 40 个字符。"} and t[3]["level"] == "error" and t[4] is None
    assert [x and x["text"] for x in t[:3]] == with_detail["tagRef"][:3]                      # the result page's metadataError texts
    assert with_detail["fi"]["keys"] == ["case_id", "patient_id", "scan_label"]
    assert with_detail["fi"]["blocking"]["key"] == "patient_id"                               # the first error, in form order
    assert with_detail["many"]["text"].startswith("名称前缀含有")                             # several files: the prefix


def test_upload_dialog_hints_block_and_release_the_send():
    out = _run(r"""
      canned['POST /api/jobs'] = {body: {job: {id: 'N9'}}};
      await boot();
      const ctl = ns.upload.open({releases: shellState().releases, cards: shellState().cards});
      const D = byId('ws-dialog');
      const field = l => walk(D, e => e.attrs['aria-label'] === l)[0];
      const hintOf = inp => inp.parentNode.children.filter(c => String(c.className).includes('up-hint'))[0];
      const submit = walk(D, e => e.tagName === 'BUTTON' && textOf(e) === '上传并检查')[0];
      const more = walk(D, e => e.tagName === 'DETAILS' && String(e.className).includes('up-more'))[0];
      const type = (l, v) => { field(l).value = v; fire(field(l), 'input'); };
      ctl.setFile(new File([new Uint8Array(40)], 'scan_a.stl'));
      const ready = submit.disabled;
      type('患者编号', 'P-0\t07');
      const bad = {hint: textOf(hintOf(field('患者编号'))), cls: hintOf(field('患者编号')).className, hidden: hintOf(field('患者编号')).hidden,
        invalid: field('患者编号').getAttribute('aria-invalid'), disabled: submit.disabled};
      ctl.send(); await wait(60);
      const blockedPosts = urls.filter(u => u === 'POST /api/jobs').length, said = textOf(byClass(D, 'upload-progress')[0]);
      // a second error inside the closed 「扫描标签、标签、备注」 opens it, even while the first one blocks
      const closedBefore = more.open;
      type('标签', Array.from({length: 13}, (_, i) => 't' + i).join(','));
      const both = more.open;
      more.open = false;
      type('患者编号', 'P-007');
      const tagsBad = {hint: textOf(hintOf(field('标签'))), open: more.open, disabled: submit.disabled};
      type('标签', 'AAA');
      type('病例名称', '张三');
      const warn = {hint: textOf(hintOf(field('病例名称'))), cls: hintOf(field('病例名称')).className, disabled: submit.disabled, invalid: field('病例名称').getAttribute('aria-invalid')};
      type('病例名称', 'CT-7');
      const clear = [field('患者编号'), field('标签'), field('病例名称')].map(f => hintOf(f).hidden);
      fire(submit, 'click'); await wait(120);
      const fd = bodies['POST /api/jobs'];
      done({ready, bad, blockedPosts, said, closedBefore, both, tagsBad, warn, clear, sent: fd ? {patient: fd.get('patient_id'), tags: fd.get('tags'), caseId: fd.get('case_id')} : null});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["ready"] is False
    b = out["bad"]
    assert b["hint"].startswith("患者编号含有换行、制表符等不可见字符") and "err" in b["cls"] and b["hidden"] is False
    assert b["invalid"] == "true" and b["disabled"] is True
    assert out["blockedPosts"] == 0 and out["said"].startswith("患者编号含有")          # nothing sent: the STL stays in the browser
    assert out["closedBefore"] is False and out["both"] is True
    assert out["tagsBad"] == {"hint": "最多 12 个标签。", "open": True, "disabled": True}
    w = out["warn"]
    assert w["hint"] == "「张三」看起来像真实姓名，请改用匿名编号。" and "err" not in w["cls"] and w["disabled"] is False and w["invalid"] is None
    assert out["clear"] == [True, True, True]
    assert out["sent"] == {"patient": "P-007", "tags": "AAA", "caseId": "CT-7"}


# ---------------------------------------------------------------------------------------------------------- #102 / #78
def test_process_card_on_unfinished_pages():
    out = _run(r"""
      const A = extra => Object.assign({input_check: IC, centerline: CL, timing_s: {ingest: 0.42, centerline: 3.51}}, extra || {});
      canned['/api/jobs/R'] = {body: {job: jobRecord('R', {status: 'running', version: 9, stage: 'B', eta: ETA_RUNNING, a: A(), events: EVENTS, timing: {queue_s: 2}})}};
      canned['/api/jobs/Q'] = {body: {job: jobRecord('Q', {status: 'awaiting_confirmation', version: 2, a: A({proposal: PROPOSAL}), eta: ETA_CONFIRM, events: EVENTS})}};
      canned['/api/v2/jobs/Q/inputcheck'] = {body: {status: 'awaiting_confirmation', input_check: IC, mesh: null, openings: OPENINGS}};
      canned['/api/jobs/Q/geometry'] = {body: {endpoints: PROPOSAL.endpoints}};
      canned['/api/jobs/U'] = {body: {job: jobRecord('U', {status: 'awaiting_input', a: {input_check: Object.assign({}, IC, {status: 'needs_confirmation'})}})}};
      canned['/api/v2/jobs/U/inputcheck'] = {body: {status: 'awaiting_input', input_check: Object.assign({}, IC, {status: 'needs_confirmation'}), mesh: null, openings: []}};
      canned['/api/jobs/N'] = {body: {job: jobRecord('N', {status: 'queued', eta: null, a: {}})}};
      canned['/api/jobs/F'] = {body: {job: jobRecord('F', {status: 'failed', a: A({centerline: {hard_pass: false, topology: {endpoints: 6}}}), events: EVENTS,
        error: {message: '中心线提取未通过', category: 'toolchain'}})}};
      canned['/api/v2/jobs/F/inputcheck'] = {body: {status: 'failed', input_check: IC, mesh: null, openings: OPENINGS}};
      canned['/api/jobs/E'] = {body: {job: jobRecord('E', {status: 'failed', a: A({centerline: {}}), events: [], error: {message: '开口数 6 ≠ 5', category: 'input_geometry', retryable: false}})}};
      canned['/api/v2/jobs/E/inputcheck'] = {body: {status: 'failed', input_check: IC, mesh: null, openings: OPENINGS}};
      await boot();
      const look = async id => {
        await hashTo('#/job/' + id, 220);
        const p = inspector(), box = byClass(p, 'input-proc')[0];
        if (!box) return null;
        const btn = box.children[0].children[0], more = byClass(box, 'proc-more')[0], hidden = more.hidden;
        fire(btn, 'click'); await wait(10);
        return {label: textOf(btn), hidden, open: !more.hidden, titles: byClass(more, 'sec-title').map(textOf), rows: kv(byClass(more, 'tbl-detail')[0]),
          dots: walk(more, e => String(e.className).includes('st-')).map(e => e.className), events: byClass(more, 'event-list').map(l => l.children.map(textOf)),
          facts: textOf(byClass(more, 'input-facts')[0]), last: p.children[1].children[p.children[1].children.length - 1] === box};
      };
      const r = await look('R');
      // the open state survives a repaint of the page (newer status of the same job)
      shellState().inputView.update(jobRecord('R', {status: 'running', version: 10, stage: 'B', eta: ETA_RUNNING, a: A(), events: EVENTS}));
      const kept = !byClass(byClass(inspector(), 'input-proc')[0], 'proc-more')[0].hidden;
      const q = await look('Q'), u = await look('U'), n = await look('N'), f = await look('F'), e = await look('E');
      done({r, kept, q, u, n, f, e});
    """)
    assert out["errors"] == [], out["errors"]
    r = out["r"]
    assert r["label"] == "输入检查与计算过程" and r["hidden"] is True and r["open"] is True and r["last"] is True
    assert r["titles"] == ["输入检查", "计算过程"]
    assert r["rows"] == [["中心线检查", "通过"], ["中心线端点 / 分叉", "5 / 3"]]
    assert any("st-ok" in c for c in r["dots"])
    assert r["events"] == [] or all("进度" not in t for t in r["events"][0])
    assert "单位 mm" in r["facts"] and "5 个开口" in r["facts"]
    assert out["kept"] is True
    q = out["q"]
    assert q["label"] == "输入检查与计算过程" and q["titles"] == ["输入检查", "计算过程"] and q["rows"][0] == ["中心线检查", "通过"]
    assert out["u"] is None and out["n"] is None                     # units waiting (classic: no processCard) / nothing checked yet
    f = out["f"]
    assert f["label"] == "计算过程" and f["titles"] == ["计算过程"]  # the failure card keeps its own input facts
    assert f["rows"] == [["中心线检查", "未通过"], ["中心线端点 / 分叉", "6 / —"]] and any("st-error" in c for c in f["dots"])
    assert out["e"]["rows"] == []                                     # the centreline never ran: no rows, no 「未通过」


def test_centerline_rows_pure():
    out = _run(r"""
      await boot();
      const D = ns.detail;
      done({rows: [D.centerlineRows({a: {centerline: CL}}), D.centerlineRows({stage_a: {centerline: {hard_pass: false}}}), D.centerlineRows({a: {centerline: {}}}),
        D.centerlineRows({summary: {centerline: {hard_pass: true, topology: {endpoints: 5, junctions: null}}}}), D.centerlineRows(null)]});
    """)
    assert out["errors"] == [], out["errors"]
    rows = out["rows"]
    assert rows[0] == [{"k": "中心线检查", "v": "通过", "ok": True}, {"k": "中心线端点 / 分叉", "v": "5 / 3"}]
    assert rows[1] == [{"k": "中心线检查", "v": "未通过", "ok": False}, {"k": "中心线端点 / 分叉", "v": "— / —"}]
    assert rows[2] == [] and rows[4] == []
    assert rows[3][1] == {"k": "中心线端点 / 分叉", "v": "5 / —"}


# ---------------------------------------------------------------------------------------------------------- W30 / W35
def test_input_page_outlet_caveat_and_summary_flags():
    out = _run(r"""
      const q = (id, p, extra) => {
        canned['/api/jobs/' + id] = {body: {job: jobRecord(id, Object.assign({status: 'awaiting_confirmation', version: 2, a: {proposal: p, centerline: CL, input_check: IC}, eta: ETA_CONFIRM}, extra || {}))}};
        canned['/api/v2/jobs/' + id + '/inputcheck'] = {body: {status: 'awaiting_confirmation', input_check: IC, mesh: null, openings: OPENINGS}};
        canned['/api/jobs/' + id + '/geometry'] = {body: {endpoints: PROPOSAL.endpoints}};
      };
      q('Q', Object.assign({}, PROPOSAL, {confidence_gate: GATE}), {summary: {flags: ['左右髂总在 x 轴上只差 2.1 mm，左右判定置信度 61.0%，请人工核对']}});
      q('V', Object.assign({}, PROPOSAL, {confidence_gate: Object.assign({}, GATE, {calibration_status: 'validated'})}));
      q('X', Object.assign({}, PROPOSAL, {confidence: null}));
      await boot();
      const tag = () => tips(byClass(inspector(), 'sec-tagrow')[0])[0];   // the ⓘ beside 「自动命名置信度 x%」
      await hashTo('#/job/Q', 250);
      const qTip = tag();
      const reasons = [textOf(byClass(inspector(), 'note-warn')[0])].concat(byClass(inspector(), 'reasons')[0].children.map(textOf));
      await hashTo('#/job/V', 250);
      const vTip = tag();
      await hashTo('#/job/X', 250);
      const xTip = tag();
      done({qTip, vTip, xTip, reasons});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["qTip"].startswith(CAVEAT + " STL 不带患者方向")
    assert out["vTip"].startswith("已绑定独立校准 profile。 STL 不带患者方向")
    assert out["xTip"].startswith("STL 不带患者方向")                      # no percentage shown: no caveat
    assert out["reasons"][0] == "右侧髂内与髂外的区分把握度只有 88%，请重点核对右侧的两个出口。"
    assert out["reasons"][-1] == "左右髂总在 x 轴上只差 2.1 mm，左右判定置信度 61.0%，请人工核对"   # summary flag, classic words (no rule → as written)
    assert len(out["reasons"]) == len(set(out["reasons"]))


def test_tools_tab_outlet_caveat_flags_and_centreline_rows():
    out = _run(r"""
      const proposal = Object.assign({}, PROPOSAL, {confidence_gate: GATE});
      serve('A', {mapping: proposal.mapping, mapping_history: [{source: 'manual_confirmation'}], a: {proposal, centerline: CL, input_check: IC},
        summary: {timing_s: {ingest: 0.1, centerline: 3.5, total: 3.6}}, events: EVENTS});
      canned['/api/v2/jobs/A/manifest'].body.analysis.flags = ['右侧髂内/髂外区分置信度 88.4%（score 1.47），请人工核对', '左右髂总在 x 轴上只差 2.1 mm，左右判定置信度 61.0%，请人工核对'];
      serve('B', {mapping: proposal.mapping, a: {proposal: Object.assign({}, proposal, {flags: []}), centerline: {}}});
      await boot();
      const tab = id => walk(app(), e => e.dataset && e.dataset.tab === id)[0];
      const body = () => byClass(app(), 'insp-body')[0];
      const sec = cls => byClass(body(), cls)[0];
      await hashTo('#/job/A', 200);
      fire(tab('tools'), 'click'); await wait(40);
      const o = sec('sec-outlets');
      const a = {source: textOf(byClass(o, 'outlet-source')[0]), tips: tips(byClass(o, 'outlet-source')[0]), flagsHead: textOf(byClass(o, 'outlet-flags-head')[0]),
        flags: byClass(o, 'reasons')[0].children.map(textOf), rows: kv(byClass(sec('sec-process'), 'tbl-detail')[0])};
      fire(walk(o, e => e.tagName === 'BUTTON' && textOf(e) === '修改并重算…')[0], 'click'); await wait(20);
      const editingFlags = byClass(sec('sec-outlets'), 'outlet-flags').length;
      await hashTo('#/job/B', 200);
      fire(tab('tools'), 'click'); await wait(40);
      const b = {flags: byClass(sec('sec-outlets'), 'outlet-flags').length, rows: byClass(sec('sec-process'), 'tbl-detail').length};
      done({a, editingFlags, b});
    """)
    assert out["errors"] == [], out["errors"]
    a = out["a"]
    assert a["source"].startswith("人工确认 · 自动命名把握度 88%") and a["tips"] == [CAVEAT]
    assert a["flagsHead"] == "命名时的核对提示"
    assert a["flags"] == ["右侧髂内与髂外的区分把握度只有 88%，请重点核对右侧的两个出口。", "左右髂总在 x 轴上只差 2.1 mm，左右判定置信度 61.0%，请人工核对"]
    assert a["rows"] == [["中心线检查", "通过"], ["中心线端点 / 分叉", "5 / 3"]]
    assert out["editingFlags"] == 1
    assert out["b"] == {"flags": 0, "rows": 0}
