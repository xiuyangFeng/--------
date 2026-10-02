"""Lane B (S5b): the one-page report in the workspace v2 look, and the report template dialog of the workspace.

The one-pager restyle is presentation only — these tests pin the new markup hooks (status tones, key-number strip,
decision marks, patient banner, print rules, the single allowed inline handler) and check that the new wrappers add no
words.  The template dialog of the workspace runs through the real shell and sends the classic payload.
"""
from __future__ import annotations

import base64
import copy
import hashlib
from html.parser import HTMLParser

from tests.test_cycle_content import m1_summary
from tests.test_onepager import MORPHOLOGY, wall_summary
from tests.test_v2_workspace_js import WS_FILES, _run
from wss_deploy import server as S
from wss_deploy.onepager import DECISION_LABELS, render_onepage


class _Text(HTMLParser):
    """Visible text (no <style>/<title>) plus every on* attribute."""
    def __init__(self):
        super().__init__()
        self.parts, self.skip, self.handlers = [], 0, []

    def handle_starttag(self, tag, attrs):
        self.skip += tag in ("style", "title", "script")
        self.handlers += [(k, v) for k, v in attrs if k.startswith("on")]

    def handle_endtag(self, tag):
        self.skip -= tag in ("style", "title", "script")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _text(html: str) -> _Text:
    parser = _Text()
    parser.feed(html)
    return parser


def _reviewed(summary):
    s = copy.deepcopy(summary)
    s["findings"]["review"] = {"items": {"F1": {"decision": "confirmed", "note": "与影像一致"}, "F2": {"decision": "rejected", "note": "切口效应"}},
                               "added": [{"id": "M1", "xyz_mm": [1, 2, 3], "text": "左髂外局部狭窄", "kind": "manual", "branch": "左髂外", "severity": "attention"}],
                               "legacy": [{"id": "F9", "decision": "rejected", "note": "", "kind": "max_wss", "label": "全场最大 WSS", "branch": "左髂内",
                                           "value": 48.6, "units": "Pa", "reason": "全场最大值已并入高值簇，该簇已有自己的判定"}]}
    s["annotations"] = {"items": [{"id": "A1", "xyz_mm": [0, 0, 0], "text": "钉在瘤囊", "branch": "主动脉", "s_from_root_mm": 88}]}
    return s


def test_page_is_script_free_except_the_print_handler_the_csp_allows():
    html = render_onepage(_reviewed(wall_summary()), {"id": "job-1"})
    parsed = _text(html)
    assert "<script" not in html and parsed.handlers == [("onclick", "window.print()")] and "打印 / 保存 PDF" in html
    # the one-page CSP allows exactly the hash of this handler text
    digest = base64.b64encode(hashlib.sha256(parsed.handlers[0][1].encode()).digest()).decode()
    assert S.ONEPAGE_HANDLER_HASH == f"'sha256-{digest}'"
    assert '<svg class="logo"' in html and 'aria-hidden="true"' in html      # decorative marks carry no text


def test_status_words_carry_the_workspace_tones():
    summary = wall_summary()
    assert '<strong class="ok">已审阅签字</strong>' in render_onepage(summary, {"id": "j"})
    assert '<p class="muted record ok">电子审阅记录：' in render_onepage(summary, {"id": "j"})
    summary["review"] = {"status": "reopened", "by": "R", "at": "2026-09-30", "note": "改判", "version": 3}
    assert '<strong class="warn">已重新打开</strong>' in render_onepage(summary, {"id": "j"})
    summary["review"] = {"status": "unreviewed"}
    page = render_onepage(summary, {"id": "j"})
    assert '<strong class="idle">未审阅</strong>' in page and "电子审阅记录" not in page


def test_key_numbers_form_a_ruled_strip_with_the_agreement_card_as_a_status_word():
    wall = render_onepage(wall_summary(), {"id": "j"})
    assert '<div class="cards c5 ">' in wall                                     # 4 WSS numbers + agreement in one row
    assert '<div class="card q ok"><span class="label"><span class="g" data-gloss="quality_grade">多模型一致性</span></span><strong>多模型一致</strong>' in wall
    m1 = render_onepage(m1_summary(), {"id": "j"})
    assert '<div class="cards c4 ">' in m1                                       # 8 cycle + peak numbers: two rows of four
    summary = wall_summary(); summary["morphology"] = MORPHOLOGY
    assert '<div class="cards c4 morph">' in render_onepage(summary, {"id": "j"})
    summary["quality"] = {"level": "review", "label": "x", "reasons": ["离散度偏大"]}
    assert '<div class="card q warn">' in render_onepage(summary, {"id": "j"})


def test_findings_carry_decision_marks_and_row_states():
    html = render_onepage(_reviewed(wall_summary()), {"id": "job-1"})
    page1, _, appendix = html.partition('<article class="page appendix">')
    assert f'<span class="dec confirmed">{DECISION_LABELS["confirmed"]}</span>' in page1
    assert '<tr class="manual"><td>M1</td><td>【人工】左髂外局部狭窄</td>' in page1
    assert '<tr class="d-confirmed"><td>F1</td>' in page1 and "F2" not in page1.split("<h2>重点发现")[1]   # rejected: appendix only
    rejected = appendix.split("<h3>附录：已驳回的自动发现</h3>")[1].split("</table>")[0]
    assert '<tr class="d-rejected"><td>F2</td>' in rejected and f'<span class="dec rejected">{DECISION_LABELS["rejected"]}</span>' in rejected
    legacy = appendix.split("<h3>规则更新前的判定")[1]
    assert '<table class="findings legacy">' in legacy and '<span class="dec rejected">✕ 已驳回</span>' in legacy
    assert "<h2>标注</h2>" in appendix and "钉在瘤囊" in appendix
    summary = wall_summary(); summary["findings"]["items"].append(
        {"id": "F3", "kind": "max_diameter", "label": "主动脉管腔最大直径", "branch": "主动脉", "value": 63.4, "units": "mm", "rank": 3, "severity": "info"})
    assert '<span class="sev info geo">几何</span>' in render_onepage(summary, {"id": "j"})


def test_patient_banner_keeps_the_words_and_separators():
    html = render_onepage(wall_summary(), {"id": "j"})
    banner = html.split('<p class="idline">')[1].split("</p>")[0]
    assert banner.count('<span class="sep"> · </span>') == 3 and '<span class="idf">患者编号 <b>P-01</b></span>' in banner
    words = "".join(_text(banner).parts)
    # the fixture is bound to X5D_v51, retired on 2026-10-02: its card names it 「（旧版）峰值 WSS」
    assert words == "患者编号 P-01 · 扫描 基线 / 2026-09-01 · 模型 （旧版）峰值 WSS · 生成 2026-09-20 10:00", words


def test_print_rules_keep_page_one_on_one_sheet_and_the_screen_tokens_are_v2():
    html = render_onepage(wall_summary(), {"id": "j"})
    css = html.split("<style>")[1].split("</style>")[0]
    for rule in ("@page{size:A4", ".page+.page{break-before:page", ".page.first{display:flex;min-height:272mm}", ".page.first>footer.pf{margin-top:auto}",
                 "print-color-adjust:exact", "body.flow .page.first{display:block}", "h2{display:flex", "break-after:avoid",
                 "--accent:#0e6e8a", "--ink:#101828", "@media screen and (max-width:840px)"):
        assert rule in css, rule
    assert "box-shadow:0 4px 18px" not in css and "linear-gradient" not in css      # v2: no decorative shadow, no gradient


def test_the_key_number_strip_adds_no_words():
    """``_cards`` only wraps the builders' (label, value, note) triples: the strip reads exactly their text, in order."""
    from wss_deploy.onepager import _quality_card, _wall_cards
    summary = wall_summary()
    html = render_onepage(summary, {"id": "j"})
    strip = html.split("<h2>关键数字</h2>")[1].split('<p class="muted tiers">')[0]
    expected = "".join("".join(_text(part).parts) for card in _wall_cards(summary, {"id": "j"}) + [_quality_card(summary, {"id": "j"})] for part in card)
    assert "".join(_text(strip).parts) == expected and expected.startswith("WSS 全场 p99")


# ---------------------------------------------------------------------------------------------------------------
# The workspace dialog (ws_onepage.js) through the real shell: user menu → 「报告模板…」.
# ---------------------------------------------------------------------------------------------------------------
FILES = [f for f in WS_FILES if f not in ("ws_shell.js", "ws_main.js")] + ["ws_onepage.js", "ws_shell.js", "ws_main.js"]

_TEMPLATE = r"""
canned['/api/report-template'] = {body: {template: {schema_version: 'wss-deploy.report-template/v1', institution: '某某医院', department: '',
  report_title: '', footer_note: '仅限内部', signature_lines: ['报告人', '审阅人'], show_glossary: 'all', appendix: false}, editable: __EDITABLE__}};
const puts = [];
canned['PUT /api/report-template'] = o => { puts.push({body: JSON.parse(o.body), csrf: o.headers['X-CSRF-Token']}); return __PUT__; };
const openMenu = async () => { fire(walk(app(), e => e.className === 'top-avatar')[0], 'click'); await wait(20); return byId('ws-menu'); };
const menuLabels = m => byClass(m, 'menu-label').map(textOf);
const field = label => walk(byId('ws-dialog'), e => e.attrs['aria-label'] === label)[0];
"""


def _dialog(scenario: str, *, editable: bool = True, put: str = "({body: {template: {}, editable: true}})") -> dict:
    return _run(_TEMPLATE.replace("__EDITABLE__", "true" if editable else "false").replace("__PUT__", put) + scenario, files=FILES)


def test_pure_helpers_match_the_classic_rules():
    out = _dialog(r"""
      await boot();
      const O = ns.onepage;
      done({lines: O.parseLines(' 报告人，审阅人,主任医师、报告人\n\n 签收 '), text: O.linesText(['a', 'b']),
        body: O.payload({institution: ' 某院 ', department: '', report_title: ' T ', footer_note: ' x \n', signature_lines: '甲，乙', show_glossary: 'bogus', appendix: false}),
        tooMany: O.problem(O.payload({signature_lines: 'a,b,c,d,e'})), tooLong: O.problem(O.payload({signature_lines: 'x'.repeat(21)})), ok: O.problem(O.payload({signature_lines: '甲'})),
        e403: O.errorText({status: 403}), off: O.menuItems({offline: () => true, api: () => ({})}).length, on: O.menuItems({offline: () => false, api: () => ({})}).map(x => x.label)});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["lines"] == ["报告人", "审阅人", "主任医师", "签收"] and out["text"] == "a，b"
    assert out["body"] == {"institution": "某院", "department": "", "report_title": "T", "footer_note": "x", "signature_lines": ["甲", "乙"],
                           "show_glossary": "used", "appendix": False}
    assert out["tooMany"] == out["tooLong"] == "签字栏最多 4 项，每项不超过 20 个字符。" and out["ok"] is None
    assert "管理员" in out["e403"] and out["off"] == 0 and out["on"] == ["报告模板…"]


def test_user_menu_opens_the_template_dialog_and_saves_the_classic_payload():
    out = _dialog(r"""
      await boot();
      await hashTo('#/job/A', 150);
      const menu = await openMenu();
      const labels = menuLabels(menu);
      fire(walk(menu, e => e.className && e.className.includes('menu-item') && textOf(e).includes('报告模板'))[0], 'click');
      await wait(40);
      const dlg = byId('ws-dialog');
      const loaded = {open: dlg.open, title: textOf(walk(dlg, e => e.tagName === 'H2')[0]), inst: field('机构名称').value, sign: field('签字栏').value,
        foot: field('页脚声明').value, gloss: field('术语表').value, appx: field('打印附录页').checked,
        sketch: textOf(byClass(dlg, 'otp-inst')[0]), sketchSign: byClass(dlg, 'otp-line').length, back: byClass(dlg, 'otp-back')[0].hidden};
      field('机构名称').value = '某某大学附属医院'; fire(field('机构名称'), 'input');
      field('签字栏').value = '报告医师，审核医师，主任医师'; fire(field('签字栏'), 'input');
      field('打印附录页').checked = true; fire(field('打印附录页'), 'change');
      const live = {sketch: textOf(byClass(dlg, 'otp-inst')[0]), sketchSign: byClass(dlg, 'otp-line').map(textOf), back: byClass(dlg, 'otp-back')[0].hidden};
      fire(walk(dlg, e => e.tagName === 'BUTTON' && textOf(e) === '保存')[0], 'click');
      await wait(40);
      done({labels, loaded, live, puts, closed: !byId('ws-dialog').open, toast: textOf(byId('ws-toast'))});
    """)
    assert out["errors"] == [], out["errors"]
    labels = out["labels"]
    assert "报告模板…" in labels and labels.index("报告模板…") < labels.index("运维中心")   # S7: 运维中心 replaced 经典工作台
    assert "GET /api/report-template" in out["calls"]
    assert out["loaded"] == {"open": True, "title": "报告模板", "inst": "某某医院", "sign": "报告人，审阅人", "foot": "仅限内部", "gloss": "all", "appx": False,
                             "sketch": "某某医院", "sketchSign": 2, "back": True}
    assert out["live"]["sketch"] == "某某大学附属医院" and out["live"]["back"] is False
    assert [s.split("日期")[0] for s in out["live"]["sketchSign"]] == ["报告医师", "审核医师", "主任医师"]
    assert out["puts"] == [{"csrf": "csrf-1", "body": {"institution": "某某大学附属医院", "department": "", "report_title": "", "footer_note": "仅限内部",
                                                      "signature_lines": ["报告医师", "审核医师", "主任医师"], "show_glossary": "all", "appendix": True}}]
    assert out["closed"] is True and "报告模板已保存" in out["toast"] and "打开一页纸" in out["toast"]


def test_template_dialog_errors_and_read_only_mode():
    bad = _dialog(r"""
      await boot();
      const menu = await openMenu();
      fire(walk(menu, e => e.className && e.className.includes('menu-item') && textOf(e).includes('报告模板'))[0], 'click');
      await wait(40);
      const dlg = byId('ws-dialog'), save = () => fire(walk(dlg, e => e.tagName === 'BUTTON' && textOf(e) === '保存')[0], 'click');
      field('签字栏').value = 'a,b,c,d,e'; save(); await wait(20);
      const local = textOf(byClass(dlg, 'otp-form')[0]);
      field('签字栏').value = '报告人'; save(); await wait(40);
      done({local, puts: puts.length, server: textOf(byClass(dlg, 'note-error')[0]), open: dlg.open});
    """, put="({status: 403, body: {error: {message: '只有本机回环模式或管理员可以修改报告模板。'}}})")
    assert bad["errors"] == [], bad["errors"]
    assert "签字栏最多 4 项" in bad["local"] and bad["puts"] == 1          # the local check sends nothing
    assert bad["server"] == "没有修改权限：只有管理员可以修改。" and bad["open"] is True
    ro = _dialog(r"""
      await boot();
      const menu = await openMenu();
      fire(walk(menu, e => e.className && e.className.includes('menu-item') && textOf(e).includes('报告模板'))[0], 'click');
      await wait(40);
      const dlg = byId('ws-dialog');
      done({disabled: ['机构名称', '签字栏', '页脚声明', '术语表', '打印附录页'].map(l => field(l).disabled), buttons: walk(dlg, e => e.tagName === 'BUTTON').map(textOf).filter(Boolean),
        note: textOf(byClass(dlg, 'note-warn')[0])});
    """, editable=False)
    assert ro["errors"] == [], ro["errors"]
    assert all(ro["disabled"]) and "保存" not in ro["buttons"] and "恢复默认" not in ro["buttons"] and "关闭" in ro["buttons"]
    assert "管理员" in ro["note"]
