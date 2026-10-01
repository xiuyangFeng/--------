"""Phase 3 lane 3 (结果信息与安全, PHASE3_LANES.md §3): tests.

* The warning bar (ws_notice.js, W65 / V45): trigger and sentence equal to the classic wall report's ``bannerInfo`` (its
  source is cut out of report.py and run in Node) and to the classic volume page's ``VolumeViewerCore.warningText``;
  real jobs, when their copies are present, give the same sentence through the v2 manifest; no dispersion number.
* The overview: quality / reference rows (W29, W31), geometry table and Murray shares (W37, W36), the volume terms
  (V38, V39), the follow-up growth of diameter and sac volume with the most recent interval and every release's line
  (#103); the admin's timeline asks for all users only with 「看全部用户」.
* Technical information (W62): the rows equal the classic ``renderTech`` popover (run in Node) for real jobs, open
  from the overview footer in every tier and offline; the manifest carries the new provenance fields.
* Sections at a centreline station (V15 pressure drop at 10 %, V41 profile click follows), the review checklist (#82)
  equal to the classic workbench's, the glossary (W63 / V38) online and embedded in the offline page.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from wss_deploy import v2_data as V
from wss_deploy import v2_offline as O
from wss_deploy.paths import PROJECT_ROOT, STATIC_DIR

V2 = STATIC_DIR / "v2"
PKG = Path(V.__file__).resolve().parent
LANE = ["ws_notice.js", "ws_overview.js", "ws_lens.js", "ws_detail.js", "ws_profile.js", "ws_unroll.js", "ws_review.js"]
JOB_ROOTS = [PROJECT_ROOT / "outputs" / "wss_deploy_jobs", PROJECT_ROOT / "outputs" / "wss_deploy_preview_jobs"]
BANNER_JOBS = ["20260918_152113_e2e53937b25a", "20260930_045135_c66b989ddd40", "20260918_154058_a72784880799", "20260920_144135_673ccd0e36b1",
               "20260929_230442_a7273f1670e4", "20260929_230450_d1428792b846"]


def _job_dir(job_id: str) -> Path | None:
    for root in JOB_ROOTS:
        if (root / job_id / "report.html").is_file() and (root / job_id / "job.json").is_file():
            return root / job_id
    return None


def _need_node():
    if not shutil.which("node"):
        pytest.skip("Node is required for the lane 3 tests")


def _classic(name_from: str, name_to: str, source: str = "report.py") -> str:
    src = (PKG / source).read_text(encoding="utf-8") if source.endswith(".py") else (STATIC_DIR / source).read_text(encoding="utf-8")
    a, b = src.index(name_from), src.index(name_to)
    return src[a:b]


PURE = [STATIC_DIR / "report_common.js", STATIC_DIR / "volume_viewer.js", V2 / "core_util.js", V2 / "ws_ui.js", V2 / "ws_icons.js",
        V2 / "ws_notice.js", V2 / "ws_overview.js", V2 / "ws_lens.js", V2 / "ws_detail.js", V2 / "ws_profile.js", V2 / "ws_review.js"]


def _node(script: str, pre: str = "") -> dict:
    _need_node()
    prelude = "globalThis.document={getElementById:()=>null};\n" + pre
    prelude += "".join("require(" + json.dumps(str(f)) + ");\n" for f in PURE)
    prelude += ("const ns=globalThis.WSSV2, N=ns.notice, OV=ns.overview, L=ns.lens, D=ns.detail, R=ns.review, C=globalThis.WssReportCommon, VV=globalThis.VolumeViewerCore;\n"
                "const out=x=>console.log(JSON.stringify(x));\n")
    program = prelude + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"
    with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as fh:   # real manifests are large
        fh.write(program)
    try:
        result = subprocess.run(["node", fh.name], text=True, capture_output=True, timeout=120)
    finally:
        Path(fh.name).unlink(missing_ok=True)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


# The classic wall report's banner code, cut out of report.py (REF_FIELDS, referenceLabel, bannerInfo) and run as is.
def _classic_banner_prelude() -> str:
    return ("let META={};const fq=x=>globalThis.WssReportCommon.formatValue(x);\n" + _classic("const REF_FIELDS=", "function openStats(target)") +
            "\nglobalThis.classicBanner=m=>{META=m;const b=bannerInfo();return b?b.text:null;};\n")


FIXTURES = r"""
const F = {
  one: {reference_assessment: {status: 'review', checks: [{path: 'geometry.右髂外.radius_median_mm', units: 'mm', value: 3.45678, min: 3.6, max: 9.1, status: 'review'},
    {path: 'cloud.spacing_mm', units: 'mm', value: 0.5, min: 0.4, max: 0.6, status: 'pass'}], population: {status: 'pass', percentile: 45.588}},
    quality: {level: 'review', label: '存在不确定性，建议复核', reasons: ['3 个模型离散度超过常规范围，建议复核热点和分支']}},
  two: {reference_assessment: {status: 'review', checks: [{path: 'geometry.左髂总.length_mm', units: 'mm', value: 177.82165, min: 21.116, max: 141.016, status: 'review'},
    {path: 'geometry.右髂总.length_mm', units: 'mm', value: 172.7799, min: 13.206, max: 143.658, status: 'review'}], population: {status: 'unknown'}}, quality: {level: 'good', label: '模型集成稳定', reasons: []}},
  spacing: {reference_assessment: {status: 'review', checks: [{path: 'cloud.spacing_mm', units: 'mm', value: 0.91, min: 0.4, max: 0.6, status: 'review'}]}},
  unitless: {reference_assessment: {status: 'review', checks: [{path: 'geometry.主动脉.surface_variation_median', units: '1', value: 0.0312, min: 0.001, max: 0.02, status: 'review'}]}},
  statusOnly: {reference_assessment: {status: 'review', checks: [], reasons: ['部分几何测量超出当前发布包声明的参考范围，请复核。']}},
  pop: {reference_assessment: {status: 'pass', checks: [], population: {status: 'review', reasons: ['参照 profile 与本发布包不一致']}}},
  poor: {quality: {level: 'poor', label: '不稳定，建议复核', reasons: ['五模型离散度较高，结果需要人工复核']}},
  noLabel: {quality: {level: 'review'}},
  tort: {reference_assessment: {status: 'review', checks: [{path: 'geometry.左髂外.tortuosity', units: '1', value: 1.8765, min: 1, max: 1.5, status: 'review'}]}},
  good: {reference_assessment: {status: 'pass', population: {status: 'pass'}}, quality: {level: 'good', label: '模型集成稳定'}},
  none: {}
};
"""


# ---------------------------------------------------------------------------------------------------------------- files
def test_scripts_parse_bundle_places_and_offline():
    _need_node()
    for name in LANE + ["ws_shell.js"]:
        subprocess.run(["node", "--check", str(V2 / name)], check=True, capture_output=True)
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    s = bundle["scripts"]
    assert s.index("ws_notice.js") < s.index("ws_shell.js") and "ws_notice.js" not in bundle["offline_exclude"]   # the bar works offline
    assert not set(LANE) & set(bundle["offline_exclude"]) and "v2_detail.css" in bundle["styles"]
    srcs = re.findall(r'<script src="/static/v2/([^"]+)"', (V2 / "index.html").read_text(encoding="utf-8"))
    assert srcs.index("ws_batch.js") + 1 == srcs.index("ws_notice.js")                                      # still the placeholder slot
    # never the ensemble's dispersion numbers (user ruling): nothing of the lane reads the quality audit
    for name in ["ws_notice.js", "ws_overview.js", "ws_review.js", "ws_detail.js", "ws_lens.js"]:
        text = (V2 / name).read_text(encoding="utf-8")
        assert "quality_audit" not in text.replace("quality_audit.json", "") and "spread_pa" not in text and "relative_spread" not in text, name


# ---------------------------------------------------------------------------------------------------------------- banner
def test_banner_sentence_equals_the_classic_wall_and_volume_pages():
    out = _node(FIXTURES + r"""
      const r = {};
      Object.keys(F).forEach(k => {
        const m = N.model({analysis: F[k]});
        const vol = VV.warningText(VV.warningItems(F[k])) || null;
        r[k] = {mine: m ? m.text : null, wall: classicBanner(F[k]), vol, target: m ? m.target : null, items: m ? m.items.map(x => x.text) : []};
      });
      out(r);
    """, pre=_classic_banner_prelude().replace("globalThis.WssReportCommon", "(globalThis.WssReportCommon||{formatValue:String})"))
    for key, r in out.items():
        assert r["mine"] == r["wall"], (key, r)                                           # the wall report's sentence, every case
    for key in ("one", "two", "spacing", "unitless", "pop", "poor", "noLabel", "good", "none"):
        assert out[key]["mine"] == out[key]["vol"], (key, out[key])                        # and the volume page's where its table has the field
    assert out["one"]["mine"] == "注意：输入几何有 1 项超出发布包参考范围（右髂外中位半径 3.46 mm，参考 3.60–9.10 mm）；模型集成质量：存在不确定性，建议复核。预测可信度可能下降，请结合详情复核。"
    assert out["two"]["mine"].startswith("注意：输入几何有 2 项超出发布包参考范围（左髂总长度 178 mm，参考 21.1–141 mm 等）。")
    assert out["statusOnly"]["mine"] == "注意：输入几何有 部分 项超出发布包参考范围。预测可信度可能下降，请结合详情复核。"
    assert out["statusOnly"]["vol"] is None                                               # the classic volume page missed this case; the bar does not
    assert out["pop"]["mine"] == "注意：人群参照需要复核。预测可信度可能下降，请结合详情复核。" and out["pop"]["target"] == "reference"
    assert out["poor"]["target"] == "quality" and "不稳定，建议复核" in out["poor"]["mine"]
    assert "迂曲度" in out["tort"]["mine"]                                                   # wall field names (the volume table lacks it)
    assert out["good"]["mine"] is None and out["none"]["mine"] is None
    assert out["two"]["items"] == ["左髂总长度 178 mm（参考 21.1–141 mm）", "右髂总长度 173 mm（参考 13.2–144 mm）"]
    for r in out.values():                                                                # labels and reasons only
        assert not re.search(r"离散度\s*[0-9]", r["mine"] or "")


def test_real_jobs_give_the_classic_sentence_through_the_manifest():
    jobs = [(j, _job_dir(j)) for j in BANNER_JOBS]
    jobs = [(j, p) for j, p in jobs if p]
    if not jobs:
        pytest.skip("no job copies present")
    cases = {}
    for job_id, path in jobs:
        V.clear_cache()
        record = json.loads((path / "job.json").read_text(encoding="utf-8"))
        data = V.load(path)
        manifest = V.build_manifest(data, record)
        cases[job_id] = {"manifest": {"analysis": {k: manifest["analysis"].get(k) for k in ("reference_assessment", "quality")}},
                         "meta": {k: data.report.meta.get(k) for k in ("reference_assessment", "quality")}}
    V.clear_cache()
    out = _node("const CASES=" + json.dumps(cases, ensure_ascii=False) + r""";
      const r = {};
      Object.keys(CASES).forEach(k => { const m = N.model(CASES[k].manifest); r[k] = {mine: m ? m.text : null, wall: classicBanner(CASES[k].meta), vol: VV.warningText(VV.warningItems(CASES[k].meta)) || null}; });
      out(r);
    """, pre=_classic_banner_prelude())
    for job_id, r in out.items():
        assert r["mine"] == r["wall"], (job_id, r)
        if r["vol"] is not None:
            assert r["mine"] == r["vol"], (job_id, r)
    shown = {k: v["mine"] for k, v in out.items() if v["mine"]}
    if "20260930_045135_c66b989ddd40" in out:
        assert "输入几何有 2 项超出发布包参考范围" in shown["20260930_045135_c66b989ddd40"]


def test_details_quality_rows_and_no_dispersion_numbers():
    out = _node(FIXTURES + r"""
      const man = k => ({analysis: F[k], provenance: {n_models: k === 'poor' ? 5 : 3}});
      const rows = ['one', 'two', 'good', 'none', 'poor'].map(k => [k, OV.qualityRows(man(k)).map(r => [r.key, r.label, r.text, r.tone])]);
      const rm = N.referenceModel(man('one')), qm = N.qualityModel(man('poor'));
      out({rows, rm: {geometry: rm.geometry, population: rm.population, items: rm.items.map(x => x.text), status: rm.statusLine}, qm, words: [N.ensembleWord(5), N.ensembleWord(3), N.ensembleWord(null)]});
    """)
    rows = dict(out["rows"])
    assert rows["one"] == [["quality", "3 个模型一致性", "存在不确定性，建议复核", "warn"], ["geometry", "几何参考", "1 项超出已声明几何参考范围，请复核", "warn"],
                           ["population", "人群参照", "已计算同协议 CV3 折外经验分位 45.6%", "ok"]]
    assert rows["two"][0] == ["quality", "3 个模型一致性", "多模型一致（一致不代表准确）", "ok"]
    assert rows["good"][1:] == [["geometry", "几何参考", "在已声明几何参考范围内", "ok"], ["population", "人群参照", "已计算同协议 CV3 折外经验分位 —%", "ok"]]
    assert rows["none"] == [] and rows["poor"][0][1] == "五模型一致性"
    assert out["rm"]["items"] == ["右髂外中位半径 3.46 mm（参考 3.60–9.10 mm）"] and out["rm"]["status"].startswith("部分几何测量超出本发布包声明的参考范围")
    assert out["qm"]["reasons"] == ["五模型离散度较高，结果需要人工复核"] and out["words"] == ["五模型", "3 个模型", "五模型"]


# ---------------------------------------------------------------------------------------------------------------- review checklist
# The classic workbench's checklist (app.js reviewChecklist) and alert title (workbench_core.alertModel) for the
# cases below, computed at 866cf02 before S7 deleted both files (P4 lane A, F6: the comparison used to be skipped
# once the files were gone, and ``alertTitle`` had no assertion left).
CLASSIC_REVIEW = [
    {"f": "one", "j": "auto", "items": [["出口命名", "自动确认（置信度 98.8%）"], ["结果质量", "存在不确定性，建议复核"], ["几何参考范围", "有超出范围的测量，请看结果页提示"],
                                       ["需要注意", "1 项几何测量超出模型训练范围，模型集成存在不确定性，结果需要复核。"]],
     "title": "1 项几何测量超出模型训练范围，模型集成存在不确定性，结果需要复核。"},
    {"f": "two", "j": "manual", "items": [["出口命名", "已人工确认"], ["结果质量", "多模型一致（一致不代表准确）"], ["几何参考范围", "有超出范围的测量，请看结果页提示"],
                                         ["需要注意", "2 项几何测量超出模型训练范围，结果需要复核。"]],
     "title": "2 项几何测量超出模型训练范围，结果需要复核。"},
    {"f": "statusOnly", "j": "none", "items": [["出口命名", "—"], ["几何参考范围", "有超出范围的测量，请看结果页提示"], ["需要注意", "几何测量超出模型训练范围，结果需要复核。"]],
     "title": "几何测量超出模型训练范围，结果需要复核。"},
    {"f": "poor", "j": "manual", "items": [["出口命名", "已人工确认"], ["结果质量", "不稳定，建议复核"], ["几何参考范围", "当前发布包未配置"], ["需要注意", "模型集成离散度较高，结果需要复核。"]],
     "title": "模型集成离散度较高，结果需要复核。"},
    {"f": "good", "j": "auto", "items": [["出口命名", "自动确认（置信度 98.8%）"], ["结果质量", "多模型一致（一致不代表准确）"], ["几何参考范围", "在参考范围内"]], "title": None},
    {"f": "none", "j": "none", "items": [["出口命名", "—"], ["几何参考范围", "当前发布包未配置"]], "title": None},
]


def test_review_checklist_is_the_classic_workbench_list():
    out = _node(FIXTURES + r"""
      const jobs = {auto: {mapping_history: [{source: 'manual_confirmation'}, {source: 'automatic_high_confidence', confidence: 0.9876}]}, manual: {mapping_history: [{source: 'manual_override'}]}, none: {}};
      const cases = [['one', 'auto'], ['two', 'manual'], ['statusOnly', 'none'], ['poor', 'manual'], ['good', 'auto'], ['none', 'none']];
      out(cases.map(([f, j]) => ({f, j, mine: R.checklistModel({analysis: F[f]}, jobs[j]), title: R.alertTitle({analysis: F[f]})})));
    """)
    assert [(r["f"], r["j"]) for r in out] == [(c["f"], c["j"]) for c in CLASSIC_REVIEW]
    for mine, classic in zip(out, CLASSIC_REVIEW):
        assert mine["mine"] == classic["items"], mine
        assert mine["title"] == classic["title"], mine
    one = out[0]["mine"]
    assert one == [["出口命名", "自动确认（置信度 98.8%）"], ["结果质量", "存在不确定性，建议复核"], ["几何参考范围", "有超出范围的测量，请看结果页提示"],
                   ["需要注意", "1 项几何测量超出模型训练范围，模型集成存在不确定性，结果需要复核。"]]
    assert out[1]["mine"][:2] == [["出口命名", "已人工确认"], ["结果质量", "多模型一致（一致不代表准确）"]]
    assert out[2]["mine"][-1] == ["需要注意", "几何测量超出模型训练范围，结果需要复核。"]
    assert out[3]["mine"][-1] == ["需要注意", "模型集成离散度较高，结果需要复核。"]
    assert out[5]["mine"] == [["出口命名", "—"], ["几何参考范围", "当前发布包未配置"]]


# ---------------------------------------------------------------------------------------------------------------- tech info
def test_tech_rows_equal_the_classic_popover_for_real_jobs():
    jobs = [(j, _job_dir(j)) for j in ("20260930_045135_c66b989ddd40", "20260920_144135_673ccd0e36b1", "20260929_230442_a7273f1670e4")]
    jobs = [(j, p) for j, p in jobs if p]
    if not jobs:
        pytest.skip("no job copies present")
    cases = {}
    for job_id, path in jobs:
        V.clear_cache()
        record = json.loads((path / "job.json").read_text(encoding="utf-8"))
        data = V.load(path)
        manifest = V.build_manifest(data, record)
        cases[job_id] = {"manifest": {k: manifest.get(k) for k in ("data_version", "job", "result", "time", "mapping", "frame", "provenance")},
                         "meta": {k: v for k, v in data.report.meta.items() if k not in ("profiles", "zones", "findings", "morphology")}}
    V.clear_cache()
    pre = ("let META={},FRAME_SCHEMA=[],CYCLE=null;const COMMON=globalThis.WssReportCommon;const esc=s=>String(s);const POP={innerHTML:''};"
           "const el=id=>id==='tech-pop'?POP:{};\n" + _classic("function schemaFrames(m){", "const FIELD_SCHEMA=") +
           _classic("function createdIso(){", "function renderFooter(){") + _classic("function renderTech(){", "function toggleTech(){") +
           "\nglobalThis.classicTech=m=>{META=m;FRAME_SCHEMA=schemaFrames(m);CYCLE=m.cycle&&typeof m.cycle==='object'?m.cycle:null;renderTech();"
           "const rows=[];const re=/<b>(.*?)(?:<button[^>]*>\\?<\\/button>)?<\\/b><span><code>(.*?)<\\/code><\\/span>/g;let x;while((x=re.exec(POP.innerHTML)))rows.push([x[1],x[2]]);return rows;};\n")
    out = _node("const CASES=" + json.dumps(cases, ensure_ascii=False) + r""";
      const r = {};
      Object.keys(CASES).forEach(k => { r[k] = {mine: D.techRows(CASES[k].manifest, null), classic: classicTech(CASES[k].meta)}; });
      out(r);
    """, pre=pre.replace("const COMMON=globalThis.WssReportCommon;", "let COMMON;").replace("globalThis.classicTech=m=>{", "globalThis.classicTech=m=>{COMMON=globalThis.WssReportCommon;"))
    rename = {"run_identity": "运行身份"}
    for job_id, r in out.items():
        mine = dict(r["mine"])
        assert r["classic"], job_id
        for k, v in r["classic"]:
            assert mine.get(rename.get(k, k)) == v, (job_id, k, v, mine.get(rename.get(k, k)))
        assert {"数据版本", "分析版本", "部署版本", "计算设备", "坐标架方向来源"} <= set(mine)          # the workspace's own rows as well


def test_manifest_carries_the_new_provenance_and_offline_page_embeds_the_glossary(tmp_path):
    meta = {"model_release": {"registry_id": "REL_X", "name": "REL_X", "model_family": "M1_3head", "models": [1, 2, 3]}, "created_at": "2026-09-20 01:04:10",
            "audit": {"mapping_history": [{"at": "2026-09-20 09:00:00"}, {"at": "2026-09-20T14:41:50+08:00"}, {"at": "2026-09-21T01:00:00Z"}], "report_rebuilt_at": "2026-09-30T05:01:55+08:00"},
            "gpu": "RTX"}
    tech = V._tech_provenance(meta, {})
    assert tech == {"model_family": "M1_3head", "n_models": 3, "created_at": "2026-09-20 01:04:10", "time_hint": "2026-09-20T14:41:50+08:00",
                    "report_rebuilt_at": "2026-09-30T05:01:55+08:00", "gpu": "RTX"}
    assert V._tech_provenance({"model_release": {"weights": ["a", "b"]}}, {"mapping_history": [{"at": "2026-01-01T00:00:00+08:00"}]})["n_models"] == 2
    assert V._tech_provenance({}, {"mapping_history": [{"at": "2026-01-01T00:00:00+08:00"}]})["time_hint"] == "2026-01-01T00:00:00+08:00"
    assert V._tech_provenance({}, {})["n_models"] is None
    # offline page: the glossary travels with the page (fake bundle as in test_v2_offline)
    spec = importlib.util.spec_from_file_location("offline_t", Path(__file__).with_name("test_v2_offline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    static, v2 = mod._fake_bundle(tmp_path)
    job = next((p for p in (_job_dir(j) for j in BANNER_JOBS) if p), None)
    if job is None:
        pytest.skip("no job copies present")
    V.clear_cache()
    record = json.loads((job / "job.json").read_text(encoding="utf-8"))
    data = V.load(job)
    html = O.build_offline_html(data, V.build_manifest(data, record), record=record, exported_at="2026-09-30T10:00:00+08:00", static_dir=static, v2_dir=v2)
    V.clear_cache()
    gloss = mod._scripts(html, "wss-glossary")
    assert gloss["terms"]["quality_grade"]["zh_desc"].endswith("一致不代表准确。")
    assert json.loads((STATIC_DIR / "glossary.json").read_text(encoding="utf-8"))["terms"] == gloss["terms"]
    manifest = mod._scripts(html, "wssv2-manifest")
    assert "n_models" in manifest["provenance"] and "time_hint" in manifest["provenance"]


def test_tech_rows_times_and_fields():
    out = _node(r"""
      const m = {data_version: 'dv', result: {run_identity: 'rid', release_id: 'REL', analysis_version: 'a1', deploy_version: '0.15', model_frame: {target: 'peak_systole', time_s: 0.21}},
        job: {input_sha256: 'sha'}, frame: {direction_source: 'unknown_stl'}, mapping: {display_interpolation: {method: 'gaussian', sigma_mm: 0.5, max_dist_mm: 1.5}},
        time: {axis: [{index: 0, label: 'peak_systole', step: 1162, time_s: 0.21}], cycle: {frames: '0-79 of 81', period_s: 0.8}},
        provenance: {release_hash: 'h', model_release: {registry_id: 'REL_X'}, model_family: 'M1_3head', n_models: 3, feature_contract: {version: 'wss-features/1.0', source_hash: 'b4b8'},
          created_at: '2026-09-20 01:04:10', time_hint: '2026-09-20T14:41:50+08:00', report_rebuilt_at: '2026-09-30T05:01:55+08:00', report_schema: 'wss-deploy.summary/v1',
          device: 'cuda', gpu: 'RTX', git_describe: 'v0.15'}};
      const rows = D.techRows(m, null);
      out({rows, created: C.localTime('2026-09-20T01:04:10+08:00', {seconds: true}) + '（2026-09-20 01:04:10）',
        rebuilt: C.localTime('2026-09-30T05:01:55+08:00', {seconds: true}) + '（2026-09-30T05:01:55+08:00）', empty: D.techRows({}, null)});
    """)
    rows = dict(out["rows"])
    assert rows["发布包"] == "REL_X" and rows["模型族"] == "M1_3head · 3 个模型" and rows["特征合同"] == "wss-features/1.0 · b4b8"
    assert rows["模型帧"] == "peak_systole · step 1162 · 0.21 s" and rows["周期定义"] == "0-79 of 81 · 周期 0.8 s"
    assert rows["生成时间"] == out["created"] and rows["报告重建"] == out["rebuilt"]
    assert rows["计算设备"] == "cuda · RTX" and rows["坐标架方向来源"] == "按解剖坐标架推断（STL 无患者方向）" and rows["显示插值"] == "gaussian，σ 0.5 mm，最大距离 1.5 mm"
    assert out["empty"] == []


# ---------------------------------------------------------------------------------------------------------------- overview models
def test_geometry_outlets_volume_notes_models():
    out = _node(r"""
      const m = {result: {family: 'wall'}, geometry: {branches: [{id: 0, name: '主动脉'}, {id: 2, name: '左髂总'}],
          openings: [{name: 'inlet', label: '入口（主动脉）', kind: 'inlet', radius_mm: 12.617, flow_share: null}, {name: 'out-le', label: '左髂外', kind: 'outlet', radius_mm: 3.887, flow_share: 0.39187},
            {name: 'out-li', label: '左髂内', kind: 'outlet', radius_mm: 2.486, flow_share: 0.10813}]},
        frame: {orientation: {left_right: 'confirmed'}},
        analysis: {branch_geometry: {'左髂总': {length_mm: 177.82, radius_min_mm: 4.69, radius_median_mm: 6.67, max_diameter_mm: 18.59, tortuosity: 1.1147},
          '主动脉': {length_mm: 101.1, radius_min_mm: 8.963, radius_median_mm: 10.78, max_diameter_mm: 23.39, tortuosity: 1.0539}, '右髂外': {length_mm: 50}}}};
      const vol = {result: {family: 'volume'}, analysis: {streamlines: {line_count: 527}, pressure_reference: '相对于该病例当前帧的体积平均压力；不可解释为绝对血压。'}};
      out({geo: OV.geometryModel(m), os: OV.outletShareModel(m, null), osAuto: OV.outletShareModel(m, {mapping_history: [{source: 'automatic_high_confidence'}]}).confirmed,
        osMan: OV.outletShareModel(m, {mapping_history: [{source: 'manual_override'}]}).confirmed, osOff: OV.outletShareModel(Object.assign({}, m, {frame: {}}), null).confirmed,
        vol: OV.volumeNotesModel(vol), wall: OV.volumeNotesModel(m), noGeo: OV.geometryModel({analysis: {}}), pr: L.pressureReference(vol), prObj: L.pressureReference({analysis: {pressure_reference: {label: 'X 参考'}}}),
        prNone: L.pressureReference({analysis: {}})});
    """)
    assert out["geo"][0] == {"name": "主动脉", "length": "101", "rmin": "9.0", "rmed": "10.8", "dmax": "23.4", "tort": "1.05"}   # classic fmt(x,0/1/1/1/2)
    assert [g["name"] for g in out["geo"]] == ["主动脉", "左髂总", "右髂外"] and out["geo"][2]["rmin"] == "—"
    assert out["os"]["rows"] == [{"name": "入口（主动脉）", "kind": "inlet", "radius": "12.6", "share": None}, {"name": "左髂外", "kind": "outlet", "radius": "3.9", "share": "39%"},
                                 {"name": "左髂内", "kind": "outlet", "radius": "2.5", "share": "11%"}]
    assert out["os"]["confirmed"] == "出口映射已人工确认" and out["osOff"] == "出口映射为自动建议"
    assert out["osAuto"].startswith("出口映射为自动确认") and out["osMan"] == "出口映射已人工确认"
    v = out["vol"]
    assert v["lineCount"] == 527 and v["streamlines"].startswith("已载入 527 条由预测速度向量积分的流线；离开有效支撑区域即停止。")
    assert v["pressure"] == "压力参考：相对于该病例当前帧的体积平均压力；不可解释为绝对血压。"
    assert [t["label"] for t in v["terms"]] == ["相对压力", "ΔP 压差", "速度大小", "采样支撑", "近切口", "几何越界", "插值无支撑"]
    assert out["wall"] is None and out["noGeo"] is None and out["prObj"] == "X 参考" and out["prNone"].startswith("请参阅该发布包的参考压定义")


def test_followup_growth_recent_and_release_lines():
    out = _node(r"""
      const tl = {scans: [
        {date: '2023-09-01', date_source: 'scan_date', geometry: {max_diameter_mm: 50, sac_volume_ml: 90}, models: {X5D: {wss_p99_pa: 10}}, jobs: [{job_id: 'w1', release_id: 'X5D', release_label: '壁面 WSS'}]},
        {date: '2024-03-15', date_source: 'scan_date', geometry: {max_diameter_mm: 52, sac_volume_ml: 96}, models: {M1: {tawss_mean_pa: 0.7, wss_p99_pa: 11}}, jobs: [{job_id: 'a', release_id: 'M1', release_label: 'WSS + TAWSS + OSI'}]},
        {date: '2025-09-20', date_source: 'created_at', geometry: {max_diameter_mm: 55, sac_volume_ml: null}, models: {M1: {tawss_mean_pa: 0.6, wss_p99_pa: 12}, X5D: {wss_p99_pa: 13}},
          jobs: [{job_id: 'b', release_id: 'M1'}, {job_id: 'w2', release_id: 'X5D'}]}],
        growth: {max_diameter_mm: {per_year: 2.52, delta: 5, days: 750, from: {date: '2023-09-01'}, to: {date: '2025-09-20'}}, sac_volume_ml: {per_year: 11.9, delta: 6, days: 196, from: {date: '2023-09-01'}, to: {date: '2024-03-15'}}},
        growth_recent: {max_diameter_mm: {per_year: 1.9, delta: 3, days: 554, from: {date: '2024-03-15'}, to: {date: '2025-09-20'}}}, notes: ['n1']};
      const fu = OV.followupModel(tl, 'M1');
      const charts = OV.followupCharts(fu, 'a');
      out({ga: fu.growthAll, releases: fu.releases, charts: charts.map(c => [c.label, c.lines.map(l => [l.id, l.current, l.points.map(p => [p.value, p.jobId, p.current])])]), metric: fu.metricKey,
        rows: fu.rows.map(r => [r.dateSource, r.sacVolume])});
    """)
    ga = out["ga"]
    assert ga["diameter"] == {"perYear": 2.52, "delta": 5, "days": 750, "from": "2023-09-01", "to": "2025-09-20"}
    assert ga["volume"]["perYear"] == 11.9 and ga["diameterRecent"]["perYear"] == 1.9 and ga["volumeRecent"] is None
    assert [r["id"] for r in out["releases"]] == ["X5D", "M1"] and [r["current"] for r in out["releases"]] == [False, True]
    charts = dict((c[0], c[1]) for c in out["charts"])
    assert list(charts) == ["管腔最大直径", "瘤体体积", "TAWSS 均值", "WSS p99"]
    assert charts["瘤体体积"][0][2] == [[90, "w1", False], [96, "a", True]]                  # a missing value is not a zero
    assert [l[0] for l in charts["WSS p99"]] == ["M1", "X5D"] and charts["WSS p99"][0][1] is True   # current release first
    assert charts["WSS p99"][1][2] == [[10, "w1", False], [13, "w2", False]]                   # the other release's own jobs
    assert out["metric"] == "tawss_mean_pa" and out["rows"][2] == ["created_at", None]


def test_glossary_loads_and_falls_back():
    out = _node(r"""
      const before = L.termText('quality_grade', '内置');
      const offline = await L.loadGlossary({offline: true});
      globalThis.fetch = async url => ({ok: true, json: async () => ({terms: {quality_grade: {zh: '多模型一致性', zh_desc: '来自术语表'}}})});
      const terms = await L.loadGlossary({offline: false});
      out({before, offline, keys: Object.keys(terms), after: L.termText('quality_grade', '内置'), missing: L.termText('nope', '内置'), tip: typeof L.termTip});
    """)
    assert out["before"] == "内置" and out["offline"] is None
    assert out["keys"] == ["quality_grade"] and out["after"] == "来自术语表" and out["missing"] == "内置"


# ---------------------------------------------------------------------------------------------------------------- shell integration
def _workspace():
    spec = importlib.util.spec_from_file_location("ws_harness_p3", Path(__file__).with_name("test_v2_workspace_js.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shell(scenario: str, offline: bool = False) -> dict:
    ws = _workspace()
    files = list(ws.WS_FILES)
    at = files.index("ws_lens.js") + 1
    files[at:at] = ["ws_detail.js", "ws_unroll.js", "ws_profile.js", "ws_review.js", "ws_notice.js"]
    if offline:
        files = [f for f in files if f not in ws.OFFLINE_EXCLUDE]
    return ws._run(scenario, offline=offline, files=["../report_common.js"] + files)


_SETUP = r"""
  const B = id => canned['/api/v2/jobs/' + id + '/manifest'].body;
  const warn = id => { const b = B(id); b.analysis.reference_assessment = {status: 'review', note: '仅检查已声明的几何参考范围。',
      checks: [{path: 'geometry.左髂总.length_mm', units: 'mm', value: 177.82, min: 21.116, max: 141.016, status: 'review'}], population: {status: 'unknown'}};
    b.analysis.quality = {level: 'review', label: '存在不确定性，建议复核', reasons: ['3 个模型离散度超过常规范围，建议复核热点和分支']};
    b.provenance = Object.assign({}, b.provenance, {n_models: 3}); };
  const host = () => byClass(app(), 'ws-notices')[0];
  const dlg = () => byId('ws-dialog');
  const btn = (root, text) => walk(root, e => e.tagName === 'BUTTON' && textOf(e).trim() === text)[0];
  const tab = id => walk(app(), e => e.dataset && e.dataset.tab === id)[0];
"""


def test_banner_on_the_stage_details_close_and_reopen():
    out = _shell(_SETUP + r"""
      warn('A');
      await boot();
      await hashTo('#/job/A', 150);
      const bar = textOf(host()), role = (byClass(host(), 'ws-notice')[0] || {attrs: {}}).attrs.role;
      fire(btn(host(), '详情'), 'click'); await wait(20);
      const details = textOf(dlg());
      const dlgOpen = dlg() && dlg().open;
      if (dlg()) dlg().close();
      const qsec = byClass(app(), 'sec-quality').map(textOf)[0] || '';
      fire(walk(host(), e => e.tagName === 'BUTTON' && (e.attrs['aria-label'] || e.title || '').indexOf('关闭提示') >= 0)[0], 'click'); await wait(10);
      const closed = textOf(host());
      await hashTo('#/job/B', 150);
      const other = textOf(host());
      await hashTo('#/job/A', 150);
      const again = textOf(host());
      done({bar, role, details, dlgOpen, qsec, closed, other, again});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["bar"].startswith("注意：输入几何有 1 项超出发布包参考范围（左髂总长度 178 mm，参考 21.1–141 mm）；模型集成质量：存在不确定性，建议复核。") and out["role"] == "alert"
    assert "左髂总长度 178 mm（参考 21.1–141 mm）" in out["details"] and "3 个模型一致性" in out["details"] and "逐点离散度只保存在质量审计文件里" in out["details"]
    assert out["details"].count("仅检查已声明的几何参考范围。") == 1                 # the note once
    assert out["dlgOpen"] is True
    assert "质量与参照" in out["qsec"] and "1 项超出已声明几何参考范围，请复核" in out["qsec"] and "存在不确定性，建议复核" in out["qsec"]
    assert out["closed"] == "" and out["other"] == ""                        # × hides it for this opening; B has nothing to say
    assert out["again"].startswith("注意：")                                # opening A again shows it again (classic: 仅本次打开)


def test_banner_and_tech_info_offline():
    out = _shell(r"""
      const m = manifestFor('A');
      m.analysis.quality = {level: 'poor', label: '不稳定，建议复核', reasons: []};
      m.provenance = {release_hash: 'abc', model_family: 'M1_3head', n_models: 3, feature_contract: {version: 'wss-features/1.0'}};
      const s1 = new Element('script'); s1.id = 'wssv2-manifest'; s1.textContent = JSON.stringify(m); body.appendChild(s1);
      const s2 = new Element('script'); s2.id = 'wssv2-offline'; s2.textContent = JSON.stringify({exported_at: '2026-09-30T09:00:00+08:00', view: {}}); body.appendChild(s2);
      const s3 = new Element('script'); s3.id = 'wss-glossary'; s3.textContent = JSON.stringify({terms: {quality_grade: {zh: '多模型一致性', zh_desc: '术语表：一致不代表准确。'}}}); body.appendChild(s3);
      const appEl = new Element('div'); appEl.id = 'ws-app'; body.appendChild(appEl);
      calls.length = 0;
      await boot();
      await wait(120);
      const bar = textOf(byClass(app(), 'ws-notices')[0]);
      const techBtn = walk(app(), e => e.tagName === 'BUTTON' && textOf(e).trim() === '技术信息')[0];
      fire(techBtn, 'click'); await wait(20);
      const tech = textOf(byId('ws-dialog'));
      const links = walk(byId('ws-dialog'), e => e.tagName === 'A').length;
      const qtip = (walk(app(), e => e.dataset && e.dataset.q === 'quality')[0] || {}).title;
      done({bar, tech, links, qtip, fetches: calls.filter(c => /^(GET|POST|PUT) /.test(c))});
    """, offline=True)
    assert out["errors"] == [], out["errors"]
    assert out["bar"].startswith("注意：模型集成质量：不稳定，建议复核。")
    assert "模型族" in out["tech"] and "M1_3head · 3 个模型" in out["tech"] and "特征合同" in out["tech"] and out["links"] == 0
    assert out["qtip"].startswith("术语表：一致不代表准确。")                # the embedded glossary, no request
    assert out["fetches"] == []


def test_review_dialog_facts_and_admin_timeline_follows_view_all():
    out = _shell(_SETUP + r"""
      warn('A');
      canned['/api/jobs/A'] = {body: {job: jobRecord('A', {mapping_history: [{source: 'automatic_high_confidence', confidence: 0.9612}], patient_id: 'P-1'})}};
      B('A').job.patient_id = 'P-1';
      const urls = [];
      const f0 = global.fetch; global.fetch = (u, o) => { urls.push(String(u)); return f0(u, o); };
      canned['/api/patients/P-1/timeline'] = {body: {patient_id: 'P-1', n_scans: 1, scans: [{date: '2025-01-01'}], growth: {}, notes: []}};
      await boot();
      await hashTo('#/job/A', 150);
      const first = urls.filter(u => u.indexOf('/timeline') >= 0);
      fire(btn(byClass(app(), 'insp-head')[0], '技术复核'), 'click'); await wait(20);
      const facts = byClass(dlg(), 'review-facts').map(textOf)[0] || '';
      dlg().close();
      ns.admin = {viewAll: () => true};
      await hashTo('#/job/B', 100); await hashTo('#/job/A', 150);
      const second = urls.filter(u => u.indexOf('/timeline') >= 0);
      done({first, second, facts});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["first"] == ["/api/patients/P-1/timeline"]                 # admin, 「看全部用户」 off: own jobs only
    assert out["second"][-1] == "/api/patients/P-1/timeline?all=1"
    assert "出口命名自动确认（置信度 96.1%）" in out["facts"] and "几何参考范围有超出范围的测量" in out["facts"]
    assert "需要注意1 项几何测量超出模型训练范围，模型集成存在不确定性，结果需要复核。" in out["facts"]


def test_followup_section_renders_growth_legend_and_table():
    out = _shell(_SETUP + r"""
      B('A').job.patient_id = 'P-2';
      canned['/api/jobs/A'] = {body: {job: jobRecord('A', {patient_id: 'P-2'})}};
      canned['/api/patients/P-2/timeline'] = {body: {patient_id: 'P-2', n_scans: 3, notes: ['年增长率只在两次扫描都填写了扫描日期时计算。'],
        scans: [{date: '2023-09-01', date_source: 'scan_date', geometry: {max_diameter_mm: 50, sac_volume_ml: 90}, models: {X5D: {wss_p99_pa: 10}, M1_3head_3seed_20260922: {tawss_mean_pa: 0.8, wss_p99_pa: 9}},
            jobs: [{job_id: 'w1', release_id: 'X5D', release_label: '壁面 WSS'}, {job_id: 'm0', release_id: 'M1_3head_3seed_20260922'}]},
          {date: '2024-03-15', date_source: 'scan_date', geometry: {max_diameter_mm: 52, sac_volume_ml: 96}, models: {M1_3head_3seed_20260922: {tawss_mean_pa: 0.7, wss_p99_pa: 11}}, jobs: [{job_id: 'A', release_id: 'M1_3head_3seed_20260922'}]},
          {date: '2025-09-20', date_source: 'scan_date', geometry: {max_diameter_mm: 55, sac_volume_ml: 104}, models: {X5D: {wss_p99_pa: 13}}, jobs: [{job_id: 'w2', release_id: 'X5D', release_label: '壁面 WSS'}]}],
        growth: {max_diameter_mm: {per_year: 2.52, delta: 5, days: 750, from: {date: '2023-09-01'}, to: {date: '2025-09-20'}}, sac_volume_ml: {per_year: 6.8, delta: 14, days: 750, from: {date: '2023-09-01'}, to: {date: '2025-09-20'}}},
        growth_recent: {max_diameter_mm: {per_year: 2.03, delta: 3, days: 554, from: {date: '2024-03-15'}, to: {date: '2025-09-20'}}}}};
      await boot();
      await hashTo('#/job/A', 200);
      const sec = byClass(app(), 'sec-follow')[0];
      const growth = byClass(sec, 'fu-growth').map(textOf), charts = byClass(sec, 'fu-chart').map(e => textOf(walk(e, x => x.tagName === 'FIGCAPTION')[0]));
      const legend = byClass(sec, 'fu-legend').map(textOf);
      const head = walk(sec, e => e.tagName === 'TH').map(textOf);
      done({growth, charts, legend, head});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["growth"][0] == "+2.5mm/年管腔最大直径最近两次 +2.0 mm/年" and out["growth"][1] == "+6.8mL/年瘤体体积"
    assert out["charts"] == ["管腔最大直径mm", "瘤体体积mL", "TAWSS 均值Pa", "WSS p99Pa"]
    assert out["legend"] == ["周期指标（本结果）壁面 WSS"]                        # the model card's short name; the other release by its family
    assert out["head"] == ["扫描", "管腔最大直径 mm", "瘤体体积 mL", "TAWSS 均值 Pa"]


def test_geometry_section_and_tools_tech_rows():
    out = _shell(_SETUP + r"""
      localStorage.setItem('wssv2:prefs:2', JSON.stringify({tier: 'full'}));
      B('A').analysis.branch_geometry = {'主动脉': {length_mm: 101.1, radius_min_mm: 8.963, radius_median_mm: 10.78, max_diameter_mm: 23.39, tortuosity: 1.0539}};
      B('A').geometry.openings = [{name: 'out-le', label: '左髂外', kind: 'outlet', radius_mm: 3.887, flow_share: 0.39187}];
      B('A').provenance = {release_hash: 'abc', model_family: 'M1_3head', n_models: 3};
      await boot();
      await hashTo('#/job/A', 150);
      const sec = byClass(app(), 'sec-geo')[0];
      const hiddenBefore = byClass(sec, 'geo-body')[0].hidden;
      fire(btn(sec, '展开'), 'click'); await wait(10);
      const shown = !byClass(byClass(app(), 'sec-geo')[0], 'geo-body')[0].hidden, text = textOf(sec);
      fire(tab('tools'), 'click'); await wait(30);
      const tools = textOf(byClass(app(), 'insp-body')[0]);
      done({hiddenBefore, shown, text, tools});
    """)
    assert out["errors"] == [], out["errors"]
    assert out["hiddenBefore"] is True and out["shown"] is True
    assert "主动脉101" in out["text"] and "23.4" in out["text"] and "1.05" in out["text"] and "左髂外3.939%" in out["text"]
    assert "模型族M1_3head · 3 个模型" in out["tools"] and "运行身份run-A" in out["tools"]   # the 工具 tab shows the same rows


# The section tool on the harness: a stub ws_slice session records what the shell asks for.
_SLICE = r"""
  const sliceLog = [];
  ns.slice = {supported: () => true, requiredArrays: () => [],
    create(v, r, o) {
      let st = Object.assign({basis: 'centerline', segment: 0, fraction: 0.5, pitch: 0, yaw: 0, offU: 0, offV: 0, thickness: 2, cut: 'none'}, o.state || {});
      const panels = [];
      sliceLog.push(['create', JSON.parse(JSON.stringify(st))]);
      return {state: () => JSON.parse(JSON.stringify(st)), set(p) { Object.assign(st, p); sliceLog.push(['set', JSON.parse(JSON.stringify(st))]); panels.forEach(x => { if (x.alive()) x.update(); }); },
        look() { sliceLog.push(['look']); }, hudText: () => '截面', colorbarInfo: () => null, cut: () => 'none', dispose() {}, disposed: () => false, picking: () => false,
        clickTaken: () => false, key: () => false, arcOf: sid => ({0: 200, 4: 40})[sid] || 0, addPanel(p) { panels.push(p); }, refresh() {}};
    },
    panel(body) { body.replaceChildren(); }};
  const volume = id => { const b = B(id); b.result.family = 'volume';
    b.fields = [{id: 'speed', label: '速度', short_label: '速度', units: 'm/s', kind: 'scalar', arrays: {read: 'f.speed.r'}}, {id: 'pressure', label: '相对压力', short_label: '压力', units: 'Pa', kind: 'scalar', arrays: {read: 'f.p.r'}}];
    b.geometry.branches = [{id: 0, name: '主动脉', length_mm: 200}, {id: 4, name: '左髂内', length_mm: 40}];
    b.analysis.pressure_reference = '相对于该病例当前帧的体积平均压力；不可解释为绝对血压。';
    b.analysis.findings = {items: [{id: 'F3', kind: 'pressure_drop', label: '左髂内沿程压降', branch: '左髂内', segment_id: 4, value: 981.8, units: 'Pa', xyz_mm: [1, 2, 3], severity: 'note', definition: '近端 − 远端'}]};
    b.analysis.profiles = {bin_mm: 2, branches: [{segment_id: 0, name: '主动脉', s_local_mm: [0, 100, 200], s_from_root_mm: [10, 110, 210], radius_mm: [10, 10, 10],
      volume: {n: [5, 5, 5], speed_mean_m_s: [0.3, 0.2, 0.1], speed_max_m_s: [0.5, 0.4, 0.3], pressure_mean_pa: [10, 5, 0], pressure_min_pa: [-5, -6, -7]}}]};
    b.analysis.zones = null; };
"""


def test_pressure_drop_puts_the_section_at_ten_percent_and_the_profile_click_moves_it():
    out = _shell(_SETUP + _SLICE + r"""
      volume('A');
      await boot();
      await hashTo('#/job/A', 150);
      const toasts = [];
      const t0 = ns.ui.toast; ns.ui.toast = (m, o) => { toasts.push(String(m)); return t0(m, o); };
      fire(walk(app(), e => e.dataset && e.dataset.findingId === 'F3')[0], 'click'); await wait(40);
      const afterFinding = sliceLog.slice(), tabNow = shellState().tab;
      // the lens of the finding offers 近端 10% / 远端 90%, and names the pressure reference
      shellState().tab = 'reading'; ns.shell && 0;
      fire(tab('reading'), 'click'); await wait(20);
      const lens = textOf(byClass(app(), 'insp-body')[0]);
      fire(btn(byClass(app(), 'insp-body')[0], '远端 90%'), 'click'); await wait(20);
      const far = sliceLog[sliceLog.length - 2];
      // 沿程 tab: a click on the curve moves the open section (stub rect 30 px: clientX 25 → the middle of the x range)
      fire(tab('along'), 'click'); await wait(40);
      const bodyEl = byClass(app(), 'insp-body')[0];
      const byCls = (root, cls) => walk(root, e => String((e.attrs && e.attrs.class) || e.className || '').split(/\s+/).includes(cls));
      const hint = textOf(byClass(bodyEl, 'along-read')[0]);
      const n0 = sliceLog.length;
      fire(byCls(bodyEl, 'along-main')[0], 'click', {clientX: 25, clientY: 20}); await wait(20);
      const moved = sliceLog.slice(n0);
      const mark = (byCls(byClass(app(), 'insp-body')[0], 'along-mark-slice')[0] || {attrs: {}}).attrs.visibility;
      done({afterFinding, tabNow, toasts, lens, far, hint, moved, mark});
    """)
    assert out["errors"] == [], out["errors"]
    create = [x for x in out["afterFinding"] if x[0] == "create"][0][1]
    assert create["basis"] == "centerline" and create["segment"] == 4 and create["fraction"] == pytest.approx(0.1) and create["pitch"] == 0
    assert out["tabNow"] == "slice"
    assert any(t.startswith("压降定义：左髂内 近端 10% 与远端 10% 弧长段的平均压差。截面已放在近端 10%；把位置滑到 90% 查看远端。") for t in out["toasts"])
    assert "截面：近端 10%" in out["lens"] and "压力参考相对于该病例当前帧的体积平均压力" in out["lens"]
    assert out["far"][0] == "set" and out["far"][1]["segment"] == 4 and out["far"][1]["fraction"] == pytest.approx(0.9)
    set_ = [x for x in out["moved"] if x[0] == "set"]
    assert set_ and set_[-1][1]["segment"] == 0 and set_[-1][1]["basis"] == "centerline"
    # x range 10–210 mm from the root, local = root − 10; the middle (110 mm) is local 100 of a 200 mm centreline
    assert set_[-1][1]["fraction"] == pytest.approx(0.5, abs=0.02)
    assert out["mark"] == "visible"


def test_followup_end_labels_never_overlap():
    out = _shell(r"""
      await boot();
      const OVm = ns.overview;
      const charts = [{label: 'WSS p99', units: 'Pa', tier: 'model', points: [], lines: [
        {id: 'M1', label: 'M1', current: true, points: [{date: '2024-03-15', value: 11, jobId: 'a'}, {date: '2025-09-20', value: 17.0, jobId: 'b'}]},
        {id: 'X5D', label: 'X5D', current: false, points: [{date: '2023-09-01', value: 10, jobId: 'c'}, {date: '2025-09-20', value: 16.6, jobId: 'd'}]},
        {id: 'PF6', label: 'PF6', current: false, points: [{date: '2023-09-01', value: 16.8, jobId: 'e'}, {date: '2025-09-20', value: 16.8, jobId: 'f'}]}]}];
      const el = OVm.followSparks(charts, () => {}, {});
      const vals = walk(el, e => String((e.attrs && e.attrs.class) || '').split(/\s+/).includes('fu-val')).map(e => [Number(e.attrs.x), Number(e.attrs.y), textOf(e)]);
      const legend = walk(el, e => String(e.className || '').split(/\s+/).includes('fu-leg')).map(textOf);
      done({vals, legend});
    """)
    assert out["errors"] == [], out["errors"]
    vals = out["vals"]
    assert vals[0][2] == "17.0"                                                     # the current release keeps its label above
    for i, a in enumerate(vals):
        for b in vals[i + 1:]:
            assert not (abs(a[0] - b[0]) < 30 and abs(a[1] - b[1]) < 10), vals    # no two labels on top of each other
    assert len(vals) == 2 and out["legend"] == ["M1（本结果）", "X5D", "PF6"]          # the third is left to the tip and the table
