"""§19.2 cycle quantities (M1 three-head release) through findings, profiles, narrative, quality, tables and one-pager.

Every addition is keyed on the presence of TAWSS / OSI: without them each output must stay byte-for-byte what the
peak-WSS releases (X5D) and the volume release (PF6/VF6) produced before.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_analysis import NAMES, _atlas, _wall_points
from tests.test_narrative import VOLUME, WALL
from tests.test_onepager import wall_summary
from wss_deploy import analysis as A, cycle_fields as CF, narrative as N
from wss_deploy.comparison import compare_summaries
from wss_deploy.export_table import COLUMNS, CYCLE_COLUMNS, WALL_COLUMNS, VOLUME_COLUMNS, build_row
from wss_deploy.onepager import LIMIT_CYCLE_FRAME, LIMIT_SINGLE_FRAME, render_onepage, top_findings
from wss_deploy.quality import ensemble_quality, model_count_phrase

KW = dict(thresholds=[0.4, 4, 7], total_area_mm2=1000.0, spacing_mm=1.0, branch_names=NAMES, min_cluster_points=20)


@pytest.fixture(scope="module")
def tube():
    atlas = _atlas()
    pts, feats = _wall_points(atlas)
    n = len(pts)
    wss = np.full(n, 2.0)
    wss[(feats["segment_id"] == 1) & (feats["s_local_mm"] >= 5) & (feats["s_local_mm"] <= 7)] = 12.0
    tawss = np.full(n, 1.5); osi = np.full(n, 0.02)
    sac = (feats["segment_id"] == 0) & (feats["s_local_mm"] >= 15) & (feats["s_local_mm"] <= 25)      # 21 rings
    small = (feats["segment_id"] == 0) & (feats["s_local_mm"] >= 2) & (feats["s_local_mm"] <= 4)      # 5 rings
    tawss[sac | small] = 0.2; osi[sac] = 0.2; osi[small] = 0.15
    hot = (feats["segment_id"] == 2) & (feats["s_local_mm"] >= 14) & (feats["s_local_mm"] <= 16)      # high OSI only
    osi[hot] = 0.4; osi[hot & (feats["s_local_mm"] == 15)] = 0.45
    return atlas, pts, feats, wss, {"tawss": tawss, "osi": osi}, {"sac": sac, "small": small, "hot": hot}


def test_findings_without_cycle_are_byte_identical_and_cycle_items_are_appended(tube):
    atlas, pts, feats, wss, cycle, masks = tube
    base = A.findings_wall(pts, wss, feats, atlas, {}, **KW)
    assert json.dumps(A.findings_wall(pts, wss, feats, atlas, {}, cycle=None, **KW), sort_keys=True) == json.dumps(base, sort_keys=True)
    # The historical listing (no area floor) first; the 2026-09-30 1 cm² floor is checked at the end.
    out = A.findings_wall(pts, wss, feats, atlas, {}, cycle=cycle, area_floor_mm2=0.0, **KW)
    n_base = len(base["items"])
    assert json.dumps(out["items"][:n_base], sort_keys=True) == json.dumps(base["items"], sort_keys=True)   # prefix unchanged
    extra = out["items"][n_base:]
    assert [i["kind"] for i in extra] == ["stagnation_cluster", "stagnation_cluster", "high_osi_cluster"]
    assert [i["id"] for i in out["items"]] == [f"F{k}" for k in range(1, len(out["items"]) + 1)]
    big, small, hot = extra
    n = len(pts)
    assert big["n_points"] == int(masks["sac"].sum()) and big["severity"] == "attention" and big["branch"] == "主动脉"
    assert big["units"] == "cm²" and big["value"] == pytest.approx(1000.0 * masks["sac"].sum() / n / 100.0)
    assert big["area_mm2"] == pytest.approx(100.0 * big["value"]) and big["label"] == "主动脉滞留区"
    assert big["tawss_mean_pa"] == pytest.approx(0.2) and big["osi_mean"] == pytest.approx(0.2) and big["point_indices"]
    assert 15 <= big["s_from_root_mm"] <= 25 and "TAWSS < 0.4 Pa 且 OSI > 0.1" in big["definition"]
    assert small["severity"] == "info" and small["n_points"] == int(masks["small"].sum())
    assert hot["branch"] == "右髂总" and hot["label"] == "右髂总高 OSI 区" and hot["units"] == "1" and hot["severity"] == "info"
    assert hot["value"] == pytest.approx(0.45) and hot["n_points"] == int(masks["hot"].sum())
    assert out["cycle_criteria"] == {"stagnation": {"tawss_lt_pa": 0.4, "osi_gt": 0.1}, "high_osi_gt": 0.3, "max_clusters": 3}
    assert "cycle_criteria" not in base
    json.dumps(out, allow_nan=False)
    # OSI alone still yields the high-OSI clusters; malformed arrays fail loudly.
    osi_only = A.findings_wall(pts, wss, feats, atlas, {}, cycle={"osi": cycle["osi"]}, area_floor_mm2=0.0, **KW)
    assert [i["kind"] for i in osi_only["items"][n_base:]] == ["high_osi_cluster"]
    # U12 (2026-09-30): with the default 1 cm² floor the small stagnation (≈ 31 mm²) and high-OSI (≈ 31 mm²)
    # clusters are not listed; the big one stays first (and stays 'attention'); the drop is recorded.
    floored = A.findings_wall(pts, wss, feats, atlas, {}, cycle=cycle, **KW)
    assert [i["kind"] for i in floored["items"][n_base:]] == ["stagnation_cluster"]
    assert floored["items"][n_base]["severity"] == "attention" and floored["items"][n_base]["n_points"] == big["n_points"]
    suppressed = floored["listing_rules"]["suppressed"]
    assert suppressed["stagnation_cluster"]["count"] == 1 and suppressed["high_osi_cluster"]["count"] == 1
    assert floored["listing_rules"]["area_floor_mm2"] == 100.0
    with pytest.raises(ValueError):
        A.findings_wall(pts, wss, feats, atlas, {}, cycle={"tawss": cycle["tawss"][:-1], "osi": cycle["osi"]}, **KW)


def test_profiles_add_cycle_curves_on_the_wss_bins(tube):
    atlas, pts, feats, wss, cycle, _ = tube
    keep = ~((feats["segment_id"] == 0) & (feats["s_local_mm"] >= 30) & (feats["s_local_mm"] < 34))
    base = A.profiles(atlas, feats, {"wss": wss}, point_mask=keep, branch_names=NAMES)
    out = A.profiles(atlas, feats, {"wss": wss, **cycle}, point_mask=keep, branch_names=NAMES)
    for before, after in zip(base["branches"], out["branches"]):
        assert {k: v for k, v in after.items() if k not in ("tawss", "osi")} == before
        assert set(after["tawss"]) == {"mean_pa", "min_pa"} and set(after["osi"]) == {"mean", "p90"}
        for block in (after["tawss"], after["osi"]):
            for values in block.values():
                assert len(values) == after["n_bins"]
                assert [v is None for v in values] == [c == 0 for c in after["wss"]["n"]]
    root = out["branches"][0]
    assert root["tawss"]["mean_pa"][9] == pytest.approx(0.2) and root["osi"]["p90"][9] == pytest.approx(0.2)
    assert root["tawss"]["mean_pa"][15] is None                           # the masked gap stays empty
    json.dumps(out, allow_nan=False)


def _cycle_block(n=1000):
    seg = np.repeat([0, 1, 2], [700, 200, 100])
    tawss = np.where(seg == 0, 0.3, 1.2); tawss[:50] = 3.3
    osi = np.where(seg == 0, 0.2, 0.05); osi[:20] = 0.35
    return CF.cycle_block(CF.with_derived({"tawss": {"values": tawss}, "osi": {"values": osi}}), seg, {0: "主动脉", 1: "左髂总", 2: "右髂总"}, 34000.0)


def test_narrative_adds_one_cycle_sentence_and_keeps_the_single_frame_text():
    plain = N.build_narrative(WALL)
    summary = {**copy.deepcopy(WALL), "cycle": _cycle_block()}
    out = N.build_narrative(summary)
    # C2 (2026-09-30): morphology → cycle quantities → one peak-frame sentence; the peak low-WSS fraction is gone.
    assert out["zh"][0] == plain["zh"][0] and out["en"][0] == plain["en"][0] and len(out["zh"]) == 4
    assert out["zh"][-1] == N.CYCLE_DISCLAIMER_ZH == "以上为标准血流条件下单周期 TAWSS / OSI 与收缩期峰值 WSS 预测的参考描述，非诊断结论。"
    assert out["en"][-1] == N.CYCLE_DISCLAIMER_EN and plain["zh"][-1] == N.DISCLAIMER_ZH
    # 650 aortic points at 0.3 Pa (50 at 3.3 Pa) → 65 % low TAWSS, mean 0.72 Pa; OSI > 0.1 on the aortic 700 → 70 %.
    assert out["zh"][1] == ("周期平均 TAWSS 均值 0.72 Pa，低 TAWSS（< 0.4 Pa）区占壁面 65%，以主动脉为主；OSI > 0.1 占 70%；"
                            "滞留区（TAWSS < 0.4 Pa 且 OSI > 0.1）占 65%（约 221 cm²），主要位于主动脉。")
    assert out["en"][1].startswith("Cycle-averaged TAWSS has a mean of 0.72 Pa, low TAWSS (< 0.4 Pa) covers 65% of the wall, mostly in the aorta;")
    assert out["zh"][2] == "收缩期峰值帧：峰值 WSS 最高的区域（不低于本例 p99，16.6 Pa）2 处，最高 21.9 Pa 位于左髂内；p99 处于 136 例参照人群第 32 百分位。"
    assert out["en"][2].startswith("At the peak-systolic frame there are 2 region(s) of highest peak WSS (at or above this case's p99, 16.6 Pa)")
    assert "低 WSS（" not in "".join(out["zh"]) and "Low WSS" not in "".join(out["en"])
    # The volume family ignores a stray cycle block; its sentences stay identical.
    volume = N.build_narrative(VOLUME)
    assert N.build_narrative({**copy.deepcopy(VOLUME), "cycle": _cycle_block()})["zh"] == volume["zh"]


def test_real_m1_numbers_give_the_contract_sentence():
    """Numbers of the LV_GUO_YOU M1 run (contract §19.2 example)."""
    cycle = {"fields": {"tawss": {"mean": 0.6648, "thresholds": [0.4, 4.0, 7.0], "area_frac": {"low": 0.6479},
                                  "per_branch": {"主动脉": {"area_mm2": 25984.0, "frac_low": 0.8}, "左髂总": {"area_mm2": 1770.0, "frac_low": 0.5}}},
                        "osi": {"mean": 0.1348, "thresholds": [0.1, 0.2, 0.3], "area_frac": {"above_t0": 0.5663}}},
             "stagnation": {"criteria": {"tawss_lt_pa": 0.4, "osi_gt": 0.1}, "area_frac": 0.4557, "area_mm2": 15544.8,
                            "per_branch": {"主动脉": {"area_mm2": 14511.6}, "左髂总": {"area_mm2": 736.4}}}}
    zh = N.build_narrative({"fields": {"wss": {}}, "cycle": cycle})["zh"]
    assert zh[0] == ("周期平均 TAWSS 均值 0.66 Pa，低 TAWSS（< 0.4 Pa）区占壁面 65%，以主动脉为主；OSI > 0.1 占 57%；"
                     "滞留区（TAWSS < 0.4 Pa 且 OSI > 0.1）占 46%（约 155 cm²），主要位于主动脉。")


def test_quality_text_follows_the_model_count():
    assert model_count_phrase(None) == model_count_phrase(5) == "五模型" and model_count_phrase(3) == "3 个模型"
    values, spread = np.array([1.0, 2.0]), np.array([0.8, 1.0])
    five = ensemble_quality(values, spread, seed_count=5)["quality"]
    assert five["reasons"] == ["五模型离散度较高，结果需要人工复核"] == ensemble_quality(values, spread)["quality"]["reasons"]
    three = ensemble_quality(values, spread, seed_count=3)["quality"]
    assert three["reasons"] == ["3 个模型离散度较高，结果需要人工复核"] and three["level"] == five["level"]
    review = ensemble_quality(np.array([2.0, 4.0, 1.0]), np.array([0.5, 0.9, 0.3]), seed_count=3)["quality"]
    assert all("五模型" not in reason for reason in review["reasons"])


def test_export_table_cycle_columns_sit_after_the_wall_columns():
    assert COLUMNS.index(WALL_COLUMNS[-1]) + 1 == COLUMNS.index(CYCLE_COLUMNS[0])
    assert COLUMNS.index(CYCLE_COLUMNS[-1]) + 1 == COLUMNS.index(VOLUME_COLUMNS[0]) and len(CYCLE_COLUMNS) == 8
    block = _cycle_block()
    row = build_row({"id": "j"}, {**wall_summary(), "cycle": block})
    assert row["tawss_mean_pa"] == pytest.approx(block["fields"]["tawss"]["mean"]) and row["tawss_p99_pa"] == pytest.approx(block["fields"]["tawss"]["p99"])
    assert row["tawss_low_frac"] == pytest.approx(0.65) and row["osi_mean"] == pytest.approx(block["fields"]["osi"]["mean"])
    assert row["osi_high_frac"] == pytest.approx(0.70) and row["osi_very_high_frac"] == pytest.approx(0.02)
    assert row["stagnation_frac"] == pytest.approx(0.65) and row["stagnation_area_cm2"] == pytest.approx(34000.0 * 0.65 / 100.0)
    plain = build_row({"id": "j"}, wall_summary())
    assert all(plain[c] is None for c in CYCLE_COLUMNS) and plain["wss_p99_pa"] == 16.64


def _comparable(**extra):
    summary = {"input_sha256": "a" * 64, "case_id": "A", "mapping": {"1": "out-le"},
               "fields": {"wss": {"units": "Pa", "location": "wall", "kind": "scalar", "components": 1}},
               "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21, "label": "peak_systole"}],
               "statistics_protocol": {"field": "fixed_peak_systolic_frame"}, "wss_field_pa": {"thresholds_pa": [0.4, 4.0, 7.0], "mean": 1.0},
               "run_parameters": {"smooth_mm": 1.0, "spacing_mm": 0.5}, "peak": {"p99_pa": 10.0, "max_pa": 20.0}}
    summary.update(extra)
    return summary


def test_comparison_adds_cycle_rows_with_deltas_only_when_both_sides_match():
    block = _cycle_block()
    other = copy.deepcopy(block); other["fields"]["tawss"]["mean"] += 0.1
    both = compare_summaries(_comparable(cycle=block), _comparable(cycle=other))
    rows = {row["id"]: row for row in both["rows"] if row["scope"] == "cycle"}
    assert both["compatible"] and set(rows) == {"cycle.tawss.mean_pa", "cycle.tawss.p99_pa", "cycle.osi.mean", "cycle.rrt.mean_per_pa",
                                                "cycle.ecap.mean_per_pa", "cycle.stagnation.area_frac"}
    assert rows["cycle.tawss.mean_pa"]["delta"] == pytest.approx(0.1) and rows["cycle.osi.mean"]["units"] == "1"
    assert both["cycle"] == {"comparable": True, "note": None} and "note" not in rows["cycle.osi.mean"]
    one = compare_summaries(_comparable(), _comparable(cycle=block))
    cycle_rows = [row for row in one["rows"] if row["scope"] == "cycle"]
    assert all(row["left"] is None and row["right"] is not None and row["delta"] is None for row in cycle_rows)
    assert cycle_rows[0]["note"] == "只有右侧有周期量（TAWSS / OSI），差值留空" and one["cycle"]["comparable"] is False
    changed = copy.deepcopy(other); changed["definition"]["period_s"] = 1.0
    differ = compare_summaries(_comparable(cycle=block), _comparable(cycle=changed))
    assert all(row["delta"] is None for row in differ["rows"] if row["scope"] == "cycle") and "定义或阈值不同" in differ["cycle"]["note"]
    plain = compare_summaries(_comparable(), _comparable())
    assert "cycle" not in plain and not any(row["scope"] == "cycle" for row in plain["rows"])


def m1_summary():
    summary = wall_summary()
    summary.update(cycle=_cycle_block(), run_parameters={"seed_count": 3, "smooth_mm": 1.0, "spacing_mm": 0.5},
                   model_release={"registry_id": "M1_3head_3seed_20260922", "fingerprint": "ab" * 32})
    summary["fields"] = {**summary["fields"], "tawss": {"units": "Pa"}, "osi": {"units": "1"}}
    summary["findings"]["items"] += [
        {"id": "F3", "kind": "stagnation_cluster", "label": "主动脉滞留区", "branch": "主动脉", "value": 121.6, "units": "cm²", "rank": 3, "severity": "attention", "definition": "TAWSS < 0.4 Pa 且 OSI > 0.1 的连通簇"},
        {"id": "F4", "kind": "high_osi_cluster", "label": "主动脉高 OSI 区", "branch": "主动脉", "value": 0.357, "units": "1", "rank": 4, "severity": "info", "definition": "OSI > 0.3 的连通簇"},
        {"id": "F5", "kind": "max_wss", "label": "全场最大 WSS", "branch": "左髂内", "value": 48.6, "units": "Pa", "rank": 5, "severity": "attention", "definition": "单点最大值"},
        {"id": "F6", "kind": "min_radius", "label": "主动脉最小半径", "branch": "主动脉", "value": 7.6, "units": "mm", "rank": 6, "severity": "info", "definition": "狭窄指数"}]
    summary["review"] = {"status": "unreviewed"}
    return summary


def test_m1_onepage_cards_limitations_model_count_and_findings():
    summary = m1_summary()
    html = render_onepage(summary, {"id": "job-m1"})
    page1 = html.split('<article class="page appendix">')[0]
    # U13 (2026-09-30): three significant digits; U14: the model card name instead of the release code;
    # C2: the cycle cards come before the peak-frame cards.
    for text in ("TAWSS 均值", "0.720 Pa", "OSI 均值", "滞留区面积", "221 cm²", "3 个模型离散度在常规范围内", "周期指标 TAWSS · OSI",
                 "峰值帧 WSS 全场 p99", "多模型一致", "一致不代表准确", '<span class="tier">模型</span>', '<span class="tier">派生</span>'):
        assert text in page1, text
    assert "M1 · WSS + TAWSS + OSI" not in page1 and page1.index("TAWSS 均值") < page1.index("峰值帧 WSS 全场 p99")
    assert LIMIT_CYCLE_FRAME in html and LIMIT_SINGLE_FRAME not in html and "没有 TAWSS" not in html and "五模型" not in html
    assert "主动脉滞留区" in page1 and "122 cm²" in page1 and "0.357" in page1
    assert "<h2>模型验证" in html and "R² 0.74，一致性系数 0.93" in html and "34 例留出" in html
    kinds = [item["kind"] for item in top_findings(summary)]
    assert kinds[:3] == ["high_wss_cluster", "low_wss_cluster", "stagnation_cluster"] and "max_wss" not in kinds[:4]
    assert "high_osi_cluster" in kinds and len(kinds) == 5
    # The merged appendix table carries the per-branch cycle columns.
    assert "TAWSS 均值</span></th>" in html and "滞留区</span></th>" in html
    # Glossary "used": the cycle terms appear because the page uses them; volume-only terms do not.
    glossary = html.split("<h2>术语说明")[1]
    assert "TAWSS 周期平均壁面切应力" in glossary and "OSI 振荡剪切指数" in glossary and "滞留区" in glossary
    assert "流线" not in glossary and "相对压力" not in glossary


def test_cycle_arrays_are_read_back_from_field_npz(tmp_path):
    from wss_deploy.rebuild_report import _cycle_arrays
    np.savez(tmp_path / "f.npz", wss_pa=np.ones(3, np.float32), tawss_pa=np.full(3, 0.5, np.float32), osi=np.full(3, 0.1, np.float32))
    z = np.load(tmp_path / "f.npz")
    meta = {"fields": {"wss": {"array_key": "wss_pa"}, "tawss": {"array_key": "tawss_pa"}, "osi": {"array_key": "osi"}}}
    got = _cycle_arrays(meta, z)
    assert set(got) == {"tawss", "osi"} and got["tawss"].dtype == np.float64 and got["osi"][0] == pytest.approx(0.1)
    assert _cycle_arrays({"fields": {"wss": {"array_key": "wss_pa"}}}, z) == {}


def test_m1_reference_sidecar_is_geometry_only_and_bound_to_the_release(tmp_path):
    from wss_deploy.build_reference_profiles import build
    from wss_deploy.infer import load_reference_sidecar
    from tests.test_reference_profiles import _collected
    profile = build("M1_3head_3seed_20260922", _collected(), today="2026-09-22", geometry_only=True,
                    geometry_note="同一 v5.1 train136", population_note="M1 无 CV3 折外预测")
    assert "population_reference" not in profile and profile["population"] == {"status": "unknown", "reason": "M1 无 CV3 折外预测"}
    (tmp_path / "reference.json").write_text(json.dumps(profile), encoding="utf-8")
    info = {"release": "M1_3head_3seed_20260922"}
    assert load_reference_sidecar(tmp_path, "M1_3head_3seed_20260922", info) and set(info) == {"release", "geometry_reference"}
    outputs = Path(__file__).resolve().parents[1] / "outputs"
    shipped = next((p for p in (outputs / "wss_deploy_release/M1_3head_3seed_20260922/reference.json",
                                outputs / "wss_deploy_release_retired/M1_3head_3seed_20260922/reference.json") if p.is_file()), None)
    if shipped is not None:
        info = {"release": "M1_3head_3seed_20260922"}
        assert load_reference_sidecar(shipped.parent, "M1_3head_3seed_20260922", info)
        assert info["geometry_reference"]["status"] == "validated" and "population_reference" not in info
