"""C line (2026-09-30, WORKSPACE_V2_CONTRACT.md §4–§5): model cards, reference v2, anatomical zones,
findings listing rules (merge / area floor / cohort grading), review remap across rule changes, display name."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tests._c_helpers import finished, manager
from tests.test_analysis import NAMES, _atlas, _wall_points
from wss_deploy import analysis as A, model_cards as MC, reference as REF
from wss_deploy.schema import display_name, summary_display_name

RELEASE_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "wss_deploy_release"
KW = dict(thresholds=[0.4, 4, 7], total_area_mm2=1000.0, spacing_mm=1.0, branch_names=NAMES, min_cluster_points=20)


@pytest.fixture(scope="module")
def tube():
    atlas = _atlas()
    pts, feats = _wall_points(atlas)
    return atlas, pts, feats


# ----------------------------------------------------------------------------- model cards (§4)
def test_repository_cards_load_validate_and_carry_provisional_windows():
    assert MC.card_ids() == ["M1_3head_3seed_20260922", "PF6_VF6_peak_3seed_20260920", "X5D_v51_5seed_20260916"]
    for rid in MC.card_ids():
        card = MC.load(rid)
        assert card is not None and card["schema_version"] == MC.SCHEMA_VERSION and card["release_id"] == rid
        assert card["caveat"] == "研究用途，不是诊断" and card["card_source"] == "repository"
        assert set(card["field_tiers"].values()) <= set(MC.TIERS) and card["field_tiers"]["max_diameter"] == "geometry"
        for windows in card["display_windows"].values():
            assert all(w["provisional"] is True and w["range"][0] < w["range"][1] for w in windows)
        assert any("30.38 mL/s" in line and "0.8 s" in line for line in card["protocol"])
        assert any("待核对" in line for line in card["protocol"])          # rigid wall not verified
    m1 = MC.load("M1_3head_3seed_20260922")
    assert [w["label"] for w in MC.display_windows(m1, "tawss")] == ["低剪切窗", "常规窗"]
    assert MC.display_windows(m1, "osi")[1]["range"] == [0.1, 0.5] and MC.display_windows(m1, "missing") == []
    assert MC.field_tier(m1, "rrt") == "derived" and MC.field_tier(m1, "nope", "model") == "model"
    assert MC.field_validation(m1, "tawss")["holdout_n"] == 34 and MC.field_validation(m1, "nope") is None
    rows = MC.validation_rows(m1)
    assert [r["field"] for r in rows] == ["max_diameter", "tawss", "wss", "osi", "stagnation", "rrt", "ecap"]
    pf6 = MC.load("PF6_VF6_peak_3seed_20260920")
    assert pf6["validation"]["holdout_n"] is None and pf6["display_windows"]["pressure"] == []
    assert "尚未建立" in pf6["validation"]["note"]
    assert MC.load("../etc") is None and MC.load("") is None and MC.load("NOPE") is None


def test_card_numbers_match_the_release_packages():
    """Every validation number of the shipped cards is the release.json / metrics value (rounded to 4 decimals)."""
    x5d_dir, m1_dir, pf6_dir = (RELEASE_ROOT / n for n in ("X5D_v51_5seed_20260916", "M1_3head_3seed_20260922", "PF6_VF6_peak_3seed_20260920"))
    if not (x5d_dir / "release.json").is_file() or not (m1_dir / "release.json").is_file():
        pytest.skip("release packages not present")
    close = lambda a, b: a == pytest.approx(round(b, 4), abs=6e-5)
    x5d = json.loads((x5d_dir / "release.json").read_text(encoding="utf-8"))
    x5d_metrics = json.loads((x5d_dir / "metrics" / "x5d_v51_five_seed_test34.json").read_text(encoding="utf-8"))["values"]
    card = MC.load("X5D_v51_5seed_20260916")["validation"]["fields"]["wss"]
    assert close(card["r2_pa"], x5d["metrics"]["test34_pa_r2_cb_ensemble"]) and close(card["cv3_oof_r2_pa"], x5d["metrics"]["cv3_v51_out_of_fold_pooled"])
    assert close(card["r2_case_mean"], x5d_metrics["test34_summary"]["case_r2_mean"])
    assert close(card["r2_case_p10"], x5d_metrics["test34_summary"]["case_r2_p10"])
    assert close(card["top10_pred_over_cfd"], x5d_metrics["test34_summary"]["calibration_pooled"]["top10_pred_true_ratio"])
    assert card["cv3_n"] == x5d_metrics["cv3_summary"]["n"] and MC.load("X5D_v51_5seed_20260916")["version_date"] == x5d["frozen_on"]
    m1 = json.loads((m1_dir / "release.json").read_text(encoding="utf-8"))
    ens, fields = m1["metrics"]["test34_ensemble"], MC.load("M1_3head_3seed_20260922")["validation"]["fields"]
    for key in ("wss", "tawss", "osi"):
        assert close(fields[key]["r2_pa"], ens[key]["pa_r2cb"]) and close(fields[key]["r2_case_mean"], ens[key]["r2_casemean"])
        assert close(fields[key]["r2_case_p10"], ens[key]["r2_casep10"])
    assert close(fields["tawss"]["ccc"], ens["tawss"]["ccc"]) and close(fields["osi"]["ccc"], ens["osi"]["ccc"])
    assert close(fields["stagnation"]["iou_case_mean"], m1["metrics"]["test34_stagnation"]["iou_casemean"])
    assert close(fields["stagnation"]["dice_median"], m1["metrics"]["test34_stagnation"]["dice_med"])
    assert fields["rrt"]["r2_pa"] is None and fields["ecap"]["r2_pa"] is None
    assert MC.load("M1_3head_3seed_20260922")["version_date"] == m1["frozen_on"]
    if (pf6_dir / "release.json").is_file():
        pf6 = json.loads((pf6_dir / "release.json").read_text(encoding="utf-8"))
        assert "metrics" not in pf6 and "not been established" in pf6["validation"]
        assert MC.load("PF6_VF6_peak_3seed_20260920")["version_date"] == pf6["frozen_on"]


def test_release_dir_card_wins_and_bad_cards_fail_closed(tmp_path):
    card = MC.load("X5D_v51_5seed_20260916")
    card.pop("card_source")
    own = {**copy.deepcopy(card), "display_name": "自带说明卡"}
    (tmp_path / "model_card.json").write_text(json.dumps(own, ensure_ascii=False), encoding="utf-8")
    assert MC.load("X5D_v51_5seed_20260916", tmp_path)["display_name"] == "自带说明卡"
    assert MC.load("X5D_v51_5seed_20260916", tmp_path)["card_source"] == "release_dir"
    # A card bound to another release, an unknown tier or a reversed window is ignored (repository copy used).
    for bad in ({**own, "release_id": "OTHER"}, {**own, "field_tiers": {"wss": "guess"}},
                {**own, "display_windows": {"wss": [{"id": "x", "label": "x", "range": [5, 1]}]}}, {"schema_version": "v0"}):
        (tmp_path / "model_card.json").write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        assert MC.load("X5D_v51_5seed_20260916", tmp_path)["display_name"] == "峰值 WSS"
    assert set(MC.all_cards(["X5D_v51_5seed_20260916", "NOPE"])) == {"X5D_v51_5seed_20260916", "NOPE"}
    assert MC.all_cards(["NOPE"])["NOPE"] is None and MC.all_cards(None) == {}
    assert MC.all_cards({"X5D_v51_5seed_20260916": None})["X5D_v51_5seed_20260916"]["release_id"] == "X5D_v51_5seed_20260916"


def test_all_cards_reads_a_registry(tmp_path):
    class Registry:
        def list(self):
            return [{"id": "M1_3head_3seed_20260922"}, {"id": "ZZZ"}]

        def describe(self, rid):
            raise KeyError(rid)
    cards = MC.all_cards(Registry())
    assert cards["M1_3head_3seed_20260922"]["short_name"] == "周期指标" and cards["ZZZ"] is None


# ----------------------------------------------------------------------------- reference v2 (§4 end)
def _v1_profile(values, release="R1"):
    protocol = {"field": "fixed_peak_systolic_frame", "support": "prediction_point_cloud", "quantile_method": "linear"}
    axis = [{"index": 0, "step": 1162, "time_s": 0.21, "label": "peak_systole"}]
    field = {"units": "Pa", "location": "wall", "kind": "scalar", "components": 1}
    info = {"release": release, "population_reference": {
        "schema_version": "wss-deploy.population-reference/v1", "id": "pop", "status": "validated", "release": release,
        "reference_set": "oof", "source": "cv3_oof_predictions", "fold_count": 3, "metric": "p99_pa", "field": field,
        "time_axis": axis, "statistics_protocol": protocol, "values_pa": list(values)}}
    meta = {"release": release, "fields": {"wss": field}, "time_axis": axis, "statistics_protocol": protocol, "peak": {}}
    return info, meta


def test_reference_reads_v1_and_v2_and_gives_the_p90():
    values = list(np.linspace(10.0, 40.0, 31))
    info, meta = _v1_profile(values)
    meta["peak"]["p99_pa"] = 30.0
    population = REF.evaluate_population(meta, info)
    assert population["status"] == "pass" and population["reference_p90_pa"] == pytest.approx(np.quantile(values, 0.9))
    assert REF.population_p90(population) == pytest.approx(37.0) and REF.population_p90({"status": "unknown"}) is None
    assert REF.quantile([1, 2, 3, 4], 0.5) == pytest.approx(2.5) and REF.quantile([], 0.5) is None
    info["population_references"] = [
        {"id": "tawss-mean-AAA", "metric": "tawss_mean_pa", "field": "tawss", "statistic": "mean", "units": "Pa", "n": 3,
         "subgroup": "AAA", "source": "cv5_oof_predictions", "model_release": "R1", "data_version": "v5.2", "built_on": "2026-10-01",
         "values": [0.5, 0.7, 0.9]},
        {"id": "osi-q", "metric": "osi_mean", "field": "osi", "statistic": "mean", "units": "1", "subgroup": "all",
         "source": "cv5", "model_release": "R1", "quantiles": {"p10": 0.05, "p50": 0.1, "p90": 0.2}},
        {"id": "bad-subgroup", "metric": "x", "field": "x", "statistic": "x", "units": "1", "subgroup": "XYZ", "source": "s",
         "model_release": "R1", "values": [1, 2]},
        {"id": "bad-n", "metric": "x", "field": "x", "statistic": "x", "units": "1", "subgroup": "all", "source": "s",
         "model_release": "R1", "n": 5, "values": [1, 2]},
        {"id": "other-release", "metric": "y", "field": "y", "statistic": "y", "units": "1", "subgroup": "all", "source": "s",
         "model_release": "R2", "values": [1, 2]},
    ]
    entries = REF.population_references(info)
    assert [e["id"] for e in entries] == ["pop", "tawss-mean-AAA", "osi-q", "other-release"]
    assert entries[0]["schema"] == "v1" and entries[0]["n"] == 31 and entries[1]["schema"] == "v2"
    found = REF.find_population_reference(info, metric="tawss_mean_pa", field="tawss", subgroup="AAA")
    assert found["id"] == "tawss-mean-AAA" and REF.reference_quantile(found, 0.5) == pytest.approx(0.7)
    osi = REF.find_population_reference(info, metric="osi_mean")
    assert REF.reference_quantile(osi, 0.9) == pytest.approx(0.2) and REF.reference_quantile(osi, 0.75) is None
    assert REF.find_population_reference(info, metric="p99_pa")["schema"] == "v1"
    assessed = REF.evaluate(meta, info)
    assert [e["id"] for e in assessed["population_references"]] == ["tawss-mean-AAA", "osi-q"]   # bound to R1, no values
    assert "values" not in assessed["population_references"][0] and assessed["population"]["percentile"] is not None
    del info["population_references"]
    assert "population_references" not in REF.evaluate(meta, info)


def test_merge_sidecar_v2_reads_reference_json(tmp_path):
    (tmp_path / "reference.json").write_text(json.dumps({"population_references": [
        {"id": "a", "model_release": "R1"}, {"id": "b", "model_release": "R2"}]}), encoding="utf-8")
    info = {"release": "R1"}
    merged = REF.merge_sidecar_v2(info, tmp_path)
    assert [e["id"] for e in merged["population_references"]] == ["a"] and "population_references" not in info
    assert REF.merge_sidecar_v2(info, None) == info and REF.merge_sidecar_v2(info, tmp_path / "missing") == info


# ----------------------------------------------------------------------------- zones (U10, §5.1)
SAC_MORPHOLOGY = {"aorta": {"segment_id": 0, "name": "主动脉",
                            "neck": {"present": True, "s_start_mm": 5.0, "s_end_mm": 12.0},
                            "sac": {"present": True, "s_start_mm": 15.0, "s_end_mm": 25.0}}}
FULL_NAMES = {"0": "主动脉", "1": "左髂总", "2": "右髂总"}


def test_zones_split_the_aorta_by_neck_and_sac_and_pair_the_iliacs(tube):
    atlas, pts, feats = tube
    n = len(pts)
    tawss = np.where(feats["segment_id"] == 0, 0.3, 1.0); tawss[feats["segment_id"] == 2] = 2.0
    osi = np.full(n, 0.05); osi[(feats["segment_id"] == 0) & (feats["s_local_mm"] >= 15) & (feats["s_local_mm"] <= 25)] = 0.2
    wss = np.full(n, 2.0)
    out = A.zones_wall(feats["segment_id"], feats["s_from_root_mm"], {"wss": wss, "tawss": tawss, "osi": osi},
                       morphology=SAC_MORPHOLOGY, branch_names=FULL_NAMES, total_area_mm2=1000.0)
    assert out["schema_version"] == "wss-deploy.zones/v1" and out["primary_fields"] == ["tawss", "osi", "wss"]
    ids = [z["id"] for z in out["zones"]]
    assert ids == ["aorta_proximal", "neck", "sac", "aorta_distal", "left_cia", "right_cia"]
    by = {z["id"]: z for z in out["zones"]}
    assert by["neck"]["s_range_mm"] == [5.0, 15.0] and by["neck"]["gap_merged_mm"] == pytest.approx(3.0)   # shoulder → neck
    assert by["sac"]["s_range_mm"] == [15.0, 25.0] and by["sac"]["n_points"] == 21 * 48
    assert by["aorta_proximal"]["n_points"] == 10 * 48 and by["aorta_distal"]["branch_key"] == "root"
    assert sum(z["n_points"] for z in out["zones"] if z["segment_id"] == 0) == int((feats["segment_id"] == 0).sum())
    assert by["sac"]["area_mm2"] == pytest.approx(1000.0 * 21 * 48 / n)
    assert by["sac"]["fields"]["osi"]["frac_high"] == pytest.approx(1.0) and by["neck"]["fields"]["osi"]["frac_high"] == 0.0
    assert by["sac"]["fields"]["tawss"]["frac_low"] == pytest.approx(1.0) and set(by["sac"]["fields"]["tawss"]) == {
        "mean", "median", "p10", "p90", "frac_low", "frac_high"}
    assert set(by["sac"]["fields"]["wss"]) == {"mean", "p99", "frac_low", "frac_high"}
    assert by["left_cia"]["branch_key"] == "left_cia" and by["left_cia"]["label"] == "左髂总"
    cia = out["pairs"][0]
    assert cia["id"] == "cia" and cia["fields"]["tawss"]["ratio_left_over_right"] == pytest.approx(0.5)
    assert [p["id"] for p in out["pairs"]] == ["cia"]                    # no external / internal branches in the tube
    json.dumps(out, allow_nan=False)


def test_zones_without_sac_missing_fields_and_zero_right_mean(tube):
    atlas, pts, feats = tube
    n = len(pts)
    wss = np.full(n, 2.0); wss[feats["segment_id"] == 2] = 0.0
    plain = A.zones_wall(feats["segment_id"], feats["s_from_root_mm"], {"wss": wss}, morphology={"aorta": {"sac": {"present": False}}},
                         branch_names=FULL_NAMES, total_area_mm2=None)
    assert [z["id"] for z in plain["zones"]] == ["aorta", "left_cia", "right_cia"] and plain["primary_fields"] == ["wss"]
    assert plain["zones"][0]["area_mm2"] is None and plain["pairs"][0]["fields"]["wss"]["ratio_left_over_right"] is None
    assert A.zones_wall(feats["segment_id"], feats["s_from_root_mm"], {"wss": wss}, morphology=None,
                        branch_names=FULL_NAMES, total_area_mm2=1.0)["zones"][0]["id"] == "aorta"
    # Neck absent but sac present → the proximal zone runs to the sac and is labelled accordingly.
    no_neck = {"aorta": {"sac": {"present": True, "s_start_mm": 15.0, "s_end_mm": 25.0}, "neck": {"present": False}}}
    out = A.zones_wall(feats["segment_id"], feats["s_from_root_mm"], {"wss": wss}, morphology=no_neck, branch_names=FULL_NAMES,
                       total_area_mm2=1.0)
    assert [z["id"] for z in out["zones"]][:3] == ["aorta_proximal", "sac", "aorta_distal"] and out["zones"][0]["label"] == "瘤体以上主动脉"
    # Missing names → no zones (with a note); no fields → None; unknown branch names are skipped.
    empty = A.zones_wall(feats["segment_id"], feats["s_from_root_mm"], {"wss": wss}, morphology=None, branch_names=None, total_area_mm2=1.0)
    assert empty["zones"] == [] and empty["notes"]
    assert A.zones_wall(feats["segment_id"], feats["s_from_root_mm"], {}, morphology=None, branch_names=FULL_NAMES, total_area_mm2=1.0) is None
    odd = A.zones_wall(feats["segment_id"], feats["s_from_root_mm"], {"wss": wss}, morphology=None,
                       branch_names={"0": "主动脉", "1": "分支 1"}, total_area_mm2=1.0)
    assert [z["id"] for z in odd["zones"]] == ["aorta"]
    with pytest.raises(ValueError):
        A.zones_wall(feats["segment_id"], feats["s_from_root_mm"], {"wss": wss[:-1]}, morphology=None, branch_names=FULL_NAMES, total_area_mm2=1.0)


# ----------------------------------------------------------------------------- findings rules (U11 / U12, §5.2)
def test_high_findings_are_graded_against_the_cohort_p90(tube):
    atlas, pts, feats = tube
    wss = np.full(len(pts), 2.0)
    wss[(feats["segment_id"] == 1) & (feats["s_local_mm"] >= 5) & (feats["s_local_mm"] <= 7)] = 12.0
    p99 = float(np.quantile(wss, .99))
    above = {"population": {"status": "pass", "reference_p90_pa": p99 - 1.0, "value_pa": p99, "reference_count": 136}}
    below = {"population": {"status": "pass", "reference_p90_pa": p99 + 1.0, "value_pa": p99, "reference_count": 136}}
    graded = A.findings_wall(pts, wss, feats, atlas, {}, reference_assessment=above, **KW)
    high = [i for i in graded["items"] if i["kind"] == "high_wss_cluster"]
    assert high and all(i["severity"] == "attention" and i["grading"] == "reference_p90" for i in high)
    assert graded["high_grading"]["reference_p90_pa"] == pytest.approx(p99 - 1.0) and graded["high_grading"]["reference_count"] == 136
    A.grade_high_findings(graded, below)                   # re-grading a stored list is idempotent and in place
    assert all(i["severity"] == "note" for i in graded["items"] if i["kind"] == "high_wss_cluster")
    A.grade_high_findings(graded, {"population": {"status": "unknown"}})
    assert graded["high_grading"]["status"] == "no_reference" and graded["high_grading"]["reference_p90_pa"] is None
    # Low-value severity is untouched by the grading.
    assert A.grade_high_findings({"items": [{"kind": "low_wss_cluster", "severity": "attention"}]}, above)["items"][0]["severity"] == "attention"


def test_global_max_outside_every_listed_cluster_stays_listed(tube):
    atlas, pts, feats = tube
    n = len(pts)
    wss = np.full(n, 2.0)
    wss[(feats["segment_id"] == 1) & (feats["s_local_mm"] >= 5) & (feats["s_local_mm"] <= 7)] = 12.0   # 240-point cluster
    lone = int(np.flatnonzero(feats["segment_id"] == 2)[0])
    wss[lone] = 50.0                                                                                # single hot point
    out = A.findings_wall(pts, wss, feats, atlas, {}, **KW)
    kinds = [i["kind"] for i in out["items"]]
    assert kinds.count("max_wss") == 1 and out["listing_rules"]["global_max_merged"] is False
    item = next(i for i in out["items"] if i["kind"] == "max_wss")
    assert item["value"] == pytest.approx(50.0) and item["severity"] == "note" and item["grading"] == "no_reference"
    assert not any(i.get("contains_global_max") for i in out["items"])


def test_low_wss_clusters_below_one_square_centimetre_are_not_listed(tube):
    atlas, pts, feats = tube
    n = len(pts)
    wss = np.full(n, 2.0)
    sac = (feats["segment_id"] == 0) & (feats["s_local_mm"] >= 15) & (feats["s_local_mm"] <= 25)   # 1008 points
    small = (feats["segment_id"] == 2) & (feats["s_local_mm"] >= 14) & (feats["s_local_mm"] <= 16)  # 240 points
    wss[sac | small] = 0.1
    out = A.findings_wall(pts, wss, feats, atlas, {}, **KW)                  # 1000 mm² total: small ≈ 31 mm², sac ≈ 129 mm²
    low = [i for i in out["items"] if i["kind"] == "low_wss_cluster"]
    assert len(low) == 1 and low[0]["n_points"] == int(sac.sum())
    assert out["listing_rules"]["suppressed"]["low_wss_cluster"]["count"] == 1
    # The severity rule of the low side is unchanged: it counts every ≥ 20-point cluster, listed or not.
    assert out["low_wss_total_fraction"] == pytest.approx((sac.sum() + small.sum()) / n)
    assert low[0]["severity"] == ("attention" if out["low_wss_total_fraction"] > 0.2 else "note")
    legacy = A.findings_wall(pts, wss, feats, atlas, {}, area_floor_mm2=0.0, **KW)
    assert len([i for i in legacy["items"] if i["kind"] == "low_wss_cluster"]) == 2


# ----------------------------------------------------------------------------- review remap (§5.2)
OLD_ITEMS = [
    {"id": "F1", "kind": "high_wss_cluster", "segment_id": 4, "xyz_mm": [1.0, 2.0, 3.0], "label": "左髂内高 WSS 区", "value": 52.6, "units": "Pa"},
    {"id": "F2", "kind": "max_wss", "segment_id": 4, "xyz_mm": [1.0, 2.0, 3.0], "label": "全场最大 WSS（左髂内）", "value": 52.6, "units": "Pa"},
    {"id": "F3", "kind": "low_wss_cluster", "segment_id": 2, "xyz_mm": [9.0, 9.0, 9.0], "label": "左髂总低 WSS 区", "value": 0.34, "units": "Pa"},
    {"id": "F4", "kind": "max_diameter", "segment_id": 0, "xyz_mm": [0.0, 0.0, 100.0], "label": "主动脉最大直径", "value": 72.2, "units": "mm"},
]
NEW_ITEMS = [
    {"id": "F1", "kind": "high_wss_cluster", "segment_id": 4, "xyz_mm": [1.0, 2.0, 3.0], "contains_global_max": True},
    {"id": "F2", "kind": "max_diameter", "segment_id": 0, "xyz_mm": [0.0, 0.0, 100.0]},
]


def test_remap_moves_matched_decisions_and_keeps_the_rest_as_legacy():
    review = {"schema_version": "wss-deploy.findings_review/v1", "added": [{"id": "M1", "text": "人工"}],
              "items": {"F1": {"decision": "confirmed", "note": ""}, "F2": {"decision": "rejected", "note": "噪声"},
                        "F3": {"decision": "confirmed", "note": "小簇"}, "F4": {"decision": "confirmed", "note": ""}}}
    out, report = A.remap_review(review, OLD_ITEMS, NEW_ITEMS, from_version="2026-09-26", to_version="2026-09-30")
    assert out["items"] == {"F1": {"decision": "confirmed", "note": ""}, "F2": {"decision": "confirmed", "note": ""}}
    assert report["moved"] == {"F4": "F2"} and sorted(report["legacy"]) == ["F2", "F3"]
    legacy = {entry["id"]: entry for entry in out["legacy"]}
    assert legacy["F2"]["kind"] == "max_wss" and legacy["F2"]["decision"] == "rejected" and "已并入" in legacy["F2"]["reason"]
    assert legacy["F3"]["status_label"] == A.LEGACY_REVIEW_LABEL == "规则更新前的判定" and legacy["F3"]["note"] == "小簇"
    assert out["added"] == review["added"] and out["remap"]["map"]["F4"] == "F2" and out["remap"]["from_analysis_version"] == "2026-09-26"
    # The rejected decision of the old F2 (max) never lands on the new F2 (the diameter).
    assert out["items"]["F2"]["decision"] == "confirmed"
    # Idempotent: a second pass over the same list moves nothing and returns the same object.
    again, report2 = A.remap_review(out, NEW_ITEMS, NEW_ITEMS)
    assert again is out and not report2["moved"] and not report2["legacy"]
    # Without a matching kind nearby the max merges into the cluster when the cluster itself was undecided.
    lone, rep = A.remap_review({"items": {"F2": {"decision": "confirmed", "note": ""}}}, OLD_ITEMS, NEW_ITEMS)
    assert lone["items"] == {"F1": {"decision": "confirmed", "note": ""}} and rep["moved"] == {"F2": "F1"}
    assert A.remap_review(None, OLD_ITEMS, NEW_ITEMS)[0] is None


def test_rebuild_remaps_the_sidecar_and_the_record(tmp_path):
    from wss_deploy.rebuild_report import remap_findings_review
    (tmp_path / "findings_review.json").write_text(json.dumps({"items": {"F4": {"decision": "confirmed", "note": ""},
                                                                         "F3": {"decision": "rejected", "note": ""}}}), encoding="utf-8")
    record = {"findings_review": {"items": {}}}
    meta = {"findings": {"items": NEW_ITEMS}}
    report = remap_findings_review(tmp_path, meta, {"items": OLD_ITEMS, "analysis_version": "2026-09-26"}, record)
    stored = json.loads((tmp_path / "findings_review.json").read_text(encoding="utf-8"))
    assert report["moved"] == {"F4": "F2"} and stored["items"] == {"F2": {"decision": "confirmed", "note": ""}}
    assert [e["id"] for e in stored["legacy"]] == ["F3"] and record["findings_review"] == stored
    assert remap_findings_review(tmp_path / "none", meta, {"items": OLD_ITEMS}, record) is None


def test_findings_review_save_keeps_legacy_decisions(tmp_path):
    mgr = manager(tmp_path)
    job = finished(mgr, case_id="A")
    with mgr.lock:
        mgr.jobs[job["id"]]["findings_review"] = {"items": {}, "legacy": [{"id": "F9", "decision": "rejected"}],
                                                  "remap": {"map": {"F9": None}}}
    doc = mgr.findings_review(job["id"], "owner", {"items": {"F1": {"decision": "confirmed", "note": ""}}})["findings_review"]
    assert doc["items"]["F1"]["decision"] == "confirmed" and doc["legacy"] == [{"id": "F9", "decision": "rejected"}]
    assert doc["remap"] == {"map": {"F9": None}}


# ----------------------------------------------------------------------------- display name (C7)
def test_display_name_prefers_the_patient_id(tmp_path):
    assert display_name("LV_GUO_YOU", "") == "LV_GUO_YOU" and display_name("LV_GUO_YOU", " P-07 ") == "P-07"
    assert display_name(None, None) == "" and summary_display_name({"case_id": "C", "case_metadata": {"patient_id": "P"}}) == "P"
    mgr = manager(tmp_path)
    job = finished(mgr, case_id="ZHANG_SAN")
    assert job["display_name"] == "ZHANG_SAN"
    rows = mgr.list("owner")
    assert rows and all(row["display_name"] == "ZHANG_SAN" for row in rows)
    updated = mgr.update_metadata(job["id"], "owner", {"patient_id": "P-0001"})
    assert updated["display_name"] == "P-0001"
    summary = json.loads((tmp_path / job["id"] / "summary.json").read_text(encoding="utf-8"))
    assert summary["display_name"] == "P-0001" and summary["case_id"] == "ZHANG_SAN"
    from wss_deploy.cases import group_cases
    cards = group_cases([{**mgr._snapshot(mgr.jobs[job["id"]], detail=False)}])
    assert cards[0]["display_name"] == "P-0001" and cards[0]["runs"][0]["display_name"] == "P-0001"
