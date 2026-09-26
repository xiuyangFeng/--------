"""v0.13 derived cycle indices: RRT = 1 / [(1 − 2·OSI)·TAWSS], ECAP = OSI / TAWSS (both 1/Pa).

Computed point by point from the predicted TAWSS / OSI; they ride the generic extra-field channel into the
summary, CSV / VTP and the report, never into field.npz.  Older three-head jobs gain them through
``rebuild_report`` (full rebuild) or, UI-only, through the viewer (see tests/test_report.py).
"""
import json

import numpy as np
import pytest

from wss_deploy import cycle_fields as CF
from wss_deploy import onepager as OP
from wss_deploy import rebuild_report as RB


def test_formula_floors_and_nan_passthrough():
    got = CF.derive_indices(np.array([0.4, 0.0, 2.0, np.nan]), np.array([0.1, 0.5, 0.0, 0.1]))
    # 0.4 Pa / OSI 0.1 = the stagnation corner: RRT 1/(0.8·0.4) = 3.125, ECAP 0.25
    assert got["rrt"][0] == pytest.approx(3.125) and got["ecap"][0] == pytest.approx(0.25)
    # TAWSS floored at 0.01 Pa and 1 − 2·OSI at 0.01: finite, never inf
    assert got["rrt"][1] == pytest.approx(1 / (0.01 * 0.01)) and got["ecap"][1] == pytest.approx(0.5 / 0.01)
    assert got["rrt"][2] == pytest.approx(0.5) and got["ecap"][2] == 0.0
    assert np.isnan(got["rrt"][3]) and np.isnan(got["ecap"][3])
    with pytest.raises(ValueError):
        CF.derive_indices(np.ones(3), np.ones(2))


def test_with_derived_needs_both_fields_and_recomputes():
    t, o = np.array([0.2, 1.0]), np.array([0.25, 0.05])
    assert set(CF.with_derived({"tawss": {"values": t}})) == {"tawss"}
    out = CF.with_derived({"tawss": {"values": t, "seed_pred": np.ones((2, 2))}, "osi": {"values": o},
                           "rrt": {"values": np.zeros(2)}})
    assert list(out) == ["tawss", "osi", "rrt", "ecap"]
    assert out["rrt"]["derived"] is True and out["rrt"]["derived_from"] == ["tawss", "osi"]
    np.testing.assert_allclose(out["rrt"]["values"], [10.0, 1 / (0.9 * 1.0)])   # the stale entry was recomputed
    np.testing.assert_allclose(out["ecap"]["values"], [1.25, 0.05])
    assert "seed_pred" in out["tawss"] and "seed_pred" not in out["rrt"]


def test_descriptors_summary_and_direction():
    v = np.linspace(0.5, 30.0, 200)
    d = CF.descriptor("rrt", v)
    assert (d["units"], d["array_key"], d["source"], d["derived_from"]) == ("1/Pa", "rrt_per_pa", "derived", ["tawss", "osi"])
    assert d["display"]["threshold_direction"] == "above" and d["display"]["thresholds"] == [5.0, 10.0, 20.0] and d["display"]["log_scale"] is True
    assert "1 - 2*OSI" in d["definition"]
    assert CF.descriptor("ecap", v)["display"]["thresholds"] == [1.4, 2.8, 4.2]
    assert CF.descriptor("tawss", v)["display"]["threshold_direction"] == "below" and CF.descriptor("tawss", v)["source"] == "prediction"
    seg = np.repeat([1, 2], 100)
    summary = CF.scalar_field_summary("ecap", v / 10, seg, {1: "主动脉", 2: "左髂总"}, 1000.0)
    assert set(summary["area_frac"]) == {"above_t0", "above_t1", "above_t2"}
    assert summary["area_frac"]["above_t0"] == pytest.approx(np.mean(v / 10 > 1.4))
    assert summary["units"] == "1/Pa" and "OSI / TAWSS" in summary["definition"]
    assert set(summary["per_branch"]["左髂总"]) >= {"frac_above_t0", "mean", "p99"}
    block = CF.cycle_block(CF.with_derived({"tawss": {"values": v / 20}, "osi": {"values": np.full(200, .2)}}), seg, {1: "a", 2: "b"}, 10.0)
    assert set(block["fields"]) == {"tawss", "osi", "rrt", "ecap"} and "stagnation" in block


def test_display_arrays_use_the_interpolated_vertex_fields():
    fields = {"tawss": {"array_key": "tawss_pa"}, "osi": {"array_key": "osi"}}
    extra = CF.with_derived({"tawss": {"values": np.array([0.5, 1.0])}, "osi": {"values": np.array([0.1, 0.2])}})
    points, vertices = CF.derived_display_arrays(fields, extra, {"tawss_pa": np.array([0.5, np.nan, 2.0]), "osi": np.array([0.1, 0.1, 0.25])})
    assert set(points) == {"rrt_per_pa", "ecap_per_pa"} and points["rrt_per_pa"].dtype == np.float32
    np.testing.assert_allclose(vertices["ecap_per_pa"][[0, 2]], [0.2, 0.125], rtol=1e-6)
    assert np.isnan(vertices["rrt_per_pa"][1])                      # an uncovered vertex stays uncovered
    assert CF.derived_display_arrays(fields, {"tawss": {"values": np.ones(2)}}, {}) == ({}, {})
    with pytest.raises(ValueError):
        CF.derived_display_arrays(fields, extra, {"tawss_pa": np.ones(3)})


def test_rebuild_adds_derived_blocks_only_for_three_head_jobs():
    plain = {"fields": {"wss": {}}}
    assert RB._add_derived_indices(plain, {}, np.ones(4), {}, 10.0) == {} and plain == {"fields": {"wss": {}}}
    tawss, osi = np.array([0.2, 0.5, 1.5, 3.0]), np.array([0.25, 0.05, 0.3, 0.12])
    meta = {"fields": {"wss": {}, "tawss": CF.descriptor("tawss", tawss), "osi": CF.descriptor("osi", osi)},
            "cycle": {"definition": {"period_s": 0.8}, "fields": {}}, "results": {"fields": {}, "statistics": {}}}
    got = RB._add_derived_indices(meta, {"tawss": tawss, "osi": osi}, np.ones(4, int), {"1": "主动脉"}, 10.0)
    assert set(got) == {"rrt", "ecap"} and got["rrt"][0] == pytest.approx(10.0)
    assert meta["fields"]["rrt"]["source"] == "derived" and meta["cycle"]["fields"]["ecap"]["mean"] == pytest.approx(np.mean(osi / tawss))
    assert meta["cycle"]["definition"]["period_s"] == 0.8 and "rrt" in meta["cycle"]["definition"]
    assert meta["results"]["fields"]["rrt"] == meta["fields"]["rrt"] and "ecap" in meta["results"]["statistics"]


def test_rebuild_appends_csv_columns_and_keeps_the_existing_text(tmp_path):
    path = tmp_path / "points_wss.csv"
    path.write_text("x_mm,y_mm,wss_pa,branch,tawss_pa,osi\n1.000,2.000,3.1234,主动脉,0.2000,0.2500\n4.000,5.000,6.0000,左髂总,1.0000,0.0500\n", encoding="utf-8")
    cols = {"rrt_per_pa": np.array([10.0, 1.1111111]), "ecap_per_pa": np.array([1.25, 0.05])}
    RB._append_csv_columns(path, cols)
    first = path.read_text(encoding="utf-8")
    assert first.splitlines() == ["x_mm,y_mm,wss_pa,branch,tawss_pa,osi,rrt_per_pa,ecap_per_pa",
                                  "1.000,2.000,3.1234,主动脉,0.2000,0.2500,10.0000,1.2500",
                                  "4.000,5.000,6.0000,左髂总,1.0000,0.0500,1.1111,0.0500"]
    RB._append_csv_columns(path, cols)                              # a second rebuild replaces, never duplicates
    assert path.read_text(encoding="utf-8") == first
    with pytest.raises(ValueError, match="行数"):
        RB._append_csv_columns(path, {"rrt_per_pa": np.ones(3)})


def test_one_pager_shows_one_rrt_ecap_card():
    seg = np.repeat([1, 2], 50)
    tawss, osi = np.linspace(0.1, 3.0, 100), np.linspace(0.3, 0.01, 100)
    block = CF.cycle_block(CF.with_derived({"tawss": {"values": tawss}, "osi": {"values": osi}}), seg, {1: "主动脉", 2: "左髂总"}, 1000.0)
    cards = OP._cycle_cards({"cycle": json.loads(json.dumps(block))})
    titles = [c[0] for c in cards]
    assert len(cards) == 4 and 'data-gloss="rrt"' in titles[2] and 'data-gloss="ecap"' in titles[2]
    assert cards[2][1].endswith(" Pa⁻¹") and "ECAP &gt; 1.4" in cards[2][2] and "RRT &gt; 5.0" in cards[2][2]
    legacy = CF.cycle_block({"tawss": {"values": tawss}, "osi": {"values": osi}}, seg, {1: "a", 2: "b"}, 1000.0)
    assert len(OP._cycle_cards({"cycle": legacy})) == 3            # older summaries: unchanged card set
