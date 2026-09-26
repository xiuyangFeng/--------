"""§19.3: patient follow-up timeline — one scan per input geometry, growth only between dated scans ≥ 30 days apart,
model quantities connected only within one release, owner isolation, empty result."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

import pytest

from wss_deploy.timeline import build_timeline, geometry_values, model_values
from tests._c_helpers import Service, call, finished, manager

X5D, M1, VOL = "X5D_v51_5seed_20260916", "M1_3head_3seed_20260922", "PF6_VF6_peak_3seed_20260920"
# Checked-in copies of two real summaries (the M1 three-head LV_GUO_YOU run and the PF6/VF6 volume run);
# WSS_DEPLOY_TEST_FIXTURES may point at another fixtures root holding the same job ids.
FIXTURES = Path(os.environ.get("WSS_DEPLOY_TEST_FIXTURES") or Path(__file__).resolve().parent / "fixtures")


def _morph(diameter, sac_volume=None):
    sac = {"present": sac_volume is not None, "length_mm": 70.0, "volume_ml": sac_volume}
    return {"aorta": {"max": {"max_diameter_mm": diameter}, "sac": sac, "neck": {"present": True, "diameter_mean_mm": 19.0, "length_mm": 22.0}},
            "lumen_volume_ml": 150.0, "branches": [{"segment_id": 0, "name": "主动脉", "delta_p_pa": 80.0}]}


def _job(i, sha, release, *, patient="P-1", scan_date="", created="2026-09-2%dT10:00:00+08:00", status="done"):
    protocol = {"X5D": "single_frame_wss", "M1_": "single_frame_wss_cycle_multi", "PF6": "single_frame_volume"}[release[:3]]
    return {"id": f"j{i}", "status": status, "patient_id": patient, "scan_date": scan_date, "scan_label": f"扫描{i}", "case_id": f"C{i}",
            "created_at": created % i if "%d" in created else created, "input_sha256": sha,
            "model_release": {"id": release, "contract": {"protocol": protocol}}, "review": {"status": "unreviewed"}}


def _wall(p99, diameter, sac=None, cycle=None):
    out = {"fields": {"wss": {}}, "peak": {"p99_pa": p99}, "wss_field_pa": {"area_frac_low": 0.3, "area_frac_high": 0.02}, "morphology": _morph(diameter, sac)}
    if cycle:
        out["fields"].update(tawss={}, osi={})
        out["cycle"] = {"fields": {"tawss": {"mean": 0.66, "area_frac": {"low": 0.65}}, "osi": {"mean": 0.13, "area_frac": {"above_t0": 0.57}}},
                        "stagnation": {"area_frac": 0.46}}
    return out


def test_scans_merge_by_input_geometry_and_growth_needs_dates():
    jobs = [_job(1, "a" * 64, X5D, scan_date="2025-03-01"), _job(2, "a" * 64, M1, scan_date="2025-03-01"),
            _job(3, "b" * 64, X5D, scan_date="2026-03-01"), _job(4, "b" * 64, VOL, scan_date="2026-03-01"),
            _job(5, "c" * 64, X5D, status="failed"), _job(6, "d" * 64, X5D, patient="P-2")]
    summaries = {"j1": _wall(16.6, 52.1, 90.0), "j2": _wall(17.0, 52.1, 90.0, cycle=True), "j3": _wall(18.0, 55.3, 96.3),
                 "j4": {"fields": {"pressure": {}, "velocity": {}}, "volume_statistics": {"speed_m_s": {"p99": 1.2}}, "morphology": _morph(55.3, 96.3)}}
    timeline = build_timeline("P-1", jobs, lambda job: summaries.get(job["id"]))
    assert timeline["patient_id"] == "P-1" and timeline["n_scans"] == 2
    first, second = timeline["scans"]
    assert first["input_sha256"] == "a" * 64 and first["date"] == "2025-03-01" and first["date_source"] == "scan_date"
    assert {job["job_id"] for job in first["jobs"]} == {"j1", "j2"} and set(first["models"]) == {X5D, M1}
    assert {job["release_label"] for job in first["jobs"]} == {"壁面 WSS", "WSS + TAWSS + OSI"}
    assert first["geometry"] == {"max_diameter_mm": 52.1, "sac_present": True, "sac_length_mm": 70.0, "sac_volume_ml": 90.0,
                                 "neck_diameter_mm": 19.0, "neck_length_mm": 22.0, "lumen_volume_ml": 150.0}
    assert first["models"][M1]["tawss_mean_pa"] == 0.66 and first["models"][M1]["stagnation_frac"] == 0.46
    assert second["models"][VOL] == {"speed_p99_m_s": 1.2, "aorta_delta_p_pa": 80.0}
    growth = timeline["growth"]["max_diameter_mm"]
    assert growth["days"] == 365 and growth["basis"] == "first_last" and growth["delta"] == pytest.approx(3.2)
    assert growth["per_year"] == pytest.approx(3.2 * 365.25 / 365) and growth["from"]["value"] == 52.1 and growth["to"]["date"] == "2026-03-01"
    assert timeline["growth"]["sac_volume_ml"]["delta"] == pytest.approx(6.3)
    assert "growth_recent" not in timeline                  # two scans: the recent interval is first → last
    diameter = next(s for s in timeline["series"] if s["key"] == "max_diameter_mm")
    assert diameter["group"] == "geometry" and [p["value"] for p in diameter["points"]] == [52.1, 55.3]
    assert len(timeline["notes"]) == 2


def test_model_series_never_connect_across_releases():
    jobs = [_job(1, "a" * 64, X5D, scan_date="2025-01-01"), _job(2, "b" * 64, M1, scan_date="2025-06-01"), _job(3, "c" * 64, X5D, scan_date="2026-01-01")]
    summaries = {"j1": _wall(10.0, 40.0), "j2": _wall(12.0, 41.0, cycle=True), "j3": _wall(14.0, 43.0)}
    timeline = build_timeline("P-1", jobs, lambda job: summaries[job["id"]])
    p99 = {s["release_id"]: [p["value"] for p in s["points"]] for s in timeline["series"] if s["key"] == "wss_p99_pa"}
    assert p99 == {X5D: [10.0, 14.0], M1: [12.0]}
    assert all(s.get("release_id") is None for s in timeline["series"] if s["group"] == "geometry")
    assert [p["value"] for p in next(s for s in timeline["series"] if s["key"] == "max_diameter_mm")["points"]] == [40.0, 41.0, 43.0]
    tawss = [s for s in timeline["series"] if s["key"] == "tawss_mean_pa"]
    assert len(tawss) == 1 and tawss[0]["release_id"] == M1 and tawss[0]["release_label"] == "WSS + TAWSS + OSI"
    assert timeline["growth"]["max_diameter_mm"]["delta"] == pytest.approx(3.0)
    assert timeline["growth_recent"]["max_diameter_mm"]["delta"] == pytest.approx(2.0) and timeline["growth_recent"]["max_diameter_mm"]["days"] == 214


def test_missing_dates_or_short_gaps_give_no_growth():
    jobs = [_job(1, "a" * 64, X5D), _job(2, "b" * 64, X5D, scan_date="2026-01-01")]
    summaries = {"j1": _wall(10.0, 40.0), "j2": _wall(11.0, 42.0)}
    timeline = build_timeline("P-1", jobs, lambda job: summaries[job["id"]])
    assert timeline["n_scans"] == 2 and timeline["growth"] == {} and "growth_recent" not in timeline
    assert timeline["scans"][1]["date_source"] == "created_at" or timeline["scans"][0]["date_source"] == "created_at"
    assert any("没有填写扫描日期" in note for note in timeline["notes"])
    jobs = [_job(1, "a" * 64, X5D, scan_date="2026-01-01"), _job(2, "b" * 64, X5D, scan_date="2026-01-20")]
    timeline = build_timeline("P-1", jobs, lambda job: summaries[job["id"]])
    assert timeline["growth"] == {} and any("不足 30 天" in note for note in timeline["notes"])


def test_empty_result_shape():
    timeline = build_timeline("P-9", [_job(1, "a" * 64, X5D)], lambda job: {})
    assert timeline["n_scans"] == 0 and timeline["scans"] == [] and timeline["series"] == [] and timeline["growth"] == {}
    assert timeline["patient_id"] == "P-9" and timeline["notes"]


def test_real_summaries_give_the_contract_numbers():
    m1, vol = FIXTURES / "20260922_173705_27065942b465/summary.json", FIXTURES / "20260920_174150_e0502f7ea512/summary.json"
    missing = [str(path) for path in (m1, vol) if not path.is_file()]
    if missing:
        pytest.fail(f"test fixture missing: {', '.join(missing)} (restore tests/fixtures/ or set WSS_DEPLOY_TEST_FIXTURES)", pytrace=False)
    m1 = json.loads(m1.read_text(encoding="utf-8"))
    values = model_values(m1)
    assert values["tawss_mean_pa"] == m1["cycle"]["fields"]["tawss"]["mean"]
    assert values["osi_high_frac"] == m1["cycle"]["fields"]["osi"]["area_frac"]["above_t0"]
    assert values["stagnation_frac"] == m1["cycle"]["stagnation"]["area_frac"] and values["wss_p99_pa"] == m1["peak"]["p99_pa"]
    assert geometry_values(m1)["max_diameter_mm"] == m1["morphology"]["aorta"]["max"]["max_diameter_mm"]
    vol = json.loads(vol.read_text(encoding="utf-8"))
    values = model_values(vol)
    assert values["speed_p99_m_s"] == vol["volume_statistics"]["speed_m_s"]["p99"] and "wss_p99_pa" not in values
    assert values["aorta_delta_p_pa"] == next(b["delta_p_pa"] for b in vol["morphology"]["branches"] if b["name"] == "主动脉")


def test_http_route_owner_isolation_and_onepage_timeline(tmp_path, monkeypatch):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        a = finished(service.manager, owner=owner, case_id="A", content=b"scan-one", patient_id="P 1/α", scan_date="2025-01-01",
                     summary_extra={"morphology": _morph(50.0, 80.0)})
        b = finished(service.manager, owner=owner, case_id="B", content=b"scan-two", patient_id="P 1/α", scan_date="2026-01-01",
                     summary_extra={"morphology": _morph(53.0, 88.0)})
        finished(service.manager, owner="intruder", case_id="X", content=b"scan-three", patient_id="P 1/α", scan_date="2025-06-01")
        path = "/api/patients/" + quote("P 1/α", safe="") + "/timeline"
        status, payload, _ = call(api, "GET", path, {"Cookie": cookie})
        assert status == 200 and payload["n_scans"] == 2 and [s["case_id"] for s in payload["scans"]] == ["A", "B"]
        assert payload["growth"]["max_diameter_mm"]["delta"] == pytest.approx(3.0)
        status, payload, _ = call(api, "GET", "/api/patients/" + quote("nobody") + "/timeline", {"Cookie": cookie})
        assert status == 200 and payload["n_scans"] == 0 and payload["scans"] == []
        status, payload, _ = call(api, "GET", "/api/patients/" + "x" * 81 + "/timeline", {"Cookie": cookie})
        assert status == 400
        status, _, _ = call(api, "GET", path, {})
        assert status == 401
        seen = {}

        def fake_render(summary, job=None, *, job_dir=None, snapshots=None, timeline=None):
            seen["timeline"] = timeline
            return "<html>one</html>"
        monkeypatch.setattr("wss_deploy.onepager.render_onepage", fake_render)
        status, body, _ = call(api, "GET", f"/api/jobs/{a['id']}/onepage", {"Cookie": cookie})
        assert status == 200 and seen["timeline"]["n_scans"] == 2 and seen["timeline"]["patient_id"] == "P 1/α"
        lone = finished(service.manager, owner=owner, case_id="L", content=b"lone", patient_id="P-lone")
        status, _, _ = call(api, "GET", f"/api/jobs/{lone['id']}/onepage", {"Cookie": cookie})
        assert status == 200 and seen["timeline"] is None
        # a one-pager without the keyword (W1 not merged yet) is called without it
        monkeypatch.setattr("wss_deploy.onepager.render_onepage", lambda summary, job=None, *, job_dir=None, snapshots=None: "<html>old</html>")
        status, body, _ = call(api, "GET", f"/api/jobs/{b['id']}/onepage", {"Cookie": cookie})
        assert status == 200 and body == b"<html>old</html>"
    finally:
        api.close(); service.close()


def test_manager_timeline_admin_view_and_light_snapshot_labels(tmp_path):
    mgr = manager(tmp_path)
    finished(mgr, owner="alice", case_id="A", content=b"one", patient_id="P-7", scan_date="2025-01-01")
    finished(mgr, owner="bob", case_id="B", content=b"two", patient_id="P-7", scan_date="2025-09-01")
    assert mgr.patient_timeline("P-7", "alice")["n_scans"] == 1
    assert mgr.patient_timeline(" P-7 ", "carol", all_owners=True)["n_scans"] == 2
    row = mgr.list("alice")[0]
    assert row["family_label"] == "壁面 WSS" and row["release_short"] == "REL_A" and row["has_cycle"] is False
    mgr.close()
