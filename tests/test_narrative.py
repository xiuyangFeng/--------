"""§17.2 automatic conclusion: generated sentences, the reviewer's edit and the PUT endpoint."""
from __future__ import annotations

import json

import pytest

from wss_deploy import narrative as N
from wss_deploy.jobs import JobError
from tests._c_helpers import Service, call, finished, manager

MORPHOLOGY = {
    "schema_version": "wss-deploy.morphology/v1", "station_mm": 1.0,
    "aorta": {"segment_id": 0, "name": "主动脉", "reference_diameter_mm": 18.2,
              "max": {"max_diameter_mm": 63.4, "equivalent_diameter_mm": 58.9, "s_from_root_mm": 120.0,
                      "distance_from_inlet_mm": 120.0, "xyz_mm": [1.0, 2.0, 3.0]},
              "sac": {"present": True, "length_mm": 85.0, "volume_ml": 96.3, "threshold_mm": 27.3},
              "neck": {"present": True, "length_mm": 80.0, "diameter_mean_mm": 19.1}},
    "lumen_volume_ml": 158.2, "branches": [],
}
WALL = {
    "fields": {"wss": {"units": "Pa"}}, "morphology": MORPHOLOGY,
    "peak": {"p99_pa": 16.6, "max_pa": 21.9},
    "wss_field_pa": {"area_frac_low": 0.34, "thresholds_pa": [0.4, 4.0, 7.0]},
    "reference_assessment": {"population": {"percentile": 32.35, "reference_count": 136}},
    "findings": {"items": [
        {"id": "F1", "kind": "high_wss_cluster", "branch": "左髂内", "value": 21.9},
        {"id": "F2", "kind": "high_wss_cluster", "branch": "右髂内", "value": 9.1},
        {"id": "F3", "kind": "low_wss_cluster", "branch": "主动脉", "value": 0.2}]},
}
VOLUME = {
    "fields": {"pressure": {"units": "Pa"}, "velocity": {"units": "m/s"}}, "morphology": MORPHOLOGY,
    "volume_statistics": {"speed_m_s": {"max": 1.234, "p99": 1.0}},
    "findings": {"items": [
        {"id": "F1", "kind": "max_speed", "branch": "右髂外", "value": 1.234},
        {"id": "F2", "kind": "pressure_drop", "branch": "主动脉", "value": 266.64},
        {"id": "F3", "kind": "pressure_drop", "branch": "左髂总", "value": 12.0},
        {"id": "F4", "kind": "low_speed_region", "branch": "主动脉", "value": 0.02}]},
}


def test_wall_narrative_states_shape_wss_and_the_population_percentile():
    out = N.build_narrative(WALL)
    assert out["schema_version"] == "wss-deploy.narrative/v1" and out["family"] == "wall"
    assert out["edited"] is None and out["edited_by"] is None and out["edited_at"] is None
    zh = out["zh"]
    assert len(zh) == 4 and zh[-1] == N.DISCLAIMER_ZH and out["en"][-1] == N.DISCLAIMER_EN
    # C1 (2026-09-30): lumen diameter wording; C2: the hotspot sentence states its real definition (≥ the case p99).
    assert zh[0] == ("主动脉管腔最大直径 63.4 mm（截面最大 Feret 直径，不含附壁血栓与管壁），位于入口下 120.0 mm；"
                     "瘤体长 85.0 mm；体积约 96 mL；近端瘤颈长 80.0 mm、平均直径 19.1 mm。")
    assert zh[1] == "低 WSS（< 0.4 Pa）区占壁面 34%，主要位于主动脉；峰值 WSS 最高的区域（不低于本例 p99，16.6 Pa）2 处，最高 21.9 Pa 位于左髂内。"
    assert out["en"][0].startswith("Largest aortic lumen diameter 63.4 mm (maximum Feret diameter of the cross-section, "
                                   "excluding mural thrombus and the wall)")
    assert "2 region(s) of highest peak WSS (at or above this case's p99, 16.6 Pa)" in out["en"][1]
    assert zh[2] == "壁面 WSS 空间 p99 为 16.6 Pa，处于 136 例参照人群第 32 百分位。"
    assert "63.4 mm" in out["en"][0] and "left internal iliac" in out["en"][1]
    assert "at the 32nd percentile of the 136-case reference cohort" in out["en"][2]
    assert [N._ordinal(v) for v in ("1", "2", "3", "4", "11", "12", "13", "21", "112")] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "112th"]


def test_volume_narrative_uses_the_aorta_pressure_drop_and_the_peak_speed():
    out = N.build_narrative(VOLUME)
    assert out["family"] == "volume"
    zh = out["zh"]
    assert zh[1] == "主动脉近远端压差 266.6 Pa，最大速度 1.23 m/s 位于右髂外。"
    assert "低速（滞留）区 1 处" in zh[2] and zh[-1] == N.DISCLAIMER_ZH
    assert "266.6 Pa" in out["en"][1] and "right external iliac" in out["en"][1]


def test_narrative_states_only_what_exists():
    bare = N.build_narrative({"fields": {"wss": {"units": "Pa"}}})
    assert bare["zh"] == [N.DISCLAIMER_ZH] and bare["en"] == [N.DISCLAIMER_EN]
    assert N.build_narrative(None)["zh"] == [N.DISCLAIMER_ZH]
    no_sac = json.loads(json.dumps(WALL))
    no_sac["morphology"]["aorta"]["sac"] = {"present": False}
    no_sac["morphology"]["aorta"]["neck"] = {"present": False}
    del no_sac["reference_assessment"]
    out = N.build_narrative(no_sac)
    assert out["zh"][0] == ("主动脉管腔最大直径 63.4 mm（截面最大 Feret 直径，不含附壁血栓与管壁），位于入口下 120.0 mm，"
                            "管腔未见瘤样扩张（< 1.5 × 参考直径 18.2 mm），不能据此排除动脉瘤。")
    assert out["en"][0].endswith("; no aneurysmal dilatation of the lumen (< 1.5 × the reference diameter of 18.2 mm); "
                                 "an aneurysm cannot be excluded on this basis.")
    assert out["zh"][2] == "壁面 WSS 空间 p99 为 16.6 Pa。"   # no cohort clause without a percentile


def test_merge_edit_and_display_text():
    auto = N.build_narrative(WALL)
    edited = N.merge_edit(auto, "  审阅人重写  ", edited_by="R", edited_at="2026-09-22T10:00:00+08:00")
    assert edited["edited"] == "  审阅人重写  " and edited["edited_by"] == "R" and edited["zh"] == auto["zh"]
    assert N.display_text(edited) == "审阅人重写"
    restored = N.merge_edit(auto, "")
    assert restored["edited"] is None and restored["edited_by"] is None and restored["edited_at"] is None
    assert N.display_text(restored).startswith("主动脉管腔最大直径")
    assert N.display_text(restored, "en").startswith("Largest aortic lumen diameter")
    assert N.display_text(None) == ""


# ------------------------------------------------------------------ service layer
def test_narrative_endpoint_writes_the_sidecar_mirrors_it_and_respects_the_lock(tmp_path):
    mgr = manager(tmp_path)
    job = finished(mgr, case_id="A", summary_extra={"morphology": MORPHOLOGY, "narrative": N.build_narrative(WALL)})
    result = mgr.narrative(job["id"], "owner", {"text": "审阅人结论", "version": job["version"]})
    document = result["narrative"]
    assert document["edited"] == "审阅人结论" and document["edited_by"] == "owner" and document["edited_at"]
    assert document["auto"]["zh"][-1] == N.DISCLAIMER_ZH
    assert result["version"] == job["version"] + 1
    job_dir = tmp_path / job["id"]
    assert json.loads((job_dir / "narrative.json").read_text(encoding="utf-8")) == document
    stored = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))["narrative"]
    assert stored["edited"] == "审阅人结论" and stored["zh"][-1] == N.DISCLAIMER_ZH
    html = (job_dir / "report.html").read_text(encoding="utf-8")
    embedded = json.loads(html.split('type="application/json">', 1)[1].split("</script>", 1)[0])
    assert embedded["narrative"]["edited"] == "审阅人结论"
    snapshot = mgr.get(job["id"], "owner")
    assert snapshot["events"][-1]["action"] == "narrative_updated"
    # An empty string restores the automatic text.
    restored = mgr.narrative(job["id"], "owner", {"text": ""})["narrative"]
    assert restored["edited"] is None and restored["edited_by"] is None
    assert json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))["narrative"]["edited"] is None
    with pytest.raises(JobError):
        mgr.narrative(job["id"], "owner", {"text": "x" * 4001})
    with pytest.raises(JobError):
        mgr.narrative(job["id"], "owner", {"text": 3})
    version = mgr.get(job["id"], "owner")["version"]
    mgr.review(job["id"], "owner", {"version": version, "decision": "approve", "reviewer": "R"})
    with pytest.raises(JobError) as info:
        mgr.narrative(job["id"], "owner", {"text": "锁定后不可改"})
    assert info.value.status == 409
    mgr.close()


def test_http_narrative_put_and_served_file(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        job = finished(service.manager, owner=owner, case_id="A", summary_extra={"morphology": MORPHOLOGY})
        status, _, _ = call(api, "GET", f"/api/jobs/{job['id']}/files/narrative.json", {"Cookie": cookie})
        assert status == 404
        status, payload, _ = call(api, "PUT", f"/api/jobs/{job['id']}/narrative", headers, {"text": "结论 A"})
        assert status == 200 and payload["narrative"]["edited"] == "结论 A"
        status, served, _ = call(api, "GET", f"/api/jobs/{job['id']}/files/narrative.json", {"Cookie": cookie})
        assert status == 200 and served["edited"] == "结论 A"
        status, _, _ = call(api, "PUT", f"/api/jobs/{job['id']}/narrative", headers, {"text": "x" * 4001})
        assert status == 400
    finally:
        api.close(); service.close()
