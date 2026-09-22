"""The one-page report is rendered from summary.json only, escapes everything, and is served on demand."""
from __future__ import annotations

import http.client
import json
import threading

from wss_deploy.jobs import JobManager
from wss_deploy.onepager import LIMITATIONS, family_of, render_onepage
from wss_deploy.server import ServiceHTTPServer, SessionStore


def wall_summary():
    return {
        "case_id": "CASE-<b>1</b>", "created_at": "2026-09-20 10:00:00",
        "case_metadata": {"patient_id": "P-01", "scan_label": "基线", "scan_date": "2026-09-01", "tags": ["AAA"], "notes": ""},
        "model_release": {"registry_id": "X5D_v51_5seed_20260916", "fingerprint": "ab" * 32},
        "feature_contract": {"version": "wss-features/1.0", "source_hash": "cd" * 32},
        "run_identity": "ef" * 32, "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21, "label": "peak_systole"}],
        "fields": {"wss": {"units": "Pa", "location": "wall", "kind": "scalar"}},
        "input_check": {"unit": "mm", "vertices": 1000, "faces": 2000, "area_mm2": 40000.0, "bbox_size_mm": [100, 90, 250],
                        "openings": 5, "components": 1, "quality": {"grade_label": "已检查项通过（存在未评估项）"}, "flags": ["1 条边绕序不一致"]},
        "geometry": {"主动脉": {"length_mm": 200.0, "radius_min_mm": 7.5, "radius_median_mm": 12.7, "max_diameter_mm": 63.4, "stenosis_index": 0.4, "tortuosity": 1.06}},
        "peak": {"p99_pa": 16.64, "max_pa": 48.6, "branch": "左髂内", "s_from_inlet_mm": 263.9},
        "wss_field_pa": {"area_frac_low": 0.433, "area_frac_high": 0.169, "area_frac_very_high": 0.097, "area_low_mm2": 14780.0, "thresholds_pa": [0.4, 4.0, 7.0]},
        "per_branch": {"主动脉": {"wss_mean_pa": 0.79, "wss_p99_pa": 5.09, "wss_max_pa": 8.14, "frac_low": 0.56, "frac_high": 0.02, "area_mm2": 25984.0}},
        "quality": {"level": "good", "label": "模型集成稳定", "reasons": []},
        "findings": {"schema_version": "wss-deploy.findings/v1", "items": [
            {"id": "F2", "kind": "low_wss_cluster", "label": "主动脉低 WSS 区", "branch": "主动脉", "value": 0.2, "units": "Pa", "rank": 2, "severity": "attention", "definition": "< 0.4 Pa 的连通簇"},
            {"id": "F1", "kind": "high_wss_cluster", "label": "左髂内高 WSS 区", "branch": "左髂内", "value": 21.9, "units": "Pa", "rank": 1, "severity": "attention", "definition": "≥ p99 的连通簇"}]},
        "trust": {"schema_version": "wss-deploy.trust/v1", "bits": {"1": "interpolation_uncovered", "2": "rough_surface"},
                  "fractions": {"interpolation_uncovered": 0.02, "rough_surface": 0.05},
                  "sources": [{"bit": 1, "label": "插值无支撑", "rule": "1.5 mm 内无预测点"}]},
        "review": {"status": "reviewed", "by": "R-01", "at": "2026-09-20T18:00:00+08:00", "note": "已核对", "version": 7},
        "timing_s": {"ingest": 0.1, "centerline": 3.3, "total": 30.3}, "device": "cuda", "gpu": "RTX 4090",
    }


def test_wall_onepage_contains_key_numbers_findings_trust_review_and_escapes(tmp_path):
    html = render_onepage(wall_summary(), {"id": "job-1"})
    assert "<script" not in html and "CASE-&lt;b&gt;1&lt;/b&gt;" in html
    for text in ("16.64 Pa", "48.60 Pa", "43.3%", "16.9%", "左髂内高 WSS 区", "主动脉低 WSS 区", "插值无支撑", "5.0%",
                 "已审阅签字", "R-01", "X5D_v51_5seed_20260916", "wss-features/1.0", "cdcdcdcdcdcdcdcd", "63.4", "@page{size:A4",
                 "模型集成稳定", "job-1", "RTX 4090"):
        assert text in html, text
    assert html.index("F1") < html.index("F2")  # findings ordered by rank
    for item in LIMITATIONS:
        assert item in html
    assert family_of(wall_summary()) == "wall"


MORPHOLOGY = {
    "schema_version": "wss-deploy.morphology/v1", "station_mm": 1.0,
    "method": {"max_diameter": "环上最大 Feret 直径", "equivalent_diameter": "2·sqrt(面积/π)"},
    "aorta": {"segment_id": 0, "name": "主动脉", "reference_diameter_mm": 18.2,
              "max": {"max_diameter_mm": 63.4, "equivalent_diameter_mm": 58.9, "distance_from_inlet_mm": 120.0,
                      "s_from_root_mm": 120.0, "xyz_mm": [1, 2, 3]},
              "sac": {"present": True, "length_mm": 85.0, "volume_ml": 96.3, "threshold_mm": 27.3},
              "neck": {"present": True, "length_mm": 80.0, "diameter_mean_mm": 19.1, "diameter_min_mm": 17.8, "diameter_max_mm": 21.0}},
    "lumen_volume_ml": 158.2, "notes": ["最大直径站的截面明显斜切"],
    "branches": [{"segment_id": 0, "name": "主动脉", "length_mm": 211.1, "tortuosity": 1.06, "diameter_min_mm": 15.1,
                  "diameter_max_mm": 63.4, "diameter_mean_mm": 30.2, "wss_p99_pa": 16.6, "wss_mean_pa": 2.1,
                  "area_frac_low": 0.34, "area_frac_high": 0.02, "delta_p_pa": None,
                  "speed_mean_m_s": None, "speed_max_m_s": None}],
}


def test_onepage_leads_with_the_conclusion_and_lists_the_morphology(tmp_path):
    summary = wall_summary()
    summary["morphology"] = MORPHOLOGY
    summary["narrative"] = {"schema_version": "wss-deploy.narrative/v1",
                            "zh": ["主动脉最大直径 63.4 mm。", "以上为固定收缩期单帧预测的参考描述，非诊断结论。"],
                            "en": ["..."], "edited": None, "edited_by": None, "edited_at": None}
    html = render_onepage(summary, {"id": "job-1"})
    assert "<h2>结论（参考）" in html and '<span class="badge auto">自动生成</span>' in html
    assert '<span class="badge">审阅人已修改</span>' not in html
    assert html.index("<h2>结论（参考）") < html.index("<h2>关键数字")
    for text in ("瘤体形态", "63.4 mm", "85 mm", "96 mL", "19.1", "158 mL", "18.2", "斜切",
                 "血管分支表", "211", "1.06", "15.1–63.4", "最大 Feret 直径"):
        assert text in html, text
    assert html.index("<h2>瘤体形态") < html.index("<h2>血管分支表") < html.index("<h2>几何")
    # The reviewer's text replaces the generated one and is badged.
    summary["narrative"]["edited"] = "审阅人重写的结论"
    summary["narrative"]["edited_by"] = "R-01"
    edited = render_onepage(summary, {"id": "job-1"})
    assert '<span class="badge">审阅人已修改</span>' in edited and "审阅人重写的结论" in edited
    assert "主动脉最大直径 63.4 mm。</p>" not in edited and "R-01" in edited
    # Without the blocks the sections simply disappear.
    plain = render_onepage(wall_summary(), {"id": "job-1"})
    assert '<h2>结论（参考）' not in plain and "<h2>瘤体形态" not in plain and "<h2>血管分支表" not in plain


def test_volume_onepage_branch_table_uses_the_volume_columns():
    summary = {"case_id": "V", "fields": {"pressure": {"units": "Pa"}, "velocity": {"units": "m/s"}},
               "morphology": {**MORPHOLOGY, "branches": [{"segment_id": 0, "name": "主动脉", "length_mm": 211.1, "tortuosity": 1.06,
                                                          "diameter_min_mm": 15.1, "diameter_max_mm": 63.4, "diameter_mean_mm": 30.2,
                                                          "delta_p_pa": 266.6, "speed_mean_m_s": 0.21, "speed_max_m_s": 0.69,
                                                          "wss_p99_pa": None, "wss_mean_pa": None}]}}
    html = render_onepage(summary, {})
    assert "血管分支表" in html and "ΔP Pa" in html and "266.6" in html and "0.690" in html and "p99 Pa" not in html


def test_volume_onepage_uses_volume_numbers_and_pressure_drops():
    summary = {"case_id": "V", "fields": {"pressure": {"units": "Pa"}, "velocity": {"units": "m/s"}},
               "volume_statistics": {"speed_m_s": {"p99": 0.597, "mean": 0.21, "max": 0.69}, "pressure_interior_pa": {"min": -489.5, "max": 321.0},
                                     "pressure_wall_pa": {"min": -480.0, "max": 328.0}},
               "streamlines": {"line_count": 521}, "notes": ["PF6 压力相对于当前帧体积平均压力；不能恢复绝对血压。"],
               "findings": {"items": [{"id": "F3", "kind": "pressure_drop", "label": "主动脉压差", "branch": "主动脉", "value": 266.6, "units": "Pa", "rank": 3, "severity": "note", "definition": "近端 10% 与远端 10% 平均压差"}]}}
    html = render_onepage(summary, {})
    assert family_of(summary) == "volume"
    for text in ("0.597 m/s", "-490 ～ 321 Pa", "521 条", "各分支近远端压差", "266.6", "2.00", "未审阅", "PF6 压力相对于当前帧体积平均压力"):
        assert text in html, text
    assert "全场 p99" not in html


def test_onepage_tolerates_minimal_summary():
    html = render_onepage({}, None)
    assert "匿名病例" in html and "局限性声明" in html and "无数据" in html


def test_http_onepage_route_and_embeddable_headers(tmp_path):
    manager = JobManager(tmp_path / "jobs", stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})
    sessions = SessionStore(tmp_path / "sessions.json", shared=False)
    server = ServiceHTTPServer(("127.0.0.1", 0), manager, sessions)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    api = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        api.request("GET", "/api/session")
        response = api.getresponse(); response.read()
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        owner = sessions.lookup(cookie)[1]["owner"]
        job = manager.create(owner, content=b"solid", filename="case.stl", case_id="ONE")
        api.request("GET", f"/api/jobs/{job['id']}/onepage", headers={"Cookie": cookie})
        response = api.getresponse(); response.read()
        assert response.status == 409  # not done yet
        with manager.lock:
            record = manager.jobs[job["id"]]
            record.update(status="done", stage="B")
            manager._event(record, "finished")
            job_dir = manager.root / job["id"]
            (job_dir / "summary.json").write_text(json.dumps(wall_summary()), encoding="utf-8")
            (job_dir / "report.html").write_text("<html></html>", encoding="utf-8")
        api.request("GET", f"/api/jobs/{job['id']}/onepage", headers={"Cookie": cookie})
        response = api.getresponse(); body = response.read().decode("utf-8")
        assert response.status == 200 and response.getheader("Content-Type").startswith("text/html")
        assert "frame-ancestors 'self'" in response.getheader("Content-Security-Policy")
        assert response.getheader("X-Frame-Options") == "SAMEORIGIN"
        assert "16.64 Pa" in body and "左髂内高 WSS 区" in body and job["id"] in body
        assert not (job_dir / "onepage.html").exists()  # rendered on demand only
        api.request("GET", f"/api/jobs/{job['id']}/report", headers={"Cookie": cookie})
        response = api.getresponse(); response.read()
        assert response.status == 200 and "frame-ancestors 'self'" in response.getheader("Content-Security-Policy")
        api.request("GET", f"/api/jobs/{job['id']}/files/summary.json", headers={"Cookie": cookie})
        response = api.getresponse(); response.read()
        assert response.status == 200 and "frame-ancestors 'none'" in response.getheader("Content-Security-Policy")
        api.request("GET", f"/api/jobs/{job['id']}", headers={"Cookie": cookie})
        response = api.getresponse(); response.read()
        assert "frame-ancestors 'none'" in response.getheader("Content-Security-Policy")
    finally:
        api.close(); server.shutdown(); server.server_close(); manager.close()


def test_onepage_picture_section_hint_and_embedded_images(tmp_path):
    """§15.1: the 配图 section embeds snapshot PNGs as data URIs, or explains how to make them."""
    import base64
    from wss_deploy.onepager import NO_SNAPSHOTS_HINT
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
    assert NO_SNAPSHOTS_HINT in render_onepage(wall_summary(), {"id": "job-1"})           # no job directory at all
    assert NO_SNAPSHOTS_HINT in render_onepage(wall_summary(), {"id": "job-1"}, job_dir=tmp_path)  # directory without pictures
    (tmp_path / "snapshot_front.png").write_bytes(png)
    (tmp_path / "snapshot_evil.png").write_bytes(png)
    (tmp_path / "snapshots.json").write_text(json.dumps({"schema_version": "wss-deploy.snapshots/v1", "items": [
        {"name": "front", "file": "snapshot_front.png", "caption": "前视 · WSS · Pa"},
        {"name": "bad", "file": "../outside.png", "caption": "路径逃逸"},
        {"name": "missing", "file": "snapshot_missing.png", "caption": "文件不在"}]}), encoding="utf-8")
    html = render_onepage(wall_summary(), {"id": "job-1"}, job_dir=tmp_path)
    assert NO_SNAPSHOTS_HINT not in html and "前视 · WSS · Pa" in html
    assert html.count("data:image/png;base64,") == 1  # escaping/containment drops the other two entries
    assert "路径逃逸" not in html and "文件不在" not in html
