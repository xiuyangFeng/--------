"""Workspace v2 data layer (WORKSPACE_V2_CONTRACT.md §2–§3): manifest assembly, byte-exact arrays, derived fields."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import numpy as np
import pytest

from wss_deploy import cycle_fields as CF
from wss_deploy import v2_data as V
from wss_deploy.paths import BRANCH_CN, PROJECT_ROOT
from wss_deploy.report import _script_json

DEV_ROOT = PROJECT_ROOT / "outputs" / "wss_deploy_jobs"
DEV_JOBS = {
    "20260929_230442_a7273f1670e4": ("wall", ["wss", "tawss", "osi", "rrt", "ecap"]),
    "20260929_230450_d1428792b846": ("volume", ["pressure", "speed", "velocity", "wall_pressure"]),
    "20260922_173705_27065942b465": ("wall", ["wss", "tawss", "osi", "rrt", "ecap"]),
    "20260920_113510_0aa3ba155bfa": ("volume", ["pressure", "speed", "velocity", "wall_pressure"]),
    "20260920_144135_673ccd0e36b1": ("wall", ["wss"]),
    "20260918_152113_e2e53937b25a": ("wall", ["wss"]),
    "20260918_154058_a72784880799": ("wall", ["wss"]),
    "20260918_162707_f22146842b96": ("wall", ["wss"]),
}
# manifest key -> embedded key (wall ``wss-report-arrays`` / volume ``volume-arrays``); xf fields are checked separately
WALL_SOURCES = {"mv": "mv", "mf": "mf", "ms": "ms", "mt": "mt", "pv": "pv", "ps": "ps", "p_s": "p_s", "p_th": "p_th", "p_r": "p_r",
                "p_dj": "p_dj", "cv": "cv", "cr": "cr", "ce": "ce", "cs": "cs", "f.wss.d": "mw", "f.wss.r": "pw"}
VOLUME_SOURCES = {"mv": "vertices", "mf": "faces", "mt": "wall_trust", "vpts": "pts", "vseg": "segment", "vis_wall": "is_wall",
                  "vtrust": "trust", "v_s": "s_from_root_mm", "v_r": "radius_mm", "v_dw": "dist_to_wall_mm", "cv": "center",
                  "cr": "center_radius_mm", "ce": "edges", "cs": "center_segment", "ct": "tangent", "f.pressure.r": "pressure_pa",
                  "f.velocity.r": "velocity_m_s", "f.wall_pressure.d": "wall_pressure_pa"}


def _embedded(job_dir: Path) -> tuple[dict, dict]:
    text = (job_dir / "report.html").read_text(encoding="utf-8")
    meta = json.loads(V._script_payload(text, "wss-report-meta"))
    arrays_id = "volume-arrays" if 'id="volume-arrays"' in text else "wss-report-arrays"
    return meta, json.loads(V._script_payload(text, arrays_id))


def _record(job_dir: Path) -> dict:
    return json.loads((job_dir / "job.json").read_text(encoding="utf-8"))


def _typed(data: V.JobData, key: str) -> np.ndarray:
    return data.report.array(key)


dev = pytest.mark.skipif(not all((DEV_ROOT / job / "report.html").is_file() for job in DEV_JOBS),
                         reason="development job copies are not present")


@pytest.fixture(autouse=True)
def _fresh_cache():
    V.clear_cache()
    yield
    V.clear_cache()


# -------------------------------------------------------------------------------------------- the eight dev jobs
@dev
@pytest.mark.parametrize("job_id", sorted(DEV_JOBS))
def test_manifest_builds_for_every_development_job_and_arrays_are_the_embedded_bytes(job_id):
    job_dir = DEV_ROOT / job_id
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in job_dir.iterdir()}
    family, field_ids = DEV_JOBS[job_id]
    data = V.load(job_dir)
    manifest = V.build_manifest(data, _record(job_dir))
    json.dumps(manifest, ensure_ascii=False, allow_nan=False)            # serialisable as the server sends it
    assert manifest["schema"] == V.SCHEMA and manifest["result"]["family"] == family
    assert [f["id"] for f in manifest["fields"]] == field_ids
    assert len(manifest["data_version"]) == 16 and manifest["job"]["id"] == job_id and manifest["job"]["status"] == "done"
    meta, embedded = _embedded(job_dir)
    sources = WALL_SOURCES if family == "wall" else VOLUME_SOURCES
    for key, entry in manifest["arrays"].items():
        raw = data.report.raw(key)
        assert len(raw) == entry["bytes"] == int(np.prod(entry["shape"])) * np.dtype(V.DTYPES[entry["dtype"]]).itemsize
        assert entry["url"] == f"/api/v2/jobs/{job_id}/arrays/{key}"
        if key in sources:
            assert raw == base64.b64decode(embedded[sources[key]]), key
    # every embedded array the classic viewer uses is offered
    for key, name in sources.items():
        if isinstance(embedded.get(name), str):
            assert key in manifest["arrays"], key
    for field in manifest["fields"]:
        assert field["tier"] in {"geometry", "model", "derived"}
        for which in ("display", "read"):
            if field["arrays"][which]:
                assert field["arrays"][which] in manifest["arrays"]
    # every file of the job directory is untouched
    assert before == {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in job_dir.iterdir()}


@dev
def test_cycle_fields_match_the_embedded_arrays_and_rrt_ecap_follow_cycle_fields():
    job_dir = DEV_ROOT / "20260929_230442_a7273f1670e4"
    data = V.load(job_dir)
    manifest = V.build_manifest(data, _record(job_dir))
    meta, embedded = _embedded(job_dir)
    xf = embedded["xf"]
    fields = {f["id"]: f for f in manifest["fields"]}
    for field_id in ("tawss", "osi"):
        key = meta["fields"][field_id]["array_key"]
        assert data.report.raw(fields[field_id]["arrays"]["display"]) == base64.b64decode(xf[key]["m"])
        assert data.report.raw(fields[field_id]["arrays"]["read"]) == base64.b64decode(xf[key]["p"])
        assert fields[field_id]["temporal"] == "cycle_summary" and fields[field_id]["time_index"] is None
    for which, part in (("read", "p"), ("display", "m")):
        tawss = np.frombuffer(base64.b64decode(xf[meta["fields"]["tawss"]["array_key"]][part]), "<f4")
        osi = np.frombuffer(base64.b64decode(xf[meta["fields"]["osi"]["array_key"]][part]), "<f4")
        expected = CF.derive_indices(tawss, osi)
        for name in ("rrt", "ecap"):
            got = _typed(data, fields[name]["arrays"][which])
            assert got.dtype == np.float32
            np.testing.assert_array_equal(got, expected[name].astype(np.float32))
    assert fields["rrt"]["tier"] == fields["ecap"]["tier"] == "derived"
    assert fields["rrt"]["derived_from"] == ["tawss", "osi"] and fields["rrt"]["units"] == "1/Pa"
    assert fields["wss"]["tier"] == "model" and fields["wss"]["temporal"] == "frame" and fields["wss"]["time_index"] == 0
    assert fields["tawss"]["display"]["log_scale"] is True and fields["osi"]["display"]["threshold_direction"] == "above"
    assert manifest["time"]["cycle"]["period_s"] == 0.8


@dev
def test_volume_speed_and_streamlines_come_from_the_embedded_velocity():
    job_dir = DEV_ROOT / "20260929_230450_d1428792b846"
    data = V.load(job_dir)
    manifest = V.build_manifest(data, _record(job_dir))
    _, embedded = _embedded(job_dir)
    velocity = np.frombuffer(base64.b64decode(embedded["velocity_m_s"]), "<f4").reshape(-1, 3)
    np.testing.assert_array_equal(_typed(data, "f.speed.r"), np.linalg.norm(velocity, axis=1).astype(np.float32))
    fields = {f["id"]: f for f in manifest["fields"]}
    assert fields["velocity"]["kind"] == "vector" and manifest["arrays"]["f.velocity.r"]["shape"] == [len(velocity), 3]
    assert fields["pressure"]["arrays"] == {"display": None, "read": "f.pressure.r"}
    assert fields["wall_pressure"]["arrays"] == {"display": "f.wall_pressure.d", "read": None}
    assert fields["speed"]["tier"] == fields["wall_pressure"]["tier"] == "model"
    lines = embedded["streamlines"]
    xyz = np.concatenate([np.frombuffer(base64.b64decode(line["points"]), "<f4") for line in lines]).reshape(-1, 3)
    speed = np.concatenate([np.frombuffer(base64.b64decode(line["speed_m_s"]), "<f4") for line in lines])
    np.testing.assert_array_equal(_typed(data, "sl.xyz"), xyz)
    np.testing.assert_array_equal(_typed(data, "sl.speed"), speed)
    offsets = _typed(data, "sl.offsets")
    assert offsets[0] == 0 and offsets[-1] == len(xyz) and len(offsets) == len(lines) + 1
    assert manifest["geometry"]["streamlines"]["xyz"] == "sl.xyz" and manifest["geometry"]["points"]["xyz"] == "vpts"
    assert manifest["analysis"]["modules"] == embedded["modules"]
    assert manifest["geometry"]["display_mesh"]["segment"] is None and manifest["geometry"]["centerline"]["tangent"] == "ct"


@dev
def test_branch_keys_come_from_the_branch_name_table_and_openings_are_listed():
    job_dir = DEV_ROOT / "20260929_230442_a7273f1670e4"
    manifest = V.build_manifest(V.load(job_dir), _record(job_dir))
    branches = manifest["geometry"]["branches"]
    assert {b["key"] for b in branches} == set(BRANCH_CN)
    for branch in branches:
        assert BRANCH_CN[branch["key"]] == branch["name"]
    root = next(b for b in branches if b["key"] == "root")
    assert root["parent"] == -1 and root["length_mm"] > 100
    assert len(manifest["geometry"]["openings"]) == 5
    assert manifest["job"]["companions"] == [{"release_id": "PF6_VF6_peak_3seed_20260920", "job_id": "20260929_230450_d1428792b846"}]


# ------------------------------------------------------------------------------------- synthetic reports (old shapes)
def _write_report(job_dir: Path, meta: dict, arrays: dict, *, volume: bool = False) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    array_id = "volume-arrays" if volume else "wss-report-arrays"
    (job_dir / "report.html").write_text(
        '<!doctype html><html><body><!--WSS_META_START--><script id="wss-report-meta" type="application/json">'
        + _script_json(meta) + '</script><!--WSS_META_END-->'
        + f'<script id="{array_id}" type="application/json">' + _script_json(arrays) + "</script></body></html>", encoding="utf-8")


def _b64(values, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(values, dtype=dtype).tobytes()).decode("ascii")


def _old_wall(tmp_path: Path, *, meta_extra=None, nan=False) -> Path:
    """A pre-v0.14 wall page: int32 branch ids (no ``dt``), no trust / junction distance, no fields or time axis."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 3]], np.uint32)
    mw = np.array([1.0, np.nan if nan else 2.0, 3.0, 4.0], np.float32)
    pts = np.arange(15, dtype=np.float32).reshape(5, 3)
    arrays = {"mv": _b64(vertices, np.float32), "mf": _b64(faces, np.uint32), "mw": _b64(mw, np.float32),
              "ms": _b64([0, 0, 1, 1], np.int32), "pv": _b64(pts, np.float32), "pw": _b64([1, 2, 3, 4, 5], np.float32),
              "ps": _b64([0, 0, 1, 1, 2], np.int32), "p_s": _b64(np.arange(5), np.float32), "p_th": _b64(np.zeros(5), np.float32),
              "p_r": _b64(np.ones(5), np.float32), "cv": _b64(pts[:3], np.float32), "cr": _b64([1, 1, 1], np.float32),
              "ce": _b64([[0, 1], [1, 2]], np.uint32), "cs": _b64([0, 0, 1], np.int32), "unroll_branches": []}
    meta = {"case_id": "OLD_CASE", "peak": {"p99_pa": 4.9, "max_pa": 5.0}, "wss_field_pa": {"p99": 4.9},
            "branch_names": {"0": "主动脉", "1": "左髂总", "2": "某未知分支"}, **(meta_extra or {})}
    job_dir = tmp_path / "20260101_000000_old"
    _write_report(job_dir, meta, arrays)
    return job_dir


def test_a_pre_v014_wall_report_yields_a_manifest_with_int32_branch_ids_and_nan_kept(tmp_path):
    job_dir = _old_wall(tmp_path, nan=True)
    data = V.load(job_dir)
    manifest = V.build_manifest(data, {"id": job_dir.name, "status": "done", "case_id": "OLD_CASE"})
    assert [f["id"] for f in manifest["fields"]] == ["wss"]
    assert manifest["arrays"]["ms"]["dtype"] == "int32" and manifest["arrays"]["ps"]["dtype"] == "int32"
    assert "mt" not in manifest["arrays"] and "p_dj" not in manifest["arrays"]
    assert manifest["geometry"]["display_mesh"]["trust"] is None and manifest["geometry"]["points"]["dist_to_junction_mm"] is None
    display = data.report.array("f.wss.d")
    assert np.isnan(display[1]) and display[0] == 1.0                         # missing stays NaN, never 0
    assert manifest["frame"]["rotation"] is None and manifest["frame"]["orientation"]["anterior_posterior"] == "unknown"
    assert manifest["time"]["mode"] == "single_frame" and len(manifest["time"]["axis"]) == 1
    keys = {b["id"]: b["key"] for b in manifest["geometry"]["branches"]}
    assert keys == {0: "root", 1: "left_cia", 2: None}                        # unknown names give null keys
    assert manifest["analysis"]["zones"] is None and manifest["model_card"] is None
    assert manifest["job"]["display_name"] == "OLD_CASE"
    json.dumps(manifest, allow_nan=False)


def test_malformed_optional_arrays_are_left_out_and_missing_required_ones_fail(tmp_path):
    job_dir = _old_wall(tmp_path)
    text = (job_dir / "report.html").read_text(encoding="utf-8")
    meta = json.loads(V._script_payload(text, "wss-report-meta")); arrays = json.loads(V._script_payload(text, "wss-report-arrays"))
    arrays["p_r"] = "abc"                                       # not valid base64 length
    _write_report(job_dir, meta, arrays)
    manifest = V.build_manifest(V.load(job_dir), {"status": "done"})
    assert "p_r" not in manifest["arrays"] and manifest["geometry"]["points"]["radius_mm"] is None
    del arrays["pv"]
    _write_report(job_dir, meta, arrays)
    V.clear_cache()
    with pytest.raises(V.DataError):
        V.load(job_dir)
    (job_dir / "report.html").unlink()
    with pytest.raises(V.DataError) as missing:
        V.load(job_dir)
    assert missing.value.status == 404


def test_orientation_rules(tmp_path):
    frame = {"rotation": np.eye(3).tolist(), "origin_mm": [0, 0, 0], "direction_source": "unknown_stl", "convention": "c"}
    job_dir = _old_wall(tmp_path, meta_extra={"frame_transform": frame, "outlets_confirmed": True})
    data = V.load(job_dir)
    manual = [{"source": "manual_confirmation", "acknowledged": True}]
    automatic = [{"source": "automatic_high_confidence", "acknowledged": False}]
    got = V.build_manifest(data, {"status": "done", "mapping_history": manual})["frame"]
    assert got["orientation"] == {"left_right": "confirmed", "superior_inferior": "derived", "anterior_posterior": "inferred"}
    assert got["direction_source"] == "unknown_stl" and got["rotation"] == np.eye(3).tolist()
    got = V.build_manifest(data, {"status": "done", "mapping_history": automatic})["frame"]
    assert got["orientation"]["left_right"] == "inferred"
    patient = _old_wall(tmp_path / "p", meta_extra={"frame_transform": {**frame, "direction_source": "dicom_patient"}, "outlets_confirmed": False})
    got = V.build_manifest(V.load(patient), {"status": "done"})["frame"]["orientation"]
    assert got["anterior_posterior"] == "confirmed" and got["left_right"] == "inferred"


def test_review_narrative_and_identity_follow_the_record_and_summary(tmp_path):
    narrative = {"schema_version": "wss-deploy.narrative/v1", "zh": ["自动一句。"], "en": ["Auto."], "edited": None,
                 "edited_by": None, "edited_at": None}
    job_dir = _old_wall(tmp_path, meta_extra={"narrative": narrative, "review": {"status": "unreviewed"},
                                              "zones": {"schema_version": "wss-deploy.zones/v1", "zones": []}})
    (job_dir / "summary.json").write_text(json.dumps({"narrative": {**narrative, "zh": ["摘要里的最新一句。"]},
                                                      "findings": {"review": {"items": {"F1": {"decision": "confirmed"}}}}},
                                                     ensure_ascii=False), encoding="utf-8")
    record = {"id": job_dir.name, "status": "done", "case_id": "OLD_CASE", "patient_id": "P-001", "display_name": "",
              "review": {"status": "reviewed", "by": "r1", "at": "2026-09-30T10:00:00+08:00", "note": "ok", "version": 4},
              "narrative": {"auto": narrative, "edited": "审阅人改写的结论。", "edited_by": "r1", "edited_at": "2026-09-30"}}
    manifest = V.build_manifest(V.load(job_dir), record)
    assert manifest["job"]["display_name"] == "P-001" and manifest["job"]["patient_id"] == "P-001"
    assert manifest["job"]["review"]["status"] == "reviewed" and manifest["job"]["review"]["version"] == 4
    assert manifest["job"]["narrative_edited"] is True
    block = manifest["analysis"]["narrative"]
    assert block["edited"] == "审阅人改写的结论。" and block["zh"] == ["摘要里的最新一句。"]
    assert manifest["analysis"]["findings"]["review"]["items"]["F1"]["decision"] == "confirmed"
    assert manifest["analysis"]["zones"]["schema_version"] == "wss-deploy.zones/v1"
    record["display_name"] = "显示名"
    assert V.build_manifest(V.load(job_dir), record)["job"]["display_name"] == "显示名"
    del record["narrative"], record["review"]
    manifest = V.build_manifest(V.load(job_dir), record)
    assert manifest["job"]["narrative_edited"] is False and manifest["job"]["review"]["status"] == "unreviewed"


def test_model_card_fills_display_name_windows_and_validation(tmp_path, monkeypatch):
    card = {"schema_version": "wss-deploy.model-card/v1", "release_id": "REL", "display_name": "峰值 WSS",
            "validation": {"holdout_n": 34, "fields": {"wss": {"r2_pa": 0.74}}, "source": "release.json"},
            "display_windows": {"wss": [{"id": "low", "label": "低值窗", "range": [0, 1], "provisional": True}]}}

    class Cards:
        @staticmethod
        def load(release_id, release_dir=None):
            return card if release_id == "REL" else None

        @staticmethod
        def all_cards(registry):
            return {"REL": card}
    monkeypatch.setattr(V, "_model_cards_module", lambda: Cards)
    data = V.load(_old_wall(tmp_path))
    manifest = V.build_manifest(data, {"status": "done", "model_release": {"id": "REL"}}, card=V.load_card("REL"))
    wss = manifest["fields"][0]
    assert manifest["result"]["display_name"] == "峰值 WSS" and manifest["model_card"] == card
    assert wss["windows"][0]["id"] == "low" and wss["validation"] == {"holdout_n": 34, "r2_pa": 0.74, "source": "release.json"}
    assert V.all_cards(object(), ["REL", "OTHER"]) == {"REL": card, "OTHER": None}
    monkeypatch.setattr(V, "_model_cards_module", lambda: None)
    assert V.load_card("REL") is None and V.all_cards(None, ["REL"]) == {"REL": None}
    plain = V.build_manifest(data, {"status": "done"}, card=None)
    assert plain["fields"][0]["windows"] == [] and plain["fields"][0]["validation"] is None


def test_cache_is_lru_bounded_and_reparses_after_a_change(tmp_path):
    dirs = [_old_wall(tmp_path / str(i)) for i in range(V.CACHE_JOBS + 2)]
    for job_dir in dirs:
        V.load(job_dir)
    info = V.cache_info()
    assert len(info["reports"]) == V.CACHE_JOBS and str(dirs[0] / "report.html") not in info["reports"]
    first = V.load(dirs[-1])
    assert V.load(dirs[-1]).report is first.report                            # cache hit: same parsed object
    (dirs[-1] / "summary.json").write_text("{}", encoding="utf-8")
    second = V.load(dirs[-1])
    assert second.report is first.report and second.data_version != first.data_version
    path = dirs[-1] / "report.html"
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 5_000_000))
    third = V.load(dirs[-1])
    assert third.report is not first.report and third.data_version != second.data_version


def test_hide_names_scrubs_every_string_but_not_numbers():
    manifest = {"job": {"display_name": "ZHANG_SAN", "case_id": "ZHANG_SAN", "patient_id": "", "source_filename": "ZHANG_SAN.stl"},
                "analysis": {"narrative": {"zh": ["ZHANG_SAN 的主动脉"]}, "peak": {"p99_pa": 17.0}}, "note": "病例 ZHANG_SAN"}
    out, (bookmarks,) = V.hide_names(manifest, {"filename": "input.stl"}, extra=[[{"name": "ZHANG_SAN 瘤颈", "camera": [1, 2]}]])
    text = json.dumps([out, bookmarks], ensure_ascii=False)
    assert "ZHANG_SAN" not in text
    assert out["job"]["display_name"] == out["job"]["case_id"] == out["job"]["source_filename"] == "病例"
    assert out["job"]["patient_id"] == "" and out["analysis"]["peak"]["p99_pa"] == 17.0
    assert bookmarks[0]["name"] == "病例 瘤颈" and bookmarks[0]["camera"] == [1, 2]
    assert manifest["job"]["case_id"] == "ZHANG_SAN"                          # the input is not modified


# ------------------------------------------------------------------------------------------------------ input check
def _ascii_stl(path: Path, triangles) -> None:
    lines = ["solid t"]
    for tri in triangles:
        lines += ["facet normal 0 0 1", "outer loop", *[f"vertex {x} {y} {z}" for x, y, z in tri], "endloop", "endfacet"]
    path.write_text("\n".join(lines + ["endsolid t"]) + "\n", encoding="utf-8")


def test_inputcheck_after_a_failed_input_check_uses_the_uploaded_stl_in_mm(tmp_path):
    job_dir = tmp_path / "20260930_000000_fail"; job_dir.mkdir()
    _ascii_stl(job_dir / "input.stl", [[(0, 0, 0), (0.01, 0, 0), (0, 0.01, 0)], [(0, 0, 0), (0.01, 0, 0), (0, 0, 0.01)]])
    quality = {"schema_version": "wss-deploy.input-quality/v1", "grade": "fail", "checks": [],
               "opening_geometry": [{"opening_index": 0, "equivalent_radius_mm": 0.3}]}
    stage_a = {"stage": "input", "input_check": {"status": "fail", "ok": False, "scale_factor": 1000.0, "clean_stl": None,
                                                 "errors": ["开口数 1 ≠ 5"], "quality": quality},
               "centerline": None, "proposal": None}
    record = {"status": "failed", "stage": "A", "error": {"message": "输入检查未通过", "category": "input_geometry", "retryable": False}}
    payload = V.build_inputcheck(job_dir, record, stage_a, input_filename="input.stl")
    assert payload["status"] == "failed" and payload["error"]["category"] == "input_geometry"
    assert payload["input_check"]["opening_geometry"] == quality["opening_geometry"] and payload["input_check"]["quality"] == quality
    assert payload["openings"] == [] and payload["mapping_proposal"] is None
    mesh = payload["mesh"]
    assert mesh["source"] == "input_stl" and mesh["units"] == "mm" and mesh["n_faces"] == 2
    vertices = np.frombuffer(base64.b64decode(mesh["vertices"]), "<f4").reshape(-1, 3)
    assert np.isclose(vertices.max(), 10.0)                                   # 0.01 m × 1000 → 10 mm
    faces = np.frombuffer(base64.b64decode(mesh["faces"]), "<u4").reshape(-1, 3)
    assert faces.max() < len(vertices)
    assert sorted(p.name for p in job_dir.iterdir()) == ["input.stl"]         # nothing written
    empty = V.build_inputcheck(job_dir, {"status": "failed"}, None, input_filename="../outside.stl")
    assert empty["input_check"] is None and empty["mesh"] is None and empty["openings"] == []


def test_inputcheck_prefers_the_stage_a_preview_and_carries_the_proposal(tmp_path):
    job_dir = tmp_path / "20260930_000000_wait"; job_dir.mkdir()
    preview = {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "faces": [[0, 1, 2]], "method": "vertex_cluster", "display_faces": 1}
    proposal = {"mapping": {"3": "out-le"}, "endpoints": [{"segment_id": 3}], "confidence": 0.9, "preview_polylines": [{"segment_id": 0}]}
    stage_a = {"input_check": {"status": "pass", "quality": {"opening_geometry": []}}, "preview": preview, "proposal": proposal,
               "centerline": {"openings": [{"opening_id": 0, "radius_mm": 9.0, "role": "inlet"}]}}
    payload = V.build_inputcheck(job_dir, {"status": "awaiting_confirmation", "version": 3}, stage_a)
    assert payload["mesh"]["source"] == "stage_a_preview" and payload["mesh"]["n_faces"] == 1
    assert payload["openings"][0]["radius_mm"] == 9.0 and payload["version"] == 3
    assert payload["mapping_proposal"]["mapping"] == {"3": "out-le"} and "preview_polylines" not in payload["mapping_proposal"]
    assert payload["centerline_polylines"] == [{"segment_id": 0}]
