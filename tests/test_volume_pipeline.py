"""Cross-layer volume tests: real portable exporters, synthetic predictions."""
import base64
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wss_deploy import pipeline, volume_geometry, streamlines
from wss_deploy.io_utils import file_sha256
from wss_deploy.jobs import JobManager


def _script_json(html, script_id):
    return json.loads(re.search(r'<script id="' + script_id + r'" type="application/json">(.*?)</script>', html, re.S).group(1))


@pytest.fixture
def volume_run(tmp_path, monkeypatch):
    """Replace expensive/model inputs; keep every exported representation real."""
    vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    internal = np.array([[.2, .2, .2], [.4, .2, .2], [.2, .4, .2]])
    points = np.concatenate([vertices, internal])
    rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    origin = np.array([.2, .3, .4])
    pressure = np.array([10., 11., 12., 13., 100., 200., 300.], np.float32)
    velocity = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]], np.float32)
    # The analysis layer (profiles / findings / trust) reads the atlas columns below and the
    # per-point arc-length features; the stub provides them so the real exporters run end to end.
    atlas = SimpleNamespace(xyz=np.array([[.2, .2, .1], [.2, .2, .4], [.2, .2, .8]]),
                            tangent=np.tile([0., 0., 1.], (3, 1)),
                            semantic_of_segment={0: 0, 1: 1},
                            segments=[{"segment_id": 0, "parent_id": -1, "starts_at_root": True, "length_mm": .7, "s_offset_mm": 0.}],
                            endpoints=lambda: [{"label": "inlet", "center_mm": np.array([.2, .2, .1]), "radius_mm": 1., "segment_id": 0}],
                            col=lambda name: {"segment_id": np.array([0, 0, 0]),
                                              "sample_index": np.arange(3), "radius_mm": np.ones(3),
                                              "s_local_mm": np.array([0., .3, .7]), "s_from_root_mm": np.array([0., .3, .7]),
                                              "curvature_per_mm": np.zeros(3)}[name])
    captured = {}
    case = {"n_wall": len(vertices), "wall_coords_raw": points, "frame_rotation": rotation,
            "dist_to_wall_mm": np.array([0., 0., 0., 0., .2, .2, .2], np.float32)}
    aux = {"geom": {"segment_id": np.array([0, 0, 1, 1, 0, 1, 1]), "frame_origin_mm": origin,
                    "s_local_mm": np.array([0., .1, .2, .3, .2, .4, .6], np.float32),
                    "s_from_root_mm": np.array([0., .1, .2, .3, .2, .4, .6], np.float32),
                    "radius_mm": np.ones(7, np.float32)},
           "diag": {"geometry_version": "synthetic-contract/v1"},
           "closed_surface": {"vertices": vertices, "faces": faces}}

    def geometry(wall, smoothed, got_faces, got_atlas, input_features, **kwargs):
        np.testing.assert_array_equal(wall, vertices)
        np.testing.assert_array_equal(got_faces, faces)
        assert got_atlas is atlas
        captured["geometry_kwargs"] = kwargs
        return case, aux

    def trace(got_points, got_velocity, seeds, inside):
        np.testing.assert_array_equal(got_points, internal)
        np.testing.assert_array_equal(got_velocity, velocity)
        assert inside(internal).all()
        return [{"points": internal, "speed_m_s": np.linalg.norm(velocity, axis=1)}], {"count": 1, "kind": "steady"}

    monkeypatch.setattr(pipeline.CL, "load_vessel_geom_atlas", lambda *_: atlas)
    monkeypatch.setattr(pipeline.CL, "apply_mapping", lambda a, mapping: a)
    monkeypatch.setattr("wss_features.stl.load_stl", lambda *_: (vertices.copy(), faces.copy()))
    monkeypatch.setattr(pipeline.G, "smooth_and_resample", lambda *_a, **_kw: (vertices.copy(), vertices.copy()))
    monkeypatch.setattr(volume_geometry, "build_volume_case", geometry)
    monkeypatch.setattr(streamlines, "centerline_seeds", lambda a: a.xyz[:1])
    monkeypatch.setattr(streamlines, "integrate_streamlines", trace)
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    (release_dir / "MANIFEST.sha256").write_text("synthetic verified test release\n")
    prediction = {"pressure_pa": pressure, "velocity_m_s": velocity,
                  "seed_pressure_pa": np.tile(pressure, (3, 1)),
                  "seed_velocity_m_s": np.tile(velocity, (3, 1, 1)),
                  "seconds_per_model": [0.] * 6, "device": "cpu", "gpu": None}
    release = SimpleNamespace(dir=release_dir, name="PF6_VF6-test", input_features=["x", "y", "z"],
                              info={"target": "pressure_velocity", "release": "PF6_VF6-test"},
                              predict=lambda _: prediction)

    def prepare(name="run", *, units="mm", clean=b"clean-test-STL"):
        job_dir = tmp_path / name
        job_dir.mkdir()
        (job_dir / "input.stl").write_bytes(b"input-stl")
        clean_path = job_dir / "input_clean_mm.stl"
        clean_path.write_bytes(clean)
        stage_a = {"stage": "A", "input_sha256": "a" * 64, "timing_s": {"ingest": .1},
                   "input_check": {"ok": True, "clean_stl": str(clean_path), "resolved_units": units},
                   "centerline": {"hard_pass": True}, "proposal": {"flags": []}}
        (job_dir / "stage_a.json").write_text(json.dumps(stage_a))
        (job_dir / "job.json").write_text(json.dumps({"owner": "session-owner-secret", "patient_id": "anon-1",
            "mapping_history": [{"source": "manual", "confirmed": {"1": "out-le"}, "acknowledged": True}]}))
        return job_dir

    def run(name="run", **kwargs):
        job_dir = prepare(name, **kwargs)
        meta = pipeline.stage_b(job_dir, {"1": "out-le"}, release, confirmed=True, case_id="anon-case")
        return job_dir, meta

    return SimpleNamespace(run=run, prepare=prepare, release=release, captured=captured,
                           pressure=pressure, velocity=velocity, internal=internal, points=points,
                           rotation=rotation, origin=origin, prediction=prediction)


def test_volume_pipeline_exports_separate_domains_and_real_report(volume_run):
    import pyvista as pv

    run = volume_run
    directory, meta = run.run()
    assert set(meta["fields"]) == {"pressure", "velocity"}
    assert meta["fields"]["pressure"]["location"] == "wall_and_interior"
    assert meta["fields"]["pressure"]["reference"] == "volume_mean_relative"
    assert meta["fields"]["velocity"]["location"] == "interior"
    assert meta["fields"]["velocity"]["units"] == "m/s"
    assert meta["fields"]["velocity"]["frame"] == "world"
    assert meta["volume_statistics"]["pressure_wall_pa"]["count"] == 4
    assert meta["volume_statistics"]["pressure_wall_pa"]["mean"] == pytest.approx(11.5)
    assert meta["volume_statistics"]["pressure_interior_pa"]["count"] == 3
    assert meta["volume_statistics"]["pressure_interior_pa"]["mean"] == pytest.approx(200.)
    assert meta["volume_statistics"]["speed_m_s"]["mean"] == pytest.approx(2.)
    np.testing.assert_allclose(meta["frame_transform"]["rotation"], run.rotation)
    np.testing.assert_allclose(meta["frame_transform"]["origin_mm"], run.origin)
    # analysis layer is present and consistent with the synthetic predictions
    assert meta["profiles"]["kind"] == "volume" and meta["profiles"]["branches"][0]["segment_id"] == 0
    kinds = [item["kind"] for item in meta["findings"]["items"]]
    assert kinds[:2] == ["max_speed", "min_pressure"]
    assert next(i for i in meta["findings"]["items"] if i["kind"] == "max_speed")["value"] == pytest.approx(3.)
    assert next(i for i in meta["findings"]["items"] if i["kind"] == "min_pressure")["value"] == pytest.approx(100.)
    assert set(meta["trust"]["fractions"]) == {"interpolation_uncovered", "geometry_out_of_range", "low_sample_support", "near_opening"}
    assert meta["reference_assessment"]["status"] == "unknown"
    assert run.captured["geometry_kwargs"]["case_name"] == "input-" + "a" * 24
    with np.load(directory / "field.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["pressure_pa"], run.pressure)
        np.testing.assert_array_equal(data["velocity_m_s"], run.velocity)
        np.testing.assert_array_equal(data["internal_pts"], run.internal.astype(np.float32))
        assert data["point_kind"].tolist() == [0] * 4 + [1] * 3
        assert data["seed_pressure_pa"].shape == (3, 7)
        assert data["seed_velocity_m_s"].shape == (3, 3, 3)
        assert "wss_pa" not in data
    csv = np.genfromtxt(directory / "points_volume.csv", delimiter=",", names=True)
    assert len(csv) == 3
    np.testing.assert_array_equal(csv["pressure_pa"], run.pressure[4:])
    np.testing.assert_array_equal(csv["speed_m_s"], [1., 2., 3.])
    volume = pv.read(directory / "volume_fields.vtp")
    assert volume.n_points == 3
    np.testing.assert_array_equal(volume["velocity_m_s"], run.velocity)
    wall = pv.read(directory / "wall_pressure.vtp")
    assert wall.n_points == 4 and wall.n_cells == 4
    assert np.all((wall["pressure_pa"] >= 10.) & (wall["pressure_pa"] <= 13.))
    assert "velocity_m_s" not in wall.point_data
    assert pv.read(directory / "streamlines.vtp").n_lines == 1
    html = (directory / "report.html").read_text()
    assert "session-owner-secret" not in html
    report_meta = _script_json(html, "wss-report-meta")
    assert report_meta["run_identity"] == meta["run_identity"]
    assert report_meta["timing_s"] == meta["timing_s"]
    arrays = _script_json(html, "volume-arrays")
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(arrays["pressure_pa"]), np.float32), run.pressure[4:])
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(arrays["velocity_m_s"]), np.float32).reshape(-1, 3), run.velocity)
    assert "slice-position" in html and "auto-presets" in html
    manifest = json.loads((directory / "run_manifest.json").read_text())
    for name in ("summary.json", "report.html", "field.npz", "volume_fields.vtp", "wall_pressure.vtp", "points_volume.csv", "streamlines.vtp"):
        assert manifest["outputs"][name]["present"] is True
        assert manifest["outputs"][name]["sha256"] == file_sha256(directory / name)
    assert manifest["provenance"]["frame_transform"] == meta["frame_transform"]


def test_volume_identity_tracks_units_and_clean_geometry(volume_run):
    directory, baseline = volume_run.run("mm")
    _, repeat = volume_run.run("same-input-other-job")
    _, converted = volume_run.run("cm", units="cm")
    _, repaired = volume_run.run("repaired", clean=b"changed-clean-STL")
    assert baseline["run_identity"] == repeat["run_identity"]
    assert len({baseline["run_identity"], converted["run_identity"], repaired["run_identity"]}) == 3
    assert baseline["run_parameters"]["resolved_units"] == "mm"
    assert baseline["run_parameters"]["clean_stl_sha256"] == file_sha256(directory / "input_clean_mm.stl")


@pytest.mark.parametrize("invalid", ["wall_velocity", "nonfinite"])
def test_volume_rejects_invalid_model_outputs_before_export(volume_run, invalid):
    directory = volume_run.prepare()
    if invalid == "wall_velocity":
        volume_run.prediction["velocity_m_s"] = np.zeros((7, 3), np.float32)
    else:
        volume_run.prediction["pressure_pa"][0] = np.nan
    with pytest.raises(ValueError, match="不匹配|非有限"):
        pipeline.stage_b(directory, {"1": "out-le"}, volume_run.release, confirmed=True)
    assert not (directory / "field.npz").exists()
    assert not (directory / "report.html").exists()


def test_wss_release_keeps_existing_stage_b_dispatch(tmp_path, monkeypatch):
    def wrong_dispatch(*_args, **_kwargs):
        pytest.fail("a WSS release was routed to volume inference")
    monkeypatch.setattr("wss_deploy.volume_pipeline.stage_b_volume", wrong_dispatch)
    (tmp_path / "stage_a.json").write_text(json.dumps({"stage": "A", "input_check": {"ok": False}, "timing_s": {}}))
    release = SimpleNamespace(info={"target": "wall shear stress magnitude (Pa)"})
    with pytest.raises(ValueError, match="输入检查"):
        pipeline.stage_b(tmp_path, {}, release, confirmed=True)


def test_cross_release_rerun_resets_five_seed_wss_subset_for_three_seed_volume(tmp_path):
    calls = []
    class Descriptor:
        def __init__(self, name, count):
            self.name, self.fingerprint, self.count = name, name + "-fingerprint", count
            self.info = {}
        def public(self):
            return {"id": self.name, "release": self.name, "fingerprint": self.fingerprint, "models_count": self.count}
    class Registry:
        records = {"wss-five": Descriptor("wss-five", 5), "volume-three": Descriptor("volume-three", 3)}
        def describe(self, name=None): return self.records[name or "wss-five"]
        def load(self, name, **kwargs):
            calls.append((name, kwargs)); return self.describe(name)
    def stage_a(_input, directory, **_kwargs):
        directory = Path(directory)
        (directory / "centerline").mkdir()
        clean = directory / "input_clean_mm.stl"; clean.write_bytes(b"clean")
        return {"stage": "A", "input_sha256": "a" * 64, "input_check": {"status": "pass", "ok": True, "clean_stl": str(clean)},
                "centerline": {"hard_pass": True}, "proposal": {"auto_ok": False, "mapping": {
                    "1": "out-le", "2": "out-li", "3": "out-re", "4": "out-ri"}}, "timing_s": {}}
    def stage_b(directory, _mapping, release, **kwargs):
        result = {"release": release.name, "timing_s": {}, "run_identity": release.name}
        (Path(directory) / "summary.json").write_text(json.dumps(result))
        return result
    manager = JobManager(tmp_path, registry=Registry(), stage_a_fn=stage_a, stage_b_fn=stage_b, mapping_validator=lambda *_: [])
    original = manager.create("owner", content=b"STL", filename="case.stl", release_id="wss-five", seed_count=5, device="cpu", threads=2)
    manager.run_next()
    pending = manager.get(original["id"], "owner")
    manager.confirm(original["id"], "owner", {"version": pending["version"], "stage": "A",
                    "mapping": pending["a"]["proposal"]["mapping"], "acknowledged": True})
    manager.run_next()
    source = manager.get(original["id"], "owner")
    old_summary = (tmp_path / original["id"] / "summary.json").read_bytes()
    new = manager.rerun(source["id"], "owner", {"version": source["version"], "release_id": "volume-three"})
    assert new["compute"] == {"device": "cpu", "seed_count": None, "threads": 2}
    assert source["compute"]["seed_count"] == 5
    manager.run_next()
    assert manager.get(new["id"], "owner")["status"] == "done"
    assert calls == [("wss-five", {"device": "cpu", "seed_count": 5}),
                     ("volume-three", {"device": "cpu", "seed_count": None})]
    assert (tmp_path / original["id"] / "summary.json").read_bytes() == old_summary
    assert (tmp_path / new["id"] / "input_clean_mm.stl").read_bytes() == b"clean"
    manager.close()
