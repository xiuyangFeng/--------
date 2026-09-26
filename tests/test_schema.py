import json
from pathlib import Path

from wss_deploy.schema import (
    FIELD_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    build_results,
    field_descriptor,
    model_release_metadata,
    single_frame_time_axis,
    write_run_manifest,
)
from wss_deploy.infer import Release


def test_generic_single_frame_result_keeps_wss_compatibility():
    axis = single_frame_time_axis({"target": "peak_systole", "step": 1162, "time_s": 0.21})
    field = field_descriptor("wss", label="壁面切应力", units="Pa", location="wall",
                             array_key="wss_pa", statistics_key="wss")
    result = build_results(time_axis=axis, fields={"wss": field},
                           statistics={"wss": {"p99": 3.0}},
                           compatibility={"peak": {"p99_pa": 3.0}})
    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["time_axis"][0]["step"] == 1162
    assert result["fields"]["wss"]["schema_version"] == FIELD_SCHEMA_VERSION
    assert result["compatibility"]["peak"]["p99_pa"] == 3.0


def test_model_release_metadata_records_release_and_checkpoint_hashes(tmp_path):
    release = tmp_path / "release"
    (release / "models" / "Demo_s1").mkdir(parents=True)
    (release / "release.json").write_text(json.dumps({
        "release": "demo-v2", "git_commit": "abc123", "target": "wall field",
        "models": [{"seed": 1, "path": "models/Demo_s1"}],
    }), encoding="utf-8")
    checkpoint = release / "models" / "Demo_s1" / "ckpt_best.pt"
    checkpoint.write_bytes(b"weights")
    from wss_deploy.io_utils import file_sha256
    (release / "MANIFEST.sha256").write_text(
        f"{file_sha256(checkpoint)}  models/Demo_s1/ckpt_best.pt  {checkpoint.stat().st_size}\n",
        encoding="utf-8")
    got = model_release_metadata(release)
    assert got["release"] == "demo-v2"
    assert got["git_commit"] == "abc123"
    assert got["weights"][0]["seed"] == "1"
    assert got["weights"][0]["sha256"] == file_sha256(checkpoint)


def test_release_can_declare_a_new_weight_family_without_x5d_paths():
    release = Release.__new__(Release)
    release.info = {"models": [{"seed": "fold-a", "path": "models/new_family_a"},
                                {"seed": "fold-b", "path": "models/new_family_b"}]}
    assert release._model_specs((1234,)) == [
        {"seed": "fold-a", "path": "models/new_family_a"},
        {"seed": "fold-b", "path": "models/new_family_b"},
    ]


def test_run_manifest_is_portable_and_hashes_outputs(tmp_path):
    (tmp_path / "input.stl").write_bytes(b"input")
    (tmp_path / "input_clean_mm.stl").write_bytes(b"clean")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    meta = {
        "case_id": "case-1", "created_at": "2026-09-18 17:00:00",
        "input_sha256": "a" * 64,
        "input_check": {"clean_stl": str(tmp_path / "input_clean_mm.stl"), "resolved_units": "mm"},
        "model_release": {"release": "demo-v2"}, "time_axis": [{"index": 0}],
        "fields": {"wss": {"units": "Pa"}}, "results": {"schema_version": "x"},
        "mapping": {"1": "out-le"}, "run_parameters": {"spacing_mm": 0.5},
        "device": "cpu", "timing_s": {"total": 1.0}, "release_hash": "b" * 64,
    }
    manifest = write_run_manifest(tmp_path, meta, outputs=("input.stl", "summary.json", "missing.bin"))
    assert manifest["input"]["clean_stl"] == "input_clean_mm.stl"
    assert manifest["outputs"]["summary.json"]["sha256"]
    assert manifest["outputs"]["missing.bin"]["present"] is False
    saved = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == "wss-deploy.run-manifest/v1"


def test_code_provenance_records_dirty_flag_version_and_source_hash(tmp_path, monkeypatch):
    """v0.14 (J4): git describe --dirty, deploy version and a hash over the code actually used."""
    import wss_deploy
    from wss_deploy import schema as S
    answers = {("rev-parse", "HEAD"): "abc123" * 6 + "abcd", ("describe", "--always", "--dirty", "--long"): "e8d3b69-dirty"}
    monkeypatch.setattr(S, "_git", lambda root, *args: answers.get(args))
    code = S.code_provenance(refresh=True)
    assert code["git_dirty"] is True and code["git_describe"] == "e8d3b69-dirty" and code["deploy_version"] == wss_deploy.__version__
    assert len(code["source_hash"]) == 64 and "dataset" in code["training_modules"] and "evaluate" in code["training_modules"]
    names = {p.name for p in S.code_source_files()}
    assert {"jobs.py", "schema.py", "families.py", "dataset.py", "evaluate.py"} <= names
    answers[("describe", "--always", "--dirty", "--long")] = "e8d3b69"
    assert S.code_provenance(refresh=True)["git_dirty"] is False
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    manifest = write_run_manifest(tmp_path, {"input_sha256": "a" * 64}, outputs=("summary.json",))
    assert manifest["deploy_version"] == wss_deploy.__version__ and manifest["git_dirty"] is False
    assert manifest["deploy_version_source"] == "manifest_writer" and manifest["analysis_version"] is None
    runtime = manifest["provenance"]["code_runtime"]
    assert runtime["git_describe"] == "e8d3b69" and runtime["source_hash"] == code["source_hash"] and runtime["git_commit"]
    computed = write_run_manifest(tmp_path, {"deploy_version": "0.13.0", "git_dirty": True, "analysis_version": S.ANALYSIS_VERSION})
    assert computed["deploy_version"] == "0.13.0" and computed["git_dirty"] is True and computed["deploy_version_source"] == "summary"
    assert S.summary_provenance()["analysis_version"] == S.ANALYSIS_VERSION == "2026-09-24"
    S.code_provenance(refresh=True)       # leave no monkeypatched values in the cache
    monkeypatch.undo()
    S.code_provenance(refresh=True)


def test_old_manifest_without_provenance_keys_still_reads(tmp_path):
    from wss_deploy.jobs import JobManager
    job_dir = tmp_path / "20260101_000000_old"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({"id": job_dir.name, "owner": "o", "status": "done"}), encoding="utf-8")
    (job_dir / "run_manifest.json").write_text(json.dumps({"schema_version": "wss-deploy.run-manifest/v1",
                                                             "provenance": {"code_runtime": {"git_commit": "x", "source_hash": "x"}}}), encoding="utf-8")
    manager = JobManager(tmp_path, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})
    assert manager.get(job_dir.name, "o")["status"] == "done"
