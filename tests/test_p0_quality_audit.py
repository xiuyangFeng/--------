import json

import numpy as np

from wss_deploy.quality import ensemble_quality
from wss_deploy.infer import Release
from wss_deploy.schema import build_run_manifest
from wss_deploy.pipeline import _archive_previous_run


def test_quality_summary_hides_pointwise_seed_spread():
    got = ensemble_quality(np.array([2.0, 4.0, 1.0]), np.array([0.05, 0.1, 0.02]), seed_count=5)
    assert got["quality"]["level"] == "good"
    assert "seed_sd_pa" not in got["quality"]
    assert got["audit"]["seed_count"] == 5
    assert got["audit"]["spread_pa"]["p90"] > 0


def test_quality_summary_requires_review_for_large_spread():
    got = ensemble_quality(np.array([1.0, 2.0]), np.array([0.8, 1.0]), seed_count=5)
    assert got["quality"]["level"] in {"review", "poor"}
    assert got["quality"]["reasons"]


def test_release_manifest_hash_parser_only_accepts_checkpoint_records(tmp_path):
    (tmp_path / "MANIFEST.sha256").write_text(
        "a" * 64 + "  models/fold/ckpt_best.pt  4\n" +
        "b" * 64 + "  models/fold/config.json  4\n", encoding="utf-8")
    release = Release.__new__(Release)
    release.dir = tmp_path
    assert release._manifest_records() == {"models/fold/ckpt_best.pt": "a" * 64}


def test_manifest_contains_code_runtime_and_audit_without_owner(tmp_path):
    manifest = build_run_manifest({
        "input_sha256": "a" * 64,
        "model_release": {"release": "demo"},
        "audit": {"mapping_history": [{"source": "manual_override", "confirmed": {"1": "out-le"}}]},
    }, tmp_path)
    assert "provenance" in manifest and "code_runtime" in manifest["provenance"]
    assert "source_hash" in manifest["provenance"]["code_runtime"]
    assert manifest["audit"]["mapping_history"][0]["source"] == "manual_override"
    assert "owner" not in json.dumps(manifest)


def test_completed_run_is_archived_before_recompute(tmp_path):
    (tmp_path / "summary.json").write_text("old", encoding="utf-8")
    (tmp_path / "report.html").write_text("old report", encoding="utf-8")
    archive = _archive_previous_run(tmp_path)
    assert archive and (tmp_path / archive / "summary.json").read_text() == "old"
    assert (tmp_path / archive / "report.html").is_file()
