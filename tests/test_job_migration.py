import json

from wss_deploy.jobs import JobManager


class _Descriptor:
    def public(self):
        return {
            "id": "legacy-release",
            "release": "legacy-release",
            "fingerprint": "f" * 64,
            "contract": {"protocol": "single_frame_wss"},
        }


class _Registry:
    def __init__(self):
        self.calls = []

    def restore_legacy_binding(self, release_id, *, manifest_sha256, release_json_sha256=None):
        self.calls.append((release_id, manifest_sha256, release_json_sha256))
        return _Descriptor()


def test_reload_binds_legacy_release_from_summary_and_persists_migration(tmp_path):
    job_dir = tmp_path / "legacy-job"
    job_dir.mkdir()
    job = {
        "id": "legacy-job",
        "owner": "owner",
        "case_id": "case",
        "source_filename": "case.stl",
        "filename": "input.stl",
        "status": "failed",
        "stage": "B",
        "version": 4,
        "params": {"units": "mm", "remove_fragments": False, "inlet": None},
        "mapping": {"1": "out-le", "2": "out-li", "3": "out-re", "4": "out-ri"},
        "a": {"input_check": {"status": "pass", "ok": True}, "centerline": {"hard_pass": True}},
        "events": [],
    }
    (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    (job_dir / "summary.json").write_text(json.dumps({
        "release": "legacy-release",
        "release_hash": "a" * 64,
    }), encoding="utf-8")

    registry = _Registry()
    manager = JobManager(job_dir.parent, registry=registry, legacy_owner="owner")

    restored = manager.jobs["legacy-job"]
    assert registry.calls == [("legacy-release", "a" * 64, None)]
    assert restored["model_release"]["id"] == "legacy-release"
    assert restored["model_release"]["fingerprint"] == "f" * 64
    assert restored["model_release"]["legacy_manifest_sha256"] == "a" * 64
    assert restored["events"][-1]["action"] == "legacy_release_bound"
    persisted = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert persisted["model_release"]["fingerprint"] == "f" * 64
    assert persisted["events"][-1]["action"] == "legacy_release_bound"

