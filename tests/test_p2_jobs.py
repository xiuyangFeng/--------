import json

import pytest

from wss_deploy.jobs import JobError, JobManager


def _manager(tmp_path):
    return JobManager(tmp_path, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})


def test_case_metadata_is_persisted_and_owner_scoped(tmp_path):
    manager = _manager(tmp_path)
    job = manager.create(
        "alice", content=b"solid", filename="scan.stl", case_id="case-a",
        patient_id="anon-17", scan_label="before", scan_date="2026-09-19",
        tags=["AAA", "review", "aaa"], notes="check inlet",
    )
    assert job["patient_id"] == "anon-17"
    assert job["tags"] == ["AAA", "review"]
    saved = json.loads((tmp_path / job["id"] / "job.json").read_text())
    assert saved["scan_label"] == "before"
    assert saved["notes"] == "check inlet"
    assert manager.list("bob") == []
    assert manager.query("alice", patient_id="anon-17")["total"] == 1


def test_query_filters_searches_and_paginates_without_changing_legacy_list(tmp_path):
    manager = _manager(tmp_path)
    for index, (patient, tag) in enumerate((("p1", "follow-up"), ("p2", "baseline"), ("p1", "follow-up"))):
        manager.create("owner", content=b"stl", filename=f"scan-{index}.stl", case_id=f"case-{index}",
                       patient_id=patient, tags=[tag], notes=f"note {index}")
    assert len(manager.list("owner")) == 3
    page = manager.query("owner", patient_id="p1", tag="FOLLOW-UP", page=1, page_size=1)
    assert page["total"] == 2 and page["page"] == 1 and page["page_size"] == 1
    assert len(page["jobs"]) == 1
    assert manager.query("owner", q="note 1")["jobs"][0]["case_id"] == "case-1"
    assert manager.query("other", q="case")["total"] == 0


def test_query_rejects_invalid_filters(tmp_path):
    manager = _manager(tmp_path)
    with pytest.raises(JobError):
        manager.query("owner", status="not-a-status")
    with pytest.raises(JobError):
        manager.query("owner", page=0)
    with pytest.raises(JobError):
        manager.query("owner", page_size=101)


def test_batch_upload_keeps_successes_when_one_item_is_invalid(tmp_path):
    manager = _manager(tmp_path)
    result = manager.create_batch("owner", [
        {"content": b"one", "filename": "one.stl", "patient_id": "p1"},
        {"content": b"two", "filename": "bad.txt", "patient_id": "p2"},
        {"content": b"three", "filename": "three.stl", "tags": ["ok"]},
    ])
    assert result["created_count"] == 2 and result["failed_count"] == 1
    assert len(result["jobs"]) == 2
    assert all(item["batch_id"] == result["batch_id"] for item in result["jobs"])
    assert result["results"][1]["error"]["status"] == 400
    assert manager.query("owner")["total"] == 2


def test_batch_limit_and_metadata_validation(tmp_path):
    manager = _manager(tmp_path)
    with pytest.raises(JobError):
        manager.create_batch("owner", [{"content": b"x", "filename": "x.stl"}] * 21)
    with pytest.raises(JobError):
        manager.create("owner", content=b"x", filename="x.stl", scan_date="19-09-2026")
    with pytest.raises(JobError):
        manager.create("owner", content=b"x", filename="x.stl", notes="\x00")


def test_rerun_retains_case_metadata(tmp_path):
    manager = JobManager(
        tmp_path,
        stage_a_fn=lambda *_a, **_k: {
            "input_sha256": "a" * 64,
            "input_check": {"status": "pass", "ok": True},
            "centerline": {"hard_pass": True},
            "proposal": {"auto_ok": False, "mapping": {"1": "out-le", "2": "out-li", "3": "out-re", "4": "out-ri"}},
        },
        stage_b_fn=lambda *_a, **_k: {"timing_s": {}, "run_identity": "id"},
        mapping_validator=lambda *_a: [],
    )
    source = manager.create("owner", content=b"stl", filename="scan.stl", patient_id="anon-1", tags=["before"])
    manager.run_next()
    waiting = manager.get(source["id"], "owner")
    manager.confirm(source["id"], "owner", {"version": waiting["version"], "stage": "A",
                       "mapping": waiting["a"]["proposal"]["mapping"], "acknowledged": True})
    manager.run_next()
    done = manager.get(source["id"], "owner")
    rerun = manager.rerun(source["id"], "owner", {"version": done["version"]})
    assert rerun["patient_id"] == "anon-1"
    assert rerun["tags"] == ["before"]


def test_restore_old_summary_release_binding_before_retry(tmp_path):
    job_id = "20260919_legacy"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "job.json").write_text(json.dumps({
        "id": job_id, "owner": "owner", "status": "failed", "stage": "B",
        "version": 4, "params": {"units": "mm", "remove_fragments": False, "inlet": None},
        "events": [], "created_at": "2026-09-19T00:00:00+08:00",
    }))
    (job_dir / "summary.json").write_text(json.dumps({
        "release": "old-release", "release_hash": "a" * 64,
    }))

    class Descriptor:
        def public(self):
            return {"id": "old-release", "release": "old-release", "fingerprint": "b" * 64}

    class Registry:
        def __init__(self): self.calls = []
        def restore_legacy_binding(self, release_id, **kwargs):
            self.calls.append((release_id, kwargs)); return Descriptor()

    registry = Registry()
    manager = JobManager(tmp_path, registry=registry,
                         stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})
    job = manager.jobs[job_id]
    assert job["model_release"]["id"] == "old-release"
    assert job["model_release"]["fingerprint"] == "b" * 64
    assert job["legacy_release_migration"]["manifest_sha256"] == "a" * 64
    assert job["version"] == 5 and registry.calls == [("old-release", {"manifest_sha256": "a" * 64, "release_json_sha256": None})]
