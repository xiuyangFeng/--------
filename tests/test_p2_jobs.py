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


# ---------------------------------------------------------------- §19.2: cycle + findings_top in the job record
from wss_deploy.jobs import CYCLE_RECORD_LIMIT, cycle_record, findings_top, release_short  # noqa: E402

_CYCLE = {"definition": {"period_s": 0.8}, "fields": {"tawss": {"mean": 0.66, "area_frac": {"low": 0.65}, "per_branch": {"主动脉": {"mean": 0.3}}},
                                                     "osi": {"mean": 0.13, "area_frac": {"above_t0": 0.57}}},
          "stagnation": {"area_frac": 0.46, "area_mm2": 155.0, "per_branch": {"主动脉": {"area_mm2": 150.0}}}}
_FINDINGS = {"items": [{"id": f"F{i}", "kind": "high_wss_cluster", "label": f"区 {i}", "value": float(i), "point_indices": list(range(50))}
                       for i in range(1, 8)]}


def test_finished_job_records_cycle_and_top_findings(tmp_path):
    from tests._c_helpers import MAPPING, manager as c_manager

    def stage_b(*_a, **_k):
        return {"peak": {"p99_pa": 17.0}, "fields": {"wss": {}, "tawss": {}, "osi": {}}, "timing_s": {}, "run_identity": "r" * 64,
                "cycle": _CYCLE, "findings": _FINDINGS}
    mgr = c_manager(tmp_path, stage_b=stage_b)
    job = mgr.create("owner", content=b"stl", filename="m1.stl")
    mgr.run_next()
    waiting = mgr.get(job["id"], "owner")
    mgr.confirm(job["id"], "owner", {"version": waiting["version"], "mapping": dict(MAPPING), "acknowledged": True})
    mgr.run_next()
    done = mgr.get(job["id"], "owner")
    assert done["status"] == "done" and done["summary"]["cycle"] == _CYCLE
    assert [item["id"] for item in done["summary"]["findings_top"]] == ["F1", "F2", "F3", "F4", "F5"]
    assert all("point_indices" not in item for item in done["summary"]["findings_top"])
    light = mgr.list("owner")[0]
    assert light["has_cycle"] is True and light["family_label"] == "WSS + TAWSS + OSI" and light["release_short"] == "REL_A"
    assert done["has_cycle"] is True and done["family_label"] == "WSS + TAWSS + OSI"
    saved = json.loads((tmp_path / job["id"] / "job.json").read_text())
    assert saved["summary"]["cycle"] == _CYCLE and len(saved["summary"]["findings_top"]) == 5
    mgr.close()


def test_old_finished_records_are_backfilled_in_memory_only(tmp_path):
    job_id = "20260920_old"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    record = {"id": job_id, "owner": "owner", "status": "done", "stage": "B", "version": 7, "events": [],
              "created_at": "2026-09-20T00:00:00+08:00", "summary": {"peak": {"p99_pa": 1.0}, "fields": {"wss": {}, "tawss": {}, "osi": {}}},
              "model_release": {"id": "M1_3head_3seed_20260922", "contract": {"protocol": "single_frame_wss_cycle_multi"}}}
    (job_dir / "job.json").write_text(json.dumps(record))
    morphology = {"aorta": {"max": {"max_diameter_mm": 72.2, "polygon_world": [[0, 0, 0]]}, "stations": {"s": [1, 2]}}, "branches": [], "method": {}}
    (job_dir / "summary.json").write_text(json.dumps({"cycle": _CYCLE, "findings": _FINDINGS, "morphology": morphology}))
    before = (job_dir / "job.json").read_bytes()
    mgr = JobManager(tmp_path, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})
    summary = mgr.jobs[job_id]["summary"]
    assert summary["cycle"] == _CYCLE and [item["id"] for item in summary["findings_top"]] == ["F1", "F2", "F3", "F4", "F5"]
    assert summary["morphology"]["aorta"]["max"] == {"max_diameter_mm": 72.2}
    light = mgr.list("owner")[0]
    assert light["has_cycle"] is True and light["max_diameter_mm"] == 72.2 and light["release_short"] == "M1_3head"
    assert light["family"] == "wall" and light["family_label"] == "WSS + TAWSS + OSI"
    assert (job_dir / "job.json").read_bytes() == before            # not rewritten on load
    # a finished X5D job without a cycle gets an explicit null and still loads
    (job_dir / "summary.json").write_text(json.dumps({"findings": {"items": []}}))
    record["summary"] = {"peak": {"p99_pa": 1.0}, "fields": {"wss": {}}}
    record["model_release"] = {"id": "X5D_v51_5seed_20260916", "contract": {"protocol": "single_frame_wss"}}
    (job_dir / "job.json").write_text(json.dumps(record))
    mgr = JobManager(tmp_path, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})
    summary = mgr.jobs[job_id]["summary"]
    assert summary["cycle"] is None and summary["findings_top"] == [] and "morphology" not in summary
    assert mgr.list("owner")[0]["has_cycle"] is False and mgr.list("owner")[0]["family_label"] == "壁面 WSS"


def test_cycle_record_is_compacted_when_large_and_release_short_names():
    big = json.loads(json.dumps(_CYCLE))
    big["fields"]["tawss"]["per_branch"] = {f"b{i}": {"mean": i, "text": "x" * 200} for i in range(100)}
    compact = cycle_record(big)
    assert compact["compact"] is True and "per_branch" not in compact["fields"]["tawss"] and "per_branch" not in compact["stagnation"]
    assert compact["fields"]["tawss"]["mean"] == 0.66 and len(json.dumps(compact)) < CYCLE_RECORD_LIMIT
    assert cycle_record(None) is None and cycle_record({}) is None and findings_top(None) == [] and findings_top({"findings": {}}) == []
    assert [release_short({"id": rid}) for rid in ("X5D_v51_5seed_20260916", "M1_3head_3seed_20260922", "PF6_VF6_peak_3seed_20260920")] == \
        ["X5D_v51", "M1_3head", "PF6_VF6_peak"]
