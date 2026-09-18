from pathlib import Path

from wss_deploy.ingest import _unit_confidence
from wss_deploy.jobs import JobManager


def _stage_a(confidence=0.99, required=False):
    return {
        "stage": "A",
        "input_check": {"status": "pass", "ok": True},
        "centerline": {"hard_pass": True},
        "proposal": {
            "auto_ok": True,
            "confirmation_required": required,
            "confidence": confidence,
            "mapping": {"1": "out-li", "2": "out-le", "3": "out-ri", "4": "out-re"},
            "flags": [],
        },
        "timing_s": {},
    }


def test_auto_unit_confidence_requires_unique_candidate():
    high, reasons = _unit_confidence(239.0, "auto", "mm")
    assert high >= 0.95
    assert reasons == []
    low, reasons = _unit_confidence(100.0, "auto", "mm")
    assert low < 0.95
    assert reasons
    explicit, reasons = _unit_confidence(100.0, "mm", "mm")
    assert explicit == 1.0
    assert reasons == []


def test_high_confidence_proposal_auto_queues_stage_b(tmp_path):
    seen = []

    def stage_a(path, job_dir, **kwargs):
        return _stage_a()

    def stage_b(job_dir, mapping, release, **kwargs):
        seen.append(mapping)
        return {"peak": {}, "wss_field_pa": {}, "timing_s": {}, "flags": []}

    manager = JobManager(tmp_path, stage_a_fn=stage_a, stage_b_fn=stage_b,
                         mapping_validator=lambda *_: [])
    job = manager.create("owner", content=b"stl", filename="input.stl")
    assert manager.run_next()
    after_a = manager.get(job["id"], "owner")
    assert after_a["status"] == "queued"
    assert after_a["stage"] == "B"
    assert after_a["mapping_history"][-1]["source"] == "automatic_high_confidence"
    assert manager.run_next()
    assert manager.get(job["id"], "owner")["status"] == "done"
    assert seen


def test_low_confidence_proposal_still_waits_for_confirmation(tmp_path):
    manager = JobManager(tmp_path, stage_a_fn=lambda *_args, **_kw: _stage_a(0.7, True),
                         stage_b_fn=lambda *_args, **_kw: {},
                         mapping_validator=lambda *_: [])
    job = manager.create("owner", content=b"stl", filename="input.stl")
    manager.run_next()
    after_a = manager.get(job["id"], "owner")
    assert after_a["status"] == "awaiting_confirmation"
    assert after_a["stage"] == "A"


def test_input_failure_never_enters_confirmation_or_auto_route(tmp_path):
    result = _stage_a()
    result["stage"] = "input"
    result["input_check"] = {"status": "fail", "ok": False, "errors": ["bad surface"]}
    manager = JobManager(tmp_path, stage_a_fn=lambda *_args, **_kw: result,
                         stage_b_fn=lambda *_args, **_kw: {},
                         mapping_validator=lambda *_: [])
    job = manager.create("owner", content=b"stl", filename="input.stl")
    manager.run_next()
    failed = manager.get(job["id"], "owner")
    assert failed["status"] == "failed"
    assert "bad surface" in failed["error"]["message"]


def test_done_job_can_be_manually_overridden_and_recomputed(tmp_path):
    calls = []

    def stage_b(job_dir, mapping, release, **kwargs):
        calls.append(mapping)
        return {"peak": {}, "wss_field_pa": {}, "timing_s": {}, "flags": []}

    manager = JobManager(tmp_path, stage_a_fn=lambda *_args, **_kw: _stage_a(),
                         stage_b_fn=stage_b, mapping_validator=lambda *_: [])
    job = manager.create("owner", content=b"stl", filename="input.stl")
    manager.run_next(); manager.run_next()
    done = manager.get(job["id"], "owner")
    mapping = {"1": "out-le", "2": "out-li", "3": "out-ri", "4": "out-re"}
    changed = manager.confirm(job["id"], "owner", {"version": done["version"], "stage": "B",
                                                     "mapping": mapping, "acknowledged": True,
                                                     "override": True})
    assert changed["status"] == "queued"
    assert changed["stage"] == "B"
    assert changed["mapping_history"][-1]["source"] == "manual_override"
    manager.run_next()
    assert manager.get(job["id"], "owner")["status"] == "done"
    assert calls[-1] == mapping
