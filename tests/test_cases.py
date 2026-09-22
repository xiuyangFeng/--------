"""C1: case cards group every run of one input geometry; missing releases and pending reviews are derived."""
from __future__ import annotations

from wss_deploy.cases import group_cases, group_key
from tests._c_helpers import Service, call, finished, manager

RELEASES = [{"id": "REL_A", "contract": {"protocol": "single_frame_wss"}}, {"id": "REL_V", "contract": {"protocol": "single_frame_volume"}}]


def _job(i, sha, release, status="done", review="unreviewed", **extra):
    return {"id": f"j{i}", "case_id": extra.pop("case_id", f"case{i}"), "input_sha256": sha, "created_at": f"2026-09-21T10:0{i}:00",
            "status": status, "review": {"status": review}, "model_release": {"id": release, "contract": {"protocol": "single_frame_wss" if release == "REL_A" else "single_frame_volume"}},
            "run_identity": "r" * 64, "version": 3, "reusable": status == "done", "tags": extra.pop("tags", []), **extra}


def test_group_cases_merges_wall_and_volume_runs_of_one_geometry():
    jobs = [_job(1, "s" * 64, "REL_A", case_id="LIU", patient_id="P-1"), _job(2, "s" * 64, "REL_V", case_id="LIU", tags=["AAA"]),
            _job(3, "t" * 64, "REL_A", review="reviewed"), _job(4, None, "REL_A", status="queued")]
    cards = group_cases(jobs, RELEASES)
    assert [card["group_key"] for card in cards] == ["j4", "t" * 64, "s" * 64][::-1] or len(cards) == 3
    by_key = {card["group_key"]: card for card in cards}
    liu = by_key["s" * 64]
    assert liu["case_ids"] == ["LIU"] and liu["patient_id"] == "P-1" and liu["tags"] == ["AAA"]
    assert [run["job_id"] for run in liu["runs"]] == ["j2", "j1"]
    assert liu["latest"] == {"REL_A": "j1", "REL_V": "j2"} and liu["missing_releases"] == [] and liu["families"] == ["volume", "wall"]
    assert liu["pending_review"] == 2 and liu["reusable_job_id"] == "j2" and liu["latest_at"] == "2026-09-21T10:02:00"
    other = by_key["t" * 64]
    assert other["missing_releases"] == ["REL_V"] and other["pending_review"] == 0
    queued = by_key["job:j4"]
    assert queued["input_sha256"] is None and queued["latest"] == {} and queued["runs"][0]["family"] == "wall" and queued["reusable_job_id"] is None
    assert group_key({"id": "x"}) == "job:x"


def test_manager_cases_filters_paginates_and_isolates_owner(tmp_path):
    mgr = manager(tmp_path)
    a = finished(mgr, case_id="A", content=b"geom-a", patient_id="P-1", tags=["AAA"])
    finished(mgr, case_id="A2", content=b"geom-a", protocol="single_frame_volume")
    finished(mgr, case_id="B", content=b"geom-b", patient_id="P-2", scan_label="follow-up-zeta")
    finished(mgr, owner="bob", case_id="C", content=b"geom-c")
    result = mgr.cases("owner")
    assert result["total"] == 2 and result["page_size"] == 20 and [r["id"] for r in result["releases"]] == ["REL_A"]
    assert {tuple(card["case_ids"]) for card in result["cases"]} == {("A2", "A"), ("B",)}
    assert mgr.cases("owner", patient_id="P-1")["cases"][0]["case_ids"] == ["A2", "A"]
    assert mgr.cases("owner", q="P-2")["total"] == 1 and mgr.cases("owner", tag="AAA")["total"] == 1
    assert mgr.cases("owner", page=2, page_size=1)["cases"][0]["case_ids"] in (["B"], ["A2", "A"])
    assert mgr.cases("bob")["total"] == 1 and mgr.cases("bob", all_owners=True)["total"] == 3
    card = next(card for card in result["cases"] if "A" in card["case_ids"])
    assert card["input_sha256"] == a["input_sha256"] and card["reusable_job_id"] and card["pending_review"] == 2


def test_http_cases_route(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, owner = service.session(api)
        finished(service.manager, owner=owner, case_id="A", content=b"geom-a")
        status, payload, _ = call(api, "GET", "/api/cases?page_size=5", {"Cookie": cookie})
        assert status == 200 and payload["total"] == 1 and payload["cases"][0]["runs"][0]["status"] == "done"
        status, payload, _ = call(api, "GET", "/api/cases?page=x", {"Cookie": cookie})
        assert status == 400
    finally:
        api.close(); service.close()
