"""Case cards (C1): every run of one input geometry grouped under one card.

Pure functions over light job snapshots (``JobManager._snapshot(detail=False)``), so the grouping is
unit-testable without a manager and never changes the job model.
"""
from __future__ import annotations

from .jobs import FAMILY_OF_PROTOCOL, family_of_release
from .schema import display_name


def _run(job: dict) -> dict:
    release = job.get("model_release") if isinstance(job.get("model_release"), dict) else {}
    review = job.get("review") if isinstance(job.get("review"), dict) else {}
    return {"job_id": job.get("id"), "release_id": release.get("id"), "family": job.get("family") or family_of_release(release),
            "status": job.get("status"), "review": review.get("status") or "unreviewed", "created_at": job.get("created_at") or "",
            "run_identity": job.get("run_identity"), "version": job.get("version"), "case_id": job.get("case_id"),
            # C7 (2026-09-30): patient id when entered, else case id.
            "display_name": job.get("display_name") or display_name(job.get("case_id"), job.get("patient_id")),
            "reusable": bool(job.get("reusable")), "reused_from": job.get("reused_from"), "source_job_id": job.get("source_job_id"),
            # §19.2 display labels from the light snapshot (absent on hand-built records → None / False).
            "family_label": job.get("family_label"), "release_short": job.get("release_short"), "has_cycle": bool(job.get("has_cycle"))}


def group_key(job: dict) -> str:
    sha = job.get("input_sha256")
    return str(sha) if sha else f"job:{job.get('id')}"


def group_cases(jobs: list[dict], releases: list[dict] | None = None) -> list[dict]:
    """Group light snapshots by input SHA256 (contract §11.1); cards ordered by latest run, newest first."""
    known = [(r.get("id"), family_of_release(r)) for r in (releases or []) if isinstance(r, dict) and r.get("id")]
    groups: dict[str, list[dict]] = {}
    for job in jobs:
        if not isinstance(job, dict) or not job.get("id"):
            continue
        groups.setdefault(group_key(job), []).append(job)
    cards = []
    for key, members in groups.items():
        # Stable newest-first order: the manager feeds snapshots newest-first already, so a plain stable sort on
        # created_at keeps that order for jobs created within the same second (ids are random and must not decide).
        members = sorted(members, key=lambda j: j.get("created_at") or "", reverse=True)
        runs = [_run(job) for job in members]
        latest: dict[str, str] = {}
        for run in runs:   # newest first, so the first done run per release wins
            rid = run["release_id"]
            if rid and run["status"] == "done" and rid not in latest:
                latest[rid] = run["job_id"]
        newest = members[0]
        case_ids = []
        for job in members:
            cid = job.get("case_id")
            if cid and cid not in case_ids:
                case_ids.append(cid)
        tags = []
        for job in members:
            for tag in job.get("tags") or []:
                if tag not in tags:
                    tags.append(tag)

        def first(field):
            return next((job.get(field) for job in members if job.get(field)), "")
        reusable = next((run["job_id"] for run in runs if run["reusable"]), None)
        cards.append({"input_sha256": key if not key.startswith("job:") else None, "group_key": key, "case_ids": case_ids,
                      "display_name": display_name(case_ids[0] if case_ids else "", first("patient_id")),
                      "patient_id": first("patient_id"), "scan_label": first("scan_label"), "scan_date": first("scan_date"),
                      "tags": tags, "latest_at": newest.get("created_at") or "",
                      "pending_review": sum(1 for run in runs if run["status"] == "done" and run["review"] != "reviewed"),
                      "runs": runs, "latest": latest,
                      "missing_releases": [rid for rid, _ in known if rid not in latest],
                      "families": sorted({run["family"] for run in runs if run["family"]}),
                      "reusable_job_id": reusable})
    cards.sort(key=lambda card: (card["latest_at"], card["group_key"]), reverse=True)
    return cards


__all__ = ["FAMILY_OF_PROTOCOL", "group_cases", "group_key"]
