"""Outlet naming released without a person (2026-10-01): the library-calibrated sidecar profile, the release binding,
the method binding and the left/right anatomical-handedness check."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wss_deploy import centerline as CL

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "wss_deploy" / "naming_profiles"
GOLDEN = ROOT / "outputs" / "wss_deploy_golden" / "20260920_baseline"
RELEASES = ("M1_3head_3seed_20260922", "PF6_VF6_peak_3seed_20260920", "X5D_v51_5seed_20260916")


class _Release:
    """A release descriptor without an embedded profile (the frozen release folders are never edited)."""
    def __init__(self, name):
        self.name, self.info = name, {"release": name}


def _proposal(confidence=0.97, consistent=True, method=CL.CONFIDENCE_METHOD, hand_method=CL.HANDEDNESS_METHOD):
    return {"auto_ok": True, "confirmation_required": False, "confidence": confidence, "confidence_method": method,
            "mapping": {"3": "out-le", "4": "out-li", "5": "out-ri", "6": "out-re"}, "flags": [],
            "handedness": {"evidence": 2.0 if consistent else -1.0, "consistent": consistent, "method": hand_method}}


def test_the_committed_profile_is_validated_and_bound_to_the_naming_code():
    files = sorted(PROFILES.glob("*.json"))
    assert files, "no naming profile committed"
    profile = json.loads(files[0].read_text(encoding="utf-8"))
    assert profile["status"] == "validated" and profile["releases"] == "*"          # naming never reads weights
    assert profile["naming_fingerprint"] == CL.naming_fingerprint()                  # measured with this code
    assert profile["joint_lower_bound"] >= CL.CONFIDENCE_THRESHOLD and profile["sample_count"] >= 59
    assert profile["statistics"]["errors_passed"] == 0
    assert profile["proxy_method"] == CL.CONFIDENCE_METHOD and profile["handedness_method"] == CL.HANDEDNESS_METHOD
    assert profile["require_handedness"] is True and profile["allowed_orientation_sources"] == ["unknown_stl"]
    text = files[0].read_text(encoding="utf-8")
    assert "ZHANG" not in text and "unit_id" not in text and "__" not in text           # no case names


@pytest.mark.parametrize("release", RELEASES + ("X5Dcap_asym2_v52d_future",))   # a later release needs no new profile
def test_a_confident_consistent_proposal_passes_for_every_release(release):
    gate = CL.evaluate_confidence_gate(_proposal(), orientation_source="unknown_stl", release=_Release(release))
    assert gate["passed"] is True, gate["reasons"]
    assert gate["calibration_status"] == "validated" and gate["profile_release"] == release


def test_the_gate_stops_inconsistent_handedness_low_proxy_and_other_methods():
    rel = _Release(RELEASES[0])
    cases = {
        "handedness": _proposal(consistent=False),
        "proxy": _proposal(confidence=0.94),
        "method": _proposal(method="geometry_margin_proxy_v2"),
        "hand_method": _proposal(hand_method="other"),
    }
    for name, proposal in cases.items():
        gate = CL.evaluate_confidence_gate(proposal, orientation_source="unknown_stl", release=rel)
        assert gate["passed"] is False, name
    no_hand = _proposal(); no_hand.pop("handedness")                     # a stage A from before the check
    assert CL.evaluate_confidence_gate(no_hand, orientation_source="unknown_stl", release=rel)["passed"] is False


@pytest.mark.skipif(not GOLDEN.is_dir(), reason="golden jobs not present")
def test_golden_centrelines_name_as_confirmed_and_agree_on_handedness():
    jobs = sorted(p for p in GOLDEN.iterdir() if (p / "centerline").is_dir())
    assert jobs
    for job in jobs:
        record = json.loads((job / "job.json").read_text())
        proposal = CL.propose_outlets(CL.load_vessel_geom_atlas(job / "centerline"))
        assert proposal["mapping"] == record["mapping"], job.name
        assert proposal["handedness"]["consistent"] is True and proposal["handedness"]["method"] == CL.HANDEDNESS_METHOD
        # the orientation is no longer part of the proposal's own flag (the profile decides)
        assert proposal["confirmation_required"] == bool(proposal["flags"] or proposal["confidence"] < CL.CONFIDENCE_THRESHOLD)


def test_a_changed_naming_code_closes_the_gate(monkeypatch):
    monkeypatch.setattr(CL, "_NAMING_FINGERPRINT", "0" * 16)
    gate = CL.evaluate_confidence_gate(_proposal(), orientation_source="unknown_stl", release=_Release(RELEASES[0]))
    assert gate["passed"] is False and any("重新校准" in r for r in gate["reasons"])
