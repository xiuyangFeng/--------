"""The frozen feature program must not change without a deliberate contract bump."""
from __future__ import annotations

import wss_features


def test_contract_record_matches_sources():
    recorded = wss_features.recorded_contract()
    assert recorded.get("version") == wss_features.FEATURE_CONTRACT_VERSION
    assert recorded.get("source_hash") == wss_features.contract_hash(), (
        "wss_features sources changed. Re-validate against the training implementation "
        "(tests/test_wss_features_equivalence.py) and then run `python -m wss_features record`.")
    assert wss_features.contract()["matches_record"] is True


def test_package_has_no_training_dependencies():
    import importlib
    import sys
    for name in ("wss_features.atlas", "wss_features.cloud", "wss_features.curvature", "wss_features.flowref",
                 "wss_features.frame", "wss_features.sampling", "wss_features.stl"):
        importlib.import_module(name)
    forbidden = [m for m in sys.modules if m.split(".")[0] in {"wss_v5", "training_wss_min", "wss_pinn"}]
    # Other tests may legitimately import the training packages; only assert when this test ran in isolation.
    if not any(m.startswith("tests.") or m.startswith("test_") for m in sys.modules):
        assert not forbidden, forbidden
