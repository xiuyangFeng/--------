from __future__ import annotations

import json

import numpy as np
import pytest

from training_wss_min import config as C
from training_wss_min import dataset as D


def test_case_features_are_case_equal_in_stats_and_broadcast() -> None:
    feature = C.CASE_FEATURE_KEYS[0]
    cases = [
        {feature: 1.0, "pos": np.zeros((2, 3)), "cohort": "AG/fast"},
        {feature: 3.0, "pos": np.zeros((20, 3)), "cohort": "AG/slow"},
    ]
    stats = D.compute_feature_stats(cases, (feature,))
    assert stats[feature]["mean"] == pytest.approx(2.0)
    built = D.build_features(cases[0], np.asarray([0, 1]), (feature,), stats)
    assert built.shape == (2, 1)
    assert np.allclose(built[:, 0], built[0, 0])


def test_case_feature_table_is_strict(tmp_path) -> None:
    feature = C.CASE_FEATURE_KEYS[0]
    path = tmp_path / "case_features.json"
    path.write_text(
        json.dumps({"feature_names": [feature], "cases": {"AG/fast/X": {feature: 1.5}}}),
        encoding="utf-8",
    )
    loaded = D.load_case_feature_table(path)
    assert loaded["AG/fast/X"][feature] == pytest.approx(1.5)


def test_case_feature_requires_sidecar() -> None:
    cfg = C.ExpConfig()
    cfg.data.input_features = ("x", C.CASE_FEATURE_KEYS[0])
    with pytest.raises(ValueError, match="case_features_path"):
        C.validate_features(cfg)
