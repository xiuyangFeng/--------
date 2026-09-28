"""v0.15.11 (P0-3): an ensemble's device inputs are prepared once per case and read by every member."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wss_deploy import pipeline as P
from wss_deploy.prepared_inference import PreparedInference, _fingerprint

RELEASES = Path(__file__).resolve().parents[1] / "outputs" / "wss_deploy_release"


def _release_or_skip(name):
    if not (RELEASES / name / "release.json").is_file():
        pytest.skip(f"release {name} not available")
    from wss_deploy.infer import Release
    return Release(RELEASES / name, device="cpu")


@pytest.mark.parametrize("name, keys", [
    ("M1_3head_3seed_20260922", ("wss_pa", "seed_pred_pa", "tawss_pa", "seed_tawss_pa", "osi", "seed_osi")),
    ("X5D_v51_5seed_20260916", ("wss_pa", "seed_pred_pa", "seed_sd_pa")),
    ("PF6_VF6_peak_3seed_20260920", ("pressure_pa", "seed_pressure_pa", "velocity_m_s", "seed_velocity_m_s")),
])
def test_real_ensembles_are_bit_identical_with_prepared_inputs_on_the_cpu(monkeypatch, name, keys):
    from wss_deploy.infer import synthetic_case
    release = _release_or_skip(name)
    case_a, case_b = synthetic_case(release, n_points=6000, seed=2), synthetic_case(release, n_points=6000, seed=2)
    with P.torch_threads(4):
        monkeypatch.setenv("WSS_DEPLOY_PREPARED_INPUTS", "0")
        plain = release.predict(case_a)
        monkeypatch.setenv("WSS_DEPLOY_PREPARED_INPUTS", "1")
        prepared = release.predict(case_b)
    stats = prepared["prepared_inputs"]
    assert stats["fallbacks"] == 0 and stats["reused"] > 0 and stats["calls"] == len(release.models)
    assert plain["prepared_inputs"]["fallbacks"] == len(release.models)
    for key in keys:
        assert np.array_equal(plain[key], prepared[key]), key


@pytest.fixture(scope="module")
def m1():
    return _release_or_skip("M1_3head_3seed_20260922")


def _args(release, case):
    member = release.models[0]
    return (member["model"], case, member["cfg"].data.input_features, member["feat_stats"], "cpu")


def test_only_the_stock_evaluator_and_known_models_inside_the_scope_are_prepared(m1, monkeypatch):
    from training_wss_min import evaluate as E
    from wss_deploy.infer import synthetic_case
    case = synthetic_case(m1, n_points=3000, seed=3)
    model, _, features, stats, device = _args(m1, case)
    cfg = m1.models[0]["cfg"]
    calls = []

    def patched(*args, **kwargs):
        calls.append(1)
        return E.predict_case_norm(*args, **kwargs)

    outside = PreparedInference()
    outside.predict(E.predict_case_norm, model, case, features, stats, device, cfg=cfg)   # never entered
    with PreparedInference() as scope:
        scope.predict(patched, model, case, features, stats, device, cfg=cfg)
        with pytest.raises(AttributeError, match="encode_support"):   # an unknown model reaches the evaluator unchanged
            scope.predict(E.predict_case_norm, torch.nn.Identity(), case, features, stats, device, cfg=cfg)
    assert outside.stats["fallbacks"] == 1 and scope.stats["fallbacks"] == 2 and calls == [1]
    assert scope.stats["built"] == 0
    monkeypatch.setenv("WSS_DEPLOY_PREPARED_INPUTS", "off")
    with PreparedInference() as scope:
        scope.predict(E.predict_case_norm, model, case, features, stats, device, cfg=cfg)
    assert scope.stats["fallbacks"] == 1 and scope.stats["built"] == 0


def test_budget_versions_and_replaced_entries_force_a_rebuild(m1, monkeypatch):
    from training_wss_min import evaluate as E
    from wss_deploy.infer import synthetic_case
    case = synthetic_case(m1, n_points=3000, seed=4)
    model, _, features, stats, device = _args(m1, case)
    cfg = m1.models[0]["cfg"]
    with P.torch_threads(4):
        reference = E.predict_case_norm(model, case, features, stats, device, cfg=cfg, return_all_channels=True)
        with PreparedInference() as scope:
            first = scope.predict(E.predict_case_norm, model, case, features, stats, device, cfg=cfg, return_all_channels=True)
            built = scope.stats["built"]
            # an in-place change of a retained tensor is detected by its version counter
            block = next(iter(scope._entries.values()))[0]
            block["x"].add_(0)
            again = scope.predict(E.predict_case_norm, model, case, features, stats, device, cfg=cfg, return_all_channels=True)
            assert scope.stats["built"] == built + 1
            # replacing a case entry changes the case fingerprint: everything is rebuilt from the new content
            case["pos"] = case["pos"].copy()
            case["pos"][0, 0] += 0.25
            moved = scope.predict(E.predict_case_norm, model, case, features, stats, device, cfg=cfg, return_all_channels=True)
            assert scope.stats["built"] == 2 * built + 1
        moved_plain = E.predict_case_norm(model, case, features, stats, device, cfg=cfg, return_all_channels=True)
        monkeypatch.setenv("WSS_DEPLOY_PREPARED_INPUTS_MB", "0")
        with PreparedInference() as scope:
            scope.predict(E.predict_case_norm, model, case, features, stats, device, cfg=cfg, return_all_channels=True)
            assert scope.stats["fallbacks"] == 1 and not scope._entries
    assert np.array_equal(reference, first) and np.array_equal(reference, again)
    assert np.array_equal(moved_plain, moved) and not np.array_equal(moved, reference)


def test_fingerprint_is_exact_and_refuses_opaque_values():
    base = {"a": np.arange(4, dtype=np.float32), "b": 1, "c": [1.0, "x", None, True]}
    assert _fingerprint(base) == _fingerprint({"c": [1.0, "x", None, True], "b": 1, "a": np.arange(4, dtype=np.float32)})
    for changed in ({**base, "a": np.arange(4, dtype=np.float64)}, {**base, "a": np.arange(4, dtype=np.float32).reshape(2, 2)},
                    {**base, "b": 1.0}, {**base, "b": True}, {**base, "c": (1.0, "x", None, True)[:3]}):
        assert _fingerprint(changed) != _fingerprint(base)
    for opaque in ({"a": object()}, {1: "non-string key"}, {"a": np.array([object()], dtype=object)}):
        with pytest.raises(TypeError):
            _fingerprint(opaque)
