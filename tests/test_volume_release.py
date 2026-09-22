import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wss_deploy.infer import Release
from wss_deploy.io_utils import file_sha256
from wss_deploy.registry import (ReleaseError, ReleaseRegistry, VOLUME_FEATURES,
                                 VOLUME_FIELDS, _contract)


def contract():
    return {"release": "volume-demo", "target": "pressure_velocity", "model_family": "PF6_VF6",
            "time_axis": [{"index": 0, "step": 1162, "time_s": .21}],
            "fields": copy.deepcopy(VOLUME_FIELDS),
            "models": [{"field": field, "seed": seed, "path": f"models/{field}_s{seed}"}
                       for field in ("pressure", "velocity") for seed in (1234, 7, 2025)]}


def package(root):
    info = contract()
    root.mkdir()
    (root / "release.json").write_text(json.dumps(info))
    for spec in info["models"]:
        run = root / spec["path"]
        run.mkdir(parents=True)
        pressure = spec["field"] == "pressure"
        cfg = {"data": {"target": "pressure_mixed" if pressure else "velocity",
                        "timesteps": "peak", "target_normalization": "global_stats",
                        "required_frame_version": "v5_atlas_frame_v1", "input_features": VOLUME_FEATURES},
               "model": {"out_dim": 1 if pressure else 3},
               "train": {"seed": spec["seed"]}, "eval": {"fixed_support": True}}
        stats = {"method": "linear", "required_frame_version": "v5_atlas_frame_v1",
                 "timesteps_scope": "peak", "peak_step": 1162,
                 "linear": {"mean": -2., "std": 3.},
                 "velocity": {"mean": [1., 2., 3.], "std": [2., 3., 4.]}}
        for name, data in (("config", cfg), ("feature_stats", {}), ("wss_global_stats", stats),
                           ("target_normalization", {"mode": "global_stats", "physical_recovery_enabled": True})):
            (run / f"{name}.json").write_text(json.dumps(data))
        (run / "ckpt_best.pt").write_bytes(b"mock checkpoint")
    (root / "rules").mkdir()
    (root / "rules/flow_split_rule_train136.json").write_text("{}")
    manifest(root)
    return root


def manifest(root):
    paths = [p for sub in ("models", "rules") for p in sorted((root / sub).rglob("*")) if p.is_file()]
    (root / "MANIFEST.sha256").write_text("".join(f"{file_sha256(p)}  {p.relative_to(root)}\n" for p in paths))


def test_volume_contract_requires_explicit_gauge_frame_and_consistent_seeds():
    assert _contract(contract())["protocol"] == "single_frame_volume"
    for path, value in (("reference", "absolute"), ("reference", None)):
        bad = contract(); bad["fields"]["pressure"][path] = value
        with pytest.raises(ReleaseError):
            _contract(bad)
    bad = contract(); bad["fields"]["velocity"]["frame"] = "aligned"
    with pytest.raises(ReleaseError):
        _contract(bad)
    bad = contract(); bad["models"].pop()
    with pytest.raises(ReleaseError):
        _contract(bad)
    bad = contract(); bad["time_axis"].append(bad["time_axis"][0])
    with pytest.raises(ReleaseError):
        _contract(bad)


def test_volume_manifest_and_runtime_protocol_fail_closed(tmp_path):
    root = package(tmp_path / "volume")
    registry = ReleaseRegistry(root, loader=lambda *a, **kw: SimpleNamespace())
    assert registry.list()[0]["models_count"] == 3
    assert registry.list()[0]["weights_count"] == 6
    registry.load(device="cpu")
    cfg_path = root / "models/velocity_s7/config.json"
    cfg = json.loads(cfg_path.read_text()); cfg["data"]["target_normalization"] = "case_max"
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ReleaseError, match="校验失败"):
        registry.load(device="cpu")
    manifest(root)
    with pytest.raises(ReleaseError, match="归一化合同"):
        ReleaseRegistry(root, loader=lambda *a, **kw: SimpleNamespace()).load(device="cpu")


def test_volume_inference_restores_gauge_rotates_vectors_and_never_queries_wall_velocity(tmp_path, monkeypatch):
    root = package(tmp_path / "volume")
    calls = []

    def load(run, device, checkpoint):
        raw = json.loads((run / "config.json").read_text())
        cfg = SimpleNamespace(data=SimpleNamespace(**raw["data"]))
        model = SimpleNamespace(eval=lambda: None, value=raw["train"]["seed"])
        return cfg, {}, model, {}

    def predict(model, case, features, stats, device, *, cfg, return_all_channels):
        velocity = cfg.data.target == "velocity"
        rows = case["query_pool"]
        assert return_all_channels is velocity
        assert np.array_equal(case["support_pool"], np.arange(2))
        assert np.array_equal(rows, np.arange(2, 5) if velocity else np.arange(5))
        calls.append((cfg.data.target, model.value))
        return np.full((len(rows), 3) if velocity else (len(rows),), 1 if model.value == 1234 else 3.)

    monkeypatch.setattr("wss_deploy.infer.E.load_model_from_run", load)
    monkeypatch.setattr("wss_deploy.infer.E.predict_case_norm", predict)
    monkeypatch.setattr("wss_deploy.infer.D.load_case", lambda *a, **kw: pytest.fail("CFD data read"))
    monkeypatch.setattr("wss_deploy.infer.E.load_wss_stats_for_run", lambda *a, **kw: pytest.fail("stats fallback"))
    release = ReleaseRegistry(root).load(device="cpu", seed_count=2)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    case = {"pos": np.zeros((5, 3)), "n_wall": 2, "support_pool": np.arange(2),
            "frame_rotation": rotation, "p_ref_pa": 1e9}
    result = release.predict(case)
    assert calls == [("pressure_mixed", 1234), ("pressure_mixed", 7), ("velocity", 1234), ("velocity", 7)]
    # p = z*3 - 2; do not add the unavailable CFD reference.
    np.testing.assert_allclose(result["pressure_pa"], 4.)
    assert result["seed_pressure_pa"].shape == (2, 5)
    aligned = np.array([2, 2, 2]) * np.array([2, 3, 4]) + np.array([1, 2, 3])
    np.testing.assert_allclose(result["velocity_m_s"], np.tile(aligned @ rotation, (3, 1)))
    np.testing.assert_allclose(result["speed_m_s"], np.linalg.norm(aligned))
    assert result["seed_velocity_m_s"].shape == (2, 3, 3)
    assert "query_pool" not in case
    with pytest.raises(ValueError, match="正交"):
        release.predict(dict(case, frame_rotation=np.ones((3, 3))))
    with pytest.raises(ValueError, match="壁面"):
        release.predict(dict(case, support_pool=np.arange(5)))
    with pytest.raises(ValueError, match="模型数量"):
        Release(root, device="cpu", seed_count=4)
