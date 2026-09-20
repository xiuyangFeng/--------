"""Wave-5 (wss_local_wave5_20260913) contracts: sidecar v1.3 stenosis index, T3 region-focus loss and residual gate."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.objectives import compute_loss, region_focus_weights

C1 = C.PROJECT_ROOT / "training_wss_min/configs/wss_direct_recovery_20260912/C1_s1234.json"


def test_new_fields_default_off_and_registered():
    assert C.TrainConfig().loss_region_feature is None and C.EvalConfig().residual_gate_quantile is None
    assert "log_r_over_rdistal" in C.V7_FLOW_FEATURE_KEYS and "log_wss_base" in C.V8_CASCADE_FEATURE_KEYS
    assert "log_wss_base" in C.SIDECAR_FEATURE_KEYS and "log_wss_base" in C.FEATURE_KEYS
    raw = json.loads(C1.read_text())
    cfg = C.ExpConfig.from_dict(raw)
    assert C.v6_point_features(cfg) == tuple(f for f in raw["data"]["input_features"] if f in C.SIDECAR_FEATURE_KEYS)
    ok = copy.deepcopy(raw)
    ok["data"]["input_features"] += ["log_r_over_rdistal", "log_wss_base"]
    assert C.v6_point_features(C.ExpConfig.from_dict(ok))[-2:] == ("log_r_over_rdistal", "log_wss_base")


def test_region_and_gate_config_contracts():
    raw = json.loads(C1.read_text())
    bad = copy.deepcopy(raw); bad["train"]["loss_region_feature"] = "log_wss_base"
    with pytest.raises(ValueError):  # not an input feature
        C.ExpConfig.from_dict(bad)
    bad = copy.deepcopy(raw); bad["data"]["input_features"] = raw["data"]["input_features"] + ["log_wss_base"]
    bad["train"].update(loss_region_feature="log_wss_base", loss_region_quantile=1.0)
    with pytest.raises(ValueError):
        C.ExpConfig.from_dict(bad)
    bad = copy.deepcopy(raw); bad["eval"]["residual_gate_quantile"] = 0.8
    with pytest.raises(ValueError):  # gate needs a residual offset feature
        C.ExpConfig.from_dict(bad)
    ok = copy.deepcopy(raw)
    ok["data"]["input_features"] = raw["data"]["input_features"] + ["log_wss_base"]
    ok["data"]["target_log_offset_feature"] = "log_wss_base"
    ok["train"].update(loss_region_feature="log_wss_base", loss_region_quantile=0.8, loss_region_outside_weight=0.2)
    ok["eval"]["residual_gate_quantile"] = 0.8
    cfg = C.ExpConfig.from_dict(ok)
    assert cfg.train.loss_region_feature == "log_wss_base" and cfg.eval.residual_gate_quantile == 0.8


def test_region_focus_weights_are_case_relative():
    values = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    w = region_focus_weights(values, batch, 0.75, 0.2)
    # per-case 75% quantile: case 0 -> 2.25, case 1 -> 12.25 (only the top point of each case is inside)
    torch.testing.assert_close(w, torch.tensor([0.2, 0.2, 0.2, 1.0, 0.2, 0.2, 0.2, 1.0]))


def test_region_loss_matches_manual_weighting_and_is_noop_when_off():
    gen = torch.Generator().manual_seed(0)
    pred = torch.randn(40, generator=gen)
    y = torch.randn(40, generator=gen)
    batch = {"y": y, "y_raw": torch.exp(y), "batch": torch.repeat_interleave(torch.arange(4), 10),
             "curv": torch.zeros(40), "invr": torch.zeros(40), "loss_region_value": torch.randn(40, generator=gen)}
    off = C.TrainConfig(loss="mse", loss_pinball_lambda=0.0)
    on = C.TrainConfig(loss="mse", loss_pinball_lambda=0.0, loss_region_feature="log_wss_base",
                       loss_region_quantile=0.7, loss_region_outside_weight=0.25)
    plain = compute_loss(pred, batch, off, "cpu")
    torch.testing.assert_close(plain, ((pred - y) ** 2).mean())
    focused = compute_loss(pred, batch, on, "cpu")
    w = region_focus_weights(batch["loss_region_value"], batch["batch"], 0.7, 0.25)
    torch.testing.assert_close(focused, (((pred - y) ** 2) * w).sum() / w.sum())
    with pytest.raises(ValueError):
        compute_loss(pred, {k: v for k, v in batch.items() if k != "loss_region_value"}, on, "cpu")


def test_residual_gate_keeps_stage_one_outside_the_region():
    stats = {"method": "log_z", "eps": 1e-6, "log": {"mean": 0.5, "std": 1.3},
             "offset": {"feature": "log_wss_base", "mean": 0.02, "std": 0.4}}
    rng = np.random.default_rng(1)
    offset = rng.normal(0.0, 1.0, 200)
    z_res = rng.normal(0.0, 1.0, 200)
    gated = D.gate_residual_prediction(z_res, offset, stats, 0.8)
    inside = offset >= np.quantile(offset, 0.8)
    np.testing.assert_array_equal(gated[inside], z_res[inside])
    back = D.residual_to_standard_logz(gated, offset, stats)
    # outside the region the standard log_z prediction equals the stage-one prediction exactly
    stage_one = (offset - stats["log"]["mean"]) / stats["log"]["std"]
    np.testing.assert_allclose(back[~inside], stage_one[~inside], rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        D.gate_residual_prediction(z_res, offset, {"method": "log_z", "log": stats["log"]}, 0.8)


def test_flowref_v13_stenosis_index_matches_distal_radius_recipe():
    root = C.PROJECT_ROOT / "data_wss_v5/views/wss_min_flowref_v1"
    view = C.PROJECT_ROOT / "data_wss_v5/views/wss_min_view_v1"
    case = "AG/fast/WANG_CHUN_MING"
    if not (root / case / "features.npz").is_file():
        pytest.skip("flow-reference sidecar not built")
    with np.load(root / case / "features.npz") as z, np.load(view / case / "bundle.npz", allow_pickle=True) as b:
        if "wall_log_r_over_rdistal" not in z.files:
            pytest.skip("sidecar older than v1.3")
        report = json.loads((root / case / "flowref_report.json").read_text())
        distal = {int(k): float(v) for k, v in report["distal_radius_by_segment"].items()}
        seg = b["wall_segment_id"].astype(int)
        expected = np.log(np.clip(b["wall_local_radius"].astype(np.float64), 1e-3, None)) - np.log(
            np.array([max(distal[int(s)], 1e-3) for s in seg]))
        np.testing.assert_allclose(z["wall_log_r_over_rdistal"], expected, rtol=1e-5, atol=1e-5)
        # Murray's log_tau0 minus the flow share is -3 ln R; the stenosis index is the part of it that depends on
        # where the point sits along its segment, so within a segment the two move together with slope -3
        mask = seg == seg[0]
        slope = np.polyfit(z["wall_log_r_over_rdistal"][mask], (z["wall_log_tau0_murray"] - z["wall_log_q_branch_murray"])[mask], 1)[0]
        assert abs(slope + 3.0) < 1e-3
