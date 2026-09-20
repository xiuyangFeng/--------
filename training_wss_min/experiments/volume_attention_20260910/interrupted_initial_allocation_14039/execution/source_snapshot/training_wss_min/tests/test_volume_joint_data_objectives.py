"""Joint supervision preserves the historical tasks' draws, units and case weights."""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from training_wss_min import config as C, dataset as D
from training_wss_min.objectives import compute_loss


STATS = {"linear": {"mean": -21.2, "std": 590.3},
         "velocity": {"mean": [.03, -.1, .2], "std": [.21, .25, .27]}}


def volume_case(target="velocity_pressure", name="SYNTH", n_wall=6000, n_vol=14000):
    rng = np.random.default_rng(17)
    case = {"cohort": "AG/fast", "case": name, "unit_id": f"AG/fast/{name}",
            "pos": rng.normal(size=(n_wall, 3)).astype(np.float32),
            "local_radius": np.full(n_wall, 8., dtype=np.float32),
            "curvature": rng.random(n_wall).astype(np.float32),
            "coord_scale_scalar": 120.}
    vol = {"vol_coords_norm": rng.normal(size=(n_vol, 3)).astype(np.float32),
           "vol_local_radius": np.full(n_vol, 8., dtype=np.float32),
           "vol_curvature": rng.random(n_vol).astype(np.float32),
           "vol_radial_aligned": rng.normal(size=(n_vol, 3)).astype(np.float32),
           "vol_dist_to_wall_mm": rng.random(n_vol).astype(np.float32),
           "wall_pressure_rel_peak": (rng.normal(size=n_wall) * 600).astype(np.float32),
           "vol_pressure_rel_peak": (rng.normal(size=n_vol) * 600).astype(np.float32),
           "vol_velocity_aligned_peak": rng.normal(size=(n_vol, 3)).astype(np.float32),
           "p_ref_pa": 12000.}
    case["y_raw"] = (vol["wall_pressure_rel_peak"].copy() if target == "pressure_mixed"
                     else np.zeros((n_wall, 3), dtype=np.float32))
    D._attach_volume(case, vol, target, STATS)
    return case


def data_config(target="velocity_pressure"):
    return C.DataConfig(target=target, input_features=("x", "y", "z"),
                        support_n_points=5000, support_sampling="random", sampling="random",
                        query_mode="independent", query_n_points=5000, query_sampling="random",
                        query_wall_fraction=.5, rot_aug=False)


def test_joint_labels_match_single_task_normalizations_bitwise():
    joint, pressure, velocity = (volume_case(t) for t in
                                 ("velocity_pressure", "pressure_mixed", "velocity"))
    for key in ("y_raw", "y_norm"):
        np.testing.assert_array_equal(joint[key][:, :3], velocity[key])
        np.testing.assert_array_equal(joint[key][:, 3], pressure[key])
    assert np.isfinite(joint["y_raw"]).all()
    assert np.isfinite(joint["y_norm"]).all()
    assert not joint["y_raw"][:joint["n_wall"], :3].any()
    np.testing.assert_array_equal(D.query_rows(joint), np.arange(len(joint["pos"])))


@pytest.mark.parametrize("epoch", [0, 1, 399])
def test_union_retains_exact_single_task_queries_features_and_supervision_budget(epoch):
    samples = {}
    for task in ("velocity_pressure", "pressure_mixed", "velocity"):
        ds = D.WSSMinDataset([volume_case(task)], data_config(task), {}, base_seed=1234)
        ds.set_epoch(epoch)
        samples[task] = ds[0]
    joint = samples["velocity_pressure"]
    for task, channels in (("velocity", slice(0, 3)), ("pressure_mixed", 3)):
        mask = joint["velocity_mask" if task == "velocity" else "pressure_mask"]
        single = samples[task]
        assert mask.sum().item() == 5000
        for key in ("query_idx", "pos", "x"):
            torch.testing.assert_close(joint[key][mask], single[key], rtol=0, atol=0)
        for key in ("y", "y_raw"):
            torch.testing.assert_close(joint[key][mask, channels], single[key], rtol=0, atol=0)
        for key in ("support_idx", "support_pos", "support_x"):
            torch.testing.assert_close(joint[key], single[key], rtol=0, atol=0)
    assert (joint["query_idx"][joint["velocity_mask"]] >= 6000).all()
    assert (joint["query_idx"][joint["pressure_mask"]] < 6000).sum() == 2500
    assert (joint["query_idx"][joint["pressure_mask"]] >= 6000).sum() == 2500
    assert (joint["velocity_mask"] & joint["pressure_mask"]).any()
    assert (joint["velocity_mask"] | joint["pressure_mask"]).all()
    assert joint["query_idx"].unique().numel() == len(joint["query_idx"])


def test_collate_preserves_case_boundaries_and_both_masks():
    cases = [volume_case(name="FIRST"), volume_case(name="SECOND")]
    ds = D.WSSMinDataset(cases, data_config(), {}, base_seed=1234)
    items = [ds[0], ds[1]]
    batch = D.collate(items)
    for i, item in enumerate(items):
        rows = batch["batch"] == i
        for key in ("velocity_mask", "pressure_mask", "y", "y_raw", "pos", "x"):
            torch.testing.assert_close(batch[key][rows], item[key], rtol=0, atol=0)
        assert batch["velocity_mask"][rows].sum() == 5000
        assert batch["pressure_mask"][rows].sum() == 5000
        assert (batch["support_batch"] == i).sum() == 5000
    assert batch["unit_ids"] == [c["unit_id"] for c in cases]


def test_joint_rejects_undersized_task_pool_and_mixed_unit_quantiles():
    case = volume_case(n_vol=4999)
    with pytest.raises(ValueError, match="supervision budget"):
        D.WSSMinDataset([case], data_config(), {})
    with pytest.raises(ValueError, match="mixed-unit quantiles"):
        D.compute_train_weight_quantiles([volume_case()])


def hand_batch():
    # Case 0: vMSE=1, pMSE=9. Case 1: vMSE=4, pMSE=16.
    pred = torch.tensor([[1., 1., 1., 99.], [99., 99., 99., 3.],
                         [2., 2., 2., 4.], [2., 2., 2., 4.], [99., 99., 99., 4.]])
    batch = {"y": torch.zeros_like(pred), "batch": torch.tensor([0, 0, 1, 1, 1]),
             "velocity_mask": torch.tensor([True, False, True, True, False]),
             "pressure_mask": torch.tensor([False, True, True, True, True])}
    return pred.requires_grad_(), batch


def test_joint_loss_hand_calculation_equal_cases_and_equal_tasks():
    pred, batch = hand_batch()
    components = {}
    loss = compute_loss(pred, batch, C.TrainConfig(), "cpu", components=components)
    assert loss.item() == pytest.approx(((1 + 9) / 2 + (4 + 16) / 2) / 2)
    assert components["velocity_mse"].item() == pytest.approx(2.5)
    assert components["pressure_mse"].item() == pytest.approx(12.5)
    assert components["total"].item() == pytest.approx(7.5)
    loss.backward()
    assert not pred.grad[~batch["velocity_mask"], :3].any()
    assert not pred.grad[~batch["pressure_mask"], 3].any()


@pytest.mark.parametrize("unobserved", [1e20, float("nan")])
def test_unobserved_labels_and_predictions_cannot_change_loss_or_gradient(unobserved):
    pred, batch = hand_batch()
    base_loss = compute_loss(pred, batch, C.TrainConfig(), "cpu")
    base_gradient = torch.autograd.grad(base_loss, pred)[0]
    changed = copy.deepcopy(batch)
    perturbed = pred.detach().clone()
    for tensor in (changed["y"], perturbed):
        tensor[~batch["velocity_mask"], :3] = unobserved
        tensor[~batch["pressure_mask"], 3] = unobserved
    perturbed.requires_grad_()
    loss = compute_loss(perturbed, changed, C.TrainConfig(), "cpu")
    gradient = torch.autograd.grad(loss, perturbed)[0]
    torch.testing.assert_close(loss, base_loss, rtol=0, atol=0)
    torch.testing.assert_close(gradient, base_gradient, rtol=0, atol=0)


def test_joint_loss_cannot_silently_drop_masks_or_a_task():
    pred, batch = hand_batch()
    with pytest.raises(ValueError, match="explicit supervision masks"):
        compute_loss(pred, {"y": batch["y"]}, C.TrainConfig(), "cpu")
    batch["velocity_mask"][:2] = False
    with pytest.raises(ValueError, match="both velocity and pressure"):
        compute_loss(pred, batch, C.TrainConfig(), "cpu")


def test_joint_loss_rejects_wss_auxiliary_losses():
    pred, batch = hand_batch()
    with pytest.raises(ValueError, match="equal-weight masked MSE"):
        compute_loss(pred, batch, C.TrainConfig(loss_pinball_lambda=.2), "cpu")


def test_joint_training_logs_masked_tasks_and_selects_one_caseweighted_checkpoint(tmp_path, monkeypatch):
    from training_wss_min import train as T

    cases = []
    for i in range(3):
        case = volume_case(name=f"CASE{i}", n_wall=10, n_vol=20)
        case["y_norm"][:, :3] = i + 1
        case["y_norm"][:10, :3] = float("nan")  # cannot enter metrics or the objective
        case["y_norm"][:, 3] = i + 3
        cases.append(case)
    stats_path, features_path = tmp_path / "stats.json", tmp_path / "features.json"
    stats_path.write_text(json.dumps(STATS))
    features_path.write_text("{}")
    cfg = C.ExpConfig(name=str(tmp_path / "run"))
    cfg.data = replace(data_config(), support_n_points=4, query_n_points=4, num_workers=0,
                       feature_stats_path=str(features_path), wss_stats_path=str(stats_path))
    cfg.train = C.TrainConfig(epochs=1, batch_cases=2, selection_rule="train_loss",
                              ckpt_top_k=1, amp=False, eval_every=1)

    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward_support_query(self, support_pos, support_x, support_batch, pos, x, batch, **kwargs):
            return torch.zeros((len(pos), 4), device=pos.device) + self.anchor * 0

    monkeypatch.setattr(C.ExpConfig, "from_json", lambda path: cfg)
    monkeypatch.setattr(D, "load_partition", lambda *args, **kwargs: cases)
    monkeypatch.setattr(D, "load_split_cases", lambda *args: [])
    def reject_quantiles(*args, **kwargs):
        raise AssertionError("joint training must not flatten physical units")
    monkeypatch.setattr(D, "compute_train_weight_quantiles", reject_quantiles)
    monkeypatch.setattr(T, "build_model", lambda *args: ZeroModel())
    monkeypatch.setattr(T, "plot_history", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["train", "--config", "mock.json"])
    T.main()
    record = json.loads((cfg.run_dir / "history.jsonl").read_text())
    expected_velocity = (1 + 4 + 9) / 3
    expected_pressure = (9 + 16 + 25) / 3
    expected_total = (expected_velocity + expected_pressure) / 2
    assert record["train_loss"] == pytest.approx(expected_total)
    assert record["selection_score"] == pytest.approx(-expected_total)
    assert record["train_velocity_mse_norm"] == pytest.approx(expected_velocity)
    assert record["train_pressure_mse_norm"] == pytest.approx(expected_pressure)
    assert record["train_velocity_sampled_points"] == 12
    assert record["train_pressure_sampled_points"] == 12
    assert record["loss_components"]["total"] == pytest.approx(expected_total)
    checkpoint = torch.load(cfg.run_dir / "ckpt_best.pt", weights_only=False)
    assert checkpoint["metric"] == pytest.approx(-expected_total)
    assert json.loads((cfg.run_dir / "weight_quantiles.json").read_text())["mode"] == "not_used"
    source = json.loads((cfg.run_dir / "feature_stats_source.json").read_text())
    assert source["mode"] == "frozen" and source["source_path"] == str(features_path)
