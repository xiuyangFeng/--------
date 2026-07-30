from __future__ import annotations

import pytest
import torch

from training_wss_min import config as C
from training_wss_min.objectives import (
    casebalanced_hotspot_bce,
    compute_loss,
    pinball_loss,
)


def test_hotspot_bce_prefers_correct_case_relative_ordering() -> None:
    target = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0] * 2)
    batch = torch.tensor([0] * 5 + [1] * 5)
    correct = torch.tensor([-3.0, -2.0, -1.0, 0.0, 3.0] * 2)
    flipped = -correct
    assert casebalanced_hotspot_bce(correct, target, batch, 0.8) < (
        casebalanced_hotspot_bce(flipped, target, batch, 0.8)
    )


def test_pinball_q90_penalizes_underprediction_more_than_overprediction() -> None:
    target = torch.tensor([1.0])
    under = pinball_loss(torch.tensor([0.0]), target, 0.9)
    over = pinball_loss(torch.tensor([2.0]), target, 0.9)
    assert under == pytest.approx(0.9)
    assert over == pytest.approx(0.1)


def test_compute_loss_accepts_regression_plus_hotspot_channels() -> None:
    cfg = C.TrainConfig(
        loss="mse",
        loss_hotspot_bce_lambda=0.2,
        hotspot_quantile=0.8,
    )
    prediction = torch.tensor(
        [[0.0, -2.0], [1.0, -1.0], [2.0, 0.0], [3.0, 1.0], [4.0, 2.0]],
        requires_grad=True,
    )
    batch = {
        "y": torch.arange(5, dtype=torch.float32),
        "batch": torch.zeros(5, dtype=torch.long),
    }
    loss = compute_loss(prediction, batch, cfg, "cpu")
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 1].abs().sum() > 0


def test_hotspot_config_requires_second_output_channel() -> None:
    cfg = C.ExpConfig()
    cfg.train.loss_hotspot_bce_lambda = 0.2
    cfg.model.out_dim = 1
    with pytest.raises(ValueError, match="out_dim=2"):
        C.validate_features(cfg)


def test_pinball_auxiliary_keeps_scalar_output_contract() -> None:
    cfg = C.ExpConfig()
    cfg.train.loss_pinball_lambda = 0.2
    cfg.train.pinball_quantile = 0.9
    cfg.model.out_dim = 1
    C.validate_features(cfg)
