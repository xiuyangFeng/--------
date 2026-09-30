import numpy as np
import torch

from training_wss_min.velocity_phase_objectives import (case_phase_mean, direction_loss, direction_coverage,
                                                       velocity_objective)
from training_wss_min.velocity_phase_metrics import case_metrics


def test_case_phase_mean_preserves_both_weighting_levels():
    values = torch.tensor([[1., 10.], [3., 30.], [5., 50.]])
    weights = torch.tensor([1., 3., 100.])
    batch = torch.tensor([0, 0, 1])
    multiplier = torch.tensor([[2., 1.], [1., 2.], [2., 2.]])
    # case0 phase0=(2+9)/5; phase1=(10+180)/7; case1=5/50.
    expected = ((11/5 + 190/7)/2 + 27.5)/2
    assert torch.allclose(case_phase_mean(values, weights, batch, multiplier), torch.tensor(expected))


def test_zero_prediction_direction_has_finite_nonzero_gradient():
    pred = torch.zeros(2, 2, 3, requires_grad=True)
    y = torch.tensor([[[.1, 0., 0.], [0., 0., 0.]], [[0., .1, 0.], [0., 0., 0.]]])
    loss = direction_loss(pred, y, torch.ones(2), torch.zeros(2, dtype=torch.long))
    assert float(loss) == .5  # empty second phase counts zero rather than disappearing.
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    assert pred.grad[:, 1].abs().sum() == 0


def test_direction_coverage_counts_empty_case_phases_and_volume():
    truth = torch.zeros(3, 2, 3)
    truth[0, 0, 0] = .1
    truth[2, :, 0] = .1
    result = direction_coverage(truth, torch.tensor([1., 3., 100.]), torch.tensor([0, 0, 1]))
    assert result == {"eligible_volume_fraction": .5625, "empty_case_phases": 1,
                      "total_case_phases": 4}


def test_d2_tail_is_phase_local_and_uses_original_z_error():
    z = torch.tensor([[[1., 1., 1.], [2., 2., 2.]], [[3., 3., 3.], [4., 4., 4.]]])
    q = {"y": torch.zeros_like(z), "y_raw": torch.tensor([[[.2, 0, 0], [.1, 0, 0]],
                                                            [[.1, 0, 0], [.3, 0, 0]]]),
         "weight": torch.ones(2), "batch": torch.zeros(2, dtype=torch.long)}
    loss, parts = velocity_objective(z, q, {}, "D2", q80=torch.tensor([.15, .2]))
    assert np.isclose(float(loss), ((2+9)/3 + (4+32)/3)/2)
    assert np.isclose(float(parts["z_mse"]), 7.5)


def test_anatomical_metrics_preserve_original_vector_results():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(6, 80, 3))*.1
    p = y.copy()
    p[..., 0] *= -1
    w = np.arange(1, 7, dtype=float)
    frames = np.tile(np.eye(3), (6, 1, 1))
    met = case_metrics(y, p, w, frames, np.ones(6, bool))
    assert met["anatomy"]["cycle"]["transverse_vector_rmse_m_s"] == 0
    assert met["anatomy"]["cycle"]["axial_rmse_m_s"] > 0
    assert np.isclose(met["error_decomposition"]["cycle"]["vector_mse_m2_s2"],
                      3*met["cycle"]["mse"])
    assert abs(met["error_decomposition"]["cycle"]["speed_mse_m2_s2"]) < 1e-12
