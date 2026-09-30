"""Train-threshold tail diagnostics must expose spurious high-speed regions."""
import numpy as np
import pytest

from training_wss_min.velocity_phase_metrics import case_metrics


def metrics(truth_speed, pred_speed, threshold, weights):
    truth_speed, pred_speed = np.asarray(truth_speed), np.asarray(pred_speed)
    y = np.zeros((*truth_speed.shape, 3))
    p = np.zeros_like(y)
    y[..., 0], p[..., 0] = truth_speed, pred_speed
    frames = np.tile(np.eye(3), (len(y), 1, 1))
    return case_metrics(y, p, np.asarray(weights, float), frames, np.ones(len(y), bool),
                        tail_q80=np.asarray(threshold))


def test_predicted_tail_and_false_positives_use_full_weighted_domain():
    truth = np.repeat([[.3], [.1], [.05]], 80, axis=1)
    pred = np.repeat([[.3], [.3], [.05]], 80, axis=1)
    out = metrics(truth, pred, np.full(80, .2), [1, 2, 7])
    for segment in ("cycle", "peak", "trough", "decel", "early_trough", "late_trough"):
        tail = out["train_threshold_tail"][segment]
        assert tail["volume_phase_coverage"] == pytest.approx(.1)
        assert tail["predicted_volume_phase_coverage"] == pytest.approx(.3)
        assert tail["false_positive_volume_phase_coverage"] == pytest.approx(.2)
        assert tail["tail_precision"] == pytest.approx(1/3)
        assert tail["tail_recall"] == pytest.approx(1.)
        # Perfect predictions inside the true tail must not hide false highs.
        assert tail["vector_rmse_m_s"] == 0
        assert tail["pred_true_speed_ratio"] == pytest.approx(1.)


def test_threshold_remains_phase_local_and_absent_truth_tail_has_no_recall():
    truth = np.full((2, 80), .05)
    pred = np.full((2, 80), .15)
    threshold = np.r_[np.full(40, .1), np.full(40, .2)]
    out = metrics(truth, pred, threshold, [1, 4])["train_threshold_tail"]
    assert out["cycle"]["predicted_volume_phase_coverage"] == pytest.approx(.5)
    assert out["cycle"]["false_positive_volume_phase_coverage"] == pytest.approx(.5)
    assert out["peak"]["false_positive_volume_phase_coverage"] == pytest.approx(1.)
    assert out["trough"]["false_positive_volume_phase_coverage"] == pytest.approx(.25)
    assert out["cycle"]["tail_precision"] == 0
    assert out["cycle"]["tail_recall"] is None
    assert out["late_trough"]["tail_precision"] is None
    assert out["late_trough"]["false_positive_volume_phase_coverage"] == 0


def test_inclusive_threshold_and_no_predicted_tail_are_explicit():
    truth = np.full((2, 80), .2)
    pred = np.repeat([[.2], [.1]], 80, axis=1)
    tail = metrics(truth, pred, np.full(80, .2), [1, 3])["train_threshold_tail"]["cycle"]
    assert tail["volume_phase_coverage"] == pytest.approx(1.)
    assert tail["predicted_volume_phase_coverage"] == pytest.approx(.25)
    assert tail["tail_precision"] == pytest.approx(1.)
    assert tail["tail_recall"] == pytest.approx(.25)
    assert tail["false_positive_volume_phase_coverage"] == 0
    no_predictions = metrics(truth, np.zeros_like(pred), np.full(80, .2), [1, 3])
    tail = no_predictions["train_threshold_tail"]["cycle"]
    assert tail["tail_precision"] is None
    assert tail["tail_recall"] == 0
    assert tail["predicted_volume_phase_coverage"] == 0
