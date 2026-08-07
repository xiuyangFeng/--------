import numpy as np

from training_wss_min import metrics as M


def test_linear_fit_r2_is_one_for_affine_biased_prediction():
    truth = np.linspace(1.0, 9.0, 17)
    prediction = 2.5 * truth - 3.0

    fit = M.linear_fit_metrics(truth, prediction)

    assert np.isclose(fit["r2_linear_fit"], 1.0)
    assert np.isclose(fit["linear_fit_slope"], 2.5)
    assert np.isclose(fit["linear_fit_intercept"], -3.0)
    assert M.r2_score(truth, prediction) < 0.0


def test_linear_fit_r2_matches_squared_pearson_correlation():
    truth = np.asarray([0.2, 1.1, 2.0, 3.7, 5.4, 8.0])
    prediction = np.asarray([0.6, 1.7, 1.4, 4.8, 5.0, 7.1])

    fit = M.linear_fit_metrics(truth, prediction)
    expected = float(np.corrcoef(truth, prediction)[0, 1] ** 2)

    assert np.isclose(fit["r2_linear_fit"], expected)


def test_casebalanced_fit_is_invariant_to_within_case_duplication():
    truth = [np.asarray([0.0, 1.0]), np.asarray([2.0, 4.0, 7.0])]
    prediction = [np.asarray([0.4, 1.2]), np.asarray([2.5, 3.7, 6.1])]
    base = M.casebalanced_linear_fit_metrics(truth, prediction)

    duplicated = M.casebalanced_linear_fit_metrics(
        [np.tile(truth[0], 20), truth[1]],
        [np.tile(prediction[0], 20), prediction[1]],
    )

    for key in ("r2_linear_fit", "linear_fit_slope", "linear_fit_intercept"):
        assert np.isclose(base[key], duplicated[key])


def test_constant_prediction_has_defined_line_but_undefined_fit_r2():
    fit = M.linear_fit_metrics(np.arange(5.0), np.full(5, 3.0))

    assert np.isnan(fit["r2_linear_fit"])
    assert np.isclose(fit["linear_fit_slope"], 0.0)
    assert np.isclose(fit["linear_fit_intercept"], 3.0)
