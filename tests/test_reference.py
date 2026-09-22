import copy
import json

import pytest

from wss_deploy.reference import evaluate, evaluate_population


def fixture():
    meta = {"model_release": {"registry_id": "demo-v1"},
            "cloud": {"spacing_mm": .5, "surface_variation_median": .002},
            "geometry": {"主动脉": {"radius_min_mm": 5., "length_mm": 150.},
                         "左髂外": {"radius_min_mm": 2., "length_mm": 50.}},
            "fields": {"wss": {"units": "Pa", "location": "wall", "kind": "scalar"}},
            "time_axis": [{"index": 0, "step": 1162, "time_s": .21}],
            "statistics_protocol": {"field": "fixed_peak_systolic_frame",
                                    "support": "prediction_point_cloud", "quantile_method": "linear"},
            "peak": {"p99_pa": 4.}}
    release = {"release": "demo-v1", "geometry_reference": {
        "schema_version": "wss-deploy.geometry-reference/v1",
        "id": "geometry-1", "status": "validated", "release": "demo-v1",
        "reference_set": "anatomy-reference-1",
        "bounds": [{"path": "cloud.spacing_mm", "units": "mm", "min": .4, "max": .6},
                   {"path": "cloud.surface_variation_median", "units": "1", "min": 0., "max": .01},
                   {"path": "geometry.*.radius_min_mm", "units": "mm", "min": 1., "max": 30.},
                   {"path": "geometry.主动脉.length_mm", "units": "mm", "min": 100., "max": 300.}]},
        "population_reference": {
            "schema_version": "wss-deploy.population-reference/v1",
            "id": "population-1", "status": "validated", "release": "demo-v1",
            "reference_set": "cv3-oof-reference-1", "source": "cv3_oof_predictions",
            "fold_count": 3, "metric": "p99_pa", "field": copy.deepcopy(meta["fields"]["wss"]),
            "time_axis": copy.deepcopy(meta["time_axis"]),
            "statistics_protocol": copy.deepcopy(meta["statistics_protocol"]),
            "values_pa": [2., 4., 4., 8.]}}
    return meta, release


def test_reference_is_optional_and_does_not_invent_a_training_distribution():
    got = evaluate({}, {})
    assert got["status"] == "unknown" and got["checks"] == []
    assert got["population"]["status"] == "unknown"
    assert got["population"]["percentile"] is None
    json.dumps(got, allow_nan=False)


def test_reviewed_geometry_ranges_check_all_branches_and_do_not_mutate_inputs():
    meta, release = fixture()
    before = copy.deepcopy((meta, release))
    got = evaluate(meta, release)
    assert got["status"] == "pass"
    assert len(got["checks"]) == 5
    assert all(row["status"] == "pass" for row in got["checks"])
    assert "校准概率" in got["note"]
    assert (meta, release) == before


def test_out_of_range_is_review_and_missing_measurement_is_unknown():
    meta, release = fixture()
    meta["geometry"]["左髂外"]["radius_min_mm"] = .5
    meta["cloud"]["spacing_mm"] = float("nan")
    got = evaluate(meta, release)
    assert got["status"] == "review"
    checks = {row["path"]: row for row in got["checks"]}
    assert checks["geometry.左髂外.radius_min_mm"]["status"] == "review"
    assert checks["cloud.spacing_mm"]["status"] == "unknown"
    assert checks["cloud.spacing_mm"]["value"] is None
    json.dumps(got, allow_nan=False)
    meta["geometry"]["左髂外"]["radius_min_mm"] = 2.
    assert evaluate(meta, release)["status"] == "unknown"


@pytest.mark.parametrize("patch", [
    {"path": "cloud.spacing_mm", "units": "cm", "min": .4, "max": .6},
    {"path": "geometry.*.wss_p99_pa", "units": "Pa", "min": 0., "max": 30.},
    {"path": "cloud.spacing_mm", "units": "mm", "min": .6, "max": .4},
    {"path": "cloud.spacing_mm", "units": "mm", "min": .4, "max": float("inf")},
    {"path": "cloud.spacing_mm", "units": "mm", "min": True, "max": .6},
])
def test_invalid_geometry_profile_never_partially_passes(patch):
    meta, release = fixture()
    release["geometry_reference"]["bounds"][0] = patch
    got = evaluate(meta, release)
    assert got["status"] == "unknown" and got["checks"] == []
    json.dumps(got, allow_nan=False)


@pytest.mark.parametrize("key,value", [("status", "draft"), ("release", "other"),
                                       ("schema_version", "other"), ("reference_set", "")])
def test_unverified_or_wrong_release_profiles_remain_unknown(key, value):
    meta, release = fixture()
    for name in ("geometry_reference", "population_reference"):
        release[name][key] = value
    got = evaluate(meta, release)
    assert got["status"] == got["population"]["status"] == "unknown"


def test_result_from_another_release_cannot_use_current_reference():
    meta, release = fixture()
    meta["model_release"]["registry_id"] = "old-version"
    got = evaluate(meta, release)
    assert got["status"] == got["population"]["status"] == "unknown"


def test_population_rank_uses_empirical_midrank_and_bound_protocol():
    meta, release = fixture()
    got = evaluate_population(meta, release)
    assert got["status"] == "pass"
    assert got["percentile"] == 50.0 and got["reference_count"] == 4
    meta["peak"]["p99_pa"] = 9.
    assert evaluate_population(meta, release)["percentile"] == 100.
    meta["peak"]["p99_pa"] = 1.
    assert evaluate_population(meta, release)["percentile"] == 0.


@pytest.mark.parametrize("patch", [
    {"source": "CFD peak-frame wall WSS, train136"},
    {"source": "in_sample_predictions"},
    {"fold_count": 5},
    {"metric": "max_pa"},
    {"field": {"units": "kPa", "location": "wall", "kind": "scalar"}},
    {"time_axis": [{"index": 0, "step": 1163, "time_s": .22}]},
    {"statistics_protocol": {"support": "interpolated_wall_surface"}},
    {"values_pa": [1., float("nan"), 4.]},
    {"values_pa": [1., True, 4.]},
    {"values_pa": [4.]},
])
def test_population_withholds_rank_for_mismatched_or_invalid_reference(patch):
    meta, release = fixture()
    release["population_reference"].update(patch)
    got = evaluate_population(meta, release)
    assert got["status"] == "unknown" and got["percentile"] is None
    json.dumps(got, allow_nan=False)


def test_population_withholds_rank_when_current_p99_is_invalid():
    meta, release = fixture()
    meta["peak"]["p99_pa"] = float("inf")
    assert evaluate_population(meta, release)["percentile"] is None


def test_missing_wildcard_geometry_does_not_pass_vacuously():
    meta, release = fixture()
    meta["geometry"] = {}
    got = evaluate(meta, release)
    assert got["status"] == "unknown"
    assert any(row["path"] == "geometry.*.radius_min_mm" and row["status"] == "unknown"
               for row in got["checks"])
