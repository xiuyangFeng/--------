import json

from wss_deploy.comparison import compare_summaries


def _summary(*, input_sha="a" * 64, case_id="A", p99=9.0, max_pa=12.0,
             mean=4.0, branches=None, mapping=None, protocol=None,
             thresholds=(0.4, 4.0, 7.0), time_axis=None, params=None):
    return {
        "schema_version": "wss-deploy.summary/v1",
        "case_id": case_id,
        "input_sha256": input_sha,
        "model_release": {"release": "demo", "manifest_sha256": "f" * 64},
        "fields": {"wss": {"id": "wss", "units": "Pa", "location": "wall",
                             "kind": "scalar", "components": 1, "array_key": "wss_pa"}},
        "time_axis": time_axis or [{"index": 0, "step": 1162, "time_s": 0.21,
                                     "label": "peak_systole"}],
        "statistics_protocol": protocol or {
            "field": "fixed_peak_systolic_frame", "support": "prediction_point_cloud",
            "quantile_method": "linear", "p99_definition": "spatial p99",
            "position_definition": "maximum point", "area_method": "point_fraction",
            "area_label": "estimated area",
        },
        "run_parameters": params or {"smooth_mm": 1.0, "spacing_mm": 0.5},
        "mapping": mapping or {"3": "out-li", "4": "out-le", "5": "out-ri", "6": "out-re"},
        "wss_field_pa": {"mean": mean, "p99": p99, "max": max_pa,
                         "thresholds_pa": list(thresholds)},
        "peak": {"p99_pa": p99, "max_pa": max_pa},
        "per_branch": branches or {"left common iliac": {"wss_p99_pa": p99 - 1}},
    }


def test_same_input_multi_release_returns_scalar_rows_and_delta():
    left = _summary(p99=9.0)
    right = _summary(p99=10.0, max_pa=13.0, mean=4.5,
                     branches={"left common iliac": {"wss_p99_pa": 9.0}})
    right["model_release"] = {"release": "demo-v2", "manifest_sha256": "e" * 64}
    got = compare_summaries(left, right)

    assert got["compatible"] is True
    assert got["kind"] == "same_input"
    assert got["reasons"] == []
    by_id = {row["id"]: row for row in got["rows"]}
    assert by_id["peak.p99_pa"]["delta"] == 1.0
    assert by_id["peak.max_pa"]["delta"] == 1.0
    assert by_id["wss_field.mean_pa"]["delta"] == 0.5
    assert by_id["per_branch.left common iliac.p99_pa"]["delta"] == 1.0
    json.dumps(got, allow_nan=False)


def test_cross_case_comparison_allows_missing_branch_without_pointwise_claim():
    left = _summary(input_sha="a" * 64, case_id="A")
    right = _summary(input_sha="b" * 64, case_id="B",
                     branches={"right common iliac": {"wss_p99_pa": 6.0}})
    got = compare_summaries(left, right)

    assert got["compatible"] is True
    assert got["kind"] == "cross_case"
    branch_rows = [r for r in got["rows"] if r["scope"] == "per_branch"]
    assert {r["branch"] for r in branch_rows} == {"left common iliac", "right common iliac"}
    assert any(r["delta"] is None for r in branch_rows)
    assert all("point" not in str(row).lower() for row in got["rows"])


def test_same_input_mapping_mismatch_is_incompatible_and_withholds_deltas():
    left = _summary()
    right = _summary(mapping={"3": "out-le", "4": "out-li", "5": "out-ri", "6": "out-re"})
    got = compare_summaries(left, right)

    assert got["compatible"] is False
    assert got["kind"] == "same_input"
    assert any("mapping" in reason for reason in got["reasons"])
    assert all(row["delta"] is None for row in got["rows"])


def test_protocol_threshold_frame_and_preprocessing_mismatches_are_explicit():
    base = _summary()
    changed = _summary(time_axis=[{"index": 0, "step": 1180, "time_s": 0.22, "label": "other"}],
                       thresholds=(0.5, 4.0, 7.0), params={"smooth_mm": 2.0, "spacing_mm": 0.5})
    changed["statistics_protocol"] = {**base["statistics_protocol"], "quantile_method": "nearest"}
    got = compare_summaries(base, changed)

    assert got["compatible"] is False
    assert "time_axis differs" in got["reasons"]
    assert "statistics_protocol differs" in got["reasons"]
    assert "WSS thresholds_pa differ" in got["reasons"]
    assert "run_parameters smooth_mm/spacing_mm differ" in got["reasons"]


def test_missing_protocol_is_not_guessed_from_legacy_wss_values():
    left = _summary()
    right = _summary()
    right.pop("statistics_protocol")
    got = compare_summaries(left, right)

    assert got["compatible"] is False
    assert any("missing statistics_protocol" in reason for reason in got["reasons"])
    assert all(row["delta"] is None for row in got["rows"])


def test_legacy_metadata_and_model_frame_are_supported_when_contract_is_explicit():
    modern = _summary()
    legacy = _summary()
    # Move declarations into the legacy metadata envelope and retain the
    # historical explicit model_frame in place of time_axis.
    declarations = {key: legacy.pop(key) for key in (
        "fields", "statistics_protocol", "run_parameters", "wss_field_pa",
        "peak", "per_branch", "mapping")}
    legacy.pop("time_axis")
    legacy["model_frame"] = {"target": "peak_systole", "step": 1162,
                              "time_s": 0.21, "label": "peak_systole"}
    legacy["metadata"] = declarations
    got = compare_summaries(modern, legacy)

    assert got["compatible"] is True
    assert got["kind"] == "same_input"


def test_invalid_field_and_missing_identity_are_fail_closed():
    left = _summary(input_sha=None)
    right = _summary()
    left["fields"]["wss"]["kind"] = "vector"
    got = compare_summaries(left, right)

    assert got["compatible"] is False
    assert got["kind"] == "unknown"
    assert any("input_sha256" in reason for reason in got["reasons"])
    assert any("scalar wall field" in reason for reason in got["reasons"])
    json.dumps(got, allow_nan=False)
