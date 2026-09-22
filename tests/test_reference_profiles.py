"""Reference sidecar: build, bind to a release, and evaluate against a result summary."""
from __future__ import annotations

import json

import pytest

from wss_deploy.build_reference_profiles import build, population_protocol
from wss_deploy.infer import load_reference_sidecar
from wss_deploy.reference import evaluate


def _collected():
    cases = [f"AG/fast/CASE_{i}" for i in range(6)]
    return {"train_cases": cases, "oof_p99_pa": {c: 10.0 + 2 * i for i, c in enumerate(cases)},
            "cfd_p99_pa": {c: 11.0 + 2 * i for i, c in enumerate(cases)},
            "geometry": {"主动脉": {"length_mm": [100, 300], "radius_min_mm": [5, 12], "radius_median_mm": [8, 16], "radius_max_mm": [10, 40]}},
            "validation": {"n_cases": 6, "spearman_oof_vs_cfd_p99": 0.9, "median_ratio_oof_over_cfd_p99": 0.95, "log_ratio_sd": 0.2,
                           "oof_p99_pa_p10_p50_p90": [11, 15, 19], "cfd_p99_pa_p10_p50_p90": [12, 16, 20],
                           "fold_runs": {}, "cascade_pack": "test", "seed": 1234},
            "split_version": "test-split", "split_sha256": "0" * 64}


def _meta(p99=15.0, aorta_radius_max=20.0, spacing=0.5):
    protocol = dict(population_protocol(), top5_connectivity_radius_mm=2 * spacing)  # per-case volatile key present
    return {"release": "rel-x", "model_release": {"registry_id": "rel-x"}, "statistics_protocol": protocol,
            "fields": {"wss": {"units": "Pa", "location": "wall", "kind": "scalar", "components": 1}},
            "time_axis": [{"index": 0, "step": 1162, "time_s": 0.21, "label": "peak_systole"}],
            "peak": {"p99_pa": p99}, "cloud": {"spacing_mm": spacing},
            "geometry": {"主动脉": {"length_mm": 200, "radius_min_mm": 7, "radius_median_mm": 10, "radius_max_mm": aorta_radius_max}}}


def test_build_binds_release_and_evaluate_ranks_and_checks_ranges():
    profile = build("rel-x", _collected(), today="2026-09-20")
    info = {"release": "rel-x", **{k: profile[k] for k in ("geometry_reference", "population_reference")}}
    result = evaluate(_meta(p99=15.0), info)
    assert result["status"] == "pass" and len(result["checks"]) == 5
    pop = result["population"]
    assert pop["status"] == "pass" and pop["reference_count"] == 6
    assert pop["percentile"] == pytest.approx(50.0)  # values 10,12,14,16,18,20: three below 15, none equal
    assert pop["caveats"] and pop["reference_support"]
    review = evaluate(_meta(aorta_radius_max=60.0), info)
    assert review["status"] == "review"
    assert [c["path"] for c in review["checks"] if c["status"] == "review"] == ["geometry.主动脉.radius_max_mm"]


def test_volatile_protocol_keys_do_not_block_population_matching():
    profile = build("rel-x", _collected(), today="2026-09-20")
    info = {"release": "rel-x", "population_reference": profile["population_reference"]}
    meta = _meta(spacing=0.42)  # different connectivity radius than any other case
    assert evaluate(meta, info)["population"]["status"] == "pass"
    meta["statistics_protocol"]["quantile_method"] = "nearest"
    assert evaluate(meta, info)["population"]["status"] == "unknown"


def test_sidecar_must_bind_to_the_loading_release(tmp_path):
    profile = build("rel-x", _collected(), today="2026-09-20")
    (tmp_path / "reference.json").write_text(json.dumps(profile), encoding="utf-8")
    info = {"release": "rel-x"}
    digest = load_reference_sidecar(tmp_path, "rel-x", info)
    assert digest and len(digest) == 64 and "population_reference" in info and "geometry_reference" in info
    with pytest.raises(ValueError):
        load_reference_sidecar(tmp_path, "rel-other", {"release": "rel-other"})
    assert load_reference_sidecar(tmp_path / "missing", "rel-x", {}) is None
