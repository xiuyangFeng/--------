#!/usr/bin/env python3
"""Audit whether high-WSS failures are scale, ordering, or resolution limited."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from batch_validate_cfd import summarize
from calibrate_v4_high_tail_oof import case_tail_metrics, within_case_rank
from calibrate_v4_oof import EPS, load_cache


def magnitude(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.linalg.norm(array, axis=1) if array.ndim == 2 else array


def r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    variance = float(np.sum((truth - np.mean(truth)) ** 2))
    return 1.0 - float(np.sum((truth - prediction) ** 2)) / max(variance, EPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prediction-key", required=True)
    parser.add_argument("--profile-cache-dir", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    source = np.load(args.predictions, allow_pickle=False)
    groups = np.asarray(source["point_case"])
    truth = np.asarray(source["truth_mag"], dtype=np.float64)
    prediction = magnitude(source[args.prediction_key])
    profile = load_cache(Path(args.profile_cache_dir))
    profile_groups_full = np.asarray(profile["point_case"])
    keep = np.isin(profile_groups_full, np.unique(groups))
    profile_groups = profile_groups_full[keep]
    if not np.array_equal(groups, profile_groups):
        raise RuntimeError("prediction and profile-cache point order differ")

    diagnostic_keys = (
        "depth3__min_cell_depth_mm",
        "depth3__depth_p50_mm",
        "depth3__depth_span_mm",
        "depth3__depth_coefficient_2_ratio",
        "depth3__depth_coefficient_3_ratio",
        "depth3__fit_residual_nrmse",
        "depth3__fit_gradient_spread",
        "depth3__cv_score",
        "depth3__condition",
    )
    diagnostics = {
        key: np.asarray(profile[key], dtype=np.float64)[keep]
        for key in diagnostic_keys
    }
    rows = []
    for case in np.unique(groups):
        mask = groups == case
        y_all = truth[mask]
        p_all = prediction[mask]
        high = y_all >= np.quantile(y_all, 0.90)
        y = y_all[high]
        p = p_all[high]
        alpha = float(np.dot(y, p) / max(float(np.dot(p, p)), EPS))
        design = np.column_stack([p, np.ones(len(p))])
        affine_coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
        affine_prediction = design @ affine_coefficient
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic_prediction = isotonic.fit_transform(p, y)
        tail = case_tail_metrics(y_all, p_all)
        peak_index = int(np.argmax(y_all))
        prediction_rank = within_case_rank(p_all, np.zeros(len(p_all), dtype=int))
        row = {
            "canonical_id": str(case),
            "n_high": int(np.sum(high)),
            "observed_high_r2": tail["high_r2"],
            "oracle_scale_high_r2": r2(y, alpha * p),
            "oracle_affine_high_r2": r2(y, affine_prediction),
            "oracle_monotonic_high_r2": r2(y, isotonic_prediction),
            "observed_peak_underestimate_fraction": tail[
                "peak_underestimate_fraction"
            ],
            "truth_peak_prediction_rank": float(prediction_rank[peak_index]),
            "truth_peak_in_predicted_top10pct": bool(
                prediction_rank[peak_index] >= 0.90
            ),
            "top10_jaccard": tail["top10_jaccard"],
            "diagnostics_at_truth_peak": {},
        }
        for key, values in diagnostics.items():
            local = values[mask]
            finite = np.isfinite(local)
            value = float(local[peak_index]) if np.isfinite(local[peak_index]) else None
            percentile = None
            if value is not None and np.any(finite):
                percentile = float(np.mean(local[finite] <= value))
            row["diagnostics_at_truth_peak"][key] = {
                "value": value,
                "within_case_percentile": percentile,
            }
        rows.append(row)

    metric_names = (
        "observed_high_r2",
        "oracle_scale_high_r2",
        "oracle_affine_high_r2",
        "oracle_monotonic_high_r2",
        "observed_peak_underestimate_fraction",
        "truth_peak_prediction_rank",
        "top10_jaccard",
    )
    aggregate = {
        name: summarize([float(row[name]) for row in rows]) for name in metric_names
    }
    aggregate["counts"] = {
        "observed_high_r2_at_least_0p9": int(
            sum(row["observed_high_r2"] >= 0.90 for row in rows)
        ),
        "oracle_scale_high_r2_at_least_0p9": int(
            sum(row["oracle_scale_high_r2"] >= 0.90 for row in rows)
        ),
        "oracle_affine_high_r2_at_least_0p9": int(
            sum(row["oracle_affine_high_r2"] >= 0.90 for row in rows)
        ),
        "oracle_monotonic_high_r2_at_least_0p9": int(
            sum(row["oracle_monotonic_high_r2"] >= 0.90 for row in rows)
        ),
        "truth_peak_in_predicted_top10pct": int(
            sum(row["truth_peak_in_predicted_top10pct"] for row in rows)
        ),
        "peak_underestimate_at_most_10pct": int(
            sum(row["observed_peak_underestimate_fraction"] <= 0.10 for row in rows)
        ),
    }
    payload = {
        "status": "high_wss_identifiability_oracle_audit",
        "predictions": str(Path(args.predictions).resolve()),
        "prediction_key": args.prediction_key,
        "profile_cache_dir": str(Path(args.profile_cache_dir).resolve()),
        "n_cases": len(rows),
        "interpretation": {
            "oracle_scale": "upper bound if only a per-case multiplicative scale is missing",
            "oracle_affine": "upper bound if a per-case scale and offset are missing",
            "oracle_monotonic": "in-sample optimistic upper bound if local ordering is retained",
        },
        "aggregate": aggregate,
        "cases": rows,
    }
    output = Path(args.json_out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)
    print(output)


if __name__ == "__main__":
    main()
