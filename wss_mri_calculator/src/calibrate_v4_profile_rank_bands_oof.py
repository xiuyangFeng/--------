#!/usr/bin/env python3
"""Grouped OOF study of case-predicted high-tail rank-band shape calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibrate_v4_high_tail_oof import within_case_rank
from calibrate_v4_oof import EPS, build_case_features, load_cache
from sweep_v4_depth3_overlay_oof import summarize_prediction
from v4_profile_features import build_profile_features
from validate_v4_high_tail_holdout import pooled_high_metrics


BANDS = ((0.80, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 1.001))
BAND_CENTERS = np.asarray([0.85, 0.925, 0.965, 0.99], dtype=np.float64)


def case_band_targets(
    cases: np.ndarray,
    groups: np.ndarray,
    truth: np.ndarray,
    source: np.ndarray,
    rank: np.ndarray,
) -> np.ndarray:
    targets = np.ones((len(cases), len(BANDS)), dtype=np.float64)
    for row, case in enumerate(cases):
        local = groups == case
        for column, (lower, upper) in enumerate(BANDS):
            band = local & (rank >= lower) & (rank < upper)
            if not np.any(band):
                continue
            targets[row, column] = float(
                np.dot(truth[band], source[band])
                / max(float(np.dot(source[band], source[band])), EPS)
            )
    return np.clip(targets, 0.65, 1.80)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-predictions", required=True)
    parser.add_argument("--source-key", default="tail_w10_peak10_s80_oof")
    parser.add_argument("--profile-cache-dir", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    source_npz = np.load(args.source_predictions, allow_pickle=False)
    groups = np.asarray(source_npz["point_case"])
    cohorts = np.asarray(source_npz["point_cohort"])
    truth = np.asarray(source_npz["truth_mag"], dtype=np.float64)
    source = np.asarray(source_npz[args.source_key], dtype=np.float64)
    rank = within_case_rank(source, groups)

    data = load_cache(Path(args.profile_cache_dir))
    if not np.array_equal(groups, np.asarray(data["point_case"])):
        raise RuntimeError("source predictions and profile cache point order differ")
    x, feature_names = build_profile_features(data)
    cases, case_x = build_case_features(x, groups)
    case_cohorts = np.asarray([cohorts[groups == case][0] for case in cases])
    band_targets = case_band_targets(cases, groups, truth, source, rank)

    strengths = (0.25, 0.50, 0.75, 1.00)
    predictions = {strength: source.copy() for strength in strengths}
    predicted_band_alpha = np.ones_like(band_targets)
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_rows = []
    for fold, (train_case_index, valid_case_index) in enumerate(
        splitter.split(case_x, case_cohorts), start=1
    ):
        models = []
        for band_index in range(len(BANDS)):
            model = make_pipeline(
                StandardScaler(),
                RidgeCV(alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0)),
            )
            model.fit(
                case_x[train_case_index],
                np.log(band_targets[train_case_index, band_index]),
            )
            models.append(model)
            predicted_band_alpha[valid_case_index, band_index] = np.clip(
                np.exp(model.predict(case_x[valid_case_index])), 0.75, 1.50
            )

        for case_row in valid_case_index:
            case = cases[case_row]
            local = groups == case
            anchors_x = np.concatenate([[0.80], BAND_CENTERS])
            anchors_y = np.concatenate([[1.0], predicted_band_alpha[case_row]])
            curve = np.interp(rank[local], anchors_x, anchors_y, left=1.0, right=anchors_y[-1])
            for strength in strengths:
                predictions[strength][local] = source[local] * (
                    1.0 + strength * (curve - 1.0)
                )
        fold_rows.append(
            {
                "fold": fold,
                "train_cases": int(len(train_case_index)),
                "valid_cases": int(len(valid_case_index)),
            }
        )
        print(
            f"fold={fold}/{args.folds} train_cases={len(train_case_index)} "
            f"valid_cases={len(valid_case_index)}",
            flush=True,
        )

    candidates = []
    for strength in strengths:
        case_balanced = summarize_prediction(truth, predictions[strength], groups)
        pooled = pooled_high_metrics(truth, predictions[strength], groups)
        candidates.append(
            {
                "strength": strength,
                "case_balanced": case_balanced,
                "pooled_high_wss": pooled,
            }
        )
    baseline = {
        "strength": 0.0,
        "case_balanced": summarize_prediction(truth, source, groups),
        "pooled_high_wss": pooled_high_metrics(truth, source, groups),
    }
    candidates.insert(0, baseline)
    eligible = [
        row
        for row in candidates
        if row["case_balanced"]["raw_r2"]["mean"] >= 0.9600
        and row["case_balanced"]["peak_underestimate_fraction"]["mean"] <= 0.10
        and row["pooled_high_wss"]["r2"] >= 0.90
        and row["pooled_high_wss"]["mean_underestimate_fraction"] <= 0.10
    ]

    def key(row: dict) -> tuple[float, ...]:
        gates = row["case_balanced"]["target_gates"]
        return (
            float(row["case_balanced"]["high_r2"]["mean"]),
            float(gates["cases_high_r2_at_least_0p9"]),
            float(gates["cases_meeting_both"]),
            float(row["case_balanced"]["raw_r2"]["mean"]),
        )

    selected = max(eligible, key=key)
    selected_prediction = source if selected["strength"] == 0 else predictions[selected["strength"]]
    payload = {
        "status": "grouped_oof_profile_rank_band_shape_calibration",
        "source_predictions": str(Path(args.source_predictions).resolve()),
        "source_key": args.source_key,
        "profile_cache_dir": str(Path(args.profile_cache_dir).resolve()),
        "feature_names": feature_names,
        "bands": [list(band) for band in BANDS],
        "band_centers": BAND_CENTERS.tolist(),
        "fold_rows": fold_rows,
        "selection_constraints": {
            "overall_raw_r2_mean_at_least": 0.9600,
            "case_peak_underestimate_mean_at_most": 0.10,
            "pooled_high_r2_at_least": 0.90,
            "pooled_mean_underestimate_at_most": 0.10,
        },
        "selected": selected,
        "candidates": candidates,
    }
    output = Path(args.json_out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    prediction_output = Path(args.predictions_out).resolve()
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    with prediction_output.open("wb") as handle:
        np.savez_compressed(
            handle,
            point_case=groups,
            point_cohort=cohorts,
            truth_mag=truth.astype(np.float32),
            source_prediction=source.astype(np.float32),
            predicted_band_alpha=predicted_band_alpha.astype(np.float32),
            selected_prediction=selected_prediction.astype(np.float32),
        )
    print(json.dumps(selected, indent=2, ensure_ascii=False), flush=True)
    print(output)


if __name__ == "__main__":
    main()
