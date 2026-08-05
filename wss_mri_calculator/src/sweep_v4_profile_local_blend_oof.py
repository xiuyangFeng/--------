#!/usr/bin/env python3
"""Select a train-only blend of profile-tail and profile-local OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sweep_v4_depth3_overlay_oof import summarize_prediction
from validate_v4_high_tail_holdout import pooled_high_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-predictions", required=True)
    parser.add_argument("--profile-key", default="tail_w10_peak10_s80_oof")
    parser.add_argument("--local-predictions", required=True)
    parser.add_argument("--local-key", default="local_tail_safe_oof")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    args = parser.parse_args()

    profile_npz = np.load(args.profile_predictions, allow_pickle=False)
    local_npz = np.load(args.local_predictions, allow_pickle=False)
    groups = np.asarray(profile_npz["point_case"])
    truth = np.asarray(profile_npz["truth_mag"], dtype=np.float64)
    if not np.array_equal(groups, np.asarray(local_npz["point_case"])):
        raise RuntimeError("profile and local OOF point order differs")
    profile = np.asarray(profile_npz[args.profile_key], dtype=np.float64)
    local = np.asarray(local_npz[args.local_key], dtype=np.float64)

    candidates = []
    predictions = {}
    for local_weight in np.linspace(0.0, 1.0, 21):
        prediction = (1.0 - local_weight) * profile + local_weight * local
        case_balanced = summarize_prediction(truth, prediction, groups)
        pooled = pooled_high_metrics(truth, prediction, groups)
        row = {
            "local_weight": float(local_weight),
            "profile_weight": float(1.0 - local_weight),
            "case_balanced": case_balanced,
            "pooled_high_wss": pooled,
        }
        candidates.append(row)
        predictions[float(local_weight)] = prediction

    eligible = [
        row
        for row in candidates
        if row["case_balanced"]["raw_r2"]["mean"] >= 0.9600
        and row["case_balanced"]["peak_underestimate_fraction"]["mean"] <= 0.10
        and row["pooled_high_wss"]["r2"] >= 0.90
        and row["pooled_high_wss"]["mean_underestimate_fraction"] <= 0.10
    ]
    if not eligible:
        raise RuntimeError("no blend passed the fixed train-only constraints")

    def key(row: dict) -> tuple[float, ...]:
        gates = row["case_balanced"]["target_gates"]
        return (
            float(row["case_balanced"]["high_r2"]["mean"]),
            float(gates["cases_high_r2_at_least_0p9"]),
            float(gates["cases_meeting_both"]),
            float(gates["cases_peak_underestimate_at_most_10pct"]),
            float(row["case_balanced"]["raw_r2"]["mean"]),
        )

    selected = max(eligible, key=key)
    ranked = sorted(eligible, key=key, reverse=True)
    selected_prediction = predictions[selected["local_weight"]]
    payload = {
        "status": "train_only_profile_local_blend_selection",
        "profile_predictions": str(Path(args.profile_predictions).resolve()),
        "profile_key": args.profile_key,
        "local_predictions": str(Path(args.local_predictions).resolve()),
        "local_key": args.local_key,
        "selection_constraints": {
            "overall_raw_r2_mean_at_least": 0.9600,
            "case_peak_underestimate_mean_at_most": 0.10,
            "pooled_high_r2_at_least": 0.90,
            "pooled_mean_underestimate_at_most": 0.10,
        },
        "selected": selected,
        "eligible": ranked,
        "all_candidates": candidates,
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
            truth_mag=truth.astype(np.float32),
            profile_prediction=profile.astype(np.float32),
            local_prediction=local.astype(np.float32),
            selected_prediction=selected_prediction.astype(np.float32),
        )
    print(json.dumps(selected, indent=2, ensure_ascii=False), flush=True)
    print(output)


if __name__ == "__main__":
    main()
