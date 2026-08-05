#!/usr/bin/env python3
"""Apply a train-selected depth-3 overlay to fixed holdout or test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calculate_wss_cfd import wss_from_gradient
from calibrate_v4_high_tail_oof import smooth_gate, within_case_rank
from calibrate_v4_oof import EPS, load_cache
from sweep_v4_depth3_overlay_oof import summarize_prediction
from validate_v4_high_tail_holdout import pooled_high_metrics


def magnitude(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    return np.linalg.norm(values, axis=1) if values.ndim == 2 else values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--source-predictions", required=True)
    parser.add_argument("--prediction-key", required=True)
    parser.add_argument("--fusion-cache-dir", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    args = parser.parse_args()

    selection_payload = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
    selected = selection_payload["selected"]
    source = np.load(args.source_predictions, allow_pickle=False)
    groups = np.asarray(source["point_case"])
    truth = np.asarray(source["truth_mag"], dtype=np.float64)
    source_prediction = magnitude(source[args.prediction_key])
    source_base_vector = np.asarray(source["base_wss_vec"], dtype=np.float64)
    source_base_mag = np.linalg.norm(source_base_vector, axis=1)
    predicted_rank = within_case_rank(source_base_mag, groups)

    fusion = load_cache(Path(args.fusion_cache_dir))
    fusion_groups_full = np.asarray(fusion["point_case"])
    keep = np.isin(fusion_groups_full, np.unique(groups))
    fusion_groups = fusion_groups_full[keep]
    if not np.array_equal(groups, fusion_groups):
        raise RuntimeError("source predictions and selected fusion-cache points differ")
    physics_base = np.linalg.norm(
        wss_from_gradient(
            np.asarray(fusion["gradient_v4_base"], dtype=np.float64)[keep], "carreau"
        ),
        axis=1,
    )
    physics_fusion = np.linalg.norm(
        wss_from_gradient(
            np.asarray(fusion["gradient_v4"], dtype=np.float64)[keep], "carreau"
        ),
        axis=1,
    )
    physics_ratio = np.maximum(
        physics_fusion / np.maximum(physics_base, EPS), 1.0
    )
    gate = smooth_gate(predicted_rank, float(selected["gate_start"]))
    capped_ratio = np.clip(physics_ratio, 1.0, float(selected["ratio_cap"]))
    overlay = 1.0 + float(selected["strength"]) * gate * (capped_ratio - 1.0)
    prediction = source_prediction * overlay

    payload = {
        "status": "fixed_depth3_overlay_evaluation",
        "selection_json": str(Path(args.selection_json).resolve()),
        "source_predictions": str(Path(args.source_predictions).resolve()),
        "prediction_key": args.prediction_key,
        "fusion_cache_dir": str(Path(args.fusion_cache_dir).resolve()),
        "selected_parameters": selected,
        "n_cases": int(len(np.unique(groups))),
        "n_points": int(len(groups)),
        "source": {
            "case_balanced": summarize_prediction(truth, source_prediction, groups),
            "pooled_high_wss": pooled_high_metrics(truth, source_prediction, groups),
        },
        "overlay": {
            "case_balanced": summarize_prediction(truth, prediction, groups),
            "pooled_high_wss": pooled_high_metrics(truth, prediction, groups),
        },
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
            base_wss_vec=source_base_vector.astype(np.float32),
            source_prediction=source_prediction.astype(np.float32),
            physics_ratio=physics_ratio.astype(np.float32),
            overlay_ratio=overlay.astype(np.float32),
            overlay_prediction=prediction.astype(np.float32),
        )
    print(
        json.dumps(
            {
                "source": payload["source"],
                "overlay": payload["overlay"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(output)


if __name__ == "__main__":
    main()
