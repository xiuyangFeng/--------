"""Screen monotone case-relative tail expansion on grouped OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from batch_validate_cfd import summarize
from calculate_wss_cfd import evaluate
from calibrate_v4_high_tail_oof import case_tail_metrics, smooth_gate


EPS = 1e-8


def transform_case(
    prediction: np.ndarray,
    rank: np.ndarray,
    *,
    beta: float,
    tail_scale: float,
    gate_start: float,
) -> np.ndarray:
    q90 = float(np.quantile(prediction, 0.90))
    relative = np.maximum(prediction / max(q90, EPS), 1.0)
    gate = smooth_gate(rank, gate_start)
    factor = (1.0 + (tail_scale - 1.0) * gate) * relative ** (beta * gate)
    return prediction * np.clip(factor, 1.0, 1.75)


def evaluate_candidate(
    groups: np.ndarray,
    truth: np.ndarray,
    truth_vec: np.ndarray,
    base_vector: np.ndarray,
    prediction: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = []
    for case in np.unique(groups):
        mask = groups == case
        base_mag = np.linalg.norm(base_vector[mask], axis=1)
        ratio = prediction[mask] / np.maximum(base_mag, EPS)
        standard = evaluate(
            truth[mask], truth_vec[mask], base_vector[mask] * ratio[:, None]
        )
        tail = case_tail_metrics(truth[mask], prediction[mask])
        rows.append({"canonical_id": str(case), **standard, **tail})
    metrics = (
        "raw_r2",
        "mae_pa",
        "nrmse",
        "high_r2",
        "high_mae_pa",
        "high_nrmse",
        "high_bias_pa",
        "peak_underestimate_fraction",
        "peak_ratio",
        "top10_jaccard",
    )
    result = {metric: summarize([row[metric] for row in rows]) for metric in metrics}
    peak_under = np.asarray([row["peak_underestimate_fraction"] for row in rows])
    high_r2 = np.asarray([row["high_r2"] for row in rows])
    result["peak_underestimate_fraction"]["max"] = float(np.max(peak_under))
    result["gates"] = {
        "cases_high_r2_at_least_0p9": int(np.sum(high_r2 >= 0.90)),
        "cases_peak_underestimate_at_most_10pct": int(np.sum(peak_under <= 0.10)),
        "all_cases_peak_underestimate_at_most_10pct": bool(np.max(peak_under) <= 0.10),
    }
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--source-method", default="current_safe_oof")
    args = parser.parse_args()

    with np.load(args.predictions, allow_pickle=False) as source:
        groups = np.asarray(source["point_case"])
        truth = np.asarray(source["truth_mag"], dtype=np.float64)
        truth_vec = np.asarray(source["truth_vec"], dtype=np.float64)
        base_vector = np.asarray(source["base_wss_vec"], dtype=np.float64)
        rank = np.asarray(source["case_pred_rank"], dtype=np.float64)
        current = np.asarray(source[args.source_method], dtype=np.float64)

    candidates = []
    case_predictions: dict[str, np.ndarray] = {}
    for gate_start in (0.75, 0.80, 0.85):
        for beta in (0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24):
            for tail_scale in (1.00, 1.025, 1.05, 1.075, 1.10):
                name = f"g{gate_start:.2f}_b{beta:.2f}_s{tail_scale:.3f}"
                transformed = np.zeros_like(current)
                for case in np.unique(groups):
                    mask = groups == case
                    transformed[mask] = transform_case(
                        current[mask],
                        rank[mask],
                        beta=beta,
                        tail_scale=tail_scale,
                        gate_start=gate_start,
                    )
                aggregate, _ = evaluate_candidate(
                    groups, truth, truth_vec, base_vector, transformed
                )
                candidates.append(
                    {
                        "name": name,
                        "gate_start": gate_start,
                        "beta": beta,
                        "tail_scale": tail_scale,
                        "aggregate": aggregate,
                    }
                )
                case_predictions[name] = transformed

    baseline = next(
        row
        for row in candidates
        if row["gate_start"] == 0.80
        and row["beta"] == 0.0
        and row["tail_scale"] == 1.0
    )
    baseline_raw = float(baseline["aggregate"]["raw_r2"]["mean"])
    eligible = [
        row
        for row in candidates
        if float(row["aggregate"]["raw_r2"]["mean"]) >= baseline_raw - 0.003
    ]
    eligible.sort(
        key=lambda row: (
            float(row["aggregate"]["high_r2"]["mean"]),
            -float(row["aggregate"]["peak_underestimate_fraction"]["p95"]),
        ),
        reverse=True,
    )
    selected = eligible[0]
    selected_prediction = case_predictions[str(selected["name"])]
    selected_aggregate, selected_cases = evaluate_candidate(
        groups, truth, truth_vec, base_vector, selected_prediction
    )
    payload = {
        "status": "train_only_oof_monotone_tail_screen",
        "source_predictions": str(Path(args.predictions).resolve()),
        "source_method": args.source_method,
        "selection_rule": (
            "maximize case-balanced high-WSS R2 subject to mean overall raw R2 "
            "dropping by no more than 0.003 from current-safe OOF"
        ),
        "baseline": baseline,
        "selected": {**selected, "aggregate": selected_aggregate},
        "selected_cases": selected_cases,
        "top20": eligible[:20],
        "n_candidates": len(candidates),
    }
    output = Path(args.json_out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline": {
                    "raw_r2": baseline["aggregate"]["raw_r2"]["mean"],
                    "high_r2": baseline["aggregate"]["high_r2"]["mean"],
                    "peak_under_p95": baseline["aggregate"]["peak_underestimate_fraction"]["p95"],
                },
                "selected": {
                    "name": selected["name"],
                    "raw_r2": selected_aggregate["raw_r2"]["mean"],
                    "high_r2": selected_aggregate["high_r2"]["mean"],
                    "high_nrmse": selected_aggregate["high_nrmse"]["mean"],
                    "peak_under_p95": selected_aggregate["peak_underestimate_fraction"]["p95"],
                    "peak_under_max": selected_aggregate["peak_underestimate_fraction"]["max"],
                    "gates": selected_aggregate["gates"],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
