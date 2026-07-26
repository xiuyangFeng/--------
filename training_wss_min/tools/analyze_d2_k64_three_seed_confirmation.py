#!/usr/bin/env python3
"""Strict matched-seed analysis for M1/S2/S3 on the frozen mixed test36."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

from training_wss_min.tools.analyze_d2_k64_ilo_structure_results import flatten_metrics


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "d2_k64_ilo_structure_three_seed_confirmation_prepared.json"
RUNS = ROOT / "training_wss_min/runs/pointnetpp_d2_k64_ilo_structure/outputs"
OUT_JSON = PREFLIGHT / "d2_k64_ilo_structure_three_seed_confirmation_analysis.json"
OUT_SUMMARY = PREFLIGHT / "d2_k64_ilo_structure_three_seed_confirmation_summary.csv"
OUT_CASES = PREFLIGHT / "d2_k64_ilo_structure_three_seed_confirmation_paired_case_seed.csv"
METRICS = ("physical_r2_casebalanced", "physical_mae", "physical_rmse", "physical_r2_case_mean", "physical_r2_negative_cases", "high_wss_r2", "physical_p99_amplitude_ratio", "physical_top10_iou", "spearman_case_mean")
PAIRS = (("S2_minus_M1", "S2", "M1"), ("S3_minus_S2", "S3", "S2"), ("S3_minus_M1", "S3", "M1"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def values(metrics: dict) -> dict:
    test = metrics["test"]
    flat = flatten_metrics(test)
    groups = {key: {"r2": data["r2"], "mae": data["mae"], "rmse": data["rmse"]} for key, data in test["group_casebalanced"].items()}
    cases = {case: {"r2": data["overall"]["r2"], "mae": data["overall"]["mae"], "rmse": data["overall"]["rmse"], "high_wss_r2": data["high_wss"]["r2"], "top10_iou": data["hotspot"]["top10_iou"], "spearman": data["hotspot"]["spearman_all"], "p99_ratio": data["calibration"]["p99_pred_true_ratio"]} for case, data in test["per_case"].items()}
    return {"flat": flat, "groups": groups, "cases": cases}


def summary(xs: list[float]) -> dict:
    n = len(xs); mean = sum(xs) / n
    return {"values": xs, "mean": mean, "std": math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0, "min": min(xs), "max": max(xs), "positive": sum(x > 0 for x in xs), "negative": sum(x < 0 for x in xs), "zero": sum(x == 0 for x in xs)}


def bootstrap(xs: list[float], seed: int) -> dict:
    rng = random.Random(seed); n = len(xs)
    draws = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(20000))
    return {"mean": sum(xs) / n, "ci95_low": draws[500], "ci95_high": draws[19499], "wins": sum(x > 0 for x in xs), "losses": sum(x < 0 for x in xs), "ties": sum(x == 0 for x in xs), "n_cases": n}


def main() -> None:
    manifest = read(MANIFEST)
    rows = manifest["configs"]
    if len(rows) != 6:
        raise RuntimeError("expected 6 confirmation configs")
    source = {
        ("M1", 1234): RUNS / "m1_d2k64_mixed138_test36",
        ("S2", 1234): RUNS / "s2_d2k64_pnxr_mixed",
        ("S3", 1234): RUNS / "s3_d2k64_pnxr_geope_mixed",
    }
    config_sha = {}
    for row in rows:
        source[(row["arm"], row["seed"])] = Path(row["run_dir"])
        config_sha[(row["arm"], row["seed"])] = row["sha256"]
    records = {}
    for key, run in source.items():
        required = [run / "ckpt_best.pt", run / "ckpt_last.pt", run / "eval/ckpt_best/metrics.json", run / "eval/ckpt_last/metrics.json", run / "eval/best_last_comparison.json"]
        missing = [str(x) for x in required if not x.exists()]
        if missing: raise RuntimeError({"missing": missing})
        records[key] = {"run_dir": str(run), "best": values(read(run / "eval/ckpt_best/metrics.json")), "last": values(read(run / "eval/ckpt_last/metrics.json"))}
    result, summary_rows, case_rows = [], [], []
    for index, (name, treatment, control) in enumerate(PAIRS):
        per_seed = []
        for seed in (1234, 7, 2025):
            t, c = records[(treatment, seed)]["best"], records[(control, seed)]["best"]
            if set(t["cases"]) != set(c["cases"]): raise RuntimeError(f"unpaired case set: {name} seed={seed}")
            delta = {metric: t["flat"][metric] - c["flat"][metric] for metric in METRICS}
            group_delta = {group: {metric: t["groups"][group][metric] - c["groups"][group][metric] for metric in ("r2", "mae", "rmse")} for group in ("AG", "AAA", "AAA_ruputer", "AAA_unruputer", "ILO", "ILO_0", "ILO_1")}
            last_delta = {metric: records[(treatment, seed)]["last"]["flat"][metric] - records[(control, seed)]["last"]["flat"][metric] for metric in METRICS}
            per_seed.append({"seed": seed, "best_delta": delta, "last_delta": last_delta, "group_delta": group_delta})
            for case in sorted(t["cases"]):
                row = {"comparison": name, "seed": seed, "case": case}
                for metric in t["cases"][case]: row[f"delta_{metric}"] = t["cases"][case][metric] - c["cases"][case][metric]
                case_rows.append(row)
        aggregate = {metric: summary([row["best_delta"][metric] for row in per_seed]) for metric in METRICS}
        last = {metric: summary([row["last_delta"][metric] for row in per_seed]) for metric in METRICS}
        group = {g: {m: summary([row["group_delta"][g][m] for row in per_seed]) for m in ("r2", "mae", "rmse")} for g in per_seed[0]["group_delta"]}
        case_boot = {metric: bootstrap([row[f"delta_{metric}"] for row in case_rows if row["comparison"] == name], 20260723 + index) for metric in ("r2", "mae", "rmse", "high_wss_r2", "top10_iou", "spearman", "p99_ratio")}
        result.append({"comparison": name, "treatment": treatment, "control": control, "per_seed": per_seed, "best_seed_summary": aggregate, "last_seed_summary": last, "group_seed_summary": group, "paired_case_bootstrap": case_boot})
        for metric, stat in aggregate.items(): summary_rows.append({"comparison": name, "checkpoint": "best", "scope": "overall", "metric": metric, **{k: stat[k] for k in ("mean", "std", "min", "max", "positive", "negative", "zero")}})
        for metric, stat in last.items(): summary_rows.append({"comparison": name, "checkpoint": "last", "scope": "overall", "metric": metric, **{k: stat[k] for k in ("mean", "std", "min", "max", "positive", "negative", "zero")}})
        for g, stats in group.items():
            for metric, stat in stats.items(): summary_rows.append({"comparison": name, "checkpoint": "best", "scope": g, "metric": metric, **{k: stat[k] for k in ("mean", "std", "min", "max", "positive", "negative", "zero")}})
    s3s2, s3m1 = result[1], result[2]
    decision = {"s3_advances_to_drop_single_variable": bool(s3s2["best_seed_summary"]["physical_r2_casebalanced"]["mean"] > 0 and s3s2["best_seed_summary"]["physical_r2_casebalanced"]["positive"] >= 2 and s3m1["best_seed_summary"]["physical_r2_casebalanced"]["mean"] > 0), "rule_note": "Automated core-rule check only; safety guards require review of MAE/RMSE, high-WSS, AAA/ILO and case distribution.", "if_pass_next_configs": ["S3 + DropPath 0.05", "S3 + DropPath 0.10", "S3 + NeighborDrop 0.05"]}
    payload = {"schema_version": 1, "scope": "frozen mixed test36 historical engineering screening set; not a new independent external test set", "manifest": str(MANIFEST), "manifest_sha256": sha(MANIFEST), "seeds": [1234, 7, 2025], "records": {f"{arm}_s{seed}": {"run_dir": data["run_dir"]} for (arm, seed), data in records.items()}, "comparisons": result, "decision": decision}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison", "checkpoint", "scope", "metric", "mean", "std", "min", "max", "positive", "negative", "zero"]); writer.writeheader(); writer.writerows(summary_rows)
    with OUT_CASES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison", "seed", "case", "delta_r2", "delta_mae", "delta_rmse", "delta_high_wss_r2", "delta_top10_iou", "delta_spearman", "delta_p99_ratio"]); writer.writeheader(); writer.writerows(case_rows)
    print(json.dumps({"status": "passed", "json": str(OUT_JSON), "summary_csv": str(OUT_SUMMARY), "paired_case_seed_csv": str(OUT_CASES), "decision": decision}, ensure_ascii=False))


if __name__ == "__main__": main()
