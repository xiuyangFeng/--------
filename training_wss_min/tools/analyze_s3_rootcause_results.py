#!/usr/bin/env python3
"""Audit & summarize the 2026-07-23 S3-GEOPE root-cause matrix (12 arms).

Reuses the exact metric flattening / paired bootstrap from the D2-K64 analyzer
so every number lines up with the previous round.  Each arm is compared against
the S3-GEOPE parent at the SAME seed; three-seed arms also get a mean-over-seed
aggregate delta.  Per-domain (AG/AAA/ILO) case-balanced R2 is surfaced for the
negative-transfer verdict.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from training_wss_min.tools.analyze_d2_k64_ilo_structure_results import (
    AGG_KEYS,
    CASE_METRICS,
    bootstrap_mean,
    finite_numbers,
    flatten_metrics,
    history_summary,
    metric_partition,
    per_case,
    sha256,
)

REPO = Path(__file__).resolve().parents[2]
PREFLIGHT = REPO / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "s3_rootcause_matrix_prepared.json"
D2_ROOT = REPO / "training_wss_min/runs/pointnetpp_d2_k64_ilo_structure/outputs"
OUT_JSON = PREFLIGHT / "s3_rootcause_results_analysis.json"
OUT_SUMMARY = PREFLIGHT / "s3_rootcause_results_summary.csv"
OUT_PAIRS = PREFLIGHT / "s3_rootcause_paired_case_stats.csv"
BOOTSTRAP_DRAWS = 20_000

# S3-GEOPE parent runs (three-seed confirmed), keyed by train seed.
PARENT = {
    1234: D2_ROOT / "s3_d2k64_pnxr_geope_mixed",
    7: D2_ROOT / "s3_d2k64_pnxr_geope_mixed_s7",
    2025: D2_ROOT / "s3_d2k64_pnxr_geope_mixed_s2025",
}
DOMAINS = ("AG", "AAA", "ILO")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_record(experiment_id: str, run_dir: Path, seed: int, variant: str,
                 config: Path | None = None, sha_expected: str | None = None) -> dict:
    errors: list[str] = []
    required = (
        "ckpt_best.pt", "ckpt_last.pt", "config.json",
        "eval/best_last_comparison.json", "eval/best_last_per_case.csv",
        "eval/ckpt_best/metrics.json", "eval/ckpt_best/per_case_metrics.csv",
        "eval/ckpt_last/metrics.json", "eval/ckpt_last/per_case_metrics.csv",
    )
    for rel in required:
        if not (run_dir / rel).exists():
            errors.append(f"missing:{rel}")
    best_part, best_metrics = metric_partition(run_dir / "eval/ckpt_best/metrics.json")
    last_part, last_metrics = metric_partition(run_dir / "eval/ckpt_last/metrics.json")
    if best_part != last_part:
        errors.append("best_last_partition_mismatch")
    finite_numbers(best_metrics, f"{experiment_id}.best", errors)
    finite_numbers(last_metrics, f"{experiment_id}.last", errors)
    if config is not None and sha_expected is not None and sha256(config) != sha_expected:
        errors.append("prepared_config_hash_changed")
    history = history_summary(run_dir, errors)
    best = flatten_metrics(best_metrics)
    last = flatten_metrics(last_metrics)
    domains = {d: best["groups"].get(d, {}).get("r2") for d in DOMAINS}
    return {
        "experiment_id": experiment_id, "variant": variant, "seed": seed,
        "run_dir": str(run_dir), "evaluation_partition": best_part,
        "history": history, "best": best, "last": last, "domains": domains,
        "last_minus_best": {k: last[k] - best[k]
                            for k in ("physical_r2_casebalanced", "physical_mae", "physical_rmse")},
        "per_case": per_case(best_metrics),
        "integrity": "passed" if not errors else "failed", "integrity_errors": errors,
    }


def comparison(name: str, control: dict, treatment: dict, seed: int) -> dict:
    if set(control["per_case"]) != set(treatment["per_case"]):
        raise RuntimeError(f"{name}: paired case sets differ")
    agg = {k: treatment["best"][k] - control["best"][k] for k in AGG_KEYS if k != "parameters"}
    dom = {d: (None if treatment["domains"][d] is None or control["domains"][d] is None
               else treatment["domains"][d] - control["domains"][d]) for d in DOMAINS}
    cases = sorted(control["per_case"])
    case_stats = {}
    for offset, metric in enumerate(CASE_METRICS):
        values = [treatment["per_case"][c][metric] - control["per_case"][c][metric] for c in cases]
        case_stats[metric] = bootstrap_mean(values, seed + offset)
    return {"comparison": name, "control": control["experiment_id"], "treatment": treatment["experiment_id"],
            "aggregate_treatment_minus_control": agg, "domain_r2_delta": dom, "paired_case_stats": case_stats}


def three_seed(variant: str, records: dict) -> dict | None:
    seeds = [s for s in (1234, 7, 2025) if f"{variant}_s{s}" in records]
    if len(seeds) < 2:
        return None
    per_seed = {s: records[f"{variant}_s{s}"]["best"]["physical_r2_casebalanced"]
                - records[f"s3_parent_s{s}"]["best"]["physical_r2_casebalanced"] for s in seeds}
    vals = list(per_seed.values())
    return {"variant": variant, "seeds": seeds, "delta_r2cb_per_seed": per_seed,
            "mean": sum(vals) / len(vals), "wins": sum(v > 0 for v in vals), "losses": sum(v < 0 for v in vals)}


def main() -> None:
    manifest = read_json(MANIFEST)
    records: dict[str, dict] = {}
    # S3 parent (all three seeds) as comparison baselines
    for seed, run in PARENT.items():
        records[f"s3_parent_s{seed}"] = build_record(
            f"s3_parent_s{seed}", run, seed, "s3_parent")
    # 12 new arms from the manifest
    for row in manifest["configs"]:
        records[row["experiment_id"]] = build_record(
            row["experiment_id"], Path(row["run_dir"]), row["seed"], row["variant"],
            Path(row["config"]), row["sha256"])

    failed = {eid: r["integrity_errors"] for eid, r in records.items() if r["integrity"] != "passed"}
    if failed:
        raise RuntimeError(f"integrity failures: {failed}")

    # Per-seed comparisons vs S3 parent (same seed)
    comparisons = []
    for row in manifest["configs"]:
        eid, seed = row["experiment_id"], row["seed"]
        comparisons.append(comparison(f"{eid}_vs_s3parent", records[f"s3_parent_s{seed}"],
                                       records[eid], 20260724 + len(comparisons) * 100))
    # Caliber decomposition (seed 1234): pooled138 -> casebal / cohortbal
    for treat in ("caliber_casebal_s1234", "caliber_cohortbal_s1234"):
        comparisons.append(comparison(f"{treat}_vs_caliber_pooled138_s1234",
                                       records["caliber_pooled138_s1234"], records[treat],
                                       20261024 + len(comparisons) * 100))

    three_seed_arms = [ts for ts in (three_seed(v, records)
                       for v in ("caliber_casebal", "caliber_cohortbal", "cohort_onehot")) if ts]

    # Rank all mixed-test arms (new + S3 parent seed1234) by R2_cb
    rankable = [r for r in records.values() if r["evaluation_partition"] == "test"]
    rank = sorted(rankable, key=lambda r: r["best"]["physical_r2_casebalanced"], reverse=True)

    payload = {
        "schema_version": 1, "analysis_date": "2026-07-24",
        "scope": "S3-GEOPE root-cause matrix; anchored on three-seed-confirmed S3; mixed test36",
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "s3_parent_r2cb": {s: records[f"s3_parent_s{s}"]["best"]["physical_r2_casebalanced"] for s in PARENT},
        "records": records, "paired_comparisons": comparisons, "three_seed_vs_s3parent": three_seed_arms,
        "rank_by_physical_r2_casebalanced": [
            {"rank": i, "experiment_id": r["experiment_id"], "variant": r["variant"], "seed": r["seed"],
             "physical_r2_casebalanced": r["best"]["physical_r2_casebalanced"],
             "domains": r["domains"]} for i, r in enumerate(rank, 1)],
        "interpretation_limits": [
            "Caliber/tail/combined single-seed arms (pooled138, rawhuber02, cohortbal_onehot) support screening only.",
            "casebal/cohortbal/cohort_onehot have three seeds; mean-over-seed deltas are paired against S3 same seed.",
            "test36 is a historical engineering screening set; no independent generalization claim.",
            "All arms share the frozen mixed train138/test36 split and support-query protocol.",
        ],
    }

    # summary CSV
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as fh:
        fields = ["experiment_id", "variant", "seed", "best_epoch", *AGG_KEYS,
                  "AG_r2", "AAA_r2", "ILO_r2", "last_minus_best_r2cb", "integrity"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records.values():
            w.writerow({"experiment_id": r["experiment_id"], "variant": r["variant"], "seed": r["seed"],
                        "best_epoch": r["history"].get("best_epoch"),
                        **{k: r["best"][k] for k in AGG_KEYS},
                        "AG_r2": r["domains"]["AG"], "AAA_r2": r["domains"]["AAA"], "ILO_r2": r["domains"]["ILO"],
                        "last_minus_best_r2cb": r["last_minus_best"]["physical_r2_casebalanced"],
                        "integrity": r["integrity"]})
    # paired CSV
    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as fh:
        fields = ["comparison", "control", "treatment", "delta_r2cb", "delta_AG", "delta_AAA", "delta_ILO",
                  "case_r2_mean_delta", "case_r2_ci95_low", "case_r2_ci95_high", "case_r2_wins", "case_r2_losses"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in comparisons:
            cr2 = c["paired_case_stats"]["case_r2"]
            w.writerow({"comparison": c["comparison"], "control": c["control"], "treatment": c["treatment"],
                        "delta_r2cb": c["aggregate_treatment_minus_control"]["physical_r2_casebalanced"],
                        "delta_AG": c["domain_r2_delta"]["AG"], "delta_AAA": c["domain_r2_delta"]["AAA"],
                        "delta_ILO": c["domain_r2_delta"]["ILO"], "case_r2_mean_delta": cr2["mean_delta"],
                        "case_r2_ci95_low": cr2["ci95_low"], "case_r2_ci95_high": cr2["ci95_high"],
                        "case_r2_wins": cr2["wins"], "case_r2_losses": cr2["losses"]})

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "records": len(records),
                      "winner": rank[0]["experiment_id"], "winner_r2cb": round(rank[0]["best"]["physical_r2_casebalanced"], 4),
                      "json": str(OUT_JSON)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
