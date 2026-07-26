#!/usr/bin/env python3
"""Prepare the S3-GEOPE root-cause matrix (2026-07-23 overnight batch).

Anchor/template: the three-seed-confirmed S3-GEOPE arm
`s3_d2k64_pnxr_geope_mixed.json` (D2-K64, mixed train138/test36, PointNeXt-R +
LocalGeoPE).  This batch probes the three documented root causes of the ILO
negative transfer, each as a SINGLE-VARIABLE change against that frozen S3 base:

  root #1 (normalization caliber, point-pooled dominated by dense cohorts):
    caliber_pooled138  -> refit target stats on mixed train138 (still pooled)
    caliber_casebal    -> per-case-balanced (A1) target stats on train138
    caliber_cohortbal  -> per-family-balanced (A1b) target stats on train138  [PRIMARY]
  root #2 (missing disease-domain condition):
    cohort_onehot      -> append cohort_ag/aaa/ilo one-hot (input_dim 6->9)
  root #3 (log-MSE under-weights high-WSS tail):
    rawhuber02         -> add physical raw-Huber tail term lambda=0.2
  exploratory combined (root #1+#2; NON-attributable, single seed, labelled):
    cohortbal_onehot   -> caliber_cohortbal + cohort_onehot

Only the fields listed in `only_allowed_differences` may differ from the S3
seed-1234 base.  Model geometry / split / support-query protocol / selection
rule stay frozen.  The manifest is the audit source for every later gate,
submission, and analysis step.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig, input_dim


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "training_wss_min/configs/pointnetpp_d2_k64_ilo_structure_20260723/s3_d2k64_pnxr_geope_mixed.json"
CONFIG_DIR = ROOT / "training_wss_min/configs/pointnetpp_s3_rootcause_20260723"
PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "s3_rootcause_matrix_prepared.json"

FOLD = ROOT / "data_wss_min/fold_stats"
STATS_POOLED138 = FOLD / "q2v_ilo_20260718/q2v_pool2025_train138_global_stats.json"
STATS_CASEBAL = FOLD / "norm_loss_20260720/q2v_pool2025_train138_casebal_global_stats.json"
STATS_COHORTBAL = FOLD / "norm_loss_20260720/q2v_pool2025_train138_cohortbal_global_stats.json"
COHORT_FEATURES = ["cohort_ag", "cohort_aaa", "cohort_ilo"]

# variant -> (mutator(cfg) -> only_allowed_diff_fields, seeds, human note)
VARIANTS = {
    "caliber_pooled138": dict(seeds=[1234], note="root#1 decomposition anchor: refit pooled stats on mixed train138"),
    "caliber_casebal": dict(seeds=[1234, 7, 2025], note="root#1: per-case-balanced (A1) target stats"),
    "caliber_cohortbal": dict(seeds=[1234, 7, 2025], note="root#1 PRIMARY: per-family-balanced (A1b) target stats"),
    "cohort_onehot": dict(seeds=[1234, 7, 2025], note="root#2: cohort one-hot condition (input_dim 6->9)"),
    "rawhuber02": dict(seeds=[1234], note="root#3 probe: physical raw-Huber tail lambda=0.2"),
    "cohortbal_onehot": dict(seeds=[1234], note="exploratory combined root#1+#2 (non-attributable)"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_variant(cfg: dict, variant: str) -> list[str]:
    """Mutate cfg in place for the variant; return the changed data/train fields."""
    changed: list[str] = []
    if variant == "caliber_pooled138":
        cfg["data"]["wss_stats_path"] = str(STATS_POOLED138)
        changed.append("data.wss_stats_path")
    elif variant == "caliber_casebal":
        cfg["data"]["wss_stats_path"] = str(STATS_CASEBAL)
        changed.append("data.wss_stats_path")
    elif variant == "caliber_cohortbal":
        cfg["data"]["wss_stats_path"] = str(STATS_COHORTBAL)
        changed.append("data.wss_stats_path")
    elif variant == "cohort_onehot":
        cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + COHORT_FEATURES
        changed.append("data.input_features")
    elif variant == "rawhuber02":
        cfg["train"]["loss_raw_huber_lambda"] = 0.2
        changed.append("train.loss_raw_huber_lambda")
    elif variant == "cohortbal_onehot":
        cfg["data"]["wss_stats_path"] = str(STATS_COHORTBAL)
        cfg["data"]["input_features"] = list(cfg["data"]["input_features"]) + COHORT_FEATURES
        changed.extend(["data.wss_stats_path", "data.input_features"])
    else:
        raise ValueError(f"unknown variant {variant}")
    return changed


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"refusing to replace existing manifest: {MANIFEST}")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for stats in (STATS_POOLED138, STATS_CASEBAL, STATS_COHORTBAL):
        if not stats.is_file():
            raise FileNotFoundError(stats)

    base_raw = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    base_cfg = ExpConfig.from_json(BASE_CONFIG)
    base_split_sha = sha(Path(base_cfg.data.split_path))
    if input_dim(base_cfg) != 6:
        raise RuntimeError(f"unexpected base input_dim {input_dim(base_cfg)}")

    # every caliber stats file must be computed on the identical split
    for stats in (STATS_POOLED138, STATS_CASEBAL, STATS_COHORTBAL):
        meta = json.loads(stats.read_text())
        if meta.get("split_sha256") and meta["split_sha256"] != base_split_sha:
            raise RuntimeError(f"stats/base split mismatch: {stats.name}")

    rows = []
    for variant, spec in VARIANTS.items():
        for seed in spec["seeds"]:
            cfg = copy.deepcopy(base_raw)
            changed = apply_variant(cfg, variant)
            cfg["name"] = f"pointnetpp_s3_rootcause/outputs/{variant}_s{seed}"
            cfg["train"]["seed"] = seed
            allowed = ["name", "train.seed", *changed]
            child = CONFIG_DIR / f"s3rc_{variant}_s{seed}.json"
            if child.exists():
                raise FileExistsError(f"refusing to replace existing config: {child}")
            write_json(child, cfg)

            parsed = ExpConfig.from_json(child)  # validates schema + geope indices
            expected_dim = 9 if variant in ("cohort_onehot", "cohortbal_onehot") else 6
            if input_dim(parsed) != expected_dim:
                raise RuntimeError(f"{child.name}: input_dim {input_dim(parsed)} != {expected_dim}")
            rows.append({
                "experiment_id": f"{variant}_s{seed}",
                "arm": "S3",  # architecture is S3 (PointNeXt-R + LocalGeoPE) for every row
                "variant": variant,
                "root_cause": spec["note"],
                "seed": seed,
                "config": str(child.resolve()),
                "sha256": sha(child),
                "parent_config": str(BASE_CONFIG.resolve()),
                "parent_config_sha256": sha(BASE_CONFIG),
                "run_dir": str(parsed.run_dir.resolve()),
                "input_dim": input_dim(parsed),
                "split": str(Path(parsed.data.split_path).resolve()),
                "split_sha256": sha(Path(parsed.data.split_path)),
                "target_stats": str(Path(parsed.data.wss_stats_path).resolve()),
                "target_stats_sha256": sha(Path(parsed.data.wss_stats_path)),
                "feature_stats": str(Path(parsed.data.feature_stats_path).resolve()),
                "feature_stats_sha256": sha(Path(parsed.data.feature_stats_path)),
                "only_allowed_differences": allowed,
            })

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": "s3_geope_rootcause_matrix_20260723",
        "anchor_template": str(BASE_CONFIG.resolve()),
        "anchor_template_sha256": sha(BASE_CONFIG),
        "anchor_note": "three-seed confirmed S3-GEOPE (D2-K64 mixed train138/test36)",
        "split_policy": "fixed pool2025 mixed train138/test36; never regenerated per train seed",
        "evaluation_support_seed": 1234,
        "directions": {
            "root1_normalization_caliber": ["caliber_pooled138", "caliber_casebal", "caliber_cohortbal"],
            "root2_domain_condition": ["cohort_onehot"],
            "root3_tail_loss": ["rawhuber02"],
            "exploratory_combined": ["cohortbal_onehot"],
        },
        "configs": rows,
        "expected_new_jobs": len(rows),
    }
    write_json(MANIFEST, payload)
    print(json.dumps({"status": "prepared", "configs": len(rows), "manifest": str(MANIFEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
