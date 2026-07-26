#!/usr/bin/env python3
"""Prepare the frozen M1/S2/S3 seed-7/2025 confirmation matrix.

This intentionally clones the completed seed-1234 JSON verbatim and mutates
only the output run name and train seed.  The manifest is the audit source for
all later gates, submission, and analysis.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "training_wss_min/configs/pointnetpp_d2_k64_ilo_structure_20260723"
PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "d2_k64_ilo_structure_three_seed_confirmation_prepared.json"
BASES = {
    "m1": "m1_d2k64_mixed138_test36.json",
    "s2": "s2_d2k64_pnxr_mixed.json",
    "s3": "s3_d2k64_pnxr_geope_mixed.json",
}
SEEDS = (7, 2025)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"refusing to replace existing manifest: {MANIFEST}")
    rows = []
    for arm, filename in BASES.items():
        parent = CONFIG_DIR / filename
        if not parent.is_file():
            raise FileNotFoundError(parent)
        raw = json.loads(parent.read_text(encoding="utf-8"))
        for seed in SEEDS:
            cfg = copy.deepcopy(raw)
            base_name = raw["name"]
            cfg["name"] = f"{base_name}_s{seed}"
            cfg["train"]["seed"] = seed
            child = CONFIG_DIR / f"{parent.stem}_s{seed}.json"
            if child.exists():
                raise FileExistsError(f"refusing to replace existing config: {child}")
            write_json(child, cfg)
            parsed = ExpConfig.from_json(child)
            rows.append({
                "experiment_id": f"{arm}_d2k64_mixed_s{seed}",
                "arm": arm.upper(),
                "seed": seed,
                "config": str(child.resolve()),
                "sha256": sha(child),
                "parent_config": str(parent.resolve()),
                "parent_config_sha256": sha(parent),
                "run_dir": str(parsed.run_dir.resolve()),
                "split": str(Path(parsed.data.split_path).resolve()),
                "split_sha256": sha(Path(parsed.data.split_path)),
                "target_stats": str(Path(parsed.data.wss_stats_path).resolve()),
                "target_stats_sha256": sha(Path(parsed.data.wss_stats_path)),
                "feature_stats": str(Path(parsed.data.feature_stats_path).resolve()),
                "feature_stats_sha256": sha(Path(parsed.data.feature_stats_path)),
                "only_allowed_differences": ["name", "train.seed"],
            })
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": "d2_k64_ilo_structure_m1_s2_s3_three_seed_confirmation",
        "frozen_seed1234": {
            arm: str((CONFIG_DIR / filename).resolve()) for arm, filename in BASES.items()
        },
        "new_train_seeds": list(SEEDS),
        "evaluation_support_seed": 1234,
        "split_policy": "fixed pool2025 mixed train138/test36; never regenerated per train seed",
        "configs": rows,
        "expected_new_jobs": 6,
        "forbidden_variables": [
            "dropout", "drop_path", "neighbor_drop", "coarse_attention", "sep",
            "mfeat", "new_loss", "new_normalization_stats", "cohort_one_hot", "new_split",
            "support_query_protocol_change",
        ],
    }
    write_json(MANIFEST, payload)
    print(json.dumps({"status": "prepared", "configs": len(rows), "manifest": str(MANIFEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
