#!/usr/bin/env python3
"""Prepare the three-seed S3-GEOPE regularization matrix (2026-07-24).

Each child clones the completed S3 parent at the SAME seed and changes exactly
one regularization field:

- head_dropout01: final regression-head dropout=0.10
- droppath005: PointNeXt-R linear stochastic-depth schedule, max=0.05
- droppath010: PointNeXt-R linear stochastic-depth schedule, max=0.10
- neighbordrop005: preserve-nearest neighborhood edge dropout=0.05

The frozen evaluation target is the historical mixed test36 engineering
screening set. It is suitable for paired route selection, not for a new
independent generalization claim.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
PARENT_DIR = ROOT / "training_wss_min/configs/pointnetpp_d2_k64_ilo_structure_20260723"
CONFIG_DIR = ROOT / "training_wss_min/configs/pointnetpp_s3_regularization_20260724"
PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "s3_regularization_matrix_prepared.json"
PARENTS = {
    1234: PARENT_DIR / "s3_d2k64_pnxr_geope_mixed.json",
    7: PARENT_DIR / "s3_d2k64_pnxr_geope_mixed_s7.json",
    2025: PARENT_DIR / "s3_d2k64_pnxr_geope_mixed_s2025.json",
}
VARIANTS = {
    "head_dropout01": ("model.dropout", 0.10),
    "droppath005": ("model.drop_path_rate", 0.05),
    "droppath010": ("model.drop_path_rate", 0.10),
    "neighbordrop005": ("model.neighbor_drop_rate", 0.05),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def set_dotted(cfg: dict, dotted: str, value: object) -> None:
    section, field = dotted.split(".", 1)
    cfg.setdefault(section, {})[field] = value


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"refusing to replace existing manifest: {MANIFEST}")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    parent_shas = {}
    for seed, parent in PARENTS.items():
        if not parent.is_file():
            raise FileNotFoundError(parent)
        parent_cfg = ExpConfig.from_json(parent)
        if int(parent_cfg.train.seed) != seed:
            raise RuntimeError(f"parent seed mismatch: {parent}")
        if not parent_cfg.model.local_geope or not any(parent_cfg.model.sa_blocks):
            raise RuntimeError(f"parent is not S3-GEOPE: {parent}")
        parent_shas[str(seed)] = sha(parent)
        raw = json.loads(parent.read_text(encoding="utf-8"))
        for variant, (field, value) in VARIANTS.items():
            child_raw = copy.deepcopy(raw)
            child_raw["name"] = (
                f"pointnetpp_s3_regularization/outputs/{variant}_s{seed}"
            )
            set_dotted(child_raw, field, value)
            child = CONFIG_DIR / f"s3reg_{variant}_s{seed}.json"
            if child.exists():
                raise FileExistsError(
                    f"refusing to replace existing config: {child}"
                )
            write_json(child, child_raw)
            parsed = ExpConfig.from_json(child)
            rows.append({
                "experiment_id": f"{variant}_s{seed}",
                "arm": "S3",
                "variant": variant,
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
                "only_allowed_differences": ["name", field],
            })
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": "s3_geope_regularization_three_seed_20260724",
        "scope": (
            "fixed mixed train138/test36 historical engineering screening set; "
            "paired route selection only, not independent external confirmation"
        ),
        "parents_by_seed": {
            str(seed): str(path.resolve()) for seed, path in PARENTS.items()
        },
        "parent_sha256_by_seed": parent_shas,
        "seeds": list(PARENTS),
        "evaluation_support_seed": 1234,
        "single_variable_variants": {
            variant: {"field": field, "value": value}
            for variant, (field, value) in VARIANTS.items()
        },
        "forbidden_combinations": [
            "drop_path+neighbor_drop",
            "drop+coarse_attention",
            "drop+sep",
            "drop+normalization_change",
            "drop+loss_change",
        ],
        "configs": rows,
        "expected_new_jobs": len(rows),
    }
    write_json(MANIFEST, payload)
    print(json.dumps({
        "status": "prepared",
        "configs": len(rows),
        "manifest": str(MANIFEST),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
