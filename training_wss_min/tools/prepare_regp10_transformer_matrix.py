#!/usr/bin/env python3
"""Freeze and register the complete REG-P10 Transformer stage matrix."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    ROOT
    / "training_wss_min/configs/pointnetpp_s3_regularization_20260724"
    / "s3reg_droppath010_s1234.json"
)
CONFIG_DIR = (
    ROOT / "training_wss_min/configs/pointnetpp_regp10_transformer_20260726"
)
MANIFEST = (
    ROOT
    / "training_wss_min/preflight/regp10_transformer_matrix_20260726_prepared.json"
)
ARMS = (
    ("localtf_sa1_s1234", "regp10_localtf_sa1_s1234.json", (1,), False),
    ("localtf_sa2_s1234", "regp10_localtf_sa2_s1234.json", (2,), False),
    ("localtf_sa3_s1234", "regp10_localtf_sa3_s1234.json", (3,), False),
    ("localtf_sa12_s1234", "regp10_localtf_sa12_s1234.json", (1, 2), False),
    ("localtf_sa13_s1234", "regp10_localtf_sa13_s1234.json", (1, 3), False),
    ("localtf_sa23_s1234", "regp10_localtf_sa23_s1234.json", (2, 3), False),
    ("localtf_sa123_s1234", "regp10_localtf_sa123_s1234.json", (1, 2, 3), False),
    ("globaltf_sa3_s1234", "regp10_globaltf_sa3_s1234.json", (), True),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"refusing to replace existing manifest: {MANIFEST}")
    parent = ExpConfig.from_json(PARENT)
    if (
        parent.train.seed != 1234
        or tuple(parent.model.sa_blocks) != (1, 1, 0)
        or not parent.model.local_geope
        or float(parent.model.drop_path_rate) != 0.1
        or parent.model.local_transformer_stages
        or parent.model.coarse_attention
    ):
        raise RuntimeError("parent is not the frozen REG-P10 seed1234 anchor")

    rows = []
    for experiment_id, filename, expected_stages, expected_global in ARMS:
        path = CONFIG_DIR / filename
        cfg = ExpConfig.from_json(path)
        if tuple(cfg.model.local_transformer_stages) != expected_stages:
            raise RuntimeError(
                f"{filename}: expected local stages {expected_stages}, "
                f"got {tuple(cfg.model.local_transformer_stages)}"
            )
        if bool(cfg.model.coarse_attention) != expected_global:
            raise RuntimeError(
                f"{filename}: coarse_attention mismatch"
            )
        if cfg.train.seed != 1234 or float(cfg.model.drop_path_rate) != 0.1:
            raise RuntimeError(f"{filename}: REG-P10 anchor drift")
        if cfg.run_dir.exists():
            raise FileExistsError(f"run already exists: {cfg.run_dir}")
        allowed = ["name", "notes"]
        if expected_stages:
            allowed.append("model.local_transformer_stages")
        if expected_global:
            allowed.append("model.coarse_attention")
        rows.append(
            {
                "experiment_id": experiment_id,
                "variant": "local" if expected_stages else "global",
                "local_transformer_stages": list(expected_stages),
                "coarse_attention": expected_global,
                "seed": 1234,
                "config": str(path.resolve()),
                "sha256": sha(path),
                "parent_config": str(PARENT.resolve()),
                "parent_config_sha256": sha(PARENT),
                "run_dir": str(cfg.run_dir.resolve()),
                "split": str(Path(cfg.data.split_path).resolve()),
                "split_sha256": sha(Path(cfg.data.split_path)),
                "target_stats": str(Path(cfg.data.wss_stats_path).resolve()),
                "target_stats_sha256": sha(Path(cfg.data.wss_stats_path)),
                "feature_stats": str(Path(cfg.data.feature_stats_path).resolve()),
                "feature_stats_sha256": sha(Path(cfg.data.feature_stats_path)),
                "only_allowed_differences": allowed,
            }
        )

    write(
        MANIFEST,
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "prepared",
            "protocol": "regp10_complete_local_stage_factorial_plus_global_control",
            "scope": (
                "REG-P10 seed1234; all seven non-empty local SA-stage subsets "
                "plus the existing SA3 coarse-center global-attention control"
            ),
            "parent": str(PARENT.resolve()),
            "parent_sha256": sha(PARENT),
            "seeds": [1234],
            "evaluation_support_seed": 1234,
            "local_stage_subsets": [
                [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]
            ],
            "global_control": "coarse_attention=true; local stages empty",
            "configs": rows,
            "expected_new_jobs": len(rows),
        },
    )
    print(
        json.dumps(
            {"status": "prepared", "configs": len(rows), "manifest": str(MANIFEST)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
