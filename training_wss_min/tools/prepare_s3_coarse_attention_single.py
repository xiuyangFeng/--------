#!/usr/bin/env python3
"""Prepare one seed-1234 S3-GEOPE + SA3 coarse-attention experiment."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    ROOT / "training_wss_min/configs/pointnetpp_d2_k64_ilo_structure_20260723/"
    "s3_d2k64_pnxr_geope_mixed.json"
)
CONFIG_DIR = ROOT / "training_wss_min/configs/pointnetpp_s3_architecture_20260724"
CONFIG = CONFIG_DIR / "s3arch_sa3_coarse_attention_s1234.json"
MANIFEST = (
    ROOT / "training_wss_min/preflight/s3_sa3_coarse_attention_single_prepared.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for path in (CONFIG, MANIFEST):
        if path.exists():
            raise FileExistsError(f"refusing to replace existing file: {path}")
    parent_cfg = ExpConfig.from_json(PARENT)
    if (
        parent_cfg.train.seed != 1234
        or not parent_cfg.model.local_geope
        or parent_cfg.model.coarse_attention
        or tuple(parent_cfg.model.sa_center_counts) != (125, 125, 32)
        or tuple(parent_cfg.model.sa_blocks) != (1, 1, 0)
    ):
        raise RuntimeError("parent is not the frozen seed-1234 S3-GEOPE anchor")
    raw = json.loads(PARENT.read_text(encoding="utf-8"))
    child = copy.deepcopy(raw)
    child["name"] = (
        "pointnetpp_s3_architecture/outputs/sa3_coarse_attention_s1234"
    )
    child["model"]["coarse_attention"] = True
    write_json(CONFIG, child)
    cfg = ExpConfig.from_json(CONFIG)
    row = {
        "experiment_id": "sa3_coarse_attention_s1234",
        "arm": "S3",
        "variant": "sa3_coarse_attention",
        "seed": 1234,
        "config": str(CONFIG.resolve()),
        "sha256": sha(CONFIG),
        "parent_config": str(PARENT.resolve()),
        "parent_config_sha256": sha(PARENT),
        "run_dir": str(cfg.run_dir.resolve()),
        "split": str(Path(cfg.data.split_path).resolve()),
        "split_sha256": sha(Path(cfg.data.split_path)),
        "target_stats": str(Path(cfg.data.wss_stats_path).resolve()),
        "target_stats_sha256": sha(Path(cfg.data.wss_stats_path)),
        "feature_stats": str(Path(cfg.data.feature_stats_path).resolve()),
        "feature_stats_sha256": sha(Path(cfg.data.feature_stats_path)),
        "only_allowed_differences": ["name", "model.coarse_attention"],
    }
    write_json(MANIFEST, {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": "s3_geope_sa3_coarse_attention_single_seed_20260724",
        "scope": (
            "single seed=1234 architecture screen on historical mixed "
            "train138/test36; not a multi-seed or independent confirmation"
        ),
        "parent": str(PARENT.resolve()),
        "parent_sha256": sha(PARENT),
        "seeds": [1234],
        "attention_contract": {
            "stage": "SA3",
            "coarse_centers": 32,
            "heads": 4,
            "local_geope": True,
            "dropout": 0.0,
            "drop_path_rate": 0.0,
            "neighbor_drop_rate": 0.0,
        },
        "configs": [row],
        "expected_new_jobs": 1,
    })
    print(json.dumps({
        "status": "prepared",
        "configs": 1,
        "manifest": str(MANIFEST),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
