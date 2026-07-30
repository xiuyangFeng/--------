#!/usr/bin/env python3
"""Prepare SA1/SA2 local-Transformer controls on the fixed D2 PNXR-GeoPE anchor."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
PARENT = (
    ROOT
    / "training_wss_min/configs/"
    "pointnetpp_d2_c125_k64_pnxr_geope_20260726/"
    "d2_c125_k64_pnxr_geope.json"
)
CONFIG_DIR = (
    ROOT
    / "training_wss_min/configs/"
    "pointnetpp_d2_c125_k64_pnxr_geope_transformer_20260726"
)
MANIFEST = (
    ROOT
    / "training_wss_min/preflight/"
    "d2_c125_k64_pnxr_geope_transformer_20260726_prepared.json"
)
ARMS = (
    (
        "d2_pnxr_geope_localtf_sa1_s1234",
        "d2_c125_k64_pnxr_geope_localtf_sa1_s1234.json",
        (1,),
    ),
    (
        "d2_pnxr_geope_localtf_sa2_s1234",
        "d2_c125_k64_pnxr_geope_localtf_sa2_s1234.json",
        (2,),
    ),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"refusing to replace existing manifest: {MANIFEST}")
    parent_payload = json.loads(PARENT.read_text(encoding="utf-8"))
    parent = ExpConfig.from_json(PARENT)
    if (
        parent.train.seed != 1234
        or tuple(parent.model.sa_blocks) != (1, 1, 0)
        or not parent.model.local_geope
        or parent.model.local_transformer_stages
        or parent.model.coarse_attention
        or float(parent.model.drop_path_rate) != 0.0
        or tuple(parent.model.sa_center_counts) != (125, 125, 32)
        or tuple(parent.model.sa_nsample) != (64, 16, 16)
        or parent.data.query_mode != "same"
    ):
        raise RuntimeError("parent is not the frozen D2 c125×k64 PNXR-GeoPE anchor")

    rows = []
    for experiment_id, filename, stages in ARMS:
        path = CONFIG_DIR / filename
        if path.exists():
            raise FileExistsError(f"refusing to replace existing config: {path}")
        child = copy.deepcopy(parent_payload)
        child["name"] = (
            "pointnetpp_d2_c125_k64_pnxr_geope_transformer/outputs/"
            f"localtf_sa{''.join(map(str, stages))}_s1234"
        )
        child["notes"] = (
            "Strict fixed-split D2 c125×k64 PointNeXt-R + LocalGeoPE "
            f"control; only local_transformer_stages={list(stages)} is enabled. "
            "AG/AAA 106/0/27, stats, random5000/SAME, centers 125/125/32, "
            "neighbors 64/16/16, seed1234, 400 epochs and train-loss selection "
            "remain frozen."
        )
        child["model"]["local_transformer_stages"] = list(stages)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(child, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        cfg = ExpConfig.from_json(path)
        if tuple(cfg.model.local_transformer_stages) != stages:
            raise RuntimeError(f"{filename}: local Transformer stage drift")
        if (
            tuple(cfg.model.sa_blocks) != (1, 1, 0)
            or not cfg.model.local_geope
            or cfg.model.coarse_attention
            or float(cfg.model.drop_path_rate) != 0.0
        ):
            raise RuntimeError(f"{filename}: D2 PNXR-GeoPE anchor drift")
        if cfg.run_dir.exists():
            raise FileExistsError(f"run already exists: {cfg.run_dir}")

        rows.append(
            {
                "experiment_id": experiment_id,
                "variant": "local_transformer",
                "local_transformer_stages": list(stages),
                "coarse_attention": False,
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
                "feature_stats": (
                    None
                    if cfg.data.feature_stats_path is None
                    else str(Path(cfg.data.feature_stats_path).resolve())
                ),
                "feature_stats_sha256": (
                    None
                    if cfg.data.feature_stats_path is None
                    else sha(Path(cfg.data.feature_stats_path))
                ),
                "only_allowed_differences": [
                    "name",
                    "notes",
                    "model.local_transformer_stages",
                ],
            }
        )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "prepared",
                "protocol": (
                    "D2_c125_k64_fixed_106_0_27_PointNeXtR_LocalGeoPE_"
                    "local_Transformer_SA1_SA2"
                ),
                "scope": (
                    "Two seed1234 single-variable local-Transformer controls "
                    "on the completed fixed-split D2 PNXR-GeoPE anchor"
                ),
                "parent": str(PARENT.resolve()),
                "parent_sha256": sha(PARENT),
                "frozen_contract": {
                    "split": "AG/AAA 106/0/27",
                    "sampling": "random5000/SAME",
                    "sa_centers": [125, 125, 32],
                    "sa_nsample": [64, 16, 16],
                    "sa_blocks": [1, 1, 0],
                    "local_geope": True,
                    "drop_path_rate": 0.0,
                    "input_features": [
                        "x",
                        "y",
                        "z",
                        "abscissa_norm",
                        "local_radius",
                        "curvature",
                    ],
                    "train": "seed1234; 400 epochs; train-loss selection",
                },
                "local_stage_subsets": [[1], [2]],
                "configs": rows,
                "expected_new_jobs": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "configs": len(rows),
                "manifest": str(MANIFEST),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
