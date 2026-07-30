#!/usr/bin/env python3
"""Freeze the REG-P10 static EdgeConv SA1/SA2/SA1+SA2 matrix."""

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
CONFIG_DIR = ROOT / "training_wss_min/configs/pointnetpp_regp10_edgeconv_20260727"
SWEEP = ROOT / "training_wss_min/configs/sweeps/pointnetpp_regp10_edgeconv_20260727.txt"
MANIFEST = (
    ROOT
    / "training_wss_min/preflight/regp10_edgeconv_matrix_20260727_prepared.json"
)
ARMS = (
    ("edgeconv_sa1_s1234", "regp10_edgeconv_sa1_s1234.json", (1,)),
    ("edgeconv_sa2_s1234", "regp10_edgeconv_sa2_s1234.json", (2,)),
    ("edgeconv_sa12_s1234", "regp10_edgeconv_sa12_s1234.json", (1, 2)),
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
    targets = [CONFIG_DIR / filename for _, filename, _ in ARMS]
    collisions = [path for path in [*targets, SWEEP, MANIFEST] if path.exists()]
    if collisions:
        raise FileExistsError(
            "refusing to replace existing EdgeConv artifacts: "
            + ", ".join(map(str, collisions))
        )

    parent_cfg = ExpConfig.from_json(PARENT)
    if (
        parent_cfg.train.seed != 1234
        or tuple(parent_cfg.model.sa_blocks) != (1, 1, 0)
        or not parent_cfg.model.local_geope
        or float(parent_cfg.model.drop_path_rate) != 0.1
        or parent_cfg.model.edgeconv_stages
        or parent_cfg.model.local_transformer_stages
        or parent_cfg.model.coarse_attention
    ):
        raise RuntimeError("parent is not the frozen REG-P10 seed1234 anchor")
    parent_raw = json.loads(PARENT.read_text(encoding="utf-8"))

    rows = []
    for experiment_id, filename, stages in ARMS:
        raw = json.loads(json.dumps(parent_raw))
        raw["name"] = (
            "pointnetpp_regp10_edgeconv/outputs/"
            + experiment_id
        )
        raw["notes"] = (
            "REG-P10 frozen anchor: S3 PointNeXt-R + LocalGeoPE + "
            "DropPath0.10, mixed train138/test36, seed1234; only static "
            f"geometry-constrained EdgeConv stages={list(stages)} is enabled."
        )
        raw["model"]["edgeconv_stages"] = list(stages)
        path = CONFIG_DIR / filename
        write_json(path, raw)

        cfg = ExpConfig.from_json(path)
        if (
            tuple(cfg.model.edgeconv_stages) != stages
            or cfg.train.seed != 1234
            or float(cfg.model.drop_path_rate) != 0.1
            or tuple(cfg.model.sa_blocks) != (1, 1, 0)
            or not cfg.model.local_geope
            or cfg.model.local_transformer_stages
            or cfg.model.coarse_attention
        ):
            raise RuntimeError(f"{filename}: EdgeConv/REG-P10 contract drift")
        if cfg.run_dir.exists():
            raise FileExistsError(f"run already exists: {cfg.run_dir}")
        rows.append(
            {
                "experiment_id": experiment_id,
                "variant": "static_edgeconv",
                "edgeconv_stages": list(stages),
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
                "only_allowed_differences": [
                    "name",
                    "notes",
                    "model.edgeconv_stages",
                ],
            }
        )

    SWEEP.parent.mkdir(parents=True, exist_ok=True)
    SWEEP.write_text(
        "\n".join(str(path.resolve()) for path in targets) + "\n",
        encoding="utf-8",
    )
    write_json(
        MANIFEST,
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "prepared",
            "protocol": "regp10_static_geometry_constrained_edgeconv_sa1_sa2",
            "scope": (
                "REG-P10 seed1234; static EdgeConv correction on existing SA "
                "geometry graph; SA1, SA2, and SA1+SA2 arms"
            ),
            "parent": str(PARENT.resolve()),
            "parent_sha256": sha(PARENT),
            "seeds": [1234],
            "evaluation_support_seed": 1234,
            "edgeconv_stage_subsets": [[1], [2], [1, 2]],
            "dynamic_graph": False,
            "configs": rows,
            "expected_new_jobs": len(rows),
        },
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "configs": len(rows),
                "manifest": str(MANIFEST),
                "sweep": str(SWEEP),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
