#!/usr/bin/env python3
"""Static hard gate for the single S3-GEOPE + SA3-attention experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.baseline_models import CoarseGlobalBlock
from training_wss_min.config import ExpConfig, input_dim
from training_wss_min.models import build_model
from training_wss_min.tools.audit_s3_rootcause_static import diff_paths


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or len(rows) != 1:
        raise RuntimeError("manifest must contain exactly one prepared config")
    row = rows[0]
    config_path = Path(row["config"])
    parent_path = Path(row["parent_config"])
    for path, expected in (
        (config_path, row["sha256"]),
        (parent_path, row["parent_config_sha256"]),
        (Path(row["split"]), row["split_sha256"]),
        (Path(row["target_stats"]), row["target_stats_sha256"]),
        (Path(row["feature_stats"]), row["feature_stats_sha256"]),
    ):
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"missing or drifted input: {path}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    child = json.loads(config_path.read_text(encoding="utf-8"))
    observed = diff_paths(parent, child)
    expected_diff = {"name", "model.coarse_attention"}
    if observed != expected_diff:
        raise RuntimeError(
            f"S3/attention diff is {sorted(observed)}, expected {sorted(expected_diff)}"
        )
    cfg = ExpConfig.from_json(config_path)
    if cfg.run_dir.exists():
        raise FileExistsError(f"run already exists: {cfg.run_dir}")
    if (
        cfg.train.seed != 1234
        or not cfg.model.local_geope
        or not cfg.model.coarse_attention
        or cfg.model.coarse_attention_heads != 4
        or tuple(cfg.model.sa_center_counts) != (125, 125, 32)
        or tuple(cfg.model.sa_blocks) != (1, 1, 0)
        or float(cfg.model.dropout) != 0.0
        or float(cfg.model.drop_path_rate) != 0.0
        or float(cfg.model.neighbor_drop_rate) != 0.0
    ):
        raise RuntimeError("S3 + SA3 coarse-attention contract drift")
    model = build_model(cfg.model, input_dim(cfg))
    if not isinstance(model.coarse_global, CoarseGlobalBlock):
        raise RuntimeError("coarse-attention runtime module is absent")
    if not all(sa.geo_pe is not None for sa in model.sa):
        raise RuntimeError("LocalGeoPE runtime modules are absent")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha(args.manifest),
        "checks": {
            "exactly_one_seed_1234": True,
            "only_name_and_coarse_attention_changed": True,
            "s3_geope_runtime": True,
            "sa3_attention_runtime": True,
            "four_heads_at_32_centers": True,
            "all_drop_disabled": True,
            "run_dir_absent": True,
        },
        "config": str(config_path.resolve()),
        "config_sha256": sha(config_path),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "rows": 1}, ensure_ascii=False))


if __name__ == "__main__":
    main()
