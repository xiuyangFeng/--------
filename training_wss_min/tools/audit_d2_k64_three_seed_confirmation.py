#!/usr/bin/env python3
"""Static hard gate for the M1/S2/S3 three-seed confirmation matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig, input_dim
from training_wss_min.models import build_model


EXPECTED = {(arm, seed) for arm in ("M1", "S2", "S3") for seed in (7, 2025)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without(payload: dict, fields: set[str]) -> dict:
    out = json.loads(json.dumps(payload))
    for dotted in fields:
        current = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            current = current[part]
        current.pop(parts[-1], None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or len(rows) != 6:
        raise RuntimeError("confirmation manifest must contain exactly six prepared rows")
    if {(r["arm"], r["seed"]) for r in rows} != EXPECTED:
        raise RuntimeError("matrix is not M1/S2/S3 x seeds{7,2025}")
    detail = []
    canonical_by_arm = {}
    for row in rows:
        path, parent = Path(row["config"]), Path(row["parent_config"])
        if not path.is_file() or not parent.is_file():
            raise FileNotFoundError(path if not path.is_file() else parent)
        if sha(path) != row["sha256"] or sha(parent) != row["parent_config_sha256"]:
            raise RuntimeError(f"config hash drift: {path.name}")
        for key, hash_key in (("split", "split_sha256"), ("target_stats", "target_stats_sha256"), ("feature_stats", "feature_stats_sha256")):
            if sha(Path(row[key])) != row[hash_key]:
                raise RuntimeError(f"frozen {key} hash drift: {path.name}")
        raw, raw_parent = json.loads(path.read_text()), json.loads(parent.read_text())
        if without(raw, {"name", "train.seed"}) != without(raw_parent, {"name", "train.seed"}):
            raise RuntimeError(f"{path.name} has a non-permitted config difference")
        if raw["train"]["seed"] != row["seed"] or raw["name"] == raw_parent["name"]:
            raise RuntimeError(f"incorrect run-name/seed clone: {path.name}")
        cfg = ExpConfig.from_json(path)
        if cfg.run_dir.exists():
            raise FileExistsError(f"run directory already exists: {cfg.run_dir}")
        contract = {
            "split": Path(cfg.data.split_path).name,
            "counts": (cfg.data.wall_n_points, cfg.data.support_n_points, cfg.data.query_n_points),
            "query": (cfg.data.query_mode, cfg.data.support_sampling, cfg.data.query_sampling),
            "centers": tuple(cfg.model.sa_center_counts),
            "nsample": tuple(cfg.model.sa_nsample),
            "grouping": tuple(cfg.model.sa_grouping),
            "width": cfg.model.width,
            "loss": cfg.train.loss,
            "epochs": cfg.train.epochs,
            "selection": (cfg.train.ckpt_metric, cfg.train.selection_rule),
            "support_seed": cfg.eval.support_seed,
            "surface": cfg.eval.surface_metric_mode,
        }
        expected = {
            "split": "split_AG_AAA_ILO_q2v_pool2025_train138_test36.json",
            "counts": (5000, 5000, 5000), "query": ("same", "random", "random"),
            "centers": (125, 125, 32), "nsample": (64, 16, 16),
            "grouping": ("knn_cover", "ball", "ball"), "width": 32, "loss": "mse",
            "epochs": 400, "selection": ("train_loss", "train_loss"),
            "support_seed": 1234, "surface": "legacy_vertex",
        }
        if contract != expected:
            raise RuntimeError(f"frozen protocol drift in {path.name}: {contract}")
        model = build_model(cfg.model, input_dim(cfg))
        blocks = [len(sa.blocks) for sa in model.sa]
        local_geope = bool(getattr(model, "local_geope", False))
        expected_blocks = [0, 0, 0] if row["arm"] == "M1" else [1, 1, 0]
        expected_geo = row["arm"] == "S3"
        if blocks != expected_blocks or local_geope != expected_geo:
            raise RuntimeError(f"runtime module contract failed for {path.name}")
        canonical_by_arm.setdefault(row["arm"], raw)
        detail.append({"experiment_id": row["experiment_id"], "config": str(path), "config_sha256": sha(path), "runtime_sa_blocks": blocks, "runtime_local_geope": local_geope, "parameters": sum(p.numel() for p in model.parameters())})
    for arm, payload in canonical_by_arm.items():
        siblings = [json.loads(Path(r["config"]).read_text()) for r in rows if r["arm"] == arm]
        if without(siblings[0], {"name", "train.seed"}) != without(siblings[1], {"name", "train.seed"}):
            raise RuntimeError(f"seed pair drift in {arm}")
    out = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "status": "passed", "manifest": str(args.manifest.resolve()), "manifest_sha256": sha(args.manifest), "checks": {"six_exact_cells": True, "only_name_and_train_seed_changed": True, "frozen_protocol": True, "frozen_split_and_stats_hashes": True, "runtime_module_contract": True, "run_dirs_absent": True}, "configs": detail}
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "configs": len(detail), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
