#!/usr/bin/env python3
"""Prepare the missing single-rate and pairwise S3 regularization arms."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "training_wss_min/configs/pointnetpp_d2_k64_ilo_structure_20260723/s3_d2k64_pnxr_geope_mixed.json"
CONFIG_DIR = ROOT / "training_wss_min/configs/pointnetpp_s3_regularization_completion_20260725"
MANIFEST = ROOT / "training_wss_min/preflight/s3_regularization_completion_20260725_prepared.json"
SINGLES = (
    ("head_dropout005", {"model.dropout": 0.05}),
    ("head_dropout015", {"model.dropout": 0.15}),
    ("head_dropout020", {"model.dropout": 0.20}),
    ("droppath015", {"model.drop_path_rate": 0.15}),
    ("droppath020", {"model.drop_path_rate": 0.20}),
    ("neighbordrop010", {"model.neighbor_drop_rate": 0.10}),
    ("neighbordrop015", {"model.neighbor_drop_rate": 0.15}),
    ("neighbordrop020", {"model.neighbor_drop_rate": 0.20}),
)
PAIRS = tuple(
    (f"pair_hd{hd:02d}_dp{dp:02d}", {"model.dropout": hd / 100, "model.drop_path_rate": dp / 100})
    for hd in (5, 10) for dp in (5, 10)
) + tuple(
    (f"pair_hd{hd:02d}_nd{nd:02d}", {"model.dropout": hd / 100, "model.neighbor_drop_rate": nd / 100})
    for hd in (5, 10) for nd in (5, 10)
) + tuple(
    (f"pair_dp{dp:02d}_nd{nd:02d}", {"model.drop_path_rate": dp / 100, "model.neighbor_drop_rate": nd / 100})
    for dp in (5, 10) for nd in (5, 10)
)
ARMS = SINGLES + PAIRS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_dotted(payload: dict, dotted: str, value: float) -> None:
    section, key = dotted.split(".", 1)
    payload.setdefault(section, {})[key] = value


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(f"refusing to replace existing manifest: {MANIFEST}")
    parent_cfg = ExpConfig.from_json(PARENT)
    if parent_cfg.train.seed != 1234 or not parent_cfg.model.local_geope or tuple(parent_cfg.model.sa_blocks) != (1, 1, 0):
        raise RuntimeError("parent is not the frozen S3-GEOPE seed1234 template")
    if len(ARMS) != 20:
        raise RuntimeError(f"expected 20 completion arms, got {len(ARMS)}")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    raw_parent = json.loads(PARENT.read_text(encoding="utf-8"))
    rows = []
    for variant, fields in ARMS:
        payload = copy.deepcopy(raw_parent)
        experiment_id = f"{variant}_s1234"
        payload["name"] = f"pointnetpp_s3_regularization_completion/outputs/{experiment_id}"
        for dotted, value in fields.items():
            set_dotted(payload, dotted, value)
        path = CONFIG_DIR / f"s3regc_{experiment_id}.json"
        if path.exists():
            raise FileExistsError(f"refusing to replace existing config: {path}")
        write(path, payload)
        cfg = ExpConfig.from_json(path)
        if cfg.run_dir.exists():
            raise FileExistsError(f"run already exists: {cfg.run_dir}")
        rows.append({
            "experiment_id": experiment_id, "arm": "S3", "variant": variant,
            "seed": 1234, "config": str(path.resolve()), "sha256": sha(path),
            "parent_config": str(PARENT.resolve()), "parent_config_sha256": sha(PARENT),
            "run_dir": str(cfg.run_dir.resolve()),
            "split": str(Path(cfg.data.split_path).resolve()), "split_sha256": sha(Path(cfg.data.split_path)),
            "target_stats": str(Path(cfg.data.wss_stats_path).resolve()), "target_stats_sha256": sha(Path(cfg.data.wss_stats_path)),
            "feature_stats": str(Path(cfg.data.feature_stats_path).resolve()), "feature_stats_sha256": sha(Path(cfg.data.feature_stats_path)),
            "regularizers": fields, "only_allowed_differences": ["name", *fields],
        })
    write(MANIFEST, {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "prepared",
        "protocol": "s3_geope_regularization_completion_single_seed_20260725",
        "scope": "frozen S3-GEOPE seed1234; historical mixed train138/test36 engineering screen; no independent confirmation claim",
        "parent": str(PARENT.resolve()), "parent_sha256": sha(PARENT), "seeds": [1234], "evaluation_support_seed": 1234,
        "single_rate_completion": {"head_dropout": [0.05, 0.15, 0.20], "drop_path": [0.15, 0.20], "neighbor_drop": [0.10, 0.15, 0.20]},
        "pairwise_cross": {"rates": [0.05, 0.10], "pairs": ["head_dropout+drop_path", "head_dropout+neighbor_drop", "drop_path+neighbor_drop"], "arms": 12},
        "configs": rows, "expected_new_jobs": len(rows),
    })
    print(json.dumps({"status": "prepared", "configs": len(rows), "singles": len(SINGLES), "pairs": len(PAIRS), "manifest": str(MANIFEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
