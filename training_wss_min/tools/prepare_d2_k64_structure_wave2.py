#!/usr/bin/env python3
"""Prepare wave 2 of the approved D2-K64 mixed-split structure screen.

S3 and S4 are single-module additions to S2.  SEP cannot be observed under
SAME because support=query takes the exact identity path, so S5 is a matched
pair: independent-query interpolation control (S5C) versus the conservative
learnable convex kernel (S5).
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig


REPO = Path(__file__).resolve().parents[2]
S2_CONFIG = (
    REPO / "training_wss_min/configs/pointnetpp_d2_k64_ilo_structure_20260723/"
    "s2_d2k64_pnxr_mixed.json"
)
CONFIG_DIR = REPO / "training_wss_min/configs/pointnetpp_d2_k64_ilo_structure_20260723"
MANIFEST = REPO / "training_wss_min/preflight/d2_k64_ilo_structure_wave2_prepared.json"
EXPECTED_SPLIT_SHA = "d16fc4978b82661e45e7863b19ae5ae5407528f7000c9aeeb9be01b904bdd8f1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def clone(parent: dict, experiment_id: str, note: str) -> dict:
    cfg = copy.deepcopy(parent)
    cfg["name"] = f"pointnetpp_d2_k64_ilo_structure/outputs/{experiment_id}"
    cfg["notes"] = note
    model = cfg["model"]
    model["local_geope"] = False
    model["local_geope_feature_indices"] = [3, 4, 5]
    model["coarse_attention"] = False
    model["coarse_attention_heads"] = 4
    model["query_decoder"] = "interpolate"
    model["sep_hidden"] = 32
    model["sep_beta_init"] = 2.0
    cfg["data"]["query_mode"] = "same"
    cfg["data"]["query_n_points"] = 5000
    cfg["data"]["query_sampling"] = "random"
    cfg["train"]["seed"] = 1234
    cfg["eval"]["support_seed"] = 1234
    return cfg


def assert_common_contract(cfg: ExpConfig) -> None:
    expected = {
        "split_sha": EXPECTED_SPLIT_SHA,
        "seed": 1234,
        "features": ("x", "y", "z", "abscissa_norm", "local_radius", "curvature"),
        "centers": (125, 125, 32),
        "neighbors": (64, 16, 16),
        "grouping": ("knn_cover", "ball", "ball"),
        "blocks": (1, 1, 0),
        "dropout": 0.0,
        "epochs": 400,
    }
    actual = {
        "split_sha": sha(Path(cfg.data.split_path)),
        "seed": cfg.train.seed,
        "features": tuple(cfg.data.input_features),
        "centers": tuple(cfg.model.sa_center_counts),
        "neighbors": tuple(cfg.model.sa_nsample),
        "grouping": tuple(cfg.model.sa_grouping),
        "blocks": tuple(cfg.model.sa_blocks),
        "dropout": float(cfg.model.dropout),
        "epochs": cfg.train.epochs,
    }
    if actual != expected:
        raise RuntimeError(f"wave2 common contract drift: {actual} != {expected}")


def main() -> None:
    if not S2_CONFIG.is_file():
        raise FileNotFoundError(S2_CONFIG)
    parent = json.loads(S2_CONFIG.read_text(encoding="utf-8"))
    rows: list[tuple[str, str, dict]] = []

    s3 = clone(
        parent,
        "s3_d2k64_pnxr_geope_mixed",
        "S3: S2 plus LocalGeoPE v1 only; normalized relative xyz/distance and "
        "delta(abscissa,radius,curvature); mixed train138/test36; no Drop.",
    )
    s3["model"]["local_geope"] = True
    rows.append(("structure_geope", "s3_d2k64_pnxr_geope_mixed", s3))

    s4 = clone(
        parent,
        "s4_d2k64_pnxr_attn_mixed",
        "S4: S2 plus one 4-head relative-bias global block at SA3 32 centers "
        "only; mixed train138/test36; no Drop.",
    )
    s4["model"]["coarse_attention"] = True
    rows.append(("structure_coarse_attention", "s4_d2k64_pnxr_attn_mixed", s4))

    s5c = clone(
        parent,
        "s5c_d2k64_pnxr_independent_interp_mixed",
        "S5C: matched independent-query control for S5; S2 plus query_mode="
        "independent and unchanged fixed 3-NN inverse-distance-squared interpolation.",
    )
    s5c["data"]["query_mode"] = "independent"
    rows.append(("sep_matched_control", "s5c_d2k64_pnxr_independent_interp_mixed", s5c))

    s5 = clone(
        parent,
        "s5_d2k64_pnxr_sep_mixed",
        "S5: S5C plus Conservative SEP-Kernel only; convex 3-NN weights, "
        "beta=2 and zero correction initialize exactly as parent interpolation.",
    )
    s5["data"]["query_mode"] = "independent"
    s5["model"]["query_decoder"] = "sep_kernel"
    rows.append(("structure_conservative_sep", "s5_d2k64_pnxr_sep_mixed", s5))

    config_rows = []
    for family, experiment_id, payload in rows:
        path = CONFIG_DIR / f"{experiment_id}.json"
        write_json(path, payload)
        cfg = ExpConfig.from_json(path)
        assert_common_contract(cfg)
        config_rows.append({
            "family": family,
            "experiment_id": experiment_id,
            "config": str(path.resolve()),
            "sha256": sha(path),
            "run_dir": str(cfg.run_dir.resolve()),
            "seed": cfg.train.seed,
            "split": str(Path(cfg.data.split_path).resolve()),
            "split_sha256": sha(Path(cfg.data.split_path)),
            "target_stats": str(Path(cfg.data.wss_stats_path).resolve()),
            "target_stats_sha256": sha(Path(cfg.data.wss_stats_path)),
            "feature_stats": str(Path(cfg.data.feature_stats_path).resolve()),
            "feature_stats_sha256": sha(Path(cfg.data.feature_stats_path)),
            "query_mode": cfg.data.query_mode,
            "query_decoder": cfg.model.query_decoder,
            "local_geope": cfg.model.local_geope,
            "coarse_attention": cfg.model.coarse_attention,
        })

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": "d2_k64_ilo_structure_wave2_single_seed_mixed_split",
        "seed_policy": {"train": [1234], "deferred_confirmation": [2025, 7]},
        "common_parent": {
            "experiment_id": "s2_d2k64_pnxr_mixed",
            "config": str(S2_CONFIG.resolve()),
            "config_sha256": sha(S2_CONFIG),
        },
        "methodological_note": (
            "SEP is inactive under SAME identity decoding. S5C is therefore the "
            "required independent-query interpolation control; S5C/S5 differ only "
            "in query_decoder."
        ),
        "configs": config_rows,
        "expected_new_jobs": 4,
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({
        "status": "prepared",
        "configs": len(config_rows),
        "manifest": str(MANIFEST),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
