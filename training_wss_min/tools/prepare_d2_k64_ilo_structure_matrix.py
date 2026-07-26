#!/usr/bin/env python3
"""Prepare the approved single-seed D2-K64 ILO + structure screening matrix.

This writer reuses the audited ILO splits, freezes target/feature statistics
within each paired comparison, and creates five new seed1234 configs:
F1, M0, M1/S0, S1-MFEAT, and S2-PointNeXt-R.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min import dataset as D
from training_wss_min.config import ExpConfig


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data_wss_min"
BASE_CONFIG = (
    REPO / "training_wss_min/configs/pointnetpp_sa1_scale_single_seed_20260721/"
    "d2_rand5000_c125_k64.json"
)
BASE_RUN = (
    REPO / "training_wss_min/runs/pointnetpp_sa1_scale_single_seed/outputs/"
    "d2_rand5000_c125_k64"
)
BASE_FEATURE_STATS = BASE_RUN / "feature_stats.json"
BASE_TARGET_STATS = (
    DATA_ROOT / "fold_stats/v4/AG_AAA_v4_stratified_train106_global_stats.json"
)
BASE_SPLIT = REPO / "training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"

FIXED_ILO_SPLIT = (
    REPO / "training/splits/split_AG_AAA_ILO_q2v_fixed_test27_train147.json"
)
MIXED_CONTROL_SPLIT = (
    REPO / "training/splits/split_AG_AAA_ILO_q2v_pool2025_control106_test36.json"
)
MIXED_SPLIT = (
    REPO / "training/splits/split_AG_AAA_ILO_q2v_pool2025_train138_test36.json"
)
MIXED_TARGET_STATS = (
    DATA_ROOT / "fold_stats/q2v_ilo_20260718/q2v_pool2025_control106_global_stats.json"
)
MIXED_FEATURE_STATS = (
    DATA_ROOT / "fold_stats/q2v_ilo_20260718/q2v_pool2025_control106_feature_stats.json"
)
MFEAT_STATS = (
    DATA_ROOT / "fold_stats/d2_k64_ilo_structure_20260723/"
    "mixed_train138_feature_stats_8d.json"
)

CONFIG_DIR = REPO / "training_wss_min/configs/pointnetpp_d2_k64_ilo_structure_20260723"
MANIFEST = REPO / "training_wss_min/preflight/d2_k64_ilo_structure_wave1_prepared.json"

BASE_FEATURES = (
    "x", "y", "z", "abscissa_norm", "local_radius", "curvature",
)
MFEAT_FEATURES = (*BASE_FEATURES, "radius_gradient", "coord_scale")
EXPECTED_SPLIT_HASHES = {
    FIXED_ILO_SPLIT: "7f5a4f993d4fdf2e1d7f43aa107b232bf4b67b868f67c7b86f3421df62f2f7a6",
    MIXED_CONTROL_SPLIT: "7b01f8ceac2743d50cb5a3c97decfb0492c48d9f1cd589f20d2811c976f4887d",
    MIXED_SPLIT: "d16fc4978b82661e45e7863b19ae5ae5407528f7000c9aeeb9be01b904bdd8f1",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def cohort_counts(units: list[str]) -> dict[str, int]:
    out = {"AG": 0, "AAA/ruputer": 0, "AAA/unruputer": 0, "ILO/0": 0, "ILO/1": 0}
    for unit in units:
        if unit.startswith("AG/"):
            key = "AG"
        elif unit.startswith("AAA/ruputer/"):
            key = "AAA/ruputer"
        elif unit.startswith("AAA/unruputer/"):
            key = "AAA/unruputer"
        elif unit.startswith("ILO/"):
            key = f"ILO/{unit.split('/', 2)[1].rsplit('-', 1)[-1]}"
        else:
            raise ValueError(f"unknown unit {unit!r}")
        out[key] += 1
    return out


def validate_split(path: Path, expected: tuple[int, int, int, int]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parts = {
        name: list(payload.get(f"{name}_cases", []))
        for name in ("train", "val", "test", "unused")
    }
    actual = tuple(len(parts[name]) for name in ("train", "val", "test", "unused"))
    if actual != expected:
        raise RuntimeError(f"wrong split counts for {path.name}: {actual} != {expected}")
    sets = {name: set(values) for name, values in parts.items()}
    names = tuple(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if sets[left] & sets[right]:
                raise RuntimeError(f"split leakage {left}/{right}: {path}")
    active = set().union(*sets.values())
    if any(unit.startswith("ILO/") and not unit.endswith("/before") for unit in active):
        raise RuntimeError(f"non-before ILO unit in {path}")
    if not all((DATA_ROOT / unit / "bundle.npz").is_file() for unit in active):
        raise FileNotFoundError(f"missing bundle referenced by {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha(path),
        "counts": dict(zip(("train", "val", "test", "unused"), actual)),
        "train_strata": cohort_counts(parts["train"]),
        "test_strata": cohort_counts(parts["test"]),
        "unused_strata": cohort_counts(parts["unused"]),
    }


def clone(base: dict, *, experiment_id: str, note: str, split: Path,
          target_stats: Path, feature_stats: Path, features=BASE_FEATURES,
          blocks=(0, 0, 0)) -> dict:
    cfg = copy.deepcopy(base)
    cfg["name"] = f"pointnetpp_d2_k64_ilo_structure/outputs/{experiment_id}"
    cfg["notes"] = note
    cfg["data"]["split_path"] = str(split.resolve())
    cfg["data"]["wss_stats_path"] = str(target_stats.resolve())
    cfg["data"]["feature_stats_path"] = str(feature_stats.resolve())
    cfg["data"]["input_features"] = list(features)
    cfg["data"]["query_mode"] = "same"
    cfg["data"]["query_n_points"] = 5000
    cfg["data"]["query_sampling"] = "random"
    cfg["model"]["sa_blocks"] = list(blocks)
    cfg["model"]["invres_expansion"] = 2
    cfg["model"]["residual_scale_init"] = 1e-3
    cfg["model"]["invres_radius_scale"] = 1.0
    cfg["model"]["query_decoder"] = "interpolate"
    cfg["model"]["dropout"] = 0
    cfg["train"]["seed"] = 1234
    cfg["eval"]["support_seed"] = 1234
    return cfg


def assert_d2_contract(cfg: ExpConfig) -> None:
    if tuple(cfg.data.input_features[:6]) != BASE_FEATURES:
        raise RuntimeError("D2-K64 base feature order drift")
    expected = {
        "support": 5000,
        "query_mode": "same",
        "centers": (125, 125, 32),
        "nsample": (64, 16, 16),
        "grouping": ("knn_cover", "ball", "ball"),
        "width": 32,
        "seed": 1234,
        "epochs": 400,
    }
    actual = {
        "support": cfg.data.support_n_points,
        "query_mode": cfg.data.query_mode,
        "centers": tuple(cfg.model.sa_center_counts),
        "nsample": tuple(cfg.model.sa_nsample),
        "grouping": tuple(cfg.model.sa_grouping),
        "width": cfg.model.width,
        "seed": cfg.train.seed,
        "epochs": cfg.train.epochs,
    }
    if actual != expected:
        raise RuntimeError(f"D2-K64 contract drift: {actual} != {expected}")


def build_mfeat_stats() -> None:
    target_stats = D.load_wss_stats(MIXED_TARGET_STATS)
    train = D.load_partition(
        str(MIXED_SPLIT), "train", target_stats, strict=True,
        data_root=DATA_ROOT, required_frame_version="stl_landmarks_v4",
    )
    stats = D.compute_feature_stats(train, MFEAT_FEATURES, "signed_log1p")
    if set(stats) != {"abscissa_norm", "local_radius", "curvature",
                      "radius_gradient", "coord_scale"}:
        raise RuntimeError(f"unexpected M-FEAT stats keys: {sorted(stats)}")
    write_json(MFEAT_STATS, stats)


def main() -> None:
    required = (
        BASE_CONFIG, BASE_FEATURE_STATS, BASE_TARGET_STATS, BASE_SPLIT,
        FIXED_ILO_SPLIT, MIXED_CONTROL_SPLIT, MIXED_SPLIT,
        MIXED_TARGET_STATS, MIXED_FEATURE_STATS,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path, expected_hash in EXPECTED_SPLIT_HASHES.items():
        if sha(path) != expected_hash:
            raise RuntimeError(f"audited split hash drift: {path}")

    splits = [
        validate_split(FIXED_ILO_SPLIT, (147, 0, 27, 0)),
        validate_split(MIXED_CONTROL_SPLIT, (106, 0, 36, 32)),
        validate_split(MIXED_SPLIT, (138, 0, 36, 0)),
    ]
    fixed = json.loads(FIXED_ILO_SPLIT.read_text(encoding="utf-8"))
    mixed_control = json.loads(MIXED_CONTROL_SPLIT.read_text(encoding="utf-8"))
    mixed = json.loads(MIXED_SPLIT.read_text(encoding="utf-8"))
    if fixed["test_cases"] != json.loads(BASE_SPLIT.read_text(encoding="utf-8"))["test_cases"]:
        raise RuntimeError("fixed ILO protocol changed the historical test27")
    if mixed_control["test_cases"] != mixed["test_cases"]:
        raise RuntimeError("mixed control/treatment test36 mismatch")
    if set(mixed_control["unused_cases"]) != {
        unit for unit in mixed["train_cases"] if unit.startswith("ILO/")
    }:
        raise RuntimeError("mixed control unused ILO32 mismatch")

    build_mfeat_stats()
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    rows = (
        ("data_fixed", "f1_d2k64_ilo41_fixed_test27",
         "F1: D2-K64; add all approved ILO-before41 to train; fixed historical AG/AAA test27; frozen parent target+feature stats.",
         FIXED_ILO_SPLIT, BASE_TARGET_STATS, BASE_FEATURE_STATS, BASE_FEATURES, (0, 0, 0)),
        ("data_mixed_control", "m0_d2k64_zeroshot_mixed36",
         "M0: D2-K64 mixed-test control; AG/AAA train106, ILO32 unused, common AG/AAA/ILO test36.",
         MIXED_CONTROL_SPLIT, MIXED_TARGET_STATS, MIXED_FEATURE_STATS, BASE_FEATURES, (0, 0, 0)),
        ("data_mixed", "m1_d2k64_mixed138_test36",
         "M1/S0: D2-K64 common mixed parent; AG/AAA/ILO train138 and common mixed test36; frozen M0 stats.",
         MIXED_SPLIT, MIXED_TARGET_STATS, MIXED_FEATURE_STATS, BASE_FEATURES, (0, 0, 0)),
        ("structure_mfeat", "s1_d2k64_mfeat_mixed",
         "S1: M1 plus radius_gradient and coord_scale only; mixed train138/test36.",
         MIXED_SPLIT, MIXED_TARGET_STATS, MFEAT_STATS, MFEAT_FEATURES, (0, 0, 0)),
        ("structure_pnxr", "s2_d2k64_pnxr_mixed",
         "S2: M1 plus PointNeXt-R blocks=(1,1,0), expansion=2, gamma=1e-3; no Drop.",
         MIXED_SPLIT, MIXED_TARGET_STATS, MIXED_FEATURE_STATS, BASE_FEATURES, (1, 1, 0)),
    )

    config_rows = []
    for family, experiment_id, note, split, target_stats, feature_stats, features, blocks in rows:
        payload = clone(
            base, experiment_id=experiment_id, note=note, split=split,
            target_stats=target_stats, feature_stats=feature_stats,
            features=features, blocks=blocks,
        )
        path = CONFIG_DIR / f"{experiment_id}.json"
        write_json(path, payload)
        cfg = ExpConfig.from_json(path)
        assert_d2_contract(cfg)
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
            "input_features": list(cfg.data.input_features),
            "sa_blocks": list(cfg.model.sa_blocks),
        })

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": "d2_k64_ilo_structure_wave1_single_seed_parallel",
        "seed_policy": {"train": [1234], "deferred_confirmation": [2025, 7]},
        "parent": {
            "experiment_id": "d2_rand5000_c125_k64",
            "config": str(BASE_CONFIG.resolve()),
            "config_sha256": sha(BASE_CONFIG),
            "run_dir": str(BASE_RUN.resolve()),
            "feature_stats": str(BASE_FEATURE_STATS.resolve()),
            "feature_stats_sha256": sha(BASE_FEATURE_STATS),
        },
        "split_audit": splits,
        "sources": [
            {"path": str(path.resolve()), "sha256": sha(path)}
            for path in required
        ],
        "generated_feature_stats": {
            "path": str(MFEAT_STATS.resolve()),
            "sha256": sha(MFEAT_STATS),
            "source_split": str(MIXED_SPLIT.resolve()),
            "features": list(MFEAT_FEATURES),
        },
        "configs": config_rows,
        "expected_new_jobs": 5,
        "deferred_wave2": [
            "s3_pnxr_geope_mixed",
            "s4_pnxr_attention_mixed",
            "s5_pnxr_sep_mixed",
        ],
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({
        "status": "prepared",
        "configs": len(config_rows),
        "manifest": str(MANIFEST),
        "mfeat_stats": str(MFEAT_STATS),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
