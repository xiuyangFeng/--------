#!/usr/bin/env python3
"""Prepare the three-arm RCR information oracle on the frozen mixed 138/0/36 S3 protocol."""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from training_wss_min.config import CASE_FEATURE_KEYS, ExpConfig


ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "training/splits/split_AG_AAA_ILO_q2v_pool2025_train138_test36.json"
BASE = (
    ROOT
    / "training_wss_min/configs/pointnetpp_d2_k64_ilo_structure_20260723"
    / "s3_d2k64_pnxr_geope_mixed.json"
)
BASE_STATS = (
    ROOT
    / "data_wss_min/fold_stats/q2v_ilo_20260718"
    / "q2v_pool2025_control106_feature_stats.json"
)
DATA_NEW = ROOT / "data_new"
CONFIG_DIR = ROOT / "training_wss_min/configs/pointnetpp_rcr_oracle_20260728"
PREFLIGHT = ROOT / "training_wss_min/preflight"
TRUE_FEATURES = PREFLIGHT / "rcr_oracle_true_case_features_20260728.json"
SHUFFLED_FEATURES = PREFLIGHT / "rcr_oracle_shuffled_case_features_20260728.json"
FEATURE_STATS = PREFLIGHT / "rcr_oracle_feature_stats_20260728.json"
MANIFEST = PREFLIGHT / "rcr_oracle_matrix_20260728_prepared.json"
STATIC_AUDIT = PREFLIGHT / "rcr_oracle_matrix_20260728_static_audit.json"

OUTLETS = ("outle", "outli", "outri", "outre")
PARAMETERS = ("r1", "r2", "c")
SHUFFLE_SEED = 20260728


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def source_candidates(unit_id: str) -> list[Path]:
    case_dir = DATA_NEW / unit_id
    direct = sorted(case_dir.glob("*.c"))
    if direct:
        return direct
    return sorted((case_dir / "libudf/src").glob("*.c"))


def parse_rcr_source(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    row: dict[str, float] = {}
    for outlet in OUTLETS:
        match = re.search(
            rf"DEFINE_PROFILE\s*\(\s*pressure_{outlet}\b"
            rf"(?P<body>.*?)(?=\n\s*DEFINE_PROFILE\s*\(|\Z)",
            text,
            flags=re.S,
        )
        if match is None:
            raise ValueError(f"missing pressure_{outlet} block")
        body = match.group("body")
        for parameter, source_name in zip(PARAMETERS, ("R1", "R2", "C")):
            value_match = re.search(
                rf"\b{source_name}\s*=\s*"
                rf"([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?)\s*;",
                body,
            )
            if value_match is None:
                raise ValueError(f"missing {source_name} for pressure_{outlet}")
            value = float(value_match.group(1))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {source_name}={value} for pressure_{outlet}")
            row[f"log_rcr_{outlet}_{parameter}"] = math.log(value)
    if tuple(row) != tuple(CASE_FEATURE_KEYS):
        raise RuntimeError("RCR feature order drift")
    return row


def extract_all(unit_ids: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, dict]]:
    cases: dict[str, dict[str, float]] = {}
    provenance: dict[str, dict] = {}
    failures = []
    for unit_id in unit_ids:
        valid = []
        errors = []
        for source in source_candidates(unit_id):
            try:
                valid.append((source, parse_rcr_source(source)))
            except Exception as exc:
                errors.append(f"{source}: {type(exc).__name__}: {exc}")
        if len(valid) != 1:
            failures.append(
                {"unit_id": unit_id, "valid_sources": [str(x[0]) for x in valid], "errors": errors}
            )
            continue
        source, row = valid[0]
        cases[unit_id] = row
        provenance[unit_id] = {
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
        }
    if failures:
        raise RuntimeError(f"RCR extraction failed: {json.dumps(failures, ensure_ascii=False)}")
    return cases, provenance


def stratum(unit_id: str) -> str:
    parts = unit_id.split("/")
    if parts[0] == "AG":
        return "AG"
    if parts[0] == "AAA":
        return f"AAA/{parts[1]}"
    if parts[0] == "ILO":
        return f"ILO/{parts[1].rsplit('-', 1)[-1]}"
    raise ValueError(f"unsupported unit id: {unit_id}")


def shuffled_rows(
    true_rows: dict[str, dict[str, float]], split: dict
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled: dict[str, dict[str, float]] = {}
    donors: dict[str, str] = {}
    for partition in ("train_cases", "test_cases"):
        groups: dict[str, list[str]] = {}
        for unit_id in split[partition]:
            groups.setdefault(stratum(unit_id), []).append(unit_id)
        for group in sorted(groups):
            recipients = sorted(groups[group])
            if len(recipients) < 2:
                raise RuntimeError(f"cannot shuffle singleton group {partition}/{group}")
            order = list(rng.permutation(recipients))
            shift = int(rng.integers(1, len(order)))
            donor_order = order[shift:] + order[:shift]
            mapping = dict(zip(order, donor_order))
            if any(recipient == donor for recipient, donor in mapping.items()):
                raise RuntimeError(f"shuffle is not a derangement: {partition}/{group}")
            for recipient in recipients:
                donor = mapping[recipient]
                shuffled[recipient] = dict(true_rows[donor])
                donors[recipient] = donor
    return shuffled, donors


def make_feature_stats(
    base_stats: dict, true_rows: dict[str, dict[str, float]], train_ids: list[str]
) -> dict:
    merged = deepcopy(base_stats)
    for feature in CASE_FEATURE_KEYS:
        values = np.asarray([true_rows[unit_id][feature] for unit_id in train_ids], dtype=np.float64)
        merged[feature] = {
            "mean": float(values.mean()),
            "std": float(values.std() + 1e-6),
            "transform": "none",
            "scope": "train138_case_equal",
        }
    return merged


def make_configs() -> list[Path]:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    geometry_features = list(base["data"]["input_features"])
    condition_features = list(CASE_FEATURE_KEYS)
    rows = [
        (
            "o0_geometry_s1234",
            "O0 geometry-only paired control; exact S3 PointNeXt-R + LocalGeoPE parent.",
            None,
            geometry_features,
            str(BASE_STATS.resolve()),
        ),
        (
            "o1a_true_rcr_s1234",
            "O1a legal information oracle: S3 parent plus true four-outlet R1/R2/C.",
            TRUE_FEATURES,
            geometry_features + condition_features,
            str(FEATURE_STATS.resolve()),
        ),
        (
            "o2_shuffled_rcr_s1234",
            "O2 negative control: S3 parent plus partition/stratum-preserving shuffled RCR.",
            SHUFFLED_FEATURES,
            geometry_features + condition_features,
            str(FEATURE_STATS.resolve()),
        ),
    ]
    paths = []
    for slug, notes, case_features, input_features, feature_stats in rows:
        cfg = deepcopy(base)
        cfg["name"] = f"pointnetpp_rcr_oracle/outputs/{slug}"
        cfg["notes"] = (
            notes
            + " Frozen mixed 138/0/36 development protocol; test36 is reused and "
            "therefore not an unbiased final estimate."
        )
        cfg["data"]["input_features"] = input_features
        cfg["data"]["feature_stats_path"] = feature_stats
        cfg["data"]["case_features_path"] = (
            str(case_features.resolve()) if case_features is not None else None
        )
        cfg["eval"]["surface_metric_mode"] = "legacy_vertex"
        path = CONFIG_DIR / f"{slug}.json"
        dump(path, cfg)
        ExpConfig.from_json(path)
        paths.append(path)
    return paths


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_ids = list(split["train_cases"])
    test_ids = list(split["test_cases"])
    unit_ids = train_ids + test_ids
    if (len(train_ids), len(split.get("val_cases", [])), len(test_ids)) != (138, 0, 36):
        raise RuntimeError("expected exact mixed 138/0/36 split")

    true_rows, provenance = extract_all(unit_ids)
    shuffled, donors = shuffled_rows(true_rows, split)
    base_stats = json.loads(BASE_STATS.read_text(encoding="utf-8"))
    feature_stats = make_feature_stats(base_stats, true_rows, train_ids)

    common = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": list(CASE_FEATURE_KEYS),
        "feature_transform": "natural logarithm of positive UDF R1/R2/C",
        "split": str(SPLIT.resolve()),
        "split_sha256": sha256(SPLIT),
    }
    dump(
        TRUE_FEATURES,
        {
            **common,
            "kind": "true_rcr_input",
            "cases": true_rows,
            "provenance": provenance,
        },
    )
    dump(
        SHUFFLED_FEATURES,
        {
            **common,
            "kind": "negative_control_shuffled_rcr",
            "shuffle_seed": SHUFFLE_SEED,
            "shuffle_policy": "independent train/test; within AG, AAA subtype, or ILO 0/1; derangement",
            "cases": shuffled,
            "donor_by_recipient": donors,
        },
    )
    dump(FEATURE_STATS, feature_stats)
    config_paths = make_configs()

    manifest_rows = []
    for index, path in enumerate(config_paths):
        cfg = ExpConfig.from_json(path)
        manifest_rows.append(
            {
                "index": index,
                "experiment_id": path.stem,
                "config": str(path.resolve()),
                "config_sha256": sha256(path),
                "run_dir": str(cfg.run_dir.resolve()),
                "input_dim": len(cfg.data.input_features),
                "case_features_path": cfg.data.case_features_path,
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": "S3 PointNeXt-R + LocalGeoPE; mixed 138/0/36; seed1234; legacy_vertex",
        "split": str(SPLIT.resolve()),
        "split_sha256": sha256(SPLIT),
        "base_config": str(BASE.resolve()),
        "base_config_sha256": sha256(BASE),
        "true_case_features": str(TRUE_FEATURES.resolve()),
        "true_case_features_sha256": sha256(TRUE_FEATURES),
        "shuffled_case_features": str(SHUFFLED_FEATURES.resolve()),
        "shuffled_case_features_sha256": sha256(SHUFFLED_FEATURES),
        "feature_stats": str(FEATURE_STATS.resolve()),
        "feature_stats_sha256": sha256(FEATURE_STATS),
        "configs": manifest_rows,
    }
    dump(MANIFEST, manifest)

    true_matrix = np.asarray(
        [[true_rows[unit_id][feature] for feature in CASE_FEATURE_KEYS] for unit_id in train_ids]
    )
    shuffled_matrix = np.asarray(
        [[shuffled[unit_id][feature] for feature in CASE_FEATURE_KEYS] for unit_id in train_ids]
    )
    static = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "checks": {
            "split_counts": [138, 0, 36],
            "rcr_cases": len(true_rows),
            "rcr_features": len(CASE_FEATURE_KEYS),
            "all_sources_unique": len({row["source"] for row in provenance.values()}) == len(unit_ids),
            "shuffle_has_no_fixed_points": all(k != v for k, v in donors.items()),
            "shuffle_preserves_train_column_multisets": bool(
                np.allclose(np.sort(true_matrix, axis=0), np.sort(shuffled_matrix, axis=0))
            ),
            "config_input_dims": [row["input_dim"] for row in manifest_rows],
            "surface_metric_mode": "legacy_vertex",
        },
    }
    if static["checks"]["config_input_dims"] != [6, 18, 18]:
        raise RuntimeError("unexpected oracle config dimensions")
    if not all(
        (
            static["checks"]["all_sources_unique"],
            static["checks"]["shuffle_has_no_fixed_points"],
            static["checks"]["shuffle_preserves_train_column_multisets"],
        )
    ):
        raise RuntimeError("static oracle audit failed")
    dump(STATIC_AUDIT, static)
    print(json.dumps({"status": "prepared", "manifest": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
