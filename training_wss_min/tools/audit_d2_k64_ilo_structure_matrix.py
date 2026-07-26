#!/usr/bin/env python3
"""Static hard-gate audit for the D2-K64 ILO/structure wave-1 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.config import ExpConfig, input_dim
from training_wss_min.models import build_model


EXPECTED = {
    "f1_d2k64_ilo41_fixed_test27",
    "m0_d2k64_zeroshot_mixed36",
    "m1_d2k64_mixed138_test36",
    "s1_d2k64_mfeat_mixed",
    "s2_d2k64_pnxr_mixed",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(payload: dict, allowed: set[str]) -> dict:
    out = json.loads(json.dumps(payload))
    for dotted in allowed:
        parent = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            parent = parent[part]
        parent.pop(parts[-1], None)
    return out


def assert_only_diff(left: dict, right: dict, allowed: set[str], label: str) -> None:
    if normalized(left, allowed) != normalized(right, allowed):
        raise RuntimeError(f"{label} differs outside allowed fields {sorted(allowed)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared":
        raise RuntimeError("manifest is not prepared")
    rows = manifest.get("configs", [])
    if len(rows) != 5 or {row["experiment_id"] for row in rows} != EXPECTED:
        raise RuntimeError("wave1 must contain the exact five approved experiments")

    raw: dict[str, dict] = {}
    cfgs: dict[str, ExpConfig] = {}
    details = []
    for row in rows:
        path = Path(row["config"])
        if not path.is_file() or sha(path) != row["sha256"]:
            raise RuntimeError(f"missing or drifted config: {path}")
        if sha(Path(row["split"])) != row["split_sha256"]:
            raise RuntimeError(f"split hash drift: {row['split']}")
        if sha(Path(row["target_stats"])) != row["target_stats_sha256"]:
            raise RuntimeError(f"target stats drift: {row['target_stats']}")
        if sha(Path(row["feature_stats"])) != row["feature_stats_sha256"]:
            raise RuntimeError(f"feature stats drift: {row['feature_stats']}")
        cfg = ExpConfig.from_json(path)
        if cfg.run_dir.exists():
            raise FileExistsError(f"run already exists: {cfg.run_dir}")
        if cfg.train.seed != 1234:
            raise RuntimeError("non-1234 training seed entered wave1")
        if tuple(cfg.model.sa_center_counts) != (125, 125, 32):
            raise RuntimeError("center contract drift")
        if tuple(cfg.model.sa_nsample) != (64, 16, 16):
            raise RuntimeError("neighborhood contract drift")
        if tuple(cfg.model.sa_grouping) != ("knn_cover", "ball", "ball"):
            raise RuntimeError("grouping contract drift")
        if cfg.data.query_mode != "same" or cfg.model.query_decoder != "interpolate":
            raise RuntimeError("wave1 must retain SAME + interpolate")
        model = build_model(cfg.model, input_dim(cfg))
        params = sum(p.numel() for p in model.parameters())
        block_counts = [len(sa.blocks) for sa in model.sa]
        raw[row["experiment_id"]] = json.loads(path.read_text(encoding="utf-8"))
        cfgs[row["experiment_id"]] = cfg
        details.append({
            "experiment_id": row["experiment_id"],
            "config": str(path.resolve()),
            "config_sha256": row["sha256"],
            "parameters": params,
            "input_dim": input_dim(cfg),
            "sa_blocks_runtime": block_counts,
        })

    common_meta = {"name", "notes"}
    assert_only_diff(
        raw["m0_d2k64_zeroshot_mixed36"],
        raw["m1_d2k64_mixed138_test36"],
        common_meta | {"data.split_path"},
        "M0/M1",
    )
    assert_only_diff(
        raw["m1_d2k64_mixed138_test36"],
        raw["s1_d2k64_mfeat_mixed"],
        common_meta | {"data.input_features", "data.feature_stats_path"},
        "M1/S1",
    )
    assert_only_diff(
        raw["m1_d2k64_mixed138_test36"],
        raw["s2_d2k64_pnxr_mixed"],
        common_meta | {"model.sa_blocks"},
        "M1/S2",
    )

    by_id = {row["experiment_id"]: row for row in details}
    m0 = by_id["m0_d2k64_zeroshot_mixed36"]
    m1 = by_id["m1_d2k64_mixed138_test36"]
    s1 = by_id["s1_d2k64_mfeat_mixed"]
    s2 = by_id["s2_d2k64_pnxr_mixed"]
    if m0["parameters"] != m1["parameters"]:
        raise RuntimeError("M0/M1 parameter count mismatch")
    if not (s1["parameters"] > m1["parameters"] and s2["parameters"] > m1["parameters"]):
        raise RuntimeError("structure arms did not change parameter count")
    if m1["sa_blocks_runtime"] != [0, 0, 0] or s2["sa_blocks_runtime"] != [1, 1, 0]:
        raise RuntimeError("sa_blocks runtime trace mismatch")
    if s1["input_dim"] != 8 or m1["input_dim"] != 6:
        raise RuntimeError("M-FEAT input dimension mismatch")

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha(args.manifest),
        "checks": {
            "exact_five_rows": True,
            "single_seed_1234": True,
            "frozen_hashes": True,
            "d2_k64_contract": True,
            "m0_m1_only_split_diff": True,
            "m1_s1_only_feature_diff": True,
            "m1_s2_only_blocks_diff": True,
            "runtime_block_trace": True,
            "run_dirs_absent": True,
        },
        "configs": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "passed",
        "configs": len(details),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
