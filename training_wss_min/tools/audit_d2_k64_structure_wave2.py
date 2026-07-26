#!/usr/bin/env python3
"""Static hard-gate audit for the D2-K64 structure wave-2 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min.baseline_models import CoarseGlobalBlock, ConservativeSEPKernel
from training_wss_min.config import ExpConfig, input_dim
from training_wss_min.models import build_model


EXPECTED = {
    "s3_d2k64_pnxr_geope_mixed",
    "s4_d2k64_pnxr_attn_mixed",
    "s5c_d2k64_pnxr_independent_interp_mixed",
    "s5_d2k64_pnxr_sep_mixed",
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
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared":
        raise RuntimeError("manifest is not prepared")
    if len(rows) != 4 or {row["experiment_id"] for row in rows} != EXPECTED:
        raise RuntimeError("wave2 must contain the exact four auditable experiments")

    parent_path = Path(manifest["common_parent"]["config"])
    if sha(parent_path) != manifest["common_parent"]["config_sha256"]:
        raise RuntimeError("S2 parent config drift")
    parent_raw = json.loads(parent_path.read_text(encoding="utf-8"))
    raw: dict[str, dict] = {}
    details = []
    common_hashes = None

    for row in rows:
        path = Path(row["config"])
        if not path.is_file() or sha(path) != row["sha256"]:
            raise RuntimeError(f"missing or drifted config: {path}")
        hashes = (
            row["split_sha256"],
            row["target_stats_sha256"],
            row["feature_stats_sha256"],
        )
        if common_hashes is None:
            common_hashes = hashes
        elif hashes != common_hashes:
            raise RuntimeError("wave2 split/stats hashes are not frozen")
        for key in ("split", "target_stats", "feature_stats"):
            if sha(Path(row[key])) != row[f"{key}_sha256"]:
                raise RuntimeError(f"{key} hash drift: {row[key]}")
        cfg = ExpConfig.from_json(path)
        if cfg.run_dir.exists():
            raise FileExistsError(f"run already exists: {cfg.run_dir}")
        if cfg.train.seed != 1234 or tuple(cfg.model.sa_blocks) != (1, 1, 0):
            raise RuntimeError("wave2 seed or PointNeXt-R parent drift")
        if tuple(cfg.model.sa_center_counts) != (125, 125, 32):
            raise RuntimeError("wave2 center contract drift")
        if tuple(cfg.model.sa_nsample) != (64, 16, 16):
            raise RuntimeError("wave2 neighbor contract drift")
        if tuple(cfg.model.sa_grouping) != ("knn_cover", "ball", "ball"):
            raise RuntimeError("wave2 grouping contract drift")
        if float(cfg.model.dropout) != 0.0:
            raise RuntimeError("Drop entered wave2")

        model = build_model(cfg.model, input_dim(cfg))
        raw[row["experiment_id"]] = json.loads(path.read_text(encoding="utf-8"))
        details.append({
            "experiment_id": row["experiment_id"],
            "config": str(path.resolve()),
            "config_sha256": row["sha256"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "local_geope_runtime": all(sa.geo_pe is not None for sa in model.sa),
            "coarse_attention_runtime": isinstance(model.coarse_global, CoarseGlobalBlock),
            "sep_runtime": isinstance(model.sep_kernel, ConservativeSEPKernel),
            "query_mode": cfg.data.query_mode,
            "query_decoder": cfg.model.query_decoder,
        })

    meta = {"name", "notes"}
    s3 = "s3_d2k64_pnxr_geope_mixed"
    s4 = "s4_d2k64_pnxr_attn_mixed"
    s5c = "s5c_d2k64_pnxr_independent_interp_mixed"
    s5 = "s5_d2k64_pnxr_sep_mixed"
    defaults = {
        "model.local_geope", "model.local_geope_feature_indices",
        "model.coarse_attention", "model.coarse_attention_heads",
        "model.sep_hidden", "model.sep_beta_init",
    }
    assert_only_diff(parent_raw, raw[s3], meta | defaults, "S2/S3")
    assert_only_diff(parent_raw, raw[s4], meta | defaults, "S2/S4")
    assert_only_diff(
        parent_raw, raw[s5c], meta | defaults | {"data.query_mode"}, "S2/S5C"
    )
    assert_only_diff(raw[s5c], raw[s5], meta | {"model.query_decoder"}, "S5C/S5")

    by_id = {row["experiment_id"]: row for row in details}
    if not by_id[s3]["local_geope_runtime"] or by_id[s3]["coarse_attention_runtime"]:
        raise RuntimeError("S3 runtime trace mismatch")
    if not by_id[s4]["coarse_attention_runtime"] or by_id[s4]["local_geope_runtime"]:
        raise RuntimeError("S4 runtime trace mismatch")
    if by_id[s5c]["query_mode"] != "independent" or by_id[s5]["query_mode"] != "independent":
        raise RuntimeError("SEP matched pair must train on independent queries")
    if by_id[s5c]["sep_runtime"] or not by_id[s5]["sep_runtime"]:
        raise RuntimeError("SEP runtime trace mismatch")
    if by_id[s5]["parameters"] <= by_id[s5c]["parameters"]:
        raise RuntimeError("SEP treatment did not add parameters")

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha(args.manifest),
        "checks": {
            "exact_four_rows": True,
            "single_seed_1234": True,
            "mixed_split_and_frozen_stats": True,
            "d2_k64_pnxr_contract": True,
            "drop_absent": True,
            "s2_s3_only_geope": True,
            "s2_s4_only_attention": True,
            "s5c_s5_only_decoder": True,
            "runtime_module_trace": True,
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
