#!/usr/bin/env python3
"""从签核真源和训练质量白名单生成 v4 splits 与 train-only WSS 统计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EPS = 1e-6
FRAME = "stl_landmarks_v4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def canonical_ag(label: str) -> str:
    parts = label.split("/")
    if len(parts) == 2 and parts[0] in {"fast", "slow"}:
        return "AG/" + label
    if len(parts) == 3 and parts[0] == "AG":
        return label
    raise ValueError(label)


def make_stats(data_root: Path, units: list[str], split_path: Path, output: Path) -> dict:
    chunks: list[np.ndarray] = []
    rows = []
    n_zero = 0
    for unit in units:
        bp = data_root / unit / "bundle.npz"
        with np.load(bp, allow_pickle=False) as z:
            frame = str(np.asarray(z["transform_frame_version"]).item())
            if frame != FRAME:
                raise RuntimeError(f"wrong frame: {unit}={frame}")
            steps = z["steps"].tolist(); peak = int(np.asarray(z["peak_step"]).item())
            wss = np.asarray(z["wall_wss"][steps.index(peak)], dtype=np.float64).ravel()
        if not np.isfinite(wss).all() or (wss < 0).any() or (wss <= 0).mean() > 0.01:
            raise RuntimeError(f"WSS gate failed while computing stats: {unit}")
        chunks.append(wss.astype(np.float32))
        n_zero += int((wss <= 0).sum())
        rows.append({"unit_id": unit, "bundle_path": str(bp.resolve()), "bundle_sha256": sha256(bp),
                     "peak_step": peak, "n_points": len(wss)})
    values = np.concatenate(chunks).astype(np.float64)
    logs = np.log(np.clip(values, 0, None) + EPS)
    stats = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "log_z", "eps": EPS, "required_frame_version": FRAME,
        "timesteps_scope": "peak", "statistics_scope": "train partition only",
        "split_path": str(split_path.resolve()), "split_sha256": sha256(split_path),
        "n_cases": len(units), "n_points": int(values.size),
        "zero_frac": float(n_zero / values.size),
        "linear": {"mean": float(values.mean()), "std": float(values.std())},
        "log": {"mean": float(logs.mean()), "std": float(logs.std())},
        "raw_min": float(values.min()), "raw_max": float(values.max()),
        "raw_percentiles": {f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 95, 99)},
        "train_units": units, "bundle_manifest": rows,
    }
    write_json(output, stats)
    return stats


def run(repo: Path) -> dict:
    data_root = repo / "data_wss_min"
    report_dir = data_root / "pipeline_reports/v4_cutover_20260715_1921"
    old_split_path = repo / "training/splits/split_AG_wss_min_v1_traintest.json"
    truth_path = report_dir / "v4_final_whitelist.json"
    train_wl_path = report_dir / "v4_training_whitelist.json"
    decisions_path = report_dir / "manual_review_decisions_20260716.json"
    quality_path = report_dir / "v4_training_quality_exclusions.json"
    old = json.loads(old_split_path.read_text()); truth = json.loads(truth_path.read_text())
    train_wl = json.loads(train_wl_path.read_text()); quality = json.loads(quality_path.read_text())
    ag_train = [canonical_ag(x) for x in old["train_cases"]]
    ag_test = [canonical_ag(x) for x in old["test_cases"] if not x.endswith("/WANG_DENG_FENG")]
    aaa_train = list(train_wl["AAA"])
    if (len(ag_train), len(ag_test), len(aaa_train)) != (61, 15, 57):
        raise RuntimeError("unexpected v4 partition counts")
    if not set(ag_train + ag_test).issubset(truth["AG"]):
        raise RuntimeError("AG split contains unit outside final AG76")
    final_exclusions = json.loads((report_dir / "v4_final_exclusions.json").read_text())
    geometry_excluded = [x["unit_id"] for x in final_exclusions["manual_exclusions"]]
    quality_excluded = [x["unit_id"] for x in quality["excluded_units"]]
    forbidden = set(geometry_excluded + quality_excluded)
    if forbidden & set(ag_train + ag_test + aaa_train):
        raise RuntimeError("excluded unit leaked into v4 partitions")
    common = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "pipeline_wss_min", "data_root": str(data_root.resolve()),
        "required_frame_version": FRAME, "id_format": "canonical cohort/subset/case",
    }
    ag_split = {
        **common, "split_version": "split_AG_wss_min_v4_traintest",
        "inherits_partition_from": str(old_split_path.resolve()),
        "derivation": "preserve historical train61; remove only AG/slow/WANG_DENG_FENG from historical test16",
        "train_cases": ag_train, "val_cases": [], "test_cases": ag_test,
        "excluded_cases": sorted(forbidden),
        "counts": {"train": 61, "val": 0, "test": 15},
        "expected_counts": {"train": 61, "val": 0, "test": 15},
        "fair_comparison": "do not compare old E2 test16 directly; use saved-prediction common-test15 anchor",
    }
    ag_path = repo / "training/splits/split_AG_wss_min_v4_traintest.json"
    write_json(ag_path, ag_split)
    mixed = {
        **common, "split_version": "split_AG_AAA_wss_min_v4_trainpool",
        "source_manual_decisions": str(decisions_path.resolve()),
        "source_manual_decisions_sha256": sha256(decisions_path),
        "source_final_geometry_whitelist": str(truth_path.resolve()),
        "source_final_geometry_whitelist_sha256": sha256(truth_path),
        "source_training_whitelist": str(train_wl_path.resolve()),
        "source_training_whitelist_sha256": sha256(train_wl_path),
        "source_quality_exclusions": str(quality_path.resolve()),
        "source_quality_exclusions_sha256": sha256(quality_path),
        "AG_train_cases": ag_train, "AAA_train_cases": aaa_train,
        "locked_AG_test_cases": ag_test,
        "train_cases": ag_train + aaa_train, "val_cases": [], "test_cases": ag_test,
        "excluded_cases": sorted(forbidden),
        "counts": {"train": 118, "val": 0, "test": 15, "AG_train": 61, "AAA_train": 57,
                   "locked_AG_test": 15},
        "expected_counts": {"train": 118, "val": 0, "test": 15},
    }
    mixed_path = repo / "training/splits/split_AG_AAA_wss_min_v4_trainpool.json"
    write_json(mixed_path, mixed)
    stats_dir = data_root / "fold_stats/v4"
    ag_stats = make_stats(data_root, ag_train, ag_path,
                          stats_dir / "AG_v4_train61_global_stats.json")
    mixed_stats = make_stats(data_root, ag_train + aaa_train, mixed_path,
                             stats_dir / "AG_AAA_v4_trainpool118_global_stats.json")
    return {"AG_split": ag_split["counts"], "mixed_split": mixed["counts"],
            "AG_stats_cases": ag_stats["n_cases"], "mixed_stats_cases": mixed_stats["n_cases"]}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", type=Path, default=Path.cwd())
    args = ap.parse_args(); print(json.dumps(run(args.repo.resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
