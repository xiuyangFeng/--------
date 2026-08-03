"""Merge full-wall train shards and derive the frozen v2 WSS scalar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from compare_multiscale_v2 import aggregate  # noqa: E402


def scale_weight(report: dict) -> float:
    """Recover ||pred||² / truth variance from raw/scaled R² and alpha."""
    alpha = float(report["alpha"])
    denominator = (alpha - 1.0) ** 2
    if denominator < 1e-12:
        return 0.0
    return max(
        float(report["scaled_r2"] - report["raw_r2"]) / denominator,
        0.0,
    )


def derive_scale(cases: list[dict], method: str = "multiscale_v2") -> float:
    """Maximize mean case-level R² for pred -> scale * pred."""
    weights = np.asarray(
        [scale_weight(case["methods"][method]) for case in cases],
        dtype=np.float64,
    )
    alpha = np.asarray(
        [case["methods"][method]["alpha"] for case in cases],
        dtype=np.float64,
    )
    if float(weights.sum()) <= 0:
        raise ValueError("could not recover positive scale weights")
    return float(np.dot(weights, alpha) / weights.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    shards = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    if not shards:
        raise ValueError("no shard inputs")
    split_sha = {shard["split"]["sha256"] for shard in shards}
    if len(split_sha) != 1:
        raise ValueError(f"split SHA mismatch: {sorted(split_sha)}")
    if any(shard["config"].get("sample_count") != 0 for shard in shards):
        raise ValueError("full-wall scale derivation requires sample_count=0")
    if any(shard["config"].get("calibrated_scale", 0.0) != 0.0 for shard in shards):
        raise ValueError("train shards must be raw, with calibrated_scale=0")

    cases = [case for shard in shards for case in shard["cases"]]
    failures = [failure for shard in shards for failure in shard["failures"]]
    canonical_ids = [case["canonical_id"] for case in cases]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("duplicate cases across shards")
    cases.sort(key=lambda case: case["canonical_id"])
    scale = derive_scale(cases)
    reference_v1_scale = derive_scale(cases, method="adaptive_cv_v1")

    payload = {
        "experiment": "pointcloud_multiscale_v2_fullwall_train_scale",
        "objective": "maximize mean case-level raw R2 on train138",
        "method": "multiscale_v2",
        "prediction_scale": scale,
        "reference_adaptive_cv_v1_scale": reference_v1_scale,
        "n_cases": len(cases),
        "n_failures": len(failures),
        "n_wall_points": int(sum(case["n_sampled"] for case in cases)),
        "split_sha256": next(iter(split_sha)),
        "source_shards": [str(Path(path).resolve()) for path in args.inputs],
        "aggregate_raw": aggregate(cases),
        "cases": cases,
        "failures": failures,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in (
        "prediction_scale", "reference_adaptive_cv_v1_scale",
        "n_cases", "n_failures", "n_wall_points"
    )}, indent=2, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
