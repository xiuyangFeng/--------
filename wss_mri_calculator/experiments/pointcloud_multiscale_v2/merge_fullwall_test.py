"""Merge full-wall calibrated test shards into one reproducible result JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from compare_multiscale_v2 import aggregate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    shards = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    split_sha = {shard["split"]["sha256"] for shard in shards}
    scales = {float(shard["config"]["calibrated_scale"]) for shard in shards}
    if len(split_sha) != 1:
        raise ValueError(f"split SHA mismatch: {sorted(split_sha)}")
    if len(scales) != 1 or next(iter(scales)) <= 0:
        raise ValueError(f"calibrated scale mismatch: {sorted(scales)}")
    if any(shard["config"].get("sample_count") != 0 for shard in shards):
        raise ValueError("full-wall merge requires sample_count=0")

    cases = [case for shard in shards for case in shard["cases"]]
    failures = [failure for shard in shards for failure in shard["failures"]]
    canonical_ids = [case["canonical_id"] for case in cases]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("duplicate cases across shards")
    cases.sort(key=lambda case: case["canonical_id"])

    payload = {
        "experiment": "pointcloud_multiscale_v2_fullwall_test_calibrated",
        "prediction_scale": next(iter(scales)),
        "n_cases": len(cases),
        "n_failures": len(failures),
        "n_wall_points": int(sum(case["n_sampled"] for case in cases)),
        "split_sha256": next(iter(split_sha)),
        "source_shards": [str(Path(path).resolve()) for path in args.inputs],
        "aggregate": aggregate(cases),
        "aggregate_by_cohort": {
            cohort: aggregate([case for case in cases if case["cohort"] == cohort])
            for cohort in sorted({case["cohort"] for case in cases})
        },
        "cases": cases,
        "failures": failures,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "prediction_scale": payload["prediction_scale"],
        "n_cases": payload["n_cases"],
        "n_failures": payload["n_failures"],
        "n_wall_points": payload["n_wall_points"],
        "aggregate": payload["aggregate"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
