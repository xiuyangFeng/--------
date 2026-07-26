#!/usr/bin/env python3
"""Create a checkpoint sensitivity report without changing checkpoint selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.run_dir / "eval"
    best = json.loads((root / "ckpt_best/metrics.json").read_text())
    last = json.loads((root / "ckpt_last/metrics.json").read_text())
    payload = {"selection_unchanged": "ckpt_best selected by train_loss",
               "partitions": {}}
    rows = []
    for part in sorted(set(best) & set(last)):
        bp, lp = best[part], last[part]
        keys = {
            "pooled_physical_r2": (bp["field"]["r2"], lp["field"]["r2"]),
            "casebalanced_vertex_r2": (bp["field_casebalanced"]["r2"], lp["field_casebalanced"]["r2"]),
        }
        area_key = "primary_physical_or_normalized_casebalanced_area"
        if area_key in bp and area_key in lp:
            keys["casebalanced_area_r2"] = (bp[area_key]["r2"], lp[area_key]["r2"])
        payload["partitions"][part] = {
            key: {"best": a, "last": b, "last_minus_best": b - a}
            for key, (a, b) in keys.items()
        }
        bpc, lpc = bp["per_case"], lp["per_case"]
        for case in sorted(set(bpc) & set(lpc)):
            section = "area_overall" if (
                "area_overall" in bpc[case] and "area_overall" in lpc[case]
            ) else "overall"
            br = bpc[case][section]["r2"]
            lr = lpc[case][section]["r2"]
            rows.append({"partition": part, "case": case, "metric_section": section,
                         "best_r2": br, "last_r2": lr, "last_minus_best": lr - br,
                         "direction": "improved" if lr > br else "degraded" if lr < br else "same"})
    (root / "best_last_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    with (root / "best_last_per_case.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["partition", "case"])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
