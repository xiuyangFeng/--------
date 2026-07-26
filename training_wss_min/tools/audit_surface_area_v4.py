#!/usr/bin/env python3
"""Write the 133-case STL-to-wall area mapping gate, including failures."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from training_wss_min import config as C, dataset as D, surface as S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    cfg = C.ExpConfig.from_json(args.config)
    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    rows = []
    for partition in ("train", "test"):
        cases = D.load_partition(
            cfg.data.split_path, partition, stats, strict=True,
            target=cfg.data.target,
            target_normalization=cfg.data.target_normalization,
            data_root=cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version,
        )
        for case in cases:
            _, report = S.area_weights_for_case(case, strict=False)
            report["partition"] = partition
            rows.append(report)
    failures = [row for row in rows if row["status"] != "passed"]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "hard_gates": {"bidirectional_p95_over_bbox_max": 0.02,
                       "bidirectional_max_over_bbox_max": 0.10,
                       "positive_wall_weights_min": 5000},
        "n_cases": len(rows),
        "n_passed": len(rows) - len(failures),
        "n_failed": len(failures),
        "failed_cases": [row["unit_id"] for row in failures],
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "n_cases", "n_passed",
                                              "n_failed", "failed_cases")},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
