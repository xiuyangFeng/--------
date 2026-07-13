#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build train-only, peak-only gauge-pressure normalization stats.

Wall pressure carries a large per-case DC offset (~1.5e4 Pa) with only ~1e2 Pa
of spatial variation.  We predict *gauge* pressure = per-case demeaned wall
pressure, so the target isolates the geometry-predictable spatial pattern and
pooled R2_field is not dominated by cross-case DC differences.

Stats are computed on the per-case-demeaned train values at the frozen peak
step and written with ``method="linear"`` so dataset.normalize_wss /
denormalize_wss handle them without WSS-specific log transforms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(
        C.PROJECT_ROOT / "training" / "splits" / "split_AG_wss_min_v2_dev1.json"))
    ap.add_argument("--partition", default="train")
    ap.add_argument("--out", default=str(
        C.PROJECT_ROOT / "data_wss_min" / "fold_stats"
        / "pressure_gauge_stats_v2_dev1.json"))
    args = ap.parse_args()

    labels = D.load_split_cases(args.split, args.partition)
    gauges = []
    n_cases = 0
    for cohort_rel, case_name in labels:
        bundle = C.DATA_ROOT / cohort_rel / case_name / "bundle.npz"
        if not bundle.is_file():
            raise FileNotFoundError(bundle)
        with np.load(bundle, allow_pickle=True) as d:
            steps = d["steps"].tolist()
            si = steps.index(int(d["peak_step"]))
            p_raw = d["wall_pressure"][si].astype(np.float64)
        gauges.append(p_raw - p_raw.mean())  # per-case demean
        n_cases += 1

    allg = np.concatenate(gauges)
    stats = {
        "method": "linear",
        "target": "pressure",
        "normalization": "per_case_demean_then_global_z",
        "eps": 0.0,
        "n_points": int(allg.size),
        "n_cases": int(n_cases),
        "timesteps_scope": "peak",
        "linear": {"mean": float(allg.mean()), "std": float(allg.std())},
        "raw_gauge_percentiles": {
            "p01": float(np.percentile(allg, 1)),
            "p50": float(np.percentile(allg, 50)),
            "p99": float(np.percentile(allg, 99)),
        },
        "raw_gauge_min": float(allg.min()),
        "raw_gauge_max": float(allg.max()),
        "partitions": [args.partition],
        "split_path": str(args.split),
        "note": "gauge pressure = per-case demeaned wall_pressure at peak step; "
                "denormalize returns gauge (relative) Pa, can be negative.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"gauge-pressure stats -> {out}")
    print(f"  n_cases={n_cases} n_points={allg.size} "
          f"mean={stats['linear']['mean']:.4g} std={stats['linear']['std']:.4g} "
          f"[p01,p99]=[{stats['raw_gauge_percentiles']['p01']:.4g},"
          f"{stats['raw_gauge_percentiles']['p99']:.4g}]")


if __name__ == "__main__":
    main()
