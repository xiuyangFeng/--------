#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1 覆盖审计：fixed/multi-start FPS 对 high-WSS 的覆盖（CPU，train-only）。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D


def top10_mask(y: np.ndarray) -> np.ndarray:
    thr = np.percentile(y, 90.0)
    return y >= thr


def audit_case(case, k_list, pool_size, epochs, run_seed):
    n = len(case["pos"])
    high = top10_mask(case["y_raw"])
    n_high = int(high.sum())
    row = {"case": f"{case['cohort']}/{case['case']}", "n": n, "n_high": n_high}
    for k in k_list:
        cfg_f = C.DataConfig(wall_n_points=k, sampling="fps")
        # 清缓存，避免跨 k 污染
        case.pop("_fps_subset", None)
        case.pop("_fps_subset_k", None)
        case.pop("_fps_pool", None)
        idx = D.sample_indices(case, cfg_f, case["geom_seed"], epoch=0, case_index=0,
                               run_seed=run_seed)
        row[f"fixed{k}_cover"] = float(len(idx) / n)
        row[f"fixed{k}_high_hit"] = float(high[idx].sum() / max(1, n_high))
        # multi-start：只预计算 pool，再按 epoch 取子集做 union（避免重复 FPS）
        cfg_m = C.DataConfig(wall_n_points=k, sampling="fps_multistart",
                             fps_pool_size=pool_size, resample_each_epoch=True)
        case.pop("_fps_pool", None)
        case.pop("_fps_pool_k", None)
        case.pop("_fps_pool_size", None)
        # 触发一次 pool 构建
        _ = D.sample_indices(case, cfg_m, case["geom_seed"], epoch=0, case_index=0,
                             run_seed=run_seed)
        union = np.zeros(n, dtype=bool)
        hit_counts = np.zeros(n, dtype=np.int32)
        for ep in range(epochs):
            idx_m = D.sample_indices(case, cfg_m, case["geom_seed"], epoch=ep,
                                     case_index=0, run_seed=run_seed)
            union[idx_m] = True
            hit_counts[idx_m] += 1
        row[f"ms{k}_union_cover"] = float(union.mean())
        row[f"ms{k}_high_union_hit"] = float(high[union].sum() / max(1, n_high))
        row[f"ms{k}_high_mean_hits"] = float(hit_counts[high].mean()) if n_high else float("nan")
        row[f"ms{k}_high_min_hits"] = float(hit_counts[high].min()) if n_high else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(C.DEFAULT_SPLIT))
    ap.add_argument("--stats", default=str(C.GLOBAL_STATS))
    ap.add_argument("--k", nargs="+", type=int, default=[2000, 4000])
    ap.add_argument("--pool-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="training_wss_min/runs/_audit/coverage_audit.csv")
    args = ap.parse_args()

    wss = D.load_wss_stats(args.stats)
    cases = D.load_partition(args.split, "train", wss, strict=True)
    for i, c in enumerate(cases):
        c["geom_seed"] = args.seed + 7919 * i
    rows = []
    for i, c in enumerate(cases):
        c["geom_seed"] = args.seed + 7919 * i
        print(f"[{i+1}/{len(cases)}] {c['cohort']}/{c['case']} n={len(c['pos'])}", flush=True)
        rows.append(audit_case(c, args.k, args.pool_size, args.epochs, args.seed))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split, "stats": args.stats,
        "n_cases": len(rows), "k": args.k, "pool_size": args.pool_size,
        "epochs": args.epochs,
        "mean_fixed2000_high_hit": float(np.mean([r.get("fixed2000_high_hit", np.nan) for r in rows])),
        "mean_ms2000_high_union_hit": float(np.mean([r.get("ms2000_high_union_hit", np.nan) for r in rows])),
        "worst_fixed2000_high_hit": float(np.min([r.get("fixed2000_high_hit", np.nan) for r in rows])),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
