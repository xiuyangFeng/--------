#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WSS-min 数据规整性审计：只统计 split 中 included（train/val/test）bundle。

用法：
  python docs/02-推进与变更/assets_第四轮/audit_bundles.py \
      [--root <repo>] [--split <json>] [--stats <json>] [--out <csv>]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2]


def audit(root: Path, split_path: Path, stats_path: Path, out_csv: Path) -> pd.DataFrame:
    data = root / "data_wss_min"
    sp = json.loads(split_path.read_text())
    stats = json.loads(stats_path.read_text())
    mu, sd, eps = stats["log"]["mean"], stats["log"]["std"], stats["eps"]

    rows = []
    problems = []
    for part in ("train", "val", "test"):
        for label in sp.get(f"{part}_cases", []):
            subset, case = label.split("/", 1)
            p = data / "AG" / subset / case / "bundle.npz"
            if not p.is_file():
                problems.append(f"MISSING bundle: {part} {label}")
                continue
            with np.load(p, allow_pickle=True) as d:
                steps = d["steps"].tolist()
                peak = int(d["peak_step"])
                si = steps.index(peak)
                pos = d["wall_coords_norm"].astype(np.float64)
                wss = d["wall_wss"][si].astype(np.float64)
                lr = d["wall_local_radius"].astype(np.float64)
                curv = d["wall_curvature"].astype(np.float64)
                absc = d["wall_abscissa_norm"].astype(np.float64)
                cs = float(d["coord_scale"])
                crop = bool(d["wall_crop_applied"])
                uem = bool(d["unit_extent_mismatch"])
                n = len(pos)

            y_norm = (np.log(np.clip(wss, 0, None) + eps) - mu) / sd
            clog = np.sign(curv) * np.log1p(np.abs(curv))
            r = dict(
                part=part, label=label, n_wall=n, n_steps=len(steps),
                peak_idx=si, coord_scale=cs, crop=crop, unit_mismatch=uem,
                pos_maxabs=float(np.abs(pos).max()),
                pos_com=float(np.linalg.norm(pos.mean(axis=0))),
                pos_finite=bool(np.isfinite(pos).all()),
                wss_min=float(wss.min()), wss_max=float(wss.max()),
                wss_mean=float(wss.mean()),
                wss_p99=float(np.percentile(wss, 99)),
                wss_zero_frac=float((wss <= 0).mean()),
                wss_finite=bool(np.isfinite(wss).all()),
                y_norm_min=float(y_norm.min()), y_norm_max=float(y_norm.max()),
                y_norm_mean=float(y_norm.mean()), y_norm_std=float(y_norm.std()),
                lr_min=float(lr.min()), lr_max=float(lr.max()),
                lr_nonpos_frac=float((lr <= 0).mean()),
                curv_absmax=float(np.abs(curv).max()),
                curv_log_absmax=float(np.abs(clog).max()),
                curv_finite=bool(np.isfinite(curv).all()),
                absc_min=float(absc.min()), absc_max=float(absc.max()),
                lr_unique=int(len(np.unique(lr))),
            )
            rows.append(r)
            if not r["pos_finite"] or not r["wss_finite"] or not r["curv_finite"]:
                problems.append(f"NONFINITE: {part} {label}")
            if abs(r["pos_maxabs"] - 1.0) > 1e-3:
                problems.append(f"POS_MAXABS!=1: {part} {label} {r['pos_maxabs']:.4f}")
            if r["wss_zero_frac"] > 0:
                problems.append(
                    f"WSS_ZERO: {part} {label} frac={r['wss_zero_frac']:.4f} "
                    f"y_norm_min={r['y_norm_min']:.2f}"
                )
            if r["lr_nonpos_frac"] > 0:
                problems.append(f"LR_NONPOS: {part} {label} frac={r['lr_nonpos_frac']:.4f}")

    if not rows:
        raise RuntimeError("no included bundles audited")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)

    print("=== counts ===")
    print(df.groupby("part").size())
    print("\n=== per-part summary (median [min,max]) ===")
    for col in ("n_wall", "coord_scale", "wss_mean", "wss_p99", "wss_max",
                "y_norm_mean", "y_norm_std", "y_norm_min", "y_norm_max",
                "pos_com", "curv_log_absmax", "lr_unique"):
        g = df.groupby("part")[col]
        line = f"{col:16s} " + "  ".join(
            f"{p}: {g.median()[p]:.3f} [{g.min()[p]:.3f},{g.max()[p]:.3f}]"
            for p in ("train", "val", "test") if p in g.groups
        )
        print(line)

    print("\n=== PROBLEMS ===")
    if problems:
        for p in problems:
            print(p)
    else:
        print("none")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "split_path": str(split_path),
        "stats_path": str(stats_path),
        "n_rows": len(df),
        "n_problems": len(problems),
        "partitions": {p: int((df.part == p).sum()) for p in ("train", "val", "test")},
        "excluded_pending_read": False,
    }
    out_csv.with_name(out_csv.stem + "_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    print(f"\nsaved {out_csv}  (n={len(df)}, problems={len(problems)})")
    return df


def main():
    ap = argparse.ArgumentParser(description="Audit WSS-min included bundles")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument(
        "--split", type=Path,
        default=None,
        help="split JSON (default: training/splits/split_AG_wss_min_v1.json)",
    )
    ap.add_argument(
        "--stats", type=Path, default=None,
        help="WSS stats JSON (default: data_wss_min/wss_global_stats.json)",
    )
    ap.add_argument("--out", type=Path, default=HERE / "audit_bundles.csv")
    args = ap.parse_args()
    root = args.root.resolve()
    split = (args.split or (root / "training/splits/split_AG_wss_min_v1.json")).resolve()
    stats = (args.stats or (root / "data_wss_min/wss_global_stats.json")).resolve()
    audit(root, split, stats, args.out.resolve())


if __name__ == "__main__":
    main()
