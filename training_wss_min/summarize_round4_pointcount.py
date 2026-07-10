#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总第四轮 xyz+geom 点数—精度曲线（dev1 / seed1234）。

  python -m training_wss_min.summarize_round4_pointcount

产出 training_wss_min/runs/_summary_round4_pointcount/：
  pointcount_metrics_dev1_s1234.csv
  pointcount_r2_curve_dev1_s1234.png
  pointcount_tail_curve_dev1_s1234.png
  pointcount_density_audit.csv
  pointcount_verdict_dev1_s1234.json
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from . import config as C

RUNS = C.PROJECT_ROOT / "training_wss_min" / "runs"
OUT = RUNS / "_summary_round4_pointcount"
AUDIT = RUNS / "_audit" / "coverage_audit_v2_dev1.csv"

MAP = [
    (1000, "r4_dev1_pc_xyzgeom_fps1000_s1234", "new"),
    (1500, "r4_dev1_pc_xyzgeom_fps1500_s1234", "new"),
    (2000, "r4_dev1_b1_tgtw_fixedq_s1234", "reuse_b1"),
    (3000, "r4_dev1_pc_xyzgeom_fps3000_s1234", "new"),
    (4000, "r4_dev1_b3_fps4000_s1234", "reuse_b3"),
    (6000, "r4_dev1_pc_xyzgeom_fps6000_s1234", "new"),
]


def _best_from_history(hist_path: Path):
    best_ep, best_score, last_ep = None, None, None
    if not hist_path.is_file():
        return best_ep, best_score, last_ep
    for line in hist_path.read_text().splitlines():
        if not line.strip():
            continue
        h = json.loads(line)
        last_ep = h.get("epoch")
        sc = h.get("val_selection_score")
        if sc is not None and (best_score is None or sc > best_score):
            best_score = sc
            best_ep = h.get("epoch")
    return best_ep, best_score, last_ep


def load_rows():
    rows = []
    for n, name, src in MAP:
        d = RUNS / name
        m = json.loads((d / "eval" / "metrics.json").read_text())
        part = "val"
        a, f = m[part]["aggregate"], m[part]["field"]
        cal, hot = m[part]["calibration"], m[part]["hotspot"]
        hi = m[part]["regional_field"]["high_wss"]
        best_ep, best_score, last_ep = _best_from_history(d / "history.jsonl")
        rows.append({
            "wall_n_points": n,
            "run": name,
            "source": src,
            "partition": part,
            "r2_field": float(f["r2"]),
            "r2_casemean": float(a["r2_casemean"]),
            "mae": float(f["mae"]),
            "top10_pred_true_ratio": float(cal["top10_pred_true_ratio"]),
            "top10_iou": float(hot["top10_iou"]),
            "high_wss_r2": float(hi["r2"]),
            "high_wss_mae": float(hot["high_wss_mae"]),
            "p95_pred_true_ratio": float(cal["p95_pred_true_ratio"]),
            "p99_pred_true_ratio": float(cal["p99_pred_true_ratio"]),
            "max_pred_true_ratio": float(cal["max_pred_true_ratio"]),
            "best_epoch": best_ep,
            "best_selection_score": float(best_score) if best_score is not None else float("nan"),
            "last_epoch": last_ep,
        })
    return rows


def load_density():
    audit_rows = list(csv.DictReader(open(AUDIT)))
    dens = []
    for n, _, _ in MAP:
        covers = [float(r[f"fixed{n}_cover"]) for r in audit_rows]
        hits = [float(r[f"fixed{n}_high_hit"]) for r in audit_rows]
        dens.append({
            "wall_n_points": n,
            "mean_cover": float(np.mean(covers)),
            "mean_high_hit": float(np.mean(hits)),
            "worst_high_hit": float(np.min(hits)),
            "best_high_hit": float(np.max(hits)),
            "n_cases": len(audit_rows),
            "split": "split_AG_wss_min_v2_dev1",
            "stats": "wss_stats_v2_dev1",
        })
    return dens


def verdict(rows):
    peak = max(rows, key=lambda r: r["best_selection_score"])
    max_rf = max(r["r2_field"] for r in rows)
    max_rc = max(r["r2_casemean"] for r in rows)
    peak_top10 = peak["top10_pred_true_ratio"]
    peak_iou = peak["top10_iou"]

    def near_platform(r):
        return (
            abs(r["r2_field"] - max_rf) <= 0.01
            and abs(r["r2_casemean"] - max_rc) <= 0.01
            and (peak_top10 - r["top10_pred_true_ratio"]) <= 0.03
            and (peak_iou - r["top10_iou"]) <= 0.03
        )

    platform = [r for r in rows if near_platform(r)]
    simp = min(platform, key=lambda r: r["wall_n_points"]) if platform else peak
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "B1 xyz+geom fixed FPS tgtw-fixedq v2_dev1 val-only seed1234",
        "peak_by_selection_score": {
            "wall_n_points": peak["wall_n_points"],
            "run": peak["run"],
            "best_selection_score": peak["best_selection_score"],
            "r2_field": peak["r2_field"],
            "r2_casemean": peak["r2_casemean"],
            "top10_pred_true_ratio": peak["top10_pred_true_ratio"],
            "top10_iou": peak["top10_iou"],
        },
        "max_r2_field": max_rf,
        "max_r2_casemean": max_rc,
        "platform_points": [r["wall_n_points"] for r in platform],
        "simplified_kink": simp["wall_n_points"],
        "note": "Single-seed only; not a final best-n claim. Next: multi-seed on peak vs 2000 anchor.",
    }


def plot_curves(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["wall_n_points"] for r in rows]
    peak_n = max(rows, key=lambda r: r["best_selection_score"])["wall_n_points"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, [r["r2_field"] for r in rows], "o-", label="val R²_field", lw=2)
    ax.plot(ns, [r["r2_casemean"] for r in rows], "s--", label="val R²_casemean", lw=2)
    ax.axvline(peak_n, color="C2", ls=":", alpha=0.8, label=f"peak score @ {peak_n}")
    ax.set_xlabel("train wall_n (eval = full cloud, val-only)")
    ax.set_ylabel("R²")
    ax.set_title("r4 xyz+geom point-count curve (dev1, seed1234, B1 recipe)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pointcount_r2_curve_dev1_s1234.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, [r["top10_pred_true_ratio"] for r in rows], "o-", label="top10 pred/true", lw=2)
    ax.plot(ns, [r["top10_iou"] for r in rows], "s--", label="top10 IoU", lw=2)
    ax.set_xlabel("train wall_n")
    ax.set_ylabel("tail metric")
    ax.set_title("r4 xyz+geom point-count tail metrics (dev1, seed1234)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "pointcount_tail_curve_dev1_s1234.png", dpi=120)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    dens = load_density()
    meta = verdict(rows)

    with open(OUT / "pointcount_metrics_dev1_s1234.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "pointcount_density_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dens[0].keys()))
        w.writeheader()
        w.writerows(dens)
    (OUT / "pointcount_verdict_dev1_s1234.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    plot_curves(rows)

    print(f"\n共 {len(rows)} 个点数\n")
    print(f"{'n':>6}  {'R2_f':>7}  {'R2_c':>7}  {'top10':>7}  {'IoU':>7}  {'score':>7}  {'best':>5}")
    for r in rows:
        print(
            f"{r['wall_n_points']:6d}  {r['r2_field']:7.3f}  {r['r2_casemean']:7.3f}  "
            f"{r['top10_pred_true_ratio']:7.3f}  {r['top10_iou']:7.3f}  "
            f"{r['best_selection_score']:7.3f}  {r['best_epoch']:5d}"
        )
    print("\nverdict:", json.dumps(meta["peak_by_selection_score"], ensure_ascii=False))
    print(f"platform={meta['platform_points']} simplified_kink={meta['simplified_kink']}")
    print(f"\n写入 {OUT}/")


if __name__ == "__main__":
    main()
