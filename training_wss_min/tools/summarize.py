#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 runs/*/eval/metrics.json：跨配置对比表 + 点数-精度曲线。

  python -m training_wss_min.summarize
产出 training_wss_min/runs/_summary/：summary.csv、pointcount_curve.png、regional_bar.png
并在控制台打印对比表。
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parent.parent / "runs"
OUT = RUNS / "_summary"


def load_all():
    rows = []
    for d in sorted(RUNS.glob("*/")):
        mp = d / "eval" / "metrics.json"
        if not mp.is_file():
            continue
        m = json.loads(mp.read_text())
        cfg = json.loads((d / "config.json").read_text())
        row = {
            "name": d.name,
            "wall_n": cfg.get("data", {}).get("wall_n_points", 0),
            "sampling": cfg.get("data", {}).get("sampling", cfg.get("method", "")),
            "features": "+".join(cfg.get("data", {}).get("input_features", [])),
            "nfeat": len(cfg.get("data", {}).get("input_features", [])),
        }
        for part in ("val", "test"):
            if part not in m:
                continue
            a, f, rf = m[part]["aggregate"], m[part]["field"], m[part].get("regional_field", {})
            row[f"{part}_r2_casemean"] = a["r2_casemean"]
            row[f"{part}_r2_field"] = f["r2"]
            row[f"{part}_nrmse_field"] = f["nrmse_range"]
            row[f"{part}_mae"] = f["mae"]
            cal = m[part].get("calibration", {})
            for ck in ("top10_pred_true_ratio", "p95_pred_true_ratio",
                       "p99_pred_true_ratio", "max_pred_true_ratio",
                       "calibration_slope"):
                row[f"{part}_{ck}"] = cal.get(ck, float("nan"))
            for reg in ("bifurcation", "stenosis", "high_wss"):
                row[f"{part}_{reg}_r2"] = rf.get(reg, {}).get("r2", float("nan"))
        rows.append(row)
    return rows


def _fmt(v):
    return f"{v:7.3f}" if isinstance(v, float) and np.isfinite(v) else f"{str(v):>7s}"


def print_table(rows):
    cols = ["name", "wall_n", "sampling", "features",
            "test_r2_field", "test_r2_casemean", "test_nrmse_field", "test_mae",
            "test_bifurcation_r2", "test_stenosis_r2", "test_high_wss_r2",
            "test_top10_pred_true_ratio", "test_p99_pred_true_ratio"]
    short = {"test_r2_field": "te_R2f", "test_r2_casemean": "te_R2cm",
             "test_nrmse_field": "te_NRMSE", "test_mae": "te_MAE",
             "test_bifurcation_r2": "te_bif", "test_stenosis_r2": "te_sten",
             "test_high_wss_r2": "te_hiW",
             "test_top10_pred_true_ratio": "te_top10",
             "test_p99_pred_true_ratio": "te_p99"}
    hdr = [short.get(c, c) for c in cols]
    print("  ".join(f"{h:>28s}" if h == "name" else f"{h:>10s}" for h in hdr))
    for r in sorted(rows, key=lambda x: -(x.get("test_r2_field") or -9)):
        cells = []
        for c in cols:
            v = r.get(c)
            if c == "name":
                cells.append(f"{v:>28s}")
            elif c in ("wall_n", "sampling", "features"):
                cells.append(f"{str(v):>10s}")
            else:
                cells.append(f"{v:10.3f}" if isinstance(v, float) and np.isfinite(v) else f"{'nan':>10s}")
        print("  ".join(cells))


def write_csv(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["sampling"], x["features"], x["wall_n"])):
            w.writerow(r)


def plot_pointcount(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pc = sorted([r for r in rows if r["name"].startswith("pc_xyz_fps_")],
                key=lambda x: x["wall_n"])
    if not pc:
        return
    n = [r["wall_n"] for r in pc]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, lab, mk in [("test_r2_field", "test R2 (field)", "o-"),
                         ("val_r2_field", "val R2 (field)", "s--"),
                         ("test_r2_casemean", "test R2 (casemean)", "^:")]:
        ax.plot(n, [r.get(key, float("nan")) for r in pc], mk, label=lab)
    ax.set_xlabel("train wall_n (eval always on full cloud)")
    ax.set_ylabel("R2")
    ax.set_title("point-count vs accuracy (xyz, fps, peak, scalar WSS)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.grid(alpha=0.3); ax.legend()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT / "pointcount_curve.png", dpi=110); plt.close(fig)


def plot_regional(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    regs = ["test_r2_field", "test_bifurcation_r2", "test_stenosis_r2", "test_high_wss_r2"]
    labs = ["overall", "bifurcation", "stenosis", "high_wss"]
    rows = sorted(rows, key=lambda x: x["name"])
    x = np.arange(len(rows)); w = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (k, lb) in enumerate(zip(regs, labs)):
        ax.bar(x + i * w, [r.get(k, float("nan")) for r in rows], w, label=lb)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([r["name"] for r in rows], rotation=35, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("test R2"); ax.set_title("regional R2 per config (test, full cloud)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT / "regional_bar.png", dpi=110); plt.close(fig)


def main():
    rows = load_all()
    print(f"\n共 {len(rows)} 个实验\n")
    print_table(rows)
    write_csv(rows)
    plot_pointcount(rows)
    plot_regional(rows)
    print(f"\n写入 {OUT}/  (summary.csv, pointcount_curve.png, regional_bar.png)")


if __name__ == "__main__":
    main()
