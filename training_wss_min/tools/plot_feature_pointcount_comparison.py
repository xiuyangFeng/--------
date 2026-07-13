"""Plot available point-count comparisons for XYZ, geometry, and XYZ+geometry."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "runs" / "_summary" / "summary.csv"
PC_SUMMARY = ROOT / "runs" / "_summary_round4_pointcount" / "pointcount_metrics_dev1_s1234.csv"
OUT_DIR = ROOT / "runs" / "_summary"


def main() -> None:
    with SUMMARY.open(newline="") as f:
        base = {r["name"]: r for r in csv.DictReader(f)}
    with PC_SUMMARY.open(newline="") as f:
        all_pc = list(csv.DictReader(f))
    pc = {int(r["wall_n_points"]): r for r in all_pc if "xyzgeom" in r["run"]}
    # The 2000-point entry reuses the fixed-q XYZ+geometry anchor.
    pc[2000] = next(r for r in all_pc if r["run"] == "r4_dev1_b1_tgtw_fixedq_s1234")

    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
    })

    points = np.array([1000, 1500, 2000, 3000, 6000])
    xyz_names = [f"pc_xyz_fps_w{n}_peak" for n in points]
    xyz_field = np.array([float(base[n]["test_r2_field"]) for n in xyz_names])
    xyz_case = np.array([float(base[n]["test_r2_casemean"]) for n in xyz_names])
    geom_field = float(base["feat_geomonly_fps_w2000_peak"]["test_r2_field"])
    geom_case = float(base["feat_geomonly_fps_w2000_peak"]["test_r2_casemean"])
    xyzgeom_field = np.array([float(pc[n]["r2_field"]) for n in points])
    xyzgeom_case = np.array([float(pc[n]["r2_casemean"]) for n in points])

    colors = {"xyz": "#2563EB", "geom": "#16A34A", "xyzgeom": "#EA580C"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.1), dpi=180, sharex=True)
    for ax, series, ylabel, title in [
        (axes[0], [(xyz_field, "纯 XYZ", "xyz"), (xyzgeom_field, "XYZ + 几何", "xyzgeom"),
                   ([geom_field], "纯几何（仅 2000 点）", "geom")], "R²_field", "字段级 R²"),
        (axes[1], [(xyz_case, "纯 XYZ", "xyz"), (xyzgeom_case, "XYZ + 几何", "xyzgeom"),
                   ([geom_case], "纯几何（仅 2000 点）", "geom")], "R²_casemean", "病例平均 R²"),
    ]:
        ax.plot(points, series[0][0], marker="o", lw=2.5, ms=7, color=colors[series[0][2]], label=series[0][1])
        ax.plot(points, series[1][0], marker="s", lw=2.5, ms=7, color=colors[series[1][2]], label=series[1][1])
        ax.scatter([2000], series[2][0], marker="D", s=70, color=colors[series[2][2]], label=series[2][1], zorder=5)
        for x, y in zip(points, series[0][0]):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", color=colors["xyz"], fontsize=8.5)
        for x, y in zip(points, series[1][0]):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, -14), textcoords="offset points", ha="center", color=colors["xyzgeom"], fontsize=8.5)
        ax.annotate(f"{series[2][0][0]:.3f}", (2000, series[2][0][0]), xytext=(28, 10), textcoords="offset points", ha="left", color=colors["geom"], fontsize=8.5)
        ax.axhline(0, color="#334155", lw=0.8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("输入点数")
        ax.set_ylabel(ylabel)
        ax.set_xticks(points)
        ax.set_ylim(-0.12, 0.42)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, loc="upper left", fontsize=9)

    fig.suptitle("不同输入特征的点数—预测精度对比", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015, "纯 XYZ：baseline 测试集 n=16；XYZ+几何：dev1 验证集 n=8；纯几何目前仅有 2000 点实验。R² 越高越好。",
             ha="center", color="#64748B", fontsize=9.2)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94), w_pad=3.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "feature_pointcount_comparison.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "feature_pointcount_comparison.pdf", bbox_inches="tight")

    with (OUT_DIR / "feature_pointcount_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input", "points", "r2_field", "r2_casemean", "split_metric"])
        for n, fval, cval in zip(points, xyz_field, xyz_case):
            w.writerow(["xyz", n, fval, cval, "baseline_test"])
        for n, fval, cval in zip(points, xyzgeom_field, xyzgeom_case):
            w.writerow(["xyz+geom", n, fval, cval, "dev1_val_s1234"])
        w.writerow(["geom", 2000, geom_field, geom_case, "baseline_test"])


if __name__ == "__main__":
    main()
