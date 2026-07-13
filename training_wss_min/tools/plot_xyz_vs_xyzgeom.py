"""Plot the baseline XYZ versus XYZ+geometry accuracy comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "runs" / "_summary" / "summary.csv"
OUT_DIR = ROOT / "runs" / "_summary"


def load_rows() -> dict[str, dict[str, str]]:
    with SUMMARY.open(newline="") as f:
        rows = {row["name"]: row for row in csv.DictReader(f)}
    names = [
        "pc_xyz_fps_w1000_peak",
        "pc_xyz_fps_w1500_peak",
        "pc_xyz_fps_w2000_peak",
        "pc_xyz_fps_w3000_peak",
        "pc_xyz_fps_w6000_peak",
        "feat_geomonly_fps_w2000_peak",
        "feat_xyzgeom_fps_w2000_peak",
    ]
    return {name: rows[name] for name in names}


def main() -> None:
    rows = load_rows()
    # Register the system CJK font explicitly; the bundled matplotlib font cache
    # may otherwise fall back to DejaVu Sans and render Chinese as boxes.
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 11,
        }
    )

    blue = "#2563EB"
    orange = "#EA580C"
    gray = "#64748B"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2), dpi=180)
    fig.patch.set_facecolor("white")

    # Panel A: pure XYZ point-count curve.
    n_points = np.array([1000, 1500, 2000, 3000, 6000])
    names = [f"pc_xyz_fps_w{n}_peak" for n in n_points]
    field = np.array([float(rows[n]["test_r2_field"]) for n in names])
    case = np.array([float(rows[n]["test_r2_casemean"]) for n in names])
    ax1.plot(n_points, field, marker="o", lw=2.5, ms=7, color=blue, label="R²_field")
    ax1.plot(n_points, case, marker="s", lw=2.5, ms=7, color=orange, label="R²_casemean")
    for x, y in zip(n_points, field):
        ax1.annotate(f"{y:.3f}", (x, y), xytext=(0, 10), textcoords="offset points", ha="center", color=blue, fontsize=9)
    ax1.set_title("纯 XYZ：点数影响", loc="left", fontweight="bold")
    ax1.set_xlabel("输入点数")
    ax1.set_ylabel("测试集 R²（越高越好）")
    ax1.set_xticks(n_points)
    ax1.set_ylim(-0.12, 0.24)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(frameon=False, loc="upper left")

    # Panel B: fixed 2000 points, feature ablation.
    labels = ["纯几何\n(4特征)", "纯 XYZ\n(3特征)", "XYZ + 几何\n(7特征)"]
    names = ["feat_geomonly_fps_w2000_peak", "pc_xyz_fps_w2000_peak", "feat_xyzgeom_fps_w2000_peak"]
    field = np.array([float(rows[n]["test_r2_field"]) for n in names])
    case = np.array([float(rows[n]["test_r2_casemean"]) for n in names])
    x = np.arange(len(labels))
    width = 0.34
    bars1 = ax2.bar(x - width / 2, field, width, color=blue, label="R²_field")
    bars2 = ax2.bar(x + width / 2, case, width, color=orange, label="R²_casemean")
    for bars in (bars1, bars2):
        for bar in bars:
            ax2.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 4 if bar.get_height() >= 0 else -14), textcoords="offset points",
                         ha="center", va="bottom" if bar.get_height() >= 0 else "top", fontsize=9)
    ax2.set_title("固定 2000 点：输入特征对比", loc="left", fontweight="bold")
    ax2.set_xticks(x, labels)
    ax2.set_ylabel("测试集 R²（越高越好）")
    ax2.set_ylim(-0.10, 0.26)
    ax2.axhline(0, color="#334155", lw=0.8)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(frameon=False, loc="upper left")

    fig.suptitle("WSS 预测精度对比：XYZ 与 XYZ + 几何特征", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015, "同一 baseline 协议；FPS 采样；peak-WSS；测试集 n=16。R² 越高表示预测精度越好。",
             ha="center", color=gray, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), w_pad=3.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "xyz_vs_xyzgeom_accuracy.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "xyz_vs_xyzgeom_accuracy.pdf", bbox_inches="tight")

    with (OUT_DIR / "xyz_vs_xyzgeom_accuracy.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "input", "points", "test_r2_field", "test_r2_casemean"])
        for name in ["feat_geomonly_fps_w2000_peak", "pc_xyz_fps_w2000_peak", "feat_xyzgeom_fps_w2000_peak"]:
            row = rows[name]
            writer.writerow([name, row["features"], row["wall_n"], row["test_r2_field"], row["test_r2_casemean"]])
        for name, n in zip([f"pc_xyz_fps_w{n}_peak" for n in n_points], n_points):
            row = rows[name]
            writer.writerow([name, row["features"], n, row["test_r2_field"], row["test_r2_casemean"]])


if __name__ == "__main__":
    main()
