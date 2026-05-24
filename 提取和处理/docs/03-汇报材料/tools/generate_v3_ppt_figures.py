#!/usr/bin/env python3
"""Generate V3 PPT supplement figures from outputs/ and docs.

Auto-generated figures (run via main):
  - v3p_wss_comparison.png
  - v3p_wss_components.png
  - v3p_val_wss_components_history.png
  - v3d_per_domain_metrics.png

User-generated externally (see PPT补充计划.md §7):
  - v3_path_map.png
  - v3_todo_priority.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "docs" / "03-汇报材料" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "Main-PW": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wall13000_near2000_split_AG_v1_seed1_20260508_001936",
    "AsymW-a": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260522_124946",
    "WssDO-a": ROOT
    / "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wssdo_a_wall13000_near2000_split_AG_v1_seed1_20260522_131813",
    "WSS-a-PW": ROOT
    / "outputs/field/field_v3_pointnext_localpool_wss01a_geom_pw_lambda005_wall13000_near2000_split_AG_v1_seed1_20260507_001902",
}

V3D_PROBE_P = ROOT / (
    "outputs/field/field_v3d_pointnext_localpool_probe_p01_geom_wall13000_near2000_"
    "split_data_new_v3_v3_seed1_20260521_103843/eval_by_domain_test/metrics_by_domain.json"
)
V3D_PROBE_WSS = ROOT / (
    "outputs/field/field_v3d_pointnext_localpool_probe_wss01_geom_wall13000_near2000_"
    "split_data_new_v3_v3_seed1_20260521_101738/eval_by_domain_test/metrics_by_domain.json"
)

# --- Style constants (16:9 PPT-friendly) ---
FIG_W, FIG_H = 12.8, 7.2
DPI = 250
PALETTE = {
    "bg": "#fafafa",
    "text": "#1f2937",
    "muted": "#6b7280",
    "grid": "#e5e7eb",
    "baseline": "#94a3b8",
    "accent_blue": "#2563eb",
    "accent_orange": "#ea580c",
    "accent_green": "#059669",
    "accent_amber": "#d97706",
    "highlight": "#1d4ed8",
    "ref_line": "#dc2626",
    "wss_x": "#dc2626",
    "wss_y": "#7c3aed",
    "wss_z": "#059669",
}


def setup_style() -> str:
    from matplotlib import font_manager

    candidates = (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "SimHei",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = "DejaVu Sans"
    for fam in candidates:
        if fam in available:
            font = fam
            break
    plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelcolor": PALETTE["text"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.facecolor": "white",
            "figure.facecolor": PALETTE["bg"],
            "text.color": PALETTE["text"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "legend.framealpha": 0.95,
            "legend.edgecolor": PALETTE["grid"],
            "axes.unicode_minus": False,
        }
    )
    return font


def style_axes(ax, *, y_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_grid:
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)


def save_fig(fig, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {path.name}")


def best_wss_metrics(run_dir: Path) -> dict:
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        s = json.load(f)
    return s.get("test_metrics_best_wss") or s["test_metrics"]


def annotate_bars(ax, bars, vals, *, fmt: str = "{:.3f}", offset: float = 0.003, fontsize: int = 10):
    ymax = max(vals) if vals else 1.0
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + offset * ymax,
            fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=PALETTE["text"],
            fontweight="medium",
        )


def fig_wss_comparison():
    labels = ["Main-PW", "WSS-a-PW", "AsymW-a", "WssDO-a"]
    keys = ["Main-PW", "WSS-a-PW", "AsymW-a", "WssDO-a"]
    vals = [best_wss_metrics(RUNS[k])["wss_r2_wss"] for k in keys]
    best_idx = int(np.argmax(vals))

    colors = [
        PALETTE["baseline"],
        PALETTE["accent_blue"],
        PALETTE["accent_green"],
        PALETTE["accent_amber"],
    ]
    colors[best_idx] = PALETTE["highlight"]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.bar(
        labels,
        vals,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        width=0.62,
        zorder=3,
    )
    bars[best_idx].set_edgecolor(PALETTE["highlight"])
    bars[best_idx].set_linewidth(2.0)

    ax.axhline(
        0.40,
        color=PALETTE["ref_line"],
        ls="--",
        lw=1.4,
        alpha=0.75,
        label="~0.40 经验上限",
        zorder=2,
    )
    ax.set_ylim(0.32, 0.43)
    ax.set_ylabel("test wss_r2_wss（best_wss_model）")
    ax.set_title("V3P 主线 WSS 对比（split_AG_v1, seed=1）", pad=14)
    annotate_bars(ax, bars, vals, offset=0.012, fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    style_axes(ax)
    fig.tight_layout(pad=1.2)
    save_fig(fig, "v3p_wss_comparison.png")


def fig_wss_components():
    exps = ["Main-PW", "AsymW-a"]
    comps = ["wss_r2_wss_z", "wss_r2_wss_x", "wss_r2_wss_y"]
    comp_labels = ["wss_z", "wss_x", "wss_y"]
    x = np.arange(len(comp_labels))
    width = 0.34
    exp_colors = [PALETTE["baseline"], PALETTE["highlight"]]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for i, exp in enumerate(exps):
        m = best_wss_metrics(RUNS[exp])
        vals = [m[c] for c in comps]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=exp,
            color=exp_colors[i],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            y_pos = v + 0.012 if v >= 0 else v - 0.018
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{v:.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9,
                color=PALETTE["text"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel("test R²（best_wss_model）")
    ax.set_title("WSS 分量对比：Main-PW vs AsymW-a", pad=14)
    ax.axhline(0, color=PALETTE["muted"], lw=0.9, zorder=2)
    ax.legend(loc="upper right", fontsize=10)
    style_axes(ax)
    fig.tight_layout(pad=1.2)
    save_fig(fig, "v3p_wss_components.png")


def fig_val_wss_history():
    paths = {
        "Main-PW": RUNS["Main-PW"] / "history.csv",
        "AsymW-a": RUNS["AsymW-a"] / "history.csv",
    }
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.72), sharey=True)
    cols = [
        ("val_wss_r2_wss_x", "wss_x"),
        ("val_wss_r2_wss_y", "wss_y"),
        ("val_wss_r2_wss_z", "wss_z"),
    ]
    styles = {
        "wss_x": PALETTE["wss_x"],
        "wss_y": PALETTE["wss_y"],
        "wss_z": PALETTE["wss_z"],
    }

    for ax, (name, csv_path) in zip(axes, paths.items()):
        if not csv_path.is_file():
            ax.text(0.5, 0.5, f"缺少 {csv_path.name}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name, pad=10)
            style_axes(ax)
            continue
        df = pd.read_csv(csv_path)
        for col, lab in cols:
            if col in df.columns:
                ax.plot(
                    df["epoch"],
                    df[col],
                    label=lab,
                    color=styles[lab],
                    lw=2.0,
                    alpha=0.92,
                )
        ax.axhline(0, color=PALETTE["muted"], lw=0.8)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val R²")
        ax.set_title(name, pad=10)
        ax.legend(loc="lower right", fontsize=9)
        style_axes(ax)

    fig.suptitle("验证集 WSS 分量 R² 曲线（过拟合诊断）", y=1.02, fontsize=14, fontweight="bold")
    fig.tight_layout(pad=1.4)
    save_fig(fig, "v3p_val_wss_components_history.png")


def fig_v3d_per_domain():
    with open(V3D_PROBE_P, encoding="utf-8") as f:
        p_data = json.load(f)
    with open(V3D_PROBE_WSS, encoding="utf-8") as f:
        w_data = json.load(f)

    domains = ["AAA", "AG", "ILO"]
    p_vals = [p_data["by_domain"][d]["metrics"]["r2_p"] for d in domains]
    w_vals = [w_data["by_domain"][d]["metrics"]["wss_r2_wss"] for d in domains]

    x = np.arange(len(domains))
    width = 0.34
    fig, ax1 = plt.subplots(figsize=(FIG_W, FIG_H))
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)

    b1 = ax1.bar(
        x - width / 2,
        p_vals,
        width,
        label="Probe-P  r2_p",
        color=PALETTE["accent_blue"],
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    b2 = ax2.bar(
        x + width / 2,
        w_vals,
        width,
        label="Probe-WSS  wss_r2_wss",
        color=PALETTE["accent_orange"],
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(domains)
    ax1.set_ylim(0.88, 1.005)
    ax2.set_ylim(0.17, 0.31)
    ax1.set_ylabel("r2_p", color=PALETTE["accent_blue"])
    ax2.set_ylabel("wss_r2_wss", color=PALETTE["accent_orange"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["accent_blue"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["accent_orange"])
    ax1.set_title("V3D post-4901 分域 test（split_data_new_v3_v3, seed=1）", pad=14)

    for b, v in zip(b1, p_vals):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v - 0.012,
            f"{v:.3f}",
            ha="center",
            color="white",
            fontsize=9,
            fontweight="medium",
        )
    for b, v in zip(b2, w_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            v + 0.006,
            f"{v:.3f}",
            ha="center",
            fontsize=9,
            color=PALETTE["text"],
            fontweight="medium",
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10)
    style_axes(ax1)
    fig.tight_layout(pad=1.2)
    save_fig(fig, "v3d_per_domain_metrics.png")


# ---------------------------------------------------------------------------
# Optional: matplotlib placeholders (deprecated — use external tools instead)
# See PPT补充计划.md §7 for draw.io / Figma / Mermaid prompts.
# ---------------------------------------------------------------------------


def fig_path_map_optional():
    """Deprecated placeholder. Prefer user-generated v3_path_map.png."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(5, 6.5, "当前带宽：V3P ~0.40 / V3D ~0.24", ha="center", fontsize=13, fontweight="bold")
    boxes = [
        (0.3, 4.2, 2.8, 1.4, "路径 A (P0)\nWSS 局部坐标 / magnitude\nTODO-5,9,10,11", "#dcfce7"),
        (3.6, 4.2, 2.8, 1.4, "路径 B (P1)\n近壁几何与梯度\nTODO-12,17,18", "#dbeafe"),
        (6.9, 4.2, 2.8, 1.4, "路径 C (P1–P2)\n三域数据与归一化\nTODO-1,3,4,19", "#fef3c7"),
        (2.0, 2.0, 2.8, 1.3, "路径 D (P1 并行)\n病例级 / top-k 评估\nTODO-8,20", "#e0e7ff"),
        (5.2, 2.0, 2.8, 1.3, "路径 E (暂缓)\n架构/MTL/发散\nTODO-7,13–16,21–26", "#f3f4f6"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#374151", lw=1.2
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    ax.text(
        5,
        1.0,
        "推荐顺序：阶段0 等 4999/5000/5001 → 阶段1 并行 TODO-8 + TODO-5 + TODO-19",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    save_fig(fig, "v3_path_map.png")


def fig_todo_priority_optional():
    """Deprecated placeholder. Prefer user-generated v3_todo_priority.png."""
    rows = [
        ("TODO-5", "WSS 局部坐标系", "A", "P0", "未开始（下一跳）"),
        ("TODO-8", "病例级/高 WSS 评估", "D", "P1", "未开始"),
        ("TODO-19", "三域 per-domain 归一化", "C", "P1", "未开始"),
        ("TODO-6", "非对称权重 AsymW", "E→A", "P2", "进行中 seed2/3"),
        ("TODO-7", "WSS head Dropout", "E", "P2", "暂缓"),
        ("TODO-1", "数据扩充 257→150+", "C", "P0", "进行中"),
        ("TODO-12", "近壁几何特征", "B", "P1", "未开始"),
        ("TODO-21–26", "发散储备", "E", "P3", "未立项"),
    ]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 0.55))
    ax.axis("off")
    table = ax.table(
        cellText=[[r[0], r[1], r[2], r[3], r[4]] for r in rows],
        colLabels=["ID", "标题", "路径", "优先级", "状态"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1e40af")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 3 and row > 0:
            pri = rows[row - 1][3]
            if pri == "P0":
                cell.set_facecolor("#fef2f2")
            elif pri == "P1":
                cell.set_facecolor("#fffbeb")
    ax.set_title("V3 后续优化待办（节选，2026-05-23）", pad=12, fontsize=12)
    fig.tight_layout()
    save_fig(fig, "v3_todo_priority.png")


def main():
    font = setup_style()
    print(f"font: {font}, dpi: {DPI}, size: {FIG_W}x{FIG_H}")
    fig_wss_comparison()
    fig_wss_components()
    fig_val_wss_history()
    fig_v3d_per_domain()
    print(f"figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
