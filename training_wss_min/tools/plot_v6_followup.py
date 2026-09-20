"""Plot only the verified, complete A5 follow-up summaries (no model inference)."""
from __future__ import annotations

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .report_v6_followup import EXP


def main():
    summaries, hashes = {}, {}
    for checkpoint in ("best", "last"):
        path = EXP / f"matrix_summary_{checkpoint}.json"
        summaries[checkpoint] = json.loads(path.read_text())
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary = summaries[checkpoint]
        if (summary["completed_count"] != 15 or summary["source_integrity"]["changed"]
                or not summary.get("matrix_finalized", False)):
            raise RuntimeError(f"{checkpoint}: all 15 verified results required before final plotting")
    ids = list(summaries["best"]["arms"])
    if ids != list(summaries["last"]["arms"]):
        raise ValueError("best/last arms differ")
    labels = {
        "D1": "管道坐标", "D2": "分叉/端区距离", "D3": "半径梯度", "D4": "显式法向",
        "D5": "细尺度曲率", "D6": "粗尺度曲率", "D7": "切向·法向",
        "L1": "纯 MSE", "L2": "原损失 + Pa-Huber", "L3": "MSE + Pa-Huber",
        "M1": "逐点容量对照", "M2": "独立 query / 3NN", "M3": "独立 query / 16NN",
        "M4": "可学局部解码", "M5": "单一邻域半径",
    }
    font_path = EXP / "assets/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": font_name, "font.size": 10,
                         "axes.unicode_minus": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(17, 9.5), sharey=True)
    panels = [("physical_r2_cb", "物理 R²_cb：相对 A5 变化"),
              ("normalized_r2_cb", "log_z R²_cb：相对 A5 变化"),
              ("top10_iou", "top10 IoU：相对 A5 变化"),
              ("p99_ratio", "p99 幅值比：接近 1 的改善量")]
    y = np.arange(len(ids))
    for ax, (key, title) in zip(axes, panels):
        for checkpoint, offset, color, marker in (("best", -.13, "#1763a6", "o"),
                                                  ("last", .13, "#cf6a21", "s")):
            summary = summaries[checkpoint]
            ref = summary["reference"][key]
            actual = np.asarray([summary["arms"][i][key] for i in ids])
            delta = abs(ref - 1) - abs(actual - 1) if key == "p99_ratio" else actual - ref
            ax.scatter(delta, y + offset, s=34, color=color, marker=marker,
                       label=checkpoint, zorder=3)
            ax.hlines(y + offset, 0, delta, color=color, alpha=.25)
        ax.axvline(0, color="#4b5563", linewidth=.9)
        for boundary in (6.5, 9.5):
            ax.axhline(boundary, color="#94a3b8", linewidth=.7)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=.2)
        ax.set_xlabel("← 退化     改善 →")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, [f"{i}  {labels[i]}" for i in ids])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left")
    fig.suptitle("V6-A5 后续 15 臂：单 seed=1234 的统一 WSS 评估", fontsize=17, y=.97)
    fig.text(.04, .025,
             "D 行为屏蔽对应输入；best/last 分别与历史 A5 的同类 checkpoint 比较。"
             "所有点来自 400 epoch 与 test34 完整评估。\n"
             "无误差条：本轮未重复种子，图中变化不能解释为显著性或稳定性。"
             "p99 面板 = |A5 比值−1| − |该臂比值−1|，正值表示更接近 1。", fontsize=10)
    fig.subplots_adjust(left=.19, right=.98, top=.89, bottom=.13, wspace=.25)
    for extension in ("png", "pdf"):
        fig.savefig(EXP / f"followup_comparison.{extension}", dpi=170)
    plt.close(fig)
    (EXP / "figure_provenance.json").write_text(json.dumps({"source_sha256": hashes,
        "checkpoint_comparison": "each checkpoint against same historical A5 checkpoint",
        "single_seed": 1234, "uncertainty_estimated": False,
        "font_sha256": hashlib.sha256(font_path.read_bytes()).hexdigest()}, indent=2) + "\n")
    print("Saved followup_comparison.png/pdf from 30 verified evaluations")


if __name__ == "__main__":
    main()
