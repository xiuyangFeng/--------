"""Plot only the verified, complete multi-radius BT summaries (no model inference)."""
from __future__ import annotations

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .report_v6_multiradius import EXP


def main():
    summaries, hashes = {}, {}
    for checkpoint in ("best", "last"):
        path = EXP / f"matrix_summary_{checkpoint}.json"
        summaries[checkpoint] = json.loads(path.read_text())
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary = summaries[checkpoint]
        if (summary["completed_count"] != 6 or summary["source_integrity"]["changed"]
                or not summary.get("matrix_finalized", False)):
            raise RuntimeError(f"{checkpoint}: all 6 verified results required before final plotting")
    ids = list(summaries["best"]["arms"])
    if ids != list(summaries["last"]["arms"]):
        raise ValueError("best/last arms differ")
    labels = {"MS1": "单半径接入对照", "MS2": "单半径 + BT", "MS3": "多半径分组与融合", "MS4": "多半径 + 独立BT", "MS5": "相同半径 + 独立BT", "MS6": "多半径 + 近容量FFN"}
    font_path = EXP.parent / "v6_followup_20260909/assets/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": font_name, "font.size": 10,
                         "axes.unicode_minus": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.0), sharey=True)
    panels = [("physical_r2_cb", "物理 R²_cb：相对 M2 变化"),
              ("normalized_r2_cb", "log_z R²_cb：相对 M2 变化"),
              ("top10_iou", "top10 IoU：相对 M2 变化"),
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
        for boundary in (2.5,):
            ax.axhline(boundary, color="#94a3b8", linewidth=.7)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=.2)
        ax.set_xlabel("← 退化     改善 →")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, [f"{i}  {labels[i]}" for i in ids])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left")
    fig.suptitle("多半径集合与独立 BT：6组、单 seed=1234 的 WSS 评估", fontsize=17, y=.97)
    fig.text(.04, .02,
             "best/last 分别与历史 M2 的同类 checkpoint 比较；新增SA3统一cap32。"
             "所有点来自 400 epoch 与 test34 完整评估。\n"
             "无误差条：本轮未重复种子，图中变化不能解释为显著性或稳定性。"
             "p99 面板 = |M2 比值−1| − |该臂比值−1|，正值表示更接近 1。", fontsize=10)
    fig.subplots_adjust(left=.19, right=.98, top=.89, bottom=.19, wspace=.25)
    for extension in ("png", "pdf"):
        fig.savefig(EXP / f"multiradius_comparison.{extension}", dpi=170)
    plt.close(fig)
    (EXP / "figure_provenance.json").write_text(json.dumps({"source_sha256": hashes,
        "checkpoint_comparison": "each checkpoint against same historical M2 checkpoint",
        "single_seed": 1234, "uncertainty_estimated": False,
        "font_sha256": hashlib.sha256(font_path.read_bytes()).hexdigest()}, indent=2) + "\n")
    print("Saved multiradius_comparison.png/pdf from 12 verified evaluations")


if __name__ == "__main__":
    main()
