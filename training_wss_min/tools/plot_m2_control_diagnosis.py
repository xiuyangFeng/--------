"""Plot the completed control discrepancy without changing the experiment gate."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training_wss_min.tools import report_m2_optimization as R
from training_wss_min.tools.m2_optimization_common import ANCHOR_RUN, EXP, RUNS, save_json, sha, stamp
from training_wss_min.tools.plot_m2_optimization import case_label, configure_font


def main():
    configure_font()
    control_run = RUNS / "m2_optimization_20260909/MO0_s1234"
    paths = [ANCHOR_RUN / "history.jsonl", control_run / "history.jsonl"]
    histories = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
    if any([h["epoch"] for h in history] != list(range(400)) for history in histories):
        raise RuntimeError("Both control histories must contain the original 400 epochs")
    comparisons = {}
    for ck in ("best", "last"):
        original = R.load_metrics(R.ANCHOR, ck)
        current = R.load_metrics("m2_optimization_20260909/MO0_s1234", ck)
        comparisons[ck] = R.case_comparison(current, original)
        paths.extend([run / "eval" / f"ckpt_{ck}" / "metrics.json" for run in (ANCHOR_RUN, control_run)])
    before = {str(path): sha(path) for path in paths}
    old_loss, new_loss = (np.array([h["train_loss"] for h in history]) for history in histories)
    epoch = np.arange(1, 401)
    fig = plt.figure(figsize=(14, 15))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 3.4), hspace=.28)
    axis = fig.add_subplot(grid[0, 0])
    axis.plot(epoch, old_loss, label="历史 M2", color="#246aa0", linewidth=1.5)
    axis.plot(epoch, new_loss, label="同期 MO0", color="#ce701d", linewidth=1, alpha=.8)
    axis.set(title="原配方训练总损失", xlabel="epoch", ylabel="log-MSE + 0.2 × q90 pinball", yscale="log")
    axis.legend()
    axis.grid(alpha=.15)
    axis = fig.add_subplot(grid[0, 1])
    axis.plot(epoch, new_loss - old_loss, color="#70528b", linewidth=.8)
    axis.axhline(0, color="black", linewidth=.6)
    axis.set(title=f"逐轮损失差；两条曲线相关 {np.corrcoef(old_loss, new_loss)[0, 1]:.6f}",
             xlabel="epoch", ylabel="MO0 − 历史 M2")
    axis.grid(alpha=.15)
    axis = fig.add_subplot(grid[1, :])
    best_cases = comparisons["best"]["per_case"]
    cases = sorted(best_cases, key=lambda c: best_cases[c]["mse_r2_contribution"])
    y = np.arange(len(cases))
    for ck, offset, color in (("best", -.17, "#246aa0"), ("last", .17, "#ce701d")):
        values = [comparisons[ck]["per_case"][case]["mse_r2_contribution"] for case in cases]
        axis.barh(y + offset, values, height=.31, color=color, label=f"{ck}：总和 {sum(values):+.5f}")
    axis.set_yticks(y, [case_label(case) for case in cases], fontsize=8)
    axis.invert_yaxis()
    axis.axvline(0, color="black", linewidth=.7)
    axis.grid(axis="x", alpha=.15)
    axis.set_title("完整 34 病例：对同期 MO0 相对历史 M2 的 Pa R²_cb 差值贡献", pad=12)
    axis.set_xlabel("由病例 MSE 和固定真值方差计算；34 项之和等于整体 ΔPa R²_cb")
    axis.legend(loc="lower left")
    fig.suptitle("同期对照偏差：相近训练曲线，完整壁面 Pa 指标存在差距", fontsize=16, y=.965)
    fig.text(.5, .025,
        "单 seed1234；400 轮；同一 test34 与固定 support。GPU 短步重复已有数值分叉，不能据此定量解释最终差距。\n"
        "best 均按自身训练总损失选取（第 379 轮），last 均为第 400 轮；不按测试结果改选 checkpoint。",
        ha="center", fontsize=10)
    fig.subplots_adjust(left=.20, right=.97, bottom=.09, top=.92)
    artifacts = []
    for extension in ("png", "pdf"):
        path = EXP / f"contemporary_control_diagnosis.{extension}"
        fig.savefig(path, dpi=170, facecolor="white")
        artifacts.append({"path": str(path), "sha256": sha(path)})
    plt.close(fig)
    if before != {str(path): sha(path) for path in paths}:
        raise RuntimeError("Control inputs changed while plotting")
    save_json(EXP / "control_plot_manifest.json", {"passed": True, "created_at": stamp(),
        "script_sha256": sha(__file__), "inputs": before, "artifacts": artifacts,
        "case_order": cases, "contribution_sums": {ck: comparisons[ck]["contribution_sum"] for ck in comparisons}})
    print(json.dumps({"passed": True, "artifacts": artifacts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
