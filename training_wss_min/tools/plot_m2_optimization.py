"""Scientific figures from verified M2 reports, histories and per-case changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from training_wss_min.tools import report_m2_optimization as R


def configure_font():
    font = R.EXP.parent / "v6_followup_20260909/assets/NotoSansCJK-Regular.ttc"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        name = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        raise FileNotFoundError(f"The established Chinese report font is missing: {font}")
    plt.rcParams.update({"font.family": name, "font.size": 9, "axes.unicode_minus": False, "pdf.fonttype": 42})
    return font


def save_figure(fig, directory, stem):
    result = []
    for extension in ("png", "pdf"):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=165, bbox_inches="tight")
        result.append(path)
    plt.close(fig)
    return result


def comparison_figure(reports, directory):
    ids = [aid for aid in reports["best"]["arms"] if all(reports[ck]["arms"][aid]["evidence"]["valid_scientific_result"] for ck in ("best", "last"))]
    if not ids:
        return []
    panels = [("physical_r2_cb", "Pa R²_cb 增量", False), ("normalized_r2_cb", "log_z R²_cb 增量", False),
              ("mae", "MAE Pa 减少量", True), ("top10_iou", "热点 IoU 增量", False),
              ("high_wss_r2", "高 WSS R² 增量", False), ("p99_ratio", "p99 比距 1 的改善量", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(23, max(7, len(ids) * .31 + 2)), sharey=True)
    y = np.arange(len(ids))
    for axis, (key, title, reverse) in zip(axes, panels):
        for ck, offset, color, mark in (("best", -.14, "#1763a6", "o"), ("last", .14, "#c86519", "s")):
            summary = reports[ck]
            ref = summary["reference"][key]
            actual = np.array([summary["arms"][aid][key] for aid in ids])
            difference = abs(ref - 1) - abs(actual - 1) if key == "p99_ratio" else ref - actual if reverse else actual - ref
            axis.scatter(difference, y + offset, s=24, color=color, marker=mark, label=ck, zorder=3)
            axis.hlines(y + offset, 0, difference, color=color, alpha=.22)
        axis.axvline(0, color="#64748b", linewidth=.8)
        axis.set_title(title)
        axis.grid(axis="x", alpha=.2)
        axis.set_xlabel("← 退化   改善 →")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, ids)
    axes[0].invert_yaxis()
    axes[0].legend(loc="best")
    fig.suptitle("M2 损失、参数、结构与优胜组合：相对历史 M2 的同 checkpoint 变化", fontsize=15, y=1.01)
    fig.text(.08, -.025, "s1234；400轮；已暴露test34。best由各臂自身训练总损失选择，last固定epoch399。无误差条，不表示显著性或稳定性。\n"
             "同期MO0与历史M2的差异见复现审计；保留全部有效单项和组合，不以筛选门槛过滤负结果。", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, directory, "m2_optimization_comparison")


def training_figure(reports, directory, runs_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    source_hashes, has_data = {}, False
    groups = [("L", "损失"), ("P", "参数"), ("S", "结构"), ("C", "组合")]
    arms = reports["best"]["arms"]
    for axis, (family, title) in zip(axes.flat, groups):
        ids = [aid for aid, a in arms.items() if (a.get("phase") == "combination" if family == "C" else a.get("phase") == "base" and R.family_of(a) == family)]
        for aid in ["MO0", *ids]:
            entry = arms.get(aid)
            if not entry:
                continue
            path = Path(runs_dir) / entry["run_name"] / "history.jsonl"
            if not path.exists():
                continue
            history = []
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                if "train_rmse_norm" in row or "train_mse_norm" in row:
                    history.append(row)
            if not history:
                continue
            source_hashes[str(path)] = R.sha256(path)
            epoch = [r["epoch"] + 1 for r in history]
            error = [r["train_rmse_norm"] if "train_rmse_norm" in r else r["train_mse_norm"] ** .5 for r in history]
            axis.plot(epoch, error, label=aid, color="black" if aid == "MO0" else None,
                      linewidth=2 if aid == "MO0" else 1, alpha=1 if aid == "MO0" else .85)
            has_data = True
        axis.set_title(f"{title}族：训练采样点 log_z RMSE")
        axis.set_xlabel("epoch")
        axis.set_ylabel("log_z RMSE")
        axis.set_xlim(1, 400)
        axis.set_yscale("log")
        axis.grid(alpha=.15)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8, ncol=2)
    fig.suptitle("统一训练误差曲线；不横向比较不同损失的总 loss", fontsize=15)
    fig.text(.08, .01, "每轮重新抽样；图为训练采样误差。所有臂固定400轮，不以测试结果早停或选择checkpoint。", fontsize=10)
    fig.tight_layout(rect=(0, .03, 1, .97))
    if not has_data:
        plt.close(fig)
        return [], source_hashes
    return save_figure(fig, directory, "m2_training_curves"), source_hashes


def case_label(name):
    pieces = name.split("/")
    return "/".join(pieces[-2:]) if pieces[-1] in {"before", "after"} else pieces[-1]


def contribution_figures(reports, directory):
    outputs = []
    for ck in ("best", "last"):
        entries = reports[ck]["arms"]
        ids = [aid for aid, a in entries.items() if a["evidence"]["valid_scientific_result"] and "cases_vs_mo0" in a]
        if not ids:
            continue
        cases = list(entries[ids[0]]["cases_vs_mo0"]["per_case"])
        # Sort by historical M2 case R² rather than the new winner's failures.
        cases.sort(key=lambda c: entries[ids[0]]["cases_vs_historical_m2"]["per_case"][c]["r2_before"])
        data = np.array([[entries[aid]["cases_vs_mo0"]["per_case"][case]["mse_r2_contribution"] for case in cases] for aid in ids])
        limit = max(float(np.max(np.abs(data))), 1e-5)
        fig, axis = plt.subplots(figsize=(18, max(7, len(ids) * .29 + 2)))
        plot = axis.imshow(data, aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit)
        axis.set_yticks(np.arange(len(ids)), ids)
        axis.set_xticks(np.arange(len(cases)), [case_label(c) for c in cases], rotation=75, ha="right", fontsize=8)
        axis.set_title(f"{ck}：每病例对 ΔPa R²_cb 的贡献（相对同期 MO0）", fontsize=14, pad=15)
        fig.colorbar(plot, ax=axis, shrink=.75, label="病例MSE减少 / (34 × 固定病例等权真值方差)")
        fig.text(.08, -.015, "每行34例贡献之和等于该臂相对MO0的整体ΔPa R²_cb；蓝色改善、红色退化。病例按历史M2 R²排序。", fontsize=10)
        fig.tight_layout()
        outputs.extend(save_figure(fig, directory, f"m2_case_contributions_{ck}"))
    return outputs


def plot_all(experiment_dir=R.EXP, runs_dir=R.RUNS, require_complete=True):
    directory = Path(experiment_dir)
    font = configure_font()
    reports, hashes = {}, {str(font): R.sha256(font)}
    for ck in ("best", "last"):
        path = directory / f"matrix_summary_{ck}.json"
        reports[ck] = json.loads(path.read_text())
        hashes[str(path)] = R.sha256(path)
        if require_complete and not reports[ck]["matrix_finalized"]:
            raise RuntimeError(f"{ck}: final figures require all registered runs to be verified")
    if list(reports["best"]["arms"]) != list(reports["last"]["arms"]):
        raise ValueError("best/last arm sets differ")
    files = comparison_figure(reports, directory)
    curves, history_hashes = training_figure(reports, directory, runs_dir)
    files += curves + contribution_figures(reports, directory)
    hashes.update(history_hashes)
    evidence = {"source_sha256": hashes, "figure_sha256": {p.name: R.sha256(p) for p in files},
                "single_seed": 1234, "uncertainty_estimated": False,
                "metric_comparison_reference": "historical M2; each checkpoint against like checkpoint",
                "case_contribution_reference": "contemporaneous MO0; each checkpoint against like checkpoint",
                "all_registered_runs_finalized": all(reports[ck]["matrix_finalized"] for ck in ("best", "last"))}
    (directory / "figure_provenance.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=R.EXP)
    parser.add_argument("--runs-dir", type=Path, default=R.RUNS)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    result = plot_all(args.experiment_dir, args.runs_dir, not args.allow_partial)
    print(json.dumps({"files": list(result["figure_sha256"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
