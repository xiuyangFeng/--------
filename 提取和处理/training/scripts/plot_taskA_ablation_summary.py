from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from ..core.field_plot_paths import CAT_ABLATION, category_dir
from ._figure_utils import load_json, read_regional_metric, resolve_run_dirs, save_json

# 与 plot_taskA_multimodel_* 对齐，便于 Fig A6 横条使用短标签
_SHORT_LABELS_EXP_ID: Dict[str, str] = {
    "A-Opt-05": "A-Opt-05 (baseline)",
    "A-Abl-02-01": "Opt-05 w/o Abscissa",
    "A-Abl-02-02": "Opt-05 w/o NormRadius",
    "A-Abl-02-03": "Opt-05 w/o Curvature",
    "A-Abl-02-04": "Opt-05 w/o Tangent",
}


def load_run_metric(
    run_dir: Path,
    metric_key: str,
    region: str = "all",
    group_by: str = "experiment_name",
) -> Dict[str, object]:
    summary = load_json(run_dir / "summary.json")
    manifest = load_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    test_metrics = summary.get("test_metrics", {})
    if not isinstance(test_metrics, dict):
        raise ValueError(f"summary.json 中缺少 test_metrics: {run_dir}")

    experiment_name = str(summary.get("experiment_name", manifest.get("experiment_name", run_dir.name)))
    exp_id = str(summary.get("exp_id", manifest.get("exp_id", "")))
    seed = int(summary.get("seed", manifest.get("seed", 0)))

    if group_by == "exp_id":
        group_key = exp_id if exp_id else experiment_name
    else:
        group_key = experiment_name

    metric_value = None
    if region != "all":
        metric_value = read_regional_metric(run_dir, region, metric_key)
    if metric_value is None:
        metric_value = test_metrics.get(metric_key)
    if metric_value is None:
        raise ValueError(f"{run_dir} 中缺少指标 {metric_key} (region={region})")
    return {
        "group_key": group_key,
        "experiment_name": experiment_name,
        "exp_id": exp_id,
        "seed": seed,
        "metric_value": float(metric_value),
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="聚合多 seed 实验并生成消融总结图")
    parser.add_argument("--runs-root", default="outputs/field", help="run 根目录")
    parser.add_argument("--pattern", action="append", default=[], help="相对 runs-root 的匹配模式，可重复传入；默认 */summary.json")
    parser.add_argument("--run-dir", action="append", default=[], help="显式指定 run 目录，可重复传入")
    parser.add_argument(
        "--run-list",
        default="",
        help="文本文件：每行一个 run 目录（相对仓库根或绝对路径），# 开头为注释；与 --run-dir 可混用",
    )
    parser.add_argument("--metric-key", default="rmse_vel_mag", help="用于比较的 test_metrics 指标")
    parser.add_argument(
        "--region", default="interior",
        choices=["all", "interior", "wall"],
        help="指标来源区域（默认 interior；优先读 regional_eval，回退 summary.json）",
    )
    parser.add_argument("--baseline", default="", help="基线实验名；默认取第一组")
    parser.add_argument(
        "--group-by",
        default="experiment_name",
        choices=["experiment_name", "exp_id"],
        help="聚合键：按 experiment_name 或 exp_id（推荐与 --order 联用）",
    )
    parser.add_argument(
        "--order",
        default="",
        help="逗号分隔的分组键列表，控制横条顺序（须与 --group-by 一致）；为空则按分组键排序",
    )
    parser.add_argument(
        "--short-labels",
        action="store_true",
        help="横条 Y 轴使用内置短标签（仅当 --group-by exp_id 且为已知 A-Opt-05 / A-Abl-02-xx）",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="输出目录，默认 <runs-root>/plots/ablation",
    )
    parser.add_argument("--title", default="", help="图标题（为空则自动生成）")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    run_dir_args: List[str] = list(args.run_dir)
    if args.run_list.strip():
        list_path = Path(args.run_list).resolve()
        if not list_path.is_file():
            raise SystemExit(f"--run-list 文件不存在: {list_path}")
        with list_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                raw = Path(line).expanduser()
                p = raw.resolve() if raw.is_absolute() else (Path.cwd() / raw).resolve()
                if not p.is_dir():
                    raise SystemExit(f"--run-list 中目录不存在: {p}")
                run_dir_args.append(str(p))
    # 显式传入 --run-dir / --run-list 时，除非同时指定 --pattern，否则不再全盘 glob，避免混入其它 run。
    if args.pattern:
        patterns = args.pattern
    elif run_dir_args:
        patterns = []
    else:
        patterns = ["*/summary.json"]
    run_dirs = resolve_run_dirs(runs_root, patterns, run_dir_args)
    run_dirs = [p for p in run_dirs if (p / "summary.json").exists()]
    if not run_dirs:
        raise SystemExit("未找到任何包含 summary.json 的 run 目录")

    try:
        from ..analysis.stats import compare_experiments, summarize_seeds
    except ImportError as exc:
        raise SystemExit(
            "运行消融统计失败：当前环境缺少 scipy。请先安装依赖，例如 `pip install -r training/requirements.txt`。"
        ) from exc
    try:
        from ..analysis.visualization import plot_ablation_summary
    except ImportError as exc:
        raise SystemExit(
            "生成图像失败：当前环境缺少 matplotlib。请先安装可视化依赖，例如 `pip install -r training/requirements.txt`。"
        ) from exc

    grouped: Dict[str, Dict[int, float]] = {}
    for run_dir in run_dirs:
        record = load_run_metric(
            run_dir, args.metric_key, region=args.region, group_by=args.group_by
        )
        grouped.setdefault(str(record["group_key"]), {})[int(record["seed"])] = float(record["metric_value"])

    if args.order.strip():
        order_keys = [x.strip() for x in args.order.split(",") if x.strip()]
        experiment_names = [k for k in order_keys if k in grouped]
        missing_order = [k for k in order_keys if k not in grouped]
        if missing_order:
            raise SystemExit(f"--order 中有未在 run 中出现的分组键: {missing_order}")
        extra = sorted(k for k in grouped if k not in experiment_names)
        experiment_names.extend(extra)
    else:
        experiment_names = sorted(grouped)
    means: List[float] = []
    stds: List[float] = []
    rows: List[Dict[str, object]] = []
    stats_payload: Dict[str, object] = {"metric_key": args.metric_key, "experiments": {}, "comparisons": {}}
    for name in experiment_names:
        summary = summarize_seeds(grouped[name])
        means.append(summary["mean"])
        stds.append(summary["std"])
        rows.append({"group": name, **summary})
        stats_payload["experiments"][name] = summary

    stats_payload["meta"] = {"group_by": args.group_by, "region": args.region}

    baseline_name = args.baseline or (experiment_names[0] if experiment_names else "")
    for name in experiment_names:
        if name == baseline_name:
            continue
        stats_payload["comparisons"][f"{baseline_name}__vs__{name}"] = compare_experiments(
            grouped[baseline_name],
            grouped[name],
            lower_is_better=True,
        )

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else category_dir(runs_root, CAT_ABLATION)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    highlight_idx = experiment_names.index(baseline_name) if baseline_name in experiment_names else None

    region_tag = "" if args.region == "all" else f"_{args.region}"
    figure_path = output_dir / f"fig_A6_ablation_summary{region_tag}.png"
    csv_path = output_dir / f"fig_A6_ablation_summary{region_tag}.csv"
    stats_path = output_dir / f"fig_A6_ablation_summary_stats{region_tag}.json"

    if args.short_labels and args.group_by == "exp_id":
        bar_labels = [_SHORT_LABELS_EXP_ID.get(k, k) for k in experiment_names]
    else:
        bar_labels = list(experiment_names)

    min_seeds = min(len(grouped[k]) for k in experiment_names) if experiment_names else 0
    max_seeds = max(len(grouped[k]) for k in experiment_names) if experiment_names else 0
    if args.title:
        title = args.title
    elif min_seeds >= 2 and min_seeds == max_seeds:
        title = (
            f"Figure A6 Ablation Summary ({args.region} nodes) — "
            f"mean ± sample std over {min_seeds} seeds"
        )
    elif min_seeds != max_seeds:
        title = (
            f"Figure A6 Ablation Summary ({args.region} nodes) — "
            f"unequal seeds per group (min={min_seeds}, max={max_seeds})"
        )
    else:
        title = f"Figure A6 Ablation Summary ({args.region} nodes)"

    plot_ablation_summary(
        experiment_names=bar_labels,
        metric_means=means,
        metric_stds=stds,
        metric_name=f"{args.metric_key} ({args.region})",
        highlight_idx=highlight_idx,
        save_path=figure_path,
        title=title,
    )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json(stats_path, stats_payload)

    print(figure_path)
    print(csv_path)
    print(stats_path)


if __name__ == "__main__":
    main()
