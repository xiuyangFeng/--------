from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from ..core.field_plot_paths import CAT_ABLATION, category_dir
from ._figure_utils import load_json, read_regional_metric, resolve_run_dirs, save_json


def load_run_metric(
    run_dir: Path, metric_key: str, region: str = "all",
) -> Dict[str, object]:
    summary = load_json(run_dir / "summary.json")
    manifest = load_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    test_metrics = summary.get("test_metrics", {})
    if not isinstance(test_metrics, dict):
        raise ValueError(f"summary.json 中缺少 test_metrics: {run_dir}")

    experiment_name = str(summary.get("experiment_name", manifest.get("experiment_name", run_dir.name)))
    seed = int(summary.get("seed", manifest.get("seed", 0)))

    metric_value = None
    if region != "all":
        metric_value = read_regional_metric(run_dir, region, metric_key)
    if metric_value is None:
        metric_value = test_metrics.get(metric_key)
    if metric_value is None:
        raise ValueError(f"{run_dir} 中缺少指标 {metric_key} (region={region})")
    return {
        "experiment_name": experiment_name,
        "seed": seed,
        "metric_value": float(metric_value),
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="聚合多 seed 实验并生成消融总结图")
    parser.add_argument("--runs-root", default="outputs/field", help="run 根目录")
    parser.add_argument("--pattern", action="append", default=[], help="相对 runs-root 的匹配模式，可重复传入；默认 */summary.json")
    parser.add_argument("--run-dir", action="append", default=[], help="显式指定 run 目录，可重复传入")
    parser.add_argument("--metric-key", default="rmse_vel_mag", help="用于比较的 test_metrics 指标")
    parser.add_argument(
        "--region", default="interior",
        choices=["all", "interior", "wall"],
        help="指标来源区域（默认 interior；优先读 regional_eval，回退 summary.json）",
    )
    parser.add_argument("--baseline", default="", help="基线实验名；默认取第一组")
    parser.add_argument(
        "--output-dir",
        default="",
        help="输出目录，默认 <runs-root>/plots/ablation",
    )
    parser.add_argument("--title", default="", help="图标题（为空则自动生成）")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    patterns = args.pattern or ["*/summary.json"]
    run_dirs = resolve_run_dirs(runs_root, patterns, args.run_dir)
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
        record = load_run_metric(run_dir, args.metric_key, region=args.region)
        grouped.setdefault(str(record["experiment_name"]), {})[int(record["seed"])] = float(record["metric_value"])

    experiment_names = sorted(grouped)
    means: List[float] = []
    stds: List[float] = []
    rows: List[Dict[str, object]] = []
    stats_payload: Dict[str, object] = {"metric_key": args.metric_key, "experiments": {}, "comparisons": {}}
    for name in experiment_names:
        summary = summarize_seeds(grouped[name])
        means.append(summary["mean"])
        stds.append(summary["std"])
        rows.append({"experiment_name": name, **summary})
        stats_payload["experiments"][name] = summary

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

    title = args.title or f"Figure A6 Ablation Summary ({args.region} nodes)"
    plot_ablation_summary(
        experiment_names=experiment_names,
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
