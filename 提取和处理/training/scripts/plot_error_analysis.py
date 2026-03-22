from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

from ..analysis.visualization import (
    plot_error_cdf,
    plot_error_distribution,
    plot_per_case_boxplot,
    scatter_pred_vs_true,
)
from ._figure_utils import (
    aggregate_predictions,
    compute_case_metrics,
    ensure_dir,
    load_manifest,
    maybe_subsample,
    save_json,
)


def save_per_case_csv(per_case_metrics: Dict[str, Dict[str, float]], save_path: Path) -> None:
    rows = []
    for case_name, metrics in sorted(per_case_metrics.items()):
        row = {"case_name": case_name}
        row.update(metrics)
        rows.append(row)
    if not rows:
        return
    with save_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 predictions manifest 生成误差分析图")
    parser.add_argument(
        "--manifest",
        required=True,
        help="predict_field.py 导出的 manifest.json 路径",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="图和汇总表输出目录，默认保存到 manifest 同级 error_analysis",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=200000,
        help="散点图最多使用多少个节点，默认 200000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="散点抽样随机种子，默认 42",
    )
    parser.add_argument(
        "--title",
        default="Prediction Error Analysis",
        help="图标题前缀",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    items = manifest.get("items", [])
    if not isinstance(items, list):
        raise SystemExit("manifest.json 缺少合法的 items 列表")

    output_dir = (
        ensure_dir(args.output_dir)
        if args.output_dir
        else ensure_dir(manifest_path.parent / "error_analysis")
    )

    y_true_all, y_pred_all, per_case_metrics = aggregate_predictions(items)
    scatter_true, scatter_pred = maybe_subsample(
        y_true_all,
        y_pred_all,
        max_points=args.max_scatter_points,
        seed=args.seed,
    )

    try:
        scatter_pred_vs_true(
            scatter_pred,
            scatter_true,
            save_path=output_dir / "fig_scatter_pred_vs_true.png",
            title=f"{args.title}: Scatter",
        )
        plot_error_distribution(
            y_pred_all,
            y_true_all,
            save_path=output_dir / "fig_error_distribution.png",
            title=f"{args.title}: Error Distribution",
        )
        plot_error_cdf(
            y_pred_all,
            y_true_all,
            save_path=output_dir / "fig_error_cdf.png",
            title=f"{args.title}: Error CDF",
        )
        plot_per_case_boxplot(
            per_case_metrics,
            save_path=output_dir / "fig_per_case_boxplot.png",
            title=f"{args.title}: Per-Case Metrics",
        )
    except ImportError as exc:
        raise SystemExit(
            "生成图像失败：当前环境缺少 matplotlib。请先安装可视化依赖，例如 `pip install -r training/requirements.txt`。"
        ) from exc

    save_per_case_csv(per_case_metrics, output_dir / "per_case_metrics.csv")

    overall_metrics = compute_case_metrics(y_true_all, y_pred_all)
    save_json(
        output_dir / "summary.json",
        {
            "manifest_path": str(manifest_path),
            "num_cases": len(per_case_metrics),
            "num_nodes_total": int(y_true_all.shape[0]),
            "scatter_nodes_used": int(scatter_true.shape[0]),
            "overall_metrics": overall_metrics,
        },
    )

    print(f"已生成误差分析目录: {output_dir}")
    print(output_dir / "fig_scatter_pred_vs_true.png")
    print(output_dir / "fig_error_distribution.png")
    print(output_dir / "fig_error_cdf.png")
    print(output_dir / "fig_per_case_boxplot.png")
    print(output_dir / "per_case_metrics.csv")
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
