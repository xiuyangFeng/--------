from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..analysis.visualization import plot_per_case_boxplot
from ._figure_utils import aggregate_predictions, ensure_dir, load_manifest


def save_metrics_csv(save_path: Path, per_case_metrics: dict) -> None:
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
    parser = argparse.ArgumentParser(description="根据 predictions manifest 生成任务A per-case 箱线图")
    parser.add_argument("--manifest", required=True, help="predict_field.py 导出的 manifest.json 路径")
    parser.add_argument("--output", default="", help="输出图片路径，默认 manifest 同级 fig_A4_per_case_boxplot.png")
    parser.add_argument("--metrics-output", default="", help="per-case 指标 CSV 输出路径，默认 manifest 同级 fig_A4_per_case_metrics.csv")
    parser.add_argument(
        "--metric-key",
        action="append",
        default=[],
        help="箱线图指标，可重复传入；默认 rmse_vel_mag、rmse_p、rmse",
    )
    parser.add_argument("--title", default="Figure A4 Per-Case Boxplot", help="图标题")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    items = manifest.get("items", [])
    if not isinstance(items, list):
        raise SystemExit("manifest.json 缺少合法的 items 列表")

    _y_true_all, _y_pred_all, per_case_metrics = aggregate_predictions(items)
    output_path = Path(args.output).resolve() if args.output else ensure_dir(manifest_path.parent) / "fig_A4_per_case_boxplot.png"
    metrics_output = (
        Path(args.metrics_output).resolve()
        if args.metrics_output
        else ensure_dir(manifest_path.parent) / "fig_A4_per_case_metrics.csv"
    )
    metric_keys = args.metric_key or ["rmse_vel_mag", "rmse_p", "rmse"]

    try:
        plot_per_case_boxplot(
            per_case_metrics,
            metric_keys=metric_keys,
            save_path=output_path,
            title=args.title,
        )
    except ImportError as exc:
        raise SystemExit(
            "生成图像失败：当前环境缺少 matplotlib。请先安装可视化依赖，例如 `pip install -r training/requirements.txt`。"
        ) from exc

    save_metrics_csv(metrics_output, per_case_metrics)
    print(output_path)
    print(metrics_output)


if __name__ == "__main__":
    main()
