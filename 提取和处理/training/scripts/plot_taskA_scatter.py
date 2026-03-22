from __future__ import annotations

import argparse
from pathlib import Path

from ..analysis.visualization import scatter_pred_vs_true
from ._figure_utils import aggregate_predictions, ensure_dir, load_manifest, maybe_subsample


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 predictions manifest 生成任务A散点图")
    parser.add_argument("--manifest", required=True, help="predict_field.py 导出的 manifest.json 路径")
    parser.add_argument("--output", default="", help="输出图片路径，默认 manifest 同级 fig_A3_scatter.png")
    parser.add_argument("--max-points", type=int, default=200000, help="散点图最多使用多少个节点")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--title", default="Figure A3 Scatter", help="图标题")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    items = manifest.get("items", [])
    if not isinstance(items, list):
        raise SystemExit("manifest.json 缺少合法的 items 列表")

    y_true_all, y_pred_all, _per_case = aggregate_predictions(items)
    y_true_plot, y_pred_plot = maybe_subsample(y_true_all, y_pred_all, args.max_points, args.seed)

    output_path = Path(args.output).resolve() if args.output else ensure_dir(manifest_path.parent) / "fig_A3_scatter.png"
    try:
        scatter_pred_vs_true(
            y_pred_plot,
            y_true_plot,
            save_path=output_path,
            title=args.title,
        )
    except ImportError as exc:
        raise SystemExit(
            "生成图像失败：当前环境缺少 matplotlib。请先安装可视化依赖，例如 `pip install -r training/requirements.txt`。"
        ) from exc

    print(output_path)


if __name__ == "__main__":
    main()
