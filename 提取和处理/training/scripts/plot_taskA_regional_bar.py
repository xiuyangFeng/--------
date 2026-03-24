from __future__ import annotations

import argparse
from pathlib import Path

from ..analysis.visualization import plot_regional_bar
from ._figure_utils import ensure_dir, load_manifest, load_prediction_payload, save_json


def compute_aggregate_regional_metrics(items):
    torch = __import__("torch")
    from ..analysis.regional_eval import (
        compute_regional_metrics,
        load_node_features_for_region_masks,
    )

    preds = []
    targets = []
    features = []
    for item in items:
        payload = load_prediction_payload(Path(str(item["prediction_path"])).resolve())
        preds.append(payload["y_pred"].detach().cpu())
        targets.append(payload["y_true"].detach().cpu())
        features.append(load_node_features_for_region_masks(payload))

    if not preds:
        raise ValueError("manifest 中没有可用预测项")

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    x = torch.cat(features, dim=0)
    return compute_regional_metrics(pred, target, x)


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 predictions manifest 生成任务A分区域误差图")
    parser.add_argument("--manifest", required=True, help="predict_field.py 导出的 manifest.json 路径")
    parser.add_argument("--output-dir", default="", help="输出目录，默认 manifest 同级 regional_eval")
    parser.add_argument(
        "--metric-key",
        action="append",
        default=[],
        help="要绘制的区域指标，可重复传入；默认 rmse_vel_mag 和 rmse_p",
    )
    parser.add_argument("--title-prefix", default="Figure A5 Regional Error", help="图标题前缀")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    items = manifest.get("items", [])
    if not isinstance(items, list):
        raise SystemExit("manifest.json 缺少合法的 items 列表")

    output_dir = ensure_dir(Path(args.output_dir).resolve()) if args.output_dir else ensure_dir(manifest_path.parent / "regional_eval")
    metric_keys = args.metric_key or ["rmse_vel_mag", "rmse_p"]

    try:
        regional_metrics = compute_aggregate_regional_metrics(items)
        for metric_key in metric_keys:
            plot_regional_bar(
                regional_metrics,
                metric_key=metric_key,
                save_path=output_dir / f"fig_A5_regional_bar_{metric_key}.png",
                title=f"{args.title_prefix}: {metric_key}",
            )
    except ImportError as exc:
        msg = str(exc)
        if "torch" in msg:
            raise SystemExit(
                "读取预测结果失败：当前环境缺少 torch。请先安装训练依赖，例如 `pip install -r training/requirements.txt`。"
            ) from exc
        raise SystemExit(
            "生成图像失败：当前环境缺少 matplotlib。请先安装可视化依赖，例如 `pip install -r training/requirements.txt`。"
        ) from exc

    save_json(output_dir / "fig_A5_regional_metrics.json", regional_metrics)
    for metric_key in metric_keys:
        print(output_dir / f"fig_A5_regional_bar_{metric_key}.png")
    print(output_dir / "fig_A5_regional_metrics.json")


if __name__ == "__main__":
    main()
