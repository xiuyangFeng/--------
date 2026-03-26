"""跨模型散点对比图脚本（Fig A3 多模型对比版）

从各 run 目录的 ``predictions_test/manifest.json`` 加载预测结果（.pt 文件），
生成多模型并排散点/hexbin 图，用于直观比较各模型的拟合质量。

⚠️  本脚本依赖 torch 和预测 .pt 文件，需在集群上运行。

用法示例
--------
# 默认：对 runs-root 下所有 seed=1 的 run，绘制 vel_mag 散点对比图
python -m training.scripts.plot_taskA_multimodel_scatter \\
    --runs-root outputs/field \\
    --seed-filter 1 \\
    --variable vel_mag \\
    --output-dir outputs/field/plots

# 绘制压力场（p）的对比图
python -m training.scripts.plot_taskA_multimodel_scatter \\
    --runs-root outputs/field \\
    --seed-filter 1 \\
    --variable p \\
    --output-dir outputs/field/plots

# 只对比 Transformer 无几何 vs 有几何
python -m training.scripts.plot_taskA_multimodel_scatter \\
    --runs-root outputs/field \\
    --seed-filter 1 \\
    --exp-filter A-Base-03 --exp-filter A-Main-01 \\
    --variable vel_mag \\
    --output-dir outputs/field/plots
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_EXP_LABELS: Dict[str, str] = {
    "A-Base-01": "MLP",
    "A-Base-02": "GraphSAGE",
    "A-Base-03": "Transformer",
    "A-Main-01": "Transformer+Geom",
}
_EXP_ORDER: List[str] = ["A-Base-01", "A-Base-02", "A-Base-03", "A-Main-01"]
_VALID_VARIABLES = {"u", "v", "w", "p", "vel_mag"}


def _discover_runs(runs_root: Path) -> List[Path]:
    return sorted(
        p for p in runs_root.iterdir()
        if p.is_dir() and (p / "summary.json").exists()
    )


def _read_summary(run_dir: Path) -> Dict[str, object]:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_manifest(manifest_path: Path) -> Dict[str, object]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate_from_manifest(
    manifest_path: Path,
    region: str = "all",
) -> tuple[np.ndarray, np.ndarray]:
    """从 manifest 中聚合所有 .pt 文件，返回 (y_true, y_pred)，shape (N, 4)。

    当 ``region`` 不为 ``"all"`` 时，仅保留对应区域的节点。
    """
    from ._figure_utils import _resolve_wall_mask_from_payload, build_region_mask

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "读取预测结果失败：当前环境缺少 torch。请在集群环境中运行。"
        ) from exc

    manifest = _load_manifest(manifest_path)
    items = manifest.get("items", [])
    if not items:
        raise ValueError(f"manifest 没有 items：{manifest_path}")

    y_trues: List[np.ndarray] = []
    y_preds: List[np.ndarray] = []
    for item in items:
        pred_path = Path(str(item["prediction_path"]))
        payload = torch.load(pred_path, map_location="cpu")
        yt = payload["y_true"].detach().cpu().numpy()
        yp = payload["y_pred"].detach().cpu().numpy()
        if region != "all":
            wall_mask = _resolve_wall_mask_from_payload(payload)
            node_mask = build_region_mask(wall_mask, region)
            yt = yt[node_mask]
            yp = yp[node_mask]
        y_trues.append(yt)
        y_preds.append(yp)

    return np.concatenate(y_trues, axis=0), np.concatenate(y_preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成跨模型散点对比图（Fig A3 多模型版，需集群运行）"
    )
    parser.add_argument("--runs-root", default="outputs/field", help="run 根目录")
    parser.add_argument(
        "--seed-filter", type=int, default=1,
        help="只使用该 seed 的 run（避免多 seed 叠加导致图太密），默认 1",
    )
    parser.add_argument(
        "--variable", default="vel_mag",
        choices=list(_VALID_VARIABLES),
        help="绘制的目标变量：u / v / w / p / vel_mag（默认 vel_mag）",
    )
    parser.add_argument(
        "--exp-filter",
        action="append", default=[],
        help="只包含指定 exp_id，可重复传入；默认全部",
    )
    parser.add_argument(
        "--max-points", type=int, default=100_000,
        help="每个模型最多采样的节点数（散点图减速），默认 100000",
    )
    parser.add_argument(
        "--region", default="interior",
        choices=["all", "interior", "wall"],
        help="节点过滤区域（默认 interior，仅内部节点）",
    )
    parser.add_argument("--output-dir", default="", help="输出目录，默认 <runs-root>/plots")
    parser.add_argument("--title", default="", help="图标题（为空则自动生成）")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    exp_filter: Optional[List[str]] = args.exp_filter or None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (runs_root / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按 exp_id 收集符合 seed-filter 的 manifest 路径
    candidates: Dict[str, Path] = {}
    for run_dir in _discover_runs(runs_root):
        summary = _read_summary(run_dir)
        exp_id = str(summary.get("exp_id", run_dir.name))
        seed = int(summary.get("seed", 0))
        if seed != args.seed_filter:
            continue
        if exp_filter and exp_id not in exp_filter:
            continue
        manifest_path = run_dir / "predictions_test" / "manifest.json"
        if not manifest_path.exists():
            continue
        candidates[exp_id] = manifest_path

    if not candidates:
        raise SystemExit(
            f"在 {runs_root} 下未找到 seed={args.seed_filter} 的有效 run"
        )

    exp_ids_sorted = [e for e in _EXP_ORDER if e in candidates] + sorted(
        e for e in candidates if e not in _EXP_ORDER
    )

    # 加载预测数据
    models_predictions: Dict[str, tuple] = {}
    for exp_id in exp_ids_sorted:
        label = _EXP_LABELS.get(exp_id, exp_id)
        print(f"  加载 {exp_id} ({label}) ...")
        y_true, y_pred = _aggregate_from_manifest(candidates[exp_id], region=args.region)
        models_predictions[label] = (y_pred, y_true)

    try:
        from ..analysis.visualization import plot_multimodel_scatter
    except ImportError as exc:
        raise SystemExit(
            "生成图像失败：当前环境缺少 matplotlib。请先安装可视化依赖。"
        ) from exc

    region_tag = "" if args.region == "all" else f"_{args.region}"
    suffix = f"_{args.variable}{region_tag}"
    if exp_filter:
        suffix += "_geo_only"
    title = args.title or f"Figure A3 Scatter Comparison ({args.region} nodes)"
    output_path = output_dir / f"fig_A3_multimodel_scatter{suffix}.png"
    plot_multimodel_scatter(
        models_predictions,
        variable=args.variable,
        max_points=args.max_points,
        save_path=output_path,
        title=title,
    )
    print(output_path)


if __name__ == "__main__":
    main()
