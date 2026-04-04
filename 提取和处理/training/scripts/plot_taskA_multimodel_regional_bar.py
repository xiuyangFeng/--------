"""跨模型分区域误差对比图脚本（Fig A5 多模型对比版）

从各 run 目录的 ``predictions_test/regional_eval/fig_A5_regional_metrics.json``
读取区域指标，按 ``exp_id`` 分组，跨 seed 对每个区域取均值，生成多模型 grouped bar chart。

本脚本不依赖 GPU / torch，可在本地直接运行。

用法示例
--------
python -m training.scripts.plot_taskA_multimodel_regional_bar \\
    --runs-root outputs/field

# 只对比 Transformer 无几何 vs 有几何（聚焦几何贡献）
python -m training.scripts.plot_taskA_multimodel_regional_bar \\
    --runs-root outputs/field \\
    --exp-filter A-Base-03 --exp-filter A-Main-01 \\
    --metric-key rmse_vel_mag
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..core.field_plot_paths import CAT_MULTIMODEL_BASELINE, category_dir

_EXP_LABELS: Dict[str, str] = {
    "A-Base-01": "MLP",
    "A-Base-02": "GraphSAGE",
    "A-Base-03": "Transformer",
    "A-Main-01": "Transformer+Geom",
    "A-Opt-01": "Transformer+Geom (tw22205)",
    "A-Opt-02": "Transformer+Geom (Pre-Norm)",
    "A-Opt-02_warmup": "Transformer+Geom (Pre-Norm+Warmup5)",
    "A-Opt-03": "Transformer+Geom (P0-4 h128)",
    "A-Opt-05": "Transformer+Geom (Opt-05 256×4L)",
    "A-Opt-07": "Transformer+Geom (Opt-07 interior boost×3)",
    "A-Abl-02-01": "Opt-05 w/o Abscissa",
    "A-Abl-02-02": "Opt-05 w/o NormRadius",
    "A-Abl-02-03": "Opt-05 w/o Curvature",
    "A-Abl-02-04": "Opt-05 w/o Tangent",
}
_EXP_ORDER: List[str] = [
    "A-Base-01",
    "A-Base-02",
    "A-Base-03",
    "A-Main-01",
    "A-Opt-01",
    "A-Opt-02",
    "A-Opt-02_warmup",
    "A-Opt-03",
    "A-Opt-05",
    "A-Abl-02-01",
    "A-Abl-02-02",
    "A-Abl-02-03",
    "A-Abl-02-04",
    "A-Opt-07",
]


def _discover_runs(runs_root: Path) -> List[Path]:
    return sorted(
        p for p in runs_root.iterdir()
        if p.is_dir() and (p / "summary.json").exists()
    )


def _read_exp_id_and_seed(run_dir: Path) -> tuple[str, int]:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        d = json.load(f)
    exp_id = str(d.get("exp_id", run_dir.name))
    seed = int(d.get("seed", -1))
    return exp_id, seed


def _load_regional_json(json_path: Path) -> Dict[str, Dict[str, float]]:
    with json_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    result: Dict[str, Dict[str, float]] = {}
    for region, metrics in d.items():
        if not isinstance(metrics, dict):
            continue
        result[region] = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成跨模型分区域误差对比图（Fig A5 多模型版）"
    )
    parser.add_argument("--runs-root", default="outputs/field", help="run 根目录")
    parser.add_argument(
        "--metric-key",
        action="append",
        default=[],
        help="要绘制的区域指标，可重复传入；默认 rmse_vel_mag、rmse_p",
    )
    parser.add_argument(
        "--exp-filter",
        action="append",
        default=[],
        help="只包含指定 exp_id，可重复传入；默认包含全部",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        help="只显示指定区域，可重复传入；默认显示全部",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="输出目录，默认 <runs-root>/plots/multimodel_baseline",
    )
    parser.add_argument("--title-prefix", default="Figure A5 Regional Error", help="图标题前缀")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="若指定则只纳入 summary.json 中该 seed 的 run（用于与单 seed 消融公平对比）",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    metric_keys = args.metric_key or ["rmse_vel_mag", "rmse_p"]
    exp_filter: Optional[List[str]] = args.exp_filter or None
    region_filter: Optional[List[str]] = args.region or None
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else category_dir(runs_root, CAT_MULTIMODEL_BASELINE)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集：exp_id → {region → {metric → [values across seeds]}}
    raw: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for run_dir in _discover_runs(runs_root):
        json_path = run_dir / "predictions_test" / "regional_eval" / "fig_A5_regional_metrics.json"
        if not json_path.exists():
            continue
        exp_id, run_seed = _read_exp_id_and_seed(run_dir)
        if args.seed is not None and run_seed != args.seed:
            continue
        if exp_filter and exp_id not in exp_filter:
            continue
        regional = _load_regional_json(json_path)
        for region, metrics in regional.items():
            for metric_key, value in metrics.items():
                raw[exp_id][region][metric_key].append(value)

    if not raw:
        raise SystemExit(
            f"在 {runs_root} 下未找到任何包含 fig_A5_regional_metrics.json 的 run 目录"
        )

    # 跨 seed 取均值
    models_regional: Dict[str, Dict[str, Dict[str, float]]] = {}
    exp_ids_sorted = [e for e in _EXP_ORDER if e in raw] + sorted(
        e for e in raw if e not in _EXP_ORDER
    )
    for exp_id in exp_ids_sorted:
        label = _EXP_LABELS.get(exp_id, exp_id)
        models_regional[label] = {
            region: {k: float(sum(vs) / len(vs)) for k, vs in mdict.items()}
            for region, mdict in raw[exp_id].items()
        }

    try:
        from ..analysis.visualization import plot_multimodel_regional_bar
    except ImportError as exc:
        raise SystemExit(
            "生成图像失败：当前环境缺少 matplotlib。请先安装可视化依赖。"
        ) from exc

    for metric_key in metric_keys:
        seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
        suffix = ("_geo_only" if exp_filter else "") + seed_tag
        output_path = output_dir / f"fig_A5_multimodel_regional_bar_{metric_key}{suffix}.png"
        plot_multimodel_regional_bar(
            models_regional,
            metric_key=metric_key,
            regions=region_filter if region_filter else None,
            save_path=output_path,
            title=f"{args.title_prefix}: {metric_key}",
        )
        print(output_path)


if __name__ == "__main__":
    main()
