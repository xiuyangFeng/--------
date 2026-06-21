"""跨模型 per-case 箱线图汇总脚本（Fig A4 多模型对比版）

从各 run 目录的 ``predictions_test/fig_A4_per_case_metrics.csv`` 读取 per-case 指标，
按 ``exp_id`` 分组，跨 seed 对每个 case 取均值后，生成多模型并排箱线图。

本脚本不依赖 GPU / torch，可在本地直接运行。

用法示例
--------
python -m training.scripts.plot_taskA_multimodel_per_case_boxplot \\
    --runs-root outputs/field

python -m training.scripts.plot_taskA_multimodel_per_case_boxplot \\
    --runs-root outputs/field \\
    --metric-key rmse_vel_mag --metric-key rmse_p
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..core.field_plot_paths import CAT_MULTIMODEL_BASELINE, category_dir

# exp_id → 论文显示标签，未在表中的 exp_id 直接原样显示
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
    "A-Opt-G01": "Opt-05 + G01 (bifurcation)",
    "A-Opt-G04": "Opt-05 + G04 (wall distance)",
    "A-Opt-G05": "Opt-05 + G05 (d tangent/ds)",
}

# 控制多模型在图中的显示顺序
_EXP_ORDER: List[str] = [
    "A-Base-01",
    "A-Base-02",
    "A-Base-03",
    "A-Main-01",
    "A-Opt-G01",
    "A-Opt-G04",
    "A-Opt-G05",
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


def _load_per_case_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """读取 per-case CSV，返回 {case_name: {metric: value}}。"""
    result: Dict[str, Dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_name = str(row.get("case_name", "")).strip()
            if not case_name:
                continue
            metrics: Dict[str, float] = {}
            for k, v in row.items():
                if k == "case_name":
                    continue
                try:
                    metrics[k] = float(v)
                except (ValueError, TypeError):
                    pass
            result[case_name] = metrics
    return result


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成跨模型 per-case 箱线对比图（Fig A4 多模型版）"
    )
    parser.add_argument("--runs-root", default="outputs/field", help="run 根目录")
    parser.add_argument(
        "--metric-key",
        action="append",
        default=[],
        help="箱线图指标，可重复传入；默认 rmse_vel_mag、rmse_p",
    )
    parser.add_argument(
        "--exp-filter",
        action="append",
        default=[],
        help="只包含指定 exp_id，可重复传入；默认包含全部",
    )
    parser.add_argument(
        "--region", default="interior",
        choices=["all", "interior", "wall"],
        help="节点过滤区域（默认 interior）；对应读取 fig_A4_per_case_metrics_<region>.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="输出目录，默认 <runs-root>/plots/multimodel_baseline",
    )
    parser.add_argument("--title", default="", help="图标题（为空则自动生成）")
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
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else category_dir(runs_root, CAT_MULTIMODEL_BASELINE)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    region_tag = "" if args.region == "all" else f"_{args.region}"
    csv_filename = f"fig_A4_per_case_metrics{region_tag}.csv"

    # 收集：exp_id → {case_name → {metric → [values across seeds]}}
    raw: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for run_dir in _discover_runs(runs_root):
        exp_id, run_seed = _read_exp_id_and_seed(run_dir)
        if args.seed is not None and run_seed != args.seed:
            continue
        if exp_filter and exp_id not in exp_filter:
            continue
        search_paths = [
            run_dir / "predictions_test" / csv_filename,
            run_dir / "predictions_test" / "fig_A4_per_case_metrics.csv",
        ]
        if args.region == "interior":
            search_paths.append(
                run_dir / "predictions_test" / "error_analysis_interior" / "per_case_metrics.csv"
            )
        csv_path = next((p for p in search_paths if p.exists()), None)
        if csv_path is None:
            continue
        per_case = _load_per_case_csv(csv_path)
        for case_name, metrics in per_case.items():
            for metric_key, value in metrics.items():
                raw[exp_id][case_name][metric_key].append(value)

    if not raw:
        raise SystemExit(
            f"在 {runs_root} 下未找到任何包含 fig_A4_per_case_metrics.csv 的 run 目录"
        )

    # 对每个 case 跨 seed 取均值
    models_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    exp_ids_sorted = [e for e in _EXP_ORDER if e in raw] + sorted(
        e for e in raw if e not in _EXP_ORDER
    )
    for exp_id in exp_ids_sorted:
        label = _EXP_LABELS.get(exp_id, exp_id)
        models_data[label] = {
            case_name: {k: float(sum(vs) / len(vs)) for k, vs in mdict.items()}
            for case_name, mdict in raw[exp_id].items()
        }

    try:
        from ..analysis.visualization import plot_multimodel_per_case_boxplot
    except ImportError as exc:
        raise SystemExit(
            "生成图像失败：当前环境缺少 matplotlib。请先安装可视化依赖。"
        ) from exc

    title = args.title or f"Figure A4 Per-Case Comparison ({args.region} nodes, all models)"
    seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
    suffix_f = ("_exp_subset" if exp_filter else "") + seed_tag
    output_path = output_dir / f"fig_A4_multimodel_per_case_boxplot{region_tag}{suffix_f}.png"
    plot_multimodel_per_case_boxplot(
        models_data,
        metric_keys=metric_keys,
        save_path=output_path,
        title=title,
    )
    print(output_path)


if __name__ == "__main__":
    main()
