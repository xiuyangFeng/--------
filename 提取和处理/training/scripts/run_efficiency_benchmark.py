"""任务 A：在已训练 run 目录上批量测量推理效率并导出图表。

默认对每个基线实验的 **全部 seed（1/2/3）** 各测一次，汇总 **mean ± std**，
并输出单 seed 对比图与汇总图。

用法示例：

  cd GNN
  conda run -n GNN python -m training.scripts.run_efficiency_benchmark \\
      --output-dir outputs/field/plots/efficiency

依赖：各 run 的 ``config.snapshot.json``、``best_model.pt``、``summary.json``。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from ..analysis.benchmark import benchmark_full_case, build_efficiency_table
from ..core.field_plot_paths import CAT_EFFICIENCY
from ..analysis.visualization import (
    plot_efficiency_bars,
    plot_efficiency_bars_per_seed,
    plot_pareto_accuracy_latency,
    plot_pareto_per_seed_points,
)
from ..core.config import ExperimentConfig
from ..core.data import FieldGraphDataset, build_required_data_keys, build_feature_mask
from ..core.io import load_checkpoint
from ..core.models import build_model
from ..core.splits import SplitSpec
from ..core.denylist import resolve_split_subset
from ..core.utils import dump_json, ensure_dir, resolve_device, set_seed


def resolve_cases(split: SplitSpec, subset: str, data_root: str):
    return resolve_split_subset(split, subset, data_root)


def _case_name_for_path(dataset: FieldGraphDataset, data_path: Path) -> str:
    case_path = data_path
    for _ in range(dataset._subdir_depth + 1):
        case_path = case_path.parent
    return str(case_path.relative_to(dataset.root))


def collect_graphs_first_case(dataset: FieldGraphDataset) -> Tuple[str, List[object]]:
    """取测试集中按文件序第一个病例的全时序图列表。"""
    if not dataset.data_files:
        raise ValueError("数据集中没有图文件")
    target_case = _case_name_for_path(dataset, dataset.data_files[0])
    graphs = []
    for i, p in enumerate(dataset.data_files):
        if _case_name_for_path(dataset, p) == target_case:
            graphs.append(dataset[i])
    return target_case, graphs


def _seed_from_run_dir(p: Path) -> int:
    m = re.search(r"seed(\d+)", p.name)
    return int(m.group(1)) if m else 0


def _short_label(config: ExperimentConfig) -> str:
    m = config.model.name
    fs = config.meta.feature_set
    if m == "mlp":
        return "MLP"
    if m == "graphsage":
        return "GraphSAGE"
    if m == "transformer" and fs and "geom" in fs:
        return "Transformer+geom"
    if m == "transformer":
        return "Transformer"
    return m


def discover_experiment_groups(output_root: Path) -> List[List[Path]]:
    """四个基线实验各自目录下所有 seed（有 best_model.pt）。"""
    patterns = [
        "field_mlp_coord_t_bc_split_AG_v1_seed*",
        "field_graphsage_coord_t_bc_wall_split_AG_v1_seed*",
        "field_transformer_coord_t_bc_wall_split_AG_v1_seed*",
        "field_transformer_coord_t_bc_geom_wall_split_AG_v1_seed*",
    ]
    groups: List[List[Path]] = []
    for pat in patterns:
        matches = [
            m
            for m in output_root.glob(pat)
            if m.is_dir() and (m / "best_model.pt").exists()
        ]
        matches.sort(key=lambda p: (_seed_from_run_dir(p), p.name))
        if matches:
            groups.append(matches)
    return groups


def group_flat_run_dirs(run_dirs: Sequence[Path]) -> List[List[Path]]:
    """按 ``experiment_name`` 分组并依 seed 排序。"""
    by_exp: Dict[str, List[Path]] = OrderedDict()
    for rd in run_dirs:
        cfg_path = rd / "config.snapshot.json"
        if not cfg_path.exists():
            continue
        cfg = ExperimentConfig.from_json(cfg_path)
        by_exp.setdefault(cfg.run.experiment_name, []).append(rd)
    out: List[List[Path]] = []
    for _name, paths in by_exp.items():
        paths = sorted(paths, key=lambda p: (_seed_from_run_dir(p), p.name))
        out.append(paths)
    return out


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.array([v for v in vals if v == v], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    mu = float(np.mean(arr))
    if arr.size == 1:
        return mu, 0.0
    return mu, float(np.std(arr, ddof=1))


def aggregate_group_rows(seed_rows: List[Dict[str, object]]) -> Dict[str, object]:
    """跨 seed 汇总；``label`` 用短名 + 换行 + mean±std 说明。"""
    if not seed_rows:
        raise ValueError("empty seed_rows")
    base = seed_rows[0]
    short = str(base.get("short_label", base["label"]))

    keys_agg = [
        "mean_ms",
        "std_ms",
        "peak_memory_mb",
        "rmse_vel_mag",
        "full_case_total_case_ms",
        "full_case_per_snapshot_ms",
        "full_case_peak_memory_mb",
    ]
    agg: Dict[str, object] = {
        "experiment_name": base["experiment_name"],
        "exp_id": base["exp_id"],
        "model": base["model"],
        "feature_set": base["feature_set"],
        "short_label": short,
        "label": f"{short}\n(mean±std, n={len(seed_rows)})",
        "n_seeds": len(seed_rows),
        "seeds": [int(r["seed"]) for r in seed_rows],
        "total_params": int(base["total_params"]),
        "subset": base["subset"],
        "benchmark_case": base["benchmark_case"],
    }

    for k in keys_agg:
        vals = []
        for r in seed_rows:
            v = r.get(k)
            if v is not None and v == v:
                vals.append(float(v))
        mu, sd = _mean_std(vals)
        agg[k] = mu
        agg[f"{k}_std"] = sd

    return agg


def benchmark_one_run(
    run_dir: Path,
    device: torch.device,
    subset: str,
    n_warmup: int,
    n_runs: int,
    cfd_time_hours: float | None,
) -> Dict[str, object]:
    config_path = run_dir / "config.snapshot.json"
    ckpt = run_dir / "best_model.pt"
    summary_path = run_dir / "summary.json"

    if not config_path.exists():
        raise FileNotFoundError(f"缺少配置: {config_path}")
    if not ckpt.exists():
        raise FileNotFoundError(f"缺少权重: {ckpt}")

    config = ExperimentConfig.from_json(config_path)
    config.validate()
    split = SplitSpec.from_json(config.data.split_file)

    set_seed(config.system.seed, deterministic=config.system.deterministic)

    feature_mask = build_feature_mask(
        enabled_node_features=config.data.enabled_node_features,
        enabled_global_features=config.data.enabled_global_features,
    )
    required_keys = build_required_data_keys(config.model.name, wss_dim=config.model.wss_dim)
    dataset = FieldGraphDataset(
        root=config.data.data_root,
        case_names=resolve_cases(split, subset, config.data.data_root),
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
        required_keys=required_keys,
    )

    sample_data = dataset[0]
    case_name, case_graphs = collect_graphs_first_case(dataset)

    model = build_model(
        model_name=config.model.name,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        heads=config.model.heads,
        use_transformer_prenorm=config.model.use_transformer_prenorm,
        wss_dim=config.model.wss_dim,
    ).to(device)
    load_checkpoint(model, ckpt, device)

    eff = build_efficiency_table(
        model,
        sample_data,
        device,
        cfd_time_hours=cfd_time_hours,
        n_warmup=n_warmup,
        n_runs=n_runs,
    )

    full_case = benchmark_full_case(model, case_graphs, device)

    rmse_vel_mag = None
    regional_json = run_dir / "predictions_test" / "regional_eval" / "fig_A5_regional_metrics.json"
    if regional_json.exists():
        with open(regional_json, "r", encoding="utf-8") as f:
            regional = json.load(f)
        interior_data = regional.get("interior", {})
        rmse_vel_mag = interior_data.get("rmse_vel_mag")
        if rmse_vel_mag is not None:
            rmse_vel_mag = float(rmse_vel_mag)
    if rmse_vel_mag is None and summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summ = json.load(f)
        rmse_vel_mag = float(summ.get("test_metrics", {}).get("rmse_vel_mag", float("nan")))

    label = f"{config.model.name}"
    if config.meta.feature_set:
        label = f"{config.model.name}\n({config.meta.feature_set})"

    row: Dict[str, object] = {
        "run_dir": str(run_dir.resolve()),
        "exp_id": config.meta.exp_id,
        "experiment_name": config.run.experiment_name,
        "short_label": _short_label(config),
        "label": label,
        "model": config.model.name,
        "feature_set": config.meta.feature_set,
        "seed": config.system.seed,
        "subset": subset,
        "benchmark_case": case_name,
        "n_snapshots_case": full_case["n_snapshots"],
        "rmse_vel_mag": rmse_vel_mag,
        **eff,
        **{f"full_case_{k}": v for k, v in full_case.items()},
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="任务 A 推理效率基准（多 seed + 汇总）")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"outputs/field/plots/{CAT_EFFICIENCY}",
        help="JSON 与图的输出目录（默认 plots/efficiency）",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="outputs/field",
        help="自动发现 run 时使用的根目录（默认 outputs/field）",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=None,
        help="可选：显式给出 run 目录；将按 experiment_name 分组。默认自动发现全部 seed",
    )
    parser.add_argument(
        "--subset",
        default="test",
        choices=["train", "val", "test"],
        help="用于抽样的数据子集（单图与全病例计时均来自该子集）",
    )
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument(
        "--cfd-time-hours",
        type=float,
        default=None,
        help="单次 CFD  wall-clock（小时），用于 speedup_vs_cfd；不设则省略该字段",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = ensure_dir((root / args.output_dir).resolve())
    runs_root = (root / args.runs_root).resolve()

    if args.run_dirs:
        groups = group_flat_run_dirs([Path(p).resolve() for p in args.run_dirs])
    else:
        groups = discover_experiment_groups(runs_root)

    if not groups:
        raise SystemExit(
            f"未找到任何 run 目录，请检查 {args.runs_root} 或传入 --run-dirs"
        )

    device = resolve_device("auto")
    rows_per_seed: List[Dict[str, object]] = []

    for group in groups:
        for rd in group:
            print(f"测量: {rd}")
            row = benchmark_one_run(
                rd,
                device=device,
                subset=args.subset,
                n_warmup=args.n_warmup,
                n_runs=args.n_runs,
                cfd_time_hours=args.cfd_time_hours,
            )
            rows_per_seed.append(row)

    aggregated: List[Dict[str, object]] = []
    grouped_for_plots: List[List[Dict[str, object]]] = []
    by_exp: Dict[str, List[Dict[str, object]]] = OrderedDict()
    for r in rows_per_seed:
        by_exp.setdefault(str(r["experiment_name"]), []).append(r)
    for _name in by_exp:
        seed_rows = sorted(by_exp[_name], key=lambda x: int(x["seed"]))
        grouped_for_plots.append(seed_rows)
        aggregated.append(aggregate_group_rows(seed_rows))

    payload = {
        "device": str(device),
        "n_warmup": args.n_warmup,
        "n_runs": args.n_runs,
        "cfd_time_hours": args.cfd_time_hours,
        "rows_per_seed": rows_per_seed,
        "aggregated": aggregated,
    }
    json_path = out_dir / "fig_A7_efficiency_benchmark.json"
    dump_json(payload, json_path)
    print(f"已写入: {json_path}")

    # —— 汇总 mean±std：柱图 + Pareto（误差棒） ——
    plot_efficiency_bars(
        aggregated,
        save_path=str(out_dir / "fig_A7_efficiency_bars_mean_std.png"),
        title="Task A — inference efficiency (mean ± std over seeds, single snapshot)",
    )
    print(f"已写入: {out_dir / 'fig_A7_efficiency_bars_mean_std.png'}")

    plot_pareto_accuracy_latency(
        aggregated,
        save_path=str(out_dir / "fig_A7_pareto_rmse_vel_mag_vs_latency_mean_std.png"),
        title="RMSE |v| (interior) vs latency (mean ± std over seeds)",
    )
    print(f"已写入: {out_dir / 'fig_A7_pareto_rmse_vel_mag_vs_latency_mean_std.png'}")

    # —— 分 seed：并列柱图（延迟 / 峰值显存 / 全病例每帧） ——
    if grouped_for_plots:
        plot_efficiency_bars_per_seed(
            grouped_for_plots,
            metric_key="mean_ms",
            ylabel="Time (ms)",
            save_path=str(out_dir / "fig_A7_latency_per_seed.png"),
            title="Single-snapshot latency by seed",
        )
        print(f"已写入: {out_dir / 'fig_A7_latency_per_seed.png'}")

        plot_efficiency_bars_per_seed(
            grouped_for_plots,
            metric_key="peak_memory_mb",
            ylabel="Peak GPU memory (MB)",
            save_path=str(out_dir / "fig_A7_peak_memory_per_seed.png"),
            title="Peak GPU memory (single-snapshot bench) by seed",
        )
        print(f"已写入: {out_dir / 'fig_A7_peak_memory_per_seed.png'}")

        plot_efficiency_bars_per_seed(
            grouped_for_plots,
            metric_key="full_case_peak_memory_mb",
            ylabel="Peak GPU memory (MB)",
            save_path=str(out_dir / "fig_A7_fullcase_peak_memory_per_seed.png"),
            title="Peak GPU memory (full case, all snapshots) by seed",
        )
        print(f"已写入: {out_dir / 'fig_A7_fullcase_peak_memory_per_seed.png'}")

        plot_pareto_per_seed_points(
            grouped_for_plots,
            save_path=str(out_dir / "fig_A7_pareto_per_seed_points.png"),
            title="RMSE |v| (interior) vs latency — each seed",
        )
        print(f"已写入: {out_dir / 'fig_A7_pareto_per_seed_points.png'}")

    # 兼容旧文件名：单 seed 时与汇总相同
    if len(rows_per_seed) == len(aggregated):
        plot_efficiency_bars(
            aggregated,
            save_path=str(out_dir / "fig_A7_efficiency_bars.png"),
            title="Task A — inference efficiency (single snapshot)",
        )
    else:
        # 多 seed 时仍写一版 fig_A7_efficiency_bars 为 mean±std，避免旧文档断链
        plot_efficiency_bars(
            aggregated,
            save_path=str(out_dir / "fig_A7_efficiency_bars.png"),
            title="Task A — inference efficiency (mean ± std, single snapshot)",
        )
    print(f"已写入: {out_dir / 'fig_A7_efficiency_bars.png'}")

    plot_pareto_accuracy_latency(
        aggregated,
        save_path=str(out_dir / "fig_A7_pareto_rmse_vel_mag_vs_latency.png"),
        title="RMSE |v| (interior) vs inference latency (mean ± std over seeds)",
    )
    print(f"已写入: {out_dir / 'fig_A7_pareto_rmse_vel_mag_vs_latency.png'}")


if __name__ == "__main__":
    main()
