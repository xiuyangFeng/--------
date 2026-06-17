from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..analysis.visualization import plot_multi_model_curves, plot_training_curves
from ..core.field_plot_paths import CAT_TRAINING_CURVES, category_dir

# ── Short display names for known experiment configurations ───────────────────
# Ordered longest-first so prefix matching is unambiguous.
_MODEL_SHORT: Dict[str, str] = {
    "field_transformer_coord_t_bc_geom_wall": "Transformer+Geom",
    "field_transformer_coord_t_bc_wall":      "Transformer",
    "field_graphsage_coord_t_bc_wall":        "GraphSAGE",
    "field_mlp_coord_t_bc":                   "MLP",
}


def _shorten_label(label: str) -> str:
    """Convert a full experiment label to a concise display string.

    e.g. 'field_transformer_coord_t_bc_geom_wall_seed1' → 'Transformer+Geom seed1'
    """
    for prefix, short in _MODEL_SHORT.items():
        if label.startswith(prefix):
            rest = label[len(prefix):].lstrip("_")  # e.g. 'seed1' or ''
            return f"{short} {rest}" if rest else short
    return label


def _label_to_group(short_label: str) -> str:
    """Extract the model-family group name from a short label.

    e.g. 'Transformer+Geom seed1' → 'Transformer+Geom'
    """
    parts = short_label.rsplit(" ", 1)
    return parts[0] if len(parts) == 2 else short_label


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _read_history_csv(csv_path: Path) -> Dict[str, List[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"history.csv 为空: {csv_path}")

    history: Dict[str, List[float]] = {}
    for key in reader.fieldnames or []:
        values: List[float] = []
        for row in rows:
            raw = row.get(key, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        history[key] = values
    return history


def _load_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _resolve_run_dirs(runs_root: Path, patterns: Sequence[str], run_dirs: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    seen = set()

    for run_dir in run_dirs:
        path = Path(run_dir).resolve()
        if (path / "history.csv").exists() and path not in seen:
            resolved.append(path)
            seen.add(path)

    for pattern in patterns:
        for csv_path in sorted(runs_root.glob(pattern)):
            candidate = csv_path.parent if csv_path.name == "history.csv" else csv_path
            candidate = candidate.resolve()
            if (candidate / "history.csv").exists() and candidate not in seen:
                resolved.append(candidate)
                seen.add(candidate)

    return resolved


def _pick_best_row(history: Dict[str, List[float]], metric_key: str) -> Dict[str, float]:
    metric_values = history.get(metric_key)
    epochs = history.get("epoch")
    if not metric_values or not epochs or len(metric_values) != len(epochs):
        raise ValueError(f"history 中缺少有效字段: {metric_key}")

    best_idx = min(range(len(metric_values)), key=lambda i: metric_values[i])
    best_row = {key: values[best_idx] for key, values in history.items() if len(values) == len(epochs)}
    return best_row


def _resolve_label(run_dir: Path) -> str:
    manifest = _load_json_if_exists(run_dir / "run_manifest.json")
    summary = _load_json_if_exists(run_dir / "summary.json")
    for source in (manifest, summary):
        experiment_name = source.get("experiment_name")
        seed = source.get("seed")
        if experiment_name and seed is not None:
            return f"{experiment_name}_seed{seed}"
        if experiment_name:
            return str(experiment_name)
    return run_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 history.csv 批量生成训练曲线和对比图")
    parser.add_argument(
        "--runs-root",
        default="outputs/field",
        help="训练输出根目录，默认 outputs/field",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="相对 runs-root 的匹配模式，可重复传入；默认匹配 */history.csv",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="显式指定某个 run 目录，可重复传入",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="汇总图和汇总表输出目录，默认为 <runs-root>/plots/training_curves",
    )
    parser.add_argument(
        "--compare-metric",
        default="val_loss",
        help="多实验对比图使用的指标，默认 val_loss",
    )
    parser.add_argument(
        "--best-metric",
        default="val_loss",
        help="汇总表挑选最佳 epoch 使用的指标，默认 val_loss",
    )
    parser.add_argument(
        "--single-plot-name",
        default="fig_training_curves.png",
        help="每个 run 内单独曲线图文件名",
    )
    parser.add_argument(
        "--compare-title",
        default="Training Comparison",
        help="多实验对比图标题",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    # 显式指定 run 时不要默认再 glob 整个 runs-root（大目录下极慢）。
    patterns = (
        args.pattern
        if args.pattern
        else ([] if args.run_dir else ["*/history.csv"])
    )
    run_dirs = _resolve_run_dirs(runs_root, patterns=patterns, run_dirs=args.run_dir)
    if not run_dirs:
        raise SystemExit("未找到任何包含 history.csv 的 run 目录")

    output_dir = (
        ensure_dir(args.output_dir)
        if args.output_dir
        else ensure_dir(category_dir(runs_root, CAT_TRAINING_CURVES))
    )

    compare_histories: Dict[str, Dict[str, List[float]]] = {}
    group_map: Dict[str, str] = {}
    best_rows: List[Dict[str, object]] = []

    try:
        for run_dir in run_dirs:
            history_path = run_dir / "history.csv"
            history = _read_history_csv(history_path)
            full_label = _resolve_label(run_dir)
            short_label = _shorten_label(full_label)

            # Per-run training curve uses the short label as title
            plot_training_curves(
                history,
                save_path=run_dir / args.single_plot_name,
                title=short_label,
            )
            compare_histories[short_label] = history
            group_map[short_label] = _label_to_group(short_label)

            best_row = _pick_best_row(history, args.best_metric)
            best_rows.append(
                {
                    "run_dir": str(run_dir),
                    "label": full_label,
                    "best_epoch": int(round(best_row.get("epoch", 0.0))),
                    args.best_metric: best_row.get(args.best_metric, ""),
                    "val_rmse": best_row.get("val_rmse", ""),
                    "val_rmse_vel_mag": best_row.get("val_rmse_vel_mag", ""),
                    "val_rmse_p": best_row.get("val_rmse_p", ""),
                    "lr": best_row.get("lr", ""),
                }
            )

        if len(compare_histories) >= 2:
            compare_path = output_dir / f"compare_{args.compare_metric}.png"
            plot_multi_model_curves(
                compare_histories,
                metric_key=args.compare_metric,
                save_path=compare_path,
                title=args.compare_title,
                group_map=group_map,
            )
            print(f"已生成多实验对比图: {compare_path}")
    except ImportError as exc:
        raise SystemExit(
            "生成图像失败：当前环境缺少 matplotlib。请先安装可视化依赖，例如 `pip install -r training/requirements.txt`。"
        ) from exc

    summary_path = output_dir / "best_metrics.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_dir",
                "label",
                "best_epoch",
                args.best_metric,
                "val_rmse",
                "val_rmse_vel_mag",
                "val_rmse_p",
                "lr",
            ],
        )
        writer.writeheader()
        writer.writerows(best_rows)
    print(f"已生成最佳指标汇总表: {summary_path}")

    for run_dir in run_dirs:
        print(f"已生成单次训练曲线: {run_dir / args.single_plot_name}")


if __name__ == "__main__":
    main()
