"""单次实验分析：生成精度对比柱图 + manifest，供 analysis_compare 目录归档。"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.field_plot_paths import CAT_ANALYSIS_COMPARE, category_dir, plots_root

# 默认对比指标（按 checkpoint 类型可删减）
_METRICS_BEST_MODEL = [
    ("r2_p", "R² p"),
    ("r2_vel_mag", "R² |v|"),
    ("wss_r2_wss", "R² WSS"),
    ("rmse_p", "RMSE p"),
]
_METRICS_BEST_WSS = [
    ("wss_r2_wss", "R² WSS (primary)"),
    ("wss_rmse_wss", "RMSE WSS"),
    ("r2_p", "R² p"),
    ("r2_vel_mag", "R² |v|"),
]


def _load_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 summary.json: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _test_metrics(summary: Dict[str, Any], checkpoint: str) -> Dict[str, Any]:
    if checkpoint == "best_wss":
        block = summary.get("test_metrics_best_wss")
        if isinstance(block, dict) and block:
            return block
    block = summary.get("test_metrics")
    return block if isinstance(block, dict) else {}


def _exp_label(run_dir: Path, summary: Dict[str, Any]) -> str:
    snap = run_dir / "config.snapshot.json"
    if snap.exists():
        with snap.open(encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            meta = cfg.get("meta") or {}
            if isinstance(meta, dict) and meta.get("exp_id"):
                return str(meta["exp_id"])
            if cfg.get("experiment_name"):
                return str(cfg["experiment_name"])
    return run_dir.name[:48]


def _pick_metrics(checkpoint: str, primary_metric: str) -> List[Tuple[str, str]]:
    base = list(_METRICS_BEST_WSS if checkpoint == "best_wss" else _METRICS_BEST_MODEL)
    keys = [k for k, _ in base]
    if primary_metric and primary_metric not in keys:
        base.insert(0, (primary_metric, primary_metric))
    seen = set()
    out: List[Tuple[str, str]] = []
    for k, lbl in base:
        if k not in seen:
            seen.add(k)
            out.append((k, lbl))
    return out


def _metric_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    v = metrics.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _plot_metrics_bar(
    rows: List[Dict[str, Any]],
    save_path: Path,
    title: str,
    primary_metric: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "需要 matplotlib：pip install -r training/requirements.txt"
        ) from exc

    metric_keys = []
    for row in rows:
        for m in row["metrics"]:
            if m["key"] not in metric_keys:
                metric_keys.append(m["key"])

    labels = [r["label"] for r in rows]
    n_metrics = len(metric_keys)
    n_groups = len(labels)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_groups, 1)

    fig, ax = plt.subplots(figsize=(max(6, n_metrics * 1.4), 5))
    colors = ["#3182bd", "#e6550d", "#31a354", "#756bb1"]

    for gi, row in enumerate(rows):
        vals = []
        for key in metric_keys:
            found = next((m["value"] for m in row["metrics"] if m["key"] == key), None)
            vals.append(found if found is not None else 0.0)
        offset = (gi - (n_groups - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=row["label"], color=colors[gi % len(colors)], alpha=0.88)
        for bar, key in zip(bars, metric_keys):
            if key == primary_metric:
                bar.set_edgecolor("#000000")
                bar.set_linewidth(1.5)

    ax.set_xticks(x)
    short_names = []
    for key in metric_keys:
        short_names.append(
            next((m["label"] for r in rows for m in r["metrics"] if m["key"] == key), key)
        )
    ax.set_xticklabels(short_names, rotation=15, ha="right")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_compare_bundle(
    *,
    run_dir: Path,
    baseline_run_dir: Optional[Path],
    runs_root: Path,
    slug: str,
    primary_metric: str,
    checkpoint: str,
    extra_run_dirs: List[Path],
) -> Path:
    metric_specs = _pick_metrics(checkpoint, primary_metric)
    out_dir = category_dir(runs_root, CAT_ANALYSIS_COMPARE) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs_ordered: List[Tuple[str, Path, str]] = []
    if baseline_run_dir is not None:
        run_dirs_ordered.append(("baseline", baseline_run_dir.resolve(), _exp_label(baseline_run_dir, _load_summary(baseline_run_dir))))
    run_dirs_ordered.append(("current", run_dir.resolve(), _exp_label(run_dir, _load_summary(run_dir))))
    for extra in extra_run_dirs:
        run_dirs_ordered.append(("extra", extra.resolve(), _exp_label(extra, _load_summary(extra))))

    rows: List[Dict[str, Any]] = []
    manifest_runs: List[Dict[str, Any]] = []

    for role, rd, label in run_dirs_ordered:
        summary = _load_summary(rd)
        tm = _test_metrics(summary, checkpoint)
        metrics_row = []
        for key, disp in metric_specs:
            val = _metric_value(tm, key)
            if val is not None:
                metrics_row.append({"key": key, "label": disp, "value": val})
        rows.append({"role": role, "label": label, "metrics": metrics_row})
        manifest_runs.append(
            {
                "role": role,
                "label": label,
                "run_dir": str(rd),
                "primary_metric": primary_metric,
                "primary_value": _metric_value(tm, primary_metric),
                "checkpoint": checkpoint,
                "best_epoch": summary.get("best_epoch"),
            }
        )

    current_primary = next((r["primary_value"] for r in manifest_runs if r["role"] == "current"), None)
    baseline_primary = next((r["primary_value"] for r in manifest_runs if r["role"] == "baseline"), None)
    delta = None
    if current_primary is not None and baseline_primary is not None:
        delta = current_primary - baseline_primary

    manifest = {
        "analysis_date": date.today().isoformat(),
        "slug": slug,
        "primary_metric": primary_metric,
        "checkpoint": checkpoint,
        "delta_vs_baseline": delta,
        "runs": manifest_runs,
        "figures": {
            "metrics_compare_png": str(out_dir / "metrics_compare.png"),
            "metrics_compare_csv": str(out_dir / "metrics_compare.csv"),
        },
    }

    title = f"Test metrics ({checkpoint})"
    if delta is not None:
        title += f" · Δ{primary_metric}={delta:+.4f}"

    _plot_metrics_bar(rows, out_dir / "metrics_compare.png", title, primary_metric)

    csv_path = out_dir / "metrics_compare.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("role,label,metric_key,metric_label,value\n")
        for row in rows:
            for m in row["metrics"]:
                f.write(
                    f"{row['role']},{row['label']},{m['key']},{m['label']},{m['value']}\n"
                )

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"已写入: {out_dir / 'metrics_compare.png'}")
    print(f"已写入: {out_dir / 'manifest.json'}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="实验分析精度对比图（analysis_compare）")
    parser.add_argument("--run-dir", required=True, help="当前实验 run 目录")
    parser.add_argument("--baseline-run-dir", default="", help="对照基线 run 目录")
    parser.add_argument("--extra-run-dir", action="append", default=[], help="额外对比 run，可重复")
    parser.add_argument("--runs-root", default="outputs/field")
    parser.add_argument(
        "--slug",
        default="",
        help="子目录名；默认 YYYYMMDD_<current>_vs_<baseline>",
    )
    parser.add_argument(
        "--primary-metric",
        default="wss_r2_wss",
        help="主指标字段名（summary test_metrics）",
    )
    parser.add_argument(
        "--checkpoint",
        choices=("best_model", "best_wss"),
        default="best_wss",
        help="读取 test_metrics 或 test_metrics_best_wss",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    baseline = Path(args.baseline_run_dir).resolve() if args.baseline_run_dir else None
    extras = [Path(p).resolve() for p in args.extra_run_dir]
    runs_root = Path(args.runs_root).resolve()

    cur_label = _exp_label(run_dir, _load_summary(run_dir))[:32]
    if args.slug:
        slug = args.slug
    elif baseline is not None:
        base_label = _exp_label(baseline, _load_summary(baseline))[:32]
        slug = f"{date.today().strftime('%Y%m%d')}_{cur_label}_vs_{base_label}"
    else:
        slug = f"{date.today().strftime('%Y%m%d')}_{cur_label}"

    out_dir = build_compare_bundle(
        run_dir=run_dir,
        baseline_run_dir=baseline,
        runs_root=runs_root,
        slug=slug,
        primary_metric=args.primary_metric,
        checkpoint=args.checkpoint,
        extra_run_dirs=extras,
    )
    print(f"对比图目录: {out_dir}")
    print(
        "建议继续生成损失曲线:\n"
        f"  python -m training.scripts.plot_training_history "
        f"--run-dir {run_dir}"
        + (f" --run-dir {baseline}" if baseline else "")
        + f" --output-dir {out_dir} --compare-metric val_loss"
    )


if __name__ == "__main__":
    main()
