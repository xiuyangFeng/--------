from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from ._figure_utils import load_json, read_regional_metrics_dict, resolve_run_dirs

_REGION_PREFIX = {
    "all": "all_",
    "interior": "",
    "wall": "wall_",
}


def extract_row(run_dir: Path, primary_region: str = "interior") -> Dict[str, object]:
    summary = load_json(run_dir / "summary.json")
    manifest = load_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    test_metrics = summary.get("test_metrics", {})
    if not isinstance(test_metrics, dict):
        raise ValueError(f"summary.json 中缺少 test_metrics: {run_dir}")

    model_info = manifest.get("model", {})
    model_name = summary.get("model", model_info.get("name", "unknown"))
    hidden_dim = model_info.get("hidden_dim", "")
    num_layers = model_info.get("num_layers", "")
    params = ""

    row: Dict[str, object] = {
        "run_dir": str(run_dir),
        "experiment_name": summary.get("experiment_name", run_dir.name),
        "study_group": summary.get("study_group", manifest.get("study_group", "")),
        "feature_set": summary.get("feature_set", manifest.get("feature_set", "")),
        "seed": summary.get("seed", manifest.get("seed", "")),
        "model": model_name,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "best_epoch": summary.get("best_epoch", manifest.get("best_epoch", "")),
        "best_val_loss": summary.get("best_val_loss", manifest.get("best_val_loss", "")),
    }

    pri_metrics = read_regional_metrics_dict(run_dir, primary_region)
    if pri_metrics is None and primary_region != "all":
        pri_metrics = read_regional_metrics_dict(run_dir, "all")
    if pri_metrics is None:
        pri_metrics = {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}

    for key in ("rmse_u", "rmse_v", "rmse_w", "rmse_p", "rmse_vel_mag", "r2_p"):
        row[key] = pri_metrics.get(key, test_metrics.get(key, ""))

    if primary_region != "all":
        all_metrics = read_regional_metrics_dict(run_dir, "all")
        if all_metrics is None:
            all_metrics = {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
        row["all_rmse_vel_mag"] = all_metrics.get("rmse_vel_mag", test_metrics.get("rmse_vel_mag", ""))

    row.update({
        "physics_enabled": summary.get("physics_enabled", manifest.get("physics", {}).get("enabled", "")),
        "num_test_graphs": summary.get("num_test_graphs", manifest.get("dataset_sizes", {}).get("num_test_graphs", "")),
        "params": params,
        "metric_scope": primary_region,
    })
    return row


def write_markdown(rows: List[Dict[str, object]], save_path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="聚合多个 run 的主结果表")
    parser.add_argument("--runs-root", default="outputs/field", help="run 根目录")
    parser.add_argument("--pattern", action="append", default=[], help="相对 runs-root 的匹配模式，可重复传入；默认 */summary.json")
    parser.add_argument("--run-dir", action="append", default=[], help="显式指定 run 目录，可重复传入")
    parser.add_argument(
        "--region", default="interior",
        choices=["all", "interior", "wall"],
        help="主指标来源区域（默认 interior；同时保留 all 作为参考列）",
    )
    parser.add_argument("--output-csv", default="", help="CSV 输出路径，默认 <runs-root>/plots/fig_A1_main_table.csv")
    parser.add_argument("--output-md", default="", help="Markdown 输出路径，可选")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    patterns = args.pattern or ["*/summary.json"]
    run_dirs = resolve_run_dirs(runs_root, patterns, args.run_dir)
    run_dirs = [p for p in run_dirs if (p / "summary.json").exists()]
    if not run_dirs:
        raise SystemExit("未找到任何包含 summary.json 的 run 目录")

    rows = [extract_row(run_dir, primary_region=args.region) for run_dir in run_dirs]
    rows.sort(key=lambda row: (str(row["study_group"]), str(row["experiment_name"]), str(row["seed"])))

    default_dir = runs_root / "plots"
    default_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv).resolve() if args.output_csv else default_dir / "fig_A1_main_table.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(output_csv)

    if args.output_md:
        output_md = Path(args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(rows, output_md)
        print(output_md)


if __name__ == "__main__":
    main()
