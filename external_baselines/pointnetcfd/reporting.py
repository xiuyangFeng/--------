from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = PACKAGE_ROOT / "experiments"

LATEST_RUN_START = "<!-- pointnetcfd:latest-run:start -->"
LATEST_RUN_END = "<!-- pointnetcfd:latest-run:end -->"


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def render_run_snapshot(manifest: Mapping[str, Any], run_dir: Path) -> str:
    exp = manifest["experiment_name"]
    split = manifest["split_version"]
    seed = manifest["seed"]
    target_mode = manifest["target_mode"]
    outputs = list(manifest["output_names"])
    tm = manifest["test_metrics"]
    best_ep = manifest.get("best_epoch", "—")
    best_val = manifest.get("best_val_loss")
    run_rel = run_dir.as_posix()

    lines = [
        "# PointNetCFD · 单次 run 分析报告",
        "",
        f"- **实验族**：`{exp}`",
        f"- **run_dir**：`{run_rel}`",
        f"- **split**：`{split}` · seed **{seed}**",
        f"- **target_mode**：`{target_mode}` · 输出 `{', '.join(outputs)}`",
        f"- **best_epoch**：{best_ep}"
        + (f" · best_val_loss={_fmt(float(best_val))}" if best_val is not None else ""),
        "",
        "## 指标口径",
        "",
        "目标场为 pipeline **z-score 归一化**值；MSE / RMSE / MAE / R² 均在归一化空间。",
        "",
        "## Test 集整体指标",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
        f"| loss_mse | {_fmt(float(tm['loss_mse']))} |",
        f"| RMSE | {_fmt(float(tm['rmse']))} |",
        f"| MAE | {_fmt(float(tm['mae']))} |",
        f"| n_points | {int(tm.get('n_points', 0))} |",
    ]
    for name in outputs:
        for suffix in ("mae", "rmse", "r2"):
            key = f"{name}_{suffix}"
            if key in tm:
                lines.append(f"| {key} | {_fmt(float(tm[key]))} |")

    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- `manifest.json` · `metrics_test.json` · `metrics_test_by_case.csv` · `history.csv` · `best_model.pt`",
            "",
            "## 回填",
            "",
            f"- 实验族汇总：`external_baselines/pointnetcfd/experiments/{exp}/实验分析记录.md`",
            "- 总账：`docs/02-推进与变更/代码修改与实验推进记录.md` 文首",
            "- 规范：`docs/00-规范与记录/外部baseline实验记录规范.md`",
            "",
        ]
    )
    return "\n".join(lines)


def _render_latest_run_block(manifest: Mapping[str, Any], run_dir: Path) -> str:
    tm = manifest["test_metrics"]
    run_name = run_dir.name
    lines = [
        LATEST_RUN_START,
        "## 最新 run（自动同步）",
        "",
        f"- **run_dir**：`{run_name}`",
        f"- **best_epoch**：{manifest.get('best_epoch', '—')}",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
        f"| test MSE | {_fmt(float(tm['loss_mse']))} |",
        f"| test RMSE | {_fmt(float(tm['rmse']))} |",
        f"| test MAE | {_fmt(float(tm['mae']))} |",
        f"| n_points | {int(tm.get('n_points', 0))} |",
        "",
        f"完整快照：[`analysis_report.md`](../../../../outputs/external_baselines/pointnetcfd/{run_name}/analysis_report.md)",
        "",
        LATEST_RUN_END,
    ]
    return "\n".join(lines)


def _upsert_latest_run_block(text: str, block: str) -> str:
    pattern = re.compile(
        re.escape(LATEST_RUN_START) + r".*?" + re.escape(LATEST_RUN_END),
        re.DOTALL,
    )
    if pattern.search(text):
        return pattern.sub(block, text)
    if "## 最新 run" in text or "Job " in text:
        return text.rstrip() + "\n\n" + block + "\n"
    return block + "\n\n" + text


def write_run_reports(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "analysis_report.md"
    snapshot.write_text(render_run_snapshot(manifest, run_dir), encoding="utf-8")

    exp_name = str(manifest["experiment_name"])
    exp_dir = EXPERIMENTS_ROOT / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    family_doc = exp_dir / "实验分析记录.md"
    latest_block = _render_latest_run_block(manifest, run_dir)

    if family_doc.exists():
        existing = family_doc.read_text(encoding="utf-8")
        family_doc.write_text(_upsert_latest_run_block(existing, latest_block), encoding="utf-8")
    else:
        header = (
            f"# {exp_name} · 实验分析记录\n\n"
            f"> 实验族目录 · 配置见 `configs/local/{exp_name}_split_*.json`\n\n"
        )
        family_doc.write_text(header + latest_block + "\n\n" + render_run_snapshot(manifest, run_dir), encoding="utf-8")

    return snapshot


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate PointNetCFD analysis reports from manifest")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir)
    path = write_run_reports(run_dir, manifest)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
