from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = PACKAGE_ROOT / "experiments"

LATEST_RUN_START = "<!-- crown_beihang:latest-run:start -->"
LATEST_RUN_END = "<!-- crown_beihang:latest-run:end -->"


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def render_run_snapshot(manifest: Mapping[str, Any], run_dir: Path) -> str:
    exp = manifest["experiment_name"]
    split = manifest["split_version"]
    seed = manifest["seed"]
    outputs = list(manifest["output_names"])
    tm = manifest["test_metrics"]
    best_ep = manifest.get("best_epoch", "—")
    best_val = manifest.get("best_val_loss")
    run_rel = run_dir.as_posix()
    pinn = manifest.get("physics_enabled", False)

    lines = [
        "# CROWN/Beihang · 单次 run 分析报告",
        "",
        f"- **实验族**：`{exp}`",
        f"- **run_dir**：`{run_rel}`",
        f"- **split**：`{split}` · seed **{seed}**",
        f"- **PINN**：{'是' if pinn else '否'} · 输出 `{', '.join(outputs)}`",
        f"- **point_filter**：`{manifest.get('point_filter', 'all')}`",
        f"- **best_epoch**：{best_ep}"
        + (f" · best_val_loss={_fmt(float(best_val))}" if best_val is not None else ""),
        "",
        "## 指标口径",
        "",
        "速度 `u,v,w` 为物理量；`p` 经 train-only min-max 训练，test 指标已反归一化到物理压力。",
        "",
        "## Test 集整体指标",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
        f"| loss_mse | {_fmt(float(tm.get('loss_mse', 0)))} |",
        f"| RMSE | {_fmt(float(tm.get('rmse', 0)))} |",
        f"| MAE | {_fmt(float(tm.get('mae', 0)))} |",
        f"| NMAE | {_fmt(float(tm.get('nmae', 0)))} |",
        f"| n_points | {int(tm.get('n_points', 0))} |",
        "",
        "## NMAE",
        "",
        "| 指标 | 值 |",
        "| --- | ---: |",
    ]
    for name in outputs:
        key = f"{name}_nmae"
        if key in tm:
            lines.append(f"| {key} | {_fmt(float(tm[key]))} |")
    if "vel_mag_nmae" in tm:
        lines.append(f"| vel_mag_nmae | {_fmt(float(tm['vel_mag_nmae']))} |")
    lines.extend(
        [
            "",
            "## 点级 R²",
            "",
            "| 指标 | 值 |",
            "| --- | ---: |",
        ]
    )
    for name in outputs:
        key = f"{name}_r2"
        if key in tm:
            lines.append(f"| {key} | {_fmt(float(tm[key]))} |")
    if "vel_mag_r2" in tm:
        lines.append(f"| vel_mag_r2 | {_fmt(float(tm['vel_mag_r2']))} |")
    lines.extend(
        [
            "",
            "## 分场 MAE / RMSE / NMAE",
            "",
            "| 指标 | 值 |",
            "| --- | ---: |",
        ]
    )
    for name in outputs:
        for suffix in ("mae", "rmse", "nmae", "r2"):
            key = f"{name}_{suffix}"
            if key in tm:
                lines.append(f"| {key} | {_fmt(float(tm[key]))} |")
    if "vel_mag_r2" in tm:
        lines.append(f"| vel_mag_r2 | {_fmt(float(tm['vel_mag_r2']))} |")
    if "vel_mag_nmae" in tm:
        lines.append(f"| vel_mag_nmae | {_fmt(float(tm['vel_mag_nmae']))} |")

    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- `manifest.json` · `metrics_test.json` · `history.csv` · `best_model.pt`",
            "",
            "## 回填",
            "",
            f"- 实验族：`external_baselines/crown_beihang/experiments/{exp}/实验分析记录.md`",
            "- 总账：`docs/02-推进与变更/代码修改与实验推进记录.md` 文首",
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
        f"| test MSE | {_fmt(float(tm.get('loss_mse', 0)))} |",
        f"| test RMSE | {_fmt(float(tm.get('rmse', 0)))} |",
        f"| test NMAE | {_fmt(float(tm.get('nmae', 0)))} |",
        "",
        f"完整快照：`outputs/external_baselines/crown_beihang/{run_name}/analysis_report.md`",
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
    return block + "\n\n" + text


def write_run_reports(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
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
            f"> 配置见 `configs/local/{exp_name}_split_*.json`\n\n"
        )
        family_doc.write_text(
            header + latest_block + "\n\n" + render_run_snapshot(manifest, run_dir),
            encoding="utf-8",
        )
    return snapshot
