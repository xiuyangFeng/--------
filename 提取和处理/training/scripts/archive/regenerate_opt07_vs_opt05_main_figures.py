"""生成 P1-2 / A-Opt-07 与母版 A-Opt-05、叙事基线 A-Main-01 的对照图。

在项目根目录、GNN conda 环境下执行::

    python -m training.scripts.regenerate_opt07_vs_opt05_main_figures

输出目录：``outputs/field/plots/optimization/A_Opt07_vs_Opt05_Main01/``

依赖：各 ``exp_id`` 的 run 已具备 ``predictions_test/manifest.json`` 与
``predictions_test/regional_eval/fig_A5_regional_metrics.json``（需先 ``predict_field`` +
``plot_taskA_regional_bar``；可选 ``plot_taskA_per_case_boxplot`` 生成 Fig A4 CSV）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _chdir_repo_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    import os

    os.chdir(root)
    return root


def _runs_for_exp(runs_root: Path, exp_id: str) -> list[Path]:
    found: list[Path] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        summary_path = p / "summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(data.get("exp_id", "")) == exp_id:
            found.append(p)
    return sorted(found, key=lambda x: x.name)


def main() -> None:
    root = _chdir_repo_root()
    runs_root = root / "outputs" / "field"
    out = root / "outputs" / "field" / "plots" / "optimization" / "A_Opt07_vs_Opt05_Main01"
    out.mkdir(parents=True, exist_ok=True)

    common_filters = [
        "--runs-root",
        str(runs_root),
        "--exp-filter",
        "A-Main-01",
        "--exp-filter",
        "A-Opt-05",
        "--exp-filter",
        "A-Opt-07",
    ]

    argv_bak = sys.argv
    try:
        from training.scripts import plot_taskA_multimodel_per_case_boxplot as box_mod
        from training.scripts import plot_taskA_multimodel_regional_bar as reg_mod
        from training.scripts import plot_taskA_multimodel_scatter as scatter_mod
        from training.scripts import plot_training_history as hist_mod

        for seed in (1, 2, 3):
            for var in ("vel_mag", "p"):
                sys.argv = [
                    "plot_taskA_multimodel_scatter",
                    *common_filters,
                    "--seed-filter",
                    str(seed),
                    "--variable",
                    var,
                    "--region",
                    "interior",
                    "--output-dir",
                    str(out),
                    "--tag",
                    f"seed{seed}",
                ]
                scatter_mod.main()

        sys.argv = [
            "plot_taskA_multimodel_regional_bar",
            *common_filters,
            "--output-dir",
            str(out),
            "--title-prefix",
            "Figure A5: A-Main-01 vs A-Opt-05 vs A-Opt-07",
        ]
        reg_mod.main()

        sys.argv = [
            "plot_taskA_multimodel_per_case_boxplot",
            *common_filters,
            "--output-dir",
            str(out),
            "--region",
            "interior",
            "--title",
            "Figure A4 (interior): Main-01 vs Opt-05 vs Opt-07",
        ]
        box_mod.main()

        run_dirs: list[Path] = []
        for eid in ("A-Main-01", "A-Opt-05", "A-Opt-07"):
            run_dirs.extend(_runs_for_exp(runs_root, eid))
        if len(run_dirs) < 3:
            print(
                "警告: 未找到足够的 run 目录用于 training_history（"
                f"期望至少 3 个 exp 各 1 run，实际收集 {len(run_dirs)} 个目录），已跳过训练曲线对比。"
            )
        else:
            hist_argv: list[str] = [
                "plot_training_history",
                "--runs-root",
                str(runs_root),
                "--pattern",
                "__regenerate_opt07_no_glob__/history.csv",
                "--output-dir",
                str(out),
                "--compare-title",
                "A-Main-01 vs A-Opt-05 vs A-Opt-07 (val_loss)",
                "--compare-metric",
                "val_loss",
            ]
            for rd in run_dirs:
                hist_argv.extend(["--run-dir", str(rd)])
            sys.argv = hist_argv
            hist_mod.main()
    finally:
        sys.argv = argv_bak

    print(f"对照图已写入: {out}")


if __name__ == "__main__":
    main()
