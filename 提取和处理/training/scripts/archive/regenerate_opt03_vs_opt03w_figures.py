"""生成 A-Opt-03 / A-Opt-03w（Pre-Norm+tw22205 与 +Warmup5）对照图。

在项目根目录执行::

    conda activate GNN
    python -m training.scripts.regenerate_opt03_vs_opt03w_figures

输出目录：``outputs/field/plots/optimization/prenorm_A_Opt03_vs_Opt03w/``
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
    runs_root = root / "outputs/field"
    out = root / "outputs/field/plots/optimization/prenorm_A_Opt03_vs_Opt03w"
    out.mkdir(parents=True, exist_ok=True)

    common = [
        "--runs-root",
        str(runs_root),
        "--exp-filter",
        "A-Opt-03",
        "--exp-filter",
        "A-Opt-03w",
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
                    *common,
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
            *common,
            "--output-dir",
            str(out),
            "--title-prefix",
            "Figure A5: A-Opt-03 vs A-Opt-03w",
        ]
        reg_mod.main()

        sys.argv = [
            "plot_taskA_multimodel_per_case_boxplot",
            *common,
            "--output-dir",
            str(out),
            "--region",
            "interior",
            "--title",
            "Figure A4 (interior): A-Opt-03 vs A-Opt-03w",
        ]
        box_mod.main()

        run_dirs = _runs_for_exp(runs_root, "A-Opt-03") + _runs_for_exp(
            runs_root, "A-Opt-03w"
        )
        if len(run_dirs) < 2:
            print(
                "警告: 未找到足够的 run 目录用于 training_history（"
                f"期望 6 个，实际 {len(run_dirs)}），已跳过训练曲线对比。"
            )
        else:
            # 仅使用下方 --run-dir；避免 plot_training_history 默认再 glob 全目录 */history.csv
            hist_argv: list[str] = [
                "plot_training_history",
                "--runs-root",
                str(runs_root),
                "--pattern",
                "__regenerate_opt03_no_glob__/history.csv",
                "--output-dir",
                str(out),
                "--compare-title",
                "A-Opt-03 vs A-Opt-03w (val_loss)",
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
