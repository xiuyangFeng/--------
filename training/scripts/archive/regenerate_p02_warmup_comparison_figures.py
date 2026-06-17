"""生成 A-Main-01 / A-Opt-02 / A-Opt-02_warmup 三模型对照图（与 P0-2 优化线作图方式对齐）。

在项目根目录执行::

    conda activate GNN
    python -m training.scripts.regenerate_p02_warmup_comparison_figures

输出目录：``outputs/field/plots/optimization/prenorm_Main_P02_P02w/``
"""
from __future__ import annotations

import sys
from pathlib import Path


def _chdir_repo_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    import os

    os.chdir(root)
    return root


def main() -> None:
    root = _chdir_repo_root()
    out = root / "outputs/field/plots/optimization/prenorm_Main_P02_P02w"
    out.mkdir(parents=True, exist_ok=True)
    runs_root = root / "outputs/field"
    common = [
        "--runs-root",
        str(runs_root),
        "--exp-filter",
        "A-Main-01",
        "--exp-filter",
        "A-Opt-02",
        "--exp-filter",
        "A-Opt-02_warmup",
    ]

    argv_bak = sys.argv
    try:
        from training.scripts import plot_taskA_multimodel_scatter as scatter_mod
        from training.scripts import plot_taskA_multimodel_regional_bar as reg_mod
        from training.scripts import plot_taskA_multimodel_per_case_boxplot as box_mod

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
            "Figure A5: Main vs Pre-Norm vs Pre-Norm+Warmup5",
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
            "Figure A4: Per-case (interior) Main vs P0-2 vs P0-3",
        ]
        box_mod.main()
    finally:
        sys.argv = argv_bak

    print(f"对照图已写入: {out}")


if __name__ == "__main__":
    main()
