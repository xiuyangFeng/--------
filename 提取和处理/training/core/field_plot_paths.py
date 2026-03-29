"""任务 A 汇总图目录约定：<runs-root>/plots/<category>/…

避免所有 PNG/CSV 堆在 plots/ 根目录；与 ``optimization/<campaign>/`` 子目录并列。
"""
from __future__ import annotations

from pathlib import Path

# ── 一级分类（plots 直下）──────────────────────────────────────────────────
CAT_SUMMARY = "summary"
CAT_CASE_PANELS = "case_panels"
CAT_MULTIMODEL_BASELINE = "multimodel_baseline"
CAT_ABLATION = "ablation"
CAT_EFFICIENCY = "efficiency"
CAT_TRAINING_CURVES = "training_curves"
CAT_OPTIMIZATION = "optimization"

# 优化/消融子课题（optimization 下），供脚本或文档引用
SUBDIR_P0_1_OPT01_VS_MAIN01 = "P0-1_A-Opt01_vs_Main01"
SUBDIR_P0_2_PRENORM_VS_MAIN01 = "prenorm_A_Opt02_vs_Main01"


def plots_root(runs_root: str | Path) -> Path:
    return Path(runs_root).resolve() / "plots"


def category_dir(runs_root: str | Path, category: str) -> Path:
    p = plots_root(runs_root) / category
    p.mkdir(parents=True, exist_ok=True)
    return p


def optimization_campaign_dir(runs_root: str | Path, campaign: str) -> Path:
    p = plots_root(runs_root) / CAT_OPTIMIZATION / campaign
    p.mkdir(parents=True, exist_ok=True)
    return p
