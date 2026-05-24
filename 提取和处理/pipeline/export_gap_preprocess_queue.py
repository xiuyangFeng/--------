#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出「缺口」预处理队列：ILO 全部 preprocess 就绪病例 + AAA/unruputer 指定 5 例。

`PREPROCESS_DENYLIST`：**预处理 / 全流程导出 / 后续 V3 `data_new` 全量实验**统一的排除路径（相对 data_root），导出脚本会跳过；
含历史上无法 preprocess 的病例、虽可有 `merged` 但缺 STL/BC 或步骤 2+ 未产出者、**AG 侧 OOD 已排除病例**（与 `training/splits/split_AG_v1.json` 的 `excluded_cases` / `docs/…/V3_数据质量与OOD诊断记录.md` 同源）、以及 **AAA+ILO 终审复核**认定的 CFD/BC/WSS 异常（见注释）。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.audit_inputs import discover_case_dirs
    from pipeline.config import DATA_ROOT
    from pipeline.validation import inspect_case_inputs
else:
    from .audit_inputs import discover_case_dirs
    from .config import DATA_ROOT
    from .validation import inspect_case_inputs

# 预处理 + 全流程队列 + 后续 data_new 三域（AAA+AG+ILO）实验：禁止入队 / 勿进 split 候选（路径相对 data_root）
# 当前共 17 条；摘名后须重跑相关步骤/审计再纳入。
PREPROCESS_DENYLIST: Set[str] = {
    # ---- 历史：无法 preprocess（或明确不入队）----
    "AAA/ruputer/FU_GUO_JUN",
    "AAA/ruputer/LV_GUO_YOU",
    "ILO/LI_FA_XIANG-1/before",
    # ---- 2026-05-16：审计为「非全流程齐套」或缺几何/BC，不纳入实验 ----
    "ILO/ZHAO_JIAN_PING-0/before",  # merged 齐，features/后续全缺
    "AAA/unruputer/ZHANG_GUI_HUA",  # 缺 Global_conditions O4（p-outri-rfile.out）
    "ILO/SUN_CHUN_PU-0/before",
    "ILO/SUN_CHUN_PU-0/after",  # 缺表面模型 STL/VTP
    # ---- V3 · AG：OOD / split 已排除（与 `split_AG_v1.excluded_cases` 同步）----
    "AG/fast/PENG_JI_MING",  # 见 `docs/01-任务/任务A/03-V3路线/V3_数据质量与OOD诊断记录.md`
    # ---- V3D · 三域 Tier 0（2026-05-19）：训练离群，污染全局归一化 ----
    "AAA/ruputer/SU_KAI_LI",  # 壁面 p_mean≈1.46e5 Pa；见 V3_三域数据修复计划 §5.1
    # ---- V3D · ILO 壁面近零（2026-05-20）：Fluent 重导未完成，暂剔出 train/候选池 ----
    "ILO/GUO_AI_JUN-0/after",  # 壁面 ~80/81 帧≈0；见 V3_三域数据问题诊断 §6.3
    "ILO/GUO_AI_JUN-0/before",
    # ---- V3 · AAA+ILO：2026-05-16 终审（作业 4662）CFD/BC inlet 或 WSS 尾部异常 ----
    "AAA/ruputer/CHEN_FU",  # Global_conditions 抽样 inlet p05≈0；图侧 BC_Inlet≈0
    "ILO/WANG_CAI-0/before",
    "ILO/XUE_YOU_TANG-0/before",
    "ILO/ZHANG_WAN_ZENG-1/before",
    "ILO/ZHAO_CHANG_SHAN-0/after",
    "ILO/LIU_BAO_JUN-0/after",  # 壁面 ‖WSS‖ 在 z-score 标签空间尾部极端
}

# AAA/unruputer 先前未完成 merged 的 5 例（preprocess 就绪）
UNRUPUTER_GAP: List[str] = [
    "AAA/unruputer/ZHANG_XUN_LIAN",
    "AAA/unruputer/ZHANG_YONG_ZHI",
    "AAA/unruputer/ZHANG_ZHI_HUA",
    "AAA/unruputer/ZHOU_XI_SHENG",
    "AAA/unruputer/ZHU_ZI_HAI",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 ILO 就绪 + unruputer 缺口预处理队列")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.data_root.resolve()
    lines: List[str] = []

    for d in discover_case_dirs(root, ["ILO"], []):
        rel = d.resolve().relative_to(root).as_posix()
        if rel in PREPROCESS_DENYLIST:
            continue
        if not inspect_case_inputs(d)["has_preprocess_inputs"]:
            continue
        lines.append(rel)

    for rel in UNRUPUTER_GAP:
        if rel in PREPROCESS_DENYLIST:
            continue
        p = root / rel
        if not p.is_dir():
            print(f"跳过（目录不存在）: {rel}")
            continue
        if not inspect_case_inputs(p)["has_preprocess_inputs"]:
            print(f"跳过（审计未就绪 preprocess）: {rel}")
            continue
        lines.append(rel)

    lines = sorted(set(lines))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"写入 {len(lines)} 条 -> {args.output}")


if __name__ == "__main__":
    main()
