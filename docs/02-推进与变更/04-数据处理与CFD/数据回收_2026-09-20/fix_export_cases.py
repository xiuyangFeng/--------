"""2026-09-22 第四批：3 例 cas 导出配置错（入库链第 2 波拓扑审计抓到），改法与 09-21 fix_cfd_cases.py 相同（只改路径与导出定义，不改物理设置；备份 *.orig_20260922_export）。
  ILO/WANG_LI_MIN-0/after    体场导出 (surfaces inlet) → (surfaces)：原导出只有入口面 726 行
  ILO/WEI_QING_FENG-1/after  体场导出 quantities 只有 "Static Pressure" → 补 Velocity Magnitude / X / Y / Z Velocity（与正常 ILO 例逐字相同）
  ILO/LIU_CHUN_YANG-1/before 壁面导出 cellzones blood blood1..5 + cell-centered → cellzones blood + node-based（与正常 ILO 例逐字相同）
三例 2.jou / cas 内路径还是 1 月的 GNN/data/ILO/ILO/… 旧前缀 → 当前目录。用法：python fix_export_cases.py  → fix_export_cases_log.md
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fcc", HERE / "fix_cfd_cases.py"); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
import os
F.STAMP = os.environ.get("FIX_STAMP", "orig_20260922_export")
VOL_Q = b'(quantities "Static Pressure" "Velocity Magnitude" "X Velocity" "Y Velocity" "Z Velocity")'
WALL_Q = b'(quantities "Static Pressure" "Wall Shear Stress" "X-Wall Shear Stress" "Y-Wall Shear Stress" "Z-Wall Shear Stress")'
CASES = {
    # 2026-09-22 晚（第 3 波拓扑审计抓到）：体场导出 quantities 是壁面剪切那一组、壁面导出 cellzones 带 blood1..5（全域式）
    "ILO/WANG_LI_MIN-0/before": [(b'(surfaces) (cellzones blood) ' + WALL_Q, b'(surfaces) (cellzones blood) ' + VOL_Q),
                                 (b'(surfaces wall) (cellzones blood blood1 blood2 blood3 blood4 blood5) ' + WALL_Q, b'(surfaces wall) (cellzones blood) ' + WALL_Q)],
    "ILO/WANG_LI_MIN-0/after": [(b'(surfaces inlet) (cellzones blood) ' + VOL_Q, b'(surfaces) (cellzones blood) ' + VOL_Q)],
    "ILO/WEI_QING_FENG-1/after": [(b'(surfaces) (cellzones blood) (quantities "Static Pressure") (vector-quantities)', b'(surfaces) (cellzones blood) ' + VOL_Q + b' (vector-quantities)')],
    "ILO/LIU_CHUN_YANG-1/before": [(b'(surfaces wall) (cellzones blood blood1 blood2 blood3 blood4 blood5) ' + WALL_Q + b' (vector-quantities) (append-type . "time-step") (digits-in-file . 6) (export/cell-centered? . #t) (node? . #f)',
                                    b'(surfaces wall) (cellzones blood) ' + WALL_Q + b' (vector-quantities) (append-type . "time-step") (digits-in-file . 6) (export/cell-centered? . #f) (node? . #t)')],
}
def main() -> None:
    from wss_pinn.v4 import new_case_sources as ncs
    rows = ["| 病例 | 2.jou | .cas.gz |", "|---|---|---|"]
    only = os.environ.get("ONLY_CASES", "").split(",") if os.environ.get("ONLY_CASES") else None
    for cid, extra in CASES.items():
        if only and cid not in only: continue
        d = F.ROOT / "data_new" / cid; base = ncs.fluent_case_from_raw(d).name
        base_j, j = F.fix_jou(d, scale=False); assert base_j == base, (cid, base_j, base); c = F.fix_cas(d, base, extra)
        rows.append(f"| `{cid}` | {j} | {c} |"); print(rows[-1])
    (HERE / "fix_export_cases_log.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
if __name__ == "__main__":
    sys.path.insert(0, str(F.ROOT)); main()
