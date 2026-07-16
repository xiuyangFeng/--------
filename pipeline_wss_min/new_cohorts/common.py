"""AAA/ILO 新队列白名单、路径解析与共享常量。"""

from __future__ import annotations

import pandas as pd

from pipeline_wss_min import config as C


ASSET_DIR = C.PROJECT_ROOT / "docs" / "02-推进与变更" / "assets_新队列审计"
CLASSIFIED = ASSET_DIR / "audit_classified.csv"
FRAME_AUDIT = ASSET_DIR / "landmark_frame_v4_audit.csv"
PASS_CATEGORIES = {"F_clean_pass", "E_dense_clean"}


def split_unit(unit_id: str) -> tuple[str, str]:
    """把标准单元编号拆成 pipeline 使用的队列路径和病例路径。"""
    parts = unit_id.split("/")
    if parts[0] == "AAA" and len(parts) == 3:
        return "/".join(parts[:2]), parts[2]
    if parts[0] == "ILO" and len(parts) == 3:
        return "ILO", "/".join(parts[1:])
    raise ValueError(f"非法新队列 unit_id: {unit_id}")


def candidate_units() -> list[str]:
    """返回同时通过数据层审计和 v4 坐标架审计的确定性白名单。"""
    data = pd.read_csv(CLASSIFIED)
    units = data[data["cat"].isin(PASS_CATEGORIES)]["unit_id"].tolist()
    if FRAME_AUDIT.is_file():
        frame = pd.read_csv(FRAME_AUDIT)
        passed = set(frame[frame["frame_pass"] == True]["unit_id"])
        units = [unit for unit in units if unit in passed]
    return units
