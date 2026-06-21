from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pipeline.config import INNER_DIR, SURFACE_DIR
from pipeline.utils.io import clean_cfd_data, load_ascii_df


def _frame_key_from_path(path: Path) -> str:
    """例如 ascii_in/ZHANG_CHUN-1120 -> merged-1120"""
    stem = path.name
    match = re.search(r"-(\d+)$", stem)
    if match:
        return f"merged-{match.group(1)}"
    return stem


def list_raw_frames(case_dir: Path) -> Dict[str, Tuple[Path, Path | None]]:
    """
    返回 {merged-XXXX: (inner_path, surface_path|None)}，与 pipeline preprocess 编号对齐。
    """
    inner_dir = case_dir / INNER_DIR
    surface_dir = case_dir / SURFACE_DIR
    inner_files: Dict[str, Path] = {}
    if inner_dir.is_dir():
        for p in inner_dir.iterdir():
            if p.is_file():
                inner_files[_frame_key_from_path(p)] = p
    surface_files: Dict[str, Path] = {}
    if surface_dir.is_dir():
        for p in surface_dir.iterdir():
            if p.is_file():
                surface_files[_frame_key_from_path(p)] = p
    keys = sorted(set(inner_files) | set(surface_files))
    return {k: (inner_files.get(k), surface_files.get(k)) for k in keys if k in inner_files}


def load_raw_frame(
    case_dir: Path,
    frame_key: str,
    point_filter: str,
    convert_to_mm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从 Fluent ascii 读取物理量点云（CROWN 论文口径，不经 pipeline FPS 降采样）。

    point_filter:
      - volume / interior: 仅 ascii_in 体点（论文 volume）
      - all: ascii_in + ascii 壁面
    """
    frames = list_raw_frames(case_dir)
    if frame_key not in frames:
        raise FileNotFoundError(f"无原始帧 {frame_key} under {case_dir}")
    inner_path, surface_path = frames[frame_key]
    parts: List[pd.DataFrame] = []

    if inner_path is not None and point_filter in ("volume", "interior", "all"):
        inner_df = clean_cfd_data(load_ascii_df(inner_path), convert_to_mm=convert_to_mm, is_wall=False)
        parts.append(inner_df)

    if point_filter == "all" and surface_path is not None:
        wall_df = clean_cfd_data(load_ascii_df(surface_path), convert_to_mm=convert_to_mm, is_wall=True)
        parts.append(wall_df)

    if not parts:
        raise ValueError(f"point_filter={point_filter} 未产生任何点")

    df = pd.concat(parts, ignore_index=True)
    required = ["x", "y", "z", "u", "v", "w", "p"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"原始帧缺少列 {missing}: {frame_key}")

    points = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    targets = df[["u", "v", "w", "p"]].to_numpy(dtype=np.float64)
    return points, targets
