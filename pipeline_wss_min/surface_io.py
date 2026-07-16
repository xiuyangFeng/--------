#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""病例原始 STL 的确定性读取、尺度推断与权威版本选择。

新队列中少数病例同时保留多个 STL（例如术前/术后导出的中间版和 final 版）。
配准不能依赖文件名排序碰运气，因此按 STL 与 Fluent 壁面点云在原始世界坐标中的
包围盒中心和三轴跨度共同评分，选择几何最吻合的一份。这里只允许尺度换算，不做
平移/旋转拟合；这样 STL 的原始世界方向仍可作为解剖左右符号锚。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class SurfaceSelection:
    points_mm: np.ndarray
    path: Path
    scale_to_mm: float
    match_score: float
    n_candidates: int


def _bbox(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lo = np.min(pts, axis=0)
    hi = np.max(pts, axis=0)
    span = hi - lo
    diag = float(np.linalg.norm(span))
    return (lo + hi) * 0.5, span, np.maximum(span, 1e-9), diag


def find_case_stls(case_dir: Path) -> List[Path]:
    """只取病例目录顶层 STL，避免误读子目录里的派生可视化文件。"""
    return sorted(
        p for p in case_dir.iterdir()
        if p.is_file() and p.name.lower().endswith(".stl")
    )


def read_stl_points(path: Path) -> np.ndarray:
    try:
        import vtk  # type: ignore
        from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("读取原始 STL 需要 vtk") from exc

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetPoints() is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"STL 无有效点: {path}")
    return vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)


def _candidate_scales(stl_raw: np.ndarray, wall_mm: np.ndarray, unit_factor: float) -> List[float]:
    _, _, _, stl_diag = _bbox(stl_raw)
    _, _, _, wall_diag = _bbox(wall_mm)
    ratio = wall_diag / stl_diag if stl_diag > 1e-12 else 1.0
    vals = [1.0, float(unit_factor), float(ratio)]
    out: List[float] = []
    for value in vals:
        if np.isfinite(value) and value > 0 and not any(np.isclose(value, x, rtol=1e-8) for x in out):
            out.append(value)
    return out


def _bbox_match_score(stl_mm: np.ndarray, wall_mm: np.ndarray) -> float:
    """0 为包围盒完全相同；中心错位和各轴跨度不符都会显著加分。"""
    sc, ss, ss_safe, sd = _bbox(stl_mm)
    wc, ws, ws_safe, wd = _bbox(wall_mm)
    ref = max(wd, 1e-9)
    center_error = float(np.linalg.norm(sc - wc) / ref)
    span_error = float(np.linalg.norm(np.log(ss_safe / ws_safe)))
    diag_error = abs(float(np.log(max(sd, 1e-9) / ref)))
    return center_error + 0.35 * span_error + 0.25 * diag_error


def select_case_surface(
    case_dir: Path,
    wall_mm: np.ndarray,
    unit_factor: float,
    max_points: int = 30000,
) -> SurfaceSelection:
    """选择与 CFD 壁面最匹配的原始 STL，并换算到 pipeline 毫米坐标。

    评分不允许刚性对齐，故选出的点仍保留原始 STL 世界坐标方向。返回点为确定性
    等间隔下采样，足够做解剖关键点检测，同时避免 10 万级 STL 拖慢批处理。
    """
    paths = find_case_stls(case_dir)
    if not paths:
        raise FileNotFoundError(f"病例目录缺原始 STL: {case_dir}")

    best = None
    errors = []
    for path in paths:
        try:
            raw = read_stl_points(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}:{exc}")
            continue
        for scale in _candidate_scales(raw, wall_mm, unit_factor):
            pts = raw * scale
            score = _bbox_match_score(pts, wall_mm)
            item = (score, str(path), scale, pts, path)
            if best is None or item[:2] < best[:2]:
                best = item

    if best is None:
        raise RuntimeError(f"病例 STL 均不可读: {case_dir}; {'; '.join(errors)}")

    score, _, scale, pts, path = best
    if len(pts) > int(max_points):
        # 等间隔索引不依赖随机种子，跨 Python/NumPy 版本完全可复现。
        idx = np.linspace(0, len(pts) - 1, int(max_points), dtype=np.int64)
        pts = pts[idx]
    return SurfaceSelection(
        points_mm=np.asarray(pts, dtype=np.float64),
        path=path,
        scale_to_mm=float(scale),
        match_score=float(score),
        n_candidates=len(paths),
    )
