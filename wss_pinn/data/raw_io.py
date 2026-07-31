"""Fluent ASCII 导出与病例 split 清单的只读解析。

学习要点
--------
上游 ``data_new`` 存的是 CFD 导出表（体域 ``ascii_in``、壁面等）。
本模块只负责：

- 识别表头是逗号还是空白分隔；
- 兼容历史列名（``cellnumber`` / ``nodenumber``）；
- 把 split JSON 归一成统一的病例字典列表（含 raw 目录与 WSS-min bundle 路径）。

不在这里做任何写盘；写 PINN 产物走 ``sidecar.py``。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils import ROOT


# Fluent 体域 ASCII 期望列
INTERIOR_COLUMNS = {
    "id": "cellnumber",
    "id_candidates": ["cellnumber", "nodenumber"],
    "xyz": ["x-coordinate", "y-coordinate", "z-coordinate"],
    "pressure": "pressure",
    "velocity": ["x-velocity", "y-velocity", "z-velocity"],
    "velocity_magnitude": "velocity-magnitude",
}
# 壁面 ASCII 期望列
WALL_COLUMNS = {
    "id_candidates": ["nodenumber", "cellnumber"],
    "xyz": ["x-coordinate", "y-coordinate", "z-coordinate"],
    "pressure": "pressure",
    "wss": "wall-shear",
    "wss_vector": ["x-wall-shear", "y-wall-shear", "z-wall-shear"],
}


def read_table(path: str | Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """读一张 Fluent ASCII：根据首行是否含逗号选择 csv / 空白分隔。"""
    source = Path(path)
    with source.open(encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    kwargs = {"usecols": list(usecols)} if usecols is not None else {}
    if "," in header:
        frame = pd.read_csv(source, skipinitialspace=True, **kwargs)
    else:
        frame = pd.read_csv(source, sep=r"\s+", engine="python", **kwargs)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def list_steps(case_dir: str | Path, subdir: str = "ascii_in") -> list[int]:
    """扫描病例目录下所有 ``*-{step}`` 文件名，返回排序后的时间步列表。"""
    steps: list[int] = []
    for path in Path(case_dir, subdir).iterdir():
        match = re.search(r"-(\d+)$", path.name)
        if path.is_file() and match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def step_file(
    case_dir: str | Path, step: int, subdir: str = "ascii_in"
) -> Path:
    """定位指定时间步的 ASCII 文件（若有多个匹配取排序后第一个）。"""
    candidates = sorted(Path(case_dir, subdir).glob(f"*-{int(step)}"))
    files = [path for path in candidates if path.is_file()]
    if not files:
        raise FileNotFoundError(f"no {subdir} file for step {step}: {case_dir}")
    return files[0]


def read_interior(path: str | Path) -> dict[str, np.ndarray]:
    """读取体域速度/压力；自动选择可用的 ID 列名。"""
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    columns_in_file = [
        value.strip()
        for value in (
            header.split(",") if "," in header else header.split()
        )
    ]
    id_column = next(
        (
            name
            for name in INTERIOR_COLUMNS["id_candidates"]
            if name in columns_in_file
        ),
        None,
    )
    if id_column is None:
        raise KeyError(
            "interior file has neither cellnumber nor historical nodenumber ID"
        )
    columns = [
        id_column,
        *INTERIOR_COLUMNS["xyz"],
        INTERIOR_COLUMNS["pressure"],
        *INTERIOR_COLUMNS["velocity"],
    ]
    frame = read_table(path, columns)
    return {
        "cell_id": frame[id_column].to_numpy(np.int64),
        "id_column": id_column,
        "coords": frame[INTERIOR_COLUMNS["xyz"]].to_numpy(np.float64),
        "pressure": frame[INTERIOR_COLUMNS["pressure"]].to_numpy(np.float64),
        "velocity": frame[INTERIOR_COLUMNS["velocity"]].to_numpy(np.float64),
    }


def read_wall(path: str | Path) -> dict[str, np.ndarray]:
    """读取壁面 WSS 标量与向量、压力。"""
    frame = read_table(path)
    id_column = next(
        (name for name in WALL_COLUMNS["id_candidates"] if name in frame.columns),
        None,
    )
    if id_column is None:
        raise KeyError(f"wall file has no node ID: {path}")
    return {
        "node_id": frame[id_column].to_numpy(np.int64),
        "coords": frame[WALL_COLUMNS["xyz"]].to_numpy(np.float64),
        "pressure": frame[WALL_COLUMNS["pressure"]].to_numpy(np.float64),
        "wss": frame[WALL_COLUMNS["wss"]].to_numpy(np.float64),
        "wss_vector": frame[WALL_COLUMNS["wss_vector"]].to_numpy(np.float64),
    }


def load_cases(path: str | Path) -> list[dict]:
    """加载 split JSON，归一成病例列表。

    支持两种格式：
    1. ``{"cases": [...]}`` 已展开的病例条目；
    2. ``{"train_cases": [...], "test_cases": [...]}`` canonical ID 列表
       （``AG/...`` / ``AAA/...`` / ``ILO/...``），自动填 raw/bundle 路径。
    """
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "cases" in payload:
        cases = payload["cases"]
    elif isinstance(payload, dict) and (
        "train_cases" in payload or "test_cases" in payload
    ):
        cases = []
        for role, key in (("train", "train_cases"), ("test", "test_cases")):
            for canonical_id in payload.get(key, []):
                parts = str(canonical_id).split("/")
                if len(parts) < 3 or parts[0] not in {"AG", "AAA", "ILO"}:
                    raise ValueError(f"invalid canonical case ID: {canonical_id}")
                cohort = parts[0]
                relative = "/".join(parts[1:])
                cases.append(
                    {
                        "canonical_id": str(canonical_id),
                        "case_id": relative,
                        "cohort": cohort,
                        "role": role,
                        "raw_case_dir": str(ROOT / "data_new" / cohort / relative),
                        "bundle_path": str(
                            ROOT / "data_wss_min" / cohort / relative / "bundle.npz"
                        ),
                    }
                )
    else:
        cases = payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("split must contain a non-empty cases list")
    required = {"case_id", "cohort", "raw_case_dir", "bundle_path"}
    normalized = []
    seen = set()
    for case in cases:
        missing = required.difference(case)
        if missing:
            raise ValueError(f"case entry missing {sorted(missing)}")
        row = dict(case)
        row.setdefault("canonical_id", f"{row['cohort']}/{row['case_id']}")
        row.setdefault("role", "unspecified")
        canonical_id = str(row["canonical_id"])
        if canonical_id in seen:
            raise ValueError(f"duplicate canonical case ID: {canonical_id}")
        seen.add(canonical_id)
        normalized.append(row)
    return normalized
