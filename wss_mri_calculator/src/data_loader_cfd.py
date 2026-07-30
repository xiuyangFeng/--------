"""
CFD ASCII 数据加载模块（适配 data_new 队列）
=============================================
原仓库 `data_loader.py` 假定输入是规则体素网格的 HDF5（4D Flow MRI 影像）。
本项目的数据是 Fluent 导出的**非结构化点云** ASCII：

  ascii/<CASE>-<step>      壁面节点
    nodenumber, x, y, z, pressure, wall-shear, x-wall-shear, y-wall-shear, z-wall-shear
  ascii_in/<CASE>-<step>   内部单元中心
    cellnumber, x, y, z, pressure, velocity-magnitude, x-velocity, y-velocity, z-velocity

单位约定
--------
  - 坐标：**m**（Fluent 原生），本模块统一转成 **mm** 返回
  - 速度：m/s
  - WSS / 压力：Pa
  - STL：与壁面点同坐标系，单位 mm（文件头标 `<stl unit=MM>`），无需额外变换
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

M_TO_MM = 1e3


def _load_ascii(path: str | Path) -> np.ndarray:
    """
    读取 Fluent ASCII 导出（首行为表头）。

    分隔符自动判定：全库以逗号分隔为主，但少数病例（如 AG/fast/FAN_JIAN_MING）
    导出成空白分隔。按表头是否含逗号来选，空白分隔交给 numpy 默认切分。
    """
    source = Path(path)
    with source.open(encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
    delimiter = "," if "," in header else None
    return np.loadtxt(str(source), delimiter=delimiter, skiprows=1)


def load_wall(path: str | Path) -> dict:
    """
    加载壁面节点数据（真值侧）。

    返回
    ----
    dict:
      coords_mm : (n,3) 壁面节点坐标，mm
      pressure  : (n,)  压力，Pa
      wss_mag   : (n,)  CFD 壁面剪切应力模长，Pa —— **对比基准**
      wss_vec   : (n,3) CFD 壁面剪切应力矢量，Pa
    """
    d = _load_ascii(path)
    if d.shape[1] < 9:
        # 少数病例（如 AG/slow/LIU_XI_QUAN）只导出到 pressure 列，没有 wall-shear，
        # 无法作为真值使用。明确报错，不要让下游拿到 IndexError。
        raise ValueError(
            f"壁面导出缺少 wall-shear 列（只有 {d.shape[1]} 列，需要 9 列）：{path}"
        )
    return {
        "coords_mm": d[:, 1:4] * M_TO_MM,
        "pressure": d[:, 4],
        "wss_mag": d[:, 5],
        "wss_vec": d[:, 6:9],
    }


def load_interior(path: str | Path) -> dict:
    """
    加载内部速度场（输入侧，等价于 4D Flow MRI 的 u/v/w，但为非结构点云）。

    返回
    ----
    dict:
      coords_mm  : (n,3) 单元中心坐标，mm
      pressure   : (n,)  压力，Pa
      vel_mag    : (n,)  速度模长，m/s
      velocity   : (n,3) 速度矢量 (u,v,w)，m/s
    """
    d = _load_ascii(path)
    return {
        "coords_mm": d[:, 1:4] * M_TO_MM,
        "pressure": d[:, 4],
        "vel_mag": d[:, 5],
        "velocity": d[:, 6:9],
    }


def _scan(case_dir: str | Path, subdir: str) -> dict[int, Path]:
    """
    扫描导出目录，返回 {时间步: 路径}。

    文件名形如 `<前缀>-<step>`，前缀不一定等于目录名：AG/AAA 队列前缀就是
    病例名，而 ILO 队列的病例目录下还有 before/after 两层，文件前缀仍是
    病例名（如 `ILO/GAO_SHU_CAI-0/before/ascii/GAO_SHU_CAI-1120`）。
    因此按后缀数字解析，不对前缀做假设。
    """
    root = Path(case_dir) / subdir
    if not root.is_dir():
        raise FileNotFoundError(f"missing export dir: {root}")
    found: dict[int, Path] = {}
    for item in root.iterdir():
        if not item.is_file():
            continue
        head, _, tail = item.name.rpartition("-")
        if head and tail.isdigit():
            found[int(tail)] = item
    return found


def list_steps(case_dir: str | Path, subdir: str = "ascii") -> list[int]:
    """列出某病例已导出的所有时间步编号（升序）。"""
    return sorted(_scan(case_dir, subdir))


def step_path(case_dir: str | Path, step: int, subdir: str = "ascii") -> Path:
    """取某时间步的文件路径。"""
    found = _scan(case_dir, subdir)
    if step not in found:
        raise FileNotFoundError(f"step {step} not exported under {Path(case_dir) / subdir}")
    return found[step]
