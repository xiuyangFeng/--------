#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""原始 CFD 读取。

- 壁面 ascii：nodenumber, x/y/z, pressure, wall-shear(+分量)
- 内部 ascii_in：cellnumber, x/y/z, pressure, velocity-magnitude(+分量)
- centerline：单条有序路径（abscissa 0->L）
- 入口波形 vf-in：定位峰值收缩期时间步

坐标在所有时间步内是静态的（已核实），所以几何只读一次；场值按时间步堆叠。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import config as C


def _read_ascii(path: Path) -> pd.DataFrame:
    """读 Fluent ascii。分隔符不统一：部分病例逗号分隔，部分空白分隔。
    用表头首行探测分隔符，逗号走快速 C 解析，空白走正则解析。"""
    with open(path) as fh:
        header = fh.readline()
    if "," in header:
        df = pd.read_csv(path, skipinitialspace=True)
    else:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def _header_line(path: Path) -> str:
    with open(path) as fh:
        return fh.readline().strip()


def delimiter_of(path: Path) -> str:
    """探测分隔符：'comma' | 'whitespace'。"""
    return "comma" if "," in _header_line(path) else "whitespace"


def header_columns(path: Path) -> List[str]:
    """返回列名（对逗号/空白两种格式都适用）。"""
    return [c.strip() for c in _header_line(path).replace(",", " ").split()]


def wall_id_column(columns: List[str]) -> str | None:
    """返回壁面节点 ID 列名。

    多数壁面 ascii 使用 `nodenumber`，少数历史导出误命名为 `cellnumber`。
    二者都可作为跨时间步对齐键；真正缺 ID 才应视为不可做行序守卫。
    """
    for name in (C.WALL_COLUMNS["id"], C.INTERIOR_COLUMNS["id"]):
        if name in columns:
            return name
    return None


def list_timesteps(case_dir: Path) -> List[int]:
    """从壁面 ascii 文件名解析导出时间步（升序）。"""
    wall_dir = case_dir / C.RAW_LAYOUT["wall_ascii_dir"]
    steps = []
    for f in wall_dir.iterdir():
        if not f.is_file():
            continue
        m = re.search(r"-(\d+)$", f.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def _step_file(case_dir: Path, sub_dir: str, case_name: str, step: int) -> Path:
    """解析某时间步文件。文件名前缀不一定等于目录名（如目录 HOU_SHEN_QIAN、
    文件 HOU_SHEN_QIAN3-1120），故先试精确名，再按 `-<step>` 后缀回退匹配。"""
    d = case_dir / sub_dir
    exact = d / f"{case_name}-{step}"
    if exact.is_file():
        return exact
    matches = sorted(p for p in d.glob(f"*-{step}") if p.is_file())
    return matches[0] if matches else exact


def read_wall_geometry(case_dir: Path, case_name: str, step: int) -> np.ndarray:
    """壁面点坐标 (N_wall, 3)（任一时间步，坐标静态）。"""
    col = C.WALL_COLUMNS
    df = _read_ascii(_step_file(case_dir, C.RAW_LAYOUT["wall_ascii_dir"], case_name, step))
    return df[[col["x"], col["y"], col["z"]]].to_numpy(dtype=np.float64)  # 原生单位，缩放在 preprocess


def read_interior_geometry(case_dir: Path, case_name: str, step: int) -> np.ndarray:
    """内部 cell 坐标 (N_int, 3)，原生单位（缩放到 mm 在 preprocess 逐病例做）。"""
    col = C.INTERIOR_COLUMNS
    df = _read_ascii(_step_file(case_dir, C.RAW_LAYOUT["interior_ascii_dir"], case_name, step))
    return df[[col["x"], col["y"], col["z"]]].to_numpy(dtype=np.float64)


def read_wall_fields(case_dir: Path, case_name: str, step: int) -> Dict[str, np.ndarray]:
    """壁面场：nodenumber/坐标 + wss 标量 (N,)、wss 矢量 (N,3)、pressure (N,)。

    `nodenumber`/`cellnumber` 是跨时间步静默错位的 P0 守卫键：预处理会先按首个
    时间步的节点顺序对齐，再堆叠 WSS/pressure。
    """
    col = C.WALL_COLUMNS
    df = _read_ascii(_step_file(case_dir, C.RAW_LAYOUT["wall_ascii_dir"], case_name, step))
    id_col = wall_id_column(list(df.columns))
    if id_col is None:
        raise KeyError("壁面导出缺少 nodenumber/cellnumber ID 列")
    return {
        "nodenumber": df[id_col].to_numpy(dtype=np.int64),
        "coords": df[[col["x"], col["y"], col["z"]]].to_numpy(dtype=np.float64),
        "wss": df[col["wss"]].to_numpy(dtype=np.float64),
        "wss_vec": df[[col["wss_x"], col["wss_y"], col["wss_z"]]].to_numpy(dtype=np.float64),
        "pressure": df[col["pressure"]].to_numpy(dtype=np.float64),
    }


def read_interior_fields(case_dir: Path, case_name: str, step: int) -> Dict[str, np.ndarray]:
    """内部场：velocity 矢量 (N,3)、vel_mag (N,)、pressure (N,)。"""
    col = C.INTERIOR_COLUMNS
    df = _read_ascii(_step_file(case_dir, C.RAW_LAYOUT["interior_ascii_dir"], case_name, step))
    return {
        "vel": df[[col["u"], col["v"], col["w"]]].to_numpy(dtype=np.float64),
        "vel_mag": df[col["vel_mag"]].to_numpy(dtype=np.float64),
        "pressure": df[col["pressure"]].to_numpy(dtype=np.float64),
    }


def read_centerline(case_dir: Path) -> Dict[str, np.ndarray]:
    """中心线（单条有序路径）。"""
    col = C.CENTERLINE_COLUMNS
    df = pd.read_csv(case_dir / C.RAW_LAYOUT["centerline_csv"])
    df.columns = [c.strip() for c in df.columns]
    out = {
        "coords": df[[col["x"], col["y"], col["z"]]].to_numpy(dtype=np.float64),
        "abscissa": df[col["abscissa"]].to_numpy(dtype=np.float64),
        "radius": df[col["radius"]].to_numpy(dtype=np.float64),
        "curvature": df[col["curvature"]].to_numpy(dtype=np.float64),
        "tangent": df[[col["tan_x"], col["tan_y"], col["tan_z"]]].to_numpy(dtype=np.float64),
    }
    out.update(_read_centerline_vtp_optional(case_dir, expected_n=len(df)))
    return out


def _read_centerline_vtp_optional(case_dir: Path, expected_n: int) -> Dict[str, np.ndarray]:
    """Read optional VMTK centerline topology arrays from centerline.vtp.

    The CSV export carries the ordered path used by older code.  The VTP keeps
    richer VMTK point arrays such as DistToBifurcation and BranchId, which are
    needed for the stricter anatomical frame.  Missing VTK/VTP or point-count
    mismatch is treated as a soft fallback to the CSV-only path.
    """
    vtp_path = case_dir / C.RAW_LAYOUT["centerline_vtp"]
    if not vtp_path.is_file():
        return {"vtp_available": np.array(False)}

    try:
        import vtk  # type: ignore
        from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
    except Exception:
        return {"vtp_available": np.array(False)}

    try:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(vtp_path))
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetPoints() is None:
            return {"vtp_available": np.array(False)}
        pts = vtk_to_numpy(poly.GetPoints().GetData())
        if len(pts) != expected_n:
            return {"vtp_available": np.array(False)}

        pdata = poly.GetPointData()

        def _arr(name: str):
            arr = pdata.GetArray(name)
            if arr is None:
                return None
            val = vtk_to_numpy(arr)
            return val if len(val) == expected_n else None

        out: Dict[str, np.ndarray] = {"vtp_available": np.array(True)}
        mapping = {
            "dist_to_bifurcation": "DistToBifurcation",
            "branch_id": "BranchId",
            "torsion": "Torsion",
            "dR_ds": "dR_ds",
            "tangent_change_rate": "TangentChangeRate",
        }
        for key, name in mapping.items():
            arr = _arr(name)
            if arr is not None:
                out[key] = arr.astype(np.float64)
        frenet = _arr("FrenetTangent")
        if frenet is not None and frenet.ndim == 2 and frenet.shape[1] == 3:
            out["frenet_tangent"] = frenet.astype(np.float64)
        return out
    except Exception:
        return {"vtp_available": np.array(False)}


def read_inlet_waveform(case_dir: Path) -> Dict[int, float]:
    """解析 vf-in-rfile.out -> {timestep: inlet_flow}。峰值收缩期 = 该值最大处。"""
    path = case_dir / C.RAW_LAYOUT["inlet_waveform"]
    out: Dict[int, float] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        # 数据行形如: "<step:int> <vf:float> <flow-time:float>"
        try:
            step = int(parts[0])
            vf = float(parts[1])
        except ValueError:
            continue
        out[step] = vf
    return out


def peak_systole_step(case_dir: Path, exported_steps: List[int]) -> int:
    """导出时间步区间内 入口流量 vf-in 最大的时间步。缺波形则回退到区间中点。"""
    wf = read_inlet_waveform(case_dir)
    cand = {s: wf[s] for s in exported_steps if s in wf}
    if not cand:
        return exported_steps[len(exported_steps) // 2]
    return max(cand, key=cand.get)
