#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中心线批量提取工具（适用于 data 文件夹结构）

用于从 data 目录下的各病例 STL 文件中提取血管中心线。
支持扫描 data/AAA/rupture、data/AAA/unrupture、data/fast、data/ILO 等子目录。

使用示例:
  # 处理所有病例
  python centerline_extract_min.py
  
  # 处理指定病例
  python centerline_extract_min.py --case FENG_LI_XIN
  
  # 覆盖已存在的中心线
  python centerline_extract_min.py --overwrite
  
  # 仅生成 VTP，不导出 CSV
  python centerline_extract_min.py --no-csv
"""

import os
import argparse
import traceback
from pathlib import Path
from typing import Optional, List

import pandas as pd
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

import sys
# 添加父目录到路径，以便导入 vmtk_core
sys.path.insert(0, str(Path(__file__).parent.parent))
import vmtk_core


# 数据源路径配置
DATA_SOURCES = [
    "AAA/rupture",
    "AAA/unrupture",
    "fast",
    "slow",
    "ILO/sq",
    "ILO/sh",
]


def find_stl(case_dir: Path) -> Optional[Path]:
    """
    在病例目录下寻找可用的 STL 文件。
    优先匹配与目录同名的文件（如 FENG_LI_XIN/FENG_LI_XIN.stl），否则取第一个 .stl。
    """
    case_name = case_dir.name
    
    # 尝试多种可能的命名格式
    possible_names = [
        f"{case_name}.stl",
        f"{case_name.replace('_', ' ')}.stl",
        f"{case_name.replace(' ', '_')}.stl",
    ]
    
    for name in possible_names:
        prefer_path = case_dir / name
        if prefer_path.exists():
            return prefer_path
    
    # 否则查找任意 .stl 文件
    for fname in case_dir.iterdir():
        if fname.suffix.lower() == ".stl":
            return fname
    
    return None


def export_centerline(centerline, vtp_path: Path, csv_path: Optional[Path]):
    """
    将中心线同时保存为 VTP 与 CSV（便于快速查看）。
    """
    vtp_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存 VTP
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(vtp_path))
    writer.SetInputData(centerline)
    if writer.Write() == 0:
        raise RuntimeError(f"写入 VTP 失败: {vtp_path}")

    if not csv_path:
        return

    pts = vtk_to_numpy(centerline.GetPoints().GetData())
    pd_obj = centerline.GetPointData()

    def get_arr(name: str):
        arr = pd_obj.GetArray(name)
        return vtk_to_numpy(arr) if arr is not None else None

    data = {
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
    }

    # 尝试导出常用几何属性，若不存在则跳过
    for key in [
        "Abscissas",
        "MaximumInscribedSphereRadius",
        "Curvature",
    ]:
        arr = get_arr(key)
        if arr is not None:
            data[key] = arr

    tangent = get_arr("FrenetTangent")
    if tangent is not None and len(tangent.shape) > 1 and tangent.shape[1] == 3:
        data["Tangent_X"] = tangent[:, 0]
        data["Tangent_Y"] = tangent[:, 1]
        data["Tangent_Z"] = tangent[:, 2]

    pd.DataFrame(data).to_csv(csv_path, index=False)


def process_case(case_dir: Path, overwrite: bool, save_csv: bool) -> bool:
    """
    针对单个病例执行中心线提取。
    
    参数:
        case_dir: 病例目录（内含 STL 文件）
        overwrite: 是否覆盖已存在的输出
        save_csv: 是否同时导出 CSV 格式
    
    返回:
        是否成功处理
    """
    case_name = case_dir.name
    stl_path = find_stl(case_dir)
    
    if not stl_path:
        print(f"  ⚠️  跳过: 未找到 STL 文件")
        return False

    # 输出到病例目录下的 centerline 子目录
    centerline_dir = case_dir / "centerline"
    vtp_path = centerline_dir / "centerline.vtp"
    csv_path = centerline_dir / "centerline_points.csv" if save_csv else None

    if vtp_path.exists() and not overwrite:
        print(f"  ⏩ 已存在输出，使用 --overwrite 可重算")
        return True

    try:
        print(f"  🚀 提取中心线: {stl_path.name}")
        surface = vmtk_core.read_surface(str(stl_path))
        centerline = vmtk_core.extract_rich_centerline(surface)
        export_centerline(centerline, vtp_path, csv_path)
        print(f"  ✅ 完成，保存至 {centerline_dir}")
        return True
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        traceback.print_exc()
        return False


def scan_cases(data_root: Path, target_case: Optional[str] = None) -> List[Path]:
    """
    扫描 data 目录下的所有病例目录。
    
    参数:
        data_root: data 根目录
        target_case: 指定处理的病例名（可选）
    
    返回:
        病例目录路径列表
    """
    cases = []
    
    for source in DATA_SOURCES:
        source_path = data_root / source
        if not source_path.exists():
            continue
        
        for case_dir in source_path.iterdir():
            if not case_dir.is_dir() or case_dir.name.startswith('.'):
                continue
            
            # 如果指定了目标病例，则只处理匹配的病例
            if target_case:
                # 标准化名称进行比较
                case_std = case_dir.name.replace(' ', '_').replace('-', '_').upper()
                target_std = target_case.replace(' ', '_').replace('-', '_').upper()
                if case_std != target_std:
                    continue
            
            cases.append(case_dir)
    
    # 按名称排序
    cases.sort(key=lambda p: p.name)
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="批量提取中心线（适用于 data 文件夹结构）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据源路径:
  - data/AAA/rupture/
  - data/AAA/unrupture/
  - data/fast/
  - data/slow/
  - data/ILO/sq/
  - data/ILO/sh/

输出位置:
  中心线将保存到各病例目录下的 centerline/ 子目录:
  - centerline/centerline.vtp (VTK 格式)
  - centerline/centerline_points.csv (CSV 格式，可选)

示例:
  # 处理所有病例
  python centerline_extract_min.py
  
  # 处理指定病例
  python centerline_extract_min.py --case FENG_LI_XIN
  
  # 指定 data 目录位置
  python centerline_extract_min.py --data-root ../data
  
  # 覆盖已存在的中心线
  python centerline_extract_min.py --overwrite
        """
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data",
        help="data 根目录路径，默认 ../data",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="指定处理的病例名称（可选，不指定则处理所有）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出已存在则强制重算",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="仅生成 VTP，不导出 CSV",
    )
    args = parser.parse_args()

    # 解析路径
    script_dir = Path(__file__).parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = script_dir / data_root
    data_root = data_root.resolve()

    save_csv = not args.no_csv

    if not data_root.is_dir():
        print(f"❌ data 目录不存在: {data_root}")
        return

    # 扫描病例
    cases = scan_cases(data_root, args.case)

    if not cases:
        if args.case:
            print(f"❌ 未找到病例: {args.case}")
        else:
            print(f"❌ 未找到任何病例")
        return

    print(f"📂 找到 {len(cases)} 个待处理病例 (data 根目录: {data_root})")
    print()

    ok = 0
    for idx, case_dir in enumerate(cases, start=1):
        # 计算相对路径用于显示
        try:
            rel_path = case_dir.relative_to(data_root)
        except ValueError:
            rel_path = case_dir.name
        
        print(f"[{idx}/{len(cases)}] {rel_path}")
        if process_case(case_dir, args.overwrite, save_csv):
            ok += 1
        print()

    print(f"🎉 完成: {ok}/{len(cases)} 个病例生成中心线")


if __name__ == "__main__":
    main()

