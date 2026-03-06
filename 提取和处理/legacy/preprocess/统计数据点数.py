#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计新数据格式中各病例的点数

功能：
1. 统计每个病例 ascii/ 文件夹中壁面点的数量
2. 统计每个病例 ascii_in/ 文件夹中内部点的数量
3. 输出统计表格，便于后续降采样策略设计
"""

import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

# 数据根目录
DATA_ROOT = Path(__file__).resolve().parents[2] / "data_new" / "AG" / "fast"


def count_lines_in_file(file_path: Path) -> int:
    """统计文件行数（不含表头）"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        # 跳过表头行
        return len(lines) - 1 if len(lines) > 0 else 0
    except Exception as e:
        print(f"  读取失败: {file_path.name}, 错误: {e}")
        return -1


def get_files_in_dir(dir_path: Path) -> list:
    """获取目录中的所有文件（排除隐藏文件）"""
    if not dir_path.is_dir():
        return []
    return sorted([f for f in dir_path.iterdir() 
                   if f.is_file() and not f.name.startswith('.')])


def analyze_case(case_dir: Path) -> dict:
    """分析单个病例的数据"""
    case_name = case_dir.name
    result = {
        "病例名": case_name,
        "壁面文件数": 0,
        "壁面点数(首帧)": 0,
        "壁面点数(末帧)": 0,
        "壁面点数(平均)": 0,
        "内部文件数": 0,
        "内部点数(首帧)": 0,
        "内部点数(末帧)": 0,
        "内部点数(平均)": 0,
        "STL文件": "无",
    }
    
    # 检查 STL 文件
    stl_files = list(case_dir.glob("*.stl"))
    if stl_files:
        result["STL文件"] = stl_files[0].name
    
    # 统计壁面数据 (ascii/)
    ascii_dir = case_dir / "ascii"
    ascii_files = get_files_in_dir(ascii_dir)
    if ascii_files:
        result["壁面文件数"] = len(ascii_files)
        
        # 统计点数
        wall_counts = []
        for f in ascii_files:
            count = count_lines_in_file(f)
            if count > 0:
                wall_counts.append(count)
        
        if wall_counts:
            result["壁面点数(首帧)"] = wall_counts[0]
            result["壁面点数(末帧)"] = wall_counts[-1]
            result["壁面点数(平均)"] = int(sum(wall_counts) / len(wall_counts))
    
    # 统计内部数据 (ascii_in/)
    ascii_in_dir = case_dir / "ascii_in"
    ascii_in_files = get_files_in_dir(ascii_in_dir)
    if ascii_in_files:
        result["内部文件数"] = len(ascii_in_files)
        
        # 统计点数
        inner_counts = []
        for f in ascii_in_files:
            count = count_lines_in_file(f)
            if count > 0:
                inner_counts.append(count)
        
        if inner_counts:
            result["内部点数(首帧)"] = inner_counts[0]
            result["内部点数(末帧)"] = inner_counts[-1]
            result["内部点数(平均)"] = int(sum(inner_counts) / len(inner_counts))
    
    return result


def main():
    print("=" * 80)
    print("新数据格式点数统计")
    print("=" * 80)
    print(f"数据目录: {DATA_ROOT}")
    print()
    
    # 获取所有病例目录
    case_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    
    if not case_dirs:
        print("未找到病例目录！")
        return
    
    print(f"找到 {len(case_dirs)} 个病例\n")
    
    # 分析每个病例
    results = []
    for case_dir in case_dirs:
        print(f"分析: {case_dir.name}...")
        result = analyze_case(case_dir)
        results.append(result)
    
    # 转换为 DataFrame 并显示
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("统计结果")
    print("=" * 80)
    
    # 显示壁面数据统计
    print("\n【壁面数据统计 (ascii/)】")
    wall_df = df[["病例名", "壁面文件数", "壁面点数(首帧)", "壁面点数(末帧)", "壁面点数(平均)", "STL文件"]]
    print(wall_df.to_string(index=False))
    
    # 显示内部数据统计
    print("\n【内部数据统计 (ascii_in/)】")
    inner_df = df[["病例名", "内部文件数", "内部点数(首帧)", "内部点数(末帧)", "内部点数(平均)"]]
    print(inner_df.to_string(index=False))
    
    # 汇总统计
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    
    total_wall = df["壁面点数(平均)"].sum()
    total_inner = df["内部点数(平均)"].sum()
    avg_wall = df["壁面点数(平均)"].mean()
    avg_inner = df["内部点数(平均)"].mean()
    min_wall = df["壁面点数(平均)"].min()
    max_wall = df["壁面点数(平均)"].max()
    min_inner = df["内部点数(平均)"].min()
    max_inner = df["内部点数(平均)"].max()
    
    print(f"\n壁面点数:")
    print(f"  - 平均每病例: {avg_wall:,.0f} 点")
    print(f"  - 最小: {min_wall:,} 点")
    print(f"  - 最大: {max_wall:,} 点")
    print(f"  - 所有病例总计: {total_wall:,} 点")
    
    print(f"\n内部点数:")
    print(f"  - 平均每病例: {avg_inner:,.0f} 点")
    print(f"  - 最小: {min_inner:,} 点")
    print(f"  - 最大: {max_inner:,} 点")
    print(f"  - 所有病例总计: {total_inner:,} 点")
    
    print(f"\n壁面:内部 比例: 1:{avg_inner/avg_wall:.1f}")
    
    # 降采样建议
    print("\n" + "=" * 80)
    print("降采样建议")
    print("=" * 80)
    
    target_totals = [20000, 40000, 60000, 80000]
    print("\n不同目标点数下的降采样比例:")
    print(f"{'目标总点数':>12} | {'壁面采样率':>10} | {'内部采样率':>10}")
    print("-" * 40)
    
    for target in target_totals:
        # 假设壁面点全部保留，只降采样内部点
        wall_ratio = min(1.0, target * 0.3 / avg_wall)  # 30% 给壁面
        inner_ratio = min(1.0, target * 0.7 / avg_inner)  # 70% 给内部
        print(f"{target:>12,} | {wall_ratio:>10.1%} | {inner_ratio:>10.1%}")
    
    # 保存结果到 CSV
    output_path = DATA_ROOT.parent.parent.parent / "数据点数统计.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n统计结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
