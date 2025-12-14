#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成各出口流量文件脚本

功能：
1. 读取入口质量流量曲线 (report-def-2-rfile.out)
2. 转换为体积流量：体积流量 = 质量流量 / 密度(1060 kg/m³)
3. 读取出口流量比 (outlet-flow-ratio.csv)
4. 计算各出口流量 = (入口质量流量 / 1060) × 出口流量比
5. 保存为对应的 .out 文件

支持单病例处理和批量处理。
"""

import os
import csv
from pathlib import Path
import argparse


def read_fluent_report_file(file_path):
    """
    读取 FLUENT report 格式的 .out 文件
    
    Parameters:
    -----------
    file_path : str or Path
        .out 文件路径
    
    Returns:
    --------
    list of dict
        包含三列：Time Step, Value, flow-time 的列表
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过前3行（标题和列名）
    data_lines = lines[3:]
    
    # 解析数据
    data = []
    for line in data_lines:
        line = line.strip()
        if line:  # 非空行
            parts = line.split()
            if len(parts) == 3:
                data.append({
                    'Time Step': int(parts[0]),
                    'Value': float(parts[1]),
                    'flow-time': float(parts[2])
                })
    
    return data


def read_outlet_flow_ratio(file_path):
    """
    读取出口流量比例文件
    
    Parameters:
    -----------
    file_path : str or Path
        outlet-flow-ratio.csv 文件路径
    
    Returns:
    --------
    dict
        {outlet_name: ratio} 的字典
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # 第一行：列名
        values = next(reader)   # 第二行：数值
    
    # 构建字典
    ratios = {}
    for col_name, value in zip(headers, values):
        ratios[col_name] = float(value)
    
    return ratios


def write_fluent_report_file(file_path, data, outlet_name, title_suffix="etc.."):
    """
    写入 FLUENT report 格式的 .out 文件
    
    Parameters:
    -----------
    file_path : str or Path
        输出文件路径
    data : list of dict
        包含 Time Step, Value, flow-time 的数据列表
    outlet_name : str
        出口名称（用于文件头）
    title_suffix : str
        标题后缀
    """
    file_name = Path(file_path).stem
    
    with open(file_path, 'w') as f:
        # 写入文件头（FLUENT格式）
        f.write(f'"{file_name}"\n')
        f.write(f'"Time Step" "{outlet_name} {title_suffix}"\n')
        f.write(f'("Time Step" "{outlet_name}" "flow-time")\n')
        
        # 写入数据
        for row in data:
            f.write(f"{row['Time Step']} {row['Value']:.15g} {row['flow-time']:.15g}\n")


def process_single_case(case_dir, inlet_file="report-def-2-rfile.out", 
                       ratio_file="outlet-flow-ratio.csv", overwrite=False, density=1060.0):
    """
    处理单个病例，生成各出口流量文件
    
    Parameters:
    -----------
    case_dir : str or Path
        病例目录路径
    inlet_file : str
        入口流量文件名
    ratio_file : str
        流量比文件名
    overwrite : bool
        是否覆盖已存在的文件
    density : float
        血液密度 (kg/m³)，默认 1060，用于将质量流量转换为体积流量
    
    Returns:
    --------
    list
        生成的文件路径列表
    """
    case_dir = Path(case_dir)
    inlet_path = case_dir / inlet_file
    ratio_path = case_dir / ratio_file
    
    # 检查文件是否存在
    if not inlet_path.exists():
        print(f"⚠️  未找到入口流量文件: {inlet_path}")
        return []
    
    if not ratio_path.exists():
        print(f"⚠️  未找到流量比文件: {ratio_path}")
        return []
    
    print(f"🔄 处理病例: {case_dir.name}")
    
    # 读取入口流量数据
    inlet_data = read_fluent_report_file(inlet_path)
    print(f"   ✓ 读取入口质量流量数据: {len(inlet_data)} 个时间步")
    
    # 读取出口流量比
    flow_ratios = read_outlet_flow_ratio(ratio_path)
    print(f"   ✓ 读取出口流量比: {len(flow_ratios)} 个出口")
    print(f"   ✓ 使用血液密度: {density} kg/m³")
    
    # 为每个出口生成流量文件
    generated_files = []
    
    for outlet_name, ratio in flow_ratios.items():
        # 构造输出文件名（从 flow-outle-rfile 变为 flow-outle-rfile.out）
        output_file = case_dir / f"{outlet_name}.out"
        
        # 检查是否已存在
        if output_file.exists() and not overwrite:
            print(f"   ⊙ 跳过已存在: {output_file.name}")
            continue
        
        # 计算该出口的流量 = (入口质量流量 / 密度) × 流量比
        # 即：体积流量 = 质量流量 / 密度，然后按比例分配
        outlet_data = []
        for row in inlet_data:
            outlet_data.append({
                'Time Step': row['Time Step'],
                'Value': (row['Value'] / density) * ratio,
                'flow-time': row['flow-time']
            })
        
        # 写入文件
        write_fluent_report_file(output_file, outlet_data, outlet_name)
        generated_files.append(output_file)
        print(f"   ✓ 生成: {output_file.name} (比例: {ratio:.6f})")
    
    return generated_files


def batch_process(root_dir, inlet_file="report-def-2-rfile.out",
                 ratio_file="outlet-flow-ratio.csv", overwrite=False, density=1060.0):
    """
    批量处理多个病例
    
    Parameters:
    -----------
    root_dir : str or Path
        包含多个病例子目录的根目录
    inlet_file : str
        入口流量文件名
    ratio_file : str
        流量比文件名
    overwrite : bool
        是否覆盖已存在的文件
    density : float
        血液密度 (kg/m³)，默认 1060，用于将质量流量转换为体积流量
    
    Returns:
    --------
    dict
        {case_name: [generated_files]} 的字典
    """
    root_dir = Path(root_dir)
    
    if not root_dir.exists():
        print(f"❌ 目录不存在: {root_dir}")
        return {}
    
    print(f"📁 批量处理目录: {root_dir}")
    print("=" * 60)
    
    results = {}
    
    # 遍历所有子目录
    case_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    
    if not case_dirs:
        print("⚠️  未找到任何子目录")
        return results
    
    print(f"找到 {len(case_dirs)} 个病例目录\n")
    
    for case_dir in case_dirs:
        generated_files = process_single_case(
            case_dir, 
            inlet_file=inlet_file,
            ratio_file=ratio_file,
            overwrite=overwrite,
            density=density
        )
        results[case_dir.name] = generated_files
        print()
    
    # 统计信息
    print("=" * 60)
    total_files = sum(len(files) for files in results.values())
    processed_cases = sum(1 for files in results.values() if len(files) > 0)
    print(f"✅ 处理完成！")
    print(f"   - 处理病例数: {processed_cases}/{len(case_dirs)}")
    print(f"   - 生成文件数: {total_files}")
    
    return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="生成各出口流量文件（从入口流量和流量比计算）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个病例
  python generate_outlet_flows.py --case 点云/001
  
  # 批量处理所有病例
  python generate_outlet_flows.py --batch 点云
  
  # 覆盖已存在的文件
  python generate_outlet_flows.py --batch 点云 --overwrite
  
  # 自定义文件名
  python generate_outlet_flows.py --case 点云/001 \\
      --inlet report-def-2-rfile.out \\
      --ratio outlet-flow-ratio.csv
        """
    )
    
    # 互斥参数组：单个病例或批量处理
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--case', type=str, help='单个病例目录路径')
    group.add_argument('--batch', type=str, help='批量处理的根目录路径')
    
    # 文件名参数
    parser.add_argument('--inlet', type=str, default='report-def-2-rfile.out',
                       help='入口流量文件名（默认: report-def-2-rfile.out）')
    parser.add_argument('--ratio', type=str, default='outlet-flow-ratio.csv',
                       help='流量比文件名（默认: outlet-flow-ratio.csv）')
    
    # 其他选项
    parser.add_argument('--density', type=float, default=1060.0,
                       help='血液密度 kg/m³（默认: 1060.0），用于将质量流量转换为体积流量')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的文件')
    
    args = parser.parse_args()
    
    # 执行处理
    if args.case:
        # 单个病例处理
        generated = process_single_case(
            args.case,
            inlet_file=args.inlet,
            ratio_file=args.ratio,
            overwrite=args.overwrite,
            density=args.density
        )
        if generated:
            print(f"\n✅ 成功生成 {len(generated)} 个文件")
        else:
            print("\n⚠️  未生成任何文件")
    
    else:
        # 批量处理
        batch_process(
            args.batch,
            inlet_file=args.inlet,
            ratio_file=args.ratio,
            overwrite=args.overwrite,
            density=args.density
        )


if __name__ == "__main__":
    main()

