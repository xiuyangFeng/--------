#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成入口体积流量和各出口流量文件脚本

功能：
1. 读取入口质量流量曲线 (report-def-2-rfile.out)
2. 生成入口体积流量文件 (vf-in-rfile.out):
   - 入口体积流量 = 入口质量流量 / 密度(1060 kg/m³)
3. 读取出口流量比 (outlet-flow-ratio.csv 或 .xlsx)
4. 生成各出口体积流量文件:
   - 出口体积流量 = 入口体积流量 × 出口流量比
5. 保存为对应的 .out 文件 (FLUENT report 格式)

输出文件：
- vf-in-rfile.out: 入口体积流量（与压力边界条件格式一致）
- flow-outle-rfile.out: 左外髂支动脉出口体积流量
- flow-outli-rfile.out: 左内髂支动脉出口体积流量
- flow-outre-rfile.out: 右外髂支动脉出口体积流量
- flow-outri-rfile.out: 右内髂支动脉出口体积流量

支持单病例处理和批量处理。
"""

import os
import csv
from pathlib import Path
import argparse
import pandas as pd


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
    读取出口流量比例文件 (支持 .csv 和 .xlsx)
    
    Parameters:
    -----------
    file_path : str or Path
        outlet-flow-ratio.csv 或 .xlsx 文件路径
    
    Returns:
    --------
    dict
        {outlet_name: ratio} 的字典
    """
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.xlsx':
        # 读取 Excel 文件
        df = pd.read_excel(file_path)
        # 假设格式：第一行为列名（出口名），第二行为数值（比例）
        headers = df.columns.tolist()
        values = df.iloc[0].tolist()
    else:
        # 默认按 CSV 读取
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)  # 第一行：列名
                values = next(reader)   # 第二行：数值
            except StopIteration:
                print(f"⚠️  文件内容为空: {file_path}")
                return {}
    
    # 构建字典
    ratios = {}
    for col_name, value in zip(headers, values):
        try:
            # 过滤掉非数值列
            val = float(value)
            ratios[col_name] = val
        except (ValueError, TypeError):
            continue
    
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
                       ratio_file="outlet-flow-ratio.csv", overwrite=False, density=1060.0,
                       inlet_output_file="vf-in-rfile.out"):
    """
    处理单个病例，生成各出口流量文件和入口体积流量文件
    
    Parameters:
    -----------
    case_dir : str or Path
        病例目录路径
    inlet_file : str
        入口质量流量文件名
    ratio_file : str
        流量比文件名（若设为默认值，将优先尝试寻找 .xlsx 格式）
    overwrite : bool
        是否覆盖已存在的文件
    density : float
        血液密度 (kg/m³)，默认 1060，用于将质量流量转换为体积流量
    inlet_output_file : str
        入口体积流量输出文件名，默认 vf-in-rfile.out（与压力边界格式一致）
    
    Returns:
    --------
    list
        生成的文件路径列表
    """
    case_dir = Path(case_dir)
    inlet_path = case_dir / inlet_file
    
    # 自动识别比例文件：如果默认的 .csv 不存在但 .xlsx 存在，则使用 .xlsx
    if ratio_file == "outlet-flow-ratio.csv":
        ratio_path_xlsx = case_dir / "outlet-flow-ratio.xlsx"
        ratio_path_csv = case_dir / "outlet-flow-ratio.csv"
        if ratio_path_xlsx.exists():
            ratio_path = ratio_path_xlsx
        else:
            ratio_path = ratio_path_csv
    else:
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
    
    # ========================================
    # 1. 生成入口体积流量文件
    # ========================================
    inlet_volume_output = case_dir / inlet_output_file
    
    if inlet_volume_output.exists() and not overwrite:
        print(f"   ⊙ 跳过已存在: {inlet_volume_output.name}")
    else:
        # 计算入口体积流量 = 入口质量流量 / 密度
        inlet_volume_data = []
        for row in inlet_data:
            inlet_volume_data.append({
                'Time Step': row['Time Step'],
                'Value': row['Value'] / density,  # 质量流量 / 密度 = 体积流量
                'flow-time': row['flow-time']
            })
        
        # 写入文件
        write_fluent_report_file(inlet_volume_output, inlet_volume_data, "vf-in")
        generated_files.append(inlet_volume_output)
        print(f"   ✓ 生成入口体积流量: {inlet_volume_output.name}")
    
    # ========================================
    # 2. 生成各出口流量文件
    # ========================================
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
        print(f"   ✓ 生成出口流量: {output_file.name} (比例: {ratio:.6f})")
    
    return generated_files


def batch_process(root_dir, inlet_file="report-def-2-rfile.out",
                 ratio_file="outlet-flow-ratio.csv", overwrite=False, density=1060.0,
                 inlet_output_file="vf-in-rfile.out"):
    """
    批量处理多个病例
    
    Parameters:
    -----------
    root_dir : str or Path
        包含多个病例子目录的根目录
    inlet_file : str
        入口质量流量文件名
    ratio_file : str
        流量比文件名
    overwrite : bool
        是否覆盖已存在的文件
    density : float
        血液密度 (kg/m³)，默认 1060，用于将质量流量转换为体积流量
    inlet_output_file : str
        入口体积流量输出文件名
    
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
    
    print(f"🔍 正在递归搜索有效病例目录...")
    
    # 递归查找所有包含入口流量文件的目录
    # 只要目录下有 inlet_file，我们就认为这是一个需要处理的病例目录
    valid_case_dirs = []
    try:
        # 使用 rglob 匹配文件名
        for p in root_dir.rglob(inlet_file):
            valid_case_dirs.append(p.parent)
    except Exception as e:
        print(f"❌ 搜索目录时出错: {e}")
        return results
    
    # 去重并排序
    valid_case_dirs = sorted(list(set(valid_case_dirs)))
    
    if not valid_case_dirs:
        print(f"⚠️  在 {root_dir} 及其子目录下未找到包含 {inlet_file} 的有效目录")
        return results
    
    print(f"找到 {len(valid_case_dirs)} 个有效病例目录\n")
    
    for case_dir in valid_case_dirs:
        # 计算相对于根目录的路径，方便日志显示
        try:
            display_name = case_dir.relative_to(root_dir)
        except ValueError:
            display_name = case_dir.name

        generated_files = process_single_case(
            case_dir, 
            inlet_file=inlet_file,
            ratio_file=ratio_file,
            overwrite=overwrite,
            density=density,
            inlet_output_file=inlet_output_file
        )
        if generated_files:
            results[str(display_name)] = generated_files
        print("-" * 40)
    
    # 统计信息
    print("=" * 60)
    total_files = sum(len(files) for files in results.values())
    processed_cases = sum(1 for files in results.values() if len(files) > 0)
    print(f"✅ 处理完成！")
    print(f"   - 处理病例数: {processed_cases}/{len(valid_case_dirs)}")
    print(f"   - 生成文件数: {total_files} (含入口体积流量)")
    
    return results


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="生成入口体积流量和各出口流量文件（从入口质量流量和流量比计算）",
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
      --ratio outlet-flow-ratio.csv \\
      --inlet-output vf-in-rfile.out

功能说明:
  1. 读取入口质量流量文件 (report-def-2-rfile.out)
  2. 生成入口体积流量文件: 体积流量 = 质量流量 / 密度(1060)
  3. 生成各出口体积流量文件: 出口流量 = 入口体积流量 × 出口流量比
        """
    )
    
    # 互斥参数组：单个病例或批量处理
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--case', type=str, help='单个病例目录路径')
    group.add_argument('--batch', type=str, help='批量处理的根目录路径')
    
    # 文件名参数
    parser.add_argument('--inlet', type=str, default='report-def-2-rfile.out',
                       help='入口质量流量文件名（默认: report-def-2-rfile.out）')
    parser.add_argument('--ratio', type=str, default='outlet-flow-ratio.csv',
                       help='流量比文件名（默认: outlet-flow-ratio.csv）')
    parser.add_argument('--inlet-output', type=str, default='vf-in-rfile.out',
                       help='入口体积流量输出文件名（默认: vf-in-rfile.out）')
    
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
            density=args.density,
            inlet_output_file=args.inlet_output
        )
        if generated:
            print(f"\n✅ 成功生成 {len(generated)} 个文件（含入口体积流量）")
        else:
            print("\n⚠️  未生成任何文件")
    
    else:
        # 批量处理
        batch_process(
            args.batch,
            inlet_file=args.inlet,
            ratio_file=args.ratio,
            overwrite=args.overwrite,
            density=args.density,
            inlet_output_file=args.inlet_output
        )


if __name__ == "__main__":
    main()

