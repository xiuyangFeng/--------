import os
import argparse
import glob
import time
import traceback
import re


def _load_processing_functions():
    try:
        from .Script_Scenario_B_Volumetric import (
            process_volumetric_dataset,
            prepare_geometry_data,
            process_single_cloud,
        )
    except ImportError:
        from Script_Scenario_B_Volumetric import (  # type: ignore
            process_volumetric_dataset,
            prepare_geometry_data,
            process_single_cloud,
        )

    return process_volumetric_dataset, prepare_geometry_data, process_single_cloud

def find_surface_file(case_dir):
    """
    在病例目录中查找唯一的 .stl 或 .vtp 表面文件。
    """
    # 查找所有 stl 和 vtp 文件
    surface_files = glob.glob(os.path.join(case_dir, "*.stl")) + \
                    glob.glob(os.path.join(case_dir, "*.vtp"))
    
    if len(surface_files) == 0:
        print(f"⚠️  [警告] 在 {case_dir} 中未找到表面文件 (.stl/.vtp)")
        return None
    elif len(surface_files) > 1:
        print(f"⚠️  [警告] 在 {case_dir} 中找到多个表面文件，使用第一个: {surface_files[0]}")
        return surface_files[0]
    else:
        return surface_files[0]

def find_cloud_files(case_dir, cloud_subdir="ascii_merged"):
    """
    在病例目录中查找所有的 .csv 或 .npy 点云文件。
    cloud_subdir: 清洗后点云所在的子目录，默认 ascii_merged。
    排除以 'result_' 开头的文件，避免重复处理输出文件。
    """
    search_dir = os.path.join(case_dir, cloud_subdir)
    if not os.path.isdir(search_dir):
        # 回退到病例根目录
        search_dir = case_dir
    all_files = glob.glob(os.path.join(search_dir, "*.csv")) + \
                glob.glob(os.path.join(search_dir, "*.npy"))
    
    # 过滤掉可能是输出结果的文件（假设输出文件包含 'result' 或 'processed'，或者我们只处理特定命名的文件）
    cloud_files = [f for f in all_files if "result_" not in os.path.basename(f)]
    
    if len(cloud_files) == 0:
        print(f"⚠️  [警告] 在 {search_dir} 中未找到点云文件 (.csv/.npy)")
    
    return cloud_files

def detect_boundary_condition_type(case_dir):
    """
    检测边界条件类型。
    
    基于出口文件类型来区分：
    - 压力边界：有 p-out*.out 文件（出口为压力）
    - 流量边界：有 flow-out*.out 文件（出口为流量）
    
    注意：两种边界条件现在都使用 vf-in-rfile.out 作为入口文件
    （流量边界的 vf-in-rfile.out 由 generate_outlet_flows.py 从质量流量转换而来）
    
    返回: 
        1 = 压力边界
        0 = 流量边界
        None = 无法检测
    """
    # 检查入口文件（两种边界条件都需要）
    inlet_file = os.path.join(case_dir, "vf-in-rfile.out")
    
    # 通过出口文件类型来区分边界条件类型
    pressure_outlet = os.path.join(case_dir, "p-outle-rfile.out")
    flow_outlet = os.path.join(case_dir, "flow-outle-rfile.out")
    
    if os.path.exists(inlet_file) and os.path.exists(pressure_outlet):
        return 1  # 压力边界
    elif os.path.exists(inlet_file) and os.path.exists(flow_outlet):
        return 0  # 流量边界
    else:
        # 回退检测：检查原始流量边界文件
        original_flow_inlet = os.path.join(case_dir, "report-def-2-rfile.out")
        if os.path.exists(original_flow_inlet) and os.path.exists(flow_outlet):
            print(f"⚠️  [提示] 检测到原始流量边界文件，请先运行 generate_outlet_flows.py 生成 vf-in-rfile.out")
            return 0  # 流量边界（但入口文件需要转换）
        return None


def read_bc_file(file_path):
    """
    读取单个边界条件文件，返回 step -> value 的字典。
    """
    data = {}
    if not os.path.exists(file_path):
        return data
        
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # 跳过头部，找到数据开始的地方
        start_idx = 0
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                start_idx = i
                break
        
        # 读取数据
        for line in lines[start_idx:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            try:
                step = int(parts[0])
                val = float(parts[1])
                data[step] = val
            except ValueError:
                continue
                
    except Exception as e:
        print(f"❌ 读取 {file_path} 失败: {str(e)}")
        
    return data


def load_boundary_conditions(case_dir):
    """
    加载该病例的所有边界条件文件 (.out)。
    
    自动检测边界条件类型：
    - BC_Flag=1: 压力边界 (vf-in + p-out*)
    - BC_Flag=0: 流量边界 (vf-in + flow-out*)
    
    注意：两种边界条件现在都使用 vf-in-rfile.out 作为入口文件。
    流量边界的 vf-in-rfile.out 由 generate_outlet_flows.py 从质量流量转换而来。
    
    返回一个字典: time_step -> [BC_Flag, in, O1, O2, O3, O4]
    其中:
        - BC_Flag: 边界条件类型标志 (1=压力边界, 0=流量边界)
        - in: 入口体积流量 (两种边界条件统一为体积流量)
        - O1: 左外髂支动脉出口 (压力边界时为压力, 流量边界时为流量)
        - O2: 左内髂支动脉出口
        - O3: 右外髂支动脉出口
        - O4: 右内髂支动脉出口
    """
    # 检测边界条件类型
    bc_type = detect_boundary_condition_type(case_dir)
    
    if bc_type is None:
        print(f"⚠️  [警告] 无法检测边界条件类型，缺少必要的边界条件文件")
        return {}
    
    bc_type_name = "压力边界" if bc_type == 1 else "流量边界"
    print(f"📊 检测到边界条件类型: {bc_type_name} (BC_Flag={bc_type})")
    
    # 根据边界条件类型定义文件映射
    # 注意：两种边界条件现在都使用 vf-in-rfile.out 作为入口文件
    # 流量边界的 vf-in-rfile.out 由 generate_outlet_flows.py 从质量流量转换而来
    if bc_type == 1:  # 压力边界
        file_mapping = {
            "inlet": "vf-in-rfile.out",       # 入口体积速率
            "O1": "p-outle-rfile.out",        # 左外髂支动脉出口压力
            "O2": "p-outli-rfile.out",        # 左内髂支动脉出口压力
            "O3": "p-outre-rfile.out",        # 右外髂支动脉出口压力
            "O4": "p-outri-rfile.out"         # 右内髂支动脉出口压力
        }
    else:  # 流量边界
        file_mapping = {
            "inlet": "vf-in-rfile.out",       # 入口体积流量 (由 generate_outlet_flows.py 生成)
            "O1": "flow-outle-rfile.out",     # 左外髂支动脉出口流量
            "O2": "flow-outli-rfile.out",     # 左内髂支动脉出口流量
            "O3": "flow-outre-rfile.out",     # 右外髂支动脉出口流量
            "O4": "flow-outri-rfile.out"      # 右内髂支动脉出口流量
        }
    
    # 读取各个文件的数据
    bc_raw_data = {}
    for key, filename in file_mapping.items():
        file_path = os.path.join(case_dir, filename)
        if not os.path.exists(file_path):
            print(f"⚠️  [警告] 未找到边界条件文件: {filename}")
            bc_raw_data[key] = {}
        else:
            bc_raw_data[key] = read_bc_file(file_path)
    
    # 获取所有时间步的并集
    all_steps = set()
    for key in bc_raw_data:
        all_steps.update(bc_raw_data[key].keys())
    
    # 构建最终的边界条件数据: step -> [BC_Flag, in, O1, O2, O3, O4]
    bc_data = {}
    for step in all_steps:
        inlet_val = bc_raw_data["inlet"].get(step, 0.0)
        o1_val = bc_raw_data["O1"].get(step, 0.0)
        o2_val = bc_raw_data["O2"].get(step, 0.0)
        o3_val = bc_raw_data["O3"].get(step, 0.0)
        o4_val = bc_raw_data["O4"].get(step, 0.0)
        
        bc_data[step] = [float(bc_type), inlet_val, o1_val, o2_val, o3_val, o4_val]
    
    if bc_data:
        print(f"✅ 成功加载 {len(bc_data)} 个时间步的边界条件数据")
    
    return bc_data

def process_case(case_dir, output_root_dir, cloud_subdir="ascii_merged", output_subdir="ascii_mapped"):
    """
    处理单个病例目录。
    cloud_subdir: 已清洗的 CFD 点云所在子目录（默认 ascii_merged）。
    output_subdir: 输出特征文件的子目录（默认 ascii_mapped）。
    """
    case_name = os.path.basename(case_dir)
    _, prepare_geometry_data, process_single_cloud = _load_processing_functions()
    print(f"\n========================================")
    print(f"📂 处理病例: {case_name}")
    print(f"========================================")
    
    # 1. 查找表面模型
    surface_path = find_surface_file(case_dir)
    if not surface_path:
        print(f"❌ 跳过 {case_name}: 缺少表面模型。")
        return False

    # 2. 查找点云文件
    cloud_files = find_cloud_files(case_dir, cloud_subdir=cloud_subdir)
    if not cloud_files:
        print(f"❌ 跳过 {case_name}: 缺少点云数据。")
        return False
        
    # 3. 准备输出目录
    # 输出结构保持与输入一致：output_root_dir/case_name/output_subdir
    case_output_dir = os.path.join(output_root_dir, case_name, output_subdir)
    os.makedirs(case_output_dir, exist_ok=True)
    
    success_count = 0
    
    # 3.5 预处理几何数据 (只做一次)
    try:
        geo_data = prepare_geometry_data(surface_path)
    except Exception as e:
        print(f"❌ 预处理几何 {surface_path} 失败: {str(e)}")
        traceback.print_exc()
        return False

    # 3.6 加载全局边界条件
    global_bcs_map = load_boundary_conditions(case_dir)
    if not global_bcs_map:
        print(f"⚠️  [警告] {case_name} 未加载到任何边界条件数据。")

    # 4. 遍历处理每个点云文件
    for cloud_path in cloud_files:
        cloud_filename = os.path.basename(cloud_path)
        # 构建输出文件名，例如: result_features_original_name.csv
        output_filename = f"result_features_{os.path.splitext(cloud_filename)[0]}.csv"
        output_path = os.path.join(case_output_dir, output_filename)
        
        try:
            print(f"   处理点云: {cloud_filename} ...")
            start_time = time.time()
            
            # 提取时间步
            # 假设文件名格式: CaseName-TimeStep.csv (例如 004-1121.csv)
            # 使用正则提取文件名末尾的数字
            time_step = None
            match = re.search(r'-(\d+)\.', cloud_filename)
            if match:
                time_step = int(match.group(1))
            
            current_bcs = None
            if time_step is not None and time_step in global_bcs_map:
                current_bcs = global_bcs_map[time_step]
            elif time_step is not None and global_bcs_map:
                # 如果找不到精确匹配的时间步，尝试找最近的时间步
                available_steps = list(global_bcs_map.keys())
                closest_step = min(available_steps, key=lambda s: abs(s - time_step))
                current_bcs = global_bcs_map[closest_step]
                # print(f"   ⚠️  警告: 时间步 {time_step} 使用最近的 BC 数据 (step={closest_step})。")
            
            # 调用核心处理函数 (使用预处理好的几何数据)
            process_single_cloud(geo_data, cloud_path, output_path, global_bcs=current_bcs)
            
            elapsed = time.time() - start_time
            print(f"   ✅ 完成，耗时 {elapsed:.2f}s. 输出: {output_filename}")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ 处理 {cloud_filename} 时出错: {str(e)}")
            traceback.print_exc()
            
    if success_count > 0:
        print(f"✨ 病例 {case_name} 完成。成功处理 {success_count}/{len(cloud_files)} 个文件。")
        return True
    else:
        print(f"⚠️  病例 {case_name} 失败。没有文件被成功处理。")
        return False

def main():
    parser = argparse.ArgumentParser(description="批量处理血管几何特征提取 (Batch Processing for Vessel Geometric Features)")
    parser.add_argument("--input_dir", type=str, default="点云", help="输入数据根目录，包含多个病例文件夹 (默认: 点云)")
    parser.add_argument("--output_dir", type=str, default="outdate", help="输出数据根目录 (默认: outdate)")
    parser.add_argument("--cloud_subdir", type=str, default="ascii_merged", help="清洗后的 CFD 点云子目录，默认 ascii_merged")
    parser.add_argument("--output_subdir", type=str, default="ascii_mapped", help="特征输出子目录，默认 ascii_mapped")
    
    args = parser.parse_args()
    
    input_root = args.input_dir
    output_root = args.output_dir
    
    if not os.path.exists(input_root):
        print(f"❌ 错误: 输入目录 '{input_root}' 不存在。")
        return

    # 获取所有子目录（病例）
    # 过滤掉隐藏文件夹
    subdirs = [os.path.join(input_root, d) for d in os.listdir(input_root) 
               if os.path.isdir(os.path.join(input_root, d)) and not d.startswith('.')]
    
    subdirs.sort()
    
    print(f"🚀 开始批量处理...")
    print(f"📁 输入根目录: {input_root}")
    print(f"📂 输出根目录: {output_root}")
    print(f"📊 找到的总病例数: {len(subdirs)}")
    
    total_cases = len(subdirs)
    processed_cases = 0
    failed_cases = 0
    
    start_total = time.time()
    
    for i, case_dir in enumerate(subdirs):
        print(f"\n[{i+1}/{total_cases}] 正在检查 {os.path.basename(case_dir)}...")
        if process_case(case_dir, output_root, cloud_subdir=args.cloud_subdir, output_subdir=args.output_subdir):
            processed_cases += 1
        else:
            failed_cases += 1
            
    end_total = time.time()
    duration = end_total - start_total
    
    print(f"\n========================================")
    print(f"🎉 批量处理完成!")
    print(f"⏱️  总耗时: {duration:.2f}s")
    print(f"✅ 成功处理的病例: {processed_cases}")
    print(f"⚠️  失败/跳过的病例: {failed_cases}")
    print(f"📂 结果保存至: {output_root}")
    print(f"========================================")

if __name__ == "__main__":
    main()
