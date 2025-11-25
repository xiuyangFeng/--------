import os
import argparse
import glob
import time
import traceback
from Script_Scenario_B_Volumetric import process_volumetric_dataset

def find_surface_file(case_dir):
    """
    在病例目录中查找唯一的 .stl 或 .vtp 表面文件。
    """
    # 查找所有 stl 和 vtp 文件
    surface_files = glob.glob(os.path.join(case_dir, "*.stl")) + \
                    glob.glob(os.path.join(case_dir, "*.vtp"))
    
    if len(surface_files) == 0:
        print(f"⚠️  [Warning] No surface file (.stl/.vtp) found in {case_dir}")
        return None
    elif len(surface_files) > 1:
        print(f"⚠️  [Warning] Multiple surface files found in {case_dir}, using the first one: {surface_files[0]}")
        return surface_files[0]
    else:
        return surface_files[0]

def find_cloud_files(case_dir):
    """
    在病例目录中查找所有的 .csv 或 .npy 点云文件。
    排除以 'result_' 开头的文件，避免重复处理输出文件。
    """
    all_files = glob.glob(os.path.join(case_dir, "*.csv")) + \
                glob.glob(os.path.join(case_dir, "*.npy"))
    
    # 过滤掉可能是输出结果的文件（假设输出文件包含 'result' 或 'processed'，或者我们只处理特定命名的文件）
    # 这里简单起见，处理所有非 result_ 开头的文件
    cloud_files = [f for f in all_files if "result_" not in os.path.basename(f)]
    
    if len(cloud_files) == 0:
        print(f"⚠️  [Warning] No cloud files (.csv/.npy) found in {case_dir}")
    
    return cloud_files

def process_case(case_dir, output_root_dir):
    """
    处理单个病例目录
    """
    case_name = os.path.basename(case_dir)
    print(f"\n========================================")
    print(f"📂 Processing Case: {case_name}")
    print(f"========================================")
    
    # 1. 查找表面模型
    surface_path = find_surface_file(case_dir)
    if not surface_path:
        print(f"❌ Skipping {case_name}: Missing surface model.")
        return False

    # 2. 查找点云文件
    cloud_files = find_cloud_files(case_dir)
    if not cloud_files:
        print(f"❌ Skipping {case_name}: Missing cloud data.")
        return False
        
    # 3. 准备输出目录
    # 输出结构保持与输入一致：output_root_dir/case_name/
    case_output_dir = os.path.join(output_root_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)
    
    success_count = 0
    
    # 4. 遍历处理每个点云文件
    for cloud_path in cloud_files:
        cloud_filename = os.path.basename(cloud_path)
        # 构建输出文件名，例如: result_features_original_name.csv
        output_filename = f"result_features_{os.path.splitext(cloud_filename)[0]}.csv"
        output_path = os.path.join(case_output_dir, output_filename)
        
        try:
            print(f"   Processing cloud: {cloud_filename} ...")
            start_time = time.time()
            
            # 调用核心处理函数
            process_volumetric_dataset(surface_path, cloud_path, output_path)
            
            elapsed = time.time() - start_time
            print(f"   ✅ Done in {elapsed:.2f}s. Output: {output_filename}")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {cloud_filename}: {str(e)}")
            traceback.print_exc()
            
    if success_count > 0:
        print(f"✨ Case {case_name} completed. {success_count}/{len(cloud_files)} files processed.")
        return True
    else:
        print(f"⚠️  Case {case_name} failed. No files processed successfully.")
        return False

def main():
    parser = argparse.ArgumentParser(description="批量处理血管几何特征提取 (Batch Processing for Vessel Geometric Features)")
    parser.add_argument("--input_dir", type=str, required=True, help="输入数据根目录，包含多个病例文件夹")
    parser.add_argument("--output_dir", type=str, required=True, help="输出数据根目录")
    
    args = parser.parse_args()
    
    input_root = args.input_dir
    output_root = args.output_dir
    
    if not os.path.exists(input_root):
        print(f"❌ Error: Input directory '{input_root}' does not exist.")
        return

    # 获取所有子目录（病例）
    # 过滤掉隐藏文件夹
    subdirs = [os.path.join(input_root, d) for d in os.listdir(input_root) 
               if os.path.isdir(os.path.join(input_root, d)) and not d.startswith('.')]
    
    subdirs.sort()
    
    print(f"🚀 Starting batch processing...")
    print(f"📁 Input Root: {input_root}")
    print(f"📂 Output Root: {output_root}")
    print(f"📊 Total Cases Found: {len(subdirs)}")
    
    total_cases = len(subdirs)
    processed_cases = 0
    failed_cases = 0
    
    start_total = time.time()
    
    for i, case_dir in enumerate(subdirs):
        print(f"\n[{i+1}/{total_cases}] Checking {os.path.basename(case_dir)}...")
        if process_case(case_dir, output_root):
            processed_cases += 1
        else:
            failed_cases += 1
            
    end_total = time.time()
    duration = end_total - start_total
    
    print(f"\n========================================")
    print(f"🎉 Batch Processing Complete!")
    print(f"⏱️  Total Time: {duration:.2f}s")
    print(f"✅ Successful Cases: {processed_cases}")
    print(f"⚠️  Failed/Skipped Cases: {failed_cases}")
    print(f"📂 Results saved to: {output_root}")
    print(f"========================================")

if __name__ == "__main__":
    main()
