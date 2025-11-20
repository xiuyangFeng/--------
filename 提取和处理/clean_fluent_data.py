import pandas as pd
import numpy as np
import os

def convert_fluent_to_csv(input_path, output_path, convert_to_mm=True):
    print(f"🧹 正在清洗 CFD 数据: {input_path}")
    
    # 1. 读取非标准格式
    # sep=r'\s+' 表示分隔符是任意数量的空格
    # skiprows 需要根据文件实际情况调整，如果第一行就是表头，就不需要 skip
    try:
        df = pd.read_csv(input_path, sep=r'\s+', engine='python')
    except Exception as e:
        print(f"读取失败，请检查表头: {e}")
        return

    print(f"   - 原始数据列名: {df.columns.tolist()}")
    
    # 2. 重命名列 (标准化为我们代码通用的名字)
    # 根据你的数据片段，建立映射关系
    rename_map = {
        'x-coordinate': 'x',
        'y-coordinate': 'y',
        'z-coordinate': 'z',
        'pressure': 'p',
        'x-velocity': 'u',
        'y-velocity': 'v',
        'z-velocity': 'w',
        'wall-shear': 'wss',  # 可选
        'nodenumber': 'node_id' # 可选
    }
    
    # 过滤掉无关列，只保留映射中存在的
    # 注意：有的软件导出列名可能略有不同，如 "x-velo" 等，请核对
    df = df.rename(columns=rename_map)
    
    # 确保只保留我们需要的列
    needed_cols = ['x', 'y', 'z', 'u', 'v', 'w', 'p']
    # 如果有 wss 也保留，后面分析用
    if 'wss' in df.columns:
        needed_cols.append('wss')
        
    # 检查列是否存在
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        print(f"❌ 警告: 缺少列 {missing_cols}，请检查原始文件的列名拼写！")
        return

    df = df[needed_cols]

    # 3. 关键步骤：单位转换 (米 -> 毫米)
    # 你的数据看起来是 E-02 (0.01米级)，通常是米。
    # 而 STL 也是医学影像导出的，通常是毫米。
    if convert_to_mm:
        print("   - [Unit] 检测到数据可能是米(m)，正在转换为毫米(mm)...")
        df['x'] = df['x'] * 1000.0
        df['y'] = df['y'] * 1000.0
        df['z'] = df['z'] * 1000.0
        # 注意：速度单位 m/s 不需要变，压力 Pa 不需要变
        # 只有几何坐标需要对齐

    # 4. 生成 is_wall 标签 (可选，但对 GNN 很重要)
    # 逻辑：速度为0的点是壁面
    speed = np.sqrt(df['u']**2 + df['v']**2 + df['w']**2)
    df['is_wall'] = (speed < 1e-6).astype(int)
    print(f"   - 识别到 {df['is_wall'].sum()} 个壁面点，{len(df) - df['is_wall'].sum()} 个内部点。")

    # 5. 保存
    df.to_csv(output_path, index=False)
    print(f"✅ 清洗完成! 已保存至: {output_path}")
    print(df.head())

if __name__ == "__main__":
    # 输入你的文件名 (把你的数据保存为 cfd_raw.txt)
    input_file = "cfd_raw.txt" 
    output_file = "cfd_clean.csv"
    
    # 创建一个假的测试文件（为了演示，你可以直接用你的真实文件）
    # 实际使用时请注释掉下面生成文件的代码
    if not os.path.exists(input_file):
        with open(input_file, 'w') as f:
            f.write("""nodenumber x-coordinate y-coordinate z-coordinate pressure velocity-magnitude x-velocity y-velocity z-velocity wall-shear x-wall-shear y-wall-shear z-wall-shear
1 -2.819E-02 1.528E-01 -8.884E-01 1.026E+02 0.000E+00 0.000E+00 0.000E+00 0.000E+00 4.853E-01 -1.26E-01 -2.80E-01 -3.72E-01
2 -2.820E-02 1.527E-01 -8.883E-01 1.026E+02 2.308E-03 -6.057E-04 -1.343E-03 -1.777E-03 0.000E+00 0.000E+00 0.000E+00 0.000E+00
""")

    convert_fluent_to_csv(input_file, output_file, convert_to_mm=True)