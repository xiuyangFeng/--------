import os
from pathlib import Path
import pandas as pd
import numpy as np

# 将 Fluent 输出列名映射到统一字段，方便后续代码直接使用
RENAME_MAP = {
    "x-coordinate": "x",
    "y-coordinate": "y",
    "z-coordinate": "z",
    "pressure": "p",
    "x-velocity": "u",
    "y-velocity": "v",
    "z-velocity": "w",
    "velocity-magnitude": "vel_mag",
    "wall-shear": "wss",
    "x-wall-shear": "wss_x",
    "y-wall-shear": "wss_y",
    "z-wall-shear": "wss_z",
    "face-area-magnitude": "face_area",
    "nodenumber": "node_id",
}


def convert_fluent_to_csv(input_path: Path, output_path: Path, convert_to_mm: bool = True) -> None:
    """
    清洗单个 Fluent 导出的 ASCII 数据文件并写出 CSV。
    参数:
        input_path: 输入 ASCII 文件路径
        output_path: 输出 CSV 文件路径
        convert_to_mm: 是否将坐标从米转换为毫米
    """
    print(f"🧹 正在清洗 CFD 数据: {input_path}")

    try:
        df = pd.read_csv(input_path, sep=r"\s+", engine="python")
    except Exception as e:
        print(f"❌ 读取失败，请检查表头: {e}")
        return

    # 重命名列，统一字段名
    df = df.rename(columns=RENAME_MAP)
    print(f"   - 原始数据列名: {df.columns.tolist()}")

    # 确保几何坐标存在
    missing_xyz = [c for c in ("x", "y", "z") if c not in df.columns]
    if missing_xyz:
        print(f"❌ 缺少坐标列 {missing_xyz}，跳过该文件。")
        return

    # 可选的常见列顺序，便于后续模型直接读取
    preferred_cols = [
        "node_id",
        "x",
        "y",
        "z",
        "u",
        "v",
        "w",
        "p",
        "vel_mag",
        "wss",
        "wss_x",
        "wss_y",
        "wss_z",
        "face_area",
    ]
    available_cols = [c for c in preferred_cols if c in df.columns]

    # 坐标单位转换：米 -> 毫米
    if convert_to_mm:
        df["x"] = df["x"] * 1000.0
        df["y"] = df["y"] * 1000.0
        df["z"] = df["z"] * 1000.0

    # is_wall 标记：有速度时用速度模判定，没有速度时默认全是壁面
    if {"u", "v", "w"}.issubset(df.columns):
        speed = np.sqrt(df["u"] ** 2 + df["v"] ** 2 + df["w"] ** 2)
        df["is_wall"] = (speed < 1e-6).astype(int)
    else:
        df["is_wall"] = 1

    # 保留优先列 + 其他未识别列
    ordered_cols = available_cols + [c for c in df.columns if c not in available_cols]
    df = df[ordered_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ 清洗完成! 已保存至: {output_path}")
    print(df.head())


def batch_clean_fluent_data(
    root_dir: Path,
    ascii_dir_name: str = "ascii",
    output_dir_name: str = "ascii_clean",
    convert_to_mm: bool = True,
) -> None:
    """
    批量遍历点云数据目录，清洗所有 ASCII 文件。
    目录结构约定:
        root_dir/
            病例编号/
                ascii/            <- Fluent 导出的原始 ASCII 文件
                ascii_clean/      <- 本函数生成的清洗后 CSV (自动创建)
    参数:
        root_dir: 点云根目录（包含多个编号文件夹）
        ascii_dir_name: 原始 ASCII 子目录名
        output_dir_name: 输出 CSV 子目录名
        convert_to_mm: 是否执行坐标单位转换
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        print(f"❌ 根目录不存在: {root_dir}")
        return

    case_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not case_dirs:
        print(f"❌ 未在 {root_dir} 下找到病例子目录。")
        return

    for case_dir in sorted(case_dirs):
        ascii_dir = case_dir / ascii_dir_name
        if not ascii_dir.is_dir():
            continue

        out_dir = case_dir / output_dir_name
        print(f"📂 处理病例 {case_dir.name} -> {out_dir}")
        for ascii_file in sorted(ascii_dir.iterdir()):
            if not ascii_file.is_file():
                continue
            # 文件名本身无扩展名，统一加 .csv 输出
            output_path = out_dir / f"{ascii_file.name}.csv"
            convert_fluent_to_csv(ascii_file, output_path, convert_to_mm=convert_to_mm)


if __name__ == "__main__":
    # 默认从当前目录下的“点云”文件夹批量处理所有病例的 ASCII 数据
    batch_clean_fluent_data(Path("点云"))
