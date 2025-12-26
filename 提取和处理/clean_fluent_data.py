import os
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO

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
    # ascii_in 文件的列名
    "Node Number": "node_id",
    "X [ m ]": "x",
    "Y [ m ]": "y",
    "Z [ m ]": "z",
    "Pressure [ Pa ]": "p",
    "Velocity [ m s^-1 ]": "vel_mag",
    "Velocity u [ m s^-1 ]": "u",
    "Velocity v [ m s^-1 ]": "v",
    "Velocity w [ m s^-1 ]": "w",
}


def _load_ascii_df(input_path: Path) -> pd.DataFrame:
    """根据文件内容读取 ascii 或 ascii_in 数据。
    - ascii: 空格分隔，无表头干扰。
    - ascii_in: 带 [Data] 区段与逗号分隔表头，这里截取表头行开始读取。
    """
    raw_text = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # 尝试定位 ascii_in 的逗号分隔表头
    header_idx = None
    for idx, line in enumerate(raw_text):
        if "Node Number" in line and "," in line:
            header_idx = idx
            break
        if line.strip().lower() == "[data]" and idx + 1 < len(raw_text):
            # [Data] 下一行通常是表头
            header_idx = idx + 1
            break

    try:
        if header_idx is not None:
            sliced = "\n".join(raw_text[header_idx:])
            df = pd.read_csv(StringIO(sliced), sep=",", engine="python")
        else:
            df = pd.read_csv(input_path, sep=r"\s+", engine="python")
    except Exception as e:
        raise ValueError(f"无法读取 {input_path}: {e}") from e

    # 去除列名首尾空格，便于映射
    df.columns = [c.strip() for c in df.columns]
    return df


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
        df = _load_ascii_df(input_path)
    except Exception as e:
        print(f"❌ 读取失败，请检查表头: {e}")
        return

    # 重命名列，统一字段名（大小写、格式差异都统一在这里处理）
    lower_map = {k.lower(): v for k, v in RENAME_MAP.items()}
    rename_dict = {}
    for col in df.columns:
        key = col.strip()
        mapped = RENAME_MAP.get(key) or lower_map.get(key.lower())
        if mapped:
            rename_dict[col] = mapped
    df = df.rename(columns=rename_dict)
    print(f"   - 原始数据列名: {df.columns.tolist()}")

    # 如果有节点编号，清洗后不再保留
    if "node_id" in df.columns:
        df = df.drop(columns=["node_id"])

    # 确保几何坐标存在
    missing_xyz = [c for c in ("x", "y", "z") if c not in df.columns]
    if missing_xyz:
        print(f"❌ 缺少坐标列 {missing_xyz}，跳过该文件。")
        return

    # 可选的常见列顺序，便于后续模型直接读取
    preferred_cols = ["x", "y", "z", "u", "v", "w", "p", "vel_mag", "wss", "wss_x", "wss_y", "wss_z"]
    available_cols = [c for c in preferred_cols if c in df.columns]

    # 壁面 ascii 可能缺少速度列，补齐为 0 便于后续与内部点合并
    for vel_col in ("u", "v", "w", "vel_mag"):
        if vel_col not in df.columns:
            df[vel_col] = 0.0
            if vel_col not in available_cols:
                available_cols.append(vel_col)

    # 坐标单位转换：米 -> 毫米
    if convert_to_mm:
        df["x"] = df["x"] * 1000.0
        df["y"] = df["y"] * 1000.0
        df["z"] = df["z"] * 1000.0

    # ascii_in 缺失壁面剪切力，统一补 0，保证列齐全
    for shear_col in ("wss", "wss_x", "wss_y", "wss_z"):
        if shear_col not in df.columns:
            df[shear_col] = 0.0

    # is_wall 标记：有速度时用速度模判定，没有速度时默认全是壁面
    if {"u", "v", "w"}.issubset(df.columns):
        speed = np.sqrt(df["u"] ** 2 + df["v"] ** 2 + df["w"] ** 2)
        df["is_wall"] = (speed < 1e-6).astype(int)
    else:
        df["is_wall"] = 1

    # 去除不需要的面积字段，便于后续内外点合并
    if "face_area" in df.columns:
        df = df.drop(columns=["face_area"])

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
    ascii_in_dir_name: str = "ascii_in",
    output_dir_name: str = "ascii_clean",
    output_in_dir_name: str = "ascii_in_clean",
    convert_to_mm: bool = True,
) -> None:
    """
    批量遍历点云数据目录，清洗所有 ASCII 文件。
    目录结构约定:
        root_dir/
            病例编号/
                ascii/            <- Fluent 导出的原始 ASCII 文件
                ascii_clean/      <- 本函数生成的清洗后 CSV (自动创建)
                ascii_in/         <- 内部点 ASCII（若存在）
                ascii_in_clean/   <- 内部点清洗后的 CSV (自动创建)
    参数:
        root_dir: 点云根目录（包含多个编号文件夹）
        ascii_dir_name: 原始 ASCII 子目录名
        ascii_in_dir_name: 内部点 ASCII 子目录名
        output_dir_name: 输出 CSV 子目录名
        output_in_dir_name: 内部点输出 CSV 子目录名
        convert_to_mm: 是否执行坐标单位转换
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        print(f"❌ 根目录不存在: {root_dir}")
        return

    # 需要处理的目录对 (输入子目录, 输出子目录)
    dir_pairs = [(ascii_dir_name, output_dir_name)]
    if ascii_in_dir_name:
        dir_pairs.append((ascii_in_dir_name, output_in_dir_name))

    case_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not case_dirs:
        print(f"❌ 未在 {root_dir} 下找到病例子目录。")
        return

    for case_dir in sorted(case_dirs):
        for in_dir_name, out_dir_name in dir_pairs:
            ascii_dir = case_dir / in_dir_name
            if not ascii_dir.is_dir():
                continue

            out_dir = case_dir / out_dir_name
            print(f"📂 处理病例 {case_dir.name} : {in_dir_name} -> {out_dir}")
            for ascii_file in sorted(ascii_dir.iterdir()):
                if not ascii_file.is_file():
                    continue
                # 输出统一加 .csv，避免重复后缀
                output_path = out_dir / f"{ascii_file.stem}.csv"
                convert_fluent_to_csv(ascii_file, output_path, convert_to_mm=convert_to_mm)


if __name__ == "__main__":
    # 默认从当前目录下的“点云”文件夹批量处理所有病例的 ASCII 数据
    batch_clean_fluent_data(Path("data"))
