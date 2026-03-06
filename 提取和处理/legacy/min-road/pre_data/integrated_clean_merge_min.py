import os
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO
from scipy.spatial import cKDTree
import argparse

# 将 Fluent 输出列名映射到统一字段
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
    """根据文件内容读取 ascii 或 ascii_in 数据。"""
    raw_text = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # 尝试定位 ascii_in 的逗号分隔表头
    header_idx = None
    for idx, line in enumerate(raw_text):
        if "Node Number" in line and "," in line:
            header_idx = idx
            break
        if line.strip().lower() == "[data]" and idx + 1 < len(raw_text):
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

    df.columns = [c.strip() for c in df.columns]
    return df


def clean_cfd_data_in_memory(df: pd.DataFrame, convert_to_mm: bool = True) -> pd.DataFrame:
    """
    在内存中清洗单个 CFD 数据的 DataFrame。
    
    参数:
        df: 原始数据 DataFrame
        convert_to_mm: 是否将坐标从米转换为毫米
    
    返回:
        清洗后的 DataFrame
    """
    # 重命名列
    lower_map = {k.lower(): v for k, v in RENAME_MAP.items()}
    rename_dict = {}
    for col in df.columns:
        key = col.strip()
        mapped = RENAME_MAP.get(key) or lower_map.get(key.lower())
        if mapped:
            rename_dict[col] = mapped
    df = df.rename(columns=rename_dict)
    
    # 移除节点编号
    if "node_id" in df.columns:
        df = df.drop(columns=["node_id"])
    
    # 确保几何坐标存在
    missing_xyz = [c for c in ("x", "y", "z") if c not in df.columns]
    if missing_xyz:
        raise ValueError(f"缺少坐标列 {missing_xyz}")
    
    # 补齐速度列
    for vel_col in ("u", "v", "w", "vel_mag"):
        if vel_col not in df.columns:
            df[vel_col] = 0.0
    
    # 坐标单位转换：米 -> 毫米
    if convert_to_mm:
        df["x"] = df["x"] * 1000.0
        df["y"] = df["y"] * 1000.0
        df["z"] = df["z"] * 1000.0
    
    # 补齐壁面剪切力
    for shear_col in ("wss", "wss_x", "wss_y", "wss_z"):
        if shear_col not in df.columns:
            df[shear_col] = 0.0
    
    # is_wall 标记
    if {"u", "v", "w"}.issubset(df.columns):
        speed = np.sqrt(df["u"] ** 2 + df["v"] ** 2 + df["w"] ** 2)
        df["is_wall"] = (speed < 1e-6).astype(int)
    else:
        df["is_wall"] = 1
    
    # 移除面积字段
    if "face_area" in df.columns:
        df = df.drop(columns=["face_area"])
    
    # 整理列顺序
    preferred_cols = ["x", "y", "z", "u", "v", "w", "p", "vel_mag", "wss", "wss_x", "wss_y", "wss_z"]
    available_cols = [c for c in preferred_cols if c in df.columns]
    ordered_cols = available_cols + [c for c in df.columns if c not in available_cols]
    df = df[ordered_cols]
    
    return df


def farthest_point_sampling(points: np.ndarray, n_samples: int, seed: int | None = None) -> np.ndarray:
    """
    最远点采样（Farthest Point Sampling, FPS）算法。
    确保采样点在空间上均匀分布，防止血管几何结构被截断。
    
    优点：空间分布均匀，防止几何截断
    缺点：计算较慢（O(n*m)）
    
    参数:
        points: 点云坐标，形状 (N, 3)
        n_samples: 需要采样的点数
        seed: 随机种子，用于初始点选择
    
    返回:
        采样点的索引数组
    """
    n_points = len(points)
    
    # 如果需要采样的点数大于等于总点数，返回所有索引
    if n_samples >= n_points:
        return np.arange(n_points)
    
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    # 初始化：随机选择第一个点
    sampled_indices = [np.random.randint(n_points)]
    distances = np.full(n_points, np.inf)
    
    # 迭代选择最远点
    for i in range(1, n_samples):
        # 计算所有点到最新采样点的距离
        last_point = points[sampled_indices[-1]]
        dists_to_last = np.sum((points - last_point) ** 2, axis=1)
        
        # 更新每个点到已采样点集的最小距离
        distances = np.minimum(distances, dists_to_last)
        
        # 选择距离最远的点
        farthest_idx = np.argmax(distances)
        sampled_indices.append(farthest_idx)
    
    return np.array(sampled_indices)


def random_sampling(points: np.ndarray, n_samples: int, seed: int | None = None) -> np.ndarray:
    """
    随机采样算法。
    快速简单，但可能导致空间分布不均。
    
    优点：速度快（O(n)）
    缺点：可能空间分布不均，有截断风险
    
    参数:
        points: 点云坐标，形状 (N, 3)（此参数仅为保持接口一致，实际未使用）
        n_samples: 需要采样的点数
        seed: 随机种子
    
    返回:
        采样点的索引数组
    """
    n_points = len(points)
    
    # 如果需要采样的点数大于等于总点数，返回所有索引
    if n_samples >= n_points:
        return np.arange(n_points)
    
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    # 随机采样
    sampled_indices = np.random.choice(n_points, size=n_samples, replace=False)
    
    return sampled_indices


def stratified_sampling_by_distance(
    surface_df: pd.DataFrame,
    inner_df: pd.DataFrame,
    boundary_threshold: float = 2.0,
    boundary_core_ratio: tuple[float, float] = (0.7, 0.3),
    target_total: int = 40000,
    sampling_method: str = "fps",
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于距离的分层降采样合并。
    
    采样策略：
    1. 优先级1：无条件保留所有壁面点
    2. 优先级2：内部点分层
       - 近壁层 (< 2.0mm)：按 7:3 比例优先分配预算
       - 核心层 (>= 2.0mm)：次优先
    3. 动态补位：如果某层点数不足配额，将多余配额转给另一层
    4. 采样方法：
       - FPS：最远点采样，空间均匀，防止截断（推荐，但较慢）
       - Random：随机采样，速度快，但可能分布不均
    
    参数:
        surface_df: 壁面点数据
        inner_df: 内部点数据
        boundary_threshold: 近壁区阈值（mm）
        boundary_core_ratio: (近壁层比例, 核心层比例)，默认 (0.7, 0.3)
        target_total: 目标总点数
        sampling_method: 采样方法，"fps" 或 "random"
        seed: 随机种子
    
    返回:
        (merged_df, sampled_inner_df): 合并后的完整数据 和 降采样后的内部点数据
    """
    coord_cols = ['x', 'y', 'z']
    
    surface_coords = surface_df[coord_cols].values
    inner_coords = inner_df[coord_cols].values
    
    print(f"  壁面点数: {len(surface_df)}, 内部点数: {len(inner_df)}")
    
    # 优先级1：无条件保留所有壁面点
    n_surface = len(surface_df)
    remaining_budget = target_total - n_surface
    
    if remaining_budget <= 0:
        print(f"  ⚠️ 壁面点已达 {n_surface}，超过目标点数 {target_total}，仅保留壁面点")
        return surface_df.copy(), pd.DataFrame()
    
    print(f"  剩余预算: {remaining_budget} 个点")
    
    # 优先级2：使用 KDTree 计算内部点到壁面的距离
    tree = cKDTree(surface_coords)
    distances, _ = tree.query(inner_coords, k=1)
    
    # 分层：近壁层 vs 核心层
    boundary_mask = distances < boundary_threshold
    boundary_indices = np.where(boundary_mask)[0]
    core_indices = np.where(~boundary_mask)[0]
    
    n_boundary_available = len(boundary_indices)
    n_core_available = len(core_indices)
    
    print(f"  近壁层点数: {n_boundary_available}, 核心层点数: {n_core_available}")
    
    # 动态配额分配：7:3 比例，实现动态补位
    boundary_ratio, core_ratio = boundary_core_ratio
    n_boundary_quota = int(remaining_budget * boundary_ratio)
    n_core_quota = int(remaining_budget * core_ratio)
    
    # 动态补位逻辑
    if n_boundary_available < n_boundary_quota:
        # 近壁层不足，将多余配额转给核心层
        surplus = n_boundary_quota - n_boundary_available
        n_boundary_final = n_boundary_available
        n_core_final = min(n_core_available, n_core_quota + surplus)
        print(f"  ⚠️ 近壁层不足配额，转移 {surplus} 个配额到核心层")
    elif n_core_available < n_core_quota:
        # 核心层不足，将多余配额转给近壁层
        surplus = n_core_quota - n_core_available
        n_core_final = n_core_available
        n_boundary_final = min(n_boundary_available, n_boundary_quota + surplus)
        print(f"  ⚠️ 核心层不足配额，转移 {surplus} 个配额到近壁层")
    else:
        # 两层都充足
        n_boundary_final = n_boundary_quota
        n_core_final = n_core_quota
    
    # 确保不超过总预算
    total_allocated = n_boundary_final + n_core_final
    if total_allocated > remaining_budget:
        # 按比例缩减
        scale = remaining_budget / total_allocated
        n_boundary_final = int(n_boundary_final * scale)
        n_core_final = remaining_budget - n_boundary_final
    
    print(f"  最终分配: 近壁层 {n_boundary_final}/{n_boundary_available}, "
          f"核心层 {n_core_final}/{n_core_available}")
    
    # 选择采样函数
    if sampling_method.lower() == "fps":
        sampling_func = farthest_point_sampling
        method_name = "FPS"
    elif sampling_method.lower() == "random":
        sampling_func = random_sampling
        method_name = "随机"
    else:
        raise ValueError(f"不支持的采样方法: {sampling_method}，请使用 'fps' 或 'random'")
    
    # 进行采样
    sampled_indices = []
    
    # 采样近壁层
    if n_boundary_final > 0 and n_boundary_available > 0:
        boundary_coords = inner_coords[boundary_indices]
        if n_boundary_final < n_boundary_available:
            print(f"  执行近壁层{method_name}采样...")
            sampled_idx = sampling_func(boundary_coords, n_boundary_final, seed)
            sampled_boundary = boundary_indices[sampled_idx]
        else:
            # 全部保留
            sampled_boundary = boundary_indices
        sampled_indices.append(sampled_boundary)
    
    # 采样核心层
    if n_core_final > 0 and n_core_available > 0:
        core_coords = inner_coords[core_indices]
        if n_core_final < n_core_available:
            print(f"  执行核心层{method_name}采样...")
            sampled_idx = sampling_func(core_coords, n_core_final, seed)
            sampled_core = core_indices[sampled_idx]
        else:
            # 全部保留
            sampled_core = core_indices
        sampled_indices.append(sampled_core)
    
    # 合并采样的内部点索引
    if sampled_indices:
        sampled_inner_indices = np.concatenate(sampled_indices)
        sampled_inner_df = inner_df.iloc[sampled_inner_indices].copy()
    else:
        sampled_inner_df = pd.DataFrame()
    
    # 对齐列
    all_cols = list(surface_df.columns)
    for col in sampled_inner_df.columns:
        if col not in all_cols:
            all_cols.append(col)
    
    surface_df_aligned = surface_df.reindex(columns=all_cols)
    sampled_inner_df_aligned = sampled_inner_df.reindex(columns=all_cols)
    
    # 合并并打乱
    merged = pd.concat([surface_df_aligned, sampled_inner_df_aligned], ignore_index=True)
    if seed is not None:
        merged = merged.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        merged = merged.sample(frac=1).reset_index(drop=True)
    
    print(f"  ✅ 合并后总点数: {len(merged)} (目标: {target_total})")
    print(f"  预算利用率: {len(merged)/target_total*100:.1f}%")
    
    return merged, sampled_inner_df


def process_single_case_integrated(
    case_dir: Path,
    surface_dir: str = "ascii",
    inner_dir: str = "ascii_in",
    output_dir: str = "ascii_merged",
    replace_ascii_in: bool = True,
    seed: int | None = 1234,
    boundary_threshold: float = 2.0,
    boundary_core_ratio: tuple[float, float] = (0.7, 0.3),
    target_total: int = 40000,
    sampling_method: str = "fps",
    convert_to_mm: bool = True,
) -> None:
    """
    整合流程：读取原始数据 -> 内存清洗 -> 分层采样合并 -> 输出 -> 替换原始内部点文件。
    
    采样策略：
    - 优先级1：无条件保留所有壁面点
    - 优先级2：内部点按 7:3 比例分配给近壁层和核心层
    - 采样方法可选：FPS（推荐，质量高但较慢）或 Random（快速但质量一般）
    - 动态补位：如果某层不足配额，将多余配额转给另一层
    
    参数:
        case_dir: 病例目录
        surface_dir: 壁面点原始目录名（默认 ascii）
        inner_dir: 内部点原始目录名（默认 ascii_in）
        output_dir: 合并输出目录名（默认 ascii_merged）
        replace_ascii_in: 是否用降采样后的内部点替换 ascii_in 中的原始文件
        seed: 随机种子
        boundary_threshold: 近壁区阈值（mm），默认 2.0
        boundary_core_ratio: (近壁层比例, 核心层比例)，默认 (0.7, 0.3)
        target_total: 目标总点数，默认 40000
        sampling_method: 采样方法，"fps"（最远点采样）或 "random"（随机采样）
        convert_to_mm: 是否转换坐标单位为毫米
    """
    case_dir = Path(case_dir)
    surface_path = case_dir / surface_dir
    inner_path = case_dir / inner_dir
    
    if not surface_path.is_dir() or not inner_path.is_dir():
        print(f"⚠️  跳过 {case_dir.name}: 缺少 {surface_dir} 或 {inner_dir}")
        return
    
    # 匹配文件编号
    surface_files = {}
    for p in surface_path.glob("*.csv"):
        stem = p.stem
        if '-' in stem:
            number_part = stem.split('-')[-1]
            if number_part.isdigit():
                surface_files[number_part] = p
    
    # 对于 ascii 文件夹中可能没有扩展名的文件
    for p in surface_path.iterdir():
        if p.is_file() and p.suffix == '':
            stem = p.stem
            if '-' in stem:
                number_part = stem.split('-')[-1]
                if number_part.isdigit():
                    surface_files[number_part] = p
    
    inner_files = {}
    for p in inner_path.glob("*.csv"):
        stem = p.stem
        if '-' in stem:
            number_part = stem.split('-')[-1]
            if number_part.isdigit():
                inner_files[number_part] = p
    
    # 对于 ascii_in 文件夹中可能没有扩展名的文件
    for p in inner_path.iterdir():
        if p.is_file() and p.suffix == '':
            stem = p.stem
            if '-' in stem:
                number_part = stem.split('-')[-1]
                if number_part.isdigit():
                    inner_files[number_part] = p
    
    common_keys = sorted(set(surface_files) & set(inner_files))
    if not common_keys:
        print(f"⚠️  跳过 {case_dir.name}: 未找到同编号的壁面与内部点文件")
        return
    
    # 创建输出目录
    out_dir = case_dir / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 开始处理病例: {case_dir.name}")
    print(f"   近壁区阈值: {boundary_threshold}mm")
    print(f"   预算分配比例: 近壁层 {boundary_core_ratio[0]*100:.0f}% : 核心层 {boundary_core_ratio[1]*100:.0f}%")
    print(f"   目标总点数: {target_total}")
    
    # 显示采样方法
    method_display = {
        "fps": "最远点采样 (FPS) - 质量高，速度慢",
        "random": "随机采样 (Random) - 速度快，质量一般"
    }
    print(f"   采样方法: {method_display.get(sampling_method.lower(), sampling_method)}")
    print(f"   替换原始 ascii_in: {'是' if replace_ascii_in else '否'}")
    
    for key in common_keys:
        try:
            print(f"\n🔄 处理编号 {key}...")
            
            # 1. 读取并清洗壁面点数据
            print(f"   读取壁面数据: {surface_files[key].name}")
            surface_raw_df = _load_ascii_df(surface_files[key])
            surface_df = clean_cfd_data_in_memory(surface_raw_df, convert_to_mm=convert_to_mm)
            
            # 2. 读取并清洗内部点数据
            print(f"   读取内部数据: {inner_files[key].name}")
            inner_raw_df = _load_ascii_df(inner_files[key])
            inner_df = clean_cfd_data_in_memory(inner_raw_df, convert_to_mm=convert_to_mm)
            
            # 3. 分层降采样合并
            print(f"   执行分层降采样合并...")
            merged_df, sampled_inner_df = stratified_sampling_by_distance(
                surface_df, 
                inner_df, 
                boundary_threshold=boundary_threshold,
                boundary_core_ratio=boundary_core_ratio,
                target_total=target_total,
                sampling_method=sampling_method,
                seed=seed
            )
            
            # 4. 保存合并结果
            output_name = inner_files[key].stem
            merged_output_path = out_dir / f"{output_name}.csv"
            merged_df.to_csv(merged_output_path, index=False)
            print(f"   ✅ 合并文件已保存: {merged_output_path.name}")
            
            # 5. 可选：替换原始 ascii_in 文件
            if replace_ascii_in and len(sampled_inner_df) > 0:
                original_inner_path = inner_files[key]
                original_size = original_inner_path.stat().st_size / (1024 * 1024)  # MB
                
                # 保存降采样后的内部点数据
                sampled_inner_df.to_csv(original_inner_path, index=False)
                new_size = original_inner_path.stat().st_size / (1024 * 1024)  # MB
                
                print(f"   🔄 已替换原始内部点文件: {original_inner_path.name}")
                print(f"      原始大小: {original_size:.2f}MB -> 新大小: {new_size:.2f}MB")
                print(f"      节省空间: {original_size - new_size:.2f}MB ({(1 - new_size/original_size)*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ 处理编号 {key} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 {case_dir.name} 处理完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="整合清洗与合并流程，支持 FPS 和随机采样",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（默认 FPS 采样，7:3 比例，会替换原始文件）
  python integrated_clean_merge_min.py "../data/AAA/rupture/FENG_LI_XIN"
  
  # 使用随机采样（速度快）
  python integrated_clean_merge_min.py "../data/AAA/rupture/FENG_LI_XIN" --sampling-method random
  
  # 自定义比例（8:2）
  python integrated_clean_merge_min.py "../data/AAA/rupture/FENG_LI_XIN" --boundary-ratio 0.8 --core-ratio 0.2
  
  # 不替换原始文件
  python integrated_clean_merge_min.py "../data/AAA/rupture/FENG_LI_XIN" --no-replace
        """
    )
    parser.add_argument(
        "case_dir", 
        type=str, 
        help="病例目录路径"
    )
    parser.add_argument(
        "--no-replace", 
        action="store_true",
        help="不替换 ascii_in 中的原始文件（默认会替换以节省空间）"
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        choices=["fps", "random"],
        default="fps",
        help="采样方法：fps=最远点采样（质量高，较慢），random=随机采样（速度快），默认 fps"
    )
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=2.0,
        help="近壁区阈值（mm），默认 2.0"
    )
    parser.add_argument(
        "--boundary-ratio",
        type=float,
        default=0.7,
        help="近壁层预算分配比例，默认 0.7 (70%%)"
    )
    parser.add_argument(
        "--core-ratio",
        type=float,
        default=0.3,
        help="核心层预算分配比例，默认 0.3 (30%%)"
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=40000,
        help="目标总点数，默认 40000"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="随机种子，默认 1234"
    )
    
    args = parser.parse_args()
    
    # 验证比例之和
    if abs(args.boundary_ratio + args.core_ratio - 1.0) > 0.01:
        print(f"⚠️  警告：近壁层比例 ({args.boundary_ratio}) + 核心层比例 ({args.core_ratio}) != 1.0")
        print(f"   将自动归一化为: {args.boundary_ratio/(args.boundary_ratio + args.core_ratio):.2f} : {args.core_ratio/(args.boundary_ratio + args.core_ratio):.2f}")
        total = args.boundary_ratio + args.core_ratio
        args.boundary_ratio /= total
        args.core_ratio /= total
    
    process_single_case_integrated(
        case_dir=Path(args.case_dir),
        replace_ascii_in=not args.no_replace,
        boundary_threshold=args.boundary_threshold,
        boundary_core_ratio=(args.boundary_ratio, args.core_ratio),
        target_total=args.target_total,
        sampling_method=args.sampling_method,
        seed=args.seed,
    )

