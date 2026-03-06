#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline 配置文件

统一管理所有处理流程的配置参数。

环境变量支持（用于集群环境）:
  - PIPELINE_DATA_ROOT: 数据根目录路径
  - PIPELINE_MODE: 处理模式 (debug/production)
"""

import os
from pathlib import Path

# ============================================================================
# 路径配置
# ============================================================================

# 获取项目根目录
_SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = _SCRIPT_DIR.parent

# 数据根目录（支持环境变量覆盖）
_env_data_root = os.environ.get("PIPELINE_DATA_ROOT")
if _env_data_root:
    DATA_ROOT = Path(_env_data_root)
else:
    DATA_ROOT = PROJECT_ROOT / "data_new"

# ============================================================================
# 数据源配置
# ============================================================================

DATA_SOURCES = {
    "AG/fast": {"enabled": True, "description": "动脉移植物-快速增长"},
    "AG/slow": {"enabled": False, "description": "动脉移植物-慢速增长"},
    "AAA/rupture": {"enabled": False, "description": "腹主动脉瘤-破裂"},
    "AAA/unrupture": {"enabled": False, "description": "腹主动脉瘤-未破裂"},
    "ILO/sq": {"enabled": False, "description": "髂支闭塞-SQ"},
    "ILO/sh": {"enabled": False, "description": "髂支闭塞-SH"},
}

# ============================================================================
# 输入目录配置
# ============================================================================

# 壁面点数据目录
SURFACE_DIR = "ascii"

# 内部点数据目录
INNER_DIR = "ascii_in"

# 边界条件目录（固定为 Global_conditions）
BC_DIR = "Global_conditions"

# ============================================================================
# 输出目录配置
# ============================================================================

# 所有输出都在 processed/ 子目录下
OUTPUT_BASE = "processed"

# 合并降采样后的数据目录
MERGED_DIR = f"{OUTPUT_BASE}/merged"

# 添加几何特征后的数据目录
FEATURES_DIR = f"{OUTPUT_BASE}/features"

# 坐标系归一化后的数据目录
COORD_NORMALIZED_DIR = f"{OUTPUT_BASE}/coord_normalized"

# 特征归一化后的数据目录
NORMALIZED_DIR = f"{OUTPUT_BASE}/normalized"

# 图数据目录
GRAPHS_DIR = f"{OUTPUT_BASE}/graphs"

# ============================================================================
# 降采样配置
# ============================================================================

SAMPLING_CONFIG = {
    # 目标总点数
    "target_total_points": 40000,
    
    # 壁面点保留比例（1.0 = 全保留）
    "wall_ratio": 1.0,
    
    # 近壁区阈值（mm）
    "boundary_threshold": 2.0,
    
    # 预算分配比例 (近壁层, 核心层)
    "boundary_core_ratio": (0.7, 0.3),
    
    # 采样方法: "fps" (最远点采样), "random" (随机采样), 或 "hybrid" (混合采样)
    "sampling_method": "hybrid",
    
    # 混合采样参数（仅当 sampling_method="hybrid" 时生效）
    # FPS 占比: 0.2 = 20%（推荐），可调整为 0.3 或 0.4 以增强空间覆盖
    "hybrid_fps_ratio": 0.2,
    
    # 随机种子
    "seed": 1234,
}

# ============================================================================
# 边界条件配置
# ============================================================================

# 血液密度 (kg/m³)，用于质量流量转体积流量
BLOOD_DENSITY = 1060

# 边界条件文件映射（固定映射，不再需要 BC_Flag）
BC_FILE_MAPPING = {
    "inlet": {
        "primary": "vf-in-rfile.out",      # 体积流量（优先）
        "fallback": "report-file-2.out",   # 质量流量（备用，需转换）
    },
    "O1": "p-outle-rfile.out",   # 左外髂支出口压力
    "O2": "p-outli-rfile.out",   # 左内髂支出口压力
    "O3": "p-outre-rfile.out",   # 右外髂支出口压力
    "O4": "p-outri-rfile.out",   # 右内髂支出口压力
}

# ============================================================================
# 归一化配置
# ============================================================================

NORMALIZATION_CONFIG = {
    # 保持不变的特征
    "keep_unchanged": [
        "Abscissa",      # 已归一化到 [0, 1]
        "Tangent_X",     # 单位向量分量
        "Tangent_Y",
        "Tangent_Z",
        "is_wall",       # 二值标记
    ],
    
    # 使用 min-max 归一化的特征
    "min_max": ["NormRadius"],
    
    # 使用 Z-score 标准化的特征
    "z_score": [
        "Curvature",
        "u", "v", "w",
        "p",
        "vel_mag",
        "wss", "wss_x", "wss_y", "wss_z",
    ],
    
    # 边界条件缩放参数
    "bc_scaling": {
        # 入口流量: Q_in × 1e5 → 0.5~5.0
        "inlet": {"scale_factor": 1e5},

        # 出口压力默认使用统计量标准化，避免把单批数据先验硬编码进 pipeline。
        # 可选 strategy:
        #   - "z_score": 使用全局 mean/std
        #   - "fixed": 使用 offset/scale 的固定缩放
        "outlet_pressure": {
            "strategy": "z_score",
            "offset": 15000,
            "scale": 1000,
        },
    }
}

# 坐标系归一化配置
COORD_NORMALIZATION_CONFIG = {
    # PCA 第一主成分对齐到的目标轴，可选 "x" / "y" / "z"
    "principal_axis_target": "z",
}

# ============================================================================
# 图构建配置
# ============================================================================

GRAPH_CONFIG = {
    # KNN 邻居数
    "k_neighbors": 6,
}

# ============================================================================
# 特征维度配置
# ============================================================================

# 节点特征名称（存储在 data.x 中，逐点不同）
NODE_FEATURE_NAMES = [
    "x", "y", "z",                                     # 坐标 (3)
    "Abscissa", "NormRadius", "Curvature",              # 几何标量 (3)
    "Tangent_X", "Tangent_Y", "Tangent_Z",              # 切线向量 (3)
    "is_wall",                                          # 壁面标记 (1)
]
NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)  # = 10

# 全局条件名称（存储在 data.global_cond 中，整个图共享）
GLOBAL_COND_NAMES = [
    "t_norm",                                           # 归一化时间 (1)
    "BC_Inlet",                                         # 入口体积流量 (1)
    "BC_O1", "BC_O2", "BC_O3", "BC_O4",               # 出口压力 (4)
]
GLOBAL_COND_DIM = len(GLOBAL_COND_NAMES)  # = 6

# 目标输出名称
TARGET_NAMES = ["u", "v", "w", "p"]
TARGET_DIM = len(TARGET_NAMES)  # = 4

# 模型输入维度（节点特征 + 全局条件拼接后）
MODEL_INPUT_DIM = NODE_FEATURE_DIM + GLOBAL_COND_DIM  # = 16

# 节点特征中各组的索引范围
FEATURE_INDICES = {
    "coords": (0, 3),       # x, y, z
    "geom_scalar": (3, 6),  # Abscissa, NormRadius, Curvature
    "tangent": (6, 9),      # Tangent_X, Tangent_Y, Tangent_Z
    "is_wall": (9, 10),     # is_wall
}

# ============================================================================
# 处理模式配置
# ============================================================================

# debug: 保留中间文件，便于检查
# production: 只保留最终输出
# 支持环境变量覆盖
MODE = os.environ.get("PIPELINE_MODE", "debug")

# ============================================================================
# 辅助函数
# ============================================================================

def get_enabled_sources():
    """获取已启用的数据源列表"""
    return [source for source, config in DATA_SOURCES.items() if config["enabled"]]


def get_case_dirs(data_root=None, sources=None):
    """
    获取所有病例目录
    
    参数:
        data_root: 数据根目录，默认使用 DATA_ROOT
        sources: 数据源列表，默认使用已启用的数据源
    
    返回:
        病例目录路径列表
    """
    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)
    
    if sources is None:
        sources = get_enabled_sources()
    
    case_dirs = []
    for source in sources:
        source_path = data_root / source
        if not source_path.exists():
            continue
        
        for case_dir in source_path.iterdir():
            if case_dir.is_dir() and not case_dir.name.startswith('.'):
                case_dirs.append(case_dir)
    
    return sorted(case_dirs, key=lambda p: p.name)


if __name__ == "__main__":
    # 测试配置
    print("Pipeline 配置信息")
    print("=" * 50)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据根目录: {DATA_ROOT}")
    print(f"已启用数据源: {get_enabled_sources()}")
    print(f"目标点数: {SAMPLING_CONFIG['target_total_points']}")
    print(f"处理模式: {MODE}")
    
    print("\n病例目录:")
    for case_dir in get_case_dirs():
        print(f"  - {case_dir.relative_to(DATA_ROOT)}")
