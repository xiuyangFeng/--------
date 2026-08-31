#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pipeline_wss_min 全局配置。

所有可调开关集中在这里；预处理与样本装配都从这里读参数。
配置驱动的目的：坐标/稀疏化/时间步/特征掩码 都能只改配置就跑，不用改代码。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data_new"           # 原始 CFD 根目录（只读）
OUT_ROOT = PROJECT_ROOT / "data_wss_min"       # 新流程产物（独立目录，不动旧数据）
DEFAULT_SPLIT_NAME = "split_AG_wss_min_v1"

# 默认数据范围：AG 队列的 fast 与 slow 子集。
COHORTS: Dict[str, str] = {
    "AG/fast": "AG/fast",
    "AG/slow": "AG/slow",
}

# 网格坐标单位统一：Fluent ASCII 多为 SI 米，中心线为毫米，但个别病例网格单位异常。
# 默认按中心线包围盒逐病例反推“网格坐标→毫米”因子，使近壁阈值、壁面距离和
# 局部半径同量纲；异常因子会在审计中标记待人工复核。
MESH_COORD_TO_MM = 1000.0  # 固定单位模式下的回退因子

# 已由“权威 STL ↔ Fluent 壁面”同坐标核验确认的逐病例单位覆盖。
# 不能继续用中心线包围盒反推这些病例的单位；中心线略短于完整表面时会把
# 正确的米→毫米因子错误压缩。键使用 data_new 下的 canonical 相对路径。
UNIT_CASE_OVERRIDES: Dict[str, Dict] = {
    "ILO/YANG_YU_QING-1/before": {
        "mode": "fixed",
        "fixed_factor": 1000.0,
        "reason": "2026-08-27 regenerated centerline from authoritative ANG_YU_QINF-sq.stl.stl",
    },
}

# 原始 CFD 子文件（相对每个病例目录）
RAW_LAYOUT = {
    "wall_ascii_dir": "ascii",          # 壁面节点：含 wall-shear
    "interior_ascii_dir": "ascii_in",   # 内部 cell：含 velocity
    "centerline_csv": "centerline/centerline_points.csv",
    "centerline_vtp": "centerline/centerline.vtp",
    "inlet_waveform": "Global_conditions/vf-in-rfile.out",  # 入口流量波形 -> 峰值收缩期
}

# ascii 文件的原始列名（去空格后）
WALL_COLUMNS = {
    "id": "nodenumber",
    "x": "x-coordinate", "y": "y-coordinate", "z": "z-coordinate",
    "pressure": "pressure",
    "wss": "wall-shear",
    "wss_x": "x-wall-shear", "wss_y": "y-wall-shear", "wss_z": "z-wall-shear",
}
INTERIOR_COLUMNS = {
    "id": "cellnumber",
    "x": "x-coordinate", "y": "y-coordinate", "z": "z-coordinate",
    "pressure": "pressure",
    "vel_mag": "velocity-magnitude",
    "u": "x-velocity", "v": "y-velocity", "w": "z-velocity",
}
CENTERLINE_COLUMNS = {
    "x": "x", "y": "y", "z": "z",
    "abscissa": "Abscissas",
    "radius": "MaximumInscribedSphereRadius",
    "curvature": "Curvature",
    "tan_x": "Tangent_X", "tan_y": "Tangent_Y", "tan_z": "Tangent_Z",
}


@dataclass
class UnitConfig:
    """网格坐标 -> 毫米 的单位统一策略。"""
    mode: str = "auto_centerline"   # 'auto_centerline' 逐病例用中心线反推 | 'fixed' 固定因子
    fixed_factor: float = 1000.0    # mode='fixed' 时使用
    # |log10(factor) - 3| 超过该阈值 -> 标记单位异常（正常米->毫米 factor≈1000）
    anomaly_log10_tol: float = 0.5
    # --- 覆盖范围不一致鲁棒兜底 ---
    # 比值法 factor = cl_diag / wall_diag 只在“壁面与中心线覆盖同一段血管”时成立。
    # 个别病例中心线只描远端一段、壁面网格却含整条近端主动脉，比值法会把 factor
    # 压到真实值的一半，坐标缩放/配准全线偏移。此时改用“米制整十次幂”兜底：
    # 选使壁面对角线最接近生理尺度的 10^k 作 factor，并标记 extent_mismatch 待复核。
    phys_diag_mm: float = 250.0     # 主动脉-髂动脉包围盒对角线的典型量级
    # 比值法结果与整十次幂偏离超过该倍数 -> 判定覆盖范围不一致，改用整十次幂
    ratio_trust: float = 1.5


@dataclass
class RegistrationConfig:
    """解剖学刚性配准（中心线主导）。"""
    # v4：由原始 STL 自动选取分叉中心、近端主干和双髂支端点建立有符号坐标架。
    # +Z 恒指向近端主动脉（故髂支在 -Z，下方）；+X 用原始 STL 世界 +X 定号。
    frame_mode: str = "stl_landmarks"       # 'stl_landmarks' | 'legacy_centerline'
    superior_axis_sign: str = "trunk_positive"
    lr_world_axis: str = "x"
    # STL 末端切片：分别在无符号主轴两端取最外侧比例，比较横向双支展开度。
    landmark_terminal_quantile: float = 0.12
    landmark_min_side_points: int = 30
    landmark_min_fork_ratio: float = 1.10
    landmark_min_lr_sep: float = 1.25
    # 中心线与 STL/壁面明显只差纯平移时自动修复。只有偏移超过一条血管尺度
    # 且平移后中心线确实落回管腔附近才接受，避免正常弯曲病例被误校正。
    centerline_translation_repair: bool = True
    centerline_translation_trigger_diag_frac: float = 0.80
    centerline_translation_max_p50_diag_frac: float = 0.08
    centerline_translation_max_p90_diag_frac: float = 0.16
    # 平移原点：默认使用 flow_divider（三臂等权三叉连接点）作为解剖原点。
    # 可切回 bifurcation 使用 VMTK DistToBifurcation≈0 分叉区域普通均值。
    # 若缺 VTP 拓扑或分叉数组，自动退回 wall/all/centerline 几何中心。
    center_on: str = "flow_divider"         # 'bifurcation' | 'flow_divider' | 'wall' | 'all' | 'centerline'
    # 主轴目标：中心线 inlet->outlet 方向对齐到该轴
    principal_axis_target: str = "z"        # 'x' | 'y' | 'z'
    # 主轴定义：bifurcation 原点下优先用入口端 -> 分叉点的 trunk 方向；
    # 若不可用则退回完整中心线首尾弦向。
    main_axis_mode: str = "inlet_to_bifurcation"  # 'inlet_to_bifurcation' | 'inlet_to_outlet'
    # 主轴鲁棒兜底：若中心线入口端无法清楚区分"单主干侧/双髂支侧"，
    # 用壁面点云 PCA 长轴并按双髂支侧定号，专门处理中心线缺主干或端点落在分支的病例。
    main_axis_wall_fallback: bool = True
    main_axis_wall_fallback_sep_delta: float = 0.10
    main_axis_wall_fallback_min_sep: float = 0.45
    # 滚转（绕主轴）用中心线曲率平面固定，保证朝向唯一可复现
    fix_roll_with_centerline: bool = True
    # roll（左右）轴来源：
    #   'wall_branches'（新默认）——从分叉下游"壁面点"求真正的左右轴。中心线常只描一条
    #      髂支（BranchId 只有 {0,1}、abscissa 单调），而壁面/STL 两条髂支都在、横断面呈双峰，
    #      取两团质心连线为左右轴，稳定且解剖正确。
    #   'branches'（旧，可回退）——用分叉后中心线下游点横向 PCA（依赖中心线含两支，常不成立）。
    #   'curvature'（旧兜底）——中心线曲率平面主方向。
    roll_source: str = "wall_branches"      # 'wall_branches' | 'branches' | 'curvature'
    # roll 符号锚：
    #   'trunk_bending'（新默认）——用主干（入口->分叉）偏离直弦的弯曲方向作 A-P 内在手性参照，
    #      L-R 符号由 (A-P × 主轴) 决定；只依赖主干（恒存在），跨病例自洽，不依赖世界坐标朝向。
    #   'world_axis'（旧，可回退）——把 roll 符号锚到固定世界轴投影（网格世界系不共享时会翻转）。
    roll_sign_mode: str = "trunk_bending"   # 'trunk_bending' | 'world_axis'
    # wall_branches 下游选择：沿主轴投影 > 该比例 * 下游最大投影 记为下游髂支段（避开分叉合流区）。
    downstream_wall_axis_frac: float = 0.05
    # roll 符号置信阈值：|cos(roll, 参照)| 低于此值判定不可靠 -> 记审计 + 回退世界轴锚。
    roll_sign_min_cos: float = 0.2
    # DistToBifurcation <= 此阈值的中心线点用于估计分叉原点。
    bifurcation_distance_mm: float = 2.0
    # flow_divider 原点：将 DistToBifurcation≈0 附近点聚成三臂（主干端 + 左右髂支端），
    # 对三臂中心等权平均，减少采样密度/单侧分支对普通均值的拉偏。
    flow_divider_distance_mm: float = 2.0
    flow_divider_n_clusters: int = 3
    flow_divider_min_cluster_sep_mm: float = 5.0
    # 分支远端候选：在分叉后点中取 DistToBifurcation 的高分位（'branches' 回退路径用）。
    branch_endpoint_quantile: float = 0.85
    # roll 符号锚定到原始世界坐标某一轴（'world_axis' 模式 / 兜底用）。
    roll_sign_world_axis: str = "x"         # 'x' | 'y' | 'z'
    # 主轴方向由 abscissa 最小(入口)->最大(出口) 锚定符号
    orient_by_abscissa: bool = True
    # 配准后横向二次居中：刚性旋转后，若主干低位段在横断面内仍明显偏离中心轴，
    # 只做 X/Y（或非主轴两个方向）平移，把主干低位段压回共同视角中心。
    trunk_centering: bool = True
    trunk_centering_quantile: float = 0.20
    trunk_centering_min_offset_frac: float = 0.08
    trunk_centering_stat: str = "median"     # 'median' | 'mean'
    # 未描主动脉尾巴裁剪：个别病例壁面网格含一段中心线没描到的近端主动脉，
    # 破坏跨病例尺度一致性。配准后（基于中心线，不受尾巴影响）、归一化前，
    # 按中心线入口端轴向覆盖裁掉壁面/内部点里明显超出的那段。
    # 只裁“入口(主动脉)侧”，不动“出口(髂支)侧”（髂支双支自然外展，属正常）；
    # 仅当入口侧超出量 > crop_trigger_overshoot_frac × 中心线轴向跨度 才触发，
    # 对覆盖一致的正常病例不产生任何裁剪。
    crop_untraced_inlet: bool = True
    # 入口裁剪采用中心线入口端局部切平面，而不是全局主轴一刀切；弯曲主干下更稳。
    crop_use_inlet_tangent: bool = True
    crop_axial_margin_frac: float = 0.12
    crop_trigger_overshoot_frac: float = 0.20
    # 保险：擦边小裁剪通常是正常解剖/中心线端点误差，不作为未描入口段处理。
    crop_min_wall_frac: float = 0.08


@dataclass
class NormalizationConfig:
    """坐标逐病例、WSS 全局（用户确认口径）。"""
    # 坐标：逐病例各向同性缩放到 [-1, 1]（保形，视角一致）
    coord_scope: str = "per_case"           # 'per_case' | 'global'
    coord_method: str = "max_abs"           # 'max_abs' -> [-1,1] | 'std'
    # 缩放参照：'wall' 只按壁面范围（默认，跨病例视角一致，不受内部 CFD 流动延伸段
    # 长度影响）；'all' 按壁面+内部（旧口径，会被内部延伸段带偏，壁面只填约半框）。
    # 注：'wall' 下内部点归一化坐标可能超出 ±1（WSS 任务不以内部为输入，无影响）。
    coord_scale_on: str = "wall"            # 'wall' | 'all'
    # WSS 标量目标：全局标准化（保留跨病例物理量级）；
    # 第三轮 clean-data 主线用 train peak-only stats，all-phase 需显式切换。
    wss_scope: str = "global"               # 'global' | 'per_case'
    wss_method: str = "log_z"               # 'z' | 'log_z'（WSS 近似对数正态）
    # 全局统计存放
    global_stats_name: str = "wss_global_stats.json"


@dataclass
class PointTaggingConfig:
    """壁面/近壁/内部 掩码。"""
    # 近壁层定义：内部点到最近壁面点距离 <= 阈值(mm) 记为 near_wall
    near_wall_threshold_mm: float = 1.5
    # 近壁层最多保留多少内部点（0 = 不限）；用于第二条路径(近壁->速度->算子->WSS)
    near_wall_max_points: int = 40000
    # --- 存储控制（bundle 体积）---
    # 壁面 WSS 标量 + pressure 永远存全时间步（小）。以下为可选大块：
    store_wall_wss_vector: bool = True          # 壁面 WSS 矢量分量(全时间步)
    store_interior_coords: bool = True          # 内部点归一化坐标 + 掩码 + 几何(静态)
    store_near_wall_timeseries: bool = False    # 近壁层速度(全时间步) —— 第二条路径用，默认关
    deep_interior_full_timeseries: bool = False # 深层内部速度(全时间步)，默认关


@dataclass
class TimestepConfig:
    """时间步处理：预处理一次性全做，装配阶段再选。"""
    # 导出时间步从 ascii 文件名解析；这里可限制只处理某区间（None=全部导出步）
    step_min: int | None = None
    step_max: int | None = None
    # 峰值收缩期 = 导出区间内 入口波形 vf-in 最大的时间步
    peak_from_waveform: bool = True


# 稀疏化 / 样本装配（配置驱动，方便扫点数）
@dataclass
class SampleConfig:
    name: str = "wss_min_peak_w2000"
    # 稀疏化在 归一化之后 执行
    wall_n_points: int = 2000               # {1500, 2000, 3000} 可扫
    near_wall_n_points: int = 0             # 第二条路径用；0 = 不装配近壁点
    interior_n_points: int = 0             # 深层内部；默认 0
    wall_sampling: str = "fps"              # 'fps' | 'random'
    # 时间步选择：'peak' 单样本 | 'all' 全相位 | 具体步列表
    timesteps: str = "peak"                 # 'peak' | 'all'
    # 网络输入特征（列名）与目标；其余列写入文件但被 mask 屏蔽
    input_features: Tuple[str, ...] = ("x", "y", "z")
    target: str = "wss"
    # 保留但默认屏蔽（可通过改配置解锁）：几何 + 速度等
    retained_masked: Tuple[str, ...] = (
        "dist_to_wall", "abscissa_norm", "local_radius", "curvature",
        "wss_x", "wss_y", "wss_z", "u", "v", "w", "pressure",
    )
    seed: int = 1234


@dataclass
class PipelineConfig:
    unit: UnitConfig = field(default_factory=UnitConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    tagging: PointTaggingConfig = field(default_factory=PointTaggingConfig)
    timestep: TimestepConfig = field(default_factory=TimestepConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)


DEFAULT = PipelineConfig()

# 逐病例配准覆盖项（相对路径 fast/XXX 或 slow/XXX）。
# 用于 trunk_bending 与世界轴冲突、但 wall_branches 左右轴仍可信的个案。
REGISTRATION_CASE_OVERRIDES: Dict[str, Dict] = {
    "fast/FAN_JIAN_MING": {"roll_sign_mode": "world_axis"},
    "fast/LI_SHI_QIANG": {"roll_sign_mode": "world_axis"},
    "fast/LI_ZHEN_SHAN": {"roll_sign_mode": "world_axis"},
    "fast/ZHANG_HAO": {"roll_sign_mode": "world_axis"},
    "slow/YIN_YU_RONG": {"roll_sign_mode": "world_axis"},
    "slow/ZANG_YU_SHU": {"roll_sign_mode": "world_axis"},
    "slow/LI_CHONG_ZENG": {"roll_sign_mode": "world_axis"},
    "slow/XU_YI_CAI": {"roll_sign_mode": "world_axis"},
    "slow/QIN_SI_FU": {"roll_sign_mode": "world_axis"},
}


def case_rel_path(cohort_rel: str, case_name: str) -> str:
    """逐病例 override 键。

    AG 保持历史 ``fast/XXX`` / ``slow/XXX``；AAA 使用 ``ruputer/XXX``；
    单段 cohort（ILO）则使用 ``ILO/<patient>/<phase>``。不能假定 cohort 必含
    ``/``，否则新队列会在尚未查询 override 前就异常退出。
    """
    cohort = str(cohort_rel).strip("/")
    if not cohort:
        raise ValueError("cohort_rel 不能为空")
    subset = cohort.rsplit("/", 1)[-1]
    return f"{subset}/{case_name}"


def split_path(split_name: str = DEFAULT_SPLIT_NAME) -> Path:
    return PROJECT_ROOT / "training" / "splits" / f"{split_name}.json"


def load_split(split_name: str = DEFAULT_SPLIT_NAME) -> Dict:
    return json.loads(split_path(split_name).read_text())


def split_case_labels(
    split_name: str = DEFAULT_SPLIT_NAME,
    partitions: Tuple[str, ...] = ("train", "val", "test"),
) -> List[str]:
    """返回 split 中的相对病例标签，如 fast/XXX 或 slow/XXX。"""
    sp = load_split(split_name)
    labels: List[str] = []
    for part in partitions:
        key = part if part.endswith("_cases") else f"{part}_cases"
        labels.extend(sp.get(key, []))
    return labels


def list_split_cases(
    cohort_rel: str,
    split_name: str = DEFAULT_SPLIT_NAME,
    partitions: Tuple[str, ...] = ("train", "val", "test"),
) -> List[str]:
    """按 split 白名单列出某 cohort 的病例名；不会返回 excluded/pending。"""
    subset = cohort_rel.split("/", 1)[1]
    prefix = f"{subset}/"
    return [label.split("/", 1)[1]
            for label in split_case_labels(split_name, partitions)
            if label.startswith(prefix)]


def registration_for_case(cohort_rel: str, case_name: str,
                          base: RegistrationConfig | None = None) -> RegistrationConfig:
    """返回该病例生效的配准配置（含逐病例 override）。"""
    reg = base or DEFAULT.registration
    over = REGISTRATION_CASE_OVERRIDES.get(case_rel_path(cohort_rel, case_name))
    return replace(reg, **over) if over else reg


def unit_for_case(cohort_rel: str, case_name: str,
                  base: UnitConfig | None = None) -> tuple[UnitConfig, str | None]:
    """返回逐病例生效的单位配置及审计原因。"""
    unit = base or DEFAULT.unit
    canonical_id = f"{str(cohort_rel).strip('/')}/{str(case_name).strip('/')}"
    override = UNIT_CASE_OVERRIDES.get(canonical_id)
    if not override:
        return unit, None
    values = {key: value for key, value in override.items() if key != "reason"}
    return replace(unit, **values), str(override.get("reason", "case_unit_override"))


def out_case_dir(
    cohort_rel: str,
    case_name: str,
    out_root: str | Path | None = None,
) -> Path:
    """返回病例产物目录；``out_root`` 用于隔离 staging 重建。"""
    root = Path(out_root).resolve() if out_root is not None else OUT_ROOT
    return root / cohort_rel / case_name


def raw_case_dir(cohort_rel: str, case_name: str) -> Path:
    return RAW_ROOT / cohort_rel / case_name


def list_cases(cohort_rel: str) -> List[str]:
    d = RAW_ROOT / cohort_rel
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if p.is_dir() and (p / RAW_LAYOUT["wall_ascii_dir"]).is_dir():
            out.append(p.name)
    return out
