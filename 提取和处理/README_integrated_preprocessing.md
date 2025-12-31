# 血管数据一体化预处理工具

## 简介

`integrated_preprocessing.py` 是一个用于血管 CFD（计算流体力学）数据预处理的一体化工具，整合了数据准备、中心线提取、出口流量生成、数据清洗、点云合并和几何特征提取等功能。

## 依赖环境

```bash
# 基础环境
conda activate rag_venv

# 必需的 Python 包
- pandas
- numpy

# VTK 相关（中心线提取和几何特征提取需要）
conda install -c conda-forge vtk netcdf4
```

## 使用方法

### 交互式菜单模式

```bash
python integrated_preprocessing.py
```

运行后会显示交互式菜单：

```
======================================================================
  血管数据一体化预处理工具
======================================================================

功能列表:
  1. 完整流程 (依次执行所有步骤)
  2. 数据准备 (从 data 和 stl_data 整理到点云文件夹)
  3. 中心线提取 (批量提取血管中心线)
  4. 生成出口流量 (根据入口流量和流量比计算)
  5. 数据清洗 (清洗 Fluent ASCII 文件)
  6. 合并点云 (合并壁面和内部点云)
  7. 几何特征提取 (提取血管几何特征)
  8. 数据准备 [预演模式] (仅显示计划，不实际执行)
  0. 退出
```

## 功能模块详解

### 1. 完整流程

按顺序执行所有预处理步骤（2-7），每个步骤执行前会询问确认。

### 2. 数据准备

**功能**: 从 `data` 和 `stl_data` 文件夹整理病例数据到统一的 `点云` 文件夹。

**数据源扫描路径**:
- `data/fast/` - 快速进展型
- `data/slow/` - 慢速进展型
- `data/AAA/rupture/` - 破裂型 AAA
- `data/AAA/unrupture/` - 未破裂型 AAA
- `data/ILO/sq/` - ILO sq 类型
- `data/ILO/sh/` - ILO sh 类型

**复制的文件**:
- STL 文件（从 `stl_data/anonymized/`）
- `ascii/` 文件夹（壁面 CFD 点云数据）
- `ascii_in/` 文件夹（内部 CFD 点云数据）
- 边界条件文件（`*.out`、`outlet-flow-ratio.csv`）

**输出结构**:
```
点云/
├── P001/
│   ├── P001.stl
│   ├── ascii/
│   ├── ascii_in/
│   ├── vf-in-rfile.out
│   ├── p-outle-rfile.out
│   └── ...
├── P002/
└── ...
```

### 3. 中心线提取

**功能**: 使用 VMTK 从 STL 模型提取血管中心线。

**依赖**: 需要 VTK 和 vmtk_core 模块。

**输出**:
- `centerline/centerline.vtp` - VTK 格式的中心线
- `centerline/centerline_points.csv` - CSV 格式的中心线点数据

**CSV 包含的属性**:
- x, y, z - 坐标
- Abscissas - 弧长
- MaximumInscribedSphereRadius - 最大内切球半径
- Curvature - 曲率
- Tangent_X, Tangent_Y, Tangent_Z - Frenet 切向量

### 4. 生成出口流量

**功能**: 根据入口质量流量和出口流量比例计算各出口的体积流量。

**所需文件**:
- `report-def-2-rfile.out` - 入口质量流量（kg/s）
- `outlet-flow-ratio.csv` - 各出口流量比例

**计算公式**:
```
出口体积流量 = (入口质量流量 / 血液密度) × 流量比例
```

**默认参数**:
- 血液密度: 1060 kg/m³

**输出**:
- `{outlet_name}.out` - 各出口流量文件

### 5. 数据清洗

**功能**: 清洗 Fluent 导出的 ASCII 数据文件，转换为标准 CSV 格式。

**处理内容**:
- 列名标准化（如 `x-coordinate` → `x`）
- 坐标单位转换（米 → 毫米）
- 补齐缺失的速度列和壁面剪切力列
- 添加 `is_wall` 标记（速度接近 0 的点标记为壁面）
- 移除节点编号和面积字段

**输入/输出目录**:
- `ascii/` → `ascii_clean/`
- `ascii_in/` → `ascii_in_clean/`

**标准化后的列名**:
| 原始列名 | 标准化列名 |
|---------|-----------|
| x-coordinate | x |
| y-coordinate | y |
| z-coordinate | z |
| pressure | p |
| x-velocity | u |
| y-velocity | v |
| z-velocity | w |
| velocity-magnitude | vel_mag |
| wall-shear | wss |
| x-wall-shear | wss_x |
| y-wall-shear | wss_y |
| z-wall-shear | wss_z |

### 6. 合并点云

**功能**: 合并壁面点云和内部点云数据。

**输入**:
- `ascii_clean/` - 清洗后的壁面数据
- `ascii_in_clean/` - 清洗后的内部数据（可选）

**输出**:
- `ascii_merged/` - 合并后的完整点云

### 7. 几何特征提取

**功能**: 使用 STL 模型和中心线为点云数据添加几何特征。

**依赖**: 需要 `Script_Scenario_B_Volumetric.py` 模块。

**输入**:
- STL 文件
- `ascii_merged/` 目录下的点云文件
- 边界条件文件

**输出**:
- `outdata/{case_name}/ascii_mapped/result_features_*.csv`

**添加的特征**:
- 距离中心线距离
- 血管半径
- 曲率
- 边界条件值（入口速度、出口压力等）

## 目录结构

### 处理前

```
项目根目录/
├── data/
│   ├── fast/
│   │   └── PATIENT_NAME/
│   │       ├── ascii/
│   │       ├── ascii_in/
│   │       └── *.out
│   ├── AAA/
│   │   ├── rupture/
│   │   └── unrupture/
│   └── ILO/
├── stl_data/
│   ├── anonymized/
│   │   └── P001.stl, P002.stl, ...
│   └── new_mapping_*.json
└── integrated_preprocessing.py
```

### 处理后

```
项目根目录/
├── 点云/
│   └── P001/
│       ├── P001.stl
│       ├── ascii/
│       ├── ascii_in/
│       ├── ascii_clean/
│       ├── ascii_in_clean/
│       ├── ascii_merged/
│       ├── centerline/
│       │   ├── centerline.vtp
│       │   └── centerline_points.csv
│       └── *.out
├── outdata/
│   └── P001/
│       └── ascii_mapped/
│           └── result_features_*.csv
└── ...
```

## 映射文件

工具依赖 `stl_data/` 目录下的映射文件来关联原始病例名和匿名 ID：

- `new_mapping_*.json` - JSON 格式（优先使用）
- `new_mapping_*.csv` - CSV 格式（备用）

映射文件由 `anonymization_tool.py` 生成。

## 注意事项

1. **首次运行**: 建议先运行 "数据准备 [预演模式]"（选项 8）查看计划，确认无误后再执行实际操作。

2. **VTK 依赖**: 中心线提取和几何特征提取功能需要安装 VTK。如果只需要数据清洗功能，可以不安装 VTK。

3. **覆盖行为**: 
   - 默认情况下，已存在的文件不会被覆盖
   - 中心线提取可通过代码中的 `overwrite` 参数控制

4. **坐标单位**: 数据清洗默认将坐标从米转换为毫米，以匹配 STL 模型的单位。

5. **血液密度**: 出口流量计算默认使用 1060 kg/m³，可在代码中修改。

## 错误处理

- 遇到错误时，程序会显示错误信息并询问是否显示详细堆栈
- 按 `Ctrl+C` 可以中断当前操作
- 程序会在每个主要步骤后暂停，按回车继续

## 示例工作流

```bash
# 1. 激活环境
conda activate rag_venv

# 2. 运行预处理工具
python integrated_preprocessing.py

# 3. 选择功能 1 运行完整流程
# 或选择单独的功能进行处理
```

