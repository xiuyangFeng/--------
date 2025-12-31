# 显式几何特征工程项目说明

面向血管 CFD 点云的几何特征提取与 GNN 训练全流程。核心功能：自动提取中心线、计算曲率/切向量/归一化半径等特征，并将它们映射到表面或体点云上，可单病例运行，也可批量处理。

## 目录速览
- `vmtk_core.py`：中心线提取与几何特征计算的核心实现，封装 VMTK 并补充手动计算。
- `Script_Scenario_A_Surface.py`：单病例表面映射（壁面节点特征）。
- `Script_Scenario_B_Volumetric.py`：单病例体点云映射（CFD 体节点特征）。
- `batch_centerline_extract.py`：批量生成中心线（VTP/CSV），用于质检或后续处理。
- `batch_process.py`：批量完成几何预处理与特征映射，主力入口。
- `generate_outlet_flows.py`：根据入口流量和流量比生成各出口流量文件（支持批量处理）。
- `prepare_data.py`：数据整理/重命名/合并工具（可选，用于整理目录）。
- `dataset.py`、`train.py`：将特征 CSV 转为 PyG 图数据并训练 PI-GNN。
- `outdata/`、`点云/`：示例/输出目录，实际路径可自定义。

## 环境依赖
- Python 3.8+
- 几何与点云处理：`vmtk`、`vtk`、`numpy`、`pandas`
- 训练相关（可选）：`torch`、`torch-geometric`、`scikit-learn`

> `vmtk` 通常通过 Conda 安装；确保 `vmtk` 与 `vtk` 版本匹配，启动前激活对应环境。

为了做好数据来源标注，来源于fast文件夹的病例需要在人名编号后做一个fast标注，如一个病人来自于fast文件夹，其对应的人名映射为001，那么他的文件夹命名修改为001-fast。同理如果一个数据来自slow文件夹，其人名映射为002，那么他的文件夹命名为002-slow。同理还有来自AAA文件的病例，来自AAA文件夹的病例分别有来自其中子文件夹rupture和unrupture的，若人名映射为003且来自rupture，则对该数据的文件夹命名为003-AAA-R,若来自unrupture则命名为003-A-UR。同理还有来自ILO文件夹的病例，此文件夹中的数据分别存在为sq文件夹，sh文件夹，分别代表术前，术后，然后各自文件夹中的人名为格式为 名字-1（0），1代表ILO，0代表没有ILO，若人名映射为005，则对应到005-sh-1，代表005患者在术前文件夹并且发生了ILO。


使用001，002.....等数据是为了去敏感化
## 数据要求与目录约定
第一部分属于（fast，slow）病例要求
```
点云/                        # 数据根目录
├─ 004/                      # 病例编号（任意命名，批处理会遍历子目录）
│  ├─ 004.stl                # 必需：血管表面模型（STL 或 VTP 格式）
│  ├─ ascii/                 # 原始 CFD 点云（无扩展名文件，160个时刻）
│  │   ├─ 004-1121
│  │   ├─ 004-1122
│  │   └─ ... (到 004-1280)
│  ├─ ascii_clean/           # 清洗后的 CFD 点云（推荐用于特征提取）
│  │   ├─ 004-1121.csv       # 包含 x,y,z 及流场变量的 CSV 文件
│  │   └─ ... (160个.csv文件)
│  ├─ ascii_in/              # 入口边界点云
│  │   └─ 004-1121.csv 
│  │   └─ ... (160个.csv文件)
│  ├─ ascii_in_clean/       # 清洗后的入口点云
│  │   └─ 004-1121.csv 
│  │   └─ ... (160个.csv文件)
│  ├─ ascii_merged/          # 合并后的点云
│  │   └─ 004-1121.csv 
│  │   └─ ... (160个.csv文件)
│  ├─ centerline/            # 中心线数据（批处理自动生成）
│  │   ├─ centerline.vtp     # 中心线 VTK 文件（ParaView 可视化）
│  │   └─ centerline_points.csv  # 中心线坐标与几何特征
│  └─ *.out                  # 出口压力和入口流量（重要全局参数）
│      ├─ p-outle-rfile.out  # 左外髂支出口压力
│      ├─ p-outli-rfile.out  # 左内髂支出口压力
│      ├─ p-outre-rfile.out  # 右外髂支出口压力
│      ├─ p-outri-rfile.out  # 右内髂支出口压力
│      └─ vf-in-rfile.out    # 入口流量
├─ 010/                      # 其他病例
│  └─ ...（结构相同）
└─ ...
```
第二部分属于（腹主动脉瘤数据）病例要求
点云/                        # 数据根目录
├─ 001/                      # 病例编号（任意命名，批处理会遍历子目录）
│  ├─ 001.stl                # 必需：血管表面模型（STL 或 VTP 格式）
│  ├─ ascii/                 # 原始 CFD 点云（无扩展名文件，160个时刻）
│  │   ├─ 001-0161
│  │   ├─ 001-0162
│  │   └─ ... (到 001-0240)  #一共80个文件
│  ├─ ascii_clean/           # 清洗后的 CFD 点云（推荐用于特征提取）
│  │   ├─ 001-0161.csv       # 包含 x,y,z 及流场变量的 CSV 文件
│  │   └─ ... (80个.csv文件)
│  ├─ ascii_in/              # 入口边界点云
│  │   └─ 001-0161.csv 
│  │   └─ ... (80个.csv文件)
│  ├─ ascii_in_clean/       # 清洗后的入口点云
│  │   └─ 001-0161.csv 
│  │   └─ ... (80个.csv文件)
│  ├─ ascii_merged/          # 合并后的点云
│  │   └─ 001-0161.csv 
│  │   └─ ... (80个.csv文件)
│  ├─ centerline/            # 中心线数据（批处理自动生成）
│  │   ├─ centerline.vtp     # 中心线 VTK 文件（ParaView 可视化）
│  │   └─ centerline_points.csv  # 中心线坐标与几何特征
│  └─ *.out                  # 出口流量和入口流量（重要全局参数）
│      ├─ report-def-2-rfile.out  # 入口质量流量曲线
│      ├─ outlet-flow-ratio.csv # 出口流量比
│      ├─ flow-outle-rfile.out  # 左外髂支出口流量
│      ├─ flow-outli-rfile.out  # 左内髂支出口流量
│      ├─ flow-outre-rfile.out  # 右外髂支出口流量
│      ├─ flow-outri-rfile.out  # 右内髂支出口流量
├─ 010/                      # 其他病例
│  └─ ...（结构相同）
└─ ...

第三部分属于 （髂支闭塞）病例要求
点云/                        # 数据根目录
├─ 002/                      # 病例编号（任意命名，批处理会遍历子目录）
│  ├─ 002.stl                # 必需：血管表面模型（STL 或 VTP 格式）
│  ├─ ascii/                 # 原始 CFD 点云（无扩展名文件，160个时刻）
│  │   ├─ 002-0161
│  │   ├─ 002-0162
│  │   └─ ... (到 004-0240)  #一共80个文件
│  ├─ ascii_clean/           # 清洗后的 CFD 点云（推荐用于特征提取）
│  │   ├─ 002-0161.csv       # 包含 x,y,z 及流场变量的 CSV 文件
│  │   └─ ... (80个.csv文件)
│  ├─ ascii_in/              # 入口边界点云
│  │   └─ 002-0161.csv 
│  │   └─ ... (80个.csv文件)
│  ├─ ascii_in_clean/       # 清洗后的入口点云
│  │   └─ 002-0161.csv 
│  │   └─ ... (80个.csv文件)
│  ├─ ascii_merged/          # 合并后的点云
│  │   └─ 002-0161.csv 
│  │   └─ ... (80个.csv文件)
│  ├─ centerline/            # 中心线数据（批处理自动生成）
│  │   ├─ centerline.vtp     # 中心线 VTK 文件（ParaView 可视化）
│  │   └─ centerline_points.csv  # 中心线坐标与几何特征
│  └─ *.out                  # 出口流量和入口流量（重要全局参数）
│      ├─ report-def-2-rfile.out  # 入口质量流量曲线
│      ├─ outlet-flow-ratio.csv # 出口流量比
│      ├─ flow-outle-rfile.out  # 左外髂支出口流量
│      ├─ flow-outli-rfile.out  # 左内髂支出口流量
│      ├─ flow-outre-rfile.out  # 右外髂支出口流量
│      ├─ flow-outri-rfile.out  # 右内髂支出口流量
├─ 010/                      # 其他病例
│  └─ ...（结构相同）
└─ ...
**文件格式要求：**
- **表面模型**：`.stl` 或 `.vtp` 格式均可；若病例目录中有多个模型文件，脚本会自动选择第一个并给出提示。
- **点云文件**：
  - **CSV 格式**：至少包含 `x,y,z` 列（前三列），其他列（如 `u,v,w,p,k,epsilon` 等流场变量）会被原样保留并随几何特征一起写回输出文件。
  - **NPY 格式**：二维数组，前 3 列必须为 `x,y,z` 坐标，其余列按 `Feature_0, Feature_1...` 命名追加。
  - **无扩展名格式**：`ascii/` 目录中的原始 FLUENT 导出文件（会被清洗脚本转换为 CSV）。
- **全局参数文件（.out）**：
  - FLUENT 时序监测数据，格式为三列：`Time Step`、监测值、`flow-time`（物理时间）。
  - 包含边界条件（入口速度、出口压力等）随时间的变化，用于物理一致性检验和全局特征提取。
  - 每个病例通常包含 5 个监测文件，对应入口和多个出口的边界参数。
- **输出目录**：批处理默认写入 `output_root/病例名/ascii_mapped/`，文件名格式为 `result_features_<原文件名>.csv`。

## 核心特征说明
输出 CSV 列：
- `x,y,z`：原始坐标
- `Abscissa`：沿中心线的归一化弧长（0=入口，1=出口）
- `NormRadius`：径向距离 / 局部半径（壁面通常接近 1）
- `Curvature`：中心线曲率
- `Tangent_X/Y/Z`：中心线切向量
- 其余列：原始 CFD 物理量或 `Feature_k`（来自输入点云）

## 单病例快速使用
### 1) 表面映射（Scenario A）
将中心线特征映射到壁面节点：
```bash
python - <<'PY'
from Script_Scenario_A_Surface import process_surface_dataset
process_surface_dataset("MA+XIAO+DONG-new.stl", "surface_features.csv")
PY
```

### 2) 体点云映射（Scenario B）
适用于 CFD 体点云（CSV 或 NPY）：
```bash
python - <<'PY'
from Script_Scenario_B_Volumetric import process_volumetric_dataset
process_volumetric_dataset(
    stl_path="MA+XIAO+DONG-new.stl",
    cfd_cloud_path="outdata/004/ascii_clean/004-1121.csv",
    output_csv_path="outdata/004/ascii_mapped/result_features_004-1121.csv",
)
PY
```
> 运行时会先预处理几何（读取表面、提取中心线、构建 KDTree），随后对单个点云做特征映射。

## 批量处理指南（重点）
### 1) 批量提取中心线（可选质检）
若希望先生成所有病例的中心线并导出 VTP/CSV：
```bash
python batch_centerline_extract.py \
  --input_dir 点云 \
  --output_dir 点云 \
  --overwrite      # 可选，重算已存在结果
  # --no_csv       # 仅保存 VTP
```
输出示例：`点云/004/centerline/centerline.vtp`、`centerline_points.csv`。

### 2) 批量特征映射（主流程）
遍历输入根目录下的所有病例，自动读取唯一的表面模型，映射所有点云：
```bash
python batch_process.py \
  --input_dir 点云 \
  --output_dir outdata \
  --cloud_subdir ascii_clean \   # 点云所在子目录（若直接在病例根目录可省略/改为空）
  --output_subdir ascii_mapped   # 输出子目录名
```
行为说明：
- 每个病例仅在首次点云处理前预处理一次几何（中心线+KDTree），后续点云复用，效率更高。
- 点云文件名默认过滤包含 `result_` 的文件，避免重复处理输出。
- 输出命名：`result_features_<原文件名去扩展名>.csv`。

### 3) 数据整理（可选）
若需要批量重命名/合并 STL 与点云目录，可在修改 `prepare_data.py` 顶部开关后运行：
```bash
python prepare_data.py
```
推荐先保持 `DRY_RUN = True` 查看计划，再执行实际操作。

### 4) 生成出口流量文件（腹主动脉瘤/髂支闭塞数据）
对于腹主动脉瘤和髂支闭塞数据，需要根据入口流量和出口流量比生成各个出口的流量时序文件：
```bash
# 单个病例处理
python generate_outlet_flows.py --case 点云/001

# 批量处理所有病例
python generate_outlet_flows.py --batch 点云

# 覆盖已存在的文件
python generate_outlet_flows.py --batch 点云 --overwrite
```
功能说明：
- 读取 `report-def-2-rfile.out`（入口质量流量曲线）
- 读取 `outlet-flow-ratio.csv`（各出口流量比）
- 自动计算并生成4个出口流量文件：`flow-outle-rfile.out`、`flow-outli-rfile.out`、`flow-outre-rfile.out`、`flow-outri-rfile.out`
- 详细使用说明请参考 `出口流量生成说明.md`

## 物理信息 GNN 训练 (GNN_train 模块)

该模块位于 `min-road/GNN_train` 下，提供了从 CSV 特征数据到 GNN 模型训练与验证的完整流水线。

- **输入特征 (16维)**：包括空间坐标 `(x, y, z)`、归一化时间 `t` (0~1s)、以及提取的几何特征和边界条件。
- **预测目标 (5维)**：`u, v, w` (速度)、`p` (压力)、`wss` (壁面剪切应力)。
- **图构建**：使用 K=6 的 KNN 算法基于节点坐标构建局部连接。

### 快速启动
1. **数据转换**：将 CSV 转换为图格式 `.pt`
   ```bash
   python min-road/GNN_train/data_converter.py --data-root ./data --output-dir ./min-road/GNN_train/processed_data
   ```
2. **模型训练**：
   ```bash
   python min-road/GNN_train/train.py --data-dir ./min-road/GNN_train/processed_data --epochs 100
   ```
3. **模型验证**：
   ```bash
   python min-road/GNN_train/validate.py --model-path ./min-road/GNN_train/checkpoints/best_model.pt --data-list test_files.txt
   ```

更多详情见 [GNN_train/README.md](file:///Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/min-road/GNN_train/README.md)。

## 常见问题与排查
- **未找到 `vmtk`/`vtk`**：确认已在 Conda 环境安装并激活；Mac/Linux 可用 `python -c "import vmtk"` 试运行。
- **未找到表面模型**：确保病例目录仅有一个 `.stl/.vtp`；若有多个，脚本会提示并取第一个。
- **点云缺少 `x,y,z`**：CSV/NPY 首三列必须是坐标，否则会报错。
- **输出为空或数量为 0**：检查 `cloud_subdir`、文件过滤规则以及输出目录权限。

## 参考路径
- 批处理示例输出：`outdata/004/ascii_mapped/result_features_004-1121.csv`
- 代码主入口：`batch_process.py`、`Script_Scenario_B_Volumetric.py`
- 几何核心：`vmtk_core.py`

如需扩展或调整特征，请优先修改 `vmtk_core.py`（新增数组）及对应映射脚本。

## 分支分析 (vmtk-分叉)
用于对血管模型进行分支提取和几何分析（如 CIA 扭转度、EIA 直径、DRI 等）。
```bash
python vmtk-分叉/branch_analysis.py "MA XIAO DONG-new.stl"
```
输出：
- `*_branched.vtp`: 包含分支 ID 的中心线文件，可用于 ParaView 可视化。
- `*_analysis.csv`: 包含每个分支的几何指标及 DRI 计算结果。
