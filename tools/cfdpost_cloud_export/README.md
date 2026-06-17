# CFD-Post 云图导出工具包

将 GNN 场重建预测导出为物理单位点云，并支持**三条对比路线**给老师展示：

| 路线 | 内容 | 文档 |
| --- | --- | --- |
| **A** | Fluent 原生 `.cas` + **你导出的 `.dat`** | [三条对比路线.md §A](./三条对比路线.md#路线-a原生-cfd-真值你在-fluent-中导出-dat) |
| **B** | 点云 → STL 面片 / Fluent 插值 | [三条对比路线.md §B](./三条对比路线.md#路线-b点云--面片插值映射) |
| **C** | 点云 / VTP 直接 Contour | [三条对比路线.md §C](./三条对比路线.md#路线-c点云直接展示不插值到面片) |

## 快速开始

```bash
# 在仓库根目录
cd /path/to/GNN
conda activate GNN

# 推荐：一键跑齐 B + C（CSV + STL 映射 + VTP + Fluent 插值 CSV）
bash tools/cfdpost_cloud_export/run_all_routes.sh

# 仅导出 CSV
bash tools/cfdpost_cloud_export/run_export.sh

# 或自定义参数
MANIFEST=outputs/field/<run>/predictions_test_best_wss/manifest.json \
SAMPLE_ID=result_features_merged-1120 \
CASE_NAME=slow/GUO_XI_JIANG \
OUTPUT_DIR=tools/cfdpost_cloud_export/output \
bash tools/cfdpost_cloud_export/run_export.sh
```

等价 Python 调用：

```bash
python tools/cfdpost_cloud_export/export_for_cfdpost.py \
  --manifest outputs/field/<run>/predictions_test_best_wss/manifest.json \
  --sample-id result_features_merged-1120 \
  --case-name slow/GUO_XI_JIANG \
  --output-dir tools/cfdpost_cloud_export/output
```

## 目录结构

```text
tools/cfdpost_cloud_export/
├── README.md                      # 本文件
├── 三条对比路线.md                 # ★ 给老师看图：A/B/C 总览与排版
├── CFDPOST_操作指南.md             # CFD-Post 逐步操作（含 .dat/.cdat）
├── export_for_cfdpost.py          # 公共：.pt → CSV
├── map_to_stl_surface.py          # 路线 B1：点云 → STL 面片 VTP
├── prepare_fluent_interpolation.py # 路线 B2：Fluent 插值用 CSV
├── export_pointcloud_vtp.py       # 路线 C：点云 VTP
├── run_export.sh                  # 仅 CSV
├── run_all_routes.sh              # ★ 一键 B+C
├── fluent_native/                 # 路线 A：放置你导出的 .dat
└── output/
    ├── *.csv
    ├── route_interp/              # B1 VTP + B2 fluent_cloud/
    └── route_pointcloud/          # C：*.vtp
```

## 输入与输出

### 输入

| 来源 | 路径 | 用途 |
| --- | --- | --- |
| manifest | `outputs/field/<run>/predictions_test(_best_wss)/manifest.json` | 定位 `.pt` 预测文件 |
| 预测 | `*.pt` | `y_pred` / `y_wss_pred`（z-score 归一化） |
| features CSV | `data_new/AG/<case>/processed/features/<sample>.csv` | **优先**原始 Fluent 坐标与 CFD 真值 |
| 反归一化 | `data_new/AG/normalization_params_global.json` | `value * std + mean` |
| 坐标逆变换（备用） | `processed/coord_normalized/transform_params.json` | features 缺失时从 `.pt` 的 `x` 还原坐标 |

### 输出 CSV 列

`x, y, z, is_wall, u_cfd, v_cfd, w_cfd, p_cfd, wss_cfd, vel_mag_cfd, u_pred, v_pred, w_pred, p_pred, wss_pred, vel_mag_pred, err_u, err_v, err_w, err_p, err_wss, err_vel_mag`

单位：**压力 Pa**，**速度 m/s**，**WSS Pa**（与 `normalization_params_global.json` 统计口径一致）。

每个样本生成三个文件：

- `*__all.csv` — 全部 15000 节点
- `*__wall.csv` — 壁面节点（`is_wall=1`）
- `*__interior.csv` — 腔内节点（`is_wall=0`）

## 前置条件

1. 已完成对应 run 的 `predict_field` 预测，`manifest.json` 与 `.pt` 存在。
2. 病例 `processed/features/` 下有同名 CSV（推荐；与 `.pt` 行序一致，15000 行）。
3. 在 **GNN** conda 环境中运行（依赖：`torch`, `pandas`, `numpy`；与训练环境相同）。
4. CFD-Post 侧需有网格 case 文件；详见 [CFDPOST_操作指南.md](./CFDPOST_操作指南.md) 中关于 `.cas` / `.dat` / `.cdat` 的说明。

## 可选依赖

- **scipy**（`cKDTree`）：仅当需要在 Fluent 体网格上做最近邻映射时自行使用；本工具包导出脚本不强制依赖。

## 环境说明

| 脚本 | 环境 |
| --- | --- |
| `export_for_cfdpost.py` | **GNN**（torch, pandas） |
| `map_to_stl_surface.py`, `export_pointcloud_vtp.py` | **GNN_vmtk** 或含 **vtk** 的环境 |
| 路线 A `.dat` | 本地 **Fluent** 导出，本仓库不提供 |

## 相关文档

- **三条路线总览**：[三条对比路线.md](./三条对比路线.md)
- CFD-Post 逐步点击：[CFDPOST_操作指南.md](./CFDPOST_操作指南.md)
- 坐标逆变换：`pipeline/coord_normalize.py`
