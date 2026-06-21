# CFD-Post 云图对比操作指南

本文说明如何将 GNN 预测结果导入 ANSYS CFD-Post，与 Fluent CFD 真值做壁面 WSS、腔内压力等云图对比。

> **给老师做对比图**：请先读 [三条对比路线.md](./三条对比路线.md)（路线 A 原生 `.dat` / 路线 B 插值面片 / 路线 C 点云直接展示），本文是 CFD-Post 侧逐步补充。

---

## 1. 你需要哪些 Fluent / CFD-Post 文件？

### 1.1 标准 Fluent Case + Data（最常见）

| 扩展名 | 含义 | 是否必需 |
| --- | --- | --- |
| **`.cas` / `.cas.gz`** | **Case（网格 + 边界 + 求解设置）** | **必需** — CFD-Post 需要网格才能显示云图 |
| **`.dat` / `.dat.gz`** | **Data（该时刻的流场解：压力、速度、WSS 等）** | **显示 CFD 真值云图时必需**；仅叠加外部点云时可暂不加载 |
| **`.stl`** | 表面三角网格（可视化辅助） | 可选；本仓库 GUO_XI_JIANG 有 `GUO_XI_JIANG.stl` |

在 Fluent 中：**File → Read → Case** 读入 `.cas`，再 **File → Read → Data** 读入同名 `.dat`，两者合体后 CFD-Post 才能显示完整 CFD 解。

### 1.2 `.cdat` 是什么？

`.cdat` 常见于以下场景，**不要与 `.dat` 混为一谈**：

1. **CFD-Post 原生 Case 格式**（旧版或导出为 CFD-Post case 时）：有时整套 case 以 `.cas` + `.cdat` 命名，其中 `.cdat` 承载解数据，作用类似 Fluent 的 `.dat`。
2. **瞬态 / 多时间步**：如 `foo-0123.cdat` 表示某一时间步的 data。
3. **教程笔误**：部分资料将 “case + data” 笼统写成 `.dat` 和 `.cdat`，实际在 **Fluent 稳态单帧** 工作流里，对应的是 **`.cas` + `.dat`**。

**本仓库 GUO_XI_JIANG 病例现状（已核查）：**

```text
data_new/AG/slow/GUO_XI_JIANG/
├── GUO_XI_JIANG.cas.gz    ✅ 有（约 21 MB，Fluent case）
├── GUO_XI_JIANG.stl       ✅ 有
└── GUO_XI_JIANG.dat       ❌ 仓库内未找到
```

- 做 **CFD 真值云图**：需从原始 Fluent 工作目录或备份中找回与 `GUO_XI_JIANG.cas.gz` **同一次计算、同一时间步** 的 `GUO_XI_JIANG.dat`（或 `.dat.gz`），解压后与 case 一起读入。
- 做 **仅 GNN 预测云图 / 点云插值对比**：有 `.cas.gz` 即可加载网格；预测 CSV 由本工具包导出，通过 **Point Cloud / External Data** 方式叠加。
- `ascii/GUO_XI_JIANG-1120` 等文件是 **按样本导出的 ASCII 点表**（压力、WSS），单位与坐标系可能与 `processed/features/*.csv` 不同，**不要**直接当作 Fluent `.dat` 使用。

### 1.3 本项目的“真值”数据来源（不依赖 .dat）

GNN 训练/评估用的 CFD 真值已落在：

```text
data_new/AG/<speed>/<CASE>/processed/features/result_features_merged-<id>.csv
```

列含 `x,y,z,u,v,w,p,wss,is_wall` 等，**已是物理单位**，与 `.pt` 行序一致（15000 节点）。  
导出脚本优先读此 CSV，**不强制**你有 Fluent `.dat`。

---

## 2. 从 GNN 侧导出预测点云

### 2.1 环境

```bash
conda activate GNN
cd /path/to/GNN
```

### 2.2 一键导出（示例）

```bash
bash tools/cfdpost_cloud_export/run_export.sh
```

默认使用 run `...20260609_124213` 的 `predictions_test_best_wss/manifest.json`，样本 `result_features_merged-1120`，病例 `slow/GUO_XI_JIANG`。

### 2.3 自定义 run / 样本

```bash
python tools/cfdpost_cloud_export/export_for_cfdpost.py \
  --manifest outputs/field/<你的run>/predictions_test_best_wss/manifest.json \
  --sample-id result_features_merged-1120 \
  --case-name slow/GUO_XI_JIANG \
  --output-dir tools/cfdpost_cloud_export/output
```

### 2.4 导出物说明

每个样本 3 个 CSV：

| 文件后缀 | 内容 |
| --- | --- |
| `__all.csv` | 全部节点 |
| `__wall.csv` | 壁面（WSS 对比用） |
| `__interior.csv` | 腔内（压力对比用） |

脚本会打印：坐标来源（`features_csv` 或 `inverse_transform`）、壁面/腔内节点数、输出路径。

---

## 3. CFD-Post 加载网格与 CFD 解

### 步骤 A：打开 Case

1. 启动 **ANSYS CFD-Post**（或 Workbench → CFD-Post）。
2. **File → Open**（或 *Load Case*）。
3. 选择 `GUO_XI_JIANG.cas.gz`（Post 通常可直接读 gzip case）。
4. 若提示加载 Data：
   - 有 `.dat`：选择同名 `GUO_XI_JIANG.dat`；
   - **无 `.dat`**：可仅加载 case 看网格，CFD 变量列表为空或不可用——此时用下文第 4 节导入 GNN CSV 中的 `*_cfd` 列作为对照。

### 步骤 B：确认变量单位

- Fluent 默认压力常为 **Pa**，速度 **m/s**，WSS **Pa**（与导出 CSV 一致）。
- 若你曾在 Fluent 中改过单位，请在 Post 的 *Variables* 面板核对，并与 CSV 列对齐。

---

## 4. 导入 GNN 导出 CSV（点云 / 外部数据）

两种常用路径（版本菜单名略有差异）：

### 方法 1：Point Cloud（推荐用于对比图）

1. **File → Import → Import Point Cloud**（或 *Insert → Point Cloud*）。
2. 选择 `*__wall.csv` 或 `*__interior.csv`。
3. 坐标列映射：`x, y, z`。
4. 标量变量：导入 `wss_pred`、`p_pred` 或误差列 `err_wss`、`err_p`。
5. 在同一视图中对 Point Cloud 做 **Contour** 或 **Vector**，网格上可同时显示 CFD 变量。

### 方法 2：Fluent 内插值后写回（需 Fluent 许可证）

若希望预测场显示在**体网格面/体**上而非离散点：

1. 在 Fluent 中 **File → Read → Case**（`GUO_XI_JIANG.cas.gz`）。
2. **Mesh → Interpolate**，选择 *Cloud of Points*，读入 `*__all.csv` 的 `x,y,z` 与 `p_pred`（或 `wss_pred`）。
3. 写回为自定义场（如 `p_gnn`），再 **File → Write → Data** 得到新的 `.dat`。
4. 在 CFD-Post 中加载 case + 新 dat，用标准 Contour 对比 `Pressure` vs `p_gnn`。

本工具包**不自动调用 Fluent**；上列为手工流程参考。

---

## 5. 壁面 WSS 云图工作流

1. 导出 `*__wall.csv`。
2. CFD-Post 加载 case（+ dat 若有）。
3. 导入壁面点云，变量选 `wss_pred`。
4. 创建 **Contour**：
   - CFD 侧：边界上 `Wall Shear` / `Skin Friction`（名称取决于 Fluent 导出）。
   - GNN 侧：Point Cloud 上 `wss_pred` 或 `wss_cfd`（CSV 内真值，无需 .dat）。
5. 对比技巧：
   - 统一 colorbar 范围（建议用 CFD 与 pred 的 5–95 分位，避免极值拉伸）。
   - 叠加 `err_wss` 等值面查看系统误差分布。
   - 壁面节点约 1 万级，点云渲染已足够；不必先重网格化。

---

## 6. 腔内压力云图工作流

1. 导出 `*__interior.csv`。
2. 在通过中心线的截面上：
   - CFD：有 `.dat` 时用原生 `Pressure` Contour；
   - 无 `.dat` 时用 CSV 的 `p_cfd` 做点云 Contour。
3. GNN：同截面过滤点云，对 `p_pred` 做 Contour。
4. 使用 **Turbo** 或 **Plane** 切面工具对齐同一几何位置再截图对比。

---

## 7. CFD vs DL 对比建议

| 项目 | 建议 |
| --- | --- |
| 坐标系 | 优先 `features_csv` 坐标（与 Fluent 物理坐标一致）；避免仅用 PCA 逆变换截图叠图 |
| 样本时间 | `merged-1120` 对应 ascii 子目录 `GUO_XI_JIANG-1120`；确保对比同一时间步 |
| 壁面 WSS | 只看 `is_wall=1`；腔内 WSS 在 CSV 中多为 0 |
| 压力 | 腔内对比 `p_pred` vs `p_cfd`；壁面压力边界条件可能相同，差异主要在内部场 |
| 误差图 | 直接使用 `err_p`、`err_wss`，比心算差更省事 |
| 定量 | CSV 内可后处理 RMSE；本工具包不跑 WSS 批量评估脚本 |

---

## 8. 故障排查

| 现象 | 处理 |
| --- | --- |
| Post 提示缺少 data | 找回 `.dat`，或仅用 CSV 中 `*_cfd` / `*_pred` 做点云 |
| 点云与网格错位 | 确认导出日志为 `坐标来源: features_csv`；若 `inverse_transform`，检查 `transform_params.json` |
| 行数不是 15000 | 检查 features CSV 与 `.pt` 是否同一样本 |
| manifest 多条同名 sample | 必须指定 `--case-name`（含 `slow/` 或 `fast/` 前缀） |
| `ImportError: pipeline` | 在仓库根目录运行，并使用 GNN 环境 |

---

## 9. 文件清单速查

```text
# GNN 预测
outputs/field/<run>/predictions_test_best_wss/manifest.json
outputs/field/<run>/predictions_test_best_wss/slow__GUO_XI_JIANG__result_features_merged-1120.pt

# 原始 CFD 特征（坐标 + 真值）
data_new/AG/slow/GUO_XI_JIANG/processed/features/result_features_merged-1120.csv

# Fluent 网格（本病例）
data_new/AG/slow/GUO_XI_JIANG/GUO_XI_JIANG.cas.gz

# 导出结果
tools/cfdpost_cloud_export/output/slow__GUO_XI_JIANG__result_features_merged-1120__wall.csv
```

更多参数说明见 [README.md](./README.md)。
