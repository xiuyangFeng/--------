---
name: postview-surface-viz
description: >-
  GNN/CROWN 点云标量回插 STL 面片并生成病例级可视化（postview 交付包、三联图、ParaView 包），
  以及把体点云转成可在 ParaView Slice 出连续填充截面的 .vtu 体网格。
  在用户要求后处理可视化、面片云图、点云插值到 STL、点云转 VTU/VTP、做截面/切片、postview、
  merged-1146 汇报图、CFD|Pred|Error 三联图时使用。遵循 docs/paper_reproduction/05-点云预测值与真值回插到面片方法.md。
---

# 点云 → 面片后处理可视化（postview）

## 触发

「后处理可视化」「面片云图」「回插 STL」「点云转 VTU/VTP」「做截面/切片/Slice」「postview」「merged-1146 图」「三联图」「ParaView 包」等；或给出 `manifest.json` / CROWN checkpoint + 病例名。

**默认假定**：V3P/GNN 场重建；CROWN baseline 走专用分支。**禁止**把插值面片指标当作正式 R²。

---

## 0. 核心口径（每次必守）

| 目的 | 口径 |
| --- | --- |
| **数值指标** | 原始同点点云 CSV（`__wall.csv`），**不**在插值面片上算 R² |
| **病例级云图** | 点云标量 → **同一 STL** + **同一插值法/参数** → VTP/PNG |
| **展示帧** | 默认 **`result_features_merged-1146`**（`t_norm≈0.16`，收缩期上升段）；**不用** 1120 作主图 |

图注必写：`merged-1146 · t_norm≈0.16 · 收缩期上升段（近似）· Gaussian r=3 mm, sharpness=2`；若报数字须注明 **81 帧 pooled**。

方法细节与 QC 清单 → [reference.md](reference.md) · 交付索引 → `outputs/field/postview/README.md` · 方法论文档 → `docs/paper_reproduction/05-点云预测值与真值回插到面片方法.md`

---

## 0.5 两条产物路线（先判别需求）

| 需求 | 产物 | 几何 | 标量挂载 | 脚本 | 节点用途 |
| --- | --- | --- | --- | --- | --- |
| **只看壁面**（WSS/压力面片云图、三联图） | **VTP 面片**（三角面） | STL 三角网格 | 壁面点云 → 插值到 STL 顶点 | `map_to_stl_surface.py`（§3.1/§3.4） | 旋转看面 |
| **要做截面/切片**（腔内压力、速度切面） | **VTU 体网格** | Fluent `.cas` 四面体单元 | 体点云 → 插值到网格节点 | `build_sliceable_volume.py`（§3.5） | ParaView Slice 出连续填充截面 |

判别关键：**ParaView/CFD-Post 对纯点云做 Slice 只会切到平面附近的稀疏散点**，要得到连续填充截面，标量必须挂在**带体单元连接**的网格上（即 `.vtu`，与文件后缀 `.cas/.dat` 无关）。仅看壁面则无需体网格，VTP 面片即可。

### 0.6 先锁定病例范围，避免无意全量导出

- 对已完成的 baseline，默认只导出**一个最佳与一个最差**的开发集病例；以正式同点 `per_case_metrics.csv` / `metrics.json` 的逐病例 R² 排序，并在批次 README 记录模型、分区、排序指标与数值。
- 仅当用户明确要求“全部病例 / 全 val / 全 test / 批量”时，才提交全病例后处理作业；test 仍遵守原实验访问门禁。
- 用户要求 `wss / wss(max)` 时，CFD、预测、signed error 与 absolute error 必须都除以**同一病例 CFD 壁面最大 WSS**；在 VTP 使用明确后缀（如 `*_over_cfd_max`），并在 manifest 写入分母（Pa）。不得各自按预测最大值归一化。

---

## 1. 锁定输入

### 1.1 GNN / V3P

| 项 | 路径/说明 |
| --- | --- |
| manifest | `outputs/field/<run>/predictions_test_best_wss/manifest.json`（或 `predictions_test`） |
| 病例 | `slow/GUO_XI_JIANG` · `slow/ZHANG_JUN_HUA` · `fast/CHEN_SHI_MING`（汇报三例） |
| STL | `data_new/AG/<CASE_NAME>/<CASE_SHORT>.stl` |
| 展示帧 | `SAMPLE_ID=result_features_merged-1146` |

从 `config.snapshot.json` / 用户说明确认 run；**不要**与 V3D 或 CROWN 指标混表。

### 1.2 CROWN baseline

| 项 | 路径/说明 |
| --- | --- |
| config | `external_baselines/crown_beihang/configs/local/crown_original_vp_*.json` |
| checkpoint | `outputs/external_baselines/crown_beihang/.../best_model.pt` |
| 说明 | CROWN **无 wss_pred**；主变量 `p` / `vel_mag`；WSS 仅 `wss_cfd` 真值 |

---

## 2. 默认插值参数（与现有 postview 对齐）

```text
METHOD=gaussian
RADIUS=3.0      # mm
SHARPNESS=2.0
MAX_DIST=3.0    # mm
FIELD_CMAP=GNN_BWR
ERR_CMAP=GNN_BWR
```

**一次性回插标量**（GNN 壁面）：

```text
wss_cfd,wss_pred,err_wss,abs_err_wss,
wss_cfd_over_cfd_max,wss_pred_over_cfd_max,
err_wss_over_cfd_max,abs_err_wss_over_cfd_max,
p_cfd,p_pred,err_p,abs_err_p
```

debug 坐标对齐可用 `--method nearest`；汇报主图用 **gaussian**。

---

## 3. 执行流程（按场景选一条）

**环境**：仓库根目录；`export_for_cfdpost.py` → **GNN**；`map_to_stl_surface.py` / `plot_stl_mapped_triptych.py` → 现有 shell 默认 **GNN**（需 vtk）。

### 3.1 ★ GNN 完整 ParaView 交付包（推荐）

每病例生成 `surface_wall.vtp` + 配色 + 说明 + `manifest_bundle.json`：

```bash
cd /path/to/GNN
conda activate GNN

MANIFEST=outputs/field/<run>/predictions_test_best_wss/manifest.json \
CASE_NAME=slow/GUO_XI_JIANG \
SAMPLE_ID=result_features_merged-1146 \
RUN_TAG=v3p_i6diag_t016_report \
bash tools/cfdpost_cloud_export/package_postview_case.sh
```

产出：`outputs/field/postview/<RUN_TAG>/<CASE>__result_features_merged-1146/`

| 主文件 | 用途 |
| --- | --- |
| `<CASE>__surface_wall.vtp` | ★ ParaView 面片云图（含全部标量） |
| `<CASE>__pointcloud_wall.vtp` | 原始壁面点云（对照插值平滑） |
| `<CASE>__mapping_report.json` | 覆盖率 / map_dist 统计 |
| `GNN_blue_white_red.xml` | 蓝-白-红色标 |
| `README_后处理打开说明.md` | ParaView 操作 |

三例批量：对 `GUO_XI_JIANG` / `ZHANG_JUN_HUA` / `CHEN_SHI_MING` 各跑一遍（改 `CASE_NAME`）。

### 3.2 GNN 仅出 WSS/P 三联 PNG（无 ParaView 包）

```bash
MANIFEST=outputs/field/<run>/predictions_test_best_wss/manifest.json \
CASE_NAME=slow/GUO_XI_JIANG \
SAMPLE_ID=result_features_merged-1146 \
RUN_TAG=<slug> \
bash tools/cfdpost_cloud_export/run_case_surface_compare.sh
```

产出：
- PNG：`outputs/field/plots/stl_surface_compare/<RUN_TAG>/<stem>/fig_{wss,p}_triptych.png`
- 中间件：`tools/cfdpost_cloud_export/output/<RUN_TAG>/route_interp/*__stl_mapped_wall.vtp`

### 3.3 CROWN 完整包（推理 + 映射 + 压力/速度三联图）

```bash
CROWN_CONFIG=external_baselines/crown_beihang/configs/local/crown_original_vp_split_AG_v1_seed1.json \
CROWN_CKPT=outputs/external_baselines/crown_beihang/<run>/best_model.pt \
CASE_NAME=slow/GUO_XI_JIANG \
SAMPLE_ID=result_features_merged-1146 \
RUN_TAG=crown_vp_t016_report \
CROWN_METHOD_LABEL=non-PINN \
bash tools/cfdpost_cloud_export/package_crown_postview_case.sh
```

三例批量：

```bash
bash tools/cfdpost_cloud_export/run_crown_surface_batch.sh
```

PINN 变体：改 `CROWN_CONFIG` / `CROWN_CKPT` / `RUN_TAG=crown_pinn_t016_report` / `CROWN_METHOD_LABEL=PINN`。

已有 `_export` CSV 时仅重映射+补图：

```bash
bash tools/cfdpost_cloud_export/refresh_crown_surface_plots.sh
```

### 3.4 已有 wall CSV，仅重跑映射

```bash
conda activate GNN

python tools/cfdpost_cloud_export/map_to_stl_surface.py \
  --csv <path/to/*__wall.csv> \
  --stl data_new/AG/<case>/<CASE>.stl \
  --method gaussian \
  --radius 3.0 \
  --sharpness 2.0 \
  --max-dist 3.0 \
  --scalars wss_cfd,wss_pred,err_wss,abs_err_wss,p_cfd,p_pred,err_p,abs_err_p \
  --output-dir <out>/surface_gaussian
```

补三联图：

```bash
python tools/cfdpost_cloud_export/plot_stl_mapped_triptych.py \
  --vtp <out>/*__stl_mapped_wall.vtp \
  --render surface \
  --variable wss|p|vel_mag \
  --output <out>/fig_<var>_triptych.png \
  --field-cmap GNN_BWR --err-cmap GNN_BWR \
  --report-json <out>/fig_<var>_triptych_report.json
```

### 3.5 ★ 体点云 → 可切面 .vtu（做截面用）

**前提**：已有 `<CASE>__volume_merged-1146.vtp`（§3.1 的交付包里已含；它是体点云=Fluent 单元中心点 + 合并 pred/cfd 标量）。

**环境**：`conda activate GNN_vmtk`（`.cas` 模式需 `vtkFLUENTReader`，比纯 GNN 环境更稳）。

**默认 `.cas` 模式（推荐，最严谨）**——从 Fluent `.cas` 读体单元，再把体点云标量高斯插值到网格节点：

```bash
conda activate GNN_vmtk

CASEDIR=outputs/field/postview/v3p_i6diag_t016_report/GUO_XI_JIANG__result_features_merged-1146
python tools/cfdpost_cloud_export/build_sliceable_volume.py \
  --cas data_new/AG/slow/GUO_XI_JIANG/GUO_XI_JIANG.cas.gz \
  --source $CASEDIR/GUO_XI_JIANG__volume_merged-1146.vtp \
  --output $CASEDIR/GUO_XI_JIANG__volume_merged-1146.vtu \
  --cas-scale 1000 --radius 2.0 --sharpness 2.0 --fallback nearest
```

- `--cas-scale 1000`：Fluent `.cas` 多为米，点云为 mm，必须 ×1000 对齐（默认即 1000）。
- 验证：日志打印 `单元数 > 0`；节点数应等于 `.cas` 网格节点（≠ 来源点数，说明确实挂到了体单元）。
- `--interior-only`：仅用 `is_wall==0` 的点，去壁面只看腔内场（可选）。

**无 `.cas` 兜底**——直接对体点云 Delaunay3D 四面体化（凸包会在凹陷/分叉外侧补料，精度不如 `.cas`）：

```bash
python tools/cfdpost_cloud_export/build_sliceable_volume.py \
  --delaunay --alpha 4.0 \
  --source $CASEDIR/_export/*__all.csv \
  --output $CASEDIR/GUO_volume_delaunay.vtu
```

**ParaView**：Open `.vtu` → Filters → **Slice**（选法向/原点）→ Coloring 选 `p_cfd`/`p_pred`/`err_p`/`vel_mag_cfd` → 即得连续填充截面。多切面用 Slice 的 Plane 偏移或 Filters → Clip。

---

## 4. 出图规范

### 4.1 三联图布局

| CFD 真值 | GNN/CROWN 预测 | 误差 |
| --- | --- | --- |
| `wss_cfd` / `p_cfd` / `vel_mag_cfd` | `*_pred` | `err_*` 或 `abs_err_*` |

- CFD 与 Pred **同一色标范围**（`plot_stl_mapped_triptych.py` 默认共用 field range）
- 误差列单独色标
- 标题含：病例 · 帧号 · 变量 · 插值参数

### 4.2 ParaView 交互出图

读 `README_后处理打开说明.md`：`Representation=Surface` · 导入 `GNN_blue_white_red.xml` · CFD/Pred 关 separate color scales。

### 4.3 交付目录约定

```text
outputs/field/postview/<RUN_TAG>/
├── README.md                    # 批次说明（可选，多 run 对照时写）
├── <CASE>__result_features_merged-1146/
│   ├── <CASE>__surface_wall.vtp
│   ├── plots/fig_{wss,p,vel_mag}_triptych.png
│   └── manifest_bundle.json
└── V3P_vs_CROWN_对照表_1146.md   # 跨方法并排时
```

路径与 `manifest.json` 写入 `docs/02-推进与变更/代码修改与实验推进记录.md` 文首（与 `实验分析对比图目录说明.md` 的 analysis_compare **分开**）。

---

## 5. 质量控制（出图前必查）

复制 checklist 并逐项确认 → [reference.md §QC](reference.md#qc-出图前清单)

最低要求：
- [ ] `mapping_report` 中 `valid_ratio` ≥ 95%
- [ ] STL 与 CSV 无错位/镜像
- [ ] CFD 与 Pred 同插值法、同参数
- [ ] 图注含帧号 + 插值参数 +「指标为 pooled CSV」
- [ ] 正式 R² 仍来自 run `summary.json`，非面片 VTP

**截面出图关键坑**：单截面内变量跨度（几十 Pa）远小于全腔全局范围（压力 ~3000 Pa），若色标用全局范围，截面会几乎一个颜色。出截面图时务必把色标 **Rescale 到该截面的局部/可见范围**（ParaView: Rescale to Visible Data Range），才能复现 Fluent 那种丰富的截面云图。（另注：某些 run 的 `vel_mag_pred` 可能退化为常数；压力单位为 Pa，转 mmHg 需 ÷133.322。）

---

## 6. 任务完成检查

- [ ] 产物路径可打开（VTP 或 PNG 存在）
- [ ] `mapping_report.json` 已读并记录 coverage
- [ ] 需要跨方法对比时更新 `outputs/field/postview/*对照表*.md`
- [ ] 推进记录文首追加条目（日期 · run · 病例 · 帧 · 产物路径）

---

## 7. 禁止

- 用 **1120** 作主汇报帧（除非用户明确要求对照）
- 在插值面片上算正式 R² 与 pooled 指标混报
- CFD 真值（Fluent 原生网格）与 GNN 插值面片**不说明口径差异**就并排
- 把 CROWN/V3P postview 数字写入 V3 母版实验表（见 `docs/00-规范与记录/外部baseline实验记录规范.md`）
- 未经用户明确要求代跑 WSS 批量对比脚本（见 workspace wss 规则）

---

## 8. 相关脚本与文档

| 资源 | 路径 |
| --- | --- |
| 工具包 README | `tools/cfdpost_cloud_export/README.md` |
| 壁面点云 → STL 面片 VTP | `tools/cfdpost_cloud_export/map_to_stl_surface.py` |
| 体点云 → 可切面 .vtu | `tools/cfdpost_cloud_export/build_sliceable_volume.py` |
| 体点云合并 pred/cfd → VTP | `tools/cfdpost_cloud_export/export_volume_merged_vtp.py` |
| 三条路线 A/B/C | `tools/cfdpost_cloud_export/三条对比路线.md` |
| postview 索引 | `outputs/field/postview/README.md` |
| **落盘目录规范** | `docs/00-规范与记录/点云回插面片可视化目录说明.md` |
| 方法论文档 | `docs/paper_reproduction/05-点云预测值与真值回插到面片方法.md` |
| 精度分析对比图（另一套） | `docs/00-规范与记录/实验分析对比图目录说明.md` |
