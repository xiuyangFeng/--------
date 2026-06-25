---
name: postview-surface-viz
description: >-
  GNN/CROWN 点云标量回插 STL 面片并生成病例级可视化（postview 交付包、三联图、ParaView 包）。
  在用户要求后处理可视化、面片云图、点云插值到 STL、postview、merged-1146 汇报图、
  CFD|Pred|Error 三联图时使用。遵循 docs/paper_reproduction/05-点云预测值与真值回插到面片方法.md。
---

# 点云 → 面片后处理可视化（postview）

## 触发

「后处理可视化」「面片云图」「回插 STL」「postview」「merged-1146 图」「三联图」「ParaView 包」等；或给出 `manifest.json` / CROWN checkpoint + 病例名。

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
wss_cfd,wss_pred,err_wss,abs_err_wss,p_cfd,p_pred,err_p,abs_err_p
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
| 三条路线 A/B/C | `tools/cfdpost_cloud_export/三条对比路线.md` |
| postview 索引 | `outputs/field/postview/README.md` |
| **落盘目录规范** | `docs/00-规范与记录/点云回插面片可视化目录说明.md` |
| 方法论文档 | `docs/paper_reproduction/05-点云预测值与真值回插到面片方法.md` |
| 精度分析对比图（另一套） | `docs/00-规范与记录/实验分析对比图目录说明.md` |
