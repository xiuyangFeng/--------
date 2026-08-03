# AAA / WSS 数字孪生实验仓库

本仓库包含血管几何预处理、WSS 场重建训练，以及 velocity→WSS 物理估算对照。
**当前日常主攻**是 WSS-only 最小化线与 `wss_mri_calculator` CFD 适配；任务 A 的 V3 训练栈仍在 `training/` + `docs/01-任务/任务A/03-V3路线/`。

## 当前主攻（先看这里）

| 路线 | 代码 | 数据 / 产物 | 说明入口 |
| --- | --- | --- | --- |
| **WSS-min 预处理** | [`pipeline_wss_min/`](pipeline_wss_min/) | `data_wss_min/` | [`pipeline_wss_min/README.md`](pipeline_wss_min/README.md) |
| **WSS-min 训练** | [`training_wss_min/`](training_wss_min/) | `outputs/wss_min/`（及本目录 `runs/`） | [`training_wss_min/README.md`](training_wss_min/README.md) |
| **velocity→WSS 估算** | [`wss_mri_calculator/`](wss_mri_calculator/) | `outputs/wss_pinn/audits/…`、本目录 `experiments/` | [`README_CFD_ADAPTATION.md`](wss_mri_calculator/README_CFD_ADAPTATION.md) |
| 峰值体域 `u,v,w,p` PINN | [`wss_pinn/`](wss_pinn/) | `data_wss_pinn/volume_uvwp_peak_v1_*`、`outputs/wss_pinn/volume_uvwp_peak_v1/` | [`wss_pinn/README.md`](wss_pinn/README.md) · [`docs/.../WSS_PINN/`](docs/02-推进与变更/WSS_PINN/README.md) |

文档总索引：[`docs/README.md`](docs/README.md)。
WSS-min 推进记录（专用，勿混入 V3 大日志）：[`docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`](docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md)。

### 状态速览（2026-08）

- **WSS-min 数据**：v4 活动口径；AG76（`stl_landmarks_v4`）；AAA 几何签核 63 / 训练白名单 57；ILO 术前审核通过 41（未进正式 split）。产物独立于 `data_new/`。
- **WSS-min 训练**：单 seed 开发锚点为 **LSA2 SAME-H2 + `log(local_radius)`**（`R²_cb≈0.3506`）；暂不做多 seed。矩阵真源见 [PointNet baseline 进度跟踪](docs/02-推进与变更/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)。
- **wss_mri_calculator**：原 MRI 仓库已做 CFD 点云适配；正式对比病例池绑定 WSS-PINN **train138/test35（173 例）**。adaptive-CV **v1** 已冻结；多尺度 **v2** 在 test35 上相对 v1 raw R² mean 约 `+0.009`。默认模式仍是 v1，v2 需显式指定。
- **峰值体域 PINN**：首轮 SEP 8 臂 `11128` 已完成并审计；第二轮 random SAME5K、固定 7500 epoch / 517,500 step 的 GPU preflight `11137` 已完成，训练数组 `11138_[0-7%4]` 运行中；首批四臂30分钟健康监控通过，Slurm 时间限制为0（GPU 分区 `MaxTime=UNLIMITED`）。

---

## 1. WSS-only 最小化（`pipeline_wss_min` + `training_wss_min`）

任务：几何点云 `x,y,z[+几何]` → 壁面 WSS；坐标架为原始 STL landmark **v4**；全局统计 train-only、峰值步、`log_z`。

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# 预处理四阶段（写 data_wss_min/，不改 pipeline/ 与 data_new/）
$PY -m pipeline_wss_min.run --stage preprocess
$PY -m pipeline_wss_min.run --stage qa-gate
$PY -m pipeline_wss_min.run --stage global-stats --stats-timesteps peak
$PY -m pipeline_wss_min.run --stage build-samples

# 训练 / 评估（只读 data_wss_min）
$PY -m training_wss_min.train    --config <config.json>
$PY -m training_wss_min.evaluate --config <config.json>
```

更多说明：

- 预处理：[`pipeline_wss_min/README.md`](pipeline_wss_min/README.md) · Agent 约束：[`pipeline_wss_min/AGENTS.md`](pipeline_wss_min/AGENTS.md)
- 训练：[`training_wss_min/README.md`](training_wss_min/README.md)
- 进度：[PointNet baseline 矩阵](docs/02-推进与变更/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md) · [训练实验跟踪](docs/02-推进与变更/WSS最小化_训练实验跟踪.md)
- 历史诊断归档：[`docs/02-推进与变更/_archive/WSS最小化/`](docs/02-推进与变更/_archive/WSS最小化/)

---

## 2. velocity→WSS：`wss_mri_calculator`

上游原项目面向 4D Flow MRI；本仓库在其上增加了 **Fluent 非结构化点云** 适配（`src/*_cfd.py`、`src/wss_multiscale.py`），原 MRI 入口未改。可视化在 `viz/`，算子在 `src/`。

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
cd wss_mri_calculator/src

# 单病例（默认读 data_wss_min bundle 的 peak_step；默认 neighbor-mode=adaptive-CV v1）
$PY calculate_wss_cfd.py --case-dir /public/newhome/cy/Digital_twin/GNN/data_new/AG/slow/LIU_JIN_LIANG

# 多尺度 v2（需显式指定，避免改动已冻结 v1 口径）
$PY calculate_wss_cfd.py --case-dir <病例目录> --neighbor-mode multiscale_v2

# 正式批量：仅 PINN 冻结 173 例（勿对全库随机抽样当下结论）
$PY batch_validate_cfd.py \
  --json-out ../../outputs/wss_pinn/audits/cfd_velocity_wss_explore/pinn173_peak.json
```

| 文档 / 实验 | 内容 |
| --- | --- |
| [`wss_mri_calculator/README.md`](wss_mri_calculator/README.md) | 上游 MRI 原说明（clone / demo） |
| [`README_CFD_ADAPTATION.md`](wss_mri_calculator/README_CFD_ADAPTATION.md) | **本仓库 CFD 适配与批量入口（必读）** |
| [`experiments/pointcloud_adaptive_v1/`](wss_mri_calculator/experiments/pointcloud_adaptive_v1/) | adaptive-CV v1 冻结结果 |
| [`experiments/pointcloud_multiscale_v2/`](wss_mri_calculator/experiments/pointcloud_multiscale_v2/) | 多尺度 v2 冻结结果 |

依赖：`GNN` conda 环境即可（CFD 路径不强制 pyvista）。

---

## 3. 目录导航

| 路径 | 职责 |
| --- | --- |
| [`pipeline_wss_min/`](pipeline_wss_min/) | WSS-only 最小化预处理（正式四阶段） |
| [`training_wss_min/`](training_wss_min/) | WSS-min PointNet / PointNeXt 训练与评估 |
| [`wss_mri_calculator/`](wss_mri_calculator/) | MRI WSS 计算器 + CFD 点云适配 / 实验 |
| [`wss_pinn/`](wss_pinn/) | 峰值体域 `u,v,w,p` 的 PointNet / PointNet++ data-only 与非牛顿 PINN 活动实现；旧 WSS-target 路线已归档 |
| [`pipeline/`](pipeline/) | 历史主线几何/图数据流程（`data_new/`） |
| [`training/`](training/) | 任务 A V1/V2/V3 场重建训练 |
| [`external_baselines/`](external_baselines/) | 外部论文复现（如 PointNetCFD、CROWN） |
| [`docs/`](docs/) | 实验总纲、路线文档、推进记录 |
| [`legacy/`](legacy/) | 归档脚本；非稳定入口 |
| `data_new/` · `data_wss_min/` · `stl_data/` | 原始与处理数据（勿随意重排） |

---

## 4. 其他常用入口

### 任务 A / V3

见 [`docs/01-任务/任务A/03-V3路线/README.md`](docs/01-任务/任务A/03-V3路线/README.md) 与 [`docs/README.md`](docs/README.md)。推进记录：[`docs/02-推进与变更/代码修改与实验推进记录.md`](docs/02-推进与变更/代码修改与实验推进记录.md)。

### 旧 `pipeline/` 完整流程

```bash
conda activate GNN
python -m pipeline.audit_inputs --groups AAA AG ILO
python -m pipeline.run_all --case ZHANG_CHUN \
  --geometry-python /public/newhome/cy/.conda/envs/GNN_vmtk/bin/python
```

日志：`data_new/pipeline_reports/logs/run_all.log`；单病例步骤日志在
`data_new/<病例路径>/processed/logs/progress.log`。
几何步骤需要 `GNN_vmtk`；其余用 `GNN`。详见 [`pipeline/README.md`](pipeline/README.md)。

### 外部 baseline（PointNetCFD 示例）

```bash
conda activate rag_venv   # 或文档指定环境
python -m external_baselines.pointnetcfd.train \
  --config external_baselines/pointnetcfd/configs/pointnetcfd_original_vp.json \
  --dry-run
```

详见 [`external_baselines/pointnetcfd/README.md`](external_baselines/pointnetcfd/README.md)。

### 历史脚本

```bash
python -m legacy.preprocess.batch_process --help
```

`pipeline.extract_features` 依赖 [`pipeline/vmtk_core.py`](pipeline/vmtk_core.py)；`legacy/preprocess/vmtk_core.py` 仅为兼容层。

---

## 约定摘要

- 集群批量任务按 `pipeline/cluster/` / 各子目录 `cluster/` 提交；登录节点不做大规模训练或批处理。
- 日志至少细到病例级，便于判断作业是否仍在推进。
- WSS **对比/批量评估**未明确要求时不代跑（见项目 `wss.mdc`）；文档与代码路径可照常维护。
- 改 WSS-min / `wss_mri_calculator` 相关代码或文档后，在 [`WSS最小化_代码修改与实验推进记录.md`](docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md) **文首**追加记录。
