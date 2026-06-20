# CROWN/Beihang 脑血管 PointNet-PINN 私有数据适配计划

> 适用代码：`external_baselines/CROWN_Beihang/`
> 适用论文：Physics-informed neural networks for three-dimensional cerebrovascular hemodynamic prediction: A point cloud preprocessing strategy based on limited data
> 当前状态：raw_ascii v1 预处理 **5738 完成** · **5739 No-Go（OOM 截断）** · **5740 evaluate 后 No-Go** · lazy 加载优化 **已审核并用于 5751 重训** · 旧 5625/5626 **作废**（见推进记录）。

## 1. 源码当前结构梳理

| 文件 | 作用 | 备注 |
| --- | --- | --- |
| `source_snapshot/sampling_buid_dataset.py` | 体素化参考实现 | 可运行逻辑见 `crown_beihang/voxelize.py` |
| `source_snapshot/model_train_*.py` | 原训练入口 | 已重构为 `crown_beihang/train.py` |

## 2. 论文方法和本项目边界

这篇论文的核心是“有限病例 + 点云预处理 + PointNet-style 全局特征 + PINN 物理残差”，监督目标是 `Vx,Vy,Vz,P`，不是 WSS。第一轮适配应严格仿照论文，先把本项目私有数据上的完整速度场和压力场预测跑通：即点级 `u,v,w,p` 全场回归、PINN 与非 PINN 对照、测试集速度/压力指标和预测 CSV/NPZ 导出。

WSS 暂不进入第一轮代码任务。合理逻辑是：如果速度场和压力场预测足够稳定，后续再评估两条 WSS 路线：一是基于预测速度场做近壁梯度/后处理推导 WSS；二是在 CROWN 架构上新增 WSS 预测头或多任务监督。FR/PD 和 WSS 都属于第二阶段或后处理阶段，不能提前写成第一轮复现结果。

本项目使用它时应分成两类结论：

| 类型 | 可以主张 | 不能主张 |
| --- | --- | --- |
| 第一轮源码复现 | 在私有 AG/AAA/ILO 数据上复现论文的 `u,v,w,p` 点云映射，比较 PINN 与非 PINN | 加 WSS head、改成 WSS baseline、混入 V3 内部路线 |
| 几何特征保留 | 预处理阶段保留显式几何特征，方便后续消融和第二阶段增强 | 第一轮 paper-original 配置默认不启用几何输入 |
| 后续 WSS 扩展 | 若 `u,v,w,p` 准确，再做速度后处理 WSS 或新增 WSS 监督头 | 未走完整 WSS 链前，不报告为 WSS 复现结果 |

## 3. 新预处理结果目录

为避免污染本项目原有 `processed/`、`graphs/`、`outputs/field/`，CROWN 私有数据适配单独使用：

```text
external_baselines/CROWN_Beihang/private_preprocessed_raw_ascii_v1/   # 当前论文口径（raw_ascii 体素化）
external_baselines/CROWN_Beihang/private_preprocessed/                # 旧链路（features 15000，只读对照）
├── pkl/                      # crown_volume_* / crown_all_* 或 partial/
├── manifests/
├── stats/
└── audit/                    # preprocess_cases.jsonl · pilot_timing.json · logs/
```

该目录只保存 CROWN 论文适配产生的中间数据。原先项目预处理结果继续保留在原路径，不覆盖、不迁移、不改名。

## 4. 中间数据层设计

**输入源（2026-06-17 v2）**：`data_new/AG/<case>/ascii_in/`（及 `ascii/` 壁面，当 `point_filter=all`）**全量 Fluent 体点**，不经 pipeline FPS 15000。

**体素化**：`crown_beihang/export_pkl.py` · `export.source=raw_ascii` · `voxel_size_mm=0.05` · `point_filter=volume|all`（`volume` 等价论文 interior）。

**训练采样**：体素化 pkl 存**全部体素点**；`train.py` 每 epoch **随机 10000**（与源码一致）。

**显式几何（非第一轮）**：不在全量体素 export 预挂；后续 `crown_geom_vp` 在**训练循环内对已采样 10000 点**做中心线/几何查询。`export.source=features` 仅留几何消融。

**归一化**：`u,v,w` 物理量；`p` train-only min-max（`stats/train_stats.json`）；PINN 坐标 `×1000`；壁面 loss 用速度阈值 0.01（非 `is_wall`）。

第一轮 raw_ascii pkl 记录（`feature_names=["x","y","z"]`）：

```python
{
    "case_name": str,
    "time_index": int,
    "split": "train" | "val" | "test",
    "point_filter": "volume" | "all",
    "export_source": "raw_ascii",
    "feature_names": ["x", "y", "z"],
    "target_names": ["u", "v", "w", "p"],
    "features": np.ndarray,   # (3, N_voxel)
    "targets": np.ndarray,    # (4, N_voxel)
    "global_cond": np.ndarray,
}
```

核心原则：

- 第一轮 paper-original：`input_features=["x","y","z"]`，不启用几何列。
- 几何增强消融在采样 10000 点之后挂载，与 V3 pipeline 全量几何 export 分离。
- 归一化统计 train-only；测试指标含 MAE/RMSE 与**点级 R²**。
- manifest/audit 记录 split、点数、`n_raw`/`n_voxel`、导出 Job ID。

## 5. 配置 mask 方案

建议新增 CROWN 专用配置，而不是把输入列写死在训练脚本里：

```json
{
  "experiment_name": "crown_geom_vp_nopinn_split_AG_v1_seed1",
  "data": {
    "root": "external_baselines/CROWN_Beihang/private_preprocessed/pkl",
    "train_pkl": "crown_train.pkl",
    "val_pkl": "crown_val.pkl",
    "test_pkl": "crown_test.pkl",
    "input_features": ["x", "y", "z"],
    "target_features": ["u", "v", "w", "p"],
    "sample_points": 10000,
    "sample_policy": "random_each_epoch",
    "use_wall_mask": true
  },
  "model": {
    "input_dim": 3,
    "output_dim": 4,
    "hidden": [256, 512, 512, 256],
    "global_pool": "max"
  },
  "physics": {
    "enabled": false,
    "reynolds": 300,
    "use_explicit_wall_mask": true
  }
}
```

第一轮 paper-original 复现只要求 `crown_original_vp` 和 `crown_original_vp_pinn` 跑通。几何特征消融配置可以先准备，但不作为第一轮必须提交的集群任务；后续只需要改 `input_features` 和 `model.input_dim`：

| 配置 | `input_features` | 用途 |
| --- | --- | --- |
| `crown_original_vp` | `x,y,z` | 最接近论文源码的 paper-original 对照 |
| `crown_wall_vp` | `x,y,z,is_wall` | 检查壁面标记是否改善近壁速度/压力 |
| `crown_geom_vp` | `x,y,z,Abscissa,NormRadius,Curvature,Tangent_X,Tangent_Y,Tangent_Z,is_wall` | 检查原 V3 显式几何特征 |
| `crown_geom_gall_vp` | 上一组 + `dist_to_bifurcation,branch_id,dR_ds,torsion,d_tangent_ds,dist_to_wall` | 检查 Line G 几何扩展特征 |

给后续 Agent 的边界：第一轮可以生成几何配置文件，但不要默认提交几何增强训练；默认提交清单只包含 `crown_original_vp` 与 `crown_original_vp_pinn`。

## 6. PINN 版本适配要点

PINN 版本不要只把 `model_train_pinn.py` 直接换数据路径运行，需要先修下面几处口径：

1. `get_pinn(model, pv_phy)` 当前假设模型输入只有 `x,y,z`。当启用几何特征时，应构造 `model_input = concat(xyz_requires_grad, selected_noncoord_features)`，并明确残差只对 `x,y,z` 求导。
2. 非坐标几何特征来自离线预处理，不是 autograd 可微函数。因此几何增强 PINN 的物理残差是“对坐标通道的偏导近似”，不能写成严格完整的几何可微 PINN。
3. `loss_wall` 当前通过 `labels[:,0:3]^2 <= 0.01` 推断壁面，必须改为读取 `wall_mask`。
4. `Re=300`、血液密度/黏度、坐标单位缩放要写进配置；当前源码里 PINN 版本把坐标乘 `1000`，这会直接影响二阶导数量级。
5. `lphy=epoch_data/epoch_func` 可能极端放大或缩小物理项，集群第一轮建议同时记录 `loss_data/loss_phy/loss_wall/lphy`，并设置上下界。

## 7. 第一轮实施计划

### 阶段 A：只读审计和源码整理

- 保留原始 `model_train_1.py`、`model_train_pinn.py` 作为 source snapshot。
- 把共用模型抽到一个 CROWN 专用模块，训练和测试共用同一 `get_model(input_dim, output_dim)`。
- 修正未定义返回变量、硬编码路径、硬编码 batch size、硬编码 `cuda:0`、训练输出目录覆盖等工程问题。
- 新增 `configs/local/`，所有实验只通过配置切换。

### 阶段 B：私有数据导出

- 输入：项目已有预处理后的特征 CSV 或等价中间层，要求包含 `u,v,w,p`、`is_wall` 和显式几何列。
- 输出：`private_preprocessed/pkl/crown_train.pkl`、`crown_val.pkl`、`crown_test.pkl`。
- 同步输出：`manifests/split_*.json`、`stats/train_stats.json`、`audit/preprocess_audit.json`。
- 采样：先保留每个样本全量点，训练时沿用源码每 epoch 随机抽 `10000` 点；若集群显存不稳，再加固定 10000 点快照。

### 阶段 C：非 PINN 跑通

- 先跑 `crown_original_vp`，确认 CROWN 架构在私有数据上能正常学习 `u,v,w,p`。
- 第一轮不要求提交 `crown_wall_vp`、`crown_geom_vp`；这些只作为后续消融候选。
- 指标至少输出 `rmse/r2/mae` for `u,v,w,p`、`vel_mag`，并按 wall / near-wall / interior 分区汇总。

### 阶段 D：PINN 跑通

- 先跑 `crown_original_vp_pinn`，只用 `x,y,z`，严格对齐论文源代码逻辑。
- 第一轮 PINN 只提交 paper-original 输入；`crown_geom_vp_pinn` 作为后续探索组，暂不要求实现或提交。
- 必须单独保存 `loss_data/loss_pde/loss_continuity/loss_wall/lphy`，否则无法解释 PINN 是否真正贡献。

### 阶段 E：结果回收与是否进入第二阶段

- 如果 `u,v,w,p` 指标足够稳定，第二阶段再考虑走速度后处理 WSS 或新增 WSS 预测头。
- 如果速度场不稳定，CROWN 只作为“脑血管 PointNet-PINN 复现对照”，不进入 WSS 主线。
- 所有结果只回填到外部 baseline 目录，不混入 V3 内部优化结论。

## 8. 第一轮实验矩阵

第一轮目标是完整复现论文的速度场和压力场预测，不做 WSS 预测头，不做 WSS 后处理，不默认启用显式几何输入。预处理结果中仍保留几何列，保证后续消融可以直接复用同一套数据。

### 8.1 第一轮必跑矩阵

| 实验名 | PINN | 输入列 | 输出 | 目的 |
| --- | --- | --- | --- | --- |
| `crown_original_vp` | 否 | `x,y,z` | `u,v,w,p` | paper-original 非 PINN 对照 |
| `crown_original_vp_pinn` | 是 | `x,y,z` | `u,v,w,p` | paper-original PINN 对照 |

### 8.2 后续可选矩阵

| 实验名 | PINN | 输入列 | 输出 | 目的 |
| --- | --- | --- | --- | --- |
| `crown_wall_vp` | 否 | `x,y,z,is_wall` | `u,v,w,p` | 壁面标记消融，检查近壁速度/压力是否改善 |
| `crown_geom_vp` | 否 | `x,y,z` + V3 基础几何 + `is_wall` | `u,v,w,p` | 显式几何特征消融主组 |
| `crown_geom_vp_pinn` | 是 | `x,y,z` + V3 基础几何 + `is_wall` | `u,v,w,p` | 几何 + PINN 探索组 |
| `crown_vp_to_wss_post` | 可选 | 已训练 `u,v,w,p` 预测 | 后处理 `wss` | 基于速度场近壁梯度推导 WSS |
| `crown_geom_vp_wss_head` | 可选 | 几何输入 + `is_wall` | `u,v,w,p,wss_*` | 多任务监督或新增 WSS head |

后两组不是论文原始方法，必须另开 CROWN-WSS 变体并单独标注为“源码架构借鉴 + 本项目任务改造”。

## 9. 给后续 Agent 的执行交接

后续 Agent 如果负责代码修改或集群提交，应按下面边界执行：

1. 先实现 CROWN 私有数据 adapter，输出 `private_preprocessed/pkl/crown_train.pkl`、`crown_val.pkl`、`crown_test.pkl` 和对应 manifest/audit/stats。
2. adapter 必须保留全量几何列，但 `crown_original_vp*` 配置的 `input_features` 只能是 `x,y,z`。
3. 训练代码必须把模型输入维度配置化，不能继续硬编码 `input_channel = 3` 后又在几何配置里直接拼列。
4. 第一轮只提交 `crown_original_vp` 与 `crown_original_vp_pinn` 两个集群任务。
5. 测试与评估必须导出完整 `u,v,w,p` 预测结果、逐变量指标、`vel_mag` 指标、按病例指标和预测文件。
6. 不要在第一轮实现 WSS head；不要把速度/压力结果直接汇报成 WSS baseline。

## 10. 集群提交前检查清单

- [ ] `private_preprocessed/audit/preprocess_audit.json` 中 train/val/test 病例数、点数、列名、缺失率通过。
- [ ] `stats/train_stats.json` 只来自 train split。
- [ ] `input_features` 与 `model.input_dim` 一致。
- [ ] PINN 配置的坐标单位、`Re`、`lphy` 上下界写清楚。
- [ ] 测试脚本读取同一份配置和同一份模型定义。
- [ ] 输出目录包含 `config.json`、`train_loss.csv`、`best_epoch.pth`、`metrics_test.json`、`metrics_test_by_case.csv`。
- [ ] 实验记录进入 `external_baselines/CROWN_Beihang/experiments/`，矩阵跑齐后再更新 `梳理记录.md`。
- [ ] 第一轮集群提交清单只包含 `crown_original_vp` 与 `crown_original_vp_pinn`；几何增强和 WSS 变体不混入第一轮。

## 11. 当前判断

CROWN/Beihang 源码第一轮只做“脑血管小样本 PointNet-PINN 速度/压力复现线”：完整预测 `u,v,w,p`，比较 PINN 与非 PINN，判断该论文路线在本项目私有数据上的速度场/压力场可学性。2026-06-20 的执行口径是：5739 只作为 OOM 截断 No-Go，5740 已作为 PINN No-Go 记录，5751 完成后再重新判断非 PINN raw_ascii 充分训练上限。若非 PINN 充分训练结果仍不稳定，CROWN 只作为外部速度/压力复现对照；若结果好，再进入第二阶段 WSS：可以从预测速度场推导 WSS，也可以加 WSS head 或用 WSS 数据监督。适配时最重要的是把原论文源码复现、私有数据中间层、显式几何特征保留、后续 WSS 扩展分开。这样交给另一个 Agent 执行时，不会把第一轮任务扩大成 WSS 改造。
