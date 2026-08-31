# `outputs/field` 冗余审查与可再生性清理方案（已执行）

> 审查日期：2026-08-30  
> 执行状态：**2026-08-30 已完成，本文归档作为执行证据**  
> 删除原则：仅删除已用当前代码、当前数据与现存 checkpoint 验证可再生的产物；历史代码或历史图资产不兼容时，优先归档而不是删除。

## 0. 执行结果

- 构建 `outputs/field/_reproducibility/legacy_graph_snapshot_pre_20260609/`：17 例 / 1377 帧，约 0.68 GiB。
- 6 个旧 checkpoint 通过当前 `predict_field --legacy-snapshot` 重算，输入完全一致，field/WSS 均通过 `allclose(rtol=1e-5, atol=1e-6)`。
- 删除 14 个大型 prediction 目录中 17,820 个逐样本 `.pt`，共 85,885,367,175 字节（79.987 GiB）。
- 保留全部 prediction manifest、run config、checkpoint、summary、history 和紧凑 evaluation。
- 删除 4 个空目录；8 个失败启动移入 `outputs/field/_archive/aborted_starts_20260830/`；54 个当前代码不兼容 run 移入 `outputs/field/_archive/legacy_pre_wss_schema/`。
- `outputs/field` 从约 82.98 GiB 降到约 3.7 GiB。删除执行记录见 `outputs/field/_reproducibility/cleanup_execution_20260830.json`。
- `predict_field.py` 新增 `compact/full`、`--legacy-snapshot` 和 `--max-samples`；默认 compact，防止再次出现同类空间膨胀。

以下章节保留执行前审查口径与分级依据。

## 1. 执行前总体结论

- `outputs/field/` 当前约 **82.98 GiB**，共 222 个一级目录。
- 129 个 `predictions_*` 目录合计 **80.174 GiB**，是几乎全部的空间来源。
- 其中 14 个大型预测目录占 **79.995 GiB**；其余 115 个预测目录总共只有 **0.179 GiB**，不值得为节省空间批量删除。
- 已证明可用当前 `training.scripts.predict_field` 路径数值再生的大型预测 `.pt` 为 **43.629 GiB**，可作为第一批清理对象。
- 另有 **36.358 GiB** 旧预测使用了已变更的历史图资产；执行前不能直接删除，后续已通过 legacy snapshot 固化后完成清理。
- 198 份 run 配置中，194 份可由当前 `ExperimentConfig` 解析；4 份旧 PINN 配置含已移除字段 `physics.loss_mode`。另有 53 个 2026-03 旧 MLP/GraphSAGE/Transformer checkpoint 与当前输入维度和 head 命名不兼容。这 57 个 legacy 目录合计仅 **0.263 GiB**，应保留或归档。

## 2. 为什么预测目录会达到 80 GiB

`training/scripts/predict_field.py` 每个时间步导出一个 `.pt`，除了 `y_pred` / `y_wss_pred`，还重复保存：

- `x`
- `global_cond`
- `edge_index`
- `y_true`
- `y_wss_true`
- `wall_mask`

一个 15,000 节点样本约 4.596 MiB，其中 `edge_index` 约 2.747 MiB，而真正的场预测和 WSS 预测各约 0.229 MiB。同一病例 81 个时间步又重复保存静态 `x` / `edge_index`，因此完整 test 导出会膨胀到约 5.8–6.2 GiB。

## 3. 第一批：可删除逐样本 `.pt`（约 43.629 GiB）

下表目录的配置、split、数据根目录和 checkpoint 均存在，当前模型可完整加载 checkpoint。代表样本的当前输入与原导出 `x/y/edge_index/global_cond` 一致，重算预测与原预测在 `rtol=1e-5, atol=1e-6` 下全部 `allclose`。

| run / prediction 目录 | 可删除 `.pt` |
| --- | ---: |
| `field_v3_pointnext_i6diag_..._20260619_174001/predictions_test/` | 5.817 GiB |
| `field_v3_pointnext_i6diag_..._20260619_174001/predictions_test_best_wss/` | 5.817 GiB |
| `field_v3_pointnext_i6diag_..._20260619_174001/predictions_val_best_wss/` | 2.909 GiB |
| `field_v3_pointnext_k1branchfeat_..._20260701_150937/predictions_test/` | 5.817 GiB |
| `field_v3_pointnext_k1branchfeat_..._20260701_150937/predictions_test_best_wss/` | 5.817 GiB |
| `field_v3_pointnext_localpool_main01_..._20260609_124213/predictions_test_best_wss/` | 5.817 GiB |
| `field_v3_pointnext_j4v3dlite_..._20260630_131820/predictions_test_best_wss/` | 5.817 GiB |
| `field_v3_pointnext_j5activedata_..._20260701_002549/predictions_test_best_wss/` | 5.817 GiB |

数值证据：

- I6：`best_model` 代表样本完全一致；`best_wss_model` 的 field / WSS 最大绝对误差不超过 `5.37e-7 / 1.08e-6`。
- K1：`best_model` 和 `best_wss_model` 代表样本均完全一致。
- 2026-06-09 Main：代表样本完全一致。
- J4：field / WSS 最大绝对误差为 `9.54e-7 / 7.16e-7`。
- J5：field / WSS 最大绝对误差为 `7.16e-7 / 8.95e-7`。

清理时只删除这些目录中的逐样本 `.pt`，保留：

- `manifest.json`
- `manifest_surface_compare_partial.json`（如存在）
- run 根目录中的 checkpoint、`config.snapshot.json`、`split.snapshot.json`、`run_manifest.json`、`summary.json`、`history.csv`
- `evaluation/`、`fig_training_curves.png`、`outputs/field/f0_decision/` 中的紧凑诊断结果

K1/J4/J5 虽没有 run 内 `evaluation/`，但 `summary.json` 已保存 test/best-WSS 完整指标，且已有以下紧凑证据：

- `outputs/field/f0_decision/v3p_l3_subset_eval_20260630.json`
- `outputs/field/f0_decision/v3p_l3_subset_eval_j5_20260701.json`
- `outputs/field/f0_decision/v3p_l3_subset_eval_k1_bestmodel_20260704.json`
- `outputs/field/f0_decision/v3p_l3_subset_eval_k1_bestwss_20260704.json`
- `outputs/field/f0_decision/v3p_k1_branch_error_probe_20260701.json`

## 4. 第二批：必须先做历史输入快照（约 36.358 GiB）

下列六个预测目录的 checkpoint 仍可被当前模型加载，但它们使用的图资产与现有 `processed/graphs` 不同：

| run / prediction 目录 | 当前状态 |
| --- | --- |
| `field_v3_pointnext_localpool_main01_..._seed1_20260522_124946/predictions_test_best_wss/` | 保留，旧图资产 |
| `field_v3_pointnext_localpool_main01_..._seed2_20260523_124511/predictions_test_best_wss/` | 保留，旧图资产 |
| `field_v3_pointnext_localpool_main01_..._seed3_20260523_124511/predictions_test_best_wss/` | 保留，旧图资产 |
| `field_v3_pointnext_localpool_fpc_pgradfeat_..._20260530_141452/predictions_test_best_wss/` | 保留，旧图资产 |
| `field_v3_pointnext_localpool_pcv2_blctx_..._20260602_145347/predictions_test_best_wss/` | 保留，旧图资产 |
| `field_v3_pointnext_localpool_main01_..._20260607_115628/predictions_test_best_wss/` | 保留，旧图资产 |

代表样本核验结果：

- 当前图与旧导出的 `x` 最大绝对差为 `0.066976`，`y_true` 和 `global_cond` 也已变化。
- 直接用当前图重算，field 最大绝对误差约 `0.283–0.796`，WSS 约 `0.202–0.669`，不属于数值浮动。
- 早期 4 个目录含 17 例 / 1377 帧；当前 denylist 会跳过 `slow/TE_JIN_WANG` 的 81 帧，默认只会输出 16 例 / 1296 帧。

同时，已证明这六个 run 的历史输入具有很高的去重潜力：

- 随机抽取的 12 个共同样本，六个 run 的 `x/global_cond/edge_index/y_true/y_wss_true` 完全一致。
- 17 个病例的首末时间步中，`x/edge_index/wall_mask` 全部为病例内静态。
- 若每例只保存一份静态图，每帧只保存 `global_cond/y_true/y_wss_true`，估算历史输入快照约 **0.677 GiB**。

因此第二批的正确清理顺序是：

1. 从一个完整 1377 帧旧导出中构建去重的 legacy input snapshot，固定 17 例历史输入。
2. 扩展当前预测代码，支持从该 snapshot 加载数据，并显式支持历史 denylist 病例。
3. 对六个 checkpoint 执行数值重算，确认与原 `.pt` 在指定容差内一致。
4. 再删除六个原始预测目录中的逐样本 `.pt`。

这一步完成后，第二批预计净释放约 **35.68 GiB**。

## 5. 必须保留的内容

- 所有有效 run 的 `config.snapshot.json`、`split.snapshot.json`、`run_manifest.json`、`summary.json`、`history.csv`。
- `best_model.pt`、`best_wss_model.pt`，以及实验结论所需的 `last_model.pt`。
- I6 的 20 个 `checkpoint_epoch_*.pt`：I7 checkpoint probe 依赖完整 epoch 序列，而 I6-a 配置直接依赖 `checkpoint_epoch_10.pt`，删除后无法在不重训的情况下重现。
- `evaluation/`、`outputs/field/f0_decision/`、`outputs/field/diagnostics/`、`outputs/field/plots/`、`outputs/field/postview/`。这些是结论、图件和路线判读的紧凑证据，体积显著小于原始预测。
- 115 个小型预测目录（合计 0.179 GiB）。其中包含旧格式和当前模型不兼容的早期结果，批量删除风险大于空间收益。
- 57 个当前配置或 checkpoint 不兼容的 legacy run（合计 0.263 GiB）。建议收入一个明确的 `_archive/legacy_pre_wss_schema/`，不建议删除。

## 6. 可归档或延后清理的小项

### 6.1 空目录

以下 4 个目录完全为空，无文档精确引用，可直接删除：

- `g4_2d_unwrap_overfit1c_s4_seed1_20260611_151055`
- `g4_2d_unwrap_overfit1c_s4_seed1_20260611_151203`
- `g4c_patch_overfit1c_fast__CHEN_SHI_MING_p12_r3.0_k20_seed1_20260630_110159`
- `g4c_patch_overfit1c_slow__GUO_XI_JIANG_p12_r3.0_k20_seed1_20260630_110243`

### 6.2 只有配置/空 history 的失败启动

根目录中还有 8 个小于 10 KiB、无 summary、无 checkpoint、无文档引用的失败启动目录。建议先移入 `_archive/aborted_starts_20260830/`，待一个阶段后再整包删除；这样可减少根目录噪声，又不会立即丢失旧配置。

- `field_transformer_opt05_pinn_steady_split_AG_v1_seed1_20260612_230102`
- `field_transformer_opt05_pinn_steady_cont_split_AG_v1_seed1_20260612_230102`
- `field_transformer_opt05_pinn_steady_contnoslip_split_AG_v1_seed1_20260612_230102`
- `field_transformer_pinn_a_pinn_01_split_AG_v1_seed1_20260611_192702`
- `field_transformer_pinn_a_pinn_01_split_AG_v1_seed1_20260611_192815`
- `field_transformer_pinn_a_pinn_01_split_AG_v1_smoke_seed1_20260611_193858`
- `field_v3_pointnext_j4v3dlite_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v3lite_v1_seed1_20260630_131154`
- `field_v3_pointnext_localpool_base01_nogeom_pw_wall13000_near2000_split_AG_v1_seed1_20260506_183801`

### 6.3 checkpoint 重复

300 个根目录模型文件合计约 2.006 GiB。按张量内容计算，仅有 9 组语义相同的 checkpoint，可节省约 0.058 GiB。由于文档和脚本可能依赖 `best_model.pt` / `best_wss_model.pt` / `last_model.pt` 的固定文件名，不建议为这点空间删除路径；如需极致去重，可用硬链接保留全部文件名。

## 7. 建议的执行顺序

1. **Phase A，低风险**：删除第 3 节 8 个目录中的逐样本 `.pt`，保留 manifest 和紧凑证据，释放约 43.63 GiB。
2. **Phase B，历史输入固化**：实现约 0.68 GiB 的 legacy input snapshot 和当前代码加载入口，验证六个旧 checkpoint 后，再释放约 35.68 GiB。
3. **Phase C，目录治理**：删除 4 个空目录，将 aborted/legacy 小运行收入 `_archive/`。
4. **Phase D，防止再次膨胀**：给 `predict_field.py` 增加 `compact/full` 导出模式；默认 compact 只保存预测和必要索引，只有任务 B/CFD 对照需要时才显式导出 full payload。

两批大型预测都完成后，理论净释放约 **79.31 GiB**，同时保留可复现的配置、checkpoint、历史输入和评估证据。

## 8. 本轮验证口径

- 环境：`/public/newhome/cy/.conda/envs/GNN/bin/python`
- 配置：`ExperimentConfig.from_json(...); config.validate()`
- 模型：`build_field_model_from_config(config)` + `load_state_dict`
- 数据：检查 split/data root/case graph 数量与 manifest，并用当前 `FieldGraphDataset` 特征 mask 口径加载代表样本
- 数值比较：比较 `x/y/edge_index/global_cond`，并对 `y_pred/y_wss_pred` 执行最大绝对误差与 `torch.allclose(rtol=1e-5, atol=1e-6)`
