# WSS 最小化路线 · 训练实验跟踪

> 用途：跟踪 `training_wss_min/`（PointNeXt 残差 baseline，独立于 V3P `training/`）的逐轮实验：
> 设计、完整指标表、结论、待办。**每完成一轮/一次任务，回填本文档。**
> 上位：[WSS最小化_代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md) / [training_wss_min/README](../../training_wss_min/README.md)。

## 第四轮补充｜点数—精度曲线（§14，2026-07-10 ✅单 seed + 多 seed）

- **单 seed 曲线**：见下表；峰值曾在 1000，但属单点。
- **多 seed 确认（1000 vs 2000 × {1234,7,2025}）**：

| n | R²_field | R²_casemean | top10 | IoU | score |
|---:|---|---|---|---|---|
| 1000 | 0.337±0.017 | **0.221±0.019** | 0.413±0.032 | 0.327±0.011 | 0.276±0.025 |
| 2000 | 0.339±0.020 | 0.188±0.025 | 0.416±0.025 | 0.324±0.016 | 0.259±0.026 |

- **裁决**：仅 R²_casemean 三 seed 一致偏向 1000；R²_field/top10/IoU/score 持平或不一致 → **默认锚点仍为 2000**；1000 为更稀采样候选。Jobs 6976–6979 + 既有 6969/6960。
- **产物**：`runs/_summary_round4_pointcount/`（含 `pointcount_multiseed_*`）。
- 归档计划：[第四轮执行总结与归档 §14.5](../02-推进与变更/_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)。

### 单 seed 6 点曲线（历史，seed1234）

| n | R²_field | R²_casemean | top10 | IoU | score | 来源 |
|---:|---:|---:|---:|---:|---:|---|
| 1000 | **0.351** | **0.241** | **0.448** | 0.330 | **0.301** | Job 6969 |
| 1500 | 0.327 | 0.208 | 0.408 | 0.296 | 0.262 | 6970 |
| 2000 | **0.351** | 0.214 | 0.436 | 0.330 | 0.283 | B1 6960 |
| 3000 | 0.298 | 0.150 | 0.365 | 0.315 | 0.205 | 6971 |
| 4000 | 0.259 | 0.207 | 0.346 | 0.296 | 0.227 | B3 6962 |
| 6000 | 0.335 | 0.207 | 0.412 | **0.343** | 0.268 | 6972 |

## 第四轮状态（Stage A→B→C 单 seed 已执行，2026-07-10）

- **协议**：v2_dev1 开发划分 + fold stats；val-only；`persistent_workers=False`；固定 train 分位权重；复合选模 + early stop（160 / patience=6）。
- **B 组**：B0/B1 持平（B1 为协议胜者）；B2 multi-start、B3 FPS4000 均 Gate-1 No-Go（B3 `R²_field` 大跌；§14 后改为补全曲线而非永久不开 6000）。
- **C 组**：C1 raw-Huber、C4 coord_scale No-Go；C2/C3/C5 按条件跳过（A3 smearing 失败；C5 Ridge 增量≈0）。
- **A5 补齐**：raw top10 差 + 邻域 cap（全量 cap≈0.97，子采样≈0.37）。
- **覆盖**：旧 v1 审计 fixed2000 high-WSS hit ~10%、multi-start 40ep union ~34%；dev1 对齐审计见 Job 6968。
- **当前锚点**：`r4_dev1_b1_tgtw_fixedq_s1234`（val R²_c=0.214 / R²_f=0.351 / top10=0.436）。未扩 3 seed、未跑 legacy test。
- 归档计划：[WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md](../02-推进与变更/_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)。

## 第四轮状态（计划 v2.3 终审通过，历史快照）

- 开发阶段仍须 val-only；`test16` 仅保留为最后一次的 legacy benchmark。
- A5 只读诊断已完成**三 seed 标准化口径**（终审独立复现，seed1234 与原引用逐位一致）：8 个 val 病例、同一 FPS-2000 点上比较子采样/全量推理同索引输出，MAE/RMSE/Pearson/mean-shift：s1234 `0.1645/0.2278/0.9381/-0.050`、s7 `0.1413/0.1884/0.9605/-0.033`、s2025 `0.1191/0.1638/0.9717/-0.066`。三 seed 方向一致：全量推理系统性偏高 `0.03–0.07σ`，且点数最多的病例漂移最大。证据：`docs/02-推进与变更/assets_第四轮/a5_density_probe.{py,csv}`。这证明推理密度敏感，尚不能推出全量评估不正确或采用下采样插值作为修复。
- 下一步固定为 Stage A：split/bundle 完整性、worker-safe sampler、固定 train-only loss 阈值、A5 剩余项（同索引 raw top10 差 + 邻域 cap 审计）、残差校准、repeated-holdout；不直接启动 NLL、法向、rot_aug 或 6000 点训练。
- 现行状态入口：本跟踪文档与 [WSS最小化代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md)；第四轮计划已归档为[执行总结](../02-推进与变更/_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)。

## 任务定义与通用设置
- **任务**：几何点云 `(x,y,z[+几何]) → 壁面 WSS 标量`，全局 `log_z` 归一化，单头。矢量三分量为二期。
- **路线 A（部署导向）**：部署有完整几何、缺 CFD 标签 → 训练用稀疏子采样，**评估恒在完整壁面点云上**。
- **模型**：PointNeXt-S 残差版（InvResMLP + ball-query，密度鲁棒）。约 4.4M 参数。
- **当前数据口径**：split `split_AG_wss_min_v1`（train 53 / val 8 / test 16 / excluded 9 / pending 1）；坐标逐病例 [-1,1]；WSS 为 train-only、peak-only 全局 `log_z`（std=1.0907）。第一/二轮旧 stats 仅作历史对照。
- **第三轮训练**：AdamW + cosine（warmup 10）+ AMP，240 epoch，batch 8 病例，eval_every 10；best 按 `val_r2_casemean`。第一/二轮 400 epoch 设置见各轮记录。
- **集群**：GPU 分区 `master`（4×RTX4090），`submit_baseline_sweep.sh` 提交，自动排队 4 并行；第三轮每 config 实测约 3–4 min 训练，随后做完整点云评估。
- **产物**：`training_wss_min/runs/<name>/`（`train.log`、`history.jsonl`、`ckpt_best/last.pt`、`config.json`、`feature_stats.json`、`eval/metrics.json`、`eval/per_case_metrics.csv`、`eval/heatmaps/`）。
- **汇总**：`python -m training_wss_min.summarize` → `runs/_summary/{summary.csv,pointcount_curve.png,regional_bar.png}`（聚合全部轮次）。

## 指标口径
- 均在**原始 WSS 空间**（denormalize 后）计算。
- `R²_field`：所有点 pool 起来算（受高 WSS 病例主导）；`R²_casemean`：逐病例算再平均（跨病例更平衡）。
- 分区：`bifurcation`(距原点≤0.25) / `stenosis`(local_radius 最小 20%) / `high_wss`(原始 WSS 前 10%)。

---

## 第三轮 clean-data ✅完训并完成判读（2026-07-10，Slurm 6952–6958）

**目的**：移除污染病例、重算 peak-only 统计后，以相同 `xyz+geom / FPS 2000 / PointNeXt-S` 配方比较 MSE 与 target-weight（α=2），各跑 3 个 seed，判断数据清理和尾部加权是否真正改善完整点云预测。

**数据侧已完成**：
- split 已更新为 train 53 / val 8 / test 16 / excluded 9 / pending 1；`slow/ZHANG_HUAN_LI` 已移入 excluded，`slow/ZHAO_XIU_XUAN` 保持 pending，excluded/pending 不参与正式数据集。
- scope guard 已加硬：`--all-raw` 仅允许诊断性 `preprocess`，正式 `qa-gate/global-stats/build-samples/all` 均按 split，pending/excluded 历史 bundle 不会进入 stats、训练或评估口径。
- `preprocess` 已重跑 77 个 included bundle：Slurm Job **6952**，`ok=77 / skipped=0 / error=0`；`nodenumber/cellnumber` 对齐守卫落盘，`nodenumber_reordered_cases=0`，`wall_coord_mismatch_cases=0`。
- QA gate：fatal=0，warning=3（`fast/ZHANG_QING_WANG`、`slow/GUAN_TONG_XIANG`、`fast/RAN_QING_BO` 仅 `trunk_centering_offset_frac>0.05`，按最终诊断 P2 作为复核项，不默认剔除）。
- 新 stats：train peak-only，53 cases / 711412 wall points，zero_frac=0，log mean/std=`1.1595 / 1.0907`，raw p90/p99/max=`12.8847 / 32.6039 / 220.7968`。

**模板基线（test，完整点云）**：

| 配置 | R²_field | R²_casemean | high_wss R² | top10 pred/true | p99 比 |
|---|---:|---:|---:|---:|---:|
| template_mean_clean | -0.114 | -0.139 | -2.533 | 0.172 | 0.118 |
| template_voxel_clean | 0.035 | -0.023 | -2.058 | 0.260 | 0.392 |
| template_knn_clean | -0.039 | -0.225 | -1.994 | 0.289 | 0.576 |

**单 run 结果（test，完整点云；best 仍由 val `R²_casemean` 选择）**：

| Job / 配置 | R²_field | R²_casemean | bif | stenosis | high_wss | top10 比 | p99 比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6953 / `mse_s1234` | 0.210 | 0.200 | 0.040 | -0.186 | -1.591 | 0.359 | 0.437 |
| 6954 / `mse_s7` | 0.187 | 0.163 | 0.012 | -0.168 | -1.669 | 0.332 | 0.425 |
| 6955 / `mse_s2025` | 0.175 | 0.165 | -0.003 | -0.255 | -1.673 | 0.319 | 0.421 |
| 6956 / `tgtw_s1234` | **0.251** | 0.220 | **0.097** | -0.129 | **-1.366** | **0.393** | **0.486** |
| 6957 / `tgtw_s7` | 0.186 | 0.182 | 0.002 | -0.273 | -1.699 | 0.320 | 0.410 |
| 6958 / `tgtw_s2025` | 0.239 | **0.234** | 0.064 | **-0.097** | -1.528 | 0.357 | 0.406 |

**三 seed 汇总（mean ± sample std）**：

| loss | R²_field | R²_casemean | bif | stenosis | high_wss | top10 比 | p99 比 | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MSE | 0.191±0.018 | 0.176±0.021 | 0.016±0.022 | -0.203±0.046 | -1.644±0.046 | 0.337±0.020 | 0.428±0.008 | 2.847±0.038 |
| target-weight α=2 | **0.225±0.034** | **0.212±0.027** | **0.055±0.048** | **-0.166±0.094** | **-1.531±0.167** | **0.357±0.037** | **0.434±0.045** | **2.788±0.040** |

**结果判读**：

1. **target-weight 方向仍成立，但不是稳定获胜。** 三 seed 均值相对 MSE：`R²_field +0.035`、`R²_casemean +0.036`、MAE `-0.059`，区域指标也平均回升；但 seed 7 的 field 持平、stenosis/high-WSS 反而退化。当前只能下“2/3 seed 有效、均值正收益”的结论，不能把单 seed 最优 0.251 当作稳定水平。
2. **第三轮 clean-data 组合改善了幅值刻度，但没有突破整体 R² 平台。** 第二轮旧口径 tgtw 三 seed `R²_field=0.222±0.022`，第三轮为 `0.225±0.034`，几乎持平；但 top10 预测/真值均值约从 `23.4%` 提高到 `35.7%`，p99 比约从 `35.1%` 提高到 `43.4%`。这说明清理数据/peak stats 是必要修复，主要收益是减少峰值压扁，而非自动提高空间拟合上限。注意两轮还同时改变了 split、curvature transform、epoch/eval 设置，故该跨轮比较不是“只改 stats”的严格因果消融。
3. **高 WSS 仍是明确 No-Go 项。** 6 个 run 的 high-WSS R² 全为负，tgtw 均值仍为 `-1.531`；top10 真值只恢复约 36%，max 比均值仅约 13%。典型热力图显示模型大致知道热点在分叉/髂支附近，但输出过度平滑、峰值幅值系统性偏低。
4. **checkpoint 选择噪声仍大。** tgtw 三个 best epoch 为 39/19/19，MSE 为 149/59/99；240 epoch 对 tgtw 明显冗余。val 只有 8 例，单一 `val_r2_casemean` 会偏向很早的偶然峰值，也无法约束 high-WSS 校准。
5. **病例级失败不是单个坏例造成。** tgtw 三 seed 平均最差病例为 `WANG_DENG_FENG`、`LU_ZHEN_QING`、`LV_FU_LONG`、`QIN_SI_FU`；fast/slow 两组病例均值接近，没有证据把问题归因于某一个 cohort。QA warning 3 例都在训练集，也不能解释 test 的系统性尾部低估。

**汇总产物**：已重跑 `python -m training_wss_min.summarize`，`training_wss_min/runs/_summary/` 现聚合 27 个实验（含第三轮 6 个深度模型与 3 个 clean 模板基线）。

**下一轮优化优先级**（详见已归档的 [第四轮执行总结与归档 v2](../02-推进与变更/_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)）：

- **P0｜协议与正确性**：修复 persistent worker 下 epoch 采样状态、train-global 固定 target-weight 阈值、val-only 开发评估、top-k checkpoint 与 early stopping；不采用跨 checkpoint 指标滑窗直接选当前权重。
- **P1｜采样覆盖**：先做 train-only 覆盖审计，再依次比较 fixed FPS 2000、multi-start FPS 2000、fixed FPS 4000；4000 Go 后才开 6000。
- **P2｜尾部目标假设**：先做 3 seed 残差分位与病例留一校准 Gate-0；raw-space Huber 辅助为主 probe，高斯 NLL 仅在 Gate-0 支持时探索，expectile/quantile 不进入默认均值回归矩阵。
- **P3｜内在局部几何**：先 `radius_gradient`，再 radial/surface rotation-invariant 特征；不直接输入未经符号/局部 frame QA 的全局法向。
- **P4｜统计固化**：train+val 61 例做版本化 repeated holdout 与 fold-specific stats；legacy test16 停止逐配置查看，最终锁定后只运行一次，并做病例级 bootstrap。

## 第一轮 sweep ✅完成（2026-07-08，Slurm 5945–5953）
**目的**：三条正交问题——点数-精度曲线、特征消融、采样策略。均标量 WSS + 峰值收缩期 + 400 epoch。

**结果（test，完整点云，按 R²_field 降序）**：

| 配置 | 点数 | 采样 | 特征 | R²_field | R²_casemean | bif | stenosis | high_wss |
|---|---|---|---|---|---|---|---|---|
| feat_xyzgeom | 2000 | fps | xyz+几何 | **0.200** | 0.188 | 0.017 | −0.196 | −1.634 |
| feat_geomonly | 2000 | fps | 纯几何 | 0.191 | 0.189 | 0.014 | −0.263 | −1.712 |
| pc_xyz_w6000 | 6000 | fps | xyz | 0.141 | 0.119 | −0.074 | −0.277 | −1.805 |
| pc_xyz_w3000 | 3000 | fps | xyz | 0.118 | 0.108 | −0.072 | −0.345 | −1.842 |
| samp_geomw | 2000 | 几何加权 | xyz | 0.108 | 0.100 | −0.110 | −0.349 | −1.934 |
| samp_random | 2000 | random | xyz | 0.084 | 0.062 | −0.139 | −0.326 | −1.956 |
| pc_xyz_w1000 | 1000 | fps | xyz | 0.058 | 0.070 | −0.151 | −0.474 | −2.037 |
| pc_xyz_w2000 | 2000 | fps | xyz | 0.053 | 0.047 | −0.160 | −0.482 | −2.062 |
| pc_xyz_w1500 | 1500 | fps | xyz | −0.072 | −0.051 | −0.315 | −0.749 | −2.446 |

**结论**：
1. **几何特征是压倒性杠杆**：xyz+几何(0.200) ≈ 纯几何(0.191) ≫ 纯 xyz@2000(0.053)，约 4×。纯几何≈xyz+几何 → 绝对坐标几乎不贡献，**标量任务里那套 flow-divider 配准/左右轴框架不值分**（模型学局部几何而非绝对位置）。
2. **几何特征更省点**：2000 点几何 > 6000 点纯 xyz；纯 xyz 才靠堆点数往上爬。`w1500` 负值是单种子方差异常。
3. **几何加权采样有用**：0.108 > random 0.084 > fps 0.053（xyz@2000）。
4. **主瓶颈**：所有配置在 stenosis / high_wss 区 R² 全负（最好也 −0.20 / −1.63），val/test gap 大（w1000 val 0.29 vs test 0.06，小样本过拟合）。

---

## 第二轮 sweep ✅完成（2026-07-08 提交，Slurm 5961–5969；2026-07-09 收官汇总）
**目的**：以第一轮最优方向为默认（**xyz+几何 / fps2000**），专打尾部（狭窄/高 WSS）崩溃，并稳方差、缩小 gap。

**新增代码能力（配置驱动，向后兼容）**：
- 训练期**随机 3D 旋转增广**（`DataConfig.rot_aug`）：只扰动 xyz 输入列（几何/标量标签旋转不变、ball-query 图结构不变），逼模型别背绝对朝向。
- **目标幅值加权 loss**（`TrainConfig.loss_weight_target`）：`weight += α·clamp01(y_norm)`，用训练标签给高 WSS 点更大权重（非泄漏），直接补 high_wss 欠拟合。
- Huber loss 选项、几何加权 loss（`loss_geom_weight`）。

**配置矩阵（9）**：`mse`(参考) / `huber` / `geomwloss`(几何加权 loss) / `tgtwloss`(目标加权 loss) / `geomw_tgtw`(双加权) / `geomwsamp_tgtw`(几何加权采样+目标加权) / `rotaug`(旋转增广) / `tgtw_s2025` `tgtw_s7`(目标加权多种子)。

**最终结果（test，完整点云，按 R²_field 降序；best epoch 为 `val_r2_casemean` 选中的 ckpt）**：

| 配置 | R²_field | R²_casemean | bif | stenosis | high_wss | best epoch |
|---|---|---|---|---|---|---|
| r2_xyzgeom_geomw_tgtw | **0.249** | 0.222 | **0.086** | −0.111 | −1.401 | 119 |
| r2_xyzgeom_tgtwloss (s1234) | 0.246 | **0.233** | 0.074 | **−0.068** | **−1.377** | 79 |
| r2_xyzgeom_geomwsamp_tgtw | 0.235 | 0.226 | 0.064 | −0.128 | −1.524 | 59 |
| r2_xyzgeom_geomwloss | 0.229 | 0.196 | 0.049 | −0.154 | −1.398 | 279 |
| r2_xyzgeom_tgtw_s7 | 0.217 | 0.220 | 0.040 | −0.139 | −1.587 | 99 |
| r2_xyzgeom_tgtw_s2025 | 0.203 | 0.165 | 0.048 | −0.209 | −1.460 | 59 |
| r2_xyzgeom_huber | 0.203 | 0.180 | 0.017 | −0.169 | −1.574 | 339 |
| r2_xyzgeom_mse (参考) | 0.189 | 0.156 | −0.005 | −0.191 | −1.634 | 159 |
| r2_xyzgeom_rotaug | 0.186 | 0.173 | −0.015 | −0.094 | −1.577 | 59 |

**高 WSS 分位校准探针（test 全场 pool，ckpt_best 完整推理；真值 top10% 均值 18.51，p99 26.98，max 126.74）**：

| 配置 | top10% pred/true | p99 比 | max 比 |
|---|---|---|---|
| r2_xyzgeom_mse | 3.90/18.51 = **21.1%** | 34.4% | 33.5% |
| r2_xyzgeom_tgtwloss | 4.10/18.51 = 22.1% | 34.3% | 31.1% |
| r2_xyzgeom_tgtw_s2025 / _s7 | 22.0% / 26.0% | 31.2% / 39.9% | 41.9% / 34.4% |
| r2_xyzgeom_geomw_tgtw | 4.86/18.51 = 26.2% | 40.4% | **138%**（max 溢出） |
| r2_xyzgeom_geomwsamp_tgtw | 9.76/18.51 = **52.7%** | 78.5% | **2248%**（max 2849，爆炸） |

**结论**：
1. **目标幅值加权方向确认有效，且是唯一稳定正收益**：tgtw 家族（tgtwloss/geomw_tgtw/geomwsamp_tgtw）R²_field 0.235–0.249 全面领先 mse 参考(0.189)/huber(0.203)/rotaug(0.186)，stenosis 从 −0.19 回拉到 −0.07~−0.13、high_wss 从 −1.63 回拉到 −1.38~−1.52。
2. **但分位校准揭示：R² 回拉主要来自中段，真峰值几乎没恢复**。top10% 高 WSS 校准 mse 21.1% → tgtwloss 仅 22.1%（p99/max 比甚至持平或更低）；`geomwsamp_tgtw` 可推到 52.7% 但 max 溢出 22 倍，不可用。**印证最终诊断的 P1 判断：旧 all-time stats（log std≈5.58）把峰值压扁是硬上限，loss 加权修不了，必须 clean-data 重算 stats。**
3. **seed 方差大到吞掉多数配置间差异**：tgtwloss 三种子 R²_field = 0.246/0.217/0.203（均值 0.222，极差 0.043）。除"tgtw 家族 > 非加权"这一档间隔外，家族内部排序（如 geomw_tgtw 0.249 vs tgtwloss 0.246）不可作数；第三轮必须多 seed。
4. **Huber 无收益**（0.203，在 seed 噪声内）；**rotaug 全场最差**（0.186）且与 canonical flow-divider 坐标框架假设冲突——第三轮 `xyz+geom` 主线不启用（与最终诊断 6.2 一致）。
5. **400 epoch 明显过长**：best epoch 集中在 59–159（仅 huber 339 / geomwloss 279 例外），后期都是过拟合区；val(8例)→test 落差 R²_field 约 0.05–0.11。支持最终诊断 5.1/5.2：缩短训练/early stopping + 复合选模指标。

**产物**：`runs/_summary/{summary.csv,pointcount_curve.png,regional_bar.png}`（已含全部 18 run）；分位探针脚本口径见最终诊断文档 §9 交付物要求（top10%/p95/p99/max 比值）。

---

## 待办 / 下一轮候选
- [x] **第三轮 clean-data 重启已提交**：`ZHANG_HUAN_LI` 移 excluded、QA gate、`nodenumber/cellnumber` 对齐守卫、clean peak stats、模板基线、`mse/tgtw` 各 3 seed 已完成或提交。
- [x] eval 固化高 WSS 校准指标（top10% mean 比、p95/p99/max 比、分位校准斜率），并入 `metrics.json` 与 summarize。
- [x] 第三轮 Job 6953–6958 完训并完成三 seed、区域指标、high-WSS 校准和典型热力图判读。
- [ ] **P0**：复合/平滑选模 + early stopping，固定在 val 上决策，不用 test 反选 checkpoint。
- [ ] **P1**：`coord_scale` 与 peak inlet-flow/可部署边界条件标量的单变量信息上限探针。
- [ ] **P2**：α=1/2/4 + 小权重 raw-space/分位辅助 loss；暂不组合激进几何加权采样。
- [ ] **P3**：重复 split / 5-fold 或病例 bootstrap CI，确认 0.02–0.04 级差异是否超出抽样噪声。
- [ ] 二期：WSS 矢量三分量（幅值复用标量 + 内在系方向，避开全局配准符号一致性问题）。
- [ ] 全相位（TAWSS/OSI 衍生量）与近壁速度第二条路径。
