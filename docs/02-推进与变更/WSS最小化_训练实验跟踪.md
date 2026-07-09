# WSS 最小化路线 · 训练实验跟踪

> 用途：跟踪 `training_wss_min/`（PointNeXt 残差 baseline，独立于 V3P `training/`）的逐轮实验：
> 设计、完整指标表、结论、待办。**每完成一轮/一次任务，回填本文档。**
> 上位：[WSS最小化_代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md) / [training_wss_min/README](../../training_wss_min/README.md)。

## 任务定义与通用设置
- **任务**：几何点云 `(x,y,z[+几何]) → 壁面 WSS 标量`，全局 `log_z` 归一化，单头。矢量三分量为二期。
- **路线 A（部署导向）**：部署有完整几何、缺 CFD 标签 → 训练用稀疏子采样，**评估恒在完整壁面点云上**。
- **模型**：PointNeXt-S 残差版（InvResMLP + ball-query，密度鲁棒）。约 4.4M 参数。
- **数据**：split `split_AG_wss_min_v1`（train 54 / val 8 / test 16）；坐标逐病例 [-1,1]；WSS 全局 log_z（train-only，std≈5.6）。
- **训练**：AdamW + cosine（warmup 10）+ AMP，400 epoch，batch 8 病例，eval_every 20；best 按 `val_r2_casemean`。
- **集群**：GPU 分区 `master`（4×RTX4090），`submit_baseline_sweep.sh` 提交，自动排队 4 并行；每 config 约 10–15 min。
- **产物**：`training_wss_min/runs/<name>/`（`train.log`、`history.jsonl`、`ckpt_best/last.pt`、`config.json`、`feature_stats.json`、`eval/metrics.json`、`eval/per_case_metrics.csv`、`eval/heatmaps/`）。
- **汇总**：`python -m training_wss_min.summarize` → `runs/_summary/{summary.csv,pointcount_curve.png,regional_bar.png}`（聚合全部轮次）。

## 指标口径
- 均在**原始 WSS 空间**（denormalize 后）计算。
- `R²_field`：所有点 pool 起来算（受高 WSS 病例主导）；`R²_casemean`：逐病例算再平均（跨病例更平衡）。
- 分区：`bifurcation`(距原点≤0.25) / `stenosis`(local_radius 最小 20%) / `high_wss`(原始 WSS 前 10%)。

---

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

## 第二轮 sweep 🚧进行中（2026-07-08，Slurm 5961–5969）
**目的**：以第一轮最优方向为默认（**xyz+几何 / fps2000**），专打尾部（狭窄/高 WSS）崩溃，并稳方差、缩小 gap。

**新增代码能力（配置驱动，向后兼容）**：
- 训练期**随机 3D 旋转增广**（`DataConfig.rot_aug`）：只扰动 xyz 输入列（几何/标量标签旋转不变、ball-query 图结构不变），逼模型别背绝对朝向。
- **目标幅值加权 loss**（`TrainConfig.loss_weight_target`）：`weight += α·clamp01(y_norm)`，用训练标签给高 WSS 点更大权重（非泄漏），直接补 high_wss 欠拟合。
- Huber loss 选项、几何加权 loss（`loss_geom_weight`）。

**配置矩阵（9）**：`mse`(参考) / `huber` / `geomwloss`(几何加权 loss) / `tgtwloss`(目标加权 loss) / `geomw_tgtw`(双加权) / `geomwsamp_tgtw`(几何加权采样+目标加权) / `rotaug`(旋转增广) / `tgtw_s2025` `tgtw_s7`(目标加权多种子)。

**已完成部分结果（test，完整点云）**：

| 配置 | R²_field | bif | stenosis | high_wss | 状态 |
|---|---|---|---|---|---|
| r2_xyzgeom_tgtwloss | **0.246** | 0.074 | **−0.068** | **−1.377** | ✅ |
| r2_xyzgeom_geomwloss | 0.229 | 0.049 | −0.154 | −1.398 | ✅ |
| r2_xyzgeom_huber | 0.203 | 0.017 | −0.169 | −1.574 | ✅ |
| r2_xyzgeom_mse | 0.189 | −0.005 | −0.191 | −1.634 | ✅ (参考) |
| r2_xyzgeom_geomw_tgtw | — | — | — | — | 🚧 |
| r2_xyzgeom_geomwsamp_tgtw | — | — | — | — | 🚧 |
| r2_xyzgeom_rotaug | — | — | — | — | 🚧 |
| r2_xyzgeom_tgtw_s2025 / _s7 | — | — | — | — | 🚧 |

**早期结论（待全部跑完后补全）**：
- **目标加权 loss 是当前最优**：R²_field 0.246（超第一轮 0.200），并把 stenosis −0.19→−0.07、high_wss −1.63→−1.38 明显回拉 → **尾部攻击方向验证有效**。
- 相同配置 `r2_xyzgeom_mse`(0.189) vs 第一轮 `feat_xyzgeom`(0.200) 差 0.011，印证**单种子方差**问题（多种子结果待收）。

**待回填**：`geomw_tgtw` / `geomwsamp_tgtw` / `rotaug` / 多种子跑完后更新上表、补 `summarize` 图、定第三轮方向。

---

## 待办 / 下一轮候选
- [ ] 回填第二轮剩余 5 例结果，定最优组合。
- [ ] 若目标加权仍不足以救 high_wss（仍负）：加大 α 扫描、或改 robust target（分位裁剪/分段标准化）。
- [ ] 多种子确认方差量级，必要时加 EMA / 正则。
- [ ] 二期：WSS 矢量三分量（幅值复用标量 + 内在系方向，避开全局配准符号一致性问题）。
- [ ] 全相位（TAWSS/OSI 衍生量）与近壁速度第二条路径。
