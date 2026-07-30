# WSS 最小化路线 · 训练实验跟踪

> 用途：跟踪 `training_wss_min/`（PointNeXt 残差 baseline，独立于 V3P `training/`）的逐轮实验：
> 设计、完整指标表、结论、待办。**每完成一轮/一次任务，回填本文档。**
> 上位：[WSS最小化_代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md) / [training_wss_min/README](../../training_wss_min/README.md)。

## 2026-07-29｜SAME-H2 并发复现 / +log(local_radius)（`11032→11033_[0-1]`；✅2/2 完成，处理臂 Go）

按用户决议暂不做多 seed，直接锁定 `lsa2_same_h2_pinball_q90_lam020_s1234` 为后续父模型。本轮不是重复旧 MSE/H2 矩阵，而是用两个 seed1234 臂隔离输入表达：

| Array | ID | 相对锁定 H2 的变化 | 作用 |
|---:|---|---|---|
| 0 | `lsa2_h2_same_repro_s1234` | 仅 `name/notes` | 精确同-seed并发 control，控制 CUDA 训练轨迹漂移 |
| 1 | `lsa2_h2_logradius_s1234` | 输入末尾新增 train-only 标准化 `log_local_radius`；特征统计路径相应更新 | 检验对数半径是否补充原始 `local_radius` |

两臂均冻结 REG-P10-LSA2、SAME、H2 q90 pinball λ0.20、DropPath0.10、SA2 4-head Local Transformer、LocalGeoPE 原索引 `[3,4,5]`、mixed `138/0/36`、random5000、400 epoch、train-loss 选模和 `legacy_vertex`。处理臂保留原始 `local_radius`，不把 log 半径送入 LocalGeoPE，只把它作为第7列输入，因此不混入特征替换或架构变化。新列使用与父模型一致的 control106 train-only 来源统计：106例、3,208,800点，natural-log 均值 `2.116623`、标准差 `0.611423`。

预注册主门：相对并发 control，`Δphysical R²_cb≥+0.012` 或 `Δhigh-WSS nRMSE≤-0.002`；保护线为 `Δnormalized R²_cb≥-0.01`、`Δphysical MAE_cb≤+0.05 Pa`、`Δtop10 IoU≥-0.005`。历史 ★ H2 只作参考，不作正式 Gate 对照。Jobs `11032`、`11033_0`、`11033_1` 均 `COMPLETED (0:0)`；两臂均完成 400 epoch、best/last checkpoint、test36 全云评估和 36 例明细，配置哈希、有限值、CUDA checkpoint/full-chunk 与病例集合完整性通过。

主结果固定取 `ckpt_best(train_loss)`：

| 臂 | best epoch | R²_cb | ΔR² | normalized R²_cb | Δnormalized | MAE (Pa) | ΔMAE | high-WSS nRMSE | ΔnRMSE | top10 IoU | ΔIoU | AG / AAA / ILO R² | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| H2 concurrent control | 378 | 0.3070 | — | 0.6397 | — | 2.5580 | — | 0.06906 | — | **0.1916** | — | 0.325 / 0.248 / 0.313 | 并发基准 |
| **H2 + log(radius)** | 378 | **0.3506** | **+0.0436** | **0.6440** | **+0.0042** | **2.5238** | **-0.0342** | **0.06611** | **-0.00295** | 0.1887 | -0.0029 | **0.363 / 0.271 / 0.377** | **Go** |

处理臂同时通过两个主门，且 normalized R²、MAE、IoU 三条保护线全部通过；三域 R² 分别提高 `+0.0385/+0.0233/+0.0640`。相对历史 ★ H2，处理臂仍有 `ΔR²_cb=+0.0267`、`ΔMAE=-0.0133 Pa`、`Δhigh-WSS nRMSE=-0.00224`，但正式裁决只使用并发 control。

病例层面没有同样强的全域一致性：逐病例 R² 均值差 `-0.0020`，bootstrap 95%CI `[-0.0326,+0.0281]`，16/20 改善/退化；high-WSS R² 为 23/13，均值差 95%CI `[-0.0374,+0.4000]`。这不推翻已注册的聚合 Gate，但说明当前只可晋级为**单 seed 开发锚点**，不能写成已确认的跨病例稳定增益。

**终裁**：`lsa2_h2_logradius_s1234` 晋级为新的正式开发锚点；保留原始 `local_radius`、SAME-H2、L-SA2 和 LocalGeoPE，后续新增实验均以该配置为父配置。按用户既定决议暂不做多 seed，也不回到 RCR 架构、面积口径、IND 或 H1+H2。结构化真源为 `training_wss_min/preflight/lsa2_h2_logradius_matrix_20260729_results_{analysis.json,summary.csv}` 与同前缀 `paired_case_stats.csv`。

## 2026-07-29｜REG-P10-LSA2 SAME/IND × MSE/H1/H2（`11019→11020_[0-5]%3`；✅6/6 完成）

六臂均完成 400 epoch、best/last checkpoint、test36 全云评估和 36 例明细；Jobs `11019`、`11020_0–5` 均 `COMPLETED (0:0)`。配置哈希、CUDA smoke、有限值、best/last 和病例集合完整性全部通过。

主结果固定取 `ckpt_best(train_loss)`；Δ 只使用本轮并发控制：

| 臂 | 对照 | R²_cb | ΔR² | normalized R²_cb | Δnormalized | MAE (Pa) | ΔMAE | high-WSS nRMSE | ΔnRMSE | top10 IoU | ΔIoU | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SAME MSE | — | 0.2862 | — | 0.6390 | — | 2.5892 | — | 0.07040 | — | 0.1786 | — | 并发基准 |
| SAME H1 | SAME MSE | 0.3050 | +0.0188 | 0.6401 | +0.0011 | 2.5589 | -0.0303 | 0.06866 | -0.00174 | 0.1962 | +0.0175 | **No-Go** |
| SAME H2 | SAME MSE | **0.3239** | **+0.0376** | **0.6453** | **+0.0063** | **2.5371** | **-0.0521** | 0.06835 | **-0.00205** | 0.1800 | +0.0014 | **Go** |
| IND MSE | SAME MSE | 0.2876 | +0.0014 | 0.6367 | -0.0024 | 2.5825 | -0.0067 | 0.06991 | -0.00049 | 0.1921 | +0.0135 | **IND No-Go** |
| IND H1 | IND MSE | 0.3021 | +0.0145 | 0.6373 | +0.0007 | 2.5687 | -0.0138 | 0.06932 | -0.00059 | 0.1903 | -0.0018 | **No-Go** |
| IND H2 | IND MSE | 0.3198 | +0.0322 | 0.6390 | +0.0023 | 2.5709 | -0.0117 | **0.06784** | **-0.00207** | **0.2017** | +0.0096 | H2 Go；IND 不晋级 |

Gate 判读：

- H1：SAME/IND 的 ΔIoU 为 `+0.0175/-0.0018`，均未到 `+0.020`，关闭 H1，不组合；
- H2：SAME/IND 的 Δhigh-WSS nRMSE 为 `-0.002050/-0.002069`，两者均过 `-0.002`，且共同保护线通过；
- IND：MSE 主效应只有 `ΔR²_cb=+0.0014`，未到 `+0.012`。IND-H2 相对 SAME-H2 的 R² `-0.0041`、normalized R² `-0.0063`、MAE `+0.0337 Pa`，不晋级 IND；
- SAME-H2 三域 R² 均提高，病例 high-WSS R² 均值差 95%CI `[+0.0014,+0.5319]`；`ckpt_last` 仍有 `ΔR²_cb=+0.0383`、`Δhigh-WSS nRMSE=-0.00204`；
- SAME-H2 的主门只多出约 `0.00005`，病例全场 R² CI 仍跨零；因此它是临时工程锚点，不是确认性结论。

**终裁**：`lsa2_same_h2_pinball_q90_lam020_s1234` 作为父配置进入上方 `log(local_radius)` 两臂；后续处理臂已过门并替代其成为新开发锚点。结构化真源为 `training_wss_min/preflight/regp10_lsa2_objective_ind_matrix_20260728_results_{analysis.json,summary.csv}` 与同前缀 `paired_case_stats.csv`；工作簿已回填并保持 5 张表、6 页。

## 2026-07-28｜O0 热点/高 WSS 两臂（`11012→11013_[0-1]%2`；✅2/2 完成，2/2 No-Go）

按最新执行决策，不补 RCR 多 seed、不做 RCR 架构；以 O0 geometry-only S3 `PointNeXt-R + LocalGeoPE` 为唯一父模型。H1 增加病例相对 top10 balanced BCE 辅助输出，H2 增加 q90 pinball 辅助项；两臂均不使用 RCR，不改变 mixed `138/0/36`、seed1234、random5000/SAME、400 epoch、train-loss 选模或 `legacy_vertex`。

| 臂 | R²_cb | ΔR²_cb | normalized R²_cb | Δnormalized | MAE_cb (Pa) | ΔMAE | high-WSS nRMSE | ΔnRMSE | top10 IoU | ΔIoU | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| O0 geometry | 0.2756 | — | 0.6300 | — | 2.5974 | — | 0.07068 | — | 0.1772 | — | 基准 |
| H1 hotspot BCE | **0.2990** | **+0.0234** | **0.6336** | +0.0035 | **2.5843** | -0.0131 | **0.06935** | -0.00133 | **0.1943** | **+0.0170** | **No-Go：主门未达 +0.020** |
| H2 q90 pinball | 0.2947 | +0.0190 | 0.6274 | -0.0026 | 2.5917 | -0.0057 | 0.06996 | **-0.00072** | 0.1818 | +0.0046 | **No-Go：主门未达 -0.002** |

共同保护线：`Δnormalized R²_cb ≥ -0.01`、`Δphysical MAE_cb ≤ +0.05 Pa`。两臂保护线均通过，且 AG/AAA/ILO 的 physical R² 都相对 O0 提升；但 H1 的热点主门差 `0.0030`，H2 的尾部主门差 `0.00128`。病例配对的 ΔIoU 95%CI 也均跨零：H1 `[-0.0004,+0.0343]`（24/12），H2 `[-0.0103,+0.0190]`（20/16）。

Jobs `11012`、`11013_0`、`11013_1` 均 `COMPLETED (0:0)`，分别耗时 `00:03:32 / 00:22:12 / 00:21:57`；checkpoint、best/last test36 评估、36 例 CSV、配置哈希和有限值检查全部通过。`ckpt_last` 仍未过门，不改变结论。**终裁：2/2 No-Go，不组合、不补 H2 相邻 q/λ；当时提出的 O0 IND 单臂已被上方 REG-P10-LSA2 2×3 并发矩阵取代。**结构化真源为 `training_wss_min/preflight/o0_hotspot_tail_matrix_20260728_results_{analysis.json,summary.csv}` 与同前缀 `paired_case_stats.csv`。

## 2026-07-28｜S3 RCR Oracle O0/O1a/O2（`11008→11009_[0-2]%3`；✅3/3 完成）

以三种子通过的 S3 `D2-K64 + PointNeXt-R + LocalGeoPE` 为父模型，固定 mixed `138/0/36`、seed1234、400 epoch、train-loss 选模、random5000/SAME 与 `legacy_vertex`。O0 为 geometry-only；O1a 追加四出口真实 \(R_1/R_2/C\) 的 12 维自然对数病例条件；O2 追加在 train/test 内、按 AG/AAA subtype/ILO 0/1 分层无固定点打乱的同维 RCR，作为负对照。

| 臂 | R²_cb | ΔR²_cb vs O0 | MAE (Pa) | ΔMAE | high-WSS nRMSE | ΔnRMSE | top10 IoU | ΔIoU | p99 幅值比 | AG / AAA / ILO R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| O0 geometry | 0.2756 | — | 2.5974 | — | 0.07068 | — | 0.1772 | — | 0.3429 | 0.290 / 0.240 / 0.267 |
| O1a true RCR | **0.3561** | **+0.0805** | **2.4620** | **-0.1354** | **0.06728** | **-0.00340** | **0.2109** | **+0.0337** | **0.4356** | **0.400 / 0.268 / 0.357** |
| O2 shuffled RCR | 0.2723 | -0.0033 | 2.6053 | +0.0080 | 0.07148 | +0.00079 | 0.1840 | +0.0068 | 0.3240 | 0.332 / 0.233 / 0.221 |

配对复算结论：

- O1a − O0 的病例均值 ΔR²=`+0.0536`，bootstrap 95%CI `[+0.0127,+0.1021]`，22/36 病例改善；top10 IoU 的 95%CI `[+0.0102,+0.0550]`；
- O2 − O0 的病例均值 ΔR²=`-0.0134`，95%CI `[-0.0622,+0.0356]`，与零相容；
- O1a − O2 的 \(R^2_{cb}\) 为 `+0.0838`，病例均值 ΔR² 95%CI `[+0.0352,+0.1011]`；
- O1a 的 high-WSS 内 Spearman `0.0357`，相对 O0 `0.0359` 基本不变：真实 RCR 改善幅值/总体误差/热点定位，但没有解决热点内部排序。

**裁决：Go-to-follow-up。** 真实 RCR 同时显著优于几何基准和等维打乱负对照，确认四出口边界条件包含几何之外的有效 WSS 信息；下一步先补 O1a/O0 的多种子或新协议确认，并评估 RCR 可得性/可部署代理。由于仍是单种子、复用 test36、train-loss 选模，本结果不作为最终无偏泛化结论。O1b 流量占比因直接覆盖仅 58/174 且属于求解器输出，继续暂缓。

结果真源：`training_wss_min/preflight/rcr_oracle_matrix_20260728_results_{analysis.json,summary.csv}` 与 `rcr_oracle_matrix_20260728_paired_case_stats.csv`；配置在 `training_wss_min/configs/pointnetpp_rcr_oracle_20260728/`。

## 2026-07-28｜REG-P10 静态 EdgeConv SA1/SA2/SA12（`10993→10994_[0-2]%3`；✅3/3 完成）

以 REG-P10 `S3 + LocalGeoPE + DropPath0.10, seed1234` 为冻结父模型，只在既有 SA 几何分组内增加残差 EdgeConv 消息 `[x_i, x_j-x_i, Δp_ij/r]`。三臂分别启用 SA1、SA2、SA1+SA2；没有改变邻接关系，也没有执行特征空间动态 KNN。本组仍使用 mixed `138/0/36` 历史 test36、random5000/SAME、`125/125/32` centers、`64/16/16` 邻域、400 epoch 和 train-loss 选模。

| 处理臂 | R²_cb | ΔR²_cb | ΔMAE / ΔRMSE (Pa) | Δhigh-WSS / ΔIoU | ΔAG / ΔAAA / ΔILO | 病例均值 ΔR² 95%CI；胜/负 | 判定 |
|---|---:|---:|---:|---:|---:|---|---|
| REG-P10 | 0.2957 | — | 2.5858 / 6.3283 | -0.4127 / 0.1883 | 0.2954 / 0.2366 / 0.3195 | — | 父模型 |
| `EC-SA1` | 0.2904 | -0.0053 | -0.0001 / +0.0237 | -0.0210 / -0.0014 | -0.0095 / +0.0172 / -0.0167 | -0.0013 `[-0.0264,+0.0222]`；17/19 | No-Go |
| `EC-SA2` | 0.2935 | -0.0023 | -0.0240 / +0.0102 | -0.0256 / +0.0122 | +0.0143 / +0.0168 / -0.0324 | +0.0083 `[-0.0170,+0.0332]`；21/15 | No-Go |
| `EC-SA12` | 0.2619 | -0.0338 | +0.0207 / +0.1502 | -0.0925 / +0.0084 | -0.0108 / -0.0089 / -0.0753 | +0.0059 `[-0.0185,+0.0312]`；18/18 | No-Go |

三臂和父模型的配置哈希、400 epoch、best/last、全云指标、逐病例 CSV 与有限值完整性 `4/4` 通过；Job 均 `COMPLETED (0:0)`。`EC-SA2` 的 MAE、IoU、AG/AAA 及病例胜负存在局部正向信号，但未满足主 `R²_cb`、high-WSS 和 ILO 保护线，病例 CI 也跨零。终裁为**静态 EdgeConv 三臂全部 No-Go**，不补动态图、不补 seeds `7/2025`；这不能否定所有 DGCNN 变体，只说明当前“在 REG-P10 既有几何邻域内增加残差 EdgeConv”的最小实现没有净收益。

## 2026-07-26｜高 WSS 优化方案的第一性原理重估与对抗性审查（**无训练**，全库 192 例只读诊断）

不提交任何作业。对 [WSS高值区域预测优化方案](WSS高值区域预测优化方案.md) 做第一性原理重估与对抗性审查，结论回填该文档 §13–§17。诊断脚本 `training_wss_min/tools/diagnose_high_wss_ceilings.py`，产物 `training_wss_min/preflight/high_wss_ceilings_20260726.json`。

**为什么这轮不训练**：现有锚点 `regp10_localtf_sa2_s1234` 的落盘产物已足以确定"下一步该往哪投"，而按原方案 §9 直接开 A0–A4 + B0–B5 十一臂会在一个**门禁自相矛盾、口径混用、且处于过拟合区**的框架里烧掉整轮预算。

### 协议天花板（全库 192 例，真值在 5000 支撑点完美 → 模型实际的 3-NN 插值到全点云）

| 上界 | ALL | AG | AAA | ILO |
|---|---:|---:|---:|---:|
| 物理 R² | 0.939 | 0.969 | 0.914 | 0.894 |
| 标准化 R² | 0.976 | 0.984 | 0.972 | 0.946 |
| top10 幅值比 | 0.963 | 0.976 | 0.955 | 0.946 |
| **top10 IoU** | **0.813** | 0.858 | 0.784 | 0.738 |
| high-WSS Spearman | 0.847 | 0.901 | 0.808 | 0.774 |

当前模型 IoU 0.194 / top10 幅值比 0.371（pooled）——**分辨率不是瓶颈，还有 0.62 IoU 的空间不需要动分辨率**。各 cohort 天花板不同（网格 1.11 / 0.45 / 0.44 mm），跨域 IoU 直接比较不在同一标尺。

### 五条决定性量化结论

| # | 结论 | 关键数字 | 影响的原方案条目 |
|---|---|---|---|
| 1 | 物理尾部亏欠 = 标准化亏欠经 exp 放大，非独立故障 | \(\hat y_q/y_q=\exp[\sigma(\hat z_q-z_q)]\)，σ=1.3669，逐例最大偏差 3e-6 | §2.2 二分故障论 |
| 2 | 网络已是 log 空间条件均值，收缩是 MSE 的定义 | pooled slope 0.6325 vs R² 0.6181（差 0.014） | §5 全部重加权方案的上界 |
| 3 | log 方差 **85% 在病例内**、14.9% 在病例间 | between 0.2421 / within 1.3854 | §4 oracle、§7.4 尺度头、§7.5 分域容量**共享** ~15% 上限 |
| 4 | 标签不是瓶颈 | 8 近邻平滑后 IoU 0.901、幅值保留 0.976；峰值±1 步 IoU 0.921、幅值比 0.995 | §4.1 两条排查可结案为「否」 |
| 5 | 明确过拟合且在最深处选模 | train R²≈0.793 vs test 0.6425；插值只占 0.024 → 真实差距 ≥0.13；best epoch 378/400，`val_cases=0` | §9.1/§9.2 的臂序 |

### 对抗性审查（13 条，全部有落盘证据）

最关键四条：

- **D-03 门禁自相矛盾**：三种子配对 sd 为 R²_cb `0.0143–0.0227`、top10_IoU `0.0069–0.0119`、p99 `0.0105–0.0207`。§9.3 的"R² 不降 >0.01"**低于 1 sd**（中性模型单种子 24–33% 被误杀）；而"top10/p99 各 +0.08"是 9 臂全谱宽的 2.5 倍。**历史最佳的 S3−M1（R²_cb +0.0224 / IoU +0.0219 / p99 +0.0138 / MAE −0.128）自己过不了这道门禁。** 且能过幅值轴的过不了定位轴（单调重标定对 IoU/Spearman 逐点无效），8 条合取对任一单臂不可满足。
- **D-06 O1 定义含泄漏**：`Global_conditions/vf-out*` 是 Fluent report-definition 监视器 = **求解器输出**；RCR 参数在 `.cas.gz` 的四个 `pressure-outlet` 中，才是合法输入。原 §4 写作"真实出口 RCR/流量占比"把二者并列，喂流量占比会让 O1 **必然**假阳性并触发 §4.1 的转向判据。须拆成 O1a（RCR，合法不可部署）/ O1b（流量占比，显式标注泄漏探针）。
- **D-02 winner's curse**：§2.1 引用的 L-SA2 在 9 臂中 5 项第一、2 项第二（同数据同种子）。取 9 次最大的期望偏置约 1.49σ → R²_cb 高估约 **+0.016**、high-WSS Spearman 高估约 +0.013（臂均值 0.0288）。且比较发生在已复用多轮的 test36 上。
- **D-01 口径**：§2.1 一张表混了 pooled / casemean / case-balanced 三种聚合。pooled→逐例（均为物理空间）：top10 `0.371→0.428`、p99 `0.396→0.521`、max `0.125→0.265`、动态范围 `0.125→0.261`。另发现 `per_case_metrics.csv` 的**无前缀列是标准化空间**（`write_reports` 的 `norm_res` 赋值），与 `metrics.json` 同名物理区块并存；已核对分析器读 `metrics.json`，**已发布结果表未受污染**。

其余：D-04 尾部校准损失与 §2.2 矛盾且与被测指标同构；D-05 IoU/precision/recall 在当前实现下只有一个自由度（实测 precision=recall=0.31803376878722456）；D-09 §5.1「病例等权」在现采样器下已恒成立（no-op），cohort 平衡已被 S3 根因矩阵证否；D-10 曲率跨 7 个数量级（q02 0.00108 / q98 21495.6），§5.5 的曲率边权不可直接实现；D-11 面积映射 127/133 通过、**ILO 51 例从未审计**；D-12 §10.2 否决 NLL 的理由与本任务数学结构相反（log 空间下 \(\mathbb E[y|x]=\exp(\hat\mu+\hat\sigma^2/2)\)，预测方差正是去偏所需）。

### λ 前沿（零训练，已算出，唯二待实测的是物理 R²/MAE）

\(\hat z_\lambda=c+\lambda(\hat z-c)\)：\(R^2_\lambda=R^2\lambda(2-\lambda)\)，且 **IoU / 各 Spearman 逐点严格不变**。

| λ | 标准化 R²_cb | p99（逐例·物理） | top10（逐例·物理） | p99（pooled） | top10（pooled） |
|---:|---:|---:|---:|---:|---:|
| 1.000 现状 | 0.6425 | 0.520 | 0.428 | 0.396 | 0.371 |
| 1.150 | 0.6280 | 0.656 | 0.492 | 0.552 | 0.456 |
| 1.250 | 0.6023 | 0.769 | 0.541 | 0.688 | 0.524 |
| 1.400 | 0.5397 | 0.981 | 0.628 | 0.958 | 0.645 |

真值 top-10% 占物理 MSE 的 **70%（逐例）/ 85%（pooled）**，所以抬升尾部预计**提高**物理 R²——与 §9.3 用物理 R² 当幅值护栏的前提相反。λ 必须在 train/OOF 上拟合。

### 新增优化空间（原方案未覆盖）

`log(local_radius)` 输入列（单特征解释 **38.5%** 病例内 log 方差，实测指数 −1.328 而非 −3）；库内已有但零引用的 **81 心动时相 / `wall_wss_vec(81,N,3)` / `wall_pressure(81,N)`**；零代码的 `query_mode="independent"`（当前训练期因 `data_ptr()` 短路**从不执行** `knn_interpolate`，评估期总是执行，插值误差从未进入训练目标，约值 0.024 标准化 R²）；以及用 pinball/物理 Huber/NLL **换目标泛函**而非往 MSE 上加尾部项。

### 待办

修订后的优先级见方案 §16：**P0 七项全部不需要训练**（口径修复、λ 前沿、指标去冗余、grouped OOF + val 选模、按实测 sd 重写 Gate、面积口径定案）。P1 训练臂改为「正则/协议 → 输入信息 → 监督信息 → 目标泛函 → 才轮到 §5 的 hotspot 分类头」，每臂以 λ 前沿为对照。P2 oracle 降为 1 日诊断并预注册 ~15% 上限。P3 推迟 full-resolution refiner / cohort adapter / Transolver。**当前 No-Run，等用户裁定。**

## 2026-07-27｜固定 D2 PNXR+GeoPE 的 local-SA1 / local-SA2 对照（`10979→10980_[0-1]%2`；✅2/2 完成）

按追加对照要求，以已完成的 `d2_c125_k64_pnxr_geope` 为父模板，只新增 `local_transformer_stages=[1]` 与 `[2]` 两臂。与上一节 REG-P10 不同，本组固定 AG/AAA `106/0/27`、train106 stats、random5000/SAME、`125/125/32`、`64/16/16`、6D xyz+geom、7D LocalGeoPE、PointNeXt-R `(1,1,0)`，并保持 `drop_path_rate=0`；因此回答的是“SA1/SA2 局部 Transformer 能否叠加到原始 fixed-test27 D2 PNXR+GeoPE”，不是复跑 mixed test36 的 REG-P10。

两份配置位于 `training_wss_min/configs/pointnetpp_d2_c125_k64_pnxr_geope_transformer_20260726/`。静态严格差分 `2/2` 通过，除运行标识外唯一模型差异是 `model.local_transformer_stages`；门禁和两条训练任务均 `COMPLETED (0:0)`。两臂均有 400 行 history、best/last checkpoint、best/last test27 全云指标和 27 例逐病例 CSV，完整性审计通过。

| 处理臂 | 物理 R²_cb（Δ） | 归一化 R²_cb（Δ） | ΔMAE_cb / ΔRMSE_cb | Δhigh-WSS / ΔIoU | ΔAG / ΔAAA | 病例 ΔR² 95%CI | 判定 |
|---|---:|---:|---:|---:|---:|---|---|
| D2 PNXR+GeoPE | 0.2975 | 0.6712 | — | — | — | — | 父锚点 |
| `D2-L-SA1` | 0.2683（-0.0293） | 0.6491（-0.0222） | +0.0469 / +0.1298 Pa | -0.0753 / -0.0158 | -0.0217 / -0.0369 | -0.0184 `[-0.0466,+0.0090]` | No-Go |
| `D2-L-SA2` | 0.2990（+0.0015） | 0.6581（-0.0132） | -0.0160 / -0.0069 Pa | -0.0105 / +0.0073 | -0.0012 / +0.0042 | -0.0045 `[-0.0437,+0.0291]` | 持平，不晋级 |

主结论固定使用 `ckpt_best(train_loss)`：SA1 明确退化；SA2 的物理主 R² 仅近似持平，归一化 R²/high-WSS 回退且病例 CI 跨零，不支持继续扩展。SA2 last checkpoint 相对父模型 last 有 `+0.0198` 物理 R²_cb 敏感性，但不得按 test 事后改选 checkpoint。可复跑分析器为 `training_wss_min/tools/analyze_d2_c125_k64_transformer_results.py`；工作簿已更新「实验矩阵总览」「教师汇报视图」「汇总对比」。

## 2026-07-26｜REG-P10 局部 SA Transformer 完整矩阵 + SA3 全局对照（`10967→10968_[0-7]%4`；✅8/8 完成）

以 `s3reg_droppath010_s1234.json` 为本轮工作锚点 REG-P10。新增的局部 Transformer 不在 SA center 之间做全局 attention，而是在每个 center 自己的邻域内，把逐边 MLP 与 LocalGeoPE 融合后的消息当作 token，在 max 聚合前执行 pre-norm MHA+FFN；因此它直接检验“LocalGeoPE 已证明有效后，能否进一步提升邻域关系建模”。配置字段 `local_transformer_stages` 使用 1-based SA 编号，默认空列表，旧配置加载和旧 checkpoint key 不变。

SA3 全局分支不新增同义实现：既有 `CoarseGlobalBlock` 已经是 32 个 SA3 centers 上的病例内全局 Transformer（4-head self-attention + 相对 xyz/距离 bias + FFN + LayerScale），所以 `regp10_globaltf_sa3_s1234` 直接复用 `coarse_attention=true`。它与 2026-07-24 的旧 `S3+coarse_attention` 模块相同，但父配置从无 DropPath 的 S3 改成 REG-P10；因此新实验回答的是“全局块是否能叠加到 REG-P10”，不能把旧结果当作本轮结果。`local_transformer_stages=[3]` 则是 SA3 各自邻域内部、聚合前的局部 Transformer，和 coarse centers 间的全局块不同，两者均已列入矩阵用于机制区分。

完整矩阵包含局部 stage 的 7 个非空子集：`SA1`、`SA2`、`SA3`、`SA1+SA2`、`SA1+SA3`、`SA2+SA3`、`SA1+SA2+SA3`，再加 `SA3-global` 对照，共 8 臂。所有配置保持 mixed `138/0/36`、5000 random/SAME、`125/125/32`、`64/16/16`、PointNeXt-R `(1,1,0)`、7D LocalGeoPE、DropPath 0.10、seed1234、400 epoch 和 train-loss 选模不变。配置位于 `training_wss_min/configs/pointnetpp_regp10_transformer_20260726/`；本轮不同时打开 local 与 global，以保持机制可归因。

本地配置准备与静态审计 `8/8` 通过；默认关闭时旧 REG-P10 checkpoint 严格重载无缺失/多余 key。全测试集现为 `102/102` 通过：此前唯一失败不是模型问题，而是既有未跟踪的 SA-grouping 测试夹具把 `sa_radius/sa_nsample/sa_ratios` 配成三层，却遗漏对应的三层 `sa_blocks`，于是沿用默认四层并被配置一致性校验正确拒绝；已只在该测试夹具补上 `sa_blocks=[1,1,1]`，未放宽生产校验。正式 GPU 门禁 `10967` 与训练/评估数组 `10968_[0-7]%4` 均 `COMPLETED (0:0)`。8 个 run 均有 400 行 history、best/last checkpoint、best/last test36 全云指标和 36 例逐病例 CSV；配置哈希与全部数值完整性审计通过。

| 处理臂 | R²_cb | ΔR²_cb | ΔMAE / ΔRMSE (Pa) | Δhigh-WSS | ΔAG / ΔAAA / ΔILO | 病例 ΔR² 95%CI | 判定 |
|---|---:|---:|---:|---:|---:|---|---|
| REG-P10 | 0.2957 | — | — | — | — | — | 父锚点 |
| `L-SA1` | 0.2851 | -0.0106 | -0.0005 / +0.0474 | -0.0425 | +0.0008 / +0.0205 / -0.0440 | -0.0101 `[-0.0404,+0.0181]` | No-Go |
| **`L-SA2`** | **0.3171** | **+0.0214** | **-0.0332 / -0.0968** | **+0.0453** | **+0.0105 / +0.0198 / +0.0341** | +0.0114 `[-0.0101,+0.0324]` | **唯一过 Gate** |
| `L-SA3` | 0.2798 | -0.0159 | +0.0000 / +0.0712 | -0.0523 | +0.0014 / +0.0274 / -0.0641 | +0.0004 `[-0.0251,+0.0255]` | No-Go |
| `L-SA12` | 0.3021 | +0.0063 | -0.0101 / -0.0285 | -0.0088 | +0.0222 / +0.0017 / -0.0065 | +0.0023 `[-0.0290,+0.0322]` | 低于主 Gate |
| `L-SA13` | 0.2952 | -0.0005 | -0.0072 / +0.0024 | -0.0209 | +0.0017 / +0.0094 / -0.0096 | +0.0088 `[-0.0122,+0.0306]` | No-Go |
| `L-SA23` | 0.2909 | -0.0049 | +0.0139 / +0.0218 | -0.0082 | -0.0023 / +0.0206 / -0.0251 | -0.0208 `[-0.0664,+0.0162]` | No-Go |
| `L-SA123` | 0.2873 | -0.0084 | +0.0199 / +0.0376 | -0.0261 | +0.0015 / +0.0041 / -0.0273 | -0.0181 `[-0.0474,+0.0090]` | No-Go |
| `G-SA3` | 0.2909 | -0.0048 | +0.0068 / +0.0215 | -0.0246 | -0.0061 / +0.0184 / -0.0197 | -0.0026 `[-0.0316,+0.0249]` | No-Go |

`L-SA2` 是唯一满足 `ΔR²_cb≥+0.012`、MAE/high-WSS/分域护栏的处理臂，下一步只补它的 seeds `7/2025`，与同 seed REG-P10 做配对确认。病例 CI 跨零，因此当前仍是历史 test36 上的单种子强筛选信号。`L-SA12` 不到主门槛，不扩展；SA3 局部与既有 SA3 全局 attention 都没有叠加收益，且多层全开反而退化。可复跑审计器为 `training_wss_min/tools/analyze_regp10_transformer_results.py`，结构化结果位于 `training_wss_min/preflight/regp10_transformer_matrix_20260726_results_{analysis.json,summary.csv}` 和 `..._paired_case_stats.csv`；工作簿已更新「实验矩阵总览」「教师汇报视图」「汇总对比」。

## 2026-07-26｜D2 c125×k64 固定 106/0/27：PointNeXt-R + LocalGeoPE（恢复链 `10958→10959`；✅1/1 完成）

以已完成的 `d2_rand5000_c125_k64` 为唯一父配置，新增 `d2_c125_k64_pnxr_geope`。严格冻结 AG/AAA `106/0/27` split、原 train106 全局统计、6D xyz+geom 输入、random5000/SAME、固定 SA centers `125/125/32`、`sa_nsample=64/16/16`、半径、width/head、seed=1234、400 epoch、train-loss 选模和 test27 legacy-vertex 评估。唯一模型变化是 `sa_blocks: (0,0,0)→(1,1,0)` 以启用 PointNeXt-R，以及 `local_geope: false→true`；LocalGeoPE 继续使用 D2 已有的 `abscissa_norm/local_radius/curvature` 三项差分，形成原标准 7D 边编码。

静态逐字段审计 `1/1` 通过，manifest 记录 split/stats/config SHA。首轮门禁 `10956` 的静态/SA1 几何均通过，但通用 GPU smoke 将 D2 合法的空 `feature_stats_path` 误作路径，未执行模型前后向即失败；依赖任务 `10957` 已取消。烟测已改为与 D2 训练时一致地从 train106 重算特征统计，恢复门禁 `10958` 与训练/评估 `10959` 均 `COMPLETED (0:0)`；完成 400 epoch、best/last test27 全云评估、checkpoint strict reload 与 full/chunk 一致性验证。

**结果（best checkpoint，唯一严格对照为原 D2 c125×k64）**：`R²_cb=0.2628→0.2975`（**`+0.0347`**），MAE `2.9260→2.7947 Pa`（`-0.1313`），RMSE `6.4475→6.2938 Pa`（`-0.1537`），high-WSS R² `-0.5266→-0.4461`（`+0.0805`），top10 IoU `+0.0273`。AG/AAA R² 同向提高 `+0.0526/+0.0181`；病例 R² 19 胜/8 负，均差 `+0.0480`、bootstrap 95% CI `[+0.0134,+0.0837]`。这说明 PointNeXt-R + LocalGeoPE 的增益可在完整 AG/AAA `106/0/27` 协议下复现，作为单种子正向筛选信号；仍不能与 mixed138/test36 的 S3 数值直接排名，也不能代替多 seed 确认。

真源：`training_wss_min/preflight/d2_c125_k64_pnxr_geope_20260726_{prepared,static_audit,submission,resubmission,results_analysis,results_summary,paired_case_stats}.json/csv`。

## 2026-07-25｜纯 XYZ + LocalGeoPE 兼容对照（`10929→10930`；✅1/1 完成）

以已完成的 `S2 PointNeXt-R + xyz` 为唯一父配置，新增 `s2_d2k64_pnxr_mixed_xyz_geope`。原 S3 六维输入的 LocalGeoPE 保持 `[(Δxyz/r), ||Δxyz/r||, Δabscissa, Δradius, Δcurvature]` 七维边编码；纯 XYZ 版显式将 `local_geope_feature_indices=[]`，改为 `[(Δxyz/r), ||Δxyz/r||]` 四维边编码，不访问不存在的语义几何列。除 `name/notes/local_geope/local_geope_feature_indices` 外不改任何配置。

新增兼容性单测验证四维 GeoPE MLP 与梯度传播，既有六维 GeoPE 行为保持不变；配置读取、静态差分 `1/1`、提交 dry-run 均通过。GPU 门禁 `10929` 与训练/评估 `10930` 均 `COMPLETED (0:0)`，完成 400 epoch、best/last test36 全云评估。

**结果（best checkpoint）**：纯 XYZ 的 S2 PointNeXt-R 从 `R²_cb=0.1798` 提升至 **`0.2469`**（`Δ=+0.0671`），同时 `MAE -0.2833 Pa`、`RMSE -0.2853 Pa`、high-WSS `R² +0.0139`、top10 IoU `+0.0247`；AG/AAA/ILO 的 R² 分别 `+0.1132/+0.0041/+0.0655`。逐病例 R² 为 27 胜/9 负，均差 `+0.3051`，bootstrap 95% CI `[+0.1062,+0.5750]`。相对同一 S2 的 xyz+geom 父模型 `R²_cb=0.2528`，仍差 `-0.0059`，MAE 几乎持平（`-0.0014 Pa`）、RMSE `+0.0256 Pa`；病例均差 `-0.0061`、95% CI `[-0.0813,+0.0540]` 跨零。故 4D 相对位置编码可显著修复纯 XYZ 的信息缺口，但当前单 seed 证据只支持“接近 xyz+geom”，**不能替换含语义几何输入的 S2/S3 主线**。

真源：`training_wss_min/preflight/xyz_local_geope_20260725_{prepared,static_audit,submission,results_analysis,results_summary,paired_case_stats}.json/csv`。

## 2026-07-25｜S3 正则化强度补齐 + 两两交叉（`10923→10924_[0-19]%4`；✅20/20 完成）

母模板固定为 S3-GEOPE seed=1234：mixed `138/0/36`、xyz+geom、5000 random/SAME、`125/125/32`、`64/16/16`、PointNeXt-R `(1,1,0)`、LocalGeoPE、400 epoch 与 train-loss 选模全部冻结。已完成的单变量 `head dropout=0.10`、DropPath=`0.05/0.10`、NeighborDrop=`0.05` 不重跑；新建 **20** 个独立 run：单变量补齐 8 臂（head=`.05/.15/.20`、DropPath=`.15/.20`、NeighborDrop=`.10/.15/.20`），以及 HeadDrop×DropPath、HeadDrop×NeighborDrop、DropPath×NeighborDrop 三对 `{.05,.10}×{.05,.10}` 交叉共 12 臂。

静态逐字段审计 `20/20` 通过；每臂相对 S3 仅改 `name` 与声明的一个或两个正则字段。GPU 门禁 `10923` 与数组 `10924_[0-19]` 全部 `COMPLETED (0:0)`，各臂完成 400 epoch、best/last test36 全云评估。

**结果（均为 seed=1234 对 S3-GEOPE 的严格配对）**：S3 父模型为 `R²_cb=0.2924`。单变量中最优是 **HeadDrop=0.15：`0.2983`，`ΔR²_cb=+0.0059`**，但 MAE `+0.0105 Pa`；DropPath 的温和区间是 `0.05/0.10`（分别 `+0.0017/+0.0034`），`0.15` 明显退化 `-0.0179`，`0.20` 近零 `+0.0006`。NeighborDrop 四个强度均为负（最佳也为 `0.10: -0.0037`），关闭该方向。12 个两两交叉中仅 DropPath=.05 + NeighborDrop=.10（`+0.0031`）及 HeadDrop=.10 + DropPath=.10（`+0.0007`）为正，但均未超过 HeadDrop=.15 单变量；其余组合为负。因此没有可执行的组合协同信号，也不提升任何臂为 S3 新锚点。

这批是单种子、已反复使用的 mixed test36 工程筛选，不能用 `+0.0059` 等小幅差异作正式 Go。若后续需要确认，只将 HeadDrop=.15 和 DropPath=.10 作为互不组合的候选补 seeds `7/2025`，并保留 MAE/high-WSS/分域护栏。

真源：`training_wss_min/preflight/s3_regularization_completion_20260725_{prepared,static_audit,submission,results_analysis,results_summary,paired_case_stats}.json/csv`。

## 2026-07-25｜7 月 24 日三批实验结果回填（✅16/16 新训练完成）

`10871→10872_[0-11]` 的 S3 正则化 12 臂、`10882→10883` 的 SA3 coarse-attention 单臂，以及 `10903→10904_[0-2]` 的三锚点纯 XYZ 对照均已完成；每个 run 均有 400 epoch history、train-loss 选出的 best、last 与 test 全云评估，完整性审计全部通过。以下仍仅是反复使用的历史 test36/test27 工程筛选结果。

### 纯 XYZ：三个锚定协议一致 No-Go

| 严格对照（xyz+geom → xyz） | R²_cb | ΔR²_cb | ΔMAE Pa | ΔRMSE Pa | Δhigh-WSS R² | 病例 ΔR² 95% CI | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| D2 c125×k64，test27 | 0.2628 → 0.2047 | -0.0581 | +0.1812 | +0.2492 | -0.0833 | -0.0510 `[-0.1113,+0.0094]` | No-Go |
| M1/S0 mixed138/test36 | 0.2614 → 0.1656 | -0.0958 | +0.3353 | +0.4073 | -0.0217 | -0.3597 `[-0.8510,-0.0748]` | No-Go |
| S2 PointNeXt-R mixed138/test36 | 0.2528 → 0.1798 | -0.0730 | +0.2819 | +0.3109 | -0.0122 | -0.3112 `[-0.6287,-0.0753]` | No-Go |

几何输入是三条协议共同的必要信息，不再扩展纯 XYZ 主线。纯 XYZ 内 S2 残差核相对 M1 只有 `ΔR²_cb=+0.0142`，病例 CI `[-0.1504,+0.2471]` 跨零，不能把该结构小信号解释为对缺失几何的补偿；也不能以 M1/S2 的 IoU 小幅上升抵消主 R²、MAE/RMSE 的一致退化。真源：`training_wss_min/preflight/xyz_input_ablation_20260724_{results_analysis,results_summary,paired_case_stats}.json/csv`。

### S3 正则化：无正式 Go；DropPath 0.05 为最强弱信号

相对同 seed S3-GEOPE，四个严格单变量的三种子均值为：head dropout 0.10 `ΔR²_cb=+0.0057±0.0140`（2/3 正）、DropPath 0.05 `+0.0089±0.0103`（**3/3 正**）、DropPath 0.10 `+0.0076±0.0055`（**3/3 正**）、NeighborDrop 0.05 `-0.0043±0.0223`（2/3 正但均值负）。前三者 MAE/RMSE 均值下降且 high-WSS R² 均约 `+0.016`；所有域均值护栏通过。由于没有任何臂达到预注册正式 Go 门槛 `mean ΔR²_cb≥+0.015`，本批不替换 S3 锚点、不自动组合正则；DropPath 0.05 仅登记为优先的 **Weak-Go** 后续确认候选，DropPath 0.10 次之，head dropout 需谨慎，NeighborDrop 关闭。真源：`training_wss_min/preflight/s3_regularization_{results_analysis,results_summary,paired_case_stats}.json/csv`。

### SA3 coarse attention：单种子 Weak-Go，必须补 seed

S3-GEOPE seed1234 的 R²_cb `0.2924→0.3007`（`+0.00835`），RMSE `-0.0375 Pa`、high-WSS R² `+0.0353`，AAA/ILO 分域分别 `+0.0326/+0.0117`；代价是 MAE `+0.0114 Pa`、AG `-0.0113`、top10 IoU `-0.0023`，病例平均 ΔR²=`-0.0145`、95% CI `[-0.0436,+0.0122]` 跨零。故只能记录为单种子 Weak-Go 信号，不能与 DropPath 或其它结构模块组合，也不能提升为主锚；若要继续，应首先作相同配置的 seeds `7/2025` 严格确认。真源：`training_wss_min/preflight/s3_sa3_coarse_attention_single_{results_analysis,results_summary,paired_case_stats}.json/csv`。

## 2026-07-24｜D2-K64 三锚点纯 XYZ 输入对照（`10903→10904_[0-2]%3` 已提交，等待 GPU）

为分离结构/数据协议与输入特征的作用，新增三个严格对照：`d2_c125_k64_xyz`（D2 c125×k64，AG/AAA `106/0/27`）、`m1_d2k64_mixed138_test36_xyz`（M1/S0，mixed `138/0/36`）和 `s2_d2k64_pnxr_mixed_xyz`（S2 PointNeXt-R 残差版，mixed `138/0/36`）。三者均只把 `data.input_features` 从 `xyz+abscissa_norm+local_radius+curvature` 改为 `x/y/z`；split、target/feature stats、5000 random/SAME、SA `125/125/32`、`64/16/16`、seed=1234、400 epoch、train-loss 选模和评估协议均保持锚定配置不变。

静态差分审计已通过 `3/3`，配置与父配置均记录 SHA256；GPU 预检 Job `10903` 当前因集群资源不足按 Slurm 队列等待，训练与 best/last 全云评估数组 `10904_[0-2]%3` 设为 `afterok:10903`，不会在门禁未过时启动。真源：`training_wss_min/preflight/xyz_input_ablation_20260724_{prepared,static_audit,submission}.json`。

## 2026-07-24｜S3 + SA3 coarse attention（单种子 `1234`；`10882→10883` 已提交）

以已完训 S3-GEOPE seed1234 为唯一父配置，仅打开 `model.coarse_attention=true`：LocalGeoPE 保持开启，attention 只位于 SA3 的 32 个 coarse centers，4 heads；数据 split/stats、SAME、`125/125/32`、`64/16/16`、PointNeXt-R `(1,1,0)`、400 epoch、train-loss 选模与 support seed 全部冻结，head dropout/DropPath/NeighborDrop 均为 0。本轮按当前决定只跑 seed1234，不补多种子。

配置生成、静态硬门禁、GPU 门禁和训练脚本均已补齐。本地逐字段审计确认相对 S3 仅 `name/model.coarse_attention` 两处差异，`test_d2_k64_wave2_modules` 10/10 通过。正式门禁 Job `10882` 已提交；训练与 best/last test36 全云评估 Job `10883` 使用 `afterok:10882`。当前 4 卡由正则化数组占用，两个新作业在队列中等待资源/依赖。

结果只与同 seed S3 的 `R²_cb=0.2924` 作严格配对；`ΔR²_cb≥+0.012` 且 MAE、high-WSS 和分域护栏无实质回退，才记录为单种子正信号。test36 是历史工程筛选集，且本实验不做多种子，因此不能形成稳定泛化结论。

## 2026-07-24｜S3-GEOPE 正则化三种子矩阵（门禁通过；`10872_[0-11]%4` 训练中）

锚点固定为 S3 `D2-K64 + PointNeXt-R (1,1,0) + LocalGeoPE v1`。S3 的 absolute `R²_cb=0.2924/0.2743/0.2713`（均值 `0.2793±0.0114`）；相对 M1 的三种子增益 `+0.0310/+0.0059/+0.0303`，3/3 为正。test36 仍是历史工程筛选集。

首波共 12 个严格单变量配置：

| 变体 | 唯一变化 | Seeds | 当前状态 |
|---|---|---|---|
| `head_dropout01` | `model.dropout=0.10` | `1234/7/2025` | submitted |
| `droppath005` | `model.drop_path_rate=0.05` | `1234/7/2025` | submitted |
| `droppath010` | `model.drop_path_rate=0.10` | `1234/7/2025` | submitted |
| `neighbordrop005` | `model.neighbor_drop_rate=0.05` | `1234/7/2025` | submitted |

实现中 DropPath 对每个病例图共享 mask，并沿 PointNeXt-R block 深度线性增加；NeighborDrop 在 SA 与 InvRes 邻域内训练时随机丢边，但每个 center 固定保留最近边，评估时关闭。默认值均为 0，不改变历史配置与 checkpoint key。

本地 `compileall`、新增 3 项定向测试及 `test_d2_k64_wave2_modules` 全部 10/10 通过；prepared manifest 为 `training_wss_min/preflight/s3_regularization_matrix_prepared.json`，静态单变量差分 12/12 passed。正式门禁 Job `10871` 已 `COMPLETED (0:0)`：SA1 几何合同、12 个 CUDA 前后向、checkpoint strict reload 与 full/chunk 一致全部 passed；训练数组 `10872` 已按 `afterok` 启动并占用 4 张 GPU。

当前执行与后续架构优先级见 [S3-GEOPE 锚定正则化与架构优化执行计划](WSS最小化_S3-GEOPE锚定_正则化与架构优化执行计划_2026-07-24.md)。

## 2026-07-23｜D2 c125×k64：ILO 两协议 + PointNeXt/GeoPE/Attention/SEP（single seed=1234；✅9/9完成）

### 三种子确认回填（`10848→10849[0-5]`；✅6/6完成）

seed=`7/2025` 的每个配置只改变 `name` 和 `train.seed`；mixed `train138/test36`、split/stats SHA、SAME、`125/125/32`、`64/16/16` 和 `support_seed=1234` 均冻结。门禁确认 S2/S3 残差块、S3 LocalGeoPE、checkpoint 严格重载、full/chunk 一致与无 NaN/Inf；训练/eval 6/6 `COMPLETED (0:0)`。

| 比较（best） | ΔR²_cb seeds `1234/7/2025` | mean±sd；正/负 | 结论 |
|---|---|---|---|
| S2−M1 | `-0.0086/+0.0007/+0.0335` | `+0.0086±0.0221`；2/1 | 裸 PointNeXt-R 不稳定，维持 No-Go。 |
| S3−S2 | `+0.0396/+0.0052/-0.0033` | `+0.0138±0.0227`；2/1 | LocalGeoPE 独立增量为正；MAE 三 seed 均降 `-0.1172 Pa`，case-mean R² `+0.0619`（78/30；bootstrap `[+0.0401,+0.0837]`）。 |
| S3−M1 | `+0.0310/+0.0059/+0.0303` | `+0.0224±0.0143`；3/0 | 端到端确认：MAE `-0.1280 Pa`、RMSE `-0.0987 Pa`、high-WSS `+0.0349`、IoU `+0.0219`、Spearman `+0.0592`；case-mean R² 80/28，bootstrap `[+0.0425,+0.0868]`。 |

S3 通过本批决策规则，只提出后续互不组合的 `S3+DropPath 0.05`、`S3+DropPath 0.10`、`S3+NeighborDrop 0.05`，本批未自动混入 Drop。AAA unrupture 轻微回退、少数 high-WSS seed 回退且绝对 high-WSS R²仍为负，继续作护栏。test36 是历史工程筛选集，未用于 checkpoint 选择，也不是独立外部测试集。真源：`training_wss_min/preflight/d2_k64_ilo_structure_three_seed_confirmation_{analysis.json,summary.csv,paired_case_seed.csv}`。

**完整性**：门禁 `10837/10843` 和数组 `10838_[0-4]/10844_[0-3]` 全部 `COMPLETED (0:0)`。9 个新 run 均完成 400 epoch、best/last checkpoint、best/last 全云评估、逐病例 CSV、配置哈希和有限数值审计。主结果固定使用 `ckpt_best(train_loss)`；病例 bootstrap 为 20,000 次。所有结论均为单种子筛选，不能替代后续三种子确认。

### 数据协议结果

| 严格配对 | 对照 → 处理 | R²_cb（Δ） | MAE_cb（Δ, Pa） | RMSE_cb（Δ, Pa） | high-WSS R²（Δ） | top10 IoU（Δ） | 病例 R² 胜/负；均差95% CI | 判定 |
|---|---|---:|---:|---:|---:|---:|---|---|
| F1−F0 | fixed test27：AG/AAA train106 → +ILO41 train147 | 0.2459 (**-0.0169**) | 2.9432 (+0.0172) | 6.5211 (+0.0736) | -0.5511 (-0.0245) | 0.1739 (+0.0105) | 10/17；`-0.0116 [-0.0340,+0.0118]` | **No-Go 信号** |
| M1−M0 | mixed test36：ILO zero-shot → ILO32 入训 | 0.2614 (-0.0059) | 2.7145 (+0.0003) | 6.4809 (+0.0258) | -0.4914 (-0.0150) | 0.1613 (**-0.0173**) | 15/21；`+0.0071 [-0.0282,+0.0539]` | **不支持总体晋级** |

F1 在相同 AG/AAA test27 上，AG/AAA R² 分别 `0.2396→0.2229`、`0.2750→0.2577`，说明加入全部 ILO41 没有带来跨域正迁移。M1 在 mixed test36 上的 ILO R² `0.2647→0.2756`（`+0.0109`），但 AAA `0.2279→0.1948`（`-0.0331`），overall/high-WSS/IoU 同时回退；这更像训练容量在队列间重新分配，而不是总体模型变好。故 ILO 混训只记录为**条件性域权衡**，不以 ILO9 小样本的单一正数晋级。

### 数据扩容负迁移归因（跨两轮综合｜为什么「加了 ILO 反而更差」）

**现象是稳健的，不是单次波动。** 两轮、五个严格配对的整体 `R²_cb` 全部为负差：

| 轮次 | 配对 | 变化 | R²_cb | Δ |
|---|---|---|---:|---:|
| Q2V/ILO（07-18） | D1 固定 test27、冻结统计，+ILO41 训练 | 加 ILO | 0.2724→0.2516 | **-0.0208** |
| Q2V/ILO（07-18） | D2 同 test36，+ILO32 vs 零样本 | 加 ILO | 0.2646→0.2493 | **-0.0153** |
| Q2V/ILO（07-18） | D3 同 test36 控制，+ILO32（重算统计再降到 0.2459） | 加 ILO | 0.2981→0.2915 | **-0.0066** |
| D2-K64（07-23） | F1−F0 固定 test27，AG/AAA train106→+ILO41 train147 | 加 ILO | 0.2628→0.2459 | **-0.0169** |
| D2-K64（07-23） | M1−M0 混合 test36，ILO 零样本→ILO32 入训 | 加 ILO | 0.2672→0.2614 | **-0.0058** |

**机制是「容量再分配」而非「模型变好/变坏」。** M1−M0 分域拆解最能说明问题：加入 ILO 训练后，只有 ILO 自身 `0.2647→0.2756`（`+0.0109`）微升，AG `0.2785→0.2745`（`-0.0040`）、AAA `0.2279→0.1948`（`-0.0331`）双降，整体净降。模型没有获得跨域正迁移，只是把有限容量从 AAA/AG 匀给了 ILO。F1−F0 在纯 AG/AAA test27 上 AG/AAA 同步退（`-0.0167/-0.0173`）则说明：即便测试端不含 ILO，把 ILO41 塞进训练也污染了原有两域。

**三个根因（真源：[代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md) 2026-07-20 A1b 归一化条目 + 2026-07-15 v3→v4 混合条目）：**

1. **归一化目标统计是 point-pooled，被高密度队列主导（主因）。** AAA/ILO 每例 5–7 万壁面点、AG 仅 ~1.3 万点（密度差 4–6×），AAA 稠密网格贡献混合 target 统计约 78.8% 的全云点。node03 实测 pooled 统计 `μ=0.6456`，而逐病例等权 `μ=0.8166`、逐父系等权 `μ=0.7798`——pooled 被稠密队列拉低；混入 AAA/ILO 使 log-WSS 全局统计从 `1.122±1.096` 变 `0.553±1.373`，稀疏的 AG 在归一化空间被错误居中、高幅值区被压缩。
2. **缺少疾病域条件（cohort condition）。** ILO 相对 AG/AAA 是不同解剖形状先验，无 cohort 标识 + 单目标定义下混训只能折中 → 对 ILO 属净负迁移。
3. **log-MSE 天然轻视高 WSS 尾部**，样本更杂时该偏差被摊得更明显（high-WSS R²、p99 幅值比反复回退印证）。几何层面还有放大器：ILO ~8 万点在 `random/FPS 5000 + nsample=16~64` 下被严重截断（评估分辨率截断率≈1.0），物理尺度进不了邻域。测试端 ILO-1 仅 3 例、ILO-0 仅 6 例，小亚组只作探索性线索。

**反证：修正根因后掉点消失（说明是口径问题、非 ILO 数据不可用）。**

- **修归一化口径**（07-20 A1b 逐父系等权）：R²_cb `0.2459→0.2753`（**+0.0295**），high-WSS、p99 幅值比同升。
- **补局部几何**（本轮 S3 LocalGeoPE）：是唯一对 AG、AAA、ILO 三域同时提高的臂，绝对 `R²_cb=0.2924` 为 mixed test36 全场最高（见下节结构模块结果）。

因此正确结论不是「撤掉 ILO」，而是「默认 point-pooled 归一化 + 无域条件 + log-MSE 口径下直接混入高密度异解剖 ILO 会触发负迁移」；下一步走 **A1b 逐父系等权归一化 ×（cohort 条件 / LocalGeoPE）** 并补 seeds `{7,2025}` 多种子确认。分域对比图件：[`docs/03-汇报材料/WSS_ILO扩数据分域R2对比_2026-07-23.png`](../03-汇报材料/WSS_ILO扩数据分域R2对比_2026-07-23.png)。

### 结构模块结果（共同 mixed `138/0/36`）

| Arm | 唯一变化/严格对照 | R²_cb | ΔR²_cb | MAE_cb | high-WSS R² | top10 IoU | AG / AAA / ILO R² | 判定 |
|---|---|---:|---:|---:|---:|---:|---|---|
| M1/S0 | D2-K64 mixed 公共父控制 | 0.2614 | — | 2.7145 | -0.4914 | 0.1613 | 0.2745 / 0.1948 / 0.2756 | 父控制 |
| S1-MFEAT | M1 + `radius_gradient/coord_scale` | 0.2631 | +0.0018 | 2.7174 | -0.4859 | 0.1589 | 0.2683 / 0.2062 / 0.2790 | 暂不支持 |
| S2-PNXR | M1 + PointNeXt-R `(1,1,0)` | 0.2528 | -0.0086 | 2.7455 | -0.5043 | 0.1558 | 0.2698 / 0.1810 / 0.2666 | No-Go 信号 |
| **S3-GEOPE** | **S2 + LocalGeoPE v1** | **0.2924** | **+0.0396** | **2.5747** | **-0.4495** | **0.1716** | **0.3112 / 0.2173 / 0.3077** | **强晋级候选** |
| S4-ATTN | S2 + SA3 coarse attention | 0.2706 | +0.0178 | 2.7092 | -0.4606 | 0.1587 | 0.2899 / 0.2133 / 0.2725 | 弱正候选 |
| S5C | independent query + 原3NN | 0.2498 | — | 2.7447 | -0.5220 | 0.1680 | 0.2770 / 0.2100 / 0.2311 | SEP匹配控制 |
| S5-SEP | S5C + Conservative SEP-Kernel | 0.2733 | +0.0235 | 2.7234 | -0.4587 | 0.1693 | 0.2805 / 0.2103 / 0.2917 | 条件正候选 |

S3−S2 同时改善 `R²_cb +0.0396`、case-mean R² `+0.0921`、MAE `-0.1708 Pa`、RMSE `-0.1750 Pa`、high-WSS R² `+0.0548`、Spearman `+0.0594` 与 IoU `+0.0157`；逐病例 R² 29胜7负，均差95% CI `[+0.0600,+0.1259]`，逐病例 MAE 32例改善、4例退化。它也是本轮唯一对 AG、AAA、ILO 三个父域都同时提高的结构臂。S4 的病例 R² 19胜17负且 CI 跨零，效果弱于 S3。S5 必须只与 S5C 比：其 aggregate `R²_cb +0.0235`、RMSE `-0.1031 Pa`、high-WSS `+0.0633`，但 case-mean R² `-0.0047`、负例 `3→7`、病例 R² CI `[-0.0316,+0.0211]`，暂不能称为稳定 decoder 改进。

**下一步**：

1. 第一优先补 `M1/S2/S3 × seeds {7,2025}`，与现有 seed1234 组成三种子配对：同时确认“PointNeXt-R 单独效应”“LocalGeoPE 增量效应”和“S3 对原 D2-K64 mixed 父模型的净收益”。
2. 若 S3 三种子仍为正，再做单变量轻正则 Gate；优先残差分支 `DropPath 0.05/0.10` 或 `NeighborDrop 0.05`，不直接把较大的 head dropout 混进 S3。Drop 是正交问题，不用于解释本轮结果。
3. S4 与 S5 暂不和 S3 组合；分别等 S3 和 decoder 多种子证据后再决定。S1 与裸 S2 停止扩展。

### 2026-07-23→24｜S3-GEOPE 根因矩阵 12 配置 ✅12/12 完训并回填（门禁 `10857`／array `10858_[0-11]%6`）

以三种子确认的 **S3-GEOPE**（父模板 `s3_d2k64_pnxr_geope_mixed`，D2-K64 mixed `138/0/36`）为锚，对上文 [数据扩容负迁移归因](#数据扩容负迁移归因跨两轮综合为什么加了-ilo-反而更差) 的三个根因各开单变量臂。均 400ep、train-loss 选模、冻结 split 与 support-query 协议；每臂只允许一处字段差异（`prepare_s3_rootcause_matrix.py` 生成、`audit_s3_rootcause_static.py` 12/12 passed）。门禁与 12 个 array 任务全部 `COMPLETED (0:0)`，best/last checkpoint、全云评估、逐病例 CSV、配置 SHA、有限性审计与 400-epoch history 全部通过。

**终裁：整体无净增（No-Go），但机制上证实了根因 #1。** S3 父模板 R²_cb 基线为 `0.2924/0.2743/0.2713`（seed 1234/7/2025）。没有任何臂能跨种子稳定超过 S3——口径臂在零附近震荡（casebal 三种子均值 `+0.0040` 但 seed1234 `-0.0166`；cohortbal 均值 `-0.0018`，seed1234 case-R² 95%CI `[-0.056,-0.002]` 显著为负），尾部与组合臂单种子均为负。真正稳健的信号在**分域**：几乎每个平衡口径/域条件/尾部臂都把 **AAA 抬回**（cohortbal 三种子 `+0.0255/+0.0362/+0.0249`、casebal `+0.0165/+0.0471/+0.0140`、rawHuber `+0.0281`），AAA 正是被 ILO 压掉的域——但代价是 **ILO 相应回落**（cohortbal `-0.0291/+0.0084/-0.0149`、cohort_onehot 三种子全降、组合臂 `-0.0672`），AG 基本持平。因此这是把容量在 AAA↔ILO 间**重新分配**（原 ILO 负迁移的镜像），总体持平。含义：**LocalGeoPE(S3) 已吸收了归一化口径在无几何 Q2V 基座上贡献的净收益（A1b 曾 `+0.0295`），在 S3 骨干上两条杠杆不叠加**，口径修正只是拿 ILO 换 AAA。high-WSS R² 全部仍为负（rawHuber02 `-0.456` vs S3 `-0.450`，尾部未解决）。S3-GEOPE 仍为参考模型。

| 臂（vs S3 父同种子） | R²_cb | ΔR²_cb | ΔAG | ΔAAA | ΔILO | high-WSS | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| RC0 pooled138 (s1234) | 0.2784 | -0.0139 | -0.029 | **+0.008** | -0.014 | -0.468 | 单seed，负 |
| RC1 casebal (s1234/7/2025) | 0.276/0.296/0.279 | 均值 **+0.0040**（2/1） | — | **全+**（+0.017/+0.047/+0.014） | -0.046/+0.029/-0.017 | — | 震荡跨零 |
| RC2 cohortbal (s1234/7/2025) | 0.273/0.281/0.278 | 均值 -0.0018（2/1） | — | **全+**（+0.026/+0.036/+0.025） | -0.029/+0.008/-0.015 | — | AAA最稳回补但总体持平 |
| RC3 cohort_onehot (三seed) | 0.281/0.279/0.253 | 均值 -0.0084（1/2） | — | +0.029/+0.041/-0.010 | 全降 | — | No-Go |
| RC4 rawHuber0.2 (s1234) | 0.2838 | -0.0086 | -0.001 | +0.028 | -0.042 | -0.456 | 尾部No-Go |
| RC5 cohortbal+onehot (s1234) | 0.2655 | -0.0268 | -0.002 | -0.006 | **-0.067** | -0.511 | 过校正，最差 |

**下一步**：停止在 S3 骨干上继续试归一化口径变体（已被 GeoPE 吸收）。剩余真问题是 high-WSS 尾部（所有臂仍负）与 AAA↔ILO 容量竞争；后续应转向**尾部感知/域感知的损失加权或分域容量**（如逐域 head、按域重加权），而非更多全局归一化微调。DropPath/NeighborDrop 仍待主干实现后单独 Gate。真源：`training_wss_min/preflight/s3_rootcause_results_{analysis.json,summary.csv,paired_case_stats.csv}`；xlsx 已回填「实验矩阵总览/教师汇报视图/汇总对比」。所有结论仍为历史工程筛选集，不作独立泛化声明。

---

<details><summary>提交时设定（2026-07-23 晚，已完训）</summary>

均 400ep、train-loss 选模、冻结 split 与 support-query 协议；每臂只允许一处字段差异（`prepare_s3_rootcause_matrix.py` 生成、`audit_s3_rootcause_static.py` 本地 12/12 passed）。

| 方向（根因） | 臂 | 唯一变化 | 种子 | 目标 log μ |
|---|---|---|---|---|
| 归一化口径 #1 | `caliber_pooled138` | 目标统计→train138 pooled 重算 | 1234 | 0.6456 |
| 归一化口径 #1 | `caliber_casebal` | →逐病例等权 A1 | 1234/7/2025 | 0.8166 |
| 归一化口径 #1（**主**） | `caliber_cohortbal` | →逐父系等权 A1b | 1234/7/2025 | 0.7798 |
| 域条件 #2 | `cohort_onehot` | +cohort one-hot（dim 6→9） | 1234/7/2025 | 0.5718（父，不变） |
| 尾部损失 #3 | `rawhuber02` | +raw-Huber λ=0.2 探针 | 1234 | 0.5718（父，不变） |
| 探索组合 #1+#2 | `cohortbal_onehot` | cohortbal+one-hot（非归因） | 1234 | 0.7798 |

父模板当前口径为 `control106` pooled（log μ=0.5718，且不含 ILO）——恰好双重踩中根因 #1。`pooled138→casebal/cohortbal` 用于分离"含 ILO 重算统计"与"跨域平衡"两效应；DropPath/NeighborDrop 需改主干本批不含（仅 head dropout 已实现），推迟到单独实现。真源：`training_wss_min/preflight/s3_rootcause_matrix_{prepared,submission}.json`。

</details>
4. high-WSS R² 虽由 S2 的 `-0.5043` 改善到 S3 的 `-0.4495`，仍为负；三种子确认后再开尾部目标/排序校准实验，不能用结构提升掩盖高值区仍未达标。

结构化真源：`training_wss_min/preflight/d2_k64_ilo_structure_{results_analysis.json,results_summary.csv,paired_case_stats.csv}`。

## 2026-07-22｜bridge random5000+500c+k64 的 SEP 对照臂（single seed=1234；historical test27；✅完成｜No-Go）

在 `bridge random5000 + 500c + k64`（`bridge_rand5000_fixed500_k64`，SAME）基础上，补一个唯一变量为 SAME→SEP 的对照臂 `bridge_rand5000_fixed500_k64_sep`：`data.query_mode` 由 `"same"` 改为 `"independent"`，其余 support 点数（random5000）、SA 链（固定 center 500/125/32）、SA1 `knn_cover k=64`、width=32、batch_cases=8、seed=1234、400 epoch、train-loss 选模、`legacy_vertex` 评估口径全部与父实验相同（逐字段 diff 已确认仅此一处差异）。门禁 Job `10804`（SA1 knn_cover 覆盖率几何审计 + GPU 预检）→训练 Job `10805`（`afterok:10804`）均 `COMPLETED (0:0)`；400 epoch、best/last checkpoint、27 例 CSV、配置哈希与 `runtime_preserves_frozen_config`（除 `query_mode` 外逐字段一致）均通过审计，无 NaN/Inf。

| 配置 | 物理R²_cb（Δ） | 归一化R²_cb（Δ） | 病例均值R²（Δ） | 负例 | MAE（Δ） | RMSE（Δ） | Spearman（Δ） | top10 IoU（Δ） | high-WSS R²（Δ） | p99比（Δ） |
|---|---|---|---|---|---|---|---|---|---|---|
| bridge random5000 + 500c + k64（SAME，复用父实验） | 0.2439 (0.0000) | 0.6139 (0.0000) | 0.1910 (0.0000) | 2 | 2.9345 (0.0000) | 6.5294 (0.0000) | 0.7351 (0.0000) | 0.1694 (0.0000) | -0.5651 (0.0000) | 0.6688 (0.0000) |
| **bridge random5000 + 500c + k64 SEP（唯一变量 SAME→SEP）** | 0.2329 (-0.0111) | 0.6065 (-0.0074) | 0.1688 (-0.0222) | **1** | 2.9693 (+0.0348) | 6.5770 (+0.0476) | 0.7235 (-0.0116) | 0.1776 (+0.0082) | -0.6019 (-0.0368) | 0.6774 (+0.0086) |

**判读**：与既有 Q1V/Q2V（SAME/SEP）对照不同，本臂的 SEP 在绝大多数主指标上同向回退——物理/归一化 R²_cb、病例均值 R²、MAE、RMSE、Spearman、high-WSS R² 全部劣于 SAME；仅负例（2→1）、top10 IoU（+0.0082）与 p99 比（+0.0086）小幅改善，且改善幅度均小于回退幅度。因此判定 **bridge 家族下 SAME→SEP 为 No-Go**，不将 SEP 并入 bridge 的后续候选；主线仍以 SAME（`bridge_rand5000_fixed500_k64`）代表该点数标度对照臂，与 SA1-scale 矩阵①点数标度、③降center×大邻域两节的现有判读一致。best/last 敏感性：SEP 的 `last` 物理 R²_cb 为 0.2508，比 best 高 `+0.0179`，不改变 train-loss 选模规则，也不逆转上述 No-Go 判断。仍是历史 test27 同协议工程比较，非独立确认。

真源：`training_wss_min/preflight/bridge_sep_followup_results_analysis.json`、`bridge_sep_followup_results_summary.csv`。

## 2026-07-21｜PointNet++ SA1-scale 矩阵结果（全点/10k/降center×大k/宽度/Stem；single seed=1234；historical test27）

**状态**：17/17 新 run 完成且 17/17 审计通过。锚点 Q1V random5000+ball16（0.2763）；导师标准 KNN-8-cover w32（0.2589）。协议同 SA1 分组矩阵（106/0/27、seed1234、400ep、train-loss 选模、train106 global log-z、legacy_vertex、物理 R²_cb 主指标）。**test27 为多轮复用的工程比较集，本轮全部是同协议筛查，非独立确认**；D1-prop 族 batch=2（其余 batch=8）存在优化混杂，跨家族比较需声明。审计逐项检查冻结 SHA、seed、split、400 epoch、best/last eval、27 例 CSV、有限数值与 knn_cover 覆盖硬门（全部 100%）。

### 1) 点数标度（固定 500/125/32 center；Δ 相对 Q1V 锚点）

| 配置 | 物理R²_cb（Δ） | 归一化R²_cb（Δ） | 病例均值R²（Δ） | 负例 | MAE（Δ） | Spearman（Δ） | top10 IoU（Δ） | high-WSS R²（Δ） | p99比（Δ） |
|---|---|---|---|---|---|---|---|---|---|
| Q1V random5000 + ball16（复用锚点） | 0.2763 (0.0000) | 0.5982 (0.0000) | 0.2114 (0.0000) | 1 | 2.8844 (0.0000) | 0.7222 (0.0000) | 0.1803 (0.0000) | -0.5134 (0.0000) | 0.6912 (0.0000) |
| bridge random5000 + 500c + k64 | 0.2439 (-0.0324) | 0.6139 (0.0157) | 0.1910 (-0.0204) | 2 | 2.9345 (0.0501) | 0.7351 (0.0129) | 0.1694 (-0.0109) | -0.5651 (-0.0517) | 0.6688 (-0.0224) |
| D3 random10000 + k64 | 0.2262 (-0.0502) | 0.5882 (-0.0100) | 0.1710 (-0.0404) | 1 | 2.9642 (0.0798) | 0.7201 (-0.0021) | 0.1523 (-0.0280) | -0.6347 (-0.1213) | 0.6745 (-0.0167) |
| D3 random10000 + k128 | 0.2240 (-0.0523) | 0.5926 (-0.0056) | 0.1665 (-0.0449) | 1 | 2.9527 (0.0683) | 0.7217 (-0.0004) | 0.1580 (-0.0223) | -0.6443 (-0.1308) | 0.6680 (-0.0232) |
| D3 random10000 + k256 | 0.2608 (-0.0155) | 0.6060 (0.0077) | 0.2085 (-0.0029) | 1 | 2.9103 (0.0259) | 0.7310 (0.0089) | 0.1772 (-0.0031) | -0.5510 (-0.0376) | 0.6945 (0.0033) |
| D1-fixed 全点 + k64 | 0.2459 (-0.0304) | 0.5954 (-0.0028) | 0.1830 (-0.0284) | 2 | 2.9566 (0.0722) | 0.7177 (-0.0045) | 0.1779 (-0.0024) | -0.5342 (-0.0208) | 0.6859 (-0.0053) |
| D1-fixed 全点 + k128 | 0.2227 (-0.0536) | 0.5899 (-0.0083) | 0.1742 (-0.0372) | 1 | 2.9830 (0.0986) | 0.7192 (-0.0030) | 0.1600 (-0.0203) | -0.6377 (-0.1243) | 0.6370 (-0.0542) |
| D1-fixed 全点 + k256 | 0.2472 (-0.0291) | 0.5964 (-0.0018) | 0.1791 (-0.0323) | 3 | 2.9759 (0.0915) | 0.7158 (-0.0064) | 0.1706 (-0.0097) | -0.5651 (-0.0516) | 0.6728 (-0.0184) |

### 2) 比例 center vs 固定 center（全点；Δ 相对同 k 的 D1-fixed；batch 2 vs 8 混杂）

| 配置 | 物理R²_cb（Δ） | 归一化R²_cb（Δ） | 病例均值R²（Δ） | 负例 | MAE（Δ） | Spearman（Δ） | top10 IoU（Δ） | high-WSS R²（Δ） | p99比（Δ） | 覆盖率 min | 最大重叠 | 组大小 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1-prop 全点比例center + k64（batch2） | 0.2132 (-0.0328) | 0.5886 (-0.0068) | 0.1602 (-0.0228) | 3 | 3.0144 (0.0579) | 0.7172 (-0.0005) | 0.1679 (-0.0101) | -0.6284 (-0.0942) | 0.6351 (-0.0508) | 1.0000 | 0.9062 | 64–64 |
| D1-prop 全点比例center + k128（batch2） | 0.2054 (-0.0173) | 0.5825 (-0.0074) | 0.1707 (-0.0035) | 5 | 2.9941 (0.0111) | 0.7163 (-0.0029) | 0.1795 (0.0194) | -0.6318 (0.0059) | 0.6600 (0.0229) | 1.0000 | 0.9844 | 128–128 |
| D1-prop 全点比例center + k256（batch2） | 0.2024 (-0.0448) | 0.5841 (-0.0123) | 0.1418 (-0.0373) | 2 | 3.0215 (0.0456) | 0.7180 (0.0022) | 0.1716 (0.0010) | -0.6240 (-0.0590) | 0.6386 (-0.0342) | 1.0000 | 1.0000 | 256–256 |

### 3) 降 center × 大邻域（random5000；Δ 相对 KNN-8-cover w32）

| 配置 | 物理R²_cb（Δ） | 归一化R²_cb（Δ） | 病例均值R²（Δ） | 负例 | MAE（Δ） | Spearman（Δ） | top10 IoU（Δ） | high-WSS R²（Δ） | p99比（Δ） | 覆盖率 min | 最大重叠 | 组大小 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1V KNN-8-cover w32（复用） | 0.2589 (0.0000) | 0.6042 (0.0000) | 0.1917 (0.0000) | 1 | 2.9443 (0.0000) | 0.7232 (0.0000) | 0.1717 (0.0000) | -0.5181 (0.0000) | 0.7302 (0.0000) | 1.0000 | 0.8750 | 8–84 |
| bridge random5000 + 500c + k64 | 0.2439 (-0.0150) | 0.6139 (0.0097) | 0.1910 (-0.0007) | 2 | 2.9345 (-0.0098) | 0.7351 (0.0118) | 0.1694 (-0.0023) | -0.5651 (-0.0470) | 0.6688 (-0.0614) | 1.0000 | 1.0000 | 64–82 |
| D2 c250 × k64 | 0.2474 (-0.0116) | 0.6099 (0.0057) | 0.1983 (0.0066) | 1 | 2.9169 (-0.0275) | 0.7310 (0.0078) | 0.1722 (0.0004) | -0.5675 (-0.0494) | 0.6834 (-0.0467) | 1.0000 | 0.9688 | 64–136 |
| D2 c125 × k64 | 0.2628 (0.0039) | 0.6114 (0.0073) | 0.1907 (-0.0010) | 1 | 2.9260 (-0.0183) | 0.7287 (0.0055) | 0.1634 (-0.0083) | -0.5266 (-0.0085) | 0.6900 (-0.0401) | 1.0000 | 0.7969 | 64–251 |
| D2 c250 × k128 | 0.2599 (0.0010) | 0.5991 (-0.0051) | 0.1904 (-0.0013) | 1 | 2.9304 (-0.0139) | 0.7199 (-0.0033) | 0.1749 (0.0032) | -0.5290 (-0.0109) | 0.7101 (-0.0201) | 1.0000 | 1.0000 | 128–138 |
| D2 c125 × k128 | 0.2765 (0.0176) | 0.6042 (0.0001) | 0.2145 (0.0228) | 2 | 2.8961 (-0.0482) | 0.7339 (0.0107) | 0.1756 (0.0039) | -0.5040 (0.0141) | 0.7088 (-0.0213) | 1.0000 | 0.9922 | 128–251 |

### 4) width=64 与 Stem 变种（Δ 相对各自父配置）

| 配置 | 物理R²_cb（Δ） | 归一化R²_cb（Δ） | 病例均值R²（Δ） | 负例 | MAE（Δ） | Spearman（Δ） | top10 IoU（Δ） | high-WSS R²（Δ） | p99比（Δ） |
|---|---|---|---|---|---|---|---|---|---|
| Q1V ball16 w64（复用） | 0.2600 (0.0000) | 0.5932 (0.0000) | 0.1891 (0.0000) | 3 | 2.9517 (0.0000) | 0.7258 (0.0000) | 0.1812 (0.0000) | -0.5276 (0.0000) | 0.6939 (0.0000) |
| Q1V KNN-8-cover w32（复用） | 0.2589 (0.0000) | 0.6042 (0.0000) | 0.1917 (0.0000) | 1 | 2.9443 (0.0000) | 0.7232 (0.0000) | 0.1717 (0.0000) | -0.5181 (0.0000) | 0.7302 (0.0000) |
| Q1V KNN-10-cover w32（复用） | 0.2425 (0.0000) | 0.5952 (0.0000) | 0.1739 (0.0000) | 1 | 2.9511 (0.0000) | 0.7276 (0.0000) | 0.1903 (0.0000) | -0.5438 (0.0000) | 0.6911 (0.0000) |
| D4 KNN-8-cover w64 | 0.2183 (-0.0406) | 0.5760 (-0.0281) | 0.1646 (-0.0271) | 3 | 2.9998 (0.0555) | 0.7070 (-0.0162) | 0.1709 (-0.0008) | -0.6143 (-0.0963) | 0.6500 (-0.0801) |
| D4 KNN-10-cover w64 | 0.2172 (-0.0253) | 0.5791 (-0.0161) | 0.1567 (-0.0172) | 4 | 3.0033 (0.0521) | 0.7085 (-0.0192) | 0.1744 (-0.0159) | -0.5919 (-0.0482) | 0.6814 (-0.0097) |
| D5-A stem 6→32→64（w64 KNN-8-cover） | 0.2316 (0.0133) | 0.5941 (0.0181) | 0.1739 (0.0093) | 3 | 2.9704 (-0.0294) | 0.7167 (0.0096) | 0.1791 (0.0082) | -0.5622 (0.0521) | 0.6586 (0.0086) |

**D5-B 触发判定**：D5-A 物理 R²_cb=0.2316 vs 父 d4_knn8_cover_w64 0.2183（Δ=+0.0133）；final train_loss 0.1605 vs 0.1613。满足主指标不劣条件，若判读认为仍有欠拟合余量，可用 prepare/submit --variant-b 提交 D5-B（stem 6→256→512→64）。

**判读与下一步**：①点数标度全线未超锚点——全点/10k + 大 k 的 8 个臂全部低于 random5000+ball16（0.2763），bridge 表明 5000 点下 k64 本身就 -0.0324，点数放大到 10k/全点没有补回该损失；"用全部原始 CFD 点"在当前 500/125/32 center 协议下不成立。家族内 k256 一致优于 k64/k128（D3、D1-fixed 同趋势），但都不及小邻域基线。②比例 center 全败：三个 k 全部低于同 k 的固定 center（k64/k128/k256 分别 -0.0328/-0.0173/-0.0448），负例也更多（含 batch2 混杂），不支持"跨队列 center 密度一致"假设，方向关闭。③**降 center × 大邻域是本轮唯一正向家族**：c125×k128=0.2765（+0.0176 vs KNN-8-cover），与 Q1V 锚点打平（+0.0002），high-WSS R²=-0.504 为全轮最好，且趋势单调——同 k 下 center 500→250→125 递增、同 center 下 k64→k128 递增；等预算对角（250×64 vs 125×128）由"更少 center + 更大邻域"一侧胜出。建议下一轮沿此方向延伸（c125×k256、c64×k128/k256）并将 c125×k128 列为 3-seed/独立确认候选。④w64 在 cover 分组下显著回退（KNN-8/10-cover 从 0.2589/0.2425 掉到 0.2183/0.2172），比 ball16 的 w64 效应（-0.016）严重得多；D5-A 瓶颈 Stem 相对标准 w64 Stem +0.0133、high-WSS +0.052，但绝对值仍低于一切 w32 基线，且两者 final train_loss 几乎相同——"容量不足"证据弱，**建议不自动提交 D5-B**，与导师确认后再定。另注意 bridge 的归一化 R²_cb=0.6139 为本轮最高，物理/归一化排名分裂的既有模式延续，主指标仍按预注册的物理 R²_cb。

真源：`training_wss_min/preflight/pointnetpp_sa1_scale_results_analysis.json`、`pointnetpp_sa1_scale_results_summary.csv`、`pointnetpp_sa1_scale_per_case_deltas.csv`。

## Q2V-pool2025 归一化口径×尾部损失（2026-07-20｜`10711_[0-4]` 5/5完成｜A1b 单seed候选）

共同协议：复用 `q2v_ilo_d3_pool2025_refit` 的 pool2025 `138/0/36`、PointNet++ SA3、vertex random5000 SEP、`5000→500→125→32`、400 epoch、train-loss 选模、seed1234、`legacy_vertex`。A0 是既有 point-pooled 统计 checkpoint（`10487_5`，未重训）；`10711_[0-4]` 只提交 A1–A4。5个 GPU 任务均完成400 epoch，best/last test36、逐例 CSV、配置 SHA 和数值有限性均完整；下表为预注册 `ckpt_best(train_loss)` 的物理空间结果。

| arm / Job | 唯一变化（相对 A0/A1） | R²_cb / pooled / case mean | 负例 | MAE / RMSE (Pa) | Spearman / IoU | high-WSS R² / p99 比 | 判读 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A0 / `10487_5` | point-pooled log-z（复用） | 0.2459 / 0.2176 / 0.1974 | 4 | 2.727 / 7.058 | 0.6833 / 0.1697 | -0.523 / 0.334 | 历史同协议对照 |
| A1 / `10711_0` | 逐病例等权 log-z + MSE | 0.2449 / 0.2168 / 0.1858 | 7 | 2.778 / 7.061 | 0.6759 / **0.1754** | -0.499 / 0.359 | 主 R² 不增、病例护栏变差；不晋级 |
| **A1b / `10711_1`** | **逐父系等权 log-z + MSE** | **0.2753 / 0.2411 / 0.2119** | 5 | 2.731 / **6.951** | 0.6795 / 0.1740 | **-0.457 / 0.381** | **主 R²、RMSE 与高尾护栏同步最好；仅保留为确认候选** |
| A2 / `10711_2` | A1 + cohort one-hot | 0.2461 / 0.2240 / 0.1795 | 7 | 2.752 / 7.029 | 0.6714 / 0.1507 | -0.493 / 0.361 | 不优于 A1；关闭 cohort 特征单臂 |
| A3 / `10711_3` | A1 + raw-Huber λ=0.2 | 0.2347 / 0.2023 / 0.1879 | 6 | 2.754 / 7.126 | 0.6811 / 0.1655 | -0.539 / 0.340 | 相对 A1 主 R²/误差/高尾回退；关闭 |
| A4 / `10711_4` | A1 + cohort + raw-Huber | 0.2491 / 0.2229 / 0.1603 | 6 | 2.775 / 7.033 | 0.6590 / 0.1485 | -0.492 / 0.323 | 交互未补回，case mean/IoU 最差；关闭 |

结论：point-pooled→逐病例等权本身没有净收益；进一步改为**逐父系等权**后取得单 seed 的一致正向信号（`Δphysical R²_cb=+0.0295`，`ΔRMSE=-0.107 Pa`，p99 比 `+0.047`），但负例 `+1` 且 test36 已是历史开发集，不能作为独立泛化结论。后续只允许 A1b 进入独立 split/3-seed 确认；cohort one-hot、raw-Huber λ=0.2 与交互均停止扩展。

## QAD 精确Q2V协议三种子对照（2026-07-19｜Jobs `10549→10550` 5/5完成｜**No-Go**）

目标是在完全相同的Q2V-10477历史协议下回答QAD decoder是否有效：`106/0/27`、train106 global stats、vertex random5000 / SEP、FPS `500→125→32`、radius `0.05/0.10/0.20`、`nsample=16`、`width=32`、400 epoch、train-loss best、legacy-vertex。test27已是历史开发集，所以本轮只是同协议比较，不是独立确认。

| seed | R0 / QAD normalized R²_cb | Δnormalized | R0 / QAD physical R²_cb | Δphysical | ΔSpearman | ΔIoU | Δnormalized p99比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1234 | 0.62097 / 0.59873 | **-0.02225** | 0.27243 / 0.22807 | -0.04436 | -0.00862 | +0.02331 | -0.06431 |
| 7 | 0.59272 / 0.60015 | +0.00743 | 0.28033 / 0.26645 | -0.01388 | -0.00191 | +0.01637 | +0.04497 |
| 2025 | 0.60092 / 0.60147 | +0.00055 | 0.26170 / 0.26688 | +0.00518 | +0.00021 | -0.00357 | -0.00088 |
| **3seed均值** | **0.60487 / 0.60012** | **-0.00476** | **0.27149 / 0.25380** | **-0.01769** | **-0.00344** | **+0.01204** | **-0.00674** |

GPU预检 `10549` 与数组 `10550_[0-4]` 全部 `COMPLETED (0:0)`；5个新run均400 epoch完整，best/last、27例CSV、配置哈希和数值有限性检查均通过。QAD将normalized `R²_cb` 的种子标准差从 `0.01187` 压到 `0.00112`，但是收敛到更低的三种子均值；这是“方差下降、偏差增大”，不是精度增益。物理MAE均值仅 `-0.00385 Pa`，RMSE反而 `+0.01705 Pa`，seed1234的RMSE恶化约`3.02%`；high-WSS `R²` 均值仍为负且由 `-0.5269` 降至 `-0.5361`。

逐例先在seed内配对、再对seed取平均：normalized R²差均值 `-0.00630`，bootstrap 95% CI `[-0.02196,+0.00969]`，8胜19负，Wilcoxon `p=0.141`；normalized p99比差均值 `-0.02136`，95% CI `[-0.03672,-0.00333]`，5胜22负，`p=0.00102`，是最一致的负向尾部证据。IoU均值 `+0.01204`且主要2/3 seed改善，但不足以抵消normalized/物理R²、RMSE与p99护栏的失败。`ckpt_last` 会缩小差异，但不逆转R²结论，也不改写train-loss选模规则。

**决策**：精确Q2V协议下QAD判为 **No-Go**，停止QAD-REG/更小gate和以QAD为父实验的R2-DUAL。用户重点关注的历史test27 normalized `R²_cb` 最高仍是Q2V-10477/seed1234的 `0.62097`；同seed QAD为 `0.59873`，三种子QAD均值为 `0.60012`。由于test27已多次参与历史选择，这是同协议工程排除证据，不是新的独立确认。结构化真源：[analysis JSON](../../training_wss_min/preflight/pointnetpp_qad_q2v_test27_results_analysis.json)、[summary CSV](../../training_wss_min/preflight/pointnetpp_qad_q2v_test27_results_summary.csv)、[seed deltas](../../training_wss_min/preflight/pointnetpp_qad_q2v_test27_seed_deltas.csv)、[per-case deltas](../../training_wss_min/preflight/pointnetpp_qad_q2v_test27_per_case_deltas.csv)。

## PointNet++ R1 QAD-Lite（2026-07-19｜Job `10540` 5/5完成｜**No-Go for direct R2**）

本轮在统一 `dev85/val21`、`random5000 SEP`、`5000→500→125→32`、`n32_w32`、global log-z stats、MSE/400 epoch/train-loss best、legacy-vertex 协议下，严格配对 `interpolate` 与 QAD-Lite。五个新任务均 `COMPLETED (0:0)`；六个 run（含复用的 `10488_2`）均有400行history、best/last checkpoint、val21指标和21例逐例CSV，配置哈希无漂移、数值无NaN/Inf，原 test27 未访问。

| seed | R0 normalized R²_cb | R1 normalized R²_cb | Δnormalized R²_cb | Δphysical R²_cb | ΔSpearman | Δtop10 IoU | Δnormalized p99比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1234 | 0.52937 | 0.56612 | **+0.03676** | -0.00828 | +0.02476 | +0.01012 | -0.03166 |
| 7 | 0.53601 | 0.55877 | **+0.02277** | +0.01959 | +0.00970 | -0.00562 | +0.01275 |
| 2025 | 0.54864 | 0.53189 | **-0.01675** | -0.01304 | -0.01384 | -0.00448 | -0.02811 |
| **3seed均值** | **0.53800** | **0.55226** | **+0.01426** | **-0.00058** | **+0.00688** | **+0.00001** | **-0.01567** |

主判据有正向信号，但3seed均值增益 `+0.01426` 低于预注册 `+0.015`，尽管满足2/3 seed为正，主门仍未通过。物理 `R²_cb` 均值基本不变（`0.26490→0.26432`）；MAE小幅改善 `-0.0187 Pa`，RMSE平均几乎持平，但seed2025恶化约`2.14%`。Spearman非劣门、RMSE非劣门和p99高尾方向门均失败；IoU非劣门通过但均值无改善。high-WSS `R²` 仍为负且均值 `-0.6030→-0.6261`。

三种子逐例差先在seed内配对、再对seed平均：normalized R²逐例均差 `+0.01107`，bootstrap 95% CI `[-0.00176,+0.02516]`，21例中13胜8负，Wilcoxon `p=0.179`，不确定性仍跨零。探索性分组信号主要集中在AAA rupture（三种子均值约`+0.0446`），但seed2025同样反转，不可作为确证性结论。best/last的normalized `R²_cb` 差约在±0.0035内，不改变决策。

**当时决策与后续更新**：val21结果当时只支持“不直接进R2”，并曾保留QAD-REG/更小gate作为候选。本页顶部精确Q2V三种子复核已用更接近历史锚点的协议将该候选更新为 **No-Go**：不再执行QAD-REG/小gate，R2-DUAL不继承QAD。本节保留为val21阶段证据，不代表当前待执行计划。结构化真源：[analysis JSON](../../training_wss_min/preflight/pointnetpp_qad_results_analysis.json)、[summary CSV](../../training_wss_min/preflight/pointnetpp_qad_results_summary.csv)、[seed deltas](../../training_wss_min/preflight/pointnetpp_qad_seed_deltas.csv)、[per-case deltas](../../training_wss_min/preflight/pointnetpp_qad_per_case_deltas.csv)。

<!-- Q2V_ILO_20260718_START -->
## Q2V 数据扩容与 Point++ 架构结果（2026-07-18｜定量12/12完成）

共同设置冻结为 Q2V-10477：Point++、vertex random5000 SEP、完整 support→SA center 链 `5000→500→125→32`、radius `0.05/0.10/0.20`、MSE、400 epoch、train-loss选模、seed1234、`legacy_vertex`。其中 `500/125/32` 仅表示三层 SA center 数，不表示输入只有500点；所有主表取预注册 `ckpt_best(train_loss)`，last只作敏感性审计，不按评估结果改选checkpoint。

| 数据实验 / Job | test | 物理 R²_cb / pooled / case mean | 负例 | MAE / RMSE (Pa) | Spearman / IoU | high-WSS R² | 流程状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Q2V-10477 / `10477` | 27 | 0.2724 / 0.2647 / 0.2157 | 2 | 2.562 / 6.230 | 0.7358 / 0.1568 | -0.521 | export-only `10489` 后 PostView 27/27 |
| D1 fixed frozen / `10487_0` | 27 | 0.2516 / 0.2506 / 0.1881 | 2 | 2.603 / 6.290 | 0.7128 / 0.1756 | -0.549 | ✅ 定量 + PostView 27/27 |
| D1 fixed refit / `10487_1` | 27 | 0.2543 / 0.2486 / 0.1846 | 2 | 2.583 / 6.298 | 0.7237 / 0.1756 | -0.579 | ✅ 定量 + PostView 27/27 |
| Q2V zero-shot36 / `10490` | 36 | 0.2646 / 0.2561 / 0.2138 | 2 | 2.959 / 7.620 | 0.7182 / 0.1727 | -0.452 | ✅ 只读评估完成 |
| D2 extended frozen / `10487_2` | 36 | 0.2493 / 0.2419 / 0.2026 | 1 | 2.967 / 7.693 | 0.7156 / 0.1860 | -0.485 | ✅ 定量 + PostView 36/36 |
| D3 pool2025 control / `10487_3` | 36 | **0.2981** / **0.2820** / 0.2131 | 7 | 2.694 / **6.761** | 0.6800 / 0.1660 | **-0.368** | ⚠️ 定量完成；PostView 35/36 |
| D3 pool2025 frozen / `10487_4` | 36 | 0.2915 / 0.2507 / **0.2162** | **3** | **2.692** / 6.907 | **0.6898** / **0.1734** | -0.457 | ⚠️ 定量完成；PostView 35/36 |
| D3 pool2025 refit / `10487_5` | 36 | 0.2459 / 0.2176 / 0.1974 | 4 | 2.727 / 7.058 | 0.6833 / 0.1697 | -0.523 | ⚠️ 定量完成；PostView 35/36 |

配对判读：D1加入ILO41但保持原test27与Q2V统计时，`R²_cb -0.0208`、MAE `+0.041 Pa`；重算train147统计只回补 `+0.0027`，且pooled/case mean/high-WSS方向相反。D2同test36下，加入ILO32训练相对zero-shot `R²_cb -0.0153`、RMSE `+0.073 Pa`，虽负例 `2→1`、IoU `+0.0133`，仍不能判整体改善。D3同test36/控制统计加入ILO32后 `R²_cb -0.0066`、pooled `-0.0313`，负例 `7→3`，但ILO组 `0.3255→0.2842`、high-WSS继续回退；再重算统计 `R²_cb -0.0457`。因此本轮数据扩容结论是“未获得稳健整体增益，局部收益与幅值/高尾代价并存”。ILO-1测试仅3例、ILO-0仅6例，组间差异只作探索性线索。

同一原 test27 的归一化空间对照为：Q2V-10477 `normalized R²_cb=0.6210`、D1 frozen `0.5961`、D1 refit `0.6092`。因此 Q2V 仍是该归一化空间下最好的历史锚点；D1 refit 虽较 frozen 回升，但仍未超过 Q2V。该结论只限相同 test27，不与 D2/D3 的 test36 或架构 val21 横比。

| 架构 / Job（统一val21） | 参数量 | 物理 / 归一化 R²_cb | pooled / case mean（物理） | MAE / RMSE | Spearman / IoU | high-WSS R² | 开发判读 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| n16_w32 / `10488_0` | 0.224M | 0.2456 / 0.5481 | 0.2800 / 0.2042 | 2.384 / 5.479 | 0.6830 / 0.1961 | -0.702 | **Q2V-structure dev control** |
| n16_w64 / `10488_1` | 0.880M | 0.2368 / **0.5709** | 0.2862 / 0.2102 | 2.327 / 5.456 | **0.7036** / 0.2028 | -0.723 | normalized数值第一；物理主指标回退0.0089 |
| n32_w32 / `10488_2` | 0.224M | 0.2589 / 0.5294 | 0.3142 / 0.1952 | 2.346 / 5.348 | 0.6743 / 0.1858 | **-0.593** | 参数效率候选 |
| **n32_w64 / `10488_3`** | 0.880M | **0.2603** / 0.5431 | **0.3164** / **0.2241** | **2.329** / **5.339** | 0.6842 / 0.1985 | -0.635 | **预注册物理主指标第一；仅领先n32_w32 0.0014** |
| n64_w32 / `10488_4` | 0.224M | 0.2327 / 0.5506 | 0.2889 / 0.1941 | 2.364 / 5.445 | 0.6904 / 0.1886 | -0.672 | n64回退 |
| n64_w64 / `10488_5` | 0.880M | 0.2436 / 0.5540 | 0.3024 / 0.2225 | 2.337 / 5.393 | 0.6906 / **0.2021** | -0.661 | 扩宽恢复但物理仍低于n32 |

预注册架构主指标是物理 `R²_cb`：两种width均由 `nsample=32` 取得最高值，`n32_w64=0.2603` 数值第一。normalized val21 的数值第一则是 `n16_w64=0.5709`；两种空间排名不同，不能事后把 normalized 排名替换为主判据。width64的物理作用随nsample交互（n16 `-0.0089`、n32 `+0.0014`、n64 `+0.0108`），不能概括为“扩宽必然有效”。`n32_w64` 单seed、val21、小幅领先且参数约4倍，只作为独立重复/确认候选。

`n16_w32` 与 Q2V 的模型、采样、support→center 链和训练预算一致，仅因开发协议改为 dev85/val21 并使用 train85 统计而重训，所以它是可比较的 **Q2V-structure dev control**。原 Q2V checkpoint 不得在 val21 排名：`dev_train85∪dev_val21=Q2V train106`，21个val病例全部参与过原 Q2V 训练，直接评估会形成 `21/21` 泄漏。Q2V 原 test27 normalized `R²_cb=0.6210` 只作为历史锚点展示，与 val21 的不同 split/stats 不可直接排名。原test27从未被六个开发任务访问，确认任务尚未提交。

完整性方面，12个run均有400行history、best/last checkpoint和metrics，提交配置哈希一致、无NaN/Inf。数据任务best/last `R²_cb` 差在 `-0.0201～+0.0182`，不改变预注册选模规则。`10487_3–5` 的Slurm失败均发生于定量评估之后：同一 `ILO/YU_XIANG_SHENG-1/before` 可视化STL只有 `7193/26942=26.70%` 顶点在3 mm Gaussian支持域内，其余35例为100%；先做几何坐标域/裁剪血缘审计，不放宽门禁、不重训。AreaRandom六组继续冻结。
<!-- Q2V_ILO_20260718_END -->

## Q2V/test27 半径与采样探索（2026-07-18｜4/4完成｜仅历史锚定）

本节是用户明确授权的**历史 test27 探索**：所有新任务仍用 `106/0/27`、`5000 support→500→125→32`、MSE、400 epoch、seed1234、legacy-vertex 和 `ckpt_best(train_loss)`。因此可以用于筛掉不值得确认的方向，但不能当作新的独立 test27 确认；`last` 只做敏感性审计。

| 任务 / Job | 相对历史控制的唯一变化 | 物理 R²_cb / pooled / case mean | MAE / RMSE (Pa) | high-WSS R² / IoU | PostView | 判读 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Q2V-10477（历史） | vertex random5000 SEP；r=0.05/0.10/0.20 | 0.2724 / 0.2647 / 0.2157 | **2.562 / 6.230** | -0.521 / 0.1568 | 27/27 | Q2V半径对照 |
| `q2v_radius_r80_sep` / `10506_0` | **仅**r→0.04/0.08/0.16 | 0.2433 / 0.2419 / 0.1760 | 2.655 / 6.326 | -0.505 / 0.1596 | 27/27 | 对Q2V `ΔR²_cb=-0.0291`；不晋级 |
| `q2v_radius_r60_sep` / `10506_1` | **仅**r→0.03/0.06/0.12 | 0.2406 / 0.2242 / 0.1808 | 2.665 / 6.400 | -0.590 / 0.1647 | 27/27 | 对Q2V `Δ=-0.0319`；进一步回退 |
| Q1V-10476（历史） | vertex random5000 SAME；r=0.05/0.10/0.20 | **0.2763 / 0.2623 / 0.2114** | 2.588 / 6.240 | **-0.513 / 0.1803** | 27/27 | Q1V采样对照 / 当前优先候选 |
| `q1v_fpsmultistart5000_same` / `10506_2` | **仅**vertex-random→fps_multistart5000 | 0.2297 / 0.2264 / 0.1752 | 2.630 / 6.390 | -0.601 / 0.1760 | 27/27 | 对Q1V `Δ=-0.0466`；不支持替换 |
| `q2v_to_q1v_same_finetune` / `10506_3` | Q2V best→Q1V SAME，权重warm-start、优化器重置 | 0.2670 / **0.2646** / 0.1982 | 2.571 / 6.231 | -0.523 / 0.1755 | 27/27 | 对Q1V `Δ=-0.0093`；非纯采样对照且无稳定增益 |

四个run都完成400个epoch，submission config哈希未漂移，数值无NaN/Inf，best/last均齐全；最后checkpoint相对best的 `R²_cb` 改变量为 `+0.0016/-0.0041/-0.0021/+0.0004`，不改变train-loss选模。r80的high-WSS R²局部略好于Q2V，但主R²、误差和病例表现同时回退，不能以单指标选择。所有新臂的high-WSS R²仍为负。本轮结论为：保留Q1V作为独立重复候选，停止缩半径、fps_multistart5000与该warm-start路线；AreaRandom不在本矩阵内，继续冻结。

结构化真源：[分析 JSON](../../training_wss_min/preflight/q2v_sampling_radius_test27_results_analysis.json) 与 [汇总 CSV](../../training_wss_min/preflight/q2v_sampling_radius_test27_results_summary.csv)。

## Phase-V 六组完训回填（2026-07-18｜定量完成｜五组 PostView 待 export-only）

- `10473–10478` 都完成400 epoch、best/last test27 legacy-vertex 评估；主结果固定为 `ckpt_best(train_loss)`，不因 test 结果改选 last。Q0=`10475` 在 `2026-07-17 22:53:55+08:00` 完成，并已通过 PostView `27/27` 验证。
- P1V/P2V/Q1V/Q2V/Q3V 的训练和两套评估指标同样完整；它们在旧 exporter 的 `AAA/ruputer/SHI_YUN_XI` 严格面积调用处停止，当前各留20个病例目录但无批次 verification。该错误发生在定量评估之后，五个模型无需重训，只需按 `legacy_vertex` 修复路径 export-only 补齐。

| ID / Job | 物理 R²_cb | case mean / 负例 | MAE / RMSE (Pa) | Spearman / IoU | high-WSS R² | 状态判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| P1V / 10473 | 0.1592 | 0.1186 / 8 | 2.678 / 6.599 | 0.6914 / 0.1546 | -0.737 | 相对 B0 整体改善；P1V 为桥接组 |
| **P2V / 10474** | **0.2193** | **0.1805 / 1** | 2.606 / 6.463 | 0.7161 / 0.1607 | -0.656 | **PointNet 分支优先候选** |
| Q0 / 10475 | 0.2212 | 0.1657 / 4 | 2.637 / 6.407 | 0.7112 / 0.1645 | -0.615 | fixed-center/full-query 桥接，PostView 27/27 |
| **Q1V / 10476** | **0.2763** | 0.2114 / **1** | 2.588 / 6.240 | 0.7222 / **0.1803** | **-0.513** | **PointNet++ 分支优先候选** |
| Q2V / 10477 | 0.2724 | **0.2157 / 2** | **2.562 / 6.230** | **0.7358** / 0.1568 | -0.521 | SEP 结果混合，不判优于Q1V |
| Q3V / 10478 | 0.2521 | 0.1852 / 2 | 2.634 / 6.357 | 0.7107 / 0.1595 | -0.579 | Random center 整体回退，停止扩展 |

完整配对差、口径边界和下一步顺序见 [PointNet baseline 矩阵 §0C](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#0c-phase-v-完训结果与判读2026-07-18test27-legacy-vertex)。面积六组仍严格冻结；不得将上述 legacy-vertex 指标写作 area-weighted 结论。

## 面积口径合同与 Phase-V PostView 修复（2026-07-17｜无需重训）

- 实验矩阵现固定为两条可独立解释的路线：Phase-V 使用 `legacy_vertex`，训练/评估/PostView 均不消费 STL 面积；Phase-A 使用 `both_strict + area_random`，只有严格面积映射 133/133 通过后才能提交。面积加权是物理表面积口径的必要条件，但不是所有点云实验的通用前置条件。
- 代码复核发现旧 `export_wss_postview.py` 在所有模式下都无条件调用 `area_weights_for_case`。因此 Phase-V 的 `10473/10474/10476/10477/10478` 实际均已完成400 epoch训练及 best/last vertex eval，随后才在 test 例 `AAA/ruputer/SHI_YUN_XI` 的 PostView 面积调用处退出；Slurm `FAILED (1:0)` 不代表模型或已有指标失效，五组 checkpoint/metrics 无需重训，只需后续 export-only 补齐可视化。`10478` 的 exporter 在修复前已启动并载入旧模块，运行中不会热更新。
- 已修复 PostView：`legacy_vertex` 只输出逐点/vertex-top10 结果且 manifest 写明 `highrisk_mask_basis=legacy_vertex`、`surface_area_metrics_status=not_requested`；`both_strict` 才加载面积并输出 area 字段。resume 还会拒绝复用旧的无口径/错口径半成品包，避免同一 PostView 混合 vertex 与 area 标量。
- 同时修复一个采样门禁边界：当 `query_mode=same` 时，未实际使用的 `query_sampling=area_random` 不再触发面积加载；FPS pool 预热使用有效的 `support_sampling`。相关 Support/Query 与 PostView 回归共28项通过。
- `10475/Q0` 已于 `2026-07-17 22:53:55+08:00` 完训并完成 PostView `27/27` 验证；其余五组的定量结果已于2026-07-18回填，PostView 仍只需 export-only。Phase-A 六组继续冻结，修复六个失败病例并重跑严格 preflight 前不得提交。

判读时务必区分：PostView Gaussian `mapping coverage=100%` 只保证可视化 STL 顶点能找到邻近 CFD 点，不代表原始 STL 三角面积已严格转移成功。面积失败本身也不证明 bundle 中的中心线派生特征错误；四个 crop 例主要是支撑域不同，`ZHOU_KE_XUN` 需审计坐标变换，`LIU_WEN_QI` 需核查数据血缘。只有确认原始几何配错或重发布后，才重提中心线/几何特征。详细合同与后续采样顺序见[PointNet 矩阵 §0B](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#0b-面积依赖合同与判读边界后续实验按此执行)。

## Job 9170 test27 结果（2026-07-17｜400 epoch完成｜正式面积eval阻断｜诊断结果已回填）

- Slurm `9170` 最终为 `FAILED (1:0)`、用时 `06:45:54`；这不是训练失败。run 内400行 history、best/last checkpoint、运行时 config/stats 均完整，`ckpt_best` 为 epoch397、train loss `0.139726`。训练结束后的正式 evaluate 在首个面积失败 train 例 `AG/fast/LI_ZHEN_SHAN` 按硬门退出。
- 保持正式 evaluate 默认严格，另用诊断入口对 best/last 做 test27：全27只报告 legacy vertex 指标；面积只报告映射通过的26例，并显式排除 `AAA/ruputer/SHI_YUN_XI`。9170 best 的物理/归一化 `R²_cb=0.0530/0.4569`、pooled物理 `R²=0.0831`、case mean/median=`0.0405/0.0221`、负例12/27、MAE/RMSE=`2.919/6.957 Pa`、Spearman=`0.6512`、legacy IoU=`0.1727`、high-WSS R²=`-0.8620`。
- 同 split/stats 的9169为 `0.1331/0.5307`、pooled物理 `0.1371`、case mean/median=`0.0861/0.0785`、负例6/27、`2.736/6.749 Pa`、`0.6959`、`0.1846`、`-0.8004`。9170逐例10例改善、17例退化；只在峰值点距离/bbox更小。当前冻结配置下 SA3 胜出，但参数量0.22M vs0.79M，仍不是容量配平架构终裁。
- 9170 last−best 的物理 `R²_cb=+0.0009`，不改 train-loss 选出的 best。26例有效面积子集 physical/normalized area `R²_cb=0.1559/0.5243`、area IoU=`0.2564`；因为不是27例且9169无同口径面积结果，只作数据修复前诊断。
- 指标真源：`runs/pointnet_v4/outputs/ag_aaa_v4_stratified_e2_global_fps2000/eval_diagnostic/`；汇总 xlsx 已回填并渲染检查。9170 标准 `eval/` 与 PostView 仍缺失。

## test27 Support/Query 两阶段矩阵（2026-07-18｜Phase-V 定量完成/五组导出待补｜Phase-A 6组待修）

- 已实现 `support_n_points/support_sampling/query_mode/query_n_points/query_sampling` 与 fixed-support/full-query eval；缺省仍复现旧 `wall_n_points/sampling + SAME`。PointNet SAME 数值等价；PointNet++ support 每次完整预测只编码一次，固定中心500/125/32，支持 FPS/按 seed+epoch+case+stage 派生的 Random center。
- 评估门禁改为显式两态：旧配置默认 `legacy_vertex`，不依赖 STL 面积且不伪造 area 指标；面积配置必须显式 `both_strict`，对133例严格 mapping，禁止均匀权重回退。这是协议分层，不是把 AreaRandom 静默换成 vertex random。
- **Phase-V 已完训并回填**：P1V=`10473`、P2V=`10474`、Q0=`10475`、Q1V=`10476`、Q2V=`10477`、Q3V=`10478` 均完成训练和 best/last vertex eval。Q0通过PostView `27/27`；另五组均在旧 PostView 隐式面积调用失败，留下20个病例目录，待 export-only 补齐。P2V/Q1V 分别是两条分支的单次优先候选；Q2V结果混合、Q3V回退。
- 正式 preflight 6/6通过：split106/0/27与strata、九例排除、train-only stats和全部bundle SHA、133/133 frame/load、SAME/SEP与epoch重采样、双病例CPU、batch8 RTX4090 AMP均通过。PointNet峰值约559 MiB；PointNet++约127–181 MiB，无OOM；中心逐病例精确500/125/32；full-query/chunk最大差`7.45e-8`，support encoder调用符合一次/完整预测。
- **Phase-A 保留未提交**：P1/P2/Q1/Q2/Q3/Q4 共6组；面积审计仍为127/133，失败六例及可视化/审计哈希写入 `training_wss_min/preflight/ag_aaa_v4_area_phase_backlog.json`。修复后必须重跑严格面积 preflight，不能沿用本次 vertex preflight 放行。
- Job `9170` 没有重复提交。Job `9169` 未重训：export-only `9818` 已 `COMPLETED (0:0)`；最终27/27每例4 VTP/3 PNG/3 CSV、mapping coverage 100%，checkpoint、best/last metrics 与 per-case CSV 前后 SHA完全一致。

| ID | Job | 模型/Support/Query/center | 严格对照 | 单一变化或桥接变化 | 提交时状态 |
| --- | ---: | --- | --- | --- | --- |
| P1V | 10473 | PointNet；vertex random5000；SAME | B0/9170 | FPS2000固定→每epoch random5000（采样协议+点数） | best R²_cb=0.1592；PostView export-only待补 |
| P2V | 10474 | PointNet；vertex random5000；SEP | P1V | SAME→SEP | best R²_cb=**0.2193**；PointNet优先候选；PostView待补 |
| Q0 | 10475 | PointNet++；FPS2000；SAME；FPS center | B1/9169 | legacy ratio/eval→固定中心与fixed-support/full-query桥接 | best R²_cb=0.2212；PostView **27/27** |
| Q1V | 10476 | PointNet++；vertex random5000；SAME；FPS center | Q0 | FPS2000固定→每epoch random5000（采样协议+点数） | best R²_cb=**0.2763**；PointNet++优先候选；PostView待补 |
| Q2V | 10477 | PointNet++；vertex random5000；SEP；FPS center | Q1V | SAME→SEP | best R²_cb=0.2724；混合，不判优于Q1V；PostView待补 |
| Q3V | 10478 | PointNet++；vertex random5000；SAME；Random center | Q1V | FPS center→Random center | best R²_cb=0.2521；回退；PostView待补 |

## v4 PointNet++ 三协议结果（2026-07-16/17｜`9167–9169` 全流程闭环）

- 三组均完整400 epoch；当前 PointNet++ 为 0.22M 参数的三层 SA（`2000→500→125→32`，radius `0.05/0.10/0.20`，nsample16，FP k=3），统一 xyzgeom / GLOBAL log-z / FPS-2000 / MSE / seed1234 / train-loss 选模。
- `9167`：AG `61/0/15`，best epoch 364 / train loss `0.169471`；best/last eval 齐全，PostView `15/15`。
- `9168`：锁定 AG test15 的混合 `118/0/15`，best epoch 394 / train loss `0.139474`；best/last eval 齐全，PostView `15/15`。
- `9169`：新分层 `106/0/27`，best epoch 340 / train loss `0.142360`；best/last eval 齐全。原 AAA bundle 路径错解由 export-only Job `9818` 修复，现 PostView `27/27`，checkpoint/metrics 哈希未改变。

| common-test15 协议 | 模型 | 物理 `R²_cb` | 归一化 `R²_cb` | MAE / RMSE | Spearman | top10 幅值比 / IoU | high-WSS R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AG `61/0/15` | PointNet E2 | **0.1866** | **0.4676** | **2.847 / 5.143** | **0.6788** | **0.3508** / 0.1314 | **-1.8279** |
| AG `61/0/15` | PointNet++ SA3 | 0.1500 | 0.4463 | 2.867 / 5.264 | 0.6598 | 0.3078 / **0.1451** | -2.0479 |
| 混合 `118/0/15` | PointNet E2 | **0.2054** | **0.4906** | **2.772 / 5.068** | **0.7041** | **0.3405** / 0.1575 | **-1.8473** |
| 混合 `118/0/15` | PointNet++ SA3 | 0.1490 | 0.4726 | 2.829 / 5.261 | 0.6779 | 0.2858 / **0.1633** | -2.1388 |

**判读**：common-test15 上当前 SA3 配置在两个同病例协议都未超过 E2，只有 top10 IoU 小幅改善，仍判该分支 **No-Go**。独立 test27 则出现相反结果：同 split 的9169 SA3在主要 legacy vertex 指标上超过9170 E2，说明结论依赖数据划分/域组成，不能从 common-test15 外推。SA3 只有 E2 约28%参数，且 sampled-train/full-cloud 仍可能有密度落差，所以两组结果都不写成“PointNet++ 架构终裁”。三组 SA3 与9170的 high-WSS R² 仍全为负，高尾幅值恢复未解决。完整表见[PointNet 矩阵 §0.1](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#01-第三混合-split-与-pointnet-结果)。

## PointNet E4 deeper 追加探针与三层 SA foundation（2026-07-15 ✅完训完评｜No-Go）

- `E4-DEEP-GLOBAL` 严格复用 E2-GLOBAL 的 train61 / 固定 FPS-2000 / global log-z / 400 epoch / train-loss 选模协议，仅将网络改为 `6→64→128→256→512；1024→512→256→128→64→1`；参数量 `874,561`，相对 E2 增加 10.3%。
- Job `8999` 已 `COMPLETED (0:0)`，用时 `01:37:11`；400 epoch 完整，best=第 379 epoch / train loss `0.121590`，比 E2 的 `0.172548` 降低 29.5%。best/last train61+test16 全点评估齐全，best 的 16/16 PostView 包完整，mapping coverage 100%，无 Traceback/OOM/NaN。
- PointNet++ fine-tune 本轮未启动。前置三层 SA 结构审计已完成：FPS-2000 后中心数 `500/125/32`，radius `0.05/0.10/0.20`，nsample=16；`fast/RAN_QING_BO` 的覆盖率 `99.90%/100%/100%`。完整 PNG、VTP、assignment CSV 与 manifest 位于 `例子/06_PointNet++_SA三层采样与分组/`。

| best 配对指标 | E2 | E4 | E4−E2 |
| --- | ---: | ---: | ---: |
| train / test 物理 `R²_cb` | 0.6124 / **0.2140** | **0.6966** / 0.1629 | +0.0841 / **−0.0511** |
| 物理 train−test gap | 0.3985 | 0.5337 | +0.1352（变差） |
| train / test 归一化 `R²_cb` | 0.7717 / **0.4606** | **0.8375** / 0.4476 | +0.0658 / −0.0130 |
| test 物理 MAE / RMSE | **2.8005 / 5.1141** | 2.8510 / 5.2778 | +0.0505 / +0.1637 |
| test high-WSS R² / top10 幅值比 | **−1.486 / 0.378** | −1.680 / 0.333 | 均变差 |
| test Spearman / top10 IoU | **0.661** / 0.146 | 0.657 / **0.153** | −0.004 / +0.007 |
| test 双 self-max `R²_cb` / 负例 | −4.857 / 16 | **−4.208** / 16 | 小幅改善，仍 No-Go |

**结论**：E4 把 train-fit 做得更好，却使物理/归一化 test R² 下降、gap 扩大、MAE/RMSE 和 high-WSS 幅值恢复变差；16 例中物理 R² 仅 6 例改善、10 例退化。top10 IoU 和 self-max 的小幅好转不足以抵消整体泛化恶化，因此导师追加深度探针判为 **No-Go**，保留 E2 为 PointNet 锚点，不继续纯深度扫描。`ckpt_last` 的 test 物理/归一化 R²=`0.1618/0.4451`，与 best 同结论。test16 已被反复使用，本结果只作为导师驱动的同协议配对证据，不表述为无偏最终测试。完整表见[PointNet 矩阵 §4.4](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#44-导师追加深度探针-e4-deep-global2026-07-15-完训完评no-go)。

## PointNet baseline 新矩阵（2026-07-15 ✅五组完训+完评）

- 当前只保留 `PointNet+xyzgeom`；共同协议为无 val/无早停/400 epoch。
- 正式父实验分为导师通道对齐的容量组和 5000 点 random 不放回/每 epoch 重采样组；两组各做全局与逐病例归一化配对。
- 该矩阵的公式讨论、Job 状态、high-risk 评价与每例可视化产物统一转到[PointNet baseline 实验矩阵与进度跟踪](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)；本文仅保留已完成的原 baseline 数值与其他路线证据。
- **2026-07-14 实现回填**：`N-CASE` 冻结为 `WSS/WSSmax`；E2/E3 的 GLOBAL/CASE 四份主配置、`E23-GLOBAL`（精确导师宽网 + random-5000）交互配置、random-5000 协议校验、best/last 隔离评估和 test16 STL/VTP 流水线均已写入。
- 训练阶段新增 sampled normalized MSE/MAE/RMSE；完整点云 R²、Spearman、top10 与热点位置仍由训练后的 eval 计算。`ckpt_best(train_loss)` 为主报告和 PostView，`ckpt_last` 只作指标审计。
- **Slurm 结案（2026-07-15）**：`8976–8980` 五个 Job 均 `COMPLETED (0:0)`；5/5 run 的 best/last train61+test16 全点评估齐全，best 共导出 80/80 个 test case PostView 包，surface mapping coverage 为 100%。

| Run | 变量 | test 主空间 `R²_cb` | Spearman | top10 IoU | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `E0-GLOBAL` | 原 PointNet + FPS-2000 | 物理 0.1414 / norm 0.3964 | 0.635 | 0.128 | 共同对照 |
| **`E2-GLOBAL`** | **导师宽网** | **物理 0.2140 / norm 0.4606** | 0.661 | 0.146 | 本轮最强；容量有效，但热点仍 No-Go |
| `E3-GLOBAL` | random-5000 | 物理 0.1637 / norm 0.4218 | 0.645 | 0.124 | 点数/重采样单独增益弱 |
| `E23-GLOBAL` | 宽网 + random-5000 | 物理 0.1988 / norm 0.4598 | **0.674** | 0.148 | 未超过 E2，无明确协同 |
| `E2-CASE` | E2 + `WSS/WSSmax` | norm 0.1724 | 0.574 | 0.140 | **No-Go** |
| `E3-CASE` | E3 + `WSS/WSSmax` | norm 0.1551 | 0.559 | **0.159** | 局部 IoU 改善但整体分布 **No-Go** |

**本轮结论**：导师宽通道是主要有效因素；random-5000 不是稳定主增量，且与加宽没有明确协同。逐病例 `WSS/WSSmax` 使 R²、Spearman、p99 与动态范围明显退化，虽在 E3 上出现局部 hotspot IoU 改善，仍不足以支持替换全局 log-z。E2 的 physical high-WSS R² 仍为 −1.486、top10 幅值比仅 0.378，所以只判为相对改进，不判为可部署 Go。完整配对、best/last 审计和失败病例可视化见[独立矩阵文档 §4.2](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#42-正式矩阵结果2026-07-15)。

**导师 self-max 补充（2026-07-15）**：五组 80 个既有 test PostView 已无推理回填 `WSScfd/WSScfd,max` 与 `WSSpred/WSSpred,max`。五组的 16/16 逐病例 self-max R² 均为负；相对较好的 E2-GLOBAL pooled/case-balanced R² 仍为 −4.802/−4.857，说明去掉绝对幅值后空间型态仍未学准。CASE 线性输出还产生负点：E2/E3-CASE 病例平均占 7.34%/6.03%。定义、完整表、负值解释和 VTP/图件路径见[独立矩阵文档 §4.3](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#43-导师补充指标cfdcfdmax-对-predpredmax2026-07-15)。

**下一轮暂不自动执行**：以 E2-GLOBAL 为锚点，讨论顺序冻结为开发协议（train61 内 group-dev/repeated holdout）→ 推理期可得 BC/病例级信息审计 → shape/scale 拆分与热点 loss → 局部拓扑表示 → 几何分层采样。random-5000、原样 `WSS/WSSmax` 和继续盲目加宽不列为优先项；详见[独立矩阵文档 §9](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#9-下一阶段优化方向待讨论不自动开跑)。

## PointNet 无 val / train_loss 选模 / 400 epoch（2026-07-14 ✅完训+完评｜`8968` / eval `8974`/`8975`）

- 协议：`split_AG_wss_min_v1_traintest`（61/0/16）、train_loss 选模、400 epoch、FPS-2000、seed=1234。
- **主结果 `R²_field_cb`**：

| 臂 | train61 | test16 | gap | test 负例 | test high-WSS R² |
|---|---:|---:|---:|---|---:|
| xyz | 0.5704 | 0.0825 | 0.488 | 5/16 | −1.88 |
| **xyz+geom** | 0.5303 | **0.1414** | 0.389 | 5/16 | −1.76 |

- **结论**：No-Go（泛化）；几何臂 test 略好，保留 xyz+geom；train-fit 有容量信号但选模协议不适合作精度主线。产物：`runs/pointnet_trainloss_e400/outputs/*/eval/metrics.json`。

## PointNet 加宽容量探针（2026-07-14 ✅完训｜Job `8970`｜No-Go）

- 相对 2×3 最佳格只改容量：`PointNet + xyz+geom`，`width=128` / `head_hidden=256`，其余同冻结 val 协议。
- **val8 `R²_field_cb=0.3071`** vs 锚点 **0.3015**（Δ≈+0.006）；RMSE/MAE/热点几乎持平。
- **结论**：单变量加宽无实质增益；下一步按矩阵做老师通道对齐与 5k random，不再扩宽本结构。

## 最小 2×3 baseline 矩阵（2026-07-14 ✅DONE｜Job `8700[0-5]` 全部 `0:0`）

- 重新从最基础形式起步：`MLP / PointNet / PointNet++` × `xyz / xyz+geom`，共 6 个单 seed（1234）作业；`xyz+geom=xyz+abscissa_norm+local_radius+curvature`。
- 统一使用 `split_AG_wss_min_v1` 的 AG 划分（train/val/test=`53/8/16`），只训练和选择 `val`，暂不读取 test；峰值收缩期、WSS 单标量、FPS-2000、完整壁面 val 推理。
- 训练仅用未加权标准化空间 MSE，关闭旋转增强、采样/几何/目标加权、多任务和 raw-space 辅助项；PointNet++ 为经典 SA+FP，无 PointNeXt 残差/倒置瓶颈。
- 实验包位于 `training_wss_min/runs/baseline_2x3_simple/`：`make_configs.py` 固化 6 份配置，`submit.sh` 提交最多 4 并发的 Slurm array，所有产物落在同目录 `outputs/`。`8700[0-5]` 于 2026-07-14 依次完成，6/6 均 `COMPLETED (0:0)`；全程只做 val，`test16` 未读。

| 模型·输入 | `R²_field_cb` | `R²_field_raw` | case mean / median / P10 R² | 负例 | RMSE / MAE (Pa) | high-WSS R² | top10 比 / IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP · xyz | 0.1980 | 0.2043 | 0.0080 / 0.1114 / −0.3414 | 1/8 | 4.424 / 2.615 | −1.3788 | 0.361 / 0.189 |
| MLP · xyz+geom | 0.2442 | 0.2632 | 0.0669 / 0.0496 / −0.2142 | 3/8 | 4.257 / 2.476 | −1.0972 | 0.447 / 0.182 |
| PointNet · xyz | 0.1727 | 0.1704 | 0.1194 / 0.0699 / −0.0554 | 2/8 | 4.517 / 2.454 | −1.6736 | 0.313 / 0.150 |
| **PointNet · xyz+geom** | **0.3015** | **0.3122** | **0.1842 / 0.1404 / +0.0143** | **1/8** | 4.113 / 2.291 | −1.1690 | 0.457 / 0.235 |
| PointNet++ · xyz | 0.2169 | 0.2367 | 0.0156 / 0.0694 / −0.3146 | 2/8 | 4.333 / 2.583 | −1.0329 | 0.461 / 0.207 |
| PointNet++ · xyz+geom | 0.2587 | 0.2668 | 0.1287 / 0.1427 / −0.0950 | 1/8 | **4.247 / 2.281** | −1.4213 | 0.387 / 0.226 |

**判读**：几何特征对三种架构的 `R²_field_cb` 均有增益（MLP `+0.046`、PointNet `+0.129`、PointNet++ `+0.042`）。本矩阵的最强且最稳单格是 **PointNet + xyz+geom**：主指标/pooled R² 最高，逐病例 P10 唯一为正，且只 1/8 负例。PointNet++ 的 `xyz+geom` MAE 最低但主 R² 较 PointNet 低 `0.043`；在纯 xyz 条件下 PointNet++ 的主指标最好（0.217），但仍明显低于加入几何后的 PointNet。六格 high-WSS R² 全为负、top10 幅值比仅 `0.313–0.461`，说明基础网络能学到中低 WSS 空间趋势，却仍严重低估高 WSS，不能把单 seed 结果视为最终结论或架构终裁。

**后续边界**：这轮完成“最简单形式”的单 seed baseline，不扩展模块、不访问 test；若要把 PointNet+xyz+geom 作为正式比较锚点，需要另行决定是否补多 seed。

### baseline 壁面 ParaView 包（2026-07-14 ✅DONE｜`8966` / `8967` 均 `0:0`）

- 原 `8961[0-5]` 的全量 6×8 导出在用户收窄范围后已取消，已生成的全量/冒烟产物已清理；不产生腔内切片，不读取 test。
- 当前范围只保留当前最佳 baseline `PointNet+xyz+geom` 的两个 val 病例，按同点逐病例 R² 排序：最佳 `slow/CHENG_LU_LI=0.3959`，最差 `slow/XU_YI_CAI=-0.0577`。单作业 `8966` 与依赖汇总 `8967` 均 `COMPLETED (0:0)`。
- 每个 `surface_wall.vtp` 同时挂载 `wss_cfd`、`wss_pred`、`err_wss`、`abs_err_wss`（Pa），以及四个 `*_over_cfd_max` 字段。归一化的分母固定为**同一病例 CFD 壁面最大 WSS**，所以 CFD、预测与误差可在 ParaView 中用同一 0–1 标尺比较。
- CFD 和预测同用 Gaussian `r=3 mm, sharpness=2, max_dist=3 mm` 回插到同一经配准变换的 STL；另写 `map_dist` / `map_valid`、mapping report、同点 wall CSV、CFD/Pred/signed-error 三联预览。正式 R² 始终只读同点 CSV。
- 两个最终 VTP 均验证 10 个字段齐全、mapping coverage=100%。`CHENG_LU_LI` 的 CFD max=48.595 Pa、同点 R²=0.3959；`XU_YI_CAI` 的 CFD max=36.417 Pa、同点 R²=−0.0577。最终批次输出与打开说明见 `training_wss_min/runs/baseline_2x3_simple/postview/README.md`；`comparison.csv` 与 `comparison_by_run.csv` 已生成。

## 第六轮指标/调度修订（2026-07-13 ✅DONE｜覆盖下方旧的 36-run 默认调度）

- WSS W0–W3 恢复为 P0 主线；横向 36 runs 仅为理论上限，不再是必做表。
- H-PW 已完成并收口。Track B 先完成 adapter/QA，默认只执行 `|v|+geom` 和联合 `u,v,w+geom` 单 seed sanity；其他目标/三 seed 按 Gate 触发。
- WSS 首要点级指标改为 `R²_field_casebalanced` + 物理单位 RMSE/MAE，并强制同报逐病例中位数/P10/失败数和热点护栏。`R²_field_raw` 与 `R²_casemean` 保留为历史衔接。
- V3P 0.429 与 wss_min 0.31–0.36 只作 B 级协议化参考，不报精确 gap。具体见[跨路线评估口径](../00-规范与记录/WSS跨路线评估与横向对比口径.md)。
## 第六轮对抗性审查回写（2026-07-13 ✅DONE｜覆盖旧版 F3 调度）

- **P0 先修指标合同**：现有 `R²_casemean` 是逐病例 spatial R² 的平均，不是病例 mean-WSS 的跨病例 R²。新合同拆成 A 病例 level、B 病例内 pattern、C hotspot，以及病例等权 Pa 误差；文档、`evaluate/train/gate/checkpoint` 完全一致并补单元测试前，不启动新训练。
- **E 结论降级**：E−D `field_cb +0.021` 只是在 dev1-val8/旧选模合同下通过开发筛选；在新版指标只读重评和 duplicate-grouped repeated validation 前，E 是临时候选，不称为已确认最佳可部署输入。
- **P0 密度修复前置**：`nsample=16`/完整点云评估的密度错配先做同 seed 单变量实验，形成新的冻结 control；旧版受混杂的 `~0.54` train-fit 不直接进入信息天花板裁决。
- **F3 改为 2×2**：架构 `{冻结 control/高容量或无下采样}` × 输入 `{geometry/geometry+RCR}`，加入 shuffled-RCR、RCR-only/随机病例特征负对照，同时报告 train-fit 与病例外 validation。信息效应和架构效应允许并存；取消单一 `train R²≥0.85` 二分。
- **当前顺序**：`P0-Metric → P0-Density → 2×2 F3 → 分支优化 → repeated validation`。oracle 特征继续强制 `oracle_non_deployable=true`；压力/速度保持条件路线。

## 全链路基础检查（2026-07-12 ✅DONE｜当前标量主线无致命错位，下一轮先补 3 项基础 Gate）

- 已从预处理、配准/正交旋转、逐病例缩放、节点 ID 对齐、FPS 稀疏化、训练特征/标签索引、模型/激活/loss/选模和完整点云评估逐项检查。
- 77/77 included bundle 齐全；`det(R)≈1`、最大正交误差 `4.44e-16`、逆变换最大误差约 `7.17e-05 mm`；数组同长，跨时间步 `nodenumber`/坐标守卫全过；FPS-2000 后坐标与标签同索引误差为 0；train-only peak stats 与 split 无泄漏。
- 结论：既有 `R²≈0.31` 不能归因于“基础坐标/标签整体错位”。优先风险是 3 个 included 病例 roll-sign 不可靠但 QA gate 未告警、ball-query 半径固定在逐病例归一化尺度而非毫米尺度、预留速度路径尚缺 cell ID/裁剪同步守卫。
- 激活函数 GELU + 线性输出合理，不是当前首要提分项；下一轮优先级为严格尺度进入 FPS/ball-query、log+raw/case-balanced loss、密度鲁棒邻域，再做 global-local 和网络宽度/激活微调。
- 完整报告：[WSS最小化_全链路基础检查报告_2026-07-12](WSS最小化_全链路基础检查报告_2026-07-12.md)。本轮不读新 test16、不改代码/配置、不启动新训练。

## 第六轮｜A/B/D XYZ 尺度诊断（2026-07-12 ✅DONE｜尺度信号 B−A 成立，几何仍是主杠杆）

- 目的：判断旧纯 XYZ 较差是否主要由“坐标逐病例归一化到 `[-1,1]`”丢失物理尺度造成。
- A=`xyz`；B=`xyz+coord_scale`；D=`xyz+abscissa_norm+local_radius+curvature`。三组均为 dev1 / fixed FPS-2000 / B1 fixed target-weight / val-only / `seed={1234,7,2025}`。
- 预注册：B−A 的三 seed 均值在 field/casemean 均 `>0.02` 才判为可辨识尺度信号；D−B 报告显式几何增量和 seed 方差。
- 范围声明：B 是尺度诊断，不是严格物理尺度纯 XYZ 终审；本轮不读 test16，不外推几百/几千例数据上限。
- 详细协议与 Job 表：[WSS最小化_第六轮XYZ尺度诊断计划与执行](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)。

### ⚠ 首批提交（7029–7037）5/9 因 config 路径失效而失败，已重跑补齐

- 首批提交记录 `submitted_20260712_120420.txt` 用的是**旧目录名** `configs/round6/r6_scale_*.json`；但同日“目录规整”已把这批 config 迁到 `configs/xyz_scale_diag/scale_*.json` 并删除 `round6/`。
- 早启动的 `7029–7031`(A×3) 与 `7032`(B_s1234) 赶在删除前解析成功；`7033`(B_s7) 训练成功但评估阶段目录已删 → 崩溃；`7034`(B_s2025)、`7035–7037`(D×3) 载 config 即 `FileNotFoundError`，**从未训练**。根因为提交清单路径与实际目录不一致，与协议/数据/模型无关，未污染任何已完成 run。
- 修复：按正确 manifest `configs/sweeps/xyz_scale_abd.txt` 重跑——`7039`(B_s2025)、`7040–7042`(D×3) 完整 train+eval，`7043`(B_s7) 复用 ckpt 仅重评（新增 `cluster/run_eval_only.slurm`）。记录 `cluster/logs/resubmit_20260712_001011.txt`。**5 个重跑均 `COMPLETED (0:0)`，9/9 eval 齐全。**

### 完整结果（val 完整壁面点云；MAE 单位 Pa）

| 组·seed | Job | `R²_field` | `R²_casemean` | `R²_casemed` | 负例 | top10 比 | IoU | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A·1234 | 7029 | +0.089 | −0.032 | +0.042 | 3/8 | 0.236 | 0.226 | 3.591 |
| A·7 | 7030 | +0.178 | −0.083 | +0.063 | 2/8 | 0.287 | 0.156 | 3.519 |
| A·2025 | 7031 | +0.227 | −0.141 | +0.024 | 4/8 | 0.333 | 0.241 | 3.497 |
| **A `xyz` 三 seed** | | **+0.164±0.070** | **−0.085±0.055** | | | 0.285±0.048 | 0.208±0.045 | 3.536 |
| B·1234 | 7032 | +0.173 | −0.010 | +0.068 | 3/8 | 0.309 | 0.148 | 3.710 |
| B·7 | 7043 | +0.204 | −0.080 | +0.147 | 2/8 | 0.335 | 0.159 | 3.563 |
| B·2025 | 7039 | +0.248 | +0.115 | +0.114 | 1/8 | 0.352 | 0.230 | 3.342 |
| **B `xyz+coord_scale` 三 seed** | | **+0.208±0.038** | **+0.008±0.099** | | | 0.332±0.022 | 0.179±0.044 | 3.538 |
| D·1234 | 7040 | +0.326 | +0.208 | +0.242 | 1/8 | 0.406 | 0.286 | 3.113 |
| D·7 | 7041 | +0.308 | +0.208 | +0.206 | 1/8 | 0.385 | 0.302 | 3.087 |
| D·2025 | 7042 | +0.298 | +0.174 | +0.143 | 1/8 | 0.362 | 0.375 | 3.129 |
| **D `xyz+geom` 三 seed** | | **+0.311±0.014** | **+0.197±0.020** | | | 0.384±0.022 | 0.321±0.048 | 3.110 |

**对照（三 seed 均值差）**：B−A field `+0.044` / casemean `+0.093`；D−B field `+0.103` / casemean `+0.189`；D−A field `+0.146` / casemean `+0.282`。

### 判读（终裁）

1. **尺度信号 B−A 成立。** B−A field `+0.044`、casemean `+0.093`，两项三 seed 均值均过预注册 `+0.02` 门槛；且**逐 seed 方向一致**（matched-seed field 增量 +0.084/+0.026/+0.021，casemean +0.022/+0.003/+0.256，3/3 seed 为正）。**结论：逐病例 `[-1,1]` 归一化确实丢失了对 WSS 有用的病例物理尺度，补回 `coord_scale` 标量能稳定回收一部分——主要体现在跨病例可辨识性（casemean 由 −0.085 转正到 +0.008）。**
1. **旧开发口径下尺度信号 B−A 成立。** B−A field `+0.044`、旧 per-case mean R² `+0.093`，三 seed 方向一致。该结果说明 `coord_scale` 提供了可用上下文，但旧 `casemean` 不是病例 level R²，不能据此宣称“跨病例整体水平被恢复”；病例 level 结论等待新版指标只读重评。
2. **但尺度标量只补回约三成缺口，几何仍是压倒性主杠杆。** 从 A(0.164)→B(0.208)→D(0.311)：coord_scale 把 field 抬 `+0.044`，而显式几何（D）再抬 `+0.103`（D−B）、相对 A 共 `+0.146`。与第一轮“纯几何≈xyz+几何≫纯 xyz”的结论一致。
3. **与第四轮 C4「coord_scale No-Go」不矛盾，反而互补澄清。** C4 是在 `xyz+geom` 之上再加 coord_scale（`local_radius` 已带尺度 → 冗余无增量）；本轮 B 是在**无 geom 的裸 xyz** 上加 coord_scale（非冗余 → 有增量）。两者一起说明：**尺度信息本身有用，但一旦有 `local_radius` 等局部几何，标量尺度基本被覆盖**。这提示 C（严格物理 mm-XYZ）要想跑赢 D，必须靠“物理尺度同时进 FPS/ball-query 邻域”带来的、局部几何特征无法替代的增量，而非仅把 mm 尺度塞进 feature。
4. **绝对精度和稳健性仍不足。** 本轮最好的 D（`xyz+geom`，三 seed field `0.311±0.014`、casemean `0.197±0.020`）的高 WSS 护栏（top10 比 0.384 / IoU 0.321）延续“定位有信号、幅值系统性低估”。尺度诊断解释了旧纯 XYZ 差的一部分成因，但未证明当前协议的绝对信息上限；`0.70` 仅作长期理想参考。

### 交叉验证与下一步（详见[第六轮总路线](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)与其拆分文档）

- 已对第六轮各新思路做证据交叉核对；内容随文档重构分入[WSS 精度突破](WSS最小化_第六轮_WSS精度突破计划与执行.md)与[边界条件与速度路线](WSS最小化_第六轮_边界条件与速度路线.md)。
- **用户批准的下一步优先级（2026-07-12）**：`L1/L2 loss` 与 `C/E 严格尺度` **并列第一**；架构 `M1/M2/M3` 次之；速度→WSS 只做 `V0/V1` oracle 复核 0.632 天花板，**oracle 未过门前不建 data_new adapter、不训速度 surrogate**。
- **多目标扩展（2026-07-13 修订）**：原 36 runs 保留为理论上限，不再是必做主表。压力-壁面 H-PW 已完成；Track B 的 data_new adapter 完成 QA 后，默认只跑近壁 `|v|+geom` 和联合 `u,v,w+geom` 单 seed sanity，其余实验按机制/论文需要触发。近壁速度与速度→WSS oracle 复用同一 adapter。完整方案与结果见[横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md)。

### 结果返回后的预注册后续（待用户批准）

- 当前 A/B/D 九个作业的 config、Gate 和评估不变；不在运行中修改协议。
- 尺度补充：C=严格 mm-XYZ（输入/FPS/ball-query 同时使用物理尺度）；E=`XYZ+coord_scale+geom`，与 A/B/D 构成嵌套因子对照。
- 输入信息：先做残差↔RCR/压力/分流只读诊断和 oracle-BC 上限；临床实测、估计与 CFD oracle 强制分开。
- 架构：global-local/FiLM、density-robust neighborhood、解析尺度 residual 和小模型对照；不先扩大参数量。
- loss：当前 B1 vs `log + raw scaled-Huber` vs case-balanced raw robust 的三 seed 小矩阵。
- 速度→WSS：先做 CFD-velocity oracle 和降采样上限；当前光滑剖面 oracle 约 `R²=0.632`，未过 `0.70`，不直接训全速度 surrogate。
- 详细 Gate、数据边界和分阶段顺序见[第六轮总路线与执行入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)。

## 第六轮｜W0 审计 + W1 因子(E) + C 邻域预审计 + W2-L1 + W3 组合（2026-07-12 ✅DONE｜E 险胜、尺度/几何冗余、raw-Huber seed 脆弱、W3 组合阴性）

> 承接上节 A/B/D，本节补 **W0 只读审计**、**E 组**（补齐 A/B/D/E 2×2 嵌套因子）、**C 邻域预审计** 与 **W2-L1 raw-Huber λ Gate**。协议全程冻结 B1/dev1/FPS-2000/val-only，seed=1234/7/2025。作业 `7557–7562`。产物：`runs/_audits/{w0_coord_scale,w1_c_neighborhood,round6_w1w2_summary}/report.md`。

### W0｜coord_scale 只读审计（`runs/_audits/w0_coord_scale/report.md`）

- **val 无尺度外插**：val `coord_scale ∈ [147, 214]` 全落在 train `[128, 262]` 内；dev1 **无任何壁面裁剪**（`wall_crop_applied` 全 False）。
- **coord_scale 是血管尺寸代理，但与 WSS 幅值无关**：Spearman(coord_scale, bbox 对角线)=+0.74、(z 向 extent)=+0.79、(入口面积代理)=+0.61、(local_radius 中位)=+0.65；而 (WSS case mean)=−0.15、(p95)=−0.03、(p99)=+0.02。**结论：B 组 coord_scale 的增益来自跨病例可辨识性/归一化上下文，不是幅值定标**——这与下面 casemean 的抬升方向一致，也解释了为何 B 组抬 casemean 多于 field。
- **coord_scale 是血管尺寸代理，但与 WSS 幅值无关**：Spearman(coord_scale, bbox 对角线)=+0.74、(z 向 extent)=+0.79、(入口面积代理)=+0.61、(local_radius 中位)=+0.65；而与 WSS case mean/p95/p99 的相关约为 −0.15/−0.03/+0.02。结论仅限于“它提供尺寸/归一化上下文”，不再用旧 `casemean` 推导病例 level 机制。

### W1｜A/B/D/E 2×2 因子（E 补齐，三 seed 均值；MAE 单位 Pa）

| 组 | 输入 | R²_field | R²_field_cb | R²_casemean | R²_casemed | 负例(∑/24) | top10 比 | IoU | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | xyz | +0.164±0.057 | +0.164±0.050 | −0.085±0.044 | +0.043 | 9 | 0.285 | 0.208 | 3.536 |
| B | xyz+scale | +0.208±0.031 | +0.213±0.036 | +0.008±0.081 | +0.110 | 6 | 0.332 | 0.179 | 3.538 |
| D | xyz+geom | +0.311±0.012 | +0.309±0.016 | +0.197±0.016 | +0.197 | 3 | 0.384 | 0.321 | 3.110 |
| **E** | **xyz+scale+geom** | **+0.331±0.014** | **+0.330±0.014** | **+0.199±0.018** | **+0.204** | **2** | **0.404** | **0.341** | **3.094** |

**严格归因（逐 seed 配对差，mean±std）**：

| 对比 | 含义 | ΔR²_field | ΔR²_field_cb | ΔR²_casemean |
|---|---|---:|---:|---:|
| B−A | 尺度主效应 | +0.044±0.028 | +0.049±0.019 | +0.093±0.115 |
| D−A | 几何主效应 | +0.146±0.069 | +0.145±0.066 | +0.282±0.031 |
| E−B | 几何｜有尺度 | +0.123±0.044 | +0.118±0.049 | +0.191±0.098 |
| E−D | 尺度｜有几何 | +0.020±0.009 | +0.021±0.012 | +0.002±0.011 |
| E−D−B+A | 尺度×几何交互 | −0.023±0.033 | −0.027±0.031 | −0.091±0.115 |

**判读**：

1. **E 是当前最佳可部署输入，但仅险胜 D。** E 三 seed field `0.331`、field_cb `0.330`、casemean `0.199`、**负例仅 2/24（全组最少）**、top10/IoU 均最高。E−D 在 field_cb `+0.021`（过预注册 Gate-1 `+0.02`），负例不增、top10/IoU 反升，**E 干净通过 Gate-1，取代 D 成为最佳输入**。
2. **尺度与几何冗余、非协同（关键新结论）。** 交互项 E−D−B+A 三项**全为负**（field −0.023 / casemean −0.091）；尺度的边际价值从"无几何"的 B−A(+0.044 field/+0.093 casemean) 坍缩到"有几何"的 E−D(+0.020 field/**+0.002 casemean**)。**一旦有 `local_radius` 等局部几何，coord_scale 标量的贡献基本被吸收**——量化印证上节 §判读3 与 C4 No-Go 的猜想。W0 也从另一侧印证：coord_scale 编码的是尺寸而非幅值，几何特征已覆盖其可用信息。
3. **E 的增益虽小但一致**：E−D field 逐 seed +0.009~+0.031 全正，主要抬 field/pooled 与热点护栏，对 casemean 无增量。

1. **E 是 dev1 旧口径下的临时候选。** E 三 seed field `0.331`、field_cb `0.330`、旧 per-case mean R² `0.199`、负例 2/24；E−D field_cb `+0.021` 通过旧开发筛选，但尚未通过新版指标与 grouped repeated validation，不再称为最佳可部署输入。
2. **尺度与几何无协同证据。** 交互项 E−D−B+A 为负；在已有 `local_radius` 等显式几何时，`coord_scale` 的边际收益很小。该结论限于 dev1/旧指标，不外推为普遍机制。
3. **E 的增益虽小但一致**：E−D field 逐 seed +0.009~+0.031 全正，主要抬 field/pooled 与热点护栏，对 casemean 无增量。

> ⚠ **2026-07-13 代码核验更正**：上文"E 干净通过 Gate-1、取代 D 成为最佳输入"依据的是文档 `field_cb` 口径；但提交的 `gate1_compare.py` 判 GO 需 `Δr2_field>0.02 且 Δr2_casemean>0.02`，`field_cb` 被加载却不参与判定。E−D 的 `Δr2_casemean=+0.002` 使 `common_improvement=False` → 代码实际判为 **INDIFFERENT/NO_GO**；且产出 E 的 checkpoint 由 composite（`0.6·casemean+0.4·field`）选出。故"E>D"仅为探索性，**不作冻结 control/最佳输入结论**；须先统一"文档 Gate=代码 gate1=选模 rule"口径。详见 [WSS 精度突破 §2026-07-13（第二轮·本地代码核验）](WSS最小化_第六轮_WSS精度突破计划与执行.md)。

### W1｜C 组邻域预审计（`runs/_audits/w1_c_neighborhood/report.md`，只读未训练）

- C 定义：train 拟合全局共享常数 `s_global=max(train coord_scale)=261.9` 缩放所有病例，固定物理半径。复现 PointNeXt-S 的 SA 级联对比 A/C 每层邻居。
- **Gate 结论**：① **C 不产生退化邻域**（A/C 所有层孤立率=0.000，无空邻域）→ 邻域结构上可训；② **C 更密而非更稀**（训练分辨率邻居中位约 A 的 1.5–1.9×）；③ **`nsample=16` 截断已主导**（评估分辨率 A/C 截断率≈1.00，训练细层≈0.93–1.00），把 C"物理尺度进邻域"的预期收益大部分抹平，真正分化只存活到最粗层。④ 结合 W1 已证尺度/几何冗余，**C 先验跑赢 D 的理由弱，判为低优先级（低于 E/L1）**；若仍训 C 需先调 radius 协议（提高 nsample 或改 median 参考的 s_global）。

### W2-L1｜raw-Huber λ Gate（固定 D 输入，单 seed 1234，`7560–7562`）

log-z MSE 主损失 + raw-space scaled Huber 辅助。L0=`r6_scale_D_xyzgeom_s1234`（λ=0）。

| run | λ | R²_field | R²_field_cb | R²_casemean | top10 比 | p99 比 | IoU | high_wss_MAE | max 比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 0.0 | +0.326 | +0.331 | +0.208 | 0.406 | 0.369 | 0.286 | 13.85 | 0.14 |
| L1 | 0.1 | +0.374 | +0.376 | +0.245 | 0.447 | 0.478 | 0.347 | 13.03 | 0.17 |
| L1 | 0.3 | +0.362 | +0.362 | +0.235 | 0.421 | 0.370 | 0.330 | 13.48 | 0.14 |
| **L1** | **1.0** | **+0.385** | **+0.384** | **+0.253** | **0.463** | 0.450 | **0.360** | **12.61** | 0.16 |

**判读**：**三个 λ 全面优于 L0**（field/casemean/top10/IoU/high_wss_MAE 无一退化），且**无爆峰**（max 比 0.14–0.17，远低于 1.5 阈值）。λ=1.0 最佳（vs L0 同 seed：field +0.059、casemean +0.045、top10 +0.057、high_wss_MAE 13.85→12.61）。单 seed 仅筛选，**选中 λ=1.0 补三 seed 确认**。注意 high_wss_MAE 只降 ~9%、max 比仍 ~0.16——raw-Huber 抬中高段与整体 R²/定位，但**极端峰值坍缩仍未解决**（延续第二轮"峰值压扁是硬上限"判断）。

### W2-L1 λ=1.0 三 seed 确认（`7563–7564`+s1234）+ W3 唯一组合（`7565–7567`，2026-07-12 ✅DONE）

**L1 λ=1.0 三 seed（D 输入，loss-control）**：单 seed s1234 的 field 0.385 属"幸运高 seed"，三 seed 均值回落：

| 配置 | 输入 | λ | R²_field | R²_field_cb | R²_casemean | 负例(∑) | top10 比 | IoU | high_wss_MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D (baseline) | xyz+geom | 0.0 | +0.311±0.012 | +0.309±0.016 | +0.197±0.016 | 3 | 0.384 | 0.321 | 14.36 |
| D+rawHuber | xyz+geom | 1.0 | +0.347±0.033 | +0.346±0.035 | +0.199±0.054 | 4 | **0.429** | 0.310 | **13.37** |
| E (input-control) | xyz+scale+geom | 0.0 | +0.331±0.014 | +0.330±0.014 | +0.199±0.018 | **2** | 0.404 | 0.341 | 13.90 |
| **W3 = E+rawHuber** | xyz+scale+geom | 1.0 | +0.316±0.037 | +0.315±0.036 | +0.189±0.009 | **2** | 0.400 | **0.351** | 14.08 |

**W3 Gate（配对 seed 差）**：W3−E field `−0.015±0.032` / top10 `−0.004`；W3−(D+rawHuber) field `−0.030±0.022` / top10 `−0.029`；两侧仅 IoU 微升（+0.011 / +0.042）。

**判读（W3 = 阴性，按 §6 停止组合）**：

1. **W3 未通过组合 Gate。** 逐 seed 看，W3 在**全部 3 seed 低于 D+rawHuber**（field_cb 0.366/0.294/0.286 vs 0.384/0.300/0.353），在 **2/3 seed 低于 E**（仅幸运 seed s1234 反超）。W3 在主指标（field/field_cb/casemean/top10）不优于任一控制，只在热点 IoU 微升。**按 §6："组合未同时优于两个 control → 不再扩展交互组合"，正式停止 scale-feature × raw-Huber 的组合线。**
2. **机制：尺度特征与 raw-space Huber 冗余/轻微互斥。** 两者都在推高 WSS/物理尺度表示，叠加不增益反而略降——与 W1 尺度×几何交互为负一致。
3. **L1 raw-Huber 收益真实但 seed 脆弱。** D+rawHuber 三 seed 抬 field +0.036、top10 +0.045、high_wss_MAE −7%（幅值定向有效），但 casemean 持平、+1 负例、方差大（field_cb ±0.035，主要靠 s1234）。**单 seed 0.385 不可外推**——印证 §5 W2"单 seed 只用于廉价筛选"。
4. **两条互斥候选，均未跨越稳健性瓶颈。** E（最稳、负例最少、IoU 最高）与 D+rawHuber（field/top10 幅值最好但脆弱）**不可叠加**，且都未改善 casemean/负例。**下一步交由用户裁决 W5 确认候选（E vs D+rawHuber），或启动尚未尝试的 L2 case-balanced robust loss 直击病例稳健性**。汇总真源：`runs/_audits/round6_w1w2_summary/report.md`。
4. **两条互斥候选，均未跨越稳健性瓶颈。** E（最稳、负例最少、IoU 最高）与 D+rawHuber（field/top10 幅值最好但脆弱）**不可叠加**，且都未改善旧 per-case mean R²/负例。最终审查后两者均冻结，不再直接进入 W5 或 L2；先完成 P0-Metric、P0-Density 与 2×2 F3。汇总真源：`runs/_audits/round6_w1w2_summary/report.md`。

## 第六轮 横向对比 H-PW Track A｜压力-壁面（2026-07-12 ✅DONE｜压力空间型态比 WSS 易学）

- 目标：同最小协议（FPS-2000/PointNeXt-S/dev1/B1 schedule）预测壁面 **gauge 压力**（peak-step `wall_pressure` 逐例去均值，隔离 ~1.5e4 Pa 的 DC 偏置），`xyz` vs `xyz+geom` 矩阵，纯 MSE（关掉 WSS 长尾加权）。
- 数据/代码：管线支持 `target` 切换（原硬编码 `wall_wss`）；gauge stats `pressure_gauge_stats_v2_dev1.json`（53 例/707705 点，std=529.2 Pa，gauge∈[-1880,1164]）；配置 `configs/multitarget/press_wall_{xyz,xyzgeom}_s{1234,7,2025}.json`（6 个）。
- Jobs `7551–7556` 全部 `COMPLETED (0:0)`（记录 `submitted_20260712_050410.txt`）；CPU 冒烟先行通过。

| 组·seed | Job | `R²_field` | `R²_casemean` | 负例 | top10 比 | IoU | MAE(Pa) |
|---|---:|---:|---:|---:|---:|---:|---:|
| press·xyz·{1234,7,2025} | 7551–3 | +0.406/+0.424/+0.448 | +0.286/+0.207/+0.271 | 1/2/3 | — | — | ~261 |
| **press `xyz` 三 seed** | | **+0.426±0.021** | **+0.254±0.042** | | 0.601±0.058 | 0.298±0.021 | 261 |
| press·xyzgeom·{1234,7,2025} | 7554–6 | +0.506/+0.495/+0.592 | +0.493/+0.522/+0.513 | 0/0/0 | — | — | ~225 |
| **press `xyz+geom` 三 seed** | | **+0.531±0.053** | **+0.509±0.015** | | 0.575±0.046 | 0.389±0.097 | 225 |

- **判读**：同协议下 gauge pressure 空间型态比 WSS 易学——`xyz+geom` 压力 `0.531/0.509` vs WSS `0.311/0.197`；`xyz` 压力 `0.426/0.254` vs WSS `0.164/−0.085`。压力 casemean 全程为正、`xyz+geom` 下 **0/8 失败例**（WSS 每 seed 都有负例）。几何增益 field `+0.105`/casemean `+0.255`。该结果不使用跨目标通用 `0.70` 门槛，不外推其他目标。
- **选模**：沿用 WSS 复合选模，列为**探索性**；只读复核显示复合最优 epoch 与压力 R² 最优 epoch 一致，数值应接近压力专用选模；正式复选待用户批准（预计不改数值）。
- 完整表与读法：[横向多目标对比 §8](WSS最小化_第六轮_横向多目标对比计划与执行.md#8-h-pw-结果壁面-gauge-pressure2026-07-12-done探索性选模)。Track B（压力-内部 + 近壁速度）待 data_new adapter。
- **选模**：沿用 WSS 复合选模，列为**探索性**；只读复核显示复合最优 epoch 与压力 R² 最优 epoch 一致。仅在需要对外确认 H-PW 时，按压力专用指标做 val-only 只读重评；Track B 保持暂停。
- 完整表与当前读法：[横向多目标对比 §3](WSS最小化_第六轮_横向多目标对比计划与执行.md#3-已完成-h-pw-结果)。

## 汇报｜A0E-ctrl 两例 postview（2026-07-12 ✅DONE）

- 模型 `r5_a0e_b1_ctrl_s1234`；病例 `slow/WU_FENG_YAN`、`fast/RAN_QING_BO`（均为 train）。
- 产物：`docs/03-汇报材料/figures/WSS最小路线_20260712/postview_a0e_ctrl/`（`*__surface_wall.vtp` 含 CFD/Pred/Error；映射覆盖率 100%）。
- 同点 wall R²：`0.3225` / `0.4959`；脚本 `training_wss_min/tools/export_wss_postview.py`。

## 第五轮｜F0 结案（2026-07-12 ✅DONE / 科学结案·工程未达标）

- 全部 §8.1 机制问题均有可复核裁决 → **第五轮科学结案**；dev1 field/casemean ~0.31/0.21 ≪ 0.70 → **未达内部工程目标**（§8.2），未跑 OOF（无达标候选）。
- 机制链：拟合足（A0D）→ 当前协议泛化锚点 ~0.34（A0E）→ 现有 61 例范围内 ~0.31 暂时平台（LC）→ 可部署 BC 无新杠杆（B-BC）→ 标签噪声小（CFD，R²_cap ~0.92–0.96）。该结论限于当前数据池、输入和模型协议，不外推数千个高质量独立病例的上限。
- 报告：`training_wss_min/runs/_round5/final_report/round5_final_report.md`。可选后续（非部署路径、待用户定）：oracle RCR 增量探针 / P2 loss 探针。
- 归档口径：用户确认无达标候选时不强行运行 15-run OOF；L0/OOF/T16 经记录未触发，`test16` 保持未读；P2 转交第六轮 loss 小矩阵。见[第五轮归档说明](_archive/WSS最小化/WSS最小化_第五轮结案与归档说明_2026-07-12.md)。

## 第五轮｜CFD 可信性审计 §6（2026-07-12 ✅DONE）

- read-only；dev61。产物 `training_wss_min/runs/_round5/cfd_audit/{cfd_audit.py,cfd_per_case.csv,cfd_summary.json,cfd_audit_report.md}`。
- peak 相位按**固定步**（1162/idx21）统一取，结构上无跨病例相位错配；仅 3/61 真峰晚 ≥12 步（含 2 val：`LIU_JUN_FENG` +13.8%、`CHENG_GUANG_SEN` +12.2%）→ n=8 val 数值脆弱。
  - ⚠ **2026-07-13 与固定 peak 的交互**：这 2 个 val 病例（占 dev1-val8 的 25%）在固定步与真峰之间有约 12–14% 的幅值差；`CHENG_GUANG_SEN` 又是 A0R 常见失败例。固定 peak 协议保持不变，但新版病例级 level 报告必须把两例的真峰敏感性单列，不得据 val 结果重新选择时相。见 [WSS 精度突破计划](WSS最小化_第六轮_WSS精度突破计划与执行.md)。
- 近壁 QA 全清（normal_invalid=0、basis≤0.066、radial≤0.066、mesh≥786k）；高 WSS 尖峰为真实几何热点（不与 QA 旗标共现），但单节点 max 受尖峰主导，报告以 p95/p99 为准。
- **重复几何**：独立证实 `HOU_SHEN_QIAN`=`KANG_XI_MING` 同一几何（且均 dev1 train）→ OOF 须整组；`LIU_XI_QUAN`（excluded）peak 全零损坏。
- **裁决：~0.31 平台主要不能由简单 CFD 复现噪声解释。** 复现 floor ~2%，即便按 10–15% per-case 噪声估算，理论 cap 仍约 0.92–0.96；但该审计不能区分模型容量、密度协议、缺失条件信息和目标定义，不再写成“信息上限已确认”。

## 第五轮｜B-BC 资产审计（2026-07-12 ✅DONE / 可部署 B-BC 关闭）

- read-only 审计 61/61 dev 病例 BC 完整（udf-inlet.c + vf-in + 5 压力监测）。产物 `training_wss_min/runs/_round5/bc_audit/{audit_ag_bc.py,bc_per_case.csv,bc_summary.json,bc_audit_report.md}`。
- **决定性发现：入口流量不是病人特异的。** Fourier 流量模板（a0..b8,w,T,A1/B/D/E/n）在所有病例逐字节相同；入口 BC 为平速 `v=1e-6·template/area`，故 `Q=v·area=1e-6·template` 与面积无关 → 各病例入口流量本质相同（dev CoV≈1.85e-4，Fourier↔实测比 0.999–1.000）。唯一 per-case 入口标量是入口**面积**（几何可得）。
- **可部署 vs oracle**：可部署 = 入口面积/速度/流量（但零方差或与几何冗余，非信息量）；**唯一有跨病例方差的是出口 RCR（CoV 0.55–0.67）与出口压力/流量分配，全部 `oracle_non_deployable`**（CFD 设定/解产物，新病人不可知）。
- 数据质量旗标：两对拷贝 BC（`HOU_SHEN_QIAN=KANG_XI_MING` 均 dev1 train、`LIU_XI_QUAN=LI_BING_YI`）→ 其面积/RCR 不独立；`vf-outri` 20 例退化（15 在 dev1）；两例异常入口监测均已 excluded。
- **裁决：可部署 B-BC 关闭**——现有资产中没有可部署、有信息量且病人特异的 BC 输入。出口 RCR 只有方差证据，尚未证明能在病例外解释 WSS 残差；最终审查后仅允许把它放入带 shuffled/RCR-only 负对照的 2×2 oracle 探针，不能写成已确认根因。

## 第五轮｜LC 三链 learning curve（2026-07-12 ✅DONE）

- G1 批准后启动；A0E-ctrl（标准 B1，min_lr=1e-5）配方，val 恒为 dev1 固定 8 例；3 链嵌套 `LC13⊂26⊂40⊂53`，硬分层 cohort×train-only WSS 三分位；case-drop（chain salt）与 model seed 分离。生成器 `make_configs_round5_lc.py`，per-subset 划分/WSS stats，feature stats 与 loss 分位运行时重算。
- 30 个 run（3 链×3 seed×{13,26,40} + 3 端点）Job `6994–7002`/`7003–7022` 全部 `COMPLETED`。
- **field R² 均值±std：13 `0.258±0.023` / 26 `0.297±0.025` / 40 `0.312±0.033` / 53 `0.310±0.044`**；casemean `0.087/0.158/0.177/0.208`。
- **配对 field 增量：13→26 +0.039±0.037、26→40 +0.014±0.027、40→53 −0.002±0.049**——**field 在当前 13–53 例范围内约 40 例后出现 ~0.31 暂时平台**，40→53 增量与 0 不可分。
- 端点 53 逐 seed field `0.359/0.252/0.319`（std 0.044，超过 26→53 整段增量）；stage-1 单 seed 的"53 仍在上升"是 s1234 偏高伪影，补 seed 后纠正——多 seed 必要性再次印证。
- **裁决（修正后口径）：现有 61 例池内的小步扩展没有显示可将 field R² 从 ~0.31 推到 0.70 的证据**。剩余差距与输入信息、坐标/尺度表示和 loss 目标错位有关；13–53 例 LC 不能否定几百/几千例高质量独立数据的潜在收益。高 WSS 护栏全程未改善（端点 top10 ratio 0.466 / IoU 0.302）。
- 产物：`training_wss_min/runs/_round5/learning_curve/{learning_curve_report.md,lc_points.csv,lc_curve.png,lc_verdict.json}`。下一步见 G2 路由：B-BC 输入信息（首选，先 read-only 审计）+ P2 loss 探针（并行）；不启动 L0/OOF（当前配置远低于 0.70）。

## 第五轮｜A0E dev1 control 重锚（2026-07-12 ✅DONE）

- 目的：判定历史 `0.34` 是真实泛化上限还是训练预算伪影，并冻结 LC 训练协议。control=历史 B1 anchor（field/casemean `0.3511/0.2138`）。
- `ctrl`（B1 逐字，min_lr=1e-5）Job `6992`：field/casemean **`0.3587/0.2300`**，best epoch 29，负 R² 病例 0——复现 anchor（±0.02 内），确认 `0.34/0.23` 是真实泛化上限。
- `nsl`（仅抬 LR 下限到 2e-4）Job `6993`：field/casemean `0.3419/0.2125`，2 例负 R²（失败率 0.25）——非饿死 LR 对 dev1 无益且略有害。
- 裁决：LC 用标准 B1 schedule（min_lr=1e-5，即 ctrl）；更正预注册时"dev1 需非饿死 LR"的假设。A0D"按 optimizer step 计预算"规则仍成立，但 dev1（batch8、约 1120 step、best-val 时 LR 健康）本就满足。产物 `training_wss_min/runs/_round5/a0e_control/a0e_control_report.md`。

## 第五轮｜G1 第二次分支裁决（2026-07-12 ✅DONE）

- 合并 A0D（拟合能力 GO）+ A0E（`0.34` 真实泛化上限）+ A0R（53 例即过拟合）+ A1（密度受 mapping 阻断）证据。
- 裁决：**批准 LC**（泛化/病例数主导）；**关闭 B-REP**（表示坏了前提被证伪）；维持 B-DEN/B-BC `BLOCKED`；normalized-MSE↔raw-R² 目标错位（79.75×）列 P2 并行探针。
- 产物：`training_wss_min/runs/_round5/branch_experiments/branch_decision_g1.md`。

## 第五轮｜A0D 基础拟合链（2026-07-12 ✅GO）

- 只读审计：stats/逐点对齐无异常；AMP 首步跳过不足以解释缺口；旧四病例 micro 只有 160 次 optimizer updates。
- Job `6986`：四个单病例简化 MLP 全部达 R² `0.991–0.999`。
- Job `6990`：four-case shared plain MSE，field/casemean `0.979526/0.982533`，逐例最低 `0.975705`。
- Job `6991`：只恢复 fixed target-weight alpha2，field/casemean `0.993723/0.992423`，相对 6990 `+0.014197/+0.009890`。
- 结论：`0.844` 不是四个已见病例的拟合上限，target-weight 单独不是原缺口的充分原因。这些仍是 train-only/canonical-2000 结果，不代表新病例精度。
- 产物：`training_wss_min/runs/_round5/a0d_fit_chain/`。下一步为预注册的 dev1 val-only 单变量候选。

## 第五轮｜B-REP/C2 global context（2026-07-11 ⛔NO_GO）

- Job `6984`：四病例 train-only、fixed FPS-2000、seed1234、last checkpoint，`COMPLETED (0:0)`，未读 val/test16。
- train `R²_field_raw=0.740682`、`R²_casemean=0.695389`，未达 `0.95/0.95`，dev1 未提交。
- C2 `NO_GO`；已列 B-REP 候选均未通过 micro Gate，暂停新候选训练、LC 和 B-DEN，转入 normalization/loss/标签对齐与几何可辨识性审计。

## 第五轮｜B-REP/C1 radius normalization（2026-07-11 ⛔NO_GO）

- Job `6983`：四病例 train-only、fixed FPS-2000、seed1234、last checkpoint，`COMPLETED (0:0)`，未读 val/test16。
- train `R²_field_raw=0.803401`、`R²_casemean=0.760602`，未达 `0.95/0.95`，dev1 未提交。
- C1 `NO_GO`；下一个仅执行 C2 单输出 global context 的四病例 micro。

## 第五轮｜B-REP/M1 逐点 MLP（2026-07-11 ⛔NO_GO）

- Job `6982`：四病例 train-only、fixed FPS-2000、seed1234、last checkpoint，`COMPLETED (0:0)`，未读 val/test16。
- train `R²_field_raw=0.880817`、`R²_casemean=0.844018`，虽高于 B1 micro，仍未达 `0.95/0.95`，所以 dev1 未提交。
- M1 `NO_GO`；STL 表面特征受 mapping Gate 阻塞，下一个只执行 C1 radius-normalized relative position 的四病例 micro。

## 第五轮｜A1 与 G0（2026-07-11 ✅已裁决）

- density probe 确认同索引预测会随 canonical/full 密度变化（三 seed 预测间 R² `0.865–0.882`），但相对真值的两项主指标方向不跨 seed 一致。
- full density 的 SA L1–L4 query cap 截断率约 `0.9998/1/1/1`，密度敏感有明确结构证据。
- 真值 IDW3 oracle 通过：field `0.9293`、casemean `0.9174`、top10 ratio `0.9353`、IoU `0.8003`。
- 但预注册 STL mapping 总 Gate 为 `0/8`；失败集中在连续三角面采样到离散 CFD wall 节点的 1 mm 覆盖门槛，因此 A1 `NO_GO`并阻断 D1/D2。
- G0 只批准 B-REP 的逐点 MLP 候选；先用已锁定四病例重跑 micro-overfit，通过后才能提交 dev1 Gate-1。

## 第五轮｜A0R 多 checkpoint 只读诊断（2026-07-11 ✅DONE）

- 范围：B1 `s{1234,7,2025}` 的 15 个 best/last/candidate checkpoint，train/val × canonical-2000/full；无训练、无 test16。
- best canonical train `R²_field_raw=0.538–0.573`、`R²_casemean=0.494–0.528`；容量/基础拟合不足信号明确。
- last 相对 best 的 canonical train field 平均 `+0.070`，canonical val field 平均 `-0.061`，继续训练不是同时修复拟合与泛化的答案。
- best full-canonical：train field/casemean 平均 `-0.159/-0.171`，val 平均 `-0.055/-0.080`，top10 ratio/IoU 也在三 seed 中全部下降；密度迁移方向一致。
- 最集中的 val 失败病例是 `slow/XU_YI_CAI`，其次为部分 seed/checkpoint 的 `slow/CHENG_GUANG_SEN`。产物见 `training_wss_min/runs/_round5/a0_readonly/`。

## 第五轮｜A0M 四病例 micro-overfit（2026-07-11 ⛔NO_GO）

- 四例固定为 fast/low `SUN_ZHI_YU`、fast/high `WANG_DAO_CHUN`、slow/low `ZANG_YU_SHU`、slow/high `MA_TIAN_YI`；全部来自 dev1 正式 train。
- Slurm Job `6981`：B1 配方、seed1234、fixed canonical FPS-2000、160 epoch、last checkpoint、无 val 选模、未读 test16；状态 `COMPLETED (0:0)`。
- train `R²_field_raw=0.76548`、`R²_casemean=0.72381`，未达同时 `>=0.95` 的 Go 阈值，因此 A0M `NO_GO`。
- 产物：`training_wss_min/runs/_round5/a0_micro/`；按停止规则暂停 LC/大规模 sweep，待 A0R/A1 后进入 G0。

## 第五轮｜P0 评价协议（2026-07-11 ✅DONE）

- 评价现同时输出 pooled `R²_field_raw`、逐病例等权 `R²_casemean` 与病例总权重相等的 `R²_field_casebalanced`，并固化 median/P10/负 R² 数/失败率。
- Gate-1 仅在 field/casemean 相对 control 都改善 `>0.02` 且 top10 ratio/IoU 均未下降 `>0.05` 时为 Go；不再允许高 WSS 单指标旁路 Go。
- `evaluate` 默认 val-only；test 必须额外显式 `--allow-test`。P0 未访问 legacy test16。
- 旧 control `r4_dev1_b1_tgtw_fixedq_s1234` / best epoch 49 的 val-only 复评：`R²_field_raw=0.3510821344`、`R²_casemean=0.2138313493`、`MAE=3.0635919684`，与历史值最大偏差 `3.24e-9`；新 `R²_field_casebalanced=0.3451643087`。
- 产物：`training_wss_min/runs/_round5/protocol/protocol_report.md`、`protocol_regression.json`。下一步可并行 A0R/A0M/A1。

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
- `R²_field`：所有点 pool 起来算（受高 WSS 病例主导）；旧 `R²_casemean`：逐病例分别计算 spatial R² 后再平均，虽为病例等权，但不是病例 mean-WSS 的跨病例 level R²。
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

历史轮次到此结束。当前待办不在历史段落重复维护，以文首最终审查回写和[WSS 精度突破计划](WSS最小化_第六轮_WSS精度突破计划与执行.md)为唯一准绳。
