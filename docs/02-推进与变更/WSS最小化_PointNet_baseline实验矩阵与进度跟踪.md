# WSS 最小化·PointNet baseline 实验矩阵与进度跟踪

> **🧊 已冻结（2026-08-07 标注）**：2026-08-06 起主线收束为峰值体域 `u,v,w,p`，本矩阵不再新增实验或续写；只作为 PointNet/PointNet++ baseline 的结果档案供论文对照引用。

> **转交体域 PINN 的架构口径（2026-09-04 对齐）**：本页 2026-07-18 的 P2V/Q1V
> 只是当时两个分支的 single-seed/test27 优先候选。体域 PINN V1/V2 后来实际冻结的是
> PointNet=P2V、PointNet++=2026-07-21 的纯 D2 `c125-k128`；后者
> `R²_cb=0.2765`，与 Q1V `0.2763` 基本打平且未获独立多种子确认。2026-09-04 用户决定
> Centerline-V2 formal V4 继续采用这两个 PINN provenance 锚点，只继承骨干结构、不加载
> 旧权重。2026-07-23 以后 direct-WSS 线发展的 D2-k64 + PointNeXt-R/LocalGeoPE 与
> L-SA2/H2/log-radius 是另一条后续开发线，不得倒写成 V1/V2 当时使用的父模型。

> 建立日期：2026-07-14
>
> 更新日期：2026-07-29（同 seed H2 control / +log(local_radius) 已完成；处理臂晋级新开发锚点）

> 三种子：S2−M1 ΔR²_cb=`-0.0086/+0.0007/+0.0335`，均值`+0.0086`但不稳定；S3−S2=`+0.0396/+0.0052/-0.0033`，均值`+0.0138`、2/3正；S3−M1=`+0.0310/+0.0059/+0.0303`，均值`+0.0224`、3/3正。S3 MAE 3/3下降、ILO平均R²提升；AAA unrupture 与少数 high-WSS seed 仍是护栏。test36 仍为历史工程筛选集。
>
> 用途：跟踪导师提出的 PointNet baseline 容量、采样点数与 WSS 归一化实验；本文是这一小矩阵的状态真源。
>
> 相关入口：[WSS 训练实验跟踪](WSS最小化_训练实验跟踪.md) · [WSS 代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md) · [`training_wss_min`](../../training_wss_min/README.md)

> **v4 当前状态（2026-07-18 回填）**：`10473–10478` 均已完成400 epoch和 `ckpt_best/last` 的 test27 legacy-vertex 评估；预注册主结果一律仍取 `ckpt_best(train_loss)`。PointNet 分支以 **P2V** 为本次单 seed 优先候选，PointNet++ 分支以 **Q1V** 为优先候选；Q2V 的 SEP 结果混合、Q3V 的随机 SA center 整体回退，均不进入下一步优先确认。`10475/Q0` 已通过 PostView 验证 `27/27`；另五组均在旧 exporter 的严格面积调用处止于 `AAA/ruputer/SHI_YUN_XI`，已有20个病例目录但未形成批次验证，故只需修复后 export-only 补齐，不得重训或误写成面积结果。面积六组 `P1/P2/Q1/Q2/Q3/Q4` 仍因严格映射仅127/133通过而冻结。没有重复提交 `9169/9170`；ILO-before41 未进入本段 Phase-V 原106 split，但已按下方 §0D 的独立 Q2V/ILO 数据矩阵协议完成扩容实验。

## 0Q. SAME-H2 锚定后的 `log(local_radius)` 输入臂（2026-07-29｜Jobs `11032→11033_[0-1]`｜✅2/2 完成，处理臂 Go）

用户明确暂不做多 seed，并以 §0P 的 `★ LSA2 SAME H2 q90 λ0.20 s1234` 为后续锚点。为避免把已观察到的 CUDA 轨迹漂移误判为特征收益，本轮保留精确同-seed并发 control：

| Array | ID | 输入 | 其他协议 |
|---:|---|---|---|
| 0 | `lsa2_h2_same_repro_s1234` | 原6D `xyz + abscissa + radius + curvature` | 冻结 SAME-H2 全部设置 |
| 1 | `lsa2_h2_logradius_s1234` | 原6D末尾新增 natural-log `log_local_radius` | 与 control 完全相同 |

处理臂不替换原始 `local_radius`，LocalGeoPE 仍读取索引 `[3,4,5]`，因此唯一有效变量是输入维度 `6→7` 与对应冻结统计路径。新列按 control106 train-only 的3,208,800点计算，均值/标准差为 `2.116623/0.611423`；无 RCR、面积、H1 组合或新 seed。

正式 Gate 只用 treatment − concurrent control：`Δphysical R²_cb≥+0.012` 或 `Δhigh-WSS nRMSE≤-0.002`，并要求 `Δnormalized R²_cb≥-0.01`、`ΔMAE_cb≤+0.05 Pa`、`ΔIoU≥-0.005`。Job `11032` 的 CUDA 前后向、严格 checkpoint 重载和 full/chunk 一致性门禁通过；`11033_0/1` 均完成400 epoch、best/last checkpoint、best/last test36 和 36 例逐病例导出。

| 臂 | best epoch | R²_cb | ΔR² | normalized R²_cb | Δnormalized | MAE (Pa) | ΔMAE | high-WSS nRMSE | ΔnRMSE | top10 IoU | ΔIoU | AG / AAA / ILO R² | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| H2 concurrent control | 378 | 0.3070 | — | 0.6397 | — | 2.5580 | — | 0.06906 | — | **0.1916** | — | 0.325 / 0.248 / 0.313 | 并发基准 |
| **H2 + log(radius)** | 378 | **0.3506** | **+0.0436** | **0.6440** | **+0.0042** | **2.5238** | **-0.0342** | **0.06611** | **-0.00295** | 0.1887 | -0.0029 | **0.363 / 0.271 / 0.377** | **Go** |

处理臂同时满足两条主门，normalized R²、MAE、top10 IoU 保护线全部通过，AG/AAA/ILO 均正向。相对历史 ★ H2 的 \(R^2_{cb}\) 仍高 `+0.0267`，但历史结果只作参考；正式判断以并发 control 为准。

逐病例 R² 均值差为 `-0.0020`，95%CI `[-0.0326,+0.0281]`，16/20 改善/退化；病例 high-WSS R² 为23/13，95%CI仍跨零。因此结果足以按预注册规则晋级工程主线，但仍是单 seed、复用 test36、train-loss 选模的开发结论。

**终裁**：`lsa2_h2_logradius_s1234` 晋级为新的正式开发锚点；下一轮保留 SAME-H2、L-SA2、原始半径和 LocalGeoPE，并默认加入 train-only 标准化 `log(local_radius)`。暂不补多 seed，不回到 RCR、面积、IND 或 H1+H2。结构化真源为 `training_wss_min/preflight/lsa2_h2_logradius_matrix_20260729_results_{analysis.json,summary.csv}` 与同前缀 `paired_case_stats.csv`。

## 0P. REG-P10-LSA2 历史正式锚点：SAME/IND × MSE/H1/H2（2026-07-29｜Jobs `11019→11020_[0-5]%3`｜✅6/6 完成）

O0 只是 RCR Oracle 的 geometry-only 配对基准，并非当前最优的非 RCR 模型。正式开发锚点改为 `REG-P10 + L-SA2 s1234`：S3 `PointNeXt-R + LocalGeoPE`、DropPath0.10，且只在 SA2 启用 4-head Local Transformer。历史 best epoch 378 的 test36 指标为 physical `R²_cb=0.3171`、normalized `R²_cb=0.6425`、MAE `2.5527 Pa`、high-WSS nRMSE `0.06808`、top10 IoU `0.1940`，是当前不含 RCR 的最高精度配置；其单 seed 和复用 test36 边界继续保留。

本轮用并发 SAME MSE 重训消除历史运行时差异，并形成完整 2×3：

| Array | ID | Query | 目标 | 主要比较 |
|---:|---|---|---|---|
| 0 | `lsa2_same_mse_s1234` | SAME | MSE | 并发锚点 |
| 1 | `lsa2_same_h1_bce_q90_lam020_s1234` | SAME | H1 | − SAME MSE |
| 2 | `lsa2_same_h2_pinball_q90_lam020_s1234` | SAME | H2 | − SAME MSE |
| 3 | `lsa2_ind_mse_s1234` | IND | MSE | − SAME MSE |
| 4 | `lsa2_ind_h1_bce_q90_lam020_s1234` | IND | H1 | − IND MSE；并补 IND−SAME H1 |
| 5 | `lsa2_ind_h2_pinball_q90_lam020_s1234` | IND | H2 | − IND MSE；并补 IND−SAME H2 |

除 `query_mode` 和目标相关字段外，六臂均冻结 mixed `138/0/36`、random5000、seed1234、400 epoch、train-loss 选模、`legacy_vertex` 及 REG-P10-LSA2 结构；`case_features_path=null`，不含 RCR、面积指标或多 seed。SAME 与 IND 都走常规 support→query 插值；二者差异是 SAME 重用 support 坐标/索引，IND 独立抽取 off-support query。

Gate：H1 `Δtop10 IoU≥+0.020`；H2 `Δhigh-WSS nRMSE≤-0.002`；共同保护线 `Δnormalized R²_cb≥-0.01`、`ΔMAE_cb≤+0.05 Pa`。IND 主门 `Δphysical R²_cb≥+0.012`，并要求 normalized R²、MAE、high-WSS nRMSE 和 IoU 不越过预注册保护线。H1/H2 不组合。

配置生成、静态/CUDA 审计和提交合同通过。Job `11019` 与数组 `11020_0–5` 均 `COMPLETED (0:0)`；六臂全部完成 400 epoch、best/last checkpoint、best/last test36 全云评估和 36 例 CSV。

| 臂 | 对照 | R²_cb | ΔR² | normalized R²_cb | Δnormalized | MAE (Pa) | ΔMAE | high-WSS nRMSE | ΔnRMSE | top10 IoU | ΔIoU | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SAME MSE | — | 0.2862 | — | 0.6390 | — | 2.5892 | — | 0.07040 | — | 0.1786 | — | 并发基准 |
| SAME H1 | SAME MSE | 0.3050 | +0.0188 | 0.6401 | +0.0011 | 2.5589 | -0.0303 | 0.06866 | -0.00174 | 0.1962 | +0.0175 | H1 No-Go |
| SAME H2 | SAME MSE | **0.3239** | **+0.0376** | **0.6453** | **+0.0063** | **2.5371** | **-0.0521** | 0.06835 | **-0.00205** | 0.1800 | +0.0014 | **H2 Go；候选锚点待复跑** |
| IND MSE | SAME MSE | 0.2876 | +0.0014 | 0.6367 | -0.0024 | 2.5825 | -0.0067 | 0.06991 | -0.00049 | 0.1921 | +0.0135 | IND No-Go |
| IND H1 | IND MSE | 0.3021 | +0.0145 | 0.6373 | +0.0007 | 2.5687 | -0.0138 | 0.06932 | -0.00059 | 0.1903 | -0.0018 | H1 No-Go |
| IND H2 | IND MSE | 0.3198 | +0.0322 | 0.6390 | +0.0023 | 2.5709 | -0.0117 | **0.06784** | **-0.00207** | **0.2017** | +0.0096 | H2 Go；IND 不晋级 |

SAME-H2 的 AG/AAA/ILO R² 分别比 SAME MSE 提高 `+0.0341/+0.0140/+0.0585`；病例 high-WSS R² 差的 95%CI 为 `[+0.0014,+0.5319]`，best/last 结论一致。IND-H2 相对 SAME-H2 虽提高 IoU `+0.0217`，但 R² `-0.0041`、normalized R² `-0.0063`、MAE `+0.0337 Pa`，不满足 IND 晋级条件。另有并发 SAME MSE 相对历史同配置 L-SA2 的 `ΔR²_cb=-0.0309`，因此不得跨运行拿历史锚点做 Gate。

**终裁**：关闭 IND、H1 和 H1+H2 组合；SAME-H2 q90 pinball λ0.20 以“单 seed、复用 test36、主门仅多约 0.00005”的边界进入上方 §0Q 的同-seed H2 control / +`log(local_radius)` 单变量实验。该处理臂已过门并替代其成为新开发锚点。结构化真源为 `training_wss_min/preflight/regp10_lsa2_objective_ind_matrix_20260728_results_{analysis.json,summary.csv}` 与同前缀 `paired_case_stats.csv`；工作簿已回填并保持 5 张表、6 页。

## 0O. O0 geometry-only 热点/高 WSS 单变量矩阵（2026-07-28 回填｜Jobs `11012→11013_[0-1]%2`｜✅2/2 完成）

不继续 RCR 多种子或 RCR 架构；O0 geometry-only S3 `PointNeXt-R + LocalGeoPE` 重新成为执行父模型。为避免重复 target-weight/raw-Huber 历史路线，本轮只做两个新辅助目标：

| ID | 相对 O0 的唯一变化 | 核心指标 | Gate |
|---|---|---|---|
| H1 | 第二输出通道 + 每病例 q90 top10 balanced BCE，λ=0.20 | top10 IoU | ΔIoU ≥ +0.02 |
| H2 | MSE + q=0.90 pinball，λ=0.20 | high-WSS nRMSE | ΔnRMSE ≤ -0.002 |

共同保护线为 `Δnormalized R²_cb ≥ -0.01`、`Δphysical MAE_cb ≤ +0.05 Pa`。mixed `138/0/36`、seed1234、random5000/SAME、400 epoch、train-loss 选模、骨干和 `legacy_vertex` 均冻结。H1 的辅助 logit 不替代 WSS 输出，所有正式指标仍读取第一回归通道。

Jobs `11012`、`11013_0/1` 均 `COMPLETED (0:0)`；主结果固定取 `ckpt_best(train_loss)`，完整性审计 3/3 通过：

| 臂 | best epoch | R²_cb | ΔR²_cb | normalized R²_cb | Δnormalized | MAE_cb (Pa) | ΔMAE | high-WSS nRMSE | ΔnRMSE | top10 IoU | ΔIoU | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| O0 | 350 | 0.2756 | — | 0.6300 | — | 2.5974 | — | 0.07068 | — | 0.1772 | — | 基准 |
| H1 | 378 | **0.2990** | **+0.0234** | **0.6336** | +0.0035 | **2.5843** | -0.0131 | **0.06935** | -0.00133 | **0.1943** | **+0.0170** | **No-Go** |
| H2 | 350 | 0.2947 | +0.0190 | 0.6274 | -0.0026 | 2.5917 | -0.0057 | 0.06996 | **-0.00072** | 0.1818 | +0.0046 | **No-Go** |

H1/H2 的保护线和三域 R² 均未退化，但各自主终点没有达到预注册门槛；病例配对 ΔIoU 的 95%CI 均跨零，best/last 结论一致。**终裁：两臂不组合，H2 不补相邻 q/λ；O0 不再作为后续开发父模型，后续转入上方 §0P 的 REG-P10-LSA2 完整矩阵。**结构化真源：`training_wss_min/preflight/o0_hotspot_tail_matrix_20260728_results_{analysis.json,summary.csv}` 与同前缀 `paired_case_stats.csv`。

## 0N. S3 RCR 边界条件信息上限 Oracle（2026-07-28 回填｜Jobs `11008→11009_[0-2]%3`｜✅3/3 完成）

共同父模型为 S3 `D2-K64 + PointNeXt-R + LocalGeoPE`，协议冻结为 mixed `138/0/36`、seed1234、random5000/SAME、400 epoch、train-loss 选模和 `legacy_vertex`。唯一主变量是病例条件：O0 无额外条件，O1a 使用四出口真实 \(R_1/R_2/C\) 的 12 维 log 特征，O2 使用 train/test 内按分域分层打乱的同维 RCR。

| 臂 | 输入 | R²_cb | ΔR²_cb | MAE (Pa) | high-WSS nRMSE | top10 IoU | p99 幅值比 | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| O0 | geometry-only | 0.2756 | — | 2.5974 | 0.07068 | 0.1772 | 0.3429 | 配对基准 |
| O1a | true RCR | **0.3561** | **+0.0805** | **2.4620** | **0.06728** | **0.2109** | **0.4356** | **Go-to-follow-up** |
| O2 | shuffled RCR | 0.2723 | -0.0033 | 2.6053 | 0.07148 | 0.1840 | 0.3240 | 负对照≈O0 |

O1a − O0 的病例均值 ΔR² 95%CI `[+0.0127,+0.1021]`，O2 − O0 为 `[-0.0622,+0.0356]`；O1a − O2 的 \(R^2_{cb}\) 差为 `+0.0838`。因此真实 RCR 的收益不是“多 12 个输入维度”或域标签伪增益。三域 R² 均改善（AG/AAA/ILO `+0.110/+0.028/+0.089`），但 high-WSS Spearman 不变，说明 RCR 主要补病例级幅值/条件信息，局部热点排序仍需另解。

本组只授权条件信息方向进入多种子或新协议确认；单种子、复用 test36 和 train-loss 选模不支持最终泛化声称。详表与配置见 [训练实验跟踪](WSS最小化_训练实验跟踪.md) 和 [高值区域优化方案 §4.2](_archive/WSS最小化/WSS高值区域预测优化方案.md)。

## 0M. REG-P10 静态几何约束 EdgeConv 矩阵（2026-07-28 回填｜Jobs `10993→10994_[0-2]%3`｜✅3/3 完成）

为检验 UDGCNN/DGCNN 的局部邻点差分是否能补充 LocalGeoPE，本组继续以 `s3reg_droppath010_s1234.json` 为唯一父模型 REG-P10，冻结 mixed `138/0/36`、random5000/SAME、`125/125/32` centers、`64/16/16` 邻域、PointNeXt-R `(1,1,0)`、7D LocalGeoPE、DropPath `0.10`、seed1234、400 epoch 和 train-loss 选模。新增模块只在既有 SA 几何邻域内计算残差消息 `[x_i, x_j-x_i, Δp_ij/r]`，不改变 center、邻接关系、解码器或损失；本轮**没有开启特征空间动态图重连**。

| ID | 相对 REG-P10 的唯一变化 | 配置 | 状态 |
|---|---|---|---|
| `EC-SA1` | `edgeconv_stages=[1]` | `regp10_edgeconv_sa1_s1234.json` | ✅完成 |
| `EC-SA2` | `edgeconv_stages=[2]` | `regp10_edgeconv_sa2_s1234.json` | ✅完成 |
| `EC-SA12` | `edgeconv_stages=[1,2]` | `regp10_edgeconv_sa12_s1234.json` | ✅完成 |

`edgeconv_stages` 使用 1-based 编号，默认空列表时不实例化模块、不增加 state_dict key。108 项回归测试全部通过；GPU 门禁验证了三个配置的目标层执行、有限梯度、严格 checkpoint 重载以及 full/chunk 推理一致性。Job `10993` 与数组 `10994_[0-2]` 均 `COMPLETED (0:0)`；三臂完成 400 epoch、best/last checkpoint、best/last test36 全云评估和 36 例逐病例 CSV。父模型加三臂的产物完整性审计 `4/4` 通过，预注册主结果固定取 `ckpt_best(train_loss)`（三臂 best epoch 均为 378）。

| 处理臂 | R²_cb | ΔR²_cb | ΔMAE / ΔRMSE (Pa) | Δhigh-WSS / ΔIoU | ΔAG / ΔAAA / ΔILO | 病例均值 ΔR² 95%CI；胜/负 | 单种子判定 |
|---|---:|---:|---:|---:|---:|---|---|
| REG-P10 父模型 | 0.2957 | — | 2.5858 / 6.3283（绝对值） | -0.4127 / 0.1883（绝对值） | 0.2954 / 0.2366 / 0.3195（绝对值） | — | 锚点 |
| `EC-SA1` | 0.2904 | -0.0053 | -0.0001 / +0.0237 | -0.0210 / -0.0014 | -0.0095 / +0.0172 / -0.0167 | -0.0013 `[-0.0264,+0.0222]`；17/19 | No-Go |
| `EC-SA2` | 0.2935 | -0.0023 | **-0.0240** / +0.0102 | -0.0256 / **+0.0122** | +0.0143 / +0.0168 / **-0.0324** | +0.0083 `[-0.0170,+0.0332]`；21/15 | No-Go |
| `EC-SA12` | 0.2619 | **-0.0338** | +0.0207 / +0.1502 | **-0.0925** / +0.0084 | -0.0108 / -0.0089 / **-0.0753** | +0.0059 `[-0.0185,+0.0312]`；18/18 | No-Go |

**终裁**：沿用 REG-P10 Transformer 矩阵的预注册 Gate（`ΔR²_cb≥+0.012`、`ΔMAE≤+0.03 Pa`、high-WSS 不下降超过 `0.01`、任一域 R² 不下降超过 `0.02`），三臂均未晋级。`EC-SA2` 是三者中最接近父模型的一臂，MAE、top10 IoU、AG/AAA 和病例胜负方向为正，但主指标 `R²_cb=-0.0023`、high-WSS `-0.0256` 且 ILO `-0.0324`，不能因局部指标改善而事后晋级；`EC-SA12` 明显退化，说明两层同时叠加没有协同。当前静态 EdgeConv 设计判 No-Go，**不继续补特征空间动态图或多种子**。结构化真源为 `training_wss_min/preflight/regp10_edgeconv_matrix_20260727_results_{analysis.json,summary.csv}` 与同前缀 `paired_case_stats.csv`；工作簿已回填。

## 0L. D2 c125×k64 PNXR+GeoPE 的 SA1/SA2 局部 Transformer 对照（2026-07-27 回填｜Jobs `10979→10980_[0-1]%2`｜✅2/2 完成）

为让 REG-P10/mixed test36 的 SA2 信号能在原始 AG/AAA 固定协议上得到同结构对照，本组改用已完成的 `D2 c125×k64 + PointNeXt-R + LocalGeoPE` 作为唯一父模型。父配置为 `pointnetpp_d2_c125_k64_pnxr_geope_20260726/d2_c125_k64_pnxr_geope.json`，冻结 AG/AAA `106/0/27`、train106 统计、6D xyz+geom、7D LocalGeoPE、random5000/SAME、`125/125/32` centers、`64/16/16` 邻域、PointNeXt-R `(1,1,0)`、无 DropPath、seed1234、400 epoch 和 train-loss 选模。

| ID | 相对固定 D2 PNXR+GeoPE 的唯一变化 | 配置 | 状态 |
|---|---|---|---|
| `D2-L-SA1` | `local_transformer_stages=[1]` | `d2_c125_k64_pnxr_geope_localtf_sa1_s1234.json` | ✅完成 |
| `D2-L-SA2` | `local_transformer_stages=[2]` | `d2_c125_k64_pnxr_geope_localtf_sa2_s1234.json` | ✅完成 |

两份配置的静态逐字段审计 `2/2` 通过，观察到的差异只有 `name/notes/model.local_transformer_stages`；SA1/SA2 模块只在目标层实例化，参数量分别为 `473,991/572,999`。正式 GPU 门禁 Job `10979` 与两条训练任务均 `COMPLETED (0:0)`；两臂均完成 400 epoch、best/last checkpoint、best/last test27 全云评估和 27 例逐病例 CSV，配置哈希、有限值和产物完整性审计全部通过。预注册主结果固定取 `ckpt_best(train_loss)`。

| 处理臂 | 物理 R²_cb | Δ物理 R²_cb | 归一化 R²_cb | Δ归一化 | ΔMAE_cb / ΔRMSE_cb (Pa) | Δhigh-WSS / ΔIoU | ΔAG / ΔAAA | 病例均值 ΔR² 95%CI；胜/负 | 主结果判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| D2 PNXR+GeoPE 父模型 | 0.2975 | — | 0.6712 | — | — | — | — | — | 锚点 |
| `D2-L-SA1` | 0.2683 | **-0.0293** | 0.6491 | -0.0222 | +0.0469 / +0.1298 | -0.0753 / -0.0158 | -0.0217 / -0.0369 | -0.0184 `[-0.0466,+0.0090]`；12/15 | **No-Go** |
| `D2-L-SA2` | 0.2990 | +0.0015 | 0.6581 | **-0.0132** | -0.0160 / -0.0069 | -0.0105 / +0.0073 | -0.0012 / +0.0042 | -0.0045 `[-0.0437,+0.0291]`；17/10 | **持平，不晋级** |

**终裁**：SA1 局部 Transformer 在 fixed test27 上各主要维度一致退化，明确 No-Go。SA2 的物理 `R²_cb` 仅 `+0.0015`，低于工程筛选量级；归一化 R² 与 high-WSS 同时回退、病例 CI 跨零，因此不构成可复现提升。SA2 `ckpt_last` 相对父模型 last 的物理 `R²_cb` 为 `+0.0198`，但 last 仅作敏感性检查，不能替换预注册的 train-loss-selected best；不据此事后晋级。该结果说明 REG-P10/mixed test36 的 SA2 单种子正信号没有在“无 DropPath、无 ILO、fixed test27”的 D2 协议中直接复现，可能涉及正则化/数据协议交互，当前不再扩展 D2 local Transformer。结构化真源为 `training_wss_min/preflight/d2_c125_k64_pnxr_geope_transformer_20260726_results_{analysis.json,summary.csv}` 与同前缀 `paired_case_stats.csv`；工作簿已回填。

## 0K. REG-P10 局部/全局 Transformer 完整矩阵（2026-07-26｜Jobs `10967→10968_[0-7]%4`｜✅8/8 完成）

当前按讨论将 `S3-GEOPE + DropPath 0.10, seed=1234` 固定为本轮工作锚点（简称 **REG-P10**），父配置为 `pointnetpp_s3_regularization_20260724/s3reg_droppath010_s1234.json`。split/stats、mixed `138/0/36`、random5000/SAME、`125/125/32` centers、`64/16/16` 邻域、PointNeXt-R `(1,1,0)`、7D LocalGeoPE、400 epoch 和 train-loss 选模全部冻结。

**SA3 是否与旧 attention 重复**：若指 SA3 输出的 32 个 coarse centers 之间做全局 Transformer，答案是**重复**。已有 `CoarseGlobalBlock` 已执行病例内全局多头自注意力、相对 xyz/距离 bias、FFN 与残差缩放，结构语义就是“SA3 全局 Transformer”。因此不再新增第二套同义模块；全局对照直接复用 `coarse_attention=true`。新代码补充的是机制不同的局部模块：在每个 SA center 的邻域内部，以逐边 MLP+LocalGeoPE 后的消息为 token，在 max 聚合之前执行 MHA+FFN；不同 center 和不同病例之间不互相注意。`local_transformer_stages=[3]` 表示“SA3 每个局部分组内部、聚合前”的局部 Transformer，并不等于 32 centers 间的全局建模，所以矩阵同时保留 local-SA3 与 global-SA3 两个机制对照。

| ID | 相对 REG-P10 的唯一有效变化 | 回答的问题 | 配置 |
|---|---|---|---|
| `L-SA1` | `local_transformer_stages=[1]` | 最细局部邻域是否受益 | `regp10_localtf_sa1_s1234.json` |
| `L-SA2` | `local_transformer_stages=[2]` | 中尺度局部邻域是否受益 | `regp10_localtf_sa2_s1234.json` |
| `L-SA3` | `local_transformer_stages=[3]` | 粗尺度局部分组内部是否受益 | `regp10_localtf_sa3_s1234.json` |
| `L-SA12` | `local_transformer_stages=[1,2]` | SA1 与 SA2 的局部关系建模是否协同 | `regp10_localtf_sa12_s1234.json` |
| `L-SA13` | `local_transformer_stages=[1,3]` | 细尺度与粗尺度局部模块是否协同 | `regp10_localtf_sa13_s1234.json` |
| `L-SA23` | `local_transformer_stages=[2,3]` | 中尺度与粗尺度局部模块是否协同 | `regp10_localtf_sa23_s1234.json` |
| `L-SA123` | `local_transformer_stages=[1,2,3]` | 三层局部 Transformer 全开是否受益 | `regp10_localtf_sa123_s1234.json` |
| `G-SA3` | `coarse_attention=true` | REG-P10 上是否复现/增强旧 SA3 全局信号 | `regp10_globaltf_sa3_s1234.json` |

局部 stage 使用 **1-based** 编号；空列表完全关闭且不生成新参数/checkpoint key。7 个局部非空子集构成完整 `2³−1` 因子矩阵，另加 1 个旧机制的 SA3 全局对照；本轮不把 local 与 global 同时开启，避免混淆两类机制。配置读取与逐字段静态审计 `8/8` 通过，全测试集 `102/102` 通过；默认关闭时旧 REG-P10 checkpoint 严格重载无缺失/多余 key。正式 GPU 门禁 Job `10967` 与训练/评估数组 `10968_[0-7]%4` 均 `COMPLETED (0:0)`；8 臂全部完成 400 epoch、best/last checkpoint、test36 全云指标和 36 例逐病例 CSV，配置哈希、有限值、best/last 产物完整性均通过。预注册结果固定取 `ckpt_best(train_loss)`。

| 处理臂 | R²_cb | ΔR²_cb | ΔMAE / ΔRMSE (Pa) | Δhigh-WSS / ΔIoU | ΔAG / ΔAAA / ΔILO | 病例均值 ΔR² 95%CI；胜/负 | 单种子判定 |
|---|---:|---:|---:|---:|---:|---|---|
| REG-P10 父模型 | 0.2957 | — | 2.5858 / 6.3283（绝对值） | -0.4127 / 0.1883（绝对值） | 0.2954 / 0.2366 / 0.3195（绝对值） | — | 锚点 |
| `L-SA1` | 0.2851 | -0.0106 | -0.0005 / +0.0474 | -0.0425 / +0.0027 | +0.0008 / +0.0205 / -0.0440 | -0.0101 `[-0.0404,+0.0181]`；20/16 | No-Go |
| **`L-SA2`** | **0.3171** | **+0.0214** | **-0.0332 / -0.0968** | **+0.0453 / +0.0058** | **+0.0105 / +0.0198 / +0.0341** | +0.0114 `[-0.0101,+0.0324]`；19/17 | **唯一过 Gate；补 seeds 7/2025** |
| `L-SA3` | 0.2798 | -0.0159 | +0.0000 / +0.0712 | -0.0523 / +0.0015 | +0.0014 / +0.0274 / -0.0641 | +0.0004 `[-0.0251,+0.0255]`；15/21 | No-Go |
| `L-SA12` | 0.3021 | +0.0063 | -0.0101 / -0.0285 | -0.0088 / -0.0017 | +0.0222 / +0.0017 / -0.0065 | +0.0023 `[-0.0290,+0.0322]`；18/18 | 正向但未过主 Gate |
| `L-SA13` | 0.2952 | -0.0005 | -0.0072 / +0.0024 | -0.0209 / +0.0071 | +0.0017 / +0.0094 / -0.0096 | +0.0088 `[-0.0122,+0.0306]`；19/17 | No-Go |
| `L-SA23` | 0.2909 | -0.0049 | +0.0139 / +0.0218 | -0.0082 / +0.0012 | -0.0023 / +0.0206 / -0.0251 | -0.0208 `[-0.0664,+0.0162]`；18/18 | No-Go |
| `L-SA123` | 0.2873 | -0.0084 | +0.0199 / +0.0376 | -0.0261 / +0.0028 | +0.0015 / +0.0041 / -0.0273 | -0.0181 `[-0.0474,+0.0090]`；17/19 | No-Go |
| `G-SA3` | 0.2909 | -0.0048 | +0.0068 / +0.0215 | -0.0246 / -0.0110 | -0.0061 / +0.0184 / -0.0197 | -0.0026 `[-0.0316,+0.0249]`；20/16 | No-Go |

**终裁**：按预注册 Gate（`ΔR²_cb≥+0.012`、`ΔMAE≤+0.03 Pa`、high-WSS 不下降超过 `0.01`、任一域 R² 不下降超过 `0.02`），只有 **`L-SA2`** 四项全部通过。它是本轮唯一进入 seeds `7/2025` 配对确认的臂，但病例均值 CI 仍跨零，当前只能写成“单种子强筛选信号”，不能写成稳定泛化提升。`L-SA12` 虽方向为正，但 `+0.0063` 未达到主门槛，不与 SA2 一起扩展。SA1/SA3 单层、全部含 SA3 的局部组合和三层全开均未受益，说明局部 Transformer 不是“层数越多越好”，收益集中在 **SA2 中尺度邻域**。`G-SA3` 在 REG-P10 上回退，也说明旧 coarse global attention 不能直接叠加到当前锚点。结构化真源为 `training_wss_min/preflight/regp10_transformer_matrix_20260726_results_{analysis.json,summary.csv}` 与同前缀 `paired_case_stats.csv`；工作簿已同步回填。

## 0J. D2-K64 ILO 两协议与结构模块矩阵（2026-07-23｜Jobs `10837/10838/10843/10844`｜9/9 完成）

> **固定完整数据对照结果（2026-07-26，`10958/10959` ✅完成）**：为避免把后续 mixed138/test36 的结构结论外推到完整 AG/AAA 数据协议，`d2_c125_k64_pnxr_geope` 锚定 D2 c125×k64 的 `106/0/27`、random5000/SAME、`125/125/32`、`64/16/16`、6D xyz+geom、seed1234 与 400 epoch；仅将残差块改为 PointNeXt-R `(1,1,0)` 并开启 7D LocalGeoPE。首轮门禁 `10956` 仅因 smoke harness 对空 `feature_stats_path` 的错误处理而失败，未执行模型；修正后 `10958/10959` 均 clean completed。相对原 D2：`R²_cb 0.2628→0.2975`（`+0.0347`）、MAE/RMSE `-0.1313/-0.1537 Pa`、high-WSS `+0.0805`、IoU `+0.0273`，AG/AAA `+0.0526/+0.0181`，病例 R² 19/8、95% CI `[+0.0134,+0.0837]`。这是一条完整数据协议上的单种子正信号；只与原 D2 配对，不与 mixed S3 直接排名。

> **正则化补齐结果（2026-07-25，`10923/10924_[0-19]` ✅20/20）**：20 臂均为 S3-GEOPE seed1234、mixed `138/0/36` 的严格配对。HeadDrop=.15 最优（`R²_cb=0.2983`，`Δ=+0.0059`，但 MAE `+0.0105 Pa`）；DropPath=.05/.10 为温和正信号（`+0.0017/+0.0034`），.15 明显退化（`-0.0179`）；NeighborDrop 全强度均为负。两两交叉没有超过 HeadDrop=.15，故不组合、不改 S3 主锚。该批是单 seed 的历史 test36 筛选，只将 HeadDrop=.15、DropPath=.10 保留为互不组合的多 seed 确认候选。

### 纯 XYZ 补充对照（2026-07-24｜`10903→10904_[0-2]` ✅完成）

| 新臂 | 锚定配置 | 唯一变化 | 协议 | 状态 |
|---|---|---|---|---|
| `d2_c125_k64_xyz` | D2 c125×k64 | `input_features: xyzgeom → xyz` | AG/AAA `106/0/27`，其余字段冻结 | ✅完成 |
| `m1_d2k64_mixed138_test36_xyz` | M1/S0 D2-K64 | `input_features: xyzgeom → xyz` | mixed `138/0/36`，其余字段冻结 | ✅完成 |
| `s2_d2k64_pnxr_mixed_xyz` | S2 D2-K64 + PointNeXt-R | `input_features: xyzgeom → xyz` | mixed `138/0/36`，残差块 `(1,1,0)` 保留 | ✅完成 |

静态逐字段审计限定差异为 `name`、`notes` 和 `data.input_features`，已 `3/3` 通过；`10903` 与 `10904_[0-2]` 均已完成 400 epoch、best/last checkpoint 与 test 全云评估。

### 结果回填（2026-07-25｜`10903/10904_[0-2]` ✅完成）

| 对照（xyz+geom → xyz） | physical R²_cb | ΔR²_cb | ΔMAE Pa | ΔRMSE Pa | Δhigh-WSS R² | 结论 |
|---|---:|---:|---:|---:|---:|---|
| D2 c125×k64，AG/AAA test27 | 0.2047 | -0.0581 | +0.1812 | +0.2492 | -0.0833 | No-Go |
| M1/S0，mixed test36 | 0.1656 | -0.0958 | +0.3353 | +0.4073 | -0.0217 | No-Go |
| S2 PointNeXt-R，mixed test36 | 0.1798 | -0.0730 | +0.2819 | +0.3109 | -0.0122 | No-Go |

M1/S2 的严格病例平均 ΔR² 95% CI 分别为 `[-0.8510,-0.0748]` 与 `[-0.6287,-0.0753]`，支持去除几何特征的实质性退化；D2 的同一 CI 为 `[-0.1113,+0.0094]`，但聚合 R²、MAE、RMSE、high-WSS 和 18/27 病例同时指向负向。纯 XYZ 内的 S2−M1 `ΔR²_cb=+0.0142` 且 CI 跨零，不能作为残差结构补回几何信息的证据。故关闭 xyz-only 扩展，xyz+geom 继续作为这三条协议的冻结输入。

### 纯 XYZ 的 4D LocalGeoPE 补偿（2026-07-25｜`10929→10930` ✅完成）

S2-PNXR 纯 XYZ 上仅开启 LocalGeoPE，并将边编码从原 xyz+geom 的 7D `Δxyz/r + distance + Δ(abscissa,radius,curvature)` 改为不读取语义属性的 4D `Δxyz/r + distance`。该分支 `R²_cb=0.2469`：相对纯 XYZ `+0.0671`（MAE/RMSE `-0.2833/-0.2853 Pa`，病例 CI `[+0.1062,+0.5750]`），但相对 xyz+geom 的 S2 `0.2528` 仍 `-0.0059`，病例 CI `[-0.0813,+0.0540]` 跨零。结论是相对位置编码能补回大部分损失，但不取代原含几何属性的输入；既有 7D GeoPE 路径保持不变。

### S3 衍生结构 7 月 24 日结果回填（✅13/13 完成）

| 相对同 seed S3-GEOPE | ΔR²_cb（种子/均值） | 平均 ΔMAE / Δhigh-WSS R² | 判定 |
|---|---|---|---|
| Head dropout 0.10 | `-0.0089/+0.0190/+0.0068`；`+0.0057` | `-0.0135 / +0.0163` | Weak-Go，2/3 正 |
| DropPath 0.05 | `+0.0017/+0.0043/+0.0206`；**`+0.0089`** | **`-0.0176 / +0.0162`** | 最强 Weak-Go，3/3 正 |
| DropPath 0.10 | `+0.0034/+0.0057/+0.0138`；`+0.0076` | `-0.0115 / +0.0159` | Weak-Go，3/3 正 |
| NeighborDrop 0.05 | `-0.0296/+0.0037/+0.0129`；`-0.0043` | `+0.0063 / -0.0021` | No-Go |
| SA3 coarse attention（seed1234） | `0.2924→0.3007`；`+0.00835` | `+0.0114 / +0.0353` | 单种子 Weak-Go；须补 seeds `7/2025` |

四个正则化臂都未越过预注册 `mean ΔR²_cb≥+0.015` 的正式 Go 门槛，不能取代 S3 或彼此组合；仅把 DropPath 0.05 作为优先确认候选。attention 的病例 CI 跨零、AG 小回退，亦不得晋级为主锚。详见 [训练实验跟踪](WSS最小化_训练实验跟踪.md)；xlsx 已回填「实验矩阵总览」和「汇总对比」，未保留额外专用页。

本轮父模板是 **D2 c125×k64**，不是 c125×k128。F0/F1 只在固定 AG/AAA test27 配对；M0/M1 与 S1–S5 统一使用 AG/AAA/ILO mixed `138/0/36` test36。所有新训练均为 `seed=1234`、400 epoch、train-loss 选模；门禁和训练评估均 `COMPLETED (0:0)`，9 个 run 的 best/last、逐病例 CSV、配置 SHA 与有限性审计全部通过。

| 比较 | 对照 → 处理 | physical R²_cb | Δ | MAE_cb Δ | high-WSS R² Δ | IoU Δ | 病例R²胜/负；均差95% CI | 筛选结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| F1−F0 | AG/AAA train106 → +ILO41，test27不变 | 0.2459 | **-0.0169** | +0.0172 | -0.0245 | +0.0105 | 10/17；`-0.0116 [-0.0340,+0.0118]` | No-Go |
| M1−M0 | mixed test36；ILO zero-shot → ILO32入训 | 0.2614 | -0.0059 | +0.0003 | -0.0150 | **-0.0173** | 15/21；`+0.0071 [-0.0282,+0.0539]` | 总体不晋级 |
| S1−M1 | +M-FEAT | 0.2631 | +0.0018 | +0.0029 | +0.0055 | -0.0025 | 16/20；`-0.0137 [-0.0367,+0.0084]` | 暂不支持 |
| S2−M1 | +PointNeXt-R core | 0.2528 | -0.0086 | +0.0310 | -0.0129 | -0.0055 | 17/19；`-0.0279 [-0.0559,-0.0025]` | No-Go 信号 |
| **S3−S2** | **+LocalGeoPE v1** | **0.2924** | **+0.0396** | **-0.1708** | **+0.0548** | **+0.0157** | **29/7；`+0.0921 [+0.0600,+0.1259]`** | **强晋级候选** |
| S4−S2 | +SA3 coarse attention | 0.2706 | +0.0178 | -0.0363 | +0.0437 | +0.0029 | 19/17；`+0.0175 [-0.0106,+0.0463]` | 弱正候选 |
| S5−S5C | independent query 下 3NN → SEP-Kernel | 0.2733 | +0.0235 | -0.0213 | +0.0633 | +0.0013 | 19/17；`-0.0047 [-0.0316,+0.0211]` | 条件正候选 |

**终裁**：直接加入 ILO 没有带来总体收益；M1 仅把 ILO R² `+0.0109`，同时 AAA `-0.0331`，按域间权衡归档。PointNeXt-R 单独不增益，但与局部几何位置编码组合后，S3 在 AG/AAA/ILO 三域都提高；后续三种子 S3−M1 为 `+0.0310/+0.0059/+0.0303`（3/3正），因此 S3-GEOPE 成为当前工程锚点。完整分析见 [训练实验跟踪](WSS最小化_训练实验跟踪.md) 与 [当前 S3 优化执行计划](WSS最小化_S3-GEOPE锚定_正则化与架构优化执行计划_2026-07-24.md)。

> **后续（2026-07-24 完训）**：`M1/S2/S3 × {7,2025}` 三种子确认完成（S3−M1 `+0.0310/+0.0059/+0.0303`，3/3正）后，以 S3-GEOPE 为父模板跑 **S3 根因矩阵**（门禁 `10857`／array `10858_[0-11]%6`，✅12/12 完训）：针对 ILO 负迁移三根因各开单变量臂——归一化口径 `caliber_{pooled138,casebal,cohortbal}`、域条件 `cohort_onehot`、尾部 `rawhuber02`、探索组合 `cohortbal_onehot`。**终裁：整体 No-Go 但机制证实根因 #1。** S3 父 R²_cb `0.2924/0.2743/0.2713`；无臂跨种子稳定超过 S3（casebal 均值 `+0.0040`/seed1234 `-0.0166`、cohortbal `-0.0018`、cohort_onehot `-0.0084`、rawHuber02/组合单种子皆负）。分域上平衡口径把被 ILO 压掉的 **AAA 稳定抬回**（cohortbal 三种子 `+0.0255/+0.0362/+0.0249`）但 **ILO 相应回落**、AG 持平——AAA↔ILO 容量再分配、总体持平，说明 **LocalGeoPE 已吸收归一化口径在无几何基座上的净收益、二者不叠加**；high-WSS 全负，尾部未解决。S3-GEOPE 仍为参考模型，下一步转尾部/域感知损失加权、不再试全局归一化变体。详见 [训练实验跟踪 §S3-GEOPE 根因矩阵](WSS最小化_训练实验跟踪.md) 与 [代码记录](WSS最小化_代码修改与实验推进记录.md)；xlsx 已回填。

> **当前执行（2026-07-24）**：S3 正则化首波已实现并提交。12 个配置为 `head dropout 0.10 / DropPath 0.05 / DropPath 0.10 / NeighborDrop 0.05 × seeds 1234/7/2025`；每臂只改变一个模型字段，静态差分 12/12 通过。正式门禁 Job `10871`，训练与 best/last test36 评估 Job `10872_[0-11]%4`。当前结果仍只用于历史 test36 同协议工程筛选。

## 0I. SA1 全覆盖、低重叠与 support 矩阵（2026-07-20｜Jobs `10557–10561`｜运行中）

本轮以 **Q1V-10476（vertex-random5000 / SAME / FPS center / `106/0/27`）** 为主基准，不做3-seed；17个新训练任务全部只用 `seed=1234`。Q2V/SEP只作 support 方法的配套敏感性矩阵，不替代Q1V主线。所有新任务继续用历史test27做同协议工程比较，不写成新的独立确认结论。

| 子矩阵 | 复用对照 | 新训练（每项仅1 seed） |
| --- | --- | --- |
| `nsample × width` 2×2 | Q1V `n16_w32` | `n32_w32` / `n16_w64` / `n32_w64` |
| Q1V SA1 grouping | Q1V `ball16` | `ball32`，raw `KNN-8/10`，`KNN-8/10 + coverage repair`，`adaptive_cover` |
| Q1V/SAME support 3×2 | random+ball16；历史 fps-multistart+ball16 | random+adaptive，fixed-FPS+ball/adaptive，fps-multistart+adaptive |
| Q2V/SEP support 3×2 | Q2V random+ball16 | random+adaptive，fixed-FPS+ball/adaptive，fps-multistart+ball/adaptive |

`adaptive_cover` 不再把16当作强制上限：先将每个support点归给最近center，因而SA1 support覆盖必为100%；如某center的主分区大于16，group自然可超过16。再从最近16个center候选中至多增加一个次归属，对每个center pair执行 `floor(min(primary_size)×1/3)` 共享点预算；由此对任意group pair硬保证 `|交集|/min(|G_i|,|G_j|)≤1/3`。raw KNN-8/10保留为负对照，不假定它们全覆盖；coverage-repair版才执行100%覆盖门禁。

fixed-FPS5000与pool8 FPS-multistart5000均先按病例生成并验证持久化索引，Q1V/Q2V共用，不在每个DataLoader worker里重复计算。提交链为：`10557` 随机support几何审计+GPU预检 → `10558_[0-9]%4` 训练；`10559_[0-132]%7` 离线FPS缓存 → `10560` 几何审计+GPU预检 → `10561_[0-6]%4` 训练。正式提交前，adaptive已通过CPU反传、RTX4090 batch8 AMP反传、`500/125/32` center计数与full/chunk query一致性预检；完整几何门禁作为Slurm训练的`afterok`前置，未通过就不会启动对应训练。冻结真源为 `training_wss_min/preflight/sa_grouping_single_seed_{prepared,submission}.json`。

## 0H. QAD 精确Q2V `106/0/27` 对照（2026-07-19｜Jobs `10549→10550`｜**No-Go**）

本轮不直接拿 `85/21` QAD checkpoint 与Q2V横比，而是完整恢复Q2V-10477的train106 stats、random5000/SEP、FPS center、`nsample16/width32`和 `106/0/27` split。复用Q2V/seed1234/interpolate，新训R0 seeds`7/2025`与QAD seeds`1234/7/2025`，共5个任务。这样每个seed的R1−R0只能归因于decoder，同时可与历史Q2V展示同协议绝对分数。

GPU正式预检 `10549` 5/5 passed，数组 `10550_[0-4]` 5/5均 `COMPLETED (0:0)`；五个新run均400 epoch、best/last、27例CSV、配置哈希和有限性检查完整。QAD相对R0仅新增`2,625`参数。

| seed | R0 / QAD normalized R²_cb | Δnormalized | R0 / QAD physical R²_cb | Δphysical | ΔSpearman / IoU / normalized p99比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1234 | 0.62097 / 0.59873 | **-0.02225** | 0.27243 / 0.22807 | -0.04436 | -0.00862 / +0.02331 / -0.06431 |
| 7 | 0.59272 / 0.60015 | +0.00743 | 0.28033 / 0.26645 | -0.01388 | -0.00191 / +0.01637 / +0.04497 |
| 2025 | 0.60092 / 0.60147 | +0.00055 | 0.26170 / 0.26688 | +0.00518 | +0.00021 / -0.00357 / -0.00088 |
| **3seed均值** | **0.60487 / 0.60012** | **-0.00476** | **0.27149 / 0.25380** | **-0.01769** | **-0.00344 / +0.01204 / -0.00674** |

QAD将normalized `R²_cb` 种子标准差从`0.01187`降至`0.00112`，但均值同时下降，不能称为稳健性改进。物理MAE均值基本持平（`-0.00385 Pa`），RMSE均值恶化`+0.01705 Pa`，high-WSS `R²` 由`-0.5269`降至`-0.5361`。三种子平均后的27例配对normalized R²为8胜19负、均差`-0.00630`、95% CI `[-0.02196,+0.00969]`；p99比为5胜22负、均差`-0.02136`、95% CI `[-0.03672,-0.00333]`、Wilcoxon `p=0.00102`。IoU的`+0.01204`不足以抵消R²、RMSE与p99护栏回退。

**结论**：精确Q2V协议下QAD为 **No-Go**，不再做QAD-REG/更小gate，R2-DUAL不以QAD为父实验。就用户重点关注的历史test27 normalized `R²_cb` 而言，Q2V-10477/seed1234的`0.62097`仍高于同seed QAD `0.59873`和QAD三种子均值`0.60012`。该test27已参与多轮历史选择，因此仅用于同协议工程排除，不是新的独立确认。真源见 [analysis JSON](../../training_wss_min/preflight/pointnetpp_qad_q2v_test27_results_analysis.json)、[summary CSV](../../training_wss_min/preflight/pointnetpp_qad_q2v_test27_results_summary.csv)和 `pointnetpp_qad_q2v_test27_{seed_deltas,per_case_deltas}.csv`。

## 0G. PointNet++ QAD-Lite 三种子配对（2026-07-19｜val21｜No-Go for direct R2）

R1以 `n32_w32` 为唯一父实验，只把历3-NN插值decoder改为query-relative gated residual；新增`2,625`参数，不改SA、采样、loss、stats、400 epoch预算或train-loss选模。Job `10540_[0-4]` 5/5 `COMPLETED (0:0)`，连同已有R0/seed1234形成三组严格配对；全部评估只在val21，原test27未访问。

| seed | R0 / R1 normalized R²_cb | Δnormalized R²_cb | R0 / R1 physical R²_cb | ΔSpearman / IoU / normalized p99比 |
| ---: | ---: | ---: | ---: | ---: |
| 1234 | 0.52937 / 0.56612 | **+0.03676** | 0.25892 / 0.25065 | +0.02476 / +0.01012 / -0.03166 |
| 7 | 0.53601 / 0.55877 | **+0.02277** | 0.26739 / 0.28698 | +0.00970 / -0.00562 / +0.01275 |
| 2025 | 0.54864 / 0.53189 | **-0.01675** | 0.26838 / 0.25534 | -0.01384 / -0.00448 / -0.02811 |
| **3seed均值** | **0.53800 / 0.55226** | **+0.01426** | **0.26490 / 0.26432** | **+0.00688 / +0.00001 / -0.01567** |

normalized `R²_cb` 的三种子均值有正向信号且2/3 seed改善，但 `+0.01426` 刚好低于预注册 `+0.015` 主门。物理 `R²_cb` 均值不变，MAE小幅下降`0.0187 Pa`但RMSE无稳定改善；seed2025同时破坏Spearman与RMSE非劣门，p99比在2/3 seed下降，high-WSS `R²` 均值 `-0.6030→-0.6261`。三种子平均的逐例normalized R²差为`+0.01107`，bootstrap 95% CI `[-0.00176,+0.02516]`，13胜8负，不确定性跨零。

因此val21阶段当时只能判定 **R2-DUAL不直接启动**，并曾保留residual幅度惩罚/更小gate作为微变体候选。上方§0H的精确Q2V三种子复核已将这一候选更新为 **No-Go**：不再做QAD-REG/小gate，R2-DUAL不以QAD为父实验。本节仅保留val21的阶段证据，不是当前执行建议。完整真源见 [analysis JSON](../../training_wss_min/preflight/pointnetpp_qad_results_analysis.json)与[summary CSV](../../training_wss_min/preflight/pointnetpp_qad_results_summary.csv)。

<!-- Q2V_ILO_20260718_START -->
## 0D. Q2V 数据扩容与 Point++ 架构矩阵（2026-07-18｜结果已回填）

本轮以 Q2V-10477 为单seed工程锚点，数据归因6组按各自预注册test评估；架构6组只在 grouped dev `85/21` 的val21开发，原test27保持不可见。12组均完成400 epoch与best/last定量评估；主结果固定取 `ckpt_best(train_loss)`。Q2V及本轮所有配置的完整链是 `5000 support→500→125→32`，其中 `500/125/32` 是三层 SA center 数，不是输入点数。

| ID / Job | train/val/test | 主对照与唯一变化 | 物理 / 归一化 R²_cb | 配对差 | 闭环状态 |
| --- | ---: | --- | ---: | ---: | --- |
| `Q2V-10477` / `10477` | `106/0/27` | 原test27唯一历史锚点；D1同test27对照 | 0.2724 / **0.6210** | D1同test27对照 | PostView 27/27；不参与val21排名 |
| `q2v_ilo_d1_fixed_frozen` / `10487_0` | `147/0/27` | Q2V；原test27不变，新增ILO41训练，冻结统计 | 0.2516 / 0.5961 | vs Q2V `-0.0208` | 定量+PostView 27/27 |
| `q2v_ilo_d1_fixed_refit` / `10487_1` | `147/0/27` | D1 frozen；只重算train147 target/feature stats | 0.2543 / 0.6092 | `+0.0027` | 定量+PostView 27/27 |
| `Q2V-zero-shot-D2` / `10490` | `106/0/36` | Q2V checkpoint；只读test36 override | 0.2646 / 0.6004 | D2对照 | 只读评估完成 |
| `q2v_ilo_d2_extended_frozen` / `10487_2` | `138/0/36` | zero-shot；加入ILO32训练，冻结Q2V统计 | 0.2493 / 0.5943 | `-0.0153` | 定量+PostView 36/36 |
| `q2v_ilo_d3_pool2025_control` / `10487_3` | `106/0/36` | D3五层test36控制组，ILO32不参与训练 | **0.2981** / 0.5499 | 控制锚点 | 定量完成；PostView 35/36 |
| `q2v_ilo_d3_pool2025_frozen` / `10487_4` | `138/0/36` | D3 control；只加入ILO32训练并冻结control统计 | 0.2915 / 0.5692 | `-0.0066` | 定量完成；PostView 35/36 |
| `q2v_ilo_d3_pool2025_refit` / `10487_5` | `138/0/36` | D3 frozen；只重算train138统计 | 0.2459 / 0.5637 | `-0.0457` | 定量完成；PostView 35/36 |

数据归因结论：三个加入ILO训练的主配对在物理 `R²_cb` 上均为负差，重算统计也没有稳定补偿；局部出现负例数减少、IoU或ILO-0改善，但与pooled误差、ILO整体或high-WSS回退并存。因此不晋级“数据扩容提高整体性能”的结论，只保留为单seed探索证据。同一test27归一化空间中，Q2V `0.6210` 仍高于D1 frozen/refit `0.5961/0.6092`。D3的五层固定计数仍为 AG `61/15`、AAA rupture `21/6`、AAA unrupture `24/6`、ILO-0 `22/6`、ILO-1 `10/3`；小亚组不作确认性解释。

| nsample × width / Job | 参数量 | 物理 / 归一化 val R²_cb | width64−width32（物理） | 排名与动作 |
| --- | ---: | ---: | ---: | --- |
| 16 × 32 / `10488_0` | 0.224M | 0.2456 / 0.5481 | — | **Q2V-structure dev control** |
| 16 × 64 / `10488_1` | 0.880M | 0.2368 / **0.5709** | -0.0089 | normalized数值第一；物理回退 |
| 32 × 32 / `10488_2` | 0.224M | 0.2589 / 0.5294 | — | 物理第2；效率候选 |
| **32 × 64 / `10488_3`** | 0.880M | **0.2603** / 0.5431 | **+0.0014** | **预注册物理主指标第1；数值开发候选** |
| 64 × 32 / `10488_4` | 0.224M | 0.2327 / 0.5506 | — | 物理第6 |
| 64 × 64 / `10488_5` | 0.880M | 0.2436 / 0.5540 | +0.0108 | 物理第4 |

架构归因结论：预注册主指标是物理 `R²_cb`，因此 `nsample=32` 在两种width下都最好，`n32_w64` 是物理主指标候选；normalized val21 数值第一是 `n16_w64`，只能作为次指标现象，不能事后改写主排名。`n32_w64` 仅领先 `n32_w32` 0.0014、参数约4倍，仍只登记为确认候选。`n16_w32` 是配置匹配且在dev85重训的Q2V结构控制；原Q2V checkpoint不能补到val21排名，因为这21例全部来自其原train106，直接评估是21/21泄漏。Q2V原test27归一化 `R²_cb=0.6210` 只作历史锚点，与val21不同split/stats不可横比。现在test27已被多轮历史探索和本次QAD复核使用，不再具备独立确认资格；若架构候选进入确认阶段，应使用未参与选择的独立holdout或前瞻数据。

集群状态：`10484/10485/10488/10489/10490`完成，当前队列为空；`10487_3–5`的Slurm `FAILED`仅来自定量评估后的PostView门禁。共同失败病例 `ILO/YU_XIANG_SHENG-1/before` 映射覆盖为 `7193/26942=26.70%`，其他35例100%；模型指标有效，但可视化包未闭环。finalizer `10486` 在完成全部提交后因node03缺少Python `uno`失败，只影响自动回填。结构化审计真源为 `training_wss_min/preflight/q2v_ilo_arch_matrix_results_analysis.json` 与同名summary CSV。AreaRandom六组继续冻结。
<!-- Q2V_ILO_20260718_END -->

## 0E. Q2V/test27 半径与采样探索（2026-07-18｜4/4完成｜非确认性）

这是以 Q2V-10477 的历史test27为锚的用户授权探索，不新增独立确认集。共同冻结 `106/0/27`、vertex legacy指标、400 epoch、train-loss best、完整SA链 `5000→500→125→32`；`10505` GPU preflight 4/4通过，`10506_[0–3]` 4/4均完成400 epoch、best/last评估和PostView `27/27`，配置哈希无漂移、无NaN/Inf。因此下表只能用于**排除方向**，不得按test27事后定稿。

| ID / Job | 对照与唯一变化 | 物理 R²_cb | 对照差 | 主要伴随指标 | 决策 |
| --- | --- | ---: | ---: | --- | --- |
| Q2V-10477（历史） | SEP；r=`0.05/0.10/0.20` | 0.2724 | — | MAE/RMSE `2.562/6.230`，high `-0.521` | 半径锚点 |
| `q2v_radius_r80_sep` / `10506_0` | 对Q2V仅r=`0.04/0.08/0.16` | 0.2433 | `-0.0291` | MAE/RMSE `2.655/6.326`；high `-0.505` | 主指标与误差回退，不晋级 |
| `q2v_radius_r60_sep` / `10506_1` | 对Q2V仅r=`0.03/0.06/0.12` | 0.2406 | `-0.0319` | MAE/RMSE `2.665/6.400`；high `-0.590` | 进一步回退，不晋级 |
| Q1V-10476（历史） | vertex-random5000 SAME | **0.2763** | — | high `-0.513`、IoU `0.1803` | 采样匹配控制 / 现优先候选 |
| `q1v_fpsmultistart5000_same` / `10506_2` | 对Q1V仅采样换fps_multistart5000 | 0.2297 | `-0.0466` | MAE/RMSE `2.630/6.390`；high `-0.601` | 不支持用该FPS替换vertex-random |
| `q2v_to_q1v_same_finetune` / `10506_3` | Q2V best warm-start→Q1V SAME；优化器重置 | 0.2670 | 对Q1V `-0.0093` | MAE/RMSE `2.571/6.231`；high `-0.523` | 非纯采样对照，未见稳定warm-start增益 |

结论：四个新臂都没有超过Q1V `R²_cb=0.2763`，且high-WSS R²仍全部为负。r80在high-WSS R²有局部数值改善，却伴随整体R²和误差回退，不能单指标选它。下一步只保留Q1V做预先固定的独立重复；缩小ball半径、fps_multistart5000和本次warm-start路线停止扩展。结构化审计真源为 `training_wss_min/preflight/q2v_sampling_radius_test27_results_analysis.json` 与同名summary CSV；xlsx 的教师视图和总览均已回填4行。

## 0F. Q1V/SAME 半径探索（2026-07-19~20｜4个半径臂完成｜非确认性）

用户在上述 Q2V/SEP 缩半径无增益结论之后，明确要求以 **Q1V-10476**（vertex-random5000 / SAME）追加半径探索。因此本矩阵不改写上一节的停止建议，也不把历史 test27 用作确认集：只允许用于检验 Q1V 的 SAME 合同是否存在与 SEP 不同的半径响应。共同冻结 `106/0/27`、`stl_landmarks_v4`、seed1234、400 epoch、train-loss best、vertex legacy 指标、无旋转、`5000→500→125→32`、nsample16、width32/head64、FPS center 与 fixed-support/full-cloud query；**唯一变量为三层 SA radius**。原始 Q1V `0.05/0.10/0.20` 已完成，不重训，作为所有新臂的共同对照。

`10545` GPU preflight `3/3` 通过；`10546_[0–2]` 均完成400 epoch、best/last test27 评估，配置哈希未漂移，best PostView `27/27` 验证通过，且未出现 NaN/Inf。用户随后追加的0.6× SAME 补充臂也由 `10555` preflight 通过、`10556` 完成相同训练—评估—PostView闭环。训练仍严格按 train loss 选 `ckpt_best`。下表的数值排序仅用于描述半径方向，不能以已复用的历史 test27 选定最终半径。

| ID / Job | Q1V 单变量半径 | 物理 R²_cb | 对 Q1V 差 | MAE / RMSE | high-WSS R² / IoU | 判读 |
| --- | --- | ---: | ---: | --- | --- | --- |
| Q1V-10476（历史对照） | `0.05/0.10/0.20`（100%） | 0.2763 | — | `2.588 / 6.240` | `-0.513 / 0.1803` | 共同 SAME 对照 |
| `q1v_radius_r60_same` / `10556` | `0.03/0.06/0.12`（60%，补充） | 0.2584 | `-0.0179` | `2.624 / 6.245` | `-0.523 / 0.1608` | pooled R²几乎持平，但病例等权R²、MAE、high-WSS与IoU均回退；p99幅值比虽从`0.411→0.432`改善，不能抵消整体退化 |
| `q1v_radius_r80_same` / `10546_0` | `0.04/0.08/0.16`（80%） | 0.2263 | `-0.0501` | `2.649 / 6.361` | `-0.589 / 0.1709` | 所有主伴随指标回退，3个负例；排除缩小半径 |
| `q1v_radius_r120_same` / `10546_1` | `0.06/0.12/0.24`（120%） | **0.2848** | **`+0.0085`** | `2.590 / 6.164` | **`-0.478`** / `0.1757` | R²_cb、pooled R²、RMSE 与 high-WSS R²改善；但病例均值 `-0.0136`、负例 `+1`、IoU `-0.0046`，只登记为独立复核候选 |
| `q1v_radius_r150_same` / `10546_2` | `0.075/0.15/0.30`（150%） | 0.2654 | `-0.0109` | `2.595 / 6.191` | `-0.495 / 0.1528` | 误差/high-WSS 局部改善不足以抵消病例均值、主 R² 与热点 IoU回退；不晋级 |

结果解释：SAME 合同下半径响应不是单调的，但缩小半径的方向已得到两个点的负向证据。60%相对Q1V的 R²_cb `-0.0179`、病例均值 R² `-0.0306`、MAE `+0.036 Pa`、IoU `-0.0195`；80%则更明显回退，故停止继续缩小。150% 虽降低 pooled RMSE `0.049 Pa`、改善 high-WSS R² `+0.0186`，却降低病例等权 R² 与热点 IoU；只有120%在物理 pooled/病例平衡 R²、RMSE和high-WSS R²上同向改善。但该增益很小（R²_cb `+0.0085`），病例均值 R²下降、负例由1增至2，热点IoU下降，且high-WSS R²仍为负，不能判为稳健优胜。best/last敏感性也不支持用一次测试排名替代确认：60% 的 last R²_cb 比 best `-0.0030`，80%与150%的 last 分别高于 best `+0.0294/+0.0094`，而120%则低 `-0.0098`；因此只可将 `r=0.06/0.12/0.24` 与原始Q1V带入预先固定的独立重复，停止继续扩展其他半径。

结构化审计真源为 `training_wss_min/preflight/q1v_radius_test27_results_analysis.json`、`q1v_radius_r60_same_results_analysis.json` 及各自summary CSV；xlsx 的教师视图和总览已回填全部四个Q1V/SAME半径臂。

## 0A. test27 Support/Query 与采样矩阵（2026-07-18｜Phase-V 定量完成，Phase-A 待修）

共同冻结：split `106/0/27`、train strata `61/21/24`、test `15/6/6`、`stl_landmarks_v4`、seed1234、400 epoch、peak WSS、train106 point-pooled global log-z、xyzgeom、MSE、train-loss 选模、AMP、无旋转；PointNet++ 为 radius `0.05/0.10/0.20`、nsample16、FP k3、width32/head64。`random` 是壁面训练点 vertex-uniform 无放回；`area_random` 是 STL 三角面积1/3分摊后映射到训练壁面点，二者不可混名。

SA 链统一按“输入 support→三层 center”书写：B1/9169 与 Q0/10475 均为 `2000→500→125→32`；Q1V/Q2V/Q3V 及本轮 Q2V 数据/架构矩阵为 `5000→500→125→32`。两者后三层 center 数相同，但第一层分别从2000个 FPS support或5000个 vertex-random support中分组，邻域样本和输入信息量不同，因此不能把它们视为同一个下采样输入。B1 还使用 legacy ratio-based full-cloud 评估路径，Q0起使用 fixed-support/full-cloud query，推理合同也不同。

| ID | 配置/Job | Support / Query | SA center | 严格对照与唯一变化 | 真实状态 |
| --- | --- | --- | --- | --- | --- |
| B0 | `9170` | FPS2000 / SAME | — | 既有 PointNet E2 基线 | 模型资产完成；未重复提交 |
| B1 | `9169` | FPS2000 / SAME legacy | ratio/FPS | 既有 PointNet++ SA3 基线 | 模型/评估完成；PostView 27/27；未重训 |
| **P1V** | `10473` | vertex random5000 / SAME | — | B0：FPS2000固定→每epoch vertex-random5000（采样协议+点数） | ✅ 400 epoch + best/last eval；旧 PostView 留20目录，待 export-only |
| **P2V** | `10474` | vertex random5000 / independent random5000 | — | P1V：只改 SAME→SEP | ✅ 400 epoch + best/last eval；**PointNet 本次优先候选**；待 export-only |
| **Q0** | `10475` | FPS2000 / SAME | FPS 500/125/32 | B1：legacy ratio/eval→固定中心+fixed-support/full-query 路径 | ✅ 400 epoch + best/last eval + PostView **27/27** 验证通过 |
| **Q1V** | `10476` | vertex random5000 / SAME | FPS 500/125/32 | Q0：FPS2000固定→每epoch vertex-random5000（采样协议+点数） | ✅ 400 epoch + best/last eval；**PointNet++ 本次优先候选**；待 export-only |
| **Q2V** | `10477` | vertex random5000 / independent random5000 | FPS 500/125/32 | Q1V：只改 SAME→SEP | ✅ 400 epoch + best/last eval；结果混合；待 export-only |
| **Q3V** | `10478` | vertex random5000 / SAME | Random 500/125/32 | Q1V：**只改 center FPS→Random** | ✅ 400 epoch + best/last eval；整体回退；待 export-only |
| P1 | `...e2_global_area_random5000_same` | area random5000 / SAME | — | P1V：只改 vertex random→AreaRandom | **DEFERRED，未提交** |
| P2 | `...e2_global_area_random5000_sep` | area random5000 / independent area random5000 | — | P2V：只改 vertex random→AreaRandom | **DEFERRED，未提交** |
| Q1 | `...sa3_area_random5000_fpscenter_same` | area random5000 / SAME | FPS 500/125/32 | Q1V：只改 vertex random→AreaRandom | **DEFERRED，未提交** |
| Q2 | `...sa3_area_random5000_fpscenter_sep` | area random5000 / independent area random5000 | FPS 500/125/32 | Q2V：只改 vertex random→AreaRandom | **DEFERRED，未提交** |
| Q3 | `...sa3_area_random5000_randomcenter_same` | area random5000 / SAME | Random 500/125/32 | Q1：只改 center FPS→Random | **DEFERRED，未提交** |
| Q4 | `...sa3_area_random5000_randomcenter_sep` | area random5000 / independent area random5000 | Random 500/125/32 | Q2/Q3：center sampling 或 SAME→SEP | **DEFERRED，未提交** |

Phase-V 六组显式使用 `eval.surface_metric_mode=legacy_vertex`，不读取 STL 面积、不输出伪 area 指标；旧 config 缺省仍保持该行为。Phase-A 六组显式 `both_strict`，只有它们消费面积映射并保留2%/10%硬门。正式 Phase-V preflight 为6/6通过：133/133 bundle/frame及stats哈希通过，batch8 RTX4090 AMP无OOM，PointNet峰值约559 MiB、PointNet++约127–181 MiB，中心精确500/125/32，full/chunk最大差`7.45e-8`。

面积硬门失败病例为 `AG/fast/LI_ZHEN_SHAN`、`AAA/ruputer/{XIE_JIN_QUAN,ZHOU_KE_XUN,SHI_YUN_XI}`、`AAA/unruputer/{GUO_BAO_CHUN,LIU_WEN_QI}`。诊断、失败ID与六份冻结配置已固化到 `training_wss_min/preflight/ag_aaa_v4_area_phase_backlog.json`；后续不得放宽阈值、自动退化或把同一模型的技术重复当额外病例。

失败病例可视化见 [`area_mapping_failures_summary.png`](_archive/assets_新队列审计/area_mapping_failures_20260717/area_mapping_failures_summary.png) 与同目录6张逐例三视图。图中四个 crop 例的高距离区集中在完整 STL 尾端；`ZHOU_KE_XUN` 原坐标整体相隔约一个大平移，bbox-center 平移后主体高度重合；`LIU_WEN_QI` 平移后分支与瘤体轮廓仍不一致，因此不能把两例都自动按“只修平移”处理。

### 0B. 面积依赖合同与判读边界（后续实验按此执行）

| 环节 | `legacy_vertex` / Phase-V | `both_strict` / Phase-A |
| --- | --- | --- |
| 训练采样 | FPS 或壁面点 vertex-uniform；不读 STL 面积 | `area_random`；必须先把原始 STL 三角面积严格转移到有效 CFD 壁面点 |
| preflight | 检查 bundle/frame/split/stats/显存/Support-Query，不要求面积映射 | 除左侧检查外，133/133 病例必须通过面积映射硬门 |
| 评估与热点 | 逐壁面点指标、vertex top10；字段必须明确标为 legacy vertex | 同时输出 vertex 与 area-weighted 指标；面积字段只在严格映射成功时存在 |
| PostView | 可把点预测插值到有效 STL 作**可视化**，不得因此加载或伪造面积权重 | 可视化之外，还可输出严格面积 top10/IoU 等物理表面积指标 |
| 结论用途 | 回答“当前离散 CFD 点集上的模型表现”，可作为完整独立主线 | 回答“连续物理表面积意义下的表现”，是独立敏感性/物理口径线 |

这里必须区分两种“映射通过”：PostView 的 Gaussian `mapping coverage=100%` 只表示每个**有效可视化 STL 顶点**附近找到了壁面点；它不等于“原始完整 STL 三角面积已无偏、严格地转移到 CFD 壁面点”。旧导出器混淆了两者，导致 `10473/10474/10476/10477/10478` 在完成训练与 best/last vertex eval 后，运行到 `AAA/ruputer/SHI_YUN_XI` 才因面积硬门退出；`10478` 的导出进程在代码修复前已经启动并载入旧模块，因此不会热更新。代码现已按 `surface_metric_mode` 分流：后续新启动的 `legacy_vertex` 导出不再调用 `area_weights_for_case`，旧的半成品 manifest 也不会跨口径复用。上述五组只需 export-only 补齐 PostView，不需要重训。

面积审计失败也**不自动等于中心线几何特征错误**。当前 `abscissa_norm/local_radius/curvature` 来自 bundle 预处理和已注册坐标帧；四个 crop 例更像“完整 STL 尾段 vs CFD 有效裁剪壁面”的面积支撑域不一致，不能据此推翻既有点口径训练。`ZHOU_KE_XUN` 需先审计刚体变换，`LIU_WEN_QI` 需核查 STL/CFD 数据血缘；只有确认原始几何配错或重发布几何后，才需要重提中心线和几何特征。修复前禁止把这两例的 area-weighted 结果当正式证据，但既有 vertex/FPS 结果仍可保留并明确标注口径。

后续采样优化按单变量顺序推进：先完成 Phase-V 的 `FPS2000 ↔ vertex-random5000`、`SAME ↔ SEP`、`FPS center ↔ Random center` 配对；若随机点覆盖不稳，优先考虑多起点 FPS 或按中心线弧长/分支分层的等额采样；只有严格面积映射 133/133 修复后，再比较 AreaRandom。不要把面积采样与点数、Support/Query、中心策略一次同时改变。

### 0C. Phase-V 完训结果与判读（2026-07-18｜test27 legacy vertex）

以下均为同一 split `106/0/27`、同一 train106 global log-z 统计、全27 test 病例和 `ckpt_best(train_loss)`；因此可在各自预注册配对内比较。它们是**离散 CFD 壁面点**的 legacy-vertex 结果，不含也不替代严格面积指标。

| ID / Job | 物理 R²_cb / point / case-mean | 负例 | MAE / RMSE (Pa) | Spearman / top10 IoU | high-WSS R² / MAE | 本次判读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B0 / `9170` | 0.0530 / 0.0831 / 0.0405 | 12 | 2.919 / 6.957 | 0.6512 / 0.1727 | -0.862 / 13.234 | PointNet FPS2000 诊断锚点 |
| P1V / `10473` | 0.1592 / 0.1750 / 0.1186 | 8 | 2.678 / 6.599 | 0.6914 / 0.1546 | -0.737 / 13.256 | 相对 B0 整体显著改善，但 IoU 与 high-WSS MAE 未同步改善 |
| **P2V / `10474`** | **0.2193 / 0.2087 / 0.1805** | **1** | 2.606 / 6.463 | 0.7161 / 0.1607 | -0.656 / 12.567 | **PointNet 分支优先候选；相对 P1V 的 SAME→SEP 是主增益** |
| B1 / `9169` | 0.1331 / 0.1371 / 0.0861 | 6 | 2.736 / 6.749 | 0.6959 / 0.1846 | -0.800 / 13.756 | PointNet++ legacy 锚点 |
| Q0 / `10475` | 0.2212 / 0.2224 / 0.1657 | 4 | 2.637 / 6.407 | 0.7112 / 0.1645 | -0.615 / 12.580 | fixed-center/full-query 桥接有效，但相对 B1 含推理路径变化，非单一采样归因 |
| **Q1V / `10476`** | **0.2763 / 0.2623 / 0.2114** | **1** | 2.588 / 6.240 | 0.7222 / **0.1803** | **-0.513 / 12.074** | **PointNet++ 分支优先候选；相对 Q0 随机5000点全面提升主指标** |
| Q2V / `10477` | 0.2724 / **0.2647** / **0.2157** | 2 | **2.562 / 6.230** | **0.7358** / 0.1568 | -0.521 / 12.212 | SEP 只带来部分点/病例误差收益，R²_cb、IoU、负例和高尾略差，**不判优于 Q1V** |
| Q3V / `10478` | 0.2521 / 0.2345 / 0.1852 | 2 | 2.634 / 6.357 | 0.7107 / 0.1595 | -0.579 / 12.486 | 随机 SA center 相对 Q1V 整体回退；仅峰值距离略小，不足以继续扩展 |

1. **PointNet：P2V 是可复核的下一步候选。** P1V 相对 B0 的 R²_cb 增加 `+0.1063`、负例 `12→8`；在固定 P1V 的前提下，SEP 再将 R²_cb 增加 `+0.0600`、case-mean 增加 `+0.0618`、负例 `8→1`，且 high-WSS MAE 降 `0.689 Pa`。P1V 同时改变点数与采样协议，不能把 B0→P1V 写成纯采样因果。
2. **PointNet++：Q1V 是本次主候选。** Q0 相对 B1 的 R²_cb 增加 `+0.0881`，但含 fixed-center/full-query 桥接，不能单独归因；Q1V 相对 Q0 再增加 `+0.0551`，负例 `4→1`，high-WSS R² `-0.615→-0.513`，是当前最干净的随机5000点增益。Q2V 与 Q1V 的差异没有一致方向，Q3V 相对 Q1V 的主指标全部回退，均不作为优先扩展路线。
3. **不能把单次 test27 排名写成发布结论。** P2V/Q1V 仅用于确定下一轮复核优先级；所有 checkpoint 仍严格按 train loss 选出，best/last 敏感性也没有改变该规则。六组 high-WSS R² 均为负、最高仅 `-0.513`，p99/peak 仍明显压缩，故本轮不判物理高尾问题已解决。
4. **闭环与后续顺序**：先对 P1V/P2V/Q1V/Q2V/Q3V 使用修复后的 exporter 做 export-only 并通过27/27批次验证；随后只把 P2V 与 Q1V 带入预先约定的独立重复/确认协议。Phase-A 仍须先修复133/133严格面积映射，不能由本节任何 vertex 结果放行。

## 0. v4 结果结论（2026-07-16）

三列均是同一 AG common-test15 的物理 WSS，逐点真值一致。旧锚点从已保存预测纯后处理重算，没有重训/重推理；两套 v4 取预注册的 `ckpt_best(train_loss)`。`point` 为全点池化；`case-balanced` 对每例赋相同总权重；`case-mean NMAE` 是逐例 `MAE/(true max−true min)` 后平均。

| 口径 | train | point R² | point MAE / RMSE / NMAE | case-balanced R² | case-balanced MAE / RMSE | case-mean R² / NMAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧 v3 E2 common-test15 锚点 | AG61 | **0.2597** | **2.754 / 4.902 / 0.02431** | **0.2554** | **2.723 / 4.840** | **0.1928 / 0.05122** |
| AG-v4 E2 | AG61 | 0.1852 | 2.847 / 5.143 / 0.02513 | 0.1866 | 2.802 / 5.059 | 0.1465 / 0.05300 |
| AG+AAA-v4 E2 | AG61+AAA57 | 0.2086 | 2.772 / 5.068 / 0.02447 | 0.2054 | 2.741 / 5.000 | 0.1572 / 0.05148 |

| 口径 | high-WSS point R² / MAE | p95 / p99 / peak 幅值比（病例均值） | Spearman / top10 IoU | 峰值 / 热点质心距离÷bbox |
| --- | ---: | ---: | ---: | ---: |
| 旧 v3 E2 common-test15 锚点 | **-1.5357 / 10.411** | **0.624 / 0.533 / 0.269** | 0.6756 / 0.1528 | **0.1679 / 0.0849** |
| AG-v4 E2 | -1.8279 / **10.919** | **0.630 / 0.538 / 0.254** | 0.6788 / 0.1314 | 0.2114 / 0.0895 |
| AG+AAA-v4 E2 | -1.8473 / 11.195 | 0.531 / 0.443 / 0.206 | **0.7041 / 0.1575** | 0.2084 / **0.0785** |

1. **AG-v4 单独重跑 No-Go**：对同一 test15，相对旧锚点 field/case-mean R² 下降 `0.0744/0.0463`，MAE/RMSE 增加 `0.093/0.241 Pa`，15例仅3例 R² 改善。因此不能把 v4 坐标/数据发布解读为 E2 精度提升。
2. **AAA 扩容是小幅、非均匀的增益**：相对 AG-v4，混合 field/case-mean R² 增加 `0.0234/0.0106`，MAE/RMSE 下降 `0.075/0.074 Pa`，15例中8例改善；Spearman 提高 `0.0254`，top10 IoU 提高 `0.0261`，热点质心距离/bbox 由 `0.0895` 降到 `0.0785`。
3. **但幅值与高尾更差**：high-WSS point R² `-1.8279→-1.8473`、MAE `10.919→11.195 Pa`，病例均值 p99/peak 幅值比 `0.5380/0.2540→0.4433/0.2058`。混合模型仍比旧锚点低 `0.0510` point R²，只是 MAE 已接近（高 `0.018 Pa`）。
4. **病例异质性值得优先处理**：混合相对 AG-v4 对 `BAI_WEN_JIE`、`WANG_YONG_FAN`、`LU_ZHEN_QING` 改善最大；对高 WSS 观察病例 `LI_SHU_KUN` 恶化最大（R² 约 `-0.293`），`GUO_XI_JIANG` 约 `-0.149`。下一步应优先检查 AG/AAA 域比例、高尾损失/采样和困难病例，不建议先切 random-5000。
5. `last` 对 AG-v4 略差；混合 last 的 field R² 为 `0.2193`，比 best 高 `0.0106`，但不能用 test 指标改写预注册选模。这只说明 best/last 敏感性低且主结论不变。
6. **归一化有一个需要先隔离的混杂因素**：混合训练按病例是 AG:AAA=`61:57`，但 train-only global stats 按全云点汇总；AAA 密网格贡献了混合统计约 `78.8%` 的点。因此 log-WSS 均值/标准差从 AG 的 `1.122/1.096` 变为混合的 `0.553/1.373`，p50 从 `3.088` 降到 `1.683 Pa`。这与 AG test 高幅值进一步压缩一致，但目前只是机制性推断；下一个最干净的单变量应是“病例平衡/域平衡 target stats”，而不是同时改采样点数。
7. 混合 best 的物理 train `R²_cb` 从 AG-v4 的 `0.6041` 降到 `0.4922`，而test 从 `0.1866` 升到 `0.2054`，train−test gap 由 `0.4174` 缩到 `0.2867`。这更像“AAA 提供了正则化/空间排序信号，但幅值和域匹配仍未解决”；两套的 train loss 因 target stats 不同不能直接比较。

产物完整性：Jobs `9138` / `9140` 均 `COMPLETED (0:0)`；两套 best/last checkpoint、train+test 全云 metrics/per-case CSV 齐全，best PostView 均 15/15 病例、每例4个 VTP、mapping coverage 100%。统计 bundle 哈希全匹配，日志无 NaN/Inf、OOM、Traceback、frame/path 或数量错误，九个排除病例无 partition/stats 泄漏。结构化真源见 `data_wss_min/pipeline_reports/v4_cutover_20260715_1921/{v4_e2_global_acceptance.json,v4_e2_global_result_analysis.json,v4_e2_common_test15_per_case_comparison.csv}`。

### 0.1 第三混合 split 与 PointNet++ 结果

第三协议不再锁定旧 AG test15。从 AG76 + AAA57 合格池按 `AG / AAA rupture / AAA unrupture` 分层、seed1234、固定 strata 顺序与排序后的 canonical ID 确定性重划：

| partition | AG | AAA rupture | AAA unrupture | 合计 |
| --- | ---: | ---: | ---: | ---: |
| train | 61 | 21 | 24 | **106** |
| val | 0 | 0 | 0 | **0** |
| test | 15 | 6 | 6 | **27** |

- 新 test27 与旧 AG test15 仅重合 `AG/slow/GUO_XI_JIANG`、`AG/slow/ZHANG_JUN_HUA`；这是独立协议，聚合指标不能与前两组 common-test15 直接横比。
- 已知相关几何 `HOU_SHEN_QIAN/KANG_XI_MING` 同组分配；精确几何指纹未发现其他跨 partition 重复。
- train106 独立 point-pooled peak-WSS `log_z` 统计中，AG/AAA 贡献 `778,189/2,360,613` 点（`24.79%/75.21%`），算法与前两组保持一致。

| 配置 / Job | 模型 | split | 统计 | 当前状态 |
| --- | --- | --- | --- | --- |
| `ag_v4_sa3_e2_global_fps2000` / `9167` | PointNet++ SA3 | AG `61/0/15` | AG train61 | ✅ 400 epoch + best/last eval + PostView `15/15` |
| `ag_aaa_v4_locked_sa3_e2_global_fps2000` / `9168` | PointNet++ SA3 | 锁定 AG test15 `118/0/15` | mixed train118 | ✅ 400 epoch + best/last eval + PostView `15/15` |
| `ag_aaa_v4_stratified_sa3_e2_global_fps2000` / `9169` | PointNet++ SA3 | 新混合 `106/0/27` | 新 train106 | ✅ 400 epoch + best/last eval；export-only `9818` 后 PostView `27/27` |
| `ag_aaa_v4_stratified_e2_global_fps2000` / `9170` | PointNet E2 | 新混合 `106/0/27` | 新 train106 | ⚠️ 400 epoch完成；正式eval被面积门禁阻断；全27 vertex best/last诊断完成 |

PointNet++ 结构冻结为 `2000→500→125→32`、radius `0.05/0.10/0.20`、nsample16、FP k=3、width32、head64，参数量约 0.22M；PointNet E2 约 0.79M。四组均为 xyzgeom、GLOBAL log-z、FPS-2000、MSE、seed1234、400 epoch、无旋转/每 epoch 重采样，按 train loss 选模。

#### common-test15 配对结果

| 协议 | 模型 | 物理 `R²_cb` | 归一化 `R²_cb` | 物理 MAE / RMSE | Spearman | top10 幅值比 / IoU | high-WSS R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AG `61/0/15` | PointNet E2 | **0.1866** | **0.4676** | **2.847 / 5.143** | **0.6788** | **0.3508** / 0.1314 | **-1.8279** |
| AG `61/0/15` | PointNet++ SA3 | 0.1500 | 0.4463 | 2.867 / 5.264 | 0.6598 | 0.3078 / **0.1451** | -2.0479 |
| 锁定 AG test15 `118/0/15` | PointNet E2 | **0.2054** | **0.4906** | **2.772 / 5.068** | **0.7041** | **0.3405** / 0.1575 | **-1.8473** |
| 锁定 AG test15 `118/0/15` | PointNet++ SA3 | 0.1490 | 0.4726 | 2.829 / 5.261 | 0.6779 | 0.2858 / **0.1633** | -2.1388 |

- AG-only 中 PointNet++ 的物理/归一化 `R²_cb` 相对 E2 分别下降 `0.0367/0.0213`，15例仅4例物理 R² 改善；锁定混合中分别下降 `0.0564/0.0180`，15例仅6例改善。top10 IoU 的 `+0.0138/+0.0058` 局部改善不足以抵消整体精度、排序和高尾幅值的回退，因此只对**当前冻结 SA3 配置**判 No-Go。
- PointNet++ 混入 AAA 后，相对自身 AG-only 的归一化 `R²_cb` / Spearman / top10 IoU 提高 `0.0263/0.0180/0.0182`，但物理 `R²_cb` 几乎不变（`-0.0009`）、top10 物理幅值比更低。这与 E2 的早期结论一致：AAA 主要补充空间排序/热点信号，没有解决物理幅值和高 WSS 压缩。
- 这不是容量配平的架构终裁：SA3 参数只有 E2 的约 28%，且其完整壁面 train 物理 `R²_cb` 也更低（AG `0.3076 vs 0.6041`；混合 `0.3035 vs 0.4922`）。同时 sampled train loss 并未同比例变差，提示固定 FPS-2000 训练与 full-cloud 评估、固定 SA 中心数/radius 之间可能存在密度敏感的推理落差。这是机制性假设，需用“同 checkpoint 的 FPS-2000 评估 vs full-cloud 评估 + full-cloud SA coverage”只读诊断确认，本轮不自动开新训练。

#### 分层 test27 同协议配对结果

以下全是同一 split/stats/test27 的 **legacy vertex** 口径；`9169` 取正式 `ckpt_best(train_loss)`，`9170` 取相同选模规则的有效 best checkpoint 后诊断补算。它们可直接配对，但不是完整面积主指标。

| 模型 / Job | 物理 `R²_cb` | 归一化 `R²_cb` | case-mean R² / 负例 | MAE / RMSE | Spearman | top10 幅值比 / IoU | high-WSS R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **PointNet++ SA3 / `9169`** | **0.1331** | **0.5307** | **0.0861 / 6·27** | **2.736 / 6.749** | **0.6959** | **0.2301 / 0.1846** | **-0.8004** |
| PointNet E2 / `9170` | 0.0530 | 0.4569 | 0.0405 / 12·27 | 2.919 / 6.957 | 0.6512 | 0.2075 / 0.1727 | -0.8620 |

`9169−9170` 的物理/归一化 `R²_cb` 为 `+0.0801/+0.0738`，legacy IoU/Spearman 为 `+0.0119/+0.0447`，且负例少6例；逐例物理 R² 中9170改善10例、退化17例。9170 的优势只出现在峰值点距离/bbox（`0.1399 vs 0.1725`），但其热点质心距离更大（`0.0765 vs 0.0667`）。因此判“当前冻结 SA3 配置在同协议上优于 E2”，仍保留0.22M vs 0.79M容量混杂，不写成架构终裁。两者 high-WSS R² 仍为负、top10物理幅值比都低于0.24，高尾压缩没有解决。

面积口径必须另列：9170 仅26/27通过映射的子集可算 physical/normalized area `R²_cb=0.1559/0.5243`、area top10 IoU=`0.2564`；`AAA/ruputer/SHI_YUN_XI` 被排除，且9169旧评估没有同口径面积结果，因此这些数值不得用于上面的架构胜负。9170 last 相对 best 的物理 `R²_cb` 仅 `+0.0009`，不改变预注册 best 主结论。该 test27 与 common-test15 只重合2例，仍禁止跨协议横比。9169 PostView 已由 `9818` 补齐27/27；9170 的标准 `eval/` 与 PostView 尚未生成。

## 1. 当前冻结决策

1. 历史正式矩阵停止继续扩展 MLP / xyz-only；用户本次明确批准的 PointNet++ 仅限 §0.1 三层 SA 三协议，并配一组同 split PointNet E2。不得扩回旧四层结构或通用架构扫参。
2. 历史正式协议为 AG `61/0/16`；v4 公平协议为 `61/0/15`，只删除 `WANG_DENG_FENG`，不补病例。训练期不读 test，关闭早停，训练400 epoch，按 train loss 保存 best。
3. 容量实验按导师结构对齐：局部编码 `6→256→512`，全局拼接后解码 `1024→512→256→1`，`dropout=0`。
4. 历史 E3 点数消融定义为每例每 epoch 随机不放回5000点；本次 v4 PointNet++/重划矩阵明确保持 **FPS-2000**，不切 random-5000。
5. 归一化实验只研究无量纲的 WSS 空间分布和 high-risk position，不要求恢复测试病例的物理 WSS。
6. Job `8970` 已完训（旧 `53/8/16 + val-only + 160 epoch` 协议的 width=128 探针），**仅作补充证据，不纳入下述正式配对矩阵，也不代替 `E2-GLOBAL`**。
7. `N-CASE` 已冻结为 **`WSS/WSSmax`**；不再把逐病例 log-z 混入本矩阵。
8. `ckpt_best(train_loss)` 为预注册主模型并导出 test16 PostView；`ckpt_last` 只做完整指标审计。连通性指标本轮暂缓。
9. 导师追加的 `E4-DEEP-GLOBAL` 是矩阵结案后的单次深度探针：严格对齐 E2，只改网络深度；Job `8999` 已完训并判为 **No-Go**，不并入已完成的五组正式矩阵，也不恢复 random-5000 或 CASE 路线。
10. v4 发布后的 AG 协议为 `61/0/15`；旧 `61/0/16` 结果保留为历史事实，公平配对列使用 common-test15。

## 2. 要回答的科学问题

### Q1｜模型容量

在同一输入、点数、损失和训练协议下，将 PointNet 对齐到导师的宽通道结构，是否能改善 WSS 非线性空间分布与高 WSS 位置识别。

### Q2｜单例点数与每 epoch 重采样

在不改模型的情况下，将每例训练点从固定 FPS-2000 改为随机不放回 5000 点，并在每个 epoch 重采样，是否能让网络看到更完整的病例内 WSS 空间型态。

### Q3｜跨病例幅值是否遮蔽空间型态

对照全局归一化与逐病例归一化，重点检验：当各 case 的目标数值被压到近似尺度后，网络是否更能预测病例内的 WSS 相对分布、高风险区域与峰值位置，而不再主要受病例间绝对幅值差异影响。

## 3. 归一化两套实验

> 术语说明：z-score 严格由均值和标准差定义，不是由 range 定义。导师口头所说的“全局 range”，在本文按“全局共享统计量”理解。

### N-GLOBAL｜全局 train-only log-z

\[
y_G=\frac{\log(\mathrm{WSS}+\epsilon)-\mu_{\mathrm{train}}}
{\sigma_{\mathrm{train}}}
\]

- `mu_train` / `sigma_train` 由 61 个 train case 的峰值时相壁面点共同计算，所有病例共享。
- 当前统计：`eps=1e-6`、`log mean=1.12116`、`log std=1.09615`。
- 保留病例间幅值差异；模型同时承担“跨病例水平 + 病例内空间型态”。
- `E2-GLOBAL` / `E3-GLOBAL` 就是容量实验（2）和采样实验（3）本身，不重复训第二份同配置。

### N-CASE｜逐病例相对尺度

正式冻结为 case-max 口径：

\[
y_C=\frac{\mathrm{WSS}}
{\max_j\mathrm{WSS}_j}
\]

- 每个 case 的真值最大值都为 1，直接移除病例间绝对幅值差异。
- 测试时模型只输出相对场；真实 `WSSmax` 只用于构造归一化评价真值，不作模型输入，也不用于恢复物理 WSS。
- 该任务回答“哪里相对更高”，不回答“绝对值是多少 Pa”。

#### 已排除的本轮替代口径

如果“第二个实验使用逐病例 z-score”指的是每例独立均值/标准差，则另一个公式是：

\[
y_{CZ}=\frac{\log(\mathrm{WSS}+\epsilon)-\mu_i}{\sigma_i}
\]

| 口径 | 对齐效果 | high-risk 解释 | 主要风险 |
| --- | --- | --- | --- |
| `WSS/WSSmax` | 每例 max=1，保留非负相对幅值 | 适合直观画 0–1 分布和 top-q% | max 对单点尖峰/异常值敏感 |
| 逐例 log-z | 每例 mean≈0、std≈1，分布对齐更强 | 须用病例内分位数定义高风险 | 输出有正负，不再是直观的 WSS 比例 |

**冻结结论**：本矩阵只实现 `WSS/WSSmax`。逐例 log-z 若未来获批，必须使用新实验 ID 和独立汇总，不复用 `E2-CASE/E3-CASE`。

## 4. 正式实验矩阵

| ID | 模型/采样变量 | 目标归一化 | 与父实验的唯一差异 | 状态 |
| --- | --- | --- | --- | --- |
| `E0-GLOBAL` | 原 PointNet，FPS-2000 | 全局 log-z | 无 val/无早停共同对照 | ✅ 完训+完评（`8968_1` / eval `8975`）；见 §4.1 |
| `E2-GLOBAL` | 导师宽 PointNet，FPS-2000 | 全局 log-z | 相对 E0 只改容量 | ✅ 完训+完评，Job `8976` |
| `E3-GLOBAL` | 原 PointNet，random-5000/不放回/每 epoch 重采 | 全局 log-z | 相对 E0 只改采样 | ✅ 完训+完评，Job `8977` |
| `E2-CASE` | 与 E2-GLOBAL 相同 | `WSS/WSSmax` | 相对 E2-GLOBAL 只改目标归一化 | ✅ 完训+完评，Job `8979` |
| `E3-CASE` | 与 E3-GLOBAL 相同 | `WSS/WSSmax` | 相对 E3-GLOBAL 只改目标归一化 | ✅ 完训+完评，Job `8980` |
| `E23-GLOBAL` | 导师宽 PointNet，random-5000/不放回/每 epoch 重采 | 全局 log-z | 同时合并容量（2）与采样（3），用于检查交互效应 | ✅ 完训+完评，Job `8978` |

主矩阵对应“2+4”和“3+4”。全局列是原始容量/采样实验，逐病例列是各自的配对归一化实验。`E23-GLOBAL` 是后补的交互对照：只在 E2/E3 单变量结果之外解释“加宽与 5000 点是否协同”，不能替代 E2 或 E3，也不用于单独归因容量/采样贡献。

### 4.1 `E0-GLOBAL` 结果摘要（2026-07-14）

对照臂为 Job `8968` 的 **PointNet + xyz+geom**（`pointnet_trainloss_e400/outputs/pointnet_xyzgeom`）；纯 xyz 仅作输入消融，不进矩阵主线。

| 分区 | `R²_field_cb` | `R²_field_raw` | case mean/med/P10 | 负例 | RMSE/MAE (Pa) | high-WSS R² | top10 比/IoU |
| --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| train61 | 0.5303 | 0.5277 | 0.454 / 0.462 / 0.326 | 0/61 | 4.51 / 2.19 | −0.20 | 0.581 / 0.415 |
| **test16** | **0.1414** | 0.1433 | 0.106 / 0.074 / −0.062 | **5/16** | 5.42 / 2.95 | **−1.76** | 0.328 / 0.174 |

- **判读**：train-fit 有容量信号，但 test16 泛化弱、热点失败；本协议下 `E0` 作为「可部署精度」**No-Go**，仍保留为后续 `E2/E3` 的共同对照锚点（同 split / 同选模）。
- 同协议纯 xyz test16 `R²_cb=0.0825`（更差）→ 确认只保留 xyz+geom。
- 证据：`eval/metrics.json`（`8975`）；详细分析见推进记录文首 `2026-07-14｜8968 train+test16 与 8970 …`。
- **注意**：不把本 test16 数与 2×3 的 val8 `0.3015` 混表。

### 补充探针（不进正式配对矩阵）

| Job | 协议 | 当前作用 |
| --- | --- | --- |
| `8700[0-5]` | `53/8/16`、val-only、160 epoch、MLP/PointNet/PointNet++ × xyz/xyzgeom | 已完成的最小 2×3 架构下限；PointNet+xyzgeom val `R²_cb=0.3015` 为当轮最佳单格 |
| `8968[0-1]` | `61/0/16`、无早停、400 epoch、train-loss 选模 | 已完成；**xyzgeom = `E0-GLOBAL`**；xyz 仅消融 |
| `8970` | `53/8/16`、val-only、160 epoch、`6→128→256→512` | ✅ 完训；val `R²_cb=0.3071` vs 锚点 0.3015（Δ≈+0.006）→ **旧协议加宽 No-Go**；不代替 `E2-GLOBAL` |

### 4.2 正式矩阵结果（2026-07-15）

五个 Job 均为 `COMPLETED (0:0)`，训练、best/last 全点评估和 best 的 test16 PostView 已全部结束。以下均使用预注册的 `ckpt_best(train_loss)` 作主结果；`last` 只用于敏感性审计。

#### GLOBAL：物理 WSS 对照

| Run | best epoch | test `R²_field_cb` | MAE / RMSE (Pa) | high-WSS R² | top10 幅值比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `E0-GLOBAL` | 383 | 0.1414 | 2.913 / 5.345 | −1.76 | 0.328 |
| **`E2-GLOBAL`** | **364** | **0.2140** | **2.801 / 5.114** | **−1.486** | **0.378** |
| `E3-GLOBAL` | 354 | 0.1637 | 2.876 / 5.275 | −1.665 | 0.343 |
| `E23-GLOBAL` | 354 | 0.1988 | 2.806 / 5.163 | −1.525 | 0.372 |

- `E2-GLOBAL` 相对 E0 的主 R² 增加 **0.0725**，是本轮最强的物理 WSS 配置；说明导师宽通道确有容量增益。
- `E3-GLOBAL` 相对 E0 只增加 **0.0223**；单独把 2000 点改为每 epoch random-5000 不是主要突破口。
- `E23-GLOBAL` 没有超过 E2（`−0.0152`），因此没有“加宽 × 5000 点”的明确协同收益；它比 E3 高 `0.0351`，进一步说明本轮主要增量来自容量。
- 即使最优 E2 的 high-WSS R² 仍为 **−1.486**、top10 幅值比仅 **0.378**，所以它是“相对改进”，仍不是可部署精度的 Go。

#### 归一化空间分布与 high-risk position

GLOBAL 行在 train61 共享 log-z 空间评价，CASE 行在逐病例 `WSS/WSSmax` 空间评价。两种目标变换下的 MAE/RMSE 数值尺度不同，不直接横比；R²、排序、top10 和归一化位置指标用于回答本轮空间分布问题。

| Run | test `R²_cb` | case med / P10 | 负例 | Spearman | top10 IoU | 峰值距 / bbox | 质心距 / bbox | p99 比 | 动态范围比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `E0-GLOBAL` | 0.3964 | 0.319 / 0.118 | 0/16 | 0.635 | 0.128 | 0.175 | 0.090 | 0.659 | 0.569 |
| **`E2-GLOBAL`** | **0.4606** | **0.431 / 0.202** | 1/16 | 0.661 | 0.146 | 0.172 | 0.085 | 0.684 | 0.577 |
| `E3-GLOBAL` | 0.4218 | 0.372 / 0.234 | 0/16 | 0.645 | 0.124 | 0.178 | **0.073** | 0.642 | 0.581 |
| `E23-GLOBAL` | 0.4598 | 0.394 / **0.282** | 0/16 | **0.674** | 0.148 | **0.158** | 0.090 | **0.693** | **0.605** |
| `E2-CASE` | 0.1724 | 0.122 / −0.328 | 4/16 | 0.574 | 0.140 | 0.172 | 0.094 | 0.486 | 0.456 |
| `E3-CASE` | 0.1551 | 0.077 / −0.406 | 5/16 | 0.559 | **0.159** | 0.162 | 0.078 | 0.547 | 0.454 |

配对判读：

1. **容量是本轮主增量**：E2 相对 E0 的 normalized `R²_cb` 增加 `0.0642`，Spearman 增加 `0.0261`，top10 IoU 增加 `0.0177`。E23 相对 E3 也全面更好；但 E23 相对 E2 基本持平，5000 点没有稳定附加收益。
2. **`WSS/WSSmax` 主结论为 No-Go**：E2-CASE 相对 E2-GLOBAL 的 `R²_cb` 从 `0.4606` 降至 `0.1724`，Spearman 从 `0.661` 降至 `0.574`，top10 IoU 也没有改善。E3-CASE 的 IoU 从 `0.124` 升至 `0.159`，但 `R²_cb` 从 `0.4218` 降至 `0.1551`、负例从 0 增至 5，且 p99/动态范围明显压缩；局部热点重叠的单项改善不足以支持替换 GLOBAL。
3. 抽查 `WANG_DENG_FENG`（E2）和 `GONG_HUI_XIA`（E3）的 top10 overlay 与指标一致：CASE 偶尔收窄或移动部分假阳性，但仍存在大块漏检和假阳性，没有恢复真实高风险带。
4. 这回答了本轮核心问题：当病例目标被压到相近尺度后，网络**没有更可靠地学会完整 WSS 分布和 high-risk position**；相反，case-max 的尖峰尺度使大部分分布被压缩，尤其不利于 p99 与动态范围。

#### best / last 敏感性审计

| Run | best normalized `R²_cb` | last normalized `R²_cb` | last − best |
| --- | ---: | ---: | ---: |
| `E2-GLOBAL` | 0.4606 | 0.4455 | −0.0151 |
| `E3-GLOBAL` | 0.4218 | 0.4188 | −0.0030 |
| `E23-GLOBAL` | 0.4598 | 0.4591 | −0.0006 |
| `E2-CASE` | 0.1724 | 0.1748 | +0.0025 |
| `E3-CASE` | 0.1551 | 0.1556 | +0.0004 |

best/last 不改变容量、采样或归一化结论。虽然个别 last 的某项 test 指标略高，仍严格保留 `ckpt_best(train_loss)` 为主模型，不根据 test16 反选 checkpoint。

#### 产物完整性

- 5/5 run 均具有 `ckpt_best.pt`、`ckpt_last.pt`、`eval/ckpt_best/` 和 `eval/ckpt_last/`；日志未发现 traceback、OOM 或失败退出。
- 5 × 16 = **80/80** 个 best-test PostView case 包完整；每例具有对齐 STL、同点 CSV、点云/表面 VTP、共享色标三联图、top10 overlay、指标 JSON 与 normalization manifest。
- 80/80 manifest 的必需标量字段齐全，引用文件全部存在；表面 Gaussian 回插 mapping coverage 的最小值和均值均为 **100%**。
- 结果根目录：`training_wss_min/runs/pointnet_distribution_matrix/outputs/`。

### 4.3 导师补充指标：`CFD/CFDmax` 对 `Pred/Predmax`（2026-07-15）

导师补充要求比较每个病例内两个场各自除以自身最大值后的空间型态：

\[
y_{\mathrm{CFD,self}}=\frac{WSS_{\mathrm{CFD}}}{\max(WSS_{\mathrm{CFD}})},
\qquad
y_{\mathrm{Pred,self}}=\frac{WSS_{\mathrm{Pred}}}{\max(WSS_{\mathrm{Pred}})}
\]

这与原先已经保存的 `WSSpred/WSScfd,max` 不是同一个量：

- `WSSpred/WSScfd,max` 使用 CFD 真值尺度，保留预测幅值是否正确的信息；
- `WSSpred/WSSpred,max` 使用预测自身尺度，主动移除绝对幅值，只检查相对空间分布；
- 两个分母均为该病例峰值时相的**完整壁面点场最大值**，先算最大值再映射到 STL；Gaussian 表面回插只用于展示，不参与正式同点指标；
- 正比例缩放不改变点排序，因此 top10 high-risk 位置集合不会因 self-max 操作本身改变；新指标是对既有空间排序/定位结果的补充，不替代 top10 IoU 与距离指标。

既有 PostView 已保存完整壁面 true/pred，无需重新训练或重新跑模型前向。五组共 80 个 test case 已用纯后处理回填：

| Run | pooled R² | case-balanced R² | case R² med / P10 | R²<0 病例 | MAE / RMSE | 负预测点占比（病例均值） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`E2-GLOBAL`** | **−4.802** | **−4.857** | −7.083 / −14.812 | 16/16 | **0.185 / 0.247** | 0% |
| `E3-GLOBAL` | −6.557 | −6.605 | −6.915 / −17.654 | 16/16 | 0.215 / 0.281 | 0% |
| `E23-GLOBAL` | −5.454 | −5.422 | **−6.283** / −14.369 | 16/16 | 0.190 / 0.259 | 0% |
| `E2-CASE` | −8.814 | −8.922 | −12.088 / −22.508 | 16/16 | 0.250 / 0.321 | 7.34% |
| `E3-CASE` | −8.868 | −9.079 | −11.177 / −35.641 | 16/16 | 0.252 / 0.324 | 6.03% |

判读：

1. self-max 后五组的 16/16 逐病例 R² 均为负，说明主要问题不只是整体幅值偏小；分别把两张场都缩放到峰值 1 后，预测的相对空间型态仍明显错误。负 R² 表示误差大于“预测为 CFD 归一化场均值”的基线，**不等于所有归一化点值都为负**。
2. `E2-GLOBAL` 的 pooled/case-balanced R² 与 MAE/RMSE 相对最好，仍支持“导师宽网是本轮较优锚点”；但它的 self-max R² 远低于 0，因此不能据此宣称学会高 WSS 空间分布。
3. CASE 网络使用线性输出层，没有非负约束。`E2-CASE` 为 15/16 个病例出现至少一个负预测点，病例平均占 7.34%、最差 40.48%；`E3-CASE` 为 14/16，平均 6.03%、最差 20.83%。除以正的 `Predmax` 不会消除负数。正式指标保留未裁剪结果，并报告负值比例；如做裁剪图，必须标注为可视化/敏感性分析，不能覆盖正式结果。
4. GLOBAL 的物理 WSS 由 log 目标逆变换得到非负场，因此本批物理 self-max 字段无负点；这与 CASE 的线性 0–1 目标输出机制不同。

新增字段为 `wss_cfd_over_cfd_max`、`wss_pred_over_pred_max`、`err_wss_selfmax`、`abs_err_wss_selfmax`，已写入同点 CSV、点云 VTP 和表面 VTP；每例新增 `plots/fig_wss_selfmax_triptych.png`。五组汇总为 `outputs/selfmax_test_summary.json`，逐病例表为各 run 的 `postview/ckpt_best/test/selfmax_per_case_metrics.csv`。

### 4.4 导师追加深度探针 `E4-DEEP-GLOBAL`（2026-07-15 完训完评｜No-Go）

该探针不改写 §4.2 已结案的五组正式矩阵，只回答“在 E2 宽度锚点上增加编码/解码深度是否继续有收益”。

| Run | 局部编码 | global 拼接后解码 | 参数量 | 其余协议 | 状态 |
| --- | --- | --- | ---: | --- | --- |
| `E2-GLOBAL` | `6→256→512` | `1024→512→256→1` | 792,833 | train61 / FPS-2000 / global log-z / 400ep / train-loss | ✅ 已完成，配对锚点 |
| `E4-DEEP-GLOBAL` | `6→64→128→256→512` | `1024→512→256→128→64→1` | 874,561 | 与 E2 完全相同 | ✅ Job `8999` `COMPLETED (0:0)`｜**No-Go** |

E4 相对 E2 增加 `81,728` 个参数（10.3%），因此它主要是深度探针而非再次大幅加宽。Job `8999` 用时 `01:37:11`；400 epoch 齐全，best=第 379 epoch，train loss `0.121590`（E2 为第 364 epoch / `0.172548`，下降 29.5%）。

| best 全点指标 | `E2-GLOBAL` | `E4-DEEP-GLOBAL` | E4−E2 |
| --- | ---: | ---: | ---: |
| train 物理 `R²_field_cb` | 0.6124 | **0.6966** | +0.0841 |
| test 物理 `R²_field_cb` | **0.2140** | 0.1629 | **−0.0511** |
| 物理 train−test gap | 0.3985 | **0.5337** | +0.1352（变差） |
| train 归一化 `R²_field_cb` | 0.7717 | **0.8375** | +0.0658 |
| test 归一化 `R²_field_cb` | **0.4606** | 0.4476 | −0.0130 |
| 归一化 train−test gap | 0.3111 | **0.3899** | +0.0788（变差） |
| test 物理 MAE / RMSE（Pa，病例等权） | **2.8005 / 5.1141** | 2.8510 / 5.2778 | +0.0505 / +0.1637 |
| test 归一化 MAE / RMSE | **0.5732 / 0.7354** | 0.5799 / 0.7442 | +0.0067 / +0.0088 |
| test 物理 high-WSS R² | **−1.4861** | −1.6798 | −0.1937 |
| test 物理 top10 幅值比 | **0.3781** | 0.3328 | −0.0452 |
| test Spearman / top10 IoU | **0.6613** / 0.1457 | 0.6570 / **0.1531** | −0.0043 / +0.0073 |
| test 病例 R² mean / median / P10 | **0.1717 / 0.2010 / −0.0075** | 0.1390 / 0.0962 / −0.0288 | 全部变差 |
| test R²<0 病例 | **2/16** | 4/16 | +2 |
| test 双 self-max `R²_cb` / 负例 | −4.857 / 16 | **−4.208** / 16 | +0.649，仍 No-Go |

配对判读：

1. 更深网络确实提高了 train-fit，但物理/归一化 test R² 均下降，两种空间的 train−test gap 均扩大；这是更强的过拟合证据，不是容量突破。
2. 16 个 test 病例中，E4 的物理 R² 仅 6 例改善、10 例退化，配对中位变化 −0.0295；不是由少数离群例拉低总表。
3. top10 IoU 和双 self-max R² 有小幅改善，说明更深网络对部分相对型态/定位有弱信号；但 self-max 仍 16/16 负 R²，且高 WSS R²、幅值恢复、MAE/RMSE 和整体排序均变差，不足以改变 No-Go。
4. `ckpt_last` 的 test 物理/归一化 R² 为 `0.1618/0.4451`，top10 IoU `0.1540`，与 best 一致；结论不是 train-loss checkpoint 巧合。保留 E2 为 PointNet 锚点，不启动后续纯深度扫描。

完整产物审计：`ckpt_best/last`、best/last train61+test16 全点评估均齐全；best PostView 为 16/16 病例，manifest 引用文件无缺失，STL mapping coverage 全为 100%，self-max/top10 图均为 16/16。日志无 Traceback/OOM/NaN。上述 test16 只是与 E2 同协议的导师驱动配对证据，不是无偏最终测试。

进入 PointNet++ fine-tune 前的三层 SA foundation 已独立完成，但未启动训练。冻结结构为输入 FPS-2000，中心 `500→125→32`，radius `0.05→0.10→0.20`、nsample=16；代表病例 `fast/RAN_QING_BO` 的 source coverage 为 `99.90%→100%→100%`。总览 PNG、代表 group 图、逐层 VTP/CSV 和可复核 trace hash 位于 `例子/06_PointNet++_SA三层采样与分组/`。

## 5. 评价口径：优先检查空间分布和高风险位置

### 5.1 归一化空间分布

每个 test case 独立计算，再做病例等权汇总：

- 归一化空间 `R²`：中位数、P10、负 R² 病例数；
- 归一化 MAE / RMSE；
- Spearman 相关：检查病例内高低排序是否对齐；
- true/pred 的 p50/p90/p95/p99 和动态范围比；
- 绘制每例归一化目标的分布摘要，确认“各 case 数值更相近”确实发生，而不是只改了名称。

### 5.2 high-risk position

由于逐病例归一化后不再有可比的绝对 Pa 阈值，high-risk 主定义使用每例真值 **top 10%** 点，并固定报告：

- top10 IoU；
- top10 precision / recall；
- 峰值点欧氏距离，同时除以该病例坐标 bbox 对角线作尺度归一化；
- high-risk 区域质心距离；
- high-risk 区域连通性/主区域命中本轮暂缓，不引入未冻结邻接定义。

首要判断不是“pred max 是否等于 1”，而是高 WSS 区的位置、范围和排序是否与真值一致。

### 5.3 实验配对判读

- `E2-CASE` 只与 `E2-GLOBAL` 比，`E3-CASE` 只与 `E3-GLOBAL` 比。
- `E23-GLOBAL` 分别与 E2-GLOBAL（隔离 5000 点在宽网下的增量）及 E3-GLOBAL（隔离宽度在 5000 点下的增量）比较，只报告交互证据。
- 主结论回答“归一化是否改善病例内 pattern / hotspot”，不使用物理 Pa 指标判定逐病例归一化组。
- 在公式冻结前不预设数值 Go 线；至少要求归一化 R²/Spearman 和 top10 IoU/峰值距离中同时有“分布 + 定位”两类证据，不以单一 pooled R² 宣布成功。
- 因本协议取消 val，且 test16 会被用于多个连续实验的比较，本矩阵属于导师驱动的探索性实验；不将 test 结果宣称为未经反复使用的无偏最终泛化估计。
- `E0` 已显示 train–test 大 gap；后续 `E2/E3` 仍报 train-fit 与 test16，但**不以压低 train loss  alone 作为成功**。

## 6. 每个 test case 必须保存的可视化与数据

每个正式 run 的 test 评估至少保存：

1. 归一化 true / pred / absolute error 三联图，true 和 pred 强制共用同一色标范围。
2. true-top10 / pred-top10 / 交集与漏检的 high-risk 位置覆盖图。
3. 逐点结果文件，至少包含 `xyz`、`true_norm`、`pred_norm`、`abs_err_norm`、`true_highrisk`、`pred_highrisk`。
4. 逐例指标 JSON/CSV，并记录 `normalization_mode`、归一化公式和当例使用的统计量。
5. 同时将归一化字段写入 VTP 供 ParaView 表面查看；定量指标仍只在同点点云上计算。
6. 导师补充 self-max 图：`CFD/CFDmax`、`Pred/Predmax` 和二者绝对误差共用可比较色标；CSV/VTP 保留未裁剪负预测值，并在指标 JSON 中报告负点比例。

> `E0` 只有标准 `evaluate` 产物（`metrics.json` / `per_case_metrics.csv` / test heatmaps）；§6 完整清单已在 `E2/E3/E23` 五个正式 run 中全部补齐，完整性见 §4.2。

## 7. 执行顺序

1. ~~旧协议 Job `8970`~~ → ✅ 已完成（补充 No-Go，不代替 `E2-GLOBAL`）。
2. ~~对齐导师精确通道表并运行 `E2-GLOBAL`~~ → ✅ 已完成。
3. ~~运行 `E3-GLOBAL`：5000 点、random、不放回、每 epoch 重采样~~ → ✅ 已完成。
4. ~~运行 `E2-CASE` / `E3-CASE` 的 `WSS/WSSmax` 配对~~ → ✅ 已完成。
5. ~~运行 `E23-GLOBAL` 检查容量×点数交互~~ → ✅ 已完成。
6. ~~生成全部 test16 归一化空间图、high-risk 位置图并完成配对分析~~ → ✅ 已完成；结论见 §4.2。

## 8. 进度表

| 日期 | 事项 | 状态 | 证据/下一步 |
| --- | --- | --- | --- |
| 2026-07-14 | 最小 2×3 baseline | ✅ 完成 | Job `8700[0-5]`；PointNet+xyzgeom val `R²_cb=0.3015` |
| 2026-07-14 | `E0-GLOBAL`（无 val/400ep PointNet+xyzgeom） | ✅ 完训+完评 | 训 `8968_1`；eval `8975`；test16 `R²_cb=0.1414`，train 0.5303；见 §4.1 |
| 2026-07-14 | 同协议 xyz 消融 | ✅ 完评（不进主线） | `8968_0` / `8974`；test16 `R²_cb=0.0825` |
| 2026-07-14 | 旧协议 width=128 探针 | ✅ 完成｜补充 No-Go | Job `8970`；val `R²_cb=0.3071` vs 0.3015 |
| 2026-07-15 | 正式 `E2-GLOBAL` | ✅ 完训+完评 | Job `8976`；test 物理 `R²_cb=0.2140`，本轮最强，但 high-WSS 仍 No-Go |
| 2026-07-15 | 正式 `E3-GLOBAL` | ✅ 完训+完评 | Job `8977`；test 物理 `R²_cb=0.1637`，random-5000 单独增益弱 |
| 2026-07-15 | `E2-CASE` / `E3-CASE` | ✅ 完训+完评｜No-Go | Jobs `8979` / `8980`；整体分布 R²/Spearman 明显下降，见 §4.2 |
| 2026-07-15 | `E23-GLOBAL` 宽网+random-5000 交互对照 | ✅ 完训+完评 | Job `8978`；未超过 E2，未见明确协同效应 |
| 2026-07-15 | 五组 best/last + test16 PostView 审计 | ✅ 完成 | 5/5 run、80/80 case 包完整，mapping coverage 100% |
| 2026-07-14 | 完整产物流水线 | ✅ 实现并通过单测/smoke | 34 项单测通过；1 epoch 端到端 smoke 完成 train → best/last eval → test STL/VTP（100% mapping）；临时产物已清理，未提交正式 GPU 作业 |
| 2026-07-15 | `E4-DEEP-GLOBAL` 深度探针 | ✅ 完训+完评｜No-Go | Job `8999` `COMPLETED (0:0)`；test 物理 `R²_cb=0.1629` < E2 `0.2140`，gap 扩大；见 §4.4 |
| 2026-07-15 | PointNet++ 三层 SA foundation | ✅ 可视化完成｜未训练 | `500/125/32` 中心；VTP/CSV/PNG/manifest 完整；40 项训练模块单测通过 |
| 2026-07-15 | §9 第一性原理重排 + 对接第五轮天花板 + 对抗性审查 | ✅ 文档更新｜No-Run | 对接 `runs/_round5` 信息天花板；核实 E2 test16 0.214<PointNeXt 0.31、AAA65+ILO106 例壁面 WSS 有效（p99≈19–23 Pa）、无多队列 split；重排 P0/P0b/P1/P2/P3，新增 §9.4 对抗性审查；待导师批 |

## 9. 下一阶段优化方向（第一性原理重排 · 待讨论 · 不自动开跑）

> 本节 2026-07-15 按第一性原理重写：先对接第五轮已结案的信息天花板结论（§9.0），再据此把「提高预测精度」的杠杆按 **物理可解释性 × 可部署性 × 成本** 重排（§9.1–§9.3），最后对本重排本身做对抗性审查（§9.4）。所有条目仍为 **No-Run**，等导师确认 §9.2 的 P0 开发协议与 P1/P2 范围后再冻结最小矩阵。

### 9.0 第一性原理定位与第五轮对接（关键）

WSS 的物理定义是 \(\tau_w=\mu\,(\partial u/\partial n)|_{\text{wall}}\)：由**几何** + **入口流量波形** + **出口流量分配**三者共同决定。第五轮（PointNeXt、AG 单队列、dev1）已对这三项做过量化结案（`runs/_round5/final_report/round5_final_report.md`），结论必须先接住，否则本矩阵会重复其已排除的路径：

1. **入口无信息**：AG 86 例入口为共享人群模板（Fourier 波形 + Carreau-Yasuda 黏度全同，实测流量 CoV≈1.85e-4），逐例唯一差异是入口面积（几何）。
2. **出口是唯一隐藏杠杆，且不可部署**：4 髂动脉出口各自 RCR 三元 Windkessel，逐例几乎全不同；单支峰值流量占比在 4%–46% 间摆动（~39pp），这是跨病例 WSS 残差的主控变量，但属 `oracle_non_deployable`，不在壁面点云里。
3. **标签可信**：CFD 复现 floor ~2%，`R²_cap≈0.92–0.96`，因此天花板是**信息/表示**上限，不是标签噪声。
4. **几何-only 天花板**：dev1 达 `R²_field≈0.31 / casemean≈0.21`，且拟合足（A0D 四例 0.98–0.99）、预算非限、AG 池内 40 例后出现暂时平台、局部表示（邻域/半径归一化/global-context）无跨病例增量。

**本矩阵与第五轮的数值对接**：本矩阵最优 `E2-GLOBAL` 物理 test16 `R²_cb=0.214`，**低于**第五轮 PointNeXt 的 dev1 `≈0.31`。二者 split、协议、主干均不同，因此当前 0.214 与 0.31 之间同时含两条缝：

- **主干/容量缝（约 +0.09，可能可回收）**：本矩阵用的是导师指定的干净 PointNet，比第五轮的 PointNeXt（残差 + ball-query）弱；这段差距**未必是信息天花板**，可能只是主干不足。
- **信息天花板缝（约 0.31 处）**：越过主干缝后，几何-only 仍受第 2 条出口流量分配的硬约束。

> 结论：**「加宽 PointNet / 加点数 / 换归一化」这类在信息天花板以下的调参已接近收口**（本矩阵 E2/E3/E23/CASE 与第五轮 B-REP 双重证据）。要真正提高精度，杠杆必须落在 **(a) 把主干补到第五轮已验证的水平以回收容量缝、(b) 引入直指出口流量分配的可部署代理输入、(c) 用真实新队列扩大数据与多样性以检验平台是否 AG 同质性造成的假象、(d) 把目标改写成几何真正能学的「型态 + 热点」而非绝对幅值**——而不是继续堆 PointNet 宽度。

### 9.1 证据约束（更新）

- 天花板以下的调参已收口：宽通道相对 E0 一次性 +0.0725 后，random-5000 单独增益弱、E23 未超 E2、`WSS/WSSmax` 与双 self-max 均 No-Go；这些与第五轮「局部表示无跨病例增量」互证，下一轮**不再盲目扩宽度/点数/换归一化**。
- **当前「型态学得好」的表象被高估**：log-z normalized `R²_cb=0.4606`、Spearman 0.661 看似不差，但同一批预测在 self-max（`Pred/Predmax` vs `CFD/CFDmax`）下 16/16 例 R² 全负（pooled −4.8）。差异来源是 log 压缩 + 保留了病例间「整体水平」这一相对容易、且部分由血管尺寸几何决定的方差；去掉水平后暴露出**病例内精细型态与峰值定位仍弱**。因此下一轮**主指标改用 self-max（p99 稳健尺度）+ Spearman + top-k IoU**，不再以 log-z normalized R² 领头。
- **主要矛盾定位**：从「能否拟合训练集」转为「主干是否补到位 + 输入信息是否够 + 目标是否可学」；而输入信息的物理主控变量（出口流量分配）已知不可部署，只能用几何代理逼近。
- **test16 已被连续复用**（E0/E2/E3/E23/E2-CASE/E3-CASE/E4），它已不是未使用的最终测试集，其 R² 对「选模」而言是乐观偏置。**当前 split 上没有一个干净的无偏泛化估计**，这是 §9.2-P0 必须先补的方法学前提。

### 9.2 重排后的优先级（第一性原理）

排序依据：先补方法学前提与最便宜的判别性实验，再上两条真正的精度杠杆（信息 + 数据规模），最后才是目标改写；纯容量/采样/归一化扫描全部降级。

| 优先级 | 方向 | 为什么现在做（第一性原理） | 最小可证伪动作 | 进入/成功判据 |
| --- | --- | --- | --- | --- |
| **P0（前提）** | 恢复无泄漏病例级开发协议 | test16 已被反复使用，任何「涨点」都含选模偏置；没有干净 dev 就无法可信判断任何杠杆 | train pool 内做 grouped repeated holdout 或 grouped K-fold/OOF；**按病人分组**（ILO before/after 同病人、重复几何 `HOU_SHEN_QIAN=KANG_XI_MING` 必须整组同 fold）；预注册 split/统计量/选模规则；test16 只在方案锁定后审计**一次** | 有 OOF 曲线可选模；不再用 test16 逐轮调参 |
| **P0b（便宜判别）** | 把第五轮 PointNeXt 主干搬到当前 split | 直接判定 0.214→0.31 的「容量缝」是否可回收；`pointnext.py` 已存在，一次作业即可再锚定，避免继续在弱主干上做结论 | 用 `pointnext.py` 在 `split_v1_traintest` 上按 E2 协议跑一臂；与 E2-GLOBAL 严格配对 | 若逼近 ~0.31 → 容量缝真实、优先补主干；若仍 ~0.21 → 天花板随 split 前移，转向 P1/P2 |
| **P1（最高真实杠杆·导师已点名·第五轮未测）** | 真实新队列扩数据 + 病例数 learning curve | 第五轮平台只在**同质 AG 池**内出现（40 例），从未测跨解剖扩容；导师已明确要求先用真实 AAA/ILO 做 learning curve 判定「数据量是否瓶颈」再谈合成数据。现有 AAA 65 + ILO 106 例壁面 WSS 有效（峰值 p99≈19–23 Pa，与 AG≈20 Pa 同量级，可比） | 前置门：复用 `bc_audit` 脚本刻画 AAA/ILO 入口/出口 BC 是否与 AG 共享模板；再分层（cohort 作特征、按病人分组、控制点预算）跑 AG-only → AG+AAA/ILO 的病例数曲线，逐队列 + 混合分别评 | 平台随数据/多样性上移，且 **AG-test16 不退化** → 数据是瓶颈；否则跨解剖再证几何-only 天花板（本身可发表） |
| **P2（可部署·直指隐藏变量）** | 出口流量分配的物理代理**空间特征** | 出口 RCR 是已证的隐藏主控变量且不可部署；唯一可部署代理是出口截面积（几何免费）。第五轮 B1 只用了**全局标量**出口面积（≈+0.03，边际），未试把 Murray 定律（流量∝r³）预测的**逐分支流量占比映射到每个壁面点** | 在 E2/PointNeXt 锚点上，仅新增：分支归属 + 上下游面积比 + 距分叉距离构成的逐点「流量占比代理场」；保持采样/target/loss 不变 | 必须**专门降低远端(髂)分支误差**；若无效则确认几何无法代理流量分配（亦为有价值结论）。Murray 在瘤体/病变段失效，预注册 null |
| **P3（目标改写·诚实指标）** | 把「尺度」与「型态」拆开 + loss 直接约束热点 | 幅值受信息天花板限，几何真正能学的是病例内相对型态与热点位置（临床初筛也主要用这个）；应把可学部分单独优化并如实汇报 | 预注册 `global log-z` 锚点，对照双头 `shape + case-level scale`（scale 必须由网络/可部署几何预测，不借测试真值）；在锚点上加**一个**可微 rank/quantile/hotspot 辅助项，保留全场 MSE；不重复既有仅调 raw-Huber/target-weight 的路线 | 同时改善 self-max/Spearman 与 top-k IoU/峰值·质心距离；不能只改 MAE、不能退化全场 R² |
| **P4（降级·已收口）** | 纯容量/深度/点数/归一化扫描 | 本矩阵 + 第五轮双重证据显示已近天花板；E4-deep(8999) 已证实 train-fit 增强但 test 退化；`WSS/WSSmax` 已 No-Go | 不主动扩；仅在 P0b 判定容量缝真实时，才把「补主干」并入 P0b，不再单独堆 PointNet 宽度/深度 | 仅作为 P0b 的附带结论存在 |
| **P5（依赖 P1 结果·非本轮）** | 合成几何补数据（GAN/扩散） | 导师提议，但前置是「真实数据 learning curve 证明数据量确是瓶颈」+「合成几何的 WSS 标签来源与配准口径先解决」 | 仅当 P1 判定数据量为瓶颈后再评估 | P1 未证瓶颈前不启动 |

### 9.3 输出约束与暂不优先项

1. 逐病例 `[0,1]` 目标若继续，`Sigmoid`/`Softplus` 只作**输出非负性敏感性实验**并同时报原始/裁剪指标；能消负点但不解型态错误，不列 P0/P1。
2. high-risk 主定义仍固定为每例 CFD 真值 top10%（top5% 次指标）；正比例 self-max 不改排序，**不得**把「归一化后 top10 不变」当作新增定位能力。
3. 下一轮主判据必须同时含两类证据——分布（self-max/Spearman/动态范围）与定位（top-k IoU、峰值/质心归一化距离）；不得以单一 pooled R²、NMAE 或一张平滑表面图宣布 Go。self-max 用 p99 稳健尺度、不用单点 max。
4. 当前仅形成讨论清单：**未生成下一轮配置、未提交新作业**。导师确认 P0 开发协议 + P0b/P1/P2 范围后再冻结最小矩阵。

### 9.4 对抗性审查（对本重排本身，2026-07-15）

对 §9.0–§9.3 逐条自证伪，避免把「看似合理的方向」写成结论：

1. **「扩数据能提精度」可能是把「数据量」与「多样性」混为一谈。** AG 池 40 例平台可能来自**同质性**而非样本不足；混入 AAA/ILO 同时改变了量与多样性，单条混合曲线无法把增益归因于「量」。→ 缓解：P1 必须同时跑逐队列曲线（AG-only 延长、AAA-only、ILO-only）与混合曲线，令「量 vs 多样性」可分离。
2. **跨队列合并可能拉低 AG，而非抬高。** 三队列峰值 WSS 量级可比（去风险），但解剖形状先验不同、网格密度差 4–6×（AG≈1.3 万 vs ILO≈8.1 万壁面点），FPS 覆盖与点预算不匹配；AAA/ILO 入口/出口 BC 是否共享 AG 模板**尚未刻画**，若不共享则混合模型面对更多隐藏 BC 方差。→ 缓解：cohort 作特征、统一点预算、逐队列评估、并把「AG-test16 不退化」设为硬验收门。
3. **P0b 可能直接推翻本节的天花板叙事。** 若 PointNeXt 在当前 split 上仍只有 ~0.21，则天花板随 split 前移；若达 ~0.31，则当前 PointNet 只是容量不足、「信息天花板」对本 split 属**过早断言**。这正是把 P0b 排在便宜前置的原因——**在 PointNeXt + OOF 复核前，不得把第五轮 dev1 的天花板直接钉在本矩阵 test16 上**。
4. **Murray 代理很可能边际甚至系统性错。** 第五轮 B1 全局面积仅 +0.03；Murray 定律在瘤体/病变/分叉段常失效，恰好在 WSS 最关键处代理偏差最大。→ 处理：预注册 null 为有价值结论（证明几何无法代理流量分配），不把 P2 写成必胜项。
5. **self-max R² 可能被单点 max 放大而过苛。** 病例 `max/p99` 中位 2.36，除以尖峰 max 对单点异常敏感，可能低估真实型态技能。→ 缓解：self-max 一律用 p99 稳健尺度，并与 Spearman/top-k 三角互证，不单独据 self-max 下结论。
6. **test16 复用使任何「涨点」含选模偏置。** 跨这些杠杆在 test16 上测得的改进都部分是「按 test 选出来的」；唯一干净信号是 P0 的 OOF + 最终对 test16 一次性审计。**严禁**在 test16 上迭代。
7. **不可无视导师给本矩阵设定的框架。** 本矩阵是导师要求的干净 PointNet baseline + 指定消融；整体转向 PointNeXt + 多队列属方向变更，必须提交导师确认，不能单方面开跑——故本节维持 No-Run，与既有 §9.3.4 及第五轮 No-Run 立场一致。

## 10. 配置与运行入口（2026-07-14）

- 五份配置：四个主矩阵配置加一个可选 `E23-GLOBAL`，位于 `training_wss_min/configs/pointnet_distribution_matrix/`。
- 第一阶段清单：`configs/sweeps/pointnet_distribution_global_stage1.txt`；第二阶段清单：`pointnet_distribution_case_stage2.txt`，不会自动串联提交。
- 可选交互清单：`configs/sweeps/pointnet_distribution_e23_optional.txt`，只包含 `E23-GLOBAL`。
- 单配置流水线：`cluster/run_pointnet_distribution_matrix.slurm`。训练日志仅报告采样目标空间 loss/MAE/RMSE；完整 R²/Spearman/top10/位置指标由训练后的全点 eval 生成。
- E4 配置：`configs/pointnet_deeper/e4_deep_global.json`；提交入口：`cluster/pointnet_deeper/submit.sh`；Job `8999` 已完训完评，复用上述完整流水线。
- 三层 SA foundation：`configs/pointnetpp_sa_foundation/sa3_xyzgeom.json`；运行 `python -m training_wss_min.tools.visualize_pointnetpp_sa` 只生成结构审计图件，不训练模型。
- evaluator 分别写 `eval/ckpt_best/` 与 `eval/ckpt_last/`；PostView 只为 best 导出全部 test16。STL 只保存几何，标量保存在 VTP/同点 CSV。

### 10.1 本轮 Slurm 提交记录

- 提交时间：2026-07-14；执行入口：`cluster/run_pointnet_distribution_matrix.slurm`。
- GLOBAL：`E2=8976`、`E3=8977`、`E23=8978`。
- CASE：`E2-CASE=8979`、`E3-CASE=8980`；依赖统一冻结为 `afterok:8976:8977`。
- 完成状态（2026-07-15 核验）：五个 Job 均 `COMPLETED (0:0)`；队列中无本矩阵残留任务。
- 用时：`8976=01:21:36`、`8977=00:08:20`、`8978=00:08:20`、`8979=01:20:40`、`8980=00:07:58`。固定 FPS-2000 宽网组明显更慢主要与该协议的数据准备/固定 FPS 路径有关，不能把墙钟时间差直接解释为容量代价。
- 每个 Job 已自动完成训练、best/last 的 train61+test16 全点评估，以及 best 的 test16 STL/VTP 导出；未根据 test 反选 checkpoint。
- E4 追加记录：`8999` 于 2026-07-15 `18:31:12–20:08:23` 运行，用时 `01:37:11`，状态 `COMPLETED (0:0)`；best/last eval 和 best test16 PostView 齐全。
## 2026-07-20｜PointNet++ SA1 覆盖/重叠矩阵（single seed=1234；historical test27）

**状态**：17/17 新 run 已完成且审计通过。Q1V-10476（vertex-random5000、SAME、FPS center 500/125/32、106/0/27、seed=1234、width32、ball16）为主基准；Q2V-10477 仅为 SEP 配套敏感性基准。全部结论都基于 historical test27 的同协议工程比较，**不是独立确认结论**；本轮没有 3-seed，也未按 test27 重选 checkpoint（主结果均取 `ckpt_best(train_loss)`）。

审计：冻结 manifest SHA、seed、split、400 epoch、best/last checkpoint、best/last test27、27 个病例、有限数值与冻结 SA 协议均逐项检查。几何真源 `sa_grouping_*_geometry_audit.json` 均为 passed。adaptive_cover 在所有对应 contract 上 coverage=100%、最大 pair overlap≤1/3；实际 group size 为 1–84（random）、1–24（fixed-FPS）、1–26（FPS-multistart），故 `nsample=16` 是补充目标而不是硬上限。KNN-cover 同为100%覆盖，但最大重叠仍为0.875/0.900；raw KNN-8/10 覆盖率仅 0.511/0.684 与 0.575/0.782（min/median）。ball16/ball32 覆盖率分别为 0.201/0.303 与 0.382/0.550（random）；ball32 的更高覆盖来自更大且更重复的邻域（最大重叠均为1）。

### 1) nsample × width（统一对照：Q1V n16/w32）

| 配置 | 物理R²_cb（Δ） | 归一化R²_cb（Δ） | 病例均值R²（Δ） | 负例 | MAE（Δ） | Spearman（Δ） | top10 IoU（Δ） | high-WSS R²（Δ） |
|---|---|---|---|---|---|---|---|---|
| Q1V random5000 + ball16（复用） | 0.2763 (0.0000) | 0.5982 (0.0000) | 0.2114 (0.0000) | 1 | 2.8844 (0.0000) | 0.7222 (0.0000) | 0.1803 (0.0000) | -0.5134 (0.0000) |
| Q1V n32 / w32 | 0.2438 (-0.0325) | 0.5996 (0.0014) | 0.1750 (-0.0365) | 2 | 2.9384 (0.0540) | 0.7199 (-0.0023) | 0.1663 (-0.0140) | -0.5975 (-0.0840) |
| Q1V n16 / w64 | 0.2600 (-0.0163) | 0.5932 (-0.0050) | 0.1891 (-0.0223) | 3 | 2.9517 (0.0673) | 0.7258 (0.0036) | 0.1812 (0.0009) | -0.5276 (-0.0142) |
| Q1V n32 / w64 | 0.2364 (-0.0399) | 0.5835 (-0.0147) | 0.1697 (-0.0417) | 2 | 2.9747 (0.0903) | 0.7246 (0.0024) | 0.1695 (-0.0108) | -0.5450 (-0.0316) |

### 2) SA1 grouping（统一对照：Q1V ball16）

| 配置 | 物理R²_cb（Δ） | 归一化R²_cb（Δ） | 病例均值R²（Δ） | 负例 | MAE（Δ） | Spearman（Δ） | top10 IoU（Δ） | high-WSS R²（Δ） | 覆盖率 min/median | 最大重叠 | 组大小 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1V random5000 + ball16（复用） | 0.2763 (0.0000) | 0.5982 (0.0000) | 0.2114 (0.0000) | 1 | 2.8844 (0.0000) | 0.7222 (0.0000) | 0.1803 (0.0000) | -0.5134 (0.0000) | 0.2006/0.3025 | 1.0000 | 2–16 |
| Q1V ball32 | 0.2504 (-0.0259) | 0.5880 (-0.0103) | 0.1858 (-0.0256) | 2 | 2.9519 (0.0675) | 0.7150 (-0.0072) | 0.1703 (-0.0100) | -0.5632 (-0.0498) | 0.3816/0.5504 | 1.0000 | 2–32 |
| Q1V raw KNN-8 | 0.2712 (-0.0051) | 0.5970 (-0.0012) | 0.2066 (-0.0048) | 1 | 2.9147 (0.0303) | 0.7199 (-0.0023) | 0.1709 (-0.0094) | -0.4954 (0.0180) | 0.5110/0.6840 | 0.8750 | 8–8 |
| Q1V raw KNN-10 | 0.2568 (-0.0195) | 0.5955 (-0.0028) | 0.1959 (-0.0155) | 2 | 2.9320 (0.0476) | 0.7227 (0.0005) | 0.1739 (-0.0064) | -0.5396 (-0.0262) | 0.5752/0.7818 | 0.9000 | 10–10 |
| Q1V KNN-8-cover | 0.2589 (-0.0174) | 0.6042 (0.0059) | 0.1917 (-0.0197) | 1 | 2.9443 (0.0599) | 0.7232 (0.0010) | 0.1717 (-0.0086) | -0.5181 (-0.0046) | 1.0000/1.0000 | 0.8750 | 8–84 |
| Q1V KNN-10-cover | 0.2425 (-0.0338) | 0.5952 (-0.0031) | 0.1739 (-0.0375) | 1 | 2.9511 (0.0667) | 0.7276 (0.0055) | 0.1903 (0.0100) | -0.5438 (-0.0304) | 1.0000/1.0000 | 0.9000 | 10–84 |
| Q1V adaptive_cover | 0.2233 (-0.0530) | 0.5885 (-0.0098) | 0.1764 (-0.0351) | 2 | 2.9812 (0.0968) | 0.7134 (-0.0088) | 0.1715 (-0.0088) | -0.6008 (-0.0874) | 1.0000/1.0000 | 0.3333 | 1–84 |

### 3) Q1V/SAME support（每列 grouping 与 random 同 grouping 对照）

| 配置 | 物理R²_cb（Δ） | 归一化R²_cb（Δ） | 病例均值R²（Δ） | 负例 | MAE（Δ） | Spearman（Δ） | top10 IoU（Δ） | high-WSS R²（Δ） |
|---|---|---|---|---|---|---|---|---|
| Q1V random5000 + ball16（复用） | 0.2763 (0.0000) | 0.5982 (0.0000) | 0.2114 (0.0000) | 1 | 2.8844 (0.0000) | 0.7222 (0.0000) | 0.1803 (0.0000) | -0.5134 (0.0000) |
| Q1V adaptive_cover | 0.2233 (0.0000) | 0.5885 (0.0000) | 0.1764 (0.0000) | 2 | 2.9812 (0.0000) | 0.7134 (0.0000) | 0.1715 (0.0000) | -0.6008 (0.0000) |
| Q1V fixed-FPS + ball16 | 0.2298 (-0.0465) | 0.5931 (-0.0051) | 0.1853 (-0.0261) | 2 | 2.9388 (0.0544) | 0.7193 (-0.0029) | 0.1876 (0.0073) | -0.6103 (-0.0968) |
| Q1V fixed-FPS + adaptive | 0.2440 (0.0206) | 0.5875 (-0.0010) | 0.1869 (0.0106) | 3 | 2.9321 (-0.0491) | 0.7136 (0.0002) | 0.1623 (-0.0091) | -0.5502 (0.0506) |
| Q1V FPS-multistart5000 + ball16（历史复用） | 0.2297 (-0.0466) | 0.5880 (-0.0102) | 0.1752 (-0.0362) | 2 | 2.9582 (0.0738) | 0.7128 (-0.0094) | 0.1760 (-0.0043) | -0.6009 (-0.0875) |
| Q1V FPS-multistart + adaptive | 0.2163 (-0.0070) | 0.5829 (-0.0055) | 0.1662 (-0.0101) | 1 | 2.9796 (-0.0015) | 0.7191 (0.0057) | 0.1695 (-0.0020) | -0.6451 (-0.0443) |

### 4) Q2V/SEP support（每列 grouping 与 random 同 grouping 对照）

| 配置 | 物理R²_cb（Δ） | 归一化R²_cb（Δ） | 病例均值R²（Δ） | 负例 | MAE（Δ） | Spearman（Δ） | top10 IoU（Δ） | high-WSS R²（Δ） |
|---|---|---|---|---|---|---|---|---|
| Q2V random5000 + ball16（复用） | 0.2724 (0.0000) | 0.6210 (0.0000) | 0.2157 (0.0000) | 2 | 2.8783 (0.0000) | 0.7358 (0.0000) | 0.1568 (0.0000) | -0.5207 (0.0000) |
| Q2V random + adaptive | 0.2358 (0.0000) | 0.5820 (0.0000) | 0.1620 (0.0000) | 2 | 2.9833 (0.0000) | 0.7143 (0.0000) | 0.1768 (0.0000) | -0.5951 (0.0000) |
| Q2V fixed-FPS + ball16 | 0.2601 (-0.0123) | 0.5955 (-0.0255) | 0.1719 (-0.0438) | 5 | 2.9611 (0.0828) | 0.7175 (-0.0184) | 0.1732 (0.0164) | -0.5077 (0.0130) |
| Q2V fixed-FPS + adaptive | 0.2436 (0.0078) | 0.5896 (0.0076) | 0.1804 (0.0184) | 2 | 2.9424 (-0.0409) | 0.7132 (-0.0011) | 0.1784 (0.0016) | -0.5756 (0.0195) |
| Q2V FPS-multistart + ball16 | 0.2418 (-0.0306) | 0.5869 (-0.0341) | 0.1811 (-0.0346) | 1 | 2.9683 (0.0900) | 0.7105 (-0.0253) | 0.1600 (0.0032) | -0.5639 (-0.0432) |
| Q2V FPS-multistart + adaptive | 0.2314 (-0.0045) | 0.5829 (0.0009) | 0.1702 (0.0082) | 2 | 2.9818 (-0.0015) | 0.7119 (-0.0023) | 0.1775 (0.0007) | -0.6240 (-0.0289) |

**判读与下一步**：容量×邻域未显示一个跨 width 的稳定 nsample 增益；ball32 不构成 Go。raw KNN 的不完全覆盖伴随总体表现回退，KNN-cover 虽100%覆盖但高重叠下也没有稳定收益。adaptive 同时通过几何门禁，但在 Q1V/SAME 与 Q2V/SEP 的性能效应需结合全场、high-WSS、hotspot、负例及 best/last 敏感性审慎解读，单 seed/test27 不足以宣布最终 Go。fixed-FPS 与 FPS-multistart 的结论均只限本协议探索；尤其历史 Q1V FPS-multistart+ball16 仍须作为工程对照，不可写作新独立确认。

真源：`training_wss_min/preflight/sa_grouping_single_seed_results_analysis.json`、`sa_grouping_single_seed_results_summary.csv`、`sa_grouping_single_seed_per_case_deltas.csv`。

## 2026-07-21｜SA1-scale 矩阵（已提交待跑：门禁 `10746` → 训练 `10747_[0-16]%4`）

**动机**：SA1 分组矩阵显示 raw KNN-8 最接近对照且唯一改善 high-WSS，KNN-8-cover 归一化最高但物理回退。导师方向：①放弃 random5k、用全部原始 CFD 点 + 大 k（64/128/256）+ 全覆盖；②KNN-8-cover 基准下降 center、升邻域；③10k 采样 + FPS center；④cover 系补 w64 对照；⑤w64 下检验 PointNet 式大容量逐点前端能否为 PointNet++ 分层建模带来增益（先 Stem 6→32→64 vs 6→64→64 隔离前端瓶颈，欠拟合证据成立再上 6→256→512→64 + 64→128→256→512 主干）。

**协议**（与 SA1 分组矩阵一致）：split `106/0/27` v4-stratified、单 seed=1234、400 epoch、train-loss 选模、train106 global log-z（不混 A1b）、`legacy_vertex` 全点评估、eval fixed_support + chunk16384；预注册主指标 = 物理 `R²_cb`。SA1 全部 `knn_cover`（100% 覆盖硬门），SA2/3 保持 ball16。**test27 已多轮复用，本轮仍是同协议工程筛查，非独立确认**；胜出臂再进 3-seed/独立 split。

**17 臂**（对照复用：Q1V 锚点、`q1v_sa1_knn8_cover`、`q1v_sa1_knn10_cover`、`q1v_n16_w64`）：

| 家族 | run | 支撑 | SA center | SA1 k | width/Stem | batch |
|---|---|---|---|---|---|---|
| D1-fixed ×3 | `d1_allpts_fixed500_k{64,128,256}` | 全点 | 500/125/32 | 64/128/256 | 32 | 8 |
| D1-prop ×3 | `d1_allpts_prop10pct_k{64,128,256}` | 全点 | 比例 0.1N/0.25/0.25（三层） | 64/128/256 | 32 | **2**（干跑 k256@batch4 峰值 23.3GB，按 >19GB 门槛降档） |
| D2 ×4 | `d2_rand5000_c{250,125}_k{64,128}` | random5000 | {250,125}/125/32 | 64/128 | 32 | 8 |
| D3 ×3 | `d3_rand10000_fixed500_k{64,128,256}` | random10000（5 个 <1w 的 AG 例回退全点） | 500/125/32 | 64/128/256 | 32 | 8 |
| D4 ×2 | `d4_rand5000_knn{8,10}_cover_w64` | random5000 | 500/125/32 | 8/10 | 64 | 8 |
| D5-A ×1 | `d5_rand5000_knn8_cover_w64_stem32_64` | random5000 | 500/125/32 | 8 | 64 / Stem 6→32→64 | 8 |
| bridge ×1 | `bridge_rand5000_fixed500_k64` | random5000 | 500/125/32 | 64 | 32 | 8 |

**设计要点**：D2 中 250×64 与 125×128 等预算（16000 连接 = 基线 500×8 的 4×），构成同预算粗细粒度对比；Q1V(5k)→D3(10k)→D1-fixed(全点) 逐 k 列构成点数标度线，bridge 把 5k 上的 k64 从点数效应中分离；D1-prop 检验跨队列 center 密度一致是否重要（与 Q1V 不可直接比，参照系为 D1-fixed）。**D5-B（Stem 6→256→512→64）条件触发**：D5-A 主指标 ≥ `d4_rand5000_knn8_cover_w64` 且 train loss 仍显欠拟合，用 `prepare --variant-b` + `submit --variant-b` 单独提交。

**判读注意**：①D1-prop 的 batch=2 与其余 batch=8 存在优化混杂，家族内 k 网格内部可比、跨家族比较需声明；②D3 对 AG 约等于全点（AG 原始 9k–14k），实际主要检验 AAA 的 1w 下采样；③固定 500 center 下同一 k 在 AG（1w 点）与 AAA（最大 10.4w 点）处于完全不同覆盖/重叠区间，分队列判读；④k>100 的 SA1 KNN 走新的分块 cdist+topk 精确路径（k≤100 与历史逐位一致）。

**状态**：2026-07-21 提交（门禁 `10746` 通过 → 训练 `10747_[0-16]`）；2026-07-22 **17/17 全部完成并通过审计**。**结果与判读见下一节「2026-07-21｜PointNet++ SA1-scale 矩阵结果」**。真源：`training_wss_min/preflight/pointnetpp_sa1_scale_{prepared,submission,results_analysis}.json`。

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

## 2026-07-22｜bridge SEP 对照臂（门禁 `10804` → 训练 `10805`，均 COMPLETED｜✅完成｜No-Go）

**动机**：上表 `bridge random5000 + 500c + k64`（`bridge_rand5000_fixed500_k64`）沿用 Q1V/SAME 支路；用户要求补一个唯一变量为 SAME→SEP 的对照臂，检验 bridge 的“5000 点下 k64 单独 -0.0324”这一判读是否受 SAME/SEP 支路影响。

**配置**：`bridge_rand5000_fixed500_k64_sep`，逐字段 diff 确认仅 `data.query_mode: same → independent`；random5000 support、固定 center 500/125/32、SA1 `knn_cover k=64`、width=32、batch_cases=8、seed=1234、400 epoch、train-loss 选模、`legacy_vertex` 评估口径均与父实验相同。配置：`training_wss_min/configs/pointnetpp_sa1_scale_bridge_sep_followup_20260722/bridge_rand5000_fixed500_k64_sep.json`。

**状态**：门禁 Job `10804`、训练 Job `10805` 均 `COMPLETED (0:0)`；400 epoch、best/last checkpoint、27 例 CSV、无 NaN/Inf，配置哈希与 `runtime_preserves_frozen_config`（除 `query_mode` 外逐字段一致）均通过审计。提交记录：`training_wss_min/preflight/bridge_sep_followup_submission.json`。

| 配置 | 物理R²_cb（Δ） | 归一化R²_cb（Δ） | 病例均值R²（Δ） | 负例 | MAE（Δ） | RMSE（Δ） | Spearman（Δ） | top10 IoU（Δ） | high-WSS R²（Δ） | p99比（Δ） |
|---|---|---|---|---|---|---|---|---|---|---|
| bridge random5000 + 500c + k64（SAME，复用父实验） | 0.2439 (0.0000) | 0.6139 (0.0000) | 0.1910 (0.0000) | 2 | 2.9345 (0.0000) | 6.5294 (0.0000) | 0.7351 (0.0000) | 0.1694 (0.0000) | -0.5651 (0.0000) | 0.6688 (0.0000) |
| **bridge random5000 + 500c + k64 SEP（唯一变量 SAME→SEP）** | 0.2329 (-0.0111) | 0.6065 (-0.0074) | 0.1688 (-0.0222) | **1** | 2.9693 (+0.0348) | 6.5770 (+0.0476) | 0.7235 (-0.0116) | 0.1776 (+0.0082) | -0.6019 (-0.0368) | 0.6774 (+0.0086) |

**判读**：与 Q1V/Q2V 的混合式 SAME/SEP 差异不同，bridge 家族的 SEP 在绝大多数主指标上同向回退——物理/归一化 R²_cb、病例均值 R²、MAE、RMSE、Spearman、high-WSS R² 全部劣于 SAME；仅负例（2→1）、top10 IoU（+0.0082）与 p99 比（+0.0086）小幅改善，幅度均小于回退幅度。判定 **bridge 下 SAME→SEP 为 No-Go**；不将该 SEP 臂并入 bridge 后续候选，也不改变 SA1-scale 矩阵①点数标度、③降center×大邻域两节已有的判读（`bridge_rand5000_fixed500_k64` 的 SAME 结果继续代表该点数标度对照）。`last` 相对 `best` 的物理 R²_cb 为 `+0.0179`（last=0.2508），不改变 train-loss 选模规则，也不逆转上述 No-Go。仍是历史 test27 同协议工程比较，非独立确认。真源：`training_wss_min/preflight/bridge_sep_followup_results_analysis.json`、`bridge_sep_followup_results_summary.csv`。
