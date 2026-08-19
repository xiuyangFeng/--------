# WSS 高值区域预测优化方案

> **⏸️ 已暂停（2026-08-07 归档）**：2026-08-06 起模型开发收束到峰值体域 `u,v,w,p` 主线，WSS 侧优化全部暂停；本文保留为高 WSS 尾部/损失设计的候选思路库，Stage 2 之后若重启 WSS 线再回读。

> 日期：2026-07-28  
> 适用路线：`training_wss_min` / WSS 标量场预测  
> 当前任务：血管壁几何点云与几何特征 → 峰值时相壁面 WSS  
> 核心目标：在不破坏中低 WSS 和全场性能的前提下，提高高 WSS 区域的幅值、局部排序和热点定位精度。

> **路线边界与 2026-08-04 相关证据**：本文主体研究的是
> `training_wss_min` 的“几何点云 → 直接预测壁面 WSS”，不能与
> `wss_mri_calculator` 的“已知 CFD 速度场 → 后处理计算 WSS”混写。后者当前推荐的
> [Profile-Secant V3 结果模型](../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/PROFILE_SECANT_HIGH_TAIL_V3_RESULTS.md)
> 已把 test35 pooled high-WSS R² 提高到 `0.9415`、平均峰值低估降到 `9.31%`，
> 但逐病例 high-WSS R² 均值仅 `0.7721`，仅 4/35 病例同时满足
> `R²≥0.90` 和峰值低估 `≤10%`。Oracle 审计表明当前剩余误差主要受近壁速度层
> 过远限制，而不是峰值位置未找到。这个结果证明：即使上游速度真值可用，严格的
> 病例级高 WSS 仍需要更靠壁的速度信息；它不代表 geometry-only 直接预测路线已经
> 达到同样精度。该选择面向总体和平均高 WSS 指标；V4 final 仅作为冻结基线保留。

> **⚠ 阅读提示（2026-07-26 追加）**：本文档 §1–§12 为初稿方案。§13–§17 是基于落盘数据与已存 run 产物的第一性原理重估与对抗性审查，**推翻或大幅修订了 §1、§2.1、§4、§5.4、§9.3、§10.2、§11、§12**。请先读 §13 的量化结论与 §16 的修订优先级，再回看 §1–§12 的原始论证。§14 逐条给出对 §1–§12 的判定。

> **当前执行决议（2026-07-29，优先于下文旧建议）**：
>
> 1. 当前工程筛选继续使用 mixed `138/0/36`，暂不建立 val、不立即做五折；代价是 test36 已成为复用的开发榜，不能再解释为无偏最终测试精度。
> 2. 暂不启用面积口径，正式配置保持 `surface_metric_mode="legacy_vertex"`；主汇总使用逐病例等权，避免 pooled vertex 被病例点数主导。
> 3. 日常选模指标做减法，只保留全场 \(R^2\)、physical MAE、high-WSS 条件误差和 top10 IoU；high-WSS Spearman 仅用于排序臂，幅值 ratio 仅作诊断。
> 4. 边界条件 Oracle 已在 S3 上完成并证明真实 RCR 含几何外信息，但本阶段不补 RCR 多 seed，也不开发 RCR 架构；O0 上首轮 H1/H2 均未过各自主 Gate，仅保留为历史证据。
> 5. REG-P10-LSA2 的 SAME/IND × MSE/H1/H2 六臂已完成。IND 主效应 `ΔR²_cb=+0.0014`，未过 `+0.012`；H1 在 SAME/IND 下均未过 IoU 主门。H2 在 SAME/IND 内均刚好过 high-WSS nRMSE 主门，但 IND-H2 相对 SAME-H2 的 R²、normalized R² 和 MAE 更差，因此不晋级 IND。
> 6. REG-P10-LSA2 + SAME + H2 的精确同-seed并发 control / +train-only 标准化 `log(local_radius)` 两臂已完成。处理臂 \(R^2_{cb}=0.3506\)、MAE `2.5238 Pa`、high-WSS nRMSE `0.06611`，相对并发 control 的 `ΔR²_cb=+0.0436`、`ΔnRMSE=-0.00295`，全部保护线通过；现晋级为新的单 seed 开发锚点，暂不做多 seed。

## 1. 结论摘要

当前下一步不应继续盲目增加 PointNet/PointNet++ 的宽度、深度，或在所有层堆叠 Transformer。更合适的主线是：

> **保留 S3 PointNeXt-R + LocalGeoPE 骨干，将单头全场 MSE 回归改成“全场回归 + hotspot 识别 + 尾部幅值校准 + 局部排序约束”，并行执行一次出口 RCR/流量分配 oracle 实验。**

这条路线可以回答两个决定性问题：

1. 高 WSS 失败是否主要来自现有损失函数和采样协议对中低值的偏好；
2. geometry-only 输入是否缺少决定病例间高 WSS 幅值的边界条件信息。

如果真实 RCR/流量条件能显著改善高 WSS，而打乱条件不能，则后续重点应转向边界条件或其可部署代理，不应继续依赖纯架构调参。

---

## 2. 当前表现与问题诊断

### 2.1 当前代表性结果

当前表现较好的 `REG-P10 + local-SA2` 大致为：

| 指标 | 当前结果 |
|---|---:|
| normalized case-balanced R² | 0.6425 |
| physical case-balanced R² | 0.3171 |
| 全场 Spearman | 0.742 |
| CFD top10% WSS 均值 | 21.89 Pa |
| 预测在 CFD top10% 区的均值 | 8.12 Pa |
| top10 幅值恢复比例 | 37.1% |
| p99 幅值比例 | 39.6% |
| top10 IoU | 0.194 |
| high-WSS 内 Spearman | 0.042 |
| physical high-WSS R² | -0.367 |

相关配置：

- [`training_wss_min/configs/pointnetpp_regp10_transformer_20260726/regp10_localtf_sa2_s1234.json`](../../training_wss_min/configs/pointnetpp_regp10_transformer_20260726/regp10_localtf_sa2_s1234.json)
- [`training_wss_min/objectives.py`](../../training_wss_min/objectives.py)
- [`training_wss_min/baseline_models.py`](../../training_wss_min/baseline_models.py)

### 2.2 这不是单纯的全局幅值缩小

当前模型的全场 Spearman 较高，但进入 high-WSS 区域以后 Spearman 接近 0，说明问题至少包含两部分：

1. **幅值校准失败**  
   高 WSS 被系统性低估，预测 top10 均值和 p99 仅恢复真值的约 40%。

2. **热点内部型态失败**  
   模型对全场高低趋势有一定能力，但对热点内部“哪个位置更高、热点边缘在哪里”几乎没有可靠排序。

因此，以下操作都不足以单独解决问题：

- 给预测结果统一乘一个后处理系数；
- 只提高 normalized R²；
- 只改变输出非负约束；
- 继续增加普通 MLP/Transformer 容量；
- 只优化全场 Spearman。

### 2.3 当前训练目标会奖励平滑预测

当前强配置仍使用：

- train-only global `log_z` 目标；
- 全点平均 MSE；
- `loss_weight_target=false`；
- `selection_rule=train_loss`；
- 400 epoch；
- `legacy_vertex` 指标口径。

这意味着占绝大多数的中低 WSS 点主导优化，网络预测条件均值即可获得较低损失。经过 log 变换后，原始物理空间的高 WSS 差异还会进一步被压缩。

现有代码中的 target weighting 主要是线性形式：

\[
w_i=1+\alpha\cdot \operatorname{scale}(y_i)
\]

它比纯 MSE 更重视高值，但仍没有直接约束热点类别、局部排序、热点边界和尾部校准。

### 2.4 geometry-only 可能存在信息上限

历史审计已经发现：

- 病例间入口流量近似相同；
- 真正具有较大病例间差异的是出口 RCR、出口压力和流量分配；
- 这些变量没有进入当前 geometry-only 网络；
- wall gauge pressure 明显比 WSS 更容易由几何预测；
- 更深网络提高 train-fit，却降低 test R² 并扩大泛化差距。

这说明高 WSS 误差可能不只是网络能力问题。若相似几何在不同出口条件下可以产生不同 WSS，确定性 geometry-only MSE 网络会学习这些可能解的条件均值，结果自然是热点被平滑、峰值被压低。

---

## 3. P0：先修评价和开发协议

在继续大规模训练前，建议先完成以下前置项。

### 3.1 当前保留 mixed 138/0/36

当前不改变 split，继续使用 mixed `138/0/36`：

- 138 例全部进入训练，避免为了临时 val 减少单次模型的训练样本；
- checkpoint 继续按 `train_loss` 选择，所有横向实验冻结 400 epoch、seed、父模型、采样和统计文件；
- test36 继续用于当前工程筛选和配对比较；
- grouped K-fold / repeated grouped holdout 后置到方案基本确定以后，再用于稳健性确认。

该选择对“模型精度”和“精度估计”有不同影响：

- 多保留训练病例可能使单次模型精度略高，不会因为 split 本身直接降低模型容量；
- 没有 val 时无法识别过拟合拐点，按 train loss 选择的 checkpoint 可能不是泛化最优点；
- test36 已被多轮用于选择方向，其结果存在 winner's curse，后续表格只能称为**复用开发集结果**，不能称为无偏最终测试精度。

因此本阶段目标是保持历史横向可比性，不再把 test36 数字用于对外声称最终泛化上限。若以后需要论文级最终结论，应补充 grouped OOF 或新的独立审计集。

### 3.2 暂不启用面积口径

当前正式配置固定：

```json
"surface_metric_mode": "legacy_vertex"
```

理由是现有面积 mapping 尚未覆盖全部 192 单元，部分病例无法严格合规。当前固定 5000 点/病例训练和逐病例等权汇总已经缓解病例间网格密度差异，但不能完全消除病例内局部网格疏密对 vertex 指标的影响。

执行约束：

- 日常实验不再以面积合同作为阻塞项；
- 不使用 pooled vertex 指标作为主结论；
- 面积指标后续只在合规病例子集上做敏感性审计；
- 面积 mapping 修复和 ILO 扩审后，再单独决定是否切换正式口径。

### 3.3 日常指标做减法

日常筛选只保留四项：

| 角色 | 指标 | 用途 |
|---|---|---|
| 全场护栏 | normalized case-balanced \(R^2\) | 防止全场能力明显退化 |
| 物理护栏 | physical case-balanced MAE | 防止物理误差恶化 |
| 尾部误差 | high-WSS physical nRMSE（必要时并列 MAE） | 测量高值条件误差 |
| 热点定位 | top10 IoU | 测量热点位置和形态 |

附加指标按实验类型启用：

- ranking loss：增加 high-WSS Spearman；
- 幅值/NLL/物理损失：top10 mean ratio、p99 ratio 只作诊断；
- 最终审计：再展开质心距离、固定阈值 precision/recall 和分域结果。

等基数 top10 定义下 precision、recall 与 IoU 冗余，日常只保留 IoU。high-WSS \(R^2\) 对病例内窄真值范围过于敏感，不作为主终点。

---

## 4. P1：边界条件信息上限 oracle

Oracle 固定在当前 **S3 PointNeXt-R + LocalGeoPE** 父模型上，除病例条件输入外保持 mixed `138/0/36`、seed1234、400 epoch、train-loss 选模、random5000/SAME 和 `legacy_vertex` 不变。

| ID | 输入 | 目的 |
|---|---|---|
| O0 | 当前 geometry-only | 严格配对基准 |
| O1a | 几何 + 四出口真实 \(R_1/R_2/C\) | 合法信息上限 Oracle |
| O2 | 几何 + 分区内、分层内随机打乱的 RCR | 排除增加维度和域标签伪增益 |
| O1b（暂缓） | 几何 + 峰值步出口流量占比 | 泄漏式信息探针，永不作为部署输入 |

2026-07-28 实查 `data_new/**/Global_conditions`：

- 入口流量与四出口压力监视文件可用于一致性审计；
- 四出口流量在峰值步可直接读取的病例为 **58/174**；
- 其余 **116/174** 存在出口流量文件缺失，或文件只有 step/time 而没有流量值；
- 因此本轮只提交 O0/O1a/O2，不为补齐 O1b 引入未经验证的流量反演。

日常只比较 §3.3 的四项核心指标。Oracle 的作用轴主终点为 high-WSS physical nRMSE；top10 IoU 用于判断病例级条件是否进一步改善空间热点，幅值 ratio 仅作诊断。

### 4.1 oracle 结果的决策规则

#### 如果 O1a 明显提高高 WSS，而 O2 不提高

说明 geometry-only 信息不足。优先路线应转为：

- 输入真实患者流量、压力或 RCR；
- 从医学检查或低维血流模型获得病例条件；
- 使用可部署的流量分配代理。

可部署代理可以包括：

- 分支出口面积；
- 基于 Murray 关系的 \(r^3\) 流量占比；
- 上游/下游截面积比；
- 分支 ID；
- 距分叉距离；
- 分支级局部半径统计；
- 入口面积与主干尺度。

更完整的框架可以是：

```text
几何 + 可用边界条件
          │
          ▼
1D 中心线血流网络
预测分支流量/压力
          │
          ▼
PointNeXt/GNN 壁面网络
预测逐点 WSS
```

#### 如果 O1a 也没有明显提高

优先检查：

- RCR 和壁面点的分支映射是否正确；
- 峰值时相与条件是否对应；
- 当前点云是否缺少热点所需的表面局部结构；
- 尾部标签是否受局部网格/插值影响；
- loss、采样和 decoder 平滑是否仍是主因。

### 4.2 2026-07-28 实际配置、结果与裁决

本轮配置、训练和 best/last test36 评估均已完成：

| 臂 | 配置 | 输入维度 |
|---|---|---:|
| O0 | `training_wss_min/configs/pointnetpp_rcr_oracle_20260728/o0_geometry_s1234.json` | 6 |
| O1a | `training_wss_min/configs/pointnetpp_rcr_oracle_20260728/o1a_true_rcr_s1234.json` | 18 |
| O2 | `training_wss_min/configs/pointnetpp_rcr_oracle_20260728/o2_shuffled_rcr_s1234.json` | 18 |

数据与运行合同：

- 174/174 病例均从 UDF 源文件提取四出口 \(R_1/R_2/C\)，取自然对数后使用 train138 病例等权统计做 z-score；
- O2 在 train/test 内分别按 AG、AAA rupture/unrupture、ILO 0/1 分层做无固定点置换；
- O0/O1a/O2 的父模型、训练协议和评估口径均冻结；
- GPU 前置检查 Job `11008` 为 `COMPLETED (0:0)`；
- 三臂训练与 best/last test36 评估数组 Job `11009_[0-2]` 均为 `COMPLETED (0:0)`；
- 三臂 `ckpt_best.pt`、`ckpt_last.pt`、best/last `metrics.json` 和逐病例 CSV 完整性检查通过。

`ckpt_best` 结果如下；Δ 均为 treatment − O0：

| 臂 | best epoch | physical \(R^2_{cb}\) | Δ\(R^2_{cb}\) | physical MAE (Pa) | ΔMAE | high-WSS nRMSE | ΔnRMSE | top10 IoU | ΔIoU | p99 幅值比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| O0 geometry | 350 | 0.2756 | — | 2.5974 | — | 0.07068 | — | 0.1772 | — | 0.3429 |
| O1a true RCR | 378 | **0.3561** | **+0.0805** | **2.4620** | **-0.1354** | **0.06728** | **-0.00340** | **0.2109** | **+0.0337** | **0.4356** |
| O2 shuffled RCR | 342 | 0.2723 | -0.0033 | 2.6053 | +0.0080 | 0.07148 | +0.00079 | 0.1840 | +0.0068 | 0.3240 |

分域 physical \(R^2\) 也支持相同方向：

| 臂 | AG | AAA | ILO |
|---|---:|---:|---:|
| O0 | 0.2903 | 0.2404 | 0.2671 |
| O1a | **0.4000** | **0.2681** | **0.3565** |
| O2 | 0.3318 | 0.2329 | 0.2209 |

逐病例配对 bootstrap 进一步显示：

- O1a − O0 的病例均值 Δ\(R^2\) 为 `+0.0536`，95%CI `[+0.0127,+0.1021]`，22/36 病例改善；
- O1a − O0 的 top10 IoU 均值差为 `+0.0337`，95%CI `[+0.0102,+0.0550]`；
- O2 − O0 的病例均值 Δ\(R^2\) 为 `-0.0134`，95%CI `[-0.0622,+0.0356]`，与零相容；
- O1a − O2 的 physical \(R^2_{cb}\) 差为 `+0.0838`，病例均值 Δ\(R^2\) 95%CI `[+0.0352,+0.1011]`。

**裁决：RCR 信息上限 Oracle 为 Go-to-follow-up。** O1a 同时优于几何基准和等维打乱负对照，说明四出口 \(R_1/R_2/C\) 包含当前几何输入没有表达的有效 WSS 信息。它主要改善幅值、全场误差和热点定位，但 high-WSS 内 Spearman 基本不变（`0.0359→0.0357`），因此不能把问题简化成“只缺 RCR”；病例内热点排序仍需局部几何/目标函数方向继续处理。

该结论仍受三项边界约束：单种子、test36 多轮复用、train-loss 选模。因此它授权后续验证 RCR 或可部署代理，不授权直接对外声称最终泛化提升。完整复算产物：

- `training_wss_min/preflight/rcr_oracle_matrix_20260728_results_analysis.json`
- `training_wss_min/preflight/rcr_oracle_matrix_20260728_results_summary.csv`
- `training_wss_min/preflight/rcr_oracle_matrix_20260728_paired_case_stats.csv`

### 4.3 2026-07-28 O0 热点/高值首轮：2/2 No-Go

RCR Oracle 保留为“几何之外存在条件信息”的诊断证据，但本阶段不继续补多 seed，也不开发 RCR 融合架构。执行父模型重新固定为 O0 geometry-only；RCR 特征和 `case_features_path` 均不得进入下一轮。

为避免重复既有 target-weight/raw-Huber 路线，本轮只提交两个新单变量臂：

| 臂 | 相对 O0 的唯一变化 | 主攻指标 | 配置 |
|---|---|---|---|
| H1 | 增加逐点第二输出通道；按每病例 q90 生成 top10 标签，加入病例等权、正负平衡 BCE，\(\lambda=0.20\) | top10 IoU | `h1_hotspot_bce_q90_lam020_s1234.json` |
| H2 | 保持单回归通道；在 MSE 上加入 q=0.90 pinball 辅助项，\(\lambda=0.20\) | high-WSS physical nRMSE | `h2_pinball_q90_lam020_s1234.json` |

共同冻结：mixed `138/0/36`、seed1234、S3 PointNeXt-R + LocalGeoPE、random5000/SAME、400 epoch、train-loss 选模和 `legacy_vertex`。H1 的热点 logit 只作为训练辅助；正式 WSS 指标仍由第一回归通道计算。

预注册 Gate 只使用四项日常指标：

- H1：`Δtop10 IoU ≥ +0.02`；
- H2：`Δhigh-WSS nRMSE ≤ -0.002`；
- 共同保护线：`Δnormalized R²_cb ≥ -0.01`，`Δphysical MAE_cb ≤ +0.05 Pa`；
- H1/H2 只有各自独立过门后才允许组合，不因单项幅值 ratio 变化自动晋级。

GPU 门禁 Job `11012` 与训练/best-last test36 评估数组 `11013_0/1` 均已 `COMPLETED (0:0)`。预注册主结果固定取 `ckpt_best(train_loss)`：

| 臂 | best epoch | physical R²_cb | ΔR²_cb | normalized R²_cb | Δnormalized | MAE_cb (Pa) | ΔMAE | high-WSS nRMSE | ΔnRMSE | top10 IoU | ΔIoU | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| O0 geometry | 350 | 0.2756 | — | 0.6300 | — | 2.5974 | — | 0.07068 | — | 0.1772 | — | 基准 |
| H1 hotspot BCE | 378 | **0.2990** | **+0.0234** | **0.6336** | +0.0035 | **2.5843** | -0.0131 | **0.06935** | -0.00133 | **0.1943** | **+0.0170** | **No-Go：IoU 主门差 0.0030** |
| H2 q90 pinball | 350 | 0.2947 | +0.0190 | 0.6274 | -0.0026 | 2.5917 | -0.0057 | 0.06996 | **-0.00072** | 0.1818 | +0.0046 | **No-Go：nRMSE 主门差 0.00128** |

分域 physical R²（AG / AAA / ILO）从 O0 的 `0.290/0.240/0.267` 提高到 H1 的 `0.314/0.256/0.296`、H2 的 `0.312/0.247/0.292`，两臂没有分域负迁移。逐病例配对证据也没有达到稳健正向：H1 的病例均值 ΔIoU 95%CI 为 `[-0.0004,+0.0343]`（24/12 改善/退化），H2 为 `[-0.0103,+0.0190]`（20/16）；病例均值 ΔR² 的区间均跨零。

**裁决**：

- H1 是接近 Gate 的正向筛选信号，但不能因其总体 R²、MAE 和三域同时改善而事后降低热点主门；
- H2 对 high-WSS nRMSE 的改善只有预注册门槛的约 36%，不补相邻 q/λ；
- 两臂均通过共同保护线但未过各自主门，按组合政策禁止 H1+H2；
- `ckpt_last` 不改变结论：H1 的 ΔIoU 约 `+0.0150`，H2 的 Δhigh-WSS nRMSE 约 `-0.00077`；
- 当时提出的下一项是 `query_mode="independent"`；该问题已由 §4.4 的完整 2×3 矩阵回答并判 No-Go，后续不再补 IND。

可复跑结果真源：

- `training_wss_min/tools/analyze_o0_hotspot_tail_results.py`
- `training_wss_min/preflight/o0_hotspot_tail_matrix_20260728_results_analysis.json`
- `training_wss_min/preflight/o0_hotspot_tail_matrix_20260728_results_summary.csv`
- `training_wss_min/preflight/o0_hotspot_tail_matrix_20260728_paired_case_stats.csv`

### 4.4 REG-P10-LSA2 正式开发锚点与 SAME/IND × H1/H2 矩阵

O0 是 RCR Oracle 内为病例条件输入建立的几何对照，并不是当前精度最高的 geometry-only 模型。后续热点与高 WSS 优化统一回到历史最高非 RCR 配置 **REG-P10-LSA2**：

- S3 `PointNeXt-R + LocalGeoPE`；
- DropPath `0.10`；
- 只在 SA2 启用 Local Transformer（4 heads、FFN ratio 2、dropout 0）；
- random5000、seed1234、400 epoch、train-loss 选模、`legacy_vertex`；
- 不使用 RCR/病例条件、面积口径或多 seed。

历史锚点来自 `regp10_localtf_sa2_s1234.json`，best epoch 378；test36 指标为 physical \(R^2_{cb}=0.3171\)、normalized \(R^2_{cb}=0.6425\)、MAE `2.5527 Pa`、RMSE `6.2314 Pa`、high-WSS nRMSE `0.06808`、top10 IoU `0.1940`。它仍是单 seed、复用 test36 的工程锚点，不是无偏最终泛化结论。

本轮不把 IND 单独追加在旧 O0 上，而是在新锚点上重跑完整并发对照：

| 索引 | Query | 目标 | 实验 ID |
|---:|---|---|---|
| 0 | SAME | MSE 并发基准 | `lsa2_same_mse_s1234` |
| 1 | SAME | H1 q90 hotspot BCE, λ=0.20 | `lsa2_same_h1_bce_q90_lam020_s1234` |
| 2 | SAME | H2 q90 pinball, λ=0.20 | `lsa2_same_h2_pinball_q90_lam020_s1234` |
| 3 | IND | MSE | `lsa2_ind_mse_s1234` |
| 4 | IND | H1 q90 hotspot BCE, λ=0.20 | `lsa2_ind_h1_bce_q90_lam020_s1234` |
| 5 | IND | H2 q90 pinball, λ=0.20 | `lsa2_ind_h2_pinball_q90_lam020_s1234` |

预注册比较只使用本轮并发控制：SAME H1/H2 各自减 SAME MSE；IND 主效应为 IND MSE 减 SAME MSE；IND H1/H2 各自减 IND MSE，并补充同一目标下 IND 减 SAME。H1 主门仍为 `Δtop10 IoU ≥ +0.02`，H2 主门仍为 `Δhigh-WSS nRMSE ≤ -0.002`；共同保护线为 `Δnormalized R²_cb ≥ -0.01`、`Δphysical MAE_cb ≤ +0.05 Pa`。IND 主门为 `Δphysical R²_cb ≥ +0.012`，保护线为 `Δnormalized R²_cb ≥ -0.01`、`ΔMAE_cb ≤ +0.03 Pa`、`Δhigh-WSS nRMSE ≤ +0.001`、`Δtop10 IoU ≥ -0.005`。

GPU 门禁 Job `11019` 与训练/评估数组 `11020_[0-5]` 均已 `COMPLETED (0:0)`；六臂均完成 400 epoch、best/last checkpoint、test36 全云评估和 36 例明细，配置哈希、有限值及产物完整性通过。

主结果固定取 `ckpt_best(train_loss)`；Δ 使用表中“并发对照”，而不是历史 O0 或旧 L-SA2 run：

| 臂 | 对照 | R²_cb | ΔR² | normalized R²_cb | Δnormalized | MAE (Pa) | ΔMAE | high-WSS nRMSE | ΔnRMSE | top10 IoU | ΔIoU | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SAME MSE | — | 0.2862 | — | 0.6390 | — | 2.5892 | — | 0.07040 | — | 0.1786 | — | 并发基准 |
| SAME H1 | SAME MSE | 0.3050 | +0.0188 | 0.6401 | +0.0011 | 2.5589 | -0.0303 | 0.06866 | -0.00174 | 0.1962 | +0.0175 | **No-Go：IoU 未到 +0.020** |
| SAME H2 | SAME MSE | **0.3239** | **+0.0376** | **0.6453** | **+0.0063** | **2.5371** | **-0.0521** | 0.06835 | **-0.00205** | 0.1800 | +0.0014 | **Go：候选锚点待复跑** |
| IND MSE | SAME MSE | 0.2876 | +0.0014 | 0.6367 | -0.0024 | 2.5825 | -0.0067 | 0.06991 | -0.00049 | 0.1921 | +0.0135 | **IND No-Go：R² 主门未达** |
| IND H1 | IND MSE | 0.3021 | +0.0145 | 0.6373 | +0.0007 | 2.5687 | -0.0138 | 0.06932 | -0.00059 | 0.1903 | -0.0018 | **H1 No-Go** |
| IND H2 | IND MSE | 0.3198 | +0.0322 | 0.6390 | +0.0023 | 2.5709 | -0.0117 | **0.06784** | **-0.00207** | **0.2017** | +0.0096 | H2 过门，但 **IND 不晋级** |

补充稳健性证据：

- SAME-H2 的 AG/AAA/ILO R² 均提高 `+0.0341/+0.0140/+0.0585`；病例 high-WSS R² 均值差 95%CI `[+0.0014,+0.5319]`，但病例全场 R² CI `[-0.0034,+0.0426]` 仍跨零；
- `ckpt_last` 下 SAME-H2 仍有 `ΔR²_cb=+0.0383`、`Δhigh-WSS nRMSE=-0.00204`，不改变 Gate；
- IND-H2 相对 SAME-H2 虽提高 IoU `+0.0217`，但 R² `-0.0041`、normalized R² `-0.0063`、MAE `+0.0337 Pa`，其中 MAE 超出 IND 的 `+0.03 Pa` 保护线；
- 并发 SAME MSE 比历史同配置 L-SA2 的 R² 低 `0.0309`，说明运行漂移不可忽略。因此所有结论只依赖本轮并发控制，历史 `0.3171` 不作为 Gate 基线。

**裁决**：关闭 IND 和 H1，不做 H1+H2 组合。SAME-H2 以“单 seed、复用 test36、刚过主门”的边界作为 §4.5 输入信息实验的冻结父配置；该后续实验已完成并由 `log(local_radius)` 处理臂过门，因此本节不再保留“待复跑”的未决状态。

可复跑真源：

- `training_wss_min/tools/analyze_regp10_lsa2_objective_ind_results.py`
- `training_wss_min/preflight/regp10_lsa2_objective_ind_matrix_20260728_results_analysis.json`
- `training_wss_min/preflight/regp10_lsa2_objective_ind_matrix_20260728_results_summary.csv`
- `training_wss_min/preflight/regp10_lsa2_objective_ind_matrix_20260728_paired_case_stats.csv`

### 4.5 SAME-H2 + `log(local_radius)` 输入信息臂

本轮以 §4.4 的 SAME-H2 为父配置，并发提交两个 seed1234 臂：`lsa2_h2_same_repro_s1234` 只改 `name/notes`，`lsa2_h2_logradius_s1234` 只在原6D输入末尾追加 train-only 标准化的自然对数半径。处理臂保留原始 `local_radius`，LocalGeoPE 仍使用 `[abscissa_norm, local_radius, curvature]`，因此没有混入特征替换、位置编码或结构改动。

两臂冻结 mixed `138/0/36`、SAME、H2 q90 pinball λ0.20、DropPath0.10、SA2 Local Transformer、random5000、400 epoch、train-loss 选模和 `legacy_vertex`。新列来自 control106 train-only 的3,208,800点，natural-log 均值/标准差为 `2.116623/0.611423`。Jobs `11032`、`11033_0/1` 均 `COMPLETED (0:0)`。

| 臂 | R²_cb | ΔR² | normalized R²_cb | Δnormalized | MAE (Pa) | ΔMAE | high-WSS nRMSE | ΔnRMSE | top10 IoU | ΔIoU | AG / AAA / ILO R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H2 concurrent control | 0.3070 | — | 0.6397 | — | 2.5580 | — | 0.06906 | — | **0.1916** | — | 0.325 / 0.248 / 0.313 |
| **H2 + log(radius)** | **0.3506** | **+0.0436** | **0.6440** | **+0.0042** | **2.5238** | **-0.0342** | **0.06611** | **-0.00295** | 0.1887 | -0.0029 | **0.363 / 0.271 / 0.377** |

预注册主门为 `ΔR²_cb≥+0.012` 或 `Δhigh-WSS nRMSE≤-0.002`；处理臂同时通过两项。`Δnormalized R²_cb=+0.0042`、`ΔMAE=-0.0342 Pa`、`ΔIoU=-0.0029`，三条保护线也全部通过。相对历史 ★ H2，处理臂仍有 `ΔR²_cb=+0.0267` 与 `ΔnRMSE=-0.00224`，但因训练轨迹漂移，历史结果不参与正式 Gate。

病例全场 R² 均值差为 `-0.0020`、95%CI `[-0.0326,+0.0281]`，16/20病例改善/退化；病例 high-WSS R² 为23/13且CI仍跨零。该反证要求保留：新增输入改善的是注册的聚合场指标和三域整体表现，尚不能声称对多数病例稳定增益。

**裁决**：`lsa2_h2_logradius_s1234` 晋级为新的正式开发锚点；直接输入列通过后，N-02 的“解析锚残差参数化”可以进入下一轮单变量筛选，但仍须相对这个新锚点设置并发 control。暂不补多 seed、RCR、面积、IND 或 H1+H2。

可复跑真源：

- `training_wss_min/tools/analyze_lsa2_h2_logradius_results.py`
- `training_wss_min/preflight/lsa2_h2_logradius_matrix_20260729_results_analysis.json`
- `training_wss_min/preflight/lsa2_h2_logradius_matrix_20260729_results_summary.csv`
- `training_wss_min/preflight/lsa2_h2_logradius_matrix_20260729_paired_case_stats.csv`

---

## 5. P2：尾部感知多任务损失

推荐的总目标为：

\[
L =
L_{\text{balanced-reg}}
+ \lambda_h L_{\text{hotspot}}
+ \lambda_r L_{\text{rank}}
+ \lambda_t L_{\text{tail-cal}}
+ \lambda_g L_{\text{gradient}}
\]

初始可以从以下权重开始：

| 项 | 初始权重建议 |
|---|---:|
| balanced regression | 1.0 |
| hotspot classification | 0.20 |
| local ranking | 0.05 |
| tail calibration | 0.10 |
| surface gradient | 0.05 |

这些数值只是第一轮起点，后续应检查各项梯度范数，避免某一个辅助项完全压过基础回归。

### 5.1 连续标签平衡回归

不要直接使用无上限的 inverse-frequency 权重。建议：

1. 在每个 fold 的 train-only `log1p(WSS)` 上估计标签密度；
2. 每个病例的总损失权重相同；
3. 病例内划分：
   - q0–q50；
   - q50–q80；
   - q80–q90；
   - q90–q95；
   - q95–q99；
   - q99–q100；
4. 权重平滑并限制最大值，建议不超过最低箱的 4–6 倍；
5. 同时保持病例平衡和必要的 cohort 平衡。

优先候选：

- Balanced MSE；
- Label Distribution Smoothing；
- 有上限的 effective-number weighting。

算法参考：

- [Balanced MSE for Imbalanced Visual Regression](https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html)
- [Delving into Deep Imbalanced Regression](https://proceedings.mlr.press/v139/yang21m.html)

### 5.2 hotspot 分类辅助头

在共享 backbone 特征上增加逐点 hotspot probability head：

\[
p_i=\sigma(f_{\text{hot}}(h_i))
\]

标签建议：

- 主标签：病例内 CFD top10%；
- 辅助标签：病例内 CFD top5%；
- 可选软标签：在 q85–q95 附近用连续过渡，减少硬阈值噪声。

损失可以使用：

\[
L_{\text{hotspot}}
=L_{\text{focal-BCE}}+L_{\text{soft-Dice}}
\]

分类头不替代 WSS 回归，而是强迫特征空间把热点与背景分开。

参考：

- [A step towards understanding why classification helps regression](https://openaccess.thecvf.com/content/ICCV2023/html/Pintea_A_step_towards_understanding_why_classification_helps_regression_ICCV_2023_paper.html)

### 5.3 局部排序损失

在同一病例内采样点对：

- 高点来自 top10/top5；
- 低点来自 q50–q90 或热点边缘；
- 优先在同一局部表面 patch 内采样；
- 避免全点两两比较。

可使用 RankNet 或 margin ranking：

\[
L_{\text{rank}}
=
\log\left(1+\exp\left[-s_{ij}(\hat y_i-\hat y_j)\right]\right)
\]

其中 \(s_{ij}=\operatorname{sign}(y_i-y_j)\)。

当前 high-WSS 内 Spearman 接近 0，因此该项比继续提高全场 Spearman 更直接。

参考：

- [RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression](https://proceedings.mlr.press/v162/gong22a.html)

### 5.4 条件尾部校准

不要再次对所有 raw WSS 点统一增加 raw-Huber。更有针对性的方式是只约束真值尾部：

- true-top10 区域平均预测；
- true-top5 区域平均预测；
- p95/p99；
- \(WSS-q90\) 的超额部分。

可定义：

\[
L_{\text{tail-cal}}
=
\left|
\frac{\operatorname{mean}(\hat y_i \mid y_i\ge q90)}
{\operatorname{mean}(y_i \mid y_i\ge q90)+\epsilon}
-1
\right|
\]

并对超额部分使用：

\[
e_i=\max(WSS_i-q90,0)
\]

\[
L_{\text{excess}}
=
\operatorname{Huber}
\left(
\log(1+\hat e_i),
\log(1+e_i)
\right)
\]

这与历史“所有点 raw-Huber”不同，它直接优化被系统低估的尾部。

### 5.5 表面梯度损失

为避免热点边界被平滑，可在 STL 邻接或表面 kNN 图上约束：

\[
L_{\nabla}
=
\operatorname{mean}_{(i,j)\in E}
\left|
(\hat y_i-\hat y_j)-(y_i-y_j)
\right|
\]

边权可在真值梯度较大、top10 边缘或高曲率区域提高。

优先使用：

- STL mesh 邻接；
- 表面测地邻域；
- 有分支约束的局部图。

谨慎使用纯欧氏 kNN，避免两个空间上接近、表面上不连通的血管壁被错误连边。

---

## 6. P3：双流采样和热点 patch

建议每个病例同时保留全局流与尾部流。

### 6.1 全局流

- 5000 个 uniform-random 或 area-random 点；
- 保持现有全场分布学习；
- 基础回归损失按全局流计算；
- 必要时使用 importance correction。

### 6.2 尾部流

- 从 top10/top5 选择多个热点中心；
- 沿表面邻接提取局部 patch；
- 每个 patch 同时包含热点、边界和 hard negative；
- 尾部流主要用于 hotspot、rank、tail calibration 和 gradient loss。

建议第一版每个病例：

- 4–8 个热点/边界中心；
- 每个 patch 128–256 点；
- 高点、边界点和普通点比例约为 `1:1:1`；
- 热点中心做空间非极大值抑制，避免全部落在同一个小区域。

### 6.3 Hard-example mining

可以在基础模型 warmup 后，按训练残差更新困难点集合，但需要：

- 限制困难点最高占比；
- 对邻近困难点做空间去重；
- 不连续追逐单节点异常值；
- 同时保留随机背景；
- 只使用训练病例残差。

否则容易从“高值低估”转成“局部爆峰”。

---

## 7. P4：架构优化

### 7.1 保留的骨干

推荐继续使用：

- PointNeXt-R；
- LocalGeoPE；
- 当前验证有效的 `xyz+geom`；
- SA2 中尺度上下文；
- 轻量 DropPath 候选。

现有证据不支持：

- 去除几何特征；
- 全层 local Transformer；
- SA3 全局 Transformer；
- 单纯继续加宽；
- 单纯继续加深；
- NeighborDrop；
- 用全点/10000点直接替代当前5000点协议；
- 通用 QAD decoder 替换。

### 7.2 Coarse-to-fine hotspot refiner

建议的结构为：

```text
PointNeXt-R + LocalGeoPE
          │
          ├── base WSS head
          │
          ├── hotspot probability head
          │
          └── hotspot candidate generator
                         │
                         ▼
              full-resolution patch refiner
                         │
                         ▼
              tail residual / excess prediction
                         │
                         ▼
          final WSS = base + gated residual
```

精修器可以输入：

- SA1 细尺度特征；
- SA2 中尺度特征；
- 原始 xyz；
- abscissa、local radius、curvature；
- base WSS；
- hotspot probability；
- 局部表面梯度；
- 分支/流量代理特征。

最终残差建议使用门控：

\[
\hat y_i
=
\hat y_i^{base}
+p_i^{hot}\cdot \operatorname{softplus}(\Delta_i)
\]

这样可以限制尾部修正主要作用在高风险候选点，减少全场爆峰。

### 7.3 为什么不只改当前 decoder

现有 PointNet++ 解码使用 3-NN 插值与 MLP：

```text
coarse feature
    → 3-NN interpolation
    → skip concat
    → MLP
```

这会天然平滑高频尖峰。历史 QAD 没有稳定增益，说明通用 decoder residual 不足以解决问题。

新 refiner 与 QAD 的核心差异应是：

- 只针对 hotspot 候选区域；
- 预测尾部 residual/excess；
- 使用 hotspot、rank、tail calibration 和 gradient supervision；
- 在 full-resolution 表面 patch 上工作；
- 同时使用 SA1 和 SA2 特征。

### 7.4 病例尺度与空间型态双头

可以增加：

1. 病例级尺度头：从全局特征预测 `log(p99 WSS)`；
2. 点级 shape 头：预测相对空间型态；
3. 最终由网络预测的尺度与 shape 重建物理 WSS。

重要约束：

> 推理时不能使用真实 CFD p99/WSSmax。

历史 `WSS/WSSmax` 直接目标失败，并不等于 scale-shape 框架无效。历史方案的问题是病例真实尺度在部署时不可用；新方案必须由网络根据几何和可用边界条件预测尺度。

### 7.5 分域容量

若继续混合 AG/AAA/ILO，可在共享 backbone 后尝试：

- cohort-specific 小型 adapter；
- cohort-specific output head；
- FiLM 条件层；
- shared expert + domain expert 的轻量 mixture-of-experts。

这比简单拼接 cohort one-hot 更能缓解 AAA↔ILO 容量竞争。

前提：

- cohort 在部署时已知；
- 每个域有足够病例；
- 必须保留 shared head 对照；
- 必须防止小域专用 head 过拟合。

---

## 8. 数据增强边界

### 8.1 推荐

- 每 epoch 点重采样；
- 合法重网格化、降采样和字段插值；
- 点密度扰动；
- 小幅坐标噪声，幅度不超过配准/网格误差；
- 几何特征轻噪声；
- feature dropout；
- hotspot patch 尺度和中心扰动；
- 表面邻域随机遮挡；
- 训练后期有上限的 hard-example mining。

### 8.2 不推荐

- 任意三维旋转：已有负结果，且破坏 canonical anatomical frame；
- 对几何做缩放后保留原 WSS；
- 弹性形变后保留原 CFD 标签；
- 病例间点级 MixUp/CutMix；
- 合成几何后复制某个真实病例 WSS；
- 为增加高值样本而复制单点极值；
- 用单节点 max 作为主要增强或选模目标。

几何发生变化时，真实 CFD 解也会变化。合成几何只有在重新运行 CFD，或使用经过独立验证并带不确定性过滤的 teacher 重新标注后，才可进入正式训练。

---

## 9. 最小实验矩阵

### 9.1 第一阶段：损失单变量筛选

固定三种子确认过的 S3-GEOPE 骨干，在新 grouped val/OOF 上运行：

| ID | 唯一变化 | 目的 |
|---|---|---|
| A0 | 当前 log-MSE | 科学对照 |
| A1 | Balanced MSE/标签密度平衡 | 检验连续标签不均衡 |
| A2 | A0 + hotspot classification | 检验热点特征分离 |
| A3 | A0 + local ranking | 检验热点内部排序 |
| A4 | A0 + conditional tail calibration | 检验幅值系统低估 |

执行规则：

1. 先在固定开发 fold、seed1234 筛选；
2. 只保留综合表现最好的两个；
3. 补 seeds `7/2025`；
4. 做逐病例配对 bootstrap；
5. 不按单个 pooled 指标选择。

### 9.2 第二阶段：采样与架构

以前一阶段胜者为父模型：

| ID | 唯一变化 | 目的 |
|---|---|---|
| B0 | 第一阶段胜者 | 新父对照 |
| B1 | 双流 hotspot patch 采样 | 提高尾部训练覆盖 |
| B2 | scale-shape 双头 | 分离病例尺度与空间型态 |
| B3 | coarse-to-fine hotspot refiner | 修复 decoder 平滑和局部尖峰 |
| B4 | 真实 RCR/流量条件 | 信息上限 oracle |
| B5 | 可部署流量代理场 | 检验真实部署路线 |

### 9.3 精简后的 Gate

不再要求所有实验同时通过一长串异质指标。每个实验只设置一个作用轴主终点，并检查两个全场护栏：

| 实验类型 | 主终点 |
|---|---|
| RCR Oracle / 幅值目标 | high-WSS physical nRMSE |
| hotspot 分类 | top10 IoU |
| ranking loss | high-WSS Spearman |

共同护栏为 normalized case-balanced \(R^2\) 和 physical case-balanced MAE。top10/p99 ratio 只作诊断，不作为晋级门禁。

Oracle 的必要条件是 **O1a 同时优于 O0 和 O2**；若 O1a 与 O2 同步改善，不能归因于真实 RCR 信息。当前 `138/0/36` 单 seed 结果只用于工程筛选，后续是否补多 seed 或 OOF 由效应量决定。

---

## 10. 不建议当前优先启动的方向

### 10.1 全局 Transformer/大模型

现有 local/global Transformer 结果显示，收益集中在特定 SA2 中尺度，且不同协议不稳定。直接换成更大的 Transformer 或神经算子不能自动补回缺失边界条件，也可能在当前病例规模下加重过拟合。

Transolver 等物理注意力算子可作为后期候选：

- [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/abs/2402.02366)

但其优先级应低于：

1. 无泄漏开发协议；
2. RCR/流量 oracle；
3. tail-aware loss；
4. hotspot refiner；
5. 数据规模与条件输入。

### 10.2 Gaussian NLL 作为主解决方案

异方差 NLL 可以给出不确定性，但也可能通过增大 high-WSS 区域预测方差来降低损失，并不一定提高均值预测。它适合在确定 geometry-only 存在不可辨识性后，用于输出风险区间，不适合作为第一轮峰值修复方案。

### 10.3 极值理论直接拟合

Generalized Pareto 等尾部模型要求足够多且近似独立的超阈值样本。当前高 WSS 点具有强空间相关性，病例数也有限，不建议直接把 EVT 作为第一主线。

---

## 11. 推荐执行顺序

### 阶段 0：不训练或最小训练

1. 保持 mixed `138/0/36` 历史横向口径，并标注 test36 为复用开发集；
2. 面积口径暂缓，固定 `legacy_vertex`；
3. 日常指标收敛为 §3.3 四项；
4. 运行 S3 父模型的 O0/O1a/O2 RCR Oracle；
5. O1b 等出口流量覆盖和 QA 闭环后再决定是否补跑。

### 阶段 1：损失

1. A0–A4 单变量；
2. 单 seed筛选；
3. 最优两个补三 seed；
4. 只选一个作为后续父模型。

### 阶段 2：采样和精修器

1. 双流 hotspot patch；
2. coarse-to-fine residual refiner；
3. scale-shape 双头；
4. 按需要加入 gradient loss；
5. 三 seed和逐病例配对确认。

### 阶段 3：物理条件和部署

1. 若 oracle 成立，设计真实 BC 输入接口；
2. 若真实 BC 部署不可得，训练流量/RCR代理；
3. 比较真实条件、预测条件、几何代理和无条件；
4. 对条件预测误差做端到端敏感性分析。

---

## 12. 最终建议

当前最值得立即推进的单条路线是：

> **S3-GEOPE 骨干 + 病例/尾部分布平衡回归 + top10 hotspot 分类头 + 条件尾部校准；并行执行真实出口 RCR/流量分配 oracle。**

判断成功时必须同时满足：

- 全场 physical R² 和 MAE 不明显退化；
- 高 WSS 幅值恢复提高；
- p99 校准提高；
- hotspot IoU 和质心距离提高；
- high-WSS 内排序提高；
- 不出现局部爆峰；
- AG/AAA/ILO 没有明显容量转移；
- 多 seed 和逐病例配对证据方向一致。

如果真实边界条件 oracle 明显成功，应及时停止纯 geometry-only 架构扫描，把主要精力转向条件输入和低维血流—壁面场级联模型。若 oracle 无效，则优先执行 tail-aware 多任务损失和 full-resolution hotspot refiner。

---
---

# 第二部分：第一性原理重估与对抗性审查（2026-07-26）

> 本部分所有数字均由**本仓库落盘数据**或**已存 run 产物**直接测得，脚本与出处见 §17。
> 未训练任何新模型。凡属推导而非实测的量，均已显式标注。

## 13. 第一性原理重估

### 13.1 物理空间尾部亏欠不是独立故障，而是标准化空间亏欠的确定性放大

当前目标是 \(z=(\log(y+\varepsilon)-\mu)/\sigma\)，指标在 \(\hat y=\exp(\sigma\hat z+\mu)-\varepsilon\) 之后计算。
因为分位数与单调变换可交换，**物理分位比是一个恒等式**：

\[
\frac{\hat y_q}{y_q}=\exp\!\big[\sigma\,(\hat z_q-z_q)\big]
\]

代入本 run 实测值（`wss_stats` 的 \(\sigma=1.3669\)；标准化空间 \(\hat z_{p99}=1.5830,\ z_{p99}=2.2601\)）：

\[
\exp[1.3669\times(1.5830-2.2601)]=\exp(-0.9255)=\mathbf{0.3963}
\]

而 `metrics.json` 记录的 `calibration.p99_pred_true_ratio` = **0.39630136585239173**。**逐位吻合。**

含义（这是本轮最重要的一条）：

- 标准化空间只差 **0.68 个 z 单位**，经 \(\exp\) 放大后就是物理空间 **2.5 倍**的亏欠；
- 所以 §2.2 把"幅值校准失败"和"热点型态失败"并列成两个故障是**误判**。幅值项没有独立的故障机制，它是全场收缩经过 \(\sigma=1.37\) 的指数杠杆后的读数；
- 任何只盯物理幅值比的诊断，都在观察一个被放大 1.37 倍指数的量，其对模型改动的敏感度天然大于其信息量。

### 13.2 网络已经是 log 空间的条件均值——收缩是损失函数的定义，不是缺陷

`metrics.py` 中 `calibration_slope` = \(\mathrm{cov}(y,\hat y)/\mathrm{var}(y)\)。对 MSE 最优的 \(\hat z=\mathbb E[z\mid x]\) 有 \(\mathrm{cov}(z,\hat z)=\mathrm{var}(\hat z)=R^2\mathrm{var}(z)\)，因此 **slope 应当等于 \(R^2\)**。实测（均为标准化空间）：

| 口径 | calibration_slope | \(R^2\) | 差 |
|---|---:|---:|---:|
| pooled | 0.6325 | 0.6181 | **0.014** |
| 逐例平均 | 0.6236 | 0.5525 | 0.071（逐例 \(\|{\rm slope}-R^2\|\) 均值 0.089，相关 0.82，36 例中 18 例差 <0.05） |

pooled 口径上两者几乎相等，逐例口径上方向一致但有离散——**足以支持"网络已经在做 MSE 要求它做的事"这一读法**（离散来自有限样本与过拟合，见 §13.8），但不是严格恒等式。

结论：**收缩不是可以靠"更重视高值"消除的偏置**。在存在不可约条件方差时，条件均值预测器的幅值收缩是数学必然，不是可以靠"更重视高值"消除的偏置。因此：

> §5 中所有"加权/重加权"类方案，其可达上界受制于同一条前沿；能真正越过前沿的只有**增加信息**（特征、监督、样本），不是重新分配已有信息的权重。

### 13.3 单参数方差重标定前沿：零训练即可拿到大部分幅值，且排序指标严格不变

令 \(\hat z_\lambda=c+\lambda(\hat z-c)\)（\(c\) 取预测中位数，\(\lambda>0\)）。两个精确性质：

1. \(R^2_\lambda=R^2\cdot\lambda(2-\lambda)\)，在 \(\lambda=1\) 取最大——**任何幅值恢复都必然付出标准化 \(R^2\)**；
2. \(\hat z_\lambda\) 与 \(\hat y_\lambda\) 都是 \(\hat z\) 的严格单调变换，所以 **top10 IoU / precision / recall / spearman_all / spearman_high_wss 逐点完全不变**。

前沿表：**逐例应用**上式后按 §14/D-01 的两种聚合分别汇总（\(\sigma=1.3669\)，\(c_i\) 取各例预测中位数）。
p99 列由 §13.1 的恒等式给出，已逐例验证 \(\lambda=1\) 与落盘值最大偏差 **3×10⁻⁶**；top10 列含一个实测 1.020 的 Jensen 修正。

| \(\lambda\) | 标准化 \(R^2_{cb}\) | Δ | p99（逐例·物理） | top10（逐例·物理） | p99（pooled·物理） | top10（pooled·物理） | IoU / 各 Spearman |
|---:|---:|---:|---:|---:|---:|---:|:--|
| **1.000（现状）** | 0.6425 | — | **0.520** | **0.428** | **0.396** | **0.371** | 不变 |
| 1.100 | 0.6361 | −0.006 | 0.607 | 0.469 | 0.494 | 0.426 | 不变 |
| 1.150 | 0.6280 | −0.015 | 0.656 | 0.492 | 0.552 | 0.456 | 不变 |
| 1.200 | 0.6168 | −0.026 | 0.710 | 0.516 | 0.616 | 0.489 | 不变 |
| 1.250（≈\(1/\sqrt{R^2}\)，方差匹配） | 0.6023 | −0.040 | 0.769 | 0.541 | 0.688 | 0.524 | 不变 |
| 1.300 | 0.5847 | −0.058 | 0.833 | 0.568 | 0.769 | 0.561 | 不变 |
| 1.400 | 0.5397 | −0.103 | 0.981 | 0.628 | 0.958 | 0.645 | 不变 |

对照 §9.3 的两条幅值门禁（top10 与 p99 各 **+0.08**）：**\(\lambda=1.15\) 一个标量就同时满足两条**（逐例 top10 +0.064、p99 +0.136；pooled top10 +0.085、p99 +0.156），代价是标准化 \(R^2_{cb}\) −0.015。而它**不可能**满足同一门禁里的 IoU +0.02 与 high-WSS Spearman +0.05——这两项在任何 \(\lambda\) 下逐点不变。

**而物理 \(R^2\) 的方向与标准化 \(R^2\) 相反。** 真值 top-10% 那批点占物理 MSE 的份额：

| 口径 | 份额 |
|---|---:|
| pooled（`regional_field.high_wss` n·rmse² / `field` n·rmse²） | **0.854** |
| 逐例平均（`per_case[*].high_wss` vs `per_case[*].overall`） | **0.698**（中位 0.696，p10–p90 0.530–0.870） |

即 **物理 MSE 的 70–85% 来自模型低估 2.3–2.7 倍的那批点**（top10 幅值比逐例 0.428 / pooled 0.371）。因此在中等 \(\lambda\) 下抬升尾部会**降低**物理 MSE、**抬高**物理 \(R^2\)（幅度需实测，见下）。

这直接击穿 §9.3 的门禁设计前提：它用**物理** \(R^2\) 作为幅值干预的护栏，但物理 \(R^2\) 有 70–85% 由该幅值干预所修的那批点决定——**护栏和被测项不是正交的，而是同向的**。幅值与 \(R^2\) 的真实取舍发生在**标准化空间**，不在物理空间。

**行动**：这是全案投入产出比最高的一步，且**不需要训练**——重跑已存 ckpt 的 evaluate，扫 \(\lambda\in[1.0,1.5]\)，出 (λ, 标准化 \(R^2\), **物理 \(R^2\)**, 物理 MAE, top10, p99) 六列前沿。上表已给出除物理 \(R^2\)/MAE 外的全部列（它们由恒等式确定，无需训练）；**物理 \(R^2\) 与 MAE 是唯一必须实测的两列**，因为 \(\exp\) 非线性使其无解析式。
\(\lambda\) 必须在 **train/OOF** 上拟合、在测试集上仅作应用，否则又是一次在 test36 上调参。
此后 §5 的每一个损失臂，都必须以"是否跑赢同等 top10 提升下的 λ 前沿"来判定，而不是"是否提高了幅值"。

### 13.4 剩余方差主要在病例内，不在病例间——这给 §4 的 oracle 计划设了硬上限

全库 192 例的 \(\log\)WSS 方差分解：

| 分量 | 数值 |
|---|---:|
| 病例间 var(病例平均 log WSS) | 0.2421 |
| 病例内平均 var | 1.3854 |
| **病例间占比** | **14.9%** |

而模型当前标准化 \(R^2_{cb}\) 已达 **0.6425**，**远超病例间分量的全部**。

含义：

- **任何纯病例级条件信号（出口 RCR、cohort one-hot、入口面积、病例尺度头）的收益上界约为 15% 的 log 方差**；分支级流量分配能再吃掉一部分病例内份额，但四个出口只切分远端髂支；
- §4（P1 oracle）、§7.4（scale-shape 双头）、§7.5（分域容量）**三条方案共享同一个 ~15% 的上限**，且相互重叠；
- 真正待解的 36% 残差绝大部分是**病例内空间型态**。

### 13.5 标签不是瓶颈：热点是网格已解析的连续大斑块，峰值时相也稳定

全库 192 例，中位数（p10–p90）：

| 实测量 | 数值 | 读法 |
|---|---:|---|
| 8 近邻平滑后 top10 IoU | **0.901** (0.833–0.942) | 热点不是散粒噪声 |
| 8 近邻平滑后 top10 幅值保留 | 0.976 | 峰值不是单点毛刺 |
| 热点内聚度（热点点的 8 近邻仍是热点的比例） | **0.848** (0.775–0.901) | 热点是大片连通区 |
| 峰值时相 vs 峰值±1 步（10 ms）IoU | **0.921** (0.882–0.948) | 选步不是噪声源 |
| 峰值 vs ±1 步 top10 幅值比 | 0.995 | 同上 |
| 峰值 vs ±1 步 high-WSS Spearman | 0.978 | 同上 |
| 峰值 vs ±2 步 IoU | 0.850 | |
| 峰值 vs ±4 步 IoU / 幅值比 | **0.742** / 0.876 | ±4 步才是真正不同的场 |

**§4.1 中"尾部标签是否受局部网格/插值影响""峰值时相与条件是否对应"这两条排查，现在可以直接结案为「否」。**
同时，IoU = 0.194 **不能**用标签噪声解释。

### 13.6 协议天花板（实测，零训练）：分辨率不是主因，但各 cohort 的天花板不同

做法：用**真值**在 5000 支撑点上"完美预测"，再走模型实际的 3-NN 反距离插值到全点云，算指标。这是当前 support/query 协议下任何模型的上界。全库 192 例中位数：

| 上界 | ALL | AG | AAA | ILO |
|---|---:|---:|---:|---:|
| 物理 \(R^2\) | 0.939 | 0.969 | 0.914 | 0.894 |
| top10 幅值比 | 0.963 | 0.976 | 0.955 | 0.946 |
| p99 幅值比 | 0.974 | 0.983 | 0.967 | 0.964 |
| **top10 IoU** | **0.813** | **0.858** | **0.784** | **0.738** |
| high-WSS Spearman | 0.847 | 0.901 | 0.808 | 0.774 |
| **标准化空间 \(R^2\)** | **0.976** | 0.984 | 0.972 | 0.946 |
| 支撑点间距 (mm) | 1.70 | 1.74 | 1.70 | 1.61 |
| 网格间距 (mm) | 0.75 | 1.11 | 0.45 | 0.44 |
| 5000 点覆盖率 | 23.2% | 39.5% | 8.7% | 7.6% |
| 病例数 | 192 | 76 | 65 | 51 |

（ALL 的 IoU 天花板 p10–p90 = 0.712–0.883；high-WSS Spearman p10–p90 = 0.725–0.922。）

三条结论：

1. **分辨率不是当前主要瓶颈。** IoU 天花板 0.813 vs 当前 0.194，有 **0.62 的空间不需要动分辨率**就能拿。§7.2 的 full-resolution refiner 所攻击的机制，其全部价值上限是 top10 幅值 ≤0.037、IoU ≤0.187——应当**降级到后置**，不是与损失并行。
2. **各 cohort 天花板不同（IoU 0.858/0.784/0.738；high-WSS Spearman 0.901/0.808/0.774）**，因为 AG 网格粗（1.11 mm）而 AAA/ILO 细（0.44 mm），真值场的细节量本身就不同。**跨 cohort 的 IoU / Spearman 直接比较不在同一标尺上**，§7.5 诊断"AAA↔ILO 容量竞争"之前必须先扣掉这一项。
3. 支撑点的**物理**间距三个 cohort 反而接近（1.61–1.74 mm）——这是网格密度差异（5.2×）与固定 5000 点两个效应恰好抵消的巧合，不是设计。**任何改动采样协议（面积采样、改点数、改 cohort 配比）都会破坏这个巧合**，必须同时重算本表。

### 13.7 一个解析特征解释了 38.5% 的病例内 log 方差，而它没有进网络

逐例最小二乘 \(\log(\mathrm{WSS})\sim a+b\log r_{\text{local}}\)（全库 192 例）：

| 量 | 中位数 | p10 | p90 | AG / AAA / ILO 中位 |
|---|---:|---:|---:|---|
| 指数 \(b\) | **−1.328** | −1.754 | −0.860 | −1.253 / −1.413 / −1.331 |
| log 空间 \(R^2\) | **0.385** | 0.150 | 0.635 | 0.367 / 0.432 / 0.344 |

`local_radius` **是物理量**（mm，全库中位 7.9 mm，`data_wss_min/**/bundle.npz:wall_local_radius`），但在 `dataset.py:build_features` 中只做**线性 z-score** 进网络，\(\log r\) 从未构造。Poiseuille 尺度 \(\tau\propto \mu Q/r^3\) 在 log 空间是**线性**的；当前网络必须先在内部逼近 \(\log\) 再逼近幂律，而尾部恰好是 \(r\) 最小、非线性最强的地方。

同时注意实测指数是 **−1.33 而非 −3**：入口共享波形下这是脉动/入口段/分流共同作用的结果，因此**不应硬编码 −3**，而应把 \(\log r\) 作为输入列、让网络学一个自由指数（或用可学习指数的解析锚 + 残差参数化）。

### 13.8 当前处于明确过拟合，而选模在过拟合最深处

| 量 | 值 | 出处 |
|---|---:|---|
| 末期 train MSE（标准化，5000 采样点） | 0.1784 | `history.jsonl` e399 |
| 测试集标准化 \(R^2_{cb}\) | 0.6425 | `metrics.json` |
| 由测试指标反解的 case-balanced var(z) | 0.8615 | \(0.3080/0.3575\) |
| **等价 train \(R^2\)** | **≈0.793** | \(1-0.1784/0.8615\) |
| **泛化差距** | **≈0.151** | |
| 其中支撑点→全点云插值贡献 | ≈0.024 | §13.6 标准化天花板 0.976 |
| **真实泛化差距** | **≈0.13** | |

且 train loss 是在 BN batch 统计 + DropPath 开启、每 epoch 重采样下测得的，**系统性高估训练误差**，所以 0.13 是**下界**。
选模配置：`selection_rule="train_loss"`、`val_cases=0`、best epoch **378/400**——即在过拟合最深处选点。

含义：§9.1 的 A1–A4 全部是**增加目标项**，§9.2 的 B1–B3 全部是**增加容量**。在 0.13+ 的过拟合区里，加容量的期望符号是负的。§3.1 被正确识别为 P0，但方案随后没有据此改变 §9 的臂序。

---

## 14. 对抗性审查（逐条判定）

判定口径：**证否** = 有实测证据表明该条不成立；**须改写** = 方向对但表述/阈值错；**已被历史结果覆盖** = 该臂跑过且为空。

### D-01 §2.1 一张表里混了三种聚合口径；且 `per_case_metrics.csv` 的无前缀列静默地是**标准化空间** —— **须改写 + 代码隐患**

**(a) 聚合口径混用。** `metrics.py` 中 `calibration_metrics` / `regional_metrics` 在**全部 36 例 1,405,566 点池化后**计算，`aggregate_hotspot_metrics` 是逐例平均，`casebalanced_field_metrics` 是病例等权。§2.1 把三者并列而未标注：

| §2.1 行 | 实际口径 | pooled·物理 | **逐例·物理** | 差 |
|---|---|---:|---:|---:|
| top10 幅值恢复比例 | **pooled** | 0.3708 | **0.4278** | +0.057 |
| p99 幅值比例 | **pooled** | 0.3963 | **0.5205** | **+0.124** |
| （未列）max 幅值比 | **pooled** | 0.1250 | **0.2652** | **+0.140** |
| （未列）动态范围比 | **pooled** | 0.1247 | **0.2608** | **+0.136** |
| （未列）calibration slope | **pooled** | 0.2497 | **0.3398** | +0.090 |
| physical high-WSS \(R^2\) | **pooled**(`regional_field`) | −0.367 | 逐例·物理未落盘 | — |
| top10 IoU / high-WSS Spearman / 全场 Spearman | casemean ✓ | — | 0.194 / 0.042 / 0.742 | 空间无关 |
| normalized / physical \(R^2_{cb}\) | case-balanced ✓ | — | 0.6425 / 0.3171 | — |

放大机制：pooled top-10% 掩码选的是**全局**最高点，集中在少数高 WSS 病例；且 pooled 是**逐顶点**加权，而 ILO 中位 65,815 点 vs AG 12,644 点——**一个 ILO 病例约等于 5.2 个 AG 病例**。对 `max_ratio` 尤其致命：pooled 的 0.125 是全部 140 万点里的单个最大值之比，由一例决定，作为汇报量无信息。

**后果**：p99 与 max 两行相对高估亏欠 30–50%。但 top10 行只差 0.057——所以**"只恢复约 40%"这个核心前提基本成立**，需要修正的是 p99/max/动态范围三行，以及"表内口径不一致"本身。

**(b) 代码隐患（更危险）。** `evaluate.py:write_reports` 开头是
`norm_res = res.get("normalized", res)`，随后遍历 `norm_res["per_case"]`。
因此 `per_case_metrics.csv` 中所有**无前缀**列——`cal_*`、`dist_*`、`overall_*`、`high_wss_*`、`bifurcation_*`、`stenosis_*`、`legacy_vertex_*`——**全部是标准化空间的值**，却与 `metrics.json` 里同名的**物理**区块并存。实测同一例：

| 列 | 值 | 空间 |
|---|---:|---|
| `cal_top10_pred_true_ratio`（逐例平均） | 0.5458 | **标准化** |
| `physical_calibration_top10_pred_true_ratio`（逐例平均） | 0.4278 | 物理 |
| `high_wss_r2`（逐例平均） | **−13.82** | **标准化** |
| `regional_field.high_wss.r2`（metrics.json） | −0.367 | 物理·pooled |

`hot_*` 恰好安全（IoU/Spearman 是秩量，对 \(\exp\) 不变，metrics.json 里两个空间的值逐位相同）。但任何扫 CSV 的人读 `cal_top10_pred_true_ratio` 会拿到一个比物理值高 0.12 的数。
（已核对：`tools/analyze_regp10_transformer_results.py` 走的是 `metrics.json` 而非该 CSV，故已发布的 9 臂汇总表口径正确，未受污染。）

**行动**：§2.1 每行标注「空间 + 聚合」；主口径统一为**物理·逐例（或 case-balanced）**，pooled 仅作副指标且 max_ratio 不再单列；`write_reports` 给无前缀列补空间标注或直接弃用（它们只为历史列兼容而存在）。§9.3 的所有阈值按选定口径重新推导。

### D-02 §2.1 的锚点是 9 臂 test36 扫描的逐指标最大值（winner's curse）—— **须改写**

`regp10_transformer_matrix_20260726_results_summary.csv`，同数据同种子的 9 个臂：

| 指标 | L-SA2（§2.1 引用） | 9 臂范围 | 9 臂 sd | L-SA2 排名 |
|---|---:|---|---:|---|
| physical \(R^2_{cb}\) | 0.3171 | 0.2798–0.3171 | 0.0109 | **1/9** |
| top10 幅值比 | 0.3708 | 0.3392–0.3708 | 0.0093 | **1/9** |
| p99 幅值比 | 0.3963 | 0.3348–0.3963 | 0.0205 | **1/9** |
| high-WSS Spearman | 0.0416 | 0.0028–0.0416 | 0.0115 | **1/9** |
| physical MAE | 2.5527 | 2.5527–2.6057 | 0.0153 | **1/9（最小）** |
| top10 IoU | 0.1940 | 0.1772–0.1954 | 0.0052 | 2/9 |
| normalized \(R^2_{cb}\) | 0.6425 | 0.6239–0.6441 | 0.0060 | 2/9 |

L-SA2 在 5 个指标上同时为 9 臂最优、另 2 个次优。9 次抽样取最大的期望偏置约 \(1.49\sigma\)：**\(R^2_{cb}\) 约高估 +0.016，high-WSS Spearman 约高估 +0.013**（臂均值 0.0288）。且这 9 个臂是在**已复用多轮的 test36** 上比较的——§3.1 已指出复用问题，此处给出了量级。

**行动**：锚点改用臂均值 ± sd 报告；或在新协议上把 L-SA2 重训 3 seed 后再作父模型。否则后续所有"改进量"都是相对一个偏高 0.016 的基线测量的。

### D-03 §9.3 的 Gate 同时"过松于噪声"与"严于历史最佳"，且历史上最大的一次真实改进自己也过不了 —— **证否**

已测噪声（`d2_k64_ilo_structure_three_seed_confirmation_summary.csv`，三种子逐例配对差的 sd，ckpt_best）与 9 臂谱宽：

| §9.3 门禁 | 阈值 | 三种子配对 sd | 9 臂全幅 | 判定 |
|---|---:|---:|---:|---|
| physical \(R^2_{cb}\) 不降 >0.01 | 0.01 | **0.0143–0.0227** | 0.0373 | 阈值 **< 1 sd**：真正中性的模型单种子 24–33% 被误杀，取三种子均值仍有 11–22% |
| physical MAE 不恶化 >0.03 Pa | 0.03 | **0.0211–0.0465** | 0.0530 | 阈值 ≈ 1 sd，同上量级 |
| top10 幅值比 ≥ +0.08 | 0.08 | — | **0.0316** | 阈值是 9 臂**全部谱宽的 2.5 倍**（架构改动够不到；但一个标量 λ 就够，见下） |
| p99 幅值比 ≥ +0.08 | 0.08 | 0.0105–0.0207 | **0.0615** | 同上 |
| top10 IoU ≥ +0.02 | 0.02 | 0.0069–0.0119 | 0.0181 | ≈ 历史最佳单次改动的实测值 (+0.0219±0.0119) |
| high-WSS Spearman ≥ +0.05 | 0.05 | — | **0.0388** | 阈值 > 该指标**曾出现过的全部取值范围**（0.0028→0.0416）；要求当前值翻一倍以上 |
| 三域任一 \(R^2\) 不降 >0.02 | 0.02 | AG 0.0083 / AAA 0.0099 | — | 三域取"任一"，误杀概率约 ×3 |

**决定性反例**：项目历史上最大的一次真实骨干改进 S3−M1（三种子配对）为
`R²_cb +0.0224 / top10_IoU +0.0219 / p99 +0.0138 / spearman_all +0.0592 / MAE −0.128`——
**在 top10 幅值、p99 幅值、high-WSS Spearman 三条上全部不达标，会被本门禁拒绝。**

**两条幅值门禁的真正问题不是"太高"，而是"选错了工具轴"**：架构改动的全部谱宽只有 0.032/0.062，够不到 +0.08；但 §13.3 的一个后验标量 \(\lambda=1.15\) **零训练**就给出 pooled top10 +0.085、p99 +0.156。也就是说这两条门禁**只筛得出重标定，筛不出任何表示层面的进步**——它们奖励的正是 §2.2 声明要拒绝的那类操作。

再叠加 §13.3 的正交性：任何近似单调重标定的干预**在数学上不可能**移动 IoU 与 high-WSS Spearman，所以 A4（尾部校准臂）**必然**在门禁的定位轴上得 0。综合起来，8 条合取门禁**对任一单臂都不可满足**：能过幅值轴的过不了定位轴，能过定位轴的过不了幅值轴。

**行动**：改为「1 个预注册主终点 + 若干护栏」结构；护栏容差按已测配对 sd 定（如 −2 sd），不是拍脑袋的 0.01/0.03；主终点按臂的作用轴分配（幅值臂考幅值，排序臂考排序），并用非劣性检验而非阈值合取。

### D-04 §5.4 的尾部校准损失与 §2.2 自相矛盾，且与被优化指标同构 —— **须改写**

三个问题：

1. **与 §2.2 矛盾**：\(L_{\text{tail-cal}}=\big|\mathrm{mean}(\hat y\mid y\ge q90)/\mathrm{mean}(y\mid y\ge q90)-1\big|\) 对掩码内每一点的梯度**同号同量**，只推高不压低。推理时模型看不到掩码，它能实际实现的下降方向就是"把与尾部重叠的那片特征空间整体抬高"——这正是 §2.2 明确否决的"统一乘一个系数"。
2. **教到考题上**：该损失与 `metrics.py:calibration_metrics` 的 `top10_pred_true_ratio` 是**同一个泛函**。优化它必然改善该指标，但不构成物理改进的证据。
3. **正交性**：它（近似）是单调重标定，因此对 `top10_iou`、`precision`、`recall`、`spearman_all`、`spearman_high_wss` **严格无效**（§13.3）。

**行动**：保留"尾部超额 \(L_{\text{excess}}\)"（它有空间选择性），删除 mean-ratio 项；把 ratio 类指标降为诊断，**门禁只用不可被单调变换伪造的两类量**：条件误差（high-WSS MAE / nRMSE）与定位（IoU / high-WSS Spearman / 质心距）。任何幅值臂必须先跑赢 §13.3 的 λ 前沿才算有效。

### D-05 §3.3 要求同时报 IoU / precision / recall，但当前实现下三者互为函数 —— **证否**

`metrics.py:hotspot_localization_metrics` 用 `thr_p = np.percentile(yp, 90)`，即预测掩码的**基数恒等于**真值掩码。于是
\(\text{precision}\equiv\text{recall}\equiv 2\,\mathrm{IoU}/(1+\mathrm{IoU})\)。
`metrics.json` 实测：precision = recall = **0.31803376878722456**（完全相等）。三个数只有一个自由度。

**行动**：要么只报 IoU；要么引入**固定阈值**变体（如 pred ≥ 真值 q90 的物理值），使 precision/recall 解耦并对幅值敏感——这也顺带给了一个既测定位又测幅值的复合量。

### D-06 §4 的 O1 把"出口 RCR"与"流量占比"混为一谈，后者是求解器输出 = 标签泄漏 —— **证否**

落盘核查：

- `data_new/*/*/Global_conditions/` **只有** `p-in/p-out{le,li,re,ri}-rfile.out` 与 `vf-*-rfile.out`，是 Fluent report-definition 监视器，**即解的输出**；
- `*.cas.gz` 中四个出口为 `pressure-outlet`（`out-li/out-ri/out-le/out-re`），RCR 参数内嵌于 case 文件（"rcr" 命中 12 处）——**RCR 参数是合法的逐例输入**；
- 入口侧 347/347 共享同一硬编码 Fourier 波形，\(Q_{peak}\) 92% 落在中位 ±1%（2026-07-26 审计）。

即：**RCR 参数（输入，合法但不可部署）** 与 **出口流量占比（输出，与 WSS 同源）** 是两种完全不同的东西，§4 的 O1 把它们并列写成"真实出口 RCR/流量占比"。若把流量占比喂进去，O1 **必然**显著改善高 WSS——因为它是产生 WSS 的同一次求解的函数。届时 §4.1 的判据会**假阳性触发**，把整个项目导向"转边界条件输入"。

**行动**：拆成 **O1a = 仅 RCR 参数**（合法 oracle，明确标注不可部署）与 **O1b = 出口流量占比**（**显式标注为泄漏探针**，只用于测信息上限，永不作为部署路线）。§4.1 的判据只允许由 O1a 触发。

### D-07 §4 整个 oracle 计划受 14.9% 的方差硬上限约束，且与 §7.4、§7.5 重叠 —— **须降级**

见 §13.4（全库 192 例）：病例间只占 log WSS 方差的 **14.9%**，而模型已解释 64.25%。纯病例级条件（RCR、cohort、入口面积、病例尺度头）的收益上界就是这 ~15%（分支级分流能再吃一点病例内份额，但四出口只切远端髂支）。

同时，第五轮 F0 已就"可部署 BC 无杠杆"结案（入口共享模板；per-case RCR 是 oracle 非部署），本方案 §4 相当于重开该结论。

**行动**：P1 从"并行主线"降为**预注册上限 ~0.16 log 方差的 1 日诊断**；并明确写入：**空结果是预期结果**，不构成"转向条件输入"的授权。§4.1 中"如果 O1 明显提高"分支下的六条可部署代理（出口面积、Murray \(r^3\)、截面积比…）**全部是几何量**，本来就该走 §15 的输入特征臂，不需要 oracle 授权。

### D-08 §7.2/§7.3 的 full-resolution refiner 所攻击机制的上限已实测，应后置 —— **须降级**

见 §13.6：协议天花板 IoU 0.813 / top10 幅值比 0.963。当前 0.194 / 0.371（逐例·物理 0.428）。**在完全不动分辨率的前提下仍有 0.62 IoU 的空间**。且 §7.3 声称的差异化（"只针对候选区、预测尾部 residual、full-resolution patch"）解决的只是剩下那 0.187 的部分。

**行动**：B3 从第二阶段并列项移到 P3 后置；先做 §15 的 N-06（训练/评估解码口径对齐），它用**零代码**吃掉同一机制中已量化的 0.024 标准化 \(R^2\)。

### D-09 §5.1「每个病例的总损失权重相同」在当前采样器下已恒成立；真正未做的 cohort 平衡已被证否 —— **删除**

`dataset.py` 每例恰好采 `support_n_points=5000`，`objectives.compute_loss` 在拼接 batch 上取（加权）均值，因此每例贡献恒为 \(1/\text{batch\_cases}\)。§5.1 第 2 条是 **no-op**。

而"必要的 cohort 平衡"已在 S3 根因矩阵跑过：cohortbal 三种子 AAA **+0.0255/+0.0362/+0.0249**，但 ILO 同步回落、AG 持平，净 **−0.0018**。机制是 AAA↔ILO 容量再分配，不是净增益。

**行动**：删掉 no-op 条款；不要在没有新机制的情况下把 cohort 平衡重列为新臂。若要重开，必须先按 §13.6 扣除各 cohort 的协议天花板差异（IoU 0.858/0.784/0.738、high-WSS Spearman 0.901/0.808/0.774），否则测的是分辨率不是域。

### D-10 §5.5 的曲率加权在当前数据上不可直接实现 —— **实现陷阱**

`weight_quantiles.json`：`curv_q02` = **0.00108**，`curv_q98` = **21495.6**——原始 \(|\text{curvature}|\) 跨 7 个数量级。`objectives.fixed_quantile_scale` 是线性映射，套上去会把 >99% 的点压到 ~0。输入路径已用 `signed_log1p`（`dataset.py:_transform_feature_values`），但**权重分位是在原始值上算的**（`compute_train_weight_quantiles` 直接用 `np.abs(c["curvature"])`）。

**行动**：§5.5 的"高曲率区提高边权"必须在 `signed_log1p` 后取分位并裁剪；同时修 `compute_train_weight_quantiles` 的口径（现有 `loss_geom_weight` 路径同样受影响，只是当前配置未启用）。

### D-11 §3.2 的面积口径条件"严格通过后使用"当前不可达，且 ILO 从未被审计 —— **须改写**

`ag_aaa_v4_surface_area_mapping_133.json`：`status=failed`，**127/133 通过**，6 例失败（`AG/fast/LI_ZHEN_SHAN`、`AAA/ruputer/{XIE_JIN_QUAN,ZHOU_KE_XUN,SHI_YUN_XI}`、`AAA/unruputer/{GUO_BAO_CHUN,LIU_WEN_QI}`）。该审计生成于 2026-07-16，**早于 2026-07-17 的 ILO 提升**——**ILO 的 51 个单元从未做过面积映射审计**。

**行动**：§3.2 二选一并写死：(a) 排除这 6 例、在 127 例上启用 `surface_metric_mode="both_strict"`；或 (b) 先把审计扩到全部 192 单元（含 ILO）。当前"等全部严格通过"是一个不会到来的条件，会无限期挂起面积口径。

### D-12 §10.2 否决 Gaussian NLL 的理由与本任务的数学结构相反 —— **证否**

§10.2 称 NLL"可能通过增大 high-WSS 区域预测方差来降低损失，并不一定提高均值预测"。但本任务的目标在 **log 空间**，物理条件均值是

\[
\mathbb E[y\mid x]=\exp\!\big(\hat\mu+\tfrac12\hat\sigma^2\big)
\]

**预测方差正是去除指数化偏置所需的那个量**。所以在这里 NLL 不是"附加不确定性"，它是**异方差版的 §13.3 重标定**——比单一 \(\lambda\) 更强，因为修正量随位置变化。

**行动**：把 NLL 从"§10.2 不建议"移到 §15 的**校准臂**，按物理幅值 + 物理 MAE 判定（注意 `config.py` 已支持 `loss="gaussian_nll"` + `out_dim=2`，`objectives.compute_loss` 已实现，无需新代码）。

### D-13 §12「最终建议」的三件事都受上面的判定影响 —— **须改写**

原建议 =「病例/尾部分布平衡回归 + top10 hotspot 分类头 + 条件尾部校准；并行执行 RCR/流量 oracle」。逐项：

- 病例平衡 = no-op（D-09）；分布平衡的可达上界受 §13.2 前沿约束；
- 条件尾部校准 = 与 §2.2 矛盾且与被测指标同构（D-04），且被零训练的 λ 前沿支配（§13.3）；
- 并行 oracle = 上限 ~15%（D-07），且 O1 定义含泄漏（D-06）；
- hotspot 分类头是四项中**唯一**不受上述任一条否定的（它作用于病例内空间型态 = 85% 的方差所在），应保留并提级。

**行动**：§12 按 §16 重写。

---

## 15. 原方案未覆盖的优化空间（按投入产出排序）

### N-01 λ 方差重标定（零训练，最高 ROI）

见 §13.3。重跑已存 ckpt 的 evaluate、扫 \(\lambda\)，出前沿表。**它同时是所有后续损失臂的对照基线**：一个损失臂只有在"同等 top10 提升下标准化 \(R^2\) 损失更小"或"移动了 λ 无法移动的 IoU/Spearman"时才算真的有效。

### N-02 `log(local_radius)` 输入列 + 解析锚残差参数化

见 §13.7：单个 \(\log r\) 解释 **38.5%** 的病例内 log 方差（全库 192 例中位），实测指数 −1.328。做法：
(a) 在 `config.FEATURE_KEYS` 增加 `log_local_radius`（一列，`dataset.build_features` 一个分支）；
(b) 进一步，把输出写成 \(\hat z = \hat z_{\text{anchor}}(\log r) + \Delta\)，其中 anchor 是逐例拟合或可学习指数的一维项，让网络只学 \(O(1)\) 的残差。尾部正是 \(r\) 最小、非线性最强处，这直接减轻 §13.2 的收缩。
**不要硬编码 −3**（全库实测中位 −1.328，p10–p90 为 −1.754…−0.860）。

### N-03 多时相监督（81 个心动时相已在 bundle 内，零新数据）

每个 `bundle.npz` 的 `wall_wss` 形状是 **(81, N)**（实测），而 `config.py` 只支持 `timesteps="peak"`，`dataset.load_case` 取 `[si]` 后丢弃其余 80 相。

- 相邻相高度冗余（峰值±1 步 IoU 0.921），但 **±4 步 IoU 0.743 / 幅值比 0.878** 是**同一几何、同一 BC 下真正不同的场**；有效独立标签约 15–20 个/例；
- 在 138 例、泛化差距 ≥0.13 的区间里，这是**唯一零成本的样本量杠杆**，且标签是真 CFD，不像 §8.2 否决的合成几何；
- **更重要**：它免费给出 §7.4 想要的 scale–shape 分解的**真实监督**——空间型态跨相基本不变而幅值随相位缩放，正是该分解本身。§7.4 要凭空发明一个尺度头，多时相直接提供标注。

### N-04 矢量 WSS 辅助头（`wall_wss_vec` (81,N,3) 在库、`training_wss_min` 零引用）

两个理由，都是第一性原理：

1. **符号正确的 Jensen**：回归三分量再取模，\(\|\mathbb E[\hat\tau]\|\le\mathbb E[\|\hat\tau\|]\)，方向与"系统性低估"相反；直接回归 \(|\tau|\) 则把一个折叠（fold）后的量当成光滑目标学。
2. **热点边界信息**：分离/再附着线（\(\tau\) 方向翻转处）正是 §5.5 想用梯度损失合成的"热点边缘"，而矢量场里它是**直接可监督**的。

### N-05 壁面压力辅助任务（`wall_pressure` (81,N) 在库未用）

历史证据：wall gauge pressure 明显比 WSS 更容易由几何预测（§2.4）。在 0.13+ 过拟合区里，共享骨干的辅助头是**免费的正则化**；且壁面切向压力梯度与 WSS 在边界层动量方程中直接耦合，不是无关任务。`dataset.load_case` 已支持 `target="pressure"`，只需多头而非新数据。

### N-06 训练/评估解码口径对齐（**纯配置，零代码**）

`baseline_models.PointNetPlusPlusRegressor.decode_query` 只有在 query/support 确实共用同一 tensor storage 时才短路。当前 `dataset.collate` 会分别拼接 support 和 query，因此即使 `query_mode="same"`，常规训练批次也会进入 `knn_interpolate`；SAME 的真实含义是 query 与 support 使用相同坐标/索引，而不是“训练期从不插值”。

`query_mode="independent"` + `query_sampling="random"` 已在 `config.validate_features` 与 `dataset.WSSMinDataset.__getitem__` 中实现并校验。它使训练 query 成为独立抽样的 off-support 坐标，直接检验模型是否能从 support 特征泛化到未作为支撑点出现的位置；这是采样支持域的变化，不应再写成“第一次启用插值”。

### N-07 显式选择"目标泛函"，而不是往 MSE 上加尾部项

§13.2 说明收缩是 MSE 的定义。与其加惩罚项对抗它，不如换泛函：

- **分位损失（pinball, q=0.9/0.99）**：按定义给出校准正确的尾部分位，不需要任何 ratio 型辅助损失；
- **物理空间 Huber**（注意：不是 §5.4 否决的"全点 raw-Huber"，而是**取代** log-MSE 作为主损失）：直接以 \(\mathbb E[y\mid x]\) 为目标；
- **NLL + 解析回变换**（D-12）。

三者都是单臂、单行改动，且都**位于 λ 前沿之外**（不是沿前沿滑动），因此是真正的对照。这应当**取代** §5.4。

### N-08 面积口径 + cohort 分辨率对齐（在诊断"域竞争"之前）

按 D-11 先解锁面积口径；同时在每个分域指标旁**并列印出该域的协议天花板**（§13.6 的 IoU 0.858/0.784/0.738）。§7.5 的 cohort adapter / FiLM / MoE 只有在扣除天花板差异后仍见容量竞争时才立项。

### N-09 winner's-curse 防护（流程）

任何在 test36 上从 \(k\) 个臂里选出的锚点，报告时必须给出臂均值 ± sd（或先在新协议重训 3 seed）。当前 §2.1 的偏置量级已量化为 \(R^2_{cb}\) 约 +0.016（D-02）。

---

## 16. 修订后的优先级与最小动作集（取代 §11、§12）

### P0 — 当前执行口径（2026-07-28 已定）

| # | 动作 | 产物 | 依据 |
|---|---|---|---|
| 1 | 当前筛选保留 mixed `138/0/36`，test36 标注为复用开发集 | 冻结 split + 风险说明 | 当前工程决议 |
| 2 | 面积口径暂缓，固定 `legacy_vertex` | Oracle 与后续配置统一口径 | 当前数据合规性 |
| 3 | 日常指标缩减为 §3.3 四项 | 精简后的 §9.3 Gate | D-03, D-05 |
| 4 | 提取 174 例四出口 \(R_1/R_2/C\)，配置 O0/O1a/O2 | RCR sidecar + 三臂配置 | D-06, D-07 |
| 5 | grouped OOF、五折与面积扩审后置 | 仅在方案基本确定后启动 | 算力与当前横向可比性 |

### P1 — 训练，**严格顺序执行**（每步以 λ 前沿为对照）

1. **RCR Oracle（已完成、当前不扩展）**：O1a 同时优于 O0/O2，但本阶段不补 RCR 多 seed、不做 RCR 架构；只保留信息上限证据。
2. **O0 热点/高值首轮（已完成，2/2 No-Go）**：H1 的 `ΔIoU=+0.0170` 未到 `+0.020`，H2 的 `Δhigh-WSS nRMSE=-0.00072` 未到 `-0.002`；保护线均通过，但按 §4.3 不组合。
3. **REG-P10-LSA2 目标/IND 矩阵（已完成）**：IND 主效应和 H1 No-Go；H2 在 SAME/IND 内均过门，但 IND 不晋级。
4. **H2 锚定 + 输入信息臂（已完成，处理臂 Go）**：`log(local_radius)` 相对同 seed H2 control 的 `ΔR²_cb=+0.0436`、`Δhigh-WSS nRMSE=-0.00295`，保护线全过；晋级为新单 seed 开发锚点。
5. **输入信息臂后续**：以新 log-radius 锚点为并发基准，优先验证解析锚残差参数化（N-02）；不得把历史 H2 作为 Gate 对照。
6. **hotspot/ranking 后续**：H1 已两种 query 协议 No-Go，不做 H1+H2 组合；病例层面热点定位仍不稳定，ranking 后置于解析锚单变量臂。
7. **监督信息臂**：多时相（N-03）→ 矢量 WSS 辅助头（N-04）→ 壁面压力辅助头（N-05）。
8. full hotspot refiner、表面梯度损失和多项联合目标继续后置。

### P2 — 诊断（1 天，非主线，预注册上限 ~0.16 log 方差）

- **O1a**：仅 RCR 参数（合法输入，不可部署）；
- **O2**：分区/分层内打乱 RCR 的负对照；
- **O1b**：出口流量占比仅作泄漏探针；当前直接可读覆盖仅 58/174，暂不提交；
- 预先写入：**空结果是预期结果**，不构成转向条件输入的授权（D-06、D-07）。

### P3 — 明确推迟

full-resolution hotspot refiner（§7.2，D-08）、cohort adapter / FiLM / MoE（§7.5，D-09 + N-08）、Transolver 等算子（§10.1）。

### 修订后的一句话结论（取代 §12）

> **当前保持 mixed `138/0/36` 与 `legacy_vertex` 的历史横向口径；RCR 信息上限证据保留但不继续扩展。IND 和 H1 已关闭；REG-P10-LSA2 + SAME + H2 + train-only `log(local_radius)` 已过并发 Gate并成为新单 seed 开发锚点。下一步只在该锚点上验证解析半径锚残差，不组合、不补面积或多 seed。**
> 因为：物理尾部亏欠是标准化亏欠经 \(\sigma=1.37\) 指数放大的确定性函数，逐位可验（§13.1）；网络已是 log 空间条件均值，收缩是 MSE 的定义而非缺陷（§13.2）；log 方差的 85% 在病例内空间型态、只有 15% 在病例间（§13.4）；协议天花板 IoU 0.813 说明分辨率不是瓶颈（§13.6）；而当前泛化差距 ≥0.13 且在最大过拟合点选模（§13.8）。

---

## 17. 复核方法与数据出处

所有数字可复现。§13.4–§13.7 由一个只读诊断脚本产出（不训练、不改数据）：

```bash
/usr/bin/python3 training_wss_min/tools/diagnose_high_wss_ceilings.py \
    --n-per-cohort 0 \
    --out training_wss_min/preflight/high_wss_ceilings_20260726.json
```

该脚本逐位复现 `baseline_models.py` 解码器所用的 `knn_interpolate`（权重 \(1/d^2\)，k 近邻归一化），
并使用本 run 实际的 `q2v_pool2025_control106_global_stats.json` 计算标准化空间天花板。
产物：`training_wss_min/preflight/high_wss_ceilings_20260726.json`（逐例明细 + 方差分解）。

落盘出处：

| 结论 | 出处 |
|---|---|
| 空间/聚合四象限表；收缩恒等式逐例验证（最大偏差 3e-6）；尾部占物理 MSE 70–85% | `training_wss_min/runs/pointnetpp_regp10_transformer/outputs/localtf_sa2_s1234/eval/ckpt_best/{metrics.json,per_case_metrics.csv}` |
| CSV 无前缀列是标准化空间 | `training_wss_min/evaluate.py:write_reports` 的 `norm_res = res.get("normalized", res)` |
| 9 臂谱宽与 winner's curse | `training_wss_min/preflight/regp10_transformer_matrix_20260726_results_summary.csv` |
| 三种子配对 sd；S3−M1 历史最佳效应量 | `training_wss_min/preflight/d2_k64_ilo_structure_three_seed_confirmation_summary.csv` |
| 过拟合区间（train MSE 0.1784 / best epoch 378） | 同 run 的 `history.jsonl`、`train.log` |
| 曲率 7 个数量级 | 同 run 的 `weight_quantiles.json` |
| 面积映射 127/133、ILO 未审计 | `training_wss_min/preflight/{ag_aaa_v4_surface_area_mapping_133.json,ag_aaa_v4_area_phase_backlog.json}` |
| 81 时相 / 矢量 WSS / 壁面压力在库未用 | `data_wss_min/**/bundle.npz` 的 `wall_wss(81,N)`、`wall_wss_vec(81,N,3)`、`wall_pressure(81,N)`；`training_wss_min/config.py` 仅支持 `timesteps="peak"` |
| 出口 RCR 是输入 / 流量占比是输出 | `data_new/*/*/Global_conditions/`（仅 `p-*`、`vf-*` 监视器）、`*.cas.gz` 中四个 `pressure-outlet` + RCR 参数 |
| 训练期不执行插值 | `training_wss_min/baseline_models.py:decode_query` 的 `data_ptr()` 短路 vs `evaluate.py:predict_case_norm` |
| 网格密度 AG 12.6k / AAA 57.5k / ILO 65.8k（中位） | 全部 192 个 `bundle.npz` 遍历 |

**样本量声明**：§13.4–§13.7 基于 **`data_wss_min/` 全部 192 个单元**（AG 76 / AAA 65 / ILO 51），无抽样；报告的是逐例值的中位数，p10–p90 已在文中给出。§13.1–§13.3、§13.8 与 §14 基于**完整的 36 例测试集产物**（`metrics.json` / `per_case_metrics.csv` / 9 臂 summary / 三种子 paired summary），亦无抽样。

注意 §13.4–§13.7 的样本是**全库 192 例**（含 138 训练例），而 §13.1–§13.3 的样本是 **test36**。天花板与方差分解是数据本身的性质（与 split 无关），因此用全库；模型指标只能用 test36。两者不可互相代入。

**推导 vs 实测**：§13.3 前沿表的 p99 列由 §13.1 的恒等式给出，已逐例验证 \(\lambda=1\) 与落盘值最大偏差 \(3\times10^{-6}\)；top10 列含一个实测 1.020 的 Jensen 修正。\(R^2_\lambda=R^2\lambda(2-\lambda)\) 假定 \(\hat z=\mathbb E[z\mid x]\)，该假定由 pooled slope 0.6325 vs pooled \(R^2\) 0.6181（差 0.014）支持，逐例口径下方向一致但有离散（见 §13.2 表）。**物理 \(R^2\) 与物理 MAE 在 λ 下的行为未做解析预测，必须由 P0-2 实测**——这是整张前沿表唯一未定的两列。

**本轮的一处自我更正**：初稿曾把 `per_case_metrics.csv` 的 `cal_*` 列当作物理值，据此声称"逐例 top10 = 0.546、pooled 高估 47%"。核对 `write_reports` 后确认该列是**标准化空间**，正确的逐例·物理值是 **0.4278**，pooled 与逐例的真实差是 **+0.057**（p99 为 +0.124、max 为 +0.140）。D-01 已按更正后的数字重写；结论方向不变（口径必须标注），但"核心前提被严重高估"这一说法**不成立**——top10 一行基本可用，需要修正的是 p99/max/动态范围三行。
