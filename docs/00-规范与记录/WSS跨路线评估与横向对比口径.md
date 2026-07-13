# WSS 跨路线评估与横向对比口径

> 用途：统一 WSS 最小化线、V3P 和多物理目标横向对比的指标、选模和结论边界。
> 生效日期：2026-07-13。
> 优先级：本文是跨路线指标的共享口径；具体 Job 和数值仍以各路线跟踪文档为准。

## 1. 先判断“能不能比”

任意两个数值并排前，必须同时核对以下八项：

1. 病例划分：同一 train/val/test 病例，且 duplicate group 不跨 fold。
2. 时相：同一 peak frame，或同一组心动周期帧。
3. 空间域：同一 wall mask、裁剪边界和点集映射。
4. 目标：同为 raw WSS magnitude，不将分量、对数目标或 gauge pressure 混入。
5. 输入信息：几何-only、临床可测 BC 和 CFD-oracle BC 分层。
6. 采样与权重：同一点数、病例权重和完整点云评估方式。
7. 指标公式：同一 R² 中心、pooling 和物理空间。
8. 选模与报告集：只用 val 选 checkpoint，不用 test 选模；对比时使用同一 checkpoint 规则。

只要有一项不同，就不得使用“公平对比”、“只差 0.1”或“已证明达到同一上限”等表述。

## 2. 三级可比性标签

| 级别 | 定义 | 允许的结论 |
|---|---|---|
| A｜直接对比 | 第 1 节八项全部一致 | 可报绝对差、配对病例差和统计不确定性 |
| B｜协议化参考 | 目标和指标公式一致，但 split/时相/输入等不同 | 只能说处于什么数值区间，不得宣称优劣或精确 gap |
| C｜叙事背景 | 目标、域或 R² 分母不同 | 只用于解释任务差异，不进入定量排名 |

当前 V3P `wss_r2_wss≈0.429` 与 wss_min `R²_field_raw≈0.31–0.36` 最多是 **B 级参考**：两者都是 wall WSS magnitude 的 pooled 点级 R²，但 split、时相、节点数、输入（V3P 含 `t_norm+BC`）、选模和数据版本不同。现有数值不能用来认定“真实差距约 0.1”或“信息上限已锁定”。

V3P 绝对压力 `r2_p≈0.92–0.96` 与 wss_min gauge pressure `R²≈0.53` 属于 **C 级背景**：目标中心、输入 BC 和 R² 分母都不同。

## 3. WSS 主结果的指标层级

WSS 不使用单一 pooled R² 决定路线成败。主表固定为四层：

| 层级 | 必报指标 | 作用 |
|---|---|---|
| 点级总体 | `R²_field_casebalanced`、`RMSE`、`MAE` | 病例等权地评估整体重建；作为首要点级指标 |
| 病例稳健性 | 逐病例 R² 中位数、P10、负 R² 病例数；配对 bootstrap CI | 防止少数大病例/高方差病例主导结论 |
| 幅值与定位 | 每例 top10 ratio、top10 IoU、high-WSS MAE、Spearman | 分开“幅值收缩”与“热点位置错误” |
| 下游任务 | 区域 mean/p95/高 WSS 面积误差；若有多时相再报 TAWSS/OSI/RRT | 判断是否支撑实际科学/临床用途 |

`R²_field_raw`（按点 pooled）保留为与历史结果衔接的次指标。`R²_casemean` 是“每例 R² 的算术平均”，对低方差病例非常敏感，不再单独作为部署主指标；必须与中位数、P10、失败率和物理单位误差同报。

## 4. 压力和速度不复用 WSS 的成功标准

### 4.1 压力

- gauge pressure 主问题是病例内空间型态；主报 case-balanced R²、每例空间 R² 分布、MAE/RMSE（Pa）、Spearman 和压差/梯度误差。
- 绝对压力必须另报“只预测病例/时相均值”的 level-only baseline，并分开 within-case 和 between-case/time 方差。
- CFD 出口压力/RCR 输入始终标记 `oracle_non_deployable=true`。

### 4.2 速度

- 主结果优先使用联合向量模型，报 `u/v/w`、`|v|`、方向夹角/余弦、符号错误率和速度分层误差。
- 三个彼此独立的标量 `u/v/w` 模型只是坐标帧诊断，不作为速度物理主基线，也不直接用于恢复 WSS。
- 速度预测精度不能代替“速度→WSS” oracle Gate；后者必须在同点、同法向深度、同采样密度下独立验证。

## 5. 选模与 Go/No-Go

1. 每个目标使用自己的 val 选模规则；压力/速度不复用 WSS top10 复合分数。
2. 开发 Gate 使用“相对冻结 control 的配对改善 + 护栏”，不使用跨目标通用的 `R²=0.70` 单阈值。
3. 建议 WSS 单 seed 开发 Gate：`R²_field_casebalanced` 或中位每例 R² 至少 `+0.02`，且另一项不明显退化；top10 ratio/IoU 下降不超过 `0.05`；失败率不增加。
4. 单 seed 只决定是否扩展。确认性结论需三 seed，并使用 duplicate-grouped repeated validation 或按病例配对 bootstrap。
5. `0.70` 可保留为长期理想目标，但在给定实际用途、临床容忍误差和统一外部验证前，不得写成所有物理目标的通用“工程可用线”。

## 6. V3P 桥接实验的最小定义

若需回答“wss_min 与 V3P 到底差多少”，不从现有两个总结数值相减，而是新建 Bridge 协议：

1. 同一 duplicate-grouped split 和同一 peak frame。
2. 同一批 wall points，同一 raw WSS magnitude 真值和物理单位。
3. 同一评估脚本输出第 3 节四层指标。
4. 至少分两个输入层：`geometry-only` 与 `geometry+measurable-BC`；CFD-oracle BC 单独作上限。
5. 冻结 val 选模，最终报告集只评估一次。

在 Bridge 协议完成前，V3P 只是历史参考带，不是 wss_min 的直接对手。
