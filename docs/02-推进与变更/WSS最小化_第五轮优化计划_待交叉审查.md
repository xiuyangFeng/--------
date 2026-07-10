# WSS 最小化路线 · 第五轮优化计划（待交叉审查）

> 版本：v0.1（讨论稿）  
> 日期：2026-07-10  
> 状态：**待其他智能体交叉审查，未授权执行**  
> 执行边界：本文仅冻结讨论结论、实验假设、最小矩阵和 Go/No-Go 规则；在交叉审查完成且用户明确批准前，**不得据此修改训练代码、生成新 split、改动数据 manifest、提交 Slurm 作业或查看 legacy test16**。  
> 当前状态入口：[WSS 最小化训练实验跟踪](WSS最小化_训练实验跟踪.md) / [WSS 最小化代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md)  
> 上一轮归档：[第四轮优化计划执行总结与归档](_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)  
> 新队列证据：[AAA / ILO 新队列数据可用性审计](新队列数据可用性审计_AAA_ILO_2026-07-10.md)

---

## 0. 本轮已冻结的决策

第五轮在执行前先冻结以下边界；交叉审查若建议改变其中任何一项，必须形成显式修订记录，不得静默改写。

1. **实验主队列仅为 AG。** AAA、ILO 与 AG 保持三个独立队列；第五轮不做三队列直接混池、不用 AAA/ILO 反向选择 AG 模型。
2. **AAA 与 ILO 只推进数据治理。** 输入、WSS、配准、方向、边界条件口径或病人分组存在问题的单元，先从 active pool 移出，保留原始数据、证据和可恢复状态。
3. **任务仍为峰值收缩期逐点 WSS 标量回归。** 当前预测一个 WSS 标量，`out_dim=1`；第五轮不训练 WSS 三分量，不增加多任务监督头。
4. **主目标为完整壁面 `R²_field`。** `R²_casemean`、MAE 和高 WSS 指标作为次指标与安全护栏，不再让尾部指标主导选模。
5. **先做 AG 内部 learning curve。** 先回答病例数是否仍是主瓶颈，再决定是否投入跨队列预训练、复杂模型或生成式扩充。
6. **密度问题仍需验证。** AG 原始点云约 1–2 万点并不意味着无需处理；关键矛盾是训练常用 1000/2000 点、推理使用完整点云，局部邻域统计不一致。
7. **暂不启用病例级多任务头。** 若后续需要全局上下文，只允许先探索“单 WSS 输出 + 无额外监督的 global feature broadcast”，不得新增 mean/p99 等辅助监督头。

## 1. 第五轮要回答的问题

本轮不是继续扩大 sweep，而是按因果顺序回答五个问题：

1. **现有 PointNeXt 在训练集上能拟合到什么程度？** 若 train 也低，则瓶颈在表示、密度或优化；若 train 高而 val 低，则病例数和泛化更重要。
2. **训练/完整推理的密度错位是否实质压低 `R²_field`？** 先用既有 checkpoint 做无需重训的诊断，再决定是否改训练协议。
3. **AG 的 13→26→40→53 病例 learning curve 是否仍持续上升？** 这是判断“补真实数据是否值得”的核心证据。
4. **当前实现是否充分利用局部邻域？** 用逐点 MLP 对照和 PointNeXt 相对位置归一化做低成本判断。
5. **CFD 标签和评价口径本身是否构成精度上限？** 从网格、求解、相位定义、表面积权重和医学目标检查模型之外的上限。

## 2. 已有证据与当前锚点

### 2.1 第四轮锚点

- 配方：`xyz + abscissa_norm + local_radius + curvature`，PointNeXt-S 残差版，fixed FPS 2000，fixed target-weight α=2，v2_dev1，val-only。
- 三 seed 2000 点：`R²_field=0.339±0.020`、`R²_casemean=0.188±0.025`、top10 ratio `0.416±0.025`。
- 1000 点与 2000 点的 `R²_field` 持平；1000 的 `R²_casemean` 稳定略高，但不足以推翻 2000 协议锚点。
- raw-Huber、multi-start FPS、4000 点、`coord_scale`、`radius_gradient` 均未形成可升级主线的稳定证据。

### 2.2 密度敏感性证据

- 同一 FPS-2000 索引上，稀疏推理与完整点云推理的标准化输出 RMSE 约 `0.164–0.228`，三 seed 方向一致。
- 完整推理相对稀疏推理存在约 `0.03–0.07σ` 的系统性偏移。
- ball-query `max_num_neighbors=16` 的 cap 命中比例：子采样平均约 `0.37`，完整点云平均约 `0.97`。
- AG 训练病例原始点数约 `9700–21500`、中位约 `12700`；虽然绝对点数不算极端，但相对 2000 点训练密度仍相差约 5–10 倍。

因此第五轮所谓“密度处理”指**模型侧密度协议一致化**，不代表删除原始点云，也不代表把完整壁面评估改成只评 2000 点。

### 2.3 高 WSS 错误结构

- 模型已经学到部分全场排序和热点大致位置，但高 WSS 内部排序、幅值和局部细节明显不足。
- 残差随真值分位单调转为欠估；最高两个 decile 的欠估率接近 90% 和 97%。
- 全局 smearing/偏置校准没有稳定恢复 top10；单变量 raw-space Huber 也未通过第四轮 Gate。

这支持“回归均值化存在”，但不支持把第五轮直接升级为 Gaussian NLL、多任务分位头或更激进尾部损失。

## 3. 三队列数据治理方案

### 3.1 队列边界

| 队列 | 第五轮角色 | 是否参与 AG 选模 | 当前动作 |
|---|---|---:|---|
| AG | 唯一主实验队列 | 是 | density、train-fit、learning curve、单任务表示诊断 |
| AAA | 独立候选队列 | 否 | 清单治理、异常隔离、病人分组、保留证据 |
| ILO | 独立候选队列 | 否 | before/after 病人级分组、异常隔离、保留证据 |

### 3.2 状态机

新队列不物理删除数据，使用版本化 manifest 控制 active scope：

| 状态 | 含义 | 是否进入 stats/train/val/test |
|---|---|---:|
| `ready` | 已通过当前队列全部硬门槛 | 是，但仅限本队列未来实验 |
| `excluded_hard` | 空数据、缺关键输入、死导出、不可接受混合区域等 | 否 |
| `quarantine_recoverable` | 选步、centerline 平移、配准等已有可验证修复方向 | 否 |
| `quarantine_bc` | `vf-in` 定义、单位或数量级不一致 | 否 |
| `watch` | 方向、轻度配准、疑似重复病人等尚未定性 | 否；人工确认后再转状态 |
| `recovered` | 修复后重新通过全链 QA | 是，需记录修复版本 |

每条 manifest 记录至少包含：

- `unit_id`、`patient_group_id`、队列、phase；
- `status`、`reason_codes`、数值证据和证据文件；
- 首次发现日期、最后复核日期；
- 是否可恢复、建议修复方式；
- 是否允许进入 stats/train/val/test；
- 人工复核结论与修复版本。

建议 reason code 至少覆盖：

- `DATA_EMPTY`、`CENTERLINE_MISSING`；
- `WSS_ZERO`、`WSS_NEAR_ZERO`、`MIXED_WALL_ZONE`；
- `REG_ORIGIN_OFFSET`、`REG_AXIS_TILT`、`ORIENTATION_FLIP`；
- `BC_SCALE_OR_SCHEMA_MISMATCH`；
- `PATIENT_DUPLICATE_OR_PAIRED`。

### 3.3 新增隔离项

在既有 `exclude_manifest_AAA_ILO.json` 基础上，交叉审查应重点确认以下新增标记：

1. `ILO/LIU_BAO_JUN-0/after`：峰值步 WSS mean≈`0.0068`、p90≈`0.0161`；全 81 步 p90 最高仍仅≈`0.435`，先标 `WSS_NEAR_ZERO`，不得继续计入 ready 数量。
2. 以下 `vf-in` 与主流病例数量级差异极大，先标 `BC_SCALE_OR_SCHEMA_MISMATCH`，不得直接定性为 CFD 错误：
   - `AAA/ruputer/WANG_SHUN_WEN`
   - `AAA/ruputer/ZHANG_ZAO_SHUAN`
   - `AAA/unruputer/GUO_YU_YING`
   - `ILO/DONG_KE_QIN-0/before`
   - `ILO/GUO_YU_SHU-0/after`
   - `ILO/LIU_YUE_DONG-0/after`
   - `ILO/WANG_LI_MIN-0/before`（已因配准层隔离，追加 BC reason）
3. `orientation_flip_watch` 内单元在方向未确认前统一从 active pool 移出。
4. ILO before/after 必须共享 `patient_group_id`；AAA 与 ILO 的同名病例需人工确认是否同一病人。原始队列发现的同名包括 `GUO_AI_JUN`、`SHEN_CHUN_WANG`、`ZHANG_MAO_JIN`。

> 计数规则：任何 ready 数量必须由 manifest 现算，不再沿用“162”作为固定事实。第五轮不以新队列 ready 数量作为完成指标。

## 4. 任务定义与评价协议

### 4.1 任务

- 输入：病例壁面几何点云及已批准的单任务几何特征。
- 输出：每个壁面点一个峰值收缩期 WSS **标量**。
- 目标空间：训练沿用当前 train-only `log_z` 口径；评价在反归一化后的原始 WSS 空间。
- 不包含：WSS 三方向分量、TAWSS、OSI、RRT、多任务辅助头。

### 4.2 主指标

第五轮主指标固定为：

1. **`R²_field`**：完整壁面、原始 WSS、所有验证病例点汇总后的 pooled R²。
2. `R²_casemean`：次指标，防止整体提升只来自少数高方差病例。
3. MAE：次指标，约束绝对误差。
4. top10 ratio、top10 IoU、max ratio：安全护栏和错误分析，不主导选模。

AG 第五轮为保持与历史可比，主表继续报告现有顶点级 `R²_field`；同时新增 surface-area-weighted 指标作为诊断列。面积权重实现未通过合成测试和病例 QA 前，不替代历史主指标。

### 4.3 checkpoint 选择

第五轮不继续使用以 `R²_casemean` 为主的第四轮复合 score。预注册规则为：

- checkpoint 第一排序键：val `R²_field` 最大；
- 若差值 `<0.005`：优先 `R²_casemean` 更高者；
- 仍近似并列：优先 MAE 更低者；
- 始终保存 val 前 3 个 checkpoint，但不得用 legacy test 反选。

安全护栏：

- 相对同 fold、同 seed control，`R²_casemean` 不得下降超过 `0.02`；
- MAE 不得恶化超过 `3%`；
- max ratio 不得超过 `1.5`，不得出现单病例爆峰；
- 提升不得只来自一个 val 病例。

## 5. Stage A：零/低 GPU 诊断

### A0｜训练集拟合能力审计

对当前 B1 三 seed best checkpoint，在不改权重的前提下补齐：

- train / val 的完整密度指标；
- train / val 的 canonical-2000 指标；
- `R²_field`、`R²_casemean`、MAE、top10、IoU、残差 decile；
- train-val gap 与三 seed 方差。

判读：

- train 高、val 低：优先归因于病例数、分布和过拟合；
- train 与 val 均低：优先归因于表示、密度协议或优化不足；
- train 整体高但 train high-WSS 仍差：尾部目标/局部几何/标签可靠性优先。

在这一步完成前，不启动大模型实验。

### A1｜无需重训的 canonical-density 推理

对同一 checkpoint、同一 val 病例比较：

| ID | 网络输入 | 完整壁面输出方式 |
|---|---|---|
| D0 | 原始完整点云 | 直接逐点输出（现行 control） |
| D1 | 固定 FPS-2000 | 表面感知插值回完整壁面 |
| D2 | 多个起点的 FPS-2000 | 各自插值后 ensemble mean |

要求：

- 完整壁面真值和最终指标口径不变；
- 优先使用表面连接或受半径约束的局部插值，禁止跨相邻但不连通的管壁“抄近路”；
- 若只能用欧氏 kNN，必须专项检查分叉、近壁对置面和细颈处的跨壁泄漏；
- 同时报告插值误差上限：用真值在采样点插值回全场，区分“插值损失”与“模型损失”。

A1 Go 条件满足任一：

- 相对 D0，val `R²_field >= +0.02`，且通过全部护栏；或
- 同索引 density drift RMSE 至少下降 50%，且没有指标退化。

最终密度协议的工程目标：同索引标准化输出 RMSE `<0.05σ`、mean shift `<0.01σ`。若当前结构无法达到，记录为模型密度敏感性上限，不通过插值掩盖。

### A2｜表面积权重可行性

- 若存在可靠表面三角形连接，按顶点分摊邻接三角形面积；
- 若只有点云，kNN 局部面积估计仅作敏感性分析，不直接作为正式指标；
- 合成均匀/非均匀采样表面上验证加权指标对重采样近似不变；
- 对 AG 报告 unweighted 与 area-weighted 指标排序是否一致。

### A3｜简单模型与邻域价值基线

增加逐点 MLP baseline，输入特征、loss、split、stats 与 B1 对齐，但不做邻域聚合。目的不是追求最优，而是回答 PointNeXt 的邻域模块提供了多少增量。

- 若 PointNeXt 相对 MLP 的 `ΔR²_field <0.02`：优先检查邻域实现/特征信息；
- 若增量明显：保留 PointNeXt 路线，再做相对位置归一化。

## 6. Stage B：AG 病例数 learning curve

### 6.1 子集设计

以 v2_dev1 的 53 个训练病例为母集，生成一次性版本化嵌套子集：

- `LC13 ⊂ LC26 ⊂ LC40 ⊂ LC53`；
- 按 fast/slow、病例 mean log-WSS、p99 WSS、`n_wall`、`coord_scale` 分层；
- val 8 例固定不变；
- 每个子集只用自身训练标签计算 WSS stats、feature stats 和 loss quantiles；
- 记录各子集 stats 漂移，禁止使用 LC53 标签统计帮助 LC13/26/40。

### 6.2 最小执行顺序

| 阶段 | split | seed | 规模 | 用途 |
|---|---|---|---|---|
| LC-1 | dev1 | 1234 | 13/26/40/53 | 低成本判断曲线形状 |
| LC-2 | dev1 | 7/2025 | 仅 LC-1 曲线可解释时补齐 | 判断 seed 稳定性 |
| LC-3 | dev2/dev3 | 1234/7/2025 | 端点 + 拐点 | 判断 split 稳定性 |

LC-1 前必须冻结密度协议；不得在不同数据规模上同时改变采样、模型、loss 或选模指标。

### 6.3 learning curve 判读

“数据量仍是主瓶颈”的支持条件：

- `R²_field` 随 13→26→40→53 总体非下降；
- 40→53 的三 seed mean `ΔR²_field >= +0.02`；
- 至少 2/3 dev split 方向一致；
- 改善不是仅由一个病例或 WSS stats 偶然变化驱动。

平台判定：

- 40→53 的 mean `|ΔR²_field| <0.01`，且多个 seed/split 一致；
- 则“继续补同分布 AG 病例”降为次要杠杆，优先表示、密度和标签上限。

非单调或方差主导：

- 若规模差异小于 seed/split 方差，不得宣称数据量有效或无效；
- 优先检查子集分层、stats 漂移、训练稳定性和近重复几何。

## 7. Stage C：单任务模型诊断与优化

Stage C 只在 Stage A/B 给出清晰结论后启动，一次只允许一个变量。

### C1｜PointNeXt 相对位置半径归一化

当前局部相对坐标使用 `p_j-p_i`；第五轮首个结构候选为：

`delta_p = (p_j - p_i) / query_radius`

其余模型、数据、loss、seed 和 split 全部固定。该实验优先于加宽/加深模型。

### C2｜单输出 global context（条件候选）

仅当 A0 显示病例级幅值/全局形态是主要误差，才允许探索：

- 编码器 deepest/global pooling 得到病例 embedding；
- embedding 广播并拼接到逐点 decoder；
- 最终仍只有一个 WSS 标量 head；
- 不添加 mean/p99/分位等辅助监督，不改变任务数。

### C3｜密度一致训练（条件候选）

若 A1 支持 density mismatch 为主因，按成本从低到高比较：

1. canonical-2000 train + canonical-2000 inference + 完整壁面插值；
2. multi-density 2000/4000/8000 train + 预注册推理密度；
3. full AG train + full AG inference，batch 1–2 + gradient accumulation。

不得把 `nsample`、训练密度、推理密度和模型宽度同时修改。

### C4｜loss 路线（本轮默认冻结）

第五轮默认沿用 fixed target-weight α=2 锚点。只有满足以下条件才重新讨论 loss：

- 密度协议已稳定；
- learning curve 已判读；
- train-set 仍表现出系统性、空间连续的尾部欠拟合；
- 高 WSS 标签已通过连通性和网格敏感性检查。

本轮不默认启动 Gaussian NLL、quantile/expectile、多任务 moment loss 或激进 label-aware sampling。

## 8. Stage M：医工交叉审计

本阶段以只读审计和小规模复算为主，不与模型超参 sweep 混合。

### M1｜任务声明边界

当前模型应表述为：

> 在固定 CFD 建模、边界条件和峰值时相口径下，从血管几何预测模拟得到的逐点 WSS 标量场。

在没有患者特异流量、血压、黏度或出口阻力输入时，不得扩大表述为“由几何预测真实患者 WSS”。

### M2｜峰值时相适用性

检查：

- 入口流量峰值与局部/全场 WSS 峰值是否同步；
- 病例间是否存在明显相位滞后；
- peak-WSS 是否符合当前疾病问题，还是 TAWSS/OSI/RRT 更相关。

该检查不改变第五轮标量 peak-WSS 任务，只用于限定论文/报告表述和后续路线。

### M3｜CFD 标签精度上限

从代表性低/中/高 WSS 病例中选 3–5 例，核对：

- 网格无关性与近壁层设置；
- 求解残差、周期稳定性和收敛；
- 刚性壁、黏度模型、入口/出口截断和边界条件一致性；
- p95/p99 热点在网格变化下是否稳定。

若 CFD 重复设置本身的 WSS 差异已接近模型误差，必须把该差异报告为可达到精度的标签上限。

### M4｜高 WSS 标签空间可靠性

对热点检查：

- 是否形成空间连续区域，而非单个/少量孤立节点；
- p95/p99、top10 区域和 max 对网格变化是否稳定；
- 热点位置是否与狭窄、分叉、局部半径或流动机制相符。

max 只作爆峰护栏，不作为模型主目标。

### M5｜医学评价补充

在不替换 `R²_field` 主指标的前提下，建议并行报告：

- surface-area-weighted MAE/RMSE/R²；
- top10 或临床阈值以上的面积比例误差；
- 热点连通区域 IoU/Dice；
- 热点中心或峰值点的表面距离；
- 病例 mean/p95/p99 的 Bland–Altman；
- 最差病例、OOD 病例和自动拒绝率。

### M6｜不确定性与 OOD

不新增 NLL 头，先复用三 seed ensemble：

- ensemble mean 作为可选预测；
- seed 间标准差作为 epistemic uncertainty；
- 检查低 `R²_field` 病例是否同时高不确定；
- 未来用于 AAA/ILO 的 OOD/拒绝提示，不用于本轮 AG 选模。

## 9. 统一 Go/No-Go 规则

所有模型增量必须相对同 fold、同 seed、同 density protocol control 比较。

### Gate-1｜单 seed / dev1

- `ΔR²_field >= +0.02`；
- `ΔR²_casemean >= -0.02`；
- MAE 恶化不超过 3%；
- 无 max 爆峰、无单病例支配；
- 训练无 NaN/Inf，stats 与输入口径可复现。

### Gate-2｜三 seed

- 至少 2/3 seed 的 `ΔR²_field >0`；
- mean `ΔR²_field >= +0.02`；
- seed 标准差不高于 control 的 1.25 倍；
- 若只有单 seed 大幅提升，其余退化，判为不稳定 No-Go。

### Gate-3｜重复开发划分

- 至少 2/3 dev split 方向为正；
- 病例级 bootstrap 95% CI 不显示明显整体退化；
- 通过后才允许锁定最终配置。

### legacy test16

- 已被前三轮多次查看，只能称为 legacy benchmark；
- 第五轮开发过程中禁止逐配置运行；
- 配置、density protocol、seed 和 selection rule 全部锁定后，最多运行一次；
- 若需要严格确认性结论，应使用未来未参与决策的新 AG 病例。

## 10. 明确不进入第五轮主线的事项

- AAA/ILO 与 AG 直接混池训练；
- WSS 三分量或幅值+方向联合预测；
- 病例 mean/p99 等多任务监督头；
- Gaussian NLL、quantile/expectile 主线；
- GAN/扩散生成几何；
- 直接扩大 PointNeXt 宽度/深度；
- 同时修改 loss、sampling、点数、邻域和特征；
- 用 legacy test 选择 checkpoint；
- 把 CFD surrogate 的结果直接解释为临床真实 WSS 或患者结局。

## 11. 建议执行顺序与依赖

```text
三队列 manifest 治理（只读/清单）
          │
          ├── AG A0：train-fit 上限
          │
          ├── AG A1：无需重训的 density 推理诊断
          │       └── 若 density Go → C3
          │
          ├── A2：面积权重可行性
          ├── A3：逐点 MLP 邻域价值对照
          │
          └── 固定 density protocol
                    │
                    └── Stage B：13/26/40/53 learning curve
                              │
                              ├── 数据受限 → 优先补真实 AG / 后续独立讨论迁移
                              └── 平台 → C1 相对位置归一化
                                         └── 条件触发 C2 单输出 global context

Stage M 医工交叉审计与上述流程并行，但不得改变主实验口径。
```

## 12. 交叉审查清单

其他智能体审查时，应逐项给出“同意 / 修改 / 否决 + 证据”，至少覆盖：

### 数据与 split

- 三队列隔离是否足够严格；
- 新增 quarantine 名单是否遗漏或过度排除；
- `patient_group_id` 是否覆盖 ILO 配对与跨队列疑似重复；
- learning curve 嵌套/分层是否会泄漏 val/test 信息；
- 子集特定 stats 是否实现且可追踪。

### 模型与密度

- A5 cap 统计是否足以支持 density hypothesis；
- D1/D2 插值是否存在跨壁泄漏；
- canonical inference 是否仍满足“完整壁面评价”；
- train/full 点数差异是否还影响 BatchNorm、SA 数量和最深层拓扑；
- 相对位置除以 radius 是否与当前实现兼容。

### 指标与统计

- `R²_field` 主选模是否可能被高方差病例支配；
- tie-break 和 guardrail 是否预注册充分；
- area-weighted 指标实现是否正确；
- learning curve 的 `+0.02` 阈值是否高于 seed/split 噪声；
- 病例 bootstrap 是否按病人而非点进行。

### 医工边界

- peak-WSS 目标是否符合当前疾病问题；
- CFD 标签是否具备可接受的网格/求解一致性；
- “几何→WSS”的临床表述是否越界；
- 高 WSS 极值是否为稳定物理区域而非数值尖峰。

### 工程执行

- 是否存在未授权自动评估 test 的脚本默认值；
- 新产物是否版本化并可复现；
- 是否保留原始数据和历史 run；
- 是否有一次只改一个变量的 config 对照。

## 13. 第五轮完成定义

第五轮只有同时满足以下条件才可结案：

1. AAA/ILO 独立 manifest 完成状态治理，异常单元不会进入 active scope；
2. AG train-fit、canonical/full density 差异有三 seed 证据；
3. AG 13/26/40/53 learning curve 至少完成 dev1 三 seed，并对关键点做重复 split；
4. 明确裁决当前主瓶颈是数据量、density、表示、标签上限或其组合；
5. 至少完成逐点 MLP baseline 与 PointNeXt 相对位置归一化的条件性判读；
6. 主表以完整壁面 `R²_field` 排序，同时报告病例级和医学护栏；
7. 没有用 legacy test 参与第五轮开发决策；
8. 所有 Go/No-Go、配置、split、stats、seed、代码版本和产物路径回填训练跟踪；
9. 若锁定最终模型，只在全部协议冻结后运行一次 legacy test；
10. 形成“结论、适用边界、失败病例、医学解释和下一轮建议”的最终报告。

## 14. 当前状态与下一步

当前仅完成计划讨论与证据整理，尚未执行第五轮任何实验。下一步固定为：

1. 由其他智能体对本文做代码、统计、点云密度、CFD/医学四个方向的交叉审查；
2. 将审查意见以新版本小节或独立审查记录回填，保留 v0.1 决策轨迹；
3. 用户明确批准修订版后，才从三队列 manifest 治理与 AG Stage A 开始执行。

