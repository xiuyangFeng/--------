# WSS 最小化路线 · 第五轮执行计划

> 版本：v1.1  
> 日期：2026-07-12  
> 状态：**F0 已结案——第五轮科学结案 / 当前 61 例与当前协议下内部工程未达标**；不外推大样本学习上限  
> 决策依据：[第五轮优化计划正式版](WSS最小化_第五轮优化计划_正式版.md)  
> 实验状态源：[WSS 最小化训练实验跟踪](WSS最小化_训练实验跟踪.md)  
> 变更日志：[WSS 最小化代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md)

---

## 1. 这份文档怎么用

本文档是第五轮的**唯一执行入口**，用于把正式优化方案交给其他智能体实施。优化计划负责科学口径，本文档负责任务卡和执行约束。

执行智能体必须遵守：

1. 只执行用户明确指派的任务 ID，不自动领取后续阶段。
2. 任务的前置依赖未通过时，不得“先跑了再说”。
3. 早期开发只用 val；不得读取、评估或据此调整 `legacy test16`。
4. 一次只改一个主变量，必须保留 control，不允许无记录的组合调参。
5. 每完成一张任务卡，先回填训练跟踪和推进记录，再由协调者决定下一步。
6. 不修改 `pipeline/` / `training/` 旧主线；第五轮只在 `pipeline_wss_min/`、`training_wss_min/`、对应 split/stats 和 WSS 文档内工作。
7. 不重跑 CFD，不将 AAA/ILO 混入 AG，不将 oracle BC 写成可部署输入。

### 1.1 冲突优先级

出现冲突时按以下顺序处理：

1. 用户当前明确指令；
2. `pipeline_wss_min/AGENTS.md`；
3. 本执行计划；
4. 第五轮优化计划正式版；
5. 历史训练与归档文档。

如代码与文档不一致，先停止执行并记录“代码事实”，不得为了继续作业而自行改变科学口径。

---

## 2. 冻结的任务与验收口径

### 2.1 任务边界

- 主队列：仅 AG（腹主动脉瘤生长速度队列）。
- 输入：病例几何，以及经证明可在实际部署时获取的条件。
- 输出：峰值收缩期 WSS **标量场**，`out_dim=1`。
- 形式：单任务、单头；本轮不做 WSS 三分量、多任务头、TAWSS/OSI/RRT。
- 工程用途：快速生成供医生参考的 WSS 初筛图，不声称自动诊断、治疗决策或直接预测生长/破裂。

### 2.2 共同主指标

所有 R² 均在 denormalize 后的原始 WSS 空间计算。

| 指标 | 用途 | 最终工程阈值 |
|---|---|---:|
| `R²_field_raw` | 所有完整壁面点 pooled，反映全场重建 | `>=0.70` |
| `R²_casemean` | 逐病例 R² 等权平均，防止大/高 WSS 病例主导 | `>=0.70` |
| `R²_field_casebalanced` | 病例等权全场辅助口径 | 必须报告 |
| median / P10 / 负 R² 数 / 失败率 | 描述病例间风险 | 必须报告 |
| top10 ratio / top10 IoU | 高 WSS 保真护栏 | 不得隐藏劣化 |

开发 Gate 中，`R²_field_raw` 与 `R²_casemean` 是共同主目标。对 control 的变化在 `±0.02` 以内视为 indifference band；不得仅依据其中一项宣布 Go。

### 2.3 三种完成状态

| 状态 | 定义 |
|---|---|
| 第五轮科学结案 | 完成密度、数据量、表示、输入信息和 CFD 标签上限的可复核裁决；即使 R² 未达 0.70 也可结案 |
| 内部工程达标 | 锁定配置在全 61 例 OOF 上同时达到两项 R² `>=0.70` |
| 对外/临床确认 | 未来还需未参与开发的新 AG 病例前瞻性确认；不属于第五轮 |

---

## 3. 执行组织与共享工作区规则

### 3.1 角色

| 角色 | 责任 |
|---|---|
| 协调者 | 分配任务 ID、验收证据、更新状态表、主持 G0 分支裁决和 L0 配置锁定 |
| 协议/指标执行者 | 实施 P0，只改评估、Gate 和回归测试相关代码 |
| 诊断执行者 | 执行 A0R/A0M，不修改正式数据划分 |
| 密度/表面 QA 执行者 | 执行 A1，负责 STL→CFD mapping 与 oracle 证据 |
| 实验执行者 | 仅执行 G0 触发的单变量分支、LC 或 OOF |

一个智能体可兼任多个角色，但同一时间不得有两个执行者修改同一文件。

### 3.2 状态值

状态只使用：`NOT_STARTED` / `IN_PROGRESS` / `BLOCKED` / `NO_GO` / `GO` / `DONE`。

- `DONE`：产物、测试、报告和文档回填全部完成。
- `GO`：证据支持进入指定后续分支，不等于后续任务已获用户指派。
- `NO_GO`：已按预注册规则停止该路线，不得改参后无限重试。
- `BLOCKED`：缺前置、证据或用户权限；必须写明 blocker。

### 3.3 任务领取和回填

开始任务前：

1. 读取本文档、`pipeline_wss_min/AGENTS.md`、训练跟踪和相关 README。
2. 记录当前 `git status --short`，不覆盖用户或其他智能体的变更。
3. 在本文档 §4 将任务状态改为 `IN_PROGRESS`，填写执行者和开始时间。
4. 将预注册的 control、主变量、seed、split、指标和停止条件写入任务报告头。

完成任务后：

1. 产物必须落盘，不接受只在对话中汇报。
2. 在训练跟踪回填指标、失败病例、Gate 和结论。
3. 在推进记录文首增加“本次主要修改 / 对应代码文档 / 推进到实验步骤 / 当前状态判断”。
4. 运行聚焦测试和 `git diff --check`；不得自行 stage、commit 或 push，除非用户另行要求。

### 3.4 统一产物根目录

新证据统一放在：

```text
training_wss_min/runs/_round5/
├── protocol/
├── a0_readonly/
├── a0_micro/
├── a1_density_surface/
├── branch_experiments/
├── learning_curve/
├── lock/
├── oof/
└── final_report/
```

数据 QA 资产可放在 `data_wss_min/audits/round5/`，但必须由 `_round5/` 中的报告链接到具体路径。不要覆盖第三/四轮产物。

---

## 4. 总任务看板

| ID | 任务 | 依赖 | 资源 | 当前状态 | 执行者 | 核心产物 |
|---|---|---|---|---|---|---|
| P0 | 评价协议与 Gate 实现 | 无 | CPU | `DONE` | Codex `/root` | `protocol_report.md` |
| A0R | B1 多 checkpoint 只读诊断 | P0 | CPU/GPU 可选 | `DONE` | Codex `a0r_readonly` | `a0_readonly_report.md` |
| A0M | 4 病例 micro-overfit | P0 | 1 GPU | `NO_GO` | Codex `a0m_micro` | `a0_micro_report.md` |
| A1 | 当前 B1 密度与 STL mapping/oracle | P0 | CPU/GPU 可选 | `NO_GO` | Codex `a1_density` | `a1_evidence_report.md` |
| G0 | 第一次分支裁决 | A0R+A0M+A1 | CPU | `DONE` | Codex `/root` | `branch_decision.md` |
| B-REP | 表示/实现修复分支 | G0 触发 | 1–4 GPU | `NO_GO` | Codex 协调 | 单变量报告 |
| A0D | 基础拟合链分解审计 | A0M+B-REP No-Go | CPU + 1 GPU | `DONE` | Codex 协调 | `a0d_fit_chain_report.md` |
| A0E | step-based 预算修正后的 dev1 control 重锚 | A0D | 1 GPU | `DONE` | Claude | `a0e_control_report.md` |
| G1 | 第二次分支裁决（纳入 A0D/A0E） | A0D+A0E | CPU | `DONE` | Claude | `branch_decision_g1.md` |
| B-DEN | 密度修复 D1/D2 分支 | G0 触发 | 1–4 GPU | `BLOCKED` | 待分配 | 密度对照报告 |
| B-BC | BC 资产与 A0.5 分支 | G0 触发 | CPU + GPU | `BLOCKED` | 待分配 | BC 审计/对照报告 |
| LC | 3 链 learning curve | G1 指向数据量且协议冻结 | 最多 4 GPU | `DONE` | Claude | `learning_curve_report.md` |
| G2 | 第三次分支裁决（纳入 LC） | LC | CPU | `DONE` | Claude | `branch_decision_g2.md` |
| B-BC审计 | BC 资产 read-only 审计（B-BC 门） | G2 触发 | CPU | `DONE`(PASS/可部署 B-BC 关闭) | Claude | `bc_audit_report.md` |
| CFD审计 | §6 CFD 可信性/标签上限审计 | G2 触发 | CPU | `DONE`(标签噪声小) | Claude | `cfd_audit/cfd_audit_report.md` |
| P2 | loss 目标错位单变量探针 | G2 触发 | 1–2 GPU | `NOT_STARTED`(可选/非部署路径) | 待分配 | `p2_loss_probe_report.md` |
| L0 | 最终配置锁定 | 开发分支完成 | CPU | `BLOCKED` | 协调者 | `model_lock.json` |
| OOF | 61 例 5-fold × 3 seeds | L0 | 最多 4 GPU | `BLOCKED` | 待分配 | `oof_engineering_report.md` |
| T16 | legacy test16 一次弱确认 | OOF | GPU 可选 | `BLOCKED` | 待分配 | `legacy_test16_report.md` |
| F0 | 第五轮结案 | 开发杠杆用尽（LC/B-BC/CFD）；T16/OOF 经记录跳过 | CPU | `DONE`(科学结案·工程未达标) | Claude | `final_report/round5_final_report.md` |

**可并行边界**：P0 通过后，A0R、A0M、A1 可由不同智能体并行，前提是不修改同一文件。G0 之后不得默认并行所有分支；只启动被证据触发的最小分支。

---

## 5. 任务卡

### P0｜评价协议与 Gate 实现

**目标**：先固定量尺，不运行新训练。

**允许改动**：`training_wss_min/metrics.py`、评估/Gate 脚本、对应 tests 和 README。

**必做**：

1. 新增 `R²_field_casebalanced`，并保留 `R²_field_raw` 和 `R²_casemean`。
2. Gate 实现 `±0.02` indifference band 和 field/casemean 共同主目标。
3. 开发评估默认只允许 val；为 test 访问加显式 guard/参数和回归测试。
4. 补充 case-balanced 手算小样例、常数标签、单病例、病例点数悬殊和 test guard 测试。
5. 用一个旧 run 复评，确认原有指标在容差内不变。

**验收**：聚焦测试通过；旧 run 回归对齐；未读 test16；产出 `_round5/protocol/protocol_report.md`。

**停止条件**：发现历史 `R²_field` 命名/聚合口径与文档不一致时，先标记 `BLOCKED`，不得静默改写历史结论。

### A0R｜B1 多 checkpoint 只读诊断

**目标**：区分模型容量不足、早停/选模问题与密度迁移。

**固定对象**：`r4_dev1_b1_tgtw_fixedq_s{1234,7,2025}`；使用各 run 内 config 和 stats 快照。

**必做**：

1. 枚举 `ckpt_best.pt`、`ckpt_last.pt` 和已有 top-k/candidate checkpoints；只读，不训练。
2. 对 train/val × canonical-2000/full 四象限计算 field/casemean/case-balanced/MAE/top10。
3. 报告 best↔last 的 train-fit 差异、canonical↔full 的密度差异及逐病例异常。
4. 不允许根据 train 指标重新反选正式 checkpoint。

**验收**：产出逐 seed/checkpoint/partition/density CSV、摘要 JSON 和 `_round5/a0_readonly/a0_readonly_report.md`。

### A0M｜4 病例 micro-overfit sanity

**目标**：验证基础前向链、标签、归一化、loss 和优化器能否拟合一个极小训练集。

**预注册**：

- 从正式 train 固定 4 例，覆盖 fast/slow 和低/高 WSS；名单一旦生成不更换。
- 固定 canonical-2000，关闭 val 选模；仅以拟合训练集为目的。
- 使用 B1 配方起步；如失败，允许进行最小数值排错，但每次改动必须记录。

**Go**：四例训练集 `R²_field>=0.95` 且 `R²_casemean>=0.95`。

**No-Go**：在排除评估错误后仍达不到阈值；立即暂停 LC 和大规模 sweep，转 B-REP。

**产物**：固定病例清单、config、完整 loss/指标曲线、逐病例图和 `_round5/a0_micro/a0_micro_report.md`。

### A1｜当前 B1 密度与 STL mapping/oracle

**目标**：判断约 1.5 万壁面点是否需要密度处理，并先证明稀疏预测能否可信地回到 CFD 壁面。

**必做**：

1. 在 B1/v2_dev1 三 seed 重做同索引 canonical-2000/full probe。
2. 重做邻域 cap 审计，统计每层/每病例有效邻域数与截断率。
3. 验证 STL 三角面到 CFD wall 点的坐标系、单位、覆盖率、距离分位和多对一情况。
4. 做真值插值 oracle：仅用稀疏真值回插完整壁面，测量映射本身的误差上限。
5. 仅当 oracle 通过后，才得提议 D0/D1/D2；正式指标仍在完整 CFD wall 真值上计算。

**oracle 验收阈值**：执行者必须在运行前把数值阈值写入报告头，至少包含 field/casemean、top10 ratio/IoU 和 mapping 距离阈值；不得看结果后补阈值。

**产物**：mapping 逐病例 CSV、oracle 指标、失败图例、可复跑脚本和 `_round5/a1_density_surface/a1_evidence_report.md`。

### G0｜第一次分支裁决

G0 不运行新实验。协调者必须把 A0R/A0M/A1 证据填入下表，一次只批准最小必要分支：

| 证据 | 主判读 | 批准分支 |
|---|---|---|
| micro-overfit 失败，或 canonical train-fit 仍低 | 基础链/表示/优化问题 | B-REP；暂停 LC |
| canonical 明显优于 full，且 oracle 通过 | 密度迁移主导 | B-DEN |
| train 高、val 低，密度差异不主导 | 病例数/泛化可能主导 | LC |
| 几何模型 train 高但跨病例出现不可辨识差异 | 疑似缺少全局 BC | B-BC 的证据审计阶段 |
| 多种机制同时存在 | 无法因果归因 | 按影响量和成本排序，仍只先启动一支 |

G0 产物为 `_round5/branch_experiments/branch_decision.md`，必须列出：证据、裁决、被拒绝分支、下一张获准任务卡和停止条件。

### B-REP｜表示/实现修复分支

候选顺序：

1. 逐点 MLP 基线，判断邻域是否真正提供增量；
2. STL 表面特征（法向、面积/密度权重等），前提是 A1 mapping QA 通过；
3. C1 PointNeXt 相对位置半径归一化；
4. C2 单输出 global context，仍保持 `out_dim=1`；
5. 其他单任务容量/优化修复。

每次仅允许一个候选相对冻结 B1 运行 Gate-1。只有共同主指标超过 indifference band，且 top10 护栏未明显劣化，才进入三 seed Gate-2；否则 `NO_GO`。

> **A0D 后关闭本分支**：B-REP 的触发前提「四病例 micro 拟合失败＝基础链/表示坏了」已被 A0D 证伪——拟合失败实为「每 epoch 仅 1 个 batch、全程仅 160 次 optimizer update、余弦 LR 提前退火」的训练预算伪影，同样四病例在修正配方下可到 `0.98–0.99`。因此不再新增表示/实现 micro 候选。若后续 LC 显示输入信息受限，另走全局条件特征或 B-BC，而非回到此分支。

### A0D｜基础拟合链分解审计

**触发原因**：A0M 与 M1/C1/C2 的四病例 train-only micro 均未达 `R²_field>=0.95` 且 `R²_casemean>=0.95`，不能直接进入 dev1、LC 或新架构 sweep。

**先做 CPU 只读审计**：

1. normalization/denormalization 往返误差、run 内 stats 与 fixed quantile 使用口径；
2. canonical-2000 索引、特征、WSS 标签的逐点对齐与重复/近重复特征冲突；
3. AMP 首步跳过、scheduler、BatchNorm、梯度与训练目标/原始空间 R² 是否错位。

**最小训练链**：只能根据上述证据冻结。默认优先为单病例、fixed canonical-2000、逐点 MLP、AMP off、无 scheduler、plain normalized-space MSE、last checkpoint。单病例未过 `0.95` 则停止并返回标签/表示/实现审计；单病例通过后才可进入四病例简化对照。所有训练仍必须通过 Slurm 提交，不读 val/test16。

**执行结论（2026-07-12，`GO`）**：三份只读审计（normalization/loss、点—标签对齐、optimizer/AMP/BN）均未发现 stats/对齐/实现错误；单病例简化 MLP `0.991–0.999`、four-case shared `0.980/0.983`、只恢复 fixed target-weight `0.994/0.992`，全部达 `0.95/0.95`。证明旧 micro `NO_GO` 是训练预算伪影而非拟合上限，target-weight 单独也不是缺口原因。两个附带信号写入 G1：跨病例局部映射不可迁移（仅其他病例 15NN casemean `-4.10`），normalized-MSE 与 raw-R² 目标错位（高 WSS 同等 normalized 扰动的 raw 代价约 `79.75×`）。裁决交 A0E→G1。

### A0E｜step-based 预算修正后的 dev1 control 重锚

**触发原因**：A0D 证明旧四病例 micro 的 `NO_GO` 主因是「每 epoch 仅 1 个 batch、全程仅 160 次 optimizer update、余弦 LR 在指标仍上升时已退火到 `min_lr`」。进入 LC 前必须先在正式 dev1 上把这个预算口径纠正并复核，避免用被污染的训练历程测量数据量增量。

**唯一主变量**：训练预算从「epoch」改为「optimizer step」，并采用不会中途饿死的 LR（step-based，或摊到真实长 horizon 的余弦，floor 不得在训练指标仍上升时触达）。其余保持 B1（`r4_dev1_b1_tgtw_fixedq`）配方、`split_AG_wss_min_v2_dev1`、fold stats、seed `1234` 不变。

**必做**：

1. 记录等价 optimizer step 预算与逐 step LR/指标曲线；确认末段 LR 未在 val 指标仍改善时触底。
2. val-only 复评两项主 R² 与 top10 护栏，与历史 B1（val `R²_field≈0.351` / `R²_casemean≈0.214`）逐项对照。
3. 明确区分两种结果：与历史持平（预期）→ 确认 `0.34` 是真实泛化上限、非训练历程伪影；显著更高 → 历史基线本身被旧预算低估，需连带复核 A0R 与第四轮结论。
4. 训练 Slurm-only，不读 test16。

**产物**：`_round5/a0e_control/a0e_control_report.md`、逐 step history、val-only 指标与对照表。

**Gate**：本任务是协议卫生与基线重锚，不设增量 Go/No-Go 阈值；输出用于冻结 LC 的训练协议与 control。

### G1｜第二次分支裁决（纳入 A0D/A0E）

G1 不运行新实验。协调者把 A0D、A0E 与既有 A0R/A1 证据合并，改写 G0 判读并只批准一支后续分支。

| 证据（A0D 后） | 主判读 | 批准分支 |
|---|---|---|
| 拟合能力已确认（`0.98–0.99`），dev1 ~epoch49 后 val 退化，跨病例局部映射不可迁移 | 病例数/泛化主导 | LC（先做，公共终点 53 例） |
| 若 LC 显示 53 例仍有明显斜率但未达标 | 输入信息可能受限（缺全局/BC 条件） | 再评估 B-BC 证据阶段或全局条件特征，不回 B-REP |
| normalized-MSE 与 raw-R² 目标错位（`79.75×`） | 与数据量正交的独立杠杆 | 作 P2 并行探针，不抢在 LC 前搅混变量 |
| 密度迁移 | 受 A1 STL mapping `0/8` 阻断 | 维持 B-DEN `BLOCKED` |

G1 产物为 `_round5/branch_experiments/branch_decision_g1.md`，须列出：改写后的证据、裁决、关闭的 B-REP、下一张获准任务卡（LC）与停止条件。

### B-DEN｜密度修复 D1/D2 分支

仅当 A1 oracle 通过且 G0 确认密度迁移主导时执行：

- D1：在 STL/表面连接上将稀疏预测回插完整 CFD wall。
- D2：稀疏回插与 full-density 直接推理的受控 ensemble。
- D2 必须设置 top10 ratio/IoU 护栏，防止平滑到丢失峰值。
- full-density train 后置；BatchNorm 与 batch size 未隔离前，不得称为纯 density 消融。

### B-BC｜BC 资产与 A0.5 分支

该分支先做证据，后决定是否训练。

**BC 资产验收**：

- `audit_ag_bc.py` 或等价可复跑脚本；
- 逐病例入口 UDF 系数/固定分母、RCR、监测完整性、peak flow split CSV；
- train/val/test scope、缺失值规则、源文件路径、时间戳和摘要 JSON；
- 至少 5 个病例人工对照与单位核对。

审计通过后才可预注册 A/B 对照。B1 出口面积可作可部署几何候选；B3p/B3s 必须标记 `oracle_non_deployable=true`。`B3−A` 只称为在相同模型/协议下加入记录 BC 的**预测增量**，不得写成因果效应或全部可解释方差。

### LC｜3 链 learning curve

**前置**：A0D 证明基础拟合能力（取代原「A0M 通过」要求）；A0E 完成 step-based 训练预算修正与 control 重锚；G1 判定泛化/病例数是主要假设。

**矩阵**：

- `R=3` 条独立嵌套链，每条 `LC13 ⊂ LC26 ⊂ LC40 ⊂ LC53`；53 例为公共终点。
- case-drop seed 与 model seed 分离并落盘。
- fast/slow × train-only WSS 三分位为硬分层；不使用 val/test 标签构造分层。
- 在相同链内做配对增量；分别报告病例选择、模型 seed 和 dev split 波动。
- 不自动将 AAA/ILO 或 excluded 病例加入曲线。

**判读**：使用配对斜率、链间方差和 53 例终点的 seed 方差，判断增加同分布 AG 是否还有明确边际收益。不得用单条曲线外推“再加数据必到 0.70”。

**执行结论（2026-07-12，`DONE`）**：field R² 均值 `13→0.258 / 26→0.297 / 40→0.312 / 53→0.310`，配对 40→53 增量 `−0.002±0.049`——**当前 13–53 例范围内约 40 例后出现 ~0.31 暂时平台**，端点 seed std 0.044 超过 26→53 整段增量。stage-1 单 seed"仍在上升"为 s1234 端点偏高伪影，多 seed 纠正。裁决：现有 61 例池内不再继续同配方小步扩展，交 G2；不外推几百/几千例数据上限。产物 `_round5/learning_curve/`。

### G2｜第三次分支裁决（纳入 LC）

G2 不运行新实验。它把 LC（现有 61 例池内暂时平台）与既有 A0D/A0E/A1 证据合并，裁决第五轮剩余可执行杠杆。

| 证据 | 主判读 | 批准分支 |
|---|---|---|
| LC 在 13–53 例范围内约 40 例后出现 ~0.31 暂时平台，40→53 增量≈0 | 现有池的同配方小步加例非第五轮路径 | 不扩展现有池 LC；不作大样本外推 |
| A0D 对齐：跨病例局部映射不可迁移（仅其他病例 15NN casemean −4.10） | 缺全局/病例级（BC）信息是最受指证瓶颈 | **B-BC（首选）：先 read-only BC 资产审计** |
| A0D norm_loss：normalized-MSE↔raw-R² 79.75× | 与输入信息正交的独立杠杆 | **P2：loss 目标错位单变量探针（并行）** |
| 当前 dev 指标 ~0.31/0.21 远低于 0.70 | 无可锁定达标配置 | **不启动 L0/OOF** |

裁决：启动 **B-BC 证据阶段（read-only 审计，B-BC 训练的门）** 为首选杠杆；并行启动 **P2 loss 探针**（不与 B-BC 混变量）。L0/OOF 维持 BLOCKED，直至某杠杆把 dev 指标推过工程门槛。若两杠杆均无法打开平台，则按 §8.1 走"第五轮科学结案 + 未达工程目标"。产物 `_round5/branch_experiments/branch_decision_g2.md`。

### L0｜最终配置锁定

开发阶段结束后，协调者创建 `_round5/lock/model_lock.json`，至少冻结：

- git commit/hash 或代码快照识别；
- 模型、特征、loss、密度、采样、选模和推理协议；
- 配置文件哈希、stats 生成规则、seed 列表；
- OOF fold 生成脚本版本和度量口径；
- 可部署输入列表以及 oracle 输入排除列表。

L0 之后禁止修改配置。如修改任何一项，原 OOF 声明作废，需生成新 lock 并重跑全部 OOF。

### OOF｜全 61 例 5-fold × 3 seeds

**数据协议**：

1. 61 个开发病例每例恰好一次作为 OOF val。
2. 同病人/近重复几何必须整组进入同一 fold。
3. 5-fold 分层仅用 fast/slow × fold-train-only WSS 水平。
4. 每 fold 的 stats/quantiles 只使用该 fold train。
5. 每 fold 运行 seeds `{1234,7,2025}`；先报告各 seed，再以三 seed ensemble mean 生成工程候选图。
6. OOF 期间不得根据已完成 fold 调整配置。

**强制预检**：fold 完整性/去重、病人组泄漏、stats scope、config hash、seed 和 test guard 全部通过后才能提交 GPU 作业。

**工程验收**：用 61 例 OOF ensemble 预测同时计算两项主 R²、case-balanced field、median/P10/负 R² 数/失败率、top10 指标和逐病例 CSV。只有两项主 R² 同时 `>=0.70` 才标记“内部工程达标”。

### T16｜legacy test16 一次弱确认

- 只有 OOF 报告完成且配置未变时才可执行。
- 最多运行一次；不得根据 test16 返回开发。
- 仅作历史弱确认，不代替新 AG 前瞻性验证。

### F0｜第五轮结案

结案报告必须分开写：

1. 科学上学到了什么：密度、数据量、表示、BC 和标签上限裁决；
2. 内部工程是否达标：两项 R² 是否同时达 0.70；
3. 失败病例和适用边界；
4. 可部署输入与 oracle 输入的严格分隔；
5. 与临床应用之间仍缺少的前瞻性证据。

如 R² 未达 0.70，必须明确写“未达内部工程目标”；只要本轮的机制问题均有可复核结论，仍可标记“第五轮科学结案”。

---

## 6. Gate 和止损规则

### 6.1 开发 Gate

| Gate | 范围 | 通过规则 |
|---|---|---|
| Gate-0 | 脚本/协议 | 单测、数据 scope、旧 run 回归和 test guard 通过 |
| Gate-1 | dev1 + seed1234 | 两项主 R² 相对 control 共同改善并超过 0.02，top10 无明显劣化 |
| Gate-2 | dev1 × 3 seeds | 方向在 seeds 间稳定，报告 mean±std 和逐 seed 结果 |
| Gate-3 | dev2/dev3 | 只对 Gate-2 候选检查开发划分稳定性，不用于 0.70 声明 |

如一项主 R² 改善、另一项劣化超过 0.02，默认不通过；只能作为 trade-off 结果报告，不得冒充总体提升。

### 6.2 硬停止条件

- 训练预算口径（A0D 后新增）：所有诊断/正式训练按 optimizer step（而非 epoch）冻结预算，且 LR schedule 的 floor 不得在训练指标仍上升时触达；违反此口径得到的 No-Go 不成立，须按修正预算重跑后再判。
- micro-overfit 未过：停 LC 和大规模架构 sweep；但须先按上一条确认预算充分（A0D 已证明未达阈值可能是预算/LR 伪影而非拟合上限）。
- STL mapping/oracle 未过：停 D1/D2，不得将插值图当定量结果。
- 两个连续 Gate 对同一单变量 `NO_GO`：封闭该分支，不继续扩参。
- 发现 split/stats/test 泄漏：当前受影响结果全部作废。
- L0 后改变配置：已运行 OOF 全部作废。
- 无法确认单位、坐标系或标签对齐：停止训练，先处理数据 QA。

---

## 7. 报告最小模板

每个任务报告至少包含：

```markdown
# <Task ID> 执行报告

## 预注册
- control：
- 唯一主变量：
- split / stats / seeds：
- 主指标与护栏：
- Go/No-Go 和停止条件：

## 实际执行
- 代码/config hash：
- 命令或 Slurm Job IDs：
- 与预注册的偏差：

## 结果
- 汇总指标：
- 逐病例异常：
- 失败/缺失产物：

## 裁决
- Gate：GO / NO_GO / BLOCKED
- 结论：
- 允许的下一任务 ID：
```

---

## 8. 最终交付清单

第五轮结案时必须能从一个索引找到：

- 所有代码、config、split、fold stats 和生成脚本；
- 每个运行的 seed、checkpoint、Slurm Job ID 和日志；
- 每个 Gate 的 control/候选逐 seed 指标；
- 61 例 OOF 预测、逐病例 CSV 和汇总报告；
- density/STL/oracle、learning curve、BC（如触发）的证据资产；
- 科学结案、内部工程状态和临床适用边界；
- 训练跟踪、推进记录和 README 中一致的当前状态。

---

## 9. 当前交接状态

- 第五轮优化计划已收敛为正式版，不再保留历轮审查展开内容。
- 本文档已将正式路线转为可领取的任务卡。
- P0 已于 `2026-07-11` 由 Codex `/root` 完成；control 为 `r4_dev1_b1_tgtw_fixedq_s1234` 的现有 val-only best checkpoint 评估，本任务未运行新训练。
- P0 冻结 `split_AG_wss_min_v2_dev1` / run 内 stats / seed `1234`；两项共同主指标为 `R²_field_raw` 与 `R²_casemean`，护栏为 top10 ratio/IoU，indifference band 为 `±0.02`。
- P0 回归结果：旧 `R²_field_raw` / `R²_casemean` / MAE 最大绝对偏差 `3.24e-9`，在 `1e-8` 容差内；新 `R²_field_casebalanced=0.345164`，未发现历史 field 口径冲突。
- P0 Gate-0 已通过；A0R/A0M/A1 现可并行领取。
- A0R `DONE`；A0M 与 A1 均 `NO_GO`；G0 已裁决优先 B-REP。
- B-REP 已列候选的四病例 micro 结果：M1 逐点 MLP `0.880817/0.844018`，C1 radius normalization `0.803401/0.760602`，C2 global context `0.740682/0.695389`，均未过 field/casemean `0.95/0.95`；对应 dev1 均未提交。
- 当前暂停 LC、B-DEN 与新的 B-REP 训练；下一步需返回 normalization/loss/标签对齐与几何可辨识性审计，不应无记录继续扩参。
- A0D 已完成：四个 single-case 简化 MLP 均达 `0.991–0.999`；four-case shared plain MSE 达 field/casemean `0.979526/0.982533`；只恢复 fixed target-weight 后达 `0.993723/0.992423`。
- 机制修正：旧 `0.844` casemean 不是四病例/六维输入/MLP 的拟合上限，target-weight 单独也不是缺口充分原因；主差异在 160 optimizer steps 与 AMP/scheduler/WD/clip 训练历程组合。
- 执行进展（2026-07-12）：A0E/G1/LC/G2/B-BC审计/CFD审计/**F0 均 `DONE`**；P2 `NOT_STARTED`（可选/非部署）；B-DEN/L0/OOF/T16 经记录跳过或维持 BLOCKED。
- **F0 结案**：§8.1 机制问题全部有可复核裁决 → 第五轮**科学结案**；dev1 field/casemean ~0.31/0.21 ≪ 0.70 → **未达内部工程目标**（§8.2），未跑 OOF（无达标候选）。报告 `_round5/final_report/round5_final_report.md`。
- **CFD 审计**：peak 相位固定步统一、近壁 QA 全清、高 WSS 为真实几何热点、复现 floor ~2%（R²_cap ~0.92–0.96）→ ~0.31 上限**不是**标签噪声，是真实模型/信息上限。附带证实 `HOU_SHEN_QIAN=KANG_XI_MING` 同一几何且均 dev1 train（OOF 须整组）。
- **B-BC 审计（决定性）**：61/61 dev 覆盖完整；入口 Fourier 流量模板跨病例逐字节相同、`Q` 与面积无关（入口流量 CoV≈1.85e-4）→ 入口流量是共享人群模板、非病人特异；唯一 per-case 变化的 BC（出口 RCR/压力/流量分配）全部 `oracle_non_deployable`。**可部署 B-BC 杠杆关闭：无可部署、有信息量、病人特异的 BC 输入可加**；跨病例 WSS 差异由出口 RCR（oracle）驱动。
- A0E：`ctrl`（标准 B1）field/casemean `0.3587/0.2300` 复现 anchor（best epoch 29）→ `0.34/0.23` 是真实泛化上限；`nsl` 更差，LC 用标准 B1 schedule（更正"非饿死 LR"假设）。
- G1：批准 LC、关闭 B-REP。
- LC（30 run，3 链×3 seed）：**field 均值 `13→0.258 / 26→0.297 / 40→0.312 / 53→0.310`，40→53 增量 `−0.002±0.049`——当前 13–53 例范围内约 40 例后出现 ~0.31 暂时平台**；端点 seed std 0.044，不外推大样本上限。
- G2 裁决：现有 61 例池内不继续同配方小步加例 → **B-BC 输入信息** + **P2 loss 目标探针**；不启动 L0/OOF。
- 机制总结（限当前协议）：拟合能力足（A0D）→ dev1 锚点 ~0.34（A0E）→ 现有池 ~0.31 暂时平台（LC）→ 当前可部署 BC 无新杠杆。该机制不否定坐标物理尺度或未来几百/几千例高质量数据的潜在收益。
- 机制定性：本轮瓶颈已从「模型/实现/拟合 bug」收敛为「泛化受限（53 训练例即在 val 过拟合）＋ 现有六维局部输入不携带跨病例/全局条件信息」。「跨病例不可迁移」指同一局部几何在不同病人对应不同 WSS、需补全局/BC 信息，不是「输入特征越多越差」；也不把 train-only `0.992` 误写为新病例精度。
