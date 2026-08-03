# WSS-PINN 阶梯实验矩阵与进度跟踪（历史归档）

> 建立日期：2026-07-30
>
> 当前状态：**PINN 专用 train138/test35 与 173 例 sidecar 已冻结；full F0-UP completed / audited / No-Go；full F1 被配对科学 Gate 阻断且未提交；F2 blocked**
>
> 上位入口：[WSS-PINN 独立路线总入口](README.md)

本文只记录 WSS-PINN 实验族、单次运行、Gate 和结论。第一性原理、物理公式和长篇论证不在这里重复。

## 1. 实验矩阵

| Exp ID | 阶段 | Control | 唯一新增内容 | 主要问题 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| `PINN-S0` | 隔离 | 现有仓库 | 独立目录、数据根、输出根和记录合同 | 能否零污染启动新路线 | completed / No-Run |
| `PINN-P0A` | 数据审计 | 无 | 体点 identity、坐标、单位、时相稳定性 | 数据能否安全对齐 | completed / pass |
| `PINN-P0B` | WSS Oracle | P0A | CFD velocity→WSS 算子 | 真值速度能否恢复真值 WSS | completed / pass |
| `PINN-P0C` | PDE Oracle | P0A | CFD continuity residual | 拟采用算子是否惩罚 CFD 真值 | completed / pass |
| `PINN-P0D` | BC/几何审计 | P0A | 法向、面积、zone 可得性 | 哪些 BC 能 hard/soft 使用 | completed；hard flux/RCR blocked |
| `PINN-P1` | sidecar | P0 | wall/near-wall/core 分槽数据 | 是否形成可复现 physics 数据合同 | pilot + full173 completed / pass |
| `PINN-F0U` | data-only | W0 | 辅助 \(u,v,w\) decoder | 速度场是否可学 | v1、v2 No-Go；v2ext completed / Go |
| `PINN-F0UP` | data-only | F0-U | 压力头与 gauge 监督 | 压力是否可学且不伤害主干 | pilot Go；full173 completed / audited / No-Go |
| `PINN-F1` | 一阶物理 | F0-UP | continuity + no-slip | 一阶物理约束是否有效 | pilot selected；full173 blocked / not submitted |
| `PINN-F2` | WSS 物理 | F1 | \(WSS_{\mathrm{phys}}\) 一致性 | 是否直接改善峰值/高值 WSS | blocked |
| `PINN-F3` | 完整 PDE | F2 | 非定常 momentum | 完整 NS 是否有额外增益 | blocked |
| `PINN-C1` | 开发筛选 | 最优 F0–F3 | mixed test36 单 seed | 是否值得确认 | blocked |
| `PINN-C2` | 确认 | C1 Go | 三 seed + grouped validation/holdout | 是否形成确认性结论 | blocked |

## 2. 不可变的配对原则

每次从父阶段进入子阶段，必须保持：

- 同一患者 split；
- 同一时相集合；
- 同一 wall/near-wall/core sampling manifest；
- 同一 geometry/BC 输入；
- 同一 encoder、decoder 容量和初始化策略；
- 同一训练轮数、优化器和 checkpoint 规则；
- 同一完整壁面评估；
- 只改变矩阵“唯一新增内容”列。

如果因显存或稳定性必须同时改 batch、网络宽度或采样点数，该 run 必须另建 ID，不能冒充严格消融。

## 3. 运行 ID

建议格式：

```text
<exp_id>_<variant>_<split>_s<seed>_<YYYYMMDD>
```

例如：

```text
PINN-F1_cont-noslip_mixed138dev36_s1234_20260815
```

禁止使用 `test`、`new`、`latest`、`final2`、`try_again` 这类不能表达实验含义或版本的名称。

## 4. 每个 run 的登记字段

每个 seed 单独一行，不把多 seed 合并：

| 字段 | 必填内容 |
| --- | --- |
| `exp_id` | 实验族 ID |
| `run_id` | 唯一运行 ID |
| `date_started/date_finished` | 绝对日期 |
| `status` | 统一状态词 |
| `hypothesis` | 本次要验证的单一问题 |
| `control_run_id` | 直接父对照 |
| `only_change` | 唯一新增字段/loss/数据 |
| `split_version` | 明确 split，不写 latest |
| `reporting_role` | development screen / confirmation |
| `seed` | 一个 seed 一行 |
| `data_manifest_sha256` | physics sidecar 与采样合同 |
| `config_sha256` | resolved config |
| `code_commit/dirty` | 代码版本和 dirty flag |
| `parent_checkpoint_sha256` | 若复用 W0 |
| `job_id` | Slurm Job ID；本地运行写 `local` |
| `output_path` | 唯一结果目录 |
| `primary_metric/value` | 预注册主指标 |
| `guardrails` | WSS、速度、压力、residual 护栏 |
| `gate_result` | pass/fail + 具体阈值 |
| `decision` | go / no-go / blocked |
| `notes` | OOM、重跑、缺失、偏离协议 |

## 5. 单次运行记录模板

```markdown
### <run_id>

- 日期：
- Exp ID：
- 状态：
- 假设：
- Control：
- 唯一改动：
- Split / reporting role：
- Seed：
- Data manifest / SHA256：
- Config / SHA256：
- Code commit / dirty：
- Parent checkpoint / SHA256：
- Job ID：
- 输出目录：

#### Preflight

- [ ] 病例和 duplicate group 无泄漏
- [ ] wall / near-wall / core 槽位满足下限
- [ ] 坐标、速度、压力和时间单位已审计
- [ ] 非牛顿参数与 UDF 一致
- [ ] `lambda=0` 退化测试通过
- [ ] 无 NaN/Inf，梯度有限

#### 结果

- 主指标：
- 次指标：
- 高 WSS：
- 逐病例稳健性：
- velocity / pressure：
- continuity / no-slip / momentum residual：
- best/last 敏感性：

#### Gate 与结论

- Gate：
- 结论：go / no-go / blocked
- 失败原因或下一步：
```

## 6. 预注册 Gate

### P0

- `P0-A`：所有纳入病例必须能由稳定 ID 或明确坐标匹配对齐；失败病例显式排除并登记。
- `P0-B`：CFD 真值 velocity→WSS 必须在 AG/AAA/ILO 均稳定，并明显优于 W0；否则 F2 blocked。
- `P0-C`：CFD 真值 residual 必须在无量纲口径下有限、稳定；否则对应 PDE loss blocked。
- `P0-D`：壁面法向必须可用；zone 缺失允许 F0–F2 继续，但 hard inlet/outlet BC blocked。

### F0-U/F0-UP

- 3–5 例过拟合无 NaN/Inf；
- 速度和 gauge pressure 的高拟合阈值在运行前冻结；
- 初始候选为训练病例 \(R^2\ge0.95\)；
- 达不到时不进入 physics loss 调权。

### F1

- continuity 和 wall speed 相对 F0-UP 明显下降；
- 速度/压力监督指标不越过预注册退化护栏；
- `WSS_direct` 不因辅助物理显著退化；
- `lambda_physics=0` 与 F0-UP 行为一致。

### F2/F3/C1

至少一个主门：

- `ΔR²_field_casebalanced ≥ +0.02`；
- `Δhigh-WSS nRMSE ≤ -0.002`。

同时满足护栏：

- normalized/physical R² 退化不超过 `0.01`；
- physical MAE 增加不超过 `0.05 Pa`；
- top10 ratio/IoU 下降不超过 `0.05`；
- 负 R² 病例数不增加；
- residual 改善不是由 loss 尺度爆炸或预测塌缩造成。

### C2

- 三 seed；
- duplicate-grouped repeated validation 或新 holdout；
- 报告病例配对 bootstrap CI；
- `test36` 不作为 C2 的独立确认集。

## 7. 决策记录

### D-001｜2026-07-30｜建立独立 WSS-PINN 路线

- **决定**：代码、派生数据、输出、记录分别使用
  `wss_pinn/`、`data_wss_pinn/`、`outputs/wss_pinn/`、
  `docs/02-推进与变更/WSS_PINN/`。
- **理由**：保护现有 W0 复现合同，并允许 PINN 失败后无成本回退。
- **影响**：旧目录只读；后续新增实现全部落本路线。

### D-002｜2026-07-30｜最终目标仍是 WSS

- **决定**：\(u,v,w,p\) 是训练期辅助场；主报告输出仍是峰值壁面 WSS。
- **理由**：WSS 依赖速度梯度，完整动量需要压力，但工程交付不必输出全体域场。

### D-003｜2026-07-30｜体域点独立分槽

- **决定**：保留 W0 `random5000` 壁面 support；PINN 另加 near-wall/core 体点。
- **理由**：壁面点只能施加 BC，不能单独支撑体域 Navier–Stokes residual。

### D-004｜2026-07-30｜物理口径

- **决定**：非滑移刚性壁面、不可压缩、\(\rho=1060\ \mathrm{kg/m^3}\)、
  UDF 一致的 Carreau–Yasuda 非牛顿流变。

### D-005｜2026-07-30｜P0-C 稳定性诊断分母修正

- **决定**：邻域敏感性由病态的
  `|div24-div48|/mean(|div|)` 改为
  `|div24-div48|/mean(||grad u||)`。
- **理由**：不可压缩目标附近 `div≈0`，旧分母会把小的绝对差放大成无界比值。
- **影响**：首轮旧诊断失败保留在推进记录；修正后重新跑完整 near-wall/core
  分层，三个队列均通过。物理方程、病例、采样和主 residual 阈值未变。

### D-006｜2026-07-30｜F0-U v1 No-Go 后建立配对 v2

- **决定**：保留 v1 结果，不提交 F0-UP；v2 将 WSS 与辅助场拆为独立 trunk，
  增加 field 容量并把 velocity data 权重提高到 10。
- **理由**：v1 有限且 WSS 可拟合，但三病例 velocity R² 仅
  `0.654/0.586/0.567`，未达到 0.95。
- **影响**：F0-UP/F1 同步建立完全配对的 v2 配置；后两阶段仍只在前级结果过门后提交。

### D-007｜2026-07-30｜F0-U 严格逐分量 Gate 与显式续训

- **决定**：F0-U 的 `R²≥0.95` 按每病例的向量聚合、`u/v/w` 和速度模长逐项检查，
  不以 case-balanced WSS 或单一聚合 velocity 指标替代。
- **理由**：v2 的 case-balanced WSS R² 已为 `0.991`，三病例聚合 velocity R²
  为 `0.964/0.948/0.953`，但 AG、AAA、ILO 仍分别有速度分量或模长低于 0.95。
- **影响**：v2 保留为 completed / No-Go。另建 `v2ext`，显式读取 v2
  `last.pt`，只增加 12000 个训练 epoch；F0-UP/F1 更新为同架构、同 20000
  总训练步的配对配置，但继续由科学 Gate 锁定。

### D-008｜2026-07-30｜F0-UP 通过能力 Gate，F1 解锁

- **决定**：F0-UP 的 best/last 均按每病例 velocity 聚合、`u/v/w/speed`
  和 gauge-pressure R² 复核；全部高于 0.95，判定 completed / Go。
- **证据**：best 的最低 velocity R² 为 `0.9872`，三病例 pressure R² 为
  `0.9987/0.9994/0.9998`；last 的对应最低值为 `0.9876/0.9988`。
  CPU 只读重算与落盘 best JSON 的 126 个数值字段最大绝对差
  `2.29e-5`，来自 CUDA/CPU 的 continuity 浮点差。
- **边界**：这是三病例训练期过拟合能力 Gate，不是独立测试泛化结论。
  F0-UP 没有启用物理 loss，continuity/no-slip 不作为其 Go 条件，也没有改善；
  F1 才负责检验这两个 residual。
- **影响**：F1 提交器 dry-run 的前级科学 Gate 已通过；随后按独立提交动作
  启动 F1，提交事实见 D-009 和运行登记。

### D-009｜2026-07-30｜冻结并提交 F1 continuity + no-slip

- **决定**：复用已与 F0-UP 配对的 `f1_pilot_v2.json` 作为正式配置，避免再建
  内容重复但 ID 不同的配置；F1 只把 `continuity_weight` 和
  `no_slip_weight` 从 0 调为 0.1。
- **边界**：`wss_physics_weight=0`、`momentum_weight=0`；模型容量、数据、
  sampling manifest、seed、优化器、batch 和 20000 epoch 与 F0-UP 保持配对。
- **证据**：本地 compile、12 项单测、CPU 单步 dry-run 通过；GPU preflight
  Job `11045` 以 `COMPLETED (0:0)` 通过相同 12 项测试、科学 Gate 和 CUDA
  单步 dry-run。依赖作业 `11046` 已启动正式训练并持续写出有限 loss。
- **影响**：提交当时 F1 只记为 `running`；最终结果与 Gate 见 D-010。

### D-010｜2026-07-30｜F1 continuity 有效但整体 No-Go

- **决定**：F1 Job `11046` 程序完整结束并通过结果审计，但科学 Gate 判
  No-Go；F2 继续 blocked。
- **证据（best）**：相对 F0-UP，三病例 mean continuity RMS 从
  `70.1526` 降到 `0.3359`（`-99.52%`），但 mean no-slip RMS 从
  `0.12447` 升到 `0.13396`（`+7.63%`）。mean velocity R² 从
  `0.99502` 降到 `0.82491`，所有病例/分量/speed 的最低 R² 从
  `0.98720` 降到 `0.69678`。pressure mean R² 仍为 `0.99824`。
- **WSS 护栏（best）**：case-balanced WSS
  `ΔR²=-0.00038`、`ΔMAE=+0.02446 Pa`、`Δtop10 IoU=-0.00913`，
  基本护栏未越界；但 high-WSS nRMSE 从 `0.01340` 升到 `0.02186`，
  没有物理收益。
- **稳健性**：last 同样表现为 continuity 大降、no-slip 上升和 velocity
  R² 大幅退化；CPU 只读重算 best/last 各 126 个数值字段，与落盘 JSON
  的最大绝对差分别为 `1.88e-5/1.43e-5`。
- **解释边界**：这证明当前权重下 continuity 可以被优化，但不能证明学到了
  更准确的物理速度场。结果与“低散度解牺牲数据拟合、no-slip 权重相对过小或
  多目标梯度冲突”一致。last step 的加权 loss 中 velocity/continuity/no-slip
  约占 `83.4%/14.5%/1.4%`，进一步提示 no-slip 约束过弱；但要区分权重、
  尺度和梯度冲突仍需单独的 F1 诊断实验，不能直接进入 F2。

### D-011｜2026-07-30｜完成 F1 内部八臂诊断并冻结候选权重

- **决定**：以 F0-UP 三病例 pilot 为共同对照，完成 continuity-only、
  no-slip-only 与 4 个联合权重臂；预注册要求联合臂的 best/last 两个
  checkpoint 都同时通过 residual、velocity、pressure 和 WSS 护栏。
- **结果**：唯一稳健通过的联合臂为
  `continuity_weight=1e-4`、`no_slip_weight=10`。best/last 的
  continuity mean ratio 为 `0.07687/0.07153`，no-slip mean ratio 为
  `0.12323/0.14650`；最低 velocity R² 为 `0.96886/0.96970`，
  最低 gauge-pressure R² 为 `0.99847/0.99845`。
- **WSS 护栏**：best 的 `ΔR²=-0.00014`、`ΔMAE=+0.00690 Pa`、
  `Δhigh-WSS nRMSE=+0.00319`、`Δtop10 IoU=+0.00131`；last 的对应值为
  `+0.00012/-0.01567 Pa/-0.00500/-0.00389`，均通过。
- **因果解释**：no-slip-only 臂在权重 1/10 下均能显著降低壁面速度且保持
  velocity R²；旧 F1 的 `0.1/0.1` 失败主要由 continuity 相对过强、
  no-slip 相对过弱造成。`1e-3/10` 联合臂则使最低 velocity R²
  降至约 `0.945`，说明连续性权重不能简单放大。
- **证据**：训练 Jobs `11049/11051/11053/11055/11057/11059/11061/11063`
  与汇总 Job `11064` 均 `COMPLETED (0:0)`；机器可读 Gate 位于
  `outputs/wss_pinn/matrices/f1_diagnostics_20260730/report.json`，
  SHA256 `3b1a51c3…00761f72`。选中 config source/resolved SHA256 为
  `ba304ff0…8c24fa` / `62af55d7…f2de6a`。
- **边界**：这是三病例训练期 pilot 的内部权重诊断，不是全库泛化结果；
  F1 仍只允许 continuity + no-slip，WSS-physics/momentum 均为 0。

### D-012｜2026-07-30｜全量 PINN 数据审查发现 1 个不可用测试病例

- **决定**：在生成全量 sidecar 和提交训练前，对冻结
  `train138/test36` 的 174 例逐例检查 bundle、原始峰值体域场、坐标对齐、
  ID、有限性、速度尺度、采样容量、角色计数和 duplicate group。
- **结果**：复审 Job `11067` 完整写出报告后按 Gate 合同以
  `FAILED (2:0)` 退出，报告得到 `173/174` pass（`99.43%`）。
  170 例为直接行对齐，4 例通过冻结坐标变换后的唯一 KD-tree 子集映射；
  170 例使用 `cellnumber`，4 例为历史 `nodenumber`。split 计数
  `train=138/test=36` 正确，无 duplicate-group 泄漏。
- **唯一失败**：`AAA/ruputer/SHI_YUN_XI`（role=`test`）的峰值与抽查
  多个时间步体域 `u/v/w` 全零；坐标、压力和壁面 WSS 本身有效，但该病例
  不能用于 PINN 速度监督、continuity 或 no-slip 评估。
- **失败保留**：首轮 Job `11047` 因审查器只接受等行数和 `cellnumber`
  误报 7 例，证据未删除；修复历史格式兼容和空间子集对齐后重跑，才收敛到
  上述 1 个真实失败。依赖旧审查的 sidecar/audit Jobs `11065/11066`，
  以及依赖复审但会包含该坏病例的 `11068/11069`，均在启动前取消。
- **当前门禁**：未生成全量 sidecar、未提交正式训练。推荐只从 PINN 专用
  test 移除 `SHI_YUN_XI`，冻结为 `train138/test35`（173 例），不从 train
  补位；该动作改变冻结数据划分，必须经用户确认。
- **证据**：
  `outputs/wss_pinn/audits/full_data_train138_test36_20260730_v2/report.json`
  （SHA256 `02b567a4…79fffc9`）；原 split SHA256
  `d16fc497…bdd8f1`。

### D-013｜2026-07-30｜冻结 full-data 配对提交合同

- **决定**：full F0-UP 作为 173 例 data-only control 先独立训练；三病例
  F0-U 只通过 `scientific_gate_control_run_dir` 提供阶段能力解锁，不写入
  full F0-UP 的配对评估。full F1 的 `control_run_dir` 必须指向已完成的
  full F0-UP。
- **提交门禁**：source audit 必须与 config 的 split SHA256 一致；
  sidecar audit 必须同时与 split 和 aggregate manifest SHA256 一致；
  full F1 还必须与 full F0-UP 的 data、sampling、model、训练协议和共享
  loss 完全一致，且 continuity/no-slip 权重等于诊断矩阵选中臂。
- **实现**：
  `wss_pinn/tools/prepare_full_training_configs.py` 在三个报告均 pass 后一次
  生成严格配对的 `f0up_full.json`、`f1_full.json` 和哈希清单；确认 split
  前没有预生成配置。
- **验证**：23 项单测通过；选中 pilot 配置的 launcher dry-run 仍显示
  control evaluation 与严格配对合同全部 pass。

### D-014｜2026-07-30｜冻结 PINN 专用 test35 并提交 full F0-UP

- **用户授权边界**：只允许修改 PINN 路线 split，不得改变 baseline_wss
  原 split。已从 PINN 专用 test 仅移除
  `AAA/ruputer/SHI_YUN_XI`，train138 列表逐项不变且不补位。
- **split**：
  `wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_train138_test35_exclude_SHI_YUN_XI_v1.json`
  / SHA256 `964d7021…9361f2b`。baseline 原
  `training/splits/split_AG_AAA_ILO_q2v_pool2025_train138_test36.json`
  仍为 SHA256 `d16fc497…bdd8f1`。
- **全量数据 Gate**：source audit Job `11070` 为 `173/173 pass`；
  sidecar build/audit Jobs `11071→11072` 为 `173/173 pass`。
  aggregate manifest SHA256 `81a32066…43b5e14`；source/sidecar report
  SHA256 为 `14a732ca…5eda3` / `979f5793…8719c`。
- **配对配置**：full F0-UP source/resolved SHA256
  `d27c215c…12e073` / `85136a44…fe606`；full F1 source/resolved SHA256
  `c41cf992…11273` / `110a3102…b06484`。两者仅在 F1 的
  `continuity=1e-4`、`no-slip=10` 有计划差异；WSS-physics/momentum 均为 0。
- **提交事实**：full F0-UP GPU preflight Job `11073`
  `COMPLETED (0:0)`；train/eval Job `11074` 已启动并持续写出有限 loss。
  full F1 launcher dry-run 为 code-ready，但因 full F0-UP
  `evaluation_best.json` 尚未产生而 `would_submit=false`，没有越过 Gate。

### D-015｜2026-07-30｜full F0-UP 完成但未通过全量能力 Gate

- **程序事实**：GPU preflight Job `11073` 与 train/eval Job `11074`
  均 `COMPLETED (0:0)`；正式训练和 best/last 评估总用时 `00:16:15`，
  `train_11074.err` 为空，最后训练 epoch 为 `19999`。best/last checkpoint
  SHA256 分别为 `19b0fb46…4ab316` / `cccb3d2e…542cb`。
- **best 结果**：全体 case-balanced WSS 为
  `R²=0.4451`、`MAE=1.7527 Pa`、high-WSS nRMSE `0.4745`、
  top10 IoU `0.4869`。train138 的 mean/min velocity R² 为
  `0.6568/0.4844`，test35 为 `0.1800/-0.1372`；两侧均无病例达到
  aggregate velocity `R²≥0.95`。train/test pressure mean R² 为
  `0.9715/0.5662`；test WSS mean R² 为 `-0.8743`。
- **last 稳健性**：结论不变。train/test velocity mean R² 为
  `0.6591/0.1770`，仍为 `0/138`、`0/35` 病例达到 0.95；test pressure
  mean R² `0.5694`、test WSS mean R² `-0.7637`。
- **科学 Gate**：严格速度能力 Gate 明确失败，压力与泛化护栏也未通过；
  full F0-UP 判为 `completed / audited / No-Go`。机器可读 Gate 报告为
  `outputs/wss_pinn/audits/full_f0up_to_f1_gate_train138_test35_exclude_shi_v1_20260730/report.json`
  / SHA256 `daccfd1d…75cfed`，其中配对合同、source/sidecar/matrix
  绑定均通过，但 `velocity_r2_ge_0_95=false`、
  `pressure_r2_ge_0_95=false`，总 Gate 为 false。
- **解释边界**：本 run 没有打开 continuity/no-slip，因此不能用于判断物理
  loss。其 20000 step 每步只抽 3/138 个训练病例，单病例期望曝光约
  `20000×3/138≈435` 次，仅为三病例 pilot 每病例 20000 次曝光的约
  `1/46`；快速结束来自固定 step 预算，而不是完整病例训练已经充分收敛。
- **影响**：full F1 虽已 code-ready，仍不得提交；未产生 full-data F1
  训练或结果。下一步如继续，应先为 full F0-UP 重新冻结按病例曝光量或续训
  预算，再重新通过配对 Gate，不能把本次 No-Go 写成 F1 的物理结论。

## 8. 当前运行登记

### PINN-F0UP-full-train138-test35-exclude-shi-v1-s1234-20260730

- 日期：2026-07-30
- 状态：completed / audited / No-Go
- 角色：173 例 full-data F0-UP data-only 配对 control
- Split / SHA256：PINN 专用 `train138/test35` /
  `964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b`
- baseline split：只读、未修改；SHA256
  `d16fc4978b82661e45e7863b19ae5ae5407528f7000c9aeeb9be01b904bdd8f1`
- Sampling manifest / SHA256：
  `data_wss_pinn/full_train138_test35_exclude_shi_v1/sampling_manifest.json` /
  `81a3206664e06436ede107d70ffe6eb532e02ab7ed3bae932067e59eb43b5e14`
- Config / SHA256：
  `wss_pinn/configs/full_train138_test35_exclude_shi_v1_20260730/f0up_full.json` /
  `d27c215c8d3e6c7771e288faecc20cb45b285b7079200ae08efc4a210412e073`
- Resolved config SHA256：`85136a448d03ee765b50e06906e24aee9e9f2a06468c7d4d462f93bc448fe606`
- Job ID：GPU preflight `11073`、train/eval `11074` 均
  `COMPLETED (0:0)`；train/eval 用时 `00:16:15`
- 输出目录：
  `outputs/wss_pinn/runs/PINN-F0UP-full-train138-test35-exclude-shi-v1-s1234-20260730`
- Checkpoint SHA256：best `19b0fb460dfb10a3cd8ad28d7b4d0890bb34a8e4133bfd14af8ef470fd4ab316`；
  last `cccb3d2e11431bfb77783a6f899f5452293cecdb309c3904b2e0a248096542cb`
- 结果：best 全体 WSS `R²=0.4451`、train/test velocity mean R²
  `0.6568/0.1800`；last 对应为 `0.6591/0.1770`
- 恢复：保留 `checkpoints/last.pt`，复制配置到新 experiment/run_dir，
  将 `train.resume` 指向该 checkpoint，再运行统一提交器；不得覆盖当前 run。
- Gate：fail；报告
  `outputs/wss_pinn/audits/full_f0up_to_f1_gate_train138_test35_exclude_shi_v1_20260730/report.json`
  / SHA256 `daccfd1df0539b533011734f744befd4d747bdb2271f7cf6d0591dc6be75cfed`
- 决策：full F1 blocked / not submitted。

### PINN-F1D 八臂内部诊断矩阵

共同合同：3-case overfit pilot、seed 1234、20000 epoch、同一
F0-UP control；`direct_wss/velocity/pressure` 权重为 `1/10/1`，
WSS-physics/momentum 均为 0。下表的 ratio 均相对同 checkpoint 的 F0-UP；
只有 continuity 与 no-slip 同时非零的臂才有资格成为 F1 候选。

| 实验 | λcont | λnoslip | best cont/ns | best min velocity R² | last cont/ns | last min velocity R² | 双 checkpoint Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `C0-N1` | 0 | 1 | `1.029/0.237` | `0.9863` | `1.015/0.259` | `0.9868` | 非联合诊断 |
| `C0-N10` | 0 | 10 | `1.050/0.081` | `0.9856` | `1.059/0.084` | `0.9821` | 非联合诊断 |
| `C1e3-N0` | 1e-3 | 0 | `0.030/0.863` | `0.9581` | `0.033/0.858` | `0.9591` | 非联合诊断；last WSS 护栏失败 |
| `C1e3-N1` | 1e-3 | 1 | `0.033/0.428` | `0.9578` | `0.039/0.429` | `0.9573` | fail：best WSS 护栏 |
| `C1e3-N10` | 1e-3 | 10 | `0.034/0.189` | `0.9457` | `0.037/0.199` | `0.9444` | fail：velocity |
| `C1e4-N0` | 1e-4 | 0 | `0.062/0.859` | `0.9777` | `0.063/0.843` | `0.9755` | 非联合诊断 |
| `C1e4-N1` | 1e-4 | 1 | `0.069/0.308` | `0.9752` | `0.063/0.306` | `0.9724` | fail：WSS 护栏 |
| `C1e4-N10` | 1e-4 | 10 | `0.077/0.123` | `0.9689` | `0.072/0.147` | `0.9697` | **pass / selected** |

- Config：`wss_pinn/configs/f1_diagnostics_20260730/`
- 汇总：`outputs/wss_pinn/matrices/f1_diagnostics_20260730/{report.json,matrix.csv}`
- Slurm：训练 Jobs `11049,11051,11053,11055,11057,11059,11061,11063`；
  汇总 Job `11064`
- 状态：completed / audited；正式全量训练仍 blocked on split。

### PINN-F0U-overfit3-s1234-20260730

- 日期：2026-07-30
- Exp ID：`PINN-F0U`
- 状态：completed / No-Go
- Control：P1 sidecar / W0 direct-WSS task
- 唯一改动：新增平滑 \(u,v,w\) data-only 辅助场；pressure/physics/momentum 全关
- Split / reporting role：3-case overfit pilot / `reused_development_screen`
- Seed：1234
- Data manifest / SHA256：`data_wss_pinn/pilot_v1/sampling_manifest.json` /
  `c5bdd7b7ec89b43b16d8a1eccd664caf9490b39eb4aca6e1c4e8ec05177678f8`
- Config / SHA256：`wss_pinn/configs/f0u_pilot.json` /
  `561991e446e1f666fd9c28a1026654b3c66b99ba4ff920505a913681884fecdd`
- Job ID：preflight `11037`；train/eval `11038`；均 `COMPLETED (0:0)`
- 输出目录：`outputs/wss_pinn/runs/PINN-F0U-overfit3-s1234-20260730`
- 结果：best velocity R² AG/AAA/ILO = `0.654/0.586/0.567`；
  case-balanced WSS R² `0.859`
- Gate 与结论：速度 R² 未达到 0.95，No-Go；不得提交 F0-UP。
- Notes：CUDA 报告未设置 `CUBLAS_WORKSPACE_CONFIG` 的确定性警告；无 NaN/Inf、
  路径、导入或 checkpoint 错误。

### PINN-F0U-overfit3-v2-s1234-20260730

- 日期：2026-07-30
- Exp ID：`PINN-F0U`
- 状态：completed / No-Go
- Control：上方 v1 失败诊断；不冒充与 v1 的单变量消融
- 唯一改动：split WSS/field trunk、field hidden 256×5、8 Fourier bands、
  velocity data weight 10、8000 epochs
- Split / reporting role：同一 3-case overfit pilot / `reused_development_screen`
- Seed：1234
- Data manifest / SHA256：同上，`c5bdd7b7…77678f8`
- Config / SHA256：`wss_pinn/configs/f0u_pilot_v2.json` /
  `08157b19d47d8bc28bd168f08d055df764b3b71195f0db2ca5a95ca72bf4bdb6`
- Job ID：preflight `11039`、train/eval `11040`，均 `COMPLETED (0:0)`
- 输出目录：`outputs/wss_pinn/runs/PINN-F0U-overfit3-v2-s1234-20260730`
- 结果：best case-balanced WSS R² `0.9906`；聚合 velocity R²
  AG/AAA/ILO = `0.9638/0.9478/0.9532`。逐分量最低项分别为
  AG `u=0.9290`、AAA `speed=0.9280`、ILO `w=0.9250`。
- Gate：严格逐病例、逐分量和速度模长均不低于 0.95；未通过，No-Go。

### PINN-F0U-overfit3-v2ext-s1234-20260730

- 日期：2026-07-30
- Exp ID：`PINN-F0U`
- 状态：completed / Go
- Control：`PINN-F0U-overfit3-v2-s1234-20260730`
- 唯一改动：从 v2 `last.pt` 显式恢复，增加 12000 个 epoch；模型、数据、
  loss、优化器和 seed 不变
- Split / reporting role：同一 3-case overfit pilot / `reused_development_screen`
- Seed：1234
- Data manifest / SHA256：同上，`c5bdd7b7…77678f8`
- Config / SHA256：`wss_pinn/configs/f0u_pilot_v2_extended.json` /
  `169fb4eb1e444a0f3091b66b524a7807e5a7e44ba0aeb422286dee579c74625c`
- Resolved config SHA256：
  `1740356024c6b100cd7dad795440b5aed1b2e6beae20d3215908d8544d2649d9`
- Parent checkpoint / SHA256：v2 `checkpoints/last.pt` /
  `dcfee8d51a3edd603a5623afefb44fae483eed925903037495144bbff7c0bbad`
- Job ID：preflight `11041`、train/eval `11042`，均 `COMPLETED (0:0)`
- 输出目录：`outputs/wss_pinn/runs/PINN-F0U-overfit3-v2ext-s1234-20260730`
- 监控：`/public/slurm/bin/squeue -j 11042`；
  `/public/slurm/bin/sacct -j 11041,11042 --format=JobID,State,Elapsed,ExitCode`；
  `tail -f outputs/wss_pinn/runs/PINN-F0U-overfit3-v2ext-s1234-20260730/slurm/train_11042.out`
- 恢复：保留当前 `last.pt`，另建新 config/run_dir，并把 `train.resume`
  指向该 checkpoint 后重新运行统一提交器。
- 结果：best 的三病例聚合 velocity R² 为
  AG/AAA/ILO `0.9961/0.9948/0.9950`；逐病例全部 `u/v/w/speed`
  的最小 R² 为 `0.9888`。case-balanced WSS R² `0.99934`、
  high-WSS nRMSE `0.02179`、top10 IoU `0.9763`。last 的速度门禁最小
  R² 为 `0.9896`，结论不依赖 best/last 选择。
- Gate：逐病例聚合 velocity、`u/v/w` 和 speed R² 均不低于 0.95，
  pass / Go；允许提交 F0-UP。

### PINN-F0UP-overfit3-v2ext-s1234-20260730

- 日期：2026-07-30
- Exp ID：`PINN-F0UP`
- 状态：completed / Go
- Control：`PINN-F0U-overfit3-v2ext-s1234-20260730`
- 唯一改动：启用 pressure decoder 和 gauge-pressure data loss；模型容量、
  数据、velocity/WSS loss、优化器、seed 和 20000 总 epoch 保持配对
- Split / reporting role：同一 3-case overfit pilot / `reused_development_screen`
- Seed：1234
- Data manifest / SHA256：同上，`c5bdd7b7…77678f8`
- Config / SHA256：`wss_pinn/configs/f0up_pilot_v2.json` /
  `f426414387de7a6ecfb1afc7daa9ce789d2f276f5d4c07187effd6ca235101bb`
- Resolved config SHA256：
  `d58fad649e5a731d7f8c37d8e964daf5f819e5c2c0908964ebc2d65c4eace54d`
- Job ID：preflight `11043`、train/eval `11044`，均 `COMPLETED (0:0)`
- 运行时间：2026-07-30 01:48:37–02:03:20（Asia/Shanghai），`00:14:43`
- 输出目录：`outputs/wss_pinn/runs/PINN-F0UP-overfit3-v2ext-s1234-20260730`
- Checkpoint / SHA256：best
  `a332af6b9eba4efaeca4182777dc226e2a77f16c5254cc18816c53e91d2a3124`；
  last `fb3829df3f22a2aaaa98671263b69c4894cda6390dc6b02d68f5a16b90275e49`
- 结果（best）：三病例 velocity 聚合 R²
  AG/AAA/ILO `0.9957/0.9939/0.9954`；所有 velocity 聚合与
  `u/v/w/speed` 的最低 R² `0.9872`。gauge-pressure R²
  `0.9987/0.9994/0.9998`。case-balanced WSS R² `0.99974`、
  MAE `0.05997 Pa`、high-WSS nRMSE `0.01340`、top10 IoU `0.98282`。
- best 相对 F0-U v2ext：case-balanced WSS
  `ΔR²=+0.00040`、`ΔMAE=-0.01367 Pa`、
  `Δhigh-WSS nRMSE=-0.00839`、`ΔIoU=+0.00652`；
  病例 velocity `ΔR²=-0.00033/-0.00093/+0.00045`，未破坏速度能力。
- best/last 敏感性：last 的最低 velocity / pressure R² 为
  `0.9876/0.9988`，仍通过；last case-balanced WSS R² `0.99942`，
  结论不依赖 checkpoint 选择。
- residual 基线：F0-UP 没有 physics loss。best 的 mean
  continuity RMS 从 F0-U 的 `66.99` 上升到 `70.15`，mean no-slip RMS
  从 `0.1223` 上升到 `0.1245`；不能把 F0-UP 写成物理 residual 改善。
- Gate：逐病例 velocity 聚合、`u/v/w/speed` 与 gauge-pressure R²
  全部不低于 0.95，pass / Go；F1 已解锁并于后续独立动作提交。
- Notes：本 run 与 F0-U v2ext 的总训练步、模型容量、数据、velocity/WSS
  loss 和 seed 对齐，但 F0-U 是显式续训、F0-UP 是从头训练并新增 pressure
  输出，因此这里只把它用于能力 Gate，不把二者差异解释为压力头的严格因果效应。
  CUDA 日志只有未设置 `CUBLAS_WORKSPACE_CONFIG` 的确定性警告，无
  NaN/Inf、导入、路径、checkpoint 或评估失败。

### PINN-F1-cont-noslip-overfit3-v2ext-s1234-20260730

- 日期：2026-07-30 11:32:02 提交；11:50:17 完成（Asia/Shanghai）
- Exp ID：`PINN-F1`
- 状态：completed / audited / No-Go
- 假设：在保持 F0-UP 数据监督与模型协议不变时，continuity + no-slip 能降低
  对应 residual，且不越过 velocity、pressure 和 `WSS_direct` 退化护栏
- Control：`PINN-F0UP-overfit3-v2ext-s1234-20260730`
- 唯一改动：`continuity_weight: 0→0.1`、
  `no_slip_weight: 0→0.1`
- 关闭项：`wss_physics_weight=0`、`momentum_weight=0`
- Split / reporting role：同一 3-case overfit pilot /
  `reused_development_screen`
- Seed：1234
- Data manifest / SHA256：`data_wss_pinn/pilot_v1/sampling_manifest.json` /
  `c5bdd7b7ec89b43b16d8a1eccd664caf9490b39eb4aca6e1c4e8ec05177678f8`
- Config source / SHA256：`wss_pinn/configs/f1_pilot_v2.json` /
  `85ecb2c14748018c086dff9349b8de781e0657e9280c7c21428253cbde49ff7a`
- Resolved config SHA256：
  `8eae38ddf26824314ad0c1b10a283706dcf42722b75eb61a5a65b90e505ad1b1`
- Code commit / dirty：`ea24602ee2f57e6f79e916b256e04038bbb2a4f1` / `dirty=true`
- Parent checkpoint：不恢复 F0-UP checkpoint；与 F0-UP 一样从 seed 1234
  初始化，作严格配置配对
- Job ID：GPU preflight `11045`、train/eval `11046`，均
  `COMPLETED (0:0)`
- 运行时间：preflight `00:01:27`；train/eval `00:16:48`
- 输出目录：
  `outputs/wss_pinn/runs/PINN-F1-cont-noslip-overfit3-v2ext-s1234-20260730`
- 提交记录：上述输出目录下 `submission.json`
- Checkpoint / SHA256：best（epoch `19736`）
  `f676f50343f235f91a5f9892b535e7262dfd3dda37b6834c38e3c973a005c917`；
  last（epoch `19999`）
  `6b94a312656627155effecfb0520fea65f7fb9af1077038b0212e5eb63127b29`
- 完整性：训练、best/last 评估均完成；日志仅有未设置
  `CUBLAS_WORKSPACE_CONFIG` 的 CuBLAS 确定性警告，无 NaN/Inf、路径、
  checkpoint 或评估错误。
- 结果（best）：mean continuity RMS `70.1526→0.3359`（`-99.52%`）；
  mean no-slip RMS `0.12447→0.13396`（`+7.63%`）；mean velocity R²
  `0.99502→0.82491`，严格最低 velocity R² `0.69678`；mean pressure R²
  `0.99824`。case-balanced WSS R² `0.99936`、MAE `0.08443 Pa`、
  high-WSS nRMSE `0.02186`、top10 IoU `0.97370`。
- best/last 敏感性：last 的 mean continuity/no-slip RMS 为
  `0.43589/0.13266`，mean velocity R² `0.82696`；结论不依赖 checkpoint。
- 只读复算：CPU 重算 best/last 各 126 个数值字段，最大绝对差
  `1.88e-5/1.43e-5`，与 CUDA 落盘评估一致。
- 训练为何快：这是 3 病例、单峰值时相的过拟合 pilot，不是全库完整
  Navier–Stokes PINN。模型只有 `613,829` 个参数；每个 optimizer step
  只处理三病例各 `2048 wall + 2048 near-wall + 2048 core` 点，数据预载内存。
  F1 只计算一阶 continuity 导数和简单 no-slip MSE，没有 momentum 的高阶导数、
  非牛顿黏度梯度、非定常多时相或 WSS-physics。RTX 4090 上 20000 step 加
  best/last 评估耗时 `16:48`，约 `50 ms/step`；相比 F0-UP 的 `14:43`
  只慢 `2:05`（约 `14%`），与新增一阶自动微分的规模一致。
- 监控：`/public/slurm/bin/squeue -j 11045,11046`；
  `/public/slurm/bin/sacct -j 11045,11046 --format=JobID,State,Elapsed,ExitCode`；
  `tail -f outputs/wss_pinn/runs/PINN-F1-cont-noslip-overfit3-v2ext-s1234-20260730/slurm/train_11046.out`
- 恢复：保留当前 `checkpoints/last.pt`，另建新 config/run_dir，把
  `train.resume` 显式指向该 checkpoint，再运行
  `python wss_pinn/cluster/submit_experiment.py --config <new-config>`。
- Gate 与结论：No-Go。continuity 目标通过，但 no-slip 未下降且 velocity
  护栏严重失守；F2 保持 blocked。若继续，应先做 F1 内部的单项/权重/尺度或
  curriculum 诊断，不得把本 run 写成“物理场改善”。
