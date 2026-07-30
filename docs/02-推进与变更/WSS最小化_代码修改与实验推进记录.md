# WSS 最小化路线代码修改与实验推进记录

> 用途：单独记录 `pipeline_wss_min/` 这条 WSS-only 最小化数据线的代码、坐标 QA、图件和实验推进。
> V3P / 训练主线 / 通用代码修改记录见：[代码修改与实验推进记录](代码修改与实验推进记录.md)。
> 当前执行入口：[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md) / [第六轮总入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md) / [横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md) / [WSS 精度突破](WSS最小化_第六轮_WSS精度突破计划与执行.md) / [BC/速度条件路线](WSS最小化_第六轮_边界条件与速度路线.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)。

## 2026-07-30｜WSS-PINN full F0-UP 完训审计并阻断 full F1

**本次主要修改**：核验 full F0-UP Slurm、best/last checkpoint、完整
train/test 评估与配对科学 Gate，并将工程完成状态和科学 No-Go 分开回填。
复核 baseline 与 PINN 专用 split 哈希，未改动任何 baseline 数据划分。

**对应代码/文档**：
`outputs/wss_pinn/runs/PINN-F0UP-full-train138-test35-exclude-shi-v1-s1234-20260730/`、
`outputs/wss_pinn/audits/full_f0up_to_f1_gate_train138_test35_exclude_shi_v1_20260730/report.json`、
[WSS-PINN 总入口](WSS_PINN/README.md)、
[阶梯矩阵](WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md)、
`wss_pinn/README.md`。

**推进到实验步骤**：Jobs `11073→11074` 均 `COMPLETED (0:0)`；
train/eval 用时 `00:16:15`。best 的 train/test velocity mean R² 为
`0.6568/0.1800`，last 为 `0.6591/0.1770`，两侧达到 aggregate
velocity `R²≥0.95` 的病例数均为 0。Gate 报告 SHA256
`daccfd1d…75cfed`。

**当前状态判断**：full F0-UP 为 completed / audited / No-Go；full F1
虽 code-ready，但因配对 control Gate 为 false 而 blocked / not submitted。
20000 step 在 full train138 下每病例期望仅曝光约 435 次，约为三病例 pilot
的 `1/46`，因此快速结束反映固定 step 预算不足，不代表全量学习充分。F1
仍只允许 continuity + no-slip，WSS-physics/momentum 保持 0。

## 2026-07-30｜PINN 专用 test35 冻结、全量 sidecar 通过并提交 full F0-UP

**本次主要修改**：按用户授权仅在 `wss_pinn/configs/splits/` 派生
`train138/test35`，只移除体域速度全零的 `AAA/ruputer/SHI_YUN_XI`，
train 列表不变且不补位；新增可复核的 PINN split 派生工具与单测。完成
173 例 source audit、sidecar 构建与完整性审查，生成严格配对的 full
F0-UP/F1 配置并提交 full F0-UP。

**对应代码/文档**：`wss_pinn/tools/derive_pinn_split.py`、
`wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_train138_test35_exclude_SHI_YUN_XI_v1.json`、
`wss_pinn/configs/p1_full_train138_test35_exclude_shi_v1.json`、
`wss_pinn/configs/full_train138_test35_exclude_shi_v1_20260730/`、
`data_wss_pinn/full_train138_test35_exclude_shi_v1/`、
`outputs/wss_pinn/audits/{full_data_train138_test35_exclude_shi_v1_20260730,full_sidecars_train138_test35_exclude_shi_v1_20260730}/`、
[WSS-PINN 总入口](WSS_PINN/README.md)与
[阶梯矩阵](WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md)。

**推进到实验步骤**：baseline 原 split SHA256 保持
`d16fc497…bdd8f1`；PINN 专用 split SHA256 `964d7021…9361f2b`。
source Job `11070`、sidecar build/audit Jobs `11071/11072` 均 completed，
173/173 pass；manifest SHA256 `81a32066…43b5e14`。full F0-UP preflight
Job `11073` completed，train/eval Job `11074` running 并持续写出有限 loss。

**当前状态判断**：完整可用 PINN 数据集已 frozen / audited。full F0-UP
正式训练 running，不写成 completed；full F1 config 已 code-ready，但
`evaluation_best.json` 尚未产生，所以 launcher 正确返回
`would_submit=false`，不得越过配对科学 Gate。F1 仍只计划开启
`continuity=1e-4`、`no-slip=10`，WSS-physics/momentum 为 0。

## 2026-07-30｜WSS-PINN F1 内部八臂 Gate 通过；全量数据审查停在 split 门禁

**本次主要修改**：完成 F1 continuity-only、no-slip-only 与 4 个联合权重臂，
并以 best/last 双 checkpoint 统一 Gate 汇总；新增全量源数据、sidecar 完整性
审查及 Slurm 入口，支持历史 `nodenumber` 和经冻结坐标变换后的唯一空间子集
映射。对冻结 `train138/test36` 做 174 例深审计，不改旧 bundle 或既有 run。
另将 full F0-UP 的阶段解锁证据与 full F1 的配对 control 分离，并强制审计报告
绑定 split/manifest SHA256 和 F0-UP/F1 严格配置配对。

**对应代码/文档**：`wss_pinn/{config.py,train.py,evaluate.py}`、
`wss_pinn/data/{raw_io.py,sidecar.py,dataset.py}`、
`wss_pinn/tools/{analyze_f1_matrix.py,audit_full_dataset.py,audit_sidecar_dataset.py,preflight.py}`、
`wss_pinn/tools/prepare_full_training_configs.py`、
`wss_pinn/cluster/`、`wss_pinn/configs/f1_diagnostics_20260730/`；
`outputs/wss_pinn/matrices/f1_diagnostics_20260730/report.json`；
`outputs/wss_pinn/audits/full_data_train138_test36_20260730_v2/report.json`；
[WSS-PINN 总入口](WSS_PINN/README.md)与
[阶梯矩阵](WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md)。

**推进到实验步骤**：八个训练 Jobs
`11049/11051/11053/11055/11057/11059/11061/11063` 与汇总 Job `11064`
全部 `COMPLETED (0:0)`。唯一双 checkpoint 通过的联合臂为
`continuity=1e-4`、`no-slip=10`；best/last continuity ratio
`0.0769/0.0715`、no-slip ratio `0.1232/0.1465`，最低 velocity R²
`0.9689/0.9697`。全量复审 Job `11067` 完整写出报告后按 Gate 合同以
`FAILED (2:0)` 退出，结果为 `173/174` pass。

**当前状态判断**：F1 pilot diagnostics completed / selected；full-data
sidecar 与正式训练 blocked on split，未提交。唯一失败
`AAA/ruputer/SHI_YUN_XI` 位于 test，其多个时相体域速度全零，不能用于
PINN 速度监督/物理评估。推荐从 PINN 专用 test 只移除该例并冻结
`train138/test35`，不从 train 补位；因这会改变冻结数据划分，等待用户确认。
F2 继续 blocked，WSS-physics/momentum 未开启。正式链条已 code-ready，
22/22 单测通过；确认 split 后仍必须依次通过 source audit、sidecar audit、
full F0-UP 结果 Gate，才能提交 full F1。

## 2026-07-30｜WSS-PINN F1 完训审计：continuity 有效但整体 No-Go

**本次主要修改**：只读核验 F1 Jobs `11045→11046` 的 Slurm accounting、
best/last checkpoint、训练摘要和配对评估；以 CPU 独立重算 best/last 各
126 个数值字段。按 F1 预注册 Gate 回填路线矩阵、入口状态、总纲与进度日志，
并解释 20000 step 仅耗时约 17 分钟的计算规模。

**对应代码/文档**：
`outputs/wss_pinn/runs/PINN-F1-cont-noslip-overfit3-v2ext-s1234-20260730/{evaluation_best.json,evaluation_last.json,training_summary.json,checkpoints/,slurm/}`；
[WSS-PINN 总入口](WSS_PINN/README.md)；
[阶梯矩阵 §7–§8](WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md)；
[历史执行合同](WSS_PINN/WSS_PINN_下一智能体目标提示词_推进至F1.md)；
[体域物理约束路线](WSS最小化_体域物理约束与PINN训练路线_2026-07-29.md)；
`wss_pinn/README.md`；根 `README.md`、`docs/README.md`、
`docs/实验设计总纲.md`。

**推进到实验步骤**：Jobs `11045/11046` 均 `COMPLETED (0:0)`；
train/eval 用时 `00:16:48`。best 相对 F0-UP 的 mean continuity RMS
`70.1526→0.3359`（`-99.52%`），但 mean no-slip RMS
`0.12447→0.13396`（`+7.63%`），mean velocity R²
`0.99502→0.82491`，严格最低 velocity R² `0.69678`。pressure mean R²
仍为 `0.99824`；WSS `ΔR²=-0.00038`、`ΔMAE=+0.02446 Pa`、
`Δhigh-WSS nRMSE=+0.00846`。last 给出相同方向，CPU 重算最大绝对差
`1.88e-5/1.43e-5`。

**当前状态判断**：F1 completed / audited / No-Go，F2 blocked。continuity
确实被压低，但 no-slip 未改善且速度场数据护栏严重失守，不能写成“物理场整体
改善”。训练快并非少跑：这是 3 病例、单时相、`613,829` 参数的过拟合 pilot，
每 step 仅处理 `18,432` 个查询点，只含一阶 continuity 与简单 no-slip，
没有 momentum 高阶导数、非牛顿黏度梯度、多时相或 WSS-physics；RTX 4090 上
约 `50 ms/step` 合理。若继续应先做 F1 单项/权重/尺度或 curriculum 诊断，
不得直接提交 F2。

## 2026-07-30｜WSS-PINN F1 配置冻结并提交训练

**本次主要修改**：复核并冻结已与 F0-UP 配对的
`wss_pinn/configs/f1_pilot_v2.json`，不另建重复配置。该配置相对 F0-UP
只把 continuity/no-slip 权重从 0 调为 0.1，WSS-physics/momentum 保持 0。
完成 compile、12 项单测、CPU 单步 dry-run、launcher Gate dry-run 后，通过统一
Slurm 提交器正式提交 GPU 预检与训练/评估依赖作业。

**对应代码/文档**：`wss_pinn/configs/f1_pilot_v2.json`；
`outputs/wss_pinn/runs/PINN-F1-cont-noslip-overfit3-v2ext-s1234-20260730/`；
[WSS-PINN 总入口](WSS_PINN/README.md)；[阶梯矩阵 §7–§8](WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md)；
`wss_pinn/README.md`；根 `README.md`、`docs/README.md`、
`docs/实验设计总纲.md`。

**推进到实验步骤**：配置 source/resolved SHA256 分别为
`85ecb2c1…49ff7a` / `8eae38dd…5ad1b1`，sampling manifest SHA256 为
`c5bdd7b7…77678f8`。GPU preflight Job `11045` 已
`COMPLETED (0:0)`，通过 12 项测试、F0-UP 科学 Gate 和 CUDA 单步 dry-run；
train/eval Job `11046` 已在 GPU 节点运行并持续写出
`training_progress.jsonl`、best/last checkpoint 和有限 loss。

**当前状态判断**：F1 `running`，不是 completed；当前日志只有未设置
`CUBLAS_WORKSPACE_CONFIG` 的 CuBLAS 确定性警告，无 NaN/Inf、路径或
checkpoint 错误。必须等待 best/last 评估并与 F0-UP 配对检查 continuity、
wall speed、velocity、pressure 与 WSS 护栏后再判 Gate；此前不得提交 F2。
监控与恢复命令已写入阶梯矩阵和 run 的 `submission.json`。

## 2026-07-30｜WSS-PINN F0-UP 完成、能力 Gate 通过并解锁 F1

**本次主要修改**：只读核验 F0-UP Jobs `11043→11044` 的 Slurm accounting、best/last checkpoint、训练摘要、机器可读评估与父对照 F0-U v2ext；按逐病例 velocity 聚合、`u/v/w/speed`、gauge-pressure 和 WSS 护栏回填路线矩阵、入口状态、总纲和进度日志。另以 CPU 直接加载 best checkpoint 重算评估，不改写 run；126 个数值字段相对落盘 JSON 的最大绝对差 `2.29e-5`。

**对应代码/文档**：`outputs/wss_pinn/runs/PINN-F0UP-overfit3-v2ext-s1234-20260730/{evaluation_best.json,evaluation_last.json,training_summary.json,checkpoints/,slurm/}`；[WSS-PINN 总入口](WSS_PINN/README.md)；[阶梯矩阵 §7–§8](WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md)；`wss_pinn/README.md`；根 `README.md`、`docs/README.md`、`docs/实验设计总纲.md`。

**推进到实验步骤**：F0-UP preflight/train-eval Jobs `11043/11044` 均 `COMPLETED (0:0)`。best 的三病例 velocity 聚合 R² 为 `0.9957/0.9939/0.9954`，全部 velocity 聚合与 `u/v/w/speed` 的最低 R² `0.9872`；gauge-pressure R² `0.9987/0.9994/0.9998`。last 的最低 velocity/pressure R² `0.9876/0.9988`，结论不依赖 checkpoint。F1 launcher dry-run 已复核 `would_submit=true`，但本轮未提交。

**当前状态判断**：F0-UP completed / Go，F1 code-ready / unlocked。该结论只证明三病例训练期速度与 gauge-pressure 可高拟合，不是独立测试泛化证据；F0-UP 未启用 physics loss，best 的 mean continuity/no-slip RMS 相对 F0-U 从 `66.99/0.1223` 变为 `70.15/0.1245`，不能写成物理 residual 改善。F1 仍严格只允许 continuity + no-slip，WSS-physics/momentum 为 0，并需单独审批后提交。

## 2026-07-30｜WSS-PINN P0/P1、F0-U Gate 完成并进入 F0-UP

**本次主要修改**：在独立 `wss_pinn/` 中实现配置驱动的 P0-A/B/C/D、P1 sidecar、统一 F0-U/F0-UP/F1 模型/损失/训练/评估入口、阶段门禁、写路径守卫、12 项单元测试和幂等 Slurm 提交器。真实审计选取 AG/AAA/ILO 各 1 例；生成 5k wall / 8k near-wall / 8k core 三槽 sidecar。F1 配置只启用 continuity + no-slip，WSS-physics 与 momentum 强制为 0。首次 P1 因误读冻结 `int_type`（实际 core=0、near-wall=1）在落盘前失败；修正后重跑。P0-C 首轮邻域差使用近零 divergence 作分母而病态，改用梯度范数归一化后重新跑完整 near-wall/core 诊断。

**对应代码/文档**：`wss_pinn/{config.py,data/,models/,physics/,tools/,tests/,cluster/,train.py,evaluate.py}`；`wss_pinn/configs/{pilot_cases,p0_pilot,p1_pilot,f0u_pilot,f0u_pilot_v2,f0u_pilot_v2_extended,f0up_pilot_v2,f1_pilot_v2}.json`；`data_wss_pinn/pilot_v1/`；`outputs/wss_pinn/{audits,runs}/`；[WSS-PINN 总入口](WSS_PINN/README.md)；[阶梯矩阵](WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md)。

**推进到实验步骤**：P0-A/B/C/D 与 P1 completed。P0-B 三病例最佳 velocity→WSS R² 为 `0.816/0.826/0.801`；P0-C 相对 divergence p95 为 `0.069/0.129/0.088`。P0-D 法向可用，但 exact zone/connectivity 缺失，hard flux/RCR residual blocked。P1 聚合 manifest SHA256 为 `c5bdd7b7ec89b43b16d8a1eccd664caf9490b39eb4aca6e1c4e8ec05177678f8`。F0-U v1 Jobs `11037→11038` 均 completed，速度 R² `0.654/0.586/0.567` 未过 0.95，No-Go。v2 Jobs `11039→11040` 也 completed；case-balanced WSS R² `0.9906`，但逐病例 `u/v/w/speed` 最低项仍为 `0.925–0.929`，严格 Gate No-Go。v2ext 从 v2 `last.pt` 显式恢复，只增加 12000 epoch；Jobs `11041→11042` completed，best/last 的严格速度 Gate 最低 R² 为 `0.9888/0.9896`，Go。F0-UP Jobs `11043→11044` 中 preflight completed、训练 running。F1 v2ext 已通过 CPU/launcher dry-run，但仍由 F0-UP Gate 锁定、未提交。

**当前状态判断**：实现层已覆盖到 F1，执行层按 Gate 推进到 F0-UP running。F0-U v2ext 已证明三病例速度场可严格过拟合，且 case-balanced WSS R² 为 `0.99934`。只有 F0-UP 的逐病例 velocity 各项与 gauge-pressure R² 均过 0.95 后才允许提交 F1。GPU 日志存在未设置 `CUBLAS_WORKSPACE_CONFIG` 的确定性警告，但无 NaN/Inf、导入、CUDA、路径或 checkpoint 错误。

## 2026-07-30｜P0→F1 下一智能体执行提示词与长作业交接合同（No-Run）

**本次主要修改**：新增一份可直接交给下一智能体的执行型目标提示词，把本轮实现边界固定为 P0-A/P0-B/P0-C/P0-D→P1→F0-U→F0-UP→F1；要求实际修改独立路线代码、所有实验只通过配置字段切换、正式训练统一走 Slurm。明确“实现层必须完整做到 F1、执行层按科学 Gate 推进”的双层完成口径，以及长训练只需提交、检查首轮状态/日志并可靠记录，无需原地等待完训。

**对应代码/文档**：[WSS-PINN 下一智能体目标提示词：实现并推进至 F1](WSS_PINN/WSS_PINN_下一智能体目标提示词_推进至F1.md)；[WSS-PINN 独立路线总入口](WSS_PINN/README.md)；`wss_pinn/README.md`。

**推进到实验步骤**：仍为 S0/No-Run；本次只建立执行交接合同，未实现 P0/P1 或模型代码，未生成 sidecar，未提交 Slurm 作业。下一智能体应先完成 P0/P1 pilot，同时把 F0-U/F0-UP/F1 的共用代码、配置、测试和集群入口准备到 code-ready。

**当前状态判断**：提示词已避免“长 F0-U 未完导致 F1 代码也不实现”和“只靠 Slurm `afterok` 越过科学 Gate”两类问题。F2 的 WSS 梯度一致性和 F3 momentum 明确不在本轮范围；若训练尚未完成，只能记录 `submitted/running/pending_review`，不得宣称 F1 实验已完成。

## 2026-07-30｜WSS-PINN 独立目录、阶梯计划与记录合同建立（No-Run）

**本次主要修改**：建立顶层 `wss_pinn/` 专用入口及硬隔离规则，冻结 `data_new/`、`data_wss_min/`、`pipeline_wss_min/`、`training_wss_min/` 为只读上游；PINN 派生数据、结果和执行记录分别固定到 `data_wss_pinn/`、`outputs/wss_pinn/`、`docs/02-推进与变更/WSS_PINN/`。新增 S0→P0→P1→F0-U→F0-UP→F1→F2→F3→C1/C2 阶梯、配对消融、运行落盘字段、状态词和 Go/No-Go 模板。

**对应代码/文档**：`wss_pinn/{AGENTS.md,README.md}`；[WSS-PINN 独立路线总入口](WSS_PINN/README.md)；[WSS-PINN 阶梯实验矩阵与进度跟踪](WSS_PINN/WSS_PINN_阶梯实验矩阵与进度跟踪.md)；[体域物理约束与 PINN 路线](WSS最小化_体域物理约束与PINN训练路线_2026-07-29.md)；根 `README.md`、`docs/README.md`、`docs/实验设计总纲.md` 和 `.gitignore`。

**推进到实验步骤**：完成 S0 目录隔离和预注册；P0-A/P0-B/P0-C/P0-D 均为 planned。未实现数据或训练代码，未生成 sidecar，未提交 GPU/Slurm 作业，未修改旧数据、配置、checkpoint 或 run。

**当前状态判断**：独立目录方案可以保护原实验，且比在 `training_wss_min/` 内直接叠加 PINN 更容易做严格归因和失败回退。下一步只允许先实现 P0-A 体点身份/单位审计，再做 P0-B velocity→WSS Oracle；P0 未通过前，P1–C2 保持 blocked。

## 2026-07-29｜非滑移壁面与体域分槽采样口径冻结（No-Run）

**本次主要修改**：根据用户补充确认，将“刚性壁面”精确冻结为“非滑移刚性壁面”；明确现有 `random5000` 仅继续承担 W0 直接 WSS 锚点的壁面 support，PINN 训练必须另加近壁体点和核心体点，不能用全壁面 5k 点计算体域 Navier–Stokes 残差。

**对应代码/文档**：[体域物理约束与 PINN 训练路线讨论稿 §0.5/§6.1/§10.1](WSS最小化_体域物理约束与PINN训练路线_2026-07-29.md)；体点候选源为只读的 `data_new/**/ascii_in/`，现有 WSS 锚点协议不变。

**推进到实验步骤**：仍为 P0/No-Run，未改代码、数据、配置或作业。后续 sidecar/batch 合同预注册为 wall、near-wall、core 分槽；F0 使用体内 \(u,v,w,p\) 监督，F1/F3 在体域点计算 continuity/momentum，壁面点只承担 no-slip、直接 WSS 和梯度一致性。

**当前状态判断**：体内点是完整 PINN 路线的必要条件，但不要求一次载入全部百万级体点，也不应破坏 `random5000` 基线。下一步仍先做 cell identity、velocity→WSS、CFD residual 与法向/边界可得性四类 P0 审计，再冻结 pilot 的每槽点数。

## 2026-07-29｜非牛顿流变正式确认并冻结 PINN 物理口径（No-Run）

**本次主要修改**：根据用户最终确认，将体域 PINN 路线中的流变状态从“UDF/`.cas.gz` 已证明、待口头确认”升级为正式决策：统一使用 UDF 定义的剪切率相关非牛顿流变；删除“可能本意为常黏度或误挂接”的待定表述。

**对应代码/文档**：[体域物理约束与 PINN 训练路线讨论稿 §0.4/§1.2/§10.2](WSS最小化_体域物理约束与PINN训练路线_2026-07-29.md)；物理参数真源仍为 `data_new/**/udf-inlet4.c` 和对应 `.cas.gz`。

**推进到实验步骤**：仍为 P0/No-Run，未改代码、数据或作业。后续 velocity→WSS Oracle、\(WSS_{\mathrm{phys}}\) 和 momentum residual 统一使用 UDF 的 \(\mu(\dot{\gamma})\)；constant-\(\mu\) 只允许作为命名明确的简化消融。

**当前状态判断**：流变模型不再是阻塞项。第一阶段的目标、BC、数据划分、Fluent 导出边界和非牛顿物理方程族均已冻结；当前只等待 UDF/单位、cell identity、CFD velocity→WSS 和 CFD residual 四类 P0 闭环。

## 2026-07-29｜PINN 五项边界确认与 UDF 流变/BC 只读审计（No-Run）

**本次主要修改**：将用户确认的五项边界正式回填体域 PINN 路线：第一阶段只做峰值 WSS；沿用现有 `test36` 做 reused development screen；工程 BC 使用共享入口流量模型和几何分支面积；暂不新增 Fluent mesh/zone/connectivity 导出。只读核对 DING 病例 UDF、`.cas.gz`、Global_conditions 和全库 188 份 AAA/ILO 顶层 UDF，修正原“常黏度牛顿流体”假设。

**对应代码/文档**：[体域物理约束与 PINN 训练路线讨论稿 §0.4/§1.2/§10](WSS最小化_体域物理约束与PINN训练路线_2026-07-29.md)；只读证据为 `data_new/AAA/ruputer/DING_JUN_FENG/{udf-inlet4.c,DING_JUN_FENG.cas.gz,2.jou,Global_conditions/}` 及 `data_new/**/udf-inlet4.c`。

**推进到实验步骤**：仍处于 P0 设计与数据合同阶段，未改代码、数据或作业。新增正式前置审计：解析共享 \(Q(t)\)、病例入口/四出口面积、四组 RCR、流变参数和 `F_FLUX` 单位，并检验 RCR 是否可由出口面积/分支身份确定；P0-B velocity→WSS Oracle 改为使用实际的 variable \(\mu(\dot{\gamma})\)。

**当前状态判断**：DING `.cas.gz` 明确挂接 `cell_viscosity::libudf`，密度为 \(1060\ \mathrm{kg/m^3}\)；188 份顶层 UDF 的波形与流变常数一致，入口面积除数有 180 个不同值。当前 CFD 真源应表述为非滑移刚性壁面、不可压缩、统一广义牛顿 Carreau–Yasuda 型流变，而非全场 \(\mu=0.0035\) 常数；该流变口径已由上方最新条目正式确认。第一阶段可在不新增 Fluent 导出的前提下推进 F0/F1/F2，但不做精确出口面 hard RCR residual 或 connectivity-based residual。

## 2026-07-29｜体域物理约束/PINN 第一性原理路线留档（No-Run）

**本次主要修改**：只读核对当前 WSS-only 锚点、`data_new/` 体内/壁面数据、v4 bundle 字段、旧 `pipeline/` 采样和 V1 physics loss 实现；新增体域物理约束讨论真源。方案明确不在现有壁面 WSS 单头上直接叠加 Navier–Stokes loss，而是保留冻结锚点，新增几何/边界条件条件化的平滑 \(u,p\) 神经场辅助分支，并按 data-only → continuity/no-slip → WSS 梯度一致性 → unsteady momentum 逐级验证。

**对应代码/文档**：[体域物理约束与 PINN 训练路线讨论稿](WSS最小化_体域物理约束与PINN训练路线_2026-07-29.md)；只读证据包括 `pipeline_wss_min/{config.py,preprocess.py,raw_io.py}`、`pipeline/{config.py,utils/sampling.py}`、`training/core/losses.py`、`data_wss_min/**/{bundle.npz,report.json}` 和历史 V1 PINN 推进记录。

**推进到实验步骤**：推进到 P0 数据/数值闭环预注册，尚未进入实现或训练。下一步依次审核全队列 `cellnumber/coords` 时相稳定性、CFD velocity→WSS Oracle、CFD 真值 PDE residual、壁面法向与入口/出口 zone；通过后才建立独立 physics sidecar，并先做 3–5 例 F0 data-only 过拟合。

**当前状态判断**：现有 v4 bundle 已有全体点坐标、距离和 81 时相壁面 WSS/pressure，但默认没有体内速度/压力时间序列；旧 V1 physics batch 在壁面点超过 2048 时只选壁面，Jobs 5536–5538/5540/5600 也未形成完整可判读对照，不能作为 PINN No-Go。本条提出的五项待确认边界已由上方“PINN 五项边界确认与 UDF 流变/BC 只读审计”收敛；其中流变口径按 `.cas.gz` 实际挂接修正为 variable viscosity。本次未改代码/数据，未提交作业。

## 2026-07-29｜SAME-H2 + `log(local_radius)` 两臂完训、分析与工作簿回填

**本次主要修改**：
- 在向后兼容的特征表中新增 `log_local_radius`，`load_case` 从正的物理 mm `local_radius` 计算自然对数；新增独立回归测试。
- 以 ★ SAME-H2 为父配置生成精确同-seed并发 control 和唯一新增第7列 `log_local_radius` 的处理臂；原始半径、LocalGeoPE 索引、H2 损失和所有训练协议冻结。
- 从父模型相同的 control106 train-only 来源生成扩展特征统计；新增静态审计、CUDA 前后向/checkpoint/full-chunk门禁与幂等 Slurm 提交合同。
- 完成两臂 400 epoch、best/last test36、36 例配对 bootstrap 和预注册 Gate 复算；新增结果分析器及幂等工作簿更新器，在既有三张结果表回填两臂，不增加工作表。

**对应代码/配置**：`training_wss_min/{config.py,dataset.py,tests/test_log_local_radius_feature.py}`；`training_wss_min/configs/pointnetpp_lsa2_h2_logradius_20260729/`；`training_wss_min/tools/{prepare_lsa2_h2_logradius_matrix.py,smoke_lsa2_h2_logradius_matrix.py,analyze_lsa2_h2_logradius_results.py,update_lsa2_h2_logradius_xlsx.py}`；`training_wss_min/cluster/{preflight_lsa2_h2_logradius_matrix.slurm,run_lsa2_h2_logradius_matrix.slurm,submit_lsa2_h2_logradius_matrix.py}`；`training_wss_min/preflight/lsa2_h2_logradius_matrix_20260729_{results_analysis.json,results_summary.csv,paired_case_stats.csv}`；`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`。

**推进到实验步骤**：24项核心/Support-Query回归、5项 Local Transformer 回归、新特征2项测试、配置差异审计和提交 dry-run 通过。Jobs `11032`、`11033_0/1` 均 `COMPLETED (0:0)`；两臂 checkpoint、400 epoch、best/last test36、36 例 CSV、配置哈希和有限值完整性通过。结构化分析、文档和 `.xlsx` 已回填；工作簿保持5张表、6页，未恢复 `D2K64三种子确认`。

**当前状态判断**：处理臂相对并发 H2 control 的 `ΔR²_cb=+0.0436`、`Δnormalized R²_cb=+0.0042`、`ΔMAE=-0.0342 Pa`、`Δhigh-WSS nRMSE=-0.00295`、`ΔIoU=-0.0029`，同时通过两个主门和全部保护线，晋级为新单 seed 开发锚点。病例全场 R² 均值差95%CI跨零，故不得写成稳健确认；暂不补多 seed，下一项可在新锚点上单变量验证解析锚残差参数化。

## 2026-07-29｜REG-P10-LSA2 SAME/IND × H1/H2 六臂完训、分析与工作簿回填

**本次主要修改**：
- 完成 `SAME/IND × MSE/H1/H2` 六臂训练、best/last test36 评估、三域与 36 例配对 bootstrap；所有 Gate 只使用本轮并发控制。
- 新增严格结果分析器，核验 Slurm、400 epoch、checkpoint、配置哈希、有限值、病例集合和 best/last；结构化输出 JSON 与两张 CSV。
- 新增幂等工作簿更新器；向 `实验矩阵总览`、`教师汇报视图` 和 `汇总对比` 回填六臂，不增加 sheet，并把打印首页切换为本轮紧凑结论页。保持 5 张表、6 页，未恢复 `D2K64三种子确认`。

**对应代码/文档**：
- `training_wss_min/configs/pointnetpp_regp10_lsa2_objective_ind_20260728/*.json`
- `training_wss_min/tools/{prepare_regp10_lsa2_objective_ind_matrix.py,smoke_regp10_lsa2_objective_ind_matrix.py}`
- `training_wss_min/tools/{analyze_regp10_lsa2_objective_ind_results.py,update_regp10_lsa2_objective_ind_xlsx.py}`
- `training_wss_min/cluster/{preflight_regp10_lsa2_objective_ind_matrix.slurm,run_regp10_lsa2_objective_ind_matrix.slurm,submit_regp10_lsa2_objective_ind_matrix.py}`
- `training_wss_min/preflight/regp10_lsa2_objective_ind_matrix_20260728_{results_analysis.json,results_summary.csv,paired_case_stats.csv}`
- `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`
- [高值区域优化方案 §4.4](WSS高值区域预测优化方案.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)、[PointNet baseline 矩阵 §0P](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)

**推进到实验步骤**：Jobs `11019`、`11020_0–5` 全部 `COMPLETED (0:0)`；六臂完整性 `6/6` 通过。结构化分析、文档和 `.xlsx` 已回填；工作簿保存回读和 LibreOffice 6 页 PDF 渲染通过。

**当前状态判断**：IND 主效应 `ΔR²_cb=+0.0014`，No-Go；H1 在 SAME/IND 下 `ΔIoU=+0.0175/-0.0018`，均 No-Go。H2 在两种 query 模式内均过 high-WSS nRMSE Gate，但 IND-H2 相对 SAME-H2 的 R²/normalized R²/MAE 更差，IND 不晋级。SAME-H2 以 `R²_cb=0.3239`、`Δhigh-WSS nRMSE=-0.00205` 进入后续输入臂；H2 control / +`log(local_radius)` 已在上方完成并产生新锚点，不组合 H1/H2。

## 2026-07-28｜O0 hotspot BCE / q90 pinball 2/2 完训、分析与工作簿回填

**本次主要修改**：
- 不补 RCR 多 seed、不做 RCR 架构；以 O0 geometry-only S3 `PointNeXt-R + LocalGeoPE` 为唯一父模型。
- 在 `TrainConfig/objectives` 增加两个默认关闭的单变量辅助目标：病例相对 q90 balanced hotspot BCE，以及 q90 pinball；默认值保持历史模型行为不变。
- H1 使用第二输出通道作为 hotspot logit，评估与 PostView 仍严格读取第一 WSS 回归通道；H2 保持单输出。新增配置差异审计、单元/梯度测试、CUDA smoke、Slurm 门禁/训练和幂等提交入口。
- 完成 O0/H1/H2 的 best/last、分域和 36 例配对复算；新增独立结果分析器与幂等工作簿更新器，工作簿不新增 sheet，只向既有三张结果表追加两臂，保持 5 张表、6 页紧凑版式。

**对应代码/文档**：
- `training_wss_min/{config.py,objectives.py,evaluate.py}`
- `training_wss_min/tests/test_hotspot_tail_objectives.py`
- `training_wss_min/configs/pointnetpp_o0_hotspot_tail_20260728/*.json`
- `training_wss_min/tools/{prepare_o0_hotspot_tail_matrix.py,smoke_o0_hotspot_tail_matrix.py}`
- `training_wss_min/tools/{analyze_o0_hotspot_tail_results.py,update_o0_hotspot_tail_xlsx.py}`
- `training_wss_min/cluster/{preflight_o0_hotspot_tail_matrix.slurm,run_o0_hotspot_tail_matrix.slurm,submit_o0_hotspot_tail_matrix.py}`
- `training_wss_min/preflight/o0_hotspot_tail_matrix_20260728_results_{analysis.json,summary.csv}`
- `training_wss_min/preflight/o0_hotspot_tail_matrix_20260728_paired_case_stats.csv`
- `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`
- [高值区域优化方案 §4.3](WSS高值区域预测优化方案.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)、[PointNet baseline 矩阵 §0O](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)

**推进到实验步骤**：静态矩阵审计、目标函数有限值/梯度检查和 CUDA smoke 通过；Jobs `11012`、`11013_0`、`11013_1` 均 `COMPLETED (0:0)`。两臂均完成 400 epoch、best/last checkpoint、best/last test36 全云评估和 36 例逐病例 CSV；配置哈希、产物完整性、有限值与 Slurm 状态审计全部通过。工作簿回读、专门结果区域渲染和整本 6 页 PDF 渲染通过，`D2K64三种子确认` 未恢复。

**当前状态判断**：H1 的 `Δtop10 IoU=+0.0170`，距主门 `0.0030`；H2 的 `Δhigh-WSS nRMSE=-0.00072`，仅达到门槛量级约 36%。两臂 normalized R²/physical MAE 保护线和三域 R² 均通过，但按预注册规则仍为 **2/2 No-Go**；best/last 结论一致，不组合、不补 H2 相邻 q/λ。当时提出的 O0 IND 单臂已由本日志上方的 REG-P10-LSA2 2×3 矩阵取代。

## 2026-07-28｜S3 RCR Oracle 3/3 完训、配对复算与工作簿回填

**本次主要修改**：
- 基于 `data_new/**/Global_conditions` 对应 UDF 源文件提取 174/174 病例四出口 \(R_1/R_2/C\)，形成 12 维自然对数病例特征；train138 病例等权统计，O2 在 train/test 内按 AG、AAA subtype、ILO 0/1 分层做无固定点置换。
- 固定 S3 `PointNeXt-R + LocalGeoPE`、mixed `138/0/36`、seed1234、400 epoch、train-loss 选模和 `legacy_vertex`，完成 O0 geometry、O1a true RCR、O2 shuffled RCR 三臂训练与 best/last 评估。
- 新增结果审计与幂等工作簿回填工具；复算聚合指标、分域 R²、best/last 敏感性和逐病例 bootstrap，并在原工作簿增加 `RCR Oracle` 专页。

**对应代码/文档**：
- `training_wss_min/tools/{prepare_rcr_oracle_matrix.py,analyze_rcr_oracle_results.py,update_rcr_oracle_xlsx.py}`
- `training_wss_min/configs/pointnetpp_rcr_oracle_20260728/*.json`
- `training_wss_min/preflight/rcr_oracle_matrix_20260728_results_{analysis.json,summary.csv}`
- `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`
- [高值区域优化方案](WSS高值区域预测优化方案.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)、[PointNet baseline 矩阵 §0N](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)

**推进到实验步骤**：Jobs `11008→11009_[0-2]%3` 全部 `COMPLETED (0:0)`；三臂 checkpoint、best/last test36 指标、逐病例 CSV 与配置哈希完整性通过。工作簿已回填 `实验矩阵总览`、`教师汇报视图`、`汇总对比` 并新增 `RCR Oracle` 专页，LibreOffice 渲染验证通过。

**当前状态判断**：O1a 相对 O0 的 physical `ΔR²_cb=+0.0805`、`ΔMAE=-0.1354 Pa`、`Δtop10 IoU=+0.0337`；O2 的 `ΔR²_cb=-0.0033`。真实 RCR 同时优于几何基准和等维负对照，判 **Go-to-follow-up**；下一步补 O1a/O0 多种子或新协议确认，并评估可部署代理。单种子、复用 test36、train-loss 选模仍是结论边界。

## 2026-07-28｜REG-P10 静态 EdgeConv 3/3 完训、分析与工作簿回填

**本次主要修改**：
- 在 `PointNetSetAbstraction` 中新增几何约束的 `StaticEdgeConvCorrection`：只复用既有 SA 邻域，计算 `[x_i, x_j-x_i, Δp_ij/r]` 残差消息，不执行动态 KNN；`ModelConfig.edgeconv_stages` 为 1-based 开关，默认 `()` 不创建模块或 checkpoint key。
- 以 REG-P10 `S3 + LocalGeoPE + DropPath0.10, seed1234` 为父配置生成 `SA1/SA2/SA12` 三个单变量臂，补齐配置生成、静态审计、GPU smoke、Slurm 门禁/训练/提交、单元测试、结果分析和幂等写表工具。
- 新增可复跑分析产物，严格比较同一 REG-P10 父模型，核验 400 epoch、best/last、test36 全云指标、36 例逐病例 CSV、配置哈希和有限值，并复用已注册的 R²/MAE/high-WSS/分域 Gate。

**对应代码/文档**：
- `training_wss_min/{baseline_models.py,config.py}`
- `training_wss_min/tests/test_static_edgeconv.py`
- `training_wss_min/configs/pointnetpp_regp10_edgeconv_20260727/*.json`
- `training_wss_min/tools/{prepare_regp10_edgeconv_matrix.py,smoke_regp10_edgeconv_matrix.py}`
- `training_wss_min/tools/{analyze_regp10_edgeconv_results.py,update_regp10_edgeconv_xlsx.py}`
- `training_wss_min/cluster/{preflight_regp10_edgeconv_matrix.slurm,run_regp10_edgeconv_matrix.slurm,submit_regp10_edgeconv_matrix.py}`
- `training_wss_min/preflight/regp10_edgeconv_matrix_20260727_{prepared,static_audit,geometry_audit,gpu_smoke,submission,results_analysis}.json`
- `training_wss_min/preflight/regp10_edgeconv_matrix_20260727_{results_summary,paired_case_stats}.csv`
- `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`
- [PointNet baseline 矩阵 §0M](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md) 与 [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：
- 108 项回归测试全部通过；三份配置的逐字段静态审计 `3/3`、138 个训练病例的 SA 几何合同、目标层 CUDA 执行、有限梯度、严格 checkpoint 重载和 full/chunk 推理一致性全部通过。
- GPU 门禁 Job `10993` 与训练/评估数组 `10994_[0-2]%3` 全部 `COMPLETED (0:0)`；三臂均完成 400 epoch、best/last checkpoint、best/last test36 全云评估和 36 例逐病例 CSV，父模型加三臂完整性 `4/4` 通过。
- 分析器产出 JSON/CSV；工作簿在既有「实验矩阵总览」第 `179–181` 行、「教师汇报视图」第 `153–155` 行和「汇总对比」第 `244` 行起追加 EdgeConv 结果，保存回读与 LibreOffice PDF 渲染通过。

**当前状态判断**：
- `EC-SA1/SA2/SA12` 的物理 `ΔR²_cb` 分别为 `-0.0053/-0.0023/-0.0338`，没有一臂超过 REG-P10。
- `EC-SA2` 虽有 MAE `-0.0240 Pa`、top10 IoU `+0.0122`、AG/AAA 正向和 21/15 病例胜负，但 high-WSS `-0.0256`、ILO `-0.0324` 且病例 CI 跨零，未过主指标和保护线，不能事后晋级。
- 本轮静态 EdgeConv 3/3 No-Go；不继续补动态特征图或多种子。该结论只覆盖当前 REG-P10 上的残差 EdgeConv 最小实现，不外推否定所有 UDGCNN/DGCNN 架构。

## 2026-07-26｜《WSS 高值区域预测优化方案》第一性原理重估与对抗性审查 ✅只读诊断、方案回填

**本次主要修改**：
- 新增只读诊断脚本 `training_wss_min/tools/diagnose_high_wss_ceilings.py`：不训练、不改数据，逐位复现 `baseline_models.py` 解码器的 `knn_interpolate`（\(1/d^2\)、k 近邻归一化），在 **全库 192 个 bundle** 上产出四组量——(1) support→全点云的**协议天花板**（物理与标准化两个空间）、(2) **标签稳定性**（8 近邻平滑 / 相邻心动时相漂移）、(3) log WSS 的**病例间/病例内方差分解**、(4) 逐例 **Poiseuille 先验** \(\log\tau\sim b\log r\) 的指数与 \(R^2\)。支撑点抽样改用 `hashlib` 派生种子（内置 `hash()` 按进程加盐，会破坏跨运行复现）。
- 在 `docs/02-推进与变更/WSS高值区域预测优化方案.md` 追加第二部分 §13–§17（约 480 行）：§13 第一性原理重估、§14 对抗性审查 D-01…D-13、§15 未覆盖优化空间 N-01…N-09、§16 修订优先级（取代原 §11/§12）、§17 复核方法与出处。文首加阅读提示，声明 §13–§17 推翻或修订原 §1/§2.1/§4/§5.4/§9.3/§10.2/§11/§12。
- **未改动任何训练/评估代码，未提交任何作业。**

**对应代码/文档**：
- `training_wss_min/tools/diagnose_high_wss_ceilings.py`（新增）
- `training_wss_min/preflight/high_wss_ceilings_20260726.json`（新增，192 例逐例明细 + 方差分解）
- [WSS高值区域预测优化方案 §13–§17](WSS高值区域预测优化方案.md)
- 数据源（只读）：`runs/pointnetpp_regp10_transformer/outputs/localtf_sa2_s1234/eval/ckpt_best/{metrics.json,per_case_metrics.csv}`、`preflight/regp10_transformer_matrix_20260726_results_summary.csv`、`preflight/d2_k64_ilo_structure_three_seed_confirmation_summary.csv`、`data_wss_min/**/bundle.npz`、`data_new/*/*/{Global_conditions/,*.cas.gz}`

**推进到实验步骤**：
- 全库 192 例诊断跑通（AG 76 / AAA 65 / ILO 51），产物落 preflight；结论量级与 24 例分层预跑一致。
- 收缩恒等式 \(\hat y_q/y_q=\exp[\sigma(\hat z_q-z_q)\,]\) 逐例验证，\(\lambda=1\) 与落盘 `physical_calibration_p99_pred_true_ratio` 最大偏差 \(3\times10^{-6}\)。
- 发现 `evaluate.py:write_reports` 的 `norm_res = res.get("normalized", res)` 使 `per_case_metrics.csv` 的**无前缀列**（`cal_*`/`dist_*`/`overall_*`/`high_wss_*`/`bifurcation_*`/`stenosis_*`/`legacy_vertex_*`）静默为**标准化空间**，与 `metrics.json` 同名的物理区块并存。已核对 `tools/analyze_regp10_transformer_results.py` 读 `metrics.json`，**已发布的 9 臂汇总表未受污染**；本轮不改代码，只在方案 D-01 登记为待修隐患。

**当前状态判断**：
- 五条量化结论：① 物理尾部亏欠是标准化亏欠经 \(\sigma=1.3669\) 的 \(\exp\) 放大，**不是独立故障**；② 网络已是 log 空间条件均值（pooled slope 0.6325 vs \(R^2\) 0.6181），收缩是 MSE 的定义；③ log 方差 **85% 在病例内**、仅 14.9% 在病例间——给 §4 oracle / §7.4 尺度头 / §7.5 分域容量**共同**设了 ~15% 上限；④ 协议天花板 IoU **0.813**（AG/AAA/ILO = 0.858/0.784/0.738）、标准化 \(R^2\) 0.976，**分辨率不是瓶颈**，full-resolution refiner 应后置；⑤ 泛化差距 ≥0.13 且 `selection_rule=train_loss` + `val_cases=0` 在 best epoch 378/400 选模。
- 已识别但**尚未执行**的零训练动作：λ 方差重标定前沿（唯一待实测的是物理 \(R^2\)/MAE 两列）。
- 新增未用监督清单（均已在 bundle 内）：81 心动时相、`wall_wss_vec(81,N,3)`、`wall_pressure(81,N)`；以及零代码的 `query_mode="independent"` 训练/评估解码口径对齐。
- 下一步待用户裁定：是否按方案 §16 的 P0 七项启动（全部不训练）。

## 2026-07-27｜固定 D2 PNXR+GeoPE 的 local-SA1 / local-SA2 对照 ✅完训、分析与工作簿回填

**本次主要修改**：
- 以已完成的 `D2 c125×k64 + PointNeXt-R + LocalGeoPE` 为唯一父配置，新增 `local_transformer_stages=[1]` 与 `[2]` 两个 seed1234 单变量臂；固定 AG/AAA `106/0/27`、train106 stats、random5000/SAME、`125/125/32`、`64/16/16`、6D 输入、7D GeoPE、无 DropPath、400 epoch 与 train-loss 选模。
- 新增独立配置生成、门禁、训练数组与提交器；训练完成后自动执行 best/last test27 全云评估和 best/last 对比。
- Transformer GPU smoke 补充历史 D2 兼容路径：配置未显式给出 `feature_stats_path` 时，严格复用训练循环从冻结 train partition 重算统计的合同；显式 stats 配置行为不变。
- 新增可复跑结果分析器与幂等写表器：区分物理/归一化 R²，核验 Slurm、配置哈希、400 epoch、best/last、27 例 CSV 与有限值，计算同病例 bootstrap CI；Excel 只更新既有三张结果表并执行回读和 LibreOffice 渲染。

**对应代码/文档**：
- `training_wss_min/tools/prepare_d2_c125_k64_transformer_matrix.py`
- `training_wss_min/tools/smoke_regp10_transformer_matrix.py`
- `training_wss_min/tools/{analyze_d2_c125_k64_transformer_results.py,update_d2_c125_k64_transformer_xlsx.py}`
- `training_wss_min/configs/pointnetpp_d2_c125_k64_pnxr_geope_transformer_20260726/*.json`
- `training_wss_min/cluster/{preflight_d2_c125_k64_transformer.slurm,run_d2_c125_k64_transformer.slurm,submit_d2_c125_k64_transformer.py}`
- `training_wss_min/preflight/d2_c125_k64_pnxr_geope_transformer_20260726_{prepared,static_audit,submission}.json`
- `training_wss_min/preflight/d2_c125_k64_pnxr_geope_transformer_20260726_results_{analysis.json,summary.csv}`
- `training_wss_min/preflight/d2_c125_k64_pnxr_geope_transformer_20260726_paired_case_stats.csv`
- `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`
- [PointNet baseline 矩阵 §0L](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md) 与 [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：
- 配置读取、逐字段静态审计 `2/2`、Python compile、Slurm shell syntax、5 项局部 Transformer unittest、目标层模块实例化与提交 dry-run 均通过；run_dir 均不存在。
- GPU 门禁 Job `10979` 与两条训练任务全部 `COMPLETED (0:0)`；两臂均完成 400 epoch、best/last checkpoint、best/last test27 全云评估和 27 例逐病例 CSV，父模型加两臂的完整性审计 `3/3` 通过。
- 分析器产出 JSON/CSV；写表器新增「实验矩阵总览」第 `177–178` 行、「教师汇报视图」第 `151–152` 行和「汇总对比」第 `240` 行起的区块，保存回读与 PDF 渲染通过。

**当前状态判断**：
- 本组是 fixed `106/0/27` D2 协议下的 SA1/SA2 对照，不含 REG-P10 的 DropPath0.10，也不含 ILO/mixed test36，故只与同一 D2 PNXR+GeoPE 父模型严格配对。
- `D2-L-SA1` 物理 `R²_cb -0.0293`，MAE/RMSE、high-WSS、IoU、AG/AAA 全部退化，判 No-Go。
- `D2-L-SA2` 物理 `R²_cb +0.0015` 近似持平，但归一化 `R²_cb -0.0132`、high-WSS `-0.0105`，病例 CI 跨零，判“持平、不晋级”。last checkpoint 的正向敏感性不用于事后改选。
- REG-P10/mixed test36 的 SA2 信号未在无 DropPath、无 ILO的 fixed test27 D2 协议上直接复现，当前停止 D2 local Transformer 扩展。

## 2026-07-26｜REG-P10 局部 SA Transformer 完整矩阵 ✅8/8 完训、分析与工作簿回填

**本次主要修改**：
- 新增 `LocalNeighborhoodTransformer`：在单个 SA center 的邻域内部，对逐边 MLP+LocalGeoPE token 执行 pre-norm 多头自注意力与 FFN，再做 max 聚合；通过 group mask 保证不跨 center、不跨病例。
- `ModelConfig` 新增前后兼容字段 `local_transformer_stages/heads/ffn_ratio/dropout`。stage 为 1-based；默认 `()` 时不实例化模块、不增加 state_dict key，旧 JSON/checkpoint 保持兼容。
- 明确 SA3 不再新增第二套 Transformer：既有 `CoarseGlobalBlock/coarse_attention` 已经完成 32 个 coarse centers 间的全局 self-attention、相对几何 bias 与 FFN。
- 以 `S3+DropPath0.10, seed1234`（REG-P10）为父配置完成局部 stage 的全部 7 个非空子集：`SA1/SA2/SA3/SA12/SA13/SA23/SA123`，另加复用旧模块的 `SA3-global` 对照，共 8 臂。
- 修正既有未跟踪 SA-grouping 测试夹具的层数声明：三层 SA 参数显式补 `sa_blocks=[1,1,1]`；生产配置校验未放宽。
- 新增可复跑结果审计器和安全写表器；统一核验配置哈希、400 epoch、best/last、全云/逐病例产物与有限数值，计算同病例 bootstrap CI，并按预注册四类 Gate 自动判定。工作簿只更新既有「实验矩阵总览」「教师汇报视图」「汇总对比」，保存后回读并经 LibreOffice 渲染。

**对应代码/文档**：
- `training_wss_min/{baseline_models.py,config.py}`
- `training_wss_min/tests/test_local_sa_transformer.py`
- `training_wss_min/tests/test_sa_grouping_matrix.py`
- `training_wss_min/configs/pointnetpp_regp10_transformer_20260726/*.json`
- `training_wss_min/configs/sweeps/pointnetpp_regp10_transformer_20260726.txt`
- `training_wss_min/tools/{prepare_regp10_transformer_matrix.py,smoke_regp10_transformer_matrix.py}`
- `training_wss_min/tools/{analyze_regp10_transformer_results.py,update_regp10_transformer_xlsx.py}`
- `training_wss_min/cluster/{preflight_regp10_transformer_matrix.slurm,run_regp10_transformer_matrix.slurm,submit_regp10_transformer_matrix.py}`
- `training_wss_min/preflight/regp10_transformer_matrix_20260726_{prepared,static_audit,submission,results_analysis}.json`
- `training_wss_min/preflight/regp10_transformer_matrix_20260726_{results_summary,paired_case_stats}.csv`
- `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`
- [PointNet baseline 矩阵 §0K](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md) 与 [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：
- `GNN` 环境下 compileall、Slurm shell syntax、8 份配置读取、逐字段静态审计 `8/8` 与提交 dry-run 均通过；默认关闭路径对旧 REG-P10 checkpoint 严格重载通过。
- 全测试集 `102/102` 通过。此前报告的 `101/102` 中唯一失败，原因是测试夹具自身将三层 SA 列表与默认四层 `sa_blocks` 混用，不代表训练代码回归。
- 正式 GPU 门禁 Job `10967` 与训练/评估数组 `10968_[0-7]%4` 均 `COMPLETED (0:0)`；8 个 run 均完成 400 epoch、best/last checkpoint、best/last test36 全云评估和 36 例逐病例 CSV，审计 `9/9`（含父模型）通过。
- `analyze_regp10_transformer_results.py` 固定比较同一 REG-P10 父模型并产出 JSON/CSV；`update_regp10_transformer_xlsx.py` 新增总览第 `169–176` 行、教师视图第 `143–150` 行和汇总第 `230` 行起的严格对照区块，回读与 PDF 渲染均通过。

**当前状态判断**：
- “SA3 centers 之间加 Transformer”与已做的 coarse attention 是同一结构问题；local-SA3 则发生在每个 SA3 邻域的聚合前，两者不是同一机制。
- **只有 `L-SA2` 过预注册 Gate**：`R²_cb 0.2957→0.3171`（`+0.0214`），MAE/RMSE `-0.0332/-0.0968 Pa`，high-WSS `+0.0453`，AG/AAA/ILO `+0.0105/+0.0198/+0.0341`；下一步只补 seeds `7/2025`。病例均值 CI `[-0.0101,+0.0324]` 跨零，所以仍只写单种子强筛选信号。
- `L-SA12` 仅 `+0.0063`，未过主门槛；其余局部臂和 `G-SA3` 均不超过 REG-P10。多层全开退化说明收益集中于 SA2 中尺度邻域，不能把“加入更多 Transformer 层”作为默认方向。

## 2026-07-26｜D2 c125×k64 固定 split 的 PointNeXt-R + LocalGeoPE 对照已提交

- 新增 `prepare_d2_c125_k64_pnxr_geope.py`：从 `d2_rand5000_c125_k64` 克隆，严格保留 AG/AAA `106/0/27`、train106 stats、6D xyz+geom、random5000/SAME、`125/125/32`、`64/16/16`、seed1234 和 400 epoch；仅开启 `model.sa_blocks=[1,1,0]` 与 `model.local_geope=true`。
- LocalGeoPE 沿用完整输入的 7D 公式 `Δxyz/r + distance + Δ(abscissa,radius,curvature)`，不复用纯 XYZ 的 4D 分支；未修改既有 D2 配置或已完成 run。
- 新增独立 Slurm 门禁、训练/评估与提交脚本。`py_compile`、shell syntax、配置读取、逐字段静态审计 `1/1` 和提交 dry-run 均通过。首轮 `10956→10957` 因 smoke harness 将 D2 合法的空 `feature_stats_path` 当作路径而在模型执行前失败，已取消不会启动的 `10957`；烟测改为复用训练时的 train106 统计重算逻辑后，恢复链 `10958→10959`（`afterok`）均 `COMPLETED (0:0)`。
- 新增可复跑分析器与安全写表器 `analyze_d2_c125_k64_pnxr_geope.py` / `update_d2_c125_k64_pnxr_geope_xlsx.py`。相对原 D2 的 best/test27，PNXR+7D LocalGeoPE 的 `R²_cb +0.0347`（`0.2628→0.2975`）、MAE/RMSE `-0.1313/-0.1537 Pa`、high-WSS `+0.0805`、top10 IoU `+0.0273`，AG/AAA `+0.0526/+0.0181`；病例 R² 19/8、均差 95% CI `[+0.0134,+0.0837]`。工作簿仅在既有「实验矩阵总览」「汇总对比」追加该行和严格配对区块，37 列指标映射回读与 LibreOffice 渲染均通过，未新增页签。
- manifest/审计/分析/提交记录位于 `training_wss_min/preflight/d2_c125_k64_pnxr_geope_20260726_*`。

## 2026-07-25｜纯 XYZ LocalGeoPE 公式兼容、训练完成与结果回填

- LocalGeoPE 由固定 7 维边特征扩展为兼容模式：保留三项语义属性时仍是 `Δxyz/r + distance + Δ(abscissa,radius,curvature)`；`local_geope_feature_indices=[]` 时切换为纯 XYZ 的 `Δxyz/r + distance` 四维边特征。既有六维 config/checkpoint 的结构不变。
- 新增 XYZ 专用单测，并以 `S2 PointNeXt-R + xyz` 为父配置生成一条严格对照；只开启 LocalGeoPE 与空属性索引，不混入其它模块。
- 静态差分 `1/1`、新增四维 GeoPE 单测和既有模块测试通过；GPU 门禁 `10929`、训练/评估 `10930` 均 `COMPLETED (0:0)`，完成 400 epoch 与 best/last test36 全云评估。
- 新增可复跑分析器 `analyze_xyz_local_geope.py`。best checkpoint 相对 S2-PNXR 纯 XYZ：`R²_cb +0.0671`（`0.1798→0.2469`）、MAE/RMSE `-0.2833/-0.2853 Pa`，病例 R² 27/9、95% CI `[+0.1062,+0.5750]`；相对 S2 的 xyz+geom 仍为 `-0.0059`，病例 CI 跨零。因此 4D LocalGeoPE 证明相对坐标信息可以弥补大部分纯 XYZ 缺口，但不替换原 7D GeoPE/xyz+geom 主线。
- 原六维输入 GeoPE 路径的 7D 边编码与 MLP 输入维度保持不变；本次只是为 `indices=[]` 增加独立 4D 分支，已提交的历史 7D GeoPE 配置、checkpoint 与训练结果均不受影响。

## 2026-07-25｜S3 正则化补齐矩阵：20 个 seed1234 配置完成、分析与回填

- 新增 `prepare_s3_regularization_completion_matrix.py`：基于 S3-GEOPE seed1234，仅补未跑的 8 个单变量强度（head=`.05/.15/.20`、DropPath=`.15/.20`、NeighborDrop=`.10/.15/.20`），不重复 `.05/.10` 已完成臂。
- 新增 12 个两两组合，三对正则器均执行 `{.05,.10}×{.05,.10}` 完整交叉；没有加入第三项正则，也不与 attention/loss/归一化组合。
- 新增独立静态审计、GPU 门禁、训练与提交脚本。静态逐字段差分 `20/20` 通过；门禁 `10923` 与训练/评估数组 `10924_[0-19]` 均 `COMPLETED (0:0)`，每臂保留 400 epoch、best/last 与全云评估产物。
- 新增 `analyze_s3_regularization_completion.py` 与 `update_s3_completion_xyzgeope_xlsx.py`。单变量最强为 HeadDrop=.15（`R²_cb=0.2983`，对 S3 `+0.0059`），DropPath=.05/.10 为温和正信号（`+0.0017/+0.0034`），DropPath=.15 和全部 NeighborDrop 为负。两两组合没有超过最优单变量；仅 DP=.05+ND=.10（`+0.0031`）和 HD=.10+DP=.10（`+0.0007`）为小正数，不构成可执行协同。
- 本轮只使用 seed1234、且 test36 为历史工程筛选集，故不改变 S3 锚定，也不将小差异提升为正式 Go；如果继续，只把 HeadDrop=.15 与 DropPath=.10 分开补 seeds `7/2025`。
- 工作簿仅更新现有「实验矩阵总览」「汇总对比」：追加 20 条补齐配置和 1 条 xyz+LocalGeoPE 配置，按既有 37 列字段映射写入并完成 LibreOffice 渲染回读；未新增工作表。

## 2026-07-25｜7 月 24 日三批训练完成：分析、文档与工作簿回填

- `10871→10872_[0-11]`：S3 正则化 12/12 完成。新增/运行 `analyze_s3_regularization_results.py`，逐 run 核验配置 SHA、400 epoch、best/last、有限值与同 seed S3 配对；DropPath `0.05/0.10` 均 3/3 正但平均 `ΔR²_cb=+0.0089/+0.0076`，未达到正式 Go `+0.015`，只记录 Weak-Go；NeighborDrop 0.05 为 No-Go。
- `10882→10883`：S3+SA3 32-center/4-head attention 完成。新增 `analyze_s3_coarse_attention_single.py`；相对 seed1234 S3 的 `R²_cb +0.00835`、high-WSS `+0.0353`，但病例 CI 跨零、MAE 与 AG 小退，因此仅为单种子 Weak-Go，待 `7/2025` 才能确认。
- `10903→10904_[0-2]`：纯 XYZ 三对照完成。新增 `analyze_xyz_input_ablation.py`；D2/M1/S2 相对各自 xyz+geom 父模型的 R²_cb 均退 `-0.0581/-0.0958/-0.0730`，M1/S2 的病例 CI 均完全为负，关闭纯 XYZ 扩展。
- 工作簿已运行 `update_s3_regularization_xlsx.py`，在「实验矩阵总览」「教师汇报视图」「汇总对比」加入 12 条正则化结果；`update_xyz_attention_results_xlsx.py` 将纯 XYZ 三臂与 SA3 attention 单臂严格按「实验矩阵总览」既有 37 列指标映射写入，并在「汇总对比」追加对应严格配对区块。此前新增的两个专用页已删除；保存已回读并经 LibreOffice 渲染验证。

## 2026-07-24｜D2-K64 三锚点纯 XYZ 输入对照已提交（`10903→10904_[0-2]%3`）

- 新增 `prepare_xyz_input_ablation.py` 与 `audit_xyz_input_ablation.py`：从 `D2 c125×k64`、`M1/S0 D2-K64 mixed138/test36`、`S2 D2-K64 + PointNeXt-R mixed138/test36` 各复制一臂；逐字段合同仅允许 `name`、`notes` 和 `data.input_features=[x,y,z]` 不同。三条锚的 split、统计文件、采样/SAME、SA、残差块、训练种子与预算均冻结。
- 新增 3 份配置于 `training_wss_min/configs/pointnetpp_xyz_input_ablation_20260724/`，以及 Slurm GPU 门禁、依赖训练数组和提交记录。`py_compile`、`bash -n`、配置读取、静态差分 `3/3` 和提交前 dry-run 均通过。
- 正式 GPU 门禁 Job `10903` 已排队（当前 4 张 4090 被 `10872` 占用）；训练/评估数组 `10904_[0-2]%3` 以 `afterok:10903` 等待。所有任务由 Slurm 自动等待可用 GPU，不会绕过队列或抢占运行作业。

## 2026-07-24｜S3 + SA3 coarse attention 单种子实验已提交

**本次主要修改**：
- 新增单配置生成器与静态硬门禁，从 S3-GEOPE seed1234 克隆并严格只改变 `name/model.coarse_attention`。
- 新增独立 Slurm GPU 门禁、训练/评估和幂等提交脚本；门禁覆盖输入 SHA、运行目录、SA1 几何、CUDA 前后向/optimizer step 与 checkpoint/full-chunk smoke。
- 明确 attention 只作用于 SA3 的 32 个 coarse centers、4 heads；LocalGeoPE 保持开启，三类 Drop 全部关闭。本轮仅做 seed1234，不追加多种子。

**对应代码/文档**：
- `training_wss_min/tools/{prepare_s3_coarse_attention_single.py,audit_s3_coarse_attention_static.py}`
- `training_wss_min/cluster/{preflight_s3_coarse_attention_single.slurm,run_s3_coarse_attention_single.slurm,submit_s3_coarse_attention_single.py}`
- `training_wss_min/configs/pointnetpp_s3_architecture_20260724/s3arch_sa3_coarse_attention_s1234.json`
- `training_wss_min/preflight/s3_sa3_coarse_attention_single_{prepared,submission}.json`
- [S3-GEOPE 当前优化执行计划](WSS最小化_S3-GEOPE锚定_正则化与架构优化执行计划_2026-07-24.md) 与 [训练实验跟踪](WSS最小化_训练实验跟踪.md)。

**推进到实验步骤**：本地 compile、shell syntax、严格单变量审计与模块测试 10/10 均通过。正式门禁 Job `10882` → 训练和 best/last test36 评估 Job `10883`（`afterok:10882`）已提交。

**当前状态判断**：当前 4 张 GPU 正运行正则化数组首波，`10882` 等待 GPU，`10883` 等待门禁依赖。结果只用于单种子架构筛选；相对同 seed S3 达到 `ΔR²_cb≥+0.012` 且 high-WSS/分域无实质退化才记正信号，不作跨种子结论。

## 2026-07-24｜S3-GEOPE 锚定正则化首波 · 门禁通过 · 12 配置训练中

**本次主要修改**：
- `ModelConfig` 新增前后兼容的 `drop_path_rate` 与 `neighbor_drop_rate`（默认均为 0）；PointNeXt-R block 使用病例图级 DropPath，最大率沿 block 深度线性递增。
- SA 与 InvRes 邻域新增保覆盖 NeighborDrop：训练时随机丢消息边，每个 center 永远保留最近边；评估时关闭。
- 新增 4 变体 × 3 seeds 的严格单变量配置生成、静态差分、正式 GPU 门禁、数组训练与提交脚本。
- 新建 [S3-GEOPE 当前优化执行计划](WSS最小化_S3-GEOPE锚定_正则化与架构优化执行计划_2026-07-24.md)，并将已完成的 D2-K64 两协议终审与 PointNet++ 路线讨论稿移入 `_archive/WSS最小化/`。

**对应代码/文档**：
- `training_wss_min/{config.py,baseline_models.py}`
- `training_wss_min/tests/test_d2_k64_wave2_modules.py`
- `training_wss_min/tools/prepare_s3_regularization_matrix.py`
- `training_wss_min/cluster/{preflight_s3_regularization_matrix.slurm,run_s3_regularization_matrix.slurm,submit_s3_regularization_matrix.py}`
- `training_wss_min/configs/pointnetpp_s3_regularization_20260724/`
- `training_wss_min/preflight/s3_regularization_matrix_{prepared,submission}.json`
- 本文、训练实验跟踪、PointNet baseline 矩阵、`docs/README.md`、归档 README。

**推进到实验步骤**：S3 absolute `R²_cb=0.2924/0.2743/0.2713`，相对 M1 三种子 `Δ=+0.0310/+0.0059/+0.0303`，据此冻结 S3 为工程锚点。12 个子配置与同 seed 父配置仅差 `name` 和一个正则化字段，静态审计 12/12 passed；正式提交 Job `10871`（门禁）→ `10872_[0-11]%4`（400 epoch + best/last test36 eval）。

**当前状态判断**：代码编译通过；新增正则化定向测试 3/3、模块全套测试 10/10 通过。4 张 RTX4090 提交前均空闲。Job `10871` 已 `COMPLETED (0:0)`，静态、几何与 12 个 CUDA smoke 全部 passed；`10872_0–3` 已进入第一波训练，其余任务由 `%4` 并发上限自动续跑。test36 仍是历史工程筛选集，本轮只能决定下一工程锚点，不能形成独立泛化结论。

## 2026-07-24｜S3-GEOPE 根因矩阵 12 配置 ✅12/12 完训、分析与回填｜整体No-Go、机制证实根因#1

**完训与分析**：门禁 `10857` 与 array `10858_[0-11]%6` 全部 `COMPLETED (0:0)`；新增可复跑分析器 `training_wss_min/tools/analyze_s3_rootcause_results.py`（复用上一轮 flatten/20,000 次配对 bootstrap 口径，导入其函数保证一致），核验 12 臂 + S3 父三种子的完整性并算每臂 vs S3 父（同种子）分域配对差与三种子聚合；新增 `update_s3_rootcause_xlsx.py` 安全追加 12 行到「实验矩阵总览/教师汇报视图」并在「汇总对比」加分域判读段（temp 复制→保公式→回读校验→覆盖）。

**终裁**：S3 父 R²_cb 基线 `0.2924/0.2743/0.2713`。**无任何臂跨种子稳定超过 S3**——casebal 三种子均值 `+0.0040`（seed1234 `-0.0166`）、cohortbal `-0.0018`（seed1234 case CI `[-0.056,-0.002]`）、cohort_onehot `-0.0084`；单种子 pooled138 `-0.0139`、rawHuber02 `-0.0086`、组合 `-0.0268`（最差）。稳健信号在**分域**：平衡口径/域条件把被 ILO 压掉的 **AAA 抬回**（cohortbal 三种子 `+0.0255/+0.0362/+0.0249`），但 **ILO 相应回落**、AG 持平——是 AAA↔ILO 容量**再分配**（原负迁移的镜像），故总体持平。含义：**LocalGeoPE(S3) 已吸收归一化口径在无几何 Q2V 基座上的净收益（A1b 曾 `+0.0295`），S3 骨干上两杠杆不叠加**；high-WSS R² 全部仍为负，尾部（rawHuber02）未改善。S3-GEOPE 仍为参考模型。下一步转向尾部感知/域感知损失加权或分域容量，不再试全局归一化变体；DropPath/NeighborDrop 待主干实现后单独 Gate。

**对应代码/文档**：分析/写表脚本见上；真源 `training_wss_min/preflight/s3_rootcause_results_{analysis.json,summary.csv,paired_case_stats.csv}`；12 run 位于 `training_wss_min/runs/pointnetpp_s3_rootcause/outputs/`；同步回填 [训练实验跟踪](WSS最小化_训练实验跟踪.md)、[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md) 与 `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`。

<details><summary>提交记录（2026-07-23 晚）</summary>

**背景/动机**：承接"加入 ILO 数据反而更差"的负迁移归因（见 [训练实验跟踪](WSS最小化_训练实验跟踪.md) §数据扩容负迁移归因）。三种子已确认 **S3-GEOPE** 为当前最优且唯一三域同升的臂，因此以它为父模板，针对三个根因各开**单变量**臂过夜跑，供明日总结分析。DropPath/NeighborDrop 需改主干（当前仅 head dropout 已实现），本批**不含**，按"元工具/主干改动不混入数据对照"原则推迟到单独实现。

**本次主要修改（纯新增，未改主干/既有配置/判定）**：
- 新增 `training_wss_min/tools/prepare_s3_rootcause_matrix.py`：克隆三种子确认过的 S3-GEOPE seed1234 配置，逐臂只改单一字段，产出 12 个子配置 + `preflight/s3_rootcause_matrix_prepared.json`（记录父配置、split/stats/feature SHA、`only_allowed_differences`、input_dim）。
- 新增 `training_wss_min/tools/audit_s3_rootcause_static.py`：纯 JSON 静态单变量审计，校验每个子配置对父配置只在声明字段差异、SHA 一致、run_dir 不存在（本地已跑通 12/12 passed）。
- 新增集群三件套 `cluster/{preflight_s3_rootcause_matrix.slurm,run_s3_rootcause_matrix.slurm,submit_s3_rootcause_matrix.py}`：门禁复用通用几何审计 `audit_sa1_scale_matrix` 与 GPU module/checkpoint/full-chunk smoke `smoke_d2_k64_three_seed_confirmation`；array 以 `afterok` 依赖门禁，逐任务复核 SHA/run_dir/smoke 覆盖，训练后自动 best/last 全云评估 + `compare_best_last`。

**矩阵（父模板 = `s3_d2k64_pnxr_geope_mixed`，D2-K64 mixed `138/0/36`；均 400ep / train-loss 选模 / 冻结 split 与 support-query 协议）**：

| 方向（根因） | 臂 | 唯一变化 | 种子 |
|---|---|---|---|
| 归一化口径（#1） | `caliber_pooled138` | 目标统计→mixed train138 pooled 重算（μ 0.5718→0.6456） | 1234 |
| 归一化口径（#1） | `caliber_casebal` | →逐病例等权 A1（μ 0.8166） | 1234/7/2025 |
| 归一化口径（#1，**主**） | `caliber_cohortbal` | →逐父系等权 A1b（μ 0.7798） | 1234/7/2025 |
| 域条件（#2） | `cohort_onehot` | +cohort one-hot（input_dim 6→9） | 1234/7/2025 |
| 尾部损失（#3） | `rawhuber02` | +物理 raw-Huber λ=0.2（探针） | 1234 |
| 探索组合（#1+#2） | `cohortbal_onehot` | cohortbal + one-hot（非归因） | 1234 |

口径三档 μ 递增（0.5718 control106-pooled → 0.6456 train138-pooled → 0.7798 cohortbal / 0.8166 casebal）验证根因 #1"pooled 被高密度队列拉低"，`pooled138→casebal/cohortbal` 可分离"含 ILO 重算"与"跨域平衡"两效应。

**对应代码/文档**：新增脚本见上；配置 `training_wss_min/configs/pointnetpp_s3_rootcause_20260723/`；真源 `training_wss_min/preflight/s3_rootcause_matrix_{prepared,submission}.json`。集群作业：门禁 `10857`、训练 array `10858_[0-11]%6`（`afterok:10857`）。

**提交时状态**：仅已提交，未等待结果（结果与终裁见本节顶部 2026-07-24 完训回填）。

</details>

## 2026-07-23｜D2-K64 M1/S2/S3 三种子确认 ✅6/6 完训、审计与回填

新增配置克隆、静态/几何/GPU module/checkpoint/full-chunk gate、6-task array 与三种子分析器；子配置逐项只允许 `name/train.seed` 差异，manifest 记录父配置、split/stats SHA。`10848` 与 `10849[0-5]` 全部 `COMPLETED (0:0)`；S3−M1 三种子 ΔR²_cb=`+0.0310/+0.0059/+0.0303`（mean `+0.0224`，3/3正），S3−S2 mean `+0.0138`、2/3正且 MAE 3/3下降。因此只提出 `DropPath 0.05/0.10`、`NeighborDrop 0.05` 单变量计划，本批不混入 Drop。

## 2026-07-23｜D2-K64 ILO 两协议与结构模块矩阵 ✅9/9 完训并回填｜S3-GEOPE 晋级单种子候选

**本次主要修改**：完成第一批 `10838_[0-4]`（F1/M0/M1/S1/S2）与第二批 `10844_[0-3]`（S3/S4/S5C/S5）的统一结果审计；新增可复跑分析器，核验 9 个 run 的 400 epoch、best/last checkpoint、best/last 全云指标、逐病例 CSV、配置 SHA、有限数值与 Slurm `COMPLETED 0:0`，并计算 7 组严格同协议差、20,000 次病例配对 bootstrap 及病例胜负数。新增安全写表脚本，把 9 个新 run 追加到 Excel「实验矩阵总览/教师汇报视图」，把严格配对表追加到「汇总对比」，保存后重新打开回读校验。

**对应代码/文档**：`training_wss_min/tools/{analyze_d2_k64_ilo_structure_results.py,update_d2_k64_ilo_structure_xlsx.py}`；结构化结果 `training_wss_min/preflight/d2_k64_ilo_structure_{results_analysis.json,results_summary.csv,paired_case_stats.csv}`；9 个 run 位于 `training_wss_min/runs/pointnetpp_d2_k64_ilo_structure/outputs/`；同步回填 [终审与执行计划](_archive/WSS最小化/WSS最小化_D2-c125-k64_ILO两协议对照_终审与执行计划_2026-07-23.md)、[前沿算法路线讨论稿](_archive/WSS最小化/WSS最小化_PointNet++下一轮算法优化路线_第一性原理与前沿研究讨论稿_2026-07-22.md)、[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md) 与 `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`。

**推进到实验步骤**：门禁 `10837/10843` 与训练评估 `10838_[0-4]/10844_[0-3]` 全部完成。F0/F1 仅在固定 AG/AAA test27 配对；M0/M1 与 S1–S5 仅在相同 mixed test36 配对；SEP 只允许 S5−S5C，因为两臂共同使用 independent query。结果固定取预注册 `ckpt_best(train_loss)`；last 只作敏感性检查。

**当前状态判断**：F1−F0 `ΔR²_cb=-0.0169`，判为 ILO41 对 AG/AAA 的负迁移 No-Go 信号；M1−M0 `ΔR²_cb=-0.0059`，ILO 分域 `+0.0109` 但 AAA `-0.0331`、IoU `-0.0173`，属于域间权衡，不支持总体晋级。S1 增益仅 `+0.0018`，不超单种子噪声；S2 单独 `-0.0086`，PointNeXt-R core 本身不成立。S3 相对 S2 为 `ΔR²_cb=+0.0396`、MAE `-0.1708 Pa`、RMSE `-0.1750 Pa`、high-WSS R² `+0.0548`、IoU `+0.0157`，逐病例 R² 29胜7负、均值差95% CI `[+0.0600,+0.1259]`，是本轮唯一强晋级候选；绝对 `R²_cb=0.2924` 也是 mixed test36 八臂最高。S4 为较弱正信号；S5−S5C 虽 `R²_cb +0.0235`，但 case-mean R² `-0.0047`、负例 `3→7` 且病例 CI 跨零，只保留条件候选。下一步优先用 seeds `7/2025` 补 M1/S2/S3 的配对确认；Drop 不与本轮结论混合，待 S3 确认后再以单变量小强度正则臂测试。全部新结果仍只有 seed1234，且 test27/test36 均为历史工程筛选集，不写成稳定泛化结论。

## 2026-07-22｜D2 c125×k64 中位数病例 PostView 导出 ✅完成

**本次主要修改**：为 `d2_rand5000_c125_k64` 补导出归一化空间 R² 中位数病例 `AG/fast/LI_ZHI_LIN`（`normalized_overall_r2=0.4896`，与 Excel `case med` 一致）的 PostView 包；新增 `training_wss_min/cluster/run_d2_c125_k64_postview_median.slurm`。

**对应代码/文档**：Job `10803` COMPLETED；产出目录 `training_wss_min/runs/pointnetpp_sa1_scale_single_seed/outputs/d2_rand5000_c125_k64/postview/ckpt_best/test/AG__fast__LI_ZHI_LIN__peak_wss/`（含 `plots/fig_wss_triptych.png` 等；映射覆盖率 100%）。

**推进到实验步骤**：单例 export-only（未跑 test27 全量 verify）；未改权重/评估指标。

**当前状态判断**：D2 中位数病例可视化已齐，可与 P1V（`GAO_FENG_SHAN`）/ Q1V（`WANG_DAO_CHUN`）并排整理。若需 D2 全 test27 PostView，另提作业。

## 2026-07-22｜bridge random5000+500c+k64 的 SEP 对照臂 ✅完成｜No-Go

**背景/动机**：SA1-scale 矩阵（2026-07-21）的 `bridge_rand5000_fixed500_k64`（random5000、固定 center 500/125/32、SA1 knn_cover k=64）沿用 Q1V/SAME 支路，未验证该 bridge 配置在 SEP（query 与 support 独立重采样）下是否同向。按用户要求补一个唯一变量为 SAME→SEP 的对照臂，其余保存配置一律不变。

**本次新增**：
- 新配置 `training_wss_min/configs/pointnetpp_sa1_scale_bridge_sep_followup_20260722/bridge_rand5000_fixed500_k64_sep.json`：以 `bridge_rand5000_fixed500_k64.json` 为唯一模板，逐字段 diff 确认仅 `data.query_mode` 由 `"same"` 改为 `"independent"`（即 Q1V→Q2V 同款 SAME→SEP 写法），其余 data/model/train/eval 字段（`support_n_points=5000`、`sa_center_counts=[500,125,32]`、`sa_nsample=[64,16,16]`、`sa_grouping=["knn_cover","ball","ball"]`、`width=32`、`batch_cases=8`、`seed=1234`、400 epoch、train-loss 选模等）逐项相同。
- 单臂人工 manifest `training_wss_min/preflight/bridge_sep_followup_prepared.json`（记录 parent/single_change/sha256），供 SA1 geometry audit 复用；新增 `training_wss_min/cluster/{run_bridge_sep_followup_preflight.slurm,run_bridge_sep_followup.slurm,submit_bridge_sep_followup.py}`，结构对齐既有单臂追加先例 `submit_q1v_radius_r60_same.py`：门禁作业先跑 `audit_sa1_scale_matrix`（knn_cover 覆盖率硬门）与 `preflight_v4_jobs`，训练作业按 `afterok` 依赖并在启动前二次校验配置 sha256、拒绝已存在的 run_dir。
- 新增 `training_wss_min/tools/analyze_bridge_sep_followup_results.py`：复核配置哈希、`runtime_preserves_frozen_config`（除 `query_mode` 外逐字段一致）、seed/split/epochs/选模规则、400 行 history、best/last 27 例 CSV 与全量有限数值，读取已冻结的 `pointnetpp_sa1_scale_results_analysis.json` 中的父实验（`bridge_rand5000_fixed500_k64`）指标作对照锚点、不重算，输出 `bridge_sep_followup_results_{analysis.json,results_summary.csv}`。

**验证与提交**：本地 `py_compile`/`bash -n` 通过；`ExpConfig.from_json` 加载新配置确认 `query_mode=independent`、`sa_nsample=(64,16,16)`、`sa_grouping=('knn_cover','ball','ball')`、`sa_center_counts=(500,125,32)`，run_dir 未存在；`--dry-run` 通过全部哈希/存在性前置检查后正式提交：门禁 Job `10804`→训练 Job `10805`（`afterok:10804`）均 `COMPLETED (0:0)`。提交记录：`training_wss_min/preflight/bridge_sep_followup_submission.json`。

**当前状态判断**：400 epoch、best/last 27 例评估、配置哈希与逐字段一致性审计全部通过，无 NaN/Inf。物理 `R²_cb` `0.2439→0.2329`（`-0.0111`）、归一化 `R²_cb` `0.6139→0.6065`（`-0.0074`）、病例均值 R² `-0.0222`、MAE `+0.0348 Pa`、RMSE `+0.0476 Pa`、Spearman `-0.0116`、high-WSS R² `-0.0368`；仅负例（`2→1`）、top10 IoU（`+0.0082`）与 p99 比（`+0.0086`）小幅改善，改善幅度均小于回退幅度。判定 **bridge 家族下 SAME→SEP 为 No-Go**，不并入后续候选；主线仍以 SAME（`bridge_rand5000_fixed500_k64`）代表该点数标度对照臂。结果已回填 [训练实验跟踪](WSS最小化_训练实验跟踪.md) 与 [PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)，`WSS_PointNet实验矩阵与结果汇总last.xlsx` 同步新增：「实验矩阵总览」第109行、「教师汇报视图」第109行、「汇总对比」新增 bridge_sep_followup 对比区块（第140–143行），均已回读校验。新增 `training_wss_min/tools/update_bridge_sep_followup_xlsx.py` 完成该次性追加，样式/数值均与父行一致。仍是历史 test27 同协议工程比较，非独立确认。

## 2026-07-21｜实验矩阵汇总 PPT 与图件交付 ✅完成（汇报材料，不改变任何实验判定）

**背景/动机**：导师要求把 `WSS_PointNet实验矩阵与结果汇总last.xlsx` 的指标汇总为一个 PPT，判定哪些指标/实验必要、哪些可删并给出原因，同时用图示记录网络结构与 width/nsample/knn/ball 等调参含义，以及 CFD mesh、模型设置、数据前后处理口径。

**本次交付**：
- 新增 `docs/03-汇报材料/tools/build_wss_matrix_summary_ppt.py`，组装 22 页 PPT → `docs/03-汇报材料/WSS_PointNet实验矩阵汇总_组会汇报_20260721.pptx`（python-pptx；LibreOffice 渲染逐页抽查通过）。
- 复用当日图件 `figures/WSS_PointNet矩阵汇总_20260721/{f1..f10,s1..s6}.png`（生成脚本 `build_wss_matrix_summary_figures.py` / `build_wss_matrix_schematics.py`，此前未回填记录，一并登记）。
- PPT 结构：一页结论 → 数据/CFD 网格与四阶段预处理链路 → 归一化·训练协议·PostView → PointNet(E2) 与 PointNet++ SA3 结构图（标注 width/nsample/radius/SA1 分组/中心采样五类旋钮）→ 采样与 SA1 分组机制示意 → 指标四句话体系 → 11 阶段 79 行实验总览 → 六组结果图 → 指标取舍 → 实验线取舍 → 开放问题 → 附录指标速查。
- 指标取舍建议（30 列→12 列，附全矩阵相关性证据）：保留 物理 R²_cb/case mean/负例/RMSE/NMAE(pooled+case-mean)/high-WSS R²/top10 幅值比/p99 比/top10 IoU + 归一化 R²_cb/Spearman；删除 R²_raw（与 R²_cb r=0.845 共线）、case med/P10（r=0.89/0.73）、range-NRMSE 族（与 NMAE r=0.98/0.89）、物理 MAE、归一化误差 6 列、high-WSS Spearman（全矩阵≈0 从未参与判定）、归一化 top10 幅值比。
- 实验线取舍建议：8 条归档（容量/深度扫描、case-max 归一化、AG-v4 重跑、Q3V、半径 5 臂、QAD 全系、SA1 17 臂、扩容+NormLoss A1/A2/A3/A4）；4 条保留待独立确认（Q1V/Q2V/P2V、Q1V 半径 1.2×、n32_w64、A1b）。
- 口径声明：PPT 只汇报、不改变 xlsx 数据与既有 Go/No-Go 判定；"删除"仅指从汇报口径移出，「实验矩阵总览」sheet 与 runs/ 产物全部保留归档。

## 2026-07-20｜归一化口径×尾部损失 Q2V-pool2025 5-arm 矩阵 ✅完成｜A1b 晋级候选

**背景/动机**：数据审查确认 AG/AAA/ILO 混训收益有限、ILO 甚至掉点，根因是 (1) 全局 log-z 目标统计为 **point-pooled**，被高点密度队列（AAA/ILO 每例 5–7 万点 vs AG ~1.3 万点）主导，稀疏 AG 在归一化空间被错误居中；(2) 缺少疾病域条件，ILO 相对 AG/AAA 属负迁移；(3) log-MSE 天然轻视高 WSS 尾部，与"准确预测高 WSS 区"目标冲突。锚点定为 Q2V-10477（pool2025 train138/test36 通用集，含 ILO）。

**本次主要修改（前后兼容）**：
- `config.py` 新增 cohort one-hot 特征键 `cohort_ag/cohort_aaa/cohort_ilo`（并入 `FEATURE_KEYS`）；旧配置不含即不启用，`input_dim=len(input_features)` 自动派生。
- `dataset.py` 新增 `COHORT_FEATURE_KEYS`/`cohort_family()`；`build_features` 对 cohort 键按病例父系产出 0/1 常量列（**不 z-score**），`compute_feature_stats` 跳过 cohort 键。
- `train.py` 冻结特征统计的必需集减去 cohort 键（防御性，避免未来冻结路径误判缺列）。
- 新增 `training_wss_min/tools/prepare_norm_loss_matrix.py`：从 pool2025 train138 逐病例读峰值 WSS，产出 **逐病例等权**（case-balanced）与 **逐父系等权**（cohort-balanced）两份 log-z 目标统计，并克隆 Q2V 基座写 5 个 arm 配置 + manifest。
- 新增 `cluster/{prepare_norm_loss_matrix_cpu.slurm(node03),run_norm_loss_matrix.slurm(GPU array),submit_norm_loss_matrix.py}`。

**对应代码/配置**：`training_wss_min/{config.py,dataset.py,train.py}`；`training_wss_min/tools/prepare_norm_loss_matrix.py`；`training_wss_min/configs/pointnetpp_norm_loss_20260720/`；`data_wss_min/fold_stats/norm_loss_20260720/`；`training_wss_min/cluster/{prepare_norm_loss_matrix_cpu.slurm,run_norm_loss_matrix.slurm,submit_norm_loss_matrix.py}`；真源 `training_wss_min/preflight/norm_loss_matrix_{prepared,submission}.json`。

**矩阵（单一自变量，均 seed1234 / 400ep / train-loss / legacy-vertex）**：A0=复用已训练 `q2v_ilo_d3_pool2025_refit`（point-pooled 统计，不重跑）。A1=改逐病例等权统计；A1b=改逐父系等权统计；A2=A1+cohort one-hot（input_dim 6→9）；A3=A1+物理空间 raw-Huber 尾部项 λ=0.2；A4=A1+cohort+raw-Huber。对比链：A1↔A0(口径 point-pooled→逐病例)、A1b↔A1(病例→父系)、A2↔A1(加 cohort 特征)、A3↔A1(加尾部损失)、A4↔A1(交互)。

**验证、提交与完训**：本地已验证代码前后兼容——cohort 键通过 `validate_features`，one-hot 列 AG=[1,0,0]/ILO=[0,0,1] 且不进入 feature_stats，旧 6 特征配置 `input_dim` 仍为 6；stats 平衡口径数学与配置克隆单测通过。node03 prepare 作业 `10710` 已 `passed`：138 例统计 casebal log(μ=0.8166,σ=1.2647) / cohortbal(μ=0.7798,σ=1.2737)，均较 pooled(μ=0.6456,σ=1.3021) 上移（证实 pooled 被高密度队列拉低）。GPU array `10711_[0-4]%4` 已全部 `COMPLETED`；5 个 arm 均完成400 epoch，`ckpt_best(train_loss)` 与 `ckpt_last` 的 test36 `legacy_vertex` 评估、36例 CSV 和配置哈希均齐全，无 NaN/Inf、OOM 或提前停止。主表固定使用 best，不按 test 结果改选 last。

**结果与判断（physical test36，A0 为复用的 point-pooled `10487_5`）**：A1b（逐父系等权）最完整：`R²_cb 0.2459→0.2753`（+0.0295）、pooled `0.2176→0.2411`、case mean `0.1974→0.2119`、RMSE `7.058→6.951 Pa`、high-WSS `-0.523→-0.457`、p99 比 `0.334→0.381`；负例 `4→5`、MAE `+0.005 Pa` 是保留代价。A1（逐病例等权）主 R² 近乎持平且负例增至7，只在 high-WSS/p99 有小幅改善。A2（加 cohort one-hot）未优于 A1；A3（raw-Huber λ=0.2）相对 A1 主 R²、误差及高尾均退化；A4 交互也未补回，case mean 与 IoU 最差。故本轮只将 **A1b 归一化口径**列为后续独立确认候选；不以本次单 seed/historical test36 结果宣布泛化提升，cohort 特征、raw-Huber 及其交互不进入组合扩展。结构化真源为各 run 的 `eval/ckpt_best/metrics.json`，汇总已同步至 `WSS_PointNet实验矩阵与结果汇总last.xlsx`。

## 2026-07-20｜SA1全覆盖/低重叠分组 + 单种子完整矩阵 🚧已提交

**本次主要修改**：PointNet++ 新增逐层 `sa_grouping`，支持原始 ball、raw KNN、KNN+coverage repair 和 coverage-first `adaptive_cover`。adaptive先做每点最近center主归属，再做受半径、目标group大小与center-pair重叠预算约束的至多一个次归属，因而硬保证SA1 support覆盖100%、任意组对重叠率不超过1/3。该改动不增加模型参数，旧checkpoint `state_dict` 的102个key保持一致。数据层新增fixed/pool8 FPS5000持久化缓存、坐标hash与shape强验证，避免worker重复FPS。

**对应代码/配置**：`training_wss_min/{baseline_models.py,config.py,dataset.py}`；`training_wss_min/tools/{prepare_sa_grouping_single_seed_matrix.py,build_fps_support_cache.py,audit_sa1_grouping_matrix.py}`；`training_wss_min/tests/test_sa_grouping_matrix.py`；`training_wss_min/configs/pointnetpp_sa_grouping_single_seed_20260720/`；`training_wss_min/cluster/{build_sa_grouping_fps_cache.slurm,preflight_sa_grouping_single_seed.slurm,run_sa_grouping_single_seed.slurm,submit_sa_grouping_single_seed_matrix.py}`。

**验证与提交**：6个新单元测试全部通过；ball/KNN/KNN-repair/adaptive均通过同一PointNet++ support-query CUDA反传；adaptive的完整Q1V配置通过CPU反传、RTX4090 batch8 AMP反传、center计数和full/chunk query一致性预检。实测首例FPS缓存为5000个无重复索引，fixed跨epoch不变、multistart跨epoch切换。冻结17个新配置，全部是 `seed=1234` 和精确 `106/0/27`；10个随机support任务由 `10557→10558_[0-9]%4`门控，7个FPS support任务由 `10559_[0-132]%7→10560→10561_[0-6]%4`门控。提交时两个前置任务已运行，训练数组按`afterok`等待；不在本记录预写训练结果。

**当前状态判断**：Q1V/SAME是主基准；Q2V/SEP只补齐support方法的配套矩阵。raw KNN-8/10是允许未覆盖点的负对照，只有repair/adaptive受100%覆盖硬门禁。本轮仍是已多次使用的历史test27上的工程比较，不到做3-seed确认的阶段。冻结和提交真源为 `training_wss_min/preflight/sa_grouping_single_seed_{prepared,submission}.json`。

## 2026-07-19｜Q2V SA 重叠审计汇总包 + xlsx ✅无训练几何复核

**本次主要修改**：新建 `例子/06_PointNet++_SA三层采样与分组/汇总_重叠审计_Q2V-10477_2026-07-19/`，统一输出 `SA_group重叠统计汇总.xlsx`、SA1/SA2/SA3 最近 center 局部图、ball 半径 `0.6×/0.8×/1.0×/1.2×/1.5×` 全部分布与同一最近邻 pair 图、KNN `k=6/8/10/16` 全部分布与同一全局最近 pair 图。新增 `09–10`：SA2/SA3 的“左侧完整源点云 + 右侧局部放大”图（SA2 背景严格为500个 SA1 center，SA3 严格为125个 SA2 center）；新增 `11–15`：固定同一 SA1 最近 center pair 的五种 ball 半径完整点云定位 + 局部放大图，便于逐半径看圈和共享点如何变化。xlsx 新增“半径×SA分层”sheet，完整列出五种半径下 SA1/SA2/SA3 的15行统计；xlsx 同时记录每种方法的中位重叠、P10、`≤1/4`/`≤1/3`/`≤1/2` 比例、100%重叠比例/个数和逐 pair 明细；同时清理未被文档引用的旧顶层 PNG/CSV/说明（`00–04`、`08–09`），保留原始 case trace、VTP/CSV 和 Q1V 专项半径历史图。

**对应代码/文档**：`training_wss_min/tools/{build_q2v_sa_overlap_audit_package.py,plot_q2v_full_vessel_group_topology.py,plot_q2v_pointcloud_global_local_pair.py,plot_q2v_stage_radius_pointcloud_views.py,visualize_pointnetpp_sa_q2v.py}`；`例子/06_PointNet++_SA三层采样与分组/README_ParaView.md`；上述汇总目录；本页。

**推进到实验步骤**：五种 ball 半径均在 `Q2V-10477 / AG/fast/RAN_QING_BO` 上以原模型的固定 eval support、deterministic FPS 和 `torch_cluster.radius(..., max_num_neighbors=16)` 重新导出 trace；脚本断言五份反事实在 SA1/SA2/SA3 的 FPS centre 坐标均完全一致。KNN 对照仅用这些已固定的 SA1 source/centre 坐标重算最近 `k` 个点，不加载 checkpoint、不训练。完整血管图选择最近邻关系中 `5/16=31.25%` 的 pair `283/488`，再把成员映射到三角网格，使用表面测地/欧氏距离比 `≥2.5` 作风险筛查；新半径全局/局部系列固定另一条真实最近邻 pair `99/332`（距离 `7.286 mm`），因此只改变半径，不改变对照对象。xlsx 已以 `openpyxl` 复开，验证5个 sheet、“半径×SA分层”15行、图件索引含 `09–15` 和所有图件文件存在。

**当前状态判断**：Q2V SA1 的 ball 中位重叠随半径为 `25.0% (0.6×) / 31.25% (0.8×) / 12.5% (1.0×) / 0% (1.2×) / 0% (1.5×)`；因每层只保留最多16个半径候选，成员截断集合会变化，不能按“半径越大重叠越高”的直觉解读。KNN-6/8/10/16 的中位数为 `0%/12.5%/20.0%/31.25%`，且均无100% pair。完整血管示例中有 `2/32` 个中心—成员边触发测地/欧氏比风险标记（最大 `2.62`），证明纯欧氏 ball query 存在拓扑捷径风险，但该筛查不是已确认的跨支解剖标签；该几何指标用于筛选候选 grouping，不构成训练性能结论，任何正式 grouping 改写仍须独立训练验证。

## 2026-07-19｜QAD 精确Q2V协议三种子历史test27对照 ✅完成｜No-Go

**本次主要修改**：按用户要求，把QAD比较从 `85/21/27 + n32 + train85 stats` 切到与 `Q2V-10477` 精确匹配的 `106/0/27 + n16_w32 + train106 stats`。复用Q2V/seed1234/interpolate历史锚点，新增R0/interpolate seeds `{7,2025}` 和R1/QAD seeds `{1234,7,2025}`，共5个新任务。每个seed内R0/R1只改decoder；采样、SA中心、radius、nsample、width、loss、400 epoch、train-loss选模与评估support seed均冻结。

**对应代码/文档**：`training_wss_min/configs/pointnetpp_qad_q2v_test27_20260719/`；`training_wss_min/tools/{prepare_pointnetpp_qad_q2v_test27_matrix.py,analyze_pointnetpp_qad_q2v_test27_results.py,update_pointnetpp_qad_q2v_test27_xlsx_uno.py}`；`training_wss_min/cluster/{run_pointnetpp_qad_q2v_test27_preflight.slurm,run_pointnetpp_qad_q2v_test27_matrix.slurm,submit_pointnetpp_qad_q2v_test27_matrix.py}`；`training_wss_min/preflight/pointnetpp_qad_q2v_test27_{prepared,gpu_preflight,submission,results_analysis}.json`及三份结果CSV；本页/训练跟踪/PointNet矩阵/架构执行稿；`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`。

**推进到实验步骤**：Python compile、cluster shell语法、Support/Query+PointNet distribution回归 `32/32` 通过；准备门确认5份配置、run dir、hash、Q2V `106/0/27`、n16/w32、SEP和train106 stats未漂移，QAD参数仅`+2,625`。正式GPU预检Job `10549` 在`00:01:54`内 `COMPLETED (0:0)` 且5/5 passed；训练数组 `10550_[0-4]` 全5项均 `COMPLETED (0:0)`，用时`00:09:50–00:10:53`。五个新run的400 epoch、best/last checkpoint、test27 legacy-vertex指标和27例CSV完整，配置哈希无漂移、无NaN/Inf。分析器生成JSON+三份CSV；Excel通过LibreOffice UNO原生写入，总览新5条run+2条均值、教师视图新2条均值，并完成回读和版式验证。

**当前状态判断**：同seed配对中QAD的normalized `R²_cb` 差为 `-0.02225/+0.00743/+0.00055`，三种子均值由 `0.60487` 降至 `0.60012`（`-0.00476`）；物理 `R²_cb` 均值由 `0.27149` 降至 `0.25380`（`-0.01769`）。IoU均值改善 `+0.01204`，但RMSE、p99与high-WSS护栏不支持晋级；逐例normalized p99差95% CI为 `[-0.03672,-0.00333]`，Wilcoxon `p=0.00102`。QAD虽把normalized R²种子标准差从`0.01187`压到`0.00112`，却稳定在更低均值，属偏差—方差交换而非准确性提升。因此精确Q2V协议判定 **No-Go**：Q2V-10477仍是历史test27 normalized `R²_cb` 最高锚点`0.62097`，停止QAD-REG与以QAD为父实验的R2-DUAL。该test27已多次参与历史决策，本结论只是同协议工程排除证据，不写成新的独立确认。

## 2026-07-19｜Q2V SA1 ball-query 与 KNN-16 局部重叠审计 ✅无训练反事实可视化

**本次主要修改**：新增同一 `Q2V-10477 / AG/fast/RAN_QING_BO` SA1 的四联局部放大图：历史高重叠 ball-query 示例、满足教师 `5/16=31.25%` 参考的附近但非最小距离 pair、500 个 FPS centre 的全局最近 pair 的原始 ball-query，以及该完全相同 centre pair 的 KNN-16 反事实。按唯一原始 source-point ID 计数共同成员，避免 padding/重复边被错误当作重叠；没有训练、模型推理或配置改写。

**对应代码/文档**：`training_wss_min/tools/plot_q2v_sa1_ball_knn_overlap.py`；`例子/06_PointNet++_SA三层采样与分组/Q2V-10477_vertex-random5000_FPS-center_fast_RAN_QING_BO_ParaView/08_SA1_ball与KNN_最近center及局部重叠对比.{png,csv,json}`；本页。

**推进到实验步骤**：复核旧 `07_SA相邻group重叠示例.png` 的生成器后，明确其 pair 按最大 Jaccard 选取，是“强重叠示例”而非“全局最近 center”。新图固定真实 trace 的 center、support、ball-query 半径及上限 `nsample=16`；KNN 只替换成员选择规则为同一 support 中欧氏距离最近的 16 个点。脚本编译、绘图、CSV/JSON 输出及四个 panel 的成员计数已复核。

**当前状态判断**：旧示例 `292/482` 的 ball group 确为 `16/16=100%`，但距离最近 pair 实为 `126/499`（`5.76 mm`），其 traced ball group 为 `0/16=0%`；同一 pair 的 KNN-16 为 `3/16=18.75%`。故不能将旧图的 100% 误写为“所有最近 group 都 100%”；它说明现有 `radius(..., max_num_neighbors=16)` 在局部候选截断下可出现高冗余样本。KNN 图仅用于无训练几何反事实，若要替换训练 grouping，必须以独立训练/验证评估其性能与跨支风险。

## 2026-07-19｜Q1V SA1 半径 0.8× / 1.2× / 1.5× 相邻 group 重叠图 ✅无训练几何审计

**本次主要修改**：以 Q1V-10476 的 `vertex-random5000 + SAME + FPS centre` 配置，对 `AG/fast/RAN_QING_BO` 重放固定评估support，导出三份仅改 SA radius 的SA trace：`0.04/0.08/0.16`、`0.06/0.12/0.24`、`0.075/0.15/0.30`。新增固定同一邻近 center pair 的三栏图和全最近邻对统计；图中不改变support、500个SA1 center、病例或模型权重。

**对应代码/文档**：`training_wss_min/tools/{visualize_pointnetpp_sa_q2v.py,plot_q1v_radius_overlap.py}`；`例子/06_PointNet++_SA三层采样与分组/Q1V-10476_{r80,r120,r150}_vertex-random5000_SAME_fast_RAN_QING_BO_ParaView/`；`例子/06_PointNet++_SA三层采样与分组/10_Q1V_SA1_半径0.8x_1.2x_1.5x_同一相邻group重叠对比.{png,csv,json}`；本页。

**推进到实验步骤**：三个trace均用同一Q1V配置、同一support seed和 deterministic FPS；脚本逐一断言三份SA1中心坐标完全一致。展示pair为center `39/484`，在所有半径下均有共同成员且三半径均值最接近教师的1/3参考，不能替代整体分布；另对全部376个最近邻pair计算平均/中位/p90/≤1/3比例。Python编译和图像渲染已通过。

**当前状态判断**：展示pair的共享率随半径为`37.5% → 31.25% → 31.25%`，前者略高于1/3、后两者低于阈值；全pair平均为`30.0% → 25.6% → 25.1%`。由于每个ball group的 `nsample=16` 上限保持不变，扩大半径会改变最多16个候选成员的截断集合，不能假设“半径增大=交叠单调增大”。这是采样/分组几何审计，不是训练性能结论。

## 2026-07-19｜PointNet++ R1 QAD-Lite 三种子配对 ✅5/5完成｜不直接晋级R2

**本次主要修改**：实现向后兼容的 `query_decoder=qad_lite`：保留原 3-NN support context 和预测 head，在其上用 query 与插值 support 的 `Δ输入特征、Δxyz、距离` 构造轻量局部分支，并由 support context gate 调制后输出 zero-init residual。默认 `interpolate` 不改变历史行为；QAD 仅新增 `2,625` 参数。新增严格配对的五个 val21 配置：复用已有 R0/seed1234，新增 R1 seeds `{1234,7,2025}` 与 R0 seeds `{7,2025}`；每对除 decoder 和 run 元数据外完全一致。

**对应代码/文档**：`training_wss_min/{baseline_models.py,config.py,tests/test_support_query_v4.py}`；`training_wss_min/configs/pointnetpp_qad_20260719/`；`training_wss_min/cluster/{run_pointnetpp_qad_matrix.slurm,submit_pointnetpp_qad_matrix.py}`；`training_wss_min/tools/{analyze_pointnetpp_qad_results.py,update_pointnetpp_qad_xlsx_uno.py}`；`training_wss_min/preflight/pointnetpp_qad_{matrix_submission,results_analysis}.json`；`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`；本页/训练跟踪/PointNet矩阵/架构方案。

**推进到实验步骤**：GNN 环境 Python compile 和 Support/Query 回归 `17/17` 通过后提交数组 `10540_[0-4]%3`。现五个新任务全部 `COMPLETED (0:0)`；连同复用的R0/seed1234，六个run均有400 epoch、best/last checkpoint、val21 metrics与21例CSV，配置哈希无漂移、无NaN/Inf，test27未访问。新分析工具输出三种子配对、逐例bootstrap/Wilcoxon、预注册门判定及可重跑CSV/JSON；xlsx总览回填6条原始结果+2条均值，教师视图回填2条三种子均值。

**当前状态判断**：QAD的normalized `R²_cb` 三种子均值 `0.53800→0.55226`，`+0.01426`且2/3 seed为正，低于 `+0.015` 主门；物理 `R²_cb` `0.26490→0.26432`，无稳定增益。Spearman、物理RMSE与normalized p99高尾护栏均有seed级回退，只有IoU非劣门通过。三种子平均的逐例normalized R²差 `+0.01107`，95% CI `[-0.00176,+0.02516]`仍跨零。因此判 **No-Go for direct R2**：保留QAD机制，先做residual幅度审计与正则化单变量诊断；R2-DUAL、R3-AXIAL和test27继续冻结。

## 2026-07-18｜PointNet 实验汇总 xlsx 教师视图重排 ✅展示列收敛

**本次主要修改**：按教师汇报阅读路径，删除 `教师汇报视图` 的 P:U 执行/溯源说明列和 `实验矩阵总览` 的 AK:AP 对应列；后续Q2V结果回填器同步改为只写保留指标列，避免再次添加这些列。教师视图改为“设置→核心物理/归一化/热点指标”的摘要计分表，矩阵总览保留完整数值并按实验行交替底色、候选高亮、统一数值精度、分组标题和横向打印版式。另将指标单元格由 LibreOffice 导出的 `[$-409]` locale 标记改为无货币符号的纯数值格式，避免部分查看器显示前导 `$`；NMAE 与 range-NRMSE 的物理/归一化列已统一按百分数显示并在列标题标注 `(%)`。

**对应代码/文档**：`training_wss_min/tools/{style_pointnet_results_xlsx_uno.py,update_q2v_sampling_radius_test27_xlsx_uno.py}`；`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`；本页。

**推进到实验步骤**：格式化在独立临时副本中执行，xlsx容器完整性、两张表的列范围（教师 A:O；总览 A:AJ）、4个新Q2V探索行的ID与物理 R²_cb 数值均已复核；PDF渲染检查了教师页两页续表重复标题/表头，完整矩阵拆为3个横向指标页以保证印刷可读性。不改动训练结果、公式口径或原始run产物。

**当前状态判断**：教师展示页不再承载Slurm状态、协议描述和文件路径，这些审计信息保留在训练跟踪与结构化 JSON；主表更适合当场比较，完整矩阵仍可用于追溯所有指标。颜色只强调当前候选与warm-start参考，不将单seed test27结果包装为确认性结论。

## 2026-07-18｜Q2V/test27 半径与采样探索矩阵 + Q1V-SAME warm-start ✅4/4完成并回填

**本次主要修改**：按用户授权，以 `Q2V-10477` 的历史 `test27` 为探索性锚点新增四组 PointNet++ 配置：Q2V-SEP 的半径 `r80=(0.04,0.08,0.16)` 与 `r60=(0.03,0.06,0.12)`；以 Q1V-SAME 为匹配控制的 `fps_multistart5000` 采样；以及加载 Q2V `ckpt_best`、改用完整 Q1V-10476 `random5000 + SAME + FPS 500→125→32` 协议的 400-epoch warm-start。warm-start 仅加载模型权重，优化器、学习率日程与 checkpoint 选择全部重新开始，并写入初始化 checkpoint SHA。`test27` 在此矩阵只作历史锚定探索，不能表述为独立确认。

**对应代码/文档**：`training_wss_min/{config.py,dataset.py,train.py,evaluate.py,tools/preflight_v4_jobs.py}`；`training_wss_min/tools/{prepare_q2v_sampling_radius_test27_matrix.py,analyze_q2v_sampling_radius_test27_results.py,update_q2v_sampling_radius_test27_xlsx_uno.py}`；`training_wss_min/configs/pointnetpp_q2v_sampling_radius_test27_20260718/`；`training_wss_min/cluster/{run_q2v_sampling_radius_test27_preflight.slurm,run_q2v_sampling_radius_test27_matrix.slurm,submit_q2v_sampling_radius_test27_matrix.py}`；`training_wss_min/preflight/q2v_sampling_radius_test27_results_{analysis.json,summary.csv}`；`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`；本页/训练跟踪/PointNet矩阵。

**推进到实验步骤**：GPU formal preflight `10505` 4/4通过；数组 `10506_[0–3]` 均完成400 epoch、`ckpt_best/last` 评估和 PostView `27/27`。结果审计复核了history精确400行、best/last checkpoint、数值有限、提交配置哈希未漂移、随机初始化/strict warm-start元数据和全27例可视化验证；`last-best R²_cb` 在 `-0.0041～+0.0016`，不改变预注册的 train-loss best 选模。xlsx 使用 LibreOffice 写入后重新以 xlsx 容器和单行ID/指标一致性复核。

**当前状态判断**：新四组均未超过历史 Q1V 的物理 `R²_cb=0.2763`。相对 Q2V，r80/r60 分别为 `0.2433/0.2406`（`Δ=-0.0291/-0.0319`）；相对严格匹配Q1V-SAME，`fps_multistart5000` 为`0.2297`（`Δ=-0.0466`）；Q2V→Q1V SAME warm-start为`0.2670`（相对Q1V `-0.0093`）。所以缩小ball radius、以fps_multistart替换vertex-random、以及该权重初始化均不晋级；Q1V继续是下一步独立复核候选。全部是用户授权的历史test27探索，不能写成确认性比较；AreaRandom继续冻结。

## 2026-07-18｜PointNet++ 架构精度方案清理定稿 ✅文档完成｜No-Run

**本次主要修改**：将 PointNet++ 第一性原理工作稿收敛为单一最终执行版；删除跨智能体审查日志、逐题答复、作者/版本流水和重复优先级。有效结论已合并到唯一的架构路线、实验矩阵、Go/No-Go 门和启动清单中。

**对应代码/文档**：[`WSS最小化_PointNet++架构精度优化_第一性原理方案与交叉论证工作稿_2026-07-18.md`](WSS最小化_PointNet++架构精度优化_第一性原理方案与交叉论证工作稿_2026-07-18.md)；本页。

**推进到实验步骤**：已冻结最小顺序 `Stage-0 审计 → R1-QAD → R2-DUAL → R3-AXIAL → 条件触发 R4-RES/R5-MSG → 三 seed 确认`。根据既有第四轮 A3 三 seed smearing 变差证据，将“回变校正/NLL 前置”纠正为历史 No-Go/条件触发支线。本轮未改代码、配置或实验产物，未提交作业。

**当前状态判断**：最终文档可直接用于实现和提交实验；val21 继续使用 `n32_w32` 效率锚点，test27 只在结构与三 seed 结论冻结后做一次确认。AreaRandom 继续受 `133/133` 严格面积映射门禁阻塞。

## 2026-07-18｜Excel 第19–21行 SA 容量纠错与 Q2V 重复行清理 ✅已回填

**本次主要修改**：逐格复核“实验矩阵总览”第20行 Q2V-10477、第21行 Q3V-10478，并追查到第19行 Q1V-10476 有同一模板遗留：三行的每例点数和采样列已正确写为 random5000，但容量结构仍误沿用 Q0 的 `2000→500→125→32`。现按运行时配置统一修正为 `5000→500→125→32；nsample=16；width=32`；Q0第18行继续保留真实的 `2000→500→125→32`。同时清除上轮追加的 `Q2V-10477-reference-test27` 重复行，以第20行原 Q2V 结果作为唯一 test27 历史锚点。

**对应代码/文档**：`training_wss_min/tools/update_q2v_results_xlsx_uno.py`；`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`；[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)；本页。

**推进到实验步骤**：运行时 config 与提交 config 双重核对 Q1V/Q2V/Q3V 的 `support_n_points=5000`、`sa_center_counts=[500,125,32]`；Q2V 为 SEP+FPS center，Q3V 为 SAME+Random center。Excel 更新器改为13个新增结果行并幂等清理旧重复锚点；本轮没有训练、评估或集群提交。

**当前状态判断**：Excel 只保留一个 Q2V-10477 test27 主行，不再通过重复实验行表达对照关系；D1仍以该唯一行作为同test27控制。Q1V/Q2V/Q3V 的真实完整链均为 `5000 support→500→125→32`，此前第19–21行的 `2000` 只是展示错误，不影响已完成模型或指标。

## 2026-07-18｜Q2V 对照、SA 链与归一化排名展示修正 ✅已回填

**本次主要修改**：修正“Q2V-10477 是否缺少对照”和“`2000→500→125→32` 与 `500→125→32` 是否为不同 SA 下采样”的展示歧义。Q2V/本轮数据与架构配置的完整链均写为 `5000 support→500→125→32`；其中 `500/125/32` 是三层 SA center 数，不是输入 support 数。旧 B1/9169 与 Q0/10475 是 `2000 support→500→125→32`，因此中心数相同不代表第一层输入点集、邻域或推理路径相同。Excel 新结果区补一行 `Q2V-10477-reference-test27`，只读重复原 test27 历史锚点，并将 `q2v_arch_dev_n16_w32` 明确标为 `Q2V-structure dev control`。

**对应代码/文档**：`training_wss_min/tools/update_q2v_results_xlsx_uno.py`；`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`；[训练实验跟踪](WSS最小化_训练实验跟踪.md)；[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)；本页。

**推进到实验步骤**：仅做结果展示和口径修正，没有新增训练、评估或 Slurm 提交。原 Q2V checkpoint 不在架构 val21 上补评：dev split 的 `train85∪val21` 恰好等于 Q2V 原 train106，val21 的21例全部曾参与原 Q2V 训练，直接计算会是 `21/21` 数据泄漏。架构可比控制因此是按 dev85 重训、只改 split/train-only stats 的 `n16_w32`，不是原10477 checkpoint。

**当前状态判断**：Q2V 仍是同 test27 归一化空间的历史锚点，normalized `R²_cb=0.6210`；D1 frozen/refit 分别为 `0.5961/0.6092`，未超过 Q2V。架构 val21 的预注册主指标是物理 `R²_cb`，数值第一仍为 `n32_w64=0.2603`；normalized val21 数值第一是 `n16_w64=0.5709`，两者排名不同但不构成冲突，不事后更换主指标。Q2V test27 的 `0.6210` 与架构 val21 不同 split/stats，明确不参与 val21 排名。

<!-- Q2V_ILO_20260718_START -->
## 2026-07-18｜Q2V/ILO 6组 + Point++ 架构6组结果回填｜定量12/12完成

**集群结果**：CPU准备 `10484`、GPU preflight `10485`（12/12 smoke passed）、架构数组 `10488_0–5`、Q2V PostView export-only `10489` 和扩展test36零样本评估 `10490` 均完成；当前用户队列为空。数据数组 `10487_0–2` 全流程完成，`10487_3–5` 虽被 Slurm 标为 `FAILED`，但三者均已完成400 epoch、best/last定量评估和36例导出，退出点是共同病例 `ILO/YU_XIANG_SHENG-1/before` 的 PostView Gaussian 映射覆盖仅 `7193/26942=26.70%`，因此应记为“定量完成、PostView 35/36未闭环”，不重训也不伪写为全流程通过。其余35例覆盖均为100%。

**结果审计与回填修改**：新增 `training_wss_min/tools/analyze_q2v_ilo_arch_results.py`，复核12个run均有400行history、best/last checkpoint与metrics，数值无NaN/Inf，提交配置哈希未漂移；生成 `training_wss_min/preflight/q2v_ilo_arch_matrix_results_{analysis.json,summary.csv}`。新增 `training_wss_min/tools/update_q2v_results_xlsx_uno.py`，以临时副本+有效xlsx容器检查的方式幂等回填14行（12个训练结果+Q2V zero-shot36+Q2V原test27历史锚点），并同步三份WSS跟踪文档。

**主要判读**：D1在同test27冻结统计下加入ILO41训练，物理 `R²_cb 0.2724→0.2516`；D2在同test36下加入ILO32训练相对Q2V零样本 `0.2646→0.2493`；D3加入ILO32训练相对控制 `0.2981→0.2915`，且D3重算统计再降至 `0.2459`。三条协议都没有一致的整体扩容增益，只有负例数或局部IoU/ILO-0等分项改善，不能写成数据扩容有效。架构val21两种width都以 `nsample=32` 最好；`n32_w64` 的 `R²_cb=0.2603` 数值第一，但仅比 `n32_w32=0.2589` 高 `0.0014`、参数量为 `0.880M vs 0.224M`，只登记为开发候选，原test27仍未访问。

**流程异常**：finalizer `10486` 在成功提交全部训练后，因node03系统Python缺少 `uno` 而 `FAILED (1:0)`；该错误没有影响GPU提交或训练，仅阻断自动文档/xlsx回填，本次已在登录环境完成回填。Q2V export-only `10489` 已补齐并验证27/27。AreaRandom六组继续冻结；下一步先审计 `YU_XIANG_SHENG` 的STL/CFD坐标域与裁剪血缘，再决定是否只做PostView修复，不得用放宽阈值掩盖映射问题。
<!-- Q2V_ILO_20260718_END -->

## 2026-07-18｜Phase-V 六组完训结果、状态真源与汇总表回填 ✅定量完成｜五组 PostView 待补

**本次主要修改**：只读核对 Jobs `10473–10478` 的400 epoch history、运行时 config、best/last checkpoint、test27 legacy-vertex metrics、best/last 比较文件及 PostView 输出；将定量结果回填到 PointNet 矩阵、训练跟踪和 `WSS_PointNet实验矩阵与结果汇总last.xlsx`。本次未修改训练/评估代码、未重训、未提交 export-only 或面积阶段作业。

**对应代码/文档**：`training_wss_min/runs/{pointnet_v4,pointnetpp_v4}/outputs/ag_aaa_v4_stratified_{e2_global_random5000_{same,sep},sa3_fps2000_fixed_same,sa3_random5000_fpscenter_{same,sep},sa3_random5000_randomcenter_same}/`；`training_wss_min/cluster/logs/v4sq_{p1v_10473,p2v_10474,q0_10475,q1v_10476,q2v_10477,q3v_10478}.{out,err}`；[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)、`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`。

**推进到实验步骤**：六组的 best/last test27 legacy-vertex 指标均已生成；Q0=`10475` 于 `2026-07-17 22:53:55+08:00` 完成并通过 PostView `27/27`。另五组在旧 exporter 处理 `AAA/ruputer/SHI_YUN_XI` 时触发不应属于 legacy-vertex 路线的面积硬门，训练与评估均已完成但 PostView 仅留下20个病例目录、没有 batch verification。P2V 的 best 物理 R²_cb=`0.2193`，Q1V为`0.2763`；Q2V=`0.2724`但IoU/负例/高尾未同步改善，Q3V=`0.2521`整体回退。所有主结果仍使用 train-loss best，未根据 test27 切换到 last。

**当前状态判断**：P2V 与 Q1V 仅作为下一轮独立重复/确认的优先候选，不构成发布或架构终裁；六组 high-WSS R² 仍为负，最高Q1V也仅`-0.513`。P1V/P2V/Q1V/Q2V/Q3V 必须先 export-only 补齐并逐批验证27/27；Phase-A 六组继续停在严格面积映射`127/133`，不得由本轮 vertex 指标放行。

## 2026-07-17｜Phase-V PostView 隐式面积依赖修复 ✅代码/测试完成｜五组无需重训

**本次主要修改**：复核面积门禁、Support/Query 数据集和 PostView 全链路，发现旧 PostView 即使运行配置为 `legacy_vertex` 也会无条件加载严格 STL 面积，造成点口径作业在训练和 best/last eval 完成后被误判失败。现按 `surface_metric_mode` 分流 high-risk mask/manifest：legacy 只输出 vertex 指标且不读/伪造面积，strict 才输出 area 指标；resume 拒绝复用缺少模式或模式不一致的旧半成品。另修复 `query_mode=same` 时未使用的 `query_sampling=area_random` 误触面积加载，以及 FPS pool 按有效 support sampler 预热的边界。

**对应代码/文档**：`training_wss_min/{dataset.py,tools/export_wss_postview.py,tests/test_support_query_v4.py,README.md}`；[PointNet baseline 矩阵 §0A–0B](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#0a-test27-supportquery-与采样矩阵2026-07-17phase-v-运行中phase-a-待修)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)。

**推进到实验步骤**：GNN 环境运行 Support/Query 与 PointNet distribution 两组回归共28项全部通过。只读核对 Slurm 与日志：Jobs `10473/10474/10476/10477/10478` 均400 epoch训练完成，best/last legacy vertex eval完成，只在 test `AAA/ruputer/SHI_YUN_XI` 的旧 PostView 严格面积调用处退出；`10478` 的导出进程在代码修复前已经载入旧模块，运行中不会热更新。`10475/Q0` 在19:47仍运行。本轮未重提、取消或补交任何作业。

**当前状态判断**：上述五组的 checkpoint 与已有 vertex 指标有效，不需要重训；后续仅用修复后的 exporter 做 export-only 补齐。Phase-A 六组仍因面积映射127/133冻结，不能借本修复绕过严格门。Gaussian 可视化 `mapping coverage=100%` 与原始 STL 面积严格转移是两个独立合同；四个 crop 例不能据此推翻既有中心线特征，`ZHOU_KE_XUN` 需审计坐标变换，`LIU_WEN_QI` 需核查 STL/CFD 数据血缘，确认原始几何错误后才重提中心线与几何特征。

## 2026-07-17｜ILO 术前最终签核完成 ✅41/41 PASS｜旧轮次已归档

**本次主要修改**：在已有人工排除6例基础上，再排除 `WANG_LI_MIN-0`、`YANG_QING_REN-1`；固化人工排除8例、AAA 优先重名策略和 before-only 入口。最终白名单、排除清单、人工决定、增量指纹和 QA 报告集中到唯一活动目录；49/43 例两轮旧报告、旧图包和说明移入 `_archive` 追溯。

**对应代码文档**：`pipeline_wss_min/new_cohorts/{ilo_before.py,visualize_ilo_before.py,qa.py,README.md}`；`pipeline_wss_min/tests/test_ilo_before.py`；[ILO 术前最终审核通过](ILO术前队列最终审核通过_2026-07-17.md)；`data_wss_min/pipeline_reports/ilo_before_final_approved_20260717/`；`assets_新队列审计/ilo_before_final_approved_20260717/`。

**推进到数据步骤**：原始 before61、最终通过41、总排除20；41/41 WSS/节点/peak/原始 STL v4/坐标与归一化硬门通过，WSS watch=0、软旗标=0、ILO 平移修复=0。统一活动 bundle 复验 AAA65+ILO41=106/106 PASS；可视化按 AG76+AAA63+ILO41、固定 X/Y±75mm、Z±275mm、归一化±1、WSS log1p 0–300Pa 重建，分组为20/20/1。8项 unittest 与 Python 编译通过。

**当前状态判断**：人工几何审核已通过，活动 after bundle/report=0，61个原始 after 目录继续保留。本轮未创建 split、全局 WSS 统计或训练，也未修改 AG/AAA 白名单与训练产物。

## 2026-07-17｜test27 两阶段采样协议落地｜vertex/FPS 6组 Jobs `10473–10478` 已提交 🚀

**本次主要修改**：把原本被面积映射共同阻塞的矩阵显式拆成 Phase-V 与 Phase-A。旧配置/旧run缺省为 `legacy_vertex`，只计算既有vertex指标且不读取STL面积、不伪造area字段；只有显式`both_strict`的面积采样/面积评估才触发133例严格mapping。新增Q3V（PointNet++、vertex-random5000、SAME、Random center 500→125→32），并用配置等价测试约束其相对Q1V只改变center FPS→Random。preflight复用同一冻结协议的一次全量bundle/hash审计，六个模型的CPU/CUDA门禁仍逐项执行；提交器冻结为六组vertex/FPS，六组area保持独立backlog。

**对应代码/文档**：`training_wss_min/{config.py,evaluate.py}`；`training_wss_min/tools/{preflight_v4_jobs.py,compare_best_last.py}`；`training_wss_min/cluster/submit_v4_support_query_matrix.py`；Q3V配置及其余11份显式评估模式配置；`training_wss_min/tests/test_support_query_v4.py`；`training_wss_min/preflight/{ag_aaa_v4_vertex_phase_6_preflight.json,ag_aaa_v4_vertex_phase_submission_manifest.json,ag_aaa_v4_area_phase_backlog.json}`；本页、[PointNet baseline矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)、根/docs/训练/config README。

**推进到实验步骤**：相关Python编译、Support/Query回归11项、既有v4协议回归5项、两个Slurm脚本`bash -n`通过。正式Phase-V preflight为6/6 passed：split精确106/0/27、strata与排除/统计/133 bundle SHA/frame均通过；双病例CPU、batch8 RTX4090 AMP均通过且无OOM，PointNet峰值约559 MiB、PointNet++约127–181 MiB；SA中心逐病例精确500/125/32，full-query与chunk最大差`7.45e-8`。Jobs为P1V `10473`、P2V `10474`、Q0 `10475`、Q1V `10476`、Q2V `10477`、Q3V `10478`；提交后首项RUNNING，其余因Resources/Priority正常排队。`9169/9170`均未重复提交或重训。

**当前状态判断**：适合让六个vertex/FPS作业继续排队/训练；它们不消费失败的面积映射，且运行时快照明确记录`surface_metric_mode=legacy_vertex`。Phase-A的P1/P2/Q1/Q2/Q3/Q4仍不适合提交：严格审计仅127/133，通过/失败清单、六例可视化manifest及哈希均保存在`ag_aaa_v4_area_phase_backlog.json`，后续修复后必须重新跑面积门禁，不得把本次vertex结果冒充area结论。ILO、xlsx、Git commit/push均未处理。

## 2026-07-17｜Job `9170` 400 epoch 结果回填 ✅CHECKPOINT VALID｜正式面积评估未闭环｜6例映射可视化

**本次主要修改**：以 `sacct`、Slurm 日志、运行时 config、history 和 checkpoint 为真源验收 Job `9170`。确认训练完整结束后，新增不放宽正式面积门禁的诊断评估入口：全27例只报告 legacy vertex 指标，面积指标只报告通过冻结映射门的26例子集。将 `9170` 与同 split/stats 的 `9169` 做严格配对，回填 PointNet 矩阵、训练跟踪和 `WSS_PointNet实验矩阵与结果汇总-new.xlsx`；同时为133例面积审计的6个失败病例生成原始坐标叠加、仅平移诊断和距离热图。

**对应代码/文档**：`training_wss_min/tools/{evaluate_mapping_blocked_run.py,visualize_area_mapping_failures.py}`；`training_wss_min/runs/pointnet_v4/outputs/ag_aaa_v4_stratified_e2_global_fps2000/eval_diagnostic/`；`docs/02-推进与变更/assets_新队列审计/area_mapping_failures_20260717/`；本页、[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)、`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总-new.xlsx`、根/训练/docs README。

**推进到实验步骤**：`9170` 的 Slurm 最终状态为 `FAILED (1:0)`、用时 `06:45:54`，但400/400 history、`ckpt_best(epoch397, train loss=0.139726)` 与 `ckpt_last(epoch399)` 完整；失败点是训练结束后正式 evaluate 首次处理 train 例 `AG/fast/LI_ZHEN_SHAN` 时被新增面积映射硬门正确阻断。best/last 均已对 test27 诊断补算：best 的全27 legacy vertex 物理/归一化 `R²_cb=0.0530/0.4569`、pooled物理 `R²=0.0831`、病例 R² mean/median=`0.0405/0.0221`、负例12/27、MAE/RMSE=`2.919/6.957 Pa`、Spearman=`0.6512`、top10 IoU=`0.1727`、high-WSS R²=`-0.8620`。同口径 `9169` 分别为 `0.1331/0.5307`、`0.1371`、`0.0861/0.0785`、6/27、`2.736/6.749`、`0.6959`、`0.1846`、`-0.8004`；9170 逐例10例改善、17例退化。9170 last 与 best 的物理 `R²_cb` 只差 `+0.0009`，不改变 train-loss 选出的 best 主结论。面积有效子集26/27的 physical/normalized area `R²_cb=0.1559/0.5243`、area top10 IoU=`0.2564`，因排除 `SHI_YUN_XI` 且9169没有同口径结果，未用于架构判优。xlsx 已经 LibreOffice 两页渲染 QA；新增 Python 文件编译通过。

**当前状态判断**：同一独立 test27 上，可以判定“当前冻结配置的0.22M SA3 优于0.79M E2”，但不能写成容量配平后的 PointNet++ 架构终裁；9170 仅峰值点距离/bbox 更小（`0.1399 vs 0.1725`），热点质心、IoU、排序和高尾总体仍由9169更好。四个失败图明确是完整 STL 尾段超过裁剪 CFD 壁面；`ZHOU_KE_XUN` 主要是世界坐标原点大偏移，`LIU_WEN_QI` 在仅平移后仍存在几何差异。正式 test27 面积指标、9170 标准 `eval/` 和 PostView 仍未闭环；11组 AreaRandom/vertex-random 训练继续保持0/11提交，不能据26例诊断子集放行。

## 2026-07-17｜test27 Support/Query + Area/Vertex Random 实现 ⚠️训练提交被面积映射硬门阻断｜9169 PostView 修复重跑

**本次主要修改**：实现向后兼容的 Support/Query 配置、PointNet/PointNet++ 分离编码、固定 SA `500→125→32`、FPS/确定性 Random center、vertex-uniform `random` 与独立 `area_random`、fixed-support/full-query 分块推理、面积加权风险区/病例等权物理指标、11 份冻结配置及提交/manifest 工具。PostView 路径统一以 canonical ID 解析；对 pipeline 已裁剪壁面使用 bundle 冻结 STL 尺度并同步裁剪 STL 尾段。Job `9169` 只跑 export，不重训、不重写 best/last 指标。

**对应代码/文档**：`training_wss_min/{config.py,dataset.py,baseline_models.py,evaluate.py,metrics.py,surface.py}`；`training_wss_min/tools/{export_wss_postview.py,verify_postview_batch.py,audit_surface_area_v4.py,preflight_v4_jobs.py,compare_best_last.py}`；`training_wss_min/configs/{pointnet_v4,pointnetpp_v4}/` 的 P1V/P2V/P1/P2/Q0/Q1V/Q2V/Q1/Q2/Q3/Q4；`training_wss_min/cluster/{run_v4_support_query.slurm,submit_v4_support_query_matrix.py,run_9169_postview_export_only.slurm}`；本页、[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)。

**推进到实验步骤**：Python compile、GNN unittest `55/55`、全部 cluster shell `bash -n` 通过；11 个 config/run/job/log 路径互不冲突。用8个已通过面积映射的真实 train 病例做非正式 RTX4090 batch8 诊断，11/11 AMP forward/backward 通过、无 OOM，PointNet 峰值0.546 GiB、PointNet++ 0.123–0.177 GiB，SA中心逐例精确500/125/32；PointNet、PointNet++ FPS/Random center 的10887点 full-query vs chunk1024 最大差分别 `5.22e-8/1.49e-8/2.98e-8`，每次完整预测 support encoder 仅调用一次。133/133 bundle/frame 可加载，但严格面积审计仅 `127/133` 通过：四例因 pipeline 裁剪壁面而与完整 STL 尾段不一致，`ZHOU_KE_XUN/LIU_WEN_QI` 的 STL/CFD 原始世界坐标明显失配；因此正式 11-config preflight 为 failed，11 个训练作业均未提交，未生成伪 submission manifest，也未重复提交 Job `9170`。9169 export-only 首次 Job `9817` 正确拒绝 `SHI_YUN_XI` 的 6.2% mapping；修正冻结尺度/裁剪后 Job `9818` `COMPLETED (0:0)`，27/27病例均为4 VTP/3 PNG/3 CSV、mapping 100%，且 checkpoint/best-last metrics/per-case CSV 前后 SHA 完全一致。

**当前状态判断**：代码和配置已准备，但不适合启动11组训练；协议明确要求任一病例双向 mapping p95>bbox 2% 或 max>10% 即硬失败，当前不得放宽阈值、不得把 AreaRandom 静默替换为 vertex random，也不得只提交矩阵子集。冻结面积审计见 `training_wss_min/preflight/ag_aaa_v4_surface_area_mapping_133.json`；需先决定并完成可审计的“裁剪后有效 STL 子面”与两例跨坐标系 STL 的数据修复/重发布，再重跑11组正式 CPU/CUDA/显存 preflight 和提交。ILO、xlsx、数据删除、Git commit/push 均未处理。

## 2026-07-16｜PointNet++ Jobs `9167–9169` 结果回填 ⚠️SA3 No-Go｜分层对照未闭环

**本次主要修改**：只读核对三组 PointNet++ 的400 epoch history、best/last checkpoint、train/test 全云 metrics 与 PostView 完整性；将结果回填到 PointNet 状态真源、训练跟踪、根/ docs 入口与 `WSS_PointNet实验矩阵与结果汇总-new.xlsx`。本次未修改训练/评估代码，未启动新作业。

**对应代码/文档**：`training_wss_min/runs/pointnetpp_v4/outputs/{ag_v4_sa3_e2_global_fps2000,ag_aaa_v4_locked_sa3_e2_global_fps2000,ag_aaa_v4_stratified_sa3_e2_global_fps2000}/`；`training_wss_min/cluster/logs/pnpp_*_916{7,8,9}.{out,err}`；[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)、`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总-new.xlsx`、`README.md`、`docs/README.md`。

**推进到实验步骤**：Jobs `9167/9168` 已完成训练、best/last 全云评估与 test15 PostView `15/15`；Job `9169` 已完成训练及 best/last test27 评估，但 PostView 在第一个 AAA 病例因 bundle 路径错解停止，当前 `15/27`；同 split PointNet E2 Job `9170` 仍在训练。

**当前状态判断**：当前 0.22M SA3 配置在 AG-only 与锁定 AG test15 混合两个协议上，物理 `R²_cb` 分别为 `0.1500/0.1490`，均低于 0.79M PointNet E2 的 `0.1866/0.2054`；只有 top10 IoU 轻微改善，整体判当前配置 **No-Go**。由于参数量未配平且 sampled-train/full-cloud 可能存在密度落差，不写成 PointNet++ 架构终裁。分层 `9169` 的物理/归一化 `R²_cb=0.1331/0.5307`，但在 `9170` 与 27/27 PostView 完成前只能作独立基线，不作架构胜负结论。

## 2026-07-16｜v4 PointNet++ 三协议 + PointNet 同 split 对照提交 🚀RUNNING/QUEUED

四份已验收配置在提交前再次核对 SHA-256、配置解析、Slurm `bash -n` 和正式 run 目录不存在，随后使用统一的 train→best/last 全云 train+test eval→best test PostView 流水线提交：PointNet++ AG `61/0/15` = Job `9167`，PointNet++ 锁定 AG test15 混合 `118/0/15` = `9168`，PointNet++ 新分层混合 `106/0/27` = `9169`，PointNet E2 新分层同 split 对照 = `9170`。提交后 `9167/9168` 已在 `master` 启动，`9169/9170` 分别因 Resources/Priority 正常排队；`9167` 已确认 CUDA、train61 并推进至 epoch1，`9168` 已确认 CUDA、train118 和正确模型构建。启动日志未见 NaN/Inf、OOM、Traceback、frame/path 错误。本条只记录正式提交与早期启动核查，不把临时 loss 当结果，也未修改实验汇总 xlsx。

## 2026-07-16｜v4 正式验收、混合重划与 PointNet++ 四组预检 ✅READY｜未提交新训练

**正式验收**：新增 `training_wss_min/tools/accept_v4_pointnet_results.py`，以 `sacct`、运行时 `config.json`、run 路径和实际产物为真源重新验收 Jobs `9138/9140`。两项均 `COMPLETED (0:0)`；配置冻结为 seed1234、FPS-2000、400 epoch、`stl_landmarks_v4`，split 精确为 `61/0/15` 与 `118/0/15`。best/last checkpoint 元数据、train+test 全云指标、per-case CSV、best PostView 15/15、每例4个 VTP（各60个）和 mapping coverage 100% 全部通过；统计 manifest 中所有 bundle SHA-256 与现文件一致。日志未发现 NaN/Inf、OOM、Traceback、frame/path 或病例数错误；九个质量/人工排除病例均未进入任何 partition、train-only stats 或特征统计。

**common-test15 公平重算**：三组均直接读取已保存的逐点预测，且真值数组逐点一致；没有重训旧 E2、没有用 test 反选 checkpoint。统一指标真源为 `data_wss_min/pipeline_reports/v4_cutover_20260715_1921/{v4_e2_global_result_analysis.json,v4_e2_common_test15_per_case_comparison.csv}`。旧 v3 / AG-v4 / AG+AAA-v4 的 pooled `R²` 为 `0.2597/0.1852/0.2086`，MAE 为 `2.754/2.847/2.772 Pa`，RMSE 为 `4.902/5.143/5.068 Pa`，NMAE 为 `0.02431/0.02513/0.02447`；病例等权 `R²` 为 `0.2554/0.1866/0.2054`。混合相对 AG-only 改善整体误差、Spearman（`0.6788→0.7041`）和 top10 IoU（`0.1314→0.1575`），但 high-WSS MAE `10.919→11.195 Pa`、high-WSS R² `-1.8279→-1.8473`，仍有高尾幅值压缩。

**病例证据与判读边界**：混合相对 AG-only 退化最大为 `LI_SHU_KUN`（病例 R² `-0.2932`）、`GUO_XI_JIANG`（`-0.1492`）；混合 MAE 最高为 `ZHANG_JUN_HUA 4.5599 Pa`、`LU_ZHEN_QING 3.9195 Pa`、`QIN_SI_FU 3.6872 Pa`。最低峰值幅值比分别为 `GUO_XI_JIANG 0.1027`、`LIU_FENG_MING 0.1057`、`LI_SHU_KUN 0.1213`；热点峰值距离/bbox 最大为 `LU_ZHEN_QING 0.3625`、`LIU_FENG_MING 0.3168`。证据更符合模型高尾压缩叠加 AG/AAA 密度与统计权重变化，不构成自动删病例依据。高 WSS 尾部观察病例 `ZHOU_KE_XUN`、`WU_GUANG_CUN`、`DING_JUN_FENG`、`ZHANG_YONG_ZHI`、`LI_ZHI_LIN` 已在结构化分析中单列训练侧整体/高尾/热点证据，全部继续保留；pressure gauge offset 也不触发自动排除。

**第三 split 与统计**：新增 `split_AG_AAA_wss_min_v4_stratified_seed1234.json`，从合格池 AG76+AAA57 按固定 strata 顺序、canonical ID 排序和 seed1234 确定性分层为 train106/test27：train=`AG61 + AAA rupture21 + AAA unrupture24`，test=`AG15 + 6 + 6`，无 val。旧 AG test15 不锁定，新 test27 只重合 `GUO_XI_JIANG`、`ZHANG_JUN_HUA` 两例，禁止与 common-test15 总指标直接横比。`HOU_SHEN_QIAN/KANG_XI_MING` 作为相关几何同组分配，未发现其他精确重复几何跨 partition。新统计只读取 train106；AG/AAA 分别贡献 `778,189/2,360,613` 点，即 `24.79%/75.21%`。

**PointNet++/PointNet 准备**：新增三份固定三层 SA PointNet++ 配置（`2000→500→125→32`、radius `0.05/0.10/0.20`、nsample16、FP k=3、width32、head64），覆盖 AG-v4、锁定 AG test15 的混合、以及新混合重划；另新增同 split 的 PointNet E2 对照。四组均为 xyzgeom、GLOBAL log-z、FPS-2000、MSE、seed1234、400 epoch、无旋转/重采样、train-loss 选模。全量病例加载、frame/排除/泄漏/统计哈希、双病例 CPU 前后向、RTX4090 CUDA AMP 前后向均通过；正式 run 目录尚不存在。当前结论为 **可提交但等待用户确认**，本轮没有执行 `sbatch`。

**空间清理边界**：新增只读清单工具 `build_v4_archive_inventory.py`，为旧 AG 快照、staging、HAN 修复快照、活动数据、fold stats、历史 WSS run 和日志生成逐文件 SHA-256 manifest、逻辑/独占空间与保留理由。旧 AG v3 快照可释放 `2.685 GB`，AG staging 因硬链接仅约 `30.3 MB`，HAN 旧快照约 `135.8 MB`、修复 staging 约0；历史 run 归档候选约 `7.248 GB`，日志约 `1.46 MB`，合计 `10,100,359,168 B`（`10.10 GB / 9.41 GiB`）。建议归档根为 `/data/user_data/cy/Digital_twin/GNN_archive/wss_min/20260716_v4_pointnet_acceptance/`；本轮只盘点，不打包、不删除。活动 AG/AAA、v4 stats、当前 v4 runs 与 common-test15 锚点继续禁止删除。

## 2026-07-16｜AG/AAA v4 E2-GLOBAL 完训公平评估 ✅DONE｜v4 No-Go｜AAA 小幅增益但高尾未解

**本次主要修改**：只读收敛 Job `9138` / `9140` 的400 epoch、best/last 全云 train+test15 评估和 best PostView；用预注册的 `ckpt_best(train_loss)` 作主结果，`last` 仅作敏感性。公平对照仅使用旧 E2 已保存预测纯后处得到的 common-test15，没有重训、重推理或用 test 反选 checkpoint。

**对应产物**：`training_wss_min/runs/pointnet_v4/outputs/{ag_v4_e2_global_fps2000,ag_aaa_v4_e2_global_fps2000}/`；结果真源 `data_wss_min/pipeline_reports/v4_cutover_20260715_1921/v4_e2_global_result_analysis.json`；旧锚点为 `training_wss_min/runs/pointnet_distribution_matrix/outputs/e2_global/eval/ckpt_best/common_test15_from_saved_predictions/` 。

**推进到实验步骤**：Job `9138` / `9140` / 延迟监控 `9141` 均 `COMPLETED (0:0)`；AG 用时 `01:18:25`、best epoch `364`、loss `0.1703`，混合用时 `06:14:49`、best epoch `395`、loss `0.1353`。两套 best/last 评估齐全，PostView 均为15/15病例、15/15 manifest bundle、60份 VTP，mapping coverage 最小100%；日志无 NaN/OOM/Traceback/路径或 frame 失败。

**当前状态判断**：旧 v3 E2 common-test15 锚点为 field/case-mean `R²=0.2597/0.1928`、MAE/RMSE `2.754/4.902 Pa`；AG-v4 best 为 `0.1852/0.1465`、`2.847/5.143 Pa`，同病例口径明显回退，v4 单独重跑判 **No-Go**。混合 best 为 `0.2086/0.1572`、`2.772/5.068 Pa`，相对 AG-v4 的 field/case-mean R² 增加 `+0.0234/+0.0106`，15例中8例改善，Spearman `0.6788→0.7041`、top10 IoU `0.1314→0.1575`；但 high-WSS R² `-1.8279→-1.8473`，p99 幅值比 `0.4873→0.4263`、动态范围比 `0.2147→0.1540`，且仍未追平旧锚点。AAA 密网格还贡献了混合 target stats 约 `78.8%` 的全云点，使 log-WSS 统计从 `1.122±1.096` 变为 `0.553±1.373`，这是高幅值压缩的一个待单变量验证混杂因素。结论是 AAA 带来小幅整体/排序/热点定位增益，但高 WSS 幅值压缩更重；`LI_SHU_KUN` 在混合后病例 R² 下降约 `0.293`，下一步优先做病例/域平衡统计、高尾加权与困难病例诊断，不因 last 的小幅 test 改善改变主结论，也不优先切 random-5000。ILO 未处理；旧快照/staging 仍保留，待用户验收并指定归档位置后再清理。

## 2026-07-16｜AG/AAA v4 发布、数值门禁与 E2-GLOBAL GPU 提交 ✅DONE｜后续状态见上条

**数据发布**：发布前复核旧活动 AG 84 bundle + 84 report 共168个 SHA-256 全部匹配终签清单；事务切换后活动 AG 为76例且全部 `stl_landmarks_v4`，`WANG_DENG_FENG` 缺席、`LI_ZHEN_SHAN` 存在。旧84例完整保存在 `data_wss_min/_snapshots/AG_legacy_v3_20260716_signedoff/`，原77例 staging 未删除，`promotion_record.json` 已提交。`AAA/unruputer/HAN_JIAN_FU` 的旧 bundle/report 已做病例级 SHA-256 快照，修正版已原子换入；从原始81个时间步复核 82417 节点，step1120 精确坐标重映射后节点ID、坐标、WSS、pressure 逐点误差均为0。

**训练门禁与协议**：几何终签真源仍为 AG76/AAA63。对139个签核候选扫描 frame、shape、节点、NaN/Inf、负值、全零、非正比例、peak/all 分位数、时间突峰、入口波形、压力和哈希，hard failure=0。派生训练白名单额外排除 AAA 六例：既有 denylist `CHEN_FU`、`SU_KAI_LI`、`ZHANG_GUI_HUA`，以及入口波形量级异常 `ZHANG_ZAO_SHUAN`、`GUO_YU_YING`、`WANG_SHUN_WEN`；AAA 入训57。高 WSS 稀疏尾部病例保留观察，pressure gauge offset 只记录不自动排除。新 split 为 AG `61/0/15`、混合 `118/0/15`（AG61+AAA57），统计只写 `data_wss_min/fold_stats/v4/`。旧 E2 best 已从保存的逐点预测纯后处理重汇总 common-test15，没有重新推理/训练。

**代码/验证**：`DataConfig` 新增向后兼容的显式 `data_root`/`required_frame_version`，split loader 同时支持旧 AG 短 ID 与 canonical AG/AAA ID，并拒绝非法 ID、重复和 partition 泄漏；train/evaluate/PostView 传递显式路径与 frame。AG promotion 增加切换后验证/记录失败自动回滚测试。base pytest 的 cutover 11/11 通过；base 环境因没有 `torch_geometric` 不能收集训练测试，GNN 环境 direct unittest 42/42 通过；Python 编译、Slurm `bash -n`、全量加载、统计病例哈希与 RTX4090 AMP 前后向通过。

**GPU 作业**：两份 E2-GLOBAL/FPS-2000 配置已独立提交：AG-v4 Job `9138` 已连续监控10.18分钟，日志确认 train61、RTX4090、推进至 epoch50，当前/期间最优 loss=`0.3539/0.3333`，无 NaN/OOM/Traceback/path/frame 错误；AG+AAA-v4 Job `9140` 已提交，等待 GPU 资源，Slurm 自行调度。初次 pending Job `9139` 在观察名单路径勘误后、尚未启动前安全取消并由重新预检通过的 `9140` 替代。由于 `9140` 尚未得到 GPU，另挂只读监控 Job `9141`（`after:9140+10`），在其实际启动10分钟后自动核对 CUDA、train118、epoch推进与异常关键字。正式作业均配置训练后自动运行 best/last 全云 train+test15 评估及 best PostView。本轮已启动训练但未等待最终结果，也未分析临时指标；ILO 未处理。

**保留/清理边界**：本轮保留旧 AG84 快照、AG77 staging、AAA/HAN 病例快照与 fixes staging。只有次日结果验收并指定归档位置后，才允许按保留清单打包、复核归档 SHA 再删除旧路径。

## 2026-07-16｜AG/AAA v4 人工终签、最终白名单与发布合同 ✅READY｜未切换

**本次主要修改**：固化用户逐图审核结论：排除 `AAA/unruputer/CAO_DIAN_HE`、`AAA/ruputer/LIU_YU_MING`、`AG/slow/WANG_DENG_FENG`；`AG/fast/LI_ZHEN_SHAN` 的 `CROP` 明确放行，其余软标记病例全部放行。新增 `finalize-review` 将人工决定合并到最终 manifest/白名单/排除清单；发布命令改为从原 77 例 staging 按终签白名单物化 76 例 AG 候选目录，避免误发布 `WANG_DENG_FENG`。新增终签图，只把 3 个排除病例标红，所有已放行病例标黑。

**对应代码/产物**：`pipeline_wss_min/{v4_cutover.py,visualize_v4_cutover.py,tests/test_v4_cutover.py}`；`data_wss_min/pipeline_reports/v4_cutover_20260715_1921/{manual_review_decisions_20260716.json,v4_final_dataset_manifest.*,v4_final_whitelist.json,v4_final_exclusions.json,v4_manual_review_summary.json,v4_soft_flag_definitions.md}`；`assets_新队列审计/alignment_v4_cutover_signedoff_20260716/`。

**推进到实验步骤**：最终 AG=`76`、AAA=`63`，待审=`0`，硬 QA 失败=`0`，`ready_for_promotion=true`。10 项 v4 cutover 回归测试、Python 编译和终签图检查通过；其中已在临时目录完整演练“旧 84 例快照 + 原 staging 77 例保留 + 白名单 76 例原子提升”。

**当前状态判断**：人工终签已完成，但活动 `data_wss_min/AG` 仍为旧 84 例，`promotion_authorized=false`，本条未执行原子切换。`WANG_DENG_FENG` 原属历史 test16；排除后 v4 为 train61/test15，配对比较必须把旧 E2 也重汇总到 common-test15。

## 2026-07-15｜v4 人工审核图按 20 例分组与红黑姓名标记 ✅DONE

**本次主要修改**：在 bundle 直读的 cutover 可视化中新增 AAA/AG 每 20 例分组审核。每组同时生成固定毫米坐标的 X–Z 逐例小图和仿 `05_AG_AAA_v4_common_mm_ortho_overlay.png` 的 X–Z/Y–Z/X–Y 三视图叠加图。按用户反馈，待审与过审病例改为按组容量分层随机混排，固定 seed 保证可复现；待审姓名标红，过审姓名标黑，三视图图例保留逐例颜色与软标记。

**对应代码/产物**：`pipeline_wss_min/visualize_v4_cutover.py`；`assets_新队列审计/alignment_v4_cutover_review_20260715_1921/{grouped20_AAA,grouped20_AG}/` 共 16 张图，`grouped20_review_index.csv` 记录 142 例的分组、顺序、审核状态和图件路径。

**推进到实验步骤**：AAA 65 例分 4 组（20/20/20/5），21 个红名待审病例按容量比例分散到各组，再与 44 个黑名过审病例组内随机交错；AG 77 例同样分层随机混排为 4 组。逐图检查红黑标色、姓名、共同坐标范围和图例均正常。

**当前状态判断**：仅扩充人工审核材料，未修改 bundle、QA 结论或发布状态；`promotion_authorized=false`，仍等待用户逐组签核。

## 2026-07-15｜E4 deeper PointNet Job 8999 结果审核与配对回填 ✅DONE｜No-Go

**本次主要修改**：只读审核 Job `8999` 的 Slurm 状态、训练日志、checkpoint、best/last 全点评估和 best test16 PostView；将 `E4-DEEP-GLOBAL` 与同协议 `E2-GLOBAL` 做单变量配对，回填物理/归一化 R²、train−test gap、MAE/RMSE、high-WSS、top10 幅值比/IoU、Spearman 和双 self-max。本条同时保留前置实现事实：E4 结构为 `6→64→128→256→512；1024→512→256→128→64→1`，PointNet++ 三层 SA foundation 为 `500/125/32`、radius `0.05/0.10/0.20`、nsample=16。本次未修改训练/评估代码，未提交新作业，未启动 PointNet++ fine-tune。

**对应代码/文档**：`training_wss_min/{baseline_models.py,tools/visualize_pointnetpp_sa.py}`、`training_wss_min/configs/{pointnet_deeper/,pointnetpp_sa_foundation/,sweeps/pointnet_deeper_e4.txt}`、`training_wss_min/cluster/pointnet_deeper/`、`training_wss_min/tests/test_pointnet_deeper_sa.py`、`例子/06_PointNet++_SA三层采样与分组/`；[PointNet baseline 实验矩阵 §4.4](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#44-导师追加深度探针-e4-deep-global2026-07-15-完训完评no-go)、[WSS 训练实验跟踪](WSS最小化_训练实验跟踪.md)、`training_wss_min/configs/README.md`、`docs/README.md`；结果位于 `training_wss_min/runs/pointnet_deeper/outputs/e4_deep_global/`，日志为 `training_wss_min/cluster/logs/wsspn_e4_deep_8999.{out,err}`。

**推进到实验步骤**：Job `8999` 已 `COMPLETED (0:0)`，用时 `01:37:11`；400 epoch 齐全，best=第 379 epoch / train loss `0.121590`。`ckpt_best/last`、best/last train61+test16 评估均齐全，best PostView 16/16 病例 manifest 引用文件无缺失，mapping coverage 全为 100%，self-max/top10 图均为 16/16；日志无 Traceback/OOM/NaN。

**当前状态判断**：E4 相对 E2 将 best train loss 降低 29.5%，train 物理 `R²_cb` 从 0.6124 升至 0.6966；但 test 物理 `R²_cb` 从 0.2140 降至 0.1629，gap 从 0.3985 扩至 0.5337。test 归一化 `R²_cb` 从 0.4606 降至 0.4476，物理 MAE/RMSE 从 `2.8005/5.1141` 升至 `2.8510/5.2778`，high-WSS R² 从 −1.486 降至 −1.680，top10 幅值比从 0.378 降至 0.333。16 例物理 R² 为 6 例改善、10 例退化。虽然 top10 IoU `0.146→0.153` 和双 self-max `R²_cb −4.857→−4.208` 小幅好转，self-max 仍 16/16 负例，不足以抵消整体泛化恶化。结论为 **深度探针 No-Go**：保留 E2 锚点，停止纯 PointNet 深度扫描。best/last 同结论；test16 已被反复使用，只作导师驱动配对证据，不表述为无偏最终测试。

## 2026-07-15｜AG/AAA v4 staging、硬 QA 与可视化签核包 ⏳等待人工签核

**本次主要修改**：在预处理、批量报告、QA 和可视化入口增加显式 `--out-root`，bundle/report 改为临时文件后原子替换；新增 `v4_cutover.py` 的 staging 重建、bundle 硬门、旧版 SHA-256 清单和签核后原子提升命令，以及直接读 bundle 的 v4 对齐审核图。为处理 AAA `HAN_JIAN_FU` 唯一时间步的节点顺序循环错位，增加严格的坐标集一对一重映射；只有坐标集完全一致时才启用，真正移动网格仍会拒绝。

**对应代码/文档**：`pipeline_wss_min/{config.py,preprocess.py,reporting.py,run.py,qa_gate.py,v4_cutover.py,visualize_v4_cutover.py}`、`pipeline_wss_min/cluster/run_ag_v4_{staging,finalize}.slurm`、`pipeline_wss_min/tests/test_v4_cutover.py`、[pipeline README](../../pipeline_wss_min/README.md)、[新队列审计](新队列数据可用性审计_AAA_ILO_2026-07-10.md)、[PointNet baseline 矩阵](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)。数据与审核产物位于 `data_wss_min/_staging/ag_v4_20260715_cutover/`、`data_wss_min/_staging/aaa_v4_20260715_fixes/`、`data_wss_min/pipeline_reports/v4_cutover_20260715_1921/` 和 `assets_新队列审计/alignment_v4_cutover_review_20260715_1921/`。

**推进到实验步骤**：Slurm `9007` 完成 AG 正式 77 例 v4 staging 重建，`9085` 完成 AG+AAA 终检与图件。AG `77/77`、AAA `65/65` 通过 bundle 硬门；旧 AG 84 例共 168 个 bundle/report 已生成路径、大小和 SHA-256 快照清单。可视化包含新旧 AG 对照、AG/AAA 共同毫米坐标与归一化三视图、`07_frame_metrics.csv/json` 以及所有软标记专项页。本轮没有训练、没有生成 v4 统计。

**当前状态判断**：已到达可视化人工签核门，不是已发布状态。硬 QA 零失败，但仍有 AG 1 例（`LI_ZHEN_SHAN`）和 AAA 21 例需人工确认软标记；`HAN_JIAN_FU` 修正 bundle 仍仅在 AAA fix staging。活动 `data_wss_min/AG` 仍为旧 84 例，`promotion_authorized=false`；未获用户明确签核前禁止执行 `promote`。签核后才能切换为 AG 77 例 v4，随后用独立 v4 实验 ID 配对复跑 `AG-v4 E2-GLOBAL`。

## 2026-07-15｜PointNet self-max 指标回填与下一轮优化讨论清单 ✅DONE

**本次主要修改**：按导师补充口径新增 `WSScfd/WSScfd,max`、`WSSpred/WSSpred,max` 及二者误差字段；扩展 PostView 的 CSV/VTP/manifest 和共享色标三联图。利用既有完整壁面 true/pred 对五组共 80 个 test case 做纯后处理回填，没有重新训练或运行模型前向。同步把 self-max 完整指标、负 R²/负预测点解释和下一阶段 P0–P5 优化方向写入 PointNet 矩阵状态真源。

**对应代码/文档**：`training_wss_min/tools/export_wss_postview.py`、`training_wss_min/tools/backfill_postview_selfmax.py`、`training_wss_min/tests/test_pointnet_distribution_matrix.py`；[PointNet baseline 实验矩阵 §4.3/§9](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)。汇总产物：`training_wss_min/runs/pointnet_distribution_matrix/outputs/selfmax_test_summary.json`；每例图：`postview/ckpt_best/test/<case>/plots/fig_wss_selfmax_triptych.png`。

**推进到实验步骤**：五组 80/80 图、80/80 manifest、240 个 CSV 和 320 个 VTP 字段校验通过；14 项 PointNet 矩阵单元测试通过。当前完成结果归档与下一轮讨论预备，未生成新配置、未提交新 Slurm 作业。

**当前状态判断**：五组 self-max 的 16/16 逐病例 R² 均为负；E2-GLOBAL pooled R² 相对最好但仍为 −4.802，说明主要瓶颈不只是绝对幅值低估，病例内空间型态也明显失配。CASE 的线性输出产生负点（E2/E3 病例平均 7.34%/6.03%），应保留负值比例作为物理有效性护栏。下一轮先冻结 train61 内开发协议并审计推理期可得 BC/病例级信息，再讨论 shape/scale 双头、热点 loss 和局部拓扑表示；不自动重跑 random-5000 或原样 case-max。

## 2026-07-15｜给老师补齐 NMAE 表 + R² regression 图 ✅DONE

**本次主要修改**：原汇总 xlsx 只有 MAE/RMSE、无 NMAE，也无 true–pred 回归图。新增脚本 `docs/03-汇报材料/tools/build_pointnet_matrix_nmae_r2_report.py`，按 `NMAE=MAE/(max−min)` 汇总正式矩阵 6 组（物理 Pa + 归一化），并生成 hexbin 回归图；回填 `WSS_PointNet实验矩阵与结果汇总.xlsx`（新工作表 `NMAE与R2`，主表插入 NMAE 列）。E0 无 PostView，用 `model.eval()` 重推理补齐。

**对应代码/文档**：图与 CSV → `docs/03-汇报材料/figures/WSS_PointNet矩阵_NMAE与R2_20260715/`；xlsx 同上。

**推进到实验步骤**：汇报材料可直接发给老师（表 + 单 run 图 + 汇总网格）。

**当前状态判断**：物理 NMAE（range）因高峰点分母大而数值偏小，须与 R²/热点图一起看；CASE 的 NMAE≈MAE（分母≈1），不可与 GLOBAL log-z 的 NMAE 横比。

## 2026-07-15｜PointNet 分布矩阵五组结果、PostView 与归一化结论回填 ✅DONE

**本次主要修改**：只读核验 Slurm Jobs `8976–8980`、训练日志、checkpoint、best/last 全点评估和全部 test16 PostView 后，完成 E2/E3/E23 与 GLOBAL/CASE 的配对分析。五个 Job 均 `COMPLETED (0:0)`，无 traceback/OOM；5/5 run 均有 best/last train61+test16 指标，best 共 80/80 个病例包完整，VTP surface mapping coverage 100%。抽查 `WANG_DENG_FENG` 与 `GONG_HUI_XIA` 的 top10 overlay，确认定量热点结论与空间图一致。

**对应代码/文档**：[PointNet baseline 实验矩阵与进度跟踪 §4.2](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md#42-正式矩阵结果2026-07-15)、[WSS 训练实验跟踪](WSS最小化_训练实验跟踪.md)、`training_wss_min/configs/README.md`、`docs/README.md`；产物位于 `training_wss_min/runs/pointnet_distribution_matrix/outputs/`。本次未修改训练/评估代码，未提交新作业。

**推进到实验步骤**：E2-GLOBAL、E3-GLOBAL、E23-GLOBAL、E2-CASE、E3-CASE 全部完成训练 → best/last eval → best test16 可视化。预注册主模型仍为 `ckpt_best(train_loss)`，没有根据 test 指标反选 last。

**当前状态判断**：导师宽网 E2 是本轮最强相对改进，test 物理 `R²_cb=0.2140`，比 E0 的 0.1414 增加 0.0725；random-5000 单独增益弱，E23 没有超过 E2，说明容量是主要因素且未见点数协同。`WSS/WSSmax` 在 E2/E3 下将 normalized `R²_cb` 分别从 0.4606/0.4218 降至 0.1724/0.1551；E3-CASE 虽提高 top10 IoU，但整体 R²、Spearman 和动态范围退化，CASE 主结论为 **No-Go**。E2 的 high-WSS R² 仍为 −1.486、top10 幅值比仅 0.378，因此本矩阵不判为可部署 Go；test16 已被连续用于探索性比较，后续也不能把它表述为无偏最终泛化估计。

## 2026-07-14｜PointNet baseline 矩阵文档同步 `E0`/`8970` 结案 ✅DONE

**本次主要修改**：更新 [PointNet baseline 实验矩阵与进度跟踪](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)：`E0-GLOBAL` 标为完训+完评并写入 §4.1 指标表；`8970` 标为补充 No-Go；进度表/执行顺序去掉「排队/运行中」；注明 §6 完整可视化合同从 `E2/E3` 起补齐。

**推进到实验步骤**：矩阵真源已与实测一致；下一跳仍是实现/提交 `E2-GLOBAL`。

**当前状态判断**：文档状态与推进记录文首分析对齐。

## 2026-07-14｜`8968` train+test16 与 `8970` wide128 结果分析 ✅DONE

**本次主要修改**：只读复盘完训/完评：`8968` 事后 eval `8974`/`8975`（`train,test`），以及宽网探针 `8970`（val-only）。产物：`runs/pointnet_trainloss_e400/outputs/{pointnet_xyz,pointnet_xyzgeom}/eval/`、`runs/pointnet_wide128/outputs/pointnet_xyzgeom/eval/`。

**关键指标**（主指标 `R²_field_cb`；**禁止**把 test16 与 2×3 的 val8 当成同一列比）：

| Run | 分区 | R²_cb | R²_raw | case mean/med/P10 | 负例 | RMSE/MAE | high-WSS R² | top10 比/IoU |
|---|---|---:|---:|---|---|---:|---:|---|
| 8968 PN·xyz | train61 | 0.5704 | 0.5677 | 0.505/0.522/0.388 | 0/61 | 4.32/2.09 | −0.04 | 0.635/0.435 |
| 8968 PN·xyz | **test16** | **0.0825** | 0.0832 | 0.071/0.033/−0.143 | **5/16** | 5.60/3.10 | **−1.88** | 0.284/0.095 |
| 8968 PN·xyz+geom | train61 | 0.5303 | 0.5277 | 0.454/0.462/0.326 | 0/61 | 4.51/2.19 | −0.20 | 0.581/0.415 |
| 8968 PN·xyz+geom | **test16** | **0.1414** | 0.1433 | 0.106/0.074/−0.062 | **5/16** | 5.42/2.95 | **−1.76** | 0.328/0.174 |
| 8970 wide128·xyz+geom | val8 | 0.3071 | 0.3167 | 0.173/0.169/−0.007 | 1/8 | 4.10/2.30 | −1.17 | 0.406/0.224 |
| 2×3 PN·xyz+geom（锚） | val8 | 0.3015 | 0.3122 | 0.184/0.140/+0.014 | 1/8 | 4.11/2.29 | −1.17 | 0.457/0.235 |

**判读**：
1. **`8968` 过拟合**：train≈0.53–0.57，test16 仅 0.08–0.14（gap≈0.39–0.49）；无 val + train_loss 选模不适合作精度主线。
2. **几何略优泛化**：test 上 xyz+geom 比纯 xyz **+0.059**；后续保留 xyz+geom。
3. **高 WSS 仍崩**：test high-WSS R²≈−1.8，top10 比~0.3。
4. **`8970` 加宽 No-Go**：val 0.3071 vs 锚点 0.3015（Δ≈+0.006），容量单变量无实质增益。

**Go-NoGo**：`8968` 作可部署精度 **No-Go**（test 过低）；作 train-fit 容量信号有。`8970` 相对锚点 **No-Go**。

**下一步**：按 PointNet baseline 矩阵推老师通道 / 5k random；不把 8968 test16 当新追分基线。

**推进到实验步骤**：8968/8970 数值已结案；正式 `E2/E3` 尚未提交。

**当前状态判断**：瓶颈在泛化与高 WSS，不在能否压低 train loss。

## 2026-07-14｜PointNet baseline 实验矩阵与归一化讨论跟踪建立 📄

**本次主要修改**：新建 PointNet baseline 独立跟踪文档，冻结后续只跑 `PointNet+xyzgeom`、无 val/无早停、5000 点 random 不放回且每 epoch 重采样；将容量/采样两条父实验各自拆成全局 log-z 与逐病例归一化配对。记录逐病例 `WSS/WSSmax` 与逐例 log-z 的待冻结差异，并定义归一化空间分布、top10 high-risk 定位和逐 test case 产物口径。

**对应代码/文档**：[PointNet baseline 实验矩阵与进度跟踪](WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md)、[WSS 训练实验跟踪](WSS最小化_训练实验跟踪.md)、`docs/README.md`、根 `README.md`及本记录。本次未改训练代码或配置。

**推进到实验步骤**：`8968`/`8970`/`8974`/`8975` 数值已出（见文首）；正式 `E2-GLOBAL` / `E3-GLOBAL` 尚未实现或提交。

**当前状态判断**：实验矩阵合同已建立；旧探针结案后进入矩阵正式父实验。逐病例公式未冻结前不启动 `E2-CASE` / `E3-CASE`。

## 2026-07-14｜`8968` train+test16 事后 eval 已提交 `8974`/`8975` ✅COMPLETED

**本次主要修改**：取消仅 train 的 `8972`/`8973`；改为一次评 `train,test` 并解锁 `--allow-test`。作业 `8974`/`8975` 均 `COMPLETED (0:0)`；数值见文首分析条。

**当前状态判断**：eval 产物齐全（`metrics.json` 含 train+test）。

## 2026-07-14｜`8968` train-fit 事后 eval 已提交 `8972`/`8973` ↪已取消，改 `8974`/`8975`

**本次主要修改**：对 `pointnet_trainloss_e400` 两臂提交完整壁面 **train** 分区评估（不读 val/test）。后因需保留指标并做 test16，已 `scancel`，由上条 `train,test` 作业替代。

**当前状态判断**：已取消，不作为有效产物。

## 2026-07-14｜PointNet train_loss×400 完训分析 `8968[0-1]` ✅完训｜待 eval

**本次主要修改**：只读复盘作业 `8968[0-1]`（无 val / `selection_rule=train_loss` / 400 epoch / split `61/0/16`）。两枪均 `COMPLETED`：`pointnet_xyz` ~84.1 min，`pointnet_xyzgeom` ~85.6 min；均跑满 400 epoch，无早停。产物：`training_wss_min/runs/pointnet_trainloss_e400/outputs/{pointnet_xyz,pointnet_xyzgeom}/`（`ckpt_best.pt`、`history.jsonl`、`history.png`）。

**关键指标**（仅 train_loss；本协议训练期不写 R²，test16 未读）：

| 臂 | best train_loss | best epoch | last loss | late50 mean±std |
|---|---:|---:|---:|---|
| PointNet · xyz | 0.2164 | 379 | 0.2242 | 0.2261±0.0060 |
| PointNet · xyz+geom | **0.2105** | 383 | 0.2167 | 0.2206±0.0051 |

- 几何臂在 400 epoch 中 **91.75%** 步 train_loss 低于纯 xyz；全程均值差约 `+0.010`（xyz−geom），末 100 epoch 约 `+0.006`。
- 曲线：两臂均前 ~20 epoch 陡降，其后缓慢下行至 ~0.21–0.23；末段仍有小幅下降与 batch 抖动，未完全平台。
- 对照冻结 2×3（**不可直接比 R²**）：同架构 baseline 在 val 选模下 `xyz` best train_loss≈0.257（150 ep）、`xyz+geom` 早停 70 ep 且 best val 落在 e9（当时 train_loss≈0.43）。8968 把原 val8 并入 train 并强制训满 400，train fit 更深属协议预期，不构成泛化 Go。

**Go-NoGo**：待定 — 缺泛化指标。本枪只能证明「无 val + train_loss 选模 + 400 ep」可稳定完训，且几何输入在 train fit 上仍略优；**不能**相对 baseline `R²_field_cb=0.3015` 裁决升降。

**下一步**：
1. 开发侧：`evaluate --partitions train` 报 train-fit 天花板（不碰 test）。
2. 获批后再 `--partitions test --allow-test` 做一次 test16 终评。
3. 勿用原 val8 当 held-out（已并入 train61）。

**对应代码/文档**：本条；[训练实验跟踪](WSS最小化_训练实验跟踪.md)；日志 `training_wss_min/cluster/logs/wss_pn_tl_8968_{0,1}.{out,err}`。

**推进到实验步骤**：训练完成；正式 R² / 物理误差尚未写出。

**当前状态判断**：完训健康；结论卡在事后 eval。

## 2026-07-14｜导师建库脚本改为读 bundle.npz（去掉 ascii 全流程） ✅DONE

**本次主要修改**：`teacher/build_wss_dataset.py` 改为老师式「读数据→pkl」：从 `data_wss_min/AG/*/bundle.npz` 取 xyz+几何与峰值 WSS，按正式 split 写出 pkl；不再包含 ascii/ascii_in/STL 预处理。交付说明同步。

**对应代码/文档**：`teacher/{build_wss_dataset.py,交付说明_WSS最小化适配老师接口.md}` 及本记录。

**推进到实验步骤**：建库脚本可直接在有 bundle 的环境运行；原始 Fluent 预处理仍走 `pipeline_wss_min`。

**当前状态判断**：与老师「读已有点云再打 pkl」的职责更一致。

## 2026-07-14｜导师交付脚本按老师源码风格重写 ✅DONE

**本次主要修改**：按用户澄清「外形跟老师、内容用我们项目」，重写 `teacher/build_wss_dataset.py` / `model_train_wss.py`：函数骨架与老师 `sampling_buid_dataset` / `model_train_1` 对齐（process_*、get_model/train_step/my_collate、Adam+Plateau、train-loss 存 best）；内容改为 WSS-min 全流程建库、6 维 xyz+几何、log_z WSS、FPS-2000。去掉 argparse 工程壳。交付说明改为「风格对齐」表述。

**对应代码/文档**：`teacher/{build_wss_dataset.py,model_train_wss.py,交付说明_WSS最小化适配老师接口.md}` 及本记录。

**推进到实验步骤**：可与老师原版并排审阅；尚未实跑建库/训练。

**当前状态判断**：交付形态符合「老师风格模板 + 本项目适配」。

## 2026-07-14｜PointNet 加宽容量探针（width=128，xyz+geom）⏳已提交 `8970`

**本次主要修改**：相对冻结 2×3 最佳格 `PointNet+xyz+geom`，仅加宽通道（`width=32→128`、`head_hidden=64→256`，局部 `in→128→256→512`，相对老师 `256→512` 多一层 128 过渡；`dropout=0`），其余协议不变（FPS-2000、val-only、seed=1234、未加权 MSE）。不重跑 preprocess。后续计划：代码严格对齐老师通道表后再提交两枪——(A) 老师对齐 + FPS-2000；(B) 老师对齐 + 5k random / 每 epoch 重采。

**对应代码/文档**：`training_wss_min/configs/pointnet_wide128/pointnet_xyzgeom.json`、`training_wss_min/configs/sweeps/pointnet_wide128.txt`、`training_wss_min/cluster/pointnet_wide128/`、`training_wss_min/configs/README.md` 及本记录。

**推进到实验步骤**：作业 `8970` 已提交；产物将写入 `training_wss_min/runs/pointnet_wide128/outputs/pointnet_xyzgeom/`；对照基线 `baseline_2x3_simple/outputs/pointnet_xyzgeom`（`R²_field_cb=0.3015`）。

**当前状态判断**：第一枪只隔离「加宽」杠杆；老师严格对齐与 5k 随机采留待第二批，避免与容量效应混杂。

## 2026-07-14｜导师两文件交付代码落地（建库全流程 + PointNet 读 pkl） ✅DONE

**本次主要修改**：用户确认交付口径后，在 `teacher/` 落地 `build_wss_dataset.py`（Fluent/STL/中心线全流程 → `wss_{train,val,test}.pkl` + log-z 统计，特征 6 维 xyz+几何）与 `model_train_wss.py`（老师式 get_model/collate/采点循环；FPS-2000；AdamW+warmup+cosine；完整壁面 val；case-balanced R² 或 train_loss 选模）。同步更新交付说明为已实现状态。`py_compile` 通过。未跑全量建库/训练（需集群与 conda）。

**对应代码/文档**：`teacher/{build_wss_dataset.py,model_train_wss.py,交付说明_WSS最小化适配老师接口.md}` 及本记录。

**推进到实验步骤**：交付脚本可发给老师审阅；试跑可用 `--limit 1`（建库需 `GNN_vmtk`/vtk，训练需 `GNN`）。

**当前状态判断**：两文件接口与文档口径一致；全量 77 例建库应走集群，勿在登录节点直接跑。

## 2026-07-14｜导师两文件交付口径说明（pkl 建库 + 训练）已定稿 📄待实现代码

**本次主要修改**：在 `teacher/` 写清发给导师的交付说明：保留老师「建库脚本 + 训练脚本」两文件形态；建库须 WSS-min **全流程**（原始 Fluent/STL/中心线 → 配准/归一化/几何/QA → pkl）；特征为 **xyz+几何（6 维）**，标签为峰值壁面 WSS；训练脚本模仿老师单文件读 pkl / PointNet / 采点循环，协议采用本项目 log-z、FPS-2000、case-balanced R² 等。老师原版 `sampling_buid_dataset.py` / `model_train_1.py` 保留对照；适配代码 `build_wss_dataset.py` / `model_train_wss.py` 标为待实现。

**对应代码/文档**：`teacher/交付说明_WSS最小化适配老师接口.md`、`teacher/{sampling_buid_dataset.py,model_train_1.py}` 及本记录。

**推进到实验步骤**：交付口径与 pkl 协议已文档化，可直接发给老师确认；尚未落地两份适配脚本、未重跑 preprocess/训练。

**当前状态判断**：用户确认输入 xyz+几何、建库全流程、训练内容用本项目协议后，下一步是在 `teacher/` 实现上述两脚本。↪ 已由上条落地代码替代。

## 2026-07-14｜导师展示 train / pipeline 改为真正自包含单文件 ✅DONE

**本次主要修改**：按导师“全部功能压缩到一个文件”的要求，重写两份展示代码并删除对项目内部模块的调用。PointNet 文件内直接实现 JSON 配置解析、bundle/split 读取、train-only 几何统计、log-z、确定性 FPS、DataLoader、PointNet、AMP、MSE/AdamW、warmup+cosine、梯度裁剪、完整壁面验证、case-balanced R²、早停和 best/last checkpoint。预处理文件内直接实现 Fluent ASCII/中心线/STL 读取、入口峰值选择、节点 ID 对齐、单位换算、中心线平移修复、STL 解剖坐标架、未描入口裁剪、坐标归一化、近壁标注、中心线几何特征、全时间步 WSS/压力/矢量堆叠、QA、冻结 log-z 与 FPS 峰值样本；展示产物只写 `outputs/`，不覆盖正式 bundle。两份代码的说明性注释均使用中文。

**对应代码/文档**：`training_wss_min/examples/pointnet_baseline_train.py`、`pipeline_wss_min/examples/preprocess_pipeline.py`、两侧相关测试与 README，以及本记录。

**推进到实验步骤**：训练文件 224/250 行，使用冻结 baseline JSON 完成 1 epoch 的 53 例训练和 8 例完整壁面 val 实跑，成功写出 loss、指标和 best/last checkpoint；预处理文件 248/250 行，使用 included 病例 `AG/fast/ZHANG_HAO` 从 81 个原始时间步实跑，生成 12,521 壁面点、770,914 内部点的 bundle、QA 报告和 FPS-64 峰值样本。训练/预处理共 29 项单元测试、v4 配准合成测试、编译、CLI、内部 import 禁止项和 PointNet 冻结实现输出一致性均通过；临时产物已清理，未改正式数据、未访问 test16。

**当前状态判断**：两份文件现在都可脱离仓库内部 Python 包独立审阅和执行，只依赖通用第三方库、JSON 配置/split/stats 与原始/预处理数据。生产级模块化入口继续保留用于正式实验，但老师看到的文件不再把核心逻辑藏在调用后面。

## 2026-07-14｜`pipeline_wss_min` 目录重构、历史归档与中文注释收口 ✅DONE

**本次主要修改**：将根目录收敛为 10 个正式核心模块；AAA/ILO v4 的白名单、预处理、只读几何审计、bundle 终检、AG 回归和共同坐标可视化统一迁入 `new_cohorts/`，并把重复的白名单常量与 `unit_id` 解析合并到 `common.py`。2026-07-07 至 2026-07-08 的 AG flow-divider/LR/STL/居中 QA 脚本迁入 `archive/alignment_v3/`，统一冻结为 `legacy_centerline` 且不保留旧根模块兼容层。`run.py` 改为基于 dataclass 复制构造命令行配置，避免采样覆盖项原地污染全局 `DEFAULT`；保留源码的说明性注释/docstring 和用户可见提示统一为中文，技术字段名与文件格式名保持不变。删除 494 个已完成 Slurm `.out/.err`、275 个已完成本地 WSS-min 日志和全部 Python 缓存；所有 bundle、JSON/CSV 审计、图件、split 和训练产物均保留。

**对应代码/文档**：`pipeline_wss_min/{README.md,run.py,config.py,registration.py,raw_io.py,preprocess.py,reporting.py,examples/,tests/}`、`pipeline_wss_min/new_cohorts/`、`pipeline_wss_min/archive/`、`pipeline_wss_min/cluster/`、根 `README.md`、`.gitignore`、归档交接记录及本记录。

**推进到实验步骤**：工程重构与回归验证完成；`compileall` 通过，7 个 WSS-min 单元测试和 5 个 v4 配准合成回归测试通过，3 份 Slurm 脚本通过 `bash -n`，新队列入口仍解析出 171 个双白名单单元。当次展示入口以复用正式模块方式验证；其后已由本文首条记录中的 248 行自包含实现替代。未改写正式 bundle，也未访问 test16。

**当前状态判断**：当前入口、目录职责和代码事实已经一致；AG 正式四阶段继续使用 `pipeline_wss_min.run`，AAA/ILO 使用 `pipeline_wss_min.new_cohorts.*`，历史 AG 坐标 QA 只从归档路径复核。旧命令会直接失败，避免调用者误以为仍在执行当前 v4 口径。

## 2026-07-14｜导师展示用单文件 WSS-min 预处理旧实现 ↪ 已由自包含版替代

**本次主要修改**：新增 229 行的单文件预处理展示入口。默认直接调用正式 `preprocess_case`，完整保留原始 CFD 校验、稳定节点对齐、单位换算、解剖坐标架/配准、裁剪、坐标归一化、壁面标记、几何特征及全时间步 WSS/压力/矢量堆叠；随后在同一文件中显式展示病例 QA、冻结的 train-only peak WSS log-z 统计校验、归一化坐标 FPS 和峰值训练样本构建。入口只接受当前 split 中 included 的 AG train/val/test 病例，并提供 `--skip-preprocess` 安全复用既有 bundle；展示产物仅写入 `outputs/wss_min/teacher_preprocess/`。

**对应代码/文档**：`pipeline_wss_min/examples/{preprocess_pipeline.py,__init__.py}`、`pipeline_wss_min/tests/{test_preprocess_example.py,__init__.py}`、`pipeline_wss_min/README.md` 及本记录。

**推进到实验步骤**：展示代码实现与验证完成；行数门限为 229/250，4 个展示入口单元测试、CLI/编译检查通过。使用既有 included 病例 `AG/fast/ZHANG_HAO` 以 `--skip-preprocess --wall-n 128` 跑通 QA、53 例冻结统计加载和样本构建；v4 配准合成回归测试通过。未重跑正式病例预处理、未重算全局统计、未改写正式 bundle，也未访问 test16。

**当前状态判断**：本条记录的是最初“紧凑入口调用正式模块”的版本，已不符合导师对自包含文件的要求；当前事实以本文首条 248 行自包含实现为准，且展示运行不会更新正式 bundle。

## 2026-07-14｜展示用预处理示例注释改为中文 ✅DONE

**本次主要修改**：将 `pipeline_wss_min/examples/preprocess_pipeline.py` 的模块说明、函数 docstring 与 argparse help 改为中文；逻辑与对外行为不变。

**对应代码/文档**：`pipeline_wss_min/examples/preprocess_pipeline.py` 及本记录。

**推进到实验步骤**：文档可读性调整完成；行数仍 ≤250，相关单元测试通过。

**当前状态判断**：仅注释语言变更，不影响正式 preprocess 路径与训练作业。

## 2026-07-14｜PointNet 无 val / train_loss 选模 / 400 epoch ✅完训 `8968[0-1]`（分析见文首）

**本次主要修改**：新增 train/test-only 划分（原 val8 并入 train→61/0/16），按新 train 重算 WSS 全局统计（不覆盖默认 `wss_global_stats.json`）；`train.py` 支持 `selection_rule=train_loss`（按 epoch 训练损失选 best，`early_stop_patience<=0` 关早停，不加载 val）；冻结 PointNet `xyz` / `xyz+geom` 两份 400-epoch 配置与 Slurm 入口。未重跑 preprocess，未访问 test16。作业 `8968` 已提交（array 0–1）并完训。

**对应代码/文档**：`training/splits/split_AG_wss_min_v1_traintest.json`、`data_wss_min/fold_stats/wss_stats_v1_traintest.json`、`training_wss_min/{train.py,config.py,objectives.py}`、`training_wss_min/configs/pointnet_trainloss_e400/`、`training_wss_min/configs/sweeps/pointnet_trainloss_e400.txt`、`training_wss_min/cluster/pointnet_trainloss_e400/`、`training_wss_min/configs/README.md` 及本记录。

**推进到实验步骤**：训练完成；复盘见文首分析条。

**当前状态判断**：协议与冻结 2×3（val-only）隔离；待 train-fit / 获批 test16 事后 eval。

## 2026-07-14｜导师展示用单文件 PointNet baseline 训练旧实现 ↪ 已由自包含版替代

**本次主要修改**：新增 245 行的单文件 PointNet WSS 训练入口，默认复现 2×3 baseline 中 `PointNet + xyz+geom` 配置，也可通过 `--config` 切换冻结的 `PointNet + xyz`。文件内完整展示 PointNet shared MLP、病例级 max-pool、WSS decoder、FPS-2000 数据加载、MSE/AdamW、warmup+cosine、AMP、梯度裁剪、完整壁面 val、case-balanced R² 选模、早停和 best/last checkpoint；只复用已审计的数据解析与指标公式，明确不读取 test16。

**对应代码/文档**：`training_wss_min/examples/pointnet_baseline_train.py`、`training_wss_min/examples/__init__.py`、`training_wss_min/tests/test_core_refactor.py`、`training_wss_min/README.md` 及本记录。

**推进到实验步骤**：展示代码实现与等价性验证完成；行数门限为 245/250，模型可直接加载正式 `PointNetRegressor` state dict，固定随机输入下输出逐值一致；全套 22 个单元测试与 CLI/编译检查通过。未启动训练、未重评、未访问 test16。

**当前状态判断**：本条记录的是最初仍复用项目数据/指标模块的版本，已不符合导师对自包含文件的要求；当前事实以本文首条 224 行自包含实现为准。冻结 JSON 协议和正式模块化训练入口继续保留。

## 2026-07-14｜`training_wss_min` 核心代码重构与历史入口清理 ✅DONE

**本次主要修改**：按用户确认的保留边界清理训练库：删除全部配置生成器和根目录旧命令兼容 shim，删除 Python 缓存与已完成 Slurm 的原始 `.out/.err`，保留所有 run checkpoint、指标、训练日志、可视化结果、JSON 配置和 manifest。将模型工厂、loss/选模、日志/随机种子从 `train.py` 拆到独立核心模块；训练协议、指标口径和 test 锁未改变。2×3 baseline 的 6 份 JSON 从被忽略的 `runs/` 迁到正式配置目录并新增冻结 manifest/Slurm 入口，不再依赖生成器。

**对应代码/文档**：`training_wss_min/{models.py,objectives.py,runtime.py,train.py,evaluate.py,pointnext.py,README.md}`、`training_wss_min/configs/{README.md,baseline_2x3/,sweeps/baseline_2x3_simple.txt}`、`training_wss_min/cluster/baseline_2x3/`、`training_wss_min/{tools,experiments,tests}/` 及本记录。历史 `dist_to_wall` 配置在 `configs/README.md` 明确标为仅审计保留。

**推进到实验步骤**：工程清理与等价性验证完成；20 个单元测试通过，四种模型合成点云前向/反向通过，17 份 manifest 的 98 个 JSON 引用全部存在。未重训、未重评、未访问 test16。

**当前状态判断**：核心根目录只保留 11 个正式 Python 模块；2×3 baseline、历史 JSON 和现有实验结果均可追溯。后续新增实验需直接提交审查后的 JSON/manifest，不再恢复一次性配置生成脚本或旧入口兼容层。

## 2026-07-14｜2×3 baseline 后处理收窄为最佳/最差两例 ✅DONE，`8966/8967` 均 `0:0`

**本次主要修改**：按用户要求取消全量 `8961[0-5]`（及其依赖汇总 `8962`），清理全量/冒烟输出；postview 作业改为只导出最佳 baseline `PointNet+xyz+geom` 的最佳与最差 val 病例。排序依据是同点 `metrics.json` 的逐病例 R²：`slow/CHENG_LU_LI=0.3959`、`slow/XU_YI_CAI=-0.0577`。同步更新 `postview-surface-viz` skill：已完成 baseline 默认只做 best/worst；全量必须由用户明确要求；`wss/wss(max)` 的 CFD、预测和误差强制共用 CFD max 分母。

**对应代码/文档**：`.cursor/skills/postview-surface-viz/SKILL.md`、`training_wss_min/runs/baseline_2x3_simple/postview/{README.md,run_dirs.txt,val_cases.txt,export_array.slurm,submit.sh}`、[训练实验跟踪](WSS最小化_训练实验跟踪.md)。

**推进到实验步骤**：单例端到端 VTP/mapping QC 冒烟已通过；两例最终导出 `8966` 与自动汇总 `8967` 均完成。两个 VTP 均含 10 个规定数组，Gaussian mapping coverage 均为 100%；`comparison.csv` / `comparison_by_run.csv` / `batch_manifest.json` 已写入。未访问 test16。

**当前状态判断**：最终交付只含两例和 PointNet+xyz+geom 一个模型，避免无意生成 48 个病例包；最佳例 `CHENG_LU_LI` 同点 R²=0.3959，最差例 `XU_YI_CAI`=−0.0577。VTP 标量及同点指标口径保持不变。

## 2026-07-14｜2×3 baseline 壁面 ParaView 可视化与比较 ⏳已提交 `8961[0-5]`

**本次主要修改**：扩展 WSS-min 后处理器，使同一 STL 面片 VTP 同时包含 CFD 真值、预测、signed/absolute error 与 `wss÷wss_max` 归一化显示字段；归一化一律使用同一病例 CFD 壁面最大值。面片回插器同步改为由归一化后的 CFD/Pred 字段重算误差，避免误差被独立二次插值。新增 baseline 6 run × val8 的 array 导出、mapping/QC 汇总器和交付说明。

**对应代码/文档**：`training_wss_min/tools/export_wss_postview.py`、`tools/cfdpost_cloud_export/map_to_stl_surface.py`、`training_wss_min/runs/baseline_2x3_simple/postview/{README.md,export_array.slurm,summarize.py,summarize.slurm,submit.sh}`、[训练实验跟踪](WSS最小化_训练实验跟踪.md)。

**推进到实验步骤**：一例 CPU 端到端冒烟已通过：面片 VTP 含 10 个必须数组，Gaussian mapping coverage=100%。完整导出 array `8961[0-5]` 已提交（最多并发 4），依赖汇总 job `8962` 会在 48 个 VTP 都成功后写入比较 CSV；只读取既有 val run 和其 bundle，未访问 test16。

**当前状态判断**：可视化结果严格用于病例云图和软件检查；R²/Pa 误差继续以同点 `_export/*__wall.csv` 与既有 `eval/metrics.json` 为唯一口径，不在插值 STL 上重算。

## 2026-07-14｜最小 2×3 baseline 全部完成 ✅（Job `8700[0-5]`，6/6 `0:0`）

**本次主要修改**：无新增训练代码；完成并汇总既有 `MLP / PointNet / PointNet++` × `xyz / xyz+geom` 六格的完整壁面 val 评估。模型、输入、AG v1 split、FPS-2000、seed、MSE 和 val-only 协议均保持冻结。

**对应代码/文档**：作业与产物 `training_wss_min/runs/baseline_2x3_simple/`；完整指标、逐病例 CSV、checkpoint 和日志均在各 `outputs/<run>/`；结果表与判读写入[训练实验跟踪](WSS最小化_训练实验跟踪.md)。

**推进到实验步骤**：Slurm array `8700[0-5]` 全部 `COMPLETED (0:0)`，每格完成训练与完整壁面 val 评估；未访问 test16。

**当前状态判断**：PointNet+xyz+geom 为本单 seed 最佳单格（`R²_field_cb=0.3015`，逐病例 P10=+0.014，1/8 负例）；所有 xyz+geom 组都优于相应的 xyz 组。六格 high-WSS R² 均为负且 top10 幅值明显偏低，故它是“最简单形式”的下限基线，非最终模型或可推广的架构排名。

## 2026-07-14｜AAA/ILO 原始 STL 自动解剖坐标架 v4 与 centerline 错位修复 ✅

**本次主要修改**：从第一性原理重建有符号解剖坐标架：由原始 STL 自动选取近端主干、分叉中心和双髂支端点，固定近端主干为 `+Z`、髂支为 `-Z`，用原始 STL 世界 `+X` 对左右轴定号，并保持 `det(R)=+1`。新增 centerline↔壁面纯平移守卫：只在偏移跨过一条血管尺度且平移后最近邻残差通过时修复。壁面拓扑参考改为首步/峰值步/末步三点共识；只允许丢弃极少量非稳定额外节点，稳定节点缺失仍硬失败。

**对应代码/文档**：`pipeline_wss_min/{surface_io.py,registration.py,config.py,preprocess.py,reporting.py,audit_new_cohort_frame.py,audit_ag_frame_v4.py,preprocess_new_cohorts.py,qa_new_cohorts_v4.py,visualize_new_cohort_frame_v4.py}`、`pipeline_wss_min/cluster/{run_new_cohorts_preprocess.slurm,run_new_cohorts_finalize.slurm}`、`tests/test_wss_min_registration_v4.py`、[新队列数据审计](新队列数据可用性审计_AAA_ILO_2026-07-10.md)、`assets_新队列审计/alignment_v4/`。

**推进到实验步骤**：AAA/ILO 数据层干净的 171 单元全量几何门 `171/171` 通过；用户指定的 6 个平移错位单元已修复，并额外发现/修复同类 `ILO/YU_XIANG_SHENG-1/after`，共 7 个。AG included=77 只读回归 `77/77` 通过且无误修复。已修复 ILO 单段 cohort 路径兼容；`ILO/ZHAO_JIAN_PING-0/after` 确认为首步多 1 个瞬态节点，稳定参考选择峰值步 `1162`，只从 `1120` 丢弃 node `111167`。最终作业 `8956` 与终检 `8957` 均 `COMPLETED (0:0)`；汇总 `n_ok=171, n_missing=0, n_other=0`，bundle QA `n_qa_pass=171, n_failed=0`，7 个修复单元与几何审计名单完全一致。

**当前状态判断**：新队列 171 个 bundle/report 已全部入库，仅写 `data_wss_min/AAA/**` 和 `data_wss_min/ILO/**`；未重跑 AG preprocess/global-stats/build-samples，`data_wss_min/AG/**` 当日改写数为 0。已提交的 AG 2×3 实验不受影响。如后续要混合 AG+AAA/ILO，须将 AG 整体重建为 v4，禁止混用 v3/v4 bundle。

## 2026-07-14｜第五轮正式计划与执行计划归档 ✅

**本次主要修改**：确认第五轮已完成诊断性科学结案，但未达到内部工程目标；将正式计划与历史执行计划集中移入 `_archive/WSS最小化/`，不再占用当前推进目录。

**对应代码/文档**：[第五轮正式计划](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)、[第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md)、[第五轮结案说明](_archive/WSS最小化/WSS最小化_第五轮结案与归档说明_2026-07-12.md)、归档 README。

**推进到实验步骤**：第五轮停止新增任务；L0/OOF/T16 按止损决策保持未触发，后续优化统一进入第六轮。

**当前状态判断**：第五轮历史证据和产物路径保留，当前状态、指标和待办以第六轮文档及训练跟踪为准；同步修复了归档后相互引用的相对链接。

## 2026-07-14｜合并 `main` 最新推进与变更文档 ✅

**本次主要修改**：将 GitHub `main` 最新提交 `9254c11`（2026-07-13）中 `docs/02-推进与变更` 的路线计划、审计报告、训练跟踪、历史归档和 CSV 证据合并到本分支；保留本分支已有的 2×3 baseline 等补充记录。

**对应代码/文档**：仅修改 `docs/02-推进与变更/**`；未修改训练代码、配置、数据及其他目录。

**推进到实验步骤**：文档路线统一到 WSS 主线、横向最小诊断矩阵与条件路线的最新状态；新增/更新文件路径已逐项核对，未启动新实验。

**当前状态判断**：文档内容以 `main` 最新审计口径为基线，同时保留本分支实验事实；当前等待既有作业结果，`test16` 仍保持未读。

## 2026-07-13｜最小 2×3 基线（MLP / PointNet / PointNet++ × xyz / xyz+geom）⏳已提交 `8700[0-5]`

**本次主要修改**：训练统一模型工厂新增最基础的逐点 `MLP`、全局 max-pool `PointNet` 和经典 SA+FP `PointNet++`；后者不含 PointNeXt 的残差或倒置瓶颈。新增冻结的 2×3 配置生成器与 Slurm array：统一 AG v1 `53/8/16`、peak WSS 单标量、FPS-2000、seed=1234、未加权 MSE、val-only；禁用旋转增强、采样/几何/目标加权、多任务和 raw-space 辅助项。

**对应代码/文档**：`training_wss_min/{config.py,baseline_models.py,pointnext.py,train.py}`、`training_wss_min/runs/baseline_2x3_simple/{README.md,make_configs.py,train_array.slurm,submit.sh}`、[训练实验跟踪](WSS最小化_训练实验跟踪.md)。

**推进到实验步骤**：6 份配置已生成并通过 JSON 解析；MLP、PointNet、PointNet++ 均完成合成双病例点云的 forward/backward 冒烟，Python 编译与 Slurm shell 语法检查通过。已通过 `submit.sh` 提交 Slurm array `8700[0-5]`（最多并发 4）；每格训练后只做完整壁面 val 评估，产物固定写入该 runs 子目录。

**当前状态判断**：矩阵只改变模型与输入（`xyz` 或 `xyz+abscissa_norm+local_radius+curvature`），可作为后续复杂模型/模块的可解释下限；`test16` 保持未读。当前等待队列调度和六格完成，完成后回填指标。

## 2026-07-12｜第六轮 W0 审计 + W1 因子补全(E) + C 邻域预审计 + W2-L1 loss Gate + W3 组合 ✅DONE
## 2026-07-13｜第六轮最终审查后文档收敛整理 ✅DONE（文档）

**本次主要修改**：
- 将四份第六轮执行文档改写为当前状态页，删除旧版 F3、旧优先级、已完成 W0–W3 的重复计划和 36-run 理论矩阵。
- 冻结唯一口径：固定 peak/WSS-only；`P0-Metric → P0-Density → 2×2 F3 → 分支优化 → repeated validation`。
- 统一指标语义：level、pattern、hotspot、病例等权 Pa 误差分列；旧 `R²_casemean` 不再解释为病例 level R²。
- 保留实验证据在训练跟踪、变更历史在本记录；当前计划只保留有效结论、Gate、暂停项和待办。
- 在 `docs/README.md` 增加 WSS-only 当前入口，避免从 V3P 文档误入本路线。

**对应代码/文档**：[第六轮总入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)、[WSS 精度突破计划](WSS最小化_第六轮_WSS精度突破计划与执行.md)、[横向多目标计划](WSS最小化_第六轮_横向多目标对比计划与执行.md)、[BC/速度条件路线](WSS最小化_第六轮_边界条件与速度路线.md)、[训练实验跟踪](WSS最小化_训练实验跟踪.md)、[文档索引](../README.md)与本记录。

**推进到实验步骤**：文档已收敛，下一项唯一 P0 是指标/选模合同代码闭环与已有 checkpoint 的只读重评；本次未修改代码、配置或数据，未启动作业。

**当前状态判断**：没有通过新版指标和病例划分确认的最终候选；E/D+raw-Huber 保持冻结，E×raw-Huber No-Go，test16 未读。

## 2026-07-12｜第六轮 W0 审计 + W1 因子补全(E) + C 邻域预审计 + W2-L1 loss Gate + W3 组合 ✅DONE（历史实验裁决；E 排名已由上文降级）

**本次主要修改（代码/配置/审计工具）**：
- `training_wss_min/tools/make_configs_xyz_scale.py`：`GROUPS` 增 **E 组**（`xyz+coord_scale+geom`），补齐 A/B/D/E 2×2 嵌套因子；协议 JSON 升级为 `xyz_scale_abde_v1`，写入 `factor_design`（B−A/E−D/D−A/E−B/交互 E−D−B+A）；新增增量 manifest `configs/sweeps/xyz_scale_e.txt`（只列 E，避免重跑 A/B/D）。
- `training_wss_min/tools/make_configs_loss_l1.py`（新增）：W2-L1 raw-Huber λ∈{0.1,0.3,1.0} 单 seed 配置，基于 `r6_scale_D_xyzgeom_s1234`（固定 D 输入），只改 `loss_raw_huber_lambda`；预注册协议 `loss_l1_rawhuber_v1`。
- `training_wss_min/tools/make_configs_round6_w3.py`（新增）：W3 确认阶段——L1 λ=1.0 补 s7/s2025、W3 组合 E×raw-Huber λ=1.0 三 seed。
- `training_wss_min/tools/config_paths.py`：新增路由 `r6_l1_`→`loss_aug_ablation/`、`r6_w3_`→`round6_w3/`。
- `training_wss_min/tools/audit_coord_scale_w0.py`（新增）：W0 只读尺度审计（coord_scale 范围/OOD、与几何/WSS 的 Spearman、fast-slow 与裁剪分层）。
- `training_wss_min/tools/audit_c_neighborhood.py`（新增）：W1 C 邻域预审计，复现 SA 级联对比 A/C 每层邻居中位/孤立率/截断率；限线程、探测上限 64。
- `training_wss_min/tools/summarize_round6_w1w2.py`（新增）：只读汇总 A/B/D/E 因子（配对 seed 差+交互项）、L1 λ Gate、W3 组合 Gate。

**作业与结果**：`7557–7559`(E×3)、`7560–7562`(L1 λ-gate)、`7563–7567`(L1 λ1 confirm s7/s2025 + W3×3) 全部 `sbatch`，**11/11 COMPLETED**。产物 `runs/_audits/{w0_coord_scale,w1_c_neighborhood,round6_w1w2_summary}/report.md`。

**核心结论**：① **E 成为最佳可部署输入**（E−D field_cb +0.021 过 Gate-1，负例最少）但**尺度/几何冗余**（交互 E−D−B+A<0，E−D casemean≈0）；② W0 印证 coord_scale 编码尺寸非幅值；③ C 邻域不退化但被 nsample=16 截断抹平预期收益，判低优先级；④ **L1 raw-Huber 单 seed 全面正、λ=1.0 最佳、无爆峰**，但**三 seed 确认后收益缩水**（单 seed field 0.385→三 seed 0.347±0.033）、seed 脆弱；⑤ **W3 组合阴性**——E×raw-Huber 未过 §6 组合 Gate（全 3 seed 低于 D+rawHuber、2/3 seed 低于 E），尺度特征与 raw-space Huber 冗余互斥，**正式停止组合线**。待用户裁决 W5 候选（E vs D+rawHuber）或启动 L2。详见[训练实验跟踪：第六轮 W0–W3 节](WSS最小化_训练实验跟踪.md)。

## 2026-07-13｜横向指标与 V3P 可比性重构，WSS 恢复为 P0 主线 ✅DONE（文档/决策）

**本次主要修改**：
- 新增跨路线共享指标口径，将对比分为 A（直接对比）/B（协议化参考）/C（叙事背景）三级，并冻结 WSS 的病例等权点级、逐病例稳健性、热点和下游四层指标。
- 修正原“V3P vs wss_min 是公平 WSS 对比、gap 约 0.1、已证明共同信息上限”的过强结论：压力定为 C 级，WSS 定为 B 级；只有统一 split/时相/点集/输入/选模的 Bridge 协议才能给精确 gap。
- 将第六轮调度从“先补齐 36-run 压力/速度横向表”改为“WSS W0–W3 是 P0，横向是最小诊断矩阵”。Track B 先做 adapter/QA，默认只跑 `|v|` 和联合 `u,v,w` 单 seed sanity；内部压力、独立速度分量与三 seed 按 Gate 触发。
- 将 `R²_field_casebalanced` + 物理单位误差提为 WSS 首要点级指标；`R²_field_raw` 保留为历史衔接，`R²_casemean` 不再单独作部署主指标。`0.70` 保留为长期理想目标，不作跨物理目标通用工程门槛。

**对应代码/文档**：[跨路线评估与横向对比口径](../00-规范与记录/WSS跨路线评估与横向对比口径.md)、[第六轮总入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)、[横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md)、[WSS 精度突破](WSS最小化_第六轮_WSS精度突破计划与执行.md)、[BC/速度条件路线](WSS最小化_第六轮_边界条件与速度路线.md)、`docs/README.md`、`实验设计总纲.md`。本轮未改训练代码/配置，未启动新作业。

**推进到实验步骤**：第六轮路线重排完成；WSS W0–W3 可独立推进，Track B 仍等 adapter/QA，V3P Bridge 仅完成协议定义。

**当前状态判断**：原路线的主要风险是“不同协议的头条 R² 直接相减”和“为横向补表延误 WSS 主线”，而不是 gauge pressure 或 WSS-only 目标本身选错。当前 WSS 结果只能证明学到部分信号且稳健性/热点仍不足，不证明绝对信息上限已锁定。`test16` 保持未读。

## 2026-07-12｜V3P vs 最小路线：压力/WSS 数值可比性分析落档（已由 2026-07-13 口径修订取代）

> 历史记录保留；其中“WSS 公平对比、gap 约 0.1、已锁定信息上限”不再作当前结论，以本日志顶部 2026-07-13 条目和共享指标口径为准。

**本次主要修改**：向[横向多目标对比 §9](WSS最小化_第六轮_横向多目标对比计划与执行.md) 写入 V3P 数值可比性分析（§9.1 压力·重点、§9.2 WSS·公平对比、§9.3 部署指标口径），并在 §4.1 加交叉引用。

**核查与直算结论**：
- **压力方差分解**：本仓直算，壁面 bundle 与 `data_new` 全场 `y[:,3]` 一致——单峰帧 95.9% 是病例间"水平"、仅 4.1% 空间型态（跨全周期口径 ~90% 水平、其中 96.7% 是心动时间波）。"只预测每例均值"pooled R²=**0.959**，V3P R²_p 0.92–0.96 卡在此送分基线上。
- **BC = 压力水平**：V3P `global_cond={t_norm,BC_Inlet,BC_O1..O4}`，出口压力 BC 均值/方差≈压力目标本身；OLS 仅 BC→场均压力 R²=0.925，no-geom V3P 压力仍 0.936。出口 BC 是第五轮判定的 `oracle_non_deployable`。**故压力 0.92 与本路线 gauge 0.53 不可直接比。**
- **WSS 方差分解**：**11.5% 病例间 / 88.5% 空间型态**（与压力相反），"只预测每例均值"pooled R²=0.115——无水平可白送。V3P 最好 WSS ~**0.43**（平台 0.40–0.45，Go 线 0.459 未过），本路线纯几何 field **0.31–0.36**，**差距仅 ~0.1、都远低于 0.70**。V3P WSS 模型参数 0.25–1.3M（比本路线 4.4M 还小）→ 瓶颈是信息/表示上限，非模型容量。
- **§9.3 工程验收口径扩写**：明确"能不能用"要看物理单位误差（MAE/RMSE/分位 + NRMSE + 逐病例 casemean），不用 pooled 绝对 R²；附工程可用性小表——压力 gauge `xyz+geom` MAE **225 Pa**/RMSE 374 Pa（≈空间信号 531 Pa 的 0.42/0.70），WSS `xyz+geom` MAE **3.11 Pa**/RMSE 6.13 Pa（≈信号 6.2 Pa 的 0.50/0.99）；两者逐点误差都约为空间信号一半量级，趋势可见但离直接工程可用仍有距离。

**对应代码/文档**：[横向多目标对比 §9](WSS最小化_第六轮_横向多目标对比计划与执行.md)；证据源 `training/core/{models,metrics,losses}.py`、`pipeline/config.py`、`data_new/normalization_params_global.json`、V3 路线文档与 `任务A实验状态表.md`。

**当前状态判断**：分析性结论，未训练/未改训练代码。部署主指标口径确定为**逐病例 casemean**（压力用 gauge，WSS 用 raw+hotspot 护栏）；绝对压力/oracle BC 仅作上限探针。`test16` 保持未读。
**核心结论（当时口径）**：① E 在旧 dev1/Gate 下暂列最佳输入（E−D field_cb `+0.021`，负例最少），但已被顶部对抗性审查降级为临时候选；② W0 印证 coord_scale 编码尺寸非幅值；③ C 邻域不退化但被 `nsample=16` 截断抹平预期收益；④ L1 raw-Huber 单 seed 全面正、λ=1.0 最佳、无爆峰，但三 seed 确认后收益缩水、seed 脆弱；⑤ W3 组合阴性并停止组合线。历史数值详见[训练实验跟踪：第六轮 W0–W3 节](WSS最小化_训练实验跟踪.md)。

## 2026-07-12｜WSS 最小化预处理—训练全链路基础检查 ✅DONE

**本次主要修改**：
- 新增全链路基础检查报告，从原始读取、单位、配准/正交旋转、`[-1,1]` 各向同性缩放、节点 ID 对齐、无损压缩/FPS、训练特征/标签同索引、PointNeXt/激活/loss/超参数、完整点云评估逐项审计。
- 对 77 个 included bundle 做实际数值复核；同步在训练实验跟踪顶部加入结论入口。
- 本轮仅修改文档，不改预处理/训练代码与配置，不启动新训练，不读取新的 test16 指标。

**对应代码/文档**：
- 检查对象：`pipeline_wss_min/`、`training_wss_min/`、`data_wss_min/`、`training/splits/split_AG_wss_min_v1.json`
- 新报告：[WSS最小化_全链路基础检查报告_2026-07-12](WSS最小化_全链路基础检查报告_2026-07-12.md)
- 状态摘要：[训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：完成下一轮 C/E 严格尺度与 L1/L2 loss 前的基础审计 Gate；17 个现有单元测试通过，`compileall` 通过。

**当前状态判断**：当前 WSS 标量主线无致命坐标—标签错位；77/77 bundle 的正交旋转、逆变换、数组同长、ID/坐标守卫和采样同索引均通过。下一轮前必须优先处理/加固：3 个 included roll-sign 不可靠病例、毫米尺度真正进入 FPS/ball-query、速度路径 cell ID/裁剪同步守卫。GELU + 线性输出合理，不把激活替换作为首要提分项。

## 2026-07-12｜第六轮文档拆分与优先级重排（优先级已由 2026-07-13 再修订）

**本次主要修改**：
- 将原单篇“XYZ 尺度诊断+后续所有路线”重构为第六轮总入口和三份独立执行文档，分开横向多目标对比、WSS 精度突破、BC/速度条件路线。
- 冻结新优先级：先补齐壁面/内部压力与近壁 `u/v/w/|v|` 横向主表，再集中执行 WSS C/E、L1/L2、M1/M2；空余 GPU 可并行已预注册且互不依赖的 run。
- 保留 BC I0/I1/I2、入口裁剪分层和速度→WSS V0/V1/V2；明确横向速度基线不受 V0 oracle Gate 阻断，但 V2 仍必须过门。
- 吸收计划审核结论：C 改为 train-only 全局共享尺度及邻域 QA；L1/L2 固定 D 输入；压力/速度不再以 WSS top10 惩罚作通用选模规则；候选增加 grouped repeated validation。

**对应代码/文档**：
- [第六轮总入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)
- [横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md) / [WSS 精度突破](WSS最小化_第六轮_WSS精度突破计划与执行.md) / [BC/速度条件路线](WSS最小化_第六轮_边界条件与速度路线.md)
- `docs/README.md`、`training_wss_min/README.md`、`training_wss_min/configs/README.md`

**推进到实验步骤**：文档和执行调度重构；未修改代码/配置，未提交新作业，不中止已提交 Jobs `7551–7556`。

**当前状态判断**：横向 Track A 继续运行；Track B 必须先通过 adapter/坐标帧/指标 QA；WSS 训练主矩阵排在横向主表之后。test16 保持未读。

## 2026-07-12｜第六轮 横向对比 H-PW 压力-壁面：目标可切换 + 6 run 完训回填 ✅DONE

**本次主要修改**：
- 训练/评估管线支持 `target` 切换（此前 `target` 字段是摆设、硬编码 `wall_wss`）：
  - `dataset.load_case`/`load_partition` 增 `target` 形参；`target='pressure'` 读 `wall_pressure[peak]` 并**逐例去均值**（gauge/相对压力，可为负），复用 `normalize_wss`/`denormalize_wss` 的 `linear` 分支。
  - `train.py` 两处 `load_partition` 透传 `cfg.data.target`。
  - `evaluate.py`：`load_partition` 透传 target；**去掉对非 WSS 目标的 `clip(…,0,None)`**（gauge 压力可为负，仅 `method==log_z` 时裁剪）。
  - `tools/config_paths.py` 增 `r6_press_*` → `configs/multitarget/`。
- 新增 gauge-pressure 归一化 stats 生成器 `tools/make_pressure_stats.py` → `data_wss_min/fold_stats/pressure_gauge_stats_v2_dev1.json`（train-only/peak-only；53 例/707705 点；`method=linear`，mean≈0、std=529.2 Pa，gauge∈[-1880,1164]）。
- 新增配置生成器 `tools/make_configs_pressure.py` → `configs/multitarget/press_wall_{xyz,xyzgeom}_s{1234,7,2025}.json`（共 6）+ manifest `configs/sweeps/pressure_wall.txt` + protocol；相对 B1 control 仅改 `target=pressure`、stats 路径、`input_features`、`loss_weight_target=false`（纯 MSE，关掉 WSS 长尾加权）、name/seed。
- CPU 冒烟通过：gauge 目标逐例均值≈0、~40% 负值；normalize/denormalize 往返误差 6e-5；clip-gate 保留负值；负值下 metrics 不崩且可 JSON 序列化（`nrmse_mean` 因均值≈0 而巨大，属预期次要指标伪影，主指标 R² 正常）。

**对应代码/文档**：
- `training_wss_min/{dataset.py,train.py,evaluate.py}`、`tools/{config_paths.py,make_pressure_stats.py,make_configs_pressure.py}`
- `training_wss_min/configs/multitarget/`、`configs/sweeps/pressure_wall.txt`、`data_wss_min/fold_stats/pressure_gauge_stats_v2_dev1.json`
- [横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md)、[训练实验跟踪·第六轮](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：Track A 压力-壁面 6 个作业 `7551–7556` 全部 `COMPLETED (0:0)`，val-only 评估齐全。三 seed 均值——压力 `xyz` field/casemean `0.426±0.021 / 0.254±0.042`；压力 `xyz+geom` `0.531±0.053 / 0.509±0.015`（xyz+geom 下 0/8 失败例）。

**当前状态判断**：**同最小协议下压力比 WSS 好学得多**（xyz+geom 压力 0.531/0.509 vs WSS 0.311/0.197；xyz 压力 0.426/0.254 vs WSS 0.164/−0.085），压力 casemean 全程为正、xyz+geom 无失败例——空间压力型态比近壁剪切更由几何决定。但仍 `<0.70`，不外推其他目标。选模沿用 WSS 复合规则→列**探索性**；只读复核显示复合最优 epoch 与压力 R² 最优一致，数值应接近压力专用选模，正式复选待用户批准（预计不改数值）。Track B（压力-内部 + 近壁速度）待 data_new adapter。`test16` 保持未读。

## 2026-07-12｜第六轮 §14 多目标扩展（压力/速度）方案落地 ✅DONE（方案+口径已定，未执行）

**本次主要修改**：
- 按与老师讨论，向第六轮计划新增 §14：同最小协议（FPS-2000/PointNeXt-S/dev1/B1）下把目标从 WSS 换成压力+速度，做 `xyz`/`xyz+geom` 矩阵。
- 核对数据事实并写入方案：`wall_pressure` 已在 wss_min bundle（壁面即刻可跑，仅需 `load_case` 加 `target='pressure'` 分支）；**速度不在壁面（无滑移≈0）、wss_min bundle 无速度标签**，须走 `data_new/AG/**/result_features_merged-1162.pt` 的 `y=[u,v,w,p]`（15000 节点，已核对覆盖全部 84 个 AG 例）。
- **老师确认口径（已回填 §14）**：压力做**壁面+内部**；速度**只做近壁**；速度目标 = `u,v,w` 三分量 + `|v|` 幅值旋转不变对照。→ 6 目标 × 2 输入 × 3 seed = **36 runs**，分 Track A（压力-壁面，即刻可跑，6）+ Track B（data_new adapter：压力-内部 6 + 近壁速度 24）。
- 预注册逐目标归一化（压力不 log、建议逐例去均值；速度线性 z-norm + `|v|` 旋转不变对照）、target-weight 默认关、adapter 双采样域（内部全场 / 近壁带）要求与执行 Gate；**近壁速度与 §11 速度→WSS oracle 复用同一 adapter**。

**对应代码/文档**：
- [横向多目标对比](WSS最小化_第六轮_横向多目标对比计划与执行.md)、[训练实验跟踪·第六轮](WSS最小化_训练实验跟踪.md)
- 关联既有：`training_wss_min/dataset.py:load_case`（现 `target` 字段未启用、硬编码 `wall_wss`）；`data_new/AG/**/result_features_merged-1162.pt`（`y=[u,v,w,p]`）

**推进到实验步骤**：仅方案落地，未提交作业。Track A（压力-壁面）无新数据即可实现；Track B 前置 data_new adapter（双采样域）。

**当前状态判断**：口径已定，等指令开工。建议先实现 Track A 压力-壁面（`target='pressure'` 分支 + 逐例去均值归一化）跑单 seed gate，再落地 adapter。`test16` 保持未读。
**当前状态判断**：**同最小协议下压力比 WSS 好学得多**（xyz+geom 压力 0.531/0.509 vs WSS 0.311/0.197；xyz 压力 0.426/0.254 vs WSS 0.164/−0.085），但不能外推其他目标。该批次沿用 WSS 复合选模，仅列为探索性历史基线；如需对外确认，只做压力专用 val-only 只读重评。Track B 已在最终审查后暂停，`test16` 保持未读。

## 2026-07-12｜第六轮 A/B/D 尺度诊断收口 + 新思路交叉验证 ✅DONE

**本次主要修改**：
- 复核首批 Jobs `7029–7037`：`7029–7032`(A×3+B_s1234) eval 完整；`7033`(B_s7) 训练成功但 eval 崩溃；`7034–7037`(B_s2025+D×3) 训练即失败。
- 定位根因：首批提交清单引用旧目录 `configs/round6/r6_scale_*.json`，而同日“目录规整”已把 config 迁至 `configs/xyz_scale_diag/scale_*.json` 并删 `round6/`；早启动作业赶在删除前解析成功，晚启动作业 `FileNotFoundError`。与训练协议/数据/模型无关，未污染已完成 run。
- 新增 `training_wss_min/cluster/run_eval_only.slurm`（仅对已有 `ckpt_best.pt` 的 run 重评，不重训）。
- 按正确 manifest `configs/sweeps/xyz_scale_abd.txt` 重跑 `7039–7043`，5 个均 `COMPLETED (0:0)`，**A/B/D 9/9 eval 齐全**，回填完整三 seed 结果与终裁。
- 对第六轮 §6–§12 各新思路做证据交叉核对，写入第六轮计划新增 §13（含已核实引用表与需重新掂量的 7 条先验/张力）。

**对应代码/文档**：
- `training_wss_min/cluster/run_eval_only.slurm`、`cluster/logs/resubmit_20260712_001011.txt`
- [第六轮总入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md) / [WSS 精度突破](WSS最小化_第六轮_WSS精度突破计划与执行.md)（尺度结论 + 后续优化路由）
- [训练实验跟踪·第六轮](WSS最小化_训练实验跟踪.md)（完整 A/B/D 结果表 + 终裁）

**推进到实验步骤**：三 seed 均值——A(`xyz`) field/casemean `0.164±0.070 / −0.085±0.055`；B(`xyz+coord_scale`) `0.208±0.038 / +0.008±0.099`；D(`xyz+geom`) `0.311±0.014 / +0.197±0.020`。B−A field `+0.044`/casemean `+0.093` 均过 `+0.02` 门槛且逐 seed 方向一致（3/3）。

**当前状态判断**：**尺度信号 B−A 成立**——逐病例归一化确丢失有用物理尺度，`coord_scale` 稳定回收一部分（casemean 由负转正）；但只补回约三成缺口，**几何仍是压倒性主杠杆**（D−B field +0.103）。与第四轮 C4「coord_scale No-Go」不矛盾（C4 是在已含 `local_radius` 的 geom 上加、冗余）。绝对精度仍全面未达 0.70。用户批准下一步：`L1/L2 loss` 与 `C/E 严格尺度`并列先行，`M1/M2/M3` 次之，速度路线仅做 V0/V1 oracle。`test16` 保持未读。
**当前状态判断**：旧开发口径下 B−A 的尺度信号成立，但旧 `casemean` 不是病例 level R²；几何仍是主要增量（D−B field +0.103）。本条只保留当轮实验裁决，后续顺序已由 2026-07-13 顶部最终审查覆盖；`test16` 保持未读。

## 2026-07-12｜第五轮止损归档 + 第六轮后续路线预注册 ✅DONE

**本次主要修改**：
- 将第五轮口径冻结为“诊断性科学结案 / 内部工程 No-Go”；用户确认无达标候选时不强行运行 L0/15-run OOF/T16，临床/生产验证后置。
- 新增第五轮归档说明，明确原正式计划与实际止损的偏差；保留原文档/产物路径作审计证据。
- 扩展第六轮计划：补 C 严格物理 XYZ、E 嵌套因子组、BC 可辨识性、global-local/density-robust/残差/小模型矩阵、raw/case-balanced loss 矩阵、生成模型边界和速度→WSS oracle Gate。

**对应代码/文档**：
- [第五轮正式计划](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md) / [历史执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [结案归档说明](_archive/WSS最小化/WSS最小化_第五轮结案与归档说明_2026-07-12.md)
- [第六轮 XYZ 尺度诊断计划与执行](WSS最小化_第六轮XYZ尺度诊断计划与执行.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md) / [文档索引](../README.md)

**推进到实验步骤**：文档归档与后续预注册；不改当前 A/B/D Jobs `7029–7037` 的任何 config/Gate，不提交 C/E/架构/loss 新作业。

**当前状态判断**：第五轮已按止损口径归档，`test16` 保持未读；第六轮先等 A/B/D 九个 run 完整返回并交叉审查，再由用户批准下一阶段。

## 2026-07-12｜training_wss_min 目录规整（语义 configs + tools/experiments） ✅DONE

**本次主要修改**：
- `configs/` 按主题分目录：`baseline_sweep` / `loss_aug_ablation` / `clean_data` / `protocol_gates` / `pointcount_curve` / `fit_lc_diagnosis` / `xyz_scale_diag` / `sweeps`；去掉 `round5`/`round6` 目录名与文件名前缀。
- JSON 内 `name` 与既有 `runs/<name>/` **未改**（含 `r5_*`/`r6_*`），保证在跑/历史实验可复评。
- 根目录脚本分层：`tools/`（生成器/汇总/导出）与 `experiments/`（诊断）；旧 `-m` 入口保留兼容 shim；历史 `run_round5_*.slurm` 移入 `cluster/archive/`。

**对应代码/文档**：
- `training_wss_min/configs/README.md`、`training_wss_min/README.md`
- `training_wss_min/tools/`、`training_wss_min/experiments/`、`training_wss_min/tools/config_paths.py`
- [第六轮计划](WSS最小化_第六轮XYZ尺度诊断计划与执行.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md) 活跃路径已同步

**推进到实验步骤**：工程整理；不改训练协议、不重跑作业。

**当前状态判断**：活跃入口为 `configs/sweeps/xyz_scale_abd.txt` 与 `python -m training_wss_min.tools.make_configs_xyz_scale`；验收：锚点 `name` 与 `runs/` 一致，`python -m training_wss_min.config` 通过。

## 2026-07-12｜A0E-ctrl 两例 postview 面片包（CFD|Pred|Error） ✅DONE

**本次主要修改**：
- 按 `postview-surface-viz` 默认口径，用第五轮最新最佳锚点 `r5_a0e_b1_ctrl_s1234` 对 `slow/WU_FENG_YAN`、`fast/RAN_QING_BO` 做完整壁面推理与 STL 回插。
- 新增导出脚本 `training_wss_min/export_wss_postview.py`（wall CSV → Gaussian r=3/sharpness=2 → `*__surface_wall.vtp` + 三联图）。

**对应代码/文档**：
- `training_wss_min/export_wss_postview.py`
- `training_wss_min/cluster/run_postview_a0e_ctrl.slurm`
- `docs/03-汇报材料/figures/WSS最小路线_20260712/postview_a0e_ctrl/`
- [图表说明与汇报口径](../03-汇报材料/figures/WSS最小路线_20260712/图表说明与汇报口径.md)

**推进到实验步骤**：汇报可视化交付；映射覆盖率两例均为 100%；同点 wall R²：`WU_FENG_YAN=0.3225`、`RAN_QING_BO=0.4959`（均为 train，不作泛化结论）。

**当前状态判断**：ParaView 主入口为各例 `*__surface_wall.vtp`（含 `wss_cfd/wss_pred/err_wss/abs_err_wss`）；正式指标只读 `_export/*__wall.csv`。

## 2026-07-12｜第六轮 A/B/D XYZ 尺度诊断已预注册并提交 ⏳RUNNING

**本次主要修改**：
- 新增第六轮配置生成器和机器可读预注册；A=`xyz`、B=`xyz+coord_scale`、D=`xyz+abscissa/local_radius/curvature`。
- 9 个配置逐字段继承 B1/dev1/FPS-2000 协议，除输入特征和 `seed={1234,7,2025}` 外无其他变量。
- 同步修正 LC 外推口径：当前证据只支持“61 例范围内 40→53 无可辨识 field 增益”，不支持“几百/几千例数据无用”。

**对应代码/文档**：
- `training_wss_min/make_configs_round6_scale.py`
- `training_wss_min/configs/round6/` / `configs/sweep_round6_scale_abd.txt`
- [第六轮 XYZ 尺度诊断计划与执行](WSS最小化_第六轮XYZ尺度诊断计划与执行.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md) / [training README](../../training_wss_min/README.md)

**推进到实验步骤**：配置不变量审计和 B 组特征构造通过；Jobs `7029–7037` 已提交，提交记录 `submitted_20260712_120420.txt`；默认 val-only，未访问 test16。

**当前状态判断**：本轮只允许判断 B−A 的尺度信号和 D−B 的显式几何增量；B 不是严格物理尺度纯 XYZ 终审。待 9 个 run 完成后按三 seed 均值±sample std 裁决是否实现 C 组。

## 2026-07-12｜第五轮 CFD 审计 + F0 结案 ✅DONE

**本次主要修改**：
- 子智能体完成 §6 CFD 可信性审计（read-only，dev61）；据全轮证据写 F0 结案报告。

**对应代码/文档**：
- `training_wss_min/runs/_round5/cfd_audit/{cfd_audit.py,cfd_per_case.csv,cfd_summary.json,cfd_audit_report.md}`
- `training_wss_min/runs/_round5/final_report/round5_final_report.md`（F0）、`round5_status_synthesis.md`（SUPERSEDED）
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：CFD 审计——peak 相位固定步统一、近壁 QA 全清、高 WSS 为真实几何热点、复现 floor ~2%（R²_cap ~0.92–0.96）；独立证实 `HOU_SHEN_QIAN=KANG_XI_MING` 同一几何且均 dev1 train。

**当前状态判断**：当前 61 例、当前几何-only 输入与 PointNeXt 协议下的 ~0.31 平台主要不是 CFD 标签噪声。§8.1 机制问题全部有可复核裁决 → **第五轮科学结案**；dev1 ~0.31/0.21 ≪ 0.70 → **未达内部工程目标**（§8.2），未跑 OOF。该结论不外推到几百/几千个高质量独立病例的极限。

## 2026-07-12｜第五轮 B-BC 资产审计：可部署 BC 杠杆关闭 ✅DONE

**本次主要修改**：
- 子智能体完成 read-only BC 资产审计（未训练、未改 data_new）：解析 61 dev 病例的 udf-inlet.c、vf-in、5 路压力监测、RCR。

**对应代码/文档**：
- `training_wss_min/runs/_round5/bc_audit/{audit_ag_bc.py,bc_per_case.csv,bc_summary.json,bc_audit_report.md}`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md)；G2 产物原记录路径为 `training_wss_min/runs/_round5/branch_experiments/branch_decision_g2.md`，当前工作区未保留该文件。

**推进到实验步骤**：61/61 dev 覆盖完整；Fourier 入口模板跨病例逐字节相同、`Q` 与面积无关（入口流量 CoV≈1.85e-4）；top-3 方差全为 oracle RCR（CoV 0.55–0.67）。

**当前状态判断**：**入口流量是共享人群模板、非病人特异；唯一 per-case 变化的 BC（出口 RCR/压力/流量分配）全部 `oracle_non_deployable`。可部署 B-BC 杠杆关闭。** RCR 仅是尚待病例外负对照验证的信息候选，不能从资产方差直接推断为跨病例 WSS 差异的根因；后续只允许按顶部最终审查的 2×2 oracle 探针量化。

## 2026-07-12｜第五轮 LC 收口：当前 61 例范围内暂时平台 ✅DONE

**本次主要修改**：
- 完成 LC stage-2（seed 7/2025）并用 `summarize_round5_lc.py` 汇总 30 个 run，产出曲线/方差分解/图/verdict。
- 新增 LC 汇总器 `training_wss_min/summarize_round5_lc.py` 与报告 `learning_curve_report.md`。

**对应代码/文档**：
- `training_wss_min/summarize_round5_lc.py`
- `training_wss_min/runs/_round5/learning_curve/{learning_curve_report.md,lc_points.csv,lc_curve.png,lc_verdict.json}`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：field R² 均值曲线 `13→0.258 / 26→0.297 / 40→0.312 / 53→0.310`；配对增量 `13→26 +0.039`、`26→40 +0.014`、`40→53 −0.002±0.049`。端点 53 逐 seed `0.359/0.252/0.319`（std 0.044）。

**当前状态判断**：**field R² 在当前观测范围内约 40 例后进入 ~0.31 暂时平台，40→53 增量与 0 不可分**。stage-1 单 seed"仍在上升"被证明是 s1234 端点偏高伪影（多 seed 纠正）。该 LC 只覆盖 13–53 例，不能用于否定几百/几千例高质量数据的潜在收益；它仅支持在现有 61 例池中不再继续小步扩展同配方 LC。

## 2026-07-12｜第五轮 A0E 重锚 + G1 裁决 + LC 启动 ✅DONE / ⏳LC 运行中

**本次主要修改**：
- 新增 A0E dev1 control 重锚：两支 val-only 训练（`ctrl` B1 逐字 min_lr=1e-5、`nsl` 仅抬 LR 下限到 2e-4），判定历史 `0.34` 是否为训练预算伪影。
- 完成 G1 第二次分支裁决：合并 A0D/A0E/A0R/A1 证据，批准 LC、关闭 B-REP、维持 B-DEN/B-BC BLOCKED。
- 新增确定性 LC 生成器 `make_configs_round5_lc.py`：3 链嵌套分层子集 + per-subset 划分/WSS stats + B1 配方 config；提交 stage-1 seed1234 的 9 个子集训练。

**对应代码/文档**：
- `training_wss_min/make_configs_round5_lc.py`、`training_wss_min/configs/round5/a0e_b1_{ctrl,nsl}_s1234.json`、`training_wss_min/configs/round5/lc/`
- `training/splits/split_AG_wss_min_v2_dev1_lc_*.json`、`data_wss_min/fold_stats/wss_stats_v2_dev1_lc_*.json`
- `training_wss_min/runs/_round5/a0e_control/a0e_control_report.md`、`training_wss_min/runs/_round5/branch_experiments/branch_decision_g1.md`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：A0E `ctrl` Job `6992` field/casemean `0.3587/0.2300`（复现 anchor），`nsl` Job `6993` `0.3419/0.2125`（更差、2 例负 R²）；G1 `DONE`；LC stage-1 Job `6994–7002` 运行中，LC53 端点复用 ctrl。

**当前状态判断**：`0.34/0.23` 经 A0E 确认为真实泛化上限、非预算伪影；非饿死 LR 假设被证据推翻，LC 改用标准 B1 schedule（`如果结果不合理则修改` 的一次实际更正）。瓶颈定性维持"泛化受限 + 目标函数与 raw-R² 错位"。下一步等 LC 曲线判读增加同分布 AG 数据的边际收益。

## 2026-07-12｜第五轮 A0D 基础拟合链分解 ✅GO

**本次主要修改**：
- 并行完成 normalization/loss、点—标签对齐/可辨识性、optimizer/AMP/BatchNorm 三路只读审计。
- 新增 A0D cluster-only 简化 overfit runner/config/Slurm，按四个 single-case→four-case shared→target-weight-only 串行执行。
- 将训练预算从易受病例数影响的 epoch 口径明确为 optimizer steps，识别旧 micro 仅 160 updates 的口径缺陷。

**对应代码/文档**：
- `training_wss_min/{a0d_simple_overfit.py,a0d_target_weight_only.py}`
- `training_wss_min/configs/round5/a0d_*.json`
- `training_wss_min/cluster/run_round5_a0d_*.slurm`
- `training_wss_min/runs/_round5/a0d_fit_chain/`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：Jobs `6986/6990/6991` 均完成。single-case R² `0.991–0.999`，four-case shared plain MSE field/casemean `0.979526/0.982533`，target-weight-only `0.993723/0.992423`。

**当前状态判断**：A0D `GO`。旧 `R²_casemean=0.844` 不是拟合上限；target-weight 单独不是缺口原因。下一步应冻结 optimizer-step 预算后做一个 dev1 val-only 候选，不启动 LC/B-DEN/test16。

## 2026-07-11｜第五轮 B-REP/C2 global context ⛔NO_GO / 已列候选止损

**本次主要修改**：
- 新增 B1 PointNeXt 解码特征的病例内 mean+max global context，广播拼接后仍由单输出 head 预测 WSS。
- 新增 C2 cluster-only 串行 runner/config/Slurm，以及前后向和病例隔离 smoke test。

**对应代码/文档**：
- `training_wss_min/{pointnext_global_context.py,brep_c2_global_context.py}`
- `training_wss_min/configs/round5/brep_c2_global_context_*.json`
- `training_wss_min/cluster/run_round5_brep_c2_global_context_{micro,dev1}.slurm`
- `training_wss_min/runs/_round5/branch_experiments/global_context/`

**推进到实验步骤**：Job `6984` `COMPLETED (0:0)`，last field/casemean `0.740682/0.695389`，未达 `0.95/0.95`，dev1 未提交。

**当前状态判断**：C2 `NO_GO`；M1/C1/C2 均未过 micro Gate，STL 特征又被 A1 mapping 阻塞。暂停 B-REP 训练扩展、LC 和 B-DEN，下一步返回 normalization/loss/标签对齐与几何可辨识性审计。

## 2026-07-11｜第五轮 B-REP/C1 radius-normalized relative position ⛔NO_GO

**本次主要修改**：
- 新增与 B1 同拓扑/同初始权重的 PointNeXt C1 候选，唯一改动是 SA/InvRes 局部 MLP 中的相对 xyz 除以对应层 radius。
- 新增 cluster-only 串行 micro/dev1 runner、config、Slurm 和唯一变量/前后向预检。

**对应代码/文档**：
- `training_wss_min/{pointnext_radius_norm.py,brep_c1_radius_norm.py}`
- `training_wss_min/configs/round5/brep_c1_radius_norm_*.json`
- `training_wss_min/cluster/run_round5_brep_c1_radius_norm_{micro,dev1}.slurm`
- `training_wss_min/runs/_round5/branch_experiments/radius_norm/`

**推进到实验步骤**：Job `6983` `COMPLETED (0:0)`，last epoch 159 的 field/casemean 为 `0.803401/0.760602`，均未达 `0.95`，dev1 未提交。

**当前状态判断**：C1 `NO_GO`。B-REP 进入正式计划中最后一个已列候选 C2 单输出 global context，仍先跑四病例 micro。

## 2026-07-11｜第五轮 B-REP/M1 逐点 MLP micro ⛔NO_GO

**本次主要修改**：
- 新增不读取 `pos/batch` 的逐点 MLP 对照、cluster-only 串行 runner、micro/dev1 config 与 Slurm 脚本。
- 先执行锁定四病例 micro；预注册规则要求两项 train R² 均 `>=0.95` 才允许 dev1。

**对应代码/文档**：
- `training_wss_min/{point_mlp.py,brep_mlp.py}`
- `training_wss_min/configs/round5/brep_m1_mlp_*.json`
- `training_wss_min/cluster/run_round5_brep_m1_mlp_{micro,dev1}.slurm`
- `training_wss_min/runs/_round5/branch_experiments/mlp/`

**推进到实验步骤**：Job `6982` `COMPLETED (0:0)`，last epoch 159 的 field/casemean 为 `0.880817/0.844018`，均未达 `0.95`，dev1 未提交。

**当前状态判断**：M1 `NO_GO`。候选 2 STL/表面特征受 A1 mapping `0/8` 阻塞；B-REP 按顺序进入 C1 相对位置 radius 归一化的四病例 micro。

## 2026-07-11｜第五轮 A1 + G0 密度/映射证据与首次分支裁决 ✅DONE

**本次主要修改**：
- 新增 A1 可复跑审计入口，完成三 seed 同索引 density probe、逐层邻域 cap、8 例 STL mapping、FPS-2000 真值 IDW3 oracle 与推理重复性检查。
- G0 合并 A0R/A0M/A1 证据，只批准 B-REP 的第一个单变量候选（逐点 MLP），拒绝当前 B-DEN/LC/B-BC。

**对应代码/文档**：
- `training_wss_min/a1_density_surface.py`
- `training_wss_min/runs/_round5/a1_density_surface/`
- `training_wss_min/runs/_round5/branch_experiments/branch_decision.md`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：A1 `NO_GO`、G0 `DONE`；IDW3 oracle 通过，但 mapping 总 Gate `0/8`，因此 D1/D2 被硬阻断。

**当前状态判断**：基础拟合不足是首要可执行问题；密度敏感同时存在，但映射协议未过，不能先做回插/ensemble。B-REP MLP 候选先跑四病例 micro sanity，未过则不进入 dev1 Gate-1。

## 2026-07-11｜第五轮 A0R B1 多 checkpoint 只读诊断 ✅DONE

**本次主要修改**：
- 新增 A0R 可复跑脚本，对 B1 三 seed 的 best/last/已有 candidate 共 15 个 checkpoint 执行 train/val × canonical-2000/full 四象限只读评估。
- 产出 60 行汇总、1830 行逐病例、42 行对比和机器可读摘要；未训练、未访问 test16。

**对应代码/文档**：
- `training_wss_min/runs/_round5/a0_readonly/{run_a0_readonly.py,rerun.sh,*.csv,summary.json,a0_readonly_report.md}`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：A0R 已完成。best canonical train field/casemean 仅 `0.538–0.573 / 0.494–0.528`；last 将 train field 平均提升 `0.070` 时，canonical val field 平均下降 `0.061`。

**当前状态判断**：继续训练不能同时修复 train-fit 与 val 泛化；full 相对 canonical 在三 seed 中一致劣化，但密度分支是否可行仍需 A1 oracle/mapping 裁决。

## 2026-07-11｜第五轮 A0M 四病例 micro-overfit ⛔NO_GO

**本次主要修改**：
- 新增 train-only 四病例锁定/验证/训练入口、独立 config/协议与 Slurm 脚本；正式训练只能在 Slurm 环境中启动。
- 从 dev1 正式 train 内按 fast/slow × WSS 均值最低/最高锁定 `SUN_ZHI_YU`、`WANG_DAO_CHUN`、`ZANG_YU_SHU`、`MA_TIAN_YI`；不允许看结果后更换。
- Job `6981` 用 B1 / fixed FPS-2000 / seed1234 训练 160 epoch，仅使用 last checkpoint 计算 train-fit，未加载 val/test。

**对应代码/文档**：
- `training_wss_min/a0_micro_overfit.py`
- `training_wss_min/configs/round5/a0_micro_{protocol,b1_s1234}.json`
- `training_wss_min/cluster/run_round5_a0_micro.slurm`
- `training_wss_min/runs/_round5/a0_micro/`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：A0M 已完成；Job `6981` `COMPLETED (0:0)`，`R²_field_raw=0.76548`、`R²_casemean=0.72381`，均低于 `0.95`。

**当前状态判断**：A0M `NO_GO`；按硬停止条件暂停 LC 和大规模 sweep，待 A0R/A1 完成后由 G0 裁决是否进入 B-REP。

## 2026-07-11｜第五轮 P0 评价协议与 Gate 实现 ✅DONE

**本次主要修改**：
- 新增病例等权全场 `R²_field_casebalanced`，并补充逐病例 R² median/P10/负 R² 数/失败率。
- 将第五轮 Gate 固化为 field/casemean 两项主 R² 同时超过 `0.02` 才可 Go，取消旧高 WSS 单路径 Go；top10 ratio/IoU 下降超过 `0.05` 会阻断 Go。
- 对评估 CLI 增加 test guard；开发默认 val-only，读取 test 必须显式传入 `--allow-test`。
- 新增 case-balanced 手算、常数标签、单病例、点数悬殊、共同 Gate 和 test guard 回归测试。

**对应代码/文档**：
- `training_wss_min/{metrics.py,evaluate.py,gate1_compare.py,tests/test_round4_protocol.py,README.md}`
- `training_wss_min/runs/_round5/protocol/{protocol_report.md,protocol_regression.json}`
- [第五轮执行计划](_archive/WSS最小化/WSS最小化_第五轮执行计划.md) / [训练实验跟踪](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：P0 Gate-0 已通过；17 项聚焦测试通过，旧 B1 seed1234 val-only best checkpoint 回归的 field/casemean/MAE 最大偏差 `3.24e-9 < 1e-8`。

**当前状态判断**：P0 `DONE`，未访问 test16，未提交新训练。A0R/A0M/A1 可按任务卡并行；G0 仍被三者联合产物阻塞。

## 2026-07-11｜第五轮优化计划收敛为正式版 ✅已定稿

**本次主要修改**：
- 将 1088 行终审与历轮交叉审查轨迹收敛为正式优化计划，删除审查者过程、重复提案、已被覆盖口径和过度详细的实现草案。
- 保留最终科学边界、当前基线、共同主指标、容量/密度/learning curve/BC 分支、CFD 可信性审计、全 61 例 OOF 和完成标准。
- 明确文档分工：正式优化计划负责科学与技术口径，独立执行计划负责智能体任务 ID、依赖和产物。
- 同步修复执行计划中指向旧 §18 的链接和文档索引。本次未修改训练代码、split、stats 或 manifest，未提交作业。

**对应代码/文档**：
- [WSS最小化_第五轮优化计划_正式版.md](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)
- [WSS最小化_第五轮执行计划.md](_archive/WSS最小化/WSS最小化_第五轮执行计划.md)
- [docs/README.md](../README.md)

**推进到实验步骤**：优化计划已正式定稿；执行状态仍从 P0 `NOT_STARTED` 开始。

**当前状态判断**：项目现在有一份简洁的正式方案和一份独立的智能体执行计划，两者职责分离，后续不需要从历轮审查文本中重新解释最终口径。

## 2026-07-11｜第五轮终审稿转为智能体执行计划 ✅已定稿 / 未启动

**本次主要修改**：
- 保留第五轮优化计划为决策与审查轨迹，新建第五轮执行计划作为后续智能体的唯一交接入口。
- 将终审 §18.9 拆为 P0、A0R、A0M、A1、G0、B-REP/B-DEN/B-BC、LC、L0、OOF、T16 和 F0 任务卡，为每张卡固定依赖、允许改动、产物、Gate 和停止条件。
- 增加共享工作区的领取/回填规则、统一 `_round5/` 证据目录、最小报告模板和并行边界，避免多智能体同时改动同一文件或跳过裁决。
- 同步更新文档索引和终审稿状态指针。本次未修改训练代码、split、stats 或 manifest，未提交作业。

**对应代码/文档**：
- [WSS最小化_第五轮执行计划.md](_archive/WSS最小化/WSS最小化_第五轮执行计划.md)
- [WSS最小化_第五轮优化计划_正式版.md](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)
- [docs/README.md](../README.md)

**推进到实验步骤**：第五轮已从终审阶段转为可分派任务卡；第一个可领取任务为 P0，本次未开始 P0。

**当前状态判断**：文档已具备多智能体顺序交接条件。P0 通过后可分派 A0R/A0M/A1；G0 之前不允许启动条件性模型分支或 learning curve。

## 2026-07-11｜第五轮计划 v1.0：Codex 最终审定 ✅有条件通过 / 待执行批准

**本次主要修改**：
- 完成第五轮最终审定，新增 §18 作为唯一执行入口；旧 §0–§17 保留审查轨迹，但执行时不得自行择取冲突口径。
- 发现 dev1/dev2/dev3 的 val 并集仅 20 个独立病例、41/61 从未作 val；终审否决用该并集声明 0.70，改为配置锁定后的全 61 例 5-fold OOF ×3 seeds 内部工程评估。
- A0 增加 best/last/top-k train-fit 对照与 4 病例 micro-overfit sanity，避免把早停/选模问题误判为模型容量不足。
- A0.5 BC probe 降为条件候选：入口 UDF 抽查发现固定分母，与 §17 的逐病例 `Q/A_inlet` 表述不一致；RCR/流量分配统计需先补可复跑脚本、CSV 和元数据；`B3−A` 只解释为预测增量，不作因果/全部方差表述。
- 区分第五轮科学结案、内部工程达标和未来前瞻性确认；R² 未达 0.70 可科学结案，但不得标记工程达标。
- 本次仅更新计划与推进记录，未修改训练代码、split、manifest，未提交作业。

**对应代码/文档**：
- [WSS最小化_第五轮优化计划_正式版.md](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)（当时终审结论现已并入正式版）
- 当前训练证据：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)

**推进到实验步骤**：终审有条件通过，仍处于 No-Run；等待用户明确批准后从 §18.9 第 1 步开始。

**当前状态判断**：计划已从多轮审查草案收敛为唯一执行顺序。当前首要任务仍是能力边界诊断而非扩大 sweep；工程 0.70 需全 61 例 OOF 内部评估，未来临床/生产声明仍需新 AG 前瞻性确认。

## 2026-07-11｜第五轮计划 v0.2：第二轮交叉审查 + 工程精度目标裁决 ⏳待执行批准

**本次主要修改**：
- 对第五轮 v0.1 与第一轮外部审查做第二轮代码/统计/密度/医工交叉核对；纠正“第四轮 B1 为纯 casemean 选模”的误读，实际为 `r4_composite_v1` 复合规则。
- 根据用户工程目标冻结：第五轮只做 AG 生长队列、peak-WSS 单标量；AAA 破裂队列独立；不要求每例 R²≥0.7，但工程目标要求逐病例完整壁面 `R²_casemean` 与 pooled `R²_field_raw` 向 0.70 验收。
- 将 A0 train-fit 提升为最高优先级；当前 B1/v2_dev1 重做 density probe 后才允许 D1/D2；全部 STL 用于表面连接/插值前 QA。
- learning curve 改为三条独立嵌套链，显式估计病例选择、模型 seed 和 dev split 三类噪声；不自动触发 AAA/ILO 混入 AG。
- AG 科学结案与 AAA/ILO 数据治理解耦；第五轮仍为 No-Run，未修改训练代码、split、manifest，未提交作业。

**对应代码/文档**：
- [WSS最小化_第五轮优化计划_正式版.md](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)（历史审查口径现已并入正式版）
- [WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md) / [新队列数据可用性审计_AAA_ILO_2026-07-10.md](新队列数据可用性审计_AAA_ILO_2026-07-10.md)

**推进到实验步骤**：推进到 v0.2 审查裁决与工程验收定义；未进入 A0/A1，等待用户明确批准执行。

**当前状态判断**：目标 `R²≈0.70` 相对当前 field≈0.34 / casemean≈0.19 属结构性提升，必须先用 train-fit 与当前 B1 density 四象限判断能力边界，再决定 learning curve 或模型修复；不适合直接扩大 sweep。

## 2026-07-10｜第五轮优化计划 v0.1：冻结 AG 单队列、WSS 标量与 field-R² 主目标 ⏳待交叉审查

**本次主要修改**：
- 新建第五轮优化计划讨论稿，明确状态为“待其他智能体交叉审查、未授权执行”；本次未修改训练代码、split、数据 manifest，未提交任何实验。
- 冻结第五轮边界：AG 为唯一主实验队列；AAA/ILO 分队列治理且异常单元先隔离；任务保持峰值收缩期 WSS 单标量、单输出头；完整壁面 `R²_field` 为主指标。
- 把第五轮执行顺序收敛为：三队列清单治理 → AG train-fit/density 诊断 → AG 13/26/40/53 learning curve → 条件性单任务结构优化 → 医工交叉审计。
- 纳入新队列讨论中发现的待隔离项：`ILO/LIU_BAO_JUN-0/after` 近零 WSS、7 个 `vf-in` 数量级/口径异常单元、方向 watch 和跨队列病人分组风险。

**对应代码/文档**：
- 新计划：[WSS最小化_第五轮优化计划_正式版.md](_archive/WSS最小化/WSS最小化_第五轮优化计划_正式版.md)（后续收敛为正式版）
- 证据：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md) / [新队列数据可用性审计_AAA_ILO_2026-07-10.md](新队列数据可用性审计_AAA_ILO_2026-07-10.md)

**推进到实验步骤**：仅推进到第五轮 v0.1 计划冻结与交叉审查入口；未进入 Stage A，未授权执行。

**当前状态判断**：第五轮的首要目标不是扩大模型或混入新队列，而是先在 AG 上区分病例数、训练/推理密度、局部表示和 CFD 标签上限。交叉审查完成并经用户批准前，本文只作为讨论基线。

## 2026-07-10｜第四轮执行计划归档 ✅已完成

**本次主要修改**：
- 核对第四轮实际产物后确认：Stage A–C 与 §14 点数首轮曲线已完成；Stage D 的多 seed、重复 dev split 和最终 legacy test 尚未执行。
- 将计划从当前目录移入 `_archive/WSS最小化/`，更名为 `WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md`，并在文首明确归档边界，避免将“计划阶段完成”误写成“最终模型确认完成”。
- 同步更新 `docs/README.md`、`training_wss_min/README.md`、WSS 训练跟踪和归档索引；实验数据、run 目录与 Slurm 日志未移动。

**对应代码/文档**：
- 归档：[WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md](_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)
- 当前状态源：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md) / 本记录

**推进到实验步骤**：第四轮计划执行阶段结案；如继续推进，从 Stage D 的 1000/2000 多 seed确认开始，不再回到已判 No-Go 的单 seed配置。

**当前状态判断**：第四轮基础几何、协议修复和点数首轮探索基本完成；尚未形成跨 split、跨 seed、legacy test 的最终确认性结论。

## 2026-07-10｜点数曲线多 seed 确认完训（1000 vs 2000）✅已分析

**本次主要修改**：
- Jobs 6976–6979 完训；与 seed1234（6969/6960）合并为三 seed 对照。
- 产出 `runs/_summary_round4_pointcount/pointcount_multiseed_{1000_vs_2000.csv,summary.csv,1000_vs_2000.png,verdict.json}`。

**三 seed 均值 ± sample std（val）**：

| n | R²_field | R²_casemean | top10 | IoU | score |
|---:|---|---|---|---|---|
| 1000 | 0.337±0.017 | **0.221±0.019** | 0.413±0.032 | 0.327±0.011 | 0.276±0.025 |
| 2000 | 0.339±0.020 | 0.188±0.025 | 0.416±0.025 | 0.324±0.016 | 0.259±0.026 |

**判读**：
- 仅 **R²_casemean** 三 seed 一致且幅度超过 seed 噪声（均值Δ=+0.033）偏向 1000。
- R²_field / top10 / IoU / score：**方向不一致或持平** → 不能把默认点数改为 1000。
- **协议锚点维持 2000**；1000 保留为更稀采样候选（病例均衡更好）。稳健「最佳点数」若需要，再上 dev2/dev3。

**对应代码/文档**：计划 §14.5；训练跟踪文首；`training_wss_min/README`；方案快照完成定义；本记录。

**推进到实验步骤**：§14 第二阶段完成；第三阶段（dev2/dev3）按需，非必须。

**当前状态判断**：单 seed「1000 全面领先」已证伪；点数选择上 1000≈2000（field），1000 略优（casemean）。

## 2026-07-10｜点数曲线多 seed 确认已提交（1000 vs 2000）✅已完训

**本次主要修改**：
- 生成 `r4_dev1_pc_xyzgeom_fps1000_s{7,2025}`；2000 复用已有 B1 `s7/s2025` 配置。
- 提交 Jobs **6976–6979**（val-only）：1000×{7,2025} + 2000×{7,2025}；seed1234 已有 6969/6960。

**对应代码/文档**：`make_configs_round4_pointcount.py`（multiseed manifest）；`configs/sweep_round4_pointcount_multiseed.txt`；计划 §14.5；训练跟踪。

**推进到实验步骤**：已完训并转入上一条分析。

**当前状态判断**：见上一条。

## 2026-07-10｜第四轮 xyz+geom 点数—精度曲线完训汇总 ✅已分析

**本次主要修改**：
- Jobs 6969–6972 完训；与 B1-2000 / B3-4000 合并为 6 点曲线；覆盖审计 Job 6968 落盘。
- 新增汇总脚本 `summarize_round4_pointcount.py`，产出 `runs/_summary_round4_pointcount/`（metrics CSV、R²/尾部图、density audit、verdict JSON）。

**关键数字（val / seed1234 / B1 配方）**：

| n | R²_f | R²_c | top10 | IoU | score |
|---:|---:|---:|---:|---:|---:|
| **1000** | 0.351 | **0.241** | **0.448** | 0.330 | **0.301** |
| 1500 | 0.327 | 0.208 | 0.408 | 0.296 | 0.262 |
| 2000 | 0.351 | 0.214 | 0.436 | 0.330 | 0.283 |
| 3000 | 0.298 | 0.150 | 0.365 | 0.315 | 0.205 |
| 4000 | 0.259 | 0.207 | 0.346 | 0.296 | 0.227 |
| 6000 | 0.335 | 0.207 | 0.412 | 0.343 | 0.268 |

**判读**：
- 精度峰值（composite）= **1000**；近似平台仅含 1000。2000 与 1000 的 R²_field 持平，但 casemean/top10/score 更低。
- 曲线**非单调**：3000/4000 下凹，6000 回升仍不及 1000/2000 → 「堆点数」在本配方下无稳定收益。
- 覆盖随 n 升（fixed high-hit 约 5%→25%），精度峰值却在最稀端 → 更像密度/邻域错配，而非单纯标签覆盖不足。
- **单 seed 边界**：不得写成「1000 已是最优」；下一步对 1000 vs 2000 补 seed7/2025。

**对应代码/文档**：`summarize_round4_pointcount.py`；`runs/_summary_round4_pointcount/`；`runs/_audit/coverage_audit_v2_dev1.*`；计划 §14.3–14.4；训练跟踪文首；`training_wss_min/README.md`。

**推进到实验步骤**：§14 第一阶段（单 seed 完整曲线）完成；进入多 seed 确认（峰值 1000 × 锚点 2000）。

**当前状态判断**：已回答「geom 下是否越多点越好」——至少在 dev1/seed1234 上**不是**；2000 仍是历史协议锚点，但单 seed 峰值在 1000。

## 2026-07-10｜第四轮点数—精度曲线最小矩阵已提交 ✅已完训

**本次主要修改**：
- 生成 B1 配方点数横扫配置：`r4_dev1_pc_xyzgeom_fps{1000,1500,3000,6000}_s1234`（仅改 `wall_n_points`）。
- 提交 GPU Jobs **6969–6972**（val-only）；复用 B1-2000 Job 6960、B3-4000 Job 6962。
- 提交 node03 覆盖审计 Job **6968**：`v2_dev1 + fold stats`，k=`1000…6000` → `runs/_audit/coverage_audit_v2_dev1.*`。

**对应代码/文档**：
- `training_wss_min/make_configs_round4_pointcount.py`
- `training_wss_min/configs/sweep_round4_pointcount_s1234.txt` / `sweep_round4_pointcount_map_s1234.txt`
- `training_wss_min/cluster/run_coverage_audit_dev1.slurm`
- 计划 §14.3、训练跟踪文首、本记录。

**推进到实验步骤**：已完训并转入上一条汇总分析。

**当前状态判断**：见上一条。

## 2026-07-10｜第四轮点数—精度曲线补充方案归档 + 审计边界修正 ✅已转执行

**本次主要修改**：
- 将导师复核后的问题收敛为一个单变量补充实验：固定第四轮 B1 单头 `xyz+基础几何` 配方，横扫 `1000/1500/2000/3000/4000/6000` 点，画 `R²_field/R²_casemean`—点数曲线。
- 修正原计划的结论边界：B3 只证明 4000 在 dev1/seed1234 下不如 2000，不证明 2000 已是最佳点数；原“4000 No-Go 则不开 6000”只是算力 Gate。
- 明确本组不加辅助头、新几何特征、新 loss 或新采样器；先建立简单单头模型的点数平台。
- 记录覆盖审计的实际口径缺口：旧审计使用 v1 split/全局 stats，补充曲线执行前必须按 `v2_dev1 + fold stats` 重跑。

**对应代码/文档**：
- 归档新增：[`_archive/WSS最小化/WSS最小化_第四轮补充实验_点数精度曲线方案快照_2026-07-10.md`](_archive/WSS最小化/WSS最小化_第四轮补充实验_点数精度曲线方案快照_2026-07-10.md)。
- 更新：[第四轮执行总结与归档](_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)（新增 §14 的执行结果）、归档 README、本记录。
- 后续已生成 config 并提交作业（见上一条）。

**推进到实验步骤**：方案归档后已转入 §14.3 执行。

**当前状态判断**：见上一条「最小矩阵已提交」。

## 2026-07-10｜第四轮 Stage A→B→C 执行：协议落地 + 单 seed 矩阵 ✅已完成

**本次主要修改**：
- **Stage A**：审计脚本参数化；`load_partition` 缺失 bundle 报错；移除 `dist_to_wall`；`persistent_workers=False` + DataLoader `generator`/`worker_init_fn`；`fps_multistart` pool；hotspot 定位指标 + `r4_composite_v1` 选模/top-3/early-stop；3× `v2_dev` split + fold stats；A3 Gate-0 残差校准；A5 补 raw top10 与邻域 cap。
- **Stage B（val-only，Jobs 6959–6962）**：B0 batch-quantile 锚点 → B1 固定阈值（持平）→ B2 multi-start / B3 FPS4000。相对 B1：B2/B3 均 **Gate-1 No-Go**（B3 `ΔR²_field≈-0.09` 明确退化，不开 6000）。
- **Stage C（Jobs 6963–6964）**：C1 raw-Huber、C4 `coord_scale` 均为 Gate-1 No-Go。按条件 **跳过** C2（B1 未达增益 Go）、C3（A3 smearing 未通过）、C5（CPU Ridge 增量≈0）。
- 开发默认 `EVAL_PARTS=val`；未跑 legacy test（Stage D 未做）。

**对应代码/文档**：
- 训练侧：`training_wss_min/{config,dataset,train,metrics,evaluate}.py`、`make_configs_round4.py`、`make_v2_dev_splits.py`、`coverage_audit.py`、`a3_residual_calibration.py`、`c5_feature_probe.py`、`gate1_compare.py`、`tests/test_round4_protocol.py`、`cluster/run_train.slurm`。
- 资产：`assets_第四轮/{audit_bundles.py,a5_density_probe.py,a3_residual_calibration.json,c5_radius_gradient_probe.json,a5_density_probe_cap.csv}`；`training/splits/split_AG_wss_min_v2_dev{1,2,3}.json`；`data_wss_min/fold_stats/wss_stats_v2_dev*.json`；`runs/r4_dev1_*`。
- 单测 8/8 通过。

**关键数字（dev1 val）**：

| run | Job | R²_c | R²_f | top10 | IoU | Gate-1 |
|---|---:|---:|---:|---:|---:|---|
| B0 batchq | 6959 | 0.213 | 0.351 | 0.429 | 0.328 | 锚点 |
| B1 fixedq | 6960 | 0.214 | 0.351 | 0.436 | 0.330 | 持平（协议胜者） |
| B2 fpsms | 6961 | 0.211 | 0.352 | 0.425 | 0.341 | No-Go |
| B3 fps4000 | 6962 | 0.207 | 0.259 | 0.346 | 0.296 | No-Go |
| C1 rawhuber | 6963 | 0.200 | 0.329 | 0.413 | 0.362 | No-Go |
| C4 coordscale | 6964 | 0.215 | 0.313 | 0.396 | 0.325 | No-Go |

**其他闸门**：覆盖审计 fixed2000 high-WSS hit 均值仅 ~10%（worst ~5%），multi-start 40ep union ~34%；A5 全量推理邻域 cap 均值 ~0.97 vs 子采样 ~0.37；A3 smearing 三 seed 均使 top10 变差 → 不支持 NLL。

**推进到实验步骤**：第四轮单 seed 主矩阵已跑完；无配置达到 Gate-1 增益阈值，故不扩 3 seed / 不开 6000 / 不进 Stage D。后续若继续，应围绕密度错配（cap/train-eval 密度）或信息上限另立假设，而非重复采样/点数/简单辅助损失。

**当前状态判断**：协议与可复现性已修好；在新协议下，固定阈值≈原 batch 分位，multi-start/4000/raw-Huber/coord_scale 均无稳定收益。当前最优开发锚点为 **B1**（`r4_dev1_b1_tgtw_fixedq_s1234`）。

## 2026-07-10｜第四轮计划终审 v2.3：A5 复现落盘 + rot_aug 证据修正 ✅已完成

**本次主要修改**：
- 终审对 v2.2 的两个可检验事实主张做独立验证。A5 预检数字此前只存在于文档、无落盘脚本/CSV（不符合计划 §2.4 自定证据标准）；终审编写复现脚本并运行，`r3_clean_xyzgeom_tgtw_s1234` 的 MAE/RMSE/Pearson=`0.1645/0.2278/0.9381` **逐位复现**，并补齐 s7（`0.1413/0.1884/0.9605`）与 s2025（`0.1191/0.1638/0.9717`）两个 seed。
- 三 seed 新结论：24 个病例-seed 对中 20 个 mean-shift 为负，即**全量推理系统性高于训练密度推理约 0.03–0.07σ**；点数最多的病例（17682 点）在三 seed 中漂移均最大，支持漂移随密度差增大。A5 标准化口径至此完成，剩余 raw top10 差与邻域 cap 比例两项。
- rot_aug 历史证据核实：`r2_xyzgeom_rotaug` test `R²_field=0.186` 确为第二轮最低，但与 mse 控制组仅差 0.004（在 seed 噪声内），且 val `R²_field=0.281` 反而高于控制组 0.236。No-Go 决定维持，但依据由“历史证据最低”改写为“无正收益证据 + 单 seed 不可判 + 优先级”。
- 计划升版 v2.3：新增 §13 终审结论（含对 v2.2 各项裁决的逐项判定表和 Stage A 前遗留清单），文档冻结为第四轮执行基线，后续改动只能以新增小节/附录记录。

**对应代码/文档**：
- 新增证据资产：`docs/02-推进与变更/assets_第四轮/a5_density_probe.py`（复现脚本，FPS seed 与训练完全一致 `train.seed+7919*i`）、`a5_density_probe.csv`（24 行逐病例）、`a5_density_probe_meta.json`。
- 更新：[WSS最小化_第四轮优化计划_执行总结与归档](_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)（v2.3，§12.2-1/§12.3-3/§12.6/§11 修订 + 新增 §13）、[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)、`training_wss_min/README.md`、本文。
- 复核依据：`training_wss_min/runs/_summary/summary.csv`（rot_aug 数字）、`training_wss_min/runs/r3_clean_xyzgeom_tgtw_s{1234,7,2025}/` checkpoint（只读推理）。
- 本轮仅审核、复现诊断与文档修订；未改训练/评估代码、未生成第四轮配置、未提交训练。

**推进到实验步骤**：第四轮计划审核链闭合（v1 提案 → v2 交叉审核 → v2.2 裁决 → v2.3 终审），进入 Stage A 执行：A0/A0+（审计脚本参数化、split 严格化、常量列防护）、A1/A1+（worker-safe 采样与单测）、A2–A4，以及 A5 剩余两项。

**当前状态判断**：密度敏感性已由三 seed 落盘证据确立为第四轮最强新线索——方向为全量推理整体偏高，恰与尾部欠估方向相反，说明"下采样+插值"不是修复而只是诊断基线；主控制仍是 raw B0，一切结构/特征改动按预注册条件走独立 Go/No-Go。

## 2026-07-10｜第四轮计划第三轮代码/实测交叉审核 v2.2 ✅已完成

**本次主要修改**：
- 在既有逐行核对基础上，使用 `r3_clean_xyzgeom_tgtw_s1234` 完成 A5 首个只读诊断：8 个 val 病例、同一批 FPS-2000 点上，子采样推理与全量推理取同索引的标准化输出平均 MAE=`0.1645`、RMSE=`0.2278`、Pearson=`0.9381`。密度会实质改变输出，但不能由此宣称全量评估错误或直接替换为“下采样+插值”。
- 收紧新增建议的证据等级：保留缺失 bundle 报错、固定 train-only 分位、显式 DataLoader RNG、常量列防护、A5 三 seed 和邻域 cap 审计；取消“radius 一定返回先找到的邻居”“rel 未归一化必为 bug”等过度断言。
- 将 EMA/SWA 从 B0 默认搭载降为 B0 稳定后的预注册候选；将 `rot_aug` 保持为历史 No-Go（r2 field R²=0.186），不再列为 C8 probe。
- 修正 multi-start FPS 方案：预计算多个 FPS-2000 子集而非全 FPS order，且 pool 不能自动解决 persistent-worker 的 epoch 同步；`dist_to_throat` / `stenosis_ratio` 因缺少 branch-aware 定义和病例级 QA，撤出 C5 首批特征训练。

**对应代码/文档**：
- 更新：[WSS最小化_第四轮优化计划_执行总结与归档](_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)（v2.2；重写 §12，Stage A 新增 A5 正式口径）。
- 复核依据：`training_wss_min/{dataset,train,pointnext,evaluate,metrics,config}.py`、`training_wss_min/cluster/run_train.slurm`、`training_wss_min/runs/r3_clean_*/`、`pipeline_wss_min/preprocess.py`；A5 使用既有 checkpoint，仅推理、未产生训练产物。
- 本轮仅审核/修改文档，未实现计划代码、未生成第四轮配置、未提交训练。

**推进到实验步骤**：下一步仍是 Stage A：先让 split/worker/weight/RNG 口径可验证，再补齐 A5 其余两个 seed、A3 残差校准和 A4 重复开发划分；不得先开 NLL、法向或大点数矩阵。

**当前状态判断**：最强的新线索是密度敏感性，但它还是诊断结果，不是修复结论。第四轮的主控制仍应是 raw B0；任何 EMA、结构改动或新特征都必须在预注册条件和独立 Go/No-Go 下进入。

## 2026-07-10｜第四轮优化计划交叉审核定稿 v2 ✅已完成

**本次主要修改**：
- 交叉核对 77 例 bundle 审计、第三轮 6 个 run、`dataset/train/evaluate/metrics/pointnext` 与 Slurm 提交链，重写第四轮计划的因果判断、执行顺序、实验矩阵和 Go/No-Go。
- 将“log-MSE 是唯一第一性根因、NLL+Jensen 为最高优先级”降级为需先过残差/病例留一校准 Gate-0 的假设；raw-space Huber 辅助改为默认目标函数 probe，NLL 改为条件触发项。
- 新确认 persistent worker 下 epoch 状态同步风险：`resample_each_epoch` 不仅对 FPS 无效，random/geom-weighted 也可能复用 epoch 0 子集；同时固定 batch 分位 target-weight、val-only 开发评估和 fold-specific WSS stats 被提升为 P0/P1。
- 实测 `vf-in` 77/77 可读但 train peak-flow 变异系数仅约 0.02%，与 mean/p99 WSS 相关约 0.13/0.05，因此入口流量从当前第四轮主矩阵移除。

**对应代码/文档**：
- 重写：[WSS最小化_第四轮优化计划_执行总结与归档](_archive/WSS最小化/WSS最小化_第四轮优化计划_执行总结与归档_2026-07-10.md)。
- 同步：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)、`docs/README.md`、`training_wss_min/README.md`、本文和全项目推进记录。
- 复核依据：`docs/02-推进与变更/assets_第四轮/{audit_bundles.py,audit_bundles.csv}`、`data_new/AG/*/*/Global_conditions/vf-in-rfile.out` 和 `training_wss_min/` 当前实现。
- 本轮仅审核/修改文档，未实现计划代码、未生成第四轮配置、未提交训练。

**推进到实验步骤**：第四轮从“单方建议、待交叉验证”推进到“Stage A–D 可执行计划已定稿”；下一步从审计脚本可复现化、sampler 单测、hotspot 指标和 v2 dev split/fold stats 开始。

**当前状态判断**：当前最高优先级是实验正确性和开发协议，不是直接扩模型或启动 NLL/法向/6000 点矩阵。legacy test16 已被前三轮反复使用，第四轮开发阶段必须停止逐配置查看 test。

## 2026-07-10｜第三轮 clean-data 完训判读 + WSS 文档收口归档 ✅已完成

**本次主要修改**：
- 读取 Slurm 6953–6958 的 6 份完整 `eval/metrics.json`、训练曲线、逐病例指标与代表性热力图，完成 MSE/target-weight 各 3 seed 的均值、样本标准差、区域指标和 high-WSS 校准分析。
- 第三轮 target-weight 相对 MSE 的三 seed 均值：test `R²_field 0.225±0.034 vs 0.191±0.018`，`R²_casemean 0.212±0.027 vs 0.176±0.021`，MAE `2.788±0.040 vs 2.847±0.038`；但 seed 7 未稳定获益。
- 第三轮 clean-data 组合将 target-weight 的 top10 预测/真值均值由第二轮约 23.4% 提高到 35.7%，p99 比由约 35.1% 提高到 43.4%；整体 `R²_field` 却与第二轮 tgtw 均值 0.222 基本持平，说明数据修复改善幅值校准但未突破 geometry-only 信息/选模上限。两轮还同时改变 split、curvature transform 和训练时长，因此不把差异归因于 stats 单一变量。
- 将已执行完的预处理交接、工程建议和两版问题诊断移入 `_archive/WSS最小化/`；当前目录只保留 WSS 训练跟踪与推进记录两个活跃事实源，并新增归档索引。

**对应代码/文档**：
- 回填：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)、本文、`training_wss_min/README.md`、`docs/README.md`、根 `README.md`。
- 归档：`docs/02-推进与变更/_archive/WSS最小化/`，其中原诊断结论与交接历史保留，不删除失败证据。
- 实验依据：`training_wss_min/runs/r3_clean_xyzgeom_{mse,tgtw}_s*/eval/{metrics.json,per_case_metrics.csv,heatmaps/}` 与对应 `history.jsonl`、Slurm 6953–6958 日志。
- 汇总产物：已重跑 `training_wss_min.summarize`，`training_wss_min/runs/_summary/` 当前聚合 27 个实验并刷新 `summary.csv`、点数曲线与区域柱状图。
- 本轮未修改训练、评估或预处理代码，也未重新训练模型。

**推进到实验步骤**：
- 第三轮 clean-data 已从“产物待分析”推进到“完训、完整评估、三 seed 判读完成”。
- 下一轮优先级固定为：P0 复合/平滑选模与 early stopping；P1 `coord_scale` 和入口流量/可部署边界条件信息上限探针；P2 稳健尾部 loss；P3 重复 split/病例 bootstrap 置信区间。

**当前状态判断**：
- target-weight 是均值正收益方向，但目前仅 2/3 seed 明显获益，不能宣称稳定突破；单 run 最高 test `R²_field=0.251` 只作结果描述，不作为按 test 选出的最终模型。
- high-WSS 仍 No-Go：target-weight 三 seed high-WSS `R²=-1.531±0.167`、top10 比 `0.357±0.037`，热点大致定位但幅值过度平滑。继续只调 α 或堆模型的优先级低于补边界条件/尺度信息和修 checkpoint 选择。

## 2026-07-10｜第三轮 clean-data 口径落地 + bundle 重跑 + 训练矩阵提交 ✅已提交

**背景与目标**：用户确认第三轮不能只改 split，必须在移除污染病例后重跑 included bundle，并保证 pending/excluded 历史 bundle 不进入 stats/training；预处理阶段需要等待完成，训练阶段只提交 Slurm job，结果后续再分析。

**本次主要修改**：
- split 修正：`slow/ZHANG_HUAN_LI` 已从 `train_cases` 移入 `excluded_cases`，理由写明 WSS 近全零、`n_wall` 极端和多重审计异常；当前 split 为 train 53 / val 8 / test 16 / excluded 9 / pending 1，`slow/ZHAO_XIU_XUAN` 保持 pending。
- 数据守卫：`raw_io.read_wall_fields` 补读 ID 与坐标；预处理按首步 `nodenumber/cellnumber` 对齐每个时间步 WSS/pressure/vector，集合或长度异常即失败；兼容少数壁面文件把 ID 列命名为 `cellnumber` 的历史导出口径。
- QA gate：新增 split included QA，excluded/pending 只报告为 not participating；WSS 零值、极端点数、单位/覆盖/裁剪、非有限值、nodenumber/坐标错位为 fatal，单独 `trunk_centering_offset_frac>0.05` 按最终诊断作为 warning 复核项，不默认剔除。
- scope guard：`--all-raw` 仅允许诊断性 `preprocess`；正式 `qa-gate/global-stats/build-samples/all` 在 CLI 和函数入口均拒绝 all-raw，防止 pending/excluded 历史 bundle 被读入正式口径。
- stats 与训练：`global-stats` 默认支持 train peak-only；训练 run 保存 `wss_global_stats.json` 快照；dataset 支持 `coord_scale` 输入，curvature 新 run 默认 `signed_log1p` robust 统计；eval/summarize 固化 top10/p95/p99/max 校准指标。
- 第三轮实验：新增 template mean/voxel/KNN 基线脚本，新增 `make_configs_round3_clean.py` 生成 clean `mse/tgtw` 各 3 seed；提交训练 Job 6953–6958。

**对应代码/文档**：
- 更新：`training/splits/split_AG_wss_min_v1.json`、`pipeline_wss_min/{raw_io.py,preprocess.py,qa_gate.py,global_stats.py,run.py,reporting.py,config.py}`、`training_wss_min/{config.py,dataset.py,train.py,evaluate.py,metrics.py,summarize.py,template_baseline.py,make_configs_round3_clean.py}`、`training_wss_min/cluster/submit_baseline_sweep.sh`。
- 新增/生成：`training_wss_min/configs/r3_clean_xyzgeom_{mse,tgtw}_s*.json`、`training_wss_min/configs/sweep_round3_clean_v1.txt`、`training_wss_min/runs/template_{mean,voxel,knn}_clean/`。
- 文档同步：`pipeline_wss_min/README.md`、`training_wss_min/README.md`、[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)、[WSS最小化_数据与训练问题最终诊断_2026-07-10.md](_archive/WSS最小化/WSS最小化_数据与训练问题最终诊断_2026-07-10.md)。

**推进到实验步骤**：
- 预处理：Slurm Job **6952** 完成，`ok=77 / skipped=0 / error=0`；`nodenumber_reordered_cases=0`，`wall_coord_mismatch_cases=0`。
- QA gate：fatal=0，warning=3（`fast/ZHANG_QING_WANG`、`slow/GUAN_TONG_XIANG`、`fast/RAN_QING_BO`，均为 trunk offset 复核项）。
- clean stats：train peak-only 53 cases / 711412 wall points，zero_frac=0，log mean/std=`1.1595 / 1.0907`，raw p90/p99/max=`12.8847 / 32.6039 / 220.7968`。
- 模板基线（test）：mean R²_field=-0.114 / top10=0.172；voxel R²_field=0.035 / top10=0.260；KNN R²_field=-0.039 / top10=0.289。
- GPU 训练：`r3_clean_xyzgeom_mse_s{1234,7,2025}` 与 `r3_clean_xyzgeom_tgtw_s{1234,7,2025}` 已提交 Job **6953–6958**；终检时 6 个 run 均已写出 `eval/metrics.json`，本轮只记录产物入口，不做结果判读。

**当前状态判断**：第三轮 clean-data 已从“计划”推进到“训练产物已生成、结果待分析”阶段。数据污染源已移除，旧 stats 的 `std≈5.58` 已被 clean peak stats 替代；下一步不再改口径，按三 seed 均值/方差和 high-WSS 校准指标判读。

## 2026-07-10｜WSS 数据与训练问题最终诊断文档合并 ✅已完成

**背景与目标**：用户已进入第二轮实验，但第三轮是否需要重跑、怎么改，需要先把 `WSS最小化_训练结果差诊断_Claude与Codex联合审查_2026-07-09.md` 与本文 2026-07-09 数据 QA 复盘中的数据问题、训练症状和修改清单合并为一个后续智能体可直接执行的问题源。

**本次主要修改（文档合并，不改代码）**：
- 新增最终问题文档：[WSS最小化_数据与训练问题最终诊断_2026-07-10.md](_archive/WSS最小化/WSS最小化_数据与训练问题最终诊断_2026-07-10.md)（2026-07-10 执行完成后归档）。
- 明确第三轮前的最终判断：`AG/slow/ZHANG_HUAN_LI` 是当前 included 中唯一必须立即从 train 统计移除的重大污染源；`AG/slow/WANG_BAO_SHAN` 已 excluded，但必须防止直接 glob 历史 bundle 时误纳入。
- 合并训练问题链条：旧 all-time log stats 被坏病例和全时序共同压扁、high-WSS 系统性低估、peak 单步样本少、FPS 无有效重采样、val 选模噪声大、`nodenumber` 行序对齐仍需作为 P0 完整性守卫。
- 合并下一轮修改要求：修 split/denylist、增加 QA gate、重算 clean peak WSS stats、重算 feature stats、补 `nodenumber` 对齐、补 KNN/voxel/template 基线、clean-data `mse` 与 `target-weight loss` 多 seed 复跑，并记录 high-WSS 分位校准指标。
- 明确不要误删真实高 WSS 长尾病例：`LI_ZHI_LIN`、`MA_TIAN_YI`、`SUN_ZONG_GE`、`LIU_ZONG_YANG`、`SHEN_ZHI_GANG` 等只作为长尾样本保留，不按高值剔除。

**对应代码/文档**：
- 新增：原 `docs/02-推进与变更/WSS最小化_数据与训练问题最终诊断_2026-07-10.md`，执行完成后移至 `docs/02-推进与变更/_archive/WSS最小化/`。
- 更新：本文档顶部推进记录；`docs/README.md` 的 WSS-only 入口增加最终诊断文档链接。
- 本次未修改 `training_wss_min/`、`pipeline_wss_min/`、`training/splits/` 或任何实验配置。

**推进到实验步骤**：第二轮实验可以继续作为旧 stats 口径下的方向性证据；第三轮 clean-data 实验启动前，应以最终诊断文档为准完成数据口径修复与重新统计。

**当前状态判断**：问题源已合并完成；后续智能体无需再从两份源文档交叉抽取结论，可直接按最终诊断文档的 P0/P1/P2 顺序实施修改。

## 2026-07-09｜第二轮 sweep 收官汇总 + 高 WSS 分位校准复核 ✅已完成

**背景与目标**：第二轮 9 个 config（Slurm 5961–5969）全部跑完并完成 eval。本次做收官汇总（`summarize` 重跑聚合全部 18 run）、补做高 WSS 分位校准探针（与 2026-07-09 数据 QA 复盘中第一轮探针同口径），并回填跟踪文档与最终诊断文档。不改训练代码。

**第二轮最终结果（test 完整点云）**：
- 最优：`geomw_tgtw` R²_field **0.249** / `tgtwloss` **0.246**（R²_casemean 0.233、sten −0.068、hiW −1.377 均为全场最优）。tgtw 家族(0.235–0.249)全面领先 mse 参考(0.189)、huber(0.203)、rotaug(0.186，全场最差)。
- **seed 方差**：`tgtwloss` 三种子(1234/7/2025) R²_field = 0.246/0.217/0.203，极差 0.043，吞掉家族内配置排序；只有"tgtw 家族 > 非加权"这档结论可信。
- **分位校准探针（新证据，test 全场 pool）**：真值 top10% 均值 18.51 / p99 26.98 / max 126.74。mse 预测 top10% 比 21.1%；tgtwloss 仅 22.1%（几乎没动）；geomw_tgtw 26.2% 但 max 溢出至 138%；geomwsamp_tgtw 52.7% 但 max 爆到 2849(22 倍)。**结论：目标加权的尾部 R² 回拉主要来自中段，真峰值在旧 stats(log std≈5.58)下修不回来，clean-data 重算 stats 是第三轮前置硬条件。**
- best epoch 集中在 59–159（400 epoch 过长）；val(8例)→test R²_field 落差约 0.05–0.11，选模噪声大。

**对应代码/文档**：
- 更新：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)（第二轮完整表 + 分位校准表 + 5 条结论 + 待办改为第三轮 clean-data）、[WSS最小化_数据与训练问题最终诊断_2026-07-10.md](_archive/WSS最小化/WSS最小化_数据与训练问题最终诊断_2026-07-10.md)（第二轮收官证据回填 §0/§1/§3/§5/§6；执行完成后归档）。
- 产物：`training_wss_min/runs/_summary/{summary.csv,pointcount_curve.png,regional_bar.png}`（已含全部 18 run）。
- 本次未修改 `training_wss_min/` 代码与 split。

**推进到实验步骤**：第一/二轮（旧 stats 口径）到此收官，全部结论已固化为方向性证据；下一步严格按最终诊断文档 §7 启动第三轮 clean-data。

**当前状态判断**：loss 方向（目标幅值加权）已验证、数据问题已定位、旧口径上限已探明（R²_field≈0.25 / top10% 校准≈22%）。继续在旧 stats 上调参没有收益，第三轮必须先完成 split 修正 + QA gate + stats 重算。

## 2026-07-09｜第一/二轮训练后数据 QA 复盘：确认 ZHANG_HUAN_LI 为训练统计污染源 + 第三轮修改清单 ⚠️待执行

**背景与目标**：第一轮 `training_wss_min` baseline 暴露出系统性问题：整体 R² 已有弱信号，但 high-WSS / stenosis 区域 R² 全负，最佳模型在 test 高 WSS 区明显低估。为避免继续在被污染数据统计上做二轮/三轮调参，本次只做**数据侧复查与后续修改清单记录**，不改训练代码、不停止已进入第二轮的作业。

**本次主要修改（文档记录 + 数据 QA 结论）**：
- 复查范围：`training/splits/split_AG_wss_min_v1.json` 当前 included=78（train54/val8/test16）、2026-07-08 21:55:12 预处理审计 `data_wss_min/pipeline_reports/preprocess_audit_20260708_215512.csv`、以及 `data_wss_min/AG/*/*/bundle.npz` 中的全时序 WSS、peak WSS、曲率、局部半径、坐标范围和审计字段。
- **必须处理的训练污染病例：`AG/slow/ZHANG_HUAN_LI`**。
  - 当前仍在 `train_cases`，但 peak/all-time WSS 零值比例均为 `0.9166`，peak `p90=0`、peak mean `0.45`。
  - 壁面点数 `209299`，而剔除该例后的 included 正常范围约为 `9338-21452`，该例大约是正常最大值的 9.8 倍。
  - 审计字段同时触发 `unit_anomaly=True`、`unit_extent_mismatch=True`、`wall_crop_frac=0.324`、`trunk_centering_offset_frac=0.0999`。
  - 曲率 99 分位约 `3.13e6`，显著高于其它训练病例；它把第一轮几何特征标准化中的 curvature clip 拉到 `3.13e6`。
  - 结论：该例“坐标视觉回正”不等于“WSS 标签/几何统计可用”；继续留在 train 会污染 WSS global stats、feature stats 和训练 loss。
- **已排除但需防误读的同类风险：`AG/slow/WANG_BAO_SHAN`**。
  - 该例已在 `excluded_cases`，磁盘上仍有历史 bundle，peak/all-time WSS 零值比例同样约 `0.9166`，`n_wall=238116`。
  - 后续脚本必须严格按 split 读取 train/val/test，不允许直接 glob 全部 bundle 参与统计或训练。
- **未发现第二个 included 的同等级 WSS 标签污染病例**。
  - 除 `ZHANG_HUAN_LI` 外，train/val/test 的 peak_zero_frac 与 all_zero_frac 均为 0，未见 peak `p90=0`、非有限值、坐标超界或半径非正等致命问题。
  - `LI_ZHI_LIN`、`MA_TIAN_YI`、`SUN_ZONG_GE`、`LIU_ZONG_YANG`、`SHEN_ZHI_GANG` 等 peak 高值属于长尾高 WSS 样本，不能当作坏数据剔除。
- **需要复核/robust 处理但暂不建议剔除的病例**：
  - 曲率 per-case p99 较高：train `CHENG_GUANG_SEN`(`~7.99e5`)、`ZHANG_SONG_TIAN`(`~7.88e5`)；val `XU_YI_CAI`(`~6.45e5`)；test `WANG_YONG_FAN`(`~6.41e5`)、`GUO_XI_JIANG`(`~6.23e5`)、`QIN_SI_FU`(`~5.23e5`)。
  - 这些病例没有 WSS 零值污染或预处理致命审计字段；建议进入人工图件复核和 curvature robust clip，不作为第三轮默认剔除名单。
  - 坐标/方向复核清单：`LI_SHI_QIANG` 为 `bifurcation_fallback`；`ZHANG_HAO`、`ZHANG_QING_WANG`、`RAN_QING_BO`、`CHENG_LU_LI`、`GUAN_TONG_XIANG` 为 `wall_pca_fallback`；`ZHANG_XIU_WEN`、`CHENG_GUANG_SEN`、`CHEN_SHI_MING` 的 roll sign 置信较弱。当前只作为复核项，不直接判坏。
- **WSS global stats 被污染的量级**：
  - 当前 `wss_global_stats.json`：log mean/std = `-3.161 / 5.582`，train all-time zero_frac = `0.208`。
  - 若从 train 中排除 `ZHANG_HUAN_LI`：all-time log mean/std 约 `-0.353 / 1.270`，zero_frac 变为 0；若只用 clean train peak 统计：log mean/std 约 `1.159 / 1.091`。
  - 当前 stats 会把 raw WSS `20-200` 压到 log_z 约 `1.10-1.52`，峰值差异被严重压扁；这解释了 high-WSS 系统性低估。
- **第一轮模型低估证据**：
  - `feat_xyzgeom_fps_w2000_peak` 在 test：真值 p99/max = `26.98 / 126.74`，预测 p99/max = `9.37 / 16.71`；top10% 高 WSS 均值真值/预测 = `18.51 / 4.08`，只到 `22%`。
  - `feat_geomonly_fps_w2000_peak` 同样只到约 `23%`。因此 high-WSS R² 大负主要是峰值被系统压低，不是单个热力图显示问题。

**第三轮前必须安排的修改清单（给后续智能体）**：
1. **修 split / denylist**：将 `slow/ZHANG_HUAN_LI` 从 `train_cases` 移入 `excluded_cases`，理由写清楚为“WSS 标签近全零 + n_wall 极端 + unit/coverage/crop/trunk 多重异常，污染训练统计”。保持 `WANG_BAO_SHAN`、`SUN_WEN_QING`、`NIE_QUAN_ZHONG` 等既有 excluded 不自动回纳。
2. **增加数据 QA gate**：在 global-stats / build-samples / training 入口加入 split 内病例检查；遇到以下任一条件应 hard fail 或进入 denylist：`peak_zero_frac>0.01`、`all_zero_frac>0.01`、`peak_p90<=0`、`n_wall>50000`、`unit_anomaly=True`、`unit_extent_mismatch=True`、`wall_crop_frac>0.05`、`trunk_centering_offset_frac>0.05`、非有限值。曲率 p99 `>5e5` 先 warning + 图件复核，不默认剔除。
3. **重算 WSS stats**：排除 `ZHANG_HUAN_LI` 后重跑 `global-stats`；第三轮 peak 单步任务优先使用 clean train peak 的 log stats，至少不能继续使用当前 `std≈5.58` 的 all-time stats。训练与评估必须使用同一份新 stats。
4. **重算 feature stats**：所有 run 的 `feature_stats.json` 都不能复用旧值。排除 `ZHANG_HUAN_LI` 后 curvature 的全训练集 99% clip 会从 `~3.13e6` 降到约 `~1.0e5`；同时考虑对 curvature 做 `sign(x)*log1p(abs(x))` 或分位裁剪后 z-score。
5. **第三轮最小实验矩阵**：以 `xyz+geom / fps2000` 为主，重跑 `mse` 与 `target-weight loss`，各至少 3 个 seed；第二轮已跑结果只作为“loss 方向有效”的旁证，不作为 clean-data 最终指标。
6. **补高 WSS 监控指标**：每个 eval 除 R²/NRMSE/MAE 外，记录 test top10% high-WSS 的 `mean_pred/mean_true`、p99 预测/真值比、max 预测/真值比；checkpoint 选择可用 `val_r2_casemean + high_wss/stenosis` 的组合指标，避免只优化平滑低值背景。
7. **复核但不阻塞的病例清单**：对上述曲率尖峰和方向 fallback 病例补一页 QA 图；只有出现形态/方向明显错误或标签异常，才进入下一版 excluded。

**对应代码/文档**：
- 本次只修改本文档。
- 复查依据：`training/splits/split_AG_wss_min_v1.json`、`data_wss_min/pipeline_reports/preprocess_audit_20260708_215512.csv`、`data_wss_min/AG/*/*/bundle.npz`、`training_wss_min/runs/_summary/summary.csv`、第一轮 best checkpoint 完整 test 推理统计。

**推进到实验步骤**：第二轮实验可继续跑完，用于验证 loss/采样/增广方向；但第三轮 clean-data 实验启动前，必须先完成 split 修正、QA gate、WSS stats 和 feature stats 重算。

**当前状态判断**：当前 included 数据中，`ZHANG_HUAN_LI` 是唯一需要立即从训练统计中移除的重大污染源；其余 included 病例没有发现同等级 WSS 标签坏例。第三轮应视作“数据口径修正版”重新起跑，不能把第一/二轮在旧 stats 上得到的指标作为最终可发表结论。

## 2026-07-08｜training_wss_min：PointNeXt 残差 baseline 训练/评估框架 + 第一/二轮 sweep ✅完成

> 训练侧实验跟踪主文档：[WSS最小化_训练实验跟踪.md](WSS最小化_训练实验跟踪.md)（逐轮 sweep 设计、完整指标表、结论、待办）。

**背景与目标**：flow-divider 预处理口径落地后推进到训练侧。目标：把
`几何点云 (x,y,z[+几何]) → 壁面 WSS 标量`（全局 log_z 归一化，单头）的 baseline 搭起来并集群并行提交；
最终服务工程部署（**A 路**：部署有完整几何、缺 CFD 标签，故训练用稀疏子采样、**评估恒在完整壁面点云上**）。
要求与既有 V3P `training/` **完全独立**。

**本次主要修改（新建独立包 `training_wss_min/`）**：
- `pointnext.py`：PointNeXt-S **残差版**（InvResMLP 残差块 + **ball-query** 分组，密度鲁棒 → 训练稀疏点、推理整条血管可迁移）；`out_dim=3` 即切矢量（二期）。
- `dataset.py`：直读 `data_wss_min/**/bundle.npz`（不经 build_samples）；采样 fps/random/几何加权；特征标准化；训练期随机旋转增广（第二轮加）；完整点云评估接口。
- `metrics.py`：R²/NRMSE/MAE + 分区（分叉/狭窄/高WSS）。
- `train.py`：AdamW+cosine+AMP；日志 `train.log`+`history.jsonl`；best/last ckpt；loss 支持 MSE/Huber + 几何加权 + 目标幅值加权（第二轮加）。
- `evaluate.py`：加载 best，**完整点云**推理 → 指标 + 逐病例 CSV + 真值/预测/误差热力图。
- `make_configs.py` / `make_configs_round2.py`：两轮 sweep 配置生成。
- `cluster/run_train.slurm` + `submit_baseline_sweep.sh`：GPU 分区(master 4×4090) 训练+评估一条龙，多 config 自动排队并行。
- `summarize.py`：聚合 `runs/*/eval/metrics.json` → 对比表 + 点数曲线 + 分区柱状。

**数据口径**：split `split_AG_wss_min_v1`（train54/val8/test16）；坐标逐病例各向同性归一化 [-1,1]；WSS 全局 log_z（train-only 统计，std≈5.6）；评估在原始 WSS 空间。

**第一轮 sweep（9 config，Slurm 5945–5953）✅完成**（详见跟踪文档）：
- 几何特征是压倒性杠杆：xyz+几何(test R²_field=0.200) ≈ 纯几何(0.191) ≫ 纯 xyz@2000(0.053)。纯几何≈xyz+几何 → **标量任务里绝对坐标/配准框架几乎不值分**（正面印证前期讨论口径）。
- 几何特征更省点：2000 点几何 > 6000 点纯 xyz。几何加权采样(0.108) > random(0.084) > fps(0.053)。
- **所有配置狭窄区/高 WSS 区 R² 全负**（最好也 sten −0.20 / hiW −1.63），为主瓶颈。

**第二轮 sweep（9 config，Slurm 5961–5969）✅完成**（收官汇总见 2026-07-09 条目与跟踪文档）：以 xyz+几何为默认，专打尾部崩溃；新增旋转增广 + 目标幅值加权 loss。最终：tgtw 家族 R²_field 0.235–0.249 全面领先（最优 geomw_tgtw 0.249 / tgtwloss 0.246，sten −0.19→−0.07、hiW −1.63→−1.38）；但分位校准显示真峰值仅恢复 21%→22~26%，旧 stats 是硬上限；huber/rotaug 无收益；tgtw 三种子极差 0.043。

**对应代码/文档**：
- 代码：`training_wss_min/`（整目录，独立于 `training/`）。
- 文档：新增 `docs/02-推进与变更/WSS最小化_训练实验跟踪.md`（实验跟踪主文档）、`training_wss_min/README.md`。

**推进到实验步骤**：从预处理/坐标 QA 正式推进到 `x,y,z → wss` 训练与评估；建立点数-精度、特征消融、采样、loss 加权、旋转增广的可复现 sweep 与完整点云评估基线。

**当前状态判断**：baseline 已跑通、可复现、集群并行；最优 test R²_field 目前 **0.246**（第二轮目标加权 loss，仍在跑完最后几例）。主攻方向明确——几何特征 >> 坐标 >> 点数；尾部（狭窄/高 WSS）靠目标/几何加权可回拉但仍为负，需继续。矢量三分量为二期（幅值 + 内在系方向）。

## 2026-07-08｜flow-divider 解剖基点算法化：三叉连接点最终口径 + 78例对照报告 ✅已采纳

**用户新想法**：不再只取 `DistToBifurcation≈0` 区域普通均值，而是取“左右髂支连接到中间主动脉的三叉连接点”作为解剖基点；希望所有血管在同一框架下更规整，方便后续数据增强和训练。

**本次算法化实现**：
- `RegistrationConfig.center_on` 已切换为默认 `flow_divider`，作为当前最终预处理口径。
- `registration.py` 新增 `_flow_divider_origin()`：
  1. 读取 VMTK `centerline.vtp` 中的 `DistToBifurcation`。
  2. 取 `DistToBifurcation≈0` 的分叉近邻中心线点。
  3. 对近邻点做确定性 K-means，聚成 3 臂：主干端 + 左髂支起始端 + 右髂支起始端。
  4. 对 3 个臂中心做**等权平均**作为 flow-divider 原点，避免某一臂采样点更多时把普通均值拉偏。
  5. 若三臂分离不足或数据不完整，自动回退到原 `bifurcation` 原点，并写 `origin_kind=bifurcation_fallback`。
- `visualize_final_case_pages.py` 默认输出到 `outputs/wss_min/flow_divider_origin_QA_78例/`，可用 `--center-on` 做旧口径对照。
- 新增 `compare_flow_divider_origin.py`，批量比较旧 `bifurcation` 原点与当前 `flow_divider` 原点。

**对照报告（当前 included=78）**：
- 报告目录：`outputs/wss_min/flow_divider_origin_compare/`
- CSV：`outputs/wss_min/flow_divider_origin_compare/flow_divider_origin_compare_78cases.csv`
- 汇总图：`outputs/wss_min/flow_divider_origin_compare/flow_divider_origin_summary.png`
- 说明：`outputs/wss_min/flow_divider_origin_compare/说明.md`
- 原点位移 median/mean/p95/max：`4.78 / 5.21 / 10.23 / 20.08 mm`。
- 当前原点 |COM| mean/median：`0.2140 / 0.2228`。
- flow-divider |COM| mean/median：`0.2029 / 0.2133`。
- |COM| 改善/变差：`61 / 16`；其中明确改善（delta≤-0.005）`50` 例，轻微变化（|delta|<0.005）`17` 例，明确变差（delta≥0.005）`11` 例。
- flow-divider 直接成功 `77/78`，仅 `fast/LI_SHI_QIANG` 回退到 `bifurcation_fallback`。

**最终 QA 图**：
- 输出目录：`outputs/wss_min/flow_divider_origin_QA_78例/`
- 汇总表：`outputs/wss_min/flow_divider_origin_QA_78例/当前78例_坐标QA汇总.csv`
- 关键结果：
  - `center_on=flow_divider`：78/78。
  - `origin_kind=flow_divider`：77/78；`bifurcation_fallback`：1/78。
  - `roll_sign_reliable=True`：75/78；弱置信 3 例仍为 `fast/ZHANG_XIU_WEN`、`slow/CHENG_GUANG_SEN`、`fast/CHEN_SHI_MING`。
  - `wall_crop_applied=True`：1/78，仍仅为既有裁剪病例。

**17:47 人工复核后追加修正**：
- 发现 `fast/ZHANG_HAO` 在 flow-divider 口径下触发 `wall_pca_fallback` 后整例翻转。
- 根因：`wall_pca_fallback` 原本用壁面双峰分离度判断“哪侧是髂支侧”，但该例主干弯曲/截面展布更强，导致主干侧被误判成髂支侧，主轴正负号翻转。
- 修正：fallback 仍用壁面 PCA 修正主轴倾斜，但主轴**正负号继承中心线 trunk→bifurcation 方向**；即壁面 PCA 只替换方向，不再单独决定符号。
- 重扫 5 个 `wall_pca_fallback` 病例，只有 `fast/ZHANG_HAO` 存在“PCA 符号与中心线方向冲突且侧判误导”的情况；其他 fallback 病例不受影响。
- 已重画 `outputs/wss_min/flow_divider_origin_QA_78例/` 并重算 `flow_divider_origin_compare/`；总体统计不变，`fast/ZHANG_HAO` 已回到正常朝向。

**STL page03 复核后追加修正**：
- 发现 `02_STL同框_每20例/page03` 中浅橙色 STL 大幅偏离，经颜色/顺序定位为 `slow/ZHANG_HUAN_LI`。
- 该例 CFD wall 点云本身在 `01_点云同框` 和逐病例 X-Z 图中未飞出；异常只发生在 STL 可视化。
- 根因：该例 `unit_extent_mismatch=True`，wall 原始网格含未描入口尾巴，bbox 比例法会把 STL 过度放大。其 STL 原始坐标实际已接近 mm 尺度并与 centerline 覆盖段匹配。
- 修正：`visualize_stl_point_overlap.py` 新增 `_stl_scale_to_pipeline()`，在 `bbox_ratio`、`stl_mm`、`stl_native` 三种尺度假设中打分选择；覆盖不一致病例优先匹配 centerline 尺度。
- `visualize_final_case_pages.py` 复用该函数，并在最终 QA 汇总表新增 `stl_scale_kind` 字段。
- 修正后 `slow/ZHANG_HUAN_LI` 的 STL/wall 重心偏差从 `1.65` 降至 `0.025`，STL 顶点出框比例从 `71.8%` 降至 `0%`；该例使用 `stl_scale_kind=stl_mm`，其余有 STL 的 76 例仍为 `bbox_ratio`。
- 已重画 `outputs/wss_min/flow_divider_origin_QA_78例/`；新增单例复核图：
  `outputs/wss_min/flow_divider_origin_QA_78例/单例复核_slow_ZHANG_HUAN_LI_STL尺度修正.png`。

**最终图件归档与默认口径确认**：
- 当前最终 QA 入口：`outputs/wss_min/flow_divider_origin_QA_78例/`。
- 当前新旧原点对照报告：`outputs/wss_min/flow_divider_origin_compare/`。
- `RegistrationConfig.center_on` 默认已切到 `flow_divider`；`visualize_final_case_pages.py` 默认输出目录也已切到 `flow_divider_origin_QA_78例`。
- 旧的 `当前_78例坐标QA/`、`单位修复前后对比_点云同框/`、`stl_point_overlap_20260708_trunk_centering_cases/` 已归档到：
  `outputs/wss_min/归档_旧口径_20260708/flow_divider定稿前旧图_20260708/`。

**集群 preprocess 重跑（21:48）**：
- 已通过 Slurm 提交到 `node03`：`/public/slurm/bin/sbatch --parsable pipeline_wss_min/cluster/run_preprocess.slurm preprocess`，作业号 `5941`，状态 `COMPLETED`，退出码 `0:0`，耗时 `00:06:39`。
- 运行日志：`logs/wss_min_preprocess_20260708_214835.log`；Slurm 日志：`pipeline_wss_min/cluster/logs/wss_min_pre_5941.out` / `.err`。
- 新审计：`data_wss_min/pipeline_reports/preprocess_audit_20260708_215512.csv` / `.json`。
- 结果核对：split 为 train 54 / val 8 / test 16，included=78；78 个 included bundle 均在本次作业时间窗内更新，`missing=0`、`bad_load=0`、`stale=0`。
- 审计结果：`ok=78`、`skipped=0`、`error=0`；`origin_kind=flow_divider` 77 例，`bifurcation_fallback` 1 例；`coord_scale_on=wall` 78 例。
- `excluded_cases` 和 `pending_cases` 即便磁盘上存在历史 `bundle.npz`，本次作业没有更新，默认 `preprocess/global-stats/build-samples` 也不会读取它们。
- `pending=1` 的 `slow/ZHAO_XIU_XUAN` 含义：原始目录中存在该病例，但不在继承的 `split_AG_v1` train/val/test/excluded 任一名单中，因此暂挂起，不纳入当前 78 例训练口径，待后续单独评估后再决定是否补入。

**当前判断**：
- 这个口径更符合“解剖锚点”表述：把三叉连接点作为同一坐标框架的原点，比普通分叉近邻均值更不受局部采样密度影响。
- 数值上多数病例重心更接近统一框架，但仍有 16 例 |COM| 变差，且 1 例回退；经人工复核后采用该口径作为当前最终 QA 和后续预处理口径。
- `preprocess` 已按该口径在集群完成并生成 78 例新 bundle；后续还需继续跑 `global-stats -> build-samples`，保证 WSS 统计和训练样本也与当前 QA 图同口径。

## 2026-07-08｜人工复核再剔除 2 例：最终 included=78 + 重出 QA ✅

**用户复核意见**：`当前_80例坐标QA/page02` 同框与逐病例 XZ 中仍有 2 例肉眼偏差较大，第一版 baseline 先保守剔除。

**本次 split 修改**：
- 从 `train_cases` 移入 `excluded_cases`：
  - `slow/WANG_BAO_SHAN`：虽经单位兜底与未描入口段裁剪后回到统计正常簇，但 page02 同框/逐病例视图仍呈明显形态与朝向离群。
  - `slow/SUN_WEN_QING`：`coord_scale≈93.24mm`，为当前 80 例最小，归一化后视觉占比/上下位置离群。
- split 计数更新：train 54 / val 8 / test 16 / excluded 8 / pending 1 / total_found 87，当前 included=78。

**图件刷新**：
- 旧 80 例最终图归档到：
  `outputs/wss_min/归档_旧口径_20260708/旧版_最终80例坐标QA_剔除WANG和SUN前/`
- 新 78 例最终图：
  `outputs/wss_min/当前_78例坐标QA/`
- 汇总表：
  `outputs/wss_min/当前_78例坐标QA/当前78例_坐标QA汇总.csv`

**刷新后 QA 摘要**：
- included rows=78；`WANG_BAO_SHAN`、`SUN_WEN_QING` 不在 included，已在 excluded。
- `coord_scale` min/median/max = `128.30 / 192.16 / 266.18`，最低尺度离群被移除。
- 当前只剩 `slow/ZHANG_HUAN_LI` 触发入口裁剪（32.40%）和主干二次居中。

**关于“有的偏上/偏下”**：
- 当前坐标原点固定在分叉点，缩放用壁面 `max_abs`；入口端最低点常被压到接近 -1，但分叉上方能到多高取决于髂支/出口截断长度、分叉角度、主干弯曲和真实血管大小。
- 因此剩余轻微上下差异主要是解剖/截断范围差异，不是单纯归一化失败；只有像 `SUN_WEN_QING` 这种尺度极端小、归一化后视觉占比明显离群的病例才进入剔除。

## 2026-07-08｜最终 QA 收口：旧图归档 + 入口切平面裁剪保险 + 壁面归一化最终图 ✅

**本次收口动作**：
- 归档两个容易误读的旧图目录：
  - `outputs/wss_min/归档_旧口径_20260708/旧版_80例坐标QA_旧单位未裁剪all归一化/`
  - `outputs/wss_min/归档_旧口径_20260708/旧版_新口径QA_轴向裁剪壁面归一化_保险前/`
- `RegistrationConfig` 新增：
  - `crop_use_inlet_tangent=True`：入口裁剪改用中心线入口端局部切平面，避免单纯主轴一刀切。
  - `crop_min_wall_frac=0.08`：裁剪比例低于 8% 视为端点/正常解剖擦边，不执行裁剪。
- `untraced_inlet_crop_masks()`：优先用入口端局部中心线切向判断“中心线覆盖外”的未描入口段；切向不可用时回退旧主轴逻辑。
- `visualize_final_case_pages.py`：最终 QA 汇总 CSV 新增 `unit_extent_mismatch`、`wall_crop_applied/wall_crop_frac`、`coord_scale_on`；STL 视图也按同一入口裁剪口径过滤显示。
- `preprocess.py` / `reporting.py`：bundle/report/audit 补写 `coord_scale_on`。

**最终验证（当前代码直接复算 80 例）**：
- `coord_scale_on=wall`：80/80。
- 只裁剪 2 例：`slow/WANG_BAO_SHAN` 裁 `30.21%`，`slow/ZHANG_HUAN_LI` 裁 `32.40%`。
- 旧版擦边触发的 `fast/LI_ZHEN_SHAN` 不再裁剪（保留完整壁面）。
- |COM| 分布：mean `0.213` / median `0.220` / max `0.362`；两例回到正常簇。
- 主干二次居中仅 `slow/ZHANG_HUAN_LI` 触发；`slow/WANG_BAO_SHAN` 裁剪后无需二次居中。

**最终图件**：
- 主入口：`outputs/wss_min/当前_80例坐标QA/`
- 汇总表：`outputs/wss_min/当前_80例坐标QA/当前80例_坐标QA汇总.csv`
- 重点看：
  - `01_点云同框_每20例/page02_点云同框_病例21-40.png`
  - `01_点云同框_每20例/page03_点云同框_病例41-60.png`
  - `03_逐病例XZ主轴视图_每20例/page02_逐病例XZ_病例21-40.png`
  - `03_逐病例XZ主轴视图_每20例/page03_逐病例XZ_病例41-60.png`

**注意**：
- `slow/WANG_BAO_SHAN` / `slow/ZHANG_HUAN_LI` 只能匹配到 `stl_data/name_data` 中的 STL，STL 与 CFD wall 点云不是逐点同源；专项 STL 图只看形状方向，不作为剔除依据。
- 正式训练前仍需全量重跑 `preprocess -> global-stats -> build-samples`，让 bundle 与样本完全落到最终口径。

## 2026-07-08｜同框偏位三段式收口：单位因子 + 未描主动脉尾巴裁剪 + 壁面归一化 ⏳待全量重跑

问题同下（少数病例同框整体偏位）。逐层定位后确认需要三段修复叠加，缺一不可：

**① 单位因子鲁棒兜底**（详见本条下半部分）：3 例 `unit_factor` 被“壁面/中心线覆盖不一致”压到约 1/2 → 改 `_resolve_unit_factor` 整十次幂兜底，factor→1000。修复后 |COM| 0.50/0.66→0.32，但仍偏。

**② 未描主动脉尾巴裁剪**：实测分叉点其实已在原点（bif_norm≈0），偏位真因是这 2 例壁面网格含一段中心线没描到的近端主动脉（占 31–34% 壁面点），在 per-case 归一化里撑大尺度。
- `config.py`：`RegistrationConfig` 新增 `crop_untraced_inlet=True / crop_axial_margin_frac=0.12 / crop_trigger_overshoot_frac=0.20`。
- `registration.py`：新增 `untraced_inlet_crop_masks()`——配准后（基于中心线，不受尾巴影响）、归一化前，只裁“入口(主动脉)侧”超出中心线轴向覆盖且超出量>触发阈值的壁面/内部点；对覆盖一致的正常病例是零剪裁 no-op。
- `preprocess.py`：配准后应用裁剪 mask，壁面时间步场（WSS/压力/矢量）按同 mask 过滤；新增审计 `wall_crop_applied/wall_crop_frac`（report/bundle/批量审计/日志）。
- 裁剪后 aorta/transv 比 1.13–1.14 ≈ 正常例 1.07–1.15；但发现残余仍在。

**③ 坐标改按壁面归一化**：残余真因是尺度 `max_abs(壁面+内部)` 被内部点带偏——正常例内部含 CFD 流动延伸段（≈2.2×壁面），壁面只填约 0.46 框；这 2 例无延伸段，壁面填满。
- `config.py`：`NormalizationConfig` 新增 `coord_scale_on='wall'`（默认；旧口径 `'all'`）。
- `preprocess.py` / `visualize_final_case_pages.py`：缩放参照改为只按壁面 `max_abs`。

**三段叠加后验证（GNN_vmtk 已实跑这 2 例 + 正常例）**：
- 壁面归一化下全 80 例 |COM| 分布 mean 0.213 / median 0.220；`ZHANG_HUAN_LI` 0.226（**rank 38/80，正好中位**）、`WANG_BAO_SHAN` 0.190（**rank 57/80，中位偏下**）——两例彻底回到正常簇、不再离群。最大偏位反而是普通例（SUN_ZONG_GE 0.36）。
- 同框叠加图目视：两例落进灰色主簇、同轴同心同尺度。
- 裁剪对正常例（如 LI_HUAN_GE）为 no-op（0% 裁剪）。
- 对比图/机制图：`outputs/wss_min/单位修复前后对比_点云同框/`。

**决策记录**：用户拍板“裁掉未描主动脉尾巴 + 改壁面归一化”，不删这 2 例。

**待办**：
- 壁面归一化改动**波及全部 80 例**，需在 GNN_vmtk 全量重跑 `--stage preprocess`（80 例）后再 `global-stats`。
- 用户要求**先审查预处理再做训练样本**，故 `build-samples` 暂缓；先重出点云类 QA（`outputs/wss_min/新口径QA_裁剪加壁面归一化/`）供人工复核。
- STL 同框页需 `trimesh`（GNN 环境未装），本轮 QA 跳过 STL，只出点云/主轴/WSS 视图。
- `NIE_QUAN_ZHONG`（已 excluded）根因同属①②，如需可一并重跑评估是否回纳。
- 代码尚未 `git commit`，待用户审完 QA 一起提。

---
以下为 ① 单位因子修复的原始定位与验证细节（保留）：

## 2026-07-08｜单位因子鲁棒兜底：修复“中心线覆盖不全”导致的整体偏位（①，详情） ⏳待重跑

**问题定位（坐标 QA）**：
- 同框图中少数病例整体偏位，根因不是居中策略，而是 **单位因子 `mesh->mm` 反推被壁面/中心线覆盖范围不一致带偏**。
- 逐病例扫描 `unit_factor`：80 例正常（932–984≈米制）、`PENG_JI_MING` 967789（已知异常，自动处理），
  仅 3 例落在 492–622（≈真实值一半）：`slow/WANG_BAO_SHAN`、`slow/ZHANG_HUAN_LI`、`slow/NIE_QUAN_ZHONG`。
- 复核这 3 例原始几何：中心线只描了远端一段（含分叉），壁面网格却含整条近端主动脉，
  只有约 55–65% 壁面点落在中心线包围盒内；`cl_diag/wall_diag` 因此把 factor 压到约 1/2，
  坐标缩放/配准/二次居中全线偏移（居中偏移量高达 73–77mm），即同框图所见的“整体偏位”。

**本次主要修改**：
- `pipeline_wss_min/config.py`：`UnitConfig` 新增 `phys_diag_mm=250.0`、`ratio_trust=1.5`。
- `pipeline_wss_min/preprocess.py`：`_resolve_unit_factor` 改鲁棒版——比值法结果与
  “米制整十次幂（使壁面对角线≈生理尺度的 10^k）”偏离超过 `ratio_trust` 倍时，
  判定壁面/中心线覆盖不一致，改用整十次幂并返回 `extent_mismatch=True`；正常 81 例比值≈整十次幂，行为不变。
- 新增审计字段 `unit_extent_mismatch`（report.json / bundle / 批量审计 / run 日志）。
- `pipeline_wss_min/reporting.py`、`run.py`：批量审计与日志新增 `unit_extent_mismatch_cases` 待复核清单。

**验证（未跑 sklearn 依赖步骤，仅几何/配准层）**：
- 全量重算 `_resolve_unit_factor`：**只有** 上述 3 例被标记 `extent_mismatch`，factor→1000；其余 84 例（含 PENG_JI_MING 967789）变化<1%。
- 用 `compute_transform` 模拟 factor→1000 后归一化重心 |COM|：`WANG_BAO_SHAN` 0.50→0.14、`NIE_QUAN_ZHONG` 0.52→0.14（回到正常簇 ≈0.13），
  `ZHANG_HUAN_LI` 0.66→0.27（残余为其未被中心线覆盖的近端主动脉横向弯曲）。`center_on` 由 bifurcation 改 wall 无差异；改 `all` 反而把正常例推离原点，不采用。

**对应代码/文档**：`pipeline_wss_min/{config,preprocess,reporting,run}.py`。

**当前状态判断 / 待办**：
- 代码改完并通过语法/几何层验证，但 **未重跑** `preprocess`（本会话环境缺 sklearn）。需在原环境执行
  `python -m pipeline_wss_min.run --stage all`（或先 `--stage preprocess --cohort AG/slow --case ZHANG_HUAN_LI/WANG_BAO_SHAN` 冒烟），再重出 `当前_80例坐标QA`。
- 训练集内两例 `ZHANG_HUAN_LI`、`WANG_BAO_SHAN` 预计随重跑自动回正，无需剔除。
- `NIE_QUAN_ZHONG` 当初因“整体偏位”被 excluded；根因已修，可评估是否重新纳入（决策待定，本次未改 split）。
- `ZHANG_HUAN_LI` 残余 0.27 与 3 例中心线覆盖不全是更深层问题；如需完全同口径，可选：裁剪近端主动脉至中心线段 / 重抽全长中心线（VMTK），二选一待定。

## 2026-07-08｜当前 80 例最终 QA 分页图 + 旧口径图件归档 ✅

**本次主要修改**：
- 新增 `pipeline_wss_min.visualize_final_case_pages`，基于当前 `split_AG_wss_min_v1` included=80 直接生成最终人工复核图。
- 新增中文输出目录 `outputs/wss_min/当前_80例坐标QA/`，子目录按图件用途命名：
  - `01_点云同框_每20例/`
  - `02_STL同框_每20例/`
  - `03_逐病例XZ主轴视图_每20例/`
  - `04_峰值WSS正视图YZ_每20例/`
  - `05_峰值WSS俯视图XY_每20例/`
  - `06_LR左右轴复核/`
- 每类分页图按 20 例一页输出，共 4 页；同时写出 `当前80例_坐标QA汇总.csv` 和 `说明.md`。
- 将 `outputs/wss_min/` 下旧口径英文目录归档到 `outputs/wss_min/归档_旧口径_20260708/`，并把归档子目录改成中文描述名，保留旧图不删除。

**对应代码/文档/图件**：
- 代码：`pipeline_wss_min/visualize_final_case_pages.py`
- 文档：`pipeline_wss_min/README.md`、`outputs/wss_min/当前_80例坐标QA/说明.md`、`outputs/wss_min/归档_旧口径_20260708/说明.md`
- 当前主看目录：`outputs/wss_min/当前_80例坐标QA/`
- 旧图归档目录：`outputs/wss_min/归档_旧口径_20260708/`

**推进到实验步骤**：
- 推进到 WSS-only 最小化数据线的最终坐标 QA 阶段：当前 80 例已能按点云/STL 同框、逐病例主轴视图、峰值 WSS 正/俯视图成套人工复核。

**当前状态判断**：
- 当前输出顶层只保留两个入口：`当前_80例坐标QA/` 与 `归档_旧口径_20260708/`，降低误读旧图风险。
- 汇总 CSV 显示：STL 缺失 1 例（`slow/ZHANG_JUN_HUA`）；LR 符号继续观察 2 例（`fast/ZHANG_XIU_WEN`、`fast/CHEN_SHI_MING`）；非默认主轴来源 4 例（其中 `ZHANG_XIU_WEN` 为 `centerline_chord_ambiguous`）。
- 图件已生成但未重跑 `preprocess/global-stats/build-samples`；正式训练前仍需基于当前 split 执行 `python -m pipeline_wss_min.run --stage all`。

## 2026-07-08｜P3：剔除 `slow/NIE_QUAN_ZHONG` + split-aware 流程过滤 ✅

**本次主要修改**：
- 按人工复核决策，将 `slow/NIE_QUAN_ZHONG` 从 `training/splits/split_AG_wss_min_v1.json` 的 `train_cases` 移入 `excluded_cases`。
- split 计数更新为：train 56 / val 8 / test 16 / excluded 6 / pending 1 / total_found 87，当前 included=80。
- `pipeline_wss_min/config.py` 新增 split 读取工具：`load_split()`、`split_case_labels()`、`list_split_cases()`。
- `run.py` 默认按 `split_AG_wss_min_v1` 的 train/val/test included 病例运行；新增 `--all-raw` 仅用于原始目录排查。
- `global_stats.py` 默认只用 `train_cases` 计算 WSS 全局统计，避免 val/test/excluded 旧 bundle 混入。
- `build_samples.py` 默认只装配 split included 病例；manifest 写入 `split_name` 与 `partitions`。
- `visualize_stl_point_overlap.py --cases` 改为可指定 split 外病例做 QA，便于后续复核 excluded 个案。

**剔除原因**：
- `NIE_QUAN_ZHONG` 的主轴需要 `wall_pca_fallback`，STL→壁面点云 p95 约 `0.155 mm`，同框图显示整体偏位；这类问题更接近几何域/中心线质量异常，不适合第一版 `x,y,z -> wss` baseline 训练集。

**验证记录**：
- split 检查：included 80、train 56，`slow/NIE_QUAN_ZHONG` 不在 included/train，已在 excluded。
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m compileall pipeline_wss_min`
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m pipeline_wss_min.visualize_alignment --split split_AG_wss_min_v1 --tag p3_exclude_nie`
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m pipeline_wss_min.visualize_lr_check --split split_AG_wss_min_v1 --n 80 --tag p3_exclude_nie`

**图件**：
- `outputs/wss_min/alignment_viz_p3_exclude_nie/`
- `outputs/wss_min/lr_check/lr_side_grid_p3_exclude_nie.png`

**当前状态判断**：
- `NIE_QUAN_ZHONG` 已不会进入后续默认 preprocess/global-stats/build-samples；若旧 bundle 留在磁盘，也不会被默认统计和样本装配读取。
- 当前 split included=80；LR QA 剩余 `unreliable=2`，对应继续观察的 `fast/ZHANG_XIU_WEN` 与 `fast/CHEN_SHI_MING`。
- 正式训练前建议重新执行 `python -m pipeline_wss_min.run --stage all`，用新 split 与 split-aware 代码刷新 bundle、train-only WSS stats 和样本 manifest。

## 2026-07-08｜保留 3 例 STL/点云指定病例 QA 图 ✅

**本次主要修改**：
- `pipeline_wss_min.visualize_stl_point_overlap` 新增 `--cases` 参数，可直接指定病例列表出图，不再只能随机抽样。
- 对 P2 后保留继续复核的 3 例生成 STL/点云对比图：
  `fast/ZHANG_XIU_WEN`、`slow/NIE_QUAN_ZHONG`、`fast/CHEN_SHI_MING`。

**图件与审计**：
- 输出目录：`outputs/wss_min/stl_point_overlap_p2_keep3/`
- 逐例 3D 叠加：`01_selected3_stl_point_overlap_3d.png`
- 三投影叠加：`02_selected3_stl_point_overlap_projections.png`
- 局部放大：`03_selected3_stl_point_overlap_zoomed_projections.png`
- 同框点云：`04_selected3_same_frame_pointcloud_overlay.png`
- 同框 STL：`05_selected3_same_frame_stl_overlay.png`
- CSV：`selected3_stl_point_overlap_summary.csv`

**当前状态判断**：
- `fast/ZHANG_XIU_WEN` 与 `fast/CHEN_SHI_MING` 的 STL/壁面点云几乎逐点贴合，STL→点云 p95 分别约 `8.01e-05 mm`、`3.23e-05 mm`。
- `slow/NIE_QUAN_ZHONG` 的中位距离很小，但 STL→点云 p95 约 `0.155 mm`，明显高于另外两例；结合其 `main_axis_source=wall_pca_fallback` 与同框图中的整体偏位，继续保留复核是合理的。

## 2026-07-08｜P2：5 例 LR 符号加入 override，保留 3 例继续复核 ✅

**本次主要修改**：
- 按人工复核决策，将剩余 `roll_sign_unreliable` 中除 `ZHANG_XIU_WEN`、`CHEN_SHI_MING`、`NIE_QUAN_ZHONG` 以外的 5 例加入 `REGISTRATION_CASE_OVERRIDES`，统一使用 `roll_sign_mode="world_axis"`。
- 新增 override 病例：
  `slow/YIN_YU_RONG`、`slow/ZANG_YU_SHU`、`slow/LI_CHONG_ZENG`、`slow/XU_YI_CAI`、`slow/QIN_SI_FU`。
- 当前 override 总数为 9 例：此前 4 例 + 本次 5 例。

**验证结果**：
- 全量 81 例 LR QA：`unreliable=3`，输出图：
  `outputs/wss_min/lr_check/lr_side_grid_p2_override_keep3.png`
- 只读 transform 诊断确认剩余 3 例为：
  `fast/ZHANG_XIU_WEN`（`centerline_chord_ambiguous`）、
  `slow/NIE_QUAN_ZHONG`（`wall_pca_fallback`）、
  `fast/CHEN_SHI_MING`（`centerline_chord`）。

**当前状态判断**：
- 这 5 例不再作为待复核阻塞项；后续全量 preprocess 时会直接按 override 生效。
- 剩余 3 例继续保守保留，建议后续单独看 STL/点云同框与 LR 细图后再决定是否剔除、override 或保留审计标记。

## 2026-07-08｜P0/P1：LR 复核口径统一 + 主轴壁面 PCA 兜底 ✅

**本次主要修改**：
- **P0：复核图与正式配准口径统一。** `visualize_lr_check.py` 不再自己用单侧 `z>阈值` 选下游点，而是复用正式配准里的 `_select_wall_branch_side()`：对分叉两侧分别算横断面双峰分离度，取更像双髂支的一侧。这样消除了 `fast/LI_SHI_QIANG` 这类“配准正确、复核图全蓝”的诊断假象。
- **P0：QA 默认套用逐病例 override。** `visualize_lr_check.py`、`coord_check.py`、`visualize_alignment.py`、`visualize_stl_point_overlap.py` 均改为走 `registration_for_case()`，保证 QA 图看到的是正式 preprocess 会使用的配准配置；只有显式传 `--roll-source/--roll-sign-mode` 做旧口径对照时才关闭 override。
- **P1：主轴增加壁面 PCA 鲁棒兜底。** 默认仍用中心线入口端→分叉点 trunk 弦向；若该弦向无法清楚区分“单主干侧/双髂支侧”（两侧双峰分离度差过小），再用壁面点云 PCA 长轴，并按双髂支侧定号，使主干/分叉重新落到统一 +Z 视角。
- **P1：审计字段补全。** `RigidTransform` / `bundle.npz` / `report.json` / 批量审计新增 `main_axis_source` 与 `main_axis_wall_sep_delta`，后续能直接看到某例是 `centerline_chord`、`wall_pca_fallback` 还是 `centerline_chord_ambiguous`。

**诊断结论**：
- `fast/FAN_JIAN_MING`、`fast/LI_ZHEN_SHAN` 的中心线 trunk 弦向与壁面主方向夹角不大，属于血管自身弯曲/局部形态造成的视觉差异，不是主轴估计失败；本次保持 `centerline_chord`，避免过度旋转。
- `slow/NIE_QUAN_ZHONG` 是真实主轴问题：中心线只有局部 trunk/branch 信息，入口端→分叉弦向不能稳定代表主干方向；开启壁面 PCA 兜底后，从斜放恢复为主干近似竖直。
- 全 split 轻量诊断中，主轴来源为 `centerline_chord=76`、`wall_pca_fallback=4`、`centerline_chord_ambiguous=1`。fallback 病例进入审计清单，便于后续人工复核。
- 全量 81 例 LR QA 重跑后，待人工确认从 9 例降至 **8 例**；其中 `slow/CHENG_GUANG_SEN` 因主轴兜底后 trunk-bending 置信提高，已从 unreliable 中移出。剩余 8 例主要是 `roll_sign` 符号锚置信弱，红蓝两侧本身大多已清楚分开，属于 P2 是否加入个案 override 的范围。

**对应代码/文档/图件**：
- 代码：`pipeline_wss_min/registration.py`、`config.py`、`preprocess.py`、`reporting.py`、`visualize_lr_check.py`、`coord_check.py`、`visualize_alignment.py`、`visualize_stl_point_overlap.py`
- 主轴前后对照图：`outputs/wss_min/main_axis_p1_compare.png`
- 新 81 例对齐图：`outputs/wss_min/alignment_viz_p0p1_main_axis/`
- 新全量 LR 复核图：`outputs/wss_min/lr_check/lr_side_grid_p0p1_all81_full.png`
- 剩余待确认 9→8 例复核图：`outputs/wss_min/lr_check/lr_side_grid_p0_remaining9.png`

**验证记录**：
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m compileall pipeline_wss_min`
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m pipeline_wss_min.visualize_alignment --split split_AG_wss_min_v1 --tag p0p1_main_axis`
- `/public/newhome/cy/.conda/envs/GNN/bin/python -m pipeline_wss_min.visualize_lr_check --split split_AG_wss_min_v1 --n 81 --tag p0p1_all81_full`

**当前状态判断**：
- P0/P1 核心修复已完成：QA 图与正式处理逻辑一致，中心线主轴异常病例有可审计 fallback，且不会把普通弯曲病例强行拉直。
- 正式训练前仍需按新配准口径全量重跑 `preprocess -> global-stats -> build-samples`，否则旧 `bundle.npz` 仍可能混用旧坐标。
- 后续 P2 建议只处理剩余 8 例 `roll_sign_unreliable`：逐例决定是否接受 `world_axis_weakbend`，或加入 `REGISTRATION_CASE_OVERRIDES`。

## 2026-07-08｜左右轴 wall_branches 改造 + 13 例复核图诊断（待 P0/P1 修复）🔍

> 本条记录 2026-07-08 复核发现与待修项，代码收尾在该记录时尚未完成。

**背景（2026-07-08 已改）**：左右轴配准从「中心线分支 PCA + 世界轴定号」整体改为
「分叉下游**壁面点**双峰求真实 L-R + 主干弯曲(A-P)×主轴内在手性 + 世界轴两锚交叉校验」，
逐级回退 `wall_branches → branches → curvature`；并对四个 QA 脚本加 `--tag` 出到 `_new` 目录（旧图保留）。

**复核发现（诊断）**：
1. 左右轴纯度（两髂支沿 X 轴分离度）全 81 例 **≥0.97**（旧法最低 0.56），朝向抖动/左右翻转基本消除；
   **80/81 走 `wall_branches`**，`slow/WANG_BAO_SHAN` 因分叉原点落在血管轴向端点回退 `branches`（仍出正常 Y）。
2. `outputs/wss_min/lr_check/lr_unreliable_13cases_20260708.png` 复核图**结构正确**，但
   **`fast/LI_SHI_QIANG` 的「全蓝」是该图下游选取的假象**：该图用**单侧** `z>阈值`，恰好抓到只含一条髂支的
   一侧（1472 点全 x<0）；而配准里的 `_wall_branch_lr_axis` 是**两侧自适应**，选中的另一侧（8639 点）
   正常双峰（左 39% / 右 61%）→ **LI_SHI_QIANG 的左右轴其实是对的**。
3. 待复核 13 例中 **12 例可见清楚蓝左/红右**（轴对，仅主干弯曲太弱无法自证符号）；
   真正「长得歪」的是 `fast/FAN_JIAN_MING`、`fast/LI_ZHEN_SHAN`、`slow/NIE_QUAN_ZHONG`——
   属 **主轴倾斜**（主干方向估计问题），与左右轴无关。
4. 已加 4 例 override（`FAN_JIAN_MING`/`LI_SHI_QIANG`/`LI_ZHEN_SHAN`/`ZHANG_HAO` → `world_axis`），
   **待复核由 13 降至 9**：`ZHANG_XIU_WEN`、`YIN_YU_RONG`、`NIE_QUAN_ZHONG`、`ZANG_YU_SHU`、
   `LI_CHONG_ZENG`、`CHENG_GUANG_SEN`、`XU_YI_CAI`、`CHEN_SHI_MING`、`QIN_SI_FU`。

**待修复 / 待决**：
1. 把 13 例复核图的**下游选取改成与配准一致的两侧自适应**（消除 LI_SHI_QIANG 假象）；
   或直接用已是两侧逻辑的 `visualize_lr_check.py` 按当前 9 例重生成。
2. 剩余 9 例：目视基本是正常 Y，定夺是「接受 `world_axis` 回退」还是逐个加入 `REGISTRATION_CASE_OVERRIDES`。
3. 主轴倾斜的 3 例（`FAN_JIAN_MING`/`LI_ZHEN_SHAN`/`NIE_QUAN_ZHONG`）属 `main_axis` 估计问题，单独一轮处理。
4. 配准口径定稿后，对整个 split 重跑 `preprocess`（现有 bundle 仍是旧坐标），再跑 `global-stats` / `build-samples`。

**对应代码/文档/图件**：
- 代码：`pipeline_wss_min/registration.py`（`_wall_branch_lr_axis` 两侧自适应 + `_trunk_bending_*` + `_sign_lr_axis` 两锚交叉校验）、
  `config.py`（`roll_source`/`roll_sign_mode`/`downstream_wall_axis_frac`/`roll_sign_min_cos`/`REGISTRATION_CASE_OVERRIDES`）、
  `preprocess.py`、`reporting.py`、`visualize_lr_check.py`（新增）、
  `visualize_alignment.py`/`coord_check.py`/`visualize_stl_point_overlap.py`（加 `--tag`；小图 ⚠ 改为「回退/待复核才告警」）
- 图件：`outputs/wss_min/lr_check/`、`alignment_viz_new/`、`coord_check_new/`、`stl_point_overlap_new/`
- 审计：`data_wss_min/pipeline_reports/preprocess_audit_latest.json`（`roll_sign_unreliable_cases`）

**当前状态判断**：
- 左右轴主问题（朝向不一致 / 偶尔翻转）已在配准层修复并有量化证据（纯度 ≥0.97、`det=+1`、可复现）。
- 剩余为三件收尾：复核图取样口径统一、9 例符号确认、3 例主轴倾斜；均不阻塞「现算配准看新朝向」。

## 2026-07-08｜4 例 roll 符号 override + 重跑 preprocess ✅

**本次主要修改**：
- 新增 `config.REGISTRATION_CASE_OVERRIDES` 与 `registration_for_case()`：对 trunk_bending 与世界轴冲突、但 `wall_branches` 左右轴仍可信的个案，显式锁定 `roll_sign_mode="world_axis"`（几何与 weakbend 回退一致，仅消除 unreliable 标记）。
- 已处理 4 例：`fast/FAN_JIAN_MING`、`fast/LI_SHI_QIANG`、`fast/LI_ZHEN_SHAN`、`fast/ZHANG_HAO`；重跑 preprocess 并刷新审计。

**对应代码/文档**：
- 代码：`pipeline_wss_min/config.py`、`preprocess.py`、`README.md`
- 审计：`data_wss_min/pipeline_reports/preprocess_audit_latest.json`（`roll_sign_unreliable` 13→9）

**推进到实验步骤**：
- split 内 4 例已纳入正式 override 清单；审计待复核由 13 例降至 9 例。

**当前状态判断**：
- 4 例 `roll_sign_reliable=True`、`roll_sign_source=world_axis`，bundle 旋转矩阵与 override 前 weakbend 结果逐元素一致。
- 剩余 9 例仍待同类处理或人工 LR 复核（含 val `XU_YI_CAI`、test `CHEN_SHI_MING` 等）。

## 2026-07-08｜P4 主干横向二次居中 + 当前 80 例 QA 刷新 ✅

**背景问题**：
- 用户复核 `当前_80例坐标QA/01_点云同框_每20例/page02/page03` 时发现两例整体偏离中心轴。
- 量化排查确认不是已剔除的 `slow/NIE_QUAN_ZHONG`，而是当前 split 内的两例 train 病例：
  - `slow/WANG_BAO_SHAN`：page02 第 24 例，低位主干横向 offset_frac≈`0.278`。
  - `slow/ZHANG_HUAN_LI`：page03 第 46 例，低位主干横向 offset_frac≈`0.236`。

**本次主要修改**：
- `RegistrationConfig` 新增 `trunk_centering=True`、`trunk_centering_quantile=0.20`、`trunk_centering_min_offset_frac=0.08`、`trunk_centering_stat="median"`。
- `registration.py` 在主轴/roll 旋转确定后，检测低位 20% 主干壁面点的横向中心；若 normalized offset 超阈值，则只做非主轴两个方向的刚性平移，并折算进 `RigidTransform.centroid`。
- 新增审计字段：
  - `transform_trunk_centering_applied`
  - `transform_trunk_centering_offset_mm`
  - `transform_trunk_centering_offset_frac`
- `preprocess.py`、`reporting.py`、`coord_check.py`、`visualize_stl_point_overlap.py`、`visualize_final_case_pages.py` 同步写入/展示该字段。
- `visualize_final_case_pages.py` 新增 `07_二次居中病例复核/二次居中病例_STL点云三视图.png`。

**验证结果**：
- 全 80 例复测中，仅 `slow/WANG_BAO_SHAN` 和 `slow/ZHANG_HUAN_LI` 触发二次居中。
- 修正后两例低位主干残余横向偏移接近 0：
  - `WANG_BAO_SHAN`：mean_r≈`0.0016`，median_r≈`0.0000`。
  - `ZHANG_HUAN_LI`：mean_r≈`0.0004`，median_r≈`0.0000`。
- 已重跑两例 bundle：
  - `data_wss_min/AG/slow/WANG_BAO_SHAN/bundle.npz`
  - `data_wss_min/AG/slow/ZHANG_HUAN_LI/bundle.npz`
- 已刷新当前 80 例 QA：
  - `outputs/wss_min/当前_80例坐标QA/01_点云同框_每20例/page02_点云同框_病例21-40.png`
  - `outputs/wss_min/当前_80例坐标QA/01_点云同框_每20例/page03_点云同框_病例41-60.png`
  - `outputs/wss_min/当前_80例坐标QA/07_二次居中病例复核/二次居中病例_STL点云三视图.png`
  - `outputs/wss_min/当前_80例坐标QA/当前80例_坐标QA汇总.csv`
- 二次居中前的旧 QA 图已归档到：
  - `outputs/wss_min/归档_旧口径_20260708/旧版_80例坐标QA_二次居中前/`

**注意事项**：
- 两例只能匹配到 `stl_data/name_data/` 中的 STL，STL 与 CFD wall 点云不是逐点贴合口径；专项图中 STL-点云距离较大，不能作为本轮剔除依据。
- 当前仓库没有 `data_wss_min/wss_global_stats.json` 与默认 samples 目录，因此本轮未强行生成训练样本；正式训练前仍需按当前配准口径全量跑 `preprocess -> global-stats -> build-samples`。

## 2026-07-07｜随机 10 例点云/STL 同框叠加图 + 推进记录拆分 ✅

**本次主要修改**：
- `pipeline_wss_min.visualize_stl_point_overlap` 新增两张“同框叠加”图，不再只做逐病例小格子：
  - 10 例壁面点云放在同一张图、同一坐标范围、同一 3D 相机视角下叠加；
  - 同一批 10 例 STL 顶点/面片也放在同一张图中叠加。
- 两张同框图都保留原点参考线和 X-Z / Y-Z / X-Y 三投影，用来直接检查分叉原点是否对齐、主干/分叉朝向是否一致。
- 新建本文档，将 WSS-only 最小化路线的详细推进记录从 V3P 大日志中独立出来；原 `代码修改与实验推进记录.md` 后续保留 V3P / 训练主线记录和拆分说明。

**对应代码/文档**：
- 代码：`pipeline_wss_min/visualize_stl_point_overlap.py`
- 文档：`README.md`、`pipeline_wss_min/README.md`、`docs/README.md`、`docs/02-推进与变更/_archive/WSS最小化/WSS最小化预处理流程_搭建与交接记录_2026-07-07.md`、`docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`、`docs/02-推进与变更/代码修改与实验推进记录.md`
- 图件：
  - `outputs/wss_min/stl_point_overlap_20260707/04_random10_same_frame_pointcloud_overlay.png`
  - `outputs/wss_min/stl_point_overlap_20260707/05_random10_same_frame_stl_overlay.png`

**推进到实验步骤**：
- 推进到 WSS-only 最小化数据线的几何输入 QA：从“逐病例检查 STL/点云是否贴合”进一步推进到“多病例放入同一解剖坐标框架后是否原点对准、视角一致”。

**当前状态判断**：
- 同一随机种子 `20260707` 抽取的 10 例中，壁面点云同框叠加与 STL 同框叠加均显示：分叉附近围绕统一原点，主干整体位于 `z<0`，分叉/出口位于 `z≈0` 到正值区域。
- STL 版本与点云版本视觉结论一致，支持当前分叉原点配准和逐病例归一化可作为 `x,y,z -> wss` 简化输入的前置规整步骤。

## 2026-07-07｜随机 10 例 STL-点云逐病例重叠 QA 图 ✅

**本次主要修改**：
- 新增 `pipeline_wss_min.visualize_stl_point_overlap`：从 `split_AG_wss_min_v1` 随机抽取 10 个可匹配 STL 的病例，读取壁面 ascii 点云与 `stl_data/` 中对应 STL。
- 图件生成时不做 ICP 或额外拟合，只把 STL 按 ascii/STL 原始单位比例换到 pipeline 坐标尺度，再与壁面点云套用同一个单位换算、分叉原点刚性配准和逐病例归一化尺度。
- 输出 3D 叠加图、固定坐标范围三投影图、局部放大三投影图与 CSV 距离审计，用于确认后续 `x,y,z -> wss` 的几何输入确实处于同一处理框架。

**对应代码/文档**：
- 代码：`pipeline_wss_min/visualize_stl_point_overlap.py`
- 文档：`pipeline_wss_min/README.md`
- 图件：`outputs/wss_min/stl_point_overlap_20260707/01_random10_stl_point_overlap_3d.png`、`02_random10_stl_point_overlap_projections.png`、`03_random10_stl_point_overlap_zoomed_projections.png`
- 审计：`outputs/wss_min/stl_point_overlap_20260707/random10_stl_point_overlap_summary.csv`

**推进到实验步骤**：
- 推进到 WSS-only 最小化数据线的几何 QA：在解剖原点坐标系之外，补充 STL 面片与壁面训练点云的一致性检查，为只输入 `x,y,z` 的壁面 WSS 预测提供坐标规整证据。

**当前状态判断**：
- `split_AG_wss_min_v1` 中 80 例可匹配 STL，1 例未匹配到 STL；随机种子 `20260707` 抽取的 10 例全部成功出图。
- 10 例 `native_to_stl≈1000`，说明 STL 基本是 ascii 原生坐标的毫米版本；套用 pipeline 坐标尺度后，STL 到壁面点云的 p95 NN 距离约 `1.19e-05` 到 `9.66e-05 mm`，视觉上三投影完全贴合。
- 当前证据支持：壁面点云与 STL 几何源一致，且经新版分叉原点配准后可放入同一个坐标框架继续做 `x,y,z -> wss`。

## 2026-07-07｜解剖原点版配准 v2 + 全量坐标 QA 图 ✅

**本次主要修改**：
- `pipeline_wss_min` 的坐标配准从“壁面重心原点 + 中心线主轴 + 曲率 roll”升级为**分叉原点版解剖刚性配准 v2**：默认读取 `centerline.vtp` 中的 `DistToBifurcation` / `BranchId`，以分叉区域为原点，用入口端→分叉 trunk 方向对齐 +Z，用分叉后远端分支平面固定横轴；缺 VTP 拓扑时保留旧策略兜底。
- `bundle.npz`、单病例 `report.json`、批量审计 CSV 新增 `origin_kind` / `main_axis_mode` / `roll_source` 追踪字段，避免 fallback 病例静默混入。
- 重新生成 WSS-min 坐标 QA 图：`outputs/wss_min/alignment_viz_anatomical_v2/` 与 `outputs/wss_min/coord_check_20260707_anatomical_v2/`；81 例 included 全部 `origin=bifurcation`、`main=inlet_to_bifurcation`、`roll=branches`，无单位异常，`det(R)=1`，最大正交误差约 `3e-16`。
- 更新 `pipeline_wss_min` README、WSS-min 交接记录、根 README 与 docs 索引，说明新版配准口径、图件位置与旧 bundle 需重跑 preprocess 的注意事项。

**对应代码/文档**：
- 代码：`pipeline_wss_min/config.py`、`raw_io.py`、`registration.py`、`preprocess.py`、`reporting.py`、`visualize_alignment.py`、`coord_check.py`
- 文档：`pipeline_wss_min/README.md`、`docs/02-推进与变更/_archive/WSS最小化/WSS最小化预处理流程_搭建与交接记录_2026-07-07.md`、`README.md`、`docs/README.md`
- 图件：`outputs/wss_min/alignment_viz_anatomical_v2/*.png`、`outputs/wss_min/coord_check_20260707_anatomical_v2/*.png`

**推进到实验步骤**：
- 推进到任务 A 的 WSS-only 最小化数据线：完成 `x,y,z -> wss` 前置的解剖坐标统一与全量 QA，可作为下一步重跑 `preprocess -> global-stats -> build-samples` 和壁面点数扫描的输入标准。

**当前状态判断**：
- 新版坐标系比旧壁面重心原点更符合“同一视角 / 同一框架 / 基本朝向一致”：分叉原点固定，主干统一位于 `z<0`，分叉/出口统一位于 `z≈0` 到正值区域。
- 注意旧 `data_wss_min/*/bundle.npz` 若在本次前生成，仍是旧坐标，正式训练前必须全量重跑 `pipeline_wss_min.run --stage preprocess`。
## 2026-07-16｜v4 终签合规病例对齐图重绘 ✅DONE

**本次主要修改**：为 `visualize_v4_cutover.py` 增加 `--final-eligible-only` 与 `--training-whitelist`。前者强制读取终签 manifest，只渲染 `final_eligible=true` 的病例；后者再按正式训练白名单过滤。小图标题同步实际分母；新旧 AG 对照也不会再把已排除的 `WANG_DENG_FENG` 回填进图。

**对应代码/产物**：`pipeline_wss_min/visualize_v4_cutover.py`；`docs/02-推进与变更/assets_新队列审计/alignment_v4_cutover_{final_eligible,training_eligible}_20260716/`；终签真源 `data_wss_min/pipeline_reports/v4_cutover_20260715_1921/{v4_final_dataset_manifest.json,v4_training_whitelist.json}`。

**推进到实验步骤**：已导出两套只读复核图：几何终签版 AG 76、AAA 63（ruptured 31、unruptured 32），共139例；正式训练可用版 AG 76、AAA 57，共133例。后者额外排除 6 个已有入口/标签质量排除的 AAA。两套中对应排除病例均不在任何小图、叠加图、分组图、数值 CSV/JSON 或专项页中。

**当前状态判断**：这是展示与复核材料的只读再导出，不修改 bundle、白名单、split、统计、训练结果或发布状态；对老师展示优先使用133例训练可用版，139例几何终签版保留用于区分“几何签核”与“训练质量”的两个口径。
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

## 2026-07-21｜SA1-scale 矩阵：全点/10k/降center×大k/宽度/Stem 17 臂落地与提交 🚀

**背景**：以 SA1 分组矩阵（Q1V ball16=0.2763、raw KNN-8=0.2712 且唯一改善 high-WSS、KNN-8-cover 归一化最高 0.6042）为锚，按导师意见开五个方向：①全点支撑 + SA1 knn_cover k∈{64,128,256}（center 固定 500/125/32 与三层比例 0.1/0.25/0.25 两套）；②KNN-8-cover 基准下降 SA1 center {250,125} × 升 k {64,128} 全 2×2；③random10000 × 同一 k 网格（AG 欠 1w 点病例回退全点）；④knn{8,10}_cover 的 width=64 对照；⑤w64 KNN-8-cover 锚点上的 Stem 6→32→64 变种（6→256→512→64 条件触发）。另加 bridge（random5000+500center+k64）分离"加点数"与"加 k"。

**代码修改**（`training_wss_min/`）：
- `config.py`：新增 `ModelConfig.stem_channels`（空=历史 `(width,width)`；末元素必须==width，仅 pointnetpp）与 `DataConfig.support_allow_undersized`（仅 random 族；欠点病例回退全点），`validate_features` 同步加校验。
- `baseline_models.py`：Stem 构造支持显式通道序列（默认路径与历史 state_dict 逐 key 兼容，102 key 实测一致）；**新增 `_knn_group_chunked`**：torch_cluster 的 CUDA knn 核存在 `k<=100` 内部断言（`knn_cuda.cu:98`，CPU 无此限制，故此前未暴露），k>100 时走分块 cdist+topk 精确 KNN（每块 ~2^27 距离项，fp32、no_grad、禁 autocast），k≤100 保持 torch_cluster 原路径与既有 knn8/knn10 矩阵逐位一致。CPU/CUDA 边集合等价性、k>n 钳位、cover 100% 覆盖均单测通过。
- `dataset.py`：`support_allow_undersized` 门控原"每例点数≥support_n"硬断言；欠点病例落入 `sample_indices` 的 `k>=n→arange(n)` 全点分支。
- `tools/preflight_v4_jobs.py`：epoch 重采样门禁改为寻找第一个真正被下采样的病例做探针（全点/关闭 resample 记 `not_applicable`，否则 D1/D3 会误报）；新增 `--probe-batch largest` 用 train 最大的 batch_cases 个病例做 CUDA 冒烟（原首序探测取小 AG 例，低估峰值显存 >20×）。
- 新增 `tools/audit_sa1_scale_matrix.py`（兼容空 `sa_center_counts` 的比例制 center，eval 口径 `random_start=False`；全点/比例合同按 `--max-cases-large 4` 抽样含最小/最大例）、`tools/prepare_pointnetpp_sa1_scale_matrix.py`（17 配置 + manifest + `--variant-b` 物化 D5-B）、`cluster/{preflight,run}_pointnetpp_sa1_scale.slurm`、`cluster/submit_pointnetpp_sa1_scale_matrix.py`。

**提交前验证**：单元冒烟 10 项全过（state_dict 回溯兼容、stemA/B 前向与参数量 877.7k/1.043M、比例 center=ceil(0.1N) 且 cover 覆盖 100%、wall=0 全点恒定、D3 回退/护栏）；2-epoch GPU 干跑 5 臂全过（d2_c125_k128 83s/1.6GB、d3_k256 14s/2.4GB、d1_fixed_k256 17s/4.8GB@batch8、d1_prop_k256 44s/**23.3GB@batch4**、d5_stemA 45s/1.8GB；`train_sampled_points=3,138,802` 确认全点支撑生效）；**按预注册门槛（>19GB）d1_prop 族整体降 batch_cases=2**（判读时注意与 batch8 臂的混杂）。两种全点变体的 eval 冒烟（test27 全点云分块解码）通过后删除全部 smoke run。首轮干跑正是被 torch_cluster k≤100 断言击穿（4/5 臂失败），回退实现后重跑全过——冒烟拦截了一次必然的整矩阵集群失败。

**提交记录**：prepare manifest `training_wss_min/preflight/pointnetpp_sa1_scale_prepared.json`（17 配置，audit 冒烟 16 唯一分组合同）；门禁 Job `10746`（新 audit + `preflight_v4_jobs --probe-batch largest`）→ 训练 array `10747_[0-16]%4`（`afterok`，48h/64G/1GPU，训完自动 eval best+last test27 + compare_best_last）。提交清单 `preflight/pointnetpp_sa1_scale_submission.json`。跑完后用 `tools/analyze_pointnetpp_sa1_scale_results.py`（待写）汇总并判读 D5-B 触发。

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
