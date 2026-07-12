# WSS 最小化路线 · 训练实验跟踪

> 用途：跟踪 `training_wss_min/`（PointNeXt 残差 baseline，独立于 V3P `training/`）的逐轮实验：
> 设计、完整指标表、结论、待办。**每完成一轮/一次任务，回填本文档。**
> 上位：[WSS最小化_代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md) / [training_wss_min/README](../../training_wss_min/README.md)。

## 第六轮｜A/B/D XYZ 尺度诊断（2026-07-12 ⏳RUNNING）

- 目的：判断旧纯 XYZ 较差是否主要由“坐标逐病例归一化到 `[-1,1]`”丢失物理尺度造成。
- A=`xyz`；B=`xyz+coord_scale`；D=`xyz+abscissa_norm+local_radius+curvature`。三组均为 dev1 / fixed FPS-2000 / B1 fixed target-weight / val-only / `seed={1234,7,2025}`。
- 9 个配置唯一变量审计和 B 组特征构造通过；Jobs `7029–7037` 已提交，记录 `training_wss_min/cluster/logs/submitted_20260712_120420.txt`。
- 预注册：B−A 的三 seed 均值在 field/casemean 均 `>0.02` 才判为可辨识尺度信号；D−B 报告显式几何增量和 seed 方差。
- 范围声明：B 是尺度诊断，不是严格物理尺度纯 XYZ 终审；本轮不读 test16，不外推几百/几千例数据上限。
- 详细协议与 Job 表：[WSS最小化_第六轮XYZ尺度诊断计划与执行](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)。

## 汇报｜A0E-ctrl 两例 postview（2026-07-12 ✅DONE）

- 模型 `r5_a0e_b1_ctrl_s1234`；病例 `slow/WU_FENG_YAN`、`fast/RAN_QING_BO`（均为 train）。
- 产物：`docs/03-汇报材料/figures/WSS最小路线_20260712/postview_a0e_ctrl/`（`*__surface_wall.vtp` 含 CFD/Pred/Error；映射覆盖率 100%）。
- 同点 wall R²：`0.3225` / `0.4959`；脚本 `training_wss_min/export_wss_postview.py`。

## 第五轮｜F0 结案（2026-07-12 ✅DONE / 科学结案·工程未达标）

- 全部 §8.1 机制问题均有可复核裁决 → **第五轮科学结案**；dev1 field/casemean ~0.31/0.21 ≪ 0.70 → **未达内部工程目标**（§8.2），未跑 OOF（无达标候选）。
- 机制链：拟合足（A0D）→ 当前协议泛化锚点 ~0.34（A0E）→ 现有 61 例范围内 ~0.31 暂时平台（LC）→ 可部署 BC 无新杠杆（B-BC）→ 标签噪声小（CFD，R²_cap ~0.92–0.96）。该结论限于当前数据池、输入和模型协议，不外推数千个高质量独立病例的上限。
- 报告：`training_wss_min/runs/_round5/final_report/round5_final_report.md`。可选后续（非部署路径、待用户定）：oracle RCR 增量探针 / P2 loss 探针。

## 第五轮｜CFD 可信性审计 §6（2026-07-12 ✅DONE）

- read-only；dev61。产物 `training_wss_min/runs/_round5/cfd_audit/{cfd_audit.py,cfd_per_case.csv,cfd_summary.json,cfd_audit_report.md}`。
- peak 相位按**固定步**（1162/idx21）统一取，结构上无跨病例相位错配；仅 3/61 真峰晚 ≥12 步（含 2 val：`LIU_JUN_FENG` +13.8%、`CHENG_GUANG_SEN` +12.2%）→ n=8 val 数值脆弱。
- 近壁 QA 全清（normal_invalid=0、basis≤0.066、radial≤0.066、mesh≥786k）；高 WSS 尖峰为真实几何热点（不与 QA 旗标共现），但单节点 max 受尖峰主导，报告以 p95/p99 为准。
- **重复几何**：独立证实 `HOU_SHEN_QIAN`=`KANG_XI_MING` 同一几何（且均 dev1 train）→ OOF 须整组；`LIU_XI_QUAN`（excluded）peak 全零损坏。
- **裁决：~0.31 上限主要不是 CFD 标签噪声**——复现 floor ~2% → R²_cap ~0.999；即便 10–15% per-case 噪声也仅 cap 到 0.92–0.96，远高于 0.31 → 真实模型/信息上限。

## 第五轮｜B-BC 资产审计（2026-07-12 ✅DONE / 可部署 B-BC 关闭）

- read-only 审计 61/61 dev 病例 BC 完整（udf-inlet.c + vf-in + 5 压力监测）。产物 `training_wss_min/runs/_round5/bc_audit/{audit_ag_bc.py,bc_per_case.csv,bc_summary.json,bc_audit_report.md}`。
- **决定性发现：入口流量不是病人特异的。** Fourier 流量模板（a0..b8,w,T,A1/B/D/E/n）在所有病例逐字节相同；入口 BC 为平速 `v=1e-6·template/area`，故 `Q=v·area=1e-6·template` 与面积无关 → 各病例入口流量本质相同（dev CoV≈1.85e-4，Fourier↔实测比 0.999–1.000）。唯一 per-case 入口标量是入口**面积**（几何可得）。
- **可部署 vs oracle**：可部署 = 入口面积/速度/流量（但零方差或与几何冗余，非信息量）；**唯一有跨病例方差的是出口 RCR（CoV 0.55–0.67）与出口压力/流量分配，全部 `oracle_non_deployable`**（CFD 设定/解产物，新病人不可知）。
- 数据质量旗标：两对拷贝 BC（`HOU_SHEN_QIAN=KANG_XI_MING` 均 dev1 train、`LIU_XI_QUAN=LI_BING_YI`）→ 其面积/RCR 不独立；`vf-outri` 20 例退化（15 在 dev1）；两例异常入口监测均已 excluded。
- **裁决：可部署 B-BC 关闭**——无可部署、有信息量、病人特异的 BC 输入可加。跨病例 WSS 差异由出口 RCR（oracle）驱动，可部署几何模型无法在部署期获取。剩余仅两条：oracle RCR 增量探针（仅量化上限，非部署，需全局输入代码）、P2 loss 探针（末条可部署杠杆）。

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
