# WSS 最小化路线代码修改与实验推进记录

> 用途：单独记录 `pipeline_wss_min/` 这条 WSS-only 最小化数据线的代码、坐标 QA、图件和实验推进。
> V3P / 训练主线 / 通用代码修改记录见：[代码修改与实验推进记录](代码修改与实验推进记录.md)。
> 当前执行入口：[WSS 最小化训练实验跟踪](WSS最小化_训练实验跟踪.md) / [pipeline_wss_min README](../../pipeline_wss_min/README.md) / [training_wss_min README](../../training_wss_min/README.md)。

## 2026-07-10｜第五轮优化计划 v0.1：冻结 AG 单队列、WSS 标量与 field-R² 主目标 ⏳待交叉审查

**本次主要修改**：
- 新建第五轮优化计划讨论稿，明确状态为“待其他智能体交叉审查、未授权执行”；本次未修改训练代码、split、数据 manifest，未提交任何实验。
- 冻结第五轮边界：AG 为唯一主实验队列；AAA/ILO 分队列治理且异常单元先隔离；任务保持峰值收缩期 WSS 单标量、单输出头；完整壁面 `R²_field` 为主指标。
- 把第五轮执行顺序收敛为：三队列清单治理 → AG train-fit/density 诊断 → AG 13/26/40/53 learning curve → 条件性单任务结构优化 → 医工交叉审计。
- 纳入新队列讨论中发现的待隔离项：`ILO/LIU_BAO_JUN-0/after` 近零 WSS、7 个 `vf-in` 数量级/口径异常单元、方向 watch 和跨队列病人分组风险。

**对应代码/文档**：
- 新计划：[WSS最小化_第五轮优化计划_待交叉审查.md](WSS最小化_第五轮优化计划_待交叉审查.md)
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
