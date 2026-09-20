# V3 历史场重建分析

仅当 run 的 `config.snapshot.json` / `meta.exp_id` 确认为 V3P/V3D 时读取。V3 是历史路线；其路径 G/F 计划和数值锚点不构成当前任务的新训练计划。

## 最少入口与产物

1. `docs/01-任务/任务A/03-V3路线/README.md` 文首当前状态与口径对照。
2. [v3-docs.md](v3-docs.md) 的 L0/L1 中与当前实验有关的块；历史 run 通过 exp_id/job ID 定向搜索，不必从最新日志逐块回溯。
3. [paths.md](paths.md)：用 `experiment_index.csv` 或用户路径锁定 run，读取 `summary.json`、`history.csv`、`config.snapshot.json`、存在的 `run_manifest.json`。

## 对照与判据

| 分支 | 基线身份与主指标 | 判据来源 |
| --- | --- | --- |
| V3P | AsymW-a job 4957；`best_wss`、`wss_r2_wss`；三 seed 身份 4957/4999/5000 | 该实验跟踪记录与历史计划 |
| `V3P-F-*` | 立项声明的中间基线（如 VelSup 5266、PGradFeat 5277）与母版分别注明 | `02-历史路线/V3_发散优化探索路线.md` §6 |
| `V3P-G-*` | 4957；G1 阶梯还比较上一档 | `00-历史主线/路径G_下一代架构与精度突破方案_2026-06-05.md` §10；G0 oracle 读 §3 |
| V3D | WSS-01 post-4901；不能与 V3P 比绝对值 | 对应跟踪日志/待办的分域判据；val15 5331 不升级成基线 |

表中身份用于定位，实际基线数值仍读取其 run。所有路径均相对于 `docs/01-任务/任务A/03-V3路线/`。

主结果读取 `summary.json` 的 `test_metrics` / `test_metrics_best_wss`，并确认键对应 checkpoint。辅助看 `r2_p`、`r2_vel_mag`、`wss_x/y/z`；路径 G 判定若需病例 p95/Pa、分量/角度、高 WSS 区域，则读已有 regional/full eval。缺少门禁所需证据时不能声称全部通过，也不自动补跑 test。

曲线重点：收敛、过拟合、`best_model` 与 `best_wss` 分叉，以及辅助目标改善但 WSS 回退。val→test 脱节单独说明。不要只凭一个 pooled 指标解释病例级或结构性突破。

## 对比图

确需 V3 对比图时，核实实际脚本参数后使用 `training.scripts.plot_experiment_analysis_compare` 与 `training.scripts.plot_training_history`；调用示例见 [paths.md](paths.md)。输出仍在 `outputs/field/plots/analysis_compare/<slug>/`，按 `docs/00-规范与记录/实验分析对比图目录说明.md` 记录。

V3D 用其自己的指标、checkpoint 与 WSS-01 run，不能与 V3P 画成同口径优劣柱图。G0 only 读取 `outputs/field/f0_decision/v3_g0_oracle_*.json`，不因分析而重训。

## 下一步与历史回填

既有 V3 下一跳从相关待办、README 与实验记录核实；路径 G 的 G0/oracle 和 H 观察池门禁只用于解释历史决策，用户未要求恢复路线时不自动开训。用户要求新设计时可以基于本次证据提出新假设，标明与冻结历史计划的关系。

新分析写入 `docs/02-推进与变更/代码修改与实验推进记录.md` 文首，并补充 `V3_实验执行跟踪日志.md` 相应记录；F 结论有实质修订时补 `V3_发散优化探索路线.md` §10。保留原日期与历史结论，不将旧 TODO 标成当前待执行；确有路线判断变化再同步 README/待办。G0 only 更新已有审计说明与推进记录即可。

不回填 V1 的任务 A 状态表或跨路线 xlsx，除非用户要求且按其入账规范完成。
