# wss_v52d_merge_20261001

母库全量审计后把 v5.2c 一次合并成 v5.2d 的全部脚本和记录。说明文档：
`docs/02-推进与变更/04-数据处理与CFD/母库全量审计_2026-10-01.md` §11。没有训练。

| 步骤 | 文件 | 作用 |
| --- | --- | --- |
| 清单 | `units.json` | 18 个改动单元：同网格重算 1、从 STL 重建 1、出口改名 16 |
| 01 | `01_swap_raw.py` | 两例重算结果换入 `data_new`，旧文件归档后删除 |
| 02 | `02_refresh_audits.py` | 刷新这 18 个单元的拓扑、壁面、求解审计 |
| 03 | `03_build_stage.slurm`、`03b_wtq_centerline.slurm` | 在暂存区重建快照与特征包（作业 16438、16442、16443） |
| 04 | `04_integrate.py` | 根目录改名为 v5.2d，换入 18 个单元（`integrate_log.json`） |
| 05 | `05_refit_rule.py`、`05_rebuild_flowref_phys1d.slurm`、`05_swap_flowref_phys1d.py` | 分流规则重拟合，全库重建分流与一维物理特征包并逐数组比对（作业 16444，`flowref_phys1d_compare.json`） |
| 06 | `06_stats.py` | 各分区 WSS、周期量、输入特征统计重算（作业 16445，`stats_report.json`；旧统计在 `stats_v52c_backup/`） |
| 07 | `07_joint_cache.py` | 全周期缓存迁到 `joint_cycle_v52d_20261001`，重建改动单元（作业 16447） |
| 08 | `08_load_check.py` | v5.2d 配置严格加载检查（作业 16457，`load_check.json`） |
| 09 | `09_verify_joint_cache.py` | 全周期缓存 261 个单元从来源重新推导、逐数组比对（作业 16458，`joint_cache_verify.json`） |
| 10 | `10_finish_configs.py` | 三头配置的合并统计与配对参照改到 v5.2d（`configs_finish.json`）。**已被 `tools/prepare_wss_v52d_retrain.py` 取代**：正式配置由该工具重新生成，与这一步的结果逐字段相同（notes 除外）；这一步的产物留在 `superseded_configs_step10/` |

验收审计（对 v5.2d 重跑整套库审计，作业 16446）在
`outputs/cfd_auto_trial_20260927/_protocol_study/library_audit_20260930/`（`per_unit_v52d.json`、`flags_v52d.json`、`analyze_out_v52d.txt`）。
