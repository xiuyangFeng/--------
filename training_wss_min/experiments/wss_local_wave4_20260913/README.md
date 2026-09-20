# 波 4：切口盖面拟合分流替换 Murray + seed 集成扩充（wss_local_wave4_20260913）

用户 2026-09-13 拍板"按你的想法逐步确认后续路线"，且只允许部署可得输入。8 次单 run，Slurm 4 卡 × 2 槽（预检 14200 → 矩阵 14201，`COMPLETED 0:0`，3 h 35 min；提交记录 `submission.json`）。

| 臂 | 变化 | 配对参照 |
| --- | --- | --- |
| X5A_s1234 / s7 / s2025 / s11 / s2026 | C1 + [log_q_branch_capfit, log_tau0_capfit]：末支分流用 train138 拟合的规则 log(Q外/Q内) = 1.072·log((r盖,外/r盖,内)²) + 0.164（盖面半径来自点云几何程序），根 1、髂总 0.5 保持协议 | 同 seed 的 X5（同 C1_s<seed> 参考链，只有两列零初始化输入不同） |
| X5_s11 / X5_s2026 | C1 + Murray 先验，再加两个 seed（与波 1/1b 的三个 seed 组成 5 seed 集成） | — |
| X5B_s1234 | C1 + Murray + capfit 四列一起 | X5_s1234 / X5A_s1234 |

- 分流规则：`tools/fit_flow_split_rule.py` → `data_wss_v5/views/wss_min_flowref_v1/flow_split_rule_train138.json`（train 272 侧拟合，test 67 侧 R² 0.81，Murray R³ 同口径 0.62；4 例因出口峰值通量非正未参与拟合，sidecar 侧仍按规则计算）。
- sidecar v1.2：`wss_v5/views/wall_flowref_v1.py` 新增 `_capfit_shares`，172 例重建 15 s，旧 19 个键逐位不变、prior 不变（`tests/test_flowref_capfit.py`）。`config.V7_FLOW_FEATURE_KEYS` / `VOLUME_EXTENDABLE_SIDECAR_KEYS` 登记新键；`dataset._attach_volume` 的 log_tau0 广播按键名泛化（体场以后也能用）。
- 生成：`tools/prepare_wss_local_wave4.py`（seed 11/2026 的 C1→M2 参考链在 `refs/`；capfit 键的 train138 统计 `feature_stats/union_capfit_wall_train138.json`）；本地验证四组配对的继承张量逐位相同、追加列为零。
- 判读：X5A vs 同 seed X5 五组配对 Δ；离线用保存的 test 预测算 X5 与 X5A 各自的 5 seed 集成；X5B 单 seed 看"两族份额同时给"有没有额外价值。结果回填跟踪文档 §20.7。

## 结果（2026-09-13；全表 [results.md](results.md)，离线分析 `ensemble_analysis.py`）

| 比较 | 结果 | 决定 |
| --- | --- | --- |
| X5A vs 同 seed X5（5 对） | 物理 −0.015/−0.006/−0.019/−0.010/+0.007（均值 −0.009，带内）；归一化 +0.004～+0.009（5/5 正）；AG +0.007、AAA +0.017、ILO −0.042（射流病例 SUN_XU_XIA-1 主导） | capfit 未取代 Murray；X5 仍为底座 |
| X5 五 seed 集成 | 0.7354 / 归一化 0.8712（单 seed 均值 0.7172，三 seed 组合均值 0.7325） | 部署形态 = 五 seed 集成 |
| X5A 五 seed 集成 / 十模型混合 | 0.7255 / 0.7337 | 不采用 |
| X5B（Murray + capfit） | 0.7095 / 0.8603，带内 | 无额外价值 |

判读与下一步见跟踪文档 §20.7；工作簿 WSS 表 378–385 行（`tools/update_wss_local_wave1_xlsx.py --name wss_local_wave4_20260913 --group …`）。
