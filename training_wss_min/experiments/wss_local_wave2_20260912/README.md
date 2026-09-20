# 波 2：X5+X11 三 seed、F6 变体、F6 接压力/速度体场（wss_local_wave2_20260912）

用户 2026-09-12 拍板。10 次单 run，Slurm 4 卡 × 2 槽（预检 14184 → 矩阵 14185，`COMPLETED 0:0`，4 h 12 min；14180/14182 两次预检因病例缓存键未含目标而失败并已修复）。

| 臂 | 变化 | 参照 |
| --- | --- | --- |
| X5X11_s1234 / _s7 / _s2025 | C1 + Murray 先验 + 截面 token 上下文，三个 seed（同 seed 配对参考链） | X5_s\<seed\>、X0_s\<seed\>、X11_s1234 |
| X5e233_s1234 | Murray 指数 2.33（只 log_q_branch_murray233） | X5q_s1234 / X5_s1234 |
| X5cap_s1234 | 末支半径改用几何程序虚拟盖半径（只 log_q_branch_murray_cap） | X5q_s1234 / X5_s1234 |
| X5res_s1234 | X5 输入 + 残差目标 ln(WSS) − log_tau0_murray（评估换回标准 log_z） | X5_s1234 |
| PF6_s1234 | P02（压力混合，§17 最优）+ F6；内部单元按 vol_segment_id 广播分支份额 | 历史 P02、同期 P02r |
| VF6_s1234 | V07（速度，§17 最优）+ F6 | 历史 V07、同期 V07r |
| P02r_s1234 / V07r_s1234 | 同配置同期重跑（第二阶段） | — |

未做："真实截面积"版 Murray（几何候选仍 training_allowed=false）。

- 数据：sidecar v1.1（`wss_v5/views/wall_flowref_v1.py`，原键逐位不变，新增两个 log_q 变体）；残差目标统计 `wss_global_stats_train138_offset_logtau0.json`（ln(WSS) − log τ0 的方差为 ln(WSS) 的 38%）；体场 F6 特征统计 `feature_stats/union_murray_volume_train138.json`（train138 壁面 ∪ 内部 6368 万行）。
- 代码：`data.target_log_offset_feature`（dataset/evaluate；训练在残差 z 空间，评估前 `residual_to_standard_logz` 换回标准 log_z）；`_attach_volume` 的 Murray 广播；paired init 可扩展布局补 `qad_local.0.weight`、`bottleneck.pos.0.weight`、`sa.*.contexts.*.pos.0.weight`。
- 生成/执行/汇总：`tools/prepare_wss_local_wave2.py`；`tools/preflight_wss_local_wave1.py --config-dir ...`；`tools/run_wss_local_wave1_queue.py --config-dir ...`；`tools/report_wss_local_wave1.py --config-dir ...`。

## 结果（2026-09-13；全表 [results.md](results.md)，机器可读 [results.json](results.json)）

| 臂 | Pa R²_cb / 归一化 | 配对参照 | Δ | 判读 |
| --- | --- | --- | --- | --- |
| X5X11 s1234 / s7 / s2025 | 0.7247 / 0.7262 / 0.7274（归一化 0.8557 / 0.8548 / 0.8553） | 同 seed X5 0.7178 / 0.7210 / 0.7131 | +0.007 / +0.005 / +0.014（归一化 0） | 整体在单 seed 带内；high-WSS R²、top10/p99 比三 seed 一致上升 → 尾部候选，不是主结论 |
| X5e233 | 0.6987 | X5q 0.7164 | −0.018 | 指数 3 更好 |
| X5cap | 0.6834 | X5q 0.7164 | −0.033 | atlas 半径更好 |
| X5res | 0.7151 | X5 0.7178 | −0.003 | 持平；AG 升 ILO 降 |
| PF6（压力） | 0.7843 | P02r 0.7497（历史 0.7491） | +0.035，MAE −11.6%，29/34 例变好 | F6 对压力有效 |
| VF6（速度） | 0.7952 | V07r 0.7720（历史 0.7762） | +0.023，高速区 R² 0.17→0.32，24/34 例变好；向量 RMSE −2.6% | F6 对速度有效（未过 3% 向量 RMSE 门槛） |

判读与下一步见跟踪文档 §20.4。工作簿：WSS 表 372–377 行（`tools/update_wss_local_wave1_xlsx.py --name wss_local_wave2_20260912 --group …`），速度与压力表 74–81 行（`tools/update_wss_local_wave2_volume_xlsx.py`）；验收 `xlsx_acceptance.json` / `xlsx_acceptance_volume.json`；回填前备份在本目录。
