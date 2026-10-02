# wss_v52d_retrain_20261001

v5.2d 重训：用户 09-30 确认的那套矩阵（v5.2c 上提交后被取消），10-01 在 v5.2d 上重新提交。

- 21 臂：X5Dcap_asym2 full265 × 3 seed（B）、三头 M1cap full265 × 3 seed（C）、X5Dcap_asym2 CV5 5 折 × 3 seed（A）。
- 生成：`python -m training_wss_min.tools.prepare_wss_v52d_retrain`。A / B 由 `configs/wss_v52c_retrain_20260930/` 的配置换根，
  工具校验只有数据路径、name、notes 变了；C 用 v5.2c 的同一个构造函数在 v5.2d 路径上重建。配方都没动。
- 全部 run 在 `runs/wss_v52d_retrain_20261001/`；统计、预检锚点、E5 目录、队列状态和读数在 `experiments/wss_v52d_retrain_20261001/`。
- 执行：冻结副本 `GNN_v52d_frozen_20261001`；`cluster/{preflight,run,post}_wss_v52d_retrain*.slurm`
  （预检 16459 → 队列 16460 → 收尾评估与读数 16461）。
- 读数：`python -m training_wss_min.tools.report_wss_v52d_retrain --checkpoint best`。
- 数据说明：`docs/02-推进与变更/04-数据处理与CFD/母库全量审计_2026-10-01.md` §11；实验记录：01 块跟踪 §41。
