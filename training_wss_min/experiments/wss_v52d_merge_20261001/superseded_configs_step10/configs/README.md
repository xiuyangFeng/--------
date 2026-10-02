# wss_v52d_retrain_20261001

21 个配置，全部指向数据版本 v5.2d（`data_wss_v5/views_v5_2d_20261001`）。**未训练，也没有准备队列 / 预检脚本。**

- 来源：`configs/wss_v52c_retrain_20260930/` 的 21 个配置经 `training_wss_min.tools.repoint_data_root` 换根，配方不变；
  run 名前缀改为 `wss_v52d_retrain_20261001/`，不会写进 v5.2c 的半截 run 目录。
- 18 个 X5Dcap_asym2（CV5 5 折 × 3 seed + full265 × 3 seed），3 个三头 M1cap full265。
- 三头配置的合并统计重建为 `experiments/wss_v52d_retrain_20261001/stats/multi_stats_full265_train265.json`（v5.2d 统计），
  `init_reference_config` 指向本目录的同 seed X5Dcap_asym2 full265 配置。
- v5.2c 那次的 `matrix.json`（队列、预检锚点、附加评估）没有带过来。预检锚点「旧模型在 recover8 上复现存档值」
  按理仍成立（recover8 标签没变；输入用纯几何的 `*_murray_cap`，不受分流规则重拟合影响），没有重跑确认。
- 记录：`experiments/wss_v52d_merge_20261001/`（`load_check.json`、`configs_finish.json`），
  文档 `docs/02-推进与变更/04-数据处理与CFD/母库全量审计_2026-10-01.md` §11。
