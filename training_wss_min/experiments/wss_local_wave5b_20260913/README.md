# 波 5b：T3 热点级联 + cv3 seed 7（wss_local_wave5b_20260913）

用户 2026-09-13 拍板的第二批收尾。12 次单 run；预检 Slurm 14213（通过，含 C1 锚点重评）；矩阵先以 14214（4 卡）提交，因另一用户占了一张卡改为 14217（3 卡 × 4 槽，`COMPLETED`，12 训练 + 24 评估全部退出码 0，2026-09-14 00:45–06:48；`submission.json`）。

| 臂 | 变化 | 配对参照 |
| --- | --- | --- |
| T3_s1234 / s7 / s2025 | X5 输入 + 一阶段预测 `log_wss_base`（`wss_min_cascade_v1`：train138 用 cv3 折外 X5 预测、test34 用折模型轮转），目标改为残差 ln(WSS) − log_wss_base，损失聚焦一阶段预测的逐例 top 20%（区外 0.2），评估门控只在区内叠加残差 | 同 seed 的 X5（同 C1_s<seed> 参考链，只多一列零初始化输入） |
| T3n_s1234 / s7 / s2025 | 同上输入与残差目标，不聚焦、不门控（残差堆叠对照） | 同 seed 的 X5；T3 vs T3n 分离聚焦/门控的效应 |
| X5_f0/f1/f2_s7、X5X11_f0/f1/f2_s7 | cv3 三折在 seed 7（与波 5 的 seed 1234 折臂配对读） | 同折 X5_f<k>_s7 |

- 代码（配置驱动、默认关、旧配置逐位不变）：`train.loss_region_feature / loss_region_quantile / loss_region_outside_weight`（`objectives.region_focus_weights`，trainer 把标准化输入列放进 `batch["loss_region_value"]`）、`eval.residual_gate_quantile`（`dataset.gate_residual_prediction`）、`config.V8_CASCADE_FEATURE_KEYS`；`tests/test_wss_local_wave5.py`（6 项）。
- 一阶段 sidecar：`tools/make_cascade_sidecar.py` → `data_wss_v5/views/wss_min_cascade_v1/`（残差统计 `wss_global_stats_train138_offset_logwssbase.json`：offset 均值 −0.065、std 0.529；一阶段 ln R² 折外 0.805 / test34 0.810，两边分布匹配）。
- 生成：`tools/prepare_wss_local_wave5b.py`；结果 `results.md`、`../wss_local_wave5_20260913/offline/analyze_wave5.json`；工作簿 WSS 表 398–409 行。

## 结果（test34，best；判读见跟踪文档 §21.5、§21.7）

| 比较 | seed 1234 / 7 / 2025 | 均值 | 决定 |
| --- | --- | --- | --- |
| T3n − X5（物理） | −0.003 / +0.019 / +0.027 | +0.014（sd 0.016，符号不一致） | 尾部候选；归一化 3/3 负（−0.004），逐例 14/20、13/21、16/18 |
| T3 − X5（物理） | +0.004 / −0.000 / +0.014 | +0.006（带内） | 归一化 3/3 负（−0.008）；聚焦+门控不如不聚焦（T3 − T3n −0.009） |
| cv3 seed 7：X5X11 − X5（折外） | +0.004 / +0.017 / +0.005（三折） | 合并 +0.010 / −0.001，62 胜 76 负 | 与 seed 1234 合读：不进主线 |
