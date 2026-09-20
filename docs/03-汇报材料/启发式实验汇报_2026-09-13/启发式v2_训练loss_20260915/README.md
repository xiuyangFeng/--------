# 启发式 v2 训练 loss history（2026-09-15）

从 wave-2 已保存的 `history.jsonl` 重绘 PF6 / VF6 / X5X11 训练曲线。只读日志，不训练、不评估。

| 图 | 打开 | 结论 |
| --- | --- | --- |
| PF6 压力 | [PNG](figures/PF6_loss_history.png) | 400 轮完成；`ckpt_best` 按在线训练 loss 最低取 epoch 375，不是 last，也不是测试 R²。重尾压力目标使淡线尖峰较多 |
| VF6 速度 | [PNG](figures/VF6_loss_history.png) | 400 轮完成；`ckpt_best` 取 epoch 374。体内速度三分量更平滑，末期波动明显小于压力 |
| X5X11 | [PNG](figures/X5X11_loss_history.png) | 与病例 R² 分布相同的三 seed 形态一致；best 在 379 / 379 / 393。曲线是 total loss（含 0.2×pinball），不是纯 MSE |

逐轮数据：[PF6 CSV](source_data/PF6_loss_history.csv)、[VF6 CSV](source_data/VF6_loss_history.csv)、[X5X11 CSV](source_data/X5X11_loss_history.csv)。

## 口径

- run：`training_wss_min/runs/wss_local_wave2_20260912/{PF6,VF6,X5X11}_s*`
- epoch 为 `history.jsonl` 的 0-based 记录（0–399）
- 选模：`selection_rule=train_loss`，`eval_every=400`，没有逐轮 val 曲线
- 淡线为全部逐轮 loss（尖峰保留）；粗线为 21 轮居中均值，只辅助读趋势
- 图注中的 test R²_cb 来自已有 `ckpt_best` 评估，未参与选模；test34 已暴露
- PF6/VF6 仅 seed 1234；X5X11 为固定 v2 的 1234 / 7 / 2025。三 seed 同向不能当作统计显著性

重绘：`/tmp/nature-report-20260915/bin/python plot_loss_history.py`（与 09-15 汇报图同一 Python）。
