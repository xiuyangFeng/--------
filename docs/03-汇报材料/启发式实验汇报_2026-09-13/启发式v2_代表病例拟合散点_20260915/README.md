# 启发式 v2 代表病例拟合散点（2026-09-15）

用已交付病例的同点 CSV，画 Pred vs CFD 与 `pred = a·CFD + b`。只读后处理包，不推理、不训练。

| 图 | 打开 | 结论 |
| --- | --- | --- |
| X5X11 | [PNG](figures/X5X11_scatter_fit.png) | 最好例 a=0.950，预测/拟合 R² 0.871 / 0.877；最差例 a=0.452，拟合 R² 只到 0.541，不是截距能解释的偏差 |
| VF6 | [PNG](figures/VF6_scatter_fit.png) | 最好例 a=0.973；最差例 a=0.478，拟合 R² 0.456 几乎不抬预测 R² 0.438 |
| PF6 | [PNG](figures/PF6_scatter_fit.png) | 最好例接近恒等；中位例拟合 R² 0.934 但 a=0.693（幅值压缩）；最差例预测/拟合 0.349 / 0.387 一起掉 |
| VF6_wss | [PNG](figures/VF6_wss_scatter_fit.png) | VELWSS2 派生 WSS：最好例 a=0.790，预测/拟合 0.793 / 0.794；最差例预测 R² −0.221，拟合 R² 0.605 不能挽救 |
| X5D_v51 | [PNG](figures/X5D_v51_scatter_fit.png) | v5.1 重训完成的密度增广 X5D：最好例 a=0.926，预测/拟合 0.884 / 0.886；最差例 a=0.611，预测与拟合都停在 0.610 |
| V4_EMA | [PNG](figures/V4_EMA_scatter_fit.png) | 历史 PointNet EMA 派生 WSS：最好例 a=0.211，预测/拟合 0.260 / 0.277；最差例预测 R² −0.889，拟合 R² 0.292 不能挽救 |
| V4_EMA_speed | [PNG](figures/V4_EMA_speed_scatter_fit.png) | 同三例体内速度（按 WSS 选例，不是速度排名）：三例预测 R² 均为负，斜率 0.02–0.11 |
| V4_EMA_pressure | [PNG](figures/V4_EMA_pressure_scatter_fit.png) | 同三例体内相对压力：WSS 最差例压力 R² 反而最高（0.256），斜率仍只有 0.14 |

逐例数值：[汇总 CSV](source_data/all_scatter_fit.csv)。完整同点数组仍在原包 `_export/same_point_fields.csv.gz`。

## 口径

- 病例包：wave-2 九例、X5D_v51 三例与 V4 EMA 三例见 [代表病例后处理](../启发式v2_代表病例后处理_20260914/README.md)；V4 EMA 嵌在 [V4_EMA/](../启发式v2_代表病例后处理_20260914/V4_EMA/README.md)
- 选例：X5X11/VF6/PF6/VF6_wss 为 seed 1234/7/2025 均值；X5D_v51 为五 seed 1234/7/2025/11/2026 均值；V4 EMA 为单 seed 1234 派生 WSS。Median 为距分布中位最近的真实病例
- 图上场值与拟合：wave-2 / X5D_v51 为 s1234 / `ckpt_best` / peak 1162；V4 EMA 为 s1234 / `last_converged`（未收敛）/ 历史稳态 peak，不是 V5 1162
- X5X11 / X5D_v51：原壁面 WSS（Pa）；VF6：体内速度大小（m/s）；PF6：壁面∪体内相对压力（Pa）；VF6_wss / V4_EMA：预测速度经冻结 Profile-Secant V3 得到的派生壁面 WSS（Pa），不是直接回归；V4_EMA_speed / V4_EMA_pressure：与 WSS 相同三例的体内速度/体内相对压力
- R² 预测 = 1 − SSE/SST，与包内 `visualization_metrics.r2` 一致；R² 拟合必须连 a、b 一起读
- 全部同点绘制；颜色是 6000 点 KDE 的密度近似，不改变拟合
- 高斯插值面未用于拟合。各面板坐标范围独立，避免裁点
- test34 已暴露

重绘：`/tmp/nature-report-20260915/bin/python plot_scatter_fit.py`；只补 X5D_v51 时加 `--models X5D_v51`。
