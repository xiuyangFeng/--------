# V4 seed1234 index 0–15 训练曲线与诊断图

> 日期：2026-08-31
> 范围：`volume_uvwp_bc_rcr_v4` seed1234 全部 16 臂（array index 0–15），已完训 +
> official test35 + 同协议 WSS 下游
> 定性边界：**单 seed 探索性 screen**，非 3-seed 正式入账，不能 Holm；且按
> 2026-08-30 数据审阅结论，本矩阵属 `pre-Centerline-V2 / old-registration-frame
> historical matrix`，只作历史 screen，不得当作新中心线数据的性能结论。

## 用途

给"seed1234 十六臂为什么是这个结果"的分析与老师汇报提供图件。核心结论：

1. 16 臂 `E_rel_l2` 全部落在 0.89–1.18（1.0 = 全零预测器），无一臂实质性重建速度场；
2. 测试失败主因是 train-only 停止协议造成的系统性过拟合（图1/4/5）；
3. 物理臂在 E 上的"优势"主要来自幅值压缩/趋零收缩，不是学到结构（图6）；
4. EMA 动态权重开局即贴死 clamp 上限 10、全程未调节，等价于 λ_pde≡10 固定臂，
   并打开平凡解通道（图2/3）；
5. 瞬态压力 R̄²=−20 是 2 个 gauge 口径异常病例（真值约 0.2/0.7 kPa，全库主流
   13–16 kPa）拉爆的均值假象，中位数 +0.87（图7）；
6. WSS 全负完全跟随近壁速度场质量，Profile-Secant V3 本身（真速度上限 ≈0.96）
   不背锅（图8）。

## 图件清单

| 文件 | 内容 | 关键读数 |
| --- | --- | --- |
| `fig1_train_data_total.png` | 16 臂 raw `data_total` 训练曲线（2 时间模式 × 2 backbone） | DATA 0.19 → +BC 0.49 → +PDE-F 0.73 → +EMA 0.83 阶梯；DATA 至 10000 epoch 仍缓降，无真平台 |
| `fig2_train_channels.png` | u / p 分通道训练曲线（PN；PNPP 同臂差约 0.002） | 瞬态 EMA 的 u 通道全程钉在 ≈0.99（完全未拟合）；准稳态 PDE 臂 p 通道 0.5/0.8 对 DATA/BC 的 0.016 |
| `fig3_physics_dynamics.png` | λ_pde 轨迹、continuity/momentum 残差、瞬态 RCR 残差 | λ_pde 开局 ~100 epoch 内贴死 10；瞬态 EMA continuity 在 ~3200–5600 epoch 跌至 1e-12（平凡解区段）；瞬态 PDE-F momentum 在 ~1000–5500 epoch 尖峰到 1e3–1e4；RCR 残差 DATA+BC(3e-4) < PDE-F(1e-3) << EMA(8e-2) |
| `fig4_overfit_scatter.png` | 训练末段 data_total vs official `E_rel_l2` 散点 | 训练拟合与测试主指标完全脱钩：DATA 训练最好测试最差，EMA 几乎不拟合却 E 最低 |
| `fig5_primary_bars.png` | 16 臂 `E_rel_l2` 全景条形 + epoch_07500 黑菱形 | 准稳态 8 臂 7500 均优于 last（+0.01–0.07）→ 泛化仍在随训练变差；瞬态 PDE 臂基本无此退化 |
| `fig6_variance_ratio.png` | 逐病例 speed 方差比 pred/truth 箱线图（log） | DATA→BC→F→EMA 逐级压向零；准稳态中位 0.39/0.16/0.07/0.04；瞬态 PN·EMA 中位 ≈3e-4（纯零场） |
| `fig7_pressure_outliers.png` | 瞬态逐病例 pressure R² vs 真值 gauge 均值（TR-PN-PDE-F） | 中位 +0.87、29/35 例 >0；ZHANG_YONG_SHENG-0/before（196 Pa，R²=−552）与 WANG_FU_SHUN（688 Pa，R²=−159）拉爆均值 |
| `fig8_wss_nearwall.png` | 16 臂 WSS R²_cb 条形 + 准稳态 near-wall R² 耦合散点 | WSS 排序与近壁速度 R² 同构；最好 TR-PNPP-PDE-F ≈ 0.00（中位 +0.09） |
| `V4_seed1234_0-15_训练曲线与诊断图_2026-08-31.pdf` | 上述 8 图合订 | 汇报用 |
| `plot_v4_curves.py` | 生成脚本（可复现） | `python3 plot_v4_curves.py` 在本目录原位重生成全部图件 |

## 数据真源（全部只读，未改任何训练/评估产物）

- 训练曲线：`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/{steady_peak,transient_autograd}/V4-*-s1234/epoch_progress.jsonl`
  （raw 分量 loss、`mean_lambda_phy`；平滑为 51-epoch 滑动均值，淡色为原始逐 epoch 值）；
- 训练末段汇总：`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/midterm_train_only_20260821.json`（缺的 11/14/15 由各自 `epoch_progress.jsonl` 末 200 epoch 均值补齐）；
- official test35 场指标（last_converged 与 epoch_07500）：
  `outputs/wss_pinn/audits/v4_workbook_0_14_20260823/field_metrics_v4_0_15.json`；
- 逐病例明细（方差比、压力 R²、gauge 均值）：各 run 的
  `evaluation_official_last_converged_full.json`；
- WSS 下游（Profile-Secant V3 × 1200 壁面点）：
  `outputs/wss_pinn/audits/v4_workbook_0_14_20260823/arms/*/wss_metrics.json`。

## 口径与限制

- `E_rel_l2` 为预注册 primary endpoint（病例等权速度向量相对 L2；瞬态 81 帧 pooled）；
  `E=1.0` 恰为全零预测器；瞬态与准稳态的评估宇宙不同，TR<SP 不得读作物理增益；
- 图5 的 epoch_07500 点属 **checkpoint 敏感性诊断**，不用于也不得用于反选 checkpoint；
- 图7 只展示 TR-PN-PDE-F 一臂；同一 2 例异常病例在全部 8 个瞬态臂中同样拉爆均值
  （各臂 R̄² −20 至 −24，中位 +0.81 至 +0.87 量级）；
- 本目录不回填工作簿数值；xlsx 主表行 19–34 仍以 2026-08-23/28 入账为准。
