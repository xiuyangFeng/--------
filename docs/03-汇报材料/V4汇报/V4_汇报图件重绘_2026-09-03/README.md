# V4 汇报图件重绘（2026-09-03）

> 对应分析文档 §6「汇报答疑」Q3 / Q4 / Q9 / Q10（`../../WSS_PINN_V4实验分析与汇报提纲_2026-09-02.md`）。
> 口径更新（2026-09-04）：图中 V4 均为 pre-Centerline-V2 matched-v1.2 简化
> PN/PNPP，不是 formal 已选的 P2V/D2 `c125-k128`；后者尚无训练结果。
> 只读训练/评估产物，不改任何 run 目录；定性仍是 pre-Centerline-V2 单 seed 探索性 screen。

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `fig4_overfit_trajectory.png` | 图 4 重绘：准稳态 / 瞬态分两幅，16 臂各画 epoch 1000→2500→5000→7500→10000 的轨迹（横轴 milestone 前 100 epoch 的训练 `data_total`，纵轴该 checkpoint 的 test `E_rel_l2`），PN 实线圆 / PNPP 虚线方 | 替换旧 `fig4_overfit_scatter.png`；"训练 loss 一路降、test E 一路升"直接可见 |
| `fig7_pressure_outliers_v2.png` | 图 7 重绘：(a) TR-PN-PDE-F 35 例逐病例 pressure R² 排序条形（两例离群红色截断，标真值 gauge；含两例均值 −20 → 去掉两例 +0.36）；(b) 真值 σ 与模型 RMSE 对 gauge 均值（双 log），解释 R² = 1 − RMSE²/σ² 为何到 −552 | 替换旧 `fig7_pressure_outliers.png`；中文统一 Noto Sans CJK SC |
| `fig11_gradient_conflict.png` | 新增：cos(∇L_data, ∇L_no-slip) ≈ −0.9 的四联图——向量示意 / 日志实测曲线 / 一维剖面演示 / 末 500 epoch 分布 | §3.3 与 Q4 |
| `fig12_E_vs_speedR2.png` | 新增：16 臂 last 与 epoch 1000 在 E–speed R²_cb 平面上，说明 E 对幅值压缩不敏感 | §2.2 / §3.8 与 Q3 |
| `gradcos_V4-*-s1234.csv` | 从 `training_progress.jsonl` 抽出的每 50 step 梯度诊断（6 个 PN 臂，每臂 27600 条） | 图 11 数据；λ 时间线数字 |
| `summary_numbers.json` | 图 7 / 图 11 的关键数字（离群病例 σ/RMSE、各臂 cos 中位、裁剪比例等） | 文档引用 |
| `extract_grad_cos.py` | 第一步：抽梯度诊断（只读日志） | `python3 extract_grad_cos.py` |
| `plot_v4_report_figs_v2.py` | 第二步：生成四张图 + summary_numbers.json | `python3 plot_v4_report_figs_v2.py` |

关键数字（seed1234）：

- 梯度余弦（末 500 epoch 中位）：SP-PN-BC −0.88、SP-PN-PDE-F −0.97、SP-PN-PDE-EMA −0.97、TR-PN-BC −0.33、TR-PN-PDE-F −0.61；
  准稳态三臂 71–99% 的诊断 step 低于 −0.8；SP-PN-BC 83% 的 step 触发 grad clip。
- λ 控制器：`λ_applied` 从 1.09 起步，epoch 15 即 ≥ 9.9，之后 99.8%（SP）/ 99.5%（TR）时间贴 10；未夹紧目标 α·EMA_data/EMA_pde 在 warm-up 起点已是 130–220。
- 瞬态压力：35 例中位 +0.87、29 例 > 0；`ZHANG_YONG_SHENG-0` 真值 gauge 196 Pa、σ 551 Pa、RMSE 12943 Pa → R² −552；
  `WANG_FU_SHUN` 688 Pa、σ 992 Pa、RMSE 12561 Pa → R² −159；其余 33 例 RMSE 中位 312 Pa。

真源：`outputs/wss_pinn/volume_uvwp_bc_rcr_v4/{steady_peak,transient_autograd}/V4-*-s1234/{training_progress.jsonl, epoch_progress.jsonl, evaluation_official_last_converged_full.json}`、
`outputs/wss_pinn/audits/v4_workbook_0_14_20260823/field_metrics_v4_0_15.json`、`../V4_milestone补评_epoch曲线_2026-09-02/milestone_metrics.csv`。
