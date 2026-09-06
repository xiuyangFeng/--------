# V4 milestone 补评：epoch 1000 / 2500 / 5000 / 7500 / 10000 的 test35 诊断曲线

> 日期：2026-09-02
> 范围：`volume_uvwp_bc_rcr_v4` seed1234 16 臂，各自保留的 milestone checkpoint
> 定性：**test35 在此只做 epoch 诊断（development-exposed），不反选 checkpoint；单 seed；
> pre-Centerline-V2 历史矩阵。**
> 骨干：约 250k 参数的 matched-v1.2 简化 PN/PNPP，不是 2026-09-04 formal 已选的
> P2V/D2 `c125-k128`。

## 图件

| 文件 | 内容 | 关键读数 |
| --- | --- | --- |
| `fig9_milestone_E_and_speedR2.png` | 16 臂 speed R²_cb 随 milestone 变化（2 时间模式 × 2 backbone；不再画 E） | DATA 臂 epoch 1000 时 TR-DATA speed R² 0.32，是全矩阵最好点；之后单调退化到 R² ≈ 0。物理臂随 epoch 基本不退化，但 ceiling 更低；准稳态 BC/PDE 臂在 1000 时 speed R² 已是 −0.45～−0.58 |
| `fig10_milestone_nearwall_pressure.png` | 准稳态 4 臂：左两栏近壁速度 R²，右两栏**全体域**压力 R² | DATA 近壁 −1.4 → −2.4 持续恶化；BC/PDE 臂近壁「更好」只是预测趋零；EMA 压力 R² 全程 0.11–0.15 |
| `milestone_metrics.csv` | 80 行（16 臂 × 5 checkpoint）：E、speed/near-wall/core/pressure R²_cb | 机器可读；E 只保留在此表，不进 fig9 |
| `plot_v4_milestones.py` | 生成脚本（只读评估 JSON） | `python3 plot_v4_milestones.py` 原位重生成 |

## 近壁面怎么定义（fig10 左两栏）

fig10 的 `near_wall_speed` **不是**壁面点，也 **不是** V3 历史 `regional_eval.near_wall = interior & NormRadius > 0.8`。

本补评读的是 pre-Centerline-V2 sidecar 的 `interior_region`，原样来自 WSS-min preprocess 的 `int_type`（`pipeline_wss_min/preprocess.py`，默认 `pipeline_wss_min/config.py`）：

1. 评估宇宙先丢掉壁面重复行（`dist_to_wall ≤ 1e-6 mm`），只保留严格体域内部点。
2. 在**原始毫米坐标**下，对每个内部点求到最近 Fluent 物理壁面节点的欧氏距离（不用归一化坐标）。
3. `distance ≤ 1.5 mm` → `near_wall`（`int_type=1`）；否则 → `core`（`int_type=0`）。
4. 若近壁候选超过 **40,000** 点，只保留距离最近的 4 万个为 near_wall；其余即使 ≤ 1.5 mm 也标回 core。
5. 评估再在 `region == 1` 的子集上算 speed R²_cb。

fig10 **右两栏压力 R² 是全体域** `pressure`，不是近壁压力。2026-08-31 Centerline V2
staging 存储的是数值阈值 `1.5` 且不再做 4 万点封顶，但后续单位审计证明其
`distance_to_wall_mm` 使用的冻结因子不是统一物理 mm；正式 anatomy-only 重建必须按
Fluent `m×1000` 重新计算真实 1.5 mm。本图仍是 pre-Centerline-V2 旧 cap 数据。

## 真源与生成过程

- 评估：`wss_pinn/v4/evaluate.py`（2026-09-02 放开 checkpoint 白名单 `epoch_02500`、`epoch_01000`），
  启动脚本 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/eval_logs/launch_eval_milestones_20260902.sh`，
  日志 `eval_logs/gpu{1,2}_milestones_*_20260902.log`；
- 产物：各 run 目录 `evaluation_official_epoch_0{1000,2500,5000}_full.json`（7500 与 last_converged 为既有）；
- 每 100 epoch 的 `last.pt` 会被覆盖，最早可评 checkpoint 是 1000，因此"最优点"只能定性为 **≤1000**。

## 结论

1. V3→V4 的退步主要是协议（无 validation、训满 10000 epoch），不是数据。
2. 物理项的实际作用是防记忆的正则，不是提升；幅值压缩从 epoch 1000 就有，与梯度冲突机制一致。
3. 新一轮预算量级应为几百 epoch，且必须配 CV 早停（详见
   `docs/03-汇报材料/WSS_PINN_V4实验分析与汇报提纲_2026-09-02.md` §2.5 与 Gate B）。
