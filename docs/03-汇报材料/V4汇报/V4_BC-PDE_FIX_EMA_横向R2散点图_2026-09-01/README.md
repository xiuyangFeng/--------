# V4 BC+PDE-FIXED / BC+PDE-EMA 汇报图（含 full-wall WSS）

生成日期：2026-09-01。数据来自最新工作簿主表的 V4 seed1234 8 个目标臂，以及对应臂的 `evaluation_official_last_converged_full.json`。

> 口径补充（2026-09-04）：这里的 V4 是 pre-Centerline-V2 BC/RCR matched-v1.2，
> 使用约 250k 参数的简化 PN/PNPP，不是 P2V/D2。formal V4 已选择 P2V / 纯 D2
> `c125-k128`，但尚未训练，本目录不含其结果。

每张单臂图的横轴是逐病例 speed R²，纵轴是按 speed R² 从 worst 到 best 排序的病例序号；蓝线连接排序后的散点，红/橙/绿线分别标出 worst / most frequent / best。

8 个实验各自使用一个独立子文件夹；根目录只保留汇总图、PDF、CSV 和本说明。

另外输出的 `*_speed_linear_fit_r2_horizontal.png` 与此前 `v2_v3_linear_regression_fullpoints_20260807` 图的 `Linear-fit R²` 口径一致。V4 JSON 没有保存点级协方差，因此该值由 truth/prediction 的总体均值、方差和官方 raw R² 精确重构；它不替代工作簿主表的 raw speed R²。

每个实验子文件夹还包含 `speed_representative_regressions.png` 和 `wss_representative_regressions.png` 三联图；其原始三张代表病例图保存在该子文件夹的 `representative_cases/` 下。

WSS 图和 WSS 指标已经改为 **full-wall** 版本：每个病例使用全部冻结壁面节点，而不是旧版每例 1200 点。模型输入仍使用该实验配置的 support 点；WSS 计算则先预测峰值完整体域速度场，再在每个病例的全部冻结壁面节点上运行 Profile-Secant V3。因此 support 输入点与 WSS 评估点不是同一批点。

各实验文件夹中的 full-wall WSS 输出包括：

- `wss_representative_regressions.png`：worst / most frequent / best 三联图；
- `wss_case_r2_ranking.png`：35 个病例的 WSS linear-fit R² 排名；
- `wss_r2_distribution.png`：35 个病例的 WSS linear-fit R² 分布；
- `representative_cases/*/wss_regression.png`：三例 full-wall 点密度回归图。

根目录的 `V4_fullwall_wss_per_case.csv` 保存 8 个实验 × 35 例的 full-wall WSS 指标，`V4_fullwall_wss_manifest.json` 保存生成协议、壁面点数范围和代表病例。旧的 1200 点 WSS 图片已删除或由同名 full-wall 图片覆盖；speed 图片不受影响。

`V4_BC-PDE_FIXED_EMA_speed_r2_horizontal_report.pdf` 和 `V4_BC-PDE_FIXED_EMA_speed_linear_fit_r2_horizontal_report.pdf` 分别合并了两套 8 张单臂图，便于直接汇报或打印。

## 8 个目标臂

| 编号 | 实验ID | 阶段 | 架构 | 训练模式 |
|---:|---|---|---|---|
| 2 | `V4-SP-PN-BC-PDE-F-s1234` | volume_uvwp_bc_rcr_v4 / steady peak | PointNet | BC+PDE-FIXED |
| 3 | `V4-SP-PN-BC-PDE-EMA-s1234` | volume_uvwp_bc_rcr_v4 / steady peak | PointNet | BC+PDE-EMA |
| 6 | `V4-SP-PNPP-BC-PDE-F-s1234` | volume_uvwp_bc_rcr_v4 / steady peak | PointNet++ | BC+PDE-FIXED |
| 7 | `V4-SP-PNPP-BC-PDE-EMA-s1234` | volume_uvwp_bc_rcr_v4 / steady peak | PointNet++ | BC+PDE-EMA |
| 10 | `V4-TR-PN-BC-PDE-F-s1234` | volume_uvwp_bc_rcr_v4 / transient 81 | PointNet | BC+PDE-FIXED |
| 11 | `V4-TR-PN-BC-PDE-EMA-s1234` | volume_uvwp_bc_rcr_v4 / transient 81 | PointNet | BC+PDE-EMA |
| 14 | `V4-TR-PNPP-BC-PDE-F-s1234` | volume_uvwp_bc_rcr_v4 / transient 81 | PointNet++ | BC+PDE-FIXED |
| 15 | `V4-TR-PNPP-BC-PDE-EMA-s1234` | volume_uvwp_bc_rcr_v4 / transient 81 | PointNet++ | BC+PDE-EMA |

横向排序图生成脚本：`docs/03-汇报材料/tools/plot_v4_bc_pde_r2_horizontal.py`。
代表病例回归三联图生成脚本：`docs/03-汇报材料/tools/plot_v4_representative_regressions.py`（需使用 GNN conda 环境）。
full-wall WSS 图生成脚本：`docs/03-汇报材料/tools/plot_v4_fullwall_wss_regressions.py`。
full-wall WSS 工作簿回写脚本：`docs/03-汇报材料/tools/update_v4_fullwall_wss_workbook.py`。
