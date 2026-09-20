# V4 EMA 代表病例后处理（历史 PointNet BC+PDE-EMA）

臂：`V4-SP-PN-BC-PDE-EMA-s1234`（矩阵 index 3，seed1234，`last_converged` / epoch9999 / `converged:false`）。
不是正在训练的 X5D_long，也不是 formal P2V/D2 V4。坐标为原始 CFD 毫米，不是 V5 atlas。本包未写 native CFD 三角面。

## 选例

按 test35 全壁面派生 WSS 物理预测 R²；Median 是距分布中位最近的真实病例。
速度/压力是**同一三例**的体场，不是按速度或压力独立选的最好/最差。
WSS 来自 2026-09-01 full-wall 缓存 + 冻结 Profile-Secant V3；速度/压力在 master GPU2 剩余显存上按历史 BC 变换重推理，正式 R² 与 `evaluation_official_last_converged_full.json` 对齐。

| 角色 | 病例 | WSS R² | 速度 R² | 压力 R² |
| --- | --- | ---: | ---: | ---: |
| best | `AG/fast/RAN_QING_BO` | 0.2599 | −0.1695 | 0.1446 |
| median | `ILO/LI_YOU_ZHI-0/before` | −0.0403 | −0.6143 | 0.0108 |
| worst | `AG/fast/LOU_YANG` | −0.8893 | −0.0500 | 0.2556 |

分布：test35 派生 WSS 均值 / 中位 / 负 R² = −0.1051 / −0.0403 / 20/35。

## 打开文件

| 变量 | 打开 | 字段 |
| --- | --- | --- |
| WSS 壁面 | 各例 `surface_gaussian.vtp` | `wss_cfd_pa` / `wss_pred_pa` |
| 速度体内点云 | `*__interior_velocity.vtp` | `speed_cfd` / `speed_pred`；向量 `velocity_*_vector` |
| 压力点云 | `*__volume_pressure.vtp`（`point_kind` 0 壁面 / 1 体内） | `pressure_cfd_pa` / `pressure_pred_pa` |
| 压力壁面 | `pressure_surface_gaussian.vtp` | 同上；**不覆盖** WSS 的 `surface_gaussian.vtp` |

正式速度/压力 R² 是严格体内同点，不是插值面。压力 selfmax 分母取 wall∪interior。Gaussian：r=3 mm，sharpness=2，max_dist=3 mm，三例覆盖率 100%。

清单：[打开文件清单.csv](打开文件清单.csv) · [归一化分母清单.csv](归一化分母清单.csv)

## 预览

原量 / selfmax 三联图在上层 [`../plots/`](../plots/)：

- WSS：`V4_EMA_*_surface_triptych.png`、`V4_EMA_surface*_triptychs_contact_sheet.png`
- 速度：`V4_EMA_*_volume_pointcloud*.png`
- 压力：`V4_EMA_pressure_*_surface*.png`

拟合散点：[V4_EMA](../../启发式v2_代表病例拟合散点_20260915/figures/V4_EMA_scatter_fit.png) · [速度](../../启发式v2_代表病例拟合散点_20260915/figures/V4_EMA_speed_scatter_fit.png) · [压力](../../启发式v2_代表病例拟合散点_20260915/figures/V4_EMA_pressure_scatter_fit.png)
