# PF6 / best / AG/fast/YAO_CUN_HONG

ParaView 打开 **surface_gaussian.vtp**，点击 Apply，Representation 选 Surface。

CFD：`pressure_cfd_pa`；Pred：`pressure_pred_pa`；有符号误差：`pressure_error_pred_minus_cfd_pa`（Pa）。CFD/Pred 使用相同色标，误差用对称色标。

选例 R²（三种子均值）：0.965040；当前 s1234 全病例 R²：0.967960。peak 1162；v5_atlas_frame_v1；坐标 mm。

壁面 Gaussian 插值：r=3 mm，sharpness=2，max_dist=3 mm，fallback=mask。覆盖率 100.000%。面片内含 `map_valid`、`map_dist_mm`；配准和跨壁诊断见 `mapping_report.json`。

`cfd_wall_native.vtp` 保留原生 CFD 三角面上的同点场值，无高斯平滑；用于核对局部热点。原点云：`PF6__best__AG__fast__YAO_CUN_HONG__volume_pressure.vtp`。

压力点云包含壁面和体内点（`point_kind`：0 壁面，1 体内）；压力面片仅使用壁面点插值。相对压力 p−p_ref，参考值保存在 manifest.json。

自身最大值归一化：`pressure_cfd_selfmax` = CFD / 原始 CFD max；`pressure_pred_selfmax` = Pred / 原始 Pred max；差值 `pressure_selfmax_error_pred_minus_cfd`，绝对差 `pressure_selfmax_abs_error`。均无量纲，仅作空间分布对照，幅值差已移除。

原始同点域 `wall_union_interior`：CFD max = 302.78302002 Pa，Pred max = 362.791229248 Pa。点云/原生面/Gaussian 共享此对分母，插值后不再重求最大值；元数据嵌入 VTP FieldData 和 manifest.json。

压力 selfmax 按带符号相对压力除以各自 max，保留原 p_ref，不改成 maxabs/minmax；允许负值和小于 −1，不强制使用 0–1 色标。
