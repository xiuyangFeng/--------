# VF6 / best / AG/fast/ZHANG_LIANG

ParaView 打开 **VF6__best__AG__fast__ZHANG_LIANG__volume_velocity.vtp**，点击 Apply，Representation 选 Points。

CFD：`speed_cfd`；Pred：`speed_pred`；有符号误差：`speed_error_pred_minus_cfd`（m/s）。CFD/Pred 使用相同色标，误差用对称色标。

选例 R²（三种子均值）：0.916771；当前 s1234 全病例 R²：0.916368。peak 1162；v5_atlas_frame_v1；坐标 mm。

仅保留体内速度点云，不做壁面插值。含速度大小和 CFD/Pred 三分量向量；`dist_to_wall_mm` 可筛选近壁区域。点云没有体单元，不能当作连续体网格直接 Slice。

自身最大值归一化：`speed_cfd_selfmax` = CFD / 原始 CFD max；`speed_pred_selfmax` = Pred / 原始 Pred max；差值 `speed_selfmax_error_pred_minus_cfd`，绝对差 `speed_selfmax_abs_error`。均无量纲，仅作空间分布对照，幅值差已移除。

原始同点域 `interior`：CFD max = 1.33508220454 m/s，Pred max = 1.42629833082 m/s。点云/原生面/Gaussian 共享此对分母，插值后不再重求最大值；元数据嵌入 VTP FieldData 和 manifest.json。
