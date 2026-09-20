# VF6_wss / best / AG/fast/RAN_QING_BO

ParaView 打开 **surface_gaussian.vtp**，点击 Apply，Representation 选 Surface。

CFD：`wss_cfd_pa`；Pred：`wss_pred_pa`；有符号误差：`wss_error_pred_minus_cfd_pa`（Pa）。CFD/Pred 使用相同色标，误差用对称色标。

选例 R²（三种子均值）：0.756170；当前 s1234 全壁面派生 WSS R²：0.792808。peak 1162；v5_atlas_frame_v1；坐标 mm。

壁面 Gaussian 插值：r=3 mm，sharpness=2，max_dist=3 mm，fallback=mask。覆盖率 100.000%。面片内含 `map_valid`、`map_dist_mm`；配准和跨壁诊断见 `mapping_report.json`。

`cfd_wall_native.vtp` 保留原生 CFD 三角面上的同点场值，无高斯平滑；用于核对局部热点。原点云：`VF6_wss__best__AG__fast__RAN_QING_BO__wall_wss.vtp`。

这是 VF6 预测速度经冻结 Profile-Secant V3（校准）得到的派生 WSS，不是直接 WSS 回归。

自身最大值归一化：`wss_cfd_selfmax` = CFD / 原始 CFD max；`wss_pred_selfmax` = Pred / 原始 Pred max；差值 `wss_selfmax_error_pred_minus_cfd`，绝对差 `wss_selfmax_abs_error`。均无量纲，仅作空间分布对照，幅值差已移除。

原始同点域 `wall`：CFD max = 51.2995376587 Pa，Pred max = 43.108934093 Pa。点云/原生面/Gaussian 共享此对分母，插值后不再重求最大值；元数据嵌入 VTP FieldData 和 manifest.json。
