# VELWSS2 派生 WSS 代表病例后处理包

VF6 三 seed 预测速度经冻结 Profile-Secant V3（病例内校准，legacy 半径）得到的壁面 WSS。不是直接 WSS 回归。场值来自 **s1234**，选例按 seed 1234/7/2025 的逐例预测 R² 均值。

| 角色 | 病例 | 选例 R²：三 seed 均值 | 文件对应 R²：s1234 | 打开文件 |
| --- | --- | ---: | ---: | --- |
| best | AG/fast/RAN_QING_BO | 0.756170 | 0.792808 | [点云](best/VF6_wss__best__AG__fast__RAN_QING_BO__wall_wss.vtp) · [Gaussian壁面](best/surface_gaussian.vtp) · [原生壁面](best/cfd_wall_native.vtp) |
| median | ILO/LU_FU_SHAN-0/before | 0.487924 | 0.514595 | [点云](median/VF6_wss__median__ILO__LU_FU_SHAN-0__before__wall_wss.vtp) · [Gaussian壁面](median/surface_gaussian.vtp) · [原生壁面](median/cfd_wall_native.vtp) |
| worst | AAA/ruputer/KANG_YONG | −0.217338 | −0.220553 | [点云](worst/VF6_wss__worst__AAA__ruputer__KANG_YONG__wall_wss.vtp) · [Gaussian壁面](worst/surface_gaussian.vtp) · [原生壁面](worst/cfd_wall_native.vtp) |

Median 是距 34 例分布中位数（0.48785）最近的真实病例，与 R² 分布图的橙色「最常见区间代表」AG/slow/LI_HUAN_GE 不是同一例。

ParaView：`File → Open → Apply`，壁面 Representation = Surface。CFD `wss_cfd_pa`、Pred `wss_pred_pa`、误差 `wss_error_pred_minus_cfd_pa`（Pa）。selfmax 字段 `wss_cfd_selfmax` / `wss_pred_selfmax`。Gaussian：r=3 mm，sharpness=2，max_dist=3 mm，三例覆盖率 100%。peak 1162，v5_atlas_frame_v1，坐标 mm。

中位例 STL（`LU_FU_SHAN-sq.stl`）比 CFD 壁面更密（119,374 vs 79,580 顶点），不是一一对应；单位仍是 mm，最近壁面距离最大 0.97 mm。看局部热点请并排打开 `cfd_wall_native.vtp`。该例有 44 个跨局部三角图的邻域、2185 个法向相反邻域，是几何诊断，不能当成面积映射验收。

预览：[原量总览](../plots/VF6_wss_surface_triptychs_contact_sheet.png) · [selfmax 总览](../plots/VF6_wss_surface_selfmax_triptychs_contact_sheet.png)。核验 346 项通过，见 [verification.json](verification.json)。未训练、未推理。
