# V4_EMA / worst / AG/fast/LOU_YANG

病例按 **派生 WSS R²** 选取（不是速度/压力独立排名）。checkpoint 名为 last_converged，实际 epoch9999、converged:false。坐标为原始 CFD 毫米，不是 V5 atlas。

## WSS

ParaView 打开 **surface_gaussian.vtp**，Representation 选 Surface。
CFD `wss_cfd_pa`；Pred `wss_pred_pa`。选例/出图 R²：-0.889265。

## 速度（体内点云）

打开 **V4_EMA__worst__AG__fast__LOU_YANG__interior_velocity.vtp**。CFD `speed_cfd`；Pred `speed_pred`；向量 `velocity_*_vector`。
正式体内速度 R²：-0.050014（n=746864）。不做速度壁面插值。

## 压力（壁面∪体内）

点云 **V4_EMA__worst__AG__fast__LOU_YANG__volume_pressure.vtp**（`point_kind` 0 壁面 / 1 体内）。壁面 Gaussian：**pressure_surface_gaussian.vtp**（不覆盖 WSS 的 surface_gaussian.vtp）。
正式体内相对压力 R²：0.255620（n=746864，不含壁面分母）。 p_ref = 16169.731139 Pa。压力 selfmax 分母取 wall∪interior。
壁面 Gaussian 覆盖率 100.000%，r=3 mm / sharpness=2 / max_dist=3 mm。
