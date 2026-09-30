# SHEN_FANG_JIN_flow.case：EnSight 打开要点

完整说明见上一级的 `README_打开说明.md`。这个文件夹可以单独拷走，`.case`、`.geo` 和 `vars/` 要放在一起。

1. 拷到纯英文路径（例如 `D:\SHEN_FANG_JIN_ensight\`），然后 File → Open → `SHEN_FANG_JIN_flow.case`。
2. 只有一个部件 `lumen (Fluent blood zone)`：该例 Fluent `blood` 区，677082 个多面体。坐标单位是 mm，+Z 指向入口。
3. 建 Clip，类型 XYZ：Z = 24.5 是瘤体最宽处的横截面，Y = 10.5 是过瘤体中心的纵剖面。拖数值滑块，截面随之移动。
4. 选中 Clip 部件，Color by 选变量：`speed_CFD` / `speed_VT0` / `speed_VTB4`（m/s），`pressure_CFD` / `pressure_PT0` / `pressure_PTB8`（Pa），
   `*_err_*` 是预测 − CFD，`velocity_*` 是矢量。
5. 时间条共 4 步：0.13 s 加速、0.21 s 峰值、0.35 s 减速、0.48 s 谷底。
6. 如果显示成点：关掉 Fast display，并检查 Preferences → Performance 里的静态快速显示。

壁面速度 = 0 是无滑移边界条件。本 case 只用于显示，正式 R² / MAE 在同点 CSV 上算。
