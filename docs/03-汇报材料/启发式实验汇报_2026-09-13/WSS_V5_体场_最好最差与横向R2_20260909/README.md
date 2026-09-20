# V5 R5P 混合压力 / R5V 内部速度：最好、最差病例后处理

2026-09-09；峰值步 1162；原 best checkpoint；test34；固定 5000 壁面 support / seed1234；全量 query。

两组均按原 metrics.json 的逐病例物理预测 R² 选首尾。压力是壁面∪内部的相对压力 p−p_ref（Pa）；速度是预测三分量的幅值 |u|（m/s）。

| 目标 | 选例 | 病例 | 预测 R² | 拟合 R² | 斜率 a | 截距 b | MAE |
|---|---|---|---:|---:|---:|---:|---:|
| pressure | best | AAA/ruputer/LI_ZHEN_HUA | 0.956971 | 0.958695 | 0.9436 | -0.7530 | 65.6636 |
| pressure | worst | AAA/ruputer/KANG_YONG | -1.674758 | 0.764155 | 2.0245 | 11.7004 | 75.2500 |
| speed | best | AAA/ruputer/LI_ZHEN_HUA | 0.890054 | 0.897595 | 0.8459 | 0.0393 | 0.0931 |
| speed | worst | AAA/ruputer/KANG_YONG | 0.353737 | 0.832836 | 1.3323 | 0.0032 | 0.0802 |

## 直接查看

- `index.html`：全部图件导航。`field_best_worst_overview.png`：压力壁面/速度内部场总览。
- `scatter_fit_best_worst.png`：四张全点密度散点拟合；单例另有独立图，压力还分壁面/内部画图。
- `horizontal_r2_overview.png`：两组 × 预测R²/拟合R²；每张也单独导出 PNG。
- `per_case_r2_and_linear_fit.csv`：完整68行结果；拟合 pred = a·CFD+b，以全部同点数据计算。

## ParaView 包

完整文件目录：`/public/newhome/cy/Digital_twin/GNN/outputs/field/postview/wss_v5_r5_volume_best_worst_20260909`。

1. 压力打开每例 `cfd_wall_mesh.vtp` → Apply → Surface：真实 CFD 壁面三角网格直接挂同点压力，无插值。
2. 内部场打开 `interior_pointcloud.vtp` → Apply → Points（Point Size 2–4）。完整源场为 `full_pointcloud.vtp`。
3. 看腔内打开 `interior_3mm_slab.vtp`，或对内部点云用两个 Clip/Box 保留薄层。原始点云没有体单元拓扑，不能把普通 Slice 当连续 CFD 截面。
4. 真值/预测字段：`pressure_cfd / pressure_pred`（Pa，相对压），`speed_cfd / speed_pred`（m/s）；误差为 `err_*`（Pred−CFD）及 `abs_err_*`。
5. 速度方向可用 Glyph，Vectors 选 `velocity_pred_aligned_m_s` 或 `velocity_cfd_aligned_m_s`；`wall_geometry_only.vtp` 可叠加透明外壳。
6. CFD 与 Pred 色标用同一区间；可导入根目录 `GNN_blue_white_red.xml`。全部坐标与速度向量为同一 atlas 解剖系，长度 mm、速度 m/s。
7. `point_kind_0_wall_1_interior` 标明点类型；速度文件只有内部点。`source_index`：压力为壁面后接内部的query行号，速度为volume内部行号。

## 图与指标口径

- 预测 R² = 1−SSE/SST；拟合 R² = Pearson r²，必须结合 a、b 判断。选例依预测 R²；拟合横向图的红绿线依拟合 R² 自己排序，病例可能不同。橙线取等宽5箱中最高频箱中心最近病例，非概率密度估计。
- 压力标签沿用训练视图 v1.1：壁面压力来自最近内部单元；p_ref 为该例峰值步体积平均压力，不是壁面原导出压力，也不是绝对压力。未对测试预测另做校准或重新居中。
- 压力KANG_YONG真值标准差约78 Pa；尽管相关性较高，幅值偏差使预测R²为负，MAE约75 Pa。负R²没有裁掉。
- 静态云图CFD与Pred使用共享范围，误差用对称范围。内部全云是沿Y投影，可能遮挡；另给Y方向3mm薄层及Z方向三个2mm横截薄层。薄层只含原始单元中心，无插值。
- 速度总览使用 `fig_interior_depth_mean.png`：在0.75mm的XZ网格内，沿Y对原内部单元的值作点数等权平均，空格透明；帮助观察被近壁低速点遮挡的内部场。这是显示用深度平均，不是CFD截面或体积加权平均，不用于拟合和正式指标。原三维点场不变。
- PNG内部点最多显示85,000点；散点密度统计、拟合、VTP与NPZ均使用全部同点数据。壁面静态图三角面颜色取三顶点均值，正式指标仍按原节点。
- 速度另存CFDmax共同分母和Predmax自分母字段；双自最大值仅供看形状。压力为有符号相对压，不做自最大值归一化。
- 每例 `same_point_fields.npz` 保存全量同点值、点索引和坐标；manifest记录checkpoint、输入及数据SHA256。

## 验证与复现

- 两组34例共476项原始逐病例基本指标复现通过，并检查全部68个原斜率；模型和原评估文件未改动。速度复用已验证全点缓存，压力按原协议重推理。
- 全部点云VTP重新读回后坐标、标量、向量逐值一致；压力网格点数、三角数及标量一致。
- `python -m training_wss_min.tools.export_v5_volume_postview --device cuda:0`
- `python -m training_wss_min.tools.plot_v5_volume_postview`
- 运行环境：`/public/newhome/cy/.conda/envs/GNN/bin/python`。单seed、已暴露test34，仅为既有模型结果展示。

<!-- R5_SCATTER_STYLE -->
## 参考R4样式的散点图（2026-09-09更新）

- `pressure_scatter_fit_best_worst.png`、`speed_scatter_fit_best_worst.png`：各自上排最好、下排最差，左物理、右linear z。
- 复用 `plot_postview_case_scatter.py`：viridis密度着色的细散点、红色拟合直线、灰色y=x虚线、左上指标框和右下方程。每个原始query点均实际绘制。
- 大点云的颜色采用固定seed抽取6000点拟合Gaussian KDE，再在256×256格上求值并插值回各点，仅近似颜色。拟合、R²和点数均用全量原值。
- 压力和速度幅值的归一化均为训练集统计的线性z，不使用WSS的log_z。因此左右R²与斜率相同，单位和截距改变；已核对这一关系及原始物理指标。
- 独立重绘命令：`python -m training_wss_min.tools.plot_v5_volume_scatter`，无需重新推理或导出VTP。
