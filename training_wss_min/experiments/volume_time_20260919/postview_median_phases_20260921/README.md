# 体场中位病例四相位后处理包（与 WSS 同一例）

- 病例：`AAA/unruputer/SHEN_FANG_JIN`（cv3 fold 2 留出）。沿用 WSS `postview_median_phases_20260920` 的选例，**没有按体场 R² 重新选例**。
- 预测臂：压力 `PT0` / `PTB8`，速度 `VT0` / `VTB4`，均为 fold2 `ckpt_best.pt`。
- 压力是相对压力 p − p_体均(t)（Pa）；速度是 m/s；坐标是 V5 配准后的毫米。
- 正式 R² / MAE 只在同点 CSV / 点云上算。

## 要在 EnSight 里切截面：打开这个

**`AAA__unruputer__SHEN_FANG_JIN/ensight/SHEN_FANG_JIN_flow.case`**

点云已经插到该例自己的 Fluent 体网格（`blood` 区 677082 个单元）的节点上，写成 EnSight Gold 瞬态 case：
拖时间条换相位，拖 Clip 位置换截面，切换变量换指标（CFD / 预测 / 误差 / 压力）。
操作步骤、变量表、建议色标和核对结果见 [病例打开说明](AAA__unruputer__SHEN_FANG_JIN/README_打开说明.md)。

## 相位与同点 R²

| 相位 | 帧 | 协议步 | 时间（EnSight） | Q/Qpeak | 定义 | 压力 PT0 | 压力 PTB8 | 速度 VT0 | 速度 VTB4 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 加速 accel | 13 | 1146 | 0.13 s | 0.406 | Q 上升、尚未到 0.8 Qpeak | 0.954 | 0.985 | 0.798 | 0.765 |
| 峰值 peak | 21 | 1162 | 0.21 s | 1.000 | 峰值帧 | 0.654 | 0.602 | 0.828 | 0.770 |
| 减速 decel | 35 | 1190 | 0.35 s | 0.431 | 峰值窗之后的减速射血 | 0.718 | 0.859 | 0.578 | 0.556 |
| 谷底 trough | 48 | 1216 | 0.48 s | 0.003 | 协议波形最低 Q | −0.143 | 0.533 | −0.154 | 0.041 |

R² 是物理单位、同点（压力：壁面 ∪ 体内 762386 点；速度：体内 677082 点）的值，
取自 `manifest.json` 各包的 `pointcloud_metrics`，不是在截面上算的。

## 目录

| 路径 | 内容 |
| --- | --- |
| `AAA__unruputer__SHEN_FANG_JIN/ensight/` | **EnSight / ParaView 切截面用**（插值到 CFD 网格，4 个相位） |
| `AAA__unruputer__SHEN_FANG_JIN/README_打开说明.md` | 打开步骤与核对 |
| `AAA__unruputer__SHEN_FANG_JIN/velocity/<臂>/<相位>/` | 速度原始点云 `interior_pointcloud.vtp`、同点 CSV `_export/`、预览图 |
| `AAA__unruputer__SHEN_FANG_JIN/pressure/<臂>/<相位>/` | 压力原始点云（`wall_` / `interior_` / `full_pointcloud.vtp`）、壁面面片（`surface_gaussian.vtp`、`cfd_wall_native.vtp`）、同点 CSV |
| `AAA__unruputer__SHEN_FANG_JIN/aligned_geometry.stl` | 同坐标系 STL |
| `打开文件清单.csv` | 上述入口的清单 |
| `manifest.json` / `verification.json` / `case_selection.json` | 原导出元数据（2026-09-21，未改动） |

## 生成脚本

- 点云与壁面面片：`training_wss_min/experiments/volume_time_20260919/export_phase_postview.py`
- EnSight 插值 case：`training_wss_min/tools/build_ensight_cfd_mesh_case.py --case-dir <病例目录>`
