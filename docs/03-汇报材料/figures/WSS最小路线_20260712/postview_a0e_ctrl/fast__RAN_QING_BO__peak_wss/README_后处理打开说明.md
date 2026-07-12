# 后处理软件打开说明（WSS-min · ParaView）

## 病例 / 模型

- 病例：`fast/RAN_QING_BO`（split 分区：**train**）
- 模型：`r5_a0e_b1_ctrl_s1234` · `ckpt_best.pt`
- 帧口径：WSS-min **peak 收缩期**（`peak_step=1162`），非 V3P merged-1146
- 插值：Gaussian `r=3 mm, sharpness=2, max_dist=3 mm`
- 坐标系：bundle 刚性配准帧（mm）；STL 已变换到同一帧

## 主文件

| 文件 | 用途 |
| --- | --- |
| **`RAN_QING_BO__surface_wall.vtp`** | ParaView 面片云图（含 wss_cfd / wss_pred / err_wss / abs_err_wss） |
| `RAN_QING_BO__pointcloud_wall.vtp` | 原始壁面点云（未插值） |
| `plots/fig_wss_triptych.png` | CFD | Pred | Error 三联预览 |
| `_export/RAN_QING_BO__wall.csv` | 同点指标口径源（正式 R2 只在此算） |
| `GNN_blue_white_red.xml` | 蓝-白-红色标 |

## ParaView 步骤

1. Open `RAN_QING_BO__surface_wall.vtp` -> Apply
2. Representation = **Surface**
3. Coloring：`wss_cfd` / `wss_pred` / `err_wss` / `abs_err_wss`
4. 导入同目录 `GNN_blue_white_red.xml`；CFD 与 Pred **共用同一 Data Range**
5. 正式数字读 `_export/*__wall.csv` 或 `manifest_bundle.json` 的 `pointcloud_metrics`，不要在面片上算 R2
