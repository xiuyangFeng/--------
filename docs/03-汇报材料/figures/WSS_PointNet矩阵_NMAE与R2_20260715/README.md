# WSS PointNet 正式矩阵 + E4 deeper · NMAE 与 R² regression（给老师）

生成日期：2026-07-15

## 口径

- **NMAE** = MAE / (max(true) − min(true))
  - `pooled`：test16 全部壁面点合在一起算
  - `case-mean`：每例先算 NMAE 再平均（更不受极端高峰病例主导）
- **物理空间**：Pa；仅 GLOBAL 可逆回；CASE 不报物理 NMAE
- **归一化空间**：GLOBAL = train61 全局 log-z；CASE = WSS/WSSmax（真值 max≈1，故 NMAE≈MAE）
- **R² regression 图**：横轴 true、纵轴 pred，红色虚线 y=x；色块为 log-count hexbin
- 主模型：`ckpt_best(train_loss)`；分区：test16

## 数字摘要

| Run | 物理 NMAE pooled | 物理 R² cb | 归一化 NMAE pooled | 归一化 R² cb |
| --- | ---: | ---: | ---: | ---: |
| E0-GLOBAL | 2.33% | 0.141421 | 8.25% | 0.396355 |
| E2-GLOBAL | 2.23% | 0.213966 | 7.79% | 0.4606 |
| E3-GLOBAL | 2.30% | 0.163705 | 8.09% | 0.421763 |
| E23-GLOBAL | 2.25% | 0.198763 | 7.85% | 0.459778 |
| E4-DEEP-GLOBAL | 2.28% | 0.162851 | 7.90% | 0.447638 |
| E2-CASE | — | — | 5.77% | 0.172354 |
| E3-CASE | — | — | 5.90% | 0.155149 |

## 文件

- `nmae_r2_summary.csv` / Excel 工作表 `NMAE与R2`
- `r2_regression_<RUN>_{physical|normalized}.png`
- `r2_regression_grid_physical.png`、`r2_regression_grid_normalized.png`

## 注意

- 物理 NMAE（range）因个别高峰点把分母拉大，数值会偏小；判读请同时看 R² / Spearman / top10，不要单靠 NMAE 宣布 Go。
- CASE 与 GLOBAL 的归一化 MAE/NMAE **不可直接横比**（目标尺度不同）。
- E0 使用已有 `model.eval()` 全点重推理缓存；E2/E3/E23/E4/CASE 使用各自 best PostView 同点 CSV。
