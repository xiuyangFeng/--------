# T-null / T0 / TB8 中位病例相位后处理包

- 病例：`AAA/unruputer/SHEN_FANG_JIN`（cv3 fold 2 留出）
- 选例：T0 ckpt_best physical_overall_r2 (peak frame 1162, held-out 136 cases)
- 中位数 R²=0.7156；本例 T0 峰值 R²=0.7159（Δ=0.000321）
- checkpoint：`ckpt_best.pt`；T-null = D2 B_scale（cascade 折外峰值 × 训练折帧统计）
- 插值：Gaussian r=3 mm, sharpness=2, max_dist=3 mm
- 正式 R²/MAE 只在同点 CSV 上算，不要在面片上重算

## 相位代表帧

| 相位 | frame | step | Q/Qpeak | 定义 |
| --- | ---: | ---: | ---: | --- |
| `accel` | 13 | 1146 | 0.406 | 加速（Q 上升、尚未到 0.8 Qpeak） |
| `peak` | 21 | 1162 | 1.000 | 峰值帧（协议步 1162，Q/Qpeak=1） |
| `decel` | 35 | 1190 | 0.431 | 减速射血（峰值窗之后） |
| `trough` | 48 | 1216 | 0.003 | 谷底（协议波形最低 Q） |

## 打开这些 Gaussian 面片

| 臂 | 相位 | VTP |
| --- | --- | --- |
| `T-null` | `accel` | `AAA__unruputer__SHEN_FANG_JIN/T-null/accel/surface_gaussian.vtp` |
| `T-null` | `peak` | `AAA__unruputer__SHEN_FANG_JIN/T-null/peak/surface_gaussian.vtp` |
| `T-null` | `decel` | `AAA__unruputer__SHEN_FANG_JIN/T-null/decel/surface_gaussian.vtp` |
| `T-null` | `trough` | `AAA__unruputer__SHEN_FANG_JIN/T-null/trough/surface_gaussian.vtp` |
| `T0` | `accel` | `AAA__unruputer__SHEN_FANG_JIN/T0/accel/surface_gaussian.vtp` |
| `T0` | `peak` | `AAA__unruputer__SHEN_FANG_JIN/T0/peak/surface_gaussian.vtp` |
| `T0` | `decel` | `AAA__unruputer__SHEN_FANG_JIN/T0/decel/surface_gaussian.vtp` |
| `T0` | `trough` | `AAA__unruputer__SHEN_FANG_JIN/T0/trough/surface_gaussian.vtp` |
| `TB8` | `accel` | `AAA__unruputer__SHEN_FANG_JIN/TB8/accel/surface_gaussian.vtp` |
| `TB8` | `peak` | `AAA__unruputer__SHEN_FANG_JIN/TB8/peak/surface_gaussian.vtp` |
| `TB8` | `decel` | `AAA__unruputer__SHEN_FANG_JIN/TB8/decel/surface_gaussian.vtp` |
| `TB8` | `trough` | `AAA__unruputer__SHEN_FANG_JIN/TB8/trough/surface_gaussian.vtp` |

ParaView：Open `surface_gaussian.vtp` → Representation=Surface → Coloring 选 `wss_cfd` / `wss_pred` / `err_wss`。CFD 与 Pred 共用同一 Data Range（Pa）。`wss_*_selfmax` 无量纲，只比分布。
