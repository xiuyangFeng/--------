# V4 数据审查与中心线 / 标签修复图（2026-09-06）

> 汇报用途：终审版 §5。数字与诊断来自
> [`WSS_PINN_V4_anatomy-only预处理准备与173例数据核查_2026-09-03.md`](../../../02-推进与变更/WSS_PINN/_archive/WSS_PINN_V4_anatomy-only预处理准备与173例数据核查_2026-09-03.md)
> §12–13 / §23 / §26–27。本目录只收图，不改 `outputs/` 真源。

| 文件 | 说明 |
| --- | --- |
| `YANG_YU_QING-1_before_vs_after_iso.png` | 错 STL：中心线整体脱离血管 → 权威表面重提后回到管腔内 |
| `XIE_JIN_QUAN_before_vs_after_iso.png` | 旧算法漏支 → Centerline V2 找回 5 端点 / 3 分支 |
| `stl_vs_cfd_wall_three_mismatch_cases.png` | 三例权威 STL 不是 CFD 壁面（对照 CHEN 重合） |
| `fig_meshwall_three_iso.png` | 三例从 `.cas` 解剖壁面重提后 `pass`，`|d_wall−R|/R` p50 0.002 |
| `AAA__ruputer__LI_LAO_PING_adaptive_overlay_20260905.png` | 瘤腔伪折：固定 SG11 vs 半径自适应 |
| `LI_LAO_PING_sac_curvature_profile_adaptive_20260905.png` | 同例曲率 / `k·R` / 窗长剖面 |
| `end_zone_profiles_20260906.png` | 端点 1R 持值：四例端部伪峰压回 |
| `fig_pressure_wss_recalc.png` | (a) 10 例壁面–体域 Δp 重算前后对照；(b) D 层 4 例 WSS 20→60 次迭代对照 |
| `pressure_wss_before_after.csv` | 八例低压力族 / Q 长尾 / D 层：旧 V4 `steady_reference_pa`、81 帧均压、壁面峰值压、Δp、WSS p50/p99/max 前后 |

重新出图：`python3 plot_v4_data_review_figs.py`（先拷审查原图，再画两张合成图）。
