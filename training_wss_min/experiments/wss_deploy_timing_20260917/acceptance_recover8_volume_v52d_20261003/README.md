# v5.2d 体场发布包上线验收：recover8（2026-10-03）

被验收的包：`PF6_VF6_v52d_3seed_20261003`（压力 PF6 × 3 seed + 速度 VF6 × 3 seed），由 `python -m wss_deploy.build_v52d_volume_release` 从 `training_wss_min/runs/pf6vf6_v52d_retrain_20261002` 的 full265 三 seed 冻结；对照旧包 `PF6_VF6_peak_3seed_20260920`（v5.0，train138，已下线）。验收时新包在暂存根 `outputs/wss_deploy_release_staging_20261003/`，通过后原样移入发布根。

## 1. 模型级等价（`model_equivalence_volume.py` → `model_equivalence_volume.json`，作业 16975）

训练侧 recover8 的 8 个病例（v5.2d 体场数据根，壁面行在前、内部行在后，坐标变换取单位阵）直接喂部署加载器 `Release.predict`，与训练评估保存的三 seed 逐点均值比对：压力全部点最大绝对差 ≤ 8.6e-4 Pa，速度全部内部点 ≤ 1.5e-6 m/s，R² 对保存值 1 − 1e-14。GPU 浮点抖动量级；权重、特征统计、目标统计、集成口径打包无误。

## 2. 端到端（`e2e_recover8_volume.py` → `e2e_recover8_volume.json`，作业 16975，9 min）

recover8 的标准 STL → 部署 A 段（vessel_geom 中心线 + 自动出口命名，不人工干预）→ B 段新旧两个体场包（同一份 A 段输出、同一批查询点）→ 与 v5.2d CFD 体场标签在最近 CFD 点上比（压力：壁面节点 + 内部单元，相对体积平均；速度：内部单元，世界坐标）。病例等权 R²_cb：

| 包 | 压力 | 速率 | 逐例最低（压力 / 速率） |
|---|---:|---:|---:|
| PF6_VF6_peak_3seed_20260920（旧，已下线） | 0.913 | 0.870 | 0.673 / 0.697 |
| **PF6_VF6_v52d_3seed_20261003** | **0.930** | **0.889** | 0.754 / 0.734 |

- 压力 6/8 例、速率 8/8 例好于旧包；改善最大 LI_JIE-1/after（压力 0.67 → 0.89、速率 0.73 → 0.82）；变差 LIU_YU_MING 压力 0.82 → 0.75、ZHANG_ZAO_SHUAN 压力 0.97 → 0.95。
- 与训练侧 recover8 读数（CFD 网格口径）对照：压力 0.930 对 0.939、速率 0.889 对 0.903。最近点配对不是训练的 CFD 网格口径，只用于新旧配对比较。

## 3. 黄金回归（`golden_regress_retired_fallback.json`，作业 16976）

`python -m wss_deploy.regress --jobs-root outputs/wss_deploy_golden/20260920_baseline`：6/6 PASS（摘要差 0、数组差 0）；其中 20260920_113510 与 20260922_173705 绑旧体场包，经 `outputs/wss_deploy_release_retired/` 回退加载。

任务目录留在计算节点临时目录，没有拷进来。
