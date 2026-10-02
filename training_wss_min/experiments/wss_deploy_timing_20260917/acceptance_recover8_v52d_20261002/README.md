# v5.2d 发布包上线验收：recover8（2026-10-02）

被验收的两个包：`X5Dcap_asym2_v52d_3seed_20261002`（峰值 WSS）、`M1cap_v52d_3seed_20261002`（峰值 WSS + TAWSS + OSI），由 `python -m wss_deploy.build_v52d_releases` 从 `training_wss_min/runs/wss_v52d_retrain_20261001` 的 full265 三 seed 冻结。验收用的是临时目录里同一工具打出的包，与正式包的 `release.json`、`MANIFEST.sha256`、`README.md` 逐字节相同。

## 1. 模型级等价（`model_equivalence.py` → `model_equivalence.json`）

训练侧 recover8 的 8 个病例输入（`dataset.load_partition`，v5.2d 视图）直接喂部署加载器 `Release.predict`，与训练评估时保存的三 seed 逐点预测均值比对。

- 峰值包：8 例全部壁面点，最大绝对差 ≤ 1.3e-5 Pa，最大相对差 ≤ 4e-7。
- 三头包：峰值 WSS / TAWSS / OSI 三通道，最大相对差 ≤ 1.1e-6。

即 GPU 浮点抖动量级，权重、特征统计、目标统计和集成口径都打包无误。

## 2. 端到端（`e2e_recover8.py` → `e2e_recover8.json`，GPU 2，10 min）

recover8 的标准 STL（`find_stl`，读标准清单）→ 部署 A 段（vessel_geom 中心线 + 自动出口命名，不人工干预）→ B 段分别用新旧四个包 → 与 v5.2d CFD 标签在最近 CFD 壁面节点上比（峰值帧 WSS；三头包另比 TAWSS / OSI）。病例等权 Pa R²_cb：

| 包 | 峰值 WSS | TAWSS | OSI | 逐例最低（峰值） |
|---|---:|---:|---:|---:|
| X5D_v51_5seed_20260916（旧，已下线） | 0.830 | — | — | 0.660 |
| **X5Dcap_asym2_v52d_3seed_20261002** | **0.854** | — | — | 0.699 |
| M1_3head_3seed_20260922（旧，已下线） | 0.808 | 0.809 | 0.445 | 0.611 |
| **M1cap_v52d_3seed_20261002** | **0.839** | **0.850** | **0.489** | 0.674 |

- 峰值包 8 例中 7 例好于旧包（WANG_SHUN_WEN 0.855 对 0.871 例外）；三头包 TAWSS 8/8、OSI 7/8 好于旧包。
- 与训练侧 recover8 读数对照：峰值 0.854 对 0.839、三头 TAWSS 0.850 对 0.849、OSI 0.489 对 0.542。OSI 的差来自部署点云与 CFD 节点的最近邻配对（OSI 空间变化最碎），新旧包同样受影响。
- 出口命名 8/8 四个出口全对；自动门放行 5 例，FU_GUO_JUN / LIU_YU_MING / SHI_YUN_XI 把握度 74–94% 低于 95% 门槛，部署时会停下等人工确认（名字本身是对的）。
- B 段用时 6–20 s/例/包，与旧包相同。

## 3. 黄金回归走归档目录（`golden_regress_retired_fallback.json`）

把发布根换成只含现役三包的临时目录（`WSS_DEPLOY_RELEASE_ROOT`），`python -m wss_deploy.regress --jobs-root outputs/wss_deploy_golden/20260920_baseline`：6/6 PASS。其中 5 个任务绑定旧 X5D / M1，只能经 `outputs/wss_deploy_release_retired/` 回退加载，说明旧包下线后回归门照常可用。

任务目录（1.8 GB）留在会话临时目录，没有拷进来。
