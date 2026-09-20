# 波 6：部署点云密度 — A（STL 重采样模拟）与 B（密度增广 X5D）（wss_local_wave6_20260915）

用户 2026-09-15 拍板"做 A 和 B"。判读见跟踪文档 §22。

## B：训练矩阵

3 次单 run（预检 Slurm 14264 通过含 C1 锚点重评 → 矩阵 14265，3 卡各 1 槽，`COMPLETED`，3 训练 + 6 评估全部退出码 0；`submission.json`）。

| 臂 | 变化 | 配对参照 |
| --- | --- | --- |
| X5D_s1234 / s7 / s2025 | X5 配方 + 训练期密度增广：每例每轮 p=0.6 换成 70/50/35/25% 体素抽稀云（`data_wss_v5/views/wss_min_density_v1`，`tools/build_density_sidecars.py`，法向/曲率族/tn_dot 在抽稀云上重算），输入与结构逐位不变 | 同 seed 的 X5（同 C1_s<seed> 参考链） |

- 代码：`data.density_aug_root / density_aug_levels / density_aug_prob`（config.py 校验：只允许 random 采样、峰值帧 WSS）；`dataset.load_density_level / density_view` 与 `WSSMinDataset.__getitem__` 钩子；`tests/test_density_augmentation.py`。
- 生成：`tools/prepare_wss_local_wave6.py`；工作簿 WSS 表 410–412 行。

## A：STL 重采样部署模拟（只推理）

`tools/deployment_stl_simulation.py`：每例原始 STL（与 CFD 同帧同 mm）按目标间距重采样（校准到中位最近邻距离），用部署几何程序从零建 case（中心线投影、解剖坐标架、PCA 法向 + 虚拟盖、曲率族、`wall_flowref_v1.compute_point_features`），真值取最近 CFD 节点，参照 = 同 checkpoint 全云预测按最近节点映射。结果 `offline/stl_x5/`（0 = 原生密度、0.5 / 0.8 / 1.2 mm）、`offline/stl_x5_fine/`（0.35 mm）、`offline/stl_x5d/`。

- 训练数据密度调查：`offline/cfd_wall_spacing.json`（AG 1.17 mm、AAA 0.47、ILO 0.46）。
- X5D 验收：`offline/run_x5d_acceptance.sh` → `offline/deployment_reeval_x5d/`、`offline/deployment_reeval_x5_3seed/`（同 3 seed 的 X5 对照）、`offline/stl_x5d/`。

## 结果摘要

| 比较 | 结果 |
| --- | --- |
| A：X5 五 seed 在 STL 重采样云 | 0.35 / 0.5 / 0.8 / 原生 / 1.2 mm 对同点参照：−0.021 / **−0.012** / −0.043 / −0.070 / −0.147 → 部署合同"处处不粗于约 0.5 mm" |
| B：X5D 全密度 test34 | 三 seed 配对 Δ物理 −0.012 / −0.001 / +0.021（均值 +0.003，带内），Δ归一化 +0.004（3/3 正），逐例 23/11、21/13、21/13 |
| B：X5D 密度扫描（3 seed 集成，X5 同 3 seed 对照） | 抽稀 50/25/10%：−0.022/−0.068/−0.171（X5 −0.061/−0.183/−0.397）；STL 0.5/0.8/原生/1.2 mm：−0.022/−0.036/−0.029/−0.047（X5 −0.012/−0.043/−0.070/−0.147）；全密度相同 → X5D 定为部署底座候选，仍建议重采样到约 0.5 mm；待补 seed 11/2026 成五 seed 集成 |

## 波 6b：补 seed 11 / 2026（wss_local_wave6b_20260915）

预检 14282 → 矩阵 14283（2 卡各 1 槽，`COMPLETED`，退出码全 0）；`tools/prepare_wss_local_wave6b.py`；工作簿 413–414 行。五 seed 对照 `../wss_local_wave6b_20260915/five_seed_ensemble.json`：X5D 单 seed 0.7247 ± 0.012（X5 0.7172 ± 0.005）；五 seed 配对 Δ物理 +0.0075（带内）、Δ归一化 +0.0043（5/5 正）；**五 seed 集成 Pa 均值 X5D 0.7465 / 0.8737 对 X5 0.7402 / 0.8712 → 部署形态改为 X5D 五 seed 集成**。五 seed 的 STL 0.5 / 1.2 mm 读数：`offline/stl_x5d_5seed/`。
