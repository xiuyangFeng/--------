# 波 3：PF6 / VF6 三 seed 配对确认（wss_local_wave3_20260913）

用户 2026-09-13 拍板：把 F6（Murray 分支流量先验）接体场的单 seed 结果补成三 seed 配对证据，通过后 PF6 / VF6 作为压力 / 速度新底座。8 次单 run，Slurm 4 卡 × 2 槽（预检 14198 → 矩阵 14199，`COMPLETED 0:0`，1 h 10 min；提交记录 `submission.json`）。

| 臂 | 变化 | 配对参照 |
| --- | --- | --- |
| PF6_s7 / PF6_s2025 | P02（压力最优，SA3 后全局 BT）+ F6，seed 7 / 2025 | P02_s7 / P02_s2025（同 seed、同继承初始权重） |
| VF6_s7 / VF6_s2025 | V07（速度最优，新 SA3 单半径 0.20 + BT）+ F6 | V07_s7 / V07_s2025 |
| P02_s7 / P02_s2025 / V07_s7 / V07_s2025 | 同 seed 对照（从 `refs/P00_s<seed> → refs/P02_s<seed>` 参考链配对初始化，从零训练） | — |

- seed 1234 成员来自波 2（PF6_s1234 / VF6_s1234 / P02r_s1234 / V07r_s1234）与体场注意力矩阵（P02_s1234 / V07_s1234）。
- 生成：`tools/prepare_wss_local_wave3.py`（复用波 2 的体场 Murray 特征统计 `union_murray_volume_train138.json`）；预检 / 队列 / 报告复用波 1 工具（`--config-dir` / `--experiment-dir`）。
- 判读：三 seed 配对 Δ 同号且体场主误差（压力 MAE、速度向量 RMSE）不劣才晋级；结果与判读写入跟踪文档 §20.5。

## 结果（2026-09-13；全表 [results.md](results.md)）

| 目标 | seed 1234（波 2）/ 7 / 2025 配对 Δ R²_cb | 三 seed 均值 | 主误差 | 决定 |
| --- | --- | --- | --- | --- |
| 压力 PF6 vs P02 | +0.035 / +0.062 / +0.039 | +0.045（F6 0.7894 vs 对照 0.7445） | MAE −11.6% / −9.8% / −8.5% | 晋级压力底座 |
| 速度 VF6 vs V07 | +0.023 / +0.026 / +0.033 | +0.027（0.7964 vs 0.7690） | 向量 RMSE −2.6% / −2.7% / −5.0%；高速区 R² 0.17→0.29–0.32 | 晋级速度底座 |

判读见跟踪文档 §20.5；工作簿速度与压力表 82–97 行（`tools/update_wss_local_wave2_volume_xlsx.py --name wss_local_wave3_20260913 --group …`），验收 `xlsx_acceptance_volume.json`。
