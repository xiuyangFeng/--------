# 波 5：狭窄指数显式化 + 患者分组 3-fold（wss_local_wave5_20260913）

用户 2026-09-13 拍板"完成第一和第二批并对 X5+X11 做 3-fold"，只允许部署可得输入。12 次单 run，Slurm 4 卡 × 3 槽共享显存并行（预检 14208 → 矩阵 14209，`COMPLETED`，12 训练 + 24 评估全部退出码 0，约 4 h 50 min；提交记录 `submission.json`）。

| 臂 | 变化 | 配对参照 |
| --- | --- | --- |
| X5I_s1234 / s7 / s2025 | C1 + Murray 先验 + `log_r_over_rdistal`（sidecar v1.3：ln(R_local/R_distal(segment))，Murray 分流同一末端半径配方） | 同 seed 的 X5（同 C1_s<seed> 参考链，只多一列零初始化输入） |
| X5AI_s1234 / s7 / s2025 | C1 + capfit 分流 + `log_r_over_rdistal` | 同 seed 的 X5A |
| X5_f0/f1/f2_s1234 | X5 配方在 cv3 fold k（train = 其余两折 91–93 例，test = 折外 45/47/46 例；逐折 log_z 与特征 z-score） | — |
| X5X11_f0/f1/f2_s1234 | X5 + S3 截面 token 上下文，同折 | 同折 X5_f<k>（同一 C1 参考链，241 继承张量逐位相同） |

- 数据：`wss_v5/views/wall_flowref_v1.py` 升 v1.3（新键一列，172 例重建 24 s，旧 22 键逐位不变，`tests/test_wss_local_wave5.py`）；cv3 切分 `tools/make_cv3_splits.py` → `data_wss_v5/views/wss_min_view_v1/cv3_20260913/`（按 AG/AAA/ILO 分层、重复几何组同折、seed 20260913，test34 列为 unused）。
- 生成：`tools/prepare_wss_local_wave5.py`（狭窄列 train138 统计 `feature_stats/union_idx_wall_train138.json`，逐折统计 `feature_stats/fold{k}_train_feature_stats.json`）；本地验证四组配对的继承张量逐位相同、追加列为零。
- 结果：`results.md`（通用报告，fold 臂的"对 C1"列无意义，读折外列）、`offline/analyze_wave5.{py,json}`（配对 Δ、cv3 合并折外、折模型在冻结 test34）、工作簿 WSS 表 386–397 行（`tools/update_wss_local_wave1_xlsx.py --name wss_local_wave5_20260913 --group …`，回填前备份见 `xlsx_acceptance.json`）。
- 同目录 `offline/` 还有本轮不训练的三项：`f7_pressure_gradient_prereg`（F7 预登记，未过门槛）、`ensemble_protocol`（集成口径）、`deployment_reeval_x5/`（部署重采样复评，工具 `tools/deployment_resample_reeval.py`）。

## 结果（best；判读见跟踪文档 §21.4–§21.6）

| 比较 | 结果 | 决定 |
| --- | --- | --- |
| X5I vs 同 seed X5（3 对） | 物理 −0.002 / +0.006 / +0.001（均值 +0.001），归一化 0.000；逐例 17/17、21/13、17/17 | 零效应 |
| X5AI vs 同 seed X5A（3 对） | 物理 +0.004 / +0.001 / −0.003（均值 +0.001），归一化 −0.001（3/3 负）；ILO 仍比 X5 低 0.02–0.06 | 狭窄列补不回 capfit 的 ILO 损失；X5 仍为底座 |
| cv3 X5X11 vs X5（3 折，seed 1234） | 折外 Δ −0.004 / −0.009 / +0.018，合并 138 例 +0.000 / −0.003，逐例 54 胜 84 负；high-WSS R² 三折升，IoU 三折略降 | X5+X11 不进主线 |
| cv3 X5 合并折外 | Pa 0.6802 / 归一化 0.8461（AG/AAA/ILO 0.741/0.697/0.596）；折模型在冻结 test34 0.701/0.672/0.700 | 训练例少 1/3 约 −0.02 |
| 部署重采样复评（X5 五 seed） | 抽稀到 50/25/10% 重算特征：−0.068 / −0.189 / −0.406（同点参照）；只动模型邻域：−0.031 / −0.090 / −0.204 | 部署点云须重采样到训练密度（0.41 mm） |
| F7 预登记 | PF6 预测压力梯度可解释 X5 残差 0.3%（真值压力 25%） | 不训 |
| 集成口径 | Pa 空间平均 0.7402 ≈ seed 方差 Jensen 0.7413 > log 均值 0.7354 | 部署口径改 Pa 空间平均 |

后续：波 5b（`wss_local_wave5b_20260913`）= T3 热点级联三 seed + 对照 + cv3 seed 7。
