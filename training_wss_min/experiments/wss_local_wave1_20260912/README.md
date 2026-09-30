# 波 1 局部信息筛选矩阵（wss_local_wave1_20260912）

> 目的：按 2026-09-12 头脑风暴（`docs/02-推进与变更/_archive/WSS_PINN/_archive/WSS_局部信息与框架优化头脑风暴_2026-09-12.md`）把可快速实现的候选一次性做成单 seed 筛选。**每臂 seed 1234、400 轮、C1 底座、只改一处**（两个组合臂除外），Slurm 4 卡共享显存并行。结果只作筛选：单 seed 对单 seed 的 95% 带约 ±0.034 物理 R²_cb / ±0.009 归一化，不作显著性或泛化结论。

## 底座与协议

- 锚点：`configs/wss_direct_recovery_20260912/C1_s1234.json`（E2 切平面方向邻域 + E3 完整壁面 K16 query patch/FiLM；历史 best Pa R²_cb 0.6575 / 归一化 0.8410）。
- 所有臂 `train.init_reference_config` = C1：继承张量与 C1 初始权重逐位相同（`initialization.json` 的 `reference_state_sha256` = C1 的 `initial_state_sha256`），追加的输入列在 `stem` 与 patch MLP 中零初始化，新模块用各自子流初始化且初始为恒等（零初始化输出层）。
- 数据：冻结 V5 train138/test34、峰值 1162、support/query 各 5000、batch 8、AMP；评估固定 support + 全壁面 query、`legacy_vertex` 口径。
- 新几何 sidecar：`data_wss_v5/views/wss_min_flowref_v1/`（`wss_v5/views/wall_flowref_v1.py`，172/172，行与 bundle `wall_node_id_cas` 逐行一致）。
- 冻结特征统计：`feature_stats/<arm>_feature_stats.json` = C1 的 25 维统计 + 新键的 train138 统计（`feature_stats/union_new_keys_train138.json`）。

## 臂

| 臂 | 唯一变化 | 头脑风暴编号 |
| --- | --- | --- |
| X0 | C1 同配置同期重跑（同期对照） | — |
| X1 | 权重 EMA 0.999（多评一个 ckpt_ema） | P4 |
| X2 | +5 维弯曲参考角：bend_cos、bend_signed、torsion_rr、bend_cos_up2、bend_cos_up5 | F1 |
| X3 | +4 维分叉参考角：carina_cos、bif_plane_cos、bif_angle_cos、sibling_radius_ratio | F2 |
| X4 | +5 维上游历史：s_over_d、up_min_r_ratio_2d/5d、up_max_r_ratio_5d、up_kappa_5d | F3 |
| X5 | +2 维 Murray 分支流量先验：log_q_branch_murray、log_tau0_murray | F6 |
| X6 | +1 维 train138 人群先验 atlas_prior_logwss（训练例留一） | P5 |
| X7 | query patch 相对偏移改到 query 的 (轴向, 周向, 法向) 坐标架 | S1a |
| X8 | patch 的 mean 池化改为 query 条件注意力池化（零初始化 logit） | S1b |
| X9 | X7 + X8 + K 16→32 | S1c |
| X10 | 切平面 WSS 方向辅助头（2 通道，1−cos 损失 λ=0.1），标签来自 wall_wss_vec 峰值帧 | T1 |
| X11 | 中心线截面 token 上下文（4 mm bin、7×64 token、2 层 4 头 Transformer，零初始化残差加到 FiLM 上下文） | S3 |
| X12 | 混合专家头（4 个 head 拷贝 + 以 query 输入特征门控，门控零初始化） | S5 |
| X13a | 全帧预训练：random_frame + 相位特征 + 全帧统计（峰值保底 1/9） | P1 阶段 1 |
| X13b | 从 X13a ckpt_last 热启动，只训峰值帧 150 轮（lr 5e-4，峰值统计） | P1 阶段 2 |
| X15 | F1 + F2 + F3 + F6 共 16 维一起追加 | 特征组合 |
| X16 | X9 + X15 + X10 | 结构 + 特征 + 目标组合 |

本轮明确没做：T4 各向异性差分损失、F4 测地/HKS、F5 截面形状（几何候选仍 training_allowed=false）、F7 压力梯度级联、S2 窗口注意力、S4 (s,θ) 展开 U-Net、S6 DiffusionNet、P2/P3 预训练数据、T3 热点级联。

## 执行

- 生成：`python -m training_wss_min.tools.prepare_wss_local_wave1`（`configs/wss_local_wave1_20260912/matrix.json` 记录每臂相对 C1 的机械字段差）。
- 预检（GPU Slurm）：`cluster/preflight_wss_local_wave1.slurm` → `tools/preflight_wss_local_wave1.py`：C1 best 用新代码重评并逐字段比对存档 metrics（容差 5e-4）；17 臂各 3 步 AMP 前后向 + 全壁面推理分块一致性；X16/X11/X13a 走真实 CLI 2 轮训练 + 评估。产物 `runtime_preflight.json`。
- 矩阵：`cluster/run_wss_local_wave1.slurm` → `tools/run_wss_local_wave1_queue.py`（4 卡 × 2 槽、显存余量 ≥7000 MiB 才启动、源码/配置指纹每次启动前复核；X13b 等 X13a 完成后启动；EMA 臂多评 `ema`）。
- 汇总：`tools/report_wss_local_wave1.py` → `results.md` / `results.json`（相对 X0 与历史 C1 的 Δ、分域、逐例配对）。

提交记录见 `submission.json`；队列状态 `queue_status.json`；日志 `logs/`。

## 中期读数（第一轮 8 臂，2026-09-12 09:50）与追加批次 1b

第一轮 best 物理 R²_cb：X0 同期对照 0.6173（历史 C1 0.6575，同配置同初始权重的重跑仍差 −0.040，训练损失轨迹逐轮一致，差异全在 test 侧，即单 seed 读数带本身）；X5 Murray 分支流量先验 **0.7178**（vs X0 +0.100、vs 历史 C1 +0.060；归一化 +0.021；high-WSS R² 0.42、IoU 0.507、case P10 0.643，28/34 例变好）；X2 弯曲角 0.6510、X7 切平面 patch 0.6548、X3 0.6459、X4 0.6419、X6 0.6327、X1 EMA 0.6297——除 X5 外都在 X0 的单 seed 带内或边缘。

据此追加 **wave-1b**（`configs/wss_local_wave1b_20260912/`，`tools/prepare_wss_local_wave1b.py`，Slurm 依赖链接在 14160 之后）：X0/X5 各补 seed 7、2025（seed 配对的参考链 refs/C1_s<seed>→refs/M2_s<seed>，同 seed 的 X0 与 X5 共享初始权重）；X5q（只 log_q）与 X5t（只 log_tau0）拆分；X5F1（X5+X2）与 X5S1a（X5+X7）组合。共 8 次训练、16 次评估。

中期诊断（best 保存预测，test34）：X0 与历史 C1 的逐例 Pa R² 之差均值 −0.011、**标准差 0.071**（极差 −0.137 … +0.235），而两者 log_z 预测的逐例相关中位 0.982——同配置同初始权重的两次训练在逐例读数上就有 ±0.07 的抖动，这是本矩阵所有单臂读数的底噪。X5 相对 X0：逐例 ΔR² 均值 +0.082、28/34 例变好；Pa 校准斜率中位 0.698→0.800（幅值收缩减轻）；分支均值残差的跨支标准差 0.143→0.096（log_z），即 Murray 先验主要修掉了**逐支的整体偏置**，与头脑风暴 F6 的假设一致。

## 最终结果（作业 14160，7 h 46 min，17 训练 + 35 评估全部退出码 0；2026-09-12 14:00）

全表见 `results.md` / `results.json`（自动生成；best/last/ema、分域、逐例配对）；工作簿总览第 347–363 行（`xlsx_acceptance.json`，回填前备份 `WSS_PointNet实验矩阵与结果汇总last_backup_before_wave1_20260912_140050.xlsx`）；判读写在跟踪文档 §20。

要点（best Pa R²_cb / 归一化；X0 = 0.6173 / 0.8347，历史 C1 = 0.6575 / 0.8410）：X5 Murray 先验 0.7178 / 0.8559（唯一大效应，逐支尺度偏置被修掉）；X11 截面 token 上下文 0.6703 / 0.8457（结构侧唯一超带）；X15 0.7118 / 0.8627、X16 0.7140 / 0.8642（F6 主导，其余特征/结构只在归一化上再加 +0.007/+0.008）；X2 0.6510 / 0.8433 恰在带上；其余在带内。模块激活核对（best ckpt 零初始化层范数）：X8 patch 注意力 logit 1.78、X12 门控 1.35（专家末层两两相对差 0.45）、X11 截面残差输出 9.6、X10 方向头训练到方向余弦 0.87。

1b 批次已完成（14168/14169）：X5 三 seed 配对 Δ +0.069 物理 / +0.017 归一化，均值 0.7173（sd 0.004）；增益来自 log Q 份额；X5+X2、X5+X7 无物理增益。详见 `../wss_local_wave1b_20260912/README.md` 与跟踪文档 §20.3。
