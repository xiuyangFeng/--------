# 峰值体域 `u,v,w,p` PINN 路线

> 当前活动路线：`volume_uvwp_peak_field_v4`；`volume_uvwp_peak_qs_smooth_v3` 为冻结历史基线
>
> 更新日期：2026-08-06
>
> 状态：**field-v4 Stage 0-a/0-b、B0、四臂 seed1234、前二臂三种子确认、导数/support
> Gate 和 full-volume val15 晋级审计均 completed；26 tests、CPU smoke、静态/GPU
> preflight 与所有 Slurm 作业通过。最终保留 G-Raw，G-PE 与 local conditioner No-Go。**

V3 的完整预注册目标见
[已归档的准稳态平滑场六臂预注册提示词](./_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)。
当时冻结的 inlet/wall 与 monitor-derived outlet 资产已完成 173/173 Gate；无 3NN 条件 PointNet 平滑场
按 `xyz/xyz+geom × DATA/DATA+BC/DATA+BC+PDE` 六臂完训。test35 只在训练结束后按
预注册的三类 checkpoint 统一评估，未用于训练控制或主 checkpoint 选择。

> **2026-08-05 对抗性复核限定**：已完成 V3 使用的四出口目标应精确称为
> **exact-timestep, monitor-derived pressure proxy**。它尚未被证明等于真实
> pressure-outlet face 上的 RCR UDF profile，也未被证明是部署时可得输入。
> 这不改变已完成六臂的历史数值，但会限制 BC 机制解读。同时，
> test35 虽未参与 V3 训练/选 checkpoint，但已被用于后续架构、采样和 loss
> 设计；下一轮应把它视为 development-exposed screen，不再当作新模型的
> 未触碰确认集。详见[核心代码诊断与下一轮设计建议](./核心代码诊断与下一轮设计建议_2026-08-05.md)。

> **2026-08-06 当前范围**：下一轮唯一优化主线是峰值体域 `u,v,w,p`。已验证的
> Profile-Secant V3 velocity→WSS 算法冻结为下游检查器，不再继续调算法；本路线不新增
> 直接 WSS 输出、WSS loss 或 WSS 辅助监督，也不允许用 WSS 指标反选 checkpoint、loss
> 或架构。WSS 只在基于 validation 的 `u/v/w/speed/p`、区域、流量和压降指标选定主
> checkpoint 后运行一次，作为 downstream sanity check。
>
> **执行状态**：`FROZEN FOR IMPLEMENTATION v1.0` 的 Stage 0–1 已于 2026-08-06
> 在独立 route 完成；V3 历史产物未覆盖，Stage 1 未读取 test35、未输入 BC、未使用
> PDE 或 WSS 选模。执行合同见
> [冻结诊断后 Stage 0–1 提示词](./WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1.md)。

## field-v4 Stage 0–1 结果（2026-08-06）

### Gate 与数据合同

- Stage 0-a：pass；修复病例等权 `S_field^cb`、固定 val15×5000 query、主 checkpoint
  `best_validation_field_cb.pt`，并保留 point/legacy batch 聚合审计；
- Stage 0-b：pass；173 例覆盖、138 train/val deep audit、35 test35 仅 guard 未读取；
  552 个 outlet 记录中 536 个来自日志双路径、16 个使用 face-pressure fallback；
  train-only 阈值 `22.2161 Pa`；
- B0：train-only 病例盲 atlas，val15 `S_field^cb=0.699472`；
- Stage 1：统一 `xyz+geom`、data-only、无 BC/PDE、无 warm-start，参数量最大/最小比
  `1.0222 < 1.10`。

### 单种子矩阵与多种子确认

| 臂 | seed1234 `S_field^cb` | 状态 |
| --- | ---: | --- |
| G-PE | **0.567612** | 单种子第 1，进入确认种子 |
| G-Raw | 0.578311 | 单种子第 2，进入确认种子 |
| L-Raw | 0.588856 | 无单种子增量，停止 local decoder |
| L-PE | 0.597688 | 排名第 4，不晋级 |

确认 seeds `[1234,2345,3456]` 后，G-Raw 为 `0.576116±0.007477`，G-PE 为
`0.581208±0.012599`。G-PE−G-Raw 的配对差为
`[-0.010699,+0.015734,+0.010240]`，只在 1/3 seeds 改善；病例 bootstrap 均值差
`+0.004572`，95% CI `[-0.013164,0.030957]`。两臂都相对 B0 有明确增量，但 PE
的跨种子改善方向不稳定。

### 稳健性与 Go/No-Go

- G-PE 相对 G-Raw 的 pressure RMSE 退化 `2.36%`，超过预注册 `2%` 通道护栏；
- 最差 ILO 队列的病例分数退化 `5.58%`，超过 `2%` 最差队列护栏；
- near-wall 四通道 RMSE 护栏通过；八个训练 checkpoint 的一/二阶导数与 support
  重采样 Gate 全通过，最大相对误差/敏感度为 `1.79e-8 / 1.91e-5 / 2.997%`；
- full-volume val15 每 seed 覆盖 `12,502,817` 个严格体域点，G-Raw/G-PE 为
  `0.576816±0.007717 / 0.581096±0.013807`，与固定 5k 结论一致。

最终决定：**保留 G-Raw；G-PE No-Go 并关闭继续扫频；L-Raw 单 seed 无增量，停止
继续堆 local decoder。** 当前证据把下一假设转向 patient-specific conditioning /
缺失可部署 BC；Stage 2 尚未启动。本轮未运行 test35、WSS、BC loss/input 或 PDE loss。

机器可读真源：

- `outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0a/report.json`
- `outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0b/report.json`
- `outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage1_raw_gate/report.json`
- `outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage1_single_seed_gate/report.json`
- `outputs/wss_pinn/volume_uvwp_peak_field_v4/b0/evaluation_val15.json`
- `outputs/wss_pinn/volume_uvwp_peak_field_v4/stage1_multiseed_and_promotion_report.json`
- `outputs/wss_pinn/volume_uvwp_peak_field_v4/stage1_b0_case_deltas.csv`
- `outputs/wss_pinn/volume_uvwp_peak_field_v4/stage1_trained_derivative_support_gate.json`

## V2/V3 冻结后全点线性回归审计（2026-08-06）

为散点参考图新增带截距 OLS `y=a·x+b` 诊断（a、b、拟合线 R²），用于判断各臂是
高值段还是低值段未学会。该审计只做下游重新推理与统计，不改任何训练、checkpoint
或 WSS 算法；原 raw/scaled R²、MAE、RMSE、NMAE 指标全部保留。

- 冻结边界：V2 八臂用 `last@7500`、V3 六臂用 `best_validation_data`，均在 test35
  上评估，共 14 臂 × 35 例 = 490 份逐病例结果，全部完成并通过校验；
- 速度口径：CFD 与预测速度标量均由 `u,v,w` 重新合成；逐例用解剖 STL 封口
  enclosed-point 裁剪去除 CFD 人工延长段后取全部体内点（35 例严格体域
  27,803,241 点保留 15,757,669 点，保留率 56.68%）；
- WSS 口径：全部冻结壁面节点（每例 9,704–96,719 点，远超原 1200 点协议），
  预测 WSS 由冻结 Profile-Secant V3 velocity→WSS 算法从预测速度场重算；
- 关键读数：所有 14 臂速度斜率 `a` 仅 0.20–0.43，系统性远小于 1，即高速段
  被普遍压缩低估；speed fit R² V2-PNPP 臂 0.32–0.34、V2-PN 臂 0.14–0.20、
  V3 六臂 0.24–0.29（`QSF-XYZG-DATA` 最高 0.2878±0.1320）；WSS fit R² 最高为
  V2 `VF-PNPP-XYZG` 双臂 ≈0.294；
- 产物：`outputs/wss_pinn/audits/v2_v3_linear_regression_fullpoints_20260807/`
  （目录名为 UTC 日期），每例含 `metrics.json`、speed/wss 回归密度图与
  `density_histograms.npz`；每臂 `summary.json` 与 6 张汇总图（worst/most-
  frequent/best 代表病例回归图、R² 分布、按 R² 从 worst 到 best 排序的
  case–R² 图）；ROI 审计见同目录 `anatomical_roi/manifest.json`；
- 工具与回填：`docs/03-汇报材料/tools/add_wss_pinn_v2_v3_linear_regression.py`
  （roi/arm/summarize/workbook/verify 子命令，病例级可恢复）；
  `WSS_PINN_V1_V2_V3_field-v4实验矩阵与指标汇总_2026-08-06.xlsx` 主表由 34 列扩为
  40 列并新增「线性回归汇总」「线性回归逐病例」两表，修改前原件已备份为
  `…_备份_新增线性回归指标前.xlsx`。

## 0. V3 准稳态平滑场六臂（2026-08-05 完训并评估）

- 科学定位：“Fluent 瞬态 peak 标签 + Carreau–Yasuda 准稳态 PINN 正则”；PDE 不含
  `du/dt`，不能写成稳态 CFD surrogate。
- 数据：train123/val15/test35；split SHA256
  `c80cb65ad95f7d76fadaff82ea02c35c27ad24ab2474e1050bf981eff97493f9`。
- boundary Gate：173/173 passed；6 例保留“UDF 分母与真实网格面积不同”的 provenance
  warning，入口速度使用 exact peak 实测流量除真实入口面积，不把陈旧分母当真值。
- 模型：support `xyz` 或 `xyz+geom` 编码病例 latent；query decoder 只接 query xyz +
  latent，纯 Linear+tanh，无 3NN/IDW/BN/LN/dropout/AMP。
- 六臂：DATA、DATA+BC、DATA+BC+PDE × `xyz/xyz+geom`，2500 epoch，seed1234，
  每组三臂初始化 SHA256 相同。
- 资源：master 四臂由 Slurm array `11301_[0-3]` 调度四张 4090；node04 两臂为
  PID `1241185/1241335`，分别使用 A100 GPU0/1。
- 机器可读真源：
  `outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_boundary/report.json`、
  `outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_preflight/report.json`、
  `outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_gpu_preflight/report.json`、
  `outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/submission_six_gpu.json` 和
  `monitor_snapshots.jsonl` / `monitor_30min_summary.json`。

正式启动后 31/31 快照均 healthy、六臂全程 running、错误关键字 0，前 30 分钟监控
已通过。最终六臂均 `status=completed`、2500 epoch / 155000 step，
`best_validation_data/best_validation_total/last` checkpoint 6/6 齐全；三类 checkpoint
的 test35 全严格体域评估共 18/18 完成。主表、增量、逐病例和收敛真源见
[`summary/README.md`](../../../outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/README.md)。

### 0.1 主结果：`best_validation_data`

| 臂 | 输入 / 模式 | speed R²_cb / pooled | speed MAE / RMSE (m/s) | near-wall / core R²_cb | p R²_cb / pooled |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | xyz / DATA | 0.0344 / 0.2729 | 0.2064 / 0.2901 | -3.1933 / 0.0703 | -0.1848 / 0.5294 |
| E2 | xyz / DATA+BC | 0.0359 / 0.2658 | 0.2050 / 0.2906 | -2.5279 / 0.0610 | 0.0958 / 0.5038 |
| E3 | xyz / DATA+BC+PDE | -0.0143 / 0.2420 | 0.2113 / 0.2974 | -2.6543 / 0.0130 | 0.0336 / 0.3875 |
| E4 | xyz+geom / DATA | 0.2263 / 0.3541 | 0.1948 / 0.2715 | -1.3163 / 0.2445 | 0.5369 / 0.6648 |
| E5 | xyz+geom / DATA+BC | **0.2336** / **0.3523** | **0.1949** / 0.2724 | -1.0154 / **0.2470** | **0.6035** / **0.6486** |
| E6 | xyz+geom / DATA+BC+PDE | 0.1799 / 0.3144 | 0.1996 / 0.2809 | **-0.9561** / 0.1869 | 0.5971 / 0.5360 |

`R²_cb` 为逐病例等权平均。E5 的 case-balanced speed R² 最高，E4 的 pooled speed R²
略高；二者应按口径并列报告，不能只挑一个汇总量宣称绝对最优。

### 0.2 BC、PDE 与 geom 增量

- BC：E2−E1 的 speed R²_cb/pooled 为 `+0.0015/-0.0070`，E5−E4 为
  `+0.0073/-0.0017`。BC 对总体速度近中性且两种汇总口径方向不一致；但其直接监督的
  入口流量 ARE 与出口压力 MAE 分别改善 17.0%/19.0%（xyz）和 46.5%/32.8%
  （xyz+geom）。这些边界指标是相对当时冻结的 monitor-derived outlet
  proxy 计算，不能外推为对真实 RCR boundary profile 的精确恢复。
- PDE on BC：E3−E2 的 speed R²_cb/pooled 为 `-0.0502/-0.0239`，E6−E5 为
  `-0.0537/-0.0379`；speed MAE 同时增加。另一方面 continuity/momentum RMS 分别
  降低 88.5%/60.3% 与 90.1%/61.6%。这是准稳态正则改善方程一致性、却与瞬态 peak
  标签竞争的负结果，不能解释为“还没训够”。
- 完整 physics：E3−E1、E6−E4 的 speed R²_cb 分别为 `-0.0487/-0.0463`，首轮 No-Go。
- geom：E4−E1、E5−E2、E6−E3 的 speed R²_cb 分别为
  `+0.1919/+0.1977/+0.1942`，pooled 为 `+0.0812/+0.0865/+0.0725`，且 MAE 同步下降；
  这是六臂中唯一稳定、跨三种损失模式一致的正增量。

### 0.3 收敛、loss dominance 与限制

- `best_validation_data` 位于 epoch 92–203；最后 10% validation data loss 六臂均小幅
  上升 `0.28%–0.45%`，没有任何臂满足“仍下降超过 1%”的统一续训条件，不续到 5000。
- `last` 的 speed R²_cb 六臂均低于主 checkpoint，说明后期 validation 泛化退化；
  test35 未用于这一判断，也未用于反选 checkpoint。
- PDE 臂 late-stage 梯度裁剪均值约 21.7%–21.8%，高于 DATA/BC 臂的 4.7%–8.4%，
  但未占多数 step。PDE group 无单项长期 >90%。
- BC validation group 的 outlet pressure 子项在 E2/E3/E5/E6 最后 10% 平均占
  96.4%/95.4%/95.5%/91.9%，单项 >90% 的 epoch 比例均为 100%；下一轮若继续 BC，
  应先修分 zone/尺度平衡，而不是盲目加 epoch。
- near-wall speed R² 六臂仍全部为负，不能加入 WSS 可靠性结论。未发现零速度或病例间
  常数场塌缩，但病例内 speed 预测方差中位数仅是真值的 0.19–0.38，存在振幅压缩。
- 冻结的 monitor-derived outlet pressure proxy 可评估；outlet flow 因冻结资产没有 exact target、面法向和面积
  权重，只保留 PCA 法向质量守恒诊断，不作为 CFD 出口流量误差结论。
- 老师汇报用的六臂 data/physics loss 图、每臂最后 10% 放大图、BC/PDE 子项图和 9 页
  PDF 见
  [`loss_convergence/README.md`](../../../outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/loss_convergence/README.md)。图上可直接看到 train loss 已平台，而
  validation data/physics loss 在 epoch 69–204 达到最优后持续变差，因此是“优化已
  收敛但泛化过拟合”，不是“尚未收敛”。
- 教师查看与汇报用的统一工作簿见
  [`WSS_PINN_V3_六臂核心实验与指标汇报.xlsx`](../../../outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/WSS_PINN_V3_六臂核心实验与指标汇报.xlsx)，
  7 张表覆盖六臂主指标、九组增量、checkpoint敏感性、loss收敛、逐病例和评估口径。

## 1. 历史 V1/V2 目标与边界

历史 V1/V2 只回答一个问题：在相同 PointNet / PointNet++、相同峰值体域监督、相同随机
初始化和相同均匀采样下，加入不可压缩非牛顿流的 PDE 与 no-slip，能否比 data-only
更稳定地重建患者级 `u,v,w,p`。

本轮明确不做：

- 直接 WSS 输出或 WSS loss；
- PointNeXt、Transformer、LocalGeoPE；
- 全周期或非定常 `du/dt`；
- 从 data-only checkpoint 热启动 PINN；
- 旧 8k near-wall + 8k core sidecar 复用；
- 未经新数据 Gate 的正式训练。

旧的 WSS-target PINN 方案、F0/F1/F2 阶梯和既有运行记录保留为历史证据，入口见
[_archive/wss_target_v1_20260730](./_archive/wss_target_v1_20260730/README.md)。

### 1.1 与冻结 velocity→WSS 验证器的边界

本路线只重建体域 `u,v,w,p`，不直接输出 WSS。后处理计算器的 2026-08-04 推荐结果见
[Profile-Secant V3 结果](../../../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/PROFILE_SECANT_HIGH_TAIL_V3_RESULTS.md)。
该模型在 CFD 真速度输入下的 test35 全壁面病例 overall R² 均值约 `0.9604`、pooled
high-WSS R² 约 `0.9440`，已经足以作为冻结下游验证器；但逐病例 high-WSS R² 均值约
`0.7735`、平均峰值低估约 `15.81%`，不能当作无误差真值算子。

当前研发瓶颈因此前移到 `u,v,w,p`：优先改善 `u/v/w` 分量、速度向量、near-wall/core、
截面与分支流量，以及病例 pressure gauge、压力梯度和入口—出口压降。WSS 不参与训练
和选模；只有主 checkpoint 冻结后才做一次下游审计，以判断预测速度误差是否被梯度算子
放大。

## 2. 八实验矩阵

| 编号 | 架构 | 输入 | 训练模式 | 实验 ID |
| --- | --- | --- | --- | --- |
| E1 | PointNet | xyz | data-only | `VF-PN-XYZ-DATA-s1234-v1` |
| E2 | PointNet | xyz | PINN | `VF-PN-XYZ-PINN-s1234-v1` |
| E3 | PointNet | xyz+3 geom | data-only | `VF-PN-XYZG-DATA-s1234-v1` |
| E4 | PointNet | xyz+3 geom | PINN | `VF-PN-XYZG-PINN-s1234-v1` |
| E5 | PointNet++ | xyz | data-only | `VF-PNPP-XYZ-DATA-s1234-v1` |
| E6 | PointNet++ | xyz | PINN | `VF-PNPP-XYZ-PINN-s1234-v1` |
| E7 | PointNet++ | xyz+3 geom | data-only | `VF-PNPP-XYZG-DATA-s1234-v1` |
| E8 | PointNet++ | xyz+3 geom | PINN | `VF-PNPP-XYZG-PINN-s1234-v1` |

严格配对是 E1↔E2、E3↔E4、E5↔E6、E7↔E8。配对除 physics enabled 和三个
physics loss 权重外必须完全一致；静态 preflight 会比较 resolved protocol、参数量和
初始化 state SHA256。

### 2.1 SAME5K-E7500 v2 矩阵

第二轮保持相同四对架构/输入/模式，只把监督采样改为每例每 epoch 随机 5000 点且
`support_idx == query_idx`，固定训练 7500 epoch。实验 ID 在 v1 基础上增加
`SAME5K-E7500-s1234-v2`；配置真源为
`wss_pinn/configs/volume_uvwp_peak_same5k_e7500_v2/`。本轮仍为 train138 / val0 /
test35，不使用 test35 调度、早停或 checkpoint 选择。

## 3. 架构锚点

架构由
`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`
中的历史结果选择：

- PointNet：P2V 宽度与 global random5000 设计；
- PointNet++：纯 D2 `c125-k128` 与 random5000 设计；
- 只继承结构，不加载历史权重。

对应父配置、SHA256 和工作簿 SHA256 已冻结在
`wss_pinn/configs/volume_uvwp_peak_v1/matrix.json`，preflight 会检查来源漂移。

PINN 所需的适配只发生在连续查询路径：平滑 SiLU、query-local LayerNorm、独立查询
坐标和可微逆距离 3NN 权重。PointNet++ 的 support encoder 可以保留离散 FPS/kNN；
PDE 导数只要求给定 support 条件下，查询函数对 query coordinate 可一、二阶求导。

2026-08-04 的零重训审计表明，上述“连续查询”前提在 v2 SAME 点上并不成立：当
`query_pos` 与某个 support 点精确重合时，`1/d²` 权重由该点支配，而距离平方在
`d=0` 处对坐标的一阶导数为零，插值 latent 对查询坐标近似常量。网络输出仍有显式
`query_pos` 支路，所以不是完全零导数；但 PointNet++ latent 的主要空间变化被屏蔽，
导致 SAME 点 PDE residual 被系统性压低。独立 query 才能检验 support-conditioned
连续场的真实坐标导数。

## 4. 数据合同与合规 Gate

### 4.1 固定 split

- 文件：`wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_q2v_pool2025_train138_test35_exclude_SHI_YUN_XI_v1.json`
- 数量：train138 / test35；test 仍是 reused development screen。
- SHA256：`964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b`。

### 4.2 schema v2 sidecar

每例保存完整峰值体域：注册坐标、3 个几何特征、core/near-wall 标记、
`interior_is_wall`、旋转后的速度、相对压力、独立壁面坐标和壁面几何。

关键修正：

1. 原始速度向量按 row-vector 约定执行
   `velocity_registered = velocity_raw @ transform_rotation`；
2. 三个 bundle 含精确壁面坐标重复行，schema v2 保存 mask，并从 support/query/PDE、
   压力 gauge 和 train-only 统计中排除；
3. 原始曲率尖峰先做 `signed_log1p`，再由 train138 严格体域估计 p99 clip；
4. 压力按每例严格体域固定均值中心化，单位 Pa；绝对 gauge 只作诊断。

### 4.3 首轮 v1 训练采样

- 每例每 epoch：5000 support + 独立 5000 query；
- PINN：query 中均匀取 512 个 PDE 点；
- no-slip：独立壁面云均匀取 1024 点；
- 所有 sampling seed 可复现，配对实验完全相同。

训练数据放行条件：173/173 case manifest、文件 hash、shape/dtype、有限性、注册速度、
压力 gauge、壁面重复行、split role、train-only stats membership/hash 全通过；deep-source
审计必须重新对照父 bundle 与原始 peak ASCII。

### 4.4 第二轮 v2 SAME5K 训练采样

- 每例每 epoch 从严格体域均匀随机抽取 5000 点；support 与监督 query 使用同一索引；
- support 只含坐标/几何，不含 `u,v,w,p` 标签，因此 SAME 不构成标签输入泄露；
- PINN 从这 5000 query 中均匀取 512 个 PDE 点；no-slip 仍使用独立壁面云 1024 点；
- 每 epoch 重新采样；配对实验共享 seed、病例顺序、初始化和采样协议；
- SAME5K 训练改变了任务口径，结果不能把增益单独归因于 epoch 或 SAME，也不能与
  v1 full-volume 主指标无注释混表。

## 5. 四通道监督与压力含义

data-only 和 PINN 都监督四个输出：

\[
\mathcal L_{data}=\mathcal L_u+\mathcal L_v+\mathcal L_w+\mathcal L_p.
\]

这里“压力监督”指网络预测的第四通道直接与 CFD 相对压力真值比较，并不是只靠动量
方程间接推压力。PINN 中压力同时承担两个角色：

- `L_p` 提供有标签的第四通道监督；
- `grad(p)` 进入三个动量方程残差。

由于 CFD 压力允许整体加常数，标签先用病例固定 gauge 去除常数自由度；禁止按每个
mini-batch 再中心化，否则模型看到的目标会随抽样漂移。

## 6. 物理损失

PINN 从随机初始化的第一个 step 同时优化：

\[
\mathcal L=\mathcal L_{data}
+\lambda_c\mathcal L_{continuity}
+\lambda_m(\mathcal L_{mx}+\mathcal L_{my}+\mathcal L_{mz})
+\lambda_w\mathcal L_{no-slip}.
\]

物理方程为：

\[
\nabla\cdot\mathbf u=0,
\]

\[
\rho(\mathbf u\cdot\nabla)\mathbf u+\nabla p-
\nabla\cdot(2\mu(\dot\gamma)\mathbf D)=0,
\]

\[
\mathbf u|_{\Gamma_w}=0.
\]

采用 Carreau–Yasuda、`rho=1060 kg/m^3`、`U0=1 m/s` 和病例已知几何尺度。
momentum 使用完整变黏度应力散度，因此包含黏度随剪切率变化引起的空间导数。

## 7. 非热启动协议

PINN 不加载 data-only checkpoint。这样比较的是“同一初始化下，加入物理损失改变了
什么”，不会把 data-only 已学到的解混入 PINN 优势。

允许的 `resume` 仅指同一个 run 因中断后继续：run 目录、实验 ID、模式、架构、输入、
初始化哈希必须相同；代码会拒绝跨 run checkpoint。恢复时不覆盖最初的
`checkpoints/initialization.pt`，配置的 `epochs` 被解释为总 epoch，不会在恢复后再额外
跑一整轮相同预算。

## 8. 日志、收敛与评估

### 8.1 训练日志

- `training_progress.jsonl`、`history.csv`：逐 step raw/weighted `u/v/w/p` data
  loss、continuity、momentum-x/y/z、no-slip、physics total、total、梯度范数、学习率；
- `epoch_progress.jsonl`：上述 loss 的逐 epoch 均值；
- PINN 额外记录 continuity 的无量纲/SI RMS、三分量 momentum RMS、对流/压力/黏性项
  RMS、剪切率、黏度和壁面速度。

判读时至少同时画 `data_total`、`physics_total`、`total`、continuity、三个 momentum
分量和 no-slip。`total` 下降但某个 physics 分量发散，不算物理收敛。

### 8.2 评估

三个 checkpoint 都评估：

- 物理单位 `u/v/w/speed/p` 的 R²、MAE、RMSE；
- near-wall/core 分区指标；
- 壁面速度 RMS/p95/max；
- continuity RMS 和 momentum RMS；
- 预测/真值压力均值以及两者 gauge offset。

首轮结果统一使用 `evaluation_best_total.json` 的 test35 case-balanced 指标，与此前配对
汇总口径一致。这里的 `best_total` 只是在各自训练目标内部选择 checkpoint：data-only
的 total 等于 data loss，PINN 的 total 包含 physics loss，两个训练标量本身不能直接
跨模式排名。`best_data` 与 `last` 已同时完成评估，后续可作 checkpoint 敏感性分析。
test35 的标签是 reused development screen，不是新的独立确认集。

对 PointNet++ 的 SAME5K 评估还必须附加一个限制：回归指标与压力 gauge 仍是有效的
同点监督指标，但 SAME support 上的 continuity/momentum 只代表该退化查询位置，不能
再解释为完整连续体域上的 PDE 满足度。独立体域 query residual 见第 10.6 节。

## 9. 首轮八实验结果（2026-08-02）

### 9.1 完整性

- 数据构建 `11122`、deep-source audit `11123`、launcher `11124`、GPU preflight
  `11127` 与训练数组 `11128_[0-7]` 均完成；`11125/11126` 是正式训练前因 CuBLAS
  确定性环境缺失而主动取消的修正记录。
- 8/8 实验均执行 400 epoch；train138、`batch_cases=2`，即每 epoch 69 step、每臂
  27,600 optimizer step。
- 每个 run 的 checkpoint、配置、逐 step/epoch 日志和三份评估均完整；共 24 份
  evaluation JSON。四个 data-only/PINN 配对的初始化 state SHA256 分别一致，且全部
  `warm_start=false`。
- 所有 JSONL 有限，无 NaN/Inf、OOM、traceback 或异常退出。Slurm stderr 只有
  NumPy 非可写数组转 tensor 的非致命 warning，不影响本轮数值与 checkpoint 哈希。

### 9.2 `best_total` / test35 case-balanced 指标

| 架构/输入 | 模式 | u R² | v R² | w R² | speed R² | p R² | continuity RMS | momentum RMS (Pa/m) | wall RMS (m/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PointNet / `xyz` | data-only | -0.0524 | 0.1995 | -0.1048 | 0.0609 | -0.2669 | 1.9782 | 11368.8609 | 0.3217 |
| PointNet / `xyz` | PINN | 0.0070 | 0.1722 | -0.2151 | -0.0521 | 0.2566 | 0.4693 | 2234.8839 | 0.3048 |
| PointNet / `xyz+geom` | data-only | 0.1081 | 0.2989 | 0.0827 | 0.2092 | 0.6288 | 2.8640 | 14168.0822 | 0.3360 |
| PointNet / `xyz+geom` | PINN | 0.2421 | 0.3225 | 0.1127 | 0.2183 | 0.6569 | 0.4104 | 1988.6599 | 0.2984 |
| PointNet++ / `xyz` | data-only | 0.3955 | 0.3477 | 0.0751 | 0.0351 | -0.0898 | 1.2498 | 6118.0351 | 0.3467 |
| PointNet++ / `xyz` | PINN | 0.3687 | 0.3660 | 0.0520 | 0.0490 | 0.4492 | 0.3102 | 936.9650 | 0.2683 |
| PointNet++ / `xyz+geom` | data-only | 0.4601 | 0.4528 | 0.2270 | 0.2040 | 0.3453 | 1.0611 | 5064.4145 | 0.2975 |
| PointNet++ / `xyz+geom` | PINN | 0.3836 | 0.3947 | 0.1468 | 0.1167 | 0.4948 | 0.2927 | 909.3047 | 0.2473 |

### 9.3 配对变化

| 配对 | Δu R² | Δv R² | Δw R² | Δspeed R² | Δp R² | continuity 降幅 | momentum 降幅 | wall RMS 降幅 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PointNet / xyz | +0.0594 | -0.0273 | -0.1103 | -0.1130 | +0.5234 | 76.3% | 80.3% | 5.3% |
| PointNet / xyz+geom | +0.1339 | +0.0236 | +0.0300 | +0.0091 | +0.0281 | 85.7% | 86.0% | 11.2% |
| PointNet++ / xyz | -0.0268 | +0.0183 | -0.0231 | +0.0139 | +0.5390 | 75.2% | 84.7% | 22.6% |
| PointNet++ / xyz+geom | -0.0766 | -0.0581 | -0.0802 | -0.0874 | +0.1495 | 72.4% | 82.0% | 16.9% |

![八实验配对指标](../../../outputs/wss_pinn/volume_uvwp_peak_v1/summary/paired_metrics_best_total.png)

配对结论：

- continuity RMS 下降 72.4%–85.7%，momentum RMS 下降 80.3%–86.0%，壁面速度
  RMS 下降 5.3%–22.6%；四对 pressure R² 全部提高。
- PointNet + `xyz+geom` 是唯一加入 PINN 后 `u/v/w/speed/p` 五项全部提高的配对。
- 另外三对均有部分速度指标退化：PointNet / xyz 的 `v/w/speed` 下降，
  PointNet++ / xyz 的 `u/w` 下降，PointNet++ / xyz+geom 的四个速度指标均下降。
- 因此 PINN 的可靠结论是“物理一致性显著改善、压力普遍改善、速度收益依赖架构和
  几何输入”，不能写成全面优于 data-only。

### 9.4 收敛曲线判读

![八实验收敛曲线](../../../outputs/wss_pinn/volume_uvwp_peak_v1/summary/convergence_curves.png)

- 400 epoch 内所有 data loss 都显著下降；最后 50 epoch 相比前一个 50 epoch 仍下降
  约 0.59%–1.29%，说明还没有完全平台。
- PointNet 两个 PINN 的 late-stage physics total 仍小幅下降约 2.0%–2.7%，但 no-slip
  同期轻微上升约 0.8%–1.0%，不能只看 `physics_total`。
- PointNet++ 两个 PINN 的 late-stage physics total 基本平台（变化约 0.01%–0.03%），
  momentum/no-slip 略升。其训练初期近零场天然产生较低 residual，随后学习真实速度场
  会打破这种“伪低 physics loss”，不能把起点低或单一 total 最小当作已收敛。
- 最高 speed R² 仍只有约 0.218，首轮尚不足以选择最终模型或宣称速度场已充分收敛。

### 9.5 机器可读结果与复现

结果目录：`outputs/wss_pinn/volume_uvwp_peak_v1/summary/`

- `paired_metrics_best_total.csv/json`
- `convergence_summary.json`
- `paired_metrics_best_total.png/svg`
- `convergence_curves.png/svg`
- `manifest.json`

复现命令：

```bash
/public/newhome/cy/.conda/envs/GNN/bin/python \
  -m wss_pinn.tools.summarize_results
```

## 10. 第二轮 SAME5K-E7500 v2

> 状态：**static preflight passed / GPU preflight `11137` completed 8/8 /
> training array `11138_[0-7%4]` completed 8/8 / results audited**。

### 10.1 完整性与判读口径

- 固定 `epochs=7500`，不划 inner validation，不启用早停；train138 / val0 / test35。
- 每 epoch 69 step，总预算 517,500 optimizer step；data-only 与 PINN 使用相同固定预算。
- scheduler 保持现有 cosine 训练实现，只把总周期重参数化为 7500；因此新 run 的
  epoch400 学习率与旧 400-epoch v1 不同，不能把 epoch400 checkpoint 当作严格 SAME
  单变量对照。
- 固定保存 epoch 400 / 1000 / 2500 / 5000 / 7500；配对主读数预注册为 `last@7500`，
  `best_data/best_total` 只作 train-selected checkpoint 敏感性诊断，禁止按 test35 选 epoch。
- 训练结束自动对 `best_data/best_total/last` 生成两套评估：
  `evaluation_same5k_*` 是固定同点 5000 主协议，`evaluation_fullvolume_*` 是固定 5000
  support 解码完整严格体域的连续场泛化协议。
- 每套 evaluation 保留逐病例 `u/v/w/speed/p` 与 near-wall/core 的 R²、MAE、RMSE，
  以及 case-balanced 汇总、continuity、三分量 momentum、wall RMS/P95/max 和压力
  gauge 诊断。
- Slurm `#SBATCH --time=0`；GPU 分区实查 `MaxTime=UNLIMITED`。数组最多 4 卡并发，
  GPU preflight 通过后才释放正式训练。
- 新输出只写 `outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/`，不覆盖 v1。
- 8/8 run 均为 `completed`，每臂 7500 epoch / 517,500 step，逐臂
  `epoch_progress.jsonl` 均为 7500 行；共 48 份 SAME5K/full-volume ×
  `best_data/best_total/last` evaluation JSON 完整。
- 最低 train data/total loss 出现在 epoch 6815–7487；最后 50 epoch 相比此前 50
  epoch 的 data loss 变化仅约 `-0.07%` 至 `+0.06%`。同一 run 三个 checkpoint 的
  SAME5K speed R² 极差为 `0.0006–0.0230`，说明本轮结论不依赖偶然 checkpoint，
  训练已基本平台。
- 老师汇报版 train data / physics 收敛图（本轮 val0，仅 train）：
  [`summary/loss_convergence/`](../../../outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/summary/loss_convergence/)；
  主文件 `teacher_loss_convergence.pdf`，复现
  `python -m wss_pinn.tools.plot_v2_loss_convergence`。
- 早期 30 分钟健康快照仍保留为运行过程证据；最终判读以 8/8 完训后的
  `last@7500` 为主，不以 test35 反选 checkpoint。

### 10.2 SAME5K 主协议：`last@7500` / test35 case-balanced

| 架构/输入 | 模式 | u R² | v R² | w R² | speed R² | speed MAE (m/s) | speed RMSE (m/s) | core speed R² | near-wall speed R² | p R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PointNet / `xyz` | data-only | -2.3062 | -2.8884 | -6.7245 | -1.5266 | 0.2793 | 0.4211 | -1.3928 | -11.5665 | -0.1247 |
| PointNet / `xyz` | PINN | -0.2148 | 0.0222 | -0.5382 | -0.1331 | 0.2216 | 0.3177 | -0.1030 | -2.4695 | 0.4096 |
| PointNet / `xyz+geom` | data-only | -1.3572 | -1.8127 | -6.1607 | -0.9661 | 0.2597 | 0.3918 | -0.8901 | -7.4225 | 0.5075 |
| PointNet / `xyz+geom` | PINN | -0.0678 | 0.1835 | -0.2493 | 0.0923 | 0.2095 | 0.2950 | 0.0991 | -0.9891 | 0.6511 |
| PointNet++ / `xyz` | data-only | 0.4023 | 0.3297 | 0.1381 | 0.0870 | 0.1825 | 0.2766 | 0.1295 | -3.0818 | -0.1164 |
| PointNet++ / `xyz` | PINN | 0.4270 | 0.4051 | 0.2008 | 0.1575 | 0.1807 | 0.2730 | 0.1883 | -2.1265 | -0.0412 |
| PointNet++ / `xyz+geom` | data-only | 0.4705 | 0.4386 | 0.2259 | 0.2063 | 0.1773 | 0.2672 | 0.2371 | -2.0509 | 0.1729 |
| **PointNet++ / `xyz+geom`** | **PINN** | **0.4811** | **0.4498** | **0.2557** | **0.2380** | **0.1764** | **0.2668** | **0.2588** | **-1.4770** | **0.3225** |

按主协议，当前速度锚点是 **PointNet++ + `xyz+geom` + PINN**。其 35 例 speed R²
均值/中位数为 `0.2380/0.3145`，30/35 病例为正；但 near-wall speed R² 仍为
`-1.4770`，说明全体域相关性尚不能代表近壁速度可靠。`AAA/ruputer/YANG_BAO_KUI`
在固定采样和完整体域中均无 near-wall 标记点，因此 near-wall case-balanced 指标实际
基于其余 34 例。

### 10.3 完整体域连续场协议：固定 5k support → full volume

| 架构/输入 | 模式 | speed R² | speed MAE (m/s) | speed RMSE (m/s) | core speed R² | near-wall speed R² | p R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PointNet / `xyz` | data-only | -1.5439 | 0.2791 | 0.4206 | -1.4150 | -10.6626 | -0.1391 |
| PointNet / `xyz` | PINN | -0.1253 | 0.2199 | 0.3159 | -0.0965 | -2.3181 | 0.4017 |
| PointNet / `xyz+geom` | data-only | -0.9554 | 0.2585 | 0.3908 | -0.8835 | -6.3722 | 0.4982 |
| PointNet / `xyz+geom` | PINN | 0.0940 | 0.2087 | 0.2941 | 0.1006 | -0.9431 | 0.6521 |
| PointNet++ / `xyz` | data-only | 0.0018 | 0.1938 | 0.2881 | 0.0492 | -3.3713 | -0.1265 |
| PointNet++ / `xyz` | PINN | 0.0798 | 0.1921 | 0.2841 | 0.1129 | -2.3521 | -0.0501 |
| PointNet++ / `xyz+geom` | data-only | 0.1506 | **0.1859** | **0.2744** | 0.1839 | -2.1994 | 0.1570 |
| **PointNet++ / `xyz+geom`** | **PINN** | **0.1820** | 0.1863 | 0.2748 | **0.2036** | **-1.5754** | **0.3166** |

完整体域协议下最佳 speed R² 仍来自 PointNet++ + `xyz+geom` + PINN，但从同点
`0.2380` 降至 `0.1820`。其 MAE/RMSE 与 data-only 基本持平且略差
（`+0.0004/+0.0004 m/s`），说明 PINN 的均值 R² 增益主要来自病例间分布改善，
并不是逐点绝对误差的一致下降。

### 10.4 PINN 配对变化与稳定性

| 配对 | Δspeed R² | Δp R² | continuity 降幅 | momentum 降幅 | wall RMS 降幅 | speed R² 改善病例 | speed MAE 改善病例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PointNet / `xyz` | +1.3936 | +0.5342 | 84.2% | 82.3% | 40.1% | 30/35 | 32/35 |
| PointNet / `xyz+geom` | +1.0585 | +0.1436 | 90.8% | 90.3% | 37.2% | 32/35 | 33/35 |
| PointNet++ / `xyz` | +0.0705 | +0.0752 | 98.7% | 97.8% | 9.1% | 19/35 | 18/35 |
| PointNet++ / `xyz+geom` | +0.0317 | +0.1496 | 98.8% | 98.2% | 10.1% | 18/35 | 16/35 |

- 与首轮 v1 不同，V2 四个配对的聚合 `u/v/w/speed/p` R² 全部提高；物理一致性
  和压力改善仍是最稳定的 PINN 收益。
- PointNet 两个 data-only 在 V2 上出现严重跨病例泛化失败，PINN 的大幅增益主要是
  “挽救负 R²”，不能据此估计在稳定架构上的通用收益。PointNet++ 两对更可信的
  speed R² 增益只有 `+0.0705/+0.0317`，且逐病例改善约一半，尚不足以宣称 PINN
  稳定优于 data-only。
- PointNet++ + `xyz+geom` 的聚合增益主要来自 ILO：SAME5K speed R² 从
  `-0.0453` 提高到 `0.0797`；AG 为 `0.3295→0.3281`，AAA 为
  `0.2441→0.2446`，基本持平。当前 PINN 速度收益具有明显域依赖。
- 几何特征对四种架构/模式均有正贡献：加入 `xyz+geom` 后 speed R² 分别提高
  `+0.5605/+0.2254/+0.1193/+0.0805`。PointNet++ 在所有同输入/同模式比较中均优于
  PointNet，是当前更可靠的体域速度架构。
- 最佳速度模型的 5 个负 R² 病例在 SAME5K 与 full-volume 中一致：
  `ILO/SUN_DONG_XIN-0/before`、`ILO/LI_YOU_ZHI-0/before`、
  `AAA/ruputer/KANG_YONG`、`AG/slow/HE_SHU_ZHEN`、
  `AAA/unruputer/SUN_SHU_MING`。其中 `SUN_DONG_XIN` 是最主要均值拖累项。
- 最佳压力模型不是最佳速度模型：PointNet + `xyz+geom` + PINN 的 p R² 为
  `0.6511/0.6521`（SAME5K/full-volume），说明速度、压力和物理 residual 之间仍有
  多目标取舍。

### 10.5 当前结论与下一步

1. 冻结 **PointNet++ + `xyz+geom` + PINN / last@7500** 为 V2 速度主锚点；
   PointNet + `xyz+geom` + PINN 仅作为压力优先对照。
2. 7500 epoch 已基本平台，train-selected checkpoint 也不能改变结论；零重训诊断已
   确认 SAME-IDW 导数退化与准稳态方程缺项，下一轮若获准训练，首要变量应是独立
   physics query / 连续坐标场头与可观测边界条件，而不是继续延长 epoch。
3. 当前 near-wall speed R² 全部为负，说明 overall speed 指标掩盖了边界层区域的
   严重失真。下一轮把 near-wall `u/v/w` 作为场重建困难区单独优化和验收；冻结 WSS
   只在主模型选定后检查误差放大，不反向参与选模。
4. test35 是 reused development screen，且本轮只有 seed1234。任何“PINN 稳定提高
   速度”的论文结论仍需独立确认集或多 seed 复核。

### 10.6 零重训验证 1：SAME 点 residual 是否虚低

审计固定使用当前最佳 checkpoint，不更新任何权重；每例仍按正式 evaluator 抽取
5000 support，再比较 512 个精确 support 点、512 个排除 support 的独立严格体域点，
以及在同一批 support 坐标上施加 `1e-4/1e-3` 归一化长度微扰。主结论以 35 例
`independent_volume` 为准；jitter 只作导数探针，因为微扰点可能离开血管域。

| 查询条件 | continuity RMS | momentum RMS (Pa/m) | 速度梯度 Frobenius RMS | 相对 exact 的动量倍率 |
| --- | ---: | ---: | ---: | ---: |
| 精确 support | 0.00718 | 22.79 | 0.03185 | 1.0× |
| 独立严格体域 | 3.2065 | 7330.05 | 5.9748 | **331.2×** |
| support + `1e-4` | 0.3222 | 2673.83 | 0.6220 | **119.9×** |
| support + `1e-3` | 1.8582 | 4216.28 | 3.3890 | **192.0×** |

35 例逐病例倍率没有反例：独立点 continuity 为 exact 的 `255–731×`，动量为
`190–546×`，速度梯度为 `113–300×`。尤其 `1e-4` 微扰时，预测速度的 matched RMS
变化平均只有 `1.84e-5 m/s`，但动量 residual 已放大约 `120×`。因此问题不是场值在
极小位移后突然失真，而是 exact support 处的逆距离插值导数发生退化。

这会修正此前的物理解读：V2 表中的“PINN momentum 降幅 98.2%”只说明在 SAME
退化位置上，PINN 把显式 query head/no-slip 等训练目标压低；不能证明模型在独立
体域坐标上满足动量方程。速度 R²、MAE、RMSE 与压力 R² 不依赖坐标二阶导数，因此
这些监督指标和“速度仍不够准”的结论保持不变。

### 10.7 零重训验证 2：CFD 真值、`du/dt` 与老师论文对照

#### CFD 真值 residual

从 AG/AAA/ILO 各预注册 1 个 test 病例，在严格 core 体域各固定抽 256 点；使用物理
坐标中的加权局部二次拟合恢复速度一/二阶导数与压力梯度，并以 `k=48/96` 检查邻域
敏感性。三例 Fluent journal 均明确 `time-step=0.005 s`；峰值前后导出步为
`1160/1164`，所以中心差分分母是 `0.02 s`。三时相坐标均为 direct-row 对齐，最大
归一化误差 `≤1.43e-7`。

| 邻域 | continuity RMS (s⁻¹) | 准稳态 residual (Pa/m) | 加 `rho·du/dt` 后 (Pa/m) | 瞬态/准稳态 | 相对 residual：准稳态→瞬态 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k=48` | 10.80 | 2863 | 2721 | 0.939 | 0.253→0.236 |
| `k=96` | 3.12 | 1737 | 1494 | **0.831** | **0.175→0.142** |

`k=96` 下三例加入时间项后分别下降约 `6.5%/19.2%/24.9%`；说明 peak 瞬间并非严格
稳态，当前训练方程遗漏 `du/dt` 是真实模型误差，但它只解释部分 residual，不能单独
解释低速度 R²。另一方面，虽然两组拟合均为 100% 满秩、`k=96` 的 condition p95
约 33，点级 residual 的 `k48/k96` 相对 L2 差异仍约 `1.0–2.2`。这说明从导出的
非结构化 cell-center 点反推二阶导数对邻域选择敏感；当前结果支持“加入 `du/dt`
总体改善、且 `k=96` 三例一致改善方程闭合”，不支持把某个绝对 Pa/m 数值当成
Fluent 原生离散残差。

#### 为什么老师论文的速度看起来更准

| 设计维度 | Liao et al. 2025 | 当前 V2 | 对当前问题的含义 |
| --- | --- | --- | --- |
| 查询函数 | PointNet 直接在 `(x,y,z)` 上输出 `u,v,w,p` | PointNet++ support latent 经 3NN `1/d²` 插值后与 query 坐标解码 | 论文在同点算 PDE 不会触发 support→query 插值奇点；V2 会 |
| CFD/方程 | 严格稳态、常黏度 Newtonian | 脉动 CFD 的 peak，Carreau–Yasuda 准稳态训练 | V2 缺少已证实非零的 `du/dt` |
| 边界条件可辨识性 | 所有病例入口调到 WSS=1.5 Pa，出口按面积平方律 | 患者特异入口/RCR，但网络不输入流量、入口波形、出口压力/RCR | 论文更接近“几何→标准化流场”；V2 是缺少关键条件的多解映射 |
| loss 权重 | `omega=loss_data/loss_phy`，每 10 epoch 更新 | data/continuity/momentum/no-slip 固定 1:1 权重 | V2 最后 epoch 的 physics 中 no-slip 占 **99.85%**，内部 PDE 实际权重很弱 |
| 点云 | 0.05 mm voxel + 每例 10000 均匀点 | 每例 5000 uniform support，近壁点比例低 | V2 对近壁速度与梯度的信息预算更少 |
| 评估 | 速度/压力 NMAE；FR/PD 是病例级 R² | case-balanced 点级 speed R²、MAE/RMSE与近壁 R² | 原文没有报告可直接比较的点级速度 R² |
| 验证 | 40/11，四组 Monte Carlo 划分 | reused test35、单 seed1234 | 原文对划分方差检查更完整 |

指标差异本身足以解释一部分“论文很准、我们 R² 很低”的观感。按老师源码快照使用的
test 全局 `u/v/w` range 作为 NMAE 分母重算，当前 V2 竟可得到 SAME5K
`2.18±0.70%`、full-volume `2.15±0.67%`，数值上还低于论文 `7.79±2.14%`；但同一个
模型的 speed R² 只有 `0.2380/0.1820`。这证明低 NMAE 可以与低场分布 R² 同时出现，
不能从论文 NMAE 或 FR/PD 病例级 R² 推断其逐点速度空间结构一定比 V2 更准确。

老师论文最值得直接借鉴的不是照搬 15000 epoch，而是：使用对坐标直接、平滑可导的
场函数；让训练 PDE 与生成标签的稳态/瞬态假设一致；显式标准化或输入 inlet/outlet
条件；动态平衡 data/PDE/BC；并在多划分下同时报告 NMAE 与点级 R²/近壁指标。

### 10.8 代码维护性精简（2026-08-05）

本次只整理执行代码，不改变 V1/V2/V3 的科学合同、已完成结果或配置提交方式：

- `ExperimentConfig` 与 `build_model` 成为推荐入口，旧名称保留兼容；route 的 mode、
  architecture、activation 和冻结 split 由一张合同表校验。
- V1/V2 与 V3 共用四通道数据项和 PDE residual 计算；route 只负责各自已注册的
  data/BC/PDE 组合，`compute_losses` 直接返回 loss 字典。
- 训练 step 日志删除没有下游消费者的剪切率/黏度等重复诊断；这些物理量继续由
  checkpoint 评估统一计算。`training_progress.jsonl`、`history.csv`、
  `epoch_progress.jsonl` 和 checkpoint 命名保持不变。
- `evaluate.py` 的病例主循环拆成分块预测、区域指标、V3 边界和 PDE 四个直接 helper，
  评估 JSON schema 与历史汇总字段保持不变。
- 标准 `cluster.submit_matrix` 默认以 `matrix.json` 为唯一实验列表真源，并在输出目录
  生成 Slurm 使用的 resolved list；旧 `--config-list` 仅作为历史兼容入口。
- 测试从 7 个文件收敛为 4 个核心文件、18 项合同测试；完整数据 Gate、矩阵静态
  preflight 和逐配置 GPU dry-run 继续作为正式提交前验证，因此没有削弱配置驱动实验。

当前最终验证：26/26 核心测试通过；field-v4 原矩阵/补种子矩阵静态 preflight、CPU
smoke、master/node04 GPU dry-run 和全部正式 Slurm 训练/评估均为 `pass/completed`。

## 11. 真源与归档

- 代码说明：`wss_pinn/README.md`
- 核心代码诊断与下一轮设计建议（2026-08-05，第一性原理/对抗性修订版）：
  [核心代码诊断与下一轮设计建议_2026-08-05.md](./核心代码诊断与下一轮设计建议_2026-08-05.md)
- Stage 0–1 已完成执行合同：
  [冻结诊断后 Stage 0–1 提示词](./WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1.md)
- field-v4 决策合同与结果：
  `wss_pinn/configs/volume_uvwp_peak_field_v4/decision_contract.json`、
  `outputs/wss_pinn/volume_uvwp_peak_field_v4/stage1_multiseed_and_promotion_report.json`
- 历史六臂预注册：
  [准稳态平滑场六臂实验提示词（已完成）](./_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)
- 归档索引：[`_archive/README.md`](./_archive/README.md)
- 矩阵：`wss_pinn/configs/volume_uvwp_peak_v1/matrix.json`
- SAME5K-E7500 v2 矩阵：
  `wss_pinn/configs/volume_uvwp_peak_same5k_e7500_v2/matrix.json`
- 首轮结果：`outputs/wss_pinn/volume_uvwp_peak_v1/summary/manifest.json`
- 第二轮提交记录：`outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/submission.json`
- 第二轮 30 分钟健康快照：
  `outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/monitor_30min.json`
- 第二轮 8 个 run 与 48 份评估：
  `outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/runs/`
- 零重训诊断工具：`wss_pinn/tools/diagnose_v2_physics.py`
- 零重训机器可读结果：
  `outputs/wss_pinn/audits/volume_uvwp_peak_same5k_e7500_v2_residual_diagnosis/`
- 老师论文 PDF：`docs/paper_idea/1-s2.0-S0952197625020421-main.pdf`
- 论文复现与当前代码对照：
  `docs/paper_reproduction/papers/hemodynamics_pointcloud_pinn/README.md`
- 详细推进记录：`docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`
- 项目级摘要：`docs/02-推进与变更/代码修改与实验推进记录.md`
- 历史 WSS-target 文档：`_archive/wss_target_v1_20260730/`
- 历史 WSS-target 源码：
  [`wss_pinn/archive/wss_target_v1_20260730/`](../../../wss_pinn/archive/wss_target_v1_20260730/README.md)
- 历史源码快照：Git commit
  `bca002025d40b290d171570e6470f484fd4feec7`
