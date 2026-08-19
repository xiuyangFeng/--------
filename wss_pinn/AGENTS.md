# wss_pinn — 体域 `u,v,w,p` PINN 指令

本目录当前活动主线是已实现并提交受 Gate 保护训练链的 `volume_uvwp_bc_rcr_v4`，设计真源为
`docs/02-推进与变更/WSS_PINN/WSS_PINN_V4大重构设计方案_2026-08-08.md`。
已完成的 `volume_uvwp_peak_field_v4` Stage 0–1 与
`volume_uvwp_peak_qs_smooth_v3` 六臂均为冻结历史结果，不得覆盖或冒充新 V4。
`volume_uvwp_peak_v1` 与 SAME5K-E7500 v2 保留为历史对照；旧“直接 WSS 输出 + 阶梯
F0/F1/F2”路线已归档。

## 隔离边界

- `data_new/`、`data_wss_min/`、`pipeline_wss_min/`、`training_wss_min/` 是只读上游。
- 新派生数据只写 `data_wss_pinn/volume_uvwp_peak_v1_train138_test35/`。
- V3 新派生边界只写
  `data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/`。
- 已完成旧 field-v4 的派生合同只写
  `data_wss_pinn/volume_uvwp_peak_field_v4_train123_val15/`；不得复制或覆盖 V3 sidecar。
- 新 V4 派生数据只允许写入
  `data_wss_pinn/volume_uvwp_bc_rcr_v4_train138_test35/`，配置只写
  `wss_pinn/configs/volume_uvwp_bc_rcr_v4/`，输出只写
  `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/`；不得复用旧 field-v4 写路径。
- 新结果只写 `outputs/wss_pinn/volume_uvwp_peak_v1/`、
  `outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/`、
  `outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/`、
  `outputs/wss_pinn/volume_uvwp_peak_field_v4/`、
  `outputs/wss_pinn/volume_uvwp_bc_rcr_v4/` 与对应 audit 目录；历史结果只读。
- 活动训练、评估、数据、模型、物理、工具和 Slurm 入口直接位于 `wss_pinn/`
  根层；活动配置位于 `wss_pinn/configs/volume_uvwp_peak_v1/`、
  `wss_pinn/configs/volume_uvwp_peak_same5k_e7500_v2/` 和
  `wss_pinn/configs/volume_uvwp_peak_qs_smooth_v3/`；已完成旧 field-v4 配置位于
  `wss_pinn/configs/volume_uvwp_peak_field_v4/`；Stage 1 确认种子只写
  `wss_pinn/configs/volume_uvwp_peak_field_v4_multiseed_top2/`。测试只保留配置、采样、
  模型可微性和物理公式四类核心合同，位于 `wss_pinn/tests/test_volume_*.py`；
  真实数据与 GPU 冒烟由 preflight 承担，不再复制大型 synthetic fixture。
- 旧直接 WSS/F0–F2 源码与配置冻结在
  `wss_pinn/archive/wss_target_v1_20260730/`；`data/raw_io.py`、
  `data/alignment.py` 和 `utils.py` 因仍被当前路线复用而保留在活动区。
- 不删除旧数据或旧输出；历史 Git 快照仍为
  `bca002025d40b290d171570e6470f484fd4feec7`。

## 冻结科学合同

### 新 V4 v1.2 实现与运行边界

- split 为 train138/test35，不设 val；原 val15 合并回训练集。test35 是
  development-exposed screen，禁止用于训练停止、checkpoint、动态权重、采样或架构选择。
- 输入固定为 `xyz+geom` 加显式病例条件：`A_mesh`、`Q_actual_peak`、四出口面积与每出口
  `R1/R2/C`；禁止输入旧 V3 monitor pressure、CFD outlet-face pressure 或 UDF 求解后
  `P_n`。
- 面积/RCR/`Q_actual_peak` 显式进入网络；no-slip、入口 `Q_nom/A_udf` Dirichlet BC 与
  瞬态质量流量 RCR 一致性进入 BC loss，`lambda_bc=1`；continuity/momentum 属于 PDE 组。
- 架构固定比较 PointNet / PointNet++；每种、每个时间模式均保留 DATA、DATA+BC、
  BC+PDE-fixed (`lambda_pde=1`) 与 BC+PDE-EMA 四臂，共 16 臂；正式 seeds 为
  `[1234,2345,3456]`，合计 48 run。四臂共享初始化结构与采样协议，不加载旧 checkpoint。
- 物理通过配置区分 `steady_peak/quasi_steady` 与
  `transient_81/transient_autograd`；后者必须显式输入时间和 `Q_in(t)`，由 autograd 计算
  `du/dt`，不得用峰值单帧伪装瞬态。
- data `u/v/w/p`、continuity、momentum x/y/z、no-slip、raw/weighted、动态权重、梯度
  范数和冲突余弦必须单独记录。三轴 momentum 可分别监控，但动态平衡优先作为向量组。
- 默认动态权重为老师建议的 detached EMA ratio：
  `lambda_phy = clip(alpha * EMA(L_data)/(EMA(L_phy)+eps))`，默认每 50 step 更新并做权重
  EMA 平滑。ratio/EMA/weight 必须在计算图外更新；不 detach 会使总损失代数退化为
  `2*L_data`，属于硬失败。ReLoBRaLo/GradNorm 不作为首轮默认 controller。
- 无 val 时主 checkpoint 为 train-only 原始分量达到预注册平台后的 `last_converged`；
  weighted total 或 test35 均不能单独决定停止。
- 2026-08-08 文件级核对确认 173/173 每例有 81 帧 `u/v/w/p`、81 个精确匹配
  `Q_in(t)`、0.005 s solver step、0.01 s 场导出间隔、四组 RCR 和正值入口/四出口面积。
  Stage 0 builder 已实现跨时相点身份、时间映射、pressure gauge、注册向量和 train-only
  stats Gate；173/173 全量 Gate 已在本地通过，Slurm CPU `11970` 负责提交后的复核。
- 用户已明确授权正式多 GPU 训练提交；最终链为 `11970 → 11971 → 11972_[0-47%4]`，
  正式数组必须依赖 Stage 0 与 GPU preflight 成功。2026-08-09 用户截断 `11972_7`
  与 `11972_[8-47]`。2026-08-14 补提交 `12210_[7-47%4]`。2026-08-16 用户要求为师姐
  空出两卡：保留 `12210_{11,13}`（GPU0/3），杀死并重排 `14/15`，后续未启动任务改为
  `ArrayTaskThrottle=2`。2026-08-17 凌晨 master 失联，`12210_{11,16}` 被
  `NODE_FAIL` 自动 requeue，`12210_17` 半成品后同样被 requeue。11/16/17 半成品已隔离，
  再启动为全新开跑，不得 resume。2026-08-20 用户授权把等待作业改回四卡并行：
  `scontrol update JobId=12210 ArrayTaskThrottle=4`。同日用户要求使用空闲的
  node04：该节点只有 2×A100、不在 GPU 分区，已直启 `11/14`（PID
  `2098215/2098627`），并从数组 `scancel` 对应 task。当前 master 跑 `22/23/24`。
  test35 批量评估、WSS 和工作簿回填仍未获授权。

### 历史 V3 与旧 field-v4 冻结合同

- 科学定位固定为“Fluent 瞬态 peak 标签 + Carreau–Yasuda 准稳态 PINN 正则”，
  不含 `du/dt`，不得写成稳态 CFD surrogate。
- 2026-08-06 起，下一轮唯一优化目标是 `u,v,w,p` 场重建。禁止恢复直接 WSS 输出、
  新增 WSS loss/WSS 辅助监督，或用 WSS 指标参与训练、early stopping、checkpoint
  选择、loss 调权和架构排序。`wss_mri_calculator` 的 Profile-Secant V3 冻结为下游
  验证器；只允许在基于 validation 的 `u/v/w/speed/p`、区域、流量和压降指标确定主
  checkpoint 后做一次 sanity check，不得在本路线继续调 WSS 算法。
- 下一轮科学合同已冻结在
  `docs/02-推进与变更/WSS_PINN/核心代码诊断与下一轮设计建议_2026-08-05.md`
  的 `FROZEN FOR IMPLEMENTATION v1.0`。实现必须先完成 Stage 0-a 的病例等权
  `S_field^cb`、validation 聚合、B0 atlas 与探针入库；任何 BC 实验前完成 Stage 0-b。
  Stage 1 不输入病例 BC、不使用 PDE、不读取 test35，且不得覆盖现有 V3 配置/产物。
- Stage 1 决策已冻结为：保留 G-Raw；G-PE 跨种子方向不稳定且触发 pressure/最差
  ILO 队列护栏，禁止继续扫 PE；L-Raw 单 seed 无增量，禁止继续堆 local decoder。
  Stage 2 尚未启动，任何 patient-specific conditioning / 可部署 BC 实验必须建立新的
  独立配置、Gate 与 decision contract，不得把 Stage 1 结果扩写成已验证 BC 路线。
- split 固定 train123/val15/test35，SHA256：
  `c80cb65ad95f7d76fadaff82ea02c35c27ad24ab2474e1050bf981eff97493f9`；
  test35 不参与训练、选模或续训判断。
- 六臂固定为 `xyz/xyz+geom × DATA/DATA+BC/DATA+BC+PDE`，每个输入组三臂共享
  seed、初始化、采样协议和 2500 epoch 预算，不 warm-start。
- query decoder 只能直接接收 query xyz 与病例 latent；禁止 3NN、IDW、BatchNorm、
  LayerNorm、dropout、AMP 和 query 邻域离散选择。
- 已完成 V3 的 BC 历史合同使用 Fluent wall、`vf-in-rfile.out` 的 exact peak
  实测入口流量，以及四个 exact-timestep, monitor-derived outlet pressure
  proxy。UDF 分母只作 provenance 诊断；入口速度由实测流量除真实网格面积。
  下一轮在恢复 pressure-outlet face 上的 RCR UDF profile 前，不得再把该
  outlet proxy 写成已确认的 exact boundary-face pressure，也不得默认
  把求解后 monitor 作为可部署输入。
- 正式运行：master 四臂通过 Slurm GPU array `11301_[0-3]` 完成；node04 两臂因未
  注册 GPU GRES，按 UID/公共路径/GPU 空闲 Gate 后一臂一卡直启，原 PID
  `1241185/1241335` 已正常退出。六臂均为 2500 epoch / 155000 step completed，三类
  checkpoint 的 test35 评估 18/18 已完成；主结论固定使用 `best_validation_data`。

### 历史 V1/V2

- 只做 peak 时刻，采用准稳态形式，不伪造 `du/dt`。
- 输出固定为四通道 `u,v,w,p`；压力使用真值监督。
- 压力标签是病例严格体域均值中心化后的相对压力 Pa，不按 mini-batch 重新定零。
- 输入只允许 `xyz` 或 `xyz + abscissa_norm + local_radius + signed_log1p(curvature)`。
- 架构只允许 PointNet 和纯 PointNet++；不接 PointNeXt、Transformer、LocalGeoPE
  或旧 WSS checkpoint。
- 数据划分固定为 train138/test35，split SHA256：
  `964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b`。
- 采样是全体严格体域均匀随机并每 epoch 重采样；首轮 v1 的 support/query 独立，
  SAME5K-E7500 v2 明确使用 `support_idx == query_idx`，二者不得混写。
- 壁面重复行不得进入 support、query 或 PDE；no-slip 使用独立壁面点云。
- 速度向量必须与注册坐标使用同一 `transform_rotation`。
- 非牛顿流变固定为 Carreau–Yasuda，`rho=1060 kg/m^3`，`U0=1 m/s`。
- PINN 从第一个 step 同时启用 continuity、完整三分量动量残差和 no-slip。
- 动量项保留 `div(2 mu D)` 的变黏度应力散度，禁止简化成 `mu laplacian(u)`。

## 八实验与初始化

矩阵是 `PointNet/PointNet++ × xyz/xyz+geom × data-only/PINN`，共 8 个实验。
每一对 data-only/PINN 必须使用相同 seed、初始化哈希、support/query 采样协议和
监督损失。PINN 不得加载 data-only 权重。

`train.resume` 只允许同一个 run 目录内的断点续训；它不是热启动。任何跨 run
checkpoint 初始化都视为热启动并拒绝。正式提交前必须保留初始化 checkpoint 与
初始化 state SHA256。

## 数据与提交 Gate

正式训练前必须全部满足：

1. 173 例 schema-v2 sidecar 构建完成；
2. sidecar + deep-source audit 均通过；
3. train138-only 严格体域统计与 split、病例 manifest hash 完整绑定；
4. `test_volume_*.py`、八臂静态 preflight 和八臂 GPU dry-run 通过；
5. 用户再次明确授权正式训练。

V3 另要求 173/173 真实边界 Gate、六臂静态 preflight、六张物理 GPU dry-run 全部
通过。提交后的前 30 分钟监控和完训后的三 checkpoint test35 评估均已完成；后续不得
按 test35 结果反选 checkpoint、决定续训或改写预注册主口径。

没有新的 schema-v2 数据 Gate 时，不得提交训练。默认提交器只做 dry-run；
`--submit` 是显式且受 Gate 保护的操作。

## 日志与评估

每个正式 run 至少写出：

- `resolved_config.json`、`environment.json`、`checkpoints/initialization.pt`；
- `training_progress.jsonl` 与 `history.csv`：逐 step 的 data/physics/total loss；
- `epoch_progress.jsonl`：逐 epoch 的均值 data loss、continuity、三分量 momentum、
  no-slip、physics total 和 total；
- `best_data.pt`、`best_total.pt`、`last.pt`；
- 三个 checkpoint 的物理单位评估 JSON；SAME5K-E7500 v2 同时保留 SAME5K 主协议
  与固定 5k support→full-volume 副协议。

V3 的对应命名为 `best_validation_data.pt`、`best_validation_total.pt`、`last.pt`，三类
checkpoint 均使用固定 5k support→full strict-volume test35 协议；跨臂主比较只能用
`best_validation_data`。

已有 V1/V2/V3 和后续诊断已读取 test35 并用于新设计；因此下一轮可继续把
test35 用作历史统一 screen，但不得再宣称它是新架构的未触碰确认集。

评估必须同时报告 `u/v/w/speed/p`、近壁/核心分区、压力 gauge 诊断、壁面速度、
continuity 和 momentum residual。PointNet++ SAME support 上的 residual 必须标注
IDW 导数退化限制，并用独立 query 作连续场审计；不得只看总 loss 宣称 PINN 收敛。

## 文档责任

- 代码入口：`wss_pinn/README.md`
- 路线真源：`docs/02-推进与变更/WSS_PINN/README.md`
- 当前执行提示词：`docs/02-推进与变更/WSS_PINN/WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1.md`
- WSS 详细推进记录：`docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`
- 项目级短摘要：`docs/02-推进与变更/代码修改与实验推进记录.md`
- 历史路线：`docs/02-推进与变更/WSS_PINN/_archive/wss_target_v1_20260730/`
- 历史源码：`wss_pinn/archive/wss_target_v1_20260730/`

修改本路线代码、数据合同、配置、Gate、作业入口或科学口径后，必须同步更新活动
README 和 WSS 专用推进记录；不得把旧路线结论复制成当前路线状态。
