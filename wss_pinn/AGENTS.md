# wss_pinn — 峰值体域 `u,v,w,p` PINN 指令

本目录当前活动实验是 `volume_uvwp_peak_qs_smooth_v3`：用 Fluent 瞬态 peak
`u,v,w,p` 标签和 Carreau–Yasuda 准稳态 PINN 正则，比较无 3NN/IDW 的条件
PointNet 平滑场六臂。`volume_uvwp_peak_v1` 与 SAME5K-E7500 v2 已完成并保留为历史
对照；旧“直接 WSS 输出 + 阶梯 F0/F1/F2”路线已归档。

## 隔离边界

- `data_new/`、`data_wss_min/`、`pipeline_wss_min/`、`training_wss_min/` 是只读上游。
- 新派生数据只写 `data_wss_pinn/volume_uvwp_peak_v1_train138_test35/`。
- V3 新派生边界只写
  `data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/`。
- 新结果只写 `outputs/wss_pinn/volume_uvwp_peak_v1/`、
  `outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/`、
  `outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/` 与对应 audit 目录。
- 活动训练、评估、数据、模型、物理、工具和 Slurm 入口直接位于 `wss_pinn/`
  根层；活动配置位于 `wss_pinn/configs/volume_uvwp_peak_v1/`、
  `wss_pinn/configs/volume_uvwp_peak_same5k_e7500_v2/` 和
  `wss_pinn/configs/volume_uvwp_peak_qs_smooth_v3/`。测试只保留配置、采样、
  模型可微性和物理公式四类核心合同，位于 `wss_pinn/tests/test_volume_*.py`；
  真实数据与 GPU 冒烟由 preflight 承担，不再复制大型 synthetic fixture。
- 旧直接 WSS/F0–F2 源码与配置冻结在
  `wss_pinn/archive/wss_target_v1_20260730/`；`data/raw_io.py`、
  `data/alignment.py` 和 `utils.py` 因仍被当前路线复用而保留在活动区。
- 不删除旧数据或旧输出；历史 Git 快照仍为
  `bca002025d40b290d171570e6470f484fd4feec7`。

## 冻结科学合同

### 当前 V3

- 科学定位固定为“Fluent 瞬态 peak 标签 + Carreau–Yasuda 准稳态 PINN 正则”，
  不含 `du/dt`，不得写成稳态 CFD surrogate。
- split 固定 train123/val15/test35，SHA256：
  `c80cb65ad95f7d76fadaff82ea02c35c27ad24ab2474e1050bf981eff97493f9`；
  test35 不参与训练、选模或续训判断。
- 六臂固定为 `xyz/xyz+geom × DATA/DATA+BC/DATA+BC+PDE`，每个输入组三臂共享
  seed、初始化、采样协议和 2500 epoch 预算，不 warm-start。
- query decoder 只能直接接收 query xyz 与病例 latent；禁止 3NN、IDW、BatchNorm、
  LayerNorm、dropout、AMP 和 query 邻域离散选择。
- BC 使用真实 Fluent wall、`vf-in-rfile.out` 的 exact peak 实测入口流量和四出口
  exact peak 压力。UDF 分母只作 provenance 诊断；入口速度由实测流量除真实网格面积。
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

评估必须同时报告 `u/v/w/speed/p`、近壁/核心分区、压力 gauge 诊断、壁面速度、
continuity 和 momentum residual。PointNet++ SAME support 上的 residual 必须标注
IDW 导数退化限制，并用独立 query 作连续场审计；不得只看总 loss 宣称 PINN 收敛。

## 文档责任

- 代码入口：`wss_pinn/README.md`
- 路线真源：`docs/02-推进与变更/WSS_PINN/README.md`
- WSS 详细推进记录：`docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`
- 项目级短摘要：`docs/02-推进与变更/代码修改与实验推进记录.md`
- 历史路线：`docs/02-推进与变更/WSS_PINN/_archive/wss_target_v1_20260730/`
- 历史源码：`wss_pinn/archive/wss_target_v1_20260730/`

修改本路线代码、数据合同、配置、Gate、作业入口或科学口径后，必须同步更新活动
README 和 WSS 专用推进记录；不得把旧路线结论复制成当前路线状态。
