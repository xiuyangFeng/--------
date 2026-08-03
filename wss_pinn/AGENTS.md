# wss_pinn — 峰值体域 `u,v,w,p` PINN 指令

本目录当前只维护 `volume_uvwp_peak_v1`：用 PointNet / PointNet++ 从患者点云
预测峰值时刻体域四通道 `u,v,w,p`，并比较 data-only 与从随机初始化直接训练的
PINN。旧的“直接 WSS 输出 + 阶梯 F0/F1/F2”路线已经归档，不再作为活动口径。

## 隔离边界

- `data_new/`、`data_wss_min/`、`pipeline_wss_min/`、`training_wss_min/` 是只读上游。
- 新派生数据只写 `data_wss_pinn/volume_uvwp_peak_v1_train138_test35/`。
- 新结果只写 `outputs/wss_pinn/volume_uvwp_peak_v1/` 与对应 audit 目录。
- 活动代码、配置、测试和 Slurm 入口位于 `wss_pinn/volume_field/`、
  `wss_pinn/configs/volume_uvwp_peak_v1/` 和 `wss_pinn/tests/test_volume_*.py`。
- 不删除旧代码、旧数据或旧输出；历史源码以 Git commit
  `bca002025d40b290d171570e6470f484fd4feec7` 为冻结快照。

## 冻结科学合同

- 只做 peak 时刻，采用准稳态形式，不伪造 `du/dt`。
- 输出固定为四通道 `u,v,w,p`；压力使用真值监督。
- 压力标签是病例严格体域均值中心化后的相对压力 Pa，不按 mini-batch 重新定零。
- 输入只允许 `xyz` 或 `xyz + abscissa_norm + local_radius + signed_log1p(curvature)`。
- 架构只允许 PointNet 和纯 PointNet++；不接 PointNeXt、Transformer、LocalGeoPE
  或旧 WSS checkpoint。
- 数据划分固定为 train138/test35，split SHA256：
  `964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b`。
- 采样是全体严格体域均匀随机；support/query 独立且每 epoch 重采样。
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

没有新的 schema-v2 数据 Gate 时，不得提交训练。默认提交器只做 dry-run；
`--submit` 是显式且受 Gate 保护的操作。

## 日志与评估

每个正式 run 至少写出：

- `resolved_config.json`、`environment.json`、`checkpoints/initialization.pt`；
- `training_progress.jsonl` 与 `history.csv`：逐 step 的 data/physics/total loss；
- `epoch_progress.jsonl`：逐 epoch 的均值 data loss、continuity、三分量 momentum、
  no-slip、physics total 和 total；
- `best_data.pt`、`best_total.pt`、`last.pt`；
- 三个 checkpoint 的物理单位评估 JSON。

评估必须同时报告 `u/v/w/speed/p`、近壁/核心分区、压力 gauge 诊断、壁面速度、
continuity 和 momentum residual。不得只看总 loss 宣称 PINN 收敛。

## 文档责任

- 代码入口：`wss_pinn/README.md`
- 路线真源：`docs/02-推进与变更/WSS_PINN/README.md`
- WSS 详细推进记录：`docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`
- 项目级短摘要：`docs/02-推进与变更/代码修改与实验推进记录.md`
- 历史路线：`docs/02-推进与变更/WSS_PINN/_archive/wss_target_v1_20260730/`

修改本路线代码、数据合同、配置、Gate、作业入口或科学口径后，必须同步更新活动
README 和 WSS 专用推进记录；不得把旧路线结论复制成当前路线状态。
