# 峰值体域 `u,v,w,p` PINN

> 当前活动路线：`volume_uvwp_peak_field_v4`；`volume_uvwp_peak_qs_smooth_v3` 为冻结历史基线
>
> 当前状态（2026-08-06）：**Stage 0-a/0-b、B0、四臂 seed1234、前二臂三种子确认、
> 训练后导数/support Gate 与 full-volume val15 晋级审计均完成。主选模标量固定为
> 病例等权 `S_field^cb`；最终保留 G-Raw，关闭 PE 扫频，并停止继续堆 local decoder。**

> **当前开发范围**：只优化 `u,v,w,p`。Profile-Secant V3 velocity→WSS 路线冻结为
> 下游验证器；本目录不新增直接 WSS 输出、WSS loss 或 WSS 辅助监督，不用 WSS 指标
> 选择 checkpoint、loss 或架构，也不在本轮继续调 WSS 后处理算法。主模型先由
> `u/v/w/speed/p`、区域、流量、pressure gauge/梯度和压降指标确定，再做一次冻结 WSS
> downstream sanity check。

> **执行结果**：科学设计以
> [`核心代码诊断与下一轮设计建议`](../docs/02-推进与变更/WSS_PINN/核心代码诊断与下一轮设计建议_2026-08-05.md)
> 的 `FROZEN FOR IMPLEMENTATION v1.0` 为准，并已在独立 route 完成。执行合同见
> [`冻结诊断后 Stage 0–1 提示词`](../docs/02-推进与变更/WSS_PINN/WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1.md)。
> 新配置、派生合同和输出分别落到
> `wss_pinn/configs/volume_uvwp_peak_field_v4/`、
> `data_wss_pinn/volume_uvwp_peak_field_v4_train123_val15/` 和
> `outputs/wss_pinn/volume_uvwp_peak_field_v4/`；补种子配置位于
> `wss_pinn/configs/volume_uvwp_peak_field_v4_multiseed_top2/`，V3 路径保持只读。

历史 V1/V2 重构不直接预测 WSS，也不复用旧 WSS 最优网络或 checkpoint；其目标是：
给定峰值时刻的患者体域点云，用 PointNet 或 PointNet++ 输出四通道
`u,v,w,p`，并在严格配对实验中比较纯数据监督与非牛顿 PINN。

活动路线真源见
[WSS_PINN/README.md](../docs/02-推进与变更/WSS_PINN/README.md)。旧的 WSS-target
PINN 路线已冻结在
[_archive/wss_target_v1_20260730](../docs/02-推进与变更/WSS_PINN/_archive/wss_target_v1_20260730/README.md)。
对应历史源码位于
[`archive/wss_target_v1_20260730/`](archive/wss_target_v1_20260730/README.md)。

## field-v4 Stage 0–1（已完成）

本轮只读 train123/val15 资产，test35 病例未进入数据合同；四臂全部为 data-only，
没有 BC input/loss、PDE loss 或 WSS 选模。Stage 0-a 与 Stage 0-b Gate 均为 pass，
B0 病例盲 atlas 的 val15 `S_field^cb=0.699472`。

seed1234 的冻结排名为：

| 排名 | 臂 | `S_field^cb` |
| ---: | --- | ---: |
| 1 | G-PE | 0.567612 |
| 2 | G-Raw | 0.578311 |
| 3 | L-Raw | 0.588856 |
| 4 | L-PE | 0.597688 |

按主标量只给前二补跑 seeds 2345/3456。三种子结果为：

| 臂 | mean ± sample std | 相对 B0 mean Δ | 胜过配对臂的 seeds |
| --- | ---: | ---: | ---: |
| G-Raw | **0.576116 ± 0.007477** | -0.123355 | 2/3 |
| G-PE | 0.581208 ± 0.012599 | -0.118263 | 1/3 |

G-PE−G-Raw 的配对均值为 `+0.005092`（越低越好），方向仅 1/3 seeds 改善；病例
bootstrap 的固定评估分数差为 `+0.004572`，95% CI `[-0.013164, 0.030957]`。
PE 同时触发 pressure RMSE `+2.36%` 和最差 ILO 队列 `+5.58%` 的预注册 2% 护栏；
near-wall 护栏和八个训练 checkpoint 的导数/support Gate 通过。六个前二臂
checkpoint 的 full-volume val15（每个 seed `12,502,817` 点）仍为
G-Raw `0.576816 ± 0.007717`、G-PE `0.581096 ± 0.013807`，不改变结论。

因此 Stage 1 的 Go/No-Go 为：保留 **G-Raw** 作为后续场表示参考；G-PE No-Go，
不继续扫频；L-Raw 单 seed 无增量，不继续堆 local decoder。下一科学假设转向
patient-specific conditioning / 缺失可部署 BC，但 Stage 2 未在本轮启动。

机器可读真源：

- Stage 0-a/0-b：`outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0{a,b}/report.json`；
- Raw 与单种子 Gate：`outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage1_{raw_gate,single_seed_gate}/report.json`；
- B0：`outputs/wss_pinn/volume_uvwp_peak_field_v4/b0/evaluation_val15.json`；
- 多种子/晋级总报告：`outputs/wss_pinn/volume_uvwp_peak_field_v4/stage1_multiseed_and_promotion_report.json`；
- 逐病例 B0 差值：`outputs/wss_pinn/volume_uvwp_peak_field_v4/stage1_b0_case_deltas.csv`；
- 训练后导数/support Gate：`outputs/wss_pinn/volume_uvwp_peak_field_v4/stage1_trained_derivative_support_gate.json`。

V3 已实现为“Fluent 瞬态 peak 标签 + Carreau–Yasuda 准稳态 PINN 正则”：使用无
3NN/IDW 的条件 PointNet 平滑场，运行
`xyz/xyz+geom × DATA/DATA+BC/DATA+BC+PDE` 六臂。完整预注册要求见
[已归档的准稳态平滑场六臂预注册提示词](../docs/02-推进与变更/WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md)。

## V3 六臂（完训并完成 test35 评估）

| 资源 | 实验 |
| --- | --- |
| master / Slurm `11301_0` | `QSF-XYZ-DATA-s1234-v3` |
| master / Slurm `11301_1` | `QSF-XYZ-BC-s1234-v3` |
| master / Slurm `11301_2` | `QSF-XYZ-BCPDE-s1234-v3` |
| master / Slurm `11301_3` | `QSF-XYZG-DATA-s1234-v3` |
| node04 / GPU0 / PID `1241185` | `QSF-XYZG-BC-s1234-v3` |
| node04 / GPU1 / PID `1241335` | `QSF-XYZG-BCPDE-s1234-v3` |

- split：train123/val15/test35，SHA256
  `c80cb65ad95f7d76fadaff82ea02c35c27ad24ab2474e1050bf981eff97493f9`；
- 边界 Gate：`outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_boundary/report.json`；
- 静态/GPU Gate：`outputs/wss_pinn/audits/volume_uvwp_peak_qs_smooth_v3_{preflight,gpu_preflight}/report.json`；
- 提交清单：`outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/submission_six_gpu.json`；
- 健康快照：`outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/monitor_{latest,snapshots.jsonl}`。
- 前 30 分钟摘要：`outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/monitor_30min_summary.json`，
  `health_result=pass`、31/31 快照 healthy、6/6 全程 running、错误关键字 0。
- 六臂训练摘要：每臂 `status=completed`、2500 epoch、155000 step；三类 checkpoint
  6/6 齐全，node04 两个直启 PID 已退出。
- 结果入口：
  [`summary/README.md`](../outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/README.md)；
  主表为 `six_arm_metrics_best_validation_data.csv`，逐病例表、增量表、收敛摘要与
  图件位于同目录。
- 老师汇报损失图：
  [`loss_convergence/README.md`](../outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/loss_convergence/README.md)；
  `teacher_loss_convergence.pdf` 汇总 data loss、physics loss、最后 10% 放大和 BC/PDE
  子项。
- Excel 汇报工作簿：
  [`WSS_PINN_V3_六臂核心实验与指标汇报.xlsx`](../outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/WSS_PINN_V3_六臂核心实验与指标汇报.xlsx)；
  包含“教师汇报、六臂主指标、增量分析、Checkpoint敏感性、收敛与损失、逐病例、
  口径与文件”7张表，并嵌入核心指标图和loss总览。

主 checkpoint `best_validation_data` 的 speed R²_cb / pooled：E1 `0.0344/0.2729`、
E2 `0.0359/0.2658`、E3 `-0.0143/0.2420`、E4 `0.2263/0.3541`、
E5 `0.2336/0.3523`、E6 `0.1799/0.3144`。BC 未稳定提升总体速度；PDE 显著降低
独立 collocation residual，但伤害瞬态 peak 速度 R²；`xyz+geom` 在 DATA/BC/BCPDE
三组均稳定提高 speed R²。near-wall speed R² 六臂仍全为负，因此本路线不支持 WSS
可靠性结论。

2026-08-05 的第一性原理/对抗性复核进一步限定了上述解读：V3 出口压力
目标是当时冻结的 exact-timestep monitor-derived proxy，未证明等于真实
pressure-outlet face 上的 RCR profile；test35 也已经因用于后续设计而成为
development-exposed screen。详见
[`docs/02-推进与变更/WSS_PINN/核心代码诊断与下一轮设计建议_2026-08-05.md`](../docs/02-推进与变更/WSS_PINN/核心代码诊断与下一轮设计建议_2026-08-05.md)。

2026-08-06 起，下一轮开发不再围绕 WSS 算子、WSS loss 或 WSS 选模展开。near-wall
仍是 `u/v/w` 场重建的困难区域，但其优化与验收直接使用速度分量、速度向量和区域误差；
冻结 WSS 结果只用于主模型完成后的误差放大审计。

## 历史 V1/V2 八个实验

| 架构 | 输入 | data-only | PINN |
| --- | --- | --- | --- |
| PointNet | `xyz` | `VF-PN-XYZ-DATA-s1234-v1` | `VF-PN-XYZ-PINN-s1234-v1` |
| PointNet | `xyz+geom` | `VF-PN-XYZG-DATA-s1234-v1` | `VF-PN-XYZG-PINN-s1234-v1` |
| PointNet++ | `xyz` | `VF-PNPP-XYZ-DATA-s1234-v1` | `VF-PNPP-XYZ-PINN-s1234-v1` |
| PointNet++ | `xyz+geom` | `VF-PNPP-XYZG-DATA-s1234-v1` | `VF-PNPP-XYZG-PINN-s1234-v1` |

每对实验共享 seed、初始参数哈希、病例顺序与采样协议。PINN 从随机初始化直接同时
优化 data loss、continuity、三个动量分量和 no-slip；不从 data-only 热启动。

## 架构来源

- PointNet：沿用历史 P2V 宽度，来源配置
  `training_wss_min/configs/pointnet_v4/ag_aaa_v4_stratified_e2_global_random5000_sep.json`。
- PointNet++：沿用纯 PointNet++ D2 `c125-k128`，来源配置
  `training_wss_min/configs/pointnetpp_sa1_scale_single_seed_20260721/d2_rand5000_c125_k128.json`。
- 选择证据：`docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`。
- 只继承架构和采样设计，不加载历史权重；PointNeXt 暂不进入本矩阵。

为了让坐标一、二阶导数可用于 PINN，查询路径使用 SiLU、query-local LayerNorm 和
可微逆距离 3NN 权重；PyG kNN 的离散邻居选择不求导，距离权重对查询坐标求导。

## 数据合同

- split：固定 train138/test35，SHA256
  `964d7021f2d12baadd630e7b936456a4e62294fa72c5ada4fc70abe4d9361f2b`。
- 时相：只读每例 `peak_step`。
- 输出真值：注册坐标系中的 `u,v,w`（m/s）和相对压力 `p`（Pa）。
- 压力：以每例严格体域均值为固定 gauge；训练仍对压力真值做第四通道监督。
- 几何特征：`abscissa_norm`、`local_radius`、`signed_log1p(curvature)`；曲率裁剪和
  均值/标准差只由 train138 严格体域统计。
- 采样：每 epoch 从严格体域均匀随机抽取 5000 support + 独立 5000 query；PINN
  从 query 中取 512 PDE 点，并从独立壁面云取 1024 no-slip 点。
- 三个历史 bundle 含体表壁面重复行；schema v2 保存 `interior_is_wall`，这些行只作
  审计来源，不进入 support/query/PDE，也不污染归一化统计。

旧的 8k near-wall + 8k core sidecar 不符合本路线，不能复用。

## PINN 物理项

第一版是不可压缩、准稳态、广义牛顿非牛顿流：

\[
\nabla\cdot\mathbf u=0,
\]

\[
\rho(\mathbf u\cdot\nabla)\mathbf u+\nabla p-
\nabla\cdot\left(2\mu(\dot\gamma)\mathbf D\right)=0,
\]

\[
\mathbf u|_{\Gamma_w}=0.
\]

其中 `rho=1060 kg/m^3`，`U0=1 m/s`，病例长度尺度来自已知
`coord_scale_mm × 1e-3`，黏度使用 Carreau–Yasuda。动量残差保留变黏度应力的完整
散度，不使用常黏度拉普拉斯近似。

## 首轮结果入口

- 完整数值、配对判读与下一轮候选协议：
  [峰值体域 PINN 路线结果](../docs/02-推进与变更/WSS_PINN/README.md#9-首轮八实验结果2026-08-02)
- 可复现汇总：`outputs/wss_pinn/volume_uvwp_peak_v1/summary/`
- 8 个原始 run：`outputs/wss_pinn/volume_uvwp_peak_v1/runs/`

按 `best_total` checkpoint 的 test35 case-balanced 口径，四个 PINN 配对均降低
continuity、momentum 与壁面速度 RMS，压力 R² 也全部提高；速度泛化收益不一致，
PointNet + `xyz+geom` 是唯一 `u/v/w/speed/p` 五项全部提高的配对。因此首轮结论是
“物理一致性显著改善，但不能宣称 PINN 全面优于 data-only”。

## 第二轮 SAME5K-E7500 v2

- 配置：`wss_pinn/configs/volume_uvwp_peak_same5k_e7500_v2/`；
- 输出：`outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/`；
- 采样：严格体域 random5000，`support_idx == query_idx`，每 epoch 重采样；
- split：train138 / val0 / test35；test35 不参与训练控制；
- 预算：7500 epoch / 517,500 optimizer step；固定保存 400/1000/2500/5000/7500；
- 主 checkpoint：`last@7500`；训练结束同时输出 SAME5K 与 full-volume 评估；
- Slurm：preflight `11137`、正式 array `11138` 均 8/8 completed，`--time=0`，
  最多 4 卡并发；8 个 run 均为 7500 epoch / 517,500 step，训练与评估正常完成。

主协议 `last@7500` 的 test35 case-balanced 结果：

| 模型 | speed R² | speed MAE (m/s) | speed RMSE (m/s) | near-wall speed R² | p R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| PointNet++ / `xyz+geom` / data-only | 0.2063 | 0.1773 | 0.2672 | -2.0509 | 0.1729 |
| **PointNet++ / `xyz+geom` / PINN** | **0.2380** | **0.1764** | **0.2668** | **-1.4770** | **0.3225** |

完整体域协议下两者 speed R² 为 `0.1506/0.1820`，但 MAE/RMSE 几乎持平。PINN
显著改善 physics residual 和压力，速度聚合 R² 小幅提高；近壁 speed R² 仍为负，
因此当前结果不能直接支持可靠 velocity→WSS。完整八臂表、逐病例异常与下一步见
[路线真源第 10 节](../docs/02-推进与变更/WSS_PINN/README.md#10-第二轮-same5k-e7500-v2)。

### V2 零重训 residual 诊断

2026-08-04 使用冻结 `last@7500` 对全部 test35 做了 exact support、独立体域 query
和微扰 query 对照。独立点的 continuity/momentum/速度梯度病例均衡 RMS 分别是
exact support 的约 `453×/331×/187×`；仅移动 `1e-4` 归一化长度，预测速度变化
RMS 仅 `1.84e-5 m/s`，动量 residual 已放大约 `120×`。根因是 PointNet++ 查询头的
`1/d²` 3NN 插值在 support 重合点发生 latent 坐标导数退化。因此既有 SAME5K 回归
指标仍有效，但 SAME 点 physics residual 不能代表连续体域 PDE 满足度。

另对 AG/AAA/ILO 各 1 例 CFD 真值做局部二次导数审计。`k=96` 下加入相邻时相的
`rho·du/dt` 后，动量 residual 均值从 `1737` 降至 `1494 Pa/m`，说明 peak 并非严格
稳态；但二阶导数对 `k=48/96` 仍敏感，绝对 residual 只作趋势诊断。完整结果和老师
论文对照见[路线真源第 10.6–10.7 节](../docs/02-推进与变更/WSS_PINN/README.md)。

## 代码结构

```text
wss_pinn/
├── config.py / train.py / evaluate.py  # 当前训练与评估主入口
├── data/                               # schema-v2 builder/dataset/audit + 共享 raw I/O
├── models/                             # PointNet / PointNet++ 连续查询场
├── physics/                            # Carreau–Yasuda 与 PDE residual
├── tools/                              # build/audit/preflight/结果汇总
├── cluster/                            # Slurm 与默认 dry-run 提交器
├── configs/volume_uvwp_peak_v1/        # 首轮 8 configs + matrix
├── configs/volume_uvwp_peak_same5k_e7500_v2/ # 第二轮 8 configs + matrix
├── configs/volume_uvwp_peak_qs_smooth_v3/ # 当前六臂 configs + matrix
├── configs/volume_uvwp_peak_field_v4/ # Stage 1 原始四臂 + decision contract
├── configs/volume_uvwp_peak_field_v4_multiseed_top2/ # 前二臂确认种子
├── tests/test_volume_*.py              # 4 个核心合同测试文件
└── archive/wss_target_v1_20260730/     # 旧直接 WSS/F0–F2 源码与配置
```

2026-08-04 起，当前体域路线不再隔着 `volume_field/` 子包；训练、评估和工具命令
统一从 `wss_pinn` 根包启动。归档代码仅用于追溯，不作为可执行入口。

2026-08-05 的维护性精简没有修改科学合同或实验配置入口：配置类和模型构造器的推荐
名称分别为 `ExperimentConfig`、`build_model`，历史名称仍保留兼容；`losses.py` 统一
处理 V1/V2 与 V3 的数据、BC、PDE 组合，训练入口只记录优化所需 loss，物理场诊断
集中在 `evaluate.py`。测试不再重复构造完整数据 Gate 和 resume 集成场景，真实提交
仍由“静态 preflight → 数据 Gate → GPU dry-run → 配置矩阵提交”保护。标准提交器
默认直接读取配置目录的 `matrix.json`，不再要求同时维护一份重复的 config 文本清单；
历史清单仍可通过 `--config-list` 显式复用。

## 当前允许执行的命令

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# 核心合同测试与静态矩阵预检
$PY -m unittest discover -s wss_pinn/tests -p 'test_*.py' -v
$PY -m wss_pinn.tools.preflight

# 只查看提交计划，不提交 Slurm
$PY -m wss_pinn.cluster.submit_matrix

# 从首轮 v1 已完成 run 重新生成 CSV/JSON 与 PNG/SVG
$PY -m wss_pinn.tools.summarize_results

# 不重训：复核 V2 SAME/独立 query、CFD 准稳态/瞬态 residual 和论文 NMAE 桥接
$PY -m wss_pinn.tools.diagnose_v2_physics --section all --device cuda:0

# field-v4 Gate、结果汇总与最终决策（只读已完成 run）
$PY -m wss_pinn.tools.gate_stage1_single_seed_field_v4
$PY -m wss_pinn.tools.finalize_stage1_field_v4 --device cuda:0
```

173 例数据构建和 deep-source audit 已完成；以下命令用于按原合同复现：

```bash
sbatch wss_pinn/cluster/build_dataset.slurm
sbatch --dependency=afterok:<build_job_id> \
  wss_pinn/cluster/audit_dataset.slurm
```

首轮正式训练已通过 Slurm array `0-7%4` 在最多 4 张 GPU 上完成。后续新 run 仍须
先通过 schema-v2 数据 Gate 和 GPU preflight，且不得覆盖本轮 v1 目录。

## 运行日志

正式 run 会保留：

- `training_progress.jsonl` / `history.csv`：逐 step 四通道 data loss、
  continuity、momentum-x/y/z、no-slip、physics total、total、梯度范数；
- `epoch_progress.jsonl`：逐 epoch 对上述 loss 取均值，用于看 data loss 与 phy loss
  是否共同收敛；
- `initialization.pt` 与初始化哈希、`best_data.pt`、`best_total.pt`、`last.pt`；
- 三个 checkpoint 的物理单位全体域评估，包括 `u/v/w/speed/p`、壁面速度、
  continuity、momentum 和压力 gauge 偏移诊断。

同一 run 的 `resume` 只用于故障恢复；跨 run checkpoint 初始化会被代码拒绝，因而
不会把断点续训误写成 data-only 热启动。

第二轮已切换为固定 7500 epoch 的 SAME5K 协议，不划 inner validation、不早停；
test35 仍不参与调度或 checkpoint 选择。配对主表统一使用 `last@7500`；8/8 run 与
SAME5K/full-volume × 三 checkpoint 的 48 份评估均已完成并审计。
