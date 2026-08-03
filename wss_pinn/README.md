# 峰值体域 `u,v,w,p` PINN

> 路线：`volume_uvwp_peak_v1`
>
> 当前状态（2026-08-02）：**schema-v2 173 例数据 Gate 通过；GPU preflight
> Job `11127` 与 8 臂训练数组 `11128_[0-7%4]` 全部完成。每臂 400 epoch、
> 27,600 optimizer step，日志与 24 份 checkpoint 评估完整；第二轮 SAME5K-E7500
> GPU preflight `11137` 已完成，训练数组 `11138_[0-7%4]` 运行中，首批四臂30分钟
> 健康监控通过。**

这次重构不再直接预测 WSS，也不复用旧 WSS 最优网络或 checkpoint。当前目标是：
给定峰值时刻的患者体域点云，用 PointNet 或 PointNet++ 输出四通道
`u,v,w,p`，并在严格配对实验中比较纯数据监督与非牛顿 PINN。

活动路线真源见
[WSS_PINN/README.md](../docs/02-推进与变更/WSS_PINN/README.md)。旧的 WSS-target
PINN 路线已冻结在
[_archive/wss_target_v1_20260730](../docs/02-推进与变更/WSS_PINN/_archive/wss_target_v1_20260730/README.md)。

## 八个实验

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
- Slurm：preflight `11137` 8/8 completed，正式 array `11138` running，`--time=0`，
  最多 4 卡并发；30分钟快照无 NaN/OOM/Traceback，见输出根 `monitor_30min.json`。

## 代码结构

```text
wss_pinn/
├── configs/volume_uvwp_peak_v1/       # 8 configs + matrix
├── configs/volume_uvwp_peak_same5k_e7500_v2/ # 第二轮 SAME5K 8 configs + matrix
├── tests/test_volume_*.py              # 数据、模型、物理与训练 smoke
└── volume_field/
    ├── data/                            # schema-v2 builder/dataset/audit
    ├── models/                          # PointNet / PointNet++ 连续查询场
    ├── physics/                         # Carreau–Yasuda 与 PDE residual
    ├── tools/                           # build/audit/preflight
    ├── cluster/                         # Slurm 与默认 dry-run 提交器
    ├── losses.py
    ├── train.py
    └── evaluate.py
```

## 当前允许执行的命令

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# 单元测试与静态八臂预检
$PY -m unittest discover -s wss_pinn/tests -p 'test_*.py' -v
$PY -m wss_pinn.volume_field.tools.preflight

# 只查看提交计划，不提交 Slurm
$PY -m wss_pinn.volume_field.cluster.submit_matrix

# 从已完成 run 重新生成 CSV/JSON 与 PNG/SVG
$PY -m wss_pinn.volume_field.tools.summarize_results
```

173 例数据构建和 deep-source audit 已完成；以下命令用于按原合同复现：

```bash
sbatch wss_pinn/volume_field/cluster/build_dataset.slurm
sbatch --dependency=afterok:<build_job_id> \
  wss_pinn/volume_field/cluster/audit_dataset.slurm
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
test35 仍不参与调度或 checkpoint 选择。配对主表统一使用 `last@7500`，完整结果待
array `11138` 完训后分析回填。
