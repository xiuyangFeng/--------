# 峰值体域 `u,v,w,p` PINN 路线

> 路线 ID：`volume_uvwp_peak_v1`
>
> 冻结日期：2026-08-02
>
> 状态：**schema-v2 data Gate passed / GPU preflight `11127` completed /
> training array `11128_[0-7%4]` 8/8 completed / results audited；
> SAME5K-E7500 v2 GPU preflight `11137` completed / training array
> `11138_[0-7%4]` running / first 30-minute health check passed**

## 1. 本轮目标与边界

当前路线只回答一个问题：在相同 PointNet / PointNet++、相同峰值体域监督、相同随机
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
  -m wss_pinn.volume_field.tools.summarize_results
```

## 10. 第二轮 SAME5K-E7500 v2

> 状态：**static preflight passed / GPU preflight `11137` completed 8/8 /
> training array `11138_[0-7%4]` running / first 30-minute health check passed**。

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
- 首批四个 PointNet 臂连续观察 30:35：累计约 22,432 条已落盘 step 日志均 finite，
  四份 stderr 均 0 字节，无 NaN/Inf/OOM/Traceback/CUDA error/restart；epoch400/1000
  里程碑与定期 `last.pt` 写盘通过。梯度裁剪仍较频繁，作为后续优化诊断保留，不在
  运行中改协议。机器可读快照：`monitor_30min.json`。

## 11. 真源与归档

- 代码说明：`wss_pinn/README.md`
- 矩阵：`wss_pinn/configs/volume_uvwp_peak_v1/matrix.json`
- SAME5K-E7500 v2 矩阵：
  `wss_pinn/configs/volume_uvwp_peak_same5k_e7500_v2/matrix.json`
- 首轮结果：`outputs/wss_pinn/volume_uvwp_peak_v1/summary/manifest.json`
- 第二轮提交记录：`outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/submission.json`
- 第二轮 30 分钟健康快照：
  `outputs/wss_pinn/volume_uvwp_peak_same5k_e7500_v2/monitor_30min.json`
- 详细推进记录：`docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`
- 项目级摘要：`docs/02-推进与变更/代码修改与实验推进记录.md`
- 历史 WSS-target 文档：`_archive/wss_target_v1_20260730/`
- 历史源码快照：Git commit
  `bca002025d40b290d171570e6470f484fd4feec7`
