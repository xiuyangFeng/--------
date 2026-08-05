# Liao 2025 脑血管 PointNet/PINN 论文与当前 V2 对照

> 论文：Jing Liao et al., *Physics-informed neural networks for three-dimensional
> cerebrovascular hemodynamic prediction: A point cloud preprocessing strategy
> based on limited data*, EAAI 160 (2025) 112034。
>
> 本地 PDF：`docs/paper_idea/1-s2.0-S0952197625020421-main.pdf`
>
> 状态更新：2026-08-04。论文原始路线在 AG 私有数据上的非 PINN/PINN 复现均已
> 结案；当前新增与峰值体域 SAME5K-E7500 V2 的零重训机制对照。

## 1. 原文到底报告了什么

- 51 例右侧 ICA C3–C7，40 train / 11 test，另做四组 Monte Carlo 划分；
- CFD 是严格稳态、常黏度 Newtonian：`rho=1050 kg/m³`、`mu=0.004 Pa·s`；
- 所有病例 no-slip，入口速度统一调到入口 WSS=`1.5 Pa`，出口按面积平方律分流；
- 0.05 mm voxelization + 距离加权插值，每例 10000 点；
- PointNet-style 直接映射 `(x,y,z) -> (Vx,Vy,Vz,P)`，`tanh`，dropout=0.5；
- `loss = loss_data + omega*loss_phy`，`omega=loss_data/loss_phy` 每 10 epoch 更新；
- Adam lr=0.003，ReduceLROnPlateau，batch=1，15000 epoch；
- 主指标是 velocity/pressure NMAE=`7.79±2.14% / 6.63±2.80%`；
- FR R²=`0.9624`、PD R²=`0.9355` 是病例级派生指标，不是逐点速度 R²。

原文没有报告 pointwise velocity/speed R²、near-wall R² 或由预测速度恢复 WSS 的精度。

## 2. 与当前峰值体域 V2 的核心区别

| 维度 | 原文 | 当前 V2 | 影响 |
| --- | --- | --- | --- |
| 场函数 | 直接坐标 PointNet | PointNet++ support latent → 3NN IDW query head | 原文同点 PDE 不会触发 support 插值奇点；V2 会 |
| 时间假设 | 稳态标签 + 稳态 PDE | 瞬态 CFD peak + 准稳态 PDE | V2 漏掉非零 `du/dt` |
| 流变 | 常黏度 Newtonian | Carreau–Yasuda | V2 方程更复杂，但与标签 UDF 更一致 |
| BC | 入口/出口协议跨病例标准化 | 患者特异入口与 RCR，网络不输入 BC | V2 的几何→流场映射更欠定 |
| 点数/采样 | 每例均匀 10000 点 | 5000 uniform support；近壁信息少 | V2 近壁速度与梯度更难 |
| loss 平衡 | 动态 `omega` | 固定 1:1 原始权重 | V2 末期 physics loss 中 no-slip 占 99.85% |
| 验证 | 四组划分 | reused test35、单 seed | V2 方差证据更弱 |
| 指标 | NMAE + 病例级 FR/PD R² | 点级 case-balanced R²/MAE/RMSE + 分区 | 数字不能直接横比 |

### 2.1 SAME-IDW 是当前实现特有问题

V2 的 `differentiable_knn_interpolate` 使用 `weight=1/d²`。当 physics query 正好等于
support 点时，重合点权重压倒其余邻居，且 `d²` 在零位移处的一阶导数为零，使
interpolated latent 对 query coordinate 近似常量。直接坐标 PointNet 没有这层
support→query 插值，因此论文里“physics 点与数据点相同”不能为 V2 SAME5K 辩护。

冻结 checkpoint 的 test35 审计结果：

| 条件 | continuity RMS | momentum RMS (Pa/m) | 相对 exact 动量 |
| --- | ---: | ---: | ---: |
| exact support | 0.00718 | 22.79 | 1.0× |
| independent volume | 3.2065 | 7330.05 | 331.2× |
| support + `1e-4` | 0.3222 | 2673.83 | 119.9× |

因此 V2 的低 SAME residual 是查询几何造成的数值假象，不是连续场已满足 PDE。

### 2.2 准稳态只解释部分误差

AG/AAA/ILO 各 1 例、每例 256 个 strict-core 点的 CFD 数值导数审计显示，`k=96`
下加入 `rho*du/dt` 后动量 residual 从 `1737` 降到 `1494 Pa/m`，三例分别下降约
`6.5%/19.2%/24.9%`。所以 peak 不能严格当稳态，但缺少时间项也不是 speed R² 低的
唯一原因。由于局部二次拟合的二阶导数对 `k=48/96` 敏感，绝对 Pa/m 只作趋势判断。

## 3. 为什么原文 NMAE 很低但不能说明 R² 一定高

按现有老师源码快照的口径，NMAE 对每个 `u/v/w` 分量使用 test 全局 range 作分母，
再对病例和三分量求均值。用完全相同口径重算当前 V2：

| 模型/协议 | velocity NMAE | speed R² |
| --- | ---: | ---: |
| 当前 V2 SAME5K | **2.18±0.70%** | 0.2380 |
| 当前 V2 full-volume | **2.15±0.67%** | 0.1820 |
| 原文主模型 | 7.79±2.14% | 未报告 |

这不是说 V2 比原文更准，而是说明跨数据集 global-range NMAE 的分母很宽，低 NMAE
可以与低空间方差解释度同时出现。原文 FR/PD 的高 R² 也只证明病例级积分量相关性，
不能替代逐点速度、near-wall 速度或 WSS 梯度验证。

## 4. 真正值得借鉴的设计

1. **直接、平滑的坐标场函数**：physics 点可以与监督点相同，但中间不能有 exact
   support 奇点；优先考虑直接 coordinate decoder、连续 kernel 或独立 collocation。
2. **标签与 PDE 假设一致**：稳态标签配稳态方程；当前脉动 peak 若继续做 PINN，
   应输入时间/相位并加入 `du/dt`，或改用真正稳态 CFD 标签。
3. **把 BC 变成可观测条件**：至少输入 flow rate/波形幅值、入口截面条件与出口
   RCR/压力；否则几何无法唯一决定患者特异速度尺度和分流。
4. **动态平衡 data/PDE/BC**：不能只设相同标量权重；需要监控每项数值和梯度贡献。
5. **多划分和多指标**：保留 NMAE，同时补 pointwise R²、MAE/RMSE、near-wall 与
   病例级积分量，避免单一指标制造“很准”的错觉。
6. **点云均匀化要服从终点**：对 `u,v,w,p` 可能有效；若目标包含 WSS，必须另保
   wall/near-wall 预算，不能用全域均匀性替代边界层分辨率。

`tanh`、dropout=0.5 和 15000 epoch 可以作为消融候选，但不是当前首要差异；V2 已
使用平滑 SiLU，且 7500 epoch 已平台，继续加 epoch 不会修复不可辨识 BC 或 IDW 导数。

## 5. 复现与证据入口

- 原文 PDF：`docs/paper_idea/1-s2.0-S0952197625020421-main.pdf`
- 老师/研究代码快照：`wss_pinn/PIPN-QN Code/`
- 外部原始源码快照：`external_baselines/CROWN_Beihang/source_snapshot/`
- 可运行适配：`external_baselines/crown_beihang/`
- 非 PINN/PINN 合并汇报：
  `external_baselines/crown_beihang/experiments/CROWN_非PINN与PINN复现汇报_合并.md`
- 当前 V2 路线真源：`docs/02-推进与变更/WSS_PINN/README.md`
- 零重训工具：`wss_pinn/tools/diagnose_v2_physics.py`
- 审计结果：
  `outputs/wss_pinn/audits/volume_uvwp_peak_same5k_e7500_v2_residual_diagnosis/`

论文正文与代码快照不完全一致时，以论文为方法事实真源。当前 `PIPN-QN Code/` 还包含
`t`、WSS、固定 `Re` 等其它实验变体，不能全部归因于这篇稳态四输出论文。
