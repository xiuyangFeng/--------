# CROWN/Beihang 复现汇报（非 PINN + PINN · 合并版）

> **论文**：Jing Liao et al., *Physics-informed neural networks for three-dimensional cerebrovascular hemodynamic prediction: A point cloud preprocessing strategy based on limited data*, Engineering Applications of Artificial Intelligence 160 (2025) 112034  
> **数据**：`private_preprocessed_raw_ascii_v1` · `point_filter=volume` · `split_AG_v1` post-denylist（与 V3P / PointNetCFD 一致）  
> **实验族**：paper-original · 仅 `x,y,z` → `u,v,w,p` · 体素 0.05 mm · lazy 训练 · 正式 test 为 **全点** `paper_full`（10.27 亿点 · 对齐原文 NMAE 口径）

| 分支 | 实验族 | 训练 Job | Evaluate Job | Run 目录 |
| --- | --- | ---: | ---: | --- |
| **非 PINN**（原文消融） | `crown_original_vp` | **5751** | **5774** | `crown_original_vp_split_AG_v1_seed1_20260619_162738/` |
| **PINN**（原文主模型） | `crown_original_vp_pinn` | **5757** | **5806** | `crown_original_vp_pinn_split_AG_v1_seed1_20260620_152546/` |

**汇报指标优先级**：

1. **对齐原文**：§5.1 **NMAE**（Job **5774 / 5806** `paper_full`）— 原文主指标，可粗比 Nagahama 表 1/2  
2. **本项目诊断**：§5.2 **pooled R²** — 原文未报；判 AG 压力是否可用  
3. **分区 R²（wall/interior）** — 原文未报；GT 速度阈值在 CROWN 体点云上切分，**勿与 V3P 壁面 R² 混表**  
4. **定性对照**：§九 1146 云图 — 项目 features 壁面点 · 仅辅助说明空间结构

---

## 一、结论（汇报用总括）

| 维度 | 非 PINN（5751→**5774**） | PINN（5757→**5806**） | 说明 |
| --- | --- | --- | --- |
| **工程复现** | ✅ | ✅ | 体素 export、lazy 训练、PINN 物理项、paper 全点 evaluate、NMAE 口径全链路跑通 |
| **论文主指标（NMAE）** | ⚠️ p **16.0%** | ⚠️ p **13.2%** | 压力与原文消融 **15.16%** 同量级；PINN 改善 ~18% 但未达 **6.63%**；速度 NMAE ~2.2% **跨数据集仅粗比**（AG range 与 Nagahama 不同） |
| **点级 R²（压力）** | ❌ **−2.07** | ⚠️ **−0.63** | PINN 显著改善但仍为负 → **均不可用作工程压力场** |
| **AG 实用价值** | ❌ | ❌ | 同 split 下 PointNetCFD p R²≈0.90、V3P r2_p≈0.93–0.96 |
| **后续** | ❌ 结案 | ❌ 结案 | 不扩 seed、不进 WSS 二阶段；paper-original 路线收口 |

**三句话给老师的版本**：

1. **实现层面**：我们在 AG 私有数据上完整复现了 Liao 2025 的两条 paper-original 路线（有预处理 · 无/有 PINN），评估链路与原文 NMAE 口径一致，说明代码与数据管线可信。
2. **数值层面**：非 PINN 压力 NMAE **16.0%** 与原文消融 **15.16%** 同量级；加 PINN 后降至 **13.2%**（改善 ~18%），方向与原文一致，但都 **未接近** 原文 Nagahama 主模型 **6.63%**。
3. **项目层面**：在 AG 上，仅 xyz + global max-pool 的 CROWN（无论是否 PINN）**远劣于** 带 BC/几何的 V3P/PointNetCFD，**不能**替代主路线，也不适合作为 WSS 后处理 baseline。

---

## 二、论文方法（原文在做什么）

### 2.1 数据与 CFD 设定

| 项 | 原文 |
| --- | --- |
| 病例 | Nagahama **51 例**右侧 ICA C3–C7（49 例有动脉瘤） |
| 划分 | 40 train / 11 test；另 4 组 Monte Carlo 交叉验证 |
| CFD | 稳态不可压 N-S · ANSYS Fluent · ~50 万 mesh/例 · no-slip 壁面 |
| 入口 | 速度入口，入口 WSS 调至 **1.5 Pa** |
| 出口 | 流量按出口面积平方律分配 |
| 物性 | ρ=1050 kg/m³ · μ=0.004 Pa·s |

### 2.2 点云预处理（论文核心之一）

- **体素化** 0.05 mm，体素内随机生成新点；
- 同体素内原始 CFD 点 **距离加权插值** 得到 u,v,w,p；
- 训练时每步随机 **10 000 点**；
- uniform loss 从 26366 降至 606（论文称均匀性改善 97.7%）。

**注意**：预处理使点分布更均匀，但可能 **平滑近壁高梯度**；论文未直接评估 WSS。

### 2.3 网络与 PINN

| 模块 | 功能 |
| --- | --- |
| Module 1 | Shared Conv1d 提点特征 → **global max-pool** 得全局特征 |
| Module 2 | 局部特征 + 全局特征拼接 → 回归 u,v,w,p |
| Module 3（PINN） | 自动微分算 NS 残差 + 壁面 no-slip |

总损失：`loss = loss_data + ω · loss_phy`，ω 每 10 epoch 按 data/physics loss 比例动态更新。

训练：batch=1（原文）· epoch=15000 · Adam lr=0.003 · dropout=0.5。

**输出**：u,v,w,p 四场；**不直接预测 WSS**。临床叙事走 FR/PD 等病例级 surrogate（FR R²=0.96，PD R²=0.94）。

### 2.4 原文四组对照与主指标

| 组别 | 预处理 | PINN | 压力 NMAE | 速度 NMAE | 本次复现 |
| --- | --- | --- | ---: | ---: | --- |
| SOTA 1 | 无 | 无 | 12.55 ± 4.91 % | 11.15 ± 2.78 % | — |
| SOTA 2 | 无 | 有 | 10.05 ± 5.03 % | 9.90 ± 2.82 % | — |
| **消融 3** | **有** | **无** | **15.16 ± 7.20 %** | **12.30 ± 3.79 %** | **5751 / 5774** |
| **主模型** | **有** | **有** | **6.63 ± 2.80 %** | **7.79 ± 2.14 %** | **5757 / 5806** |

关键观察：

- **仅预处理不加 PINN，压力反而最差**（15.16%）；
- PINN 将压力从 ~15% 拉到 ~6.6%；
- 原文用 **NMAE** 作主指标，不用 R²（因壁面速度=0 会影响 MSE 解释）。

---

## 三、本项目复现实现

### 3.1 两分支共同对齐项

| 项目 | 原文 / 源码 | 本项目 |
| --- | --- | --- |
| 输入 | 仅 (x,y,z) | `input_features=["x","y","z"]` |
| 结构 | PointNet + global max-pool | `CrownPointNet` |
| 训练采样 | 每步随机 10 000 点 | `sample_points=10000` |
| 压力归一化 | train 全集 min-max | `stats/train_stats.json`（train-only） |
| Test 推理 | batch=1 · 全点 | `eval_mode=paper_full` |
| NMAE 分母 | test GT global min/max | `nmae_range_source=test_global_gt` |
| 体素化 | 0.05 mm | `voxel_size_mm=0.05` · `point_filter=volume` |

### 3.2 两分支差异

| 项目 | 非 PINN 5751 | PINN 5757 |
| --- | --- | --- |
| `physics.enabled` | false | **true** |
| 损失 | `loss = MSE(u,v,w,p)` | `loss = loss_data + lphy·(loss_phy + loss_wall)` |
| batch_size | 16 | **2**（PINN autograd 显存） |
| steps/epoch | 289 | **2309** |
| best_epoch | 55 | **41** |
| 训练 epoch | 105 早停 | **91 早停** |
| val_loss（best） | 0.00809 | — |

**PINN 物理项配置**：

| 项 | 配置 |
| --- | --- |
| 物理项 | NS 残差（PDE + continuity）+ 壁面 no-slip |
| 坐标尺度 | `coord_scale=1000` |
| 壁面判据 | `\|u\|²+\|v\|²+\|w\|² ≤ 0.01` |
| Reynolds | `Re=300` |
| 采样 | data / physics **两套独立** 随机 10000 idx |

### 3.3 与原文的差异（跨数据集固有）

| 项 | 原文 Nagahama | 本项目 AG |
| --- | --- | --- |
| 解剖范围 | 右侧 ICA C3–C7 段 | **全脑血管** AG 域 |
| 病例数 | 51（40/11） | **81 活跃**（57/8/16 post-denylist） |
| 帧数 | — | train 4617 / val 648 / test **1296** 帧 |
| 每帧体素点数 | ~10 000（训练采样） | **57–130 万**（训练再采 1 万） |
| BC 协议 | 入口 WSS=1.5 Pa 标准化 · **稳态** | 患者特异 **脉动** 入口流量 + 出口压力（**未输入网络**） |
| 时间维度 | 单时刻 | **81 帧/例**（末心动周期 · 见下） |
| 任务终点 | u,v,w,p → FR/PD | 项目主目标 **WSS**；CROWN 仅 u,v,w,p 复现 |

**AG 脉动 CFD 时间轴（简要）**：

- Fluent **Δt=0.005 s**，共算 **1280 步（6.4 s）**；丢弃前 1119 步启动瞬态，保留末周期步号 **1120–1280**（偶数步 **81 帧**，时长 **0.8 s** → **75 bpm**）。
- `t_norm = (step−1120)/160`；**1120 与 1280 为周期首尾**（均低流量，近似舒张末期）。
- 入口流量峰值在 **step≈1162（`t_norm≈0.26`）**；流量低谷在 **step≈1216（`t_norm≈0.60`）**；汇报云图常用 **1146（`t_norm≈0.16`，收缩期上升段）**。
- 相位为 **血流动力学近似**，未与 ECG 对齐；CROWN 未吃 BC/时相 → 81 帧当独立快照训练。详见 [`CROWN训练与采样流程说明.md`](../docs/CROWN训练与采样流程说明.md) §1.1。

跨数据集 NMAE **只能粗对比**；同 split 内与 PointNetCFD / V3P 对比更公平。

### 3.4 数据管线

```
data_new/AG/<case>/ascii_in/  (Fluent 全量体点)
    → export_pkl.py (raw_ascii · 0.05mm 体素化)
    → private_preprocessed_raw_ascii_v1/pkl/
    → CrownLazyDataset (lazy_load · partial_cache)
    → train.py (random 10000/step)
    → evaluate.py (paper_full · 全点)
```

**规模**（raw_ascii v1 · volume）：

| split | 病例 | 帧数 | merged pkl |
| --- | ---: | ---: | ---: |
| train | 57 | 4617 | ~100 GB |
| val | 8 | 648 | ~14 GB |
| test | 16 | 1296 | ~27 GB |

每帧体素点均值 ~81.5 万（范围 57.5–130.4 万）。**正式 test**：1296 帧 **全点** forward，共 **10.27 亿点**。

**与「壁面+内部点」的关系**：`point_filter=volume` 仅用 Fluent **`ascii_in` 体网格**（流动域 interior mesh，含近壁低速单元），**不含** `ascii` 显式壁面网格（那需 `point_filter=all`）。论文原文也是 volume 体点体素化，不是 STL 壁面点云。

**名单**：`split_AG_v1.json` 与 V3P、PointNetCFD 相同；**5 例 denylist**；pkl 与 ascii 压力抽样核对 **< 1 mPa**。

**export 中的 `global_cond`**：预处理阶段保留 BC/时相信息，但 paper-original 配置 **未将其接入网络**——这是与 V3P/PointNetCFD 的核心输入差距之一。

> 完整训练/采样说明：[`docs/CROWN训练与采样流程说明.md`](../docs/CROWN训练与采样流程说明.md)

### 3.5 Run 时间线

| Job | 分支 | 状态 | 说明 |
| --- | --- | --- | --- |
| 5626 | 非 PINN | 作废 | 旧 FPS 15000 features 路径，非论文口径 |
| 5739 | 非 PINN | OOM · 4 epoch | 旧全量 pkl 驻内存 · **不代表方法上限** |
| **5751** | 非 PINN | ✅ | lazy · **105 epoch 早停 · best_ep=55** |
| **5774** | 非 PINN | ✅ | paper evaluate · **正式指标来源** |
| 5740 | PINN | TIMEOUT · 作废 | 24h · 23 epoch · 已被 5757 取代 |
| **5757** | PINN | ✅ | lazy · **91 epoch 早停 · best_ep=41** |
| **5806** | PINN | ✅ | paper evaluate · **正式指标来源** |

**5739 → 5751 工程修复**：`CrownLazyDataset` 替代全量 merged pkl 驻内存；Slurm mem 32G→96G；partial_cache_cases=4。

### 3.6 训练/验证损失曲线

| 分支 | Job | PNG | SVG |
| --- | ---: | --- | --- |
| 非 PINN | 5751 | [5751.png](../docs/plots/crown_original_vp_train_val_loss_5751.png) | [5751.svg](../docs/plots/crown_original_vp_train_val_loss_5751.svg) |
| PINN | 5757 | [5757.png](../docs/plots/crown_original_vp_pinn_train_val_loss_5757.png) | [5757.svg](../docs/plots/crown_original_vp_pinn_train_val_loss_5757.svg) |

---

## 四、指标定义

### 4.1 NMAE（原文主指标）

$$
\text{NMAE} = \frac{\text{MAE}}{\max(y) - \min(y)}
$$

- 分母：test 集该物理量 GT 的 **global min/max**（跨全部 test 帧/点）。
- **mean ± std 的实现**：`metrics_test.json` 中 `paper_*_nmae_mean/std` 是对 **1296 帧逐帧 NMAE** 再聚合（与 evaluate 逐 sample 日志一致）；原文 Nagahama 是对 **11 个 test 病例** 各算一例 NMAE 再 mean±std。AG 上若改为 **16 病例 pooled** 再聚合，压力 mean 仍为 **16.0%**，std 约 **6.8%**（低于逐帧的 9.0%）。
- **越小越好**。

### 4.2 R²（本项目补充 · 原文未报）

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

**为何压力 R² 深负而 NMAE 仍「尚可」**：帧内 p 波动仅几百 Pa；global max-pool 模型易输出 **整帧常数或水平偏移** → MAE 相对全局 range 不大，但不解释帧内方差 → R² 深负。这是判 CROWN「不可用」的关键诊断，与 NMAE 互补。

### 4.3 正式 evaluate 口径

| | subsample（训练内） | **paper_full（正式）** |
| --- | --- | --- |
| 方式 | 每帧随机 1 万点 | test 1296 帧 **全点** forward |
| 总点数 | ~12,960,000 | **1,026,749,520** |
| **正式结论** | ❌ 不作为 Go/No-Go 依据 | ✅ **5774 / 5806 为准** |

---

## 五、实验结果（核心对比）

### 5.1 论文口径 NMAE（全点 evaluate）

| 指标 | 非 PINN 5774 | PINN 5806 | 原文·无 PINN 消融 | 原文·主模型 |
| --- | ---: | ---: | ---: | ---: |
| **速度 NMAE** | **2.24 ± 0.96 %** | **2.35 ± 2.11 %** | 12.30 ± 3.79 % | 7.79 ± 2.14 % |
| **压力 NMAE** | **16.0 ± 9.0 %** | **13.2 ± 8.7 %** | 15.16 ± 7.20 % | **6.63 ± 2.80 %** |
| 四场 pooled 均值 | 5.67 % | 5.22 % | — | — |

**PINN vs 非 PINN 变化**：

| 指标 | PINN | 非 PINN | 变化 |
| --- | ---: | ---: | --- |
| 压力 NMAE | **13.2%** | 16.0% | **↓ 18%** |
| p R² | **−0.63** | −2.07 | **↑ 显著** |
| vel_mag NMAE | 5.39% | 4.75% | ↑ 略差 |
| 速度 NMAE | 2.35% | 2.24% | ≈ 持平 |

**分量 pooled NMAE**（10.27 亿点）：

| 分支 | u | v | w | p | \|v\| |
| --- | ---: | ---: | ---: | ---: | ---: |
| 非 PINN | 1.13 % | 1.88 % | 3.72 % | **15.9 %** | 4.75 % |
| PINN | 1.09 % | 3.20 % | 3.05 % | **13.6 %** | 5.39 % |

**逐样本压力 NMAE 分布**（非 PINN）：mean 16.0 % · median **15.4 %** · std **9.0 %**（病例间差异大）。

**典型病例**（非 PINN · 逐帧 p NMAE，对齐论文 mean±std 的逐样本口径）：

| 类型 | 病例 | 病例 81 帧均值 | 单帧极值 | 说明 |
| --- | --- | ---: | ---: | --- |
| 难例 | `BAI_WEN_JIE` | ~**38%** | 最高 ~**49%** | 合法 test OOD |
| 中等 | test  pooled | **16.0%** | median ~**15%** | 与 §5.1 一致 |
| 易帧 | `XI_CHENG_JIANG` / `LI_SHU_KUN` | ~**11–12%** | 最低 ~**1.6–1.7%** | 部分帧可学，病例级仍中等 |

### 5.2 点级 R²（全点 evaluate）

> **口径说明（与原文 / V3P 均不同，汇报时勿与 NMAE 主指标混为一谈）**
>
> - **数据来源**：仍是 CROWN 自己的 test 集——`point_filter=volume` 体素化 pkl（`ascii_in` 体点 · Job 5774/5806 `paper_full` 全点），**不是** V3P 的 features CSV / wall-rich 采样 evaluate。
> - **原文无此项**：Liao 2025 只报 **NMAE mean±std**（速度/压力），**不分 wall/interior、不报 R²**；下表 wall/interior 是本仓库 `evaluate.py` **额外补充的诊断指标**（见 README「点级 R²」）。
> - **分区判据**：与 CROWN 源码 / PINN 训练一致，用 **GT 速度** 代理壁面——`\|u\|²+\|v\|²+\|w\|² ≤ 0.01` → **wall**；其余体点 → **interior**。**不用** 项目 `is_wall`、不用 `ascii` 显式壁面网格（除非 `point_filter=all`）。
> - **语义**：wall 是体点云里「GT 速度近零的近壁体素点」，**不等于** V3P 的 STL 壁面点或 WSS 监督点；§九 1146 可视化才用到项目 features 点集做与 V3P 并排对照。

| 区域 | 场 | 非 PINN | PINN |
| --- | --- | ---: | ---: |
| **Pooled** | u | −0.04 | −0.11 |
| | v | −0.12 | −2.82 |
| | w | −0.07 | −0.16 |
| | **p** | **−2.07** | **−0.63** |
| | \|v\| | 0.06 | −0.62 |
| interior（\|v\|² **>** 0.01） | p | −3.13 | −1.06 |
| wall（\|v\|² **≤** 0.01 · GT 代理） | p | −1.81 | −0.56 |

**解读**：

- PINN 将 pooled p R² 从 −2.07 拉到 −0.63，物理项 **缓解** 了压力水平偏移，但 **未** 学到足够帧内空间结构。
- v 分量 R² 在 PINN 下 **更差**（−2.82 vs −0.12），wall loss / PDE 在 AG 边界条件下 **未有效协同**。
- 速度 NMAE 低但 R²≈0：**平均误差可接受，空间细节/梯度学不充分** → 不能后处理推 WSS。

---

## 六、同 split 方法对比（`split_AG_v1` seed1）

> **V3P 不是图网络（GNN）**：采用 **PointNeXt-style 点云**（`FieldPointNeXt`），在采样点 kNN 局部池化邻域学习。

| 方法 | 输入 | 架构 | p R² | 压力 NMAE | 主指标 | 任务重心 |
| --- | --- | --- | ---: | ---: | --- | --- |
| **CROWN 非 PINN** | 仅 x,y,z | PointNet + global max-pool | **−2.07** | 16.0% | NMAE | 外部 u,v,w,p 复现 |
| **CROWN PINN** | 仅 x,y,z + PINN | 同上 + NS/no-slip | **−0.63** | **13.2%** | NMAE | 外部 u,v,w,p 复现 |
| **PointNetCFD** | x,y,z + **t + BC** | 点云 | **0.897** | — | MAE 0.063 | 外部 baseline |
| **V3P AsymW-a** | 坐标+几何+BC+is_wall | PointNeXt 局部池化 · 双域 | **0.93–0.96** | — | **WSS R²≈0.40** | 项目主模型 |

**对比解读**：

1. 同 AG split、同标签源下 PointNetCFD（带 BC）p R²≈0.90 → 数据与 split **可学压力**，evaluate pipeline 可信。
2. CROWN 去掉 BC/时相/几何后压力崩溃；加 PINN **有改善但未达标** → 差距来自 **输入信息 + inductive bias + 跨数据集 BC 不匹配**。
3. V3P 在 **壁面 WSS** 上建立优势（~0.40），CROWN **不能**替代 V3P 做场重建或 WSS 对照。
4. 原文 FR/PD 病例级 R²>0.93 的叙事 **依赖可用压力场**；CROWN 当前压力不可用，此路径在 AG 上 **无法复现**。

### 6.1 与 V3P 的结构性差异

| 维度 | Liao 2025（CROWN） | V3P 主路线 |
| --- | --- | --- |
| 解剖 | ICA C3–C7 · 51 例 | AG 全脑血管 · 81 例 |
| 点云策略 | 体素均匀化 · 1 万点/步 | **wall-rich** · 壁面+近壁优先 |
| 节点输入 | 主要 xyz | xyz + 中心线几何 + is_wall |
| 全局 BC | 隐含在 CFD · **未进网络** | **显式 BC_Inlet + 出口压力** |
| 输出 | u,v,w,p | 壁面 **WSS+p** + 近壁 u,v,w,p |
| 物理约束 | PINN（主模型） | 数据驱动 + 非对称 WSS 损失等 |
| 主指标 | NMAE（v/p） | **WSS R²**（r2_p 为副指标） |

---

## 七、判读：复现的三层含义

| 层次 | 非 PINN | PINN | 依据 |
| --- | --- | --- | --- |
| **A. 实现复现** | ✅ | ✅ | 模型/训练/evaluate 与源码一致；全点 NMAE 口径；lazy/eager 逐值一致 |
| **B. 数值复现** | ⚠️ | ⚠️ | 跨数据集不可严格对标；非 PINN p NMAE≈原文消融；PINN 改善方向与原文一致 |
| **C. AG 可用性** | ❌ | ❌ | p R² 深负/仍负 · **No-Go** |

| 场 | 评价 |
| --- | --- |
| u, v, w | NMAE 低，R²≈0 或更差 → 空间结构/梯度学不充分 |
| p | ❌ 不可用（非 PINN −2.07 · PINN −0.63） |
| 整体 | **两条 paper-original 路线均结案 No-Go** |

---

## 八、根因分析

### 8.1 已确认的方法性原因

| # | 原因 | 机制 |
| --- | --- | --- |
| 1 | **无 BC / 时相** | 压力强依赖入口流量、出口压力、心动相位；export 有 `global_cond` 但未进网络 |
| 2 | **global max-pool** | 整帧 → 单向量，难表达病例级边界 + 帧级相位差异 |
| 3 | **PINN 增益有限** | 改善 NMAE/R² 但未达 6.63%；`Re=300` 固定 · `lphy≈0.011` 物理项仍弱 |
| 4 | **MSE 在归一化 p 上优化** | 不保证物理水平与帧间一致；R² 对水平偏移极敏感 |
| 5 | **均匀体素化 vs wall-rich** | 内部点占比高，近壁梯度样本相对 V3P 更少 |
| 6 | **壁面 no-slip 未有效** | 1146 可视化：PINN Pred \|v\| 近 **常数 ~0.044 m/s** → 不能推 WSS |

### 8.2 已排除的非原因

- denylist 病例泄漏 · pkl/ascii 标签不一致 · evaluate 实现错误
- 训练不充分（5751 105 epoch · 5757 91 epoch 均已早停至平台）
- subsample 评估偏差（5774/5806 全点已修正）

### 8.3 与原文现象的一致性

| 现象 | 原文 | 本项目 AG |
| --- | --- | --- |
| 无 PINN 压力最差 | 15.16% | **16.0%** |
| PINN 改善压力 | 6.63%（主模型） | **13.2%**（有改善但未达标） |

说明：**不是复现错了**，而是 paper-original 在 AG 上 **本身就不适合作为压力/WSS baseline**；要接近原文主模型，除 PINN 外还需 BC/几何等（→ `crown_geom_vp` 未开）。

---

## 九、可视化定性对比（1146 · 三病例）

与 V3P 并排：GUO / ZHANG / CHEN · 帧 `merged-1146`（`t_norm≈0.162`，收缩期上升段）· Gaussian r=3mm → STL。

> **口径说明**：本节 **单帧 R² / err_p** 在 **项目 features 壁面点** 上计算（与 V3P 同点集 · CROWN 全点 forward 后 NN 映射到壁面）；§9.1 中 CROWN pooled 数字仍来自 §5（CROWN 体点云 `paper_full`）。**V3P 与 CROWN 的 evaluate 点集不同**，并排表仅供定性，不作严格数值排名。

### 9.1 全局 test pooled（正文数字 · 与 §5 同源）

| 方法 | R²_p | R²_wss | vel_mag NMAE | evaluate 点集 |
| --- | ---: | ---: | ---: | --- |
| **V3P-I6-diag** | **0.930** | **0.429** | — | features · ~15000 点/帧 |
| CROWN 非 PINN | −2.07 | — | 4.75 % | 体点云 · 全点 10.27 亿 |
| CROWN PINN | −0.63 | — | 5.39 % | 体点云 · 全点 10.27 亿 |

### 9.2 单帧壁面压力（1146 · features 壁面点 · 与云图一致）

| 病例 | V3P R²_p | CROWN PINN R²_p | CROWN 非 PINN R²_p | PINN err_p p99 (Pa) |
| --- | ---: | ---: | ---: | ---: |
| GUO | **0.84** | −0.05 | −0.99 | **868** |
| ZHANG | **0.47** | −0.19 | −1.23 | **1246** |
| CHEN | **0.21** | −1.85 | −0.21 | **1171** |

GUO/ZHANG：PINN 误差云图 **较非 PINN 更收敛**；CHEN 单帧 PINN 更差（病例间方差大）。

### 9.3 误差云图 err_p p99（Pa · 非 PINN）

| 病例 | V3P | CROWN 非 PINN |
| --- | ---: | ---: |
| GUO | ~344 | ~**1597** |
| ZHANG | ~806 | ~**1935** |
| CHEN | ~559 | ~**839** |

CROWN **无 WSS 预测**；WSS 三联图仅 V3P 有。PINN 壁面 \|v\| Pred 近似常数 → **不能** 后处理推 WSS。

**路径**：[`V3P_vs_CROWN_对照表_1146.md`](../../../outputs/field/postview/V3P_vs_CROWN_对照表_1146.md) · [`V3P_vs_CROWN_PINN_对照表_1146.md`](../../../outputs/field/postview/V3P_vs_CROWN_PINN_对照表_1146.md)

---

## 十、工程化落地距离

| 阶段 | 要求 | CROWN 非 PINN | CROWN PINN | V3P（参考） |
| --- | --- | --- | --- | --- |
| **1. 场重建** | test p R² > 0.9 | ❌ −2.07 | ❌ −0.63 | ✅ ~0.96 |
| **2. WSS** | 壁面 WSS 可用 | ❌ | ❌（壁面 \|v\| 常数） | ⚠️ ~0.40 |
| **3. 病例级 surrogate** | FR/PD 等 | ❌ | ❌ | 🔶 G5 预研 |
| **4. 推理效率** | 可接受 | 全 test **181s**（1296 帧 · 16 病例） | 同左 | 图采样 1.5 万点/帧 |

**阶段判断**：paper-original 两条路线均处于「文献复现完成 → 方法边界已验证 → **不可用**」。若继续 CROWN，需 **`crown_geom_vp`（几何+BC）+ PINN + WSS 二阶段**，工作量接近重新立项。

---

## 十一、与 V3P 路线的关系（项目演进背景）

本项目 **并非一开始就做 WSS**：V1 最早与 CROWN 类似，主目标也是 **u,v,w,p 场重建**（V1 压力 R² 多数 ~0.92）。后续发现速度/压力场精度 **不等于** WSS 梯度质量，故 V3P 转向 **直接监督 WSS**。

CROWN 复现的价值：

- **外部文献对照**：验证「仅 xyz + max-pool（±PINN）」在 AG 上的边界；
- **排除选项**：不宜作为 V3P 的替代 baseline；
- **可借鉴**：PINN 物理约束、体素预处理思想（需与 wall-rich 采样权衡）；
- **不可假设**：速度/压力 NMAE 低 → WSS 可后处理推导。

---

## 十二、后续分支

| 分支 | 状态 | 说明 |
| --- | --- | --- |
| **paper-original 非 PINN** | ✅ **结案 No-Go** | Job 5751 + 5774 |
| **paper-original PINN** | ✅ **结案 No-Go** | Job 5757 + 5806 |
| **crown_geom_vp** | 未开 | xyz + 几何 + BC · 公平 baseline |
| **扩 seed / WSS 二阶段** | ❌ 不建议 | 压力未达标 |

---

## 十三、产物索引

| 类型 | 非 PINN | PINN |
| --- | --- | --- |
| 指标 JSON | `outputs/.../crown_original_vp_split_AG_v1_seed1_20260619_162738/metrics_test.json` | `outputs/.../crown_original_vp_pinn_split_AG_v1_seed1_20260620_152546/metrics_test.json` |
| 训练日志 | `cluster/logs/crown_5751.{out,err}` | `cluster/logs/crown_pinn_5757.{out,err}` |
| Evaluate 日志 | `cluster/logs/crown_eval_5774.out` | `cluster/logs/crown_eval_5806.out` |
| 实验分析记录 | [`crown_original_vp/实验分析记录.md`](crown_original_vp/实验分析记录.md) | [`crown_original_vp_pinn/实验分析记录.md`](crown_original_vp_pinn/实验分析记录.md) |
| 训练曲线 | [PNG](../docs/plots/crown_original_vp_train_val_loss_5751.png) | [PNG](../docs/plots/crown_original_vp_pinn_train_val_loss_5757.png) |

**共享文档**：[`CROWN训练与采样流程说明.md`](../docs/CROWN训练与采样流程说明.md) · [`5739_5740_根因诊断.md`](../docs/5739_5740_根因诊断.md) · [`CROWN_Beihang私有数据适配计划.md`](../../../docs/paper_reproduction/papers/hemodynamics_pointcloud_pinn/CROWN_Beihang私有数据适配计划.md)

---

*2026-06-25 · 非 PINN（5751+5774）+ PINN（5757+5806）· paper-original 正式结论合并汇报*
