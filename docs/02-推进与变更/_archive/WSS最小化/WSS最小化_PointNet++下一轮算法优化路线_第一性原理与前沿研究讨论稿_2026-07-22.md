# WSS 最小化 PointNet++ 下一轮算法优化路线：第一性原理与前沿研究讨论稿

> 日期：2026-07-22（对抗性审查修订版；2026-07-23 增补首轮实验证据）
> 状态：**ARCHIVED（2026-07-24）**；其中已执行结论已吸收进当前 S3-GEOPE 锚定路线，本文只保留历史论证与候选池
> 当前执行入口：[S3-GEOPE 锚定正则化与架构优化执行计划](../../WSS最小化_S3-GEOPE锚定_正则化与架构优化执行计划_2026-07-24.md)
> 范围：PointNet++ baseline、残差/PointNeXt 化、Drop 正则化、SEP 支持-查询解码、局部几何编码、低分辨率全局交互及后续表面/物理建模
> 修订说明：本版对初稿做了一次对抗性审查，逐条对照 `training_wss_min` 现有代码。核实结论与主要修正见 §0.5，可审计的逐条记录见附录 A。所有引用文献均已核验存在（LitePT CVPR 2026 = arXiv:2512.13689；Sci. Rep. `s41598-026-47410-z`；npj Digit. Med. `s41746-026-02404-z`）。

## 0. 结论先行

下一轮不建议直接做“完整 PointNeXt + 大 Drop + SEP”的组合实验。当前证据更支持以下顺序：

1. **先确认父模型与噪声地板**：在新的独立分层划分上，用 3 个配对种子复核 Q1V 与 D2，并首次估计**总体指标的种子间标准差**。D2 的总体点估计略高，但病例级置信区间跨 0，尚不足以宣布其稳定优于 Q1V；在拿到种子方差前，任何小数点后四位的排序都不可信。
2. **先做零成本的“已备而未用特征”实验**：`radius_gradient`、`coord_scale` 已在 `FEATURE_KEYS` 中就绪，但当前 6D 主配置未使用。把它们作为逐点输入加入，是成本最低、无需改网络的第一步（详见 §6.1、§12 的 `M-FEAT`）。
3. **再加入真正的局部残差 PointNeXt 核心，不加 Drop**：保留已验证的采样、分组、损失和解码，只把固定支持 baseline 的 SA 内部单次 MLP-max 改为轻量残差局部聚合。注意：残差块本身（`InvResMLP`）**已存在于 `pointnext.py`**，本轮工作不是从零实现，而是把它移植进固定中心/支持-查询协议、加入残差缩放并调小 expansion（详见 §4）。
4. **再做三个正交模块的单变量筛选**：局部几何位置编码、仅在最粗层使用的轻量注意力、受约束的 SEP 插值核。三者不要首轮组合。
5. **Drop 是条件分支，不是默认增益项**：只有在残差网络出现训练-验证间隙、种子方差或过平滑证据后，才分别测试 head dropout、DropPath、保覆盖的 NeighborDrop。
6. **如果结构改造仍只带来约 0.01 以内的收益，应转向信息瓶颈**：补充壁面法向/中心线切向/分支结构，或引入表面网格分支与可部署边界条件；继续单纯扩大邻域、点数、宽度的回报已很低。

我建议把下一代模型暂称为 **WSS-PNX-Hybrid-S**：`PointNeXt-R 局部残差编码器 + SA3 粗层全局交互 + 保守 SEP 插值`。其中每一部分必须先单独过门，最后才允许组合。

---

## 0.1 2026-07-23 首轮实验反馈：原路线哪些成立、哪些被推翻

D2 c125×k64 的 ILO 两协议和 mixed test36 结构矩阵已经按本文“单变量、先不加 Drop”的原则完成，9 个新 run 全部为 seed1234。结果对本文假设的更新如下：

| 原假设/模块 | 严格配对结果 | 对路线的更新 |
|---|---|---|
| ILO 直接增训可能扩大通用数据覆盖 | fixed test27 F1−F0 `ΔR²_cb=-0.0169`；mixed test36 M1−M0 `-0.0059`，ILO `+0.0109` 但 AAA `-0.0331` | **不成立为总体增益**；按域间容量权衡归档 |
| M-FEAT 是低成本优先项 | S1−M1 `+0.0018`，case-mean `-0.0137`，病例 CI 跨零 | 成本低但收益不成立，停止扩展 |
| PointNeXt-R core 可独立提升 | S2−M1 `-0.0086`，病例均值差95% CI `[-0.0559,-0.0025]` | **裸残差核心 No-Go 信号**；“PointNeXt”不能脱离几何编码单独宣传 |
| LocalGeoPE 能补回局部几何表达 | S3−S2 `R²_cb +0.0396`、MAE `-0.1708 Pa`、high-WSS `+0.0548`、IoU `+0.0157`；29胜7负，病例均值差95% CI `[+0.0600,+0.1259]` | **本轮最强支持**；S3 进入三种子确认 |
| 粗层 attention 是低成本全局交互 | S4−S2 `+0.0178`，但病例 19胜17负、CI 跨零 | 保留二级弱正候选，不先组合 |
| 受约束 SEP 优于固定3NN | S5−S5C aggregate `+0.0235`，但 case-mean `-0.0047`、负例 `3→7`、病例 CI 跨零 | 仅条件正候选；必须保留 independent-query 匹配控制并补多种子 |
| Drop 只由证据触发 | 本轮未加入 Drop；S3 已出现强表示增益，但尚无多种子方差证据 | 原判断保持：先确认 S3，再单独测小 `DropPath/NeighborDrop` |

因此，本文原先设想的 `PointNeXt-R + attention + SEP` 混合终态应暂缓。当前更精确的候选是 **D2-K64 + PointNeXt-R + LocalGeoPE**；关键贡献来自 LocalGeoPE 对残差核心的补偿，而不是 PointNeXt-R 本身。三种子结果为：S2−M1 mean ΔR²_cb=`+0.0086±0.0221`但不稳定；S3−S2=`+0.0138±0.0227`、2/3正且 MAE三 seed全降；S3−M1=`+0.0224±0.0143`、3/3正。S3 通过，只进入 `DropPath 0.05/0.10` 或 `NeighborDrop 0.05` 的互不组合单变量计划；S4、SEP 与 S3 的组合仍排在其后。

需要保留的物理限制没有改变：S3 的 high-WSS R² 仍为 `-0.4495`，只是从 S2 的 `-0.5043` 改善，不能称为高 WSS 已解决。完整结果见 [训练实验跟踪](../../WSS最小化_训练实验跟踪.md) 和 [D2-K64 终审与执行结果](WSS最小化_D2-c125-k64_ILO两协议对照_终审与执行计划_2026-07-23.md)。

---

## 0.5 本次对抗性审查的关键修正

以下修正均以现有代码为依据，改变了初稿的技术判断或实验设计，不只是措辞润色：

| # | 初稿说法 | 修正后判断 | 代码依据 |
|---|---|---|---|
| 1 | 朴素 QAD 失败是因为“破坏了 3-NN 插值的同点恒等先验” | **诊断不准**。`qad_lite` 残差已零初始化并在 query=support 时走显式 identity，初始与父模型逐位一致，却仍退化 `-0.01769`。真正失败源是**训练后自由无界的加性残差学到了平均修正**。这改变了 SEP-Kernel 的立论：它的价值不是“初始不伤害”（QAD 也做到了），而是**把假设空间约束为凸插值权重，训练后也无法无界外推** | `baseline_models.py:456-468`（零初始化 467-468）、`baseline_models.py:495-508`（identity 短路） |
| 2 | SEP-Kernel 用 `softmax(-d/T)`，零初始化后“等价于逆距离/温度插值” | **公式错误**。PyG `knn_interpolate` 用逆距离平方 `1/d²` 加权，`softmax(-d/T)` 不等于它。要使初始严格等于父插值，基线 logit 必须是 `-β·ln d`（β 初始=2），详见 §7.2 | `baseline_models.py:499-500`、`config.py:121`（`fp_knn=3`） |
| 3 | “引入轻量 PointNeXt-R block” | **组件已存在**。`InvResMLP`（倒置瓶颈+残差，expansion 默认 4）已在 `pointnext.py`，且 `sa_blocks` 已被 `PointNeXtSeg` 消费——只是**没有被 `pointnetpp` baseline 消费**。真正的增量是：移植进固定支持路径、加残差缩放 γ、expansion 4→2、让 `pointnetpp` 真正读取 `sa_blocks` 并加断言 | `pointnext.py:65-83,101-128`；`models.py:22-42`；`baseline_models.py:552-562`（不传 `sa_blocks`） |
| 4 | §14 单测“随机刚体旋转下标量输出一致性” | **架构上不成立**。模型吃绝对 xyz 且把原始 Δx 送进 MLP，本身不是旋转不变，仅靠 `rot_aug` 近似。此测试要么放宽为“近似不变（容差匹配增广强度）”，要么只对“仅用不变特征”的变体做严格测试。由此引出一个有物理动机的设计决策（§6.5） | `config.py:77`（默认 `input_features`）、`config.py:75`（`rot_aug`）、`baseline_models.py:396`（原始 Δx 入 MLP） |
| 5 | 输入“仅 6D，缺方向/尺度信息” | **部分已就绪**。`radius_gradient`（扩张/收缩）、`coord_scale`（病例尺度）、`cohort_*`（疾病域 one-hot）已在 `FEATURE_KEYS`。补方向信息前，应先用这些零成本特征做对照 | `config.py:29-34` |
| 6 | Go/No-Go 阈值 `+0.012/+0.015`；五项误差“分解” | 阈值未与**实测种子方差/最小可检测效应**挂钩，且单种子筛选门几乎与确认门等高（召回过低）；五项误差非正交，应称“误差归类”而非可加分解。见 §2.2、§13 | 需 A0-3 首次测量种子 SD |

---

## 1. 当前证据：下一轮优化必须解释什么

### 1.1 已完成实验给出的事实

| 观察 | 结果 | 对下一轮的约束 |
|---|---:|---|
| Q1V：random5000 + ball16 | physical `R²_cb = 0.276310` | 当前强基线之一 |
| D2：c125 × k128 | physical `R²_cb = 0.276512` | 仅比 Q1V 高 `+0.000202`，不能据此宣称稳定更优 |
| D2 对 Q1V 病例级差值 | mean `ΔR² = +0.00307`，bootstrap 95% CI `[-0.02174, +0.02648]`，15/27 胜，Wilcoxon `p=0.7007` | D2 更适合保留为候选父模型，而非新默认基线 |
| D2 高 WSS | `R² = -0.504015`，Q1V 为 `-0.513433` | 局部改善，但高值区依然没有被可靠解释 |
| 10k/all points + 大 k | 低于 Q1V | 简单增加采样密度和邻域规模不是主线 |
| width 32→64（同分组） | KNN8：val `R²_cb` 由 `0.2589` 降到 `0.2183`；KNN10：由 `0.2425` 降到 `0.2172` | 加宽在当前样本量下净退化（过拟合/优化风险），非增益轴 |
| D5-A 深 stem | 相对其 w64 父模型改善，但仍低于 w32 | 不能把“更深”当作独立充分条件 |
| bridge SAME | `0.2439` | 支持-查询桥接未自然恢复主模型性能 |
| bridge SEP | `0.2329` | 当前朴素 SEP 为 No-Go，不应直接重复 |
| QAD（`qad_lite`，Q2V），3 seeds | physical 平均 `Δ=-0.01769`，normalized 平均 `Δ=-0.00476` | **注意：QAD 残差已零初始化+identity 短路，初始与父模型等价，仍退化**——问题在训练后的自由残差，不在初始先验（见 §7.1） |
| raw-Huber / tail weighting | 局部可能改善，但种子敏感、组合曾退化 | 在架构稳定前不作为主线 |

> §1.1 中的 width 行已澄清：`0.2589→0.2183` 指 **width 从 32 增到 64 后，同一分组配置下验证 `R²_cb` 的下降**，是净退化而非收敛过程。若该口径与实验矩阵原表不一致，以 `WSS_PointNet实验矩阵与结果汇总last.xlsx` 为准并回填本表。

最重要的判断不是“D2 比 Q1V 高 0.0002”，而是：**增加局部覆盖范围只能移动很小的总体指标，高 WSS 仍为负，且简单的查询残差、宽度和 SEP 均没有兑现预期。** 这说明剩余误差不只来自模型容量，更可能来自几何表达、缺失条件、局部—全局耦合和目标定义。

### 1.2 当前代码结构中的关键限制

| 当前实现 | 限制 | 可验证改造 |
|---|---|---|
| SA：`[相对坐标, 邻点特征] → 2 层 MLP → max`（`baseline_models.py:352,396-398`） | 单次局部变换，没有同分辨率残差堆叠 | 移植 `InvResMLP` 式残差块进固定支持 SA |
| `_mlp = Linear + BN + ReLU`（`baseline_models.py:28-36`） | 缺少 inverted bottleneck、残差缩放、可控深度 | expansion=2，残差尺度小值/零初始化 |
| `sa_blocks` 只被 `pointnext_s` 消费，**未被 `pointnetpp` baseline 消费**（`models.py:22-42` vs `baseline_models.py:552-562`） | 对 `pointnetpp` 改 `sa_blocks` 会被**静默忽略**，不报错 | 先做 forward trace + 断言；让 `pointnetpp` 真正读取该字段 |
| dropout 主要在最终 head（`config.py:127`，`baseline_models.py:453`） | 无法正则化中间局部表征 | 仅在确有过拟合时测试 DropPath/NeighborDrop |
| 默认 query decoder 为 3-NN 逆距离平方插值（`config.py:121`，PyG `knn_interpolate`） | 对同点查询有强恒等先验，但对独立 query 较僵硬 | 设计受约束、从逆距离平方初始化的 SEP-Kernel（§7.2） |
| 主配置输入 6D：xyz、`abscissa_norm`、`local_radius`、`curvature`；而 `radius_gradient`、`coord_scale`、`cohort_*` **已在 `FEATURE_KEYS` 但未使用**（`config.py:29-34`） | 已备特征未用；仍缺壁面法向、中心线切向、分支关系、边界条件 | 先零成本加入已备特征，再做可部署的局部几何/拓扑特征 |
| 预处理可保留 `wall_wss_vec`（`preprocess.py:504`，受 `store_wall_wss_vector` 开关） | 当前主要优化标量 WSS，方向信息未利用；`out_dim=3` 矢量为二期占位（`config.py:126`） | 经向量坐标 QA 后做向量辅助头 |

---

## 2. 第一性原理：WSS 误差到底从哪里来

### 2.1 WSS 不是纯局部形状函数

壁面切应力是壁面 traction 的切向分量：

$$
\boldsymbol{\tau}_w=(\mathbf I-\mathbf n\mathbf n^\top)\,\boldsymbol{\sigma}\mathbf n,
\qquad
\boldsymbol{\sigma}=-p\mathbf I+\mu(\nabla\mathbf u+\nabla\mathbf u^\top).
$$

一个常被忽略、但对“该补哪种条件”有直接影响的事实：**切向投影 $(\mathbf I-\mathbf n\mathbf n^\top)$ 把压力项消掉了**——因为 $(\mathbf I-\mathbf n\mathbf n^\top)(-p\mathbf I)\mathbf n=-p(\mathbf n-\mathbf n)=\mathbf 0$。所以 WSS 是**纯黏性/近壁速度梯度**量：

$$
\boldsymbol{\tau}_w=\mu\,(\mathbf I-\mathbf n\mathbf n^\top)(\nabla\mathbf u+\nabla\mathbf u^\top)\mathbf n .
$$

推论：压力、出口 RCR、入口阻抗**不会**直接进入 WSS，只通过它们塑造的近壁速度场间接起作用。因此在“补缺失条件”时，优先级应是**能改变近壁速度剖面的几何/流量信息**（入口流量波形、分支分流比、壁面法向与曲率、上下游管径变化），而不是压力真值本身。这也再次说明为什么把 CFD 压力当输入既是泄漏又低效（§10.2、§15）。

真实映射更接近：

$$
\boldsymbol{\tau}_w=F(\text{三维几何},\text{入口/分支流量},\mu,\text{时间/相位},\text{全局流路}).
$$

当前模型看到的主要是采样后的局部点云几何。若多个病例具有近似局部形状，但入口流量、分支阻抗或相位不同，那么在均方误差下，网络只能逼近条件均值 $\hat y(x)=\mathbb E[y\mid x]$，从而自然压平峰值。换句话说，高 WSS 预测差不一定全是“网络不够强”，也可能是输入不能唯一确定标签。

### 2.2 用五项误差归类组织所有实验（是分类框架，不是可加分解）

下式是**便于归因的分类清单，并非严格正交的可加分解**（各项相互耦合，不应按字面相加）：

$$
E_{total}\;\longleftarrow\;\{E_{missing\ info},\,E_{representation},\,E_{sampling},\,E_{optimization},\,E_{objective}\}.
$$

- `E_missing info`：入口/分支流量、时间相位、法向/切向或分支结构缺失。
- `E_representation`：局部 MLP-max 无法表达方向性、多尺度和长程耦合。
- `E_sampling`：有限支持点、支持/查询不一致、插值误差。
- `E_optimization`：深/宽网络训练不稳、过拟合、过平滑或梯度传播差。
- `E_objective`：整体 MSE 与高 WSS、热点排序、病例公平性不一致。

任何新模块都必须声明它主要针对哪一项；否则多个模块同时加入后，结果无法归因。

### 2.3 当前结果最可能对应的瓶颈排序

1. **几何表达与缺失条件**：高 WSS 长期为负，且增加点数/邻域未解决。
2. **局部—全局耦合**：局部动量变化受上游、分支和管径变化共同影响。
3. **支持—查询解码偏差**：SAME 明显优于朴素 SEP，说明查询恢复仍是难点。
4. **优化稳定性**：w64 退化，简单加深也未超越 w32。
5. **目标函数错配**：存在，但历史 tail/raw-Huber 结果说明它不是应最先动的旋钮。

---

## 3. 前沿研究带来的可迁移启发

这里关注“能否转化成受控实验”，不按论文榜单照搬整套模型。全部链接已核验存在。

| 研究方向 | 关键启发 | 对本项目的可迁移方案 | 当前优先级 |
|---|---|---|---:|
| [PointNeXt, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9318763d049edf9a1f2779b2a59911d3-Abstract-Conference.html) | PointNet++ 的训练与残差/InvResMLP 设计可以显著现代化 | 在现有 SA 内做同分辨率轻量残差，不先换采样协议 | 最高 |
| [Point Transformer, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Point_Transformer_ICCV_2021_paper.html) | 相对位置编码和邻域内自适应聚合优于固定 max 的表达力 | 先做便宜的 LocalGeoPE；之后再测局部权重聚合 | 高 |
| [Point Transformer V3, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html) | 序列化邻域可扩大感受野并提高效率 | 数据量扩大后可考虑；当前不宜先承担序列化复杂度 | 低 |
| [LitePT, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yue_LitePT_Lighter_Yet_Stronger_Point_Transformer_CVPR_2026_paper.html)（arXiv:2512.13689） | ①高分辨率早期层用局部算子、注意力放到低分辨率深层；②参数无关的相对位置编码 PointROPE 保住空间布局 | 仅在 32 个 SA3 centers 上放 1 个注意力块；相对位置编码思路直接支撑 §6 的 LocalGeoPE | 高 |
| [Point Cloud Mamba, 2024](https://arxiv.org/abs/2403.00762) | 序列状态空间模型可线性复杂度建模全局关系 | 需要可靠排序/序列化和更多数据，作为后续扩展 | 低 |
| [DiffusionNet, 2020](https://arxiv.org/abs/2012.00888) | 表面内在扩散对重网格和采样变化更稳健 | 构建表面网格分支，补足欧氏点云近邻的拓扑错误 | 中长期 |
| [SE(3)-Equivariant Mesh WSS, 2022](https://arxiv.org/abs/2212.05023) | 表面三角网格、等变特征和边界条件适合 WSS 向量预测 | 在法向/向量 QA 后做 mesh/vector 辅助路径 | 中长期 |
| [PI-GNN WSS, Scientific Reports 2026](https://www.nature.com/articles/s41598-026-47410-z) | 混合几何图边、残差 GNN 和物理软约束可共同建模 WSS（狭窄冠脉，1000 合成几何） | 借鉴“几何边 + 物理 guardrail”，但其数据规模远大于本项目，数值不可横比 | 中长期 |
| [Physics-constrained aneurysm GNN, npj Digital Medicine 2026](https://www.nature.com/articles/s41746-026-02404-z) | 节点特征、边界条件和物理约束对时变场预测重要 | 支持“先补可部署条件，再扩大网络”的判断 | 中长期 |
| [GINO, 2023](https://arxiv.org/abs/2309.00583) | 局部几何图与全局算子耦合可处理任意几何 | 多边界条件/多时间任务后再考虑 operator 路线 | 低 |
| [Transolver, 2024](https://arxiv.org/abs/2402.02366) | 物理感知 token 可高效建模全局 PDE 关联 | 数据规模和任务覆盖扩大后研究 | 低 |
| [Stochastic Depth, 2016](https://arxiv.org/abs/1603.09382) | 残差分支可按深度随机丢弃，提高深网训练与泛化 | 只有形成真正残差栈后，DropPath 才有明确对象 | 条件高 |
| [DropEdge, ICLR 2020](https://openreview.net/forum?id=J3scyJneWcE) | 随机丢边可缓解图网络过平滑 | 仅丢重复/次级边，并保持每个中心的覆盖边（本项目已有覆盖保障机制，见 §5、§8） | 条件中 |
| [Fourier Features, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html) | 高频位置编码有助于坐标回归中的高频函数 | 仅编码相对弧长/分支距离，不直接编码绝对 xyz | 中 |
| [Vector Neurons, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Deng_Vector_Neurons_A_General_Framework_for_SO3-Equivariant_Networks_ICCV_2021_paper.html) | 向量通道可保持旋转等变 | 适合 WSS 向量辅助头，也是 §6.5 “严格旋转不变/等变”路线的候选实现 | 中长期 |

近期工作的共同信号是：**局部几何编码、残差稳定性、低分辨率全局交互、边界/物理信息，通常比单纯把所有层做宽或把 attention 铺满高分辨率点更有性价比。**

---

## 4. 路线 A：PointNeXt-R 局部残差核心——下一轮主线

### 4.1 为什么它比继续加宽更合理

当前 SA 中的信息路径很短：邻域特征一次映射后立刻 max pool。增加宽度只是让这次映射更大，并不能增加“局部关系反复精炼”的次数。残差块则把表示写成：

$$
\mathbf h^{l+1}=\mathbf h^l+\gamma_l\,F_l(\mathbf h^l,\mathcal N),
$$

其中 `γ_l` 从 0 或 `1e-3` 初始化。这样初始模型接近恒等映射，新分支只在数据支持时逐渐贡献，尤其适合当前小样本和 w64 已退化的情形。

> 与现有代码的关系（重要）：`pointnext.py:65-83` 的 `InvResMLP` **已经是**“ball-query 深度聚合 + 倒置瓶颈 pointwise + 残差”的块，`PointNeXtSeg` 也**已经**按 `sa_blocks` 堆叠它（`pointnext.py:120-128`）。但它有两个不符合本轮设计的点：①残差是 `act(x + F(x))` 形式，**没有可学习残差缩放 γ、没有零/小初始化**（`pointnext.py:81-82`）；②它挂在 ratio-SA + ball-query + 无支持/查询解码的 `PointNeXtSeg` 上，而不是当前胜出的固定中心/固定支持 `PointNetPlusPlusRegressor`。所以本轮不是“新写一个残差块”，而是**把已验证的块移植进固定支持路径，并补上 γ 与 expansion 调整**。

### 4.2 最小可归因实现

建议首版只做：

- 父模型：由新划分上的 Q1V/D2 复核决定；若两者仍无显著差异，优先 Q1V，因为计算更省、协议更简单。
- 保持 center 数、邻域选择、输入、loss、FP/query decoder 不变。
- `sa_blocks=(1,1,0)`：SA1、SA2 各增加 1 个同分辨率 residual local block，SA3 不堆局部块。**前置条件**：先让 `pointnetpp` 真正消费 `sa_blocks`（当前只有 `pointnext_s` 消费），并在 A0-2 用 forward trace 断言执行图确实改变，否则会“改了配置但模型没变”。
- pointwise inverted bottleneck expansion=`2`（现有 `InvResMLP` 默认 `4`，首版调小）。
- 激活可先沿用 ReLU；Norm 首轮沿用 BN，避免同时变更多个训练因素（注意 `InvResMLP` 现用 GELU，移植时先统一为 baseline 的 ReLU/BN）。
- residual scale `γ=1e-3` 或零初始化末层（现有块无此项，需新增）。
- 参数量目标 `<0.45M`，FLOPs/step 不超过父模型约 2 倍；父模型实际参数量由 A0-2 测得后写入，不凭空设限。
- `dropout=0`、`drop_path=0`，先回答残差核心是否有效。

这应命名为 **PointNeXt-R core**，而不是直接宣称“完整 PointNeXt”。不宜把 `pointnext.py` 整模型替换 baseline，因为它的 ratio-SA、ball-query、层级与当前固定中心/固定支持协议不同，否则无法知道收益来自哪里。

### 4.3 必须记录的机制指标

- 每层实际 block 数、参数量、激活尺寸。
- residual/main 分支输出范数比（验证 γ 确实被学到非零，块没有“形同虚设”）。
- best-last gap、train-val gap、梯度范数。
- SA1/SA2 特征余弦相似度和有效秩，判断是否过平滑。
- 3 个配对种子的总体、病例级和高 WSS 指标。

---

## 5. 路线 B：Drop 的正确用法——条件式正则，而不是“加了就会更好”

### 5.1 Drop 要解决哪一种失败

| 现象 | 对应 Drop | 不应使用的情形 |
|---|---|---|
| head 明显过拟合，train 好而 val 差 | head dropout | train/val 都欠拟合 |
| 残差栈变深后种子波动、路径共适应 | DropPath / stochastic depth | 尚无残差块时 |
| 邻域冗余高、特征快速同质化 | NeighborDrop / DropEdge | 邻域本就稀疏或覆盖不足 |
| 输入点采样过度固定 | point dropout | 当前 random5000 本身已有较强随机采样时 |

### 5.2 推荐的三个独立实验

1. `REG-H`：只在最终 head 使用 dropout=`0.1`。
2. `REG-P`：PointNeXt-R 残差块采用线性递增 DropPath，最大值=`0.05`。
3. `REG-E`：邻域边丢弃率=`0.05`，仅允许丢重复或次级邻边；每个中心保留 coverage 主边和最小邻居数。实现上可直接复用现有覆盖保障机制——`_knn_cover_group`/`_adaptive_cover_group` 已保证每个 source 有主归属、100% 覆盖（`baseline_models.py:166-179,182-193`），只在“次级边”上做丢弃即可。

首轮不要把三者组合，也不要直接用 `0.3–0.5` 的视觉任务常见 dropout。当前模型与数据规模下，大 Drop 更可能进一步损失高 WSS 稀有模式。

### 5.3 DropPath 的启动条件

满足以下至少一项才启动：

- PointNeXt-R 的 train-val `R²` gap 明显高于父模型；
- 三种子标准差上升；
- 后层特征有效秩下降且平均余弦相似度异常升高；
- 加 residual blocks 后 best epoch 明显提前、last checkpoint 回退。

如果 PointNeXt-R 本身欠拟合，Drop 不是修复方向。

---

## 6. 路线 C：LocalGeoPE——比绝对坐标更贴近血管物理

### 6.0 先用“已备而未用”的逐点特征（零成本）

在设计任何相对编码之前，先做最便宜的一步：`FEATURE_KEYS` 中已有但主配置未用的 `radius_gradient`（沿中心线的管径变化，编码扩张/收缩）与 `coord_scale`（病例尺度）可直接作为逐点输入加入，无需改网络（对应 §12 的 `M-FEAT`）。`cohort_*` one-hot 也可作为疾病域条件化输入。若这些已备特征就能移动指标，则说明瓶颈部分是“输入未用满”而非“表示不足”，可先廉价获益再决定是否上相对编码。

### 6.1 相对几何首版特征

对中心 `i` 与邻点 `j`，构建：

$$
e_{ij}=[
\underbrace{\Delta\mathbf x_{ij}/r_i}_{\text{方向，旋转下会变}},
\underbrace{\|\Delta\mathbf x_{ij}\|/r_i}_{\text{不变}},
\underbrace{\Delta s_{ij}/L}_{\text{不变}},
\underbrace{\log(r_j/r_i)}_{\text{不变}},
\underbrace{\kappa_j-\kappa_i}_{\text{不变}}
].
$$

- `Δx/r_i`：尺度归一化后的局部方向（**注意：该分量在刚体旋转下会随之旋转，是唯一破坏旋转不变性的项，见 §6.5**）。
- 距离：帮助网络区分邻域中心与边缘。
- `Δs/L`：沿中心线的相对上下游关系。
- `log(r_j/r_i)`：扩张/收缩趋势（与已备的逐点 `radius_gradient` 互补，一个是边量、一个是点量）。
- 曲率差：弯曲变化而非单点曲率绝对量。

这些编码进入局部消息函数或注意力 bias，不取代原始输入。

### 6.2 第二版才增加的特征

- 壁面法向 `n`、中心线切向 `t`；
- 局部坐标：轴向、径向、环向分量；
- 分支 ID、到分叉距离、父子分支方向；
- Fourier features，但仅作用于相对弧长/分支距离。

不建议直接对全局绝对 xyz 做高频 Fourier 编码：这容易学习数据集的摆放方向和坐标范围，而非旋转稳健的血流规律。

### 6.3 前置 QA

- 法向是否统一朝外；随机抽样检查符号翻转率。
- 中心线切向是否连续，分叉处如何定义父支方向。
- 壁面点到中心线/分支的映射是否稳定。

---

## 6.5 旋转不变性：一个需要先定的设计决策

初稿在 §14 提出“随机刚体旋转下标量输出一致性”单测，但**当前架构根本不是旋转不变的**：模型吃绝对 `x,y,z`（`config.py:77`），且 SA 把原始相对坐标 `Δx` 直接送进 MLP（`baseline_models.py:396`），MLP 对旋转不等变。现状只靠训练期 `rot_aug`（`config.py:75`）获得**近似**不变。因此一个“严格一致”的单测在现架构上必然失败，不能作为无条件断言。

这引出一个有物理动机的选择：**标量 WSS 幅值场对几何刚体旋转是严格不变的，向量 WSS 是等变的**（旋转几何只旋转向量场，对应点的幅值不变）。所以有两条路：

1. **严格不变路线（推荐评估）**：网络只吃旋转不变量（`abscissa_norm`、`local_radius`、`curvature`、`radius_gradient`、以及 §6.1 中除方向项外的边特征），并在需要方向时使用**局部等变标架**（壁面法向 `n`、中心线切向 `t`）的内积分量代替原始 `Δx`。这样可获得**免费的严格旋转不变**，同时消除“数据集摆放方向”这一泄漏通道。代价是需先做一次消融：去掉绝对 xyz、去掉原始 `Δx` 方向项后精度是否下降。
2. **近似不变路线（现状）**：保留绝对 xyz + 原始 `Δx`，靠 `rot_aug` 近似不变。此时单测只能是**容差匹配增广强度的“近似一致”检查**，不能是严格相等。

建议在 Stage 0 增加一个消融（§12 的 `A0-6`）先量化“移除绝对 xyz/方向项”的代价，再决定走哪条路；不要在未定此事前就把严格不变写进验收单测。

---

## 7. 路线 D：SEP 不是再加一个输出残差，而是约束查询算子

### 7.1 为什么已有 SEP/QAD 失败——修正后的诊断

现有证据表明，朴素 bridge SEP 和查询端 QAD 残差都退化。**这里要纠正初稿的一个误判**：初稿说 QAD 失败是因为“破坏了 3-NN 插值的同点恒等先验”。但代码里的 `qad_lite`：

- 残差末层**零初始化**（`baseline_models.py:467-468`）；
- query=support 时走**显式 identity 短路**（`baseline_models.py:504-508`）。

所以 QAD **在初始时刻与父模型逐位一致，并没有破坏恒等先验**，却仍退化 `-0.01769`。真正的失败机制是：

- 训练后，**自由、无界的加性残差**倾向学到“平均修正”而非热点恢复（高 WSS 稀少，MSE 下均值修正更划算）；
- 支持点编码误差被一个自由度过高的查询模块放大；
- 独立 query 的分布偏移没有被训练目标充分约束。

因此下一版 SEP 的立论**不是**“初始不伤害父模型”（`qad_lite` 已经做到这点仍失败），而应是：**把可学习空间限制在‘凸插值权重’上，使得即使训练收敛后也无法做无界外推或无约束的加性修正。**

### 7.2 SEP-Kernel：保守的可学习凸插值（修正初始化公式）

定义支持邻域权重：

$$
\alpha_{qj}=\operatorname{softmax}_j
\Big(\underbrace{-\beta\,\ln d_{qj}}_{\text{基线 logit}}+\;g_\theta(e_{qj})\Big),
\qquad
\hat h_q=\sum_{j\in\mathcal N(q)}\alpha_{qj}\,h_j.
$$

**关键修正**：初稿用 `-d_{qj}/T` 作基线 logit，声称零初始化后等价于逆距离插值——这是错的。PyG `knn_interpolate` 用的是**逆距离平方** `1/d²` 归一化权重。要让 SEP-Kernel 在 `g_θ=0` 时**严格等于**父模型的 3-NN 插值，基线 logit 必须是 `-β ln d_{qj}` 且 `β=2`：此时

$$
\alpha_{qj}\big|_{g_\theta=0}=\frac{e^{-2\ln d_{qj}}}{\sum_m e^{-2\ln d_{qm}}}=\frac{d_{qj}^{-2}}{\sum_m d_{qm}^{-2}},
$$

即逐位复现 `knn_interpolate`。把 `β` 设为可学习标量（初值 2、约束 `β≥0`）就得到“可调逆距离幂次”的推广。

约束：

- `α_qj ≥ 0` 且逐 query 和为 1，避免无界外推（这是相对 QAD 的**本质区别**）。
- `g_θ` 末层零初始化 + `β=2`，使模型初始严格等价于父 3-NN 逆距离平方插值。
- query 与 support 重合时 `d→0`，`-β ln d→+∞`，softmax 自动把全部权重给到重合点（配合 PyG 式距离下限 clamp），identity 自然成立，无需额外分支。
- 首轮只学习权重，不新增任意输出残差。
- 训练中混合 SAME 和 SEP query，使 identity 与独立查询同时被监督。
- 对权重熵、最近邻权重、有效邻居数做诊断。

### 7.3 SEP 的 Go/No-Go 额外条件

- SAME 模式相对父模型 `ΔR² ≥ -0.005`；
- query=support 的数值误差小于设定容差（并有 §14 的初始等价单测背书）；
- 独立 query 改善不能以高 WSS 和热点 IoU 系统性下降为代价；
- 权重不能长期塌缩成单一异常远邻，也不能始终退回固定均匀平均。

---

## 8. 路线 E：邻域与聚合优化——停止继续盲扫 k

### 8.1 不再把单一 k 当作主要搜索轴

D2 已接近全重叠邻域，继续增加 k 的新增信息很有限。更合理的是让邻域包含不同物理含义：

- `N_euclid`：局部欧氏邻域；
- `N_surface`：表面网格测地邻域，避免血管两侧空间接近但表面不连通；
- `N_axial`：沿中心线上游/下游邻域；
- `N_branch`：分叉相关邻域。

首版可以不构建复杂图，只在已有邻居中增加轴向符号和半径比（后者与已备特征 `radius_gradient` 一致）；若有效，再升级为混合边图。注意现有 `adaptive_cover` 分组已提供 100% 覆盖 + 每对中心 ≤1/3 重叠的保障（`config.py:318`，`baseline_models.py:182-193`），新邻域设计应保持这一覆盖不变量。

### 8.2 Gated max + mean pooling

max 擅长保留局部极值，但梯度稀疏；mean 稳定却容易平滑热点。可测试：

$$
h_i=\lambda_i\,\operatorname{max}_j m_{ij}
+(1-\lambda_i)\,\operatorname{mean}_j m_{ij},
$$

其中 `λ_i=sigmoid(a_i)`。该项排在 LocalGeoPE、粗层 attention 和 SEP-Kernel 之后，只有前三者均无效时才值得单独筛选。

---

## 9. 路线 F：只在粗层做全局交互

### 9.1 LitePT 式分工

高分辨率层负责局部几何，最低分辨率层负责跨区域依赖：

```text
输入点
  ↓
SA1 + PointNeXt-R：局部壁面几何
  ↓
SA2 + PointNeXt-R：局部形态变化
  ↓
SA3（约 32 centers）+ 1 个轻量全局注意力块
  ↓
FP / SEP-Kernel
  ↓
WSS 输出
```

推荐首版：1 block、4 heads、相对几何 bias（与 LitePT 的相对位置编码思路一致）、无高分辨率 attention。32 个 token 的二次复杂度很低，也能隔离“全局耦合是否有用”。

### 9.2 注意力需要回答的机制问题

- 分叉、瘤颈、入口附近是否形成非局部关联；
- attention 距离分布是否明显超出 SA2 局部半径；
- 增益是否主要来自 AAA rupture 或复杂分支病例；
- 若权重几乎均匀，则可能只是额外 MLP 容量，不算机制成立。

---

## 10. 路线 G：输出与目标——先利用已有 WSS 向量，再谨慎动 loss

### 10.1 WSS 向量辅助头

预处理可保存 `wall_wss_vec`（`preprocess.py:504`，需开 `store_wall_wss_vector`）。在向量坐标系和旋转一致性 QA 通过后，可联合预测：

$$
L=L_{mag}+\lambda_vL_{vec}
+\lambda_c\left\|\,\|\hat{\boldsymbol\tau}\|-\hat\tau\,\right\|_1.
$$

潜在收益：方向监督能约束局部切平面与流向特征，避免标量网络只记忆半径—WSS 的统计关联。风险是坐标方向不一致造成负迁移，因此不列入最近一轮主线。此处天然与 §6.5 的等变路线耦合：向量头应当是**等变**的，标量幅值头应当是**不变**的。

### 10.2 尺度—形状分解

可把病例内场写为：

$$
\log \tau_{c,i}=a_c+b_{c,branch(i)}+r_{c,i}.
$$

- `a_c`：病例尺度；
- `b`：分支尺度；
- `r`：局部形状残差。

**与现有代码的关系**：`config.py:84` 已有 `target_normalization='case_max'`（按逐病例壁面 WSS 最大值归一），这本质上就是在剥离 `a_c`。但 `case_max` 用的是**真值最大值**，推理时未知——若在部署时直接用它就是 oracle 泄漏。所以尺度—形状分解成立的前提是：`a_c`、`b` 能从**部署时可得的几何/入口条件**预测；不能把 CFD 输出压力、出口流量或 RCR 真值作为推理输入（与 §2.1“压力不直接进入 WSS”一致）。

### 10.3 排序/热点损失

架构稳定后，可用很小权重 `λ=0.02–0.05` 加入：

- 病例内 pairwise rank loss；或
- top-quantile hotspot soft IoU surrogate。

目标是修正热点排序，不是用大权重强行重塑整体回归。任何 tail loss 必须同时报告 overall、AAA rupture、high-WSS、p99 ratio、IoU 和 Spearman。

---

## 11. 路线 H：如果点云路线触顶，转向表面与算子建模

### 11.1 表面双流模型

```text
PointNeXt-R 点云流 ─┐
                    ├─ gated fusion ─ query decoder ─ WSS
Surface diffusion流 ─┘
```

- 点云流：处理局部三维形状与不规则采样。
- 表面流：沿三角网格/测地边传播，避免跨血管壁的错误欧氏近邻。
- 融合：在相同中心点处做门控，不一开始全层耦合。

这条路线更符合 WSS 定义在壁面切平面上的物理结构，但需要稳定网格邻接、法向和映射，工程成本高于 PointNeXt-R。

### 11.2 神经算子/physics token 的启动条件

只有当数据扩展到多入口条件、多时间相位、更多病例或可预训练仿真集合后，再评估 GINO、Transolver、PTv3/Mamba。当前样本规模下，先换成大算子很可能把问题从“表示不足”变成“数据不足且难归因”。

---

## 12. 推荐实验矩阵

### Stage 0：前置审计，不训练或只做短跑

| ID | 内容 | 输出 |
|---|---|---|
| A0-1 | 新的独立分层 train/val/test 划分审计 | 病例类型、rupture、点数、WSS 分布平衡表 |
| A0-2 | `sa_blocks` forward trace + 参数统计（含让 `pointnetpp` 真正消费该字段） | 证明配置确实改变执行图；记录父模型参数量作为 §4.2 预算基准 |
| A0-3 | train-val/best-last/**总体指标种子间标准差**汇总 | 判断是否需要 Drop，并**据此标定 §13 的 Go/No-Go 阈值与最小可检测效应** |
| A0-4 | 特征余弦相似度、有效秩 | 判断过平滑 |
| A0-5 | normal/tangent/branch mapping QA | 决定 LocalGeoPE v2 和向量头可否启动 |
| A0-6 | 移除绝对 xyz / 相对方向项的消融 | 量化“走严格旋转不变路线”的精度代价（§6.5） |

### Stage 1：确定父模型与残差主线

| ID | 唯一变化 | Seeds | 目的 |
|---|---|---:|---|
| C-Q1V | Q1V 在新划分复跑 | 3 | 稳定基线 |
| C-D2 | D2 在同划分配对复跑 | 3 | 判断 D2 是否是真增益 |
| M-FEAT | 父模型 + 已备逐点特征（`radius_gradient`、`coord_scale`，可选 `cohort_*`） | 3 | **零成本对照**：先看“用满已备特征”能否移动指标 |
| R-PNX | 胜出父模型 + PointNeXt-R `(1,1,0)`，无 Drop | 3 | 检验残差核心 |
| R-BRIDGE | 另一父模型 + 同一残差核心 | 1，可选 | 检查收益是否依赖父采样协议 |

若 Q1V 与 D2 仍无统计差异，默认选择计算更便宜的 Q1V 作为后续父模型。

### Stage 2：正交模块筛选

全部基于 `R-PNX`，先固定一个筛选种子：

| ID | 唯一变化 | 首轮 |
|---|---|---:|
| M-GEO | LocalGeoPE v1 | 1 seed |
| M-ATTN | SA3 32 centers 上 1×4-head attention | 1 seed |
| M-SEP | 受约束 SEP-Kernel（`-2 ln d` 初始化） | 1 seed |
| M-POOL | gated max+mean | 仅前三者均失败时 |

任何通过单种子门槛的项，再补 seeds `7/2025` 做三种子确认。不要先做 `GEO+ATTN+SEP`。

### Stage 3：Drop 条件分支

只在 Stage 0/1 出现过拟合、种子方差或过平滑证据时：

| ID | 唯一变化 | 候选值 |
|---|---|---|
| REG-H | head dropout | `0.1` |
| REG-P | max DropPath | `0.05` |
| REG-E | 保覆盖 NeighborDrop | `0.05` |

先单变量筛选，胜者才与 Stage 2 的单个胜出模块组合。

### Stage 4：研究型路线

- WSS vector auxiliary head（等变，与标量不变头联合）；
- surface diffusion + PointNeXt dual stream；
- branch/centerline token；
- scale-shape decomposition；
- 小权重热点排序损失；
- 数据扩大后的 GINO/Transolver/PTv3/Mamba。

---

## 13. 统一 Go/No-Go 门槛

> 前置声明：以下所有数值阈值都是**待 A0-3 标定的初值**。只有测出总体指标的种子间标准差 `σ_seed` 后，才能确认这些阈值对应的最小可检测效应（MDE）是否可实现。当前病例级 bootstrap CI 半宽约 `0.024`，提示效应尺度很小、噪声不可忽略——先估计噪声，再定阈值。

### 13.1 单种子只做筛选（应设成高召回，而非高精度）

初稿把单种子筛选门（`+0.012`）设得几乎与三种子确认门（`+0.015`）一样高。这会让筛选阶段的**假阴性率过高**：若真效应 `+0.015`、单种子 SD 约 `0.01`，一个种子低于 `+0.012` 的概率可达三成以上，会误杀真实小效应。筛选应放宽、确认才收紧。相对配对父模型：

- physical `ΔR²_cb ≥ σ_seed`（即“单种子超过一个种子标准差”，A0-3 给出 `σ_seed` 后取该值，不早于 A0-3 固定 `+0.012`）；
- MAE、RMSE 均不恶化超过 1%；
- AAA rupture `ΔR² ≥ -0.01`；
- hotspot IoU 与 Spearman 至少一项改善，另一项不低于 `-0.01`；
- high-WSS `ΔR² ≥ +0.03`，或 p99 ratio 向 1 靠近至少 `0.02`；
- 无 NaN，chunk/full 推理一致，训练成本不越过预设上限。

单种子达标只代表“值得补种子”，不代表已确认。

### 13.2 三种子确认

- mean physical `ΔR²_cb ≥ max(+0.015, 2·σ_seed/√3)`（把阈值显式绑定到种子标准差；`σ_seed` 未知前按 `+0.015` 占位）；
- 至少 2/3 seeds 为正；
- 病例级 paired bootstrap 95% CI 下界 `>0`；若仍跨 0，只能标记 candidate；
- AAA rupture、高 WSS、IoU 不得出现系统性退化；
- 对应机制指标必须非退化，例如残差分支确实被使用（γ 非零、分支范数比非零）、attention 非均匀、SEP 权重非异常塌缩。

### 13.3 为什么门槛高于当前 D2 的 0.0002

当前病例级差异的置信区间宽，极小点估计容易来自随机种子、病例组成或评估噪声。下一轮需要寻找能跨过噪声地板的改造，而不是继续按排行榜小数点后四位排序。把阈值绑定到实测 `σ_seed` 而非拍脑袋的常数，是本版相对初稿的关键方法学修正。

---

## 14. 实现落点与防伪测试

### 14.1 建议代码落点

- `training_wss_min/baseline_models.py`
  - 新增 `LocalInvResBlock`（可复用 `pointnext.py:65-83` 的 `InvResMLP` 结构，但加残差缩放 γ、可配 expansion、统一 ReLU/BN）；
  - 让 `PointNetPlusPlusRegressor` 按 `sa_blocks` 堆叠上述块（当前它根本不接收 `sa_blocks`）；
  - 新增可选 `CoarseGlobalBlock`；
  - 保留原类，避免历史实验定义漂移。
- `training_wss_min/config.py`
  - 让 `sa_blocks` 真正进入 `pointnetpp` 模型构造（现在只进 `pointnext_s`）；
  - 新增 `invres_expansion`、`residual_scale_init`、`drop_path_rate`、`local_geope`、`coarse_attention`、`sep_kernel`。
- query decoder 所在模块
  - 新增 `ConservativeSEPKernel`（`-β ln d` 基线 logit，β 初值 2），保留 3-NN 原实现作为严格 parent。
- 评估/日志
  - 增加参数量、FLOPs/step、显存、每病例 paired delta、bootstrap CI、分层指标、机制诊断，以及**总体指标种子标准差**（供 §13 标定阈值）。

### 14.2 必须有的单元/集成测试

1. `sa_blocks=(0,0,0)` 与原父模型数值等价或在明确容差内。
2. 开启 `(1,1,0)` 后参数量和 forward trace 必须改变（防止 `sa_blocks` 被 `pointnetpp` 静默忽略的历史坑）。
3. residual `γ=0` 时 block 输出等于 identity。
4. DropPath 在 eval 模式完全关闭；train 模式统计丢弃率接近配置。
5. NeighborDrop 不得产生零邻居中心，coverage 主边始终存在。
6. SEP query=support 时走 identity，结果与支持特征一致。
7. SEP 权重非负且逐 query 求和为 1；**且 `g_θ=0, β=2` 时与 `knn_interpolate` 逐位一致**（初始等价单测，针对 §7.2 的公式修正）。
8. chunk/full inference 在容差内一致。
9. 旋转一致性检查**分档写**：
   - 对“仅不变特征”变体（§6.5 路线 1），严格断言随机刚体旋转下标量输出一致、向量输出同步旋转；
   - 对含绝对 xyz/原始 Δx 的现状模型，只做**近似一致**检查，容差与 `rot_aug` 强度匹配，**不得**断言严格相等。
10. 旧 checkpoint 加载失败时必须显式报错，禁止静默部分加载后误做对比。

---

## 15. 明确不建议立即做的事情

- 不直接把项目现有 `pointnext.py` 整体替换 baseline 并与旧结果横比；采样协议变化太多。
- 不把宽度、深度、Drop、attention、SEP 和新 loss 一次性组合。
- 不继续把 10k/all points 或更大 k 作为主搜索轴。
- 不重复当前形式的 QAD 任意（无界加性）输出残差。
- 不在没有 train-val gap 证据时默认使用大 dropout。
- 不把 attention 铺在高分辨率全部点上。
- 不使用 CFD 真值压力、出口流量或 RCR 作为部署输入（`case_max` 归一化在推理端同属 oracle，见 §10.2）。
- 不在未跑 A0-6 消融、未定 §6.5 路线前，把“严格旋转不变”写进验收单测。
- 不只看 overall `R²`；必须同时看病例级 CI、AAA rupture、高 WSS、p99 ratio、IoU、Spearman。
- 不因单种子高分就自动级联下一组组合实验；也不在拿到 `σ_seed` 前用固定常数阈值下结论。

---

## 16. 建议其他智能体重点质疑的问题

（标注 ✅ 者已在本版部分回答，其余仍开放。）

1. PointNeXt-R 的收益能否在完全固定 sampling/grouping/decoder 后成立？
2. `sa_blocks=(1,1,0)` 是否足以检验残差机制，还是需要 `(2,2,0)` 的第二级剂量？（另需注意 ✅：`pointnetpp` 当前根本不消费 `sa_blocks`，必须先修）
3. LocalGeoPE 中哪些量在部署时稳定可得，哪些存在坐标或映射泄漏？（✅ 已标注各分量的旋转不变性，见 §6.1/§6.5）
4. D2 的微小优势是否来自某几类病例，能否用层级 bootstrap 或 mixed-effects 分析确认？
5. 高 WSS 负 `R²` 主要是排序失败、尺度偏差还是峰值压缩？对应修复是否不同？（§2.1 给出“压力不直接进入 WSS”这一线索，指向近壁速度梯度/入口流量）
6. SEP-Kernel 的凸组合约束会不会过度限制跨支持点外推？是否需要一个幅度严格受限的二阶段残差？（✅ 立论已改为“凸约束优于无界残差”，见 §7.1）
7. 粗层 attention 是否真的捕获跨分支依赖，还是只增加参数量？
8. 当前数据是否足以训练向量辅助头；`wall_wss_vec` 的参考坐标和法向方向是否可靠？
9. 是否应先补全边界条件，还是先用多任务/尺度分解估计病例级 WSS scale？
10. Go/No-Go 门槛是否与现有种子方差匹配？（✅ 本版把阈值显式绑定到 A0-3 的 `σ_seed`，见 §13）
11.（新）走严格旋转不变/等变路线（去绝对 xyz + 局部标架）相对现状 `rot_aug` 的精度—鲁棒性权衡如何？（由 A0-6 回答）

---

## 17. 最终优先级

| 排名 | 方向 | 原因 | 预期成本 | 决策 |
|---:|---|---|---:|---|
| 1 | 新划分三种子复核 Q1V/D2 + 估计 `σ_seed` | 先确定可信父模型和噪声地板，并据此定阈值 | 中 | 必做 |
| 2 | 已备逐点特征 `M-FEAT` | 零成本，先榨干“输入未用满”的收益 | 极低 | 优先 |
| 3 | PointNeXt-R `(1,1,0)`，无 Drop | 直接修复局部表示和优化路径，且可归因；组件已存在，工作量小 | 中 | 主线 |
| 4 | LocalGeoPE v1 | 低成本补充方向、尺度和轴向变化 | 低 | 优先筛选 |
| 5 | SA3 粗层 attention | 用低成本检验全局流路依赖 | 中 | 优先筛选 |
| 6 | 保守 SEP-Kernel（`-2 ln d` 初始化） | 针对已知支持—查询瓶颈，用凸约束替代无界残差 | 中 | 优先筛选 |
| 7 | 旋转不变性消融 A0-6 | 决定是否走“免费严格不变 + 去泄漏”路线 | 低 | 前置审计 |
| 8 | DropPath / head dropout / NeighborDrop | 只在诊断支持时降低过拟合或过平滑 | 低 | 条件分支 |
| 9 | WSS 向量辅助头（等变） | 增加物理方向监督 | 中 | QA 后启动 |
| 10 | 表面 diffusion + 点云双流 | 更贴合壁面拓扑与切平面物理 | 高 | 中长期 |
| 11 | GINO/Transolver/PTv3/Mamba | 需要更多数据、条件和工程投入 | 很高 | 暂缓 |

一句话概括：**先用已备特征和“小而真实的 PointNeXt 残差核心”验证表示收益（残差块已在代码中，工作量小），再用局部几何、粗层全局交互和‘凸约束’SEP-Kernel 分别击穿三个已知瓶颈；Drop 与旋转不变路线均由诊断/消融触发。所有 Go/No-Go 阈值绑定到实测种子方差。如果这些仍无法稳定越过噪声地板之上的总体增益门槛，就应停止继续堆网络，转向补充近壁速度梯度相关的入口/分支/几何信息与表面物理建模。**

---

## 18. 项目内依据

文档：
- `docs/02-推进与变更/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md`
- `docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`
- `docs/02-推进与变更/WSS最小化_训练实验跟踪.md`
- `docs/02-推进与变更/WSS最小化_PointNet++架构精度优化_第一性原理方案与交叉论证工作稿_2026-07-18.md`
- `docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx`

代码（本版审查引用的具体落点）：
- `training_wss_min/baseline_models.py`：SA `local` MLP-max（`352,396-398`）、`_mlp` 定义（`28-36`）、`qad_lite` 零初始化+identity（`456-468,495-508`）、覆盖保障分组（`166-179,182-193`）、`pointnetpp` 构造不传 `sa_blocks`（`552-562`）
- `training_wss_min/pointnext.py`：`InvResMLP` 残差块（`65-83`）、`PointNeXtSeg` 消费 `sa_blocks`（`101-128`）
- `training_wss_min/models.py`：`sa_blocks` 仅进 `pointnext_s`（`22-42`）
- `training_wss_min/config.py`：`FEATURE_KEYS`（`29-34`）、默认 `input_features`/`rot_aug`（`75,77`）、`target_normalization`（`84`）、`fp_knn=3`（`121`）、`out_dim` 矢量占位（`126`）、`overlap_cap≤1/3`（`318`）
- `training_wss_min/objectives.py`
- `pipeline_wss_min/preprocess.py`：`wall_wss_vec` 保存（`504`，受 `store_wall_wss_vector`）

> 注：本文中的论文结果只用于提出结构假设。不同数据规模、任务定义和评估协议下的数值不可直接与本项目横比；所有建议最终均以当前固定协议下的配对实验为准。

---

## 附录 A：对抗性审查逐条记录

| 项 | 类型 | 初稿问题 | 修正 | 代码/文献依据 |
|---|---|---|---|---|
| A1 | 技术错误 | QAD 失败归因于“破坏同点恒等先验” | `qad_lite` 已零初始化+identity 短路，初始与父模型一致仍退化；真因是训练后无界加性残差 → 重写 §7.1 立论 | `baseline_models.py:456-468,495-508` |
| A2 | 数学错误 | SEP-Kernel `softmax(-d/T)` “等价逆距离插值” | PyG 用 `1/d²`；须改基线 logit 为 `-β ln d`（β=2）才严格等价 → 重写 §7.2 + 新增单测 7 | `baseline_models.py:499-500` |
| A3 | 事实/归因 | “引入”残差块，暗示从零实现 | `InvResMLP` 与 `sa_blocks` 消费已存在于 `pointnext.py`；真正增量是移植进固定支持路径 + γ + expansion + 让 `pointnetpp` 消费 `sa_blocks` → 重写 §4 | `pointnext.py:65-83,101-128`；`models.py:22-42` |
| A4 | 一致性 | 提出严格旋转不变单测 | 现架构吃绝对 xyz + 原始 Δx，非旋转不变，仅 `rot_aug` 近似 → 新增 §6.5 设计决策 + A0-6 消融 + 分档单测 9 | `config.py:75,77`；`baseline_models.py:396` |
| A5 | 遗漏 | 称输入“仅 6D、缺尺度/方向” | `radius_gradient`/`coord_scale`/`cohort_*` 已在 `FEATURE_KEYS` 未用 → 新增 §6.0 与 `M-FEAT` 零成本实验 | `config.py:29-34` |
| A6 | 方法学 | Go/No-Go 阈值为固定常数，单种子门≈确认门 | 绑定到 A0-3 实测 `σ_seed`，筛选门放宽为高召回 → 重写 §13 | 需 A0-3 |
| A7 | 严谨性 | 五项误差写成可加“分解” | 改称“误差归类”，声明非正交 → 修订 §2.2 | — |
| A8 | 物理补强 | WSS 公式未指出压力项消去 | 补充切向投影消去压力 → WSS 为纯近壁速度梯度量，指导“补哪种条件” → 扩写 §2.1 | — |
| A9 | 表述 | §1.1 width 行 `0.2589→0.2183` 箭头含义不明 | 澄清为 w32→w64 的验证 `R²_cb` 净下降 | 以结果汇总 xlsx 为准 |
| A10 | 正面结论 | 三处 2026 高风险引用疑似臆造 | 逐一核验为真实文献（LitePT=arXiv:2512.13689；两篇 Nature 期刊 DOI 均存在），予以保留并精化 LitePT 要点 | 见 §3 |
