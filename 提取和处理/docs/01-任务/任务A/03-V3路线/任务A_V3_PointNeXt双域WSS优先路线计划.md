# 任务A V3：PointNeXt 双域 WSS 优先路线计划

> 路线名：`Route-DualDomain-PointNeXt-V3`
> 定位：基于 Fluent/CFD 导出的壁面点与内部点，使用 PointNeXt-style 点云主干建立双域监督的血流动力学快速预测模型。
> 当前状态：路线设计稿；尚未生成训练配置，尚未修改训练代码。

---

## 1. 路线定位

V3 不再沿用“真实图结构 GNN”的叙事，也不以 Transformer 作为默认主干。

本路线将当前数据明确表述为：

> 从 Fluent/CFD 后处理流程导出的非结构化三维点云。每个病例/时间步包含壁面点与内部点；内部点有速度和压力标签，壁面点有 WSS 和压力标签。

因此，V3 的核心任务不是在真实 mesh 拓扑上做消息传递，而是在非结构化点云上学习：

- 内部点：`u, v, w, p`
- 壁面点：`WSS, p`

PointNeXt-style 点云主干用于从局部点云邻域中提取几何与流动上下文；中心线几何先验作为关键输入特征保留。

需要特别说明：当前仓库中的 `FieldPointNeXt` 是 **PointNeXt-style local pooling backbone**，不是严格意义上的完整层次化 PointNeXt。它仍读取预处理图中的 `data.edge_index` 做局部 mean/max pooling。V3 第一阶段允许复用该实现，但文稿与汇报中不得直接宣称已实现 FPS/radius grouping/set abstraction 形式的完整层次化 PointNeXt。

---

## 2. 数据事实与方法边界

### 2.1 数据事实

- 数据来自 Fluent/CFD 工具导出的壁面点与内部点。
- 当前训练图中的邻域关系不是 Fluent 原始 mesh connectivity。
- 预处理阶段已经支持并默认采用 **FPS + 随机采样的混合采样**，用于保证空间覆盖与采样多样性；当前默认配置为 `target_total_points=15000`、`wall_max_points=13000`、近壁内部优先、`hybrid_fps_ratio=0.5`。
- 采样层面的 FPS 不等同于模型层面的层次化 PointNeXt。当前模型输入仍是采样后的固定点集，图转换阶段会基于采样点重新构建 kNN `edge_index`。
- 壁面点具备：
  - 坐标
  - 中心线相关几何特征
  - `is_wall = 1`
  - 压力 `p`
  - WSS 标签：`wss, wss_x, wss_y, wss_z`
- 内部点具备：
  - 坐标
  - 中心线相关几何特征
  - `is_wall = 0`
  - 速度 `u, v, w`
  - 压力 `p`

### 2.2 方法边界

V3 中的局部邻域只解释为点云局部上下文，不解释为真实血流物理连通边。

V3 第一阶段的准确表述：

> 混合 FPS/随机采样保证输入点云的空间覆盖；现有 PointNeXt-style 主干在采样点的局部 kNN 邻域上做残差局部池化。

V3 第二阶段若要宣称“完整层次化 PointNeXt”，需要新增模型侧的 FPS/radius grouping/set abstraction 或等价的多尺度层次结构，并与第一阶段实现分开命名和比较。

V3 第一轮不引入：

- Transformer 主干
- MeshGNN
- Mamba
- PINN / Navier-Stokes 物理残差
- 高低保真混合训练
- 复杂两阶段训练

第一轮重点是把监督口径、loss 配平和评价指标做干净。

---

## 3. 主干与输入输出

### 3.1 主干选择

V3 第一阶段默认主干：

- `FieldPointNeXt`（PointNeXt-style local pooling）

选择理由：

- 适合非结构化点云。
- 不依赖真实 mesh 拓扑。
- 可通过局部邻域聚合建模近壁上下文。
- 相比普通 MLP 或早期单层点级模型，残差与归一化结构更利于稳定训练。
- 能直接复用当前仓库已有 `pointnext` 模型入口。

### 3.1.1 当前实现与完整 PointNeXt 的边界

当前仓库中的 `FieldPointNeXt` 具备：

- 节点特征 + 全局条件拼接输入。
- 多层残差块。
- 基于 `edge_index` 的局部邻域 mean/max pooling。
- 共享 decoder + `field_head` / `wss_head`。

当前尚不具备：

- 模型内部 FPS 下采样。
- radius grouping。
- set abstraction / feature propagation。
- 多尺度层次 encoder-decoder。

因此，V3 第一轮实验命名和论文叙事应使用：

- `PointNeXt-style local pooling`
- `PointCloud local pooling backbone`
- `基于混合采样点云的局部邻域聚合模型`

暂不使用：

- `完整 PointNeXt`
- `层次化 PointNeXt`
- `多尺度 PointNeXt`

若第一阶段双域 loss 已验证有效，再开放第二阶段结构升级：

| Exp ID | 目的 | 唯一变化项 |
| --- | --- | --- |
| `V3P-Arch-01` | 验证完整层次化点云主干是否优于现有 local pooling | 引入模型侧 FPS/radius grouping/set abstraction |
| `V3P-Arch-02` | 验证多尺度邻域是否改善 WSS | 增加多半径或多 k 邻域聚合 |

### 3.2 输入特征

基础输入：

- `x, y, z`
- `t_norm`
- `BC_Inlet, BC_O1, BC_O2, BC_O3, BC_O4`
- `is_wall`

核心中心线几何先验：

- `Abscissa`
- `NormRadius`
- `Curvature`
- `Tangent_X, Tangent_Y, Tangent_Z`

后续候选几何增强：

- `dist_to_wall`
- `dist_to_bifurcation`
- `branch_id`
- `d_tangent_ds`
- 其他已在 Line G 中验证过的扩展几何特征

### 3.3 输出头

V3 保留两个输出头：

```text
field_head -> [u, v, w, p]
wss_head   -> [wss, wss_x, wss_y, wss_z]
```

第一轮不拆复杂 decoder；先使用共享 PointNeXt-style backbone + 两个 head 的结构。
**关键要求**：在共享 backbone 之后，`field_head` 和 `wss_head` 不能直接是单层线性映射。必须各自加入至少 2 层的独立 MLP（带非线性激活）作为**特征解耦缓冲区 (Decoupling Buffer)**，缓解 WSS 和流场的梯度在浅层直接碰撞。

---

## 4. 双域 Mask Loss 设计

### 4.1 基本原则

V3 必须使用双域 mask loss，而不是继续用全节点 `target_weights` 简化表达所有监督目标。

定义：

```text
interior_mask = is_wall == 0
wall_mask     = is_wall == 1
```

监督规则：

- 内部点只参与 `u, v, w, p` 的 field loss。
- 壁面点必须参与 `p` 和 WSS loss，同时**强制参与速度 loss，其目标真值恒定为 `u=v=w=0`**（以建立近壁面无滑移边界条件约束）。
- 内部点不得参与 WSS loss。

当前代码已经对 WSS loss 使用 wall mask；V3 需要补齐的是 field loss 的双域 mask 口径，避免壁面速度与内部 WSS 的任务边界混读。

### 4.2 总损失

V3 默认总损失：

```text
L_total =
  lambda_vel    * L_velocity (包含内部点与 target=0 的壁面点)
+ lambda_p_int  * L_interior_pressure
+ lambda_p_wall * L_wall_pressure
+ lambda_wss    * L_wall_wss
```

第一版推荐权重：

| 分项 | 默认权重 | 说明 |
| --- | ---: | --- |
| `lambda_vel` | 0.3 | 保留近壁速度上下文，但不让速度成为主导目标 |
| `lambda_p_int` | 0.5 | 保留内部压力学习 |
| `lambda_p_wall` | 1.0 | 壁面压力为 V3 主 readout 之一 |
| `lambda_wss` | 0.1 | 从轻量 WSS 权重起步，避免重现 V2 梯度劫持 |

所有分项都应先在各自 mask 内按点数取均值，再乘权重求和。这样可以避免壁面点数量远多于内部点时，loss 被点数比例隐式支配。

### 4.3 早停与调度

V3 不使用原始 `data_loss + early_stop_wss_weight * wss_loss` 作为第一轮早停指标。

推荐验证分数：

```text
val_score =
  normalized_val_wall_wss_rmse
+ normalized_val_wall_p_rmse
+ 0.3 * normalized_val_interior_vel_rmse
```

*注：原计划的 `near_wall_vel_rmse` 如果在 DataLoader 中没有预先计算好的 mask（如 `is_near_wall`），在 `val_step` 动态计算会极大拖慢验证循环。因此第一版暂用全量 `interior_vel_rmse` 替代，或由预处理脚本提前生成 `is_near_wall` 标签。*

归一化基准推荐使用训练集目标标准差。若第一版实现成本较高，可先用首个 epoch 的验证分项 loss 固定归一化，但必须在实验记录中说明。

---

## 5. V2 失败经验规避

V3 设计必须显式避开 V2P-WSSP 中已经暴露的问题。

### 5.1 避免 WSS 梯度劫持

`V2P-WSSP-02` 使用：

- `target_weights=[2,2,2,0.5]`
- `wss_loss_weight=0.5`
- `early_stop_wss_weight=1.0`

结果中 `wss_loss` 加权后远大于 `data_loss`，共享 backbone 被 WSS 梯度主导，压力预测崩溃。

V3 规避方式：

- 第一版 `lambda_wss=0.1` 起步。
- `field_head` 与 `wss_head` 前必须设置独立的 MLP 缓冲层，避免 WSS 梯度直达 Backbone。
- 记录 raw 与 weighted loss。
- 若 `weighted_wss_loss > 2 * weighted_field_loss` 连续出现，立即停止该配置，不补 seed。
- 早停不直接使用未归一化 WSS loss。

### 5.2 不再把“仅 p+WSS”作为主线

`V2P-WSSP-05/06` 三 seed 结果显示：

- `wss_r2_wss` 仍为略负。
- Huber 相对 MSE 未稳定占优。
- `r2_p` 跨 seed 方差大。

因此 V3 不把 `target_weights=[0,0,0,1] + WSS` 作为主路线，而是保留内部速度的弱监督，用近壁流动上下文辅助壁面 WSS。

### 5.3 Huber 只做单变量对照

Huber 不是 V3 默认解法。只有在训练集壁面 WSS 分布分析显示极端值比例明显时，才执行 Huber 单变量对照。

### 5.4 避免新的架构归因错误

V3 第一阶段同时改变了主干叙事、监督口径和主指标，因此必须避免把所有改进都归因于“PointNeXt 更强”。

第一阶段允许回答的问题：

- 双域 mask loss 是否比混合全节点 loss 更合理。
- 几何先验在双域点云监督中是否仍有增益。
- 现有 PointNeXt-style local pooling 是否足以作为 V3 起跑主干。

第一阶段不允许回答的问题：

- 完整层次化 PointNeXt 是否优于所有方法。
- FPS/radius grouping 是否带来独立增益。
- 多尺度点云结构是否是 WSS 提升的主要原因。

---

## 6. V3 实验矩阵

### 6.1 第一轮核心实验

| Exp ID | 目的 | 输入 | 损失口径 | seed |
| --- | --- | --- | --- | --- |
| `V3P-Diag-00` | 诊断双域 mask 与 loss 尺度 | 同 Main | 只跑 1 epoch，输出分项 loss 与 mask 统计 | `1` |
| `V3P-Base-01` | 无几何 PointNeXt 下限 | `coords + t + BC + is_wall` | 双域 mask loss，`lambda_wss=0.1` | `1 -> [1,2,3]` |
| `V3P-Main-01` | 验证中心线几何先验 | `Base + Abscissa + NormRadius + Curvature + Tangent` | 同 Base | `1 -> [1,2,3]` |
| `V3P-WSS-01` | WSS 权重轻量增强 | 同 Main | 根据 Diag/Main 动态决定（如 `lambda_wss=0.2` 或 `0.05`），其余不变 | `1 -> [1,2,3]` |
| `V3P-WSS-02` | 速度上下文增强 | 同 Main | `lambda_vel=0.5`，其余不变 | `1 -> [1,2,3]` |
| `V3P-WSS-03` | Huber WSS 单变量对照 | 同 Main | 仅 `L_wall_wss` 改为 Huber | 条件执行 |

执行规则：

- `V3P-Diag-00` 必须先于正式训练执行，用于确认 mask、loss 尺度和日志字段。
- 每组先跑 `seed=1`。
- 对于 `V3P-WSS-01`，其 `lambda_wss` 的具体取值应在跑完 `V3P-Diag-00` 和 `Main-01` 第一周期后**动态决定**：若 `weighted_wss_loss` 仍微大于 field loss，则不应再加重（即不使用 0.2），而应向 0.05 衰减。
- 只有压力不崩、WSS 有正信号，再补 `seed=2,3`。
- 单 seed 明确负结果不补 seed，先回到 loss 和数据分布排查。

### 6.1.1 公平对照要求

`V3P-Base-01` 与 `V3P-Main-01` 必须保持以下项完全一致：

- 同一数据版本与图资产。
- 同一采样口径。
- 同一 split。
- 同一训练轮数、优化器、学习率、batch size、早停策略。
- 同一双域 loss 权重。
- 同一后处理脚本。

两者唯一变化项只能是是否启用中心线几何先验。否则不能把差异归因于 geometry。

### 6.2 几何先验消融

| Exp ID | 目的 | 说明 |
| --- | --- | --- |
| `V3P-Abl-01` | 去掉全部 geometry | 与 `V3P-Main-01` 对照 |
| `V3P-Abl-02-no-normradius` | 去掉 `NormRadius` | V1 中最关键几何项，必做 |
| `V3P-Abl-02-no-tangent` | 去掉 Tangent | V1 中次关键几何项，必做 |
| `V3P-Abl-02-no-dist-wall` | 去掉 `dist_to_wall` | 若 V3 输入启用该特征，则必做 |
| `V3P-Abl-02-no-bifurcation` | 去掉分叉相关特征 | 用于验证复杂区域先验 |
| `V3P-Abl-02-no-curvature` | 去掉 `Curvature` | 弱贡献项，可作为补充 |

---

## 7. 主指标与 Go / No-Go

### 7.1 主指标顺序

V3 主表固定按以下顺序组织：

1. `wall.r2_wss`
2. `wall.rmse_wss`
3. `wall.r2_p`
4. `wall.rmse_p`
5. `near_wall.rmse_vel_mag`
6. `near_wall.r2_vel_mag`
7. `interior.rmse_vel_mag`
8. `interior.r2_vel_mag`
9. 推理时间、显存和参数量

全局 `RMSE_|v|` 不作为 V3 第一排序指标。

WSS 指标必须拆成三层报告：

| 层级 | 指标 | 目的 |
| --- | --- | --- |
| 点级 | `wall.rmse_wss`, `wall.r2_wss` | 判断壁面点预测误差 |
| 病例级 | mean WSS、p95 WSS 的 Pearson/Spearman | 判断病例排序和临床统计量是否保留 |
| 区域级 | 高 WSS top-k overlap / Dice | 判断高剪切区域定位能力 |

若点级 R2 改善但病例级或区域级不改善，不得宣称临床相关 WSS surrogate 已成立。

### 7.2 Go 标准

V3 进入后续优化的最低标准：

- `V3P-Main-01` 相对 `V3P-Base-01` 在 `wall.r2_wss` 或 `wall.rmse_wss` 上改善。
- `wall.r2_p` 不明显退化。
- `near_wall` 速度指标不崩坏。
- 几何先验增益至少在多数 seed 或多数病例上重复出现。

V3 宣称 WSS 突破的标准：

- 三 seed `wall.r2_wss` 均值稳定超过 V1 最强锚点 `A-Opt-05-wss-multi` 约 `0.46`。
- `wall.r2_p` 维持在可接受范围。
- WSS 提升不是以压力或近壁速度明显崩坏为代价。

### 7.3 No-Go 标准

出现以下任一情况，应停止继续堆结构：

- `weighted_wss_loss` 长期大于 field loss 两倍以上。
- `wall.r2_p` 接近 0 或明显崩溃。
- `V3P-Main-01` 相对 `V3P-Base-01` 无几何增益。
- WSS 只在个别病例变好，病例级均值和分位数指标不改善。
- seed 方差过大，无法稳定复现。

---

## 8. 实现注意事项

### 8.1 建议新增配置字段

建议后续实现时新增：

```text
optim.domain_loss.enabled
optim.domain_loss.lambda_vel
optim.domain_loss.lambda_p_int
optim.domain_loss.lambda_p_wall
optim.domain_loss.lambda_wss
optim.domain_loss.normalize_by_target_std
```

第一版可保持向后兼容：未开启 `domain_loss.enabled` 时，沿用现有 loss 行为。
**强烈警告**：`optim.domain_loss.normalize_by_target_std` 必须作为 **V3 的首要前置条件 (P0)** 开启。因为 p 和 WSS 的绝对量级差异巨大，不做目标空间的 Z-score 归一化，只靠手动调 `lambda_wss=0.1` 极难保证全病例稳定收敛。

### 8.2 建议新增 loss 逻辑

建议在 `training/core/losses.py` 中新增 `dual_domain_point_loss`，专门服务 V3。

必须记录以下分项：

- `loss_interior_velocity`
- `loss_interior_pressure`
- `loss_wall_pressure`
- `loss_wall_wss`
- `weighted_loss_interior_velocity`
- `weighted_loss_interior_pressure`
- `weighted_loss_wall_pressure`
- `weighted_loss_wall_wss`

### 8.3 后处理与汇表

V3 后处理必须单列：

- 壁面 WSS 指标
- 壁面压力指标
- 近壁速度指标
- 内部速度/压力诊断指标

与 V1/V2 对照时必须注明：

- backbone
- 采样口径
- 是否监督速度
- 是否监督 WSS
- WSS loss 权重
- 是否使用双域 mask loss

### 8.4 采样与模型结构的解决路径

当前最大实现风险是：预处理已经使用 FPS/随机混合采样，但模型侧仍是固定采样点上的局部 kNN pooling。该风险的解决路径分两步：

1. **V3 第一阶段不改模型结构，只改表述与监督口径。**
   将当前实现明确命名为 `PointNeXt-style local pooling`。采样层面的 FPS 用于保证输入点云空间覆盖，不把它等同于模型内部层次化 PointNeXt。

2. **V3 第二阶段再决定是否实现完整层次化 PointNeXt。**
   只有当 `V3P-Main-01` 和 `V3P-WSS-*` 证明双域 loss 有正信号后，才开放 `V3P-Arch-*`，引入模型侧 FPS/radius grouping/set abstraction。这样可以避免第一轮同时改变主干结构和 loss，导致归因混乱。

若后续实现完整层次化 PointNeXt，必须与 `V3P-Main-01` 使用同一数据、split、loss 和评估口径，只把模型结构作为唯一变化项。

---

## 9. 测试计划

### 9.1 单 batch smoke test

必须确认：

- `wall_mask` 和 `interior_mask` 数量非零。
- 内部点不进入 WSS loss。
- 壁面点不进入速度 loss。
- 四个 loss 分项均能单独计算并写入日志。
- 采样元信息或图转换报告中可追溯当前图资产来自 hybrid/FPS/random 哪一种采样口径。
- 当前模型实际读取 `edge_index` 做局部 pooling，实验记录中需标明 `neighbour_source=knn_edge_index`。

### 9.2 一轮训练 smoke test

先执行：

- `V3P-Diag-00_seed1`
- `V3P-Base-01_seed1`
- `V3P-Main-01_seed1`

检查：

- `history.csv` 是否包含 V3 分项 loss。
- WSS loss 与 field loss 数量级是否可控。
- 验证分数是否按 V3 口径选择 checkpoint。

### 9.3 正式训练后处理

每个进入正式比较的 run 必须生成：

- `predictions_test/`
- wall WSS 指标
- wall pressure 指标
- near-wall velocity 指标
- 病例级统计表
- 分区域图

---

## 10. 当前结论

V3 的关键不是再换一个更复杂模型，而是把数据事实和监督口径改正确：

- 主干使用 PointNeXt。
- 监督改为壁面/内部双域 mask。
- WSS 从轻量权重开始，避免梯度劫持。
- 几何先验继续作为核心创新点验证。
- 主指标从全局速度误差转为壁面 WSS、壁面压力和近壁速度上下文。

若 `V3P-Main-01` 能在 WSS 与壁面压力上稳定优于无几何版本，并进一步超过 V1 的 `A-Opt-05-wss-multi` WSS 锚点，则 V3 可以成为任务A后续最干净的主线。
