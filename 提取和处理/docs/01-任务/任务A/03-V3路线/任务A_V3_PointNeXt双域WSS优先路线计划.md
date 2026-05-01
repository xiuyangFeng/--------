# 任务A V3：PointNeXt 双域 WSS 优先路线计划

> 路线名：`Route-DualDomain-PointNeXt-V3`
> 定位：基于 Fluent/CFD 导出的壁面点与内部点，使用 PointNeXt-style 点云主干建立双域监督的血流动力学快速预测模型。
> 当前状态：路线设计稿（2026-04-29 严谨性修订版）；尚未生成训练配置，尚未修改训练代码。

---

## 1. 路线定位

V3 不再沿用"真实图结构 GNN"的叙事，也不以 Transformer 作为默认主干。

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

### 2.1.1 数据/采样口径锁定项（V3 P0 不可变）

> 本节为 2026-04-29 修订新增（2026-04-29 二次澄清：内部点采样实为"100% 近壁层"，不再是"近壁优先"；2026-05-01 补充：新增 `V3P-Probe-*` 单目标/双目标诊断层）。锁定项一旦写入，V3 第一阶段所有实验（含 `V3P-Diag-00 / Probe-* / Anchor-01 / Base-01 / Main-01 / WSS-* / Abl-*`）必须严格沿用；任何变化都必须升入 V3 第二阶段并改名。

| 锁定项 | 取值 | 与历史实验对齐 / 代码引用 |
| --- | --- | --- |
| `split` | `split_AG_v1` | 与 V2P-WSSP-05/06/07、V1 wss-multi 同 split |
| `target_total_points` | 15000 | `pipeline/config.py:91` |
| `wall_max_points` | 13000 | `pipeline/config.py:96` |
| `boundary_threshold` | 2.0 mm | `pipeline/config.py:99` |
| `boundary_core_ratio` | `(1.0, 0.0)` | `pipeline/config.py:102`（**100% 近壁层，0% 核心层**） |
| `allow_core_fallback` | `False` | `pipeline/config.py:106`（近壁层不足时**不**用核心层补） |
| 实际采样形态 | **13000 壁面点 + 2000 近壁内部点（dist_to_wall < 2mm）**；近壁内部点数不足时不补核心点，剩余预算空置 | 与 V2P-WSSP-01/05/06/07 同图资产 |
| `hybrid_fps_ratio` | 0.5 | `pipeline/config.py:114` |
| `enabled_node_features`（Main） | `coords + Abscissa + NormRadius + Curvature + Tangent_{X,Y,Z} + is_wall` | 与 V2P-WSSP-05/06/07 一致 |
| `enabled_global_features` | `t_norm + BC_Inlet + BC_O1..O4` | 与 V2P-WSSP 系列一致 |

**采样口径与 `near_wall` 命名注意事项**：

当前 pipeline 在预处理阶段先分开处理壁面点和内部点：

- 壁面点来自 `surface_df`，最多采样 `wall_max_points=13000` 个点；壁面点保留压力与 WSS 标签。
- 内部点预算为 `target_total_points - 实际壁面点数`，在默认 `15000 - 13000` 口径下约为 2000 个点。
- 内部点先通过壁面点 KDTree 计算到最近壁面的距离，再按 `dist_to_wall < boundary_threshold=2.0mm` 筛出近壁层。
- 当前 `boundary_core_ratio=(1.0, 0.0)` 且 `allow_core_fallback=False`，因此 V3 第一阶段图资产中的内部点全部来自 2mm 近壁层；核心内部点不进入 V3 第一阶段，近壁层不足时也不使用核心点回填。
- 在近壁内部候选点内再执行 `sampling_method="hybrid"`，`hybrid_fps_ratio=0.5`，即约 50% FPS + 50% 随机采样。

需要严格区分两个容易混淆的概念：

- **采样层面的近壁内部点**：指预处理时由 `dist_to_wall < 2mm` 筛出的内部候选点，V3 第一阶段的 2000 个内部点都来自这一层。
- **历史 `regional_eval.near_wall` 区域**：当前项目默认定义为内部点中 `NormRadius > 0.8`，见 `docs/00-规范与记录/任务A分区域评估口径.md` 与 `training/analysis/regional_eval.py`。

二者不是同一个 mask，不应在论文、汇报或主表中无说明地混用。V3 若要表达"全部内部点都来自近壁采样层"，建议写作 `sampled_near_wall_interior` 或"2mm 近壁内部采样点"；除非同步修改并声明评估口径，否则不要直接覆盖历史 `regional_eval.near_wall` 的含义。

**重要 mask 等价关系**：

由 `boundary_core_ratio=(1.0, 0.0)` 与 `allow_core_fallback=False` 两项锁定，再叠加 `pipeline/utils/sampling.py:246-247` 的 `boundary_mask = distances < boundary_threshold` 硬过滤，V3 锁定图资产中所有 `is_wall == 0` 的节点 **必然** 满足 `dist_to_wall < 2mm`。因此：

```text
sampled_near_wall_interior_mask  ≡  interior_mask  ≡  (is_wall == 0)
```

**这意味着 V3 不需要新增 `is_near_wall` 节点字段**——若指标意图是表达"2mm 近壁内部采样点"，可直接用现有的 `is_wall` 字段派生 `sampled_near_wall_interior` mask。该等价关系只适用于 V3 的 `dist_to_wall < 2mm` 采样层事实，**不等价于**现有 `regional_eval.near_wall = interior & NormRadius > 0.8` 区域定义。如果未来要修改 `boundary_core_ratio` 或 `allow_core_fallback`（比如允许 30% 核心区点重新进入），则采样层等价关系失效，必须重新评估 §4.3 / §7.1 中的 `sampled_near_wall_interior.*` 或 `near_wall.*` 指标定义；在那之前，本路线一律假设上述采样层等价关系成立，并由 `V3P-Diag-00` 第一项验证项确认（详见 §9.1）。

**锁定理由**：
1. 与 V2P-WSSP-05/06/07 直接接续，差异轴只剩 backbone × 监督口径，归因清晰。
2. 与 V1 默认采样下的 `A-Opt-05-wss-multi`（`wall.r2_wss ≈ 0.46`）不同口径，本路线不再把那条数据当 Go 阈值；V3 自建同口径锚点（见 `V3P-Anchor-01`）。
3. `boundary_core_ratio=(1.0, 0.0)` 是 WSS / 壁面血流动力学专线的预设：因为 V3 主目标是壁面 WSS / p 与近壁速度上下文，远离壁面的核心区点对 WSS 学习贡献小，且会稀释 2000 个内部点预算的近壁分辨率。

### 2.2 方法边界

V3 中的局部邻域只解释为点云局部上下文，不解释为真实血流物理连通边。

V3 第一阶段的准确表述：

> 混合 FPS/随机采样保证输入点云的空间覆盖；现有 PointNeXt-style 主干在采样点的局部 kNN 邻域上做残差局部池化。

V3 第二阶段若要宣称"完整层次化 PointNeXt"，需要新增模型侧的 FPS/radius grouping/set abstraction 或等价的多尺度层次结构，并与第一阶段实现分开命名和比较。

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
- 能直接复用当前仓库已有 `pointnext` 模型入口（`training/core/models.py:288 FieldPointNeXt`、`training/core/models.py:369 MODEL_REGISTRY['pointnext']`）。

### 3.1.1 当前实现与完整 PointNeXt 的边界

当前仓库中的 `FieldPointNeXt` 具备：

- 节点特征 + 全局条件拼接输入。
- 多层残差块。
- 基于 `edge_index` 的局部邻域 mean/max pooling。
- 共享 decoder + `field_head` / `wss_head`（**当前为单层 `nn.Linear(hidden_dim, *)`**，见 `training/core/models.py:321-322`）。

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

#### V3 第一阶段允许的"轻量结构改动"白名单

> 本节为 2026-04-29 修订新增。是为了把 3.3 节"head 必须升级为多层 MLP"的强约束与本节"V3 第一阶段不改 backbone 拓扑"的边界条款明确切开，避免文档自相矛盾。

V3 第一阶段允许的结构改动：

| 改动 | 是否允许 | 说明 |
| --- | --- | --- |
| `field_head` / `wss_head` 升级为 ≥2 层 MLP（含非线性） | ✅ 允许 | head decoupling buffer，缓解 WSS 与流场梯度在浅层碰撞；视为 V3 默认配方一部分，纳入 `V3P-Base-01` 与 `V3P-Main-01`，避免归因混乱 |
| 共享 `shared_decoder` 拓扑改动 | 🚫 不允许 | 留至 V3 第二阶段 |
| 模型内部 FPS / radius grouping / set abstraction | 🚫 不允许 | 留至 `V3P-Arch-*`（V3 第二阶段） |
| 多尺度 encoder-decoder | 🚫 不允许 | 同上 |
| 输入特征通道增加（如 `dist_to_wall`） | ✅ 允许（按 6.1 表格指定的实验组） | 与 backbone 拓扑无关 |

如果第一阶段双域 loss 已验证有效，再开放第二阶段结构升级：

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

**关键要求（V3 默认配方，通过 `model.head_layout` 配置开关启用）**：在共享 backbone 之后，`field_head` 和 `wss_head` 不能直接是单层线性映射。在 V3 配置中显式写 `model.head_layout="mlp2"`，让每个 head 升级为至少 2 层的独立 MLP（含非线性激活，如 GELU/ReLU + LayerNorm），作为**特征解耦缓冲区 (Decoupling Buffer)**，缓解 WSS 和流场的梯度在浅层直接碰撞。该升级在 `V3P-Base-01` 与 `V3P-Main-01` 中同时启用，作为 V3 默认配方一部分；不允许只在某一组启用，否则归因混乱（不能区分"head 升级"和"几何先验"哪一项有效）。

> **向后兼容硬约束（详见 §11.0）**：`model.head_layout` 默认值必须为 `"single_linear"`，与 `training/core/models.py:321-322` 当前实现一致；旧 V1 / V2 / V2P-WSSP-* JSON 不含此字段时走默认值，所有历史 checkpoint 仍能被 `predict_field` 正常加载。本节关于"head 升级"的强约束**仅适用于 V3 系列实验**。

#### WSS 4 维冗余诊断（必做）

> 本节为 2026-04-29 修订新增。

`wss_head` 默认输出 4 维 `[wss_mag, wss_x, wss_y, wss_z]`，但 `wss_mag = sqrt(wss_x² + wss_y² + wss_z²)` 是确定性关系。等权对 4 维做监督会出现 magnitude 与分量预测不一致的现象。

V3 第一阶段的处置：

- 第一阶段保留 4 维输出，便于与 `V2P-WSSP-01/05/06/07` 直接对齐；
- `V3P-Diag-00` 必须输出"`pred[wss_mag]`"与"`sqrt(pred[wss_x]^2 + pred[wss_y]^2 + pred[wss_z]^2)`"在测试集上的相对误差 CDF（`min / 25% / 50% / 75% / 95% / max`），作为是否切换到"3 分量 + magnitude 一致性约束"的依据；
- 若 50% 相对误差 > 5%，启用条件实验 `V3P-WSS-04`：仅监督 3 个分量，magnitude 由 `sqrt` 推导，并在 loss 上加一致性约束 `|pred_mag - sqrt(...)|`；该实验只有在 `V3P-WSS-01/02` 全部 No-Go 时启动。

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
- 壁面点必须参与 `p` 和 WSS loss；是否同时强制参与速度 loss（无滑移软约束）以 `V3P-Diag-00` 的壁面真值统计为准（见 4.1.1）。
- 内部点不得参与 WSS loss。

当前代码已经对 WSS loss 使用 wall mask（`training/core/losses.py:53-77 wss_supervision_loss`）；V3 需要补齐的是 field loss 的双域 mask 口径，避免壁面速度与内部 WSS 的任务边界混读。

#### 4.1.1 壁面无滑移监督的数据依赖（必做前置）

> 本节为 2026-04-29 修订新增。

CFD 导出的壁面节点是否真的满足 `u = v = w = 0`，必须由 `V3P-Diag-00` 数据统计确认；不允许在未验证之前直接强制目标真值为 0：

| 壁面真值统计结果（`max|u|`, `max|v|`, `max|w|` 在训练集上的总最大） | 处置 |
| --- | --- | 
| `< 1e-6` | 4.2 中 `lambda_vel_noslip` 默认从 1.0 **降为 0.1**，叙事写"已 ≈ 0，noslip 仅作弱正则" |
| 介于 `1e-6` 和 `1e-3` | `lambda_vel_noslip = 0.5`；叙事维持"软无滑移监督" |
| `> 1e-3` | 4.1 文字必须改写为"软无滑移监督，目标取 CFD 真值而非恒 0"，`lambda_vel_noslip = 1.0`；说明这与 CFD 数值噪声 / 边界层离散化的关系 |

`V3P-Diag-00` 必须在执行前确定该值，并把对应的处置档位写入 `V3P-Base-01 / Main-01` 的训练配置 meta 中（如 `meta.noslip_decision = "weak_reg"` / `"soft"` / `"raw_truth"`）。

### 4.2 总损失

V3 默认总损失：

```text
L_total =
  lambda_vel_int    * L_interior_velocity  (内部点的MSE，均值分母为内部点数)
+ lambda_vel_noslip * L_noslip_velocity    (壁面点target=0或CFD真值的MSE，均值分母为壁面点数)
+ lambda_p_int      * L_interior_pressure
+ lambda_p_wall     * L_wall_pressure
+ lambda_wss        * L_wall_wss
```

第一版默认权重（实际取值受 4.1.1 与 4.2.1 的 Diag 决策约束）：

| 分项 | 默认权重 | 说明 |
| --- | ---: | --- |
| `lambda_vel_int` | 0.3 | 保留近壁速度上下文，但不让速度成为主导目标；若 4.2.1 中 `std/mean > 0.5` 则上调至 0.5 |
| `lambda_vel_noslip` | 由 4.1.1 决定（0.1 / 0.5 / 1.0） | 强约束需有壁面真值数据支持 |
| `lambda_p_int` | 0.5 | 保留内部压力学习 |
| `lambda_p_wall` | 1.0 | 壁面压力为 V3 主 readout 之一；若 4.2.1 显示该项 > 其它项 5×，自动降为 0.5 |
| `lambda_wss` | 0.1 起步（`V3P-WSS-01-*` 同时跑 0.05 / 0.1 / 0.2 三档） | 从轻量 WSS 权重起步，避免重现 V2 梯度劫持 |

> **【关键防偏移策略：空间解耦计算】**
> 所有分项**绝对不能**混在同一个大张量里算全局均值。必须严格按照各自的 mask 域，独立算出门当户对的标量 MSE Loss（如 `L_noslip_velocity` 的均值分母仅为 13000，`L_interior_velocity` 的均值分母仅为 2000），然后再乘以各自的权重相加。这样即使壁面点数量占据绝对优势，也不会"淹没"内部点的梯度损失。

#### 4.2.1 权重量级对齐（必做前置）

> 本节为 2026-04-29 修订新增。

5 个分项的均值分母不同（内部 2000、壁面 13000、全节点 15000），加上 V1 全节点 p loss 与 V3 壁面 p loss 不可比，因此默认权重在数量级上不一定平衡。`V3P-Diag-00` 必须输出 `weighted_loss_*`（即 `lambda_* × L_*`）五项在前 N 个 batch 上的实际数量级（mean / max / median），并按以下规则自动调档：

- 任一 `weighted_loss_*` 在中位数上**超过其它项 5×**：把该项权重 ÷2 并写入实验记录；
- `lambda_wss × L_wall_wss` 在 raw 状态下 > `data_field_loss` 的 2×（V2P-WSSP-02 的失败模式）：直接降到 0.05 档；
- 调档结果固化到 `V3P-Base-01 / Main-01` 配置的 `meta.weight_calibration`，并在论文方法节明确说明。

### 4.3 早停与调度

V3 不使用原始 `data_loss + early_stop_wss_weight * wss_loss` 作为第一轮早停指标（当前 trainer 的实现见 `training/core/trainer.py:267-280`）。

推荐验证分数：

```text
val_score =
  normalized_val_wall_wss_rmse
+ normalized_val_wall_p_rmse
+ 0.3 * normalized_val_near_wall_vel_rmse
```

其中 `normalized_val_near_wall_vel_rmse` 在 V3 第一阶段应理解为 `sampled_near_wall_interior` 诊断口径：该 mask 由 §2.1.1 的采样层等价关系直接派生，即 `is_wall == 0`（V3 锁定采样下所有内部点都来自 `dist_to_wall < 2mm` 的近壁采样层）；**无需新增 `is_near_wall` 节点字段**，无运行时开销。该口径不等价于历史 `regional_eval.near_wall = interior & NormRadius > 0.8`。归一化基准统一使用训练集 target 的 per-channel std，固化在 `processed/graphs/<split>/stats.json`，由 trainer 在 epoch 末读取。详细实现路径见 8.1 节。

> 历史草稿中的"interior_vel_rmse 替代"兜底语已删除（不再需要——`sampled_near_wall_interior` 与 `interior` 在 V3 锁定采样下本就等价）。归一化基准（`stats.json` 或兜底的静态 norm_consts）不到位时，`V3P-Main-01` 不得开跑。

---

## 5. V2 失败经验规避

V3 设计必须显式避开 V2P-WSSP 中已经暴露的问题。

### 5.1 避免 WSS 梯度劫持

`V2P-WSSP-02` 使用：

- `target_weights=[2,2,2,0.5]`
- `wss_loss_weight=0.5`
- `early_stop_wss_weight=1.0`

结果中 `wss_loss` 加权后远大于 `data_loss`，共享 backbone 被 WSS 梯度主导，压力预测崩溃（`r2_p ≈ 0.004`）。

V3 规避方式：

- 第一版 `lambda_wss` 在 `V3P-WSS-01-*` 中以 0.05 / 0.1 / 0.2 三档穷扫，最低档先行；
- `field_head` 与 `wss_head` 前必须设置独立的 MLP 缓冲层（见 3.3 节），避免 WSS 梯度直达 backbone；
- 训练日志同时记录 raw 与 weighted 两份 loss；
- 若 `weighted_wss_loss > 2 × weighted_field_loss` 连续出现 3 个 epoch 以上，立即停止该配置，不补 seed；
- 早停不直接使用未归一化 WSS loss（见 4.3）。

### 5.2 不再把"仅 p+WSS"作为主线

`V2P-WSSP-05/06` 三 seed 结果显示：

- `wss_r2_wss` 仍为略负；
- Huber 相对 MSE 未稳定占优；
- `r2_p` 跨 seed 方差大。

因此 V3 不把 `target_weights=[0,0,0,1] + WSS` 作为主路线，而是保留内部速度的弱监督，用近壁流动上下文辅助壁面 WSS。

### 5.3 Huber 仅作单变量对照，触发条件由数据决定

Huber 不是 V3 默认解法。`V3P-Diag-00` 必须输出训练集壁面 WSS 的异常值统计（详见 9.1.1），并按以下规则决定是否启动 Huber 对照：

- 若 `>3σ` 异常值占比 ≥ 5% 或 magnitude CDF 在 95 分位以上有显著长尾（p99/p50 > 5×），则启用 `V3P-WSS-03`（Huber WSS 单变量对照，`wss_loss_type=huber`，`wss_huber_beta=1.0`，其它配置同 `V3P-Main-01`）；
- 反之，跳过 Huber 对照，正文不出 Huber 实验，避免低信号实验占用篇幅；
- 该决策必须写入 `V3P-Diag-00` 报告并在 `V3P-WSS-03` 的 `meta.notes` 中引用。

### 5.4 避免新的架构归因错误

V3 第一阶段同时改变了主干叙事、监督口径和主指标，因此必须避免把所有改进都归因于"PointNeXt 更强"。

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

### 6.0 全新主线实验树（2026-05-01 修订）

V3 从本节开始视为任务 A 的**全新正式训练路径**。V2P-WSSP-* 不再作为日常推进主线，只作为历史探索与反例库引用：它证明了 WSS 权重过大会劫持训练、仅 `p+WSS` 不会自动解决 WSS、Huber 不是稳定解法。V3 的目标不是补齐 V2 的所有分支，而是用一棵能被复述、能被汇报、能定位瓶颈的实验树重新组织问题。

V3 第一阶段只回答四类问题，必须按层推进：

1. **诊断层**：数据、mask、target 标准差、WSS 分布和 loss 数量级是否可信。
2. **单目标上限层**：压力、速度、WSS 在互不干扰时各自能达到什么上限。
3. **关键双目标层**：目标之间是互相辅助还是互相竞争，尤其是速度上下文是否帮助 WSS。
4. **正式主线层**：在确认目标可学性和冲突模式后，再跑双域 mask + 几何先验 + head 解耦的正式 V3 主线。

逻辑树如下：

```text
V3P-Diag-00
  ├─ 若 mask / 归一化 / loss 尺度不可信：停止，先修数据与训练口径
  └─ 若诊断通过：
       ├─ 单目标上限层
       │    ├─ V3P-Probe-P-01      只监督压力
       │    ├─ V3P-Probe-V-01      只监督速度
       │    └─ V3P-Probe-WSS-01    只监督 WSS
       ├─ 关键双目标层
       │    ├─ V3P-Probe-PWSS-01   压力 + WSS
       │    ├─ V3P-Probe-VP-01     速度 + 压力
       │    └─ V3P-Probe-VWSS-01   速度 + WSS
       └─ 正式主线层
            ├─ V3P-Anchor-01       同采样 V1 锚点
            ├─ V3P-Base-01         无几何 PointNeXt 下限
            ├─ V3P-Main-01         几何 PointNeXt 主线
            └─ V3P-WSS-*           WSS 权重 / Huber / 冗余消解条件实验
```

#### 6.0.1 单目标与双目标诊断的判读规则

| 观察结果 | 结论指向 | 后续动作 |
| --- | --- | --- |
| `V3P-Probe-P-01` 压力仍低 | 当前采样、模型或归一化对压力都不成立，不是 WSS 干扰问题 | 暂停正式主线，优先查 target 标准化、壁面/内部压力分布、采样口径 |
| `V3P-Probe-WSS-01` 仍低 | WSS 在当前输入和标签下本身难学，瓶颈更可能在 WSS 对齐、壁面几何、近壁采样或 head/loss | 不急着堆多任务，先做 WSS 标签与壁面区域诊断 |
| `V3P-Probe-WSS-01` 好，但 `V3P-Probe-PWSS-01` 差 | 典型多任务冲突或 loss 尺度失衡 | 保留 head 解耦、降低 `lambda_wss`，必要时引入 GradNorm / PCGrad 到第二阶段 |
| `V3P-Probe-VWSS-01` 好于 `V3P-Probe-WSS-01` | 近壁速度上下文确实帮助 WSS | V3 正式主线保留弱速度监督，`lambda_vel_int` 可进入小范围扫描 |
| `V3P-Probe-V-01` 在 wall13000+near2000 下很差 | 当前采样不适合完整速度场重建 | 速度降级为近壁上下文与诊断项，不再作为主论文指标 |
| `V3P-Probe-PWSS-01` 与历史 V2P-WSSP-05/06 一样差 | 问题不是 V2 偶然超参，而是 `p+WSS` 组合本身在当前口径下不稳 | 正式主线必须采用双域 mask、target std 归一化和轻量 WSS 权重 |

#### 6.0.2 执行纪律

- 所有 `V3P-Probe-*` 先只跑 `seed=1`，只用于定位瓶颈，不直接写成论文主结果。
- 只有当某个 probe 出现正信号，才补 `seed=2/3`；明确负结果不补 seed。
- `V3P-Probe-PWSS-01` 与历史 `V2P-WSSP-05/06` 目的相似，但 V3 版必须使用 V3 统一图资产、统一命名、统一后处理和统一诊断表；若实现成本过高，可先引用 V2P-WSSP-05/06 作为历史反例，不强制重复。
- `V3P-Anchor-01 / Base-01 / Main-01` 不得早于 `V3P-Diag-00` 与三个单目标 probe 完成；否则无法判断正式主线失败时到底是单目标不可学还是多任务冲突。
- 本节实验树优先级高于旧 §6.1 矩阵中的平铺顺序；旧矩阵保留为具体配置清单，但实际排队以本节为准。

### 6.1 第一轮核心实验

> 表头新增 `backbone / 采样 / head / 归一化` 四列。2026-05-01 修订后，第一轮不再直接从 `Anchor/Base/Main` 起跑，而是先完成 `Diag + Probe`，再进入正式主线。

| Exp ID | 目的 | backbone | 采样 | head 结构 | 归一化基准 | 输入 | 损失口径 | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `V3P-Diag-00` | 诊断双域 mask、loss 尺度、壁面真值与 WSS 分布 | PointNeXt-localpool | wall13000+near2000 | 单层（与 V2P 对齐） | 无（仅诊断） | 同 Main | 仅跑 1 epoch，输出 9.1.1 全部诊断字段 | `1` |
| `V3P-Probe-P-01` | 压力单目标上限 | PointNeXt-localpool | wall13000+near2000 | 2 层 MLP | per-channel std | 同 Main | 只监督 `p`：`lambda_p_int/lambda_p_wall > 0`，速度与 WSS 关闭 | `1 → 条件补 [2,3]` |
| `V3P-Probe-V-01` | 速度单目标上限 | PointNeXt-localpool | wall13000+near2000 | 2 层 MLP | per-channel std | 同 Main | 只监督内部速度：`lambda_vel_int > 0`，压力与 WSS 关闭 | `1 → 条件补 [2,3]` |
| `V3P-Probe-WSS-01` | WSS 单目标上限 | PointNeXt-localpool | wall13000+near2000 | 2 层 MLP | per-channel std | 同 Main | 只监督壁面 WSS：`lambda_wss > 0`，速度与压力关闭 | `1 → 条件补 [2,3]` |
| `V3P-Probe-PWSS-01` | 压力与 WSS 的相互干扰诊断 | PointNeXt-localpool | wall13000+near2000 | 2 层 MLP | per-channel std | 同 Main | 只监督 `p + WSS`，速度关闭；用于对照 V2P-WSSP-05/06 | `1 → 条件补 [2,3]` |
| `V3P-Probe-VP-01` | 速度与压力的 field 任务诊断 | PointNeXt-localpool | wall13000+near2000 | 2 层 MLP | per-channel std | 同 Main | 只监督内部速度 + 压力，WSS 关闭 | `1 → 条件补 [2,3]` |
| `V3P-Probe-VWSS-01` | 近壁速度上下文是否帮助 WSS | PointNeXt-localpool | wall13000+near2000 | 2 层 MLP | per-channel std | 同 Main | 只监督内部速度 + 壁面 WSS，压力关闭；`lambda_wss` 从低档开始 | `1 → 条件补 [2,3]` |
| `V3P-Anchor-01` | 同采样下的 V1 公平锚点 | Transformer (`A-Opt-05`) | **wall13000+near2000** | 单层 | 无 | `coords + t + BC + geometry + is_wall` | 全场监督 `target_weights=[2,2,2,0.5]` + `wss_loss_weight=0.1` | `1 → [1,2,3]` |
| `V3P-Base-01` | 无几何 PointNeXt 下限 | PointNeXt-localpool | wall13000+near2000 | 2 层 MLP | per-channel std | `coords + t + BC + is_wall` | 双域 mask loss，`lambda_wss=0.1` | `1 → [1,2,3]` |
| `V3P-Main-01` | 验证中心线几何先验 | PointNeXt-localpool | wall13000+near2000 | 2 层 MLP | per-channel std | `Base + Abscissa + NormRadius + Curvature + Tangent` | 同 Base | `1 → [1,2,3]` |
| `V3P-WSS-01-a` | WSS 权重轻量穷扫（低档） | PointNeXt-localpool | wall13000+near2000 | 2 层 MLP | per-channel std | 同 Main | `lambda_wss=0.05`，其余同 Main | `1 → [1,2,3]` |
| `V3P-WSS-01-b` | WSS 权重轻量穷扫（中档） | 同上 | 同上 | 同上 | 同上 | 同 Main | `lambda_wss=0.10`，其余同 Main | `1 → [1,2,3]` |
| `V3P-WSS-01-c` | WSS 权重轻量穷扫（上档） | 同上 | 同上 | 同上 | 同上 | 同 Main | `lambda_wss=0.20`，其余同 Main | `1 → [1,2,3]` |
| `V3P-WSS-02` | 速度上下文增强 | 同上 | 同上 | 同上 | 同上 | 同 Main | `lambda_vel_int=0.5`，其余同 Main | `1 → [1,2,3]` |
| `V3P-WSS-03` | Huber WSS 单变量对照（条件执行，门槛见 5.3） | 同上 | 同上 | 同上 | 同上 | 同 Main | 仅 `L_wall_wss` 改为 Huber | 仅在 5.3 触发条件满足时跑 |
| `V3P-WSS-04` | WSS 4 维冗余消解（条件执行，门槛见 3.3 末段） | 同上 | 同上 | 同上 | 同上 | 同 Main | `wss_head` 改为 3 分量 + magnitude 一致性约束 | 仅在 V3P-WSS-01/02 全 No-Go 且 3.3 诊断触发时跑 |

执行规则：

- `V3P-Diag-00` 必须先于正式训练执行，用于确认 mask、loss 尺度、壁面真值统计、WSS 分布、归一化基准与 §2.1.1 mask 等价关系（详见 9.1.1）。
- `V3P-Probe-P/V/WSS` 三个单目标实验必须先于 `V3P-Anchor-01 / Base-01 / Main-01` 完成；否则正式主线的失败无法归因。
- `V3P-Probe-PWSS/VP/VWSS` 原则上在三个单目标 probe 之后执行；若算力有限，优先 `V3P-Probe-VWSS-01`，其次 `V3P-Probe-PWSS-01`，最后 `V3P-Probe-VP-01`。
- 每组先跑 `seed=1`；probe 只在出现正信号或判读存在争议时补 `seed=2/3`。
- `V3P-WSS-01-a/b/c` 三档同时入队（不再"动态决定"），构成完整 ablation；论文方法节直接报告三档。
- 只有 `V3P-WSS-01-*` 中至少一档同时满足"压力不崩 + WSS 有正信号"，才补 `V3P-WSS-02`。
- 单 seed 明确负结果不补 seed，先回到 loss 和数据分布排查。

### 6.1.1 公平对照要求

`V3P-Anchor-01`、`V3P-Base-01`、`V3P-Main-01` 三组之间必须保持以下项完全一致：

- 同一数据版本与图资产（`split_AG_v1` + wall13000+near2000）。
- 同一采样口径。
- 同一 split。
- 同一训练轮数、优化器、学习率、batch size、早停策略。
- 同一后处理脚本。

变化项：

- `V3P-Anchor-01` ↔ `V3P-Base-01`：唯一变化为 backbone（Transformer vs PointNeXt-localpool）+ 监督口径（全场 + WSS 0.1 vs 双域 mask + WSS 0.1）。两者用于回答"V3 配方相对 V1 配方在同采样下是否成立"。
- `V3P-Base-01` ↔ `V3P-Main-01`：唯一变化为是否启用中心线几何先验。两者用于回答"几何先验在 V3 配方下是否仍有增益"。
- 任何其它差异都必须升级为新一行实验，否则不能把差异归因于 backbone / geometry。

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

### 7.0 指标量纲口径（必读）

> 本节为 2026-05-01 修订新增。仓库的训练标签在 `pipeline/normalize.py:301-307` 已经做了**全局 z-score 归一化**（mean/std 来自训练集），所以训练 / 验证 / 测试时的所有 RMSE 和 R² **默认在 z-score 空间下计算**（`training/core/metrics.py:148-164`、`metrics.py:177-218`）。这意味着：

| 维度 | z-score 空间下的实际含义 | 物理量纲（Pa / m·s⁻¹）下的对应数值 |
| --- | --- | --- |
| `rmse_p`, `rmse_u/v/w` | "误差占该 channel 训练集 std 的倍数" | `rmse_z * std`（从 `<data_root>/normalization_params_global.json` 反 transform） |
| `wss.rmse_*` | "误差占壁面节点 wss 训练集 std 的倍数" | `rmse_z * wall_std` |
| `r2_*` | 在 z-score 空间内，与物理空间下的 R² **数学等价**（z-score 是仿射变换，R² 不变） | 同 z-score 空间数值 |

**叙事约定（论文方法节必须写明）**：

- V3 主表的所有 `rmse_*` / `wss.rmse_*` 数值统一以 z-score 空间报告，与 V1 / V2 主表口径一致；与历史结果（如 `A-Opt-05.RMSE_p=0.6449`、`V2P-WSSP-05.r2_p=0.215`）可直接逐表比较。
- 凡涉及临床或病例级叙事（如"高 WSS 区域分布、p95 WSS"）的指标，**必须用反归一化后的物理量纲**，避免出现"z-score=1.5 的 WSS"之类无物理意义的报告。
- `R²` 在 z-score 与物理空间下数值相等，所以 `r2_p ≥ 0.85`、`wall.r2_wss` 阈值在两种口径下都适用（无需特别注明）。

**相关脚本与字段**：

- 训练 / 验证 metrics：`training/core/metrics.py:12-218` 全程在 z-score 空间。
- 后处理输出：`predict_field.py` 写入的 `predictions_test/*.pt` 同样是 z-score 空间的预测值；`summary.json.test_metrics` 中 `rmse_*` 也是 z-score 空间。
- 反归一化所需统计量来源：`<data_root>/normalization_params_global.json` 中 `statistics["u"|"v"|"w"|"p"|"wss"|"wss_x"|...].std`（`pipeline/normalize.py:443-465`）。
- V3 P1-A `eval_wss_clinical_metrics.py` **必须**在脚本内部对 pred 和 target 同时做 inverse z-score 后再算病例级 mean / p95 WSS（详见 §8.5）。

### 7.1 主指标顺序与实现位置

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

> **mask 等价关系提示（来自 §2.1.1）**：在 V3 锁定采样下 `sampled_near_wall_interior_mask ≡ interior_mask`，因此第 5/6 行若按 V3 专属采样诊断口径解释为 `sampled_near_wall_interior.*`，将与第 7/8 行的 `interior.*` 数值上**逐 seed 完全相同**。两者并列保留时，论文方法节必须明确写"V3 采样使 2mm 近壁内部采样点与全部内部点在数值上等价，第 7/8 行作为备份显示"，避免审稿人认为是重复列。注意：历史 Fig A5 / `regional_eval.near_wall` 仍按 `NormRadius > 0.8` 定义，不应与本段采样诊断口径混用。若未来放开 `boundary_core_ratio` 把核心区点重新引入，第 5/6 行与 7/8 行将出现差异，届时主表口径需重新评估。

全局 `RMSE_|v|` 不作为 V3 第一排序指标。

WSS 指标必须拆成三层报告，每层标注实现位置：

| 层级 | 指标 | 目的 | 实现位置（脚本） |
| --- | --- | --- | --- |
| 点级 | `wall.rmse_wss`, `wall.r2_wss` | 判断壁面点预测误差 | 现有 `training/scripts/_eval_wss_metrics_once.py` + `training/scripts/plot_taskA_regional_bar.py --wss` |
| 病例级 | mean WSS、p95 WSS 的 Pearson/Spearman | 判断病例排序和临床统计量是否保留 | **待新增** `training/scripts/eval_wss_clinical_metrics.py`（V3 P1 前置任务，见 8.5） |
| 区域级 | 高 WSS top-k overlap / Dice | 判断高剪切区域定位能力 | **待新增** 扩展 `_eval_wss_metrics_once.py`（V3 P1 前置任务，见 8.5） |

若点级 R² 改善但病例级或区域级不改善，不得宣称临床相关 WSS surrogate 已成立。

### 7.2 Go 标准

V3 进入后续优化的最低标准（同采样、同 split 内部对照；所有 RMSE / R² 阈值均按 §7.0 口径为 z-score 空间，但 R² 在 z-score 与物理空间下数学等价）：

- `V3P-Main-01` 相对 `V3P-Base-01` 在 `wall.r2_wss` 或 `wall.rmse_wss` 上改善（三 seed 均值、paired t 在 0.1 显著性内）。
- `V3P-Main-01` 的 `wall.r2_p` 不明显退化：**`r2_p ≥ 0.85`**（z-score 与物理空间等价），且相对 `V3P-Base-01` 下降不超过 0.05；低于该阈值视为压力崩溃（参考 V2P-WSSP-02 `r2_p ≈ 0.004` 的失败模式）。
- `near_wall.rmse_vel_mag` 不崩坏（相对 `V3P-Base-01` 上升不超过 10%；该 RMSE 是 z-score 空间下"训练集速度 std 的倍数"，10% 改动相当于约 0.1× 训练集速度 std）。
- 几何先验增益至少在多数 seed 或多数病例上重复出现。

V3 宣称 WSS 突破的标准（同口径，wall13000+near2000）：

- 三 seed `wall.r2_wss` 均值稳定超过 **`V3P-Anchor-01`** 三 seed 均值（即同采样下的 V1 锚点）；
- 同时 `wall.r2_p ≥ 0.85`；
- WSS 提升不是以压力或近壁速度明显崩坏为代价；
- 论文中 V1 默认采样下的 `A-Opt-05-wss-multi` 三 seed `wall.r2_wss ≈ 0.46` **仅作为参考线**，不作为 Go 阈值（采样口径不同，不可直接比）。

### 7.3 No-Go 标准

出现以下任一情况，应停止继续堆结构：

- `weighted_wss_loss` 长期大于 field loss 两倍以上。
- `wall.r2_p < 0.85` 或相对 `V3P-Base-01` 下降超过 0.05。
- `V3P-Main-01` 相对 `V3P-Base-01` 无几何增益。
- WSS 只在个别病例变好，病例级均值和分位数指标不改善。
- seed 方差过大，无法稳定复现。

---

## 8. 实现注意事项

### 8.1 建议新增配置字段与归一化路径

建议后续实现时新增：

```text
optim.domain_loss.enabled
optim.domain_loss.lambda_vel_int
optim.domain_loss.lambda_vel_noslip
optim.domain_loss.lambda_p_int
optim.domain_loss.lambda_p_wall
optim.domain_loss.lambda_wss
optim.domain_loss.normalize_by_target_std
optim.domain_loss.norm_consts        # 兜底：静态写入的 per-channel std
optim.domain_loss.weight_calibration # 4.2.1 自动调档结果
```

> 历史草稿中的 `data.use_near_wall_mask` 字段已删除——若指标意图是 V3 专属的 `sampled_near_wall_interior` 采样诊断口径，可直接使用 `is_wall == 0` 派生（详见 §2.1.1），无需额外字段或图资产版本；这不改变历史 `regional_eval.near_wall` 的 `NormRadius > 0.8` 定义。

第一版可保持向后兼容：未开启 `domain_loss.enabled` 时，沿用现有 loss 行为。

#### `normalize_by_target_std` 实现路径（V3 P0 前置工作）

> **2026-05-01 重要澄清**：仓库的 pipeline **已经做了完整的全局 z-score 归一化**（详见 `pipeline/normalize.py:85-94, 191-203, 301-307`），训练时 `data.y` 与 `data.y_wss` **已经是 z-score 空间数值**（即每个 channel 的训练集 std≈1）。所以 V3 的 `normalize_by_target_std` 不是"从零做归一化"，而是"在 z-score 空间下、对验证分项 RMSE 再做一次跨 channel 量级归一化以构造 `val_score`"。下方实现路径已据此修正。

#### 已存在的 pipeline 归一化（无需新增）

| 内容 | 文件:行 | 输出 |
| --- | --- | --- |
| 训练集 `u/v/w/p` 全局 z-score | `pipeline/normalize.py:301-307` | 写入 `processed/normalized/` 下的 CSV |
| 训练集 `wss/wss_x/y/z` 全局 z-score（**仅用壁面节点统计 mean/std**） | `pipeline/normalize.py:191-203` + `301-307` | 同上 |
| 持久化 mean/std/min/max | `pipeline/normalize.py:443-465 save_normalization_params` | `<data_root>/normalization_params_global.json` |
| 转图时把 z-score 后的 `y / y_wss` 装入 `Data` | `pipeline/convert_to_graph.py:146-158` | `processed/graphs/<split>/*.pt` 中的 `data.y` / `data.y_wss` 已是 z-score 空间 |

#### V3 真正要做的改造（实际 P0 工作）

- **(a) 复用 `normalization_params_global.json`，不再生成新文件**：trainer 在 epoch 末从该 JSON 的 `statistics[*].std` 读取 per-channel std；不要新建 `processed/graphs/<split>/stats.json`，避免出现两份"训练集 std"互相不一致的情况。
- **(b) trainer 适配**：`training/core/trainer.py:267-280` 在 `optim.domain_loss.normalize_by_target_std=True` 时，把 `val_score` 各分项 RMSE 除以对应 channel std 再求加权和；旧路径完全保留（详见 §11.0 向后兼容硬约束）。
- **(c) 兜底方案**：若直接读 JSON 在某些机器上有路径问题，可在 `V3P-Diag-00` 阶段把 `statistics[u/v/w/p/wss/...].std` 复制到 `optim.domain_loss.norm_consts`，trainer 直接读配置；视为临时方案。
- **(d) 注意**：因为训练标签已经 z-score（理论上 std≈1），所以 (a) 中的归一化系数对各分项 RMSE 的"放缩"幅度本来就接近 1。V3 这一步主要是**显式把"z-score 空间"作为对外承诺写进 trainer**，避免日后有人误以为 RMSE 是物理量纲值；同时保证 wss / p / vel 的 RMSE 在 `val_score` 里量级可比。

> **硬约束**：归一化系数不论来自 (a) 文件读取还是 (c) 兜底配置，都必须出现在 `history.csv` 的元信息列里（如 `norm_std_p / norm_std_wss / ...`），便于事后审计；不到位时 `V3P-Main-01` 不得开跑。

> **数据真相提示**：`pipeline/normalize.py:191-203` 中 `wss/wss_x/y/z` 仅用壁面节点统计 mean/std，但 transform（`line 301-307`）应用到所有节点。CFD 内部点原始 `wss=0`，z-score 后变为 `(0 - wall_mean)/wall_std`，是非零"伪值"。当前 `wss_supervision_loss`（`losses.py:53-77`）和 `WSSMeter`（`metrics.py:177-180`）都做了 wall_mask，所以训练 / 评估时无害；但 V3 P0-D 的 `dual_domain_point_loss` **必须显式 wall_mask**，否则会被该伪值污染（详见 §8.2 末段警告）。

### 8.2 建议新增 loss 逻辑

建议在 `training/core/losses.py` 中新增 `dual_domain_point_loss`，专门服务 V3。

必须记录以下分项：

- `loss_interior_velocity`
- `loss_noslip_velocity`
- `loss_interior_pressure`
- `loss_wall_pressure`
- `loss_wall_wss`
- `weighted_loss_interior_velocity`
- `weighted_loss_noslip_velocity`
- `weighted_loss_interior_pressure`
- `weighted_loss_wall_pressure`
- `weighted_loss_wall_wss`

#### WSS 伪值陷阱警告（必读）

> 本节为 2026-05-01 修订新增。

`pipeline/normalize.py:191-203` 中 `wss / wss_x / wss_y / wss_z` 的 mean/std **仅用壁面节点**统计；但归一化 transform（`pipeline/normalize.py:301-307`）应用到**所有节点**，包括内部点。CFD 内部点本身没有 WSS，原始数据中是 `0`（或被 `pipeline/normalize.py:312-314` 的 NaN→0 填充覆盖）。z-score 后内部点的 `y_wss` 变成 `(0 - wall_mean) / wall_std`，是**非零的"伪值"**——如果有任何 loss 或 metric 在内部点上用了这个值，会被严重污染。

当前仓库的两处用法都是安全的：

- `wss_supervision_loss`（`training/core/losses.py:53-77`）：用 `wall_mask` 屏蔽内部点；
- `WSSMeter`（`training/core/metrics.py:177-180`）：同样用 `wall_mask` 屏蔽。

V3 P0-D `dual_domain_point_loss` 的实现必须遵守的硬约束：

1. `loss_wall_wss` / `loss_wall_pressure` **必须**只在 `is_wall == 1` 的节点上求均值（分母 = 壁面点数 ≈ 13000），与现有 `wss_supervision_loss` 行为一致；
2. **不允许**对 `data.y_wss` 全节点直接求 MSE（即使乘了 0.1 的小权重）；
3. 单元测试要求：在一个 dummy batch 上断言"内部点的 `y_wss` 均值显著非零"且"`loss_wall_wss` 的梯度只回流到 `wss_head` 在壁面节点上的输出"；CI 应包含此用例（详见 §11.0 末段）。

> **可选优化**（V3 第二阶段）：在 `pipeline/normalize.py` 中把内部点的 `y_wss` 设为 NaN 或 0 之后再 z-score，并在加载图时把 `is_wall == 0` 的 `y_wss` 显式置零；这样即便有人忘记 wall_mask，loss 也不会被污染。但这会破坏向后兼容（旧图资产 `y_wss` 已写入），暂不在 V3 第一阶段做。

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

#### V3 run 目录命名规范

V3 第一阶段 run 目录必须含 `_localpool` 后缀，便于二次检索；如未来开 `V3P-Arch-*` 第二阶段，则用 `_hier` 后缀。示例：

- `outputs/field/field_v3_pointnext_localpool_main01_geom_wall13000_near2000_split_AG_v1_seed1_<timestamp>/`
- `outputs/field/field_v3_pointnext_localpool_anchor01_transformer_wall13000_near2000_split_AG_v1_seed1_<timestamp>/`（虽然 backbone 是 Transformer，但仍归属 V3 矩阵，目录前缀写 `field_v3_*`）

`outputs/field/experiment_index.csv` 同步新增列：

- `backbone_variant`：`pointnext-localpool` / `pointnext-hier` / `transformer-prenorm`
- `sampling_profile`：`wall13000_near2000` / `default`
- `head_layout`：`single_linear` / `mlp2` / `mlp3`

#### checkpoint 策略

V3 训练必须同时保存：

- `best_epoch.ckpt`（按 4.3 的 `val_score` 选取）；
- `final_epoch.ckpt`（最后一个 epoch，便于 `V3P-Diag-00` 排查"未充分训练"或"早停过早"）。

### 8.4 采样与模型结构的解决路径

当前最大实现风险是：预处理已经使用 FPS/随机混合采样，但模型侧仍是固定采样点上的局部 kNN pooling。该风险的解决路径分两步：

1. **V3 第一阶段不改 backbone 拓扑结构（FPS / SA / FP），允许 head 升级（见 3.1.1 白名单）。**
   将当前实现明确命名为 `PointNeXt-style local pooling`。采样层面的 FPS 用于保证输入点云空间覆盖，不把它等同于模型内部层次化 PointNeXt。

2. **V3 第二阶段再决定是否实现完整层次化 PointNeXt。**
   只有当 `V3P-Main-01` 和 `V3P-WSS-*` 证明双域 loss 有正信号后，才开放 `V3P-Arch-*`，引入模型侧 FPS/radius grouping/set abstraction。这样可以避免第一轮同时改变主干结构和 loss，导致归因混乱。

若后续实现完整层次化 PointNeXt，必须与 `V3P-Main-01` 使用同一数据、split、loss 和评估口径，只把模型结构作为唯一变化项。

### 8.5 待新增的评估脚本（V3 P1 前置任务）

> 本节为 2026-04-29 修订新增。7.1 主指标表中的病例级与区域级 WSS 指标当前没有对应实现脚本，必须在 `V3P-Main-01` 提交论文表前完成。

| 脚本 / 改动 | 负责的指标 | 依赖输入 | 输出 |
| --- | --- | --- | --- |
| 新增 `training/scripts/eval_wss_clinical_metrics.py` | 病例级 mean WSS / p95 WSS 的 Pearson、Spearman；逐病例 box/strip plot；**输出物理量纲（Pa）数值** | 各 run 的 `predictions_test/*.pt` + 测试 split 病例列表 + `<data_root>/normalization_params_global.json`（用于反 z-score） | `predictions_test/wss_clinical/wss_clinical_metrics.json` + `fig_case_level_wss.png` |
| 扩展 `training/scripts/_eval_wss_metrics_once.py` | 高 WSS top-k overlap / Dice（top-k 阈值由 `V3P-Diag-00` 训练集统计确定，**阈值取在物理量纲下**而非 z-score 空间） | 同上 | `predictions_test/regional_eval/fig_A5_regional_wss_metrics.json` 中新增字段 `top_k_overlap`、`top_k_dice` |
| 扩展 `training/scripts/plot_taskA_regional_bar.py --wss` | 在现有柱图上叠加 top-k overlap / Dice 列 | 同上 | 现有图件追加新分组 |

实现顺序：先 `eval_wss_clinical_metrics.py`（独立脚本，不影响现有产物），后扩展 `_eval_wss_metrics_once.py`（影响所有历史 V2P run 重跑后处理时的输出格式，需在 PR 中明确兼容性策略）。

#### 反归一化（inverse z-score）实现要点

> 本小节为 2026-05-01 修订新增，确保 §7.0 量纲口径的临床叙事链路真实可落地。

任何输出"物理量纲数值"的脚本必须遵守以下接口约定：

1. 入参显式接收 `--norm-params <data_root>/normalization_params_global.json`（默认值可从 `summary.json.run.data_root` 推导，但必须可覆盖）；
2. 在脚本顶层定义 `denormalize_field(arr, channel: str, params: Dict)`：`return arr * params['statistics'][channel]['std'] + params['statistics'][channel]['mean']`；对 `wss_x/y/z` 同样处理；`wss`（标量 magnitude）单独反归一化（其 wall_mean / wall_std 与分量不同）；
3. 反归一化**必须**在 pred 与 target 上同时执行，不允许只对其中一个反归一化；
4. 输出 JSON 中显式记录 `"unit": "Pa"`（压力 / WSS）或 `"unit": "m/s"`（速度），避免后续读者误读；
5. `predict_field.py` 输出的 `predictions_test/*.pt` 仍保持 z-score 空间（向后兼容）；反归一化只在评估 / 出图阶段做。

实现校验（CI 建议项）：

- 取一个测试病例的 `predictions_test/*.pt`，断言 `wss` 在反归一化后落在 [0, ~10] Pa 量级（生理合理范围）；
- 断言 `p` 反归一化后落在 [0, ~30000] Pa 量级（与 `pipeline/config.py:184 outlet_pressure.offset=15000, scale=1000` 对应的训练分布一致）。

### 8.6 V3 增强策略

> 本节为 2026-04-29 修订新增。

`V2P-WSSP-07` 配置（`augment.rotation_prob=0.5 / translation_prob=0.5`）中，3D 旋转增强会在每个 epoch 改变坐标，但 `edge_index` 是预处理时构建的（基于固定采样点），增强后的坐标与 `edge_index` 的拓扑可能不再对齐；同时坐标已 PCA 对齐，rotation 会破坏 PCA 方向。

V3 第一阶段建议：

- **关闭 `augment.rotation_prob`**（设为 0.0）；
- 保留 `translation`（小幅度，0.1）；
- `mirror_prob = 0.0`（同 V2P-WSSP-07）；
- `scale_prob = 0.0`。

`V3P-Diag-00` 必须同时输出"启用 augment（`rotation=0.5`，复刻 V2P-WSSP-07）"与"关闭 augment.rotation"在首 epoch 的 `L_total` 与 `weighted_loss_*` 差异，作为最终是否完全禁用 augment.rotation 的依据；最终决策写入 `V3P-Base-01 / Main-01` 的 `meta.augment_decision`。

---

## 9. 测试计划

### 9.1 单 batch smoke test

必须确认：

- `wall_mask` 和 `interior_mask` 数量非零；典型形态 ≈ 13000 / 2000。
- 内部点不进入 WSS loss。
- 壁面点不进入"内部速度 loss"（但允许进入 noslip 速度 loss，依 4.1.1 决策档位）。
- 五个 loss 分项均能单独计算并写入日志。
- 采样元信息或图转换报告中可追溯当前图资产来自 hybrid/FPS/random 哪一种采样口径，且 `boundary_core_ratio` 与 `allow_core_fallback` 与 §2.1.1 锁定值一致。
- 当前模型实际读取 `edge_index` 做局部 pooling，实验记录中需标明 `neighbour_source=knn_edge_index`。
- **mask 等价关系自检**：在样本图上验证所有 `is_wall == 0` 的节点其 `dist_to_wall < boundary_threshold`（即 V3P-Diag-00 必须输出 `interior_max_dist_to_wall`，并断言 < 2mm），否则锁定项被破坏，`V3P-Main-01` 不得开跑。

### 9.1.1 V3P-Diag-00 强制输出（数据分析）

> 本节为 2026-04-29 修订新增。`V3P-Diag-00` 不只是"smoke test"，它的诊断输出会**直接决定** `V3P-Probe-* / Base-01 / Main-01` 的若干默认权重档位。该实验必须先于 `V3P-Probe-* / Anchor-01 / Base-01 / Main-01` 完成。

输出位置：`outputs/field/diagnostics/v3p_diag00_<seed>/`。

必须输出的字段：

| 字段 | 来源 | 决定项 |
| --- | --- | --- |
| 训练集壁面节点 `u/v/w` 的 `mean / std / max|·|`（全数据集与逐病例） | 训练集统计 | 4.1.1 `lambda_vel_noslip` 档位决策 |
| 训练集 `y_field`、`y_wss` 的 per-channel std | 训练集统计 | 8.1 (c) 兜底 norm_consts 数值；保存为 `stats.json` 草稿 |
| 训练集壁面 WSS 各维 + magnitude 的直方图、CDF | 训练集统计 | 5.3 Huber 触发判定；C7 一致性诊断的 baseline |
| 壁面 WSS 各维 `>3σ` / `>5σ` 异常值占比 | 训练集统计 | 同上 |
| 跨病例的 mean WSS / p95 WSS 分布 | 训练集统计 | 7.1 区域级 top-k 阈值取值（C6） |
| 5 个 `weighted_loss_*` 分项前 N 个 batch 的 mean / max / median | 训练首 epoch 监控 | 4.2.1 自动权重调档 |
| `L_interior_velocity` 跨 batch 的 std/mean | 训练首 epoch 监控 | I4：决定 `lambda_vel_int` 是否上调到 0.5 |
| augment.rotation on/off 的首 epoch `L_total` 差异 | 训练首 epoch 监控 | 8.6 `augment_decision` |
| `pred[wss_mag]` 与 `sqrt(pred[wss_x]^2+pred[wss_y]^2+pred[wss_z]^2)` 相对误差 CDF | 测试集预测一次 | 3.3 末段 `V3P-WSS-04` 触发判定 |
| 内部点 `dist_to_wall` 分布（含 max / 95 分位 / 中位数）+ 跨病例汇总 | 训练 + 测试集统计 | 验证 §2.1.1 mask 等价关系（必须 `max < boundary_threshold = 2.0mm`，否则报错并阻断后续 Anchor/Base/Main 训练） |
| **训练集 vs 测试集** 各 z-score channel（`u/v/w/p/wss/wss_x/y/z/Curvature`）的 99 / 99.5 分位差异 | 已 z-score 数据集（`processed/normalized/`） | OOD 警示：若任一 channel 测试集 99 分位绝对值 > 训练集 99 分位绝对值 × 1.5，输出红色警告并写入 §12 论文 limitations 段；不阻断训练，但要求在 V3 论文中明列 |
| **训练集 vs 测试集** `NormRadius` 的 max 比较（`pipeline/normalize.py:124-129` 无 clamp） | 已归一化数据集 | 若测试集 `NormRadius` > 1.0（min-max 缩放后超出训练集范围），输出红色警告；在论文方法节明示是否做了 clamp |
| **训练集 vs 测试集** BC（`BC_Inlet * 1e5`、`BC_O1~O4` z-score 后）的 5/95 分位 + max | 已归一化数据集 + `bc_metadata_normalized.json` | BC OOD 警示：测试病例的入口流量 / 出口压力是否落在训练分布内；若严重 OOD，论文 limitations 须说明 |
| `<data_root>/normalization_params_global.json` 与当前 split `--train-split` 一致性核查（避免使用其它 split 的 stats） | 文件读取 | 若不一致，立即报错；P0-B 校验任务的运行时入口 |

输出文件清单（最少）：

- `wss_distribution_train.json` + `fig_wss_dist_*.png`
- `wall_velocity_truth.json`
- `weighted_loss_calibration.json`
- `interior_dist_to_wall_stats.json`（验证 §2.1.1 mask 等价关系）
- `wss_magnitude_consistency.json`
- `noslip_decision.txt` / `weight_calibration.txt` / `augment_decision.txt`（最终落档结果）
- `train_test_distribution_diff.json`（训练 vs 测试各 channel 99/99.5 分位、`NormRadius` max、BC 分位；新增）
- `norm_params_consistency.txt`（与 `<data_root>/normalization_params_global.json` 一致性校验结果；新增）
- `stats_draft.json`（**仅在 8.1 (c) 兜底场景**写入；正常情况下直接复用 `<data_root>/normalization_params_global.json`）

### 9.2 一轮训练 smoke test

先执行：

- `V3P-Diag-00_seed1`（输出 9.1.1 全部字段；据此确定 `V3P-Probe-* / Anchor-01 / Base-01 / Main-01` 的默认权重档位）
- `V3P-Probe-P-01_seed1`
- `V3P-Probe-V-01_seed1`
- `V3P-Probe-WSS-01_seed1`

若三个单目标 probe 中至少压力或 WSS 诊断出现可解释信号，再执行：

- `V3P-Probe-VWSS-01_seed1`（最高优先，判断速度上下文是否帮助 WSS）
- `V3P-Probe-PWSS-01_seed1`（次优先，判断压力与 WSS 是否互相干扰）
- `V3P-Probe-VP-01_seed1`（算力允许时补，作为 field 任务双目标对照）
- `V3P-Anchor-01_seed1`
- `V3P-Base-01_seed1`
- `V3P-Main-01_seed1`

检查：

- `history.csv` 是否包含 V3 双域分项 loss（5 项 raw + 5 项 weighted）。
- WSS loss 与 field loss 数量级是否可控。
- 验证分数是否按 V3 口径（`val_score = 归一化 WSS RMSE + 归一化 wall p RMSE + 0.3 × 归一化 near_wall vel RMSE`）选择 checkpoint。

### 9.3 正式训练后处理

每个进入正式比较的 run 必须生成：

- `predictions_test/`
- wall WSS 指标（点级）
- wall pressure 指标
- near-wall velocity 指标
- 病例级统计表（`wss_clinical_metrics.json`，依 8.5）
- 分区域图（含 top-k overlap / Dice，依 8.5）

---

## 10. 当前结论

V3 的关键不是再换一个更复杂模型，而是把数据事实和监督口径改正确：

- V3 从 `V3P-Diag-00 + V3P-Probe-*` 开始，先定位单目标可学性和多任务冲突，再进入正式 `Anchor/Base/Main` 主线。
- 主干使用 PointNeXt-style local pooling（暂不引入完整层次化 PointNeXt）。
- 监督改为壁面/内部双域 mask；head 同步升级为 2 层 MLP decoupling buffer。
- WSS 从轻量权重起步并多档穷扫，避免梯度劫持。
- 几何先验继续作为核心创新点验证。
- 主指标从全局速度误差转为壁面 WSS、壁面压力和近壁速度上下文，且全部基于同采样、同 split 的 `V3P-Anchor-01` 锚点对照。

若单目标 probe 证明压力 / 速度 / WSS 至少具备可解释上限，且 `V3P-Main-01` 能在 WSS 与壁面压力上稳定优于无几何版本，并进一步超过同采样下的 `V3P-Anchor-01`，则 V3 可以成为任务A后续最干净的主线。

---

## 11. 前置工作清单（V3 P0 / P1 任务）

> 本节为 2026-04-29 修订新增。所有前置工作必须按 P0 → P1 顺序完成；P0 不到位时 `V3P-Main-01` 不得开跑。owner 字段是建议负责模块/脚本，便于后续派单。

### 11.0 向后兼容硬约束（所有 P0 改动的实现底线）

> 本节为 2026-04-29 二次澄清新增。V3 的 P0 代码改动**不得破坏 V1 / V2 的训练、predict、后处理能力**。所有改动必须按"开关式（feature flag）"实现：默认行为与今天完全一致，新行为仅在 V3 配置开启时生效。

| V3 改动 | 开关字段 | 默认值 | 旧 V1/V2 行为 |
| --- | --- | --- | --- |
| P0-A 运行时校验 | `optim.domain_loss.enabled` | `False` | 跳过 mask 等价校验，与今天完全一致 |
| P0-B `normalization_params_global.json` 校验 | `optim.domain_loss.normalize_by_target_std` | `False` | 不读 JSON、不校验，与今天完全一致；旧训练即便 JSON 缺失也不报错 |
| P0-C trainer 归一化 | `optim.domain_loss.normalize_by_target_std` | `False` | trainer 走原 `val_loss + early_stop_wss_weight × wss_loss`（`training/core/trainer.py:267-280` 现路径） |
| P0-D 新 loss | `optim.domain_loss.enabled` | `False` | `build_loss_plugin` 仍返回 `NullPhysicsLoss` / `PhysicsConstraintLoss` |
| **P0-E head 升级** | **`model.head_layout`** | **`"single_linear"`** | **`FieldPointNeXt.field_head` / `wss_head` 仍是单层 `nn.Linear`，与 `models.py:321-322` 完全一致** |
| P0-F 配置扩展 | dataclass 默认值 | 缺失字段用默认值 | 旧 JSON 加载零报错 |
| P0-G Diag-00 脚本 | 新脚本独立 | — | 不修改 `train_field.py` 主路径 |

#### 关键点：head 升级仅作用于 `FieldPointNeXt`（`pointnext`），且必须可配置

`training/core/models.py` 当前 6 个模型类（`FieldMLP / FieldGraphSAGE / FieldTransformer / FieldMeshGraphNet / FieldPointNetPP / FieldPointNeXt`）的 `field_head` / `wss_head` **全部**是单层 `nn.Linear`（`models.py:53/73/123/206/263/321`）。V3 P0-E 的实现必须满足：

1. **只升级 `FieldPointNeXt`**：其他 5 个模型保持单层不变；
2. **`model.head_layout` 配置开关**：默认 `"single_linear"`（与今天一致），V3 配置显式写 `"mlp2"` 才升级为 2 层 MLP；
3. **`load_state_dict` 不破坏旧 checkpoint**：在 `head_layout="single_linear"` 默认值下，参数名仍是 `field_head.weight` / `field_head.bias`，所有 V2P-WSSP-01~07 的 checkpoint 仍能被 `predict_field.py` / `load_checkpoint`（见 `training/core/io.py:17-21`）正常加载；
4. **新升级版本独立 state_dict 命名**：`head_layout="mlp2"` 启用时参数名变为 `field_head.0.weight / .0.bias / .2.weight / .2.bias`，与单层版本不冲突，便于将来同时存在两种 checkpoint。

#### 影响清单（受 V3 P0 改动影响的历史实验）

| 历史实验 | 用的模型 | 是否需要修改旧配置或重跑 |
| --- | --- | --- |
| V1 全套（A-Base-* / A-Main-01 / A-Opt-* / A-Abl-* / Line G / Line W / wss-multi 系列） | Transformer / GraphSAGE / MLP | **零影响**——不改这些模型 |
| V2P-Base-01 / V2P-Main-01（bootstrap） | PointNeXt | **零影响**（旧 JSON 不含 `model.head_layout` 字段，dataclass 默认值 `"single_linear"`） |
| V2P-WSSP-01~07 | PointNeXt | **零影响**（同上） |
| V3 全套 | PointNeXt | **新建专属 JSON**，显式写 `model.head_layout="mlp2"` 与 `optim.domain_loss.enabled=true` |

> **硬约束**：任何不通过开关字段而直接硬替换 trainer / loss / models 的 PR 一律退回；CI（如有）应包含"用 V2P-WSSP-05_seed1.json 跑 1 epoch + 用一个旧 checkpoint 跑 predict_field"两个回归用例。

### 11.1 P0 任务（V3P-Main-01 前必须完成）

| ID | 任务 | 依赖文件 | owner | 关联章节 |
| --- | --- | --- | --- | --- |
| P0-A | mask 等价关系运行时校验：训练/验证/测试加载图资产时断言 `boundary_core_ratio==(1.0,0.0)` 且 `allow_core_fallback==False`，并对采样后内部点验证 `max(dist_to_wall) < boundary_threshold`；不满足则 raise（**仅当 `optim.domain_loss.enabled=True` 时启用，旧配置默认走原路径**；不再新增 `is_near_wall` 字段，预处理无新产物） | `pipeline/preprocess.py` 或图加载层（`training/core/data.py`）；样本元信息读取 | 预处理 / 数据加载模块 | §2.1.1、§4.3、§9.1、§9.1.1、§11.0 |
| P0-B | **复用现有** `<data_root>/normalization_params_global.json`（`pipeline/normalize.py:443-465` 已经持久化）；不再新增 `processed/graphs/<split>/stats.json`；只在 `training/core/data.py` 加载图时验证该 JSON 与当前 split 的 `--train-split` 一致（无需写入新产物，仅校验） | `pipeline/normalize.py:443-465`（已存在）、`training/core/data.py`（新增校验） | 数据加载模块 | 8.1 (a) |
| P0-C | trainer 在 `optim.domain_loss.normalize_by_target_std=True` 时，从 `normalization_params_global.json` 读 per-channel std，对 `val_score` 各分项 RMSE 做归一化；同时把 `norm_std_*` 写入 `history.csv` 作为元信息列；旧路径（`val_loss + early_stop_wss_weight × wss_loss`）保留 | `training/core/trainer.py:267-280`、`training/core/losses.py` | trainer 模块 | 4.3、8.1 (b)、8.2、§11.0 |
| P0-D | 新增 `dual_domain_point_loss`，输出 5 项 raw + 5 项 weighted | `training/core/losses.py` | losses 模块 | 8.2 |
| P0-E | 模型 head 可配置升级（**仅 `FieldPointNeXt`**）：新增 `model.head_layout` 配置字段，默认 `"single_linear"` 保持当前单层 `nn.Linear`；V3 显式写 `"mlp2"` 时 `field_head` / `wss_head` 各自升级为 2 层 MLP（含非线性）。其他 5 个模型（MLP/GraphSAGE/Transformer/MeshGraphNet/PointNetPP）不动；旧 V2P-WSSP-* checkpoint 在默认值下仍可正常 `load_state_dict` | `training/core/models.py:288 FieldPointNeXt`、`training/core/models.py:321-322`、`training/core/config.py`（新字段） | models / config 模块 | 3.3、3.1.1、§11.0 |
| P0-F | 配置字段扩展（`optim.domain_loss.*` 系列） | `training/core/config.py` | config 模块 | 8.1 |
| P0-G | `V3P-Diag-00` 诊断脚本与产出（9.1.1 全部字段） | 新增 `training/scripts/run_v3_diag00.py` 或扩展 `train_field.py --diag-mode` | 训练脚本模块 | 9.1.1 |

兜底替代（若 P0-C 在某些环境读 JSON 有路径问题）：直接把 `<data_root>/normalization_params_global.json` 中相关 channel 的 `statistics[*].std` 静态复制到 `optim.domain_loss.norm_consts`，trainer 直接读配置（8.1 (c)）。**P0-B 不再有"产出新文件"型工作**——文件已存在 `<data_root>/normalization_params_global.json`，P0-B 现在只负责"读到、校验一致、写进 history.csv"。

### 11.2 P1 任务（V3P-Main-01 三 seed 提交论文表前必须完成）

| ID | 任务 | 依赖文件 | owner | 关联章节 |
| --- | --- | --- | --- | --- |
| P1-A | 新增 `eval_wss_clinical_metrics.py`：病例级 mean WSS / p95 WSS Pearson、Spearman；**内部必须做 inverse z-score**（从 `<data_root>/normalization_params_global.json` 读 `statistics["wss"]["mean"|"std"]` 等）后再算物理量纲下的统计量 | 新增 `training/scripts/eval_wss_clinical_metrics.py` | 评估脚本模块 | 7.0、7.1、8.5 |
| P1-B | 扩展 `_eval_wss_metrics_once.py`：高 WSS top-k overlap / Dice | `training/scripts/_eval_wss_metrics_once.py` | 评估脚本模块 | 7.1、8.5 |
| P1-C | 扩展 `plot_taskA_regional_bar.py --wss`：叠加 top-k overlap / Dice 柱图 | `training/scripts/plot_taskA_regional_bar.py` | 出图模块 | 7.1、8.5 |
| P1-D | `experiment_index.csv` 增加 `backbone_variant` / `sampling_profile` / `head_layout` / `norm_params_path` 列；为已有 V2P-WSSP runs 回填 | 仓库根 `outputs/field/experiment_index.csv` 维护脚本 | 后处理模块 | 8.3 |
| P1-E | 新增 V3 run 目录命名约定校验（防止漏 `_localpool` 后缀） | 训练脚本输出层 | 训练脚本模块 | 8.3 |
| P1-F | 新增 `training/scripts/check_train_test_ood.py`：对 V3 lock 资产（split_AG_v1 + wall13000+near2000）输出训练 vs 测试分位 diff + `NormRadius` max + BC 分位差异，作为论文 limitations 段附录数据来源 | 新增脚本 | 评估脚本模块 | 7.0、9.1.1、§12 |
| P1-G | `predict_field.py` 写入 `predictions_test/manifest.json` 时附加 `"norm_params_path"` 字段，引用当时使用的 `normalization_params_global.json`，便于事后反归一化复盘 | `training/scripts/predict_field.py` | 后处理模块 | 7.0、8.5 |

---

## 12. 局限与未来工作

> 本节为 2026-04-29 修订新增，作为论文 limitations 段的草稿。

- **多任务平衡方法**：V3 第一阶段使用人工权重 + 多档穷扫，未引入 GradNorm / Uncertainty Weighting / PCGrad 等现代多任务平衡方法。审稿人若提出此问题，回复策略：V3 第一阶段先建立"双域 mask + decoupling buffer + 同采样锚点"的干净对照；若 `V3P-WSS-01-*` 三档全部 No-Go（`wall.r2_wss` 未达到 `V3P-Anchor-01` 三 seed 均值），再开 GradNorm / UW 对照实验，独立列为 V3 第二阶段的 `V3P-MTL-*` 系列。
- **完整层次化 PointNeXt**：V3 第一阶段仅使用 local pooling backbone；FPS / radius grouping / set abstraction 留给 `V3P-Arch-*`，与 `V3P-Main-01` 使用同数据、同 split、同 loss、同评估口径做唯一变化项对照。
- **点云序列建模（如 Mamba）**：跨时间步的非结构化点云序列建模不在 V3 第一阶段范围内；如果未来要做，建议命名为 `V3T-*` 路线，与本路线分开归档。
- **物理残差 (PINN)**：V3 不引入 PDE 残差。如未来要做高低保真混合训练或 Navier-Stokes 残差，须新建 `V3PINN-*` 路线，独立从数据分布与正则强度开始论证，不与 V3 第一阶段混表。
- **数据采样口径**：V3 锁定 wall13000+near2000；如果未来病例池扩大或导出协议变化，需要重新跑 `V3P-Diag-00` 并复核 4.1.1 / 4.2.1 / 5.3 的所有阈值。
- **WSS 4 维冗余**：V3 第一阶段保留 4 维输出便于与 V2P-WSSP 对齐；3.3 末段已规定一致性诊断与条件实验 `V3P-WSS-04`，但若一致性误差很小，本路线第二阶段会切换到 3 分量 + magnitude 推导，并在论文方法节明确该决策依据。
- **坐标系归一化的尺度近似**：`pipeline/coord_normalize.py` 是**逐病例 PCA + scale**（每个病例的 `scale_factor` 不同），但 V3 / V1 / V2 的 physics loss（如未来引入连续性 / 动量残差时）需要把"对归一化输入求导"的结果映射回物理尺度，目前在 `training/core/config.py:144-157` 中取**所有训练病例 `scale_factor` 的中位数**作为统一坐标尺度，这是一个近似（病例间 scale 差异 ≤ ~30%）。V3 第一阶段不引入 physics loss，因此该近似不影响主结果；但若 V3 第二阶段或后续 `V3PINN-*` 路线引入 PDE 残差，必须升级为"逐病例 scale 字典 + per-sample 反归一化"，并在论文方法节写明。本期论文 limitations 段需说明此点。
- **训练集 z-score 在测试集上的 OOD 风险**：`pipeline/normalize.py` 的 z-score 与 min-max 都仅用训练集统计；测试集若在某个 channel（如 `NormRadius`、`Curvature` 长尾）上出现超过训练集 max 或远大于训练 std 的样本，输入分布会偏离训练分布。V3P-Diag-00 必须输出"测试集各 channel 在训练集 z-score 后的 99/99.5 分位 vs 训练集 99/99.5 分位"，作为 OOD 警示（详见 §9.1.1）；如果发现严重 OOD（>2σ 显著偏移），需在论文 limitations 中说明，并考虑在第二阶段做"测试时坐标 / 半径 clamp"或"按病例分组 evaluate"。
- **指标量纲与论文叙事**：训练 / 验证 / 测试期所有 RMSE 在 z-score 空间报告，与 V1 / V2 历史主表口径一致；临床叙事（病例级 mean WSS / p95 WSS）必须反归一化为 Pa（详见 §7.0 与 §8.5）。该口径切换是论文方法节必写的注解，避免审稿人质疑"RMSE_p=0.61 单位是什么"。
