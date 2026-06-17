# 任务A V2修正路线实验矩阵

> 上位文档：[实验设计总纲](../../../实验设计总纲.md) | [任务A实验清单](../01-V1路线/任务A_V1实验清单.md) | [任务A实验状态表](../03-共享执行与状态/任务A实验状态表.md) | [任务A V2准备执行清单](任务A_V2准备执行清单.md) | [任务A V2首轮判定与汇报模板](任务A_V2首轮判定与汇报模板.md) | [项目缺陷分析与修正路径](../../../02-推进与变更/项目缺陷分析与修正路径.md)

## 1. 文档定位

本文件只服务于 **`Route-PhysicsAware-V2`**。

它的目的不是覆盖已有的 `A-Base-* / A-Opt-*` 工作，而是把**新增修正路线**写成一份可执行、可命名、可追踪的实验矩阵。
原有 `kNN + GNN` 工作统一记为 **`Route-KNN-GNN-V1`**，继续保留：

- 历史基线
- 汇报对照
- 方法论缺陷复盘素材

V2 的问题不是“如何继续在旧底座上调参”，而是：

1. `.cas/.msh` 网格拓扑能否支撑 **mesh-aware GNN**
2. 在相同数据口径下，**mesh-aware GNN** 和 **geometry-aware point-cloud** 谁更适合做后续论文主线
3. 谁更能把 `near_wall / WSS / TAWSS` 做准

---

## 2. V1 与 V2 的关系

### 2.1 必须保留的旧结果

下列实验不删除、不改名，继续作为 **V1 历史结果**：

- `A-Base-01`
- `A-Base-02`
- `A-Base-03`
- `A-Main-01`
- `A-Opt-01`
- `A-Opt-02`
- `A-Opt-03`
- `A-Opt-05`

它们在 V2 中的角色：

- `A-Base-01`：旧数据口径下的点级下限
- `A-Main-01`：旧路线的几何增益锚点
- `A-Opt-03 / A-Opt-05`：旧路线中较强的单尺度 GNN 参考

### 2.2 暂停继续扩展的旧线

在 V2 首轮对照完成前，下面这些**不应继续开新实验**：

- `A-Opt-06+`
- `A-Opt-G*`（**注（2026-04-14）**：V1 上 **`G01` / `G04` / `G05` 三 seed** 已在该约束提出前归档；**自本注起**新开 `G*` 仍建议等 V2 Gate-0 或团队明确放行）
- `A-Opt-W*`
- 旧 `kNN` 路线上的新物理约束消融

原因很简单：V2 的目的就是先决定**哪一类主干和哪一套空间表示**值得继续投资源。

---

## 3. 命名规则

### 3.1 路线级命名

- `Route-KNN-GNN-V1`：旧路线
- `Route-PhysicsAware-V2`：新增总路线
- `Route-MeshGNN-V2`：V2 下的 mesh-aware GNN 子路线
- `Route-PointCloud-V2`：V2 下的 point-cloud 子路线

### 3.2 实验级命名

V2 建议只用下面三类前缀：

- `V2-Ref-*`：V2 数据口径下的公共参考实验
- `V2G-*`：V2 mesh-aware GNN 实验
- `V2P-*`：V2 point-cloud 实验

### 3.3 推荐命名模板

| 类型 | 示例 | 含义 |
| --- | --- | --- |
| 公共参考 | `V2-Ref-Base-01` | V2 数据口径下的共享 MLP/点级参考 |
| MeshGNN 基线 | `V2G-Base-01` | 无 geometry 的 mesh-aware GNN |
| MeshGNN 主模型 | `V2G-Main-01` | 有 geometry 的 mesh-aware GNN |
| PointCloud 基线 | `V2P-Base-01` | 无 geometry 的 point-cloud 主干 |
| PointCloud 主模型 | `V2P-Main-01` | 有 geometry 的 point-cloud 主干 |
| 结构扩展 | `V2G-Opt-01` / `V2P-Opt-01` | 只在首轮胜出后开放 |

### 3.4 目录命名建议

建议配置与输出目录同步区分 V2：

- 配置目录：
  - `training/configs/field/generated/v2_ref/`
  - `training/configs/field/generated/v2_meshgnn/`
  - `training/configs/field/generated/v2_pointcloud/`
- 输出目录中的 `experiment_name` 建议显式带上 `v2`

例如：

- `field_v2_meshgnn_base01_coord_t_bc_wall_mesh_split_AG_v2_seed1`
- `field_v2_meshgnn_main01_coord_t_bc_geom_wall_mesh_split_AG_v2_seed1`
- `field_v2_pointcloud_main01_coord_t_bc_geom_wall_split_AG_v2_seed1`

---

## 4. V2 的硬门槛

V2 不是“想跑什么就跑什么”，必须按门槛推进。

### Gate-0：数据与拓扑准备

必须先确认：

1. `.cas/.msh` 能稳定导出节点、单元和壁面区域
2. 采样点能回溯到原始网格或至少能建立局部可信邻域
3. WSS 真值或 WSS 对齐方式可确定

若 Gate-0 失败：

- 不允许直接宣称 `Route-MeshGNN-V2` 为默认主线
- 但 `Route-PointCloud-V2` 仍可继续

### Gate-1：首轮结构对照

V2 首轮只回答两个问题：

1. 正确空间表示是否明显优于旧 `kNN` 路线
2. MeshGNN 和 PointCloud 谁更值得继续

### Gate-2：WSS 链路门槛

后续是否进入 V2 的优化和风险预测，按下面顺序判断：

1. `WSS / TAWSS`
2. `near_wall`
3. `interior`
4. 全局 `u,v,w,p`
5. 效率

---

## 5. V2 首轮实验矩阵

### 5.1 阶段 0：准备与 smoke test

执行细节、输出文件名和通过标准见：[任务A V2准备执行清单](任务A_V2准备执行清单.md)。

| Exp ID | 类型 | 目的 | 是否训练 | 优先级 |
| --- | --- | --- | --- | --- |
| `V2-Prep-01` | mesh 导出检查 | 验证 `.cas/.msh` 能否导出节点、单元、壁面区域 | 否 | 最高 |
| `V2-Prep-02` | 采样映射检查 | 验证采样点与原始 mesh 的映射/邻接可回溯 | 否 | 最高 |
| `V2-Prep-03` | WSS 对齐检查 | 确认预测节点与 CFD WSS 真值的对应关系 | 否 | 最高 |
| `V2-Prep-04` | Data smoke | 2~3 个病例跑通 V2 数据生成与评估链 | 否 | 最高 |

### 5.2 阶段 1：首轮可训练对照

这一轮**严格控制数量**，只跑最小闭环。

| Exp ID | 主干 | 输入特征 | 研究问题 | seed 计划 |
| --- | --- | --- | --- | --- |
| `V2-Ref-Base-01` | Point-wise MLP | `coords + t + BC` | V2 数据口径下的统一下限 | `1 -> [1,2,3]` |
| `V2G-Base-01` | mesh-aware GNN | `coords + t + BC + is_wall` | 正确 mesh 邻域本身是否有价值 | `1 -> [1,2,3]` |
| `V2G-Main-01` | mesh-aware GNN | `coords + t + BC + geometry + is_wall` | geometry 在 MeshGNN 上是否成立 | `1 -> [1,2,3]` |
| `V2P-Base-01` | PointNeXt（默认）/ PointNet++（轻量对照） | `coords + t + BC + is_wall` | 点云主干在 V2 数据口径下的基线能力 | `1 -> [1,2,3]` |
| `V2P-Main-01` | 与 `V2P-Base-01` 同主干 | `coords + t + BC + geometry + is_wall` | geometry 在 PointCloud 上是否成立 | `1 -> [1,2,3]` |

#### 5.2.1 Bootstrap 扩展实验（**不计入**首轮 5 组）

> 用于在 Gate-0 / 正式 `split_AG_v2` 之前探索 **壁面血流动力学 / WSS** 与 **点云主干** 的耦合；归因时须与上表 5 组区分。

| Exp ID | 主干 | 监督 / 特征要点 | 状态 |
| --- | --- | --- | --- |
| `V2P-WSSP-01` | PointNeXt | **仅 p + WSS**（`target_weights` u/v/w=0，`wss_loss_weight=1`）；壁面 13000 + 近壁 2000 点采样；全几何 + `is_wall` | seed=1 已完成（2026-04-24，`split_AG_v1` bootstrap）；见 [任务A实验状态表](../03-共享执行与状态/任务A实验状态表.md)「V2P-WSSP-01」 |
| `V2P-WSSP-02` | PointNeXt | **全场 u/v/w/p + WSS 头**（`target_weights=[2,2,2,0.5]`，`wss_loss_weight=0.5`，`early_stop_wss_weight=1.0`）；**标准采样**；全几何 + `is_wall` | seed=1 已完成（2026-04-25）；训练+预测+图已归档；见 [任务A实验状态表](../03-共享执行与状态/任务A实验状态表.md)「V2P-WSSP-02」 |
| `V2P-WSSP-03` | PointNeXt | **全场 + 轻量 WSS 辅助头**（`target_weights=[2,2,2,0.5]`，`wss_loss_weight=0.01`，`early_stop_wss_weight=0`）；用于修复 WSSP-02 梯度失衡 | seed=1 已完成（2026-04-26 后处理归档）；**`r2_p` 相对 WSSP-02 恢复**；**WSS R² 仍略负**；见 [任务A实验状态表](../03-共享执行与状态/任务A实验状态表.md)「V2P-WSSP-03」 |
| `V2P-WSSP-04` | PointNeXt | **压力 + WSS 主线，速度弱辅助**（`target_weights=[0.1,0.1,0.1,1.0]`，`wss_loss_weight=0.5`，`early_stop_wss_weight=0`）；建议配合 wall-rich + near-wall 图资产 | seed=1 已完成；**2026-04-27** 集群重训 run **`…/20260427_103849/`** 已 **`predict_field`+分区域+与 Line W 三实验多模型**；`regional_eval` 的 **`r2_p`** 与 V1 **Transformer+WSS 多任务** **不在同档**（**勿无注释与 `A-Opt-W03-*` 混表**）；总览 **`outputs/field/plots/wssp04_w03_line_seed1_20260427/`**；**03 vs 04** 对照仍见 `outputs/field/plots/v2p_wssp03_vs_04_seed1/`；见 [任务A实验状态表](../03-共享执行与状态/任务A实验状态表.md)「V2P-WSSP-04」与「**Line W：A-Opt-W03 权重复扫**」 |
| `V2P-WSSP-05` | PointNeXt | 复刻 **WSSP-01**：**仅 p+WSS**，**MSE WSS**；wall13000+near2000 | **（2026-04-28）** seed **1～3** 已训练 + 主后处理；三 seed **`wss_r2_wss`** 仍略负、**`r2_p`** 方差大；详见 [任务A实验状态表](../03-共享执行与状态/任务A实验状态表.md)「V2P-WSSP-05」 |
| `V2P-WSSP-06` | PointNeXt | 同 **05**，唯一变更 **`wss_loss_type=huber`**（Smooth L1） | **（2026-04-28）** seed **1～3** 训练完成；seed **1～2** 已有 **`predictions_test`**，**seed3 尚缺导出**；**Huber 未稳定优于 MSE**；详见 [任务A实验状态表](../03-共享执行与状态/任务A实验状态表.md)「V2P-WSSP-06」 |

#### 5.2.2 V2P-WSSP 后续主指标口径（2026-04-25）

与导师沟通后，若速度场相关系数难以继续提高，V2P-WSSP 不再以完整速度场重建为主门槛，而转为服务后续血流动力学分析的 **压力场 + 壁面 WSS 快速预测**路线：

- **主指标**：`r2_p`、`rmse_p`、`wall.r2_wss`、`wall.rmse_wss`。
- **诊断指标**：`near_wall.r2_vel_mag`、`near_wall.rmse_vel_mag`、全局 `r2_vel_mag`；这些指标用于解释近壁上下文是否有帮助，不作为本线成败判据。
- **训练原则**：壁面点与 WSS 监督优先；近壁内部速度可以给小权重辅助，核心内部速度不作为优先优化对象。

### 5.3 首轮的最小结论要求

这 5 组跑完后，必须回答下面 4 个问题：

1. V2 数据口径下，**物理可信空间表示**是否明显优于 V1
2. geometry 的增益是否在 **MeshGNN** 和 **PointCloud** 上都成立
3. `V2G-Main-01` 和 `V2P-Main-01` 谁在 `near_wall / WSS / TAWSS` 上更强
4. 谁更稳、更值得继续做第二轮优化

---

### 5.4 当前推荐执行路线（2026-04-18 新增）

本节专门回答一个现实问题：在旧 `kNN + GNN` 主线收益已经接近平台后，下一步到底是继续扩展旧线、直接转 PointCloud，还是优先押注真实拓扑的 MeshGNN。

当前建议采用一条 **稳定、可执行、便于归因** 的路线：

- **停止继续扩展旧 `Route-KNN-GNN-V1` 主线**
- **先完成 V2 Gate-0**
- **以 PointCloud 作为首个可训练主线**
- **将 MeshGNN 作为 Gate-0 通过后的受控对照线**
- **Mamba 类结构不作为首轮立项起点，只作为第二阶段可选扩展**

#### 5.4.1 路线总判断

当前最不合理的顺序是：

1. 继续在 `A-Opt-05` 上叠更多旧线优化；
2. 同时再开 PointNet++、MeshGNN、Mamba 多条新线；
3. 最后再回头解释为什么有效。

当前更合理的顺序是：

1. 先停止继续扩展旧 V1 训练线；
2. 完成 **Gate-0**，把 V2 数据与空间表示基础做硬；
3. 先跑 **PointCloud 首轮最小闭环**；
4. 只在真实拓扑、采样映射和 WSS 对齐都稳定后，再让 **MeshGNN** 进入首轮对照；
5. 只有首轮胜出的路线，才允许进入第二阶段优化与结构扩展。

#### 5.4.2 为什么当前优先 PointCloud，而不是先押 MeshGNN

原因不是 PointCloud 在理论上一定优于 MeshGNN，而是它在当前项目条件下更稳、更容易形成闭环。

1. **当前最大不确定性在空间表示，而不是 backbone 名字。**
   旧 V1 的主要问题已经明确是 `kNN` 图的物理语义不足，因此不应继续在旧图底座上叠新层和新 loss。

2. **PointCloud 对“图边是否完全正确”的依赖更低。**
   PointNet++ 中的 `kNN/radius` 更接近局部邻域查询，而不是把边本身当作方法学核心。

3. **PointCloud 天然适合多尺度局部聚合。**
   这更接近当前“内部复杂流场与 near-wall 结构表达不足”的实际瓶颈。

4. **显式 geometry 可直接迁移。**
   当前已有证据表明显式几何先验有效，因此 V2 的关键不是放弃 geometry，而是验证 geometry 在 PointCloud 主干上是否仍有稳定增益。

5. **MeshGNN 的学术上限可能更高，但前置依赖更重。**
   只有 `.cas/.msh` 拓扑、采样映射、壁面/WSS 对齐全部稳定后，MeshGNN 才值得正式投入对照。

#### 5.4.3 第一阶段的唯一目标

第一阶段不追求“直接冲最终论文结构”，只回答下面 3 个问题：

1. 在 V2 数据口径下，**PointCloud 主干能否稳定跑通**；
2. 显式 geometry 在 PointCloud 主干上是否仍有稳定增益；
3. 在 `near_wall / WSS / TAWSS` 上，PointCloud 是否已经足以替代旧 V1 主线，成为后续优化入口。

只要以上 3 个问题里有 2 个得到正面答案，就允许把 V2 PointCloud 提升为当前主线。

#### 5.4.4 详细执行顺序

建议严格按下面 4 个阶段推进。

##### 阶段 A：停止扩展旧线，收口边界

本阶段不新增任何 V1 训练实验，只做策略收口。

- 不再新增 `A-Opt-06+`
- 不再新增新的 `A-Opt-G*`
- 不再新增新的 `A-Opt-W*`
- 保留 `A-Opt-05`、`A-Opt-05-wss-multi`、`A-Main-01` 作为历史对照锚点

本阶段完成标准：

- 文档中明确 V1 为历史对照线
- 后续新增训练实验统一进入 `V2-*`

##### 阶段 B：完成 Gate-0，把数据基础做硬

本阶段完全按 [任务A V2准备执行清单](任务A_V2准备执行清单.md) 推进，不允许跳过。

必做项：

1. `V2-Prep-01`：mesh 导出检查
2. `V2-Prep-02`：采样映射检查
3. `V2-Prep-03`：WSS 对齐检查
4. `V2-Prep-04`：2~3 病例 smoke 闭环

最低通过条件：

- 至少 3 个病例完成 mesh 导出
- 采样点到 mesh 的 `mapped_ratio >= 0.99`
- WSS 真值与壁面节点/面元对齐关系明确
- smoke 数据链路可落盘并可复查

若阶段 B 未通过：

- 不启动 `V2G-*`
- 允许继续 `V2P-*`
- 不再争论“MeshGNN 是否更优”，因为真实拓扑前提尚未成立

##### 阶段 C：PointCloud 首轮最小闭环

> **（2026-04-22）阶段 C 执行状态**：`V2P-Base-01`（无 geometry）与 `V2P-Main-01`（有 geometry）seed=1 **已在 `split_AG_v1`（bootstrap 口径）完成训练**。Geometry 增益显著（`rmse_vel_mag` -22%，`r2_vel_mag` +72%）；`V2P-Main-01` 全局 `r2_vel_mag=0.609` 已超过 V1 锚点 `A-Opt-05`（~0.576）。**当前限制**：仅 seed=1、缺分区域评估、使用 bootstrap split，不作为正式结论。Gate-0 通过后在 `split_AG_v2` 上补跑正式版。
> **（2026-04-24～04-25）** 同阶段已增跑 **`V2P-WSSP-01`**（**p + WSS 专线**）与 **`V2P-WSSP-02`**（**全场 + WSS 头 + 混合早停**），均 **不计首轮 5 组**：见 **§5.2.1** 与 [任务A实验状态表](../03-共享执行与状态/任务A实验状态表.md)。

本阶段是当前推荐主线。

建议只跑 2 组核心实验，再按结果决定是否补 seed：

1. `V2P-Base-01`
2. `V2P-Main-01`

建议输入定义：

- `V2P-Base-01`：`coords + t + BC + is_wall`
- `V2P-Main-01`：`coords + t + BC + geometry + is_wall`

建议 backbone：

- 第一选择：**PointNeXt**
- 第二选择：**PointNet++**（更轻的仓库内对照）
- 暂不建议第一轮直接上 Mamba

原因：

- PointNeXt 更适合作为仓库内首发主干：保留 PointNet++ 的层级局部聚合逻辑，同时引入更稳的残差和扩展策略；
- PointNet++ 仍保留为更轻的对照骨架，便于区分“点云路线本身有效”与“更强点云骨架带来的额外收益”；
- Mamba 更适合作为第二轮长程依赖增强，而不是第一轮主干立项；
- 先用同一套点云骨架做出清晰的 “without geometry / with geometry” 对照，归因最干净。

建议训练顺序：

1. 先跑 `seed=1`
2. 只有当 `seed=1` 在验证集和测试集都优于 V1 历史对照时，再补 `seed=2,3`
3. 若 `seed=1` 已明显差于 `A-Opt-05`，立即停线排查，不补 seed

本阶段主看指标：

1. `near_wall.rmse_vel_mag`
2. `interior.rmse_vel_mag`
3. `summary.test_metrics.r2_vel_mag`
4. `summary.test_metrics.r2_p`
5. 壁面 WSS RMSE / R²

本阶段通过标准建议：

- `V2P-Main-01` 至少在以下任一方面优于 `A-Opt-05`：
  - `near_wall.rmse_vel_mag`
  - `wss_r2_wss`
  - `r2_vel_mag`
- 且 geometry 相对 `V2P-Base-01` 呈现稳定正增益

只要满足以上条件，即可把 `V2P-Main-01` 提升为当前主推荐主线。

##### 阶段 D：MeshGNN 仅作受控对照，不作抢跑主线

只有阶段 B 通过后，才允许启动：

1. `V2G-Base-01`
2. `V2G-Main-01`

本阶段目标不是“一定取代 PointCloud”，而是回答：

- 真实拓扑是否真的给 `near_wall / WSS` 带来额外收益；
- 这种收益是否足以覆盖更高的数据和工程复杂度。

建议执行规则：

1. 同样先跑 `seed=1`
2. 若 `V2G-Main-01` 在 `near_wall / WSS` 上没有明显超过 `V2P-Main-01`，不建议继续补多 seed
3. 只有当 MeshGNN 在核心指标上明确更优，才允许成为第二阶段主线

#### 5.4.5 第二阶段开放条件

只有首轮主线确定后，才允许做结构扩展。

若 PointCloud 胜出，第二阶段开放顺序建议为：

1. `V2P-Opt-01`：速度权重重加权
2. `V2P-Opt-02`：局部层数/宽度小步扩展
3. `V2P-Abl-01`：去掉全部 geometry
4. `V2P-Abl-02-*`：几何分量消融
5. `V2P-Opt-03`：可选的长程建模扩展（如 Mamba），但必须排在 PointNet++ 主线稳定之后

若 MeshGNN 胜出，第二阶段开放顺序建议为：

1. `V2G-Opt-01`：速度权重重加权
2. `V2G-Opt-02`：更稳的残差块 / 归一化
3. `V2G-Abl-01`：去掉全部 geometry
4. `V2G-Abl-02-*`：几何分量消融

两条路线共同限制：

- 第二阶段之前不引入 physics loss
- 第二阶段之前不做多 backbone 大乱斗
- 第二阶段之前不把 Mamba 当成默认主解

#### 5.4.6 一页式执行摘要

若只保留最短版本，当前推荐路线就是：

1. 停止继续扩展旧 V1 `kNN + GNN` 主线；
2. 完成 V2 Gate-0；
3. 先用 **PointNeXt** 跑 `V2P-Base-01 / V2P-Main-01`，必要时再用 **PointNet++** 做轻量对照；
4. 用 `with geometry` vs `without geometry` 判断显式几何先验是否跨主干成立；
5. 只有真实拓扑准备完全稳定后，再用 `V2G-Base-01 / V2G-Main-01` 做受控对照；
6. 谁在 `near_wall / WSS / R²_|v|` 上更稳，谁进入第二阶段；
7. Mamba 类结构只作为第二阶段扩展，不作为当前立项起点。

---

## 6. V2 第二轮实验开放条件

只有首轮胜出的路线，才允许进入第二轮。

### 6.1 若 `Route-MeshGNN-V2` 胜出

开放：

- `V2G-Opt-01`：速度权重重加权
- `V2G-Opt-02`：PreNorm / 更稳残差块
- `V2G-Abl-01`：去掉全部 geometry
- `V2G-Abl-02-*`：几何分量消融

暂不开放：

- `V2G-Opt-WSS-*`
- `V2G-Physics-*`
- `V2G-Time-*`

### 6.2 若 `Route-PointCloud-V2` 胜出

开放：

- `V2P-Opt-01`：速度权重重加权
- `V2P-Opt-02`：局部层数/宽度小步扩展
- `V2P-Abl-01`：去掉全部 geometry
- `V2P-Abl-02-*`：几何分量消融

暂不开放：

- `V2P-Physics-*`
- `V2P-Time-*`
- 多个 point-cloud backbone 横向大乱斗

### 6.3 若两条路线都不够硬

止损判断：

- 不继续追加复杂结构
- 先把论文主线收缩为高精度 `u,v,w,p`
- 将 `WSS/TAWSS` 写成部分成立或局限性

---

## 7. V2 的主表与汇报方式

执行口径、胜出规则和首轮汇报骨架见：[任务A V2首轮判定与汇报模板](任务A_V2首轮判定与汇报模板.md)。

### 7.1 主表分三层

建议正式汇报时固定三张表：

1. **表 1：V1 历史基线表**
   - `A-Base-01`
   - `A-Base-02`
   - `A-Base-03`
   - `A-Main-01`

2. **表 2：V2 首轮路线对照表**
   - `V2-Ref-Base-01`
   - `V2G-Base-01`
   - `V2G-Main-01`
   - `V2P-Base-01`
   - `V2P-Main-01`

3. **表 3：V2 胜出路线优化表**
   - 只放胜出路线的 `Opt/Abl`

### 7.2 主表排序建议

主文排序建议：

1. `near_wall RMSE_|v|`
2. `WSS R²`
3. `TAWSS Pearson`
4. `interior RMSE_|v|`
5. `RMSE_p`
6. 推理时间
7. 显存

不要再把“全局 `RMSE_|v|` 最小”默认当作第一排序标准。

---

## 8. 旧实验的保留引用规则

### 可以继续在正文引用的 V1 结果

- `A-Base-01 ~ A-Main-01`
  - 用来讲：旧路线下 geometry 确有增益
- `A-Opt-03 / A-Opt-05`
  - 用来讲：旧路线内部通过训练技巧只能得到有限改进

### 只能作为附录或内部记录的 V1 结果

- `A-Opt-05t_*`
- `A-Opt-06`
- `A-Opt-G*`
- `A-Opt-W*`

原因：这些都是建立在旧路线底座上的延展，V2 还没决定谁是主线前，不宜继续扩展成正文核心。

---

## 9. 当前建议的执行顺序

### 本周最优先

1. `V2-Prep-01`
2. `V2-Prep-02`
3. `V2-Prep-03`
4. `V2-Prep-04`

### Gate-0 通过后立即启动

1. `V2-Ref-Base-01`
2. `V2G-Base-01`
3. `V2G-Main-01`
4. `V2P-Base-01`
5. `V2P-Main-01`

### 明确禁止

在上述 5 组没有完成前，禁止：

- 新开 `A-Opt-G*`
- 新开 `A-Opt-W*`
- 新开 `A-Opt-06+`
- 新开 V2 的 physics 或时序实验

---

## 10. 一句话结论

V2 的首要目标不是“再造一个更复杂模型”，而是**用最小且干净的对照，选出真正值得写进论文主线的那条路线**。
