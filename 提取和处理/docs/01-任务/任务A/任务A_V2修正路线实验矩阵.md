# 任务A V2修正路线实验矩阵

> 上位文档：[实验设计总纲](../../实验设计总纲.md) | [任务A实验清单](任务A实验清单.md) | [任务A实验状态表](任务A实验状态表.md) | [任务A V2准备执行清单](任务A_V2准备执行清单.md) | [任务A V2首轮判定与汇报模板](任务A_V2首轮判定与汇报模板.md) | [项目缺陷分析与修正路径](../../02-推进与变更/项目缺陷分析与修正路径.md)

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
| `V2P-Base-01` | PointNet++/Point Transformer 二选一 | `coords + t + BC + is_wall` | 点云主干在 V2 数据口径下的基线能力 | `1 -> [1,2,3]` |
| `V2P-Main-01` | 与 `V2P-Base-01` 同主干 | `coords + t + BC + geometry + is_wall` | geometry 在 PointCloud 上是否成立 | `1 -> [1,2,3]` |

### 5.3 首轮的最小结论要求

这 5 组跑完后，必须回答下面 4 个问题：

1. V2 数据口径下，**物理可信空间表示**是否明显优于 V1
2. geometry 的增益是否在 **MeshGNN** 和 **PointCloud** 上都成立
3. `V2G-Main-01` 和 `V2P-Main-01` 谁在 `near_wall / WSS / TAWSS` 上更强
4. 谁更稳、更值得继续做第二轮优化

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
