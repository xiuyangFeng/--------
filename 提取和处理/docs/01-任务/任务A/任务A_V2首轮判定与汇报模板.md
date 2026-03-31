# 任务A V2首轮判定与汇报模板

> 上位文档：[任务A V2修正路线实验矩阵](任务A_V2修正路线实验矩阵.md) | [任务A V2准备执行清单](任务A_V2准备执行清单.md) | [实验设计总纲](../../实验设计总纲.md)

## 1. 文档定位

本文件只服务于 **`Route-PhysicsAware-V2`** 的 **首轮可训练对照**。

它解决的不是“还要不要再多跑几组”，而是下面三件事：

1. `V2-Prep-01 ~ 04` 通过后，首轮 5 组实验应该如何统一判定
2. 哪些结果必须进入主表，哪些只能作为补充说明
3. 什么时候该继续，什么时候必须止损

如果没有这份判定口径，首轮对照很容易在结果出来后临时改标准，最后谁都说不清。

---

## 2. 首轮唯一允许回答的问题

V2 首轮不是为了证明全部故事，只允许回答下面 4 个问题：

1. **V2 的物理可信空间表示**是否明显优于 V1 的旧 `kNN` 路线
2. **显式几何先验**是否在 V2 口径下仍然成立
3. `Route-MeshGNN-V2` 和 `Route-PointCloud-V2` 谁更值得继续
4. 胜出路线是否已经具备进入第二轮优化的资格

首轮**不允许**回答下面这些问题：

- 风险预测是否成立
- OSI / RRT 是否已经可靠
- 复杂物理约束是否有增益
- 多个 point-cloud backbone 谁更强

这些问题一旦提前混进首轮，会直接毁掉归因。

---

## 3. 首轮实验集合

首轮只认下面 5 组可训练实验：

- `V2-Ref-Base-01`
- `V2G-Base-01`
- `V2G-Main-01`
- `V2P-Base-01`
- `V2P-Main-01`

`V2-Prep-01 ~ 04` 不属于训练实验，但属于首轮判定前置条件。

如果这 5 组之外又混入额外结构、额外损失、额外采样策略，首轮结论自动失真。

---

## 4. 首轮判定优先级

### 4.1 统一排序口径

所有路线比较统一按下面顺序排序：

1. `WSS / TAWSS`
2. `near_wall`
3. `interior`
4. 全局 `u, v, w, p`
5. 训练/推理效率

这不是建议，是固定排序。

### 4.2 不允许的偷换

不允许出现下面这些“结果解释技巧”：

- `global RMSE` 好看，就回避 `near_wall / WSS`
- `near_wall` 好看，但 `WSS / TAWSS` 不稳定，却宣称 surrogate 成立
- 只挑单个病例最漂亮的图，不报告病例级汇总
- 只汇报相关性，不汇报误差
- 只汇报均值，不汇报 seed 波动

---

## 5. 首轮必须产出的核心结果

### 5.1 主表 1：路线级总表

主表 1 必须至少包含下面字段：

| Exp ID | Backbone | Geometry | WSS Corr | TAWSS Corr | near_wall Vel RMSE | interior Vel RMSE | Pressure RMSE | Params | Train Time/Epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

说明：

- `Backbone`：`MLP / MeshGNN / PointCloud`
- `Geometry`：`No / Yes`
- `WSS Corr`：优先病例级均值
- `TAWSS Corr`：优先病例级均值
- `near_wall Vel RMSE`：必须单独列
- `interior Vel RMSE`：必须单独列
- `Pressure RMSE`：保留，但不能排在前面

### 5.2 主表 2：几何先验增益表

主表 2 只回答一个问题：**geometry 是否真的有增益**。

| Pair | Metric | Base | Main | Delta | 结论 |
| --- | --- | --- | --- | --- | --- |
| `V2G-Base-01 vs V2G-Main-01` | `TAWSS Corr` |  |  |  |  |
| `V2G-Base-01 vs V2G-Main-01` | `near_wall Vel RMSE` |  |  |  |  |
| `V2P-Base-01 vs V2P-Main-01` | `TAWSS Corr` |  |  |  |  |
| `V2P-Base-01 vs V2P-Main-01` | `near_wall Vel RMSE` |  |  |  |  |

要求：

- 同一主干只和自己的 `Base/Main` 比
- 不要跨主干直接用 `Base` 去对 `Main`
- `Delta` 要写清楚方向，不能只写绝对值

### 5.3 必须有的图

首轮至少要产出下面 4 类图：

1. **病例级 TAWSS 散点图**
   - 横轴：CFD 真值
   - 纵轴：模型预测
   - 至少画 `V2G-Main-01` 与 `V2P-Main-01`

2. **区域误差条形图**
   - `wall / near_wall / interior`
   - 至少画速度模长与压力

3. **代表病例切片或表面热图**
   - 重点看分叉和高曲率区域
   - 不要只放最好看的病例

4. **seed 稳定性图**
   - `mean ± std`
   - 至少覆盖 `TAWSS Corr` 和 `near_wall Vel RMSE`

---

## 6. 首轮胜出与止损规则

### 6.1 路线胜出规则

只有同时满足下面两条，某条路线才算真正胜出：

1. 其 `Main` 在 `WSS / TAWSS` 上优于另一条路线的 `Main`
2. 这种领先不是靠严重牺牲 `near_wall` 或 `interior` 稳定性换来的

换句话说：

- `TAWSS` 略高但 `near_wall` 明显更差，不算真正胜出
- `global` 指标好看但 `WSS` 不稳，不算真正胜出

### 6.2 几何先验成立规则

只有同时满足下面两条，才能说“显式几何先验成立”：

1. `Main` 相比 `Base` 在目标主指标上稳定改善
2. 这种改善至少在多数 seed 或多数代表病例上重复出现

如果只是单 seed、单病例偶然变好，不算成立。

### 6.3 进入第二轮的最低资格

只有满足下面条件，路线才允许进入 `Opt/Abl`：

1. Gate-0 全通过
2. `Main` 相比 `Base` 有稳定增益
3. `WSS / TAWSS` 没有明显崩坏
4. 训练和评估链条已稳定，不存在结果无法复现的情况

### 6.4 必须止损的情形

出现下面任一情况，应停止追加复杂实验：

1. `WSS / TAWSS` 在首轮两条路线里都不稳定
2. `geometry` 在两条路线里都没有可靠增益
3. 路线表现高度依赖个别病例或个别 seed
4. 评估链本身还在频繁变化，导致前后结果不可比

此时正确动作不是“再多试几个花样”，而是收缩问题定义。

---

## 7. 首轮结论的四种合法写法

### 7.1 MeshGNN 胜出

允许写法：

- 在 V2 的统一数据口径下，`Route-MeshGNN-V2` 在 `WSS / TAWSS` 与 `near_wall` 上整体优于 `Route-PointCloud-V2`
- 显式几何先验在 mesh-aware 表示上获得稳定增益

不允许写法：

- 因此 GNN 在所有复杂几何 surrogate 任务中普遍优于 point-cloud

### 7.2 PointCloud 胜出

允许写法：

- 在当前数据口径和采样设定下，`Route-PointCloud-V2` 比 `Route-MeshGNN-V2` 更适合作为主干
- 显式几何先验在 point-cloud 主干上依旧有效

不允许写法：

- 因此 mesh 表示没有价值

### 7.3 两者接近

允许写法：

- 两条路线在首轮主指标上接近，当前不能宣称某一主干绝对更优
- 后续是否继续取决于效率、稳定性和第二轮小规模优化结果

不允许写法：

- 任意挑一条当默认主线，然后把另一条静默删掉

### 7.4 两者都不够硬

允许写法：

- 当前证据只支持高精度场预测，不足以支撑强 hemodynamics surrogate 结论
- `WSS / TAWSS` 结果需要被降级为部分成立或局限性结果

不允许写法：

- 用最好看的几个可视化图替代系统结论

---

## 8. 首轮结果文件命名建议

建议在 `workspace/v2/reports/round1/` 下统一保存：

```text
workspace/v2/reports/round1/
├── round1_summary_table.csv
├── round1_geometry_gain.csv
├── round1_case_metrics.csv
├── round1_seed_stats.csv
├── round1_decision.md
└── figures/
    ├── round1_tawss_scatter.png
    ├── round1_region_error_bar.png
    ├── round1_case_visuals_caseA.png
    └── round1_seed_stability.png
```

`round1_decision.md` 至少要写清楚：

1. 哪条路线胜出
2. 凭什么胜出
3. 哪些结论还不能写
4. 第二轮是否开放

---

## 9. 首轮汇报最小骨架

汇报时建议严格按下面顺序：

1. **Gate-0 是否通过**
2. **首轮 5 组实验配置是否统一**
3. **主表 1：路线级总表**
4. **主表 2：geometry 增益**
5. **病例级 TAWSS/WSS 可视化**
6. **胜出路线与止损判断**

不要一上来先讲某个模型结构有多新。

---

## 10. 当前执行建议

在 `V2-Prep-01 ~ 04` 完成前，本文件只作为判定模板存在。  
在 `V2-Ref-Base-01 / V2G-Base-01 / V2G-Main-01 / V2P-Base-01 / V2P-Main-01` 全部完成后，必须按本文件生成首轮结论，不允许跳过。
