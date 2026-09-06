# WSS_PINN V4 anatomy-only 预处理准备与 173 例数据核查

> 日期：2026-09-03（夜）
> 范围：173 例 raw Fluent case/导出/UDF/monitor/transcript，Centerline V2 产物，
> 09-03 整改清单第 1–7 项的 Phase A/B 准备工作
> 性质：只读核查 + 新增预处理代码 + 隔离审计产物；**未改动 `data_new`、任何历史 data root、
> 旧 staging、配置、checkpoint，也未提交任何训练/CFD 作业**
> 结论：**继续 No-Go / `training_ready=false`**。Phase A 的三项算法合同（feature atlas、
> zone 拓扑、5 个解剖切面）已有可运行实现并在 173/173 上通过审计；但本次新发现一项
> 清单之外的合同级错误（长度单位），以及一项全队列性质的求解质量事实（收敛标志），
> 需要用户拍板后才能进入 Phase C 正式重建。
> 上游清单：[正式重建前剩余整改问题与验收计划](./WSS_PINN_V4正式重建前剩余整改问题与验收计划_2026-09-03.md)

> **2026-09-04 决策追加**：formal split 已改为 train138/test34，总数 172；旧 173 例
> 审计仍作为发现 `YANG_BAO_KUI` 资产覆盖问题的证据。formal backbone 按用户选择 B
> 固定为 PointNet P2V / 纯 PointNet++ D2 `c125-k128`，从随机初始化训练。当前
> `wss_pinn/v4/` 仍是旧 matched-v1.2 容量配平模型，Phase C 正式重建不能直接提交训练。

## 0. 一页结论

### 0.1 已完成并可复核

| 项 | 结果 | 证据 |
| --- | --- | --- |
| Fluent zone 拓扑解析器（全 cell/face/zone，多面体/混合/楔形） | 173/173 解析、face 总数与 header 一致、cell 体积全正 | `wss_pinn/v4/fluent_topology.py` |
| ASCII 导出 ↔ `.cas` 实体身份 | 173/173 双射、无歧义（cell 质心匹配 p99 <2.5e-7 m；壁面节点 7e-11 m） | `audits/topology/cases/*.json` |
| anatomy-only 计数 | `blood` 占导出 cell 的 38.2%–81.0%（中位 53.4%）；最小 blood 池 260,668 ≫ 15,000 | 同上 |
| 5 个 `blood↔bloodN` 接口 | 173/173 恰 5 个，平面度 ≥0.9987，语义 `inlet/out-le/li/ri/re` 173/173 完整 | 同上 |
| 解剖壁面 | 173/173 = “相邻 cell ∈ blood 的 wall face”；与 `ascii` 壁面导出节点集合一致 | 同上 |
| 入口流量合同 | 173/173 `vf-in` monitor 与 UDF 推导 `Q_actual(t)` 81 帧吻合（中位 0.000%，p95 0.64%，最大 2.0%） | §5 |
| 接口面积分流量 | CHEN 入口接口 vs UDF 最大 0.18%；ZHOU 质量守恒残差 0.85% | §3.3 |
| 7 段曲率 feature atlas（SG11/poly3、端点外推、唯一 trunk） | 173/173 finite，全队列 p99 0.174 /mm、max 0.799 /mm；合成回归 4/4 通过 | `atlas/atlas_summary.json`，`wss_pinn/tests/test_centerline_atlas.py` |
| 全队列求解收敛审计 | 172/173 transcript 可解析；见 §6 | `audits/solver_convergence/summary.json` |
| raw 内容 SHA256 清单 | 后台运行中（写入 `audits/raw_hash/`），覆盖 81 帧、壁面导出、case、UDF、monitor、journal | §8 |

### 0.2 新发现（不在 09-03 清单中）

1. **P0：长度单位合同错误。** Centerline V2 冻结的 `fluent_to_mm_unit_factor` 逐例为
   810–1088（中位 957），无一例接近 1000；而 173 例中 130 例权威 STL 与 Fluent 壁面跨度比
   精确为 1000.000，40 例在 1000±0.5% 内。即 STL = 真实 mm，Fluent = m，V2 “mm” 帧实际是
   真实 mm × factor/1000。影响：`local_radius_mm`、`curvature_per_mm`、near-wall 1.5 mm、
   rim buffer、以及 `coordinate_length_m = coord_scale_mm×1e-3`（PDE 梯度换算）逐例带
   −8%～+23.5% 误差（§1）。
2. **收敛标志是全队列现象，不是 ZHOU 单例。** 全部 journal 为 `dual-time-iterate 1280 20`；
   81 个导出步全部出现 `solution is converged` 的病例仅 50/172，76 例为 0/81（每步都触发
   20 次迭代上限）。ZHOU_KE_XUN 为 58/81（§6）。
3. **`Q_actual_peak` 是退化条件。** 173 例 UDF 的入口波形系数完全相同（峰值
   `1.07274e-4 m³/s`，`t=0.209 s`），仅 4 例因 `A_mesh≠A_udf` 不同；train138 标准差仅为均值的
   1.3%，所以 CHEN_SHU_LIN/LIU_YONG_LAN/ZHANG_YAN_SHAN 的 −8.3σ/−6.5σ/−4.8σ 是
   z-score 对近常量特征的放大，不是数据错误（§5）。
4. **ASCII 导出编号不是 `.cas` 索引。** `cellnumber`/`nodenumber` 为导出顺序编号，既不是
   case 索引也不是 zone 块置换（已穷举 720 种块置换验证）；zone 归属只能靠几何身份匹配（§2）。
5. 3 例体域导出是**节点值**而非 cell 值（`nodenumber` 列）：`ILO/WANG_SHU_SHENG-0/before`、
   `ILO/ZHANG_YAN_SHAN-0/before`、`AAA/ruputer/YANG_BAO_KUI`；1 例壁面导出是 face 中心值：
   `AG/slow/ZHANG_ZHI_JUN`。正式 builder 必须按导出类型分支（§2.4）。

### 0.3 需要用户拍板的问题

见 §10。核心四问：单位合同是否改为 `真实 mm = Fluent m × 1000`；三例 STL 失配病例是否从
`.cas` 壁面重建表面后重提中心线；求解质量 whitelist 采用“残差门槛”还是“重跑 CFD”；
`Q_actual_peak` 标准化改为固定物理尺度还是保留 z-score+whitelist。

## 1. P0（新增）：长度单位合同

### 1.1 证据

| 指标 | 数值 |
| --- | --- |
| 冻结 `fluent_to_mm_unit_factor`（173 例） | min 809.6 / p5 919.9 / p50 957.1 / p95 976.0 / max 1087.8；落在 1000±1% 的病例 **0** |
| STL 跨度 / Fluent 壁面跨度 == 1000（偏差 <1e-4） | **130 例** |
| 同上，偏差 <2% | 40 例（STL 为重网格化/光顺后的近似表面，仍为 mm） |
| 偏差 >2%（STL 不是 CFD 壁面） | **3 例**：`AAA/ruputer/ZHOU_KE_XUN`（1043/977/989）、`AAA/unruputer/LIU_WEN_QI`（790/988/887）、`ILO/YU_XIANG_SHENG-1/before`（1054/1040/1129） |

来源：`fluent_to_mm_unit_factor` 取自旧 WSS-min bundle 的 `unit_factor`，后者由
`pipeline_wss_min/preprocess.py::_resolve_unit_factor` 用**旧中心线 bbox 对角线 / 壁面
bbox 对角线**反推得到（正是 08-30 审阅 P1-6 指出的反推逻辑）。V2 “冻结单位不由新中心线决定”
只是冻结了这个旧反推值，并没有回到物理单位。

### 1.2 影响链

- V2 `centerline_paths_mm.csv` / `centerline_graph_mm.vtp` 的坐标、半径、0.5 mm 重采样步长、
  曲率均在“伪 mm”帧；
- 2026-08-31 staging 的 `local_radius_mm`、`curvature_per_mm`、`distance_to_wall_mm`、
  `NEAR_WALL_THRESHOLD_MM=1.5`、inlet-wall rim buffer 均逐例单位不一致；
- `physics_scales.coordinate_length_m = coord_scale_mm×1e-3` 进入
  `wss_pinn/v4/physics.py::_velocity_jacobian` 的 `gradient_hat / length_m`，因此 PDE 残差的
  空间导数逐例偏差 1000/factor（CHEN_SHI_MING 5.4%，LIU_WEN_QI 23.5%，YU_XIANG_SHENG −8.1%），
  而速度/压力标签是 SI 真值，连续性/动量残差在物理上不自洽；
- 刚性旋转、平移和归一化 `[-1,1]` 坐标不受影响（仅整体尺度）。

### 1.3 已实现的处理

- `wss_pinn/v4/centerline_atlas.py::load_v2_graph(unit_rescale=1000/factor)` 把 V2 图
  精确重标到真实 mm（均匀缩放，无需重提中心线）；本次 173 例 atlas 已按真实 mm 构建；
- 拓扑审计逐例写出 `unit_check`（跨度比、判定、重标系数）。
- 三例 STL 失配病例：其 V2 中心线来自非 CFD 壁面的表面，半径/弧长不可信；解析器已能从
  `.cas` 直接导出解剖壁面三角面（`anatomy_wall_faces`），可生成“由网格自身定义”的表面
  重新提取中心线（见 §10 问题 2）。

## 2. Fluent zone 拓扑与 ASCII 导出身份

### 2.1 解析器

`wss_pinn/v4/fluent_topology.py`：

- 读取 `3010/2010` 节点、`3012/2012` cell zone 区间及混合区的逐 cell 类型、`3013/2013`
  全部 face section（含 interior、element type 0/2/3/4/5）、`39/45` zone 声明；
- 解码修正：混合区和多边形区每条记录首个整数即节点数（旧解析器把 5 当作“多边形标记 +
  长度”，在四面体/楔形网格上没触发，但在多面体网格上必错）；
- Fluent 兼容 cell 质心：面心 = 绕顶点均值的三角扇面积加权中心；cell 估计 = 面心均值；
  质心 = 面棱锥体积加权。四面体与 Fluent 导出坐标一致到 1e-7 cell 尺寸，楔形 <4e-3，多面体
  <1e-1（仍远小于邻居距离）；
- `match_points`：KD-tree 最近邻 + 容差 + 一/二邻比 + 双射，含三类特殊处理：重合孪生节点
  （网格接缝上坐标相同的节点，7 例）、导出重复行（同坐标写两次，15 例壁面导出）、密堆小
  多面体的互为最近邻接受；
- 拓扑推导：`anatomy_topology` 给出按相邻 cell zone 分组的壁面、`blood↔bloodN` 接口
  （两侧 cell 属不同 zone 的两侧 face）、每个延长段所封闭的唯一 flow BC zone。

### 2.2 全队列事实（173/173）

| 事实 | 数值 |
| --- | --- |
| fluid zone | 173/173 恰为 `blood, blood1…blood5` |
| 网格族 | 95 例四面体+楔形；78 例多面体（+少量六面体） |
| cell 数 | 451,674–1,310,293（中位 757,678） |
| 接口 face zone 命名 | 6 套：`in+/out-le+…`（76）、`inlet1/outlet-ln1…`（37）、`inlet1/outlet-leftnei1…`（33）、`inlet/leftnei…`（19）、`inlet1/outlet_leftnei1…`（7）、含拼写错误 `oulet_leftnei1`（1）；**必须按拓扑而非名字识别** |
| 壁面 zone 命名 | `wall/wall1–5`（118）、`wall-extending1–5`（31）、`wall_extending1–5`（8）；13 例另有自动命名 1–2 面的壁面 zone（如 `chen-shi-ming:--.2:14:3603`），拓扑上属解剖壁面 |
| 类型 `0x28` 的未挂接三角面 zone（`c0=c1=0`） | 192 个 section，存在于部分 ILO 例；不参与任何计算，审计已确认忽略 |
| `ascii_in` 导出 | 170 例为 cell 值且行数 == cell 总数（含全部延长段）；3 例为节点值（§0.2 第 5 条） |
| `ascii` 壁面导出 | 172 例节点值，且节点集合 == 解剖壁面节点集合（差异仅来自自动命名的 1–2 面 zone）；`ZHANG_ZHI_JUN` 为 face 中心值 |
| 同目录多个 `.cas.gz` | 5 例（`LIU_BAO_JUN_1`、`WANG_SHU_SHENG_1`、`GENG_CHUN_LAI_1`、`WAGN_LI_JUN.msh`、`FAN_JIAN_MING/.cas.gz`）；审计固定使用 qs-smooth-v3 边界 manifest 的 provenance 路径 |

### 2.3 anatomy-only 计数

- 导出 cell 中 `blood` 占比 38.2%–81.0%（中位 53.4%）——旧 staging 的 15k 瞬态池与
  全部 steady 监督点里有约一半来自延长段；
- 最小 blood 池 `ILO/SUN_DONG_XIN-0/before` 260,668 cells，远大于 15,000；
- 每例接口共享节点（同时在解剖壁面和某个接口上）119–366 个，是 rim buffer 的确切对象。

### 2.4 对正式 builder 的合同要求

1. 从 `.cas` 解析 → 计算质心/节点 → 与 `ascii_in` 第一帧做双射身份匹配 → 得到逐行 zone；
   其余 80 帧按坐标 identity 复用（旧 staging 已验证 81 帧坐标完全一致）；
2. cell 导出：`zone == blood` 即解剖点；节点导出（3 例）：节点属于任一 blood cell 即解剖点，
   同时落在接口上的 3–5k 节点单独标记，不进入 PDE/no-slip 池；
3. 壁面：`anatomy_wall_faces` 的 face/node；WSS 标签从 `ascii` 导出按节点 identity 回填
   （`ZHANG_ZHI_JUN` 按 face 中心 identity）；
4. 每例 manifest 写 zone id/名称、原始/保留/排除计数、mask SHA256、`.cas` SHA256。

## 3. 解剖切面（5 个 `blood↔bloodN` 接口）

### 3.1 拓扑结果

173/173 恰 5 个接口；每个延长段恰封闭 1 个 flow BC zone（1 velocity-inlet + 4 pressure-outlet；
4 例出口为 `outflow` 类型也已覆盖）；`blood` 本身不直接接触任何 flow BC。平面度
`|Σ面积向量| / Σ|面积向量|` 最小 0.9987。语义映射通过 qs-smooth-v3 边界 manifest 的
`mesh_zone_id → out-le/li/ri/re` 完成，173/173 完整；入口接口面积 2.31e-4–9.99e-4 m²。

### 3.2 产物字段

每个接口：face 连接、两侧相邻 cell（blood 侧 / 延长段侧）、面心、由解剖指向延长段的面积
向量、单位法向、远端 BC zone id/名称/类型、语义标签。这满足 09-03 §4 “面节点、相邻 cell、
法向、面积权重、opening semantic ID”。

### 3.3 面积分流量验证

用接口两侧 cell 的导出速度点乘面积向量得到 81 帧接口流量：

| 病例 | 入口接口 Q vs UDF `Q_actual(t)` | `vf-in` monitor vs UDF | 质量守恒 `max|Q_in−ΣQ_out|/Q_peak` | 出口接口 vs `vf-out*` monitor |
| --- | ---: | ---: | ---: | --- |
| `AG/fast/CHEN_SHI_MING` | 最大 0.18% | 0.013% | 3.3%（单侧）/ 2.0%（两侧均值） | 相关系数 1.0000，差 <1% |
| `AAA/ruputer/ZHOU_KE_XUN`（多面体，9-02 重算） | 与**新** UDF 一致；与旧 manifest `q_actual` 差 26.3%（旧 A_udf 错） | 同 | 0.85% | （无 vf-out） |

ZHOU 的 26% 差异正是 09-03 §1.2 所述 A_udf 修复（旧 `0.000510`→新 `0.000691`，与
ZUO_DAO_SHENG 互换）的直接证据：raw 81 帧、monitor、UDF 三者现在自洽，只有旧 V4 manifest
过期。

## 4. 曲率 feature atlas（09-03 §2）

### 4.1 实现（`wss_pinn/v4/centerline_atlas.py`）

- 从 `centerline_graph_mm.vtp` 显式树（5 端点、3 分叉）拆出 **7 个唯一分支段**，每条物理边只
  存一次；子段以父段末端 junction 为首样本，但该重复样本在展平 atlas 中剔除（无重叠、无缺口，
  由 `_validate_atlas` 强制检查）；
- 每段严格等弧长网格（`linspace` 到段长，步长 ≤0.5 mm）；坐标三分量与半径用 SG
  `window=11, polyorder=3` 拟合；端点先用局部三阶多项式外推 5 个样本再套同一 SG 算子；
  样本数不足 11 的短段退化为单一低阶多项式；曲率 `|r'×r''|/|r'|³`；
- 每样本保存 raw/smoothed 坐标、切线、曲率、raw/smoothed 半径、`dR/ds`、segment/parent id、
  沿树到最近 junction/端点的距离、junction/端点 mask、拟合残差；
- 图中孤立点（终末尖刺剪除残留，37 例各 1–3 点）被忽略并记录。

### 4.2 合成回归（`wss_pinn/tests/test_centerline_atlas.py`，4/4 通过）

直线曲率 <1e-9；半径 20 mm 圆弧中段相对误差 <2e-3、端点 <2e-2；Y 型树分解为 7 段且
节点覆盖无重复、junction 折角不泄漏到相邻段；短段退化路径。

### 4.3 173 例结果（真实 mm）

| 指标 | 数值 |
| --- | --- |
| 展平样本总数 | 198,905 |
| 曲率 p50 / p99 / max | 0.042 / **0.174** / **0.799** /mm（目标 p99<0.5、hard max<10 均满足；train138 p99 0.178） |
| 逐例 max 中位 / p90 | 0.232 / 0.334 /mm |
| SG 拟合残差 max 中位 / 最大 | 0.089 / 0.306 mm |
| 峰值位置 | 端点（≤2.5 mm）28 例、junction（≤2.5 mm）68 例、内部 77 例 |
| 旧逐路径曲率（重标后）对照 | CHEN max 0.380→0.193；ZHOU 3.246→0.420；LI_LAO_PING 1.692→0.799 |

whitelist 复核队列（max >0.5 /mm）：

| 病例 | max /mm | 位置 | 判断 |
| --- | ---: | --- | --- |
| `AAA/ruputer/LI_LAO_PING`（train） | 0.799 | trunk 内部，s=157 mm，距 junction 86 mm，局部半径 21.8 mm | V2 折线在瘤腔内 ±1 mm 折转 80°：半径 21.8 mm 的管腔不可能 1.25 mm 曲率半径弯折，属 VMTK 在宽大瘤腔内的中心线伪折，不是解剖弯曲；SG 未抹平（残差 0.31 mm）。叠图：`audits/atlas_review/AAA__ruputer__LI_LAO_PING_curvature_peak.png` |
| `ILO/YU_XIANG_SHENG-1/before`（test） | 0.655 | out-re 端点 0.5 mm 内 | 末端“钩回”；且该例 STL 非 CFD 壁面 |
| `AG/slow/LIU_FENG`（train） | 0.552 | 端点 | 末端钩回 |

## 5. 入口条件与 `Q_actual_peak` 长尾（09-03 §5.1）

- 173 例 UDF（`udf-inlet.c` / `udf-inlet4.c`，`WANG_YONG_FAN` 仅 `libudf/src/`）Fourier
  系数完全相同：`Q_nom(t)` 峰值 `1.07274e-4 m³/s` @ `t=0.209 s`；
- `A_mesh/A_udf` 偏差 >0.1% 仅 4 例：`WANG_CHUN_MING` +0.9%、`ZHANG_YAN_SHAN` −6.6%、
  `LIU_YONG_LAN` −8.9%、`CHEN_SHU_LIN` −11.2%（后三例即清单中的 −4.8σ/−6.5σ/−8.3σ）；
  ZHOU/ZUO 的 A_udf 已互换回正确值，`Q_actual` 分别变为旧值的 1/1.356 和 1/0.737；
- train138 `Q_actual_peak` 均值 1.0707e-4、标准差 1.43e-6（旧 manifest 4.29e-6）：
  169 例数值完全相同，z-score 把 4 例面积失配病例推到极端。结论：三例来源正确（CFD 确实以
  更小的实际流量运行），属 09-03 所说“来源正确的生理/设置长尾”；但更根本的问题是该条件
  在 z-score 下退化，建议改用固定物理尺度归一（如 `Q/Q_nom_peak`，取值 0.888–1.009）
  并显式 whitelist 四例（§10 问题 4）；
- `vf-in` monitor 与 UDF 推导 `Q_actual(t)` 在 81 帧上：173/173 齐全，最大相对误差中位
  0.000%、p95 0.64%、最大 2.0%（`AG/slow/HOU_SHEN_QIAN`）。

## 6. 求解收敛（09-03 §5.2）：从 ZHOU 单例到全队列

`wss_pinn/v4/audit_solver_convergence.py` 按 **mtime 最新**的 transcript（按大小选会误取
ZHOU 的 1 月旧 transcript）逐步解析：

| 指标 | 数值 |
| --- | --- |
| 可解析 transcript | 172/173（`AG/fast/ZHANG_XIU_ZHEN` 无 transcript） |
| journal 迭代上限 | 173/173 均为 `dual-time-iterate 1280 20` |
| 81 导出步全部 `solution is converged` | **50/172** |
| 0/81 收敛（每步触发 20 次上限） | **76** |
| 部分收敛 | 46（含 ZHOU_KE_XUN 58/81，上限触发 27 步，continuity p95/max 2.60e-3/3.33e-3） |
| 末次 continuity 残差 p95（逐例） | 中位 9.9e-4，p90 2.5e-3，最大 6.3e-3；>1e-3 的病例 61，>2e-3 的 28，>3e-3 的 9 |
| 最差 | `ILO/WANG_JIN_MING-0/before` 6.3e-3、`ILO/SUN_XU_XIA-1/before` 5.7e-3、`AAA/unruputer/GUO_BAO_CHUN` 4.9e-3 |

含义：Fluent 的收敛标志对应 continuity <1e-3；“converged”病例残差本就在 1e-3 量级，
ZHOU 只比它们高约 2.6 倍。若坚持 09-03 §5.2 的“81/81 达标”，需要重跑的是 122 例而非 1 例。
建议以残差门槛（例如 continuity p95 ≤2e-3、max ≤3e-3）建立 solver-quality whitelist，
并把逐例残差写入 manifest 供训练敏感性分析；ZHOU 与其余 28 例 >2e-3 的病例由用户决定是否
重跑（§10 问题 3）。速度残差全队列最大 4.0e-4（`LIU_ZONG_YANG` 1.6e-4 为次高）。

## 7. monitor / transcript 语义（09-03 §5.3）

| 类别 | 病例数 | 出口质量流量标签来源 |
| --- | ---: | --- |
| `vf-in` + `p-out*` + `vf-out*` + transcript | 76 | monitor 或 transcript `Q_ave_out*` |
| `vf-in` + `p-out*` + transcript（无 `vf-out*`，AAA/ILO 常见；ILO 另有 2 个未命名 `report-def`） | 96 | transcript `Q_ave_out*`（UDF `execute_at_end` 每迭代打印） |
| `AG/fast/ZHANG_XIU_ZHEN` | 1 | 无 transcript；`vf-outri-rfile.out` 只有表头列无数值 → 出口 `out-ri` 质量流量标签不可恢复，mask 必须显式 false；4 个 `p-out*` 完整 |

四例“缺 optional monitor”病例的 transcript（9-01/9-02 重算）均覆盖 step 1–1280 且含
`Q_ave/P_ave` 行；旧 builder 的“取最大 transcript”规则会误选 1 月旧 transcript（LI_BING_YI、
LIU_YUE_DONG、LI_SHENG_WEN 各有两个），正式 parser 必须按 mtime/与帧一致性选择。

## 8. raw provenance 内容哈希

`audits/raw_hash/<case>.json`：每例 81 帧 `ascii_in`、81 帧 `ascii` 壁面、`.cas.gz`、顶层与
`libudf/src` UDF、全部 monitor/transcript、journal/slurm 的 SHA256+size+mtime。约 1.3 TB，
后台 6 进程运行中（本文写作时 55/173），完成后由正式 builder 直接引用。

## 9. 交付物

代码（新增，未提交 git）：

- `wss_pinn/v4/fluent_topology.py`：Fluent case 全拓扑解析、Fluent 兼容质心、身份匹配、
  解剖壁面/接口推导；
- `wss_pinn/v4/centerline_atlas.py`：唯一分支段 feature atlas 与点映射；
- `wss_pinn/v4/build_feature_atlas.py`：173 例 atlas 构建（真实 mm）；
- `wss_pinn/v4/audit_anatomy_topology.py`：173 例拓扑/身份/接口/单位审计；
- `wss_pinn/v4/audit_solver_convergence.py`：173 例 transcript 收敛审计；
- `wss_pinn/tests/test_centerline_atlas.py`：合成回归（直线/圆弧/Y 型/短段）。

产物（隔离根 `outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/`）：

- `audits/topology/cases/*.json` + `summary.json`（173/173 pass）；
- `atlas/cases/<cid>/atlas.npz|summary.json` + `atlas/atlas_summary.json`；
- `audits/solver_convergence/cases/*.json` + `summary.json`；
- `audits/atlas_review/*_curvature_peak.png`（LI_LAO_PING、YU_XIANG_SHENG 叠图）；
- `audits/raw_hash/*.json`（进行中）。

## 10. 待用户确认（2026-09-03 快照）

> 本节保留 09-03 审计当时的问题队列。2026-09-04 已决定的 split、
> `YANG_BAO_KUI` 排除和骨干选择以 § 15 及当前正式重建计划为准，不再是待决项。

1. **单位合同**：是否把 V4 正式合同改为“真实 mm = Fluent m × 1000”（V2 产物按
   `1000/fluent_to_mm_unit_factor` 均匀重标，`coordinate_length_m` 相应改正）？这会改变
   `local_radius_mm/curvature_per_mm` 的数值范围和 near-wall/rim 阈值的物理含义，属 08-30
   审阅 Phase 2 “单位直接读取 V2 冻结合同”的推翻。
2. **三例 STL 失配病例**（ZHOU_KE_XUN、LIU_WEN_QI、YU_XIANG_SHENG-1）：建议用 `.cas`
   解剖壁面三角面直接生成表面 STL 后重跑 Centerline V2（GNN_vmtk 环境），而不是继续沿用
   非 CFD 壁面提取的中心线；是否同意，以及是否顺带对全部 173 例改用网格自身壁面作为唯一
   权威表面（消除 STL 依赖）。
3. **求解质量**：按残差门槛 whitelist（推荐，附训练敏感性）还是重跑不达标病例？若重跑，
   优先范围建议为 continuity p95 >2e-3 的 28 例（含 ZHOU），journal 改为更高的每步迭代上限。
4. **`Q_actual_peak` 归一化**：改为固定物理尺度（`Q/Q_nom_peak`）并 whitelist 四例，还是
   维持 train z-score 并 whitelist；`WANG_CHUN_MING`（+0.9%）是否也登记。
5. **LI_LAO_PING** 的瘤腔内折角：确认为中心线伪折后，处理方式建议为“瘤腔段（局部半径
   显著大于近端管径）曲率置零/掩码并记录”，而不是全局加大平滑窗；请确认。
6. 端点钩回（28 例峰值在端点 2.5 mm 内，含 YU_XIANG_SHENG、LIU_FENG）：是否允许 atlas 在
   端点 1 个局部半径内改用外推切线（不采用最后几个 VMTK 样本）。
7. `FAN_JIAN_MING` 目录下隐藏文件 `.cas.gz`（22.9 MB，1 月 5 日）以及 4 例 `*_1.cas.gz`/
   `*.msh.cas.gz` 旧网格是否可归档移出 raw 目录（本次未动）。

## 11. 2026-09-03 当时的下一步（现行顺序见 § 15 与正式重建计划）

1. Phase C builder：从 raw 单遍读取 81 帧 → 身份匹配 → blood-only steady/transient
   （15k 唯一 blood cell）→ 解剖壁面/no-slip → 5 接口（面节点、法向、面积权重、81 帧
   面积分流量）→ atlas 映射三通道 geometry → train138-only stats → manifest 绑定 raw SHA；
   `rcr_loss=disabled_at_anatomical_interface` 显式写入配置；
2. 若采用问题 2 方案，先跑三例（或全部）网格壁面 Centerline V2，再重建 atlas；
3. formal 172 例 CPU 全遍历、GPU dry-run、少病例过拟合（normal / LI_LAO_PING /
   ZHOU / ILO 各一例）；源队列 173 例审计只作历史证据。

## 12. 2026-09-04 复核补充（针对用户七点提问）

### 12.1 `ascii_in` 是否含延长段

Fluent 导出对话框当前只勾选 `blood`（[1/6]）反映的是现在的 GUI 状态，不是产生磁盘文件时的
自动导出设置。磁盘证据：

| 病例 | `ascii_in` 行数 | `.cas` 六 zone 总 cell | 仅 `blood` cell | 导出体域 z 跨度 / 壁面 z 跨度 (mm) | 落在壁面包围盒外 (+2 mm) 的行 |
| --- | ---: | ---: | ---: | --- | ---: |
| `AG/fast/CHEN_SHI_MING` | 575,210 | 575,210 | 265,660 | 483.9 / 222.9 | 48.6% |
| `AAA/ruputer/ZHOU_KE_XUN`（9-02 重算） | 636,955 | 636,955 | 516,157 | 381.5 / 246.4 | 15.1% |
| `ILO/LU_FU_SHAN-0/before` | 706,914 | 706,914 | 503,357 | 467.8 / 308.5 | 25.4% |

173/173 行数均等于六 zone 总 cell 数（3 例节点导出等于节点数）。壁面导出确实只含 `wall`
面（172/173 与解剖壁面节点集合完全一致），与用户截图一致。结论：体域不需要重新导出，
用拓扑身份匹配得到的 `blood` mask 过滤即可；重新导出需要已被清理脚本删除的 `.dat`，等于重跑 CFD。

### 12.2 三例“STL 错配”的确切含义

“错配”指权威 STL 不是 CFD 网格所用的那张表面（130 例 STL 点与网格壁面节点逐点重合，
40 例在 0.5% 内）。以真实 mm、仅平移拟合后：

| 病例 | STL 点数 / 壁面节点数 | 壁面→STL 距离 p50 / p95 / max (mm) | 诊断 |
| --- | --- | --- | --- |
| `ZHOU_KE_XUN` | 109,889 / 79,783 | 1.85 / 6.8 / 16.6 | 同一解剖，但原点差 ≈[9, 278, 947] mm 且为另一版重网格表面 |
| `LIU_WEN_QI` | 12,439 / 14,749 | 9.76 / 28.9 / 41.4 | 瘤腔形状不同，不是网格所用表面（另一次分割/版本） |
| `YU_XIANG_SHENG-1` | 26,942 / 57,231 | 1.42 / 3.9 / 6.6 | 同一解剖、单位本就是 mm，但 STL 入口延长段更长（z 跨度 342 vs 303 mm），旧 bbox 反推把单位因子推成 1087.8 |

叠图：`audits/atlas_review/stl_vs_cfd_wall_three_mismatch_cases.png`。三例现有 atlas
（按 `1000/factor` 重标）不可用：YU 的中心线落在管腔外（`|d_wall−R|/R` p50 0.76），即使回到
STL 自身坐标也只有 0.49；必须从 `.cas` 解剖壁面重提中心线。其余 170 例重标是精确的。

### 12.3 同目录多个 `.cas.gz` 的取舍

| 病例 | 使用 | 备选 | 依据 |
| --- | --- | --- | --- |
| `LIU_BAO_JUN-0` | `LIU_BAO_JUN.cas.gz`（2026-01-26） | `LIU_BAO_JUN_1.cas.gz`（2022-01-20） | 两者 cell/node 数相同，导出均能匹配（2.6e-6 m）；journal `read-case` 与 2026-01 运行时间指向前者 |
| `WANG_SHU_SHENG-0` | `WANG_SHU_SHENG.cas.gz`（2026-01-23） | `WANG_SHU_SHENG_1.cas.gz`（2021-11-20） | 备选 cell/node 数相同但节点坐标与导出**不匹配**（最大 0.63 m），只能用前者 |
| `GENG_CHUN_LAI-1` | `GENG_CHUN_LAI.cas.gz`（2026-01-24） | `GENG_CHUN_LAI_1.cas.gz`（2021-12-04） | 同 LIU_BAO_JUN |
| `WAGN_LI_JUN-1` | `WAGN_LI_JUN.cas.gz`（2026-01-24） | `WAGN_LI_JUN.msh.cas.gz`（2021-11-29） | 同上；`.msh.cas.gz` 为网格阶段旧件 |
| `FAN_JIAN_MING` | `FAN_JIAN_MING.cas.gz` | 隐藏文件 `.cas.gz`（同日，gzip 头 `flntgz-240203`，NTFS 来源） | 网格完全相同、导出可匹配；隐藏文件是杂散副本，`ls` 不显示，需 `ls -la` |

### 12.4 求解质量分层（替代“每例必须收敛标志”）

| 层 | 定义 | 病例数（train） |
| --- | --- | ---: |
| A | 81 导出步全部 `solution is converged` | 50（38） |
| B | 未全收敛，但末次 continuity 残差 max <2e-3 | 85（68） |
| C | max 2e-3–5e-3（含 ZHOU_KE_XUN 3.3e-3、LI_LAO_PING、CHEN_SHI_MING） | 33（28） |
| D | max ≥5e-3：`SUN_XU_XIA-1`(test)、`DING_JUN_FENG`、`WANG_JIN_MING-0`、`GUO_BAO_CHUN` | 4（3） |

多面体网格更难在 20 次迭代内达标（0/81 的 76 例中 57 例为多面体）。建议：A/B 直接使用；
C 使用但在 manifest 写 `solver_quality` 分层并做剔除敏感性；D 由用户决定是否以更高迭代上限重跑。

## 13. 2026-09-04 执行记录：三例中心线重提与多余 `.cas.gz` 归档

### 13.1 三例从 `.cas` 解剖壁面重提中心线（`tools/centerline_v2_meshwall.py`）

表面由 `.cas` 拓扑生成（真实 mm，多边形扇形三角化，重合点合并），开口语义由接口拓扑指定；
提取/门槛/输出复用 V2 原逻辑，产物在 `outputs/centerline_v2_meshwall_20260904/`。

| 病例 | 表面 | 结果 | 端点—开口 max | 壁面 p95/R | >2R 占比 | 新 atlas 曲率 p99 / max | `|d_wall−R|/R` p50 |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| ZHOU_KE_XUN | 39,914 多边形面 → 239,178 三角 | pass，无软标记 | 3.29 mm (0.27R) | 1.76 | 1.6% | 0.230 / 0.436 | 0.002 |
| LIU_WEN_QI | 29,307 三角 | pass，无软标记 | 3.21 mm (0.18R) | 1.37 | 1.0% | 0.122 / 0.205 | 0.002 |
| YU_XIANG_SHENG-1 | 28,634 多边形面 → 171,531 三角 | pass，无软标记 | 0.71 mm (0.20R) | 1.25 | 0.0% | 0.170 / 0.271 | 0.002 |

对照：正常病例 CHEN_SHI_MING 的 `|d_wall−R|/R` p50 为 0.004；三例旧 V2 产物分别为 0.24/0.70/0.76。
开口面积与接口面积逐一相等（中心距 ≤2.1 mm）。三例 atlas 已用 `--source-root` 重建并写回
`atlas/cases/<cid>/`，cohort 汇总重算：p99 0.174 /mm、max 0.799 /mm（LI_LAO_PING）；whitelist 复核队列
只剩 LI_LAO_PING（瘤腔伪折）与 LIU_FENG（端点 0.552）。YU_XIANG_SHENG 原来的端点峰消失。

### 13.2 多余 `.cas.gz` 归档（用户授权）

五个文件已移动到 `data_new/_archive/superseded_cas_20260904/<cid>/`，`archive_manifest.json`
记录原路径、SHA256、大小、mtime 与理由；raw_hash 清单中对应条目已标注 `archived_to`。
`WANG_SHU_SHENG.cas.gz` 正式文件在 9-03 23:23 被用户重新保存（四例重算之一），其 raw 清单
需在重算完成后重做。

### 13.3 待用户重算的四例

体域节点值导出：`ILO/WANG_SHU_SHENG-0/before`、`ILO/ZHANG_YAN_SHAN-0/before`、`AAA/ruputer/YANG_BAO_KUI`；
壁面 face 中心值导出：`AG/slow/ZHANG_ZHI_JUN`。用户已提交集群；完成后需：重做四例 raw_hash、
拓扑审计与求解收敛审计，再决定是否需要重提中心线（网格不变则不需要）。

## 14. 2026-09-04 四例重算复核：三例通过，`YANG_BAO_KUI` 网格被覆盖

用户在集群重算了四例并于 09-04 01:56 前全部完成。逐例复核（重跑 raw_hash、拓扑审计、收敛审计）结果：

### 14.1 三例通过

| 病例 | 角色 | 体域导出 | 壁面导出 | `blood` 行数（占比） | cell identity | 壁面 identity | 收敛层 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `ILO/WANG_SHU_SHENG-0/before` | train | **cell 值**（已修复） | node（1-23 旧导出，仍匹配） | 405,577（68.3%） | 2.5e-6 m | 7.1e-11 m | B |
| `ILO/ZHANG_YAN_SHAN-0/before` | train | **cell 值**（已修复） | node（3-12 旧导出，仍匹配） | 389,342（66.3%） | 2.2e-6 m | 5.0e-11 m | B |
| `AG/slow/ZHANG_ZHI_JUN` | train | cell 值 | **仍是 face 中心值**（1-09 旧导出） | 285,580（39.5%） | 5.0e-7 m | 4.3e-8 m | A（81/81） |

三例的 `.cas.gz` 均被重新保存（SHA 相对 qs-smooth-v3 冻结值已变更），但网格实体数与旧记录逐项相同
（cells/nodes 完全一致），且**旧壁面导出仍以 1e-10 m 量级匹配新 case 文件**，证明网格本身未变，只是重存。

`ZHANG_ZHI_JUN` 的壁面仍是 face 中心值：本轮重算只重导出了体域（journal 无壁面导出命令）。该形态
审计已支持（face 中心双射匹配，误差 4.3e-8 m，完整覆盖解剖壁面 20,069 面），**可直接使用**，
WSS 标签改挂到面而不是节点即可；若希望统一为节点值需要再导出一次壁面。

修复后全队列：**173/173 体域导出均为 cell 值**（原 3 例节点值已清零），壁面 171 例节点值 + 1 例面中心值。
拓扑审计 **172/173 通过**，`blood` 占比 38.2%–81.0%（中位 53.4%）。

### 14.2 `AAA/ruputer/YANG_BAO_KUI`（test35）：`.cas.gz` 被 `ZHANG_YAN_SHAN` 的网格覆盖

**证据（逐项比对，非推测）**：

| 项 | `YANG_BAO_KUI.cas.gz`（09-04 00:08） | `ZHANG_YAN_SHAN.cas.gz`（09-03 23:26） | `YANG_BAO_KUI` 原网格（已丢失） |
| --- | --- | --- | --- |
| cells / nodes | 587,033 / 1,758,468 | 587,033 / 1,758,468 | 918,344 / 2,401,847 |
| 节点坐标数组 | **逐点完全相等（max diff = 0）** | — | — |
| cell zone 区间 | `blood1(1–128442) blood(128443–517784) …` | **完全相同** | — |
| 壁面 face 数 | `wall 28306`, `wall-extending1 8912`… | **完全相同** | — |
| 文件 SHA / 大小 | `45145cdf…` / 70,380,838 | `e362592c…` / 70,380,874 | `b3520fc6…` / 104,792,152 |

两文件仅 gzip 头（文件名/时间戳）不同，网格内容完全一致。因此：

- 新导出的 81 帧 `ascii_in`（09-04 01:56，587,033 行）是 **`ZHANG_YAN_SHAN` 的流场**，不是 `YANG_BAO_KUI` 的；
- 壁面导出 `ascii/`（2026-01-27，96,719 节点）仍是 `YANG_BAO_KUI` 自己的解剖，与新体域数据**分属两位患者**；
  拓扑审计报错 `96,717 行超出身份容差（最大 0.458 m）`，即两者相距 458 mm；
- `YANG_BAO_KUI` 的原网格（`b3520fc6…`，104.8 MB）与原 81 帧体域数据已被覆盖，**全盘搜索无备份**
  （仅 `qs-smooth-v3` manifest 与 raw_hash 清单留有原 SHA 作为证据）。

**幸存资产**：`YANG_BAO_KUI.stl`（75.4 MB，1-12）、81 帧壁面导出（1-27）、`data_wss_min` bundle、
2026-08-31 staging 的 15k 瞬态数组（旧口径，不足以重建 anatomy-only 全池）。

**2026-09-03 当时建议（已被 § 15.1 取代）**：当时可选方案是从幸存 STL
重新划分网格并重跑 CFD。2026-09-04 用户已选择排除 `YANG_BAO_KUI`；新 split
保持 train138，test35→test34，formal 总数固定为 172。因此“重算还是剔除”不再是未决项。

### 14.3 全队列 `.cas` provenance 漂移核对

以当前磁盘 `.cas` SHA 对照 `qs-smooth-v3` 冻结 provenance，**恰好 13 例不一致**，与已知重算记录逐条吻合，
其余 159 例（`YANG_BAO_KUI` 因审计报错未产出 JSON）网格文件自 2026 年初以来未变：

| 组 | 病例 |
| --- | --- |
| 八例低压力族 raw 更新（09-01/02） | `LIU_YUE_DONG`、`LI_SHENG_WEN`、`LI_BING_YI`、`MENG_GUANG_QIN`、`CHEN_SHU_LIN`、`LIU_ZONG_YANG`、`ZHANG_YONG_SHENG`、`WANG_FU_SHUN` |
| Q 长尾修复（09-02） | `ZHOU_KE_XUN`、`ZUO_DAO_SHENG` |
| 导出口径重算（09-03/04） | `WANG_SHU_SHENG`、`ZHANG_YAN_SHAN`、`ZHANG_ZHI_JUN`（另 `YANG_BAO_KUI` 见 §14.2） |
| D 层重跑，导出路径重存、网格不变（09-05，§27） | `DING_JUN_FENG`、`GUO_BAO_CHUN`、`WANG_JIN_MING-0`、`SUN_XU_XIA-1` |
| 补 transcript 重跑，导出路径重存、网格不变（09-06，§27.5） | `ZHANG_XIU_ZHEN` |

正式 bundle 必须绑定**当前** `.cas` SHA，不得沿用 `qs-smooth-v3` manifest 里的冻结值。

### 14.4 当前全队列状态（2026-09-04）

| Gate | 状态 |
| --- | --- |
| 拓扑/身份/接口/解剖壁面 | **172/173 通过**；仅 `YANG_BAO_KUI` 因网格被覆盖失败 |
| 体域导出口径 | **173/173 为 cell 值**（原 3 例节点值已修复） |
| 壁面导出口径 | 171 例节点值 + 1 例面中心值（`ZHANG_ZHI_JUN`，可用） |
| 曲率 feature atlas | 173/173，p99 0.174 /mm、max 0.799 /mm；复核队列 `LI_LAO_PING`、`LIU_FENG` |
| 求解质量分层 | A 49（train 38）／B 86（68）／C 33（28）／D 4（3） |
| raw 内容 SHA256 清单 | 173/173 完成（四例重算后已刷新） |
| 中心线 | 170 例用 2026-08-28 产物（重标真实 mm）+ 3 例网格壁面重提 |

## 15. 2026-09-04 决策支撑与 split 变更

### 15.1 `YANG_BAO_KUI` 已按用户决定排除，新 split 为 train138/test34

新文件（**未覆盖冻结 split**）：

```text
wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_bc_rcr_v4_anatomy_only_train138_test34_s1234.json
SHA256 c4c78561d82e5922f90ca3813175ae5da98731a0cee974d36918fd6afba05655
```

- `train_cases` 与原 split **逐条完全相同**（138 例，训练集未受影响）；
- `test_cases` 由 35 → 34，移除 `AAA/ruputer/YANG_BAO_KUI`；
- 5 折 CV 定义在 train138 内部，已核实不含该例，原样保留；
- 新增 `excluded_cases` 字段记录排除原因与证据指向 §14.2；
- 原 split 文件与 SHA 保留不动，作为历史矩阵的 provenance。

后续正式 bundle、stats、manifest 与 Gate 一律以 **172 例** 为总数。

### 15.2 `Q_actual_peak` 归一化：建议改用固定物理尺度，或直接不作为输入

**为什么现在会炸到 −8.3σ**：173 例 UDF 的入口波形系数完全相同，`Q_nom_peak = 1.07274e-4 m³/s`；
逐例差异只来自 `Q_actual_peak = Q_nom_peak × A_mesh/A_udf`。该比值的偏离量级分布为：

| `|A_mesh/A_udf − 1|` | 例数 | 含义 |
| --- | ---: | --- |
| `<1e-6` | 156 | UDF 常数取整造成的浮点差，数值上就是同一个数 |
| `1e-6 ~ 1e-4` | 12 | 同上 |
| `1e-4 ~ 1e-3` | 1 | 边缘 |
| `>1e-3` | **4** | 真正的面积失配：`WANG_CHUN_MING` +0.9%、`ZHANG_YAN_SHAN` −6.6%、`LIU_YONG_LAN` −8.9%、`CHEN_SHU_LIN` −11.2% |

train138 上该字段变异系数仅 1.33%，几乎全部来自这 4 例。z-score 把“近常量 + 4 个真实偏移”放大成
±8σ，这不是数据错误，是归一化方式与特征性质不匹配。

**三个选项**：

| 方案 | 特征值 | 优点 | 代价 |
| --- | --- | --- | --- |
| A. 固定物理尺度（推荐，若保留） | `Q_actual_peak / Q_nom_peak = A_mesh/A_udf` ∈ [0.888, 1.009] | 无需 train 统计、无 whitelist、换队列不漂移；物理含义明确=“施加的速度剖面实际输送了名义流量的百分之多少” | 需改 `_transformed_bc` 与 stats schema |
| B. 从输入向量移除 | — | 最简单；该字段对 169 例本就无信息（`A_mesh` 已在向量中） | 4 例真实流量偏差最多 11%，模型无从得知 |
| C. 维持 z-score + whitelist | 原值 | 不改代码 | 4 例长期占据 ±5~8σ，且队列一变 z 就变 |

**用户问“不作为输入是否就不用额外处理”——是的**。`Q` 的其余用途都在物理量纲下，不经归一化：

- `U_c_m_s = Q_actual_peak / A_mesh`：连续性/动量残差与入口 BC 损失的无量纲化尺度（`v4/physics.py`）；
- 入口 Dirichlet 目标：稳态用 `U_c`，瞬态用 `Q_nom(t) / A_udf`（`v4/data.py:377-380`），即 Fluent UDF 实际施加的速度；
- 瞬态时间特征里的 `q_nom(t)/1e-4`（`v4/models.py:166`）是**全队列相同**的相位信号，不是病例条件。

所以选 B 时无需任何补偿；唯一建议是把 A 方案的那一个无量纲通道保留下来（成本 1 维），
否则 4 例失配病例在条件向量里与其余病例不可区分。

**顺带**：移除 `Q_actual_peak` 后，BC 向量里仍 `|z|>6` 的只剩 `ZOU_LI_SHUN` 的 `out-re` 的
`log10_R2`（+6.46）与 `log10_C`（−6.50），需单独 whitelist；若把 BC 各 log10 字段也改用
geometry 已在用的 max-aware 尺度（`std = max(population_std, max_abs_dev/6)`），则全部字段
天然落在 ±6 内，不再需要逐例 whitelist。

### 15.3 `LI_LAO_PING` 瘤腔段：曲率在瘤腔内没有定义，不是生理真峰

证据图：`audits/atlas_review/LI_LAO_PING_sac_curvature_profile.png`（三联：局部半径 / 曲率 + 拟合残差 / `k·R`）。

- 该例 trunk 段长 243 mm，瘤腔位于 `s = 141.5–173.5 mm`（局部内切半径 R>12 mm，峰值 21.8 mm），占 trunk 的 13%；
- 曲率峰 `k = 0.799 /mm` 出现在瘤腔顶部 `s = 157 mm`，对应**弯曲半径 1.25 mm**，而该处管腔半径 21.8 mm；
- 判据 `k·R = 17.4`。对一根管子，`k·R ≥ 1` 意味着内侧壁自相交，**几何上不可能**；
- 该峰是**单个孤立样本**（前后基线 0.05–0.15 /mm），SG 拟合残差同处从 0.048 mm 跳到 0.306 mm，
  说明原始折线在该处有一个局部折角，而不是一段连续弯曲；
- 原始 V2 折线在 ±1 mm 内转过约 80°。

**成因**：VMTK 中心线基于最大内切球代价，在宽大瘤腔内该代价近乎平坦，路径约束很弱，会在腔内游走/折跳。
腔体本身不是管道，"中心线曲率"在这里不对应任何解剖量。

**这不是单例问题**：全队列 `k·R > 1` 的样本共 9,255 / 199,033（4.65%），涉及 172 例，
集中在 AAA 瘤腔（`FENG_LI_XIN` 204、`CAO_HONG_TAI` 177、`WANG_AN` 177 等）。
`LI_LAO_PING` 只是最极端的一个（`k·R` 最大 17.4，其余多在 1–5）。

**影响程度**：模型通道是 `signed_log1p(k)` 再按 train max-aware 尺度 z-score，天然被限制在 ±6，
**不会数值爆炸**；真正的问题是语义——在对 AAA 血流最关键的瘤腔区，该通道编码的是提取噪声。

**三个选项**：

| 方案 | 做法 | 评价 |
| --- | --- | --- |
| A. 加可靠性掩码（推荐） | 保留 atlas 原始曲率作为 provenance；映射到点云时，`k·R > 1` 的样本标记 `curvature_reliable = 0`，模型通道改用该段的可靠邻域值（或置 0），同时把已有的 `radial_ratio` / 新增“瘤腔指示”作为独立通道 | 不伪造数据、可审计、对全队列 4.65% 样本一致处理 |
| B. 改用无量纲 `k·R` 并饱和 | 曲率通道换成 `min(k·R, 1)` | 尺度不变、天然有界；但改变了通道语义，需重训对照 |
| C. 只对 `LI_LAO_PING` 单例 whitelist | 记录后原样保留 | 掩盖了这是 172 例共有的系统现象 |

不建议加大平滑窗：那会同时抹掉真实的髂动脉弯曲（`k·R` 在正常段本就接近 1）。

## 16. 2026-09-04 二次复核：`ZHANG_ZHI_JUN` 壁面未变更；`k·R>1` 影响重新定量（修正 §15.3）

### 16.1 `ZHANG_ZHI_JUN`：本次重跑只重导出了体域，壁面未变

09-04 02:26–03:27 的运行确实重存了 `.cas.gz`（`abb907d4…`，22,953,644 B）并重写了全部 81 帧
`ascii_in`（03:19–03:26），但 `ascii/` 目录**自 2026-01-09 起未被触碰**：

| 项 | 现状 |
| --- | --- |
| 壁面文件时间 | 2026-01-09 00:58–01:06（81 帧） |
| 壁面表头 | `cellnumber, …, wall-shear, x/y/z-wall-shear` → 仍是 **Cell Center**（面中心值） |
| 壁面行数 | 20,069 = 解剖壁面 face 数 |
| 体域表头/行数 | `cellnumber, …` / 722,718，与网格 cell 数一致 |

审计结论：该例 **全部 gate 通过**（面中心值双射匹配误差 4.3e-8 m，完整覆盖 20,069 个解剖壁面 face），
收敛为 A 层（81/81，continuity p95 9.8e-4）。**当前状态可直接用于正式构建**，WSS 标签挂到面而非节点。

若仍希望统一成节点值，需要在 case 中显式增加一次表面导出：`Location = Node`、`Surfaces = wall`、
`Quantities = Static Pressure + Wall Shear Stress + X/Y/Z-Wall Shear Stress`（即用户第一张截图的设置）；
本次运行的 journal 只触发了体域导出，所以没有产生新壁面文件。

### 16.2 `k·R>1` 的实际影响远小于 §15.3 的表述（修正）

§15.3 用 `k·R>1` 同时指代了两件不同的事，这里拆开定量：

| 现象 | 判据 | 全队列样本 | 说明 |
| --- | --- | ---: | --- |
| (a) 宽腔/瘤腔区 | `R > 1.5×` 该例中位 R | 58,400（**29.3%**，逐例中位 31%，最大 45%） | `k·R>1` 主要由 **R 大** 触发，不是 k 坏 |
| (b) 孤立提取折角 | `k > 8×` 所在段中位数 且 `> 0.3 /mm` | **15 个（0.008%），仅 6 例** | 真正的伪影 |

关键判据：**瘤腔内曲率的逐样本抖动 `|Δk|` 中位数 0.0073 /mm，正常段 0.0068 /mm，比值仅 1.07** ——
即瘤腔内的曲率序列和正常段一样平滑，值本身并没有被污染，只是数值小且对"管道弯曲"这一语义不适用。

(b) 的 15 个样本逐例为：`LI_LAO_PING` 3（max 0.799）、`SUN_XU_XIA-1` 4（0.402）、`ZHANG_XUN_LIAN` 3（0.419）、
`HOU_SHI_GUO-0` 2（0.401）、`GENG_CHUN_LAI-1` 2（0.307）、`LIU_FENG` 1（0.552）。

**映射到训练点的比例**（8 例抽样，blood cell 最近邻落在 `k·R>1` 样本上）：

| 病例 | blood cells | 点占比 | 体积占比 | 这些点处 k 的 p99 |
| --- | ---: | ---: | ---: | ---: |
| `LI_LAO_PING` | 539,739 | 15.2% | 44.5% | 0.799 |
| `FENG_LI_XIN` | 703,989 | 20.2% | 40.8% | 0.162 |
| `CAO_HONG_TAI` | 503,741 | 23.0% | 33.5% | 0.157 |
| `WANG_AN` | 509,660 | 22.6% | 45.6% | 0.168 |
| `LIU_WEN_QI` | 450,033 | 10.7% | 32.7% | 0.112 |
| `LU_FU_SHAN-0` | 503,357 | 5.0% | 6.8% | 0.164 |
| `ZHANG_ZHI_JUN` | 285,580 | 3.9% | 9.0% | 0.165 |
| `CHEN_SHI_MING` | 265,660 | 3.2% | 5.5% | 0.143 |

即 AAA 瘤腔例有 10–23% 的点（33–46% 的体积）落在宽腔区，但这些点上的曲率值 p99 只有 0.11–0.17，
与全队列 p99 0.174 同量级——**不是异常值，只是"曲率在这里信息量低"**。

### 16.3 修正后的建议

1. **只处理 (b) 的 15 个样本**：按 segment 做稳健滤波（`k > max(8×段中位数, 0.3)` 的样本替换为局部中位数），
   全队列影响 0.008%，逐例记录在 manifest。这是唯一真正写错了数的地方。
2. **不再建议对 29% 的宽腔样本做掩码**（§15.3 方案 A 撤回）：那里的曲率平滑且量级正常，
   `local_radius_mm` 通道已经直接告诉模型"此处管腔很宽"，不存在误导。
3. **可选**：把无量纲 `k·R` 或瘤腔指示作为 **auxiliary** 通道落库（不进首轮模型输入），
   留给后续消融验证是否有增益。成本仅存储。
4. 数值上曲率通道经 `signed_log1p` + max-aware z-scale 天然限制在 ±6，任何情况下都不会爆炸。

## 17. 2026-09-04 壁面 WSS 标签合规审计（172 例）

实现：`wss_pinn/v4/audit_wall_wss.py`；产物：`audits/wall_wss/cases/*.json` + `summary.json`。
每例抽 3 帧（1120 / peak / 1280）做格式、自洽、朝向、覆盖、provenance 五类检查。
**结果：157/172 通过。**

### 17.1 通过的项（全队列）

| 检查 | 结果 |
| --- | --- |
| 81 帧齐全、步号与体域一致、单例内文件大小一致 | 171/171（`ZHANG_ZHI_JUN` 正在重跑，见 17.5） |
| 全部有限、`wall-shear` 非负、无 WSS=0 节点 | 171/171 |
| 逐帧坐标一致（可按行号对齐 81 帧） | 170/171（例外见 17.4） |
| WSS 向量切向于壁面（用 `.cas` 面法向独立验证） | p95 队列中位 **0.029**，最大 0.053，无一例 >0.2 |
| 标量 `wall-shear` 与向量模自洽 | 见 17.2 |
| 覆盖解剖壁面 | 168/171（3 例差 1–4 个节点，见 17.3） |
| 周期性（1120 与 1280 同相位） | 平均 WSS 相对差中位 **0.1%**，p90 7.6%；>10% 的 7 例 |

峰值帧 WSS 分布（Pa）：`p50` 中位 2.37 [0.51, 8.41]，`p99` 中位 21.8 [5.8, 106.6]，
`max` 中位 52.7 [8.6, 320.1]。

**格式变体**：4 例壁面导出是**空格分隔**而非逗号（`HAN_JIAN_JUN`、`LI_SHI_QIANG`、`LI_SHU_KUN`、
`LI_ZHEN_SHAN`），正式 parser 必须同时支持（`raw_io.read_table` 已支持）。

### 17.2 节点导出的标量与向量不等价——标签应取标量列

Fluent 把 `|τ|` 和三个分量**分别**插值到节点，所以节点上 `|向量| ≤ 标量` 恒成立：

| 导出类型 | `|向量| ≤ 标量` 比例 | `(标量−|向量|)/标量` 中位 | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| node（171 例） | 1.0000 | 队列中位 **0.28%**（最大 0.52%） | 3–5% | 0.91–0.95 |
| face-centre（`ZHANG_ZHI_JUN` 旧导出） | — | **0**（完全相等） | 0 | 0 |

max 接近 1 的节点位于分离/再附着线，相邻面剪切方向相反，矢量平均后模长趋零。

**结论：做 WSS 幅值回归时，标签取 `wall-shear` 标量列**；不要用三分量算模长，那会在分离区
系统性低估（中位偏低 0.28%，局部可低 90% 以上）。需要方向（OSI/TAWSS 矢量）时才用分量，
并知悉节点分量已被方向平均。

### 17.3 3 例导出比解剖壁面少 1–4 个节点

| 病例 | 导出节点 | 解剖壁面节点 | 差 | 原因 |
| --- | ---: | ---: | ---: | --- |
| `AG/slow/LIU_FENG` | 13,590 | 13,591 | 1 | 另有 2 个各 1 面的自动命名壁面 zone 未参与导出 |
| `AG/slow/WANG_KE` | 13,108 | 13,109 | 1 | 1 个 2 面的自动命名 zone |
| `AG/slow/ZHANG_WEI_XIAN` | 18,551 | 18,555 | 4 | 2 个共 19 面的自动命名 zone |

影响可忽略，但正式合同里 **WSS 标签点集应定义为"导出壁面"而非"解剖壁面"**，两者在这 3 例差 1–4 点。

### 17.4 `AAA/unruputer/HAN_JIAN_FU`：首帧是单独补导出的

- 帧 1120 的写入时间比其余 80 帧晚 **219.8 小时**，且**节点顺序不同**（同行坐标最大差 30.6 mm）；
- 帧 1120 壁面压力 12,214.9 Pa（绝对值，与体域一致），其余 80 帧 −47～−2 Pa（表压）；
- WSS 本身连续正常（81 帧均值 0.165→2.675 Pa，1120 与 1122 平滑衔接）。

处理：该例 81 帧**不能按行号对齐**，须按坐标/节点 ID 对齐；壁面压力列在帧 1120 与其余帧不同基准，
若下游用到壁面压力需统一或弃用该帧。另有 5 例首帧也来自独立会话
（`LU_FU_SHAN-0`、`ZHANG_YONG_SHENG-0`、`CHEN_SHU_LIN`、`LIU_BAO_JUN-0`、`CHENG_GUANG_SEN`），
但其压力与顺序均一致，无需处理。

### 17.5 **10 例壁面标签来自已被取代的求解**（最重要）

判据：壁面 `pressure` 列与同一时刻体域最近单元压力之差。同一次求解写出的壁面与体域，该差
**恰好为 0 Pa**（对照组 6 例验证）。以下 10 例差值达 4,053–14,902 Pa：

| 病例 | 角色 | Δp 中位 (Pa) | 壁面滞后 (天) | 说明 |
| --- | --- | ---: | ---: | --- |
| `AG/slow/LI_BING_YI` | train | −14,902 | 237 | 八例低压力族，壁面仍是修复前的解 |
| `AG/slow/LIU_ZONG_YANG` | train | −13,536 | 239 | 同上 |
| `ILO/LI_SHENG_WEN-0/before` | train | −12,840 | 230 | 同上 |
| `ILO/LIU_YUE_DONG-0/before` | train | −12,821 | 229 | 同上 |
| `ILO/ZHANG_YONG_SHENG-0/before` | test | −12,439 | 216 | 同上 |
| `AAA/ruputer/MENG_GUANG_QIN` | train | −12,235 | 174 | 同上 |
| `AAA/ruputer/WANG_FU_SHUN` | test | −11,658 | 218 | 同上 |
| `AAA/unruputer/CHEN_SHU_LIN` | train | −10,131 | 220 | 同上 |
| `AAA/ruputer/ZHOU_KE_XUN` | train | **+7,576** | 218 | A_udf 修复前流量高 1.356 倍 |
| `AAA/ruputer/ZUO_DAO_SHENG` | train | −4,053 | 217 | A_udf 修复前流量低 0.737 倍 |

这 10 例的壁面 WSS 标签对应的是**修复前的流场**，与同目录的体域数据不是同一个解。直接拿来做
WSS 回归，等于给 10 例（8 train + 2 test）配了错误标签。

**无法只重导出**：全库 **0 个 `.dat` 文件**（cleanup 脚本已删除），只能重跑 CFD 才能得到匹配的壁面。

### 17.6 上尾极值：不是交界环带伪影，与收敛质量相关

高 WSS 节点到 `blood↔bloodN` 交界环带的距离中位为 12–41 mm（全壁面到环带中位 49–80 mm），
**并不集中在环带**，所以剔除环带无助于压制极值（抽样 6 例，剔除 <2 mm 环带后 max 不变）。

按求解质量分层看峰值帧 WSS：

| 层 | 例数 | p50 中位 | p99 中位 | max 中位 |
| --- | ---: | ---: | ---: | ---: |
| A 81/81 收敛 | 48 | 3.17 | 23.99 | 51.31 |
| B max<2e-3 | 85 | 1.80 | 18.63 | 54.95 |
| C 2e-3~5e-3 | 33 | 2.65 | 19.19 | 48.12 |
| **D ≥5e-3** | **4** | 2.94 | **77.51** | **268.37** |

D 层 4 例（`WANG_JIN_MING-0`、`SUN_XU_XIA-1`、`DING_JUN_FENG`、`GUO_BAO_CHUN`）的上尾显著异常；
整体 Spearman 相关 0.25–0.33，说明极值部分来自收敛不足、部分来自真实的髂动脉狭窄。

> **2026-09-06 修正**：四例以每步 60 次迭代重跑后，峰值帧 WSS p50/p99/max 变化均 <1%（§27.4），
> 上尾是真实狭窄血流，不是收敛不足；分层与高 WSS 相关只是这类流动更难在 20 次内收敛。

p99 超过队列 p95 的 9 例：`WANG_JIN_MING-0`(106.6)、`ZHOU_KE_XUN`(89.2)、`WU_GUANG_CUN`(86.7)、
`SUN_XU_XIA-1`(79.5)、`YANG_WANG_QI-1`(78.6)、`DING_JUN_FENG`(75.5)、`ZHAO_CHANG_SHAN-0`(75.5)、
`MA_TIAN_YI`(69.3)、`ZHANG_JIN_CHUN-1`(66.5)。

### 17.7 结论与待办

**可直接用作 WSS 回归标签的：157 例。**

| 待办 | 病例 | 处理 |
| --- | --- | --- |
| 壁面与体域非同一解 | 10 例（8 train + 2 test） | 必须重跑 CFD（无 `.dat`）；在此之前这 10 例不得进入 WSS 回归 |
| 首帧独立导出、顺序与压力基准不同 | `HAN_JIAN_FU` | 按坐标/节点 ID 对齐 81 帧；壁面压力列该帧单独处理 |
| 导出少 1–4 节点 | `LIU_FENG`、`WANG_KE`、`ZHANG_WEI_XIAN` | 标签点集定义为"导出壁面" |
| 壁面重跑中 | `ZHANG_ZHI_JUN` | 完成后重跑本审计 |
| 上尾离群 | 9 例（D 层 4 例为主） | 建议在 WSS 回归中登记并做剔除敏感性，不建议裁剪数值 |
| 幅值标签取值 | 全部 | 用 `wall-shear` 标量列，不用三分量模长 |
| parser | 全部 | 同时支持逗号与空格分隔 |

## 18. 2026-09-04 壁面重跑进行中（滚动记录）

针对 §17.5 的 10 例壁面失配，用户在 09-04 06:00 后陆续提交整例重跑
（`2.jou` 为完整 `dual-time-iterate 1280 20`，会同时重写体域与壁面，因此壁面与体域从此同源）。

### 18.1 作业覆盖

| 病例 | 作业 | 状态（09-04 08:00） |
| --- | --- | --- |
| `AG/slow/ZHANG_ZHI_JUN` | 12952 | **已完成**，见 18.2（该例原为面中心导出，非失配） |
| `ILO/LIU_YUE_DONG-0/before` | 12953 | 运行中 |
| `ILO/LI_SHENG_WEN-0/before` | 12954 | 运行中 |
| `AG/slow/LI_BING_YI` | 12956 | 运行中 |
| `ILO/ZHANG_YONG_SHENG-0/before` | 12957 | 运行中 |
| `AAA/ruputer/WANG_FU_SHUN` | 12958 | 运行中 |
| `AAA/unruputer/CHEN_SHU_LIN` | 12959 | 排队 |
| `AAA/ruputer/MENG_GUANG_QIN` | 12960 | 排队 |
| `AAA/ruputer/ZHOU_KE_XUN` | 12961 | 排队 |
| **`AG/slow/LIU_ZONG_YANG`** | — | **未提交** |
| **`AAA/ruputer/ZUO_DAO_SHENG`** | — | **未提交** |

`MENG_GUANG_QIN` 曾在 06:08–06:09 编译 UDF 并写入 `fluent.slurm`，但当时未进入队列；
09-04 07:45 前后已作为 12960 重新提交。

### 18.2 `ZHANG_ZHI_JUN`（12952）完成后的核实

| 项 | 结果 |
| --- | --- |
| 需要的 81 个偶数步（1120–1280） | 全部存在，`nodenumber` 节点值，10,095 节点 |
| 壁面 vs 体域压力差（step 1120） | **−0.02 Pa** → 同一次求解 |
| 峰值帧 WSS | 均值 0.456、p99 1.48、max 3.25 Pa |
| 标量与向量模的中位差 | 0.36%（节点插值的正常量级） |
| **`ascii/` 目录残留** | 另有 **80 个奇数步（1121–1279）2026-01-09 的旧面中心文件**（20,070 行） |

残留原因：cleanup 脚本只在 `ascii/` 内删除 1–1119，不删奇数步；正式 builder 必须按**步号选取**
所需 81 帧，不能"读取目录内全部文件"。审计已相应改为按步号取帧，并把残留数单列为诊断
（gate `no_stale_extra_wall_files`）。这些奇数步旧文件与新解不一致，建议清理，但属破坏性操作，未执行。

### 18.3 作业结束后的既定动作

对涉及的每一例重跑四项审计（脚本 `scratchpad/reaudit.sh`）：raw 内容 SHA256、拓扑与
anatomy-only、求解收敛分层、壁面 WSS 标签；随后更新 §17 的通过计数与 §14.3 的 `.cas` 漂移表。

### 18.4 **根因：Fluent 拒绝覆盖已存在的导出文件，壁面重跑基本无效**

09-04 的重跑作业里，除 `LI_BING_YI` 外**都没有真正替换壁面文件**。原因在 transcript 里是明写的：

```text
The file ".../ascii/LI_SHENG_WEN-1122" already exists.
OK to overwrite? [cancel]
```

批处理模式下该提示的默认响应是 **cancel**，写入被跳过。因为每例 `ascii/` 里本来就存有同名的
81 个旧文件（步号 1120–1280 偶数），新解在这 81 步上**一个都没写进去**；新写成功的只有
1–1119 和奇数步 1121–1279，而它们随后被 `fluent.slurm` 的清理逻辑删除或保留为无用的奇数步。

逐例证据：

| 病例 | 作业 | transcript `already exists` | 需要的 81 步中真正更新的 | 结论 |
| --- | --- | ---: | ---: | --- |
| `ILO/LI_SHENG_WEN-0/before` | 12954 COMPLETED | **81**（全部 `[cancel]`） | **1**（仅 1120） | 壁面未更新 |
| `AG/slow/ZHANG_ZHI_JUN` | 12952 COMPLETED | **81**（全部 `[cancel]`） | **1**（仅 1120） | 壁面未更新；另留 80 个奇数步新文件 |
| `AG/slow/LI_BING_YI` | 12956 COMPLETED | **0** | **81** | **成功**，见下 |
| `ILO/LIU_YUE_DONG-0/before` | 12953 运行中 | 已累计 62 | 预计 0–1 | 将同样失败 |
| `ILO/ZHANG_YONG_SHENG-0/before` | 12957 运行中 | 已累计 63 | 预计 0–1 | 将同样失败 |
| `AAA/ruputer/WANG_FU_SHUN` | 12958 运行中（step ~884） | 0（未到 1120） | 预计 0–1 | 将同样失败 |
| `AAA/unruputer/CHEN_SHU_LIN` | 12959 运行中（step ~54） | 0（未到 1120） | 预计 0–1 | 将同样失败 |
| `AAA/ruputer/MENG_GUANG_QIN` | 12960 排队 | — | 预计 0–1 | 将同样失败 |
| `AAA/ruputer/ZHOU_KE_XUN` | 12961 排队 | — | 预计 0–1 | 将同样失败 |

同样的失败在 09-01/09-02 那批运行里就已发生：`LI_BING_YI` 的 `Fluent_12731.out`（09-01）
显示 `already exists` **1280 次**、成功写入 `ascii/` **0 次**。这解释了 §17.5 的 10 例为何壁面
一直停留在 1 月——**它们的壁面从来没有被后续重跑覆盖过**。

#### `LI_BING_YI` 为什么成功

它的 09-04 运行把导出文件名前缀改成了 `LI_BING_YI-1120`，与旧文件 `LI_BING_YI-<step>` 不冲突，
因此 161 个新文件（步号 1120–1280 全步）全部写入成功。核对：

| 步 | 新文件 `LI_BING_YI-1120-<step>` | 旧文件 `LI_BING_YI-<step>` |
| --- | --- | --- |
| 1120 | Δp 与体域 **0.00 Pa**，WSS 均值 0.463 | Δp **−14,111.60 Pa**，WSS 均值 0.441 |
| 1162（peak） | Δp **0.04 Pa**，WSS 均值 5.429 max 61.27 | Δp **−14,902.26 Pa**，WSS 均值 5.629 max 66.41 |
| 1280 | Δp **0.00 Pa**，WSS 均值 0.462 | Δp **−14,274 Pa**，WSS 均值 0.442 |

81 个偶数步全部齐备、节点数一致（13,910）。**该例壁面已修复**。
副作用：`ascii/` 里同时存在两套命名，`*-1124` 会同时匹配 `LI_BING_YI-1124` 与
`LI_BING_YI-1120-1124`，正式 builder 的按步号取文件逻辑会产生歧义，必须清掉旧的一套。

#### 建议的补救做法（需用户执行，本次未动 `data_new`）

重跑前先把旧壁面挪走，让 Fluent 能正常写入：

```bash
cd data_new/<病例>
mkdir -p ascii_superseded_20260904 && mv ascii/* ascii_superseded_20260904/
# 然后重新提交，导出文件名保持默认 <NAME>
```

比"改导出名"更干净：既避免覆盖冲突，又不产生两套命名；旧文件保留可追溯。
若沿用 `LI_BING_YI` 的改名方式，则必须在跑完后把旧的一套移出 `ascii/`。

### 18.5 审计工具相应加固（本轮新增三项判据）

为避免"静默读到错误文件"，`audit_wall_wss.py` 增加：

| 判据 | 含义 | 触发例 |
| --- | --- | --- |
| `step_to_file_unambiguous` | `*-<step>` 只能匹配一个文件 | `LI_BING_YI`（`LI_BING_YI-1124` 与 `LI_BING_YI-1120-1124` 同时匹配；且步 1120 会优先匹配到**旧**文件） |
| `frames_same_format` | 抽样帧的 ID 列与行数必须一致 | `ZHANG_ZHI_JUN`（1120 为节点值 10,095 行，1162 为旧的面中心值 20,069 行） |
| `no_stale_extra_wall_files` | `ascii/` 内不得有需要步号以外的残留 | `ZHANG_ZHI_JUN`（80 个奇数步）、`LI_BING_YI`（81 个旧平名文件） |

另外帧的选取改为**按步号显式取文件**，不再依赖目录内文件总数。

### 18.6 三例已完成作业的判定

| 病例 | 判定 | 失败判据 | 说明 |
| --- | --- | --- | --- |
| `AG/slow/LI_BING_YI` | **解已修复，目录待清理** | `no_stale_extra_wall_files`、`step_to_file_unambiguous` | 新文件 81 步齐全且 Δp=0.00 Pa；需把 81 个旧平名文件移出 `ascii/` |
| `AG/slow/ZHANG_ZHI_JUN` | **壁面未更新** | `frames_same_format` | 仅 1120 为新节点值；1122–1280 仍是 1 月面中心值；另有 80 个奇数步新文件 |
| `ILO/LI_SHENG_WEN-0/before` | **壁面未更新** | `same_solution_as_volume` | 仅 1120 为新；其余 80 步仍为 1 月旧解（Δp −12,840 Pa） |

注意：这三例的**体域**都已被本轮重跑改写（`ascii_in` 全部为 09-04），因此它们的 raw 内容
SHA256、拓扑、收敛三项审计需要重做；壁面则要等按 18.4 的做法重跑之后才有意义。

### 18.7 09-04 08:20 快照：5 例已完成，结论与预测一致

| 病例 | 作业 | `already exists` | 81 步中更新数 | 壁面结论 | 新 transcript 收敛 |
| --- | --- | ---: | ---: | --- | --- |
| `AG/slow/ZHANG_ZHI_JUN` | 12952 | 81 | 1 | 未更新（帧格式混杂） | 81/81，p95 9.8e-4 |
| `ILO/LI_SHENG_WEN-0/before` | 12954 | 81 | 1 | 未更新 | 0/81，p95 5.9e-4 |
| `AG/slow/LI_BING_YI` | 12956 | **0** | **81** | **已更新**（改名规避冲突） | 51/81，p95 3.3e-3 |
| `ILO/LIU_YUE_DONG-0/before` | 12953 | 81 | 1 | 未更新 | 0/81，p95 3.6e-4 |
| `ILO/ZHANG_YONG_SHENG-0/before` | 12957 | 81 | 1 | 未更新 | 0/81，p95 4.2e-4 |

体域侧三项审计（raw SHA256、拓扑 anatomy-only、求解收敛）已对这 5 例重做：**拓扑 5/5 通过**。

壁面 WSS 审计当前 **157/172 通过**，15 例未通过：

| 失败判据 | 例数 | 病例 |
| --- | ---: | --- |
| `same_solution_as_volume` | 10 | `LIU_ZONG_YANG`、`MENG_GUANG_QIN`、`ZHOU_KE_XUN`、`ZUO_DAO_SHENG`、`CHEN_SHU_LIN`、`HAN_JIAN_FU`、`LIU_YUE_DONG-0`、`LI_SHENG_WEN-0`、`WANG_FU_SHUN`、`ZHANG_YONG_SHENG-0` |
| `covers_anatomy_wall_exactly` | 3 | `LIU_FENG`、`WANG_KE`、`ZHANG_WEI_XIAN`（各差 1–4 节点，可忽略） |
| `frames_same_format` | 1 | `ZHANG_ZHI_JUN` |
| `no_stale_extra_wall_files` + `step_to_file_unambiguous` | 1 | `LI_BING_YI`（解已对，只需清理旧文件） |
| `coordinates_identical_across_frames` | 1 | `HAN_JIAN_FU`（首帧独立导出） |

`ZHOU_KE_XUN` 与 `ZUO_DAO_SHENG` 至今**未提交**任何重跑作业。

## 19. 2026-09-04 壁面重跑批次最终结果（9 例作业全部结束）

09-04 06:00–10:23 共 9 个作业运行结束，队列已空。**除 `LI_BING_YI` 外全部未能更新壁面**。

| 病例 | 作业 | 拒绝覆盖 | 81 步中更新 | 体域帧 | 新解收敛 | 壁面判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `AG/slow/LI_BING_YI` | 12956 | **0** | **81** | 81 | 51/81 | **已更新**，需清理旧文件 |
| `AG/slow/ZHANG_ZHI_JUN` | 12952 | 81 | 1 | 81 | 81/81 | 未更新，帧格式混杂 |
| `ILO/LIU_YUE_DONG-0/before` | 12953 | 81 | 1 | 81 | 0/81 | 未更新 |
| `ILO/LI_SHENG_WEN-0/before` | 12954 | 81 | 1 | 81 | 0/81 | 未更新 |
| `ILO/ZHANG_YONG_SHENG-0/before` | 12957 | 81 | 1 | 81 | 0/81 | 未更新 |
| `AAA/ruputer/WANG_FU_SHUN` | 12958 | 81 | 1 | 81 | 64/81 | 未更新 |
| `AAA/unruputer/CHEN_SHU_LIN` | 12959 | 81 | 1 | 81 | 0/81 | 未更新 |
| `AAA/ruputer/MENG_GUANG_QIN` | 12960 | 81 | 1 | 81 | 0/81 | 未更新 |
| `AAA/ruputer/ZHOU_KE_XUN` | 12961 | 81 | 1 | 81 | 58/81 | 未更新 |

`AG/slow/LIU_ZONG_YANG` 与 `AAA/ruputer/ZUO_DAO_SHENG` 全程未提交作业，壁面仍为 1 月旧解。

> `LI_BING_YI` 一行的"81 步中更新"需显式按新前缀统计。按 `*-<step>` 通配会得到 50，
> 那是歧义匹配的假象：`*-1120` 会优先匹配旧文件 `LI_BING_YI-1120`。显式检查
> `LI_BING_YI-1120-<step>` 时 81 步齐全（新前缀共 161 文件，旧平名 81 文件）。

### 19.1 体域侧审计已全部重做（9 例）

这 9 例的 `ascii_in` 都被本轮重跑改写，因此 raw 内容 SHA256、拓扑 anatomy-only、求解收敛
三项已逐例重跑：

- **拓扑 anatomy-only：9/9 通过**；全队列 172/173（仅 `YANG_BAO_KUI` 因网格被覆盖失败，已排除出 split）。
- **求解收敛**：新解普遍比旧解更接近门槛，`ZHANG_ZHI_JUN` 达到 81/81、`WANG_FU_SHUN` 64/81、
  `ZHOU_KE_XUN` 58/81；其余 0/81 但 continuity 残差多在 4e-4–6e-4，属 B 层。
- **raw 内容 SHA256**：9 例已刷新。

### 19.2 壁面 WSS 标签当前状态：157/172 通过

| 失败判据 | 例数 | 病例 |
| --- | ---: | --- |
| `same_solution_as_volume` | 10 | `LIU_ZONG_YANG`、`MENG_GUANG_QIN`、`ZHOU_KE_XUN`、`ZUO_DAO_SHENG`、`CHEN_SHU_LIN`、`HAN_JIAN_FU`、`LIU_YUE_DONG-0`、`LI_SHENG_WEN-0`、`WANG_FU_SHUN`、`ZHANG_YONG_SHENG-0` |
| `covers_anatomy_wall_exactly` | 3 | `LIU_FENG`、`WANG_KE`、`ZHANG_WEI_XIAN`（差 1–4 节点，可忽略） |
| `frames_same_format` | 1 | `ZHANG_ZHI_JUN` |
| `no_stale_extra_wall_files` + `step_to_file_unambiguous` | 1 | `LI_BING_YI`（解已对，仅需清理） |
| `coordinates_identical_across_frames` | 1 | `HAN_JIAN_FU` |

注意与 §17.5 相比，失配名单从 10 例变为 10 例但成员有变：`LI_BING_YI` 已修复移出，
`HAN_JIAN_FU` 因首帧问题计入。

### 19.3 下一步（需用户执行）

1. **按 §18.4 重跑 11 例**（10 例失配 + `ZHANG_ZHI_JUN` 帧格式混杂）：提交前先
   `mkdir -p ascii_superseded_20260904 && mv ascii/* ascii_superseded_20260904/`，
   导出名保持默认；跑完用 transcript 的 `already exists` 计数为 0 验证。
2. **`LI_BING_YI` 清理**：把 81 个旧平名文件移出 `ascii/`，消除 `*-<step>` 歧义。
3. **`LIU_ZONG_YANG`、`ZUO_DAO_SHENG`** 尚未提交，一并处理。
4. 全部完成后重跑四项审计；壁面 WSS 标签可用例数应从 157 升到 170/172
   （余 `HAN_JIAN_FU` 首帧与 3 例少 1–4 节点为已知可接受项）。

## 20. 待处理 15 例的完整路径清单（2026-09-04）

全部路径相对仓库根 `/public/newhome/cy/Digital_twin/GNN/`。

### A 组：壁面与体域非同一次求解，**必须重跑**（9 例）

| # | 完整路径 | 队列 | 壁面旧解日期 | 壁面 vs 体域压力差 |
| ---: | --- | --- | --- | ---: |
| 1 | `data_new/AG/slow/LIU_ZONG_YANG` | train | 2026-01-07 | −13,536 Pa |
| 2 | `data_new/AAA/ruputer/MENG_GUANG_QIN` | train | 2026-03-12 | −12,235 Pa |
| 3 | `data_new/AAA/ruputer/WANG_FU_SHUN` | **test** | 2026-01-27 | −11,658 Pa |
| 4 | `data_new/AAA/ruputer/ZHOU_KE_XUN` | train | 2026-01-27 | +7,576 Pa |
| 5 | `data_new/AAA/ruputer/ZUO_DAO_SHENG` | train | 2026-01-28 | −4,053 Pa |
| 6 | `data_new/AAA/unruputer/CHEN_SHU_LIN` | train | 2026-01-16/17 | −10,131 Pa |
| 7 | `data_new/ILO/LIU_YUE_DONG-0/before` | train | 2026-01-15 | −12,821 Pa |
| 8 | `data_new/ILO/LI_SHENG_WEN-0/before` | train | 2026-01-14 | −12,840 Pa |
| 9 | `data_new/ILO/ZHANG_YONG_SHENG-0/before` | **test** | 2026-01-20 | −12,439 Pa |

> 其中 1 与 5（`LIU_ZONG_YANG`、`ZUO_DAO_SHENG`）从未提交过重跑；其余 7 例 09-04 跑过一次但因
> 覆盖冲突失败。各例 `ascii/` 里现在都是 80 个旧文件 + 1 个新的 1120 帧。

### B 组：帧格式混杂，**必须重跑**（1 例）

| 完整路径 | 队列 | 现状 |
| --- | --- | --- |
| `data_new/AG/slow/ZHANG_ZHI_JUN` | train | `ascii/` 161 文件：81 个新节点值（1120 及全部奇数步）+ 80 个 1 月面中心值（偶数步 1122–1280） |

### C 组：解已正确，**只需清理目录**（1 例）

| 完整路径 | 队列 | 操作 |
| --- | --- | --- |
| `data_new/AG/slow/LI_BING_YI` | train | `ascii/` 242 文件：161 个新的 `LI_BING_YI-1120-<step>`（正确）+ 81 个旧的 `LI_BING_YI-<step>`。把 **81 个旧平名文件** 移出即可 |

### D 组：已知可接受，**可不处理**（4 例）

| 完整路径 | 队列 | 情况 |
| --- | --- | --- |
| `data_new/AAA/unruputer/HAN_JIAN_FU` | train | 首帧 1120 为独立补导出：节点顺序不同、壁面压力基准与其余 80 帧差 12.2 kPa。Δp 的空间标准差仅 19 Pa（对比壁面压力空间跨度 ~780 Pa），说明是**同一个解的不同压力基准**，不是旧解。WSS 本身连续正常，可用；壁面压力列需按帧统一或弃用 |
| `data_new/AG/slow/LIU_FENG` | train | 导出比解剖壁面少 1 个节点（1 面的自动命名 wall zone 未导出） |
| `data_new/AG/slow/WANG_KE` | train | 少 1 个节点（2 面的自动命名 zone） |
| `data_new/AG/slow/ZHANG_WEI_XIAN` | **test** | 少 4 个节点（共 19 面的两个自动命名 zone） |

### 重跑前必做的一步（A、B 组共 10 例）

```bash
cd /public/newhome/cy/Digital_twin/GNN
for c in \
  AG/slow/LIU_ZONG_YANG \
  AG/slow/ZHANG_ZHI_JUN \
  AAA/ruputer/MENG_GUANG_QIN \
  AAA/ruputer/WANG_FU_SHUN \
  AAA/ruputer/ZHOU_KE_XUN \
  AAA/ruputer/ZUO_DAO_SHENG \
  AAA/unruputer/CHEN_SHU_LIN \
  ILO/LIU_YUE_DONG-0/before \
  ILO/LI_SHENG_WEN-0/before \
  ILO/ZHANG_YONG_SHENG-0/before ; do
  mkdir -p "data_new/$c/ascii_superseded_20260904"
  mv "data_new/$c/ascii/"* "data_new/$c/ascii_superseded_20260904/" 2>/dev/null
done
```

C 组单独：

```bash
cd /public/newhome/cy/Digital_twin/GNN/data_new/AG/slow/LI_BING_YI
mkdir -p ascii_superseded_20260904
for s in $(seq 1120 1 1280); do mv "ascii/LI_BING_YI-$s" ascii_superseded_20260904/ 2>/dev/null; done
# 保留 ascii/LI_BING_YI-1120-<step>（161 个），移走 81 个旧平名文件
```

重跑后验证：`grep -c "already exists" <新 transcript>` 应为 **0**。

## 21. 2026-09-04 按用户要求执行的清理（为重跑做准备）

### 21.1 清空 9 例的 `ascii/` 与 `ascii_in/`

用户授权后执行，队列已确认为空。每例删除 81 + 81 个文件，**合计释放 79.0 GB**。
被删文件的 SHA256、大小、mtime 全部保留在 `audits/superseded_20260904/raw_hash/` 中。

| 路径 | ascii | ascii_in |
| --- | ---: | ---: |
| `data_new/AG/slow/LIU_ZONG_YANG` | 81 → 0 | 81 → 0 |
| `data_new/AAA/ruputer/MENG_GUANG_QIN` | 81 → 0 | 81 → 0 |
| `data_new/AAA/ruputer/WANG_FU_SHUN` | 81 → 0 | 81 → 0 |
| `data_new/AAA/ruputer/ZHOU_KE_XUN` | 81 → 0 | 81 → 0 |
| `data_new/AAA/ruputer/ZUO_DAO_SHENG` | 81 → 0 | 81 → 0 |
| `data_new/AAA/unruputer/CHEN_SHU_LIN` | 81 → 0 | 81 → 0 |
| `data_new/ILO/LIU_YUE_DONG-0/before` | 81 → 0 | 81 → 0 |
| `data_new/ILO/LI_SHENG_WEN-0/before` | 81 → 0 | 81 → 0 |
| `data_new/ILO/ZHANG_YONG_SHENG-0/before` | 81 → 0 | 81 → 0 |

`data_new/AG/slow/ZHANG_ZHI_JUN` 由用户自行清空（已核实 ascii 与 ascii_in 均为 0）。

操作记录：`audits/cleanup_for_rerun_20260904.json`。

### 21.2 `LI_BING_YI` 目录整理并改名对齐

| 步骤 | 结果 |
| --- | --- |
| 删除 81 个旧平名文件 `LI_BING_YI-<step>`（1 月旧解，0.15 GB） | 完成 |
| 80 个奇数步新文件移到 `data_new/AG/slow/LI_BING_YI/ascii_odd_steps_20260904/` | 完成（体域无对应帧，未删除，保留备查） |
| 81 个偶数步新文件 `LI_BING_YI-1120-<step>` → `LI_BING_YI-<step>` | 完成 |

最终 `ascii/` 恰好 81 个文件，步号严格等于 `1120..1280` 偶数步，命名统一为 `LI_BING_YI-N`。

**重新审计结果：全部 gate 通过。** 峰值帧 13,910 行，壁面与体域压力差中位 **0.04 Pa**，
WSS p50 3.70、p99 22.05、max 61.27 Pa，切向性 p95 0.028，无残留文件、无步号歧义。
操作记录：`audits/li_bing_yi_wall_cleanup_20260904.json`。

### 21.3 作废审计结果归档

上述 10 例（9 例清空 + `ZHANG_ZHI_JUN`）的 raw_hash / topology / solver_convergence / wall_wss
四类 JSON 已移入 `audits/superseded_20260904/`，附 README 说明它们描述的是已删除的数据。
这样做的另一个必要原因：审计脚本会跳过已存在的结果文件，不归档会导致重跑后审计被静默跳过。

当前正式审计集合：raw_hash 163、topology 162、solver 163、wall_wss 162。

### 21.4 重跑后的验收清单

1. 重跑 10 例（9 例 + `ZHANG_ZHI_JUN`），journal 保持 `dual-time-iterate 1280 20`，导出名默认；
2. 验证新 transcript 中 `grep -c "already exists"` **为 0**；
3. 验证 `ascii/` 与 `ascii_in/` 各 81 帧、步号 `1120..1280` 偶数步；
4. 重跑四项审计：`scratchpad/reaudit.sh <10 个病例>`；
5. 预期壁面 WSS 标签通过数由 158/172 升至 **170/172**（余 `HAN_JIAN_FU` 首帧与 3 例少 1–4 节点为已知可接受项）。

## 22. 2026-09-04 20:06 重跑批次提交（10 例）

`LI_BING_YI` 的 80 个奇数步备份已按用户要求删除（157 MB），`ascii/` 保持 81 个文件不变。

在每例目录下执行 `sbatch fluent.slurm`，10 例全部提交成功：

| 作业 | 病例 | 节点 | 起始状态 |
| ---: | --- | --- | --- |
| 12974 | `AG/slow/LIU_ZONG_YANG` | node02 | RUNNING |
| 12975 | `AG/slow/ZHANG_ZHI_JUN` | node01 | RUNNING |
| 12976 | `AAA/ruputer/MENG_GUANG_QIN` | node01 | RUNNING |
| 12977 | `AAA/ruputer/WANG_FU_SHUN` | node01 | RUNNING |
| ~~12978~~ → **12985** | `AAA/ruputer/ZHOU_KE_XUN` | ~~node01~~ → **node06** | 用户改节点后重新提交，RUNNING |
| 12979 | `AAA/ruputer/ZUO_DAO_SHENG` | node02 | RUNNING |
| 12980 | `AAA/unruputer/CHEN_SHU_LIN` | node02 | RUNNING |
| 12981 | `ILO/LIU_YUE_DONG-0/before` | node05 | RUNNING |
| 12982 | `ILO/LI_SHENG_WEN-0/before` | node05 | RUNNING |
| 12983 | `ILO/ZHANG_YONG_SHENG-0/before` | node05 | RUNNING |

**开跑 3 分钟后 9 例的 `already exists` 计数均为 0**，证明清空 `ascii/` 的做法有效，本轮壁面会真正写入。

### 22.1 `LIU_ZONG_YANG` 与 `ZUO_DAO_SHENG` 之前为何没跑：**从未提交，不是失败退出**

| 证据 | `AG/slow/LIU_ZONG_YANG` | `AAA/ruputer/ZUO_DAO_SHENG` |
| --- | --- | --- |
| 目录内历史 transcript | 仅 `Fluent_12859.out`（2026-09-03 02:25） | `Fluent_12813.out`（09-02 13:59）、`Fluent_14213.out`（01-28 05:45） |
| 09-04 的失败/错误文件 | 无 | 无 |
| `sacct` 09-04 作业清单（12938–12961） | 无对应作业 | 无对应作业 |
| `fluent.slurm` 最后修改 | 2026-09-01 02:59 | **2026-09-04 19:52**（提交前 14 分钟刚编辑过） |

结论：两例在 09-04 那批重跑中**根本没有被提交**，不存在"提交后失败退出"的情况。
`ZUO_DAO_SHENG` 的 `fluent.slurm` 在 19:52 刚被编辑，说明当时正在准备但尚未 `sbatch`。

唯一的模糊点：09-04 作业清单里有一个 `12955` 状态为 `CANCELLED`、`Start=None`（从未启动，
未分配节点），因为没启动所以目录里不会留下任何文件，无法反查它属于哪一例。从时间顺序
（12954 于 05:56 启动、12956 于 06:06 启动）和 `MENG_GUANG_QIN` 在 06:08–06:09 编译 UDF、
改写 `fluent.slurm` 却没有作业这一现象看，`12955` **很可能**是 `MENG_GUANG_QIN` 的首次提交被取消，
随后在 07:59 以 `12960` 重新提交。此推断无直接文件证据。

### 22.2 本轮验收要点

1. 结束后检查各例 transcript 的 `already exists` 是否仍为 0；
2. `ascii/` 与 `ascii_in/` 应各 81 帧、步号 `1120..1280` 偶数步；
3. 运行 `scratchpad/reaudit.sh` 对 10 例重跑四项审计；
4. 预期壁面 WSS 标签通过数升至 **170/172**。

### 22.3 `ZHOU_KE_XUN` 换节点重提（20:12）

`12978` 因排队等 node01 资源，用户把 `fluent.slurm` 的 `-w` 改为 **node06**（20:11:49 修改）。
执行 `scancel 12978`（状态 `CANCELLED`，从未启动、目录无残留），随后在该例目录重新
`sbatch fluent.slurm`，得到 **`12985`，已在 node06 运行**。至此 10 例全部在跑，无排队。
监控任务已同步替换为 `12985`。

## 23. 2026-09-04 重跑批次完成：壁面标签修复成功（10/10）

10 个作业 20:06 提交、23:1x 前全部 `COMPLETED`，**每例 `already exists` 计数均为 0**，
证实"重跑前清空 `ascii/`"是正确且必要的做法。

### 23.1 逐例核实：壁面与体域已同源

| 病例 | 壁面节点 | Δp 中位（1120 / peak / 1280） | 峰值帧 WSS 均值 / p99 / max (Pa) | 修复前 Δp |
| --- | ---: | --- | --- | ---: |
| `AG/slow/LIU_ZONG_YANG` | 12,138 | −0.012 / — / −0.013 | 0.67 / 5.8 / 13.7 | −13,536 |
| `AG/slow/ZHANG_ZHI_JUN` | 10,095 | −0.016 / — / — | 0.46 / 1.5 / 3.3 | 帧格式混杂 |
| `AAA/ruputer/MENG_GUANG_QIN` | 14,527 | −0.001 / −0.005 / −0.001 | 4.66 / 40.3 / 92.4 | −12,235 |
| `AAA/ruputer/WANG_FU_SHUN` | 95,287 | +0.004 / +0.029 / +0.004 | 5.03 / 40.5 / 184.5 | −11,658 |
| `AAA/ruputer/ZHOU_KE_XUN` | 79,783 | −0.001 / −0.006 / −0.001 | 4.84 / 58.6 / 209.1 | +7,576 |
| `AAA/ruputer/ZUO_DAO_SHENG` | 74,419 | 0.000 / −0.008 / −0.000 | 2.94 / 26.7 / 58.9 | −4,053 |
| `AAA/unruputer/CHEN_SHU_LIN` | 19,687 | +0.001 / −0.017 / +0.001 | 2.88 / 18.2 / 43.9 | −10,131 |
| `ILO/LIU_YUE_DONG-0/before` | 62,998 | +0.002 / −0.005 / +0.002 | 3.31 / 14.2 / 44.2 | −12,821 |
| `ILO/LI_SHENG_WEN-0/before` | 52,856 | −0.001 / −0.077 / −0.001 | 3.96 / 23.6 / 92.5 | −12,840 |
| `ILO/ZHANG_YONG_SHENG-0/before` | 60,329 | +0.002 / −0.033 / +0.002 | 2.24 / 8.9 / 25.7 | −12,439 |

压力差从数千至上万 Pa 降到 **|Δp| ≤ 0.08 Pa**，10 例壁面标签与体域已是同一次求解。
`ZHOU_KE_XUN` 的上尾从旧解的 p99 89.2 / max 320.1 降到 58.6 / 209.1（约 −35%），
方向与 A_udf 修复后流量下降 26% 吻合。

目录终检：10 例均为壁面 81 帧 + 体域 81 帧、步号严格等于 `1120..1280` 偶数步、
全部节点值、全部 09-04 同批写出、无奇数步残留（旧版清理脚本产生的残留由
`scratchpad/tidy_odd.sh` 补删：`ZHANG_ZHI_JUN` 80 个 113 MB、`LIU_ZONG_YANG` 80 个 136 MB）。

### 23.2 四项审计全部重做

| 审计 | 结果 |
| --- | --- |
| raw 内容 SHA256 | 10 例刷新，全队列 173 份 |
| 拓扑 anatomy-only | **10/10 通过**；全队列 **172/173**（仅已排除的 `YANG_BAO_KUI`） |
| 求解收敛 | 见下 |
| **壁面 WSS 标签** | **168/172 通过**（此前 157/172） |

本轮 10 例新解的收敛分层：`LIU_ZONG_YANG`、`ZHANG_ZHI_JUN`、`ZUO_DAO_SHENG` 达到 A 层（81/81）；
`MENG_GUANG_QIN`、`CHEN_SHU_LIN`、`LIU_YUE_DONG`、`LI_SHENG_WEN`、`ZHANG_YONG_SHENG` 为 B 层
（残差 3.6e-4–1.1e-3）；`WANG_FU_SHUN`（2.18e-3）与 `ZHOU_KE_XUN`（3.33e-3）为 C 层。
全队列分层：A 49 / B 86 / C 33 / D 4，与重跑前一致。

### 23.3 壁面 WSS 标签剩余 4 例未通过（均为已知可接受项）

| 病例 | 队列 | 判据 | 说明 |
| --- | --- | --- | --- |
| `AAA/unruputer/HAN_JIAN_FU` | train | `coordinates_identical_across_frames`、`same_solution_as_volume` | 首帧 1120 单独补导出：节点顺序不同、壁面压力基准差 12.2 kPa。Δp 空间标准差仅 19 Pa（压力空间跨度 ~780 Pa），是同一解的不同压力基准。**WSS 本身可用**，81 帧须按坐标/节点 ID 对齐，壁面压力列该帧单独处理 |
| `AG/slow/LIU_FENG` | train | `covers_anatomy_wall_exactly` | 导出少 1 节点（1 面的自动命名 wall zone） |
| `AG/slow/WANG_KE` | train | 同上 | 少 1 节点（2 面） |
| `AG/slow/ZHANG_WEI_XIAN` | test | 同上 | 少 4 节点（共 19 面） |

三例"少 1–4 节点"只需把标签点集定义为"导出壁面"而非"解剖壁面"即可，无需重跑。

### 23.4 当前 WSS 回归标签可用性

**168/172 例可直接使用**；加上 `HAN_JIAN_FU`（按坐标对齐后 WSS 可用）与三例节点数微差，
实质上 **172/172 的 WSS 幅值标签均可用**。使用要点不变：取 `wall-shear` 标量列，
不要用三分量模长；parser 需同时支持逗号与空格分隔。

## 24. 2026-09-05 求解质量分层定稿与 D 层重跑准备

### 24.1 "solution is converged" 标志是 case 设置产物，不是解质量

扫描全部 173 个 `.cas.gz` 的 `residuals/convergence-criterion-type`：

| 设置 | 例数 | 与标志的关系 |
| --- | ---: | --- |
| type 0（absolute，continuity 1e-3） | 96 | 49 例 81/81、47 例部分收敛 |
| type 3（**不做收敛检查**） | 77 | **77 例全部 0/81**，且无一例例外 |

type-3 病例无论残差降到多少都不会打印标志，每步必跑满 20 次上限。直接证据：`GUO_BAO_CHUN`
第 1280 步在第 20 次内迭代 continuity 已到 3.3e-4、`WANG_JIN_MING-0` 到 1.8e-4，仍无标志。
因此 §12.4 以"81/81 标志"定义的 A 层把设置差异混进了质量判断（A 层 38/49 为 AG，
只是 AG 病例多以 type 0 建案）。

### 24.2 冻结定义（残差口径，2026-09-05）

`solver_quality.tier` 只看 81 个导出步末次 continuity 残差的最大值：

| 层 | 定义 | formal 172 | train | test |
| --- | --- | ---: | ---: | ---: |
| A | max ≤ 1e-3 | 99 → **102**（09-06） | 76 → 79 | 23 |
| B | 1e-3 < max < 2e-3 | 35 | 30 | 5 |
| C | 2e-3 ≤ max < 5e-3 | 33 → 34（09-06） | 28 → 29 | 5 |
| D | max ≥ 5e-3 | 4 → **1**（09-06，仅 test `SUN_XU_XIA-1`） | 3 → 0 | 1 |
| unknown | 无 transcript（`AG/fast/ZHANG_XIU_ZHEN`） | 1 → **0**（09-06 重跑后为 C，§27.5） | 1 → 0 | 0 |

按队列：AAA 34/13/6/2，AG 38/14/23/0（+1 unknown），ILO 27/8/4/2。原 49 例标志 A 全部落在新 A 层内。
C 层仍偏 AG（train 28 例中 21 例），任何去 C 的敏感性都必须按队列分开报告。

实现：`wss_pinn/v4/audit_solver_convergence.py` 逐例写 `solver_quality`
（tier、continuity max/p95、标志计数、`convergence_criterion_type`、所用 `.cas`），
summary schema 升到 2 并含 `tier_counts` 与 `by_case`。标志计数与 criterion type 保留为 provenance，不再参与判层。

### 24.3 用法（用户 2026-09-05 拍板）

1. A/B 合并、同等对待；分层不作训练集过滤；
2. D 层 3 例训练病例（`DING_JUN_FENG`、`WANG_JIN_MING-0`、`GUO_BAO_CHUN`）以更高每步迭代上限重跑；
   test 的 `SUN_XU_XIA-1` 同批重跑，避免"含/不含"两版指标；
3. 评估按层与按队列分层报告；训练按层记录 data loss 与物理残差（batch 已带 case_ids）；
4. 敏感性只做一臂：train138 去 C∪D 对比全量，按队列报告；不做逐例残差加权；
5. `ZHOU_KE_XUN`（C 层，3.33e-3）并入分层规则，09-03 §5.2 单例项关闭；
6. 正式 manifest 字段：`solver_quality_tier`、`continuity_last_max/p95`、`flag_converged_export_steps`、
   `iteration_cap_hits`、`convergence_criterion_type`、transcript SHA256；`ZHANG_XIU_ZHEN` 09-06 重跑后有 transcript，为 C 层（§27.5），队列不再有 unknown。

### 24.4 D 层重跑：`dual-time-iterate 1280 20 → 1280 60` 足够，但另有两处必改

**迭代上限**。四例被截断步的 continuity 在后段按约 0.74×/次衰减（r20/r10 中位 0.05–0.21），
从截断值降到 1e-3 所需迭代估计：中位 23–36、p90 25–43、最大 51（`SUN_XU_XIA-1`）。60 有余量。
三例 type-3 病例没有提前停止，会每步跑满 60 次，残差可到 1e-6 量级；运行时间约为 1 月原跑的
3 倍（1.5 h → 约 4.5 h）。`DING_JUN_FENG` 为 type 0，收敛即停。**验收改用残差口径**：81 个导出步
continuity 末值全部 ≤ 1e-3（A 层）；三例 type-3 重跑后仍不会出现 "solution is converged"，属预期。

**路径**。三例 journal 的 `read-case` 与 `.cas.gz` 内嵌的自动导出路径都指向已不存在的旧根
`GNN/data/…`（`SUN_XU_XIA-1` 甚至是 `data/ILO/ILO/…`）；`WANG_JIN_MING-0` 已是 `data_new`。
不改 `.cas.gz`（避免 §14.3 的 provenance 漂移），改为临时脚手架
`GNN/data/AAA → data_new/AAA`、`GNN/data/ILO/ILO → data_new/ILO`，journal 的 `read-case` 直接改成
`data_new` 路径；批次结束后删除 `GNN/data`。

**旧导出与审计**。`ascii/`、`ascii_in/` 移入各例 `superseded_20260905/`（合计 28.8 GB，`/public` 余 3.8 T），
验收通过后再删；四例的 raw_hash / topology / solver / wall_wss JSON 移入 `audits/superseded_20260905/`。
Slurm 脚本不改：node01 三例 ×64 任务、node02 一例 ×40 任务，09-05 检查时均空闲。

准备脚本（含 `--dry-run`）：会话 scratchpad `prepare_d_rerun.sh`。
**2026-09-05 更新：用户决定四例由本人在 Fluent 中修改导出路径后自行提交**，上述脚手架方案不再使用；
`.cas.gz` 会被重存，§14.3 的 provenance 漂移表需在提交后补录四例。提交后按残差口径做四项审计验收。
**2026-09-06：四例已提交并完成，验收见 §27。**

### 24.5 第 2、3 题状态

用户 2026-09-05 明确第 2、3 题按建议执行，落地记录见 §25、§26。

## 25. 2026-09-05 第 2 题落地：BC 条件向量合同（`wss_pinn/v4/bc_contract.py`）

### 25.1 合同

| 项 | 冻结内容 |
| --- | --- |
| raw 向量（18 维） | `[A_in_mesh, inlet_area_ratio, (A_out, R1, R2, C)×4]`，`inlet_area_ratio = A_mesh/A_udf = Q_actual_peak/Q_nom_peak` |
| 比值通道变换 | `(ratio − 1)/0.1`，固定物理尺度，永不 z-score（stats 中 mean 0 / std 1） |
| log10 字段尺度 | train138 max-aware：`std = max(总体 std, 最大绝对偏差/6)`，训练集 \|z\| 天然 ≤ 6，无 whitelist |
| 近常量守卫 | 任一拟 z-score 字段 train std < 0.02 dex 时 builder 直接报错，只能改固定尺度或删除 |
| 版本校验 | stats 写 `contract = v4_bc_vector_2026-09-05`；`normalize_bc` 拒绝旧合同 stats，`transform_bc_raw` 拒绝 index 1 不在 [0.5, 2] 的旧 `Q_actual_peak` 向量 |

接入点：`build.py`（raw 向量、stats、Gate 改为 `bc_inlet_ratio_fixed_scale` / `bc_train_abs_z_within_6`）、
`build_centerline_v2.py`（逐例从 `a_in_mesh_m2/a_in_udf_m2/outlet_area_m2/rcr` 重建 `bc_vector_raw`；
`BC_Z_WHITELIST` 清空；新增 advisory `abs_z_gt_4`）、`data.py::_bc_transform → normalize_bc`（`evaluate.py` 经此生效）、
`config.py` 提示文本。模型输入维度仍为 18。`Q_actual_peak`、`U_c` 等物理量用途不变。

### 25.2 真实 train138 验证（per-case manifest；ZHOU/ZUO 按 09-02 修复取 ratio=1）

- 比值通道非零仅 4 例：`WANG_CHUN_MING` +0.09、`ZHANG_YAN_SHAN` −0.658、`LIU_YONG_LAN` −0.887、`CHEN_SHU_LIN` −1.122；
- z-score 字段最小 train std 0.109 dex（`log10_A_in`），远高于 0.02 守卫；
- max-aware 只对 `ZOU_LI_SHUN` 的 out-re R2/C 生效（尺度 ×1.08），其 \|z\| 恰为 6.0；全部 train \|z\| ≤ 6；test34 \|z\| 最大 4.56；
- 旧口径下 CHEN/LIU/ZHANG 仍为 −8.3/−6.5/−4.8σ，新口径下三例落在固定尺度 [−1.12, 0]，09-03 §5.1 的长尾复审关闭。

**Advisory \|z\|>4（不阻断，建议核对 UDF）**：`LIU_JUN_FENG` 四出口 R1 −4.7～−5.6、`WANG_CHUN_MING` 四出口 R1 −4.1～−4.9、
`MENG_GUANG_QIN` out-le R2 +4.4 / C −4.4、`ZOU_LI_SHUN` out-re R2 +6 / C −6、`WU_GUANG_CUN` 与 `ZHOU_KE_XUN` out-le 面积 −4.1、
`GUO_BAO_CHUN` out-li 面积 +4.1。前两例四个出口 R1 同向偏低，像是 UDF 单位差异，值得看一眼。

测试：`wss_pinn/tests/test_bc_contract.py` 4 项；既有 31 项 v4 测试通过。旧 staging bundle 的 stats 自此不能再被 `data.py` 读取，
与"旧 staging 已全面过期"一致。

### 25.3 LIU_JUN_FENG / WANG_CHUN_MING 低 R1 的物理影响评估（2026-09-05）

UDF 源码里两例四个出口的 `R1` 直接写为 7.8e3～1.25e4，同批 AG/fast 病例（如 LIU_LI_QUN）为 1.1e5～2.1e5；
`R1/(R1+R2)` 仅 0.5～0.7%，队列中位 11.9%（p5–p95 5.8～23.7%）。四出口 `R2·C` 均为 1.78 s，与队列一致。

- **速度 / WSS：可忽略。** 刚性域、入口流量给定，速度场只由出口分流决定；把 R1 换成队列中位后稳态分流最大改变
  0.9 个百分点（LIU_JUN_FENG）/ 0.7（WANG_CHUN_MING），四出口时间常数相同，动态分流亦不变。实测峰值帧 WSS
  p50/p99/max 在队列中的分位为 31/22/21%（LIU_JUN_FENG）和 80/45/44%（WANG_CHUN_MING），均在正常范围。
- **压力：小幅偏软，不是离群。** 出口压力的同相分量 `R1·ṁ(t)` 几乎消失，只剩顺应腔压力：76 例有完整
  压力 monitor 的队列中，两例入口均压 15,153 / 15,386 Pa（队列中位 15,606，p5 15,210）、出口脉压 2,701 / 2,567 Pa
  （队列中位 2,891，分位 17% / 0%）、出口压力峰值滞后 Q 峰 36 / 33 步（中位 31，分位 96% / 86%）。
  即均压低 1.5～3%、脉压低 7～11%、相位滞后多 10～25 ms。

结论：两例标签是合法 CFD 解，保留；不做特殊处理，manifest 记 `rcr_setup_note`。

**用户 2026-09-05 决定：后续正式训练不再显式输入 RCR 等求解前参数。** 由此：BC 18 维向量合同只服务 staging / v1.2 臂；
formal 条件输入冻结时以几何 + `inlet_area_ratio` 为准。需要正视的后果是瞬态压力标签的脉动幅值
（队列 p5–p95 2,458～5,088 Pa，±30%）由 R1、C 决定，几何无法解释，而均压跨队列几乎恒定（p5–p95 15,210～16,036 Pa，±2.6%）。
formal 的瞬态压力目标应按病例均值中心化或明确接受脉动分量的不可约误差；RCR 一致性残差随之不可用，与验收计划 §4.1 的
"解剖切面上隔离 distal RCR"一致。

**用户 2026-09-06 拍板：formal 瞬态压力监督目标按病例均值中心化**（与 steady 的 `case_strict_volume_mean_centered_pa` 同口径，
瞬态按每例 81 帧 blood-only 体域均值中心化，均值作为诊断量随 manifest 落库）；评估中的压力指标同样在中心化后报告；
RCR 一致性残差不启用。该合同随 Phase C 正式 builder 实现。

## 26. 2026-09-05 第 3 题落地：半径自适应 SG 窗、kink Gate 与辅助列

### 26.1 实现（`wss_pinn/v4/centerline_atlas.py`、`build_feature_atlas.py`）

- `smooth_and_differentiate_adaptive`：坐标的 SG 三阶窗长（mm）≈ 1.0 × 局部平滑半径，半窗夹在 [5, 30] 样本（11～61），
  段端受支撑限制退化到 11；半径本身仍用固定 SG11（避免循环依赖）；短段退化多项式不变。R ≤ 5.5 mm 时与旧算子逐位一致（差 1e-14）。
- 新列（auxiliary，不进首轮输入）：`sg_window_samples`、`curvature_times_radius`、`radius_over_case_median`；`map_points` 同步输出；
  provenance 记录策略与病例中位半径。
- kink Gate `detect_kinks`：`k > max(8×段中位数, 0.3 /mm)`，按 interior/junction/endpoint 分类，排除子段起点的重复样本；
  `no_interior_kinks` 进入逐例 gates，cohort summary（schema 2）汇总。
- 重建时按既有 summary 复用 source root，三例网格壁面重提病例不会被默认根覆盖。

### 26.2 173 例重建结果（旧产物移至 `atlas_fixed_sg11_superseded_20260905/`）

| 指标 | 固定 SG11（09-03） | 半径自适应（09-05） |
| --- | --- | --- |
| cohort 曲率 p50 / p99 / max | 0.0422 / 0.1743 / 0.7992 | 0.0358 / 0.1677 / 0.5517 |
| train138 p99 / max | 0.1779 / 0.7992 | 0.1724 / 0.5687 |
| interior kink | 15 个样本 / 6 例（§16.2） | **0 / 173** |
| review 队列 | LI_LAO_PING 0.799、LIU_FENG 0.552 | 仅 LIU_FENG 0.552（端点） |
| `k·R` p99 / max / >1 占比 | — | 1.12 / 6.19 / 1.5% |
| SG 窗 p50 / p99 / >11 占比 | 11 | 13 / 53 / 55% |

逐例 max 变化中位 0，最大上升 +0.017（18 例，窗切换处），最大下降 −0.578（LI_LAO_PING）。gate failures 0。

**LI_LAO_PING**：trunk max 0.799→0.193（全例 max 0.221），瘤腔内 max 0.799→0.152，`k·R` max 17.4→3.24，瘤腔窗 25～45 样本，
瘤腔内拟合残差最大 1.1 mm（R=21.8 mm）。图：`audits/atlas_review/LI_LAO_PING_sac_curvature_profile_adaptive_20260905.png`、
`AAA__ruputer__LI_LAO_PING_adaptive_overlay_20260905.png`。瘤腔内 `k·R` 仍在 1～3，是"瘤腔不是管道"的残余，由 auxiliary 列承载。

**瘤腔壁 abscissa 单调性**（11,404 壁面节点，`LI_LAO_PING_sac_abscissa_audit_20260905.json`）：与流向轴 Spearman 0.856→0.874；
3 mm 邻居对 Δs>10 mm 占比 0.92%→0.75%；Δs p99 10.0→8.5 mm。结论：不是主要问题，记录不处理。

### 26.3 剩余端部命中（窗均为 11，支撑受限；属 §10 第 6 问，待决）

| 病例 | 位置 | 样本 | k max |
| --- | --- | ---: | ---: |
| `AAA/unruputer/ZHANG_XUN_LIAN` | trunk 根端 s=0～1 mm | 3 | 0.417 |
| `AG/slow/LIU_FENG` | 出口端 | 1 | 0.552 |
| `ILO/SUN_XU_XIA-1/before` | 出口端 | 4 | 0.402 |
| `ILO/HOU_SHI_GUO-0/before` | 子段起始 0.5～1 mm（分叉） | 2 | 0.401 |

测试：`test_centerline_atlas.py` 新增 3 项（常半径等价、变半径圆弧、宽腔拐角与 Gate），7/7 通过。
正式 bundle builder 仍用 `geometry_v2`，atlas 在 Phase C 接入，与此前状态一致。

### 26.4 2026-09-06 端点与分叉 1R 规则落地（用户拍板"1R 采用"）

**被否决的第一版**：坐标外推（用 1R～3R 带的二次拟合改写 1R 内的原始样本）。AAA 分叉的内切球半径达 20～33 mm，
子支起点区因此长到 25～29 mm，外推位移全队列中位 7 mm、最大 36 mm（`MENG_GUANG_QIN`），会改写真实几何，不能用。

**冻结规则（`hold_end_features`）**：坐标不动；只在开口端点（trunk 根端、四个出口端）和子支起点作用，母支的分叉端不动。
区宽 = max(6 样本, rint(min(R_end, 段中位 R, 12 mm) / 0.5 mm))；区内切线取内侧 1R～2R 带的切线均值，曲率取该带中位数。
新列 `end_zone`，provenance 逐段记录区宽、带宽、改前区内最大曲率与持值。`build_feature_atlas(end_hold=False)` 可复现改前结果。

| 端类型 | 数量 | 区宽 mm p50 / p90 / max | 曲率改变 p50 / p90 / max |
| --- | ---: | --- | --- |
| trunk 根端（入口） | 173 | 11.0 / 12.0 / 12.0 | 0.057 / 0.140 / 0.379 |
| 子支起点（分叉） | 1,036 | 4.0 / 7.5 / 12.0 | 0.067 / 0.166 / 0.391 |
| 出口端 | 692 | 3.0 / 4.5 / 8.0 | 0.040 / 0.112 / 0.522 |

逐例区内样本占比中位 8.6%，最大 14.4%；改前区内最大曲率超过 0.3 /mm 的端占 1.6%。

**重建后全队列**：曲率 p50/p99/max 0.0354 / 0.1570 / 0.4448（此前 0.0358 / 0.1677 / 0.5517），train138 p99/max 0.1567 / 0.4448；
`k·R` p99/max 0.99 / 5.25；kink Gate interior/junction/endpoint 全为 0；review 队列空；峰值位置 endpoint 0 / junction 32 / interior 141。
原先四例：ZHANG_XUN_LIAN 0.417→0.288、LIU_FENG 0.552→0.259、SUN_XU_XIA-1 0.402→0.301、HOU_SHI_GUO-0 0.401→0.222。
图：`audits/atlas_review/end_zone_profiles_20260906.png`。测试 8/8 通过。

## 27. 2026-09-06 D 层四例重算验收

作业：13035 `DING_JUN_FENG`（node01，1 h 34 min）、13036 `GUO_BAO_CHUN`（node01，4 h 54 min）、
13037 `WANG_JIN_MING-0/before`（node02，5 h 03 min）、13039 `SUN_XU_XIA-1/before`（node01，4 h 11 min）。
用户在 Fluent 中改导出路径后重存 `.cas.gz` 并自行提交；journal 均为 `dual-time-iterate 1280 60`，`read-case` 指向 `data_new`。

### 27.1 目录与 provenance

- 四例 transcript 中 `already exists` 均为 0；`ascii/`、`ascii_in/` 各 81 帧，步号 1120..1280 偶数步，无奇数步、无根目录残留，
  全部于 09-05 21:45～09-06 01:48 写出；
- `.cas.gz` 四例 SHA 全部改变（大小仅变 14～703 B）。拓扑审计比对旧记录：cell/node/face 数、六个 cell zone 计数、
  解剖体积完全一致，只是导出路径改写；§14.3 漂移表已追加四例；
- UDF 源码 SHA 未变；旧导出已被用户删除，其 SHA 保留在 `audits/superseded_20260906/raw_hash/`；
  raw_hash 173/173 刷新，四例新导出合计 30.6 GB。

### 27.2 四项审计

| 病例 | 拓扑 | 求解收敛（残差口径） | 壁面 WSS | 壁面-体域 Δp 中位 |
| --- | --- | --- | --- | ---: |
| `DING_JUN_FENG` | 通过 | **A**：81/81 标志，cont max 1.0e-3，中位 15 次即停 | 通过 | −0.010 Pa |
| `GUO_BAO_CHUN` | 通过 | **A**：cont p95/max 1.0e-6 / 2.4e-6（type 3，每步跑满 60） | 通过 | −0.002 Pa |
| `WANG_JIN_MING-0` | 通过 | **A**：9.6e-5 / 1.1e-4 | 通过 | −0.074 Pa |
| `SUN_XU_XIA-1`（test） | 通过 | **仍为 D**：p50 1.4e-7，但 p95/max 4.1e-3 / 5.5e-3 | 通过 | −0.089 Pa |

全队列分层更新：formal 172 = A 102 / B 35 / C 33 / D 1（仅 test 的 `SUN_XU_XIA-1`）/ unknown 1；
壁面 WSS 168/172 通过，未过的仍是 §23.3 的四个已知可接受项。

### 27.3 `SUN_XU_XIA-1`：收缩峰相位锁定的停滞，加迭代无效

- 81 个导出步中 16 步末值 >1e-3，全部落在 t=0.18～0.30 s（步 1156～1188，收缩峰附近）；其余相位在 60 次内降到 1e-5～1e-8；
- 停滞步的内迭代轨迹是平台甚至缓升：步 1162 从第 30 次的 4.6e-3 爬到第 60 次的 5.5e-3；步 1164 第 32 次最低 4.9e-3 后回到 5.5e-3；速度残差 ≤2e-6；
- 8 个周期每周期都有 27～33 步 >1e-3，相位一致；与 20 次旧跑相比 79/81 导出步残差下降，但平台值不变。

结论：这是该例在收缩峰的求解器停滞（ILO 狭窄，峰值帧 WSS p99 80 Pa / max 275 Pa），不是迭代数不够；
要压下去得改松弛因子或压力-速度耦合，即偏离全队列统一设置。它是 test 例，按 §24.3 保留在 test34 并标 D，指标报含/不含两版。

### 27.4 重要旁证：重跑前后 WSS 标签几乎不变

| 病例 | 峰值帧 WSS p50 / p99 / max（20 次 → 60 次） | 壁面均压 Pa |
| --- | --- | --- |
| `DING_JUN_FENG` | 3.25 / 75.5 / 263.2 → 3.27 / 75.0 / 262.4 | 15,731 → 15,726 |
| `GUO_BAO_CHUN` | 0.51 / 10.0 / 59.2 → 0.51 / 10.0 / 59.0 | 13,438 → 13,441 |
| `WANG_JIN_MING-0` | 2.64 / 106.6 / 294.7 → 2.63 / 106.9 / 294.8 | 17,749 → 17,768 |
| `SUN_XU_XIA-1` | 4.14 / 79.5 / 273.5 → 4.16 / 80.1 / 275.4 | 18,129 → 18,143 |

continuity 残差从 5e-3～7e-3 降到 1e-6～1e-3，WSS 三个统计量的变化都在 1% 以内。§17.6"D 层上尾显著异常源于收敛不足"
的推断不成立：四例的高 WSS 上尾是真实狭窄血流。这反过来支持 §24.3 的用法：分层不作过滤，重跑的价值在 provenance
而非标签修正；`SUN_XU_XIA-1` 留在 test 也不会因残差污染标签。

### 27.5 `AG/fast/ZHANG_XIU_ZHEN`：补 transcript 重跑（作业 13047，node02，1 h 49 min）

用户在 Fluent 中改导出路径后重存 `.cas.gz` 并提交，journal 保持队列默认 `dual-time-iterate 1280 20`。

- 目录：`already exists` 0；`ascii/`、`ascii_in/` 各 81 帧，1120..1280 偶数步，09-06 13:28～13:42 写出，无残留；
- `.cas.gz` SHA 改变但 cell/node/face、六个 zone 计数、解剖体积与旧网格一致；UDF 未变；raw_hash 已刷新；
- 四项审计：拓扑通过；求解收敛 type 0，53/81 标志、40 步触顶，continuity p50/p95/max 9.5e-4 / 2.1e-3 / 2.3e-3，**C 层**；
  壁面 WSS 通过，壁面-体域 Δp 中位 −0.10 Pa；
- 峰值帧 WSS p50/p99/max 2.05 / 12.6 / 28.4 Pa、壁面均压 15,461 Pa，与 1 月旧导出的审计值**逐位相同**，
  说明旧导出确实出自同一设置，只是当时的 transcript 没有保存。

至此 173 例（formal 172）全部有 transcript：A 102 / B 35 / C 34 / D 1（test `SUN_XU_XIA-1`），无 unknown。

