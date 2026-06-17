# 任务E：血流动力学三维可视化执行清单

> 上位文档：[实验设计总纲](../../实验设计总纲.md) | 相关文档：[任务A](../任务A/01-V1路线/任务A_V1实验清单.md) / [任务B](../任务B/任务B指标计算规范.md) / [任务E论文可视化规范](./任务E论文可视化规范.md)
>
> **执行优先级**：任务 E 排在 A → B → C → D 全部之后，当前阶段的核心目标是**把文档框架和技术路线弄清楚**，不急于编写代码。

---

## 1. 文档定位与核心原则

任务 E 的目标是为论文和答辩提供血流动力学的三维科研可视化证据。它不是一个独立的科学贡献，而是**为任务 A–D 的结论提供视觉层面的支撑**。

### 1.1 三层证据架构

| 层级 | 名称 | 证明什么 | 数据来源 |
|------|------|----------|----------|
| **层 A** | CFD 真值可视化 | 原始 CFD 场本身合理、可信 | 壁面 `ascii/` + 内部 `ascii_in/` + STL |
| **层 B** | 模型直接输出对比 | 模型预测的物理量在空间上对齐 CFD 真值 | `predictions_test/*.pt`（`u/v/w/p`） |
| **层 C** | 后处理血流动力学指标对比 | 由预测场计算的 TAWSS/OSI/RRT 仍与 CFD 一致 | 依赖任务 B 后处理管线 |

**硬原则**：
- 层 A 不依赖任何模型输出，可独立执行
- 层 B 只验证模型**直接输出量**，不混入后处理误差
- 层 C 只在任务 B 管线就绪后进入，不允许用少量帧伪装周期结果
- 三层图件不允许混在同一脚本里

### 1.2 WSS 来源的两条路线

当前模型输出为 `["u", "v", "w", "p"]`，壁面 WSS 不是直接输出量。后续有两条路线，在文档中分别标记其依赖：

| 路线 | 说明 | 状态 | 影响范围 |
|------|------|------|----------|
| **路线 α：后处理计算 WSS** | 从预测的 `u/v/w` 通过壁面法向速度梯度近似计算 WSS | 依赖任务 B 最小可用 WSS 管线 | 层 B 壁面图（间接）、层 C 全部 |
| **路线 β：多任务直接输出 WSS** | 扩展模型输出为 `["u","v","w","p","wss"]`，直接监督壁面剪切力 | 后续规划，当前未实现 | 层 B 壁面图（直接） |

> 当路线 β 落地时，层 B 壁面主图可直接使用 `y_pred` 中的 WSS 通道；在此之前，层 B 壁面图只能走路线 α 或标记为受阻。

---

## 2. 基准病例与数据现状

### 2.1 基准病例选定

**单病例基准**：`CHEN_SHI_MING`（`AG/fast` 数据集，`split_AG_v1` 测试集成员）

选择理由：
- 已在测试集中，有完整的模型预测产物
- 数据目录结构完整：`ascii/` + `ascii_in/` + STL + `processed/`
- 有 81 帧时间步（step 1120–1280），覆盖完整心动周期

### 2.2 CFD 原始数据

| 数据 | 路径 | 列名 | 节点数 | 坐标单位 |
|------|------|------|--------|----------|
| 壁面点云 | `data_new/AG/fast/CHEN_SHI_MING/ascii/CHEN_SHI_MING-{step}` | nodenumber, x, y, z, pressure, wall-shear, x-wall-shear, y-wall-shear, z-wall-shear | ~9,338 | m |
| 内部点云 | `data_new/AG/fast/CHEN_SHI_MING/ascii_in/CHEN_SHI_MING-{step}` | cellnumber, x, y, z, pressure, velocity-magnitude, x-velocity, y-velocity, z-velocity | ~575,210 | m |
| 表面几何 | `data_new/AG/fast/CHEN_SHI_MING/CHEN_SHI_MING.stl` | 三角面片 | — | mm |
| 时间帧数 | 81（step 1120–1280，步长 2） | — | — | — |

### 2.3 模型预测产物

| 项目 | 值 |
|------|-----|
| 当前最优 run | `A-Opt-03`（`field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_split_AG_v1_seed1_20260327_215621`） |
| manifest 路径 | `outputs/field/<run_dir>/predictions_test/manifest.json` |
| CHEN_SHI_MING 预测帧数 | 81 帧 |
| `.pt` 文件字段 | `sample_id`, `case_name`, `graph_path`, `node_feature_names`, `target_names`, `wall_mask`, `time_value`, `x`, `global_cond`, `edge_index`, `y_true`, `y_pred` |
| `target_names` | `["u", "v", "w", "p"]` |
| `node_feature_names` | `["x", "y", "z", "Abscissa", "NormRadius", "Curvature", "Tangent_X", "Tangent_Y", "Tangent_Z", "is_wall"]` |
| 每帧节点数 | 15,000（壁面 ~9,338 + 内部 ~5,662，经采样） |
| `wall_mask` | `True` 的数量与 `is_wall==1` 一致 |

### 2.4 坐标系与逆变换

pipeline 处理链：`原始 m → ×1000 转 mm → 中心化 + PCA 对齐 + [-1,1] 缩放`

逆变换参数已保存于：

```
data_new/AG/fast/CHEN_SHI_MING/processed/coord_normalized/transform_params.json
```

包含 `centroid`、`rotation_matrix`、`scale_factor`，可通过 `pipeline.coord_normalize.inverse_transform()` 还原到原始 mm 坐标系。

**可视化时的坐标选择策略**：
- 层 A（CFD 真值图）：直接使用原始 ASCII 坐标（m 或 ×1000 转 mm），与 STL 对齐
- 层 B（模型对比图）：`.pt` 中的坐标是归一化后的，需要逆变换回 mm 才能与 STL 对齐；或直接在归一化空间中对比（无需 STL 叠加时）
- 层 C（后处理指标图）：壁面指标映射到 STL 表面，必须在 mm 空间

### 2.5 关键坐标约束

| 约束 | 说明 |
|------|------|
| STL 单位 mm，ASCII 单位 m | 可视化前必须统一到同一坐标系（推荐 mm） |
| `ascii_in` 内部文件可能含重复表头 | 读取时必须检测并过滤 |
| 速度场经过 PCA 旋转 | 预测 `.pt` 中的 `y_true/y_pred` 是旋转后的速度，做 WSS 计算或流线时需注意坐标系一致性 |

### 2.6 已知问题：预测文件命名冲突

`predict_field.py` 按 `sample_id`（如 `result_features_merged-1120.pt`）命名输出文件，不含病例前缀。当多个测试病例存在同名时间步时，后写入的会覆盖先写入的。

**影响**：当前目录下可能只保留了最后一个病例的文件。
**解决方案**：后续需修改 `predict_field.py`，在文件名中加入 `case_name` 前缀，或按病例建子目录。该修复应在任务 E 正式取数前完成。

---

## 3. 运行环境

| 环境 | 用途 | 关键依赖 |
|------|------|----------|
| `GNN` | 模型训练、预测、数据读取、出图脚本主入口 | PyTorch, torch_geometric, matplotlib, numpy, pandas |
| `GNN_vmtk` | 三维渲染、STL 操作、体场插值 | VTK, PyVista, VMTK |

**规则**：
- 任务 E 脚本主入口默认在 `GNN` 环境运行
- 需要 VTK/PyVista 的三维渲染子步骤，通过子进程调用 `GNN_vmtk` 环境的解释器，或在脚本内部注明需要切换环境
- 不引入第三个环境

---

## 4. 技术架构

```
层 A：CFD 真值可视化
  壁面 ascii ──×1000──▶ mm 空间 ──映射到 STL──▶ 壁面渲染 ──▶ WSS/压力真值图
  内部 ascii_in ──×1000──▶ mm 空间 ──Delaunay/RBF 插值──▶ 切面云图 ──▶ |v|/p 真值图
  内部 ascii_in ──体场重建──▶ 三角化 + 流线积分 ──▶ 流线图

层 B：模型直接输出对比（当前 u/v/w/p）
  predictions_test/*.pt ──读取 y_true/y_pred──▶
    1) 内部切面对比（|v|/p）：归一化空间或逆变换到 mm 空间
    2) 壁面直接量对比：当前可做 p（壁面压力），WSS 需路线 α 或 β 就绪

层 C：后处理血流动力学指标对比（依赖任务 B）
  多时间步预测场 ──任务 B WSS 计算管线──▶ WSS_time ──周期聚合──▶ TAWSS / OSI / RRT
  与 CFD 对应指标做区域级/病例级/空间级对比
```

### 4.1 层 A 核心技术点

**壁面真值图**：
- 读取 `ascii/` 壁面数据（m 坐标 × 1000 转 mm）
- 已有 `wall-shear`（WSS 标量）和 `x/y/z-wall-shear`（WSS 分量），无需计算
- 映射到 STL 表面方式：最近邻投影（壁面点云到 STL 顶点）
- 输出：三视角壁面 WSS/压力渲染图

**内部真值切面图**：
- 读取 `ascii_in/` 内部数据（m 坐标 × 1000 转 mm）
- 切面选择：基于血管中心线（`centerline/` 目录）确定解剖切面，不使用全局固定 z 平面
- 从非结构化点云生成切面数据：优先 Delaunay 三角化 + 重心插值；备选 RBF 径向基函数插值
- 输出：`velocity-magnitude` / `pressure` 切面云图

**流线真值图**：
- 技术路线：先对 `ascii_in/` 点云做 Delaunay 三维三角化（或 VTK `vtkDelaunay3D`）构建非结构化体网格，再用 VTK 的 `vtkStreamTracer` 做流线积分
- 备选方案：使用 PyVista 封装的流线追踪，或导入 Fluent `.cas` 文件获取原始网格拓扑后直接在结构化网格上积分
- 播种策略：固定入口截面播种，播种点数和间距在文档中记录
- 积分参数：四阶 Runge-Kutta，步长与终止条件固定后不再修改
- **风险提示**：从 ~575K 非结构化点云做体场重建是计算密集操作，需在集群上运行；如果 Delaunay 三角化质量不佳，可考虑使用工程软件（Fluent/ParaView）预先导出带网格拓扑的体场数据

### 4.2 层 B 核心技术点

**数据源**：`predictions_test/*.pt` 中的 `y_true` 和 `y_pred`

**当前可做的图**：

| 图件 | 数据列 | 可行性 |
|------|--------|--------|
| 内部 `\|v\|` 切面 CFD vs Pred | `y_true[:, :3]` → 计算模长 vs `y_pred[:, :3]` → 计算模长 | ✅ 可做 |
| 内部 `p` 切面 CFD vs Pred | `y_true[:, 3]` vs `y_pred[:, 3]` | ✅ 可做 |
| 壁面 `p` 对比 | `y_true[wall_mask, 3]` vs `y_pred[wall_mask, 3]` | ✅ 可做 |
| 壁面 `WSS` 直接输出对比 | 需要 `target_names` 包含 `wss` | ❌ 当前受阻（等待路线 β） |
| 壁面 `WSS` 后处理对比 | 从预测 `u/v/w` 计算 WSS | ⏳ 依赖任务 B 最小 WSS 管线 |

**坐标处理**：
- 若在归一化空间做对比（不叠加 STL），可直接使用 `.pt` 中的坐标
- 若需映射到 STL 表面，先用 `inverse_transform()` 还原到 mm 空间，再做最近邻投影

### 4.3 层 C 核心技术点

完全依赖任务 B。需要：
1. WSS 计算管线就绪（路线 α：从 `u/v/w` + 壁面法向 + 图邻域近似壁面速度梯度）
2. 完整心动周期的多帧预测
3. TAWSS/OSI/RRT 的离散时间积分实现
4. 区域定义（AAA 主体、左右髂动脉等）

**在任务 B 就绪前，层 C 全部冻结**。

---

## 5. 目录结构规划

```
viz/                              # 可视化模块（待实现）
├── __init__.py
├── io_utils.py                   # 读 ASCII / STL / .pt / transform_params
├── coord_utils.py                # 逆变换、mm/m 统一、STL 对齐
├── colormaps.py                  # 统一 colormap 定义与值域管理
├── wall_surface.py               # 层 A + 层 B 壁面渲染
├── interior_slice.py             # 层 A + 层 B 内部切面
├── streamlines.py                # 层 A 流线图（VTK/PyVista，需 GNN_vmtk）
├── multitask_panel.py            # 层 B 多面板对比图
└── hemo_cycle_panel.py           # 层 C 后处理指标对比图

outputs/viz/                      # 输出目录
├── wall_truth/                   # 层 A 壁面真值图
├── interior_truth/               # 层 A 内部真值切面图
├── streamlines_truth/            # 层 A 流线图
├── multitask_compare/            # 层 B 模型对比图
└── hemo_compare/                 # 层 C 后处理指标图
```

---

## 6. 执行阶段与验收标准

> 状态标记：`⬜ 待启动` | `🔄 进行中` | `✅ 已完成` | `❌ 受阻/跳过`
>
> **任务 E 整体处于规划阶段**，代码实现在任务 A–D 主线完成后启动。

### 阶段 0：接口体检与依赖确认

**状态**：⬜

**目标**：确认所有数据源可读、接口字段正确、工具链可用。

| 编号 | 任务 | 依赖 | 验收标准 |
|------|------|------|----------|
| E-0-01 | 确认 `CHEN_SHI_MING` 在 `split_AG_v1` 测试集中 | 无 | ✅ 已确认（2026-03-30） |
| E-0-02 | 读取壁面 `ascii/CHEN_SHI_MING-1120`，打印 bbox、列名、节点数 | 无 | bbox 合理，列名含 `wall-shear` |
| E-0-03 | 读取内部 `ascii_in/CHEN_SHI_MING-1120`，过滤重复表头，打印 bbox | 无 | 无 NaN，节点数 ~575K |
| E-0-04 | 读取 STL，打印 bbox，确认与 ASCII ×1000 后的 mm 坐标可对齐 | 无 | bbox 重合 |
| E-0-05 | 读取 `transform_params.json`，验证逆变换往返误差 < 1e-6 mm | 无 | 往返误差可忽略 |
| E-0-06 | 读取 `predictions_test/manifest.json`，筛选 CHEN_SHI_MING 条目 | 预测文件命名冲突修复 | 81 帧全部可读 |
| E-0-07 | 加载一个 `.pt` 文件，打印所有字段 shape 和 dtype | 同上 | 字段完整 |
| E-0-08 | 确认 `target_names`，记录当前属于路线 α 还是路线 β | 无 | 明确写入日志 |
| E-0-09 | 在 `GNN_vmtk` 中验证 `import pyvista; import vtk` 可用 | 无 | 无 ImportError |
| E-0-10 | 在 `GNN` 中验证 `matplotlib`, `scipy.spatial.Delaunay` 可用 | 无 | 无 ImportError |

**前置修复**：
- [ ] `predict_field.py` 输出文件名加入病例前缀或按病例建子目录，解决命名冲突

---

### 阶段 1：CFD 真值图建立

**状态**：⬜

**目标**：建立高质量 CFD 真值图，作为所有后续对比的底板。

#### 子阶段 1A：壁面真值图

| 编号 | 任务 | 验收标准 |
|------|------|----------|
| E-1A-01 | 实现 `viz/io_utils.py`：读 ASCII 壁面、内部、STL，坐标统一到 mm | 三种数据源均可正确加载 |
| E-1A-02 | 壁面点云映射到 STL 表面（最近邻），统计映射距离分布 | 95% 映射距离 < 1mm |
| E-1A-03 | 生成 `wall-shear` 三视角真值图 | 低 WSS 区域与解剖结构对应合理 |
| E-1A-04 | 生成 `pressure` 三视角真值图 | 压力梯度方向符合物理预期 |
| E-1A-05 | 固定视角（正视/侧视/后视）和 colorbar 值域 | 记录到配置文件 |

#### 子阶段 1B：内部真值切面图

| 编号 | 任务 | 验收标准 |
|------|------|----------|
| E-1B-01 | 基于 `centerline/` 中心线数据确定切面位置和法向量 | 切面具有解剖意义（入口段/瘤体中央/分叉处） |
| E-1B-02 | 对 `ascii_in/` 非结构化点云做 Delaunay 三角化或 RBF 插值，提取切面数据 | 切面数据无明显插值伪影 |
| E-1B-03 | 生成 `velocity-magnitude` 三切面真值图 | 主流区、回流区可辨识 |
| E-1B-04 | 生成 `pressure` 三切面真值图 | 压力梯度合理 |
| E-1B-05 | 固定切面编号、位置参数、插值方法 | 记录到配置文件 |

#### 子阶段 1C：流线真值图

| 编号 | 任务 | 验收标准 |
|------|------|----------|
| E-1C-01 | 评估体场重建方案：(a) VTK Delaunay3D 三角化 (b) 导入 Fluent `.cas` 获取原始网格 (c) ParaView 预处理导出 | 选定方案并记录理由 |
| E-1C-02 | 如选三角化方案，验证体网格质量（最小角、体积比） | 无退化四面体 |
| E-1C-03 | 固定入口播种规则、积分步长、终止条件 | 参数写入配置 |
| E-1C-04 | 生成单帧流线真值图（速度模长着色） | 主射流、回流结构清晰 |

**验收标准（阶段 1 整体）**：
- 真值图能独立回答"CFD 场的空间分布合不合理"，不依赖预测结果
- 切面与流线选择规则可复现、可解释
- 所有坐标单位、视角、colorbar 范围均已固定

---

### 阶段 2：模型直接输出对比图

**状态**：⬜

**目标**：对模型**直接输出量**做空间级对比，量化预测精度。

| 编号 | 任务 | 依赖 | 验收标准 |
|------|------|------|----------|
| E-2-01 | 实现 `viz/multitask_panel.py` 框架 | 阶段 1 完成 | 可从 `.pt` 文件生成对比面板 |
| E-2-02 | 生成壁面 `p`（压力）CFD \| Pred \| Error 三列图 | 无 | 左列/中列同值域 |
| E-2-03 | 生成内部 `\|v\|` 切面 CFD \| Pred \| Error 三列图 | 无 | 切面位置与阶段 1 一致 |
| E-2-04 | 生成内部 `p` 切面 CFD \| Pred \| Error 三列图 | 无 | 同上 |
| E-2-05 | 壁面 WSS 对比图（路线 α：后处理计算） | 任务 B 最小 WSS 管线 | WSS 计算公式与任务 B 完全一致 |
| E-2-06 | 壁面 WSS 对比图（路线 β：直接输出） | 多任务头扩展完成 | `target_names` 含 `wss` |
| E-2-07 | 图上标注逐区域 `RMSE / R²` | 无 | 数值与 `regional_eval` 一致 |

**误差图定义（固定）**：
- 绝对误差 `|y_true - y_pred|`：使用顺序 colormap（`hot_r`），值域从 0 开始
- 有符号误差 `y_true - y_pred`：使用零中心 colormap（`bwr`），值域关于 0 对称
- 同一图中不混用两种误差定义

---

### 阶段 3：周期血流动力学指标对比图

**状态**：⬜（冻结，等待任务 B）

**硬前提**：
- 任务 B 的 WSS/TAWSS/OSI/RRT 计算管线通过验收
- 完整 81 帧周期预测产物可用
- 区域定义（AAA 主体、左右髂动脉）已固定

| 编号 | 任务 | 依赖 | 验收标准 |
|------|------|------|----------|
| E-3-01 | 对接任务 B 的统一 hemo 后处理函数 | 任务 B | 公式版本完全一致 |
| E-3-02 | 固定时间步集合与周期排序规则 | 无 | 81 帧覆盖完整周期 |
| E-3-03 | 生成 CFD `TAWSS / OSI / RRT` 壁面真值图 | E-3-01 | 分布符合文献预期 |
| E-3-04 | 生成 Prediction `TAWSS / OSI / RRT` 壁面对比图 | E-3-01 | 与 E-3-03 同值域 |
| E-3-05 | 输出区域级与病例级汇总统计 | 无 | 含 Pearson/Spearman 相关 |

---

### 阶段 4：动画与汇报增强（可选）

**状态**：⬜

| 编号 | 任务 | 验收标准 |
|------|------|----------|
| E-4-01 | 壁面 WSS 随时间变化动画 | 帧间平滑 |
| E-4-02 | 内部 `\|v\|` 切面随时间变化动画 | 帧间平滑 |
| E-4-03 | 叠加心动周期时间标签 | 标签正确 |

---

## 7. 执行优先级总路线

```
任务 A / B / C / D 完成
         │
         ▼
阶段 0（接口体检）── 修复 predict_field 命名冲突
         │
         ▼
阶段 1A（壁面真值图）──▶ 阶段 1B（内部切面图）──▶ 阶段 1C（流线图，技术风险最高）
         │
         ▼
阶段 2（模型直接输出对比）── 先做 u/v/w/p ──▶ WSS 对比等路线 α 或 β 就绪
         │
         ▼
阶段 3（周期指标对比）── 等任务 B 管线
         │
         ▼
阶段 4（动画，可选）
```

**论文图件价值排序（高 → 低）**：
1. 壁面 WSS 对比图（层 B）—— 审稿人最关注
2. 壁面真值图（层 A）—— 证明 CFD 本身合理
3. 内部 `|v|/p` 切面对比图（层 B）—— 内部场精度证据
4. TAWSS/OSI/RRT 对比图（层 C）—— 端到端闭环证据
5. 内部切面真值图（层 A）—— 辅助说明
6. 流线图（层 A）—— 视觉补充，论文定量价值较低
7. 动画（层 D）—— 仅汇报用

---

## 8. 与任务 A–D 的依赖关系

| 任务 | 对任务 E 的影响 |
|------|----------------|
| **任务 A** | 层 B 的核心数据源。当前以 `A-Opt-03` 为默认 run；若后续主模型更新，层 B 需用最终版 run 重新生成。多任务头扩展（路线 β）决定壁面 WSS 是否可直接输出。 |
| **任务 B** | 层 C 的硬依赖。WSS/TAWSS/OSI/RRT 计算公式和区域定义必须与任务 B 完全一致。WSS 最小管线也是层 B 壁面图（路线 α）的前提。 |
| **任务 C** | 无直接依赖，但层 C 图件可为任务 C 的风险分层提供直观解释。 |
| **任务 D** | 层 B + 层 C 图件共同构成端到端论据。 |

---

## 9. 常用命令口径（目标接口）

```bash
# 环境
conda activate GNN          # 默认
conda activate GNN_vmtk     # 三维渲染子步骤

# 阶段 0：接口体检
python -m viz.io_utils --check --case CHEN_SHI_MING

# 阶段 1：CFD 真值图
python -m viz.wall_surface --case CHEN_SHI_MING --step 1120 --field wall-shear
python -m viz.interior_slice --case CHEN_SHI_MING --step 1120 --field velocity-magnitude
# 流线图需要 GNN_vmtk 环境
conda activate GNN_vmtk
python -m viz.streamlines --case CHEN_SHI_MING --step 1120

# 阶段 2：模型对比图
conda activate GNN
python -m viz.multitask_panel \
  --manifest outputs/field/<run_dir>/predictions_test/manifest.json \
  --case CHEN_SHI_MING --step 1120 --field vel_mag

# 阶段 3：周期指标对比
python -m viz.hemo_cycle_panel \
  --manifest outputs/field/<run_dir>/predictions_test/manifest.json \
  --case CHEN_SHI_MING
```

> 以上为目标脚本口径，复用 `predict_field.py` 产物，不另造中间格式。

---

## 10. 风险与对策

| 风险 | 级别 | 对策 |
|------|------|------|
| 非结构化点云流线积分质量差 | 高 | 备选方案：从 Fluent `.cas` 导入原始网格拓扑；或使用 ParaView 预处理后导出结构化数据 |
| STL 与 ASCII 坐标对齐偏差 | 中 | 阶段 0 强制验证 bbox 重合；映射距离分布超标时先排查单位 |
| 预测文件命名冲突 | 中 | 阶段 0 前修复 `predict_field.py`（加病例前缀或子目录） |
| 路线 β（多任务 WSS）推迟导致壁面主图缺失 | 中 | 路线 α（后处理 WSS）作为保底方案，与任务 B 并行推进 |
| 速度旋转后做 WSS 近似的方向一致性 | 中 | 明确记录 WSS 计算是在哪个坐标系下完成的，逆变换后壁面法向量是否需要同步旋转 |

---

## 11. 当前状态总结

| 阶段 | 状态 | 说明 |
|------|------|------|
| 阶段 0 | ⬜ | 部分已人工确认（E-0-01），脚本尚未编写 |
| 阶段 1 | ⬜ | 数据源确认完整，技术路线已明确，代码待实现 |
| 阶段 2 | ⬜ | 依赖阶段 1 的切面与视角固定；内部场对比可做，壁面 WSS 受阻 |
| 阶段 3 | ⬜（冻结） | 等待任务 B 管线就绪 |
| 阶段 4 | ⬜ | 非主线 |

> **当前阶段定位**：任务 E 处于**文档框架与技术路线确认期**，代码实现在任务 A–D 主线告一段落后启动。优先确保文档准确反映实际数据状态和依赖链。
