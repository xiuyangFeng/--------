# 任务A V2准备执行清单

> 上位文档：[任务A V2修正路线实验矩阵](任务A_V2修正路线实验矩阵.md) | [实验设计总纲](../../实验设计总纲.md) | [任务B指标计算规范](../任务B/任务B指标计算规范.md)

## 1. 文档定位

本文件只服务于 **`Route-PhysicsAware-V2`** 的 **Gate-0**。

目标不是训练模型，而是先把下面四件事做硬：

1. `.cas/.msh` 网格拓扑是否可导出
2. 采样点与原始 mesh 的关系是否可追溯
3. WSS 真值与采样节点是否可对齐
4. V2 数据链路是否能在 2~3 个病例上跑通最小闭环

如果这四步没有完成，后面的 `V2G-* / V2P-*` 都不应启动。

---

## 2. 环境与总目录

### 2.1 环境

本项目默认环境仍然是：

```bash
conda activate rag_venv
```

### 2.2 V2 建议目录

建议新增但不覆盖现有 V1 目录：

```text
workspace/v2/
├── mesh_export/
│   └── <case_id>/
├── sampled_mapping/
│   └── <case_id>/
├── wss_alignment/
│   └── <case_id>/
└── smoke/
    └── <case_id>/
```

说明：

- `mesh_export/`：原始网格导出结果
- `sampled_mapping/`：采样点与 mesh 的对应关系
- `wss_alignment/`：WSS 真值与采样点对齐结果
- `smoke/`：V2 小规模闭环测试记录

---

## 3. Gate-0 的四个准备项

### 3.1 `V2-Prep-01`：mesh 导出检查

#### 目标

确认 `.cas/.msh` 能导出 V2 最低需要的结构信息。

#### 最低必需字段

对每个病例，至少要能导出：

1. **节点表**
   - `node_id`
   - `x, y, z`

2. **单元表**
   - `cell_id`
   - `cell_type`
   - `node_ids`

3. **壁面面元或壁面节点标记**
   - `face_id` 或 `wall_node_id`
   - `boundary_name`
   - `is_wall`

4. **区域/边界名称**
   - 入口
   - 各出口
   - 壁面

#### 建议产物

每个病例目录建议生成：

```text
workspace/v2/mesh_export/<case_id>/
├── mesh_nodes.csv
├── mesh_cells.csv
├── mesh_wall_faces.csv
├── mesh_boundary_summary.json
└── export_log.md
```

#### 字段建议

`mesh_nodes.csv`

| 列名 | 含义 |
| --- | --- |
| `node_id` | 原始网格节点编号 |
| `x` | x 坐标 |
| `y` | y 坐标 |
| `z` | z 坐标 |

`mesh_cells.csv`

| 列名 | 含义 |
| --- | --- |
| `cell_id` | 单元编号 |
| `cell_type` | tet / hex / poly 等 |
| `node_ids` | 组成单元的节点 id 列表 |

`mesh_wall_faces.csv`

| 列名 | 含义 |
| --- | --- |
| `face_id` | 面元编号 |
| `node_ids` | 构成该面的节点 id 列表 |
| `boundary_name` | wall / inlet / outlet 等 |
| `is_wall` | 是否壁面 |

`mesh_boundary_summary.json`

建议至少包含：

```json
{
  "case_id": "CHEN_FU",
  "n_nodes": 0,
  "n_cells": 0,
  "n_wall_faces": 0,
  "boundary_names": ["inlet", "wall", "outlet_1", "outlet_2"],
  "has_wall_region": true,
  "has_inlet_region": true,
  "has_outlet_region": true
}
```

#### 通过标准

- 至少 3 个病例成功导出上述 4 类文件
- 坐标范围与现有 `ascii/ascii_in` 数据一致到同一数量级
- 壁面区域存在，且边界名不是全丢失的匿名编号

#### 失败即停机

出现下面任一情况，本项视为失败：

- 只能导出节点坐标，拿不到单元连接
- 壁面区域无法识别
- 导出的坐标与现有 CFD 点云明显不在同一空间范围

---

### 3.2 `V2-Prep-02`：采样映射检查

#### 目标

确认现有采样点可以与原始 mesh 建立可回溯关系。

这一步是 `Route-MeshGNN-V2` 能否成立的关键。

#### 需要回答的具体问题

1. 每个采样点对应哪个原始 `node_id` 或最近 mesh 节点
2. 如果采样点不是原始节点，能否给出：
   - 所属单元 `cell_id`
   - 局部重心/插值权重
3. 壁面采样点和近壁内部点能否区分

#### 建议产物

```text
workspace/v2/sampled_mapping/<case_id>/
├── sampled_nodes.csv
├── sampled_to_mesh_map.csv
├── sampled_neighbor_audit.json
└── mapping_log.md
```

#### 字段建议

`sampled_nodes.csv`

| 列名 | 含义 |
| --- | --- |
| `sample_id` | 采样点编号 |
| `x` | x 坐标 |
| `y` | y 坐标 |
| `z` | z 坐标 |
| `is_wall` | 是否壁面点 |
| `source_type` | `wall` / `inner` |

`sampled_to_mesh_map.csv`

| 列名 | 含义 |
| --- | --- |
| `sample_id` | 采样点编号 |
| `map_type` | `node` / `cell` / `nearest_node` |
| `node_id` | 对应节点 id |
| `cell_id` | 对应单元 id |
| `distance_mm` | 与映射对象的距离 |
| `inside_cell` | 是否在单元内部 |

`sampled_neighbor_audit.json`

建议至少包含：

```json
{
  "case_id": "CHEN_FU",
  "n_sampled": 15000,
  "n_wall_sampled": 0,
  "n_inner_sampled": 0,
  "mapped_ratio": 0.0,
  "exact_node_ratio": 0.0,
  "mean_map_distance_mm": 0.0,
  "p95_map_distance_mm": 0.0
}
```

#### 通过标准

- `mapped_ratio >= 0.99`
- `p95_map_distance_mm` 足够小，不能出现大量远距离“硬贴”
- 对壁面点和近壁内部点的映射可区分

#### 额外检查

建议人工抽查 3 类点：

- 壁面点
- 近壁内部点
- 分叉附近内部点

确认映射不是“看起来能对上，实际跨壁面”。

---

### 3.3 `V2-Prep-03`：WSS 对齐检查

#### 目标

确认 CFD 真值 WSS/TAWSS 能与采样节点或壁面节点建立稳定对应。

#### 要避免的错误

不要默认拿任意一个壁面 CSV 和模型输出按坐标最近邻强配，就当“真值对齐成功”。  
如果没有明确的节点/面元对应，你很容易得到一个表面上相关、实际上不可信的结果。

#### 建议产物

```text
workspace/v2/wss_alignment/<case_id>/
├── wall_truth_nodes.csv
├── sampled_wall_wss_map.csv
├── tawss_case_summary.csv
├── alignment_qc.json
└── alignment_log.md
```

#### 字段建议

`wall_truth_nodes.csv`

| 列名 | 含义 |
| --- | --- |
| `wall_node_id` | 壁面真值节点 id |
| `x` | x 坐标 |
| `y` | y 坐标 |
| `z` | z 坐标 |
| `WSS` | 某时间步瞬时 WSS |
| `TAWSS` | 时间平均 WSS |
| `OSI` | OSI |
| `RRT` | RRT |

`sampled_wall_wss_map.csv`

| 列名 | 含义 |
| --- | --- |
| `sample_id` | 采样壁面点编号 |
| `wall_node_id` | 对应真值壁面节点 |
| `distance_mm` | 对齐距离 |
| `WSS` | 对应瞬时 WSS |
| `TAWSS` | 对应 TAWSS |
| `OSI` | 对应 OSI |
| `RRT` | 对应 RRT |

`alignment_qc.json`

建议至少包含：

```json
{
  "case_id": "CHEN_FU",
  "mapped_wall_ratio": 0.0,
  "mean_wall_distance_mm": 0.0,
  "p95_wall_distance_mm": 0.0,
  "has_wss": true,
  "has_tawss": true,
  "has_osi": false,
  "has_rrt": false
}
```

#### 通过标准

- `WSS` 和 `TAWSS` 至少可以稳定对齐
- 壁面对齐距离足够小
- 至少 3 个病例可重复完成

#### 当前阶段建议

优先级顺序固定为：

1. `WSS`
2. `TAWSS`
3. `OSI`
4. `RRT`

也就是说，即使 `OSI / RRT` 暂时没法稳定对齐，只要 `WSS / TAWSS` 成立，V2 仍可继续推进。

---

### 3.4 `V2-Prep-04`：V2 数据链路 smoke test

#### 目标

在 2~3 个病例上跑通 V2 数据准备与评估的最小闭环。

#### 最小闭环内容

1. mesh 导出
2. 采样映射
3. WSS 对齐
4. 生成 V2 训练输入资产
5. 用一个最简单模型跑 1 次 smoke test

#### 建议产物

```text
workspace/v2/smoke/<case_id>/
├── smoke_manifest.json
├── smoke_checklist.md
├── smoke_metrics.json
└── smoke_failures.md
```

`smoke_manifest.json` 建议字段：

```json
{
  "case_id": "CHEN_FU",
  "mesh_export_ready": true,
  "mapping_ready": true,
  "wss_alignment_ready": true,
  "v2_asset_ready": true,
  "smoke_train_ready": true
}
```

#### 通过标准

- 至少 2 个病例完整通过
- 所有中间文件命名一致
- 不依赖手工临时改脚本路径才能复现

---

## 4. Gate-0 统一通过标准

只有同时满足下面 4 条，Gate-0 才算通过：

1. `V2-Prep-01` 通过
2. `V2-Prep-02` 通过
3. `V2-Prep-03` 至少在 `WSS / TAWSS` 上通过
4. `V2-Prep-04` 至少 2 个病例通过

若未通过：

- 不得启动 `V2G-Base-01`
- `V2P-*` 可否先行，取决于 `V2-Prep-03` 是否已足够支撑 WSS 对齐

---

## 5. Gate-0 完成后的立即动作

Gate-0 一旦通过，下一步只允许启动：

1. `V2-Ref-Base-01`
2. `V2G-Base-01`
3. `V2G-Main-01`
4. `V2P-Base-01`
5. `V2P-Main-01`

严禁直接跳到：

- `V2G-Opt-*`
- `V2P-Opt-*`
- `physics`
- `time modeling`
- `risk prediction`

---

## 6. 当前最推荐的执行顺序

### Day 1

- 抽取 3 个代表性病例
- 完成 `V2-Prep-01`

### Day 2

- 完成 `V2-Prep-02`
- 人工抽查壁面点/近壁点/分叉点映射

### Day 3

- 完成 `V2-Prep-03`
- 优先把 `WSS / TAWSS` 对齐做通

### Day 4

- 完成 `V2-Prep-04`
- 生成 smoke 资产与 checklist

### Day 5

- 复盘 Gate-0
- 只有通过后才开始准备 `V2-Ref-Base-01` 与 `V2G-* / V2P-*`

---

## 7. 一句话原则

Gate-0 的目标不是“看起来差不多能跑”，而是**把 V2 的物理可信性前提一次性钉死**。
