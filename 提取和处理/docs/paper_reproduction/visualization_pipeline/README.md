# 点云预测结果回构与 CFD 后处理可视化流程

> 创建日期：2026-06-15
> 定位：集中管理“点云/图网络预测结果如何回到表面面片，以及如何进入 Fluent、CFD-Post、ParaView 或 PyVista 做后续可视化”的工程流程、数据格式和文献依据。

## 1. 结论先行

当前最稳的工程路线不是让神经网络“重新生成 CFD 网格”，而是保留或重建一条清楚的几何映射链：

```text
原始 CFD 表面/体网格
  -> 训练采样点或 surface graph node
  -> 模型在同一批点上输出 pressure / WSS / velocity
  -> 按 node_id / face_id / 最近邻 / VTK probe 映射回目标 surface
  -> 导出 CSV / VTP / VTU / STL+CSV
  -> CFD-Post / ParaView / Fluent 后处理可视化
```

本项目建议采用两层口径：

- **指标口径**：尽量在模型输出点与标签点同点计算，不把插值后的面片云图作为主要误差来源。
- **展示口径**：可以把点云预测值映射到 STL/VTK surface 上画连续云图，但必须记录映射方法、距离、覆盖率和插值参数。

## 2. 本目录结构

```text
docs/paper_reproduction/visualization_pipeline/
├── README.md
├── 01-点云到面片回构流程.md
├── 02-Fluent与CFDPost回写流程.md
├── 03-文献与工具证据表.md
├── schemas/
│   └── prediction_surface_csv_format.md
└── scripts/
    └── README.md
```

## 3. 推荐执行链

| 阶段 | 输入 | 输出 | 推荐方法 | 是否用于指标 |
| --- | --- | --- | --- | --- |
| A. 训练前建索引 | 原始 surface mesh / CFD result | `case_id, node_id, face_id, xyz, normal, area, region` | 从 Fluent/CFD-Post/VTK 导出 surface 节点或面心，并固定 ID | 是 |
| B. 模型预测 | 采样点云 / graph nodes | `pred_pressure, pred_wss, pred_velocity` | 保持点顺序和 `node_id`，避免只存无序 xyz | 是 |
| C. 回到目标 surface | 预测点 + 目标 STL/VTP | 带预测变量的 surface | 优先 ID join；其次 VTK probe/sample；最后最近邻或 IDW | 展示为主 |
| D. 后处理软件导入 | CSV / VTP / VTU / STL | CFD-Post/ParaView/Fluent 可视化对象 | CFD-Post CSV surface/point cloud，或 ParaView VTP/VTU | 展示为主 |
| E. 论文图导出 | 已映射 surface | 云图、切面、误差图 | 同一 colorbar、同一 target surface、同一单位 | 否，除非同点 |

## 4. 三条可落地路线

### 4.1 最推荐：VTK / ParaView 主链

适合论文图件和批量生成结果。

1. 原始 CFD 表面导出为 `.vtp`，体场导出为 `.vtu`。
2. 模型预测保存为 `case_id + point_id + xyz + pred_*`。
3. 若预测点就是 surface 节点，直接把 `pred_*` 写入 `.vtp` 的 point data。
4. 若预测点只是稀疏点云，用 PyVista/VTK 插值到目标 surface，并保存距离和未命中比例。
5. 用 ParaView 批处理导出 PNG、色标和误差图。

优点：保留几何拓扑、变量数组和批处理能力，后续可继续进入 CFD-Post 或其他后处理工具。

### 4.2 CFD-Post 交付链

适合老师或同组同学希望在 CFD-Post 里看结果。

1. 输出 CSV，列至少包含 `Node No., X[m], Y[m], Z[m], pred_wss[Pa], pred_pressure[Pa]`。
2. 若要形成 surface 而不只是点云，CSV 还需要 `[Faces]` 块，三角面或四边面用点编号定义。
3. 在 CFD-Post 使用 `File -> Import -> Import Surface or Line Data`，选择导入为 `Surface or Line` 或 `Point Cloud`。
4. 在导入对象上绘制 contour，和原始 CFD case 的 wall surface 叠加对比。

适合可视化，不建议作为最终误差指标来源。

### 4.3 Fluent 回写链

适合必须在 Fluent 环境内叠加展示或继续二次处理的场景，但工程复杂度最高。

1. 若只是显示自定义表面，优先读入 `.stl` / `.msh` / `.cas` surface，并在后处理中叠加。
2. 若要把 AI 预测量作为 Fluent 可访问变量，需要使用 UDM / UDF / profile 或外部插值脚本把值写到对应 wall face 或 node 上。
3. 若写回 Fluent 数据文件，必须保证单位、坐标系、zone、face/node 对齐，否则云图可能看起来正确但数值对应错误。

本项目第一阶段不建议把 Fluent 回写作为主链；应先打通 VTK/CFD-Post 交付链。

## 5. 后续代码管理约定

- 本目录只放流程、格式、文献和工具证据。
- 后续真实可视化脚本可以放入 `docs/paper_reproduction/visualization_pipeline/scripts/`；若脚本开始依赖项目数据和模型输出，再考虑迁移到 `external_baselines/` 或 `tools/visualization/`。
- 每个生成图件的脚本都应输出一份 `mapping_report.json`，至少包含：目标 mesh、源点数、命中点数、最大/均值映射距离、插值半径、未命中比例、变量单位。

## 6. 阅读顺序

1. [点云到面片回构流程](01-点云到面片回构流程.md)
2. [Fluent 与 CFD-Post 回写流程](02-Fluent与CFDPost回写流程.md)
3. [文献与工具证据表](03-文献与工具证据表.md)
4. [prediction surface CSV 格式](schemas/prediction_surface_csv_format.md)
