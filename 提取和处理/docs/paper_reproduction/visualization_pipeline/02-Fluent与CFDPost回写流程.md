# Fluent 与 CFD-Post 回写流程

## 1. 推荐优先级

| 优先级 | 目标 | 路线 | 当前建议 |
| --- | --- | --- | --- |
| P0 | 论文/汇报云图 | VTK/ParaView/PyVista | 最稳，适合批处理和可复现 |
| P1 | 在 CFD-Post 中查看 AI 结果 | CSV surface / point cloud import | 可交付，适合老师检查 |
| P2 | 在 Fluent 内把 AI 预测当作变量继续操作 | UDM/UDF/profile/mesh surface | 可做，但先不要作为主链 |

## 2. CFD-Post 导入点云或面片

CFD-Post 官方支持导入 CSV 文件，并把数据创建为 surface、line 或 point cloud。工程上有两种格式：

### 2.1 Point Cloud CSV

只用于点云展示，不包含面片拓扑。

```text
[Name]
AI_Pred_WSS_PointCloud
[Data]
Node No., X[m], Y[m], Z[m], pred_wss[Pa], true_wss[Pa], abs_error[Pa]
0, 0.001, 0.002, 0.003, 4.20, 4.55, 0.35
1, 0.001, 0.002, 0.004, 3.80, 3.60, 0.20
```

导入时选择 `Point Cloud`。

### 2.2 Surface CSV

用于形成可画 contour 的用户 surface，需要 `[Faces]` 块。

```text
[Name]
AI_Pred_WSS_Surface
[Data]
Node No., X[m], Y[m], Z[m], pred_wss[Pa], true_wss[Pa], abs_error[Pa]
0, 0.001, 0.002, 0.003, 4.20, 4.55, 0.35
1, 0.002, 0.002, 0.003, 4.00, 4.10, 0.10
2, 0.001, 0.003, 0.003, 4.40, 4.30, 0.10
[Faces]
0 1 2
```

导入时选择 `Surface or Line`。如果目标只是和 CFD wall surface 叠加检查，可以先使用 point cloud；如果要画连续等值面，优先输出 surface。

## 3. CFD-Post 操作步骤

1. 打开原始 `.res` / `.cas` / `.dat` 或对应结果文件。
2. `File -> Import -> Import Surface or Line Data`。
3. 选择 AI 预测导出的 CSV。
4. `Import As` 选择 `Point Cloud` 或 `Surface or Line`。
5. 在 Outline 里检查导入对象是否出现在 `User Locations and Plots`。
6. 创建 Contour，Location 选导入对象，Variable 选 `pred_wss`、`true_wss` 或 `abs_error`。
7. 固定色标范围、单位和相机视角，导出图片。

## 4. Fluent 内显示或继续处理

### 4.1 只导入几何 surface

如果 Fluent 里只需要一个可叠加的几何表面，可读入 `.stl`、`.msh`、`.cas` 等 surface。这个路线适合定位、裁剪、叠加展示，但不自动携带 AI 预测变量。

### 4.2 Profile / point cloud 可视化

Fluent 2025 R2 beta 文档中已有 profile point cloud 可视化相关 TUI 命令，例如显示 profile point cloud、把 profile point cloud 叠加到 mesh 或 contour 上。这个方向适合后续验证“AI 点云结果能否在 Fluent 内直接叠加”，但版本依赖强，当前不应作为论文图主链。

### 4.3 UDM / UDF 写入变量

如果必须让 Fluent 把 `pred_wss` 当作变量参与后续场计算或导出，需要：

1. 分配 User-Defined Memory。
2. 用 UDF 或外部 profile 把预测值按 wall face/node 写入 UDM。
3. 保存 `.dat` 或 `.cdat` 时确认 UDM 变量被包含。
4. 在 CFD-Post 或 Fluent Post 中读取 UDM 并画图。

这个路线的关键风险是 face/node ID 必须完全对齐。若只按 xyz 最近邻写入，必须设最大距离阈值并输出错配报告。

## 5. 本项目第一阶段交付标准

每个 case 至少导出三类文件：

```text
case_id_surface_pred.vtp
case_id_cfdpost_surface.csv
case_id_mapping_report.json
```

其中 `mapping_report.json` 至少包含：

```json
{
  "case_id": "case001",
  "source": "model_prediction_points",
  "target": "original_surface_vtp",
  "method": "id_join",
  "source_point_count": 15000,
  "target_point_count": 15000,
  "matched_count": 15000,
  "unmatched_count": 0,
  "max_distance_m": 0.0,
  "mean_distance_m": 0.0,
  "fields": ["pred_wss", "true_wss", "abs_error"],
  "units": {"wss": "Pa", "pressure": "Pa"}
}
```

## 6. 不建议的做法

- 不要只导出截图，不导出带变量的中间文件。
- 不要把插值到 surface 后的平滑云图当作主指标。
- 不要在没有 `node_id` 或 `face_id` 的情况下承诺“已准确写回 Fluent 原网格”。
- 不要让 AI 预测面片和 CFD 真值面片使用不同色标或不同几何底板。
