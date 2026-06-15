# Prediction Surface CSV 格式

## 1. 目标

这个格式用于把模型预测结果交给 CFD-Post 或其他后处理工具。它同时兼容两种模式：

- 无 `[Faces]`：作为 point cloud 导入；
- 有 `[Faces]`：作为 surface 导入并绘制 contour。

## 2. 最小 point cloud 格式

```text
[Name]
AI_Prediction_PointCloud
[Data]
Node No., X[m], Y[m], Z[m], pred_wss[Pa], true_wss[Pa], abs_error_wss[Pa]
0, 0.001000, 0.002000, 0.003000, 4.200000, 4.550000, 0.350000
1, 0.001200, 0.002100, 0.003200, 3.800000, 3.600000, 0.200000
```

## 3. Surface 格式

```text
[Name]
AI_Prediction_Surface
[Data]
Node No., X[m], Y[m], Z[m], pred_wss[Pa], true_wss[Pa], abs_error_wss[Pa]
0, 0.001000, 0.002000, 0.003000, 4.200000, 4.550000, 0.350000
1, 0.001200, 0.002100, 0.003200, 3.800000, 3.600000, 0.200000
2, 0.001100, 0.002300, 0.003100, 4.400000, 4.300000, 0.100000
[Faces]
0 1 2
```

## 4. 推荐列

| 列 | 必需 | 说明 |
| --- | --- | --- |
| `Node No.` | 是 | CFD-Post surface/point cloud 的节点编号 |
| `X[m]`, `Y[m]`, `Z[m]` | 是 | 坐标，单位必须写入列名 |
| `pred_wss[Pa]` | WSS 图需要 | 预测 WSS，可为标量模长 |
| `true_wss[Pa]` | 对比图需要 | CFD 真值 |
| `abs_error_wss[Pa]` | 误差图需要 | 绝对误差 |
| `pred_pressure[Pa]` | 压力图需要 | 预测压力 |
| `true_pressure[Pa]` | 压力图需要 | CFD 压力，需说明参考零点 |
| `region` | 可选 | 若 CFD-Post 不识别字符串，可另存映射表 |
| `source_node_id` | 可选 | 用于追踪回原 mesh，不一定在 CFD-Post 中使用 |
| `source_face_id` | 可选 | 同上 |

## 5. 配套 mapping report

每个 CSV 旁边应保存同名 JSON：

```json
{
  "case_id": "case001",
  "csv_file": "case001_cfdpost_surface.csv",
  "mode": "surface",
  "coordinate_unit": "m",
  "field_units": {
    "pred_wss": "Pa",
    "true_wss": "Pa",
    "pred_pressure": "Pa"
  },
  "source_geometry": "case001_surface_base.vtp",
  "mapping_method": "id_join",
  "point_count": 15000,
  "face_count": 29996,
  "unmatched_count": 0,
  "mean_mapping_distance_m": 0.0,
  "max_mapping_distance_m": 0.0
}
```

## 6. 命名约定

```text
<case_id>_prediction_points.csv
<case_id>_surface_pred.vtp
<case_id>_cfdpost_surface.csv
<case_id>_mapping_report.json
```
