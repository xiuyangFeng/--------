# 可视化脚本预留目录

后续可在这里放置点云预测结果到 VTK/CFD-Post 文件的转换脚本。建议优先实现以下脚本：

| 脚本 | 用途 |
| --- | --- |
| `audit_prediction_mapping.py` | 检查预测点与原始 surface 的 ID/距离/区域匹配情况 |
| `write_surface_vtp.py` | 把预测字段写入 `.vtp` point data 或 cell data |
| `write_cfdpost_surface_csv.py` | 导出 CFD-Post 可导入的 `[Data]` + `[Faces]` CSV |
| `render_paraview_batch.py` | 批量生成 true/pred/error 三联图 |

脚本落地时必须同步输出 `mapping_report.json`，并在本目录 README 中补充调用方式。
