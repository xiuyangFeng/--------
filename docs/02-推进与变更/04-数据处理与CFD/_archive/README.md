# 04 块归档

| 文档 | 归档日期 | 状态与结论 | 结论留在哪里 |
| --- | --- | --- | --- |
| [RCR 出口面积录入核查（2026-09-15）](./RCR出口面积录入核查_2026-09-15/README.md) | 2026-09-24 | 🧊 已完成。CFD 出口分流协议就是按出口面积的 Murray r³；23 例出口面积录错（test34 占 4 例），26 例按协议重算 CFD 并通过验收，剔除 2 例重复病例 → v5.1 母库 170 例（train136 / test34） | [历史卷 §23–§25](../../00-V5设计与历史跟踪/WSS_V5_训练实验跟踪_历史卷_2026-09-06至09-20.md)；修正值 `rcr_corrected_udf_values.csv`、审计 `rcr_area_audit.csv`（本目录内） |

目录内脚本的绝对路径已改到新位置；`rcr_area_audit.py` 等仍读取 [`_archive/WSS_PINN/_archive/V5审阅_BC协议结构探针_2026-09-06/`](../../_archive/WSS_PINN/_archive/V5审阅_BC协议结构探针_2026-09-06/rcr_protocol_summary.txt) 的逐例 CSV。
