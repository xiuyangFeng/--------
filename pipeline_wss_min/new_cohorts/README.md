# AAA/ILO v4 新队列工具

本子包只处理数据层审计与 v4 坐标架审计均通过的 AAA/ILO 单元，不扫描 AG。

> 2026-07-17 起，活动 ILO 作用域严格为 `ILO/*/before`。术后 `after` 不得进入
> 白名单、预处理、统计或训练；与 AAA 精确重名的 ILO 病例按 AAA 优先排除。

| 模块 | 职责 |
| --- | --- |
| `common.py` | 唯一白名单来源、共享路径和 unit_id 解析 |
| `audit_frame.py` | 正式预处理前的只读 STL 解剖坐标架审计 |
| `preprocess.py` | 单例或 Slurm array 正式预处理及批量汇总 |
| `qa.py` | 已落盘 bundle/report 的 v4 终检 |
| `audit_ag.py` | AG included 病例的只读 v4 回归检查 |
| `visualize_frame.py` | AAA/ILO 共同毫米坐标架可视化 |
| `ilo_before.py` | ILO-before 身份/数据/WSS/v4 终审、增量 manifest 与 after 派生产物清理 |
| `visualize_ilo_before.py` | AG76 + AAA63 + 最终通过 ILO-before41 共同框架审核图 |

旧的根目录模块名已删除，不提供兼容转发。请使用：

```bash
python -m pipeline_wss_min.new_cohorts.audit_frame --workers 4
python -m pipeline_wss_min.new_cohorts.ilo_before --stage inventory
python -m pipeline_wss_min.new_cohorts.ilo_before --stage qa
python -m pipeline_wss_min.new_cohorts.visualize_ilo_before
python -m pipeline_wss_min.new_cohorts.preprocess --list
python -m pipeline_wss_min.new_cohorts.qa
```

`preprocess` 的 ILO 清单由
`data_wss_min/pipeline_reports/ilo_before_final_approved_20260717/ilo_before_whitelist.json`
驱动；若该清单不存在则拒绝活动 ILO 预处理。历史 171 单元审计文件仅用于追溯，
不再是活动 ILO 真源。
