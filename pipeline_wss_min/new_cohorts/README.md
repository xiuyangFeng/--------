# AAA/ILO v4 新队列工具

本子包只处理数据层审计与 v4 坐标架审计均通过的 AAA/ILO 单元，不扫描 AG。

| 模块 | 职责 |
| --- | --- |
| `common.py` | 唯一白名单来源、共享路径和 unit_id 解析 |
| `audit_frame.py` | 正式预处理前的只读 STL 解剖坐标架审计 |
| `preprocess.py` | 单例或 Slurm array 正式预处理及批量汇总 |
| `qa.py` | 已落盘 bundle/report 的 v4 终检 |
| `audit_ag.py` | AG included 病例的只读 v4 回归检查 |
| `visualize_frame.py` | AAA/ILO 共同毫米坐标架可视化 |

旧的根目录模块名已删除，不提供兼容转发。请使用：

```bash
python -m pipeline_wss_min.new_cohorts.audit_frame --workers 4
python -m pipeline_wss_min.new_cohorts.preprocess --list
python -m pipeline_wss_min.new_cohorts.qa
```
