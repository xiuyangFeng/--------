# PointNetCFD · 实验族索引

| 实验族 | 配置 | 目标 | 实验分析记录 |
| --- | --- | --- | --- |
| `pointnetcfd_original_vp` | `configs/local/pointnetcfd_original_vp_split_AG_v1_seed1.json` | u,v,w,p | [实验分析记录](pointnetcfd_original_vp/实验分析记录.md) |
| `pointnetcfd_wall_vp` | `configs/local/pointnetcfd_wall_vp_split_AG_v1_seed1.json` | u,v,w,p | [实验分析记录](pointnetcfd_wall_vp/实验分析记录.md) |
| `pointnetcfd_geom_vp` | `configs/local/pointnetcfd_geom_vp_split_AG_v1_seed1.json` | u,v,w,p | [实验分析记录](pointnetcfd_geom_vp/实验分析记录.md) |
| `pointnetcfd_geom_pwss` | `configs/local/pointnetcfd_geom_pwss_split_AG_v1_seed1.json` | p,wss_x,wss_y,wss_z（壁面） | [实验分析记录](pointnetcfd_geom_pwss/实验分析记录.md) |

**回填规则**：每个 Job → `experiments/<name>/` + `outputs/.../analysis_report.md`；**四组矩阵齐** → 写 [`docs/paper_reproduction/papers/pointnetcfd/梳理记录.md`](../../../docs/paper_reproduction/papers/pointnetcfd/梳理记录.md)。规范：[`04-梳理记录规范`](../../../docs/paper_reproduction/04-梳理记录规范.md)

**Job 5610–5613（2026-06-15）**：四组均已完跑，产物时间戳 `20260615_211946`。
