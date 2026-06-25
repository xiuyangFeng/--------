# CROWN/Beihang · 文档索引

> 代码与训练入口 → [../README.md](../README.md)  
> 实验族与 Job → [../experiments/README.md](../experiments/README.md)

| 文档 | 读者 | 内容 |
| --- | --- | --- |
| [**CROWN训练与采样流程说明.md**](CROWN训练与采样流程说明.md) | 理解 train/val/test 口径 | 分帧含义、53 亿点池 vs 1 万/step、PINN loss、流程文字摘要 |
| [数据加载与评估加速说明.md](数据加载与评估加速说明.md) | 改 train/eval 代码 | lazy_load、OOM 规避、evaluate 批量 |
| [5739_5740_根因诊断.md](5739_5740_根因诊断.md) | 查历史失败 | OOM 截断、PINN TIMEOUT |
| [../experiments/crown_original_vp/实验分析记录.md](../experiments/crown_original_vp/实验分析记录.md) | 非 PINN 逐 Job | 5625–5774 时间线 |
| [../experiments/CROWN_非PINN与PINN复现汇报_合并.md](../experiments/CROWN_非PINN与PINN复现汇报_合并.md) | **导师汇报（唯一入口）** | 5751/5774 + 5757/5806 · NMAE/R² · 1146 可视化 · 与 V3P 对照 |
| [../experiments/crown_original_vp_pinn/实验分析记录.md](../experiments/crown_original_vp_pinn/实验分析记录.md) | PINN 逐 Job | 5757/5806 正式结论 |

**产物（可视化，非 V3 指标表）**：

- `outputs/field/postview/crown_vp_t016_report/` — 非 PINN 1146 壁面三联图
- `outputs/field/postview/crown_pinn_t016_report/` — **PINN** 1146 壁面三联图
- `outputs/field/postview/V3P_vs_CROWN_对照表_1146.md` — 非 PINN 与 V3P 并排
- `outputs/field/postview/V3P_vs_CROWN_PINN_对照表_1146.md` — **PINN** 与 V3P 并排
