# TAWSS / OSI 直接回归 导师汇报包（2026-09-22）

- 汇报稿（叙事 + 全部图表说明）：[导师汇报_TAWSS_OSI直接回归与三头集成_2026-09-22.md](导师汇报_TAWSS_OSI直接回归与三头集成_2026-09-22.md)
- 图：`figures/fig01`–`fig16`（PNG，170 dpi）
- 表：`tables/T1`–`T9`（同名 `.md` 可直接贴、`.csv` 可进 Excel），`tables/summary_numbers.json` 供 PPT 取数
- 生成脚本：`scripts/make_report_figures.py`（只读；仓库根目录 `/public/newhome/cy/.conda/envs/GNN/bin/python <脚本>` 约 3 分钟重跑）
- 壁面图色图在脚本顶部 `CMAP_TAWSS / CMAP_OSI / CMAP_ERR = "plasma" / "viridis" / "coolwarm"`，TAWSS 按每例 p5–p95 对数拉伸、OSI 固定 0–0.4；想换 ParaView 风格改成 `"turbo"` 或 `"coolwarm"` 重跑即可

| 图 | 内容 | 回答的问题 |
|---|---|---|
| fig01 | 逐帧 WSS R² 随周期变化 + 周期均值 c0 | 为什么不逐帧、要直接回归周期量 |
| fig02 | 170 例真值 OSI–TAWSS 联合分布 + 三队列区域面积 | 为什么要设自由基线；临床区域多大 |
| fig03 | 阶段 1 六个门的三折读数 | 哪些门过了、哪些没过 |
| fig04 | 尾部分解（TAWSS 与三头峰值通道） | Pa 门不过 / 三头峰值掉分 的原因 |
| fig05 | OSI 掩膜阈值敏感性 | G2.2 不过的原因 |
| fig06 | test34 五面板汇总（逐 seed + 集成） | 最终读数 |
| fig07 | 逐例 R² 散点 | 逐例看难度 |
| fig08 | CFD 对预测逐点 hexbin | 逐点一致性 |
| fig09 | 病例级 Bland–Altman | 汇总数的误差带 |
| fig10 | 分段误差箱线 | 按血管段给数的粒度 |
| fig11 / 12 | TAWSS / OSI 壁面分布（CFD 对预测，三队列各一例） | 直观对比 |
| fig13 | 三种临床掩膜的命中/假阳/漏检 | 区域交付的可靠度 |
| fig14 | 三头对单头 Δ | 三头是否伤各通道 |
| fig15 | 文献对比（相对 L2 / MAE） | 竞争力 |
| fig16 | 部署计时 + 真实 STL 三场 | 已落地形态 |

| 表 | 内容 |
|---|---|
| T1 | 阶段 1 逐折门控 |
| T2 / T3 / T4 | test34 TAWSS / OSI / 滞留区 主表 |
| T5 / T6 | 三头对单头；尾部分解 |
| T7 | 文献对比 |
| T8 | 34 例逐例明细 |
| T9 | 部署计时 |
