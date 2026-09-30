# V5.2 第二轮全周期精度优化

状态：**十臂均完训完评（2026-09-30）**。数组16165全部150/150轮、7800步、同一批55个留出单元/80相位评价完成，返回码全0；last checkpoint身份、源码/配置/统计及47项冻结指纹核验通过。最后J2于北京时间01:56:04完成，自动报告16166于01:56:06完成。见[完成核验](completion_verified.json)、[结果分析](analysis_20260930/analysis.md)、[完整结果矩阵](analysis_20260930/results_matrix.md)；[启动快照](startup_verified.json)保留为历史记录。

首轮独立控制对本轮候选的周期R²：速度Iu0.7432→U0均值邻域0.7480（谷底0.3048→0.3003）；压力Ip0.8690→P1单任务空间交互0.9033（周期MAE−16.44%）；WSS Iw0.7817→W1局部patch0.7911，TAWSS0.7730→0.7853。TCN未超过FFN容量控制，PCGrad未恢复三任务共同提升。保留U0/P1/W1为研究候选；单折单seed，不替换部署。

- 核心合同：[PREREG.md](../../configs/joint_cycle_round2_v52_20260929/PREREG.md)
- 十臂与顺序：[matrix.json](../../configs/joint_cycle_round2_v52_20260929/matrix.json)
- 基于已完成首轮：[首轮报告](../joint_cycle_v52_20260929/report.md)
- 研究依据：[下一轮矩阵](../../../docs/02-推进与变更/02-时间建模/_archive/阶段文档_2026-09-30/下一轮全周期精度优化_实验矩阵_2026-09-29.md)
- [CPU与参数预检](preflight_summary.json)、[实际冻结参数验收](frozen/f0_s1234_v1/parameter_validation.json)、[当前结果/进度报告](report.md)。
- [分段结果CSV](analysis_20260930/results_matrix.csv)、[80帧指标CSV](analysis_20260930/per_frame_metrics.csv)、[结果分析与后续判断](analysis_20260930/analysis.md)。
- 09-30后续只读分析：[速度谷底/减速误差诊断](analysis_velocity_20260930/diagnostics.md)，含Iu/U0已有预测的幅值/方向分解、速率/距壁分箱和同入口Q配对；[下一轮8次单seed研究设计](../../../docs/02-推进与变更/02-时间建模/_archive/阶段文档_2026-09-30/速度谷底与减速段_误差机制与实验矩阵_2026-09-30.md)，未提交新训练。

每臂固定seed1234、患者分组fold0（206训练／55开发留出单元）、150epochs、last checkpoint；支持5000、各任务训练query1024、全部80相位。复用首轮已审计缓存，不重建数据，不重训已有六个基线。

| 顺序 | 臂 | 任务 | 内容 |
|---|---|---|---|
| 0/1 | UT0/UT1 | 速度 | 逐相位FFN／31帧时间TCN |
| 2/3 | WT0/WT1 | WSS | 逐相位FFN／31帧时间TCN |
| 4/5 | U0/U1 | 速度 | 16邻居均值／查询注意力 |
| 6/7 | W0/W1 | WSS | 点式／局部壁面patch残差 |
| 8 | P1 | 相对压力 | 压力独立相位空间交互 |
| 9 | J2 | 三任务 | J1共享梯度PCGrad |

提交链为十臂训练数据GPU冒烟 → 数组`0-9%4`（最多四卡）→ node03 CPU汇总。冻结内容保存在`frozen/<tag>/`，提交记录与日志保存在本目录；正式输出为`training_wss_min/runs/joint_cycle_round2_v52_20260929/<arm>_f0_s1234/`。

报告至少同时给出峰值窗、谷底窗、整周期R²/MAE/RMSE/relative-L2，另列固定峰值帧21、80帧曲线、加速／减速／平台、方向、TAWSS、时间差分与成本。当前55单元是参与过设计决策的开发留出集；单折单seed结果仅用于内部筛选。
