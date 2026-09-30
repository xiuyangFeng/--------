# V5.2速度谷底/减速段优化实验

**2026-09-30 12:21核对：8/8正式训练与评价全部完成。** 每臂150轮/7800步、last checkpoint、55个相同开发留出单元及80相位；16265/16266所有子任务和16267报告均COMPLETED/0:0，最后训练于06:49:22结束，自动报告于06:49:30结束。全量作业16192及其后续16193也已正常结束，本轮未改动其训练代码或配置。提交历史见[submission.json](submission.json)，完成审计见[completion_audit.json](analysis_20260930/completion_audit.json)。

**结果判断**：G11四个主阶段平均R²最高，峰值/谷底/减速/周期为**0.8167 / 0.3159 / 0.6876 / 0.7571**；相对U0分别提高0.0135/0.0157/0.0094/0.0091。谷底改善仍小，且对G10的谷底增幅不足0.02、方向余弦下降，未满足完整预注册条件。D1方向改善但精度增幅小；D2和A1当前方案不支持继续原样推进。G11保留为进一步研究候选，D1保留方向优化线索。完整[结果分析](analysis_20260930/analysis.md)、[成绩矩阵](analysis_20260930/results_matrix.md)、[机器汇总](report.json)。

G11固定查询推理中位0.3535s，对U0的0.2138s为1.654倍，超过1.5倍成本偏好；不含原始几何预处理，不代表临床全点云时延。本轮只有速度实验，压力P1、标量WSS W1沿用上一轮结果。

[8臂GPU门禁](frozen/f0_s1234_v1/smoke_gate.json)全部通过。D1训练侧32batch无优化器更新的标定值为λ=0.0860702960，见[标定记录](calibration/direction_lambda.json)；几何与λ分别通过独立SHA门禁。冒烟只用于可运行性/数值验收，不作为正式精度结果。

| 阶段 | Slurm Job ID | 提交时合同 |
|---|---|---|
| CPU几何sidecar、训练q80统计 | 16263 | node03，完整261单元验收 |
| GPU方向λ标定＋8臂冒烟 | 16264 | afterok:16263，仅训练单元 |
| 第一批D1/D2/A0/A1 | 16265_0–3 | afterok:16264，数组并发上限4 |
| 第二批G00/G01/G10/G11 | 16266_4–7 | afterany:16265且afterok:16264，并发上限4 |
| CPU进度/结果报告 | 16267 | afterany，失败/缺失状态如实报告 |

[43项CPU测试](implementation_checks.json)通过，58个受保护旧源码/全量配置文件指纹均未改变；[前后兼容性核验](compatibility_after.json)另确认70项新冻结指纹一致。冻结33个传递依赖源码文件，见[源码manifest](frozen/f0_s1234_v1/source_manifest.json)；[参数验收](frozen/f0_s1234_v1/parameter_validation.json)、[原缓存验收](frozen/f0_s1234_v1/cache_validation.json)及[提交/门禁核验](submission_verified.json)通过。正式结果由完整训练与留出评价产物验收，CPU/GPU冒烟结果不计入成绩。

**数据沿用前两轮实验，206训练/55开发留出单元、患者分组fold0、80相位。** 与本轮期间运行的全量训练隔离，使用原261单元缓存、原split和训练统计。150epochs/last、batch4、support5000、train query1024、eval query16384。

| 批次 | 臂 | 内容 | 主对照 |
|---|---|---|---|
| 1 | D1 | 有界方向辅助loss，训练侧32batch梯度标定λ | U0 |
| 1 | D2 | 每相位训练速率q80尾部双权重 | U0 |
| 1 | A0 | 局部frame输入、XYZ残差 | U0 |
| 1 | A1 | 同输入、局部frame残差后旋回XYZ | A0 |
| 2 | G00 | 静态点式体内锚点 | A0/2×2控制 |
| 2 | G01 | 相位条件点式体内锚点 | G00 |
| 2 | G10 | 静态体内几何图 | G00 |
| 2 | G11 | 相位条件体内几何图 | G10、G01 |

源码、配置、原缓存元数据在`frozen/<tag>/`冻结。CPU几何sidecar与训练侧λ分别在完成时保存独立SHA manifest，训练前校验。运行新入口`training_wss_min.velocity_phase`；旧入口、全量run/配置和已运行作业不修改。

Slurm：node03几何 → GPU标定与8臂冒烟 → 第一批4臂 → 第二批4臂 → afterany报告。正式任务并行上限4，实际并发随Slurm GRES空余资源分配；执行期间未绕过、取消或重启全量作业16192。构建完整几何、训练侧标定和全部冒烟通过后才进入正式训练。

- [结果工作簿](../../../docs/03-汇报材料/WSS_PointNet实验矩阵与结果汇总last.xlsx)：`V52速度周期优化`页，Iu/U0第6–7行、8个新实验第8–15行；[备份与200指标格核对记录](analysis_20260930/workbook_update.json)。
- [完成与预测完整性验收](analysis_20260930/completion_audit.json)、[7800次实际更新验收](analysis_20260930/optimizer_audit.json)、[Slurm数组/实际编号映射](analysis_20260930/scheduler_mapping.json)。
- [正式矩阵配置](../../configs/velocity_phase_v52_20260930/matrix.json)
- [预注册合同](../../configs/velocity_phase_v52_20260930/PREREG.md)
- [研究设计与机制依据](../../../docs/02-推进与变更/02-时间建模/_archive/阶段文档_2026-09-30/速度谷底与减速段_误差机制与实验矩阵_2026-09-30.md)
- [进度与结果](report.md)、[完整可机读指标](report.json)、[扁平指标CSV](metrics_flat.csv)
- [提交工具](../../tools/submit_velocity_phase_v52.py)、[只读报告工具](../../tools/report_velocity_phase_v52.py)

主报告包含峰值、谷底、减速和整周期R²/MAE，另保留方向、幅值、早晚谷底、逐帧、解剖分解、配对困难单元和成本。没有保存的历史诊断保持缺失。单折单seed开发筛选不等同临床验证。
