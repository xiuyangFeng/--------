# 压力场、速度场全局注意力：26臂单seed实验

用户于2026-09-10确认完整执行。状态：26臂400轮及全部best/last已完成，独立数值验收与工作簿回填通过。完整结果见[results.md](results.md)，完成证据见[completion_status.json](completion_status.json)。

## 当前执行状态

- 正式队列 **14040** 已完成全部26臂400轮、best/last全量评估和34病例分支长波诊断；训练源码与配置保持冻结，详见[queue_status.json](execution/queue_status.json)。
- 自动汇总 **14041** 已通过60组目标/checkpoint预测数值验收、26条训练history验收和工作簿回读核对，详见[report.json](report.json)及[xlsx_acceptance.json](xlsx_acceptance.json)。
- 真实batch8预检14034、历史复核14036及26臂两轮冒烟14038均通过；单测104项及40子测试、历史2948项检查通过。
- 最初12小时分配14039的四份部分轨迹保留[归档](interrupted_initial_allocation_14039/reason.json)，不计正式成绩；全部正式训练仅seed1234。

提交与完成依据见[submission.json](submission.json)、[completion_status.json](completion_status.json)。资源表反映单GPU四槽并发执行条件；[联合query成本](joint_query_costs.csv)单独记录联合执行和对应单任务成本，联合运行时间只计一次。

## 冻结协议

- V5现有体场view、train138/test34、峰值1162；18维输入，复用历史R5P/R5V完全相同的train-only输入统计。
- 压力为相对体积均压的Pa；壁面沿既有最近内部单元标签。参考压不输入模型。
- 5000壁面support；压力2500wall+2500volume query，速度5000volume query；独立、按epoch采样。
- QAD-lite、MSE、无pinball；seed1234、400epoch、batch8、AdamW lr0.001/wd0.0001、warmup10、cosine min1e-5、AMP、clip1。
- best按训练总损失；last为epoch399，即400轮。test34已暴露，仅开发筛选，不用于选择checkpoint。
- 本轮迁移M2局部壁面结构，固定18D，不将M2额外8维壁面曲率映射到内部；不加入新几何、PINN或时间维。

## 矩阵

每行分别训练Pxx压力与Vxx速度，共24次；另J00/J01共2次，总26次。

L为M2局部壁面支路：归一化半径0.015/0.03/0.06，cap16，通道32。G为原SA3后BT：32×256 tokens，1层8头、FFN512、dropout0、18D中心条件、相对几何bias、可学gamma初值0.001。C为相同中心条件但layers0；F为相同中心条件及近等参数逐tokenFFN。

| 结构 | 定义 | 父臂与关键对照 |
|---|---|---|
| 00 | 原R5同期从零训练 | 历史R5P/R5V |
| 01 | 00+L | 00 |
| 02 | 00+G | 00 |
| 03 | 01+G；直接接入主候选 | 01/02/04/05 |
| 04 | 01+C | 01/03 |
| 05 | 01+C+F | 03/04 |
| 06 | L+新SA3单半径0.20，无BT | 01 |
| 07 | 06+BT | 06/03 |
| 08 | L+新SA3三半径0.10/0.20/0.40，无BT | 06 |
| 09 | 08每支独立BT；老师完整主候选 | 08/07/03/10/11 |
| 10 | 09三支均用0.20半径 | 09 |
| 11 | 08每支近等参数FFN | 09/08 |
| J00 | 01共享骨干/FP/QAD，独立速度与压力heads | P01/V01，last主比较 |
| J01 | J00+G | J00与P03/V03 |

新SA3均从125源token取同32个FPS中心，半径内最近cap32，原SA3整体替换的影响由06承担；三支融合初始中尺度直通。00/01/02/03与06/07/08/09组成两个2×2对照。FFN残差结构与BT不同，只作跨token交互的容量参照。

joint输出[ux,uy,uz,p_rel]，分别复用旧P/V每任务5000点索引后union，双mask监督（最多10000query）。每病例速度分量平均z-MSE及压力z-MSE各0.5，再病例平均。joint统一best，对单任务以last400为主，不能分别挑两个best。

## 执行入口与证据

配置入口：[matrix.json](../../configs/volume_attention_20260910/matrix.json)，包含全部父臂差异与冻结筛选线。

```bash
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
$PY -m training_wss_min.tools.prepare_volume_attention_matrix --check
/public/slurm/bin/sbatch training_wss_min/cluster/preflight_volume_attention.slurm
/public/slurm/bin/sbatch training_wss_min/cluster/run_volume_attention.slurm --mode smoke
/public/slurm/bin/sbatch training_wss_min/cluster/run_volume_attention.slurm --mode references
$PY -m training_wss_min.tools.run_volume_attention_queue --make-gate <smoke_job/queue_status.json>
/public/slurm/bin/sbatch training_wss_min/cluster/start_volume_attention.slurm <completed_smoke/queue_status.json>
/public/slurm/bin/sbatch --dependency=afterany:<formal_job> training_wss_min/cluster/finalize_volume_attention.slurm
$PY -m training_wss_min.tools.report_volume_attention --validate
$PY -m training_wss_min.tools.update_volume_attention_xlsx
```

正式训练要求同源码/config哈希的单测、真实GPU预检、26臂各2轮冒烟、历史best/last完整复核均通过。每臂train→best评估/诊断→last评估/诊断；00—05先于06—11，再联合两臂；已有正式run拒绝覆盖。

源码快照、命令、阶段退出码与起止哈希见execution/；runtime_preflight.json记录真实batch和数据SHA；reference_reproduction.json记录原R5历史复核；tests.json和training_gate.json记录验收前提。

新预测存每个run的eval/ckpt_*/predictions/；joint压力、速度metrics分别在pressure/、velocity/子目录。diagnostics/有全部病例分支剖面CSV/PNG和summary.json。核心物理指标用独立NumPy从保存预测重算；原始旧run不写入任何新评估。

## 评价与预登记门槛

- 压力混合/内部/壁面分别报告R²_cb、MAE、RMSE、中位/P10、负R²病例数；低压差病例同时看Pa误差。
- 速度报告幅值、三分量/向量误差和轴向/径向/周向R²及RMSE；同时保留AG/AAA/ILO与逐病例比较。
- 全34例沿既有segment和s坐标按5mm箱做体积加权压力/轴向速度剖面，Gaussian sigma5/10/20/40mm同branch平滑、边界重归一，报告原始与去病例体积均值偏差残差。该量不是正交频带或理论上限。
- 压力best相对父臂ΔR²≥0.01且MAE下降≥3%，last同方向；速度best幅值ΔR²≥0.01且向量RMSE下降≥3%，last同方向，best/last径与周向R²均不得下降超过0.01。
- 长波改善要求best的20/40mm去bias残差MSE均下降≥10%，且通过全场精度保护。另列相对同期00与历史R5差值；模块有效不自动等同超越历史最佳。
- J01/J00分别判断两任务；联合对单任务last主比较。两任务均改善才称为联合提升。

所有26臂固定执行，不按中途test表现增删。best/last来自同一轨迹，不构成独立重复。资源开销同时报告参数、GPU峰值显存、训练和全场推理时间。

## 最终结论（2026-09-11）

压力优先P02：best/last R²_cb 0.74911/0.75283，相对P00通过精度和长波门槛；P03保留为低MAE备选。速度V07幅值R²最高（0.77625），相对V00及历史R5V精度提高，但相对V06的模块增益未通过向量RMSE门槛。联合J01未达到两任务共同提升。

详见[完整结论与证据](final_summary.md)。工作簿新增“体场注意力结论”页，最终回读证据见[xlsx_acceptance.json](xlsx_acceptance.json)。全部26臂仅seed1234，结论限定为暴露test34开发筛选。

## P02损失曲线与收敛诊断（2026-09-11）

已绘制[P02全程、末段loss和学习率](P02_loss_diagnostic/P02_training_loss.png)，并提供[PDF](P02_loss_diagnostic/P02_training_loss.pdf)、[逐轮CSV](P02_loss_diagnostic/loss_history.csv)及[判读](P02_loss_diagnostic/README.md)。best按训练loss选于第382轮（0.031375），last第400轮loss为0.037774；last测试R²较高不能直接证明epoch不足。最后50轮接近平台；本次未启动延长训练。
