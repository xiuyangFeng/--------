# 同期 MO0 与历史 M2 的复现偏差审计

结论：同一配置配方产生了不同的训练结果。已排除配置原有字段、训练统计、评估实现与选模轮次差异；训练轨迹从第 0 轮就出现小幅分歧。GPU AMP/FP32 各 12 步检查已完成，旧源码自身重复也会分叉；400 轮最终 R² 差距的归因仍未确认。不能将当前差异直接写成“GPU 噪声已证实”。

| checkpoint | 历史 M2 Pa R²_cb | 同期 MO0 | Δ | 历史/同期 log_z R²_cb |
|---|---:|---:|---:|---|
| best | 0.62841264 | 0.59258838 | -0.03582426 | 0.82369655 / 0.82119012 |
| last | 0.62429355 | 0.59648868 | -0.02780487 | 0.82462766 / 0.82148146 |

## 已核实的证据

- 原M2源config与历史run/config只差默认项展开和训练集计算统计。历史M2与MO0实际run/config原有字段没有值变化；新增为关闭的新模块默认值、配对初始化路径、loss组件日志开关和为0的Pa-MSE系数。
- feature_stats、wss_global_stats、weight_quantiles、target_normalization四份JSON内容完全相同。dataset.py、evaluate.py、runtime.py、models.py与13205源码快照逐字节相同。
- 两者均400轮；全部轮次学习率和采样点数逐项一致。两者best均为epoch378，last均epoch399；checkpoint中的epoch、score、cfg_name与各自history/目录一致。
- 第0轮loss：0.862783349223 → 0.863080945280（Δ=+0.000297596057）；400轮loss曲线相关系数0.99972805。差异已出现在训练阶段，不能归因于最后选取了不同epoch。
- 独立加载旧/新objectives，CPU上123/5000/40000点、预测FP16/FP32共6组：loss及预测梯度逐位相同；新组件日志开关不改变结果。此检查未覆盖GPU模型后向。
- 本轮smoke使用原M2 best重评完整test34：0.628412635267；历史0.628412635614，5622项原指标复现检查通过。表明旧权重仍可在当前数据/评估代码上恢复原结果。
- 原初始化元数据仅记录random_initialization，没有保存当时初始张量。本轮旧源码/新源码重建seed1234初始状态逐位相同；CPU旧/新模型完整5000点前向逐位相同，GPU差异与旧模型自身重复前向同量级。该证据属于重建检查，不能冒称已直接比过未保存的历史初始张量。
- 原13205与当前13974均master节点，同一GNN Python/Torch/NumPy等已记录版本；CPU配额32→16。两套启动脚本显式线程设置均OMP=2、OPENBLAS=1、MKL=1；新脚本另设MPLBACKEND=Agg。均未显式设TF32/CUBLAS选项，但历史进程继承环境和后端标志未归档，无法证明该层完全相同。

## 34病例及高值误差

全部34例完整配对保存在同名JSON的results.best/last.case_comparison.per_case，含病例R²、归一化R²、MAE、高WSS MAE/RMSE/R²、热点IoU、top10/p99比和MSE贡献。两种checkpoint均为11例R²改善、23例退化。

| 病例 | best ΔPa R²_cb贡献 | last贡献 | best高WSS MAE增加 Pa | best IoU变化 |
|---|---:|---:|---:|---:|
| ILO/SUN_XU_XIA-1/before | -0.01741478 | -0.01423086 | +3.4114 | +0.03920 |
| AG/fast/ZHANG_CHUN | -0.00610327 | -0.00542743 | +3.5716 | +0.00683 |
| AAA/unruputer/WANG_MAN_TIAN | -0.00365681 | -0.00277864 | +0.8190 | -0.05780 |

三例best贡献合计-0.02717486，整体为-0.03582426；last分别为-0.02243693/-0.02780487。每例贡献由MSE和同一真值方差计算，34例贡献之和已精确核回整体Δ。

best高WSS R² 0.21946→0.12090、top10幅值比0.61560→0.58047、p99比0.60993→0.58102。前两例热点IoU有所提高而Pa高值误差增加；当前偏差主要体现为高值幅度/相关性退化，不能仅用全局缩放或热点定位单指标解释。

## GPU短步训练诊断已完成

状态：`completed_short_step`。作业 13974 在同一设备、同一真实 8 病例固定 batch 上比较旧源码、旧源码重复、新源码日志关、新源码日志开、新源码配对初始化日志开五条独立优化轨迹。AMP/FP32 各完成 12 次参数更新，未改训练中的冻结源码。

- 五个重建模型初始张量和构造后的 RNG 状态一致；每步 CPU/CUDA RNG、FPS 全索引及逐调用 RNG 轨迹、DropPath 掩码、scaler 决策均一致。两种精度都是 12/12 更新；AMP scale 始终 65536，梯度范数有限，无跳步。
- 第 0 步 stem、全部 SA、SA 注意力和壁面分支输出逐位相同。包括旧源码自身重复在内，首次观察到的前向差异均在 `fp.0`；旧旧比较该模块最大绝对差为 AMP 0.00390625、FP32 1.90735e-6。
- `fp.0` 包含 PyG `knn_interpolate` 的 kNN 查询、距离平方倒数权重、加权特征/权重 scatter 求和与相除，再与 skip 特征拼接，经过两组 Linear + BatchNorm1d + ReLU。目前只钩取整个模块输出，不能直接指认某个 CUDA kernel，也未定位首个后向差异算子。

| 精度 | 第 11 步 head 最大绝对差：旧旧 | 旧新日志关 | 旧新日志开 | 旧新 paired 日志开 |
|---|---:|---:|---:|---:|
| AMP | 0.02343750 | 0.02246094 | 0.02642822 | 0.02963257 |
| FP32 | 0.01552486 | 0.01693654 | 0.01919773 | 0.01970811 |

这里的 head 差值是归一化预测空间的逐点差，不是 Pa R²。日志开关和 paired 路径没有显示独有的重建初始化/RNG 消耗差异，也没有不同的首处分叉模块；其差异量级与本次旧旧重复有重叠。每种精度只有一组旧旧重复，这不构成统计等价证明；保存的完整张量比较均以旧源码为参照，不能反推出日志开/关之间的直接张量最大差。

12 步重复使用同一 batch，不能代替历史 400 轮数据次序、AMP 行为、评估和选模的完整重放。**已经观察到同源码 GPU 重复分叉，但尚未证明它解释 best -0.03582426 / last -0.02780487 的最终 Pa R² 差距。** 当前审计仍不把具体归约、同步或其他训练因素列为已确认根因。

证据：[GPU详细分析](training_trajectory/analysis_job_13974.md)、[分析JSON](training_trajectory/analysis_job_13974.json)、[AMP原始轨迹](training_trajectory/job_13974_amp.json)、[FP32原始轨迹](training_trajectory/job_13974_fp32.json)。

不修改冻结源码、原结果、筛选门槛或正在运行的作业。继续完整保留历史M2与同期MO0两个参照，所有候选同时报告best/last和负结果。

证据：[完整JSON](contemporary_control_diagnosis.json)、[历史权重重评](smoke_13973/historical_m2_eval/metrics.json)、[兼容性预检](runtime_preflight.json)。

[训练曲线与34病例贡献图](contemporary_control_diagnosis.png) · [PDF](contemporary_control_diagnosis.pdf)。图件与输入哈希见 [control_plot_manifest.json](control_plot_manifest.json)。
