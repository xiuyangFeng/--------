# Slurm 13981 诊断失败归档

本目录由第二次 `diagnostics/` 整体改名而来。原 `provenance.json` 未修改，其中诊断产物路径仍使用当时的目录名；对应旧产物现位于本目录相同的相对位置。失败时诊断源码保存在 `diagnose_source_failed_13981.py`，SHA 与原记录一致。

MO-P2/last、`ILO/ZHANG_JIN_CHUN-1/before` 的 Pa 空间 `pred_above_one_fraction` 复算失败：74402 个壁面点中，正式评估的严格大于 1 Pa 计数为 47993，本次保存数组为 47994；其余 Pa 和 log_z 指标通过原数值容差。

原始预测已保存。最近边界的顶点索引为 42963，原节点 ID 为 1247991，归一化预测为 -0.4929712414741516，反归一化后为 1.000000000099064 Pa。相邻向下的 float32 归一化数值映射到 0.9999999613575433 Pa；此处讨论的是归一化坐标的 ULP，不能称为 Pa 输出的一个 float32 ULP。

后续完整捕获固定在一次推理所得数组上进行审查和绘图，不修改正式评估、训练结果或筛选门槛。

Slurm 13983 的 32 次重复推理复现了 47993 和 47994 两种计数，唯一跨阈值顶点仍为 42963。重复预测在该顶点出现五个归一化数值，因此不能把实际变化概括为“恰好一个 ULP”；上面的相邻值计算只说明阈值附近的数值分辨率。

[32 次重复证据](../threshold_batch_probe/job_13983/probe.json) · [诊断作业记录](../diagnostic_attempts.json) · [完整捕获](../diagnostics_collection/provenance.json) · [最终诊断](../diagnostics/provenance.json)
