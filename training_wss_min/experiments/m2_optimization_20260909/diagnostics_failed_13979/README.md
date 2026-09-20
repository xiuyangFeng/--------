# Slurm 13979 诊断失败归档

该目录由首次 `diagnostics/` 整体改名而来，保留原失败 `provenance.json` 及失败时的诊断源码快照。原 JSON 内的诊断产物绝对路径仍指向当时的 `diagnostics/`；对应的旧产物现位于本目录相同的相对位置。原记录未重写，避免混淆失败与重跑结果。

失败仅发生在 MO-P4/best、ZHANG_YONG_ZHI 的归一化 `pred_above_one_fraction` 复算：57531 个点中严格大于 1 的计数相差 1。原训练和正式 best/last 评估结果未修改。该次失败没有保存此病例的原始预测数组；后续重复推理证据来自独立 Slurm 13980，不能声称直接恢复了该次失败的逐点预测。

[离散边界审查](../discrete_fraction_audit.json) · [独立重复推理](../threshold_probe/job_13980/probe.json) · [重跑提交记录](../diagnostic_retry_submission.json)
