# 时间建模全周期 MAE 补评估

范围：用户2026-09-27授权复用已保存权重、缓存补齐13种时间模型的全周期MAE，并再次提供表格。后续用户明确允许node04 GPU1短时与既有训练共享。本目录独立于原run、缓存、配置及冻结源码；没有训练、调参、换checkpoint或访问新划分。

## 计算合同

- V5.1原cv3留出病例，三折46/46/44例，seed1234，每例全部壁面点、81帧。
- 真值使用原始Pa值，预测沿用原实现`clip(exp(ln_pred)-eps,0,None)`。
- 先在每例每帧内对点求绝对误差均值，再在病例及81帧间求均值，最后三折算术平均。
- 峰值、谷底和TAWSS MAE另外保存在JSON。全周期MAE和TAWSS MAE不同。
- T0/TB8/TB16从原`ckpt_best/metrics.json`逐帧`physical.field_casebalanced.mae`提取，未重新推理。此前汇报的0.513/0.547/0.546来自点池化MAE，本次统一病例等权为0.5242/0.5639/0.5626。
- 其余10种方法恢复原last权重：冻结时间头9个、TL6个、TT12个checkpoint；T-null3折使用原缓存及统计公式。冻结源码逐文件SHA256与原metrics记录一致，split/帧统计/cache manifest指纹核验通过。

## 验收

- 以原run的全周期/谷底/峰值/TAWSS R²及TAWSS MAE作为复现锚点，绝对误差容差在计算前固定为5e-5。
- 流式累积器在首例独立复现原`cycle_report`，误差小于1e-10；锚定峰值逐点偏差须小于1e-8。
- CPU Transformer与冻结时间头通过复现。TL-warm CPU试算谷底R²偏差0.00145，未通过；保留于`failed_cpu/`，不进入正式表。得到共享GPU授权后，TL两臂恢复至原A100复核。
- CPU使用node03 Slurm，GPU只使用已授权node04物理卡1。GPU任务串行，完成后进程自然退出；不终止或修改其他任务。

## 产物

- `evaluate_saved.py`：只评估的恢复程序，按实验族导入各自原冻结代码。
- `existing_T0_TB_metrics.json`：9个既有折的逐帧MAE提取及来源hash。
- `results/`：30个新补评估折的指标、复现差、checkpoint/source指纹及逐病例误差。
- `execution_manifest.json`、`logs/`：执行资源、作业、授权及日志。
- `aggregate.py`：只有全部39个折结果完整且验收通过后才能生成表格。
- `summary.json`、`时间建模_MAE补齐.md`、`时间建模_MAE补齐.tsv`：最终完整数值与可复制表格。

原表部分R²为近似摘要；最终表回到实际run，末位可能有修正。不同历史批次预算和选模规则不同，指标表不能解释成单变量结构消融。
