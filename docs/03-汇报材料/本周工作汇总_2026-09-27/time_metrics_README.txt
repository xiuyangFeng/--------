时间建模机器可读表（2026-09-27）

主表 time_metrics.tsv/json：主比较采用同期标准 T-null（0.4616/0.2218/约0.696/0.713），与 T0/TB/锚定臂/Transformer 同合同。T0/TB 的 TAWSS MAE 来自 stage2_report_best.json 均值；TT 四臂只持久化 TAWSS MAE，未保存全周期 MAE，因此 full_cycle_mae_pa=null。

time_phase_metrics.tsv/json：逐帧 phase_metric_suite_best.md 原口径（T-null 0.462/0.222/0.688/峰值窗0.700）保留用于峰值窗/谷底/全周期 MAE、Pearson r²、Spearman、CCC、approx.disparity 对照。它不是主表同步 T-null，不能混行比较。
