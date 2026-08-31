# Field 预测产物保留与可再生性规范

> 生效日期：2026-08-30  
> 适用范围：`training/scripts/predict_field.py` 及 `outputs/field/`

## 1. 默认导出口径

`predict_field.py` 默认使用 `--payload-mode compact`，保存：

- `sample_id` / `case_name` / `graph_path`
- `wall_mask` / `time_value`
- `y_true` / `y_pred`
- 可选 `y_wss_true` / `y_wss_pred`

compact 不再每帧重复保存 `x/global_cond/edge_index`。常规误差分析、regional
评估和 WSS 指标应优先使用 compact。

## 2. 何时使用 full

仅在以下情况显式使用 `--payload-mode full`：

- 任务 B / CFD 对照产物必须脱离原始图资产独立传递；
- 脚本直接需要 payload `x` 或 `edge_index`；
- 需要冻结一份不依赖源图的完整交付包。

full 导出后应在同一阶段完成紧凑评估，不应将多个 5–6 GiB 的全量导出长期并排保留。

## 3. run 必须保留的复现证据

- `config.snapshot.json`
- `split.snapshot.json`（如存在）
- `run_manifest.json`
- `summary.json`
- `history.csv`
- 结论依赖的 checkpoint
- prediction `manifest.json`
- 紧凑 `evaluation/`、图件和病例级/区域级指标

删除逐样本预测前，必须确认 config 可解析、checkpoint 可加载、数据资产与
split 可访问，并用代表样本执行数值重算。

## 4. 图资产发生变化时

如原 checkpoint 的输入图已被重建或覆盖，禁止直接删除旧预测。应先：

1. 构建去重 legacy input snapshot；
2. 在当前预测入口中提供显式 snapshot loader；
3. 用当前代码重算并通过指定容差；
4. 保留 snapshot manifest/hash 和验证记录后再清理旧导出。

2026-08-30 建立的快照与执行记录位于
`outputs/field/_reproducibility/`。

## 5. 目录归档

- 当前代码不兼容但有历史证据的 run：移入 `outputs/field/_archive/legacy_pre_wss_schema/`。
- 无 summary/checkpoint 的失败启动：移入按日期命名的 `aborted_starts_*` 目录。
- 空目录：可直接删除。
- 移动有文档或脚本引用的 run 后，必须同步修正路径和 `experiment_index.csv`。

