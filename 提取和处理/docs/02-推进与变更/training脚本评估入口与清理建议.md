# training 脚本评估入口与清理建议

> 日期：2026-05-08  
> 背景：V3 WSS 进入“可信性评估”阶段，需要训练后自动产出点级、区域级、病例级和 high-WSS 区域指标；同时 `training/scripts` 中历史脚本较多，需要明确哪些保留、哪些归档。

---

## 1. 新增统一评估入口

### 1.1 单 run 完整评估

新增脚本：

```bash
python -m training.scripts.evaluate_field_run_full \
  --run-dir outputs/field/<run_dir> \
  --checkpoint auto \
  --subset test
```

默认行为：

1. `checkpoint=auto` 时优先使用 `best_wss_model.pt`，不存在则使用 `best_model.pt`。
2. 调用 `predict_field` 生成预测 manifest。
3. 调用 `plot_taskA_scatter`、`plot_taskA_per_case_boxplot`、`plot_error_analysis --wss`、`plot_taskA_regional_bar --wss`。
4. 调用 `evaluate_wss_credibility` 生成 direct WSS-head 可信性评估。
5. 输出总索引：`evaluation/<subset>_<checkpoint>/evaluation_summary.json`。

可选项：

- `--run-derived-hemo`：额外运行 `export_hemo --source AI/CFD`，用于速度场派生 WSS/TAWSS/OSI/RRT；该指标与 direct WSS-head 输出分开存放。
- `--skip-predict`：复用已有 `predictions_*`。
- `--force`：清理旧输出后重跑。

### 1.2 Direct WSS-head 可信性评估

新增脚本：

```bash
python -m training.scripts.evaluate_wss_credibility \
  --manifest outputs/field/<run_dir>/predictions_test/manifest.json
```

输出：

- `wss_point_metrics.json`：壁面 `WSS magnitude` 点级 `RMSE / MAE / R² / Pearson / Spearman`
- `wss_case_metrics.csv`：病例级 `mean / p95 / max` 与每病例点级误差
- `wss_case_correlation.json`：病例级 `mean / p95 / max` 的相关性
- `wss_region_metrics.csv`：按 `regional_eval` 区域口径汇总 direct WSS-head 指标
- `high_wss_overlap.json`：true/pred top 5% 与 top 10% high-WSS 区域 overlap / Dice / Jaccard
- 图件：`fig_wss_mag_scatter.png`、`fig_case_p95_scatter.png`、`fig_case_mean_scatter.png`、`fig_high_wss_overlap_bar.png`

注意：

- 该脚本读取 `predict_field` 导出的 `y_wss_true/y_wss_pred`，评估的是 WSS head 直接输出。
- `export_hemo.py` 评估的是从速度场派生的 WSS，两者不可混用。
- 部分旧 manifest 中保存了集群绝对路径；新脚本会在原路径不存在时回退到 manifest 同目录下同名 `.pt`。

---

## 2. 已调整的现有图件口径

`plot_taskA_regional_bar --wss` 的默认 WSS 区域图从分量 RMSE 优先改为：

1. `r2_wss`
2. `rmse_wss`
3. `mae_wss`
4. `r2_wss_z`
5. `r2_wss_x`
6. `r2_wss_y`

理由：当前论文目标更关注 `WSS magnitude` 与下游可信性，分量指标放在补充分析更合适。

---

## 3. 脚本清理建议

### 3.1 主链路必须保留

这些脚本仍是训练、预测、评估或论文图件主入口：

- `train_field.py`
- `predict_field.py`
- `evaluate_field_run_full.py`
- `evaluate_wss_credibility.py`
- `recompute_dual_test_metrics.py`
- `plot_error_analysis.py`
- `plot_taskA_scatter.py`
- `plot_taskA_per_case_boxplot.py`
- `plot_taskA_regional_bar.py`
- `plot_taskA_main_table.py`
- `plot_taskA_ablation_summary.py`
- `plot_taskA_multimodel_scatter.py`
- `plot_taskA_multimodel_per_case_boxplot.py`
- `plot_taskA_multimodel_regional_bar.py`
- `export_hemo.py`
- `compare_hemo_wss_runs.py`
- `run_field_plan.py`
- `make_field_plan.py`
- `make_split.py`

### 3.2 兼容别名，可保留但不再扩展

这些脚本只是转发到新名字，保留是为了兼容旧命令；后续不建议继续在其中加逻辑：

- `plot_taskA_error_analysis.py` → `plot_error_analysis.py`
- `plot_taskA_training_curves.py` → `plot_training_history.py`

### 3.3 一次性历史分析脚本，建议迁入 `training/scripts/archive/`

这些脚本通常服务于某一次实验对照或诊断，继续放在根目录会干扰主链路视图：

- `plot_p0_1_A_Opt01_vs_Main01.py`
- `regenerate_opt03_vs_opt03w_figures.py`
- `regenerate_opt07_vs_opt05_main_figures.py`
- `regenerate_p02_warmup_comparison_figures.py`
- `_eval_wss_metrics_once.py`
- `analyze_train_wall_wss_distribution.py`
- `run_v3_ood_per_case_diag.py`

建议先移动到 `training/scripts/archive/`，并在文件头保留原用途说明；不要立即删除。

### 3.4 诊断/benchmark 类，保留但标注非主链路

- `run_v3_diag00.py`：V3 阶段诊断仍有追溯价值，但不是训练后评估主入口。
- `run_efficiency_benchmark.py`：效率实验可保留。
- `plot_taskA_case_panel.py`：病例可视化仍有论文图潜力，但不属于每次 run 必跑。

### 3.5 cluster 脚本建议

保留通用入口：

- `run_train_field.slurm`
- `run_field_predict_test_array.slurm`
- `run_wss_multitask_predict_figs_array.slurm`
- `run_recompute_dual_test_metrics.slurm`
- `run_compare_hemo_wss.slurm`
- `run_plan.slurm`
- `run_array.slurm`

建议归档历史清单与专用脚本：

- `manifest_list_v3_*_predict.tsv`
- `manifest_list_v2p_wssp05_06_predict.tsv`
- `manifest_list_A-Opt-05-*`
- `wss_runs_A_Opt03_vs_Opt05tune_seed1.tsv`
- `run_train_field_gpu013.slurm`
- `run_v3_diag00*.slurm`
- `run_v3_ood_diag.slurm`

---

## 4. 清理原则

1. **先归档，不删除**：实验代码是论文可追溯证据，除非确认无人引用，否则不要直接删。
2. **主链路集中到统一入口**：后续单模型评估优先使用 `evaluate_field_run_full.py`。
3. **direct WSS 与 derived hemo 分开**：V3 P+WSS 主线报告优先使用 direct WSS-head 指标。
4. **旧脚本只读维护**：历史对照脚本不再继续加新功能，避免逻辑分叉。
