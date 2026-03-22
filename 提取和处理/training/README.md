# Training

这个目录是独立于 `pipeline/` 的训练脚手架，优先服务任务 A 的场重建实验。

## 目录结构

```text
training/
├── core/                    核心训练组件
│   ├── config.py            实验配置 dataclass
│   ├── data.py              图数据集、增强和特征掩码
│   ├── models.py            MLP / GraphSAGE / Transformer 模型注册
│   ├── losses.py            训练损失（加权 MSE、物理约束）
│   ├── metrics.py           回归指标
│   ├── trainer.py           训练循环和早停
│   ├── splits.py            患者级数据划分读取
│   ├── io.py                checkpoint 与实验索引落盘
│   └── utils.py             通用工具函数
│
├── scripts/                 可执行入口脚本
│   ├── train_field.py       任务 A 训练入口
│   ├── eval_field.py        独立评估入口
│   ├── predict_field.py     批量预测导出入口
│   ├── export_hemo.py       任务 B 指标导出入口
│   ├── make_field_plan.py   根据任务 A 实验清单批量生成配置
│   ├── run_field_plan.py    按 manifest 顺序批量执行训练
│   └── make_split.py        根据病例名单生成患者级 split
│
├── analysis/                分析评估与可视化
│   ├── hemo.py              任务 B 血流动力学指标计算
│   ├── plan.py              任务 A baseline / ablation 计划模板
│   ├── regional_eval.py     区域级评估
│   ├── stats.py             统计检验（配对 t/Wilcoxon/Bootstrap CI）
│   ├── benchmark.py         推理性能基准
│   ├── sensitivity.py       敏感性分析配置
│   └── visualization.py     可视化工具
│
├── configs/field/           实验配置 JSON 模板
│   ├── ablations/           消融模板
│   └── generated/           批量生成的配置
│
├── splits/                  split 样例模板
│
└── cluster/                 集群提交脚本
    ├── run_train_field.slurm
    ├── run_plan.slurm
    ├── run_array.slurm
    └── README.md
```

## 设计原则

1. 不修改现有 `pipeline/` 数据处理流程。
2. 所有特征消融通过"特征掩码"实现，不改图文件结构。
3. 第一阶段只服务任务 A，先把 `MLP / GraphSAGE / Transformer` 跑通。
4. 后续加 physics loss、任务 B、任务 C 时直接扩展这套骨架，但 physics 默认走"CFD 监督主干 + 分阶段物理正则"，不走纯 PINN 起步。

当前已经补到：

- 可插拔 physics loss 接口
- 任务 B 指标计算与导出骨架
- 运行时最小字段装载：训练时只保留模型和 loss 真正会访问的 `Data` 字段，减少大图 `.pt` 文件带来的无效内存占用

physics 使用建议：

1. 先跑稳定的 `data-only` baseline。
2. 再开 `continuity_loss`，确认不会拖垮验证集数据误差。
3. 再加 `no_slip_loss` 这类边界正则。
4. `momentum_loss` 最后上，并始终配合 warmup 与分项监控。

## 使用方式

先准备一个患者级 split JSON，再选择一个配置模板：

如果你还没有整理好"哪些病例完整可用"，建议先从原始数据导出一份病例名单：

```bash
python -m pipeline.audit_inputs \
  --sources AG/fast \
  --report-name raw_input_audit_ag_fast \
  --ready-cases-output training/splits/case_names_ag_fast_ready.txt \
  --require-named-stl
```

然后用这份名单生成患者级 split：

```bash
python -m training.scripts.make_split \
  --cases-file training/splits/case_names_ag_fast_ready.txt \
  --output training/splits/split_v1.json \
  --split-version split_v1 \
  --source AG/fast
```

```bash
python -m training.scripts.train_field --config training/configs/field/transformer_geometry.json
```

如果要按 `docs/01-任务/任务A/任务A实验清单.md` 批量生成配置，可以先运行：

```bash
python -m training.scripts.make_field_plan \
  --data-root data_new/AG/fast \
  --split-file training/splits/split_example.json \
  --output-dir training/configs/field/generated
```

默认会生成 baseline、输入特征消融、几何分量消融、增强策略消融，并为每个实验生成 `seed = [1, 2, 3]` 的配置文件。坐标归一化消融可额外通过 `--coord-variant name=subdir` 指定。

如果要按生成顺序批量执行，可以继续使用：

```bash
python -m training.scripts.run_field_plan \
  --manifest training/configs/field/generated/manifest.json \
  --study-group baseline
```

先验证命令而不实际执行，可加 `--dry-run`。

任务 B 指标导出骨架可直接接 `predict_field.py` 的输出：

```bash
python -m training.scripts.export_hemo \
  --manifest outputs/field/<run_dir>/predictions_test/manifest.json \
  --source AI
```

如果要用同一批导出里的 CFD 真值场做对照，只需要把 `--source` 改成 `CFD`。

如果本机没有数据、只是在本地准备脚手架，可以先在服务器上根据病例名单生成 split：

```bash
python -m training.scripts.make_split \
  --cases-file /path/to/case_names.txt \
  --output training/splits/split_v1.json \
  --split-version split_v1 \
  --source AG/fast
```

独立评估：

```bash
python -m training.scripts.eval_field \
  --config training/configs/field/transformer_geometry.json \
  --checkpoint outputs/field/<run_dir>/best_model.pt \
  --subset test
```

批量导出预测结果：

```bash
python -m training.scripts.predict_field \
  --config training/configs/field/transformer_geometry.json \
  --checkpoint outputs/field/<run_dir>/best_model.pt \
  --subset test
```

如果已经导出了 `predictions_test/manifest.json`，可以继续直接生成误差分析图：

```bash
python -m training.scripts.plot_error_analysis \
  --manifest outputs/field/<run_dir>/predictions_test/manifest.json
```

默认会在 `predictions_test/error_analysis/` 下生成：

- `fig_scatter_pred_vs_true.png`
- `fig_error_distribution.png`
- `fig_error_cdf.png`
- `fig_per_case_boxplot.png`
- `per_case_metrics.csv`
- `summary.json`

如果你想按任务 A 论文图的命名拆开生成，也可以直接用这些脚本：

```bash
python -m training.scripts.plot_taskA_scatter \
  --manifest outputs/field/<run_dir>/predictions_test/manifest.json

python -m training.scripts.plot_taskA_per_case_boxplot \
  --manifest outputs/field/<run_dir>/predictions_test/manifest.json

python -m training.scripts.plot_taskA_regional_bar \
  --manifest outputs/field/<run_dir>/predictions_test/manifest.json

python -m training.scripts.plot_taskA_main_table \
  --runs-root outputs/field

python -m training.scripts.plot_taskA_ablation_summary \
  --runs-root outputs/field \
  --metric-key rmse_vel_mag
```

如果你已经把旧训练结果拷回本地 `outputs/field/`，可以直接基于 `history.csv` 出训练曲线和对比图：

```bash
python -m training.scripts.plot_training_history \
  --runs-root outputs/field
```

默认行为：

- 为每个 run 目录生成 `fig_training_curves.png`
- 在 `outputs/field/plots/` 下生成 `best_metrics.csv`
- 如果匹配到 2 个及以上 run，还会额外生成 `compare_val_loss.png`

如果只想画部分实验，可用 `--pattern` 或 `--run-dir`：

```bash
python -m training.scripts.plot_training_history \
  --runs-root outputs/field \
  --pattern 'field_transformer*/history.csv' \
  --pattern 'field_graphsage*/history.csv' \
  --compare-metric val_rmse
```

安装依赖可参考：

```bash
pip install -r training/requirements.txt
```

如果服务器使用 GPU，请按实际 CUDA 版本单独安装匹配的 `torch` 和 `torch_geometric` 轮子，不要机械照搬 CPU 默认安装。

如果在集群上出现“进程被系统直接 kill，而不是 Python 抛 CUDA OOM”的情况，优先检查主机 RAM 压力。当前训练入口已经会按模型自动裁掉运行时不需要的图字段：

- `MLP` 默认只保留 `x / y / global_cond`
- 图模型默认保留 `x / y / global_cond / edge_index`

这类裁剪不会改变实验定义，只是避免把 `.pt` 图中的冗余属性反复读入、拼 batch 和搬到设备。

如果需要在集群上批量提交训练，可直接参考：

- `training/cluster/README.md`
- `training/cluster/run_train_field.slurm`
- `training/cluster/run_plan.slurm`
- `training/cluster/run_array.slurm`

## split 文件示例

见：

- `training/splits/split_example.json`

注意：

1. 当前仓库本机目录里没有实际训练数据，因此这里没有生成真实的患者划分文件。
2. 真实 split 必须在服务器上根据实际病例名单生成或手工整理。
3. `data_root` 需要改成服务器上与 `pipeline/README.md` 一致的数据路径。

## 训练产物

每次 `train_field.py` 训练结束后，默认会落盘：

- `config.snapshot.json`
- `split.snapshot.json`
- `history.csv`
- `summary.json`
- `run_manifest.json`
- `best_model.pt`
- `last_model.pt`
- `outputs/field/experiment_index.csv`

其中 `run_manifest.json` 会附带 `exp_id / study_group / feature_set / enabled_features` 等元数据，便于后续按 `docs/00-规范与记录/实验记录填写规范.md` 追踪实验。

训练过程中，控制台会额外显示：

- 训练和验证阶段的 batch 级 `tqdm` 进度条，包含当前 epoch、batch 进度和最近一个 batch 的 loss
- 每个 epoch 结束后的摘要行：`train_loss / val_loss / lr / best_val / patience`
- 当验证集出现新的最优结果时，会明确打印 `已保存 best_model.pt`

如果 `save_best_only=false`，仍会继续按 `save_every` 生成周期 checkpoint；这不会影响 `best_model.pt` 和 `last_model.pt` 的保存。

如果启用了 physics 配置，`history.csv` 还会额外记录：

- `train_data_loss / val_data_loss`
- `train_continuity_loss / val_continuity_loss`
- `train_momentum_loss / val_momentum_loss`
- `train_no_slip_loss / val_no_slip_loss`

实践上更推荐这样解读这些字段：

- `data_loss` 决定模型是否真的更接近 CFD 真值。
- `continuity_loss` 适合做第一层物理一致性筛查。
- `no_slip_loss` 适合检查壁面条件是否被破坏。
- `momentum_loss` 只有在前面几项稳定后才值得比较，否则很容易成为噪声源。

配合 `training.scripts.plot_training_history` 时，通常优先观察：

- `val_loss`：总体验证收敛情况
- `val_rmse`：整体回归误差
- `val_rmse_vel_mag`：速度模长误差
- `val_rmse_p`：压力误差
- `lr`：学习率是否进入衰减阶段
- `is_best`：最佳 epoch 是否稳定出现，而不是偶然抖动

## 第一批模板配置

- `mlp_baseline.json`
- `graphsage_baseline.json`
- `transformer_no_geometry.json`
- `transformer_geometry.json`

第一批常用消融模板：

- `ablations/transformer_coords_t_bc.json`
- `ablations/transformer_no_bc.json`
- `ablations/transformer_no_is_wall.json`
- `ablations/transformer_physics_full.json`
