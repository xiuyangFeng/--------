# 任务A论文可视化与指标建议

> 上位文档：[实验设计总纲](../../实验设计总纲.md) / [任务A实验清单](./任务A实验清单.md) / [项目思路](../../paper_idea/项目思路.md)

## 1. 文档定位

本文件用于回答任务 A 在未来写论文时，应该保留哪些图、哪些表、哪些指标，以及当前 `training/` 代码里已经具备哪些可直接复用的可视化能力。

---

## 2. 当前 training 代码里已经存在的可视化能力

### 2.1 训练过程监控

当前训练代码并不只有命令行进度条，已经具备以下可视化或半可视化能力：

- `training/core/trainer.py`
  - batch 级 `tqdm` 进度条
  - 每个 epoch 的摘要打印
  - `SummaryWriter` 写入 TensorBoard 标量
  - `history.csv` 持续落盘
- `training/README.md`
  - 已明确训练产物中包含 `history.csv`
  - 已明确支持 TensorBoard、控制台摘要和 checkpoint 追踪

这部分更偏“训练监控”，适合看收敛是否稳定、不同模型是否过拟合，不直接等同于论文主图，但可以为论文训练曲线提供数据来源。

### 2.2 论文图表函数

`training/analysis/visualization.py` 已经提供了较完整的论文图表骨架：

- `scatter_pred_vs_true()`
  - 预测 vs 真值散点图，适合 `u / v / w / p`
- `plot_training_curves()`
  - 训练/验证损失与验证指标曲线
- `plot_regional_bar()`
  - 分区域误差条形图
- `plot_ablation_summary()`
  - 消融实验横向对比图
- `plot_error_distribution()`
  - 误差分布直方图
- `plot_error_cdf()`
  - 绝对误差 CDF
- `plot_per_case_boxplot()`
  - per-case 指标箱线图
- `plot_multi_model_curves()`
  - 多模型训练曲线对比

说明：

- 文件头注释里提到“误差热图”，但当前模块中还没有真正落地的 surface heatmap 绘图函数。
- 因此目前属于“主结果图已具备一半以上，病例表面热图和切片对比图仍需后续补上”。

---

## 3. 论文中建议保留的核心可视化

以下内容综合自 `任务A实验清单`、`实验设计总纲` 和当前训练代码能力。

### 3.1 必须保留的图

1. 主结果表
   - 对比 `MLP / GraphSAGE / Transformer / geometry 主模型`
   - 报告整体精度、效率、稳定性
2. 消融结果表
   - 输入组分、几何特征、physics、层次化结构等
3. 预测 vs 真值散点图
   - 至少覆盖 `u / v / w / p`
4. 典型病例流场切片对比图
   - 展示 CFD、AI 预测和误差图
5. 误差热图
   - 重点放近壁区、高曲率区、分叉区
6. per-case 箱线图
   - 展示病例间波动，避免只报整体平均
7. 分区域误差条形图
   - 至少分 `wall / interior`
   - 更推荐加入 `high_curvature / bifurcation / near_wall`

### 3.2 强烈建议保留的图

1. 训练曲线图
   - 用于说明训练稳定性和是否早停
2. 误差分布直方图
   - 看模型偏差是否集中、是否长尾
3. 误差 CDF 图
   - 看不同模型在“小误差覆盖率”上的差异
4. 多模型训练曲线对比图
   - 适合 baseline 之间做收敛速度比较
5. 精度-速度-显存 Pareto 图
   - 这是任务 A 很关键的一张图
6. hierarchy 深度 vs 推理时间折线图
   - 如果后续真的推进层次化建模，这张图很重要

### 3.3 适合写讨论部分的补充图

1. 高曲率局部放大图
2. 分叉区局部放大图
3. 最差病例可视化
4. 物理残差分布图

---

## 4. 论文中建议报告的指标

### 4.1 主精度指标

- `RMSE`
- `MAE`
- `R²`
- `RMSE_u`
- `RMSE_v`
- `RMSE_w`
- `RMSE_p`
- `RMSE_vel_mag`
- `MAE_vel_mag`

### 4.2 分层指标

不能只报全局平均，建议至少同时报告：

- 全部节点
- 壁面节点
- 内部节点
- 高曲率区域
- 分叉区域
- 近壁区域

### 4.3 效率指标

- 单个 snapshot 推理时间
- 单病例全时序推理时间
- 每 epoch 训练时间
- 峰值显存
- 参数量
- 相对 CFD 加速比

### 4.4 稳定性与物理一致性指标

- 多 seed `mean ± std`
- paired t-test / Wilcoxon
- Bootstrap 95% CI
- `continuity_loss`
- `momentum_loss`
- 近壁无滑移误差
- 质量守恒误差

---

## 5. 写论文时的推荐组织方式

### 5.1 主结果段

- 一张主表：整体精度 + 效率
- 一张 per-case 箱线图
- 一张典型病例可视化图

### 5.2 机理解释段

- 一张分区域误差条形图
- 一张高曲率/近壁误差图
- 一张 geometry 消融图

### 5.3 工程可用性段

- 一张训练曲线图
- 一张精度-速度-显存 Pareto 图

---

## 6. 完整出图规划

下面给出面向论文主文和补充材料的完整出图方案。建议把任务 A 作为整篇论文的前半段证据核心。

### 6.1 Figure A1：任务 A 主结果总表

回答的问题：

- 主模型是否在整体精度和效率上优于点级与基础图模型。

建议内容：

- 表格列固定为：
  - `Model`
  - `Graph`
  - `Geometry`
  - `BC`
  - `is_wall`
  - `Physics`
  - `RMSE_u`
  - `RMSE_v`
  - `RMSE_w`
  - `RMSE_|v|`
  - `RMSE_p`
  - `R2_p`
  - `Inference Time`
- 每个模型报告 `mean ± std`
- 至少包含：
  - `MLP`
  - `GraphSAGE`
  - `Transformer(no geometry)`
  - `Transformer(geometry)`

数据来源：

- `history.csv`
- `summary.json`
- `eval_field.py` 的 `metrics.json`

当前代码支撑：

- 指标统计已具备
- 表格需要人工汇总或后续补统一导表脚本

### 6.2 Figure A2：典型病例主可视化图组

回答的问题：

- AI 预测场在空间上是否真的接近 CFD，尤其是在复杂区域。

推荐版式：

- `2 x 3` 或 `3 x 3`
- 每列对应：
  - `CFD`
  - `Prediction`
  - `Absolute Error`
- 每行对应：
  - 全局切片或整体表面
  - 近壁区局部放大
  - 分叉区或高曲率区局部放大

建议固定：

- 同一物理量同一色条范围
- `CFD`、`Prediction`、`Error` 使用不同但固定的 colormap 语义
- 局部放大框在全局图中明确标出

数据来源：

- `predict_field.py` 导出的逐样本预测

当前代码支撑：

- 预测资产导出已具备
- 切片/表面图脚本仍需补

### 6.3 Figure A3：预测 vs 真值散点图

回答的问题：

- 四个目标变量是否整体拟合到位。

推荐版式：

- `2 x 2` 子图
- 分别为 `u / v / w / p`
- 每个子图角落统一标注：
  - `RMSE`
  - `MAE`
  - `R²`

数据来源：

- 测试集 `pred / target`

当前代码支撑：

- `training.analysis.visualization.scatter_pred_vs_true()`

### 6.4 Figure A4：病例级分布与稳健性图

回答的问题：

- 模型提升是否稳定存在于多数病例，而不是只靠少数样本拉高均值。

推荐版式：

- 左图：per-case `RMSE_|v|` 箱线图
- 中图：per-case `RMSE_p` 箱线图
- 右图：最差 3 个病例与最好 3 个病例的点图或 strip plot

建议：

- 主文用 boxplot
- 最差病例分析放补充材料

当前代码支撑：

- `training.analysis.visualization.plot_per_case_boxplot()`

### 6.5 Figure A5：分区域误差图

回答的问题：

- geometry 先验到底改善了哪些关键区域。

推荐版式：

- grouped bar chart
- 区域固定为：
  - `all`
  - `wall`
  - `interior`
  - `high_curvature`
  - `near_wall`
  - `bifurcation`
- 对比至少两组模型：
  - `Transformer(no geometry)`
  - `Transformer(geometry)`

建议：

- 主图放 `RMSE_|v|`
- 补充材料再放 `RMSE_p`

当前代码支撑：

- `training.analysis.regional_eval.compute_regional_metrics()`
- `training.analysis.visualization.plot_regional_bar()`

### 6.6 Figure A6：消融总结图

回答的问题：

- 性能提升来自哪些设计，而不是来自“模型更大”。

推荐版式：

- 横向条形图
- 消融组建议分两张：
  - 输入与几何消融
  - physics / hierarchy / 图结构消融

条形图上建议标：

- mean
- std 或 95% CI
- 显著性标记

当前代码支撑：

- `training.analysis.visualization.plot_ablation_summary()`
- `training.analysis.stats.*`

### 6.7 Figure A7：训练曲线与误差分布图

回答的问题：

- 主模型是否训练稳定，是否只是过拟合换来的结果。

推荐版式：

- 左：train / val loss 曲线
- 中：关键验证指标曲线
- 右：误差分布直方图或误差 CDF

建议：

- 训练曲线放主模型
- 多模型对比曲线放补充材料

当前代码支撑：

- `plot_training_curves()`
- `plot_error_distribution()`
- `plot_error_cdf()`
- `plot_multi_model_curves()`

### 6.8 Figure A8：精度-速度-显存图

回答的问题：

- 你的方法是不是不仅更准，而且更适合部署。

推荐版式：

- 横轴：single snapshot inference time
- 纵轴：`RMSE_|v|`
- 点大小：peak memory
- 点颜色：模型类别

如果后续做 hierarchy：

- 再补一张 `depth vs runtime` 折线图

当前代码支撑：

- 数值统计可从实验结果汇总
- 正式出图脚本仍需补

### 6.9 Supplementary A：建议放附录的图

- 全变量误差 CDF
- 最差病例个案图
- 物理残差分布图
- 多 seed 单独曲线
- 更多区域或更多病例切片

---

## 7. 任务A出图脚本规划清单

下面这部分不再停留在“该画什么图”，而是写成后续真正可以执行的出图脚本规划。目标是后面你要落地时，可以按这个清单逐个实现。

### 7.1 建议的输出目录结构

建议统一输出到单个 run 目录下，避免图表四处散落：

```text
outputs/field/<run_dir>/
├── eval_test/
│   └── metrics.json
├── predictions_test/
│   ├── manifest.json
│   └── *.pt
├── figures_taskA/
│   ├── fig_A1_main_table.csv
│   ├── fig_A3_scatter.png
│   ├── fig_A4_per_case_boxplot.png
│   ├── fig_A5_regional_bar.png
│   ├── fig_A6_ablation_summary.png
│   ├── fig_A7_training_curves.png
│   ├── fig_A7_error_distribution.png
│   ├── fig_A7_error_cdf.png
│   ├── fig_A8_pareto.csv
│   └── manifest_taskA_figures.json
└── ...
```

建议统一再保存一份：

- `manifest_taskA_figures.json`
  - 记录这次出图使用了哪个 checkpoint、哪个 split、哪个 subset、哪个病例、哪些图已经生成

### 7.2 Figure A1 主结果总表脚本

建议脚本名：

- `training/scripts/plot_taskA_main_table.py`

输入：

- 一个或多个 `eval_test/metrics.json`
- 对应 run 目录中的 `summary.json`
- 可选 `run_manifest.json`

输出：

- `fig_A1_main_table.csv`
- 可选 `fig_A1_main_table.md`

复用现有代码：

- `training/scripts/eval_field.py` 负责产生 `metrics.json`

脚本核心步骤：

1. 读取多个 run 的 `metrics.json`
2. 提取 `rmse_u / rmse_v / rmse_w / rmse_vel_mag / rmse_p / r2_p`
3. 从 `summary.json` 或其他产物中读取 runtime、参数量等字段
4. 统一整理成单表
5. 按论文主表顺序输出 CSV

当前缺口：

- 需要一个统一聚合多个 run 的导表脚本

### 7.3 Figure A2 典型病例图脚本

建议脚本名：

- `training/scripts/plot_taskA_case_panel.py`

输入：

- `predictions_test/manifest.json`
- 指定 `sample_id` 或 `case_name`
- 对应的 `*.pt` 预测文件

输出：

- `fig_A2_case_panel_<sample_id>.png`

复用现有代码：

- `training/scripts/predict_field.py` 已导出 `x / y_true / y_pred / edge_index / wall_mask`

脚本核心步骤：

1. 读取某个 sample 的 `.pt`
2. 选择目标变量，例如 `|v|` 或 `p`
3. 计算 `CFD`、`Pred`、`Abs Error`
4. 做三联图
5. 如果需要局部放大，再按 `Abscissa / NormRadius / Curvature` 或空间范围筛子区域

当前缺口：

- 当前仓库还没有切片图或表面图脚本
- 需要明确使用二维切片、三维散点还是表面投影

建议优先级：

- 第一版先做二维散点投影或规则切片
- 第二版再做更精细的表面热图

### 7.4 Figure A3 散点图脚本

建议脚本名：

- `training/scripts/plot_taskA_scatter.py`

输入：

- `predictions_test/*.pt` 或聚合后的 `pred/target` 数组

输出：

- `fig_A3_scatter.png`

复用现有代码：

- `training.analysis.visualization.scatter_pred_vs_true()`

脚本核心步骤：

1. 遍历 `predictions_test/*.pt`
2. 拼接全部 `y_true / y_pred`
3. 调用 `scatter_pred_vs_true()`
4. 保存图像

当前缺口：

- 只缺一个把 `*.pt` 聚合起来的薄脚本

优先级：

- 最高，最容易先落地

### 7.5 Figure A4 per-case 箱线图脚本

建议脚本名：

- `training/scripts/plot_taskA_per_case_boxplot.py`

输入：

- `predictions_test/*.pt`

输出：

- `fig_A4_per_case_boxplot.png`
- 可选 `fig_A4_per_case_metrics.json`

复用现有代码：

- `training.core.metrics.PerCaseMeter`
- `training.analysis.visualization.plot_per_case_boxplot()`

脚本核心步骤：

1. 遍历预测文件
2. 对每个 `case_name` 聚合 `pred / target`
3. 用 `PerCaseMeter` 计算 per-case 指标
4. 调用 `plot_per_case_boxplot()`

当前缺口：

- 需要一个直接读取预测导出并喂给 `PerCaseMeter` 的脚本

### 7.6 Figure A5 分区域误差图脚本

建议脚本名：

- `training/scripts/plot_taskA_regional_bar.py`

输入：

- `predictions_test/*.pt`

输出：

- `fig_A5_regional_bar_vel.png`
- 可选 `fig_A5_regional_bar_p.png`
- 可选 `fig_A5_regional_metrics.json`

复用现有代码：

- `training.analysis.regional_eval.compute_regional_metrics()`
- `training.analysis.visualization.plot_regional_bar()`

脚本核心步骤：

1. 逐文件读取 `x / y_true / y_pred`
2. 调 `compute_regional_metrics()`
3. 聚合多个 sample 的区域指标
4. 选定 `rmse_vel_mag` 或 `rmse_p`
5. 调 `plot_regional_bar()`

当前缺口：

- 需要定义“跨 sample 聚合”的口径
- 建议第一版先做 test subset 全部节点汇总

### 7.7 Figure A6 消融总结图脚本

建议脚本名：

- `training/scripts/plot_taskA_ablation_summary.py`

输入：

- 多个实验 run 的 `metrics.json`
- 多 seed 汇总结果

输出：

- `fig_A6_ablation_summary.png`
- `fig_A6_ablation_summary.csv`

复用现有代码：

- `training.analysis.visualization.plot_ablation_summary()`
- `training.analysis.stats.summarize_seeds()`
- `training.analysis.stats.compare_experiments()`

脚本核心步骤：

1. 按实验组读取不同 seed 的指标
2. 聚合 mean/std
3. 对主对照组做显著性检验
4. 调用 `plot_ablation_summary()`
5. 导出图和 CSV

当前缺口：

- 需要统一实验命名规则到脚本里

### 7.8 Figure A7 训练曲线与误差分布脚本

建议拆成两个脚本：

- `training/scripts/plot_taskA_training_curves.py`
- `training/scripts/plot_taskA_error_analysis.py`

输入：

- `history.csv`
- `predictions_test/*.pt`

输出：

- `fig_A7_training_curves.png`
- `fig_A7_error_distribution.png`
- `fig_A7_error_cdf.png`

复用现有代码：

- `plot_training_curves()`
- `plot_error_distribution()`
- `plot_error_cdf()`

脚本核心步骤：

- 训练曲线脚本：
  1. 读取 `history.csv`
  2. 转成 `history` dict
  3. 调 `plot_training_curves()`
- 误差分析脚本：
  1. 聚合 `y_true / y_pred`
  2. 调误差分布和 CDF 两个函数

当前缺口：

- 只缺 `history.csv` 到 dict 的转换和 `*.pt` 聚合脚本

### 7.9 Figure A8 精度-速度-显存脚本

建议脚本名：

- `training/scripts/plot_taskA_pareto.py`

输入：

- 多个 run 的：
  - `metrics.json`
  - `summary.json`
  - 手工记录或日志导出的 peak memory

输出：

- `fig_A8_pareto.png`
- `fig_A8_pareto.csv`

脚本核心步骤：

1. 汇总每个模型的 `RMSE_|v|`
2. 汇总 runtime
3. 汇总 peak memory
4. 作二维散点图，点大小编码 memory

当前缺口：

- 当前项目里还没有统一的 peak memory 汇总产物
- 这部分需要先约定显存记录口径

### 7.10 建议的执行顺序

建议不要同时开所有出图脚本，按下面顺序推进：

1. `plot_taskA_scatter.py`
2. `plot_taskA_training_curves.py`
3. `plot_taskA_error_analysis.py`
4. `plot_taskA_per_case_boxplot.py`
5. `plot_taskA_regional_bar.py`
6. `plot_taskA_main_table.py`
7. `plot_taskA_ablation_summary.py`
8. `plot_taskA_case_panel.py`
9. `plot_taskA_pareto.py`

原因：

- 前 5 个几乎都能直接复用现有 `training` 模块
- 后 4 个更依赖实验组织、病例挑选或统一资源统计

### 7.11 第一批最值得先落地的脚本

如果只做最小闭环，建议先实现这 5 个：

1. `plot_taskA_scatter.py`
2. `plot_taskA_training_curves.py`
3. `plot_taskA_per_case_boxplot.py`
4. `plot_taskA_regional_bar.py`
5. `plot_taskA_main_table.py`

这 5 个一旦完成，任务 A 的论文主图已经能支撑：

- 主表
- 散点图
- 箱线图
- 分区域图
- 训练曲线

### 7.12 脚本实现时的统一规范

每个出图脚本建议统一支持：

- `--input-dir`
- `--output-dir`
- `--subset`
- `--title`
- `--format png`
- `--dpi 300`

每个脚本都建议额外输出：

- 对应的 `.json` 或 `.csv`
- 一份小的 `manifest` 记录图对应的数据来源

这样后面论文返修时，能知道每张图到底是用哪次 run 生成的。

---

## 8. 当前状态判断

- 任务 A 的训练侧已经有较明确的可视化基础设施，不是“完全没有图可做”的状态。
- 当前最成熟、可直接复用的图是：
  - 散点图
  - 训练曲线
  - per-case 箱线图
  - 分区域条形图
  - 消融总结图
  - 误差分布图
  - 误差 CDF 图
- 当前仍缺、但论文几乎肯定需要补的图是：
  - 典型病例切片对比图
  - 表面误差热图
  - 精度-速度-显存 Pareto 图的正式出图脚本

---

## 9. 与后续任务的关系

- 任务 A 是任务 B 的前提。
- 如果任务 A 只证明“整体 RMSE 不错”，但没有区域图和病例图，后面任务 B 很难证明 AI 场在关键区域是可靠的。
- 因此任务 A 的可视化不只是为本任务服务，也是后续任务 B、D 的上游证据。
