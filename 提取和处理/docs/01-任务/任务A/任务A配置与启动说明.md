# 任务A配置与启动说明

> 配套文档：[任务A实验清单](任务A实验清单.md) / [任务A本周实验启动清单](任务A本周实验启动清单.md) / [training/README.md](../../../training/README.md)

这份文档只回答 3 件事：

1. 任务 A 当前配置文件放在哪里。
2. 配置命名和输出命名分别是什么意思。
3. 本周应该用哪些命令先跑起来。

---

## 1. 先看当前仓库已经有什么

当前训练脚手架已经具备下面这些内容，不需要你重新搭一套：

- 训练入口：`training/train_field.py`
- 独立评估入口：`training/eval_field.py`
- 批量生成任务 A 配置：`training/make_field_plan.py`
- 按 manifest 批量执行：`training/run_field_plan.py`
- 第一批配置模板目录：`training/configs/field/`
- 已生成的任务 A 配置目录：`training/configs/field/generated/`

当前已经覆盖的实验组：

- `baseline`
- `input`
- `geometry`
- `augment`

也就是说，你这周要跑的 `A-Base-01` 到 `A-Main-01`，以及后续第一批 `A-Abl-01`，训练脚手架都已经有对应配置生成逻辑。

---

## 2. 配置文件在哪里

### 2.1 手写模板

这几份是基础模板：

- [mlp_baseline.json](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/training/configs/field/mlp_baseline.json)
- [graphsage_baseline.json](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/training/configs/field/graphsage_baseline.json)
- [transformer_no_geometry.json](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/training/configs/field/transformer_no_geometry.json)
- [transformer_geometry.json](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/training/configs/field/transformer_geometry.json)

如果你只想先单独验证一个实验，可直接从这些模板起跑。

### 2.2 批量生成后的配置

批量生成后的配置在这里：

- [training/configs/field/generated](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/training/configs/field/generated)

当前目录结构是按实验组分的：

- `baseline/`
- `ablation_input/`
- `ablation_geometry/`
- `ablation_augment/`

每个 JSON 文件基本遵循：

- `{Exp ID}_seed{seed}.json`

例如：

- `A-Base-01_seed1.json`
- `A-Main-01_seed3.json`
- `A-Abl-01-04_seed2.json`

---

## 3. 配置命名规则怎么读

你需要同时看懂 4 个名字：

1. `exp_id`
2. `experiment_name`
3. `study_group`
4. `feature_set`

### 3.1 `exp_id`

这是你在论文、文档、结果表里使用的主编号。

例如：

- `A-Base-01`
- `A-Base-02`
- `A-Base-03`
- `A-Main-01`
- `A-Abl-01-01`

这个字段是你后续写表格时最重要的主键。

### 3.2 `experiment_name`

这是训练运行时更贴近配置内容的英文名，主要服务于产物目录、日志和程序追踪。

例如：

- `field_mlp_coord_t_bc`
- `field_graphsage_coord_t_bc_wall`
- `field_transformer_coord_t_bc_wall`
- `field_transformer_coord_t_bc_geom_wall`

读法很直接：

- `field`：任务 A 场重建
- `mlp / graphsage / transformer`：模型
- `coord_t_bc...`：启用了哪些特征组

### 3.3 `study_group`

这是实验属于哪一组：

- `baseline`
- `ablation_input`
- `ablation_geometry`
- `ablation_augment`

批量运行时可以用这个字段筛选。

### 3.4 `feature_set`

这是压缩后的特征组合名，用于后续汇总与索引。

例如：

- `coord_t_bc_point`
- `coord_t_bc_wall`
- `coord_t_bc_geom_wall`
- `coord_t_bc_geom`

这个字段最适合拿来检查“两个实验到底差在哪个特征组”。

---

## 4. 每个配置文件里最重要的字段

一个任务 A 配置主要看这几块：

- `run`
- `data`
- `model`
- `optim`
- `system`
- `meta`

### 4.1 `run`

主要决定输出目录和保存策略。

关键字段：

- `experiment_name`
- `output_root`
- `save_every`
- `save_best_only`

### 4.2 `data`

这是最容易出错的部分。

你本周开始前至少要核对：

- `data_root`
- `split_file`
- `graphs_subdir`
- `enabled_node_features`
- `enabled_global_features`
- `augment`
- `augment_config`

其中最关键的是后两个特征列表，因为它们直接决定你这次实验到底在比较什么。

当前训练入口还会按 `model.name` 自动裁掉运行时不用的图字段：

- `mlp` 只保留 `x / y / global_cond`
- 图模型保留 `x / y / global_cond / edge_index`

这不会改变实验语义，只是降低大图训练时的主机内存占用。

### 4.3 `model`

当前任务 A 第一阶段主要是：

- `mlp`
- `graphsage`
- `transformer`

建议本周不要同时改：

- `hidden_dim`
- `num_layers`
- `heads`

否则 baseline 的归因会被打乱。

### 4.4 `optim`

当前计划里 baseline 和第一批消融默认共用同一套训练预算。

建议本周固定，不要随实验来回改：

- `epochs`
- `lr`
- `weight_decay`
- `early_stopping_patience`
- `target_weights`

### 4.5 `system`

本周最关键的是：

- `seed`
- `device`
- `deterministic`

### 4.6 `meta`

这一段不是为了训练，而是为了后面追踪实验。

其中最有用的是：

- `exp_id`
- `study_group`
- `feature_set`
- `ablation_axis`
- `question`

如果后面结果对不上，优先先看 `meta` 有没有写对。

---

## 5. 本周推荐的启动方式

### 5.1 如果你只想先单跑一个配置

直接跑：

```bash
python -m training.scripts.train_field --config training/configs/field/transformer_geometry.json
```

这种方式适合先验证环境、依赖和单实验训练闭环。

### 5.2 如果你要按任务 A 清单批量生成配置

先生成一批任务 A 配置：

```bash
python -m training.scripts.make_field_plan \
  --data-root <你的数据根目录> \
  --split-file <你的split.json> \
  --output-dir training/configs/field/generated
```

如果你这周只想先做 baseline，可以显式限制组别：

```bash
python -m training.scripts.make_field_plan \
  --data-root <你的数据根目录> \
  --split-file <你的split.json> \
  --groups baseline \
  --output-dir training/configs/field/generated
```

如果 baseline 跑通后要接着做输入特征消融：

```bash
python -m training.scripts.make_field_plan \
  --data-root <你的数据根目录> \
  --split-file <你的split.json> \
  --groups baseline,input \
  --output-dir training/configs/field/generated
```

### 5.3 如果你要先看生成了哪些实验

先看 manifest：

- [manifest.json](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/training/configs/field/generated/manifest.json)

这个文件是当前批量实验的唯一事实来源。后续批量训练、筛选和对表，都建议以它为准。

### 5.4 如果你要按实验组批量跑

先 dry run 看命令：

```bash
python -m training.scripts.run_field_plan \
  --manifest training/configs/field/generated/manifest.json \
  --study-group baseline \
  --dry-run
```

确认没问题后再正式跑：

```bash
python -m training.scripts.run_field_plan \
  --manifest training/configs/field/generated/manifest.json \
  --study-group baseline
```

### 5.5 如果你只想跑某一个 `Exp ID`

例如只跑 `A-Main-01`：

```bash
python -m training.scripts.run_field_plan \
  --manifest training/configs/field/generated/manifest.json \
  --exp-id A-Main-01
```

如果只跑一个 seed：

```bash
python -m training.scripts.run_field_plan \
  --manifest training/configs/field/generated/manifest.json \
  --exp-id A-Main-01 \
  --seed 1
```

---

## 6. 本周推荐命令顺序

按你现在的节奏，建议严格按下面顺序执行：

1. 先确认 split 文件可用。
2. 先 `make_field_plan --groups baseline`。
3. 再 `run_field_plan --study-group baseline --dry-run`。
4. 再真正跑 baseline。
5. baseline 首轮可读后，再生成并启动 `input` 组。

对应命令顺序可以固定成：

```bash
python -m training.scripts.make_field_plan \
  --data-root <你的数据根目录> \
  --split-file <你的split.json> \
  --groups baseline \
  --output-dir training/configs/field/generated
```

```bash
python -m training.scripts.run_field_plan \
  --manifest training/configs/field/generated/manifest.json \
  --study-group baseline \
  --dry-run
```

```bash
python -m training.scripts.run_field_plan \
  --manifest training/configs/field/generated/manifest.json \
  --study-group baseline
```

---

## 7. 本周最建议先盯住的 4 个配置

如果时间紧，本周先只盯住下面 4 个：

- `A-Base-01`
- `A-Base-02`
- `A-Base-03`
- `A-Main-01`

对应重点分别是：

- `A-Base-01`：给出点级下限
- `A-Base-02`：回答图是否必要
- `A-Base-03`：给出无 geometry 的 Transformer 对照
- `A-Main-01`：形成当前单尺度主线

只要这 4 个首轮结果可读，你这周就已经进入有效推进状态。

---

## 8. 输出结果怎么对应回实验表

建议你后续整理结果时，固定下面映射关系：

- 实验表主键：`exp_id`
- 多次重复主键：`exp_id + seed`
- 配置文件主键：`config_path`
- 运行目录识别名：`experiment_name`

换句话说：

- 写论文和文档，用 `exp_id`
- 看配置内容，用 `config_path`
- 查程序运行产物，用 `experiment_name`

这样不会混。

---

## 9. 本周最容易踩的坑

1. `data_root` 和 `graphs_subdir` 指到了不同版本数据。
2. `split_file` 还是示例文件，没有换成真实患者划分。
3. baseline 还没跑稳，就开始同时改模型结构和优化器。
4. 只看训练日志，不做统一评估。
5. 结果写表时混用 `experiment_name` 和 `exp_id`。

如果你本周先把这 5 个坑避开，推进会顺很多。
