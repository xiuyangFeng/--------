# 显式几何特征工程

当前仓库以 [`pipeline/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/pipeline) 为主线，历史脚本已归档到 [`legacy/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/legacy)，数据目录位置保持不变。

## 目录导航
- [`pipeline/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/pipeline)：正式处理流程，推荐入口
- [`legacy/preprocess/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/legacy/preprocess)：旧版几何预处理、映射与整理脚本
- [`legacy/min-road/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/legacy/min-road)：历史训练与预处理链路
- [`docs/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/docs)：迁移说明、旧文档、研究思路
- `data_new/`、`stl_data/`：原始与处理中数据，未重排

## 推荐入口
批量处理前建议先做一次输入审计：

```bash
conda activate GNN
python -m pipeline.audit_inputs --groups AAA AG ILO
```

完整流程：

```bash
conda activate GNN
python -m pipeline.run_all --case ZHANG_CHUN
```

日志查看：

- 单病例步骤日志：`data_new/<病例路径>/processed/logs/progress.log`
- 批量/总流程日志：`data_new/pipeline_reports/logs/run_all.log`
- 如果直接执行单步骤批量入口，还会生成：
  - `data_new/pipeline_reports/logs/step1_preprocess_batch.log`
  - `data_new/pipeline_reports/logs/step2_extract_features_batch.log`
  - `data_new/pipeline_reports/logs/step3_coord_normalize_batch.log`
  - `data_new/pipeline_reports/logs/step4_normalize_batch.log`
  - `data_new/pipeline_reports/logs/step5_convert_to_graph_batch.log`

可直接实时查看：

```bash
tail -f data_new/pipeline_reports/logs/run_all.log
```

若几何步骤依赖 `vmtk` 的独立环境，推荐这样运行：

```bash
conda activate GNN
python -m pipeline.run_all \
  --case ZHANG_CHUN \
  --geometry-python /public/newhome/cy/.conda/envs/GNN_vmtk/bin/python
```

单步运行：

```bash
conda activate GNN
python -m pipeline.preprocess --case ZHANG_CHUN
conda activate GNN_vmtk
python -m pipeline.extract_features --case ZHANG_CHUN
conda activate GNN
python -m pipeline.coord_normalize --case ZHANG_CHUN
python -m pipeline.normalize --case ZHANG_CHUN
python -m pipeline.convert_to_graph --case ZHANG_CHUN
```

数据集加载：

```python
from pipeline.dataset import CFDAugmentedDataset
```

## 历史脚本
历史脚本不再作为稳定顶层接口，若仍需使用，请从归档目录运行，例如：

```bash
python -m legacy.preprocess.batch_process --help
python -m legacy.preprocess.normalize_features --help
```

## 迁移说明
- 路径映射见 [`docs/MIGRATION.md`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/docs/MIGRATION.md)
- 原一体化说明见 [`docs/README_integrated_preprocessing.md`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/docs/README_integrated_preprocessing.md)
- `pipeline` 详细说明见 [`pipeline/README.md`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/pipeline/README.md)
