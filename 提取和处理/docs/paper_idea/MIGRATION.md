# 代码结构迁移说明

> **状态**：迁移已完成，本文档仅供归档参考。当前正式代码入口为 `pipeline/` 目录。

## 当前推荐结构
- `pipeline/`：正式维护的数据处理主线
- `legacy/preprocess/`：旧版预处理、中心线与特征映射脚本
- `legacy/min-road/`：历史训练与旧链路实验代码
- `docs/`：补充说明、研究草稿、归档文档

## 主要路径映射
| 旧路径 | 新路径 |
| --- | --- |
| `Script_Scenario_A_Surface.py` | `legacy/preprocess/Script_Scenario_A_Surface.py` |
| `Script_Scenario_B_Volumetric.py` | `legacy/preprocess/Script_Scenario_B_Volumetric.py` |
| `vmtk_core.py` | `legacy/preprocess/vmtk_core.py` |
| `batch_process.py` | `legacy/preprocess/batch_process.py` |
| `clean_fluent_data.py` | `legacy/preprocess/clean_fluent_data.py` |
| `merge_ascii_points.py` | `legacy/preprocess/merge_ascii_points.py` |
| `normalize_features.py` | `legacy/preprocess/normalize_features.py` |
| `prepare_data.py` | `legacy/preprocess/prepare_data.py` |
| `integrated_preprocessing.py` | `legacy/preprocess/integrated_preprocessing.py` |
| `统计数据点数.py` | `legacy/preprocess/统计数据点数.py` |
| `min-road/` | `legacy/min-road/` |
| `README_integrated_preprocessing.md` | `docs/README_integrated_preprocessing.md` |
| `项目思路.md` | `docs/项目思路.md` |
| `paper_idea/` | `docs/paper_idea/` |

## 入口迁移
旧方式：

```bash
python run_all.py
python preprocess.py
python extract_features.py
```

新方式：

```bash
python -m pipeline.run_all
python -m pipeline.preprocess
python -m pipeline.extract_features
```

当前推荐的环境策略：

```bash
# 主流程
conda activate GNN
python -m pipeline.run_all --geometry-python /public/newhome/cy/.conda/envs/GNN_vmtk/bin/python

# 若分步执行，extract_features 建议单独在几何环境中运行
conda activate GNN_vmtk
python -m pipeline.extract_features
```

历史脚本入口：

```bash
python -m legacy.preprocess.batch_process
python -m legacy.preprocess.integrated_preprocessing
```

## 兼容性说明
- `data_new/`、`stl_data/`、现有病例目录结构未移动
- `pipeline` 内部导入已改为包导入优先，支持 `python -m pipeline.xxx`
- `legacy/preprocess` 内部已改为归档后可用的导入方式
- `legacy/min-road/pre_data` 对旧预处理脚本的依赖已切换到 `legacy.preprocess`
