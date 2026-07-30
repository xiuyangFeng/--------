# 显式几何特征工程

当前仓库以 [`pipeline/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/pipeline) 为主线，历史脚本已归档到 [`legacy/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/legacy)，数据目录位置保持不变。

## 目录导航
- [`pipeline/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/pipeline)：正式处理流程，推荐入口
- `pipeline_wss_min/`：WSS-only 最小化预处理流程，默认输入 `x,y,z`、标签 `wss`，使用原始 STL landmark v4 解剖坐标架与归一化后壁面采样
- `wss_pinn/`：独立 WSS-PINN 实验线；F1 八臂诊断已选中 `continuity=1e-4`、`no-slip=10`；仅在 PINN 路线冻结 `train138/test35`，baseline 原 split 未修改；173 例 source/sidecar Gate 全通过，但 full F0-UP Jobs `11073→11074` 完成后因全量 velocity/pressure Gate 失败而 No-Go；full F1 未提交，F2 blocked，WSS-physics/momentum 始终未开启
- [`training/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/training)：任务 A V1/V2/V3 内部训练、评估与集群脚本
- [`external_baselines/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/external_baselines)：外部论文 baseline 复现代码，当前包含 PointNetCFD
- [`pipeline/vmtk_core.py`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/pipeline/vmtk_core.py)：主线几何中心线提取与特征计算核心
- [`legacy/preprocess/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/legacy/preprocess)：旧版几何预处理、映射与整理脚本
- [`legacy/min-road/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/legacy/min-road)：历史训练与预处理链路
- [`docs/`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/docs)：实验总纲、路线文档、推进记录、论文复现记录
- `data_new/`、`stl_data/`：原始与处理中数据，未重排

## 外部 baseline 复现

PointNetCFD 第一轮复现入口：

```bash
conda activate rag_venv
python -m external_baselines.pointnetcfd.train \
  --config external_baselines/pointnetcfd/configs/pointnetcfd_original_vp.json \
  --dry-run
```

集群提交模板：

```bash
bash external_baselines/pointnetcfd/cluster/submit_pointnetcfd.sh \
  external_baselines/pointnetcfd/configs/pointnetcfd_original_vp.json
```

详细说明见 [`external_baselines/pointnetcfd/README.md`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/external_baselines/pointnetcfd/README.md) 与 [`docs/paper_reproduction/papers/pointnetcfd/README.md`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/docs/paper_reproduction/papers/pointnetcfd/README.md)。

## 推荐入口
WSS-only 最小化路线（独立产物 `data_wss_min/`，不改旧 `pipeline/` 数据）：

当前 v4 活动口径：AG76；AAA 几何签核63、训练质量白名单57；ILO 术前最终人工审核通过41例、总排除20例、活动术后bundle为0。ILO 已在专用协议中使用：fixed test27 的 41 例只入训练，pool2025 mixed 协议为 train138/test36；原 AG/AAA Phase-V split 不变。D2 c125×k64 的 ILO 两协议与结构矩阵已完成；`PointNeXt-R + LocalGeoPE` 的 S3 已通过三种子配对。RCR Oracle 的信息上限证据保留但当前不扩展，静态 EdgeConv 与 O0 hotspot/pinball 首轮均 No-Go。REG-P10-LSA2 的 SAME/IND × MSE/H1/H2 六臂中，IND 主效应和 H1 均 No-Go，SAME-H2 q90 pinball λ0.20 过门。其精确同-seed并发复现与唯一新增 `log(local_radius)` 输入列的两臂 Jobs `11032→11033_[0-1]` 已全部完成；处理臂相对并发 H2 对照的 `ΔR²_cb=+0.0436`、`Δhigh-WSS nRMSE=-0.00295`，全部保护线通过，现晋级为新的单 seed 开发锚点（`R²_cb=0.3506`）。暂不做多 seed。

```bash
conda activate GNN
python -m pipeline_wss_min.run --stage preprocess
python -m pipeline_wss_min.run --stage qa-gate
python -m pipeline_wss_min.run --stage global-stats --stats-timesteps peak
python -m pipeline_wss_min.run --stage build-samples
```

说明见 `pipeline_wss_min/README.md`、`training_wss_min/README.md`、`docs/02-推进与变更/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md`、`docs/02-推进与变更/WSS最小化_训练实验跟踪.md` 与 `docs/02-推进与变更/WSS最小化_代码修改与实验推进记录.md`；已完成的诊断与交接材料统一放在 `docs/02-推进与变更/_archive/WSS最小化/`。

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

- 未来无论是数据侧预处理/批处理，还是模型训练侧的集群运行，都必须输出足够详细的进度日志；日志粒度至少细到每一个病例，避免因整体数据量过大而看不到任务是否仍在有效推进，只能无效等待。
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

说明：

- `pipeline.extract_features` 现在直接依赖 [`pipeline/vmtk_core.py`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/pipeline/vmtk_core.py)，不再把主线几何核心挂在 `legacy/` 下。
- `legacy/preprocess/vmtk_core.py` 仅保留兼容层，避免旧脚本立即失效。

## 迁移说明
- 历史文档迁移说明见 [`docs/paper_idea/MIGRATION.md`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/docs/paper_idea/MIGRATION.md)
- 旧一体化预处理说明已由 [`pipeline/README.md`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/pipeline/README.md) 与 [`docs/README.md`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/docs/README.md) 接替
- `pipeline` 详细说明见 [`pipeline/README.md`](/Users/xiuyang/研究生学习/GNN-代码/显示几何特征工程/提取和处理/pipeline/README.md)
