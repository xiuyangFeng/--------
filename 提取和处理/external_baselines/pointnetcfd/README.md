# PointNetCFD Reproduction

> 论文：A point-cloud deep learning framework for prediction of fluid flow fields on irregular geometries
> 原始代码：https://github.com/Ali-Stanford/PointNetCFD
> 本目录目标：保留 PointNetCFD 的 shared point MLP + global max pooling + point-wise decoder 核心思想，在本项目私有 PyG 图快照数据上建立外部点云 baseline。

## 1. 复现口径

本实现不使用 `edge_index`，只把每个 `.pt` 图快照看成一个点云。模型结构为：

1. 每个点输入 `node_features + global_cond`；
2. shared MLP 提取逐点特征；
3. 对每个图做 global max pooling；
4. 将逐点特征、图级全局特征、`global_cond` 拼接；
5. point-wise decoder 输出目标场量。

这对应 PointNetCFD 的核心优化点：用共享点网络和全局 pooling 在不规则点云上预测 CFD 场，不依赖 mesh connectivity。

## 2. 第一轮实验矩阵

| 配置 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `pointnetcfd_original_vp.json` | `x,y,z + t,BC` | `u,v,w,p` | 最接近 paper-original 的速度/压力点云 baseline |
| `pointnetcfd_wall_vp.json` | `x,y,z,is_wall + t,BC` | `u,v,w,p` | 加壁面标记，检查 wall mask 是否改善近壁质量 |
| `pointnetcfd_geom_vp.json` | `x,y,z,is_wall,Abscissa,NormRadius,Curvature,Tangent + t,BC` | `u,v,w,p` | 验证显式几何特征对速度/压力的增益 |
| `pointnetcfd_geom_pwss.json` | `x,y,z,is_wall,Abscissa,NormRadius,Curvature,Tangent + t,BC` | wall-only `p,wss_x,wss_y,wss_z` | 验证压力 + WSS 直接预测路线 |

## 3. 本地 smoke test

```bash
conda activate rag_venv
python -m external_baselines.pointnetcfd.train \
  --config external_baselines/pointnetcfd/configs/pointnetcfd_original_vp.json \
  --dry-run
```

`--dry-run` 只读取配置并检查维度，不读取私有数据。真正训练需要把配置中的 `data_root` 和 `split_file` 改成集群上的真实路径。

## 4. 数据审计

正式提交训练前，先检查配置和私有 PyG 图快照是否匹配：

```bash
python -m external_baselines.pointnetcfd.audit_dataset \
  --config external_baselines/pointnetcfd/configs/pointnetcfd_geom_pwss.json \
  --out outputs/external_baselines/pointnetcfd/audit_geom_pwss.json
```

审计会检查 train/val/test 图文件数量、输入维度、输出目标、第一帧点数，以及 pWSS wall-only 过滤后是否还有监督点。

## 5. 集群运行

```bash
cd /path/to/提取和处理
bash external_baselines/pointnetcfd/cluster/submit_pointnetcfd.sh \
  external_baselines/pointnetcfd/configs/pointnetcfd_original_vp.json
```

如果集群数据路径不同，优先复制一份配置到 `configs/local/` 或实验记录目录中修改，不要覆盖这四个模板。

批量提交四组模板：

```bash
bash external_baselines/pointnetcfd/cluster/submit_all_pointnetcfd.sh
```

## 6. 独立评估

训练结束后可重新评估任意 split，并导出整体、按病例、按 sample 的指标：

```bash
python -m external_baselines.pointnetcfd.evaluate \
  --checkpoint outputs/external_baselines/pointnetcfd/<run_dir>/best_model.pt \
  --split test \
  --save-predictions
```

输出包括：

- `metrics_test.json`
- `metrics_test_by_case.csv`
- `metrics_test_by_sample.csv`
- `predictions_test.npz`（仅在 `--save-predictions` 时生成）

## 7. 输出

默认输出到：

```text
outputs/external_baselines/pointnetcfd/<experiment_name>_<split>_seed<seed>_<timestamp>/
```

主要文件：

- `config.json`：本次实际配置；
- `history.csv`：每 epoch train/val loss；
- `best_model.pt`：按验证集 loss 选出的 checkpoint；
- `metrics_test.json`：测试集 loss、RMSE、MAE、R2；
- `metrics_test_by_case.csv`：病例级指标；
- `metrics_test_by_sample.csv`：快照级指标；
- `manifest.json`：运行摘要。

## 8. 代码结构

| 文件 | 用途 |
| --- | --- |
| `data.py` | 读取本项目 PyG `.pt` 图快照，按配置选择输入特征与目标，pWSS 自动过滤 wall 节点 |
| `model.py` | PointNetCFD-style shared MLP + global max pooling + decoder |
| `metrics.py` | 整体、按病例、按 sample 的回归指标 |
| `train.py` | 训练、早停、checkpoint、manifest 输出 |
| `evaluate.py` | 独立 checkpoint 评估与预测导出 |
| `audit_dataset.py` | 训练前数据/配置兼容性审计 |
| `configs/*.json` | 四组第一轮实验模板 |
| `cluster/*.sh/slurm` | 单配置与批量 Slurm 提交模板 |

## 9. 注意事项

- `p_wss_vec` 目标默认只在 wall 节点训练和评估，因为 `y_wss` 是壁面监督。
- 当前指标基于图数据中已经存储的归一化目标；如需 Pa 或 m/s 物理单位，需要后续接入反归一化统计。
- 速度/压力结果不能直接代表 WSS 能恢复；若要从速度后处理 WSS，必须按 `docs/paper_reproduction/03-后处理可视化与插值方法.md` 先做 CFD velocity oracle。
